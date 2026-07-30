"""Pre-generate a batch of shadow_gasp days ahead of time: pick N distinct
unused cases, write narration+shots for each, render all 16 FLUX stills, push
day 1's still to GitHub (so it has a stable public URL), and log each day into
the "shadow_gasp - 30 day batch" Google Sheet.

This is NOT the daily automated pipeline (that's pipeline.yml / run_pipeline.py,
which does TTS/transcribe/composition/upload too). Batch days stop after the
stills: the hook clip for shot 1 is generated manually via Google Flow (or,
if Flow isn't used for a given day, CogVideoX picks up that day the same way
the automated pipeline already does), and the rest of assembly happens later
once the hook video comes back.

Each day is reserved in the real channel ledger (cases_used.json) as soon as
it's picked, so the daily automated pipeline can never independently pick the
same case while it's sitting in this batch queue.

Resumable: progress is tracked in _pipeline/batch/state.json, keyed by day
number, so a rerun after a crash/interruption only does the remaining days.

Usage: python _batch_pregen.py [N]   (N = how many NEW days to process this
run, defaults to 30; days already marked done in state.json don't count
against N and are skipped for free)
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anthropic import Anthropic

import _gen_video_content as gvc
import _pick_case as pc

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_DIR = os.path.join(PIPELINE_DIR, "batch")
STATE_PATH = os.path.join(BATCH_DIR, "state.json")
SA_KEY_PATH = os.path.join(PIPELINE_DIR, "_local", "sheets_sa_key.json")
SHEET_ID = "1aPoXPKlC9cCStUqULzR46FmvUaL8jxQbFsDWEDXn3jM"
SHEET_TAB = "Batch"
GITHUB_REPO = "Universekianukal/shadow-gasp-pipeline"
GITHUB_BRANCH = "main"

FLUX_KERNEL_TEMPLATE = '''import os, sys, subprocess, json
def pip(*a): subprocess.run([sys.executable,"-m","pip","install","-q",*a], check=False)
pip("torch==2.4.1","torchvision==0.19.1","--index-url","https://download.pytorch.org/whl/cu121")
pip("diffusers==0.32.2","transformers==4.46.3","accelerate","sentencepiece","protobuf","bitsandbytes")
pip("easyocr")

import torch, numpy as np
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN","{hf_token}"))
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)

from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig as DBnb
from transformers import T5EncoderModel, BitsAndBytesConfig as TBnb
repo="black-forest-labs/FLUX.1-schnell"
nf4=dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

tf=FluxTransformer2DModel.from_pretrained(repo, subfolder="transformer",
     quantization_config=DBnb(**nf4), torch_dtype=torch.float16)
te=T5EncoderModel.from_pretrained(repo, subfolder="text_encoder_2",
     quantization_config=TBnb(**nf4), torch_dtype=torch.float16)
pipe=FluxPipeline.from_pretrained(repo, transformer=tf, text_encoder_2=te, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
print("PIPE READY", flush=True)

import easyocr
ocr = easyocr.Reader(["en"], gpu=True)
print("OCR READY", flush=True)

def has_text(img):
    results = ocr.readtext(np.array(img))
    hits = [r for r in results if r[2] > 0.35]
    return hits

SHOTS = {shots_json}
MAX_ATTEMPTS = 3

for s in SHOTS:
    n, p = s["n"], s["prompt"]
    saved = False
    for attempt in range(MAX_ATTEMPTS):
        seed = 3000 + n + attempt * 10000
        try:
            img=pipe(p, num_inference_steps=4, guidance_scale=0.0, height=1280, width=720,
                     max_sequence_length=256, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
            m=float(np.asarray(img).mean())
            if m<5:
                print("RETRY", n, "attempt", attempt+1, "black/NaN frame", flush=True)
                torch.cuda.empty_cache()
                continue
            hits = has_text(img)
            if hits:
                texts = [h[1] for h in hits]
                print("RETRY", n, "attempt", attempt+1, "text detected:", texts, flush=True)
                torch.cuda.empty_cache()
                continue
            img.save(f"/kaggle/working/{{n:02d}}.jpeg", quality=92)
            print("DONE", n, "meanpix", round(m,1), "attempt", attempt+1, flush=True)
            saved = True
            break
        except Exception as e:
            print("FAILED", n, "attempt", attempt+1, repr(e), flush=True)
        torch.cuda.empty_cache()
    if not saved:
        print("GAVE UP", n, "after", MAX_ATTEMPTS, "attempts — saving last generation anyway with a warning", flush=True)
        img.save(f"/kaggle/working/{{n:02d}}.jpeg", quality=92)
        with open("/kaggle/working/FLAGGED.txt", "a") as f:
            f.write(f"{{n:02d}}.jpeg needs manual review (text/bubble artifact after {{MAX_ATTEMPTS}} attempts)\\n")
print("ALL DONE", flush=True)
'''


def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH, encoding="utf-8"))
    return {"days": {}}


def save_state(state):
    json.dump(state, open(STATE_PATH, "w", encoding="utf-8"), indent=1)


def run_flux_for_day(day_dir, shots, day_num):
    """Runs one Kaggle FLUX kernel for this day's 16 shots. Blocks until done."""
    import time

    seq_dir = os.path.join(day_dir, "images")
    os.makedirs(seq_dir, exist_ok=True)
    if os.path.exists(os.path.join(seq_dir, "16.jpeg")):
        print(f"day {day_num}: images already present, skipping FLUX")
        return

    kernel_dir = os.path.join(day_dir, "_kaggle_flux_kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    hf_token = os.environ["HF_TOKEN"]
    kaggle_user = os.environ.get("KAGGLE_IMAGE_USERNAME", "anuragmishra108")
    kernel_id = f"{kaggle_user}/shadow-gasp-batch-day{day_num:02d}-flux"

    code = FLUX_KERNEL_TEMPLATE.format(shots_json=json.dumps(shots), hf_token=hf_token)
    open(os.path.join(kernel_dir, "gen_flux.py"), "w", encoding="utf-8").write(code)
    json.dump({
        "id": kernel_id,
        "title": f"shadow-gasp-batch-day{day_num:02d}-flux",
        "code_file": "gen_flux.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }, open(os.path.join(kernel_dir, "kernel-metadata.json"), "w"), indent=2)

    print(f"day {day_num}: pushing kernel {kernel_id} ...")
    subprocess.run(["kaggle", "kernels", "push", "-p", "."], cwd=kernel_dir, check=True)

    print(f"day {day_num}: polling for completion ...")
    while True:
        time.sleep(30)
        r = subprocess.run(["kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
        status = r.stdout.strip()
        print(f"day {day_num}: {status}")
        if "COMPLETE" in status:
            break
        if "ERROR" in status or "CANCEL" in status:
            print(r.stdout, r.stderr, file=sys.stderr)
            raise RuntimeError(f"day {day_num}: kaggle kernel failed: {status}")

    out_dir = os.path.join(kernel_dir, "out")
    subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", out_dir], check=True)
    for i in range(1, 17):
        src = os.path.join(out_dir, f"{i:02d}.jpeg")
        dst = os.path.join(seq_dir, f"{i:02d}.jpeg")
        if os.path.exists(src):
            os.replace(src, dst)
        else:
            print(f"WARNING: day {day_num}: {src} missing from kernel output", file=sys.stderr)


def push_shot1(day_dir, day_num):
    """Copies shot 1 to a fixed, gitignore-excepted filename and pushes it so
    raw.githubusercontent.com has something stable to link to."""
    import shutil

    src = os.path.join(day_dir, "images", "01.jpeg")
    dst = os.path.join(day_dir, "shot1.jpeg")
    shutil.copyfile(src, dst)

    rel = os.path.relpath(dst, os.path.dirname(PIPELINE_DIR)).replace("\\", "/")
    repo_root = os.path.dirname(PIPELINE_DIR)
    subprocess.run(["git", "add", rel], cwd=repo_root, check=True)
    status = subprocess.run(["git", "status", "--porcelain", rel], cwd=repo_root, capture_output=True, text=True)
    if status.stdout.strip():
        subprocess.run(["git", "commit", "-m", f"batch: day {day_num:02d} shot1 still"], cwd=repo_root, check=True)
        subprocess.run(["git", "push"], cwd=repo_root, check=True)
    return f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{rel}"


def get_sheets_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(
        SA_KEY_PATH, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=creds)


def append_sheet_row(service, day_num, case, title, shot1_url, notes):
    row = [[day_num, case, "Images done", title, shot1_url, "Pending", "", notes]]
    service.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range=f"{SHEET_TAB}!A2",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": row},
    ).execute()


def main():
    new_days_budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    os.makedirs(BATCH_DIR, exist_ok=True)
    state = load_state()

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sheets = get_sheets_service()

    ledger = gvc.load_ledger()
    used_cases = [c["case"] for c in ledger["cases"]]

    new_days_done = 0
    day_num = 0
    while new_days_done < new_days_budget:
        day_num += 1
        key = str(day_num)
        day_state = state["days"].get(key, {})
        if day_state.get("done"):
            used_cases.append(day_state["case"])
            continue

        day_dir = os.path.join(BATCH_DIR, f"day{day_num:02d}")
        os.makedirs(day_dir, exist_ok=True)

        if "case" in day_state:
            case = day_state["case"]
            angle = day_state.get("angle", "")
            print(f"day {day_num}: resuming existing pick: {case}")
        else:
            picked = pc.pick(client, used_cases)
            case = picked["case"]
            angle = picked.get("angle", "")
            used_cases.append(case)
            ledger["cases"].append({"videoId": None, "case": case, "publishedAt": None})
            gvc.save_ledger(ledger)
            day_state.update({"case": case, "angle": angle})
            state["days"][key] = day_state
            save_state(state)
            print(f"day {day_num}: picked {case}")

        shots_path = os.path.join(day_dir, "shots.json")
        meta_path = os.path.join(day_dir, "meta.json")
        if os.path.exists(shots_path) and os.path.exists(meta_path):
            shots = json.load(open(shots_path, encoding="utf-8"))
            meta = json.load(open(meta_path, encoding="utf-8"))
        else:
            d = gvc.generate(client, case)
            open(os.path.join(day_dir, "narration.txt"), "w", encoding="utf-8").write(d["narration"].strip() + "\n")
            shots = d["shots"]
            json.dump(shots, open(shots_path, "w", encoding="utf-8"), indent=1)
            meta = {
                "hook_motion_prompt": d["hook_motion_prompt"],
                "caption_yt": d["caption_yt"],
                "caption_ig": d["caption_ig"],
                "title_working": d["title_working"],
            }
            json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)
            print(f"day {day_num}: generated narration + 16 shot prompts, title: {meta['title_working']}")

        run_flux_for_day(day_dir, shots, day_num)
        shot1_url = push_shot1(day_dir, day_num)

        if not day_state.get("sheet_logged"):
            append_sheet_row(sheets, day_num, case, meta["title_working"], shot1_url, angle)
            day_state["sheet_logged"] = True

        day_state["done"] = True
        state["days"][key] = day_state
        save_state(state)
        new_days_done += 1
        print(f"day {day_num}: DONE — {case} ({new_days_done}/{new_days_budget} this run)")

    print(f"\nThis run: {new_days_done} new day(s) completed, through day {day_num}.")


if __name__ == "__main__":
    main()
