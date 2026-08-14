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
            f.write(f"{{n:02d}}.jpeg needs manual review (artifact after {{MAX_ATTEMPTS}} attempts)\\n")
print("ALL DONE", flush=True)
'''

# PixArt-Sigma: switched to from day 10 onward (see MODEL_SWITCH_DAY below)
# -- a real side-by-side comparison against
# FLUX.1-schnell on this exact channel's prompts showed PixArt reads as
# noticeably more visceral/creepy for this true-crime format (real color
# grading + a more intense screaming close-up), which matters more here than
# FLUX's marginal edge on raw prompt-following. Not gated on Hugging Face
# (unlike FLUX.1-schnell), so no HF_TOKEN needed. Same retry/OCR/vision-QA
# loop as the FLUX kernel, just swapped model + its tuned inference params.
PIXART_KERNEL_TEMPLATE = '''import os, sys, subprocess, json
def pip(*a): subprocess.run([sys.executable,"-m","pip","install","-q",*a], check=False)
pip("torch==2.4.1","torchvision==0.19.1","--index-url","https://download.pytorch.org/whl/cu121")
pip("diffusers==0.32.2","transformers==4.46.3","accelerate","sentencepiece","protobuf")
pip("easyocr")

import torch, numpy as np
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)

from diffusers import PixArtSigmaPipeline
pipe = PixArtSigmaPipeline.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16)
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
            img=pipe(p, num_inference_steps=25, guidance_scale=4.5, height=1280, width=720,
                     generator=torch.Generator("cpu").manual_seed(seed)).images[0]
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
            f.write(f"{{n:02d}}.jpeg needs manual review (artifact after {{MAX_ATTEMPTS}} attempts)\\n")
print("ALL DONE", flush=True)
'''

# Day this batch switches from FLUX.1-schnell to PixArt-Sigma. Days 1-9 stay
# FLUX (1-3 already published to YouTube; redoing 4-9 would just burn Kaggle
# GPU time for no visual-consistency benefit worth the cost -- see the
# decision recorded 2026-08-01).
MODEL_SWITCH_DAY = 10


def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH, encoding="utf-8"))
    return {"days": {}}


def save_state(state):
    json.dump(state, open(STATE_PATH, "w", encoding="utf-8"), indent=1)


def run_flux_for_day(day_dir, shots, day_num):
    """Runs one Kaggle image-gen kernel for this day's 16 shots. Blocks until
    done. Model depends on day_num: FLUX.1-schnell below MODEL_SWITCH_DAY,
    PixArt-Sigma from there on (see MODEL_SWITCH_DAY's comment for why)."""
    import time

    # images/seq/, not images/ — matches _gen_flux_images.py's convention, which
    # _build_composition.py's template and the hyperframes file server both
    # hardcode as "images/seq/NN.jpeg". Getting this wrong produces a
    # composition where every artwork layer 404s and renders solid black.
    seq_dir = os.path.join(day_dir, "images", "seq")
    os.makedirs(seq_dir, exist_ok=True)
    if os.path.exists(os.path.join(seq_dir, "16.jpeg")):
        print(f"day {day_num}: images already present, skipping generation")
        return

    use_pixart = day_num >= MODEL_SWITCH_DAY
    model_slug = "pixart" if use_pixart else "flux"
    kernel_dir = os.path.join(day_dir, f"_kaggle_{model_slug}_kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    kaggle_user = os.environ.get("KAGGLE_IMAGE_USERNAME", "anuragmishra108")
    kernel_id = f"{kaggle_user}/shadow-gasp-batch-day{day_num:02d}-{model_slug}"

    if use_pixart:
        code = PIXART_KERNEL_TEMPLATE.format(shots_json=json.dumps(shots))
    else:
        hf_token = os.environ["HF_TOKEN"]
        code = FLUX_KERNEL_TEMPLATE.format(shots_json=json.dumps(shots), hf_token=hf_token)
    open(os.path.join(kernel_dir, "gen_images.py"), "w", encoding="utf-8").write(code)
    json.dump({
        "id": kernel_id,
        "title": f"shadow-gasp-batch-day{day_num:02d}-{model_slug}",
        "code_file": "gen_images.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        # NvidiaTeslaP100 (default) is sm_60, which Kaggle's stock torch no longer
        # builds for -> kernel dies at import/op time. T4 is sm_75, same 16GB, works.
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }, open(os.path.join(kernel_dir, "kernel-metadata.json"), "w"), indent=2)

    print(f"day {day_num}: pushing kernel {kernel_id} ({model_slug}) ...")
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


def commit_day_assets(day_dir, day_num):
    """Commits this day's full deliverable — narration, shots.json, meta.json,
    all 16 stills — plus a fixed-name copy of shot 1 for the Sheet's raw
    GitHub URL. These are the actual output of the batch (consumed later, once
    a hook video comes back from Flow), not disposable build state, so they
    can't be left on the ephemeral Actions runner or in a time-limited
    artifact — commit them for real."""
    import shutil

    shutil.copyfile(os.path.join(day_dir, "images", "seq", "01.jpeg"), os.path.join(day_dir, "shot1.jpeg"))

    repo_root = os.path.dirname(PIPELINE_DIR)
    rel_dir = os.path.relpath(day_dir, repo_root).replace("\\", "/")
    subprocess.run(["git", "add", rel_dir], cwd=repo_root, check=True)
    status = subprocess.run(["git", "status", "--porcelain", rel_dir], cwd=repo_root, capture_output=True, text=True)
    if status.stdout.strip():
        subprocess.run(["git", "commit", "-m", f"batch: day {day_num:02d} assets"], cwd=repo_root, check=True)
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", GITHUB_BRANCH], cwd=repo_root, check=True)
        subprocess.run(["git", "push"], cwd=repo_root, check=True)

    shot1_rel = f"{rel_dir}/shot1.jpeg"
    return f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{shot1_rel}"


def get_sheets_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(
        SA_KEY_PATH, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=creds)


def append_sheet_row(service, day_num, case, title, shot1_url, notes):
    """The Sheets/Drive APIs both hit transient SSL errors periodically in
    this environment (seen repeatedly across this project, never a real
    permissions/data problem) — retry a few times before giving up, since a
    day's real GPU-generated assets are already safely committed by this
    point and it would be wasteful to fail the whole run over a flaky
    connection on the very last step."""
    import time

    row = [[day_num, case, "Images done", title, shot1_url, "Pending", "", notes]]
    for attempt in range(4):
        try:
            service.spreadsheets().values().append(
                spreadsheetId=SHEET_ID,
                range=f"{SHEET_TAB}!A2",
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body={"values": row},
            ).execute()
            return
        except Exception as e:
            print(f"day {day_num}: sheet append attempt {attempt + 1} failed ({e!r}), retrying", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"day {day_num}: could not append sheet row after 4 attempts")


def commit_state(reason, extra_paths=()):
    """state.json AND the real channel ledger (cases_used.json) are
    checkpointed after every meaningful change (not just at the end of the
    whole run) so a mid-run crash — Kaggle hiccup, transient Sheets SSL
    error, runner timeout — never loses track of which days are genuinely
    done, and never silently drops a case reservation (which would let the
    daily auto-pipeline or a later batch run pick the same case again).
    Each day's real assets already commit themselves independently in
    commit_day_assets(); this is cheap insurance on top."""
    repo_root = os.path.dirname(PIPELINE_DIR)
    rels = [os.path.relpath(STATE_PATH, repo_root).replace("\\", "/")]
    rels += [os.path.relpath(p, repo_root).replace("\\", "/") for p in extra_paths]
    subprocess.run(["git", "add", *rels], cwd=repo_root, check=True)
    status = subprocess.run(["git", "status", "--porcelain", *rels], cwd=repo_root, capture_output=True, text=True)
    if status.stdout.strip():
        subprocess.run(["git", "commit", "-m", f"batch: {reason}"], cwd=repo_root, check=True)
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", GITHUB_BRANCH], cwd=repo_root, check=True)
        subprocess.run(["git", "push"], cwd=repo_root, check=True)


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

        pick_path = os.path.join(day_dir, "pick.json")
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
            json.dump({"case": case, "angle": angle}, open(pick_path, "w", encoding="utf-8"), indent=2)
            day_state.update({"case": case, "angle": angle})
            state["days"][key] = day_state
            save_state(state)
            commit_state(f"day {day_num:02d} case picked", extra_paths=[gvc.LEDGER_PATH, pick_path])
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
        shot1_url = commit_day_assets(day_dir, day_num)

        if not day_state.get("sheet_logged"):
            append_sheet_row(sheets, day_num, case, meta["title_working"], shot1_url, angle)
            day_state["sheet_logged"] = True
            state["days"][key] = day_state
            save_state(state)
            commit_state(f"day {day_num:02d} sheet row logged")

        day_state["done"] = True
        state["days"][key] = day_state
        save_state(state)
        commit_state(f"day {day_num:02d} complete")
        new_days_done += 1
        print(f"day {day_num}: DONE — {case} ({new_days_done}/{new_days_budget} this run)")

    print(f"\nThis run: {new_days_done} new day(s) completed, through day {day_num}.")
    done_count = sum(1 for v in state["days"].values() if v.get("done"))
    batch_complete = done_count >= TOTAL_DAYS
    notify_pregen_done(new_days_done, day_num, batch_complete)

    if not batch_complete:
        dispatch_next_chunk(new_days_budget)
    else:
        print(f"Batch fully complete: {done_count}/{TOTAL_DAYS} days done.")


# Total size of the batch. Self-chaining (see dispatch_next_chunk) keeps
# triggering new chunks of this same size until this many days are done,
# so a human only has to start the batch once, not re-trigger every chunk.
TOTAL_DAYS = 30


def dispatch_next_chunk(new_days_budget):
    """Self-chaining: triggers another run of this same workflow with the
    same chunk size, so the batch finishes unattended instead of needing a
    human to manually re-run /pregen after every chunk. Best-effort -- a
    failed dispatch here must not make an otherwise-successful chunk show as
    failed; worst case, the batch just stalls and needs a manual nudge."""
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not repo:
        print("GITHUB_TOKEN/GITHUB_REPOSITORY not set, cannot self-dispatch next chunk")
        return
    import urllib.request

    try:
        body = json.dumps({
            "ref": "main",
            "inputs": {
                "days": str(new_days_budget),
                "notify_chat_id": os.environ.get("NOTIFY_CHAT_ID", ""),
            },
        }).encode()
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/actions/workflows/batch_pregen.yml/dispatches",
            data=body, method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "shadow-gasp-batch-pregen",
            },
        )
        urllib.request.urlopen(req).read()
        print(f"Self-dispatched next chunk (days={new_days_budget})")
    except Exception as e:
        print(f"Failed to self-dispatch next chunk (non-fatal, batch will stall until manually re-triggered): {e!r}")


def notify_pregen_done(new_days_done, through_day, batch_complete=False):
    """Best-effort: tells the /pregen chat this chunk is done, mirroring
    finish_batch_day.yml's /batch/uploaded callback. Never raises -- a
    notification failure must not make an otherwise-successful pregen run
    show as failed."""
    secret = os.environ.get("BATCH_NOTIFY_SECRET", "")
    if not secret:
        print("BATCH_NOTIFY_SECRET not set, skipping Telegram notify")
        return
    import urllib.request

    try:
        body = json.dumps({
            "new_days_done": new_days_done,
            "through_day": through_day,
            "batch_complete": batch_complete,
            "chat_id": os.environ.get("NOTIFY_CHAT_ID") or None,
        }).encode()
        req = urllib.request.Request(
            "https://shadow-gasp-bot.everydayhypehq.workers.dev/batch/pregen_done",
            data=body, method="POST",
            headers={"X-Batch-Notify-Secret": secret, "Content-Type": "application/json"},
        )
        urllib.request.urlopen(req).read()
    except Exception as e:
        print(f"pregen_done notify failed (non-fatal): {e!r}")


if __name__ == "__main__":
    main()
