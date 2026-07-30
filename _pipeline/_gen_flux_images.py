"""Generate all 16 noir comic-book stills for a video via FLUX.1-schnell on
Kaggle GPU, from shots.json (produced by _gen_video_content.py).

Pushes a Kaggle kernel with the prompts baked in (no dataset needed — FLUX is
text-to-image), polls until complete, downloads outputs into images/seq/.

Requires: kaggle CLI authenticated (image-gen account), shots.json present.
Skips entirely if images/seq/16.jpeg already exists (mirrors the vo.txt-skip
pattern used elsewhere in this pipeline).
"""
import json
import os
import subprocess
import sys
import time

PROJECT = os.getcwd()
SEQ_DIR = os.path.join(PROJECT, "images", "seq")
KERNEL_DIR = os.path.join(PROJECT, "_kaggle_flux_kernel")
KAGGLE_USER = os.environ.get("KAGGLE_IMAGE_USERNAME", "anuragmishra108")
SLUG = os.path.basename(PROJECT).replace("_", "-")[:40]
KERNEL_ID = f"{KAGGLE_USER}/{SLUG}-flux"

KERNEL_TEMPLATE = '''import os, sys, subprocess, json
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
    # Any detected text region at all is a violation — these prompts explicitly
    # forbid text/logos/speech bubbles, so a real still should have none.
    # Confidence > 0.35 filters out pure noise-shaped false positives.
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


def main():
    if os.path.exists(os.path.join(SEQ_DIR, "16.jpeg")):
        print("images/seq/16.jpeg already present, skipping FLUX generation")
        return

    shots = json.load(open("shots.json"))
    assert len(shots) == 16, f"expected 16 shots, got {len(shots)}"

    os.makedirs(KERNEL_DIR, exist_ok=True)
    os.makedirs(SEQ_DIR, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", "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")
    code = KERNEL_TEMPLATE.format(shots_json=json.dumps(shots), hf_token=hf_token)
    open(os.path.join(KERNEL_DIR, "gen_flux.py"), "w", encoding="utf-8").write(code)
    json.dump({
        "id": KERNEL_ID,
        "title": f"{SLUG}-flux",
        "code_file": "gen_flux.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }, open(os.path.join(KERNEL_DIR, "kernel-metadata.json"), "w"), indent=2)

    print(f"Pushing kernel {KERNEL_ID} ...")
    subprocess.run(["kaggle", "kernels", "push", "-p", "."], cwd=KERNEL_DIR, check=True)

    print("Polling for completion (FLUX.1-schnell, ~5-10 min for 16 images)...")
    while True:
        time.sleep(30)
        r = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID], capture_output=True, text=True)
        status = r.stdout.strip()
        print(status)
        if "COMPLETE" in status:
            break
        if "ERROR" in status or "CANCEL" in status:
            print(r.stdout, r.stderr, file=sys.stderr)
            sys.exit(1)

    out_dir = os.path.join(KERNEL_DIR, "out")
    subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", out_dir], check=True)

    for i in range(1, 17):
        src = os.path.join(out_dir, f"{i:02d}.jpeg")
        dst = os.path.join(SEQ_DIR, f"{i:02d}.jpeg")
        if os.path.exists(src):
            os.replace(src, dst)
        else:
            print(f"WARNING: {src} missing from kernel output", file=sys.stderr)

    print(f"Downloaded {len([f for f in os.listdir(SEQ_DIR) if f.endswith('.jpeg')])} stills to {SEQ_DIR}")

    flagged_path = os.path.join(out_dir, "FLAGGED.txt")
    if os.path.exists(flagged_path):
        flagged = open(flagged_path).read().strip()
        print(f"\nWARNING — some shots still have text/bubble artifacts after {3} retries, needs manual review:\n{flagged}", file=sys.stderr)


if __name__ == "__main__":
    main()
