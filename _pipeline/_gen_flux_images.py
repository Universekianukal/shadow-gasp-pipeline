"""Generate all 16 noir comic-book stills for a video via PixArt-Sigma on
Kaggle GPU, from shots.json (produced by _gen_video_content.py).

Pushes a Kaggle kernel with the prompts baked in (no dataset needed — this is
text-to-image), polls until complete, downloads outputs into images/seq/.

MODEL: PixArt-Sigma-XL-2-1024-MS, matching _batch_pregen.py's MODEL_SWITCH_DAY
choice. This script ran FLUX.1-schnell until 2026-08-30. The batch path had
already switched to PixArt at day 10 — a side-by-side on this channel's own
prompts found PixArt noticeably more visceral for the true-crime format (real
color grading, stronger close-ups) — but the daily pipeline, which took over
at day 34, never inherited that switch. The result was a visible style break
between day 33 and day 34: FLUX at 4 steps / CFG 0.0 renders flatter and
cooler, and on dense wide-environment prompts it loses the "thick black ink
outlines" style anchor entirely (day 53 shot 1 is the clearest example). The
prompts were never the difference — _gen_video_content.py is unchanged
throughout — the model was. Keep this in sync with _batch_pregen.py's
PIXART_KERNEL_TEMPLATE; the two paths must not drift apart again.

Requires: kaggle CLI authenticated (image-gen account), shots.json present.
No HF_TOKEN needed — PixArt-Sigma is not gated on Hugging Face (FLUX was).
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


def main():
    if os.path.exists(os.path.join(SEQ_DIR, "16.jpeg")):
        print("images/seq/16.jpeg already present, skipping still generation")
        return

    shots = json.load(open("shots.json"))
    assert len(shots) == 16, f"expected 16 shots, got {len(shots)}"

    os.makedirs(KERNEL_DIR, exist_ok=True)
    os.makedirs(SEQ_DIR, exist_ok=True)

    # PixArt-Sigma is not gated, so unlike the old FLUX.1-schnell path this no
    # longer needs an HF_TOKEN at all. Callers (pipeline.yml) may still export
    # one; it is simply unused now, and its absence is no longer fatal.
    code = KERNEL_TEMPLATE.format(shots_json=json.dumps(shots))
    open(os.path.join(KERNEL_DIR, "gen_flux.py"), "w", encoding="utf-8").write(code)
    json.dump({
        "id": KERNEL_ID,
        "title": f"{SLUG}-flux",
        "code_file": "gen_flux.py",
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
    }, open(os.path.join(KERNEL_DIR, "kernel-metadata.json"), "w"), indent=2)

    print(f"Pushing kernel {KERNEL_ID} ...")
    subprocess.run(["kaggle", "kernels", "push", "-p", "."], cwd=KERNEL_DIR, check=True)

    print("Polling for completion (PixArt-Sigma at 25 steps, ~25-35 min for 16 images)...")
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
    os.makedirs(out_dir, exist_ok=True)

    # `kaggle kernels output` downloads the kernel's files one at a time and
    # aborts the whole command on the first connection that drops. Day47 got
    # 6 of 16 stills before a ConnectionResetError(104) killed it, throwing
    # away a kernel run that had already reached COMPLETE and ~30 minutes of
    # wall clock -- the images existed on Kaggle the whole time, only the
    # download failed. Retry, and accept a non-zero exit as long as all 16
    # files actually landed, since each attempt only has to fill the gaps.
    def missing_stills():
        return [i for i in range(1, 17)
                if not os.path.exists(os.path.join(out_dir, f"{i:02d}.jpeg"))]

    for attempt in range(1, 5):
        rc = subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", out_dir]).returncode
        missing = missing_stills()
        if not missing:
            if rc != 0:
                print(f"attempt {attempt}: kaggle exited {rc}, but all 16 stills are present — continuing")
            break
        print(f"attempt {attempt}: output fetch failed (exit {rc}), "
              f"still missing {len(missing)} stills: {missing}", file=sys.stderr)
        if attempt < 4:
            time.sleep(15 * attempt)
    else:
        raise SystemExit(
            f"kaggle kernels output failed after 4 attempts; still missing stills {missing_stills()}. "
            f"The kernel itself completed — retry the job, or fetch manually with: "
            f"kaggle kernels output {KERNEL_ID} -p <dir>"
        )

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
