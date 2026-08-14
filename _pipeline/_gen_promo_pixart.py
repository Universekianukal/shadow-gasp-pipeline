"""One-off: generate a PixArt-Sigma noir PROP SCENE (no book, no text, no
character) for the NORJAK Facebook release post. The real cover.jpg -- which
already has legible title text, since it's the actual rendered PDF page, not
AI-hallucinated text -- gets composited into this scene afterward by
_compose_promo_scene.py. That's deliberate: letting a diffusion model render
a book with its own cover text always produces garbled fake type, but a bare
prop scene needs no text at all, so there's nothing for it to get wrong.

Reuses the same Kaggle push/poll/pull pattern as _regen_shot.py but skips the
Claude vision-QA step (no ANTHROPIC_API_KEY available locally) -- kept the
black-frame and OCR fake-text retry checks since those need no API key.

Usage: python _gen_promo_pixart.py
Writes ./norjak_promo_scene.jpeg
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KAGGLE_USER = os.environ.get("KAGGLE_IMAGE_USERNAME", "anuragmishra108")
KERNEL_ID = f"{KAGGLE_USER}/shadow-gasp-norjak-promo-scene"
KERNEL_DIR = os.path.join(HERE, "_kaggle_pixart_promo_kernel")

PROMPT = (
    "Noir true-crime aesthetic, cinematic photo-illustration, gritty halftone "
    "texture, 1970s muted color palette (mustard yellow, rust orange, faded "
    "teal, deep shadow). A weathered wooden desk shot from a high angle, "
    "empty rectangular space left clear in the center-left third of the frame "
    "for a book to be placed later. Scattered around the edges: a stack of "
    "worn cash bundles bound with paper straps, an open case file folder with "
    "blank pages, a lit cigarette resting in a glass ashtray with smoke "
    "curling up, a pair of dark sunglasses, a coiled length of parachute cord. "
    "Rain streaks down a window in the background, city runway lights blurred "
    "and bokeh beyond the glass. Dramatic low side lighting, deep shadows, "
    "high contrast. No text, no logos, no watermark, no signage, no readable "
    "writing anywhere, no book or rectangular object in the empty center space."
)

KERNEL_CODE = '''import os, sys, subprocess
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
    return [r for r in results if r[2] > 0.35]

PROMPT = """{prompt}"""
MAX_ATTEMPTS = 4
for attempt in range(MAX_ATTEMPTS):
    seed = 9000 + attempt * 1000
    img = pipe(PROMPT, num_inference_steps=28, guidance_scale=4.5, height=896, width=1152,
               generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    m = float(np.asarray(img).mean())
    if m < 5:
        print("RETRY attempt", attempt+1, "black/NaN frame", flush=True)
        torch.cuda.empty_cache()
        continue
    hits = has_text(img)
    if hits:
        print("RETRY attempt", attempt+1, "text detected:", [h[1] for h in hits], flush=True)
        torch.cuda.empty_cache()
        continue
    img.save("/kaggle/working/norjak_promo_scene.jpeg", quality=94)
    print("DONE attempt", attempt+1, "meanpix", round(m,1), flush=True)
    break
else:
    print("FAILED all attempts", flush=True)
'''.format(prompt=PROMPT)


def main():
    os.makedirs(KERNEL_DIR, exist_ok=True)
    with open(os.path.join(KERNEL_DIR, "gen_promo.py"), "w", encoding="utf-8") as f:
        f.write(KERNEL_CODE)
    json.dump({
        "id": KERNEL_ID,
        "title": "shadow-gasp-norjak-promo-scene",
        "code_file": "gen_promo.py",
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

    # The account's 2-concurrent-GPU-session cap is being used by the live
    # batch-pregen Actions workflow right now -- retry the push instead of
    # failing outright, since a slot frees up every few minutes as each
    # pregen day's kernel finishes.
    print(f"Pushing kernel {KERNEL_ID} ...")
    for attempt in range(1, 21):
        r = subprocess.run(["kaggle", "kernels", "push", "-p", "."], cwd=KERNEL_DIR, capture_output=True, text=True)
        if r.returncode == 0:
            break
        if "Maximum batch GPU session count" in (r.stdout + r.stderr):
            print(f"attempt {attempt}: GPU sessions full, waiting 60s ...")
            time.sleep(60)
            continue
        print(r.stdout, r.stderr, file=sys.stderr)
        raise RuntimeError("kaggle kernels push failed (not a GPU-cap issue)")
    else:
        raise RuntimeError("GPU sessions still full after 20 retries (~20 min) -- try again later")

    print("Polling for completion ...")
    while True:
        time.sleep(30)
        r = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID], capture_output=True, text=True)
        status = r.stdout.strip()
        print(status)
        if "COMPLETE" in status:
            break
        if "ERROR" in status or "CANCEL" in status:
            print(r.stdout, r.stderr, file=sys.stderr)
            raise RuntimeError(f"kaggle kernel failed: {status}")

    out_dir = os.path.join(KERNEL_DIR, "out")
    subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", out_dir], check=True)
    src = os.path.join(out_dir, "norjak_promo_scene.jpeg")
    dst = os.path.join(HERE, "norjak_promo_scene.jpeg")
    if not os.path.exists(src):
        raise RuntimeError(f"{src} missing from kernel output -- check kernel log")
    os.replace(src, dst)
    print(f"Saved -> {dst}")


if __name__ == "__main__":
    main()
