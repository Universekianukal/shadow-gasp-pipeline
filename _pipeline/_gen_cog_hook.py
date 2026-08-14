"""Animate shot 1 (the hook still) into a ~6s video clip via CogVideoX-5b
Image-to-Video on Kaggle GPU — the same model already proven on the Embassy
Siege project. Takes images/seq/01.jpeg + meta.json's hook_motion_prompt,
outputs images/seq/01.mp4.

Requires: kaggle CLI authenticated, images/seq/01.jpeg + meta.json present.
Skips entirely if images/seq/01.mp4 already exists.
"""
import json
import os
import subprocess
import sys
import time

PROJECT = os.getcwd()
SEQ_DIR = os.path.join(PROJECT, "images", "seq")
KERNEL_DIR = os.path.join(PROJECT, "_kaggle_cog_kernel")
KAGGLE_USER = os.environ.get("KAGGLE_VIDEO_USERNAME", "anuragmishra108")
SLUG = os.path.basename(PROJECT).replace("_", "-")[:35]
DATASET_ID = f"{KAGGLE_USER}/{SLUG}-cog-src"
KERNEL_ID = f"{KAGGLE_USER}/{SLUG}-cog-hook"

KERNEL_TEMPLATE = '''import os, sys, subprocess
def pip(*a): subprocess.run([sys.executable,"-m","pip","install","-q",*a], check=False)
def unpip(*a): subprocess.run([sys.executable,"-m","pip","uninstall","-y",*a], check=False)
pip("torch==2.4.1","torchvision==0.19.1","--index-url","https://download.pytorch.org/whl/cu121")
pip("diffusers==0.32.2","transformers==4.46.3","accelerate","sentencepiece","imageio","imageio-ffmpeg","ftfy","protobuf")
unpip("flash-attn","flash_attn","flash-attn-3","flash_attn_3")
os.environ["DIFFUSERS_NO_ADVISORY_WARNINGS"]="1"
import torch, numpy as np
from diffusers.utils import export_to_video, load_image
from diffusers import CogVideoXImageToVideoPipeline
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)

STYLE=", preserve comic-book cel-shaded illustration style, thick black ink outlines intact, subtle motion only, dynamic camera push-in"
NEG="static, still, frozen, motionless, looping, worst quality, blurry, jittery, distorted, warped, melting, dissolving, photoreal, realistic photo, extra limbs, deformed, watermark, text, caption"
PROMPT={motion_prompt!r}

img_path=None
for root,_,files in os.walk("/kaggle/input"):
    for f in files:
        if f.lower().endswith((".jpeg",".jpg",".png")):
            img_path=os.path.join(root,f)
print("INPUT IMAGE", img_path, flush=True)

cog=CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.float16)
# Tested enable_model_cpu_offload() as a speed fix (2026-08-01): it DID
# complete all 20 denoising steps faster, but then got OOM-killed by the OS
# during the VAE decode step every time (confirmed on 2 separate test runs,
# 2 different input images) -- the P100's 16GB isn't enough for that step's
# peak memory even with tiling/slicing on. Reverting to the slower-but-stable
# sequential offload; the real, safe win here is the reduced step count below
# (35->20), not the offload mode.
cog.enable_sequential_cpu_offload(); cog.vae.enable_tiling(); cog.vae.enable_slicing()
print("PIPE READY", flush=True)

img=load_image(img_path).resize((720,480))
v=cog(image=img, prompt=PROMPT+STYLE, negative_prompt=NEG, num_frames=49,
      num_inference_steps=20, guidance_scale=6.0,
      generator=torch.Generator(device="cuda").manual_seed(42)).frames[0]
export_to_video(v, "/kaggle/working/01.mp4", fps=8)
print("COG DONE meanpix", round(float(np.asarray(v[0]).mean()),1), flush=True)
print("ALL DONE", flush=True)
'''


def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def main():
    dst_mp4 = os.path.join(SEQ_DIR, "01.mp4")
    if os.path.exists(dst_mp4):
        print("images/seq/01.mp4 already present, skipping CogVideoX generation")
        return

    meta = json.load(open("meta.json"))
    motion_prompt = meta["hook_motion_prompt"]
    src_still = os.path.join(SEQ_DIR, "01.jpeg")
    assert os.path.exists(src_still), f"{src_still} not found — run _gen_flux_images.py first"

    # Upload the hook still as a Kaggle dataset (CogVideoX needs a dataset_source, unlike FLUX text-to-image)
    ds_dir = os.path.join(KERNEL_DIR, "dataset")
    os.makedirs(ds_dir, exist_ok=True)
    import shutil
    shutil.copy(src_still, os.path.join(ds_dir, "hook.jpeg"))
    json.dump({"title": f"{SLUG}-cog-src", "id": DATASET_ID, "licenses": [{"name": "CC0-1.0"}]},
               open(os.path.join(ds_dir, "dataset-metadata.json"), "w"))

    # create if new, else version (dataset may already exist from a prior run)
    r = subprocess.run(["kaggle", "datasets", "status", DATASET_ID], capture_output=True, text=True)
    if r.returncode == 0:
        run(["kaggle", "datasets", "version", "-p", ".", "-m", "update hook still", "-r", "zip"], cwd=ds_dir)
    else:
        run(["kaggle", "datasets", "create", "-p", ".", "-r", "zip"], cwd=ds_dir)

    kernel_dir = os.path.join(KERNEL_DIR, "kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    code = KERNEL_TEMPLATE.format(motion_prompt=motion_prompt)
    open(os.path.join(kernel_dir, "gen_cog.py"), "w", encoding="utf-8").write(code)
    json.dump({
        "id": KERNEL_ID,
        "title": f"{SLUG}-cog-hook",
        "code_file": "gen_cog.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        # NvidiaTeslaP100 (default) is sm_60, which Kaggle's stock torch no longer
        # builds for -> kernel dies at import/op time. T4 is sm_75, same 16GB, works.
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [DATASET_ID],
        "competition_sources": [],
        "kernel_sources": [],
    }, open(os.path.join(kernel_dir, "kernel-metadata.json"), "w"), indent=2)

    print(f"Pushing kernel {KERNEL_ID} ...")
    run(["kaggle", "kernels", "push", "-p", "."], cwd=kernel_dir)

    print("Polling for completion (CogVideoX-5b is slow, expect 15-30+ min)...")
    while True:
        time.sleep(60)
        r = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID], capture_output=True, text=True)
        status = r.stdout.strip()
        print(status)
        if "COMPLETE" in status:
            break
        if "ERROR" in status or "CANCEL" in status:
            print(r.stdout, r.stderr, file=sys.stderr)
            sys.exit(1)

    out_dir = os.path.join(KERNEL_DIR, "out")
    run(["kaggle", "kernels", "output", KERNEL_ID, "-p", out_dir])

    src = os.path.join(out_dir, "01.mp4")
    if os.path.exists(src):
        os.replace(src, dst_mp4)
        print(f"Hook clip saved to {dst_mp4}")
    else:
        print(f"WARNING: {src} missing from kernel output — falling back to static Ken Burns on shot 1", file=sys.stderr)


if __name__ == "__main__":
    main()
