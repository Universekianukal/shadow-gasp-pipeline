"""Regenerate a single still for a single batch day — for fixing one bad
shot (vision-QA-catchable artifact found after the fact, or after the QA
prompt itself gets strengthened) without burning GPU time re-doing all 16.

Usage: python _regen_shot.py <day_num> <shot_num>
Reads the day's shots.json for that shot's prompt, generates ONE replacement
image (model matches whatever _batch_pregen.py would have used for that day
— FLUX below MODEL_SWITCH_DAY, PixArt-Sigma from there on), and overwrites
images/seq/NN.jpeg in place.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _batch_pregen as bp

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) != 3:
        print("Usage: python _regen_shot.py <day_num> <shot_num>", file=sys.stderr)
        sys.exit(1)
    day_num = int(sys.argv[1])
    shot_num = int(sys.argv[2])

    day_dir = os.path.join(bp.BATCH_DIR, f"day{day_num:02d}")
    shots = json.load(open(os.path.join(day_dir, "shots.json"), encoding="utf-8"))
    shot = next(s for s in shots if s["n"] == shot_num)
    print(f"day {day_num}, shot {shot_num}: {shot['prompt'][:80]}...")

    use_pixart = day_num >= bp.MODEL_SWITCH_DAY
    model_slug = "pixart" if use_pixart else "flux"
    kernel_dir = os.path.join(day_dir, f"_kaggle_{model_slug}_regen_kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
    kaggle_user = os.environ.get("KAGGLE_IMAGE_USERNAME", "anuragmishra108")
    kernel_id = f"{kaggle_user}/shadow-gasp-batch-day{day_num:02d}-shot{shot_num:02d}-regen"

    # Both templates loop over a SHOTS list -- passing a list of one is enough
    # to reuse them unmodified for a single-shot regen.
    if use_pixart:
        code = bp.PIXART_KERNEL_TEMPLATE.format(shots_json=json.dumps([shot]), anthropic_api_key=anthropic_api_key)
    else:
        hf_token = os.environ["HF_TOKEN"]
        code = bp.FLUX_KERNEL_TEMPLATE.format(shots_json=json.dumps([shot]), hf_token=hf_token, anthropic_api_key=anthropic_api_key)
    open(os.path.join(kernel_dir, "gen_images.py"), "w", encoding="utf-8").write(code)
    json.dump({
        "id": kernel_id,
        "title": f"shadow-gasp-batch-day{day_num:02d}-shot{shot_num:02d}-regen",
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

    print(f"Pushing kernel {kernel_id} ({model_slug}) ...")
    subprocess.run(["kaggle", "kernels", "push", "-p", "."], cwd=kernel_dir, check=True)

    print("Polling for completion ...")
    while True:
        time.sleep(30)
        r = subprocess.run(["kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
        status = r.stdout.strip()
        print(status)
        if "COMPLETE" in status:
            break
        if "ERROR" in status or "CANCEL" in status:
            print(r.stdout, r.stderr, file=sys.stderr)
            raise RuntimeError(f"kaggle kernel failed: {status}")

    out_dir = os.path.join(kernel_dir, "out")
    subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", out_dir], check=True)
    src = os.path.join(out_dir, f"{shot_num:02d}.jpeg")
    dst = os.path.join(day_dir, "images", "seq", f"{shot_num:02d}.jpeg")
    if not os.path.exists(src):
        raise RuntimeError(f"{src} missing from kernel output")
    os.replace(src, dst)
    if shot_num == 1:
        # shot1.jpeg (the raw-URL copy linked from the Sheet) also needs updating
        import shutil
        shutil.copyfile(dst, os.path.join(day_dir, "shot1.jpeg"))
    print(f"day {day_num}, shot {shot_num}: regenerated -> {dst}")


if __name__ == "__main__":
    main()
