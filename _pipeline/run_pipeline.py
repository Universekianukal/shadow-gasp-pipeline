"""Master orchestrator for a single shadow_gasp automated short — the local
equivalent of .github/workflows/pipeline.yml.

Usage:
    ANTHROPIC_API_KEY=... python3 run_pipeline.py /path/to/new-video-dir
    CASE="D.B. Cooper skyjacking" ANTHROPIC_API_KEY=... python3 run_pipeline.py ...

With no CASE, _pick_case.py chooses one that isn't in cases_used.json.

Each step is a skip-if-already-present script (mirrors the MindUnlocked
pattern), so re-running after a partial failure resumes instead of redoing
completed work. Copies this pipeline's helper scripts into the target
project directory on first run.

Prefer the GitHub Actions workflow for routine daily output — it renders on a
clean runner instead of an 8GB laptop, and auto-uploads. This script exists for
one-offs and for debugging a step in isolation.
"""
import os
import shutil
import subprocess
import sys

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(PIPELINE_DIR, "cases_used.json")
MUSIC_SRC = os.path.join(PIPELINE_DIR, "assets", "bed.wav")

HELPER_SCRIPTS = [
    "_gen_video_content.py", "_tts.py", "_transcribe.py",
    "_gen_flux_images.py", "_gen_cog_hook.py",
    "_build_composition.py", "_composition_template.html",
    "_gen_youtube_meta.py",
]

STEPS = [
    ("Generate script + 16 shot prompts + caption metadata", "_gen_video_content.py"),
    ("TTS (Kokoro, bm_george)", "_tts.py"),
    ("Transcribe (faster-whisper, word timestamps)", "_transcribe.py"),
    ("Generate 16 stills (FLUX.1-schnell via Kaggle)", "_gen_flux_images.py"),
    ("Animate hook clip (CogVideoX-5b I2V via Kaggle)", "_gen_cog_hook.py"),
    ("Build composition (auto shot timing + captions)", "_build_composition.py"),
    ("Generate YouTube metadata", "_gen_youtube_meta.py"),
]


def run_step(label, script, cwd, env):
    print(f"\n=== {label} ===")
    subprocess.run([sys.executable, os.path.join(cwd, script)], cwd=cwd, check=True, env=env)


def pick_case(env):
    """Resolve CASE, either from the environment or by auto-picking an unused
    one. _pick_case.py prints the pick; capture it so the rest of the run and
    the ledger agree on the exact string."""
    if env.get("CASE", "").strip():
        return env["CASE"].strip()
    print("\n=== Pick case (not already in cases_used.json) ===")
    r = subprocess.run(
        [sys.executable, os.path.join(PIPELINE_DIR, "_pick_case.py")],
        check=True, capture_output=True, text=True, env=env,
    )
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)
    line = next(l for l in r.stdout.splitlines() if l.startswith("Picked: "))
    return line[len("Picked: "):].strip()


def main():
    if len(sys.argv) < 2:
        print("usage: python3 run_pipeline.py /path/to/project-dir", file=sys.stderr)
        sys.exit(1)
    project = os.path.abspath(sys.argv[1])
    os.makedirs(os.path.join(project, "images", "seq"), exist_ok=True)
    os.makedirs(os.path.join(project, "music"), exist_ok=True)

    for f in HELPER_SCRIPTS:
        shutil.copy(os.path.join(PIPELINE_DIR, f), os.path.join(project, f))
    bed = os.path.join(project, "music", "bed.wav")
    if not os.path.exists(bed) and os.path.exists(MUSIC_SRC):
        shutil.copy(MUSIC_SRC, bed)

    env = dict(os.environ)
    env["CASE"] = pick_case(env)
    # Keep every step pointed at the channel-wide ledger, not the copy that
    # would otherwise land next to the scripts inside the project dir.
    env["CASES_LEDGER"] = LEDGER
    print(f"\nCASE = {env['CASE']}")

    for label, script in STEPS:
        run_step(label, script, project, env)

    print("\n=== Render ===")
    subprocess.run([
        "npx", "--yes", "hyperframes", "render",
        "--fps", "30", "--workers", "1", "--low-memory-mode", "-o", "final.mp4",
        "--browser-timeout", "300", "--protocol-timeout", "900000",
        "--player-ready-timeout", "180000",
    ], cwd=project, check=True, shell=(os.name == "nt"))

    print(f"\nDone. Output: {os.path.join(project, 'final.mp4')}")
    print("Upload with: CASE=... python3 _pipeline/_youtube_upload.py  (run from the project dir)")


if __name__ == "__main__":
    main()
