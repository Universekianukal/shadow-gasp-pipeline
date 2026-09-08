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
import _llm  # provider shim: Anthropic or Fireworks, see _pipeline/_llm.py

import _gen_video_content as gvc
import _kaggle_quota as kq
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

# --- Kaggle account failover -------------------------------------------------
# A long unattended batch outlives any single account's weekly GPU quota, and
# the failure mode when it runs out is the worst kind: `kernels push` succeeds
# and the kernel sits QUEUED forever, so the batch stalls without erroring.
# Rather than have a human notice and re-aim the chain at another account, the
# run now carries every slot's credentials and moves itself to the next one
# with quota the moment the current slot drops below MIN_SLOT_HOURS.
#
# kaggle==2.2.2 authenticates purely from the KAGGLE_API_TOKEN env var, so
# switching accounts is just handing the CLI a different token per subprocess
# -- no config file to rewrite, no re-auth step.
#
# One day of 16 PixArt stills costs ~0.5 GPU-hours. The threshold is 0.75 so a
# slot is abandoned while it still has room to finish, instead of being caught
# out mid-kernel with a day already reserved.
MIN_SLOT_HOURS = 0.75

# How many days may fail inside one chunk before it gives up. Low on purpose:
# each attempt reserves a real case out of the ledger, so grinding through a
# systemic problem burns unused cases as well as time.
MAX_DAY_FAILURES = 2

_dead_slots = set()   # slots proven out of quota or broken THIS run
_active_slot = None   # (slot, handle, token)


def _slot_order():
    """Configured slots, preferred one first, minus any already written off."""
    accounts = kq.parse_accounts(os.environ.get("KAGGLE_ACCOUNTS", ""))
    pref = (os.environ.get("KAGGLE_SLOT") or os.environ.get("KAGGLE_IMAGE_ACCOUNT") or "").strip()
    ordered = [a for a in accounts if a[0] == pref] + [a for a in accounts if a[0] != pref]
    return [a for a in ordered if a[0] not in _dead_slots]


def pick_slot(recheck=True):
    """-> (slot, handle, token) for an account that can actually pay for a day.

    Re-reads live quota rather than trusting the slot that worked last time:
    the whole point is to notice the current account running dry *between*
    days, which is exactly when a cached answer would be wrong."""
    global _active_slot

    if _active_slot and recheck:
        slot, handle, token = _active_slot
        r = kq.read_slot(slot, handle)
        if "error" in r:
            # Unreadable is not the same as empty -- a transient blip must not
            # cost us a working account. Keep using it; the kernel poll
            # deadline is the backstop if it really is dry.
            return _active_slot
        if r["left_h"] >= MIN_SLOT_HOURS:
            return _active_slot
        print(f"slot {slot} ({handle}) is down to {kq.fmt_hm(r['left_h'])} GPU "
              f"-- under the {MIN_SLOT_HOURS}h a day needs, switching account", flush=True)
        _dead_slots.add(slot)
        _active_slot = None

    for slot, handle in _slot_order():
        token = os.environ.get(f"KAGGLE_{slot}_API_TOKEN", "").strip()
        if not token:
            print(f"slot {slot} ({handle}): no KAGGLE_{slot}_API_TOKEN set, skipping", flush=True)
            continue
        r = kq.read_slot(slot, handle)
        if "error" not in r and r["left_h"] < MIN_SLOT_HOURS:
            print(f"slot {slot} ({handle}): only {kq.fmt_hm(r['left_h'])} GPU left, skipping", flush=True)
            _dead_slots.add(slot)
            continue
        left = "unreadable" if "error" in r else kq.fmt_hm(r["left_h"])
        print(f"stills now running on Kaggle slot {slot} ({handle}), {left} GPU left", flush=True)
        _active_slot = (slot, handle, token)
        return _active_slot

    raise RuntimeError(
        "every configured Kaggle slot is out of GPU quota "
        f"(tried: {', '.join(s for s, _ in kq.parse_accounts(os.environ.get('KAGGLE_ACCOUNTS', ''))) or 'none'}). "
        "Quota refills weekly -- the batch will resume once it does."
    )


# --- concurrent chunks --------------------------------------------------------
# Two chunks can now run at once over disjoint day ranges (one per Kaggle
# account), which halves the wall-clock for a long batch. Their day folders
# never overlap, but they do share two files -- cases_used.json and
# state.json -- so every write to those has to assume someone else is writing
# too. That is what the two helpers below are for.


def _git(*args, repo_root=None, check=True, capture=False, env=None):
    return subprocess.run(["git", *args], cwd=repo_root, check=check,
                          capture_output=capture, text=True,
                          env=({**os.environ, **env} if env else None))


def _merge_shared_files(repo_root, state):
    """Rebuild state.json and cases_used.json as origin's copy plus ours, and
    stage them. Used to resolve the one conflict two concurrent chunks
    reliably produce.

    Both files are appended to, and both chunks append in the same place (right
    after the last pre-existing day), so git sees overlapping insertions and
    stops. The merge is well-defined even though the textual one is not: day
    ranges are disjoint, so no key is genuinely contested and the union is
    simply correct."""
    for path, key, ident in ((STATE_PATH, "days", None),
                             (gvc.LEDGER_PATH, "cases", "case")):
        rel = os.path.relpath(path, repo_root).replace("\\", "/")
        # "Theirs" is HEAD: mid-rebase that is the upstream commit we are being
        # replayed onto, i.e. the other chunk's work.
        base = _read_json_at(repo_root, "HEAD", rel)
        if base is None:
            base = _read_json_at(repo_root, "FETCH_HEAD", rel)
        if base is None:
            continue
        # "Ours" must come from the commit being replayed (REBASE_HEAD), never
        # from the working tree -- mid-conflict that file is full of conflict
        # markers and does not parse as JSON, which is what made an earlier
        # version of this throw and abandon the rebase it was meant to fix.
        # state.json is the exception: the in-memory dict is authoritative and
        # already holds every day this chunk has done.
        mine = state if ident is None else _read_json_at(repo_root, "REBASE_HEAD", rel)
        if mine is None:
            continue
        if ident is None:
            merged = {**base[key], **mine[key]}          # our days win, theirs kept
        else:
            seen = {c[ident] for c in base[key]}
            merged = base[key] + [c for c in mine[key] if c[ident] not in seen]
        base[key] = merged
        json.dump(base, open(path, "w", encoding="utf-8"), indent=1)
        _git("add", rel, repo_root=repo_root, check=False)


def _read_json_at(repo_root, ref, rel):
    """The file as JSON at a git ref, or None if it is not there / not parseable."""
    r = _git("show", f"{ref}:{rel}", repo_root=repo_root, check=False, capture=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


def push_with_retry(repo_root, rels, message, remerge=None, attempts=6):
    """Commit `rels` and get them onto origin/main, conceding to whoever got
    there first.

    A plain pull --rebase && push is a coin flip once two chunks are running:
    both rebase onto the same base and one push is rejected as non-fast-forward.
    On a content conflict `remerge` rebuilds the shared JSON as a union and the
    rebase continues; only if that fails is the rebase aborted and retried,
    because a rebase left half-finished makes every later git command in the run
    fail for reasons unrelated to the real problem."""
    import time

    _git("add", *rels, repo_root=repo_root)
    status = _git("status", "--porcelain", *rels, repo_root=repo_root, capture=True)
    if not status.stdout.strip():
        return
    _git("commit", "-m", message, repo_root=repo_root)

    for attempt in range(attempts):
        _git("fetch", "origin", GITHUB_BRANCH, repo_root=repo_root, check=False)
        r = _git("pull", "--rebase", "--autostash", "origin", GITHUB_BRANCH,
                 repo_root=repo_root, check=False, capture=True)
        if r.returncode != 0:
            resolved = False
            if remerge:
                try:
                    remerge(repo_root)
                    # GIT_EDITOR=true: --continue would otherwise open an editor
                    # for the commit message and hang the runner forever.
                    c = _git("rebase", "--continue", repo_root=repo_root, check=False,
                             capture=True, env={"GIT_EDITOR": "true"})
                    resolved = c.returncode == 0
                except Exception as e:
                    print(f"remerge failed: {e!r}", file=sys.stderr)
            if not resolved:
                print(f"rebase conflict unresolved ({r.stderr.strip()[:200]}), retrying",
                      file=sys.stderr)
                _git("rebase", "--abort", repo_root=repo_root, check=False)
                time.sleep(3 + 4 * attempt)
                continue
        p = _git("push", "origin", f"HEAD:{GITHUB_BRANCH}", repo_root=repo_root,
                 check=False, capture=True)
        if p.returncode == 0:
            return
        print(f"push rejected (attempt {attempt + 1}/{attempts}), refreshing and retrying",
              file=sys.stderr)
        time.sleep(3 + 4 * attempt)
    raise RuntimeError(f"could not push '{message}' after {attempts} attempts")


def refresh_ledger():
    """Re-read cases_used.json from origin so a pick sees reservations the
    other chunk has already made.

    Without this each chunk dedups against the ledger as it looked when the
    chunk started, and two chunks running for hours would drift into picking
    the same stories. This narrows the window to the seconds between reading
    the ledger and pushing our own reservation."""
    repo_root = os.path.dirname(PIPELINE_DIR)
    rel = os.path.relpath(gvc.LEDGER_PATH, repo_root).replace("\\", "/")
    r = _git("fetch", "origin", GITHUB_BRANCH, repo_root=repo_root, check=False, capture=True)
    if r.returncode != 0:
        print(f"ledger refresh: fetch failed ({r.stderr.strip()[:150]}), using local copy",
              file=sys.stderr)
        return gvc.load_ledger()
    _git("checkout", "FETCH_HEAD", "--", rel, repo_root=repo_root, check=False)
    return gvc.load_ledger()


def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH, encoding="utf-8"))
    return {"days": {}}


def save_state(state):
    json.dump(state, open(STATE_PATH, "w", encoding="utf-8"), indent=1)


def run_flux_for_day(day_dir, shots, day_num):
    """Generate this day's 16 stills, moving to another Kaggle account if the
    current one cannot finish the job.

    Retrying the day on a fresh account is safe and cheap: the kernel writes
    nothing outside /kaggle/working, and a day is only marked done once its
    images are actually in hand, so a half-spent slot costs a repeat of that
    one day rather than the batch."""
    # images/seq/, not images/ — matches _gen_flux_images.py's convention, which
    # _build_composition.py's template and the hyperframes file server both
    # hardcode as "images/seq/NN.jpeg". Getting this wrong produces a
    # composition where every artwork layer 404s and renders solid black.
    seq_dir = os.path.join(day_dir, "images", "seq")
    os.makedirs(seq_dir, exist_ok=True)
    if os.path.exists(os.path.join(seq_dir, "16.jpeg")):
        print(f"day {day_num}: images already present, skipping generation")
        return

    global _active_slot
    last_err = None
    while True:
        slot, handle, token = pick_slot()
        try:
            _run_kernel_on_slot(day_dir, shots, day_num, seq_dir, handle, token)
            return
        except Exception as e:
            # Any failure here -- a queued-forever kernel hitting the deadline,
            # a kernel ERROR, a push rejected -- is treated as "this account
            # cannot do the job". Writing the slot off is the conservative
            # call: the alternative is retrying the same dry account in a loop.
            print(f"day {day_num}: stills failed on slot {slot} ({handle}): {e!r}", file=sys.stderr)
            last_err = e
            _dead_slots.add(slot)
            _active_slot = None
            if not _slot_order():
                raise RuntimeError(
                    f"day {day_num}: no Kaggle account left to try (last error: {last_err!r})"
                ) from last_err
            print(f"day {day_num}: failing over to the next Kaggle account", flush=True)


def _run_kernel_on_slot(day_dir, shots, day_num, seq_dir, kaggle_user, token):
    """One attempt at this day's stills on one specific Kaggle account.

    Model depends on day_num: FLUX.1-schnell below MODEL_SWITCH_DAY,
    PixArt-Sigma from there on (see MODEL_SWITCH_DAY's comment for why)."""
    import time

    # The token is handed to each kaggle subprocess explicitly rather than
    # mutated into os.environ, so which account a call spends is visible at the
    # call site and two slots can never be half-applied to one command.
    env = {**os.environ, "KAGGLE_API_TOKEN": token, "KAGGLE_IMAGE_USERNAME": kaggle_user}

    use_pixart = day_num >= MODEL_SWITCH_DAY
    model_slug = "pixart" if use_pixart else "flux"
    kernel_dir = os.path.join(day_dir, f"_kaggle_{model_slug}_kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    # kaggle_user is the account this attempt is aimed at, passed in by the
    # failover loop. It used to be re-read from the environment here, which
    # would have quietly pinned every retry to the original account and made
    # the whole failover a no-op -- the kernel id must name the account whose
    # token is about to push it.
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

    print(f"day {day_num}: pushing kernel {kernel_id} ({model_slug}) as {kaggle_user} ...")
    subprocess.run(["kaggle", "kernels", "push", "-p", "."], cwd=kernel_dir, check=True, env=env)

    # Two deadlines, because the two ways a day can hang mean different things.
    #
    # QUEUE_DEADLINE is the one that matters for failover. An account with no
    # weekly GPU quota left does not make `kernels push` fail -- Kaggle accepts
    # the kernel and leaves it QUEUED indefinitely. A healthy kernel starts
    # within a couple of minutes, so a kernel still QUEUED after 12 is not slow,
    # it is unfundable: give up on the slot immediately rather than burn most of
    # an hour proving it. This is what makes the switch to another account fast
    # instead of eventually.
    #
    # RUN_DEADLINE only applies once the kernel is actually RUNNING, where the
    # real work is ~25-35 min. Without either deadline the loop polls until
    # GitHub kills the job at its 350-minute cap, which reports as `cancelled`
    # rather than failed -- so `if: failure()` notifiers stay silent and the day
    # is stranded mid-reserve. That is exactly how day 61 burned 24 hours.
    print(f"day {day_num}: polling for completion ...")
    QUEUE_DEADLINE = 12 * 60
    RUN_DEADLINE = 90 * 60
    started = time.time()
    ever_ran = False
    while True:
        time.sleep(30)
        r = subprocess.run(["kaggle", "kernels", "status", kernel_id],
                           capture_output=True, text=True, env=env)
        status = r.stdout.strip()
        print(f"day {day_num}: {status}")
        if "COMPLETE" in status:
            break
        if "ERROR" in status or "CANCEL" in status:
            print(r.stdout, r.stderr, file=sys.stderr)
            raise RuntimeError(f"day {day_num}: kaggle kernel failed: {status}")
        if "RUNNING" in status and not ever_ran:
            ever_ran = True
            started = time.time()  # the run clock starts when the run does
        waited = time.time() - started
        if not ever_ran and waited > QUEUE_DEADLINE:
            raise RuntimeError(
                f"day {day_num}: kernel {kernel_id} never left the queue in "
                f"{QUEUE_DEADLINE // 60} min ({status}). This account is almost certainly "
                "out of weekly GPU quota -- Kaggle queues instead of refusing."
            )
        if ever_ran and waited > RUN_DEADLINE:
            raise RuntimeError(
                f"day {day_num}: kernel {kernel_id} still '{status}' after "
                f"{RUN_DEADLINE // 60} min of running."
            )

    out_dir = os.path.join(kernel_dir, "out")
    subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", out_dir], check=True, env=env)
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
    push_with_retry(repo_root, [rel_dir], f"batch: day {day_num:02d} assets")

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
            # ⚠️ POSITIONAL, not append. _sync_youtube_status writes a day's video columns at
            # `row = day + 1`, so a day's label row MUST live there too. append() puts it at the
            # bottom in write order, which matches day+1 only while days are appended in strict
            # order into an empty sheet -- true for days 1-32, which is why this went unnoticed.
            # Once days arrived out of order the sheet grew duplicates (39 and 41 twice) and
            # holes (40 and 42 had no positional row), with the video data on one row and the
            # label on another.
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range=f"{SHEET_TAB}!A{day_num + 1}:H{day_num + 1}",
                valueInputOption="USER_ENTERED",
                body={"values": row},
            ).execute()
            return
        except Exception as e:
            print(f"day {day_num}: sheet append attempt {attempt + 1} failed ({e!r}), retrying", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"day {day_num}: could not append sheet row after 4 attempts")


def commit_state(reason, extra_paths=(), state=None):
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
    remerge = (lambda root: _merge_shared_files(root, state)) if state is not None else None
    push_with_retry(repo_root, rels, f"batch: {reason}", remerge=remerge)


def main():
    new_days_budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    os.makedirs(BATCH_DIR, exist_ok=True)
    state = load_state()

    # Day range this chunk owns. Two chunks running at once are given disjoint
    # ranges (one Kaggle account each) so they never build the same day: without
    # a range both scan from day 1 for the first unfinished day and would race
    # onto the same one. START_DAY also skips the scan over hundreds of finished
    # days, and END_DAY is what stops a chain running past its half of the batch
    # and into the other chain's.
    start_day = int(os.environ.get("BATCH_START_DAY") or 1)
    end_day = int(os.environ.get("BATCH_END_DAY") or 0) or None

    client = _llm.client()
    sheets = get_sheets_service()

    ledger = gvc.load_ledger()
    used_cases = [c["case"] for c in ledger["cases"]]

    new_days_done = 0
    day_num = start_day - 1
    failures = 0
    while new_days_done < new_days_budget:
        day_num += 1
        if end_day and day_num > end_day:
            print(f"reached the end of this chunk's range (day {end_day})")
            break
        key = str(day_num)
        day_state = state["days"].get(key, {})
        if day_state.get("done"):
            used_cases.append(day_state["case"])
            continue

        # One bad day must not end the batch. Before this, any exception
        # anywhere in a day propagated out of main(), the job went red, and
        # dispatch_next_chunk() never ran -- so a single transient LLM or
        # Kaggle hiccup silently stopped an unattended run of dozens of days
        # and waited for a human. A failed day leaves its reservation in
        # state.json without `done`, and the next chunk walks the same day
        # numbers from the start, so it is picked up and retried for free.
        try:
            day_dir = os.path.join(BATCH_DIR, f"day{day_num:02d}")
            os.makedirs(day_dir, exist_ok=True)

            pick_path = os.path.join(day_dir, "pick.json")
            if "case" in day_state:
                case = day_state["case"]
                angle = day_state.get("angle", "")
                print(f"day {day_num}: resuming existing pick: {case}")
            else:
                # Re-read the ledger from origin first. The other chunk has been
                # reserving cases for however long this one has been busy, and
                # deduping against a copy loaded at startup is how two chunks
                # end up covering the same story hours apart.
                ledger = refresh_ledger()
                used_cases = [c["case"] for c in ledger["cases"]]
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
                commit_state(f"day {day_num:02d} case picked", extra_paths=[gvc.LEDGER_PATH, pick_path], state=state)
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
                commit_state(f"day {day_num:02d} sheet row logged", state=state)

            day_state["done"] = True
            state["days"][key] = day_state
            save_state(state)
            commit_state(f"day {day_num:02d} complete", state=state)
            new_days_done += 1
            print(f"day {day_num}: DONE — {case} ({new_days_done}/{new_days_budget} this run)")
        except Exception as e:
            failures += 1
            print(f"day {day_num}: FAILED ({failures}/{MAX_DAY_FAILURES} allowed this chunk): {e!r}",
                  file=sys.stderr)
            if not _slot_order():
                # Not a bad day -- there is no GPU left anywhere. Walking on
                # would reserve fresh cases for days that cannot be built.
                print("no Kaggle account has GPU quota left; ending this chunk here",
                      file=sys.stderr)
                break
            if failures >= MAX_DAY_FAILURES:
                print("too many failed days in one chunk; ending it here rather than "
                      "reserving more cases against a problem that is not going away",
                      file=sys.stderr)
                break

    print(f"\nThis run: {new_days_done} new day(s) completed, through day {day_num}, "
          f"{failures} failed.")
    done_count = sum(1 for v in state["days"].values() if v.get("done"))
    if end_day:
        # A ranged chunk is finished when ITS range is, not when the whole batch
        # is. Judging by the global count would make the first chain to finish
        # keep chaining into the other chain's days.
        batch_complete = all(state["days"].get(str(d), {}).get("done")
                             for d in range(start_day, end_day + 1))
    else:
        batch_complete = done_count >= TOTAL_DAYS
    notify_pregen_done(new_days_done, day_num, batch_complete)

    if batch_complete:
        print(f"Batch fully complete: {done_count}/{TOTAL_DAYS} days done.")
    elif new_days_done:
        dispatch_next_chunk(new_days_budget)
    else:
        # Chaining on a chunk that achieved nothing would spin the workflow
        # against the same blocker indefinitely. Stop and leave it to a human.
        print("this chunk completed no days -- not chaining another, "
              "since it would hit whatever stopped this one.", file=sys.stderr)


# Total size of the batch. Self-chaining (see dispatch_next_chunk) keeps
# triggering new chunks of this same size until this many days are done,
# so a human only has to start the batch once, not re-trigger every chunk.
#
# ⚠️ This is a cumulative TOTAL, not "how many more". It was hardcoded to 30
# and stayed there while the batch grew to 61 done days, which made
# `done_count >= TOTAL_DAYS` true before the first chunk even started: every
# run reported "Batch fully complete" and never chained, so extending the
# batch quietly degraded into one chunk per manual trigger. Overridable
# per-run via the workflow's `total` input -> BATCH_TOTAL_DAYS.
TOTAL_DAYS = int(os.environ.get("BATCH_TOTAL_DAYS") or 30)


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
        # total and kaggle_account must be carried forward explicitly. A
        # dispatch that omits them gets the workflow's defaults instead: total
        # falls back to 30 (ending the chain immediately) and the account falls
        # back to the KAGGLE_IMAGE_ACCOUNT variable, drifting the chain off the
        # account whose quota the guard actually verified.
        body = json.dumps({
            "ref": "main",
            "inputs": {
                "days": str(new_days_budget),
                "notify_chat_id": os.environ.get("NOTIFY_CHAT_ID", ""),
                "total": str(TOTAL_DAYS),
                "kaggle_account": os.environ.get("KAGGLE_SLOT", ""),
                "start_day": os.environ.get("BATCH_START_DAY", ""),
                "end_day": os.environ.get("BATCH_END_DAY", ""),
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
            # ⚠️ The User-Agent is NOT decoration. Cloudflare's bot protection fingerprints
            # Python urllib's default UA and blocks it at the edge with 403 / "error code:
            # 1010" -- before the request ever reaches the Worker or its secret check. curl
            # sails through, which is why a manual curl test does NOT prove this works and why
            # the same secret succeeds from the /batch/failed step (that one uses curl).
            # Three other call sites were patched for this on 2026-08-13; this one was missed,
            # and day 58's "pregen chunk done" message died here with a 403.
            headers={"X-Batch-Notify-Secret": secret, "Content-Type": "application/json",
                     "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/120.0.0.0 Safari/537.36")},
        )
        urllib.request.urlopen(req).read()
    except Exception as e:
        print(f"pregen_done notify failed (non-fatal): {e!r}")


if __name__ == "__main__":
    main()
