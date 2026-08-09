#!/usr/bin/env python3
"""
dewatermark.py - remove the Google Flow / Veo sparkle watermark.

METHOD
------
1. ALPHA-MATTE DECOMPOSITION (removes the star body)
   The watermark is alpha-composited onto the frame:
        observed = (1 - alpha) * background + alpha * W
   so the background is recovered by exact algebraic inversion:
        background = (observed - alpha * W) / (1 - alpha)
   alpha and W were solved once from 240 frames of real footage, so this
   recovers REAL pixels rather than inventing them.

2. RIM REPAIR (~2px collar)
   The star was composited and THEN H.264-encoded, so the codec baked ringing
   around its sharp edges. That happened after compositing and cannot be
   inverted, so a thin collar is inpainted.

SAFETY
------
* Detects the watermark before touching anything. A file without it is SKIPPED,
  never "cleaned" (applying the matte to clean footage would damage it).
* Skips files whose output already exists, so re-running over a folder is
  idempotent.
* Verifies every result with the star-vs-collar metric:
      untouched watermark ~ +57 | clean ~ 0 +/- 2

USAGE
-----
  python dewatermark.py clip.mp4
  python dewatermark.py photo.jpg
  python dewatermark.py "C:/Users/me/Downloads" -o cleaned/
  python dewatermark.py clip.mp4 --force        # ignore detection/skip guards
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

MATTE_SIZE = 160
RIGHT_OFF = 262
BOTTOM_OFF = 270
SEARCH_PAD = 30

INNER_T = 0.25            # matte value defining the star body
COLLAR_R = (42, 54)       # collar ring, immediately outside the star
DETECT_T = 18.0           # below this score, assume no watermark -> skip
CLEAN_V = 6.0             # pass threshold, video (control sd was 1.6)
CLEAN_I = 14.0            # pass threshold, single image (noisier)

# Per-clip alpha calibration. The solved matte is right on average but a few
# percent off on any given clip, and the error only becomes VISIBLE on a bright
# watermark corner: unblend() caps alpha at 0.97*observed/W, so a dark corner
# throttles alpha and hides the error, while a lit one applies it in full and
# leaves a dark blotch. Measured best scales across three real clips: 0.96
# (lit door), 1.02 (dark pavement), 1.04 (noisy control). No single constant
# serves all three, so solve it per clip against control patches instead.
CAL_RANGE = (0.80, 1.20)  # plausible bounds; outside this, distrust the fit
CAL_STEP = 0.01
CAL_MAX_CTRL_SD = 3.0     # controls noisier than this can't resolve a scale

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------------------------------------------------------------- tools

def find_media_tool(name):
    """Find FFmpeg on PATH, including the standard WinGet installation."""
    found = shutil.which(name)
    if found:
        return found

    # WinGet installs Gyan's FFmpeg outside PATH by default on some Windows
    # setups.  Search version-agnostically so an upgrade needs no code change.
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        pattern = os.path.join(
            local_app_data, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg_*",
            "ffmpeg-*", "bin", f"{name}.exe",
        )
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    return None


FFMPEG = find_media_tool("ffmpeg")
FFPROBE = find_media_tool("ffprobe")


def require_media_tools():
    missing = [name for name, path in (("ffmpeg", FFMPEG), ("ffprobe", FFPROBE))
               if not path]
    if not missing:
        return True
    print("! Missing required media tool(s): " + ", ".join(missing))
    print("  Install FFmpeg, or add its bin folder to PATH, then run again.")
    return False


# ---------------------------------------------------------------- matte

def load_matte():
    a = np.load(os.path.join(HERE, "matte_alpha.npy")).astype(np.float32)
    W = np.load(os.path.join(HERE, "matte_W.npy")).astype(np.float32)
    rim = np.load(os.path.join(HERE, "matte_rim.npy")).astype(np.uint8)
    return a, W, rim


def masks(a):
    ag = a.mean(axis=2)
    s = ag.shape[0]
    yy, xx = np.mgrid[0:s, 0:s]
    r = np.hypot(xx - s / 2.0, yy - s / 2.0)
    return ag > INNER_T, (r >= COLLAR_R[0]) & (r <= COLLAR_R[1])


def repair_mask(rim, expand=0):
    """Optionally widen the learned codec-ringing collar by `expand` pixels.

    The recovered star body is left intact; this only grows the small mask used
    for inpainting the post-encode halo around its edge.
    """
    mask = (rim > 0).astype(np.uint8)
    if expand:
        size = 2 * expand + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
    return mask * 255


def star_score(stack, a):
    """Star-body brightness minus its immediate collar. The metric that
    actually separates a watermarked file from a clean one."""
    inner, collar = masks(a)
    g = stack.mean(axis=3)
    return float((g[:, inner].mean(axis=1) - g[:, collar].mean(axis=1)).mean())


# ---------------------------------------------------------------- io

def probe(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(n) / float(d)


def anchor(w, h):
    return w - RIGHT_OFF, h - BOTTOM_OFF


def read_crop_stream(path, x, y, cw, ch):
    p = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", path, "-vf", f"crop={cw}:{ch}:{x}:{y}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=10 ** 8)
    n, fr = cw * ch * 3, []
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        fr.append(np.frombuffer(b, np.uint8).reshape(ch, cw, 3))
    p.stdout.close(); p.wait()
    return np.array(fr).astype(np.float32)


# ---------------------------------------------------------------- align

def median_excess(stack):
    h, w = stack.shape[1], stack.shape[2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2.0, yy - h / 2.0)
    ring = r > (min(h, w) / 2.0 - 12)
    bg = stack[:, ring, :].mean(axis=1)
    ex = np.median(stack - bg[:, None, None, :], axis=0)
    for c in range(3):
        ex[:, :, c] -= np.median(ex[:, :, c][ring])
    return ex


def align_video(path, x0, y0, a):
    xs, ys = max(x0 - SEARCH_PAD, 0), max(y0 - SEARCH_PAD, 0)
    side = MATTE_SIZE + 2 * SEARCH_PAD
    st = read_crop_stream(path, xs, ys, side, side)
    if len(st) < 8:
        return x0, y0
    field = median_excess(st).mean(axis=2).astype(np.float32)
    ref = a.mean(axis=2).astype(np.float32)
    _, _, _, loc = cv2.minMaxLoc(cv2.matchTemplate(field, ref, cv2.TM_CCORR))
    return xs + loc[0], ys + loc[1]


def align_image(img, x0, y0, a):
    xs, ys = max(x0 - SEARCH_PAD, 0), max(y0 - SEARCH_PAD, 0)
    side = MATTE_SIZE + 2 * SEARCH_PAD
    if ys + side > img.shape[0] or xs + side > img.shape[1]:
        return x0, y0
    reg = img[ys:ys + side, xs:xs + side].mean(axis=2).astype(np.float32)
    hp = reg - cv2.GaussianBlur(reg, (0, 0), 6.0)
    ref = a.mean(axis=2).astype(np.float32)
    ref = ref - ref.mean()
    _, _, _, loc = cv2.minMaxLoc(cv2.matchTemplate(hp, ref, cv2.TM_CCORR))
    return xs + loc[0], ys + loc[1]


def in_bounds(x0, y0, w, h):
    return x0 >= 0 and y0 >= 0 and x0 + MATTE_SIZE <= w and y0 + MATTE_SIZE <= h


# ---------------------------------------------------------------- core

def apply_matte(reg, a, W):
    """Algebraic un-blend of one region. reg may be a single HxWx3 patch or a
    stacked NxHxWx3 -- the arithmetic broadcasts either way, which is what lets
    the calibrator score a whole frame stack without re-encoding anything."""
    # cap alpha so the result can never be driven negative and crushed to black
    a_eff = np.minimum(a, 0.97 * reg / np.maximum(W, 1.0))
    return np.clip((reg - a_eff * W) / (1.0 - a_eff), 0, 255)


def control_patches(path, x0, y0):
    """Four watermark-free patches from the same frames, same size as the matte.

    The README's hardest-won lesson: never optimise a metric without a control.
    The star metric picks up ordinary scene variation too, so "score 0" is the
    wrong target -- "score matches nearby clean scene" is the right one."""
    offsets = ((MATTE_SIZE + 40, 0), (0, MATTE_SIZE + 40),
               (MATTE_SIZE + 40, MATTE_SIZE + 40), (2 * MATTE_SIZE, 0))
    out = []
    for dx, dy in offsets:
        cx, cy = x0 - dx, y0 - dy
        if cx >= 0 and cy >= 0:
            out.append(read_crop_stream(path, cx, cy, MATTE_SIZE, MATTE_SIZE))
    return out


def solve_alpha_scale(star, controls, a, W):
    """Pick the alpha scale whose corrected star region blends into the scene.

    Returns (scale, target, note). Falls back to 1.0 whenever the controls are
    too noisy to resolve a scale -- a bad fit is worse than no fit."""
    if not controls:
        return 1.0, None, "no control patches available"
    scores = [star_score(c, a) for c in controls]
    target, sd = float(np.mean(scores)), float(np.std(scores))
    if sd > CAL_MAX_CTRL_SD:
        return 1.0, target, f"controls too noisy (sd {sd:.1f}), using 1.00"
    lo, hi = CAL_RANGE
    best = None
    for s in np.arange(lo, hi + CAL_STEP / 2, CAL_STEP):
        dev = abs(star_score(apply_matte(star, a * s, W), a) - target)
        if best is None or dev < best[0]:
            best = (dev, float(s))
    scale = best[1]
    if scale <= lo + CAL_STEP / 2 or scale >= hi - CAL_STEP / 2:
        return 1.0, target, f"fit hit the {scale:.2f} bound, distrusted — using 1.00"
    return scale, target, f"control {target:+.2f} (sd {sd:.2f})"


def unblend(img, x0, y0, a, W, rim):
    h, w = img.shape[:2]
    if not in_bounds(x0, y0, w, h):
        return img
    reg = img[y0:y0 + MATTE_SIZE, x0:x0 + MATTE_SIZE].astype(np.float32)
    # cap alpha so the result can never be driven negative and crushed to black
    a_eff = np.minimum(a, 0.97 * reg / np.maximum(W, 1.0))
    out = np.clip((reg - a_eff * W) / (1.0 - a_eff), 0, 255).astype(np.uint8)
    out = cv2.inpaint(out, rim, 3, cv2.INPAINT_TELEA)
    img[y0:y0 + MATTE_SIZE, x0:x0 + MATTE_SIZE] = out
    return img


def do_video(inp, outp, a, W, rim, align=True, crf=16, force=False, rim_expand=1,
             alpha_scale=None):
    w, h, fps = probe(inp)
    x0, y0 = anchor(w, h)
    if not in_bounds(x0, y0, w, h):
        print(f"  ! {w}x{h}: watermark anchor outside frame, skipped")
        return False

    if align:
        ax, ay = align_video(inp, x0, y0, a)
        dx, dy = ax - x0, ay - y0
        if abs(dx) <= SEARCH_PAD - 2 and abs(dy) <= SEARCH_PAD - 2:
            if (dx, dy) != (0, 0):
                print(f"  auto-aligned ({dx:+d},{dy:+d})")
            x0, y0 = ax, ay

    star = read_crop_stream(inp, x0, y0, MATTE_SIZE, MATTE_SIZE)
    before = star_score(star, a)
    if before < DETECT_T and not force:
        print(f"  no watermark detected (score {before:+.1f}) - skipped, file untouched")
        return True

    if alpha_scale is None:
        alpha_scale, _, note = solve_alpha_scale(star, control_patches(inp, x0, y0), a, W)
        print(f"  alpha calibrated x{alpha_scale:.2f} — {note}")
    a = a * alpha_scale

    repair = repair_mask(rim, rim_expand)
    rd = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", inp, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=10 ** 8)
    wr = subprocess.Popen(
        [FFMPEG, "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
         "-i", inp, "-map", "0:v:0", "-map", "1:a?",
         "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", outp],
        stdin=subprocess.PIPE)

    n, count = w * h * 3, 0
    while True:
        b = rd.stdout.read(n)
        if len(b) < n:
            break
        f = np.frombuffer(b, np.uint8).reshape(h, w, 3).copy()
        wr.stdin.write(unblend(f, x0, y0, a, W, repair).tobytes())
        count += 1
    rd.stdout.close(); rd.wait()
    wr.stdin.close(); wr.wait()

    after = star_score(read_crop_stream(outp, x0, y0, MATTE_SIZE, MATTE_SIZE), a)
    ok = abs(after) < CLEAN_V
    print(f"  {count} frames  score {before:+.1f} -> {after:+.1f}  "
          f"[{'CLEAN' if ok else 'CHECK OUTPUT'}]  -> {os.path.basename(outp)}")
    return True


def do_image(inp, outp, a, W, rim, align=True, force=False, rim_expand=1,
             alpha_scale=None):
    img = cv2.imread(inp)
    if img is None:
        print("  ! unreadable")
        return False
    h, w = img.shape[:2]
    x0, y0 = anchor(w, h)
    if not in_bounds(x0, y0, w, h):
        print(f"  ! {w}x{h}: watermark anchor outside frame, skipped")
        return False

    if align:
        ax, ay = align_image(img, x0, y0, a)
        if abs(ax - x0) <= SEARCH_PAD - 2 and abs(ay - y0) <= SEARCH_PAD - 2:
            x0, y0 = ax, ay

    reg = img[y0:y0 + MATTE_SIZE, x0:x0 + MATTE_SIZE][None].astype(np.float32)
    before = star_score(reg, a)
    if before < DETECT_T and not force:
        print(f"  no watermark detected (score {before:+.1f}) - skipped, file untouched")
        return True

    # A single frame has no temporal averaging, so a fitted scale would chase
    # scene noise -- calibrate only when explicitly asked for.
    a = a * (alpha_scale if alpha_scale is not None else 1.0)
    img = unblend(img, x0, y0, a, W, repair_mask(rim, rim_expand))
    ext = os.path.splitext(outp)[1].lower()
    cv2.imwrite(outp, img,
                [cv2.IMWRITE_JPEG_QUALITY, 97] if ext in (".jpg", ".jpeg") else [])

    after = star_score(img[y0:y0 + MATTE_SIZE, x0:x0 + MATTE_SIZE][None].astype(np.float32), a)
    ok = abs(after) < CLEAN_I
    print(f"  {w}x{h}  score {before:+.1f} -> {after:+.1f}  "
          f"[{'CLEAN' if ok else 'CHECK OUTPUT'}]  -> {os.path.basename(outp)}")
    return True


# ---------------------------------------------------------------- cli

def out_path(inp, outdir):
    base, ext = os.path.splitext(os.path.basename(inp))
    return os.path.join(outdir or os.path.dirname(inp) or ".", f"{base}_clean{ext}")


def main():
    ap = argparse.ArgumentParser(description="Remove the Google Flow / Veo sparkle watermark.")
    ap.add_argument("inputs", nargs="+", help="files, or a folder")
    ap.add_argument("-o", "--output", help="output file (single input) or folder")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="ignore watermark detection and overwrite existing outputs")
    ap.add_argument("--rim-expand", type=int, default=1,
                    help="grow the learned edge-halo repair mask by N pixels (default: 1). "
                         "0 leaves a visible star-shaped outline; 3 smears a dark blotch over a "
                         "lit corner. 1 removes the outline without the smear -- picked by eye "
                         "against a real clip at 1.5x zoom, not by score alone.")
    ap.add_argument("--alpha-scale", type=float, default=None,
                    help="override the per-clip alpha calibration with a fixed scale "
                         "(1.0 = the raw solved matte); default is to solve it per clip")
    ap.add_argument("--crf", type=int, default=16)
    args = ap.parse_args()
    if args.rim_expand < 0:
        ap.error("--rim-expand must be zero or greater")

    if not require_media_tools():
        return 2

    a, W, rim = load_matte()

    files = []
    for i in args.inputs:
        if os.path.isdir(i):
            files += [os.path.join(i, f) for f in sorted(os.listdir(i))
                      if os.path.splitext(f)[1].lower() in VIDEO_EXT | IMAGE_EXT]
        else:
            files.append(i)
    files = [f for f in files if not os.path.splitext(f)[0].endswith("_clean")]
    if not files:
        print("nothing to do")
        return 0

    single = len(files) == 1
    outdir = None
    if args.output:
        # treat -o as a folder if it is one, has no file extension, or there
        # are several inputs; only a single input + a real filename is a file
        if os.path.isdir(args.output) or not os.path.splitext(args.output)[1] or not single:
            outdir = args.output
            os.makedirs(outdir, exist_ok=True)

    done = skipped = failed = 0
    for f in files:
        dest = args.output if (single and args.output and not outdir) else out_path(f, outdir)
        if os.path.exists(dest) and not args.force:
            skipped += 1
            continue
        print(os.path.basename(f))
        ext = os.path.splitext(f)[1].lower()
        try:
            ok = (do_video(f, dest, a, W, rim, not args.no_align, args.crf, args.force,
                           args.rim_expand, args.alpha_scale)
                  if ext in VIDEO_EXT else
                  do_image(f, dest, a, W, rim, not args.no_align, args.force,
                           args.rim_expand, args.alpha_scale))
            done += bool(ok)
            failed += (not ok)
        except Exception as e:
            print(f"  ! failed: {e}")
            failed += 1

    print(f"\nprocessed {done}, already-done {skipped}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
