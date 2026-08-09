# dewatermark — Google Flow / Veo sparkle watermark removal

## Daily use

**Double-click `CLEAN-DOWNLOADS.bat`.**

It scans your Downloads folder (videos *and* images), cleans anything carrying
the Flow watermark, and writes results to `Downloads\cleaned`. You can also drag
files or a folder onto it.

Or from a terminal:

```bash
cd D:/Hyperframe/shadow_gasp/_pipeline/dewatermark
python3 dewatermark.py <file-or-folder> -o cleaned/
```

Two things happen automatically on every video:

1. **Per-clip alpha calibration.** The solved matte is right on average but a
   few percent off on any given clip. The error is only *visible* on a bright
   watermark corner, because `unblend` caps alpha at `0.97*observed/W` — a dark
   corner throttles alpha and hides it, a lit one applies it in full and leaves
   a dark blotch. So the scale is now solved per clip against four
   watermark-free control patches from the same frames, and falls back to 1.00
   when those controls are too noisy (sd > 3) to resolve it. Measured: 0.95 on
   a lit-door clip, 1.00 on a noisy one, 1.02 on dark pavement — no single
   constant serves all three. Override with `--alpha-scale`.

2. **A 1px collar repair** (`--rim-expand`, default 1) for the codec ringing
   baked around the star's edge.

### Choosing `--rim-expand`

| value | result on a lit corner |
| --- | --- |
| 0 | body inverted cleanly, but a **star-shaped outline** is left |
| **1** | **outline gone, no smear — the default** |
| 2-3 | outline gone but neighbours smear in as a **dark blotch** |

Judged by eye at 1.5x zoom on a real clip, not by score alone: rim 0 scores
*better* (-1.9 vs -4.9) while looking *worse*, because the metric measures the
star body, not its border.

```bash
python3 dewatermark.py clip.mp4 -o cleaned/ --rim-expand 0   # legacy narrow mask
```

## Why it is safe to point at the whole folder

- **Detects before touching.** A file without the watermark is skipped, never
  "cleaned" — applying the matte to clean footage would damage it.
- **Idempotent.** Files already processed are skipped, so re-running is free.
- **Never edits in place.** Originals are untouched; output goes elsewhere.

Verified behaviour:

| input | score | result |
| --- | --- | --- |
| watermarked still | +61.8 -> +11.2 | cleaned |
| already-clean still | +11.4 | skipped, untouched |
| unrelated photo | +3.4 | skipped, untouched |

## The score

Star-body brightness minus the brightness of the collar immediately around it:

```
untouched watermark  ~ +57
clean (video)        ~   0  +/- 1.6   (measured on 4 control patches)
```

Thresholds: below **+18** = no watermark (skip). After cleaning, a video must
land within **+/-6** to report `CLEAN`.

An earlier metric (median excess vs a distant annulus) was abandoned — it
passed a bad output. A control test proved it was measuring ordinary scene
variation, not watermark.

## How it works

The watermark is **alpha-composited**, not painted on:

```
observed = (1 - alpha) * background + alpha * W
```

so the background is recovered by exact algebraic inversion:

```
background = (observed - alpha * W) / (1 - alpha)
```

This recovers **real pixels** — it does not invent them, so texture and grain
survive. Solved once from 240 frames:

| parameter | value |
| --- | --- |
| star centre | (899.5, 1740) in a 1080x1920 frame |
| radius | ~36 px (72x74 overall) |
| peak opacity | 0.53 |
| colour W | warm white, RGB (234, 219, 201) |
| anchor | 180 px in from the right and bottom edges |

Alpha is additionally capped per-pixel at `0.97 * observed / W` so the result can
never be driven negative and crushed to black.

A second stage inpaints a ~2px collar along the star's edge: the star was
composited and *then* H.264-encoded, so the codec baked ringing around its sharp
edges. That happened after compositing and cannot be inverted.

## Known limits

- **The collar repair is 1 px by default** (see the table above). A wider mask
  smears; a zero mask leaves the star's outline.
- **An outer halo may or may not exist.** Slope-vs-background at r=44-56px runs
  negative in the watermark corner (-0.29/-0.21) where a mirrored corner matched
  for vignetting runs positive (+0.06/+0.04), which looks like un-solved
  alpha out there. But fitting a radial alpha tail against control profiles
  returns amplitude **zero**, and converting those slopes to alpha implies
  0.25-0.48 at r=52 — impossible beside a star whose peak is 0.53. Two
  estimators, one null and one absurd. Unresolved; do not bake a wider matte
  from either without redoing the solve properly.
- **Single images are much less precise.** A clean still scored +11.4 where the
  240-frame video average scored +2.2 — one frame has no way to average scene
  variation away. Detection is still reliable (the watermark adds ~+50), but the
  image `CLEAN` verdict uses a looser +/-14 and should not be treated as proof.
- Position is anchored to the bottom-right corner and auto-aligned +/-30px per
  run, so small drift self-corrects. Bigger changes need recalibration.

## Files

`dewatermark.py`, `CLEAN-DOWNLOADS.bat`, `matte_alpha.npy` (160x160x3 matte),
`matte_W.npy` (colour), `matte_rim.npy` (rim mask).

## Recalibrating (only if Flow changes the watermark)

Scratchpad scripts, in order: `wm_diag.py` (locate) -> `wm_solve2.py` (matte) ->
`wm_hybrid.py` (rim mask). Then re-verify with `wm_control.py`.

Traps that cost real time:

1. **Validate with `abs()`, not `.max()`** — an over-corrected (too dark) result
   reads as a small positive number and looks like success.
2. **The Jacobian is `(o - W)/(1-alpha)^2`**, not `(W - M)`. At alpha 0.5 that is
   ~2.4x larger; the wrong one oscillates instead of converging.
3. **Always run a control patch.** Optimising against a blurry inpaint
   prediction (v3) forced the region smooth and carved the star shape back in —
   it scored better while looking worse.
4. **A constant-across-frames feature is not necessarily watermark.** Tell them
   apart by background-dependence: an alpha-blended layer's excess shrinks as the
   background brightens; real scene content stays flat.
