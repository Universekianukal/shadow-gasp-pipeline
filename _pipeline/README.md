# shadow_gasp automated short pipeline

Mirrors the MindUnlocked pipeline's skip-if-present, one-script-per-step
structure — but swaps stock footage for AI-generated noir stills
(FLUX.1-schnell) and one animated hook clip (CogVideoX-5b I2V), both run
free on Kaggle GPU instead of Google Flow/Gemini (manual) or Pexels (N/A for
this art style).

Every short is **15 stills + 1 motion hook clip** (shot 1). That structure is
fixed — the composition builder, the shot-prompt spec, and the shake logic all
assume it.

## Daily operation

Everything runs on GitHub Actions, the way MindUnlocked does — no local render.
`.github/workflows/pipeline.yml` at the repo root:

| Job | What it does |
|---|---|
| `build` | pick case → script + 16 prompts → TTS → transcribe → FLUX stills → CogVideoX hook → composition → YouTube metadata → commit the reserved case |
| `render` | `hyperframes render` (3 attempts) → verify duration/audio/live-frames |
| `upload` | YouTube upload → stamp `videoId` into the ledger → commit |

Triggers: `workflow_dispatch` (optional `case` input; leave blank to auto-pick)
or the daily 13:00 UTC / 18:30 IST cron. On cron it uploads; on manual dispatch
upload is opt-in via the `upload` checkbox.

Required repo secrets: `ANTHROPIC_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`,
`HF_TOKEN`, `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN`.

`run_pipeline.py` runs the same sequence locally for one-offs and debugging.

## Steps (each skips itself if its output already exists)

0. `_pick_case.py` — Claude picks a case that isn't in `cases_used.json`. The
   full 74-case exclusion list goes into the prompt, and the answer is
   re-checked locally by fuzzy match (token-overlap + sequence ratio), because
   the model will otherwise reword an existing case rather than genuinely
   pick a new one. Skipped when `CASE` is already set.
1. `_gen_video_content.py` — one Claude call: writes `tc_narration.txt`
   (retention-optimized script), `shots.json` (16 image prompts), and
   `meta.json` (hook motion prompt + captions). Requires `CASE` and
   `ANTHROPIC_API_KEY`. Reserves the case in `cases_used.json` with
   `videoId: null` so a concurrent run can't claim it.
2. `_tts.py` — Kokoro TTS, voice `bm_george` (the established shadow_gasp
   true-crime voice), speed 0.92. Produces `tc_narration.wav`.
3. `_transcribe.py` — faster-whisper word-level timestamps. Produces
   `transcript.json`.
4. `_gen_flux_images.py` — pushes one Kaggle kernel generating all 16 stills
   from `shots.json` via FLUX.1-schnell (NF4 quantized, fits free-tier GPU).
   Polls and downloads into `images/seq/01.jpeg`..`16.jpeg`. **Each image is
   OCR-checked (EasyOCR) and regenerated up to 3× if any text is detected** —
   FLUX runs at `guidance_scale=0`, which means negative prompts do nothing, so
   garbled speech-bubble lettering can only be caught after the fact. Anything
   still failing after 3 attempts is listed in `FLAGGED.txt` and surfaced to
   stderr for manual review.
5. `_gen_cog_hook.py` — uploads `images/seq/01.jpeg` as a Kaggle dataset,
   pushes a CogVideoX-5b Image-to-Video kernel using `meta.json`'s
   `hook_motion_prompt`, polls (slow — expect 15-40+ min), downloads
   `images/seq/01.mp4`. **If this step fails the composition builder degrades
   gracefully** — shot 1 becomes a Ken-Burns still like the other 15 instead of
   blocking the pipeline. The workflow marks it `continue-on-error` for exactly
   that reason.
6. `_build_composition.py` — auto shot-timing: splits the narration into 16
   roughly-even segments, snaps each cut to the nearest real pause in the
   transcript (so cuts land on breath points, not mid-word), one caption per
   transcribed word, 4 cycling Ken Burns pan patterns, crossfades, and the
   camera shake. Writes `index.html`.
7. `_gen_youtube_meta.py` — one more Claude call: title/description/tags from
   the actual script, written to `youtube.json` (categoryId 22, private).
8. `hyperframes render` → `final.mp4`.
9. `_youtube_upload.py` — uploads and writes the real `videoId` back onto the
   reserved ledger entry.

## Music bed

`assets/bed.wav` is the channel bed — the music extracted from the reference
short, normalized to -20 LUFS (it is `hh-holmes-murder-castle-short/music/
reference_short_bg_music_boosted.wav`). **This is the only correct bed.** Two
other files in the repo look like it and are not:

- `db-cooper-short/music/reference_short_bg_music.wav` — named as if it were the
  reference music, but cross-correlates at 0.05 against it. A different track.
- `isdal-woman-short/music/dark.mp3` — a generic stock bed left over from the
  June long-form videos.

To check a bed's provenance, cross-correlate against `assets/bed.wav` at 8 kHz
mono; a genuine match scores ~0.9, an unrelated track ~0.05.

The bed is 68.4s and shorts routinely run longer, so `prep_music()` loops it
(`-stream_loop -1`) before trimming and fading. Do not hand-make a `_looped`
variant — that was the old manual workaround.

`MUSIC_VOLUME = 0.26` puts the bed ~10 dB under the narration. The previous
channel-wide 0.13 put it 16 dB under, which is inaudible on a phone speaker
(how nearly every Short is actually watched) and only audible on headphones.
Both numbers were measured with
`ffmpeg -i <track> -af loudnorm=print_format=json -f null -`, not estimated.

## Camera shake

`_build_composition.py`'s `SHAKE_*` constants were **measured**, not guessed,
from `../nuclear-false-alarm-short` — phase-correlating consecutive frames and
FFT-ing the detrended motion track. That reference is ±3px at 3.80 Hz
horizontal / 5.07 Hz vertical, with clean 2× harmonics, meaning it is periodic
(two sines at different rates), not random jitter. Keyframes are emitted per
render frame so the ~5 Hz component actually resolves. The hook clip is
excluded — it already has its own real motion.

If shake ever needs re-matching against a new reference, measure it the same
way. Eyeballing it does not converge.

## Known quality tradeoffs vs. the manual (Gemini/Flow) pipeline

- FLUX.1-schnell matches Gemini on character/portrait shots but is noticeably
  softer on wide/establishing shots — it doesn't reproduce the thick black
  ink-outline look as consistently. Verified side-by-side, see
  `../_flux_noir_test/`. Accepted tradeoff for full automation.
- Prompts must avoid the words "comic-book" and "graphic novel" — the model
  associates them with panel layouts containing speech bubbles, which it then
  fills with gibberish. `_gen_video_content.py`'s system prompt enforces
  "cel-shaded illustration" / "noir illustrated aesthetic" instead.

## Not yet verified

- The OCR retry loop has only been exercised on individual shots (8 and 16).
  The first full 16-shot batch through it is the real test.
- `_pick_case.py` has not yet been run against the live API (no local
  `ANTHROPIC_API_KEY`); its dedup guard is unit-tested against the real
  74-case ledger.
- Telegram watchdog / failed-build retry bot, matching the MindUnlocked and
  everydayhype setups, is not built for this channel.

## Not part of this pipeline

`_gumroad_publish.py`, `_telegram_send.py`, `_telegram_listen.py`,
`_deliver_for_approval.py`, `state.py`, `_get_captions.py` and their
`*_secrets.json` / `pending_approvals.json` files belong to a separate
comic-PDF selling workstream that happens to live in this directory. They hold
live state (including an open Telegram approval) — leave them alone.
