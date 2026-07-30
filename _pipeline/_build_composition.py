"""Auto-build index.html from tc_narration.wav's transcript + the 16 stills
(+ optional 01.mp4 hook clip). Generalizes the shot-timing/caption/Ken-Burns
work done by hand for the D.B. Cooper video into a repeatable algorithm:

1. Split the narration into 16 roughly-even segments, but snap each boundary
   to the nearest real pause (the biggest word-gap within a search window)
   so cuts land on natural breath points instead of mid-sentence.
2. One caption per transcribed word, shown across its own (start, end).
3. Ken Burns pans cycle through 4 fixed patterns; crossfades match Holmes'
   house style (blur+opacity, 0.4s).
4. If images/seq/01.mp4 exists, shot 1 is a video layer (the animated hook);
   otherwise shot 1 is a still like every other shot (Ken Burns only, no
   separate video layer) — the composition degrades gracefully if the
   CogVideoX hook step failed or was skipped.
5. Optional continuous camera shake — handheld-style random 2D jitter,
   applied to the artwork layers only (captions stay legible/stationary).
   Skips shot 1's duration when a hook video is present, since that clip
   already has its own real motion. Tune via SHAKE_PX — 0 disables it.

Requires: transcript.json, images/seq/01..16.jpeg (+ optional 01.mp4),
music/*.wav (any file; first one found is used, trimmed+faded to length).
"""
import glob
import json
import math
import os
import re
import subprocess

WORDS_PER_SEARCH_WINDOW = 1.8  # seconds either side of an ideal boundary to look for a natural pause
TAIL_PAD = 1.0  # seconds of extra composition time after the last word, for the fadeout

# Continuous camera shake across the video, excluding the hook clip. 0 = off.
#
# Values reverse-engineered from nuclear-false-alarm-short by phase-correlating
# its frames and FFT-ing the detrended motion track. That reference measures:
#   x: 6.6px peak-to-peak, dominant 3.80 Hz (harmonic at 7.18 = 2x)
#   y: 6.0px peak-to-peak, dominant 5.07 Hz (harmonic at 10.14 = 2x)
# Clean fundamental+harmonic pairs mean it is PERIODIC, not random jitter, and
# the two axes run at DIFFERENT rates — that mismatch is what makes it read as
# organic shake instead of a visibly repeating sway. Keyframes are emitted per
# render frame so the ~5 Hz component is actually resolved.
SHAKE_PX = 3  # amplitude in px (+/-SHAKE_PX)
SHAKE_FREQ_X = 3.80  # Hz, horizontal oscillation
SHAKE_FREQ_Y = 5.07  # Hz, vertical oscillation (deliberately != X)
SHAKE_FPS = 30  # keyframes per second; must match the render fps

# Background music level, measured not guessed. The old channel-wide 0.13 put
# the bed at -37.9 LUFS against narration at -21.9 — 16 dB down, which is only
# audible on headphones and vanishes entirely on a phone speaker (which is how
# almost every Short is actually watched). 0.26 lands the bed ~10 dB under the
# VO: clearly present as atmosphere, still well clear of masking the words.
# Re-derive with: ffmpeg -i <track> -af loudnorm=print_format=json -f null -
MUSIC_VOLUME = 0.26


def load_transcript():
    return json.load(open("transcript.json"))


def merge_number_tokens(words):
    """Merge whisper's split '$200' + ',000' style tokens into one caption."""
    merged = []
    i = 0
    while i < len(words):
        w = words[i]
        if re.match(r"^\$\d", w["text"]) and i + 1 < len(words) and re.match(r"^,\d", words[i + 1]["text"]):
            merged.append({"text": w["text"] + words[i + 1]["text"], "start": w["start"], "end": words[i + 1]["end"]})
            i += 2
        else:
            merged.append(w)
            i += 1
    return merged


def find_shot_boundaries(words, n_shots=16):
    total = words[-1]["end"]
    gaps = []  # (gap_start_time, gap_size) at each inter-word silence
    for i in range(len(words) - 1):
        gap_size = words[i + 1]["start"] - words[i]["end"]
        if gap_size > 0.05:
            gaps.append((words[i]["end"], gap_size))

    ideal = [total * i / n_shots for i in range(1, n_shots)]
    boundaries = []
    last_b = 0.0
    for target in ideal:
        window = [g for g in gaps if abs(g[0] - target) <= WORDS_PER_SEARCH_WINDOW and g[0] > last_b]
        if window:
            best = max(window, key=lambda g: g[1])
            b = round(best[0], 3)
        else:
            b = round(target, 3)
        if b <= last_b:
            b = round(last_b + 0.3, 3)
        boundaries.append(b)
        last_b = b
    return [0.0] + boundaries + [round(total, 3)]


PATTERNS = [
    ({"scale": 1.16, "xPercent": -4, "yPercent": 2}, {"scale": 1.02, "xPercent": 0, "yPercent": 0}),
    ({"scale": 1.02, "xPercent": 3, "yPercent": -2}, {"scale": 1.16, "xPercent": 0, "yPercent": 1}),
    ({"scale": 1.0, "xPercent": 0, "yPercent": 0}, {"scale": 1.12, "xPercent": -2, "yPercent": 0}),
    ({"scale": 1.0, "xPercent": 0, "yPercent": 0}, {"scale": 1.14, "xPercent": 0, "yPercent": 0}),
]


def dstr(d):
    return "{scale: %s, xPercent: %s, yPercent: %s}" % (d["scale"], d["xPercent"], d["yPercent"])


def build_shake_lines(total_dur, magnitude, freq_x, freq_y, fps, skip_until=0.0):
    """Camera shake on #scene (artwork layers only, not captions, and not the
    hook clip — starts at skip_until).

    Two independent sine oscillations, x and y at different frequencies. The
    beat between the two rates is what makes it read as organic handheld shake
    rather than a metronomic sway. Matched to the measured reference spectrum;
    see SHAKE_* constants for the numbers."""
    if magnitude <= 0:
        return []
    lines = []
    step = 1.0 / fps
    t = skip_until
    while t < total_dur:
        u = t - skip_until
        dx = round(magnitude * math.sin(2 * math.pi * freq_x * u), 2)
        dy = round(magnitude * math.sin(2 * math.pi * freq_y * u), 2)
        lines.append(f'      tl.set("#scene", {{ x: {dx}, y: {dy} }}, {t:.3f});')
        t += step
    lines.append(f'      tl.set("#scene", {{ x: 0, y: 0 }}, {total_dur:.3f});')
    return lines


def prep_music(total_dur):
    os.makedirs("music", exist_ok=True)
    trimmed = "music/_bed.wav"
    existing = [f for f in glob.glob("music/*.wav") + glob.glob("music/*.mp3") if os.path.abspath(f) != os.path.abspath(trimmed)]
    if not existing:
        print("no music/*.wav|mp3 found — composition will have no background bed")
        return None
    src = existing[0]
    # -stream_loop -1 before -i, so a bed shorter than the video repeats instead
    # of leaving a silent tail. The channel bed is 68.4s and shorts routinely
    # run past that; Holmes needed a hand-looped file for exactly this reason.
    subprocess.run([
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", src, "-t", f"{total_dur:.3f}",
        "-af", f"afade=t=in:st=0:d=1.5,afade=t=out:st={max(0, total_dur - 2.5):.3f}:d=2.5",
        "-ar", "44100", trimmed, "-loglevel", "error",
    ], check=True)
    return trimmed


def main():
    words = merge_number_tokens(load_transcript())
    vo_dur = words[-1]["end"]
    total_dur = vo_dur + TAIL_PAD

    boundaries = find_shot_boundaries(words, n_shots=16)
    shots = list(zip(boundaries[:-1], boundaries[1:]))  # 16 (start, end) tuples

    has_hook_video = os.path.exists("images/seq/01.mp4")
    music_path = prep_music(total_dur)

    layers = []
    if has_hook_video:
        layers.append(
            f'        <div class="layer first" id="L0" style="z-index:1"><video class="kb clip" id="m0" '
            f'data-start="0.000" data-duration="{shots[0][1] - shots[0][0]:.3f}" data-media-start="0" '
            f'data-track-index="20" src="images/seq/01.mp4" muted playsinline></video></div>'
        )
        start_i = 1
    else:
        layers.append(
            '        <div class="layer first" id="L0" style="z-index:1"><div class="kb" '
            'data-layout-allow-overflow="true" id="k0"><img src="images/seq/01.jpeg" /></div></div>'
        )
        start_i = 1  # k1..k15 still map to L1..L15 below; k0's pan is added separately

    for i in range(1, 16):
        n = i + 1
        layers.append(
            f'        <div class="layer" id="L{i}" style="z-index:{i + 1}"><div class="kb" '
            f'data-layout-allow-overflow="true" id="k{i}"><img src="images/seq/{n:02d}.jpeg" /></div></div>'
        )

    kb_lines = []
    fade_lines = []
    if not has_hook_video:
        frm, to = PATTERNS[0]
        dur0 = shots[0][1] - shots[0][0]
        kb_lines.append(f'      tl.fromTo("#k0",{dstr(frm)},{{...{dstr(to)},duration:{dur0:.3f},ease:"none"}},0.000);')
    for i in range(1, 16):
        s, e = shots[i]
        dur = e - s
        frm, to = PATTERNS[i % 4]
        kb_lines.append(f'      tl.fromTo("#k{i}",{dstr(frm)},{{...{dstr(to)},duration:{dur:.3f},ease:"none"}},{s:.3f});')
        fade_lines.append(
            f'      tl.fromTo("#L{i}",{{opacity:0,filter:"blur(6px)"}},'
            f'{{opacity:1,filter:"blur(0px)",duration:0.4,ease:"power2.out"}},{s:.3f});'
        )

    cap_lines = []
    for i, w in enumerate(words):
        text = w["text"].replace('"', '\\"')
        cap_lines.append(f'      mk({i},"{text}",{w["start"]:.3f},{w["end"]:.3f});')

    fadeout_start = vo_dur + 0.1
    wm_fade_time = shots[0][1] if has_hook_video else 0.0

    music_audio = (
        f'      <audio id="mus1" class="clip" data-start="0" data-duration="{total_dur:.3f}" '
        f'data-track-index="91" data-volume="{MUSIC_VOLUME}" src="{music_path}"></audio>'
        if music_path else ""
    )
    wm_block = (
        f'      tl.to("#wmPatch", {{ opacity: 0, duration: 0.4, ease: "power2.out" }}, {wm_fade_time:.3f});'
        if has_hook_video else ""
    )
    wm_div = '      <div id="wmPatch"></div>' if has_hook_video else ""

    shake_skip_until = shots[0][1] if has_hook_video else 0.0
    shake_lines = build_shake_lines(total_dur, SHAKE_PX, SHAKE_FREQ_X, SHAKE_FREQ_Y, SHAKE_FPS, shake_skip_until)

    template = open(os.path.join(os.path.dirname(__file__), "_composition_template.html"), encoding="utf-8").read()
    html = template.format(
        TOTAL_DUR=total_dur,
        VO_DUR=vo_dur,
        LAYERS="\n".join(layers),
        WM_DIV=wm_div,
        WM_BLOCK=wm_block,
        KB="\n".join(kb_lines),
        FADES="\n".join(fade_lines),
        CAPS="\n".join(cap_lines),
        FADEOUT_START=fadeout_start,
        MUSIC_AUDIO=music_audio,
        SHAKE="\n".join(shake_lines),
    )
    open("index.html", "w", encoding="utf-8").write(html)
    print(f"index.html built: {total_dur:.1f}s, 16 shots, hook_video={has_hook_video}, music={bool(music_path)}, shake={SHAKE_PX}px")


if __name__ == "__main__":
    main()
