"""Generate tc_narration.txt, shots.json (16 image prompts), and a ledger check
from a single CASE string, via one LLM call. Mirrors MindUnlocked's
_gen_video_content.py pattern, adapted for shadow_gasp true-crime shorts:
noir comic-book stills (FLUX) instead of stock footage.

Only runs when tc_narration.txt isn't already committed (mirrors the
TTS-skip pattern), so a hand-authored video in this repo is never overwritten.
"""
import json
import os
import re
import datetime

from anthropic import Anthropic

MODEL = "claude-sonnet-5"
# The ledger is channel-wide state, so it must not follow this script when it
# gets copied into a per-video project dir. CASES_LEDGER points at the one true
# copy (_pipeline/cases_used.json); the sibling default only applies when this
# script is run from _pipeline/ itself.
LEDGER_PATH = os.environ.get(
    "CASES_LEDGER", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases_used.json")
)

SYS = """You are the writer for shadow_gasp, a true-crime YouTube Shorts channel.
Every short is ~65-90s, narrated over 16 noir comic-book illustrated stills
(one of which becomes an animated "hook" clip).

RETENTION RULES (this is the entire point of the format, follow them exactly):
1. Cold open. No throat-clearing, no "let me tell you about". Start mid-scene
   or on a striking fact, in the first sentence.
2. Escalating concrete specifics (exact dates, dollar amounts, distances,
   time gaps) — concrete numbers read as insider knowledge and keep people
   watching for the next detail.
3. A mid-video reversal or twist that breaks the pattern the viewer expects
   (e.g. the case closes NOT because it was solved; the "solution" turns out
   to be stranger than the mystery; an authority fails where expected to
   succeed).
4. End on an UNRESOLVED note — a genuine open loop, not tidy closure. The
   best true-crime shorts make viewers comment/rewatch instead of feeling
   done. Prefer real unsolved cases, or a genuinely eerie/ambiguous detail in
   solved ones.
5. Second person is NOT required (unlike MindUnlocked) — write in tight,
   declarative third person, present-tense energy even in past tense.
6. Length: 150-220 words. Short sentences. Fragments are fine for punch.

Also produce 16 image prompts for noir comic-book stills that visually track
the narration beat-by-beat, in order. Shot 1 is always a WIDE, FACELESS,
ARCHITECTURAL/ENVIRONMENTAL establishing shot (a building, vehicle, or
landscape — no characters) because it becomes an animated video clip and
motion-generation only works reliably on wide shots with no faces to warp.
Shots 2-16 can include characters/close-ups since they stay as stills.

CHARACTER FACES — this is what makes or breaks retention, get it right:
Faces should be FULL, FRONTAL, and EXPRESSIVE by default — direct gaze,
open mouth mid-reaction/mid-speech, wide eyes, visible emotion (fear, shock,
tension, grim resolve). This is illustrated art, not a photograph, so an
expressive drawn face is never "a real photographic likeness" — it's already
a generic, non-identifiable rendering even when fully visible. Describe the
specific expression and where the eyes/gaze point in every prompt that has a
character in it, the way a storyboard artist would (e.g. "his eyes wide with
disbelief, mouth open mid-shout, staring directly at the device in his
hands"). Do NOT default to hiding faces (back-turned, deep shadow, obscuring
angle, hat brim over eyes) — that reads as visually dead and is a bug in this
pipeline's older prompts, not a style choice. The one narrow exception: if the
case involves a real person whose actual face is so recently and widely
publicized that an illustrated version would still read as a likeness of that
specific individual (e.g. an ongoing case currently in the news), obscure
that one figure only — every other character in the same shot still gets a
full expressive face.

ETHICS for every image prompt:
- No readable text/signage/logos baked into the image (captions are a
  separate overlay).
- No graphic gore — implied dread only.

IMPORTANT — avoid the words "comic-book" and "graphic novel" in prompts.
The image model strongly associates those words with panel layouts that
include speech bubbles, and since it cannot render real text it fills them
with garbled gibberish lettering. Use "cel-shaded illustration" / "noir
illustrated aesthetic" instead — same visual style, without the bubble
association. Every prompt must also explicitly state no dialogue, no speech
bubbles, no captions, in addition to no text or logos.

Each image prompt must end with this exact style anchor phrase, adapted only
in the bracketed parts to match the case's era/setting:
"dark, moody cel-shaded digital illustration, thick black ink outlines,
dramatic cinematic lighting, muted desaturated palette ([2-3 era-appropriate
colors]), noir illustrated aesthetic, single-scene composition, no speech
bubbles, no dialogue, no captions, no text or logos, [era/setting]"

Return JSON:
{
  "title_working": "internal working title, not for publishing",
  "narration": "the full ~150-220 word script, no section labels, no stage directions",
  "shots": [
    {"n": 1, "prompt": "..."},
    ... exactly 16 entries ...
  ],
  "hook_motion_prompt": "one sentence describing ONLY the camera motion for shot 1 (e.g. slow push-in, drifting clouds) — no new content, must match shot 1's prompt exactly in subject",
  "caption_yt": "one-line YouTube Shorts caption with hashtags, under 150 chars",
  "caption_ig": "one-line Instagram caption with hashtags, under 150 chars"
}
Respond with ONLY the JSON object — no markdown code fences, no other text."""

MIN_WORDS, MAX_WORDS = 120, 260


def extract_json(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return json.loads(t)


def load_ledger():
    """cases_used.json is the channel's real published-video ledger, scraped
    from YouTube: {"cases": [{"videoId", "case", "publishedAt"}, ...]}.
    Preserve that schema — do not invent a new one, it is the dedup source."""
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, encoding="utf-8") as f:
            d = json.load(f)
        d.setdefault("cases", [])
        return d
    return {"cases": []}


def save_ledger(ledger):
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=1, ensure_ascii=False)


def generate(client, case):
    messages = [{"role": "user", "content": f"Case: {case}"}]
    for attempt in range(3):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYS,
            messages=messages,
        )
        raw = next(b.text for b in resp.content if b.type == "text")
        d = extract_json(raw)
        words = len(d["narration"].split())
        n_shots = len(d.get("shots", []))
        if MIN_WORDS <= words <= MAX_WORDS and n_shots == 16:
            return d
        print(f"attempt {attempt + 1}: {words} words, {n_shots} shots, retrying")
        messages.append({"role": "assistant", "content": raw})
        problems = []
        if not (MIN_WORDS <= words <= MAX_WORDS):
            problems.append(f"narration was {words} words, must be {MIN_WORDS}-{MAX_WORDS}")
        if n_shots != 16:
            problems.append(f"you produced {n_shots} shots, must be exactly 16")
        messages.append({"role": "user", "content": "Fix: " + "; ".join(problems) + ". Return the full corrected JSON."})
    return d


def main():
    if os.path.exists("tc_narration.txt"):
        print("tc_narration.txt already present, skipping content generation")
        return

    case = os.environ["CASE"]
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    d = generate(client, case)

    open("tc_narration.txt", "w", encoding="utf-8").write(d["narration"].strip() + "\n")
    json.dump(d["shots"], open("shots.json", "w"), indent=1)
    json.dump(
        {
            "hook_motion_prompt": d["hook_motion_prompt"],
            "caption_yt": d["caption_yt"],
            "caption_ig": d["caption_ig"],
            "title_working": d["title_working"],
        },
        open("meta.json", "w"), indent=2,
    )

    # Reserve the case in the ledger now (videoId filled in after upload, by
    # _youtube_upload.py) so a second pipeline run can't pick the same case
    # while this one is still rendering.
    ledger = load_ledger()
    ledger["cases"].append({
        "videoId": None,
        "case": case,
        "publishedAt": datetime.date.today().isoformat(),
    })
    save_ledger(ledger)

    words = len(d["narration"].split())
    print(f"Generated: {words} words, 16 shots, title: {d['title_working']}")


if __name__ == "__main__":
    main()
