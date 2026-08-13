"""Generate youtube.json (title/description/tags) from this video's actual
script (tc_narration.txt), mirroring MindUnlocked's _gen_youtube_meta.py.
"""
import json
import os
import re

from anthropic import Anthropic

MODEL = "claude-sonnet-5"

SYS = """You write YouTube Shorts metadata for shadow_gasp, a true-crime channel.
Given a video's narration script, produce:
- title: under 100 chars, curiosity-driven but accurate, no clickbait lies,
  should create an open loop (a question the viewer wants answered)
- description: 2-3 short paragraphs summarizing the actual case, end with
  "Subscribe now \U0001F447"
- tags: 5-10 relevant lowercase keyword tags, no hashtags

Return JSON: {"title": "", "description": "", "tags": ["", ...]}
Respond with ONLY the JSON object — no markdown code fences, no other text."""


def extract_json(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return json.loads(t)


def generate(client, script):
    messages = [{"role": "user", "content": script}]
    for attempt in range(3):
        resp = client.messages.create(model=MODEL, max_tokens=2048, system=SYS, messages=messages)
        # Same crash already fixed in _pick_case.py and _gen_video_content.py:
        # a response can come back with no text block at all.
        raw = next((b.text for b in resp.content if b.type == "text"), None)
        if raw is None:
            print(f"attempt {attempt + 1}: response had no text block (stop_reason={resp.stop_reason}), retrying")
            continue
        try:
            return extract_json(raw)
        except json.JSONDecodeError as e:
            print(f"attempt {attempt + 1}: invalid JSON ({e}), retrying")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"That wasn't valid JSON ({e}). Return the full corrected JSON object only, no other text."})
    raise SystemExit("could not get valid JSON after 3 attempts")


def main():
    script = open("tc_narration.txt").read()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    meta = generate(client, script)

    # A title picked via the Telegram style-picker (see worker.js's
    # commitTitleOverride) overrides only title+tags -- description still
    # comes from the normal generation above, since the style picker never
    # touches it.
    override = {}
    if os.path.isfile("TITLE_OVERRIDE.json"):
        override = json.load(open("TITLE_OVERRIDE.json"))
        print(f"Applying title override: {override.get('title')}")

    out = {
        "title": override.get("title") or meta["title"],
        "description": meta["description"],
        "tags": override.get("tags") or meta["tags"],
        "categoryId": "22",
        # Public by default -- _youtube_upload.py overrides this to "private"
        # itself when PUBLISH_AT is set, which is the only case a video should
        # ever land private (YouTube requires private+publishAt for a real
        # scheduled release, then flips it public automatically). Hardcoding
        # "private" here with no PUBLISH_AT path ever flipping it back used to
        # strand every "immediate" upload private forever (the pipeline token
        # is upload-only, so it can't even call videos.update to fix it after
        # the fact) -- confirmed on day34/day35/day37.
        "privacyStatus": "public",
    }
    json.dump(out, open("youtube.json", "w"), indent=2)
    print(f"Generated youtube.json: {out['title']}")


if __name__ == "__main__":
    main()
