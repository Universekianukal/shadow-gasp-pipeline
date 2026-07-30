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


def main():
    script = open("tc_narration.txt").read()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYS,
        messages=[{"role": "user", "content": script}],
    )
    meta = extract_json(next(b.text for b in resp.content if b.type == "text"))

    out = {
        "title": meta["title"],
        "description": meta["description"],
        "tags": meta["tags"],
        "categoryId": "22",
        "privacyStatus": "private",
    }
    json.dump(out, open("youtube.json", "w"), indent=2)
    print(f"Generated youtube.json: {out['title']}")


if __name__ == "__main__":
    main()
