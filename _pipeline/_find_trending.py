"""shadow_gasp trending-story finder (report-only, manual dispatch).

Searches YouTube for breakout true-crime/horror/mystery Shorts from OTHER
channels (high views-per-subscriber ratio = a small channel's video that
travelled, the same "breakout" signal MindUnlocked's competitor teardown
uses), filters out stories already covered in cases_used.json, asks Claude
to synthesize 3-5 fresh story ideas from what's left, and sends the report
to Telegram via the Worker's /trending/report endpoint. Never picks or
builds anything -- purely informational.
"""
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
QUERIES = [
    "unsolved mystery case",
    "true crime story shorts",
    "disappearance case explained",
    "creepy true story",
    "cold case solved",
]
MIN_VIEWS = 50_000
MAX_RESULTS_PER_QUERY = 8


def access_token():
    body = urllib.parse.urlencode({
        "client_id": os.environ["YOUTUBE_CLIENT_ID"],
        "client_secret": os.environ["YOUTUBE_CLIENT_SECRET"],
        "refresh_token": os.environ["YOUTUBE_REFRESH_TOKEN"],
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=body)
    with urllib.request.urlopen(req) as r:
        d = json.loads(r.read())
    if "access_token" not in d:
        sys.exit(f"token refresh failed: {d}")
    return d["access_token"]


def yt_get(path, token, **params):
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"https://www.googleapis.com/youtube/v3/{path}?{qs}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"{e.code} {e.read().decode(errors='replace')[:300]}"


def own_channel_id(token):
    data, err = yt_get("channels", token, part="id", mine="true")
    if err:
        print("WARNING: couldn't resolve own channel id:", err)
        return None
    return data["items"][0]["id"]


def search_breakouts(token, my_channel_id):
    published_after = (dt.datetime.utcnow() - dt.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    seen_video_ids = set()
    candidates = []
    for q in QUERIES:
        data, err = yt_get(
            "search", token, part="id,snippet", q=q, type="video",
            videoDuration="short", order="viewCount",
            publishedAfter=published_after, maxResults=MAX_RESULTS_PER_QUERY,
        )
        if err:
            print(f"search failed for {q!r}: {err}")
            continue
        for item in data.get("items", []):
            vid = item["id"]["videoId"]
            if vid in seen_video_ids:
                continue
            seen_video_ids.add(vid)
            candidates.append({"videoId": vid, "channelId": item["snippet"]["channelId"],
                                "title": item["snippet"]["title"]})

    if not candidates:
        return []

    # Batch views + channel stats (max 50 ids per call, well under our count).
    vids_data, err = yt_get("videos", token, part="statistics",
                             id=",".join(c["videoId"] for c in candidates))
    views_by_id = {}
    if not err:
        for item in vids_data.get("items", []):
            views_by_id[item["id"]] = int(item["statistics"].get("viewCount", 0))

    channel_ids = list({c["channelId"] for c in candidates})
    subs_by_channel = {}
    titles_by_channel = {}
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i:i + 50]
        ch_data, err = yt_get("channels", token, part="statistics,snippet", id=",".join(chunk))
        if err:
            continue
        for item in ch_data.get("items", []):
            subs_by_channel[item["id"]] = int(item["statistics"].get("subscriberCount", 0)) or 1
            titles_by_channel[item["id"]] = item["snippet"]["title"]

    breakouts = []
    for c in candidates:
        if c["channelId"] == my_channel_id:
            continue
        views = views_by_id.get(c["videoId"], 0)
        subs = subs_by_channel.get(c["channelId"], 1)
        if views < MIN_VIEWS:
            continue
        breakouts.append({
            "title": c["title"],
            "channel": titles_by_channel.get(c["channelId"], "?"),
            "views": views,
            "subs": subs,
            "ratio": round(views / subs, 1),
        })
    breakouts.sort(key=lambda b: -b["ratio"])
    return breakouts[:15]


def already_covered_cases():
    path = os.path.join(HERE, "cases_used.json")
    try:
        data = json.load(open(path, encoding="utf-8"))
        return [c["case"] for c in data.get("cases", [])]
    except Exception as e:
        print(f"couldn't read cases_used.json: {e!r}")
        return []


def synthesize(breakouts, covered):
    if not breakouts:
        return "No breakout videos found in the last 30 days across the search queries tried."

    prompt = f"""You help pick true-crime/horror/mystery story ideas for a YouTube Shorts channel
called shadow_gasp. Below are breakout videos from OTHER channels (high views relative to their
subscriber count -- a signal the story/angle traveled), and a list of stories shadow_gasp has
ALREADY covered.

BREAKOUT VIDEOS (title, channel, views, views-per-subscriber ratio):
{json.dumps(breakouts, indent=2)}

ALREADY COVERED (do not suggest these or close duplicates):
{json.dumps(covered, indent=2)}

Task: suggest 3-5 SPECIFIC real story/case ideas (name the actual case/incident, not a vague
genre) that are clearly trending right now based on the breakout list above, and are NOT already
covered. For each, note which breakout video(s) suggested it and why it's trending. If the
breakout titles are too vague to identify a specific real case, say so honestly instead of
guessing. Keep it under 300 words, plain text, no markdown headers."""

    body = json.dumps({
        "model": "claude-sonnet-5",
        # claude-sonnet-5 spends output budget on reasoning blocks BEFORE
        # emitting text -- a small cap can return zero text (stop_reason
        # max_tokens, only a thinking block). max_tokens is a CAP not a
        # charge, so headroom here is free. See mindunlocked-growth-agents
        # memory's max_tokens-starvation entry -- same bug, same fix.
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as r:
        resp = json.loads(r.read())
    text_blocks = [b["text"] for b in resp.get("content", []) if b.get("type") == "text"]
    if not text_blocks:
        raise RuntimeError(f"no text block in Claude response: {resp}")
    return text_blocks[0]


def notify(report_text, chat_id):
    secret = os.environ.get("BATCH_NOTIFY_SECRET", "")
    body = json.dumps({"report": report_text, "chat_id": chat_id or None}).encode()
    req = urllib.request.Request(
        "https://shadow-gasp-bot.everydayhypehq.workers.dev/trending/report",
        data=body, method="POST",
        headers={
            "X-Batch-Notify-Secret": secret,
            "Content-Type": "application/json",
            # Cloudflare bot-protection (error 1010) blocks urllib's default
            # UA fingerprint on this Worker -- see shadow-gasp-notify-cloudflare-1010-fix memory.
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        },
    )
    try:
        urllib.request.urlopen(req).read()
    except Exception as e:
        print(f"notify failed (non-fatal, report is still in this log): {e!r}")


def main():
    token = access_token()
    my_id = own_channel_id(token)
    breakouts = search_breakouts(token, my_id)
    print(f"found {len(breakouts)} breakout candidates")
    for b in breakouts:
        print(" ", b)
    covered = already_covered_cases()
    report = synthesize(breakouts, covered)
    print("\n--- REPORT ---\n" + report)
    notify(report, os.environ.get("CHAT_ID_INPUT", "").strip())


if __name__ == "__main__":
    main()
