"""shadow_gasp retention digest (report-only, manual dispatch).

Pulls last-21-day per-video retention (views, average-view-percentage,
first -30% drop-off point) via YouTube Analytics, plus net subscriber
change, and sends a Telegram report. Ranks BOTH by retention (best-held
attention) and by reach (most views) separately -- ranking only by retention
was a real bug on MindUnlocked's version of this (a widely-reached video can
have mediocre retention and still be the one worth learning from). Never
writes to any pick/build loop -- purely informational, same as /trending.
"""
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import datetime as dt

ANALYTICS = "https://youtubeanalytics.googleapis.com/v2/reports"
MIN_VIEWS = 20  # below this the retention % is too noisy to act on


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


def get(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"{e.code} {e.read().decode(errors='replace')[:300]}"


def analytics(token, **params):
    qs = urllib.parse.urlencode(params)
    return get(f"{ANALYTICS}?{qs}", token)


def parse_iso8601_duration(s):
    # e.g. "PT47S" or "PT1M12S" -> seconds
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m:
        return 0
    h, mnt, sec = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mnt * 60 + sec


def video_durations(token, video_ids):
    durations = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        data, err = get(
            f"https://www.googleapis.com/youtube/v3/videos?part=contentDetails&id={','.join(chunk)}",
            token,
        )
        if err:
            continue
        for item in data.get("items", []):
            durations[item["id"]] = parse_iso8601_duration(item["contentDetails"].get("duration", ""))
    return durations


def dropoff_point(token, video_id, duration_s):
    data, err = analytics(
        token, ids="channel==MINE", startDate="2020-01-01", endDate=dt.date.today().isoformat(),
        metrics="audienceWatchRatio", dimensions="elapsedVideoTimeRatio", filters=f"video=={video_id}",
    )
    if err or not data.get("rows"):
        return None
    for ratio, watch in data["rows"]:
        if watch <= 0.7:
            secs = round(ratio * duration_s)
            return f"{secs // 60}:{secs % 60:02d}"
    return "held past 70% throughout"


def main():
    token = access_token()
    end = dt.date.today()
    start = end - dt.timedelta(days=21)
    s, e = start.isoformat(), end.isoformat()

    lines = [f"📊 shadow_gasp retention digest (last 21 days)"]

    print("=== SUBSCRIBERS (21d) ===")
    data, err = analytics(token, ids="channel==MINE", startDate=s, endDate=e,
                           metrics="subscribersGained,subscribersLost")
    if err:
        print("ERROR:", err)
        lines.append("👥 Subscribers: couldn't fetch (Analytics error)")
    else:
        gained, lost = (data["rows"][0] if data.get("rows") else (0, 0))
        net = gained - lost
        lines.append(f"👥 Subscribers: net {'+' if net >= 0 else ''}{net} (gained {gained}, lost {lost})")

    print("=== TOP VIDEOS BY VIEWS (21d) ===")
    data, err = analytics(token, ids="channel==MINE", startDate=s, endDate=e,
                           metrics="views,subscribersGained,averageViewPercentage",
                           dimensions="video", sort="-views", maxResults=15)
    if err:
        print("ERROR:", err)
        lines.append(f"\n❌ Couldn't fetch video list: {err}")
        send(lines, token_ok=False)
        return

    cols = [h["name"] for h in data.get("columnHeaders", [])]
    rows = [dict(zip(cols, row)) for row in data.get("rows", [])]
    rows = [r for r in rows if r["views"] >= MIN_VIEWS]
    if not rows:
        lines.append("\nNo videos with enough views (21d) to report on yet.")
        send(lines, token_ok=True)
        return

    durations = video_durations(token, [r["video"] for r in rows])

    for r in rows:
        dur = durations.get(r["video"], 0)
        r["dropoff"] = dropoff_point(token, r["video"], dur) if dur else "?"

    by_views = sorted(rows, key=lambda r: -r["views"])
    # Shorts can be re-watched in a loop, so averageViewPercentage can exceed
    # 100% on a barely-viewed video (one viewer looping it 4x looks like
    # "413% watched") -- that is a low-sample artifact, not genuine
    # attention held at scale. Pick best-retained only from videos with
    # real reach (the same top-8 actually shown below), same fix as
    # MindUnlocked's reach-vs-retention ranking bug.
    shown = by_views[:8]
    by_retention = sorted(shown, key=lambda r: -r["averageViewPercentage"])
    best_retained = by_retention[0]
    widest_reach = by_views[0]

    lines.append(f"\n🎬 {len(rows)} videos, avg watched {sum(r['averageViewPercentage'] for r in rows) / len(rows):.0f}%")
    for r in by_views[:8]:
        lines.append(f"• {r['views']}v · {r['averageViewPercentage']:.0f}% watched · drop by {r['dropoff']} · +{r['subscribersGained']} subs — https://youtu.be/{r['video']}")

    if best_retained["video"] == widest_reach["video"]:
        lines.append(f"\n🏆 Best-retained AND widest-reach is the same video: {best_retained['averageViewPercentage']:.0f}% watched, {best_retained['views']} views — https://youtu.be/{best_retained['video']}")
    else:
        lines.append(f"\n🎯 Best-retained: {best_retained['averageViewPercentage']:.0f}% watched — https://youtu.be/{best_retained['video']}")
        lines.append(f"↗ Widest reach: {widest_reach['views']} views ({widest_reach['averageViewPercentage']:.0f}% watched) — https://youtu.be/{widest_reach['video']}")

    send(lines, token_ok=True)


def send(lines, token_ok):
    secret = os.environ.get("BATCH_NOTIFY_SECRET", "")
    report = "\n".join(lines)
    print("\n--- REPORT ---\n" + report)
    body = json.dumps({"report": report, "chat_id": os.environ.get("CHAT_ID_INPUT", "").strip() or None}).encode()
    req = urllib.request.Request(
        "https://shadow-gasp-bot.everydayhypehq.workers.dev/retention/report",
        data=body, method="POST",
        headers={
            "X-Batch-Notify-Secret": secret,
            "Content-Type": "application/json",
            # Cloudflare 1010 blocks urllib's default UA on this Worker -- see
            # shadow-gasp-notify-cloudflare-1010-fix memory. Required on every call.
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        },
    )
    try:
        urllib.request.urlopen(req).read()
    except Exception as ex:
        print(f"notify failed (non-fatal, report is still in this log): {ex!r}")


if __name__ == "__main__":
    main()
