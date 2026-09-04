#!/usr/bin/env python3
"""Reconcile the channel against the ledger, and PROPOSE a mapping for review.

⚠️ WHY THIS EXISTS. The ledger records a videoId at upload time and nothing ever revisits it.
When a video is deleted and re-uploaded by hand -- routine here, because a PUBLISH_AT in the past
uploads private and the fix is a manual re-upload -- the new video gets a new id that nothing
writes back. The ledger keeps pointing at the corpse.

Measured 2026-09-04: 3 days pointed at videos YouTube reports as removed, while 10 live videos
had no ledger entry at all, including one published 2026-09-02. The visible symptom was a
backlog list that jumped from 08-07 to 07-30 and stopped dead at 08-15, and days that had clearly
published still showing as unpublished.

⭐ PROPOSES, never writes. Titles are not case names -- "8 Bodies and 1 axe Who Stayed Until
Dawn" is obviously Villisca to a human and a guess to code -- and a wrong link puts a comic's
funnel on the wrong video, which is worse than a blank cell. So this prints a table to confirm,
and `--apply` only writes the pairs given to it explicitly.

    python3 _reconcile_channel.py                       # review table
    python3 _reconcile_channel.py --apply 35=vU_D1DSkvrs,33=H_u0xC8ay_A
"""
import argparse
import json
import os
import re
import sys

STATE_PATH = os.path.join("batch", "state.json")
LEDGER_PATH = "cases_used.json"
STOP = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "was", "were", "is", "it", "its",
        "for", "with", "who", "what", "why", "how", "then", "they", "this", "that", "his", "her",
        "no", "one", "after", "before", "from", "by", "shorts", "truecrime", "unsolved", "facts"}


def toks(s):
    return {w for w in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(w) > 2 and w not in STOP}


def get_youtube():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
        client_id=os.environ["YOUTUBE_CLIENT_ID"],
        client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/youtube"],
    )
    return build("youtube", "v3", credentials=creds)


def channel_videos(yt):
    ch = yt.channels().list(part="contentDetails", mine=True).execute()
    uploads = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    out, page = [], None
    while True:
        r = yt.playlistItems().list(part="contentDetails,snippet", playlistId=uploads,
                                    maxResults=50, pageToken=page).execute()
        for it in r["items"]:
            out.append({"id": it["contentDetails"]["videoId"],
                        "title": it["snippet"]["title"],
                        "published": it["snippet"].get("publishedAt", "")})
        page = r.get("nextPageToken")
        if not page:
            break
    return out


def day_blobs(state):
    """Everything known about a day that a title might echo: case, working title, caption."""
    blobs = {}
    for k, d in state.items():
        text = d["case"]
        meta = os.path.join("batch", f"day{int(k):02d}", "meta.json")
        if os.path.exists(meta):
            try:
                m = json.load(open(meta, encoding="utf-8"))
                text += " " + " ".join(str(m.get(f, "")) for f in
                                       ("title_working", "caption_yt", "caption_ig"))
            except Exception:
                pass
        blobs[int(k)] = toks(text)
    return blobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", default="", help="day=videoId pairs, comma separated")
    args = ap.parse_args()

    state = json.load(open(STATE_PATH, encoding="utf-8"))["days"]
    ledger = json.load(open(LEDGER_PATH, encoding="utf-8"))
    by_case = {c["case"].strip(): c for c in ledger["cases"]}

    if args.apply:
        n = 0
        for pair in args.apply.split(","):
            day, _, vid = pair.strip().partition("=")
            case = state[day.strip()]["case"].strip()
            entry = by_case.get(case)
            if not entry:
                entry = {"videoId": None, "case": case, "publishedAt": None}
                ledger["cases"].append(entry)
            print(f"day {day}: {entry.get('videoId')} -> {vid}   ({case[:48]})")
            entry["videoId"] = vid.strip()
            n += 1
        json.dump(ledger, open(LEDGER_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\nwrote {n} videoId(s) to the ledger")
        return

    yt = get_youtube()
    vids = channel_videos(yt)
    known = {c.get("videoId") for c in ledger["cases"] if c.get("videoId")}
    live = {v["id"] for v in vids}
    blobs = day_blobs(state)
    day_of_case = {d["case"].strip(): int(k) for k, d in state.items()}

    dead = [(day_of_case.get(c["case"].strip()), c["case"], c["videoId"])
            for c in ledger["cases"]
            if c.get("videoId") and c["videoId"] not in live]
    print(f"channel videos: {len(vids)} | ledger ids: {len(known)} | "
          f"ledger ids NOT on the channel: {len(dead)}\n")
    if dead:
        print("DEAD ledger ids (deleted or re-uploaded):")
        for day, case, vid in sorted(dead, key=lambda x: (x[0] or 999)):
            print(f"  day {str(day or '?'):>3}  {vid}  {case[:52]}")
        print()

    orphans = [v for v in vids if v["id"] not in known]
    print(f"CHANNEL VIDEOS WITH NO LEDGER ENTRY: {len(orphans)}  (newest first)\n")
    for v in sorted(orphans, key=lambda v: v["published"], reverse=True):
        vt = toks(v["title"])
        scored = sorted(((len(vt & b) / max(len(vt), 1), d) for d, b in blobs.items()),
                        reverse=True)
        best, second = scored[0], scored[1] if len(scored) > 1 else (0, None)
        day = best[1]
        cur = by_case.get(state[str(day)]["case"].strip(), {}).get("videoId")
        flag = ""
        if cur and cur not in live:
            flag = "  <-- that day's id is DEAD, likely the replacement"
        elif cur:
            flag = f"  <-- that day already has a LIVE id ({cur}), so this may be a separate upload"
        conf = "strong" if best[0] >= 0.34 and best[0] - second[0] >= 0.12 else "WEAK - check"
        print(f"  {v['id']}  {v['published'][:10]}  {v['title'][:62]}")
        print(f"      best match: day {day} ({conf}, {best[0]:.2f}) {state[str(day)]['case'][:46]}{flag}")

    print("\nNothing has been written. Confirm the pairs you want, then:")
    print("  _reconcile_channel.py --apply 33=H_u0xC8ay_A,34=KzDYKoxFev8")


if __name__ == "__main__":
    main()
