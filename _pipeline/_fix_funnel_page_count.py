"""One-off: strip the wrong page count from comic funnel blocks already in live descriptions.

The funnel blurb said "The whole story as a 25-page illustrated book" using the page count the
build was ASKED for (Worker default 25), not what the book delivered -- 40-100 page comics were
described as 25 pages (owner, 2026-09-29). _youtube_append_link.py no longer writes a count; this
rewrites the sentence in videos that already carry it, and nothing else in the description.

    DRY_RUN=true  (default) -> list what would change
    DRY_RUN=false           -> update those descriptions
"""
import os
import re
import sys

from _youtube_append_link import get_youtube_service

PATTERN = re.compile(r"The whole story as an? (?:\d+-page )?illustrated book")
FIXED = "The whole story as an illustrated book"


def uploads(yt):
    ch = yt.channels().list(part="contentDetails", mine=True).execute()["items"][0]
    pl = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    token = None
    while True:
        r = yt.playlistItems().list(part="contentDetails", playlistId=pl, maxResults=50,
                                    pageToken=token).execute()
        for it in r.get("items", []):
            yield it["contentDetails"]["videoId"]
        token = r.get("nextPageToken")
        if not token:
            return


def main():
    dry = os.environ.get("DRY_RUN", "true").strip().lower() != "false"
    yt = get_youtube_service()
    ids = list(uploads(yt))
    print(f"{len(ids)} uploads on the channel; dry_run={dry}", flush=True)
    changed = failed = 0
    for i in range(0, len(ids), 50):
        for v in yt.videos().list(part="snippet", id=",".join(ids[i:i + 50])).execute()["items"]:
            sn = v["snippet"]
            desc = sn.get("description", "")
            hits = [m.group(0) for m in PATTERN.finditer(desc) if m.group(0) != FIXED]
            if not hits:
                continue
            print(f"{v['id']}  {sn['title'][:60]!r}  {hits}", flush=True)
            if dry:
                changed += 1
                continue
            sn["description"] = PATTERN.sub(FIXED, desc)
            try:
                yt.videos().update(part="snippet", body={"id": v["id"], "snippet": sn}).execute()
                changed += 1
            except Exception as e:  # noqa: BLE001 - report and keep going
                failed += 1
                print(f"  FAILED {v['id']}: {e}", flush=True)
    verb = "would fix" if dry else "fixed"
    print(f"{verb} {changed} description(s); {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
