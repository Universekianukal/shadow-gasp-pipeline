"""Upload final.mp4 to YouTube and stamp the resulting videoId back into the
ledger. Ported from MindUnlocked's _youtube_upload.py, with two shadow_gasp
differences: the render output is final.mp4 (not renders/final.mp4), and the
upload closes the loop on cases_used.json.

_gen_video_content.py reserves the case with "videoId": null at script time so
two concurrent runs can't pick the same case; this fills that null in once the
video is actually live. If the upload never happens the null stays, which is
correct — the case is spoken for but not published.

Credentials come from env vars (GitHub Actions secrets): YOUTUBE_CLIENT_ID,
YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN.
"""
import json
import os
import sys

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

VIDEO_PATH = "final.mp4"
META_PATH = "youtube.json"
# Repo-root-relative by default (how the workflow runs it); override when
# invoking from a project dir, where _pipeline/ isn't a child of cwd.
LEDGER_PATH = os.environ.get("CASES_LEDGER", os.path.join("_pipeline", "cases_used.json"))
THUMBNAIL_CANDIDATES = ["thumbnail.jpg", "thumbnail.png"]
TITLE_LIMIT = 100  # YouTube's hard cap; one char over is a 400 invalidTitle


def clamp_title(title):
    """Keep the title inside YouTube's 100-character cap.

    _gen_youtube_meta.py asks Claude for a hook-style title and the model
    occasionally lands a character or two over the cap (day33's molasses title
    came back at 101), which YouTube rejects outright with a 400
    "invalid or empty video title" -- a confusing error, since the title is
    neither invalid-looking nor empty. Trim at a word boundary and drop any
    trailing punctuation so the result still reads like a written headline
    rather than a truncated string.
    """
    title = title.strip()
    if len(title) <= TITLE_LIMIT:
        return title
    cut = title[:TITLE_LIMIT].rsplit(" ", 1)[0].rstrip(" ,.;:-—")
    print(f"Title was {len(title)} chars, trimmed to {len(cut)}: {cut}", file=sys.stderr)
    return cut


def get_service():
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
        client_id=os.environ["YOUTUBE_CLIENT_ID"],
        client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    return build("youtube", "v3", credentials=creds)


def stamp_ledger(video_id, case):
    """Fill in the videoId on the reserved entry for this case."""
    if not case or not os.path.exists(LEDGER_PATH):
        return
    with open(LEDGER_PATH, encoding="utf-8-sig", errors="replace") as f:
        ledger = json.load(f)
    for entry in reversed(ledger.get("cases", [])):
        if entry.get("case") == case and not entry.get("videoId"):
            entry["videoId"] = video_id
            with open(LEDGER_PATH, "w", encoding="utf-8") as f:
                json.dump(ledger, f, indent=1, ensure_ascii=False)
            print(f"Ledger updated: {case} -> {video_id}")
            return
    print(f"No reserved ledger entry found for '{case}' — not stamping", file=sys.stderr)


def main():
    if not os.path.isfile(VIDEO_PATH):
        print(f"{VIDEO_PATH} not found", file=sys.stderr)
        sys.exit(1)

    meta = json.load(open(META_PATH, encoding="utf-8"))
    # PUBLISH_AT (RFC3339 UTC, e.g. 2026-07-30T23:45:00Z) schedules the video
    # instead of publishing immediately. YouTube requires privacyStatus
    # "private" for a scheduled video — it flips to public automatically at
    # publishAt, so the meta.json privacyStatus is overridden in this case.
    publish_at = os.environ.get("PUBLISH_AT", "").strip()
    status = {
        "privacyStatus": "private" if publish_at else meta.get("privacyStatus", "private"),
        "selfDeclaredMadeForKids": False,
    }
    if publish_at:
        status["publishAt"] = publish_at
    body = {
        "snippet": {
            "title": clamp_title(meta["title"]),
            "description": meta["description"],
            "tags": meta.get("tags", []),
            "categoryId": meta.get("categoryId", "22"),  # 22 = People & Blogs
        },
        "status": status,
    }

    yt = get_service()
    media = MediaFileUpload(VIDEO_PATH, chunksize=-1, resumable=True, mimetype="video/mp4")
    request = yt.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")

    video_id = response["id"]
    print(f"Uploaded: https://youtu.be/{video_id}")
    # Printed right after the upload succeeds, before any of the steps below,
    # so a Telegram notify (which greps this line out of the job log) still
    # fires even if thumbnail-setting or ledger-stamping blows up.
    print(f"video_id={video_id}")

    thumb = next((p for p in THUMBNAIL_CANDIDATES if os.path.isfile(p)), None)
    if thumb:
        try:
            yt.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumb)).execute()
            print(f"Thumbnail set from {thumb}")
        except HttpError as e:
            # Custom thumbnails need a phone-verified channel; a 403 here must
            # not fail an otherwise-successful upload.
            print(f"Thumbnail upload failed ({thumb}): {e}", file=sys.stderr)

    try:
        stamp_ledger(video_id, os.environ.get("CASE", "").strip())
    except Exception as e:
        # The video is already live at this point -- a broken ledger write
        # (e.g. a stray non-UTF-8 byte in a past LLM-generated case name)
        # must not look like an upload failure or swallow the video_id line.
        print(f"Ledger stamp failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
