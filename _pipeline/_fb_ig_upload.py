"""Post final.mp4 (via its permanent Cloudinary URL, from _cloudinary_upload.py)
to either the shadow_gasp Facebook Page or Instagram account -- never both in
one run. Split out of the old combined script so each platform can be
approved or rejected independently from Telegram, at any point after render
(see crosspost_decision.yml) -- there is no PUBLISH_AT/scheduling concept
here anymore: /publish only ever schedules YouTube now, and Facebook/
Instagram both go out immediately, the moment a human taps Approve in
Telegram, however long after render that happens to be.

Credentials from env vars (GitHub Actions secrets): FB_PAGE_ACCESS_TOKEN.

Required env vars:
  PLATFORM    "fb" or "ig"
  VIDEO_URL   permanent Cloudinary URL (from _cloudinary_upload.py, never deleted)
  DAY_DIR     the day's own directory, for the FB_POSTED/IG_POSTED marker and youtube.json
"""
import json
import os
import sys
import time

import requests

FB_PAGE_ID = "1164008466785123"
IG_USER_ID = "17841425663819735"
GRAPH = "https://graph.facebook.com/v19.0"

META_FILENAME = "youtube.json"
DAY_DIR = os.environ.get("DAY_DIR", ".")

POLL_INTERVAL_S = 10
POLL_TIMEOUT_S = 600  # IG container processing can take a few minutes for longer videos


def build_caption(meta):
    """Title + first paragraph of the description + hashtags, IG-caption-length
    shaped. youtube.json's "tags" are plain lowercase keywords with no "#"
    (meant for YouTube's separate tags field) -- this is the only place they
    get turned into real hashtags, for Facebook/Instagram captions.
    """
    title = meta["title"]
    first_para = meta["description"].split("\n\n")[0].strip()
    hashtags = " ".join(f"#{tag.replace(' ', '')}" for tag in meta.get("tags", [])[:8])
    caption = f"{title}\n\n{first_para}"
    if hashtags:
        caption += f"\n\n{hashtags}"
    if len(caption) > 2000:
        caption = caption[:1997] + "..."
    return caption


def post_to_facebook(token, caption, video_url, schedule_at=None):
    # file_url instead of a multipart binary upload -- the permanent
    # Cloudinary link means there's no need to have final.mp4 on disk at all,
    # which matters here since this can run days after the original render's
    # artifact has expired.
    data = {"description": caption, "file_url": video_url, "access_token": token}
    if schedule_at:
        # Facebook's own scheduler: the video uploads now, sits in Meta Business
        # Suite's Planner (movable/editable there) and Facebook publishes it at
        # this unix time. Meta requires it to be at least 10 minutes ahead.
        data.update({"published": "false", "scheduled_publish_time": str(schedule_at)})
    resp = requests.post(
        f"{GRAPH}/{FB_PAGE_ID}/videos",
        data=data,
        timeout=600,
    )
    if not resp.ok:
        print(f"Facebook refused: {resp.status_code} {resp.text[:500]}", file=sys.stderr)
    resp.raise_for_status()
    post_id = resp.json()["id"]
    if schedule_at:
        print(f"Facebook scheduled for unix {schedule_at}: https://facebook.com/{post_id}")
        print(f"fb_scheduled_at={schedule_at}")
    else:
        print(f"Facebook posted: https://facebook.com/{post_id}")
    print(f"fb_post_id={post_id}")
    return post_id


def post_to_instagram(token, caption, video_url, reels_only=False):
    data = {
        "media_type": "REELS",  # IG deprecated plain feed VIDEO posts; REELS is the current path
        "video_url": video_url,
        "caption": caption,
        "access_token": token,
    }
    if reels_only:
        # "🎞 IG: Reels only" button: the Reel appears ONLY in the Reels tab -- not the
        # profile grid / followers' feed. Can only be set at creation (the API can't move it later).
        data["share_to_feed"] = "false"
    resp = requests.post(
        f"{GRAPH}/{IG_USER_ID}/media",
        data=data,
        timeout=60,
    )
    resp.raise_for_status()
    container_id = resp.json()["id"]

    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        status_resp = requests.get(
            f"{GRAPH}/{container_id}",
            params={"fields": "status_code", "access_token": token},
            timeout=30,
        )
        status_resp.raise_for_status()
        status_code = status_resp.json().get("status_code")
        if status_code == "FINISHED":
            break
        if status_code == "ERROR":
            raise RuntimeError(f"Instagram container {container_id} failed processing")
        time.sleep(POLL_INTERVAL_S)
    else:
        raise TimeoutError(f"Instagram container {container_id} did not finish within {POLL_TIMEOUT_S}s")

    publish_resp = requests.post(
        f"{GRAPH}/{IG_USER_ID}/media_publish",
        data={"creation_id": container_id, "access_token": token},
        timeout=60,
    )
    publish_resp.raise_for_status()
    media_id = publish_resp.json()["id"]
    print(f"Instagram posted: media_id={media_id}")
    print(f"ig_media_id={media_id}")
    return media_id


def main():
    platform = os.environ.get("PLATFORM", "").strip().lower()
    if platform not in ("fb", "ig"):
        print("PLATFORM must be 'fb' or 'ig'", file=sys.stderr)
        sys.exit(1)

    video_url = os.environ.get("VIDEO_URL", "").strip()
    if not video_url:
        print("VIDEO_URL not set", file=sys.stderr)
        sys.exit(1)

    meta = json.load(open(os.path.join(DAY_DIR, META_FILENAME), encoding="utf-8"))
    caption = build_caption(meta)
    token = os.environ["FB_PAGE_ACCESS_TOKEN"]
    force = os.environ.get("FORCE_CROSSPOST", "").strip().lower() == "true"

    os.makedirs(DAY_DIR, exist_ok=True)
    marker_path = os.path.join(DAY_DIR, "FB_POSTED" if platform == "fb" else "IG_POSTED")
    if os.path.exists(marker_path) and not force:
        print(f"{marker_path} present -- already posted, skipping to avoid a duplicate "
              f"(set FORCE_CROSSPOST=true to force a repost)")
        return

    schedule_at = os.environ.get("SCHEDULE_AT", "").strip()
    if schedule_at and platform != "fb":
        print("SCHEDULE_AT is Facebook-only (Instagram's API cannot schedule)", file=sys.stderr)
        sys.exit(1)

    if platform == "fb":
        post_to_facebook(token, caption, video_url, schedule_at=schedule_at or None)
    else:
        reels_only = os.environ.get("IG_REELS_ONLY", "").strip().lower() == "true"
        post_to_instagram(token, caption, video_url, reels_only=reels_only)
    with open(marker_path, "w") as f:
        # FB_POSTED on a scheduled video = handed to Facebook's scheduler (blocks a double post).
        if schedule_at:
            f.write(f"scheduled_publish_time={schedule_at}\n")


if __name__ == "__main__":
    main()
