"""Cross-post final.mp4 to the shadow_gasp Facebook Page and Instagram
alongside the YouTube upload. Mirrors _youtube_upload.py's shape (same
VIDEO_PATH/META_PATH, same "print the id line before anything that can fail"
rule from the Telegram-notify bug — see shadow-gasp-telegram-notify-fix).

Facebook accepts a direct binary upload, so no public hosting is needed there.
Instagram's Graph API only accepts a video_url, so the file is temporarily
staged on Cloudinary, referenced, polled until Instagram finishes processing
it, published, then deleted from Cloudinary regardless of outcome.

Credentials come from env vars (GitHub Actions secrets):
  FB_PAGE_ACCESS_TOKEN
  CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET
    (same account/var names as everydayhypehq/scripts/upload_to_cloudinary.py)

IDs are hardcoded, not env vars, because they identify a fixed destination
(this channel's Page/IG account), same as how VIDEO_PATH is hardcoded above.
"""
import json
import os
import sys
import time

import requests

FB_PAGE_ID = "1164008466785123"
IG_USER_ID = "17841425663819735"
GRAPH = "https://graph.facebook.com/v19.0"

VIDEO_PATH = "final.mp4"
META_PATH = "youtube.json"

POLL_INTERVAL_S = 10
POLL_TIMEOUT_S = 600  # IG container processing can take a few minutes for longer videos


def build_caption(meta):
    """Title + first paragraph of the description, IG-caption-length shaped.
    youtube.json's description already ends with 'Subscribe now 👇' — kept
    as-is rather than re-appending a CTA.
    """
    title = meta["title"]
    first_para = meta["description"].split("\n\n")[0].strip()
    caption = f"{title}\n\n{first_para}"
    if len(caption) > 2000:
        caption = caption[:1997] + "..."
    return caption


def post_to_facebook(token, caption):
    with open(VIDEO_PATH, "rb") as f:
        resp = requests.post(
            f"{GRAPH}/{FB_PAGE_ID}/videos",
            data={"description": caption, "access_token": token},
            files={"source": f},
            timeout=600,
        )
    resp.raise_for_status()
    post_id = resp.json()["id"]
    print(f"Facebook posted: https://facebook.com/{post_id}")
    print(f"fb_post_id={post_id}")
    return post_id


def upload_to_cloudinary():
    import cloudinary
    import cloudinary.uploader

    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
        secure=True,
    )
    resp = cloudinary.uploader.upload_large(
        VIDEO_PATH,
        resource_type="video",
        folder="shadow_gasp/ig_staging",
        public_id=f"ig_{int(time.time())}",
    )
    return resp["public_id"], resp["secure_url"]


def delete_from_cloudinary(public_id):
    import cloudinary
    import cloudinary.uploader

    try:
        cloudinary.uploader.destroy(public_id, resource_type="video")
        print(f"Cloudinary staging asset deleted: {public_id}")
    except Exception as e:
        # Non-fatal: a leftover staging asset costs quota, not correctness.
        print(f"Cloudinary cleanup failed ({public_id}): {e}", file=sys.stderr)


def post_to_instagram(token, caption, video_url):
    resp = requests.post(
        f"{GRAPH}/{IG_USER_ID}/media",
        data={
            "media_type": "REELS",  # IG deprecated plain feed VIDEO posts; REELS is the current path
            "video_url": video_url,
            "caption": caption,
            "access_token": token,
        },
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
    if not os.path.isfile(VIDEO_PATH):
        print(f"{VIDEO_PATH} not found", file=sys.stderr)
        sys.exit(1)

    meta = json.load(open(META_PATH, encoding="utf-8"))
    caption = build_caption(meta)
    token = os.environ["FB_PAGE_ACCESS_TOKEN"]

    # Facebook first: it's a direct upload with no external staging, so it
    # can't fail because of anything Cloudinary-related.
    post_to_facebook(token, caption)

    public_id, video_url = upload_to_cloudinary()
    try:
        post_to_instagram(token, caption, video_url)
    finally:
        delete_from_cloudinary(public_id)


if __name__ == "__main__":
    main()
