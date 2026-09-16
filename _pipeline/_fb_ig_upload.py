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


def _fb_video_status(token, video_id):
    r = requests.get(f"{GRAPH}/{video_id}", params={"fields": "status", "access_token": token}, timeout=30)
    r.raise_for_status()
    return r.json().get("status") or {}


def post_fb_reel(token, caption, video_url, schedule_at=None):
    """"🎞 FB: Reel" button: publish as a Facebook REEL (Page Reels section, recommended to
    non-followers too) via the Reels Publishing API -- start -> hosted-file upload -> finish.
    Reels must be 3-90 s, 9:16; max 30 API reels per Page per 24 h."""
    start = requests.post(f"{GRAPH}/{FB_PAGE_ID}/video_reels",
                          data={"upload_phase": "start", "access_token": token}, timeout=60)
    if not start.ok:
        print(f"Facebook refused the reel start: {start.status_code} {start.text[:500]}", file=sys.stderr)
    start.raise_for_status()
    video_id = start.json()["video_id"]

    up = requests.post(f"https://rupload.facebook.com/video-upload/{GRAPH.rsplit('/', 1)[-1]}/{video_id}",
                       headers={"Authorization": f"OAuth {token}", "file_url": video_url}, timeout=600)
    if not up.ok:
        print(f"Facebook refused the reel upload: {up.status_code} {up.text[:500]}", file=sys.stderr)
    up.raise_for_status()

    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        st = _fb_video_status(token, video_id)
        phase = (st.get("uploading_phase") or {}).get("status")
        if st.get("video_status") == "error" or phase == "error":
            raise RuntimeError(f"Facebook reel {video_id} upload failed: {st}")
        if phase == "complete" or st.get("video_status") in ("ready", "upload_complete"):
            break
        time.sleep(POLL_INTERVAL_S)
    else:
        raise TimeoutError(f"Facebook reel {video_id} upload did not finish within {POLL_TIMEOUT_S}s")

    fin_data = {"upload_phase": "finish", "video_id": video_id, "video_state": "PUBLISHED",
                "description": caption, "access_token": token}
    if schedule_at:
        # "⏰ FB: Schedule Reel": Facebook holds it in the Planner (10 min - 29 days ahead).
        fin_data.update({"video_state": "SCHEDULED", "scheduled_publish_time": str(schedule_at)})
    fin = requests.post(f"{GRAPH}/{FB_PAGE_ID}/video_reels", data=fin_data, timeout=120)
    if not fin.ok:
        print(f"Facebook refused to publish the reel: {fin.status_code} {fin.text[:500]}", file=sys.stderr)
    fin.raise_for_status()
    if not fin.json().get("success"):
        raise RuntimeError(f"Facebook reel {video_id} finish returned {fin.text[:300]}")

    deadline = time.time() + (0 if schedule_at else POLL_TIMEOUT_S)  # scheduled: nothing to wait for
    while time.time() < deadline:
        st = _fb_video_status(token, video_id)
        pub = (st.get("publishing_phase") or {}).get("status")
        if st.get("video_status") == "error" or pub == "error" or (st.get("processing_phase") or {}).get("status") == "error":
            raise RuntimeError(f"Facebook reel {video_id} failed processing: {st}")
        if pub == "complete":
            break
        time.sleep(POLL_INTERVAL_S)
    else:
        if not schedule_at:
            print(f"WARNING: reel {video_id} not reported published within {POLL_TIMEOUT_S}s -- check the Page", file=sys.stderr)

    ref = video_id
    try:  # the post id is what FB comments carry (comment funnel); fall back to the video id
        r = requests.get(f"{GRAPH}/{video_id}", params={"fields": "post_id", "access_token": token}, timeout=30)
        if r.ok and r.json().get("post_id"):
            ref = f"{FB_PAGE_ID}_{r.json()['post_id']}" if "_" not in str(r.json()["post_id"]) else r.json()["post_id"]
    except requests.RequestException:
        pass
    if schedule_at:
        print(f"Facebook reel scheduled for unix {schedule_at}: video_id={video_id} https://facebook.com/{ref}")
        print(f"fb_scheduled_at={schedule_at}")
    else:
        print(f"Facebook reel posted: video_id={video_id} https://facebook.com/{ref}")
    print(f"fb_post_id={ref}")
    return ref


class IgProcessingError(RuntimeError):
    def __init__(self, container_id, detail):
        super().__init__(f"Instagram container {container_id} failed processing: {detail}")
        self.container_id = container_id


# Retry recipes, in order: a lossless remux with the index moved to the front, then a re-encode to
# the settings Meta documents as safe for Reels.
FASTSTART = ["-c", "copy", "-movflags", "+faststart"]
SAFE_REENCODE = ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                 "-crf", "20", "-maxrate", "6M", "-bufsize", "12M", "-r", "30",
                 "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-movflags", "+faststart"]


def _ffmpeg():
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # pip: imageio-ffmpeg (GitHub's ubuntu-24.04 image has no ffmpeg)
    return imageio_ffmpeg.get_ffmpeg_exe()


def _faststart_copy(video_url, recipe=FASTSTART):
    """A TEMPORARY Cloudinary copy of the video, rebuilt with `recipe` (see FASTSTART / SAFE_REENCODE).

    Instagram refused day 22 twice and day 19 twice with a bare "failed processing" while the
    same day 19 file went through minutes later (2026-09-16). The renders keep moov at the END of
    the file, which Meta advises against for URL fetches, and a refused fetch can stick to the
    URL. So the retry gets a remuxed copy (-c copy: no re-encode) under a brand-new public_id.
    Returns (url, public_id); the caller deletes it.
    """
    import subprocess
    import tempfile

    import cloudinary
    import cloudinary.uploader

    cloudinary.config(cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
                      api_key=os.environ["CLOUDINARY_API_KEY"],
                      api_secret=os.environ["CLOUDINARY_API_SECRET"])
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.mp4"), os.path.join(tmp, "out.mp4")
        with requests.get(video_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(src, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        subprocess.run([_ffmpeg(), "-v", "error", "-y", "-i", src, *recipe, dst], check=True)
        public_id = f"shadow_gasp_ig_retry_{int(time.time())}"
        resp = cloudinary.uploader.upload_large(dst, resource_type="video", public_id=public_id)
    return resp["secure_url"], public_id


def post_to_instagram(token, caption, video_url, reels_only=False):
    try:
        return _post_to_instagram(token, caption, video_url, reels_only)
    except IgProcessingError as first:
        print(f"WARNING: {first}")
        if not os.environ.get("CLOUDINARY_API_SECRET"):
            raise
        last = first
    for label, recipe in (("faststart remux", FASTSTART), ("safe re-encode", SAFE_REENCODE)):
        print(f"retrying from a temporary {label} of the video...")
        url, public_id = _faststart_copy(video_url, recipe)
        try:
            return _post_to_instagram(token, caption, url, reels_only)
        except IgProcessingError as e:
            print(f"WARNING: {e}")
            last = e
        finally:
            try:
                import cloudinary.uploader
                cloudinary.uploader.destroy(public_id, resource_type="video", invalidate=True)
            except Exception as e:
                print(f"WARNING: could not delete the temporary copy {public_id}: {e}")
    raise last


def _post_to_instagram(token, caption, video_url, reels_only=False):
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
            params={"fields": "status_code,status", "access_token": token},
            timeout=30,
        )
        status_resp.raise_for_status()
        status_code = status_resp.json().get("status_code")
        if status_code == "FINISHED":
            break
        if status_code == "ERROR":
            # `status` carries Instagram's own reason ("Error: ..."); it used to go unread.
            raise IgProcessingError(container_id, status_resp.json().get("status") or "no reason given")
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

    as_reel = os.environ.get("FB_AS_REEL", "").strip().lower() == "true"
    if as_reel and platform != "fb":
        print("FB_AS_REEL is Facebook-only", file=sys.stderr)
        sys.exit(1)

    if platform == "fb" and as_reel:
        post_fb_reel(token, caption, video_url, schedule_at=schedule_at or None)
    elif platform == "fb":
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
