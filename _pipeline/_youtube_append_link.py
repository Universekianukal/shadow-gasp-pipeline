"""Add a comic's Gumroad link to the description of the ALREADY-PUBLISHED video it came from.

This is the funnel: the short is the advertisement that already has an audience, the comic is
the product. Without this step a comic is published to nobody -- NORJAK sat on Gumroad with 0
sales while the D.B. Cooper video that would sell it never mentioned it existed.

Env:
  VIDEO_ID       the published YouTube video to edit
  PRODUCT_URL    the Gumroad permalink (https://shadowgasp.gumroad.com/l/...)
  PRODUCT_NAME   display name, e.g. "SHADOW GASP #2: HEAVEN'S GATE"
  PAGES          optional page count for the blurb, e.g. "80"
  HOOK           optional one-line hook from the script (promo_hook)
  POSITION       "top" (default) or "bottom"
  ALLOW_DRAFT    "true" to skip the published-product check (default false)
  REPLACE        "true" to strip an existing block and re-write it (for copy changes)

Requires the full "youtube" OAuth scope, same as _youtube_update_title.py -- videos.update
403s under the upload-only scope. Re-mint via _youtube_auth_remint.py if that happens.

Safety properties, each of which exists because the opposite would be worse than doing nothing:

1. IDEMPOTENT. Re-running never appends a second copy -- the product URL is searched for in the
   existing description first. The Telegram button can be tapped twice by accident, and a
   description with the link pasted three times looks like spam on a public video.
2. REFUSES DRAFTS. A Gumroad draft URL 404s for the public, so linking one from a live video
   sends real viewers to a dead page. The product is fetched and must be published, unless
   ALLOW_DRAFT is set.
3. PRESERVES THE SNIPPET. Fetches the current snippet and edits only `description`, so title,
   tags and categoryId survive -- videos.update overwrites whatever it is not given.
4. RESPECTS THE 5000-CHAR LIMIT. YouTube rejects longer descriptions; the block is not added if
   it would overflow, rather than silently truncating someone's description.
"""
import os
import sys
import urllib.request

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

DESCRIPTION_LIMIT = 5000


def link_block(name, url, pages="", hook=""):
    """The block written into the video description.

    This has to answer "why read this when I just watched the video?". The first version said
    the comic was "from the same research as this video", which is accurate and is exactly the
    wrong thing to tell a viewer -- it describes the comic as a duplicate of the thing they have
    already consumed for free.

    So: lead with the case's own hook (the script writes one for precisely this job), then name
    what the comic actually adds -- length and form. A few minutes of video cannot carry fifty
    pages of drawn narrative, and that difference is the entire offer.
    """
    size = f"{pages}-page " if pages else ""
    lines = [f"📕 THE COMIC — {name}"]
    if hook:
        lines.append(hook.strip())
    lines.append(
        f"The whole story as a {size}illustrated book: the people, the timeline and the detail "
        "there was never room for here.")
    lines.append(url)
    return "\n".join(lines) + "\n"


def gumroad_is_published(url):
    """Best-effort public check: a draft permalink is not publicly reachable."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "shadow-gasp-funnel"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return 200 <= r.status < 300
    except Exception as exc:  # noqa: BLE001 - any failure means "cannot prove it is live"
        print(f"product URL check failed: {exc}", file=sys.stderr)
        return False


def get_youtube_service():
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
        client_id=os.environ["YOUTUBE_CLIENT_ID"],
        client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/youtube"],
    )
    return build("youtube", "v3", credentials=creds)


def main():
    video_id = os.environ["VIDEO_ID"].strip()
    product_url = os.environ["PRODUCT_URL"].strip()
    product_name = os.environ["PRODUCT_NAME"].strip()
    position = os.environ.get("POSITION", "top").strip().lower()
    allow_draft = os.environ.get("ALLOW_DRAFT", "").lower() == "true"

    if not allow_draft and not gumroad_is_published(product_url):
        print(
            f"REFUSED: {product_url} is not publicly reachable, so it is probably still a "
            "Gumroad draft. Publish the product first, or set ALLOW_DRAFT=true. Linking a "
            "draft from a live video sends viewers to a 404."
        )
        sys.exit(1)

    yt = get_youtube_service()
    items = yt.videos().list(part="snippet", id=video_id).execute().get("items", [])
    if not items:
        print(f"video_id {video_id} does not resolve on YouTube (deleted? wrong channel?)",
              file=sys.stderr)
        sys.exit(1)

    snippet = items[0]["snippet"]
    description = snippet.get("description", "")

    replace = os.environ.get("REPLACE", "").lower() == "true"
    if product_url in description and not replace:
        print(f"ALREADY_LINKED: {product_url} is already in {video_id}'s description, nothing to do")
        print(f"video_id={video_id}")
        return

    if product_url in description:
        # REPLACE: strip the old block so improved copy can go in. Without this the idempotency
        # guard makes the wording permanent -- the first version told viewers the comic was
        # "from the same research as this video", which describes it as a duplicate of the free
        # thing they just watched. Match only OUR lines, never the author's own description.
        keep = [ln for ln in description.split("\n")
                if ln.strip() != product_url
                and not ln.startswith("\U0001f4d5")
                and "from the same research as this video" not in ln
                and not ln.startswith("The whole story as a")]
        description = "\n".join(keep).strip()
        print(f"REPLACING the existing block in {video_id}", flush=True)

    block = link_block(product_name, product_url,
                       os.environ.get("PAGES", "").strip(),
                       os.environ.get("HOOK", "").strip())
    new_description = (
        f"{block}\n{description}" if position == "top" else f"{description}\n\n{block}"
    ).strip()

    if len(new_description) > DESCRIPTION_LIMIT:
        print(
            f"REFUSED: adding the link would make the description {len(new_description)} chars, "
            f"over YouTube's {DESCRIPTION_LIMIT} limit. Trim the description first.",
            file=sys.stderr,
        )
        sys.exit(1)

    snippet["description"] = new_description
    yt.videos().update(part="snippet", body={"id": video_id, "snippet": snippet}).execute()

    print(f"LINKED https://youtu.be/{video_id} -> {product_name}")
    print(f"  {product_url}  (position: {position})")
    print(f"  description {len(description)} -> {len(new_description)} chars")
    print(f"video_id={video_id}")


if __name__ == "__main__":
    main()
