"""Add a comic's Gumroad link to the description of the ALREADY-PUBLISHED video it came from.

This is the funnel: the short is the advertisement that already has an audience, the comic is
the product. Without this step a comic is published to nobody -- NORJAK sat on Gumroad with 0
sales while the D.B. Cooper video that would sell it never mentioned it existed.

Env:
  VIDEO_ID       the published YouTube video to edit
  PRODUCT_URL    the Gumroad permalink (https://shadowgasp.gumroad.com/l/...)
  PRODUCT_NAME   display name, e.g. "SHADOW GASP #2: HEAVEN'S GATE"
  PAGES          optional page count for the blurb, e.g. "80"
  POSITION       "top" (default) or "bottom"
  ALLOW_DRAFT    "true" to skip the published-product check (default false)

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


def link_block(name, url, pages=""):
    # "An 80-page", not "A 80-page" -- 8, 11 and 18 read with a leading vowel sound. This is
    # public-facing copy on a channel's video description, so it has to scan.
    article = "An" if pages and pages.lstrip("0")[:1] in ("8",) or pages in ("11", "18") else "A"
    detail = f"{article} {pages}-page" if pages else "A full-length"
    return (
        f"\U0001f4d5 READ THE COMIC: {name}\n"
        f"{url}\n"
        f"{detail} documentary comic on this case, from the same research as this video.\n"
    )


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

    if product_url in description:
        print(f"ALREADY_LINKED: {product_url} is already in {video_id}'s description, nothing to do")
        print(f"video_id={video_id}")
        return

    block = link_block(product_name, product_url, os.environ.get("PAGES", "").strip())
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
