"""Change the title of an ALREADY-PUBLISHED shadow_gasp video.

Looks up the day's video_id from the "shadow_gasp - 30 day batch" sheet
(column L, same place _stamp_youtube_sheet.py writes it), fetches the video's
current snippet so unrelated fields (description, tags, categoryId) survive
untouched, and calls videos.update with just the title changed.

Requires the YOUTUBE_REFRESH_TOKEN to carry the full "youtube" scope, not the
upload-only scope the pipeline used before -- videos.update 403s under
youtube.upload. Re-mint via _youtube_auth_remint.py if this 403s with
insufficientPermissions.

Env: DAY (e.g. "13"), NEW_TITLE, SHEETS_SA_KEY_PATH (optional override),
YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN.

Prints "video_id=<id>" on success (same convention as _youtube_upload.py) so
a workflow can grep it out; prints "NOT_PUBLISHED" and exits 0 (not an error)
if the day has no video_id yet, so a caller can fall back to the pre-upload
TITLE_OVERRIDE.json path instead.
"""
import os
import sys

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SHEET_ID = "1aPoXPKlC9cCStUqULzR46FmvUaL8jxQbFsDWEDXn3jM"
SHEET_TAB = "Batch"
SA_KEY_PATH = os.environ.get(
    "SHEETS_SA_KEY_PATH", os.path.join("_pipeline", "_local", "sheets_sa_key.json")
)
TITLE_LIMIT = 100


def clamp_title(title):
    title = title.strip()
    if len(title) <= TITLE_LIMIT:
        return title
    cut = title[:TITLE_LIMIT].rsplit(" ", 1)[0].rstrip(" ,.;:-—")
    return cut


def lookup_video_id(day):
    row = day + 1
    creds = service_account.Credentials.from_service_account_file(
        SA_KEY_PATH, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    sheets = build("sheets", "v4", credentials=creds).spreadsheets().values()
    resp = sheets.get(spreadsheetId=SHEET_ID, range=f"{SHEET_TAB}!L{row}").execute()
    values = resp.get("values", [])
    return values[0][0].strip() if values and values[0] else ""


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
    day = int(os.environ["DAY"])
    new_title = os.environ["NEW_TITLE"]

    video_id = lookup_video_id(day)
    if not video_id:
        print("NOT_PUBLISHED")
        return

    yt = get_youtube_service()
    current = yt.videos().list(part="snippet", id=video_id).execute()
    items = current.get("items", [])
    if not items:
        print(f"video_id {video_id} from the sheet doesn't resolve on YouTube (deleted?)", file=sys.stderr)
        sys.exit(1)

    snippet = items[0]["snippet"]
    snippet["title"] = clamp_title(new_title)

    yt.videos().update(part="snippet", body={"id": video_id, "snippet": snippet}).execute()
    print(f"Retitled https://youtu.be/{video_id} -> {snippet['title']}")
    print(f"video_id={video_id}")


if __name__ == "__main__":
    main()
