"""Real-time stamp: called by publish_batch_day.yml right after a successful
YouTube upload so the "shadow_gasp - 30 day batch" sheet reflects the publish
the moment it happens, instead of waiting for the next _sync_youtube_status.py
poll. A scheduled upload (PUBLISH_AT set) still stamps immediately, but as
"Scheduled" -- _sync_youtube_status.py is what later flips it to "Public"
once YouTube's own publishAt fires, since nothing calls this script again at
that moment.

Env: DAY (e.g. "14"), VIDEO_ID, PUBLISH_AT (optional, RFC3339 UTC).
"""
import datetime
import os
import sys

from google.oauth2 import service_account
from googleapiclient.discovery import build

SA_KEY_PATH = os.path.join("_local", "sheets_sa_key.json")
SHEET_ID = "1aPoXPKlC9cCStUqULzR46FmvUaL8jxQbFsDWEDXn3jM"
SHEET_TAB = "Batch"


def main():
    day = int(os.environ["DAY"])
    video_id = os.environ["VIDEO_ID"].strip()
    publish_at = os.environ.get("PUBLISH_AT", "").strip()
    row = day + 1

    status = f"Scheduled for {publish_at}" if publish_at else "Public"
    stamped_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    creds = service_account.Credentials.from_service_account_file(
        SA_KEY_PATH, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    sheets = build("sheets", "v4", credentials=creds).spreadsheets().values()
    sheets.batchUpdate(
        spreadsheetId=SHEET_ID,
        body={
            "valueInputOption": "RAW",
            "data": [
                {"range": f"{SHEET_TAB}!I{row}", "values": [[status]]},
                {"range": f"{SHEET_TAB}!L{row}", "values": [[video_id]]},
                {"range": f"{SHEET_TAB}!M{row}", "values": [[stamped_at]]},
            ],
        },
    ).execute()
    print(f"day {day}: sheet row {row} stamped ({status}, {video_id})", file=sys.stderr)


if __name__ == "__main__":
    main()
