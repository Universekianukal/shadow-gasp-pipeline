"""Runs every ~15 minutes (see .github/workflows/sync_youtube_status.yml).
Safety net for the one thing the real-time stamps in publish_batch_day.yml
can't see: a video uploaded with PUBLISH_AT (scheduled) flips from private to
public automatically on YouTube's own clock, with no webhook or job to catch
the moment. This polls every day's already-known video ID and updates the
sheet's status column whenever it no longer matches what's there, so a
YouTube-scheduled publish shows up within one poll cycle instead of only
whenever a human happens to check.

Deliberately does NOT try to guess a day for a channel video that isn't in
any ledger entry -- title-matching a working title ("The Zanzibar Ghost
Ship...") against the actual AI-generated hook title on the video ("No
Boots, No Coats...") is exactly the kind of wrong-guess that produced the
stale/mismatched entries this script exists to stop making worse. Unmatched
videos are only logged, never auto-linked.
"""
import json
import os
import sys

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SA_KEY_PATH = os.path.join("_local", "sheets_sa_key.json")
SHEET_ID = "1aPoXPKlC9cCStUqULzR46FmvUaL8jxQbFsDWEDXn3jM"
SHEET_TAB = "Batch"

STATE_PATH = os.path.join("batch", "state.json")
LEDGER_PATH = "cases_used.json"


def get_sheets():
    creds = service_account.Credentials.from_service_account_file(
        SA_KEY_PATH, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=creds).spreadsheets().values()


def get_youtube():
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
        client_id=os.environ["YOUTUBE_CLIENT_ID"],
        client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/youtube.readonly"],
    )
    return build("youtube", "v3", credentials=creds)


def main():
    state = json.load(open(STATE_PATH, encoding="utf-8"))["days"]
    ledger = json.load(open(LEDGER_PATH, encoding="utf-8"))["cases"]
    case_to_video = {c["case"].strip(): c.get("videoId") for c in ledger}

    # day -> videoId, only where the pipeline's own ledger already has one
    day_video = {}
    for day, info in state.items():
        vid = case_to_video.get(info["case"].strip())
        if vid:
            day_video[int(day)] = vid

    if not day_video:
        print("no known video IDs yet, nothing to reconcile", file=sys.stderr)
        return

    sheets = get_sheets()
    sheets.update(
        spreadsheetId=SHEET_ID,
        range=f"{SHEET_TAB}!I1:M1",
        valueInputOption="RAW",
        body={"values": [["YT Status", "FB Posted", "IG Posted", "YT Video ID", "YT Published At"]]},
    ).execute()

    yt = get_youtube()
    all_ids = sorted(set(day_video.values()))
    live = {}
    for i in range(0, len(all_ids), 50):
        chunk = all_ids[i : i + 50]
        resp = yt.videos().list(part="snippet,status", id=",".join(chunk)).execute()
        for v in resp["items"]:
            live[v["id"]] = {
                "status": v["status"]["privacyStatus"],
                "publishedAt": v["snippet"]["publishedAt"],
            }

    updates = []
    for day, vid in sorted(day_video.items()):
        row = day + 1
        info = live.get(vid)
        if info is None:
            status_label = "Removed / not found on channel"
            published = ""
        else:
            status_label = info["status"].capitalize()
            published = info["publishedAt"]
        # append_sheet_row (pregen) never ran for every day (sheet_logged is
        # false for several), so some rows we're about to touch have no
        # Day/Case label at all yet -- fill those in too so the status
        # columns don't land on an otherwise-blank row.
        updates.append({"range": f"{SHEET_TAB}!A{row}", "values": [[day]]})
        updates.append({"range": f"{SHEET_TAB}!B{row}", "values": [[state[str(day)]["case"]]]})
        updates.append({"range": f"{SHEET_TAB}!I{row}", "values": [[status_label]]})
        updates.append({"range": f"{SHEET_TAB}!L{row}", "values": [[vid]]})
        if published:
            updates.append({"range": f"{SHEET_TAB}!M{row}", "values": [[published]]})

    sheets.batchUpdate(
        spreadsheetId=SHEET_ID,
        body={"valueInputOption": "RAW", "data": updates},
    ).execute()
    print(f"reconciled {len(day_video)} days", file=sys.stderr)

    # Report-only: videos on the channel with no matching ledger entry at all.
    known_ids = set(all_ids)
    page_token = None
    ch = yt.channels().list(part="contentDetails", mine=True).execute()
    uploads = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    unmatched = []
    while True:
        resp = yt.playlistItems().list(
            part="contentDetails,snippet", playlistId=uploads, maxResults=50, pageToken=page_token
        ).execute()
        for it in resp["items"]:
            vid = it["contentDetails"]["videoId"]
            if vid not in known_ids and vid not in case_to_video.values():
                unmatched.append((vid, it["snippet"]["title"], it["snippet"]["publishedAt"]))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    if unmatched:
        print(f"{len(unmatched)} channel videos have no ledger entry at all (not from this batch pipeline, or an old/manual upload) -- not auto-linked:", file=sys.stderr)
        for vid, title, published in unmatched[:10]:
            print(f"  {vid}\t{published}\t{title}", file=sys.stderr)


if __name__ == "__main__":
    main()
