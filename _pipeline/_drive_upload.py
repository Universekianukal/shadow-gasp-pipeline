"""Upload final.mp4 to the channel owner's real Google Drive (not a service
account -- those have zero storage quota on personal Gmail, which is exactly
the wall hit earlier trying to host batch stills there) so the Telegram bot
can send back a real, playable-link video instead of just a YouTube URL.
Telegram's own bot API caps file uploads at 50MB; these renders run ~100MB,
so the file itself can't go through Telegram directly.

Credentials: DRIVE_CLIENT_ID/SECRET (same OAuth client as YouTube) +
DRIVE_REFRESH_TOKEN (separate token, minted drive.file-only -- Google
rejects combining drive.file with the YouTube scopes in one consent request).

Sets "anyone with the link can view" so it opens straight from Telegram on
a phone with no extra sign-in.
"""
import os
import sys

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

VIDEO_PATH = "final.mp4"


def get_service():
    creds = Credentials(
        token=None,
        refresh_token=os.environ["DRIVE_REFRESH_TOKEN"],
        client_id=os.environ["YOUTUBE_CLIENT_ID"],
        client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    return build("drive", "v3", credentials=creds)


def main():
    if not os.path.isfile(VIDEO_PATH):
        print(f"{VIDEO_PATH} not found", file=sys.stderr)
        sys.exit(1)

    name = sys.argv[1] if len(sys.argv) > 1 else "shadow_gasp_video.mp4"
    drive = get_service()
    media = MediaFileUpload(VIDEO_PATH, mimetype="video/mp4", resumable=True)
    request = drive.files().create(body={"name": name}, media_body=media, fields="id,webViewLink")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")

    file_id = response["id"]
    drive.permissions().create(fileId=file_id, body={"role": "reader", "type": "anyone"}).execute()
    link = drive.files().get(fileId=file_id, fields="webViewLink").execute()["webViewLink"]
    print(f"drive_link={link}")


if __name__ == "__main__":
    main()
