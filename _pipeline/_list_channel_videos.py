"""One-off: list every video on the shadow_gasp YouTube channel (any privacy
status) via the uploads playlist, so we have ground truth independent of the
pipeline's own ledger. Prints id, privacyStatus, publishedAt, title as TSV to
stdout (captured from the Actions log)."""
import os
import sys

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

creds = Credentials(
    token=None,
    refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
    client_id=os.environ["YOUTUBE_CLIENT_ID"],
    client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
    token_uri="https://oauth2.googleapis.com/token",
    scopes=["https://www.googleapis.com/auth/youtube.readonly"],
)
yt = build("youtube", "v3", credentials=creds)

ch = yt.channels().list(part="contentDetails,snippet,statistics", mine=True).execute()
item = ch["items"][0]
uploads_playlist = item["contentDetails"]["relatedPlaylists"]["uploads"]
print(f"CHANNEL\t{item['snippet']['title']}\tsubs={item['statistics'].get('subscriberCount')}\tvideos={item['statistics'].get('videoCount')}", file=sys.stderr)

video_ids = []
page_token = None
while True:
    resp = yt.playlistItems().list(
        part="contentDetails,snippet",
        playlistId=uploads_playlist,
        maxResults=50,
        pageToken=page_token,
    ).execute()
    for it in resp["items"]:
        video_ids.append(it["contentDetails"]["videoId"])
    page_token = resp.get("nextPageToken")
    if not page_token:
        break

# videos.list gives privacyStatus + real publishedAt in batches of 50
for i in range(0, len(video_ids), 50):
    chunk = video_ids[i : i + 50]
    resp = yt.videos().list(part="snippet,status", id=",".join(chunk)).execute()
    for v in resp["items"]:
        print(
            f"{v['id']}\t{v['status']['privacyStatus']}\t{v['snippet']['publishedAt']}\t{v['snippet']['title']}"
        )

print(f"TOTAL\t{len(video_ids)}", file=sys.stderr)
