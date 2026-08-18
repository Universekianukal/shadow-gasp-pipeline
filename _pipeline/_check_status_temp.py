import os, json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

creds = Credentials(
    token=None,
    refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
    client_id=os.environ["YOUTUBE_CLIENT_ID"],
    client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
    token_uri="https://oauth2.googleapis.com/token",
    scopes=["https://www.googleapis.com/auth/youtube"],
)
yt = build("youtube", "v3", credentials=creds)
ids = ["HxkKGV7ya48", "gZiPNWP6F88"]
resp = yt.videos().list(part="status,snippet", id=",".join(ids)).execute()
for item in resp.get("items", []):
    print(item["id"], "|", item["snippet"]["title"][:60], "|", item["status"]["privacyStatus"], "|", item["status"].get("publishAt"))
