"""Mint a Drive-only refresh token for shadow_gasp, used to upload final.mp4
so the Telegram bot can send a real Drive link instead of the file itself
(Telegram bots cap file uploads at 50MB; these renders run ~100MB).

Separate from _youtube_auth_remint.py's token on purpose: Google rejects a
single consent request that combines drive.file with the YouTube scopes
("This request contains scopes that cannot be requested together"), so this
needs its own independent OAuth flow/token, using the same OAuth client.

Run this ONCE, signed in as the same Google account already used for YouTube
uploads (so the video lands in that account's real Drive storage).

    python _drive_auth_mint.py client_secret.json

Writes the refresh token to drive_refresh_token.txt (never commit this file)
instead of printing it, so it doesn't leak into terminal scrollback/logs.
"""
import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/drive.file",  # only files this app creates, not the whole Drive
]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive_refresh_token.txt")


def main():
    if len(sys.argv) != 2:
        print('Usage: python _drive_auth_mint.py "<client_secret....json>"')
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(sys.argv[1], SCOPES)
    creds = flow.run_local_server(port=0)

    if not creds.refresh_token:
        print(
            "No refresh token returned. Google only issues one on FIRST consent for an app+scope set.\n"
            "Revoke this app's access at https://myaccount.google.com/permissions "
            "(as the shadow_gasp channel account), then run this again.",
            file=sys.stderr,
        )
        sys.exit(2)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(creds.refresh_token)

    print(f"\nOK -- new Drive refresh token written to:\n  {OUT}")
    print("\nGranted scopes:")
    for s in (creds.scopes or []):
        print(f"  {s}")
    print("\nNow tell Claude it's done; it will read this file and delete it.")


if __name__ == "__main__":
    main()
