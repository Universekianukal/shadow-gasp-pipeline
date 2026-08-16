"""Real-time stamp: called by crosspost_decision.yml right after a Facebook
or Instagram approve/reject decision, so the sheet reflects the decision the
moment it happens rather than waiting on a poll (FB/IG have no equivalent
"list everything" cron the way YouTube does -- this immediate stamp is the
only source for these two columns).

Env: DAY (e.g. "41"), PLATFORM ("fb" or "ig"), DECISION ("approve" or "reject").
"""
import os
import sys

from google.oauth2 import service_account
from googleapiclient.discovery import build

SA_KEY_PATH = os.path.join("_local", "sheets_sa_key.json")
SHEET_ID = "1aPoXPKlC9cCStUqULzR46FmvUaL8jxQbFsDWEDXn3jM"
SHEET_TAB = "Batch"

COLUMN = {"fb": "J", "ig": "K"}


def main():
    day = int(os.environ["DAY"])
    platform = os.environ["PLATFORM"].strip().lower()
    decision = os.environ["DECISION"].strip().lower()
    row = day + 1
    col = COLUMN[platform]

    value = "Yes" if decision == "approve" else "Rejected"

    creds = service_account.Credentials.from_service_account_file(
        SA_KEY_PATH, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    sheets = build("sheets", "v4", credentials=creds).spreadsheets().values()
    sheets.update(
        spreadsheetId=SHEET_ID,
        range=f"{SHEET_TAB}!{col}{row}",
        valueInputOption="RAW",
        body={"values": [[value]]},
    ).execute()
    print(f"day {day}: {platform} column stamped {value}", file=sys.stderr)


if __name__ == "__main__":
    main()
