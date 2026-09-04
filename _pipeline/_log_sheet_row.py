#!/usr/bin/env python3
"""Append a day's row to the Batch sheet, for days that did not log their own.

⚠️ TWO JOBS CREATE DAYS AND ONLY ONE WROTE TO THE SHEET.

`_batch_pregen.py` reserves a day, generates it, and appends a sheet row. The daily short
workflow (`pipeline.yml`) also reserves a day -- writing `sheet_logged: False` into state.json
at reservation -- generates it, and then never logs anything. Nothing ever came back for it.

The result was invisible because the sheet still filled up whenever pregen ran: rows exist for
days 1-32 (2026-08-01, one pregen run) and then stop. Days 33-57 were all made by the daily job
and have no row at all, so the sheet silently drifted from being the production tracker into
being a record of one August afternoon.

Usage:
    python3 _log_sheet_row.py 59          # one day
    python3 _log_sheet_row.py 33-57       # a range, for backfilling
    python3 _log_sheet_row.py 33-57 --dry-run

Idempotent by state.json's `sheet_logged` flag, so re-running cannot double-log a day. Reads
the title from the day's meta.json and points at the shot-1 still already committed for it.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _batch_pregen as bp   # noqa: E402  -- reuse SHEET_ID, append_sheet_row, commit_state


def days_from(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", help="day number, or a range like 33-57")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Read-only view of what is actually in the sheet. Needed because two different
    # conventions decide where a day's row lives (see the note in the module docstring),
    # and neither can be checked from state.json.
    if args.days == "dump":
        svc = bp.get_sheets_service()
        vals = svc.spreadsheets().values().get(
            spreadsheetId=bp.SHEET_ID, range=f"{bp.SHEET_TAB}!A1:M80").execute().get("values", [])
        print(f"{len(vals)} sheet rows (A1:M80)")
        for i, row in enumerate(vals, 1):
            cells = (row + [""] * 13)[:13]
            print(f"  row{i:>3} | A={cells[0]!r:>6} B={cells[1][:34]!r:36} I={cells[8][:12]!r:14} L={cells[11][:12]!r}")
        return

    state = bp.load_state()
    sheets = None if args.dry_run else bp.get_sheets_service()

    logged = skipped = 0
    for n in days_from(args.days):
        key = str(n)
        day = state["days"].get(key)
        if not day:
            print(f"day {n}: not in state.json, skipping")
            skipped += 1
            continue
        if day.get("sheet_logged"):
            print(f"day {n}: already logged, skipping")
            skipped += 1
            continue

        day_dir = os.path.join(bp.BATCH_DIR, f"day{n:02d}")
        meta_path = os.path.join(day_dir, "meta.json")
        # The title is the day's own working title. Falling back to the case keeps a row
        # useful rather than blank when a day predates meta.json or lost it.
        title = day["case"]
        if os.path.exists(meta_path):
            try:
                title = json.load(open(meta_path, encoding="utf-8")).get("title_working") or title
            except Exception as e:
                print(f"day {n}: could not read meta.json ({e}), using the case as the title")

        rel = os.path.relpath(day_dir, os.path.dirname(bp.PIPELINE_DIR)).replace("\\", "/")
        shot1 = ""
        if os.path.exists(os.path.join(day_dir, "shot1.jpeg")):
            shot1 = (f"https://raw.githubusercontent.com/{bp.GITHUB_REPO}/"
                     f"{bp.GITHUB_BRANCH}/{rel}/shot1.jpeg")

        print(f"day {n}: {day['case'][:52]!r} | title={title[:40]!r} | still={'yes' if shot1 else 'MISSING'}")
        if args.dry_run:
            logged += 1
            continue

        bp.append_sheet_row(sheets, n, day["case"], title, shot1, day.get("angle", ""))
        day["sheet_logged"] = True
        state["days"][key] = day
        bp.save_state(state)
        bp.commit_state(f"day {n:02d} sheet row logged (backfill)")
        logged += 1

    print(f"\n{'would log' if args.dry_run else 'logged'} {logged} row(s), skipped {skipped}")


if __name__ == "__main__":
    main()
