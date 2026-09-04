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


⚠️ ROWS ARE POSITIONAL: day N lives on sheet row N+1.

That is not a style choice, it is what _sync_youtube_status.py assumes when it writes the video
columns (`row = day + 1`). append_sheet_row used values().append instead, which puts a row at the
BOTTOM in write order. The two agree only while days are appended in strict order into an empty
sheet -- true for days 1-32, which is why nobody noticed. The moment days arrive out of order, or
are backfilled later, the conventions diverge: the sync writes a day's video ID at day+1 while its
label row sits somewhere near the bottom, and the sheet grows duplicates (days 39 and 41 each
appeared twice) and holes (days 40 and 42 had no positional row at all).

So this writes with values().update at an explicit row, and never appends.

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



def read_sheet(svc):
    return svc.spreadsheets().values().get(
        spreadsheetId=bp.SHEET_ID, range=f"{bp.SHEET_TAB}!A1:M200"
    ).execute().get("values", [])


def row_for(day_num, state, cell_cache):
    """The full A..M row a day should have: regenerated labels, existing status preserved.

    Columns I..M (YT status, FB/IG posted, video id, published at) are written by other jobs and
    must never be clobbered by a relabel, so they are carried over from whichever existing row
    for this day holds the most data.
    """
    day = state["days"][str(day_num)]
    base = cell_cache.get(day_num, [""] * 13)
    day_dir = os.path.join(bp.BATCH_DIR, f"day{day_num:02d}")
    title = day["case"]
    meta_path = os.path.join(day_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            title = json.load(open(meta_path, encoding="utf-8")).get("title_working") or title
        except Exception:
            pass
    rel = os.path.relpath(day_dir, os.path.dirname(bp.PIPELINE_DIR)).replace("\\", "/")
    shot1 = ""
    if os.path.exists(os.path.join(day_dir, "shot1.jpeg")):
        shot1 = f"https://raw.githubusercontent.com/{bp.GITHUB_REPO}/{bp.GITHUB_BRANCH}/{rel}/shot1.jpeg"

    out = list(base) + [""] * (13 - len(base))
    out[0] = day_num                                   # A day
    out[1] = day["case"]                               # B case  (relabelled)
    out[2] = out[2] or "Images done"                   # C
    out[3] = title                                     # D title (relabelled)
    out[4] = shot1 or out[4]                           # E still
    out[5] = out[5] or "Pending"                       # F
    out[7] = out[7] or day.get("angle", "")            # H notes
    return out[:13]


def repair(args):
    """One row per day, in its positional slot, with nothing lost and nothing duplicated."""
    state = bp.load_state()
    svc = bp.get_sheets_service()
    rows = read_sheet(svc)

    # Harvest the richest existing row per day, so status columns survive the rewrite.
    cache = {}
    for r in rows[1:]:
        cells = (r + [""] * 13)[:13]
        try:
            d = int(str(cells[0]).strip())
        except (ValueError, TypeError):
            continue
        if len([c for c in cells if c]) > len([c for c in cache.get(d, []) if c]):
            cache[d] = cells

    days = sorted(int(k) for k in state["days"])
    last = days[-1]
    print(f"{len(rows)} rows on the sheet; {len(days)} days; positional block will be rows 2-{last + 1}")
    dupes = [d for d in days if sum(1 for r in rows[1:] if str((r + [''])[0]).strip() == str(d)) > 1]
    print(f"days currently duplicated: {dupes or 'none'}")
    print(f"rows below the block to clear: {max(0, len(rows) - (last + 1))}")

    data = []
    for d in days:
        want = row_for(d, state, cache)
        cur = cache.get(d)
        moved = "" if cur and str((rows[d] + [''] * 2)[0] if len(rows) > d else '') == str(d) else "  <-- moves"
        if args.dry_run and d >= last - 6:
            print(f"  row{d + 1:>3} <- day {d:<3} {str(want[1])[:40]:42} vid={want[11] or '-'}{moved}")
        data.append({"range": f"{bp.SHEET_TAB}!A{d + 1}:M{d + 1}", "values": [want]})

    if len(rows) > last + 1:
        data.append({"range": f"{bp.SHEET_TAB}!A{last + 2}:M{len(rows)}",
                     "values": [[""] * 13 for _ in range(len(rows) - (last + 1))]})

    if args.dry_run:
        print(f"\nwould write {len(days)} positional rows and blank {max(0, len(rows) - (last + 1))} stray rows")
        return
    svc.spreadsheets().values().batchUpdate(
        spreadsheetId=bp.SHEET_ID,
        body={"valueInputOption": "USER_ENTERED", "data": data},
    ).execute()
    print(f"rewrote {len(days)} rows positionally and cleared {max(0, len(rows) - (last + 1))} stray rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", help="day number, or a range like 33-57")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Read-only view of what is actually in the sheet. Needed because two different
    # conventions decide where a day's row lives (see the note in the module docstring),
    # and neither can be checked from state.json.
    if args.days == "repair":
        repair(args)
        return

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

        # POSITIONAL, never append -- see the module docstring.
        rows = read_sheet(sheets)
        cache = {}
        for r in rows[1:]:
            cells = (r + [""] * 13)[:13]
            try:
                cache.setdefault(int(str(cells[0]).strip()), cells)
            except (ValueError, TypeError):
                pass
        sheets.spreadsheets().values().update(
            spreadsheetId=bp.SHEET_ID, range=f"{bp.SHEET_TAB}!A{n + 1}:M{n + 1}",
            valueInputOption="USER_ENTERED",
            body={"values": [row_for(n, state, cache)]},
        ).execute()
        day["sheet_logged"] = True
        state["days"][key] = day
        bp.save_state(state)
        bp.commit_state(f"day {n:02d} sheet row logged (backfill)")
        logged += 1

    print(f"\n{'would log' if args.dry_run else 'logged'} {logged} row(s), skipped {skipped}")


if __name__ == "__main__":
    main()
