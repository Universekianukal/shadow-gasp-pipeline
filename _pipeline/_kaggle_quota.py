"""Report remaining Kaggle GPU quota for every account slot we can spend.

Why this exists: an account that is out of weekly GPU quota does not make
Kaggle return an error. `kernels push` is accepted, and the kernel then sits
QUEUED forever. Day 61 hung 24h on exactly that, was killed by GitHub's job
cap, reported as `cancelled` (so the if: failure() notifier stayed silent),
and the day was stranded. The quota was knowable the whole time -- nothing
had asked.

`GET /api/v1/kernels/quota` is undocumented but real, and returns seconds
used / allowed for GPU and TPU plus the refresh time. It is scoped to the
token's own account, so each slot has to be asked separately.

Usage:
    python3 _kaggle_quota.py                     # report every slot
    python3 _kaggle_quota.py --require VIDEO     # ...and fail if VIDEO is short
    python3 _kaggle_quota.py --require VIDEO --min-hours 1.5

Slots come from KAGGLE_ACCOUNTS ("IMAGE:anuragmishra108,VIDEO:kianukal"),
tokens from KAGGLE_<SLOT>_API_TOKEN. A slot with no token is reported as
unconfigured rather than skipped silently.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

QUOTA_URL = "https://www.kaggle.com/api/v1/kernels/quota"

# Kaggle sits behind Cloudflare, and urllib's default User-Agent
# ("Python-urllib/3.x") is refused at the edge with a 1010 before the request
# ever reaches an origin -- the same trap that made three other projects in
# this repo family look like they had empty logs and working curl calls.
UA = "shadow-gasp-pipeline/1.0 (+quota-check)"


def parse_accounts(raw):
    """"IMAGE:anuragmishra108,VIDEO:kianukal" -> [("IMAGE", "anuragmishra108"), ...]"""
    out = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        slot, handle = chunk.split(":", 1)
        out.append((slot.strip(), handle.strip()))
    return out


def fetch_quota(token):
    req = urllib.request.Request(
        QUOTA_URL,
        headers={"Authorization": f"Bearer {token}", "User-Agent": UA},
    )
    with urllib.request.urlopen(req, timeout=45) as r:
        # A 200 carrying HTML is Kaggle's signed-out page, not a quota reading.
        # Treat anything that is not JSON as a failure rather than as zero use.
        return json.loads(r.read().decode("utf-8"))


def secs(block, key):
    return float(block.get(key, {}).get("seconds", 0) or 0)


def hours(s):
    return s / 3600.0


def fmt_hm(h):
    return f"{int(h)}h{int(round((h - int(h)) * 60)):02d}m"


def read_slot(slot, handle):
    """-> dict with either the numbers, or an 'error' explaining why not."""
    token = os.environ.get(f"KAGGLE_{slot}_API_TOKEN", "").strip()
    if not token:
        return {"slot": slot, "handle": handle,
                "error": f"no KAGGLE_{slot}_API_TOKEN set"}
    try:
        d = fetch_quota(token)
    except urllib.error.HTTPError as e:
        return {"slot": slot, "handle": handle, "error": f"HTTP {e.code}"}
    except Exception as e:  # network, JSON, anything
        return {"slot": slot, "handle": handle, "error": f"{type(e).__name__}: {e}"}

    gpu = d.get("gpuQuota", {})
    used = secs(gpu, "timeUsed")
    reserved = secs(gpu, "timeReserved")
    total = secs(gpu, "totalTimeAllowed")
    # timeReserved is GPU already committed to a running/queued kernel. Ignoring
    # it would let two builds each be told there is room for one.
    left = max(0.0, total - used - reserved)
    return {"slot": slot, "handle": handle,
            "used_h": hours(used), "total_h": hours(total),
            "left_h": hours(left), "reserved_h": hours(reserved),
            "refresh": d.get("quotaRefreshTime", "?")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require", default="",
                    help="slot that must have quota; exit 1 if it does not")
    ap.add_argument("--min-hours", type=float, default=1.0,
                    help="hours the --require slot needs free (default 1.0; a "
                         "16-still PixArt run costs ~0.5h, so 1.0 leaves margin)")
    args = ap.parse_args()

    accounts = parse_accounts(os.environ.get("KAGGLE_ACCOUNTS", ""))
    if not accounts:
        print("KAGGLE_ACCOUNTS is empty -- nothing to check", file=sys.stderr)
        return 2

    rows = [read_slot(slot, handle) for slot, handle in accounts]

    lines = ["Kaggle GPU quota"]
    for r in rows:
        if "error" in r:
            lines.append(f"  {r['slot']:<9} {r['handle']:<16} -- {r['error']}")
            continue
        bar_left = fmt_hm(r["left_h"])
        mark = "OK " if r["left_h"] >= args.min_hours else "LOW"
        extra = f", {fmt_hm(r['reserved_h'])} reserved" if r["reserved_h"] > 0.01 else ""
        lines.append(
            f"  {mark} {r['slot']:<9} {r['handle']:<16} {bar_left} left "
            f"of {fmt_hm(r['total_h'])}{extra}   (resets {r['refresh'][:10]})"
        )
    report = "\n".join(lines)
    print(report)

    if args.require:
        want = next((r for r in rows if r["slot"] == args.require), None)
        if want is None:
            print(f"\n::error::slot '{args.require}' is not in KAGGLE_ACCOUNTS",
                  file=sys.stderr)
            return 1
        if "error" in want:
            # Unreadable is not the same as empty. Say so instead of guessing,
            # but do not block the build on a transient network blip either.
            print(f"\n::warning::could not read quota for {args.require} "
                  f"({want['error']}) -- proceeding without the check",
                  file=sys.stderr)
            return 0
        if want["left_h"] < args.min_hours:
            print(
                f"\n::error::Kaggle slot {args.require} ({want['handle']}) has only "
                f"{fmt_hm(want['left_h'])} GPU left, under the {args.min_hours}h this "
                f"job needs. Kaggle would accept the kernel and queue it forever "
                f"rather than refuse it. Quota resets {want['refresh'][:10]}; "
                f"retry on another slot before then.",
                file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
