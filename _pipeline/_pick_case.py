"""Auto-pick the next true-crime case for shadow_gasp, avoiding anything the
channel has already published.

The dedup source is cases_used.json — the real published-video ledger
({"cases": [{"videoId", "case", "publishedAt"}, ...]}). Every already-used
case title is handed to the model as an exclusion list, and the returned pick
is re-checked locally by fuzzy match before being accepted, because the model
will occasionally reword an existing case rather than genuinely picking a new
one ("Zodiac Killer" vs "The Zodiac cipher murders").

Writes the chosen case to the CASE output so the workflow can pass it to
_gen_video_content.py. No-op if CASE is already set — a hand-picked case
always wins over the auto-picker.
"""
import difflib
import json
import os
import re
import sys

import _llm  # provider shim: Anthropic or Fireworks, see _pipeline/_llm.py

MODEL = "claude-sonnet-5"
LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases_used.json")
# Above this, treat the pick as a duplicate. Tuned deliberately loose: a false
# positive only costs one retry, while a false negative publishes a case the
# channel already covered. It does flag some genuine near-misses (e.g. "Robert
# Hanssen" vs "Robert Hansen" — different men, near-identical names), which is
# the accepted cost.
SIMILARITY_LIMIT = 0.72

SYS = """You pick cases for shadow_gasp, a true-crime YouTube Shorts channel.

Pick ONE case that would make a strong ~75-second short. Good picks:
- Genuinely unsolved, or solved in a way stranger than the mystery.
- Have a concrete, visual hook (a place, a vehicle, an object, a document).
- Have at least one hard verifiable detail (a date, an amount, a distance) —
  the format runs on escalating specifics.
- Are documented enough to be factually narratable, but not so over-covered
  that every viewer already knows the ending. Avoid the 5-6 cases everyone
  has seen a hundred shorts about unless there is a genuinely underexposed
  angle, and say so in the angle field.

Rotate subject matter — do not pick the same flavour of case as the most
recent entries in the exclusion list (e.g. if the last few are serial killers,
pick a disappearance, a heist, a maritime mystery, an institutional
cover-up, a forensic puzzle).

Return FIVE different candidates, best first, as JSON:
{
  "candidates": [
    {
      "case": "short canonical case name, including a year or place if it disambiguates",
      "angle": "one sentence on the specific hook/reversal that makes this work as a short",
      "why_now": "one sentence on why this is underexposed or freshly interesting"
    }
  ]
}
Five genuinely DIFFERENT cases, not five framings of one. Vary the category across
them (a disappearance, a heist, a maritime mystery, a forensic puzzle, an
institutional cover-up) so that if the first is already published the rest are
still usable.
Respond with ONLY the JSON object — no markdown code fences, no other text."""


# Escalated on truncation. A retry that repeats the failing parameter is not a retry.
# 32000 is a backstop, not a plan: day 58 already needed 16,000, so the ladder had no
# headroom left. PROMPT_EXCLUSIONS below is the actual fix; this just buys room to notice.
# 1024 removed 2026-09-07: glm-5p2 ALWAYS spends output budget on reasoning first, so
# the smallest rung returned empty every time and simply burned an attempt -- one of
# only four -- before the ladder could climb. Start where a reasoning model can answer.
BUDGETS = (4096, 16000, 32000)

# How many recent cases to show the model. The full history is still enforced after the
# pick by is_duplicate(); this bounds only the PROMPT, which is what was growing forever.
# Raised 60 -> 200 on 2026-09-07. 60 was chosen when the worry was a prompt growing
# without bound, but the whole ledger is only ~8KB / ~2k tokens today -- while the
# budget that actually gets exhausted is the OUTPUT one, spent on reasoning. Capping
# the INPUT bought nothing and created an 81-case blind zone that killed the picker.
# At 200 the model currently sees everything; past that the stratified sample below
# degrades gracefully instead of going blind on the early famous cases again.
PROMPT_EXCLUSIONS = 200


def extract_json(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return json.loads(t)


def norm(s):
    """Strip punctuation/filler so 'The Zodiac Killer (1969)' and 'Zodiac
    killer' compare as the same string."""
    s = re.sub(r"\(.*?\)", " ", s.lower())
    s = re.sub(r"\b(the|a|an|of|case|murders?|killer|disappearance|mystery)\b", " ", s)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def is_duplicate(pick, used):
    """Word ORDER varies freely between phrasings of the same case
    ("H.H. Holmes' Murder Castle" vs "Murder Castle of H. H. Holmes"), so a
    sequence ratio alone misses real duplicates. Token overlap catches those;
    the sequence ratio still catches near-spellings the token set would split
    ("skyjacking"/"hijacking")."""
    p = norm(pick)
    if not p:
        return None
    ptok = set(p.split())
    for u in used:
        n = norm(u)
        if not n:
            continue
        if p in n or n in p:
            return u
        ntok = set(n.split())
        shared = ptok & ntok
        overlap = len(shared) / min(len(ptok), len(ntok))
        if len(shared) >= 2 and overlap >= 0.6:
            return u
        if difflib.SequenceMatcher(None, p, n).ratio() >= SIMILARITY_LIMIT:
            return u
    return None


def load_used():
    if not os.path.exists(LEDGER_PATH):
        return []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        return [c["case"] for c in json.load(f).get("cases", [])]


def pick(client, used):
    # ⭐ SHOW THE MODEL A WINDOW, NOT THE WHOLE LEDGER.
    #
    # Every published case went into the prompt, and that list grows by one EVERY DAY. It is
    # what the reasoning budget is actually spent on: day 58 needed 16,000 tokens where 1,024
    # sufficed a few months ago, and 4,096 was not enough. A budget ladder cannot outrun a
    # prompt that grows without bound -- it only moves the failure later.
    #
    # Safe because the full history is still enforced, just AFTER the pick rather than inside
    # the prompt: is_duplicate() below checks the answer against every case ever used and
    # retries with the specific clash named. The window steers the model away from recent
    # territory; correctness never depended on it being complete.
    #
    # The tail is the useful end -- `used` runs oldest to newest, and a model asked for a
    # fresh case is likelier to collide with the last few weeks than with something from May.
    # THE WINDOW MUST NOT BE RECENCY-ONLY -- THAT IS BACKWARDS FOR THIS FAILURE.
    #
    # A pure tail showed the last 60 while is_duplicate() judged against all 141, so 81 cases
    # were landmines the model could not see. And the hidden ones are the WORST to hide: a
    # channel mines the famous cases first, so the blind zone fills with exactly the cases any
    # model reaches for. On 2026-09-06 the picker died proposing Lead Masks, Hinterkaifeck and
    # Isdal Woman -- ledger indexes 75, 4 and 74 of 141, every one outside the window.
    #
    # Same prompt size, better spend: keep most of it on recent territory (what the rotation
    # guidance needs) and spread the rest evenly over the whole history, so the early famous
    # picks are represented. The full ledger is still enforced after the pick.
    recent_n = min(len(used), PROMPT_EXCLUSIONS * 2 // 3)
    recent = used[-recent_n:] if recent_n else []
    older = used[:-recent_n] if recent_n else list(used)
    spread_n = PROMPT_EXCLUSIONS - recent_n
    spread = []
    if older and spread_n > 0:
        step = max(1, len(older) / float(spread_n))
        seen = set()
        for i in range(spread_n):
            c = older[min(len(older) - 1, int(i * step))]
            if c not in seen:
                seen.add(c)
                spread.append(c)
    shown = spread + recent
    exclusion = "\n".join(f"- {c}" for c in shown)
    msg = (
        f"The channel has published {len(used)} cases in total. Here are {len(shown)} of them "
        f"-- {len(spread)} sampled across the whole history and the {len(recent)} most recent. "
        f"Pick something genuinely different from ALL of these, and from anything closely "
        f"similar. Assume every famous case of this kind is already published even if it is "
        f"not listed:\n\n{exclusion}"
    )
    messages = [{"role": "user", "content": msg}]
    budget = BUDGETS[0]
    for attempt in range(4):
        resp = client.messages.create(
            model=MODEL, max_tokens=budget, system=SYS, messages=messages
        )
        # A response can come back with no text block at all (seen in practice
        # once the exclusion list got long) -- that used to crash the whole
        # batch run with an uncaught StopIteration instead of just retrying
        # like every other malformed-response case here does.
        raw = next((b.text for b in resp.content if b.type == "text"), None)

        # ⭐ TRUNCATION IS NOT A MALFORMED RESPONSE, AND MUST NOT BE RETRIED UNCHANGED.
        #
        # glm-5p2 is a reasoning model: it can spend an entire small budget thinking and return
        # EMPTY content with finish_reason="length". _llm.py reports that faithfully as
        # stop_reason="max_tokens" -- the same recoverable condition as Anthropic truncation --
        # but this loop only checked for an ABSENT text block, so an empty string fell through
        # to json.loads("") and was reported as "invalid JSON". Every retry then went out at the
        # same 1024 tokens, which cannot fix a budget problem: four guaranteed failures, an
        # error message pointing at a JSON bug that never existed, and the day queue silently
        # stopping at 57 (run 33780705411, 2026-09-03).
        #
        # 1024 was always thin for a reasoning model. The 32,000-token failure recorded in
        # _llm.py predates FIREWORKS_REASONING_EFFORT being capped to "low", so escalating from
        # here is the honest first move -- and if it is still empty at the ceiling, the log now
        # says exactly that instead of blaming the JSON.
        if resp.stop_reason == "max_tokens" or not (raw or "").strip():
            bigger = next((b for b in BUDGETS if b > budget), None)
            if bigger is None:
                print(f"attempt {attempt + 1}: STILL no content at the {budget:,}-token "
                      f"ceiling (stop_reason={resp.stop_reason}) -- the budget is not the "
                      f"problem; suspect the {len(used)}-case exclusion list in the prompt",
                      file=sys.stderr)
            else:
                print(f"attempt {attempt + 1}: model returned no content "
                      f"(stop_reason={resp.stop_reason}) -- reasoning consumed the "
                      f"{budget:,}-token budget; retrying with {bigger:,}", file=sys.stderr)
                budget = bigger
            # Deliberately does NOT append this exchange. There is nothing to correct, and an
            # empty assistant turn is junk context that makes the next attempt worse.
            continue
        try:
            d = extract_json(raw)
        except json.JSONDecodeError as e:
            print(f"attempt {attempt + 1}: invalid JSON ({e}), retrying", file=sys.stderr)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"That wasn't valid JSON ({e}). Return the full corrected JSON object only, no other text."})
            continue
        # Take the first CANDIDATE that is not already published.
        #
        # One call, several answers. Before this the model got exactly one shot per attempt and
        # four attempts total, so a single unlucky pick burned a whole round-trip -- and with
        # 141 cases published, unlucky is the normal case. Five candidates per call turns the
        # same four attempts into up to twenty chances, at no extra request.
        cands = d.get("candidates") if isinstance(d, dict) else None
        if not isinstance(cands, list) or not cands:
            cands = [d]                      # older single-object shape, still accepted
        clashes = []
        for c in cands:
            if not isinstance(c, dict) or not (c.get("case") or "").strip():
                continue
            clash = is_duplicate(c["case"], used)
            if not clash:
                if clashes:
                    print(f"attempt {attempt + 1}: skipped {len(clashes)} already-published "
                          f"candidate(s), taking '{c['case']}'", file=sys.stderr)
                return c
            clashes.append(f"'{c['case']}' duplicates '{clash}'")
        print(f"attempt {attempt + 1}: all {len(clashes)} candidate(s) already published "
              f"-- {'; '.join(clashes)}", file=sys.stderr)
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content":
            "Every one of those is already published: " + "; ".join(clashes) + ". "
            "Pick five genuinely different cases, in different categories, and avoid the "
            "well-known ones entirely -- this channel has already covered them. "
            "Return the full JSON."})
    raise SystemExit("could not find an unused case after 4 attempts")


def main():
    if os.environ.get("CASE", "").strip():
        case = os.environ["CASE"].strip()
        print(f"CASE already set, keeping it: {case}")
    else:
        used = load_used()
        d = pick(_llm.client(), used)
        case = d["case"]
        print(f"Picked: {case}\n  angle:   {d['angle']}\n  why now: {d['why_now']}")
        print(f"  (excluded {len(used)} already-published cases)")

    # Hand the pick to later workflow steps. GITHUB_ENV covers later steps;
    # GITHUB_OUTPUT is written too because env vars set via GITHUB_ENV are NOT
    # visible inside the step that set them, and downstream jobs can only read
    # a step output. Locally, neither is set and CASE is just printed.
    for var, line in (("GITHUB_ENV", f"CASE={case}"), ("GITHUB_OUTPUT", f"case={case}")):
        path = os.environ.get(var)
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
