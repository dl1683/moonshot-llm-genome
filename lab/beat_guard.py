# -*- coding: utf-8 -*-
"""beat_guard.py — the anti-stall tripwire (user directive 2026-10-02: "never again").

Machine-checks the two failure modes of the 2026-10-02 stall:
  1. THINKING-DUE: the thinking lane has gone quiet — no new W-card
     or substantive synthesis commit in ~90 min of wall-clock lab
     work while experiment folds continued.
  2. TREADMILL-ALERT: the last K dispatches are all successor-chain
     cells (same lineage prefix), i.e., follow-on churn instead of
     fresh questions.

Run:  python lab/beat_guard.py [--hours H]
Reads only git history + QUEUE.md; writes nothing (the heartbeat
logs the verdict in STATE.json). Exit code 0 = OK, 1 = alert.
"""
import subprocess, sys, re, datetime, json

HOURS = 6  # look-back window for the balance check
FOLD_MIN_FOR_DUE = 3   # >= this many folds with no thinking => THINKING-DUE
TREADMILL_K = 4        # >= this many consecutive same-lineage dispatches => TREADMILL-ALERT


def git(args):
    out = subprocess.run(["git"] + args, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    return out.stdout or ""


def parse_commits():
    since = (datetime.datetime.now(datetime.timezone.utc)
             - datetime.timedelta(hours=HOURS)).strftime("%Y-%m-%dT%H:%M")
    log = git(["log", f"--since={since}", "--pretty=%H%x09%ad%x09%s",
               "--date=short"]).splitlines()
    folds, thinking = 0, 0
    last_thinking = None
    for line in log:
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        subj = parts[2]
        # experiment fold: a T-card fold commit or a cell DONE commit
        if re.search(r"\bT1\d\d folded|T1\d\d folded\b", subj) or re.search(
                r"\bfolded:", subj) or re.search(r"\bFOLDED\b", subj):
            folds += 1
        # thinking lane: a NEW wonder card / synthesis / fresh-questions commit
        # [R74-era fix 2026-10-10]: T-card folds (T2\d\d FOLDED) are thinking too —
        # the detector false-fired THINKING-DUE over T296/T297's window.
        if re.search(r"\bW0\d\d\b.*(wonder|:)", subj) or "WONDER" in subj or (
                "synthesis" in subj.lower() and "day" in subj.lower()) or \
           "fresh-question" in subj.lower() or re.match(r"^W0\d\d[ :]", subj) or \
           re.search(r"\bT2\d\d\b", subj):
            thinking += 1
            last_thinking = parts[1] + " " + subj[:60]
    return folds, thinking, last_thinking


def lineage_prefix(name):
    """g1bS7 -> g1bS, g14 -> g1, e223 -> e22, else the name itself."""
    m = re.match(r"^(g\d|[a-z]\d)([a-z]?)", name)
    if not m:
        return name
    return m.group(0)


def treadmill_depth():
    q = open("QUEUE.md", encoding="utf-8").read()
    rows = [ln for ln in q.splitlines() if "DISPATCHED" in ln]
    names = []
    for ln in rows:
        m = re.match(r"\|\s*([A-Za-z0-9_-]+)\s*\|", ln)
        if m:
            names.append(m.group(1))
    # walk from the newest dispatched backwards while the lineage matches
    if not names:
        return 0, ""
    newest = names[-1]
    pref = lineage_prefix(newest)
    depth = 0
    for nm in reversed(names):
        if lineage_prefix(nm).startswith(pref[:2]) and pref[:2] in lineage_prefix(nm):
            depth += 1
        else:
            break
    return depth, newest


def main():
    folds, thinking, last_thinking = parse_commits()
    depth, newest = treadmill_depth()
    verdicts = []
    if folds >= FOLD_MIN_FOR_DUE and thinking == 0:
        verdicts.append(
            f"THINKING-DUE: {folds} experiment folds in the last {HOURS}h with "
            f"ZERO new thinking commits (no W-card/synthesis). The beat's bulk is "
            f"thinking BEFORE any dispatch: write a wonder card, a synthesis section, "
            f"or a fresh-question note now.")
    if depth >= TREADMILL_K:
        verdicts.append(
            f"TREADMILL-ALERT: the last {depth} dispatched cells are one successor "
            f"chain (newest: {newest}). Break it: run a fresh-questions review or "
            f"assemble the record before dispatching another successor.")
    status = {
        "guard": "OK" if not verdicts else "ALERT",
        "window_hours": HOURS,
        "folds_in_window": folds,
        "thinking_commits_in_window": thinking,
        "successor_chain_depth": depth,
        "newest_dispatch": newest,
        "verdicts": verdicts,
    }
    print(json.dumps(status, indent=2))
    return 1 if verdicts else 0


if __name__ == "__main__":
    sys.exit(main())
