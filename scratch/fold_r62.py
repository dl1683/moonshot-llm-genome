# -*- coding: utf-8 -*-
"""Fold R62: the auditor's repairs applied + the REVIEWS entry + stamps."""
import io, json

# --- the repairs ---
# C6 ledger fix
c = "scratch/claims_ledger.md"
s = io.open(c, encoding="utf-8").read()
old = "licensed n=3 wash-draws x n=2 root draws (g1c-root: ROOT-WALL-HOLDS; the anatomy differs, the protection identical — the wall guards the function, not the wiring)"
if old in s:
    s = s.replace(old, "licensed n=3 wash x n=2 root (g1c-root HOLDS; the anatomy differs, the protection identical); 10M: WALL-FADES draw-clean (g1bS8 at 0.9351: every rung breaches +1, the blindness arithmetic per g10's FIX-IMPOTENT isomorphism; direction order-10^3 at >=+10; the flat phase root-independent, retention mins 0.895->0.790; tax +0.10); the base axis UNTESTED (g1d: the base lottery owns the expression channel — TEXTURE, unadjudicated; the wall guarded the half-expressed fact at 1.35x its root)", 1)
    io.open(c, "w", encoding="utf-8").write(s)
    print("C6 fixed")

# QUEUE rows
q = io.open("QUEUE.md", encoding="utf-8").read()
adds = []
if "| g1bS7 |" not in q: adds.append("| g1bS7 | THE REDRAWN INTERIOR DOSE | DONE 10:55Z Oct-2 (T172: PEAK-LOTTERY; the 0.94 root saved) |")
if "| g1bS8 |" not in q: adds.append("| g1bS8 | THE SIXTH TAKE | DONE 13:15Z Oct-2 (T178: WALL-FADES draw-clean; the flat phase root-independent) |")
if "| g1d |" not in q: adds.append("| g1d | THE BASE-SEED REDRAW | DONE 14:35Z Oct-2 (T186: TEXTURE — the base lottery owns the expression channel; the ratio-device record) |")
if adds:
    j = q.index("\n", q.index("| g1c-root |")) + 1
    q = q[:j] + "\n".join(adds) + "\n" + q[j:]
q = q.replace("the g1bR roots' bands are 2-3x wider", "the walled bands [RETIRED by e211/e212: the instrument shadow] read 2-3x wider") if "2-3x wider" in q else q
io.open("QUEUE.md", "w", encoding="utf-8").write(q)
print("QUEUE rows added")

# Wording fixes (order-10^3; the retention mins; W028's clause)
t = io.open("THINKING.md", encoding="utf-8").read()
t = t.replace("~1000x at every checkpoint >= +10", "order-10^3 at every checkpoint >= +10 (276x-59,000x)")
t = t.replace("separates the fact from the control by ~1000x", "separates the fact from the control at order 10^3")
t = t.replace("retention fell 0.96x -> 0.79x", "retention mins fell 0.895 -> 0.790")
t = t.replace("retention FELL 0.96x->0.79x", "retention mins FELL 0.895->0.790")
t = t.replace("protection replicates across wash x root x base x scale", "protection replicates strict at wash x root (direction/flat-phase form at 10M; the base axis untested per g1d)")
io.open("THINKING.md", "w", encoding="utf-8").write(t)
n = io.open("NOTES.md", encoding="utf-8").read()
n = n.replace("separates the fact from the control by ~1000x", "separates the fact from the control at order 10^3 (276x-59,000x)")
io.open("NOTES.md", "w", encoding="utf-8").write(n)
print("wording fixed")

# --- REVIEWS entry ---
rv = io.open("REVIEWS.md", encoding="utf-8").read()
sep = "---\n\n---"
entry = """---

## R62 — the densest arc audited: 31/34 exact, no verdict changes, the debt bookkeeping (2026-10-02, folded ~14:45Z)

Trigger: the review clock >5h; the arc T172-T186 (fifteen cards).
AUDITOR (scratch/r62_auditor.md; a pure recomputation pass): 31 of
34 headline quantities reproduce EXACTLY — the lottery numbers,
both 10M wall tables, g10's isomorphism (deltas 5.2e-8..1.8e-7),
g1c-root's, e205's desk-forced arithmetic (5 decimals), the full
margin/band chain, the 124M cross-wash find (0.99777) — NO verdict
changes, NO bar shopping. THREE numeric misquotations (all robust
to correction, all repaired this fold): "~1000x" -> "order 10^3
(276x-59,000x)"; the retention pair -> the committed mins
0.895->0.790; W028's wall clause conflated strict protection with
its 10M direction-form and the untested base axis -> scoped. ONE
STALE LEDGER ROW (C6: g1c-root's "queued" fragment beside its own
result; g1bS8/g10 never folded in) -> repaired with the full
grid. THREE MISSING QUEUE rows (g1bS7/S8/g1d) -> added; e209's
retired free find -> marked. ONE HARVEST CAUGHT (g1d's complete
record was ahead of the lab's git memory — folded as T186 before
the audit landed, the race disclosed). THE BOTTOM LINE: the arc's
arithmetic is clean; the discipline held through its densest day;
the debts were bookkeeping and are paid."""
assert sep in rv
rv = rv.replace(sep, "---\n\n" + entry + "\n\n---", 1)
io.open("REVIEWS.md", "w", encoding="utf-8").write(rv)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T14:46:00Z"
st["last_review"] = "2026-10-02T14:45:00Z"
st["current_experiment"] = ("R62 CLOSED (31/34 exact, no verdict changes; the bookkeeping paid). Fleet 0 — the "
                            "next cells named: the third wash draw, the cons-seed redraw, the conservation-of-rank."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("R62 closed")
