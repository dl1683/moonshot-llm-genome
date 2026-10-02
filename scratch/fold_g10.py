# -*- coding: utf-8 -*-
"""Fold g10: NOTES, T176, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g10 — the first-step-aware wall: FIX-IMPOTENT, cleanly — the kill is the first step's SIZE vs the radius, not its timing (the isomorphism exact); WALL-FADES stands confirmed; the continuity dial named (2026-10-02 ~12:45Z) — DONE

WHAT WE DID: T170's three named fixes (STEP-CLIP / ANCHOR-AT-ONE /
DELTA-PROJECTION) on the loaded peak root, the g1bS6 record
drift-guarded; all gates PASS; the envelope audited every poll.

WHAT WE SAW (T176): ALL THREE VARIANTS BREACHED AT +1 EXACTLY LIKE
THE ORIGINAL W1 (0.0940 / 0.0007 / 0.0940 vs the bar 0.7546) —
FIX-IMPOTENT. THE STRUCTURAL FINDING: THE KILL IS THE SIZE OF THE
FIRST AdamW STEP (3.16 raw) VS THE 1x RADIUS (1.34 raw), NOT ITS
TIMING — (1) F1's clip produced THE IDENTICAL +1 READING as the
wall's own settled projection (the isomorphism EXACT: F1-vs-W1
deltas 0.0/0.0/0.0/-0.0 at +1/+2/+4/+10 — clipping the step to
the rung lands the fact where the projection did); (2) F2 anchored
at its own DEAD theta_1 (F2@+1 == C@+1 to 0.0; a ~0.3 shadow
re-formation in the dead ball); (3) F3's per-step trust region is
a SLOWER WASH (the walk unbounded to 32 raw; fact dead by +4; CE
0.726 < C's 0.788). WALL-FADES STANDS CONFIRMED: the wall's honest
final form — A DISPLACEMENT BUDGET THAT SEPARATES MEMORY FROM
DEATH, with every-checkpoint continuity IMPOSSIBLE while one step
exceeds the radius. THE DIAL THAT COULD BUY CONTINUITY (named,
not run): the FIRST STEP'S LR (or a rung >= one step) — the
g-series' next question. HONESTY: n=1 root/wash/variant; the
C/W1 legs loaded from g1bS6's committed metrics (the drift guard
caught a transcription typo pre-run).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T175 —"
card = """## T176 — g10: size, not timing — the wall's limit confirmed by its own fix attempts (2026-10-02 ~12:45Z)

The fix cell closes the structural question in the cleanest
possible way: all three timing-based repairs breach identically to
the original, and the isomorphism (F1 == W1 to 0.0 through +10)
PROVES the equivalence — clipping the first step to the rung and
projecting after the full step land the fact at the same point:
the projection already IS a clip at the rung scale. THE MECHANISM,
FINAL FORM: the +1 kill is arithmetic (a 3.16-raw step vs a
1.34-raw ball; the fact lands at the ball's edge whatever you do
about scheduling); anchoring later anchors into a dead state; a
per-step trust region is just a smaller wash-lr. THE WALL'S
10M LEDGER, CLOSED: a displacement budget separating memory from
death (~1000x), continuity impossible while step > radius, the
dial that could buy continuity being the step's lr or a
step-scaled rung — one named cell away if ever wanted. THE
PROGRAM NOTE: this is the third structural limit found by trying
to fix it and failing cleanly (the projection IS the clip; the
recipe stack IS scale-bound; the dose IS a window) — the failed
fix as an instrument.

"""
assert anchor in t and "## T176" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| g10 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| g10 | THE FIRST-STEP-AWARE WALL | DONE 12:45Z (T176: FIX-IMPOTENT — the kill is the step's SIZE vs the radius, not timing; the F1==W1 isomorphism exact; F2 anchors dead; F3 is a slower wash; WALL-FADES confirmed; the continuity dial named: the first step's lr or a step-scaled rung) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T12:46:00Z"
st["current_experiment"] = ("g10 FOLDED (T176: size, not timing — WALL-FADES confirmed; the failed fix as an "
                            "instrument). Fleet: g1bS8 (GPU, the sixth take) + e208 (CPU, the margin census)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g10 folded")
