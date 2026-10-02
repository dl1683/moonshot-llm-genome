# -*- coding: utf-8 -*-
"""Fold e210: NOTES, T180, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e210 — the same-episode margin: SAME-EPISODE-BREAKS — the class line dies at ANY episode; the hinge is one clock; the residue: the line was about the STEP SIZE relative to the noise ball, not the margin alone (2026-10-02 ~14:40Z) — DONE

WHAT WE DID: the repair fork as a desk cell (ZERO fresh compute —
every number loaded committed; same-episode identity gated per
row); the within-episode table n=7 (three margins against seven
committed first-episode clocks at their own pristine roots).

WHAT WE SAW (T180): SAME-EPISODE-BREAKS — W5 (the f2 half-step
counterfactual clock: margin 0.616x < 1, SURVIVED 4 steps) violates
even within the honest episode; the 2x class line is DEAD AT ANY
EPISODE; THE MARGIN IS A DESCRIPTOR, FULL STOP. THE HINGE, on the
face of the adjudication: drop W5 (e208's frozen fork convention)
and the n=6 NATURAL-clock table separates perfectly — the e208
finding "restored" as a NATURAL-STEP claim only. THE RESIDUE, THE
REAL LESSON: at half speed the same below-noise organism crosses
four steps of the same noise ball — THE LINE WAS ALWAYS ABOUT THE
STEP SIZE RELATIVE TO THE NOISE BALL, not the margin alone: the
margin, the step, and the ball are one object (the fact's
signal-to-noise per step), and no two of them predict survival
without the third. Context: Spearman 0.124; W3's step-1 read 1.20x
the bar (the survivor side noise-thin). HONESTY: three margins not
seven; W2's death bracketed; W6's margin conservative; the fork on
the face.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T179 —"
card = """## T180 — e210: the margin, the step, and the ball are one object (2026-10-02 ~14:40Z)

The repair cell ends the scalar's candidacy cleanly: the class
line breaks even within the honest episode (one counterfactual
clock suffices), and the anatomy of the break delivers the real
lesson — AT HALF SPEED THE SAME BELOW-NOISE ORGANISM CROSSES FOUR
STEPS OF THE SAME NOISE BALL. The line was never about the margin
alone: THE MARGIN, THE STEP, AND THE BALL ARE ONE OBJECT — the
fact's signal-to-noise PER STEP — and survival prediction needs
all three. THE SCALAR'S FINAL FORM: the margin is a descriptor
(the per-organism noise-unit reading); the class claim survives
only as a natural-step statement (drop the counterfactual clock
and n=6 separates perfectly — disclosed, the fork on the face).
THE PROGRAM NOTE: the scalar was minted (e208), extended (e209),
and killed (e210) in three cells and one day — the census
discipline working at full speed on its own objects.

"""
assert anchor in t and "## T180" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e210 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e210 | THE SAME-EPISODE MARGIN | DONE 14:40Z (T180: SAME-EPISODE-BREAKS — the class line dead at any episode; the hinge one clock; THE MARGIN, THE STEP, AND THE BALL ARE ONE OBJECT — the fact's signal-to-noise PER STEP; the class claim survives only as a natural-step statement) |", 1)
i = q.index("\n", q.index("| e210 |")) + 1
row2 = ("| e211 | THE WALLED-BAND QUESTION (e209's free find: the noise ball GROWS 2-3x under the wall — why? the "
        "walled history's own property) | DISPATCHED 14:41Z — CPU eval |\n")
q = q[:i] + row2 + q[j:] if False else q[:i] + row2 + q[i:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T14:41:00Z"
st["current_experiment"] = ("e210 FOLDED (T180: the margin, the step, and the ball are one object). Fleet: g1c-root "
                            "(GPU) + e211 DISPATCHED (CPU: the walled-band question)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e210 folded")
