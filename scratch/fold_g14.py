# -*- coding: utf-8 -*-
"""Fold g14: NOTES, T202, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g14 — the g1f crush decomposition: MULTI-COMPONENT — no single component's removal spares g1f (the ladder 0.404/0.210/0.251/0.258/0.403, all crush) — the per-draw structure is DIMENSIONAL, not componential; the co-reads: LETHAL != CARRIER (the -s_0 direction at full norm ANNIHILATES (3e-7) yet its removal never spares; the span's top-3 at full norm are BENIGN (0.947)); the delta 83.9% span-orthogonal; first-order predicts nothing (2026-10-02 ~22:05Z) — DONE

WHAT WE DID: the component-transplant ladder on g1f's delta (the
span/span-orth/top-3/s_0-part/span-rest removals, each at the full
step's norm; the refs reproduced bit-clean 0.2577/0.2951; the span
= e211's committed pristine wash-span reproduced to 0.0 rel); all
gates PASS; 55.2s CPU.

WHAT WE SAW (T202): MULTI-COMPONENT — removing ANY one component
still crushes (0/5 spare; the ladder 0.404/0.210/0.251/0.258/
0.403): THE g1f CRUSH IS DISTRIBUTED — nothing carries it the way
s_0 carried g1e's (g1e's s_0-carriage was the DRAW, not the law).
THE CO-READS THAT SHARPEN: (1) LETHAL != CARRIER — the -s_0
direction at full norm ANNIHILATES (0.0000003) yet removing it
never spares: a direction can be maximally lethal without being
the damage's carrier; (2) the span's top-3 directions at full
norm are BENIGN (0.947 — even protective); (3) the delta is 83.9%
SPAN-ORTHOGONAL (only 16% in the family's wash-span) with s_0
nearly out-of-span (0.072); (4) first-order predicts nothing
(-0.021 vs -1.939). THE MECHANISM TABLE'S HONEST FINISH: the
crush delta-carried (n=2, licensed); the carrier PER-DRAW (g1e:
s_0; g1f: distributed — dimensional, not componential); the wall
re-captures by +2 everywhere. HONESTY: n=1 per arm; the nested-
component caveat (each removal renormalizes a 2.74M-dim remainder
— "spared" means a non-lethal remainder direction); the span the
family's reference. SUCCESSORS: the g1e mirrored ladder; a third
cons draw; the +2 re-capture anatomy.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T201 —"
card = """## T202 — g14: lethal is not carrier; the g1f crush is dimensional (2026-10-02 ~22:05Z)

The decomposition answers T200's successor with the registered
modal prediction: no single component's removal spares g1f — the
crush is DISTRIBUTED, and the per-draw structure is dimensional,
not componential. THE DISTINCTION THE CELL GIVES THE LAB: LETHAL
!= CARRIER — the -s_0 direction at full norm annihilates totally
(3e-7) yet its removal never spares; a direction's direct lethality
says nothing about whether it carries a step's damage. THE
MECHANISM TABLE, CLOSED WITH ITS HONEST SHAPE: delta-carried
(replicates n=2); carrier-per-draw (g1e: one direction; g1f:
none — distributed); first-order blind; the wall universal in
re-capture. W028'S LAW ONE LAST TIME, NOW INSIDE THE MECHANISM:
"which component carries the crush" was a height question with a
draw-specific answer; "the crush is delta-carried and the wall
re-captures" are the shape sentences that replicate.

"""
assert anchor in t and "## T202" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHED 21:40Z — bars: COMPONENT-NAMED / MULTI-COMPONENT / GRADED |",
              "| DONE 22:05Z (T202: MULTI-COMPONENT — no single removal spares (0/5); the g1f crush DISTRIBUTED; LETHAL != CARRIER (the -s0 direction annihilates at full norm yet its removal never spares); the top-3 span directions benign (0.947); the delta 83.9% span-orthogonal; the mechanism table closed: delta-carried n=2, carrier-per-draw, first-order blind) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T22:06:00Z"
st["current_experiment"] = ("g14 FOLDED (T202: MULTI-COMPONENT - the g1f crush dimensional; LETHAL != CARRIER; the "
                            "mechanism table closed). Fleet: e224 (the training exposure) alone."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g14 folded")
