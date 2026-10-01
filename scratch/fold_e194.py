# -*- coding: utf-8 -*-
"""Fold e194: NOTES, T156, paper discussion note, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e194 — the sign-front mechanism: GRADED — no named bar; the reads convict ROTATION-TO-FLEEING-SUPPORT: the fresh front tracks the fact's moved lethal subspace, and ONE recomputation buys the whole lethality bonus (2026-10-01 ~19:25Z) — DONE

WHAT WE DID: the front-overlap reads (t=0,1,2 + post-kill), the
21-point fine static grid (D 1.50-2.50), and the k-ladder
(front recomputed every k in {1,2,4,8} at matched per-step L2);
bit-exact provenance throughout (opt2's a_sign trajectory AND its
saved s2 checkpoint reproduced to 0.0; e192's static-ray md5
matched; the chart's matched-point anchor to 8.1e-15). 54s CPU.

WHAT WE SAW (T156): (1) NOT CHASE (frozen-ray frame): the front's
overlap with the frozen g-ray COLLAPSES (+0.595 -> -0.162); the
front ANTI-ROTATES off the death ray (consecutive fronts
anti-correlated -0.15, 256x the isotropic floor). (2) NOT
ACCUMULATION: the k=1 walk's efficiency is 0.919 — SUB-DIFFUSIVE
(it kills while accumulating displacement worse than a random
walk). (3) NOT A STATIC ARTIFACT: the fine grid puts the TRUE
static edge at 2.2699 (the committed 2.5 was coarse-grid); the
path's 1.7496 is 23% BELOW — the inversion is real. (4) THE
K-LADDER IS A STEP FUNCTION: k=1 kills at 1.7496; k in {2,4,8}
ALL kill at 2.2744 (= the static edge; the walk-bracket and
grid-interp agree to 0.2%) — through their kills the k>=2 walks
ARE the frozen front: ONE RE-COMPUTATION IS THE WHOLE LETHALITY
BONUS. (5) THE MECHANISM NOUN: the fresh front's matched-point
alignment with the FACT'S OWN GRADIENT RISES along the path
(+0.040 -> +0.060, post-kill +0.13) while its frozen-ray overlap
collapses — THE FRONT TRACKS THE FACT'S FLEEING SUPPORT: the
lethal subspace moves with the state, and the recomputed front
follows it. DYNAMICS CUT BOTH WAYS, NOW MEASURED IN BOTH
DIRECTIONS (the bleed's re-orientation spares; the sign front's
re-computation pursues). HONESTY: n=1 organism/fact, one stream;
the k-ladder saturates at k=2 (reported, not hidden); post-kill
reads never adjudicated. FOLLOW-ON NAMED: the rotated-ray terrain
(sign(g_1) ray map) — where did the support flee TO?
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T155 —"
card = """## T156 — e194: the lethal subspace flees and the fresh front pursues — one recomputation, the whole bonus (2026-10-01 ~19:25Z)

GRADED with the mechanism convicted anyway: NOT chase (the frozen
frame), NOT accumulation, NOT artifact — the true reading is
ROTATION-TO-FLEEING-SUPPORT. The fine static edge is 2.2699 (the
2.5 was coarse-grid); the recomputed path kills at 1.75 — a real
23% inversion — and the k-ladder is a STEP FUNCTION: k=1 at 1.75,
k>=2 at the static edge: ONE RE-COMPUTATION IS THE WHOLE BONUS.
The discriminating pair (T150's estimator lesson earning its keep
again): the front's frozen-ray overlap collapses while its
matched-point alignment with the FACT'S OWN GRADIENT rises —
the lethal subspace MOVES WITH THE STATE and the fresh front
follows it. THE SYMMETRY WITH THE BLEED completes the day's
picture: the re-orienting walk's steps rotate AWAY from the
lethal direction and spare; the sign path's steps re-computed
TOWARD the fleeing lethal direction and kill sooner — dynamics
cut both ways, now measured in both directions, both at n>=2
anchors. THE SUBLINEAR EFFICIENCY (0.919 — worse than a random
walk at accumulating displacement) says the path SPENDS its
budget on rotation, not advance: lethality is not borrowed
steepestness. FOLLOW-ON: the rotated-ray terrain names where the
support fled TO; the discussion's mechanism paragraph writes
itself from this cell.

"""
assert anchor in t and "## T156" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "R6 THE GENERATIVE TURN:"
if old in s:
    s = s.replace(old, """DISCUSSION-MECH NOTE (e194, for the mechanism paragraph): the
   lethal subspace flees with the state; a re-computed front
   pursues it (one recomputation = the whole 23% bonus; k-ladder a
   step function) while a re-orienting walk rotates away and
   spares — dynamics cut both ways, measured in both directions.
R6 THE GENERATIVE TURN:""", 1)
    io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e194 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e194 | THE SIGN-FRONT MECHANISM | DONE 19:25Z (T156: GRADED — NOT chase/accumulate/artifact; ROTATION-TO-FLEEING-SUPPORT: the front's frozen-ray overlap collapses while its fact-gradient alignment rises; the TRUE static edge 2.2699 (the 2.5 was coarse); the k-ladder is a STEP FUNCTION — one recomputation = the whole bonus; efficiency 0.919 sub-diffusive; bit-exact provenance) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T19:26:00Z"
st["current_experiment"] = ("e194 FOLDED (T156: the lethal subspace flees, the fresh front pursues — one recomputation, "
                            "the whole bonus). Fleet: g1bS2 alone (consolidation -> the wall arms). The ledger's only "
                            "open bracket: the scale verdict.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e194 folded")
