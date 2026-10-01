# -*- coding: utf-8 -*-
"""Fold e195: NOTES, T157, paper discussion update, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e195 — the rotated-ray terrain: FLEEING-IS-LETHAL — the flight direction is itself the killer (0.39 vs 2.27, an 83% drop); direction-of-flight beats current-alignment; the terrain both concentrates and rotates (2026-10-01 ~19:20Z) — DONE

WHAT WE DID: static graded jumps along three sign-ray families
(u0/u1/u2) from the root AND from the one-stepped state theta_1;
237.7s CPU; every gate bit-exact (the walk reproduces opt2's
committed trajectory to 0.0; e194's fine grid and front traces
reproduced exactly).

WHAT WE SAW (T157): (1) FLEEING-IS-LETHAL fires decisively: the
rotated ray sign(g_1) from the root kills at D 0.3875 vs
sign(g_0)'s 2.2699 — 82.9% lower (bar 15%) — the most lethal
static ray ever measured in this program; lethality concentrates
exactly where the support fled TO. (2) THE TERRAIN ROTATES TOO:
from theta_1 the panel REVERSES (u0 0.616 < u1 0.828) — BOTH
truths in one cell: the flight direction is absolutely lethal from
the root AND the lethality is state-relative. (3) THE
DISSOCIATION: the rotated ray is ANTI-ALIGNED with the root's fact
gradient (cos -0.067) yet deadliest; the ORIGINAL ray is the
aligned one (+0.040) and kills 6x later — DIRECTION-OF-FLIGHT
BEATS CURRENT-ALIGNMENT: the killer ray follows where the support
is GOING, not where the death gradient points at the start.
(4) THE VALLEY HAS A RIM: short jumps along -u1 from theta_1
first IMPROVE the fact (0.679 -> 0.824 at D 0.30) before crashing
— the fleeing-support valley's near rim is a pump. (5) CE
currency: the fact dies 10x earlier than general degradation
along the flight ray (fact dead at 0.39; val CE still climbing
smoothly). HONESTY: n=1 organism/fact, one stream; u2's gradient
is the post-kill continuation's (read-only, disclosed);
cross-panel D comparisons never adjudicated. FOLLOW-ONS NAMED: the
valley's width (the basin the support fled into); does a second
organism's front flee the same way (e193's replicate line)?
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T156 —"
card = """## T157 — e195: the flight direction is the killer — direction-of-flight beats current-alignment (2026-10-01 ~19:20Z)

FLEEING-IS-LETHAL, decisively: the rotated ray kills at 0.39 from
the root where the original kills at 2.27 — an 83% concentration
of lethality into the direction the support fled toward. THE
DISSOCIATION IS THE DAY'S DEEPEST TWIST: the flight ray is
ANTI-aligned with the root's fact gradient (cos -0.067) yet
deadliest; the aligned ray kills 6x later. The killer direction is
not where the death gradient points NOW — it is where the fact's
support is GOING. Static alignment readings (the whole day's
terrain program!) measure the wrong thing unless they state their
point AND the state's history: the lethal direction is a property
of the TRAJECTORY (which way the support moves under this wash),
not of the landscape alone. THE BOTH-TRUTHS READING: the flight
ray is absolutely lethal from the root AND the panel reverses
from theta_1 — the terrain concentrates AND rotates; e192's
order (measured on t=0 rays) stands as the root-panel biography.
THE VALLEY-WITH-RIM picture: the fleeing support lands in a basin
whose near rim pumps (short -u1 jumps from theta_1 IMPROVE the
fact 0.68 -> 0.82 before the crash) — the flight is not toward
death but THROUGH a rim into a valley whose far wall kills. THE
FOLLOW-ONS: the valley's width; the second organism's flight
direction (does e193's replicate line flee the same way?).

"""
assert anchor in t and "## T157" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "DISCUSSION-MECH NOTE (e194, for the mechanism paragraph):"
if old in s:
    s = s.replace(old, """DISCUSSION-MECH NOTE (e194+e195, for the mechanism paragraph):
   the lethal subspace flees with the state; a re-computed front
   pursues it (one recomputation = the whole 23% bonus) while a
   re-orienting walk rotates away and spares — dynamics cut both
   ways, measured in both directions; AND THE FLIGHT DIRECTION
   ITSELF IS THE KILLER (e195: the rotated ray kills at 0.39 vs
   the original's 2.27, ANTI-aligned with the root gradient —
   direction-of-flight beats current-alignment; the valley the
   support flees into has a pumping near rim):""", 1)
    io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e195 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e195 | THE ROTATED-RAY TERRAIN | DONE 19:20Z (T157: FLEEING-IS-LETHAL — the flight ray kills at 0.39 vs 2.27 (83% lower), ANTI-aligned with the root gradient yet deadliest: DIRECTION-OF-FLIGHT BEATS CURRENT-ALIGNMENT; the panel reverses from theta_1 (concentrates AND rotates); the valley-with-rim pump; the fact dies 10x before general degradation) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T19:21:00Z"
st["current_experiment"] = ("e195 FOLDED (T157: the flight direction is the killer — direction-of-flight beats "
                            "current-alignment; the valley-with-rim). Fleet: g1bS2 alone (consolidation redo -> the "
                            "wall arms). The ledger's only open bracket: the scale verdict.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e195 folded")
