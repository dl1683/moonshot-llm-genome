# -*- coding: utf-8 -*-
"""Fold e196: NOTES, T158, T157 amendment, paper discussion update,
QUEUE, STATE."""
import io, re

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e196 — the flight replicate: GRADED — organism 2 does not flee into lethality; the flight structure needs a LIVE MID-FLIGHT STATE to form (2026-10-01 ~20:30Z) — DONE

WHAT WE DID: e195's machinery on e193's f2 root (all 14 gates
bit-clean; the k=1 walk reproduces e193's committed row to 0.0);
208s CPU.

WHAT WE SAW (T158): (1) FLIGHT-REPLICATES does NOT fire —
organism 2's flight ray u1 kills at 1.358 vs its OWN u0 edge 0.525
(ratio 2.585; bar 0.60; organism 1 committed 0.171) — the OPPOSITE
direction: on organism 2 the flight ray is the SOFTEST of the
three. FLIGHT-ABSENT also fails (spread 66%): there IS structure,
but immaterial (u2 ratio 0.876). (2) THE PRE-REGISTERED ASYMMETRY
EXPLAINS IT: organism 2's step (0.916) exceeds BOTH its static
edges (g 0.20 / sign 0.58) — its walk kills AT step 1, and its
t=1/t=2 fronts are DEAD-STATE reads (theta_1 anchor 0.0068 vs
organism 1's alive 0.679). THE FLIGHT STRUCTURE NEEDS A LIVE
MID-FLIGHT STATE TO FORM. (3) CORROBORATING: organism 2's sign
path dies EXACTLY at its own static edge (0.5257 vs 0.5252 — NONE
of organism 1's 23% recomputation bonus; the e194 inversion is
absent where the mid-flight state is dead); its flight ray is
ORTHOGONAL to the fact gradient (+0.0008 vs org1's -0.067); the
ray geometry is comparable (cos(u0,u1) -0.165 vs -0.155 — not a
collinearity artifact). HONESTY: n=2 organisms total, each n=1
stream, architecture co-varies with lineage. DISCRIMINATING
FOLLOW-ON NAMED: a lineage that stays alive past t=1 (a smaller
step or a stronger fact) — does the flight concentration require
the live mid-flight state?
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T157 —"
card = """## T158 — e196: the flight is biography — and the reason found in the same cell: it needs a live mid-flight state (2026-10-01 ~20:30Z)

The replicate answers T157's follow-on NEGATIVELY with the
mechanism in hand: organism 2's flight ray is its SOFTEST
direction (ratio 2.58 vs organism 1's 0.17 — the opposite
direction), and the pre-registered asymmetry is the explanation —
organism 2's single step exceeds both its static edges, its walk
dies AT step 1, and its post-kill fronts are dead-state reads.
THE FLIGHT CONCENTRATION REQUIRES A LIVE MID-FLIGHT STATE: the
support cannot flee somewhere if it is already dead. The
corroboration is tight: where the mid-flight state is dead, the
recomputation bonus is absent too (the sign path dies exactly at
its static edge — no e194 inversion) — BOTH dynamic effects (the
pursuit and the flight) live in the alive window. T157 RESCOPES:
the flight-direction finding is organism-1 biography WITH a
mechanism hypothesis (the live-state condition); the discriminating
cell is named (a lineage alive past t=1 — smaller step or stronger
fact). THE PAPER'S DISCUSSION carries the conditional: dynamics
cut both ways WHERE THE ORGANISM IS ALIVE TO CUT; past the kill,
the terrain is static again.

"""
assert anchor in t and "## T158" not in t
t = t.replace(anchor, card + anchor, 1)

m = re.search(r"^## T157 — .*$(.*?)(?=^## T156)", t, re.M | re.S)
assert m, "t157"
body = m.group(1).rstrip("\n")
amend = """

[E196 AMENDMENT ~20:30Z]: the flight concentration is ORGANISM-1
BIOGRAPHY — organism 2's flight ray is its softest (ratio 2.58 vs
0.17) — WITH THE MECHANISM: its walk dies at step 1 (dead
mid-flight state; theta_1 0.0068 vs org1's 0.679); the
recomputation bonus is absent there too. The flight and the
pursuit both require the alive window. Rescopes this card's
universality claims."""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "AND THE FLIGHT DIRECTION\n   ITSELF IS THE KILLER (e195: the rotated ray kills at 0.39 vs\n   the original's 2.27, ANTI-aligned with the root gradient —\n   direction-of-flight beats current-alignment; the valley the\n   support flees into has a pumping near rim):"
if old not in s:
    old = "AND THE FLIGHT DIRECTION ITSELF IS THE KILLER"
assert old in s, "mech note"
i2 = s.index(old)
j2 = s.index("):", i2) + 2 if "):" in s[i2:i2+400] else i2 + len(old)
s = s[:i2] + "AND WHERE THE ORGANISM STAYS ALIVE PAST ITS FIRST STEP, THE FLIGHT DIRECTION ITSELF IS THE KILLER (e195: 0.39 vs 2.27, anti-aligned; e196: absent on the organism whose first step kills — the flight and the pursuit both require the alive window):" + s[j2:]
io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e196 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e196 | THE FLIGHT REPLICATE | DONE 20:30Z (T158: GRADED — does NOT replicate; organism 2's flight ray is its SOFTEST (ratio 2.58 vs 0.17); the pre-registered asymmetry explains: its step kills at t=1 (dead mid-flight state) and the recomputation bonus is absent there too — the flight and the pursuit both require the ALIVE window; the discriminating lineage named) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T20:31:00Z"
st["current_experiment"] = ("e196 FOLDED (T158: the flight is biography — it needs a live mid-flight state; both dynamic "
                            "effects live in the alive window). Fleet: g1bS2 alone (the last bracket)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e196 folded")
