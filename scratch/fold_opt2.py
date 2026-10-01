# -*- coding: utf-8 -*-
"""Fold opt2: NOTES, T151, claims-ledger refinement, QUEUE DONE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## opt2 — GRADED: the density ladder — DENSITY CARRIES NOTHING; MAGNITUDE-INFORMATION ORDERS THE KILL; the sign path kills BELOW its own static edge (2026-10-01 ~16:15Z) — DONE

WHAT WE DID: the optimizer arc's terminal cell at matched per-step
L2 1.6543 in the licensed e185 wash cell (all gates bit-clean; the
chart's same-point anchor pair -0.0385/+0.0396 reproduced EXACTLY —
T150's estimator lesson trajectory-anchored at both ends; dual-
estimator alignment at every checkpoint).

WHAT WE SAW (T151): THE DENSITY LADDER (densified kill-D): TOPK-10%
0.9066 < TOPK-50% 0.9203 == raw-g 0.9203 (opt1c's committed rung,
a four-digit echo) < full sign(g) 1.7496 < A0-Adam 2.4893
(checkpoint convention, cross-convention disclosed). Neither named
bar fired — every arm kills, nothing spares — and the LADDER IS THE
RESULT: (1) DENSITY CARRIES NOTHING: keeping |g|-selection + g's
own magnitudes, deleting 90% of the coordinates leaves the kill AT
the raw cliff (the top-10% front holds 75% of ||g||^2 and the
entire lethality; top-50% is indistinguishable from the full
gradient). (2) WHAT MOVES THE KILL IS MAGNITUDE INFORMATION — the
T143 lethality ordering made INTERVENTIONAL at matched size:
flattening to sign moves the kill 0.92 -> 1.75; Adam's warmed path
2.49; THE LETHAL OBJECT IS THE |g|-WEIGHTED FRONT AT ANY DENSITY
>= 10%. (3) THE SIGN PATH KILLS BELOW ITS OWN STATIC EDGE (1.75 vs
the 2.5 static ray, which reads ~0.5 alive there) — the drifting
fresh-sign front is MORE lethal per displacement than its fixed
ray; e192's disclosed sign-class path-vs-ray gap is now measured.
With the chart's shuffled-sign inertness: THE KILL LIVES IN THE
COORDINATE-MAGNITUDE PAIRING, CONCENTRATED IN A TENTH OF IT.
HONESTY: n=1, CPU fp32; every-step g-12 + along-step
densification; the metrics' canned stretch phrase reads opposite to
the number's meaning (the kill arrived EARLIER than the static
edge — the agent's disclosed caution adopted here); top-k churn
unobservable (kills at +1). FOLLOW-ONS NAMED: the top-k ladder
downward (1% — where does the front stop being the whole kill?)
and the sign-path-vs-ray mechanism.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T150 —"
card = """## T151 — opt2: the lethal object is the |g|-weighted front — density exonerated, magnitude-information convicted (2026-10-01 ~16:15Z)

The density ladder answers the optimizer arc's last open question
with digit-level cleanliness (TOPK-50% == raw at 0.9203): DENSITY
CARRIES NOTHING — a tenth of the coordinates, |g|-selected and
magnitude-weighted, carry the whole kill. WHAT ORDERS THE KILL IS
MAGNITUDE INFORMATION, now interventional at matched size: raw
0.92 -> flattened sign 1.75 -> Adam's warmed sign-path 2.49. THE
SIGN PATH KILLS BELOW ITS OWN STATIC EDGE: the re-computed front
beats the frozen ray (1.75 vs 2.5) — the trajectory is MORE lethal
than its shadow, the mirror of the bleed (whose re-orientation
SPARES what its ray kills): DYNAMICS CUT BOTH WAYS, and the
interesting objects are the two mismatches (the bleed's protective
re-orientation; the sign path's lethal re-computation). WITH THE
CHART: the kill lives in the coordinate-magnitude pairing
(the shuffled sign is inert), concentrated in the top tenth —
a sharp, small, nameable object: THE LETHAL FRONT. The estimator
lesson is now anchored at both ends on trajectories. The arc's
shape: opt1 asked CLOCK-vs-GATE; opt1c split direction from size;
e191/e192 mapped the terrain; the chart killed two pictures; opt2
names the weapon. FOLLOW-ONS: the 1% rung; the sign-front
mechanism. HONESTY: n=1, CPU fp32, the canned-phrase caution
adopted (the number, not the frozen text, is the reading).

"""
assert anchor in t and "## T151" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

c = "scratch/claims_ledger.md"
s = io.open(c, encoding="utf-8").read()
old = "| C4 | The wash kill decomposes: CLOCK = optimizer normalization (1683x/step at matched lr; the normalizer flips fact-relevance sign); GATE = displacement; CLASSES = ballistic-maximal (0.92) / ballistic-normalized (~2.5) / diffusive (stall ~2.1; [kill-D: opt1b3])"
if old not in s:
    old = "CLASSES = ballistic-maximal (0.92) / ballistic-normalized (~2.5) / diffusive (stall ~2.1; [kill-D: opt1b3])"
new = "CLASSES = ballistic-maximal (0.92; THE LETHAL FRONT = the |g|-weighted top-tenth, opt2: density carries nothing) / ballistic-normalized (~1.75 path / 2.5 ray) / diffusive (grinds ~2.25, no death)"
assert old in s, "c4"
s = s.replace(old, new, 1)
io.open(c, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| opt2 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| opt2 | THE DENSITY QUESTION | DONE 16:15Z (T151: GRADED — the ladder IS the result: TOPK-10 0.9066 < TOPK-50 0.9203 == raw (4-digit echo) < sign 1.7496 < Adam 2.4893; DENSITY CARRIES NOTHING; MAGNITUDE-INFORMATION ORDERS THE KILL (interventional); the sign path kills BELOW its static ray — dynamics cut both ways; THE LETHAL FRONT = the |g|-weighted top tenth) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T16:16:00Z"
st["current_experiment"] = ("opt2 FOLDED (T151: the lethal front is the |g|-weighted top tenth — density exonerated, "
                            "magnitude-information convicted; dynamics cut both ways). Fleet: g2g (GPU) alone; R60 "
                            "launches when it lands.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("opt2 folded")
