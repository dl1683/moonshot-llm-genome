# -*- coding: utf-8 -*-
"""Fold e197: NOTES, T160, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e197 — the alive window: ALIVE-BUT-NO-FLIGHT — aliveness is necessary but NOT sufficient; the rim exists without the concentration (2026-10-01 ~21:55Z) — DONE

WHAT WE DID: the half-step wash (s* = 0.4582, the frozen ladder's
largest rung landing above the 0.27 bar, registered BEFORE the
walk; the same seed-10902 stream, direction machinery verbatim) on
the f2 root; all 15 gates bit-clean; 190s CPU.

WHAT WE SAW (T160): THE ALIVE WINDOW OPENED EXACTLY AS DESIGNED —
g-4 alive t=1 through t=4 (0.419/0.851/0.528/0.648), killed at t=5
(D 0.8387) — AND NEITHER DYNAMIC EFFECT FORMED: (1) NO
RECOMPUTATION BONUS (walk 0.8387 vs its static edge 0.5252, ratio
1.60 — the sub-step path is SAFER than its own ray; org1 0.771;
org2 full-step 1.001); (2) NO FLIGHT CONCENTRATION (the alive
lineage's u1 ratio 3.17 — softer than even the DEAD lineage's
2.58; org1 0.17). ALIVE-BUT-NO-FLIGHT: the honest fork — aliveness
alone does not build the flight structure; SOMETHING ELSE OF
ORGANISM 1 CARRIES THE DYNAMICS (candidates: its fact's strength;
its architecture 2.74M-6L vs 873k-4L; its lineage biography).
THE RIM-WITHOUT-CONCENTRATION TEXTURE: from the alive theta_1,
u0/u2 kill almost instantly (~0.067) while the walk's own -u1
direction IMPROVES the fact (0.42 -> 0.85 at D 0.4) before
crashing at 1.83 — THE VALLEY-WITH-RIM PICTURE EXISTS IN THIS
ORGANISM TOO, but the lethality never CONCENTRATES into the flight
direction. Front rotation large here as well (cos(u0,u1) -0.263).
HONESTY: n=1 root, one stream; the sub-step lineage is a
COUNTERFACTUAL wash — the claim is causal for this biography, not
a population claim; kill-D resolution ~s*/5.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T159 —"
card = """## T160 — e197: the honest fork — aliveness is not the carrier; the rim is universal, the concentration is not (2026-10-01 ~21:55Z)

The discriminating cell fired its honest fork: the alive window
opened (the half-step lineage lived t=1..t=4 exactly as designed)
and NEITHER effect formed — no recomputation bonus (the path SAFER
than its ray), no flight concentration (softer than the dead
lineage). THE ALIVE WINDOW IS NECESSARY BUT NOT SUFFICIENT. THE
MISSING CARRIER'S CANDIDATES, ranked by testability: (1)
ARCHITECTURE/SIZE — organism 1 is 2.74M 6L; this organism 873k 4L;
e193b's fresh root was 2.74M and its MIRABEL fact replicated the
whole TERRAIN — but the FLIGHT map was never read there: THE
DISCRIMINATING CUT IS NAMED (the flight map on e193b's MIRABEL
root: if the concentration appears, architecture carries it; if
not, organism 1's specific biography); (2) FACT STRENGTH (org1's
theta_1 read 0.679; this lineage's 0.419 — a strength threshold?);
(3) lineage biography (untestable except by draws). THE TEXTURE
THAT SAVES THE PICTURE: the rim-without-concentration — from the
alive theta_1, the static rays kill instantly while the walk's own
recomputed direction IMPROVES the fact before the far crash: THE
VALLEY-GEOMETRY IS UNIVERSAL (present in both organisms), the
LETHALITY-CONCENTRATION is organism 1's. The dissection's next cut
is already sharp.

"""
assert anchor in t and "## T160" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e197 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e197 | THE ALIVE WINDOW | DONE 21:55Z (T160: ALIVE-BUT-NO-FLIGHT — the window opened (alive t=1..4) and NEITHER effect formed (no bonus: the path safer than its ray, ratio 1.60; no concentration: u1 ratio 3.17); aliveness NECESSARY NOT SUFFICIENT; the rim-without-concentration texture (the valley-geometry universal, the lethality-concentration org-1's); the next cut named: the flight map on e193b's MIRABEL root) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T21:56:00Z"
st["current_experiment"] = ("e197 FOLDED (T160: the honest fork — aliveness necessary not sufficient; the rim universal, "
                            "the concentration org-1's). Fleet: g1bS3 (GPU, arms-for-the-record) + e198 DISPATCHING "
                            "(the flight map on e193b's MIRABEL root — the architecture-vs-biography cut)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e197 folded")
