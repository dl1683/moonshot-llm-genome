# -*- coding: utf-8 -*-
"""Fold e198: NOTES, T162, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e198 — the architecture-vs-biography cut: BIOGRAPHY-CARRIES — the architecture suspect CLEARED; the flight concentration is organism-1 lineage biography; the t=2 wrinkle (u2 concentrates) and the rim picture now 2/2 architectures, 4/4 lineages (2026-10-01 ~22:25Z) — DONE

WHAT WE DID: the flight map on e193b's fresh 2.74M MIRABEL root
(org-1's EXACT architecture, the terrain-replicating fact, theta_1
ALIVE 0.3673, theta_2 alive 0.6911); 14/14 gates bit-tight; 310.6s
CPU.

WHAT WE SAW (T162): NEITHER EFFECT — (1) THE FLIGHT RAY u1 IS THE
SOFTEST DIRECTION (never crosses 0.27 within [0, 3.0]; min 0.5169
at D=3.0 vs its own u0 edge 1.9471): categorically
non-concentrating (org1 ratio 0.171; this: unresolved-high). (2)
THE WALK IS SAFER THAN ITS EDGE (k=1 kills at 2.4128 vs 1.9471;
ratio 1.239 >= 0.85) — e197's texture, now at 2.74M with a LIVE
window. BIOGRAPHY-CARRIES: a fresh 2.74M draw, same architecture,
terrain-replicating fact, alive mid-flight — and no flight
concentration. THE ARCHITECTURE SUSPECT IS CLEARED (n=1
draw/fact caveat). (3) THE T=2 WRINKLE (riding, no bar): the t=2
front u2 DOES concentrate (kill 0.8313, ratio 0.427 — below the
0.7 bar): LETHALITY ARRIVES AT t=2 HERE, t=1 IN ORG1 — the
concentration may be a TIMING matter (when the rotation finds the
lethal direction), not a whether. (4) THE RIM PICTURE HOLDS at
2/2 architectures, 4/4 lineages: the walk's recomputed direction
IMPROVES the fact (0.367 -> 0.691) before the far crash. (5)
ALIGNMENT PREDICTS NOTHING (|cos| <= 0.092 everywhere). THE
CANDIDATES LEFT: fact strength (org1's theta_1 0.679 vs 0.367);
lineage draw; FACT IDENTITY (ZEPHYRA-vs-MIRABEL — the one
variable the fork could not hold fixed). HONESTY: n=1 root/fact;
the fresh-draw caveat; the riding textures carry no bars.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T161 —"
card = """## T162 — e198: biography carries the flight — and the t=2 wrinkle reopens the timing question (2026-10-01 ~22:25Z)

The fork resolves to BIOGRAPHY: a fresh 2.74M root with org-1's
exact architecture, a terrain-replicating fact, and a live
mid-flight state shows NEITHER the t=1 flight concentration nor
the recomputation bonus — the architecture suspect is cleared.
THE FLIGHT QUESTION'S LEDGER: org1 (concentrates at t=1, ratio
0.17); org2 dead (absent); org2 alive (absent); MIRABEL alive
(absent at t=1 — BUT u2 concentrates at 0.43). THE T=2 WRINKLE IS
THE REOPENED DOOR: the rotation finds a lethal direction by t=2
here — the concentration may be a WHEN, not a WHETHER: the front
needs time (or steps) to rotate onto the fleeing support, and org1
did it in one step where MIRABEL needs two. THE TIMING CUT IS
NAMED (ripening): org1's own u2/u3 map (does its concentration
DEEPEN past t=1?) vs MIRABEL's t=3/t=4 — the concentration's
onset curve on both organisms. THE RIM PICTURE IS NOW UNIVERSAL
(2/2 architectures, 4/4 lineages: the recomputed direction always
improves the fact before the far crash) — the valley-with-rim is
the physics; the concentration's timing is the biography.
ALIGNMENT PREDICTS NOTHING (again) — the day's most repeated
negative. THE CANDIDATES: fact strength, draw, fact identity.

"""
assert anchor in t and "## T162" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e198 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e198 | THE ARCHITECTURE-VS-BIOGRAPHY CUT | DONE 22:25Z (T162: BIOGRAPHY-CARRIES — the architecture CLEARED (fresh 2.74M, terrain-replicating, alive: NEITHER effect; u1 the softest, walk safer than its edge); THE T=2 WRINKLE: u2 concentrates at 0.43 — the concentration may be a WHEN not a WHETHER; the rim picture universal 2/2 arch, 4/4 lineages; alignment predicts nothing; candidates: fact strength/draw/fact identity) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T22:26:00Z"
st["current_experiment"] = ("e198 FOLDED (T162: BIOGRAPHY-CARRIES — architecture cleared; the t=2 wrinkle reopens the "
                            "timing question). Fleet: g1bS4 (GPU, the movement-matched dose) alone; the timing cut "
                            "ripening.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e198 folded")
