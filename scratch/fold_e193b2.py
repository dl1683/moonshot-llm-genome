# -*- coding: utf-8 -*-
"""e193b fold part 2: repair the corrupted C4 row (orphaned by the
T151-era edit), upgrade C4 to n=3, resolve C5, refresh C7; paper
Fig-5; QUEUE; STATE."""
import io, json

c = "scratch/claims_ledger.md"
s = io.open(c, encoding="utf-8").read()
# repair the orphaned C4 fragment -> the full upgraded row
old = "CLASSES = ballistic-maximal (0.92; THE LETHAL FRONT = the |g|-weighted top-tenth, opt2: density carries nothing) / ballistic-normalized (~1.75 path / 2.5 ray) / diffusive (grinds ~2.25, no death) | opt1/opt1c/opt1b/opt1b2/e191 | PROPOSED (n=1 lineage; D_kill window circular per R58; e192 licenses the one-organism picture) |"
assert old in s, "c4 orphan"
new = ("| C4 | The wash kill decomposes: CLOCK = optimizer normalization (1683x/step at matched lr; attenuates fact-relevance ~2.5x at a matched point — the flip was an estimator artifact); GATE = displacement; THE RAY ORDER g < sign < random IS DRAW/FACT/ARCHITECTURE-ROBUST (n=3 organisms, 2 facts, both replication axes); the ABSOLUTE kill-Ds are fact-strength biography (windows held by gate-passing facts only); CLASSES = ballistic-maximal / ballistic-normalized (~1.75 path / soft 1.9-2.6x ray) / diffusive (grinds, no death); the lethal front: support+signs suffice, magnitude-pairing contributes | opt1/opt1c/opt2/opt1b*/e191/e192/e193/e193b | PROPOSED at n=3 with the biography clause; the sign edge is the softest number |")
s = s.replace(old, new, 1)
old = "| C5 | The pump: small displacement strengthens the fact on ORGANISM 1 ONLY (e193: absent on organism 2 — every ruler negative); the cliff is universal (order replicated n=2 lineages; the ridge is biography) | opt1 A1-A3, opt1b/c, e191, e192, e193 | pump n=1 organism; cliff/order n=2 (architecture co-varies, disclosed); the sign rung lineage-sensitive |"
assert old in s, "c5"
s = s.replace(old, "| C5 | THE PUMP IS FACT-LEVEL BIOGRAPHY (e193b decides in one organism: one fact pumps, the other does not, same rays); the cliff is physics; present on 2 of 3 facts | opt1, e191, e192, e193, e193b | per-fact n=1; the consolidation-alignment mechanism is the hypothesis (T153/T155) |", 1)
old = "| C7 | THE RHYTHM: a zero-parameter organ self-times replay (timing 3/3 roots; amplitude = ruler-geo alignment, root-draw-bound) | g2b-g2f | licensed-with-scope; construction disclosures carried (refractory; constant threat); g2g controls pending |"
assert old in s, "c7"
s = s.replace(old, "| C7 | THE RHYTHM: a zero-parameter organ (cue-pool selector + thermostat) — threat-responsive [0.5x,2x] (weak step, ceiling-saturated); advantage +0.07 over a matched fixed schedule at n=3 seeds (median +0.033), DECOMPOSED: majority replay-batch selection (+0.056), minority timing (+0.013); refractory-tunable (shorter better; the 20-45 band partly a floor); timing 3/3 roots | g2b-g2f, g2g, g2g2 | licensed-with-scope at n=3 wash seeds / n=1 root |", 1)
io.open(c, "w", encoding="utf-8").write(s)

p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "Fig 5 (THE terrain figure — LICENSED at n=2 organisms/lineages:"
if old in s:
    s = s.replace(old, "Fig 5 (THE terrain figure — LICENSED at n=3 organisms / 2 facts / both replication axes (the ORDER draw-fact-architecture-robust; absolute kill-Ds = fact-strength biography, windows in the caption):", 1)
io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e193b |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e193b | THE CRITIC'S REPLICATE | DONE 19:00Z (T155: ORDER-SCRAMBLES fires on the FACT axis — MIRABEL replicates the whole terrain (the rider at n=3); ZEPHYRA order-holds at >2x-down; THE ORDER IS PHYSICS, THE DISTANCES ARE BIOGRAPHY; the pump splits INSIDE one organism (ridge = fact-biography, cliff = physics); in-span spread is the norm (3.0x/2.0x); the breaker: support+signs suffice, magnitudes contribute; the sign edge soft at 1.9-2.6x) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T19:02:00Z"
st["current_experiment"] = ("e193b FOLDED (T155: the order is physics, the distances are biography; the ridge/cliff "
                            "decided in one organism; the corrupted C4 row REPAIRED — orphaned since the T151 fold). "
                            "Fleet: g1bS2 (consolidation) + e194 (sign-front). The ledger's only open bracket: g1bS2's "
                            "scale verdict.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("part2 OK")
