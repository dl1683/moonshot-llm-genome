# -*- coding: utf-8 -*-
"""Fold e193b: NOTES, T155, claims-ledger updates, paper Fig-5 note,
QUEUE DONE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e193b — the critic's replicate: ORDER-SCRAMBLES fires on the FACT axis, not the lineage axis; the pump splits INSIDE one organism — the ridge is fact-biography, the cliff is physics (2026-10-01 ~19:00Z) — DONE

WHAT WE DID: the fresh-root two-fact cell at organism-1's EXACT
architecture (6L/6H/192d, 2,739,072 params, seed 5301): both facts
installed (ZEPHYRA pZ 0.377; MIRABEL pZ 0.533; the second install
crushed the first to 0.0003 — e154's interference, co-reported),
e113 consolidation, then the full terrain pass per fact (five rays,
dual currency, the ladder, the rider, the in-span 3-draw range,
the magnitude-shuffle breaker). All Rule-12 gates PASS (the
seed-10902 stream bit-matches e185's md5s; measured STEP_L2 1.6544
vs organism-1's 1.6543).

WHAT WE SAW (T155): (1) ORDER-SCRAMBLES FIRES on the frozen
"any fact" rule — but the reading is sharp: MIRABEL (the fact that
passed its root-strength gate on the e192-verbatim ruler)
replicates the WHOLE terrain (g 0.8 < sign 2.0 < randoms >4.0,
both [0.5x,2x] windows, the rider dying AT the static cliff
0.827~0.8 — e192's interventional anchor at n=3); ZEPHYRA
(G_CONS'd at D=0 on g-12: 0.149 — e193's lesson on a fresh draw;
its own fallback ruler reads 0.462) holds the ORDER but its
kill-Ds move >2x DOWN on every expressed ruler. THE ORDER IS
DRAW/FACT-ROBUST (3/3 organisms, 2/2 facts); THE ABSOLUTE KILL-Ds
ARE THE FACT'S STRENGTH BIOGRAPHY. (2) PUMP-PER-FACT decides
T153's thesis IN ONE ORGANISM ALONG THE SAME RAYS: ZEPHYRA ridge
PRESENT (+0.018 g-ray), MIRABEL ridge ABSENT (+0.0004) — THE RIDGE
IS FACT-LEVEL BIOGRAPHY (consolidation alignment), THE CLIFF IS
PHYSICS. (3) THE MISSING NUMBER: in-span 3-draw spreads 3.0x
(ZEPHYRA) / 2.0x (MIRABEL) — e131's suppressed in-span spread was
the NORM, not the outlier. (4) THE MAGNITUDE-SHUFFLE BREAKER:
kills at 1.68x the topk-10 kill — support+signs ALONE suffice to
kill; magnitude-pairing CONTRIBUTES but is not necessary — neither
pure-support nor pure-pairing. (5) The ladder's topk/raw cluster
is tight at n=3 organisms (0.99/0.99/0.99 vs f1 ~1.00); the sign
rung drifts wide AGAIN (2.24 vs f1 1.90; e193 read 2.63) — the
sign edge is lineage- AND fact-sensitive where the magnitude
cluster is neither. HONESTY: fresh root n=1, each fact n=1;
ZEPHYRA's window comparison is cross-ruler (all co-ruler kills
co-reported: g-12 dies at the first grid point); one 89C thermal
hold honored, GPU released.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T154 —"
card = """## T155 — e193b: the order is physics, the distances are biography — and the ridge/cliff thesis decided inside one organism (2026-10-01 ~19:00Z)

The critic's replicate lands the cleanest generality statement of
the program: THE RAY ORDER IS DRAW-FACT-ARCHITECTURE-ROBUST (3/3
organisms, 2/2 facts, architecture pinned and lineage varied);
THE ABSOLUTE KILL DISTANCES ARE THE FACT'S STRENGTH BIOGRAPHY
(MIRABEL inside both windows; ZEPHYRA — ruler-dead at D=0 on the
imported ruler — holds the order at >2x-down distances). THE
RIDGE/CLIFF SPLIT IS DECIDED WITHOUT CONFOUND: the same organism,
the same rays, two facts — ZEPHYRA pumps, MIRABEL does not —
T153's thesis confirmed: the ridge is consolidation-alignment
biography; the cliff is physics. THE IN-SPAN SPREAD IS THE NORM
(3.0x/2.0x here; e131's suppressed spread replicated) — the
subspace's lethality is direction-heterogeneous everywhere
measured. THE BREAKER'S MIXED VERDICT refines the lethal front:
support+signs suffice (magnitude-shuffle kills at 1.68x) but
magnitude-pairing contributes — the front is support > signs >
magnitudes in necessity order. THE SIGN RUNG's sensitivity is now
triply-observed (f1 1.90, e193 2.63, here 2.24) — the sign edge
is the terrain's softest number; the paper's clause carries the
range. THE RULER BIOGRAPHY LESSON (second occurrence): a fact can
be alive on its own ruler and dead on the imported one — rulers
are facts' property too.

"""
assert anchor in t and "## T155" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

c = "scratch/claims_ledger.md"
s = io.open(c, encoding="utf-8").read()
old = "| C4 | The wash kill decomposes:"
assert old in s, "c4 anchor"
i = s.index(old); j = s.index("\n", i)
row = s[i:j]
new_row = "| C4 | The wash kill decomposes: CLOCK = optimizer normalization (1683x/step at matched lr; attenuates fact-relevance ~2.5x at a matched point); GATE = displacement; THE RAY ORDER g < sign < random is draw/fact/architecture-robust (n=3 organisms, 2 facts, both replication axes); the ABSOLUTE kill-Ds are fact-strength biography (windows held by gate-passing facts only); the front: support+signs suffice, magnitude-pairing contributes; the sign edge is the softest number (1.9-2.6x range) | opt1/opt1c/opt2/e191/e192/e193/e193b | PROPOSED at n=3 with the biography clause"
s = s.replace(row, new_row, 1)
old = "| C5 | The pump: small displacement strengthens the fact on ORGANISM 1 ONLY (e193: absent on organism 2 — every ruler negative); the cliff is universal (order replicated n=2 lineages; the ridge is biography) | opt1 A1-A3, opt1b/c, e191, e192, e193 | pump n=1 organism; cliff/order n=2 (architecture co-varies, disclosed); the sign rung lineage-sensitive |"
assert old in s, "c5"
s = s.replace(old, "| C5 | THE PUMP IS FACT-LEVEL BIOGRAPHY (e193b decides in one organism: ZEPHYRA pumps, MIRABEL does not, same rays); the cliff is physics; present on 2 of 3 facts measured | opt1, e191, e192, e193, e193b | per-fact n=1; the consolidation-alignment mechanism is the hypothesis (T153/T155) |", 1)
io.open(c, "w", encoding="utf-8").write(s)

p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "Fig 5 (THE terrain figure — LICENSED at n=2 organisms/lineages:"
assert old in s, "fig5"
s = s.replace(old, "Fig 5 (THE terrain figure — LICENSED at n=3 organisms / 2 facts / both replication axes: the ORDER g < sign < random is draw-fact-architecture-robust; absolute kill-Ds are fact-strength biography (the caption carries the windows + the biography clause):", 1)
io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e193b |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e193b | THE CRITIC'S REPLICATE | DONE 19:00Z (T155: ORDER-SCRAMBLES fires on the FACT axis — MIRABEL replicates the whole terrain (the rider at n=3); ZEPHYRA order-holds at >2x-down; THE ORDER IS PHYSICS, THE DISTANCES ARE BIOGRAPHY; the pump splits INSIDE one organism (ridge = fact-biography, cliff = physics); in-span spread is the norm (3.0x/2.0x); the breaker: support+signs suffice, magnitudes contribute; the sign edge soft at 1.9-2.6x) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T19:01:00Z"
st["current_experiment"] = ("e193b FOLDED (T155: the order is physics, the distances are biography; the ridge/cliff "
                            "thesis decided in one organism). Fleet: g1bS2 (consolidation) + e194 (sign-front). "
                            "The claims ledger's only open bracket: g1bS2's scale verdict.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e193b folded")
