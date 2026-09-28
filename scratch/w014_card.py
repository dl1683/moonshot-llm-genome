import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

t = open("THINKING.md", encoding="utf-8").read()
w_anchor = "## W013 — WONDER:"
w014 = """## W014 — WONDER: the memory layer is a semi-independent tenant — it dies to what the corpus ignores and ignores what the corpus dies to (2026-09-28 ~08:00Z)

E141's CE dissociation, savored properly: direction-scrambling
row 0 costs the corpus +0.70 nats but SPARES the fact (+6%);
removing row 0's norm kills the fact AND wrecks the corpus
(+1.40-2.00). The fact's route and the net's language function
share a pivot but have DIFFERENT failure modes: the route cares
about presence, the LM about direction. A memory system whose
Achilles heel (pivot removal) is exactly the intervention that
destroys general function, and whose insensitivity (direction)
is exactly what the general function cannot survive — the memory
is a semi-independent TENANT of the same building. Three
consequences worth ripening: (1) TARGETED UNLEARNING has a
predicted shape: route-level attack is maximally effective but
catastrophic (indiscriminate); band-level attack is useless or
BACKFIRES (the brake — deleting the old address STRENGTHENS a
routed memory); the promising surface is the fact-SPECIFIC
readout heads (e133's L0H3 class: 0.46 fact drop at 0.21 CE) —
head-level ablation should remove the fact with low collateral.
e125's three-surface design now carries a mechanism-backed
ordering: heads > route >> band, with the brake-trap as the
failure mode naive unlearning walks into. (2) The RMU arc
re-reads: RMU unlearning seals the readout gate (T052) — under
the tenant frame that is EVICTION at the head level, and e137's
question (does restoration re-wire cheaply?) becomes "does the
tenant keep its lease?" (3) SAVOR: the net's memories and its
language live together but keep separate insurance policies —
for the second paper this is the falsifiable claim that tiny-LLM
memory is an addressing layer over the LM, not knowledge IN the
LM, and the dissociation row (direction-scramble cell) is its
cleanest single exhibit.

""" + w_anchor
assert w_anchor in t
t = t.replace(w_anchor, w014, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e125 \|[^\n]*\n", q, re.M)
assert m
newrow = "| e125 | attack the moved fact (P7; three surfaces; W014 mechanism ordering PRE-REGISTERED ~08:00Z: heads > route >> band) | READY | surfaces: (a) row-0 mean-replacement (fact -97% known — price collateral CE/base-skill/held-30); (b) band deletion (expected ~null or BRAKE +0.210 — an 'unlearning' move that STRENGTHENS the routed memory); (c) fact-specific readout heads (e133's L0H3 class) — predicted to remove the fact with LOW collateral (0.46 drop at 0.21 CE was single-head; test combined fact-specific head set). Bars (R43 collateral-matched + W014): HEADS-WIN = head-set removes fact >=60% at collateral <=25% of route-level's; ROUTE-ONLY-EFFECTIVE = no head-set reaches 60% fact drop at any collateral; BRAKE-TRAP = band deletion raises fact expression >=+0.1. Removability ratio vs pre-consolidation fact at matched collateral (e043 asymmetry reference) |"
q = q[:m.start()] + newrow + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("W014 + e125 pre-registration in")
