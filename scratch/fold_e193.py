# -*- coding: utf-8 -*-
"""Fold e193: NOTES, T153, claims-ledger C5 rescope, paper updates,
QUEUE DONE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e193 — the second-organism replicate: THE ORDER IS LINEAGE-PHYSICS, THE PUMP IS NOT (and the front's sign rung drifts wide; re-orientation causal at n=2) (2026-10-01 ~17:30Z) — DONE

WHAT WE DID: one 93s eval-only CPU pass on the committed family-2
root (e157_f2_consolidated, 873k): the five-ray terrain map + the
density ladder (with the 1% rung) + the pump read + the pinned-ray
rider. All provenance gates bit-tight (root dial cells max|diff|
0.0; the stream md5-matches e185's hashes on the new net; this
organism's OWN measured step L2 0.9164 — never ported).

WHAT WE SAW (T153): (1) TERRAIN-REPLICATES FIRES: g-ray kill 0.20 <
sign 0.58 (2.90x) < random >4.0 on all three Gaussian rays —
FIG-5'S ORDER AT n=2 ORGANISMS / n=2 LINEAGES; the absolute kill-Ds
differ exactly as pre-registered (0.20 vs 0.92; ratio form
carried); ruler-robust (checked on three rulers). (2) THE FRONT:
topk-10/50 and raw replicate within +-25% in ratio form
(1.108/1.117/1.119 vs family-1's 0.985/1.000/1.000) and the 1%
rung kills at ratio 1.289 — NO FLOOR, NO SHRINK — but the SIGN rung
overshoots its band (2.63 vs [1.43, 2.38]): the magnitude-
informative cluster is lineage-stable; the sign-normalized rung is
not. (3) THE PUMP IS ABSENT ON ORGANISM 2: every small-D rise is
NEGATIVE on every ruler (the g-ray -0.079 vs family-1's +0.045
ridge) — C5's pump claim RESCOPES to n=1 organism; gradient-
specificity survives INVERTED (randoms flat; the g-ray
monotonically lethal). (4) THE RIDER: the pinned walk dies in the
static cliff's bracket [0.183, 0.275] — RE-ORIENTATION CAUSAL AT
n=2. HONESTY: architecture co-varies with lineage (873k 4L/128d
vs 2.74M 6L/192d — disclosed); the primary ruler is a trained
jitter geometry (the e192-verbatim g-12 battery reads 0.198 at the
f2 root — under the kill bar at D=0, T113's G_CONS bound —
co-rulers reported in every table, never adjudicated); CE_R canary
ordering replicates.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T152 —"
card = """## T153 — e193: the order is lineage-physics; the pump is biography — the replicate's clean split (2026-10-01 ~17:30Z)

The replicate splits the day's central objects by generality.
WHAT CROSSES LINEAGES: the terrain's ORDER (g < sign < random,
2.9x and >4.0 at n=2 — Fig-5's caption now true at two organisms);
the front's magnitude-informative cluster (topk/raw ratio-tight,
extending to the 1% rung with neither floor nor shrink); the
re-orientation causality (the pinned walk dies at the static cliff
on both organisms — Fig-5's interventional sentence at n=2); the
CE_R canary ordering. WHAT DOES NOT: THE PUMP — organism 2's g-ray
FALLS immediately (every small-D rise negative on every ruler) —
the fact-positive ridge is one organism's biography, not the
physics; C5 rescopes. THE SIGN RUNG DRIFTS WIDE (2.63 vs the
[1.43,2.38] band): the front's sign-normalized edge is
lineage-sensitive where its magnitude cluster is not. THE THINK-
CARD THE AGENT OWED (adopted here): the ridge and the cliff may
not be the same object — the ridge is fact-positivity the wash can
harvest (present where the consolidation left the fact gradient-
aligned with the wash's useful directions); the cliff is the
direction the fact cannot survive (universal); organism 2's fact
consolidated WITHOUT the alignment, so no ridge, same cliff. THE
DISCLOSED SLIP: architecture co-varies with lineage (the family-2
root is 873k 4L, not the "same class" as e131's 2.74M 6L) — the
replicate doubles lineage, not architecture-at-fixed-lineage;
e193b (the fresh-root/two-fact cell) can pin the axis. THE RULER
LESSON: the e192-verbatim battery read 0.198 at the f2 ROOT —
under the kill bar at D=0 — a fact can be alive for its ruler and
dead for an imported one; co-rulers everywhere, adjudicated on
none.

"""
assert anchor in t and "## T153" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

c = "scratch/claims_ledger.md"
s = io.open(c, encoding="utf-8").read()
old = "| C5 | The pump: small displacement STRENGTHENS the fact under every optimizer and size; the pump-cliff is terrain in the g-direction (ridge 0.05-0.50, edge [0.80,0.92]) | opt1 A1-A3, opt1b/c, e191 | n=1 organism/fact/battery; no random-ray pump control until e192 |"
assert old in s, "c5"
s = s.replace(old, "| C5 | The pump: small displacement strengthens the fact on ORGANISM 1 ONLY (e193: absent on organism 2 — every ruler negative); the cliff is universal (order replicated n=2 lineages; the ridge is biography) | opt1 A1-A3, opt1b/c, e191, e192, e193 | pump n=1 organism; cliff/order n=2 (architecture co-varies, disclosed); the sign rung lineage-sensitive |", 1)
io.open(c, "w", encoding="utf-8").write(s)

p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = """Fig 5 (THE terrain figure — LICENSED by e192 as one picture): the
   fact-vs-displacement overlay on ONE organism/ruler/dual-currency"""
assert old in s, "fig5"
s = s.replace(old, """Fig 5 (THE terrain figure — LICENSED at n=2 organisms/lineages:
   e192 primary + e193 replicate; the order g < sign < random and
   the re-orientation rider replicate; the pump ridge does NOT
   cross lineages — one panel, two organisms): the
   fact-vs-displacement overlay, dual-currency""", 1)
io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e193 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e193 | THE SECOND-ORGANISM REPLICATE | DONE 17:30Z (T153: TERRAIN-REPLICATES fires — Fig-5's order at n=2 lineages (0.20 < 0.58 < >4.0; absolutes differ as registered); THE PUMP IS ABSENT on organism 2 (C5 rescopes to n=1; gradient-specificity survives inverted); the front's magnitude cluster lineage-stable, the sign rung drifts wide; the 1% rung kills (no floor); re-orientation causal at n=2; architecture co-varies, disclosed) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T17:32:00Z"
st["current_experiment"] = ("e193 FOLDED (T153: the order is lineage-physics, the pump is biography). Fleet: g2g2 + g1bS2 "
                            "running + e193b DISPATCHING (the critic's fresh-root/two-fact replicate — pins the "
                            "lineage-vs-architecture axis).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e193 folded")
