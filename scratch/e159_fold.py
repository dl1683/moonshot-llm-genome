import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E153 — phase-switch surgery:"
entry = """## E159 — coupled-or-organism: MASK-HEALS + SITE-SPARED — the double dissociation; READ-coupled replaces sink-coupled (2026-09-28 ~11:15Z) — DONE

WHAT WE DID: the two R46-critic probes — the mask+poison joint
cell on the consolidated net, and the first-ever norm ladder on
the site-stored control; gates to 1e-7; 86s.

WHAT WE SAW (T093): MASK-HEALS — joint 0.07-under-mask heals
COMPLETELY (g0 x1.030, g-12 x1.009 @ CE +0.036, numerically
identical to mask-alone; even COMPLETE wpe[0] removal under the
mask heals: the fact needs neither row-0 content nor its
value). Manipulation check: poisoning turns row 0 into a mass
absorber (pre-mask attention on key 0 explodes 0.157 -> 1.693,
10.8x) — the kill is delivered THROUGH attention reads, not
around them (query-side/global-softmax story FALSIFIED).
SITE-SPARED — arm_b's ladder: nothing dies (0.07: x0.922 @ CE
+0.825 — the SAME organism damage the consolidated net pays at
that bracket +0.843); even full removal costs its fact only
27.5%. THE DOUBLE DISSOCIATION: equal organism damage, only the
coupled memory dies. THE NOUN: READ-COUPLED — the consolidated
fact dies of what attention READS off a degraded row 0 (the
poisoned sink's reads corrupt downstream computation); the mask
is literally the health door; a memory must itself read the
sink to die of its poison. T086's organism-death bound is
itself bounded: organism damage is real but NOT SUFFICIENT.
Honesty: joint cell doubly off-distribution (CE prices generic
damage; anchored by gated single-intervention rebuilds); single
nets (e157 owes replication); per-net retentions (bracket
comparison is the like-for-like).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T092 — SYNTHESIS:"
t093 = """## T093 — E159: READ-coupled — the mask is the health door, and only readers die of the poison (2026-09-28 ~11:15Z)

The R46 critic's mask/ladder contradiction resolves into the
day's cleanest mechanism claim: the poison is delivered THROUGH
attention reads (the manipulation check nails it — a poisoned
row 0 becomes a 10.8x mass absorber whose reads corrupt
downstream computation), and masking key 0 removes the poison
entirely (the joint cell heals even under COMPLETE wpe[0]
removal). The query-side/global-softmax alternative is
falsified. And the site-stored control pays the same organism
price (CE +0.825 vs +0.843) without its fact dying: ORGANISM
DAMAGE IS NECESSARY BUT NOT SUFFICIENT — the memory must itself
read through the sink. THE NOUN, FINAL FORM (pending replication
and e158/e154): the consolidated memory type is READ-COUPLED to
the sink — it dies of what it reads off a degraded pivot, not
of the pivot's degradation per se. T086's removable-to-
irremovable reframe is bounded accordingly: the coordinate is
not unremovable (the mask removes it benignly!) — what is
irreplaceable by surgery is the READ PATH, and what training
built (access) surgery cannot create (e153) though it can sever
(e160). The four-layer model's LAYER 4 updates: dependence =
read-paths through the sink's health; the intervention class is
poisoning OR masking (both organism-tolerable; one kills
readers, one spares all).

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t093, 1)

# T086 bound update
o2 = """This is not a
retreat from the day's findings; it is their synthesis: jitter
moves the memory's dependence from editable tissue into
essential tissue — the OPPOSITE of surgical memory, and arguably
the point of consolidation (why replay-based systems consolidation
would produce trauma-resistant memory)."""
n2 = """This is not a
retreat from the day's findings; it is their synthesis — though
E159 BOUNDS IT FURTHER: the coordinate is not unremovable (the
mask removes row 0's contributions benignly — every fact
survives); what is irreplaceable by surgery is the READ PATH,
and only memories that read through the sink die of its
poisoning (the site-stored fact pays equal organism damage and
lives). The dependence is READ-coupledness, not organ-essentiality."""
assert o2 in t
t = t.replace(o2, n2, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- Report + paper ----------
r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
o_r = "while the READOUT consolidated into\na small head-set that surgery CAN remove (e160: {L1H0,L0H0}\n kills the memory at CE +0.25"
if o_r not in r:
    o_r = "while the READOUT consolidated into\na small head-set that surgery CAN remove (e160: {L1H0,L0H0}\nkills the memory at CE +0.25"
n_r = "while the READOUT consolidated into\na small head-set that surgery CAN remove (e160) — and the dependence\nitself is READ-coupled (e159: masking the sink heals a poisoned net\ncompletely while the site-stored fact pays the same organism damage\nand lives — only readers die of the poison; {L1H0,L0H0}\nkills the memory at CE +0.25"
assert o_r in r, "report anchor"
r = r.replace(o_r, n_r, 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "consolidation moves the\n    memory's dependence from surgically-deletable\ntissue into organism-critical tissue."
if o_p not in p:
    o_p = "consolidation moves the memory's dependence from surgically-deletable tissue into organism-critical tissue."
n_p = "consolidation moves the\n    memory's dependence into READ-coupledness with the sink (e159: the\n    double dissociation — equal organism damage, only the read-coupled\n    memory dies; the mask heals a poisoned net completely)."
assert o_p in p, "paper anchor"
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

# ---------- QUEUE ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e159 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e159 | coupled-vs-organism | DONE 11:15Z (T093: MASK-HEALS + SITE-SPARED — the double dissociation; poison delivered THROUGH reads (10.8x absorber), query-side story falsified; READ-COUPLED is the noun; organism damage necessary-not-sufficient) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 1: e152 (GPU, conversion trace). e159 DONE: READ-coupled — the double dissociation."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e159 fold complete")
