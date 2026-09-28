import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E142 — row-0 at birth:"
entry = """## E150 — the flat-CE route test: ALL-KILLS-WRECK — 'routed' was never information flow; the kill is POISONING, and the reframe activates (2026-09-28 ~09:35Z) — DONE

WHAT WE DID: five probes, CE on every cell, mask instrument gated
bit-exact vs the standard forward; 57s CPU.

WHAT WE SAW (T086): ALL-KILLS-WRECK fires — all 4 killing cells
cost CE >= +0.70 (norm0@g0 93.2% @ +1.40; norm0@g-12 98.8% @
+1.40; norm0.07@g-12 84.2% @ +0.84; perm@col12 95.8% @ +0.705).
FLAT-CE-ROUTE does not fire (0/23 bar-eligible cells). THE
MECHANISM UNDERNEATH: the forced-off-sink mask (all attentional
access to position 0 blocked) SPARES the fact at the flattest CE
of any row-0-plane intervention ever measured (retention x1.009-
1.030 @ +0.03, all three net types incl. controls) — d_r0's kill
was never information flow from row 0. Coherent reading: a
low-norm wpe[0] leaves key 0 as a VALUE-LESS MASS ABSORBER that
corrupts downstream reads (poisoning); masking refunds the mass.
The presence threshold lands in (0.07, 0.15) — far sharper than
e141's (0.066, 0.382); pre-mask sink attention mass at the read
position is only ~0.007/layer. PRESENCE-AT-NOVEL fails by 0.8pt:
perm@g-12 costs 16-21% (x0.79-0.84) — the fact MILDLY consults
row-0's direction at novel geometry (unlike +4.1% spare at g0);
and P4's col-12 control DIED under perm (x0.042) — short-horizon
reads consult row-0's direction, the 129-read does not (read-
horizon texture). DIRECTION-CONSULTED does not fire: fact-at-
position-0 scramble IMPROVES the fact (x1.589, weak 0.23 base) —
W014 survives its control. L0H3-zero: 58.6% drop at CE +0.21 —
NEAR-MISS, 1.4pts under the kill bar, report-only; top-3-mean
joint x0.068 @ +0.71 (post-hoc rider). Honesty: mask is off-
distribution but CE +0.03 prices generic damage ~0 and the
site-stored control survives (x0.995); single-seed cells; P4
power-limited (base 0.23).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## READING MAP"
t086 = """## T086 — E150: the route was never information — sink-HEALTH, poisoning, and the removable-to-irremovable reframe (2026-09-28 ~09:35Z)

The reading map's ALL-KILLS-WRECK branch fired, and the
pre-registered reframe is the fold — but e150 added a MECHANISM
the map did not anticipate:

**(1) THE ROUTE WAS NEVER INFORMATION FLOW.** Blocking ALL
attentional access to position 0 (the mask) spares the fact at
CE +0.03 — across all three net types. What kills under wpe[0]
removal is POISONING: below-norm row 0 becomes a value-less mass
absorber that corrupts every downstream read (the sink's health,
not its signal). The threshold is sharp: norm 0.07 kills (CE
+0.84), 0.15 survives (CE +0.31). The pre-mask sink attention at
the fact's read position is ~0.007/layer — the 'route' carried
essentially no fact traffic. T081/T082 are HARD-BOUNDED as
registered: 'routed through row-0 presence' is now SINK-HEALTH
DEPENDENCE — the memory requires the organism-critical
coordinate to be intact, the way any organ requires the blood
supply.

**(2) THE REFRAME, ACTIVATED WITH MECHANISM: consolidation is a
movement from the REMOVABLE to the IRREMOVABLE — into machinery
whose integrity is organism-critical.** The graft (row 129) was
surgically deletable at zero collateral; the consolidated memory
rides the sink whose poisoning wrecks everything. This is not a
retreat from the day's findings; it is their synthesis: jitter
moves the memory's dependence from editable tissue into
essential tissue — the OPPOSITE of surgical memory, and arguably
the point of consolidation (why replay-based systems consolidation
would produce trauma-resistant memory).

**(3) WHAT SURVIVES OF THE TAXONOMY:** the TYPES are real but
renamed — SITE-STORED vs SINK-COUPLED (was 'routed'): the type
differences (novel-geometry 0.914 vs 0.002; D-all tolerance; the
brake sign) stand on e139/e143's cells; the CARRIER of the
sink-coupled type's geometry-generalization is now most plausibly
the fact-specific HEADS (e133's 84.5% residue) reading content —
the content-keyed alternative W011 buried gets its revenge. The
L0H3-zero near-miss (58.6% at CE +0.21, 1.4pts under bar) is the
leading candidate for the true fact-circuit — one cell away from
the flat-CE kill the frame wanted.

**(4) W014 SURVIVES ITS CONTROL, WITH TEXTURE:** fact-at-position-
0 scramble IMPROVES the fact (x1.589); direction-insensitivity
at the 129-read holds; but perm kills short-horizon reads
(col-12 control x0.042) — direction-consultation is READ-HORIZON
dependent. The tenant's insurance-policy metaphor gains a clause:
the fact ignores the pivot's direction ONLY from far away.

STANDING QUESTIONS: e147 (in flight) now measures the SWITCH
between types without a route mechanism to explain it — if
INVARIANCE-CAUSAL fires, the switch is real and the mechanism
hunt reopens at the head level; e146's matrix gains sharper
probes (mask vs poison vs perm columns — the self's row can now
distinguish health-dependence from information-dependence); the
P-b cell (one net, both types) rises in priority as the
taxonomy's last confound.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t086, 1)

# W014 amendment
o_w = "## W014 — WONDER: the memory layer is a semi-independent tenant [R45 caveat: the never-consults null is alive"
n_w = "## W014 — WONDER: the memory layer is a semi-independent tenant [e150: SURVIVED its control (fact-at-position-0 scramble improves the fact x1.589); texture added — direction-consultation is read-horizon-dependent (short-horizon reads die under perm, the 129-read does not); R45 caveat"
assert o_w in t
t = t.replace(o_w, n_w, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e150 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e150 | flat-CE route test | DONE 09:35Z (T086: ALL-KILLS-WRECK — no flat-CE fact-kill; the kill is POISONING (mask spares fact at CE +0.03; threshold (0.07,0.15)); reframe activated: consolidation = removable->irremovable, riding organism-critical sink health; taxonomy renamed SINK-COUPLED; L0H3-zero 58.6%@+0.21 near-miss = leading fact-circuit candidate; W014 survives with read-horizon texture) |\n" + q[m.end():]
# e146 gains the sharper probes
o_q = "{row-0 presence-removal, row-0 direction-scramble, fact-specific-head ablation} x {fact expression, self-recognition"
n_q = "{forced-off-sink MASK (flat-CE, spares fact — information probe), norm POISON ladder 0.07-0.15 (health probe), direction-PERM (consult probe), fact-specific-head ablation} x {fact expression, self-recognition"
assert o_q in q
q = q.replace(o_q, n_q, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE + report + paper ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 1: e147 (GPU, width ladder — now the type-switch measurement sans route mechanism). e150 DONE: ALL-KILLS-WRECK + poisoning mechanism + reframe."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)

r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
o_r = """BOUND (R45 critic, held open): every fact-killing
   intervention in the row-0 plane sits at CE +0.70 to +4.44 —
   no flat-CE fact-kill exists yet, so 'routed' vs 'dies when the
   net dies' is not fully separated until e150's cells land; the
   install-restore probe was also a no-op by norm (the presence
   conclusion rests on the perm/halfnorm/mean riders)."""
n_r = """RESOLVED by e150 (T086): ALL-KILLS-WRECK — and the
   mechanism underneath is POISONING, not routing: masking all
   attention to position 0 spares the fact at CE +0.03, while
   sub-threshold row-0 norm corrupts every read (threshold in
   (0.07, 0.15)). 'Routed' becomes SINK-COUPLED — the memory
   requires the organism-critical coordinate's HEALTH. The day's
   final reframe, pre-registered before the data: consolidation
   is a movement from the REMOVABLE to the IRREMOVABLE — the
   graft was surgically deletable; the consolidated memory rides
   essential tissue. Arguably the point of consolidation."""
assert o_r in r
r = r.replace(o_r, n_r, 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "(4) [pending e150] the memory layer's\n    failure modes are opposite the LM's — presence-typed vs direction-typed —\n    making head-level ablation the only surgical unlearning surface, with the\n    old address a suppressive brake after routing."
n_p = "(4) sink-coupling and the removable-to-irremovable\n    movement (e150): no flat-CE fact-kill exists — masking all attention to\n    position 0 spares the fact at CE +0.03 while sub-threshold row-0 norm\n    poisons every read (threshold in (0.07, 0.15)); consolidation moves the\n    memory's dependence from surgically-deletable tissue into\n    organism-critical tissue. L0H3-zero (58.6% drop at CE +0.21) is the\n    leading fact-circuit candidate for the surgical-unlearning surface."
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)
print("e150 fold complete")
