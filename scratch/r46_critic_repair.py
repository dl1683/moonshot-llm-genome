import json, datetime, re

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- DAY_SIX_REPORT: fix splice + honest closing-sentence form ----------
r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
o = """closing the geometry door globally; the cliff runs
both ways, and the read policy is its order parameter. (The self
waits at the door — e146 pending.)
to paper before their experiments ran (e143's proximity-vs-
invariance; e125's heads-ordering). The audit also repaired three
orphaned runs from earlier days (E019/E078/E088) and caught the
checkpoint-inventory misconception (102 nets on disk, gitignored)."""
n = """closing the geometry door globally [R46 critic: n=1 same-fact
cell; the different-fact and jitter-at-183 cells are queued —
"globally" is provisional]; the cliff runs
both ways, and the read policy is its order parameter [R46: a
metaphor e153 will test]. (e146 landed INSTRUMENT-INVALID — the
self-battery does not transfer to this line; the self question
parks pending a lineage-native rig.)
...two forks were committed
to paper before their experiments ran (e143's proximity-vs-
invariance; e125's heads-ordering). The audit also repaired three
orphaned runs from earlier days (E019/E078/E088) and caught the
checkpoint-inventory misconception (102 nets on disk, gitignored)."""
assert o in r, "splice anchor"
r = r.replace(o, n, 1)

# honest closing-sentence form (critic attack 4)
o2 = """and consolidation is the movement
from the removable to the irremovable."""
n2 = """and consolidation is the movement
from the removable into machinery whose CORRUPTION is
organism-fatal — whether its readout is surgically attackable
is the open frontier (the L0H3 head-ablation near-miss, 58.6%
at CE +0.21, is one cell from deciding it)."""
assert o2 in r
r = r.replace(o2, n2, 1)

# ledger precision (critic: direct800 passes; no-width-trend hides NR falloff)
r = r.replace("direct installs row-129-null", "direct installs row-129 essentially-null (direct800 passes at 0.007)")
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

# ---------- NOTES: strip leading empty separators ----------
n = open("NOTES.md", encoding="utf-8").read()
head, sep, rest = n.partition("\n---\n")
# collapse any run of pure '---' lines at the top after the header block
lines = n.split("\n")
i = 0
while i < len(lines) and (lines[i].strip() == "" or lines[i].strip() == "---"):
    i += 1
# keep any real header (first non-sep line is likely '# ...')
if i > 0 and i < len(lines):
    n = "\n".join(lines[i:])
    open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING: T087 wording bound + T088 fixed-architecture ----------
t = open("THINKING.md", encoding="utf-8").read()
o3 = """THE ROADS DIVERGE IN OPPOSITE DIRECTIONS FROM THE FIRST RUNG:"""
n3 = """MAGNITUDE BOUND (R46 critic — the honest form): the step's
core is the SIGN and the NR onset (both ~10-20x their controls);
"dies" and "no width trend" overran the numbers — the w64 A
endpoint is 20% of w8's (5x rung scatter, single seed, zero
cross-seed variance on this dial), and NR falls 0.903 -> 0.605
from w8 to w64 (W010's ghost half-alive). Bound: the key's
positive load (<= +0.327) is abolished; what replaces it is a
WEAK negative (|A| <= 0.13), sign-robust, magnitude unresolved
pending 2-3 re-seeds.

THE ROADS DIVERGE IN OPPOSITE DIRECTIONS FROM THE FIRST RUNG:"""
assert o3 in t
t = t.replace(o3, n3, 1)

o4 = "bidirectionally switchable at fixed wiring"
n4 = "bidirectionally switchable at fixed architecture (both directions required 300 AdamW steps — NOT same-weights)"
t = t.replace(o4, n4)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- PAPER SKELETON: the critic's three edits ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
p = p.replace("bidirectionally switchable at fixed wiring", "bidirectionally switchable at fixed architecture (both directions are 300 trained steps)")
p = p.replace('Working title: **"Consolidation follows error placement: routed and site-stored memories in a tiny language model"**',
              'Working title: **"Consolidation follows error placement: sink-coupled and site-stored memory phases in a tiny language model"**')
p = p.replace("SITE-STORED\n(content concentrated at a row", "SITE-STORED\n(content concentrated at a row")
p = p.replace("(3) Content\n    never moves: all states store in body organs and fact-specific heads;\n    \"migration\" is read-policy re-routing, the destination row carrying no\n    written key (install-restore is a no-op; direction-scramble spares the fact\n    while costing the LM 0.70 nats).",
              "(3) Content\n    never moves: all states store in body organs and fact-specific heads\n    (head share from the locality-filtered report-only table); \"migration\" is\n    access-redistribution, the destination row carrying no written key (the\n    evidence is the direction-perm rider, norm-preserved: +4.1% at g0 — itself\n    CE +0.70, and read-horizon-dependent: short-horizon reads DIE under perm;\n    install-restore was a no-op-by-norm and carries no evidential weight).")
o_fig = "e143's numbers already fill the core 3x4:\n0.278/0.002/0.003/neg; 0.238/0.205/0.156/~0; 0.722/0.914/0.903/brake."
if o_fig in p:
    p = p.replace(o_fig, "e143's numbers fill the core 3x4 (brake column\nCORRECTED per R46 audit — was transposed):\nNEAR 0.278/0.002/0.003/~0; FAR 0.238/0.205/0.156/-0.233; JITTER 0.722/0.914/0.903/brake(-0.132).")
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

# ---------- QUEUE: the missing cells as top rows ----------
q = open("QUEUE.md", encoding="utf-8").read()
o_q = "| e153 | PHASE-SWITCH SURGERY"
rows = """| e158 | THE 2x2 COMPLETION (R46 critic attack 1 — THE missing cell: variance x site) | TOP PRIORITY — next GPU slot (before e154) | (a) JITTER@183: e151's protocol verbatim with ±1-8 offsets around 183 on the consolidated root — phase claim predicts geometry door STAYS OPEN (g-12 >= 0.5); placement-closure predicts it shuts (~0.1). (b) LOCKED@BAND: locked replay at the ORIGINAL band on the consolidated root — isolates zero-variance from new-site. Bars: PHASE-BY-VARIANCE = (a) open + (b) shut; CLOSURE-BY-PLACEMENT = (a) shut; SITE-INDEPENDENT = (b) stays open (locking the home site is harmless — the door keys on novel-site teaching) |
| e159 | COUPLED-OR-ORGANISM (R46 critic attack 2 — the two cheap reconcilers) | TOP PRIORITY — CPU eval-only, minutes | (a) MASK+LADDER JOINT: norm 0.07 UNDER the mask — heals (poisoning carried by attention reads; information sneaks back through the health door) or still kills (query-side global-softmax collapse; the reframe hardens). (b) SITE-STORED LADDER: norm ladder on e131_arm_b — dies at the same bracket (sink-poisoning is organism death; 'COUPLED' is a misnomer) or survives (coupling is real) |
| e160 | THE HEAD-SET ESCALATION (R46 critic's final line — one cell from the paper's best figure or its retraction) | DISPATCHED 10:45Z (CPU eval-only) | L0H3-zero was single-head 58.6% @ CE +0.21 (1.4pts under bar). Escalate: L0H3 + top-2 and top-3 fact-specific heads (e133's list, CE<=0.35 class), zero and mean-replace, singly and jointly, graded; co-measure CE + base skills per cell. Bars: FLAT-CE-FACT-KILL = some head-set >=60% fact drop at CE <= +0.35 (the surgical surface EXISTS — paper's best figure; the removable->irremovable sentence inverts to 'attackable by head coordinates'); NEAR-MISS-CONFIRMED = best set 40-60% (the frontier stays open); WRECK-ONLY = every >=60% cell costs CE >= +0.70 (the reframe holds; noun architecture survives bounded) |
| e147R | E147 RE-SEEDS (the A-dial error bar; ~10 min GPU) | QUEUED | 2-3 ARM_SEED replicates at w1 and w64: put error bars on the w1 crossing and the plateau; W010's-ghost check on the NR falloff (0.903->0.605) |
""" + o_q
assert o_q in q
q = q.replace(o_q, rows, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["last_review"] = now
s["last_novelty"] = now
s["current_experiment"] = "Fleet 3: e152 (GPU) + e153 (CPU) + e160 (CPU, head-set escalation — the one-cell-away experiment). e146 DONE INVALID; e158/e159 top-queued."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("critic repairs + missing cells queued")
