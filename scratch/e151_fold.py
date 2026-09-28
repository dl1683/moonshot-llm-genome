import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E147 — the width ladder:"
entry = """## E151 — the P-b cell: ROUTE-OVERWRITES — the cliff is PER-NET; locked re-teaching converts the memory to site-only and closes the geometry door GLOBALLY (2026-09-28 ~10:10Z) — DONE

WHAT WE DID: one GPU re-teach (e143's locked-replay protocol,
300 steps) of the consolidated net's fact at read rows 183-189;
full before/after battery; root gated bit-exact.

WHAT WE SAW (T088): the committed TWO-DOOR prediction FAILED;
T087's fork resolves PER-NET. Before -> after: the 183 SITE
GREW (row-183 strength -0.007 -> +0.512, ratio 0.96; D-183 now
kills half: 0.998 -> 0.486) while THE ROUTE DOOR CLOSED (g-12
0.916 -> 0.102; g+12 0.948 -> 0.121; D-all 0.905 -> 0.098; the
brake vanished: A -0.132 -> +0.005; row-0 S 0.732 -> 0.106 with
base collapsed alongside, rel ~0.99). NOT WRECKAGE: CE_R IMPROVED
(1.6635 -> 1.6490); mask spares both nets (x1.00/x1.06 @ +0.04 —
e150 replicated); ladder poisons both (0.07 kills @ +0.99). THE
TRANSIENCE CLUE: the 8-step smoke retained g-12 at 0.99 — the
conversion is budget-dependent, living somewhere in 8-300 steps;
a transient two-door state may exist before the cliff re-runs.
Honesty: single seed/lineage (n=1 fork verdict); 300-step budget
matches the original consolidation but the conversion point is
unlocated; FAR-class placement confound (184-token pre-context)
entangled with locked-variance exactly as e143's FAR; site bar
weak at this readout (controls ~0) — growth carried by absolute
census + D-183 necessity.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T087 — E147:"
t088 = """## T088 — E151: the cliff is PER-NET — memory type is a global phase of one substrate, and the transition runs BOTH WAYS (2026-09-28 ~10:10Z)

The committed prediction failed honestly and the failure is the
day's cleanest structural statement: ONE locked re-teach
converted a sink-coupled, geometry-general, deletion-tolerant
memory into a site-stored, geometry-bound one — growing the new
site (+0.512) while closing the geometry door EVERYWHERE (g-12
0.916 -> 0.102, D-all 0.905 -> 0.098) and dissolving the brake
(-0.132 -> +0.005), at IMPROVED corpus CE. Consequences:

(1) THE TYPES ARE PHASES, NOT SUBSYSTEMS. "Two memory types"
becomes "two phases of one memory system" — the taxonomy's
lineage confound (R45's attack) resolves the strong way: one
net, both phases, bidirectionally switchable (variance opens the
geometry door and kills the graft; zero-variance re-teaching
grows a graft and closes the door). For the paper this is
STRONGER than two-doors would have been: a reversible phase
transition, cliff both ways, at fixed wiring — the read policy
(W013's protagonist) is the order parameter.

(2) WHY GLOBAL? The geometry-generalization cannot be a property
of one fact's private circuit (a private circuit would survive
another fact's locked training); it must live in a SHARED state
— the net's read configuration as a whole. Locked replay at ANY
site drags the global policy back toward position-keyed reading.
The 8-step smoke (g-12 still 0.99) says the drag has a TIMESCALE
— the conversion lives somewhere in 8-300 steps.

(3) THE TRANSIENT: if a two-door state exists mid-conversion,
the P-b prediction is not wrong but EARLY — doors add
transiently, then the zero-variance training consolidates the
graft and the shared state abandons the geometry door. The
TIME-TRACE (g-12 retention vs re-teach steps, 8/16/32/64/128/300
— dispatched as e152) locates the conversion point and tests
whether the transient is real: monotone decay (no plateau) =>
clean conversion; a plateau with both doors followed by decay =>
the transient two-door state exists and the cliff has a dwell
time.

(4) THE BRAKE VANISHES WITH THE PHASE: A(129) -0.132 -> +0.005 —
the negative posterior (T087's mechanism note) is not a permanent
scar but a STATE of the variance phase; re-locking releases it.
e149's anti-alignment prediction sharpens accordingly: the
anti-alignment should exist only in the variance phase.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t088, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE ----------
q = open("QUEUE.md", encoding="utf-8").read()
o_q = "| e148 | DREAM-TOPOLOGY CENSUS"
row = """| e152 | THE CONVERSION TIME-TRACE (T088's central open number: where in 8-300 steps does the phase convert? does a transient two-door state exist?) | DISPATCHED 10:10Z (GPU free; short trainings) | from e131_consolidated: locked re-teach at 183 for steps in {8, 16, 32, 64, 128, 300} (checkpoints each); measure g-12/g+12 retention, 183-site content, A(129), D-all per checkpoint. Bars: CLEAN-CONVERSION = g-12 monotone decay, no plateau (site growth anti-correlates, Spearman <= -0.8); TRANSIENT-TWO-DOOR = a checkpoint where BOTH site content clears the bar AND g-12 retention >= 0.5, followed by decay (the cliff has a dwell time; P-b was EARLY, not wrong); DELAYED-CONVERSION = g-12 flat until a step threshold then cliff (a critical mass of zero-variance steps) |
""" + o_q
assert o_q in q
q = q.replace(o_q, row, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE + report + paper ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e146 (CPU, dissociation matrix) + e152 (GPU, conversion time-trace). e151 DONE: ROUTE-OVERWRITES — per-net cliff, phases not subsystems."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)

r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
o_r = """(The self waits at the
door — e146 pending.)"""
n_r = """Postscript (e151): the
types are PHASES, not subsystems — one locked re-teach converted
the sink-coupled memory to site-stored (g-12 0.916 -> 0.102) at
improved CE, closing the geometry door globally; the cliff runs
both ways, and the read policy is its order parameter. (The self
waits at the door — e146 pending.)"""
assert o_r in r
r = r.replace(o_r, n_r, 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "switched by a BINARY CLIFF at zero-vs-any error-position variance (e147: ±1 suffices"
n_p = "switched by a BINARY CLIFF at zero-vs-any error-position variance — and the types are PHASES of one substrate, bidirectionally switchable at fixed wiring (e151: one locked re-teach converts sink-coupled to site-stored, g-12 0.916->0.102, at improved CE) (e147: ±1 suffices"
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)
print("e151 fold complete")
