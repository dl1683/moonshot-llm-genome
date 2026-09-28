import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E160 — the head-set escalation:"
entry = """## E153 — phase-switch surgery: TEXTURE — the order parameter is DISTRIBUTED (MLP-heavy), not any small head set; the geometry door is transplant-RIGID (2026-09-28 ~11:05Z) — DONE

WHAT WE DID: wiring diff + K-by-K head transplants (fact /
reader / mover / random classes, K=1/3/6) both directions
between the two phase nets; every cell CE-priced (all |dCE| <=
0.024 — nothing is wreckage); gates bit-exact.

WHAT WE SAW (T091): PHASE-IN-HEADS no (best reopen g-12 0.151 vs
bar 0.45 — the reader_K3 arm {L3H5,L0H3,L1H0}, +47.5%; rand_K6
also +32% — the site-phase net is fragile, movement not
specific). PHASE-DISTRIBUTED no as written (>20% movers exist —
head identity matters, partially). THE STORY: the 300 locked
steps' delta lives in MLPs (66.5% of ||d||^2) and late heads
(all six L5 heads top movers) — but transplanting the biggest
movers moves the door <4%; the door-movers (the RE-GROWN
POSITIONAL READER L3H5 — the L3H4-class namesake, one slot from
the twin's own — plus L0H3/L1H0 content heads) are mid-ranked
deltas that reopen under half the distance and never cross the
bar. ASYMMETRIC RIGIDITY: the open geometry door is transplant-
immovable (nothing closes it, best -9.1%); the shut door is
nudgable to ~1/3 of the bar. The reader set partially carries
the BRAKE both ways (A +0.005 -> -0.027; -0.132 -> -0.068 at
K6). Honesty: parameter-swap path dependence (donor heads land
in foreign LN contexts, deltas 0.40-0.71 — nulls conflate
'phase not in heads' with 'head off-manifold'; mitigated by
no-op/random controls and near-zero CE); single lineage, one
conversion draw.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T090 — E160:"
t091 = """## T091 — E153: doors open by training and resist surgery — the phase is a distributed, MLP-heavy state with a partial head signature (2026-09-28 ~11:05Z)

The reading map's PHASE-DISTRIBUTED branch fired in substance
(though not its letter — five arms move g-12 > 20%, so "no
head-set moves it" is false; the honest form: NO head-set
SWITCHES it). Three structural facts:

(1) T037's WRITE-ONCE CORE SURVIVES ITS SHARPEST TEST: no
non-gradient write added function — the best transplant
(reader_K3, the re-grown L3H4-class reader + the content heads)
recovers under half the geometry door, and random swaps move the
fragile net comparably. Surgery cannot open what training opens.
The e160 result completes the asymmetry: surgery CAN destroy the
readout (N2 kills at flat CE) but cannot CREATE access. The edit
law holds: subtraction works, addition doesn't.

(2) ASYMMETRIC RIGIDITY is the new texture: the OPEN door
(geometry phase) is transplant-immovable — once wired by
variance training, access resists surgery; the SHUT door
(site phase) is fragile. Rigidity tracks the phase, not the
heads: consolidation, once achieved, is surgically stable —
another sense in which it is a movement into essential tissue.

(3) THE ORDER PARAMETER'S SHAPE: MLP-heavy (66.5% of delta
energy), late-layer-skewed, with a partial head signature (the
re-grown positional reader + content heads — the SAME heads that
kill the fact in e160's N2/E2 sets appear in the best reopening
arm). One circuit, three roles: it carries the readout (killable
— e160), partially carries re-opening (transplantable to a
third — e153), and partially carries the brake. The phase itself
sits above it, in the stream.

FOR THE PAPER: Fig 4 becomes the asymmetry figure (doors: opened
by training, killed by head surgery, immovable by transplant);
claim 2's mechanism section states the distributed order
parameter honestly. The e158 cell (jitter@183) still decides
variance-vs-placement before any of this travels.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t091, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e153 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e153 | phase-switch surgery | DONE 11:05Z (T091: TEXTURE — order parameter DISTRIBUTED (MLP-heavy 66.5%, late-skewed); no K<=6 transplant switches the phase (best reopen +47.5% sub-bar, rand +32%; nothing closes, best -9%); geometry door transplant-RIGID; re-grown reader L3H5 + content heads = partial signature, same heads as e160's kill sets; T037 write-once core survives: surgery kills but cannot create access) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e152 (GPU, conversion trace) + e159 (CPU, coupled-vs-organism — dispatching). e153 DONE: order parameter distributed; doors resist surgery."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e153 fold complete")
