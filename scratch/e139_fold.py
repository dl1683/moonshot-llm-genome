import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E141 — sink-key mechanism battery:"
entry = """## E139 — row-0 universality: HYBRID (site-dominant two-door) — row 0 is NOT universal; row 0 routes, the site stores (2026-09-28 ~08:15Z) — DONE

WHAT WE DID: five probes on the splice arms + consolidated
reference; arm-c regenerated (the one permitted training, 300
steps); all gates bit-exact (|d|=0.0 vs e131 tables); 1059s CPU.

WHAT WE SAW (T082): both splice arms HYBRID with the site
dominant — D-row-0 -24.8%/-27.5% (below the -50% universal bar),
D-183 -54.4%/-30.9% (below the -80% site-locked bar), both doors
together -78%/-62% (super-additive). Row-183 CONTENT-POSITIVE
(strength 0.455/0.271, ~1000x control band — the only strong
content row in the census); row-0 content test NULL on both arms.
GENERALIZATION (e131's honesty note d closed): both arms
generalize — novel train contexts 0.597/0.708, val-split
0.591/0.699, held-out fact segments 0.620/0.683 — real facts,
not window memorization; training-geometry premium ~0.3 (e131's
0.989 overstated strength; the training read AND generalization
were both true). BRAKE ABSENT on splice arms (row-129 replacement
~0) vs the consolidated line's -0.11/-0.13 — the brake is a
re-keying scar of the jitter road and does not reach across
homes (corroborates T078). CONSOLIDATED REFERENCE (report-only):
the jitter-road net reads p(Z)=0.660 at the 183-geometry it
NEVER trained, row-0-keyed there (drops +0.657/+0.551, ratio
0.84) — row 0 is a position-invariant readout route for the fact
that re-keyed to it. ARM-C RIDER: RIDER-NULL — genuine decay,
not instrument blindness: the dreams' 34 ZEPHYRAs sit at
x-col 130 (33/34; read position 129 = the OLD address — dreams
are never position-diverse); p(Z)@onset 0.230 BELOW the base's
0.391 at the same positions (dream replay actively ERODED the
fact there); consolidated nothing anywhere readable. T076's
error-compass survives its last open edge; e120's arm-c verdict
stands un-revised. Honesty: exposure spans excluded exactly;
D-row-0's CE +1.28 bounded by row-1 control (negligible
scaffold); ~25% drop is row-0-specific but NOT content (the
content test says not) — processing gate, not store.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T081 — E141:"
t082 = """## T082 — E139: two memory types — ROUTED vs SITE-STORED — and the dreams that dream in coordinates (2026-09-28 ~08:15Z)

The taxonomy completes, and it is cleaner than any card
predicted. The splice arms (error at a fixed novel site) produce
SITE-STORED memories: row-183 content-positive at ~1000x
controls, D-183 kills half, row-0 NULL by content test — and
these memories GENERALIZE to novel contexts and val-split
(0.6-0.7) despite being site-stored. The jitter road (error
position-varied) produces ROUTED memories: row-0 presence-
keyed (e141), reading at geometries never trained (0.660 at the
183-geometry, ratio 0.84 row-0 drops) — position-invariant
access to body-stored content (e133: 84.5% head residue). ROW 0
ROUTES; THE SITE STORES. The two doors compose super-additively
in the splice arms (D-row-0+183: -78%/-62%) — a splice memory is
mostly site-read with a weak routed tail (-25%).

CONSEQUENCES: (1) T075's provisional marker RESOLVES — the
retirement stands (the 'position diversity ingredient' framing is
dead; the splice arms learned, stored, and generalized at their
error site), and the credit for what position diversity ACTUALLY
does moves fully to T079: diversity decides WHICH TYPE of memory
forms (routed vs site-stored) by deciding which features are
invariant across the error windows. (2) The brake's scope is now
precise: a scar of RE-ROUTING (absent on splice arms, present on
the jitter line — deleting the old address only helps a memory
that moved its route). (3) W011's savor (a) resolves PARTIAL:
site-stored facts DO generalize across CONTEXTS — what was never
tested is novel-GEOMETRY for the splice arms (fact shifted off
183); the routed fact generalizes across geometries (0.660 at
never-trained 183). Novel-geometry-for-site-stored is the one
missing cell in the taxonomy; prediction: it fails or degrades
steeply (a site-stored read needs its site), which would make
geometry-independence the ROUTED type's exclusive property.

THE DREAMS THAT DREAM IN COORDINATES (the arm-c rider, the
day's best savor): 33 of 34 dream ZEPHYRAs sit at x-col 130 —
read position 129, the OLD ADDRESS. The net's spontaneous
replay visits its fact in the fact's own coordinates; dreams are
never position-diverse. Under T079 this is exactly why verbatim
dream replay cannot consolidate (zero position variance -> the
positional key keeps the credit -> nothing re-routes), and the
rider adds the sharper number: dream replay left the fact BELOW
base at the dream positions (0.230 vs 0.391) — replay without
error doesn't just fail to consolidate, it ERODES. T074's dream
verdict stands; T076's compass survives its last open edge; and
e136's surprisal prediction now has a mechanism to explain the
500x: dreams protect by SLOWING EROSION at the address they
never leave, not by moving anything.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t082, 1)

# T075 provisional marker resolves
o_t = "## T075 — [RETIRED-PROVISIONAL per R44 critic: retirement announced on probe-1 (learning-at-183) before the D-183 graduation cell; e139 adjudicates]"
n_t = "## T075 — [RETIRED, RESOLVED by e139/T082: the splice arms learned, stored (row-183 content ~1000x), and generalized (0.6-0.7) at their error site — retirement stands; what position diversity actually does (choose routed vs site-stored) belongs to T079]"
assert o_t in t
t = t.replace(o_t, n_t, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE.md ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e139 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e139 | row-0 universality + 183-robustness | DONE 08:15Z (T082: HYBRID site-dominant — row 0 NOT universal; row-183 content ~1000x (site stores); arms generalize 0.6-0.7 novel+val (note d closed); brake absent on splice (re-routing scar); consolidated net reads 0.660 at never-trained 183 via row 0; arm-c RIDER-NULL — dreams sit at the OLD address 33/34, dream replay ERODES below base) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE.json ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 1: e140 (CPU, route-dependence trace + L-CYCLED). e139 DONE: HYBRID — row 0 routes, the site stores; dreams dream in coordinates."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e139 fold complete")
