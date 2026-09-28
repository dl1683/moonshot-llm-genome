import json, datetime, re

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E161 — the freeze-cell:"
entry = """## E152R — the dwell re-seeds: DWELL-SEED-DEPENDENT — the dwell dies at n=3; the brake overshoot survives 3/3; the straddle cell is unstable (2026-09-28 ~15:00Z) — DONE

WHAT WE DID: two full conversion traces (seeds 10903/10904) +
the 10/12/14/16 insert (folded into 10903's trajectory) + the
straddle settler (locked@band at seeds 10905/10906); 57.5 min
GPU (gate-checked per training, never parked); all gates
bit-exact.

WHAT WE SAW (T102): DWELL-REPLICATES does NOT fire — the cliff
brackets are 8->16 / 16->32 / 128->300 across seeds; shelf
minima 0.538/0.213/0.854; T094's "8-16-step cliff" and "~50-
step dwell" are TRAJECTORY-SPECIFIC TEXTURE. WHAT SURVIVES 3/3:
(1) the CONVERSION itself (final <= 0.27 at every seed — the
door always closes by s300); (2) the BRAKE OVERSHOOT (A(129)
deepens below -0.35 mid-conversion in 3/3, deepening further
with later cliffs; released by s300 — now a real n=3
phenomenon). THE INSERT: the door stays fully open through s16
on 10903 (1.036 at s12) — timing seed-dependence at fine
granularity. THE STRADDLE SETTLER (outside its fork):
locked@band = 0.261/0.317 at new seeds vs 0.546 CPU / 0.458 GPU
— the cell SPANS OPEN-TO-SHUT across seeds; home-locking alone
CAN shut the door at some seeds; T097's two-factor gate
WEAKENED on its home leg (the honest reading: an UNSTABLE
cell, not a SHUT cell). UNDER THE DISUSE FRAME (T101): the
conversion timing is a WASHOUT RACE — when ordinary gradient
flow happens to beat the re-teach on a given trajectory; the
robust brake overshoot is the surviving mechanism signal.
Honesty: no device mixing this run (all GPU) but the seed-10902
baseline was CPU — cross-experiment drift priced; sequential
trajectories (n=3 over paths); one lineage; texture cells not
re-run (bars only).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING ----------
t = open("THINKING.md", encoding="utf-8").read()
anchor = "## T101 — E161:"
t102 = """## T102 — E152R: the dwell dies, the brake lives — and the disuse frame absorbs the wreckage (2026-09-28 ~15:00Z)

The replication did its job: the session's most-quoted texture
(the dwell, the 8-16 cliff) is trajectory-specific — three
seeds, three shapes, timings spanning an order of magnitude.
What the replication CONFIRMS is better: the conversion is
INVARIANT (3/3 complete by s300) and the brake overshoot is
REAL (3/3 seeds, deepening with later cliffs, always released).
Under T101's disuse frame, both fall into place: the conversion
is a WASHOUT RACE (ordinary gradient flow vs the re-teach, on
each trajectory's luck — hence seed-dependent timing but
invariant outcome), and the brake overshoot is the old address
RESISTING at maximum exactly when the washout is winning —
suppression peaks at maximum competition (the negative-
posterior reading, now n=3). THE STRADDLE CELL (locked@band
0.26-0.55 across seeds/devices) also dissolves into the frame:
home-locking sometimes loses the race, sometimes wins it — an
unstable cell, not a gate leg. T097's two-factor gate is now
SUGGESTIVE ONLY (novel-site + zero-variance is where the race
is usually lost fastest; e165's ladder remains the axis test).

FOR THE PAPER: the mixed-state sentence softens to "a
mid-conversion state holding both natures was OBSERVED (one
seed) but is not a timescale law; conversion completes at all
seeds within 300 steps"; the brake overshoot enters as an n=3
finding. W018's four fates lose their timing claims (already
barred from paper text). The [n=1] markers on T094 clear —
resolved NEGATIVE for the dwell, POSITIVE for the overshoot.

""" + anchor
assert anchor in t
t = t.replace(anchor, t102, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- paper ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "the conversion PASSES THROUGH A MIXED STATE: the cliff\nfires in 8-16 steps, then a ~50-step dwell holds BOTH natures (site-store\ngenuine at 66.8x the 2x-control bar across dwell peaks\nAND >=50% geometry retention (derivation stated; n=1 until e152R) before separation\ncompletes (e152)"
n_p = "conversion completes at ALL seeds within 300 steps (e152R, n=3),\nwith seed-dependent trajectories (cliff timings spanning 8-300 steps —\na washout race); a mid-conversion state holding both natures was\nOBSERVED (one seed: site-store 66.8x the 2x-control bar, geometry\nretention 0.56) but is not a timescale law; the brake overshoot\n(A(129) deepening past -0.35 mid-conversion, released at the end)\nreplicates 3/3 (e152R)"
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

# ---------- queue + state ----------
q = open("QUEUE.md", encoding="utf-8").read()
m = re.search(r"^\| e152R \|[^\n]*\n", q, re.M)
if m:
    q = q[:m.start()] + "| e152R | dwell re-seeds | DONE 15:00Z (T102: DWELL-SEED-DEPENDENT — the dwell and 8-16 cliff are trajectory-specific; conversion INVARIANT 3/3; brake overshoot REAL 3/3; straddle cell unstable OPEN-to-SHUT — two-factor gate weakened to suggestive) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 3: e170 (anchor-neutral) + e173 (partition) + e176 (root freeze). e152R DONE: the dwell dies at n=3, the brake overshoot survives, the straddle is unstable."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e152R fold complete")
