# -*- coding: utf-8 -*-
"""Supervisor C13-1 repairs: trajectory LAW -> HYPOTHESIS (proposed, n=1)
on every claim-carrying surface; QUEUE lane reorders (C13-2/3); stamp."""
import io, json

# --- paper ---
p = "scratch/day6_paper_skeleton.md"
t = io.open(p, encoding="utf-8").read()

old = "(c) THE CONE + THE TRAJECTORY LAW (g3R-amended + g3K): killing is"
new = ("(c) THE CONE (licensed: n=3 wash-draw seeds, one organ) + THE\n"
       "   TRAJECTORY HYPOTHESIS (PROPOSED — g3K n=1 per organism, kappa\n"
       "   intervals overlap [2.8,6.6] vs [4.0,9.7]; replication owed: e188\n"
       "   + two more organisms; supervisor C13-1): killing is")
assert old in t, "r6c"
t = t.replace(old, new, 1)

old = """Framing sentence: forgetting is not distance and not even
   displacement — it is ALIGNED TRAINING: any learned path to a
   displacement kills where a random jump of the same size is 4-10x
   more forgivable; the aligned front can be walled (g1,"""
new = """Framing sentence (the hypothesis's slogan, PROPOSED per C13-1 —
   not law-graded until replicated): forgetting may be not distance
   and not even displacement but ALIGNED TRAINING — any learned path
   to a displacement kills where a random jump of the same size is
   4-10x more forgivable; the aligned front can be walled (g1,"""
assert old in t, "framing"
t = t.replace(old, new, 1)

old = """and a direction-and-trajectory-typed
   forgetting law (wash-aligned training kills at ~2x where static
   random displacement is 4-10x more forgivable; no state survives
   any learned path) each convert a dissected failure law into an"""
new = """and a direction-typed forgetting
   law with a PROPOSED trajectory hypothesis (wash-aligned training
   kills at ~2x where static random displacement is 4-10x more
   forgivable; n=1, replication owed) each convert a dissected
   failure law into an"""
assert old in t, "clause4"
t = t.replace(old, new, 1)

old = ("the mechanism per e185+e180+g3K: no robustness basin against "
       "LEARNED displacement (the law is a TRAJECTORY law — static random "
       "displacement is 4-10x more forgivable, kappa ~5-6 at matched "
       "per-coordinate RMS; every e185 arm was a trajectory), ")
new = ("the mechanism per e185+e180 (+g3K, PROPOSED n=1): no robustness "
       "basin against LEARNED displacement (the trajectory hypothesis — "
       "static random displacement is 4-10x more forgivable, kappa ~5-6 at "
       "matched per-coordinate RMS; every e185 arm was a trajectory; "
       "replication owed per C13-1), ")
assert old in t, "p2"
t = t.replace(old, new, 1)

# gap 11
old10 = """10. g3K (DONE): abstract paragraph (2) must gain the TRAJECTORY-vs-
   STATIC clause"""
new10 = """11. Trajectory-hypothesis replication (C13-1, NEW): e188's
   alignment-integral test + >=2 more organisms (kappa pairs) before
   any "trajectory law" or "aligned training" wording is law-graded.
10. g3K (DONE): abstract paragraph (2) must gain the TRAJECTORY-vs-
   STATIC clause"""
assert old10 in t, "gap11"
t = t.replace(old10, new10, 1)
io.open(p, "w", encoding="utf-8").write(t)

# --- T137 amendment ---
th = io.open("THINKING.md", encoding="utf-8").read()
old = "## T137 — g3K: the basin is graded and the no-basin is a trajectory law (2026-09-30 ~10:50Z)"
new = old + """
[C13-1 AMENDMENT ~11:00Z — the claim's stamp: PROPOSED, not law. The
card's body carried the scope (organ n=1 per organism, overlapping
kappa intervals) but the TITLE minted "law" from one cell — my paper
amendment 17 minutes after landing repeated the sin. Status until
e188's integral test + >=2 more organisms: THE TRAJECTORY
HYPOTHESIS. The cone noun (n=3 wash-draw, one organ) keeps its
licensed standing.]"""
assert old in th, "t137"
th = th.replace(old, new, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(th)

# --- QUEUE lane orders (C13-2/3) ---
q = io.open("QUEUE.md", encoding="utf-8").read()
old = "| e182c | GPT-2 forgetting control (matched non-fact probes, >=2 corpus draws, eval-cost fix; C12-3) | QUEUED — GPU, behind g1bW | separates no-basin from generic forgetting |"
new = "| e182c | GPT-2 forgetting control (matched non-fact probes, >=2 corpus draws, eval-cost fix; C12-3) | QUEUED — MOVED TO CPU LANE per C13-3 (eval-only probes at 124M are CPU-feasible; run beside GPU cells) | separates no-basin from generic forgetting |"
assert old in q, "e182c"
q = q.replace(old, new, 1)
old = "| g1bS | the wall at >=10x (~10M) (C12-4) | QUEUED — design first | scale debt before any architectural-law wording |"
new = "| g1bS | the wall at >=10x (~10M) (C12-4) | NEXT GPU SLOT after g1bW per C13-2 — design owed FIRST (this is the design step); before any further g-series cell | scale debt before any architectural-law wording |"
assert old in q, "g1bS"
q = q.replace(old, new, 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

# --- STATE ---
s = json.load(io.open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = "2026-09-30T11:00:00Z"
s["current_experiment"] = ("Fleet 2: g1bW (GPU, metrics written 10:55Z - finalizing) + opt1 (CPU, computing). "
                           "C13-1 repaired: trajectory LAW -> HYPOTHESIS (proposed, n=1) on T137 + paper + "
                           "gap 11 (e188 + 2 organisms owed); C13-2/3 adopted: g1bS next GPU slot (design first), "
                           "e182c to CPU lane. Coordination resolved (supervisor's duplicate was one-shot, now exited).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(s, indent=2, ensure_ascii=False))
print("C13 repairs done")
