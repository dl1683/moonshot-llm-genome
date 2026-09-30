# -*- coding: utf-8 -*-
"""e188 fold part B: T137/T138/T139 amendments, paper rewrites, QUEUE, STATE."""
import io, json, re

t = io.open("THINKING.md", encoding="utf-8").read()

def amend_card(t, card, text):
    m = re.search(r"^## " + card + r" .*$(.*?)(?=^## T\d)", t, re.M | re.S)
    assert m, card
    body = m.group(1).rstrip("\n")
    return t[:m.start(1)] + body + "\n\n" + text + "\n\n" + t[m.end(1):]

t = amend_card(t, "T137", """[E188 AMENDMENT — REDUCED, 12:05Z]: the trajectory hypothesis loses
its integral layer (RAW-WINS: the alignment-weighted currency lost
to raw displacement, CV 0.233 vs 0.557; the pre-registered Branch B
executed). WHAT REMAINS: the STATIC-vs-LEARNED contrast (g3K's
kappas 4-10x at matched per-coordinate RMS, n=1, replication still
owed) — "learned paths reach the gate at 1x; static jumps need
4-10x" — with no claim about alignment as the currency. The paper
sentence becomes the displacement-threshold form.""")
t = amend_card(t, "T138", """[E188 RESOLUTION, 12:05Z]: the vocabulary clause RESOLVED BY
MEASUREMENT — install-vs-wash cos in [-0.034, -0.018] on all four
organisms; |cos| < 0.3 everywhere -> task-arithmetic vocabulary
REJECTED; the wash is the corpus's adaptation direction, not the
fact's negation. The R3d pre-empt resolves: report the cosines,
reject the vocabulary.""")
t = amend_card(t, "T139", """[E188 AMENDMENT — THE GATE STRENGTHENED, 12:05Z]: e188's RAW-WINS
is the displacement gate's best evidence: D at death varies 0.6%
across three wash seeds at matched t* (2.489/2.484/2.500) — the
tightest invariance the wash arc has produced — while the aligned
currency spreads 3.7x (a seed lottery). The decomposition now
reads: CLOCK = Adam's normalization (opt1); GATE = raw displacement
(e188); CURRENCY-of-reaching = the open question (opt1b running,
opt1c dispatching); ALIGNMENT = passenger (flips positive
post-kill on the fast arm — the organism returns to the grave it
dug).""")
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# --- paper rewrites ---
p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = """Framing sentence (the hypothesis's slogan, PROPOSED per C13-1 —
   not law-graded until replicated): forgetting may be not distance
   and not even displacement but ALIGNED TRAINING — any learned path
   to a displacement kills where a random jump of the same size is
   4-10x more forgivable; the aligned front can be walled (g1,"""
new = """Framing sentence (REWRITTEN AFTER E188 — the aligned-training
   slogan died by measurement; RAW-WINS): the wash kill is a
   DISPLACEMENT THRESHOLD under training — raw D at death is the
   invariant (0.6% across seeds at matched t*) — while the
   static/learned contrast governs how fast the threshold is
   reached (learned paths at 1x, static jumps 4-10x; alignment a
   passenger, seed-lottery at death); the front can be walled (g1,"""
assert old in s, "framing"
s = s.replace(old, new, 1)
old = """and a direction-typed forgetting
   law with a PROPOSED trajectory hypothesis (wash-aligned training
   kills at ~2x where static random displacement is 4-10x more
   forgivable; n=1, replication owed) each convert a dissected
   failure law into an"""
new = """and a displacement-threshold
   forgetting law (raw displacement at death is the invariant,
   0.6% across seeds; learned paths reach the gate at 1x where
   static jumps need 4-10x; alignment demoted to passenger by the
   death-currency measurement) each convert a dissected failure law
   into an"""
assert old in s, "clause4"
s = s.replace(old, new, 1)
# R2b currency line
old = """the kill reads displacement in a two-convention
   bracket (PROPOSED — opt1b/opt1c/e188 adjudicate the gate's
   trajectory-class scope and the death currency; e189 reads the
   stitches-vs-cuts decomposition, W024); the trajectory-vs-static
   hypothesis (g3K, n=1) and its replication debt."""
new = """the kill is priced in RAW displacement (e188 DONE: RAW-WINS —
   D at death CV 0.6% across seeds at matched t*; the aligned
   currency lost, CV 0.56; alignment a passenger that flips
   positive post-kill); opt1b/opt1c adjudicate the gate's
   trajectory-class scope; the static/learned contrast (g3K kappas,
   n=1) keeps the replication debt; task-arithmetic vocabulary
   REJECTED (install-vs-wash cos in [-0.034,-0.018] on four
   organisms)."""
assert old in s, "r2b"
s = s.replace(old, new, 1)
# gap 11 narrows
old = """11. Trajectory-hypothesis replication (C13-1, NEW): e188's
   alignment-integral test + >=2 more organisms (kappa pairs) before
   any "trajectory law" or "aligned training" wording is law-graded."""
new = """11. Static/learned replication (C13-1, narrowed by e188): the
   integral layer is DEAD (RAW-WINS); the owed replication is the
   kappa contrast alone — >=2 more organisms (kappa pairs) before
   the static-vs-learned wording is law-graded."""
assert old in s, "gap11"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)

# --- QUEUE ---
q = io.open("QUEUE.md", encoding="utf-8").read()
i = q.index("| e188 |"); row = q[i:]; row = row[:row.index("\n")]
new_row = ("| e188 | THE DEATH-CURRENCY CELL | DONE 12:05Z (T141: RAW-WINS — raw D at death is the invariant "
           "(CV 0.233 vs 0.557; 0.6% seed spread at matched t*); W022b's aligned-drift law KILLED per its own "
           "pre-registered Branch B; T139's gate STRENGTHENED; T137 reduced to the static/learned contrast; "
           "task-arithmetic vocabulary REJECTED (cos in [-0.034,-0.018] x4 organisms); post-kill alignment flips "
           "positive) |")
q = q.replace(row, new_row, 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

# --- STATE ---
st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-09-30T12:06:00Z"
st["current_experiment"] = ("e188 FOLDED (T141: RAW-WINS — death priced in raw displacement; alignment a passenger; "
                            "the aligned-training slogan died by measurement; vocabulary rejected). Fleet: g1bS (GPU) "
                            "+ opt1b (CPU) + opt1c DISPATCHING (the direction-size factorial). opt2/e189 queue behind.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("part B done")
