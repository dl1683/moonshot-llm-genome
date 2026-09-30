# -*- coding: utf-8 -*-
"""R58 fold part B: T138 title fix, QUEUE repairs (two-convention, dose
ladder, opt1c, C13-1 residual), skeleton repairs, quarantine, STATE."""
import io, json, os

# ---------- T138 title residual ----------
t = io.open("THINKING.md", encoding="utf-8").read()
old = "## T138 — the literature pass: the trajectory law is new as a CONTROL, predicted as THEORY (2026-09-30 ~10:55Z)"
new = "## T138 — the literature pass: the trajectory hypothesis is new as a CONTROL, predicted as THEORY (title re-stamped per C13-1; 2026-09-30 ~10:55Z)"
assert old in t, "t138"
t = t.replace(old, new, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE repairs ----------
q = io.open("QUEUE.md", encoding="utf-8").read()
# C13-1 residual in g3K row
old = "THE NO-BASIN LAW IS A TRAJECTORY LAW)"
new = "the trajectory HYPOTHESIS: no static basin vs learned paths)"
assert old in q, "g3k row"
q = q.replace(old, new, 1)
# two-convention in opt1 row
old = "the GATE is displacement (~2.49-2.84 every Adam arm)"
new = "the GATE is displacement (two-convention: checkpoint 2.49 vs warmup 3.64; interpolated 2.18 vs 2.85 — PROPOSED)"
assert old in q, "opt1 row"
q = q.replace(old, new, 1)
# g1bW2 -> dose ladder
old = "| g1bW2 | THE DOSE CELL (g1bW's discriminator: B at 600+ steps on the unwashed control — does the ruler "
assert old in q, "g1bw2"
row = q[q.index("| g1bW2 |"):]; row = row[:row.index("\n")]
new_row = ("| g1bW2 | THE DOSE LADDER (R58-critic 2d: the reference's non-monotonicity 0.53->0.49 says the dose sat "
           "near a form-transition — B at {450, 600, 900, 1200} steps on the unwashed control locates the installable "
           "dose; then the walled contrast rerun at that dose; + B-draw replicate) | QUEUED — behind g1bS/g2g (GPU) |")
q = q.replace(row, new_row, 1)
# opt1c registered after opt1b
i = q.index("| opt1b |"); j = q.index("\n", i) + 1
opt1c = ("| opt1c | THE DIRECTION-SIZE FACTORIAL (R58-critic forced: sign-SGD — raw-gradient DIRECTION at Adam's "
         "measured step size 1.6543 L2/step; splits direction from size inside the Adam kill; adjudicates the "
         "critic's own form too) | QUEUED — CPU after opt1b; bars as opt1b's: kills at D in [2.12, 3.27] "
         "-> displacement direction-robust (alignment epiphenomenal); alive past D=2.6 (interpolated convention) "
         "-> 'any path reaching the gate kills' dies in its letter; the law moves to ruler-aligned-displacement currency |\n")
q = q[:j] + opt1c + q[j:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- skeleton repairs ----------
p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "R3b (T137, the most dangerous overlap): the trajectory law's nearest"
new = "R3b (T137, the most dangerous overlap): the trajectory hypothesis's nearest"
assert old in s, "r3b"
s = s.replace(old, new, 1)
# gap 12 wording: two-convention + PROPOSED
old = """12. Optimizer clause EVIDENCED (opt1, DONE): the two-step clock is
   Adam's sign-normalization (1683x/step at matched lr); the kill
   gate is displacement (~2.5, clock-invariant); under matched SGD
   the same stream STRENGTHENS the fact — paragraph (2)'s "at every
   lr tested" must become "under AdamW at every lr tested" and cite
   the decomposition; opt1b (the direct SGD kill) decides whether
   the gate generalizes across trajectory classes."""
new = """12. Optimizer clause EVIDENCED, decomposition PROPOSED (opt1 DONE;
   R58): the two-step clock is Adam's sign-normalization (1683x/step
   at matched lr; the normalizer flips the SIGN of fact-relevance:
   -0.0385 vs +0.0981 on the same batch); the kill gate reads
   displacement in a two-convention bracket (checkpoint 2.49 vs 3.64;
   interpolated 2.18 vs 2.85) — the small-displacement PUMP
   strengthens the fact under every optimizer (SGD lingers at
   1/1683rd speed); paragraph (2)'s "at every lr tested" becomes
   "under AdamW at every lr tested"; opt1b/opt1c/e188 adjudicate the
   gate's trajectory-class scope and the death currency."""
assert old in s, "gap12"
s = s.replace(old, new, 1)
# g1bW gap-8: adopt the onset-tax lead
old = """8. g1bW DONE (T140): A SURVIVED an active second-install attempt
   (0.83 through the install; free-run expression intact) — the
   wall's strongest positive. The MUSEUM contrast itself is
   unlicensed at this dose (B installs nowhere, walled or unwalled;
   disclosed); the measured tax is the ONSET channel (B partial-form
   0.21 vs 0.53). g1bW2 (the dose cell) adjudicates where B can
   install; until then the paper says "sequential: A held through an
   active second install; the dose-adequate contrast owed"."""
new = """8. g1bW DONE (T140 + R58-critic 2c): MUSEUM fired as registered,
   its rescope WITHHELD (the paired reference failed the same ruler —
   B installs nowhere at this dose; the museum question is OPEN
   pending g1bW2's dose LADDER). LEAD WITH THE ONSET-TAX — the one
   contrast-licensed read: the install's partial form halved inside
   the ball (B g0 peak 0.21 vs 0.53, bit-identical inputs). A held
   (min 0.65) through the 300-step attempt — real but a weak
   antagonist (it installed nothing); the paper says "the wall held
   A through a second-install attempt; the dose-adequate contrast
   owed (g1bW2)"."""
assert old in s, "gap8"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)

# ---------- quarantine the wrong-ruler g3K draft ----------
path = "lab/g3K_kappa_cell.py"
if os.path.exists(path):
    t = io.open(path, encoding="utf-8").read()
    if "SUPERSEDED" not in t:
        t = ('"""[SUPERSEDED — DO NOT RUN (R58-audit quarantine, 2026-09-30): this is the '
             'concurrent-session draft with the WRONG HOST RULER (e185 2.7M lineage-1 battery '
             'instead of the g3-lineage S-DISC). The canonical cell is lab/g3K_kappa.py '
             '(runs/g3K/metrics.json, commit 294dc6e). Retained as the double-session era\'s '
             'record; nothing here is a result."""]\n"""' + t.split('"""', 1)[-1] if t.startswith('"""') else
             '"""[SUPERSEDED — DO NOT RUN (R58-audit quarantine): wrong host ruler (e185 2.7M instead '
             'of g3-lineage S-DISC); canonical = lab/g3K_kappa.py @ 294dc6e]"""\n' + t)
        io.open(path, "w", encoding="utf-8").write(t)
        print("quarantine banner added")
else:
    print("WARN: g3K_kappa_cell.py not found")
old_png = "runs/g3K/kappa_cell.png"
new_png = "runs/g3K/QUARANTINE_wrong_ruler_kappa_cell.png"
if os.path.exists(old_png):
    os.rename(old_png, new_png)
    print("png renamed")
else:
    print("WARN: orphan png not found")

# ---------- STATE ----------
st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-09-30T12:16:00Z"
st["last_review"] = "2026-09-30T12:15:00Z"
st["last_novelty"] = "2026-09-30T12:15:00Z"
st["current_experiment"] = ("R58 FOLDED (numbers verified; three slogans corrected: teaches->small-displacement pump, "
                            "the NORMALIZER FLIPS THE SIGN of fact-relevance (-0.0385 vs +0.0981), D_kill bar was "
                            "circular -> decomposition PROPOSED; MUSEUM-withheld wording adopted; W023 rise-prediction "
                            "answered NO; opt1c registered; 1683x everywhere; g3K draft quarantined). Fleet: g1bS (GPU) "
                            "+ opt1b + e188 (CPU). Next: opt1c after opt1b; g2g after g1bS.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("part B done")
