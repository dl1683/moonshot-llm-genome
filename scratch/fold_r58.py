# -*- coding: utf-8 -*-
"""R58 full fold: REVIEWS entry (opens with C13 Lab response), card
amendments (T139/T140/W023/T138-title), NOTES corrections, QUEUE repairs
(1683x, two-convention, dose-ladder, opt1c, C13-1 residual), skeleton
repairs, g3K-draft quarantine, STATE stamps."""
import io, json

# ---------- REVIEWS.md ----------
rv = io.open("REVIEWS.md", encoding="utf-8").read()
sep = "---\n\n---"
entry = """---

## R58 — the decomposition day reviewed: numbers verified, three slogans corrected, one forced cell born (2026-09-30 ~12:15Z)

Trigger: 90 min since R57. Trio dispatched 11:52Z; all landed by
~12:10Z. Fleet during review: g1bS (GPU, building), opt1b + e188 (CPU).

SUPERVISOR ITEMS (check-in 13 Lab response):
- C13-1 (paper framing outran T137's stamp): DONE — repaired on T137,
  R6(c), framing, clause 4, paragraph 2, gaps 10/11; the R58 auditor
  found three residual "law" mints (QUEUE g3K row, skeleton R3b,
  T138 title) — ALL re-stamped in this fold.
- C13-2 (g1bS next GPU slot): DONE — dispatched 11:52Z from the
  frozen design (R in per-coordinate RMS units).
- C13-3 (e182c to CPU): DONE — QUEUE row moved.
- C13-4 (W020 provenance): with Devansh; nothing owed by the lab.
- C13-5 carried items: literature beat DONE (T138, 10:53Z); W023 card
  DONE (10:41Z) and now amended by the R58 critic (below) — the card
  wrote a registered prediction and the data answered it NO within
  the hour; recorded as the honest arc.
- Coordination: accepted — future coordination notes go to
  NOTES/THINKING; SUPERVISOR.md stays directives + check-ins +
  responses.

AUDITOR (scratch/r58_auditor.md; every number recomputed): g3K
kappa pair SOUND (host min 4.03 misses the <=4 bar by 0.03); opt1's
step ratio OVERCLAIMED at the third digit — 1683.06x not 1687x
(repaired everywhere); the "D ~ 2.49-2.84 within ~15%" is a
CONVENTION MIX (checkpoint 2.49 vs warmup 3.64; interpolated 2.18 vs
2.85 — opt1b's spared-gate is calibrated on the interpolated reading
and now says so); g1bW SOUND (one metrics prose bug noted: the
reference clause calls 0.0628 an "install" — the file stays frozen,
the contradiction documented); C13-1 residuals re-stamped; the
wrong-ruler g3K draft QUARANTINED (banner + PNG renamed — never
deleted); last_novelty re-stamped (the researcher beat happened but
the field was never updated).

CRITIC (scratch/r58_critic.md; three attacks, all evidence-backed):
  K1 "THE STREAM TEACHES UNDER SGD" is a small-displacement PUMP, not
  an optimizer property — A3 (AdamW+warmup) pumped to 0.9476 at
  D=0.368 before dying; e184 pumped +0.026 in one full-lr AdamW step;
  the SGD "teaching" is the same transient lingered in at 1/1683rd
  the speed. THE GENUINELY NEW FACT (was buried as a co-read): on the
  same bit-identical batch, Adam's step cos(delta, grad m12) =
  -0.0385 vs SGD's +0.0981 — THE NORMALIZER FLIPS THE SIGN OF
  FACT-RELEVANCE (T139's slogan inverted: the normalizer chooses the
  sign, not the stream). And "inherited moments carry nothing" was
  TAUTOLOGICAL (A0's wash starts fresh-state; A5 null by construction
  — inherited moments UNTESTED, claim withdrawn).
  K2 the displacement-gate evidence is CIRCULAR AT ITS CORE: D_kill
  2.4893 was imported from A0's own kill into the registration —
  predicting A0's t* from it is an identity; the only independent
  test (A3) is bracketed [1.45, 3.64] at 10-step resolution; and the
  e131 lineage has NO static-jump leg — the forgetting law is a
  two-organism stitch at n=1 each. T139's decomposition joins T137
  under the PROPOSED stamp.
  K3 g1bW's honesty lived in NOTES while the metrics' adjudication
  clause contradicted it (file stays frozen; contradiction
  documented); MUSEUM's fire was dose-guaranteed (W021's species —
  its paired control should have been gating); "A survives an ACTIVE
  second install" oversold a weak antagonist (installed nothing;
  non-monotonic 0.53->0.49). ADOPTED VERBATIM: the critic's 2c
  MUSEUM-WITHHELD wording on T140 + the paper; g1bW2 re-registered
  as a dose LADDER (2d: the non-monotonicity says the dose sat near a
  form-transition; a point at 600 steps may overshoot); the paper
  leads with the ONSET-TAX (the one contrast-licensed read). W023's
  rise-prediction ANSWERED NO on disk (A0 |cos| flat 0.015->0.044->
  0.015 while CE_R recovered 2.21->1.77) — amended.
  FORCED CELL: opt1c, THE DIRECTION-SIZE FACTORIAL — sign-SGD (raw-
  gradient DIRECTION at Adam's measured step size 1.6543 L2/step):
  kills at the bracket -> cumulative displacement is direction-robust
  (alignment epiphenomenal); alive past D=2.6 -> "any path reaching
  the gate kills" dies in its letter and the law moves to
  ruler-aligned-displacement currency. Registered behind opt1b.

IDEATOR (scratch/r58_ideator.md): e188 CONFIRMED + the two-currency
amendment (DISPATCHED 12:00Z); opt2 THE SIGN CARRIER (SIGN / TOPK /
WARMV at matched per-step L2; QUEUED CPU after opt1b — bars stable
under either opt1b outcome); g9 THE ADMISSION BALL (top-k PC
projection of A's install-gradient structure; spec frozen, GATED on
g1bW2's dose ladder by construction).

DECISIONS: (1) all auditor repairs applied this fold; (2) critic's
wordings adopted verbatim (T139/T140/W023/paper); (3) opt1c
registered (behind opt1b); (4) double-session guards adopted as
policy — no hardcoded reference constants in lab/*.py, dispatch
briefs cite spec provenance; (5) the forgetting law's current honest
form: DISPLACEMENT-GATED (checkpoint-bracketed), OPTIMIZER-CARRIED
SPEED, DIRECTION-STRUCTURE UNRESOLVED pending opt1b/opt1c/e188 — all
currencies PROPOSED until the factorials land."""
assert sep in rv
rv = rv.replace(sep, "---\n\n" + entry + "\n\n---", 1)
io.open("REVIEWS.md", "w", encoding="utf-8").write(rv)

# ---------- 1687 -> 1683 everywhere ----------
for path in ("NOTES.md", "THINKING.md", "QUEUE.md", "scratch/day6_paper_skeleton.md"):
    t = io.open(path, encoding="utf-8").read()
    c = t.count("1687x")
    if c:
        t = t.replace("1687x", "1683x")
        io.open(path, "w", encoding="utf-8").write(t)
    print(path, "1687->1683:", c)

# ---------- T139 amendment ----------
t = io.open("THINKING.md", encoding="utf-8").read()
import re
m = re.search(r"^## T139 .*$(.*?)(?=^## T138)", t, re.M | re.S)
assert m
amend = """
[R58 AMENDMENT — three corrections, all evidence-backed]:
(a) "THE STREAM TEACHES UNDER SGD" is retired: the fact-strengthening
is a SMALL-DISPLACEMENT PUMP, not an optimizer property — A3
(AdamW+warmup) pumped to 0.9476 at D=0.368 before dying; e184 pumped
+0.026 in one full-lr AdamW step; SGD lingers in the pump at
1/1683rd the speed. The genuinely new SGD-specific fact (was a
co-read): on the same bit-identical batch Adam's step reads
cos(delta, grad m12) = -0.0385 vs SGD's +0.0981 — THE NORMALIZER
FLIPS THE SIGN OF FACT-RELEVANCE; the slogan inverts: the normalizer
chooses the sign, the stream supplies the gradient.
(b) "Inherited moments carry nothing" WITHDRAWN as tautological —
A0's wash already starts fresh-state, so A5 was null by construction;
inherited moments are UNTESTED (opt2's WARMV arm owns the question).
(c) The decomposition joins T137 under PROPOSED: D_kill 2.4893 was
imported from A0's own kill (the registered bar was circular for A0;
identity, not prediction); the only independent test (A3) is
bracketed [1.45, 3.64] at 10-step resolution — the honest
displacement statement is two-convention (checkpoint 2.49 vs 3.64;
interpolated 2.18 vs 2.85; opt1b's gate reads the interpolated
convention); and the e131 lineage has no static-jump leg — the
forgetting law is a two-organism stitch at n=1 each. opt1b (running)
+ opt1c (registered) + e188 (running) are the adjudicators."""
t = t[:m.start(1)] + m.group(1).rstrip("\n") + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- T140 amendment ----------
t = io.open("THINKING.md", encoding="utf-8").read()
m = re.search(r"^## T140 .*$(.*?)(?=^## T139)", t, re.M | re.S)
assert m
amend = """
[R58 AMENDMENT — the critic's 2c wording adopted]: "MUSEUM fired as
registered (walled B 0.074 <= 0.27 at healthy CE 1.63). Its
registered rescope is WITHHELD: the paired unwalled reference failed
the same B-ruler (0.063) — at the 300-step dose B installs nowhere in
this lineage (partial onset form only, g0 peak 0.53 unwalled), so
B's failure inside the wall is uninformative about the wall. The
museum question is OPEN pending the dose control (g1bW2, registered
as a dose LADDER: the reference's non-monotonicity 0.53->0.49 says
the dose sat near a form-transition; a point at 600 steps may
overshoot). Licensed today: (1) the wall held A (min 0.65) through a
300-step second-install attempt — protection survives interference;
(2) THE ONSET-TAX is the day's cleanest paired read (B g0 peak 0.21
walled vs 0.53 unwalled, bit-identical inputs, 300/300) and the one
contrast-licensed claim — the paper leads with it. 'Survives an
ACTIVE second install' softens: an antagonist that installed nothing
is a weak antagonist until g1bW2 supplies a dose at which B presses.
The metrics' adjudication clause contradicts this reading (calls
0.0628 an 'install' and asserts the rescope) — the file stays frozen
as the machine record; this amendment is the interpretive record."""
t = t[:m.start(1)] + m.group(1).rstrip("\n") + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- W023 amendment ----------
t = io.open("THINKING.md", encoding="utf-8").read()
old = "Savoring: the organism never stops healing;\nthe memory dies of the healing itself."
if old not in t:
    old = "Savoring: the organism never stops healing; the memory dies of the healing itself."
assert old in t, "w023 tail"
new = old + """
[R58 AMENDMENT — the rise-prediction ANSWERED NO, 12:15Z]: opt1's
own disk data: A0's |cos| ran FLAT (0.015 -> 0.044 -> 0.015) while
CE_R recovered 2.21 -> 1.77 — the alignment does NOT rise with the
recovery; "adaptation sharpens the knife" in its alignment form is
REFUTED. What survives of the wonder: the pump-then-die texture (the
fact strengthens at small displacement, dies past the gate — the
healing and the killing remain one trajectory), and the
replay-as-mini-shock question transfers to g2g's monitor slopes.
Cards keep their predictions AND their answers."""
t = t.replace(old, new, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)
print("cards amended")
