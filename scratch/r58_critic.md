# R58 CRITIC — the day's three results attacked (2026-09-30 ~12:05Z)

Inputs: NOTES g3K/opt1/g1bW; THINKING T137-T140, W021-W023; REVIEWS R56-R57;
runs/opt1/metrics.json (all seven arms' traj + ckpt tables + adjudication
recomputed); runs/g1bW/metrics.json (adjudication, gates, both legs);
lab/opt1b_sgd_kill.py registration; scratch/r56_critic.md 54-64 (the g1bW
spec as registered). All numbers below re-read from the committed metrics,
not from the notes' prose.

---

## 1. OPT1 — the decomposition is half measured, one clause is tautological,
    and the "bombshell" is a dose effect wearing an optimizer's coat

**1a. "The stream teaches under SGD" is over-read; the pump is a
small-displacement transient that Adam exhibits too — inside opt1 itself.**
The SGD rise (g-12 0.9156 -> 0.940/0.953/0.9548 at D <= 0.285) is real and
paired, but it is not SGD-specific and probably not stream-specific:
- A3 (AdamW + warmup) pumped the read to **0.9476 at D = 0.368** (ckpt s4)
  before dying — the same +0.03 nudge at the same small displacement.
- e184 seed 10903 pumped +0.026 in ONE full-lr AdamW step (D = 1.65) before
  collapsing (NOTES e184).
So "teaching" tracks displacement-smallness, not optimizer. The SGD-specific
residue is arithmetic: SGD's step is ~1687x smaller, so it LINGERS in the
pump window. That is a restatement of the clock claim, not a new property of
the stream. What IS new and SGD-specific is the sign flip (1b below) — and
it is buried as a co-read while the slogan says the opposite.

**1b. T139's slogan inverts the data's sharpest finding.** "The stream
chooses signs, the normalizer chooses the clock" — but on the SAME batch
(ce_batch bit-identical 1.35657, gated): A0's step-1 cos(delta, grad m12) =
**-0.0385**; A1's = **+0.0981**. Same stream gradient, opposite
fact-relevance of the resulting step: the normalizer flips the SIGN of the
step's alignment with the fact's strengthening direction (and A1's
cos(grad g0) is *negative*, -0.05..-0.10 — the two readout gradients
disagree in sign under SGD; see 3a). The honest slogan: "the stream supplies
the gradient; the normalizer chooses both the clock AND the sign of the
fact's fate." n=1, ckpt-sampled — but this, not "teaches," is the candidate
mechanism, and it is the one thing opt1b cannot cleanly test (1e).

**1c. The basin-vs-step arithmetic is circular, and the only independent
test of "the gate is displacement" has a bracket, not a number.** D_kill =
2.4893 was imported into the registration as *A0's own kill displacement*
("e185 stored, A0-gated"). Dividing it by A0's own step size (1.6543) to
"predict" A0's own t* = 1.63 is an identity dressed as a prediction. A4
(beta2 0.999, D 2.485) is a genuine but near-clone arm. The ONLY independent
leverage is A3: kill bracketed between ckpt 10 (gm12 0.634 at D 1.449) and
ckpt 20 (gm12 0.065 at D 3.636); interpolated D(t_x=16.39) ≈ 2.8-2.9 —
"within ~15%" as stated, but the measurement bracket is [1.45, 3.64]: the
instrument could have hidden a 46% gate shift. With warmup's lr still
ramping through the bracket, sublinear-interp gives ~2.81, linear ~2.85 —
robust-ish, still bracket-grade. The paper sentence must be "kill
displacement bracketed [1.45, 3.64], interpolated 2.85, across a 10.08x
clock stretch," never "the same displacement." And note e185's OWN noise arms
(T114) were noise-LABEL TRAINING (trajectories), so the e131 lineage has NO
static-jump leg at all — the 4-10x static basin is a g3-lineage number. The
displacement-gate claim on e131 rests on one Adam family, one root, n=1.

**1d. Clause (3) "inherited moments carry NOTHING" is a tautology as run.**
A0's recipe is fresh-state ("fresh state" in the arm spec); A5 resets a
state that is already fresh — bit-identical BY CONSTRUCTION. As a gate
("the licensed cell never inherited moments — now proven, not assumed") it
is legitimate and honestly footnoted in honesty_reflex; as a FINDING clause
("beta2 and inherited moments carry NOTHING") it claims a test that never
ran. No arm carried consolidation-time moments into the wash. Reword or
drop; a real inherited-moment arm would need e113's optimizer state (not on
disk — say so).

**1e. The verdict label is licensed; the decomposition front-runs its own
discriminator.** ADAM-AMPLIFIES fired on the warmup clause alone; the SGD
clause ("matched-displacement SGD survives >= 5x longer") could not fire by
the registration's own cap rule — so "the kill is substantially
Adam-carried" is, as of opt1, evidenced only WITHIN the Adam family. The
projections (379/1174/3228 steps) do zero adjudicative work (correct, and
correctly labeled) but non-trivial RHETORICAL work: they linearize a
nonlinear process over 5-48x the observed window (g3K itself measured
trajectory drift, cos d1..d300 ~ 0.13 — the path curls; a constant-rate
projection ignores curvature, and A2b's rate is fitted on 69 steps). T139
already writes "SGD's slow path is fact-positive in-window" — true — but
the surrounding prose ("the killer stream TEACHES under SGD", "the
BOMBSHELL") treats the window's edge as the path's fate. opt1b is running;
the decomposition should carry T137's PROPOSED stamp until it lands, exactly
as the trajectory hypothesis was forced to.

**1f. What the 1687x is and isn't.** The number is honest arithmetic:
step-1 AdamW L2 = 1.6543 ≈ lr*sqrt(2,739,072) = 1.6548 (sign-normalized,
~every coordinate at ±lr); SGD = lr*||g|| = 0.00098. In the displacement
currency — the currency the gate is defined in — the comparison is the
right one. But "at the same lr" is matched-in-name: nobody expects
cross-family lr equality to mean equal steps; the informative matched
quantity for the gate question is displacement RATE, which is what opt1b
uses. Print both; stop letting "1687x" carry the argument.

**1g. What would falsify "the gate is displacement" that opt1b cannot.**
opt1b varies step size, normalization, and trajectory class SIMULTANEOUSLY
(SGD@1e-2 differs from Adam in all three). The clean complement is the
direction-size factorial on the same licensed cell: (i) **sign-SGD at
Adam's step size** — raw-gradient direction, ±lr per coordinate (L2/step ≈
1.65): if it kills at D ~ 2.5, displacement currency beats direction; if it
survives past D = 2.6, direction/normalizer-typing wins and "any path that
REACHES the gate kills" dies in its current letter. (ii) mirror arm: Adam
direction rescaled to SGD's step size. One CPU lane, ~5-10 min/arm, both
bars registrable exactly as opt1b's. This is the forced experiment (below).

---

## 2. G1BW — the verdict/consequence split is honest; the machine record
    contradicts itself; and the fired bar was dose-guaranteed (W021 again)

**2a. What the lab did is right; the ledger must match it.** MUSEUM fired
per its registered letter (walled B 0.0736 <= 0.27 at healthy CE 1.6285 vs
bar 1.9635). The reference leg — registered as "co-reported, NOT gating" —
failed the same ruler (0.0628), so the registered RESCOPE ("the wall is a
splint...") has a falsified causal premise and was withheld, with the
discriminator (g1bW2, dose) queued. That is the correct sequence: the
VERDICT stands forever; the CONSEQUENCE is suspended pending the
discriminator. BUT runs/g1bW/metrics.json's adjudication clause reads
"...while the reference (unwalled) leg INSTALLS B at 0.0628 — the wall is a
splint; rescope the claim..." — 0.0628 fails B_installs (>= 0.7) and even
MUSEUM's own <= 0.27 letter; the auto-clause mislabels a failure as an
install and still carries the rescope sentence. Any auto-synthesis (paper
tables, future agents) inherits the false clause. Annotate the metrics or
correct via NOTES pointer; the paper must quote T140's form, not the
clause.

**2b. Where the line is.** Proposed standing rule (for REVIEWS/paper): *a
fired bar's verdict is immutable; its registered consequence may be
suspended ONLY by (a) a paired, registered control falsifying the
consequence's causal premise, recorded as (b) a new registered question
with a named discriminator — never by re-reading the fired leg's texture,
and never silently.* g1bW satisfies (a)+(b). The inverse guard is the one
W021 already names and this cell re-demonstrates: MUSEUM's fire was
GUARANTEED at any dose too low for B to install, wall or no wall — a bar
whose fire can be produced independent of the manipulated variable must
carry its paired control as GATING (or be titled "conditional on reference
installing"). The dispatch registered the detector but left it non-gating;
that design choice is why today's question is open. Same species as the
isotropic-4x leg and the refractory band: the outcome the world did not get
to vote on.

**2c. The exact wording T140/the paper should carry:**

> "MUSEUM fired as registered (walled B-ruler 0.074 <= 0.27 at healthy CE
> 1.63). Its registered rescope is WITHHELD: the paired unwalled reference
> failed the same B-ruler (0.063) — at the registered 300-step dose B
> installs nowhere in this lineage (partial onset form only, g0 peak 0.53
> unwalled), so B's failure inside the wall is uninformative about the
> wall. The museum question is OPEN pending the dose control (g1bW2: B at
> 600+ steps unwalled; walled contrast rerun at whatever dose installs).
> What the cell licenses today: (1) the wall held fact A (min 0.65 >= 0.5)
> through an active 300-step second-install attempt — protection survives
> interference, the wall's strongest positive; (2) the install's partial
> form was halved inside the ball (B g0 peak 0.21 vs 0.53 unwalled,
> bit-identical inputs) — an onset-channel tax; one lineage, B-draw n=1."

**2d. Two texture points that matter.** (i) The reference B_g0
non-monotonicity (0.53 at s100 -> 0.49 at s300) says the dose sat near a
form-transition, not merely below one — strengthens the g1bW2 case and
warns that "600 steps" may overshoot into a different regime; register a
dose LADDER, not a point. (ii) The onset-tax contrast (0.21 vs 0.53) is the
day's cleanest paired read — bit-identical install inputs, G_INPUTS 300/300
— and it is the one piece of g1bW that IS contrast-licensed. Lead with it;
the A-survival number is real but was also MUSEUM's own ambiguity (B never
threatened A at this dose — an active install that installs nothing is a
weak antagonist; "survives an ACTIVE second install" slightly oversells
until g1bW2 supplies a dose at which B actually presses).

---

## 3. THE TRAJECTORY HYPOTHESIS — the axis is half right; the law's currency
    is still unchosen; and it is a two-organism stitch at n=1 each

**3a. "Learned vs static" is the wrong noun; "static vs multi-step
adapting" is closer, but the live question is the CURRENCY.** SGD is
learned, and — per opt1's own co-reads — currently life-aligned (cos to
grad m12 = +0.10). If opt1b's SGD dies at the gate, "any learned path
kills" is true but vacuous-ish (everything tested that reaches D kills);
if it survives, "learned" was never the classifier. The discriminating
variables the day actually measured: (i) cumulative displacement, (ii)
per-step alignment with the readout's death gradient, (iii) step-size
regime. And the alignment story has an unresolved convention split: under
SGD, cos(delta, grad g0) = -0.05..-0.10 while cos(delta, grad m12) =
+0.10 — the law must FIX the ruler gradient (the kill is adjudicated on
m12) and report both, or the integral (e188) inherits a sign ambiguity.
Also: W022's "-0.44" is the g3 STORE's wash; opt1's host wash reads
-0.015..-0.10. In 2.7M dims random |cos| ~ 0.0006, so even -0.05 is ~80x
random — alignment survives, the number does not generalize; the paper must
not let -0.44 stand in for the host.

**3b. The cleanest falsifiable form I will sign today (PROPOSED, two
organisms, n=1 each):**

> A consolidated fact's readout dies when the ruler-aligned cumulative
> displacement — SUM_t ||d_t|| * (-cos(d_t, grad_readout)), one fixed
> ruler convention — crosses a per-fact threshold A*; a single static
> jump of the same L2 survives 4-10x (the g3 kappa; unmeasured on e131);
> the optimizer enters only through the rate, size, and alignment of d_t:
> AdamW at lr 1e-3 delivers lr*sqrt(N) ~ 1.65/step at |cos| ~ 0.02-0.04
> (25-70x random) and kills in ~2 steps; the raw stream gradient is
> weakly life-aligned (SGD steps: +0.10 to the ruler gradient), so SGD's
> aligned integral grows slowly, if at all.

**3c. The fork that breaks it — and it is already running.** Under this
form, opt1b's SGD@1e-2 should NOT die at D = 2.49-2.84 (its aligned
integral is small/negative) — it should drift lifeward or die only after
far more aligned accumulation. If SGD-KILLS-AT-GATE fires, MY form dies and
the raw cumulative-displacement gate (trajectory-class-agnostic) wins; if
SGD-SPARED-AT-GATE fires, the raw-displacement form dies and some
direction/normalizer typing wins. Either way the day's law gains its third
clause by measurement, not by projection — which is why opt1b outranks
every new design until it lands. The sign-SGD arm (1g) then splits
direction from size inside the Adam kill itself: my form says a
life-aligned big-step path survives past 2.6; the displacement form says it
dies at ~2.5. One arm, both sentences at risk.

**3d. W023 should be amended now, not held "undecided."** The ckpt co-reads
ARE the per-arm curves at cadence: A0's |cos(delta, grad g0)| runs
0.0152 -> 0.0438 -> 0.0264 -> 0.0159 -> 0.0155 while CE_R recovers 2.21 ->
1.77 — flat, no rise; "adaptation sharpens the knife" is unsupported at
the resolution that exists, and sub-prediction (2) (moment-reset extends
the shock) is structurally vacuous given A5's tautology (1d). Write the
amendment; "undecided pending per-arm curves" spends credit on curves that
are already on disk.

**3e. The stitch the paper must own.** opt1/e131 (Adam-family kill gate
2.49, alignment -0.02..-0.10, no static leg, no SGD kill yet) and g3K/g3
(static 4-10x, wash ~1x, kappa_store ≈ kappa_host with OVERLAPPING n=1
intervals) are different organisms. The "whole forgetting law" is a
cross-organism synthesis at n=1 per organism — legal as PROPOSED, illegal
as law (T137's own amendment logic applies verbatim to T139's
decomposition). One unified displacement table (L2, per-coordinate RMS,
wash-1x units per organism) is owed before any sentence spans both.

---

## 4. CROSS-CUTTING

- **CPU fp32 vs GPU-era:** opt1's invariance chain is real but thin — A0
  re-runs the licensed cell on CPU and gates against stored values
  (5.96e-08 root, 3.2e-13 control), which is the right mechanism; g1bW's
  trainings are cuda-with-pauses. The load-bearing cross-texture numbers
  (t* = 2, wash reproduction to 7 decimals) hold; the paper should tag each
  headline number's device, and the SGD drift (+0.02, within-arm paired) is
  texture-safe only as a TREND, not a level.
- **n=1 everywhere, one day, three headlines.** The lab's own meta-law
  (mechanism claims at H need >= 3 nets) was enforced on T137 by amendment
  17 minutes after landing — and NOT on T139, whose decomposition prose
  ("REFINES the trajectory hypothesis") reads established. Stamp it
  PROPOSED now, before opt1b's result launders it.
- **Double-session quarantine:** g1bW caught five bugs in the concurrent
  session's draft, including F2 (fabricated g1bR constants) — caught ONLY
  by verify-before-run. That is the third saved-by-vigilance incident this
  week. Cheap standing guards, no new automation: (1) no hardcoded
  reference constants in lab/*.py — every reference number loaded from
  runs/*/metrics.json at runtime (F2 dies by construction); (2) dispatch
  briefs cite the spec file's hash when they "freeze" it. Write both into
  AGENTS.md's hard rules at next edit.

---

## THE FORCED EXPERIMENT (after opt1b, same machinery, ~10 min CPU)

**opt1c — the direction-size factorial, minimum one arm:** sign-SGD on the
licensed e185 wash cell — raw-gradient DIRECTION, per-coordinate ±lr chosen
so L2/step = 1.6543 (Adam's measured step-1 size); bars registered exactly
as opt1b's, kill bracket [2.12, 3.27] vs alive-at-D = 2.6.
- Kills at the bracket -> the cumulative-displacement currency is
  direction-robust; "the gate is displacement" survives its strongest
  available attack and the alignment integral is demoted to epiphenomenon.
- Alive past 2.6 -> the gate is direction-typed; "any path that REACHES the
  gate kills" dies in its current letter, and the law must be stated in
  alignment currency (3b) — with e188 then the decisive lr-invariance test.
Either branch rewrites the paper's mechanism sentence; the arm costs one
CPU lane for ten minutes. (Mirror arm — Adam direction at SGD step size —
optional, second priority.)

## THE THREE SHARPEST ATTACKS (summary)

1. **"The stream teaches under SGD" is a small-displacement pump, not an
   optimizer property — opt1's own warmup arm shows it (0.9476 at D 0.37
   under AdamW; e184 s10903 pumps +0.026 under full-lr AdamW), and the
   genuinely new SGD-specific fact — the normalizer flipping the step's
   fact-alignment sign (-0.0385 vs +0.0981 on the same batch) — is buried
   in a co-read while the headline slogan asserts the opposite causal
   order.**
2. **The displacement-gate evidence on e131 is one Adam family, one root,
   n=1, with a circular arithmetic "prediction" (D_kill is A0's own kill
   displacement) and a single independent test whose bracket [1.45, 3.64]
   can hide a 46% shift — "the same displacement within ~15%" is
   precision-washing an interpolation; and the e131 lineage has no
   static-jump leg at all (the 4-10x basin is g3's).**
3. **g1bW's machine record still asserts the wall-is-a-splint rescope and
   mislabels the reference's 0.0628 as "installs B" — while MUSEUM's fire
   was dose-guaranteed (W021 species: a bar whose fire is producible
   independent of the manipulated variable must carry its paired control
   as gating); the NOTES' verdict/consequence split is the honest form,
   the metrics clause is the propagating error, and "A survives an ACTIVE
   install" oversells an antagonist that installed nothing.**
