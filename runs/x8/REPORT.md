# x8 — THE FLASH THROUGH THE THERMAL LENS (first pass) — REPORT

**Cell**: `lab/x8_flash_thermal.py` (birth commit `b12f981`, bars frozen in the
docstring BEFORE any compute). CPU-only desk cell, threads<=4, no GPU, no
envelope-log writes. Data read at runtime from
`runs/e268..e271/metrics.json` with 13 equality gates (embedded literals +
e271's verbatim ladder copy vs the originals) — ALL PASS.

**Verdict: UNDERPOWERED** (the frozen composite's override clause fired).
W041's registered prediction **P-W41d ("x8 returns UNDERPOWERED") is
CONFIRMED** as a scored third-party prediction. But the verdict word is the
LEAST of what the cell measured — the point estimates it carries verbatim
are unanimously storage-flavored, and the exact sense in which 5 points
cannot discriminate is now a number, not a shrug. Details below.

---

## 1. The one-T lens on the flash: rejected at every rung (point estimates)

One-T := the x4/x5 thermal lens reduced to scalar milestones (the two-class
exact form of softmax(L/T)): **z_c(t) = z_s(t)/T**, one scalar per rung, LS
through the origin on s100..s400, z = ln(p/(1-p)).

| rung | T_full | R2_z (lens) | T_rise | T_decay | I = limb incons. | sign flip |
|------|--------|------------|--------|---------|------------------|-----------|
| 10k (e268)  | 0.1366 | **-3.31** | 0.150 (n=3) | 0.141 (n=2) | 0.067 | no |
| 40k (e269)  | 0.1121 | **-8.89** | 0.139 (n=1) | 0.112 (n=4) | 0.215 | no |
| 100k (e270) | 0.0971 | **-7.62** | 0.074 (n=2) | 0.098 (n=3) | 0.278 | no |
| 237k (e271) | 0.8095 | **-42.45** | **-14.84** (n=2) | 0.249 (n=3) | **2.000 (max)** | **YES** |

Every R2 is *deeply negative* — the single-scalar lens is worse than a
horizontal line at the flash's own mean, at every rank. The STORAGE bar's
R2 clause (any rung < 0.7) fires 4/4 at the point estimates. The top rung
adds the limb clause in its extreme form: the rise limb demands a NEGATIVE
temperature (T_rise = -14.8) while the decay limb demands T = 0.25 — a
sign disagreement, I = 2.0 (the metric's maximum).

**Structural, noise-free observation at 237k**: the serial latent's logit
CROSSES ZERO (p_s = 0.589 at s100, 0.553 at s300) while the concurrent
probe never exceeds 0.073 (z_c in [-3.9, -2.5]). No positive temperature
maps a positive logit to -3.2. The gain reading of the flash is not merely
a bad fit at the natural width — it is *unrepresentable* in the family.
Also: fitted T is NOT monotone in rank (0.137, 0.112, 0.097, 0.81) — the
top rung breaks the lens's own rank pattern exactly where the flash broke
its saturation law (T249).

## 2. The kinetic race (alpha, beta): better than one-T, still nowhere clean

dX/dt = alpha*G(t) - beta*X, G = the rung's own serial formation curve
(consult #005's kinetic form), X(0) = the concurrent floor, fit on
s100..s400 in p-space.

| rung | alpha | beta | alpha/beta | tau (steps) | R2_p | fitted X(400)/serial | measured ratio |
|------|-------|------|-----------|-------------|------|----------------------|----------------|
| 10k  | 1.90e-05 | 0.0125 | 0.00153 | 80 | **-0.08** | 0.00135 | 0.000151 |
| 40k  | 0.0774 | **3.00 (grid edge)** | 0.0258 | 0.33 | **-0.10** | 0.0258 | 0.00943 |
| 100k | 4.41e-04 | 0.0146 | 0.0302 | 68 | **+0.32** | 0.0277 | 0.0250 |
| 237k | 1.59e-03 | 0.0153 | 0.104 | 65 | **+0.20** | 0.122 | 0.0911 |

- The race **does better than one-T on the top two rungs** (R2_p +0.32/+0.20
  vs lens R2_z -7.6/-42.5) and no worse at the bottom — but 0.32 is nowhere
  near "clean" (READOUT's own bar was 0.9). **Neither family fits the flash.**
- 40k's fit ran to the beta grid edge (b=3, tau = 1/3 step): the optimizer
  wants quasi-static scaled tracking there (X ~ 0.026*G), which the
  monotone-declining 40k flash (0.0245 -> 0.0033, a 7.5x fall while G stays
  at 0.35-0.48) simply is not — a first-order (formation - wash) race with a
  healthy driver cannot overshoot and crash 7.5x below its quasi-static
  level.
- The one clean trend: **alpha/beta is monotone in rank** (0.0015 -> 0.026 ->
  0.030 -> 0.104; Spearman 1.0, n=4 disclosed) — the race's quasi-static
  retention level tracks the survival-ratio ladder's direction
  (1.5e-4 -> 9.4e-3 -> 2.5e-2 -> 9.1e-2), overshooting the measured endpoint
  ratio 9x / 2.7x / 1.2x / 1.1x — converging as rank grows. The race
  "understands" the ladder's direction but not its magnitude at low rank.
- **Driver-following is not the ceiling problem**: the race fitted to the
  serial curve with itself as driver reaches R2 = 0.9999 (fast-tracking
  limit) — so its failure on the concurrent flash is about the flash's
  SHAPE, not milestone granularity.

## 3. The shape the families can't produce: antiphase

Pearson(serial, concurrent) over the four interior milestones, per rung:
**-0.52 (10k), -0.20 (40k), -0.24 (100k), -0.71 (237k)** — negative at
every rank. The flash PEAKS where the driver dips (237k: flash peak 0.073
at s200, the serial's local minimum 0.368). A gain maps the driver's shape
positively; a lagged race rotates phase but not into uniform
anti-correlation. Descriptive only (n=4) — but it is the qualitative
datum both families miss: **the flash has its own shape, not the driver's.**

## 4. T249's decay-shape discriminator (co-report)

On the decay limb in logit space, linear-in-t (H-POISON's dose-linear
erosion) vs exponential-in-t (first-order wash), both 2-param:

| rung | n | linear R2 | exp R2 | winner |
|------|---|-----------|--------|--------|
| 40k  | 4 | 0.969 | 0.945 | linear (+0.024) |
| 100k | 3 | 0.332 | 0.304 | linear (+0.028) |
| 237k | 3 | 0.878 | 0.850 | linear (+0.028) |
| 10k  | 2 | — | — | uninformative |

Weak, consistent texture: **linear (dose-erosion) wins all three
informative rungs**, by slim margins (~0.03 R2). This mildly favors
H-POISON's erosion currency on the decay limb — against T249's
non-monotone-tail lean toward H-TRUE-FORMATION. Too thin to adjudicate
anything alone; recorded for e273's harvest.

## 5. Why UNDERPOWERED and not STORAGE-LIKE (the frozen arithmetic)

The STORAGE clauses fired 5 drivers (4x R2<0.7, 1x limb-inconsistency).
The frozen override requires stability: a driver counts only if its
bootstrap 90% CI stays on the firing side. Under the frozen S1 noise model
(sigma = the lens's own residual RMS, 2.2-3.8 z-units):

- CI90 of refit R2_z: [-2.53, 0.86], [-1.89, 0.90], [-2.25, 0.87],
  [-2.93, 0.75] — every upper bound reaches >= 0.7 → every R2 driver
  "unstable"; P(R2<0.7) = 0.80/0.71/0.83/0.93 (majority-stable, tail-not).
- 237k's limb driver: P(I>0.5) = 0.71, CI90_I = [0.07, 2.0] → crosses 0.5.

No stable driver → **UNDERPOWERED**, point estimates carried verbatim.

**The honest decomposition of what that means** (co-reported in metrics as
`verdict_sensitivity_disclosure`):
- At the *point estimates*, the read is unanimous: every rung's lens
  misfit is catastrophic, and 237k's limb temperatures disagree in sign.
- Under **S2** (the only fluctuation scale visible in-data: the serial
  driver's detrended z-wobble, 0.18-0.35 z-units), the observed lens
  misfit lies **far outside the noise null at every rung** — P(R2<0.7 |
  lens true, S2) = 0.000/0.000/0.000/0.41 vs observed R2 = -3.3..-42.5
  (S2 null bands top out at 0.96-0.999). Under S2 the one-T family is
  *calibrated-rejected* at all four rungs, and the verdict would have been
  STORAGE-LIKE.
- Under **S1** (noise as large as the misfit itself), nothing is stable.
- **Which sigma is right is unknowable at n=1 corpus seed per rung** — the
  arms are deterministic to ~1e-6 run-to-run (the e261-vs-session
  replicate), so all variance is corpus-SEED variance and the flash's seed
  variance has never been measured. THAT is the precise sense in which
  5 points x 1 seed cannot discriminate: not ambiguity in the point
  estimates (there is none), but an unmeasured noise scale under which the
  same numbers flip the frozen bar.

## 6. What would settle it (quantitative prescription)

- **S1 multipliers (half-bar / R2-width-0.1):** up to 1.3e6x / 1.0e4x —
  no feasible grid settles anything under S1 assumptions.
- **S2 multipliers:** <= 0.12x at 10k/40k/100k (already sufficient); at
  237k: 98.7x (limb half-bar) and 215x (R2 width 0.1) — the top rung is
  the binding constraint, because T_rise's 2-point rise limb has almost no
  driver leverage (sum z_s^2 ~ 0.42 vs ~0.56-5.1 elsewhere).
- **Design**: (1) FIRST a 3-corpus-seed pilot on 100k+237k (same rig,
  same grid) purely to MEASURE sigma_seed — the one number the verdict
  hinges on; (2) if S2-like (~0.3 z): milestone grid every 2 install steps
  (200 interior points, 50x) x J=5 seeds (N_eff ~ 250x) on the top two
  rungs — covers S2's worst 215x; (3) if S1-like (~3 z): no grid is worth
  buying — go straight to the full-logit MLE dump mini-cell (the x5 form;
  this desk cell's disclosed reduction is the two-class logit-gain lens,
  and the proper instrument at that point is full logits, not more
  milestones).

## 7. Conjunction reads (for the harvest, not adjudicated here)

- W041's x8 x e273 map: this cell returns the card's UNDERPOWERED branch —
  "the verdict word waits for the instrument". The seed pilot above IS the
  instrument, and it rides e273/e275's rig at ~zero cost (2 extra corpus
  seeds x 2 rungs, milestone grid unchanged).
- The structural 237k zero-crossing argument (section 1) is
  seed-independent and survives any sigma: at the natural width the flash
  is not a scaled serial latent. Whichever mechanism (H-POISON's
  v-denominator or H-TRUE-FORMATION's wash) makes the flash, it does so
  with the flash's OWN shape.
- The antiphase texture + the race's inability to overshoot (section 2-3)
  both point the same way as x7's question: the measurement channel and
  the write's expression are not cleanly separable objects at 5-milestone
  resolution.

## Disclosures

- Desk reduction: one-T here is the logit-gain (two-class) reduction of the
  x5 full-logit MLE family; no milestone logit dumps exist — the full form
  is a named follow-up, not this cell.
- s1 (the shared pre-install floor) is the race's initial condition and is
  excluded from all lens fits (frozen; it is an initial condition, not a
  channel read).
- dof: lens k=1, race k=2, n=4 fit points per curve; 40k's rise limb is a
  single point (T_rise = z_s/z_c exactly, flagged degenerate).
- 40k's race beta sits ON the grid edge (3.0) — the unconstrained optimum
  wants faster erosion; the fit is a boundary fit, disclosed.
- Bootstrap sigmas are PROXIES (S1 = misfit scale; S2 = driver-wobble
  scale); seed-level variance unmeasured; the bootstrap can only
  understate it (verbatim caveat in metrics.json).
- Post-birth script edit (narrative-only): after the first compute pass the
  generated prescription sentence claimed a coverage its own numbers
  refuted; it was replaced with the computed honest text plus the
  `S2_calibration_read` / `verdict_sensitivity_disclosure` co-reports.
  Bars, families, R2 definitions, limb rules, composite order and all
  numbers unchanged (re-run reproduces identical fits, seed 20261005).
  The birth commit preserves the original for audit.
- W041 P-W41d scored: CONFIRMED (verdict word UNDERPOWERED).

**Outputs**: `runs/x8/metrics.json` (complete, adjudicated),
`runs/x8/x8_flash_fits.png` (4 rung panels, both limbs annotated, lens +
race overlays; 2 rank-trend panels), this report.
