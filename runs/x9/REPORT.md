# x9 — THE RATE-FITS INSTRUMENT (T238's mispricing resolver)

**Status**: COMPLETE — verdict **MIXED**
**Envelope**: CPU-only desk cell (no torch import, threads 1, 0 GPU calls,
0 envelope-log writes, no NOTES/THINKING/QUEUE/STATE edits).
**Registration**: bars frozen VERBATIM in the script docstring, birth-committed
before compute (8ce682cc7ce7e9bf469c88cf75619e0ad26d9314).

## The question

x5 proved every battery carries its own one-T curve (ctrl 1.18-1.27 < fact-only
1.26-1.37 < pooled < near 1.42-1.63 < tmpl 1.51-1.65). T238 flagged the
mispricing: near/tmpl read "hotter" under one-T — genuinely faster decay
(RATE), or a different decay SHAPE that one-T translates into phantom
temperature? x7 proved the lens rescale-invariant, so the mispricing (if any)
is shape-real. This cell fits three decay families per battery per wash
(single-exponential EXP, stretched exponential STR, power-law POW) against the
bit-verified e238 discharge dumps and adjudicates the frozen bars.

## Gates (all machine-checked)

- G_SHA: x5's dump shas match its committed provenance; e238 dumps hashed.
- G_DUMP_IDENTITY: e238's ctrl/near subset bit-identical to x5's dumps
  (max |dlogit| = 0.0 across all 8 states).
- G_REPRO: one-T refit reproduces x5's 32 committed T rows to
  max |dT| = 1.10e-05; mean-p to max |dp| = 1.00e-05
  (the disclosed torch-float32 vs numpy-float64 softmax gap); t0 anchors read
  ctrl=1.00000, fact=1.00000, near=1.00000, tmpl=1.00000.

## The lambda/beta table (primary plane = stretched exponential)

| battery | lam_eff_bar (geo) | beta_bar | lam_eff w1 / w2 | beta w1 / w2 | EXP lam_bar (geo) | one-T deep ladder | gap share (x7) |
|---|---|---|---|---|---|---|---|
| ctrl | 0.0059 | 0.932 | 0.0062 / 0.0056 | 1.018 / 0.846 | 0.0060 | 1.219 | 1.147 |
| fact | 0.0076 | 1.307 | 0.0075 / 0.0078 | 1.267 / 1.346 | 0.0071 | 1.318 | 0.992 |
| near | 0.0422 | 1.076 | 0.0364 / 0.0490 | 1.124 / 1.027 | 0.0367 | 1.506 | 1.078 |
| tmpl | 0.0094 | 1.093 | 0.0107 / 0.0083 | 1.237 / 0.948 | 0.0091 | 1.551 | 0.951 |

L_ratio (HOT/COLD geometric-mean contrast) = **2.969**;
max |delta beta_bar| HOT-vs-COLD = **0.231**; beta spread over all four
batteries = **0.374**; EXP-family lambda ratio (co-report) =
2.812.

## Which family wins

AICc wins across the 8 battery-wash fits: EXP 7, STR
0, POW 0, one-T 1 —
overall winner: **EXP**. Family separation: dAICc < 2 among rate
families in 1/8 fits (blanket does not fire).

## T-induction (the mispricing localizer, co-report)

RMSE between each family's induced T(t) and the committed one-T T(t):

| battery | EXP | STR | POW |
|---|---|---|---|
| ctrl | 0.0098 | 0.0073 | 0.0362 |
| fact | 0.0267 | 0.0213 | 0.0762 |
| near | 0.0109 | 0.0104 | 0.1772 |
| tmpl | 0.0378 | 0.0322 | 0.1179 |

(The family whose induced T tracks the committed one-T ladder is the one the
lens was actually reading when it called near/tmpl "hot".)

## Verdict clause (frozen routing)

both shift: L_ratio = 2.969 > 2 AND |delta beta| = 0.231 >= 0.15

## The reading — MIXED decomposed (all numbers from metrics.json)

**The rate axis is NEAR's alone.** near's lambda_eff = 0.042 = 5.5-7.2x
ctrl/fact (0.0059/0.0076), bootstrap CIs near-disjoint (near/w1
[0.016, 0.053] vs ctrl/w1 [0.003, 0.009]), at beta_bar 1.08 ~ 1 — a pure
exponential dying fast. The EXP family's induced T(t) reproduces near's
committed one-T ladder to RMSE 0.011 / max 0.024: near's "hot" reading is
HONEST RATE (T236's public death, now in rate currency — and T238's
"the lens undercounts near" worry does not survive: near's rate premium maps
1:1 onto its T premium).

**tmpl — the battery T238 called hottest — is NOT a rate story.** Its
lambda_eff = 0.0094 is only 1.2-1.6x ctrl/fact (below the 2x bar on its own;
CI [0.006, 0.027] overlaps fact's [0.004, 0.014]). Its heat is the lens
reading SHAPE/residue: the one-T family OUTRIGHT WINS tmpl/w2 (AICc 69.55 vs
EXP 71.31 — the only non-EXP win in the table), and its committed T curve is
steeper-early/flatter-late than any exponential induces (committed w2
1.138/1.509/1.539 vs EXP-induced 1.150/1.448/1.585) — a saturating profile
the one-T family's own L0-anchored geometry produces. The mispricing's
magnitude is second-order: max |dT| = 0.062 (EXP) — a fine correction to the
pricing table, not a repricing.

**The beta leg of MIXED lives on FACT, not on near/tmpl.** beta_bar: ctrl
0.93, near 1.08, tmpl 1.09, fact 1.31 — the dB = 0.231 is the near-vs-fact
and tmpl-vs-fact contrast; fact carries the steepening tendency (bootstrap
lower bounds 1.14/1.15 > 1) on the COLD side. But STR never beats EXP by AICc
anywhere (NLL gains 0.005-0.13 nats — the shape parameter never earns its
seat), so the shape axis is thin everywhere; the frozen blanket (>= 4 of 8
inseparable) did not fire at 1/8, disclosed.

**Cross-references.** The deep-T ladder tracks the RATE axis, not the shape
axis (Spearman T-vs-lambda_eff 0.8 vs T-vs-beta 0.4, n=4 texture). The x7 gap
share is uniform and gap-dominated for all four batteries (0.95-1.15) — no
shape finding cross-references to a gap/scale split; the constructive gap
channel carries everything everywhere, consistent with x7's
CONSTRUCTIVE-FIELD.

**The mispricing in one sentence**: the one-T "hot" readings are honest
rate for near (lambda_eff 5-7x ctrl/fact at beta ~ 1, induced-T RMSE 0.011)
and a second-order shape artifact for tmpl (rate only 1.2-1.6x; one-T's own
family wins tmpl/w2 outright) — the lens prices near correctly, slightly
over-reads tmpl's heat (<= 0.06 T), and the T-ladder is to first order a
rate ladder (EXP wins 7 of 8).

Separation prescription (if underpowered): more states per trajectory (the committed grids give 3-4 post-wash points; 6-8 states bracket the mid-curve where EXP/STR/POW diverge) and more probes per battery — near n=3 above all (its bootstrap CI spans the whole plane)

## Honesty

n=1 organism; near n=3 (its bootstrap CI spans the plane — every near claim
is coarse); wash-1 CPU replay vs wash-2 GPU (inherited archive asymmetry);
beta pinned at bounds flagged in metrics; the one-T AICc counts one free T per
state (its L0 anchor is data, as the rate families' t0 anchors are); t0 never
scored (anchor + positive control); lens-not-mechanism throughout (T228/T229);
no bar shopping — the routing was frozen in the birth commit.
