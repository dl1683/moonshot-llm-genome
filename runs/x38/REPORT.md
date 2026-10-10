# X38 — THE FLIP THRESHOLD (2026-10-10T12:40:41+00:00)

**VERDICT: SHARP-FLIP** — a threshold read level R* exists: rungs below it LIFT (ratio > 1) and rungs above it COLLAPSE (move > ~5x band) with NO rung ambiguous (1 LIFT / 5 COLLAPSE / 0 ambiguous; the split is monotone in read level) — R* is BRACKETED in (1.338e-05, 0.530222]; THE TWO-CHANNEL LAW GAINS ITS CONTROL PARAMETER; the flip is a state property, and R* joins the doc's Law 7

## The question (verbatim)

> WHERE between prior (1.3e-5) and living (0.745) does the flip happen? — x37/e337 found THE SIGN FLIP: the same K10K complement LIFTS the empty Z-slot 54.6x (base) but COLLAPSES living Z-reads (root -0.53, ball -0.41 in move currency). The two-channel law's sharpest open parameter.

## P-x38a (verbatim) + scoring

> Register P-x38a BEFORE compute. Lab lean: SHARP-FLIP, weakly — the sign flip across x37's three substrates was total (all-or-nothing at the slot), and the softmax denominator is a thresholding machine; the countervailing: the middle rungs (0.53-0.56) sit near the wash-survival margin and may behave mixed. State your own read.

- P-x38a outcome: **HIT — SHARP-FLIP**
- Executor position: 4/4 registered bits — HIT ({"sharp_flip": true, "collapse_level_independent_envelope_le_2x": true, "all_living_rungs_collapse": true, "r_bracket_between_empty_and_lowest_living": true})

## THE LADDER (host contexts; both currencies per rung)

| rung | net0 class | prior | comp read | lift (ratio currency) | signed move | move / band (move currency) | band_move | class |
|---|---|---|---|---|---|---|---|---|
| base | BASE | 1.3384e-05 | 7.3080e-04 | 54.60x | +0.0007 | 0.01x | 0.0729 | LIFT |
| own_inst | BASE-formed | 5.3022e-01 | 3.5971e-01 | 0.68x | -0.1705 | 11.87x | 0.0144 | COLLAPSE |
| inst_sib | BASE-formed | 5.5631e-01 | 1.2378e-01 | 0.22x | -0.4325 | 6.93x | 0.0625 | COLLAPSE |
| ball | ROOT+washed+committed | 7.0430e-01 | 2.9813e-01 | 0.42x | -0.4062 | 7.68x | 0.0529 | COLLAPSE |
| root | ROOT | 7.4475e-01 | 2.1787e-01 | 0.29x | -0.5269 | 7.75x | 0.0679 | COLLAPSE |
| locked | ROOT-locked | 7.8504e-01 | 1.1628e-01 | 0.15x | -0.6688 | 11.00x | 0.0608 | COLLAPSE |

**R\* = (1.3384e-05, 0.530222]** — bracketed between the highest LIFT rung and the lowest COLLAPSE rung.

Neutral-site rider (site-generality, never adjudicated): base +0.0006; own_inst -0.0054; inst_sib -0.0067; ball -0.0120; root -0.0089; locked -0.0167

## THE RIDER (the gaussian at 1x/2x/4x on the living root)

| dose | L2 | Z at host | signed move | move/band | beyond 5x band? |
|---|---|---|---|---|---|
| 1x | 9.1788 | 0.7441 | -0.0007 | 0.01x | False |
| 2x | 18.3577 | 0.6738 | -0.0710 | 1.04x | False |
| 4x | 36.7154 | 0.0127 | -0.7320 | 10.77x | True |

**Rider word: GAUSS-BREAKS-AT-4x** (registered expectation NOISE-FLOOR-HOLDS-AT-4X; MISS — descriptive only, no bar).

## Gates

18/18 gates PASSED: G_PANEL, G_NAMEFREE, G_BATTERY, G_NEUTRAL, G_ENDPOINTS, G_FLATBASIS, G_LADDERLOAD, G_ROOM, G_ORTH, G_DOSEMATCH, G_INROOM, G_REGEN, G_GAUSS, G_ONESTATE, G_X24REPRO, G_X37REPRO, G_E337REPRO, G_CER

## Deviations (registered at birth)

- CPU-ONLY cell (dispatch: e338 owns the GPU lane) — CUDA_VISIBLE_DEVICES='' before torch import, torch threads 4 (x37's exact setting, reproduction-critical), pocketfft workers 4, no GPU code path, no envelope-log writes, no other runs/ touched.
- THE x16 OFF-TARGET BATTERY IS NOT RUN (disclosed omission): the dispatch's design names only the ladder + the fixed displacement + the gaussian rider; the off-target axis on base/root/locked/ball is x37/e337's committed record (their cells gated bit-exact here).
- THE RIDER runs on the LIVING ROOT only (the dispatch's naming), at 1x/2x/4x doses of the SAME seed-25001 direction; the 1x dose reproduces x37's committed root_gauss cells bit-exact (the rider's end-to-end certification); 2x/4x are new.
- The ladder's two mid-rungs (e048_repro, g1c_install_resume) are NEW SUBJECTS for the panel (e335 read them at the walk's battery geometry; x37/e337 never panel-probed them) — their bare reads are gated vs e335's committed walk reads (2e-6); their panel cells are this cell's live territory.
- The displacement vectors are FIXED vectors built on the base lineage (the K10K complement from x15's carrier) applied IDENTICALLY to every rung — substrate_fp32 + delta_fp32 per key (x15's fp32 method). On the base this re-derives the carrier's model bit-exact (gated).
- THE BALL'S SETTLE (e337's registered instrument, inherited): the artifact's raw body is one optimizer step OUTSIDE the wall; load_g1 -> _enforce_wall settles; THE SUBSTRATE IS THE SETTLED STATE; pre/post wall_report recorded.
- ROOT == CONS re-gated in-cell (e335's G_ROOTIDENT, bit-exact) — the dispatch's 'root/cons' rung is one object, certified here.
- The rung checkpoints are artifacts that may carry anchor buffers; all flat-space math uses the PARAMETER keys only (gated: key order identical across all six rungs, N = 2,739,072).
- MAMILLIUS/KING/LEONTES/ELIZABETH/FLORIZEL priors differ per rung (different organisms); the cohort is frozen by CONSTRUCTION (x24's list), measured priors co-reported per site.
- n=1 per rung (one draw per construction step, one session — the g-series standing lottery note); the bit-exact x37/e337 controls double as the in-cell instrument replicate.
- No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat folds).
- Smoke (X38_SMOKE=1): full gate path + all reads live (no battery exists to shrink), own (gitignored) smoke dir; NOTHING adjudicated.
