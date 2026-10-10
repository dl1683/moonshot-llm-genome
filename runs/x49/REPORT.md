# x49 — THE AGE-525 DOSSIER (the R81 ideator's card; T329's alias-breaker) — ADJUDICATED: AGE-LOCKED (dossier EVENT-WITH-BEFORE-AFTER; P-x49a HIT)

*2026-10-10T23:04:04Z -> 2026-10-10T23:04:07Z | CPU desk only (threads 4, zero torch/CUDA) | birth commit bcc7b1cc0848ba3d1cdede79f1dfbd8119a110e9 | smoke=False*

## 0. The registration (frozen at birth, committed BEFORE compute)

**LAB LEAN (R81 parity, verbatim):** PHASE-LOCKED, weakly (the co-movement's p-values are strongest exactly where the design put shared phase, and x48's swap-anneal correlation says the cold arm shares the dynamics).

**COUNTER (grounds stated, verbatim):** AGE-LOCKED — the swap differs from the anneal in name AND start AND menu; if the tick needs the warm substrate plus age, a cold age-125 cell cannot fire, and the swap-anneal correlation could be carried entirely by the non-s125 steps (testable at desk: recompute it excluding s125).

**P-x49a (executor's own read, registered before compute):** AGE-LOCKED, weakly (against the lab lean)

**BARS (verbatim):** {
 "PHASE-LOCKED": "the swap's s125 (age 125) residual fires (negative, comparable magnitude to the warm ticks within a disclosed factor) \u2014 the tick is a reshape-STEP event; the phase clock is protocol-independent; W057's tick source strengthens.",
 "AGE-LOCKED": "the swap's s125 is silent while the warm age-525 pair fires \u2014 the tick tracks formation age (525-class), not step; the age clock revives.",
 "NO-EVENT": "neither structure holds (the warm pair was the coincidence R81's permutation already priced at p=0.035)."
}

**Frozen operationalization:** {
 "residual": "e344 N1 full-fit residual, refit exactly per e344/x48 (z(age) within the 82-row in-band pool; menu dummies cons/varied; rows as committed incl the duplicated walk_s700)",
 "swap_fire": "resid(swap_s125) < 0 AND |resid(swap_s125)| >= 0.5 * min(|warm ticks|)  [factor 2 disclosed]",
 "warm_fire": "both warm tick residuals <= 0 AND pool top-2 most-negative == {anneal_s125, walk_cons_s125} (recomputed) AND r(walk,anneal) reproduces 0.6673 within 0.005",
 "verdict_tree": "PHASE-LOCKED iff warm-fire AND swap-fire; AGE-LOCKED iff warm-fire AND NOT swap-fire; NO-EVENT iff NOT warm-fire",
 "dossier_grades": "per warm arm: EVENT iff |resid(s125)| >= 2x arm's largest non-s125 in-band |resid|; elif (mean(before s25-s100) - mean(after s150-s275)) >= 0.15 -> LEVEL-SHIFT; else POINT-ANOMALY; healing co-report mean(s250,s275) >= 0; composite EVENT-WITH-BEFORE-AFTER iff >=1 arm EVENT or LEVEL-SHIFT",
 "exclusion": "drop-one at s125 for all three star edges; edge 'carried-by-non-s125' iff r_excl >= 0.50 (co-report only)",
 "specificity": "swap_s125 negativity rank among the swap's in-band rows; swap's most-negative tag; dip rate at the fire threshold; raw-collapse census r1 <= 0.15 per arm (co-reports)",
 "P_T329a_clause": "the swap-s125 residual is nonzero and negative if the tick is phase-locked (H2), absent if age-locked \u2014 co-scored beside the dispatch's factor-2 form (the dispatch's bar adjudicates)"
}

## 1. THE ALIAS-BREAKER — the verdict

* refit reproduces e344's committed full-fit (R2 0.2793; betas exact to 5e-7); top-2 residuals == the warm ticks (gate).
* resid(anneal_s125) = **-0.5930**, resid(walk_cons_s125) = **-0.5756** (the warm pair, age 525, phase 125).
* resid(swap_s125) = **-0.2218** (the breaker cell: age 125, phase 125, COLD).
* fire line (disclosed factor 2): |resid| >= 0.2878; swap negative: True; swap fires: **False**.
* warm-fire terms: {'both_ticks_negative': True, 'pool_top2_are_the_warm_ticks': True, 'comove_reproduced': True}.
* **VERDICT: AGE-LOCKED** — the swap's s125 is silent while the warm age-525 pair fires — the tick tracks formation age (525-class), not step; the age clock revives.
* P-T329a clause co-score: nonzero-negative (T329's plain sign form; the dispatch's factor-2 form adjudicates).

## 2. THE SPECIFICITY + RAW-COLLAPSE CO-REPORTS (context, never bar-deciding)

* swap s125 negativity rank 2/22 (empirical rank p 0.0909); the swap's own most-negative in-band step: **swap_s250** (-0.3918); dip rate at the fire line 5%.
* raw-collapse census (in-band r1 <= 0.15): {"anneal": ["anneal_s125", "anneal_s725"], "walk_cons": ["walk_cons_s125"], "swap": [], "install_cold": []}
* near-collapses (0.15 < r1 <= 0.20): {"anneal": [], "walk_cons": ["walk_cons_s100", "walk_cons_s200"], "swap": [], "install_cold": []}

## 3. THE DOSSIER — event vs anomaly

* **anneal_s125** (ANNEAL-forming): LEVEL-SHIFT — 2x discontinuity bar False (tick -0.5930 vs 2x arm max 0.8890); before +0.0405 -> after -0.2463 (down-step >= 0.15: True); last2 -0.2833 (healed: False).
* **walk_cons_s125** (ROOT-forming): LEVEL-SHIFT — 2x discontinuity bar False (tick -0.5756 vs 2x arm max 0.7416); before +0.0734 -> after -0.0779 (down-step >= 0.15: True); last2 +0.1087 (healed: True).
* neighborhood dips vs s100/s150: {"anneal_s125": {"read_dip": true, "r1_dip": true}, "walk_cons_s125": {"read_dip": false, "r1_dip": true}, "swap_s125": {"read_dip": true, "r1_dip": true}}
* **COMPOSITE: EVENT-WITH-BEFORE-AFTER**

## 4. THE COLD-WARM GRID ({phase, age} x {warm, cold})

| cell | warm | cold |
|---|---|---|
| phase=125 | anneal_s125 + walk_cons_s125 — **FIRED** (-0.593 / -0.576) | swap_s125 — SILENT (-0.222, line 0.288) |
| age=125 | EMPTY (warm substrates post-install, age >= 400) | swap_s125 (the same state) |
| age=525 | the warm pair — the tick | EMPTY (deepest cold state zeph_s400, age 400) |

* install arm's sampled steps (e343/x42 grid — 125 never sampled): [1, 8, 9, 10, 11, 12, 100, 200, 300, 400]
* net0 classes recorded: {'anneal_s125': 'ANNEAL-forming', 'walk_cons_s125': 'ROOT-forming', 'swap_s125': 'BASE-forming'}

## 5. THE CO-MOVEMENT CONTEXT — the star with s125 dropped

| edge | r full | r excl s125 | delta | carried-by-non-s125 |
|---|---|---|---|---|
| walk-anneal | 0.6673 | 0.5053 | +0.1619 | True |
| swap-anneal | 0.7233 | 0.6742 | +0.0491 | True |
| swap-walk | 0.4862 | 0.3507 | +0.1355 | False |

* where the tick sits in the star: s125's largest single-edge drop-one contribution is to walk-anneal (delta +0.1619); every edge's r excl s125 is co-reported against the 0.50 carried-by-non-s125 line.

## 6. Gates

* 33 sub-gates across 9 families: {"G_MD5": "PASS", "G_POOL": "PASS", "G_N1REFIT": "PASS", "G_COMOVE": "PASS", "G_DOSSIER": "PASS", "G_PROFILES": "PASS", "G_GRID": "PASS", "G_PARITY": "PASS", "G_ENVELOPE": "PASS"}
* all_pass: True

## 7. P-x49a scorecard

* guess: AGE-LOCKED, weakly (against the lab lean); verdict: AGE-LOCKED; **HIT** (rule: HIT iff verdict == AGE-LOCKED)

## 8. The profiles (committed rows, copied intact)

### anneal
| tag | age | read | r1 | r2 | alive g0 | ce_r | write_norm | in_room | net0 | resid_full |
|---|---|---|---|---|---|---|---|---|---|---|
| anneal_s25 | 425 | 0.684 | 0.996 | 0.995 | 1.00 | 1.848 | 11.4 | 0.730 | ANNEAL-forming | +0.230 |
| anneal_s50 | 450 | 0.803 | 0.905 | 0.934 | 1.00 | 1.810 | 12.5 | 0.656 | ANNEAL-forming | +0.152 |
| anneal_s75 | 475 | 0.382 | 0.530 | 0.768 | 0.93 | 1.731 | 13.3 | 0.609 | ANNEAL-forming | -0.211 |
| anneal_s100 | 500 | 0.454 | 0.720 | 0.797 | 0.92 | 1.726 | 14.0 | 0.573 | ANNEAL-forming | -0.009 |
| anneal_s125 | 525 | 0.309 | 0.123 | 0.366 | 0.80 | 1.740 | 14.6 | 0.542 | ANNEAL-forming | -0.593 |
| anneal_s150 | 550 | 0.428 | 0.630 | 0.695 | 0.87 | 1.742 | 15.3 | 0.515 | ANNEAL-forming | -0.074 |
| anneal_s175 | 575 | 0.576 | 0.398 | 0.555 | 0.83 | 1.732 | 15.8 | 0.492 | ANNEAL-forming | -0.294 |
| anneal_s200 | 600 | 0.694 | 0.470 | 0.557 | 0.92 | 1.710 | 16.4 | 0.471 | ANNEAL-forming | -0.210 |
| anneal_s225 | 625 | 0.737 | 0.335 | 0.333 | 0.93 | 1.685 | 16.9 | 0.454 | ANNEAL-forming | -0.333 |
| anneal_s250 | 650 | 0.772 | 0.211 | 0.241 | 0.95 | 1.714 | 17.4 | 0.437 | ANNEAL-forming | -0.445 |
| anneal_s275 | 675 | 0.561 | 0.521 | 0.682 | 0.83 | 1.696 | 17.9 | 0.422 | ANNEAL-forming | -0.122 |

### walk_cons
| tag | age | read | r1 | r2 | alive g0 | ce_r | write_norm | in_room | net0 | resid_full |
|---|---|---|---|---|---|---|---|---|---|---|
| walk_cons_s25 | 425 | 0.653 | 0.892 | 0.901 | 0.95 | 1.795 | 15.9 | 0.060 | ROOT-forming | +0.322 |
| walk_cons_s50 | 450 | 0.759 | 0.924 | 0.910 | 0.92 | 1.730 | 16.7 | 0.060 | ROOT-forming | +0.367 |
| walk_cons_s75 | 475 | 0.452 | 0.520 | -0.088 | 0.90 | 1.730 | 17.3 | 0.060 | ROOT-forming | -0.025 |
| walk_cons_s100 | 500 | 0.645 | 0.162 | 0.169 | 0.97 | 1.705 | 17.8 | 0.060 | ROOT-forming | -0.371 |
| walk_cons_s125 | 525 | 0.736 | -0.055 | 0.383 | 1.00 | 1.709 | 18.3 | 0.060 | ROOT-forming | -0.576 |
| walk_cons_s150 | 550 | 0.562 | 0.461 | 0.415 | 0.90 | 1.703 | 18.8 | 0.060 | ROOT-forming | -0.048 |
| walk_cons_s175 | 575 | 0.492 | 0.285 | 0.604 | 0.85 | 1.689 | 19.3 | 0.060 | ROOT-forming | -0.211 |
| walk_cons_s200 | 600 | 0.768 | 0.159 | 0.236 | 1.00 | 1.671 | 19.8 | 0.060 | ROOT-forming | -0.324 |
| walk_cons_s225 | 625 | 0.678 | 0.369 | 0.458 | 0.87 | 1.682 | 20.2 | 0.060 | ROOT-forming | -0.102 |
| walk_cons_s250 | 650 | 0.567 | 0.502 | 0.004 | 0.83 | 1.686 | 20.7 | 0.060 | ROOT-forming | +0.043 |
| walk_cons_s275 | 675 | 0.663 | 0.621 | 0.337 | 0.83 | 1.701 | 21.1 | 0.060 | ROOT-forming | +0.174 |

### swap
| tag | age | read | r1 | r2 | alive g0 | ce_r | write_norm | in_room | net0 | resid_full |
|---|---|---|---|---|---|---|---|---|---|---|
| swap_s25 | 25 | 0.687 | 0.956 | 0.966 | 1.00 | 1.837 | 7.3 | 0.061 | BASE-forming | +0.191 |
| swap_s50 | 50 | 0.732 | 0.958 | 0.916 | 1.00 | 1.814 | 9.1 | 0.060 | BASE-forming | +0.205 |
| swap_s75 | 75 | 0.488 | 0.872 | 0.886 | 0.83 | 1.754 | 10.3 | 0.060 | BASE-forming | +0.131 |
| swap_s100 | 100 | 0.495 | 0.745 | 0.808 | 0.98 | 1.750 | 11.3 | 0.060 | BASE-forming | +0.017 |
| swap_s125 | 125 | 0.395 | 0.494 | 0.630 | 0.90 | 1.718 | 12.1 | 0.061 | BASE-forming | -0.222 |
| swap_s150 | 150 | 0.571 | 0.544 | 0.693 | 0.85 | 1.735 | 12.9 | 0.061 | BASE-forming | -0.160 |
| swap_s175 | 175 | 0.639 | 0.709 | 0.815 | 0.88 | 1.726 | 13.6 | 0.061 | BASE-forming | +0.017 |
| swap_s200 | 200 | 0.738 | 0.678 | 0.836 | 0.95 | 1.681 | 14.3 | 0.061 | BASE-forming | -0.001 |
| swap_s225 | 225 | 0.746 | 0.665 | 0.763 | 0.93 | 1.700 | 14.9 | 0.061 | BASE-forming | -0.002 |
| swap_s250 | 250 | 0.433 | 0.263 | 0.463 | 0.85 | 1.695 | 15.5 | 0.061 | BASE-forming | -0.392 |
| swap_s275 | 275 | 0.535 | 0.792 | 0.862 | 0.82 | 1.717 | 16.1 | 0.061 | BASE-forming | +0.149 |

## Provenance

* birth commit: bcc7b1cc0848ba3d1cdede79f1dfbd8119a110e9 (the registration commit, pushed BEFORE compute); parents md5-bound: e344_metrics 1a3598fb, x48_metrics 370e30c0, x45_metrics 1dfcfeb3, e343_metrics 957b4074, e341_metrics 279bd235
* CPU desk only (threads 4, zero CUDA); timestamps datetime.now(UTC); no NOTES/THINKING/QUEUE/STATE edits (the heartbeat folds this cell); smoke -> runs/x49_smoke/
