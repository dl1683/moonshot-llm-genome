# X47 — THE SLOT-PROPERTY n=2 (R*(ZEPHYRA) vs R*(TAVIREN); the flip threshold's second slot)

* VERDICT: **UNRESOLVABLE** — too few two-sided brackets on the ZEPHYRA side — 3 two-sided-consistent cells (bar: >= 2, x44's own precedent), pooled bracket NON-EXISTENT (constraint contradiction); 15 lower + 18 upper constraints — the honest gap outcome; what resolved is reported in the table
* P-x47a (my registered read): MISSED — guess SLOT-PROPERTY-CONFIRMED, weakly (concurring with the lab lean); bits {"a_verdict_confirmed": false, "b_max_lower_in_window": false, "c_s9_two_sided": false, "d_ratio_in_0p5_1p7": false}
* R*(ZEPHYRA) pooled: {"max_lower": 0.14045459032058716, "min_upper": 0.05861958488821983, "consistent": false, "R_hat": null, "n_two_sided_consistent": 3, "two_sided_keys": ["s10:g-12", "s10:g0", "s10:g+12"], "n_lower_constraints": 15, "n_upper_constraints": 18}
* R*(TAVIREN) reference (x44, runtime-read): {"max_lower": 0.04419664666056633, "min_upper": 0.05274265632033348} -> R_T 0.04828; ratio R_Z/R_T = n/a
* bracket overlap with the TAVIREN bracket: False
* gates: 19/19 PASS

## 1. The ZEPHYRA ladder (the rung-level picture; x38's rung rule on this cell's reconstruction)

| rung | class (x42 record) | prior (g0) | post (g0) | rung class (mine) | lift | signed move | ce_r bare->comp |
|---|---|---|---|---|---|---|---|
| base | BASE | 1.338e-05 | 0.0007308 | LIFT (ratio x54.60) | 54.6 | +0.0007174 | 1.6161 -> 2.0603 |
| s1 | BASE-forming | 1.518e-05 | 0.0007921 | LIFT (ratio x52.17) | 52.2 | +0.000777 | 1.6137 -> 2.0618 |
| s8 | BASE-forming | 0.02172 | 0.02702 | LIFT (ratio x1.24) | 1.24 | +0.005293 | 1.5991 -> 2.0895 |
| s9 | BASE-forming | 0.1143 | 0.0464 | MIXED (move x0.12) | 0.406 | -0.06792 | 1.5962 -> 2.0895 |
| s10 | BASE-forming | 0.3172 | 0.06909 | MIXED (move x0.74) | 0.218 | -0.2481 | 1.5955 -> 2.0960 |
| s11 | BASE-forming | 0.4557 | 0.09604 | MIXED (move x1.03) | 0.211 | -0.3597 | 1.5970 -> 2.1010 |
| s12 | BASE-forming | 0.5052 | 0.1261 | MIXED (move x1.08) | 0.25 | -0.3791 | 1.6000 -> 2.1040 |
| own_inst | BASE-formed | 0.5302 | 0.3597 | MIXED (move x1.31) | 0.678 | -0.1705 | 1.6736 -> 2.0952 |
| inst_sib | BASE-formed | 0.5563 | 0.1238 | MIXED (move x2.00) | 0.223 | -0.4325 | 1.6668 -> 2.1248 |
| ball | ROOT+washed+committed | 0.7043 | 0.2981 | MIXED (move x2.14) | 0.423 | -0.4062 | 1.6857 -> 1.9744 |
| root | ROOT | 0.7448 | 0.2179 | MIXED (move x2.77) | 0.293 | -0.5269 | 1.6861 -> 2.0110 |
| locked | ROOT-locked | 0.785 | 0.1163 | MIXED (move x4.15) | 0.148 | -0.6688 | 1.6635 -> 2.0670 |

## 2. The per-context flip cells (the primary band form)

| rung | geo | pre mean (alive) | post mean (alive) | L/C/A | L_max | C_min | R_hat | viol | floor | satup |
|---|---|---|---|---|---|---|---|---|---|---|
| ball | g-12 | 0.9406 (1.00) | 0.1557 (0.65) | 0/35/25 | None | 0.39526 | None | 0 | 0 | 0 |
| ball | g0 | 0.7043 (0.93) | 0.2981 (0.70) | 0/44/16 | None | 0.05936 | None | 0 | 0 | 9 |
| ball | g+12 | 0.9576 (1.00) | 0.1085 (0.57) | 0/29/31 | None | 0.84175 | None | 0 | 0 | 0 |
| base | g-12 | 0.0000 (0.00) | 0.0007 (0.00) | 60/0/0 | 4e-05 | None | None | 0 | 0 | 0 |
| base | g0 | 0.0000 (0.00) | 0.0007 (0.00) | 59/0/1 | 4e-05 | None | None | 0 | 0 | 0 |
| base | g+12 | 0.0000 (0.00) | 0.0008 (0.00) | 59/0/1 | 3e-05 | None | None | 0 | 0 | 0 |
| inst_sib | g-12 | 0.1657 (0.85) | 0.0048 (0.00) | 0/6/54 | None | 0.05862 | None | 0 | 0 | 0 |
| inst_sib | g0 | 0.5563 (1.00) | 0.1238 (0.52) | 1/36/23 | 0.50688 | 0.23482 | None | 11 | 0 | 0 |
| inst_sib | g+12 | 0.1131 (0.73) | 0.0019 (0.00) | 0/3/57 | None | 0.06972 | None | 0 | 0 | 0 |
| locked | g-12 | 0.9156 (1.00) | 0.0157 (0.03) | 0/60/0 | None | 0.22975 | None | 0 | 0 | 0 |
| locked | g0 | 0.7850 (0.98) | 0.1163 (0.57) | 3/55/2 | 0.19052 | 0.0548 | None | 5 | 0 | 4 |
| locked | g+12 | 0.9478 (1.00) | 0.0214 (0.08) | 0/60/0 | None | 0.61607 | None | 0 | 0 | 0 |
| own_inst | g-12 | 0.1689 (0.95) | 0.1291 (0.65) | 15/27/18 | 0.18514 | 0.05853 | None | 32 | 0 | 13 |
| own_inst | g0 | 0.5302 (1.00) | 0.3597 (0.68) | 5/48/7 | 0.50805 | 0.19926 | None | 24 | 0 | 20 |
| own_inst | g+12 | 0.1329 (0.92) | 0.0616 (0.42) | 8/21/31 | 0.11115 | 0.08611 | None | 8 | 0 | 6 |
| root | g-12 | 0.9026 (1.00) | 0.0772 (0.48) | 0/44/16 | None | 0.33073 | None | 0 | 0 | 0 |
| root | g0 | 0.7448 (1.00) | 0.2179 (0.67) | 4/49/7 | 0.48755 | 0.09686 | None | 15 | 0 | 5 |
| root | g+12 | 0.9289 (1.00) | 0.0473 (0.38) | 0/44/16 | None | 0.69985 | None | 0 | 0 | 0 |
| s1 | g-12 | 0.0000 (0.00) | 0.0007 (0.00) | 60/0/0 | 5e-05 | None | None | 0 | 0 | 0 |
| s1 | g0 | 0.0000 (0.00) | 0.0008 (0.00) | 59/0/1 | 4e-05 | None | None | 0 | 0 | 0 |
| s1 | g+12 | 0.0000 (0.00) | 0.0008 (0.00) | 59/0/1 | 3e-05 | None | None | 0 | 0 | 0 |
| s10 | g-12 | 0.2567 (0.67) | 0.0724 (0.57) | 1/8/51 | 5e-05 | 0.4147 | 0.00464 | 0 | 0 | 0 |
| s10 | g0 | 0.3172 (0.68) | 0.0691 (0.48) | 1/12/47 | 7e-05 | 0.43458 | 0.00532 | 0 | 0 | 0 |
| s10 | g+12 | 0.2407 (0.68) | 0.0525 (0.42) | 1/5/54 | 5e-05 | 0.39332 | 0.00465 | 0 | 0 | 0 |
| s11 | g-12 | 0.3988 (0.68) | 0.1024 (0.60) | 0/26/34 | None | 0.44538 | None | 0 | 0 | 0 |
| s11 | g0 | 0.4557 (0.68) | 0.0960 (0.57) | 0/26/34 | None | 0.62983 | None | 0 | 0 | 0 |
| s11 | g+12 | 0.3787 (0.68) | 0.0742 (0.50) | 0/23/37 | None | 0.50533 | None | 0 | 0 | 0 |
| s12 | g-12 | 0.4536 (0.68) | 0.1336 (0.63) | 0/28/32 | None | 0.51506 | None | 0 | 0 | 0 |
| s12 | g0 | 0.5052 (0.68) | 0.1261 (0.62) | 0/26/34 | None | 0.69468 | None | 0 | 0 | 0 |
| s12 | g+12 | 0.4323 (0.68) | 0.0982 (0.57) | 0/25/35 | None | 0.58621 | None | 0 | 0 | 0 |
| s8 | g-12 | 0.0160 (0.05) | 0.0251 (0.17) | 42/0/18 | 0.05414 | None | None | 0 | 0 | 0 |
| s8 | g0 | 0.0217 (0.08) | 0.0270 (0.20) | 37/0/23 | 0.07676 | None | None | 0 | 0 | 0 |
| s8 | g+12 | 0.0149 (0.03) | 0.0201 (0.10) | 36/0/24 | 0.06114 | None | None | 0 | 0 | 0 |
| s9 | g-12 | 0.0834 (0.65) | 0.0457 (0.37) | 8/0/52 | 0.11225 | None | None | 0 | 0 | 0 |
| s9 | g0 | 0.1143 (0.67) | 0.0464 (0.37) | 7/0/53 | 0.14045 | None | None | 0 | 0 | 0 |
| s9 | g+12 | 0.0779 (0.57) | 0.0342 (0.28) | 6/0/54 | 0.07255 | None | None | 0 | 0 | 0 |

## 3. The pooled comparison (the frozen composite)

```json
{
 "pooled_primary": {
  "max_lower": 0.14045459032058716,
  "min_upper": 0.05861958488821983,
  "consistent": false,
  "R_hat": null,
  "n_two_sided_consistent": 3,
  "two_sided_keys": [
   "s10:g-12",
   "s10:g0",
   "s10:g+12"
  ],
  "n_lower_constraints": 15,
  "n_upper_constraints": 18
 },
 "pooled_g0_only": {
  "max_lower": 0.14045459032058716,
  "min_upper": 0.0593593455851078,
  "consistent": false,
  "R_hat": null,
  "n_two_sided_consistent": 1,
  "two_sided_keys": [
   "s10:g0"
  ],
  "n_lower_constraints": 5,
  "n_upper_constraints": 4
 },
 "pooled_sensitivity_form": {
  "max_lower": 5.039895040681586e-05,
  "min_upper": 1.9741739379242063e-05,
  "consistent": false,
  "R_hat": null,
  "n_two_sided_consistent": 0,
  "two_sided_keys": [],
  "n_lower_constraints": 6,
  "n_upper_constraints": 14
 },
 "R_T": 0.04828093355900924,
 "R_Z": null,
 "ratio": null,
 "resolvable": false,
 "two_sided_R_hats": {
  "s10:g-12": 0.004638001268369217,
  "s10:g0": 0.0053208669299817605,
  "s10:g+12": 0.004648001407155626
 },
 "x44_reference": {
  "max_lower": 0.04419664666056633,
  "min_upper": 0.05274265632033348
 }
}
```

## 4. The riders (descriptive, never bars)

### The margin map (pooled over rungs, per geometry)
```json
{
 "g-12": {
  "R_hat_used": null
 },
 "g0": {
  "R_hat_used": null
 },
 "g+12": {
  "R_hat_used": null
 }
}
```

### Noise-zone lifts + SAT-UP per cell

* ball: g-12 lifts-under-noise 0, sat-up 0; g0 lifts-under-noise 0, sat-up 9; g+12 lifts-under-noise 0, sat-up 0
* base: g-12 lifts-under-noise 60, sat-up 0; g0 lifts-under-noise 59, sat-up 0; g+12 lifts-under-noise 59, sat-up 0
* inst_sib: g-12 lifts-under-noise 0, sat-up 0; g0 lifts-under-noise 0, sat-up 0; g+12 lifts-under-noise 0, sat-up 0
* locked: g-12 lifts-under-noise 0, sat-up 0; g0 lifts-under-noise 0, sat-up 4; g+12 lifts-under-noise 0, sat-up 0
* own_inst: g-12 lifts-under-noise 0, sat-up 13; g0 lifts-under-noise 0, sat-up 20; g+12 lifts-under-noise 0, sat-up 6
* root: g-12 lifts-under-noise 0, sat-up 0; g0 lifts-under-noise 0, sat-up 5; g+12 lifts-under-noise 0, sat-up 0
* s1: g-12 lifts-under-noise 60, sat-up 0; g0 lifts-under-noise 59, sat-up 0; g+12 lifts-under-noise 59, sat-up 0
* s10: g-12 lifts-under-noise 1, sat-up 0; g0 lifts-under-noise 1, sat-up 0; g+12 lifts-under-noise 1, sat-up 0
* s11: g-12 lifts-under-noise 0, sat-up 0; g0 lifts-under-noise 0, sat-up 0; g+12 lifts-under-noise 0, sat-up 0
* s12: g-12 lifts-under-noise 0, sat-up 0; g0 lifts-under-noise 0, sat-up 0; g+12 lifts-under-noise 0, sat-up 0
* s8: g-12 lifts-under-noise 9, sat-up 0; g0 lifts-under-noise 8, sat-up 0; g+12 lifts-under-noise 8, sat-up 0
* s9: g-12 lifts-under-noise 5, sat-up 0; g0 lifts-under-noise 5, sat-up 0; g+12 lifts-under-noise 4, sat-up 0

## 5. The references (runtime-read from the md5-bound records)

* x38's committed bracket: (1.338e-05, 0.5302] (SHARP-FLIP)
* x42's committed interior window: (0.02172, 0.5052] (PARTIAL-GRADING)
* x44's committed pooled bounds: {"max_lower": 0.04419664666056633, "min_upper": 0.05274265632033348} (R*-UNIFORM); its two R_hats {"VARIED": 0.045533611079884004, "FIXED": 0.04847246848948268}

## Provenance
* birth commit: 565bf644ad79269f165704d5fdd967df85e72c81; final head: None
* RE-KEY: x46 stays reserved for R80's redundancy-profile card; this replicate runs as x47 (disclosed at birth).
* 22 md5 binds; the carrier certified end-to-end (model == base + delta bit-equal + x24's 22 host-g0 panel cells bit-exact through this cell's own reader + x44's committed ZEPHYRA-side rows at 1e-12); every rung prior + every interior per-step read reproduces x42's committed records at 2e-6; the reconstruction label COMMITTED-RUNGS; CPU-only (threads 4); timestamps UTC only; no NOTES/THINKING/QUEUE/STATE edits (the heartbeat folds)
