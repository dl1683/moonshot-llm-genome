# X51 / N3 — THE OPEN-LOOP CEILING (the hub test's differently-phased warm series)

* PRIMARY VERDICT: **FEEDBACK-COHERES** — no primary r-profile lands [sign -, consistency >= 0.75, mean|d| >= 0.15] over the 3 in-window matched pairs (r_g-12|g0: sign + 1.0, mean|d| 0.082292; r_g0|g+12: sign + 1.0, mean|d| 0.048315) — the controller states stay coherent like the open-loop arms; the open-loop ceiling stands; formation-phase feedback does not write decorrelation
* HUB RIDER: **APART-ABOVE** — r(ctrl,anneal) 0.9903 (perm p 0.1735, n=3); sign agreement 3/3; anneal nearest 3/3; median |d_ctrl-anneal| 0.515
* P-N3a (executor's registered read): HIT (guess: FEEDBACK-COHERES (moderately) + rider APART (moderately) — diverging from the lab lean's ANNEAL-ANCHORED on the rider)
* PARITY: lab lean primary HIT / rider MISSED; counter primary MISSED
* gates: 15/15 PASS

## The matched-read table (x45's frozen rule; window 0.1; many-to-one disclosed)

| row | role | read | matched panel | panel read | |d read| | in-window | r(g-12|g0) new/anneal (d) | r(g0|g+12) new/anneal (d) | alive g0 new/anneal | write |W| | in-room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctrl_t400 | primary | 0.8147 | anneal_s50 | 0.8034 | 0.011 | YES | 0.993/0.905 (+0.087) | 0.986/0.934 (+0.052) | 1.000/1.000 | 9.01 | 0.937 |
| ctrl_t600 | primary | 0.9000 | anneal_s50 | 0.8034 | 0.097 | YES | 0.984/0.905 (+0.078) | 0.983/0.934 (+0.049) | 1.000/1.000 | 9.06 | 0.932 |
| ctrl_t800 | primary | 0.8732 | anneal_s50 | 0.8034 | 0.070 | YES | 0.987/0.905 (+0.081) | 0.978/0.934 (+0.044) | 1.000/1.000 | 9.11 | 0.927 |
| passive_t400 | rider | 0.0029 | ann_s0 | 0.2859 | 0.283 | NO | 0.975/0.972 (+0.003) | 0.934/0.979 (-0.045) | 0.000/0.883 | 9.01 | 0.936 |
| passive_t800 | rider | 0.0017 | ann_s0 | 0.2859 | 0.284 | NO | 0.962/0.972 (-0.010) | 0.937/0.979 (-0.042) | 0.000/0.883 | 9.12 | 0.924 |
| ol_varied_s300 | open-loop | 0.7520 | ann_s300 | 0.7524 | 0.000 | YES | 0.326/0.325 (+0.001) | 0.460/0.458 (+0.002) | 0.983/0.983 | 18.36 | 0.408 |
| ol_fixed_s300 | open-loop | 0.7558 | anneal_s700 | 0.7532 | 0.003 | YES | 0.607/0.612 (-0.006) | 0.638/0.611 (+0.027) | 1.000/0.950 | 18.38 | 0.407 |

* read-span disclosure: anneal [0.285851,0.803375]; the anneal series' read ceiling 0.8034 (the s50 peak panel) — the controller's high states pair against it; every in-window alternative co-reported in the sensitivity table, never adjudicating
* read-sensitivity (in-window alternatives, never adjudicating): [{"tag": "ctrl_t400", "alternatives": [{"ann_tag": "anneal_s525", "ann_read": 0.7784374952316284, "abs_d_read": 0.036294758319854736, "d_r1": 0.5334848760237063}, {"ann_tag": "anneal_s250", "ann_read": 0.7715824246406555, "abs_d_read": 0.04314982891082764, "d_r1": 0.7818630457877118}, {"ann_tag": "anneal_s700", "ann_read": 0.7531909942626953, "abs_d_read": 0.06154125928878784, "d_r1": 0.38035298166659426}, {"ann_tag": "ann_s300", "ann_read": 0.7524126768112183, "abs_d_read": 0.06231957674026489, "d_r1": 0.6677412017746999}, {"ann_tag": "anneal_s600", "ann_read": 0.7369271516799927, "abs_d_read

## The primary scoring (the family's frozen decorrelation rule)

| profile | n | sign | consistency | mean|d| | separates decorrelated |
|---|---|---|---|---|---|
| r_g-12|g0 | 3 (0 None) | + | 1.0 | 0.082292 | no |
| r_g0|g+12 | 3 (0 None) | + | 1.0 | 0.048315 | no |

## The hub rider (matched phase: shared reshape steps 400/600/800; the frozen N1 residuals)

| step | age | ctrl r1 | ctrl resid (base/cons/varied) | anneal resid | walk bracket | swap bracket | |d ann| | |d min other| | ann nearest | all within 0.15 | sign agrees |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 400 | 800 | 0.9927581404727485 | 0.5654/0.607/0.4108 | 0.0504 | -0.2809 | -0.113 | 0.515 | 0.6784 | True | False | True |
| 600 | 1000 | 0.9836515452043235 | 0.6542/0.6958/0.4995 | 0.1275 | -0.2809 | -0.113 | 0.5267 | 0.7672 | True | False | True |
| 800 | 1200 | 0.986635171026022 | 0.755/0.7967/0.6004 | 0.2732 | -0.2809 | -0.113 | 0.4818 | 0.8681 | True | False | True |

* the walk/swap reshape grids END at 300 — no shared steps; the bracket pairs their terminal residuals (walk_s700 -0.2809, swap_s300 -0.113) against the controller's nearest step (t400)
* zeph_s400 (resid -0.1447) shares reshape step 400 as a COLD age-alias (age 400 vs the warm 800) — x49's territory; named, excluded from the bracket
* the controller's FEEDBACK class is not in the frozen fit; baseline coding adjudicates, cons/varied sensitivities co-reported
* n=3 — the r clause is degenerate at this n; the sign/proximity/magnitude clauses carry the adjudication; H2's 'equally with ALL arms' is only partially askable (the non-anneal arms have no shared steps; the bracket substitutes)

## The rider scores of the non-primary rows (co-reports)

* passive_t400: r_g-12|g0: sign None None, mean|d| None; r_g0|g+12: sign None None, mean|d| None
* passive_t800: r_g-12|g0: sign None None, mean|d| None; r_g0|g+12: sign None None, mean|d| None
* ol_varied_s300: r_g-12|g0: sign + 1.0, mean|d| 0.001437; r_g0|g+12: sign + 1.0, mean|d| 0.00165
* ol_fixed_s300: r_g-12|g0: sign - 1.0, mean|d| 0.005568; r_g0|g+12: sign + 1.0, mean|d| 0.027474

## The registration (frozen at birth, committed BEFORE compute)

**LAB LEAN:** FEEDBACK-COHERES, weakly, + ANNEAL-ANCHORED (the controller doses on its own cadence — a different phase structure; and the ruler is one endpoint of every correlation in this pool).

**COUNTER:** FEEDBACK-DECORRELATES — the controller's doses ARE a varied-context menu in the strict sense (each dose responds to a different read state), and W056's surviving anchor says varied menus buy decorrelation early.

**P-N3a:** FEEDBACK-COHERES (moderately) + rider APART (moderately) — diverging from the lab lean's ANNEAL-ANCHORED on the rider

GROUNDS: (1) THE SURGICAL-DOSE ARGUMENT — the controller's name exposure is 16-32 SINGLE error-gated steps through opt_F while the anneal needed 75-125 CONTINUOUS varied-curriculum steps to decorrelate (x45's panels: s25 r1 0.996, s50 0.905, s75 0.530, s125 0.124); x40's FRESH precedent says burst-style in-room writes RAISE the read without decorrelating (the subject itself: read 0.286, r1 0.972). (2) THE VARIANCE-SUPPRESSION ARGUMENT — the controller is a setpoint law on the read LEVEL; x48's split says the level is the state-clock's object and decorrelation is the phase-dynamics object; feedback that quenches read-level error should quench the reshape wiggle with it (W055's period-1 stability). (3) THE RIDER ARITHMETIC (design-time desk recon on COMMITTED literals only): the anneal's residuals at the shared steps are +0.05/+0.13/+0.27; a coherent controller lands +0.42..+0.67 — outside every arm's +-0.15 neighborhood -> APART; the lean's ANNEAL-ANCHORED requires a mid-r1 controller, which the surgical reading says will not happen.

**P-T329a (T329, frozen before this cell ran):** REGISTERED PREDICTION (P-T329a, on N3+x49, frozen before they land): the controller states' reshape residuals correlate with the anneal's at matched phase MORE strongly than with any non-anneal arm (the star persists — H1/H3) versus equally with all arms (H2); and x49's swap-s125 residual is nonzero and negative if the tick is phase-locked (H2), absent if age-locked. The lab lean: H1/H3, weakly (the ruler is one endpoint of every measured correlation — the star may be the instrument's own shape); the counter, stated plainly: H2 — the co-movement's p-values are strongest exactly where the design put the shared phase, and the walk-side miss is the only warm-specific whisper.

## The endpoint gates (my reads vs the committed milestones)

* ctrl_t400: g0 mine 0.814732 vs committed 0.814732 (|d| 0.00e+00); gm12 |d| 0.00e+00
* ctrl_t600: g0 mine 0.899997 vs committed 0.899997 (|d| 0.00e+00); gm12 |d| 0.00e+00
* ctrl_t800: g0 mine 0.873240 vs committed 0.873240 (|d| 0.00e+00); gm12 |d| 0.00e+00
* passive_t400: g0 mine 0.002891 vs committed 0.002891 (|d| 0.00e+00)
* passive_t800: g0 mine 0.001652 vs committed 0.001652 (|d| 0.00e+00)
* ol_varied_s300: g0 mine 0.751972 vs committed 0.751972 (|d| 0.00e+00); gm12 |d| 0.00e+00
* ol_fixed_s300: g0 mine 0.755755 vs committed 0.755755 (|d| 0.00e+00); gm12 |d| 0.00e+00
* the t600 branch co-read (e342 vs e339's committed milestones): |d| 6.0e-08

## Provenance
* birth commit: c7b4272f853cef6e2b72f28168018d7067103793; final head: c7b4272f853cef6e2b72f28168018d7067103793
* CPU desk only (inference + desk; threads 4; zero CUDA calls); timestamps UTC only; every artifact md5-bound (metrics.gates.G_MD5); the instrument certified against x45's anchor rows + x44's FRESH/VARIED/FIXED rows (metrics.gates.G_INSTRUMENT); x48's pooled re-scores reproduced exactly (G_X45REPRO/G_E343REPRO/G_E344REPRO/G_X48POOLED/G_X48COMPA/G_N1REFIT/G_COMOVE)
* catches + disclosures: see metrics.deviations + metrics.catches

*No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat folds).*