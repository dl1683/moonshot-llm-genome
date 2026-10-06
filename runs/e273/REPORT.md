# E273 — THE THREE-BARREL MECHANISM CELL

**VERDICT: MIXED**

the SGD-M barrel DIVERGED at the registered calibration (max CE beyond s100 8533.8505859375 vs
first-batch 1.0702) — its formation clause is DIVERGENCE-CONFOUNDED and cannot adjudicate Adam-
specificity at this scale; the x0.01 stability rider (SGD001X) carries the stable-scale read; the
antiphase-convention straddle (n=4 vs n=5) flips a clause — the n=4 PRIMARY stands, disclosed — the
trajectories verbatim, all reads, no inflation

## The question (frozen)
> the concurrent kill at k=10k — is the coupling (i) the SHARED OPTIMIZER STATE (v-poisoning), (ii) ADAM'S NORMALIZATION itself (the low-rank formation block is Adam-specific), or (iii) the TRAJECTORY (a true two-body problem in the parameters)?

## The arms' reads (n=1 per arm, one session; the WRITE read adjudicates; the landing read carried, never adjudicated)

| arm | post g0 (WRITE) | x serial | peak traj g0 | formed (>0.005)? | antiphase n4 | antiphase n5 | peak@dip? | kept med | in-own-room | v-exc | root g0 (carried) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SERIAL | 0.209721 | 1.000x | 0.315270 | n/a (the driver) | n/a | n/a | n/a | 0.0601 | 0.9445 | 1.00 | 0.6910 |
| SHARED | 0.005288 | 0.0252x | 0.005288 | YES | -0.321 | +0.209 | no (pk@400, dip@300) | 0.0597 | 0.6777 | 0.23 | 0.7477 |
| SEPARATE | 0.001599 | 0.0076x | 0.018551 | YES | +0.968 | +0.754 | no (pk@100, dip@300) | 0.0574 | 0.9397 | 0.96 | 0.3923 |
| SGDM | nan | nanx | 0.000006 | no **DIVERGED** | n/a | n/a | no (pk@100, dip@300) | 0.0599 | 0.0000 | 0.00 | nan |
| SGD05X | 0.000000 | 0.0000x | 0.000008 | no **DIVERGED** | n/a | -0.885 | no (pk@100, dip@300) | 0.0605 | 0.1794 | 12.48 | 0.0000 |
| SGD001X | 0.002527 | 0.0120x | 0.002527 | no | -0.313 | +0.178 | no (pk@400, dip@300) | 0.0596 | 0.6378 | 3.29 | 0.7859 |
| SGD2X | nan | nanx | 0.000006 | no **DIVERGED** | n/a | n/a | no (pk@None, dip@300) | 0.0000 | 0.0000 | 0.00 | nan |

- the volume-null floor (the fresh fact-free base's own g0 read): 1.338e-05
- the antiphase provenance: e268's committed 10k pair reproduced live — Pearson n4 -0.524 / n5 0.298 (the registered datum was -0.52, the n4 convention)
- the room: e272's committed K10KR (k=10,000, seeds 27215/27216), rebuilt + bit-gated; the cited serial reference: e272's K10KR (post 0.209721 / root 0.696994 / kept 0.0601)

## The clauses (each barrel's clause reported separately — the composite requires them jointly)

- **SP1_separate_write_survives**: does not fire — post_g0_separate=0.0015990192769095302, post_g0_serial_session=0.209721177816391, ratio_session=0.00762450074693676, ratio_committed=0.007624510498231792, bar=0.5
- **SP2a_antiphase_vanishes_in_separate**: does not fire — pearson_n4=0.9681400862432751, pearson_n5=0.7536083530095062, bar=0.2
- **SP2b_antiphase_present_in_shared_replay**: FIRES — pearson_n4=-0.321438038605749, pearson_n5=0.2086323680817582, bar=-0.2
- **ASB_sgdm_forms**: does not fire — peak_traj_g0=5.991389116388746e-06, bar=0.005, informative=False, divergence_confounded=False, block_level_context=0.0007
- **TTB1_separate_write_dies**: FIRES — 
- **TTB2_sgdm_dies_without_forming**: FIRES — informative=False, divergence_confounded=True
- **sgdm_divergence_signature**: {'first_batch_ce': 1.0701713562011719, 'max_ce_beyond_s100': 8533.8505859375, 'max_ce': 8533.8505859375, 'nonfinite': True, 'diverged': True}
- **sgd001x_stability_rider**: {'first_batch_ce': 1.0701713562011719, 'max_ce_beyond_s100': 1.1811984777450562, 'max_ce': 1.2731643915176392, 'nonfinite': False, 'diverged': False}
- **convention_straddle_n4_vs_n5**: True

## The SGD-M lr calibration (the live probe)

- AdamW s1 applied in-room L2: 1.321312 (gpn 0.060782; full-L2 1.6522)
- **LR_SGD = 21.738575** (the disclosed factor: x21738.6 vs the AdamW 1e-3); momentum 0.9, wd 0.0 (disclosed)
- the SGD s1 verification: in-room 1.321312, rel err 7.52e-09 (bar 0.05)

## Texture disclosures (non-halting)

- G_SERIAL_ANCHOR: install L2 4.716552780561744e-05, post |d| 0.000000, root |d| 0.0060 -> PASS
- the draw-integrity first-batch CE spread across arms: 0.0

## The reads addendum (the coordinator's message, received PRE-BIRTH; reads only, no bar moved)

> under SHARED-v relaxation, three reads move TOGETHER in the separate-AdamW arm — (i) the antiphase vanishes (|Pearson| < 0.2 — already in the spec as P-273a), (ii) the flash's PEAK AMPLITUDE falls toward the volume-null level, (iii) the ENDPOINT rises toward serial. v-ownership moves all three; trajectory-ownership moves none.

Per arm: peak traj_g0 (the table), the endpoint ratio vs serial (the table's x serial), and the peak-vs-driver-dip alignment (the table's peak@dip column). Under v-ownership all three move together; under trajectory-ownership none move.

## Provenance

- parents hard-bound: e272 metrics (eb9f624708bcb6576c4115161dfd7042), e268 metrics (c1149229b7f0191943a7b8eb0442b494), e271 metrics (16c29a3167d8523a64f66f03c717f67b), e272_rooms.pt, e246 span, e258 v-map, the K10KR vehicle; the room bit-gated (G_ROOM10KR); 13 hard gates (a failure halts)
- machinery: e261's ported whole by import (the serial driver, the cons, the hook, the envelope); this file's one new driver (chunked_install_barrel) + the live lr calibration probe
- envelope: bursts <= 175s, per-step polls both streams, 40s cooldowns, the 84C never-past line (inside the dispatch's 85C); max temp 80.0C over 7382 persisted polls (0 violations)
- bars + question frozen VERBATIM at birth (commit before compute); no bar shopping; n=1 per arm; nothing guaranteed