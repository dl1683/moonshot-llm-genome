# X64 — THE SEED BATCH + RIDERS (the coin's base-rate cell)

Generated 2026-10-11T22:55:24.382499+00:00 | smoke=False | CPU desk threads 4 zero CUDA

## THE VERDICTS (against the frozen bars)

**THE BASE RATE: MAJORITY-CARRY** — R 6.5 of 8 labeled (carry 6 / lack 1 / knife 1); CP95 on the integer carry count: [0.349, 0.968] (sensitivity 0.473, 0.997). R = 6.5 (n_carry 6 + 0.5*n_knife 1) of 8 labeled — the key coin carries in the majority; the 2-for-3 committed read was not a consecutive-seed fluke

**RIDER A (ANY-SEPARATES / ALL-SILENT): ALL-SILENT** — NO new-class feature separates carriers from non-carriers at any frozen threshold (max min-cross-gap ratio 0.16 of threshold, on J1) — the untried classes join x63's six: the key remains invisible in the draw statistics (T346's response-not-marginals reading strengthens)

**RIDER B (24315-CARRIES-ON-B / 24315-LACKS-ON-B): KNIFE-ON-B (neither bar fires — the named leftover, disclosed)** — K 0.178016; the grid: SUBJECT-INTERACTION (a B-side label flips its A-side counterpart)

**RIDER C (KEY-SHOWS-EARLY / KEY-HIDDEN-UNTIL-REFORMATION): KEY-SHOWS-EARLY** — 3 of 20 cells fire at the 25% margin: ['s200:read', 's300:read', 's400:read'] (rungs [200, 300, 400]) — an EARLY-rung fire (s1-s200): the clean early signal, not the landing-height axis

Composite: BASE:MAJORITY-CARRY + A:ALL-SILENT + B:KNIFE-ON-B (neither bar fires — the named leftover, disclosed) + C:KEY-SHOWS-EARLY | P-x64a: UNSCORED (smoke/texture/unadjudicable — all guesses stand registered).

## THE BATCH TABLE (the 8 records; the K axis frozen at birth)

| seed | origin | landing read | K | rule verdict | label | dA/dB |
|---|---|---|---|---|---|---|
| 24314 | x60 refa (runtime-read) | 0.4024 | 0.196489 | DECORRELATED | CARRY | - |
| 24315 | x59 refa2 (runtime-read) | 0.6879 | 0.144955 | COHERENT | LACK | - |
| 24316 | x63 refa3 (runtime-read) | 0.4393 | 0.259131 | DECORRELATED | CARRY | - |
| 429496730 | x64 fresh (s1) | 0.6153 | 0.269025 | DECORRELATED | CARRY | 0.5036/0.9835 |
| 1288490189 | x64 fresh (s2) | 0.4570 | 0.245983 | DECORRELATED | CARRY | 0.4805/1.1452 |
| 2147483648 | x64 fresh (s3) | 0.3950 | 0.214787 | DECORRELATED | CARRY | 0.279/0.912 |
| 3006477407 | x64 fresh (s4) | 0.5521 | 0.320080 | DECORRELATED | CARRY | 1.5247/2.2247 |
| 3865470566 | x64 fresh (s5) | 0.4354 | 0.146190 | COHERENT | KNIFE | 1.0323/0.3707 |

- exclusions: none
- the SEED is the coin's unit (T346: name-invariant); the refb twin (K 0.204653) is a co-report, not a 9th record; rider B (24315-on-B) likewise excluded from the rate
- the fork: MAJORITY-CARRY (R >= 5) / MINORITY-CARRY (R <= 3) / COIN-CLASS (3 < R < 5); R = n_carry + 0.5 * n_knife; the CP interval on the integer carry count.

## THE SEED-BY-SUBJECT GRID (rider B's co-report)

| cell | K | label | source |
|---|---|---|---|
| 24314xA | 0.196489 | CARRY | x60 refa (committed) |
| 24314xB | 0.204653 | CARRY | x59 refb (committed) |
| 24315xA | 0.144955 | LACK | x59 refa2 (committed) |
| 24315xB | 0.178016 | KNIFE | x64 refb2 (this cell) |
- reading: SUBJECT-INTERACTION (a B-side label flips its A-side counterpart)

## RIDER A — THE ANATOMY RE-SCREEN TABLE (the seven named features, thresholds frozen at birth)

| feature | min carrier-lack gap | threshold | ratio | separates |
|---|---|---|---|---|
| J1: Pearson corr of within-step (ix, aj) pairs (6400); separates | 0.0079854 | 0.05 | 0.16 | no |
| J2: Pearson corr of within-step (ix x2, rj/M) pairs (12800); sep | 0.0010713 | 0.035 | 0.03 | no |
| J3: unordered overlapping window pairs (12800 spans); separates  | 2 | 745 | 0.00 | no |
| O1: median of the 6400 ix draws; separates iff |d| >= 1.0 | 0 | 1.0 | 0.00 | no |
| O2: range (max-min) of the 100 per-step ix means; separates iff  | 0.875 | 10.0 | 0.09 | no |
| B1: max adjacent-block two-sample chi2 of the 60-bin ix histogra | 2.6403 | 48.0 | 0.06 | no |
| B2: max |block mean rj - M/2| over 4 blocks; separates iff |d| > | 727.96 | 15000.0 | 0.05 | no |

- groups: carriers ['24314', '24316', '429496730', '1288490189', '2147483648', '3006477407'] / lacks ['24315'] / knives (excluded, co-report) ['3865470566']
- family-wise caveat: seven features at ~99% each: ~7% chance of >=1 spurious firing under the pure-coin null
- per-seed feature values: metrics.rider_a.features_per_seed

## RIDER C — THE COARSE-TRAJECTORY MARGIN TABLE (the frozen 25% rule)

| rung:stat | lack value | nearest carrier gap | margin | fires |
|---|---|---|---|---|
| s1:r_g-12|g0 | 0.88936 | 0.0039039 | 0.004 | no |
| s1:r_g0|g+12 | 0.94417 | 0.0007933 | 0.001 | no |
| s1:read | 0.0007387 | 3.5533e-05 | 0.051 | no |
| s1:write_norm | 21.56 | 1.0234e-05 | 0.000 | no |
| s100:r_g-12|g0 | 0.65015 | 0.0085871 | 0.009 | no |
| s100:r_g0|g+12 | 0.61604 | 0.0092698 | 0.009 | no |
| s100:read | 0.71858 | 0.084404 | 0.146 | no |
| s100:write_norm | 21.555 | 0.010255 | 0.000 | no |
| s200:r_g-12|g0 | 0.33381 | 0.20395 | 0.204 | no |
| s200:r_g0|g+12 | 0.47549 | 0.083871 | 0.084 | no |
| s200:read | 0.20317 | 0.46173 | 0.675 | YES |
| s200:write_norm | 22.861 | 0.019046 | 0.001 | no |
| s300:r_g-12|g0 | 0.41643 | 0.041484 | 0.041 | no |
| s300:r_g0|g+12 | 0.48266 | 0.01598 | 0.016 | no |
| s300:read | 0.33933 | 0.15609 | 0.308 | YES |
| s300:write_norm | 24.051 | 0.0062564 | 0.000 | no |
| s400:r_g-12|g0 | 0.52258 | 0.067433 | 0.067 | no |
| s400:r_g0|g+12 | 0.4608 | 0.045272 | 0.045 | no |
| s400:read | 0.68791 | 0.24863 | 0.618 | YES |
| s400:write_norm | 24.963 | 0.0099648 | 0.000 | no |

- in_room EXCLUDED by construction: 0.060 at every committed rung (the write lives 94% outside the K10K room from step one); write_norm s1 sanity: all four ~21.56 (the first step is a no-event on the shared start)

## THE GATES

31/31 gates PASS (the two G_FLOOR class gates are soft by design; the rest hard): G_MD5(PASS), G_QUOTES(PASS), G_NAMEFREE(PASS), G_SPLICE(PASS), G_BATTERY(PASS), G_BASELITERALS(PASS), G_ROOM(PASS), G_ENDPOINTS(PASS), G_SEEDTWIN14(PASS), G_RIDERCRECORDS(PASS), G_INSTCERT(PASS), G_WALKREPRO(PASS), G_FLOOR_s1(PASS), G_REDRAWDIFF_s1(PASS), G_REFDRAWS_s1(PASS), G_FLOOR_s2(PASS), G_REDRAWDIFF_s2(PASS), G_REFDRAWS_s2(PASS), G_FLOOR_s3(PASS), G_REDRAWDIFF_s3(PASS), G_REFDRAWS_s3(PASS), G_FLOOR_s4(PASS), G_REDRAWDIFF_s4(PASS), G_REFDRAWS_s4(PASS), G_FLOOR_s5(PASS), G_REDRAWDIFF_s5(PASS), G_REFDRAWS_s5(PASS), G_SEEDTWIN15(PASS), G_FLOOR_B2(PASS), G_REFDRAWS_B2(PASS), G_ANATOMYRECORDS(PASS)

## THE PARITY BLOCK (verbatim from the dispatch)

- LAB LEAN: MINORITY-CARRY + ALL-SILENT + 24315-LACKS-ON-B + KEY-HIDDEN (the coin's midpoint-ish lean with the emergent reading)
- COUNTER: COIN-CLASS + ANY-SEPARATES + 24315-CARRIES-ON-B + KEY-SHOWS-EARLY (the locality artifact + the sloppy-models discriminant)
- BASE-RATE BAR VERBATIM: report the n=8 count with the CP interval; the bar is the PRE-REGISTERED fork: MAJORITY-CARRY (>= 5 of 8) / MINORITY-CARRY (<= 3) / COIN-CLASS (4 of 8 ± 1)
- RIDER A BAR VERBATIM: ANY-SEPARATES (>= 1 new-class feature separates carriers from non-carriers at the frozen threshold) / ALL-SILENT
- RIDER B BAR VERBATIM: 24315-CARRIES-ON-B / 24315-LACKS-ON-B (seed-main vs interaction)
- RIDER C BAR VERBATIM: KEY-SHOWS-EARLY (a coarse-rung statistic separates the carriers' formations) / KEY-HIDDEN-UNTIL-REFORMATION

## P-x64a (the executor's own reads, registered at birth)

- READ1: COIN-CLASS, weakly (DIVERGES from the lab lean's MINORITY-CARRY — named at birth) -> UNSCORED
- READ2: ALL-SILENT, weakly (WITH the lab lean) -> HIT
- READ3: 24315-LACKS-ON-B, weakly (WITH the lab lean) -> UNSCORED
- READ4: KEY-HIDDEN, weakly (WITH the lab lean; the recon peek of the committed rung logs disclosed — the 24315 s400 landing-read gap named as the visible counter-signal) -> UNSCORED
- COMPOUND: COIN-CLASS + ALL-SILENT + 24315-LACKS-ON-B + KEY-HIDDEN -> UNSCORED
- scored: False

## THE SEED CHOICES (disclosed at birth)

- fresh five: [429496730, 1288490189, 2147483648, 3006477407, 3865470566]
- derivation: midpoints of five equal bins of [0, 2^32): (2i+1)*2^32/10 rounded, i=0..4; adjacent pairs 858,993,458-459 apart (> 2^29); e327's convention, distinct seeds (e327's own eight consumed by its installs)

## CATCHES / DEVIATIONS

- THE THREE KILLS + RELAUNCHES (the three-proof standard, R83/R84's form; appended check-side AFTER the run — no bar-deciding logic touched, the docstring untouched birth->HEAD): the environment's ~1h background window killed the desk three times (kill 1 ~59.8 min mid refa_s2; kill 2 ~59.4 min mid refa_s4; kill 3 ~59 min mid refb2 — x59/x60/x63's own class). RESTART 1 (refa_s2): inst_s2's resume conserved 400/400 draws + the 5 rung rows (PROOF 1, verified from the resume artifact); refa_s2 re-executed from step 0 with its pre-kill rows BIT-IDENTICAL (verified from the relaunch log: s25 0.519673 / s50 0.399590 / s75 0.641776 / s100 0.656942 — PROOF 2); every gate re-derived on relaunch (PROOF 3). RESTART 2 (refa_s4): same form — inst_s4's resume conserved; refa_s4's pre-kill s150 row 0.393584 reproduced bit-identically. RESTART 3 (refb2): the automated reexec_proof recorder (added at birth) recorded 11 rows bit-identical in metrics.restart_proofs.refb2. VERDICTS UNAFFECTED (deterministic re-execution); the in-memory proof records for restarts 1-2 died with their kills and are quoted here from the logs verified at relaunch time.
- BIRTH-TIME DISCLOSURES (frozen here, before any compute): (1) the slice cap LOWERED to 900s from x63's 1800s (the environment's ~1h background-kill window; checkpoint frequency only — no bar-deciding logic touched, the docstring untouched birth->HEAD); (2) G_INSTALL-FLOOR converted from x63's hard gate to a per-seed CLASS gate (the batch's purpose is the coin's rate; a die-mode draw — e324's bimodal wheel, a DIFFERENT lottery axis, e327: 0/13 unconditioned — EXCLUDES the seed from the base-rate denominator and SKIPS its re-formation leg rather than killing the cell; the exclusion is disclosed prominently in the verdict); (3) drops vs x63: G_A2CERT (the 24315 record already certified by md5-bound x63 — zero recompute) and the N/M base literals (no b/h guests; the N channel CO-REPORTED measured-only — no committed N base literal at hand); (4) P-x64a read1 DIVERGES from the lab lean (COIN-CLASS vs MINORITY-CARRY) — named at birth; (5) the RIDER-C INFORMATION-STATE DISCLOSURE: the executor inspected the four committed coarse-rung logs during design recon (committed public artifacts; the 24315 s400 landing-read gap 0.6879 vs 0.3681-0.4393 visible and named in P-x64a read4) — the rider-C rule is nonetheless frozen as designed (all five rungs, all four statistics, 25% margin); (6) the smoke trims the batch to ONE fresh seed (the seed only changes the generator; both pool classes and both leg classes exercised); (7) the rider-A thresholds are ANALYTICALLY calibrated from construction-known sample sizes (Gaussian/Poisson approximations disclosed per feature), not from any seed's computed values — the feature list + thresholds frozen in this birth commit before the first anatomy computation.