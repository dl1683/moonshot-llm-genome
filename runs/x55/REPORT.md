# X55 — THE MID-BAND LADDER, FATE-RESOLVED

* SHAPE VERDICT: **EDGE-EARLY** — score_min ALIVE at k=12 (k12: 0.3371) — the protection edge lands in (7,12]: the lab lean's reading (k=7's 0.6% near-miss + the read-gap climb)
* P-x55a (my registered read): {'verdict': False, 'guess': "RECOVERY-ONLY (WITH the dispatch's counter, AGAINST the lab lean's EDGE-EARLY)"} (guess: RECOVERY-ONLY (WITH the dispatch's counter, AGAINST the lab lean's EDGE-EARLY))

* the critic's co-registered k7 prediction: HIT — PATCH (the min fluctuates: 2/8 in-band rungs sub-DEAD with clears between); clears at [105, 110, 115, 130]
* gates: 15/15 PASS

## THE K-BAND TABLE (fate cuts: DEAD <= 0.10, ALIVE >= 0.285851; score_min = e346's frozen convention — THE adjudicating score; score_med = the R83 MEDIAN CO-REPORT, standing, never re-adjudicating e346's committed k<=7 verdict)

| k | provenance | flip slots | p(T) s25 | p(T) s125 | p(T) s300 | score_min | fate(min) | score_med | fate(med) | p(Z) guest s300 | s1 host |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | e345's committed arm (k=0) | None | 3.60e-03 | 4.73e-04 | 3.06e-03 | 4.73e-04 | DEAD | 3.06e-03 | DEAD | 0.6352 | 6.76e-03 |
| 1 | mine (k1; flip slots (0,)) | [0] | 0.0107 | 0.0281 | 0.0424 | 0.010666 | DEAD | 0.028132 | DEAD | 0.6573 | 6.62e-03 |
| 2 | mine (k2; flip slots (0, 8)) | [0, 8] | 0.1438 | 0.0704 | 0.0699 | 0.069899 | DEAD | 0.070442 | DEAD | 0.6388 | 6.30e-03 |
| 4 | mine (k4; flip slots (0, 4, 8, 12)) | [0, 4, 8, 12] | 0.1154 | 0.0911 | 0.1610 | 0.091134 | DEAD | 0.115390 | MID | 0.6042 | 9.52e-03 |
| 7 | mine (k7; flip slots (0, 2, 5, 8, 10, 12, 15)) | [0, 2, 5, 8, 10, 12, 15] | 0.4348 | 0.0994 | 0.2994 | 0.099412 | DEAD | 0.299372 | ALIVE | 0.4072 | 0.014052 |
| 9 | mine (k9; flip slots (0, 2, 4, 5, 7, 9, 11, 12, 14)) | [0, 2, 4, 5, 7, 9, 11, 12, 14] | 0.3545 | 0.2410 | 0.3889 | 0.241018 | MID | 0.354484 | ALIVE | 0.3244 | 0.370537 |
| 12 | mine (k12; flip slots (0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15)) | [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15] | 0.4457 | 0.3371 | 0.5300 | 0.337099 | ALIVE | 0.445712 | ALIVE | 0.1450 | 0.306193 |
| 14 | mine (k14; flip slots (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15)) | [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15] | 0.6497 | 0.6084 | 0.6538 | 0.608400 | ALIVE | 0.649686 | ALIVE | 0.1012 | 0.737110 |
| 16 | the anneal ruler (k=16, x45 panels + x52 dense) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] | 0.6843 | 0.3090 | 0.7159 | 0.309011 | ALIVE | 0.684341 | ALIVE | - | 0.748952 |

* the composite order (frozen): TEXTURE (hard gates) -> EDGE-EARLY (min-ALIVE at k in {9,12}) -> EDGE-MID-(12,16) (k14 alone ALIVE; the pre-named fallback) -> RECOVERY-ONLY (no mid-band min-ALIVE + mid-band fate reads ALIVE) -> EDGE-LATE-OR-NEVER (no mid-band min-ALIVE + no mid-band fate read ALIVE) -> MIXED

## THE k7 IN-BAND CONTROL (the critic's named control)

* in-band reads s100-s135 (grain 5): s100 0.245210, s105 0.237467, s110 0.179316, s115 0.271183, s120 0.096068, s125 0.099400, s130 0.293403, s135 0.248836
* the CPU re-realization's fate trio: s25 0.434759, s125 0.099400, s300 0.294554 vs the committed GPU arm s25 0.434825, s125 0.099412, s300 0.299372 (G_K7CLASS bounds the class)
* patch class: **PATCH (the min fluctuates: 2/8 in-band rungs sub-DEAD with clears between)** — the in-band-extended min 0.096068 stays sub-DEAD — the committed k7 score stands un-flipped even in the co-report lane
* the critic's prediction: HIT (the R83 critic's k7-control prediction, adopted verbatim and co-scored: 'the min fluctuates (patch class), at least one of s105/s110/s115/s120/s130 clears 0.10')

## THE s1 MICRO-LADDER EXTENSION (the armor observable; co-report)

| k | 0 | 1 | 2 | 4 | 7 | 9 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| s1 host p(T) | 6.76e-03 | 6.62e-03 | 6.30e-03 | 9.52e-03 | 0.014052 | 0.370537 | 0.306193 | 0.737110 | 0.748952 |
* k14's s1 > 0.10 would locate an ARMOR edge inside (12,16) — the registered surprise branch, DISCLOSED ONLY (committed datum: 0.006756 (k=0) .. 0.014052 (k=7) vs 0.748952 (k=16, x52's rider_s1))

## Provenance
* birth commit: 5aed531; final head: 3ab3d0ed36ed08065b88e2aa2f7d8957eb349017
* CPU desk ONLY (threads 4, zero CUDA — the GPU lane is sibling x36's, never touched; runtime asserted); timestamps UTC only; every artifact md5-bound (metrics.gates.G_MD5); the four arms' draws bit-equal to e346's committed k7 resume record (G_DRAWS)
* catches + disclosures: see metrics.deviations

*No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat folds).*