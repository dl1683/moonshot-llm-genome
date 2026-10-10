# W055 — THE RELAY TUNER'S FIRST PROBE (R79's card; W049's graduation)

**Date:** 2026-10-10T17:59:33Z (datetime.now(UTC) stamps in metrics) · **Executor report**

## THE VERDICT (against the frozen bars)

**STABLE-PERIOD-1**

```
the 8 events all dose weakly (classes MIXED/MIXED/MIXED/MIXED/MIXED/MIXED/MIXED/MIXED — no ZERO events) with every dose interior (0.267-0.388 of the halved cap), the read alive (min declared read 0.1750 >= 0.05) and the floor stable (range 0.0347 <= 0.08; drift +0.0052) — THE INTERMEDIATE REGIME IS A USABLE SETPOINT; W055's tuner has its third knob-setting (2-cycle / period-1 / off); the calibration program's kinetics complete
```

## The 8-event trace (phase-declared — every event, both phases)

| global event | step | PRE gate read | deficit | dose lr_m (x cap) | class | POST read |
|---|---|---|---|---|---|---|
| 37 | t925 | 0.181968 | 0.3634 | 0.003987 (0.363) | MIXED | 0.579423 |
| 38 | t950 | 0.196698 | 0.3119 | 0.003422 (0.312) | MIXED | 0.544597 |
| 39 | t975 | 0.187498 | 0.3441 | 0.003775 (0.344) | MIXED | 0.576482 |
| 40 | t1000 | 0.178771 | 0.3746 | 0.004110 (0.375) | MIXED | 0.59384 |
| 41 | t1025 | 0.193634 | 0.3226 | 0.003539 (0.323) | MIXED | 0.554293 |
| 42 | t1050 | 0.175019 | 0.3877 | 0.004254 (0.388) | MIXED | 0.604586 |
| 43 | t1075 | 0.209670 | 0.2665 | 0.002924 (0.267) | MIXED | 0.514947 |
| 44 | t1100 | 0.187562 | 0.3438 | 0.003772 (0.344) | MIXED | 0.576918 |

- the rails: floor mean 0.188852 (range 0.0347, 2nd-half minus 1st-half +0.0052); upper mean 0.568136
- vs x41's LRCAP leg: classes ZERO/DOSE/MIXED/MIXED, floor 0.3378, upper 0.6403, weak 0.1429
- vs e339's committed 2-cycle: floor 0.3735 / upper 0.8800 / weak 0.1245
- aliveness: min declared read 0.175019 (>= 0.05: True) at 'event 42 PRE (gate, pre-dose)' — 23 declared reads

## The off-target battery at the end (the standing amendment's collateral check)

- gm12 (offset battery, p(T)): 0.521382 (x41 t900: 0.568724)
- g0 Z-channel (wrong token): 2.87e-05 (x41 t900: 1.87e-05)
- CE_R (the eval bank): 1.6637 (x41 t900: 1.6583)

## The base and the gates

- the base: x41's committed LRCAP leg's final state (resume md5 aa2a5737... — the model + both SGD-M buffers + both generator states at t900, the state where MIXED appeared), bit-compared vs the committed post checkpoint, read-reproduced |d| 0.0e+00
- the regime held constant: lr_m_max 0.010971232006621194 (byte-equal to x41's LRCAP cap), M=25 unchanged; THE SCHEDULE EXTENSION INST_TOTAL 1000 -> 1250 (disclosed; G_SCHED verifies the cap stayed the only binding corpus constraint, min margin 2.62x)
- the net0 class: the lineage's char-transformer base (e001.pt) (2,739,072 params)
- 19 gate classes instantiated, all PASS: G_NAMEWIN, G_NAMEFREE, G_BASE, G_VMBIND, G_SPANBIND, G_ROOMK10K, G_LR_BIND, G_SUBSTRATE, G_X41BIND, G_STAGING, G_T900_REPRO, G_CONT, G_REGIME, G_PHASEDECL, G_ORTH, G_BUFSEP, G_BUDGET, G_STREAMELIVE, G_SCHED

## Envelope

- 208 thermal polls; max temp 61.0C; burst cap 175s; cooldowns 40s between bursts (inside the driver); zero concurrent GPU jobs; every burst logged to runs/_envelope_log.jsonl tagged w055:CONT

## Registered predictions

- P-W055a (the executor's own read: STABLE-PERIOD-1, moderately): HIT
- the dispatch's lab lean (STABLE-PERIOD-1, weakly): HIT

## Catches / disclosures

- THE SCHEDULE EXTENSION: the house cosine clamps to 0 at t >= 1000; a verbatim continuation would have silently KILLED the corpus wash for the last four events (RE-BIFURCATION-by-schedule, not by the relay). E261.INST_TOTAL re-bound 1000 -> 1250, asserted + restored; the budget cap stayed the ONLY binding corpus constraint (G_SCHED) — the operative wash variable held constant.
- THE PHASE-DECLARATION (METHODS 4) is enacted in the rig: MILESTONES == ALL event steps (every event read post-dose) + the controller's own gate read (pre-dose) + window brackets; the milestone-costume failure mode (a cadence locked to one phase) is impossible by construction.
- NO TWIN, BY DESIGN: the regime under test is an ERROR-GATED-arm property (the sanctuary twin has no maintenance events).
- n=1, ONE organism, one lineage, ONE branch (the LRCAP branch's own deterministic continuation — not a draw); the fixed point is one slot's realization.
- The off-target battery is REPORTED, never adjudicating (the standing amendment's collateral check).
- Checkpoints live INSIDE runs/w055/. No NOTES/THINKING/QUEUE/STATE edits (the heartbeat folds this cell).

*This cell does not edit NOTES/THINKING/QUEUE/STATE — the heartbeat folds.*