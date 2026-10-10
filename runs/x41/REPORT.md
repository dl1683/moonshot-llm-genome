# X41 — THE RELAY'S KINETICS (R78's card; W049's dial)

**Date:** 2026-10-10T15:30:02Z (datetime.now(UTC) stamps in metrics) · **Executor report**

## THE VERDICT (against the frozen bars)

**ENTRAINABLE**

```
at least one perturbation moved the cycle's observable structure: M1 PERIOD-RELOCK (leg-A: every event doses at the doubled spacing — the cycle re-locked to the event cadence; classes ['DOSE', 'DOSE', 'MIXED']); M2 AMPLITUDE-MOVE (LRCAP upper_rail 0.6403 vs control 0.8632 (|d| 0.2229 > 0.08)); M3 CLASS-FLIP (PERIOD2X: ['DOSE', 'DOSE', 'MIXED'] vs control ['ZERO', 'DOSE', 'ZERO']; LRCAP: ['ZERO', 'DOSE', 'MIXED', 'MIXED'] vs control ['ZERO', 'DOSE', 'ZERO', 'DOSE']) — THE CYCLE IS TUNABLE; W049's dial designs a relay tuner; the calibration program gains its kinetics
```

## The three legs' cycle structures vs the committed cycle

| leg | M | lr_m_max | events (global) | classes | floor rail | upper rail |
|---|---|---|---|---|---|---|
| committed (e339 25-32) | 25 | 0.021942 | 25-32 | ZERO/DOSE alternating | 0.3735 | 0.8800 |
| CONTROL | 25 | 0.021942 | [33, 34, 35, 36] | ZERO/DOSE/ZERO/DOSE | 0.3668 | 0.8632 |
| PERIOD2X | 50 | 0.021942 | [33, 34, 35] | DOSE/DOSE/MIXED | None | 0.8624 |
| LRCAP | 25 | 0.010971 | [33, 34, 35, 36] | ZERO/DOSE/MIXED/MIXED | 0.3378 | 0.6403 |

- M1 PERIOD-RELOCK: True; M2 AMPLITUDE-MOVE: True (['LRCAP upper_rail 0.6403 vs control 0.8632 (|d| 0.2229 > 0.08)']); M3 CLASS-FLIP: True (["PERIOD2X: ['DOSE', 'DOSE', 'MIXED'] vs control ['ZERO', 'DOSE', 'ZERO']", "LRCAP: ['ZERO', 'DOSE', 'MIXED', 'MIXED'] vs control ['ZERO', 'DOSE', 'ZERO', 'DOSE']"])
- leg-A's alternation survives: False
- the move-observable margin: 0.08 (raw units); the committed cycle's rails: floor 0.3735 / upper 0.8800 / weak 0.1245

## The verification rider (CPU; e339's committed trace)

- **REFINED-UPPER-RAIL** — R1 (the strict average): |0.8732 - 0.6267| = 0.2465 (tol 0.05) — FAILS
- R2 (the upper rail): |0.8732 - 0.8800| = 0.0068 — HOLDS
- R3 (descriptive): the reconstructed cycle time-average 0.4149-0.4197 vs the plateau 0.8732 — the plateau is 2.09x the time-average
- T310's sentence, tested: the milestone cadence (every 4th event — the dose phase's parity) is phase-locked to the cycle's UPPER RAIL; the 'equilibrium' is the upper rail, not the cycle's average

## The base and the gates

- the base: e339's committed t800 resume state (md5 134034c5...), bit-compared vs the committed post checkpoint, read-reproduced |d| 0.0e+00; three digest-verified branch copies (shared stream draws by construction)
- the net0 class: the lineage's char-transformer base (e001.pt) (2,739,072 params)
- 21 gate classes instantiated, all PASS: G_NAMEWIN, G_NAMEFREE, G_BASE, G_VMBIND, G_SPANBIND, G_ROOMK10K, G_LR_BIND, G_SUBSTRATE, G_E339BIND, G_STAGING, G_T800_REPRO, G_RIDER, G_LEG_CONTROL, G_LEG_PERIOD2X, G_LEG_LRCAP, G_ORTH, G_BUFSEP, G_BUDGET, G_STREAMELIVE, G_SCHED, G_CYCLE_REPRO

## Envelope

- 361 thermal polls; max temp 71.0C; burst cap 175s; cooldown 40s between legs; zero concurrent GPU jobs; every burst logged to runs/_envelope_log.jsonl tagged x41:...

## Registered predictions

- P-x41a (the executor's own read: ENTRAINABLE via both legs): HIT
- P-x41a-rider (REFINED-UPPER-RAIL): HIT
- the dispatch's lab lean (ENTRAINABLE, weakly): HIT

## Catches / disclosures

- THE LEAN'S WORDING: the dispatch's lean says 'lr_m -> 0 on the weak phase'; the committed trace puts the zero doses on the STRONG-read phase (the pre-read exceeds the baseline -> deficit 0). Frozen verbatim regardless; the operationalizations carry the read-level fact.
- LEG-A LENGTH: 150 steps (3 events at the doubled spacing) vs the dispatch's '~100' — a birth decision (2 events cannot show a re-lock pattern), disclosed at birth.
- LEG-A'S SCHEDULE TAIL: from ~step 929 the house cosine decays below the corpus cap; the t950 event sits in the weaker-wash tail (recorded per-step in G_SCHED); the primary events t850/t900 are cap-bound.
- NO TWIN, BY DESIGN: the 2-cycle is an ERROR-GATED-arm property; the sanctuary twin has no events and no cycle (the family's x-ratio co-read not applicable).
- THE LEG POOLS are length-scaled (the per-step/per-event shares identical to e339's committed leg-2 — the wash protocol-matched; the unscaled pool rejected at birth as a confound).
- n=1 per leg, ONE organism, one lineage — the phase classes are deterministic continuations (not draws), but the cycle structure itself is one slot's realization.
- Checkpoints live INSIDE runs/x41/. No NOTES/THINKING/QUEUE/STATE edits (the heartbeat folds this cell).

*This cell does not edit NOTES/THINKING/QUEUE/STATE — the heartbeat folds.*