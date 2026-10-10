# E339 — THE LONG LANDING

**Date:** 2026-10-10T11:58:30Z (datetime.now(UTC) stamps in metrics) · **Executor report**

## THE VERDICT (against the frozen bars)

**PLATEAUS-BELOW**

```
the curve flattens below the band: ratio_800 3.0549 < 3.112, g_last -0.0473 — A GENUINE LOWER EQUILIBRIUM; 'the founding band was ZEPHYRA's slot' stands; the name-tuned rider confirms
```

## The trajectory (post-maintenance milestone reads)

| t | ratio vs own baseline | gain/100steps |
|---|---|---|
| 400 (e336 committed) | 2.8502 | +0.1571 |
| 500 | 2.9840 | +0.1339 |
| 600 | 3.1485 | +0.1644 |
| 700 | 3.1022 | -0.0463 |
| 800 | 3.0549 | -0.0473 |

- monotone rise: False; flat (|g_last| < 0.05): True; in band: False
- the slope at t800: -0.0473 per 100 steps — SETTLED
- raw read at t800: 0.873240 (ceiling 1.0 = ratio 3.4983)

## The discriminator trace (the dose's own input)

- pre-event gate reads, leg 2 (events 17-32): [0.2111, 0.1292, 0.206, 0.125, 0.2455, 0.1018, 0.2786, 0.0904, 0.3542, 0.0983, 0.3844, 0.1165, 0.3873, 0.1339, 0.3679, 0.1492]
- deficits, leg 2: [0.261, 0.548, 0.279, 0.563, 0.141, 0.644, 0.025, 0.684, 0.0, 0.656, 0.0, 0.592, 0.0, 0.531, 0.0, 0.478] (leg 1 committed: 0.947 -> 0.561)
- the dose self-limits as the floor rises toward the baseline 0.2859

## The cumulative spend (the doubled-horizon cost)

- leg 1 (committed): S_corpus 2.6849 + S_maint 1.0471 = 3.7320 (83.4% of its budget)
- leg 2 (measured): S_corpus 2.6849 + S_maint 0.8752 = 3.5601 (79.6% of the leg budget 4.4748)
- CUMULATIVE t0->t800: 7.2920 = 0.8148x the write norm (leg convention: a fresh 0.5x-write budget per 400-step leg — frozen at birth; the stretched-pool alternative rejected)

## The twin and the co-reads

- the twin at t800: 0.0016520 (x0.0058) — x-ratio_800 528.6x (x-ratio_400 committed 281.9x)
- the purity read p(Z) at t800: 1.05e-05 (the committed prior 1.88e-04) — no Z read built

## Gates

- 17 gate classes instantiated, all PASS: G_NAMEWIN, G_NAMEFREE, G_BASE, G_VMBIND, G_SPANBIND, G_ROOMK10K, G_LR_BIND, G_SUBSTRATE, G_PROTOCOL_IDENT, G_RESUME, G_T400_REPRO, G_STAGING, G_STREAMELIVE, G_SCHED_IMMATERIAL, G_ORTH, G_BUFSEP, G_BUDGET

## Envelope

- 816 thermal polls; max temp 74.0C; burst cap 175s; cooldown 40s; zero concurrent GPU jobs; every burst logged to runs/_envelope_log.jsonl tagged e339:...

## Registered predictions

- P-e339a (the executor's own read: LANDS-IN-BAND, weakly): MISSED
- the dispatch's lab lean (LANDS-IN-BAND, weakly): MISSED

## Catches / disclosures

- THE RESUME: e336's committed t400 states (model + optC + optF + both generator states) md5-bound, bit-compared vs the post checkpoints, and read-reproduced (G_T400_REPRO) BEFORE any compute; the stream CONTINUES (generators restored, never re-seeded).
- THE LEG CONVENTION: 'the same budget convention' = e336's 400-step protocol VERBATIM as a second leg (a fresh 0.5x-write-norm budget, pools split 60/40, the driver's own equal-share caps); the stretched-pool alternative REJECTED at birth (B_C exactly spent at t400 — the wash would die; a forfeit landscape).
- THE SCHEDULE: cosine_lr(step-1, 1000) VERBATIM on the global step; disclosed immaterial — the corpus cap bound below the schedule at every logged leg-2 step (verified in G_SCHED_IMMATERIAL); lr_m is schedule-free.
- The drivers are e336's COMMITTED bodies executed VERBATIM under the disclosed leg re-binding (PHASE_STEPS/MILESTONES/WIN1/WIN2 only); every protocol constant asserted == the md5-bound record first.
- The founding band is TRANSPLANTED verbatim [3.112 (hugged), 3.616]; the top unreachable (ceiling 3.4983); an in-hug landing stamped EDGE-GRAZE (e336's frozen convention).
- n=1 per arm, ONE organism — the landing curve is a single draw read against the founding class's five-draw band.
- Checkpoints live INSIDE runs/e339/. No NOTES/THINKING/QUEUE/STATE edits (the heartbeat folds this cell).

*This cell does not edit NOTES/THINKING/QUEUE/STATE — the heartbeat folds.*