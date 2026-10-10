# E342 — THE PHASE AUDIT

**Date:** 2026-10-10T19:01:59Z (datetime.now(UTC) stamps in metrics) · **Executor report**

## THE VERDICT (against the frozen bars)

**SAG-EXISTS**

```
mean sag 48.0% >= 15% (the mid-phase reads fall materially below the boundary reads; per-period sags 60%/34%/60%/31%/66%/25%/60%) — THE FOUNDING HEIGHTS ARE RAILS: the endpoint class carries the phase clause (Law 4's endpoint language joins x41's rider); the 3.16-3.62x band is an upper-rail band — the between-events state of this trajectory sits at 34%-75% of its boundary reads at mid-period and at 18%-31% at the floors (the committed floors already showed the valleys; the mid-phase point is the new datum)
```

```
THE FOUNDING NUMBERS' PHASE CHARACTER: the founding milestones (e336 leg-1: 0.6703/0.6985/0.7698/0.8147; the canon endpoints' band 3.16-3.62x) were ALL read post-dose; the committed desk half shows their own-period floors at 38.2x-6.5x below them (the gates); the live half measures the mid-period state at 52% of the boundary read on average — the founding heights are UPPER-RAIL samples of an intra-period oscillation; the phase clause joins the endpoint class
```

## The sag profile (boundary vs mid-phase per period)

| period | event step | dose (deficit) | boundary post | MID (+13) | next gate (floor) | sag | mid/floor | peak/floor |
|---|---|---|---|---|---|---|---|---|
| ev 17 | t425 | DOSE (0.261) | 0.6340 | 0.2536 | 0.1292 | 60.0% | 1.96 | 4.9x |
| ev 18 | t450 | DOSE (0.548) | 0.8394 | 0.5555 | 0.2060 | 33.8% | 2.70 | 4.1x |
| ev 19 | t475 | DOSE (0.279) | 0.6770 | 0.2723 | 0.1250 | 59.8% | 2.18 | 5.4x |
| ev 20 | t500 | DOSE (0.563) | 0.8530 | 0.5881 | 0.2455 | 31.1% | 2.40 | 3.5x |
| ev 21 | t525 | DOSE (0.141) | 0.5161 | 0.1738 | 0.1018 | 66.3% | 1.71 | 5.1x |
| ev 22 | t550 | DOSE (0.644) | 0.9032 | 0.6744 | 0.2786 | 25.3% | 2.42 | 3.2x |
| ev 23 | t575 | DOSE (0.025) | 0.3256 | 0.1304 | 0.0904 | 59.9% | 1.44 | 3.6x |

- **MEAN SAG 48.0%** vs the frozen bars (SAG-EXISTS >= 15% / NO-SAG < 5% / PARTIAL-SAG between)
- the +1 window reads (the decay's first step): event 17 (small-dose, e339-comparand): post 0.6340 -> +1 0.6155 (+2.9%); event 23 (small-dose, the new datum): post 0.3256 -> +1 0.3021 (+7.2%); event 21 (big-dose; e339's committed t600->t601, CITED): post 0.9000 -> +1 0.8983 (+0.2%)
- the zero events inside the leg: none

## The founding numbers' phase character

- DESK HALF (committed records only): the founding milestones 0.6703/0.6985/0.7698/0.8147 all post-dose; their own-period floors [0.0175, 0.1121, 0.1235, 0.1254]; peak/floor 38.2x/6.2x/6.2x/6.5x
- the canon's deficit medians 0.6-0.7 (the dispatch's own citation)
- LIVE HALF: this leg's mean sag 48.0%; the founding heights are UPPER-RAIL samples

## The e339 shared-draw co-read (non-halting)

- t500: mine 0.852994 vs e339 committed 0.852994 (|d| 0.00000)
- t600: mine 0.899997 vs 0.899997 (|d| 0.00000)
- win1 quadruple max |d| 0.00000 (tol 0.05 — disclosed, never a gate)

## Gates

- 22 gate classes instantiated, all PASS: G_E336RECORD, G_E339RECORD, G_NAMEWIN, G_NAMEFREE, G_BASE, G_VMBIND, G_SPANBIND, G_ROOMK10K, G_LR_BIND, G_SUBSTRATE, G_PROTOCOL_IDENT, G_RESUME, G_T400_REPRO, G_STAGING, G_LEG_IDENTITY, G_PHASEDECL, G_E339_COVERREAD, G_BUFSEP, G_ORTH, G_STREAMELIVE, G_BUDGET, G_NET0

## Envelope

- 218 thermal polls (source: the durable ledger runs/_envelope_log.jsonl tagged e342:* (the reused-leg output re-run stages no GPU work; the leg's own polls live there)); max temp 62.0C; burst cap 175s; cooldown 40s; zero concurrent GPU jobs; every burst logged to runs/_envelope_log.jsonl tagged e342:AUDIT-LEG:*; chunk durations 175s/177s/183s/177s/141s
- THE BURST-DURATION NOTE (disclosed): the phase-declared cadence puts CPU milestone reads INSIDE burst windows (15 vs any prior cell's 4); one chunk's RECORDED duration is 183s vs the 175s driver cap (the post-step break check + the edge milestone read) — the thermal never-past line untouched (max 62C of 84C); zero poll violations

## Registered predictions

- P-e342a (the executor's own read: SAG-EXISTS, moderately): HIT
- the dispatch's lab lean (SAG-EXISTS, weakly): HIT
- the shape predictions: mean in [0.35, 0.60]: True; every mid above its floor: True

## Catches / disclosures

- THE SUBJECT: e336's committed t400 resume state (model + optC + optF + both generator states) md5-bound, bit-compared vs the committed post checkpoint, and read-reproduced to e336's committed t400 == x39's subject literals (2e-6/5e-3) BEFORE any stepping.
- THE LEG: 8 events (the dispatch's 4-8 at its top), M=25 UNCHANGED, pools length-scaled (x41's frozen form; LR_M_MAX never scaled) — the per-step/per-event equal-share arithmetic identical to e336's leg-1 and e339's leg-2 at their starts.
- THE e339 CO-READ is a consistency check at a disclosed 0.05 tolerance, never a gate (cross-run GPU op-order determinism is not the family's law).
- THE MID OFFSET +13 frozen verbatim from the dispatch; period 24's mid (t613) sits beyond the leg — 7 complete pairs, no pair dropped.
- n=1: ONE organism, one lineage, one leg — the sag profile is a single realization's, read against the founding class's committed record.
- Checkpoints live INSIDE runs/e342/. No NOTES/THINKING/QUEUE/STATE edits (the heartbeat folds this cell).

*This cell does not edit NOTES/THINKING/QUEUE/STATE — the heartbeat folds.*