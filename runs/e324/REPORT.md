# e324 — THE INSTALL-DRAW CENSUS (run 2026-10-09T21:54:01Z)

**THE QUESTION** (dispatch verbatim): *what is the install-draw lottery's spread on the formation texture axes, at HELD room?*

## The census table (all draws, all axes)

| draw | gen | s1 g0 (side) | post g0 | gm12 | write norm | in-own-room |
|---|---|---|---|---|---|---|
| THE CANON (not rerun; artifacts) | 24314 | 1.35e-05 (DIE; 2nd room 1.35e-05) | 0.264648 | 0.10526 | 9.1788 | 0.9442 |
| FRESH 1/4 | 32401 | 0.7449 (SURVIVE) | 0.534977 | 0.70194 | 27.0284 | 0.6319 |
| FRESH 2/4 | 32402 | 0.7452 (SURVIVE) | 0.520091 | 0.65904 | 27.0825 | 0.6338 |
| FRESH 3/4 | 32403 | 0.7453 (SURVIVE) | 0.476808 | 0.66332 | 27.0870 | 0.6339 |
| FRESH 4/4 | 32404 | 0.7450 (SURVIVE) | 0.507923 | 0.72275 | 27.0342 | 0.6320 |

## The frozen bars, scored

- **SPREAD_g0** (max/min - 1 over the four fresh draws) = **12.2%** (bar: >= 40% INSTALL-DOMINATES / < 25% ROOM-COMPARABLE; e272's same-form room value 26.2%)
- **write norm varies x1.00** (bar > 2x); **in-own-room varies x1.00** (bar > 2x); gm12 varies x1.10 (co-reported)
- **VERDICT: ROOM-COMPARABLE**

> the fresh gens' g0 spread 12.2% lands within 25% (comparable to e272's room spread 26.2%) — e323's miss was a tail draw (or the room x gen interaction), not a wider gen wheel: the canon's monoculture caveat softens to a two-wheel lottery of similar sizes

## The s1-fork finding (never a bar)

**S1-FORK-CLEARS** — the draws split cleanly by step-one survival (all 5 census draws outside the [0.01, 0.15] gap: 4 SURVIVE / 1 DIE) AND the SURVIVE side's min write norm 27.03 > the DIE side's max 9.18 — survival predicts the bulky texture

## The canon's position

gen 24314 sits at rank 1/5 on g0 (EXTREME (min)), rank 1/5 on write norm (EXTREME (min)), rank 5/5 on in-own-room (EXTREME (max)) — ordered g0: 24314=0.2646, 32403=0.4768, 32404=0.5079, 32402=0.5201, 32401=0.5350

## Comparators

- e272's room lottery (same form): 26.2% (K10KR 0.2097 vs canon 0.2646, same gen 24314, two rooms)
- e323's both-wheels-fresh draw: 0.4788 (x1.81 the canon) — the census's held-room spread prices how much of that was the room's

## Gates

- 16/16 named gates PASS (all PASS)
- per-draw: G_PROTOIDENT 4/4 (the committed rig's protocol identity, verified from each run's own optimizer artifact) + G_FACTLOAD 4/4 (read-determinism re-loads)

## Envelope

- bursts <= 175s, cooldowns 40s, hard line 84C; 1624 polls tagged e324:*; max temp 78.0C; violations >= 84C: 0

## Prediction scoring

- P-e324a (lab guess, adopted): INSTALL-DOMINATES — MISS
- P-e324b (executor counter): ROOM-COMPARABLE — HIT (verdict branch); its
  discriminating texture clause (survivors cluster in-room >= ~0.85) —
  **MISS**: the survivors cluster at 0.632; the held room did NOT recapture
  the sprawl (split scoring, disclosed)

## The frozen bar's prose vs the data (the honest read)

The ROOM-COMPARABLE bar's baked-in prose said "*the e323 miss was a tail
draw, not a wider wheel*" — **the data contradicts the prose while
confirming the numeric condition**. e323's draw (0.4788) sits INSIDE the
fresh cluster [0.4768, 0.5350] — it was the TYPICAL survive-mode outcome,
not a tail; the **canon (gen 24314) is the extreme on every axis**. The
verdict token follows the frozen numeric condition (fresh-draw spread
12.2% < 25%); the census corrects the interpretation:

**THE INSTALL WHEEL AT HELD ROOM IS BIMODAL, NOT WIDE.** Within-mode
spread 12.2% (narrower than the room wheel's 26.2%); between-mode gap
(die vs survive) ~1.80-2.02x on g0, 2.95x on write norm, 6.4x on gm12 —
the modes ARE the s1-fork's sides, and the fork is gen-tracked (the canon
died at s1 in BOTH its rooms; all four fresh gens survived at the canon's
own room). The canon-vs-cluster contrast exceeds 2x on every texture
axis, but the frozen operationalization scoped the >2x clause to the four
fresh draws (registered at birth; no re-scoping after compute — the
finding reported, the bar not shopped). Consequence for the monoculture
caveat: it does not soften to "a two-wheel lottery of similar sizes" —
it sharpens to **"the canon is a die-mode draw of a bimodal gen wheel;
the survive mode (typical installs, ~0.48-0.53 at this room) reads
~1.8-2x the canon's 0.2646 with 3x the write norm"**. The die mode's
population share is unconstrained at n=4+1 (committed-selected).

## Catches / disclosures

- PLOT RENDER FIX (post-run, disclosed): the four fresh draws cluster tightly (x 0.477-0.535) and the first render piled their annotations; make_census_plot's label offsets were staggered by x-rank and the PNG regenerated from the recorded census (zero recompute; the registration sections untouched).
- ONE THERMAL-MARGIN BURST END (the envelope working as designed): GEN32402's chunk 2 ended at the 78C margin after 135 steps (resume ckpt + cooldown + continuation — zero hard violations, max 78.0C).
- THE FOUR INSTALLS RUN THROUGH e261's chunked_install BY IMPORT (the committed rig VERBATIM — extend, don't repeat): the module-global FRESH_GEN is rebound per draw (32401-32404; the committed 24314 is co-reported); the room ladder is the HELD pair 26113/26114; the step body (draws, clip, the room hook, the milestone battery) is the committed file's, NOT modified.
- NO CONSOLIDATION PHASE (disclosed): the census axes — g0 at its own battery, write norm, in-own-room, gm12, s1 — are all INSTALL-phase quantities; the canon's 0.2646 is the install-phase read (e264's post_cells); e323's +81% was install-phase too. The cons (e113 seed 10901) is a different question's machinery.
- NO FORMATION GATE (disclosed): e323's halt taught that the [0.15, 0.45] band does not contain fresh draws; a CENSUS does not reject its own data — the band is drawn on the plot as context only; every draw lands wherever it lands (no shopping, no redraws).
- THE CANON IS NOT RERUN: gen 24314's row is READ from e264's metrics (post g0/gm12/gp12, install-traj s1, in-own-room) + e290's metrics (the write norm, its G_FACTLOAD record) + e272's metrics (the K10KR s1 — the die's second-room confirmation) at RUNTIME, asserted against the birth literals; the census row carries the runtime-read values.
- e290 IS ADDED TO THE DISPATCH'S PARENT SET (e261/e264/e272/e323 + the room artifacts): the committed write norm 9.1788 lives in e290's G_FACTLOAD record (e264's metrics holds the fractions, not the norm) — an extension of the chain, never a replacement.
- THE FRESH INSTALLS EACH START FROM THE SAME COMMITTED ROOT (g1c_root.pt, bit-gated once, held in memory across draws — the census's held organism; e264's committed K10K install started from the same root).
- n=4 fresh draws, one lineage, one session (the g-series standing lottery caveat — the critic's note carried verbatim); a 4-draw census estimates a distribution's coarse spread, not its tails; nothing guaranteed.
- No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).
- Smoke mode (E324_SMOKE=1): 8-step installs, room at k=512 (the bit-bind vs e264_rooms.pt's k=10k record is VACUOUS at smoke — disclosed, LIVE and binding at the full run), all paths smoke_-prefixed, own smoke dir; NOTHING adjudicated (SMOKE stamp on every read).

*This cell does not edit NOTES/THINKING/QUEUE/STATE — the heartbeat folds.*