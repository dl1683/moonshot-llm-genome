# E323 — THE CASCADE GUARD (R71's critic pick; consult #010's adopted next-slot)

**Date:** 2026-10-09 (datetime.now, UTC stamps in metrics) · **Executor report**
**Commits:** birth `871e2f3` -> smoke `3f1b7f7` -> halt-path completion `500ba4a` -> primary complete `b22f4ce` -> rider halt-path `512966d` -> this fold.

## THE VERDICTS (against the frozen bars)

- **PRIMARY: TEXTURE (G_FORMATION)** — *not* BOTH-EDGES-HOLD and *not*
  EITHER-MOVES. The fresh draw's write landed **outside** the formation
  trust band: **g0 0.478831 vs [0.15, 0.45]** (x1.809 the committed
  0.2646; e272's room-only redraw was x0.793). The frozen no-shopping
  branch fired exactly as registered at birth: ONE draw, no redraw, the
  miss is the datum. **THE RUNGS NEVER RAN** — the EITHER-MOVES /
  BOTH-EDGES-HOLD question is **unanswered at this draw**; e290's
  bracket stays n=1 with its formation-side uncertainty now measured.
- **RIDER: RIDER-TEXTURE (G_RIDER_FORMATION)** — *not*
  COLLATERAL-REPLICATES and *not* FAMILY-DRAW-DEPENDENT. The fresh
  family redraw's completed-organism baselines landed outside
  [0.05, 0.45]: **FACT3 (the target) 0.5211** (e291's 0.2678) and
  **FACT5 0.4682** (e291's 0.3815); FACT1/2/4 in band (0.294/0.347/0.435).
  **THE ANTI ARM NEVER RAN.** The membrane law's n=1-family objection
  stands unanswered — with the reason now named.

Both registered predictions (P-e323a BOTH-EDGES-HOLD; P-e323r
COLLATERAL-REPLICATES) are **UNTESTED**, not fallen: their cells of the
decision table were never reached. The counter-branches (P-e323b/P-e323s)
equally untested.

## THE AUTOPSY (what the miss shows — the session's real finding)

**The install-draw lottery at formation dominates the room lottery, and
the committed "family" shares ONE install draw.**

1. The committed formation family {0.2646 [e261/e264], 0.2097 [e272
   K10KR]} both ran install gen **24314** — e272 varied only the room.
   Both committed trajectories read **g0 0.0000 at s1**: gen 24314's
   first AdamW step (the ±lr sign-kick) KILLS the read, and both
   formations are kill-then-recover curves (s100 ≈ 0.31, converging to
   0.21–0.26).
2. The fresh draw (gen **32302**) **survived its first step** (s1 g0
   0.7451 ≈ the root landing; CE_R moved 1.616 → 1.686, so the step
   happened) and formed a **bulkier, less-localized write**:

   | read | committed | e323 fresh | ratio |
   |---|---|---|---|
   | post g0 | 0.2646 | 0.4788 | x1.81 |
   | gm12 (localization) | 0.1053 | 0.6849 | x6.50 |
   | write norm | 9.1788 | 27.0568 | x2.95 |
   | in-own-room | 0.9442 | 0.6329 | x0.67 |
   | CE_R | 1.5921 | 1.6374 | +0.045 |

3. The rider's five fresh-seed installs (gens 32321–32325, shared-frame
   rooms 32311/32312) propagate the same texture: baselines 0.294–0.521
   vs e291's 0.267–0.382, write norm **31.84 vs 19.27** — six
   independent fresh install draws on this machine, all landing high or
   at-family-top.

**Reading (for the coordinator's THINKING fold):** the formation
instrument's family band was calibrated on the ROOM lottery only
(~21%). The INSTALL-draw component is larger (+81% here) and
regime-structured: whether the first AdamW step's sign-kick kills the
read appears to be a property of the first batch draw
(ix/aj/rj), and kill-recovery formations are leaner (write ~9.2, 94%
in-room, gm12 0.105) while survive-carry formations are bulky (~27,
63% in-room, gm12 0.68). The cascade's guard set out to price the
RETENTION bracket's draw uncertainty and found a bigger crack
UPSTREAM: the five dependents (Law 2b, Law 4's necessity consequence,
Law 5's Landauer pricing, T277's premise, e311's dose scale) inherit a
formation-side draw structure their prose hides — e290's committed
numbers stand as one-draw measurements (the frozen rider), now with the
draw's identity named.

**Sharpened follow-up designs (not run; for QUEUE):**
(a) the install-draw lottery measured as a ladder (n≥3 fresh gens at
held room 26113/26114 — isolates the gen component the way e272
isolated the room);
(b) the first-step-shock survival as the fork variable (classify
committed vs fresh formations by s1 read == 0 vs not — a 1-eval
discriminator on existing checkpoints);
(c) a retention re-ladder on e272's K10KR state (room-only redraw,
gen-held — the in-family fresh state already on disk).

## Gates

16 gates instantiated: **14 PASS**, 2 registered halts fired
(G_FORMATION, G_RIDER_FORMATION — the verdicts themselves). Parent
chain md5-bound on git-canonical hashes: e290 (the bracket), e272 (the
lottery precedent), e264/e261 (the install rig), e285, e273
(LR_STABLE), e291/e294/e291_organism.pt (the rider's literals:
BUDGET 4.58942163293615, LR_ANTI_MAX 0.02250444534525284, DOSE 0.05,
ERASE_BAR 0.01 — bound and asserted, never consumed by compute). The
fresh room certified (idem 5.6e-16, kept2 within bar, span-overlap on
expectation) and verified NEW (D and S both differ from e264's
committed K10K; index overlap 0.42%).

## Envelope (zero violations)

2,436 per-step polls tagged e323:*, **max 77.0C**, 0 events ≥ 84C;
bursts capped at 175s with 40s cooldowns throughout; GPU sequential
(the x16 CPU agent unaffected; CPU threads 4); gpu_ok() checked at
startup and the machinery polls every step.

## Catches

1. **Smoke catch (pre-compute):** G_FORMATION (and G_RIDER_FORMATION)
   wrongly fired at smoke k=512 — no committed formation record exists
   at smoke k (e285's smoke note). Made SMOKE-VACUOUS per e290's
   G_ROOMK10K precedent; LIVE at full k=10k. No bar moved.
2. **Halt-path completion (disclosed, post-halt):** the full run's
   formation miss raised before the adjudication/figure were written;
   the halt path was completed (TEXTURE adjudication + autopsy figure +
   the miss's checkpoint) and the script re-ran on resume ckpts (zero
   recompute). Same repair applied to the rider's halt path
   preemptively. No bar moved.
3. **The miss itself** — disclosed as the datum, per the birth
   registration's no-shopping clause.

## Artifacts

- `runs/e323/metrics.json` (COMPLETE; the frozen registration verbatim,
  both adjudications, full ledgers/trajectories)
- `runs/e323/e323_cascade_guard.png` (the formation autopsy: trajectories
  vs the committed family, the band, the miss, the write-structure
  comparison)
- `runs/e323/e323_rider_family.png` (the family redraw's formation
  autopsy: baselines vs e291, the five installs' trajectories)
- Checkpoints (gitignored, md5s in metrics): e323_rooms, e323_fresh_fact
  (the miss's state, for the follow-up designs), e323_rider_rooms,
  e323_rider_organism, + all resume ckpts

**Not touched:** other runs/, lab/x16*, NOTES/THINKING/QUEUE/STATE
(the heartbeat folds this cell).
