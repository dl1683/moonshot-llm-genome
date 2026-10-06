# E280 — THE CAPACITY LADDER UNDER SGD-M (the day-twelve headline cell)

**VERDICT: ADAM-CREATED** — "1k EXPRESSES under SGD-M (>= 0.05) or the edge
moves by >= 2x — the floor is Adam's preconditioned geometry." The edge
MOVED: the SGD-M bracket is **(2k, 5k]** (firing pair 2k->5k at **127.3x**,
dead side 0.001047 < 0.01) vs AdamW's committed **(1k, 2k]**. 1k did NOT
express (0.000127); the moved-edge clause fired. All 14 instantiated hard
gates PASS (G_MISSILE_ORTH's bookkeeping omission disclosed below).

The question (frozen verbatim): *is the expression edge at (1k,2k] Adam's
preconditioned geometry or the parameter space's own physics?*

## The two ladders (post g0, the WRITE read; rooms bit-bound, optimizer the only delta)

| k | AdamW (committed cites) | SGD-M @ stable lr (this cell) | ratio SGD/AdamW |
|---|---|---|---|
| 1,000 | 0.00043458 (e261) | **0.00012719** | 0.29x |
| 2,000 | 0.02661625 (e272) | **0.00104674** | 0.039x |
| 5,000 | 0.12709597 (e272) | **0.13328430** | 1.05x |
| 10,000 | 0.26464763 (e264) | **0.21675554** | 0.82x |

The structure in one sentence: **BELOW the edge SGD-M suppresses expression
3-25x; ABOVE the edge the two optimizers agree within ~20%** — the 5k rung
under SGD-M at x0.01 of matched dose actually EXCEEDS AdamW's committed 5k
(0.1333 vs 0.1271). The floor is where the optimizers differ; the climb
above it is broadly optimizer-robust. The forming phase is delayed under
SGD-M (S5K: 0.0002 at s100 -> 0.1333 at s400; S10K: 0.0048 -> 0.2168 —
the write waits for the cosine to climb), consistent with the disclosed
undermatch binding hardest early and at low k.

## THE W046 COORDINATE — (floor-bracket, exponent) per optimizer

- **AdamW: ((1k, 2k], 1.4417)** — fit over {2k, 5k, 10k} (r2 0.984);
  the 40k-included variant 0.8251 (W046's own caveat carried: the curve is
  convex-to-power early; the 2k-10k local slope is steeper than the 118x
  2k-237k slope ~0.63).
- **SGD-M: ((2k, 5k], 0.7016)** — fit over its own bracket-top domain
  {5k, 10k} (2 points, slope exact).
- Shared-domain context: over {5k, 10k} alone, AdamW 1.058 vs SGD-M 0.702.

W046's fork, answered: the bracket MOVED (Adam's geometry sets the floor)
while the above-edge scaling stays sub-linear and same-order in both
optimizers — an intercept shift with the exponent broadly intact, not a
reshaping of the whole climb.

## THE RIDERS

**(a) THE FIRING-PAIR REPLICATE (the R66-critic debt): REPLICATE-LOTTERY.**
The fresh 1k room landed 0.000154 (0.354x of the committed 0.000435 —
below the [0.5x, 2x] band); the fresh 2k room landed 0.009135 (0.343x of
the committed 0.026616 — below the band AND below the dead bar). Both rungs
of the AdamW edge pair drew ~2.9x low TOGETHER. Read verbatim: at n=2 per
rung the (1k,2k] bracket's top straddles the dead bar (2k posts: 0.0266,
0.0091) and the dead side spans 0.00015-0.00044 — the edge's cushion on
fresh rooms is real but the n=1 caveat at the pair is now PRICED, not
hypothetical. The 61x committed jump survives at the fresh pair (59x), but
any single-room claim about WHERE the edge sits inside (1k,5k] carries a
~3x room-draw uncertainty.

**(b) THE 0.5x-COMPENSATED 1K ARM: 0.5X-DEAD** — post 0.000618 < 0.01.
P-280b CONFIRMED: dose acquitted at three lr points (1x -> 0.000435; 1.865x
-> 0.000618; 3.731x -> 0.001245 — all dead, a faint monotone creep that
never approaches the floor).

**(c) THE SGD-MISSILE (P-x283b): MOTEL-IS-SPACE** — the orthogonalized
corpus stream's realized displacement under SGD-M ran in-room at interval
fracs {s100 0.4193, s200 0.4926, s300 0.4553, s400 0.4152}, median **0.4553
>= 0.35** (cumulative 0.419 -> 0.627; AdamW's committed band 0.4814-0.5988
intervals, 0.586 -> 0.673 cumulative). The gradient was exactly orthogonal
throughout (orth_max 4.6e-17). P-x283b's direction FAILED: removing Adam's
per-coordinate normalizer did NOT open the escape — the displacement-level
funnel survives the optimizer swap at ~0.42-0.49 vs Adam's ~0.48-0.60.
DISCLOSED MECHANICAL READING (the honest fine print): the construction is
e278's VERBATIM — ONE SHARED momentum buffer — so each corpus step's
realized displacement carries the install stream's in-room momentum
(injected every other step). SGD-M has no normalizer, but the shared buffer
is itself an optimizer-architecture coupling channel: the letter says
MOTEL-IS-SPACE, and what the read strictly shows is that the funnel is NOT
Adam-normalizer-specific; a SEPARATE-buffer SGD missile (the e273
separate-barrel question ported to the missile) remains untested — the
natural follow-up. The missile's own write died (post 0.00069, like AdamW's
0.00055): the orthogonal stream still kills.

## DISCLOSURES (the honest ledger)

1. **THE LR IS STABLE-BUT-UNDERMATCHED (the cell's central caveat, frozen
   at birth):** LR_STABLE = x0.01 x 21.7385748014537 = 0.2173857, momentum
   0.9, wd 0.0 — e273's SGD001X stable rider's exact scale and
   hyperparameters. The matched lr diverges (e273's committed record);
   consult #007's license adopted (e272 acquitted dose). The per-rung
   FIRST-STEP APPLIED-L2 LEDGER (measured): s1 in-room L2 = 4.4e-5 (1k),
   6.0e-5 (2k), 9.1e-5 (5k), 1.32e-4 (10k) — exactly x0.01 of the matched
   target, momentum-exact to <= 6.2e-4 rel err. The rungs DID form (5k
   expresses at AdamW's own level), so the INCONCLUSIVE-AT-THIS-LR branch
   did not bind — but the 2k rung's 25x suppression is the undermatch's
   likely fingerprint, and the bracket move (2k->5k vs 1k->2k) should be
   read with that caveat attached: the floor is optimizer-sensitive; how
   much of the move is geometry vs dose-at-2k is not separable in this
   cell. The verdict is the frozen letter's, carried verbatim.
2. **G_MISSILE_ORTH — A BOOKKEEPING OMISSION (caught at fold, disclosed):**
   the docstring's declared hard-gate set includes G_MISSILE_ORTH; the
   implementation records the orth ledger (max rel err 4.57e-17 over all
   400 steps, in the arm record + the envelope of the driver) but did not
   instantiate a separate gate ROW in metrics['gates']. The gate would
   read PASS by ~10 orders of magnitude (bar 1e-6); no result depends on
   it. The metrics were not edited post-hoc.
3. **THE RUN WAS INTERRUPTED AND RESUMED (determinism preserved):** the
   first pass completed R1KR/R2KR/K1KM05/S1K/S2K/S5K before its wrapper
   was killed at the S5K->S10K cooldown; the resume pass re-ran the gates
   and resumed all six arms from their complete step-400 checkpoints
   (reads identical to 6+ digits — the lineage's resume-determinism law).
4. **NO CONS (T259/e281; e278/e283's form):** the landing read is a cons
   property; every arm's install-final state is checkpointed
   (runs/checkpoints/e280_<arm>_post.pt) for any later landing pass.
5. **40k/237k SKIPPED** (budget/cited; e264's posts appear in the overlay
   and the fit variant only); **n=1 per arm, one lineage, one session**
   (the standing caveat — rider (a) exists to price it); **the SGD serial
   arms' displacement is 100.000000% in-own-room** (in_own_room
   0.99999999997 — SGD's step is exactly a linear combination of projected
   gradients; AdamW's 10k write sat at 0.9445 — the normalizer rotates
   ~5.5% of the write OUT of the room; a mechanism datum in its own right).

## Gates (all PASS)

G_NAMEFREE (ZEPH x0) / G_SPLICE (19+41) / G_BATTERY (shapes) / G_ANCHOR
(16-window bank) / G_INSTMASK (420) / G_PARENTS (e261+e264+e272+e273+
lr_calibration + e272_rooms + span, all md5-bound) / G_LR_BIND (LR_STABLE
provenance exact) / G_DOSE_ARITH (x1.8652783844 exact) / G_BASE (fact-free
1.4e-5) / G_ROOT (0.9026 bit-read) / G_VMBIND / G_SPANBIND / G_PROJ (6
rooms certified, idem <= 5.6e-16, kept2 within 10-sigma) / G_ROOMS (the
four ladder rooms D/S bit-identical to e272_rooms.pt — the optimizer is
the SGD arms' ONLY delta).

## Envelope

bursts <= 175 s (dispatch 180), cooldowns 40 s (dispatch 30-60), max temp
**78.0 C** at the early-end margin (never-past line 84 C, dispatch 85;
0 violations), 3,660 per-step polls persisted to runs/_envelope_log.jsonl
tagged e280:<ARM>:<phase>; serial arms only. Total ~2,329 s resumed pass
+ ~3,542 s first pass.

## Provenance

- Script: lab/e280_sgd_ladder.py (BIRTH commit 6385b2a, BEFORE any
  compute; smoke 95e9f55 — 15/15 gates at smoke scale, zero code bugs).
- Machinery: e261's chunked_install VERBATIM by import (the three AdamW
  arms); chunked_install_opt (e261's driver, optimizer parameterized +
  first-step ledger); chunked_missile_sgd (e278's MISSILE driver, the ONE
  SHARED optimizer swapped to SGD-M); e258's v-map + e246's span LOADED.
- Parents hard-bound: e261 f460475d, e264 a42ff478, e272 eb9f6247, e273
  df0f608a, lr_calibration de0b1c3e, e272_rooms 06694485.
- Predictions: P-280m REFUTED (the bracket did not hold under SGD-M);
  P-280b CONFIRMED (0.5x dead); P-x283b's direction REFUTED (the missile
  does not escape under SGD-M; the funnel is not normalizer-specific).
