# x15 — THE DOSE-MATCHED COMPLEMENT (DEAD-AT-FULL-DOSE)

**The confound (R71 critic iii):** e310's complement (the full 10k write
MINUS its room-overlap components — 0% in-room by construction) read DEAD
(3.66e-05) at its natural 10.86%-energy dose, but everything inside a ~9x
energy gap dies (e306's r10 at 14.7% energy too) — the bearer-necessity
claim was dose-confounded until a complement at the FULL write's dose was
probed. **This cell is that probe:** the complement regenerated BIT-EXACT
from e310's own construction (G_REGEN: 65/65
keys bit-equal to e310's committed checkpoint; fp64 L2 3.024342 vs
committed 3.024342), scaled x3.03499 so its fp64 norm
equals the full write's 9.1788432658723 to
1.8e-15 (G_DOSEMATCH; the dispatch letter's ~x3.033
was off by ~0.1% — derived here, not trusted), injected by e310's exact
fp32 method, probed at t0 on e261's 60-window battery.

## Controls (both re-verified in-session BEFORE any arm counted)

- FULL write: 0.26464763283729553 vs committed
  0.26464763283729553 (|d| = 0.0e+00; G_FULLREAD PASS).
- UNSCALED complement: 3.655569e-05 vs committed 3.655569e-05
  (band [0.5x, 2x]; G_COMPREF PASS).
- Base e001 root: 1.338e-05 (e310 committed
  1.338e-05).
- Gates: 10/
  10 PASS (G_ENDPOINTS md5-binds 6 parents,
  G_FLATBASIS, G_BATTERY, G_ROOM, G_ORTH, G_REGEN, G_FULLREAD,
  G_COMPREF, G_DOSEMATCH, G_INROOM_SCALED).

## THE DOSE LADDER

| arm | scale | fp64 L2 (intended) | fp32 L2 (injected) | energy vs full | in-room share | t0 g0 | read | rider top-1 p | rider entropy | role |
|---|---|---|---|---|---|---|---|---|---|---|
| comp_x1 | x1 | 3.0243 | 3.0243 | 10.86% | 2.4e-18 | 3.65557e-05 | DEAD | 0.6495 | 1.1086 | (G_COMPREF control)
| comp_x1.5 | x1.5 | 4.5365 | 4.5365 | 24.43% | 2.3e-18 | 7.03682e-05 | DEAD | 0.6321 | 1.1769 |
| comp_x2.25 | x2.25 | 6.8048 | 6.8048 | 54.96% | 2.4e-18 | 0.000312887 | DEAD | 0.6594 | 1.1370 |
| comp_xFULL | x3.035 | 9.1788 | 9.1788 | 100.00% | 2.3e-18 | 0.0007308 | DEAD | 0.5161 | 1.7404 | (THE PRIMARY ARM — dose-matched)

## Verdict: DEAD-AT-FULL-DOSE

the full-dose complement (x3.0350, norm-matched to the full write to 1e-12) reads 0.0007308 < 0.01 — THE BEARER LAW SURVIVES DOSE-MATCHED: structure without room-overlap cannot read at any dose; the read needs the room's tail to couple into the context.

**P-x15a scoring (registered pre-compute, never shopped):**
DEAD-AT-FULL-DOSE (T278's author) — the e306 mix ladder (100% in-room 0.2646 / 50% at full dose 0.0631 / 0% at 33% dose dead) trends to zero with in-room share, and the mix's 0% arm was complement-dose-dead. -> **HIT — DEAD-AT-FULL-DOSE**.
Executor sub-prediction: CONCUR (registered pre-compute); sub-prediction P-x15a-exec: full-dose complement reads < 1e-3 (>= 50x below the dead bar), ladder flat-dead within one order of the 3.66e-05 unscaled reference. ->
**HIT — g < 1e-3**.

**The honest riders (organism health, never bars):** at xFULL the
complement carries 100.00% of the full write's
energy ALL out-of-room (9.21x the full write's own out-of-room
energy). Full-write control riders: top-1 p
0.4853 / entropy
1.4644; xFULL complement riders:
top-1 p 0.5161 / entropy
1.7404
(the confidence structure HELD — the dead read is a bearer-law fact, not organism collapse).

Disclosures: the complement is orthogonal to the room BY CONSTRUCTION
(G_ORTH identity at rel 2.2e-14);
the fp32 injection layer deviates from the fp64 dose-match at ~1e-7
relative (per arm in metrics.json; the full write's own fp32 cast in e310
reproduced the committed read bit-exactly, so fp32 IS the committed
instrument); scaling cannot change direction but every arm's in-room share
was re-measured anyway (max 2.4e-18,
G_INROOM_SCALED PASS); the letter's "0% at 33% dose" mix arm is the
comp-dose mix whose ENERGY share is 10.86% (its 32.95% is the complement's
MASS share — runs/e310/metrics.json); n=1 per arm (one lineage, one
session — the g-series standing lottery note carried verbatim).
