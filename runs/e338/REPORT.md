# E338 — THE COMMIT EVENT AS CONSOLIDATOR (the consolidation chamber's first cell)

* VERDICT: **MIXED** — the table verbatim: committed retention 0.0448 (in_band False; min panel read 0.004079); twin retention 0.0036 (SAND-STRICT; min panel 0.001033). the committed read fell below 0.05 (died WITHIN the ball) but its retention is outside the sand class
* P-e338a (my registered read): MISSED (guess: BALL-FORMS, WEAKLY-TO-MODERATELY (against the lab lean); the dispatch's lab lean: STILL-SAND, weakly)
* gates: 16/16 PASS

## The event (the port disclosure)

The event is NOT re-coded: `lab/g1_anchored_ball.py`'s `CommittedGPT` runs unmodified via import, fired exactly as g1b/g1c fired it —

```python
net0 = G1.CommittedGPT(GB.G1B_CFG)   # the 2.74M family (GB rebind)
net0.load_state_dict(theta0)          # the body == the subject
net0.commit(0.7)                      # THE EVENT (R = the W1 dial)
```

THE QUOTED CONSTRUCTION (substring-verified at run time):

> `lab/g1_anchored_ball.py`: `def commit(self, R: float) -> None:` [VERIFIED] — *(the event's signature (the class method, verbatim))*
> `lab/g1_anchored_ball.py`: `Snapshot every trainable tensor into registered non-trainable` [VERIFIED] — *(the event's own docstring line 1 (the anchor snapshot))*
> `lab/g1_anchored_ball.py`: `hard in-place L2 projection onto the ball` [VERIFIED] — *(THE WALL's construction (the projection step's own words))*
> `lab/g1_anchored_ball.py`: `d > self.R:` [VERIFIED] — *(the projection's exact condition (outside-the-ball test))*
> `lab/g1_anchored_ball.py`: `p.copy_(a + (p - a) * s)` [VERIFIED] — *(the projection's exact rescale (onto the surface))*
> `lab/g1b_continuity.py`: `R_LADDER = (0.7, 1.4, 4.2)` [VERIFIED] — *(the R dial, frozen in the firing cell (W1 = 0.7))*
> `lab/g1b_continuity.py`: `net0.commit(R)` [VERIFIED] — *(the firing line itself (the event as the era ran it))*
> `lab/g1c_root_redraw.py`: `R=0.7 RAW L2` [VERIFIED] — *(the ball's birth cell: the 2.74M convention verbatim)*

The wall's semantics (the class's own words): every forward opens with a hard in-place L2 projection onto `||theta - theta_anchor||_2 <= R`; the optimizer may step one step outside between forwards and the next forward pulls the parameters back onto the surface. The panel reads on the committed arm go through the ARMED CPU eval twin (g1_wash's convention).

## The arms (e322's 100-step unbiased wash, seed 10902, bit-identical draws)

### ARM (a) COMMIT — TAVIREN + commit(0.7)

| step | read p(T)@g0 | gm12 p(T) | host p(Z) | CE_R | |d| raw | proj | note |
|---|---|---|---|---|---|---|---|
| 0 | 0.285851 | 0.1744 | 0.0002 | 1.5853 | 0.0000 | 0.0 | t0 (== the committed subject read) |
| 1 | 0.004079 | 0.0042 | 0.0001 | 1.8997 | 1.6543 | 0.7 | **< 0.05** |
| 5 | 0.010868 | 0.0098 | 0.0000 | 1.6188 | 1.1977 | 0.7 | **< 0.05** |
| 10 | 0.008039 | 0.0065 | 0.0000 | 1.6141 | 1.0901 | 0.7 | **< 0.05** |
| 25 | 0.015989 | 0.0103 | 0.0001 | 1.6172 | 1.0706 | 0.7 | **< 0.05** |
| 50 | 0.017449 | 0.0121 | 0.0001 | 1.6252 | 1.0698 | 0.7 | **< 0.05** |
| 100 | 0.012817 | 0.0080 | 0.0001 | 1.6274 | 1.0616 | 0.7 | **< 0.05** |

* retention s100/t0: **0.0448** — NOT-SAND; outside the installed-fact band

### ARM (b) TWIN — TAVIREN uncommitted

| step | read p(T)@g0 | gm12 p(T) | host p(Z) | CE_R | |d| raw | proj | note |
|---|---|---|---|---|---|---|---|
| 0 | 0.285851 | 0.1744 | 0.0002 | 1.5853 | 0.0000 | None | t0 (== the committed subject read) |
| 1 | 0.005213 | 0.0076 | 0.0000 | 3.5422 | 1.6543 | None | **< 0.05** |
| 5 | 0.001535 | 0.0024 | 0.0000 | 2.7602 | 4.1566 | None | **< 0.05** |
| 10 | 0.002690 | 0.0033 | 0.0000 | 2.1290 | 5.5602 | None | **< 0.05** |
| 25 | 0.004460 | 0.0051 | 0.0001 | 1.7530 | 7.2989 | None | **< 0.05** |
| 50 | 0.002277 | 0.0024 | 0.0000 | 1.7635 | 8.7760 | None | **< 0.05** |
| 100 | 0.001033 | 0.0025 | 0.0000 | 1.7511 | 11.0549 | None | **< 0.05** |

* retention s100/t0: **0.0036** — SAND-STRICT; outside the installed-fact band

## The bands (runtime-read from md5-bound committed records)

* installed-fact band: raw [1.0072, 1.0379] hugged by 0.05 -> **[0.9572, 1.0879]** (the dispatch's [0.957, 1.088], gated within 1e-3); legs: g1b W1 (locked root, wash 10902) 1.0072; g1bR W1_10907 (locked root) 1.0232; g1bR W1_10908 (locked root) 1.0335; g1c W1 (the ball's own leg; the subject lineage's sibling) 1.0379
* sand class: VACANT 0.0028; HOST-OCCUPIED 0.0049; FRESH 0.0017; VACANT-E25 0.0032 — strict [0.0017, 0.0049]; SAND-CLASS operationalized retention <= 0.01 AND read < 0.05
* the ball's committed retention (the reference): g1c W1 (the ball's own +100 retention) = 1.0379; e335's bare root: t0 0.7448 -> s1 0.0087 (retention 0.0003)

## The match gates

* G_DRAWS: both arms consumed bit-identical aj/rj sequences (100 steps)
* G_INPUTS: per-step input-batch md5s bit-identical
* G_TWIN_T0: both arms' t0 == the committed subject read 0.285851389169693
* G_PIN: the committed arm's max raw displacement 1.6543 <= R + 1.5 = 2.2

## Provenance
* birth commit: b329eb4 (bars + P-e338a, pushed BEFORE compute); run-start head: 428fb51 (the smoke R-gate repair); [metadata repairs, disclosed: status DONE-prefix + birth pinning]
* every artifact md5-bound (see metrics.gates: subject, parents, era rigs); timestamps UTC only; envelope polls to runs/_envelope_log.jsonl tagged e338:*
