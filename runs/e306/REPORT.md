# e306 DESK HALF — THE MINIMUM BEARER'S TRUNCATION INVENTORY (READS-CLIFF)

**The object:** the committed 10k quiet write `e261_K10K_inst_resume.pt`
(s400, md5-bound) minus the fresh root `e001.pt` (md5-bound) — x6's
registered object, and THE e290 fact write itself: ||dW|| = 9.1788
(bit-matches e290's committed write norm; the coupling constant was
measured on this very write). Truncated per x6's conventions — per-2D-matrix
SVD (fp64), top-min(r, rank_j), 1D deltas carried verbatim (SVD undefined
there; x6 discloses, never drops), rungs {1000, 100, 10, 7} verbatim from
the dispatch. CPU-only desk cell; no training, no GPU.

**Instrument:** e261's g0 battery rebuilt bit-exactly (mix FLORIZEL 19 /
ELIZABETH 41, shapes (60,130)/(60,118), corpus ZEPH-free — the committed
gate literals); this desk's probe of the FULL write returns
0.26464763283729553 vs e264's committed
0.26464763283729553 (|d| = 0.0e+00; G_FULLREAD PASS).
The base e001 root reads 1.34e-05 at t0.

## (a)+(b) The truncation inventory + the budgets each rung implies

| rank | \|\|dW_r\|\| | mass vs full | energy vs full | residual | 2D bearer dims | in-room (own) | passive-kill budget (abs L2, e290 bracket) |
|---|---|---|---|---|---|---|---|
| 1000 | 9.1788 | 100.000% | 100.000% | 0.0000 | 4930 | 89.14% | 0.00652 - 0.02968 |
| 100 | 8.0212 | 87.388% | 76.367% | 4.4622 | 2630 | 66.62% | 0.00570 - 0.02593 |
| 10 | 3.5246 | 38.399% | 14.745% | 8.4752 | 270 | 9.01% | 0.00250 - 0.01140 |
| 7 | 3.1094 | 33.875% | 11.475% | 8.6361 | 189 | 6.14% | 0.00221 - 0.01005 |

The dispatch's question — "a rank-7 write is ~0.1% of full rank's mass?"
— answers: **33.88% of the mass**
(11.48% of the energy). The 1D
passthrough carries 17.55% of the mass
(x6's 3.08% energy share). The drift budget scales with the write's own
norm (T269): the rank-7 bearer's absolute safe-drift window is
[0.00221, 0.01005] L2
vs the full write's [0.00652, 0.02968]
— the thinner the write, the proportionally smaller its tolerance.

## (c) THE INJECTION READINESS — the t0 read per rank (the desk datum)

| rank | t0 g0 mean p(Z) | vs floor 0.05 | t0 gm12 | frac >= 0.5 | frac argmax | ckpt roundtrip |
|---|---|---|---|---|---|---|
| 1000 | 0.264648 | PASS | 0.105255 | 0.133 | 0.200 | True |
| 100 | 0.092544 | PASS | 0.033403 | 0.000 | 0.000 | True |
| 10 | 0.000054 | FAIL | 0.000056 | 0.000 | 0.000 | True |
| 7 | 0.000043 | FAIL | 0.000046 | 0.000 | 0.000 | True |

Injectable checkpoints written + probe-verified from disk at every rung:
`runs/e306/e306_trunc_r{1000,100,10,7}.pt` (theta_base + truncated_delta
fp32 + the per-key delta). The 1D-ONLY diagnostic (2D zeroed, 1D verbatim;
co-report only, never a bar): t0 g0 = 0.000019.

## Verdict: READS-CLIFF

the t0 read collapses below rank 10 (< 0.05); the injection floor bracket is (10, 100] — the maintenance ladder starts above it. Per-rank t0 g0 reads verbatim:
{"1000": 0.26464763283729553, "100": 0.09254436194896698, "10": 5.3935862524667755e-05, "7": 4.340019222581759e-05}. Committed context (different objects — writes FORMED
in small rooms, not truncations of this write): e261's K1K write read
0.000435; e246's rank-10 ALIGNED read 2.86e-5.

Disclosures: rung 1000 is the full write BIT-EXACTLY (the largest 2D
matrix rank is 192 — the rung is the dispatch's verbatim sanity rung, and
its probe doubles as G_FULLREAD); rung 100 passes wte/lm_head (rank 65)
through untouched; the room is the committed K10K room (e264_rooms.pt,
seeds 26113/26114, D/S bit-verified vs fresh construction); the budget
bracket is e290's committed realized-drift fractions
[0.0007099905, 0.0032331213] of the write's norm,
applied same-object; n=1 per rung (one lineage, one session — the g-series
standing lottery note carried verbatim).
