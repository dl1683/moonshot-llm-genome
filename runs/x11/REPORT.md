# x11 — THE CONSOLIDATION LEDGER (MIXED)

**The registered question (P-x11a, T256):** is the expression edge a
CONSOLIDATION-RATE threshold? The consolidation ratio := final in-room
write mass ||P_room dW|| (inst_resume s400 checkpoints, own-room fp64
projection) / cumulative applied in-room displacement (the integrated
per-step lr x ||g'|| channel of the committed install ledgers) — how much
of what was pushed STAYED, per rung.

## The consolidation ledger (7 committed rungs)

| rung | k | role | post g0 (cited) | num ||P dW|| | cum. applied (proxy) | RATIO | ratio (hold-left) | inside share |
|---|---|---|---|---|---|---|---|---|
| K1K | 1,000 | DEAD (primary below-edge) | 0.000435 | 6.0489 | 4.828876e-03 | **1252.66** | 1251.99 | 0.8082 |
| K1KM | 1,000 | dead, kept-matched lr x3.73 (co-report) | 0.001245 | 11.9596 | 1.898507e-02 | **629.95** | 627.26 | 0.6329 |
| K2K | 2,000 | first above edge | 0.026616 | 6.5213 | 6.824598e-03 | **955.56** | 955.31 | 0.8264 |
| K5K | 5,000 |  | 0.127096 | 7.3313 | 1.084706e-02 | **675.87** | 676.00 | 0.8546 |
| K10K | 10,000 |  | 0.264648 | 8.6663 | 1.584159e-02 | **547.06** | 547.38 | 0.8914 |
| K10KR | 10,000 | 10k replicate (co-report) | 0.209721 | 8.7018 | 1.581204e-02 | **550.33** | 550.72 | 0.8921 |
| K237K | 237,123 |  | 0.384364 | 14.7640 | 9.428808e-02 | **156.58** | 156.83 | 0.9614 |

**The edge read:** J = r(K2K)/r(K1K) = **0.7628** (hold-left 0.7630).

## Verdict: MIXED

Clause trace: cliff (J >= 3) = False; inverted (J < 1) =
True; flat (1 <= J < 3) = False;
MIXED clause (a) above-edge family spread 6.103 > 3 =
True; clause (b) r(K1KM) >
r(K2K) = False (r(K1KM) = 629.95);
clause (c) hold-left flips the primary word = False
(primary RATIO-INVERTED -> RATIO-INVERTED).

## Gates

- G_ENDPOINTS: all 15 touched files md5-gated against x10's committed
  table (which carries x6's original binds); runs/x10/metrics.json
  fresh-bound (2d1cec68a0e7d471fd991e3693913d01).
- G_ROOMS: all 7 room entries bit-verified vs fresh committed-seed
  reconstruction; K1KM room re-gated bit-equal to the committed K1K room.
- G_LEDGER + G_DOSE_REPRO: every ledger covers steps
  {1,10,...,400} with finite positive gpn; this cell's reconstruction of
  the per-step proxy reproduces e272's committed
  applied_dose_proxy_median to <= 1e-9 relative on all four e272 arms —
  the lr schedule + gpn channel are the parent's, exactly.
- G_DISPL (content bind, x10's precedent): every rung's fp64 inside_share
  matches the committed in_own_room norm ratio squared within 2e-3
  (max abs diff 9.48e-12).
- G_DISP_NORM: the four e272 numerators match the committed
  in_room_disp_norm within 5e-3 relative
  (max rel diff 6.24e-12).
- G_FLATBASIS: key order == TinyGPT parameters, no buffers, N=2,739,072.

## Disclosures

1. THE PROXY FORM: the denominator is the first-order applied
   displacement lr(s) x ||g'(s)|| summed over 400 steps — Adam's
   per-coordinate preconditioner and AdamW's decoupled weight decay are
   unmodeled (e272's own dose_control caveat). It is used as a consistent
   cross-rung currency; the RATIO's absolute scale is not interpretable,
   only its cross-rung structure is. Hold-left interpolation sensitivity
   co-reported (clause c).
2. gpn between the 41 committed ledger points is piecewise-LINEAR
   interpolated (primary); every ledger point is a committed fp64 value.
3. K10K's ledger is e264's committed resume-completed record (e261's s278
   cut completed by e264's pass 1; the journal-resume design) — it covers
   s1..s400 whole.
4. K1KM's install ran at lr x3.7305567687315575 (the kept-matched
   compensation; lr_scale asserted against e272's committed
   lr_scale_applied); its denominator uses that scale.
5. The numerator includes whatever re-growth inside the room the install
   produced (projection is orthogonal; the out-of-room drift is excluded
   by construction).
6. CPU-only: threads 4/4, no envelope-log writes, no GPU code path; all
   timestamps datetime.now(UTC).
