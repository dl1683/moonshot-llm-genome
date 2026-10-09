# X17 — THE N=2 DESK BUNDLE (three replication controls)

**Probe 1 (x15R) verdict: GAP** — the full-dose TAVIREN complement reads 0.0284082 (>= 0.01 but the READS condition unmet: frac>=0.05 = 0.18) — necessity replicates at the dead bar but the named READS bar does not fire; no claim wording changes without a new registered cell

**Probe 2 (e320R) verdict: SUPPRESSION-REPLICATES** — both fresh draws suppress at both doses (worst x0.510 < x1.0) — n=2/n=3 for the claim

**Probe 3 (x16R) verdict: NULL-REPLICATES** — the second random out-of-room draw's host boost +0.000008 < 5% of the trigger's +0.084783 (0.01% if computed) — n=2 for the null

## Headline numbers

* x15R: the TAVIREN write ||dW_tav|| = 8.9496290602 (0.9424 in-room — e311's committed literals reproduced at 1e-9); complement L2 2.993528 (11.19% of the write's energy); scaled x2.989660 to the full dose (1e-12); reads p(T) 0.0284082 at the 60 g0 contexts (x15's committed DEAD read on the K10K write: 0.0007308); full-write control 0.2858513593673706 vs committed 0.2858513891696930 (|d| 3.0e-08); riders top-1 p 0.6618 / entropy 1.3591 (full-write control: 0.4488 / 1.4651).
* e320R: drawA (seed 17001) D3-r1000scale x0.217 D4-e314match x0.134; drawB (seed 17002) D3-r1000scale x0.510 D4-e314match x0.403; e320's committed seed-32001 curve: D1 x0.942 D2 x0.809 D3 x0.597 D4 x0.500.
* x16R: ctrl-2 (seed 17003) host boost +0.000008 = 0.01% of the trigger's in-session boost +0.084783 (committed +0.084783); the anchor reproduced at |d| 0.0e+00 with arm-A flat md5 f84fd2647fa5; ctrl-1 replayed bit-exact (bytes fa9d1d70, flat ecf2a210, boost +7.132e-05); ctrl-2 off-target battery: stability 1.0000, flip-or-squeeze 0.0000, mean |dmargin| 0.00054 (trigger: 0.01170), Z-excess +0.00008 (trigger: +0.01005); cos(ctrl2, trigger) -5.18e-04.

## Predictions scored (registered at birth, never shopped)

* P-x17a (DEAD-REPLICATES): **MISS**.
* P-x17b (SUPPRESSION-REPLICATES): **HIT**.
* P-x17c (NULL-REPLICATES): **HIT**.

## The gate ledger

| gate | what it binds | pass |
|---|---|---|
| G_NAMEFREE | the corpus carries no name (Z, TAVIREN) | True |
| G_SPLICE | the battery reconstruction (19+41) | True |
| G_BATTERY | the g0/gm12 battery geometry | True |
| G_PARENTS | 9 parent artifacts md5-bound + 18 committed references cross-checked from artifacts | True |
| G_FLATBASIS | the flat <-> state-dict basis | True |
| G_BASE | e001 fact-free | True |
| G_ROOM | the K10K room D/S bit-bound vs e264_rooms.pt | True |
| G_HOSTLOAD | the host loads bit-exact (3-way) | True |
| G_TAVOBJ | dW_tav reproduces e311's norm + in-room literals | True |
| G_TAVPRIOR | the base's p(T) prior <= 0.05 | True |
| G_TAVFULL | the full TAVIREN write reproduces the committed read | True |
| G_TAVDOSE | the complement scaled to the full dose (1e-12) | True |
| G_TAVINROOM | the scaled complement stays out-of-room | True |
| G_SHAM2 | both fresh in-room draws' construction | True |
| G_TRIGBIND | the committed trigger (md5/norm/in-room) | True |
| G_ANCHOR2 | the +32% anchor (state md5 + read) | True |
| G_CTRL1REPLAY | x16's first control replays bit-exact | True |
| G_CTRL2 | the second draw (in-room ~0, matched norm) | True |
| G_BATT3 | the off-target battery sanity (x16's form) | True |
| G_CE2 | ce_r sanity at every probed state | True |

**20/20 gates PASS.**

## Catches + disclosures

* THE GAP, READ HONESTLY: the full-dose TAVIREN complement reads
  0.0284 — dead by e310's original 0.05 floor and 10.1x below the full
  write's 0.2859, but NOT under this cell's frozen 0.01 dead bar
  (x15's K10K complement read 0.00073 — the null is WRITE-DEPENDENT by
  ~39x). Per the frozen GAP clause: the necessity claim survives
  WEAKENED; no laws-v3 wording changes without a new registered cell.
  The organism-health riders say the weakened read is NOT trivially
  guaranteed by collapse: battery top-1 confidence ROSE (0.6618 vs the
  full write's 0.4488; entropy 1.359 vs 1.465), though general CE
  carries a real cost (ce_r 1.9954 vs the full write's 1.5853 — +0.41
  nats, inside the [1.0, 2.2] sanity band; the 100%-dose complement
  carries 8.9x the write's own out-of-room energy, and e320's
  displaced states sat at ~1.61-1.63). p(Z) co-report 0.0003 — no host
  name leak; gm12 0.0285 == g0 0.0284, flat across geometries.
* THE DISK CATCH: the dispatch letter said runs/e311/e311_hijack_vectors.pt carries dW_tav — on disk it carries {trigger, seed, meta} only. The write's true carrier is runs/checkpoints/e311_TAVINST_post.pt (e311's G_FRESH.checkpoint); both md5-bound; dW_tav reconstructed and gated against e311's committed norm/in-room literals at 1e-9.
* Probe 2 runs e320's bars' state only (the saturated e288 co-column is not re-run — its ceiling-stamped verdict was structurally vacuous and plays no role in the n-count).
* Probe 3's battery reuses x16's registered seed 31603 (the same 300 windows — the instrument, not a draw); the trigger's off-target table is re-run in-session and cross-checked against x16's committed record.
* Every committed reference co-reported here was READ from its md5-bound metrics.json at runtime and cross-checked against the frozen literals (never retyped).

## Provenance

* birth commit: bf880324958941236a5a1da368de37aa880d2bd7; final head: bf880324958941236a5a1da368de37aa880d2bd7
* CPU only (torch threads 4, pocketfft workers 4); no GPU touched; no other runs/ written.
* No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat folds).
