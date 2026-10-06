# X14 — THE TRANSPORT INTERVENTION — REPORT

**Verdict: MIXED/INCONCLUSIVE** (the frozen bars, letter-exact). mass intact + no resurrection TO THE BAR: the in-room mass is NOT eroded below half (ratio 0.8435 >= 50%) while the orthogonal-subtraction read sits below 0.1 — a PARTIAL resurrection (arm A 0.039154 in [0.01, 0.1): 4328x above the dead state, 15% of baseline — real signal against pure overwrite, short of the transport bar); THE COUPLING CAVEAT FIRED as registered (the subtraction's linear-superposition null can be masked by LN/softmax coupling; a sub-bar read does NOT prove overwrite — the necessary-not-sufficient direction governs); the converse arm reads 5.86e-05 (dead — the out-of-room context ALONE, kept in arm B, suffices to hold the read down)

R67's named desk cell: e283's correlational dichotomy — TRANSPORT-LITERAL
(the write's in-room mass intact, killed by out-of-room context) vs
OVERWRITE (the in-room mass eroded) — both fit every committed number
(the critic's finding-3 arithmetic: the intact-write band [7.0, 10.3] vs
measured 7.31; the erosion degeneracy: the same 7.31 is consistent with
any surviving in-room fraction in [0.65, 1.03]). The states are on disk;
this cell intervenes.

## (a) The write-mass ledger (fp64, the fact's own K10K room)

| t | in-room mass P(theta-base) | out-of-room (I-P)(theta-base) | source |
|---|---|---|---|
| t0 | **8.6663** | 3.0243 | COMPUTED (the fact ckpt) |
| 100 | 8.2762 | 6.5704 | cited-derived (e283's committed ledger; state not saved) |
| 200 | 7.8240 | 10.5033 | cited-derived |
| 300 | 7.5238 | 12.8347 | cited-derived |
| 400 | **7.3096** | **14.3231** | COMPUTED (the post ckpt) |

The mass ratio t400/t0 = **0.8435** — ABOVE the
50% erosion bar. The out-of-room residual grew 3.02 -> 14.32
(the transport channel). Diagnostics (context, not bars): the surviving
mass's direction vs the write's own in-room direction cos =
0.9931; the erosion degeneracy [0.653, 1.034]
reconstructs R67's finding 3 exactly.

## (b) The subtraction intervention (fp64; arms materialized fp32)

drift decomposition: |delta| 14.4539 = in-room 1.6487 + out-of-room 14.3596
(orthogonal by construction; identity closure |theta400 - delta - theta0|
= 0.0e+00 in fp64).

| arm | construction | g0 read | prediction it answers |
|---|---|---|---|
| the fact (t0) | the loaded write | 0.26464763 | the resurrection target (baseline 0.2646) |
| t400 post | the dead state | 9.046e-06 | the death being diagnosed |
| **ARM A (primary)** | theta400 - (I-P)(drift) | **3.915367e-02** | TRANSPORT-LITERAL: >= 0.1; OVERWRITTEN: < 0.01 |
| ARM B (control) | theta400 - P(drift) | 5.864364e-05 | the converse (co-interpreted, no bar) |
| ARM ID (closure) | theta400 - delta | 0.26464763 | must equal the fact's read (abs diff 0.0e+00) |

## (c) The honesty block — the caveat's status

**Coupling caveat fired: True.** The subtraction
assumes component-wise additivity (a linear-superposition null); LN and
softmax couple out-of-room parameters into the read path, so a
transport-literal state can fail to resurrect. The test is
NECESSARY-not-SUFFICIENT: resurrection => transport; no resurrection =>
inconclusive between overwrite and coupling, NOT proof of overwrite.
This run: the mass is intact (ratio 0.8435) and the read did not resurrect TO THE BAR (0.039154 < 0.1) — it PARTIALLY resurrected (4328x above the dead state, 15% of baseline) — real signal against pure overwrite, short of the transport bar; the registered inconclusive branch holds the result by construction, not by consolation.

## The gates (all 12 PASS)

Parents md5/literal-bound (e264/e268/e278 metrics, the fact ckpt
md5/size/step + flat-md5, the room file, the span, the v-map); the base
fact-free; the root read-bound; the room certified (idem/kept2/span) and
BIT-BOUND to e264_rooms.pt; the fact behaviorally bound (g0 abs diff
0.0e+00); the post state
value-bound behaviorally + geometrically (drift norm abs diff
6.3e-12; read g0 abs diff
0.0e+00).

## Disclosures

- THE MILESTONE GRID: only t0 + t400 were checkpointed by e283; t100-300
  are cited-derived products of its committed fp64 ledger
  (cross-checked at t400: abs diff 0.0e+00); the t400 intervention
  answers the primary (the dispatch's own instruction).
- CPU-ONLY desk cell: no training, no stream, no steps; reads + fp64
  projections; torch threads 4, pocketfft workers 2; NO envelope-log
  writes.
- The post state has no committed file-md5 (e283 recorded none) — bound
  by VALUE (behavioral + geometric), the stronger bind; its md5 is
  recorded here (a98b638cbd52658e63d706cd112b26fb).
- n=1 lineage, one session; nothing guaranteed.

## Provenance

Birth commit 8620c0babc19ae05af3436064ec7ce8b5a35aeff (bars + conventions, BEFORE any compute); full run
this commit. Machinery: e261's SRCT/LadderRooms ported whole by import
(the committed file untouched). No NOTES/THINKING/QUEUE/STATE edits.
