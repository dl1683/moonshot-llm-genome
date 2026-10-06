# E278 — THE THREE-NULL COLLISION CELL — REPORT

**Verdict: UNDERTOW-REGARDLESS** (the frozen bars, letter-exact: ALL THREE
concurrent arms die < 0.5x serial, INCLUDING the missile) — with the
displacement reads naming the undertow's channel and correcting its
mechanism story (below). 15/15 hard gates PASS. Run 3190 s; thermal max
78.0 C over 2834 polls, 0 violations of the 84 C line (the 78 C margin
ended 3 isotope bursts early, by design).

## The question (verbatim, frozen at birth)

e278 asks WHAT THE COLLISION NEEDS: semantics (the corpus's content),
mere energy (any gradient stream), or mere overlap (spatial presence in
the room)?

## The arms (one rig: 1:1 interleave, ONE shared AdamW, k=10k room —
e272's committed K10KR (seeds 27215/27216), bit-gated D/S vs
e272_rooms.pt; install streams bit-identical across arms, seed 24314;
SEMANTIC and MISSILE share the ONE registered corpus generator 27801 —
the missile's ONLY delta vs SEMANTIC is the projection)

| arm | construction | post g0 (s400) | ratio vs same-session serial | peak traj g0 | verdict |
|---|---|---|---|---|---|
| SERIAL | the committed rung alone, fresh re-run | **0.209721** | 1.0000x | 0.3153 | expresses (anchored: L2 4.7e-5 vs e272's committed vehicle; \|d post g0\| 3.6e-7) |
| SEMANTIC | the control twin (e268/e273's form, corpus seed 27801) | **0.000371** | **0.0018x** | 0.0020 | DIES |
| ISOTOPE | max-entropy: 48x256 uniform tokens (seed 27811), CE pinned at chance | **0.014742** | **0.0703x** | 0.0163 | DIES (40x softer than SEMANTIC) |
| MISSILE | SEMANTIC's twin, corpus gradient projected ENTIRELY orthogonal to the room | **0.000545** | **0.0026x** | 0.0008 | DIES (essentially as hard as SEMANTIC) |

The e272-cited ratios are identical to 4 decimals (the serial twin is
bit-faithful). Corpus CE medians: SEMANTIC 0.886, ISOTOPE 4.181 (chance
ln 65 = 4.174; second-half median 4.1813 — the honesty gate PASSED with
the isotope sitting 0.007 above chance), MISSILE 0.993.

## The gates (all PASS)

- Parents md5-bound: e272 (RANK-WRITES-THE-CURVE, K10KR post 0.20972091),
  e268 (DYNAMICAL-CARRIER, 0.2646 vs 0.00004), e273 (the two-body record:
  shared 0.0053 / separate 0.0016, TTB1 fires), the room file
  (e272_rooms.pt, 066944...), the K10KR install vehicle (md5/step/size).
- G_ROOM10KR: the room rebuilt from seeds 27215/27216 is bit-identical
  (D and S exact) to e272's committed K10KR room; certified (idem
  5.6e-16, kept2 0.003619 vs 0.003651 at the 10-sigma bar).
- G_ISOTOPE_CE: second-half corpus-CE median 4.1813 in
  [3.974, 4.924] — the max-entropy stream really was max-entropy.
- **G_MISSILE_ORTH: max ||P_room g_perp||/||g_perp|| = 4.6e-17 over ALL
  400 corpus steps** (bar 1e-6; the projection is exact to machine
  precision — the dispatch's smoke centerpiece held at full scale).
  ||g_perp||/||g|| = 0.9982 (first and mean) — the degeneracy clause
  never triggered; the disclosed step-size cost of the projection is
  ~0.2%.
- G_SERIAL_ANCHOR (non-halting): install L2 4.72e-5 (bar 5e-3); post g0
  |d| 3.6e-7. PASS.
- Draw integrity: first-batch install CE identical across all four arms.

## The mechanism reads (measured, never nominal)

**THE DECISIVE DATUM — the missile's realized displacement.** The
per-milestone corpus-displacement ledger (fp64 projections of the summed
corpus-step parameter deltas):

| milestone (interval end) | SEMANTIC in-room frac | ISOTOPE in-room frac | MISSILE in-room frac |
|---|---|---|---|
| s100 | 0.573 | 0.792 | **0.586** |
| s200 | 0.572 | 0.857 | **0.599** |
| s300 | 0.508 | 0.852 | **0.537** |
| s400 | 0.461 | 0.847 | **0.481** |
| cumulative | 0.649 | 0.878 | 0.673 |

The missile's gradient NEVER stepped in the room (4.6e-17); its
displacement walked 48-60% in-room — statistically indistinguishable
from the unprojected SEMANTIC stream at every milestone. **Adam's
per-coordinate normalization re-aims orthogonal gradients into the
room** (e273's sign-step law operating on the orthogonal complement:
the optimizer's sign-like step is room-compatible regardless of the
gradient's room content). The gradient-level steering #007 proposed is
undone by the optimizer before it reaches the parameters.

**The isotope's death mode is different: total flattening, not a
targeted kill.** Its cumulative corpus displacement norm is 487
(SEMANTIC: 38; the interval norms run 27-79 vs 5-8), 85% in-room, and
the model it leaves behind is a uniform predictor: CE_R 4.198, install
CE 4.218, and its post g0 0.01474 sits at the uniform floor (1/65 =
0.01538; g0_argmax 0.0). The isotope's "40x softer kill" is not a
partially spared write — it is a sandblasted model whose battery reads
chance. SEMANTIC and MISSILE leave healthy text predictors (CE_R 1.62)
with the write specifically erased.

**The kill needs nothing but traffic through the shared optimizer.**
Semantic content modulates the depth (semantic 0.0018x vs isotope
0.070x) and the death mode (surgical vs flattening), but: (i) a
zero-semantics stream at chance CE kills inside the bar (P-C-x's
two-body/Energy prediction CONFIRMED in direction — the isotope kills
too; the shared-v account's specific "feeds the shared denominator"
mechanism remains dead per e273); (ii) removing the corpus gradient's
in-room component entirely changes the endpoint by 1.5x (0.0026x vs
0.0018x) — nothing.

## The adjudication (the registered composite, frozen at birth)

All three arms < 0.5x serial, serial expresses, no edge-zone reads
(nearest: ISOTOPE 0.070x vs the [0.375, 0.625] edge band) →
**UNDERTOW-REGARDLESS** fires on its letter-exact signature. The
ENERGY-CO-FIRE sub-clause does NOT fire under its registered
quantitative form ("as hard as": ratio_I <= 2 x ratio_S + 0.01 →
0.0703 <= 0.0136 is FALSE) — the isotope kills, but not as hard as the
semantic corpus (partial kinetic content; and per the CE_R reads, a
different death mode altogether).

**The clause's mechanism story, corrected by the measured reads (the
honesty the letter owes):** the verdict's letter says "even a corpus
that never steps in the room kills the write". Measured, the corpus's
GRADIENT never stepped in the room; its realized DISPLACEMENT did
(48-60% in-room, because the optimizer re-aims it). So the undertow is
real — no gradient-level construction on this family's standard
optimizer escapes the kill — but its channel is the optimizer's own
normalization, not a room-independent drag through parameter space. The
two-body story's second clause: **the corpus cannot be steered; the
room cannot be dodged at the gradient level; the collision is written
into the optimizer's re-aiming of every stream it carries.**

## Provenance

- Script: lab/e278_three_null.py (birth commit 4e86c6c, BEFORE any
  compute; smoke catches + fixes committed at 0698cbf; bars/gates/arms
  untouched by either).
- Machinery: e261's drivers/projector/thermal envelope PORTED WHOLE BY
  IMPORT (unmodified); this cell's one new driver is
  chunked_install_threenull + orthogonalize_grads.
- NO CONS (registered deviation: the bars are WRITE-read-only;
  T259/e281 — the rehearsal lane carries zero write information). The
  four post-install states are checkpointed (runs/checkpoints/
  e278_<ARM>_post.pt) for any later cons.
- Envelope: bursts <= 175 s, per-step polls both streams (2834 polls),
  40 s cooldowns, max 78.0 C, 0 violations; polls tagged e278:* in
  runs/_envelope_log.jsonl.
- Outputs: runs/e278/{metrics.json, e278_three_null.png, REPORT.md,
  run.log (gitignored)}.

## Standing caveats (carried verbatim)

n=1 per arm, one lineage, one session (the g-series standing lottery
note); the arms' DIFFERENCE is the registered object; nothing
guaranteed. No NOTES/THINKING/QUEUE/STATE edits (the coordinator
folds).
