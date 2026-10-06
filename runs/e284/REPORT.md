# E284 — THE SEPARATE-BUFFER MISSILE — MOMENTUM-OWNED

**Run:** 2026-10-06 (session-local ~52.5 min wall; datetime.now(UTC) stamps in
metrics.json) · **Script:** `lab/e284_buffer_missile.py` (birth commit
`f0ca112`, BEFORE any compute; smoke commit `e78058b`) · **Vehicle:** e272's
committed K10KR room (k=10,000, seeds 27215/27216), bit-bound to
`e272_rooms.pt` (D/S exact) · **Optimizer:** SGD(momentum 0.9, wd 0) at
LR_STABLE = 0.01 x 21.7385748014537 = 0.21738574801453703 x cosine — e280's
committed lr class, re-derived at runtime from the md5-bound
`runs/e273/lr_calibration.json`.

## THE VERDICT (the frozen bars, verbatim adjudication)

**MOMENTUM-OWNED** — "the separate-buffer arm's in-room share < 0.20 (vs the
shared twin's ~0.45+) — the re-aimer is buffer-sharing; THE MOTEL IS A
MOMENTUM-SHARING ARCHITECTURE; orthogonality is preservable by buffer
separation."

| read (primary = in-room share of the realized corpus displacement, median t100-400) | SEP (two buffers) | SHA (one shared) | e280 committed | e278 AdamW band |
|---|---|---|---|---|
| **primary median** | **2.32e-07** | **0.4503** | 0.4553 | 0.4814-0.5988 |
| interval fracs (t100/200/300/400) | 2.5e-7 / 1.8e-7 / 1.8e-7 / 2.3e-7 | 0.4074 / 0.4835 / 0.4503 / 0.3958 | 0.4193 / 0.4926 / 0.4553 / 0.4152 | (t100-400 band) |
| cumulative (t400) | 1.8e-7 | 0.6167 | 0.6267 | 0.6727 |

The separate-buffer arm's corpus stream walked **2.3e-07 in-room** — six
orders of magnitude below the 0.20 bar, at the fp32 accumulation floor, never
rising (cumulative flat at ~1.8e-7). The shared twin — bit-identical draws,
bit-identical room, identical lrs, the ONE delta the buffer topology —
funneled at **0.4503**, replicating e280's committed 0.4553 within 1.1%
(fresh corpus seed 28401 vs e280's 28001; intervals within ~0.01-0.02). The
twin-precedent clause passes (0.4503 >= 0.35); the composite's
MOMENTUM-OWNED branch fires on its exact letter.

**The motel's ownership is settled at this optimizer: the re-aimer is the
SHARED MOMENTUM BUFFER.** Orthogonality is preservable by buffer separation —
exactly. The funnel's arithmetic is now directly visible in the ledgers: the
twin's shared buffer held only **9.4-11.7% in-room content** (the coherent
install mass; per-step magnitude ~kept 0.06), yet its corpus displacement
realized **40-48% in-room** — because the ~1.0-magnitude orthogonal corpus
content cancels across steps (random directions) while the in-room install
content accumulates coherently. The realized SHARE is a ratio effect: a
buffer ~10% in-room delivers a walk ~45% in-room. P-e284a CONFIRMED in full
(collapse to the fp floor AND twin replication).

## THE SECONDARY (never the primary): the write is NOT spared

| write read (post g0, s400) | value | vs cited serial (e280 S10K 0.21676) | vs e280 missile (0.000692) |
|---|---|---|---|
| SEP | 0.0014973 | 0.69% — **NOT SPARED** | 2.16x |
| SHA | 0.0008973 | 0.41% — **NOT SPARED** | 1.30x |

P-e284s HOLDS (e273's two-body account: the parameters still collide — an
exactly-orthogonal corpus stream still kills ~99.3% of the write). The
disclosed counter-possibility landed as a nuance, not a rescue: separation
roughly **doubled** the surviving write vs the same-session shared twin
(0.00150 vs 0.00090; cleaner buf_I = only coherent in-room install content)
— a softening, nowhere near sparing (0.5x bar = 0.1084).

## THE BUFFER-SEPARATION VERIFICATION (the smoke's centerpiece, full-run discipline)

- **Isolation (machine-checked EVERY step):** 800/800 bitwise snapshot
  comparisons (opt_I's buffers before/after every corpus opt_C.step() and
  symmetrically for opt_C around every install opt_I.step()) — **0
  violations**. The install buffer never received corpus content and vice
  versa.
- **Composition (per milestone, CPU fp64 projection):** buf_C in-room max
  **5.87e-09** (bar 1e-4 — 17,000x below; the linear-momentum fp floor as
  registered); buf_I in-room min **0.99999999999999** (bar 0.99 — the
  install buffer IS the in-room stream).
- **The twin's single buffer** (a read, never gated): 0.0547 -> 0.1171 ->
  0.1088 -> 0.1074 -> 0.0944 in-room — the designed mixture.

## GATES (all PASS; G_MISSILE_ORTH INSTANTIATED this time)

`G_NAMEFREE, G_SPLICE (19+41), G_BATTERY, G_ANCHOR, G_INSTMASK, G_PARENTS
(e278/e280/e273-lrcal/e272-rooms md5+value bound), G_BASE (fact-free
1.44e-5), G_ROOT (|d| 0.0), G_VMBIND, G_SPANBIND, G_PROJ (idem 5.6e-16,
kept2 0.003619 vs 0.003651, 10-sigma), G_ROOM10KR (D/S bit-equal),
G_LR_BIND (LR_STABLE runtime == frozen), G_MISSILE_ORTH (INSTANTIATED —
e280's bookkeeping omission corrected: max 4.57e-17 SEP / 4.59e-17 SHA over
ALL 400 corpus steps x 2 arms, bar 1e-6), G_BUFSEP (above).`

## DISCLOSURES

1. **Runtime exceeded the dispatch's estimate** (~10-15 GPU min): ~3.2s per
   install+corpus pair at k=10k (two fp64 SRCT projections + per-step
   isolation snapshots + per-step polls) -> ~41 min GPU-burst time across
   both arms, 8 bursts/arm, all capped at 175s. The envelope itself was
   honored throughout: max temp **76.0C**, **0** readings >= 84C, 40s
   cooldowns, no concurrent GPU jobs (1630 polls persisted, tagged
   `e284:SEP:`/`e284:SHA:` phases).
2. **Draw-integrity texture (non-halting, disclosed):** the first-batch
   install CE is bit-identical across arms (1.0701713562011719); the first
   corpus CE agrees to 1.2e-7 relative (0.98404235 vs 0.98404247) — GPU
   kernel-reduction texture across arms, far below any read; the corpus
   draws themselves are identical by construction (one generator seed,
   28401, drawn identically by both arms).
3. **The SEP write trajectory is noisy** (t100 0.00070 -> t200 0.00013 ->
   t400 0.00150) — at these magnitudes the battery read rides the free
   stream's transport; the post read is the registered form and is reported
   verbatim.
4. **n=1 per arm, one lineage, one session** (the g-series standing lottery
   caveat) — the arms' DIFFERENCE on bit-identical inputs is the registered
   object; nothing guaranteed.
5. NO CONS (T259/e281 — the landing read is a cons property); both post
   states checkpointed (`e284_SEP_post.pt`, `e284_SHA_post.pt`). NO serial
   arm (e280's committed S10K rung cited + hard-bound as the survival
   denominator). No NOTES/THINKING/QUEUE/STATE edits.

## WHAT THIS SETTLES

The roach motel's forming-regime re-aiming decomposes cleanly: **the room is
a momentum-SHARING architecture, not the landscape's own geometry** — an
optimizer that never mixes its streams' momentum (or none at all beyond each
stream's own history) applies orthogonal steps orthogonally, to the fp floor.
The e278/e280 funnels (AdamW 48-60%, SGD-M 45%) were both, at bottom, the
same mechanism viewed through different optimizers: shared state carrying
the install's coherent in-room mass into every other stream's realized walk.
The landscape-owned branch — the stranger world where the loss surface
itself bends applied steps into the room — is refuted at this instrument:
with buffers separated, nothing bends. (T261's established-regime result
stands unchanged: transport, not re-aiming, kills established writes.)

**Inputs:** `runs/e284/metrics.json` (COMPLETE), `e284_buffer_missile.png`,
this report. Birth `f0ca112` -> smoke `e78058b` -> results (this commit).
