# E283 — THE ESTABLISHED-FACT COLLISION TEST — REPORT

**Verdict: ESTABLISHED-DIES** (the frozen bars, letter-exact: the
established write dies at **0.0000342x** of its baseline — far below the
0.5x bar) — and it dies **DEEPER than the forming write did** (e268's
forming-under-fire death: 0.000151x; the established write lands 4.4x
deeper, with NOTHING writing back). 15/15 hard gates PASS. Run 174.7 s;
thermal max 75.0 C over 402 per-step polls, 0 violations of the 84 C
line; one 97.7 s burst (inside the 175 s cap).

## The question (verbatim, frozen at birth)

Does an ALREADY-ESTABLISHED fact collide as hard as a forming one?
Every concurrent arm so far FORMED under fire (the install ran while
the corpus interleaved). T244's registered discriminator said "corpus
CONCURRENTLY vs AFTER"; the AFTER arm never ran. Tonight it ran.

## The arms (the established fact = the committed quiet-formed 10k
write, loaded BIT-EXACT: e261_K10K_inst_resume.pt, md5/size/step-bound,
flat-md5'd ebebb447..., and behaviorally gated — the loaded read
returned the committed post g0 0.26464763283729553 with |d| EXACTLY 0.0,
gm12 |d| 0.0; the write stood 0.9442 in-room at t=0, matching e268's
SERIAL post read 0.944)

| arm | construction | post g0 (t=400) | survival ratio | trajectory (x baseline) |
|---|---|---|---|---|
| ESTABLISHED-SERIAL-CONTROL | the same loaded fact, the stream PAUSED — NO steps at all (the pure storage-decay control, the frozen pick) | **0.26464763** | **1.0000x** | flat at x1.0000 (max read drift EXACTLY 0.0) |
| ESTABLISHED-CONCURRENT | the loaded formed fact + the family's standard corpus stream ALONE: 400 corpus steps (16 anchors + 32 random windows, full-window CE, the matching cadence lr 1e-3 x cosine_lr(t-1,1000), clip 1.0 -> opt.step FREE) through ONE fresh AdamW (0.9,0.95) wd 0.1 — NO install steps | **0.00000905** | **0.0000342x** | t100 x0.00067; t200 x0.00038; t300 x0.00022; t400 x0.000034 — monotone death |
| *(cited)* e268's FORMING write under the same interleave | the install + the corpus stream, 1:1 | 0.0000400 | 0.000151x | the SPARED bar's "0.0002x" cite (hard-bound) |
| *(cited)* e271's 237k forming write | the widest room under fire | 0.0349985 | 0.091x | the retention barrier (hard-bound) |

The control null HOLDS exactly (G_CONTROL: max |g0 drift| 0.0 over the
repeated milestone reads) — time, storage and the read pipeline are
nulled; everything the concurrent arm shows is the corpus traffic's
doing.

## The gates (all 15 PASS)

- Parents md5-bound: e264 (SHARP-THRESHOLD — the loaded fact's own
  record), e268 (DYNAMICAL-CARRIER — the forming reference), e271
  (MIXED — the 237k retention datum), e278 (UNDERTOW-REGARDLESS — the
  roach-motel record), the fact ckpt (md5 0f6dc1cf..., size, step 400,
  traj [1,100,200,300,400], ledger max 400), e264_rooms.pt, the span,
  the v-map.
- **G_FACTLOAD**: the established fact loads bit-exact — the
  behavioral read |d post g0| = 0.0 and |d gm12| = 0.0 vs the committed
  literals; flat-md5 recorded; the write's standing displacement is
  9.179 in norm, 0.9442 in its own room (the t=0 reference for the
  drift ledger).
- G_ROOMK10K: the room rebuilt from seeds 26113/26114 is bit-identical
  (D and S exact) to e264's committed K10K room — the fact's OWN
  writing room; certified (idem 5.7e-16, kept2 0.003607 vs 0.003651 at
  the 10-sigma bar).
- G_CORPUSGEN: the fresh corpus stream registered (seed 28301 — the
  family's per-cell rule; first draws logged from a scratch generator).
- G_CONTROL: the storage-decay null — exact.

## The drift ledger (the roach-motel continuation read)

Per milestone (fp64): the state's drift from the loaded fact
(theta_t - theta_fact) and the write's REMAINING displacement from
base (theta_t - base), with in-room shares vs the fact's own K10K room
(volume overlap sqrt(k/N) = 0.0604):

| t | drift norm | drift in-room | remaining-from-base in-room | corpus interval v (norm / in-room) |
|---|---|---|---|---|
| t0 | 0 | — | **0.9442** | — |
| 100 | 6.053 | 0.0876 | 0.7832 | 6.053 / 0.0876 |
| 200 | 10.394 | 0.1018 | 0.5974 | 7.625 / 0.0858 |
| 300 | 12.881 | 0.1091 | 0.5057 | 6.997 / 0.0752 |
| 400 | **14.454** | 0.1141 | **0.4546** | 6.329 / 0.0699 |

Three facts, verbatim:

1. **THE CORPUS STREAM ALONE MOVES THE STATE FARTHER THAN THE WRITE'S
   ENTIRE DISPLACEMENT** (drift 14.454 at t400 vs the write's own
   9.179) — unprojected free traffic at the family's cadence is a
   bigger object in parameter space than the fact it erases.
2. **THE WRITE'S ROOM OCCUPANCY IS SCRUBBED BELOW HALF** (0.944 ->
   0.455, monotone): the state is dragged OUT of the room the write
   lives in.
3. **THE ROACH MOTEL DOES NOT GENERALIZE TO THIS REGIME — AND THE KILL
   DOES NOT NEED IT.** The raw corpus gradient sits EXACTLY at chance
   in-room (median ||P g||/||g|| = 0.0604 = sqrt(k/N)); the realized
   displacement walks only 0.070-0.088 in-room per interval (cumulative
   0.114) — a mild 1.2-1.5x re-aiming, NOT e278's forming-phase
   0.48-0.60 funnel. The established write dies anyway: the undertow's
   lethal channel here is TOTAL FREE-STREAM DISPLACEMENT, not in-room
   funneling.

## The clause (as adjudicated)

The established write also dies (ratio_400 0.0000342x < 50%x) — the
collision is content-agnostic about age; retention and formation die
together; the 237k retention barrier (e271) generalizes down to 10k
for established facts. And sharper than the letter: the established
write dies **deeper than the forming one** (0.0000342x vs 0.000151x,
4.4x) — under fire SOMETHING was still writing (the install steps);
after establishment nothing defends the write. The registered
expectation under the roach-motel/two-body account (ESTABLISHED-DIES)
landed; the mechanism story it came with did not — the drift ledger
corrects it (above).

## Convention disclosures (frozen before compute)

- The phase counts CORPUS steps only (400; no install — the family's
  install+corpus accounting explicitly set aside); milestones at
  t=100/200/300/400 corpus steps; the matching cadence is the corpus
  steps' own paired-lr schedule cosine_lr(t-1, 1000).
- The control is the pure storage-decay null (NO steps at all — the
  dispatch's recommended pick; the zero-gradient alternative rejected:
  it would add optimizer noise that is not storage decay), read at the
  same milestone cadence; corpus exposure 0 vs 400 is the intervention.
- The optimizer is the rig's, FRESH at phase start (AdamW 0.9/0.95 wd
  0.1): its decoupled wd shrinks weights ~3.4% over the cosine — a
  mostly in-room-direction shrink, disclosed as part of the
  intervention's body (the same channel acted on e268's forming arm);
  it cannot alone explain a 29,000x collapse (corpus CE fell 0.88 ->
  0.77 while CE_R rose 1.592 -> 1.665: genuine corpus learning, no
  numeric blowup; displacement norms are sane throughout).
- NO CONS (T259/e281: the rehearsal lane carries zero write
  information; the post-phase state is checkpointed,
  e283_ESTABLISHED-CONCURRENT_post.pt).

## Smoke catches

None — the shakedown (runs/e283_smoke/, gitignored by family rule) ran
clean end-to-end; its centerpieces verified: the no-install accounting
(corpus-step milestones, no install machinery invoked), G_FACTLOAD's
exact bind, G_CONTROL's exact null. Smoke-scale reads disclosed in the
script's deviations (the k=512 room captures only 0.2167 of the
10k-written displacement — S_512 is a subset of S_10k; the funnel is a
k=10k-scale question).

## Provenance

Birth commit 3ce875f (bars + conventions, BEFORE any compute); smoke
pass 3f6b7b1; full run this commit. Machinery: e261's LadderRooms/
SRCT/thermal envelope ported whole by import; the one new driver is
chunked_corpus_phase (the corpus-only phase + the drift ledger). n=1
per arm, one lineage, one session (the standing lottery caveat);
nothing guaranteed.
