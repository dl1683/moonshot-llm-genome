# E285 — THE SANCTUARY CELL — AIM-ONLY-KILLS

**Run:** 2026-10-06 (session-local ~21.1 min wall; datetime.now(UTC) stamps in
metrics.json) · **Script:** `lab/e285_sanctuary.py` (birth commit `6667dd0`,
BEFORE any compute; smoke catch commit `c073f40`) · **Vehicle:** the committed
QUIET-FORMED 10k fact (`e261_K10K_inst_resume.pt`, loaded bit-exact, G_FACTLOAD
|d|=0.0 both reads) in ITS OWN room — the committed K10K room (seeds
26113/26114), bit-bound to `e264_rooms.pt` (D/S exact) · **The corpus stream:**
seed 28501 (this cell's ONE fresh registered stream), bit-identical draws across
arms (draw-integrity check EXACT: the t1 corpus CE + clipped ||g|| equal across
arms — the protection stack is the arms' ONLY delta).

## THE VERDICT (the frozen bars, verbatim adjudication)

**AIM-ONLY-KILLS** — "< 0.5x with the budget held but (check the ledger) —
separation+projection insufficient alone; the transport channel claims the kill
despite the budget (disclose the leakage arithmetic)."

**The lab's first BUILD-LANE result is a refutation at the composed form: the
two-channel law's minimal composition did NOT preserve the write.** The
engineered protections worked EXACTLY as designed at their own gates — the
corpus stream never aimed in-room (machine-verified), the state never moved
past its displacement budget (triangle-guaranteed, machine-held) — and the
write's read died anyway, 330x down, while its parameter-space mass stayed
93.0% in-room and 93.0%-intact. The kill rode FUNCTION-SPACE COUPLING through
the OUT-OF-ROOM CONTEXT — x14's directional-transport reading, now measured
under a live budget.

| read (survival = post g0 / 0.26464763283729553) | SANCTUARY (composed build) | UNPROTECTED-TWIN (e283's form) | e283 committed |
|---|---|---|---|
| **post g0 at t400** | 0.00080253 (**x0.0030**) | 0.0000039 (x0.0000147) | 0.0000090 (x0.0000342) |
| milestones | x0.0051 / x0.0032 / x0.0027 / x0.0030 | (dead from t100: x0.0001) | x0.0007 → x0.0000 |
| drift from the fact (t400) | **1.845** (20.1% of the write's 9.18) | 14.432 (157%) | 14.454 |
| the write's remaining in-room mass | **0.9300** (intact; began 0.9442) | 0.4553 (halved) | 0.4546 |
| drift's in-room share | **1.23e-06** (the fp floor) | 0.1134 | 0.1141 |
| corpus CE median | 0.8584 (the budget's price) | 0.8270 | 0.8284 |

The sanctuary softens the kill **~206x vs this session's twin** (~89x vs e283's
committed arm) on bit-identical draws — the protection stack is REAL — but
0.0030x sits two orders below the 0.5x bar. The bars adjudicate what they
adjudicate: AIM-ONLY-KILLS.

## THE TWO CHANNELS' LEDGERS (both verifiably cut)

**CHANNEL 1 — COLLISION (state re-aiming): CUT, machine-exact.**
- G_ORTH: max ||P_room g_perp||/||g_perp|| = **4.56e-17** over ALL 400 steps
  (bar 1e-6; INSTANTIATED — checked every step).
- G_BUFSEP: **400/400 bitwise isolation checks, 0 violations**; opt_F (the
  fact side) EMPTY throughout — never created, never touched; buf_C's in-room
  fraction **3.4e-9 → 5.2e-9** across milestones (bar 1e-4; e284's SEP
  arithmetic replicating).
- The realized corpus displacement's in-room share (cumulative): **1.2e-6 -
  1.7e-6 at every milestone** — the applied stream is out-of-room to the fp
  floor, start to finish.
- The RAW corpus gradient's in-room fraction (the unprojected contrast read):
  median 0.0605 = chance (the volume overlap sqrt(k/N)=0.0604) — there was no
  in-room pressure to cut in the first place at this phase (e283's own
  finding); the projection cut it anyway, exactly.

**CHANNEL 2 — TRANSPORT (displacement): HELD, machine-exact.**
- BUDGET = 0.5 x ||fact - base|| = 0.5 x 9.1788 = **4.5894**.
- The equal-share per-step lr cap: **S = 4.58942 at t400 = 100.00000002% of
  budget** (within the disclosed 1e-5 fp slack; the cap bound 397/400 steps).
- Realized cumulative corpus displacement: **1.8445 = 40.2% of budget = 20.1%
  of the write's own norm** (the orthogonal steps cancel below the triangle
  bound — the guarantee held with room to spare).
- The lr price: applied median **0.00600** vs nominal schedule median 0.19458
  (x0.031 of the stable class); the corpus CE improved 0.93 → 0.8584 med while
  the twin's free stream reached 0.8270 (e283's: 0.7747 at t400) — preservation
  was bought at corpus learning's price, and it still did not buy the read.

## THE LEAKAGE ARITHMETIC (the bar's demanded disclosure)

Where the kill entered despite both cuts:

1. **The write's mass is INTACT.** Remaining-from-base in-room occupancy:
   0.9442 → 0.9300 over 400 steps (1.4pp rotated out; the twin's fell to
   0.4553). The write was not collided with, not eroded, not transported past
   itself.
2. **The state moved 1.845 in parameters the write does not live in** (the
   drift's in-room share 1.23e-6) — 20.1% of the write's norm, inside budget —
   and the read fell to 0.30% of baseline. Worse: at t100 the read was ALREADY
   x0.0051 (200x down) with the state moved only **0.661 = 7.2% of the write's
   norm**. The transport kill threshold for OUT-OF-ROOM displacement sits at
   ≲ 0.07x the write's norm — roughly an ORDER below the 0.5x budget, probably
   two (the collapse is near-total before the budget is 15% consumed).
3. **The mechanism is x14's, now under a live budget:** the orthogonal-
   subtraction arm there resurrected the dead read to 0.0392 (4328x) while the
   in-room subtraction left it dead — the read lives in the OUT-OF-ROOM
   CONTEXT around the write's mass, and an exactly-orthogonal stream moves
   exactly that context. LN/softmax coupling converts a 7% parameter-space
   orthogonal perturbation into a 200x function-space read change: PARAMETER-L2
   BUDGETING DOES NOT DENOMINATE THE KILL. The two-channel law's transport
   clause is correct in direction (transport kills; 20% of the write norm
   sufficed here) but its "exceeds the write's own norm" scale is a FREE-STREAM
   scale — for an orthogonal stream at this organism the kill scale is ~x0.07
   of the write norm, and a budget tight enough to hold the read (~x0.01-0.02)
   would leave the corpus stream effectively stopped (the lr already at x0.031
   of nominal; x0.003 more would be stillness — and stillness is the only
   preservation condition this lab has ever measured, T261's perfect storage
   null).

**The build lane's founding lesson: the two-channel law is NECESSARY-shaped
but NOT SUFFICIENT as parameter-space engineering.** Cut both parameter-space
channels and the function dies anyway: the third quantity is the read's own
sensitivity — preservation must be denominated in FUNCTION space (a read-
anchored constraint, a frozen room, or a function-space budget), or bought
with stillness.

## GATES + ENVELOPE

- **18/18 hard gates PASS**: {G_NAMEFREE, G_SPLICE(19+41), G_BATTERY,
  G_ANCHOR, G_INSTMASK, G_PARENTS (e264/e268/e278/e283/e284/x14 md5-bound),
  G_BASE, G_ROOT(|d|=0.0), G_VMBIND, G_SPANBIND, G_PROJ (cert: idem 5.7e-16,
  kept2 within 1e-4 of k/N at 10-sigma), G_ROOMK10K (D/S bit-equal), G_FACTLOAD
  (|d|=0.0 both reads; flat-md5 ebebb447...), G_CORPUSGEN (28501), G_LR_BIND
  (LR_STABLE re-derived from the md5-bound calibration), G_ORTH, G_BUFSEP,
  G_BUDGET}.
- **Draw integrity EXACT** across arms (bit-identical corpus batches).
- **Envelope:** bursts <= 175s (5 chunks SANCTUARY + 3 TWIN, resume-checked),
  40s cooldowns, per-step polls (816 tags e285:*), max temp **78.0C** (at the
  margin, never past), **0 violations** of the 84C line. NO concurrent GPU
  jobs. Runtime ~21 min wall (inside the 15-20 GPU min estimate class,
  disclosed).

## PROVENANCE + COMMITS

- Birth (bars + dials frozen BEFORE compute): **6667dd0**; smoke pass 1 (one
  catch: E273_LR_SGD NameError; both centerpieces verified): **c073f40**; this
  report + metrics + figure: the final commit of this run.
- Machinery: e261's SRCT/LadderRooms ported whole by import (the committed
  file untouched); e284's orthogonalize_grads + buffer machines ported
  verbatim; e283's corpus-phase driver ported verbatim as the twin; the ONE
  new driver is chunked_sanctuary_phase (the composed build).
- Checkpoints (gitignored, md5s in metrics): e285_rooms.pt,
  e285_SANCTUARY_post.pt, e285_UNPROTECTED-TWIN_post.pt (+ resume ckpts).
- NO CONS (T259/e281); both post states checkpointed for any later landing
  pass. No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).
- n=1 per arm, one lineage, one session — the lottery note carried verbatim;
  nothing guaranteed.
