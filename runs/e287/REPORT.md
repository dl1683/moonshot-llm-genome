# E287 — THE NAME-WEIGHTED MAINTENANCE CELL — REPORT (executor-written)

**VERDICT: SAWTOOTH-CONFIRMED** (the frozen bars, adjudicated at the t400
endpoint against the birth-committed registration — commit `6a66fab`,
BEFORE any compute).

> 0.05x-0.5x survival at t400 (ratio **0.0917x**) with the windows showing
> consistent LIFTS (every sampled window x>1), the budget HELD (100.0%,
> the maintenance stream alone only 3.9%), and the stream live (corpus CE
> 0.926 -> 0.819, improving) — **the mechanism real (the name signal lifts
> the read under traffic), the dosage still short.**

Gates **20/20 PASS** (including the new G_NAMEWIN). Thermal max 75.0C over
838 polls, 0 violations. Wall 2225 s (~37 min; two arms GPU-sequential).

---

## 1. The birth-day correction (read this first)

Porting the rig exposed a **MISBIND in e286's maintenance batch**: its
driver call bound `anchor_full` (the original-host anchor bank) into the
`inst_x` slot, so the masked positions of e286's "install windows" held
the **HOST OPENINGS** (`ELIZABE`/`FLORIZE`), not ZEPHYRA — **0/60 windows
carried the name** (decode-verified: `win_i` masked positions hold ZEPHYRA
60/60; `win_i[:, :PRE]` is bit-equal to `anchor_full[:, :PRE]`).
Consequences, stated honestly:

- e286's "name_ce" 0.0615 was the CE on memorized host-opening text;
- e286's maintenance gradient carried **ZERO name content** — its flat
  in-room 0.0597-0.0612 == the corpus ~0.06 *because the step WAS a
  corpus-text step*;
- e286's win1 lift x2.203 came from a **host-text batch**.

**THE NAME SIGNAL HAD NEVER BEEN INJECTED BEFORE THIS CELL.** e287 is its
first injection. The dispatch's premise ("the name signal rides ~1% of the
mixed gradient's mass") is superseded by the stronger measured fact (0%).
The design was run exactly as dispatched; the correction is hard-gated
going forward (**G_NAMEWIN**: the masked positions must decode to ZEPHYRA
in ALL 60 windows — a HARD gate).

## 2. The two changed dials (both frozen at birth)

**(i) THE NAME-ONLY CE.** Maintenance batch := ix(16) TRUE install windows
(`win_i` — e261's `build_win` verbatim: 130 pre-context tokens | ZEPHYRA |
119 post); token-level CE; select the masked NAME positions (7 x 16 =
112); **loss := their mean**. The Dmix corpus half is dropped entirely —
the pure name signal, no corpus dilution. Measured `name_ce` 1.14 -> 1.04
nats across the 16 steps (the drifted state's real name-knowledge
deficit; e286's misbound reading was 0.06 — host text the model had
memorized).

**(ii) THE BUDGET-FITTING LR** (the momentum-capped form). Schedule
ceiling `MAINT_LR_BASE` = LR_SGD/100 = LR_STABLE = 0.21739 (e286's
denomination verbatim; the AdamW-literal 1e-5 fork stays rejected). The
applied lr at every maintenance event: `lr_m = min(MAINT_LR_BASE,
cap_m)`, `cap_m = share_m/||b_m||`, with `b_m = 0.9*buf_F + g_name` the
EXACT pending in-step momentum and `share_m = (BUDGET - S_total)/(1 +
remaining corpus steps + later maintenance events)` — the SAME equal-share
reservation the corpus cap uses, symmetric. **Every one of the 16
maintenance steps realized EXACTLY 1.00x its reserved share** (0.01107/
step) while the momentum built b_m 1.00 -> 7.61 and the applied lr_m fell
0.0111 -> 0.0015 — the cap, not the schedule, governs. e286's 89.5%
blowout is structurally excluded: **S_maint = 0.1772 = 3.9% of budget**
(vs 4.107 = 89.5% in e286); **S_total = 4.5894216311 <= BUDGET
4.5894216329 — G_BUDGET PASS**.

## 3. The two curves (survival vs the committed baseline 0.2646)

| t | NAME-MAINTAINED | SANCTUARY-TWIN (bit-identical draws) |
|---|---|---|
| 100 | **x0.0423** | x0.0053 |
| 200 | **x0.0777** | x0.0037 |
| 300 | **x0.1221** | x0.0038 |
| 400 | **x0.0917** | x0.0018 |

The build OPENED 8.0x above the twin at t100 and CLIMBED to 33x at t300
(0.042 -> 0.078 -> 0.122); the t400 endpoint dipped to 0.092 (the final
25-step interval's corpus drift) — inside the SAWTOOTH band. References:
e285's committed passive sanctuary x0.0030; e286's misbound active form
x0.0180. **The endpoint is 5.1x e286's first form and 50.9x the
same-session passive twin.**

## 4. The windows (both directions, verbatim)

| window | pre | post | lift | held after |
|---|---|---|---|---|
| win1 t25 | x0.0165 (0.004354) | **x0.0464** (0.012274) | **x2.819** | x0.0464 at t26 |
| win2 t200 | x0.0198 (0.005252) | **x0.0777** (0.020564) | **x3.916** | x0.0756 at t201 |

**Every sampled window lifted (consistent LIFTS)** — and the contrast
with e286 is decisive: e286's win2 CRASHED x0.377 (the misbound
host-text step inverted with drift); e287's name-only step lifts in BOTH
phases. The single maintenance step at t200 TRIPLED the read. This is the
sawtooth the build lane has been hunting since e286's win1: **real,
repeatable, phase-robust.**

## 5. The registered reads (the dispatch's list)

- **The maintenance gradient's in-room fraction per step**: 0.0606-0.0612,
  median **0.0610** — *NOT* >> 0.06. **The registered expectation is
  falsified**: the pure name gradient at the established-then-drifted
  state points ~93.9% OUT-OF-ROOM, the same in-room share as the raw
  corpus gradient (and e286's misbound steps). Yet the lifts fire — **the
  name signal's effect on the read does NOT ride in-room geometry** (the
  room-frame is not the operative frame for the name channel; P-e287a's
  mechanism is wrong, its outcome prediction partially right). State
  dependence disclosed: at the near-fact state (smoke t2, drift 0.006)
  the name gradient read 0.0133 in-room — the geometry moves with the
  state.
- **The budget split** (per milestone, S_corpus + S_maint vs 4.5894):
  t100: 1.0902 + 0.0443 (24.7%); t200: 2.1975 + 0.0886 (49.8%); t300:
  3.3049 + 0.1329 (74.9%); t400: 4.4122 + 0.1772 (100.0%, held). The
  maintenance's 16 steps cost 0.1772 total; the corpus stream paid its
  own way (397/400 steps cap-bound, lr ~0.0057-0.0059 applied).
- **The corpus CE**: improving — early-median (t in [1,100]) 0.9261 ->
  late-median (t in (300,400]) 0.8192. **The stream is LIVE** (the
  budget-fitting cap did not freeze the organism; e286's frozen-stream
  failure mode is gone).
- The maintenance lr realized: median 0.00208, min 0.00146 (16/16
  cap-bound) — the dosage still short is the named next dial.

## 6. The gates (20/20 PASS)

Hard gates (HALT class) all PASS: G_NAMEWIN (new — the decode gate, the
misbind's correction), G_PARENTS (e286 metrics md5-bound
`9c5a3cef...` + verdict/numbers hard-bound; the fact ckpt md5/size/step;
the room bit-bound), G_FACTLOAD (|d| = 0.0 on both committed reads),
G_ROOMK10K (D/S bit-equal), G_PROJ, G_LR_BIND, G_MAINTBIND, G_CORPUSGEN,
G_BUFSEP (bidirectional bitwise isolation: 400 corpus + 16 maintenance
checks, **0 violations**; opt_F stepped EXACTLY 16x, machine-counted;
buf_C in-room max 5.27e-09 — fp floor). Non-halting: **G_ORTH PASS**
(max 4.55e-17), **G_BUDGET PASS** (S_total 4.5894216311 <= 4.5894216329
+ slack). Draw-integrity EXACT (t1 CE + clipped ||g|| identical across
arms — the maintenance mechanism is the arms' ONLY delta).

## 7. What this cell established, and the named successor

1. **The name signal is a real maintenance mechanism** — the record's
   first active-maintenance LIFTS that survive adjudication: consistent,
   phase-robust (win1 AND win2), on bit-identical corpus draws vs a twin
   that dies at x0.0018.
2. **The dosage is the remaining dial** — 0.011/step (the equal-share
   cap) tripled the read per step but could not out-climb the corpus
   drift between steps; the trajectory climbed t100->t300 then dipped.
   The successor's dial is NOT the budget (held) and NOT the direction
   (lifts everywhere): it is the SHARE ALLOCATION — a larger maintenance
   share of the same budget (e.g., a maintenance-priority reservation, or
   a denser cadence at the same per-step displacement).
3. **The room-frame is not the name channel's frame** — in-room ~0.061 ==
   the corpus gradient's, yet the lifts fire: the read's response to the
   name gradient is a function-coupling effect, not an in-room
   displacement effect (T265's holographic law, now with a working
   actuator).

## 8. Provenance

- Birth commit `6a66fab` (bars + arms + both conventions verbatim,
  BEFORE compute); smoke verified both dials (name-only construction +
  cap arithmetic) with no code changes after birth; full run single-pass
  (no resumes across arms' completions; chunk-level resume ckpts used
  only for burst caps).
- Corpus stream seed 28701 (bit-identical across arms); install stream
  seed 28702; the fact `e261_K10K_inst_resume.pt` (md5
  `0f6dc1cf46850ce655dfafc9c853d467`) loaded bit-exact; the room = the
  fact's own K10K (seeds 26113/26114, bit-bound vs e264_rooms.pt).
- Post states checkpointed (gitignored, md5s in metrics):
  `e287_NAME-MAINTAINED_post.pt`, `e287_SANCTUARY-TWIN_post.pt`.
- Envelope: bursts <= 175 s, 40 s cooldowns, per-step polls (838), max
  75.0C, 0 violations; NO concurrent GPU jobs; timestamps
  datetime.now(UTC).
- NO CONS (T259/e281). No NOTES/THINKING/QUEUE/STATE edits (dispatch;
  the coordinator folds). n=1 per arm, one lineage, one session — the
  arms' DIFFERENCE is the registered object.
