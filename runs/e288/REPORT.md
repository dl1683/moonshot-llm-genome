# E288 — THE ERROR-GATED MAINTENANCE CELL — REPORT (executor-written)

**VERDICT: ERROR-GATED-HOLDS** (the frozen bars, adjudicated at the t400
endpoint against the birth-committed registration — commit `eee93b6`,
BEFORE any compute; smoke ledger commit `1f1cf6d`).

> ">= 0.5x — THE FIRST ACTIVE SURVIVAL, by the controller; the build
> era's founding success." Ratio at t400 (post the 16th maintenance
> step): **x3.4792** — not merely held but **3.5x the loaded baseline
> read**, with the budget HELD (S_total 87.2% <= 100%), the stream LIVE
> (corpus CE 0.9370 -> 0.8805, improving), and every hard gate PASS.

Gates **20/20 PASS** (incl. G_NAMEWIN, G_PARENTS with e287 md5-bound as
the cited control). Thermal max **72.0C over 852 polls, 0 violations**
(175s bursts, 40s cooldowns, the 84C line never approached). Wall
2098.5 s (~35 min; two arms GPU-sequential, no concurrent jobs).

---

## 1. The one changed dial (frozen at birth)

e287's rig VERBATIM (the name-only CE on win_i's 112 masked positions,
the fact-side buffer opt_F, the orthogonal corpus projection, M=25
cadence, the windows) with the maintenance dose **ERROR-GATED** — a
closed-loop controller on the SAME g0 battery the bars read:

```
deficit_t = max(0, 0.26464763283729553 - read_t) / 0.26464763283729553
lr_m_t   = LR_M_MAX * deficit_t            # the sicker, the stronger
applied  = min(lr_m_t, cap_m)              # the symmetric budget cap
cap_m    = ((B_M - S_maint)/rem_maint)/||b_m||     # b_m = 0.9*buf_F + g_name
```

**The calibration** (frozen before compute): LR_M_MAX := 0.40 x BUDGET /
sum(e287's 16 committed b_m norms = 81.5736) = **0.02250444534525284**
(re-derived at runtime from the loaded write_norm and asserted == the
frozen literal) — so that IF every step ran at full deficit the
maintenance would consume ~40% of the budget, deliberately
redistributing e287's 96.1/3.9 split to **60/40** (the reservation
pools SPLIT: B_C = 2.7537 corpus, B_M = 1.8358 maintenance, each
capped by the same symmetric equal-share form; S_total <= BUDGET by
the triangle bound, by construction). Disclosed at birth: without the
split, e287's joint pool would hold the maintenance to ~3.9% forever —
the controller structurally null.

**The controller's arithmetic was asserted LIVE at every event** (lr
never exceeds LR_M_MAX; == 0 at zero deficit; == LR_M_MAX at full
deficit) and exercised in the smoke run (event 3 of the smoke read
0.3314 ABOVE baseline -> deficit exactly 0.0000 -> lr 0.000000 ->
realized step 0.00000 — the dose verifiably VANISHES when healthy).

## 2. The curve vs e287's (the frozen comparison)

| read (x committed 0.2646) | t100 | t200 | t300 | t400 |
|---|---|---|---|---|
| **ERROR-GATED** (this cell) | **x2.6875** | **x3.4846** | **x3.4033** | **x3.4792** |
| NAME-FIXED-TWIN (e287's exact form, same session, same stream 28801) | x0.0354 | x0.0611 | x0.1050 | x0.1623 |
| e287 committed (CITED, md5-bound) | x0.0423 | x0.0777 | x0.1221 | x0.0917 |
| e287's sanctuary twin (cited) | x0.0053 | x0.0037 | x0.0038 | x0.0018 |

The build opens **64x above the twin** at t100 and ends **21.4x the
twin** and **37.9x e287's committed endpoint**. The twin faithfully
replicates e287's flat-dose class in-session (its endpoint x0.1623 vs
e287's x0.0917 — draw scatter, same class; every one of its 16 events
cap-bound at exactly 1.00x its 0.01107 share, e287's machine signature
reproduced). The arms shared bit-identical corpus draws (draw-integrity
check EXACT) — the controller is the only delta.

## 3. The controller trace (the registered read)

The dose tracked the deficit at every event; **the cap NEVER bound**
(binder 16/16 controller):

| event | t | gate read | deficit | applied lr | realized step |
|---|---|---|---|---|---|
| 1 | 25 | 0.0093 | 0.965 | 0.02171 | 0.0217 |
| 4 | 100 | 0.0047 | 0.982 | 0.02210 | 0.0684 |
| 8 | 200 | 0.0419 | 0.842 | 0.01894 | 0.0905 |
| 12 | 300 | 0.0793 | 0.700 | 0.01576 | 0.0880 |
| 16 | 400 | 0.0905 | 0.658 | 0.01481 | 0.0911 |

- **The deficit tapered 0.965 -> 0.658 as the read recovered** — the
  controller visibly self-limited (P-e288c's discriminator fired: the
  read's inter-event troughs rose from x0.018 (0.0047) to x0.34
  (0.0905), a **19x floor rise**, so each gate found the read less
  sick and dosed less).
- Applied lr_m median **0.01894** (range 0.01471-0.02211) vs LR_M_MAX
  0.02250 — and vs e287's flat median **0.00208** (a ~9x dose) with
  per-event displacement ~0.088 vs e287's 0.0111 (~8x).
- The gate read at t25/t200 was **bit-identical** to the window
  pre-read (|diff| = 0.0 exactly, both events) — the controller's
  battery is the milestones' own battery, fp-deterministic.

## 4. The windows (both directions)

- **win1 t25: pre x0.0352 -> post x0.2639 = LIFT x7.491** (held x0.2418
  at t26) — vs e287's x2.819 at the same window.
- **win2 t200: pre x0.1583 -> post x3.4846 = LIFT x22.014** (held
  x3.4599 at t201) — vs e287's x3.916. A single maintenance step
  **restored the read to 3.5x baseline** mid-phase.

The honest shape: the sawtooth persists with a LARGE amplitude — the
post-maintenance peaks (the milestone reads, post-maintenance by the
birth-frozen convention) sit at x2.7-3.5 while the pre-maintenance
troughs sit at x0.02-0.34 (rising through the phase). The read is not
statically pinned; it is **dynamically caught** — each 25-step decay
is answered by a deficit-proportional re-teaching step. The trough
trajectory (0.0047 -> 0.0905, monotone after event 3) is the
controller's converging floor, the cell's cleanest signature.

## 5. The budget split (held, both arms)

| stream | consumed | allocation | share |
|---|---|---|---|
| S_corpus (the B_C pool) | 2.7537 | B_C = 2.7537 (60%) | **100.0%** |
| S_maint (the B_M pool) | 1.2462 | B_M = 1.8358 (40%) | **67.9%** |
| **S_total (build)** | **3.9998** | BUDGET = 4.5894 | **87.2%** |
| twin (e287's joint pool) | 4.5894 | BUDGET | 100.0% |

The realized split **68.9/31.1** (corpus/maintenance of S_total) vs
e287's 96.1/3.9 — the redistribution delivered. The maintenance spent
only 67.9% of B_M **because the deficit tapered below 1** (the
controller self-limited below its full-deficit calibration — the
honest gap between "40% if always maximally sick" and "31% of total
when recovering"). Final drift from the fact 1.1437 (24.9% of budget
realized cum) vs the twin's 1.795 — less total displacement, more
read.

## 6. The registered reads, answered

- **In-room fraction (expect ~0.06): CONFIRMED** — the name-only
  gradient's per-step in-room fraction median **0.06026** (range
  0.0597-0.0611 across all 16 events) == the corpus ~0.06, T267's
  at-chance reading carried verbatim: the lifts ride function-space
  direction, not room geometry, even at the controller's 8x doses.
- **Corpus CE (live): CONFIRMED** — 0.9370 -> 0.8805 (improving); the
  stream learned at its smaller 60% allocation (398/400 steps
  cap-bound at the B_C share).
- **The budget split + the trace**: above.
- buf_C in-room max 5.53e-09 (fp floor); buf_F in-room disclosed
  0.0600 (the name stream's momentum — never a bar).

## 7. Gates and envelope

All 20 gates PASS: G_PARENTS (e287 md5 `b7d18b8b...` hard-bound, its
b_m trace asserted event-by-event), G_FACTLOAD (read delta exactly
0.0), G_ROOMK10K (bit-identical D/S), G_NAMEWIN (ZEPHYRA 60/60),
G_ORTH max 4.55e-17 both arms, G_BUFSEP bidirectional **both** arms
(400+16 checks each, 0 violations; opt_F stepped EXACTLY 16x + 16x),
G_BUDGET PASS (total + per-allocation + twin joint), G_LR_BIND (the
LR_M_MAX runtime re-derivation asserted), G_MAINTBIND (the
controller's law bound). Envelope: bursts <= 175s (5 chunks/arm), 40s
cooldowns, max 72.0C, **0 violations** over 852 polls tagged
`e288:<ARM>:<phase>`.

## 8. Disclosures (the honesty ledger)

- **The read OVERSHOOTS the baseline** (x3.48 at t400): the bars asked
  only >= 0.5x "survival"; the controller's re-teaching pushed the
  name battery to 3.5x the original write's read. The milestone reads
  are POST-maintenance by the birth-frozen convention; the
  between-event troughs (the honest floor) rose to x0.34 by phase's
  end — a floor still 25% of baseline. Both facts reported; nothing
  averaged away.
- **P-e288a confirmed** (the share hypothesis — T267's named dial);
  **P-e288b's overshoot branch half-fired** (the dose did NOT saturate
  or oscillate — the lifts grew super-linearly — but the read
  overshot the original baseline, which no bar contemplated);
  **P-e288c's discriminator fired** (the taper is visible: deficit
  0.965 -> 0.658).
- **n=1 per arm, one lineage, one session** (the g-series standing
  lottery caveat carried verbatim); the twin replicates e287's FORM on
  this session's stream, not its bits (e287's committed curve cited,
  md5-bound). NO CONS (T259/e281 — the bars read the WRITE and the
  DISPLACEMENT only; both post states checkpointed:
  `e288_ERROR-GATED_post.pt`, `e288_NAME-FIXED-TWIN_post.pt`).
- NO NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator
  folds).

## 9. The named successor question

The dose-response is now the build era's first DESIGN CHART (this
cell's two points: flat ~0.011/step -> x0.16; error-gated ~0.088/step
-> x3.48 — plus e287's committed x0.09). The open dials the trace
names: (i) the **cadence** (the between-event decay is the residual
loss — a denser gate at the same per-event dose would raise the floor
toward the peaks); (ii) the **overshoot** (whether x3.5 peaks are
benign re-teaching or the read's own scale distorting — the landing/
cons pass on the checkpointed state would settle it); (iii) the
**calibration constant** (LR_M_MAX at other shares — the same
controller, other budgets).

— executor, e288, 2026-10-06 (datetime.now(UTC) stamps in metrics.json)
