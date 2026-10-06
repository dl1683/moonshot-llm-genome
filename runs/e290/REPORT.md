# E290 — THE COUPLING-CONSTANT LADDER — REPORT (executor-written)

**Date:** 2026-10-06T12:23:54Z (datetime.now(UTC); all stamps this form)
**Script:** `lab/e290_coupling_ladder.py` (birth commit `3a8ec72`, BEFORE any
compute; bars frozen VERBATIM from the dispatch)
**Verdict:** **TIGHT-CONSTANT** — the 0.1x arm dies AND a lower rung holds;
the threshold is located inside the ladder's span.

## The question (dispatch verbatim)

Measuring the program's load-bearing constant: the orthogonal-drift threshold
at which an established write's read dies. e285 (the sanctuary) found a
20.1%-of-write orthogonal drift killed the read 330x with the budget held;
its t100 datum (7.2% drift -> 200x down) bounded the threshold at <= 0.07x —
a ONE-DRAW bound (R68's critic: the monotonicity assumed, the threshold could
sit at 0.5%). The ladder measures it. e288 just landed (ERROR-GATED-HOLDS —
active maintenance works); the constant prices when maintenance is NECESSARY.

## The arms

Four rungs at total-drift budgets {0.1x, 0.02x, 0.004x, 0.0008x} of the
write's norm 9.1788432658723 — the geometric 0.2-factor descent from e285's
0.5x anchor. The sanctuary form VERBATIM (SGD-M m0.9 wd0 at LR_STABLE
0.21738574801453703 x cosine_lr(t-1,1000); TWO SGD instances — opt_C stepped
/ opt_F never stepped with bitwise isolation probes; clip 1.0 -> g_perp =
g - P_room(g) verified every step; the equal-share per-step lr cap
triangle-guaranteeing S_400 <= BUDGET), the committed quiet-formed 10k fact
loaded bit-exact (three-way G_FACTLOAD, |d| = 0.0), the fact's own K10K room
bit-bound to e264_rooms.pt, the corpus stream seed 29001 with ALL FOUR RUNGS
drawing the IDENTICAL sequence (draw-integrity machine-checked: the t1 corpus
CE bit-identical; the budget the rungs' ONLY delta), NO maintenance, NO cons.
400 corpus steps per rung; reads at t100/200/300/400.

## The rungs (the ladder's measured constant — every number realized, never nominal)

| rung | budget (x write) | S usage | realized drift at t400 (% of write) | survival t100/t200/t300/t400 | t400 verdict |
|---|---|---|---|---|---|
| BUDGET-0.1X    | 0.1x    | 100.0% | 5.8288% | 0.0428 / 0.0198 / 0.0126 / **0.0097** | DIES |
| BUDGET-0.02X   | 0.02x   | 100.0% | 1.3909% | 0.2532 / 0.1207 / 0.0793 / **0.0599** | DIES |
| BUDGET-0.004X  | 0.004x  | 100.0% | 0.3233% | 0.7208 / 0.5236 / 0.3993 / **0.3194** | DIES |
| BUDGET-0.0008X | 0.0008x | 100.0% | 0.0710% | 0.9379 / 0.8774 / 0.8199 / **0.7674** | **HOLDS** |
| (e285 anchor, cited) | 0.5x | 100.0% | 20.10% | (0.0051 at t100) / **0.00303** | DIES |

Survival ratio = post g0(t) / the committed baseline 0.26464763283729553.
All four rungs consumed EXACTLY 100.0% of their budgets (worst overage
0.0006% = 4.4e-8 absolute, fp32 noise, 200x under the 1e-5 slack); realized
cumulative drift == drift-from-fact at t400 on every rung (nothing but the
budgeted stream moved the state).

## THE THRESHOLD (the constant, located)

- **In budget terms:** threshold in **(0.0008x, 0.004x]** of the write's norm
  (largest holding budget 0.0008x; smallest dying budget 0.004x).
- **In realized-drift terms:** threshold in **(0.0710%, 0.3233%]** of the
  write's norm — the orthogonal drift that kills an established write's read
  over a 400-step moving phase sits at a few parts in a thousand of the
  write's own size.
- **e285's one-draw bound (<= 0.07x) is tightened ~20-100x.** R68's critic
  was RIGHT to distrust the monotonicity assumption and RIGHT that the
  threshold could sit near half a percent — it sits at or just under it
  (the bracket's upper edge 0.32% is BELOW the critic's 0.5% guess).
- **Curve shape: GRADED, not step** — adjacent-rung survival factors 6.2x,
  5.3x, 2.4x (and 3.2x from e285's anchor to the 0.1x rung): survival scales
  progressively with budget across the whole ladder's span; no cliff inside
  the span. Descriptively (never a bar): across anchor + rungs, survival
  scales approximately INVERSELY with realized orthogonal drift (log-log
  slope ~ -1 over 2.5+ orders of drift magnitude: 283x less drift bought
  253x more survival), flattening as it approaches the unperturbed baseline.
- **The honest fine print on "HOLDS":** the holding rung's curve is still
  descending (x0.94 -> x0.77 over the phase). At this drift rate the 0.5x
  bar is met at t400 but not forever — passive protection buys time
  inversely proportional to the drift rate, never permanence. THIS is the
  quantity that prices maintenance: below ~0.1%-of-write drift an organism
  can move ~400 steps without an anchor; anything faster needs e288's
  controller.

## Gates (18/18 PASS — a failure would have halted)

- G_PARENTS: e285's sanctuary record + e288's controller record + e283/e284/
  e264/e268/x14 + the lr calibration + the fact artifact, all hard-bound
  (asserted literals + md5s).
- G_FACTLOAD three-way EXACT (|d post g0| = 0.0, |d gm12| = 0.0).
- G_ROOMK10K: the fact's own room bit-identical to e264_rooms.pt (D/S exact).
- G_ORTH per rung: max ||P_room g_perp||/||g_perp|| = 4.56e-17 (bar 1e-6),
  checked on EVERY step of every rung (1,600 checks).
- G_BUFSEP per rung: 400/400 bitwise isolation checks per rung (1,600
  total), 0 violations, opt_F's state EMPTY throughout; buf_C in-room at
  the fp floor (worst 5.15e-9 vs the 1e-4 bar).
- G_BUDGET per rung: S_400 <= BUDGET + 1e-5 AND realized cum <= BUDGET +
  1e-5 on all four; lr positive every ledger row; S monotone.
- Draw-integrity: the t1 corpus CE bit-identical across all four rungs
  (0.9891313314).

## Envelope

Bursts <= 175s (dispatch 180), 40s cooldowns (dispatch 30-60), per-step
thermal polls tagged `e290:<ARM>:corpus:*`; 1,642 polls, max 75.0C, zero
violations of the 84C line (dispatch 85). Wall ~65 GPU-minutes across three
passes (one external kill + resume); within the dispatch's 15-25 GPU-min
estimate per pass-class with the resume overhead disclosed.

## Deviations + disclosures (full list in metrics.json "deviations")

1. **OneDrive re-serialization catch (smoke pass 1):** G_PARENTS' raw-md5
   bind caught a CRLF re-serialization of runs/e273/lr_calibration.json
   between birth and smoke; all 8 JSON parents re-bound on GIT-CANONICAL
   md5s (content verified byte-identical to the committed blobs; .pt
   artifacts stay raw-bound). Commit `ff87ca8`.
2. **Resume-path catch (mid-run):** an external kill of the first full-run
   pass forced rungs 1-3 through the complete-resume early-return path,
   which lacked the lr-summary keys — fixed from the saved ledger exactly
   as the live path (commit `434c7c7`); the resumed rungs' numbers are
   bit-identical to the killed pass's (same resume ckpts; rung 4 continued
   from t182 with its saved generator state).
3. **The anchor is a class datum:** e285's 0.5x point ran seed 28501; this
   ladder ran 29001 (the per-cell family rule). The anchor is overlaid as
   the committed record's datum and excluded from the bracket arithmetic.
4. **Draw-integrity fine print:** the t1 clipped-||g|| is bit-identical on
   three rungs and differs by 6e-8 (relative ~7e-8) on BUDGET-0.004X —
   GPU reduction-order nondeterminism, inside the family's read-
   determinism law class (~5e-7); the t1 CE (the draws themselves) is
   bit-identical on all four.
5. **The smallest rung's realized drift reads 0.24% in-room** (vs ~4e-6 on
   the larger rungs): at 0.0065 total drift the fp32 rounding floor of the
   room projection becomes visible (expected: sqrt(k/N)-scale fp noise);
   the STEPPED gradient's orthogonality — the gate's object — stayed at
   4.54e-17 on every step. Disclosed in the ledger, never gated.
6. **T268 was absent from THINKING.md at birth** (the dispatch named it);
   e288's record was read from its committed metrics + NOTES/STATE instead.
   The coordinator folds T268; not this cell's edit.
7. NO CONS (T259/e281); all four post states checkpointed
   (runs/checkpoints/e290_BUDGET-*_post.pt, gitignored, md5s in metrics).
   NO TWIN (the death class is committed: e283's x3.42e-05 + e285's
   same-session twin x1.47e-05, both hard-bound).
8. n=1 per rung, one lineage, one session (the g-series standing lottery
   caveat carried verbatim); the rungs' MONOTONE PATTERN across
   bit-identical draws is the registered object.

## What it means (for the coordinator's fold, not adjudicated here)

The program's load-bearing constant has a value: **the read dies when the
orthogonal drift crosses a few parts in a thousand of the write's norm** —
two orders of magnitude tighter than the two-channel law's original
"displacement exceeds the write's norm" clause, and graded, not a cliff.
P-e290a CONFIRMED (the 0.1x rung died inside e285's lethal band); P-e290b
REFUTED (a passive floor EXISTS — the 0.0008x rung holds); P-e290c REFUTED.
The passive/active division now has numbers on both sides: passive
preservation works only under ~0.1%-of-write-per-400-steps drift; e288's
error-gated controller (x3.4792, 87.2% of a 0.5x budget) is the necessary
instrument above that line — and its 60/40 corpus/maintenance split at
budget 0.5x sits FAR above the passive threshold, i.e., the controller was
operating in the regime where no passive form could have survived.

## Outputs

- `runs/e290/metrics.json` (COMPLETE, progressive writes superseded)
- `runs/e290/e290_coupling_ladder.png` (the survival-vs-budget curve, log-x,
  e285's 0.5x anchor overlaid, the threshold bracket shaded)
- `runs/e290/REPORT.md` (this file)
- checkpoints: `runs/checkpoints/e290_{rooms,BUDGET-*_post}.pt` (gitignored)

Commits: birth `3a8ec72` -> smoke catch `ff87ca8` -> smoke record `860e218`
-> resume catch `434c7c7` -> final fold (this commit).
