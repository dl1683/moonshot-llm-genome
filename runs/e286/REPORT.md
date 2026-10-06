# E286 — THE RE-ANCHORING CELL — REPORT (executor-written)

**VERDICT: MAINTENANCE-FAILS** (the dispatch's gates clause — and the ratio
clause agrees: 0.0180x < 0.05x). The lab's first ACTIVE memory maintenance
design, run at its registered first form, FAILS — honestly, with the
mechanism's own signature measured every step of the way.

```
survival ratio (post g0 / the committed baseline 0.26464763283729553)
              t100     t200     t300     t400
RE-ANCHORED   x0.0311  x0.0015  x0.0014  x0.0180   (post g0 0.0047667)
SANCTUARY-TWN x0.0047  x0.0035  x0.0035  x0.0021   (post g0 0.0005631)
e285's committed sanctuary (the parent):      x0.0030 (post 8.03e-04)
```

## The two arms' curves

- **RE-ANCHORED** (the build): the read opens like the passive stack
  (x0.0121 at t24, before any maintenance), then the maintenance era
  begins: t100 x0.0311 — **6.6x the same-session twin at the same
  milestone** — then collapse (t200 x0.0015, BELOW the twin's x0.0035;
  t300 x0.0014), then a late bounce to x0.0180 at t400 (the read taken
  immediately after the 16th maintenance step, on an organism whose
  corpus stream had been frozen to lr=0 since ~t280).
- **SANCTUARY-TWIN** (the control): e285's passive stack verbatim,
  replicating the parent's class on its own draw stream (t400 x0.0021 vs
  e285's committed x0.0030; S = 100.0% of budget; cap bound 397/400 —
  the same economics).

## The between-steps window (the dispatch's demanded read — sampled twice)

```
win1 (t25):   t24 0.003200  |  t25-PRE 0.003178  ->  t25-POST 0.007001  |  t26 0.006134
              THE LIFT: x2.203 — the first maintenance step MORE THAN DOUBLED
              the read, and the lift HELD one corpus step later (x0.0061)
win2 (t200):  t199 0.001169 |  t200-PRE 0.001082  ->  t200-POST 0.000408 |  t201 0.000397
              THE CRASH: x0.377 — by mid-phase the same step LOWERS the read
```

The sawtooth is real and it INVERTS: the maintenance stream lifts the read
when the state is near the fact (win1) and drives it down when the state is
far (win2). The shape is the cell's cleanest mechanism datum: the
maintenance step is not a fixed-direction restorer — its effect on the read
changes sign with the state's drift.

## The budget split (the dispatch's demanded disclosure)

```
                  S_corpus   S_maint   S_total         BUDGET (0.5 x 9.1788)
RE-ANCHORED       1.8679  +  4.1067  = 5.9745   vs    4.5894   = 130.2%  BLOWN
                  (40.7%)    (89.5% of budget alone — the maintenance stream
                              consumed 2.2x the corpus stream's total)
SANCTUARY-TWIN    4.5894  = 100.0% (the passive stack's own triangle-tight fill)
```

- The maintenance lr is FROZEN (uncapped, per the dispatch): 16 SGD-M steps
  at MAINT_LR 0.21738574801453703 moved 0.162 (first, buffer empty) rising
  to 0.25-0.31 (momentum building in the fact-side buffer, ||bufF|| ~2.5)
  — per-step ~15-25x the corpus cap's equal share (~0.011).
- The corpus cap's amended reservation (all remaining optimizer events)
  did its job: the corpus side stayed triangle-bounded (1.87 <= 4.59) and
  drove its own lr to 0 from ~t280 — **the corpus stream paid the
  maintenance's bill and died for it**: the corpus CE DEGRADED
  (early median 0.950 -> late median 1.238; the twin improved
  0.93 -> 0.78). The HOLDS bar's "stream live" clause fails independently.
- G_BUDGET failed NON-HALTING exactly as designed (e285's assert semantics
  evolved per the dispatch's FAILS clause) — the verdict routed, nothing
  halted.

## The gates (18 bind/isolation gates PASS; G_BUDGET FAIL by outcome)

- **G_ORTH PASS** (both arms): max ||P g_perp||/||g_perp|| = 4.58e-17 over
  all 400 corpus steps — the corpus stream never aims in-room. The
  maintenance gradient is deliberately UNPROJECTED (the fact's own stream).
- **G_BUFSEP v2 PASS**: bidirectional bitwise isolation around EVERY
  optimizer event — 400 corpus checks (opt_F untouched) + 16 maintenance
  checks (opt_C untouched), 0 violations; **opt_F stepped EXACTLY 16x ==
  16 maintenance steps** (machine-counted); buf_C in-room at the fp floor
  (max 4.99e-09); buf_F's composition disclosed (in-room ~0.062 — the
  maintenance momentum's own direction, mirroring its gradient).
- G_FACTLOAD bit-exact (|d post g0| = 0.0); G_ROOMK10K D/S bit-identical;
  all parents md5-bound (e285's record now hard-bound as this cell's
  direct parent); draw-integrity EXACT (t1 corpus CE + clipped gn
  bit-identical across arms — the maintenance mechanism the arms' ONLY
  delta).

## The mechanism's direction read (P-e286c decided)

The Dmix install gradient's in-room fraction, measured at every one of the
16 maintenance steps: **0.0597-0.0612, flat** — statistically identical to
the raw corpus gradient's ~0.060. At the established (and drifting) state,
the stream that FORMED the fact points **94% out-of-room**: the union CE
carries 112 name tokens against 12,240 corpus tokens, and the name-position
teaching signal (name CE 0.02-0.12 — the write still knows its name) rides
~1% of the gradient mass on what is, directionally, another corpus step.
**The first form's obstacle is DIRECTIONAL, not merely dosage**: the
maintenance step adds out-of-room transport (win2's crash; the read below
the passive twin at t200-t300) with a name-tilt too small to re-center
against it (win1's x2.2 lift — real, held one step, and overwhelmed).

## The maintenance-lr sensitivity (the PARTIAL bar's honesty clause)

- At LR_SGD/100 (== LR_STABLE, the SGD-denominated reading of "1/100 of
  the install's formation lr"), one step per 25 corpus steps: ~0.25/step,
  4.11 total = 89.5% of budget alone. The budget is blown at t275 and the
  corpus stream freezes; the read ends at x0.0180 — 8.5x the twin but
  28x under the 0.5x bar and 2.8x under even the 0.05x PARTIAL bar.
- The dose ladder implied by the ledgers: a maintenance lr at the corpus
  share scale (~x0.01 of this one) fits the budget 16x over — but at this
  DIRECTION (6% in-room, 1% name mass) it would re-anchor ~nothing
  (win1's lift needed a 0.16-size step to move the read x2). The AdamW-
  literal reading (1e-5/step) moves 1.6e-4 total — four orders below the
  0.66 drift that killed the read ~200x at t100 in e285 (named + rejected
  at birth; the arithmetic stands).
- THE REGISTERED SENSITIVITY CONCLUSION for the next cell: tune the
  SIGNAL, not just the dose — a name-only CE (the masked name tokens
  alone, no corpus mix), or a name-weighted union, at a budget-fitting
  lr, is the design the direction ledger points to. P-e286a's lift
  signature (win1) is the existence proof that the read responds to the
  install stream's name component; P-e286c's flat 6% in-room is the proof
  the current vehicle cannot deliver it.

## The verdict clause (frozen bars, verbatim outcome)

"< 0.05x or the maintenance steps break the budget/orthogonality gates —
the active design's first form fails; the trajectories verbatim." BOTH
branches fired: ratio_400 0.0180x < 0.05x AND G_BUDGET broken (S_total
5.97 > 4.589). The orthogonality gate held (4.58e-17) — the machinery is
clean; the design's first form is what failed.

## Disclosures

- The maintenance-lr denomination fork (disclosed at birth): SGD-denominated
  LR_SGD_matched/100 chosen; the AdamW-literal 1e-5 named + rejected as a
  guaranteed-null. The chosen dose's 4.11 displacement is the measured
  consequence — the fork's outcome is now on the record either way.
- The maintenance step is e261's chunked_install Dmix step VERBATIM (the
  stream that formed the fact) — the token-mass arithmetic (112:12240) was
  registered at birth (P-e286c) and is what the direction ledger measured.
- Milestone reads are POST-maintenance (every milestone a multiple of 25);
  the t400 endpoint includes the 16th maintenance step — the x0.0180 is a
  post-lift read on a stream-frozen organism, disclosed.
- NO CONS (T259/e281); both post states checkpointed
  (e286_RE-ANCHORED_post.pt / e286_SANCTUARY-TWIN_post.pt).
- n=1 per arm, one lineage, one session (the g-series standing lottery
  note).
- Thermal envelope: bursts <= 175s, 40s cooldowns, per-event polls tagged
  e286:ARM:phase to runs/_envelope_log.jsonl; max 71C, zero >= 84C
  violations; ~44 min wall, GPU-sequential arms.
- No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).

## Provenance

- Script: lab/e286_reanchor.py (birth commit 4730715, before any compute;
  smoke pass folded at 4d128b1/e9cc5fc).
- Rig: e285's sanctuary driver VERBATIM (ported) + the maintenance
  injection; e261's machinery imported whole; the room bit-bound to
  e264_rooms.pt; the fact loaded bit-exact (md5 0f6dc1cf..., flat-md5
  ebebb447..., behavioral |d| = 0.0).
- Streams: corpus seed 28601 (bit-identical across arms), install seed
  28602 (the maintenance's own).
- Outputs: runs/e286/metrics.json (COMPLETE — adjudicated),
  runs/e286/e286_reanchor.png, this REPORT.
