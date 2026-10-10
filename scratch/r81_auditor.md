# R81 AUDITOR REPORT (written 2026-10-10T22:19Z; window under audit = R80 8c6983e 14:17 EDT -> ff30b1a 18:09 EDT)

Scope per scratch/r81_agenda.md items 1-2 (+3-4 as ordered). Receipts = file/key, not prose.
Verdicts: 1) CELLS SOUND-WITH-REPAIRS · 2) FOLD EDITS SOUND-WITH-REPAIRS · 3) V3.1 SOUND · 4) COUNTER-COLUMN SOUND (both counts exact).

## 1. THE CELLS — SOUND-WITH-REPAIRS

- **x45** birth chain ffbd479 (15:31, rig+bars) -> 79fb6ef (smoke) -> 4e161e7 (run) -> 79e3be0 (fold);
  bars frozen in metrics.adjudication.bars_verbatim, compute landed 16:28 — pre-compute freeze holds.
  Headlines traced to runs/x45/metrics.json: r_g-12|g0 0.7778/0.2335, r_g0|g+12 0.8333/0.2888
  (adjudication.separations), gates 10/10, G_STREAM n_compared 900 bit_identical true, replay drifts
  2.0e-11 (s1)/0.00212 (walk_cons s300)/0.0046 (anneal s900) — the commit's "0.0000/0.0021/0.0046".
- **g1bS9** 5728274 (rig 1613 lines WITH bars committed at birth) -> ac6811f -> d46cbff (19:40:03Z).
  annealed s1 0.010666 / control 0.0022982 vs bar 0.05; margin 4.6409; t0 0.742721; retention
  0.596 vs 0.383 (metrics.adjudication). NICK-1: commit/NOTES say "13/13 gates" but metrics.gates
  holds 11 keys; 13 = gate instances (G_POOL log-only + G_BITROOT logged [K]+[A]). Defensible,
  not single-artifact traceable.
- **e342** (window-relevant) a09b47d -> cc011dd -> bde5322: mean_sag 0.4803 vs 0.15 bar, 22/22
  gates, P hit both calls.
- **x47** de8ce29 -> 14098d7 -> 7fd122e: gates 19/19; pooled n_lower 15 / n_upper 18; max_lower
  0.14045 (the living lift) > min_upper 0.058620 -> contradiction, verdict UNRESOLVABLE per the
  frozen "UNRESOLVABLE iff not resolvable"; onset 0.0586-0.0594 (pooled_g0 min_upper 0.059359) vs
  x44 ref 0.052743 = 1.11/1.13x ("~1.1-1.2x"). P missed (0/4 bits).
- **e343** 2bfd6e7 -> fca25ae -> e8002c7: gates 13/13; separations +0.5714/0.3253 and +0.5714/0.2545
  (n=7) — failed the frozen rule (REPORT.md:26 "dominant sign - AND consistency >= 0.75 AND
  mean|d| >= 0.15") **on SIGN; no scorer-side rescue taken**: the write_norm separation (sign -,
  1.0) is disclosed as a non-primary co-finding, the primaries are the registered object.
  Segments traced to REPORT table: zeph_s10/s12 r 0.986/0.993, upper s100-s400 sign -0.75 (n=4),
  anneal_s125 0.124; G_X45REPRO reproduces 0.7778/0.2335. NICK-2: metrics.birth_commit pins
  14098d7 = the SIBLING x47 smoke commit (HEAD at run start), not e343's own birth 2bfd6e7 —
  harmless for freeze order (both pre-compute) but the field name misleads. x45 has the same
  soft nick (pins its own smoke 79fb6ef).
- **e344** 86ab9d2 -> bc06f4c -> **44f4c4f (re-freeze, 17:58:20) -> 6048acd (adjudicated, 18:06:39)**:
  order VERIFIED (metrics.git_head_final == 44f4c4f). Touched-no-bar VERIFIED: diff 86ab9d2..44f4c4f
  changes only G_LANDING plumbing + a deviations disclosure; the bar/P docstring section
  (rig lines 40-110) is byte-identical birth->final; smoke bc06f4c touches no bar constant.
  G_LANDING two-row: root-own |d| 0.0 at 2e-6, replay drift 0.058535 within class 0.15, flat
  0.02131, birth tol 0.020 disclosed inside metrics.gates.G_LANDING. All verdict numbers traced:
  23 pairs +0.7826/0.3126 + +0.7826/0.2680; young <=s50 +0.8462/0.4275; upper >=s75 +0.70/0.1632;
  N1 rider R2_age 0.2214 / R2_menu 0.0548 / full 0.2793; per-menu 0.7467/0.6442/0.5520 (ages
  10-400/2-700/400-1300); residuals anneal_s125 -0.593 and walk_cons_s125 -0.576, both age 525.
  NICK-3: "the failed pass preserved in the progressive metrics + run log" is HALF-true — no
  first-pass metrics snapshot exists; run.log preserves only the truncated first process (P3/
  G_RIGCONST at 555.6s, then restart) and no "G_LANDING FAILED" line is committed. NICK-4:
  Law 8's "draws bit-identical to the canon's stream" overstates the certification — e344 has NO
  draw-identity gate (G_POOL checks shape/mask only); identity is proven by the root-own landing
  read, which is what 6048acd itself claims.

## 2. THE FOLD EDITS — SOUND-WITH-REPAIRS

- T325 banner (26fe3be): "+0.57" == e343 metrics 0.5714. T326 (fab483a): all four x47 numbers
  above traced. T327: 0.57 / 0.986-0.993 / -0.75 / s125 0.124 all in runs/e343/REPORT.md.
  T328: 3-for-3, 1-for-6, R2 4x, s125 residuals — verified in section 4 + metrics.
- Law 7 asymmetric re-word (fab483a diff): 1.06x/0.0586-0.0594/0.0527/~1.1-1.2x all traced.
- Law 8 crosses clause, three forms: v3.1 "THE ERA'S FIRST FINGERPRINT (x45)" -> 26fe3be
  "RETRACTED AT ITS FIRST CROSS" -> ff30b1a "RETRACTED AT ITS CROSSES ... closed every general
  form" — every number in the final form traced (e343/e344 metrics + e343 REPORT).
- METHODS 5 count ticks all reconcile: 18-for-31 (R80) -> 19-for-32 (e342) -> [g1bS9/x45 folds
  carried the count in commits but did not edit the doc — v3.1 caught and disclosed this] ->
  19-for-34 (8336e11) -> 19-for-35 (x47) -> 19-for-36 (e343) -> 19-for-37 (e344). Each tick
  matches that cell's NOTES lean outcome.
- W056/W057 amendment notes: first notes added 26fe3be, second notes ff30b1a; content matches
  the two verdicts.
- NICK-5 (repair R1): METHODS 5 "two-sided p ~ 1.0" is a ONE-sided no-edge probability — for
  1-for-6, P(X>=1)=0.984; the two-sided exact binomial is 0.219. Same mislabel through the
  ~0.6/~0.7/~0.9 series. NICK-6 (repair R2): the discriminator-column sentence stops at e343;
  e344's miss is uncovered by it (T328 itself reads e344's mechanism home as narrowed-to-two,
  while the bar-verdict did close the H-A/H-B fork).

## 3. THE V3.1 RATIFICATION (8336e11) — SOUND

- Law 8 seating: the block moved verbatim out of Law 3's scope into new LAW 8 (diff shows both
  sides); sources named inline (e337/e338/e340, x39, e341, x40, x43, g1bS9, x45); g1bS9 4.6x/28x
  and x45 0.78-0.83 re-traced to metrics; x43 0.26-0.33/s1 0.115->0.427 and e341 gm12 0.688/0.131
  are the long-committed numbers in runs/x43 + runs/e341 records.
- e326 -> Law 2b: 0.6066 == runs/e326/metrics arms/BUDGET-0.0008X/250/gn_clipped 0.60667; e290's
  0.3194 cited in e326's own clause; n-disclosure INTACT in scope ("TWO RUNGS, n=1, ONE CLASS").
- e325 -> Law 6: 0.165 -> 0.275/0.312 == runs/e325/REPORT.md:63-64; maintenance ~0.3 (gate_coreport
  0.3002) vs install-class ~9.2 (REPORT:68-69 + NOTES e325); scope n-disclosures intact.
- Law 3 3a/3b split labeled; 3b = e335's root-is-cons-final 0.745. Epilogue day-fifteen line added
  at 8336e11. Footer disclosure chain extended correctly (and post-v3.1 x47 note appended at fab483a).

## 4. THE COUNTER-COLUMN ARITHMETIC — SOUND (both counts exact)

Re-derived from runs/*/metrics.json `registered` blocks (verbatim leans) + verdicts:
| cell | lab lean | outcome | executor P | contested? | counter |
|------|----------|---------|------------|-----------|---------|
| e342 | SAG-EXISTS weakly | HIT | SAG-EXISTS (concurring) HIT | no | — |
| g1bS9 | DOSE-TRANSFERS weakly | MISS | DOSE-FADES weakly, AGAINST | YES | HIT |
| x45 | ENDPOINTS-ONLY weakly | MISS | ENDPOINTS-ONLY (concurring) MISS | no | — |
| x47 | SLOT-PROPERTY-CONFIRMED weakly | MISS | CONFIRMED (concurring) MISS | no | — |
| e343 | FINGERPRINT-HOLDS weakly | MISS | VANISHES, AGAINST | YES | HIT |
| e344 | CURRICULUM-SIGNATURE moderately | MISS | NAME-ARTIFACT, AGAINST | YES | HIT |
=> lean 1-for-6 (only e342) — VERIFIED; counter 3-for-3 on exactly the three contested calls
(g1bS9, e343, e344) — VERIFIED; cumulative 18+1 for 31+6 = 19-for-37 — VERIFIED. METHODS 5's
claim is accurate as written. Caveat kept by the doc itself: "contested" is executor-selected
(n=3), and the ordering hypothesis is labeled a hypothesis, not a measurement — correct.

## LEDGER RECONCILIATION
6 window cells (e342, g1bS9, x45, x47, e343, e344), 6 NOTES entries, 6 T-cards (T323-T328),
6 runs/ dirs with metrics.json + REPORT.md + PNG (Rule 3 satisfied), 3 METHODS ticks + 3 doc
re-words + 4 W-card amendments — all reconciled; zero unreconciled items.

## REPAIRS RECOMMENDED (for the heartbeat to apply; I edit no ledger)
- R1 (METHODS 5): "two-sided p ~ 1.0" -> "one-sided no-edge p ~ 0.98 (two-sided exact binomial 0.22)".
- R2 (METHODS 5 discriminator sentence): append "e344's bar-verdict closed the H-A/H-B fork while
  the mechanism home stayed two-live — narrowed, not resolved (T328)".
- R3 (Law 8, e344 parenthetical): "draws bit-identical to the canon's stream" -> "stream identity
  proven by the root-own landing read |d| 0.0 (no draw-identity gate ran in this cell)".
- R4 (convention): pin metrics.birth_commit to the cell's OWN birth commit (x45 pinned its smoke;
  e343 pinned the sibling x47's smoke — harmless here, all pre-compute, but mislabeled provenance).
- R5 (g1bS9): reconcile "13/13 gates" with metrics.gates' 11 keys (G_POOL is log-only; G_BITROOT
  logs twice) — amend the REPORT gate list or count "11+2".

Auditor: R81 panel, GLM-5.3, 2026-10-10T22:19Z.
