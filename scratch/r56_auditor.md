# R56 AUDITOR REPORT — the replication ladder under adversarial re-read
(2026-09-29 ~21:20Z; read-only audit; all numbers recomputed from runs/*/metrics.json)

Scope: the three law-grade architectural claims (WALL g1bR, RHYTHM g2d, CONE g3R),
plus g5 store-survivor, g4R honest bounds, g2e root bound, e187, e182. Ledger
cross-check QUEUE/NOTES/THINKING/STATE/git. Verdict scale: SOUND /
SOUNDS-WITH-SCOPE / INVALID-BY-INSTRUMENT / OVERCLAIMED.

---

## 1. g1bR — WALL-REPLICATES: **SOUND**

Recomputed from `runs/g1bR/metrics.json` (adjudication.wall):
- W1_10907 g-12: 0.962/0.803/0.907/0.923/0.916/0.937/0.939/0.906 — min 0.8028@+2, +300 0.9056.
- W1_10908: min 0.7460@+2, +300 0.8999. Controls C10907/C10908 dead: 0.0028/0.0069 @+50
  (0.002-0.014 over +50..+100) — NOTES' numbers match to the digit, including the
  n=3 min band 0.746/0.777/0.803 (ref 10902: 0.7768/0.9181).
- Bars tracked verbatim: maintain = g-12 >= 0.50 at EVERY checkpoint
  {1,2,4,10,50,100,200,300}, die <= 0.27 — g1/g1b's own bars, restated in
  `registered_prediction` with the pre-registered floor band 0.55-0.85 (mins land inside).
  No bar shopping detectable.
- Scoping: T133 (THINKING.md:721-742) carries the full scope — R=0.7 specifically
  (radius-tuned), W2/W3/noise cells n=1, ROOT single, device migrations recorded
  (arms CPU vs reference cuda; margins 0.75-vs-0.50 are device-proof).
- Residual wording risk only: NOTES g1bR's closing line "memory is made
  architectural against displacement, across seeds" — "seeds" here means wash-draw
  seeds; root remains n=1 (T133 says so; the compact line alone does not).

## 2. g2d — RHYTHM-REPLICATES: **SOUNDS-WITH-SCOPE** (one misattributed citation)

Recomputed from `runs/g2d/metrics.json` (adjudication):
- 10903: 10 events, spacings [36,24,24,28,28,32,28,24,24] 100% in [20,45],
  cycle-median 0.587, duty 0.704; 10904: 11 events, 100% in band, median 0.602,
  duty 0.710; reference 0.6149/0.695. Trough texture 0.0814 (10904) matches NOTES.
  All quoted figures verify.
- Bars: events clause (n>=5, 100% in 20-45) + median clause (>=0.5, g2c's
  convention, same grid) — stated as g2b/g2c's registered band, adjudicated per
  seed with explicit pass flags. No shopping.
- Scope present and correct (NOTES g2d "HONEST BOUNDS: n=3 wash-draw realizations
  on ONE root/lineage... not root-seed generality").
- **DEFECT (the reason for the scope qualifier): the causal-loop sentence.**
  NOTES.md:323-325: "THE CAUSAL LOOP IS CLOSED: the onset read trips the replay
  (intervention), the gate-disabled contrast dies (e184's n=3)". The gate-disabled
  contrast is **g2's own cell, n=1** (`runs/g2/metrics.json` purpose: "vs the
  gate-disabled base"; contrast_gate base_ruler_50 = 0.0241, seed 10902). e184
  (ALL-DISSOLVE, n=3 seeds, verified) is the **organ-less** consolidated fact under
  the neutral wash — no organ, no gate. The sentence lends e184's n=3 to a leg that
  is n=1. The loop's other legs (onset trips replay) are g2b/g2c, also n=1
  root/wash-seed. The loop is closed *in kind* at n=1 per leg, not at n=3.

## 3. g3R — SPLIT-REPLICATES / THE CONE: **SOUNDS-WITH-SCOPE** (criterion note)

Recomputed from `runs/g3R/metrics.json`:
- Seed 10905: wash kills at +1 (g0 1.4e-06); lambda sweep 1x = 0.597, 2x = 0.156;
  iso mins 0.883/0.875/0.852 at 1x/2x/4x. Seed 10906: +1 kill (1.3e-05); 0.564/0.096;
  iso mins 0.885/0.879/0.862. Dstore(t*) 0.1318/0.1319 vs g3's 0.13184 — replicates.
  Every number in NOTES/T135 checks ("0.10-0.16 vs 0.85-0.89 at 2x identical L2" ✓).
- Anti-bar-shopping: registered_criteria carry `"dispatch": "e70f44b"` — the
  criteria were committed 20:48Z, metrics landed 20:58Z. `no_bar_shopping: true`;
  ISO-SPARES adjudicated on the MIN of 3 draws (tightened vs g3's every-point rule).
- **The criterion note:** the registered WASH-KILLS bar is "g0 <= 0.27 at lambda in
  {1,2}". g3's original phenomenon as narrated (T126, runs/g3 lambda 1x = 0.238)
  was "the wash direction kills AT 1x". Both replicates MISS the kill at 1x
  (0.597/0.564). So the registered criterion is weaker than the informal original,
  and under it "zero broken legs" is true, while under the "at 1x" reading both
  replicates would have broken. The lab handled this exactly right *after the fact*
  — T135 was sharpened (commit 18821ff), NOTES sharpened (ac6728b), the 1x kill
  relabeled a draw artifact, and the claim re-anchored on the 2x dissociation
  (~0.7 g0 gap at identical L2), which is unambiguous in the data. Verdict: the
  claim AS CURRENTLY WORDED is supported; any reversion to "kills at 1x" in the
  paper would be OVERCLAIMED. The cone's edge along the wash direction is
  draw-sensitive in (1x, 2x) — keep that sentence load-bearing.
- Instrument geometry: clean. Both legs displace the STORE ONLY at matched L2
  (own measured wash direction vs isotropic draws) — matched subspace, same readout.
  The claim is self-scoped "[the store's own]" and T135 reconciles it with
  e185/e187's organism-level isotropic kill ("the whole organism still dies
  isotropic at match"). Root n=1, organ construction n=1 — both stated (NOTES
  honesty block; T135 "the organ-draw scope restated").

## 4. g5 — store-survivor: **SOUND** (n=1, honestly scoped)

Recomputed from `runs/g5/metrics.json`:
- STW (whole-store wall) flat 0.751 -> 0.883 through +300 ✓ ("0.75->0.88"); the
  WQ-arm organ read decays 0.710 -> 0.4996 at +300 — the "misses by 0.0004"
  (0.49964 vs the 0.5 bar) is real and disclosed as not adjudicated.
- The transplant 2x2 (census_after_wash.WQ@+1): washed live retrieval a_fact 0.0831
  vs root 0.9069; wallstate_q (root W_q in WASHED host) x root-K = 0.0832 vs
  wqdrift_q (WASHED W_q in ROOT host) x root-K = 0.8914 — NOTES' numbers exact.
  This cell is the Rule-12 exemplar: the instrument reads at BOTH geometries and
  the attribution (kill site = host stream h, not W_q weights) is interventional.
- Bypass root-store-into-washed-host g0: 6.4e-05 (+1), 4.0e-05 (+50), 3.9e-05
  (+300) — "restores g0 to 0.00004" ✓ (route kill site independent).
- Tax -0.0177 ✓. STORE_SURVIVES only fires for STW; the headline "the store itself
  SURVIVES walled" correctly refers to the whole-store wall. cone_bracket null —
  the unclosed bracket is disclosed. Single seed, radii pre-operationalized from
  g3 checkpoints, transplant probes interventional — all in the honesty block.

## 5. g4R — compass + knife honest bounds: **SOUNDS-WITH-SCOPE** (prose gap, LOW)

Recomputed from `runs/g4R/metrics.json` (verdicts + replicates):
- Compass: r1 PASS (site row 5 in band 5-13, strength 0.376, control_max 0.0,
  A-arm -1.3e-5 inert); r2 site_pos FALSE (band max 0.0136; the near-fact's
  load-bearing rows sit at 0-4; row-3 zeroed readout 0.819 = NOTES' "+0.82").
  Both adjudicated HONEST BOUND exactly as NOTES/T134/QUEUE report. "Substance
  3/3" (P-floor placement, A inert) supported: A inert every draw; install
  carrier_observed = P for all four installs ("spine 4/4" ✓).
- Knife: census-rule N2 kills at ce_delta +0.9701/+1.8509, flat_ce_kill FALSE
  both ("kills at +0.97/+1.85 — kills but NOT flat" ✓). Literal g4 headset
  {L1H2,L3H0}: r1 drop 0.8875 @ +0.1697, r2 0.8673 @ +0.1773, flat TRUE both;
  with g4 draw1 (0.9877 @ +0.179) = 3/3 flat-CE, dCE spread ~0.01 ✓ ("88.8% @
  +0.170; 86.7% @ +0.177" exact).
- Prose gap (LOW): NOTES' r1 line "site +0.376 at rows 5-13" omits that r1's
  near-fact census also loads rows 2-4 (zeroed readouts 0.66-0.99 vs base 0.995) —
  the same split as g4's own draw1 (rows 0-1 strong, site row 5), so the
  convention is consistent; but a reader of NOTES alone would take the binding as
  band-only.
- Scope: install-draw only; root/lineage n=1 (T113 bound) — explicit in
  honesty_reflex. Neither claim minted unqualified, per R55.

## 6. g2e — root replicate: **SOUND** (a model honest-bound entry)

- Fresh-root ruler g+12 = 0.6842 < 0.70 gate (geos 0.4526/0.5832/0.6842); ROOT_BAR
  never lowered; gate_pass false, verdict ROOT-DRAW-BOUND — all as NOTES says.
- Adjudicated cycle-median 0.3883 / duty 0.292 on the frozen (g+12) ruler; g0
  pooling co-report 0.5627/0.599 — the dual-geometry co-report is exactly Rule-12
  practice, and the geo-shift (argmax moved to g+12) is itself reported.
- The "coin-flip zone 0.591-0.711" is pre-registered: registered_prediction cites
  "e157's ported root 0.591" and predicts 0.62-0.78 (landed 0.684). No post-hoc
  zone. 11 events, 100% in band ✓.

## 7. e187 — noise replicates: **SOUND**

- 4/4 cells kill_by2 (worst g-12@+2 8.3e-4 vs 0.27 bar; ANY-SPARES max 2.5e-2 vs
  0.50); displacement-match co-adjudication M=+2, 2.65 >= D_kill 2.489 ✓.
- Orthogonal-direction texture replicates: cos_vs_corpus -0.089..-0.099 at +1 ->
  -0.032..-0.046 after, all four cells — matches NOTES "cos ~ -0.09 -> ~ -0.04".
- Outage recovery provenance documented cell-by-cell (15/16 checkpoints, stream
  md5-matched to e185 10/10, the capped cell re-run bit-identical) — an unusually
  clean provenance chain. Scope: one stream, one root, n=3 draws per arm, stated.

## 8. e182 — GPT-2 texture: **SOUNDS-WITH-SCOPE** (title/QUEUE overstatement)

- Metrics verdict TEXTURE (neither registered bar fired) — the card body is honest:
  +10 retention 0.987/1.004 (verified from traj: lr5e5 0.9869, lr5e6 1.0043), so
  the two-step clock does NOT replicate at 124M; lr5e5 min retention 0.662@+50 with
  bank ppl IMPROVING 43.5 -> 34.7 (ppl_ratio 0.486) — the surgical signature ✓.
- **OVERSTATEMENT 1 (card title):** "the physics translates — wider basin, same
  law, same surgery" — the body itself corrects to "same in DIRECTION, slower in
  time constant (~5-10x on the lr axis)" and explicitly strikes sqrt(P)
  ("NOT proportional sqrt(P) as first read [corrected per the official report]").
  The title reads stronger than the licensed form.
- **OVERSTATEMENT 2 (QUEUE row, stale):** QUEUE.md:203 still says "the basin wider
  ~sqrt(P), the law the same" — contradicting T123's own correction. Stale ledger
  text that would propagate into any assembly.
- Metrics clause defect (LOW): "first checkpoint under the 50% bar: none by +200"
  — the arms were time-capped at steps 108/80 with three measured checkpoints
  (2/10/50); +200 was not reached (trims record the caps; R55 lists "e182's +200
  horizon" as owed; the commit message and NOTES are honest about it). The clause
  should say "none within the realized horizon".
- Geometry: the 2-shot-context conflation (fact storage vs task-following) is
  flagged but unadjudicated; probe-selection bias toward RESISTANCE is stated
  (a kill on this battery is strong; resistance is an upper bound) — good Rule-12
  hygiene. Size envelope compliant (124M <= 500M with the stated external-validity
  reason). Compute note: per-training cap 1800s / ~893s walls recorded (eval-heavy);
  outside the 180s preference but documented and dispatch-sanctioned.

---

## Ledger integrity (cross-file)

- **QUEUE vs runs:** all eight audited rows (g3R:190, g4R:191, g1bR:192, g2e:193,
  g2d:195, g5:201, e187:188, e182:203) say DONE and every one has metrics.json +
  PNG on disk and committed (git ls-files 8/8). g2f row (194, DISPATCHED 21:04Z)
  matches STATE and heartbeat. No status drift found.
- **Claimed-but-missing markers:** none. T135 exists (THINKING.md:679); R55 exists
  (REVIEWS.md:76 — the n>=3-before-law-grade rule and "no 'rhythm' without seeds");
  e098_base_s4306.pt on disk; g2c/g1b/g3/g2 reference metrics all present; g3_gen.pt
  reused with provenance gates (host bit-identical to base, root g0 reproduces,
  store-off pass).
- **STATE:** updated live during this audit (fleet line "Fleet 0" -> "Fleet 4",
  heartbeat 21:04Z) — live bookkeeping, not drift. Minor: last_review/last_novelty
  18:29:47Z vs Review 55's own stamp 18:45Z (16-min mismatch); review was overdue
  at audit time and is being discharged by this R56 trio.
- **The one systemic caveat:** "THREE law-grade architectural claims" (STATE's
  compact line, echoed in NOTES g3R and T135) rests on n=3 **wash-draw** seeds per
  claim, each on ONE root/lineage. The README meta-law reserves mechanism-grade
  for >=3 NETS; R55 demanded seeds; the replicates delivered the seed axis only.
  T133/T131/T135 all carry the root/organ-draw scope correctly — the compact forms
  (STATE, commit messages, any paper abstract) must carry "(wash-draw n=3, one
  root)" or the scoping is lost in transmission. g2f (in flight) tests exactly the
  rhythm's root axis; the wall's and cone's root redraws remain unqueued.

## Instrument-geometry sweep (the e166 lesson)

No INVALID-BY-INSTRUMENT findings. g5's transplant 2x2 is the exemplar (reads the
putative store at both host geometries; the W_q attribution flipped only because
the instrument crossed geometries). g2e co-reports the frozen-ruler adjudication
and g0 pooling. e182 flags its few-shot conflation. g3R's dissociation shares one
readout across both legs (matched subspace, matched L2). g1bR/g2d reuse the
reference batteries bit-gated.

---

## Prioritized repairs (anchored old-text -> new-text)

1. **(HIGH) g2d's causal-loop citation.** NOTES.md:323-325
   old: "the gate-disabled contrast dies (e184's n=3)"
   new: "the gate-disabled contrast dies (g2's own base cell, n=1 seed 10902;
   e184's n=3 is the organ-less analogue — every leg of the loop is n=1 in its
   own cell; the loop is closed in kind, not yet in n)"
2. **(HIGH) e182 QUEUE row stale sqrt(P).** QUEUE.md:203
   old: "the basin wider ~sqrt(P), the law the same"
   new: "the basin wider (~5-10x slower on the lr axis, NOT sqrt(P) — T123
   correction); the law the same in direction only; the two-step clock does not
   replicate at 124M"
3. **(MED) Scope-carrying compactions.** Wherever "law-grade"/"n=3" is compacted
   (STATE fleet lines, commit messages, paper abstracts), append "(wash-draw n=3,
   one root/organ)". Specifically STATE's "THREE law-grade architectural claims"
   and NOTES g3R's "THIRD ARCHITECTURAL CLAIM AT LAW GRADE" should read
   "...at law grade (wash-draw n=3, one root)". Also queue the wall's and cone's
   root-redraw cells alongside g2f.
4. **(MED) e182 metrics clause horizon.** runs/e182/metrics.json adjudication
   old: "first checkpoint under the 50% bar: lr 5e-06: none by +200, lr 5e-05: none by +200"
   new: "none within the realized horizon (checkpoints {2,10,50}; arms time-capped
   at steps 108/80 — the +200 cell owed per R55)"
5. **(LOW) g4R r1 prose.** NOTES g4R r1 clause: after "site +0.376 at rows 5-13,
   A inert" add "(the near-fact's census also loads rows 2-4 at 0.66-0.99
   zeroed-readout — same split as g4's own draw1; the site leg is the in-band row)".
6. **(LOW) STATE timestamp hygiene:** set last_review to the review entry's own
   stamp (18:45Z for R55) when resetting, so the log and the clock agree.

## Verdict table

| Claim | Verdict | One-line basis |
|---|---|---|
| g1bR WALL-REPLICATES | SOUND | every checkpoint number recomputed; bars verbatim; scope in T133 |
| g2d RHYTHM-REPLICATES | SOUNDS-WITH-SCOPE | numbers verify; the "gate-disabled (e184 n=3)" cite misattributes n |
| g3R SPLIT-REPLICATES / CONE | SOUNDS-WITH-SCOPE | 2x dissociation rock-solid and pre-registered; the 1x kill was draw-luck — never restate "kills at 1x" |
| g5 store-survivor | SOUND | STW flat 0.75->0.88 verified; transplant 2x2 exact; 0.4996 miss honestly unadjudicated; n=1 scoped |
| g4R compass/knife | SOUNDS-WITH-SCOPE | honest bounds match metrics; headset 3/3 flat-CE exact; r1 census prose gap |
| g2e ROOT-DRAW-BOUND | SOUND | gate failure and co-reports verified; zone pre-registered |
| e187 NOISE-KILLS-REPLICATES | SOUND | 4/4 kills, cosines, displacement-match all verify; provenance exemplary |
| e182 GPT-2 TEXTURE | SOUNDS-WITH-SCOPE | body honest; card title + QUEUE row overstate ("same law", "~sqrt(P)"); "+200" clause overstates horizon |
