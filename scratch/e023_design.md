# e023 design — surgical forgetting at entity granularity: letter-row surgery vs entity-granular ascent

Status: DESIGN (implementable as `lab/e023_surgical_forgetting.py`). Date: 2026-09-24.
Answers the "next family" pointer in card claim **C5** (closed negative: first-order
ascent cannot content-selectively forget; r = 1.08 memorization-symmetric at the
0.66-nat bar) and cashes the tombstone inheritance (README History: the genome era's
one survivor — *output lm_head token-row directions are causal training coordinates*).
Builds on E001 (host net), E003c (ascent protocol + r-metric conventions), E011c
(zero/rotate/random perturbation ladder doctrine: damage tracks energy, exceptions are
content), E013 (16-token local sufficiency — name completion is a local task).

**Question.** Can a memorized ENTITY (a proper name) be surgically removed from the
2.7M char-transformer with near-zero collateral, and does the answer localize entity
knowledge in the I/O row space (wte read-rows / lm_head write-rows) rather than in
the body? Arm C answers the control question: does restricting first-order ascent to
entity-containing contexts — the finest first-order instrument there is — buy the
selectivity that corpus-level ascent (C5) lacked?

**Design probes already run (2026-09-24, eval-only on `runs/checkpoints/e001.pt`,
no files written; all numbers below are measured, marked MP):** name batteries,
zero-J surgery preview, full-set zero preview, generation baselines, row geometry.
These ground the predictions; the registered experiment re-measures everything on
the full frozen batteries.

---

## 1. TARGET — JULIET (the J-name family), PROSPERO as the forgotten-name anchor

**Why not PROSPERO:** The Tempest lives in the LAST 10% of `data/input.txt` = the
val split. PROSPERO has **0 train / 63 val** occurrences (MP: NLL 13.23 nats/char,
argmax acc 0.003 — the model never knew it). There is nothing to forget. PROSPERO is
instead the **anchor**: what a truly forgotten name looks like on this net
(13.2 nats/char; note uniform = ln 65 = 4.17 — an unknown name is *far* below
uniform because uppercase continuations are confidently suppressed).

**Target: JULIET** — 125 train occurrences (Romeo & Juliet region, corpus positions
458,053–886,378), **0 val** (clean train/val split; the readout is a *memorization*
channel, exactly like e003c's train-A). MP baselines (ctx 120, per-position argmax
acc): **NLL/char 0.35, acc 0.91, per-position acc [0.43, 0.97, 1.00, 1.00, 1.00,
1.00]** — the name is deeply memorized from position 2 onward: 12.9 nats/char of
headroom above the anchor.

**Control names (frozen battery, letter-overlap o vs the target letter set
{J,U,L,I,E,T}):**

| name | train occ (val) | NLL/char (MP) | overlap letters | o |
|---|---|---|---|---|
| JULIET | 125 (0) | 0.35 | — (target) | — |
| JOHN | 33 (0) | 1.09 | {J} | 0.25 |
| ROMEO | 163 (0) | 0.24 | {E} | 0.25 |
| GLOUCESTER | 229 (0) | 0.08 | {E,L,T,U} | 0.44 |
| MENENIUS | 162 (0) | 0.28 | {E,I,U} | 0.50 |
| CORIOLANUS | 150 (0) | 0.19 | {I,L,U} | 0.33 |
| ISABELLA | 129 (0) | — (measure in-run) | {E,I,L} | 0.50 |
| LUCIO | 111 (0) | — | {I,L,U} | 0.60 |
| PETRUCHIO | 21 (137) | 13.26 on VAL occ (MP) | {E,I,T,U} | 0.44 |
| PROSPERO | 0 (63) | 13.23 (anchor) | {P,R,O,S,E} | 0.17 |

Observational bonus R0 (in-run, free): name memorization dose-response across the
battery — 21 exposures (PETRUCHIO train) did NOT memorize; 33 (JOHN) → 1.09; 125
(JULIET) → 0.35; 229 (GLOUCESTER) → 0.08. (Axis caveat: PETRUCHIO 13.26 is measured
on val occurrences = generalization axis; the in-run battery adds its 21 train
occurrences to fix the axis.)

**The J-census (collateral class, train):** J = 312 chars = **0.031% of train text**.
J-words: JULIET 125, Juliet 48, JOHN 33, John 20, Jove 15, Jack 7, Jesu 7, Justice
7, Jupiter 5, Join 4, Juno 3, Julius 3 (277 of 312 J chars in 12 words; the only
uppercase letters rarer: Q 230, Z 161, X 112). **The honest entity statement: row
surgery at char granularity can target "the J-name family" (JULIET + John/Jove/
Jack/…), not JULIET alone — JOHN is the registered same-letter control, not a
confound to be hidden.**

## 2. THE GRANULARITY PROBLEM (owned up front)

Vocab = 65 chars; an entity has NO dedicated parameters. "The name's rows" = its
letter rows (shared with all text) + diffuse body-circuit transitions. So arm A is a
**depth ladder**:

- **D1 — full letter set** {J,U,L,I,E,T}: 6 rows × 2 matrices = 2,304 params. What
  naive "name-token row surgery" means at char level. MP preview: a BOMB — val CE
  1.5975 → 1.9936 (**+0.396 nats** corpus collateral), ROMEO acc 0.94 → 0.57,
  JULIET dead (NLL 8.30, acc 0.00). The letters U/L/I/E/T carry the pronoun I,
  line-initial T/E, and every name sharing them.
- **D2 — rare-letter core** {J}: 1 row × 2 matrices = **384 params of 2.74M**. MP
  preview: a SCALPEL — zero both rows: JULIET NLL 0.35 → 7.36 (acc 0.91 → 0.14),
  JOHN dead, ROMEO flat (0.24 → 0.24), val CE **+0.0009**. JULIET owns exactly one
  row that (almost) no other name uses.

The ladder IS the experiment: it measures how entity-selectivity of row surgery
scales with row ownership. Predicted answer: selectivity is set by the rare core,
not the full set.

## 3. ARMS

All surgery arms are eval-only `state_dict` surgery on a deepcopy of E001 (assert:
exactly the 12 (D1) or 2 (D2) target row entries differ from base; all other params
bit-identical). Untied note: wte row = the READ path (what the net sees when the
letter is in context), lm_head row = the WRITE path (the letter's logit direction —
the tombstone's causal coordinate). The wte/lm split is a first-class sub-arm.

**Arm A — row surgery ladder** (cells: D × variant × matrix):

| cell | operation on each target row r | e011c rung |
|---|---|---|
| zero | r ← 0 | remove (energy −‖r‖) |
| resample | r ← fresh N(0, 0.02²) draw (seed 23100+i) | random content, init-norm |
| shuffle | r ← Q's row, Q's row ← r (J↔Q swap; lm cos(J,Q) = 0.71, Q is QUEEN's private letter — nearest private-letter neighbor) | in-class content swap |
| proj (arm B) | see below | in-manifold identity removal |

- A/D2 (9 cells): {zero, resample, shuffle} × {both, wte, lm}.
- A/D1 (3 cells): {zero, resample, shuffle} × {both}. (D1 wte/lm split registered as
  e023b if the D2 split is informative.)
- Generation probes on 4 key cells only (D2-zero-both, D2-resample-lm,
  D2-shuffle-both, D1-zero-both) + base.

**Arm B — direction (span-projection) surgery:** r′ = Proj_{span(S)} r where S =
the 25 non-J uppercase letter rows (D2), or the 20 non-target letters (D1), applied
to both matrices (D2 wte-only/lm-only if time). The row stays a *generic letter*
(a linear blend of other letters — on-manifold) while its J-specific component is
deleted. Measured geometry (MP): the letter rows are near-orthogonal — the J
residual outside the span is **67.4% (wte) / 60.2% (lm)** of ‖r‖ (‖r_J‖: wte 0.931
vs mean row 0.754; lm 1.299 vs mean 1.382). So D2-proj keeps ~35% norm along shared
letter directions: a *shrunk generic-letter* surgery between no-op and zero. It
tests whether the entity coordinate is the row's private component (then proj
erases like zero) or the shared component (then proj spares the name).

**Arm C — entity-granular ascent (the finest first-order attempt):** e003c protocol
verbatim — AdamW betas (0.9, 0.95), lr 1e-5, grad-clip 1.0, batch 32×256, up to 600
steps from E001 — but batches are drawn ONLY from 256-char train windows containing
≥1 "JULIET" occurrence (window index built once, seed 23001; sampler generator seed
23002; with replacement). Loss = −CE on the whole window (entity CONTEXT, not entity
chars: J-positions are ~6/256 of each window — that is the point: this is the best
first-order can do at char granularity). Eval every 50 steps; early stop when
JULIET NLL ≥ 13.2 (anchor level) or step 600. Checkpoint model+opt+gen-state every
100 steps to `runs/checkpoints/e023_ascent.pt` (Rule 10 resume discipline).

## 4. READOUTS (all deterministic; batteries built once, seed-stamped)

- **R1 name battery:** every train occurrence (≤125) of the 10 battery names, ctx
  120, one forward each: per-position NLL + argmax acc. Report NLL/char, acc, and
  JULIET split first-char vs chars-2+. Plus the 12-word J-census battery (all
  J-words, NLL/acc each).
- **R2 corpus CE (fixed blocks, e003c convention):** (a) val-All 400 blocks
  (seed 202); (b) val blocks containing no J char (J-free fluency); (c) 125
  target-text windows anchored at JULIET occurrences vs 125 matched windows
  anchored at ROMEO occurrences (paired local-fluency channels); (d) PETRUCHIO-
  containing vs PETRUCHIO-free val blocks (held-out name-containing vs name-free
  collateral, the task's required axis — PETRUCHIO has 137 val occurrences and no J).
- **R3 generation probes:** 12 fixed prompts (6 R&J-region train offsets, 6 generic
  train offsets), 350 tokens, temp 0.8, top-k 40, torch seed = prompt index. Count
  JULIET / ROMEO / JOHN / J-census words / total J chars. MP baseline (8 R&J
  prompts × 400 tok, seed 7): **JULIET 6, ROMEO 6, Nurse 7 per 3,200 chars**
  (nonzero baseline ⇒ headroom to vanish); generic prompts: JULIET 0, ROMEO 4 per
  1,800 chars.
- **R4 row geometry (descriptive, feeds P2):** ‖r‖, cos(r, other letters),
  projection residuals per target row.

## 5. SELECTIVITY METRICS (frozen before running)

All deltas vs the same run's baseline (gate G0). Significance floors: ΔCE_corpus
0.01 nats; ΔNLL_name 0.10 nats/char (batteries are fixed ⇒ deltas are exact, not
noisy; floors encode practical meaning).

- **S_corpus = ΔNLL_JULIET / ΔCE_val-All** (both nats). If ΔCE_val < 0.01: report
  "S_corpus > ΔNLL_JULIET/0.01" (lower bound). MP preview for D2-zero-both:
  7.01/0.0009 ⇒ > 780.
- **S_name = ΔNLL_JULIET / mean(ΔNLL_pure controls)**, pure controls = ROMEO,
  GLOUCESTER, CORIOLANUS (lowest overlap, 0.25–0.44). **THE entity-selectivity
  metric. Bar: S_name ≥ 5 counts as selective.**
- **S_letter = ΔNLL_JULIET / ΔNLL_JOHN** — entity-vs-letter granularity. Rows
  cannot split the J family: registered surgery expectation **S_letter ≈ 1**
  (preview: 7.01/7.02). This is the honest granularity limit of row surgery.
- **r_entity (ascent only) = ΔNLL_JULIET / ΔCE_val-All** at the entity forgetting
  bar (below) — directly comparable to C5's r = 1.08–1.43.
- **Forgetting bars:** Bar-1 (partial) = JULIET NLL ≥ +2.0 nats/char (≥ 2.35);
  Bar-2 (erasure) = acc ≤ 0.10 AND NLL ≥ ln 65 = 4.17. Surgery verdicts on Bar-2;
  ascent on Bar-1 (ascent reaching Bar-2 would itself be a headline).

## 6. REGISTERED PREDICTIONS (3)

**P1 — surgery erases at the rare core with near-zero collateral; the full letter
set is anti-selective.** (a) D2-zero-both reaches Bar-2 (preview NLL 7.36 ✓,
acc 0.14 — full battery must land ≤ 0.10; if acc lands 0.10–0.30 the verdict is
"damaged, not erased" and resample-lm is the erasure cell) with ΔCE_val ≤ 0.01 and
S_name > 50 (preview: ROMEO Δ ≈ 0.001). (b) D1-zero-both ALSO erases JULIET
(preview 8.30 ✓) but FAILS selectivity: ΔCE_val ≥ 0.30 (preview +0.396) and S_name
< 5 (preview: ROMEO 0.24 → 2.57 ⇒ S_name ≈ 3.4). Verdict rule: **"row surgery is
selective exactly insofar as the entity owns its rows"** — confirmed iff (a) and
(b) both hold.

**P2 — collateral is predicted by shared row structure (the metric the task asks
for).** After D1-zero-both: Spearman ≥ 0.8 between ΔNLL_control and overlap o
across the 8 control names (o range 0.25–0.60). After D2 (any variant): the 12
J-census words take ΔNLL ≥ 1.0 while every non-J name takes ≤ 0.05 — collateral is
the letter class, nothing else. **P2b (untied split, tombstone test):** lm-head-
only does the WRITE damage (J logit killed ⇒ J vanishes from generation R3 even
while battery acc stays high — preview: lm-zero NLL 2.02 but acc 0.83, the zero-row
argmax artifact: logit 0 still beats negative logits), wte-only does the READING
damage (context J becomes invisible; preview: wte-zero NLL 5.75 ≫ lm-zero 2.02,
acc 0.21). **P2c:** resample-lm kills the argmax artifact (acc ≤ 0.10 — a live
random logit beats J nowhere it mattered), i.e. resample, not zero, is the honest
erasure variant; shuffle-both produces Q-substitutions (generation emits Q-names
where J-names were). **P2d (arm B):** D2-proj erases (Bar-2) iff the J-specific
residual component carries the identity — registered both ways: if proj spares the
name (NLL stays < 1.0), the entity coordinate is the SHARED letter component and
the tombstone's "row direction is causal" refines to "shared-row direction".

**P3 — entity-granular ascent fails like corpus ascent.** At Bar-1 (+2.0 nats/char
JULIET): **r_entity < 1.3** (C5 band 1.08–1.23) AND S_name < 2 (control names
damaged comparably — memorization-symmetric damage now at entity granularity),
with target-text windows (R2c) rising ≥ half as fast as the JULIET battery itself
(ascent burns the shared local fluency of the R&J region, not the name). Sub-
prediction: ascent's damage concentrates on the J ROW (logit-J falls) — same
channel as surgery — but arrives with corpus collateral ≥ 30× surgery's at equal
target damage (surgery at +7.0 nats/char costs +0.0009 val CE; ascent at +2.0 is
predicted to cost ≥ 0.05).

## 7. WHAT REVIVES FIRST-ORDER vs CONFIRMS SURGERY-ONLY

**Revives first-order (any one suffices; each reopens C5 with a specific edit):**
1. r_entity ≥ 1.3 at Bar-1 with ΔCE_val ≤ 0.10 — context restriction (data
   granularity), not gradient modification, was the missing ingredient.
2. An early-selective transient: within 100 steps, ΔNLL_JULIET ≥ +2.0 while
   ΔCE_val < 0.05 and mean ΔNLL_pure-controls < 0.3 — an exploitable window before
   the C5 collapse (early-stopped entity ascent as an instrument).
3. Ascent reaches Bar-2 (uniform-floor erasure) at any dose with ΔCE_val ≤ +0.5.

**Confirms surgery as the only selective family:** P1 ∧ P3 (with P2 locating the
collateral). Card v2 gains candidate C6: *entity knowledge localizes in I/O row
space at LETTER granularity (rare-letter core; S_letter ≈ 1 — rows cannot split a
letter family); first-order fails even at entity granularity. Forgetting an entity
without its letter-mates requires body-circuit surgery or second-order directions
(e024+).*

**Informative third outcome (P3 fails toward symmetry, ascent dies on the letter
too):** ascent raises JOHN ≈ JULIET (S_letter ≈ 1) AND surgery S_letter ≈ 1 ⇒
NEITHER instrument is entity-granular; entity identity lives in body transitions
the rows only index. Next step becomes causal body surgery (rank attention/MLP
sites by JULIET-context selectivity, graft/zap) — the e038 playbook at name
positions.

## 8. BUDGET, DETERMINISM, CHECKPOINTS

Measured on the RTX 5090 laptop (design probes): battery pass ≈ 10–15 s; CE
batteries ≈ 2 s; generation 12×350 ≈ 34 s; ascent step ≈ 0.25 s.

| phase | cost |
|---|---|
| P0 setup, batteries, baselines (G0–G3) | ~1.0 min |
| Arm A+B: 12 cells × ~15 s + generation on 4 cells + base | ~5.5 min |
| Arm C: 600 steps (2.5 min) + 12 evals × ~12 s + final generation | ~5.5 min |
| **total** | **~12 min (≤ 15 hard cap; 3 min slack)** |

Registered fallbacks if slow: generation → 8 prompts × 300 tok (−1.5 min); Arm C
eval every 75 steps and cap 400 steps (−1.8 min); drop A/D1-resample and -shuffle
(−0.5 min). Determinism: fixed occurrence lists; fixed eval blocks via seeded
generators (e003c convention seeds 101/202/303/404 + 23xxx for new batteries);
`set_seed(23000)`; per-prompt generation seeds; surgery cells asserted-local
(G2: bitwise). Note the e003c honesty caveat: bit-determinism holds per
machine/kernel build; across builds, verdicts (thresholds) — not trajectories — are
the registered object. Checkpoints: ascent arm snapshots every 100 steps
(resumable); save the D2-zero-both surgery state_dict for follow-ups; everything
else is specs + metrics. Outputs: `runs/e023/metrics.json`,
`surgery_ladder.png` (S_name/S_letter/S_corpus by cell),
`ascent_traj.png` (r_entity(dose) with the 1.3/2.0 bars), `probes.txt`
(generations), NOTES.md entry after the run.

## 9. VERIFICATION GATES

- **G0:** reloaded E001 val CE within 0.03 of its training history (1.622).
- **G1:** two consecutive baseline battery calls bit-identical.
- **G2:** per surgery cell: `(surg_sd != base_sd).sum()` equals exactly the target
  row entries (2 / 4 / 12 / 24 entries), all others equal.
- **G3:** ascent step-0 eval == baseline battery (same code path).
- **G4:** PROSPERO anchor NLL in [10, 16] (sanity of the battery construction).

## 10. PSEUDOCODE

```python
# batteries (once): occ lists via regex on train/val text; fixed blocks via
# torch.Generator seeds 23101..23112; name_battery(model, occs) -> {nll, acc,
# per_pos}; j_census(model); gen_probes(model, n=12) with per-prompt seeds.
# surgery(model, rows, variant): deepcopy state_dict; zero | resample(seed 23100+i)
#   | shuffle(J<->Q) | proj (lstsq onto non-J uppercase span); assert G2.
# for cell in CELLS: sd = surgery(...); log R1+R2 (+R3 on the 4 key cells); restore.
# arm C: window index of JULIET-containing train windows (seed 23001); AdamW e003c
#   protocol; loop steps 1..600: batch from index (gen 23002); loss=-CE; clip;
#   step; every 50: full R1+R2; every 100: ckpt. r_entity(dose) at Bar-1 crossing.
```

## 11. HONESTY CAVEATS (pre-registered)

- JULIET readout is on TRAIN text (memorization channel — deliberate, mirrors
  e003c train-A; the name has no val occurrences; PETRUCHIO val battery covers the
  held-out axis).
- Zero-row argmax artifact (preview: acc 0.83 at NLL 2.02): never report zero-cell
  accuracy without NLL; Bar-2 requires both.
- S_letter ≈ 1 is a STRUCTURAL limit of row surgery at char granularity, not a
  failure to be spun; the J-family framing is the honest target class.
- Generation counts are small integers; report exact counts + per-10k-char rates;
  the vanish verdict uses the 6 fixed R&J prompts where the baseline is 6.
- Arm C ascent windows contain other R&J text (unavoidable at char granularity) —
  that is part of the instrument being tested, and R2c measures it.
- PROSPERO anchor contexts straddle the train/val boundary; contexts are inputs
  only, no leakage concern for a forgetting readout.
