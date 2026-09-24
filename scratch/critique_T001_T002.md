# Adversarial critique of T001 / T002 (independent review, 2026-09-24)

Reviewer read: THINKING.md, NOTES.md, runs/{e001,e002,e011a}/metrics.json, lab/common.py,
lab/e001_lesion_map.py, lab/e002_forgetting.py, lab/e011a_write_norms.py, runs/e002/probes.txt,
runs/e001_run.log, and the raw corpus. All numbers below were recomputed from the JSONs and the
corpus file unless stated otherwise.

---

## Missed alternatives

**M1 — The E002 evaluation sets are positionally cross-labeled (biggest miss).**
`CharCorpus` splits train = chars 0–90%, val = 90–100%. Then `val_a = slice("val", 0.0, 0.5)`
= corpus positions **90–95%** and `val_b = slice("val", 0.5, 1.0)` = positions **95–100%**, while
`train_a` = 0–45% and `train_b` = 45–90%. So "held-out A-text" is text from the *late-corpus
region adjacent to B*, not from A's (early-corpus) distribution. Both curves ("A" and "B") measure
collateral damage on neighboring late-corpus plays; **neither measures the held-out loss of the
material actually being ascended** (0–45%). The A-vs-B contrast the whole experiment is built on is
a 90–95% vs 95–100% contrast. None of T002's H1–H4 includes this.

**M2 — The probe prompts don't probe what they claim.** Verified in the raw corpus: `PROSPERO:`
occurs 63 times, all at positions 97.2–99.2% — i.e., **entirely inside the val set, never trained
on**; `ROMEO:` occurs at 40.3–50.7%, straddling the 45% A/B boundary. The "A prompt vs B prompt"
behavioral check contrasts a boundary-straddling name against a name the model never saw in
training. Any probe conclusion from probes.txt is uninterpretable as A/B memory.

**M3 — Baseline difficulty asymmetry.** Baseline CE: A 1.513 vs B 1.685 — a 0.17-nat gap before
any intervention. ΔA vs ΔB comparisons (including "B_damage_at_A_plus_1nat") conflate selectivity
with the two sets' different difficulty/dynamic range. No difficulty-matching or relative-damage
normalization anywhere.

**M4 — Grid-quantization artifact in the headline "anti-selectivity" number.** The metric is read
at the first 20-step grid point where ΔA ≥ 1.0, but there ΔA = 1.63 (step 20: A 3.140, B 3.197).
Linear interpolation puts ΔB at ΔA = 1.0 at **0.93 < 1.0** — collateral slightly *behind* target in
the naive arm, which is ahead at every naive grid point (step 400: ΔA 26.06 vs ΔB 25.90). Genuine
"collateral ahead of target" (ΔB > ΔA) occurs only in the **anchored** arm, steps ~120–200 (step
160: ΔA 2.91 vs ΔB 2.95; step 180: ΔA 5.46 vs ΔB 5.52) — a real and interesting phenomenon (the
retain anchor is net-positive damage to B in that window) that NOTES/T002 never mention.

**M5 — Optimizer artifacts in E002, beyond "dose too high".** (a) Fresh Adam state, no warmup:
early steps are ~sign-ascent at maximal effective step size — the least selective instrument
available. (b) `clip_grad_norm_(1.0)` on an exploding ascent objective turns every post-explosion
step into a fixed-size normalized direction walk; "dose" is lr-only, and the H1 "lr explodes"
framing misses that the step *direction* is the problem. (c) The anchored objective −CE_A + CE_B has
no equilibrium; its late curve oscillates ±1 nat per eval (loss_A 8.29 → 7.99 → 8.41 → 9.25 → 9.97
→ 8.28 over steps 200–300), making "final Δ" a stopping-step lottery (ΔA = 8.97 at step 280 vs 6.38
at step 400 — a 2.6-nat swing). (d) AdamW in e002 uses default weight_decay = 0.01 vs 0.1 in
training — unregistered config drift (probably negligible at lr 2e-5, but it should have been a
conscious choice).

**M6 — The interesting dynamics were never sampled.** Uniform-random CE over 65 chars is
ln 65 = 4.17. The naive arm passed through random between steps 20 (CE 3.14) and 40 (CE 12.94);
~94% of the 400 steps measured post-random pathology. Eval every 20 steps means the entire
forgetting-vs-destruction transition was observed through a single data point (step 20).

**M7 — Held-out CE cannot show memory deletion.** val_a was never trained on; it was never
memorized. ΔA rising measures damage to region-level statistics (style/fluency of late-corpus
text), not removal of train-A memories. The cheap unused control: track the **train-A vs val-A gap**
(memorization gap) during ascent — if train-A loss rises while the gap collapses, memory is being
deleted; if both rise in lockstep, it's distribution-level damage (this would directly discriminate
H2 from H3).

**M8 — Single seed, single run, truncated schedule (E001).** Seed 42, one model. Training stopped
by `max_seconds=240` at ~2000 of 4000 steps, i.e., mid-cosine at lr ≈ 0.5e-3, with val already
0.07 above its reported 1.55 minimum. The lesion map describes an *unconverged, un-annealed,
slightly overfit* snapshot. The e004 ladder varies size, not seed; nothing in the plan tests whether
"MLP-0 keystone" or "attention strictly monotone" is seed-stable at all.

**M9 — Positional val split + partial coverage.** E001 damage is estimated on the last 10% of the
file (one specific region of late plays), via 384 random 256-char windows over a 111,539-char val
set (~59% expected token coverage). Deterministic and paired — good for ranking — but "dispensable"
means "dispensable for this region and this realization". The L0H4 lesion *improves* val loss
(−0.0032), calibrating realization noise at ~±0.005; the 0.02 "harmless" threshold is only ~4×
that floor.

**M10 — Head vs block eval precision mismatch.** Heads were evaluated with n_batches=24, blocks and
baseline with 30. Same generator seed makes the 24-batch eval a strict prefix of the 30-batch
sequence, but block-vs-head damage comparisons still carry a small unquantified subset offset
(prefix-mean vs full-mean), relevant at the 0.02 threshold.

**M11 — The hook is exact, but "marginal damage" is not "necessity" (T001's framing gap).**
Implementation check: `c_proj` is a bias-free linear map applied to concatenated head outputs, so
zeroing head h's input slice subtracts exactly W_proj[:,slice]·h_head — the head ablation is
mathematically identical to removing that head's write. Same for zeroing attn/mlp module outputs.
The real gap: single-component zeroing measures *marginal* contribution with the other 47 components
intact. Numbers: L0 head damages sum to **0.316 vs attn-0 block damage 2.401 (7.6× superadditivity)**;
**all 36 heads summed (2.10) < attn-0 alone (2.40)**. "16/48 dispensable" and "late attention dead"
are claims about a redundant ensemble, not about vestigial organs — and H3's own planned test
(cumulative L4+L5) targets the wrong layers; the redundancy signal lives in L0–L2 heads.

**M12 — H4 was "refuted" with the wrong quantity.** E011a measured *absolute* write norms. The
pre-LN mechanism H4 actually proposes is about *relative* perturbation of the LayerNormed stream.
Embedding stream norm at block input ≈ 0.02·√192·√2 ≈ 0.39; quadrature accumulation of the measured
writes gives stream ≈ 10.7 before L5's attention. So attn-0's 2.72 write enters a ~0.4-norm stream
(~7× relative perturbation) while attn-5's 1.93 enters a ~10.7-norm stream (~0.18×) — a ~40× falloff
in *relative* perturbation with flat absolute norms. Likewise a large write nearly parallel to the
stream (MLP-5: norm 5.64, damage 0.59) changes LN'd direction little. Damage vs absolute write norm
cannot refute H4; damage vs orthogonal-innovation-to-stream could.

**M13 — Tokenization/corpus trap in the planned "dissimilar splice".** `data/input_french.txt` is
Les Misérables: 3.25M chars, vocab **118** vs 65. The dissimilar arm cannot reuse the E001
checkpoint or tokenizer; it needs a union vocab and retraining, and char-level CE baselines differ
by language entropy, so ΔA/ΔB magnitudes won't be commensurable across arms without normalization.
(`data/input_reversed.txt` shares the 65-char vocab and would be a tokenizer-matched dissimilar
control nobody has mentioned.)

---

## Prediction audit

- **T001-P1 ("write norms will NOT decline monotonically")** — trivially likely; the journal itself
  states the base rate ("they usually grow or stay flat in trained residual nets"). And the coupled
  clause "H4 will NOT **fully** explain" is unfalsifiable: "fully" is undefined, so any leftover
  role for H4 counts as a win. Right-for-wrong-reason risk: it already "won" on a measurement
  (absolute norms) that cannot test H4's actual mechanism (M12).
- **T001-P2 ("LN-only recalibration recovers ≥30% of MLP-0's +4.08")** — sharp, quantitative,
  genuinely at risk; the best prediction in the file. Weakness: no preregistered recalibration lr,
  and "recovery fraction" depends on the 200-step budget — specify lr or the 30% threshold is
  adjustable after the fact.
- **T002-P1 ("grad cosine A↔B ≥ 0.85 ⇒ H2 structural")** — **most at risk of being right for the
  wrong reason.** Any two same-corpus batches have near-parallel gradients because per-token
  gradients are dominated by shared unigram/bigram structure; cos(A,B) ≈ cos(B1,B2) ≈ high whether
  or not any A-specific memory exists. As registered, it will near-certainly "confirm" H2 while
  showing only that gradients are frequency-dominated. Without a within-half null (see below) this
  prediction is close to unfalsifiable in the confirming direction.
- **T002-P2 ("no lr achieves ΔA ≥ 1 with ΔB ≤ 0.1")** — falsifiable and meaningful, but its readout
  ΔA is the mislabeled 90–95% region (M1), so even a "pass" wouldn't mean what the journal says.
- **T002-P3 ("selectivity WILL appear for the dissimilar splice")** — one-sided: no quantified
  criterion for "appears" (what ΔB at ΔA = 1 counts as selective?), plus the vocab/retraining trap
  (M13). Also confirmatory-framed: the failure branch is described, but no number is staked to it.

---

## Hypothesis verdicts

**T001**
- **H1 (early attention does the work): needs-another-discriminator** — block zeroing conflates
  function with off-manifold shift and within-layer redundancy (L0: block 2.40 vs head-sum 0.32),
  on a single seed.
- **H2 (off-manifold / LN miscalibration): live** — untouched by E011a, which measured intact-model
  norms, not lesioned LN statistics; the mean-replace and LN-recalibrate controls have not run.
- **H3 (redundancy, not vestigiality): live and quietly the best-supported** — 7.6× within-L0
  superadditivity and all-heads-sum (2.10) < attn-0 (2.40) already argue it; but the planned
  cumulative test (L4+L5) aims at the layers with the least redundancy signal.
- **H4 (residual-scale confound): weakened, NOT refuted** — absolute write norms are the wrong
  quantity (M12); relative-perturbation and write-vs-stream-angle controls are missing.
- **H5 (under-training): live** — the checkpoint is a truncated-cosine, mid-schedule, overfit
  snapshot (~2k/4k steps at lr ≈ 0.5e-3); the 500/2000/8000-step map hasn't run.

**T002**
- **H1 (dose pathology): live and under-explored** — the transition lived in steps 0–30 and was
  sampled once; fresh-Adam sign-ascent and clip-normalized direction walks are unexamined (M5, M6).
- **H2 (structural non-separability): needs-another-discriminator** — current data cannot speak to
  it (both eval sets are B-region text, M1), and the registered cosine test lacks the within-half
  null that would make it informative.
- **H3 (wrong measurement axis): live, arguably now leading** — probes collapsed to single-character
  loops ("!!!!" naive, "EEEE" anchored), i.e., unigram/fluency death rather than content-selective
  damage; but the fluency/content split probe hasn't run, and the probes' names aren't A/B-valid
  anyway (M2).
- **H4 (wrong instrument): live** — targeted ascent untested; with clip 1.0, uniform ascent is a
  normalized direction walk, the bluntest possible instrument.

---

## The missing observation

**The within-half gradient-cosine null.** Alongside the planned cos(A,B), measure cos(A1,A2) and
cos(B1,B2) on disjoint batch halves of the *same* half — overall and per-layer — on the E001
checkpoint. Cost: minutes, no training, no new data. It is the only thing that converts T002's
headline discriminator from a tautology into a measurement:

- If cos(A,B) ≈ cos(B1,B2) (e.g., both ~0.9): there is **no A-specific gradient signal at all** —
  ascent directions are frequency-dominated, H1/H4 territory, and no lr sweep or retain weight can
  ever be selective. The registered "cos ≥ 0.85 ⇒ H2 structural" would be revealed as right for the
  wrong reason.
- If cos(A,B) is meaningfully below the within-half null: A-specific directions exist, and H4's
  targeted ascent has something to target — the e003 sweep is worth its compute.

Runners-up (in order): (1) re-run the ascent with eval every step for steps 0–30 plus the train-A /
val-A memorization gap (M6, M7) — this is the honest version of the existing curve; (2) a 3-seed
replication of E001 (~10 min of GPU) to test whether the lesion map's shape is even seed-stable
before e011 builds MLP-0 anatomy on it; (3) orthogonal-innovation write norms for the real H4 test
(M12).

---

## Number sanity checks

1. **Param count verified exactly**: 2,739,072 = 6 blocks × 444,096 + wte 12,480 + wpe 49,152 +
   ln_f 384 + lm_head 12,480. ✓
2. **THINKING.md contradicts its own bracket**: "damage/write-norm still falls **11×** across
   attention layers [0.88 → 0.02]" — 0.882/0.017 = **51.9×** (44× with the rounded endpoints).
   11× matches only L0→L4 (0.882/0.081 = 10.9), suggesting a copy error from an earlier draft.
3. **"mlp [writes] RISES with depth" is wrong**: [4.32, 1.82, 2.27, 2.73, 3.36, 5.64] *falls* 58%
   from L0 to L1 before rising (U-shape, minimum at L1). The misdescription hides that MLP
   damage-per-norm is also non-monotone [0.945, 0.083, 0.117, 0.171, 0.178, 0.105] — it *rises*
   L1→L4, so "damage/write-norm falls with depth" is an attention-specific pattern, not a network
   law, and by the journal's own logic MLP-0's outlier ratio (0.945, 11× its L1 value) deserves the
   same geometry-vs-information skepticism applied to attention.
4. **NOTES "final CE ≈ random guessing over 65 chars" is off by 6.6×**: random = ln 65 = 4.17;
   observed final naive CE = 27.57. The model is far *past* random (active anti-prediction of A
   chars), having crossed random around step 25–30.
5. **"By the time A rose +1 nat, B had ALREADY risen +1.51" (naive) misstates the curve**: at the
   only relevant grid point (step 20), ΔA = 1.63 > ΔB = 1.51; ΔA exceeds ΔB at *every* naive grid
   point; interpolated ΔB at ΔA = 1.0 is **0.93**. The stated 1.51 is ΔB at ΔA = 1.63 — a ~60%
   overstatement of early collateral. True B-ahead-of-A appears only in the anchored arm
   (steps 120–200), which the narrative doesn't mention. (Anchored's "+1.04 at A+1 nat" is fair —
   ΔA = 1.022 at that grid point.)
6. **Anchored "final" numbers are a stopping-step lottery**: ΔA ranges 6.38 (step 400) to 8.97
   (step 280); the curve oscillates ±1 nat per 20-step eval from step 200 on (M5c).
7. **Head-damage bookkeeping ✓**: recount of `harmless_below_0.02` gives exactly 16, all of them
   heads (min block damage = attn-5 at 0.0335 > 0.02), distribution L5:6, L4:3, L3:2, L2:2, L1:1,
   L0:2 — consistent with "mostly late-layer heads". But note the threshold sits ~4× above the
   realization noise floor implied by the −0.0032 L0H4 "helpful" lesion (M9, M10).
8. **E011a arithmetic ✓ but bookkeeping trap**: 2.40103/2.7218 = 0.882 and 4.07905/4.3153 = 0.945
   check out; the JSON key `"registered_prediction_H4_declining_norms": true` inverts its own name
   (true means norms do NOT decline) — a future-reader landmine.
9. **Silent pairing, good and unexploited**: E011a used generator seed 1337 on val, the same seed
   and draw order as `estimate_loss` — write norms were measured on a prefix of the exact windows
   used for lesion damage. This paired design is correct but is stated nowhere; it also means both
   measurements share one realization's idiosyncrasies.
10. **Provenance gap**: runs/e001/loss_curve.png (05:59:36) predates metrics.json (06:00:24), and
    the logged e001 run skipped training ("checkpoint found"), so `plot_history` never ran — the
    training curve is from an earlier execution, and the history behind NOTES' "bottomed ~1.55 at
    step 1500" is not persisted anywhere (e001.train.pt is absent). The overfitting claim is
    currently unverifiable from the repo.
11. **Baseline A/B gap (1.513 vs 1.685)** appears in metrics.json but is never used or discussed in
    any selectivity calculation (M3).

---

*Reviewer's one-sentence take: T001's sharpest finding (redundancy, M11) is the one it treats as a
footnote, T002's headline number is an artifact of its eval grid and mislabeled halves, and both
programs are one seed and one null-control away from knowing whether they are measuring anatomy or
geometry.*
