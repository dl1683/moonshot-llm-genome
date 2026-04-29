# Pre-registration: genome_180b cross-tokenizer Forecast (DRAFT — gates on g180 PASS)

**STATUS:** DRAFT pre-staged 2026-04-29. **LOCKS** only after g180 PASS. If g180 FAILs, this prereg is discarded and we pivot to Tokenizer-Prior Compatibility Benchmark instead (cycle 72 Q4).

- Date: 2026-04-29
- Trigger: cycle 72 direction review Q3 — *"if g180 PASSes, run g180b cross-tokenizer next ... the hardest attack on a g180 PASS will be 'you learned Qwen tokenizer/interface identity, not a portable training-health signal'."*

---

## Hypothesis

**H1 (early-geometry forecast is tokenizer-portable):** the g180 forecast model trained on Qwen-tokenizer cells generalizes to held-out tokenizers (BERT/T5/SentencePiece/Llama-native) — held-out-tokenizer MSE on final C4 NLL gain prediction beats early-loss-only baseline by ≥15% with paired bootstrap CI > 0.

**H0 (tokenizer-identity, not training-health):** held-out-tokenizer MSE improvement < 10%, OR CI crosses zero — forecast learned Qwen3-tokenizer interface specifics, not a portable training-health signal.

## Universality level claim

Level-1 portable training-diagnostic if H1 PASSes on ≥3 distinct tokenizers spanning ≥2 vocabulary families (BPE / WordPiece / SentencePiece).

## Measurement primitive

`forecast.predict(features) → ΔNLL` where features = 8 picks from cycle 66 design (mid-layer spectral invariant, depth spectral drift, TwoNN ID, kNN-10, PCA-64 Procrustes-to-Qwen3-reference, gradient-noise scale, curvature proxy, norm/variance depth ratios). Trained on Qwen-tokenizer cells from g165/g167/g172/g174/g177/g181a. **HELD OUT:** all cells using non-Qwen3 tokenizer (BERT/T5/SentencePiece variants — TBD which to instantiate).

## Systems

- **Train (Qwen3 tokenizer):** all g180 PASS-eligible cells (Qwen-family).
- **Test (held-out tokenizers):** at minimum:
  1. Llama-native tokenizer arm (g173 currently uses shared Qwen3 tokenizer; need ONE fresh native-tokenizer Llama arm)
  2. BERT-base WordPiece (run a quick scratch + KD-from-Qwen-distillation pair if possible)
  3. T5/SentencePiece variant
- **Architecture span:** tokenizer is the held-out variable; architecture should be matched-or-bracketed within the tokenizer split where possible.

## Pass / fail

| Outcome | Decision |
|---|---|
| Held-out-tokenizer MSE improvement ≥ 15% AND paired bootstrap CI > 0 | **PASS** → §0.1 → 7.3-7.6/10, then g180c at 1B+ |
| MSE improvement in [10%, 15%] | **WEAK PASS** → tokenizer-portable but workshop-grade only |
| MSE improvement < 10% OR CI crosses zero | **FAIL** → forecast is Qwen-tokenizer-identity learner, not portable |

Secondary endpoint (advisory only, NOT load-bearing): AUROC for bad-run/stop decision threshold.

## Compute envelope (CLAUDE.md §1.5 + COMPUTE.md §9)

- [ ] VRAM ≤22 GB: yes — 500M-class students at FP16/BF16 forward
- [ ] RAM ≤56 GB: yes
- [ ] Wall-clock ≤4 h: TBD pending fresh held-out cells. Estimated 2-3h if 6-9 fresh cells × 10-20 min each.
- [ ] Disk: held-out tokenizer model caches + 9 cell results JSONs ~10GB
- [ ] Quantization: BF16 forward / FP32 master — same as g180
- [ ] Save-resume: per-cell features written to results/cache/genome_180b_features/

## Required addition before LOCK

Codex must specify:
1. Which exact tokenizer arms to run (likely 1 Llama-native + 1 BERT WordPiece minimum; possibly T5).
2. Whether to reuse PCA-64 Procrustes-to-Qwen3 reference or substitute per-tokenizer references.
3. Per-tokenizer probe-batch construction — need to tokenize matched C4 text under each tokenizer.

This prereg does NOT lock until those decisions are made post-g180 PASS.
