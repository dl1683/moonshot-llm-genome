# Pre-registration: genome_182 Tokenizer-Prior Compatibility Benchmark (DRAFT — gates on g180 FAIL)

**STATUS:** DRAFT pre-staged 2026-04-29. **LOCKS** only if g180 FAILs the ≥25% MSE-improvement criterion. If g180 PASSes, this prereg is shelved and we run g180b cross-tokenizer instead (cycle 72 Q3).

- Date: 2026-04-29
- Trigger: cycle 72 direction review Q4 — *"the only non-delusional salvage pivot is a Tokenizer-Prior Compatibility Benchmark: quantify how tokenizer/embed/lm_head initialization, corpus match, and held-in-place optimization affect early training efficiency across tokenizers and small architectures."*

---

## Hypothesis

**H1 (tokenizer-prior compatibility predicts training efficiency):** for a fixed model architecture and pretraining corpus, the choice of {tokenizer, embed-init source, corpus-match} produces predictable variation in early-training efficiency that is captured by a small number of geometric features.

**H0 (effect is incoherent):** tokenizer/init/corpus combinations don't produce systematic efficiency differences, OR the differences don't compress to a useful benchmark.

## Universality level claim

Level-0 single-architecture multi-tokenizer benchmark. Could promote to Level-1 if effect generalizes to ≥3 architecture families.

## Measurement primitive

`tokenizer_prior_score(tokenizer, embed_init, corpus, target_steps) → ΔNLL_efficiency` where ΔNLL_efficiency is the held-out C4 NLL gain over scratch-from-random-init at fixed step budget.

## Systems — factorial design

Architecture: Qwen3-arch random-init recipient (fixed) at ~100M params.

Factor 1: **Tokenizer source** {Qwen3-tok, BERT-tok WordPiece, Llama-3-tok, T5-tok SentencePiece}
Factor 2: **Embed-init source** {Qwen3-trained, BERT-trained, Llama-trained, T5-trained, random}
Factor 3: **Corpus** {C4 web, Wikipedia, BookCorpus} (subset; not full crossing)

Pick a strategically-sparse subset of the 4×5×3=60 cell factorial. Target: ~15-20 cells × 3 seeds = 45-60 cells × 1000-2000 steps. Envelope-fit estimate: ~3-4h hard cap 4h.

## Pass / fail

| Outcome | Decision |
|---|---|
| Tokenizer-prior score variance explains ≥30% of the ΔNLL variance, with ≥3 tokenizers showing distinct compatibility patterns | **PASS** → §0.1 → 4.5-5.0/10 (workshop-grade tool, doesn't restore moonshot per cycle 72) |
| Variance explained <15% OR no clean pattern | **FAIL** → benchmark not actionable; consider stopping mainline cycles per cycle 72 Q4 |
| In-between | **WEAK** → maybe publishable as a negative-with-caveats benchmark |

## Compute envelope (CLAUDE.md §1.5 + COMPUTE.md §9)

- [ ] VRAM ≤22 GB: yes — 100M Qwen3-arch students at FP16/BF16
- [ ] RAM ≤56 GB: yes
- [ ] Wall-clock ≤4h: TBD — hinge on subset size. Need Codex design gate to lock the strategically-sparse cell selection.
- [ ] Disk: tokenizer model caches + 60-cell results JSONs ~5-10GB
- [ ] Quantization: BF16 forward / FP32 master
- [ ] Save-resume: per-cell at results/cache/genome_182_features/

## Required additions before LOCK

Codex must specify (post-g180 FAIL):
1. Strategically-sparse cell subset (~15-20 cells out of the 60 factorial — full Latin-square-like coverage)
2. Whether to use pretrained model checkpoints or freshly-trained per-tokenizer references
3. Definition of "training efficiency" — final NLL at fixed step budget OR steps-to-reach-target-NLL
4. Whether random init for embed is uniform, Xavier, or sampled-from-trained-distribution

This prereg does NOT lock until g180 FAILs (concrete trigger) AND those decisions are made.
