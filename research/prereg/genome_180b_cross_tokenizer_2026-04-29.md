# Pre-registration: genome_180b cross-tokenizer Forecast (LOCKED)

**STATUS:** LOCKED on 2026-04-29 after g180 WEAK PASS. g180 cleared the point-estimate gate with 61.6% held-out MSE reduction over early-loss-only, but its paired bootstrap CI crossed zero by a hair because the held-out set had only 9 Llama-family rows. g180b is therefore the tokenizer-portability lock test, not a claim-promotion run.

- Date: 2026-04-29
- Trigger: cycle 72 direction review Q3: if g180 shows real signal, run the hardest attack on that signal: "you learned Qwen tokenizer/interface identity, not a portable training-health signal."
- Locked design source: `codex_outputs/g180b_design_gate_20260429.md`

---

## Hypothesis

**H1 (early-geometry forecast is tokenizer-portable):** the g180 forecast model trained on Qwen-tokenizer cells generalizes to held-out tokenizers. On held-out-tokenizer cells, the frozen g180 geometry+early-loss Ridge beats the frozen early-loss-only Ridge by at least 15% MSE reduction, with paired bootstrap CI strictly above 0.

**H0 (tokenizer identity, not training health):** held-out-tokenizer MSE improvement is below 10%, or the paired bootstrap CI crosses 0. That means the current forecast learned Qwen3 tokenizer/interface specifics rather than a portable training-health signal.

## Universality level claim

No Level-1 claim is promoted by the prereg alone. If g180b PASSes on all three held-out tokenizer arms below, the forecast/diagnostic line may be described as a tokenizer-portable training diagnostic within small decoder-only recipients, and the section 0.1 score can move to the cycle-72 projection of 7.3-7.6/10. It still requires g180c at 1B+ before any economic-scale claim.

## Frozen forecast rule

The g180 Ridge models are applied **as-is**:

- Fit the full geometry+early-loss Ridge on the original g180 Qwen-tokenizer train rows only.
- Fit the early-loss-only baseline Ridge on the same original g180 train rows only.
- Freeze feature names, imputation medians, standardization means/scales, Ridge coefficients, and alpha.
- Treat every g180b row as held-out test data.

No g180b row may enter training, imputation, feature scaling, model selection, or hyperparameter tuning. A pooled retrain with g180b features is allowed only as an exploratory appendix and cannot affect PASS/FAIL.

## Measurement primitive

`frozen_forecast.predict(features) -> final_C4_NLL_gain`

Feature set is the g180 feature set:

- early loss
- mid-layer spectral alpha / participation-ratio features
- depth spectral drift
- TwoNN intrinsic dimension
- kNN-10 clustering coefficient
- PCA-64 Procrustes/RSA to the shared Qwen3 reference
- gradient-noise scale and gradient norm moments
- curvature top-eigen proxy
- hidden norm/variance depth ratios

The label remains paired same-seed final C4 validation NLL gain:

`label = final_nll(scratch_ce_same_tokenizer_same_seed) - final_nll(arm_same_tokenizer_same_seed)`

Scratch rows have label 0 and remain in the scored held-out set, matching g180/g173 practice.

## Tokenizer arms

Run exactly three non-Qwen3 tokenizer arms:

| Arm | HuggingFace tokenizer ID | Vocabulary family | Recipient architecture |
|---|---|---|---|
| `bert_wordpiece` | `bert-base-uncased` | WordPiece | Qwen3-arch recipient with swapped tokenizer/vocab |
| `t5_sentencepiece` | `google-t5/t5-small` | SentencePiece unigram | Qwen3-arch recipient with swapped tokenizer/vocab |
| `llama3_bpe` | `meta-llama/Llama-3.2-3B` | Llama-3 BPE/tiktoken | Qwen3-arch recipient with swapped tokenizer/vocab |

**Architecture decision:** use Qwen3-architecture recipients with swapped tokenizers for the primary test. Do not run architecture-matched BERT, T5, or Llama recipients in g180b.

Rationale:

- g180 already used g173 to test architecture movement under the shared Qwen tokenizer.
- g180b must isolate the tokenizer/interface attack. Changing tokenizer and architecture together would make a fail uninterpretable.
- BERT and T5 are not native decoder-only C4-NLL recipients; making architecture-matched recipients would change objective, masking, and loss semantics.
- "Llama-native" in this prereg means the Llama tokenizer is native; the recipient architecture remains Qwen3-style.

Recipient config for all arms:

- Qwen3-style decoder-only model, random init
- `hidden_size=768`
- `num_hidden_layers=8`
- `num_attention_heads=12`
- `num_key_value_heads=6`
- `intermediate_size=2048`
- tied input embedding / LM head
- vocab size = `len(tokenizer)`
- `max_position_embeddings >= 320`
- BF16 autocast forward on CUDA, FP32 optimizer/master parameters

Tokenizer special-token handling:

- `bert-base-uncased`: use `[SEP]` as EOS separator; use `[PAD]` as pad if padding is required; do not inject `[CLS]` into every training window.
- `google-t5/t5-small`: use `</s>` as EOS separator and `<pad>` as pad.
- `meta-llama/Llama-3.2-3B`: use native EOS; if pad is absent, set pad to EOS.

## Cell matrix

Run 27 fresh held-out cells:

`3 tokenizer arms x 3 training arms x 3 seeds = 27 cells`

Seeds:

- `42`
- `7`
- `13`

Training arms per tokenizer and seed:

| Arm | Purpose | Label role |
|---|---|---|
| `scratch_ce` | Native-tokenizer CE-only C4 training baseline | paired baseline, label 0 |
| `seq_kd_full` | Tokenizer-portable sequence-level Qwen3 teacher distillation for all steps | scored gain vs `scratch_ce` |
| `seq_kd_late_only` | CE-only for early steps, sequence-level distillation only in final third | scored gain vs `scratch_ce` |

Token-level top-k logit KD is explicitly **not** used across tokenizers. It is invalid because the teacher and recipient vocabularies do not share a token index space. The tokenizer-portable KD analogue is sequence-level distillation:

1. Lock a raw C4 prefix pool once.
2. Use the Qwen3-0.6B teacher to produce deterministic decoded continuations for those prefixes.
3. Retokenize the resulting raw teacher text under each held-out tokenizer.
4. Train the recipient with ordinary CE on those teacher-text windows.

`seq_kd_full` uses teacher-text CE for all training steps. `seq_kd_late_only` uses ordinary C4 CE for steps 1-2400 and teacher-text CE for steps 2401-3600. Final evaluation is always ordinary held-out C4 validation NLL under the recipient tokenizer, not teacher-text NLL.

Training constants:

- `TRAIN_STEPS=3600`
- `SEQ_LEN=256`
- `TRAIN_BATCH_SIZE=8`
- `LR=3e-4`
- `LR_WARMUP_STEPS=200`
- `WEIGHT_DECAY=0.1`
- `GRAD_CLIP=1.0`
- same optimizer family as g173/g180 replay: AdamW betas `(0.9, 0.95)`

## Procrustes reference

Use the **shared Qwen3-reference PCA-64 Procrustes/RSA features**, the same reference family used by g180.

Do not build per-tokenizer references for the primary result.

Rationale:

- The primary question is whether the existing g180 forecast model generalizes as-is. Per-tokenizer references would redefine the feature space and force a new forecast model.
- Shared Qwen3 reference is the hard attack: if the Ridge relies on Qwen-tokenizer identity features, held-out-tokenizer performance should fail.
- Per-tokenizer references would be fairer for geometry normalization but would test a different model. That belongs in a later repair experiment if g180b fails, not in the confirmatory g180b gate.

Reference details:

- Hidden-state Procrustes/RSA: compare each recipient's fixed-probe hidden cloud to the trained `Qwen/Qwen3-0.6B` reference hidden cloud after PCA-64, as in g180.
- Embedding and LM-head Procrustes/RSA: retain the g180 row-index comparison against the Qwen3 reference for the first `min(vocab_size, 4096)` rows. Do not attempt semantic token remapping.
- If a tokenizer has fewer than 4096 usable vocab rows, use all available rows and record `*_reference_rows_used`.

The row-index embedding comparison is intentionally harsh and partly nonsensical across vocabularies. That is acceptable for g180b because the failure mode under test is exactly whether the current model is leaning on Qwen-specific interface identity.

## Probe-batch construction

Use the **same raw C4 passages tokenized separately under each tokenizer**. Do not use tokenizer-native independent C4 batches.

Probe construction:

- Split: C4 validation.
- Raw passage seed: `180180`.
- Probe windows: `16`.
- Sequence length: fixed `256` tokens per tokenizer.
- Build one locked ordered raw text pool.
- For each tokenizer, tokenize that same raw text pool independently.
- Concatenate with the tokenizer's EOS separator between raw documents.
- Slice deterministic contiguous windows of length 256.
- Drop incomplete final windows rather than padding probe windows.
- Preserve the same raw-pool ID/order metadata in every feature cache.

Training/evaluation construction:

- Use the same raw C4 train passage pool for all tokenizers before tokenization.
- Use the same raw C4 validation passage pool for all tokenizers before tokenization.
- Tokenize per tokenizer after raw-pool selection.
- Keep `SEQ_LEN=256` fixed for train, validation, and probe.
- Keep validation window count at `N_C4_VAL_WINDOWS=1000`.
- Keep train window count at `N_TRAIN_WINDOWS=8192`, unless the smoke test projects wall-clock over 4h; if so, abort before producing a result rather than silently shrinking the prereg.

Rationale:

- Same raw passages control semantic/data-distribution variation.
- Tokenizer-native batches would confound tokenizer with different C4 samples.
- Fixed sequence length controls GPU cost, attention geometry, hidden-cloud point count, and curvature/gradient feature scale.
- Variable-length batches would make the geometry features absorb padding and context-length artifacts.

## Primary analysis

Scored test set: all 27 g180b rows, with tokenizer arm recorded for stratified diagnostics.

Primary metric:

`MSE_reduction = (MSE_early_loss_only - MSE_geometry_plus_early_loss) / MSE_early_loss_only`

Use paired bootstrap over held-out rows with `BOOTSTRAP_N=10000`, same seed convention as g180.

Primary PASS:

- held-out-tokenizer MSE reduction >= 15%, and
- paired bootstrap CI95 for MSE improvement is strictly above 0, and
- no false stop on an actionable high-gain arm under the inherited g180 stop rule.

WEAK PASS:

- MSE reduction in `[10%, 15%)`, and
- paired bootstrap CI95 is strictly above 0.

FAIL:

- MSE reduction < 10%, or
- paired bootstrap CI95 crosses 0, or
- any actionable high-gain arm is falsely stopped.

Secondary diagnostics, not load-bearing:

- per-tokenizer MSE reduction
- per-tokenizer residual means
- AUROC for bad-run / stop decision threshold
- performance with Qwen-reference embedding/lm_head features removed from the frozen feature vector; this is exploratory only and cannot rescue a primary FAIL.

## Compute envelope

This design fits the binding envelope in `COMPUTE.md`.

- Max VRAM: estimated 10-14 GB during the largest Llama-tokenizer Qwen3-arch student training cell; below the 22 GB limit.
- Max RAM: estimated 16-28 GB including raw text pools, tokenizer objects, result rows, and feature caches; below the 56 GB limit.
- Wall-clock estimate: 2.7-3.6 hours total on the RTX 5090 Laptop.
- Hard wall-clock cap: 4 hours. Run a 1-tokenizer, 1-seed, 20-step smoke test and abort if projected total exceeds 3.75 hours.
- Disk footprint: estimated 2-8 GB for tokenizer caches, teacher-text cache, per-cell JSONs, and feature caches. No non-Qwen model weights are loaded.
- Quantization/precision: students train with BF16 autocast forward and FP32 optimizer/master states; Qwen3 teacher inference uses BF16 forward.
- Save/resume: write per-cell result JSONs and per-cell feature JSONs under `results/cache/genome_180b_features/` immediately after each cell.

The run must be sequential, not multi-process parallel, to preserve Windows+CUDA reliability and keep CUDA memory below the envelope.

## Decision summary

Locked decisions:

1. **Tokenizer arms:** `bert-base-uncased`, `google-t5/t5-small`, `meta-llama/Llama-3.2-3B`; all use Qwen3-arch recipients with swapped tokenizers/vocabs.
2. **Procrustes reference:** shared Qwen3-reference PCA-64, exactly to test frozen-model portability; no per-tokenizer references in the primary result.
3. **Probe batches:** same raw C4 passages tokenized per tokenizer, fixed 256-token windows; no tokenizer-native C4 sampling and no variable-length probes.
4. **Cells:** 27 held-out cells = 3 tokenizers x 3 arms x 3 seeds.
5. **Forecast model:** frozen g180 Ridge and frozen early-loss-only Ridge applied as-is; no retraining or per-tokenizer feature fitting.

This is the correct hard gate. A PASS means the geometry forecast survived the tokenizer-identity attack. A FAIL means g180 was probably a Qwen-interface forecast, and the next mainline move is g182 Tokenizer-Prior Compatibility Benchmark rather than g180c scaling.
