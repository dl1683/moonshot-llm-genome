**Verdict**
Correctness: **WEAK PASS, with 2 pre-run blockers**  
Performance: **PASS for VRAM, WEAK PASS for wall-clock gate**

**Blocking Issues**
1. **Teacher-text training data is not deduped against C4 validation.** C4 train vs val has a 13-gram overlap check, but teacher-text windows are built afterward without the same validation-overlap audit. For seq-KD rows, that is the actual training source, so I would block final claims until teacher-text vs val overlap is checked per tokenizer. See [g180b build_tokenized_pools](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180b_cross_tokenizer.py:673>).

2. **Feature cache validation is too weak for reruns.** Cache acceptance checks only tokenizer/arm/seed, not `target_step`, tokenizer HF id, raw/probe hashes, feature schema, or g180 reference identity. `--force-rerun` without `--force-features` can pair fresh final NLLs with stale step-108 features. See [cache load](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180b_cross_tokenizer.py:795>) and [cell_done](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180b_cross_tokenizer.py:1146>).

**Specific Checks**
- Frozen Ridge isolation: **PASS.** `g180_train_rows_and_models()` reads `results/genome_180_forecast.json`, filters `split == "train"`, and fits both Ridge models only on those rows. I verified live: `train_rows=113`, baseline features `['early_loss']`, full feature count `24`. See [g180b Ridge fit](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180b_cross_tokenizer.py:1163>) and [g180 Ridge internals](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180_forecast.py:1313>).

- Step-108 feature timing: **PASS.** `TARGET_STEP = ceil(0.03 * 3600) = 108`, features are extracted immediately after optimizer step 108, matching g180’s target-step convention. See [g180b train loop](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180b_cross_tokenizer.py:901>) and [g180 target_steps](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180_forecast.py:92>).

- Sequence-level KD: **PASS.** This is teacher text -> retokenize under recipient tokenizer -> ordinary causal CE. No cross-tokenizer logit matching is used. See [teacher generation](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180b_cross_tokenizer.py:535>), [teacher retokenization](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180b_cross_tokenizer.py:695>), and [CE training](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180b_cross_tokenizer.py:912>).

- Tokenizer config: **PASS.** BERT `[SEP]`, T5 `</s>`, and GPT-2 `<|endoftext|>` handling matches the updated prereg table. Minor doc scar: the prereg substitution note says GPT-2, but the final decision summary still says Llama.

**Performance**
- Teacher generation: likely **2-4 GB VRAM**, no OOM expected.
- Recipient training: exact student sizes are ~75.4M BERT, ~76.6M T5, ~90.5M GPT-2 params. Peak likely **4-8 GB**, GPT-2 highest due vocab logits/CE.
- Feature extraction: likely peak phase, due grad-noise + HVP proxy. Estimate **8-14 GB**, still under the 22 GB effective cap.
- Ridge prediction: CPU negligible.

Main bottleneck is wall-clock, not VRAM: 27 sequential cells plus 1.67M teacher generated tokens. The smoke gate is directionally useful but too crude as a hard 3.75h gate because it samples only BERT/one arm/20 steps and does not scale full validation evals cleanly. I would trust it for “obviously too slow,” not for a precise pass.

Verification run: `python -m py_compile` passed for both reviewed files; lightweight Ridge reconstruction confirmed 113 g180 train rows and 24 full-model features.

