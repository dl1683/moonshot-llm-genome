**Review Verdict**

1. **Fix correctness:** yes for current GPT-2. GPT-2 block norms are named like `transformer.h.N.ln_1.weight` / `ln_2`, so the widened check in [genome_180_forecast.py](</mnt/c/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180_forecast.py:675>) catches them. The 1.0002 value is exactly the kind of sane result expected for scratch GPT-2 early in training.

2. **Completeness:** mostly, but one related miss remains: `_last_block_lm_head_params()` detects `transformer.h.N` when finding `last_idx`, but when selecting last-block params it only includes `layers.N` / `model.layers.N`, not `transformer.h.N`. So GPT-2 `curvature_top_eigen_proxy` is probably computed on `lm_head` only, unlike Qwen where it includes last block + head. That will not crash because curvature is optional, but it is a cross-architecture comparability bug.

3. **Other NaN risk:** no obvious GPT-2-specific required NaN left. The main required risks are generic degeneracy in spectral/TwoNN/KNN/hidden variance or gradient loss becoming non-finite. Given three GPT-2 scratch cells completed with stable feature values, those paths look healthy. Shesha, curvature, and Qwen-reference features are already optional/imputed.

4. **LOAO impact:** the dramatic architecture separation mostly **hurts** LOAO as a strict generalization test. With only two architectures, training on Qwen and testing GPT-2 means Ridge is extrapolating into a shifted feature regime. It helps scientifically if arm/health deltas preserve direction across architectures, but raw architecture offsets this large make the benchmark much harder and may swamp seed/arm signal.

5. **GPT-2 issues to watch:** watch `embed_anchor`: the anchor loss only constrains GPT-2 `transformer.wte.weight` shared-vocab rows, not a separate `lm_head` path. That is probably fine because GPT-2 ties embeddings, but it means the arm is not perfectly symmetric with Qwen. Also inspect GPT-2 curvature values after the run; if they differ strangely, fix the last-block selector.

6. **Performance:** restart is acceptable. The only waste is startup overhead: donor load, tokenizer map, teacher cache/reference geometry reloads happen before resuming cells. Since results save per completed cell, the crash only lost the partial failed cell. Run larger `--max-cells` chunks if possible to amortize startup cost.

Emergency recommendation: continue the run, but patch `_last_block_lm_head_params()` soon to include `f"transformer.h.{last_idx}."` so curvature is comparable across Qwen/GPT-2.

