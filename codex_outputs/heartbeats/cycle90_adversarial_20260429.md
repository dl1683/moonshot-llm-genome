1. **Severity 10: arm/protocol identity may kill the geometry claim.** g180 “beats early loss” is not enough when `arm_mean` beats geometry on heldout rows, and g180b interim says the null beats both. The model may only learn “KD arm hurts under tokenizer mismatch,” not early geometry.

2. **Severity 9: cross-tokenizer KD harm is overclaimed.** “Universally harmful” currently means naive Qwen teacher-text CE evaluated on C4 under foreign tokenizers. That could be sequence/vocab/EOS/distribution mismatch, not a law about KD.

3. **Severity 8: g180b is not cross-architecture.** BERT/T5/GPT-2 here are tokenizer swaps inside a Qwen-style shell. It tests interface perturbation, not BERT/T5/GPT-2 systems.

4. **Severity 8: g182 Model B is not pure geometry.** It includes loss/gradient/norm/curvature telemetry. A PASS could mean “ordinary early run telemetry forecasts outcome,” not token/embedding geometry.

5. **Severity 7: probe/eval coupling remains live.** If features and labels both lean on C4 validation, the result may be “early C4-val reactivity predicts later C4-val NLL,” not general training health.

6. **Severity 6: C23 supports the objection, not the headline.** The latest canonical PASS shows persistent tokenizer/interface prior within Qwen-family training. That narrows the mechanism; it does not validate a general neural-genome diagnostic.

**Resolving experiment:** run g182, but adjudicate only a stricter variant: train-fold residualize labels against `arm_mean`; exclude arm IDs, early/delayed loss, gradients, and Qwen-reference features; use probe/eval-disjoint validation plus OOD labels; require reference-free pure-geometry residual LOAO to beat `arm_mean` and `combined_telemetry` by >25% MSE on both folds with seed-block CI > 0. If that fails, the strongest forecast claim dies.

