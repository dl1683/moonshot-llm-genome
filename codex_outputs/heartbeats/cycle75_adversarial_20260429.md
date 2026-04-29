1. **Highest severity: g180 is not yet a real forecast claim.** The 61.6% MSE reduction is from only 9 held-out rows, effectively 3 seeds × 3 arms, not 9 independent worlds. Worse, labels include future interventions not visible at 3% training, so the task partly rewards shrinkage/arm-mean behavior, not destiny prediction. Block-bootstrap by seed/arm would likely make the CI worse.

2. **The baseline is too weak.** “Geometry beats scalar early loss” is not novel enough. The fair opponent is early-loss trajectory/slope/curvature, gradient stats, protocol/arm labels, simple held-out arm means, and shuffled-geometry controls at equal measurement budget. Until then, this could be “early telemetry predicts final NLL” with a fancy feature pack.

3. **g180b will not prove the big thing.** It tests frozen Qwen-geometry portability under swapped tokenizers inside Qwen3-style recipients. That is useful, but not real BERT/T5/GPT-2 system generalization. A PASS says the Qwen shell survived vocabulary perturbation; it does not prove a cross-architecture training-health diagnostic.

4. **Alternative explanation still alive:** the model learned tokenizer/interface/protocol signatures plus regression-to-the-mean. g181a already showed the live mechanism is tokenizer-prior dominated. g180b may only show that this prior has a stable enough fingerprint across artificial tokenizer swaps.

5. **Competitive reality:** big labs care only if this beats learning-curve forecasting early enough to save meaningful compute or prevent bad run continuation. Current small-C4 Ridge evidence is not enough; DeepMind/OpenAI/Anthropic would view it as an internal telemetry baseline, not a moonshot.

**Resolving experiment:** freeze the g180 model, then run a preregistered 60+ cell holdout with ≥10 independent seeds per tokenizer/architecture, fixed future policy visible at 3%, block-bootstrap by seed/tokenizer, and compare against early-loss trajectory + protocol/arm + shuffled-geometry baselines. PASS only if geometry beats all baselines with CI > 0 within at least two held-out tokenizer/architecture families.

