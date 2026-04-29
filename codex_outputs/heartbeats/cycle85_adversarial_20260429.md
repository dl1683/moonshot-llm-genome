1. **Severity 10: `arm_mean` already looks like it kills P17.** Post-hoc but fair: on g180 heldout Llama rows, mapping `scratch/kd_logit/kd_late` to the matching Qwen-arch train-arm means gives MSE **0.00150**, beating full geometry **0.00595** and early-loss **0.01548**. That means g180 may be “protocol identity transfers across families,” not geometry.

2. **Severity 9: `combined_telemetry` could beat g182.** If arm labels + trajectory + gradients + delayed losses win in LOAO, the product claim becomes “ordinary run telemetry forecasts training,” not “early geometry forecasts training.” g182 correctly includes this baseline; it is the real falsifier.

3. **Severity 8: g182 can PASS but still not prove token/embedding geometry.** Model B includes `early_loss`, gradient stats, curvature, hidden norm ratios, PR, ID, kNN. That is a mixed diagnostic bundle. A PASS could be from norm/gradient/curvature telemetry, not token/embedding/interface geometry.

4. **Severity 7: cycle-84 validation split fix creates a new probe/eval coupling.** It removes teacher/train overlap risk, but features are extracted on C4 validation probe windows while final NLL is also C4 validation. That can become “early heldout C4 reactivity predicts later heldout C4 NLL,” not general training health. Need disjoint validation-A probe vs validation-B/test label, ideally plus Wikitext/OOD.

5. **Severity 7: 12 seed blocks is only enough for a large effect.** Bootstrap over 12 clusters is fragile for residual claims after arm means. It can certify a huge effect, but a marginal PASS/WEAK PASS should not upgrade the story. Effective independent evidence is 12 seeds × 2 reciprocal architecture folds, not 72 independent cells.

6. **Severity 6: two-architecture LOAO is thin.** Qwen3↔GPT-2 is better than g180, but still only two Transformer families. It can show “one reciprocal transfer worked,” not architecture-general triage.

7. **Resolving experiment:** run g182, but adjudicate the decisive result as **reference-free, pure-geometry residual LOAO**: subtract train-fold `arm_mean`, exclude loss/gradient/delayed telemetry from the geometry bundle, use probe/eval-disjoint validation splits, and require >25% residual-MSE reduction vs `arm_mean` and `combined_telemetry` in both folds with seed-block CI > 0. If that fails, P17 dies. If it passes, I stop objecting.

