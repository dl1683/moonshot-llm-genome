1. **g180 is not a STRONG_PASS in the ledger.** Actual ledger: `baseline_mse=0.01548`, `full_mse=0.00595`, `61.6%` reduction, `R² -0.941 -> 0.254`, CI `[-0.0009, +0.021]`, `n=9`. That is WEAK/INTERMEDIATE, not locked.

2. **Strongest g180 kill: it may be learning seed/arm structure, not geometry.** In held-out rows, scratch and late-KD have identical step-108 features by seed, so full predictions are identical for each scratch/late pair. It predicts the midpoint: good MSE, bad causal interpretation.

3. **The “24 geometry features” are Qwen-shaped.** Train is 113 Qwen-family rows; test is 9 Llama rows. Features include Qwen-reference Procrustes/RSA and duplicated embed/lm_head features. This can be interface-fingerprint regression, not architecture-general geometry.

4. **Baselines were too weak.** Early-loss-only is not the fair opponent. The fair opponent is arm labels, arm means, loss trajectory, delayed loss, seed-block means, and combined telemetry. g182 correctly includes these. If any tie/beat geometry, g180 dies.

5. **g181b survives horizon attack but stays narrow.** Actual: `+0.513 nats` at 5000 steps, seeds `+0.531/+0.486/+0.523`, CI `[0.486, 0.531]`. Strong persistence, but only Qwen3-arch embed/lm_head anchor. Alternative explanation: tokenizer/interface initialization improves lexical prior under same tokenizer shell, not neural genome transfer.

6. **g182 LOAO may collapse.** I expect collapse risk is high unless reference-free geometry beats combined telemetry in both folds. If full geometry passes but reference-free fails, it is Qwen-reference leakage. If arm/protocol labels win, g180 was intervention identity.

7. **Resolver experiment:** run g182 exactly as preregistered: 72 fresh cells, Qwen3 vs GPT-2 LOAO, both full and reference-free geometry, seed-block bootstrap, and all non-geometry baselines. Kill condition: combined telemetry or arm labels tie/beat geometry, or Model B fails either LOAO fold.

