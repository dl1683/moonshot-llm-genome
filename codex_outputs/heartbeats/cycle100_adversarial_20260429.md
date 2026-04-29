1. **S10: tokenizer/interface-family recognition.** Qwen3 and GPT-2 are not independent enough: both English decoder Transformers, same C4/teacher-text environment, and GPT-2 was the only g180b tokenizer where geometry helped. LOAO may be “nearby tokenizer transfer,” not architecture-agnostic geometry.

2. **S9: stage-1 `arm_mean` is weak.** With scratch excluded as labels, stage 1 has only `seq_kd_full` labeled rows. `arm_mean` becomes almost a constant and cannot model arch-specific KD harm, teacher-text effects, or tokenizer compatibility.

3. **S9: Model C is not fully pure.** Current code puts curvature in telemetry, but C still includes hidden norm/variance ratios. Those can proxy optimization health, scale, or tokenizer/interface fit.

4. **S8: Ridge can manufacture a win.** Stage-1 LOAO trains on only 12 rows against 10 correlated C features; 5-fold alpha CV has tiny validation folds. A lucky shrinkage path can beat weaker baselines.

5. **S8: 12 seeds is not enough for a 9/10 claim.** Seed-block bootstrap after feature/model/alpha selection is fragile. Fine for futility triage, not canonical proof.

6. **Alternative ruled out too quickly:** tokenizer geometry as confound, not mechanism. g181a/g181b plus g180b all point to lexical/interface compatibility, not universal training health.

**Resolving experiment:** run a prospective `g182/g184 Tokenizer Residual Kill`. Freeze Model C after g182, drop hidden norm/var, add Shesha + combined telemetry + arch/arm/teacher/corpus residual baselines, then test no-refit on a native-tokenizer third family like Falcon-H1. If frozen C loses or ties any strong baseline, g182 is falsified even if Qwen3<->GPT-2 passed.

