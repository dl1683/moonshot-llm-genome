1. **S10: Shesha may erase the moat.** [Geometric Canary](https://arxiv.org/abs/2604.17698) already frames representational geometry as a predictive diagnostic, and [Shesha PyPI](https://pypi.org/project/shesha-geometry/0.1.31/) exposes RDM stability plus training-dynamics monitoring. If no-tuning Shesha at step 108 matches g182, the pivot becomes “we reimplemented a public geometry-stability diagnostic.”

2. **S10: g182 may still be arm/protocol recognition.** g180b truth is mostly scratch=0, KD negative under BERT/T5, less bad under GPT-2; permutation p=0.999 says current geometry ordering is anti-informative. Worse: `compute_normalized_labels()` appears to include scratch rows with label 0 despite logging “excluding scratch,” baking arm identity into the target.

3. **S9: Model B is not pure geometry.** “Reference-free” still includes early loss, gradient noise, grad norms, curvature, and norm ratios. A Model B PASS could mean ordinary early telemetry forecasts outcomes, not token/embedding geometry.

4. **S8: C23 is real but narrows the story.** g181b’s +0.513 nats at 5000 steps validates a persistent Qwen embed/lm_head lexical prior. It does not validate internal structure transfer or a general training-health diagnostic; it strengthens the interface-prior confound.

5. **S7: Umwelt is not just explanatory, it is a ceiling.** [URH](https://arxiv.org/abs/2604.17960) makes g180b’s tokenizer failure unsurprising. Even g182 PASS is only cross-Transformer/C4/teacher-protocol unless it escapes local training Umwelten.

6. **Resolving experiment:** run a `g182-Shesha Residual Kill`: same 72 cells, compute Shesha `feature_split`, `sample_split`, `anchor_stability`, and `rdm_drift` on the same step-108 tensors; primary analysis excludes scratch rows; residualize labels against arch × arm × teacher/corpus means plus combined telemetry; use probe/eval/OOD-disjoint labels. If Shesha or telemetry ties, pivot/moat dies. If pure geometry beats all by >25% MSE in both LOAO folds with seed-block CI >0, the claim survives.

