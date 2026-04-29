**Top 3 Devastating Attacks**

1. **The label variance may be too small for this to be a real prediction task.**  
   If Qwen3 KD labels cluster near `-0.04` with seed-to-seed SD around only `0.003-0.005`, Ridge has to predict final normalized gain within a few thousandths. For held-out `R² >= 0.20`, RMSE must be below roughly `0.89 * label_sd`; for a 25% MSE reduction over mean baseline, RMSE must be below `0.87 * label_sd`.  
   So if label SD is `0.004`, geometry needs RMSE around `0.0035` or better. That is a very thin target, especially with one training architecture per LOAO fold and many geometry features.

2. **The 48-cell staged design is statistically malformed if only `seq_kd_full` is labeled.**  
   Scratch is excluded from labels, so the first-stage labeled set is just KD rows. That means there is no arm variation at all. The model is not learning “which training intervention to triage”; it is learning tiny seed-level deviations around one harmful protocol.  
   In that regime, `arm_mean` is the correct null. If geometry beats it, the claim is only “geometry predicts small KD harm variation,” not “geometry predicts useful triage gain.” If geometry does not beat it, the experiment is dead at stage 1.

3. **Cross-architecture sign differences create an architecture-identity confound.**  
   If GPT-2 KD also hurts, both folds are compressed negative-label regressions. Low variance likely kills power.  
   If GPT-2 KD helps while Qwen3 KD hurts, then LOAO becomes even worse: labels are architecture-sign coded. A model may appear to transfer because architecture-specific geometry encodes “Qwen-like vs GPT-like,” not because it predicts an architecture-invariant training basin. This is A17’s arm-identity confound upgraded to architecture identity.

**Code Breakage / Assumption Failures**

`compute_normalized_labels` handles negative labels numerically. It does not clip, abs, or assume positivity.

But downstream interpretation breaks:

- `simulated_kill()` assumes `sum(y_true)` is positive “gain.” With all-negative labels, `gain_retained_fraction = survived_gain / total_gain` is conceptually invalid. You are retaining or removing harm, not retaining gain.
- If labels mix signs and total gain is near zero, `gain_retained_fraction` becomes unstable or meaningless.
- `bad_run_auroc()` still ranks bottom 30% correctly, but if all KD cells are bad, this is only within-bad ranking.
- `R²` becomes fragile when `ss_tot` is tiny. If labels are nearly constant, tiny absolute errors produce large negative R² or `nan`.
- The manifesto language “final C4 NLL gain” now becomes misleading. The observed KD treatment is not a gain; it is systematic damage.

**Manifesto Implication**

This does not automatically kill “geometry predicts outcomes.” It does seriously damage the current triage premise.

The prereg is framed around early geometry predicting useful intervention gains and kill/continue decisions. If KD systematically hurts C4, then the main result becomes: “teacher-text KD is a bad intervention under this setup.” Geometry can only rescue the manifesto if it predicts meaningful heterogeneity beyond protocol/architecture identity. With labels clustered near `-0.04`, that bar is very high.

My brutal read: stage 1 is already a likely futility signal unless later Qwen3 KD labels show much larger variance or GPT-2 introduces true within-architecture, within-arm spread that geometry predicts better than arm/architecture baselines.