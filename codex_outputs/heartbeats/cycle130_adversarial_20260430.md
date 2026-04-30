Strongest kill: **the permutation test is anti-conservative**. Shuffling geometry rows within architecture preserves architecture only, not dose. If geometry deltas mostly encode alpha, the real model can look significant while the null destroys the dose code. Since the prereg explicitly says alpha-only tying/winning kills the claim, the null should condition on alpha too.  
Fix: primary permutation should shuffle within `(architecture, alpha)` or residualize against `arm_mean`/`alpha_plus_arch` first, then permute residual geometry.

Ranked attacks:

1. **Permutation null is too weak**: lines 491-495 shuffle within arch only. This can validate “geometry as dose identity,” not geometry-specific prediction.  
Fix: add `(arch, alpha)`-conditioned permutation and require it to pass.

2. **Missing telemetry imputation is not clean**: `_safe_delta_partial` writes missing individual feature deltas as `0.0` (lines 267-277). That can encode missingness or falsely assert “no change.” Filtering is raw-cell based, not delta-pair based.  
Fix: drop telemetry columns unless finite for both scratch/KD across all primary rows, or use train-fold-only imputation plus missingness indicators and report both.

3. **`combined_non_geometry` is not the best non-geometry model**: it uses alpha, alpha², early loss, available telemetry, but omits arch indicator, wallclock despite being logged, and any missingness indicators. `best_baseline` separately includes `alpha_plus_arch` and `arm_mean`, but not the strongest combined competitor.  
Fix: add `combined_non_geometry_plus_arch` with all logged non-geometry features, train-fold-safe missing handling, and compare PASS against that too.

4. **Curvature dropped is not fatal, but weakens interpretation**: current results show `curvature_top_eigen_proxy` unavailable while six telemetry features are finite. Six vs eight features is not inherently unfair under Ridge; the issue is whether the telemetry baseline is complete.  
Fix: say “beats available telemetry,” not “beats full telemetry,” unless curvature/loss-slope/throughput baselines are restored.

5. **Bootstrap CI likely understates uncertainty**: it bootstraps fixed CV predictions, not refit models.  
Fix: seed-block bootstrap should refit the full CV/baseline comparison inside each resample.

Verdict: **do not accept a g186 PASS unless attacks 1-3 are fixed before final reanalysis.**


2026-04-30T01:40:41.435621Z ERROR codex_core::session: failed to record rollout items: thread 019ddc09-2e4c-7de2-8467-9d9af43a595a not found
