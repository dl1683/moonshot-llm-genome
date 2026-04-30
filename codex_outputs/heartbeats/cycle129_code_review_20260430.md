Not clean.

The six requested fixes are present: baselines include `delta_telemetry` and `delta_shesha`; under-48 rows set `rows_below_prereg_minimum` and block PASS; deltas are built in one filtered loop; permutation shuffles geometry rows for 1000 iterations; arch criterion requires both archs; alpha-1-only blocks PASS; bootstrap CI now uses actual per-row predictions.

New blocking bug: telemetry fallback zeroes the whole telemetry baseline if any one telemetry feature is missing. Current 15 cells have `curvature_top_eigen_proxy` finite in `0/15`, so `_safe_delta(...)` returns `None`, and [line 303](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_186_kd_dose_response.py:303>) replaces all 7 telemetry deltas with zeros. That neuters `delta_telemetry` and weakens `combined_non_geometry`, so geometry may beat a fake baseline. Fix by dropping unavailable telemetry columns or requiring finite telemetry for all rows before including that baseline.

Minor residual: under-48 rows cannot PASS, but can still produce `WEAK PASS` because that guard is not in the weak-pass clause.

VRAM: live `nvidia-smi` shows `20882/24463 MiB` used, GPU 91%. This is within the 22 GB effective budget but thin. Do not launch other GPU work; if it creeps above ~22 GB, stop and clear competing processes.

Compile check passed. I would patch telemetry handling before accepting any final g186 analysis.


2026-04-30T01:28:27.417821Z ERROR codex_core::session: failed to record rollout items: thread 019ddbfd-5b67-79a1-937e-0548223e59f3 not found
