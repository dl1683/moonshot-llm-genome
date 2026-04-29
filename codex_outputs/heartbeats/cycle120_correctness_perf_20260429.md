Reviewed [code/genome_182_triage_arena.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:881>).

1. `compute_normalized_labels` handles negative labels correctly.
`label = (scratch_nll - c["final_nll"]) / scratch_nll` at line 901 does not clip, abs, log, or assume positivity. Current partial JSON confirms labels around `-0.0415`, `-0.0379`, `-0.0372`.

2. `bad_run_auroc` is correct for negative labels.
Lines 1233-1240 define bad runs as `y_true <= bottom_30pct`, then score with `-geo_pred`. So “more negative predicted label” means “more likely bad.” That is directionally right.

3. `simulated_kill` mostly handles ranking correctly, but `gain_retained_fraction` assumes positive total gain.
Lines 1249-1250 kill the lowest predicted labels, which is correct. Line 1258 compares against the true lowest labels, also correct.

The broken part is lines 1253-1255:

```python
total_gain = np.sum(y_true)
survived_gain = np.sum(y_true[survived_idx])
gain_retained = survived_gain / total_gain
```

With all-negative labels, this ratio is not “retained gain.” If you kill the worst negative run, the surviving sum becomes less negative, and the ratio can look like failure or become unstable. If total gain is near zero, it can explode. This is the clearest positive-label assumption.

4. `compute_verdict` has one bad criterion under negative distributions.
MSE reduction, R², and AUROC are mathematically okay with negative labels. But `kill_retains_ge_90pct` at line 1309 inherits the broken positive-gain assumption above. For a KD-hurts regime, the useful kill metric should be something like “fraction of harm avoided” or “mean label improves among survivors,” not “retain 90% of gain.”

5. Pass criteria are too brittle for the current compressed negative distribution.
A Qwen-only narrow band around `-0.04` means:
- R² will be unstable because label variance is tiny.
- MSE reduction can be dominated by constant/arm baselines.
- AUROC is still meaningful as rank detection, but with few cells it will be noisy.
- The kill-retention gate is currently invalid if labels are mostly negative.

Performance: 77°C, 18GB VRAM, 78% util is sustainable for ~3h if stable. My live check saw `64°C`, `14.5/24.5GB`, `0% util` at that instant, with `python.exe` and `ollama.exe` on GPU, so it may have been between work bursts. I would keep running unless temp climbs above ~82°C, VRAM steadily creeps toward 24GB, or throughput stalls.

Bottom line: label computation and AUROC direction are fine. The code assumption that breaks under negative labels is the “gain retained” kill metric and verdict gate. 
