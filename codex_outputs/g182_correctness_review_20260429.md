Reviewed `CLAUDE.md`, `COMPUTE.md`, and [genome_182_triage_arena.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py>). No edits made.

1. Severity 9: Qwen `embed_anchor` will fail. Recipient Qwen is 768-wide, donor Qwen3-0.6B is 1024-wide; direct shape match at [lines 301-337](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:301>) matches zero tensors and raises.
Fix: crop shared embedding dims for Qwen too, or build recipient from donor config. Minimal fix: in the `arch == "qwen3"` branch, for 2D embed/lm_head tensors with same vocab rows, compare `[:, :min_dim]`.

2. Severity 9: partial runs are saved as complete. `hit_limit` prints partial, then analysis runs and final JSON says `"status": "complete"` at [line 1381](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:1381>).
Fix: after line 1348, if `hit_limit`, save `{status:"partial"}` and `return` before LOAO.

3. Severity 8: test-fold leakage in NaN imputation. `feature_matrix(train)` and `feature_matrix(test)` independently compute medians at [lines 815-819](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:815>); baselines repeat this at [902-906](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:902>).
Fix: build raw matrices, fit imputer on train only, apply to test before `standardize`.

4. Severity 8: NaN NLL propagates into labels. `evaluate_nll` never checks finite batch/final loss, and `compute_normalized_labels` accepts NaN scratch/final at [770-774](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:770>).
Fix: reject non-finite eval metrics immediately:
```python
if not math.isfinite(final_metrics["nll"]):
    raise RuntimeError(f"non-finite final NLL cell={cell_id}")
```

5. Severity 7: GPU memory staging is wasteful. Donor embed snapshot remains GPU-resident, then teacher/ref Qwen loads on GPU. Peak estimate: 768 Qwen cell ~4-7 GB; full Qwen ref/donor setup ~2-3 GB extra; full real-Qwen training variant would be ~8-12 GB. Still under 22 GB, but avoidable.
Fix: store donor snapshots on CPU and load `_load_qwen3_reference_geometry` on CPU or before any model is live.

6. Severity 7: trajectory eval is expensive. Eight full 1000-window val passes per cell plus scheduled evals make validation roughly 40-50% extra forward work.
Fix: use a fixed 128-window trajectory subset; keep full val for initial/final only.

7. Severity 5: wrong baseline feature names. `gradient_stats` asks for `grad_norm`, `grad_noise_ratio`, `grad_signal_to_noise`; g180 returns `gradient_noise_scale`, `grad_norm_mean`, `grad_norm_var`.

Seed `0` is fine. `shared_vocab_map={}` fails loudly via `matched == 0`; add an explicit precheck. Source file is ASCII; no Unicode print/cp1252 issue found.

