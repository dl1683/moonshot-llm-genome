Stage-1-only path: CLEAN.

Findings:

- SEV-3 performance: [code/genome_182_triage_arena.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:512) `evaluate_nll()` does not use `torch.no_grad()` / `torch.inference_mode()`. This wastes VRAM and time during initial, trajectory, and final evals. I do not think it exceeds the 22GB budget, but it is unnecessary autograd pressure.
- SEV-4 performance: [code/genome_182_triage_arena.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:2393) stage1-only still loads the Qwen3 donor and builds the shared vocab map even though `embed_anchor` is skipped. This is startup waste only, not a correctness bug.
- SEV-5 stats hygiene: [code/genome_182_triage_arena.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:1213) permutation test is randomized/reproducible, but p-value uses `n_better / n_iter`; `(n_better + 1) / (n_iter + 1)` would avoid reporting exact zero.

Checks:

- `--stage1-only` skips Stage 2 correctly via `if not args.stage1_only and not hit_limit`.
- Analysis does run after stage 1 completes, as long as `--max-cells` is not hit.
- Incremental save is atomic and happens after each completed cell.
- Missing `embed_anchor` cells should not break analysis. Labels become the 24 `seq_kd_full` cells; scratch cells are denominator only.
- LOAO split is by architecture, imputation and standardization use train-only statistics, and no test-arch features enter training.
- Peak VRAM per active cell should be comfortably under 22GB. Qwen3 stage is the peak, likely single-digit GB even with the inefficient eval path. No gradient accumulation/checkpointing needed.
- Model and optimizer are deleted after each cell, with GC and CUDA cache clear.
- NaN guard is acceptable for training loss, grad norm, required features, and final NLL. Initial/trajectory NLL are less guarded but not blocking.
- `results/cache/genome_182_features/teacher_texts.json` exists: 8,704 texts, 9,555,888 bytes.
- No stale `results/genome_182_triage_arena.json`, tmp file, or prior g182 result file found. Only the teacher cache exists.

I could not inspect PID 23734 directly because Windows process queries returned access denied.

