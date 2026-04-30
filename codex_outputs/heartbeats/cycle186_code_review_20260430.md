**Findings**

- **SEV-8** [code/genome_196_anchor_residue_factorial.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:575): `init_only` sets `anchor_mask=None`, so the init path falls through to `weight.copy_(target)` at lines 214/226. The g194 `correct_dir_uniform_norm` target has zeros for unmatched rows, so `init_only` zeroes every unmatched embedding/lm_head row instead of injecting only matched rows. This violates the prereg “Target rows injected” design and biases the `PASS_RESIDUE` gate that compares against `init_only + 0.10`.

- **SEV-6** [code/genome_196_anchor_residue_factorial.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:399): verdict fallback can emit `FAIL_REPLICATION` even when `replication_gate_passed=True`. The replication gate correctly allows `init_anchor_full >= +0.30`, but the fallback check ignores `init_anchor_mean`. I verified with a synthetic payload: code returned `FAIL_REPLICATION True`. Prereg says `FAIL_REPLICATION` only when neither anchor reference passes; active effect present but no residue/late effect should be timing ambiguous.

- **SEV-6** [code/genome_196_anchor_residue_factorial.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:613): resume loads any existing `genome_196_anchor_residue_factorial.json` without validating `surface`, `tied`, `train_steps`, or `seeds` against the current g195-selected branch. A partial run from `--surface input` could be silently resumed/skipped after g195 later selects `output` or `both`.

- **SEV-5** [code/genome_196_anchor_residue_factorial.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:296): prereg asks for cutoff drift diagnostics at the cutoff eval step. The code only evaluates every 500 steps/final, so `cutoff_50` never records step 50, and summary does not compute post-cutoff decay.

**Checked OK**

Anchor schedules match the prereg. Scaffold seeds/construction and Fro normalization check out. g194 `correct_dir_uniform_norm` target is used and normalized to `matched_fro`. Anchor gradient injection matches g191/g195 with dynamic lambda. Current g195 JSON is still `INCOMPLETE`, so g196 correctly refuses auto-launch without an override.

`py_compile` passes. VRAM looks in-envelope; runtime is tight but plausible: g195 observed ~7.36-7.65 min/cell, so 30 cells is ~3.7-3.8h plus setup. `cleanup_cuda()` runs per cell; retained scaffold arrays are CPU-side and not a VRAM leak.

