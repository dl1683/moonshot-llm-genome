Read `CLAUDE.md`, current `WIKI.md`, active files, and `git log --since='4 hours ago' --name-only -- code/genome_*.py` which returned no files. `py_compile` passes for both active scripts.

**Severity 7: g197/g192 can resume-skip corrupt cells.** Both scripts treat any dict containing `final_val_nll` as done. In `g197`, `_sanitize_nans()` writes non-finite floats as `null`, so a non-finite final eval can become a “completed” cell on the next resume.

Exact change in both files, replacing the skip check:

```python
v = cell.get("final_val_nll") if isinstance(cell, dict) else None
if isinstance(v, (int, float)) and math.isfinite(float(v)):  # g197
    print_flush(f"\n  Skipping {cond}/seed={seed} (done)")
    continue
```

For `g192`, use `np.isfinite(float(v))` instead of `math.isfinite`, or import `math`.

Also add after final evals in `g197 train_cell()`:

```python
final_nll = float(final_nll)
if not math.isfinite(final_nll):
    raise RuntimeError(f"non-finite final_val_nll cond={condition} seed={seed}")
```

**Severity 4: Unicode in source.** `code/genome_197_output_interface_canary_arena.py:936` has `§` despite the repo’s ASCII-only source rule. Change to:

```python
# Matched-row index for matched-only feature extraction (prereg Features)
```

**Performance note:** `g197` repeatedly builds expensive invariant heads/scaffolds, especially `build_neural_collapse_etf()` SVD. Not an immediate OOM under the stated RAM envelope, but cache invariant heads per condition to avoid wasting run time.

No other severity >=7 bug found in the audited code paths.

