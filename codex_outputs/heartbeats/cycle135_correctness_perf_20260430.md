Found two severity >=7 issues.

1. **Severity 8.5: `--export-ridge` can freeze an incomplete, biased model.** At current state, results have 34 cells and 24 valid delta rows, all nonzero pairs from Qwen3. `export_frozen_ridge()` only requires `len(delta_X) >= 20`, so it would export a Qwen-only Ridge before GPT-2 dose rows exist.

Exact change in [code/genome_186_kd_dose_response.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_186_kd_dose_response.py:904):

```python
expected_cells = len(ARCHS) * len(SEEDS) * len(KD_ALPHAS)
expected_rows = len(ARCHS) * len(SEEDS) * (len(KD_ALPHAS) - 1)
if len({c["cell_id"] for c in cells}) != expected_cells:
    print_flush(f"Refusing export: incomplete cells ({len(cells)}/{expected_cells})")
    return
if existing.get("dose_analysis", {}).get("verdict") != "PASS":
    print_flush("Refusing export: g186 verdict is not PASS")
    return
if len(delta_X) != expected_rows:
    print_flush(f"Refusing export: expected {expected_rows} delta rows, got {len(delta_X)}")
    return
```

2. **Severity 7: `--max-cells` can mark partial runs completed.** The `break` at line 819 exits only the seed loop; after that, the script can still run analysis and set `status = "completed"` on an incomplete file.

Exact change: track `stop_requested`, break all loops, and set status from actual cell coverage:

```python
stop_requested = False
...
if cells_run >= args.max_cells:
    print_flush(f"    MAX CELLS reached ({args.max_cells})")
    stop_requested = True
    break
...
if stop_requested:
    break
...
expected_total = len(archs) * len(doses) * len(seeds)
run_complete = len({c["cell_id"] for c in done_cells}) >= expected_total
if run_complete and len(labeled) >= 10:
    existing["dose_analysis"] = pairwise_dose_analysis(labeled, all_cells)
existing["status"] = "completed" if run_complete else "running"
```

Otherwise: Ridge standardization is correct for CV and final freeze; NaNs are mostly guarded; OOM risk looks acceptable; Windows issue is low severity: non-ASCII em dashes remain in code comments/log strings despite the repo rule.

