Not clean.

**Sev 8, g190 verdict bug:** [code/genome_190_decoder_conditioned_relearning.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_190_decoder_conditioned_relearning.py:411>) violates prereg: `anchor-only harms` must be `FAIL`, but current logic can still return `PASS`.

Exact change:
```python
anchor_only_harms = ao_mean < 0.0

if anchor_only_harms:
    verdict = "FAIL"
elif mean_gain >= 0.257 and seeds_positive >= 3 and ao_mean >= 0.15:
    verdict = "STRONG_PASS"
elif mean_gain >= 0.15 and seeds_positive >= 3:
    verdict = "PASS"
```

**Sev 8, g190 smoke/full contamination:** `--smoke` now uses 50 steps, but still writes canonical `OUT_PATH` and `PHASE1_CACHE`. A later full run can silently reuse a 50-step Phase 1 embedding.

Exact change: use smoke-specific paths and pass cache path into `run_phase1`.
```python
run_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_smoke.json") if smoke else OUT_PATH
phase1_cache = PHASE1_CACHE.with_name(PHASE1_CACHE.stem + "_smoke.pt") if smoke else PHASE1_CACHE
```
Then replace `OUT_PATH` reads/writes with `run_out_path`, and `PHASE1_CACHE` in Phase 1 load/save with `phase1_cache`.

**Sev 7, Windows encoding crash:** g190 line 252 prints `→`; I reproduced the same `UnicodeEncodeError` class via cp1252 stdout. After a long Phase 1, this can crash before payload save. Replace with ASCII:
```python
print_flush(f"\n  Phase 1 done: NLL {init_nll['nll']:.4f} -> {final_nll:.4f} ({time.time()-t0:.0f}s)")
```

**Sev 7, g189 smoke lies/runs full:** [code/genome_189_c23_content_causality.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_189_c23_content_causality.py:256>) computes `train_steps=50`, but `train_cell()` hardcodes `TRAIN_STEPS`. Add `n_steps` param and use it in schedule/range/final fallback; pass `n_steps=train_steps`.

g190 Phase 1 freezing, tied weights, and Phase 2 anchor gradient injection otherwise look correct. Recent files `py_compile` clean.

