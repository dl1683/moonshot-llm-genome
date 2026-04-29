Not clean.

1. **Sev 9: Shesha augment crashes on `embed_anchor`.** In [code/genome_182_triage_arena.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_182_triage_arena.py:1355>), `load_qwen3_donor()` returns a model, but code indexes it as a dict.

Replace lines 1355-1357 with:

```python
donor = load_qwen3_donor()
donor_embed = snapshot_donor_embed_lm_head(donor)
shared_vocab = build_shared_vocab_map(tok_qwen, tok_gpt2)
del donor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

2. **Sev 8: `seq_kd_full` Shesha replay is not exact.** It regenerates sampled teacher texts instead of reusing the texts from the saved run, so replayed step-108 tensors are not the saved cells. Persist teacher texts during main run and require them in augment mode; otherwise hard-fail rather than silently producing non-comparable Shesha features.

Minimal fix: save `teacher_texts` after line 1533, then in augment replace line 1363 regeneration with loading that artifact, raising if absent.

3. **Sev 7: verdict can false-PASS if co-primary models are absent.** `compute_verdict()` initializes `all_pass=True`; if augment skips A/B and only exploratory models exist, verdict can pass. Before line 1295 add:

```python
missing_primary = CO_PRIMARY_MODELS - set(loao_results)
if missing_primary:
    return {
        "verdict": "FAIL",
        "missing_primary_models": sorted(missing_primary),
        "details": details,
    }
```

Other checks: source files compile; no non-ASCII found in the three recent `genome_*.py` files; local Shesha import and toy feature extraction work. g184 Falcon branch is pre-staged but `make_model("falcon_h1")` is unreachable until config lookup includes `PHASE2_ARCH_CONFIGS`; not active for current g182 rerun.

