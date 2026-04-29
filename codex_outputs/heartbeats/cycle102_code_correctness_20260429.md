Not clean. Two severity-8 issues in [code/genome_182_triage_arena.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_182_triage_arena.py:1353>).

1. Sev-8: optional NaNs now survive feature extraction, but `save_incremental()` writes with `allow_nan=False`, so the first optional NaN crashes the run/save. Fix by adding a sanitizer and UTF-8 I/O:

```python
def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
```

```python
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(to_jsonable(data), f, indent=2, ensure_ascii=True, allow_nan=False)
```

Also change `load_existing()` to `open(out_path, encoding="utf-8")`.

2. Sev-8: g184 phase-2 teacher texts are sampled but not cached, so resume runs can silently train different Falcon cells on different KD corpora. Fix around lines 1671-1674:

```python
CACHE_DIR.mkdir(parents=True, exist_ok=True)
teacher_cache_path = CACHE_DIR / f"teacher_texts_{phase2_arch}.json"
if teacher_cache_path.exists():
    with open(teacher_cache_path, encoding="utf-8") as f:
        teacher_texts = json.load(f)
else:
    teacher_texts = generate_phase2_teacher_texts(phase2_arch, n_teacher)
    with open(teacher_cache_path, "w", encoding="utf-8") as f:
        json.dump(teacher_texts, f, ensure_ascii=True)
```

Clean checks: all three touched scripts are ASCII-only and `py_compile` passes.

