**Findings**

1. Severity 8: new g182 data leakage. [genome_182_triage_arena.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:352>) builds train/val/probe from the same C4 `train` stream, while [teacher generation](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_182_triage_arena.py:400>) also uses that stream. Because validation is just a shuffled slice of the same source records, seq-KD can train on prompts that later appear in held-out eval.

Exact fix: in `load_c4_pools`, draw `val_texts` and `probe_texts` from C4 `validation`, not the shared train stream.

```python
def take_texts(split: str, n: int, seed: int) -> list[str]:
    ds = load_dataset("allenai/c4", "en", split=split, streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out = []
    for item in ds:
        out.append(item["text"])
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"C4 {split} underfilled: {len(out)}/{n}")
    return out

train_texts = take_texts("train", n_train, C4_TRAIN_SEED)
val_probe_texts = take_texts("validation", n_val + PROBE_WINDOWS, C4_VAL_SEED)
val_texts = val_probe_texts[:n_val]
probe_texts = val_probe_texts[n_val:]
```

2. Severity 7: g180b can accept non-finite final eval as a completed cell. [genome_180b_cross_tokenizer.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180b_cross_tokenizer.py:1016>) checks CE during training but not final eval; [cell_done](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_180b_cross_tokenizer.py:1161>) only checks key presence.

Exact fix after final eval:

```python
final_metrics = evaluate_nll(model, pools.val_ids, pools.val_mask)
if not math.isfinite(float(final_metrics["nll"])):
    raise RuntimeError(f"non-finite final NLL tokenizer={tokenizer_spec.label} arm={arm.label} seed={seed}")
```

And in `cell_done`:

```python
try:
    final_nll = float(result["final_nll"])
except (TypeError, ValueError):
    return False
if not math.isfinite(final_nll):
    return False
```

I did not re-flag the 7 known g182 false positives. Source files are ASCII; `py_compile` passes for all three active files. g180_forecast import surface looked clean.

