Not clean.

**SEV-8:** [code/genome_191_string_match_decomposition.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_191_string_match_decomposition.py:111>) silently corrupts decomposition arms. `matched_rows_only` zeroes all unmatched rows, then `train_cell` copies the whole table. That tests “matched rows plus 8k zeroed vocab rows,” not matched-row content. Same issue affects shuffled matched controls.

Exact change: make init row-masked, not whole-table:

```python
def train_cell(..., custom_mask: np.ndarray | None = None, ...):
    ...
    if custom_embed is not None:
        emb_t = torch.from_numpy(custom_embed).to(
            model.model.embed_tokens.weight.device,
            dtype=model.model.embed_tokens.weight.dtype,
        )
        with torch.no_grad():
            if custom_mask is None:
                model.model.embed_tokens.weight.copy_(emb_t)
            else:
                mask_t = torch.from_numpy(custom_mask).to(emb_t.device)
                model.model.embed_tokens.weight[mask_t].copy_(emb_t[mask_t])
```

Then pass `custom_mask=matched_mask` for `matched_rows_only`, `row_shuffled_matched`, and `frequency_bucket_shuffle`; use `custom_embed=full_embed/shuffled/freq_shuf`, not zero-filled matrices.

**SEV-7:** [code/genome_191_string_match_decomposition.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_191_string_match_decomposition.py:212>) lacks non-finite loss guards, and JSON save allows `NaN`. Add:

```python
if not torch.isfinite(loss):
    raise RuntimeError(f"non-finite loss at step {step} arm={arm_label} seed={seed}")
```

and save with:

```python
tmp.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
```

**SEV-7 OOM risk:** [code/genome_190_decoder_conditioned_relearning.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_190_decoder_conditioned_relearning.py:288>) builds full 28-layer fp32 phase-2 models and trains all params with AdamW. This is likely outside the 22GB envelope. Minimum hardening:

```python
model = Qwen3ForCausalLM(cfg).to(device=DEVICE, dtype=FORWARD_DTYPE)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

Also replace the live non-ASCII error string at [line 524](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_190_decoder_conditioned_relearning.py:524>) with ASCII `-`.

`compileall` passed for all six reviewed scripts.

