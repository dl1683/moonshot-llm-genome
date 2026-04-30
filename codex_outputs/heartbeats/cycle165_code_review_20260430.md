Findings:
- SEV-8 g192 factory is not full Qwen3-0.6B. Cached config is `kv_heads=8`, `intermediate=3072`, `head_dim=128`, `rope_theta=1e6`, `max_pos=40960`; g192 hardcodes `4/2816/64/1e4/320`. It’s a 28-layer variant of the g188 surrogate, not true 28-layer Qwen3 with GPT-2 vocab.
- SEV-6 g192 is not strictly g191-equivalent: factory casts params to BF16, while g191 trains FP32 params under autocast. It also omits `use_cache=False`, a real VRAM risk at 28 layers.
- SEV-5 g192 `PASS_ATTENUATION` ignores `row_shuffled`; matched can pass even if shuffled helps equally/more. Require matched > shuffled by margin and/or shuffled under a cap.

g194:
- `matched_fro` fix looks correct: factorial arms normalize to matched-row Fro, and `custom_mask`/`anchor_mask` restrict training use to matched rows.
- `decompose_rows()` is fine for matched rows; zero rows become zero dirs with epsilon norm. Mask use is downstream and okay. Add `n_matched > 0` assert defensively.

`py_compile` passed.

