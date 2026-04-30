**Findings**
- SEV-4: `g195` reports `tied_mean_gain` against `scratch_untied`, so that number is confounded by architecture/parameter count. Untied has `193,122,304` params; tied reference has `141,659,136`. Primary input/output/both verdict is unaffected because those arms all use the untied baseline. Treat tied reference as diagnostic unless you add a `scratch_tied` arm. See [g195:250](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_195_untied_input_output_factorial.py:250>) and [g195:260](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_195_untied_input_output_factorial.py:260>).

- SEV-2: `g195` records `anchor_lambda=0.01` for `scratch_untied` even though no anchor target exists. Training is unaffected, but result metadata is misleading. Set `actual_lambda` only when `anchor_embed` or `anchor_lm_head` is active. See [g195:157](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_195_untied_input_output_factorial.py:157>) and [g195:232](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_195_untied_input_output_factorial.py:232>).

**Checks**
- `g195` untied creation is correct: `tie_word_embeddings=False`, verified separate `embed_tokens` and `lm_head` pointers. Tied reference correctly routes through `g188.make_gpt2_qwen3_model()`, which ties weights.
- `g195` anchor injection is independent for untied input/output: separate gradient additions hit `embed_tokens.weight` and `lm_head.weight`; tied reference only anchors the shared embed/lm_head matrix.
- `scratch_untied` is the right baseline for the primary untied factorial. The tied-vs-untied parameter difference is only a confound for interpreting `tied_reference` gain.
- VRAM: `g195` is inside envelope. Extra params over tied are ~51.5M, not 103M extra over tied. Worst-case extra FP32 AdamW training state is ~0.8 GB.
- `g192` requested config fields match actual Qwen3-0.6B: `kv_heads=8`, `intermediate=3072`, `head_dim=128`, `rope_theta=1e6`. Verified against local `AutoConfig` and Hugging Face config: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
- `g192` sets `use_cache=False` in the constructed training config.
- `g192` instantiated at `491,930,624` params with GPT-2 vocab and tied embeddings. BF16 weights plus Adam state and B=8/S=256 activations should stay well below the 22 GB VRAM envelope.

`py_compile` passed for both files:
`python -m py_compile code\genome_195_untied_input_output_factorial.py code\genome_192_28layer_replication.py`

