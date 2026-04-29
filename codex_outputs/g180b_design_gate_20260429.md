Locked the g180b design.

Updated:
- [research/prereg/genome_180b_cross_tokenizer_2026-04-29.md](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\research\prereg\genome_180b_cross_tokenizer_2026-04-29.md:1>)
- [codex_outputs/g180b_design_gate_20260429.md](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\codex_outputs\g180b_design_gate_20260429.md:1>)
- [WIKI.md](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\WIKI.md:94>)

Core lock:
- Tokenizers: `bert-base-uncased`, `google-t5/t5-small`, `meta-llama/Llama-3.2-3B`
- Recipients: Qwen3-arch with swapped tokenizer/vocab, not architecture-matched
- Procrustes: shared Qwen3 PCA-64 reference
- Probe data: same raw C4 passages, tokenized per tokenizer, fixed 256-token windows
- Cells: `3 tokenizers x 3 arms x 3 seeds = 27`
- Forecast: frozen g180 Ridge and frozen early-loss baseline, no retraining

No code/tests run; this was a doc/design-gate lock only. I left unrelated dirty/untracked files alone.

