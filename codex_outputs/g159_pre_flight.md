Wrote the audit to [g159_pre_flight.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/codex_outputs/g159_pre_flight.md>).

Main result: not launch-clean. The two real blockers are:
- `load_eval_data()` is using `c4_clean_v1` / `allenai/c4` `train` instead of the locked validation + dedup protocol.
- `fit_pca_at_sublayer()` does not implement the specified exact deterministic covariance PCA; it materializes full `X`, then randomly subsamples 50k rows before SVD.

I did **not** find a Qwen3/RWKV/Falcon module-path mismatch, missing `var_explained` in JSON, hook leakage, or a runtime `cp1252` print hazard. Falcon availability is also fine: [Falcon-H1-0.5B-Instruct](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct) exists, as do [Falcon-H1-0.5B-Base](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base) and [Falcon-H1-1.5B-Base](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Base). If the instruct artifact fails anyway, `0.5B-Base` is the better fallback than `1.5B-Base`.

My compute re-estimate for the script as written is about `79,872` window-forwards, roughly `0.9-1.6 GPU-hours`, so `15-45 min` is too optimistic.