Verdict: no SEV-6+ bug that makes the low holdout cosine look like a Frobenius normalization artifact. Cosine is measured before `generate_all_rows()` scaling, so global Frobenius scaling cannot rotate directions.

Issues found:

- SEV-6 feature bug candidate: [byte_histogram](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_193_token_row_compiler.py:73) encodes tokenizer vocab strings as UTF-8. For GPT-2 byte-level tokens, strings like `Ġ` are representation glyphs, not original bytes, so byte histogram and length can be wrong if the intended feature is decoded token bytes. This can artifactually weaken directional learning.

- SEV-6 anchor-scale candidate: [trained_fro](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_193_token_row_compiler.py:401) is full Qwen vocab Frobenius norm, then applied to the smaller GPT-2 table at [row generation](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_193_token_row_compiler.py:452). That does not explain low cosine, but it can over-scale compiled anchors and make `compiled_init_anchor` harm. Calibrate by per-row RMS or matched-row norm, not full-vocab Fro across different vocab sizes.

- Loss/training path looks mechanically correct: split indexing is consistent, targets align with matched pairs, and log-frequency is train-only. But pure MSE on raw rows is weak for directional recovery; low MSE near `1e-3` may be close to a zero/mean baseline in 1024D, so it is not strong evidence of real norm learning.

- Anchor gradient is correct for unnormalized `lambda * ||W-A||_F^2`: [manual grad add](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-llm-genome/code/genome_193_token_row_compiler.py:274). Tied embeddings are fine because g188 ties weights before copy.

Performance: remaining cells are sequential. Extra GPU anchor tensor is about 200 MB; CPU compiled/shuffled tables are about 400 MB total. No obvious VRAM risk under the 22 GB envelope.

