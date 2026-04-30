**Findings**
- SEV 8: `trained_anchor` builds pairs on a throwaway recipient: [code/genome_183_corpus_derived_init.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_183_corpus_derived_init.py:656>). `g181a.build_anchor_pairs()` stores the dummy `Parameter` in each tuple: [code/genome_181a_tokenizer_isolation.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_181a_tokenizer_isolation.py:409>). Then `train_cell()` creates a second recipient: [code/genome_183_corpus_derived_init.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_183_corpus_derived_init.py:351>). This is a real memory bug and likely transition trigger. Use the g181b pattern: build anchor pairs inside `train_cell()` against the actual recipient.

- SEV 8: no process lock. I currently see two Python CUDA processes and the result JSON still has only `scratch_ce/42`. If both are g183 resumes, they can race on the same `OUT_PATH.tmp`: [code/genome_183_corpus_derived_init.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_183_corpus_derived_init.py:576>). That can corrupt run integrity or cause misleading failures. This blocks trusting the current run.

- SEV 7: donor params are staged to GPU before any cell and kept for all arms, including scratch: [code/genome_183_corpus_derived_init.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_183_corpus_derived_init.py:591>). Full FP32 donor snapshot is ~596M params, about 2.2 GiB. `trained_anchor` only needs embed/lm_head, about 155.6M params, about 0.58 GiB FP32. This wastes VRAM across the whole run.

- SEV 6: `cleanup_cuda()` is too weak for this boundary. It only does `gc.collect()` + `empty_cache()`: [code/genome_183_corpus_derived_init.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_183_corpus_derived_init.py:109>). It cannot free tensors still referenced by `anchor_pairs`, locals, optimizer state, or dummy params. Also `train_cell()` calls cleanup before function locals fully unwind: [code/genome_183_corpus_derived_init.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_183_corpus_derived_init.py:427>).

- SEV 6: resume recomputes donor/data/corpus embeddings before skipping completed cells. Current JSON shows one completed cell, but resume has to redo preprocessing before reaching `trained_anchor`: [code/genome_183_corpus_derived_init.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_183_corpus_derived_init.py:580>). It also builds Stage B arrays even for Stage A.

**Direct Read**
Single-process peak VRAM probably fits in 22 GB, but not as cleanly as the prereg says. Qwen3-0.6B + FP32 donor + optimizer + full-vocab CE logits likely peaks roughly 12-17 GB, not `<6 GB`. The biggest per-step tensors come from full-vocab CE casting logits to FP32: [code/genome_167_kd_canonical.py](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\code\genome_167_kd_canonical.py:461>).

I do not see an internal silent `exit(0)` path. No top-level exception swallowing was present. A true Python exception should print traceback and exit nonzero. Exit code 0 after 1/9 cells points more toward external launcher behavior, duplicate-run confusion, sleep/kill, or wrapper status reporting.

**Ship Gate**
This is not clean. SEV 8 blocks ship. I would stop duplicate resumes if confirmed, patch anchor construction to happen inside `train_cell`, stage only needed donor subset lazily, add a lock file, and then resume from the existing scratch cell.

