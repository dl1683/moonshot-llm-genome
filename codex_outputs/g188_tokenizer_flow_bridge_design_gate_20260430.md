**Gate Verdict**
Revise, then approve a staged pilot. The idea is sound, but “OT on a 50K x 50K matrix” is the wrong implementation. Use a sparse character-offset alignment graph and Sinkhorn-style balancing on that graph.

One correction: local [results/genome_183_corpus_derived_init.json](</c/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_183_corpus_derived_init.json>) is marked `INCOMPLETE`, with 2/3 `ppmi_svd_anchor` seeds done. Those two are already harmful: mean PPMI NLL `6.763` vs scratch `6.456`, about `-0.307` nats. `trained_anchor` is `+0.389` nats in this artifact.

**Design Answers**
1. Soundness: yes, if the graph is an alignment graph, not sentence co-occurrence. Sentence bag co-occurrence risks recreating the g183 PPMI failure. The OT object should be a sparse source-token to target-token coupling.

2. Library: POT is fine for small smoke tests, but I would not make dense POT the primary path. Implement sparse Sinkhorn/IPFP directly over CSR/COO edges: `K_ij = exp(-C_ij / eps)`, then alternate row/column scaling. POT docs confirm Sinkhorn/regularized OT support, but dense 50K-scale plans are out of envelope: https://pythonot.github.io/

3. Alignment: use character-offset overlap as primary. Add optional neighbor diffusion within +/-1 or +/-2 tokens as a secondary ablation. Do not use same-sentence co-occurrence as the main signal.

4. Subset: do not define this as `10K x 10K` dense. Pick top tokens by train mass: likely `top 20K Qwen source x top 10K-20K GPT-2 target`, with exact mass coverage reported. Rare GPT-2 rows get fallback embeddings. Full `151,936 x 50,257` is about 7.6B entries, so one float32 matrix is ~30 GB before kernels/plans.

5. Thresholds:
   - PASS: flow bridge beats scratch by `>= +0.12` nats mean and wins `3/3` paired seeds.
   - Must beat `char_overlap_no_ot` by `>= +0.04` nats.
   - Must beat `string_match_anchor` by `>= +0.05` nats.
   - Must beat both `flow_shuffled_qwen_rows` and `flow_random_source` by `>= +0.08` nats and `3/3` seeds.
   - PARTIAL: `+0.05` to `+0.12` nats, useful but not headline.
   - STRONG PASS: `>= +0.20` nats, about half of the current g183 trained-anchor ceiling.

6. Score:
   - Section 0.1 movement: `6.4/10` if revised; `4/10` as dense/full-OT written.
   - Nobel/Turing upside: `2/10` as an experiment, `4/10` if it becomes a general tokenizer-transcoding law.
   - Feasibility: `7/10` sparse offset bridge, `2/10` dense 50K OT.

**Required Controls**
Use these arms, all GPT-2-tokenizer Qwen3-arch, same seeds:

`scratch_ce`, `flow_bridge_init_anchor`, `char_overlap_no_ot`, `direct_string_match_anchor`, `flow_shuffled_qwen_rows`, `flow_random_source`, `random_plan_same_degrees`, and ideally `flow_init_only` vs `flow_anchor_only` to separate initialization from continuous anchor force.

Calibrate all anchor arms to the same embedding Frobenius norm and comparable anchor gradient norm. Otherwise a “win” may just be norm/regularization.

**File Spec**
Add:

- `code/genome_188_tokenizer_flow_bridge.py`
- `research/prereg/genome_188_tokenizer_flow_bridge_2026-04-30.md`
- `results/genome_188_tokenizer_flow_bridge.json`
- `results/cache/genome_188_tokenizer_flow_bridge/alignment_edges.npz`
- `results/cache/genome_188_tokenizer_flow_bridge/flow_plan_topk.npz`
- Update `experiments/ledger.jsonl`, `experiments/EXPERIMENTS.md`, and `WIKI.md` after run.

Core functions: `tokenize_with_offsets`, `build_offset_alignment_edges`, `make_sparse_cost_kernel`, `sparse_sinkhorn_balance`, `barycentric_target_embeddings`, `make_bridge_controls`, `train_cell`, `compute_verdict`.

**Prior Art**
Relevant citations:
- ULD, OT for cross-tokenizer logit distillation: https://arxiv.org/abs/2402.12030
- MultiLevelOT, token and sequence-level OT for cross-tokenizer KD: https://arxiv.org/abs/2412.14528
- CDM, contextual dynamic mapping for sequence/vocab mismatch: https://aclanthology.org/2025.findings-acl.419/
- EMO, cross-tokenizer embedding-model distillation with MinED, CKA, and OT hidden-state alignment: https://aclanthology.org/2025.emnlp-main.385/

Bottom line: run it, but make the first claim narrow: “trained Qwen interface embeddings can be transcoded through a sparse tokenizer alignment graph.” Do not claim universal geometry unless the shuffled-row/random-source controls fail cleanly.

