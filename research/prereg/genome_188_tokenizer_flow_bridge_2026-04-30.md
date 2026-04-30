# Pre-registration: genome_188 — Tokenizer-Flow Bridge

**Date:** 2026-04-30
**Status:** LOCKED (2026-04-30T08:10:00Z)
**Design gate:** `codex_outputs/g188_tokenizer_flow_bridge_design_gate_20260430.md`

---

## Hypothesis

Trained Qwen3 interface embeddings (embed_in + lm_head) can be transcoded through a sparse tokenizer-alignment graph into GPT-2 tokenizer space, producing anchor targets that recover a meaningful fraction of the within-family trained-anchor effect (+0.513 nats, g181b) when applied to a GPT-2-tokenizer Qwen3-arch recipient.

## Motivation

- g183 FAIL: corpus-derived PPMI SVD embeddings ACTIVELY HURT training (-0.31 nats mean). Corpus co-occurrence statistics have the wrong geometric format for architecture internals.
- g180b FAIL: cross-tokenizer KD HURTS (-0.37 to -0.54 nats). Naive cross-tokenizer transfer is toxic.
- g181b PASS: within-family trained-anchor works (+0.513 nats persistent at 5000 steps).
- Missing piece: a topology-aware bridge that maps trained embeddings between tokenizer vocabularies while preserving the architecture-specific geometric structure.

## Prior art

- ULD (arXiv 2402.12030): OT for cross-tokenizer logit distillation
- MultiLevelOT (arXiv 2412.14528): token and sequence-level OT for cross-tokenizer KD
- CDM (ACL Findings 2025): contextual dynamic mapping for sequence/vocab mismatch
- EMO (EMNLP 2025): cross-tokenizer embedding distillation with MinED, CKA, and OT hidden-state alignment

## Approach

1. Tokenize same C4 corpus spans with both Qwen3 and GPT-2 tokenizers
2. Build sparse bipartite alignment graph via character-offset overlap (NOT sentence co-occurrence)
3. Compute sparse Sinkhorn-balanced coupling on CSR/COO edges (NOT dense 50K OT)
4. For each GPT-2 token, compute barycentric embedding from mapped Qwen3 trained embeddings
5. Use flow-bridged embeddings as anchor target for GPT-2-tokenizer Qwen3-arch recipient
6. Normalize all anchor arms to matched Frobenius norm and comparable anchor gradient norm

## Token subset

Top tokens by train mass: ~20K Qwen3 source × ~10-20K GPT-2 target. Exact mass coverage reported. Rare GPT-2 tokens get fallback embeddings (mean of nearest-mass Qwen3 tokens).

## Arms (all GPT-2-tokenizer Qwen3-arch, 3 seeds [42, 7, 13])

1. `scratch_ce` — no anchor, no init injection (baseline)
2. `flow_bridge_init_anchor` — flow-bridged embeddings as both init + continuous anchor (primary)
3. `flow_anchor_only` — flow-bridged anchor, random init (isolate anchor contribution)
4. `flow_init_only` — flow-bridged init, no anchor (isolate init contribution)
5. `char_overlap_no_ot` — character-overlap direct mapping without OT balancing (ablation)
6. `direct_string_match_anchor` — naive string-match token mapping (strong baseline)
7. `flow_shuffled_qwen_rows` — OT structure preserved, source embeddings row-shuffled (random content)
8. `flow_random_source` — same OT plan, random source embeddings (plan quality control)

## Training protocol

Same as g183: Qwen3-arch model, 5000 steps, ANCHOR_LAMBDA=0.0323, eval_every=100, C4 train/val with 13-gram dedup. Recipient is GPT-2-tokenizer Qwen3-arch (from g180b infrastructure).

## PASS criteria (Codex-approved thresholds)

**PASS (all must hold):**
- P1: flow_bridge_init_anchor beats scratch_ce by >= +0.12 nats mean AND 3/3 paired seeds
- P2: flow_bridge_init_anchor beats char_overlap_no_ot by >= +0.04 nats mean
- P3: flow_bridge_init_anchor beats direct_string_match_anchor by >= +0.05 nats mean
- P4: flow_bridge_init_anchor beats BOTH flow_shuffled_qwen_rows AND flow_random_source by >= +0.08 nats AND 3/3 seeds

**PARTIAL:** +0.05 to +0.12 nats vs scratch, useful but not headline.
**STRONG PASS:** >= +0.20 nats vs scratch (~40% of g181b +0.513 ceiling).

## FAIL criteria

- flow_bridge_init_anchor <= 0 nats vs scratch (HURTS, like g183)
- flow_bridge_init_anchor does not beat flow_shuffled_qwen_rows 3/3 seeds (OT plan carries no real information)
- flow_bridge_init_anchor within +0.02 nats of flow_random_source (source embedding quality doesn't matter)

## Stage B gate

If flow_bridge_init_anchor >= +0.05 nats: run flow_anchor_only and flow_init_only to decompose contribution.

## Compute envelope (COMPUTE.md compliance)

- Alignment graph construction: O(N_spans × L) where L = max span length. ~20K × 20K sparse edges → ~50 MB CSR
- Sparse Sinkhorn: iterative scaling on CSR, ~5 min for 100 iterations
- Barycentric embedding computation: matrix-vector multiply on sparse plan, ~1 min
- Training: 8 arms × 3 seeds × 5000 steps × ~21 min/cell = ~8.4h total. Run in stages.
- VRAM: same as g183 (~18 GB per cell). One cell at a time.
- System RAM: sparse plan + embeddings < 2 GB. Well within 64 GB.
- Wall clock: preprocessing ~30 min, Stage A (scratch + primary + baselines) ~3h, Stage B (conditional) ~1.5h

## §0.1 scoring (from Codex)

- Movement potential: 6.4/10 if revised sparse approach, 4/10 if naive dense OT
- Nobel/Turing: 2/10 as experiment, 4/10 if becomes general tokenizer-transcoding law
- Feasibility: 7/10 sparse offset bridge

## Source files

- Code: `code/genome_188_tokenizer_flow_bridge.py`
- Results: `results/genome_188_tokenizer_flow_bridge.json`
- Cache: `results/cache/genome_188_tokenizer_flow_bridge/alignment_edges.npz`, `flow_plan_topk.npz`

## Core functions

`tokenize_with_offsets`, `build_offset_alignment_edges`, `make_sparse_cost_kernel`, `sparse_sinkhorn_balance`, `barycentric_target_embeddings`, `make_bridge_controls`, `train_cell`, `compute_verdict`
