# Pre-registration: genome_187 — Ultrametric Training Diagnostic

**Date:** 2026-04-30
**Status:** LOCKED (2026-04-30, post Codex design gate approval)
**Design gate:** `codex_outputs/g187_ultrametric_design_gate_20260430.md`

---

## Hypothesis

Token embedding geometry becomes increasingly ultrametric (hierarchically structured) during training. The rate and magnitude of ultrametric convergence differs between embed_in and embed_out (Pythia has no weight tying).

## Literature gap

No published work measures ultrametric violation ratios on text LLM training checkpoints. Prior work: arXiv 2512.20926 measured on protein language models (ProtT5 ultrametricity=0.130 vs SeqVec=16.67). v-PuNNs (arXiv 2508.01010) achieve 99.96% on WordNet hierarchies using p-adic neural networks. TIER (arXiv 2603.08159, KDD 2026) uses cophenetic correlation as a regularization loss for taxonomy-aligned embeddings (prescriptive, not descriptive). Zuniga-Galindo (arXiv 2601.19070) connects p-adic tree structures to DNN thermodynamic limits. Xu et al. (npj Complexity 2024) measure hierarchical structure in MLP weights via "order-rate" but not in token embeddings.

## Models

- Primary: EleutherAI/pythia-160m (768d, 50304 vocab, no weight tying)
- Replication: EleutherAI/pythia-410m, EleutherAI/pythia-1b

## Checkpoints

20 log-spaced: step0, step1, step2, step4, step8, step16, step32, step64, step128, step256, step512, step1000, step2000, step4000, step8000, step16000, step32000, step64000, step128000, step143000.

## Token subset

Top 10,000 tokens by tokenizer rank (deterministic, frozen before measurement).

## Primary metrics

1. **Angular-distance triplet slack** (L2-normalized rows, angular distance = arccos(cos)/pi):
   - mean_slack, median_slack, p90_slack
   - violation_rate_tau_0.01, violation_rate_tau_0.05
   - 500,000 random triplets per checkpoint

2. **Cophenetic correlation coefficient** (CCC):
   - Average linkage and complete linkage on 3,000-token subset
   - CCC = Pearson correlation between original pairwise distances and cophenetic distances

## Controls

- step0 random-init baseline
- Gaussian random embeddings (shape-matched)
- Row-norm-matched random embeddings
- Spectral-matched random embeddings (same singular values, random basis)

## Extra measurements per checkpoint

- Row norm mean/std and norm-frequency Spearman correlation
- Spectral alpha, participation ratio, stable rank, top-PC variance fraction
- embed_in vs embed_out: same metrics independently

## Trajectory PASS criteria

- mean_slack decreases by >=20% from step0 to step143000 for BOTH embed_in and embed_out on Pythia-160m
- violation_rate_tau_0.01 decreases by >=20%
- Spearman rho between log(step+1) and mean_slack <= -0.75
- CCC increases by >=0.05 absolute
- Same direction holds in at least 2 of 3 model sizes
- embed_in and embed_out trajectories differ: normalized AUC gap >=0.10, bootstrap CI excludes 0

## Trajectory FAIL criteria

- mean_slack stays flat or increases during training
- No meaningful difference (>20% relative) between trained final and random-init
- Controls (Gaussian, spectral-matched) produce similar slack trajectories

## Compute envelope (COMPUTE.md compliance)

- Pythia-160m at FP32: ~0.7 GB VRAM (loaded one checkpoint at a time)
- Distance matrix: 10000x10000 x 4 bytes = 400 MB RAM
- Triplet sampling: 500K triplets = fast
- CCC linkage on 3000 subset: O(n^2 log n) = ~1 min
- Wall clock: ~3h for 3 models x 20 checkpoints x 2 matrices
- Well within envelope

## s0.1 scoring (from Codex)

- As trajectory diagnostic only: +0.1 to +0.3 for s0.1
- Nobel/Turing: 1.5/10
- Value: fills genuine literature gap, supports codebook theory
- MOST LIKELY OUTCOME: monotonicity exists but disappears after step/loss/frequency controls

## Source files

- Code: `code/genome_187_ultrametric_training_diagnostic.py`
- Results: `results/genome_187_ultrametric_training_diagnostic.json`
- Figures: `results/figures/genome_187_ultrametric_training_diagnostic.png`
