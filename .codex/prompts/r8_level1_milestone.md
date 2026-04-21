You are the senior architectural authority for the Neural Genome atlas moonshot. Act as the 3 Phase-Transition personas from CLAUDE.md §7.3: Architecture Theorist + Research Integrity Auditor + Competitive Analyst.

## Context

Session 2026-04-21 produced the first atlas-coordinate claim strong enough to warrant your review. The claim is:

**"kNN-10 clustering coefficient passes Gate-1 portability at strict δ=0.10 on 7 of 8 architecture classes spanning 5 distinct training objectives: autoregressive (CLM), masked-LM (MLM), contrastive text, self-supervised vision ViT, contrastive vision. Falcon-H1 hybrid narrow-fails at n=2000 but passes cleanly at n=4000. This constitutes evidence for Level-1 Gate-1 threshold satisfaction per `research/UNIVERSALITY_LEVELS.md`."**

Files to read (do not accept summaries):

- `research/atlas_tl_session.md` — full session trajectory, heartbeat log at bottom.
- `research/CLAIM_EVIDENCE_MAP.md` — C1..C7 the current locked claims, P1..P5 the pending.
- `experiments/EXPERIMENTS.md` — narrative entries for genome_007 through genome_011.
- `experiments/ledger.jsonl` — raw metrics for every experiment.
- `research/derivations/knn_clustering_universality.md` — LOCKED at 62338b8. This is the Gate-2 immutable artifact.
- `research/prereg/genome_knn_k10_portability_2026-04-21.md` — LOCKED Gate-1 prereg.
- `research/prereg/genome_knn_k10_batch2_2026-04-21.md` — STAGED Batch-2 prereg (8-class extension).
- `research/prereg/genome_knn_k10_causal_2026-04-21.md` — STAGED G2.4 causal prereg.
- `research/prereg/genome_knn_k10_hierarchical_2026-04-21.md` — STAGED G2.3 hierarchical prereg.
- `research/prereg/genome_knn_k10_biology_2026-04-21.md` — STAGED G2.5 biology prereg.
- `results/gate1/stim_resample_n2000_8class_full.json` — the 8-class Gate-1 verdict JSON.
- `results/gate1/stim_resample_n4000_seeds42_123_456_falcon.json` — Falcon n=4000 tip verdict.
- `results/gate1/quant_stability_n2000_seed42.json` — G1.5 4-class verdict.
- `results/gate2/hierarchical_fit_smoke.json` — G2.3 smoke fit result (underdetermined at k={5,10}).
- `code/genome_primitives.py` — primitive implementation; `knn_clustering_coefficient`.
- `code/genome_cross_arch.py` — extraction runner.
- `code/genome_stim_resample.py` — equivalence criterion machinery.
- `code/genome_ablation_schemes.py` — G2.4 schemes.
- `code/genome_causal_probe.py` — G2.4 runner.

Also read:
- `CLAUDE.md` (project-level operating manual)
- `../CLAUDE.md` (parent AI-Moonshots constitution — the manifesto)
- `COMPUTE.md` (binding envelope)

## Respond to 6 questions. Be brutal.

### Q1 — Architecture Theorist

The LOCKED derivation predicts `C(X,k) = α_d (1 − β_d·κ·k^(2/d_int))₊ + O(n^(-1/2))` from Laplace-Beltrami convergence of kNN graphs. The 8-class G1.3 result shows kNN-k10 values clustering in [0.28, 0.36] across architectures.

1a. Is the Laplace-Beltrami argument actually sound for the point clouds produced by these 8 architectures? In particular, the argument assumes **iid sampling from a smooth manifold with bounded density**. Trained LLM hidden states are (a) not iid (sentences are correlated), (b) may not form a smooth manifold (could be a stratified space, a union of charts, etc.), (c) density may be unbounded (layer-norm creates sphere-like distributions). State which of these assumptions we're getting away with violating, and which should be falsification-tested.

1b. The derivation should predict HOW kNN-k10 varies with `d_int` and `k`. With only k ∈ {5, 10} measured we cannot test the functional form — smoke fit (results/gate2/hierarchical_fit_smoke.json) confirmed underdetermination. What k-values do you recommend as the G2.3 sweep, and why? Would a measurement at k=3 or k=100 be MORE useful than your recommended sweep for identifying β_d?

### Q2 — Research Integrity Auditor

2a. Read `results/gate1/stim_resample_n2000_8class_full.json`. Confirm or refute: the 7/8 pass verdict is correctly computed from the underlying atlas rows via the equivalence criterion `|Δ| + c·SE < δ·median(|f|)`. Is `c = 2.77` Bonferroni-correct for K=18 per the current N=8 bestiary (or should K grow with bestiary size)?

2b. The 8-class claim includes CLIP and MiniLM at n=2000 × 3 seeds, but with different pooling semantics than the Batch-1 anchors. CLIP uses `cls_or_mean` on vision transformer output; MiniLM uses `seq_mean` on contrastive text encoder. Does the single-δ threshold correctly calibrate across these pooling differences, or are we leaking unmodeled variance into the "pass" column?

2c. The original LOCKED Gate-1 prereg (`genome_knn_k10_portability_2026-04-21.md`) specified 3 systems (Qwen3/RWKV/DINOv2) and K=18. Adding BERT/MiniLM/CLIP extends scope. Under LOCKED prereg discipline, does this require a NEW locked prereg before the 8-class claim is defensible, or does the existing STAGED Batch-2 prereg (`genome_knn_k10_batch2_2026-04-21.md`) cover it adequately? If the latter, should it LOCK now?

2d. PR_uncentered was empirically demoted after a 5/5 pass because the values were all ≈1 (DC artifact). Apply the same skepticism to kNN-k10: the 8-class values cluster in [0.28, 0.36]. For a random point cloud of n=2000 in h=384 to h=1536 dimensions (the ambient ranges here), what is the EXPECTED random-baseline kNN-10 clustering coefficient? If the measured values are close to the random baseline, kNN-k10 is a "random-geometry artifact" the same way PR_uncentered was a "DC artifact."

### Q3 — Competitive Analyst

3a. Huh et al.'s Platonic Representation Hypothesis (2024) argues representations converge across modalities using linear-similarity metrics. Our result uses kNN-k10. What does the literature say about the comparative robustness of kNN-graph invariants vs linear-alignment metrics for cross-architecture comparison? Name 2-3 specific papers (with years and arxiv IDs if possible).

3b. Anthropic interpretability and DeepMind's Circuit work operate at the feature-direction level, NOT the point-cloud level. Does a point-cloud-level universality claim compete, complement, or contradict the circuit-level approach? If a reviewer from Anthropic read the genome_011 result, what would be their sharpest criticism?

3c. The Aristotelian-View critique (Feb 2026) argued linear similarity metrics like CKA conflate scale with geometry. Does kNN-k10 escape that critique cleanly, or does it have its own analogous problem?

### Q4 — Decision: is it publishable NOW, or what specifically must still happen?

Score the current state against the manifesto's sacred outcomes (S1-S7 per `research/atlas_tl_session.md §0b`). Specifically:

- Is S2 (architecture-agnostic) satisfied or not?
- Is S3 (Level-1 universal, backed by derivation) satisfied, or only the Gate-1 portion?
- Is the 8-class Gate-1 claim worth a workshop paper, a tech report, or neither?

If "neither" — what's the single most important experiment to run next? Rank-order the 3 STAGED Gate-2 preregs (G2.3 hierarchical, G2.4 causal, G2.5 biology) by **expected information value per GPU-hour**.

### Q5 — Critical path to publishable Level-1 claim

Write the next 72 hours of work as an ordered task list. Be specific — name files to create, scripts to run, n-values, systems, success criteria. Prioritize the actions with the highest probability of producing a falsification-or-validation verdict rather than just more data.

### Q6 — What we might be fooling ourselves about

Name the 3 most likely ways the current 8-class result is wrong in a way the Gate-1 machinery doesn't catch. Be brutal. "We got the thing we wanted, therefore our instrument is correct" is the failure mode of bad science — tell me where the instrument could be lying.

---

**Output format:** write to the `-o` file. Max 4000 words total. Use tables for comparisons. Cite specific file:line locations. End with a numbered action list (max 10 items).

Do NOT suggest runs that exceed `COMPUTE.md` envelope (≤22 GB VRAM, ≤56 GB RAM, ≤4 h wall-clock, no cloud).
