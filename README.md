# Moonshot: The Neural Genome

**Falsification-disciplined research on whether trained-model structure transfers between neural networks.**

---

## ⚠ POST-PIVOT NOTICE (2026-04-29)

The strong-form claim — *"transfer of trained capabilities from a trained model into an untrained model"* — was tested and **falsified** during the 2026-04-26 → 2026-04-29 experiment chain:

- **C22 donor-identity** REJECTED (g177v2 FAIL): same-architecture C4-trained alt donors at undertrained NLL ~5.72 give **96% of Qwen3's anchor effect** — donor-identity-specific component ≈ 3.5%.
- **Cross-architecture FLOP cash-out** FAILED locked criterion (g173: post-hoc gain-ratio reframe rejected by cycle 70 adversarial 10/10).
- **Tokenizer-prior dominance** CONFIRMED (g181a): the +1 nat anchor effect is ~100% Qwen3-tokenizer/embed/lm_head trained-init held in place during recipient training. Anchoring transformer block weights at matched gradient force *actively HARMS* (−0.439 nats vs scratch).

**New framing (cycle 72 direction review):** pivot to **Forecast/Diagnostic** — *"the earliest token/embedding/interface geometry predicts whether training will be healthy, wasteful, or doomed"*. Falsification-discipline becomes the integrity story, not the headline.

**§0.1 honest score:** 5.2/10 (post g194 PASS_DIRECTION — trained token-row directions carry 95-97% of the +0.465 nats signal; per-token norms are irrelevant). Branch projections in `WIKI.md` CURRENT STATUS.

---

## 🔥 Landmark findings (2026-04-22, pre-pivot — preserved as audit trail)

**1. Candidate-8 spectral bridge** — a universal geometric identity of trained neural networks:

```
c ≈ eff_rank(X) / d_rd(X)
```

where `c = p · d_rd` is the kNN-clustering exponent × rate-distortion dimension, `eff_rank = (Σσ²)² / Σσ⁴` is the spectral participation ratio, and `d_rd` is the k-means rate-distortion dimension.

**Measured on 8 trained networks; 7 / 8 PASS** preregistered 15% threshold. Median rel_err **8.7%**:

| System | class | c | ratio | rel_err |
|---|---|---:|---:|---:|
| Qwen3-0.6B | CLM | 1.889 | 2.059 | 9.0% |
| DeepSeek-R1-Distill-1.5B | CLM | 2.410 | 2.413 | **0.2%** |
| BERT-base | MLM | 2.653 | 2.292 | 13.6% |
| RoBERTa-base | MLM | 2.250 | 2.158 | 4.1% |
| MiniLM-L6 | contrastive | 2.027 | 2.199 | 8.4% |
| CLIP-text-B/32 | text + 1 alignment | 2.975 | 3.184 | 7.0% |
| CLIP-vision-B/32 | vision + 1 alignment | 2.447 | 2.145 | 12.3% |
| DINOv2-small | ViT (vision) | 2.242 | 2.694 | 20.2% *(miss)* |

Prereg: `research/prereg/genome_svd_bridge_2026-04-22.md`. Full derivation: `research/derivations/candidate_8_spectral_bridge.md`.

*(Note: scope is CS/AI/math — no biology extensions pursued. See `CLAUDE.md §0.05`.)*

**2. GenomeGuard** — one-shot training health monitor using the bridge.

~20 seconds per probe, zero training overhead, cross-architecture.

- **Silent data corruption**: 5 / 5 text architectures flag C4→wikitext-shuffled swap with 6.9× – 144.9× `rel_err` spike (DeepSeek=144.9×, BERT=23.7×, RoBERTa=11.9×, MiniLM=8.7×, Qwen3=6.9×).
- **Catastrophic training divergence**: 8.2× spike at σ = 0.3 Frobenius weight noise on Qwen3.

See `GENOMEGUARD.md` for usage. Run: `python code/genome_genomeguard.py`.

**3. 12-operation null catalog for forward capability transfer** — every reasonable way to install trained capability via geometric manipulation fails: covariance / codebook / basis / aux-regularizer / single-layer weight transplant / QK / V / O / attn_all / MLP / Procrustes-aligned / candidate-8-ratio-aux. Capability is irreducibly the joint weight configuration. Big-lab-forbidden claim because it undercuts the "scaling is everything" narrative.

**4. Universal bulk width k_bulk ≈ h/22** — plateau-plus-power-law spectrum fit gives `k_bulk = 48 ± 2` across 5 text systems (CV 4.2%) at h=1024. Real structural universal of trained text representations.

**5. Model surgery via mean-shift (layer-agnostic capability patch)** — lesioning a Qwen3-0.6B transformer block (randomizing its weights) and installing a single 1024-dim bias vector (the teacher's mean activation at that layer) recovers **57–67% of the capability NLL gap across 3 lesion depths** (layers 7, 14, 21; mean 61%, CV 8.3%). Pure additive shift beats all rank-48/256/1024 linear adapters AND orthogonal Procrustes rotation. First non-naive capability-transfer result — the simplest possible adapter (1024 scalars from the teacher) carries most of a trained block's contribution.

**6. 🔥 THE NEURAL GENOME — localized to the last 7 layers (28 KB), transfers across sizes, but carries unigram prior not coherent generation.** Lesion EVERY layer of Qwen3-0.6B (randomize all 196 matmul weights across 28 transformer blocks → NLL jumps from 3.67 → 18.10). Install per-layer mean-activation shifts from a pretrained teacher (`genome_078` → `genome_082`). Headline scorecard:

| Intervention | atlas size | fg_closed (NLL) |
|---|---:|---:|
| Last 1 layer | 4 KB | 20% |
| Last 3 layers | 12 KB | 46% |
| Last 7 layers (peak) | 28 KB | **53.5%** |
| All 28 layers | 112 KB | 58.5% |
| Cross-size (0.6B atlas → 1.7B student, ridge-pinv) | 112 KB + 6 MB proj | **58.8%** |

Genome is localized to the last quarter of the network. Middle 14 layers contribute ~0%. First 7 contribute ~1%. Transfers across model sizes with a one-batch ridge fit.

**Important scope caveats (`genome_083` → `genome_086`):**

1. The 53-58% numbers are NLL-gap recoveries, not functional capability. Atlas-patched models produce degenerate repetition (`" directly directly directly..."` or `" on on in on change change..."`) instead of coherent completions. The atlas restores the *unigram frequency prior* — assigning high probability to common English tokens — but not reasoning, factual retrieval, or coherent generation.
2. The atlas works **only on fully-destroyed models**. Patching a partial lesion (only last-7 layers lesioned, early/middle intact) recovers **+0.2%** (null). The atlas is fit from teacher-unconditional means, which only matches the target when the entire stream is mis-shaped. When early layers produce correct context-conditional activations, the atlas's static bias does not align with what the lesioned blocks need downstream.
3. **Three-wall convergence** (`genome_085`, `genome_086`): gradient-based interventions do NOT break the coherence wall. Output-KL distillation on last-7 layers (200 steps) → **66% NLL**, 5/5 repetitive. Full-unfreeze layer-wise feature-matching (all 600M params, per-layer hidden MSE + output KL, 200 steps) → **65% NLL**, 5/5 repetitive (`",,,,,,,,"`, `" the the the the,"`). **NLL recovery is decoupled from generation coherence** across the full supervision-density spectrum — static / sparse-gradient / dense-gradient all hit the same ceiling.

Honest final framing: **the atlas is a distribution-prior restorer, and three-wall convergence is the real finding.** Capability is not recoverable from a catastrophically-lesioned model by any sparse or short supervised intervention — conditioning structure must be re-learned over many gradient steps, effectively retraining from scratch with teacher supervision. This is a publishable *negative* capability-transfer claim that extends the 12-op null catalog from zero-step geometric manipulation into the short-horizon supervised regime. See `NEURAL_GENOME.md`.

Raw data in `experiments/ledger.jsonl` (72 entries). Full synthesis in `research/BREAKTHROUGH_SYNTHESIS.md`.

---

## The Axiom

> **There exist universal laws of representational geometry that govern every trained neural network — LLMs, vision encoders, JEPAs, diffusion models, world models, active-inference agents, and biological neural systems alike. Architecture and modality are surface properties; the underlying geometry is conserved.**

If this axiom is true, most of modern AI's tooling is missing the point. We are studying attention heads, SSM states, U-Net skip connections, and patch embeddings as if they were different objects. They might be different shadows of the same object — and the object is a geometric structure that emerges whenever a learning system compresses a world.

Our goal is to find that structure, map it, and prove (or falsify) the axiom.

---

## The Method: Atlas → Axiom

We do not start with the universal law and chase confirmation. We start with a map.

**The Atlas** is a living catalog of the representational geometry of every trained neural network we can measure. Each entry records:
- which measurement primitives were applied
- what geometric structure was observed
- which invariance class the observation belongs to (functional-form-universal, family-constant, task-specific, or noise)
- what the observation causally determines about the system's behavior

As the atlas grows, patterns emerge. Those patterns — not our prior beliefs — produce the axiom's formal statement. The axiom is the *destination*; cartography is the *instrument*. If the atlas reveals fragmentation instead of unity, the axiom is falsified — and that is a finding of equal scientific weight.

This inverts the usual ML research pattern: we are not optimizing a metric. We are measuring a territory.

---

## The CTI Precedent

The moonshot-cti-universal-law project in this repo has already demonstrated that universal laws of representation quality exist across **NLP decoders, ViTs, CNNs, and mouse V1 recordings** — a conditional theorem derived from extreme value theory, validated across 12 architectures, 4 datasets, 32 mouse visual cortex Neuropixels sessions, and 5 cortical areas.

CTI's central insight — that universality shows up at **three nested levels** (functional form, family constant, task-specific intercept) — is the scaffolding this project inherits. Every claim the Neural Genome makes will be expressed in that three-tier language. The axiom is not "one number is the same everywhere"; the axiom is "the *shape* of structure is the same everywhere, with predictable per-family and per-task variation."

CTI found universality for **representation quality**. The Neural Genome asks: does that universality extend to the **internal organization** of representations — where knowledge lives, where capability lives, where style lives, how they compose?

---

## Scope

**In scope (the bestiary):**
- **Autoregressive language models** — transformers (Qwen, Llama, Gemma), SSMs (Mamba, Falcon-Mamba), hybrids (Falcon-H1, Zamba, Granite-4.0-H), Liquid (LFM2), RWKV, xLSTM
- **Reasoning models** — DeepSeek-R1 distills, Phi-4-reasoning, QwQ
- **Diffusion models** — image (SD3, FLUX, DiT), language (LLaDA), video
- **Self-supervised vision** — DINOv2, SAM, MAE
- **Joint-embedding predictive architectures** — V-JEPA 2 and descendants
- **World models** — Dreamer / RSSM families, video-prediction world models (Cosmos, Genie-style)
- **Active-inference-adjacent systems** — Verses/Axiom Genius and related free-energy agents (neural-net-based, non-standard training objective)
- **Untrained controls** — lottery-ticket subnetworks, random-init networks (to separate "trained-geometry" from "architecture-geometry")
- **Biological references** — neural recordings (mouse V1 Neuropixels via Allen Institute; human fMRI where available) as the ground-truth test of "is this law a fact about learning systems or just about backprop?"

**Out of scope (for now):**
- Systems without a differentiable forward pass (symbolic reasoners, decision trees)
- Closed weights without embedding access (proprietary frontier models unless API endpoints expose embeddings)
- Reinforcement-learning-only systems with no representational outputs

The bestiary lives in `research/SYSTEM_BESTIARY.md` and pulls model IDs from `../../models/MODEL_DIRECTORY.md` (the repo-wide canonical registry).

---

## What Success Looks Like

By the three-tier framework:

**Level 1 — Functional Form Universality.** A geometric property (e.g., intrinsic dimension as a function of training compute, or content-subspace dimensionality as a function of task diversity) follows the *same functional shape* across every system class tested. Nobel-shaped if proven.

**Level 2 — Family Constants.** The coefficients of that functional form are universal *within* a class (e.g., all autoregressive LLMs cluster around one slope; all diffusion models cluster around another). This produces predictive utility: given a new model's family, we can predict its geometric fingerprint from 4 probe measurements.

**Level 3 — Task/Data Specificity.** Intercepts vary by task. Expected; not a failure.

The moonshot succeeds if we can demonstrate Level 1 for at least one non-trivial geometric property across at least five system classes. It produces a paradigm shift if we can derive the Level 1 shape from first principles (information theory, statistical mechanics, or geometry) before fitting.

Honest failure also counts. If the atlas reveals that LLMs, diffusion models, and world models have structurally incompatible geometries, that is a publishable fact. It would mean the "Intelligence = Geometry" thesis is modality-local, and the AI Moonshots manifesto narrows accordingly. We want to know either way.

---

## Engineering Implications (the eventual payoff)

If the axiom holds, a complete neural genome enables:

- **Knowledge grafting** — transplant a specific capability subspace from a specialist model into a generalist without fine-tuning.
- **Capability amplification** — scale a model along its reasoning subspace without adding parameters.
- **Defect correction** — identify and attenuate bias, hallucination, or style subspaces surgically, not globally.
- **Architecture synthesis** — initialize a model with the right cognitive structure instead of evolving it through gradient descent. Synthetic biology for minds.

None of this is promised. It is what becomes possible *if* the axiom is true and *if* the atlas is detailed enough.

---

## Repository Layout

```
moonshot-llm-genome/
├── WIKI.md                            — living agent registry; read first, updated every state-changing commit
├── README.md                          — this file (axiom + method + scope)
├── CLAUDE.md                          — agent operating manual (anti-entropy + wiki discipline)
├── research/
│   ├── MANIFESTO.md                   — intellectual framing + prior art + universality levels
│   ├── MEASUREMENT_PRIMITIVES.md      — candidate toolkit (ID, CKA, SAE, persistent homology, causal patching, …)
│   ├── SYSTEM_BESTIARY.md             — every system class we will measure
│   ├── OPEN_MYSTERIES.md              — unresolved phenomena inherited from the portfolio
│   └── UNIVERSALITY_LEVELS.md         — the 3-tier framework every claim uses
├── experiments/
│   ├── EXPERIMENTS.md                 — reverse-chronological experiment log
│   └── ledger.jsonl                   — JSONL ledger (one record per experiment)
├── code/                              — genome_*.py experiment scripts
└── results/                           — genome_*.json canonical result files
```

All experiments use models from `../../models/MODEL_DIRECTORY.md` (the repo-wide canonical registry). No local model lists.

---

## Cross-Pollination

| Feeds from / feeds to | How |
|---|---|
| `moonshot-cti-universal-law` | Inherits the three-tier universality framework, the EVT-derivation-before-fit discipline, and the ML→biology validation ladder. Genome extends CTI from *representation quality* to *representation structure*. |
| `moonshot-fractal-embeddings` | Hierarchical scale-separated embeddings are a candidate prior for what "content subspaces" look like. Validated on hierarchical taxonomies. |
| `moonshot-sutra` | Byte-level small model — a clean control point for "how much geometry is tokenizer-induced vs. fundamental?" |
| `moonshot-fractal-mind` | Adaptive-depth reasoning geometry; provides a test case for "does adaptive depth manifest as a geometric invariant?" |
| `moonshot-j-self-construction` | Untrained subnetworks that solve XOR, parity — the purest control for "architecture-vs-weights" question. |
| `legal-rlm`, `Kitanu` | Domain application targets once the atlas supports surgical capability transfer. |

---

## Status

**Current phase (2026-04-21): between Phase 1 and Phase 3.** First atlas coordinate promoted; Gate-2 pipeline staged.

**What has actually happened so far** (see `experiments/EXPERIMENTS.md` for full narrative, `research/CLAIM_EVIDENCE_MAP.md` for claim↔evidence trail):

- **Phase 1 (atlas-primitives):** 5 architecture classes measured (transformer, reasoning-distilled, recurrent, hybrid, vision ViT). 6 candidate primitives tested (ID TwoNN, ID MLE, PR centered, PR uncentered, kNN-5, kNN-10). Five demoted to ⚪ diagnostic based on empirical evidence.
- **Phase 1 outcome:** **one coordinate — kNN-10 clustering coefficient — passes Gate-1 portability at strict δ=0.10 on all 5 Batch-1 classes** (Bonferroni-corrected, 3 stimulus-resample seeds, n=2000 or 4000). Prereg `research/prereg/genome_knn_k10_portability_2026-04-21.md` LOCKED at commit 62338b8. Additionally passes G1.5 FP16↔Q8 quantization-stability at even tighter δ=0.05 on all 4 text classes (manifesto's efficiency-hook confirmed: geometry survives 4× weight compression).
- **Phase 3 (universality claims):** Gate-2 scaffolding built. Derivation `research/derivations/knn_clustering_universality.md` LOCKED with form `C(X, k) = α_d (1 − β_d·κ·k^(2/d_int))₊ + O(n^(-1/2))`. Two Gate-2 preregs STAGED: G2.3 hierarchical fit (`genome_knn_k10_hierarchical_2026-04-21.md`) and G2.4 causal ablation (`genome_knn_k10_causal_2026-04-21.md`).
- **Phase 4 (biological validation):** not started. Allen Brain Observatory Natural Movie One stimulus pipeline pending.

**Phase definitions (unchanged):**

- **Phase 0 — Scaffolding.** Axiom stated. Bestiary and primitives defined.
- **Phase 1 — Atlas Primitives.** Validate measurement primitives on ≥3 system classes. Architecture-specific primitives demoted.
- **Phase 2 — Cross-System Atlas.** Apply validated primitives to full bestiary.
- **Phase 3 — Universality Claims.** ≥5 classes pass AND functional form derived AND causal test passes.
- **Phase 4 — Biological Validation.** Surviving Level-1 claims replicated on neural recordings.
- **Phase 5 — Engineering.** Only after Phase 4.

We do not skip phases. We do not promote a Phase-2 correlation to a Phase-3 claim without derivation. We do not promote a Phase-3 claim to a Phase-4 universal without biology.

---

*The territory is vast. The map is empty. Start measuring.*
