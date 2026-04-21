# SYSTEM BESTIARY

*Every system class the Neural Genome measures. Organized by class; model IDs pulled from the repo-wide canonical registry at `../../../models/MODEL_DIRECTORY.md`.*

**Never hard-code HF IDs in experiment scripts. Import from the registry via the shim pattern (see `../../../models/README.md`).**

**Never add a model to the atlas that is not in the canonical registry.** If a system needs to be tested and is not there, add it to the registry first in a dedicated commit.

---

## The nine classes

The bestiary is organized into nine classes. The project aims to include ≥1 system from each class in Phase 1 of the atlas. Level-1 universality requires the shape to hold across ≥5 classes.

1. Autoregressive language models
2. Reasoning-tuned language models
3. State-space / linear-attention models
4. Hybrid architectures
5. Diffusion models (language and image)
6. Self-supervised vision encoders
7. Joint-embedding predictive architectures
8. World models
9. Untrained / biological controls

---

## Class 1 — Autoregressive language models

**Purpose.** The canonical baseline. Most prior interpretability work lives here; our primitives must also work here, or they do not generalize.

**Minimum atlas coverage.** 3 architectures × 2 scales each.

**From the canonical registry:**

| Model ID | Family | Scale | Tier |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | qwen3 | 0.6B | 1 |
| `Qwen/Qwen3-1.7B` | qwen3 | 1.7B | 1 |
| `Qwen/Qwen3-4B` | qwen3 | 4B | 2 |
| `Qwen/Qwen3-8B` | qwen3 | 8B | 2 |
| `google/gemma-3-1b-it` | gemma3 | 1B | 1 |
| `google/gemma-3-12b-it` | gemma3 | 12B | 3 |
| `HuggingFaceTB/SmolLM3-3B` | smollm3 | 3B | 1 |
| `allenai/OLMo-2-1124-7B` | olmo2 | 7B | 2 |

**Notes.** Qwen3 series provides the cleanest scale ladder for within-family variation tests.

---

## Class 2 — Reasoning-tuned models

**Purpose.** Test whether reasoning training produces a distinct Level-2 family, or reuses the same geometry as base autoregressive LLMs.

| Model ID | Family | Scale | Tier |
|---|---|---|---|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | deepseek-r1 | 1.5B | 1 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | deepseek-r1 | 7B | 2 |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | deepseek-r1 | 8B | 2 |
| `microsoft/Phi-4-mini-reasoning` | phi4 | 3.8B | 2 |
| `microsoft/Phi-4-reasoning` | phi4 | 14B | 3 |

**Key pairing.** Qwen3-7B (base) vs. DeepSeek-R1-Distill-Qwen-7B (reasoning-distilled from the same base): isolates reasoning-training's geometric fingerprint.

---

## Class 3 — State-space / linear-attention models

**Purpose.** The strongest non-transformer alternative. `LLM exploration/` found intrinsic dim is 5.3× higher for SSMs vs. Transformers at comparable scale — this class is where family-level universality lives or dies.

| Model ID | Family | Scale | Tier |
|---|---|---|---|
| `state-spaces/mamba2-130m-hf` | mamba2 | 130M | 1 |
| `state-spaces/mamba2-370m-hf` | mamba2 | 370M | 1 |
| `state-spaces/mamba2-780m-hf` | mamba2 | 780M | 2 |
| `state-spaces/mamba2-1.3b-hf` | mamba2 | 1.3B | 2 |
| `state-spaces/mamba2-2.7b-hf` | mamba2 | 2.7B | 3 |
| `tiiuae/falcon-mamba-7b` | falcon-mamba | 7B | 2 |

**Plus RWKV-7 and xLSTM** (classed as SSM-adjacent for genome purposes):

| Model ID | Family | Scale | Tier |
|---|---|---|---|
| `RWKV/RWKV7-Goose-World3-1.5B-HF` | rwkv7 | 1.5B | 1 |
| `RWKV/RWKV7-Goose-World3-2.9B-HF` | rwkv7 | 2.9B | 1 |
| `NX-AI/xLSTM-7b` | xlstm | 7B | 2 |

**Notes.** Activation patching and path patching must be adapted for SSMs (no attention heads). Development of SSM-compatible causal primitives is a tier-1 task.

---

## Class 4 — Hybrid architectures

**Purpose.** Bridge class between Transformer and SSM. If family-level universality holds, hybrids should sit between the two — not produce their own family. Test of the family-taxonomy itself.

| Model ID | Family | Scale | Tier |
|---|---|---|---|
| `tiiuae/Falcon-H1-0.5B-Instruct` | falcon-h1 | 0.5B | 1 |
| `tiiuae/Falcon-H1-1.5B-Instruct` | falcon-h1 | 1.5B | 1 |
| `tiiuae/Falcon-H1-3B-Instruct` | falcon-h1 | 3B | 2 |
| `ibm-granite/granite-4.0-h-1b` | granite4-h | 1B | 1 |
| `ibm-granite/granite-4.0-h-350m` | granite4-h | 350M | 1 |
| `Zyphra/Zamba2-2.7B` | zamba2 | 2.7B | 2 |
| `Zyphra/Zamba2-7B` | zamba2 | 7B | 2 |
| `LiquidAI/LFM2-1.2B-Exp` | lfm2 | 1.2B | 1 |
| `LiquidAI/LFM2-2.6B-Exp` | lfm2 | 2.6B | 1 |

**Notes.** LiquidAI uses a genuinely novel LIV-convolution + GQA architecture; interesting as a potential third family point.

---

## Class 5 — Diffusion models

**Purpose.** Non-autoregressive generative models. Diffusion operates by iterative denoising, not next-token prediction — a different training dynamic. If geometry universality holds here too, the axiom survives a major stress test.

**Language diffusion:**

| Model ID | Family | Scale | Tier |
|---|---|---|---|
| `GSAI-ML/LLaDA-8B-Base` | llada | 8B | 2 |
| `GSAI-ML/LLaDA-8B-Instruct` | llada | 8B | 2 |

**Image diffusion (add to canonical registry in Phase 1):**
- `stabilityai/stable-diffusion-3-medium` (SD3, 2.6B)
- `black-forest-labs/FLUX.1-schnell` (distilled FLUX)
- A small DiT variant for probeability

**Atlas-specific considerations.** Diffusion has no "layer depth" in the transformer sense; the geometry unfolds across *noise steps*. Need diffusion-specific primitives (see `MEASUREMENT_PRIMITIVES.md` §8.1).

---

## Class 6 — Self-supervised vision encoders

**Purpose.** No language at all. Tests whether the atlas is language-dependent or genuinely cross-modal.

**To add to canonical registry (Phase 1):**
- `facebook/dinov2-small`, `dinov2-base`, `dinov2-large`
- `facebook/sam-vit-base`
- `facebook/vit-mae-base`
- `google/vit-base-patch16-224`

**Atlas notes.** These systems have well-matched biological analogues (ventral stream), making them high-value for the biology validation phase.

---

## Class 7 — Joint-embedding predictive architectures (JEPA)

**Purpose.** LeCun's proposed alternative to autoregressive / diffusion generative modeling. Trained by self-distillation on representation-level prediction, not pixel or token reconstruction.

**To add to canonical registry:**
- `facebook/vjepa2-vit-huge-384` — V-JEPA 2 (Meta, 2024–2025)
- `facebook/ijepa-vit-huge-14-448` — I-JEPA

**Atlas-specific considerations.** Dual-encoder architecture. Encoder vs. predictor may have different geometric fingerprints — a novel sub-question. Primitives from §8.2 of `MEASUREMENT_PRIMITIVES.md` apply.

---

## Class 8 — World models

**Purpose.** Learned models of environment dynamics. Different training signal (next-state prediction conditioned on action) — the genome test for action-grounded representations.

**Candidates (add to registry in Phase 1):**
- **Dreamer v3** — `danijar/dreamer-v3` (open implementation); Hafner et al. 2023. RSSM (recurrent state-space model) structure.
- **Nvidia Cosmos** (if weights released) — large-scale video world model.
- **Genie-style** or small reproductions for laptop-scale probing.

**Atlas-specific considerations.** World models have latent state dynamics; geometric primitives must handle time-evolving latents, not just single activations. Dreamer's RSSM explicitly decomposes into stochastic + deterministic state — an interesting prior on "where content lives vs. where fluency lives."

---

## Class 9 — Controls

Three control populations, each testing a different confound.

### 9a — Untrained networks

**Purpose.** Separates "trained geometry" from "architecture geometry." If the atlas finds the same structure in untrained nets, the geometry is architectural, not learned.

**Candidates.**
- Random-init Qwen3-0.6B (weights reset)
- Lottery-ticket subnetworks (pair with `moonshot-j-self-construction/` outputs)
- Networks with only first-layer training (frozen rest)

### 9b — Active-inference / free-energy-principle systems

**Purpose.** A genuinely different training paradigm — variational free energy minimization, not gradient descent on a task loss. Tests whether the axiom depends on backprop specifically.

**Candidates.**
- Verses AI / Axiom "Genius" (if weights or API access available)
- Small hand-built active-inference agents with neural-net encoders
- Published deep active-inference implementations

**Caveat.** Verses's code is not open-source; may be API-only. May require building a small active-inference reference implementation in-house.

### 9c — Biological neural recordings

**Purpose.** The ultimate generalization test. If the axiom extends to biology, it becomes a fact about learning systems; if not, it is a fact about trained neural networks.

**Datasets.**
- **Allen Institute Neuropixels** — mouse visual cortex (V1, V2L, V1p, VISl, VISal, VISp). CTI already validated 32 sessions here. Use the same access pattern (`remfile + h5py + dandi`; do NOT use `allensdk`).
- **BrainScore benchmark tasks** — for ML-to-brain alignment scoring.
- **Visual Brain Encoding Study (VBES), Algonauts, NSD** — fMRI / MEG candidates if Allen is saturated.

---

## Bestiary growth rules

1. A new class is added only with a pre-registration justifying why the existing 9 are insufficient.
2. New models within an existing class are added freely but must be in the canonical registry first.
3. Models that fail to load, produce NaN, or require dependencies we cannot install are marked ⛔ in this file and excluded from the atlas until fixed.
4. Every ledger entry records which subset of the bestiary it used.

---

## Phase 1 minimum viable bestiary

For the first atlas iteration, we cover ≥1 system per class:

| Class | Representative | Tier | Notes |
|---|---|---|---|
| 1 Autoregressive LLM | `Qwen/Qwen3-0.6B` | 1 | Canonical anchor |
| 2 Reasoning | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1 | Same base as Qwen2.5 |
| 3 SSM | `state-spaces/mamba2-370m-hf` | 1 | Cleanest pure SSM |
| 4 Hybrid | `tiiuae/Falcon-H1-0.5B-Instruct` | 1 | Active sub of Mamba+Attn |
| 5 Diffusion | `GSAI-ML/LLaDA-8B-Instruct` | 2 | Language diffusion |
| 6 Vision encoder | `facebook/dinov2-small` | 1 | Add to registry Phase 1 |
| 7 JEPA | `facebook/ijepa-vit-huge-14-448` | 3 | May need smaller variant |
| 8 World model | Dreamer-V3 small | — | Build or port |
| 9 Control | Untrained Qwen3-0.6B + Allen V1 | — | Paired controls |

If every Phase-1 system loads and a shared primitive produces readable numbers on each, the project has its first atlas row.
