# WIKI — Neural Genome

**The living project registry. Agents read this first, always. Agents update this after every experiment, finding, or doc change — in the same commit. Never stale.**

Entries are pointers (≤500 chars). If an entry needs more depth, link to the canonical doc. This file is an **index**, not a document.

Codex's Cross-System Auditor checks WIKI consistency at every PR gate. A commit that changes experiment results / docs / primitives without a corresponding WIKI patch is rejected.

---

## ⚠ SCOPE LOCK — CS/AI/MATH ONLY (read first) ⚠

We are a CS / AI / math research group. End goal: **map the learning of every AI model** so we can diagnose capability, perform model surgery (transfer a capability from Model A into Model B without retraining), and ship tools for ML practitioners. **Biology experiments are DEPRIORITIZED.** We borrow biological principles as inspiration but do not replicate biology in this repo. See `CLAUDE.md §0.05` for the full scope lock. Any experiment, outreach, or synthesis that drifts into "let's also test on mouse V1 / organoids / cortex" — **stop and redirect**. Partners: Martian / Furiosa / Weka / Liquid AI / VERSES / NVIDIA. They care about capability transfer + efficient inference + geometry of *learned ML representations*.

---

## ⚡ TIER-0 FRAMING — READ BEFORE EVERY ACTION ⚡

**We are ONE independent researcher competing against DeepMind, Anthropic, OpenAI, Google, Meta.** Workshop-grade "we measured X across 9 models" papers are what they already publish monthly. We will not stand out that way. Every action must advance toward: (a) **first-principles derivation**, not phenomenology, or (b) a finding the big labs architecturally cannot/will not publish because it contradicts "scale = capability", or (c) **electricity-grade efficiency** on a real task (10× less compute, match capability). "Tighter error bars" / "one more architecture row" / "another figure" default to NO unless they enable (a)/(b)/(c). See `CLAUDE.md §0.1` for the full framing.

**Current distinctive-direction status (2026-04-21 session):**
- (a) Derivation: **c = p × d_rd modality-stratified training invariant** (text c≈2.07, vision c≈3.18) — genome_036 / 037 / 038 / 039. First quantitatively predictive derivation candidate after A/B/C falsified. Pending: explain the specific integers 2 and 3 from first principles.
- (b) Big-lab-forbidden finding: **training-convergence negative control** (genome_028-033): random-init p spans 22× wider than trained. Big labs won't publish this because "training produces good geometry" isn't their product.
- (c) Efficiency: **pre-registered decision rule** `ΔR²(Q8) ≤ -0.003 → ΔNLL(Q4) ≥ 2%` VALIDATED on held-out Qwen3-1.7B (genome_035). Proof-of-concept, not yet electricity-grade.

**When in doubt, fire Codex with:** *"given current state, which action has the highest probability of producing a finding DeepMind cannot or will not produce?"*

### The 4-rung ladder (Codex-ratified 2026-04-21)

DONE = `Genome Equation` + `Genome Extractor` + `Genome Compiler` (derived law + reproducible extractor + causal transfusion / geometry-regularized training demos). See `research/MANIFESTO.md §0.1` + `research/CURRENT_KNOWLEDGE.md §7-§8`.

| Rung | Goal | Status |
|---|---|---|
| 1 | Stress-test universality (diffusion + JEPA + ≥2 stim banks per modality + LOCO + random-init controls) | **~2/3 done** (7 sys pass `c=p·d_rd`; LOCO + 2nd banks + dynamical stress-test pending) |
| 2 | Close the derivation (no-fit explanation for text c=2 / vision c=3, pre-reg prediction for new modality) | **Not started** |
| 3 | ⚡ **PARADIGM-SHIFT** ⚡ Make geometry causal (intervention moves coordinate → predicted capability change) | **Not started** |
| 4 | Cash it in (scaling-law replacement + ≥10× compute-reduction training demo + biology beyond V1) | **Partial** (decision-rule blind test validated, no training demo) |

**Failure modes named (6-month window):** (i) measurement artifact (collapses under replication), (ii) post-hoc fitted theory (no out-of-sample predictions), (iii) no causal lever (field files as "interesting geometry survey").

---

## How to read this file

- Every section is scannable. Scan it.
- Every claim has a pointer (`→ file:section` or `→ ledger:<id>`).
- Anything undated is current. Anything dated is as-of that date.
- If a pointer is stale, fix it in the same session you noticed it.

## How to update this file

After any of the following, patch the relevant section(s) in the **same commit** as the change:

- Experiment run (any `genome_*.py` with a ledger entry)
- Primitive added, promoted, or demoted
- Bestiary change (system added, marked broken, reclassified)
- Mystery progress (hypothesis confirmed/falsified, new priority, scar flag)
- Finding promoted to a universality level
- Doc created, renamed, or deleted
- Anti-entropy pass (files deleted, merged, renamed)

No "update WIKI later." If the change exists in git, WIKI reflects it.

---

## 1. Project state at a glance

| Field | Value |
|---|---|
| **Phase** | 1 — Instrument live; **first 🟡 coordinate promoted 2026-04-21** |
| **Axiom status** | **G1 + G2.4-text + G2.5-biology all PASS as of genome_034 2026-04-21.** 9 trained neural networks across 7 training objectives produce `C(X,k) = c_0·k^p` with `p = 0.179 ± 0.021 (CV 12.0%), R² mean 0.997` across 27 cells. The 12% CV decomposes into: **text systems converge to `p ∈ [0.158, 0.177]`, vision systems to `p ∈ [0.210, 0.223]`** (Δ ≈ 0.06 modality gap; verified via 4 systems × 3 stim seeds = 12 cells, per-system CV 1.6–3.4%). **Random-init twins span `p ∈ [0, 0.37]` (22× wider)**, across 15 cells → training is a modality-stratified convergence operation toward a shared fixed point. **Biology bridge passes (genome_034, 10/10 Allen V1 sessions at δ=0.10, 8/10 at δ=0.05, prereg criterion cleared by 40 and 20 points respectively)**. Only open criterion for Level-1 is G2.3 theoretical re-derivation (v1 FALSIFIED, 3 of 4 v2 sketches FALSIFIED, framework D untested). |
| **Bestiary coverage** | **9 / ~13 classes measured** (through genome_022 2026-04-21): classes 1 transformer / 2 reasoning / 3 recurrent / 4 hybrid / 6 vision ViT / 7 BERT-MLM / 8 MiniLM-contrastive-text / 9 I-JEPA-predictive-masked / 10 CLIP-contrastive-vision + **NEW: 11 DiT-XL/2-256 class-conditional diffusion transformer (genome_021+022, 3-seed n=2000 cluster-join)**. **kNN-k10 + power-law passes on 9 classes** (Falcon narrow-fail at n=2000, tips at n=4000). Spans **7 distinct training objectives** (CLM + reasoning-distilled + MLM + contrastive-text + self-supervised-ViT + contrastive-vision + predictive-masked + diffusion-denoising). |
| **Promoted primitives (🟢¹/🟢²)** | 0 |
| **Gate-1 passed (🟡 coordinate)** | **1 CLEAN 🟡 + now 9-class extended**: kNN-10 clustering coefficient + power-law form `C(X,k)=c_0·k^p`, scope `(modality ∈ {text, vision}, stimulus_family ∈ {c4_clean.len256.v1, imagenet1k_val.v1}, pooling ∈ {seq_mean, cls_or_mean})` on **Qwen3-0.6B + RWKV-4-169M + DINOv2-small + DeepSeek-R1-Distill-Qwen-1.5B + Falcon-H1-0.5B + BERT-base + MiniLM-L6 + CLIP-ViT-B/32 + I-JEPA-ViT-H/14 + DiT-XL/2-256**. Prereg `research/prereg/genome_knn_k10_portability_2026-04-21.md` LOCKED. **Biology interim (genome_027, n=4/10 sessions)**: 100% pass rate at δ=0.10 vs DINOv2 band. |
| **Active mysteries** | 7 (unchanged; H11-H13 are hypotheses, not mysteries) |
| **Scars (🩹)** | 0 |
| **Active hypotheses (H-register)** | 14 — H1..H10 original + H11 Koopman + H12 stimulus-dominance + H13 quantization-stability + H14 subsample-stability (→ `research/atlas_tl_session.md §1c`). H15 retired to governance rule `research/atlas_tl_session.md §2.5.8` (modality-scope is policy, not falsifiable). |
| **Open pre-registrations** | **2 locked:** `research/prereg/genome_id_portability_2026-04-21.md` (Gate-1 joint ID+PR+kNN — superseded by focused kNN prereg for promotion) and **`research/prereg/genome_knn_k10_portability_2026-04-21.md` (Gate-1 kNN-10 on Qwen3+RWKV+DINOv2, LOCKED 2026-04-21)**. Validator exits 0 on both. |
| **Phase-3 claims** | 0 (Gate-1 ≠ Level-1; v1 derivation FALSIFIED; empirical power law `C(X,k)=c_0·k^p` with **p=0.179±0.021 (CV 12.0%), R²>0.989 mean 0.997 across 27 cells (9 architectures × 3 depths × seeds)** stands as stronger-than-originally-claimed replacement. 2026-04-21 v2-derivation pilots RULED OUT 3 of 4 simple algebraic sketches: **framework A (fractal d_2/d_int) FALSIFIED** wrong-sign structurally (genome_024); **framework B (doubling-dim ratio) FALSIFIED** magnitude-absurd (genome_026); **framework C (heavy-tailed NN-degree) FALSIFIED** wrong-sign (genome_020). Only **framework D (rate-distortion) untested**. All 3 falsifications predict wrong sign or huge magnitude → v2 mechanism likely needs non-dimensional / information-theoretic / correction-to-leading-order class of argument. Pilot details: `research/derivations/power_law_v2_candidates.md`. **Separately (genome_028 negative control, 2026-04-21):** untrained-twin power-law exponents span `p ∈ [0.021, 0.355]` (16.9× spread) on 3 systems vs trained 1.1× spread → training is a CONVERGENCE operation toward the cross-arch universal, not an architectural constant. This is the strongest single manifesto-claim datum collected to date. |
| **Active TL session** | `research/atlas_tl_session.md` — Phase 1-3 drafted; Codex Round 1 complete (8/10), Round 2 running (task `b3fwyis5j`) |
| **Gate semantics** | LOCKED in `research/atlas_tl_session.md §2.5` (two-gate spec + prereg template) |
| **Next phase trigger** | Phase 1 begins when TL session converges to blueprint AND a Gate-1 prereg is locked AND smoke test passes |

→ Phase definitions: `README.md` §Status.

---

## 2. Canonical docs index

Single source of truth per topic. If a topic's pointer goes stale, update here first, then fix the pointer.

| Topic | File |
|---|---|
| Public face | `README.md` |
| Agent operating manual | `CLAUDE.md` |
| **This index** | `WIKI.md` (you are here) |
| **Compute envelope (binding)** | `COMPUTE.md` — read at every design gate |
| Intellectual framing | `research/MANIFESTO.md` |
| Universality framework | `research/UNIVERSALITY_LEVELS.md` |
| Measurement toolkit | `research/MEASUREMENT_PRIMITIVES.md` |
| System bestiary | `research/SYSTEM_BESTIARY.md` |
| Unresolved phenomena | `research/OPEN_MYSTERIES.md` |
| Experiment log (human) | `experiments/EXPERIMENTS.md` |
| Experiment ledger (JSONL) | `experiments/ledger.jsonl` |
| Pre-registrations | `research/prereg/` (one file per experiment, dated, locked) |
| Claim-to-evidence map | `research/CLAIM_EVIDENCE_MAP.md` — every public claim maps to a ledger entry + locked prereg |
| Repo-wide model registry | `../../models/MODEL_DIRECTORY.md` + `../../models/registry.py` |
| **Grafting subproject** | `grafting/OBJECTIVE.md` — geometry-first initialization for efficient training |

Any markdown file not in this table either feeds one of these or should be deleted. (CLAUDE.md §3.4.)

---

## 3. Measurement primitives status

→ Full catalog: `research/MEASUREMENT_PRIMITIVES.md`. Gate semantics locked in `research/atlas_tl_session.md §2.5`.

**Legend (four-tier per §2.5).** 🟢¹ Level-1 universal (Gate 2 passed) · 🟢² Level-2 family-local (Gate 1 on ≥5 classes + family constants) · 🟡 coordinate (Gate 1 on ≥3 classes, portability only, no universality claim) · ⚪ diagnostic (Level-0; class-local or fails semantic comparability) · ⚫ untested.

| Primitive | Status | Classes tested | Last used | Notes |
|---|---|---|---|---|
| Intrinsic dimension (TwoNN + MLE estimator pair) | ⚪ | 4 (class 1,3,4,6) | genome_007 | **DEMOTED to diagnostic.** TwoNN and MLE-k10 fail G1.3 at δ=0.10 on all systems at n=2000 (max_stat 2.4-4.4 vs margin 1.8-2.3). Negative-control test (genome_004) showed only 6-13% trained-vs-untrained gap — measures architecture more than learned geometry. |
| Participation ratio centered | ⚪ | 4 (class 1,3,4,6) | genome_007 | Fails G1.3 at δ=0.10 on all systems (max_stat 3.9-5.9 vs margin 0.8-2.8). Neg-control passes (92% gap) but stimulus-resample too noisy. |
| Participation ratio uncentered | ⚪ | 5 (class 1,2,3,4,6) | genome_009 | **DEMOTED to DC-artifact diagnostic 2026-04-21.** Passes G1.3 5/5 classes at δ=0.10 but values are all ≈1.0 (range 1.01-1.60); PR_centered is 13-39× larger across the same systems. Its 5/5 pass is documenting that all trained networks have a dominant DC activation component (top eigenvector captures ≥95% of uncentered variance), not that they share substantive learned geometry. |
| kNN-10 clustering coefficient | **🟡→🟢¹ text-scope** pending DINOv2+biology | 5 Batch-1 + 3 Batch-2 | genome_013 | **G2.4 CAUSAL PROVISIONAL PASS 2026-04-21.** Gate-1: 5/5 Batch-1 + 7/8 Batch-2 (BERT/MiniLM/CLIP) pass G1.3 at δ=0.10 across 5 training objectives. G1.5 Q8-stability δ=0.05 on 4 text. G2.4 full-grid 3/3 text PASS (Qwen3/RWKV/DeepSeek): topk-ablation effect 7.8-443% at λ=1.0, monotonic, 20-60× specific vs random-10d and PCA-10. Random-Gaussian baseline: trained 0.28-0.36 vs random 0.05-0.08 (4-7× ratio) → NOT an artifact. Gate-2 derivation LOCKED 62338b8. **Remaining for clean green-1:** DINOv2 causal test (needs linear-probe loss), G2.3 hierarchical fit (needs extended k-sweep), G2.5 biology (Allen Neuropixels, STAGED). |
| kNN-5 clustering coefficient | ⚪ | 4 (class 1,3,4,6) | genome_007 | **DEMOTED.** Fails G1.3 at δ=0.10 on 3/4 systems at n=2000 (too noisy at k=5). k=10 is the stable neighborhood size. |
| Koopman spectrum (DMD) | ⚫ | — | — | **NEW** H11 (conf medium). Strongest cross-class candidate by literature (transformer+SSM+diffusion 2025-2026). Deferred to Batch 2 per parsimony. |
| Persistent homology | ⚫ | — | — | Deferred to Batch 2; needs subsampling-stability control. |
| Ricci curvature (Ollivier) | ⚫ | — | — | Deferred to Batch 2. H3a. Null result on SSM/diffusion in 2025-2026 lit — new science opportunity. |
| Fisher info matrix trace | ⚫ | — | — | Harder on JEPAs (no probability output). |
| Lyapunov spectrum | ⚫ | — | — | Needs layer-wise adaptation for feedforward. |
| CKA (linear / RBF) | ⚪ | — | — | **DEMOTED** from 🟡 to ⚪ per Round 1 (scale-confound Feb-2026 Aristotelian-View; PC-dominance). Diagnostic only. |
| NNGS (kNN Jaccard between two embeddings) | ⚪ | — | — | **NEW** cross-system diagnostic (not a per-system coordinate). Level-0. |
| Procrustes / CCA / SVCCA | ⚫ | — | — | Prior: fails cross-arch (p=0.82); informative bound. |
| RSA (cross-system RDM) | ⚫ | — | — | Canonical bridge to biology — stimulus-bank-conditional. |
| SAE (feature decomposition) | ⚫ | — | — | High-risk, family-local per dark-matter lit; see Mystery 7. Phase-N not Phase-1. |
| PCA / SVD spectral | 🟡 | LLM, vision | — | Spectral-slope retired per Codex R2 (fragile, redundant with PR). PCA/SVD still a generic diagnostic. |
| Activation ablation | ⚫ | — | — | Minimum class-agnostic causal primitive; Gate-2 G2.4 requirement. |
| Path / activation patching | ⚪ | transformer | — | **DEMOTED** per Round 1 — transformer-native; Level-0 until class-agnostic extension derived. |
| CAA (direction steering) | ⚪ | transformer-only | — | Demoted — fails on hybrids/SSMs; see Mystery 4. |
| Linear probes | ⚪ | — | — | Diagnostic; cannot imply usability (see Mystery 2). |
| MDL probes | ⚫ | — | — | Pure information-theoretic — works anywhere. |
| Non-linear / MLP probes | ⚫ | — | — | Pair with linear probes for manifold hypothesis test. |
| Task-conditional compression | ⚫ | — | — | Carries over the 99.7%→3D coherent-divergence finding. |
| Successive-refinement D(R) curves | ⚫ | — | — | Direct CTI lineage — strong Level-1 candidate. |
| Diffusion noise-step representations | ⚫ | — | — | No analogue in transformer "layer depth". |
| JEPA predictor/encoder alignment | ⚫ | — | — | Dual-network geometry. |
| World-model latent rollout | ⚫ | — | — | Dynamics-aware, not static. |

---

## 4. System bestiary status

→ Full bestiary: `research/SYSTEM_BESTIARY.md`. Model IDs always pulled from `../../models/registry.py`.

| Class | In atlas? | Phase-1 anchor | Status |
|---|---|---|---|
| 1 Autoregressive LLM | **measured genome_001..007** | `Qwen/Qwen3-0.6B` | ACTIVE in Batch 1 at n∈{5, 500, 2000} |
| 2 Reasoning | 0/N | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Not yet loaded |
| 3 SSM / linear-attention | **measured genome_003..007** | ~~`state-spaces/mamba2-370m-hf`~~ → `RWKV/rwkv-4-169m-pile` | ACTIVE. Mamba2 Windows-blocked (mamba-ssm kernel build fail); RWKV-4 substituted as linear-attention recurrent. |
| 4 Hybrid | **2/3 seeds at n=2000** | `tiiuae/Falcon-H1-0.5B-Instruct` | **UNBLOCKED 2026-04-21** via micro-batched extractor. Natural-fallback Mamba layers work for Falcon at n=2000 with batch_size=64 (was 93 GB OOM in single-batch). |
| 5 Diffusion | 0/N | `GSAI-ML/LLaDA-8B-Instruct` | Not yet loaded |
| 6 Vision encoder | 0/N | `facebook/dinov2-small` | Not in canonical registry yet — add Phase 1 |
| 7 JEPA | 0/N | `facebook/ijepa-vit-huge-14-448` | Not in canonical registry yet — add Phase 1 |
| 8 World model | 0/N | Dreamer-V3 small | Not in canonical registry — may need to build/port |
| 9 Controls | 0/N | Untrained Qwen3-0.6B + Allen V1 | Allen access pattern known from CTI |

---

## 5. Open mysteries — status board

→ Full descriptions: `research/OPEN_MYSTERIES.md`.

| # | Mystery | Hyp. falsified | Hyp. remaining | Status |
|---|---|---|---|---|
| 1 | Orthogonality coupling at 7B+ | 3 | phase-transition, global manifold, depth-capacity | ⚫ unresolved — 1 more falsified hyp → 🩹 scar |
| 2 | Reading/writing asymmetry | 0 | 3 (manifold, distributed, activation-vs-weight) | 🔥 high priority — unlocks Phase 5 |
| 3 | 2 tokens of noise fixes reasoning | 0 | 3 (attention-sink, phase-space, derivable-2) | Cross-class replication is the genome angle |
| 4 | CAA works on tuned transformers only | 0 | 3 (post-tune artifact, non-linear concepts, wrong-layer) | Touches every causal primitive design |
| 5 | Coherent divergence (99.7% content destruction, fluency preserved) | 0 | Cross-class replication pending | Strong Level-1 candidate if replicates |
| 6 | CLIP modality gap | 0 | "gap as Level-2 family constant" | Counter-example to naive universality |
| 7 | SAE feature universality across models | 0 | Feature-vocabulary shared or family-local | Determines if SAEs are coordinate or diagnostic |

---

## 6. Active experiments

→ Running or queued. Full log: `experiments/EXPERIMENTS.md`. Raw: `experiments/ledger.jsonl`.

| ID | Status | Purpose | Systems | Primitive | Pre-reg |
|---|---|---|---|---|---|
| `genome_001_smoke` | ✅ passed 2026-04-21 | First end-to-end pipeline verification | Qwen3-0.6B (trained, FP16) | ID + PR + kNN-clustering | `research/atlas_tl_session.md §3.7` strawman via prereg |
| `genome_002_n500_c4` | ✅ passed 2026-04-21 | First real primitive values (n=500 C4, Qwen3, 2 depths) | Qwen3-0.6B | ID + PR + kNN-clustering | `research/prereg/genome_id_portability_2026-04-21.md` STAGED |
| `genome_003_cross_arch_pilot` | ✅ passed 2026-04-21 (2/3 systems) | **FIRST CROSS-CLASS atlas data** — Qwen3 transformer vs RWKV linear-attention at matched depths on matched stimuli | Qwen3-0.6B + RWKV-4-169M (Falcon-H1 hybrid deferred) | ID + PR + kNN-clustering | STAGED |
| `genome_004_neg_control` | ✅ passed 2026-04-21 | Trained vs untrained negative control — discriminates learned vs architectural geometry | Qwen3-0.6B trained + random-init; RWKV trained | ID + PR + kNN-clustering | STAGED |
| `genome_005_cross_modal` | ✅ passed 2026-04-21 (3/4 systems — Falcon blocked) | **FIRST CROSS-MODAL atlas data** — 3 systems × 3 classes × 2 modalities (text + vision). Clustering coefficient agrees within 0.06 across all three. Strongest Level-1 universality candidate. | Qwen3-0.6B + RWKV-4-169M + DINOv2-small | ID + PR + kNN-clustering (k=5 + k=10) | STAGED |
| `genome_006_stim_resample_g13` | ✅ executed 2026-04-21 | **FIRST formal Gate-1 G1.3 verdicts** — 3 seeds × 3 systems × 3 depths × 6 primitive-estimator cells. Strict δ=0.10: 3/18 pass. δ=0.20 sensitivity: kNN-k10 passes ALL 3 systems. kNN-k10 is the atlas's first 🟡 (δ-sensitive) Level-1 candidate. | Qwen3-0.6B + RWKV-4-169M + DINOv2-small | ID + PR + kNN clustering (equivalence criterion) | STAGED; scale to n=2000 for clean 🟡 at δ=0.10 |

---

## 7. Findings (by universality level)

→ Full derivations and pre-registrations under `research/`.

### Level 1 (functional-form universal)
*(none)*

### Level 2 (family constants)
*(none)*

### Level 3 (task/data intercepts)
*(none)*

### Phase-2 atlas observations (null level — not yet claimed)
*(none)*

---

## 8. Cross-project connections

→ Full context: `_meta/insights/` (sibling path outside this moonshot).

| Other project / moonshot | What it gives us | What we give it |
|---|---|---|
| `moonshot-cti-universal-law` | 3-tier framework; EVT derivation template; biology validation pattern; `cti_allen_*` access scripts | Generalizes CTI's law of representation *quality* into laws of representation *structure* |
| `moonshot-fractal-embeddings` | Hierarchical scale-separated embeddings as a prior on content-subspace structure | Test of whether fractal structure is Level-2 family-local or Level-1 universal |
| `moonshot-sutra` | Byte-level small model — control for "tokenizer-induced geometry" vs. fundamental | Identifies which atlas coordinates are tokenizer artifacts |
| `moonshot-fractal-mind` | Adaptive-depth reasoning — test of "depth-as-geometric-invariant" | Framework for interpreting its adaptive-depth measurements geometrically |
| `moonshot-j-self-construction` | Untrained networks that solve XOR/parity — pure "architecture-vs-weights" control | Rigorous characterization of what their emergent networks represent |
| `LLM exploration/` | Intrinsic-dim and orthogonality-coupling prior data | Inherits Mystery 1; provides cross-class replication |
| `llm-rosetta-stone/` | CAA/steering prior work; cross-arch linear-alignment failure (p=0.82) | Inherits Mysteries 2 and 4; reframes steering as a primitive-agnosticism question |
| `knowledge-surgeon/` | LoRA-based weight-space editing (100% on geography) | Activation-vs-weight asymmetry lens for Mystery 2 |
| `Latent-Space-Reasoning/` | 2-token noise → +19.6pp arithmetic | Inherits Mystery 3 for cross-class replication |
| `llm-platonic-geometry/` | Positive Ricci curvature in LLM embedding spaces, Lyapunov ≈ 0 | Primary prior on atlas-shape expectations |

---

## 9. Decisions log

Architectural / methodological decisions. One line each. Irreversible choices get highlighted.

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-20 | **Axiom-first framing, atlas as instrument.** | CTI template; avoids pre-specified-curve-fit trap. README §The Axiom. |
| 2026-04-20 | **Inherit CTI's 3-tier universality framework.** | Prevents "universality collapsed because one constant varied" failure. `research/UNIVERSALITY_LEVELS.md`. |
| 2026-04-20 | **Architecture-agnosticism gate: ≥3 classes before a primitive becomes a coordinate.** | Keeps LLM-specific tools from masquerading as atlas coordinates. CLAUDE.md §4.3. |
| 2026-04-20 | **Biological validation mandatory for every Level-1 claim.** | Separates "fact about trained NNs" from "fact about learning systems." CLAUDE.md §4.5. |
| 2026-04-20 | **Anti-entropy is a Tier-1 rule, not a cleanup chore.** | Atlas dies under file-bloat within 6 months without this. CLAUDE.md §3. |
| 2026-04-20 | **Repo-wide model registry at `Projects/models/` is canonical; no local model lists.** | One source of truth across every moonshot and research project. |
| 2026-04-20 | **WIKI.md is read first, updated every commit touching state.** | Agents bootstrap fast; nothing goes stale. |
| 2026-04-20 | **9 system classes + Phase-1 minimum viable bestiary defined.** | Concrete, constrained first atlas iteration. `research/SYSTEM_BESTIARY.md` §Phase 1. |
| 2026-04-20 | **`COMPUTE.md` is the binding hardware envelope.** | RTX 5090 Laptop (24 GB VRAM, ≤22 GB usable), Ultra 9 285HX, 64 GB RAM. Every Codex prompt, every prereg must comply with §9 checklist. Out-of-envelope proposals are rejected at design gate. No cloud available. |
| 2026-04-20 | **`moonshot-llm-genome` becomes its own GitHub repo** (pattern matches `moonshot-sutra`). Process docs (`CLAUDE.md`, `WIKI.md`, codex review artifacts) excluded from public push. | Separates mission-public content from agent-process scaffolding. |

---

## 10. Anti-entropy log

Every deletion, merge, and rename. Demonstrates the repo is getting simpler, not just bigger.

| Date | Action | Target | Reason |
|---|---|---|---|
| 2026-04-20 | Migrated | Old `LLM Genome Project/` → deleted in full | Superseded by axiom-first scope; content had diverged from the vision |
| 2026-04-20 | Scaffolded | `moonshot-llm-genome/` | New home under AI Moonshots umbrella |

---

## 11. Retired primitives, archived mysteries, dead ends

Kept for institutional memory. Do not resurrect without reading the retirement reason first.

*(none yet — project just scaffolded)*

---

## 12. Next actions

*(Updated 2026-04-22 T+48h, 13 experiments landed this session + cross-arch GenomeGuard + biology bridge in flight.)* Start here on next session.

**Current state: CANDIDATE-8 BRIDGE + GENOMEGUARD TOOL SHIPPED (cross-arch universal).**

Latest landings (genome_068 + genome_069):
- **GenomeGuard noise-sweep**: 8.2× rel_err spike at σ=0.3 catastrophic weight perturbation (genome_068). Detector has 2 proven failure modes.
- **GenomeGuard cross-arch**: 5/5 text systems (Qwen3, DeepSeek, BERT, RoBERTa, MiniLM) detect C4→wikitext-shuffled swap with **6.9× – 144.9× spike** (mean 39×). Mean baseline → swap: DeepSeek 0.002→0.227, tightest-baseline systems give largest spike (genome_069).
- **Candidate-8 on biology** (genome_biology_bridge): in flight on Allen V1 session 0. If ratio matches c on mouse neurons, candidate-8 extends beyond ML.

README landmark-findings block added. GENOMEGUARD.md updated with cross-arch + catastrophic-divergence tables.

- **Candidate-8 spectral bridge** `c ≈ eff_rank / d_rd` **7/8 PASS** preregistered 15% threshold (Qwen3 9% / DeepSeek 0.2% / BERT 14% / RoBERTa 4% / MiniLM 8% / CLIP-text 7% / CLIP-vision 12% / DINOv2 20%=fail by 5pt). Median rel_err 8.7%, 88% pass rate above prereg 80% target. Derivation-grade universal geometric identity across base text + MLM + contrastive + vision + cross-modal.
- **k_bulk=48 universal** (CV 4.2% across 5 text systems) — plateau-plus-power-law P2 partial fit. Pure power-law falsified.
- **12-op null forward-transfer catalog** (all geometric/weight-subset transplant and aux-loss operations fail to install capability): covariance / codebook / basis / aux-regularizer (eff_rank) / single-layer / QK / V / O / attn_all / MLP / Procrustes-aligned / candidate-8-ratio-aux.
- **Candidate-8 is stimulus-dependent** (bridge BREAKS on wikitext-raw, scrambled, reversed — rel_err rises 3-45×). This is a FEATURE, not a bug: it is the basis of GenomeGuard.
- **GenomeGuard shipping tool** (`GENOMEGUARD.md`, `code/genome_genomeguard.py`, `genome_067`): ~20s per probe; 6/6 (Qwen3+BERT) × (wiki_raw+scrambled+reversed) detect contamination with ≥3× rel_err spike. Silent data corruption detector with zero training overhead.

**Scope lock (2026-04-22):** CS/AI/MATH ONLY. No biology. End goal: capability transfer + model surgery + AI diagnostic tools. See `CLAUDE.md §0.05`.

**🔥 Session T+52h landmark — PHASE-TRANSITION + DERIVATION-GRADE INVARIANT (genome_078 → genome_088):**

Two major landings this session.

**A) Capability recovery from catastrophic lesion is a PHASE TRANSITION, not a ceiling (`genome_087`).**

The 200-step "three-wall" (atlas / output-KL / layerwise-FM all 5/5 repetitive at 49–66% NLL) dissolves at longer horizon. Layer-wise feature-matching + output KL with full-unfreeze on fully-lesioned Qwen3-0.6B over 2000 steps:

| Step | NLL | fg_closed | repetitive |
|---:|---:|---:|:---:|
| 200 | 7.72 | 71% | 5/5 |
| 1000 | 6.74 | 78% | 4/5 |
| **1500** | 6.74 | 78% | **1/5** |
| 2000 | 6.86 | 77% | **0/5** |

Coherence emerges between step 1000 and 1500. Final completions are syntactically coherent English (`"Water boils at → the following game of the city, the first three years"`). The previous "capability is irretrievable from catastrophic lesion" negative claim is REVISED: capability IS retrievable via dense layer-wise supervision, with a sharp phase transition around ~1500 gradient steps. This opens the efficiency question: can geometric auxiliary losses pull the transition earlier?

**B) First trained-ML-specific derivation-grade invariant: `sqrt(eff_rank)·α ≈ 3√2` (`genome_088`).**

Fresh extraction + matched controls on 4 text systems:

| Condition | mean sqrt(er)·α | CV | mean er·α² | CV |
|---|---:|---:|---:|---:|
| Trained | **4.279** | **5.65%** | **18.37** | 11.6% |
| Shuffled | 5.501 | 18.85% | 31.33 | 36.1% |
| Gaussian | 5.505 | 19.07% | 31.41 | 36.6% |

**5.1σ separation** between trained and shuffled/Gaussian. 3√2 = 4.243 (empirical 4.279, 0.85% off). First invariant STRICTLY SPECIFIC to trained ML — biology gives 0.95 due to shallow α=0.20 spectrum. Implies closed-form `eff_rank = 18/α²` for trained spectra. Combined with candidate-4 (`c = d_stim+1`) predicts `d_rd = 18/(α²·(d_stim+1))` — no k-means probe needed. See `research/derivations/trained_spectrum_invariant.md`.

**Atlas is scope-closed** as distribution-prior restorer only. The 200-step "three-wall" is now understood as a training-budget artifact observable at short horizon. The positive story is the phase transition around 1500 steps + the new spectral invariant.

**Further updates 2026-04-22 T+57h (genome_089 → genome_093):**
- `genome_089` — invariant tracks capability recovery as U-shape (mode-collapse-then-expand): eff_rank 78.7 → 4.72 (step 500) → 17.29 (step 2000, near teacher 20.4). **The coherence wall is mode collapse.**
- `genome_090` — NULL at γ=1e-3 weak batch-aux. 13th null-op on record. Trajectories overlap with control.
- `genome_091` — **shifted-power-law σ²=(i+k)^(-2α) FALSIFIED as spectrum model.** Fit gives k=127 (not 5), α=2.09 — reproduces log-log R²=0.85 but predicts wrong eff_rank (3-4× overshoot). My hand-picked (k=5, α=0.8) was a coincidence of invariant arithmetic.
- **Broken-power-law candidate** — numerical fit (k_brk=24, a1=0.4, a2=0.8) reproduces all 4 empirical statistics (er, α, invariant, er·α²) to ~1%. k_brk ≈ k_bulk/2 from genome_047. genome_094 will test fit across 5 empirical spectra.
- `genome_093` — **aux loss IS a spectrum knob but NOT a capability lever.** Buffered K=64, γ=1e-2, target er=16. Aux drove student eff_rank to 22.81 (above teacher 20.44) at step 1000 and sqrt(er)·α to 2.58 (closer to teacher 2.88). But NLL was NOT improved — marginally WORSE (aux 7.02 vs control 6.98 at step 2000; aux 6.90 vs control 6.78 at step 1000 where aux spectrum already matched teacher). **The invariant is a DIAGNOSTIC of mode diversity, not a CAUSAL LEVER for capability.** 15+ null-op catalog, all converging: no sparse geometric intervention transplants capability. Spectrum matching fills directions with optimization noise, not semantic content.

1. **Derive the constant 18 (`genome_091` ruled out shifted power-law).** Remaining candidate paths: (a) two-regime / broken-power-law spectrum fit — flat head + steep tail, with sharp break point; (b) variational form of trained-spectrum extremum under rate-distortion + training-objective. Genome_091 confirms that simple three-parameter rational shapes won't work.
2. **Geometry-as-auxiliary-loss efficiency training** (electricity-grade per §0.1(c)). genome_090 null at γ=1e-3 batch aux (weak leverage). **genome_093 running** — buffered K=64 at γ=1e-2 gives real signal strength. If 093 shows faster coherence emergence than control, first concrete "geometry beats scale" demonstration.
3. **Invariant N≥15 validation.** Extend genome_088 to vision systems (DINOv2, CLIP-vision) + random-init twins + aligned models (Perceiver, VLMs). Target: CV stays < 7% on trained, untrained cleanly separated. N=5 current, target N≥12.
4. **Preprint draft lives at** `research/PREPRINT_DRAFT.md`. Narrative covered: bridge + invariant + mode-diversity mechanism + phase transition + null catalog + GenomeGuard. Section 9 (aux-loss efficiency) pending genome_093 outcome.

**Further updates 2026-04-22 T+66h (genome_097 → genome_109 adversarial cycle):**

Round-1 Codex adversarial review (6 blind spots, `drafts/missing_angles_2026-04-22.md`):
| # | Blind spot | Verdict |
|---|---|---|
| 1 | Shuffled ≠ true random-init control | **REFUTED** (13.5σ separation, `genome_097`) |
| 2 | Probe-window arbitrary | partial (CV tight at fixed window, value shifts with window, `genome_098`) |
| 3 | Unconditional stimulus prior | partial (conditional subsets shift 10-20%, `genome_100`) |
| 4 | Eigenvectors might differ | **REFUTED** (top-30 overlap 0.65 = 17.4× random, `genome_099`) |
| 5 | SFT/RLHF/distill shift | **REFUTED** (CV 1.53% across regimes, `genome_101`) |
| 6 | Fisher/NTK-side invariant | **CONFIRMED** (Fisher CV 37.8% N=4, `genome_102`) — claim scopes to activations only |

Round-2 Codex adversarial review (6 new blind spots, `research/adversarial_review_round2_invariant_2026-04-22.md`):
| # | Blind spot | Verdict |
|---|---|---|
| 1 | Fractional depth wrong axis | **REFUTED** (`genome_109`) — functional-depth band [0.4-0.7] CV 2-4% (TIGHTER than layer-index 5-9%); early [0-0.3] loose as expected |
| 2 | seq_mean pooling is the story | partial refute (last_token also tight at n=800; no_pool breaks CLM/MLM, `genome_107`) |
| 3 | C4 is just dataset adjacency | **partially confirmed** (tight on C4+scrambled-C4, LOOSE on wikitext and random-chars, `genome_108`) |
| 4 | n=800 rank cap | partial confirm (CV loosens at n=3200, `genome_107`) |
| 5 | Cross-class (diffusion/RL) undefined | untested (would require non-transformer systems) |
| 6 | Content direction not pinned | not tested yet (genome_110 planned) |

**Major new positive findings from the adversarial cycle:**
- `genome_103/104` — invariant is a **shared CURVE `f(normalized_depth)` across 5 systems**, tight band [0.4, 0.8] with CV 5-9% per depth. The "universal attractor" is a function of depth, not a single number.
- `genome_105` — cross-stimulus test: **scrambled-C4 preserves universality (word-order not required), wikitext breaks it** (domain shift is the real issue, not syntax).
- `genome_106` — **the attractor is a dynamical fixed point of training.** Pythia-410m at step-0 = 9.55 (random-init-like), step-1k = 3.57 (overshoot), step-143k = 4.09 (settled). Gradient descent actively pulls networks from random-init-land toward the universal attractor.
- `genome_108` mechanistic bonus: trained LMs project OOD text (random chars) to a rank-1-to-7 degenerate subspace at mid-depth (vs rank 24-34 on C4). OOD detection signal by itself.

**Current strongest claim (post round-2):** *trained text LMs converge to a shared activation-cloud curve sqrt(er)·α = f(normalized_depth) in the mid-band [0.4, 0.8] at CV 5-9% per depth, specifically for (sentence-level pooling, n~800, natural-text-like stimuli). The attractor is reached by gradient descent dynamically (Pythia trajectory). Shape AND direction-identity are shared across 5 systems. Random-init is separated by 13.5σ. The invariant is activation-side-only (Fisher CV 38%), domain-sensitive (C4 tight, wikitext loose, OOD collapses), and probe-choice affects specific values but not the tight-CV structure.*

**Explicitly out of scope:** biology, mouse V1, neural recordings.

Key synthesis docs: `research/BREAKTHROUGH_SYNTHESIS.md`, `research/derivations/candidate_8_spectral_bridge.md`, `research/derivations/trained_spectrum_invariant.md` (new 2026-04-22), `research/adversarial_review_round2_invariant_2026-04-22.md`, `GENOMEGUARD.md`, `NEURAL_GENOME.md`.

**NEW SUBPROJECT (2026-04-24): `grafting/`**
Goal: geometry-first initialization via shared transition operators. See `grafting/OBJECTIVE.md`.

**grafting_002 PASS (2026-04-24): transition operator T_l IS genuinely shared across architectures.**
- Cross-prediction R²=0.911 vs within-model R²=0.980 (ratio 0.93) across Qwen3/DeepSeek/BERT
- T_l fit on Qwen3 predicts DeepSeek's next-layer activations with 93% of within-model accuracy
- BERT→Qwen3 and BERT→DeepSeek also transfer (R² 0.72–0.99 across mid-depth band)
- No Procrustes artifact — direct held-out R² test
- **grafting_001** (pairwise cosine sim) had inflated shuffled baseline (Procrustes artifact); **grafting_002** proper cross-prediction is the valid test

**grafting_003 PARTIAL (2026-04-24): analytical lstsq MLP transplant recovers 59% of lesioned capability at zero gradient steps.**
- Donor NLL 3.84 → Lesion NLL 18.14 → Grafted NLL 9.64 (improvement 8.49 nats, ceiling gap 5.80 nats)
- All 28 layers: rank=1500, R²_train=1.0 (underdetermined: 1500 samples in 3072-dim space)
- Minimum-norm lstsq solution fits training activations perfectly but generalizes partially (distribution shift)
- Direction alive: 59% capability recovery at zero gradient steps. Not PASS (ceiling gap 5.80 > 0.5 threshold)
- Failure mode: minimum-norm lstsq does not generalize — geometric content packed into low-rank projection
- Next: **grafting_004** — Ridge regularization (lambda sweep) + overdetermined regime (n=4096 > d=3072) to close ceiling gap

**grafting_004 PARTIAL (2026-04-24): mean-pooled lstsq/Ridge ceiling identified at ~55-60%.**
- All 7 conditions PARTIAL. Best: n=1500, λ=0.1 → NLL 10.08, 55.6% recovered
- Ridge barely helps (55.5%→55.6%). Overdetermined n=4096 is WORSE (35.9%)
- Root cause: mean-pooling over tokens discards token-level structure needed for exact MLP weight recovery. Averaging is too lossy — the MLP operates token-by-token but we fit from pooled sentence vectors. This ceiling (~55-60%) is fundamental to the pooling approximation, not fixable by regularization.
- **Next: grafting_005** — does 55% zero-step recovery provide CE training speedup vs lesioned baseline?

**grafting_005 CONTAMINATED (2026-04-24): CE speedup test showed 2.0× CtQ_75 speedup BUT experiment is invalid.**
- Arm B reuses same model object after Arm A's 300 CE training steps — Arm B attention weights were already trained
- NLL_0 Arm B = 7.94 (not ~10.08 as expected) because attention was already partially trained
- Project gate is >=10× not 2× (see `grafting/OBJECTIVE.md`). 2× is below minimum meaningful threshold.
- `load_texts_at_offset` does not guarantee disjoint compile/train/eval splits
- Result filed as INVALID. `grafting/results/grafting_005_ce_training_speedup.json` retained as reference.
- Codex verdict: "skip grafting_005 as currently written; rewrite as fresh-arm, matched-capacity, frozen-backbone adapter training"

**grafting_006 KILL (2026-04-24): token-level rank-30 adapter bootstrap — CtQ_75 speedup=1.0× (no acceleration).**
- Key fix over grafting_005: fresh model load per arm, frozen backbone, adapter-only training, token-level XtX/XtY
- Arm A (zero-init kaiming): NLL 17.83→7.73@50→5.88@500
- Arm B (token-fitted rank-30): NLL 17.83→8.63@50→6.62@500 — SLOWER than Arm A early
- CtQ_75: both arms reach target at step 150 → speedup=1.0× (TIE)
- Root cause: open-loop fitting misaligns with closed-loop context; CE gradient dominates any init advantage at step 50+
- KILL: CtQ_75 speedup 1.00× < 2× threshold. `grafting/results/grafting_006_tokenlevel_rank30_adapter_bootstrap.json`
- Codex architecture review: mean-shift (61% zero-step closure, zero params) is the real comparison baseline
- **Next: grafting_007** — mean-shift speedup test: does adding donor-minus-lesion per-layer bias provide CtQ_75 speedup ≥10×?

**grafting_007 KILL (2026-04-24): mean-shift speedup test — CtQ_75 speedup=1.0× (no acceleration).**
- Arm B: lesion + fixed (non-trainable) per-layer mean-shift bias = mean(donor) - mean(lesion), full unfreeze
- Arm B NLL at step 0: 9.703 (58.1% gap closed) — strong zero-step advantage (CtQ_50 = inf!)
- At step 25: Arm B=7.37 vs Arm A=7.46 (slight edge). Target_75=7.34 not yet crossed.
- At step 50+: Arm A BETTER than Arm B by 0.5 nats consistently — fixed bias becomes a liability
- Mechanism: backbone trains to compensate for fixed offset → misalignment grows → Arm B stuck
- KILL: CtQ_75 speedup 1.00× = 1.0×. `grafting/results/grafting_007_meanshift_speedup.json`
- Key finding: fixed prior does NOT persist through gradient updates; good step-0 effect evaporates
- Codex: "bottleneck is persistence, not step-0 effect. Test trainable carrier with anchor protection."
- **Next: grafting_008** — trainable mean-shift bias + anchor penalty + protected warmup

**grafting_008 KILL (2026-04-24): trainable mean-shift persistence test — CtQ_75 speedup=1.14× (no acceleration).**
- Arm A (zero-init trainable bias + warmup): CtQ_75=step 40
- Arm B (donor-init fixed bias, no warmup): CtQ_75=step 30 — BEST arm
- Arm C (donor-init trainable bias + anchor + warmup): CtQ_75=step 35 — WORSE than arm_b
- KILL: CtQ_75 speedup arm_c vs arm_a = 1.14× < 2× threshold. `grafting/results/grafting_008_trainable_meanshift_persistence.json`
- Critical finding: bias cosine sim stays ~0.9999 throughout (anchor lambda=1.0 over-constrains biases; they never adapt)
- Protected warmup backfired: blocks backbone for 10 steps while biases do nothing → arm_c wastes 10 steps arm_b doesn't
- Pattern confirmed: no hook/adapter/bias approach has worked. Geometry in output space ≠ geometry in weight space.
- **Next: grafting_009** — weight-space seed: directly initialize down_proj weights via outer product of donor output means × lesion inner activation means (rank-1 weight delta, no hooks needed)

**grafting_009 KILL (2026-04-24): rank-1 weight-space seed — CtQ_75 speedup=0.9× (arm_b SLOWER than arm_a).**
- Arm B step-0 NLL: 17.18 (3% gap closed — seed barely changes initial state)
- CtQ_50: arm_b=step 1, arm_a=step 5 → **5× speedup** (seed gives first gradient massive leverage: step-1 NLL 10.0 vs 17.4)
- CtQ_75: arm_b=step 50, arm_a=step 45 → **0.9× (arm_b LOSES)** — fast initial descent into shallower basin; arm_a overtakes from step ~25
- signal_frac range: [0.038, 0.288], 0 degenerate layers. Ridge stabilization worked. Seed mathematically valid.
- KILL: CtQ_75 speedup 0.9× < 2× threshold. `grafting/results/grafting_009_weightspace_seed.json`
- **Definitive conclusion: mean-based initialization family (output-space priors + weight-space seeds) is exhausted.** 7 experiments, consistent null at CtQ_75. CE gradient is too efficient — a lesioned model recovers fast from zero-init; no mean-level geometric prior provides enough of a head start to matter at ≥10× gate.
- **Next decision: pivot grafting to surgical capability transfer (specific circuit/skill) OR redirect to genome_109 atlas track.**

---

## §13 Mental Model Exploration Series (genome_110–113)

**Strategic pivot (2026-04-24):** grafting_001-009 established that global mean-level geometric priors cannot transfer capability. Codex postmortem verdict: we have been measuring trained-manifold occupancy (observability) but NOT capability-bearing control geometry (controllability). New series tests four model-native mental models derived from Codex analysis.

**Codex key insight:** "stop asking global descriptive geometry to do a circuit's job." Every new geometry object must be task-conditioned and intervention-linked.

**genome_110 NULL/KILL (2026-04-24): Syndrome Codes — FALSIFIED for Qwen3-0.6B**
- max_repair=0.0 (0 pairs above 50%, 0 above 20%). Corruption AMPLIFIED not repaired.
- repair_by_distance monotonically negative (dist1=-0.98, dist5=-3.94, dist27=-51.14).
- The model has NO syndrome code error-correction mechanism. Perturbations grow larger, not smaller, as they propagate forward. Mental model 4 falsified.
- `code/genome_110_syndrome_codes.py` -> `results/genome_110_syndrome_codes.json`

**genome_111 RUNNING (2026-04-24): Routing Constitutions** — cluster layer-14 residual stream into K=8 regimes; compute state-conditioned attention head coalitions; test if communication graph differs across regimes.
- 500 diverse contexts: 200 wikitext + 100 math + 100 code + 100 factual. Pass: >30% cluster pairs JS-div>0.30. Kill: mean JS-div<0.10.
- `code/genome_111_routing_constitutions.py` -> `results/genome_111_routing_constitutions.json`

**genome_112 PENDING: Scaffold-and-Flow Fields** — project residual streams onto top-30 shared directions (genome_099); cluster trajectories by task type; test if tasks follow distinct flow paths through scaffold space.

**genome_113 PENDING: Consistency Lattices** — measure single vs. pairwise layer-ablation damage; compute synergy = pair_damage - (A+B); test if superadditive pairs cluster spatially.

---

*End of WIKI. If anything here surprised you, fix the docs — not the wiki — and then patch the wiki pointer.*
