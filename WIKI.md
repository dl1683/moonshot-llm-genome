# WIKI — Neural Genome

**The living project registry. Agents read this first, always. Agents update this after every experiment, finding, or doc change — in the same commit. Never stale.**

Entries are pointers (≤500 chars). If an entry needs more depth, link to the canonical doc. This file is an **index**, not a document.

Codex's Cross-System Auditor checks WIKI consistency at every PR gate. A commit that changes experiment results / docs / primitives without a corresponding WIKI patch is rejected.

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
| **Axiom status** | **FIRST CLEAN 🟡 COORDINATE (genome_007, 2026-04-21).** At n=2000 × 3 seeds × Bonferroni c=2.7729, **kNN-k10 clustering coefficient passes G1.3 at strict δ=0.10 on Qwen3-0.6B (transformer) + RWKV-4-169M (recurrent) + DINOv2-small (vision ViT)** — 3 classes × 2 modalities. Falcon-H1 hybrid narrow-fail (3.5% margin excess). Prereg `genome_knn_k10_portability_2026-04-21.md` → LOCKED. Bonus: PR_uncentered also passes δ=0.10 on all 4 systems (surprise 2nd candidate). ID primitives still fail (SE-dominated). |
| **Bestiary coverage** | **8 / ~13 classes measured** (genome_011 2026-04-21): classes 1 transformer / 2 reasoning / 3 recurrent / 4 hybrid / 6 vision ViT + **NEW: 7 BERT-MLM / 8 MiniLM-contrastive-text / 10 CLIP-contrastive-vision**. **kNN-k10 passes G1.3 at δ=0.10 on 7/8 classes** (Falcon narrow-fail at n=2000, tips at n=4000 per genome_010). Spans **5 distinct training objectives** (CLM + MLM + contrastive-text + self-supervised-ViT + contrastive-vision). MiniLM contrastive has the BEST max_stat (0.0175, 74% headroom). PR_uncentered's 5/5 was a DC-artifact — demoted ⚪. |
| **Promoted primitives (🟢¹/🟢²)** | 0 |
| **Gate-1 passed (🟡 coordinate)** | **1 CLEAN 🟡**: kNN-10 clustering coefficient, scope `(modality ∈ {text, vision}, stimulus_family ∈ {c4_clean.len256.v1, imagenet1k_val.v1}, pooling ∈ {seq_mean, cls_or_mean})` on Qwen3-0.6B + RWKV-4-169M + DINOv2-small. Prereg `genome_knn_k10_portability_2026-04-21.md` LOCKED at this commit. 1 secondary 🟡 candidate (PR_uncentered) pending its own focused prereg. |
| **Active mysteries** | 7 (unchanged; H11-H13 are hypotheses, not mysteries) |
| **Scars (🩹)** | 0 |
| **Active hypotheses (H-register)** | 14 — H1..H10 original + H11 Koopman + H12 stimulus-dominance + H13 quantization-stability + H14 subsample-stability (→ `atlas_tl_session.md §1c`). H15 retired to governance rule `atlas_tl_session.md §2.5.8` (modality-scope is policy, not falsifiable). |
| **Open pre-registrations** | **2 locked:** `genome_id_portability_2026-04-21.md` (Gate-1 joint ID+PR+kNN — superseded by focused kNN prereg for promotion) and **`genome_knn_k10_portability_2026-04-21.md` (Gate-1 kNN-10 on Qwen3+RWKV+DINOv2, LOCKED 2026-04-21)**. Validator exits 0 on both. |
| **Phase-3 claims** | 0 (Gate-1 ≠ Level-1; Gate-2 derivation draft exists at `research/derivations/knn_clustering_universality.md`) |
| **Active TL session** | `atlas_tl_session.md` — Phase 1-3 drafted; Codex Round 1 complete (8/10), Round 2 running (task `b3fwyis5j`) |
| **Gate semantics** | LOCKED in `atlas_tl_session.md §2.5` (two-gate spec + prereg template) |
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
| `genome_001_smoke` | ✅ passed 2026-04-21 | First end-to-end pipeline verification | Qwen3-0.6B (trained, FP16) | ID + PR + kNN-clustering | `atlas_tl_session.md §3.7` strawman via prereg |
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
| 2026-04-20 | **Inherit CTI's 3-tier universality framework.** | Prevents "universality collapsed because one constant varied" failure. `UNIVERSALITY_LEVELS.md`. |
| 2026-04-20 | **Architecture-agnosticism gate: ≥3 classes before a primitive becomes a coordinate.** | Keeps LLM-specific tools from masquerading as atlas coordinates. CLAUDE.md §4.3. |
| 2026-04-20 | **Biological validation mandatory for every Level-1 claim.** | Separates "fact about trained NNs" from "fact about learning systems." CLAUDE.md §4.5. |
| 2026-04-20 | **Anti-entropy is a Tier-1 rule, not a cleanup chore.** | Atlas dies under file-bloat within 6 months without this. CLAUDE.md §3. |
| 2026-04-20 | **Repo-wide model registry at `Projects/models/` is canonical; no local model lists.** | One source of truth across every moonshot and research project. |
| 2026-04-20 | **WIKI.md is read first, updated every commit touching state.** | Agents bootstrap fast; nothing goes stale. |
| 2026-04-20 | **9 system classes + Phase-1 minimum viable bestiary defined.** | Concrete, constrained first atlas iteration. `SYSTEM_BESTIARY.md` §Phase 1. |
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

First things an agent should pick up on startup. Keep this short (≤5 items). Reorder by priority as phases progress.

1. **ACTIVE: TL design session on "how do we map LLM internals."** Working doc `research/atlas_tl_session.md`. Phase 1 landscape + Phase 2 mental machine drafted. Codex Round 1 queued. Will promote to `research/BLUEPRINT.md` at convergence (then add to §2 canonical index).
2. **Phase-1 primitive agnosticism sprint.** Pick intrinsic dimension (TwoNN). Run on the full Phase-1 minimum viable bestiary. Promotion criteria per CLAUDE.md §4.3. Pre-register before loading any model. (Will be re-specced by Phase 6 blueprint from the active TL session — do not pre-register ahead of that.)
2. **Add `facebook/dinov2-small`, `facebook/ijepa-vit-huge-14-448`, and a small DiT variant to `Projects/models/` canonical registry.** Separate commit per add; paradigm/tier/VRAM metadata required.
3. **Biology access smoke test.** Reproduce CTI's Allen Neuropixels data-load pattern (`remfile + h5py + dandi`) on one session — confirms Phase-4 path is live *before* Phase 1 results appear.
4. **Develop an SSM-compatible activation-patching primitive.** Mysteries 2 and 4 depend on it. Design gate before implementation.
5. **Mystery 5 (coherent divergence) cross-class replication.** Candidate for first flashy atlas entry. Pre-register as a Level-1 probe.

---

*End of WIKI. If anything here surprised you, fix the docs — not the wiki — and then patch the wiki pointer.*
