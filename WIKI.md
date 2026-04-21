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
| **Phase** | 0 — Scaffolding (scaffold complete 2026-04-20) |
| **Axiom status** | Stated; 0 atlas entries |
| **Bestiary coverage** | 0 / 9 classes measured |
| **Promoted primitives (🟢)** | 0 |
| **Active mysteries** | 7 |
| **Scars (🩹)** | 0 |
| **Open pre-registrations** | 0 |
| **Phase-3 claims** | 0 |
| **Next phase trigger** | Phase 1 begins when any primitive is tested on Phase-1 minimum viable bestiary |

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
| Claim-to-evidence map | `research/CLAIM_EVIDENCE_MAP.md` (create on first claim) |
| Repo-wide model registry | `../../models/MODEL_DIRECTORY.md` + `../../models/registry.py` |

Any markdown file not in this table either feeds one of these or should be deleted. (CLAUDE.md §3.4.)

---

## 3. Measurement primitives status

→ Full catalog: `research/MEASUREMENT_PRIMITIVES.md`.

**Legend.** 🟢 coordinate (agnosticism gate passed, ≥3 classes) · 🟡 candidate (1–2 classes) · ⚪ diagnostic (architecture-specific) · ⚫ untested.

| Primitive | Status | Classes tested | Last used | Notes |
|---|---|---|---|---|
| Intrinsic dimension (TwoNN, MLE) | ⚫ | — | — | Priority primitive #1 for Phase-1 agnosticism gate |
| Participation ratio | ⚫ | — | — | Pure covariance measure — high agnosticism prior |
| Fisher info matrix trace | ⚫ | — | — | Harder on JEPAs (no probability output) |
| Persistent homology | ⚫ | — | — | Topology is modality-agnostic by construction |
| Ricci curvature (Ollivier) | ⚫ | — | — | `llm-platonic-geometry/` reports positive Ricci in LLMs |
| Lyapunov spectrum | ⚫ | — | — | Needs layer-wise adaptation for feedforward |
| CKA (linear / RBF) | 🟡 | LLM, vision | — | Needs diffusion/JEPA/world-model test to promote |
| Procrustes / CCA / SVCCA | ⚫ | — | — | Prior: fails cross-arch (p=0.82); informative bound |
| RSA (cross-system RDM) | ⚫ | — | — | Canonical bridge to biology — priority primitive |
| SAE (feature decomposition) | ⚫ | — | — | See Mystery 7 (feature universality across models) |
| PCA / SVD spectral | 🟡 | LLM, vision | — | Spectral decay slope — candidate Level-2 constant |
| Activation ablation | ⚫ | — | — | Required for every Level-1 claim (CLAUDE.md §4.4) |
| Path / activation patching | 🟡 | transformer | — | Needs SSM + diffusion generalization (research task) |
| CAA (direction steering) | ⚪ | transformer-only | — | Demoted — fails on hybrids/SSMs; see Mystery 4 |
| Linear probes | ⚪ | — | — | Diagnostic; cannot imply usability (see Mystery 2) |
| MDL probes | ⚫ | — | — | Pure information-theoretic — works anywhere |
| Non-linear / MLP probes | ⚫ | — | — | Pair with linear probes for manifold hypothesis test |
| Task-conditional compression | ⚫ | — | — | Carries over the 99.7%→3D coherent-divergence finding |
| Successive-refinement D(R) curves | ⚫ | — | — | Direct CTI lineage — strong Level-1 candidate |
| Diffusion noise-step representations | ⚫ | — | — | No analogue in transformer "layer depth" |
| JEPA predictor/encoder alignment | ⚫ | — | — | Dual-network geometry |
| World-model latent rollout | ⚫ | — | — | Dynamics-aware, not static |

---

## 4. System bestiary status

→ Full bestiary: `research/SYSTEM_BESTIARY.md`. Model IDs always pulled from `../../models/registry.py`.

| Class | In atlas? | Phase-1 anchor | Status |
|---|---|---|---|
| 1 Autoregressive LLM | 0/N | `Qwen/Qwen3-0.6B` | Not yet loaded |
| 2 Reasoning | 0/N | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Not yet loaded |
| 3 SSM / linear-attention | 0/N | `state-spaces/mamba2-370m-hf` | Not yet loaded |
| 4 Hybrid | 0/N | `tiiuae/Falcon-H1-0.5B-Instruct` | Not yet loaded |
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
| *(none — scaffolding phase)* | — | — | — | — | — |

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
