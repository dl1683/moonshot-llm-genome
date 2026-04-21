# TL Session — How Do We Build a Map of LLM Internals?

**Status:** ACTIVE — Phase 0 complete, Phase 1 in progress.
**Session start:** 2026-04-20
**Operating mode:** `/tl` (Tesla + Leibniz + Chrome under Swarm Build governance)
**Canonical doc?** Working / session-scoped. Becomes `research/BLUEPRINT.md` at convergence (add to `CLAUDE.md` §3.4 index when promoted).

---

## Phase 0 — Bootstrap (complete)

### 0a. Loaded context (mission stack)

- `~/.claude/CLAUDE.md` — global constitution (Senior–Junior, autonomy, Codex is real CLI, anti-entropy, time-awareness)
- `../CLAUDE.md` (AI Moonshots umbrella) — **Manifesto:** AI = electricity = vaccines. Intelligence = Geometry (not Scale). Prove it on a single RTX 5090. Every moonshot scored for paradigm-shift potential.
- `./CLAUDE.md` (project) — axiom-first framing, atlas-as-instrument, architecture-agnosticism gate (≥3 system classes), causal-test mandate, biological-validation mandate, anti-entropy as Tier-1 rule.
- `./COMPUTE.md` — **BINDING envelope:** ≤22 GB VRAM, ≤56 GB RAM, ≤4 h per experiment with checkpointing, Windows+CUDA quirks, no cloud.
- `./WIKI.md` — Phase 0 scaffold. 0 atlas entries. 0 promoted primitives. 7 open mysteries. 0 running experiments. 9 system classes defined with Phase-1 anchors.
- `../../models/MODEL_DIRECTORY.md` — canonical model registry. Full ladder from Pythia-160M through Kimi-K2.
- Git state: fresh `main` branch in llm-genome, `4238fb7` initial scaffold commit.
- Memory file (AI Moonshots umbrella) — Nobel/Turing/Fields ≥9/10 bar, autonomous execution, Codex always reviews.

### 0b. Sacred outcomes (for the LLM Genome moonshot, as extracted)

These are the non-negotiable goals that every Codex confidence score in this session must rate against:

**S1. The atlas is a real instrument.** It *measures* representational geometry across system classes with mathematically well-defined coordinates — not a catalog of ad-hoc numbers. Each coordinate is a named, derivation-backed, reproducible function of the model's internals.

**S2. The atlas is architecture-agnostic.** Every primitive promoted to "coordinate" must pass the ≥3-system-class gate: LLM + at least two non-LLM classes (SSM, diffusion, vision encoder, JEPA, world model, controls). "Works on GPT-2" is not a coordinate.

**S3. The atlas produces derivation-first universal laws.** Before fitting constants, derive the functional form from first principles (information theory, statistical mechanics, geometry, EVT, etc.). Curve-fit without derivation = Phase-2 observation, not Phase-3 law.

**S4. The atlas supports causal claims.** Every Level-1 claim is validated by a do-intervention / ablation / orthogonal factorial experiment. No Level-1 claim graduates on observational data alone.

**S5. The atlas bridges to biology.** Level-1 claims are tested on biological neural recordings (Allen Neuropixels, human fMRI). This is the "genome" in Neural Genome — it's what separates this from interpretability.

**S6. The atlas stays low-entropy.** Deletions are a KPI. The repo stays tractable. Growth must increase clarity.

**S7. The atlas serves the manifesto.** Every coordinate, every law, every claim must speak to "Intelligence = Geometry." If it doesn't illuminate the efficiency arbitrage, it is a distraction.

### 0c. Convergence criteria for THIS session

This session is a **design session**, not an implementation session. Deliverable at convergence:

**D1. Phase 1 landscape artifact** — complete decomposition of "what does it mean to map LLM internals," research brief covering the current interpretability field with counter-evidence, hypothesis register with kill criteria.

**D2. Phase 2 mental-machine design** — concrete atlas construction pipeline (what goes in, what comes out, what lives between).

**D3. Phase 3 probe batch** — the 3–7 empirical probes that resolve the load-bearing uncertainties in D2.

**D4. Phase 6 blueprint** — the Phase-1 minimum-viable atlas plan. Specifies:
- Which primitives to implement first (ranked, with agnosticism-gate plan)
- Order and dependency graph
- Pre-registration templates (one per first primitive sprint)
- Triple backing per decision: theoretical derivation + cross-domain citation + probe-or-planned-probe
- Known risks + deferred uncertainties

**Exit criteria — all must hold:**

- Codex rates every sacred outcome (S1–S7) at ≥9/10 in a final round, with specific evidence per point.
- Every inherited paradigm audited — accepted with current justification or re-derived.
- No open assumption without theoretical basis, citation, or logged research question.
- Adversarial audit (fresh Codex session, hostile persona) run at convergence, all findings addressed.
- Blueprint is precise enough that another engineer could execute Phase 1 of the atlas without asking questions.

### 0d. Working cadence

- **Heartbeat:** every ~60 min of active session work, I fire the alignment checklist (CLAUDE.md §3.6). Not a `/loop` — a manual check so I don't interrupt Codex mid-round.
- **Codex cadence:** Round 1 fires after Phase 1+2 are externalized. Expect 5–15+ rounds; 3 is a floor, not a target.
- **Adversarial audit:** triggers at rounds 5, 10, 15… AND at claimed convergence.
- **Reviewer personas (Tier 1 — correctness + perf):** fire on any code Claude writes; no code in this session yet, so not triggered until probe batch execution.
- **Anti-entropy check:** at the end of every phase, sweep for stale scratch / duplicate notes.

### 0e. Known dead ends (from sibling repos / memory)

| Source | Dead end | Why it matters for this session |
|---|---|---|
| `llm-rosetta-stone` | CAA / linear steering fails cross-architecture (p=0.82 on Procrustes alignment) | Don't build atlas coordinates on steering vectors. Already encoded in `WIKI.md` Mystery 4 and §3 (CAA demoted to ⚪ diagnostic). |
| CTI hard review | "Curve-fitting universal laws without derivation = rejection" | Motivates S3. Every candidate law gets a derivation before any fitting code is written. |
| Fractal Embeddings | "Publication overclaims if stats aren't paired, HWV ranges misstated" | Motivates S4. Every claim in the atlas gets a pre-reg and paired-test plan from day one. |
| Prior `LLM Genome Project/` (deleted) | "Diverged from axiom; accumulated entropy" | Motivates S6. The new repo starts with anti-entropy as a Tier-1 rule, not a cleanup chore. |

### 0f. Inherited assumptions to audit (flag list for Codex)

These are baked into the project scaffolding and will be explicitly surfaced to Codex for challenge at every round:

- **A1.** "Representational geometry" is the right abstraction (vs. circuits, features, algorithms, attractors).
- **A2.** Primitives transfer if they pass the ≥3-class gate.
- **A3.** Derivation-first is required for Level-1; observation + correlation is acceptable for Phase 2.
- **A4.** Biological validation means RSA / Neuropixels / fMRI — not some other bridge (behavior, evolution, development).
- **A5.** The 9-class bestiary (autoregressive LLM, reasoning, SSM, hybrid, diffusion, vision encoder, JEPA, world model, controls) covers the space of "trained neural networks."
- **A6.** The RTX-5090-only constraint does not fundamentally limit what CAN be in the atlas — only how we get there.
- **A7.** The "universality levels" framework (Level 1 functional-form, Level 2 family constants, Level 3 task/data intercepts) correctly decomposes the claim space.
- **A8.** Intrinsic dimension, participation ratio, RSA, and CKA are *candidate* coordinates worth testing — not foregone conclusions.

---

## Phase 1 — Landscape

### 1a. Tesla decomposition — what does it mean to map LLM internals?

#### Components

**C1. The object being mapped.** "LLM internals" is six distinct objects in the literature, and the atlas must pick a primary object per coordinate or it becomes a Rorschach test:

1. **Activations** — residual stream, attention, MLP outputs at each layer/token. Transformer-native.
2. **Weights** — attention, MLP, embedding matrices. Architecture-specific in structure.
3. **Gradients** — during training or at inference. Dynamics over parameters.
4. **Dynamics** — trajectories through internal state space over sequences or training.
5. **Circuits** — subgraphs of computation implementing specific algorithms. Gate-compositional.
6. **Features** — directions in activation space corresponding to concepts. SAE-derived.

For architecture-agnosticism (S2), the object must exist in all 9 system classes:
- Activations exist everywhere but in different forms (transformer layers vs. SSM hidden state vs. diffusion noise-step vs. JEPA context/target).
- Weights exist but their structure is class-specific.
- **Dynamics — trajectory through internal state space under input — is the most universal object.**
  - Transformer: per-layer residual stream sequence
  - SSM: hidden-state sequence over time
  - Diffusion: state sequence across noise steps
  - JEPA: context → target embedding map
  - World model: latent rollout
  - Brain: population activity over stimulus

**Choice: the atlas's universal object is the trajectory through an internal state space.** State space is model-specific; trajectory *properties* (geometry, dynamics, information content) are the candidate universal coordinates.

**C2. The measurement primitive.** A function f: trajectory → {scalar, vector, or small structured object}. Taxonomy from the literature:

| Family | Examples | Status (from WIKI §3) |
|---|---|---|
| Geometric | intrinsic dimension (TwoNN, MLE), participation ratio, Ricci curvature, persistent homology | All ⚫ or 🟡, none promoted |
| Statistical | Fisher info trace, spectral decay, eigenvalue distribution | 🟡 PCA/SVD spectral |
| Information-theoretic | MI between layers, MDL, rate-distortion D(R) | ⚫ all |
| Dynamical | Lyapunov spectrum, Koopman eigenvalues, attractor dim | ⚫ all; Lyapunov ≈ 0 prior from `llm-platonic-geometry` |
| Causal | do-intervention size, ablation magnitude, activation patching | 🟡 transformer-only; required for Level-1 |
| Cross-system similarity | CKA, RSA (RDM), Procrustes, CCA, SVCCA | 🟡 CKA / PCA; RSA ⚫ but canonical biology bridge |

Each primitive must (a) be well-defined on trajectories, (b) computable within COMPUTE.md envelope, (c) pass ≥3-class agnosticism gate, (d) carry a first-principles derivation.

**C3. The comparison.** What the atlas compares:
- Same primitive across classes → cross-architecture universality
- Same primitive across scale → scaling laws
- Same primitive across training → time-evolution
- Same primitive under intervention → causal structure

CTI's 3-tier maps directly: Level 1 = functional form holds across all comparisons; Level 2 = constants are family-specific; Level 3 = intercepts are task-specific.

**C4. The validation.** How we know the atlas isn't a delusion:
- Cross-class replication (≥3 classes)
- Causal tests (ablation → behavior change)
- Biological bridging (RSA vs. Allen Neuropixels / fMRI)
- Adversarial robustness (survives OOD inputs, weight perturbation)

**C5. The payload.** What the atlas produces:
- Reference dataset: for each (system, layer, primitive, task), a value
- Laws: functional forms + empirical validations
- Predictions: coordinate values for unseen systems
- Anomalies: scars / mysteries

#### Mathematical structure of the problem

Formally the atlas is **functional analysis on a moduli space of learned systems**:

- **M** = moduli space of trained neural networks. Each point m ∈ M is a trained system.
- **S(m)** = space of internal states of m.
- **T(m, x)** ⊂ S(m) = trajectory induced by input x.
- **f: T(m, x) → ℝ^k** = a measurement primitive.
- The **atlas** is a map M × Inputs → ℝ^k via f.

Level-1 universality ⟺ f factors as f(m, x) = g(θ(m), x) where θ extracts a universal parameter (architecture family + scale) and g is class-agnostic. This formalization makes explicit:

- The atlas's value depends on what *structure* M has. Manifold? Stratified space? Poset? Without knowing, the atlas is a point cloud, not a chart.
- Level-1 = functional form on M up to a diffeomorphism.
- Failure mode of prior interpretability: without specifying M, "universality" is ill-defined.

#### Cross-disciplinary framings (for Phase 1b to ground)

- **Information theory:** every primitive is a statistic. Level-1 universality ≈ "same sufficient statistic under the true data-generating process." Platonic Hypothesis is a consequence: LLMs and brains trained on the same input converge on the same sufficient statistics. Rate-distortion D(R) is a natural universal coordinate — CTI's lineage.
- **Dynamical systems:** forward pass = dynamical system. Universality classes in DS (strange attractors, phase transitions) apply directly. Grokking = dimensional phase transition with self-organized criticality (Rubin et al. 2026) is a proven DS analogy.
- **Statistical mechanics:** RG flow ≈ coarse-graining across layers. If layers ARE RG transformations, fixed points of flow are universal attractors of representation. Free energy / partition functions give full thermodynamic info of output distribution.
- **Topological data analysis:** persistent homology of activation point clouds → Betti numbers stable under perturbation. Candidate modality-agnostic invariant.
- **Geometry:** Riemannian curvature (Ricci, sectional) of embedding spaces. Decoder-only LLMs are empirically hyperbolic (HELM). Brains are plausibly also non-Euclidean.

#### Inherited axioms — surfaced for Codex challenge (Round 1 verdicts applied)

Verdicts come from Codex Round 1 (`.codex/outputs/round1.md` §1 + §2). MODIFY = keep with constraint; KEEP = unchanged; REJECT = drop; SPLIT = decompose into sub-claims with separate verdicts.

| # | Axiom | Challenge (what if wrong?) | **Round 1 verdict** |
|---|---|---|---|
| A1 | "Representational geometry" is the right abstraction | Dynamics might be primary; geometry is a static snapshot. Trajectories, not snapshots, are the primitive (supported by Koopman + grokking literature). | **MODIFY** → "geometry-of-trajectory / geometry-of-update." `llm-platonic-geometry` Lyapunov≈0 and `Latent-Space-Reasoning` basin effects pressure-test toward dynamics-first. |
| A2 | Primitives transfer if they pass ≥3-class gate | 3 is arbitrary. Why not 5? | **MODIFY** → keep ≥3 as *portability gate*, not universality. Level-1 requires ≥5 classes (per `UNIVERSALITY_LEVELS.md`). Every use of "3-class" must add: "portability; Level-1 requires ≥5." |
| A3 | Derivation-first for Level-1 | RG discovered phase transitions empirically before Wilson's theory. | **KEEP** as graduation rule, not exploration blocker. Empirical-first is fine as Phase-2 observation; derivation required only for Phase-3 Level-1 promotion. |
| A4 | Biological validation = RSA / Neuropixels / fMRI | Neuropixels is mouse V1, fMRI is coarse, RSA is stimulus-dependent. | **MODIFY** → RSA is *first* bridge (cheap) but must be paired to versioned stimulus bank; first serious bio validation should target a **vision-class claim** (Allen V1-aligned), not language. |
| A5 | 9-class bestiary covers trained NNs | Missing spiking, RNN, GNN, ensembles, non-backprop. | **MODIFY** → relabel 9-class as "Phase-1/2 bestiary," not "trained-NN closure." Add "Phase-N expansion candidates" register. |
| A6 | RTX 5090 doesn't limit WHAT is in the atlas, only HOW fast | Absolutely false. | **REJECT** — compute is **epistemic**. Per COMPUTE.md §7 the atlas vocabulary IS bounded by the envelope: small-to-mid dense (≤10B), MoE with ≤8B active. Sample size, stimulus-bank size, and primitive complexity are all constrained by ≤4h/experiment. Primitives that need huge activation dumps are non-viable as atlas coordinates. |
| A7 | 3-tier framework is sufficient | Possibly needs Level-0 and Level-4. | **MODIFY** → codify Level-0 explicitly (diagnostic/scope-limited, already in primitives doc) plus existing "Phase-2 null." No Level-4 — premature complexity. |
| A8 | ID / PR / CKA / RSA are candidate coordinates | CKA scale-confounded (Aristotelian-view, Feb 2026). | **SPLIT.** ID + PR = strong candidates (distribution-conditional but semantically stable). RSA = biology bridge, stimulus-dependent. **CKA demoted to diagnostic only** — do not treat as coordinate. Promote "local neighborhood agreement" (kNN-overlap) as the thing H4 actually wants. |
| **A9** | **(NEW; Codex flag)** Depth normalization ℓ/L is meaningful cross-class. | If wrong, H1/H2 collapse — comparing curves by ℓ/L assumes an equivalence class that may not exist. | **LOAD-BEARING.** Must be tested empirically (new probe P1.6). Dynamical-systems lens: the proper universal parameter is *arc-length in state space* or *spectral measure*, not raw index. |
| **A10** | **(NEW; Codex flag)** Stimulus-bank comparability isn't the dominant confound. | If wrong, "universality" is ill-posed: we're comparing different problems. | **LOAD-BEARING.** Must be tested: variance decomposition per primitive into `{system, depth/time, stimulus}`. Universality requires stimulus variance NOT to dominate. Pipeline must treat stimulus as the *conditioning variable of every coordinate*, not a component. |

---

### 1b. Leibniz research brief (compiled from web search, 2024-2026)

Facts only. No mechanisms yet — that's Codex's job in Phase 4.

#### B1. Mechanistic interpretability — current state of the art

- **Anthropic circuit tracing (2025-2026).** Attribution graphs via cross-layer transcoders replace MLPs with interpretable sparse features. Open-sourced. Supports Gemma-2-2B, Llama-3.1-1B, Qwen3-4B. MIT Tech Review named mech-interp a 2026 Breakthrough Technology. [[Anthropic]](https://www.anthropic.com/research/open-source-circuit-tracing) [[Circuits thread]](https://transformer-circuits.pub/) [[MIT Tech Review]](https://www.technologyreview.com/2026/01/12/1130003/mechanistic-interpretability-ai-research-models-2026-breakthrough-technologies/)
- **Universal SAEs (USAE, Thasarathan et al., Feb 2025, revised Mar 2026).** Single sparse autoencoder ingests activations from multiple models, decodes to any other. Universal concept space across architectures. Strong correlation between concept universality and importance. [[paper]](https://arxiv.org/abs/2502.03714)
- **SAE cross-LLM feature universality (Lan et al. 2024).** SAE feature spaces significantly similar across multiple LLMs at multiple layers. [[paper]](https://arxiv.org/html/2410.06981v3)

**Counter-evidence to SAE-based approaches:**
- **"Dark Matter" (Engels et al., Oct 2024, ICLR 2025).** >90% of SAE reconstruction error norm is linearly predictable from the input. Error decomposes into: unlearned linear features, unlearned dense features, nonlinear SAE errors. Scaling SAE width doesn't uniformly help. [[paper]](https://arxiv.org/abs/2410.14670)
- **Specialized SAEs for rare concepts.** Wide SAEs miss rare/specific concepts; infrequent activations are essentially invisible. [[paper]](https://arxiv.org/html/2411.00743)
- **Nanda pessimism update (Sept 2025):** "I don't see a path to deeply and reliably understanding what AIs are thinking." Theoretical barriers: "feature" lacks rigorous definition; many interpretability queries are computationally intractable; fundamental math limits constrain linear interventions in chaotic deep networks.
- **Temporal/agentic interpretability frontier.** Static mechanistic interpretability insufficient for agents; long-horizon trajectories need new methods.

#### B2. Representation similarity and cross-architecture alignment

- **Platonic Representation Hypothesis (Huh et al. 2024).** Representations converge across modalities and architectures as scale increases. Both vision and language measure distance between datapoints increasingly similarly. [[paper]](https://arxiv.org/abs/2405.07987)
- **Critical revision — "Aristotelian View" (Feb 2026).** Existing similarity metrics are scale-confounded. Under calibration, global spectral convergence largely disappears. **Only local neighborhood similarity retains significant agreement across modalities.** [[paper]](https://arxiv.org/abs/2602.14486)
- **Entropic-force theory for Platonic convergence.** SGD with implicit entropic regularization enforces "perfect Platonic" alignment in deep linear models. [[ref]](https://phillipi.github.io/prh/)
- **CKA limitations (Kornblith et al., plus 2024-2025 follow-ups).** CKA is scale-invariant but PC-dominated — misalignment of leading principal components causes rapid score drop; low-variance PCs contribute weakly. Procrustes + Bures more sensitive. Subspace CKA reveals leakage global CKA misses. [[ICLR 2025]](https://proceedings.iclr.cc/paper_files/paper/2025/file/03d113a060c0ac93a5859517a0f07271-Paper-Conference.pdf)
- **Multi-way representation alignment (2026).** Extensions to simultaneous alignment across >2 systems. [[paper]](https://arxiv.org/html/2602.06205)

#### B3. Geometric primitives

- **Intrinsic dimension via TwoNN (Facco et al. 2017).** Ratio of 1st to 2nd nearest neighbor distances is Pareto-distributed; MLE gives ID estimate. Applied broadly to LLMs. Word embeddings: extrinsic 300 → ID 10-30. Fine-tuning systematically lowers local ID only on fine-tuning dataset. [[background]](https://www.emergentmind.com/topics/intrinsic-dimension-id-of-llm-representations)
- **Local ID of contextual embeddings (Ruppik et al. 2025).** TwoNN localized. Fine-tuned LLMs have markedly lower local ID than bases. Decision-making geometry in LLMs has clear low-ID structure. [[paper]](https://arxiv.org/html/2506.01034v1)
- **Ollivier-Ricci curvature on LLM embeddings.** Decoder-only LLM token embeddings show wide range of negative curvatures — spaces are more hyperbolic than Euclidean. [[HELM: Hyperbolic LLMs]](https://arxiv.org/html/2505.24722)
- **Ricci curvature for representational alignment (Fumero et al. 2025).** Ollivier-Ricci curvature + Ricci flow analyze local structure agnostic to source of representational space. Enables direct comparison between human behavior and model embeddings. [[paper]](https://arxiv.org/html/2501.00919)
- **Persistent homology for LLMs (2024-2025).** TDA tools (persistent homology, Mapper) applied to attention patterns, latent representations, training dynamics. [[review]](https://link.springer.com/article/10.1007/s10462-025-11462-w) [[LLMs]](https://www.mdpi.com/2227-7390/14/2/378) [[arXiv:2410.11042]](https://arxiv.org/pdf/2410.11042)
- **Persistent homology of brain dynamics.** Topological features covary with behavioral traits; useful at individual level. [[paper]](https://pmc.ncbi.nlm.nih.gov/articles/PMC12163041/)

#### B4. Information-theoretic and statistical-physics primitives

- **Generalized Information Bottleneck (GIB, Sept 2025, updated Jan 2026).** Reformulates IB via synergy. Demonstrates clear compression phases. Overcomes infinite-complexity-term issue of standard IB. [[paper]](https://arxiv.org/abs/2509.26327)
- **Two-phase learning dynamics.** Rapid curve-fitting phase → slower compression / coarse-graining. Three timescales (grokking-perfect-training, double-descent second descent, IB compression) empirically align. [[paper]](https://arxiv.org/html/2504.12700)
- **Information-theoretic grokking progress measures.** Grokking is an emergent phase transition attributable to synergistic neuron interactions. [[paper]](https://arxiv.org/html/2408.08944)
- **Grokking as dimensional phase transition (Rubin et al. 2026).** Effective dimensionality crosses from sub-diffusive to super-diffusive at generalization onset; self-organized criticality (SOC). [[paper]](https://arxiv.org/abs/2604.04655)
- **Provable scaling laws of feature emergence (Sept 2025).** Mathematical framework for Lazy / Independent / Interactive feature learning stages → provable scaling laws. [[paper]](https://arxiv.org/abs/2509.21519)
- **Spectral entropy collapse in grokking.** Empirical signature of delayed generalization. [[paper]](https://arxiv.org/html/2604.13123)
- **Renormalization group and neural networks.** Dynamic-neuron RG approach (Phys. Rev. Research, June 2025) reveals translational symmetry in DNNs, simplifying RG transformations. Earlier: Koch-Janusz & Ringel (Nature Physics 2018) show NNs can implement RG. [[2025]](https://arxiv.org/abs/2410.00396) [[2018]](https://www.nature.com/articles/s41567-018-0081-4)
- **Diffusion-guided tensor-network RG for neural systems.** Iterative coarse-graining via tensor networks. [[paper]](https://arxiv.org/html/2510.06361)

**Counter-evidence — IB theory:**
- Some research shows IB's core claims don't hold generally; networks that do NOT compress still generalize, and vice versa. Compression → generalization causal link contested.

#### B5. SSM / Mamba interpretability

- **Mamba-3 (ICLR 2026).** Three methodological improvements: more expressive SSM-discretization recurrence, complex-valued state update for richer state tracking, MIMO formulation. [[paper]](https://openreview.net/pdf?id=HwCvaJOiCj)
- **Mamba-2 activation patching + causal mediation.** Factual info concentrated at subject's final token in middle layers, prompt end in later layers — **paralleling transformer behavior.** First strong evidence of cross-class universality in causal structure. [[ref]](https://www.emergentmind.com/topics/mamba-based-selective-state-space-model)
- **LATIM (Latent Token-to-Token Interaction in Mamba).** Layerwise decomposition interpreting Mamba-2 recurrence as implicit attention matrix. Enables token attribution despite recurrent structure.
- **Layer-wise relevance propagation adapted for Mamba.** Stable, faithful attribution.

#### B6. Diffusion interpretability

- **Mechanistic SAE analysis of diffusion (2025).** Human-interpretable concepts in diffusion activations. Early diffusion steps: composition is predictable from spatial concept distribution BEFORE first reverse step completes. [[paper]](https://arxiv.org/html/2504.15473v1)
- **DIFFLENS (CVPR 2025).** Directly interacts with diffusion internals. Identifies and modifies bias-specific features. [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Shi_Dissecting_and_Mitigating_Diffusion_Bias_via_Mechanistic_Interpretability_CVPR_2025_paper.pdf)
- **Phase structure:** early steps control composition; middle steps set stylistic; final steps minor texture. "Chaotic early stage" is a blind spot.

#### B7. JEPA — geometric theory

- **Collapse theorems.** Sufficient target diversity + nontrivial context-target mapping ensure non-collapsed global minima of JEPA objective.
- **Koopman-invariant recovery.** JEPA loss structure recovers Koopman invariants on time-series / dynamical data. Implication: JEPA latent space IS a dynamical-systems invariant representation. [[background]](https://www.emergentmind.com/topics/joint-embedding-predictive-architecture-jepa-1c718c4a-0bb8-4b78-a67b-0990aa485ceb)
- **V-JEPA 2.1 (2026).** Dense spatio-temporal features; open challenge: simultaneous dense structure + dynamics + global understanding.
- **LeJEPA theory upgrade.** Theoretical foundation for JEPA objectives.

#### B8. Brain-LLM alignment

- **Tuckute et al. / recent fMRI-RSA work.** Alignment with human eye movement and fMRI significantly improves as model scales from 774M → 65B. [[Nature Comput Sci]](https://www.nature.com/articles/s43588-025-00863-0)
- **Brain-informed fine-tuning.** Voxel-level gains scale with both model size (GPT-2-124M → LLaMA-2-7B) and training duration (1-40h). [[ICLR]](https://openreview.net/forum?id=07S1CPoQYP)
- **Platonic Hypothesis vs. Intermediate-Layer Advantage (NeurIPS 2025 workshop).** Middle-depth layers often encode richer, more generalizable features. [[paper]](https://arxiv.org/html/2510.17833v1)
- **fMRI-LM (Nov 2025).** Foundation model for language-aligned fMRI understanding. [[paper]](https://arxiv.org/html/2511.21760)
- **Brain decoding survey (Nov 2025, bioRxiv).** Foundation models as priors, targets, generators for non-invasive brain decoding. [[survey]](https://www.biorxiv.org/content/10.64898/2025.11.30.691403v2.full)

**Gap:** brain-alignment literature is LLM-to-brain. SSM-to-brain, diffusion-to-brain, JEPA-to-brain alignments are largely unpublished. Direct genome opportunity.

#### B8b. Koopman spectral analysis — candidate universal trajectory invariant (2025-2026)

Added post-draft during heartbeat T+0.5. Koopman operator theory is emerging across ALL three major non-biological system classes the atlas cares about:

- **Learnable Koopman-enhanced transformers (Mar 2026).** Full spectral analysis — eigenvalue trajectories, stability envelopes, learned spectral distributions — compared against SSM baselines AND transformer architectures (Autoformer, Informer, PatchTST). [[paper]](https://arxiv.org/html/2602.02592)
- **Residual Koopman Spectral Profiling for transformer training instability (Feb 2026).** "Koopman Spectral Shaping" reshapes spectra during training; applied to transformers AND Mamba SSMs. [[paper]](https://arxiv.org/html/2602.22988)
- **Hierarchical Koopman Diffusion (Oct 2025).** Lifts nonlinear diffusion dynamics into a latent space where evolution is governed by globally linear Koopman operators; closed-form trajectory solutions; access to intermediate states. [[paper]](https://arxiv.org/abs/2510.12220)
- **Data-driven spectral analysis via pseudo-resolvent Koopman (Feb 2026).** Addresses spectral-pollution in finite-dimensional approximations. [[paper]](https://arxiv.org/abs/2512.24953)
- **Generative modeling via Koopman spectral Wasserstein gradient descent (Feb 2026).** Training-free generative framework via Koopman + Wasserstein. [[paper]](https://arxiv.org/html/2512.18837)
- **JEPA recovers Koopman invariants** (from B7 — Koopman appears there naturally too).

**Implication for the atlas:** Koopman spectrum is the strongest candidate I've found for a **cross-class universal trajectory invariant** (U8 research uncertainty resolved in favor of Koopman). It applies to any dynamical system by construction, is architecture-agnostic, has formal spectral theory (eigenvalues carry mode information, Koopman modes are the universal "features" of the trajectory), and has been computed on transformers, SSMs, and diffusion in the literature within the last 12 months.

**This should be flagged to Codex Round 1** (arriving too late for that round, but critical for Round 2): consider promoting **Koopman spectral analysis** to a top-tier candidate primitive alongside intrinsic dimension. Unlike TwoNN ID, Koopman has (a) a first-principles derivation (linear operator on observables, Koopman 1931), (b) already-demonstrated cross-class applicability (transformer + Mamba + diffusion all in 2025-2026), (c) natural causal interpretability (eigenmodes = dynamical modes; ablating a mode has a defined theoretical effect).

**Addendum to H-register: H11.** The Koopman spectrum of the layer/time/step-indexed trajectory is a Level-1 universal coordinate. Same functional form of spectral distribution across transformer, SSM, diffusion; class-specific scales. Conf: medium-high. Kill: if Koopman spectra of three matched systems on matched tasks show non-superimposable distributions even under scale normalization, H11 fails.

#### B8c. Null result — Ricci curvature on SSM / diffusion (2025-2026)

Added during heartbeat T+0.5. No published application of Ollivier-Ricci (or any discrete Ricci) to Mamba/SSM hidden states or diffusion U-Net latents was found in 2025-2026 literature. Only LLM embeddings (HELM and Fumero et al.) have been Ricci-measured.

**Implication:** if the atlas applies Ricci to SSM/diffusion, it is **new science**. H3 can be tested cleanly — no prior Ricci measurement in those classes means our measurement IS the measurement. Cheap Phase-3 probe candidate (adds ~2 h compute per class on 10k embeddings at most).

**Addendum to H-register: H3a.** We can strengthen H3's test: measure Ricci on Qwen3-0.6B + Mamba2-370M + a diffusion latent decoder (e.g., FLUX latent) on matched stimuli. If all three show negative mean Ricci → H3 promotes to Level-1 hyperbolic-embedding claim. If Mamba is positive or flat, H3 stays Level-2 (transformer-only).

#### B9. Critical limitations of current interpretability

- **Theoretical barriers** (2026 ICLR paper, cited in 2026 mech-interp status report):
  - "Feature" lacks rigorous definition
  - Computational complexity results prove many queries intractable
  - Fundamental math limits on linear-intervention methods in chaotic deep networks
  - Models may learn harder-to-study representations if trained against interp techniques
- **Reductionist tradeoff.** Researchers want succinct explanations of terabyte-scale models. Information-theoretic limit: detailed-AND-succinct may be impossible.
- **Temporal gap.** Mechanisms governing agent trajectories live across many forward passes; static interpretability insufficient.
- **Neel Nanda 2025 update:** pessimistic about high-risk-high-reward mech-interp; more optimistic about medium-risk-medium-reward approaches.

---

### 1c. Hypothesis register with kill criteria

Format: **H#. Claim. Level. Theoretical confidence. Kill criteria.**

**H1. Intrinsic dimension is a universal coordinate.** For any system class C, ID(layer, task) factors as d = d₀ + α·g(layer_depth_normalized, task_complexity) with C-specific α and universal g. Level-1 candidate. Conf: medium (TwoNN scales consistently within transformers; cross-class missing). **Kill:** ID(Qwen3-0.6B mid, Wikipedia) and ID(Mamba2-370M mid, Wikipedia) differ >3σ under matched tokenization and depth normalization.

**H2. Layer-wise compression-expansion is a universal primitive.** Some geometric invariant (Fisher info trace, spectral entropy, ID) traces a universal monotonic curve through normalized depth. Level-1 candidate. Conf: medium (IB two-phase dynamics known within transformers). **Kill:** phase-transition location in depth varies >20% across classes under matched tasks.

**H3. Negative Ricci curvature of embeddings is family-specific (Level-2), universal across decoder-only transformers.** Conf: high within decoders; low across classes. **Kill:** any decoder-only LLM shows positive mean Ricci ⇒ revise. Mamba/Mamba-2 shows same negative pattern ⇒ H3 promotes to Level-1 ("hyperbolic = decoder-only causal architectures").

**H4. Representational convergence is LOCAL, not global.** Level-1 candidate (after scale correction). Conf: high (Aristotelian-view paper). **Kill:** neighborhood-preservation rates across architectures do NOT exceed scale-matched controls.

**H5. Phase transitions exist in all system classes, signed by dimensional collapse.** Level-1 candidate. Conf: medium-high for transformers. **Kill:** training any class from scratch on a grokking-prone task (modular arithmetic, sparse parity) produces NO measurable dimensional collapse.

**H6. Feature universality is scale-dependent with a threshold.** Conf: medium (USAE). **Kill:** no scale threshold exists (feature overlap is flat with scale).

**H7. Activation patching generalizes principled-ly to SSM + diffusion.** Conf: medium (Mamba-2 patching shows factual recall pattern parallel to transformers). **Kill:** SSM or diffusion patching fails to produce interpretable causal effects on a well-defined task.

**H8. Brains and LLMs share Level-2 family constants, not Level-1 functional forms.** Conf: medium (Platonic + brain-LLM alignment). **Kill:** no law fits both brain and LLM with a single re-parametrization; OR, the fit is better than Level-1 (then promote).

**H9. RG coarse-graining is a valid lens on layer-wise transformation.** Conf: medium (Koch-Janusz 2018, dynamic-neuron RG 2025). **Kill:** layer-wise mutual-information structure does not exhibit RG fixed-point behavior at any scale.

**H10. The atlas cannot be built from ANY single primitive; it requires a multi-chart cover.** (A meta-hypothesis.) Conf: high (no single current primitive passes all 4 validation tests). **Kill:** some single primitive passes all validation tests (agnostic, causal, biological-bridge, adversarial-robust). Would simplify the whole design.

**H11. The Koopman spectrum of a layer/time/step-indexed trajectory is a Level-1 universal coordinate.** Same functional-form spectral distribution across transformer, SSM, diffusion; class-specific scales. Conf: **medium** (reset from medium-high after Codex Round 1's anti-overconfidence audit — literature proximity is not evidence in OUR envelope; needs in-repo pilot). **Kill:** Koopman spectra of three matched systems on matched tasks show non-superimposable distributions even under scale normalization.

**H12. Stimulus dominance (NEW; Codex Round 1 gap).** For any primitive f, the variance of f decomposes into `Var(f) = σ²_system + σ²_depth + σ²_stimulus + interactions`. Any universality claim requires `σ²_stimulus < min(σ²_system, σ²_depth) / 2`. Conf: N/A (this is an enabling meta-hypothesis, not a universality claim itself). **Kill:** stimulus variance dominates — in that case "universality" is ill-posed and we need conditional universality (Alt C below) from the ground up.

**H13. Quantization stability (NEW; Codex Round 1 gap).** Every candidate coordinate must be stable under the project quantization ladder (FP16 → Q8 → Q4_K_M) within a prereg-specified tolerance. A coordinate whose measurement changes materially under quantization is hardware-dependent, not universal. Conf: N/A. **Kill:** primitive values drift by > 2× bootstrap scatter across FP16 vs Q6 on the same model, rendering the primitive hardware-dependent.

### 1d. Probe candidates (from empirical uncertainties)

These get triaged in Phase 2 — some resolved by thought, some by research, some go into the Phase 3 probe batch.

- **U1 (PROBE candidate).** Does TwoNN on Qwen3-0.6B + Mamba2-370M + DINOv2-small on a matched input set produce equivalent ID scaling curves? **This is the Phase-1 primitive agnosticism sprint.**
- **U2 (PROBE candidate).** Does a "compression phase" aligned across layer depth exist in all 3 classes?
- **U3 (PROBE candidate).** Does ablating a middle-layer subspace at fixed ID produce equivalent behavioral degradation across classes?
- **U4 (PROBE candidate).** Do persistent-homology Betti numbers of hidden-state trajectories agree across architectures on a fixed task?
- **U5 (RESEARCH first).** Has any paper computed Ollivier-Ricci on SSM hidden states? On diffusion latents? If yes, what was found?
- **U6 (OPERATIONAL — triage deferred to Phase 3).** Can CTI's Allen pipeline be re-wired for ID computation on cortical population activity?
- **U7 (THOUGHT resolvable).** Does the moduli-space framing (§1a) force any structural requirement on the atlas that the current primitive list doesn't satisfy? [Answered: yes — forces a choice between Level-0 diagnostics and Level-1 coordinates, which was missing from the inherited framework.]
- **U8 (RESEARCH).** Is there a known mathematical object that is the "universal trajectory invariant" across continuous (SSM), step-indexed (diffusion), and layer-indexed (transformer) dynamics? (Candidates: Koopman spectrum, Lyapunov spectrum, spectral entropy.)

---

## Phase 2 — Mental machine (Round 1-revised)

The atlas construction pipeline. Round 1 flagged three kill shots; this section now addresses them:
- **Kill shot 1** (agnosticism gate broken): split into two gates — see new §2.5
- **Kill shot 2** (trajectory tensor transformer-shaped): replaced with point-cloud + operator view
- **Kill shot 3** (stimulus as component): replaced with stimulus as conditioning variable of every coordinate + mandatory variance-decomposition probe

### 2a. Component inventory (revised)

```
[model registry]           ←  ../../models/registry.py (canonical)
        ↓
[system loader]            ←  class-specific adapter: load model at the quantization
        |                     chosen for its parameter bucket (COMPUTE.md ladder).
        |                     Must declare its quantization level for H13 stability test.
        ↓
[stimulus bank (VERSIONED; conditioning variable)]
        |                  ←  every coordinate is f(m, x) where x ~ D_stimulus.
        |                     D_stimulus is versioned, its invariance class declared,
        |                     its resamplings pre-defined. No coordinate is evaluated
        |                     without naming its conditioning distribution.
        ↓
[trajectory extractor → point-cloud sequence]
        |                  ←  class-specific hook, but OUTPUT IS UNIFORM:
        |                     sequence {(k, X_k, meta_k)} where k indexes depth/time/step,
        |                     X_k ∈ R^{n_k × d_k} is a *point cloud* (each row is one state
        |                     vector sampled from the class's internal state at index k),
        |                     meta_k describes what a point is (token / patch / neuron /
        |                     spatial site) and how sampled (pooling, subsample).
        |                     Point clouds — NOT transformer-tensors. Works for diffusion
        |                     U-Nets, world-model rollouts, and biology recordings.
        |                     Codex Round 1 Alt A.
        ↓
[operator view (OPTIONAL per primitive)]
        |                  ←  estimate a linear operator K_k mapping observables across
        |                     k → k+1 (Koopman-lite). Plug-in point for H11 spectral
        |                     primitives without redesigning the pipeline.
        ↓
[primitive evaluator]      ←  f: {(k, X_k, meta_k)} or K-sequence → Measurement.
        |                     Plug-in architecture. Each primitive declares:
        |                       - invariance group G (what f(m,x) should be invariant to)
        |                       - supported class_ids
        |                       - estimator variants for the robustness check (H13)
        |                       - biology instantiation spec (Alt D — mandatory at declare time)
        ↓
[atlas row builder]        ←  assembles (system_id, k_normalized, primitive_id,
        |                     stimulus_version, pooling_choice, value, uncertainty,
        |                     estimator_variant_result) → one ledger entry
        ↓
[two-gate semantics (see §2.5)]
        |                  ←  Gate 1 PORTABILITY (≥3 classes, stability + comparability)
        |                     Gate 2 UNIVERSALITY (≥5 classes, derivation, causal, biology)
        ↓
[causal tester]            ←  class-agnostic causal estimand: "sensitivity of output to
        |                     removal of a coordinate-defined subspace at matched norm."
        |                     Ablation is the minimum cross-class primitive (patching
        |                     is transformer-native and stays as a Level-0 diagnostic
        |                     until a class-agnostic extension is derived — see H7).
        ↓
[biological bridge (CO-EQUAL AXIS — Alt D)]
        |                  ←  each coordinate ships with a declared biology instantiation:
        |                     "given neural population activity matrix N_neurons × T_timepoints
        |                     under stimulus s, the primitive is computed as …". May be
        |                     deferred in execution but MUST be specified at coordinate
        |                     declare-time.
        ↓
[law fitter (STUB ONLY UNTIL DERIVATION EXISTS)]
                           ←  per Codex Round 1: do not build until a first-principles
                              functional form is in hand. Prevents optimizing for fit-ability.
```

### 2b. Interface contracts (revised)

- **`PointCloudTrajectory`** = `{system_id: str, class_id: {1..9}, quantization: str, index_kind: "layer"|"time"|"step", sequence: List[{k: float, k_normalized: float, X: Tensor[n_k, d_k], point_kind: "token"|"patch"|"neuron"|"site", pooling: str, stimulus_version: str, seed: int}]}`.
  - Point clouds uniformly shaped across classes. No fake "token axis" for classes that don't have one.
  - Each entry carries its conditioning variables inline (stimulus version, pooling, seed).
- **`OperatorView`** (optional, per primitive's request) = `List[{k: float, K: Tensor[d, d] or LinearOp, residual: float}]` — Koopman-lite operators between consecutive point clouds.
- **`Primitive`** module contract:
  - `invariance_group: str` — formal declaration (e.g., "orthogonal rotations + token permutations + stimulus resampling within D")
  - `supported_classes: Set[int]`
  - `estimator_variants: List[str]` — each must produce a comparable value (for H13 stability)
  - `biology_instantiation: str` — how to compute on neural-population data (co-equal axis)
  - `evaluate(trajectory) → Measurement` — raises if class not supported
  - `Measurement = {value, uncertainty, estimator_variants_results, invariance_checks_passed}`
- **`AgnosticismGate`** — NO LONGER "promote on ≥3 congruent measurements." See §2.5 for the revised two-gate spec.

### 2c. Data flow — revised worked example

Probe P1.1 (ID agnosticism), revised:

1. Load **three matched-modality language systems** (Codex Round 1 §7, §11 parsimony):
   - Qwen3-0.6B (Class 1, autoregressive LLM), FP16, ~1.3 GB
   - Mamba2-370M (Class 3, SSM), FP16, ~0.75 GB
   - Falcon-H1-0.5B (Class 4, hybrid), FP16, ~0.5 GB
   - Total: ~2.6 GB concurrent. All use same text tokenization-family (we lose nothing across class comparisons because stimulus is matched).
2. **One** stimulus bank: 5000 sentences × 256 tokens from a clean C4 slice. Version = `text/c4_clean_5k_v1_seed42`. Alongside: **two disjoint resamples** (version `_v2_seed123`, `_v3_seed456`) for stimulus-resampling stability (H12).
3. **Untrained control:** random-init Qwen3-0.6B. Measures how much of the primitive is "learned geometry" vs "architectural geometry."
4. Extract point-cloud sequences per system. Per layer k: subsample 5000 sequence-averaged embeddings (seq-mean pooling) → X_k ∈ R^{5000 × d}. Also save per-token embeddings (no pooling) for the pooling-control.
5. Apply primitive ID(X_k) with both TwoNN and MLE estimators (H13 stability check). Also apply to untrained-control and pooling-variant.
6. Compute **variance decomposition (H12 probe)**: `Var[ID] = σ²_system + σ²_k + σ²_stimulus + interactions`, via two-way ANOVA on {system × stimulus_resample}.
7. Apply two-gate semantics (§2.5).

### 2.5. Agnosticism gate semantics (Round 2 priority-directive deliverable)

Derived in response to Codex Round 1 Priority Directive: *"Derive and pre-register the agnosticism gate semantics (invariances + stability tests + semantic comparability criteria) because the current design assumes 'congruent measurements' is the right promotion criterion, and that assumption will invalidate the atlas either by stalling or by false promotion."*

#### 2.5.0 Conceptual framing

The category error in the Round 1 design: it conflated **"the primitive measures the same thing in two systems"** (coordinate promotion — a statement about the primitive itself) with **"the primitive produces the same *number* in two systems"** (universality — a statement about the systems). These are different. A thermometer measures the same thing in a cup of coffee and a block of ice, but the numbers differ — and that difference IS the science. An atlas coordinate must be a valid *measurement* on each system class; whether the values agree or form a universal law is a separate question.

Two gates, in sequence.

#### 2.5.1 Gate 1 — PORTABILITY (coordinate eligibility)

A primitive `f` passes Gate 1 for a system class `C` iff ALL five criteria hold with pre-registered tolerances:

**G1.1 Computability.** There exists a finite algorithm producing `f(m, x)` for `m ∈ C` and `x ~ D_stimulus` within the COMPUTE.md envelope (≤22 GB VRAM, ≤56 GB RAM, ≤4 h wall-clock).

**G1.2 Invariance.** `f` is invariant under the primitive's declared invariance group `G_f`. For common primitives:
- ID (TwoNN / MLE): invariant to global isotropic rescaling of X; invariant to orthogonal rotations; NOT invariant to token permutations unless sequence-mean pooled (then: yes).
- Participation ratio: invariant to orthogonal rotations; NOT invariant to rescaling.
- Koopman spectrum (eigenvalue ECDF): invariant to orthogonal change of basis on X; invariant to rescaling of K; NOT invariant to reparameterizations of the time axis unless pre-registered arc-length normalization is applied.
- Persistent homology (Betti profile): invariant to ambient isometry; stable under Hausdorff perturbation (Cohen-Steiner stability theorem).
- Ricci curvature (Ollivier): invariant to graph isomorphism; sensitive to kNN choice, distance metric, and graph construction — these MUST be pre-registered per primitive use.

The primitive's declared `G_f` is checked at measurement time: apply a random element of `G_f` to `X_k`, verify `f` output is within a pre-registered tolerance.

**G1.3 Stability under stimulus resampling (H12 conditional).** For the primitive's declared `D_stimulus`, draw two disjoint resamples `x₁, x₂ ~ D`. Require
```
|f(m, x₁) − f(m, x₂)| < τ_resample · max(|f(m, x₁)|, |f(m, x₂)|)
```
with `τ_resample` pre-registered (default 0.10). If this fails, the primitive is stimulus-dominated for class `C` and does NOT pass Gate 1 on that class.

**G1.4 Stability under estimator variant (H13 conditional).** For every primitive, at least TWO independent estimators must be declared (e.g., TwoNN vs MLE for ID; centered vs uncentered PR). Require the two estimators to agree within `τ_estimator` (pre-registered, default 0.15) on at least one system from class `C` at a typical depth.

**G1.5 Stability under quantization (H13 primary).** Run the primitive at two points on the quantization ladder (e.g., FP16 and Q8 on the same model) and require agreement within `τ_quant` (pre-registered, default 0.15). If quantization-sensitive, the primitive is hardware-dependent and fails Gate 1.

**Portability promotion:** a primitive is a **coordinate for class C** iff G1.1–G1.5 pass on at least one model in C. It becomes a **coordinate (portability gate passed)** when it has passed on ≥3 distinct classes. Portability makes no universality claim — values may differ across classes.

#### 2.5.2 Gate 2 — UNIVERSALITY (Level-1 claim)

A primitive `f` passes Gate 2 iff ALL five criteria hold:

**G2.1 Portability on ≥5 classes** (per `UNIVERSALITY_LEVELS.md` Level-1 threshold; extends the Round-1 ≥3 portability gate).

**G2.2 Derivation-first functional form.** A candidate form `f(m, x) = g(θ(m), x)` has been derived from first principles BEFORE fitting. The derivation is written up in a pre-reg, cites its theoretical source (info-theory, statistical mechanics, dynamical systems, EVT, etc.), and is LOCKED at pre-reg commit.

**G2.3 Joint-fit residual bound.** Across the ≥5 classes, joint nonlinear fit of `g` with class-specific parameters `θ(m)` has residual
```
RSS_joint / RSS_per_class_independent < α_universal
```
with `α_universal` pre-registered (default 1.5). If the joint fit is worse than ~1.5× per-class independent fits, class-specific models beat universality and the primitive is Level-2 (family constants) not Level-1.

**G2.4 Causal test.** A do-intervention that ablates a coordinate-defined subspace (top-k directions from an SVD on `f`'s input, or a subspace carrying a specific coordinate value) produces a predicted behavior change in the class-appropriate loss/metric. Estimand: `E[loss(m with subspace ablated) − loss(m intact)]` must exceed a pre-registered minimum and show a monotonic response to ablation magnitude.

**G2.5 Biology instantiation specified (may be deferred in execution, not in declaration).** The primitive ships with a concrete biology instantiation: "given a neural-population activity matrix N_neurons × T_timepoints under stimulus `s`, `f` is computed by … with sampling rule …". If the primitive cannot be evaluated on Allen-style recordings even in principle, it is LLM-specific, not a genome coordinate.

**Universality promotion:** a primitive promotes from coordinate to **Level-1 universal coordinate** iff G2.1–G2.5 pass. Below G2.5 the primitive is a **Level-2 family-local coordinate** (universal within a family). Below G2.2 it is a **Phase-2 atlas observation** (pattern exists, derivation pending). These status labels already exist in `research/UNIVERSALITY_LEVELS.md` — this section operationalizes them.

#### 2.5.3 Semantic comparability without numeric agreement

Gate 1 G1.2 requires the primitive be *invariant* to its declared group. Gate 2 G2.3 tests *numeric agreement* across classes via a joint fit. The gap between them is **semantic comparability** — does the primitive "mean the same thing" across classes independent of numerical agreement?

The operational definition:
- **Naming rule.** The primitive's measurement function must be definable in a class-agnostic mathematical statement (e.g., "intrinsic dimension = log-asymptotic growth rate of ball volume / log-radius on the point cloud") whose computation depends only on the point cloud, not on class-specific artifacts.
- **Interpretation rule.** The primitive must have a single documented interpretation that applies to every supported class (not one interpretation per class).
- **Negative-control rule.** The primitive must produce *different* measurements on controls that SHOULD differ (e.g., trained vs untrained model of the same architecture). If a primitive returns the same value on trained and untrained models, it measures the architecture alone, not the representation — it is a Level-0 diagnostic, not a coordinate.

Primitives that fail semantic comparability become Level-0 diagnostics regardless of Gate 1 pass.

#### 2.5.4 Summary: four-tier status taxonomy

This replaces the informal "primitive promotion" talk with explicit status:

| Status | Gate tests passed | What you can say | What you cannot say |
|---|---|---|---|
| **Untested (⚫)** | — | — | — |
| **Diagnostic (Level-0, ⚪)** | class-local, fails semantic comparability or fails G1.2–G1.5 on some classes | "In class C, this primitive behaves like X" | "This primitive measures the same thing across classes" |
| **Coordinate / Level-2 candidate (🟡)** | Gate 1 on ≥3 classes | "This primitive is a semantically-comparable measurement in these 3+ classes; its values are (class-dependent values)" | "There is a universal law" |
| **Coordinate / Level-2 family-local (🟢²)** | Gate 1 on ≥5 classes AND joint fit shows family-specific constants | "There is a family-specific functional form" | "A single functional form holds across families" |
| **Coordinate / Level-1 universal (🟢¹)** | Gate 2 all 5 criteria pass | "This functional form is universal across trained NNs in the ≥5 tested classes, with causal test and biology instantiation" | Nothing more — science is never final |

#### 2.5.5 Pre-registration template (LOCKED at commit)

Every new coordinate promotion requires a pre-reg at `research/prereg/genome_<primitive>_<scope>_YYYY-MM-DD.md` with these fields LOCKED:

1. Primitive name + mathematical definition
2. Supported classes targeted in this prereg
3. Invariance group `G_f` + invariance check protocol
4. Stimulus distribution `D` + resampling protocol
5. Tolerances: `τ_resample`, `τ_estimator`, `τ_quant`, `α_universal` (all with justifications)
6. Estimator variants to be run
7. Quantization ladder points to be run
8. Promotion target: Gate 1 / Gate 2 / universality claim
9. If Gate 2: derivation write-up (attached), causal-test design, biology instantiation spec
10. Kill criterion: what outcome refutes the claim?
11. COMPUTE.md §9 compliance checklist filled
12. Sign-off: "locked at commit `<hash>`; modifications invalidate this prereg"

### 2d. Failure modes (revised)

- **F1. Layer/time axis mismatch.** Transformer depth L=24, Mamba2 L=24, DINOv2 L=12. Normalize by ℓ/L — but this bakes in a "depth is the universal axis" assumption (A2 challenge). If false, curves never collapse.
- **F2. Tokenization / input-space mismatch.** LLMs see tokens, DINOv2 sees patches. Matched stimulus construction is load-bearing. A primitive that depends on input distribution (e.g., ID at raw embedding layer) will never cross cleanly.
- **F3. Primitive estimator bias.** TwoNN biased at small n. Persistent homology sensitive to point-cloud size. Biological comparison limited by Neuropixels population size (≈thousands of neurons) vs. LLM embedding dim (≈ thousands).
- **F4. "Agnosticism gate" semantics.** What counts as "congruent measurement" across classes? If threshold is too loose, false positives; too tight, never promotes anything. **This is the biggest open question in the pipeline.**
- **F5. Stimulus bank dominance.** If different classes require different stimuli, we're not comparing the same function f on the same input x — we're comparing f on different inputs. The universality claim weakens.

### 2e. Self-stress-test — Round 1 outcomes

The Round-1 draft surfaced these challenges. Post-Round-1 status:

| Challenge (pre-Round-1) | Status |
|---|---|
| Trajectory axis assumes sequence/layer ordering; what about symmetric models? | **Addressed** in revised §2a — uniform point-cloud-per-index contract replaces transformer tensor. |
| ≥3-class gate is ad hoc | **Addressed** in §2.5 — ≥3 is *portability* only; Level-1 requires ≥5. |
| Biological bridge is a bolt-on | **Addressed** — co-equal axis (Alt D); every coordinate declares biology instantiation at declare-time. |
| No adversarial / OOD stability | **Partially addressed** — §2.5 G1.3 covers stimulus-resample stability; adversarial perturbation deferred to a Batch-N robustness probe. |
| 7+ primitive menu with no priority | **Addressed** — §3a/§3c prune Batch 1 to ID + PR + spectral slope. Ricci, PH, Koopman moved to Batch 2. |
| (Round-1 adds) Agnosticism gate conflates promotion with universality | **Addressed** in §2.5 two-gate spec. |
| (Round-1 adds) Depth ℓ/L is assumed meaningful cross-class | **Testable now** via P1.5 normalization probe. |
| (Round-1 adds) Stimulus treated as component, not conditioning variable | **Addressed** — stimulus is explicit conditioning variable in every `Measurement` (§2b); P1.4 variance decomposition verifies non-dominance. |

### 2f. Triage of uncertainties from §1d (Round-1 outcomes + additions)

| # | Uncertainty | Disposition | Round-1 outcome |
|---|---|---|---|
| U1 | TwoNN agnosticism across classes? | PROBE | Revised to P1.1 on 3 language classes + controls (Batch 1 lead). |
| U2 | Compression-phase alignment? | PROBE | Operationalized into P1.1 joint fit + P1.3 spectral slope + P1.5 normalization probe. |
| U3 | Ablation equivalence? | PROBE | Deferred to Batch 2 (Gate-2 causal test) with class-agnostic estimand "sensitivity of output to removal of coordinate-defined subspace at matched norm." |
| U4 | PH Betti agnosticism? | PROBE | Deferred to Batch 2 with pre-registered subsampling-stability control. |
| U5 | Ricci on SSM / diffusion literature check | RESEARCH | **Resolved** — null in 2025-2026 lit; H3a probe deferred to Batch 2 with pre-registered graph protocol. |
| U6 | Allen pipeline re-use? | OPERATIONAL | Deferred to Batch 4 (biology bridge), per Codex §5 targeted at vision-class first (DINOv2 + Allen V1). |
| U7 | Moduli-space framing → structural constraints? | THOUGHT | **Resolved** — drove the §2.5 two-gate semantics. |
| U8 | Universal trajectory invariant — exists? | RESEARCH | **Resolved** — Koopman is the strongest candidate; H11 registered; probe deferred to Batch 2 with pre-registered observable family + estimator. |
| **U9** (NEW, from Codex Round 1) | Is `ℓ/L` meaningful cross-class? (A9) | PROBE | P1.5 normalization probe in Batch 1. |
| **U10** (NEW, from Codex Round 1) | Does stimulus variance dominate? (A10, H12) | PROBE | P1.4 variance-decomposition probe in Batch 1. |

### 2g. Open questions — Round 1 outcomes + new Round 2 asks

**Answered in Round 1:**
- Q3 (agnosticism gate sufficient?) — **No.** Split into two-gate spec (§2.5).
- Q4 (H1-H10 cover load-bearing claims?) — **Not quite.** Added H11 (Koopman), H12 (stimulus dominance), H13 (quantization stability).
- Q5 (minimal first atlas?) — **ID + PR + spectral slope on 3 language classes with full Gate-1 control suite.**
- Q6 (inherited paradigm breaking on some class?) — **Yes: ℓ/L depth normalization.** Added A9; P1.5 tests it.
- Q7 (one primitive × 9 classes or 3 primitives × 3 classes?) — **3 primitives × 3 classes first (Batch 1). Then expand classes per primitive that passes Gate 1.**

**Partially answered (open for Round 2):**
- Q1 (moduli-space framing useful?) — Codex did not explicitly rule; it accepted the formalization but questioned whether H10 (multi-chart cover) reflects structural reality. Round 2 asks Codex to explicit-rule on moduli-space usefulness given the two-gate redesign.
- Q2 (trajectory as primary object vs. weights) — Codex's "operator view" addition (Alt A) and `knowledge-surgeon` cross-reference (weight-space writes succeed where activation-space fails) suggest weights may be a separate coordinate axis, not a substitute. Round 2 asks Codex to rule: one object or two?

**New questions for Round 2:**
- Q8. Is the two-gate spec (§2.5) sufficient, or are there still underspecified criteria (e.g., numerical tolerances need derivation, not declaration)?
- Q9. Does the revised Phase 3 Batch 1 actually fix the three kill shots, or does some kill shot persist in a subtler form?
- Q10. Given the three-language-class parsimony, is there a risk of "language-family-local" laws masquerading as universal? How do we detect this before premature claims?
- Q11. With Koopman/Ricci/PH deferred, is there a primitive we're missing that should actually be in Batch 1 (e.g., local-neighborhood overlap from Codex Intuition 2)?
- Q12. What is the priority directive for Round 2?

---

## Phase 3 — Probe batch (Round 1-revised)

### 3a. Batch framing (revised per Codex Round 1 §7, §11)

Batch 1 is now the **Phase-1 coordinate portability + stimulus-dominance sprint.** It answers one load-bearing question: **can ANY primitive pass Gate 1 (§2.5.1) on ≥3 matched-modality language classes, and is its signal ≫ stimulus-bank variance?**

Three critical revisions vs. the Round-1 draft:
1. **Systems restricted to language classes** (Qwen3-0.6B + Mamba2-370M + Falcon-H1-0.5B) — matched tokenization family, single text stimulus bank, no cross-modal confound.
2. **Primitive zoo pruned to Phase-1 MVP** per Codex §11: ID + Participation Ratio (+ spectral slope as auxiliary). Ricci, persistent homology, and Koopman — each scientifically attractive — **defer to Batch 2** to avoid pipeline-bloat before the first coordinate row lands.
3. **Gate 1 semantics (§2.5.1) is the promotion criterion**, not "congruent measurements." Every probe checks G1.1–G1.5 explicitly.

Later batches depend on Batch 1 landing at least one Gate-1-passing primitive. If Batch 1 yields zero, the atlas approach is falsified at the primitive-vocabulary level — **we pivot** (see §3f stop-rule).

### 3b. Shared infrastructure (revised)

**Systems (3 language classes, matched modality):**
- Qwen3-0.6B (Class 1 — autoregressive LLM) — `Qwen/Qwen3-0.6B` — FP16, ~1.3 GB
- Mamba2-370M (Class 3 — SSM) — `state-spaces/mamba2-370m-hf` — FP16, ~0.75 GB
- Falcon-H1-0.5B (Class 4 — hybrid, transformer + Mamba2 layers) — `tiiuae/Falcon-H1-0.5B-Instruct` — FP16, ~0.5 GB
- **Total VRAM footprint: ~2.6 GB concurrent. In envelope.**

**Controls (mandatory per Codex §7):**
- **Untrained control:** random-init Qwen3-0.6B. Separates "learned geometry" from "architectural geometry" (G1.4 negative-control rule).
- **Pooling control:** per-token vs sequence-mean vs last-token. Same model, same stimuli, different pooling. (Codex §7: "ID can change radically with pooling.")
- **Stimulus-resampling control:** three disjoint sub-sets of the stimulus bank. (G1.3 stability test, H12 variance decomposition.)

**Stimulus bank (single, versioned, matched):**
- 5000 sentences × 256 tokens from a clean C4 / Wikipedia slice.
- Version strings: `text/c4_clean_5k_v1_seed42`, `…v2_seed123`, `…v3_seed456`.
- Tokenizer fan-out: each model's native tokenizer applied; sequence-length capped at 256 tokens and truncated. Mild detokenization mismatch is acknowledged and part of the invariance-group declaration for the primitive.
- **No cross-modal stimuli in Batch 1.** Vision, diffusion, JEPA etc. come in later batches with their own conditioning distributions.

**Trajectory extraction → point-cloud sequence** (new interface per §2b):
- Qwen3 and Mamba2: hidden state after each of L blocks, per token. 24 layers × 5000 seq × 256 tokens × D. Store as `PointCloudTrajectory` per system.
- Falcon-H1: hybrid-block output — layer-type (attention vs Mamba) recorded in `meta_k.point_kind_extra`.
- **Pooling:** generate two point-cloud variants per (system, layer):
  - `seq_mean`: one state vector per sequence (n_k = 5000)
  - `per_token_subsample`: random subsample of 5000 tokens across all sequences (n_k = 5000)
- Depth axis `k_normalized = ℓ / L ∈ [0, 1]`. **A9 load-bearing assumption** — explicitly tested by P1.5.

**Artifacts (saved once, reused):**
- `results/activations/<system_id>_<pooling>.npz` — float16, one per (system, pooling) pair → 6 files, ~600 MB total across all three systems
- `results/trajectories/<system_id>_<pooling>.pkl` — `PointCloudTrajectory` objects with metadata
- All excluded by .gitignore (`*.npz`, `*.pkl`). Ledger logs hash + size + config.

### 3c. The 5 probes (pruned + controlled)

**P1.1 — TwoNN intrinsic dimension (LEAD, Gate-1 promotion target)**

- **Question:** Does ID pass Gate 1 on all three language classes? Does the ID(k_normalized) function admit a class-independent g under affine rescaling (H1 Level-2 test)?
- **Hypothesis (H1):** Gate 1 passes on all three. ID(k) is a monotonic-increasing-then-decreasing (hunchback) curve peaking near k ≈ 0.5-0.7 after affine rescaling `d(k) = d_0(m) + α(m)·g(k)`.
- **Counter-hypothesis:** Either (a) Gate 1 fails on one or more classes (stimulus-dominated, quantization-sensitive, or estimator-sensitive — kills ID as a coordinate), OR (b) joint fit fails (kills H1 universality; ID stays Level-2 at best).
- **MVE:**
  1. Compute TwoNN (k=2) at every k_normalized ∈ {0.0, 0.05, …, 1.0} × 3 systems × 2 pooling variants × 3 stimulus resamples × {FP16, Q8} quantization = 21 × 3 × 2 × 3 × 2 = **756 values** + 100-boot bootstrap CIs.
  2. **G1.2 invariance check:** rotate point clouds by random orthogonal, verify ID invariant to rotation within `τ_G1.2 = 0.02`.
  3. **G1.3 stimulus-resample check:** verify `|ID(v1) − ID(v2)| / ID(v1) < τ_resample = 0.10` pairwise across the 3 resamples.
  4. **G1.4 estimator check:** compute MLE-ID alongside TwoNN, verify `|ID_MLE − ID_TwoNN| / mean < τ_estimator = 0.15`.
  5. **G1.5 quantization check:** verify `|ID_FP16 − ID_Q8| / ID_FP16 < τ_quant = 0.15`.
  6. **Negative-control check:** verify `ID_trained ≠ ID_untrained` on at least one layer (else ID measures architecture, not representation — becomes Level-0).
  7. If all Gate-1 checks pass: joint fit `d(k) = d_0(m) + α(m) g(k)`, test universality via F-test (joint-fit RSS vs per-class RSS).
- **Interpretation (per §2.5):** PASS-Gate-1 on class C iff G1.1–G1.5 + negative-control all pass on C. PASS-Level-2 iff Gate 1 on ≥3 classes AND joint-fit RSS / per-class RSS < 1.5 AND g monotonic. PASS-Level-1 needs Gate 2 (separate batch; not claimed from Batch 1 alone).
- **Cost:** TwoNN ~5 s per measurement. 756 × 5 s ≈ 63 min + boot. Activation extraction ~3 h (dominant). **Total: ~4 h wall-clock, fits one experiment.**

**P1.2 — Participation ratio (Gate-1 promotion target)**

- **Question:** Does PR pass Gate 1 on all three language classes? Does PR(k) admit a class-independent functional form?
- **Hypothesis:** Gate 1 passes on all three (covariance statistics are nearly universal). PR(k) will likely show a compression-expansion shape similar to ID but with different sensitivity to pooling.
- **Counter-hypothesis:** Gate 1 fails on one or more classes due to pooling sensitivity (H12 variance decomposition will reveal this).
- **MVE:** Same point clouds as P1.1. Compute PR = `(Σ λ_i)² / Σ λ_i²` where λ_i are covariance eigenvalues. Estimator variants: centered vs uncentered. All G1.1–G1.5 checks + negative control. Joint fit test for Level-2.
- **Cost:** PR is O(D³) eigendecomp — fast at D ≤ 1024. All measurements ≈ 5 min total. Shares activations with P1.1.

**P1.3 — Spectral slope (auxiliary, Gate-1 eligibility check)**

- **Question:** Is the spectral decay exponent of the covariance eigenvalues a Gate-1 coordinate on language classes?
- **MVE:** Fit power law `λ_i ∝ i^{-β}` on top-100 eigenvalues at each (system, k). Report β(k). Apply G1.1–G1.5. (Optional — this is a cheap addition. Codex §11 allows it.)
- **Cost:** Negligible on top of P1.2.

**P1.4 — Stimulus-dominance variance-decomposition probe (H12, Codex §7 required)**

- **Question:** For each primitive in P1.1–P1.3, does stimulus variance dominate system or depth variance?
- **Hypothesis (H12):** Stimulus variance is subdominant: `σ²_stimulus < min(σ²_system, σ²_k) / 2`.
- **Counter-hypothesis:** Stimulus dominates. In that case universality is ill-posed with current stimuli and we need conditional universality (§2.5 Alt C).
- **MVE:** Two-way ANOVA on `{primitive_value ~ system + k_normalized + stimulus_resample + interactions}`. Report `σ²_system : σ²_k : σ²_stimulus` ratio per primitive.
- **Interpretation:** CONFIRM — stimulus variance < 0.5 × min(system, k) → universality claims are well-posed. REFUTE — stimulus dominates → pivot to conditional-universality framing and add more stimulus diversity.
- **Cost:** statistical analysis only on saved measurements. ~10 min.

**P1.5 — Normalization / depth-axis probe (A9 load-bearing test)**

- **Question:** Does the raw depth axis `ℓ/L` align primitive curves cross-class, or is a different reparameterization needed (e.g., arc-length in state space, spectral-measure-rank)?
- **Hypothesis:** `ℓ/L` is at best a weak alignment; a dynamics-based reparameterization (cumulative per-layer state-space speed: `s(ℓ) = Σ_{i≤ℓ} ||X_{i+1} − X_i||_F`, normalized to [0,1]) aligns curves better.
- **Counter-hypothesis:** No reparameterization improves cross-class alignment — depth is fundamentally class-specific.
- **MVE:** Using P1.1 ID curves, fit joint g under (a) raw ℓ/L axis, (b) arc-length axis, (c) spectral-rank axis (k reordered by descending spectral gap). Compare joint-fit RSS under each parameterization. Best parameterization wins.
- **Interpretation:** CONFIRM — at least one reparameterization gives joint-fit RSS < 1.5 × per-class RSS; adopt it. REFUTE — no reparameterization helps; depth is class-specific, H1/H2 stay Level-2 at best even after stimulus controls.
- **Cost:** statistical analysis only on saved measurements. ~15 min.

Note: **Koopman (H11), Ricci (H3a), and persistent-homology probes are deferred to Batch 2** per Codex Round 1 parsimony mandate (§11). Their Gate-1 checks are expensive and require their own preregs with additional estimator-stability controls. They should not slow Batch 1.

### 3d. Batch execution plan (revised; fits COMPUTE.md §9 compliance checklist)

**Three-experiment split** (each ≤4 h wall-clock, checkpointable):

**Exp A — Activation extraction.**
- Wall-clock: ~3.5 h (extraction bound; 3 systems × 2 pooling variants × 3 stimulus resamples × 2 quantizations = 36 runs but shared activations across some axes)
- Max VRAM: ~3 GB (three models loaded concurrently OR sequentially with Q8 second pass)
- Max RAM: ~12 GB (activation buffers for per-token variant)
- Disk artifact: ~800 MB total
- Checkpoint: per (system, pooling, resample, quant) — one `.npz` file each
- Quantization: both FP16 and Q8 per model (for G1.5 stability check)

**Exp B — Primitive computation + Gate 1 checks (P1.1, P1.2, P1.3).**
- Wall-clock: ~2 h (no model loading; ID + PR + spectral-slope across all saved point-clouds)
- Max VRAM: 0 (CPU-only analysis on saved tensors)
- Max RAM: ~12 GB
- Depends on Exp A artifacts
- Includes G1.2 rotation invariance check, G1.3/G1.4/G1.5/negative-control checks

**Exp C — Variance decomposition + normalization probes (P1.4, P1.5) + joint fits.**
- Wall-clock: ~0.5 h (pure statistics on saved measurements)
- Max VRAM: 0
- Max RAM: ~4 GB
- Depends on Exp B measurements
- Output: two-way ANOVA, reparameterization comparison, Gate-1 pass/fail tables, Level-2 joint-fit test

**Compliance checklist per experiment (from COMPUTE.md §9):**
- [x] Max VRAM ≤ 22 GB (3 GB peak, Exp A only)
- [x] Max RAM ≤ 56 GB (12 GB peak)
- [x] Wall-clock ≤ 4 h per experiment (3.5 h + 2 h + 0.5 h split)
- [x] Disk footprint documented (~800 MB)
- [x] Quantization logged (FP16 + Q8 per model)
- [x] Save-resume path: per-tuple `.npz` for Exp A; per-primitive `.json` for Exp B; final summary `.json` for Exp C
- [ ] **Smoke test required before launch** (5 sentences × 2 layers × 1 system, end-to-end through Exp A+B+C) — prereg checkpoint

### 3e. Dependencies and deferrals

**Not in Batch 1** (deferred for anti-entropy — one sprint at a time):
- **Koopman spectrum probe (H11)** — Batch 2, after Batch 1 validates that any coordinate can pass Gate 1 on language classes
- **Ollivier-Ricci probe (H3a)** — Batch 2, with pre-registered graph-construction protocol
- **Persistent-homology probe** — Batch 2, with pre-registered point-cloud subsampling stability check
- **Causal test for any Gate-1-passing primitive (Gate 2 G2.4)** — Batch 3
- **Biology bridge (Allen V1 first, per Codex §5 and §8 — bio-first-for-vision-class)** — Batch 4, for vision-class coordinates; LLM-specific bio bridge via fMRI-language dataset is later still
- **Cross-modal extension** (diffusion, vision encoder, JEPA, world model) — Batches 5+, sequentially per class. **First extension target: vision encoder (DINOv2) with Allen V1 bio bridge co-planned.**

### 3f. Stop-rule (revised per Gate 1 semantics)

If ALL three candidate primitives (ID, PR, spectral slope) fail Gate 1 on at least one of the three language classes: **atlas approach is falsified at the primitive-vocabulary level on language.** Options:
1. **Pivot primitive family.** Move to dynamics-first primitives (Koopman, Lyapunov, local neighborhood overlap) in a new Batch 1'.
2. **Pivot object.** Move from activation geometry to weight-space geometry (informed by `knowledge-surgeon` — LoRA weight-space edits succeed where activation-space edits fail — Mystery 2).
3. **Pivot framing.** Abandon primitive-hunting and attempt a task-based approach (e.g., CTI's rate-distortion lineage).

This is the design kill-criterion for the current Phase-1 direction, now operationalized through Gate 1 rather than informal congruence.



---

## Phase 4 — Codex rounds (active)

Codex invocation log. Each round: timestamp, session id / output file, key findings, priority directive issued.

| Round | Timestamp | Output file | Top finding | Priority directive | Sacred-outcome scores (S1..S7) | Overall (1-10) |
|---|---|---|---|---|---|---|
| 1 | 2026-04-20 | `.codex/outputs/round1.md` | Agnosticism gate conflates "measurement congruence" with "coordinate promotion"; trajectory tensor is transformer-shaped; stimulus treated as component not conditioning variable | Derive and pre-register the agnosticism gate semantics (invariances + stability + semantic comparability) | 2, 2, 3, 3, 3, 4, 3 | 8/10 |
| 2 | queued | `.codex/outputs/round2.md` | — | — | — | — |

---

## Phase 5 — Adversarial audits (scheduled at rounds 5, 10, 15, and at claimed convergence)

*(none yet — triggers at Round 5)*

---

## Phase 6 — Blueprint (pending; produced at convergence)

---

## Heartbeat log

Cron job: `cf3f1112` @ `4,34 * * * *` (session-only, 7-day auto-expire). One-line entries only — see §Heartbeat rules above.

| Time | Status | Drift? | Corrective action | Next priority |
|---|---|---|---|---|
| T+0 (session start) | ON TRACK | No | — | Phase 1a decomposition + Phase 1b research in parallel |
| T+0.5 (post-Phase-1+2 draft) | ON TRACK | No | Codex Round 1 fired in background (task `b2kmq8iou`); U5/U8 research in parallel | Process U5 (null result — Ricci-on-SSM unpublished) and U8 (Koopman strong candidate) into §1b research brief; draft Phase 3 probe batch while Codex runs |
| T+1h (post-Round-1 Codex output) | ON TRACK | No | Codex scored 8/10, flagged 3 kill shots; revised §1a axioms with verdicts, added A9/A10, H12/H13, added §2.5 agnosticism gate semantics (priority directive deliverable), rewrote §2a/§2b/§2c/§2d for point-cloud + operator view, rewrote §3 Batch 1 for language-only + controls | Fire Codex Round 2 on revised artifacts; continue compressing/aligning research brief with intuitions from Round 1 while it runs |

---

*End of session working doc. Becomes `research/BLUEPRINT.md` at convergence.*
