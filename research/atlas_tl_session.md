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

#### Inherited axioms — surfaced for Codex challenge

| # | Axiom | Challenge (what if wrong?) |
|---|---|---|
| A1 | "Representational geometry" is the right abstraction | Dynamics might be primary; geometry is a static snapshot. Trajectories, not snapshots, are the primitive (supported by Koopman + grokking literature). |
| A2 | Primitives transfer if they pass ≥3-class gate | 3 is arbitrary. Why not 5? Why not specific class-pairs with theoretical kinship (e.g., causal-attention architectures)? |
| A3 | Derivation-first for Level-1 | RG discovered phase transitions empirically before Wilson's derivation. Requiring derivation-first may miss real laws. Compromise: admit empirical-first, but demote to Phase-2 until derivation exists. |
| A4 | Biological validation = RSA / Neuropixels / fMRI | Neuropixels is single-modality (mostly mouse V1). fMRI is coarse (mm³). Behavior and evolutionary constraints may be deeper bridges. Also: brain-alignment lit is all LLM-to-brain — SSM-to-brain, diffusion-to-brain unexplored. |
| A5 | 9-class bestiary covers trained NNs | Missing: spiking NNs, pre-transformer RNNs (LSTM/GRU distinct from SSM). Also missing: ensembles, MoEs treated as separate class, graph NNs. |
| A6 | RTX 5090 doesn't limit WHAT is in the atlas, only HOW fast | Absolutely false. We cannot fit >32B models at reasonable quantization. Our atlas is biased to the ≤10B dense / ≤20B MoE regime. State this explicitly — it is a SCOPE of the atlas, not a weakness. |
| A7 | 3-tier framework is sufficient | Possibly missing Level-0 (class-specific diagnostic only) and Level-4 (meta-law relating Level-1 laws). |
| A8 | Intrinsic dim, participation ratio, CKA, RSA are candidate coordinates | Recent lit: CKA scale-confounded (Aristotelian-view paper Feb 2026), some ID estimators biased at small n, RSA depends on stimulus choice. These aren't ground truth. |

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

## Phase 2 — Mental machine (DRAFT; to be refined after Codex Round 1)

The atlas construction pipeline.

### 2a. Component inventory

```
[model registry]           ←  ../../models/registry.py (canonical)
        ↓
[system loader]            ←  class-specific adapter: load model at the quantization
        |                     chosen for its parameter bucket (COMPUTE.md ladder)
        ↓
[stimulus bank]            ←  fixed input sets per task: language (Wikipedia sample,
        |                     c4), vision (ImageNet val subset), control (random).
        ↓
[trajectory extractor]     ←  class-specific hook: extract {h_ℓ(x)} for transformer,
        |                     {h_t(x)} for SSM, {h_τ(x)} for diffusion, {h_ctx, h_tgt}
        |                     for JEPA. Returns a uniform trajectory representation
        |                     tagged with class, layer/time/step, token/patch index.
        ↓
[primitive evaluator]      ←  f: trajectory → measurement. Plug-in architecture:
        |                     each primitive is a module with a class-agnostic
        |                     interface. Returns structured measurement.
        ↓
[atlas row builder]        ←  assemble (system_id, layer_or_time, primitive_id,
        |                     task_id, value, uncertainty) → one ledger entry
        ↓
[agnosticism gate]         ←  for a primitive, tracks which classes it has been
        |                     tested on. Promotes primitive to "coordinate" when
        |                     ≥3 classes pass a prereg agreement criterion.
        ↓
[causal tester]            ←  for a candidate Level-1 claim: runs do-intervention
        |                     / ablation that predicts behavior change if the claim
        |                     is real.
        ↓
[biological bridge]        ←  same primitive applied to Allen Neuropixels / fMRI
        |                     under matched stimuli. For Level-1 claims only.
        ↓
[law fitter]               ←  derivation-first: given functional form from theory,
                              fit constants + uncertainty + cross-validation residuals.
```

### 2b. Interface contracts

- **Trajectory** = `{"system_id": str, "class": {1..9}, "states": Tensor[L_or_T, N_tokens, D], "metadata": {tokenizer, task, seed, ...}}`. L_or_T axis is "depth" for transformers, "time" for SSM, "noise step" for diffusion.
- **Primitive module** must implement: `evaluate(trajectory: Trajectory) → Measurement`. `Measurement = {value: Any, uncertainty: Any, metadata: dict}`. Must declare supported `class_ids` — primitives raise clearly if asked on an unsupported class.
- **Agnosticism gate** reads the primitive's class list + ledger; promotes when ≥3 classes have produced congruent measurements (definition of congruent varies by primitive — part of the prereg).

### 2c. Data flow — worked example (H1 test, U1 probe)

1. Load Qwen3-0.6B (FP16, ~1.3 GB), Mamba2-370M (FP16, ~0.7 GB), DINOv2-small (FP16, ~0.1 GB). Total ≤ 3 GB — well in envelope; can load all three concurrently.
2. Build matched stimulus bank: 5000 sentences from Wikipedia clean, chunked to 256 tokens for LLMs. For DINOv2, corresponding sentences have no direct visual analogue — so define a separate vision stimulus bank (ImageNet val 5000 images) and ALSO a matched-difficulty text-vision pair set (LAION/MSCOCO captions + images).
3. Extract trajectories per class. For Qwen3 and Mamba2: residual/hidden state at each of L layers for each of 256 tokens, averaged within sequence to get L × 768 for transformer (L=24) and L × 1024 for Mamba2 (L=24). For DINOv2-small: patch-token outputs at each of 12 blocks.
4. Per primitive (TwoNN), compute ID at each normalized layer position ℓ/L. Per system get ID(ℓ/L) curve.
5. **The question:** do the curves collapse under architecture-normalization? H1 predicts yes (after d₀, α rescaling).

### 2d. Failure modes

- **F1. Layer/time axis mismatch.** Transformer depth L=24, Mamba2 L=24, DINOv2 L=12. Normalize by ℓ/L — but this bakes in a "depth is the universal axis" assumption (A2 challenge). If false, curves never collapse.
- **F2. Tokenization / input-space mismatch.** LLMs see tokens, DINOv2 sees patches. Matched stimulus construction is load-bearing. A primitive that depends on input distribution (e.g., ID at raw embedding layer) will never cross cleanly.
- **F3. Primitive estimator bias.** TwoNN biased at small n. Persistent homology sensitive to point-cloud size. Biological comparison limited by Neuropixels population size (≈thousands of neurons) vs. LLM embedding dim (≈ thousands).
- **F4. "Agnosticism gate" semantics.** What counts as "congruent measurement" across classes? If threshold is too loose, false positives; too tight, never promotes anything. **This is the biggest open question in the pipeline.**
- **F5. Stimulus bank dominance.** If different classes require different stimuli, we're not comparing the same function f on the same input x — we're comparing f on different inputs. The universality claim weakens.

### 2e. Stress test — what I'd challenge if I were Codex

- "Your trajectory definition implicitly assumes a sequence/layer axis. What about bag-of-words encoders, symmetric models, models with no inherent ordering? Define your object more carefully."
- "Your ≥3-class gate is ad hoc. You should either (a) derive the minimal statistical sample for a valid universality claim (conditional on assumed effect size), or (b) admit that 3 is a convention and cite the precedent."
- "Your biological bridge is a bolt-on. It should be a co-equal axis of the design, with Level-1 claims required to include biology from the start."
- "You haven't addressed adversarial / OOD stability. A primitive that agrees across clean inputs but diverges under weight perturbation or adversarial stimuli is a false positive."
- "You list 7+ primitive families with no priority. Without explicit ranking + kill criteria per primitive, the atlas becomes a menu, not a plan."

### 2f. Triage of uncertainties from §1d

| # | Uncertainty | Disposition | Action |
|---|---|---|---|
| U1 | TwoNN collapse across 3 classes? | PROBE | Phase-3 batch: lead experiment |
| U2 | Compression-phase alignment? | PROBE | Phase-3 batch: piggyback on U1 trajectories |
| U3 | Ablation equivalence? | PROBE | Phase-3 batch: depends on U1+U2 |
| U4 | PH Betti numbers? | PROBE | Phase-3 batch: secondary (cost concern; PH is O(n³)) |
| U5 | Ricci on SSM / diffusion — literature check | RESEARCH | Fire follow-up web search before Round 2 |
| U6 | Allen pipeline re-use? | OPERATIONAL | Defer to implementation; CTI reference scripts exist |
| U7 | Moduli-space framing → structural constraints? | THOUGHT | Resolved: forces Level-0 diagnostic / Level-1 coordinate distinction |
| U8 | Universal trajectory invariant — exists? | RESEARCH | Fire follow-up: Koopman / Lyapunov / spectral-entropy comparative lit |

### 2g. Open questions for Codex Round 1 (priority-ordered)

1. Is the moduli-space framing (§1a) a useful formalization, or pretentious decoration over a simpler framing?
2. Is "trajectory through state space" the right primary object, or should weights / dynamics-of-weights also be first-class?
3. Is the agnosticism gate (≥3 classes) sufficient, or does Phase-1 need a stronger statistical framework?
4. Do the 10 hypotheses H1–H10 cover the load-bearing claims, or is a critical one missing?
5. Given the envelope, what is the MINIMAL first atlas — the smallest sprint that yields a defensible Level-1 claim?
6. Is there an inherited paradigm we're assuming (e.g., "layers are discrete, indexable by integer") that breaks on some class?
7. Should the Phase-1 minimum viable atlas be *one primitive across all 9 classes* or *three primitives across 3 classes*? What's the tradeoff?

---

## Phase 3 — Probe batch (DRAFT — will be stress-tested at Codex Round 2)

### 3a. Batch framing

Batch 1 is the **Phase-1 primitive agnosticism sprint.** One set of 3 systems, matched stimulus banks per modality, parallel extraction of activations, four geometric primitives computed on the same trajectories. Any primitive whose measurements superimpose across the 3 classes (under its per-primitive congruence criterion) becomes a Level-2 candidate coordinate. Any that additionally has a first-principles derivation and survives a causal test becomes a Level-1 candidate — the target Phase-6 deliverable.

Later batches (causal tests, biological bridge) depend on Batch 1 picking a winner, so defer.

### 3b. Shared infrastructure

**Systems (3 classes, all in-envelope):**
- Qwen3-0.6B (autoregressive LLM) — `Qwen/Qwen3-0.6B` — FP16, ~1.3 GB
- Mamba2-370M (SSM) — `state-spaces/mamba2-370m-hf` — FP16, ~0.75 GB
- DINOv2-small (vision encoder) — `facebook/dinov2-small` — FP16, ~0.1 GB
- **Total VRAM footprint: ~2.2 GB concurrent. Well under the 22 GB ceiling.**

**Stimulus bank (per-modality natural inputs, NOT content-matched):** Choice justified under A1 challenge — universality is about INTERNAL geometry, not input-matched behavior (Platonic framing). If Codex disputes, we add a content-matched run (MSCOCO captions + images) as a control.
- LLMs: 5000 sentences of 256 tokens each, sampled from a clean C4 / Wikipedia slice, seed-fixed.
- DINOv2: 5000 ImageNet-val images at native resolution.

**Trajectory extraction:** per-system adapter hooks the canonical internal state. For Qwen3/Mamba2: residual / hidden state at every layer after the block (post-LN), averaged across tokens per sequence → L × D per input. For DINOv2: CLS token + patch-mean at each block → L × D per input. Depth axis ℓ/L is normalized to [0, 1] for cross-system comparison.

**Artifacts (one per system, saved once, reused across probes):**
- `results/activations/<system_id>.npz` — `(N=5000, L, D_system)` float16 tensor + metadata
- `results/trajectories/<system_id>.npz` — averaged-per-sequence L × D tensors
- Kept out of git (.gitignore excludes `*.npz`). Logged in ledger with hash.

### 3c. The 4 probes

**P1.1 — TwoNN intrinsic dimension agnosticism test (LEAD)**

- **Question:** Does ID(ℓ/L) factor across architectures as d(ℓ/L) = d₀(system) + α(system) · g(ℓ/L) with a single universal g?
- **Hypothesis (H1):** Yes. g is a monotonic increasing-then-decreasing (hunchback) curve peaking near ℓ/L ≈ 0.6 in all three.
- **Counter-hypothesis:** Curves cross or fail to fit a single g under any affine rescaling. Architecture dictates geometry, not a universal.
- **MVE:** Compute TwoNN ID (Facco et al.) with k=2 at every ℓ/L ∈ {0.0, 0.05, ..., 1.0} on each system's trajectory set. 100 bootstrap resamples of 1000 sequences each to get confidence intervals. Fit single g by joint nonlinear least squares across systems with (d₀, α) per system. Test residual vs. per-system-independent fit via F-test.
- **Interpretation:** CONFIRM — joint fit residual < 1.5× per-system residual (i.e., universality loses little) and g is monotonic. REFUTE — joint residual > 3× per-system residual OR g is non-monotonic in any class. AMBIGUOUS — in between → add more stimuli.
- **Cost:** TwoNN ~10 s per (system, ℓ). 3 × 21 × 10 s ≈ 10 min compute after activations saved. Activation extraction dominates (~3 h for 3 systems). **Total: ~3.5 h wall-clock.** Checkpointable (per-system).

**P1.2 — Ollivier-Ricci curvature agnosticism test**

- **Question:** Does mean discrete Ollivier-Ricci curvature of a kNN-5 graph on hidden states share a sign (and sign-curve over ℓ/L) across the three classes?
- **Hypothesis (H3a):** All three show negative mean Ricci at middle layers. Promotes H3 to Level-1 (negative curvature = decoder-causal architecture with trained representations, regardless of architecture family).
- **Counter-hypothesis:** Mamba2 and DINOv2 show positive or ≈0 mean Ricci. H3 stays Level-2 (transformer-specific).
- **MVE:** For each system, subsample 1000 sequence-averaged embeddings per layer (every 4th layer — 6 layers total per system, 18 runs total). Build kNN-5 graph, compute Ollivier-Ricci via `GraphRicciCurvature` (Wasserstein-1 variant). Report mean curvature per layer + 95% bootstrap CI.
- **Interpretation:** CONFIRM — mean Ricci negative in middle layer-band for all 3 systems (no overlap with 0 in CI). REFUTE — any system has positive mean Ricci anywhere in middle band. AMBIGUOUS — mixed signs with overlapping CIs.
- **Cost:** Graph construction O(n²) on 1000 points ≈ seconds. Ricci ≈ 2–5 min per layer. 18 × 5 min = **90 min total after activations are extracted.** Activations shared with P1.1. **Fits in envelope.**

**P1.3 — Koopman spectrum agnosticism test**

- **Question:** Does the Koopman spectrum of the layer-indexed trajectory have a universal functional form across transformer/SSM/vision?
- **Hypothesis (H11):** Yes. Koopman-mode eigenvalue distributions superimpose on a log-log rank plot after per-system rescaling (Level-1 candidate).
- **Counter-hypothesis:** Distributions have architecture-family-specific shapes even after rescaling.
- **MVE:** For each system, treat the (5000, L, D) trajectory tensor as 5000 time-series of length L. Use DMD (dynamic mode decomposition, Schmid 2010) on the per-input L × D matrix; collect eigenvalue spectrum. Aggregate spectra across 5000 inputs → per-system ECDF of |λ_i|. Compare by Wasserstein-1 distance between per-system ECDFs on log-log. Use 100 bootstraps for CI. If log-log linearizes and slopes match across systems within CI: universal power-law Koopman spectrum.
- **Interpretation:** CONFIRM — Wasserstein distances between all pairs < intra-system bootstrap scatter. REFUTE — distances exceed 3× intra-system scatter. AMBIGUOUS — intermediate.
- **Cost:** DMD is cheap (SVD of L × D matrices). ~1 s per input. 5000 × 3 = 15000 DMDs ≈ 4 h, but can batch ≈ 100 DMDs at once on GPU → ~150 seconds total. Aggregation + bootstrap another few min. **Total: ~10 min after activations. Negligible cost.**

**P1.4 — Persistent homology Betti-0/Betti-1 agnosticism test**

- **Question:** Do the Betti-0 and Betti-1 curves over a Vietoris-Rips filtration on each layer's hidden-state cloud share a universal profile across classes?
- **Hypothesis (inherits from WIKI primitive priority):** Yes. Betti profiles align when normalized by characteristic distance scale.
- **Counter-hypothesis:** Topology is class-specific — Mamba2 shows distinct Betti-1 structure tied to recurrence.
- **MVE:** Subsample 500 sequence-averaged embeddings per layer (every 4th layer — 6 layers/system). Compute persistent homology Betti-0 and Betti-1 via `Ripser` with max dimension 2, max edge length normalized to kNN-5 median. Report barcode lengths and Betti numbers at persistent thresholds. Compare across systems with bottleneck distance.
- **Interpretation:** CONFIRM — bottleneck distance per layer < intra-system bootstrap scatter. REFUTE — bottleneck distance > 3× intra-system. AMBIGUOUS — between.
- **Cost:** PH on 500 points in reduced-D (SVD-to-50) is ~1–2 min per layer. 18 runs × 2 min = **36 min after activations. Fits.**

### 3d. Batch execution plan (fits COMPUTE.md §9 compliance checklist)

**Two-experiment split** (each ≤4 h wall-clock, checkpointable):

**Exp A — Activation extraction + fast primitives (TwoNN + Koopman).**
- Wall-clock: ~4 h (extraction bound)
- Max VRAM: ~3 GB (three models loaded concurrently)
- Max RAM: ~8 GB (activation buffers)
- Disk artifact: ~600 MB across all three systems
- Checkpoint: after each system's extraction finishes, save `.npz` before loading next; TwoNN and Koopman compute after save
- Quantization: FP16 for all three (per COMPUTE.md ladder for <1B dense)

**Exp B — Slow primitives (Ricci + PH) on saved activations.**
- Wall-clock: ~2.5 h (no model loading, just analysis)
- Max VRAM: ~0 GB (CPU-only analysis)
- Max RAM: ~10 GB (point-cloud processing + Ripser)
- Depends on Exp A artifacts
- Checkpoint: per (system, layer) pair

**Compliance checklist per experiment (from COMPUTE.md §9):**
- [x] Max VRAM ≤ 22 GB (3 GB peak)
- [x] Max RAM ≤ 56 GB (10 GB peak)
- [x] Wall-clock ≤ 4 h (with split, yes)
- [x] Disk footprint documented (~600 MB)
- [x] Quantization logged (FP16)
- [x] Save-resume path: per-system for Exp A, per-(system, layer) for Exp B — verified viable; smoke-test required before launch (5 inputs × 2 layers)

### 3e. Dependencies and deferrals

**Not in Batch 1:**
- **Causal tests** (for any candidate Level-1 that emerges) — Batch 2, after Batch 1
- **Biological bridge** (Allen Neuropixels RSA/ID/Koopman) — Batch 3, after a primitive is promoted
- **Additional classes** (diffusion, reasoning, hybrid, JEPA, world model, controls) — Batches 4+, sequentially per system class added to registry
- **Content-matched stimuli control** (MSCOCO) — added to Batch 1 only if Codex disputes the "natural-inputs" choice

### 3f. Stop-rule

If ALL four primitives in Batch 1 fail agnosticism at their congruence criterion: **atlas approach is falsified at the primitive level.** Pivot to a different class of object — most likely weight-space structure or dynamics-of-weights — before committing more compute. This is the design kill-criterion for the current Phase-1 direction.



---

## Phase 2 — Mental machine (pending)

---

## Phase 3 — Probe batch (pending)

---

## Phase 4 — Codex rounds (pending)

Codex invocation log. Each round: timestamp, session id / -o file, key findings, priority directive issued.

---

## Phase 5 — Adversarial audits (pending)

---

## Phase 6 — Blueprint (pending)

---

## Heartbeat log

Cron job: `cf3f1112` @ `4,34 * * * *` (session-only, 7-day auto-expire). One-line entries only — see §Heartbeat rules above.

| Time | Status | Drift? | Corrective action | Next priority |
|---|---|---|---|---|
| T+0 (session start) | ON TRACK | No | — | Phase 1a decomposition + Phase 1b research in parallel |
| T+0.5 (post-Phase-1+2 draft) | ON TRACK | No | Codex Round 1 fired in background (task `b2kmq8iou`); U5/U8 research in parallel | Process U5 (null result — Ricci-on-SSM unpublished) and U8 (Koopman strong candidate) into §1b research brief; draft Phase 3 probe batch while Codex runs |

---

*End of session working doc. Becomes `research/BLUEPRINT.md` at convergence.*
