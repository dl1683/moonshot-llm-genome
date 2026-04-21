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
| **A9** | **(NEW; Codex flag)** Depth normalization ℓ/L is meaningful cross-class. | If wrong, H1/H2 collapse — comparing curves by ℓ/L assumes an equivalence class that may not exist. | **LOAD-BEARING.** Must be tested empirically (probe P1.5 — corrected from earlier "P1.6" typo). Dynamical-systems lens: the proper universal parameter is *arc-length in state space* or *spectral measure*, not raw index. Reparameterization family + selection criterion pre-registered in §3c to prevent post-hoc axis shopping (Codex Round 2 §Q3). |
| **A10** | **(NEW; Codex flag)** Stimulus-bank comparability isn't the dominant confound. | If wrong, "universality" is ill-posed: we're comparing different problems. | **LOAD-BEARING.** Must be tested: variance decomposition per primitive into `{system, depth/time, stimulus}`. Universality requires stimulus variance NOT to dominate. Pipeline must treat stimulus as the *conditioning variable of every coordinate*, not a component. |

---

### 1b. Leibniz research brief (Round 2-compressed)

Compressed per Codex Round 2 §11 parsimony mandate. Only load-bearing-for-Batch-1 content in full; others are pointers. Full expanded brief preserved in git history (commits `dd17cc6`, `d638e5b`) + `.codex/outputs/round1.md`.

#### B2. Representation similarity & cross-architecture alignment (load-bearing for P1.3 + H4)

- **Platonic Representation Hypothesis** (Huh et al. 2024): cross-modality convergence with scale. [[arXiv:2405.07987]](https://arxiv.org/abs/2405.07987)
- **Aristotelian-view critical revision** (Feb 2026): similarity metrics are scale-confounded; global spectral convergence largely disappears under calibration; **only local neighborhood similarity survives cross-modality.** Direct motivation for P1.3 local-neighborhood primitive. [[arXiv:2602.14486]](https://arxiv.org/abs/2602.14486)
- **CKA limitations** (Kornblith et al. + 2024-2025 follow-ups): PC-dominated; Procrustes + Bures more sensitive. → CKA demoted to ⚪ diagnostic in `MEASUREMENT_PRIMITIVES §3.1`. [[ICLR 2025]](https://proceedings.iclr.cc/paper_files/paper/2025/file/03d113a060c0ac93a5859517a0f07271-Paper-Conference.pdf)

#### B3. Geometric primitives (load-bearing for P1.1 + P1.2)

- **TwoNN intrinsic dimension** (Facco et al. 2017): 1st-to-2nd NN distance ratio is Pareto-distributed; MLE gives ID. Word embeddings extrinsic 300 → ID 10-30. Fine-tuning lowers local ID.
- **Local ID of contextual embeddings** (Ruppik et al. 2025): fine-tuned LLMs show markedly lower local ID than bases. [[arXiv:2506.01034]](https://arxiv.org/html/2506.01034v1)
- **Participation ratio** — Gao & Ganguli 2015 (neural recordings) provides the biology-bridge anchor in prereg §3.7 §9a.
- **Ollivier-Ricci on LLM embeddings** (HELM 2025): decoder-only LLMs show negative Ricci → hyperbolic. Batch-2 primitive. [[arXiv:2505.24722]](https://arxiv.org/html/2505.24722)
- **Ricci for representational alignment** (Fumero et al. 2025). [[arXiv:2501.00919]](https://arxiv.org/html/2501.00919)
- **Persistent homology for LLMs** (2024-2025): TDA on attention / latent representations. Batch-2+ primitive. [[review]](https://link.springer.com/article/10.1007/s10462-025-11462-w)

#### B8d. Local-neighborhood primitives (load-bearing for P1.3 after Round-2 swap)

- **NNGS** — Jaccard of kNN graphs between two embeddings; strongly correlates with downstream task accuracy. Cross-system diagnostic, Level-0. [[arXiv:2411.08687]](https://arxiv.org/html/2411.08687v1)
- **NPE** (He et al. 2005), **LNCA** — local-manifold foundations.
- **For the atlas:** per-point kNN-5 Jaccard self-stability is a per-system Gate-1 candidate (P1.3 in Batch 1). Per-system variants: clustering coefficient, diffusion entropy at t, local reachability ratio.

#### B8c. Ricci on SSM / diffusion — NULL RESULT (Batch-2 opportunity)

No published Ollivier-Ricci measurement on Mamba/SSM or diffusion U-Net latents in 2025-2026. Only LLM embeddings (HELM + Fumero) have been Ricci-measured. → H3a Batch-2 probe is new science.

#### B8b. Koopman spectrum — strongest cross-class candidate (H11, Batch 2)

Koopman operator theory now applied across all three target classes within 2025-2026:
- Koopman-enhanced transformers (Mar 2026) [[arXiv:2602.02592]](https://arxiv.org/html/2602.02592)
- Residual Koopman Spectral Profiling for Mamba (Feb 2026) [[arXiv:2602.22988]](https://arxiv.org/html/2602.22988)
- Hierarchical Koopman Diffusion (Oct 2025) [[arXiv:2510.12220]](https://arxiv.org/abs/2510.12220)
- Koopman-Wasserstein generative (Feb 2026) [[arXiv:2512.18837]](https://arxiv.org/html/2512.18837)
- Pseudo-resolvent Koopman for spectral-pollution control (Feb 2026) [[arXiv:2512.24953]](https://arxiv.org/abs/2512.24953)
- JEPA recovers Koopman invariants.

**Caveat (per Codex Round 2):** estimator choices (observable family, Hankel depth, rank truncation) dominate the spectrum — prereg must LOCK these before any run. Not "all upside."

#### B5 / B8 — minimal pointers (secondary for Batch 1)

- **B5 SSM/Mamba interpretability:** Mamba-2 activation patching + causal mediation finds factual info in same layer bands as transformers → first cross-class causal signal (H7, Batch 2+). Mamba-3 (ICLR 2026).
- **B8 Brain-LLM alignment:** LLM-to-brain RSA scales with model size (774M → 65B) [[Nature Comput Sci]](https://www.nature.com/articles/s43588-025-00863-0). Brain-informed FT gains scale with size + duration. **Gap:** SSM/diffusion/JEPA-to-brain alignment unpublished → direct genome opportunity. fMRI-LM (Nov 2025) infrastructure; brain-decoding survey (Nov 2025 bioRxiv).

#### B1 / B4 / B6 / B7 / B9 — archived pointers (not load-bearing for Batch 1)

Full content in git history commits `d638e5b`/`dd17cc6` + `.codex/outputs/round1.md`. Summary:

- **B1 Mech-interp state-of-art:** Anthropic circuit tracing (cross-layer transcoders), USAE (Universal SAEs), SAE dark-matter limitation (>90% of error norm is linearly predictable from input; scaling SAE width doesn't help — defers SAE-family primitives to Batch 5+), Nanda Sept-2025 pessimism update, MIT Tech Review 2026 Breakthrough designation.
- **B4 Information-theoretic + statistical physics:** Generalized Information Bottleneck (GIB, synergy-based), two-phase learning dynamics (curve-fit → compression), grokking = dimensional phase transition with SOC (Rubin et al. 2026), provable feature-emergence scaling laws (Sept 2025), spectral-entropy collapse as grokking signature, RG-as-coarse-graining (dynamic-neuron RG 2025, Koch-Janusz 2018). → H5 phase-transition hypothesis + H9 RG-lens deferred to Batch 2+.
- **B6 Diffusion interp:** DIFFLENS (CVPR 2025), mechanistic-SAE on diffusion — early steps control composition, middle stylistic, late texture. → Batch 5+.
- **B7 JEPA geometric theory:** collapse theorems, Koopman-invariant recovery, V-JEPA 2.1 (2026), LeJEPA. → Batch 5+.
- **B9 Critical limitations of mech-interp:** "feature" undefined, NP-hard queries, fundamental math limits on linear interventions in chaotic deep networks, temporal-agentic gap. → Scope caveat: the atlas does NOT promise mech-interp; it promises geometry.

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

**H13. Quantization stability (NEW; Codex Round 1 gap).** Every candidate coordinate must be stable under the project quantization ladder (FP16 → Q8 → Q4_K_M) within a prereg-specified tolerance. A coordinate whose measurement changes materially under quantization is hardware-dependent, not universal. Conf: N/A. **Kill:** primitive values drift by > noise-calibrated bound (§2.5.6) across FP16 vs Q6 on the same model, rendering the primitive hardware-dependent.

**H14. Subsample stability (NEW; Codex Round 2 gap).** Point-cloud-based primitives must asymptote in n. For a primitive `f` applied on a point cloud of size n, the variance-over-n curve must flatten before the prereg's sampling budget. Conf: N/A. **Kill:** `f` has not asymptoted at the prereg-declared n; in that case the primitive is undersampled and its Gate-1 verdict is invalid. Pivot: increase n within envelope, or accept that the primitive is non-viable at the current compute budget.

**~~H15~~ (retired from H-register; moved to Governance Rule §G-modality-scope per Codex Round 3 Q6).** H15 was misclassified as a hypothesis — it is a claim-scope policy, not a falsifiable statement. See §1f (new) for the reclassification.

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

A primitive `f` passes Gate 1 for a system class `C` iff all seven criteria (G1.1–G1.7) hold under the noise-calibrated decision rule of §2.5.6:

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

**G1.5 Stability under quantization (H13 primary).** Run the primitive at two points on the quantization ladder (e.g., FP16 and Q8 on the same model) and require agreement within a noise-calibrated bound (§2.5.6). If quantization-sensitive, the primitive is hardware-dependent and fails Gate 1.

**G1.6 Subsample stability (NEW — Codex Round 2 blocking-level).** For the point-cloud–based primitives that comprise most of the atlas vocabulary, estimator behavior is n-dependent. Require an n-sweep: compute `f` at n ∈ {500, 1000, 2000, 5000} with resampling. Define `f` as Gate-1-stable iff it asymptotes (difference at n=2000 vs n=5000 within noise-calibrated bound, §2.5.6). Any primitive that has not asymptoted at n=5000 fails G1.6 and is not portable at the current sample budget.

**G1.7 Preprocessing / metric declaration (NEW — Codex Round 2 §4).** Every primitive declares its preprocessing pipeline (centering, normalization, whitening, projection) and metric (Euclidean, cosine, geodesic on a sphere, etc.). Changing preprocessing/metric produces a different primitive — not a variant of the same one. Variant-vs-estimator distinction is: estimators are swappable implementations of the SAME mathematical quantity; preprocessing/metric choices are DIFFERENT mathematical quantities. The primitive's G_f invariance group is defined with reference to its declared preprocessing — e.g., ID after whitening has trivial rescaling invariance, while ID without whitening is only scale-invariant up to global factors.

**Portability promotion:** a primitive is a **coordinate for class C** iff G1.1–G1.7 pass on at least one model in C. It becomes a **coordinate (portability gate passed)** when it has passed on ≥3 distinct classes. Portability makes no universality claim — values may differ across classes. Every portable coordinate is scope-labeled `(modality, stimulus-family, pooling, tokenizer)` and that scope is part of the coordinate identity.

#### 2.5.2 Gate 2 — UNIVERSALITY (Level-1 claim)

A primitive `f` passes Gate 2 iff ALL five criteria hold:

**G2.1 Portability on ≥5 classes** (per `UNIVERSALITY_LEVELS.md` Level-1 threshold; extends the Round-1 ≥3 portability gate).

**G2.2 Derivation-first functional form.** A candidate form `f(m, x) = g(θ(m), x)` has been derived from first principles BEFORE fitting. The derivation is written up in a pre-reg, cites its theoretical source (info-theory, statistical mechanics, dynamical systems, EVT, etc.), and is LOCKED at pre-reg commit.

**G2.3 Hierarchical model comparison (replaces RSS ratio; see §2.5.6e for full procedure).** Across the ≥5 classes, compare class-specific model A vs universal model B via LRT (α=0.01) + ΔBIC > 10 + AIC agreement + leave-one-class-out holdout error within in-sample scatter. All four conditions must hold for Level-1 promotion. Partial agreement → Level-2 family-local.

**G2.4 Causal test (operational definition for scalar primitives).** A coordinate-defined subspace is: the top-k principal directions of the point-cloud covariance that carry the coordinate's value above a pre-reg threshold (for ID: the top-k eigenvectors explaining the claimed intrinsic dimension; for PR: the top-PR eigenvectors; for clustering coefficient: the k nearest neighbors of each high-clustering-coefficient point). The causal test: ablate this subspace (project out the top-k directions, then project back to original dim) and measure behavior change in a class-appropriate metric. Estimand: `E[loss(m with subspace ablated) − loss(m intact)]` must (i) exceed a pre-registered minimum effect size and (ii) show a monotonic response to ablation magnitude (k → 2k → 4k). Pre-reg locks (a) the ablation procedure, (b) the behavior metric per class, (c) the minimum effect.

**G2.5 Biology instantiation specified (may be deferred in execution, not in declaration).** The primitive ships with a concrete biology instantiation: "given a neural-population activity matrix N_neurons × T_timepoints under stimulus `s`, `f` is computed by … with sampling rule …". If the primitive cannot be evaluated on Allen-style recordings even in principle, it is LLM-specific, not a genome coordinate.

**Universality promotion:** a primitive promotes from coordinate to **Level-1 universal coordinate** iff G2.1–G2.5 pass. Below G2.5 the primitive is a **Level-2 family-local coordinate** (universal within a family). Below G2.2 it is a **Phase-2 atlas observation** (pattern exists, derivation pending). These status labels already exist in `research/UNIVERSALITY_LEVELS.md` — this section operationalizes them.

#### 2.5.3 Semantic comparability without numeric agreement

Gate 1 G1.2 requires the primitive be *invariant* to its declared group. Gate 2 G2.3 tests *numeric agreement* across classes via a joint fit. The gap between them is **semantic comparability** — does the primitive "mean the same thing" across classes independent of numerical agreement?

The operational definition:
- **Naming rule.** The primitive's measurement function must be definable in a class-agnostic mathematical statement (e.g., "intrinsic dimension = log-asymptotic growth rate of ball volume / log-radius on the point cloud") whose computation depends only on the point cloud, not on class-specific artifacts.
- **Interpretation rule.** The primitive must have a single documented interpretation that applies to every supported class (not one interpretation per class).
- **Negative-control rule.** The primitive must produce *different* measurements on controls that SHOULD differ (e.g., trained vs untrained model of the same architecture). If a primitive returns the same value on trained and untrained models, it measures the architecture alone, not the representation — it is a Level-0 diagnostic, not a coordinate.

Primitives that fail semantic comparability become Level-0 diagnostics regardless of Gate 1 pass.

#### 2.5.4 Summary: five-tier status taxonomy

This replaces the informal "primitive promotion" talk with explicit status. Codex Round 2 flagged that the previous label "four-tier" was inconsistent with five rows — corrected here.

Every promoted coordinate additionally carries a **scope label** `(modality, stimulus-family, pooling, tokenizer)` (e.g. `(text, c4_clean_5k_v1, seq_mean, llama-family-tokenizer)`) so that a claim is never stronger than its tested scope. A coordinate only loses a scope qualifier when extended and re-validated.

| Status | Gate tests passed | What you can say | What you cannot say |
|---|---|---|---|
| **Untested (⚫)** | — | — | — |
| **Diagnostic (Level-0, ⚪)** | class-local, fails semantic comparability, or fails G1.2–G1.7 on some classes | "In class C under scope S, this primitive behaves like X" | "This primitive measures the same thing across classes" |
| **Coordinate / portability-passed (🟡)** | Gate 1 (G1.1–G1.7) on ≥3 classes at scope S + negative control | "This primitive is a semantically-comparable measurement in these ≥3 classes under scope S; its values are class-dependent" | "There is a universal law" |
| **Coordinate / Level-2 family-local (🟢²)** | Gate 1 on ≥5 classes AND hierarchical model comparison (§2.5.6) prefers family-specific fit over single universal fit | "There is a family-specific functional form under scope S" | "A single functional form holds across families" |
| **Coordinate / Level-1 universal (🟢¹)** | Gate 2 all 5 criteria pass AND hierarchical model comparison prefers universal functional form over family-specific | "This functional form is universal across tested trained-NN classes under declared scope S, with causal test and biology instantiation" | Nothing more — science is never final, scope can always widen |

#### 2.5.6 Noise-calibrated decision rule (Round 2 priority-directive; Round-3-revised)

**Round 3 criticism addressed:** `|z|<c` is a fail-to-reject rule — a high-variance primitive passes by being too noisy to distinguish from zero (Codex Round 3 NEW kill shot #1). Replacement: **equivalence / precision gating**, plus explicit K enumeration (NEW kill shot #2).

##### 2.5.6a. The generic Gate-1 stability test — EQUIVALENCE + PRECISION

Every Gate-1 stability criterion (G1.2 invariance, G1.3 stimulus resample, G1.4 estimator variant, G1.5 quantization, G1.6 subsample asymptote) is a test of the form: **is the effect of the nuisance factor *small enough that we can treat the two measurements as equivalent* AND *precise enough that the test is informative*?**

Let `f_A` and `f_B` be two measurements of primitive `f` differing only in a nuisance factor `N`. Define:
- `Δ = f_A − f_B` (point estimate of the effect)
- `SE(Δ)` (standard error of the difference)
- `δ` = a pre-registered **equivalence margin** (relative to the primitive's scale — e.g., `δ = 0.10 × median(|f|)`)

The stability criterion (TOST-style equivalence):

```
|Δ| + c · SE(Δ) < δ
```

Both summands must be small simultaneously. This forces:
- **Effect-size smallness** (`|Δ|` near zero)
- **Precision** (`c · SE(Δ)` — the uncertainty band — must itself fit under δ; otherwise the test is uninformative and the primitive fails Gate 1)

A high-variance primitive CANNOT pass by being noisy: if `SE(Δ)` is large, `c · SE(Δ) > δ` even when `|Δ| = 0`, so the criterion fails. The precision loophole is closed.

`c` is Bonferroni-corrected for the family-wise error rate `α_FWER = 0.05` across K independent tests in the prereg. Since this is a one-sided equivalence test:

```
c = z_{1 − α_FWER / K}
```

(one-sided, not two-sided, because we're only asking "is the upper bound of |Δ| below δ?")

##### 2.5.6b. K enumeration — OPERATIONAL DEFINITION (Round 3 NEW kill shot #2 closure)

**Rule:** K is the number of independent pass/fail decisions in the prereg. Each decision corresponds to ONE (criterion, system) pair. Per-layer curves, per-pooling variants, per-quantization points, and per-resample pairs are **aggregated into a single per-system pass/fail decision per criterion** using the following rule:

- **G1.2 rotation invariance:** aggregate across all (layer, pooling) → 1 decision per (system, criterion). The aggregate statistic is the max over the grid of `|Δ(ℓ, pool)|`; the SE is the max SE over the grid. Passing the aggregate = passing worst-case.
- **G1.3 stimulus resample:** aggregate across 3 resample pairs × (layer, pooling) — same max rule.
- **G1.4 estimator variant:** aggregate across (layer, pooling, quant).
- **G1.5 quantization:** aggregate across (layer, pooling).
- **G1.6 subsample asymptote:** 1 decision per system (slope across n-sweep must be within 1 SE of zero).
- **Negative control** (trained vs untrained): 1 decision per system (must be non-equivalent at δ_neg-control).

Per-system criteria count: 5 stability + 1 negative control = **6 decisions per system**.

For the Batch-1 prereg §3.7 (3 systems × 6 decisions) → **K = 18**.

Correction: `c = z_{1 − 0.05/18} = z_{0.9972} ≈ 2.77`.

Each test failing → primitive fails Gate 1 on that system. Gate 1 portability (≥3 systems pass) → Coordinate 🟡. Testing per-layer curves separately would be a different prereg (Level-2-within-system) — not part of Gate 1's portability promotion.

##### 2.5.6c. Equivalence-margin δ — PRE-REGISTRATION RULE

δ cannot be post-hoc. Pre-register:
- **δ_relative:** default 0.10 (i.e., 10% of the primitive's median value on the test cloud). Used for G1.2/G1.3/G1.4/G1.5.
- **δ_slope:** default 0.05 (absolute, applied to the G1.6 asymptote slope in log-log space).
- **δ_neg-control:** default 0.20 (relative; a trained-vs-untrained difference of more than 20% of median passes the "measures learned geometry" negative control).

δ values are prereg'd at commit and LOCKED. A prereg is invalid if δ is chosen after seeing data.

**Sensitivity check (mandatory per prereg):** also compute pass/fail at δ = 0.05 and δ = 0.20. Report how the verdict changes. If the primitive is Gate-1 pass at δ=0.10 but fail at δ=0.05, that's a weak pass — flag in the atlas as `🟡 (δ-sensitive)`.

##### 2.5.6d. Universality decision rule (Gate 2 G2.3) — hierarchical model comparison

For a primitive's measurements across `{class × depth × stimulus_resample × pooling × quant}`:

1. **Class-specific model (Model A):** `f_{c,k,…} = g_c(k, …) + ε`. Each class has its own function.
2. **Universal model (Model B):** `f_{c,k,…} = g(θ(c), k, …) + ε`. Single universal `g` with class-specific parameters.
3. **Compare via LRT + BIC + AIC triple:**
   - LRT: `Λ = 2(log L_B − log L_A)`, reject A (prefer universal) iff `Λ > χ²_{1−α, df}` with α=0.01 for Level-1.
   - BIC: prefer Model B iff `ΔBIC = BIC_A − BIC_B > 10` (Raftery scale: "very strong" evidence).
   - AIC: agreement on preferred model required.
4. **Leave-one-class-out predictive check** (Codex Round 3 Q1c): refit Model B on C−1 classes, predict on the held-out class, report prediction error. Holdout error must be within 1.5× in-sample scatter. If not, universality is overfit — demote.
5. **Decision (all four conditions must hold):**
   - Reject A (universal preferred) + ΔBIC>10 + AIC agrees + holdout passes → Level-1 candidate pending causal (G2.4) + biology (G2.5).
   - Fail to reject A OR any single disagreement → Level-2 family-local.
   - Inconclusive at current sample → "extend n" or cite failure per §2.5.7 escalation.

##### 2.5.6f. Sample-size-efficient SE estimation — COMPUTE FIX (Round 3 NEW kill shot #2)

Codex Round 3 §1b: bootstrap plan of `20 × 0.20 × 756 ≈ 3000 evals × 5s = 4.2h` is **out of envelope**. Fix per Codex Round 3 §8 (store sufficient statistics + analytical SE):

- **TwoNN ID analytical SE.** Under the TwoNN asymptotic distribution, `SE(d_hat) = d_hat / √n` (from Pareto MLE asymptotics). For n=5000, `SE ≈ d/70 ≈ 0.014 d` — tight; no bootstrap needed. Just store `log μ_i` samples per (system, layer) → ~40 KB per measurement.
- **PR delta method or Hutchinson trace estimator.** PR = `(Σλ)²/Σλ²` — delta method on eigenvalue sums gives `Var(PR)` in terms of `Σλ, Σλ², Σλ³, Σλ⁴`. Store the top-k eigenvalues (k=100) per measurement → ~800 bytes.
- **Clustering coefficient SE (new P1.3).** Standard result: variance of clustering coefficient on a sparse graph is `O(1/n)`. Store the per-point clustering values → ~40 KB per measurement.
- **Bootstrap RESERVED for primitives without analytical SE** (Batch-2 Ricci, Koopman, persistent homology).

**Revised budget:**
- Primitive computation: ~60 min for ID, ~10 min PR, ~15 min clustering = ~85 min.
- Sufficient-statistic dumping: ~1 min.
- Analytical SE computation: ~1 min.
- Total compute after activations: < 2 h. **Fits the envelope.**

Activation extraction (Exp A) remains the wall-clock constraint at ~3.5 h; Exp B (primitive + SE + Gate-1 checks) is now ~2 h; Exp C (hierarchical fit + normalization) is ~0.5 h. **Compliant.**

##### 2.5.6g. Defaults — DERIVED, not picked

- Gate 1 per-test: `|Δ| + 2.77·SE < δ_relative=0.10` (K=18, α_FWER=0.05 one-sided).
- Gate 2 universality: LRT α=0.01 + ΔBIC > 10 + AIC agreement + leave-one-class-out holdout error within in-sample scatter.
- H12 stimulus-dominance: variance-component model, `σ²_stimulus > σ²_system` rejects atlas well-posedness at current scope (pivots to cUniv per §2.5.7).
- H14 subsample asymptote: slope `log|f(n) − f(n_max)| vs log n` within 1 SE of zero at n=n_max/2 vs n_max.
- H13 quantization: same equivalence criterion as G1.5 with δ=0.10.
- Sensitivity check: report at δ ∈ {0.05, 0.10, 0.20} in every ledger entry.

All defaults are derivable from the prereg's declared α + K + δ structure. **Removing arbitrariness requires that choice of (α, K, δ) be justified once in the prereg — and Round 3 flagged that even α is picked by convention, not cost.** Acceptable compromise: α=0.05 FWER + α=0.01 for Level-1 promotion come from Bayesian/frequentist convention; prereg's "cost-rationale" section states why these are chosen for this project (false-pass > false-fail cost since an atlas row is hard to retract once published).

##### 2.5.6b. The universality decision rule

Gate 2 G2.3 previously used `RSS_joint / RSS_per_class < α_universal = 1.5`. This is an ad-hoc ratio with no distribution under the null.

**Replacement: hierarchical / mixed-effects model comparison.** For a primitive's measurements across `{class × depth × stimulus_resample × pooling × quant}`:

1. Fit a **class-specific model** (Model A):
   ```
   f_{c,k,s,p,q} = g_c(k, s, p, q) + ε_{c,k,s,p,q}
   ```
   where `g_c` is a class-specific function (one set of parameters per class).

2. Fit a **universal functional-form model** (Model B):
   ```
   f_{c,k,s,p,q} = g(θ(c), k, s, p, q) + ε_{c,k,s,p,q}
   ```
   where `g` is a single universal function with class-specific parameters `θ(c)`.

3. Compare via **AIC / BIC / likelihood ratio test**.
   - **LRT**: `Λ = 2(log L_B − log L_A)`, reject Model A (prefer universal) iff `Λ > χ²_{1−α, df}`.
   - **BIC**: prefer whichever model has lower BIC by ≥ 10 (Raftery scale: "very strong" evidence).
   - **AIC / WAIC**: for non-nested or out-of-sample comparisons.

4. **Decision** (all three must agree for a promotion, else keep status 🟡 portability-only):
   - If LRT rejects A + ΔBIC > 10 in favor of B + AIC agrees → Level-1 universal
   - If LRT fails to reject A + ΔBIC < 10 → family-local (Level-2)
   - If mixed → pre-reg says "inconclusive at current sample budget; extend n or cite failure."

**This replaces** fixed α_universal = 1.5. Pre-registered false-positive rate (α for the LRT) is 0.01 for Level-1 claims (stricter than Level-2's 0.05) because Level-1 is the stronger claim.

##### 2.5.6c. Sample-size-efficient SE estimation

Codex Round 2 §7 warned that "100-bootstrap CIs × 756 measurements likely blows the 4h budget." Mitigation:

- **Primary SE estimator:** variance across stimulus resamples (3 disjoint draws = 3 independent point estimates; SE = std / √3). Cheap; no additional compute per measurement.
- **Secondary (subset only):** bootstrap on a 20% random subset of measurements (n_b = 20 bootstraps rather than 100). Use only to verify primary SE agreement.
- **Total bootstrap budget:** 20 × 0.20 × 756 ≈ 3000 evaluations (rather than 100 × 756 = 75600). Stays well inside envelope.

##### 2.5.6d. Defaults now DERIVED (replace hand-picked τ's)

- Gate 1 per-test criterion: `|z| < 2.81` (FWER α = 0.05, K = 10 tests Bonferroni-corrected).
- Gate 2 universality: LRT α = 0.01 + ΔBIC > 10 + AIC agreement.
- H12 stimulus-dominance threshold: `σ²_stimulus > σ²_system` rejects atlas well-posedness. Threshold derived from variance-component model, not a hand-picked ratio.
- H14 subsample asymptote: primitive's slope in `log(|f(n) − f(∞̂)|) vs log(n)` is within 1 SE of zero at the maximum n tested.

All defaults are pre-registerable and can be tightened in specific preregs. The key property: **defaults come from a statistical model, not a convention.**

#### 2.5.7 Conditional universality — Alt C definition (Round 2 closure)

**Codex Round 2 §1, §Q1 flagged** that "conditional universality (Alt C) is referenced but not defined." Defining it now.

**Claim structure.** A universality claim always names a stimulus distribution `D`. The default ("unconditional" universality) asserts the functional form holds for all natural stimulus distributions in a broad class. **Conditional universality** relaxes this: the claim holds only within a specified stimulus family.

Formally, conditional universality (cUniv) is a Level-1 or Level-2 claim qualified by a stimulus-family specification:

```
(cUniv)    f(m, x) = g(θ(m), x)   for all x ~ D ∈ ℱ
```

**ℱ is a formal object (Round-3 closure per Codex Q2).** ℱ is not a prose description; it is a machine-checkable tuple:

```
ℱ = (
    scope_id: str,              # e.g. "text.c4_clean.len256.v1"
    generator: Callable,        # deterministic sampling code with seed
    dataset_hash: str,          # sha256 of the source dataset used to seed
    filter: Callable,           # predicate on raw examples (language, length, format)
    length_law: Distribution,   # the distribution of stimulus length in tokens
    invariances: List[Transform], # transformations that keep x "in the same family"
                                  # e.g. permutation of sentence order within a doc,
                                  # case normalization, whitespace normalization
    invariance_check: Callable  # takes two stimuli, returns True iff in same ℱ
)
```

A primitive's measurement `f(m, x)` is claimed conditional on `ℱ` iff:
1. x was sampled via `ℱ.generator(seed)` with the declared seed — the sample is machine-reproducible.
2. x passes `ℱ.filter`.
3. `f(m, x)` is stable under any `τ ∈ ℱ.invariances` applied to x (tested under the G1.3 stimulus-resample stability; resamples from `ℱ.generator` + samples from `τ(x)` are both valid stimuli).

The Batch-1 prereg §3.7 locks its ℱ with `scope_id = "text.c4_clean.len256.v1_seeds42_123_456"` and `invariances = [whitespace_norm, case_norm]`. Tokenization mismatch across models is NOT declared as an invariance (tokenizers produce different token sequences from the same raw text) — this is treated as a **conditioning variable** carried in the scope label, not an invariance.

No prose re-interpretation of ℱ post-lock. Extending to ℱ_2 requires a NEW prereg with a new scope_id and explicit enumeration of which invariances or filters changed.

**Escalation rule:** A primitive that passes Gate 2 within `ℱ_1` is **cUniv on ℱ_1** but NOT unconditional. To extend to ℱ_2, re-run Gate 2 with prereg-declared `ℱ_2`. To generalize ("unconditional"), re-run Gate 2 on a sequence of ℱ's that span the scope claim (e.g., text → image → audio → synthetic distributions).

**Pivot rule:** If probe P1.4 returns "stimulus dominates" (σ²_stimulus > σ²_system), the atlas direction is not falsified — it **pivots to cUniv-by-construction**. In that scenario every claim must ship with its `ℱ` declaration, and scope generalization becomes part of the research program, not a hidden assumption.

**What this changes for Batch 1.** The Batch-1 prereg (§3.7) explicitly declares its ℱ as `ℱ_1 = {C4-clean, length-256 tokens, natural-language English}`. Any Gate-1 pass from Batch 1 is conditional on ℱ_1 until extended. Modality-scope labels (§2.5.4) are the operational encoding of this conditionality on each coordinate row in the atlas.

**What this changes for the atlas taxonomy.** Status 🟢¹ (Level-1 universal) can now mean:
- `🟢¹(unconditional)` — Gate 2 passed across ≥ 2 broad stimulus families
- `🟢¹(cUniv, ℱ)` — Gate 2 passed only within ℱ

Both are valuable; the atlas makes the distinction explicit rather than pretending unconditional universality when only cUniv was demonstrated.

#### 2.5.8 Governance rule — modality-scope (reclassified from H15 per Codex Round 3 Q6)

This is a **policy**, not a hypothesis. It does not have a kill criterion — it shapes the scope of any claim that may be made from a probe result.

**Rule:** A primitive that passes Gate 1 on a set of classes within a single modality (e.g., language: transformer + SSM + hybrid) is scope-labeled **(modality-local)**. It is a coordinate only within the tested modality. It is NOT eligible for Gate 2 universality promotion until at least one class from a different modality (vision encoder or diffusion) has been tested. "Cross-modal replication" means: same Gate-1-passing primitive, re-run on a model from a different modality, with its scope label updated.

**Enforcement:** At commit time, a prereg whose claim set exceeds its tested modality is rejected by the Cross-System Auditor (Persona 9 per `CLAUDE.md §7.4`). Modality-scope is part of the scope label `(modality, stimulus-family, pooling, tokenizer)` in every coordinate row.

**Operational trigger:** If the first cross-modal extension (Batch-5 DINOv2 or similar) produces a Gate-1 FAIL for a primitive that passed on all three language classes, that primitive is relabeled **text-only** and its promotion ceiling is Level-2-text-family-local.

#### 2.5.5 Pre-registration template (LOCKED at commit)

Every new coordinate promotion requires a pre-reg at `research/prereg/genome_<primitive>_<scope>_YYYY-MM-DD.md` with these fields LOCKED:

1. Primitive name + mathematical definition
2. Supported classes targeted in this prereg
3. Invariance group `G_f` + invariance check protocol
4. Stimulus distribution `D` + resampling protocol
5. Noise-calibrated decision rule: equivalence margin `δ`, family-wise α_FWER + K enumeration (§2.5.6b), Gate-2 hierarchical comparison parameters (§2.5.6d)
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
  7. If all Gate-1 checks pass, the primitive is portable (🟡) for Batch-1 scope. A secondary Level-2 joint fit is optional and deferred — per §2.5.6d, universality is decided by the hierarchical LRT+BIC+AIC+holdout rule, not an F-test on RSS ratios. Level-1 requires a separate Gate-2 prereg with G2.2 derivation, G2.4 causal test, and G2.5 biology — none of which is in scope for this Gate-1 prereg.
- **Interpretation (per §2.5):** PASS-Gate-1 on class C iff G1.1–G1.5 + negative-control all pass on C. PASS-Level-2 iff Gate 1 on ≥3 classes AND joint-fit RSS / per-class RSS < 1.5 AND g monotonic. PASS-Level-1 needs Gate 2 (separate batch; not claimed from Batch 1 alone).
- **Cost:** TwoNN ~5 s per measurement. 756 × 5 s ≈ 63 min + boot. Activation extraction ~3 h (dominant). **Total: ~4 h wall-clock, fits one experiment.**

**P1.2 — Participation ratio (Gate-1 promotion target)**

- **Question:** Does PR pass Gate 1 on all three language classes? Does PR(k) admit a class-independent functional form?
- **Hypothesis:** Gate 1 passes on all three (covariance statistics are nearly universal). PR(k) will likely show a compression-expansion shape similar to ID but with different sensitivity to pooling.
- **Counter-hypothesis:** Gate 1 fails on one or more classes due to pooling sensitivity (H12 variance decomposition will reveal this).
- **MVE:** Same point clouds as P1.1. Compute PR = `(Σ λ_i)² / Σ λ_i²` where λ_i are covariance eigenvalues. Estimator variants: centered vs uncentered. All G1.1–G1.5 checks + negative control. Joint fit test for Level-2.
- **Cost:** PR is O(D³) eigendecomp — fast at D ≤ 1024. All measurements ≈ 5 min total. Shares activations with P1.1.

**P1.3 — kNN-5 graph clustering coefficient (Gate-1 promotion target; SINGLE-CLOUD redefinition per Codex Round 3 Q4 + NEW kill shot #3)**

- **Round 3 correction.** The Round-2 draft defined P1.3 as "mean per-point kNN-Jaccard across resamples." Codex Round 3 §Q4: that definition **conflates the coordinate with its G1.3 stability diagnostic** and is underspecified (Jaccard across DIFFERENT clouds lacks shared anchor points). Redefined here as a single-cloud scalar per Codex recommendation.
- **Primitive.** For a point cloud X, build the k-nearest-neighbor graph (k=5, Euclidean). The **local clustering coefficient** `C(i)` for point `i` is the fraction of pairs among `i`'s k neighbors that are themselves neighbors (edges in the kNN graph). The coordinate `C(X) = mean_i C(i)` — one scalar per point cloud.
- **Why this primitive:** single-cloud, class-agnostic, cheap (O(n k²)), measures local manifold density independent of global structure, directly tests Codex Intuition 2 ("local neighborhood structure survives cross-architecture"). Invariant to orthogonal rotations of the embedding (kNN distances unchanged) — clean G1.2 pass.
- **Question:** Does mean kNN-5 clustering coefficient pass Gate 1 (all of G1.1–G1.7) on all three language classes?
- **Hypothesis (supports H4):** Yes. Clustering coefficient C(k_normalized) is stable under resampling (G1.3), asymptotes in n (G1.6), and differs between trained and untrained controls (negative control). Its depth curve C(ℓ/L) may or may not factor universally across classes (that's a Level-2 question for Gate 2, not Gate 1).
- **Counter-hypothesis:** C(i) is too local-graph-topology-dependent — small changes in k break it; distance-metric choice dominates; fails G1.4 estimator variant.
- **MVE:** Compute C(X) at every ℓ/L ∈ {0.0, 0.05, …, 1.0} × 3 systems × 2 pooling × 3 stimulus resamples × 2 quantizations. Estimator variants: (a) weighted clustering coefficient (edge-weighted by 1/distance), (b) unweighted. Store per-point C(i) as sufficient statistic → analytical SE via O(1/n) variance of the mean.
- **Gate-1 checks:** G1.2 (random rotation → identical clustering, tight tolerance), G1.3 (resample stability via equivalence criterion §2.5.6a), G1.4 (weighted vs unweighted within δ=0.10), G1.5 (FP16 vs Q8 within δ=0.10), G1.6 (n-sweep asymptote), plus trained vs untrained negative control.
- **Interpretation (per §2.5 + §2.5.6):** PASS-Gate-1 per class iff all seven criteria pass the equivalence criterion at δ=0.10. Cross-class portable iff ≥3 language classes pass. **A coordinate-level measurement, not a stability measurement** — stability is tested by G1.3-G1.6, not part of the primitive definition.
- **Cost:** kNN build O(n log n), clustering O(n k²). ~5 s per (system, layer). Negligible on top of P1.1/P1.2 activations.

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

### 3g. Prereg strawman — `genome_id_portability_2026-04-20` (NOT YET LOCKED)

Demonstrates the §2.5.5 template in practice for the Batch-1 lead probe P1.1. STRAWMAN ONLY — will be reviewed by Codex Round 2 and locked at commit after approval. Current location: this section. At lock, migrates to `research/prereg/genome_id_portability_2026-04-20.md` and becomes immutable.

**1. Primitive** — Intrinsic dimension via TwoNN (Facco et al. 2017). For a point cloud `X ∈ R^{n×d}`, compute for each point its nearest and second-nearest neighbors with distances `r_1, r_2`. The ratio `μ = r_2 / r_1` is Pareto-distributed with scale parameter d (the intrinsic dimension). Estimate d via MLE on `log μ`.

Mathematical definition (class-agnostic, §2.5.3 naming rule): "the intrinsic dimension of the data manifold sampled by X, computed as the scale parameter of the nearest-two-neighbor ratio distribution."

**2. Supported classes** — Class 1 (autoregressive LLM), Class 3 (SSM), Class 4 (hybrid), via the three models in §3b. Controls: untrained Class 1.

**3. Invariance group G_f** — f is invariant to (a) isometric transformations of X (orthogonal rotations + translations), (b) global isotropic rescaling. f is NOT invariant to (c) token permutation within a sequence, (d) stimulus resampling. (c) is mitigated by seq-mean pooling; (d) is tested in G1.3.

**G1.2 check:** apply a random d×d orthogonal Q to each X_k; verify `|ID(Q·X_k) − ID(X_k)| / ID(X_k) < τ_G1.2 = 0.02`. 10 random Q per (system, layer, pooling).

**4. Stimulus distribution D** — 5000 sentences × 256 tokens from a clean C4 / Wikipedia slice. Three seed-disjoint resamples (seeds 42, 123, 456). The invariance class of D: "natural-language text from the c4_clean distribution; length 256 tokens; no task conditioning."

**G1.3 check:** verify `max_{i,j in {42,123,456}} |ID_i − ID_j| / mean(ID) < τ_resample = 0.10`.

**5. Tolerances — NOISE-CALIBRATED (per §2.5.6, replacing fixed τ's from Round 1):**

All stability gates use `|z| = |f_A − f_B| / SE(f_A − f_B) < c` where c is Bonferroni-corrected for family-wise error rate α_FWER = 0.05 across K = 10 Gate-1 tests in this prereg → `c = 2.81`.

- G1.2 (invariance): `|z_rotation| < 2.81` with SE from 10 random-Q resamples on the same cloud.
- G1.3 (stimulus resample): `|z_resample| < 2.81` pairwise across the 3 stimulus resamples, SE from std / √3.
- G1.4 (estimator variant): `|z_TwoNN vs MLE| < 2.81` with SE from bootstrap on 20% subset (n_b=20).
- G1.5 (quantization): `|z_FP16 vs Q8| < 2.81` same SE source.
- G1.6 (subsample asymptote): n-sweep `n ∈ {500, 1000, 2000, 5000}`; slope of `log|f(n) − f(5000)|` vs `log n` must be within 1 SE of zero at n=2000 vs n=5000.

For optional Level-2 fit: model comparison via LRT (α=0.05) + BIC (ΔBIC > 10 prefers richer) + AIC agreement — NOT a fixed RSS ratio. See §2.5.6b.

Justification: all decisions come from a statistical model under the null, not hand-picked constants. Codex Round 2 priority directive.

**6. Estimator variants** — (a) TwoNN Facco et al. 2017, (b) MLE Levina-Bickel 2004. Both computed on same point clouds; `|d_TwoNN − d_MLE| / mean < τ_estimator` required per class.

**7. Quantization ladder points** — FP16 and Q8 for each of the three models. Q8 via `bitsandbytes` 8-bit quantization at inference. ID computed on both; `|d_FP16 − d_Q8| / d_FP16 < τ_quant` required per class.

**8. Promotion target** — Gate 1 (portability) on all three language classes. If passed, optionally test Level-2 joint fit as secondary analysis. Level-1 claim is NOT within this prereg's scope; requires subsequent prereg with derivation + causal + biology.

**9. Derivation (Gate-2 placeholder, not within prereg scope)** — N/A for Gate-1 prereg. A future Level-1 prereg would need a first-principles functional form for d(k_normalized) from information theory or statistical mechanics.

**9a. Biology instantiation (Codex Round 3 Q5 correction — stimulus-indexed, not time-indexed).** Given a neural population recording `N_neurons × T_timepoints` under a set of stimulus conditions `{s_i}`:
- **Point identity:** each point `x_i ∈ R^{N_neurons}` is the **population response vector for stimulus condition `s_i`** — specifically, the trial-averaged z-scored firing rate (for Neuropixels) or BOLD activation (for fMRI) over a stimulus-locked window `w(s_i)`. Points index stimuli, not time.
  - Rationale (Codex Q5): time-bin-indexed points measure temporal-autocorrelation geometry, not stimulus representation geometry. The atlas coordinates are about *how the system encodes different stimuli*, which is the stimulus-indexed formulation (consistent with Gao & Ganguli 2015 for PR on neural recordings).
- **Window `w(s_i)`:** stimulus-locked — e.g., frame-locked for natural-movie stimuli (~33 ms), trial-locked for fixed-duration stimuli, or block-averaged for slow-event fMRI. Prereg'd.
- **Preprocessing invariance (G1.7):** z-scoring per-neuron across stimuli declared as part of primitive identity. Different preprocessing = different primitive.
- **Sensitivity (required per prereg):** Also compute at alternative binning widths (e.g., 10/25/50 ms for spike rate, 1/2 TR for fMRI) and report verdict consistency.
- **Preregistered pitfalls:**
  - Low neuron count in a recording may cause `n_stimuli × n_neurons < asymptote` (G1.6 fails on biology). Use Allen V1 datasets with N_stimuli ≥ 100 + N_neurons ≥ 200 for first validations.
  - For fMRI, voxel count (~10⁵) is large but trial count is small; use multi-session aggregation for stimulus diversity.

**9b. Modality-scope label (Codex Round 2 §4, §11 + §2.5.4).** This prereg's claim is scope-labeled:
`(modality=text, stimulus_family=c4_clean_5k_v1, pooling=seq_mean_and_per_token_subsample, tokenizer=per-model-native)`. Any Gate-1 pass is conditional on this scope until cross-modal re-validation.

**10. Kill criterion** —
- **Primary (primitive-level):** ID fails Gate 1 on ≥ 1 of 3 classes. In that case ID is not a coordinate for the atlas; the probe output is a NEGATIVE RESULT that narrows the primitive search space.
- **Secondary (H1 universality):** Under §2.5.6d hierarchical comparison (LRT α=0.01 + ΔBIC>10 + AIC + leave-one-class-out), the universal-form model fails any of the four checks → Level-1 refuted; ID may still be Level-2 family-local.
- **Tertiary (negative control):** ID_trained ≈ ID_untrained within stimulus-resample scatter. Refutes the "ID measures learned geometry" interpretation; demotes ID to Level-0 diagnostic.

**11. COMPUTE.md §9 compliance checklist:**
- [x] Max VRAM ≤ 22 GB — peak ~3 GB (three models FP16 concurrent, or one-at-a-time Q8)
- [x] Max RAM ≤ 56 GB — peak ~12 GB (activation buffers for per-token variant)
- [x] Wall-clock ≤ 4 h — 3-experiment split (Exp A extraction ~3.5 h, Exp B primitive compute ~2 h, Exp C stats ~0.5 h) — each ≤ 4 h
- [x] Disk footprint — ~800 MB activations in `.gitignore` path
- [x] Quantization logged — FP16 + Q8 per model, logged per ledger entry
- [x] Save-resume path verified on smoke test (pre-reg blocker — smoke test must pass before full run)

**12. Sign-off (LOCKED AT COMMIT)** — pending Codex Round 2 approval. Upon approval, commit with message `Lock prereg: genome_id_portability_2026-04-20` and move to `research/prereg/`. Post-lock modifications invalidate the experiment.

---

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
| 2 | 2026-04-21 | `.codex/outputs/round2.md` | KS2 CLOSED (point-cloud contract); KS1/KS3 PARTIAL. Arbitrary τ/α thresholds dominate gate outcomes more than science. Missing G1.6 subsample stability. Alt C referenced but not defined. 4-tier taxonomy has 5 rows. Spectral slope fragile — swap for local-neighborhood. Batch-1 extraction/uncertainty plan can silently go out-of-envelope. Biology instantiations ceremonial without concrete per-primitive specs. | Derive/pre-register noise-calibrated tolerance + universality model-comparison rule. | 3, 3, 3, 3, 3, 4, 3 | 8.3/10 (trajectory +0.3) |
| 3 | 2026-04-21 | `.codex/outputs/round3.md` | KS2 CLOSED; KS1/KS3 PARTIAL. NEW kill shots: (1) precision loophole (`|z|<c` passes noisy primitives), (2) K ambiguity (undercounted grid), (3) P1.3 conflates coordinate with stability diagnostic. Compute arithmetic fails: bootstrap plan 4.2h OOE. Biology instantiation mis-framed (time-indexed, should be stimulus-indexed). §2.5 internal drift (Gate 1 count, old RSS text). ℱ informal. | Lock Gate-1 into prereg-grade decision procedure: K enumeration + equivalence/precision criterion + P1.3 single-cloud redefinition | 4, 4, 3, 3, 4, 3, 4 | 8.4/10 (trajectory +0.1) |
| 4 | queued | `.codex/outputs/round4.md` | — | — | — | — |

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
| T+1.2h (Round 2 running) | ON TRACK | No | Round 2 fired in background (task `b3fwyis5j`); parallel work: researched local-neighborhood primitives (Codex Intuition 2) — added §B8d brief distinguishing NNGS (cross-system diagnostic) from local-connectivity statistics (per-system Gate-1 candidates); drafted prereg strawman §3.7 for ID Gate-1 test (demonstrates §2.5.5 template) | Wait for Round 2 output; while waiting, continue entropy sweep + possibly smoke-test prep for P1.1 once prereg approved |
| T+1.5h (cron heartbeat fire) | ON TRACK | No | Round 2 still running (~10 min, within 5-15 min typical); GPU/CPU idle (Codex is remote); repo clean; anti-entropy sweep next on WIKI/MEASUREMENT_PRIMITIVES/SYSTEM_BESTIARY/OPEN_MYSTERIES — session has advanced (H11-H13, §2.5 two-gate, revised Batch 1) but canonical docs haven't been patched to match | Sweep and patch canonical research docs for consistency with revised session state |
| T+1.8h (sweep complete) | ON TRACK | Minor — Round 2 now ~20 min, past typical 5-15 min window but process count stable at 4 (not stuck; longer prompt + more targeted questions vs Round 1) | Anti-entropy sweep committed: MEASUREMENT_PRIMITIVES.md status legend updated to four-tier, Koopman (§2.4) + local-connectivity (§2.5) added, CKA demoted ⚪, §9 gate replaced with cross-reference to locked §2.5. WIKI §1 project-state glance + §3 primitives table patched. Model registry verified: all three Batch-1 anchors (Qwen3-0.6B, mamba2-370m, Falcon-H1-0.5B) present in `../../models/MODEL_DIRECTORY.md`. | Wait for Round 2 output; if it lands, process + fire Round 3; if heartbeat fires before Round 2 returns, do additional research on the deferred-to-Batch-2 primitives (Koopman observable-family and DMD estimator variants) so Batch 2 can move fast once Batch 1 lands. |
| T+2h (Round 2 landed, 8.3/10) | ON TRACK | No | Processed Round 2: KS2 CLOSED, KS1/KS3 PARTIAL. Derived §2.5.6 noise-calibrated decision rule + §2.5.7 conditional universality (Alt C) — priority directive deliverable. Fixed entropy bugs (5-row taxonomy, P1.5 typo, Alt C undefined). Added G1.6 subsample stability, G1.7 preprocessing/metric declaration, H14/H15. Swapped spectral slope → local-neighborhood kNN-Jaccard (P1.3) per Codex Intuition 2. Added biology instantiations to §3.7 prereg + modality-scope labels. | Fire Round 3 fresh session; while waiting, compress §1b research brief per parsimony mandate (Codex §11) — most not load-bearing for next probe |
| T+2.3h (cron heartbeat fire, Round 3 running) | ON TRACK | Minor — session doc at 1019 lines (S6 entropy risk); Round 3 running ~10 min (in typical window) | Parsimony compression of §1b in progress — keep only Batch-1-load-bearing content (B2 Platonic critique, B3 geometric primitives, B8c-d Ricci-null + local-neighborhood), compress B1/B4/B6/B7 to pointers | Complete §1b compression; commit; continue waiting for Round 3 |
| T+2.8h (Round 3 landed, 8.4/10) | ON TRACK | No | Round 3 score +0.1 (trajectory capped by stats loopholes + drift). Processed: (1) §2.5.6 rewritten with equivalence/precision criterion `|Δ|+c·SE<δ` replacing `|z|<c`, K enumerated to 18 with per-system aggregation rule, analytical SE for TwoNN (d/√n) + sufficient-statistic storage fixing compute arithmetic; (2) P1.3 redefined as single-cloud kNN-5 clustering coefficient (not Jaccard across resampled clouds); (3) §2.5 drift fixes — Gate 1 "all seven criteria", taxonomy G1.1-G1.7, Gate 2 G2.3 hierarchical-only (RSS/α_universal purged); (4) ℱ formalized as generator+hash+invariances+scope_id machine-checkable object; (5) biology §9a corrected — stimulus-indexed points not time-indexed (Gao&Ganguli framing); (6) H15 retired from H-register → §2.5.8 governance rule. | Fire Codex Round 4 fresh session; during wait, propagate changes to WIKI + MEASUREMENT_PRIMITIVES + verify no residual legacy text |

---

*End of session working doc. Becomes `research/BLUEPRINT.md` at convergence.*
