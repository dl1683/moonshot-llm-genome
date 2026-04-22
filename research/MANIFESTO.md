# MANIFESTO — Why a Neural Genome

*Intellectual framing for the moonshot. Reread at every phase transition.*

---

## 0. The Competitive Reality

**We are one independent researcher. DeepMind, Anthropic, OpenAI, Google, Meta publish representational-geometry papers monthly with more compute, more authors, and more review cycles than we can match.** "We measured X across N architectures, pre-registered it, it mostly held up" — that is the baseline product of those labs. If our output looks like that, we are invisible.

**We win by doing exactly what those labs cannot or will not publish:**
1. **First-principles derivation**, not phenomenology. If we can predict an observed invariant from information-theoretic or geometric axioms before fitting, we have a Nature-grade claim. If we can only fit it, we have a workshop paper nobody reads.
2. **Findings that contradict "bigger model = better"**, because that is the product of every big lab. The manifesto — *intelligence is geometry, not scale* — is structurally hostile to the business model of scaling labs. They cannot push this agenda. We can.
3. **Electricity-grade efficiency** on a real task. Not "this signal correlates with quantization cost" — actually *training* a model at 10× less compute using geometry-derived tools, matching baseline capability. That is the manifesto cashing in. That is what Weka / Furiosa / Martian / VERSES will care about.
4. **Pre-registered falsification discipline**. Big labs cannot write papers this way — their output is the capability, not the epistemic process. We can, and this is a structural differentiator for serious reviewers.

**The default answer to any marginal improvement — another model row, tighter error bars, one more figure — is NO**, unless it directly enables (1)–(4). Paper polish that makes us look like a smaller DeepMind is the failure mode.

Concretely, as of 2026-04-21, the breakthrough-aligned directions are:

- **Derive why `c = p × d_rd = 2.07` for text and `3.18` for vision** from information theory. If we show these are not free parameters but predicted from stimulus intrinsic-dimension or rate-distortion axioms, we have the theoretical frame the field has been missing.
- **Train a model with geometry-as-auxiliary-loss** targeting the identified invariant and demonstrate it matches baseline at substantially less compute.
- **Transfuse trained geometry into a random-init model** surgically and observe capability emergence.
- **Derive the biology-vs-ANN equivalence from first principles**, not just show it empirically — why does mouse V1 land at the same `c` as DINOv2?

None of this is done. The current paper (8900 words, 10 tables, 4 figures) is workshop-grade respectable — useful scaffolding, not the breakthrough.

---

## 0.1. What Done Looks Like — The Three Deliverables (Codex-ratified vision, 2026-04-21)

The moonshot is not "a paper." It is three coupled artifacts whose joint landing settles whether Intelligence = Geometry:

1. **The Genome Equation.** A derived law — currently candidate `c = p · d_rd` — that predicts the invariant *a priori* from stimulus/representation geometry, holds across architectures + objectives + brains, and **predicts capability/compute**, not just fits it. The integers that land (text c=2, vision c=3 as of today) must come from derivation, not measurement.
2. **The Genome Extractor.** One command that outputs a model's or recording's genome vector + uncertainty, with locked stimulus banks and stability proofs (seed / quantization / stimulus resampling). Any lab or neuroscience group should be able to run it.
3. **The Genome Compiler.** Two causal demos that prove geometry is not descriptive but generative:
   - **Geometry transfusion**: inject genome structure into random-init weights → capability emerges without SGD.
   - **Geometry-regularized training**: ≥10× compute reduction at equal eval on a real task.

**The paradigm-shift rung is the Compiler.** Before it, geometry is "interesting survey." After it, geometry is a *controllable state variable that manufactures capability or substitutes for compute*. That is the finding DeepMind/Anthropic/OpenAI will not publish because it reframes scaling itself.

The full 4-rung ladder from current state to DONE lives in `research/CURRENT_KNOWLEDGE.md` §7–§8. Rung 1 (stress-test universality) is ~2/3 complete. Rung 2 (close derivation) has first rigorous evidence: 2/2 vision systems shift `c` toward text value when stimuli are forced to 1D structure (genome_040/041). Rung 3 (Compiler) has first transfusion attempts and **two clean nulls at the 2nd-moment level** (genome_042 covariance: geometry moves 61% toward trained, NLL unchanged; genome_043 k-means codebook: geometry overshoots, NLL unchanged). **The substrate is decisively higher-order than covariance or piecewise-constant structure.** Remaining candidates for what the substrate actually is: higher moments (skew, kurtosis), feature-direction orthogonality / tangent alignment, operator-level structure (Koopman attention patterns), or nonlinear per-token transformations preserving identity.

This narrowing is not a failure — it is precisely the kind of progress the moonshot should make. The `c = p · d_rd` invariant is *real* (genome_036 confirms on 4 text + 3 vision systems), *training-specific* (genome_038 confirms random-init breaks it), but *not sufficient for capability*. The Genome Compiler, when it lands, will need to transfer something richer than 2nd-moment geometry. This is now the explicit rung-3 problem statement.

---

## 1. The Bet

Every trained neural network — regardless of architecture, modality, or training objective — compresses some structure about its training distribution into its representations. The compression is not random: it produces geometric regularities. We bet that those regularities, properly measured and compared across systems, resolve into a small number of universal laws.

This is not a metaphor. It is a claim with a falsification condition. If the atlas grows large and reveals that LLMs, diffusion models, JEPAs, world models, and biological neural systems each have incompatible representational geometries, the bet is lost and we will say so publicly.

The bet is worth making because if it *is* true, every downstream capability — transfer, distillation, editing, steering, alignment, synthesis — rests on a shared substrate. Currently the field treats these as separate problems with separate tools. If the substrate is shared, the tools should be too.

---

## 2. Why Now

Four independent developments make this project tractable in 2026 that were not in 2022:

1. **Architectural diversity is real.** We now have competitive non-Transformer architectures in production scale: Mamba/Falcon-Mamba (pure SSM), Falcon-H1 and Granite-4.0-H (hybrid), RWKV7, Liquid LFM2, xLSTM. Diffusion LLMs (LLaDA) and JEPA-family predictors (V-JEPA 2) extend the space. "Transformer-only interpretability" has become insufficient — exactly the condition under which universality questions can be tested empirically instead of speculated about.
2. **Representation-similarity tools have matured.** CKA (Kornblith 2019), procrustes alignment, persistent homology of activations, intrinsic-dimension estimators (TwoNN), and SAEs (Bricken 2023, Templeton 2024) all work on arbitrary activation vectors. None of them is perfect; together they span enough of representation space that cross-architecture comparison is tractable.
3. **CTI proved universality is findable.** The sibling moonshot already demonstrated a universal law of representation quality across 12 NLP architectures, ViTs, CNNs, and mouse V1 neural recordings. The methodology generalizes. We inherit it.
4. **The Platonic Representations Hypothesis put this question on the map.** Huh, Cheung, Wang, Isola (2024) — "The Platonic Representation Hypothesis" — argued that representations in different neural networks are converging to a shared statistical model of reality. Our project is the operationalization of that hypothesis: convert it from a belief into a measurement program, and from a measurement program into a coordinate system.

---

## 3. Intellectual Lineage

The project draws on five distinct traditions. A healthy brief cites all of them and synthesizes.

### 3.1 Representation similarity & convergence

- Kornblith et al. 2019 — Centered Kernel Alignment as the canonical cross-model similarity measure.
- Huh et al. 2024 — Platonic Representation Hypothesis. Models converge to a shared statistical representation.
- Raghu et al. 2017 — SVCCA for comparing deep representations.
- Morcos et al. 2018 — projection-weighted CCA.

### 3.2 Mechanistic interpretability

- Olah et al. (Circuits Thread, 2020–present) — mechanistic analysis, "universality" of motifs across vision networks.
- Elhage et al. 2021 (Transformer Circuits) — mathematical framework for transformer interpretability.
- Bricken, Templeton et al. (Anthropic, 2023–2024) — sparse autoencoders for monosemantic features.
- Nanda et al. — grokking circuits, modular arithmetic.

### 3.3 Statistical mechanics of learning

- Seung, Sompolinsky, Tishby — early statistical mechanics of generalization.
- Tishby, Zaslavsky 2015 — information bottleneck for deep learning.
- Bahri et al. 2020 — statistical mechanics of deep learning.
- Zdeborová, Krzakala — replica methods and phase transitions in ML.

### 3.4 Neuroscience-AI bridges

- Yamins & DiCarlo 2016 — deep nets as models of ventral stream.
- Schrimpf et al. 2018 — BrainScore benchmark.
- Mitchell et al. 2008 (and descendants) — fMRI decoding from embeddings.
- Conwell et al. 2024 — what makes representations brain-aligned.

### 3.5 Geometric & information-theoretic foundations

- Bronstein et al. 2021 — Geometric Deep Learning.
- Amari — information geometry.
- Equitz-Cover 1991; Rimoldi 1994 — successive refinement of information (CTI's foundation).
- Shannon 1959 — rate-distortion theory.

A proper literature review for any phase-transition Codex gate must touch at least three of these traditions.

---

## 4. The Three-Tier Universality Framework (inherited from CTI)

Every claim the Genome makes is expressed in the same three-tier structure. This is not optional — it prevents the "universality collapsed because one constant varied" failure mode.

### Level 1 — Functional-Form Universality

A geometric property depends on system variables (scale, training compute, task diversity) through the **same functional shape** across every class of neural network tested.

*Example shape:* `participation_ratio(layer, model) = f(depth, task_entropy, training_compute)` where `f` has the same form for Transformers, SSMs, diffusion models, and JEPAs.

This is the strongest claim. Nobel-shaped if proven.

### Level 2 — Family Constants

The coefficients of the Level-1 shape are universal *within* a family (all autoregressive LLMs cluster around one slope; all diffusion models cluster around another). Different families have different constants, but those constants are predictable and few.

### Level 3 — Task/Data Intercepts

Additive offsets vary by task, dataset, or training distribution. Expected; these are parameters we fit, not evidence of universality failure.

**A claim is honest only if it names which level it belongs to.** "We found a universal law" without specifying whether Level 1, 2, or 3 is meaningless.

See `UNIVERSALITY_LEVELS.md` for the decision tree that assigns a claim to a level.

---

## 5. Why Cartography Before Axiom

We considered the opposite path — state a universality axiom, design experiments to falsify it, iterate. CTI did exactly this. It worked because CTI had a concrete starting derivation (EVT/Gumbel race) that specified what to measure before measuring.

Neural Genome does not yet have that derivation. The structural question ("where does knowledge live in an LLM? where does capability live? how do we even address a subspace?") is under-specified. We do not know which geometric property is the right one to derive a law about.

The atlas is the discovery instrument. By measuring many systems with many primitives, patterns reveal *which* property is the right one to formalize. Then — and only then — does derivation begin.

This sequence is:

1. **Atlas** — measure everything, tag every measurement with its universality-level candidate.
2. **Pattern** — find a property that clusters into a recognizable three-tier shape across the bestiary.
3. **Derivation** — prove the shape from first principles.
4. **Pre-registration + causal tests** — lock the hypothesis, intervene, measure.
5. **Biology** — test on neural recordings.
6. **Paradigm shift** — if all phases pass, state the axiom formally; publish.

Reversing this order produces pre-registered curve-fits that correlate with something and call it a law. That is the failure mode we are avoiding.

---

## 6. Anti-Entropy as Epistemology

The atlas will have thousands of measurements. Without ruthless pruning, signal drowns in noise — both at the file-system level (see `CLAUDE.md` §3) and at the intellectual level.

The intellectual anti-entropy discipline:

- Every atlas entry is tagged with a universality level. Entries that never promote beyond "observation" after three months are archived into an "observations that did not pattern" ledger. We publish that ledger too — honest negative space is valuable.
- Every measurement primitive that fails the architecture-agnosticism gate (§4.3 of CLAUDE.md) is explicitly demoted, not silently retained. `MEASUREMENT_PRIMITIVES.md` separates coordinates from diagnostics.
- Every open mystery that resolves moves from `OPEN_MYSTERIES.md` to the experiment ledger with a pointer to the resolving claim. Every open mystery that collects three failed resolution attempts is flagged as a *scar*: a structural feature of the atlas that the current framework cannot explain. Scars are prioritized for framework revision.

If the atlas ever becomes a dumping ground, the axiom becomes unfindable. Pruning is how the signal stays visible.

---

## 7. What Falsification Looks Like

The project has publishable value in three outcomes:

1. **Axiom proven.** A Level-1 universality law survives derivation, pre-registration, causal testing, and biological validation. Paradigm-shift paper.
2. **Axiom partially proven.** Level-2 family universality exists, but no property achieves Level-1 cross-family universality. Paper: "Representational geometry is universal within architecture families; cross-family structure is modular." Still a substantial contribution.
3. **Axiom refuted.** The atlas reveals that different neural network classes have genuinely incompatible geometric structures. Paper: "Representational Fragmentation: evidence against universal geometry in trained neural networks." This would reshape how the field thinks about transfer, distillation, and alignment — a loss for the geometry thesis, a gain for the field. Write it honestly.

The one outcome that is *not* publishable is "we ran a bunch of experiments and nothing cohered." That outcome means we failed at atlas discipline. It is prevented by the anti-entropy rules in `CLAUDE.md` §3, not by hope.

---

## 8. Why This Is a Moonshot, Not a Research Project

A standard research project asks: how do LLMs represent X? A moonshot asks: is there a universal theory of how learning systems represent anything?

Moonshot bar (from `../CLAUDE.md`):

- **Axiom questioned:** Are attention, SSMs, diffusion, and world models fundamentally different objects, or shadows of the same object? The entire interpretability field implicitly assumes the former. We are testing the latter.
- **Paradigm-shift potential:** If the axiom holds, "AI architecture" becomes a surface distinction. "AI training objective" becomes a surface distinction. The real object is a geometric structure that emerges from compression. Interpretability, transfer, editing, alignment, synthesis all become subfields of one larger field.
- **5090-demonstrable:** Every primitive must run on small models and short recordings on a single laptop. The moonshot's first demonstration should be: take a small Qwen, small V-JEPA, small diffusion net, measure one shared geometric property, show the three-tier structure. That week-long experiment is the project's MVP.
- **Novelty:** Nobody is doing this systematically. Pieces exist — Platonic Representations Hypothesis, CTI, various interpretability circuits — but the comparative-anatomy-for-trained-networks program with a three-tier framework and a biology validation path is unclaimed. First-mover advantage is real.

If the moonshot succeeds, it becomes the foundation the other AI Moonshots build on. CTI becomes a special case. Fractal Embeddings becomes a specific Level-2 family constant. CWC and J inherit the atlas as their substrate.

If it fails, we have a rigorously documented atlas and a set of negative results that tell the field exactly which tools do not cross architecture boundaries. That is also a contribution.

---

*Reread this file at the start of every phase. The axiom is the horizon; the manifesto is the compass; the atlas is the step.*
