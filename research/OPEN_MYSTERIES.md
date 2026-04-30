# OPEN MYSTERIES

*Unresolved cross-system phenomena the Neural Genome inherits. Each is a testable atlas entry in waiting; each is a potential paper if its mechanism is nailed.*

Every mystery in this file has at minimum: a one-line phenomenon statement, a candidate hypothesis landscape, a priority experimental action, and a pointer to prior work. Mysteries that collect three failed resolution attempts are flagged as **scars** — structural features of the atlas that the current framework cannot explain; these are prioritized for framework revision.

---

## Mystery 1 — Orthogonality coupling at 7B+

**Phenomenon.** At sub-3B scale, participation-ratio surgery and jitter stress are orthogonal: they are independent axes of instability. At 7B+, a coupling emerges (p=0.013).

**Prior work.** `LLM exploration/` experiments 012–016. Three mechanistic hypotheses have been falsified:
- Layer-local coupling → no (couplings are not layer-localized)
- Reasoning-training-induced → no (present in base models too)
- Predicted by local geometry → no (no local geometry signal correlates)

**What's left.** Experiment-017 was in progress at the dormant date. Remaining candidates are non-local: global representational manifold properties, phase transitions, or depth-dependent capacity allocation.

**Why it matters for the genome.** If a sharp scale-transition exists in how LLMs organize representations, that is a *scaling law of structure* — qualitatively different from parameter-count scaling laws. A Level-1 candidate.

**Priority actions:**
1. Reproduce the 7B+ measurement on the frozen canonical registry, getting SSM and Hybrid data at 7B+ (prior HF DL failures blocked this).
2. Test phase-transition hypothesis with a fine scale sweep (0.6B, 1.7B, 3B, 4B, 7B, 14B): is the coupling sharp or gradual? Sharp = phase transition; gradual = smooth emergence.
3. Cross-class replication: does the coupling appear in Mamba2 at 7B? Falcon-H1 at 7B? If yes, it is architecture-universal. If only Transformers, it is family-specific (Level-2 boundary).

**Status.** ⚫ unresolved. Entering its second year.

---

## Mystery 2 — Reading / writing asymmetry

**Phenomenon.** Probes reliably *read* concept representations at high accuracy. Steering vectors reliably *fail to write* those same concepts. Four independent write-side approaches tried in the old LLM Genome Project — all failed. Rosetta's cross-architecture linear alignment also fails (p=0.82, captures text statistics, not capabilities).

**Hypothesis landscape:**
- **(a) Off-manifold linear writes fail.** Concepts live on a lower-dimensional manifold than linear directions capture. Reading tolerates off-manifold noise; writing needs to stay on-manifold, so linear pushes land in invalid regions.
- **(b) Distributed edits succeed, single-layer edits fail.** Writing requires matched edits across many layers, not a single-direction push. The CAA result on gemma-3-1b-it supports this indirectly — it is a single layer, but instruction tuning may have pre-concentrated the "write handle."
- **(c) Activation-vs-weight asymmetry.** `knowledge-surgeon`'s LoRA writes work (100% on geographic facts) because they are weight-space, not activation-space. If the asymmetry is activation-specific, it is a statement about the geometry of activation space, not about representation per se.

**Why it matters for the genome.** If reading and writing live in geometrically different places, the "coordinate system" framing is incomplete — a genome has to address both afferent and efferent geometry. This is the central interpretability bottleneck.

**Priority actions:**
1. **Non-linear writes.** Train a small MLP edit head per concept, holding other dimensions constrained. If non-linear wins, hypothesis (a) is correct; manifolds matter.
2. **Distributed multi-layer writes at matched L2 norm.** Compare to single-layer baselines. Tests hypothesis (b).
3. **Weight-vs-activation comparison.** Replicate the `knowledge-surgeon` LoRA vs. CAA experiment on the same concept on the same model. Is LoRA-style weight edit equivalent to a hypothetical infinite-composition activation steer?

**Status.** 🔥 high priority. Resolving this opens the path to Phase-5 engineering.

---

## Mystery 3 — Why 2 tokens of random noise fixes reasoning

**Phenomenon.** Prepending two random embedding-scale soft-prompt tokens to Qwen3-4B lifts arithmetic by +19.6 percentage points. Direction-agnostic (p=1.0 vs. optimized). Non-monotonic dose-response (2 is optimal; 1 is insufficient, 4 equalizes back down, p=0.335). Transfers to legal (92% oracle win rate) and planning (rescues catastrophic failures).

**Prior work.** `Latent-Space-Reasoning/` and `AI Moonshots/moonshot-latent-space-reasoning/`. Current mechanism hypothesis: breaks attention-sink pathology in greedy decoding.

**Open questions:**
- Is attention-sink *the* mechanism or a correlate? No ablation has established causality.
- Why two? What is the principled derivation?
- Why direction-agnostic? If random == optimized, the structure that matters is *the act of perturbation*, not the direction — more a phase-space argument than an embedding-direction argument.

**Why it matters for the genome.** If the mechanism is generic (phase-space perturbation escape from a degenerate basin), it applies to any small LM with the pathology — and possibly to diffusion LMs, SSMs, and recurrent models too. Cross-class transfer would be a genome-level invariant.

**Priority actions:**
1. Ablate attention-sink directly. Load a sink-less model (StreamingLLM without sink; or a model without sink pathology) and test whether LSR still helps. If yes, mechanism is broader than attention-sink.
2. Scale test: does 2-token perturbation work at 14B? 32B? If yes, it is a general phenomenon; if not, a small-model result.
3. Replicate on SSM, hybrid, and diffusion-LM. Cross-class presence is what makes this a genome-level mystery.
4. Derive "2" theoretically. Treat decode as a dynamical system; compute fixed points; verify that two perturbation tokens are the minimum for basin escape.

---

## Mystery 4 — CAA steering works on instruction-tuned Transformers, fails elsewhere

**Phenomenon.** Contrastive Activation Addition achieves p=0.008, Cohen's h=0.800, clean dose-response on gemma-3-1b-it (instruction-tuned Transformer). Fails on Hybrids (Falcon-H1 fluency collapses under steering). Underpowered on SSMs.

**Prior work.** `llm-rosetta-stone/`.

**Hypothesis landscape:**
- **Instruction tuning concentrates the "write handle" into a specific layer** → steerability is not architectural; it is a post-training artifact.
- **Hybrids and SSMs have less-linear concept structure** → non-linear probes/writes should work where linear CAA fails.
- **Layer selection matters more for non-transformer systems** → CAA's layer-sweep methodology may systematically miss the right depth in SSMs.

**Why it matters for the genome.** CAA is the cleanest linear causal primitive we have. If it is transformer-specific, the atlas needs a *distinct* causal primitive for each system class — or a unifying primitive that subsumes CAA as a special case. Either way, a genome-level question.

**Priority actions:**
1. **Full layer sweep on Mamba2-1.3B and Falcon-H1-1.5B.** Has the right depth just been missed?
2. **Non-linear CAA variants** — MLP-steered concept edits on the same models.
3. **Relate the instruction-tuning finding to the write-handle concentration hypothesis.** If CAA succeeds only post-instruction-tuning, that tells us instruction tuning is geometrically real, not just behavioral.

---

## Mystery 5 (inherited prior) — Coherent divergence under extreme compression

**Phenomenon.** Compressing an LLM's hidden representations by 99.7% (896D → 3D, Qwen2.5-0.5B-Instruct) produced grammatically perfect, semantically coherent but *different* text — not gibberish.

**Interpretation.** Representations factorize: `h ≈ h_fluency + h_content`. Content destroyed; fluency preserved.

**Prior work.** Old LLM Genome Project (now deleted). Headline finding that motivated the original project. BERTScore F1=0.770 vs baseline confirmed content destruction; 100% grammaticality confirmed fluency preservation.

**What the genome asks.** Does the same factorization appear in:
- Other architectures (SSM, hybrid, diffusion LM)?
- Other scales (is the content-subspace dimensionality a function of scale)?
- Other modalities (does V-JEPA have an analogous "semantic-dimension destroyed, structural-dimension preserved" factorization across compression levels)?
- Biology (can V1 recordings be compressed in an analogous structured way without losing ethological response properties)?

**Why it matters.** If this factorization is Level-1 universal, it is the single strongest candidate for a genome primitive. "Content" becomes addressable across systems.

**Priority actions:**
1. Replicate on Mamba2-370M, Falcon-H1-0.5B, LLaDA-8B.
2. Scale study on Qwen3 family (0.6B, 1.7B, 4B, 8B): how does content-subspace dimension scale with model size?
3. Apply to V-JEPA: can the predictor's latent be compressed along a "content" axis while preserving "structural" prediction accuracy?
4. Biological analogue: can mouse V1 recordings be compressed along a "semantic / category" axis while preserving orientation / spatial-frequency tuning? Allen Neuropixels.

---

## Mystery 6 — The modality gap

**Phenomenon.** CLIP-style joint vision-language models consistently show a "modality gap": image and text embeddings occupy non-overlapping regions of the shared space despite being trained to align (Liang et al. 2022 "Mind the Gap").

**Why it matters for the genome.** The modality gap is a counter-example to naive universality. Two *explicitly aligned* representation spaces retain architectural signatures. What does this say about implicit universality across non-aligned systems?

**Hypothesis.** The gap is a Level-2 family constant — a quantifiable geometric offset between vision-encoder-produced activations and language-encoder-produced activations. The *distance* is architecture-specific; the *existence* of a gap may be universal.

**New context (2026-04-29):** The "Umwelt Representation Hypothesis" (arXiv 2604.17960, April 2026) directly addresses this mystery. It argues alignment arises from overlapping ecological constraints, not convergence to a universal optimum — modalities as distinct "Umwelten" with local, partial alignment. If correct, the modality gap is a *feature* of ecological specialization, not a failure of convergence. This supports our g180b finding (tokenizers = different Umwelten → geometry is tokenizer-specific) and our pivot to architecture-explicit forecasting (g182).

**Priority actions:**
1. Measure the gap in multiple joint encoders (CLIP, SigLIP, BLIP-2).
2. Test whether the gap's magnitude correlates with a predictable architectural property.
3. Test whether non-aligned bestiary systems exhibit analogous inter-class gaps (LLM activations vs. vision-encoder activations on matched stimuli).
4. **(NEW)** Read and integrate the Umwelt Representation Hypothesis framing — does their "ecological constraint" explanation predict the modality gap's magnitude?

---

## Mystery 7 — Sparse autoencoder feature universality

**Phenomenon.** SAE-discovered features are largely interpretable per-model but poorly compared across models. It is unclear whether "the same feature" appears across architectures, or whether SAEs find family-specific features.

**Why it matters for the genome.** If SAEs produce a universal feature vocabulary, the genome's content coordinates may be SAE features. If they produce family-specific vocabularies, SAEs are a diagnostic, not a coordinate.

**Priority actions:**
1. Train matched SAEs on Qwen3-0.6B, Mamba2-370M, Falcon-H1-0.5B. Compute feature similarity across models.
2. Test feature universality on a paired input set (same prompts, each model's SAE features activated).
3. Relate SAE feature geometry to manifold primitives (ID, persistent homology).

---

## How to add a mystery

A new mystery enters this file only if:

1. The phenomenon is documented (paper, prior project, or clean pilot experiment with n ≥ 30).
2. At least one hypothesis has been posed.
3. A priority action is specifiable.
4. It is not already subsumed by an existing mystery.

Codex's Cross-System Auditor reviews proposed additions. Mysteries that are actually just symptoms of deeper mysteries are merged, not duplicated.

## Mystery 8 — Architecture-specific representational charts (GOLD MINE)

**Phenomenon.** Every cross-architecture experiment fails (g173, g180b, g182, g186), while within-family effects are strong (g181b +0.513 nats, g183 +0.389 nats). Forensic analysis of 6 experiments reveals a clear hierarchy:

1. **KD cross-arch with shared tokenizer works (g173: 101% retention) but provides ZERO efficiency gain** — soft-label signal is architecture-agnostic regularization, not structure transfer.
2. **KD cross-tokenizer actively HURTS (g180b: -0.37 to -0.54 nats)** — wrong codebook = toxic signal.
3. **Weight/embed transfer only works within-family** — geometry is incommensurable across architectures.
4. **Transformer blocks without matching interface HARM (g181a: -0.44 nats)** — internal geometry is dependent on interface geometry.
5. **Geometric features are architecture fingerprints, not invariant coordinates (g182: R^2 = -11 to -19)** — same feature name, different coordinate chart.

Specific numbers: Qwen3 vs GPT-2 feature means: mid_spectral_alpha 0.666 vs 1.085, kNN10 0.645 vs 0.508, TwoNN ID 7.15 vs 11.18. These are NOT the same coordinate system.

**Key insight (Codex cycle 147):** "The transferable object is an interface codebook plus decoder. Tokenizer defines the codebook. Architecture defines the decoder." Strong-form cross-architecture transfer is fundamentally false in the current framing.

**Hypothesis landscape:**
- **(a) Tokenizer imposes a codebook; architecture imposes a decoder.** Cross-arch fails because same codebook + different decoder = misaligned priors. Supported by g173 (shared tokenizer enables KD) + g180b (different tokenizer = toxic).
- **(b) ~~A topology-aware bridge (ultrametric, OT diffusion, hierarchical wavelets) could map between codebooks.~~** **TESTED — OT bridge FAILS (g188: -0.119 HARMS). But direct string matching WORKS (+0.478 nats = 93.2% of within-family effect).** The signal is exact lexical token identity, not topological mapping. g191 decomposing whether it's content or format.
- **(c) The interface geometry IS the fundamental invariant.** The training diagnostic pivot (predicting health from interface geometry) may be more productive than cross-arch transfer.

**Why it matters.** This is the central finding of the Neural Genome project so far. Understanding WHY architectures impose different charts is either (i) a fundamental barrier to universal transfer (the finding itself becomes the contribution) or (ii) the key to designing architecture-agnostic representations.

**Priority actions:**
1. **g191 string-match decomposition (RUNNING)** — decompose +0.478 into content vs format. If PASS_CONTENT, the signal is trained semantic vectors at exact-string positions.
2. **g192 28-layer replication (gated on g191)** — test whether the signal persists in full 28-layer Qwen3 (not just 8-layer shell).
3. **g187 ultrametric diagnostic on Pythia checkpoints** — does embedding geometry become increasingly ultrametric during training? (NOVEL gap)
3. ~~**Successive-refinement codebook ladder**~~ — **RUNG 1 DEAD (g183 FAIL), RUNG 2 DEAD (g188 flow_bridge FAIL).** PPMI SVD harms (-0.291). OT-bridged trained embeddings harm (-0.119). But direct string matching (+0.478) shows the signal IS in trained embedding content at exact-string-matched positions. g191 testing whether it's content vs format.
4. **Architecture-conditioned compatibility law** — train per-arch with scratch normalization, test frozen on third arch.

**Status.** 🔥 ACTIVE GOLD MINE. This pattern, if properly characterized, IS the contribution.

---

## Scar flag

A mystery becomes a **scar** (🩹) when three independent resolution attempts have failed. Scars are elevated: they become candidates for framework revision, not just more experiments. If a scar persists, the framework might be wrong, not the experiments.

Current scar count: 0. Mystery 1 (orthogonality coupling) is at 3 failed hypotheses — one more failure and it graduates to scar status.
