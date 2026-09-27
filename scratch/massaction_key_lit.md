# Lit scan 2026-09-25 — T051 "mass-action anchor" (e089) + T053 "basis-private key" (e082)

Scope: 2018–2026, web scan only. RESEARCHER angle. Structure: per-claim closest works,
verdicts with reviewer-collapse risk + rebuttal, sources.

---

## CLAIM A — T051/e089 "the mass-action anchor"

**Claim under scan:** In a tiny char-LM's free-running generation, removing self-generated
cache entries is near-free below a mass threshold (~64–128 of ~350), then breaks sharply
convex to full off-manifold collapse. WHICH entries is irrelevant (variance collapse);
HOW MANY is everything. Pair removals sub-additive (median 0.464). Static-vs-dynamic
per-entry cost correlation ~0 (r = −0.004).

### Closest works (ranked by proximity)

1. **StreamingLLM / attention sinks — Xiao et al. 2023** (arXiv:2309.17453) and
   **Massive Activations in LLMs — Sun et al. 2024** (arXiv:2402.17762).
   KV cache has a *flat-then-cliff* quality-vs-budget curve: evict middle tokens freely,
   collapse if you evict the ~4 sink tokens. This is the known dose-response shape —
   BUT their headline is exactly the *opposite* pole on "which": a handful of specific
   entries (sinks / massive-activation tokens) carry everything. Companion work KVSink
   (quantization variant) same story.
2. **H2O: Heavy-Hitter Oracle — Zhang et al., NeurIPS 2023** (arXiv:2306.14048).
   Whole KV-eviction industry (SnapKV, PyramidKV, AdaKV, DynamicKV, ReAttention …)
   built on *which*: accumulated attention mass ranks tokens; keep heavy hitters.
   Implicitly documents "most entries are near-free to evict, a budget cliff exists,"
   but never as a *mass-count law with variance collapse across random subsets*; random
   eviction is their baseline to be beaten, not the finding.
3. **LLMLingua / LongLLMLingua — Jiang et al. 2023/24** (arXiv:2310.05736, 2310.06839).
   Prompt compression: ~4–20x removal is near-free (sometimes *improves* performance),
   past ~20x performance collapses. Empirical dose-response over HOW MANY tokens —
   but token *selection* (perplexity-based scoring) is their core contribution, i.e.,
   again "which matters."
4. **ContextCite: Attributing Model Generation to Context — Cohen-Wang et al. 2024**
   (arXiv:2409.00729). Closest methodological neighbor: ablate random context subsets,
   measure sufficiency/participation of context members. Finds sparse subsets often
   suffice for faithful attribution. Teacher-forced, not free-running; never frames a
   threshold mass law or sub-additivity.
5. **Working-memory-capacity line for LMs:** *Short-term memory in neural language
   models* (OpenReview QNW1OrjynpT); *Working Memory Constraints Scaffold Learning in
   Transformers* (arXiv:2604.20789); *Working Memory Identifies Reasoning Limits in
   LLMs* (Zhang & Jian). Capacity limits exist (Cowan-style chunk limits), but framed as
   input-capacity, not cache-mass dose-response in self-generated context.
6. **Free-running collapse literature (the destination state):** Holtzman et al.,
   *The Curious Case of Neural Text Degeneration* (arXiv:1904.09751, exposure bias /
   repetition); Shumailov et al., *Model Collapse* (arXiv:2305.17493); self-loop
   papers. These characterize degenerate attractors of free generation but never
   identify a *cache-mass threshold* guarding the on-manifold attractor.

### What is NOT in the literature (novelty pockets)

- No paper found where the dose-response curve is over a *self-generated* cache in
  *free-running* generation (attractor-stability experiment, not eval-perplexity).
- No variance-collapse result (which-subset irrelevance) — every major work sells the
  opposite (sinks, heavy hitters, position).
- No sub-additive pair-removal interaction statistics on cache entries.
- No static-vs-dynamic per-entry cost decorrelation (r≈0) falsifying importance scores
  as predictors of dynamical cost.

---

## CLAIM B — T053/e082 "the basis-private key"

**Claim under scan:** An installed fact's *positional address* is structurally universal
across seeds (same wpe coordinates carry the mass at seed 42 AND seed 43), but the *code*
in those rows is seed-orthogonal (row-cos 0.059) — a one-row key cut for one lock; a cheap
pre-graft crossmatch instrument (stream-cosine, AUC 0.919) correctly rejected the one-row
graft in advance.

### Closest works (ranked by proximity)

1. **Git Re-Basin — Ainsworth et al. 2022** (arXiv:2209.14385). Two seeds converge to
   the same basin *modulo permutation*; unaligned, weights are incomparable. This is the
   canonical "shared structure / private basis" result — for *hidden units* under
   permutation symmetry. Says nothing about embedding-row addresses of a specific memory.
2. **Model stitching — Lenc & Vedaldi 2015; Bansal et al. NeurIPS 2021**
   (arXiv:2009.10636). Independently trained nets are stitchable *only after a learned
   linear map* — representations similar in geometry, different in basis. Supports
   "code is private"; operates on whole layers, not single embedding rows or facts.
3. **Relative representations — Moschella et al. ICLR 2023** (arXiv:2209.15430).
   Sharpest "shared-structure/private-code split": latent spaces of independently trained
   models are nearly isometric; zero-shot stitching works after re-basing to similarity
   coordinates. Geometry universal, coordinates private — the general version of Claim
   B's split.
4. **Universal Neurons in GPT2 — Gurnee et al. 2024** (arXiv:2402.04163) + follow-up
   *Universal Neurons in GPT-2: Emergence, Persistence* (Nov 2025). Cross-seed
   neuron-level universality: only 1–5% of neurons universal (and monosemantic).
   Confirms most per-unit code is seed-specific; does not address *address-of-a-fact*
   universality. Related: *Quantifying Feature Space Universality Across LLMs* (2025),
   *Cross-Family Universality of Behavioral Axes* (arXiv:2605.09875).
5. **Cross-model activation intervention transfer:** *Activation Space Interventions Can
   Be Transferred Between LLMs* — Oozeer et al., ICML 2025 (arXiv:2503.04429):
   interventions transfer only via *learned* mappings of shared spaces (transfer works
   with alignment, fails raw — consistent with one-row-graft rejection);
   *On the Limits of Steering Vectors for Preference* (arXiv:2607.01802): vectors do
   NOT transfer reliably across models; *Master Key Hypothesis / "Unlock"* (NeurIPS
   2026, arXiv:2604.06377): cross-model capability transfer succeeds exactly when
   linear-subspace alignment is performed first.
6. **Fact-localization line:** Knowledge Neurons (Dai et al., arXiv:2104.08696);
   ROME (Meng et al., arXiv:2202.05262); *Does Localization Inform Editing?* (Hase et
   al., ICLR 2024, arXiv:2311.17943). Establishes facts live at localized MLP sites —
   but no cross-seed comparison of *the same fact's address*, and Hase et al. warn
   localization ≠ editability.
7. **Pre-graft instrument analog:** transferability estimation (LogME — You et al.
   ICML 2021, arXiv:2103.16850; survey arXiv:2402.15231) predicts transfer success
   cheaply *before* fine-tuning — same epistemic move (predict the graft outcome before
   performing it), never applied to cross-seed single-row knowledge grafts.

### What is NOT in the literature (novelty pockets)

- Nobody found showing the *same embedding-row coordinates* carrying a specific
  installed fact across seeds while the row *content* is orthogonal — universality at
  the granularity of a single memory's address, in a *privileged basis* (wpe rows are
  not permutable hidden units; cf. Elhage et al. 2022, toy models of superposition,
  transformer-circuits.pub/2022/toy_model). Re-basin's permutation story cannot absorb
  this, because wpe rows are index-labeled by position, not free hidden units.
- No "crossmatch" instrument (cheap cosine screen predicting graft rejection with AUC)
  exists at row granularity; LogME-style transferability scores are feature/layer-level
  and task-level.
- Platonic convergence (arXiv:2405.07987) is about geometry across data/modalities at
  scale, not addresses across seeds — adjacent framing only.

---

## VERDICTS

### Claim A (mass-action anchor): **PARTIALLY KNOWN — claimable with reframing**

- Known: flat-then-cliff cache/prompt budget curves (StreamingLLM, H2O, LLMLingua);
  free-running degeneration attractors (Holtzman, Shumailov); context capacity limits.
- Claimable: threshold *mass law* over the self-generated cache in free-running
  generation; WHICH-irrelevance (variance collapse) — this *contradicts and falsifies*
  the importance-score assumption of the entire KV-eviction literature in this regime;
  sub-additive pair interactions; static≠dynamic (r=−0.004).
- **Sharpest reviewer-collapse risk:** "This is the known KV-eviction robustness curve
  (flat-then-cliff) rediscovered in a toy model; 'which doesn't matter' is a small-model
  artifact since H2O/StreamingLLM prove sinks matter in real LLMs; and the sharp
  threshold may be a metric artifact (Schaeffer mirage, arXiv:2304.15004)."
- **Rebuttal:** (1) prior curves are teacher-forced benchmark perplexity on human
  prompts; ours is attractor stability of the model's *own* generation — a different
  dynamical object, and the prior works' own premise (which-matters) *fails* here
  (r=−0.004 static-vs-dynamic; variance collapse across random subsets) — a direct
  falsification, not a rediscovery; (2) threshold measured on continuous
  likelihood/perplexity metrics, not discontinuous accuracy — the mirage mechanism
  cannot manufacture it; (3) sub-additivity (0.464) + convexity are quantitative
  signatures absent from every published curve. Required hedge: state scope as
  ~1–10M-param char-LM regime; position vs. identity of sinks within the anchor band
  should be explicitly compared before claiming full which-irrelevance (StreamingLLM's
  sink asymmetry is the obvious counter-instantiation to test).

### Claim B (basis-private key): **PARTIALLY KNOWN — address-universality + instrument are claimable**

- Known: same-function/different-basis across seeds (git re-basin, model stitching,
  relative representations); most per-unit code seed-specific (universal neurons 1–5%);
  raw cross-model interventions fail, aligned ones succeed (Oozeer, Master Key,
  steering-limits papers).
- Claimable: (1) *fact-address universality in a privileged basis* — same wpe row
  coordinates carry the fact across seeds while row code is orthogonal (cos 0.059);
  this dissociation (address shared, code private) is sharper than and not implied by
  permutation-symmetry results, because wpe rows are index-fixed, not permutable;
  (2) the pre-graft crossmatch instrument (stream-cosine, AUC 0.919) predicting graft
  rejection — the LogME epistemic move transplanted to single-row cross-seed grafts.
- **Sharpest reviewer-collapse risk:** "Seed-orthogonal code is trivially expected from
  re-basin/permutation symmetry; graft rejection is just known unaligned-basis
  non-transfer (steering-vector limits); and n=2 seeds is an anecdote — the shared wpe
  address could be positional-geometry coincidence."
- **Rebuttal:** (1) permutation symmetry applies to hidden units; wpe rows live in a
  privileged, position-indexed basis — re-basin cannot predict *which address* a fact
  occupies, so cross-seed address agreement is a positive result, not a default;
  (2) the deliverable is the validated *predictor* (AUC 0.919) — prior cross-model
  transfer papers report transfer success/failure post hoc, never a cheap pre-graft
  rejection test at row granularity; (3) n=2 is the registered extrapolation point:
  run n=5–10 seeds; if address agreement holds above chance, claim graduates from
  anecdote to law.

### Cross-claim synergy (for the reviewer)
Claim B's private-code result explains why Claim A's cache entries have no individual
identity that predicts dynamical cost (no stable cross-run row semantics), and Claim A's
mass law explains why the fact-key can be a single row (the cache supplies redundant
mass; the key supplies the address). Neither half is in any single prior work.

---

## SOURCES

Claim A:
- StreamingLLM (attention sinks): https://arxiv.org/abs/2309.17453
- Massive Activations in LLMs: https://arxiv.org/abs/2402.17762
- H2O Heavy-Hitter Oracle: https://arxiv.org/abs/2306.14048
- LLMLingua: https://arxiv.org/abs/2310.05736 ; LongLLMLingua: https://arxiv.org/abs/2310.06839
- Lost in the Middle (position matters): https://arxiv.org/abs/2307.03172
- ContextCite: https://arxiv.org/abs/2409.00729 (project: https://gradientscience.org/contextcite/)
- Working Memory Constraints Scaffold Learning in Transformers: https://www.alphaxiv.org/abs/2604.20789v1
- Short-term memory in neural LMs: https://openreview.net/forum?id=QNW1OrjynpT
- Neural text degeneration (free-running collapse): https://arxiv.org/abs/1904.09751
- Model collapse (recursive self-consumption): https://arxiv.org/abs/2305.17493
- Emergent-abilities mirage (metric-artifact risk): https://arxiv.org/abs/2304.15004

Claim B:
- Git Re-Basin: https://arxiv.org/abs/2209.14385
- Revisiting Model Stitching: https://arxiv.org/abs/2009.10636
- Relative Representations: https://arxiv.org/abs/2209.15430
- Platonic Representation Hypothesis: https://arxiv.org/abs/2405.07987
- Universal Neurons in GPT2: https://arxiv.org/abs/2402.04163 (Nov 2025 follow-up on emergence/persistence across seeds: title-only, ID unverified)
- Activation Space Interventions Can Be Transferred (ICML 2025): https://arxiv.org/abs/2503.04429
- Limits of Steering Vectors for Preference: https://arxiv.org/abs/2607.01802
- Master Key Hypothesis / Unlock (NeurIPS 2026): https://arxiv.org/abs/2604.06377
- Cross-Family Universality of Behavioral Axes: https://arxiv.org/abs/2605.09875
- Knowledge Neurons: https://arxiv.org/abs/2104.08696 ; ROME: https://arxiv.org/abs/2202.05262
- Does Localization Inform Editing? (Hase et al.): https://arxiv.org/abs/2311.17943 / https://openreview.net/forum?id=EldbUlZtbd
- LogME (transferability estimation): https://arxiv.org/abs/2103.16850 ; survey: https://arxiv.org/abs/2402.15231
- Privileged basis (toy models of superposition): https://transformer-circuits.pub/2022/toy_model/index.html
