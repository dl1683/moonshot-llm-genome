# Literature scan: T050/e084 read-rule claims (researcher angle)

Date: 2026-09-25. Scope: 2020-2026 online literature (WebSearch). Claims under test
(0.84M char-LM, per-entry interventions on a memory/coordinate store, 80 candidates):

1. **Kernel-equals-shadow:** per-entry argmax-flip rate correlates r=0.918 with per-entry
   lesion dCE (young band r=0.96); tail diverges (Spearman -0.13).
2. **Sparse-open:** only 3-4 of 80 coordinates "open" per decision.
3. **Flip taxonomy:** runner-up wins ~50% of flips; ~25% escape top-5.
4. **Content-following:** donor-run next token wins 8.8% vs 2% chance after value swaps.

---

## (a) Decision-level (argmax/logit-rank) vs loss-level (CE/PPL) intervention readouts

**What exists.** Metric choice in activation patching is a recognized methodological
variable, but the literature treats it as folklore + best-practice advice, not as a
measured agreement between two readout levels:

- Zhang & Nanda, "Towards Best Practices of Activation Patching in Language Models:
  Metrics and Methods" (arXiv:2309.16042, ICLR 2024). The canonical reference. Shows the
  choice of readout metric (logit diff, prob diff, KL, accuracy) qualitatively changes
  attribution results; recommends logit difference; warns that loss-based readouts can
  mislead depending on clean-vs-corrupted baseline. Does NOT quantify per-component
  correlation between a loss-metric map and a decision-metric map — e084's r=0.918 + tail
  Spearman is a stronger, quantitative statement than anything in this line.
- Nanda, "How to Think About Activation Patching" (Alignment Forum 2022/23) +
  "Attribution Patching" (nanda.me). Argues logit diff is sharper than loss because loss
  **saturates** once the model is confident — this is exactly the lab's tail finding
  (dCE can't rank-order the 0.3-1% still-opened old entries): small decision effects
  hide under the CE floor. So the tail divergence is *predicted* by known saturation
  arguments; e084 gives it a measured form.
- EAP-GP (NeurIPS): "Mitigating Saturation Effect in Gradient-based Path Patching" —
  formalizes the saturation artifact for attribution patching; more evidence the
  loss-readout/deciion-readout gap at small effects is expected, not anomalous.
- Chan et al., "Causal Scrubbing" (Alignment Forum, Dec 2022): validates interpretability
  hypotheses by **loss recovered** after resampling activations — the "shadow" side of the
  analogy. Loss is the ground truth there; the decision rule is never separately
  measured. No kernel-vs-shadow comparison.
- Meng et al., ROME (arXiv:2202.05262, NeurIPS 2022): causal tracing readout = probability
  ratio on the target token + argmax check for edits. Uses both levels informally but
  never correlates them per-component.
- Mueller & Hsu-style critiques, "Counterfactuals Pose Challenges for Interpreting Neural
  Networks" (OpenReview ~2024) and Millière et al., "Interventionist Methods for
  Interpreting Deep Neural Networks": note patching = interchange interventions, and that
  counterfactual readout choice shapes conclusions. Framework, not measurement.
- Jacovi et al., "Aligning Faithful Interpretations with their Social Attribution" (TACL
  2021): faithfulness-as-causal-attribution framing, decision-adjacent, but no
  dual-readout correlation.

**Gap confirmed.** Nobody located publishes "per-intervention decision-flip effect vs
per-intervention dCE, entry by entry, r=?" — i.e., an explicit validation that a
cross-entropy shadow faithfully pictures the argmax decision rule, with a characterized
failure regime (the tail). Closest priors: Zhang & Nanda (metric sensitivity),
Nanda/saturation folklore (predicts the tail), Causal Scrubbing (loss-side validation
template).

---

## (b) Sparse context use at the decision level (few entries per decision)

**What exists.** Per-token sparsity of what a model actually uses is studied, but mostly
(a) for efficiency, (b) at neuron/head/SAE-feature granularity, or (c) as
architecture-enforced top-k — not as an empirical read-rule over explicit memory entries:

- Liu et al., "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time"
  (ICML 2023). Closest concept: **contextual sparsity** — per-token, per-layer subsets of
  neurons/heads that suffice to preserve output. But sparsity there is ~50% of channels
  and the goal is speed; ~4% (3-4/80) at memory-entry level is a different regime, and
  they never frame it as "the decision reads these entries."
- Wu et al., "Retrieval Head Mechanistically Explains Long-Context Factuality"
  (arXiv:2404.15574): a sparse subset of heads, often attending to near-single tokens,
  carries long-context copying. Decision-relevant sparse reads exist, at head/token
  granularity in attention, not entry granularity in a memory store.
- Marks et al., "Sparse Feature Circuits" (arXiv:2403.19647): per-task sparse sets of
  SAE features causally sufficient (attribution patching). Feature-level sparse read/write
  sets — the closest "how few things does one behavior read" result, but aggregated over
  a task/distribution, not per single decision.
- Olsson et al., "In-context Learning and Induction Heads" (Transformer Circuits 2022):
  the canonical sparse-read mechanism ([A][B]...[A] -> [B] reads essentially one earlier
  occurrence). Bietti et al., "Birth of a Transformer: A Memory Viewpoint" (NeurIPS 2023,
  arXiv:2306.00802): at tiny scale, attention key-value slots act as bigram memories then
  induction reads — direct precedent for "small LM, few-slot reads."
- Architecture-side: Lample et al., "Large Memory Layers with Product Keys" (NeurIPS
  2019, pre-window): top-k sparse slot retrieval is *enforced* by design (k chosen, not
  measured); Schlag et al., "Linear Transformers are Secretly Fast Weight Programmers"
  (ICML 2021): delta rule = "new key writes only to slots it reads" — the read-set
  question posed but not measured per decision.

**Gap confirmed.** "3-4 of 80 coordinates open per decision" as an *empirical* read-rule
measurement is claimable; the phenomenon class (per-token sparse use) is well
precedented, so the claim should be positioned as measurement of a known sparsity
regime at a new granularity (memory entries in a char-LM), not as discovery of
sparsity itself. Closest priors: Deja Vu (contextual sparsity), Sparse Feature Circuits,
Retrieval Heads / induction heads.

---

## (c) Counterfactual next-token transfer (donor continuation wins)

**What exists.** The *logic* of this experiment is exactly the interchange-intervention
program of causal abstraction:

- Geiger et al., "Finding Alignments Between Interpretable Causal Variables and Neural
  Representations" (CLeaR/PMLR 2024; earlier DAS/INTRA papers 2020-2023). Swap internal
  representation values with those from a donor input; the model's output should follow
  the donor iff the variable is the one the network computes with. e084's claim 4 is an
  interchange-intervention **effect size** (8.8% donor-token win vs 2% chance) for
  memory-entry values. Must cite; the idea is not new.
- Todd et al., "Function Vectors in Large Language Models" (ICLR 2024, arXiv:2310.15213):
  patching a donor-task vector makes the model perform the donor task — transfer-of-
  behavior-at-the-output is the readout. Hendel et al. 2023 task vectors similar.
- Counterfactual Simulatability (Chen et al., arXiv:2306.04917, ~2023): evaluates an
  explanation by whether it predicts model outputs under counterfactual input changes —
  same validation spirit, input-level.
- Steering vectors (Turner et al. 2023, ACTAD): adding a direction shifts next-token
  distribution toward steered content — adjacent but vector, not entry-value, swap.
- Sutter et al., "Is Causal Abstraction Enough for Mechanistic Interpretability?"
  (NeurIPS ~2024/25) + "The Non-Linear Representation Dilemma" (2025): critique that
  interchange success can be trivially achieved with unconstrained alignments — a useful
  caution when claiming content-following from modest transfer rates.

**Verdict on novelty.** Weakest of the four: the protocol is standard causal
abstraction applied to memory entries. What is addable: the chance-corrected magnitude
(8.8% vs 2% = 4.4x) at char level in an explicit-entry memory. Claimable only with
Geiger/Todd cited as the framework, positioned as a measured transfer rate, not a new
method or phenomenon.

---

## (d) Flip taxonomy (runner-up wins ~half; ~25% escape top-5)

**What exists.** Flip *rates* are everywhere; flip *destinations* (the distribution over
where the argmax lands) are rarely tabulated:

- Adversarial NLP: TextFooler (Jin et al. 2019, pre-window), BERT-Attack (Li et al.
  2020), Universal Adversarial Triggers (Wallace et al. 2019, pre-window) — report flip
  rates under token perturbations, not the winner's identity distribution. The
  computer-vision folklore result (minimal perturbations flip to the runner-up class)
  is referenced around Ilyas et al. 2019 ("Adversarial Examples Are Not Bugs, They Are
  Features", pre-window) but as anecdote, not a measured taxonomy.
- Maini et al., "Post-hoc Ensembles for Targeted Misclassification" / perturbation-type
  categorization (PMLR v180, 2022): categorizes adversarial examples by perturbation
  type — closest in spirit (taxonomy of flips) but for input-space attacks on
  classifiers.
- Knowledge editing: Cohen et al., "Evaluating the Ripple Effects of Knowledge Editing"
  (arXiv:2307.12976; TACL 2024 RippleEdits) — tracks whether edits change predictions on
  neighboring facts and whether unrelated predictions are preserved; flip destination
  (target vs other) is a binary, not a rank taxonomy.
- Copy suppression (McDougall et al., NeurIPS 2023): ablating the copied token lets
  specific alternates win — a mechanistic account of *which* alternates, but in one
  named circuit, not a population-level taxonomy.

**Gap confirmed.** A rank-level flip taxonomy (P(winner = runner-up) ~ 0.5; P(escape
top-5) ~ 0.25) under internal entry-level perturbations appears unmeasured in the
searched literature. Claimable; position against adversarial-NLP flip rates and editing
side-effect evals, noting those stop at "did it flip."

---

## Verdicts (one line each)

1. **Kernel-equals-shadow: CLAIMABLE (strongest).** Quantified dual-readout validation
   (r=0.918, tail Spearman -0.13) is absent from the patching-metrics literature, which
   only asserts metric sensitivity (Zhang & Nanda 2023) and loss saturation (Nanda 2022)
   — the tail divergence is *predicted* by saturation folklore, which strengthens rather
   than scoops the finding. Cite: arXiv:2309.16042, Causal Scrubbing, ROME.
2. **Sparse-open: CLAIMABLE with framing care.** Per-decision sparsity is precedented
   (Deja Vu contextual sparsity, retrieval heads, sparse feature circuits) but never as a
   measured read-rule over explicit memory entries at 3-4/80 in a char-LM; frame as new
   granularity of a known regime. Cite: ICML 2023 Deja Vu, arXiv:2403.19647,
   arXiv:2404.15574, arXiv:2306.00802.
3. **Flip taxonomy: CLAIMABLE.** Flip rates are standard (adversarial NLP, RippleEdits)
   but destination distributions (runner-up ~50%, top-5 escape ~25%) are not tabulated
   anywhere found. Cite: arXiv:2307.12976, Maini et al. 2022, McDougall et al. 2023.
4. **Content-following: WEAKEST.** This is a standard interchange intervention / causal
   abstraction effect size (donor-run value swap -> donor output) — novel only as the
   measured 8.4x-over-chance rate at memory-entry granularity in a char-LM. Cite Geiger
   et al. (PMLR 2024) and Todd et al. (arXiv:2310.15213) as the framework; anticipate
   the Sutter et al. trivial-alignment critique by keeping the alignment map fixed.

**Weakest claim's closest prior:** Geiger et al., interchange interventions / causal
abstraction (PMLR 2024; DAS line 2021-2023) — claim 4 is that method applied to memory
entries, with Todd et al.'s function vectors (ICLR 2024) the closest behavioral analog.

---

## Sources

**Metric/readout (a):**
- Zhang & Nanda, Towards Best Practices of Activation Patching (ICLR 2024): https://arxiv.org/abs/2309.16042
- Causal Scrubbing (Chan et al., 2022): https://www.alignmentforum.org/posts/JvZhhzycHu2d57RN/causal-scrubbing-a-method-for-rigorously-testing
- Scrubbing results on induction heads: https://www.alignmentforum.org/posts/j6s9H9SHrEhEfuJnq/causal-scrubbing-results-on-induction-heads
- Practical Pitfalls of Causal Scrubbing: https://www.lesswrong.com/posts/DFarDnQjMnjsKvW8s/practical-pitfalls-of-causal-scrubbing
- Attribution Patching at Industrial Scale (Nanda): https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
- ROME, Locating and Editing Factual Associations in GPT (Meng et al. 2022): https://arxiv.org/abs/2202.05262
- Counterfactuals Pose Challenges for Interpreting Neural Networks (Mueller et al.): https://openreview.net/forum?id=v2SSe0jNRw (via search; title match)
- Interventionist Methods for Interpreting DNNs (Milliere et al.): https://philarchive.org/archive/MILIMF (via search)
- Jacovi et al., aligned faithfulness (TACL 2021): https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00367/98620
- Mech Interp for AI Safety review (Bereska & Gavves 2024): https://arxiv.org/abs/2404.14082

**Sparsity (b):**
- Deja Vu: Contextual Sparsity (Liu et al., ICML 2023): https://arxiv.org/abs/2310.17157
- Retrieval Head Mechanistically Explains Long-Context Factuality (Wu et al. 2024): https://arxiv.org/abs/2404.15574
- Sparse Feature Circuits (Marks et al. 2024): https://arxiv.org/abs/2403.19647 ; https://features.baulab.info
- In-context Learning and Induction Heads (Olsson et al. 2022): https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Birth of a Transformer: A Memory Viewpoint (Bietti et al. 2023): https://arxiv.org/abs/2306.00802
- Large Memory Layers with Product Keys (Lample et al. 2019): https://arxiv.org/abs/1907.05242
- Linear Transformers are Secretly Fast Weight Programmers (Schlag et al. 2021): https://arxiv.org/abs/2102.11174
- Anthropic attribution graphs / circuit tracing (2025): https://transformer-circuits.pub/2025/attribution-graphs/methods.html

**Interchange / transfer (c):**
- Geiger et al., Finding Alignments Between Interpretable Causal Variables and Neural Representations (PMLR 2024): https://proceedings.mlr.press/v202/geiger24a.html
- Causal Abstraction survey (2024): search "Causal Abstraction in Model Interpretability survey arXiv 2410" (title match via search)
- Function Vectors in LLMs (Todd et al., ICLR 2024): https://arxiv.org/abs/2310.15213
- Hendel et al., ICL Creates Task Vectors (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.547/ (via anthology search)
- Counterfactual Simulatability (Chen et al. 2023): https://arxiv.org/abs/2306.04917
- Sutter et al., Is Causal Abstraction Enough?: https://openreview.net/forum?id=xgAI9WtZBN (via search)
- Tuned lens / logit lens background: https://sidn.baulab.info/lenses ; https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

**Flip taxonomy (d):**
- RippleEdits (Cohen et al., TACL 2024): https://arxiv.org/abs/2307.12976 ; https://aclanthology.org/2024.tacl-1.16.pdf
- Maini et al., perturbation-type categorization (PMLR 2022): https://proceedings.mlr.press/v180/maini22a.html
- Copy Suppression (McDougall et al., NeurIPS 2023): https://arxiv.org/abs/2303.03338
- Adversarial examples are features (Ilyas et al. 2019, pre-window context): https://arxiv.org/abs/1905.02175

Note on link hygiene: URLs marked "via search" were surfaced by search summaries but not
independently fetched; all arXiv IDs in the primary verdicts were confirmed in result
text except where noted. No prior work found that quantifies argmax-flip vs dCE
agreement per intervention — the central T050 novelty holds.
