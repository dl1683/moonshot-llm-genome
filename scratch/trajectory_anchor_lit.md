# T048 trajectory-anchoring — literature scan (RESEARCHER, 2026-09-25)

Claim scanned: mid-run deletion of a run's OWN generated KV entries costs +0.26 nats and
drives free-running generation into a self-consistent off-manifold attractor; presence
(norm-matched noise) equally bad; borrowed real content recovers only ~3/4; combined with
T045 (lesion-helpful junk is self-generated, late-generation). Slogan: the model's own
accumulated outputs anchor its continuation — removing them, even when individually
harmful, collapses the run.

## Q1 — self-conditioning / history entries as non-substitutable anchors

NO prior found (2018-2026) that frames cache/history entries as run-specific,
non-substitutable trajectory anchors, let alone with the lab's three-arm discriminator
(vzero vs norm-matched noise vs borrowed real content). Nearest partials:

- Self-preference / self-recognition line: Panickssery et al. 2024 (NeurIPS) show LLM
  judges score their own generations higher, with a linear link between self-recognition
  and self-preference. Ackman et al. 2024 control self-generated-text recognition;
  "Extreme Self-Preference" (2509.26464) shows minimal identity cues trigger it. All
  eval-time preference phenomena — no intervention on the generating run, no anchoring.
- Induction heads (Olsson et al. 2022) + copy-suppression heads (McDougall et al.,
  NeurIPS 2023): the MECHANISM by which self-history is consumed (find earlier occurrence
  of current token, copy continuation). Establishes that earlier context — including the
  run's own output — is the substrate of continuation, but ablations are head-level and
  eval-time; no deletion-of-own-history-during-generation experiment.
- "Repetition In Repetition Out" (Ivgi et al., NeurIPS 2023): repetition in the CONTEXT
  self-reinforces repetition in the continuation (data-perspective). Content of history
  shapes continuation — the weak version of anchoring — via token statistics, no cache
  intervention, no recovery controls.
- Context rot (Chroma 2025) / context collapse (agentic-memory usage): degradation with
  input length in eval settings; unrelated to run-specific anchoring.

## Q2 — flip side of exposure bias: is self-history ever shown LOAD-BEARING?

No. The exposure-bias/snowball line is entirely one-directional (self-history HURTS):

- Zhang & Press, "How Language Model Hallucinations Can Snowball" (ICML 2024 /
  arXiv:2305.13534): explicit exposure-bias framing (train on gold, condition on own
  history at inference); interventions ADD correct info to context to reduce
  snowballing. Never deletes benign self-history; never shows removing it damages the run.
- Braverman et al. 2019 (arXiv:1906.05664): entropy-rate miscalibration → perplexity of
  self-generated text grows without bound; calibration fixes it. "Memory" framing is the
  closest vocabulary, but again no mid-run deletion intervention.
- Arora et al. 2023, Stable Entropy Hypothesis (arXiv:2302.06784): model text drifts out
  of the stable entropy band of human text (the lab's entropy-band/dead-band concepts
  rhyme with this — worth citing); mitigation is entropy-constrained decoding.
- Kalai et al. 2025 (arXiv:2509.04664): hallucination as statistical inevitability;
  orthogonal.
- Hardware-fault error propagation (arXiv:2606.02430, 2026): bit-flips in hidden states
  during inference propagate — perturbation-during-inference vocabulary, but fault
  tolerance, zero content-specificity.

The lab's sign inversion (self-history entries that look individually harmful at eval
time are dynamically load-bearing in free-run) has NO counterpart found.

## Q3 — KV-cache editing/pruning during GENERATION: dynamic damage from eviction?

- StreamingLLM (Xiao et al., arXiv:2309.17453) — verified against full text:
  (a) degradation reported is streaming PERPLEXITY ON GOLD TEXT, not free-running
  generation quality (no coherence table for cache-dropped free-run);
  (b) eviction removes the EARLIEST tokens only — middle-token eviction never tested;
  (c) their one substitution control replaces the first-4 tokens with "\n" and restores
  perplexity → their explicit conclusion is POSITION, not content, is load-bearing —
  the exact OPPOSITE of the lab's content-specific, run-specific anchor;
  (d) no norm-matched-noise control. So the nearest neighbor fails on all four of the
  lab's discriminating features.
- H2O / SnapKV / Scissorhands family: eviction during generation with task-level quality
  metrics; assume attention/importance scores predict prunability (Scissorhands'
  "persistence of importance" hypothesis). The lab's eval-time-utility ≠
  generation-time-prunability inversion directly contradicts that assumption and appears
  untested in this literature.
- LagKV (arXiv:2504.04704, 2025): motivated by reasoning quality degrading under
  aggressive KV eviction as length grows — systemic evidence that generation-time
  eviction damages generation, but importance-selection framing, prompt+generated
  entries mixed, no noise/content discrimination.
- KVFundaBench (arXiv:2502.01941, 2025): benchmarks degradation beyond perplexity;
  diagnostic but not mechanistic. SparK (2025): "recoverable" KV sparsity motivated by
  eviction damage — adjacent engineering acknowledgment.

## Q4 — attractor language for degeneration

- Holtzman et al. 2020 (arXiv:1904.09751): repetition loops via positive feedback /
  self-reinforcing likelihood ("likelihood trap") — implicit attractor, no dynamical
  formalism, no perturbation-entry mechanism.
- Wang et al., "Unveiling Attractor Cycles in LLMs" (ACL 2025 / arXiv:2502.15208) —
  verified against full text: the only explicit attractor-dynamics treatment found
  (2-period cycles as stable limit cycles, convergence evidence via conditioned
  perplexity / Vendi diversity collapse). BUT: regime is ITERATIVE PARAPHRASING (a map
  applied repeatedly), attractors arise from UNPERTURBED iteration (perturbations are
  used as escape attempts), no KV/history lesion, no self-vs-external judge
  dissociation. The lab's "attractor entered under mid-run perturbation, witnessed by
  self/clean judge gap" is a different and unclaimed claim.
- Fu et al. (AAAI 2021) theoretical repetition analysis; Xu et al. "Learning to Break
  the Loop" (arXiv:2206.02369) self-reinforcement mitigation — no dynamical-systems
  framing.
- Self-scored-fine / clean-judge-bad dissociation: nearest is Panickssery 2024
  (self-preference in evaluators) — eval-time, never used as a witness of an
  off-manifold state entered by intervention.

## Q5 — closest prior + verdict

Ranked nearest neighbors (each misses ≥2 of the lab's four components — mid-run
self-entry deletion, noise/content controls, free-run attractor, judge dissociation):

1. StreamingLLM (2309.17453) — same intervention family (generation-time cache eviction
   damage), but position-sinks not content anchors, eval-time perplexity not free-run,
   earliest not self-generated middle entries.
2. Wang et al. attractor cycles (2502.15208) — same dynamical vocabulary, wrong regime
   (iterative map, unperturbed entry, no lesion).
3. Zhang & Press snowballing (2305.13534) — same emphasis on self-history, opposite
   sign (self-history as error source, never as load).
4. Panickssery self-preference (2404.13076) — self-vs-other scoring gap, eval-time only.
5. Braverman (1906.05664) / Arora stable-entropy (2302.06784) — self-text divergence
   and entropy-band drift; no interventions.

VERDICT: GENUINELY NOVEL as a conjunction. The specific claim — "self-history is
load-bearing in a run-specific, content-specific way that static (eval-time) utility
does not predict, shown by mid-run deletion vs noise vs borrowed-content controls with
a free-run attractor witnessed by self/clean judge dissociation" — is unclaimed in the
2018-2026 literature scanned. Novelty risks to pre-empt in write-ups: (a) reviewers
collapsing it into StreamingLLM (rebut with position-vs-content + eval-time-vs-free-run
+ prompt-vs-self-generation); (b) collapsing it into exposure bias (rebut with the sign
inversion: their self-history is poison, ours is also the anchor); (c) the attractor
word (rebut: Wang et al.'s attractors are unperturbed iterative-map cycles, ours are
lesion-entered during single-pass generation). External-validity caveats remain ours:
single ~2.7M char-LM, B=8; clean judge is another net, not human. Cheapest external
check worth registering: replicate the vzero-vs-noise arms on GPT-2-small mid-run.

## Sources

- Zhang & Press, How Language Model Hallucinations Can Snowball — https://proceedings.mlr.press/v235/zhang24ay.html ; https://arxiv.org/abs/2305.13534
- StreamingLLM / attention sinks — https://arxiv.org/abs/2309.17453 (full text: https://ar5iv.labs.arxiv.org/html/2309.17453)
- Wang et al., Unveiling Attractor Cycles in LLMs (ACL 2025) — https://arxiv.org/abs/2502.15208 ; https://aclanthology.org/2025.acl-long.624
- Arora et al., Stable Entropy Hypothesis — https://arxiv.org/abs/2302.06784
- Braverman et al., Calibration, Entropy Rates, and Memory — https://arxiv.org/abs/1906.05664
- Kalai et al., Why Language Models Hallucinate — https://arxiv.org/abs/2509.04664
- Holtzman et al., Neural Text Degeneration — https://arxiv.org/abs/1904.09751
- Ivgi et al., Repetition In Repetition Out (NeurIPS 2023) — https://arxiv.org/abs/2310.10226
- Xu et al., Learning to Break the Loop — https://arxiv.org/abs/2206.02369
- Panickssery et al., LLM Evaluators Recognize and Favor Their Own Generations — https://arxiv.org/abs/2404.13076
- Ackman et al., Self-Generated-Text Recognition — https://arxiv.org/abs/2410.02064
- Extreme Self-Preference in LLMs — https://arxiv.org/html/2509.26464v1
- Olsson et al., In-context Learning and Induction Heads — https://arxiv.org/abs/2209.11895
- McDougall et al., Copy Suppression — https://openreview.net/forum?id=5Hd6813x3U
- LagKV — https://arxiv.org/abs/2504.04704
- KVFundaBench — https://arxiv.org/abs/2502.01941
- Chroma, Context Rot — https://www.trychroma.com/research/context-rot
- A Systematic Study of Error Propagation in LLMs (bit-flips) — https://arxiv.org/html/2606.02430v1
