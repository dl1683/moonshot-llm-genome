# Expression-gap literature scan — RESEARCHER, 2026-09-25

Scope: the "installed knowledge that scores 92-96% on continuation batteries but never
surfaces in free generation" arc (novelty_inventory results #1/#2, experiment e055).
Question: is the transplant-at-divergence-token approach unpublished, and what is the
closest prior? Verdict up front: **yes, unpublished as a factual-recall depth-localizing
intervention** — but the blanket claim "zero interventional studies anywhere" needs one
softening (see Follow-ups, the relay-steering paper).

## Orgad and caveats

**Orgad, Toker, Gekhman, Reichart, Szpektor, Kotek, Belinkov — "LLMs Know More Than
They Show: On the Intrinsic Representation of LLM Hallucinations" (ICLR 2025,
arXiv:2410.02707, v4 May 2025).**

- Setup: linear probes (logistic regression) on MLP-output activations; models
  Mistral-7B / Mistral-7B-it-v0.2 / Llama3-8B / Llama3-8B-it; 10 tasks (TriviaQA,
  HotpotQA±ctx, NQ-ctx, movie-roles, Winogrande, Winobias, MNLI, Math, IMDB).
- Exact claims:
  1. Truthfulness signal is concentrated in the **exact-answer tokens** (probing the
     last exact-answer token gives AUC 0.85–0.95 on best cells; beats logit and P(True)
     baselines). Signal is strong right after prompt, dips, peaks at answer tokens.
  2. Probes trained on greedy activations predict a **sampling-consistency error
     taxonomy** (A=refuse, B=consistently-correct, C=consistently-wrong, D=two
     competing answers, E=>10 answers): AUC 0.81–0.90 for A/B/E but only **0.59–0.68
     for C and D**.
  3. Probe-based selection among 30 resampled answers beats greedy/random/majority,
     with the biggest gains (30–40 pts) on C2/D/E1 — cells where the model shows *no
     external preference* for the correct answer. This is their title claim: internal
     encoding of the right answer with consistent wrong generation.
- Caveats (from v4 full text + OpenReview discussion):
  - **Entirely probe-level. No logit lens, no activation patching, no causal
    interventions.** Authors state the probe is "not proposed here as an error
    mitigation strategy but rather as a diagnostic tool." The only quasi-intervention
    is using the probe to *select among* resamples (post-hoc readout, not a write).
  - OpenReview reviewers explicitly challenged the error-type-prediction claim; the
    weak C/D AUCs (0.59–0.68) corroborate that it is the paper's softest result — and
    C/D are precisely the expression-gap populations.
  - Probes **fail to generalize across datasets** (no universal truthfulness
    direction; "multifaceted" encoding). Deployment "with caution."
  - White-box open models only; QA with gold labels only; open-ended generation left
    to future work. The famous "entity-level consistency" is really repeated-sample
    consistency (same entity errs across paraphrases), not a mechanism claim.
- Relevance to us: their C2/D/E1 cells = our expression-gap population. They observed
  the gap, characterized it with probes, and never touched the network. e055 is the
  causal experiment their Discussion section implies but never runs.

## Follow-ups

- **Buckmann, Nguyen, Hill (Bank of England) — "Revealing economic facts: LLMs know
  more than they say" (arXiv:2505.08662, May 2025).** Linear probes on hidden states
  estimate/impute economic facts better than the model's own text outputs. Coins the
  term **"elicitation failure"** for the probe-vs-output gap. Probe-level only; no
  interventions; economics domain. Term is useful and unclaimed for our use.
- **"You Only Pass Once: Answering and Abstaining" (arXiv:2608.14465, Aug 2026) — the
  single most important prior for e055.** On Qwen2.5-7B sufficiency detection over
  RepLiQA: the residual stream encodes whether context is sufficient but generation
  fails to act on it ("the model knows, but does not say"): explicit permission to
  abstain still leaves 44/125 fabrications, while a difference-of-means direction from
  the same forward pass separates 124/125 (AUROC ≈0.9988). Interventions: read-only
  direction; **"relay steering"** pushes the internal verdict into speech (fidelity
  1.0 across 16 cells); trained write-access interventions (LoRA) collapse under
  domain shift (write-access ladder). Localization: causal onset consistently at
  **layer 19**; steering writes at mid-stack {12,16,20}; two interference types —
  downstream layers *overwriting* an injected verdict, and the write perturbing the
  read. Differences from us: (a) domain is abstention/sufficiency, not factual
  recall; (b) they inject a *learned direction*, not the model's own teacher-forced
  states; (c) goal is elicitation, not depth-localizing suppression of an installed
  fact. But they DO measure "where," so cite them and scope our novelty claim as
  "zero interventional studies of the *factual-recall* expression gap / of own-state
  transplant."
- **"Truth is Universal" (Nadler et al.)** — argues RLHF *widens* the elicitation gap
  between internal truth representations and outputs; lie-detection probes stay
  robust in aligned models. (Search-level confidence; verify before citing in a
  paper draft.)
- **"Do We Know What LLMs Don't Know?" (arXiv:2505.21701, ACL 2025)** — probes of
  knowledge gaps are brittle to elicitation method and prompt perturbation; cautions
  that probe→behavior transfer is unreliable. Motivates our battery-overstatement
  control (3.4× gap).
- **Yan & Jia — "Promote, Suppress, Iterate: How Language Models Answer One-to-Many
  Factual Queries" (arXiv:2502.20475, EMNLP 2025).** Mechanistic: models promote all
  candidate answers, then **suppress** already-generated ones; both attention and
  MLPs carry the suppression, layer-specific. Establishes that suppression circuits
  exist and are localizable — but for *repetition avoidance during enumeration*, not
  silent first-answer knowledge. Strong motivation citation for "suppression depth"
  being a real, addressable mechanism class.
- **"Knowing Before Saying" (Findings of ACL/EMNLP)** — representations encode CoT
  success before verbalization; read-only, same genre as Orgad.
- **FAR AI — "Preference Learning with Lie Detectors can Induce Deception" (Nov
  2025)** — training against probes yields probe evasion: a caveat that probe-visible
  knowledge can be driven deeper without disappearing. Matters for our claim that the
  knowledge is really *in* the network.
- medRxiv Sep 2025 "Probing Hidden States for Calibrated, Alignment-Resistant
  Confidence" — continues the hidden-state-probing line citing Orgad; probe-level.

## Interventions that worked

- **ITI (Li et al., NeurIPS 2023, arXiv:2306.03341)** — shift attention-head
  activations along a probe-derived truthful direction; LLaMA-7B TruthfulQA MC2
  32.5%→65.1%. Critiques: directions are probe-derived, possibly non-causal (CORAL,
  arXiv:2602.06022); TruthfulQA-MC is gameable (Turntrout: blind decision trees hit
  ~79.6%); limited cross-model/task generalization (TruthFlow positions itself as the
  fix). Scale tested: 7B-class, one benchmark family. No localization of suppression.
- **DoLa (Chuang et al., ICLR 2024, arXiv:2309.03883)** — contrast "premature" vs
  "mature" layer logits per token (JSD-based layer pick); improves TruthfulQA
  (largest on MC scoring), some FACTOR/ARC gains, training-free. Critiques (SH2
  arXiv:2410.xxxx; "Why contrastive decoding fails" arXiv:2504.10020) argue part of
  the gains are multiple-choice scoring artifacts. Crucially, DoLa *assumes* late
  layers pollute early-layer facts but **never measures where knowledge dies** —
  layer choice is per-token automatic, not an anatomical finding.
- **Contrastive Decoding (Li et al., ACL 2023)** — expert/amateur logit contrast;
  coherence/reasoning gains (O'Brien & Lewis: GSM8K), sensitive to amateur scale;
  not knowledge-specific; no internal localization.
- **Relay steering (arXiv:2608.14465)** — the only intervention found that both
  flips silent internal signal into generation AND characterizes where the
  suppression/overwrite happens (mid-stack, layer-19 onset). See above.
- **TruthPrInt (CVPR 2025)** and the LVLM head-intervention family (Yang ICLR 2025;
  Sarkar EMNLP 2025) — truthful-direction and per-query head interventions in
  vision-language models. Vision domain; method-transferable intuition (early
  intervention before divergence compounds).
- Common denominator: every method injects *externally derived* directions/logit
  contrasts. None transplants the model's **own teacher-forced states** into its
  free run, and none reports a depth-survival curve of such a transplant.

## The gap we can own

1. **Own-state transplant at the divergence token (e055 core).** Patching
   teacher-forced residuals at the token where free-run first diverges, sweeping
   depth, to localize suppression: no counterpart found. Nearest methodological
   neighbor is Heimersheim & Nanda's activation-patching best practices
   (arXiv:2309.16042) — but their clean/corrupt framing is corruption-repair, not
   teacher-forced→free-run knowledge expression. Nearest *conceptual* neighbor is
   relay steering (2608.14465), which localizes onset depth for a learned direction
   in the abstention domain. Our differentiators: the fact domain, own-state (not
   learned-direction) patching, and the explicit question "at what depth does the
   free-run context destroy the teacher-forced state's ability to produce the
   token."
2. **Sub-argmax prior of the correct token.** No dedicated paper found on the
   correct answer persistently sitting below argmax during free run while winning
   under teacher forcing (the closest vocabulary is "logit margin" work, which is
   about confidence, not latent knowledge). Claimable.
3. **Suppression depth vs regime/seed/dose.** Orgad's taxonomy is single-family and
   correlational; nothing tracks whether suppression depth moves with training
   regime. Matches our T015-T019 findings; no prior to collide with.
4. **Teacher-forcing gap at the knowledge level.** 2025-26 work is behavioral or
   theoretical: exposure-bias/self-recovery debates, distillation gains attributed
   to reduced exposure bias (ICML 2026 "Bridge-Garden"), teacher-forced log-prob
   concavity theory (OpenReview EGz8InJz6F). Nobody connects TF/free-run *state*
   differences to knowledge expression at a specific token.
5. **Watch-outs for the write-up.** (a) Cite Yan & Jia + relay-steering so reviewers
   don't surface them as "missed prior art"; scope claims to factual recall.
   (b) Orgad's error-type claims drew reviewer fire at AUC 0.59–0.68 — our battery
   should pre-register the discrimination metric. (c) "Zero interventional studies
   anywhere" (novelty_inventory #2) should become "zero interventional studies of
   factual-recall expression" to survive review.

## sources

- Orgad et al. ICLR 2025: https://arxiv.org/abs/2410.02707 (HTML v4), OpenReview
  forum KRnsX5Em3W, project https://llms-know.github.io
- Buckmann et al.: https://arxiv.org/abs/2505.08662
- You Only Pass Once: https://arxiv.org/html/2608.14465v1
- Do We Know What LLMs Don't Know: https://arxiv.org/abs/2505.21701
- Promote, Suppress, Iterate: https://arxiv.org/abs/2502.20475 (EMNLP 2025)
- ITI: https://arxiv.org/abs/2306.03341 ; critique CORAL:
  https://openreview.net/forum?id=1piRnQJhLc
- DoLa: https://arxiv.org/abs/2309.03883 ; CD critique:
  https://arxiv.org/html/2504.10020v2 ; SH2 self-highlighted hesitation (arXiv Oct 2024)
- Activation patching best practices: https://arxiv.org/abs/2309.16042
- Truth is Universal (Nadler et al.), FAR AI probe-evasion (Nov 2025), Knowing
  Before Saying (ACL Findings), TruthPrInt (CVPR 2025), Yang ICLR 2025 LVLM heads
  (openreview Bjq4W7P2Us), Sarkar EMNLP 2025 (aclanthology 2025.emnlp-main.631) —
  search-level; verify before citing in submission drafts.
