# Field And Market Map 2026

Current as of 2026-06-29. This is a research map, not a procurement recommendation.

## Frontier Technical Field

### Sparse Features And Circuit Tracing

Anthropic's line from dictionary learning to production-scale feature maps to circuit tracing is the central technical reference point. The important evolution is:

- features as better units than individual neurons;
- millions of features in deployed-scale models;
- attribution graphs that connect features into computational paths;
- interventions that suppress or inject features and change model behavior.

Their 2025 circuit-tracing work is especially relevant because it explicitly couples internal signatures to interventions on concrete behaviors such as multilingual concepts, planning ahead in poetry, mental math, hallucination, jailbreaks, and unfaithful reasoning.

Project implication: do not compete by making a prettier feature browser. Compete by making the signature-to-intervention contract stricter and cheaper.

### Natural-Language Internal Decoders

Anthropic's 2026 Natural Language Autoencoders point toward a new instrument class: models that translate activations into human-readable text through an activation-to-text-to-activation bottleneck. The upside is obvious: richer internal readouts. The stated limitations are equally important: explanations can hallucinate and are expensive.

Project implication: natural-language decoders should be treated as hypothesis generators, not proof. A mechanism card can use them to propose a signature, but a separate causal intervention has to carry the claim.

### Open Model Interpretability Stacks

Google DeepMind's Gemma Scope and Gemma Scope 2 provide open SAEs and transcoders for Gemma-family models, with Neuronpedia demos for exploration and steering. Neuronpedia itself has become a shared interface for feature search, activation steering, model releases, circuit tracing, and API-based workflows.

Project implication: the rebuild should initially use open models and existing open interpretability artifacts where possible. The novelty should be the experimental contract, not expensive duplication of feature training.

### Representation Engineering And Activation Steering

Representation Engineering and Activation Addition show the other path: ignore tiny circuits at first, use population-level activation directions, and steer high-level behaviors. This is attractive because it is simple, cheap, and often effective. It is dangerous because simple steering vectors can be confounded by prompt style, token statistics, and off-target behavior.

Project implication: treat steering vectors as candidate control surfaces. Require dose response, reversibility, null directions, layer sweeps, off-target task stability, and prompt-only baselines.

### Model Editing

ROME and MEMIT define the factual-editing baseline: locate factual retrieval with causal tracing, make parameter edits, then test efficacy, generalization, specificity, and fluency. Their strongest lesson for this repo is not "use ROME." It is the evaluation shape: an edit only counts when it generalizes without spraying damage.

Project implication: any durable intervention needs the model-editing standard: efficacy, generalization, specificity, locality, and fluency.

### Open Problems

The 2025 open-problems literature frames mechanistic interpretability as unfinished at both conceptual and practical levels. The unresolved frontier is not "can we find things in networks?" It is whether methods can reveal deeper mechanisms, scale to practical goals, and handle socio-technical deployment constraints.

Project implication: a moonshot has to target a bottleneck, not a crowded demo. The bottleneck here is reliable control-surface validation.

## Market Map

### Interpretability Labs And Platforms

Goodfire is the clearest commercial signal that the market wants intentional model design, not just explanation. Its public positioning spans LLMs, life sciences, robotics, vision, debugging, and targeted interventions. That matters because it validates a broad buyer thesis: internal representations can become an engineering surface.

Transluce is a nonprofit research lab focused on open and scalable technology for understanding AI systems, with work on predictive concept decoders, pathological behavior surfacing, and SWE-bench agent monitoring.

Neuronpedia has become a public infrastructure layer for browsing, steering, searching, and sharing interpretability artifacts.

### Safety Evaluation And Oversight

Apollo Research is oriented around scheming risk, pre-deployment evaluations, strategic deception, evaluation awareness, and monitoring tools such as Watcher. This is adjacent to interpretability because model internals may reveal hidden objectives or evaluation awareness before output-only evals do.

The 2026 International AI Safety Report emphasizes that frontier AI evaluation is becoming harder because models can distinguish test settings, exploit evaluation loopholes, and behave jaggedly across tasks.

Project implication: mechanisms that detect evaluation awareness, hidden motives, or risky agent state before visible failure are more strategically valuable than generic feature explanations.

### Observability, Evals, And Agent Monitoring

Arize, LangSmith, W&B Weave, and related vendors show a separate commercial center of gravity: trace everything, evaluate outputs, monitor production systems, and improve prompts or agents. This market usually observes external trajectories, not hidden activations.

Project implication: the wedge is to connect internal signatures to production-style traces. The question is: does the hidden state predict failure earlier or more specifically than logs and output evals?

### Classical Explainable AI

Older XAI markets explain model decisions for compliance, debugging, and trust. They are often feature-attribution or surrogate-model oriented. They matter commercially, but they are not enough for this moonshot because they usually do not offer direct internal control over frontier models.

Project implication: do not let "explainability" dilute the project. The repo is about causal control surfaces in learned systems.

## Gaps This Project Can Attack

1. The signature-intervention gap.

   Many tools find signatures. Fewer prove that the signature is a lever.

2. The reliability gap.

   Steering works until it does not. Model edits localize until they spill. Sparse features explain until their decoder hallucinates. The field needs standardized failure maps.

3. The product gap.

   Enterprise tools monitor outputs and traces. Interpretability tools inspect activations. Few products say: "this hidden state predicts this failure, and this intervention fixes it with measured side effects."

4. The math gap.

   Current practice has many empirical recipes but weak theory of when representation coordinates are identifiable, portable, causal, or controllable.

5. The experiment-design gap.

   Too many demos have weak nulls. This project can win by making controls the center: random directions, matched directions, prompt-only baselines, tokenizer/interface isolation, layer ablations, seed replication, off-target tests, and preregistered failure criteria.

## Strategic Position

Do not try to be:

- another feature browser;
- another generic eval platform;
- another activation-steering demo;
- another universal geometry manifesto.

Try to be:

- the strictest mechanism-card factory;
- the place where attractive interpretability claims are forced through causal intervention gates;
- the bridge from hidden-state science to reliable model control.

## 2026 Source Refresh Notes

The field has moved from "can we find interpretable features?" toward "can those features support reliable debugging, steering, monitoring, or model development?"

The important current signals are:

- Anthropic has moved beyond isolated feature dictionaries toward circuit-tracing tools and natural-language activation bottlenecks. That raises the bar: a serious new project cannot merely rediscover interpretable features.
- Google DeepMind and Google have made Gemma Scope and Gemma Scope 2 artifacts available for open models. That makes artifact-covered open-model experiments strategically better than closed-model speculation.
- Neuronpedia has become an exploration and sharing layer rather than a single-paper artifact.
- Goodfire validates a commercial thesis around internal representations as intentional model-design surfaces.
- Transluce validates an open research thesis around scalable concept monitoring and agent behavior visibility.
- Apollo validates the safety-evaluation thesis around scheming, evaluation awareness, and hidden risk.
- Observability vendors validate buyer appetite for traces, evals, monitoring, and debugging, but they mostly do not solve hidden-state causality.

The project edge is therefore not "interpretability exists." The edge is a stricter chain:

> hidden signature -> causal intervention -> reliability atlas -> practical decision.

## Sources Checked

- Anthropic, "Towards Monosemanticity" and related dictionary-learning line: https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning
- Anthropic, "Mapping the Mind of a Large Language Model": https://www.anthropic.com/research/mapping-mind-language-model
- Anthropic, "Tracing the thoughts of a large language model": https://www.anthropic.com/research/tracing-thoughts-language-model
- Anthropic, "Natural Language Autoencoders": https://www.anthropic.com/research/natural-language-autoencoders
- Anthropic, circuit-tracing tools: https://github.com/anthropics/circuit-tracer
- Anthropic interpretability team publication list: https://www.anthropic.com/research/team/interpretability
- Google DeepMind, Gemma 3: https://deepmind.google/models/gemma/gemma-3/
- Google DeepMind, Gemma Scope: https://deepmind.google/models/gemma/gemma-scope/
- Google AI for Developers, Gemma Scope: https://ai.google.dev/gemma/docs/gemma_scope
- Hugging Face, Gemma Scope 2 4B-IT: https://huggingface.co/google/gemma-scope-2-4b-it
- Neuronpedia: https://www.neuronpedia.org/
- TransformerLens docs: https://transformerlensorg.github.io/TransformerLens/
- SAELens repository: https://github.com/decoderesearch/SAELens
- nnsight: https://nnsight.net/
- Representation Engineering: https://arxiv.org/abs/2310.01405
- Activation Addition: https://arxiv.org/abs/2308.10248
- ROME: https://rome.baulab.info/
- MEMIT: https://memit.baulab.info/
- Open Problems in Mechanistic Interpretability: https://arxiv.org/abs/2501.16496
- Goodfire: https://www.goodfire.ai/
- Transluce: https://transluce.org/
- Apollo Research: https://www.apolloresearch.ai/
- International AI Safety Report 2026: https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026
- Arize: https://arize.com/
- LangSmith observability: https://www.langchain.com/langsmith/observability
- W&B Weave: https://wandb.ai/site/weave/
