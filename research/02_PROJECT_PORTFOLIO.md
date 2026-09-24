# Project Portfolio

This file sketches future projects. None are implementation commitments. Each project must eventually instantiate the mechanism-card contract.

## Portfolio Logic

The projects should form a ladder:

1. find internal signatures;
2. prove causal control on one behavior;
3. map reliability and failure;
4. generalize cautiously;
5. turn the method into a reusable experimental discipline.

## Project A: Mechanism Card One

Objective: produce the first serious card for one narrow behavior.

Candidate behaviors:

- hallucination versus known-answer/refusal circuits;
- sycophancy under incorrect hints;
- evaluation awareness in benchmark-like transcripts;
- refusal bypass under coherence pressure;
- cross-lingual concept mediation;
- planning-ahead in constrained generation.

Design:

- choose one open model with good tooling support;
- define behavior labels before looking at internals;
- collect activations across positives, negatives, and ambiguous cases;
- compare sparse features, linear probes, activation directions, attribution paths, and natural-language explanations if available;
- pick a candidate signature only if it predicts held-out behavior;
- intervene with activation steering, feature suppression, patching, or edit-like surgery;
- require null directions, matched-frequency controls, and prompt-only controls.

Stop condition:

- success: local intervention changes the target behavior with measured side effects;
- informative failure: signature predicts behavior but does not steer it;
- dead path: signature fails on holdout or collapses into output/token confound.

## Project B: Control Surface Benchmark

Objective: compare instrument classes on the same behaviors.

Instrument classes:

- linear probes;
- contrastive activation directions;
- SAE features;
- transcoder features;
- attribution graph nodes;
- natural-language activation decoders;
- parameter-edit targets.

Core question:

> Which instrument best predicts a reliable causal intervention, not just a readable explanation?

Metrics:

- prediction AUC for behavior;
- intervention effect size;
- dose-response monotonicity;
- reversibility;
- off-target degradation;
- fluency;
- compute cost;
- human analysis time;
- failure interpretability.

Why it matters:

This is the experiment that can stop the project from becoming loyal to one fashionable method.

## Project C: Hidden-State Early Warning

Objective: test whether internal signatures detect failures before output-only monitoring.

Target failures:

- hallucination onset;
- unsafe compliance;
- benchmark/evaluation awareness;
- agentic coding risk;
- chain-of-thought performativity;
- refusal collapse;
- overconfident uncertainty.

Design:

- pair production-style traces with activation reads at selected layers;
- compare output-only monitors, trace monitors, probes, SAE features, and steering-vector coordinates;
- preregister lead-time: how many tokens or steps before visible failure must the hidden signal fire?

Product thesis:

If hidden-state signals do not beat trace/output observability, this project has weaker market value. If they do, it becomes a bridge between interpretability and monitoring.

## Project D: Reliability Atlas

Objective: map where control surfaces break.

Axes:

- prompt paraphrase;
- language;
- topic;
- task difficulty;
- model family;
- model scale;
- instruction tuning versus base model;
- temperature;
- context length;
- adversarial prompting;
- distribution shift.

Output:

A reliability atlas is a table of "works here, breaks here, side effects here" for one mechanism. It is more valuable than a single impressive intervention.

## Project E: Mechanism Diffing

Objective: compare two related models or checkpoints and identify behavioral changes through internal deltas.

Use cases:

- base versus instruction-tuned;
- pre-safety versus post-safety;
- small versus large in same family;
- same model before and after finetuning;
- open models from different labs with similar tasks.

Claim shape:

- "This internal difference predicts this behavioral difference."
- "Intervening on the difference reduces or amplifies the behavior."
- "The result survives prompts not used to find the difference."

## Project F: Mechanism Editing

Objective: test durable edits to behavioral mechanisms, not just facts.

Targets:

- reduce sycophancy;
- increase calibrated refusal;
- reduce unsupported answer tendency;
- alter a model's response to evaluation-like contexts;
- preserve unrelated helpfulness.

Evaluation standard:

Borrow the ROME/MEMIT shape:

- efficacy;
- generalization;
- specificity;
- fluency;
- locality;
- regression suite;
- durability across context.

## Project G: Training-Health Diagnostics

Objective: determine whether internal geometry catches training or data problems earlier than loss curves.

Signals:

- feature sparsity drift;
- activation covariance spectrum;
- representation collapse;
- layerwise anisotropy;
- probe instability;
- abrupt changes in behavior-signature alignment;
- internal distribution shift under data mixture changes.

This should stay diagnostic until it predicts intervention. The first goal is "warn earlier than loss," not "fix training."

## Project H: Mechanism-Carrying Distillation

Objective: test whether a discovered mechanism can be taught or preserved during distillation.

Question:

> Can a student inherit a control surface, not just the teacher's outputs?

Design:

- teacher has a validated mechanism card;
- train or distill student under several regimes;
- measure whether behavior transfers;
- measure whether the internal signature transfers;
- test whether the same intervention works in the student;
- include output-only distillation as a control.

Why this matters:

If mechanisms can be transferred, interpretability becomes part of model development. If not, mechanisms may be model-local diagnostic objects.

## Project I: Cross-System Validation

Objective: test only the strongest LLM mechanism claims outside ordinary transformer LLMs.

Targets:

- vision-language models;
- code agents;
- recurrent or state-space language models;
- diffusion or world models;
- biological neural data only after the LLM mechanism is strong.

Rule:

Non-LLM validation is a late universality gate, not a place to start.

## Project J: Mechanism Operations

Objective: define what a future product would actually do.

Potential product primitives:

- mechanism card registry;
- hidden-state failure monitor;
- intervention sandbox;
- side-effect scanner;
- model-diff report;
- reliability atlas;
- experiment preregistration templates;
- red-team harness for mechanism claims.

Commercial wedge:

Organizations already buy evals and observability. The new value is "hidden-state evidence that predicts or fixes a failure before it becomes a visible incident."
