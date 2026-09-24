# Rebuild Doctrine

## One Sentence

This project exists to turn hidden model computation into a measurable and controllable object, not to collect interpretability artifacts.

## Starting Position

The repo is now a scratch rebuild. The root README defines the live contract:

- find a behavior;
- find an internal signature;
- intervene on the signature;
- prove the effect under strong controls;
- record failure modes without hiding them.

Nothing older than this rebuild gets to count as evidence. A past result can only be a warning about experimental design.

## Moonshot Fit

The larger AI Moonshots philosophy is not "make a useful dashboard." It is to ask whether a scientific law or engineering primitive is hiding inside modern AI systems and then force it to survive contact with measurement.

For this repo, the moonshot is:

> learned intelligence may contain stable control surfaces that are discoverable, editable, portable in limited ways, and useful for making models safer or more capable.

That sentence has to be treated as a hypothesis, not a slogan.

## Core Doctrine

1. Mechanisms beat metrics.

   A benchmark win without a mechanism is not the thing. A mechanism is also not a screenshot of a feature dashboard. The unit of progress is a causal chain: behavior, internal state, intervention, behavioral change, locality, failure.

2. Control beats explanation.

   If an explanation cannot predict an intervention, it is a story. The project should prefer weak explanations with strong intervention predictions over beautiful explanations with no causal bite.

3. Reliability beats surprise.

   The field has many surprising demos. The project should bias toward boring but hard tests: dose response, reversibility, off-target behavior, holdout prompts, cross-seed replication, and negative controls.

4. Diagnostics come before surgery.

   Geometry, sparse features, probes, attribution graphs, and natural-language decoders are instruments. Their first job is to measure. They become levers only after they predict interventions.

5. Universality is a late gate.

   A pattern shared across models is interesting. It is not a universal law until it survives tokenizer, architecture, training distribution, scale, and non-language validation challenges.

6. Failed interventions are valuable.

   A clean failure with strong controls is more useful than a weak success. The field needs maps of where steering, editing, sparse features, probes, and geometry do not carry causal information.

## What This Project Should Do Better Than The Field

The current market is good at three things:

- microscopes: finding and visualizing features;
- monitors: scoring traces, outputs, and agent failures;
- knobs: steering or editing behavior in scoped settings.

The opportunity is to integrate them into a discipline of control-surface discovery:

> A proposed internal feature only matters after it predicts a local, repeatable, behaviorally meaningful intervention that survives nulls and admits a practical use.

That is the difference between "interpretability as analysis" and "interpretability as engineering."

## The Product Of The Research

The first product is not software. It is a mechanism card that another serious researcher could attack.

The second product is a library of dead ends: controls that killed attractive hypotheses.

The third product is a method for deciding which internal signatures deserve more compute.

Only after those exist should the project become a platform, benchmark, or training system.

## Non-Goals

- no grand theory before the first surviving mechanism card;
- no inherited benchmark numbers;
- no "universal geometry" claim from representation similarity alone;
- no steering claim without off-target and fluency checks;
- no editing claim without generalization and specificity checks;
- no biology claim unless the LLM result is already strong enough to justify a non-LLM validation test;
- no implementation push while this repo is in doctrine mode.

## First Serious Mechanism Target

Pick one behavior that is narrow enough to test and important enough to matter:

- hallucination suppression versus known-answer activation;
- refusal bypass versus safety/coherence tension;
- sycophancy or motivated reasoning under incorrect hints;
- evaluation awareness in benchmark-like settings;
- cross-lingual conceptual mediation;
- planning-ahead in constrained generation;
- agentic coding risk detection.

For each candidate, the card must ask:

- What behavior changes?
- What internal signature predicts it before output?
- What intervention changes it?
- What null intervention fails?
- What off-target behaviors stay stable?
- What distribution shift breaks it?
- What practical decision would change if this card were true?

## Research Style

Think in public markdown first. Do not let context-window thoughts become the only source of design. Every important idea should end up in a file that can be reread and attacked.
