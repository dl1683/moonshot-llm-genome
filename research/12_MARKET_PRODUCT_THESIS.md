# Market And Product Thesis

This project is not a startup plan yet. But the research has to know what practical demand would make it matter.

## Core Thesis

Current AI tooling largely monitors outputs, traces, prompts, eval scores, and user reports. Interpretability tools inspect hidden states. The missing product category is:

> hidden-state evidence that predicts or controls a model failure better than output-only systems.

If that sentence is false, the project may still be scientifically valuable, but its market value is weaker.

## Buyer Categories

### Frontier Model Labs

Need:

- understand model changes between checkpoints;
- detect dangerous capabilities or behavior shifts;
- debug safety training regressions;
- decide whether an internal feature should be monitored during training;
- produce internal evidence for deployment decisions.

Why they might care:

They already run evals. A hidden-state mechanism is useful only if it catches something evals miss or explains why a regression happened.

### Agent Deployers

Need:

- reduce high-cost agent failures;
- monitor hidden risk in long traces;
- catch when an agent is about to take an unsafe action;
- distinguish uncertainty from overconfident fabrication;
- debug why prompting changes fail.

Why they might care:

Agents create long trajectories where visible failure may arrive late. Hidden-state early warning could be valuable if it produces lead time.

### Regulated Or High-Stakes Users

Need:

- auditability;
- evidence of risk controls;
- documented failure modes;
- model-change reports;
- defensible governance.

Why they might care:

Mechanism cards could become stronger internal technical evidence than generic "the eval passed" reports.

### Interpretability Teams

Need:

- stricter claim standards;
- shared artifact formats;
- null-control templates;
- reliability atlases;
- negative-result libraries.

Why they might care:

The mechanism-card contract can become a research standard.

## Competitor Categories

### Feature Browsers

Strength:

- make hidden features visible and searchable.

Weakness:

- browsing does not prove control or practical value.

Project response:

- use browsers as inputs, not products.

### Evals And Red-Team Platforms

Strength:

- align with buyer workflows;
- produce pass/fail deployment evidence.

Weakness:

- often output-only;
- may miss hidden evaluation awareness or emerging failure states.

Project response:

- compare hidden signatures against evals, not apart from them.

### Observability Platforms

Strength:

- production traces, dashboards, incidents, and regression tracking.

Weakness:

- external trajectory focus;
- limited access to internal activations.

Project response:

- future product should integrate with trace observability rather than compete head-on.

### Model-Editing And Steering Tools

Strength:

- direct behavior changes.

Weakness:

- side effects, brittle locality, and weak failure maps.

Project response:

- require reliability atlases and side-effect budgets.

## Product Sequence

### Product 0: Mechanism Card Corpus

Not software first. A credible corpus of mechanism cards and failed cards.

Value:

- proves the standard;
- gives examples for buyers and researchers;
- builds trust through negative results.

### Product 1: Mechanism Claim Workbench

Internal research tool:

- preregister a target behavior;
- attach prompt splits;
- compare probes/directions/features;
- generate null-control checklist;
- draft a mechanism card.

Value:

- makes rigorous interpretability faster.

### Product 2: Hidden-State Monitor

Deployment-adjacent tool:

- track validated hidden signatures during model use;
- alert when a failure signature appears;
- compare with output/trace monitors;
- produce incident evidence.

Value:

- catches failures earlier or with better specificity.

### Product 3: Intervention Sandbox

Controlled environment:

- test activation steering or feature suppression;
- scan side effects;
- compare prompt-only and policy-only alternatives;
- produce reliability atlas.

Value:

- helps model teams decide whether an internal lever is worth using.

## What Not To Build

- generic dashboard before a validated mechanism;
- another feature browser;
- a claim that all hidden features are deployable controls;
- a compliance wrapper without technical substance;
- a steering API that hides side effects.

## Commercial Kill Criteria

The market thesis weakens if:

- hidden signatures do not beat output/trace monitors;
- interventions are less effective than prompting;
- side effects are too broad;
- buyers cannot access activations;
- model providers lock down the required internals;
- feature interpretation remains too labor-intensive.

## Research-To-Market Bridge

Every practical card should answer:

- Who would use this?
- What decision changes?
- What existing tool is beaten?
- What failure becomes visible earlier?
- What side effect budget is acceptable?
- What would make this unsafe to deploy?

If a card cannot answer these, it remains science, not product.
