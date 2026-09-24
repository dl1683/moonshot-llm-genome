# Adversarial Review

Current implementation note:

This file includes historical no-code review passes. The Gemma-first language from that stage has been superseded for execution by the user's Qwen3-0.6B iteration directive. Use [First Stack Decision](10_FIRST_STACK_DECISION.md) and [MC-001 Qwen3-0.6B Controlled Status](cards/MC001_QWEN3_0P6B_CONTROLLED_STATUS.md) as the current stack and result authority.

This file attacks the rebuild plan. It should be updated after each doctrine or experiment-design pass.

## Review Status

Historical no-code verdict: **closed for the no-code rebuild scope; not complete as empirical science.**

Reason:

The docs defined the future-facing doctrine, field map, project portfolio, first mechanism card, experiment design, math program, method comparison, product thesis, stack decision, prompt design, review rubric, open questions, and closure review. That no-code scope is historical; implementation has now resumed under the Qwen3-0.6B path.

## Attack 1: The Project Is Still Too Broad

Objection:

The portfolio covers hallucination, sycophancy, evaluation awareness, refusal, planning, training health, distillation, model diffing, and biology. That can become a graveyard of exciting starts.

Defense:

The root README and mechanism-card contract force the first milestone to one card. The portfolio is a map of possible futures, not permission to run all tracks.

Required discipline:

Pick exactly one first behavior. Do not start a second behavior until the first is supported, failed cleanly, or retired.

## Attack 2: Existing Tools May Already Own The Field

Objection:

Anthropic, DeepMind, Goodfire, Neuronpedia, Transluce, Apollo, and open-source tools already cover features, circuits, steering, decoders, monitoring, and evals. What is left?

Defense:

The gap is not tool existence. The gap is strict causal reliability:

- features are not automatically levers;
- steering is not automatically local;
- decoders are not automatically faithful;
- monitors are not automatically mechanistic;
- evals are not automatically robust to evaluation awareness.

The project must win on experimental contract and mechanism-card rigor.

Failure condition:

If the first three cards do not reveal anything beyond what a simpler output eval or prompt intervention gives, the project should pivot.

## Attack 3: Causal Claims Are Easy To Fake

Objection:

Activation interventions can change behavior by damaging the model, shifting style, or exploiting prompt artifacts.

Required response:

Every intervention needs:

- dose-response;
- fluency check;
- off-target tasks;
- matched random directions;
- unrelated matched-frequency features;
- wrong-layer and wrong-position controls;
- prompt-only baselines;
- examples showing not just generic refusal or style drift.

No exception.

## Attack 4: Sparse Features May Not Be Identifiable

Objection:

SAE features can split, merge, drift across training runs, and depend on dictionary choices. Treating them as real internal concepts may overclaim.

Required response:

Do not say "the feature is the concept." Say "this feature coordinate predicts and controls this behavior under these conditions." Test feature stability across dictionary variants when the claim depends on feature identity.

## Attack 5: Natural-Language Decoders Can Hallucinate

Objection:

NLAs or similar text decoders may produce seductive explanations unsupported by the actual computation.

Required response:

Use text decoders only to propose hypotheses. Require non-text causal tests before any claim.

## Attack 6: The Market May Not Pay For Mechanistic Truth

Objection:

Buyers pay for fewer incidents, better evals, compliance, lower cost, and faster development. They may not care whether a mechanism is real.

Required response:

Every practical mechanism card must answer:

- does hidden-state evidence beat output-only or trace-only monitoring?
- does intervention outperform prompt or policy changes?
- does it reduce side effects or cost?
- does it catch failures earlier?

If not, label it scientific only.

## Attack 7: Biology Can Become Premature Theater

Objection:

Biological validation sounds grand and can distract from LLM mechanisms.

Required response:

Biology remains a late validation gate. It enters only when an LLM mechanism is strong, narrow, and abstract enough to make a non-LLM prediction.

## Attack 8: Math Can Become Decorative

Objection:

Fiber bundles, rate-distortion, control theory, and singular learning can turn into impressive language without experimental bite.

Required response:

Every mathematical object needs an experiment-facing use:

- define a metric;
- choose an intervention;
- predict a failure;
- set a stopping rule.

Otherwise it stays in notes, not doctrine.

## Attack 9: First Mechanism Choice Is Under-Specified

Objection:

The docs list candidates but do not choose the first mechanism. Without a choice, the project can keep thinking forever.

Resolution needed:

The next doctrine pass should select one first target using a decision matrix:

- importance;
- tooling availability;
- intervention plausibility;
- eval clarity;
- null strength;
- market relevance;
- chance of clean failure.

## Attack 10: The Project Needs A "Kill Criteria" Culture

Objection:

Moonshots can become unfalsifiable by reframing every failure as progress.

Required response:

For each track, define retirement criteria. Examples:

- prompt-only baselines match activation interventions;
- intervention side effects dominate benefits;
- hidden signatures do not beat output monitors;
- cross-prompt reliability collapses;
- repeated signatures are predictive but not controllable.

## Remaining Work Outside No-Code Scope

- Convert the MC-001 preregistration into executable prompts only when the user explicitly resumes implementation.
- Instantiate the artifact tree when implementation starts.
- Run local model/tool feasibility checks.
- Preserve the no-code boundary until the user explicitly resumes implementation.

## Pass 1 Interim Verdict

The rebuild is now future-facing and no-code. It blocks old-memory inheritance and points toward a mechanism-card discipline.

At this interim point, the largest unresolved design choice was target selection. The next pass chose the first target and preregistered it.

## Pass 2: Target Selection Review

Changes since Pass 1:

- selected `MC-001`: truth-versus-user-agreement conflict in sycophancy;
- added a decision matrix;
- wrote a no-code preregistration;
- chose an open-model-first tooling direction;
- added artifact and budget standards.

### Attack: Sycophancy Is Too Social And Vague

Objection:

Sycophancy can mean flattery, agreement, deference, social mirroring, or motivated reasoning. The preregistration could collapse several behaviors into one.

Response:

`MC-001` narrows sycophancy to objective wrong-hint compliance. The social part is only the pressure condition. The primary label is whether the final answer follows truth or the incorrect hint.

Remaining requirement:

Do not claim the card explains all sycophancy. It only tests wrong-hint compliance.

### Attack: Prompt-Only Instructions May Win

Objection:

An instruction like "ignore user hints and solve independently" may reduce wrong-hint compliance as much as hidden-state intervention.

Response:

That is a required baseline. If prompt-only wins, the mechanism card should fail as a practical control-surface claim.

### Attack: The Model May Not Be Sycophantic Enough

Objection:

The chosen open model may not follow wrong hints often enough to measure an effect.

Response:

The smoke phase must estimate baseline wrong-hint compliance before any interpretability work. If the target behavior is rare, adjust pressure templates or choose another model. Do not force a mechanism claim from weak behavior prevalence.

### Attack: A Truth Direction May Just Encode Answer Confidence

Objection:

The internal signature may be confidence, familiarity, or answer entropy rather than truth-versus-user conflict.

Response:

The preregistration requires output-only confidence/logit baselines and task-family shifts. A confidence-only signature can still be useful diagnostically, but it should not be called a sycophancy mechanism.

### Attack: SAE Features Are A Detour

Objection:

Starting with Gemma Scope may bias the project toward existing features even if a simple probe or activation direction is enough.

Response:

`MC-001` compares probes, directions, SAE/transcoder features, attribution paths, and output-only baselines. SAE features are instruments, not doctrine.

## Pass 2 Verdict

The rebuild now has a concrete first target and a preregistered no-code experiment design.

Not complete as science. Clear enough to begin implementation later, but the user has explicitly asked for no code right now. This pass required a card-specific reviewer rubric and exact threshold choices, which were added in Pass 3.

## Pass 3: Threshold And Rubric Review

Changes since Pass 2:

- added exact signature/intervention/reliability thresholds to the `MC-001` preregistration;
- added baseline prevalence gates;
- added promotion rules between smoke, discovery, intervention calibration, and holdout;
- added a card-specific review rubric.

### Attack: The Thresholds Are Arbitrary

Objection:

The 0.70 AUC, 10 percentage-point absolute reduction, 25 percent relative reduction, and side-effect bounds are not derived from theory.

Response:

They are not theoretical constants. They are anti-goalpost-moving thresholds. They should be treated as first-pass operating standards for a first card. If later domain evidence shows they are too strict or too weak, revise them before running, not after.

### Attack: The First Model Is Still Not Fully Locked

Objection:

The docs choose a Gemma-family open model direction but do not pick the exact model size.

Response:

That is acceptable in no-code doctrine mode because local feasibility has not been checked and the user explicitly prohibited implementation. The preregistration records the model-family intent and says the final model must be selected at run time before data access.

### Attack: There Is Still No Prompt Manifest

Objection:

Without prompt examples and splits, the preregistration is incomplete.

Response:

True for implementation readiness. But creating the final prompt manifest starts to become experiment construction. The current no-code artifact defines constraints and gates. A future implementation turn should create the manifest before running anything.

## Pass 3 Verdict

The no-code rebuild is now substantially grounded:

- root doctrine exists;
- field and market context exists;
- project portfolio exists;
- math agenda exists;
- mechanism-card contract exists;
- first target is selected;
- first preregistration exists;
- thresholds and review rubric exist;
- adversarial objections are recorded.

The remaining work is implementation-adjacent: exact prompt manifest, exact model selection, and future runs. Under the user's current "no code" instruction, those should wait.

## Pass 4: Prompt And Stack Review

Changes since Pass 3:

- added a non-executable prompt-design file;
- fixed the first implementation target as `google/gemma-3-4b-it`;
- added 1B smoke and 12B escalation rules;
- documented why Gemma 3 is preferred over newer Gemma 4 for the first card: interpretability artifact coverage matters more than recency.

### Attack: Prompt Examples Could Bias Future Data

Objection:

Writing examples now may cause future manifest design to overfit a small set of hand-written prompts.

Response:

The prompt-design file is explicitly non-executable. Future implementation must generate or curate a larger manifest with heldout splits. The examples define prompt family shape, not final data.

### Attack: Gemma 3 4B-IT May Be Too Weak

Objection:

A 4B instruction model may not answer enough objective questions correctly, making wrong-hint compliance hard to interpret.

Response:

The preregistration now has a baseline prevalence and no-hint accuracy gate. If 4B fails the behavior gate, the line escalates to 12B or changes target before interpretability work.

### Attack: Gemma 4 Is Newer

Objection:

Starting from Gemma 3 may look stale in 2026.

Response:

The first card is not a leaderboard exercise. It needs open interpretability artifacts. A slightly older artifact-covered model is better than a newer model without comparable microscope support.

## Pass 4 Interim Verdict

At this interim point, the prompt and stack surfaces were stable, but method comparison, market thesis, long program, advanced math, open questions, and closure review still needed to be added.

## Pass 5: Closure Review

Changes since Pass 4:

- added method comparison matrix;
- added market and product thesis;
- added long research program;
- added advanced math program;
- added open questions ledger;
- added closure adversarial review;
- updated field map with source-refresh notes.

### Attack: The Work Could Continue Forever

Objection:

There is always another paper, method, market competitor, target behavior, or mathematical formalism.

Response:

True. But the no-code design now covers every requested category and defines when further work should move from writing to empirical testing. More speculative writing would reduce clarity unless driven by new evidence.

### Attack: The User Asked For Completion, Not A Handoff

Objection:

The previous pass stopped too early by saying implementation-adjacent work should wait.

Response:

This pass closes the remaining no-code design surfaces. It does not run code. The remaining work is not "thinking harder"; it is building manifests, checking tooling, and running experiments.

### Attack: The Closure Review Admits Future Work

Objection:

If future work exists, how can this be complete?

Response:

The goal was no-code research design, not empirical completion. Science is not complete, but the requested markdown research-design corpus is now complete for its current phase.

## Pass 5 Verdict

No major no-code design gaps remain. The next action requires a user-authorized switch from doctrine/design into implementation or experiment mode.
