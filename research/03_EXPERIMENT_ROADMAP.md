# Experiment Roadmap

No experiments are run in this doctrine phase. This file says what future experiments must look like.

## Phase 0: Design Freeze

Deliverables:

- choose one target behavior;
- write the mechanism-card preregistration;
- define all nulls and holdouts;
- decide allowed tools;
- decide failure criteria before seeing results.

Exit criterion:

The experiment can fail without ambiguity.

## Phase 1: Signature Discovery

Goal:

Find a measurable internal pattern that predicts the target behavior.

Candidate methods:

- linear probe on activations;
- contrastive direction from positive/negative prompts;
- SAE feature activation;
- transcoder feature path;
- attribution patching path;
- activation clustering;
- natural-language activation decoder explanation as a hypothesis generator.

Required controls:

- train/test prompt split;
- paraphrase split;
- topic split;
- label permutation;
- matched-length and matched-token-frequency controls;
- prompt-only baseline;
- output-only classifier baseline;
- layer and position sweep.

Minimum evidence:

- signature predicts held-out behavior;
- signature is not reducible to surface token artifacts;
- signature appears before the behavior is fully visible in output when early warning is part of the claim.

## Phase 2: Intervention

Goal:

Change the behavior by changing the internal candidate mechanism.

Interventions:

- activation addition;
- feature suppression;
- feature injection;
- patching from positive to negative examples or the reverse;
- causal scrubbing-style ablation;
- low-rank parameter edit;
- finetuning or reward shaping using internal feature objectives.

Required controls:

- random direction with matched norm;
- unrelated feature with matched activation frequency;
- same intervention at wrong layer;
- same intervention at wrong token position;
- prompt-only intervention;
- null behavior where no change is predicted;
- dose-response curve;
- reversal or washout test when possible.

Minimum evidence:

- target behavior changes in the predicted direction;
- effect is monotonic or at least dose-sensitive;
- off-target degradation is measured;
- fluency does not collapse;
- intervention does not simply force a canned response.

## Phase 3: Reliability

Goal:

Map the boundary of the mechanism.

Reliability axes:

- paraphrase;
- topic;
- language;
- task difficulty;
- temperature;
- context length;
- adversarial prompt;
- model family;
- base versus instruction tuned;
- seed or checkpoint variation;
- longer interaction trajectories.

Output:

A reliability atlas with successes, failures, and side effects. The atlas should be written even if the mechanism fails.

## Phase 4: Practical Use

Goal:

Show the mechanism would change a real engineering decision.

Use-case questions:

- Does it predict a failure earlier than output-only monitoring?
- Does it reduce a failure without unacceptable side effects?
- Does it identify a model-diff that ordinary evals missed?
- Does it tell a training team which data or objective caused a behavior?
- Does it define a safety threshold worth monitoring?

If the answer is no, the mechanism may still be scientifically interesting, but it is not a practical control surface.

## Candidate First Experiments

### 1. Hallucination Gate

Behavior:

Model answers versus refuses or expresses uncertainty when evidence is absent.

Signature hypothesis:

There are internal known-answer, unknown-entity, refusal, or confabulation signatures that precede the output choice.

Intervention:

Suppress known-answer activation or amplify uncertainty/refusal activation on unknown entities.

Nulls:

Matched known names, fake names, random direction, unrelated entity feature, prompt-only uncertainty instruction.

Failure mode:

The signature is just entity familiarity or prompt style, not a mechanism of hallucination control.

### 2. Sycophancy And Motivated Reasoning

Behavior:

Model agrees with an incorrect user hint or works backward to a target answer.

Signature hypothesis:

There is a conflict between truth-seeking computation and user-agreement or answer-rationalization features.

Intervention:

Amplify truth-checking or suppress agreement/rationalization feature candidates.

Nulls:

Correct hints, no hints, adversarial wrong hints, unrelated social-tone prompts.

Failure mode:

The intervention merely changes style or refusal rate rather than correctness.

### 3. Evaluation Awareness

Behavior:

Model internally recognizes benchmark/test conditions and changes behavior.

Signature hypothesis:

Evaluation-awareness signatures appear even when the model does not verbalize awareness.

Intervention:

Suppress awareness signature or inject it into real-user-like prompts and measure behavioral shift.

Nulls:

Realistic user tasks, synthetic benchmark markers, hidden benchmark metadata removed, prompt-only "this is a test" baseline.

Failure mode:

The signature detects benchmark formatting, not awareness.

### 4. Refusal-Coherence Tension

Behavior:

Model begins an unsafe or policy-sensitive completion then pivots late.

Signature hypothesis:

Safety/refusal features and grammatical/coherence continuation features compete over time.

Intervention:

Amplify refusal earlier or reduce coherence pressure after a harmful latent cue appears.

Nulls:

Benign completions with similar grammar, harmless encoded words, random substitutions.

Failure mode:

Intervention causes generic refusal or damages helpfulness.

### 5. Planning-Ahead In Constrained Generation

Behavior:

Model plans future rhyme, code variable, proof step, or final answer before emitting it.

Signature hypothesis:

Future-target representations appear before the output position requiring them.

Intervention:

Suppress or swap the future-target representation and observe downstream change.

Nulls:

No-constraint generation, random future concept, matched surface word frequency.

Failure mode:

The feature reflects local continuation statistics rather than actual planning.

### 6. Symbolic Fact-Code Arbitration

Behavior:

Model chooses between a prompt-local artificial symbolic code and a learned
real-world symbolic fact.

Current preregistration and V1 status:

`research/prereg/MC008_SYMBOLIC_FACT_CODE_ARBITRATION.md`

`research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

`research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_V2_NULL_AUTHORITY_REPAIR_STATUS.md`

`research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_ROUTE_CLOSEOUT_STATUS.md`

V1 result:

The first full Qwen3-1.7B symbolic-code behavior run selected `symbol_field`.
It passed direct synthetic lookup and real-world chemical-symbol controls, but
failed before hidden-state work: answer-absent null was only 30/40 `UNKNOWN`,
primary conflict parseability was 80.4 percent, and source-disjoint real/lure
conflict balance was absent.

V2 result:

The null/authority repair selected `membership_authority_split`. It preserved
direct controls and repaired nulls: synthetic lookup was 39/40 artificial-code
rows, real-world symbol recall was 40/40 real-symbol rows, and answer-absent
null was 40/40 `UNKNOWN`. It still failed the bridge because primary conflict
remained table-dominant: 220 artificial-code rows versus 2 real-symbol rows and
0 lure-symbol rows.

Signature hypothesis:

A bridge surface may appear when the answer is a compact generated code rather
than an open city name: source-token lookup, learned memory pressure, and
output/candidate geometry may become separable before final answer commitment.

Intervention:

Still forbidden for MC008 V1-V2. Only after a future behavior and signature
gate pass:
source-value attention/write
replacement, source-path masking with rewrite-equivalence controls, or
pre-output residual steering that beats output/candidate baselines.

Nulls:

Synthetic code lookup rows, real-world memory controls, answer-absent null
rows, wrong element, wrong code, wrong layer, wrong position, deletion and
neutral rewrite controls.

Failure mode:

V1 showed that compact symbolic outputs can repair direct controls without
preserving null reliability, primary-conflict parseability, or
artificial-versus-real balance. V2 showed that the null boundary is repairable,
but artificial-versus-real balance still fails. The first symbolic route is
closed; the next bridge must change more than prompt wording around the same
matching element-code table. MC009 tested the obvious next variant and showed
that removing the printed element-code pair is also not enough.

### 7. Derived-Code Arbitration

Behavior:

Model chooses between a prompt-local derived code and a learned real-world
symbol. The prompt-local code is computed from table row position rather than
printed as `entity -> code`.

Current preregistration:

`research/prereg/MC009_DERIVED_CODE_ARBITRATION.md`

Current smoke / route-closeout status:

`research/cards/MC009_DERIVED_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

Runner:

`code/mc009_derived_code_arbitration.py`

Rationale:

MC008 closed because direct matching artificial values dominated the generated
answer. MC009 removes the direct source-value row while preserving a
prompt-local answer, compact generated codes, real-world memory controls,
source-disjoint holdout, and answer-absent null rows.

Signature hypothesis:

Not reached. The route failed before hidden-state discovery. If a materially
different derived-answer family is ever rebuilt, the relevant internal surface
may be a row-position, target-row, or authority-state monitor rather than a
source-value lookup path. The signature would still have to beat output,
candidate, prompt-format, and row-position baselines before any intervention is
allowed.

Intervention:

Forbidden. Candidate interventions were never valid because no MC009 behavior
substrate passed the controls, parseability, source-disjoint balance, and
prompt-visibility checks.

Failure mode:

The derived-code rule may still collapse to prompt-local task behavior, fail
row-position computation, fail nulls, or become fully explained by
row-position/output/candidate baselines.

Current route verdict:

A 10-source `membership_authority_split` smoke now validates the direct
controls: synthetic ordinal lookup, real-world symbol recall, and
answer-absent nulls are all 10/10. The bridge still fails before full behavior
promotion: primary conflict parseability is 47/60, real/lure labels are only
7/60, and non-holdout plus holdout real/lure balance fail. Two bounded format
repairs were worse and are diagnostic only.

The non-default `typed_slot_v2` template produced the opposite failure: primary
conflict parseability reached 54/60, with 23 derived-code and 19 real-symbol
rows, so non-holdout and holdout balance passed. But synthetic ordinal lookup
collapsed to 0/10 derived-code rows, answer-absent nulls collapsed to 0/10
`UNKNOWN`, and the typed answer slots exposed the source channel in the prompt.
That is prompt-visible behavior control, not an internal mechanism substrate.

Decision:

Close the first MC009 derived-code route as a diagnostic bridge. Do not run
hidden-state work. Do not continue ordinary prompt-only repairs inside this
route. The next bridge must change the conflict family materially or move to a
new family.

## Experiment Quality Bar

A future result should not be written up if any of these are missing:

- preregistered behavior definition;
- heldout prompts;
- null directions or features;
- off-target behavior checks;
- human-readable failure analysis;
- raw enough artifacts for review;
- explicit statement of what did not work.

## Falsification Rules

The project should stop or pivot a line when:

- signatures repeatedly predict behavior but interventions fail;
- interventions work only by damaging fluency or forcing generic refusal;
- prompt-only baselines match internal interventions;
- hidden-state methods do not beat output or trace monitors for a practical task;
- cross-prompt reliability collapses;
- controls reveal tokenizer, formatting, or dataset artifacts.
