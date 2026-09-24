# Advanced Math Program

This extends the math agenda into a formal research program. It should remain subordinate to experiments.

## Formal Setup

Let a model define a stochastic computation:

`Y ~ M_theta(X; r)`

where:

- `X` is context;
- `Y` is output;
- `r` is sampling randomness;
- `theta` are model parameters;
- hidden trajectory is `H = {h_t^l}`.

A behavior is a functional:

`B = b(X, Y)`

A signature is a map:

`S: H -> R^k`

An intervention family is:

`I_alpha: H -> H`

or, for durable edits:

`J_alpha: theta -> theta'`

The mechanism-card question is:

> Does `S` predict `B`, and does applying `I_alpha` change `B` in the predicted direction under bounded side effects?

## Causal Fidelity

Define causal fidelity as the degree to which a signature's predictive information is preserved under intervention.

Informally:

- high predictive power;
- high intervention effect;
- low null effect;
- low side-effect cost;
- stable across contexts.

A rough score:

`F = Predict(S, B) * Effect(I_alpha, B) * Specificity / Cost`

This is not yet a final metric. It is a prompt for math: predictive information should not be enough. Intervention effect and specificity must enter.

## Intervention Capacity

A behavior has high intervention capacity if small, local changes to hidden state can reliably move the behavior.

Candidate quantity:

`C_B = max_alpha Delta_B(I_alpha) / (||I_alpha|| * SideEffect(I_alpha))`

Questions:

- Which behaviors have high capacity?
- Does capacity correlate with probe accuracy?
- Do sparse features have higher capacity than dense directions?
- Does capacity change by layer?

## Locality

A control surface is local only relative to a side-effect set.

Let `B_target` be the target behavior and `B_off` be off-target behaviors.

Locality requires:

`Delta B_target` large and `Delta B_off` small.

The project should not talk about locality without defining the off-target set.

## Observability Rank

Borrowing from control theory:

- a behavior is observable if hidden statistics reveal its future outcome;
- a behavior is controllable if interventions can move it;
- a behavior is mechanistically accessible if both are true.

Potential empirical object:

For each layer and token position, estimate:

- predictive rank: number of dimensions needed to predict behavior;
- control rank: number of intervention directions with target effect;
- overlap rank: intersection between predictive and controllable subspaces.

The overlap rank is more important than either alone.

## Invariant Mechanisms

For contexts `c` in base space `C`, a mechanism has local charts:

`S_c(H_c)`

A mechanism generalizes if transition maps preserve intervention effect:

`T_{c->d}(S_c)` predicts or induces the analogous behavior in context `d`.

This avoids vague universality language. It asks whether intervention effects commute with context changes.

## Counterfactual Equivalence Classes

Two hidden states are behavior-equivalent if they produce the same target behavior under ordinary decoding.

They are mechanism-equivalent only if they respond similarly to interventions.

This distinction matters:

- behavior-equivalent states may have different causal internals;
- mechanism-equivalent states should share intervention response.

Future experiments should sample both.

## False Discovery Control

Interpretability scans many layers, features, directions, prompts, and doses. Without correction, false discovery is guaranteed.

Required math:

- split discovery, calibration, and holdout;
- correct for feature search when reporting discovered features;
- report the number of scanned candidates;
- treat hand-picked examples as illustrations, not evidence.

Potential tools:

- Benjamini-Hochberg for feature discovery;
- nested holdouts for dose selection;
- permutation baselines for probes;
- bootstrap confidence intervals for behavior effect.

## Minimal Theorem Targets

### Theorem Target 1: Prediction Does Not Imply Control

Construct or characterize cases where `S` predicts `B` but no local `I_alpha` on `S` changes `B` without high side effects.

Value:

Explains diagnostic-only results.

### Theorem Target 2: Sparse Feature Identifiability Is Not Enough

Even if a sparse feature is stable, it may be downstream of the true causal mediator.

Value:

Prevents overclaiming SAE features.

### Theorem Target 3: Local Control Requires Overlap

If predictive subspace and controllable subspace have low overlap, interventions based on predictors fail.

Value:

Defines why some signatures steer and others do not.

### Theorem Target 4: Reliability Is A Transition-Map Problem

Generalization across prompt families requires preservation of intervention effect under context transition maps, not just preservation of feature activation.

Value:

Turns reliability atlas into math.

## Mathematical Red Lines

Do not introduce:

- category theory without an experiment it clarifies;
- topology without a measurable invariant;
- information theory without an estimator;
- control theory without an intervention;
- universality language without transition tests.

The math should make claims harder to pass, not easier to narrate.
