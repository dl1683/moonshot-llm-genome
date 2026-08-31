# Math Agenda

The math should serve the mechanism-card contract. Do not build ornamental theory.

## 1. Control Surfaces As Causal Objects

Let a model run define hidden states `h_t^l` across token position `t` and layer `l`. A candidate control surface is not merely a statistic `s(h)`. It is a statistic plus an intervention operator `I_alpha` such that:

- `s(h)` predicts behavior `B`;
- applying `I_alpha` changes `B` in a direction predicted by `s`;
- the effect is local relative to a specified off-target set;
- null operators with matched norm or frequency do not reproduce the effect.

Mathematical question:

> What conditions make a representation statistic intervention-valid rather than merely predictive?

Useful concepts:

- causal mediation;
- do-calculus style intervention semantics;
- local average treatment effects in activation space;
- counterfactual activation patching;
- operator norm and constrained side-effect budgets.

## 2. Intervention Algebra

Model interventions can be treated as operators on hidden-state trajectories:

- addition: `h -> h + alpha v`;
- projection: `h -> h - P_S h`;
- replacement: `h_i -> h_j`;
- edit: `W -> W + Delta`;
- feature clamp: `z_k -> c` in a sparse basis.

Questions:

- When do two interventions commute?
- When does one intervention erase the evidence needed by another?
- Which interventions are reversible?
- What is the minimum-norm intervention that changes behavior?
- What is the side-effect frontier for a target behavior?

This could become a small control theory of transformer internals.

## 3. Observability And Controllability

In control theory, a system is useful when relevant state variables can be observed and controlled. For LLMs:

- observability asks whether hidden signatures reveal future behavior;
- controllability asks whether accessible interventions can move the behavior;
- locality asks whether control can be applied without broad damage.

Potential criterion:

> A behavior is mechanistically accessible when there exists an observation map `O(h)` and intervention family `I_alpha` such that `O` predicts behavior and `I_alpha` changes behavior under bounded off-target cost.

This is the cleanest mathematical framing for the repo.

## 4. Sparse Coding And Identifiability

Sparse autoencoders and transcoders assume dense activations can be represented in a more interpretable overcomplete basis. The dangerous question is identifiability:

> When is a learned feature a stable object rather than one valid coordinate system among many?

Needed tests:

- feature stability across SAE seeds;
- stability across dictionary size;
- stability across training data;
- relationship between reconstruction error and causal usefulness;
- feature splitting and merging under scale;
- comparison with dense probes and contrastive vectors.

Mathematical tools:

- dictionary learning identifiability;
- compressed sensing;
- sparse recovery;
- mutual coherence;
- overcomplete basis degeneracy;
- stability under perturbation.

## 5. Geometry Without Mysticism

Representation geometry is useful if it predicts something. Candidate geometry:

- local linear subspaces;
- curved manifolds of behavior;
- covariance spectra;
- Fisher information geometry;
- geodesic control cost;
- curvature around decision boundaries;
- anisotropy and collapse metrics.

Bad use:

- "The geometry looks universal, therefore intelligence is universal."

Good use:

- "This geometric statistic predicts which layer contains a controllable behavior signature."

## 6. Fiber-Bundle View Of Context

A useful formal metaphor:

- base space: contexts, prompts, tasks, languages, domains;
- fiber: hidden representation coordinates available in each context;
- local chart: feature basis or probe basis;
- transition map: how a mechanism appears across contexts.

Universality claim:

> A mechanism generalizes across contexts only if there are transition maps that preserve intervention effects, not just representation similarity.

Failure modes:

- chart artifact: a feature appears stable but intervention does not transfer;
- transition failure: signature transfers but control direction changes;
- base-space hole: mechanism fails in a hidden region of contexts.

This gives a rigorous language for "works here, breaks there."

## 7. Rate-Distortion And Mechanism Compression

A mechanism explanation is a compressed description of computation. Natural-language activation decoders make this explicit: an activation is compressed into text, then reconstructed.

Questions:

- What is the minimum description length of a mechanism that preserves intervention predictions?
- How much behavior-relevant information is lost by a natural-language explanation?
- Can a mechanism card be evaluated as a rate-distortion object: short enough for humans, faithful enough for control?

This can discipline explanation quality.

## 8. Singular Learning And Phase Transitions

Training dynamics may contain phase transitions where a behavior or internal signature suddenly becomes usable. Singular learning theory and related tools may help explain why feature emergence is not smooth.

Questions:

- Do control surfaces emerge abruptly during training?
- Does a signature appear before behavior, after behavior, or together with behavior?
- Are there detectable precursors to dangerous or useful capabilities?
- Can training-health diagnostics catch bad transitions before loss does?

This connects interpretability to training-time monitoring.

## 9. Mechanism Portability

Portability should be formalized as a simulation relation, not a vague similarity score.

Model A mechanism `M_A` ports to model B only if:

- a signature map exists from A to B;
- the behavior correspondence is defined;
- an intervention in B produces the analogous effect;
- nulls in B fail;
- the mapping survives heldout contexts.

Representation similarity alone is not enough. Portability is intervention preservation.

## 10. Statistical Evidence

Mechanism claims should use a sequential evidence ledger:

- preregistered hypotheses;
- heldout prompts;
- effect sizes with uncertainty;
- multiple-comparison correction when scanning many features;
- negative controls;
- adversarial splits;
- replication across seeds or models when claimed.

The math of the project must include decision rules:

- when to promote a signature;
- when to kill it;
- when to spend more compute;
- when to call a result an artifact.

## 11. State Gates And Cheapest-Mechanism Nulls (deposited from Latent-Space-Reasoning)

Five constraints from a closed native-latent-mathematics program (50 audited
experiments, 2026-08-27 → 2026-08-31) that strengthen this project's
mechanism-card contract:

**Three state gates.** Information ≠ state. A candidate internal state must
pass three gates before it counts: (1) **present** — a signature is
measurable; (2) **addressable** — an intervention can read/write it;
(3) **composable** — two states can be combined and the result predicts
behavior. Most "state" claims in interpretability stop at gate 1. This
project's mechanism-card contract (signature → intervention → reliability)
already encodes gates 1–2; gate 3 (composability) is an explicit addition.

**Transport-earned quotients.** A claim that two representations are
"equivalent" (same mechanism, different contexts) must be earned by a
transport — a held-out intervention that preserves behavior across the
claimed equivalence class. Held-out presentation similarity is not
quotient-level generalization.

**Collision witnesses for absence.** Claiming a mechanism is absent requires a
collision witness: two inputs with the same internal signature but different
downstream behavior. A failed probe is not evidence of absence — the probe
may be the wrong instrument.

**Cheapest-mechanism nulls.** Before attributing an effect to a discovered
mechanism, run the cheapest direct alternative as a control (e.g.,
identity-plus-shared-displacement before a learned transport; output-margin
before a hidden signature). This project already does matched-norm nulls;
the cheapest-mechanism null is a stronger requirement.

**Many operational latent spaces.** A model has many operational latent spaces,
indexed by (actions, observations, horizon). A mechanism card should specify
which operational space it lives in — what actions are available, what is
observable, and over what horizon the behavior is defined.

Source: `Latent-Space-Reasoning/theory/AXIOMS.md`, `STATE.md`, `NOTEBOOK.md`.
These are negative-result constraints, not positive mechanisms.

## The Central Theorem-Shaped Ambition

The deepest possible version of the project would prove something like:

> Under specified assumptions about representation sparsity, causal mediation, and intervention locality, a class of learned behaviors admits discoverable control surfaces whose intervention validity can be certified from finite experiments with bounded false-discovery risk.

That is too ambitious for the first milestone. But it is the north star for the math.
