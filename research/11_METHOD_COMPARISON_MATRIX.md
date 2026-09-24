# Method Comparison Matrix

This file prevents method loyalty. The project should use the method that best supports a mechanism card, not the method that is fashionable.

## Selection Rule

A method is useful only if it helps answer at least one of these:

1. Can we observe the behavior before output?
2. Can we intervene on the candidate mechanism?
3. Can we estimate side effects?
4. Can we distinguish mechanism from artifact?
5. Can we make a practical engineering decision?

## Method Matrix

| Method | Best Use | Main Failure | Required Control | Role In MC-001 |
| --- | --- | --- | --- | --- |
| Linear probe | Fast behavior prediction | Learns surface artifacts | label permutation, prompt-format baseline | baseline internal signature |
| Contrastive direction | Cheap steering/control candidate | Encodes style, confidence, or prompt form | random matched norm, sign flip, wrong layer | primary intervention candidate |
| SAE feature | Human-readable sparse coordinate | feature splitting, dictionary dependence | matched-frequency feature, SAE seed check | candidate conflict/agreement features |
| Transcoder feature | More mechanistic input-output path | hard to interpret, artifact dependence | path ablation and unrelated path | optional path-level mechanism |
| Activation patching | Causal localization | patch transfers many variables at once | clean/corrupt swaps, token/layer sweeps | locate hint-to-answer mediation |
| Attribution graph | Structured causal story | graph can be incomplete or overfit | ablate predicted nodes and null nodes | later evidence if simple methods work |
| Natural-language decoder | Hypothesis generation | explanation hallucination | independent causal test | no proof value by itself |
| Model editing | Durable intervention | off-target damage, brittle locality | specificity/generalization/fluency | later phase, not first lever |
| Output/logit monitor | Cheap deployed baseline | not mechanistic | compare hidden lead-time and specificity | required comparator |
| Trace/eval monitor | production relevance | sees visible behavior too late | hidden-state lead-time test | product comparator |

## Interpretation Discipline

### Probes

Probes answer:

> Is the information present?

They do not answer:

> Is the information used?

For MC-001, a probe that predicts wrong-hint compliance is useful only if it beats output-only baselines and leads to a successful intervention or early-warning claim.

### Activation Directions

Directions are the fastest path to a first intervention. They are also the easiest to fool.

Minimum requirements:

- dose curve;
- sign-flipped control;
- wrong-layer control;
- random matched-norm control;
- off-target tasks;
- prompt-only baseline.

Do not claim a direction is an "honesty vector." The maximum allowed claim is that a direction controls a scoped behavior under scoped controls.

### Sparse Features

Sparse features are strong when they provide:

- readable hypotheses;
- feature-level interventions;
- cross-prompt reuse;
- cleaner side-effect accounting.

They are weak when:

- multiple features split the same concept;
- dictionary size changes the story;
- feature labels come from human glosses without causal support;
- activation frequency explains the effect.

### Transcoders

Transcoders can expose transformations rather than static features. They are valuable if the first card needs to say not just "this hidden state contains conflict" but "this path carries hint pressure into the answer."

Do not start with transcoders if a simple direction or probe can fail the target faster.

### Attribution And Circuit Tracing

Circuit tracing is powerful because it gives a graph of causal candidates. It is expensive in analysis time. For the first card, use it only if simpler methods pass enough gates to justify deeper mechanistic resolution.

### Natural-Language Decoders

Natural-language decoders are hypothesis engines. They should be treated like a researcher writing a guess. They may be correct, incomplete, or hallucinated.

Allowed role:

- propose candidate features;
- summarize activation differences;
- help generate alternate hypotheses.

Disallowed role:

- prove that the model "knows";
- replace intervention;
- replace null controls.

### Model Editing

Editing is not the first lever because it is harder to localize and easier to damage the model. It becomes relevant only after an activation-level mechanism has survived.

## MC-001 Method Order

Run future implementation in this order:

1. Behavior prevalence and output baselines.
2. Linear probes and activation directions.
3. SAE/transcoder feature search.
4. Activation direction interventions.
5. Feature suppression or injection.
6. Patching/path tests if earlier signals survive.
7. Final mechanism card.

The order matters. The project should try to kill the hypothesis cheaply before doing expensive interpretability analysis.

## Decision Rule

If a simpler method explains the behavior as well as a complex one, the complex one does not get promoted.

If a complex method uniquely predicts a reliable intervention, it becomes the mechanism candidate.

If no method beats prompt/output baselines, the card fails.
