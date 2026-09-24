# MC007 Semi-Synthetic Familiar-Entity Lookup Preregistration

Date: 2026-07-01

## Purpose

MC005 and MC006 define the current gap in the atlas.

MC005 shows that a prompt-visible synthetic lookup table can expose a bounded
source-value mediation surface: final-query attention writes in layers 24-26
produce a large lookup effect, with rare answer-absent null locality failures.

MC006 shows that a knowledge-like capital-fact override table can support a
matched generated behavior substrate and a pre-output monitoring signal, but
the labels remain globally visible to candidate-score and final-output margins,
and additive steering fails.

MC007 tests the bridge:

> If familiar real-world entity names are paired with artificial prompt-local
> values, does the model behave more like synthetic source-value lookup, more
> like output-visible factual override, or as a measurable mixture of both?

This is the first explicit transition experiment in the control-surface atlas.

## Law Hypotheses Under Test

MC007 primarily tests:

- `source_visible_lookup_localizes_more_than_parametric_override`;
- `final_state_output_geometry_dominance`;
- `lead_time_monitor_before_lever`;
- `coarse_source_ablation_overstates_circuit_locality`;
- `null_reliability_bottleneck`.

Machine-readable hypotheses:

`data/control_surface_law_hypotheses.json`

## Model

Initial target:

- `Qwen/Qwen3-1.7B`

Reason:

- MC005's strongest bounded lookup surface is on Qwen3-1.7B;
- MC006's current knowledge-like behavior and lead-time results are on
  Qwen3-1.7B;
- the bridge should first hold model family constant.

No model-transfer claim is allowed from the initial run.

## Task Family

Use familiar entity names with artificial values in the prompt.

Candidate entity families:

- countries with artificial city-like values;
- companies with artificial product-code values;
- famous people with artificial badge values;
- landmarks with artificial region-code values.

The first implementation should use countries because MC006 already provides
capital-fact pressure and candidate-scoring tooling. The artificial values must
not be the real capitals of the countries in the row, and prompt text must make
the artificial mapping the requested source of truth.

Example contract:

```text
Use the mapping table below for this task.

Country -> task city
France -> Doveton
Japan -> Larkspur
Brazil -> Norhaven

Question: According to the mapping table, what is the task city for Japan?
Answer with only the city.
```

The target label is the artificial prompt-local value, not the real-world
capital.

## Panels

MC007 should build at least four panels.

### Panel A: Synthetic Control Lookup

Arbitrary non-semantic keys mapped to artificial values.

Purpose:

- reproduce an MC005-like source-value lookup baseline under the new runner;
- estimate how much locality is available when semantic priors are absent.

### Panel B: Familiar Entity Artificial Lookup

Familiar real-world entity names mapped to artificial values.

Purpose:

- test whether familiar entity priors weaken source-value lookup;
- measure artificial-value adherence under generated answers.

### Panel C: Familiar Entity Conflict

Familiar entity names mapped to artificial values, with a real-world lure
available for scoring and parsing.

Purpose:

- separate artificial prompt-local value from real-world prior;
- measure whether final output/candidate margins expose the artificial-vs-real
  decision.

### Panel D: Answer-Absent Null

Prompts with the same entity/value style but a query or target candidate not
supported by the table.

Purpose:

- test whether any intervention that works on lookup rows stays local on
  answer-absent rows;
- preserve MC005's null-locality discipline.

## Behavior Gate

Hidden-state probing is forbidden unless the behavior table passes.

Minimum behavior criteria for the selected generated-answer template:

1. At least 40 primary binary rows across Panels B and C.
2. At least 10 artificial-value rows and 10 real-prior/lure rows in the
   non-holdout primary split if Panel C produces both outcomes.
3. Holdout must contain at least 4 artificial-value rows and at least 4
   real-prior/lure rows if both outcomes exist.
4. Strict first-line parsing must parse at least 90 percent of primary rows.
5. No prompt may leak the artificial answer outside the mapping row for that
   entity.
6. No artificial value may equal any real-world answer candidate in the row.
7. Source-disjoint split: holdout entities must not appear in discovery or
   calibration.
8. Panel A synthetic lookup must pass at least 90 percent artificial-value
   adherence, otherwise the runner or generation contract is not lookup-clean.
9. Panel D null rows must remain parseable enough to score target/distractor
   margins, but no null behavior success is required before intervention.

If these fail, the output is a behavior diagnostic only.

## Signature Gate

Only after the behavior gate passes, test hidden signatures at:

- source key token positions;
- source artificial-value token positions;
- query entity token positions;
- post-table pre-question positions;
- final prompt token;
- intermediate response-marker or answer-prefix positions if present.

For each candidate signature, report:

- discovery AUC;
- holdout AUC;
- source-disjoint holdout AUC;
- same-position next-token output-margin AUC;
- final-prompt next-token output-margin AUC;
- candidate-score margin AUC when candidates are available;
- prompt length and token-count baselines;
- entity identity and artificial-value token-count baselines;
- shuffled-label selection p95;
- Panel A/B/C subgroup AUCs.

Signature promotion requires:

1. Holdout AUC at least 0.85.
2. Source-disjoint holdout AUC at least 0.80.
3. Hidden score beats same-position output margin by at least 0.05.
4. Hidden score beats final-output and candidate-score margins by at least
   0.02, unless the result is explicitly classified as lead-time monitoring
   only.
5. Hidden score beats prompt/entity/value token baselines by at least 0.05.
6. Hidden score beats shuffled-label p95 by at least 0.05.
7. Subgroup performance does not collapse on either familiar-entity or
   conflict rows.

If the signature only beats same-position output controls but not final-output
or candidate-score controls, classify as:

```text
lead_time_monitor_only
```

No intervention is allowed from a final-token signature that is matched by
final-output or candidate-score controls.

## Intervention Gate

Intervention is allowed only if the signature gate identifies a candidate that
is not explained by prompt/entity/value baselines and has a reason to be
causally upstream.

Allowed first intervention families:

1. Source-value attention/write replacement, modeled after MC005, if the
   selected signature or path is source-value localized.
2. Source-token path masking, only with rewrite-equivalence controls.
3. Residual steering, only if the selected coordinate beats output/candidate
   controls or is preregistered as a known-confounded causal stress test.

Required controls:

- wrong source token;
- wrong entity;
- wrong value;
- wrong layer;
- wrong position;
- matched random vector or path;
- prompt deletion and neutral rewrite equivalence;
- Panel D answer-absent nulls;
- side rows where the real prior should still win if no mapping is given.

## Mechanism Promotion Criteria

MC007 supports a mechanism-card candidate only if all gates pass:

1. Behavior gate passes.
2. Signature gate passes on source-disjoint holdout.
3. Intervention moves generated behavior or target-vs-lure margin in the
   predicted direction on holdout.
4. Selected intervention beats all wrong-source, wrong-entity, wrong-value,
   wrong-layer, wrong-position, and random controls.
5. Prompt deletion and neutral rewrite do not explain the same effect.
6. Panel D answer-absent nulls remain local under preregistered margin strata.
7. Side rows do not show broad parse or label corruption.
8. The allowed claim is specific about whether the surface is synthetic-like,
   familiar-entity-specific, or conflict-specific.

## Bounded Success Criteria

MC007 should be promoted as a bounded atlas result, not a full mechanism card,
if:

- behavior and signature gates pass, but intervention is matched by a control;
- source-value mediation appears on Panel A or B but fails on conflict rows;
- intervention works on lookup rows but answer-absent nulls fail;
- the hidden signal beats same-position output controls but remains
  final-output/candidate visible.

Bounded results should update the atlas mixture profile rather than be treated
as failed noise.

## Death Criteria

Kill the initial MC007 route if:

1. Panel A synthetic lookup fails, indicating the runner or template is broken.
2. Panels B/C do not produce enough parseable binary rows after one template
   repair.
3. All hidden signatures are matched by prompt/entity/value baselines,
   final-output margin, candidate-score margin, or shuffled-label selection.
4. Interventions reproduce only prompt deletion/rewrite effects.
5. Null or side rows show broad corruption under the only active intervention.

## Diagnostic Labels

Expected labels:

- `behavior_substrate_failed`;
- `synthetic_control_failed`;
- `semantic_prior_dominant`;
- `output_margin_confounded`;
- `candidate_score_confounded`;
- `prompt_entity_confounded`;
- `lead_time_monitor_only`;
- `source_value_surface_supported`;
- `source_deletion_not_circuit`;
- `null_row_low_margin_flip`;
- `intervention_failed`;
- `bounded_source_value_mechanism`;
- `mechanism_candidate_supported`.

If new labels are used in result JSON, update
`data/control_surface_atlas.json` controlled vocab before adding an atlas row.

## Allowed Interpretations

If MC007 behaves like MC005:

> Familiar entity names do not erase source-value mediation under this prompt
> contract. The model can treat real entity tokens as lookup keys for
> artificial values, and the source-value path may generalize beyond arbitrary
> synthetic keys.

If MC007 behaves like MC006:

> Familiar entity priors and final answer geometry dominate even when prompt
> values are artificial. The transition from lookup to knowledge-like behavior
> happens as soon as semantically loaded entity names enter the contract.

If MC007 is mixed:

> The bridge task has located a transition surface: synthetic and familiar
> source lookup share some machinery, but semantic priors, output geometry,
> null locality, or intervention dirtiness bound the claim.

## Forbidden Interpretations

MC007 may not claim:

- a general knowledge mechanism from artificial values;
- a truth vector;
- factual editing;
- model-wide recall control;
- transfer across models;
- mechanism promotion from behavior control alone;
- mechanism promotion from a final-token hidden classifier matched by
  candidate-score or output-margin baselines.

## Export To Atlas

Regardless of outcome, MC007 must update:

- `data/control_surface_atlas.json`;
- `data/control_surface_law_hypotheses.json` if a law is supported, bounded, or
  falsified;
- `research/20_CONTROL_SURFACE_ATLAS.md`;
- `research/21_CONTROL_SURFACE_LAWS_AND_GAPS.md`.

The required atlas fields are:

- behavior gate state;
- lead-time state;
- intervention state;
- mixture profile;
- diagnostic classes;
- allowed and forbidden claims;
- source-disjoint evidence;
- null locality;
- transfer state.
