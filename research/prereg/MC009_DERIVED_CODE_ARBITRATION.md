# MC009 Derived-Code Arbitration Preregistration

Date: 2026-07-01

## Purpose

MC009 is the next bridge family after the MC007 and MC008 closeouts.

MC007 showed that familiar country names plus artificial city values either
collapse to prompt-local lookup or fail parseability/authority controls. MC008
changed the answer family to compact symbolic codes. That repaired direct
controls and, in V2, repaired answer-absent nulls. But the route still closed:
as soon as a matching artificial code was attached to a real entity in the
prompt table, Qwen3-1.7B overwhelmingly emitted that prompt-local code even
when the prompt said ordinary chemistry should control.

MC009 changes the conflict construction.

The task-local answer is no longer printed as a value beside the entity. The
prompt gives a table of entities and a rule that derives a short code from the
entity's row position. The model must compute the task code if the task table
controls, or use learned real-world memory if ordinary factual knowledge
controls.

Example:

```text
Row-code rule:
row 1 -> Qa
row 2 -> Qb
row 3 -> Qc
row 4 -> Qd

Element table:
1. Sodium
2. Carbon
3. Oxygen
4. Iron

Query: Sodium
```

For task-table authority, the answer is `Qa`. For ordinary chemistry, the
answer is `Na`. The prompt-local task answer is available through source
structure and computation, not through a direct `Sodium -> Qa` mapping.

## Relationship To Closed Routes

MC009 is not another MC008 prompt repair.

It changes:

- source visibility: direct entity->value row removed;
- computation: task answer derived from row position;
- conflict construction: prompt-local value is not bound to entity text as a
  printed pair;
- diagnostic target: whether matching prompt values caused the MC008
  table-dominance collapse;
- artifact role: next bridge family after
  `MC008_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE`.

It keeps:

- compact generated answers;
- real-world factual memory controls;
- source-disjoint holdout;
- answer-absent null rows;
- strict parsing;
- output/candidate baseline reporting before hidden-state work;
- no intervention until behavior and signature gates pass.

## Law Hypotheses Under Test

Primary hypotheses:

- `source_visible_lookup_localizes_more_than_parametric_override`;
- `authority_pressure_creates_contrast_before_clean_substrate`;
- `behavior_substrate_first_or_everything_lies`;
- `null_reliability_bottleneck`;
- `final_state_output_geometry_dominance`.

Specific MC009 question:

> If the prompt-local answer is derived from table structure rather than printed
> as a matching source value, can a small LLM produce a balanced generated
> bridge between task-local lookup computation and learned factual memory?

## Model

Initial target:

- `Qwen/Qwen3-1.7B`

Reason:

- MC005, MC006, MC007, and MC008 bridge evidence is currently on Qwen3-1.7B;
- the next bridge should change the behavior family before changing model
  family.

No transfer claim is allowed from the initial run.

## Source Set

Use the same 40 common chemical elements as MC008 where possible.

Each source row must include:

- source id;
- element name;
- true real-world symbol;
- source-disjoint split;
- synthetic nonce key for non-factual controls.

The true symbol must not be printed in any conflict prompt.

## Derived Code Rule

Primary rule:

```text
row 1 -> Qa
row 2 -> Qb
row 3 -> Qc
row 4 -> Qd
```

The code for the target depends on the target's row position inside the local
four-row table. The target must rotate across all four row positions across
sources and splits.

Required checks:

1. The target's task code is not printed as `Element -> Code`.
2. The true symbol is never printed in conflict rows.
3. Row-code legend is generic and identical across rows.
4. Target row position is balanced across discovery, calibration, and holdout.
5. No row-position code equals the target true symbol.
6. Null rows list the row-code rule but omit the queried entity from the table.

This is intentionally not a hidden task. It is a prompt-local computation task.
The point is to remove direct source-value binding while preserving a real
prompt-local answer.

## Panels

MC009 must include at least six panels.

### Panel A: Synthetic Ordinal Code Lookup

Nonce keys are listed in a four-row table. The query key appears in one row.
The task answer is the row-code for the query row.

Purpose:

- verify that the model can follow the ordinal code rule without factual
  priors.

Gate:

- at least 90 percent derived-code generated answers;
- answer-absent nulls for nonce keys must produce `UNKNOWN` at least
  95 percent.

### Panel B: Familiar Element Derived-Code Lookup

Real element names appear in a four-row table. The query element appears in one
row. The task asks for the derived task code.

Purpose:

- test whether familiar element names behave as table rows when the code is
  not directly printed next to the element.

Expected labels:

- derived task code;
- real symbol;
- lure/other symbol;
- unknown;
- unparsed.

### Panel C: Real-World Memory Control

No task table is active. The prompt asks for the standard chemical symbol.

Purpose:

- verify real-symbol generation under the same answer format.

Gate:

- at least 85 percent real-symbol or accepted real/lure answers;
- at least 95 percent parseability.

### Panel D: Authority Dial Conflict

The prompt includes the ordinal code rule and element table. It asks for an
answer under a source-authority dial.

Dial values:

- 100;
- 70;
- 50;
- 30;
- 0.

At high authority, the expected answer is the derived task code. At low
authority, the expected answer is the real chemical symbol.

Purpose:

- test whether removing direct entity->code binding makes authority pressure
  capable of producing balanced task-versus-real generated labels.

### Panel E: Conflict Without Explicit Source Labels

The prompt contains the ordinal rule and element table, says ordinary chemistry
may also be relevant, and asks for only the final code.

Purpose:

- avoid explicit source labels, since MC007 V4 showed they can decouple from
  generated answers.

### Panel F: Answer-Absent Null

The prompt includes the row-code rule and a four-row table, but the queried key
or element is absent.

Purpose:

- preserve null discipline after MC008 V2 showed symbolic nulls are repairable;
- test whether ordinal rules invite row-position inference for absent queries.

Gate:

- at least 95 percent parseability;
- at least 90 percent `UNKNOWN`.

## Output Parsing

Generated answers are primary.

Strict parser:

- read the first non-empty generated line;
- accept `Qa`, `Qb`, `Qc`, `Qd`;
- accept the target's true chemical symbol;
- accept a known lure/other element symbol as `lure_symbol` or
  `other_symbol_code`;
- accept `UNKNOWN`;
- reject explanations and source declarations as `unparsed`.

Candidate scoring is a control, not the behavior label source.

## Prompt Leak Audit

Conflict rows must not print the target element's true symbol anywhere in the
prompt.

The runner must audit:

- no target true-symbol prompt leaks;
- no direct `target element -> target task code` string;
- generic row-code legend appears exactly once;
- target appears exactly once in the local table for non-null rows;
- target is absent from null tables;
- target row position is logged;
- row-position code balance is logged by split.

If true-symbol leakage occurs, the row is invalid.

## Behavior Gate

No hidden-state work is allowed unless a selected template passes all of:

1. Structural checks pass.
2. At least 40 sources.
3. Source-disjoint holdout.
4. No true-symbol prompt leaks in conflict rows.
5. No direct target->task-code mapping in conflict rows.
6. Synthetic Panel A passes at least 90 percent derived-code adherence.
7. Real-world Panel C passes at least 85 percent real-symbol or real/lure
   answers with at least 95 percent parseability.
8. Answer-absent Panel F is at least 95 percent parseable and at least
   90 percent `UNKNOWN`.
9. Primary conflict rows across Panels D/E include at least 40 binary rows
   where label is derived-code or real-symbol/lure.
10. Non-holdout conflict rows include at least 10 derived-code labels and at
    least 10 real-symbol/lure labels.
11. Holdout conflict rows include at least 4 derived-code labels and at least
    4 real-symbol/lure labels.
12. Primary conflict parseability is at least 90 percent.
13. Candidate-score margin and final-output margin are reported before hidden
    search.

If the gate fails, the result is a behavior diagnostic only.

## Behavior Baselines

Report before hidden-state work:

- final next-token derived-code-minus-real-symbol margin;
- final next-token derived-code-minus-lure-symbol margin;
- candidate mean-logprob derived-code-minus-real-symbol margin;
- candidate mean-logprob derived-code-minus-lure-symbol margin;
- target row position;
- row-position code;
- prompt length and token count;
- authority dial value;
- split and source id;
- prompt template id.

## Signature Gate

Only after behavior passes, test hidden signatures at:

- row-code legend tokens;
- target table-row element tokens;
- target row-number token;
- query element token;
- post-table boundary;
- post-authority boundary;
- answer-prefix position;
- final prompt token.

Signature promotion requires the same MC008-style controls:

1. source-disjoint holdout AUC at least 0.80;
2. holdout AUC at least 0.85;
3. hidden score beats same-position output margin by at least 0.05;
4. hidden score beats final-output and candidate-score margins by at least
   0.02, unless explicitly labeled lead-time monitor only;
5. hidden score beats prompt/template/authority/token/row-position baselines
   by at least 0.05;
6. hidden score beats shuffled-label selected-search p95 by at least 0.05;
7. no subgroup collapse by row position, split, authority value, or source id.

## Intervention Gate

Intervention is forbidden until the signature gate passes, except for an
explicitly labeled known-confounded causal stress test.

Allowed intervention families:

1. Source/row-position path masking from row-number and target-row tokens to
   answer position.
2. Row-code legend path masking, with wrong-row and wrong-position controls.
3. Residual steering at a preregistered pre-output coordinate if it beats
   output/candidate/row-position controls.

Required controls:

- wrong row;
- wrong element;
- wrong row-code;
- wrong layer;
- wrong position;
- same-norm random vector/path;
- prompt deletion;
- neutral prompt rewrite;
- synthetic lookup rows;
- real-world memory control rows;
- answer-absent null rows.

## Mechanism Promotion Criteria

MC009 can become a mechanism-card candidate only if:

1. behavior gate passes;
2. signature gate passes;
3. intervention moves generated answers or derived-versus-real margins in the
   predicted direction on source-disjoint holdout;
4. wrong-row, wrong-code, wrong-layer, wrong-position, random, deletion, and
   rewrite controls do not match the target effect;
5. real-world control rows do not collapse into derived-code behavior;
6. synthetic lookup rows retain expected behavior;
7. answer-absent null rows remain local under preregistered margin strata;
8. output/candidate/row-position baselines do not explain the result;
9. allowed claim names the exact ordinal-code behavior contract.

## Bounded Success Criteria

MC009 should become a bounded atlas result if:

- behavior passes but signatures are output/candidate/row-position visible;
- source/path lead-time monitors exist but intervention fails;
- derived-code mediation appears only on synthetic/task rows and not on
  memory-conflict rows;
- intervention works on high-margin rows but null locality fails;
- derived-code construction repairs contrast but not final-state output
  visibility.

## Death Criteria

Kill the first MC009 route if:

1. synthetic ordinal code lookup fails;
2. real-world memory control fails;
3. answer-absent nulls fail after one membership repair;
4. conflict rows do not produce both derived-code and real/lure labels after
   one authority repair;
5. primary conflict parseability stays below 90 percent after one format
   repair;
6. candidate-score/final-output/row-position baselines perfectly explain
   holdout and no pre-output monitor survives nulls;
7. hidden selection fails shuffled-label selected-search nulls.

## Expected Diagnostic Labels

Possible result labels:

- `derived_code_behavior_passed`;
- `derived_code_synthetic_lookup_failed`;
- `derived_code_real_memory_control_failed`;
- `derived_code_null_failed`;
- `derived_code_conflict_contrast_absent`;
- `derived_code_conflict_parseability_failed`;
- `derived_code_row_position_confounded`;
- `derived_code_output_margin_confounded`;
- `derived_code_candidate_score_confounded`;
- `derived_code_lead_time_monitor_only`;
- `derived_code_signature_shuffle_overfit`;
- `derived_code_intervention_failed`;
- `derived_code_bounded_bridge_surface`;
- `derived_code_mechanism_candidate_supported`.

If any of these are promoted into the atlas, add canonical uppercase diagnostic
types to `data/control_surface_atlas.json`.

## Allowed Claims

If behavior passes:

- removing direct entity->code mappings can create a cleaner bridge substrate
  than MC008's printed source-value conflict.

If signatures pass:

- specify whether the signal is row-position, target-row, authority-state,
  memory-state, or output-shadowed.

If intervention passes:

- claim only the exact ordinal-code arbitration surface tested.

## Forbidden Claims

MC009 may not claim:

- a truth vector;
- a general factual recall mechanism;
- chemical knowledge control;
- source declarations as proxy labels;
- mechanism promotion from behavior contrast alone;
- mechanism promotion from a hidden classifier matched by output, candidate,
  row-position, or prompt-format baselines;
- transfer across models without a matched transfer panel.

## Required Outputs

Every run must produce:

- generated-answer behavior table;
- prompt leak audit;
- direct target->task-code leak audit;
- row-position balance report;
- split/source-disjoint report;
- parseability report;
- output/candidate/row-position baseline report;
- selected template report;
- if hidden-state work runs, signature report with shuffled-label null;
- if intervention runs, locality/null/side-effect report;
- status card with allowed and forbidden claims.

## Atlas Update Rule

MC009 should not add an atlas row until at least a behavior status card exists.

If the behavior gate fails, add a diagnostic note only after the result artifact
and status card are written.
