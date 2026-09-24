# MC008 Symbolic Fact-Code Arbitration Preregistration

Date: 2026-07-01

## Purpose

MC008 is the next bridge family after the MC007 route closeout.

MC005 shows that prompt-visible synthetic lookup can expose a bounded internal
source-value mediation surface. MC006 shows that capital-fact override quickly
becomes output/candidate visible and intervention-blocked. MC007 showed that
familiar country names plus artificial city values are not enough: terse table
authority collapses to lookup, authority pressure creates partial prior
contrast, and open city generation plus source declarations fail parseability
and source-city consistency.

MC008 changes the behavior family.

Instead of asking for open city names, MC008 asks for short symbolic factual
codes. The first target is chemical element symbols:

```text
Element: Sodium
Real-world answer: Na
Task-local answer: Qe
```

The real answer is learned parametric knowledge. The task-local answer is a
prompt-visible synthetic mapping. Both are compact symbolic strings with the
same response format. This directly tests whether the MC005-to-MC006 transition
depends on the open-city answer interface or on factual memory conflict itself.

## Relationship To Closed Routes

MC008 is not another MC007 prompt repair.

It changes:

- entity family: countries -> chemical elements;
- answer family: city names -> compact element-symbol-like codes;
- parser target: open lexical city strings -> short symbolic code strings;
- conflict type: country-capital geography -> element-symbol factual memory;
- artifact role: first new bridge family after `MC007_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE`.

MC008 keeps the core discipline:

- no true-answer prompt leaks in conflict rows;
- generated answers, not candidate scores alone;
- source-disjoint holdout;
- answer-absent null rows;
- candidate/output baselines before hidden-state work;
- no intervention until behavior and signature gates pass.

## Law Hypotheses Under Test

Primary hypotheses:

- `source_visible_lookup_localizes_more_than_parametric_override`;
- `final_state_output_geometry_dominance`;
- `behavior_substrate_first_or_everything_lies`;
- `lead_time_monitor_before_lever`;
- `null_reliability_bottleneck`.

Specific MC008 question:

> When the factual answer is a short symbolic code instead of an open city name,
> does the model produce a clean bridge substrate where prompt-local lookup,
> learned memory, and output geometry can be separated?

## Model

Initial target:

- `Qwen/Qwen3-1.7B`

Reason:

- MC005's bounded mechanism surface is on Qwen3-1.7B;
- MC006 and MC007 bridge diagnostics are on Qwen3-1.7B;
- the next bridge should hold model family fixed before claiming transfer.

No transfer claim is allowed from the initial run.

## Source Set

Use 40 common chemical elements whose symbols are likely known by the model and
whose true symbols are short and stable.

Candidate source examples:

- Hydrogen -> H
- Carbon -> C
- Oxygen -> O
- Sodium -> Na
- Magnesium -> Mg
- Aluminum -> Al
- Silicon -> Si
- Chlorine -> Cl
- Potassium -> K
- Calcium -> Ca
- Iron -> Fe
- Copper -> Cu
- Silver -> Ag
- Gold -> Au
- Lead -> Pb

The source list must be fixed in the runner. Holdout elements must be
source-disjoint from discovery and calibration elements.

## Artificial Code Construction

Artificial task codes must be symbol-like but not true element symbols.

Rules:

1. Each artificial code is one or two letters.
2. Codes use the same capitalization style as element symbols:
   uppercase first letter, optional lowercase second letter.
3. No artificial code may equal the target element's real symbol.
4. No artificial code may equal another element's true symbol if that would
   create an ambiguous real-symbol parser hit.
5. No two rows share the same artificial code inside the same split.
6. Token length and character length should be balanced between artificial and
   real symbols where possible.

Examples of allowable artificial codes depend on tokenizer and source list
checks. The runner must log rejected codes and the final mapping.

## Panels

MC008 must include at least six panels.

### Panel A: Synthetic Code Lookup

Nonce keys mapped to artificial symbol-like codes.

Purpose:

- verify that the prompt contract can support clean symbolic lookup without
  learned factual priors.

Gate:

- at least 90 percent artificial-code generated answers;
- answer-absent nulls for nonce keys must produce `UNKNOWN` at least 95
  percent.

### Panel B: Familiar Element Task Lookup

Real element names mapped to artificial codes. The prompt asks for the task
code according to the table.

Purpose:

- test whether familiar element names collapse to table keys like MC007 V1.

Expected labels:

- artificial code;
- real symbol;
- lure symbol;
- unknown;
- unparsed.

### Panel C: Real-World Memory Control

No task mapping for the target element. The prompt asks for the real-world
chemical symbol.

Purpose:

- verify the model can generate real symbols under the same answer format;
- quantify real-symbol parseability before conflict.

Gate:

- at least 85 percent real-symbol or accepted real-symbol/lure answers;
- parseability at least 95 percent.

### Panel D: Authority Dial Conflict

The prompt includes an artificial task-code table and asks for the answer under
an authority dial.

Example:

```text
For this task, the table may be more or less authoritative.
Authority of task table: 30 out of 100.
If the table is authoritative, use the task code.
If real chemistry is more authoritative, use the real chemical symbol.
Return only the code, or UNKNOWN.
```

Purpose:

- create artificial-versus-real contrast without printing the true real symbol;
- preserve generated answer parseability.

Dial values:

- 100;
- 70;
- 50;
- 30;
- 0.

### Panel E: Conflict Without Source Labels

The prompt creates conflict between task-local code and real chemistry, but
does not ask the model to declare which source it used.

Purpose:

- avoid the MC007 V4 failure where source declarations decoupled from generated
  city answers.

Required output:

```text
Answer: <code-or-UNKNOWN>
```

No `SOURCE:` field is allowed in the primary generated answer.

### Panel F: Answer-Absent Null

The prompt asks about an element or key absent from the table, or uses an
artificial code candidate not supported by the prompt.

Purpose:

- preserve MC005's null-locality discipline before any intervention;
- expose whether symbolic code outputs produce low-margin null instability.

Gate before intervention:

- null rows must be parseable at least 95 percent;
- `UNKNOWN` should be generated at least 90 percent;
- any non-UNKNOWN answer must be logged as a null-side outcome, not discarded.

## Output Parsing

Generated answers are primary.

Strict parser:

- read the first non-empty generated line;
- accept a bare one- or two-letter symbol-like code;
- accept `UNKNOWN`;
- reject explanatory sentences as `unparsed`;
- do not infer a code from later explanation text.

Lenient parser:

- may be used only as an audit;
- must not change pass/fail gates unless preregistered before the run.

Candidate scoring is a control, not the behavior label source.

## Prompt Leak Audit

Conflict rows must not print the target element's true symbol anywhere in the
prompt.

The runner must audit:

- true symbol absent from the prompt except in real-world control rows where
  no target answer is shown;
- artificial code appears only in the task mapping row for that element and in
  any allowed candidate-control fields;
- lure symbols do not equal the target true symbol;
- no target true symbol appears in instructions, examples, comments, or source
  labels.

If true-symbol leakage occurs, the row is invalid.

## Behavior Gate

No hidden-state work is allowed unless a selected template passes all of:

1. Structural checks pass.
2. At least 40 sources.
3. Source-disjoint holdout.
4. No true-symbol prompt leaks in conflict rows.
5. Synthetic Panel A passes at least 90 percent artificial-code adherence.
6. Real-world Panel C passes at least 85 percent real-symbol or real/lure
   answers with at least 95 percent parseability.
7. Answer-absent Panel F is at least 95 percent parseable and at least 90
   percent `UNKNOWN`.
8. Primary conflict rows across Panels D/E include at least 40 binary rows
   where label is artificial-code or real-symbol/lure.
9. Non-holdout conflict rows include at least 10 artificial-code labels and at
   least 10 real-symbol/lure labels.
10. Holdout conflict rows include at least 4 artificial-code labels and at
    least 4 real-symbol/lure labels.
11. Primary conflict parseability is at least 90 percent.
12. Candidate-score margin and final-output margin are reported before any
    hidden-state search.

If the gate fails, the result is a behavior diagnostic only.

## Behavior Baselines

Report these before hidden-state work:

- final next-token artificial-minus-real margin;
- final next-token artificial-minus-lure margin;
- candidate mean-logprob artificial-minus-real margin;
- candidate mean-logprob artificial-minus-lure margin;
- answer length and token count;
- artificial-code token frequency;
- real-symbol token frequency;
- element-name token count;
- prompt template id;
- authority dial value;
- split and source id.

The behavior table is not hidden-state-ready if candidate-score or final-output
margins perfectly separate holdout labels and no source/path lead-time target
is preregistered.

## Signature Gate

Only after behavior passes, test hidden signatures at:

- mapping element-name token positions;
- mapping artificial-code token positions;
- query element-name token positions;
- post-table line boundary;
- post-authority-instruction line boundary;
- answer-prefix position;
- final prompt token.

For each selected signature report:

- discovery AUC;
- calibration AUC;
- holdout AUC;
- source-disjoint holdout AUC;
- same-position output-margin AUC;
- final-output margin AUC;
- candidate-score margin AUC;
- prompt/template/authority baselines;
- token-length and code-frequency baselines;
- shuffled-label selected-search p95;
- subgroup AUC by synthetic, task-lookup, real-control, conflict, and null
  panels.

Signature promotion requires:

1. holdout AUC at least 0.85;
2. source-disjoint holdout AUC at least 0.80;
3. hidden score beats same-position output margin by at least 0.05;
4. hidden score beats final-output and candidate-score margins by at least
   0.02, unless explicitly labeled lead-time monitor only;
5. hidden score beats prompt/template/authority/token baselines by at least
   0.05;
6. hidden score beats shuffled-label selected-search p95 by at least 0.05;
7. no subgroup collapse on holdout conflict rows.

## Intervention Gate

Intervention is forbidden until the signature gate passes, except for an
explicitly labeled known-confounded causal stress test.

Allowed intervention families:

1. Source-value attention/write replacement if the selected surface is
   localized to mapping/query source paths.
2. Residual steering at a preregistered pre-output coordinate if it beats
   output/candidate controls.
3. Path masking from mapping element/code tokens to answer position, with
   deletion/rewrite-equivalence controls.

Required controls:

- wrong element;
- wrong code;
- wrong layer;
- wrong position;
- same norm random vector/path;
- prompt deletion;
- neutral prompt rewrite;
- synthetic lookup rows;
- real-world memory control rows;
- answer-absent null rows.

## Mechanism Promotion Criteria

MC008 can become a mechanism-card candidate only if:

1. behavior gate passes;
2. signature gate passes;
3. intervention moves generated answers or artificial-versus-real margins in
   the predicted direction on source-disjoint holdout;
4. wrong-source, wrong-code, wrong-layer, wrong-position, random, deletion, and
   rewrite controls do not match the target effect;
5. real-world control rows do not collapse into task-code behavior;
6. synthetic lookup rows retain expected behavior;
7. answer-absent null rows remain local under preregistered margin strata;
8. output/candidate baselines do not explain the result;
9. allowed claim names the exact behavior contract and does not generalize to
   capital facts or factual recall broadly.

## Bounded Success Criteria

MC008 should become a bounded atlas result if:

- behavior passes but signatures are output/candidate visible;
- source/path lead-time monitors exist but intervention fails;
- source-value mediation appears only on synthetic/task lookup rows and not on
  memory-conflict rows;
- intervention works on high-margin rows but null locality fails;
- symbolic code outputs repair parseability but not final-state output
  visibility.

## Death Criteria

Kill the first MC008 route if:

1. real-world memory control cannot produce real symbols at the gate threshold;
2. synthetic code lookup fails under the generated-answer contract;
3. conflict rows do not produce both artificial and real/lure labels after one
   template repair;
4. primary conflict parseability remains below 90 percent after one template
   repair;
5. candidate-score and final-output margins perfectly explain holdout and no
   pre-output/source-path monitor survives nulls;
6. hidden selection fails shuffled-label selected-search nulls;
7. interventions reproduce only deletion/rewrite effects or corrupt real-world
   control rows.

## Expected Diagnostic Labels

Possible result labels:

- `symbolic_bridge_behavior_passed`;
- `real_symbol_memory_control_failed`;
- `synthetic_code_lookup_failed`;
- `symbolic_conflict_contrast_absent`;
- `symbolic_conflict_parseability_failed`;
- `symbolic_output_margin_confounded`;
- `symbolic_candidate_score_confounded`;
- `symbolic_lead_time_monitor_only`;
- `symbolic_source_value_surface_supported`;
- `symbolic_signature_shuffle_overfit`;
- `symbolic_intervention_failed`;
- `symbolic_null_locality_failed`;
- `symbolic_bounded_bridge_surface`;
- `symbolic_mechanism_candidate_supported`.

If any of these are promoted into the atlas, add canonical uppercase diagnostic
types to `data/control_surface_atlas.json`.

## Allowed Claims

If behavior passes:

- compact symbolic factual codes can create a cleaner bridge substrate than
  open city generation.

If signatures pass:

- specify whether the signal is source-path, authority-state, memory-state, or
  output-shadowed.

If intervention passes:

- claim only the exact symbolic-code arbitration surface tested.

## Forbidden Claims

MC008 may not claim:

- a truth vector;
- a general factual recall mechanism;
- capital-fact control;
- source declarations as proxy labels;
- mechanism promotion from behavior contrast alone;
- mechanism promotion from a final-token hidden classifier matched by output
  or candidate scoring;
- transfer across models without a matched transfer panel.

## Required Outputs

Every run must produce:

- generated-answer behavior table;
- prompt leak audit;
- split/source-disjoint report;
- parseability report;
- output/candidate baseline report;
- selected template report;
- if hidden-state work runs, signature report with shuffled-label null;
- if intervention runs, locality/null/side-effect report;
- status card with allowed and forbidden claims.

## Atlas Update Rule

MC008 should not add an atlas row until at least a behavior status card exists.

If the behavior gate fails, add a diagnostic note only after the result artifact
and status card are written.

If the behavior gate passes but hidden-state work is blocked, update:

- `research/20_CONTROL_SURFACE_ATLAS.md`;
- `research/21_CONTROL_SURFACE_LAWS_AND_GAPS.md`;
- `data/control_surface_atlas.json`;
- `data/control_surface_law_hypotheses.json`.
