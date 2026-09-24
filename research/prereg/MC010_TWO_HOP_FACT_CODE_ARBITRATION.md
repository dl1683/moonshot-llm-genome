# MC010 Two-Hop Fact-Code Arbitration Preregistration

Date: 2026-07-01

## Purpose

MC010 is the next bridge candidate after the MC007, MC008, and MC009 route
closeouts.

The generated next-experiment queue says the next bridge must not proceed to
hidden-state work unless the same behavior table passes:

- authority contrast;
- direct controls;
- answer-absent nulls;
- parseability;
- source-disjoint balance;
- prompt-channel locality.

MC010 tests a materially different bridge construction.

MC008 attached an artificial code directly to a familiar entity. That made the
model overwhelmingly follow the prompt-local value. MC009 removed the direct
entity-to-code row and derived the task answer from row position, but the first
route failed conflict balance and the typed-slot repair exposed the source
channel while breaking controls.

MC010 uses a two-hop source path:

```text
element -> nonce handle -> task code
```

The prompt-local task answer is available, but not as a direct
`element -> code` row and not as a row-position answer slot. The competing
learned answer remains the real chemical symbol.

## Machine Scaffold

Harness:

> `code/mc010_two_hop_fact_code_arbitration.py`

Structural manifest:

> `results/cards/MC010/mc010_two_hop_fact_code_structural_20260701T083427.json`

Smoke command:

```powershell
python code\mc010_two_hop_fact_code_arbitration.py --limit-sources 10
```

Full structural manifest command:

```powershell
python code\mc010_two_hop_fact_code_arbitration.py --write-manifest
```

The full structural manifest currently passes all structural criteria:

- 800 records;
- 40 sources;
- 2 templates;
- 6 panels;
- 24 discovery sources;
- 8 calibration sources;
- 8 holdout sources;
- 80 synthetic two-hop lookup rows;
- 80 familiar-entity two-hop lookup rows;
- 80 real-world memory-control rows;
- 400 authority-dial conflict rows;
- 80 unlabeled-conflict rows;
- 80 answer-absent null rows.

## Model

Initial target:

- `Qwen/Qwen3-1.7B`

Reason:

- MC005, MC006, MC007, MC008, and MC009 bridge evidence is currently centered
  on Qwen3-1.7B;
- MC010 should change the behavior construction before changing the model
  family;
- no transfer claim is allowed from the first run.

## Source Set

Use the same 40 common chemical elements as MC008 and MC009.

Each source row contains:

- source id;
- element name;
- real chemical symbol;
- lure chemical symbol;
- nonce handle;
- prompt-local task code;
- synthetic nonce key;
- source-disjoint split.

The true symbol must never be printed in conflict prompts.

The prompt-local task code must never appear on the same line as the queried
element.

## Two-Hop Construction

For a target element, the prompt can contain:

```text
Entity handle table:
Sodium uses handle HND-27.
Carbon uses handle HND-22.
Oxygen uses handle HND-24.
Iron uses handle HND-37.

Handle code table:
HND-27 yields Ql.
HND-22 yields Qg.
HND-24 yields Qh.
HND-37 yields Xp.

Query entity: Sodium
```

If the local source path controls, the expected answer is `Ql`.

If ordinary chemistry controls, the expected answer is `Na`.

The task code is visible in the prompt, but not beside the entity. The model
must follow a source path through the handle.

## Templates

MC010 starts with two templates:

- `two_hop_authority`;
- `neutral_contract`.

Both use the same final answer suffix:

```text
Return only one final token: a task code, a chemical symbol, or UNKNOWN.
Do not explain.
Answer:
```

No typed answer slot is allowed. In particular, the prompt must not expose a
separate "task answer" field and "chemistry answer" field.

## Panels

### Panel A: Synthetic Two-Hop Lookup

Synthetic nonce keys point to handles; handles point to task codes.

Purpose:

- verify that the model can follow the two-hop source path without factual
  priors.

Behavior gate:

- at least 90 percent task-code answers;
- at least 95 percent parseability.

### Panel B: Familiar Entity Two-Hop Lookup

Real element names point to handles; handles point to task codes.

Purpose:

- test whether familiar entities can still behave as prompt-local source nodes
  when the task code is not directly attached to the entity.

Behavior gate:

- at least 90 percent task-code answers;
- at least 95 percent parseability.

### Panel C: Real-World Memory Control

No local source table is active. The prompt asks for the standard chemical
symbol.

Purpose:

- verify real-symbol generation under the same answer format.

Behavior gate:

- at least 85 percent real-symbol answers;
- at least 95 percent parseability.

### Panel D: Authority-Dial Conflict

The prompt includes both source tables and an authority dial:

- 100;
- 70;
- 50;
- 30;
- 0.

High authority should favor the task code. Low authority should favor the real
symbol. The 50 setting is diagnostic and may be mixed.

Purpose:

- test whether two-hop source availability plus explicit authority pressure can
  create a balanced task-code-versus-real-symbol conflict substrate.

Behavior gate:

- selected template must contain at least 40 parseable primary binary conflict
  rows;
- non-holdout primary conflict rows must contain at least 10 task-code and 10
  real/lure-symbol rows;
- holdout primary conflict rows must contain at least 4 task-code and 4
  real/lure-symbol rows.

### Panel E: Unlabeled Conflict

The prompt includes both source tables and says both local source tables and
ordinary chemistry may be relevant, but does not provide numeric source
authority.

Purpose:

- detect whether the authority dial itself is doing all the work.

Gate:

- report task-code, real-symbol, lure-symbol, unknown, and unparsed counts;
- do not use this panel alone for hidden-state work.

### Panel F: Answer-Absent Null

The handle-code table is present, but the queried element is absent from the
entity-handle table.

Purpose:

- verify source-locality and null behavior.

Behavior gate:

- at least 90 percent `UNKNOWN`;
- at least 95 percent parseability.

## Structural Gate

The structural harness must pass before any generated behavior run.

Current structural criteria:

- `expected_row_count`;
- `all_panels_present`;
- `all_templates_present`;
- `source_split_disjoint`;
- `holdout_sources_present`;
- `calibration_sources_present`;
- `no_direct_entity_code_lines`;
- `true_symbol_hidden_in_conflicts`;
- `answer_absent_omits_query_handle`;
- `candidate_answers_parseable`;
- `single_answer_suffix`;
- `task_codes_balanced_in_primary_splits`.

The current full structural manifest passes all of them.

## Generated Behavior Gate

Generated behavior may proceed only after the structural gate passes.

The behavior run must report:

- generated answer;
- parsed answer;
- task-code label;
- real-symbol label;
- lure-symbol label;
- unknown label;
- unparsed label;
- per-panel parseability;
- per-template label counts;
- source-disjoint split counts;
- output next-token margins where applicable;
- candidate-score margins for task code, real symbol, lure symbol, and
  `UNKNOWN`.

Hidden-state search remains forbidden unless a selected template passes all
behavior gates.

## Promotion Rule

MC010 can become a hidden-state substrate only if one selected template passes:

- Panel A synthetic two-hop lookup;
- Panel B familiar two-hop lookup;
- Panel C real-world memory control;
- Panel F answer-absent null;
- primary conflict parseability;
- primary binary row volume;
- non-holdout and holdout label balance;
- source-disjoint split;
- output/candidate baseline reporting.

## Death Rule

Close the route as diagnostic if:

- two-hop source paths remain table-dominant like MC008;
- authority contrast appears only by breaking direct controls or nulls;
- answer-absent nulls fail under otherwise good primary behavior;
- conflict balance only appears under a prompt-visible typed answer channel;
- output/candidate geometry fully explains the selected behavior table before
  hidden-state work.

## Allowed Claims Before Running

Allowed:

- MC010 is a structurally clean two-hop bridge candidate.
- The current scaffold removes direct entity-to-code rows.
- The current scaffold hides true symbols from conflict prompts.
- The current scaffold has source-disjoint discovery/calibration/holdout splits.
- The current scaffold is ready for generated-answer behavior scoring.

Forbidden:

- MC010 has a behavior substrate.
- MC010 has a hidden signature.
- MC010 supports intervention or steering.
- MC010 solves the bridge problem.
- MC010 proves a knowledge-control surface.

## Next Step

Run generated-answer behavior scoring on Qwen3-1.7B, then write:

> `research/cards/MC010_TWO_HOP_FACT_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

No hidden-state probe is allowed before that status card says the behavior gate
passed.
