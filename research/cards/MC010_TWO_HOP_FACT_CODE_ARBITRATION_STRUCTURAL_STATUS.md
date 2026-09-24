# MC010 Two-Hop Fact-Code Arbitration Structural Status

Date: 2026-07-01

Status: structural gate passed; generated behavior not yet run.

## Artifact

Structural manifest:

> `results/cards/MC010/mc010_two_hop_fact_code_structural_20260701T083427.json`

Harness:

> `code/mc010_two_hop_fact_code_arbitration.py`

Preregistration:

> `research/prereg/MC010_TWO_HOP_FACT_CODE_ARBITRATION.md`

Command:

```powershell
python code\mc010_two_hop_fact_code_arbitration.py --write-manifest
```

## Result

The MC010 two-hop bridge scaffold passed all structural checks.

Counts:

- 800 records;
- 40 sources;
- 2 templates;
- 6 panels;
- 24 discovery sources;
- 8 calibration sources;
- 8 holdout sources.

Panel counts:

- `synthetic_two_hop_lookup`: 80;
- `familiar_entity_two_hop_lookup`: 80;
- `real_world_memory_control`: 80;
- `authority_dial_conflict`: 400;
- `unlabeled_conflict`: 80;
- `answer_absent_null`: 80.

Template counts:

- `two_hop_authority`: 400;
- `neutral_contract`: 400.

## Passed Structural Criteria

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

## What This Means

MC010 now has a structurally valid behavior-table candidate for the next bridge
run.

It directly addresses the top next-queue pressure:

> For the next bridge, require authority contrast, direct controls, nulls, and
> prompt-channel locality to pass together before probing.

The structural part of that requirement is satisfied. The generated behavior
part is not yet known.

## Allowed Claims

Allowed:

- MC010 removes direct entity-to-code rows.
- MC010 hides true symbols from conflict prompts.
- MC010 keeps one answer suffix across panels.
- MC010 has source-disjoint discovery/calibration/holdout source splits.
- MC010 is ready for generated-answer behavior scoring.

Forbidden:

- MC010 has passed the behavior gate.
- MC010 has authority contrast.
- MC010 has clean null behavior under generation.
- MC010 has a hidden-state signature.
- MC010 supports intervention.

## Next Step

Run Qwen3-1.7B generated behavior scoring and write:

> `research/cards/MC010_TWO_HOP_FACT_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

No hidden-state work is allowed until that generated behavior status passes all
predeclared gates.
