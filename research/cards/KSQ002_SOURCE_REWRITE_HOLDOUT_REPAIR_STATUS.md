# KSQ002 Source-Rewrite Holdout Repair Status

Date: 2026-07-02

Runner:

> `code/ksq002_source_rewrite_holdout_repair.py`

Result:

> `results/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR/ksq002_source_rewrite_holdout_repair_full_behavior.json`

Status: source_rewrite_repair_broke_locality_controls.

## Verdict

- route decision: `kill_ordinary_source_rewrite_repair`
- exported diagnostic: `SOURCE_REWRITE_REPAIR_LOCALITY_REGRESSION`
- behavior ready: `false`
- signature screen allowed next: `false`
- hidden-state claim allowed: `false`
- intervention allowed: `false`

## Behavior Gates

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `false` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `baseline_source_value_lookup_passed` | `false` |
| `neutral_rewrite_lookup_passed` | `false` |
| `source_deletion_passed` | `false` |
| `query_only_control_passed` | `true` |
| `source_disjoint_rewrite_holdout_passed` | `true` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `city_field_rewrite`
- rewrite delta from baseline: `0.250`
- source-disjoint rewrite holdout label counts: `{"artificial_value": 16}`
- source-disjoint rewrite holdout parseability: `1.000`
- source-disjoint rewrite holdout artificial-value rate: `1.000`

| Panel | Label counts |
| --- | --- |
| `baseline_source_value_lookup` | `{"artificial_value": 30, "unparsed": 10}` |
| `neutral_rewrite_lookup` | `{"artificial_value": 40}` |
| `source_deletion` | `{"unknown": 10, "unparsed": 30}` |
| `query_only_control` | `{"lure_value": 7, "real_prior": 13, "unparsed": 20}` |
| `source_disjoint_rewrite_holdout` | `{"artificial_value": 40}` |
| `rewrite_output_geometry_audit` | `{"artificial_value": 40}` |

## Boundary

The repair is not behavior-ready. It fixed the named source-disjoint
rewrite holdout boundary but damaged other admission gates, so ordinary
KSQ002 source-rewrite repair is killed rather than promoted. No internal
source-channel signature, causal intervention, or knowledge-control
mechanism follows from this run.

## Forbidden Claims

- KSQ002 repair is a mechanism card.
- KSQ002 repair supports intervention.
- KSQ002 repair found an internal source-channel or knowledge-control surface.
- A behavior pass is itself a hidden-state or causal result.
