# KSQ004 Template-Invariance Adjudication Status

Date: 2026-07-02

Runner:

> `code/ksq004_template_invariance_adjudication.py`

Result:

> `results/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION/ksq004_template_invariance_adjudication_full_behavior.json`

Status: template_invariant_bridge_behavior.

## Verdict

- route decision: `admit_behavior_substrate_only`
- exported diagnostic: `TEMPLATE_INVARIANT_BRIDGE_BEHAVIOR`
- behavior ready: `true`
- signature screen allowed: `true`
- hidden-state claim allowed: `false`
- intervention allowed: `false`
- passing templates: `["question_form", "relation_key_form"]`

## Template Gates

| Template | Passed | Conflict Expected | Atomic-Branch Atomic | Holdout Expected | Null UNKNOWN | Side Answer |
| --- | --- | --- | --- | --- | --- | --- |
| `question_form` | `true` | `0.850` | `0.700` | `0.875` | `1.000` | `0.000` |
| `neutral_sentence_form` | `false` | `0.675` | `0.525` | `0.750` | `0.925` | `0.000` |
| `relation_key_form` | `true` | `0.975` | `0.950` | `0.875` | `1.000` | `0.000` |

## Selected First-Run-Style Summary

- selected template: `relation_key_form`
- first-run-style diagnostic class: `bridge_answer_interface_behavior_ready`
- first-run-style behavior ready: `true`

## Boundary

The adjudication is behavior-only. It either admits a later signature
screen or kills the same-family answer-interface repair route. It does
not itself show an internal bridge, intervention, or mechanism.

## Forbidden Claims

- KSQ004 adjudication is a mechanism card.
- KSQ004 adjudication supports intervention.
- KSQ004 adjudication found an internal bridge or knowledge-control surface.
- A single passing template is template-invariant bridge behavior.
