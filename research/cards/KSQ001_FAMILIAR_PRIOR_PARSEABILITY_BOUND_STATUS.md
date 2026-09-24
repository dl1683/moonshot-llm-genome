# KSQ001 Familiar-Prior Parseability Bound Status

Date: 2026-07-02

Runner:

> `code/ksq001_familiar_prior_parseability_bound.py`

Result:

> `results/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND/ksq001_familiar_prior_parseability_bound_full_behavior.json`

Status: parseability_repair_failed.

## Verdict

- route decision: `closeout_familiar_prior_parseability_tradeoff`
- exported diagnostic: `FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF`
- behavior ready: `false`
- signature screen allowed: `false`
- hidden-state claim allowed: `false`
- intervention allowed: `false`
- selected template: `compact_original_replay`
- passing templates: `[]`
- collapse mode: `None`

## Template Gates

| Template | Passed | Parseable | Artificial | Prior/Lure | UNKNOWN | Collapse |
| --- | --- | --- | --- | --- | --- | --- |
| `compact_original_replay` | `false` | `0.600` | `0.450` | `0.050` | `0.100` | `None` |
| `compact_begin_city` | `false` | `0.825` | `0.525` | `0.000` | `0.300` | `None` |
| `compact_city_name_first` | `false` | `0.875` | `0.075` | `0.000` | `0.800` | `None` |

## Selected Primary

- label counts: `{"artificial_value": 18, "real_prior": 2, "unknown": 4, "unparsed": 16}`
- first-run compact parseability: `0.600`
- parseability delta: `0.000`

## Boundary

This is behavior-only. It can close or admit the KSQ001 behavior
substrate, but it does not show an internal signature, causal
intervention, or knowledge-control surface.

## Forbidden Claims

- KSQ001 bound is a mechanism card.
- KSQ001 bound supports intervention.
- KSQ001 bound found an internal knowledge-control surface.
- A parser or answer-shape repair is sufficient without branch accounting.
