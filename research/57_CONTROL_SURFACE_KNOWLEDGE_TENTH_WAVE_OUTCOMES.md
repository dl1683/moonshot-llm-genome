# Control-Surface Knowledge Tenth-Wave Outcomes

Status: generated KSQ014 slash locality outcome layer.

Generated data:

> `data/control_surface_knowledge_tenth_wave_outcomes.json`

Builder:

> `code/control_surface_knowledge_tenth_wave_outcomes.py`

Commands:

```powershell
python code\control_surface_knowledge_tenth_wave_outcomes.py --write
python code\control_surface_knowledge_tenth_wave_outcomes.py
python code\validate_control_surface_atlas.py
```

## Summary

- outcomes: `1`
- structural rows checked: `440`
- smoke rows checked: `110`
- behavior-ready outcomes: `0`
- signature-screen licenses: `0`
- hidden-state licenses: `0`

## Diagnostic Chain

The slash result is now split. Catalog-labeled slash text has a clean 10-source locality smoke under the tested prompt contract, but bare slash notation is too parse-fragile. This licenses only a narrow full-source catalog-slash behavior test, not hidden-state work.

| Step | Outcome | Result | Boundary |
| ---: | --- | --- | --- |
| 1 | `ksq014_slash_locality_packet` | `SLASH_LOCALITY_BRIDGE_LOSS` | catalog slash bridge variants pass smoke locality; all slash-only controls abstain 10/10; bare slash bridge fails at 6/10 answer and 4/10 unparsed |

## Outcome

| Outcome | Selected Template | Exported Diagnostic | Behavior Ready | Hidden State | Key Boundary |
| --- | --- | --- | --- | --- | --- |
| `ksq014_slash_locality_packet` | `tag_rows` | `SLASH_LOCALITY_BRIDGE_LOSS` | `false` | `false` | catalog slash locality survives smoke; bare slash bridge fails |

## Selected Panel Counts

- exact bridge: `9/10` answer, `1/10` unparsed
- raw answer_for alternate: `9/10` reproduced, `1/10` unparsed
- catalog slash entity alternate: `9/10` bridge answer, `1/10` unparsed
- catalog slash decoy alternate: `9/10` bridge answer, `1/10` abstain
- bare slash entity alternate: `6/10` bridge answer, `4/10` unparsed
- catalog slash reversed alternate: `9/10` bridge answer, `1/10` unparsed
- catalog slash entity-only control: `10/10` abstain
- catalog slash decoy-only control: `10/10` abstain
- bare slash entity-only control: `10/10` abstain
- catalog slash reversed-only control: `10/10` abstain
- query-only control: `10/10` abstain

## Interpretation

KSQ014 makes the slash result useful by splitting it. Catalog-labeled slash text passes the missing no-bridge locality smoke; bare slash does not preserve bridge parseability. The next admissible step is a full-source catalog-slash-only packet, not hidden-state work.

## Forbidden Claims

- KSQ014 is behavior-ready.
- KSQ014 licenses hidden-state probing, intervention, or mechanism claims.
- Bare slash notation is a clean repair surface.
- Catalog slash notation is promoted without a full-source behavior run.

## Next Required Evidence

- Run a KSQ015 full-source catalog-slash-only packet without bare slash.
- Keep raw answer_for as the active positive-control pressure channel.
- Require full-source bridge preservation, slash-only abstention, source-disjoint holdout, and margins before any signature screen.
- Do not merge catalog slash and bare slash into one nonfunction category.
