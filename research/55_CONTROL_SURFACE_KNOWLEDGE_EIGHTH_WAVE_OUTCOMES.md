# Control-Surface Knowledge Eighth-Wave Outcomes

Status: generated KSQ012 function-assignment wrapper repair outcome layer.

Generated data:

> `data/control_surface_knowledge_eighth_wave_outcomes.json`

Builder:

> `code/control_surface_knowledge_eighth_wave_outcomes.py`

Commands:

```powershell
python code\control_surface_knowledge_eighth_wave_outcomes.py --write
python code\control_surface_knowledge_eighth_wave_outcomes.py
python code\validate_control_surface_atlas.py
```

## Summary

- outcomes: `1`
- structural rows checked: `480`
- smoke rows checked: `120`
- behavior-ready outcomes: `0`
- signature-screen licenses: `0`
- hidden-state licenses: `0`

## Diagnostic Chain

Simple wrapper or placement text does not reliably quarantine the function-assignment answer channel. Splitting function and value can make no-bridge controls abstain, but it does not reliably preserve bridge answering. The boundary is now a wrapper-control leak plus repair leak, not an exact syntax issue.

| Step | Outcome | Result | Boundary |
| ---: | --- | --- | --- |
| 1 | `ksq012_function_assignment_wrapper_repair` | `FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK` | exact bridge answers 9/10; raw answer_for reproduces 9/10; inactive/comment/fence/below-cut wrappers override 8/10 to 9/10; detached overrides 7/10; unrelated-entity answer_for overrides 6/10; assignment-only inactive leaks 8/10 |

## Outcome

| Outcome | Selected Template | Exported Diagnostic | Behavior Ready | Hidden State | Key Boundary |
| --- | --- | --- | --- | --- | --- |
| `ksq012_function_assignment_wrapper_repair` | `tag_rows` | `FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK` | `false` | `false` | exact bridge and raw answer_for both passed; simple wrappers and assignment-only inactive controls leaked |

## Selected Panel Counts

- exact bridge: `9/10` answer, `1/10` unparsed
- raw answer_for alternate: `9/10` reproduced, `1/10` unparsed
- inactive block answer_for alternate: `9/10` override, `1/10` unparsed
- comment-mark answer_for alternate: `8/10` override, `1/10` abstain, `1/10` unparsed
- fenced-text answer_for alternate: `9/10` override, `1/10` abstain
- below-cut answer_for alternate: `9/10` override, `1/10` unparsed
- detached function then value: `7/10` override, `2/10` bridge answer, `1/10` abstain
- masked function plus value bank: `3/10` bridge answer, `6/10` abstain, `1/10` unparsed
- unrelated-entity answer_for alternate: `6/10` override, `4/10` bridge answer
- assignment-only inactive control: `8/10` reproduced, `1/10` abstain, `1/10` unparsed
- split-assignment-only control: `10/10` abstain
- query-only control: `10/10` abstain

## Forbidden Claims

- KSQ012 is behavior-ready.
- KSQ012 licenses hidden-state probing, intervention, or mechanism claims.
- Wrapper text reliably quarantines function-like assignment syntax.

## Next Required Evidence

- Treat visible function-like assignment text as unsafe unless a wrapper control proves otherwise.
- Search for a materially different nonfunction representation instead of adding more weak wrappers.
- If split/masked variants are reused, require both bridge preservation and no-bridge abstention across full-source holdout before hidden-state screens.
