# Control-Surface Knowledge Fifth-Wave Outcomes

Status: generated KSQ009 schema-specific value-lookup outcome layer.

Generated data:

> `data/control_surface_knowledge_fifth_wave_outcomes.json`

Builder:

> `code/control_surface_knowledge_fifth_wave_outcomes.py`

Commands:

```powershell
python code\control_surface_knowledge_fifth_wave_outcomes.py --write
python code\control_surface_knowledge_fifth_wave_outcomes.py
python code\validate_control_surface_atlas.py
```

## Summary

- outcomes: `1`
- structural rows checked: `1200`
- smoke rows checked: `300`
- behavior-ready outcomes: `0`
- signature-screen licenses: `0`
- hidden-state licenses: `0`

## Diagnostic Chain

The fifth-wave repair fails more sharply than KSQ008. The schema-specific ALLOW contract is visible enough for absent, unrelated, and query-only abstention, but not strong enough to make positive ALLOW lookup reliable or suppress the old answer_for micro-language. The answer_for string remains an answer-bearing channel even when declared out of schema.

| Step | Outcome | Result | Boundary |
| ---: | --- | --- | --- |
| 1 | `ksq009_schema_specific_value_lookup` | `SCHEMA_SPECIFIC_POSITIVE_FAILED` | selected kv_lines template answers exact ALLOW rows 2/10, counted answer_for rows reproduce 9/10, uncounted answer_for rows reproduce 7/10, and uncounted answer_for alternates override ALLOW rows 9/10 |

## Outcome

| Outcome | Selected Template | Exported Diagnostic | Behavior Ready | Hidden State | Key Boundary |
| --- | --- | --- | --- | --- | --- |
| `ksq009_schema_specific_value_lookup` | `kv_lines` | `SCHEMA_SPECIFIC_POSITIVE_FAILED` | `false` | `false` | exact ALLOW value answered `2/10`; counted `answer_for` reproduced `9/10`; uncounted `answer_for` reproduced `7/10`; uncounted `answer_for` alternate overrode ALLOW `9/10` |

## Selected Panel Counts

- exact ALLOW value: `2/10` answer, `8/10` abstain
- counted wrong-schema answer_for: `9/10` reproduced, `1/10` abstain
- uncounted wrong-schema answer_for: `7/10` reproduced, `3/10` abstain
- ALLOW value versus uncounted answer_for alternate: `9/10` answer_for override, `1/10` unparsed
- absent ALLOW, unrelated ALLOW, and query-only controls abstained `10/10`
- uncounted ALLOW reproduced `2/10`; quoted ALLOW reproduced `4/10`; conflicting ALLOW rows selected a value `5/10`

## Forbidden Claims

- KSQ009 is behavior-ready.
- KSQ009 licenses hidden-state probing, intervention, or mechanism claims.
- Schema-specific ALLOW labels solve answerability or answer_for leakage.

## Next Required Evidence

- Stop treating row labels alone as a likely repair for answerability.
- Build an interface where target extraction and counted evidence are separated from answer-like syntax.
- Measure whether the answer_for channel is a lexical parser prior, an output-format prior, or a prompt-contract prior before another repair attempt.
