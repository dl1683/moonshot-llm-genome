# Control-Surface Knowledge Sixth-Wave Outcomes

Status: generated KSQ010 two-stage codebook outcome layer.

Generated data:

> `data/control_surface_knowledge_sixth_wave_outcomes.json`

Builder:

> `code/control_surface_knowledge_sixth_wave_outcomes.py`

Commands:

```powershell
python code\control_surface_knowledge_sixth_wave_outcomes.py --write
python code\control_surface_knowledge_sixth_wave_outcomes.py
python code\validate_control_surface_atlas.py
```

## Summary

- outcomes: `1`
- structural rows checked: `1440`
- smoke rows checked: `360`
- behavior-ready outcomes: `0`
- signature-screen licenses: `0`
- hidden-state licenses: `0`

## Diagnostic Chain

The two-stage codebook interface moves the positive branch closer to viability, but it does not repair the substrate. The model can sometimes follow the entity-code/code-value path, yet the same prompt contract fails conflict abstention, locality, quoted-row exclusion, and answer_for competition.

| Step | Outcome | Result | Boundary |
| ---: | --- | --- | --- |
| 1 | `ksq010_two_stage_codebook_value_lookup` | `CODEBOOK_POSITIVE_FAILED` | selected tag_rows template answers exact bridges 8/10 but leaves 2/10 unparsed; both conflict panels select a value 10/10; answer_for-only rows reproduce 10/10; answer_for alternates override bridges 9/10 |

## Outcome

| Outcome | Selected Template | Exported Diagnostic | Behavior Ready | Hidden State | Key Boundary |
| --- | --- | --- | --- | --- | --- |
| `ksq010_two_stage_codebook_value_lookup` | `tag_rows` | `CODEBOOK_POSITIVE_FAILED` | `false` | `false` | exact bridges answered `8/10`; conflict panels selected values `10/10`; answer_for-only reproduced `10/10`; answer_for alternate overrode bridge `9/10` |

## Selected Panel Counts

- exact codebook bridge: `8/10` answer, `2/10` unparsed
- missing entity-code: `9/10` abstain, `1/10` unparsed
- missing code-value: `10/10` unparsed
- unrelated entity-code and query-only controls abstained `10/10`
- conflicting entity-code and code-value panels selected a value `10/10` each
- counted answer_for-only reproduced `10/10`
- bridge versus answer_for alternate: `9/10` answer_for override, `1/10` unparsed
- uncounted code-value reproduced `4/10`; quoted bridge reproduced `3/10`

## Forbidden Claims

- KSQ010 is behavior-ready.
- KSQ010 licenses hidden-state probing, intervention, or mechanism claims.
- Two-stage codebook rows solve answerability or answer_for leakage.

## Next Required Evidence

- Stop treating positive lookup alone as a useful admission signal for this branch.
- Probe whether answer_for competition is lexical, answer-position, or value-salience driven with answer_for string ablations.
- Repair conflict abstention and row locality before any full behavior run or hidden-state screen.
