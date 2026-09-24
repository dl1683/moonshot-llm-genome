# Control-Surface Knowledge Third-Wave Outcomes

Status: generated post-second-wave diagnostic outcome layer.

Generated data:

> `data/control_surface_knowledge_third_wave_outcomes.json`

Builder:

> `code/control_surface_knowledge_third_wave_outcomes.py`

Commands:

```powershell
python code\control_surface_knowledge_third_wave_outcomes.py --write
python code\control_surface_knowledge_third_wave_outcomes.py
python code\validate_control_surface_atlas.py
```

## Summary

- outcomes: `2`
- completed outcomes: `2`
- structural rows checked: `2040`
- smoke rows checked: `510`
- behavior-ready outcomes: `0`
- signature-screen licenses: `0`
- hidden-state licenses: `0`
- intervention licenses: `0`

## Diagnostic Chain

The current real-uncertainty bottleneck is not exhausted by familiar priors or value mentions. A slot-binding assertion can become an answer channel even when the prompt marks it as outside the counted evidence contract.

| Step | Outcome | Result | Boundary |
| ---: | --- | --- | --- |
| 1 | `ksq007_nonce_evidence_answerability` | `NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY` | claim-only rows still reproduce the nonce value too often |
| 2 | `ksq007b_claim_channel_boundary` | `ANSWER_FOR_SYNTAX_CLAIM_LEAK` | bare answer_for(entity)=value syntax remains strongly answer-bearing |

## Outcomes

| Outcome | Selected Template | Exported Diagnostic | Behavior Ready | Hidden State | Key Boundary |
| --- | --- | --- | --- | --- | --- |
| `ksq007_nonce_evidence_answerability` | `evidence_rows` | `NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY` | `false` | `false` | claim-only reproduced `6/10` under nonce evidence |
| `ksq007b_claim_channel_boundary` | `counted_uncounted_sections` | `ANSWER_FOR_SYNTAX_CLAIM_LEAK` | `false` | `false` | bare `answer_for(entity)=value` reproduced `7/10` |

## Forbidden Claims

- KSQ007 or KSQ007B is behavior-ready.
- KSQ007 or KSQ007B licenses hidden-state probing, intervention, or mechanism claims.
- Removing real-world priors is sufficient to solve answerability.
- Section labels alone solve claim-channel leakage.

## Next Required Evidence

- A behavior substrate that preserves exact evidence answering while suppressing bare slot-binding claims.
- A source-disjoint full run after any claim-channel repair.
- Candidate/output margins on the selected repaired template.
- Only after behavior repair: hidden signatures that beat prompt/output/candidate baselines.
