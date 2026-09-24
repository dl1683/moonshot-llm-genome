# MC006 Parametric Fact Override V11 Chat-Signature Status

Status: hidden signal present, but signature promotion failed; requested-mode,
output-margin, and shuffled-label controls match it.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V11_CHAT_SIGNATURE.md`
- runner:
  `code/mc006_parametric_fact_override_v11_chat_signature.py`
- source behavior artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v10_chat_generated_20260630T232318.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v11_chat_signature_20260630T233253.json`
- result SHA256:
  `d61853edd2bea85b2b628e6fcff4a7bbc4ee3ebdc5c66318b64de1bf41edb43c`

## Verdict

MC006 V11 does not pass the hidden-signature gate. Do not start intervention
from this signature.

The run found a perfect final-prompt-token hidden classifier:

- primary rows: V10 clean source-level contrasts only;
- primary shape: 20 clean sources, 5 original holdout sources, 40 rows;
- selected hidden candidate: `layer_0` at final rendered prompt token;
- selected hidden holdout AUC: 1.000;
- holdout source real-vs-fictional ordering: 5/5 sources passed.

But easy controls matched it:

- output-margin holdout AUC: 1.000;
- requested-mode holdout AUC: 1.000;
- prompt-length holdout AUC: 0.880;
- shuffled-label selection p95: 1.000.

The diagnostic class is:

```text
requested_mode_confounded
```

The stricter interpretation is broader: V11 is requested-mode confounded,
output-margin confounded, and shuffle-selection confounded.

## Criteria

| Criterion | Result |
| --- | --- |
| structural checks pass | pass |
| selected hidden holdout AUC at least 0.85 | pass |
| selected hidden beats output margin by 0.02 | fail |
| selected hidden beats requested mode by 0.02 | fail |
| selected hidden beats prompt length by 0.02 | pass |
| selected hidden beats shuffled-selection p95 by 0.05 | fail |
| every holdout source has `mode_real` score greater than `mode_fictional` | pass |

## Interpretation

V11 shows that the V10 behavior table has a strong internal separation between
real-world mode and fictional-code mode. That separation is not mechanism-grade:
the prompt explicitly contains the requested mode, output logits separate the
rows perfectly, and shuffled layer selection can also reach perfect holdout AUC
on this small table.

MC006 should remain bounded as:

- V10: chat-rendered generated-answer behavior substrate passed;
- V11: hidden signal present but requested-mode/output/null-confounded.

The next MC006 repair should not start intervention. It should either build a
signature table where the critical labels vary under a matched requested-mode
surface, or move to an intervention test only after a signature beats
requested-mode, output-margin, and shuffled-selection controls.
