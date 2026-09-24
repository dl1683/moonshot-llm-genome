# MC006 Parametric Fact Override V8 Generated-Mode Status

Status: generated-answer repair failed; strict parseability and real-world mode
both remain below gate.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V8_GENERATED_MODE.md`
- runner:
  `code/mc006_parametric_fact_override_v8_generated_mode.py`
- source artifacts:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v4_source_selected_hybrid_20260630T223734.json`
  and
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v7_mode_gated_20260630T230257.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v8_generated_mode_20260630T230955.json`
- result SHA256:
  `f7b95d9a3d3ae841917f3f206eb88803b4f0bbd5088d4af08a4d16a889977ffb`

## Verdict

MC006 V8 does not pass the generated-answer behavior gate. Removing hidden
candidate scoring did not recover a same-prompt-family knowledge/control
substrate.

The diagnostic class is:

```text
parseability_failed
```

The stricter interpretation is that V8 also failed real-world mode and source
contrasts.

## Structural Checks

All structural checks passed:

- 22 sources;
- 5 original holdout sources;
- 44 rows;
- two conditions per source;
- 22 rows per condition;
- 22 rows per requested mode;
- valid source splits;
- no duplicate row IDs or candidate strings;
- no true-capital prompt leaks.

## Behavior Results

| Criterion | Result |
| --- | --- |
| parseable rows at least 40/44 | fail: 35/44 |
| `mode_real` true answer at least 20/22 | fail: 6/22 |
| `mode_fictional` override answer at least 20/22 | pass: 20/22 |
| clean source-level contrasts at least 18/22 | fail: 6/22 |
| holdout clean contrasts at least 4/5 | fail: 1/5 |
| true capital absent from prompt | pass |

The generated interface changed the failure mode but did not repair it:

- `mode_real`: 6 true, 9 override, 7 unparsed;
- `mode_fictional`: 20 override, 2 unparsed;
- 9 total rows were unparsed by the strict first-line prefix rule.

Several unparsed real-world rows stated the true fact in prose, for example
`The actual capital of Germany is Berlin.` Those rows still fail the V8 parser
because the prompt required only the city name. This is a useful diagnostic:
raw generation can expose latent factual text, but the current prompt does not
produce a reliable compact behavior substrate.

## Interpretation

V8 shows that candidate scoring was not the only MC006 blocker. Greedy
generation reduces the hidden answer-choice concern, but it introduces
format/parse failures and still leaves real-world mode far below threshold.

MC006 should remain bounded as:

- V4: narrow prompt-bounded behavior substrate passed;
- V5: hidden signature present but output/prompt/null-confounded;
- V6: same-prompt-family two-source repair failed real-source reliability;
- V7: same-prompt-family mode-gated candidate scoring failed real-world mode;
- V8: same-prompt-family generated-answer repair failed parseability and
  real-world mode.

The next MC006 repair should test whether a stricter output contract, chat
rendering, or a preregistered lenient parser can recover real-world mode without
weakening fictional-code following. Do not start hidden-state work from V8.
