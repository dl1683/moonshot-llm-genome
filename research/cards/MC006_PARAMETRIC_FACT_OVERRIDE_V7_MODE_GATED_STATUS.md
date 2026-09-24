# MC006 Parametric Fact Override V7 Mode-Gated Status

Status: same-prompt-family mode repair failed; fictional-code contamination
dominates real-world mode.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V7_MODE_GATED.md`
- runner:
  `code/mc006_parametric_fact_override_v7_mode_gated.py`
- source artifacts:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v4_source_selected_hybrid_20260630T223734.json`
  and
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v6_condition_balanced_20260630T225327.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v7_mode_gated_20260630T230257.json`
- result SHA256:
  `eac13cd76b248c58bb9526d4674345b98b42baa59699411af352d8f8fe741a08`

## Verdict

MC006 V7 does not pass the same-prompt-family behavior gate. It removed the
true capital from the prompt and asked the model to choose between
`REAL_WORLD_CAPITAL` and `FICTIONAL_CITY_CODE` modes, but the fictional code
line dominated real-world mode.

The diagnostic class is:

```text
real_mode_failed
```

## Structural Checks

All structural checks passed:

- 22 sources;
- 5 original holdout sources;
- 44 rows;
- two conditions per source;
- 22 rows per condition;
- 22 rows per requested mode;
- valid source splits;
- no duplicate row IDs or candidate answers;
- valid candidate token counts;
- no true-capital prompt leaks.

## Behavior Results

| Criterion | Result |
| --- | --- |
| `mode_real` true answer at least 20/22 | fail: 8/22 |
| `mode_fictional` override answer at least 20/22 | pass: 22/22 |
| clean source-level contrasts at least 18/22 | fail: 8/22 |
| holdout clean contrasts at least 4/5 | fail: 1/5 |
| true capital absent from prompt | pass |

The failure is stronger than V6 on the real-world side. In `mode_real`, the
model selected the fictional override on 13/22 rows and a lure on 1/22 rows.
In `mode_fictional`, it selected the fictional override on 22/22 rows.

## Interpretation

V7 shows that this candidate-scored Qwen3-1.7B interface is highly sensitive to
the presence of a task-local fictional city line. Removing the true capital from
the prompt did not recover parametric fact use under an explicit real-world
mode request.

MC006 should remain bounded as:

- V4: narrow prompt-bounded behavior substrate passed;
- V5: hidden signature present but output/prompt/null-confounded;
- V6: same-prompt-family two-source repair failed real-source reliability;
- V7: same-prompt-family mode-gated repair failed real-world mode reliability.

The next MC006 repair should not proceed to hidden-state work. It should either
change the scoring interface away from exposed alternate city candidates or test
whether instruction-tuned/chat rendering can recover real-world mode without
destroying fictional-code following.
