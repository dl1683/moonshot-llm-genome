# MC006 Parametric Fact Override V9 Lenient Parse Audit Status

Status: lenient parser rescued parseability, but real-world mode and source
contrasts still failed.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V9_LENIENT_PARSE_AUDIT.md`
- runner:
  `code/mc006_parametric_fact_override_v9_lenient_parse_audit.py`
- source artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v8_generated_mode_20260630T230955.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v9_lenient_parse_audit_20260630T231503.json`
- result SHA256:
  `12ad8bc1d7e10bc4f171c940b41710775fe30b62605448a90670b34671d5def5`

## Verdict

MC006 V9 does not rescue the generated-answer substrate. The lenient parser
recovered several factual prose rows, but the behavior table remains far below
the real-world mode and source-contrast gates.

The diagnostic class is:

```text
real_mode_failed
```

## Structural Checks

All structural checks passed:

- source artifact run type was `parametric_fact_override_v8_generated_mode`;
- 44 source records;
- 22 sources;
- 5 original holdout sources;
- two conditions per source;
- 22 rows per condition;
- no duplicate candidate answers;
- V8 reported no true-capital prompt leaks.

## Behavior Results

| Criterion | Result |
| --- | --- |
| lenient parseable rows at least 40/44 | pass: 41/44 |
| `mode_real` true answer at least 20/22 | fail: 11/22 |
| `mode_fictional` override answer at least 20/22 | pass: 21/22 |
| clean source-level contrasts at least 18/22 | fail: 10/22 |
| holdout clean contrasts at least 4/5 | fail: 2/5 |

Compared with V8 strict parsing:

- parseable rows improved from 35/44 to 41/44;
- real-world true rows improved from 6/22 to 11/22;
- fictional-code override rows improved from 20/22 to 21/22;
- clean contrasts improved from 6/22 to 10/22;
- holdout clean contrasts improved from 1/5 to 2/5.

The parser changed 6 rows. Five `mode_real` rows became true-answer rows from
factual prose, and one `mode_fictional` row became an override row from prose.

## Interpretation

V9 shows that the V8 strict parser was indeed too narrow for some generated
answers. But the rescue is not large enough. Even after accepting full candidate
mentions anywhere in the generated suffix, real-world mode reaches only 11/22
and the source-level contrast gate remains 10/22.

MC006 should remain bounded as:

- V4: narrow prompt-bounded behavior substrate passed;
- V5: hidden signature present but output/prompt/null-confounded;
- V6: same-prompt-family two-source repair failed real-source reliability;
- V7: same-prompt-family mode-gated candidate scoring failed real-world mode;
- V8: same-prompt-family generated-answer repair failed parseability and
  real-world mode;
- V9: lenient parser improved V8 but still failed real-world mode and contrasts.

The next MC006 repair should change the prompt/rendering contract rather than
only changing the parser. Hidden-state discovery remains blocked.
