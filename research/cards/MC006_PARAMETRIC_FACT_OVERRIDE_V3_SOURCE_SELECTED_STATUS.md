# MC006 Parametric Fact Override V3 Source-Selected Status

Status: source-selected paraphrase test failed; real-world side passed, weakened fictional-code pressure failed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V3_SOURCE_SELECTED.md`
- runner:
  `code/mc006_parametric_fact_override_v3_source_selected.py`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v3_source_selected_20260630T223516.json`
- result SHA256:
  `d2483434bf26a89d09fe1c3223bc6d7914eb1945e9409d43aaf5a6fc35d1f4b1`

## Verdict

MC006 V3 does not pass the source-selected behavior-substrate gate.

The important positive result is that the V2 three-way source-selected lead
survived the real-world paraphrases:

- `direct_real_paraphrase` selected true on 23/23 rows;
- `true_fact_paraphrase` selected true on 23/23 rows;
- `false_claim_check` selected true on 22/23 rows;
- `false_claim_check` selected override on 0/23 rows.

The failure was the weakened fictional-code prompt:

- `fictional_code_lookup` selected override on only 18/23 rows;
- clean source-level contrasts were 17/23, below the 18-source floor;
- holdout clean contrasts were 5/6, passing.

The diagnostic class is:

```text
fictional_code_pressure_failed
```

## Condition Summary

| Condition | Expected | True | Override | Lure | UNKNOWN | Expected-correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `direct_real_paraphrase` | true | 23 | 0 | 0 | 0 | 23/23 |
| `true_fact_paraphrase` | true | 23 | 0 | 0 | 0 | 23/23 |
| `false_claim_check` | true | 22 | 0 | 1 | 0 | 22/23 |
| `fictional_code_lookup` | override | 4 | 18 | 1 | 0 | 18/23 |

## Criteria

| Criterion | Result |
| --- | --- |
| direct-real true at least 20/23 | pass |
| true-fact true at least 22/23 | pass |
| false-claim true at least 20/23 | pass |
| false-claim override at most 3/23 | pass |
| fictional override at least 20/23 | fail |
| clean contrast sources at least 18 | fail |
| holdout clean contrasts at least 5 | pass |

## Interpretation

V3 is a negative boundary, not a hidden-state discovery substrate. It shows that
the source-selected real-world capital side is robust to paraphrase, but the
fictional-code override side is prompt-sensitive.

The next repair should change only the fictional-code wording. Do not weaken or
replace the real-world side until the one-change fictional-pressure repair is
tested.
