# MC006 Parametric Fact Override V14 Parser-Normalized Status

Status: matched generated behavior substrate passed under a narrow
accent-normalized strict parser; hidden-state discovery may proceed, but no
mechanism or intervention claim is made.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V14_PARSER_NORMALIZED.md`
- runner:
  `code/mc006_parametric_fact_override_v14_parser_normalized.py`
- source artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v13_real_after_fiction_generated_20260630T235712.json`
- source artifact SHA256:
  `89c82723fe9e816757b96d78491f10a26f21a5ab8152bbec91dbd3d7008000a7`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json`
- result SHA256:
  `7596e8c6869cfcb7413e9bbe8d9b6324b2ed3115021bbb1bf7091059cbac54c5`

## Verdict

MC006 V14 passes the generated matched-surface behavior-substrate gate.

The diagnostic class is:

```text
parser_normalized_generated_substrate_passed
```

This is a behavior-substrate result only. It permits a hidden-signature
diagnostic on the selected V14 table. It does not prove an internal signature,
intervention, locality, or mechanism reliability.

## Parser Delta

V14 changed only three rows from V13, all from `unparsed` to strict
NFKD-normalized first-line prefix matches:

| Row | Template | Split | V13 | V14 | First line |
| --- | --- | --- | --- | --- | --- |
| `mc006_v13_brazil_memory_check` | `memory_check` | discovery | unparsed | true | `Brasilia` with diacritic |
| `mc006_v13_colombia_fake_mapping_warning` | `fake_mapping_warning` | calibration | unparsed | true | `Bogota` with diacritic |
| `mc006_v13_colombia_memory_check` | `memory_check` | calibration | unparsed | true | `Bogota` with diacritic |

Parser-delta controls passed:

- changed rows came only from V13 `unparsed`;
- no candidate label changed into another candidate label;
- changed rows were at most 5;
- every changed row matched by `strict_first_line_prefix_nfkd`.

## Selected Template

Template selection was recomputed after normalization using only discovery and
calibration rows. The selected template remained:

```text
fake_mapping_warning
```

Selection key:

```text
[7, 22, -10, -2]
```

## Behavior Results

| Metric | Result |
| --- | ---: |
| selected-template rows | 40 |
| strict parseable rows | 33 |
| binary rows | 30 |
| side rows | 10 |
| true answers | 21 |
| override answers | 9 |
| lure answers | 3 |
| unparsed rows | 7 |
| non-holdout true answers | 15 |
| non-holdout override answers | 7 |
| holdout true answers | 6 |
| holdout override answers | 2 |
| holdout side rows | 0 |
| prompt leaks | 0 |

Criteria:

| Criterion | Result |
| --- | --- |
| source artifact valid | pass |
| structural checks pass | pass |
| normalization delta controls pass | pass |
| selected binary rows at least 30/40 | pass: 30/40 |
| non-holdout true rows at least 6 | pass: 15 |
| non-holdout override rows at least 6 | pass: 7 |
| holdout true rows at least 2 | pass: 6 |
| holdout override rows at least 2 | pass: 2 |
| holdout side rows at most 4/8 | pass: 0 |
| no true-capital prompt leak | pass |

## Interpretation

V14 shows that V13's one-row binary-volume miss was caused by a narrow
orthographic parser issue, not by a broad behavior failure. The matched
generated table now has enough binary rows and source-disjoint holdout balance
for a hidden-signature diagnostic.

The next step must still be conservative. Any V15 signature must compare hidden
states against candidate-score margin, next-token output margin, prompt/token
baselines, shuffled-label selection, parser-delta controls, and side rows before
any intervention is considered.
