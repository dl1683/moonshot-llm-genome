# MC006 Parametric Fact Override V13 Real-After-Fiction Generated Status

Status: generated matched-surface behavior table nearly passed, but the
preregistered binary-volume gate failed by one row; no hidden-signature or
intervention claim is allowed.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V13_REAL_AFTER_FICTION_GENERATED.md`
- runner:
  `code/mc006_parametric_fact_override_v13_real_after_fiction_generated.py`
- source behavior artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v2_repair_20260630T222815.json`
- source diagnostic artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v12_real_after_fiction_signature_20260630T234703.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v13_real_after_fiction_generated_20260630T235712.json`
- result SHA256:
  `89c82723fe9e816757b96d78491f10a26f21a5ab8152bbec91dbd3d7008000a7`

## Verdict

MC006 V13 does not pass the generated matched-surface behavior-substrate gate.
Do not start hidden-state signature discovery or intervention from this table.

The diagnostic class is:

```text
binary_volume_failed
```

The selected template was `fake_mapping_warning`. It was selected using only
discovery/calibration rows by the preregistered balanced-binary rule:

```text
selection_key = [7, 21, -11, -2]
```

This is a useful near miss. The selected template had both true-answer and
fictional-code-following rows in non-holdout and holdout splits, and the
holdout rows were cleanly binary. But the total selected binary count was
`29/40`, below the `30/40` floor.

## Structural Checks

All structural checks passed:

- exactly 40 sources;
- exactly 6 templates;
- exactly 240 generated rows;
- exactly 40 rows per template;
- exactly 8 holdout rows per template;
- every source appeared once per template;
- no duplicate record ids;
- no duplicate candidate answers;
- no true-capital prompt leaks.

## Selected Template Results

Selected template: `fake_mapping_warning`.

| Metric | Result |
| --- | ---: |
| rows | 40 |
| strict parseable rows | 32 |
| binary rows | 29 |
| side rows | 11 |
| true answers | 20 |
| override answers | 9 |
| lure answers | 3 |
| unparsed rows | 8 |
| non-holdout true answers | 14 |
| non-holdout override answers | 7 |
| holdout true answers | 6 |
| holdout override answers | 2 |
| holdout side rows | 0 |
| prompt leaks | 0 |

Split labels for the selected template:

| Split | True | Override | Lure | Unparsed |
| --- | ---: | ---: | ---: | ---: |
| discovery | 11 | 4 | 3 | 6 |
| calibration | 3 | 3 | 0 | 2 |
| holdout | 6 | 2 | 0 | 0 |

## Criteria

| Criterion | Result |
| --- | --- |
| structural checks pass | pass |
| selected binary rows at least 30/40 | fail: 29/40 |
| non-holdout true rows at least 6 | pass: 14 |
| non-holdout override rows at least 6 | pass: 7 |
| holdout true rows at least 2 | pass: 6 |
| holdout override rows at least 2 | pass: 2 |
| holdout side rows at most 4/8 | pass: 0 |
| no true-capital prompt leak | pass |

## Interpretation

V13 repaired the biggest V12 behavior-table issue: source-disjoint holdout now
has both labels with enough rows, and the labels come from generated answers
rather than candidate scoring. The remaining blocker is strict output shape and
binary volume. Eleven selected-template rows were side rows: three lure answers
and eight unparsed responses, often because the model began with explanatory
text rather than a bare city name.

This means MC006 still lacks a passing generated-answer matched-surface
behavior table for learned capital facts versus fictional-code contamination.
The next repair should not start hidden-state work. It should repair the
generated-answer output contract or expand the matched template/source bank
until the selected table clears binary volume while preserving non-holdout and
holdout label balance.
