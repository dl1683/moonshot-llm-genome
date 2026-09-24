# MC006 Parametric Fact Override V6 Condition-Balanced Status

Status: condition-balanced behavior repair failed; do not start hidden-state
discovery from V6.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V6_CONDITION_BALANCED.md`
- runner:
  `code/mc006_parametric_fact_override_v6_condition_balanced.py`
- source behavior artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v4_source_selected_hybrid_20260630T223734.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v6_condition_balanced_20260630T225327.json`
- result SHA256:
  `3ee27a679a07b7759e7cfc8b7edcdbfb0b3113c77045bdd86abb88e51f9a7653`

## Verdict

MC006 V6 does not pass the condition-balanced behavior gate. It fixes the V5
prompt-family confound structurally, but the behavior is not reliable enough to
use as a hidden-state signature table.

The diagnostic class is:

```text
real_query_failed
```

## Structural Checks

All structural checks passed:

- 22 sources;
- 5 original holdout sources;
- 44 rows;
- two conditions per source;
- 22 rows per condition;
- valid source splits;
- no duplicate row IDs or candidate answers;
- valid candidate token counts;
- query-tag/expected-label balance: `A_true_answer=11`,
  `A_override_answer=11`, `B_true_answer=11`, `B_override_answer=11`.

## Behavior Results

| Criterion | Result |
| --- | --- |
| `balanced_real` true answer at least 20/22 | fail: 17/22 |
| `balanced_fictional` override answer at least 20/22 | pass: 20/22 |
| clean source-level contrasts at least 18/22 | fail: 15/22 |
| holdout clean contrasts at least 4/5 | pass: 4/5 |
| query-tag/expected-label balance valid | pass |

The failure is asymmetric. The model can usually follow the fictional source
tag inside the balanced prompt family, but it too often keeps selecting the
fictional city even when the queried source tag points to the real-world
capital.

## Interpretation

V6 closes one V5 confound in the right way: both target labels occur inside the
same prompt family and both `A`/`B` tags occur with both labels. The negative
result is therefore meaningful rather than a structural invalidation.

MC006 should remain bounded as:

- V4: narrow prompt-bounded behavior substrate passed;
- V5: hidden signature present but output/prompt/null-confounded;
- V6: condition-balanced prompt-family repair failed behavior reliability.

The next MC006 attempt should not probe hidden states until a same-prompt-family
behavior table passes. A repair should first recover real-world retrieval under
the balanced two-source prompt without weakening the fictional-code condition.
