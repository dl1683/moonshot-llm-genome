# MC005 Associative Lookup Same-Grammar V6 Status

Status: source-line controls clean; query-label robustness still open.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_SAME_GRAMMAR_V6.md`
- runner:
  `code/mc005_associative_lookup_same_grammar_v6.py`
- primary result:
  `results/cards/MC005/mc005_qwen3_1p7b_associative_lookup_same_grammar_v6_20260630T174151.json`
- primary result SHA256:
  `c6bec7cd693882a14bbbaff34c81271b1a1635d3f3e5008a355ec45720d63644`
- seed-17 robustness result:
  `results/cards/MC005/mc005_qwen3_1p7b_associative_lookup_same_grammar_v6_seed17_20260630T174222.json`
- seed-17 result SHA256:
  `f7449132200834cdd071b01816294205186311cb0df35a21738e2b8d53a29f91`

## Verdict

V6 moves the MC005 boundary but does not promote MC005 to a full mechanism card.

The preregistered primary run passed:

```text
scenario labels: 3 clean_null
arm labels: 12 clean
same-grammar null repaired: true
```

The seed-17 robustness diagnostic did not pass:

```text
scenario labels: 2 clean_null, 1 structural_boundary
arm labels: 10 clean, 1 weak, 1 side_effect
same-grammar null repaired: false
```

The important positive result is that the same-grammar source-line masks were
clean in both runs. Irrelevant value, key, colon, and non-source control-value
masks did not reproduce the V5 weak-null effect.

The remaining boundary is query-label/position robustness. In the seed-17
answer-absent control, masking the final query label had small mean delta
(+0.059) but changed three target-win rows, so it was a `side_effect` under the
preregistered bound.

## Primary Run Scenario Table

| Scenario | Label | Clean rows | Irrelevant value | Irrelevant key | Irrelevant colon | Non-source control | Final query label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_other_pair_original` | clean_null | 25/32 | -0.072 | -0.035 | -0.016 | n/a | n/a |
| `query_other_pair_control_word` | clean_null | 32/32 | -0.086 | -0.037 | +0.002 | -0.125 | n/a |
| `answer_absent_reference_control` | clean_null | 26/32 | -0.061 | +0.016 | +0.023 | -0.020 | +0.207 |

## Seed-17 Robustness Table

| Scenario | Label | Clean rows | Irrelevant value | Irrelevant key | Irrelevant colon | Non-source control | Final query label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_other_pair_original` | clean_null | 27/32 | -0.035 | +0.012 | +0.023 | n/a | n/a |
| `query_other_pair_control_word` | clean_null | 32/32 | -0.057 | -0.027 | +0.000 | -0.307 | n/a |
| `answer_absent_reference_control` | structural_boundary | 25/32 | +0.029 | +0.076 | -0.004 | -0.004 | +0.059 |

## Interpretation

What V6 supports:

- the V5 same-grammar weak-null effect was not reproduced by source-value,
  key-token, colon-token, or non-source control-value masks;
- answer words absent from reference values can still produce clean source-line
  nulls;
- the late-band source-line mask now looks less side-effectful than V4/V5
  suggested.

What V6 blocks:

- same-grammar reliability is not fully closed because the seed-17 final query
  label control changed three target-win rows;
- the result has only two seeds and one answer-absent prompt template;
- model-family replication is still premature until query-label/position
  robustness is resolved.

## Next Step

The next MC005 pass should isolate query-label/position sensitivity:

- add final-label variants where the label is semantically irrelevant, repeated,
  omitted, or moved away from the final token;
- separate mean-margin nulls from row-flip nulls;
- run a small seed sweep over the same source-line controls;
- then decide whether same-grammar nulls are stable enough for model-family
  replication.
