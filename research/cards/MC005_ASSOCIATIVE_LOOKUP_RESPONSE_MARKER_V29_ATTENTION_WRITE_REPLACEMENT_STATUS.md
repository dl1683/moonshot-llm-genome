# MC005 Associative Lookup Response-Marker V29 Attention Write Replacement Status

Status: lookup attention-write mediation passed exactly; strict null gate failed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V29_ATTENTION_WRITE_REPLACEMENT.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v29_attention_write_replacement.py`
- source artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
- source SHA256:
  `efdab6fad30de1762b8e92f7e46c7f4e1e160bcc988ad093c7d0bf9a5617b9cf`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v29_attention_write_replacement_20260630T214645.json`
- result SHA256:
  `c90bbd9a65fdd3866ef4e19e787d7c93b7faa693a89ddd0f843e58829097804b`

## Verdict

V29 does not pass the full preregistered reliability gate, because one
answer-absent null arm failed. But it is the strongest mechanistic localization
inside MC005 so far.

The layers-24-26 final-query attention-write replacement exactly recovered the
direct layers-24-26 target source-mask effect on V25 holdout rows:

- direct target source mask: mean delta -6.1978, 11 target-win losses;
- target attention-write replacement: mean delta -6.1978, 11 target-win losses;
- recovery: 1.0000 for both mean-delta magnitude and target-win loss.

The lookup source controls also matched the direct source-mask controls:
distractor write replacement moved the margin upward by +0.8223 with zero
target-win losses, and random write replacement was near zero at +0.0239 with
zero target-win losses.

The diagnostic class is:

```text
null_failed
```

So the correct claim is bounded:

> On the tested lookup holdout, the layers-24-26 source-mask effect is mediated
> by final-query self-attention output writes. It is not yet a fully reliable
> mechanism surface because the strict answer-absent write-replacement null
> suite did not pass.

## Lookup Result

| Source key | Direct mask delta | Direct target-win loss | Write replacement delta | Write target-win loss | Write label |
| --- | ---: | ---: | ---: | ---: | --- |
| `target_value` | -6.1978 | 11 | -6.1978 | 11 | `side_effect` |
| `distractor_value` | +0.8223 | 0 | +0.8223 | 0 | `weak` |
| `random_value` | +0.0239 | 0 | +0.0239 | 0 | `clean` |

## Layer Decomposition

Single-layer final-query write replacement did not recover the parent effect.

| Layer set | Delta | Target-win loss | Label |
| --- | ---: | ---: | --- |
| layers 24-26 | -6.1978 | 11 | `side_effect` |
| layer 24 | -0.4258 | 0 | `clean` |
| layer 25 | -1.2212 | 0 | `side_effect` |
| layer 26 | -0.7778 | 0 | `weak` |

This keeps the V20/V21 interpretation intact: the useful surface is a
three-layer interaction block, not a single-layer write.

## Nulls

| Seed | Null label | Arm | Target-win loss | Mean delta | Clean |
| --- | --- | --- | ---: | ---: | --- |
| 251 | `side_effect` | `source_value` | 0 | -0.0085 | yes |
| 251 | `side_effect` | `non_source_control_value` | -1 | +0.1406 | no |
| 251 | `side_effect` | `final_colon` | 0 | +0.0586 | yes |
| 257 | `clean_null` | `source_value` | 0 | +0.0267 | yes |
| 257 | `clean_null` | `non_source_control_value` | 0 | +0.1302 | yes |
| 257 | `clean_null` | `final_colon` | 0 | +0.0553 | yes |

The failing arm is small and margin-safe by magnitude, but the preregistered
null rule required target-win loss exactly 0. A gain of one target-win row
(`target_win_loss = -1`) still violates the exact row-change criterion.

## Criteria

| Criterion | Result |
| --- | --- |
| source artifact valid | pass |
| direct parent reference valid | pass |
| direct source controls pass | pass |
| target write delta recovery at least 0.70 | pass |
| target write target-win-loss recovery at least 0.50 | pass |
| write source controls pass | pass |
| write nulls clean | fail |

## Interpretation

What V29 supports:

- the direct layers-24-26 source-mask effect is exactly reproducible by
  replacing only final-query self-attention output writes in those layers;
- the lookup source specificity survives the write-replacement formulation;
- the parent effect still requires the three-layer write block, not a single
  layer.

What V29 does not support:

- no full mechanism-card promotion yet;
- no claim that attention-write replacement has fully clean answer-absent
  reliability;
- no claim of arbitrary-context or cross-model generality.

## Next Step

The next MC005 pass should not return to V26 row-signature steering. It should
repair or explain the V29 null boundary: repeat the answer-absent
write-replacement null with more seeds and row-level audit of the single
`non_source_control_value` target-win gain, then decide whether the failure is
sample-fragile, a harmless target-gain convention issue, or a real final-query
write side effect.

Follow-up completed:

- V30 repeated this null boundary and found fresh strict row flips on seeds 263
  and 283. The V29 failure is therefore not only an original-sample artifact.
  See
  `research/cards/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V30_WRITE_NULL_SWEEP_STATUS.md`.
