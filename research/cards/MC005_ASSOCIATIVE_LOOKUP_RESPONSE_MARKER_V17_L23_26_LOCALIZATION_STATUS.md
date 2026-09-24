# MC005 Associative Lookup Response-Marker V17 L23-26 Localization Status

Status: layers 23-26 localization passed under held-out lookup and null controls.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V17_L23_26_LOCALIZATION.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v17_l23_26_localization.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v17_l23_26_localization_20260630T192250.json`
- result SHA256:
  `be9841bac785a646f87b48508f18383341049826c2f1b6ced6355914194c6e52`

## Verdict

V17 passes the preregistered `slice_l23_26_all` localization gate. It narrows
the supported Qwen3-1.7B intervention surface from layers 20-26 to layers 23-26
under the current synthetic associative lookup contract.

Summary:

```text
model: Qwen/Qwen3-1.7B
marker: Response
pair count: 16
lookup seeds: 43, 47
answer-absent null seeds: 53, 59
lookup rows: 192
lookup baseline target wins: 192/192
primary candidate: slice_l23_26_all
candidate effect share of full: 0.7218
passed: true
diagnostic class: l23_26_localization_supported
```

## Lookup Holdout

| Path | Target mean delta | Target-win loss |
| --- | ---: | ---: |
| `full_l20_26_all` | -8.6820 | 51 |
| `slice_l23_26_all` | -6.2664 | 23 |
| `slice_l20_22_all` | -1.3416 | 0 |
| `slice_l23_24_all` | -0.3818 | 0 |
| `slice_l25_26_all` | -2.3957 | 5 |

`slice_l23_26_all` recovered 72.18 percent of the full layers-20-26 all-head
target-mask mean-delta magnitude. It also beat every smaller-slice control by
at least 0.50 mean delta.

Source controls for `slice_l23_26_all`:

| Arm | Mean delta | Target-win loss |
| --- | ---: | ---: |
| target source value | -6.2664 | 23 |
| distractor source value | +0.9113 | 0 |
| random source value | +0.0127 | 0 |

## Null Holdout

`slice_l23_26_all` preserved the pair16 answer-absent null:

| Seed | Label | Source value | Control value | Earlier colon | Final label | Final colon |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 53 | clean_null | +0.0150 | +0.1006 | +0.0039 | +0.0020 | -0.0231 |
| 59 | clean_null | +0.0072 | +0.0319 | -0.0111 | -0.0078 | -0.0397 |

## Criteria

All preregistered criteria passed:

- full benchmark effect present;
- candidate target effect present;
- candidate source controls passed;
- candidate recovered at least 60 percent of full benchmark effect;
- candidate beat all smaller-slice controls;
- candidate null holdouts were clean.

## Interpretation

What V17 supports:

- MC005's Qwen3-1.7B `Response:` lookup intervention can be narrowed from
  layers 20-26 to layers 23-26;
- the layers-23-26 block preserves target/distractor/random source specificity;
- the layers-23-26 block preserves the pair16 answer-absent null on fresh seeds;
- the V16 `slice_l23_26_all` lead replicated on disjoint lookup seeds.

What V17 does not support:

- it does not localize the mechanism to a single layer or head;
- it does not prove model-family generality;
- it does not repair the Qwen3-0.6B strict answer-absent boundary;
- it does not prove reliability outside the synthetic associative lookup
  contract.

## Next Step

The next MC005 localization pass should decompose layers 23-26:

- compare layers 23-24, 25-26, 23-25, and 24-26 on fresh holdouts;
- test head partitions inside layers 23-26;
- carry forward target/distractor/random source controls and the pair16
  answer-absent null;
- keep V10-V12 as negative cross-size reliability evidence.
