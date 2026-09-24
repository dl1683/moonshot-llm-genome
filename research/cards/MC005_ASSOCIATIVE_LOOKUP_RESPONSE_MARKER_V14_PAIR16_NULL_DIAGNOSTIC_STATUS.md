# MC005 Associative Lookup Response-Marker V14 Pair16 Null Diagnostic Status

Status: expanded answer-absent null diagnostic passed; V13 pair16 weak null was sample-fragile.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V14_PAIR16_NULL_DIAGNOSTIC.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v14_pair16_null_diagnostic.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v14_pair16_null_20260630T185711.json`
- result SHA256:
  `2349a11b2e4878886e5cc5a13c09eea983b1befad4aa48ff2c81c600ed78c8aa`

## Verdict

V14 passes the preregistered pair16 answer-absent null diagnostic on
Qwen3-1.7B. The V13 seed-23 pair16 final-label weak null did not reproduce
under the larger row bank.

Summary:

```text
model: Qwen/Qwen3-1.7B
selected band: late_20_26
selected layers: 20, 21, 22, 23, 24, 25, 26
marker: Response
seeds: 17, 23, 31, 37, 41
pair counts: 12, 14, 16
rows per pair/seed: 128
scenario count: 15
label counts:
  clean_null: 15
passed: true
diagnostic class: all_clean_sample_fragile
```

All pair-count/seed scenarios were `clean_null`. Pair counts 12, 14, and 16
each had 5/5 clean seeds. No source value, non-source control value, earlier
neutral colon, final label, or final colon arm was non-clean.

The direct V13 failure check is clean: seed 23 at pair count 16 had final-label
mean delta -0.3115 with zero target-win changes, versus V13's seed-23 pair16
final-label mean delta -0.5039.

## Scenario Table

| Pair count | Seed | Label | Clean rows | Final-label mean delta | Final-label target-win change |
| ---: | ---: | --- | ---: | ---: | ---: |
| 12 | 17 | clean_null | 128/128 | -0.2212 | 0 |
| 12 | 23 | clean_null | 128/128 | -0.1763 | 0 |
| 12 | 31 | clean_null | 128/128 | -0.2217 | 0 |
| 12 | 37 | clean_null | 128/128 | -0.2649 | 1 |
| 12 | 41 | clean_null | 128/128 | -0.3042 | 1 |
| 14 | 17 | clean_null | 128/128 | -0.4019 | 1 |
| 14 | 23 | clean_null | 128/128 | -0.3838 | 1 |
| 14 | 31 | clean_null | 128/128 | -0.2676 | 0 |
| 14 | 37 | clean_null | 127/128 | -0.2612 | 0 |
| 14 | 41 | clean_null | 128/128 | -0.1997 | 0 |
| 16 | 17 | clean_null | 128/128 | -0.2671 | 0 |
| 16 | 23 | clean_null | 128/128 | -0.3115 | 0 |
| 16 | 31 | clean_null | 128/128 | -0.2739 | 0 |
| 16 | 37 | clean_null | 128/128 | -0.3687 | 0 |
| 16 | 41 | clean_null | 128/128 | -0.3555 | 0 |

## Interpretation

What V14 supports:

- the Qwen3-1.7B `Response:` answer-absent null is clean through pair count 16
  under five seeds and 128 rows per pair/seed;
- the V13 pair16 final-label breach should be treated as a sample-fragile
  32-row boundary, not a persistent pair16 answer-absent failure;
- Qwen3-1.7B now has compact lookup reliability, off-target reliability, and
  expanded answer-absent reliability through pair count 16 for this synthetic
  associative lookup contract.

What V14 does not support:

- it does not repair the Qwen3-0.6B answer-absent reliability failure from
  V10-V12;
- it does not prove model-family generality;
- it does not localize the effect to a single head, layer, or minimal path;
- it does not prove reliability outside this prompt family and row-generation
  contract.

## Next Step

The next MC005 pass should move from atlas reliability to finer localization on
the Qwen3-1.7B positive surface:

- decompose heads or smaller layer slices inside `late_20_26`;
- preserve target/distractor/random source controls;
- carry forward the V14 pair16 answer-absent null as a required holdout;
- keep V10-V12 as negative cross-size reliability evidence rather than
  rerunning marker selection on Qwen3-0.6B.
