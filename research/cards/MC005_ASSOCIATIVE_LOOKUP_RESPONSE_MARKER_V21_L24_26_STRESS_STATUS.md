# MC005 Associative Lookup Response-Marker V21 L24-26 Stress Status

Status: layers 24-26 stress diagnostic passed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V21_L24_26_STRESS.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v21_l24_26_stress.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v21_l24_26_stress_20260630T200456.json`
- result SHA256:
  `9e684f807d34eeed566f0065909cf525d38f54106d27d08986ed6e676aabede3`

## Verdict

V21 passes the preregistered stress diagnostic for the fixed
`slice_l24_26_all` interaction block. The parent path kept a strong target
source-value effect across the dash/colon, arrow, and sentence layouts, across
a shifted lexicon, and under pair-count 20 stress. Distractor and random source
controls did not match the target effect, and both answer-absent null seeds
stayed clean.

Summary:

```text
model: Qwen/Qwen3-1.7B
marker: Response
parent candidate: slice_l24_26_all
lookup stress scenarios: 6
lookup rows: 384
answer-absent null seeds: 179, 181
null rows: 192
passed: true
diagnostic class: l24_26_stress_supported
```

## Lookup Stress

| Scenario | Layout | Pair count | Lexicon offset | Baseline target wins | Target mean delta | Target-win loss | Label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pair16_dash_base` | dash/colon | 16 | 0 | 63/64 | -5.8174 | 6 | works |
| `pair16_arrow_layout` | arrow | 16 | 0 | 64/64 | -7.4199 | 6 | works |
| `pair16_sentence_layout` | sentence | 16 | 0 | 64/64 | -6.6816 | 9 | works |
| `pair16_dash_lexicon_shift` | dash/colon | 16 | 24 | 64/64 | -5.3594 | 6 | works |
| `pair20_dash_pairstress` | dash/colon | 20 | 0 | 64/64 | -5.9854 | 6 | works |
| `pair20_arrow_lexicon_stress` | arrow | 20 | 24 | 64/64 | -7.3662 | 4 | works |

All lookup scenarios cleared the preregistered baseline, effect-size, and
target-win-loss thresholds.

## Source Controls

| Scenario | Target source delta | Distractor source delta | Random source delta |
| --- | ---: | ---: | ---: |
| `pair16_dash_base` | -5.8174 | +1.0010 | +0.0361 |
| `pair16_arrow_layout` | -7.4199 | +1.2529 | +0.0088 |
| `pair16_sentence_layout` | -6.6816 | +1.2480 | +0.0264 |
| `pair16_dash_lexicon_shift` | -5.3594 | +0.8164 | +0.0068 |
| `pair20_dash_pairstress` | -5.9854 | +0.9014 | +0.0020 |
| `pair20_arrow_lexicon_stress` | -7.3662 | +1.3516 | +0.0049 |

The distractor and random source controls stayed separated from the target
source-value effect by more than the preregistered 0.50 mean-delta margin in
every stress scenario.

## Null Holdout

`slice_l24_26_all` preserved both pair16 answer-absent null seeds:

| Seed | Label | Source value | Control value | Earlier colon | Final label | Final colon |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 179 | clean_null | +0.0033 | +0.1159 | +0.0026 | +0.0046 | +0.0273 |
| 181 | clean_null | -0.0078 | +0.1055 | -0.0085 | +0.0098 | +0.0143 |

## Criteria

All preregistered criteria passed:

- every lookup stress baseline had target wins at least 75 percent of rows;
- every lookup stress scenario had target source-value mean delta at most
  -1.0 and at least three target-win losses;
- target source-value masking beat distractor and random source controls by at
  least 0.50 mean delta in every lookup scenario;
- both answer-absent null holdout seeds were `clean_null`.

## Interpretation

What V21 supports:

- the current MC005 Qwen3-1.7B surface is a stress-supported all-head
  layers-24-26 interaction block under the synthetic associative lookup
  contract;
- the V20 interaction interpretation survives the tested layout, shifted
  lexicon, and pair-count stress axes;
- the target source-value effect remains source-specific under
  target/distractor/random source controls;
- the answer-absent null boundary remains clean on fresh seeds.

What V21 does not support:

- it does not localize the mechanism to a single layer, head partition, or
  head;
- it does not prove model-family generality;
- it does not repair the Qwen3-0.6B strict answer-absent boundary;
- it does not prove arbitrary-context or deployable reliability beyond the
  synthetic associative lookup contract.

## Next Step

The next MC005 pass should use V21 as the current reliability floor for
`slice_l24_26_all` and move to row-level interaction diagnostics or a new
preregistered model-family/null design. Another smaller-path promotion attempt
should first explain why it can beat the V19/V20 negative decomposition
boundary.
