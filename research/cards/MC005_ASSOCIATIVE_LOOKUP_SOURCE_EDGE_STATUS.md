# MC005 Associative Lookup Source-Edge Status

Status: broad source-value path control found; compact mechanism not promoted.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_SOURCE_EDGE.md`
- runner:
  `code/mc005_associative_lookup_source_edge.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_associative_lookup_source_edge_20260630T170033.json`
- result SHA256:
  `1ad09d5e1b093c7c44ab18857862ee8afb46603f69e331ff93b91b9127b32fe9`

## Verdict

MC005 is the strongest positive path result in the rebuild so far, but it is not
a supported compact mechanism card.

What passed:

- behavior gate;
- source-value attention signature gate;
- full-path target source-value intervention diagnostic.

What failed:

- selected-layer locality. The wrong-layer target-value mask was stronger than
  the discovery-selected causal layer on holdout.

The right label is:

> broad source-value path control, not localized layer/head mechanism.

## Behavior

Task:

```text
Reference pairs:
- key_1: value_1
- key_2: value_2
...
- query_key:
```

Behavior metric:

```text
target value first-token logit - distractor value first-token logit
```

Result:

| Metric | Value |
| --- | ---: |
| rows | 48 |
| clean target-retrieval rows | 41 |
| clean rate | 0.854 |
| discovery clean rows | 28 |
| holdout clean rows | 13 |

The behavior substrate passed the preregistered floor.

## Signature

Signature:

At the final query token, score attention mass to each candidate source-value
token. Target source-value candidates are positives; distractor source-value
candidates are negatives.

Selected signature:

| Field | Value |
| --- | ---: |
| layer | 16 |
| head | 14 |
| discovery AUC | 1.000 |
| holdout AUC | 1.000 |
| shuffled-selection p95 | 0.911 |

Holdout subgroup AUCs were all 1.000 across:

- target source before distractor source;
- target source after distractor source;
- query pair in the low half of the prompt;
- query pair in the high half of the prompt.

This is a clean diagnostic signature for this prompt family.

## Intervention

Discovery selected causal layer 25 by target source-value mask effect.

Discovery selected-layer result:

| Arm | Mean margin delta | Target-win loss |
| --- | ---: | ---: |
| layer 25 target source-value mask | -0.884 | 3/28 |

Holdout intervention results:

| Arm | Mean margin delta | Target-win loss |
| --- | ---: | ---: |
| selected layer target source-value | -0.606 | 2/13 |
| selected layer distractor source-value | +0.428 | 0/13 |
| selected layer random value | -0.058 | 0/13 |
| wrong layer target source-value | -0.913 | 3/13 |
| selected attention head target source-value | -0.168 | 1/13 |
| full-path target source-value | -5.154 | 11/13 |
| full-path distractor source-value | +1.952 | 0/13 |

The intervention result is real but broad:

- masking the target value across the full attention path nearly destroys the
  lookup;
- masking the distractor value moves in the opposite direction;
- masking an unrelated value is weak;
- the selected layer is not localized because a wrong-layer target-value mask
  is stronger on holdout;
- the selected head is directionally correct but small.

## Criteria

| Criterion | Result |
| --- | --- |
| at least 36 clean behavior rows | pass |
| at least 12 clean holdout rows | pass |
| clean rate at least 0.75 | pass |
| selected attention holdout AUC at least 0.85 | pass |
| selected attention beats shuffled-selection p95 by 0.05 | pass |
| source-order subgroup AUCs at least 0.75 | pass |
| selected-layer target mask reduces margin by 0.25 | pass |
| selected-layer target mask beats distractor mask by 0.25 | pass |
| selected-layer target mask beats random mask by 0.25 | pass |
| selected-layer target mask beats wrong-layer mask by 0.25 | fail |
| selected-layer target mask causes target-win loss | pass |
| full-path target mask beats full-path distractor mask by 0.50 | pass |

Overall result:

```text
behavior gate: pass
signature gate: pass
selected-layer intervention gate: fail
full-path diagnostic intervention: pass
mechanism-card promotion: fail
```

## Interpretation

MC005 answers a narrower version of the control-surface question:

> Yes, the model's in-context associative lookup depends strongly on the source
> value token. Removing that internal source path changes the behavior in the
> predicted direction under target/distractor/random controls.

It does not yet answer the compact mechanism question:

> No, this run did not localize the source-value control surface to a specific
> layer/head. The effect is a broad late-path dependency.

This is still a material improvement over the MC001-MC004 failures. The project
now has a behavior with a clean internal attention signature and a causal
source-path intervention. The next question is whether that path can be narrowed
to a layer band, head group, or sparse/path feature without losing the effect or
matching wrong-layer controls.

## Next Step

Run an MC005 layer-band/head-group localization audit before any mechanism-card
claim:

- test cumulative late-layer bands around layers 20-26;
- test selected attention head groups rather than one head or all heads;
- keep target, distractor, random-value, and wrong-band controls;
- enlarge the row bank enough that shuffled-selection max AUC is less often
  perfect by chance;
- add a prompt-format holdout with the same key/value semantics but a different
  surface layout.
