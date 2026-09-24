# MC006 Parametric Fact Override V24 Delayed-City Interface Status

## Status

Diagnostic note. V24 changed the MC006 answer interface after V23 showed that
direct city generation made final next-token city-margin sign tautological with
the parsed binary label.

The delayed interface worked as an interface repair: the generated answer starts
with a JSON wrapper rather than a city token, and the selected table no longer
has a final city-token sign barrier. It did not promote a mechanism signature:
the best hidden monitor was still beaten by a JSON-completion candidate-score
baseline on holdout.

## Artifacts

- runner:
  `code/mc006_parametric_fact_override_v24_delayed_city_interface.py`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v24_delayed_city_interface_20260701T040731.json`
- SHA256:
  `FB0AD33B02CB3A8DDC843BD7A0D164F3A50E1183EF6B162876D862142C847C10`

## Source Check

V24 validates the V19, V22, and V23 source artifacts before running:

- V19 diagnostic: `non_holdout_candidate_margin_overlap_failed`;
- V22 diagnostic: `source_path_final_margin_shadow`;
- V23 diagnostic: `greedy_final_margin_sign_barrier`.

The V24 structural prompt bank passed:

- 40 sources;
- 10 templates;
- 400 rows;
- 8 holdout rows per template;
- one source split per source;
- no duplicate record ids;
- no duplicate candidate answers;
- no true-answer prompt leakage.

## Interface

Instead of asking for the city directly, V24 asks for JSON:

```text
Return exactly JSON with one key named city, like {"city":"ExampleCity"}.
JSON:
```

This preserves generated city behavior while preventing the first generated
token from being the city token.

## Selected Table

Selected template: `game_code_then_geo`.

Selection key:

```json
[1, 1, 1, 1, 1, 1, 8]
```

Selected rows:

| Field | Value |
| --- | ---: |
| total rows | 40 |
| binary rows | 31 |
| true rows | 10 |
| override rows | 21 |
| lure rows | 1 |
| unparsed rows | 8 |
| non-holdout true rows | 8 |
| non-holdout override rows | 17 |
| holdout true rows | 2 |
| holdout override rows | 4 |
| selected-city first-token rate | 0.000 |

All behavior-gate criteria passed, including source-disjoint holdout label
coverage and first-token decoupling.

## Margin Results

V24 repaired the exact V23 failure mode:

| Margin | Row Count | Sign Accuracy | Raw Overlap |
| --- | ---: | ---: | --- |
| final next-token city margin | 31 | 0.516 | true |
| city candidate-score margin | 31 | 0.645 | true |
| JSON candidate-score margin | 31 | 0.742 | true |

This means final city-token sign no longer defines the parsed label. The
delayed wrapper successfully breaks the direct first-token sign barrier.

## Hidden Screen

The hidden screen ran on 31 selected binary rows:

- discovery rows: 18;
- calibration rows: 7;
- holdout rows: 6;
- label counts: 10 true, 21 override;
- position mapping: 31/31 complete.

Best selected hidden monitor:

| Field | Value |
| --- | --- |
| position/layer | `after_mapping_line/layer_0` |
| discovery AUC | 1.000 |
| holdout AUC | 0.875 |

Output/candidate controls:

| Control | Discovery AUC | Holdout AUC |
| --- | ---: | ---: |
| final next-token city margin | 0.631 | 0.250 |
| city candidate-score margin | 0.615 | 0.750 |
| JSON candidate-score margin | 0.754 | 1.000 |

The hidden monitor does not pass the signature gate because the
JSON-completion candidate-score baseline reaches 1.000 holdout AUC.

## Interpretation

V24 proves that V23 was not just "MC006 is impossible." It was specifically a
first-token answer-interface problem. When the city is generated later inside a
JSON value, the final city-token margin is no longer a sign barrier.

But V24 also shows that fixing the first-token barrier is not enough. A
candidate-score baseline over the same JSON answer format perfectly orders the
source-disjoint holdout labels. The behavior has moved from a first-token
tautology to a richer output/candidate geometry problem.

## Allowed Claims

- MC006 can build a delayed-city generated-answer table with no true-answer
  prompt leakage and source-disjoint holdout label coverage.
- The delayed JSON wrapper breaks the V23 first-generated-token city-margin
  sign barrier.
- V24 supports an early hidden monitor on the selected table, but it is
  monitor-only because a JSON-completion candidate-score baseline beats it on
  holdout.
- V24 changes the MC006 diagnostic class from "same-interface first-token
  barrier" to "delayed interface still candidate-score visible."

## Forbidden Claims

- V24 is a mechanism card.
- V24 supports an intervention.
- The V24 hidden monitor is mechanism-grade.
- Delayed JSON answers remove output/candidate visibility from MC006.
- The MC006 answer-interface problem is solved in general.

## Next Decision

Do not steer from V24. The next MC006 route must either:

1. design an answer interface where full-completion candidate scoring is not a
   perfect holdout baseline, or
2. treat the V24 hidden monitor as a known-confounded diagnostic and run only a
   preregistered stress test, not a promotion attempt.
