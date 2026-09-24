# MC006 Parametric Fact Override V25 Candidate-Decoupled Template Status

## Status

Diagnostic note. V25 audited the V24 row bank for a stricter MC006 target:
a behavior-passing delayed-city template where full JSON-completion candidate
scoring is not a perfect source-disjoint holdout baseline.

That target exists. The selected `untrusted_note_real` template passed the
behavior and source-disjoint holdout gates, and JSON-completion candidate-score
holdout AUC fell to `0.333`. The hidden screen then found an
`after_mapping_line/layer_10` monitor with `1.000` discovery AUC and `1.000`
holdout AUC that beat the output/candidate controls.

The result still does not pass the signature gate. The same selected-search
procedure on shuffled labels reached `1.000` p95 discovery AUC and `1.000` p95
holdout AUC across 128 split-preserving null runs. V25 therefore changes the
MC006 blocker from output/candidate visibility to shuffled-label selection
overfit on a small table. No intervention is allowed.

## Artifacts

- runner:
  `code/mc006_parametric_fact_override_v25_candidate_decoupled_template.py`
- source result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v24_delayed_city_interface_20260701T040731.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v25_candidate_decoupled_template_20260701T041843.json`
- SHA256:
  `318BE48A7C7349D117BF7B3B612324A158E7DC401902091C2FC7933EAE65ABF2`

## Source Check

V25 validates the V24 source artifact before running hidden-state work:

- source run type:
  `parametric_fact_override_v24_delayed_city_interface`;
- V24 diagnostic:
  `delayed_city_interface_decouples_first_token_margin`;
- V24 behavior ready: true;
- V24 signature ready: false;
- V24 intervention ready: false.

V25 is therefore an offline table-selection and hidden-screen audit over the
already generated V24 row bank, not a new prompt-bank generation run.

## Selected Table

Selected template: `untrusted_note_real`.

Selection key:

```json
[1, 1, 1, 0.6666666666666667, -0.75, 7, -2]
```

Selected rows:

| Field | Value |
| --- | ---: |
| total rows | 40 |
| binary rows | 37 |
| true rows | 24 |
| override rows | 13 |
| lure rows | 2 |
| unparsed rows | 1 |
| non-holdout true rows | 20 |
| non-holdout override rows | 10 |
| holdout true rows | 4 |
| holdout override rows | 3 |

This table is important because it breaks the V24 blocker:
JSON-completion candidate scoring is no longer a perfect holdout baseline.

## Control Baselines

Holdout AUCs on the selected binary table:

| Control | Holdout AUC |
| --- | ---: |
| final next-token city margin | 0.417 |
| city candidate-score margin | 0.250 |
| JSON candidate-score margin | 0.333 |
| best position-local control | 0.500 |

Template-level audits reported:

| Margin | Holdout AUC |
| --- | ---: |
| final next-token city margin | 0.583 |
| city candidate-score margin | 0.750 |
| JSON candidate-score margin | 0.333 |

The exact values differ by audit view because the hidden screen orients binary
scores after train/calibration selection, but both views agree on the central
fact: JSON-completion candidate scoring is not the dominant holdout explanation
on this selected template.

## Hidden Screen

The hidden screen ran on 37 selected binary rows:

- discovery rows: 22;
- calibration rows: 8;
- holdout rows: 7;
- label counts: 24 true, 13 override;
- position mapping: 37/37 complete.

Best selected hidden monitor:

| Field | Value |
| --- | --- |
| position/layer | `after_mapping_line/layer_10` |
| discovery AUC | 1.000 |
| holdout AUC | 1.000 |
| direction norm | 22.6715 |

The hidden monitor beats every direct output/candidate control on holdout. This
is the first MC006 delayed-city audit where the selected hidden monitor is not
blocked by the full-completion candidate-score baseline.

## Shuffled-Label Null

V25 fails because the selected-search procedure is too flexible for this small
table:

| Null Field | Value |
| --- | ---: |
| runs | 128 |
| seed | 25006 |
| discovery AUC p95 | 1.000 |
| holdout AUC p95 | 1.000 |
| max discovery AUC | 1.000 |
| max holdout AUC | 1.000 |

This is not an output/candidate confound. It is a selection-null confound.

## Interpretation

V25 is a better negative than V24. V24 showed that delayed JSON city generation
breaks the first-token city-margin sign barrier but remains
JSON-candidate-score visible on the selected template. V25 shows that the V24
row bank also contains a candidate-score-decoupled delayed-city template.

That moves the project one level deeper. MC006 is not simply "always explained
by output/candidate geometry." But the current hidden-search method is not yet
reliable enough to promote a signature on the candidate-decoupled table. The
selected hidden monitor may be real, but the current evidence cannot
distinguish it from small-table layer/position selection overfit.

## Allowed Claims

- The V24 delayed-city row bank contains at least one behavior-passing,
  source-disjoint, candidate-score-decoupled MC006 template.
- On `untrusted_note_real`, JSON-completion candidate-score holdout AUC is
  `0.333`, so the V24 full-completion candidate-score objection is not universal
  across the delayed-city prompt bank.
- A hidden `after_mapping_line/layer_10` monitor reaches `1.000` discovery and
  holdout AUC on that selected table and beats direct output/candidate controls.
- The hidden result is not mechanism-grade because shuffled-label selected
  searches also reach `1.000` p95 discovery and holdout AUC.
- V25 exports the diagnostic class
  `CANDIDATE_DECOUPLED_SHUFFLE_OVERFIT`.

## Forbidden Claims

- V25 is a mechanism card.
- V25 supports any MC006 intervention.
- `after_mapping_line/layer_10` is a reliable knowledge-control surface.
- Candidate-score decoupling alone is sufficient for MC006 signature promotion.
- The selected layer/position is robust before a preregistered or regularized
  hidden-selection procedure beats shuffled-label nulls.

## Next Decision

Do not steer from V25. The next MC006 route must either:

1. preregister or regularize hidden selection on the candidate-decoupled
   delayed-city target, then rerun the shuffled-label null; or
2. broaden the candidate-decoupled table so source-disjoint holdout size is large
   enough that selected-search shuffled-label p95 cannot trivially reach perfect
   holdout AUC.

