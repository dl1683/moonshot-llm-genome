# MC006 Parametric Fact Override Delayed-City Closeout Status

## Status

Diagnostic closeout. The MC006 delayed-city branch is closed as a monitor-only
route under the tested prompt families.

This closeout does not say MC006 is useless. It says the current delayed-city
route should no longer be treated as a candidate for mechanism promotion,
coordinate selection, transfer testing, or steering. The branch has produced a
useful map of where the behavior lives, but not a reliable internal control
surface.

Diagnostic class:

`delayed_city_route_closed_monitor_only`

Canonical atlas type:

`DELAYED_CITY_ROUTE_CLOSED_MONITOR_ONLY`

## Evidence Base

The closeout is based on the MC006 V14-V28 chain:

| Version | Result | Boundary |
| --- | --- | --- |
| V14 | Matched generated-answer behavior table passed after parser normalization. | Behavior substrate exists, but no mechanism claim yet. |
| V15 | Final-token hidden separator reached perfect holdout AUC. | Candidate-score and next-token output margins also reached perfect holdout AUC. |
| V16 | Earlier `after_mapping_line/layer_4` monitor beat same-position output controls. | Final candidate-score and final-output margins still reached perfect holdout AUC. |
| V17 | Additive steering on the V16 direction was tested. | No directional holdout label control; controls matched or exceeded movement. |
| V18 | V16 signal reproduced on the V14 table. | No true/override holdout overlap in candidate-score or final-output margins; residualized hidden AUC fell to 0.333. |
| V19 | A broader prompt bank improved balance and selected `separate_task_weak`. | Strict candidate/final margin overlap still failed. |
| V20 | Pooled strict-overlap rescue was audited. | Strict final-output overlap was absent even under pooled selection. |
| V21 | Approximate pair matching at z <= 0.5 was tested. | Candidate-score and final-output margins still perfectly ordered holdout pairs. |
| V22 | Source/path and line-boundary lead-time curve was mapped. | Monitors existed, but final candidate/output margins still perfectly ordered holdout pairs. |
| V23 | Final-margin sign barrier was diagnosed. | Final next-token margin sign predicted every binary generated label under the greedy interface. |
| V24 | Delayed JSON city answer interface broke the first-token sign barrier. | JSON-completion candidate-score holdout AUC still reached 1.000. |
| V25 | Candidate-score-decoupled delayed-city template was found. | Hidden selector failed split-preserving shuffled-label selected-search nulls. |
| V26 | V25 coordinate was locked and transferred. | Source holdout AUC was 0.750, but transfer holdout AUC was 0.500 and controls reached 0.833. |
| V27 | 640-row source/transfer expanded bank was built. | Source side repaired, but only one transfer-role template passed. |
| V28 | 640-row transfer-only repair bank was built. | One transfer template passed, but the bank gate required at least two; combined ready rows reached only 143 pooled binary rows. |

## What Survives

MC006 still supports several useful monitor-only claims:

- a matched generated-answer capital-fact override behavior substrate exists
  under V14;
- pre-output and source/path monitors exist under V16 and V22;
- final-output and candidate-score geometry often explain or shadow the label;
- the greedy answer interface has a final-margin sign barrier;
- delayed JSON answers break the first-token barrier but not completion-level
  candidate visibility in general;
- candidate-score-decoupled transfer templates exist, but they are too sparse
  and brittle to support a behavior bank;
- transfer-role prompt construction lives in a narrow band between
  candidate-visible real-geography prompts and parse/label-fragile task-local
  prompts.

These are genome-map facts. They are not mechanism-control claims.

## Closed Routes

The following MC006 delayed-city routes are now closed as promotion routes:

- final-prompt-token hidden separators;
- same-table pre-output monitor promotion;
- simple additive residual steering from the V16 direction;
- strict margin-overlap rescue on the V14/V19 banks;
- approximate pair matching as a substitute for strict overlap;
- source/path final-margin rescue;
- greedy first-token generated answer tables;
- delayed JSON city interface as a full repair;
- flexible hidden selection on the small candidate-decoupled delayed-city table;
- locked-coordinate transfer from `after_mapping_line/layer_10`;
- source-side-only expanded bank construction;
- ordinary transfer-role prompt repair under the current delayed-city family.

## Allowed Claims

- MC006 has knowledge-like generated behavior under a matched prompt contract.
- MC006 has monitor-only pre-output and source/path signals.
- MC006 has a well-mapped sequence of output/candidate/prompt confounds.
- The delayed-city branch exports a reusable diagnostic chain:
  `GLOBAL_OUTPUT_CONFOUNDED_LEADTIME`,
  `GLOBAL_MARGIN_SEPARATION_BLOCKS_MATCHING`,
  `MARGIN_OVERLAP_TABLE_FAILED`,
  `STRICT_FINAL_MARGIN_OVERLAP_ABSENT`,
  `APPROXIMATE_PAIR_MATCHING_FAILED_MARGIN_BASELINES`,
  `SOURCE_PATH_FINAL_MARGIN_SHADOW`,
  `GREEDY_FINAL_MARGIN_SIGN_BARRIER`,
  `DELAYED_INTERFACE_CANDIDATE_SCORE_VISIBLE`,
  `CANDIDATE_DECOUPLED_SHUFFLE_OVERFIT`,
  `LOCKED_COORDINATE_TRANSFER_FAILED`,
  `EXPANDED_CANDIDATE_DECOUPLED_BANK_INSUFFICIENT`,
  `TRANSFER_ROLE_REPAIR_BANK_INSUFFICIENT`, and
  `DELAYED_CITY_ROUTE_CLOSED_MONITOR_ONLY`.
- The MC006 delayed-city branch should be used as a diagnostic atlas row and
  negative control suite for future factual-override work.

## Forbidden Claims

- MC006 has a promoted knowledge mechanism.
- MC006 has a reliable hidden knowledge-control surface.
- V16, V25, V26, V27, or V28 authorizes steering.
- The V25 `after_mapping_line/layer_10` coordinate is reusable.
- The delayed JSON city interface removes output/candidate visibility.
- One transfer-ready template is enough to run hidden-state search.
- Ordinary transfer-template prompting is still a live promotion route under
  the current delayed-city family.

## Future Work Rule

Future MC006 work must satisfy one of two conditions:

1. It is explicitly labeled a known-confounded causal stress test, with no
   mechanism-promotion language; or
2. it changes the behavior family or prompt contract materially enough that
   V14-V28 are no longer the active route being repaired.

Otherwise, the work should move to another family: MC005 bounded closeout,
MC007 bridge redesign, or a new behavior family designed to measure the
prompt/output/source/internal mixture from the start.
