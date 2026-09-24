# MC006 Predecision Frontier Closeout Status

Status: monitor-only route closed by executable audit.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PREDECISION_FRONTIER_CLOSEOUT.md`
- runner:
  `code/mc006_predecision_frontier_closeout_audit.py`
- result:
  `results/cards/MC006/mc006_predecision_frontier_closeout_audit_20260701T200738.json`
- result SHA256:
  `acddfd973f58a313a2051c78f01d4b02c6f502464df54b96a9b0fb77eabe2c52`

## Verdict

The closeout audit returns:

```text
diagnostic_note
```

with route status:

```text
monitor_only_closed
```

Hidden-state work is not allowed on the current MC006 delayed-city/predecision
route. Intervention is not allowed. Ordinary route repairs are not allowed.
Only known-confounded causal stress tests or materially new behavior families
remain valid next moves.

## Evidence

The audit validates the V14-V28 chain:

| Version | Diagnostic | Closeout Role |
| --- | --- | --- |
| V14 | `parser_normalized_generated_substrate_passed` | behavior substrate exists |
| V15 | `candidate_score_confounded` | final-position hidden separator is output/candidate-visible |
| V16 | `leadtime_signal_supported_but_output_global_confounded` | predecision monitor exists but fails global baselines |
| V17 | `intervention_failed` | additive steering does not control labels |
| V18 | `global_margin_separation_blocks_matching` | residualized hidden signal does not survive global margins |
| V19 | `non_holdout_candidate_margin_overlap_failed` | expanded prompt bank still lacks overlap |
| V20 | `strict_final_margin_overlap_absent` | strict final-margin rescue is unavailable |
| V21 | `approximate_pair_matching_failed_margin_baselines` | approximate pairs remain margin-explained |
| V22 | `source_path_final_margin_shadow` | source/path monitors remain final-margin-shadowed |
| V23 | `greedy_final_margin_sign_barrier` | greedy first-token interface is sign-barriered |
| V24 | `delayed_city_interface_decouples_first_token_margin` | first-token barrier breaks, but route is not repaired |
| V25 | `candidate_decoupled_hidden_shuffle_overfit` | hidden selection fails shuffled-label selected-search nulls |
| V26 | `locked_coordinate_transfer_failed` | locked coordinate does not transfer |
| V27 | `expanded_candidate_decoupled_bank_insufficient` | source side repairs, transfer side remains insufficient |
| V28 | `transfer_role_repair_bank_insufficient` | one transfer template is not enough for a behavior bank |

Key decision criteria:

- behavior substrate passed: `true`;
- predecision monitor supported: `true`;
- final/candidate geometry blocks promotion: `true`;
- candidate-decoupled hidden selection failed shuffle null: `true`;
- causal and transfer routes closed: `true`;
- promotion gate passed: `false`;
- monitor-only closeout gate passed: `true`.

## Interpretation

MC006 does not fail because hidden signals are absent. It fails because the
signals that exist are not actionable control surfaces under the current route.

The useful genome-map claim is:

> in this capital-fact override family, predecision internal monitors can appear
> before the final answer interface, but the decision remains downstream-visible
> through candidate/output geometry or too brittle under shuffle, steering, and
> transfer tests.

That is a decision-timing result, not a mechanism card.

## Allowed Claims

- MC006 has a matched generated behavior substrate under V14.
- MC006 has predecision monitor signals under V16 and source/path monitor
  structure under V22.
- The current delayed-city/predecision route is closed as monitor-only.
- The route exports a reusable diagnostic chain for future knowledge-like
  behavior families.

## Forbidden Claims

- MC006 has a knowledge vector.
- MC006 has a promoted or bounded mechanism card.
- V16, V25, or V26 authorizes steering.
- The delayed-city interface solves output/candidate visibility.
- One transfer-ready template is enough for hidden-state probing.
- More ordinary delayed-city prompt repair is a valid promotion route.

## Next Decision

Do not run another MC006 hidden-state search inside the V14-V28 route.

Future MC006 work must be either:

- a known-confounded causal stress test explicitly labeled as such; or
- a materially new behavior family/prompt contract that reopens the behavior
  substrate from first principles.
