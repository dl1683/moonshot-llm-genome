# Control-Surface Genome Snapshot

Date: 2026-07-01

Status: generated compact genome snapshot implemented and validated.

Machine-readable artifact:

> `data/control_surface_genome_snapshot.json`

Builder:

> `code/control_surface_genome_snapshot.py`

Commands:

```powershell
python code\control_surface_genome_snapshot.py --write
python code\control_surface_genome_snapshot.py
python code\validate_control_surface_atlas.py
```

## Purpose

This is the compact current-state object for the small-model control
surface genome. It fuses the atlas, mixture law, gate geometry, route
disposition, bridge ladder, reliability matrix, transfer matrix, error
taxonomy, and next queue.

The point is not to add another interpretation layer. The point is to
make the current global claim state executable: what is controllable,
what only looks controllable, what is prompt-visible, what is
output-visible, what is internal, what is causal, and where each claim
breaks.

## Current Shape

- atlas rows: 19;
- linked result artifacts: 53;
- verdict counts: `{"bounded_mechanism_card": 1, "diagnostic_note": 17, "failed_mechanism_card": 1}`;
- bounded mechanisms: 1;
- promoted mechanisms: 0;
- diagnostic or failed rows: 18.

## Mixture Shape

| Pressure Class | Count | Ratio |
| --- | ---: | ---: |
| `behavior_or_bridge_substrate_blocked` | 11 | 0.579 |
| `internal_causal_surface` | 1 | 0.053 |
| `internal_monitor_present` | 5 | 0.263 |
| `null_boundary_or_locality_limited` | 6 | 0.316 |
| `output_geometry_visible` | 14 | 0.737 |
| `prompt_contract_visible` | 19 | 1.000 |
| `signature_or_intervention_failed` | 6 | 0.316 |
| `source_or_prompt_token_dependent` | 13 | 0.684 |
| `transfer_unproven_or_failed` | 3 | 0.158 |

## Gate Shape

| Terminal Stage | Count | Ratio |
| --- | ---: | ---: |
| `intervention_failed` | 1 | 0.053 |
| `pre_signature_behavior_substrate` | 11 | 0.579 |
| `pre_signature_prompt_channel_locality` | 1 | 0.053 |
| `reliability_null_boundary` | 1 | 0.053 |
| `signature_monitor_no_lever` | 2 | 0.105 |
| `signature_output_geometry_shadow` | 3 | 0.158 |

## Bridge Shape

- bridge rungs: 24;
- hidden-state-allowed bridge rungs: 0;
- clean unconfounded bridge rungs: 0;
- recent closed rungs: `["MC030", "MC031", "MC032", "MC033"]`.

## Global Allowed Claims

- The current small-model control-surface map is mostly prompt- and output-coupled: prompt-contract-visible pressure appears in 19/19 rows and output-geometry-visible pressure in 14/19 rows.
- The mechanism-card bar has a measured geometry: 12 rows stop before signature work, 5 stop at signature-stage controls, 1 stops at failed intervention, and 1 is bounded at reliability.
- MC005 is the single bounded internal-causal reference specimen; it is not promoted because null, locality, side-effect, and transfer gates are still not clean.
- The bridge program is still behavior-substrate work: 24 bridge rungs, 0 hidden-state-allowed rungs, and no clean unconfounded bridge.

## Global Forbidden Claims

- Do not claim a broad truth, honesty, factuality, or knowledge vector.
- Do not call output-visible, candidate-visible, prompt-visible, or monitor-only signals mechanisms.
- Do not reopen MC030 operation-leak guards, the MC031 statusless checksum route, the MC032 cross-table route, or the MC033 fact-claim route without a materially different branch/null/local/side-number/source-validity gate.
- Do not report full reliability: 0 rows currently clear the full reliability bar.

## Top Next Pressures

| Queue Item | Priority | Reason Codes | Next Test |
| --- | --- | --- | --- |
| `authority_pressure_creates_contrast_before_clean_substrate__next_test_1` | `immediate` | `["TENTATIVE_PATTERN", "BRIDGE_ROUTE_NEEDED", "INTERVENTION_RELEVANT", "OUTPUT_GEOMETRY_CONTROL", "BEHAVIOR_SUBSTRATE_GATE", "ZERO_PROMOTED_MECHANISM_PRESSURE"]` | For the next bridge, preserve MC012-level direct controls and conflict mixture while making the answer rule independent of visible trusted/untrusted source-status text, proving that outputs follow the intended rule, and showing that expected-atomic rows survive prompt-local table pressure. |
| `behavior_substrate_first_or_everything_lies__next_test_2` | `immediate` | `["STRONG_DOCTRINE_GUARDRAIL", "BRIDGE_ROUTE_NEEDED", "INTERVENTION_RELEVANT", "OUTPUT_GEOMETRY_CONTROL", "BEHAVIOR_SUBSTRATE_GATE", "ZERO_PROMOTED_MECHANISM_PRESSURE"]` | For the next bridge, require MC012-level direct controls, conflict balance, nulls, parseability, source-disjoint holdout, candidate/output baselines, and an MC013/MC014/MC015/MC016-style prompt-channel and rule-following audit that does not collapse the learned-fact side. |
| `transfer_fails_at_reliability_before_primary_effect__next_test_1` | `immediate` | `["TENTATIVE_PATTERN", "SINGLE_ROW_LAW_SUPPORT", "TRANSFER_GAP", "BEHAVIOR_SUBSTRATE_GATE"]` | Run any future transfer test with matched null panels and side rows from the start. |
| `transfer_fails_at_reliability_before_primary_effect__next_test_2` | `immediate` | `["TENTATIVE_PATTERN", "SINGLE_ROW_LAW_SUPPORT", "TRANSFER_GAP", "BEHAVIOR_SUBSTRATE_GATE"]` | Do not mark a surface as transferred unless the atlas row's null_locality and transfer fields both move beyond bounded. |
| `coarse_source_ablation_overstates_circuit_locality__next_test_1` | `immediate` | `["TENTATIVE_PATTERN", "SINGLE_ROW_LAW_SUPPORT", "INTERVENTION_RELEVANT", "ZERO_PROMOTED_MECHANISM_PRESSURE"]` | Require source deletion, neutral rewrite, and query-only path comparisons in every source-token control-surface claim. |

## Interpretation

The current genome is not a list of discovered mechanisms. It is a
measured distribution of control-surface failure and survival modes.
Most of the map is prompt-contract-visible, output-visible,
source/prompt-token-dependent, behavior-substrate-blocked, or
monitor-only. The one internal causal specimen is bounded rather
than promoted.

This compact snapshot is the artifact future experiments should move.
A new result matters if it changes one of these ratios, moves a row
to a later gate stage, creates a real transfer-ready mechanism, or
sharpens a failure bucket.

## Validation

The snapshot is validated as part of `python code\validate_control_surface_atlas.py`.
Validation fails if it loses the gate partition, reports promoted
mechanisms, loses MC005 as the sole bounded reference, reopens bridge
hidden-state work, drops MC030-MC033 closure context, or emits global
claims without forbidden-claim boundaries.
