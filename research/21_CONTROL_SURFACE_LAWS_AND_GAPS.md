# Control-Surface Laws And Gaps

Date: 2026-07-02

Status: evidence-bounded theory layer, not a completed genome.

This document turns the control-surface atlas into falsifiable working
hypotheses. The goal is to make the project better at prediction:

> Given a behavior and prompt contract, predict which internal surfaces will
> appear, which baselines will explain them, which interventions might work,
> and which reliability gates are most likely to fail.

Machine-readable backing files:

- atlas rows: `data/control_surface_atlas.json`
- law hypotheses: `data/control_surface_law_hypotheses.json`
- law audit: `data/control_surface_law_audit.json`
- next experiment queue: `data/control_surface_next_experiment_queue.json`
- smoke diagnostics: `data/control_surface_smoke_diagnostics.json`
- bridge ladder: `data/control_surface_bridge_ladder.json`
- mixture law: `data/control_surface_mixture_law.json`
- decision frontier: `data/control_surface_decision_frontier.json`
- route disposition: `data/control_surface_route_disposition.json`
- transfer matrix: `data/control_surface_transfer_matrix.json`
- reliability matrix: `data/control_surface_reliability_matrix.json`
- error taxonomy: `data/control_surface_error_taxonomy.json`
- gate geometry: `data/control_surface_gate_geometry.json`
- genome snapshot: `data/control_surface_genome_snapshot.json`
- axis interactions: `data/control_surface_axis_interactions.json`
- coverage gaps: `data/control_surface_coverage_gaps.json`
- gap closure plan: `data/control_surface_gap_closure_plan.json`
- offensive doctrine: `data/control_surface_offensive_doctrine.json`
- compositional genome audit: `data/control_surface_compositional_genome_audit.json`
- family matrix: `data/control_surface_family_matrix.json`
- knowledge ladder: `data/control_surface_knowledge_ladder.json`
- knowledge gap plan: `data/control_surface_knowledge_gap_plan.json`
- knowledge substrate admission:
  `data/control_surface_knowledge_substrate_admission.json`
- knowledge candidate queue:
  `data/control_surface_knowledge_candidate_queue.json`
- knowledge first-run pack:
  `data/control_surface_knowledge_first_run_pack.json`
- knowledge second-wave outcomes:
  `data/control_surface_knowledge_second_wave_outcomes.json`
- knowledge third-wave outcomes:
  `data/control_surface_knowledge_third_wave_outcomes.json`
- knowledge fourth-wave outcomes:
  `data/control_surface_knowledge_fourth_wave_outcomes.json`
- knowledge fifth-wave outcomes:
  `data/control_surface_knowledge_fifth_wave_outcomes.json`
- knowledge sixth-wave outcomes:
  `data/control_surface_knowledge_sixth_wave_outcomes.json`
- knowledge seventh-wave outcomes:
  `data/control_surface_knowledge_seventh_wave_outcomes.json`
- knowledge eighth-wave outcomes:
  `data/control_surface_knowledge_eighth_wave_outcomes.json`
- knowledge ninth-wave outcomes:
  `data/control_surface_knowledge_ninth_wave_outcomes.json`
- knowledge tenth-wave outcomes:
  `data/control_surface_knowledge_tenth_wave_outcomes.json`
- knowledge eleventh-wave outcomes:
  `data/control_surface_knowledge_eleventh_wave_outcomes.json`
- transfer width probe: `data/transfer_width_probe_mc005_mc003_mc004.json`
- singleton stage replication pack: `data/singleton_stage_replication_pack.json`
- artifact index: `data/control_surface_artifact_index.json`
- cross-family comparison: `data/control_surface_comparison.json`
- decision-frontier builder: `code/control_surface_decision_frontier.py`
- bridge-ladder builder: `code/control_surface_bridge_ladder.py`
- comparison builder: `code/control_surface_comparison.py`
- law-audit builder: `code/control_surface_law_audit.py`
- mixture-law builder: `code/control_surface_mixture_law.py`
- next-queue builder: `code/control_surface_next_queue.py`
- reliability-matrix builder: `code/control_surface_reliability_matrix.py`
- error-taxonomy builder: `code/control_surface_error_taxonomy.py`
- gate-geometry builder: `code/control_surface_gate_geometry.py`
- genome-snapshot builder: `code/control_surface_genome_snapshot.py`
- axis-interactions builder: `code/control_surface_axis_interactions.py`
- coverage-gaps builder: `code/control_surface_coverage_gaps.py`
- gap-closure-plan builder: `code/control_surface_gap_closure_plan.py`
- offensive-doctrine builder: `code/control_surface_offensive_doctrine.py`
- compositional-genome audit builder:
  `code/control_surface_compositional_genome_audit.py`
- family-matrix builder: `code/control_surface_family_matrix.py`
- knowledge-ladder builder: `code/control_surface_knowledge_ladder.py`
- knowledge-gap-plan builder: `code/control_surface_knowledge_gap_plan.py`
- knowledge-substrate-admission builder:
  `code/control_surface_knowledge_substrate_admission.py`
- knowledge-candidate-queue builder:
  `code/control_surface_knowledge_candidate_queue.py`
- knowledge-first-run-pack builder:
  `code/control_surface_knowledge_first_run_pack.py`
- knowledge-second-wave-outcomes builder:
  `code/control_surface_knowledge_second_wave_outcomes.py`
- knowledge-third-wave-outcomes builder:
  `code/control_surface_knowledge_third_wave_outcomes.py`
- knowledge-fourth-wave-outcomes builder:
  `code/control_surface_knowledge_fourth_wave_outcomes.py`
- knowledge-fifth-wave-outcomes builder:
  `code/control_surface_knowledge_fifth_wave_outcomes.py`
- knowledge-sixth-wave-outcomes builder:
  `code/control_surface_knowledge_sixth_wave_outcomes.py`
- knowledge-seventh-wave-outcomes builder:
  `code/control_surface_knowledge_seventh_wave_outcomes.py`
- knowledge-eighth-wave-outcomes builder:
  `code/control_surface_knowledge_eighth_wave_outcomes.py`
- knowledge-ninth-wave-outcomes builder:
  `code/control_surface_knowledge_ninth_wave_outcomes.py`
- knowledge-tenth-wave-outcomes builder:
  `code/control_surface_knowledge_tenth_wave_outcomes.py`
- knowledge-eleventh-wave-outcomes builder:
  `code/control_surface_knowledge_eleventh_wave_outcomes.py`
- transfer-width probe builder:
  `code/transfer_width_probe_mc005_mc003_mc004.py`
- singleton-stage replication builder:
  `code/singleton_stage_replication_pack.py`
- route-disposition builder: `code/control_surface_route_disposition.py`
- smoke-diagnostic builder: `code/control_surface_smoke_diagnostics.py`
- transfer-matrix builder: `code/control_surface_transfer_matrix.py`
- validator: `python code/validate_control_surface_atlas.py`
- MC005 bounded mechanism closeout:
  `research/cards/MC005_ASSOCIATIVE_LOOKUP_BOUNDED_MECHANISM_CLOSEOUT_STATUS.md`
- MC005 write-replacement closeout audit:
  `research/cards/MC005_WRITE_REPLACEMENT_CLOSEOUT_STATUS.md`
- symbolic bridge preregistration:
  `research/prereg/MC008_SYMBOLIC_FACT_CODE_ARBITRATION.md`
- symbolic bridge V1 behavior status:
  `research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_BEHAVIOR_STATUS.md`
- symbolic bridge V2 null/authority repair status:
  `research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_V2_NULL_AUTHORITY_REPAIR_STATUS.md`
- symbolic bridge route closeout:
  `research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_ROUTE_CLOSEOUT_STATUS.md`
- derived-code bridge preregistration:
  `research/prereg/MC009_DERIVED_CODE_ARBITRATION.md`
- derived-code bridge smoke status:
  `research/cards/MC009_DERIVED_CODE_ARBITRATION_BEHAVIOR_STATUS.md`
- first bridge preregistration:
  `research/prereg/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP.md`
- first bridge status:
  `research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_STATUS.md`
- first bridge V2 authority-dial status:
  `research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V2_AUTHORITY_DIAL_STATUS.md`
- first bridge V3 parse-repair status:
  `research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V3_PARSE_REPAIR_STATUS.md`
- first bridge V4 authority-interface status:
  `research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V4_AUTHORITY_INTERFACE_STATUS.md`
- first bridge route closeout:
  `research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_ROUTE_CLOSEOUT_STATUS.md`
- MC006 V18 margin-matched lead-time status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_V18_MARGIN_MATCHED_LEADTIME_STATUS.md`
- MC006 V19 overlapping-margin table status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_V19_OVERLAPPING_MARGIN_TABLE_STATUS.md`
- MC006 V20 strict-overlap selection audit status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_V20_STRICT_OVERLAP_SELECTION_AUDIT_STATUS.md`
- MC006 V21 pair-matched lead-time status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_V21_PAIR_MATCHED_LEADTIME_STATUS.md`
- MC006 V22 source/path lead-time curve status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_V22_SOURCE_PATH_LEADTIME_CURVE_STATUS.md`
- MC006 V23 final-margin sign-barrier status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_V23_FINAL_MARGIN_SIGN_BARRIER_STATUS.md`
- MC006 V24 delayed-city interface status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_V24_DELAYED_CITY_INTERFACE_STATUS.md`
- MC006 V25 candidate-decoupled template status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_V25_CANDIDATE_DECOUPLED_TEMPLATE_STATUS.md`
- MC006 V26 locked-coordinate transfer status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_V26_LOCKED_COORDINATE_TRANSFER_STATUS.md`
- MC006 V27 expanded candidate-decoupled bank status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_V27_EXPANDED_CANDIDATE_DECOUPLED_BANK_STATUS.md`
- MC006 V28 transfer-role repair bank status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_V28_TRANSFER_ROLE_REPAIR_BANK_STATUS.md`
- MC006 delayed-city closeout status:
  `research/cards/MC006_PARAMETRIC_FACT_OVERRIDE_DELAYED_CITY_CLOSEOUT_STATUS.md`
- MC006 predecision frontier closeout status:
  `research/cards/MC006_PREDECISION_FRONTIER_CLOSEOUT_STATUS.md`
- MC010 two-hop bridge preregistration:
  `research/prereg/MC010_TWO_HOP_FACT_CODE_ARBITRATION.md`
- MC010 two-hop bridge behavior status:
  `research/cards/MC010_TWO_HOP_FACT_CODE_ARBITRATION_BEHAVIOR_STATUS.md`
- MC011 atomic-number bridge preregistration:
  `research/prereg/MC011_ATOMIC_NUMBER_CODE_ARBITRATION.md`
- MC011 atomic-number bridge behavior status:
  `research/cards/MC011_ATOMIC_NUMBER_CODE_ARBITRATION_BEHAVIOR_STATUS.md`
- MC012 reliability-labeled bridge preregistration:
  `research/prereg/MC012_RELIABILITY_LABELED_NUMERIC_ARBITRATION.md`
- MC012 reliability-labeled bridge behavior status:
  `research/cards/MC012_RELIABILITY_LABELED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`
- MC013 status-channel ablation preregistration:
  `research/prereg/MC013_STATUS_CHANNEL_ABLATION_NUMERIC_ARBITRATION.md`
- MC013 status-channel ablation behavior status:
  `research/cards/MC013_STATUS_CHANNEL_ABLATION_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`
- MC014 inferred-reliability bridge preregistration:
  `research/prereg/MC014_INFERRED_RELIABILITY_NUMERIC_ARBITRATION.md`
- MC014 inferred-reliability bridge behavior status:
  `research/cards/MC014_INFERRED_RELIABILITY_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`
- MC015 parity-gated bridge preregistration:
  `research/prereg/MC015_PARITY_GATED_NUMERIC_ARBITRATION.md`
- MC015 parity-gated bridge behavior status:
  `research/cards/MC015_PARITY_GATED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`
- MC016 alphabet-gated bridge preregistration:
  `research/prereg/MC016_ALPHABET_GATED_NUMERIC_ARBITRATION.md`
- MC016 alphabet-gated bridge behavior status:
  `research/cards/MC016_ALPHABET_GATED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`
- MC031 statusless reliability bridge preregistration:
  `research/prereg/MC031_STATUSLESS_RELIABILITY_BRIDGE.md`
- MC031 statusless reliability bridge smoke status:
  `research/cards/MC031_STATUSLESS_RELIABILITY_BRIDGE_STATUS.md`
- MC032 post-checksum cross-table bridge preregistration:
  `research/prereg/MC032_POST_CHECKSUM_BRIDGE.md`
- MC032 post-checksum cross-table bridge status:
  `research/cards/MC032_POST_CHECKSUM_BRIDGE_STATUS.md`
- MC033 fact-claim bridge closeout preregistration:
  `research/prereg/MC033_FACT_CLAIM_BRIDGE_CLOSEOUT.md`
- MC033 fact-claim bridge closeout status:
  `research/cards/MC033_FACT_CLAIM_BRIDGE_CLOSEOUT_STATUS.md`

Current validated snapshot:

- 19 atlas rows;
- 9 law hypotheses;
- law-hypothesis status counts: 1 `strong_doctrine`, 4
  `supported_pattern`, and 4 `tentative_pattern`;
- 1 bounded mechanism card;
- 1 failed mechanism-card route;
- 17 diagnostic notes;
- 6 atlas rows with `OUTPUT_MARGIN_CONFUND`;
- 4 atlas rows with `SIGNATURE_NOT_CAUSAL`;
- 7 atlas rows with `BEHAVIOR_SUBSTRATE_FAILED`;
- 10 atlas rows with `PROMPT_AUTHORITY_DIAL`;
- 3 atlas rows with `SHUFFLED_SELECTION_OVERFIT`;
- 3 atlas rows with `PROMPT_CONTRACT_PARSEABILITY`;
- 2 atlas rows with `PROMPT_FORMAT_CONFUND`;
- 1 atlas row each with the MC010, MC011, MC012, MC013, MC014, MC015, and MC016 bridge diagnostics:
  `MC010_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`,
  `MC011_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`,
  `MC012_BEHAVIOR_READY_PROMPT_VISIBLE_BRIDGE`,
  `MC013_STATUS_ABLATION_FAILED_DIAGNOSTIC_BRIDGE`, and
  `MC014_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`, and
  `MC015_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`, and
  `MC016_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`;
- 1 atlas row each with `STATUS_CHANNEL_POSITIVE_CONTROL_REPRODUCED` and
  `STATUS_CHANNEL_ABLATION_COLLAPSED_CONTRAST`;
- 1 atlas row each with `CALIBRATION_STATUS_LABEL_ABSENT` and
  `CALIBRATION_INFERENCE_CONFLICT_COLLAPSED`;
- 1 atlas row each with `PARITY_GATE_LABELS_BALANCED` and
  `PARITY_GATE_NOT_FOLLOWED`;
- 1 atlas row each with `ALPHABET_GATE_LABELS_BALANCED`,
  `ALPHABET_GATE_LOCAL_COLLAPSE`, and `FEATURE_LABEL_GATE_DID_NOT_RESCUE`;
- 2 `lead_time_monitor_only` rows.
- 53 unique atlas-linked result artifacts parsed by the artifact registry.
- 53 artifacts with extracted common/family metrics.
- all 19 atlas rows have parsed result-artifact support.
- 43 enabled family-level claim checks.
- claim-audit coverage has 19/19 rows with artifacts, metrics, and family-level
  claim checks.
- claim-audit levels: 19 `family_checked_with_metrics`, 0
  `family_checked_partial_metrics`.
- claim-consistency coverage has 120 generic row-claim checks, 0 contradictions,
  and 0 rows with contradictions.
- law-audit coverage has 9/9 evidence-consistent hypotheses, 19/19 atlas rows
  with law support, 60/60 observed diagnostics cited by at least one law, and 0
  unobserved cited diagnostics.
- next-experiment queue coverage has 19 generated items, 5 immediate items, 4
  high-priority items, 9 medium-priority items, 1 watch item, and 9
  immediate-or-high items.
- next-experiment bridge closure context is available, with 24 bridge rungs,
  17 smoke rungs, 0 hidden-state-allowed bridge rungs, 0 clean unconfounded
  bridge rungs, 24 closed contract axes, and recent closed rungs MC030, MC031,
  MC032, and MC033.
- smoke-diagnostic coverage has 17 cards, 17 structural-passed cards, 0
  behavior-ready cards, 0 signature-ready cards, 0 hidden-state-allowed cards,
  and 72 validation checks.
- bridge-ladder coverage has 24 rungs, 7 atlas rungs, 17 smoke rungs, 1
  behavior-ready prompt-visible rung, 0 signature-ready rungs, 0
  hidden-state-allowed rungs, and 0 clean unconfounded bridge rungs.
- 1 checked-in compact artifact index validated against raw artifacts.
- 1 checked-in cross-family comparison validated against the atlas and artifact
  index.
- 0 promoted mechanism cards by current comparison.
- 0.052632 bounded-mechanism ratio.
- 0.947368 diagnostic-or-failed ratio.
- 0.315789 output-margin-confounded row ratio.
- 0.368421 behavior-substrate-failed row ratio.
- 0.736842 intervention not-allowed-or-failed ratio.
- reliability matrix: 0 full-reliability mechanisms, 1 bounded reliability
  reference, 11 behavior/bridge-blocked rows, 3 output-shadow diagnostic rows,
  2 monitor-only rows, 1 failed-intervention route, and 1 prompt-visible
  positive control.
- reliability missing-gate counts: 19/19 rows still miss clean predicted
  intervention, null/locality cleanliness, robustness/side-effect clearance,
  and transfer/widening; 18/19 miss a control-surviving signature and local
  internal path; 11/19 miss the behavior substrate.
- error taxonomy: 17 smoke cards, 24 bridge rungs, 0 hidden-state-allowed
  smoke cards, and MC028's 28/160 full-source operation-atomic other-number
  rows bucketed as 9 worked-example outputs, 8 other bank atomic numbers, 6
  off-bank other numbers, 3 bank local numbers, and 2 prompt-local row
  numbers, plus MC029's factorized branch/null/example-leak tradeoff where
  `rules_only` improves operation-atomic atomic to 0.875 while answer-absent
  UNKNOWN falls to 0.806, plus MC030's absence-guard repair failure where
  guards worsen nulls or collapse the learned branch, plus MC031's statusless
  checksum failure where invalid-checksum rows collapse to local answers while
  direct controls and nulls stay clean, plus MC032's cross-table consistency
  failure where mismatch rows select atomic/lure numbers 0/10 times, mostly
  collapse to the primary local row, and never copy the second-table side
  number, plus MC033's fact-claim closeout where direct controls and nulls
  stay clean but match rows return learned atomic numbers too often and
  mismatch rows split between local answers and the wrong claimed number.
- gate geometry: 12/19 atlas rows stop before signature work is licensed, 5/19
  stop at the signature stage, 1/19 stops at failed intervention, 1/19 is
  bounded at reliability, and 0/19 are promoted mechanisms. The bridge funnel
  has 23/24 rungs at behavior-substrate closure and 1/24 prompt-visible
  positive control.
- genome snapshot: 19 atlas rows, 53 linked result artifacts, 1 bounded
  mechanism card, 18 diagnostic-or-failed rows, 0 promoted mechanisms, 0
  hidden-state-allowed bridge rungs, and the top next-queue pressure still on
  a materially different bridge that preserves MC012-level controls without
  visible source-status text.
- axis interactions: 123 feature summaries, 14 pure predictive rules, 10
  broad-or-supported pure rules, and 31 mixed predictors. The strongest broad
  pure rule is `closed_before_hidden_state -> pre_signature_behavior_substrate`
  across 11 rows; output-geometry pressure is broad but mixed across
  output-shadow, monitor-only, and failed-intervention terminal stages.
- coverage gaps: 12 named gaps, including 4 critical gaps: no promoted
  mechanism, no full reliability, no clean predicted intervention, and no
  transfer-ready mechanism. The layer also records 14 transfer-untested rows,
  0 hidden-state-allowed bridge rungs, 0 clean unconfounded bridge rungs, and
  82 singleton-or-sparse feature summaries.
- gap closure plan: 6 work orders covering 12/12 coverage gaps, 4/4 critical
  gaps, and 5/5 top queue items. The plan binds each next line to promotion,
  bounded-claim, kill, containment, and export rules while preserving the
  current fact that there are 0 promoted mechanisms, 0 full-reliability rows,
  and 0 hidden-state-allowed bridge rungs.
- offensive doctrine: 6 branch contracts generated from the gap-closure plan.
  Each contract names target gaps, expected generated-layer movement, minimum
  evidence, promotion, bound, kill, containment, and export rules. The central
  validator now fails if this branch-intake harness is missing or stale.
- transfer width probe: the immediate widening branch now has a generated
  preregistration packet for MC005/MC003/MC004. It specifies MC005 as the
  bounded reference surface, MC003 as an output-shadow baseline, MC004 as a
  predecision-monitor baseline, Gemma as the first non-Qwen target family, and
  6 panels covering primary effect, null locality, side effects, prompt
  robustness, output-shadow baseline, and monitor-only baseline.
- singleton stage replication pack: the high-priority law-replication branch
  now has a generated preregistration packet for the three singleton terminal
  stages: `intervention_failed`, `pre_signature_prompt_channel_locality`, and
  `reliability_null_boundary`. The packet proposes two materially distinct row
  additions per stage and keeps law promotion forbidden until new atlas rows
  actually land in the same terminal stage.
- MC031/MC032/MC033 continuation artifacts: the statusless arithmetic-checksum
  bridge, the post-checksum cross-table bridge, and the fact-claim closeout are
  preregistered and structurally valid on 840 records. Their 10-source model
  smokes are not full-source behavior verdicts, but together they show that
  three distinct statusless source-validity cues preserve direct controls and
  nulls while failing to route invalid, mismatch, or fact-claim conflict rows
  to the intended learned/local branch. These remain outside the validated
  atlas rows and keep hidden-state work forbidden on this route.

The central empirical fact is still uncomfortable and useful:

> Most current surfaces are output-visible, prompt-coupled, source-token
> dependent, or diagnostic. Clean local causal mechanisms are the minority.

That ratio is not a disappointment. It is the first measured shape of the
genome.

## Law Layer

The current hypotheses are theory candidates. They are allowed to guide the
next experiments, but each one has falsifiers in
`data/control_surface_law_hypotheses.json`.

| Hypothesis | Current Status | Short Form |
| --- | --- | --- |
| `final_state_output_geometry_dominance` | supported pattern | Perfect final-position hidden AUC is usually a warning sign unless output/candidate geometry is controlled. |
| `lead_time_monitor_before_lever` | supported pattern | Early hidden signals are monitors until an intervention proves they are levers. |
| `source_visible_lookup_localizes_more_than_parametric_override` | supported pattern | Synthetic prompt-visible lookup localizes better than factual override. |
| `coarse_source_ablation_overstates_circuit_locality` | tentative pattern | Coarse source ablations can be real behavior controls without being localized circuits, but current evidence is single-row. |
| `null_reliability_bottleneck` | supported pattern | Primary effects can be easier than null locality; null rows define the boundary. |
| `behavior_substrate_first_or_everything_lies` | strong doctrine | Weak behavior tables make hidden-state results untrustworthy. |
| `transfer_fails_at_reliability_before_primary_effect` | tentative pattern | Primary effects may transfer before null reliability transfers. |
| `familiar_entities_can_collapse_to_lookup_keys` | tentative pattern | Familiar names alone do not create knowledge conflict under a terse table-authoritative contract. |
| `authority_pressure_creates_contrast_before_clean_substrate` | tentative pattern | Prompt-authority pressure can create contrast before the behavior table is clean enough for probing. |

## What The Current Map Predicts

### 1. Final-Token Probes Will Keep Looking Too Good

The atlas predicts that most final-token hidden classifiers on answer-selection
tasks will be matched by output-margin, next-token, candidate-score, prompt
format, or requested-mode baselines.

Operational rule:

> A final-token hidden signature is guilty until it beats final-output and
> candidate-score controls on a balanced, source-disjoint holdout.

This is not pessimism. It is a specific prediction. If a final-token signature
beats those controls, the hypothesis weakens immediately.

### 2. Lead-Time Is The Main Pre-Mechanism Frontier

MC006 V16 is important because it separates same-position output visibility
from final-output visibility. It shows a hidden trace before the local output
interface fully catches up. MC006 V17 is equally important because it shows
that this trace is not automatically a steering vector. MC006 V18 adds the
table-geometry boundary: on the current V14 split, true and override holdout
rows do not overlap in candidate-score or final-output margins, so
margin-matched promotion is blocked before intervention. MC006 V19 then tested
whether a broader prompt bank could repair that row geometry. It improved
behavior balance and produced near-matched pairs, but strict margin overlap
still failed. MC006 V20 then showed that strict final-output overlap is absent
even in the pooled V19 binary bank, so matched-template or per-source selection
cannot rescue the existing prompt bank. MC006 V21 then ran the bounded
approximate pair-matched diagnostic and found that the match did not neutralize
the easy baselines: candidate-score and final-output margins still achieved
1.000 holdout pair accuracy, while the selected pre-output hidden direction
fell to 0.286 holdout pair accuracy. MC006 V22 then replaced the single-probe
habit with a source/path and line-boundary curve. It found real monitor
structure at the queried-country token and later pre-output line boundaries,
but final candidate-score and final-output margins still achieved 1.000
holdout pair accuracy. MC006 V23 then audited the final-margin gate directly:
final next-token margin sign predicted every binary label in V18 and V19, while
candidate-score margin had overlap in V19 all-binary rows. MC006 V24 changed
the answer interface with delayed JSON city generation. That broke the
first-token city-margin sign barrier, but JSON-completion candidate-score
holdout AUC reached 1.000 and beat the selected hidden monitor. MC006 V25 then
selected a different delayed-city template where JSON-completion candidate-score
holdout AUC fell to 0.333 and the hidden monitor reached 1.000 holdout AUC, but
split-preserving shuffled-label selected searches also reached 1.000 p95
holdout AUC. MC006 V26 locked that coordinate and tested transfer to
`separate_task_weak`; source holdout AUC was 0.750, but transfer holdout AUC was
0.500 while a position-local output control and fixed-coordinate train-label
shuffle p95 both reached 0.833.
MC006 V27 then expanded the delayed-city bank to 640 rows with predeclared
source/transfer template roles. It found four candidate-decoupled templates,
138 pooled binary rows, and balanced pooled holdout labels, but only one
transfer-role template passed.
MC006 V28 then targeted transfer-role repair directly. It generated another
640 rows across 16 predeclared transfer-role templates and found one clean
transfer template, `transfer_sandbox_mapping_then_geo`, but not the two
transfer-ready templates required for a source/transfer behavior bank.
The delayed-city route is now closed as monitor-only under the tested prompt
families.

Operational rule:

> Lead-time signatures should be reported as `monitor_only` until intervention
> proves otherwise.

The next improvement is no longer "run the first curve," "generate more rows
under the same greedy answer interface," "add a wrapper before the city," or
"find any candidate-decoupled template." V22 ran the curve, V23 diagnosed the
greedy final-margin sign barrier, V24 showed that one delayed JSON city target
still loses to full-completion candidate scoring, and V25 found a
candidate-decoupled target that still fails shuffled-label selection nulls. V26
then showed that locking the V25 coordinate does not transfer across the current
candidate-decoupled templates. V27 then showed that a larger table can repair
source-role candidate decoupling but not transfer-role coverage. V28 then
showed that direct transfer-role repair can find one clean template but not a
bank. The next improvement is not another ordinary transfer-template sweep; it
is either a materially different causal stress test that treats V22-V28 evidence
as known-confounded, or a new behavior family. The delayed-city route itself is
closed as monitor-only.

### 3. Synthetic Lookup And Factual Override Are Different Parts Of The Ladder

MC005 is source-visible, synthetic, and strongly prompt-mediated. That is why
source-value attention/write paths can be localized enough to make a bounded
mechanism candidate.

MC006 is knowledge-like and final-output-visible. It has a behavior substrate
and an early monitoring signal, but the output/candidate interface already
knows too much by the time most probes read it. V18 made this sharper: the
current holdout split has zero matched true/override pairs at z <= 0.5 for both
candidate-score and final-output margins. V19 improved this to four holdout
pairs at z <= 0.5 for both margins in the selected `separate_task_weak` table,
but the candidate and final-output class ranges still did not overlap. V20
found 28 pooled holdout joint pairs at z <= 0.5, but no strict final-output
overlap; this allowed a bounded pair-matched diagnostic, not promotion. V21 ran
that diagnostic and showed the near-pairs were still ordered by candidate and
final margins on every holdout pair, so approximate 0.5z matching is too weak
to rescue the existing MC006 table. V22 then showed that source/path probing
does not rescue it either: the curve has monitors, but the final candidate and
output margins still perfectly order holdout pairs. V23 then showed why strict
final-output overlap is not a reasonable default target under this interface:
the final next-token margin sign is the generated binary-label boundary. V24
then changed the answer format so the first generated token is a JSON wrapper,
not the city. That fixed the V23 first-token barrier but exposed the next law:
completion-level candidate scoring can still be a stronger behavioral readout
than the internal monitor.

Operational rule:

> The bridge task should not jump directly from MC005 to real factual override.
> It should use real or familiar entity names with artificial values.

MC007 V1 tested that bridge and found the first boundary: familiar country
names collapsed into source-value lookup keys under a terse table-authoritative
contract. The selected `lookup_only` template produced 75/80 artificial-value
primary rows, 0 real-prior rows, 0 lure rows, and 40/40 `UNKNOWN`
answer-absent null rows. That is a behavior pass but not a signature substrate.

MC007 V2 then added an authority dial without leaking the true capital. The
numeric dial created partial real-prior/lure contrast: across all familiar
authority panels it produced 119 artificial-value rows, 24 real-prior rows, 6
lure rows, and 51 unparsed rows. The best within-panel near miss was numeric
authority 0, with 15 artificial and 15 real-prior/lure rows, but parseability
was only 75 percent. The result is useful, but still not a signature substrate.

MC007 V3 then tested the obvious parse repair: target-only or compact task
notes, a stricter `Answer:` slot, direct geography controls, synthetic lookup
controls, and answer-absent nulls. It failed with diagnostic class
`parseability_repair_failed`. The selected `compact_note` template had 62
artificial-value rows, 2 real-prior rows, 2 lure rows, and 54 unparsed rows
across 120 primary rows. Direct geography stayed clean at 38/40 prior/lure and
answer-absent nulls stayed 40/40 `UNKNOWN`. The failure is therefore specific
to the target-note conflict prompt, not the whole parser.

MC007 V4 then tested a materially different authority interface with generated
`CITY` and `SOURCE` fields. It failed with diagnostic class
`control_panel_failed`: the selected `city_then_source` template had only
26/40 artificial synthetic-control rows, 30/40 `UNKNOWN` answer-absent null
rows, 57/80 parseable explicit-authority rows, and 45/80 parseable conflict
rows. It also exposed the new boundary: source declarations and city fields
decouple. Source-city consistency was 38/80 on explicit rows and 12/80 on
conflict rows, so source-choice labels are not valid proxies for generated
city behavior.

### 4. Source Dependence Is Not Circuit Locality

MC001 proved that source tokens can causally matter without proving a compact
internal path. Coarse input masking moved behavior strongly, but literal
rewrites and prompt-state recomputation explained most of the effect.

Operational rule:

> Every source-token claim needs source deletion, neutral rewrite, and
> query-only/path-local comparisons.

Without those, a source effect is only a behavior dependence claim.

### 5. Null Rows Are The Real Mechanism Boundary

MC005 V29-V31 shows the shape clearly:

- lookup write replacement is large and stable on high-margin lookup rows;
- answer-absent null flips are rare;
- null flips cluster near low-to-moderate target/distractor margins;
- the strict 0.5-margin explanation failed.
- the executable closeout audit validates V27-V31 and freezes the exact
  write-replacement route as bounded, not promoted.

Operational rule:

> A primary intervention effect is not enough. The null margin strata are part
> of the mechanism.

That means MC005 is now a bounded mechanism card without being a promoted
mechanism card.

## Gap Table

| Gap | Why It Matters | Current Evidence | Highest-Value Fill |
| --- | --- | --- | --- |
| No second bounded mechanism card | One bounded surface could be idiosyncratic. | MC005 is now frozen bounded; MC007 V1-V4 is a closed diagnostic bridge rather than a probe substrate; MC008 V1 repaired direct symbolic controls but failed null reliability and conflict balance; MC008 V2 repaired nulls but conflict stayed table-dominant, closing the first symbolic route; MC010-MC016 each close a bridge repair route before hidden-state work; MC006 V18-V28 block current strict-overlap, approximate-pair, source/path final-margin-rescue, same-interface final-overlap row-generation, delayed-city first-token-repair, candidate-decoupled selected-hidden, locked-coordinate transfer, expanded transfer-bank, and direct transfer-role repair routes; delayed-city is closed monitor-only. | New bridge family, or materially different MC006/MC005 intervention stress with preregistered null controls. |
| No cross-family full lead-time curve | Single early probes cannot locate commitment time, and V22 only maps MC006. | MC004 and MC006 near misses; MC006 V22 now has a source/path and line-boundary curve. | Repeat the curve shape on another behavior family or new MC006 row distribution. |
| No transfer-robust candidate-decoupled MC006 signature | V25 found a behavior-passing delayed-city template where JSON candidate scoring is not perfect, but the hidden selector failed the null-selection test; V26 then showed the locked coordinate does not transfer; V27 then showed the expanded bank is transfer-insufficient; V28 then showed direct transfer repair finds one template, not a bank. | V16/V17 found monitor-only and failed steering; V18 found no holdout overlap; V19 improved balance and pair proximity; V20 showed strict final-margin overlap is absent; V21 showed approximate pairs still have perfect margin-baseline ordering; V22 showed source/path monitors remain final-margin-shadowed; V23 showed final-margin sign predicts generated labels; V24 broke that first-token barrier but JSON candidate-score holdout AUC was 1.000; V25 found `untrusted_note_real` with JSON candidate-score holdout AUC 0.333, but shuffled-label selected-search p95 also reached 1.000; V26 locked `after_mapping_line/layer_10` and transfer holdout AUC fell to 0.500 on `separate_task_weak`; V27 found 3 source-ready templates but only 1 transfer-ready template; V28 found `transfer_sandbox_mapping_then_geo` ready but only one transfer-ready template overall; delayed-city is now closed monitor-only. | Use as a closed diagnostic row, or run a materially different known-confounded causal stress. |
| No promoted mechanism card | Bounded mechanisms are useful, but full promotion still requires strict locality, transfer, and side-effect evidence. | MC005 is bounded by answer-absent null locality; MC006, MC007, MC008, MC009, MC010, MC011, MC012, MC013, MC014, MC015, and MC016 are diagnostic routes. | A repaired behavior family or materially different intervention that passes strict null/locality/transfer gates. |
| Transfer evidence is thin | Genome claims require cross-model structure. | MC005 0.6B null fragility. | Matched primary/null transfer panels for MC005 or MC007. |
| Atlas metrics are artifact-extracted but not yet verdict-complete | Current rows are no longer prose-only or partial-metric, and generic row-claim contradictions now fail validation. Family-specific verdict boundaries still need deeper automatic checks. | Parser registry now extracts 53 linked result artifacts across all 19 atlas rows, extracts metrics from all 53, writes a compact checked-in artifact index, fails direct row/artifact readiness contradictions, checks 43 family-level closure facts, and runs 120 generic claim-consistency checks with 0 contradictions. | Add deeper family-specific metric parsers so verdict, baseline, null, and intervention claims can fail automatically. |
| Source-token locality is under-separated | Source removal can masquerade as path control. | MC001 V9-V13. | Mandatory rewrite-equivalence fields in future atlas rows. |

## Next Experiment Queue

### Priority 1: New Bridge Family After MC007 And MC008 Closeouts

Closed route:

> MC007 now has four behavior facts: V1 is clean source-value lookup with no
> prior contrast, V2 creates contrast but is parse-fragile, V3 shows that
> answer-slot strictness alone does not repair the conflict panel, and V4 shows
> that explicit source declarations can decouple from generated city answers
> while controls fail. The current route is now closed as
> `MC007_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE`.

MC008 V1 result:

- selected template: `symbol_field`;
- synthetic lookup control: 39/40 artificial-code rows, 40/40 parseable;
- real-world symbol control: 39/40 real-symbol rows, 39/40 parseable;
- answer-absent null: 40/40 parseable but only 30/40 `UNKNOWN`;
- primary conflict: 176 artificial-code rows, 8 real-symbol rows, 0 lure rows,
  47 unparsed rows, and 80.4 percent parseability;
- verdict: `symbolic_null_control_failed`, diagnostic note only.

MC008 V2 result:

- selected template: `membership_authority_split`;
- synthetic lookup control: 39/40 artificial-code rows, 40/40 parseable;
- real-world symbol control: 40/40 real-symbol rows, 40/40 parseable;
- answer-absent null: 40/40 `UNKNOWN`;
- primary conflict: 220 artificial-code rows, 2 real-symbol rows, 0 lure rows,
  9 unparsed rows, and 96.25 percent parseability;
- authority-0 rows: only 1/40 real-symbol answers;
- verdict: `symbolic_conflict_contrast_absent`, route closeout diagnostic.

Next bridge decision:

- do not run MC008 hidden-state discovery from V1 or V2;
- do not treat compact symbolic outputs as a solved bridge;
- close the first symbolic route as diagnostic;
- close MC009's first derived-code route as diagnostic rather than continuing
  ordinary row-code prompt repairs;
- keep the MC007 V1 `lookup_only` template as a clean source-value baseline;
- keep the MC007 V2 numeric low-authority prompt as the prior-pressure
  baseline;
- stop relying on target-only notes, stricter `Answer:` wording, or explicit
  source labels as the main repair;
- preserve the MC008 direct-control improvement: compact generated symbolic
  codes rather than open city names;
- do not print the true factual answer in conflict prompts;
- score generated answers, not only candidate scores;
- preserve answer-absent null rows;
- only after a future bridge pass, measure source-token dependence,
  output/candidate visibility, lead-time, intervention, null locality, and
  transfer.

Promotion condition:

- a repaired generated table has both artificial-value and real-prior/lure
  outcomes on non-holdout and holdout splits, then a source/path-specific
  internal surface beats output/candidate controls and supports a local
  intervention with null locality.

Bounded success:

- source-value behavior remains strong, but prior-pressure or null rows bound
  the claim.

Death condition:

- a new bridge route cannot create real/lure contrast while preserving
  parseability, nulls, and source-disjoint holdout balance.

Why this is first:

MC007 V3 and V4 already tested the two obvious repairs and killed them, and
the route closeout makes that a formal diagnostic boundary:

> lookup localizes when the answer is prompt-visible; factual override becomes
> output-visible. V1 showed familiar names alone are not enough. V2 showed
> authority pressure moves the transition but can break the behavior substrate.
> V3 showed answer-slot repair does not make that transition measurable enough
> for probing. V4 showed source labels are not safe proxies for generated city
> behavior.

MC008 V1-V2 is now a closed symbolic bridge diagnostic. It shows compact answer
classes can repair direct controls, and a membership contract can repair nulls,
but matching prompt-local task codes still dominate artificial-versus-real
conflict rows.

MC009 tested the next hypothesis:

> direct source-value binding may be the reason MC008 collapsed to task codes.

MC009 kept compact generated answers but removed the printed
`entity -> artificial code` row. The task answer is derived from row position
under a generic rule, so the model must compute the task-local code from source
structure rather than copy a matching prompt value.

MC009 now closes the first derived-code route as diagnostic. The 10-source
`membership_authority_split` smoke validated the structural audits and direct
controls: synthetic ordinal lookup, real-symbol memory, and answer-absent nulls
were all 10/10. But the bridge still blocked hidden-state work: primary
conflict parseability was 47/60, only 7/60 primary conflict rows were
real-symbol rows, and source-disjoint real/lure balance failed.

The non-default `typed_slot_v2` repair proved why contrast alone cannot be the
gate. It reached primary conflict parseability of 54/60, produced 23
derived-code rows and 19 real-symbol rows, and passed non-holdout plus holdout
balance. But it broke the controls: synthetic ordinal lookup produced 0/10
derived-code rows, answer-absent nulls produced 0/10 `UNKNOWN`, and the typed
slot itself made the source channel prompt-visible. The lesson is sharper than
"MC009 failed": removing direct source-value copying is insufficient, and a
prompt can manufacture apparent conflict balance by changing the answer channel.

Next bridge implication:

- do not run MC009 hidden-state work;
- do not continue ordinary row-code prompt repairs;
- preserve the diagnostic classes `DERIVED_CODE_CONTROL_CONFLICT_TRADEOFF`,
  `DERIVED_CODE_TYPED_SLOT_PROMPT_VISIBLE`, and
  `MC009_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE`;
- the next bridge must materially change the conflict family or become MC010.

### Priority 2: MC006 Closed Monitor Or Known-Confounded Stress

V18 result:

> MC006 has a pre-output signal that is not merely same-stage output geometry,
> but the current V14 table cannot test margin-matched promotion because
> candidate-score and final next-token output margins have no true/override
> holdout overlap.

V19 result:

> A broader prompt bank can produce a better balanced table. The selected
> `separate_task_weak` contract has 35/40 binary rows and 4 true / 4 override
> holdout rows, but strict candidate and final-output margin overlap still
> fails. It has near-matched pairs, not a promotion substrate.

V20 result:

> The V19 bank itself lacks strict final-output overlap. Pooled holdout
> final-margin separation is only 0.0612z, and there are 28 pooled holdout
> joint pairs at z <= 0.5, but no matched-template or per-source subset can
> rescue strict overlap from the existing rows.

V21 result:

> The bounded approximate pair-matched audit failed as a control. It found 451
> non-holdout and 28 holdout joint pairs at z <= 0.5, but candidate-score and
> final-output margins still reached 1.000 holdout pair accuracy. The selected
> `after_mapping_line/layer_14` hidden direction reached 0.918 non-holdout
> pair accuracy but only 0.286 holdout pair accuracy. Approximate matching
> overfit and did not neutralize the easy baselines.

V22 result:

> The source/path curve is mapped. The best source/path candidates were at
> `question_country_token/layer_20` and `question_country_token/layer_22`;
> the AUC-selected candidate reached 0.832 holdout AUC and 0.714 holdout pair
> accuracy. Later line-boundary monitors were stronger, including
> `after_question_line/layer_22` at 0.893 holdout pair accuracy. But
> candidate-score, V19 final, and V22 final-prompt next-token margins all
> remained at 1.000 holdout pair accuracy. This is
> `source_path_final_margin_shadow`, not a mechanism signature.

V23 result:

> The strict final-margin target is malformed for the current greedy binary
> answer interface. Final next-token margin sign predicted 30/30 V18 binary
> labels and 257/257 V19 all-binary labels, with no raw final-margin overlap.
> Candidate-score margin behaved differently: V19 all-binary candidate-score
> margin had raw overlap and only 0.953 sign accuracy. This is
> `greedy_final_margin_sign_barrier`.

V24 result:

> Delayed JSON city generation fixed the narrow first-token interface problem.
> The selected `game_code_then_geo` table had 31 binary rows, 10 true rows, 21
> override rows, and source-disjoint holdout coverage of 2 true / 4 override.
> The selected-city first-token rate was 0.000, final city-token sign accuracy
> fell to 0.516 with raw overlap, and the best hidden monitor reached 0.875
> holdout AUC. But JSON-completion candidate-score holdout AUC reached 1.000.
> This is `delayed_city_interface_decouples_first_token_margin`, mapped in the
> atlas as `DELAYED_INTERFACE_CANDIDATE_SCORE_VISIBLE`.

V25 result:

> The V24 row bank contains a candidate-score-decoupled delayed-city template.
> The selected `untrusted_note_real` table had 37 binary rows, 24 true rows, 13
> override rows, and source-disjoint holdout coverage of 4 true / 3 override.
> JSON-completion candidate-score holdout AUC fell to 0.333, while the selected
> `after_mapping_line/layer_10` hidden monitor reached 1.000 holdout AUC. But
> split-preserving shuffled-label selected searches also reached 1.000 p95
> holdout AUC. This is `candidate_decoupled_hidden_shuffle_overfit`, mapped in
> the atlas as `CANDIDATE_DECOUPLED_SHUFFLE_OVERFIT`.

V26 result:

> Locking the V25 coordinate does not rescue the route. The
> `after_mapping_line/layer_10` coordinate reproduced on `untrusted_note_real`
> source holdout at 0.750 AUC, but transferred to `separate_task_weak` at only
> 0.500 holdout AUC. The best transfer control was the `after_mapping_line`
> next-token city margin at 0.833, and the fixed-coordinate train-label shuffle
> p95 was also 0.833. This is `locked_coordinate_transfer_failed`, mapped in the
> atlas as `LOCKED_COORDINATE_TRANSFER_FAILED`.

V27 result:

> The larger delayed-city bank is still not hidden-state-ready. V27 generated
> 640 rows across 16 predeclared source/transfer templates. Four templates were
> candidate-decoupled, pooled binary rows reached 138, and pooled holdout labels
> were balanced at 14 true / 14 override. But the predeclared transfer-role gate
> failed: 3 source templates passed and only 1 transfer template passed. This is
> `expanded_candidate_decoupled_bank_insufficient`, mapped in the atlas as
> `EXPANDED_CANDIDATE_DECOUPLED_BANK_INSUFFICIENT`.

V28 result:

> Direct transfer-role repair is still insufficient. V28 generated 640 rows
> across 16 predeclared transfer-role templates. It found one ready transfer
> template, `transfer_sandbox_mapping_then_geo`, with 35 binary rows, balanced
> 4 true / 4 override holdout labels, JSON candidate-score holdout AUC 0.1875,
> final-city holdout AUC 0.3125, and city-candidate holdout AUC 0.4375. But the
> predeclared repair gate required at least two transfer-ready templates; the
> combined V27+V28 ready bank had only 4 templates and 143 pooled binary rows.
> This is `transfer_role_repair_bank_insufficient`, mapped in the atlas as
> `TRANSFER_ROLE_REPAIR_BANK_INSUFFICIENT`.

Closeout:

> The MC006 delayed-city branch is closed as monitor-only. V14-V28 preserve a
> useful behavior, monitor, and confound map, but block the ordinary promotion
> routes: final-token hidden separators, pre-output monitor promotion, additive
> steering, strict/approximate margin matching, source/path rescue, delayed JSON
> repair, flexible hidden selection, locked-coordinate transfer, expanded bank
> construction, and direct transfer-role repair. This is
> `delayed_city_route_closed_monitor_only`, mapped in the atlas as
> `DELAYED_CITY_ROUTE_CLOSED_MONITOR_ONLY`.

Executable closeout:

> `research/cards/MC006_PREDECISION_FRONTIER_CLOSEOUT_STATUS.md`

The closeout audit validates V14-V28 and returns `monitor_only_closed`: behavior
and predecision monitors exist, but hidden-state work, intervention, and
ordinary route repairs remain disallowed under the current prompt family.

Next design sketch:

- do not run another ordinary transfer-template sweep under the same
  delayed-city family as a promotion route;
- treat prompt template as a first-class confound if pooling templates;
- treat the V22 source/path curve as a completed diagnostic map, not as a
  final-margin rescue;
- treat V23 as closing same-interface strict final-margin row generation;
- treat V24 as closing delayed-city first-token repair as a promotion route;
- treat V25 as closing flexible hidden-selection on the small
  candidate-decoupled delayed-city table as a promotion route;
- treat V26 as closing simple locked-coordinate transfer on the current V24
  candidate-decoupled bank;
- treat V27 as closing source-side-only expanded bank construction as a
  promotion route;
- treat V28 as closing direct transfer-role repair under the current
  delayed-city prompt family as a promotion route;
- treat the delayed-city route as closed monitor-only;
- if pursuing causality without promotion, preregister a materially different
  stress intervention on later pre-output line-boundary monitors;
- otherwise move to MC005 bounded closeout, MC007 bridge redesign, or a new
  behavior family;
- report same-stage output, final-output, and candidate-score controls beside
  every hidden score;
- do not intervene unless the signature survives those controls.

Promotion condition:

- a pre-output hidden signal beats same-stage and final-output/candidate
  controls on holdout, on an answer interface where final-margin controls are
  not label-tautological and full-completion candidate scoring is not already a
  perfect readout.

Bounded success:

- the signal beats same-stage controls but remains final-output visible:
  `lead_time_monitor_only`, `source_path_final_margin_shadow`, or
  `greedy_final_margin_sign_barrier`;
- the answer interface breaks the first-token barrier but remains
  candidate-score visible: `DELAYED_INTERFACE_CANDIDATE_SCORE_VISIBLE`.

Death condition:

- a changed answer interface still leaves labels perfectly visible to
  full-completion candidate controls, or materially different source/path
  interventions cannot move behavior beyond output/candidate controls under a
  source-disjoint holdout.

### Priority 3: MC005 Frozen Bounded Or Materially Different Stress

Closed claim:

> The current write-replacement surface is lookup-causal but bounded by
> answer-absent null locality. MC005 is now frozen as
> `MC005_BOUNDED_ATTENTION_WRITE_MEDIATION`.

Executable closeout:

> `research/cards/MC005_WRITE_REPLACEMENT_CLOSEOUT_STATUS.md`

Future-use rule:

- do not continue ordinary repairs of the exact V29-V31 write-replacement
  route;
- use MC005 as the bounded calibration specimen;
- only reopen MC005 for a materially different intervention, a matched
  transfer/null panel, or a comparative baseline for a new bridge family.

Do not keep trying small threshold repairs. V31 already failed the strict 0.5
margin explanation.

### Priority 4: Atlas Artifact Extraction

Claim to test:

> The project becomes cumulative only if more row fields are extracted from
> result artifacts instead of maintained by hand.

V1 build:

- added `code/control_surface_artifacts.py`;
- parses 53 unique result artifacts linked from all 19 atlas rows;
- parses 46 summary-schema artifacts and 7 legacy-schema artifacts;
- extracts common/family metrics from 53 artifacts;
- writes `data/control_surface_artifact_index.json`;
- extracts row id, card id, run type, model id, record count, diagnostic class,
  pass/fail, behavior/signature/intervention readiness, selected
  templates/layers/positions/coordinates/doses, failed criteria, null criteria,
  baseline/control fields, and intervention/locality fields where schemas expose
  them;
- integrated the registry into `code/validate_control_surface_atlas.py`;
- validation now fails on direct row/artifact readiness contradictions.
- validation now fails if the checked-in artifact index is stale;
- validation now also checks 43 central family-level closure facts:
  MC001 source-mask/rewrite equivalence, MC001B raw-versus-residual signal,
  MC001G intervention non-movement, MC002 behavior-substrate failure, MC002B
  pressure failure, MC003 shuffle/condition-trace signature failure, MC004
  lead-time shuffle/subgroup failure, MC005 layers-24-26 write boundary, MC005
  lookup effect with null flips, MC005 non-simple margin cutoff null boundary,
  MC006 additive-steering failure, MC006 V16 output/candidate confound, MC006
  V21 pair-matching margin-baseline failure, MC006 V22 source/path shadow,
  MC006 V24 delayed-city monitor-only boundary, MC006 V25 candidate-decoupled
  shuffle overfit, MC006 V28 one-template-not-bank transfer result, MC006
  shuffle/transfer failure, MC007 contrast/control failure plus the V1 source
  lookup, V2 authority-dial, V3 parseability-repair, and V4 source-declaration
  control failures, MC008 direct-controls-before-null failure and V2
  null-repaired/conflict-absent boundary, MC009 membership/control and
  typed-slot/control tradeoffs, MC010 two-hop direct-control and conflict-table
  failure, MC011 numeric direct controls plus conflict collapse, MC012
  reliability-labeled contrast plus visible prompt-channel block, MC013
  statused positive-control reproduction plus matched-ablation collapse,
  MC014 direct-control cleanliness plus calibration-inference collapse, MC015
  direct-control cleanliness plus parity-gate rule-following failure, and MC016
  direct-control cleanliness plus alphabet-gate local collapse.

Current command:

```powershell
python code\control_surface_artifacts.py
python code\control_surface_artifacts.py --write-index
python code\control_surface_comparison.py --write
python code\control_surface_comparison.py
python code\control_surface_law_audit.py --write
python code\control_surface_law_audit.py
python code\control_surface_next_queue.py --write
python code\control_surface_next_queue.py
python code\control_surface_smoke_diagnostics.py --write
python code\control_surface_smoke_diagnostics.py
python code\control_surface_bridge_ladder.py --write
python code\control_surface_bridge_ladder.py
python code\control_surface_mixture_law.py --write
python code\control_surface_mixture_law.py
python code\control_surface_decision_frontier.py --write
python code\control_surface_decision_frontier.py
python code\control_surface_route_disposition.py --write
python code\control_surface_route_disposition.py
python code\control_surface_transfer_matrix.py --write
python code\control_surface_transfer_matrix.py
python code\control_surface_reliability_matrix.py --write
python code\control_surface_reliability_matrix.py
python code\control_surface_error_taxonomy.py --write
python code\control_surface_error_taxonomy.py
python code\control_surface_gate_geometry.py --write
python code\control_surface_gate_geometry.py
python code\control_surface_genome_snapshot.py --write
python code\control_surface_genome_snapshot.py
python code\control_surface_axis_interactions.py --write
python code\control_surface_axis_interactions.py
python code\control_surface_coverage_gaps.py --write
python code\control_surface_coverage_gaps.py
python code\control_surface_gap_closure_plan.py --write
python code\control_surface_gap_closure_plan.py
python code\control_surface_offensive_doctrine.py --write
python code\control_surface_offensive_doctrine.py
python code\transfer_width_probe_mc005_mc003_mc004.py --write
python code\transfer_width_probe_mc005_mc003_mc004.py
python code\singleton_stage_replication_pack.py --write
python code\singleton_stage_replication_pack.py
python code\control_surface_knowledge_first_run_outcomes.py --write
python code\control_surface_knowledge_first_run_outcomes.py
python code\control_surface_knowledge_failure_topology.py --write
python code\control_surface_knowledge_failure_topology.py
python code\control_surface_knowledge_second_wave_outcomes.py --write
python code\control_surface_knowledge_second_wave_outcomes.py
python code\control_surface_knowledge_third_wave_outcomes.py --write
python code\control_surface_knowledge_third_wave_outcomes.py
python code\control_surface_knowledge_fourth_wave_outcomes.py --write
python code\control_surface_knowledge_fourth_wave_outcomes.py
python code\control_surface_knowledge_fifth_wave_outcomes.py --write
python code\control_surface_knowledge_fifth_wave_outcomes.py
python code\control_surface_knowledge_sixth_wave_outcomes.py --write
python code\control_surface_knowledge_sixth_wave_outcomes.py
python code\control_surface_knowledge_seventh_wave_outcomes.py --write
python code\control_surface_knowledge_seventh_wave_outcomes.py
python code\control_surface_knowledge_eighth_wave_outcomes.py --write
python code\control_surface_knowledge_eighth_wave_outcomes.py
python code\validate_control_surface_atlas.py
```

V2 build:

- added `code/control_surface_comparison.py`;
- writes `data/control_surface_comparison.json`;
- aggregates verdicts, lead-time states, intervention states, behavior-gate
  states, row diagnostics, artifact diagnostics, mixture-axis counts,
  null-boundary signals, artifact-row summaries, row summaries, and current
  genome-shape ratios;
- treats the comparison as a checked-in generated artifact, not an ad hoc
  notebook output;
- integrates comparison staleness checks into
  `code/validate_control_surface_atlas.py`;
- adds a generated `claim_audit` section and makes validation fail if any atlas
  row lacks linked result artifacts, extracted metrics, a family-level claim
  check, or full row-level metric coverage;
- adds a generated `claim_consistency` section and makes validation fail if
  generic verdict, intervention-state, lead-time, null-locality, behavior-gate,
  output-confound, or signature-causal consistency rules are contradicted;
- records the current shape: 0 promoted mechanism cards, 1 bounded mechanism
  card, 17 diagnostic notes, 1 failed mechanism-card route, 0.315789
  output-margin-confounded row ratio, 0.368421 behavior-substrate-failed row
  ratio, and 0.736842 intervention not-allowed-or-failed ratio.

Law-audit build:

- added `code/control_surface_law_audit.py`;
- writes `data/control_surface_law_audit.json`;
- validates every law hypothesis against cited atlas rows and observed
  diagnostics;
- fails validation if a checked-in law audit is stale, if a hypothesis has
  evidence gaps, if a law cites an unobserved diagnostic, or if an atlas row has
  no law support;
- downgraded `coarse_source_ablation_overstates_circuit_locality` from
  `supported_pattern` to `tentative_pattern` because the current evidence is
  single-row;
- removed `REQUESTED_MODE_CONFUND` from current law evidence because it is in
  the diagnostic vocabulary but not observed in any current atlas row.

Next-queue build:

- added `code/control_surface_next_queue.py`;
- writes `data/control_surface_next_experiment_queue.json`;
- compiles law-hypothesis `next_tests` into a scored queue;
- ranks tests by tentative/single-row evidence, bridge-route pressure, transfer
  gaps, intervention relevance, lead-time relevance, output/candidate-control
  relevance, behavior-substrate gates, and the current 0-promoted-card state;
- ingests bridge-ladder, route-disposition, and error-taxonomy artifacts as
  closure context, so bridge-route queue items carry active constraints from
  MC030-MC033 instead of reopening local guard/order/schema/checksum,
  cross-table, or fact-claim variants;
- fails validation if the queue is stale, empty relative to the law set, missing
  top queue ids, has no immediate/high-priority item, lacks bridge-closure
  context, has any hidden-state-allowed bridge rung, loses the MC030-MC033
  recent-closure set, or emits bridge-route items without active constraints.

Smoke-diagnostic build:

- added `code/control_surface_smoke_diagnostics.py`;
- writes `data/control_surface_smoke_diagnostics.json`;
- writes `research/26_CONTROL_SURFACE_SMOKE_DIAGNOSTICS.md`;
- extracts MC017-MC033 smoke behavior artifacts into a typed failure ledger
  without promoting them into atlas rows;
- records 17 smoke cards, 17 structural-passed cards, 0 behavior-ready cards, 0
  signature-ready cards, and 0 hidden-state-allowed cards;
- validates 72 core assertions, including MC017 atomic-selector collapse, MC018
  first-listed-choice pressure, MC019 expected-atomic route weakness with clean
  nulls, MC020 direct atomic recall surviving query-row pressure while the
  route-atomic branch fails, MC021 visible-route instability, and MC022
  semantic-source-label rule-order sensitivity, plus MC023 query-operation
  controls-clean conflict failure and MC024 few-shot operation local-branch
  repair with learned atomic-branch failure, plus MC025 constrained-choice
  atomic-control, null, and atomic-branch failure, plus MC026 numeric-option
  null repair with atomic-control abstention and weak learned-branch routing,
  plus MC027 answer-interface dispersion with only the bare-integer interface
  surviving 10-source smoke, plus MC028 bare-integer full-source learned-branch
  failure with other-number leakage despite clean controls and nulls, plus
  MC029 factorized operation-leak tradeoff where `rules_only` improves the
  learned atomic branch but fails answer-absent null reliability, plus MC030
  absence-guard repair failure where direct guards worsen nulls or collapse the
  learned atomic branch, plus MC031 statusless checksum failure where invalid
  rows collapse to local answers despite clean controls and nulls, plus MC032
  cross-table consistency failure where mismatch rows never select atomic/lure
  values and do not copy the side number, plus MC033 fact-claim closeout where
  controls and nulls stay clean while match and mismatch branches fail stable
  routing;
- integrates smoke-diagnostic staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Bridge-ladder build:

- added `code/control_surface_bridge_ladder.py`;
- writes `data/control_surface_bridge_ladder.json`;
- writes `research/27_CONTROL_SURFACE_BRIDGE_LADDER.md`;
- combines validated MC010-MC016 atlas rows with MC017-MC033 smoke diagnostics
  into one cumulative bridge map;
- records 24 rungs, 7 atlas rungs, 17 smoke rungs, 1 behavior-ready rung, 0
  signature-ready rungs, 0 hidden-state-allowed rungs, and 0 clean
  unconfounded bridge rungs;
- validates that MC012 is exactly the prompt-visible positive control, MC013
  kills that contrast under status ablation, MC016 still collapses a visible
  non-status gate, MC020 shows atomic recall is not the bottleneck, and MC022
  shows semantic labels are still not enough, while MC023 shows query-level
  operation handles clean controls without clearing conflict gates and MC024
  shows worked examples repair local routing without rescuing learned atomic
  routing, while MC025 shows constrained choices introduce new control/null
  failures, and MC026 shows numeric options preserve nulls while turning direct
  atomic recall into UNKNOWN, and MC027 shows answer schemas are behavior
  surfaces rather than neutral wrappers, while MC028 shows the bare-integer
  survivor does not promote under full-source learned atomic routing, and
  MC029 shows factorized prompt edits move branch, null, local-row, and
  example-copying axes without producing a clean bridge, while MC030 closes
  simple absence-guard repair for that branch/null tradeoff, MC031 closes
  the first statusless checksum cue as invalid-branch local collapse, and
  MC032 shows cross-table consistency repeats the statusless local-dominance
  boundary, while MC033 closes the fact-claim route as branch instability
  rather than a clean learned-fact arbitration surface;
- integrates bridge-ladder staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Mixture-law build:

- added `code/control_surface_mixture_law.py`;
- writes `data/control_surface_mixture_law.json`;
- writes `research/28_CONTROL_SURFACE_MIXTURE_LAW.md`;
- classifies every atlas row into non-exclusive pressure classes and exactly
  one primary blocker;
- records current primary blockers as 11 behavior-substrate-or-bridge-blocked
  rows, 6 output-geometry-shadow rows, 1 prompt-visible positive control, and
  1 bounded internal-causal row;
- records current pressure classes as prompt-contract-visible in 19/19 rows,
  output-geometry-visible in 14/19, source-or-prompt-token-dependent in 13/19,
  internal-monitor-present in 5/19, internal-causal in 1/19, and transfer
  unproven-or-failed in 3/19;
- validates that primary blockers cover every row exactly once, that prompt
  contract is universal current pressure, that 0 promoted mechanism cards are
  preserved, that MC005 is the only bounded internal-causal row, and that the
  bridge ladder has 0 clean unconfounded hidden-state candidates;
- integrates mixture-law staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Decision-frontier build:

- added `code/control_surface_decision_frontier.py`;
- writes `data/control_surface_decision_frontier.json`;
- writes `research/29_CONTROL_SURFACE_DECISION_FRONTIER.md`;
- classifies every atlas row by commitment-timing status:
  `frontier_not_reached`, `output_visible_at_or_before_frontier`,
  `predecision_monitor_no_lever`, `predecision_causal_candidate`, or
  `causal_surface_not_timing_frontier`;
- records the current frontier as 12 unreached rows, 4 output-visible or
  zero-lead rows, 2 monitor-only rows, 1 causal-not-timing row, and 0
  predecision causal candidates;
- validates that the frontier counts match comparison lead-time counts, that
  MC004 and MC006 are exactly the monitor-only rows, that output-visible
  frontier rows outnumber monitor-only rows, and that MC005 is not silently
  counted as a lead-time promotion;
- integrates decision-frontier staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Route-disposition build:

- added `code/control_surface_route_disposition.py`;
- writes `data/control_surface_route_disposition.json`;
- writes `research/30_CONTROL_SURFACE_ROUTE_DISPOSITION.md`;
- classifies every atlas row as bounded frozen, closed before hidden-state
  work, failed intervention/mechanism route, monitor-only closed, monitor-only
  conditional revisit, output-shadow diagnostic baseline, prompt-visible
  positive control, or diagnostic baseline;
- records 0 hidden-state-ready atlas routes;
- records atlas dispositions as 11 closed-before-hidden rows, 3 output-shadow
  baselines, 1 failed intervention/mechanism route, 1 monitor-only closed row,
  1 monitor-only conditional-revisit row, 1 prompt-visible positive control,
  and 1 bounded frozen mechanism;
- records bridge dispositions as 23 bridge rungs closed before hidden-state
  work and 1 prompt-visible positive control;
- validates that MC005 is bounded frozen, MC006 is monitor-only closed, MC012
  is prompt-visible positive control, no active hidden-state route exists, and
  no bridge rung allows hidden-state work;
- integrates route-disposition staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Transfer-matrix build:

- added `code/control_surface_transfer_matrix.py`;
- writes `data/control_surface_transfer_matrix.json`;
- writes `research/31_CONTROL_SURFACE_TRANSFER_MATRIX.md`;
- classifies every atlas row by transfer/widening status, separating bounded
  transfer-fragile reference rows, failed or bank-insufficient transfer routes,
  diagnostic cross-model evidence, output-shadow widening baselines, closed
  no-widening rows, and prompt-visible rows with no transfer claim;
- records 0 transfer-ready mechanisms;
- records transfer values as 14 untested rows, 2 low-transfer rows, 2
  medium-transfer rows, and 1 failed-transfer row;
- validates that transfer values match the comparison artifact, that MC005 is
  the bounded transfer-fragile reference, that MC006 is the failed transfer
  route, that untested transfer remains the majority, and that transfer-gap
  queue pressure still exists;
- integrates transfer-matrix staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Reliability-matrix build:

- added `code/control_surface_reliability_matrix.py`;
- writes `data/control_surface_reliability_matrix.json`;
- writes `research/32_CONTROL_SURFACE_RELIABILITY_MATRIX.md`;
- classifies every atlas row across mechanism-card reliability gates:
  behavior, signature, intervention, null/locality, local internal path,
  robustness/side effects, and transfer;
- records 0 full-reliability mechanisms and 1 bounded reliability reference;
- records reliability classes as 11 behavior/bridge-blocked rows, 3
  output-shadow diagnostic rows, 2 monitor-only no-lever rows, 1
  failed-intervention route, 1 prompt-visible positive control, and 1 bounded
  reference;
- records missing gates as 19/19 rows missing clean predicted intervention,
  null/locality cleanliness, robustness/side-effect clearance, and transfer or
  widening, 18/19 rows missing a control-surviving signature and local internal
  path, and 11/19 rows missing the behavior substrate;
- validates that reliability classes derive from route dispositions, that
  MC005 is the only bounded reliability reference, that MC004 and MC006 are the
  monitor-only rows, that MC012 is the prompt-visible positive control, and
  that no row can look reliability-complete;
- integrates reliability-matrix staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Error-taxonomy build:

- added `code/control_surface_error_taxonomy.py`;
- writes `data/control_surface_error_taxonomy.json`;
- writes `research/33_CONTROL_SURFACE_ERROR_TAXONOMY.md`;
- reads the smoke diagnostics, bridge ladder, route-disposition ledger, and
  MC028, MC029, MC030, MC031, MC032, and MC033 behavior artifacts;
- records 17 smoke cards, 24 bridge rungs, 0 hidden-state-allowed smoke cards,
  and 23 bridge rungs closed before hidden-state work;
- validates MC028 as a learned-branch other-number leak rather than a direct
  atomic-control or null failure: operation-atomic rows had 28/160 wrong
  numbers while direct atomic control, answer-absent nulls, and rule-absent
  nulls were each 160/160 clean;
- bucketizes those 28 wrong numbers into 9 worked-example outputs, 8 other
  bank atomic numbers, 6 off-bank other numbers, 3 bank local numbers, and 2
  prompt-local row numbers;
- validates MC029 as a factorized branch/null/example-leak tradeoff: no variant
  passes, `rules_only` reaches 0.875 operation-atomic atomic but only 0.806
  answer-absent UNKNOWN, and `query_before_examples` reaches 54 worked-example
  other-number rows;
- validates MC030 as an absence-guard repair failure: no variant passes, the
  row-absence guard drives answer-absent UNKNOWN to 0.231, and query-last
  reduces other-number leakage to 0.056 only while dropping operation-atomic
  atomic to 0.487;
- validates MC031 as a statusless checksum reliability-cue failure: direct
  controls and nulls stay clean, valid-checksum rows select local numbers, and
  invalid-checksum rows still select local numbers with 0.000 atomic/lure
  selection;
- validates MC032 as a statusless cross-table consistency failure: direct
  controls and nulls stay clean, mismatch rows select atomic/lure values 0.000
  of the time, mostly select the primary local row, and never copy the
  second-table side number;
- validates MC033 as a fact-claim closeout failure: direct controls and nulls
  stay clean, match rows return local only 0.400 and learned atomic 0.500 of
  the time, and mismatch rows select learned atomic only 0.100 while leaking
  local and wrong claimed/lure numbers;
- integrates error-taxonomy staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Gate-geometry build:

- added `code/control_surface_gate_geometry.py`;
- writes `data/control_surface_gate_geometry.json`;
- writes `research/34_CONTROL_SURFACE_GATE_GEOMETRY.md`;
- reads the route-disposition, reliability-matrix, decision-frontier,
  bridge-ladder, and error-taxonomy layers;
- turns failed controls into an ordered claim-bar geometry: 11 rows die at
  behavior/bridge substrate, 1 at prompt-channel locality, 3 at output-shadow
  signature, 2 at monitor-only no-lever signature, 1 at failed intervention,
  1 at bounded reliability, and 0 at promoted mechanism;
- validates that the row funnel partitions all 19 atlas rows, that the bridge
  funnel partitions all 24 bridge rungs, that MC005 is the only reliability
  boundary, MC012 is the prompt-channel boundary, MC001G is the intervention
  failure boundary, and bridge rungs still have no signature/intervention
  stage;
- integrates gate-geometry staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Genome-snapshot build:

- added `code/control_surface_genome_snapshot.py`;
- writes `data/control_surface_genome_snapshot.json`;
- writes `research/35_CONTROL_SURFACE_GENOME_SNAPSHOT.md`;
- fuses atlas, artifact index, comparison, law audit, next queue, smoke
  diagnostics, bridge ladder, mixture law, decision frontier, route
  disposition, transfer matrix, reliability matrix, error taxonomy, and gate
  geometry into one compact current-state artifact;
- records the current global claim state: 19 atlas rows, 53 linked artifacts,
  1 bounded mechanism card, 18 diagnostic-or-failed rows, 0 promoted
  mechanisms, 12 rows blocked before signature, 5 signature-stage blocks, 1
  failed intervention, 1 bounded reliability specimen, 0 transfer-ready
  mechanisms, and 0 hidden-state-allowed bridge rungs;
- validates that the snapshot preserves the gate partition, reports no
  promoted mechanisms, keeps MC005 as the only bounded reference, keeps
  MC030-MC033 as the recent bridge closure set, and emits global allowed and
  forbidden claims together;
- integrates genome-snapshot staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Axis-interactions build:

- added `code/control_surface_axis_interactions.py`;
- writes `data/control_surface_axis_interactions.json`;
- writes `research/36_CONTROL_SURFACE_AXIS_INTERACTIONS.md`;
- fuses atlas rows with mixture law, gate geometry, route disposition,
  decision frontier, reliability matrix, and genome snapshot;
- emits 123 feature summaries across diagnostics, frontier classes, mixture
  axes, pressure classes, primary blockers, reliability classes, and route
  dispositions;
- records 14 pure predictive rules and 31 mixed predictors, with every feature
  carrying row count, evidence level, dominant terminal stage, purity, and row
  membership;
- validates that row-feature entries cover all atlas rows, route-disposition
  rules preserve the main pre-signature and output-shadow regularities,
  `closed_before_hidden_state` predicts `pre_signature_behavior_substrate`
  across 11 rows, output-geometry pressure remains mixed rather than a single
  terminal-stage rule, and singleton features stay labeled as singleton
  evidence;
- integrates axis-interactions staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Coverage-gaps build:

- added `code/control_surface_coverage_gaps.py`;
- writes `data/control_surface_coverage_gaps.json`;
- writes `research/37_CONTROL_SURFACE_COVERAGE_GAPS.md`;
- reads atlas, genome snapshot, axis interactions, bridge ladder, next queue,
  reliability matrix, and transfer matrix;
- emits 12 named gaps across mechanism promotion, reliability, intervention,
  transfer, bridge substrate, predictive-law support, sample size, model
  coverage, behavior-domain coverage, mechanism reference, and knowledge
  mechanism status;
- records 4 critical gaps: `promoted_mechanism_absent`,
  `full_reliability_absent`, `clean_intervention_absent`, and
  `transfer_ready_mechanism_absent`;
- validates that the coverage map keeps the 14 transfer-untested rows, 0
  bridge hidden-state-allowed rungs, at least 80 singleton/sparse feature
  summaries, and nonempty next pressures for every gap;
- integrates coverage-gap staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Gap-closure-plan build:

- added `code/control_surface_gap_closure_plan.py`;
- writes `data/control_surface_gap_closure_plan.json`;
- writes `research/38_CONTROL_SURFACE_GAP_CLOSURE_PLAN.md`;
- reads coverage gaps, next queue, genome snapshot, reliability matrix,
  transfer matrix, bridge ladder, and axis interactions;
- emits 6 decision-bound work orders: MC005 reference closeout, post-MC033
  bridge family closeout, MC006 predecision frontier closeout, width transfer
  probe, singleton-stage law replication, and the offensive-doctrine harness;
- validates that 12/12 coverage gaps, 4/4 critical gaps, and 5/5 top queue
  items are covered by work orders with promotion, bounded-claim, kill,
  containment, and export rules;
- integrates gap-closure-plan staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Offensive-doctrine build:

- added `code/control_surface_offensive_doctrine.py`;
- writes `data/control_surface_offensive_doctrine.json`;
- writes `research/39_CONTROL_SURFACE_OFFENSIVE_DOCTRINE.md`;
- reads coverage gaps, gap-closure plan, genome snapshot, and next queue;
- emits 6 branch contracts, one for each current gap-closure work order;
- requires every future branch contract to name target gaps, expected
  generated-layer movement, minimum evidence, promotion, bound, kill,
  containment, and export rules;
- validates that all work orders have contracts, all contracts have kill
  rules, every contract names generated-layer movement, critical gaps have
  active contracts, and the harness does not imply a promoted mechanism or
  full-reliability row;
- integrates offensive-doctrine staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

MC005-reference-specimen build:

- added `code/mc005_reference_specimen_audit.py`;
- writes `data/mc005_reference_specimen_audit.json`;
- writes `research/cards/MC005_REFERENCE_SPECIMEN_AUDIT.md`;
- reads the atlas, offensive doctrine, route disposition, reliability matrix,
  transfer matrix, gate geometry, and the canonical MC005 write-replacement
  closeout result;
- targets the `close_mc005_reference_specimen` work order;
- records MC005 as the validator-backed bounded reference specimen: verdict
  `bounded_mechanism_card`, route status `bounded_frozen_not_promoted`,
  terminal stage `reliability_null_boundary`, reliability class
  `bounded_reliability_reference`, and transfer class
  `bounded_transfer_fragile_reference`;
- validates the positive control surface: V29 attention-write replacement has
  1.0 delta recovery and 1.0 target-win-loss recovery versus direct source
  masking, with write source controls passing;
- validates the boundary: strict answer-absent nulls are not clean, the null
  boundary is reproduced, V31 has 5 combined null flips, and the strict 0.5
  absolute-margin explanation fails;
- records the future-work admission rule: no same-route repair loop is
  allowed; future MC005 work must be a materially new intervention family, a
  matched transfer panel, or a comparative baseline for another family;
- integrates MC005-reference staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

MC006-predecision-frontier build:

- added `code/mc006_predecision_frontier_audit.py`;
- writes `data/mc006_predecision_frontier_audit.json`;
- writes `research/cards/MC006_PREDECISION_FRONTIER_AUDIT.md`;
- reads the atlas, offensive doctrine, decision frontier, route disposition,
  reliability matrix, transfer matrix, gate geometry, and the canonical MC006
  predecision-frontier closeout result;
- targets the `close_mc006_predecision_frontier` work order;
- records MC006 as the validator-backed knowledge-like monitor route: verdict
  `diagnostic_note`, route status `monitor_only_closed`, frontier class
  `predecision_monitor_no_lever`, terminal stage `signature_monitor_no_lever`,
  reliability class `not_reliable_monitor_only_no_lever`, and transfer class
  `transfer_failed_or_bank_insufficient`;
- validates the positive timing result: V14 has a matched generated behavior
  substrate with 30 selected binary rows and both labels on holdout, while V16
  has an `after_mapping_line` layer-4 monitor with 1.0 holdout AUC that beats
  same-position output;
- validates the boundary: the V16 monitor does not beat candidate score or
  final next-token output, final/candidate geometry blocks promotion, V17
  intervention fails, V25 fails shuffled-label selected-search nulls, and
  V26-V28 close the transfer-bank route;
- records the future-work admission rule: no ordinary delayed-city prompt
  repair is allowed; future MC006 work must be a known-confounded causal stress
  test labeled as such or a materially new behavior family;
- integrates MC006-predecision staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Compositional-genome audit build:

- added `code/control_surface_compositional_genome_audit.py`;
- writes `data/control_surface_compositional_genome_audit.json`;
- writes `research/40_CONTROL_SURFACE_COMPOSITIONAL_GENOME_AUDIT.md`;
- reads the atlas, mixture law, decision frontier, reliability matrix, transfer
  matrix, MC005 reference audit, MC006 predecision audit, and post-MC033 bridge
  closeout;
- records the reviewer insight as a generated artifact: the current
  genome-level object is the measured distribution across prompt contract,
  output geometry, source/prompt-token dependence, behavior/bridge substrate
  blockage, internal monitors, bounded causal surfaces, reliability, and
  transfer;
- validates the current ratios directly: 19/19 prompt-contract-visible rows,
  14/19 output-geometry-visible rows, 13/19 source-or-prompt-token-dependent
  rows, 11/19 behavior-or-bridge-substrate-blocked rows, 5/19 internal-monitor
  rows, 1/19 internal-causal rows, 0 full-reliability mechanisms, 0
  transfer-ready mechanisms, and 0 clean unconfounded bridge substrates;
- anchors the map to MC005 as the bounded internal-causal reference, MC006 as
  the knowledge-like monitor-only route, the post-MC033 bridge sequence as the
  substrate death condition, and MC004/MC006 as the current monitor-only
  lead-time frontier;
- records the claim boundary: the audit supports a compositional
  control-surface map, not a promoted mechanism card, general truth vector, or
  general knowledge-control surface;
- integrates compositional-genome audit staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Family-matrix build:

- added `code/control_surface_family_matrix.py`;
- writes `data/control_surface_family_matrix.json`;
- writes `research/41_CONTROL_SURFACE_FAMILY_MATRIX.md`;
- joins each atlas row to mixture-law, decision-frontier, route-disposition,
  reliability-matrix, transfer-matrix, and gate-geometry cells;
- emits the compact cross-family table the project needs for cumulative work:
  row id, behavior domain, model list, behavior gate, verdict, primary blocker,
  terminal stage, route disposition, frontier class, lead-time state,
  intervention state, reliability class, transfer class/value, prompt/output/
  source/internal/null/transfer boolean axes, failed gates, claim-bar action,
  and next decision;
- validates that the matrix covers all 19 atlas rows exactly once, preserves
  the mixture-law primary blockers, preserves the decision-frontier,
  reliability, transfer, and gate-geometry counts, keeps MC005 as the sole
  bounded reference, keeps MC004/MC006 as the monitor-only rows, and reports 0
  promotion-ready rows;
- integrates family-matrix staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Knowledge-ladder build:

- added `code/control_surface_knowledge_ladder.py`;
- writes `data/control_surface_knowledge_ladder.json`;
- writes `research/42_CONTROL_SURFACE_KNOWLEDGE_LADDER.md`;
- reads the family matrix, bridge ladder, MC005 reference audit, MC006
  predecision audit, and post-MC033 bridge closeout;
- separates five knowledge-ladder levels: pure synthetic lookup, semi-synthetic
  familiar entities, symbolic/learned-memory bridge, strong parametric-fact
  override, and real abstention/uncertainty;
- keeps MC001 truth/agreement rows, MC003 delayed copy, and MC004 in-context
  binding as auxiliary diagnostics rather than counting them as knowledge
  ladder levels;
- validates that ladder rows plus auxiliary rows partition all 19 family-matrix
  rows, that MC005 is the level-1 bounded reference, MC007 remains behavior-
  substrate-blocked at level 2, MC008-MC016 plus the 24-rung bridge ladder
  remain hidden-state-disallowed at level 3, MC006 is monitor-only at level 4,
  MC002/MC002B are not mechanism-ready at level 5, and no ladder level is
  promoted;
- integrates knowledge-ladder staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Knowledge-gap-plan build:

- added `code/control_surface_knowledge_gap_plan.py`;
- writes `data/control_surface_knowledge_gap_plan.json`;
- writes `research/43_CONTROL_SURFACE_KNOWLEDGE_GAP_PLAN.md`;
- reads the knowledge ladder, family matrix, gap-closure plan, and offensive
  doctrine;
- records 15 missing-evidence items across the five knowledge-ladder levels;
- links existing licensed work to `close_mc005_reference_specimen`,
  `run_width_transfer_probe`, `close_post_mc033_bridge_substrate_family`, and
  `close_mc006_predecision_frontier`;
- marks three levels as requiring new behavior substrates or future work
  orders before progress beyond the current map;
- validates that no new hidden-state search is licensed, MC005 is routed to
  reliability/transfer closure, the bridge route remains killed, MC006 remains
  monitor-only, real uncertainty remains below the behavior gate, and broad
  truth/knowledge/factual-correction claims remain forbidden;
- integrates knowledge-gap-plan staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Knowledge-substrate-admission build:

- added `code/control_surface_knowledge_substrate_admission.py`;
- writes `data/control_surface_knowledge_substrate_admission.json`;
- writes `research/44_CONTROL_SURFACE_KNOWLEDGE_SUBSTRATE_ADMISSION.md`;
- reads the knowledge gap plan, bridge ladder, smoke diagnostics, and gate
  geometry;
- creates three admission packets: semi-synthetic familiar entities,
  symbolic/learned-memory bridge, and real abstention/uncertainty;
- requires 11 front-door gates before hidden-state work: material novelty,
  behavior contract, parseability and label balance, direct controls, conflict
  mixture, null rows, source-disjoint holdout, prompt-channel locality,
  output/candidate baselines, side-effect/leakage checks, and split freeze;
- validates that the packets cover exactly the levels requiring new behavior
  substrates, that no new hidden-state search is licensed, that the bridge route
  remains killed unless a proposal is outside MC007-MC033, and that real
  uncertainty proposals require grounded labels and abstention controls;
- integrates knowledge-substrate-admission staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Knowledge-candidate-queue build:

- added `code/control_surface_knowledge_candidate_queue.py`;
- writes `data/control_surface_knowledge_candidate_queue.json`;
- writes `research/45_CONTROL_SURFACE_KNOWLEDGE_CANDIDATE_QUEUE.md`;
- reads the knowledge-substrate admission protocol and next experiment queue;
- creates six behavior-only candidate substrates: two for semi-synthetic
  familiar entities, two for the symbolic/learned-memory bridge, and two for
  real abstention/uncertainty;
- binds every candidate to all 11 admission gates, first-run panels, dumb
  explanations, and promote/death/containment/export rules;
- validates that every admission packet has candidate coverage, each admission
  class has two candidates, no candidate licenses hidden-state work, and the
  queue operationalizes existing next-queue items;
- integrates knowledge-candidate-queue staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Knowledge-first-run-pack build:

- added `code/control_surface_knowledge_first_run_pack.py`;
- writes `data/control_surface_knowledge_first_run_pack.json`;
- writes `research/46_CONTROL_SURFACE_KNOWLEDGE_FIRST_RUN_PACK.md`;
- reads the knowledge-candidate queue;
- creates six behavior-substrate first-run packets, one for every queue
  candidate;
- records 37 total behavior panels, 24 baseline checks, 66 admission-gate
  bindings, frozen-before-run fields, model targets, artifact paths, and
  promote/death/containment/export rules;
- validates that every packet has null, holdout, and output-geometry coverage,
  every packet binds all 11 admission gates, no packet licenses hidden-state
  work, and every packet names prereg, runner, result, and status-card paths;
- integrates knowledge-first-run-pack staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

KSQ001 familiar-entity prior-counterbalance run:

- added `code/ksq001_familiar_entity_prior_counterbalance_first_run.py`;
- writes
  `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_first_run.json`;
- writes
  `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_smoke_limit10.json`;
- writes
  `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_full_behavior.json`;
- writes
  `research/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN_STATUS.md`;
- writes
  `research/prereg/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN.md`;
- executes the highest-priority semi-synthetic familiar-entity candidate from
  the first-run pack as a behavior-substrate admission test, not a hidden-state
  search;
- builds a 480-row structural substrate from 40 familiar country entities, 4
  panels, 3 templates, and discovery/calibration/source-disjoint holdout splits;
- validates that artificial task-city values do not collide with real capitals
  or lures, real capitals are not prompt-listed in conflict rows, prompt-local
  artificial values appear only where intended, status/authority lexemes are
  absent, and hidden-state license remains false;
- 10-source smoke result: selected `compact_question`; local artificial direct
  control passed (`10/10` artificial), semantic-prior direct control passed
  (`10/10` real prior), nulls passed (`10/10` unknown), and the primary
  familiar-entity conflict showed a small mixture (`6/10` artificial, `1/10`
  real prior, `2/10` unknown, `1/10` unparsed);
- full 40-source result: direct controls and nulls still passed
  (`40/40` artificial local control, `38/40` real-prior direct control with one
  lure and one unparsed, `38/40` unknown nulls with two unparsed), candidate and
  output margins were reported, and source-disjoint holdout mixture existed;
- full behavior admission still failed because the selected conflict template
  had only `24/40` parseable primary rows: `18/40` artificial, `2/40` real
  prior, `4/40` unknown, and `16/40` unparsed;
- typed boundary: familiar-entity priors and prompt-local artificial values can
  both be separately controlled, and a small conflict mixture exists, but the
  mixture/parseability tradeoff blocks a behavior-ready substrate at full scale;
- integrates JSON-only KSQ001 structural, smoke, and full-behavior checks into
  `code/validate_control_surface_atlas.py`;
- still licenses no hidden-state signatures, interventions, or
  knowledge-control-surface claim.

KSQ002 familiar-entity source-rewrite equivalence run:

- added `code/ksq002_familiar_entity_source_rewrite_equivalence_first_run.py`;
- writes
  `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_first_run.json`;
- writes
  `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_smoke_limit10.json`;
- writes
  `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_full_behavior.json`;
- writes
  `research/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN_STATUS.md`;
- writes
  `research/prereg/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN.md`;
- executes the source-rewrite equivalence packet from the first-run pack as a
  behavior-substrate admission test, not a hidden-state search;
- builds a 720-row structural substrate from 40 familiar country entities, 6
  panels, 3 templates, and discovery/calibration/source-disjoint holdout splits;
- validates baseline source-value lookup, neutral source rewrite, source
  deletion, query-only control, source-disjoint rewrite holdout, and rewrite
  output-geometry audit panels while keeping hidden-state license false;
- 10-source smoke result: selected `registry_question`; baseline lookup,
  neutral rewrite lookup, source deletion, query-only control,
  source-disjoint rewrite holdout, and candidate/output margin reporting all
  passed, with all rewrite and baseline rows returning artificial values and
  all deletion/query-only rows returning `UNKNOWN`;
- full 40-source result: selected `sentence_rewrite`; baseline source lookup
  returned `39/40` artificial values, neutral rewrite returned `36/40`
  artificial values with `4/40` unparsed rows, and source deletion plus
  query-only controls were both `40/40` unknown;
- full behavior admission still failed because the source-disjoint rewrite
  holdout was only `14/16` parseable/artificial and `2/16` unparsed, yielding
  `0.875` parseability and artificial-value rates against the predeclared
  `0.900` gate;
- candidate and output margins were reported, and the selected rewrite delta
  from baseline was `0.07499999999999996`;
- typed boundary: source rewrite is mostly robust and source-local under the
  best full template, and deletion/query-only controls kill the simplest
  source-presence explanation, but the full source-disjoint rewrite holdout is
  not parseability-stable enough to admit hidden-state work;
- integrates JSON-only KSQ002 structural, smoke, and full-behavior checks into
  `code/validate_control_surface_atlas.py`;
- still licenses no hidden-state signatures, interventions, source-channel
  mechanism claim, or knowledge-control-surface claim.

KSQ003 statusless-evidence first-run structural gate:

- added `code/ksq003_bridge_statusless_evidence_aggregation_first_run.py`;
- writes
  `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_first_run.json`;
- writes
  `research/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN_STATUS.md`;
- writes
  `research/prereg/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN.md`;
- executes the highest-priority knowledge candidate from the first-run pack as
  a behavior-substrate admission test, not a hidden-state search;
- builds 840 rows from 40 sources, 7 panels, and 3 templates with discovery,
  calibration, and source-disjoint holdout splits;
- tests a statusless symbolic/learned-memory bridge where branch selection
  depends on chemical-symbol and atomic-parity evidence about the queried
  element rather than source status labels, checksums, equations, row codes, or
  row-local atomic-number claims;
- includes local-number direct controls, learned atomic-number direct controls,
  all-evidence-fit conflicts, one-evidence-mismatch conflicts, symbol-only
  ablations, parity-only ablations, and answer-absent nulls;
- passed the structural gate after removing accidental numeric note labels that
  made the audit treat `Evidence note 1/2` as atomic-number leakage;
- validates that conflict prompts hide the real and lure atomic numbers, local
  numbers are not used as evidence, answer-absent rows omit the query local
  number, candidates are parseable and collision-free, primary prompts avoid
  status/closed-route lexemes, and primary branch labels are balanced;
- ran a contained 10-source behavior smoke to
  `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_smoke_limit10.json`;
- the selected `compact_fit` template passed local-number direct control
  (`10/10` local), learned atomic-number direct control (`10/10` atomic),
  all-evidence-fit conflict routing (`10/10` local), and answer-absent nulls
  (`10/10` unknown);
- the same smoke failed the knowledge-bridge behavior gate because the
  one-evidence-mismatch branch returned local numbers (`10/10` local), and both
  symbol-only and parity-only ablations also returned local numbers (`10/10`
  local each) instead of `UNKNOWN`;
- typed boundary: statusless symbol/parity evidence is not yet strong enough to
  overcome local-table dominance, even though direct learned-memory recall and
  null behavior are separately available;
- integrates JSON-only KSQ003 structural and smoke-result checks into
  `code/validate_control_surface_atlas.py`;
- still licenses no hidden-state signatures, interventions, or
  knowledge-control-surface claim.

KSQ003 evidence-sufficiency redesign:

- added `code/ksq003_evidence_sufficiency_redesign.py`;
- writes
  `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_first_run.json`;
- writes
  `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_smoke_limit10.json`;
- writes
  `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_full_behavior.json`;
- writes
  `research/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN_STATUS.md`;
- writes
  `research/prereg/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN.md`;
- executes the medium-priority `redesign_statusless_bridge_substrate` work
  order as a material redesign of the first KSQ003 route, not a template tweak;
- changes the branch contract from symbol/parity fit-versus-mismatch to
  evidence sufficiency: complete identity evidence should route to the learned
  standard atomic number, contradictory or one-note evidence should route to
  `UNKNOWN`, and the local lab table remains a visible distractor/direct
  control rather than the positive conflict branch;
- structural gate passed on 840 rows from 40 sources, 7 panels, 3 templates,
  discovery/calibration/source-disjoint holdout splits, hidden real/lure atomic
  numbers in conflicts, no local number evidence leak, no status lexemes, no
  checksum/row-code/fact-claim route cues, parseable candidates, and one answer
  suffix;
- 10-source smoke already showed the new boundary: the original local-table
  collapse mostly disappeared, but templates split between learned-branch
  activation and null-stress rejection;
- full behavior result selected `compact_identity`: direct local lookup was
  `40/40` local, direct learned recall was `40/40` atomic, answer-absent nulls
  were `38/40` `UNKNOWN`, complete identity conflict was only `13/40` atomic
  and `27/40` `UNKNOWN`, and null stress was 66/120 `UNKNOWN` with 41/120
  atomic answers;
- cross-template boundary: `identity_packet` got `40/40` complete-evidence
  atomic answers and `8/8` holdout complete-evidence atomic answers, but null
  stress was only `1/120` `UNKNOWN`; `compact_identity` raised null-stress
  `UNKNOWN` to `66/120` and removed complete-conflict local answers, but
  dropped complete-evidence atomic answers to `13/40`;
- typed boundary: this is not a repeat of simple local-table dominance. It is
  `STATUSLESS_EVIDENCE_SUFFICIENCY_BOUNDARY`: statusless evidence can suppress
  the local table or activate learned recall, but this design cannot do both
  reliably;
- remains behavior-not-ready, signature-screen-disallowed,
  hidden-state-disallowed, intervention-disallowed, and mechanism-claim
  disallowed.

KSQ004 bridge answer-interface minimal-pairs first-run gate:

- added `code/ksq004_bridge_answer_interface_minimal_pairs_first_run.py`;
- writes
  `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_first_run.json`;
- writes
  `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_smoke_limit10.json`;
- writes
  `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_full_behavior.json`;
- writes
  `research/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN_STATUS.md`;
- writes
  `research/prereg/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN.md`;
- executes the answer-interface shortcut test from the first-run pack as a
  behavior-substrate admission run, not a hidden-state search;
- builds a 720-row structural substrate from 40 element sources, 5 panels, 2
  templates, 9 subtypes, and discovery/calibration/source-disjoint holdout
  splits;
- tests matched local-table and standard-atomic branches under the same bare
  numeric answer interface, with side-answer leakage rows and answer-absent
  null rows;
- structural gate passed: local/atomic expected labels are balanced, candidate
  answers are collision-free, primary conflict prompts hide the real atomic
  answer, side panels contain exactly one side number, answer-absent prompts
  omit the query local number, the response suffix is shared, and hidden-state
  license remains false;
- 10-source smoke result: selected `question_form` and passed every smoke
  behavior criterion (`19/20` conflict rows expected-correct, `0/20` side
  answer selections, `10/10` answer-absent UNKNOWN, margins reported);
- full behavior result: canonical selected `compact_form` failed with
  `bridge_minimal_pair_contrast_absent`; the selected full conflict panel
  returned `70/80` local numbers and only `9/80` atomic numbers, while the
  atomic conflict subtype selected atomic only `9/40` times;
- the important non-flat result is template fragility: `question_form` retained
  the bridge much better (`0.850` conflict expected-correct, `0.700`
  atomic-branch atomic rate, `0.875` holdout expected-correct), while
  `compact_form` collapsed toward local-table lookup (`0.6125` conflict
  expected-correct, `0.225` atomic-branch atomic rate, `0.625` holdout
  expected-correct);
- candidate sequence-logprob margins were reported, but first-token numeric
  next-token margins are degenerate for this interface: the final
  local-minus-atomic logit gap is `0.0` throughout the selected summaries, so
  this run cannot use first-token next-token geometry as a meaningful
  discriminator;
- typed boundary: a matched bare numeric answer interface is not enough by
  itself to explain away the bridge, because the `question_form` branch mostly
  works, but the bridge is not robust enough for hidden-state work because the
  compact relation-key interface collapses the learned branch back to
  prompt-local table answers;
- integrates JSON-only KSQ004 structural, smoke-result, and full-behavior
  checks into `code/validate_control_surface_atlas.py`;
- still licenses no hidden-state signatures, interventions, answer-interface
  control claim, learned-memory bridge claim, or knowledge-control-surface
  claim.

KSQ005 grounded-answerability first-run gate:

- added `code/ksq005_uncertainty_grounded_answerability_first_run.py`;
- writes
  `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_first_run.json`;
- writes
  `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_smoke_limit10.json`;
- writes
  `research/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN_STATUS.md`;
- writes
  `research/prereg/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN.md`;
- executes the highest-priority real-abstention/uncertainty candidate from
  the first-run pack as a behavior-substrate admission test, not a hidden-state
  search;
- builds a 480-row structural substrate from 40 real country sources, 40 nonce
  country controls, 4 panels, 3 templates, and discovery/calibration/source-
  disjoint holdout splits;
- tests four answerability branches under a shared answer schema: ordinary
  known factual questions, unknown nonce-country rows, unsupported context rows
  where a prompt-listed city is irrelevant to the requested capital, and
  contradicted context rows where a prompt-listed false capital competes with
  familiar geography;
- structural gate passed: candidate answers are collision-free, true capitals
  are not prompt-listed, false context answers appear only in unsupported and
  contradicted panels, lure capitals are absent, status/authority lexemes are
  absent, response suffix is shared across panels, and hidden-state license
  remains false;
- 10-source smoke result: selected `reference_note`; known direct facts passed
  at the threshold (`8/10` known correct, `1/10` wrong candidate, `1/10`
  unparsed), and contradicted familiar facts passed (`5/10` corrected, `3/10`
  abstain, `1/10` false accept, `1/10` wrong answer);
- the same smoke failed the answerability behavior gate because unknown nonce
  rows reached only `4/10` abstain with `6/10` unparsed, and unsupported
  context rows reached only `4/10` abstain with `3/10` unsupported prompt-city
  answers and `3/10` unparsed;
- candidate/output margins were reported, but this does not rescue the
  behavior gate: the failure occurs before hidden-state work because
  unanswerable and unsupported rows are not parseable or abstention-stable
  enough;
- typed boundary: correction of false familiar-context claims is easier than
  grounded abstention on unknown or unsupported entities, so apparent
  correction/refusal behavior cannot be treated as evidence of real uncertainty
  control;
- integrates JSON-only KSQ005 structural and smoke-result checks into
  `code/validate_control_surface_atlas.py`;
- still licenses no hidden-state signatures, interventions, refusal-control
  claim, uncertainty-control claim, or knowledge-control-surface claim.

KSQ006 context-support counterfactual first-run gate:

- added `code/ksq006_uncertainty_context_support_counterfactuals_first_run.py`;
- writes
  `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_first_run.json`;
- writes
  `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_smoke_limit10.json`;
- writes
  `research/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN_STATUS.md`;
- writes
  `research/prereg/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN.md`;
- executes the second real-abstention/uncertainty candidate from the first-run
  pack as a behavior-substrate admission test, not a hidden-state search;
- builds a 720-row structural substrate from 40 real country sources, 5 panels,
  3 templates, 2 control subtypes, and discovery/calibration/source-disjoint
  holdout splits;
- tests support-sensitive behavior under matched supported, irrelevant,
  contradicting, insufficient, claim-only, and context-only conditions;
- structural gate passed: candidate answers are collision-free, expected city
  mentions appear only in their intended relation slots, support-word prompt
  leakage is absent, status lexemes are absent, control subtypes are balanced,
  response suffix is shared across panels, and hidden-state license remains
  false;
- 10-source smoke result: selected `field_form`; supported rows passed
  (`9/10` supported answer, `1/10` abstain), irrelevant rows passed
  (`10/10` abstain), and contradicting rows passed (`8/10` contradiction
  detected, `1/10` false accept, `1/10` true answer despite contradiction);
- the same smoke failed the context-support behavior gate because insufficient
  context rows split `5/10` abstain and `5/10` unsupported answers, and the
  claim/context-only controls reproduced supported behavior on `18/20` rows:
  claim-only rows returned the supported answer `8/10` times and context-only
  city-mention rows returned it `10/10` times;
- candidate/output margins were reported, but this does not rescue the behavior
  gate: the failure is not lack of support-word controls, it is that mere claim
  text and mere city mention reproduce the supported answer channel;
- typed boundary: context support has a stronger partial substrate than
  KSQ005's unknown/unsupported answerability rows, but it fails exactly where
  the counterfactual controls ask whether relation support is doing the work;
- integrates JSON-only KSQ006 structural and smoke-result checks into
  `code/validate_control_surface_atlas.py`;
- still licenses no hidden-state signatures, interventions, context-support
  control claim, uncertainty-control claim, or knowledge-control-surface claim.

KSQ001/KSQ002/KSQ003/KSQ004/KSQ005/KSQ006 cross-run interpretation:

- KSQ001, KSQ002, KSQ003, KSQ004, KSQ005, and KSQ006 now provide six executed
  rows from the generated knowledge first-run pack.
- They fail differently:
  - KSQ001 keeps both direct channels alive and shows weak familiar-prior
    competition, but the conflict-mixture template loses parseability at full
    scale.
  - KSQ002 keeps baseline source-value lookup, neutral rewrite lookup, source
    deletion, and query-only controls alive, but the source-disjoint rewrite
    holdout misses the full parseability gate by two unparsed rows.
  - KSQ003 keeps direct local, direct learned, and null behavior alive, but
    mismatch and incomplete-evidence branches collapse to local-table outputs.
  - KSQ004 shows that answer-interface matching is not a complete shortcut
    explanation under `question_form`, but full behavior is template-fragile:
    compact relation-key prompts collapse expected-atomic conflict rows back
    to local-table answers.
  - KSQ005 keeps known facts and contradicted familiar-context correction/
    abstention alive, but unknown nonce and unsupported context abstention fail
    before hidden-state work.
  - KSQ006 keeps supported, irrelevant, and contradicting context behavior
    alive, but insufficient context and claim/context-only controls reproduce
    answer channels before hidden-state work.
- This supports the reviewer-derived doctrine that the useful object is not a
  single "knowledge vector" but the distribution of failure modes across prompt
  contract, local table pressure, semantic prior pressure, parseability,
  source-rewrite stability, answer-interface form, template phrasing, nulls,
  answerability, context relation support, and output geometry.
- It also creates six new diagnostic classes for future rows:
  `FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF` and
  `SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE` and
  `STATUSLESS_EVIDENCE_LOCAL_TABLE_DOMINANCE` and
  `ANSWER_INTERFACE_TEMPLATE_FRAGILITY` and
  `GROUNDED_ANSWERABILITY_UNKNOWN_UNSUPPORTED_FAILURE` and
  `CONTEXT_SUPPORT_CLAIM_OR_MENTION_CONTROL_FAILURE`.
- None of these rows licenses hidden-state work. They are behavior-substrate
  evidence and diagnostic law evidence only.

Knowledge-first-run-outcomes build:

- added `code/control_surface_knowledge_first_run_outcomes.py`;
- writes `data/control_surface_knowledge_first_run_outcomes.json`;
- writes `research/47_CONTROL_SURFACE_KNOWLEDGE_FIRST_RUN_OUTCOMES.md`;
- reads `data/control_surface_knowledge_first_run_pack.json` plus all
  KSQ001-KSQ006 structural, smoke, and available full-behavior result JSONs;
- converts the six executed first-run packets into a single outcome matrix
  with candidate id, ladder level, terminal gate, final diagnostic, exported
  diagnostic class, selected template, failure axes, survived controls, failed
  gates, artifact paths, and hidden-state license;
- records the current executed-KSQ distribution: 6/6 structural gates passed,
  3/6 candidates stopped at full behavior, 3/6 stopped at smoke behavior,
  0/6 are behavior-ready, 0/6 allow hidden-state work, and all 3 knowledge
  levels have exactly two executed candidates;
- validates six exported diagnostics:
  `FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF`,
  `SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE`,
  `STATUSLESS_EVIDENCE_LOCAL_TABLE_DOMINANCE`,
  `ANSWER_INTERFACE_TEMPLATE_FRAGILITY`,
  `GROUNDED_ANSWERABILITY_UNKNOWN_UNSUPPORTED_FAILURE`, and
  `CONTEXT_SUPPORT_CLAIM_OR_MENTION_CONTROL_FAILURE`;
- integrates outcome-matrix staleness and assertion checks into
  `code/validate_control_surface_atlas.py`;
- still licenses no hidden-state signatures, interventions, mechanism claim,
  knowledge-vector claim, or knowledge-control-surface claim.

Knowledge-failure-topology build:

- added `code/control_surface_knowledge_failure_topology.py`;
- writes `data/control_surface_knowledge_failure_topology.json`;
- writes `research/48_CONTROL_SURFACE_KNOWLEDGE_FAILURE_TOPOLOGY.md`;
- reads the executed KSQ outcome matrix and classifies the six outcomes into
  four topology nodes: structural construction solved, full-behavior
  parseability near-misses, bridge template/local-table boundary, and real
  uncertainty answer-channel failures;
- records 5 second-wave work orders:
  `repair_ksq002_source_rewrite_holdout` as the immediate narrow repair,
  `adjudicate_ksq004_template_invariance` as the high-priority bridge
  boundary test, `bound_ksq001_familiar_prior_parseability` as a medium
  repair-or-closeout, `redesign_statusless_bridge_substrate` as a medium
  material bridge redesign, and
  `redesign_real_uncertainty_answerability` as a medium uncertainty redesign;
- validates that all six KSQ candidates are covered by topology nodes and work
  orders, no hidden-state license is created, every work order binds
  promotion/kill/containment/export/evidence rules, and near-miss and redesign
  candidates remain separated instead of flattened into one generic failure
  bucket;
- integrates topology staleness and assertion checks into
  `code/validate_control_surface_atlas.py`;
- still licenses no hidden-state signatures, interventions, mechanism claim,
  knowledge-vector claim, or knowledge-control-surface claim.

Knowledge-second-wave-outcomes build:

- added `code/ksq002_source_rewrite_holdout_repair.py`;
- added `code/ksq001_familiar_prior_parseability_bound.py`;
- added `code/ksq003_evidence_sufficiency_redesign.py`;
- added `code/ksq004_template_invariance_adjudication.py`;
- added `code/ksq005_006_relation_evidence_answerability_redesign.py`;
- added `code/control_surface_knowledge_second_wave_outcomes.py`;
- writes `data/control_surface_knowledge_second_wave_outcomes.json`;
- writes `research/49_CONTROL_SURFACE_KNOWLEDGE_SECOND_WAVE_OUTCOMES.md`;
- writes the KSQ002 repair preregistration and status card:
  `research/prereg/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR.md` and
  `research/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR_STATUS.md`;
- writes the KSQ001 parseability-bound preregistration and status card:
  `research/prereg/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND.md` and
  `research/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND_STATUS.md`;
- writes the KSQ003 evidence-sufficiency redesign preregistration and status card:
  `research/prereg/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN.md` and
  `research/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN_STATUS.md`;
- writes the KSQ004 template-invariance preregistration and status card:
  `research/prereg/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION.md` and
  `research/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION_STATUS.md`;
- writes the KSQ005/KSQ006 relation-evidence answerability preregistration
  and status card:
  `research/prereg/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY.md` and
  `research/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY_STATUS.md`;
- executes the medium-priority `bound_ksq001_familiar_prior_parseability`
  work order with a compact replay plus two softer answer-shape variants, 40
  sources, the original 4 panels, prompt audit, source-disjoint holdout, and
  candidate/output margin reporting;
- records the decisive KSQ001 second-wave result: the 10-source compact replay
  remains a candidate, but the full compact replay exactly preserves the
  original conflict parseability boundary (`0.600`; 18 artificial-value, 2
  real-prior, 4 UNKNOWN, 16 unparsed), while softer answer-shape variants lose
  the prior branch or move toward UNKNOWN;
- exports `FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF` and closes ordinary
  KSQ001 parseability repair rather than admitting hidden-state work;
- executes the immediate `repair_ksq002_source_rewrite_holdout` work order
  with a fixed `city_field_rewrite` answer-channel repair, 40 sources, 6
  original panels, prompt audit, deletion/query-only controls, and
  candidate/output margin reporting;
- records the decisive second-wave result: the named source-disjoint rewrite
  holdout boundary is repaired to 16/16 artificial-value answers, but baseline
  lookup drops to 30/40 and source-deletion UNKNOWN drops to 10/40;
- exports `SOURCE_REWRITE_REPAIR_LOCALITY_REGRESSION` and kills ordinary
  KSQ002 source-rewrite repair rather than admitting hidden-state work;
- executes the medium-priority `redesign_statusless_bridge_substrate` work
  order with a full 40-source evidence-sufficiency redesign, three templates,
  seven panels, prompt audit, source-disjoint holdout, and candidate/output
  margin reporting;
- records the decisive KSQ003 second-wave result: local-table dominance mostly
  disappears, but learned-branch activation and null-stress rejection trade off
  across templates. `identity_packet` gets `40/40` complete-evidence atomic
  answers while failing null stress, and `compact_identity` improves UNKNOWN
  behavior while reducing complete-evidence atomic answers to `13/40`;
- exports `STATUSLESS_EVIDENCE_SUFFICIENCY_BOUNDARY` and bounds KSQ003 without
  admitting hidden-state work;
- executes the high-priority `adjudicate_ksq004_template_invariance` work
  order with 40 sources, 3 predeclared templates, the original matched-pair
  panels, source-disjoint holdout, nulls, side-leakage controls, and
  candidate/output margin reporting;
- records the decisive bridge result: `question_form` and `relation_key_form`
  pass all full-behavior gates, while `neutral_sentence_form` fails conflict
  routing and expected-atomic rows still collapse too often;
- exports `TEMPLATE_INVARIANT_BRIDGE_BEHAVIOR`, admits a bounded behavior
  substrate, and licenses exactly one later signature screen for KSQ004;
- executes the medium-priority `redesign_real_uncertainty_answerability` work
  order with a formal relation-evidence grammar, 40 sources, 7 panels, 3
  templates, prompt audit, source-disjoint holdout, claim-only and mention-only
  controls, contradiction rows, unsupported rows, unknown nonce rows, and
  candidate/output margin reporting;
- structural construction passed for the KSQ005/KSQ006 redesign:
  `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_first_run.json`
  has 840 rows, 40 sources, all panels/templates present, no duplicate rows,
  source-disjoint splits, clean prompt audit, absent support-word prompt
  channel, clean candidate parsing, and a shared response suffix;
- records the decisive KSQ005/KSQ006 full behavior result:
  `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_full_behavior.json`.
  The selected `compact_relation` template keeps known/direct answers at
  `39/40` and exact `REL capital_of` supported rows at `40/40`, but unknown
  nonce rows reach only `27/40` abstain with `13/40` unparsed, unsupported
  rows reach `28/40` abstain with `10/40` unsupported answers and `2/40`
  unparsed, contradiction rows abstain only `2/40` and choose the prior/true
  capital `38/40`, and claim/mention controls reproduce the supported capital
  on `79/80` rows;
- records the complementary template boundary: stricter `relation_rows`
  protects controls and abstention better (`0.825` control-abstain rate,
  `1.000` unknown/unsupported/contradiction abstain rates), but supported
  relation answering collapses to `11/40`;
- exports `RELATION_EVIDENCE_ANSWERABILITY_BOUNDARY`, kills the current
  real-uncertainty route, and licenses no hidden-state search;
- records the updated second-wave distribution: 5 completed work orders, 0
  pending work orders, 1 behavior-ready row, 1 signature-screen license, 3
  killed/closed routes, and 0 hidden-state/intervention licenses;
- integrates second-wave outcome staleness and assertion checks into
  `code/validate_control_surface_atlas.py`;
- still licenses no hidden-state claim, intervention, mechanism claim,
  knowledge-vector claim, or knowledge-control-surface claim.

KSQ007 nonce-evidence answerability calibrator:

- added `code/ksq007_nonce_evidence_answerability_calibrator.py`;
- writes
  `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_first_run.json`;
- writes
  `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_smoke_limit10.json`;
- writes `research/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY_STATUS.md`;
- writes `research/prereg/KSQ007_NONCE_EVIDENCE_ANSWERABILITY.md`;
- executes a post-KSQ005/KSQ006 calibration branch, not a hidden-state search:
  remove familiar capital facts and real city values, keep exact evidence,
  absent evidence, unrelated entity, contradiction, claim-only, mention-only,
  query-only, holdout, prompt-audit, and candidate/output controls;
- structural construction passed: the first-run artifact has 840 rows, 40
  nonce sources, 7 panels, 3 templates, clean source-disjoint splits, clean
  prompt audit, no duplicate rows, no candidate collisions, no support-word or
  status prompt channel, and no hidden-state license;
- 10-source smoke selected `evidence_rows` after the selection rule was fixed
  to maximize the weakest branch rather than the prettiest exact-answer rate;
- selected-template rates: exact evidence answered `9/10`, absent evidence
  abstained `10/10`, unrelated-entity rows abstained `10/10`, conflicting rows
  abstained `9/10` with `1/10` conflict value selected, mention-only controls
  abstained `10/10`, and query-only controls abstained `10/10`;
- the selected template still failed because claim-only controls reproduced
  the value on `6/10` rows; combined controls were `34/40` abstain with `6/40`
  claim-only reproduction, so the route remains behavior-not-ready;
- comparison across templates sharpens the axis: `compact_evidence` answers
  exact rows `10/10` but collapses absent rows to `10/10` unparsed and claim
  rows to `10/10` reproduction; `ledger_form` answers exact rows `10/10` but
  also treats unrelated and conflicting rows as answer channels; only
  `evidence_rows` makes absence, unrelated entity, conflict, mention, and
  query-only behavior mostly stable;
- exports `NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY`: the answerability problem is
  not only real-world capital prior pressure. Even with nonce values, claim
  text can act like evidence unless the control surface separates claim text
  from determinative evidence more strongly;
- licenses no signature screen, hidden-state claim, intervention, mechanism
  claim, uncertainty-control claim, or factual-correction claim.

KSQ007B claim-channel boundary audit:

- added `code/ksq007_claim_channel_boundary_audit.py`;
- writes
  `results/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY/ksq007_claim_channel_boundary_first_run.json`;
- writes
  `results/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY/ksq007_claim_channel_boundary_smoke_limit10.json`;
- writes `research/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY_STATUS.md`;
- writes `research/prereg/KSQ007_CLAIM_CHANNEL_BOUNDARY.md`;
- executes a behavior-only follow-up to classify the KSQ007 claim-only leak,
  not to rescue it into a hidden-state substrate;
- structural construction passed: the first-run artifact has 1,200 rows, 40
  nonce sources, 10 panels, 3 templates, clean source-disjoint splits, clean
  prompt audit, no duplicate rows, no candidate collisions, no support-word or
  status prompt channel, and no hidden-state license;
- 10-source smoke selected `counted_uncounted_sections`: exact evidence
  answered `9/10`, literal `CLAIM answer_for(entity)=value` abstained `9/10`,
  `NOT_EVIDENCE` rows abstained `10/10`, quoted evidence syntax abstained
  `10/10`, wrong-predicate claims abstained `10/10`, mention-only controls
  abstained `10/10`, and query-only controls abstained `10/10`;
- the selected template still failed because prose claims reproduced `4/10`,
  bare `answer_for(entity)=value` reproduced `7/10`, and evidence-looking rows
  placed outside the counted block reproduced `2/10`;
- exports `ANSWER_FOR_SYNTAX_CLAIM_LEAK`: the primary remaining leak is not
  the lexical `CLAIM` label and not simple value mention. It is the
  slot-binding form itself: when the prompt contains a bare
  `answer_for(entity)=value` assertion, the model often treats that assertion
  as answer-bearing even under explicit counted/uncounted section rules;
- licenses no signature screen, hidden-state claim, intervention, mechanism
  claim, uncertainty-control claim, or factual-correction claim.

Knowledge third-wave outcomes layer:

- added `code/control_surface_knowledge_third_wave_outcomes.py`;
- writes `data/control_surface_knowledge_third_wave_outcomes.json`;
- writes `research/50_CONTROL_SURFACE_KNOWLEDGE_THIRD_WAVE_OUTCOMES.md`;
- validates the post-second-wave diagnostic chain from KSQ007 to KSQ007B;
- records two completed outcomes, 2,040 structural rows, 510 smoke rows, zero
  behavior-ready outcomes, zero signature-screen licenses, zero hidden-state
  licenses, zero intervention licenses, and zero mechanism claims;
- pins `NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY` and
  `ANSWER_FOR_SYNTAX_CLAIM_LEAK` as a two-step boundary sequence:
  real-world prior pressure is not the whole answerability problem, and
  section labels do not fully suppress answer-bearing slot-binding syntax;
- makes the next required evidence explicit: a behavior substrate that answers
  exact evidence while suppressing bare `answer_for(entity)=value` claims,
  followed by source-disjoint full behavior, candidate/output margins, and only
  then any hidden-state screen.

KSQ008 neutral-evidence channel repair:

- added `code/ksq008_neutral_evidence_channel_repair.py`;
- writes
  `results/cards/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR/ksq008_neutral_evidence_channel_repair_first_run.json`;
- writes
  `results/cards/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR/ksq008_neutral_evidence_channel_repair_smoke_limit10.json`;
- writes `research/cards/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR_STATUS.md`;
- writes `research/prereg/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR.md`;
- builds a 1,440-row structural substrate over 40 nonce sources, 12 panels, 3
  templates, and source-disjoint splits;
- runs a 360-row, 10-source smoke with candidate/output margins;
- selects `field_registry`;
- exports `NEUTRAL_EVIDENCE_POSITIVE_FAILED`;
- exact neutral evidence answers only `5/10`;
- conflicting neutral evidence abstains `8/10` and selects a value `2/10`;
- counted wrong-schema `answer_for(entity)=value` rows reproduce `5/10`;
- neutral evidence versus forbidden bare alternate answers the counted neutral
  value only `2/10` and abstains `8/10`;
- forbidden bare answer_for, claim answer_for, prose claim, uncounted neutral,
  quoted neutral, absent, unrelated, and query-only controls abstain `10/10`
  under the selected template;
- licenses no behavior-ready substrate, signature screen, hidden-state claim,
  intervention, mechanism claim, uncertainty-control claim, or factual-
  correction claim.

Knowledge fourth-wave outcomes layer:

- added `code/control_surface_knowledge_fourth_wave_outcomes.py`;
- writes `data/control_surface_knowledge_fourth_wave_outcomes.json`;
- writes `research/51_CONTROL_SURFACE_KNOWLEDGE_FOURTH_WAVE_OUTCOMES.md`;
- validates the KSQ008 repair as one completed outcome, 1,440 structural rows,
  360 smoke rows, zero behavior-ready outcomes, zero signature-screen
  licenses, zero hidden-state licenses, zero intervention licenses, and zero
  mechanism claims;
- pins `NEUTRAL_EVIDENCE_POSITIVE_FAILED` as the primary boundary and
  counted wrong-schema answer_for reproduction as the secondary boundary;
- updates the live answerability lesson: after answer_for syntax is isolated,
  the first neutral-channel repair still trades off positive evidence
  parseability against schema specificity.

Knowledge fifth-wave outcomes layer:

- added `code/control_surface_knowledge_fifth_wave_outcomes.py`;
- writes `data/control_surface_knowledge_fifth_wave_outcomes.json`;
- writes `research/52_CONTROL_SURFACE_KNOWLEDGE_FIFTH_WAVE_OUTCOMES.md`;
- validates KSQ009 as one completed schema-specific value-lookup repair with
  1,200 structural rows, 300 smoke rows, zero behavior-ready outcomes, zero
  signature-screen licenses, zero hidden-state licenses, zero intervention
  licenses, and zero mechanism claims;
- pins `SCHEMA_SPECIFIC_POSITIVE_FAILED` as the boundary: the selected
  `kv_lines` template answers exact ALLOW rows only `2/10`, while counted
  wrong-schema `answer_for(entity)=value` rows reproduce `9/10`, uncounted
  wrong-schema `answer_for` rows reproduce `7/10`, and uncounted `answer_for`
  alternates override explicit ALLOW rows `9/10`;
- updates the live answerability lesson: explicit row labels do not by
  themselves defeat an answer-bearing competing syntax.

Knowledge sixth-wave outcomes layer:

- added `code/control_surface_knowledge_sixth_wave_outcomes.py`;
- writes `data/control_surface_knowledge_sixth_wave_outcomes.json`;
- writes `research/53_CONTROL_SURFACE_KNOWLEDGE_SIXTH_WAVE_OUTCOMES.md`;
- validates KSQ010 as one completed two-stage codebook repair with 1,440
  structural rows, 360 smoke rows, zero behavior-ready outcomes, zero
  signature-screen licenses, zero hidden-state licenses, zero intervention
  licenses, and zero mechanism claims;
- pins `CODEBOOK_POSITIVE_FAILED` as the boundary: the selected `tag_rows`
  template improves exact bridge answering to `8/10`, but conflict panels still
  select a value `10/10`, counted `answer_for(entity)=value` rows reproduce
  `10/10`, and `answer_for` alternates override the codebook bridge `9/10`;
- updates the live answerability lesson: positive lookup and answer-channel
  suppression are separable axes, so positive-answer rate cannot license
  hidden-state work unless conflict, locality, and adversary panels pass too.

Knowledge seventh-wave outcomes layer:

- added `code/control_surface_knowledge_seventh_wave_outcomes.py`;
- writes `data/control_surface_knowledge_seventh_wave_outcomes.json`;
- writes `research/54_CONTROL_SURFACE_KNOWLEDGE_SEVENTH_WAVE_OUTCOMES.md`;
- validates KSQ011 as one completed syntax-ablation diagnostic with 480
  structural rows, 120 smoke rows, zero behavior-ready outcomes, zero
  signature-screen licenses, zero hidden-state licenses, zero intervention
  licenses, and zero mechanism claims;
- pins `FUNCTION_ASSIGNMENT_ANSWER_CHANNEL_DOMINANCE` as the boundary: exact
  bridge lookup answers `9/10`, but exact `answer_for`, spaced `answer_for`,
  colon `answer_for`, `answer_to`, `value_for`, prose value notes, and quoted
  `answer_for` override `7/10` to `10/10`, while plain entity assignment and
  bare alternate mention mostly preserve the bridge at `8/10`;
- updates the live answerability lesson: the answer_for boundary is not an
  exact spelling artifact. It is a broader function-like assignment answer
  channel, separable from generic value salience.

Knowledge eighth-wave outcomes layer:

- added `code/ksq012_function_assignment_wrapper_repair.py`;
- added `code/control_surface_knowledge_eighth_wave_outcomes.py`;
- writes `data/control_surface_knowledge_eighth_wave_outcomes.json`;
- writes `research/55_CONTROL_SURFACE_KNOWLEDGE_EIGHTH_WAVE_OUTCOMES.md`;
- validates KSQ012 as one completed function-assignment wrapper-repair
  diagnostic with 480 structural rows, 120 smoke rows, zero behavior-ready
  outcomes, zero signature-screen licenses, zero hidden-state licenses, zero
  intervention licenses, and zero mechanism claims;
- pins `FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK` as the boundary:
  exact bridge lookup answers `9/10`, raw `answer_for` reproduces the alternate
  `9/10`, inactive-block wrapper overrides `9/10`, comment-mark wrapper
  overrides `8/10`, fenced text overrides `9/10`, below-cut text overrides
  `9/10`, detached function/value rows override `7/10`, unrelated-entity
  `answer_for` still overrides `6/10`, and the assignment-only inactive control
  leaks `8/10`;
- records the partial counter-signal: split-assignment-only and query-only
  controls abstain `10/10`, while masked and split bridge panels are partial
  rather than behavior-ready;
- updates the live answerability lesson: visible function-like assignment text
  is unsafe by default. Wrapper prose, comments, fences, and cut markers are
  weak controls, and any future split/masked route must prove both bridge
  preservation and no-bridge abstention before hidden-state work.

Knowledge ninth-wave outcomes layer:

- added `code/ksq013_nonfunction_representation_screen.py`;
- added `code/control_surface_knowledge_ninth_wave_outcomes.py`;
- writes `data/control_surface_knowledge_ninth_wave_outcomes.json`;
- writes `research/56_CONTROL_SURFACE_KNOWLEDGE_NINTH_WAVE_OUTCOMES.md`;
- validates KSQ013 as one completed nonfunction-representation diagnostic
  with 480 structural rows, 120 smoke rows, zero behavior-ready outcomes, zero
  signature-screen licenses, zero hidden-state licenses, zero intervention
  licenses, and zero mechanism claims;
- pins `NONFUNCTION_REPRESENTATION_CONTROL_AND_REPAIR_LEAK` as the boundary:
  exact bridge lookup answers `9/10`, raw `answer_for` reproduces the
  alternate `9/10`, value-bank and decoy-pair bridge panels answer `8/10`
  with parse losses, metadata answers `7/10`, separated entity/value answers
  `6/10`, bare alternate mention answers `8/10`, and separated entity/value
  no-bridge control leaks the alternate `4/10`;
- records the one constructive signal: catalog slash text answers the bridge
  `9/10`, matching exact bridge in the smoke, but KSQ013 does not include a
  slash-only no-bridge locality control;
- updates the live answerability lesson: nonfunction text is not automatically
  safe. It can reduce function-assignment dominance while still damaging
  bridge preservation, parseability, or locality. The next admissible repair
  is a slash-only locality packet, not a hidden-state screen.

Knowledge tenth-wave outcomes layer:

- added `code/ksq014_slash_locality_packet.py`;
- added `code/control_surface_knowledge_tenth_wave_outcomes.py`;
- writes `data/control_surface_knowledge_tenth_wave_outcomes.json`;
- writes `research/57_CONTROL_SURFACE_KNOWLEDGE_TENTH_WAVE_OUTCOMES.md`;
- validates KSQ014 as one completed slash-locality diagnostic with 440
  structural rows, 110 smoke rows, zero behavior-ready outcomes, zero
  signature-screen licenses, zero hidden-state licenses, zero intervention
  licenses, and zero mechanism claims;
- pins `SLASH_LOCALITY_BRIDGE_LOSS` as the boundary: exact bridge and raw
  `answer_for` both pass `9/10`, catalog slash entity answers `9/10`,
  catalog slash decoy answers `9/10` with `1/10` abstain, catalog slash
  reversed answers `9/10`, all matched slash-only controls abstain `10/10`,
  and bare slash bridge answers only `6/10` with `4/10` unparsed;
- updates the live answerability lesson: catalog-labeled slash text is now a
  plausible full-source behavior candidate, while bare slash is a parse-fragile
  notation. The next admissible step is a KSQ015 catalog-slash-only full-source
  packet, still behavior-only.

Knowledge eleventh-wave outcomes layer:

- added `code/ksq015_catalog_slash_full_source_packet.py`;
- added `code/control_surface_knowledge_eleventh_wave_outcomes.py`;
- writes `data/control_surface_knowledge_eleventh_wave_outcomes.json`;
- writes `research/58_CONTROL_SURFACE_KNOWLEDGE_ELEVENTH_WAVE_OUTCOMES.md`;
- validates KSQ015 as one completed full-source catalog-slash diagnostic with
  360 structural rows, 360 full-behavior rows, zero behavior-ready outcomes,
  zero signature-screen licenses, zero hidden-state licenses, zero
  intervention licenses, and zero mechanism claims;
- pins `CATALOG_SLASH_BRIDGE_POSITIVE_FAILED` as the boundary: exact bridge
  answers only `35/40` with `5/40` unparsed code-token outputs, raw
  `answer_for` remains active at `39/40`, catalog slash entity answers
  `36/40`, catalog slash decoy answers `35/40`, catalog slash reversed answers
  `35/40`, every catalog slash-only no-bridge control abstains `40/40`, and
  source-disjoint holdout fails by one exact/reversed unparsed bridge row;
- updates the live answerability lesson: catalog slash has clean no-bridge
  locality at full source, but it cannot promote while the counted bridge
  positive control is itself unreliable. The next admissible move is not a
  hidden-state screen; it is a new bridge contract whose exact positive branch
  clears full-source and holdout first.

Transfer-width-probe build:

- added `code/transfer_width_probe_mc005_mc003_mc004.py`;
- writes `data/transfer_width_probe_mc005_mc003_mc004.json`;
- writes `research/prereg/TRANSFER_WIDTH_PROBE_MC005_MC003_MC004.md`;
- reads the atlas, transfer matrix, and offensive-doctrine contract;
- targets the immediate `run_width_transfer_probe` work order;
- assigns MC005 as the bounded reference surface, MC003 as the output-shadow
  diagnostic baseline, and MC004 as the predecision-monitor diagnostic
  baseline;
- predeclares non-Qwen Gemma targets before any small-model generalization
  language is allowed;
- emits 6 required panels: primary lookup effect, answer-absent null locality,
  side rows, prompt robustness, delayed-copy output-shadow baseline, and
  in-context-binding monitor-only baseline;
- validates that the target rows, non-Qwen target, required panel types,
  decision fields, and no-transfer-success claim boundary hold;
- integrates transfer-width-probe staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Singleton-stage-replication build:

- added `code/singleton_stage_replication_pack.py`;
- writes `data/singleton_stage_replication_pack.json`;
- writes `research/prereg/SINGLETON_STAGE_REPLICATION_PACK.md`;
- reads gate geometry, axis interactions, coverage gaps, and offensive
  doctrine;
- targets the `replicate_singleton_stage_laws` work order;
- anchors the current singleton terminal stages to MC001G intervention failure,
  MC012 prompt-channel locality, and MC005 reliability/null boundary;
- emits two materially distinct replication proposals for each singleton
  terminal stage;
- validates that the work order matches, the target stages match the current
  singleton gap, each stage has two proposals, every proposal has decision
  rules, no law promotion is claimed, and the current singleton count is still
  preserved;
- integrates singleton-stage-pack staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Post-MC033 bridge closeout build:

- added `code/post_mc033_bridge_closeout_audit.py`;
- writes `data/post_mc033_bridge_closeout_audit.json`;
- writes `research/cards/POST_MC033_BRIDGE_SUBSTRATE_CLOSEOUT_STATUS.md`;
- reads offensive doctrine, genome snapshot, bridge ladder, smoke diagnostics,
  and error taxonomy;
- targets the `close_post_mc033_bridge_substrate_family` work order;
- records MC031-MC033 as the same-family statusless source-validity closure
  sequence, with MC030-MC033 preserved as the recent bridge closure set;
- validates that the bridge ladder still has 24 rungs, 17 smoke rungs, 0
  hidden-state-allowed bridge rungs, and 0 clean unconfounded bridge
  candidates;
- validates the decisive closure metrics: MC031 invalid-checksum rows select
  local 1.000 and atomic/lure 0.000; MC032 mismatch rows select atomic/lure
  0.000, local at least 0.700, and side number 0.000; MC033 fact-claim match
  rows stay below the local-pass gate and mismatch rows leak the wrong claimed
  number;
- records the surviving claim as a diagnostic family boundary, not a hidden
  signature, intervention, mechanism card, or general knowledge-control
  surface;
- integrates post-MC033 closeout staleness and assertion checks into
  `code/validate_control_surface_atlas.py`.

Next build:

- add family-specific parsers for MC005 null-margin strata, MC006 lead-time and
  output/candidate controls, and MC007-MC009 behavior-substrate gates;
- fail validation when a row's high-level verdict, null-locality field, or
  output/candidate-control claim contradicts linked artifact metrics;
- turn more allowed/forbidden claims into explicit generated consistency rules;
- add deeper family-specific metric parsers for older MC001-MC004 artifacts so
  comparison ratios become verdict-auditable, not only artifact-covered;
- add validator checks that compare row-level verdict, null-locality, and
  intervention claims against the generated comparison summaries.

This is infrastructure, but it directly serves the genome map.

## Kill Rules

The law layer must be allowed to die.

| Hypothesis | Immediate Kill Or Downgrade Condition |
| --- | --- |
| final-state output geometry dominance | Final-token hidden signatures repeatedly beat output/candidate baselines and support local control. |
| lead-time monitor before lever | Early hidden signals repeatedly steer behavior cleanly with null locality. |
| source-visible lookup localizes more than parametric override | Factual override localizes as cleanly as MC005, or semi-synthetic lookup is entirely output-visible. |
| coarse source ablation overstates locality | Query-only path interventions repeatedly recover coarse source-mask effects. |
| null reliability bottleneck | New mechanism-like surfaces pass strict nulls as easily as primary effects. |
| behavior substrate first | Weak behavior-table signatures survive unchanged after strict behavior repair. |
| transfer fails at reliability first | Transfer failures repeatedly hit primary effects before null reliability. |

## Current Verdict

The project has not mapped the full knowledge genome.

It has mapped the first compositional shape:

- answer-selection behavior often becomes output/candidate geometry;
- source-visible lookup can expose bounded internal mediation;
- familiar entity names can still behave as prompt-local lookup keys under a
  terse table-authoritative contract;
- lead-time signals can exist before local output geometry catches up;
- lead-time is not causality;
- source dependence is not locality;
- null rows define mechanism boundaries;
- function-like assignment syntax can be an answer-channel control surface
  separate from generic value salience;
- wrapper text is not a reliable quarantine boundary for answer-channel syntax;
- behavior-table quality is the first control surface.

The next serious gain is not another pretty classifier. It is locating the
transition between MC005-style source-value mediation and MC006-style
output-visible factual override.
