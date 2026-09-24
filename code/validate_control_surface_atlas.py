"""Validate the machine-readable control-surface atlas.

This is intentionally dependency-free. It checks that atlas rows use the
canonical vocabularies, evidence paths exist, and selected result artifacts
still contain the values that the atlas claims depend on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from control_surface_artifacts import (
    ARTIFACT_INDEX_PATH,
    build_artifact_index,
    registry_report,
    validate_atlas_artifact_registry,
)
from control_surface_axis_interactions import (
    AXIS_INTERACTIONS_PATH,
    build_control_surface_axis_interactions,
    validate_axis_interactions as validate_axis_interactions_payload,
)
from control_surface_bridge_ladder import (
    BRIDGE_LADDER_PATH,
    build_control_surface_bridge_ladder,
    validate_bridge_ladder as validate_bridge_ladder_payload,
)
from control_surface_comparison import COMPARISON_PATH, build_control_surface_comparison
from control_surface_compositional_genome_audit import (
    COMPOSITIONAL_GENOME_AUDIT_PATH,
    build_control_surface_compositional_genome_audit,
    validate_compositional_genome_audit as validate_compositional_genome_payload,
)
from control_surface_coverage_gaps import (
    COVERAGE_GAPS_PATH,
    build_control_surface_coverage_gaps,
    validate_coverage_gaps as validate_coverage_gaps_payload,
)
from control_surface_gap_closure_plan import (
    GAP_CLOSURE_PLAN_PATH,
    build_control_surface_gap_closure_plan,
    validate_gap_closure_plan as validate_gap_closure_plan_payload,
)
from control_surface_decision_frontier import (
    DECISION_FRONTIER_PATH,
    build_control_surface_decision_frontier,
    validate_decision_frontier as validate_decision_frontier_payload,
)
from control_surface_error_taxonomy import (
    ERROR_TAXONOMY_PATH,
    build_control_surface_error_taxonomy,
    validate_error_taxonomy as validate_error_taxonomy_payload,
)
from control_surface_family_matrix import (
    FAMILY_MATRIX_PATH,
    build_control_surface_family_matrix,
    validate_family_matrix as validate_family_matrix_payload,
)
from control_surface_gate_geometry import (
    GATE_GEOMETRY_PATH,
    build_control_surface_gate_geometry,
    validate_gate_geometry as validate_gate_geometry_payload,
)
from control_surface_genome_snapshot import (
    GENOME_SNAPSHOT_PATH,
    build_control_surface_genome_snapshot,
    validate_genome_snapshot as validate_genome_snapshot_payload,
)
from control_surface_knowledge_ladder import (
    KNOWLEDGE_LADDER_PATH,
    build_control_surface_knowledge_ladder,
    validate_knowledge_ladder as validate_knowledge_ladder_payload,
)
from control_surface_knowledge_gap_plan import (
    KNOWLEDGE_GAP_PLAN_PATH,
    build_control_surface_knowledge_gap_plan,
    validate_knowledge_gap_plan as validate_knowledge_gap_plan_payload,
)
from control_surface_knowledge_substrate_admission import (
    KNOWLEDGE_SUBSTRATE_ADMISSION_PATH,
    build_control_surface_knowledge_substrate_admission,
    validate_knowledge_substrate_admission as validate_knowledge_admission_payload,
)
from control_surface_knowledge_candidate_queue import (
    KNOWLEDGE_CANDIDATE_QUEUE_PATH,
    build_control_surface_knowledge_candidate_queue,
    validate_knowledge_candidate_queue as validate_knowledge_candidate_payload,
)
from control_surface_knowledge_first_run_pack import (
    KNOWLEDGE_FIRST_RUN_PACK_PATH,
    build_control_surface_knowledge_first_run_pack,
    validate_knowledge_first_run_pack as validate_knowledge_first_run_payload,
)
from control_surface_knowledge_first_run_outcomes import (
    KNOWLEDGE_FIRST_RUN_OUTCOMES_PATH,
    build_control_surface_knowledge_first_run_outcomes,
    validate_knowledge_first_run_outcomes as validate_knowledge_first_run_outcomes_payload,
)
from control_surface_knowledge_failure_topology import (
    KNOWLEDGE_FAILURE_TOPOLOGY_PATH,
    build_control_surface_knowledge_failure_topology,
    validate_knowledge_failure_topology as validate_knowledge_failure_topology_payload,
)
from control_surface_knowledge_second_wave_outcomes import (
    SECOND_WAVE_OUTCOMES_PATH,
    build_control_surface_knowledge_second_wave_outcomes,
    validate_knowledge_second_wave_outcomes as validate_knowledge_second_wave_outcomes_payload,
)
from control_surface_knowledge_third_wave_outcomes import (
    THIRD_WAVE_OUTCOMES_PATH,
    build_control_surface_knowledge_third_wave_outcomes,
    validate_knowledge_third_wave_outcomes as validate_knowledge_third_wave_outcomes_payload,
)
from control_surface_knowledge_fourth_wave_outcomes import (
    FOURTH_WAVE_OUTCOMES_PATH,
    build_control_surface_knowledge_fourth_wave_outcomes,
    validate_knowledge_fourth_wave_outcomes as validate_knowledge_fourth_wave_outcomes_payload,
)
from control_surface_knowledge_fifth_wave_outcomes import (
    FIFTH_WAVE_OUTCOMES_PATH,
    build_control_surface_knowledge_fifth_wave_outcomes,
    validate_knowledge_fifth_wave_outcomes as validate_knowledge_fifth_wave_outcomes_payload,
)
from control_surface_knowledge_sixth_wave_outcomes import (
    SIXTH_WAVE_OUTCOMES_PATH,
    build_control_surface_knowledge_sixth_wave_outcomes,
    validate_knowledge_sixth_wave_outcomes as validate_knowledge_sixth_wave_outcomes_payload,
)
from control_surface_knowledge_seventh_wave_outcomes import (
    SEVENTH_WAVE_OUTCOMES_PATH,
    build_control_surface_knowledge_seventh_wave_outcomes,
    validate_knowledge_seventh_wave_outcomes as validate_knowledge_seventh_wave_outcomes_payload,
)
from control_surface_knowledge_eighth_wave_outcomes import (
    EIGHTH_WAVE_OUTCOMES_PATH,
    build_control_surface_knowledge_eighth_wave_outcomes,
    validate_knowledge_eighth_wave_outcomes as validate_knowledge_eighth_wave_outcomes_payload,
)
from control_surface_knowledge_ninth_wave_outcomes import (
    NINTH_WAVE_OUTCOMES_PATH,
    build_control_surface_knowledge_ninth_wave_outcomes,
    validate_knowledge_ninth_wave_outcomes as validate_knowledge_ninth_wave_outcomes_payload,
)
from control_surface_knowledge_tenth_wave_outcomes import (
    TENTH_WAVE_OUTCOMES_PATH,
    build_control_surface_knowledge_tenth_wave_outcomes,
    validate_knowledge_tenth_wave_outcomes as validate_knowledge_tenth_wave_outcomes_payload,
)
from control_surface_knowledge_eleventh_wave_outcomes import (
    ELEVENTH_WAVE_OUTCOMES_PATH,
    build_control_surface_knowledge_eleventh_wave_outcomes,
    validate_knowledge_eleventh_wave_outcomes as validate_knowledge_eleventh_wave_outcomes_payload,
)
from control_surface_law_audit import LAW_AUDIT_PATH, build_control_surface_law_audit
from control_surface_mixture_law import (
    MIXTURE_LAW_PATH,
    build_control_surface_mixture_law,
    validate_mixture_law as validate_mixture_law_payload,
)
from control_surface_next_queue import NEXT_QUEUE_PATH, build_control_surface_next_queue
from control_surface_offensive_doctrine import (
    OFFENSIVE_DOCTRINE_PATH,
    build_control_surface_offensive_doctrine,
    validate_offensive_doctrine as validate_offensive_doctrine_payload,
)
from mc005_reference_specimen_audit import (
    MC005_REFERENCE_SPECIMEN_AUDIT_PATH,
    build_mc005_reference_specimen_audit,
    validate_mc005_reference_specimen_audit as validate_mc005_reference_payload,
)
from mc006_predecision_frontier_audit import (
    MC006_PREDECISION_FRONTIER_AUDIT_PATH,
    build_mc006_predecision_frontier_audit,
    validate_mc006_predecision_frontier_audit as validate_mc006_frontier_payload,
)
from post_mc033_bridge_closeout_audit import (
    POST_MC033_BRIDGE_CLOSEOUT_PATH,
    build_post_mc033_bridge_closeout_audit,
    validate_post_mc033_bridge_closeout_audit as validate_post_mc033_closeout_payload,
)
from control_surface_route_disposition import (
    ROUTE_DISPOSITION_PATH,
    build_control_surface_route_disposition,
    validate_route_disposition as validate_route_disposition_payload,
)
from control_surface_reliability_matrix import (
    RELIABILITY_MATRIX_PATH,
    build_control_surface_reliability_matrix,
    validate_reliability_matrix as validate_reliability_matrix_payload,
)
from control_surface_smoke_diagnostics import (
    SMOKE_DIAGNOSTICS_PATH,
    build_control_surface_smoke_diagnostics,
    validate_smoke_diagnostics as validate_smoke_diagnostics_payload,
)
from control_surface_transfer_matrix import (
    TRANSFER_MATRIX_PATH,
    build_control_surface_transfer_matrix,
    validate_transfer_matrix as validate_transfer_matrix_payload,
)
from singleton_stage_replication_pack import (
    SINGLETON_STAGE_PACK_PATH,
    build_singleton_stage_replication_pack,
    validate_singleton_stage_replication_pack as validate_singleton_stage_pack_payload,
)
from transfer_width_probe_mc005_mc003_mc004 import (
    TRANSFER_WIDTH_PROBE_PATH,
    build_transfer_width_probe,
    validate_transfer_width_probe as validate_transfer_width_probe_payload,
)


ROOT = Path(__file__).resolve().parents[1]
ATLAS_PATH = ROOT / "data" / "control_surface_atlas.json"
HYPOTHESES_PATH = ROOT / "data" / "control_surface_law_hypotheses.json"
KSQ001_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE"
    / "ksq001_familiar_entity_prior_counterbalance_first_run.json"
)
KSQ001_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE"
    / "ksq001_familiar_entity_prior_counterbalance_smoke_limit10.json"
)
KSQ001_FULL_BEHAVIOR_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE"
    / "ksq001_familiar_entity_prior_counterbalance_full_behavior.json"
)
KSQ001_FIRST_RUN_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN_STATUS.md"
)
KSQ001_FIRST_RUN_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN.md"
)
KSQ002_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE"
    / "ksq002_familiar_entity_source_rewrite_equivalence_first_run.json"
)
KSQ002_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE"
    / "ksq002_familiar_entity_source_rewrite_equivalence_smoke_limit10.json"
)
KSQ002_FULL_BEHAVIOR_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE"
    / "ksq002_familiar_entity_source_rewrite_equivalence_full_behavior.json"
)
KSQ002_FIRST_RUN_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN_STATUS.md"
)
KSQ002_FIRST_RUN_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN.md"
)
KSQ003_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION"
    / "ksq003_bridge_statusless_evidence_aggregation_first_run.json"
)
KSQ003_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION"
    / "ksq003_bridge_statusless_evidence_aggregation_smoke_limit10.json"
)
KSQ003_FIRST_RUN_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN_STATUS.md"
)
KSQ003_FIRST_RUN_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN.md"
)
KSQ005_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY"
    / "ksq005_uncertainty_grounded_answerability_first_run.json"
)
KSQ005_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY"
    / "ksq005_uncertainty_grounded_answerability_smoke_limit10.json"
)
KSQ005_FIRST_RUN_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN_STATUS.md"
)
KSQ005_FIRST_RUN_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN.md"
)
KSQ006_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS"
    / "ksq006_uncertainty_context_support_counterfactuals_first_run.json"
)
KSQ006_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS"
    / "ksq006_uncertainty_context_support_counterfactuals_smoke_limit10.json"
)
KSQ006_FIRST_RUN_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN_STATUS.md"
)
KSQ006_FIRST_RUN_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN.md"
)
KSQ007_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ007_NONCE_EVIDENCE_ANSWERABILITY"
    / "ksq007_nonce_evidence_answerability_first_run.json"
)
KSQ007_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ007_NONCE_EVIDENCE_ANSWERABILITY"
    / "ksq007_nonce_evidence_answerability_smoke_limit10.json"
)
KSQ007_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ007_NONCE_EVIDENCE_ANSWERABILITY_STATUS.md"
)
KSQ007_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ007_NONCE_EVIDENCE_ANSWERABILITY.md"
)
KSQ007B_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ007_CLAIM_CHANNEL_BOUNDARY"
    / "ksq007_claim_channel_boundary_first_run.json"
)
KSQ007B_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ007_CLAIM_CHANNEL_BOUNDARY"
    / "ksq007_claim_channel_boundary_smoke_limit10.json"
)
KSQ007B_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ007_CLAIM_CHANNEL_BOUNDARY_STATUS.md"
)
KSQ007B_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ007_CLAIM_CHANNEL_BOUNDARY.md"
)
KSQ008_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR"
    / "ksq008_neutral_evidence_channel_repair_first_run.json"
)
KSQ008_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR"
    / "ksq008_neutral_evidence_channel_repair_smoke_limit10.json"
)
KSQ008_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR_STATUS.md"
)
KSQ008_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR.md"
)
KSQ009_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP"
    / "ksq009_schema_specific_value_lookup_first_run.json"
)
KSQ009_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP"
    / "ksq009_schema_specific_value_lookup_smoke_limit10.json"
)
KSQ009_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP_STATUS.md"
)
KSQ009_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP.md"
)
KSQ010_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP"
    / "ksq010_two_stage_codebook_value_lookup_first_run.json"
)
KSQ010_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP"
    / "ksq010_two_stage_codebook_value_lookup_smoke_limit10.json"
)
KSQ010_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP_STATUS.md"
)
KSQ010_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP.md"
)
KSQ011_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ011_ANSWER_FOR_SYNTAX_ABLATION"
    / "ksq011_answer_for_syntax_ablation_first_run.json"
)
KSQ011_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ011_ANSWER_FOR_SYNTAX_ABLATION"
    / "ksq011_answer_for_syntax_ablation_smoke_limit10.json"
)
KSQ011_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ011_ANSWER_FOR_SYNTAX_ABLATION_STATUS.md"
)
KSQ011_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ011_ANSWER_FOR_SYNTAX_ABLATION.md"
)
KSQ012_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR"
    / "ksq012_function_assignment_wrapper_repair_first_run.json"
)
KSQ012_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR"
    / "ksq012_function_assignment_wrapper_repair_smoke_limit10.json"
)
KSQ012_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR_STATUS.md"
)
KSQ012_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR.md"
)
KSQ013_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ013_NONFUNCTION_REPRESENTATION_SCREEN"
    / "ksq013_nonfunction_representation_screen_first_run.json"
)
KSQ013_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ013_NONFUNCTION_REPRESENTATION_SCREEN"
    / "ksq013_nonfunction_representation_screen_smoke_limit10.json"
)
KSQ013_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ013_NONFUNCTION_REPRESENTATION_SCREEN_STATUS.md"
)
KSQ013_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ013_NONFUNCTION_REPRESENTATION_SCREEN.md"
)
KSQ014_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ014_SLASH_LOCALITY_PACKET"
    / "ksq014_slash_locality_packet_first_run.json"
)
KSQ014_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ014_SLASH_LOCALITY_PACKET"
    / "ksq014_slash_locality_packet_smoke_limit10.json"
)
KSQ014_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ014_SLASH_LOCALITY_PACKET_STATUS.md"
)
KSQ014_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ014_SLASH_LOCALITY_PACKET.md"
)
KSQ015_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET"
    / "ksq015_catalog_slash_full_source_first_run.json"
)
KSQ015_FULL_BEHAVIOR_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET"
    / "ksq015_catalog_slash_full_source_full_behavior.json"
)
KSQ015_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET_STATUS.md"
)
KSQ015_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET.md"
)
KSQ004_FIRST_RUN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS"
    / "ksq004_bridge_answer_interface_minimal_pairs_first_run.json"
)
KSQ004_SMOKE_LIMIT10_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS"
    / "ksq004_bridge_answer_interface_minimal_pairs_smoke_limit10.json"
)
KSQ004_FULL_BEHAVIOR_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS"
    / "ksq004_bridge_answer_interface_minimal_pairs_full_behavior.json"
)
KSQ004_FIRST_RUN_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN_STATUS.md"
)
KSQ004_FIRST_RUN_PREREG_PATH = (
    ROOT
    / "research"
    / "prereg"
    / "KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN.md"
)
ALLOWED_HYPOTHESIS_STATUSES = {
    "strong_doctrine",
    "supported_pattern",
    "tentative_pattern",
}


REQUIRED_ROW_FIELDS = {
    "id",
    "family",
    "models",
    "behavior_contract",
    "behavior_gate",
    "current_best_signal",
    "lead_time",
    "intervention",
    "mixture_profile",
    "diagnostics",
    "verdict",
    "allowed_claims",
    "forbidden_claims",
    "evidence",
    "next_decision",
}

REQUIRED_MIXTURE_AXES = {
    "prompt_authority",
    "prompt_format",
    "source_token_dependence",
    "output_geometry",
    "lead_time_internal_signal",
    "local_internal_path",
    "causal_control",
    "null_locality",
    "transfer",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dotted_get(payload: Any, dotted_path: str) -> Any:
    current = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            fail(f"missing artifact path component {part!r} in {dotted_path!r}")
    return current


def validate_rows(atlas: dict[str, Any]) -> None:
    vocab = atlas["controlled_vocab"]
    lead_time_states = set(vocab["lead_time_states"])
    intervention_states = set(vocab["intervention_states"])
    verdict_classes = set(vocab["verdict_classes"])
    diagnostic_types = set(vocab["diagnostic_types"])
    mixture_values = set(vocab["mixture_values"])

    seen_ids: set[str] = set()
    for row in atlas["rows"]:
        row_id = row.get("id", "<missing>")
        missing = REQUIRED_ROW_FIELDS - set(row)
        if missing:
            fail(f"{row_id}: missing fields {sorted(missing)}")
        if row_id in seen_ids:
            fail(f"duplicate row id: {row_id}")
        seen_ids.add(row_id)

        if row["lead_time"]["state"] not in lead_time_states:
            fail(f"{row_id}: invalid lead_time.state {row['lead_time']['state']!r}")
        if row["intervention"]["state"] not in intervention_states:
            fail(f"{row_id}: invalid intervention.state {row['intervention']['state']!r}")
        if row["verdict"]["class"] not in verdict_classes:
            fail(f"{row_id}: invalid verdict.class {row['verdict']['class']!r}")

        unknown_diagnostics = sorted(set(row["diagnostics"]) - diagnostic_types)
        if unknown_diagnostics:
            fail(f"{row_id}: unknown diagnostics {unknown_diagnostics}")

        missing_axes = REQUIRED_MIXTURE_AXES - set(row["mixture_profile"])
        if missing_axes:
            fail(f"{row_id}: missing mixture axes {sorted(missing_axes)}")
        bad_axis_values = {
            axis: value
            for axis, value in row["mixture_profile"].items()
            if value not in mixture_values
        }
        if bad_axis_values:
            fail(f"{row_id}: invalid mixture values {bad_axis_values}")

        if not row["allowed_claims"]:
            fail(f"{row_id}: allowed_claims cannot be empty")
        if not row["forbidden_claims"]:
            fail(f"{row_id}: forbidden_claims cannot be empty")
        if not row["evidence"]:
            fail(f"{row_id}: evidence cannot be empty")
        for rel_path in row["evidence"]:
            evidence_path = ROOT / rel_path
            if not evidence_path.exists():
                fail(f"{row_id}: missing evidence path {rel_path}")


def validate_artifact_assertions(atlas: dict[str, Any]) -> None:
    for assertion in atlas.get("artifact_assertions", []):
        rel_path = assertion["path"]
        artifact_path = ROOT / rel_path
        if not artifact_path.exists():
            fail(f"missing asserted artifact {rel_path}")
        payload = load_json(artifact_path)
        for dotted_path, expected in assertion["checks"].items():
            actual = dotted_get(payload, dotted_path)
            if actual != expected:
                fail(
                    f"{rel_path}: {dotted_path} expected {expected!r}, got {actual!r}"
                )


def validate_artifact_index(atlas: dict[str, Any]) -> None:
    if not ARTIFACT_INDEX_PATH.exists():
        fail(f"missing artifact index {ARTIFACT_INDEX_PATH.relative_to(ROOT)}")
    current_index = build_artifact_index(atlas)
    stored_index = load_json(ARTIFACT_INDEX_PATH)
    if stored_index != current_index:
        fail(
            "artifact index is stale; regenerate with "
            "python code\\control_surface_artifacts.py --write-index"
        )


def validate_comparison(atlas: dict[str, Any]) -> None:
    if not COMPARISON_PATH.exists():
        fail(f"missing comparison artifact {COMPARISON_PATH.relative_to(ROOT)}")
    artifact_index = load_json(ARTIFACT_INDEX_PATH)
    current_comparison = build_control_surface_comparison(atlas, artifact_index)
    stored_comparison = load_json(COMPARISON_PATH)
    if stored_comparison != current_comparison:
        fail(
            "control-surface comparison is stale; regenerate with "
            "python code\\control_surface_comparison.py --write"
        )
    claim_audit = stored_comparison.get("claim_audit", {}).get("summary", {})
    for field in [
        "rows_without_artifacts",
        "rows_without_metrics",
        "rows_without_family_checks",
        "rows_with_partial_metrics",
    ]:
        if claim_audit.get(field):
            fail(f"comparison claim audit has unresolved {field}: {claim_audit[field]}")
    claim_consistency = stored_comparison.get("claim_consistency", {}).get("summary", {})
    if claim_consistency.get("contradictions"):
        fail(
            "comparison claim consistency has contradictions: "
            f"{claim_consistency['contradictions']}"
        )


def validate_hypotheses(atlas: dict[str, Any]) -> dict[str, int]:
    if not HYPOTHESES_PATH.exists():
        fail(f"missing hypotheses file {HYPOTHESES_PATH.relative_to(ROOT)}")

    payload = load_json(HYPOTHESES_PATH)
    if payload.get("schema_version") != 1:
        fail("hypotheses schema_version must be 1")
    if payload.get("atlas_ref") != str(ATLAS_PATH.relative_to(ROOT)).replace("\\", "/"):
        fail("hypotheses atlas_ref must point to data/control_surface_atlas.json")

    row_ids = {row["id"] for row in atlas["rows"]}
    diagnostic_types = set(atlas["controlled_vocab"]["diagnostic_types"])
    required_fields = {
        "id",
        "status",
        "statement",
        "why_it_matters",
        "evidence_rows",
        "evidence_diagnostics",
        "current_support",
        "predicted_next_observations",
        "falsifiers",
        "next_tests",
    }

    seen_ids: set[str] = set()
    status_counts: dict[str, int] = {}
    for hypothesis in payload["hypotheses"]:
        hyp_id = hypothesis.get("id", "<missing>")
        missing = required_fields - set(hypothesis)
        if missing:
            fail(f"{hyp_id}: missing hypothesis fields {sorted(missing)}")
        if hyp_id in seen_ids:
            fail(f"duplicate hypothesis id: {hyp_id}")
        seen_ids.add(hyp_id)

        unknown_rows = sorted(set(hypothesis["evidence_rows"]) - row_ids)
        if unknown_rows:
            fail(f"{hyp_id}: unknown evidence rows {unknown_rows}")
        unknown_diagnostics = sorted(
            set(hypothesis["evidence_diagnostics"]) - diagnostic_types
        )
        if unknown_diagnostics:
            fail(f"{hyp_id}: unknown diagnostics {unknown_diagnostics}")

        for list_field in [
            "evidence_rows",
            "evidence_diagnostics",
            "predicted_next_observations",
            "falsifiers",
            "next_tests",
        ]:
            if not hypothesis[list_field]:
                fail(f"{hyp_id}: {list_field} cannot be empty")

        status = hypothesis["status"]
        if status not in ALLOWED_HYPOTHESIS_STATUSES:
            fail(f"{hyp_id}: unknown hypothesis status {status!r}")
        status_counts[status] = status_counts.get(status, 0) + 1

    return status_counts


def validate_law_audit(atlas: dict[str, Any]) -> dict[str, Any]:
    if not LAW_AUDIT_PATH.exists():
        fail(f"missing law audit {LAW_AUDIT_PATH.relative_to(ROOT)}")
    hypotheses_payload = load_json(HYPOTHESES_PATH)
    comparison = load_json(COMPARISON_PATH)
    current_audit = build_control_surface_law_audit(
        atlas,
        hypotheses_payload,
        comparison,
    )
    stored_audit = load_json(LAW_AUDIT_PATH)
    if stored_audit != current_audit:
        fail(
            "control-surface law audit is stale; regenerate with "
            "python code\\control_surface_law_audit.py --write"
        )

    summary = stored_audit.get("summary", {})
    if summary.get("hypotheses_with_gaps"):
        fail(f"law audit has hypothesis gaps: {summary['hypotheses_with_gaps']}")
    if summary.get("rows_without_law_support"):
        fail(
            "law audit has atlas rows without law support: "
            f"{summary['rows_without_law_support']}"
        )
    if summary.get("unobserved_cited_diagnostics"):
        fail(
            "law audit cites diagnostics not observed in the atlas: "
            f"{summary['unobserved_cited_diagnostics']}"
        )
    return summary


def validate_next_queue(atlas: dict[str, Any]) -> dict[str, Any]:
    if not NEXT_QUEUE_PATH.exists():
        fail(f"missing next-experiment queue {NEXT_QUEUE_PATH.relative_to(ROOT)}")
    hypotheses_payload = load_json(HYPOTHESES_PATH)
    law_audit = load_json(LAW_AUDIT_PATH)
    comparison = load_json(COMPARISON_PATH)
    current_queue = build_control_surface_next_queue(
        atlas,
        hypotheses_payload,
        law_audit,
        comparison,
    )
    stored_queue = load_json(NEXT_QUEUE_PATH)
    if stored_queue != current_queue:
        fail(
            "control-surface next-experiment queue is stale; regenerate with "
            "python code\\control_surface_next_queue.py --write"
        )

    summary = stored_queue.get("summary", {})
    if summary.get("queue_item_count", 0) < len(hypotheses_payload["hypotheses"]):
        fail("next-experiment queue has fewer items than law hypotheses")
    if not summary.get("top_queue_ids"):
        fail("next-experiment queue has no top_queue_ids")
    if not summary.get("immediate_or_high_count"):
        fail("next-experiment queue has no immediate or high-priority items")

    bridge_closure = summary.get("bridge_closure", {})
    if not bridge_closure.get("available"):
        fail("next-experiment queue has no bridge-closure context")
    if bridge_closure.get("bridge_rung_count") != 24:
        fail(
            "next-experiment queue bridge closure has unexpected rung count: "
            f"{bridge_closure.get('bridge_rung_count')}"
        )
    if bridge_closure.get("hidden_state_allowed_count") != 0:
        fail(
            "next-experiment queue bridge closure allows hidden-state work: "
            f"{bridge_closure.get('hidden_state_allowed_count')}"
        )
    if bridge_closure.get("recent_closed_rung_ids") != ["MC030", "MC031", "MC032", "MC033"]:
        fail(
            "next-experiment queue bridge closure has unexpected recent closures: "
            f"{bridge_closure.get('recent_closed_rung_ids')}"
        )
    if bridge_closure.get("closed_contract_axis_count", 0) < 20:
        fail(
            "next-experiment queue bridge closure lost closed contract axes: "
            f"{bridge_closure.get('closed_contract_axis_count')}"
        )

    bridge_items = [
        item
        for item in stored_queue.get("queue", [])
        if "BRIDGE_ROUTE_NEEDED" in item.get("reason_codes", [])
    ]
    if not bridge_items:
        fail("next-experiment queue has no bridge-route items")
    if not bridge_items[0].get("bridge_closure_context", {}).get(
        "active_constraints"
    ):
        fail("bridge-route queue items do not carry active closure constraints")
    return summary


def validate_smoke_diagnostics_layer() -> dict[str, Any]:
    if not SMOKE_DIAGNOSTICS_PATH.exists():
        fail(f"missing smoke diagnostics {SMOKE_DIAGNOSTICS_PATH.relative_to(ROOT)}")
    current_smoke = build_control_surface_smoke_diagnostics()
    stored_smoke = load_json(SMOKE_DIAGNOSTICS_PATH)
    if stored_smoke != current_smoke:
        fail(
            "control-surface smoke diagnostics are stale; regenerate with "
            "python code\\control_surface_smoke_diagnostics.py --write"
        )
    validate_smoke_diagnostics_payload(stored_smoke)
    return stored_smoke["summary"]


def validate_bridge_ladder_layer() -> dict[str, Any]:
    if not BRIDGE_LADDER_PATH.exists():
        fail(f"missing bridge ladder {BRIDGE_LADDER_PATH.relative_to(ROOT)}")
    current_ladder = build_control_surface_bridge_ladder()
    stored_ladder = load_json(BRIDGE_LADDER_PATH)
    if stored_ladder != current_ladder:
        fail(
            "control-surface bridge ladder is stale; regenerate with "
            "python code\\control_surface_bridge_ladder.py --write"
        )
    validate_bridge_ladder_payload(stored_ladder)
    return stored_ladder["summary"]


def validate_mixture_law_layer() -> dict[str, Any]:
    if not MIXTURE_LAW_PATH.exists():
        fail(f"missing mixture law {MIXTURE_LAW_PATH.relative_to(ROOT)}")
    current_mixture = build_control_surface_mixture_law()
    stored_mixture = load_json(MIXTURE_LAW_PATH)
    if stored_mixture != current_mixture:
        fail(
            "control-surface mixture law is stale; regenerate with "
            "python code\\control_surface_mixture_law.py --write"
        )
    validate_mixture_law_payload(stored_mixture)
    return stored_mixture["summary"]


def validate_decision_frontier_layer() -> dict[str, Any]:
    if not DECISION_FRONTIER_PATH.exists():
        fail(f"missing decision frontier {DECISION_FRONTIER_PATH.relative_to(ROOT)}")
    current_frontier = build_control_surface_decision_frontier()
    stored_frontier = load_json(DECISION_FRONTIER_PATH)
    if stored_frontier != current_frontier:
        fail(
            "control-surface decision frontier is stale; regenerate with "
            "python code\\control_surface_decision_frontier.py --write"
        )
    validate_decision_frontier_payload(stored_frontier)
    return stored_frontier["summary"]


def validate_route_disposition_layer() -> dict[str, Any]:
    if not ROUTE_DISPOSITION_PATH.exists():
        fail(f"missing route disposition {ROUTE_DISPOSITION_PATH.relative_to(ROOT)}")
    current_disposition = build_control_surface_route_disposition()
    stored_disposition = load_json(ROUTE_DISPOSITION_PATH)
    if stored_disposition != current_disposition:
        fail(
            "control-surface route disposition is stale; regenerate with "
            "python code\\control_surface_route_disposition.py --write"
        )
    validate_route_disposition_payload(stored_disposition)
    return stored_disposition["summary"]


def validate_transfer_matrix_layer() -> dict[str, Any]:
    if not TRANSFER_MATRIX_PATH.exists():
        fail(f"missing transfer matrix {TRANSFER_MATRIX_PATH.relative_to(ROOT)}")
    current_matrix = build_control_surface_transfer_matrix()
    stored_matrix = load_json(TRANSFER_MATRIX_PATH)
    if stored_matrix != current_matrix:
        fail(
            "control-surface transfer matrix is stale; regenerate with "
            "python code\\control_surface_transfer_matrix.py --write"
        )
    validate_transfer_matrix_payload(stored_matrix)
    return stored_matrix["summary"]


def validate_reliability_matrix_layer() -> dict[str, Any]:
    if not RELIABILITY_MATRIX_PATH.exists():
        fail(f"missing reliability matrix {RELIABILITY_MATRIX_PATH.relative_to(ROOT)}")
    current_matrix = build_control_surface_reliability_matrix()
    stored_matrix = load_json(RELIABILITY_MATRIX_PATH)
    if stored_matrix != current_matrix:
        fail(
            "control-surface reliability matrix is stale; regenerate with "
            "python code\\control_surface_reliability_matrix.py --write"
        )
    validate_reliability_matrix_payload(stored_matrix)
    return stored_matrix["summary"]


def validate_error_taxonomy_layer() -> dict[str, Any]:
    if not ERROR_TAXONOMY_PATH.exists():
        fail(f"missing error taxonomy {ERROR_TAXONOMY_PATH.relative_to(ROOT)}")
    current_taxonomy = build_control_surface_error_taxonomy()
    stored_taxonomy = load_json(ERROR_TAXONOMY_PATH)
    if stored_taxonomy != current_taxonomy:
        fail(
            "control-surface error taxonomy is stale; regenerate with "
            "python code\\control_surface_error_taxonomy.py --write"
        )
    validate_error_taxonomy_payload(stored_taxonomy)
    return stored_taxonomy["summary"]


def validate_gate_geometry_layer() -> dict[str, Any]:
    if not GATE_GEOMETRY_PATH.exists():
        fail(f"missing gate geometry {GATE_GEOMETRY_PATH.relative_to(ROOT)}")
    current_geometry = build_control_surface_gate_geometry()
    stored_geometry = load_json(GATE_GEOMETRY_PATH)
    if stored_geometry != current_geometry:
        fail(
            "control-surface gate geometry is stale; regenerate with "
            "python code\\control_surface_gate_geometry.py --write"
        )
    validate_gate_geometry_payload(stored_geometry)
    return stored_geometry["summary"]


def validate_genome_snapshot_layer() -> dict[str, Any]:
    if not GENOME_SNAPSHOT_PATH.exists():
        fail(f"missing genome snapshot {GENOME_SNAPSHOT_PATH.relative_to(ROOT)}")
    current_snapshot = build_control_surface_genome_snapshot()
    stored_snapshot = load_json(GENOME_SNAPSHOT_PATH)
    if stored_snapshot != current_snapshot:
        fail(
            "control-surface genome snapshot is stale; regenerate with "
            "python code\\control_surface_genome_snapshot.py --write"
        )
    validate_genome_snapshot_payload(stored_snapshot)
    return stored_snapshot


def validate_axis_interactions_layer() -> dict[str, Any]:
    if not AXIS_INTERACTIONS_PATH.exists():
        fail(f"missing axis interactions {AXIS_INTERACTIONS_PATH.relative_to(ROOT)}")
    current_interactions = build_control_surface_axis_interactions()
    stored_interactions = load_json(AXIS_INTERACTIONS_PATH)
    if stored_interactions != current_interactions:
        fail(
            "control-surface axis interactions are stale; regenerate with "
            "python code\\control_surface_axis_interactions.py --write"
        )
    validate_axis_interactions_payload(stored_interactions)
    return stored_interactions["summary"]


def validate_coverage_gaps_layer() -> dict[str, Any]:
    if not COVERAGE_GAPS_PATH.exists():
        fail(f"missing coverage gaps {COVERAGE_GAPS_PATH.relative_to(ROOT)}")
    current_gaps = build_control_surface_coverage_gaps()
    stored_gaps = load_json(COVERAGE_GAPS_PATH)
    if stored_gaps != current_gaps:
        fail(
            "control-surface coverage gaps are stale; regenerate with "
            "python code\\control_surface_coverage_gaps.py --write"
        )
    validate_coverage_gaps_payload(stored_gaps)
    return stored_gaps["summary"]


def validate_gap_closure_plan_layer() -> dict[str, Any]:
    if not GAP_CLOSURE_PLAN_PATH.exists():
        fail(f"missing gap closure plan {GAP_CLOSURE_PLAN_PATH.relative_to(ROOT)}")
    current_plan = build_control_surface_gap_closure_plan()
    stored_plan = load_json(GAP_CLOSURE_PLAN_PATH)
    if stored_plan != current_plan:
        fail(
            "control-surface gap closure plan is stale; regenerate with "
            "python code\\control_surface_gap_closure_plan.py --write"
        )
    validate_gap_closure_plan_payload(stored_plan)
    return stored_plan["summary"]


def validate_offensive_doctrine_layer() -> dict[str, Any]:
    if not OFFENSIVE_DOCTRINE_PATH.exists():
        fail(
            f"missing offensive doctrine {OFFENSIVE_DOCTRINE_PATH.relative_to(ROOT)}"
        )
    current_doctrine = build_control_surface_offensive_doctrine()
    stored_doctrine = load_json(OFFENSIVE_DOCTRINE_PATH)
    if stored_doctrine != current_doctrine:
        fail(
            "control-surface offensive doctrine is stale; regenerate with "
            "python code\\control_surface_offensive_doctrine.py --write"
        )
    validate_offensive_doctrine_payload(stored_doctrine)
    return stored_doctrine["summary"]


def validate_transfer_width_probe_layer() -> dict[str, Any]:
    if not TRANSFER_WIDTH_PROBE_PATH.exists():
        fail(
            f"missing transfer width probe {TRANSFER_WIDTH_PROBE_PATH.relative_to(ROOT)}"
        )
    current_probe = build_transfer_width_probe()
    stored_probe = load_json(TRANSFER_WIDTH_PROBE_PATH)
    if stored_probe != current_probe:
        fail(
            "transfer width probe is stale; regenerate with "
            "python code\\transfer_width_probe_mc005_mc003_mc004.py --write"
        )
    validate_transfer_width_probe_payload(stored_probe)
    return {
        "row_count": len(stored_probe["row_roles"]),
        "panel_count": len(stored_probe["probe_panels"]),
        "work_order": stored_probe["work_order"]["id"],
        "non_qwen_targets": stored_probe["target_models"][
            "primary_non_qwen_targets"
        ],
    }


def validate_singleton_stage_pack_layer() -> dict[str, Any]:
    if not SINGLETON_STAGE_PACK_PATH.exists():
        fail(
            f"missing singleton stage pack {SINGLETON_STAGE_PACK_PATH.relative_to(ROOT)}"
        )
    current_pack = build_singleton_stage_replication_pack()
    stored_pack = load_json(SINGLETON_STAGE_PACK_PATH)
    if stored_pack != current_pack:
        fail(
            "singleton stage replication pack is stale; regenerate with "
            "python code\\singleton_stage_replication_pack.py --write"
        )
    validate_singleton_stage_pack_payload(stored_pack)
    return {
        "stage_targets": stored_pack["summary"]["stage_target_count"],
        "proposals": stored_pack["summary"]["proposal_count"],
        "work_order": stored_pack["work_order"]["id"],
        "target_terminal_stages": stored_pack["summary"][
            "target_terminal_stages"
        ],
    }


def validate_post_mc033_bridge_closeout_layer() -> dict[str, Any]:
    if not POST_MC033_BRIDGE_CLOSEOUT_PATH.exists():
        fail(
            "missing post-MC033 bridge closeout "
            f"{POST_MC033_BRIDGE_CLOSEOUT_PATH.relative_to(ROOT)}"
        )
    current_closeout = build_post_mc033_bridge_closeout_audit()
    stored_closeout = load_json(POST_MC033_BRIDGE_CLOSEOUT_PATH)
    if stored_closeout != current_closeout:
        fail(
            "post-MC033 bridge closeout is stale; regenerate with "
            "python code\\post_mc033_bridge_closeout_audit.py --write"
        )
    validate_post_mc033_closeout_payload(stored_closeout)
    return {
        "same_family_sequence": stored_closeout["summary"][
            "same_family_sequence_ids"
        ],
        "recent_closed_rung_ids": stored_closeout["summary"][
            "recent_closed_rung_ids"
        ],
        "work_order": stored_closeout["work_order"]["id"],
        "same_family_route_status": stored_closeout["summary"][
            "same_family_route_status"
        ],
        "hidden_state_allowed": stored_closeout["summary"][
            "hidden_state_allowed_count"
        ],
        "clean_unconfounded": stored_closeout["summary"][
            "clean_unconfounded_bridge_count"
        ],
    }


def validate_mc005_reference_specimen_layer() -> dict[str, Any]:
    if not MC005_REFERENCE_SPECIMEN_AUDIT_PATH.exists():
        fail(
            "missing MC005 reference specimen audit "
            f"{MC005_REFERENCE_SPECIMEN_AUDIT_PATH.relative_to(ROOT)}"
        )
    current_audit = build_mc005_reference_specimen_audit()
    stored_audit = load_json(MC005_REFERENCE_SPECIMEN_AUDIT_PATH)
    if stored_audit != current_audit:
        fail(
            "MC005 reference specimen audit is stale; regenerate with "
            "python code\\mc005_reference_specimen_audit.py --write"
        )
    validate_mc005_reference_payload(stored_audit)
    return {
        "row_id": stored_audit["summary"]["row_id"],
        "verdict": stored_audit["summary"]["verdict"],
        "route_status": stored_audit["summary"]["route_status"],
        "terminal_stage": stored_audit["summary"]["terminal_stage"],
        "reliability_class": stored_audit["summary"]["reliability_class"],
        "transfer_class": stored_audit["summary"]["transfer_class"],
        "lookup_write_effect_exact": stored_audit["summary"][
            "lookup_write_effect_exact"
        ],
        "strict_answer_absent_nulls_clean": stored_audit["summary"][
            "strict_answer_absent_nulls_clean"
        ],
        "combined_null_flip_count": stored_audit["summary"][
            "combined_null_flip_count"
        ],
    }


def validate_mc006_predecision_frontier_layer() -> dict[str, Any]:
    if not MC006_PREDECISION_FRONTIER_AUDIT_PATH.exists():
        fail(
            "missing MC006 predecision frontier audit "
            f"{MC006_PREDECISION_FRONTIER_AUDIT_PATH.relative_to(ROOT)}"
        )
    current_audit = build_mc006_predecision_frontier_audit()
    stored_audit = load_json(MC006_PREDECISION_FRONTIER_AUDIT_PATH)
    if stored_audit != current_audit:
        fail(
            "MC006 predecision frontier audit is stale; regenerate with "
            "python code\\mc006_predecision_frontier_audit.py --write"
        )
    validate_mc006_frontier_payload(stored_audit)
    return {
        "row_id": stored_audit["summary"]["row_id"],
        "verdict": stored_audit["summary"]["verdict"],
        "route_status": stored_audit["summary"]["route_status"],
        "frontier_class": stored_audit["summary"]["frontier_class"],
        "terminal_stage": stored_audit["summary"]["terminal_stage"],
        "reliability_class": stored_audit["summary"]["reliability_class"],
        "transfer_class": stored_audit["summary"]["transfer_class"],
        "behavior_substrate_passed": stored_audit["summary"][
            "behavior_substrate_passed"
        ],
        "predecision_monitor_supported": stored_audit["summary"][
            "predecision_monitor_supported"
        ],
        "promotion_gate_passed": stored_audit["summary"]["promotion_gate_passed"],
    }


def validate_compositional_genome_audit_layer() -> dict[str, Any]:
    if not COMPOSITIONAL_GENOME_AUDIT_PATH.exists():
        fail(
            "missing compositional genome audit "
            f"{COMPOSITIONAL_GENOME_AUDIT_PATH.relative_to(ROOT)}"
        )
    current_audit = build_control_surface_compositional_genome_audit()
    stored_audit = load_json(COMPOSITIONAL_GENOME_AUDIT_PATH)
    if stored_audit != current_audit:
        fail(
            "compositional genome audit is stale; regenerate with "
            "python code\\control_surface_compositional_genome_audit.py --write"
        )
    validate_compositional_genome_payload(stored_audit)
    return {
        "row_count": stored_audit["summary"]["row_count"],
        "dominant_current_object": stored_audit["summary"][
            "dominant_current_object"
        ],
        "prompt_contract_visible_count": stored_audit["summary"][
            "prompt_contract_visible_count"
        ],
        "output_geometry_visible_count": stored_audit["summary"][
            "output_geometry_visible_count"
        ],
        "source_or_prompt_token_dependent_count": stored_audit["summary"][
            "source_or_prompt_token_dependent_count"
        ],
        "behavior_or_bridge_substrate_blocked_count": stored_audit["summary"][
            "behavior_or_bridge_substrate_blocked_count"
        ],
        "internal_monitor_present_count": stored_audit["summary"][
            "internal_monitor_present_count"
        ],
        "internal_causal_surface_count": stored_audit["summary"][
            "internal_causal_surface_count"
        ],
        "full_reliability_mechanism_count": stored_audit["summary"][
            "full_reliability_mechanism_count"
        ],
        "transfer_ready_mechanism_count": stored_audit["summary"][
            "transfer_ready_mechanism_count"
        ],
        "clean_unconfounded_bridge_substrate_count": stored_audit["summary"][
            "clean_unconfounded_bridge_substrate_count"
        ],
        "promoted_mechanism_count": stored_audit["summary"][
            "promoted_mechanism_count"
        ],
    }


def validate_family_matrix_layer() -> dict[str, Any]:
    if not FAMILY_MATRIX_PATH.exists():
        fail(f"missing family matrix {FAMILY_MATRIX_PATH.relative_to(ROOT)}")
    current_matrix = build_control_surface_family_matrix()
    stored_matrix = load_json(FAMILY_MATRIX_PATH)
    if stored_matrix != current_matrix:
        fail(
            "control-surface family matrix is stale; regenerate with "
            "python code\\control_surface_family_matrix.py --write"
        )
    validate_family_matrix_payload(stored_matrix)
    return {
        "row_count": stored_matrix["summary"]["row_count"],
        "matrix_column_count": stored_matrix["summary"]["matrix_column_count"],
        "promotion_ready_row_count": stored_matrix["summary"][
            "promotion_ready_row_count"
        ],
        "bounded_reference_rows": stored_matrix["summary"][
            "bounded_reference_rows"
        ],
        "monitor_only_rows": stored_matrix["summary"]["monitor_only_rows"],
        "primary_blockers": stored_matrix["summary"]["primary_blocker_counts"],
        "terminal_stages": stored_matrix["summary"]["terminal_stage_counts"],
        "boolean_axis_counts": stored_matrix["summary"]["boolean_axis_counts"],
    }


def validate_knowledge_ladder_layer() -> dict[str, Any]:
    if not KNOWLEDGE_LADDER_PATH.exists():
        fail(
            "missing knowledge ladder "
            f"{KNOWLEDGE_LADDER_PATH.relative_to(ROOT)}"
        )
    current_ladder = build_control_surface_knowledge_ladder()
    stored_ladder = load_json(KNOWLEDGE_LADDER_PATH)
    if stored_ladder != current_ladder:
        fail(
            "control-surface knowledge ladder is stale; regenerate with "
            "python code\\control_surface_knowledge_ladder.py --write"
        )
    validate_knowledge_ladder_payload(stored_ladder)
    return {
        "level_count": stored_ladder["summary"]["level_count"],
        "ladder_row_count": stored_ladder["summary"]["ladder_row_count"],
        "auxiliary_row_count": stored_ladder["summary"]["auxiliary_row_count"],
        "status_counts": stored_ladder["summary"]["status_counts"],
        "bounded_reference_level_count": stored_ladder["summary"][
            "bounded_reference_level_count"
        ],
        "monitor_only_level_count": stored_ladder["summary"][
            "monitor_only_level_count"
        ],
        "promoted_level_count": stored_ladder["summary"]["promoted_level_count"],
        "real_abstention_uncertainty_ready_count": stored_ladder["summary"][
            "real_abstention_uncertainty_ready_count"
        ],
        "bridge_hidden_state_allowed_count": stored_ladder["summary"][
            "bridge_hidden_state_allowed_count"
        ],
        "bridge_clean_unconfounded_count": stored_ladder["summary"][
            "bridge_clean_unconfounded_count"
        ],
    }


def validate_knowledge_gap_plan_layer() -> dict[str, Any]:
    if not KNOWLEDGE_GAP_PLAN_PATH.exists():
        fail(
            "missing knowledge gap plan "
            f"{KNOWLEDGE_GAP_PLAN_PATH.relative_to(ROOT)}"
        )
    current_plan = build_control_surface_knowledge_gap_plan()
    stored_plan = load_json(KNOWLEDGE_GAP_PLAN_PATH)
    if stored_plan != current_plan:
        fail(
            "control-surface knowledge gap plan is stale; regenerate with "
            "python code\\control_surface_knowledge_gap_plan.py --write"
        )
    validate_knowledge_gap_plan_payload(stored_plan)
    return {
        "level_count": stored_plan["summary"]["level_count"],
        "missing_evidence_item_count": stored_plan["summary"][
            "missing_evidence_item_count"
        ],
        "linked_existing_work_order_count": stored_plan["summary"][
            "linked_existing_work_order_count"
        ],
        "linked_existing_work_order_ids": stored_plan["summary"][
            "linked_existing_work_order_ids"
        ],
        "new_work_order_required_level_count": stored_plan["summary"][
            "new_work_order_required_level_count"
        ],
        "levels_requiring_new_behavior_substrate_count": stored_plan["summary"][
            "levels_requiring_new_behavior_substrate_count"
        ],
        "levels_requiring_predecision_or_reliability_count": stored_plan["summary"][
            "levels_requiring_predecision_or_reliability_count"
        ],
        "new_hidden_state_search_allowed_level_count": stored_plan["summary"][
            "new_hidden_state_search_allowed_level_count"
        ],
        "same_family_route_killed_level_count": stored_plan["summary"][
            "same_family_route_killed_level_count"
        ],
        "decision_state_counts": stored_plan["summary"]["decision_state_counts"],
        "promoted_level_count": stored_plan["summary"]["promoted_level_count"],
        "real_abstention_uncertainty_ready_count": stored_plan["summary"][
            "real_abstention_uncertainty_ready_count"
        ],
    }


def validate_knowledge_substrate_admission_layer() -> dict[str, Any]:
    if not KNOWLEDGE_SUBSTRATE_ADMISSION_PATH.exists():
        fail(
            "missing knowledge substrate admission "
            f"{KNOWLEDGE_SUBSTRATE_ADMISSION_PATH.relative_to(ROOT)}"
        )
    current_admission = build_control_surface_knowledge_substrate_admission()
    stored_admission = load_json(KNOWLEDGE_SUBSTRATE_ADMISSION_PATH)
    if stored_admission != current_admission:
        fail(
            "control-surface knowledge substrate admission is stale; regenerate with "
            "python code\\control_surface_knowledge_substrate_admission.py --write"
        )
    validate_knowledge_admission_payload(stored_admission)
    return {
        "packet_count": stored_admission["summary"]["packet_count"],
        "gate_count": stored_admission["summary"]["gate_count"],
        "unique_gate_count": stored_admission["summary"]["unique_gate_count"],
        "admission_classes": stored_admission["summary"]["admission_classes"],
        "levels_requiring_new_behavior_substrate_count": stored_admission[
            "summary"
        ]["levels_requiring_new_behavior_substrate_count"],
        "new_work_order_required_level_count": stored_admission["summary"][
            "new_work_order_required_level_count"
        ],
        "new_hidden_state_search_allowed_level_count": stored_admission[
            "summary"
        ]["new_hidden_state_search_allowed_level_count"],
        "same_family_route_killed_level_count": stored_admission["summary"][
            "same_family_route_killed_level_count"
        ],
        "bridge_hidden_state_allowed_count": stored_admission["summary"][
            "bridge_hidden_state_allowed_count"
        ],
        "bridge_clean_unconfounded_count": stored_admission["summary"][
            "bridge_clean_unconfounded_count"
        ],
        "smoke_hidden_state_allowed_count": stored_admission["summary"][
            "smoke_hidden_state_allowed_count"
        ],
        "promoted_mechanism_count": stored_admission["summary"][
            "promoted_mechanism_count"
        ],
    }


def validate_knowledge_candidate_queue_layer() -> dict[str, Any]:
    if not KNOWLEDGE_CANDIDATE_QUEUE_PATH.exists():
        fail(
            "missing knowledge candidate queue "
            f"{KNOWLEDGE_CANDIDATE_QUEUE_PATH.relative_to(ROOT)}"
        )
    current_queue = build_control_surface_knowledge_candidate_queue()
    stored_queue = load_json(KNOWLEDGE_CANDIDATE_QUEUE_PATH)
    if stored_queue != current_queue:
        fail(
            "control-surface knowledge candidate queue is stale; regenerate with "
            "python code\\control_surface_knowledge_candidate_queue.py --write"
        )
    validate_knowledge_candidate_payload(stored_queue)
    return {
        "candidate_count": stored_queue["summary"]["candidate_count"],
        "admission_packet_count": stored_queue["summary"][
            "admission_packet_count"
        ],
        "admission_gate_count": stored_queue["summary"]["admission_gate_count"],
        "total_gate_binding_count": stored_queue["summary"][
            "total_gate_binding_count"
        ],
        "hidden_state_candidate_count": stored_queue["summary"][
            "hidden_state_candidate_count"
        ],
        "candidate_count_by_level": stored_queue["summary"][
            "candidate_count_by_level"
        ],
        "candidate_count_by_admission_class": stored_queue["summary"][
            "candidate_count_by_admission_class"
        ],
        "priority_counts": stored_queue["summary"]["priority_counts"],
        "linked_next_queue_id_count": stored_queue["summary"][
            "linked_next_queue_id_count"
        ],
        "promoted_mechanism_count": stored_queue["summary"][
            "promoted_mechanism_count"
        ],
    }


def validate_knowledge_first_run_pack_layer() -> dict[str, Any]:
    if not KNOWLEDGE_FIRST_RUN_PACK_PATH.exists():
        fail(
            "missing knowledge first-run pack "
            f"{KNOWLEDGE_FIRST_RUN_PACK_PATH.relative_to(ROOT)}"
        )
    current_pack = build_control_surface_knowledge_first_run_pack()
    stored_pack = load_json(KNOWLEDGE_FIRST_RUN_PACK_PATH)
    if stored_pack != current_pack:
        fail(
            "control-surface knowledge first-run pack is stale; regenerate with "
            "python code\\control_surface_knowledge_first_run_pack.py --write"
        )
    validate_knowledge_first_run_payload(stored_pack)
    return {
        "packet_count": stored_pack["summary"]["packet_count"],
        "candidate_count": stored_pack["summary"]["candidate_count"],
        "admission_gate_count": stored_pack["summary"]["admission_gate_count"],
        "total_panel_count": stored_pack["summary"]["total_panel_count"],
        "total_baseline_count": stored_pack["summary"]["total_baseline_count"],
        "total_gate_binding_count": stored_pack["summary"][
            "total_gate_binding_count"
        ],
        "hidden_state_packet_count": stored_pack["summary"][
            "hidden_state_packet_count"
        ],
        "packet_count_by_level": stored_pack["summary"]["packet_count_by_level"],
        "packet_count_by_admission_class": stored_pack["summary"][
            "packet_count_by_admission_class"
        ],
        "priority_counts": stored_pack["summary"]["priority_counts"],
        "primary_model": stored_pack["summary"]["primary_model"],
        "secondary_model": stored_pack["summary"]["secondary_model"],
        "promoted_mechanism_count": stored_pack["summary"][
            "promoted_mechanism_count"
        ],
    }


def validate_knowledge_first_run_outcomes_layer() -> dict[str, Any]:
    if not KNOWLEDGE_FIRST_RUN_OUTCOMES_PATH.exists():
        fail(
            "missing knowledge first-run outcomes "
            f"{KNOWLEDGE_FIRST_RUN_OUTCOMES_PATH.relative_to(ROOT)}"
        )
    current_outcomes = build_control_surface_knowledge_first_run_outcomes()
    stored_outcomes = load_json(KNOWLEDGE_FIRST_RUN_OUTCOMES_PATH)
    if stored_outcomes != current_outcomes:
        fail(
            "control-surface knowledge first-run outcomes are stale; regenerate with "
            "python code\\control_surface_knowledge_first_run_outcomes.py --write"
        )
    validate_knowledge_first_run_outcomes_payload(stored_outcomes)
    return {
        "outcome_count": stored_outcomes["summary"]["outcome_count"],
        "first_run_packet_count": stored_outcomes["summary"][
            "first_run_packet_count"
        ],
        "structural_passed_count": stored_outcomes["summary"][
            "structural_passed_count"
        ],
        "terminal_gate_counts": stored_outcomes["summary"][
            "terminal_gate_counts"
        ],
        "level_counts": stored_outcomes["summary"]["level_counts"],
        "diagnostic_class_counts": stored_outcomes["summary"][
            "diagnostic_class_counts"
        ],
        "exported_diagnostic_class_counts": stored_outcomes["summary"][
            "exported_diagnostic_class_counts"
        ],
        "behavior_ready_count": stored_outcomes["summary"][
            "behavior_ready_count"
        ],
        "hidden_state_allowed_count": stored_outcomes["summary"][
            "hidden_state_allowed_count"
        ],
        "promotion_ready_count": stored_outcomes["summary"][
            "promotion_ready_count"
        ],
    }


def validate_knowledge_failure_topology_layer() -> dict[str, Any]:
    if not KNOWLEDGE_FAILURE_TOPOLOGY_PATH.exists():
        fail(
            "missing knowledge failure topology "
            f"{KNOWLEDGE_FAILURE_TOPOLOGY_PATH.relative_to(ROOT)}"
        )
    current_topology = build_control_surface_knowledge_failure_topology()
    stored_topology = load_json(KNOWLEDGE_FAILURE_TOPOLOGY_PATH)
    if stored_topology != current_topology:
        fail(
            "control-surface knowledge failure topology is stale; regenerate with "
            "python code\\control_surface_knowledge_failure_topology.py --write"
        )
    validate_knowledge_failure_topology_payload(stored_topology)
    return {
        "topology_node_count": stored_topology["summary"]["topology_node_count"],
        "work_order_count": stored_topology["summary"]["work_order_count"],
        "outcome_count": stored_topology["summary"]["outcome_count"],
        "candidate_coverage_count": stored_topology["summary"][
            "candidate_coverage_count"
        ],
        "diagnostic_coverage_count": stored_topology["summary"][
            "diagnostic_coverage_count"
        ],
        "node_type_counts": stored_topology["summary"]["node_type_counts"],
        "second_wave_decision_counts": stored_topology["summary"][
            "second_wave_decision_counts"
        ],
        "work_order_priority_counts": stored_topology["summary"][
            "work_order_priority_counts"
        ],
        "work_order_track_counts": stored_topology["summary"][
            "work_order_track_counts"
        ],
        "behavior_ready_count": stored_topology["summary"][
            "behavior_ready_count"
        ],
        "hidden_state_allowed_count": stored_topology["summary"][
            "hidden_state_allowed_count"
        ],
    }


def validate_knowledge_second_wave_outcomes_layer() -> dict[str, Any]:
    if not SECOND_WAVE_OUTCOMES_PATH.exists():
        fail(
            "missing knowledge second-wave outcomes "
            f"{SECOND_WAVE_OUTCOMES_PATH.relative_to(ROOT)}"
        )
    current_outcomes = build_control_surface_knowledge_second_wave_outcomes()
    stored_outcomes = load_json(SECOND_WAVE_OUTCOMES_PATH)
    if stored_outcomes != current_outcomes:
        fail(
            "control-surface knowledge second-wave outcomes are stale; regenerate with "
            "python code\\control_surface_knowledge_second_wave_outcomes.py --write"
        )
    validate_knowledge_second_wave_outcomes_payload(stored_outcomes)
    return {
        "outcome_count": stored_outcomes["summary"]["outcome_count"],
        "completed_work_order_count": stored_outcomes["summary"][
            "completed_work_order_count"
        ],
        "pending_work_order_count": stored_outcomes["summary"][
            "pending_work_order_count"
        ],
        "killed_route_count": stored_outcomes["summary"]["killed_route_count"],
        "route_decision_counts": stored_outcomes["summary"][
            "route_decision_counts"
        ],
        "verdict_counts": stored_outcomes["summary"][
            "verdict_counts"
        ],
        "exported_diagnostic_class_counts": stored_outcomes["summary"][
            "exported_diagnostic_class_counts"
        ],
        "behavior_ready_count": stored_outcomes["summary"][
            "behavior_ready_count"
        ],
        "signature_screen_allowed_count": stored_outcomes["summary"][
            "signature_screen_allowed_count"
        ],
        "hidden_state_claim_allowed_count": stored_outcomes["summary"][
            "hidden_state_claim_allowed_count"
        ],
        "intervention_allowed_count": stored_outcomes["summary"][
            "intervention_allowed_count"
        ],
        "pending_work_orders": stored_outcomes["summary"]["pending_work_orders"],
    }


def validate_knowledge_third_wave_outcomes_layer() -> dict[str, Any]:
    if not THIRD_WAVE_OUTCOMES_PATH.exists():
        fail(
            "missing knowledge third-wave outcomes "
            f"{THIRD_WAVE_OUTCOMES_PATH.relative_to(ROOT)}"
        )
    current_outcomes = build_control_surface_knowledge_third_wave_outcomes()
    stored_outcomes = load_json(THIRD_WAVE_OUTCOMES_PATH)
    if stored_outcomes != current_outcomes:
        fail(
            "control-surface knowledge third-wave outcomes are stale; regenerate with "
            "python code\\control_surface_knowledge_third_wave_outcomes.py --write"
        )
    validate_knowledge_third_wave_outcomes_payload(stored_outcomes)
    return stored_outcomes["summary"]


def validate_knowledge_fourth_wave_outcomes_layer() -> dict[str, Any]:
    if not FOURTH_WAVE_OUTCOMES_PATH.exists():
        fail(
            "missing knowledge fourth-wave outcomes "
            f"{FOURTH_WAVE_OUTCOMES_PATH.relative_to(ROOT)}"
        )
    current_outcomes = build_control_surface_knowledge_fourth_wave_outcomes()
    stored_outcomes = load_json(FOURTH_WAVE_OUTCOMES_PATH)
    if stored_outcomes != current_outcomes:
        fail(
            "control-surface knowledge fourth-wave outcomes are stale; regenerate with "
            "python code\\control_surface_knowledge_fourth_wave_outcomes.py --write"
        )
    validate_knowledge_fourth_wave_outcomes_payload(stored_outcomes)
    return stored_outcomes["summary"]


def validate_knowledge_fifth_wave_outcomes_layer() -> dict[str, Any]:
    if not FIFTH_WAVE_OUTCOMES_PATH.exists():
        fail(
            "missing knowledge fifth-wave outcomes "
            f"{FIFTH_WAVE_OUTCOMES_PATH.relative_to(ROOT)}"
        )
    current_outcomes = build_control_surface_knowledge_fifth_wave_outcomes()
    stored_outcomes = load_json(FIFTH_WAVE_OUTCOMES_PATH)
    if stored_outcomes != current_outcomes:
        fail(
            "control-surface knowledge fifth-wave outcomes are stale; regenerate with "
            "python code\\control_surface_knowledge_fifth_wave_outcomes.py --write"
        )
    validate_knowledge_fifth_wave_outcomes_payload(stored_outcomes)
    return stored_outcomes["summary"]


def validate_knowledge_sixth_wave_outcomes_layer() -> dict[str, Any]:
    if not SIXTH_WAVE_OUTCOMES_PATH.exists():
        fail(
            "missing knowledge sixth-wave outcomes "
            f"{SIXTH_WAVE_OUTCOMES_PATH.relative_to(ROOT)}"
        )
    current_outcomes = build_control_surface_knowledge_sixth_wave_outcomes()
    stored_outcomes = load_json(SIXTH_WAVE_OUTCOMES_PATH)
    if stored_outcomes != current_outcomes:
        fail(
            "control-surface knowledge sixth-wave outcomes are stale; regenerate with "
            "python code\\control_surface_knowledge_sixth_wave_outcomes.py --write"
        )
    validate_knowledge_sixth_wave_outcomes_payload(stored_outcomes)
    return stored_outcomes["summary"]


def validate_knowledge_seventh_wave_outcomes_layer() -> dict[str, Any]:
    if not SEVENTH_WAVE_OUTCOMES_PATH.exists():
        fail(
            "missing knowledge seventh-wave outcomes "
            f"{SEVENTH_WAVE_OUTCOMES_PATH.relative_to(ROOT)}"
        )
    current_outcomes = build_control_surface_knowledge_seventh_wave_outcomes()
    stored_outcomes = load_json(SEVENTH_WAVE_OUTCOMES_PATH)
    if stored_outcomes != current_outcomes:
        fail(
            "control-surface knowledge seventh-wave outcomes are stale; regenerate with "
            "python code\\control_surface_knowledge_seventh_wave_outcomes.py --write"
        )
    validate_knowledge_seventh_wave_outcomes_payload(stored_outcomes)
    return stored_outcomes["summary"]


def validate_knowledge_eighth_wave_outcomes_layer() -> dict[str, Any]:
    if not EIGHTH_WAVE_OUTCOMES_PATH.exists():
        fail(
            "missing knowledge eighth-wave outcomes "
            f"{EIGHTH_WAVE_OUTCOMES_PATH.relative_to(ROOT)}"
        )
    current_outcomes = build_control_surface_knowledge_eighth_wave_outcomes()
    stored_outcomes = load_json(EIGHTH_WAVE_OUTCOMES_PATH)
    if stored_outcomes != current_outcomes:
        fail(
            "control-surface knowledge eighth-wave outcomes are stale; regenerate with "
            "python code\\control_surface_knowledge_eighth_wave_outcomes.py --write"
        )
    validate_knowledge_eighth_wave_outcomes_payload(stored_outcomes)
    return stored_outcomes["summary"]


def validate_knowledge_ninth_wave_outcomes_layer() -> dict[str, Any]:
    if not NINTH_WAVE_OUTCOMES_PATH.exists():
        fail(
            "missing knowledge ninth-wave outcomes "
            f"{NINTH_WAVE_OUTCOMES_PATH.relative_to(ROOT)}"
        )
    current_outcomes = build_control_surface_knowledge_ninth_wave_outcomes()
    stored_outcomes = load_json(NINTH_WAVE_OUTCOMES_PATH)
    if stored_outcomes != current_outcomes:
        fail(
            "control-surface knowledge ninth-wave outcomes are stale; regenerate with "
            "python code\\control_surface_knowledge_ninth_wave_outcomes.py --write"
        )
    validate_knowledge_ninth_wave_outcomes_payload(stored_outcomes)
    return stored_outcomes["summary"]


def validate_knowledge_tenth_wave_outcomes_layer() -> dict[str, Any]:
    if not TENTH_WAVE_OUTCOMES_PATH.exists():
        fail(
            "missing knowledge tenth-wave outcomes "
            f"{TENTH_WAVE_OUTCOMES_PATH.relative_to(ROOT)}"
        )
    current_outcomes = build_control_surface_knowledge_tenth_wave_outcomes()
    stored_outcomes = load_json(TENTH_WAVE_OUTCOMES_PATH)
    if stored_outcomes != current_outcomes:
        fail(
            "control-surface knowledge tenth-wave outcomes are stale; regenerate with "
            "python code\\control_surface_knowledge_tenth_wave_outcomes.py --write"
        )
    validate_knowledge_tenth_wave_outcomes_payload(stored_outcomes)
    return stored_outcomes["summary"]


def validate_knowledge_eleventh_wave_outcomes_layer() -> dict[str, Any]:
    if not ELEVENTH_WAVE_OUTCOMES_PATH.exists():
        fail(
            "missing knowledge eleventh-wave outcomes "
            f"{ELEVENTH_WAVE_OUTCOMES_PATH.relative_to(ROOT)}"
        )
    current_outcomes = build_control_surface_knowledge_eleventh_wave_outcomes()
    stored_outcomes = load_json(ELEVENTH_WAVE_OUTCOMES_PATH)
    if stored_outcomes != current_outcomes:
        fail(
            "control-surface knowledge eleventh-wave outcomes are stale; regenerate with "
            "python code\\control_surface_knowledge_eleventh_wave_outcomes.py --write"
        )
    validate_knowledge_eleventh_wave_outcomes_payload(stored_outcomes)
    return stored_outcomes["summary"]


def validate_ksq001_first_run_result_layer() -> dict[str, Any]:
    if not KSQ001_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ001 first-run result "
            f"{KSQ001_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ001_FIRST_RUN_STATUS_PATH.exists():
        fail(
            "missing KSQ001 first-run status card "
            f"{KSQ001_FIRST_RUN_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ001_FIRST_RUN_PREREG_PATH.exists():
        fail(
            "missing KSQ001 first-run preregistration "
            f"{KSQ001_FIRST_RUN_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ001_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ001 first-run result schema_version must be 1")
    if result.get("candidate_id") != "ksq001_familiar_entity_prior_counterbalance":
        fail("KSQ001 first-run result candidate_id changed")
    if result.get("run_type") != "ksq001_familiar_entity_prior_counterbalance_structural_gate":
        fail("KSQ001 first-run result must be the structural gate result")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ001 first-run result must not license hidden-state work")
    structural = result.get("structural")
    if not isinstance(structural, dict):
        fail("KSQ001 first-run result missing structural payload")
    criteria = structural.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ001 first-run result missing structural criteria")
    failed_criteria = [key for key, value in criteria.items() if value is not True]
    if failed_criteria:
        fail(f"KSQ001 first-run structural criteria failed: {failed_criteria}")
    if structural.get("record_count") != 480:
        fail("KSQ001 first-run result must keep the full 480-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ001 first-run result must keep the full 40-source substrate")
    if structural.get("panel_counts") != {
        "answer_absent_irrelevant_nulls": 120,
        "semantic_prior_direct_control": 120,
        "semantic_prior_lure": 120,
        "source_local_artificial_lookup": 120,
    }:
        fail("KSQ001 first-run panel counts changed")
    if structural.get("template_counts") != {
        "association_question": 160,
        "compact_question": 160,
        "registry_question": 160,
    }:
        fail("KSQ001 first-run template counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ001 first-run source split changed")
    return {
        "status": "structural_passed",
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": result["hidden_state_allowed"],
    }


def validate_ksq001_smoke_result_layer() -> dict[str, Any]:
    if not KSQ001_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ001 10-source smoke result "
            f"{KSQ001_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ001_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ001 smoke result schema_version must be 1")
    if result.get("candidate_id") != "ksq001_familiar_entity_prior_counterbalance":
        fail("KSQ001 smoke result candidate_id changed")
    if result.get("run_type") != "ksq001_familiar_entity_prior_counterbalance_behavior":
        fail("KSQ001 smoke result must be a behavior run")
    if result.get("limit_sources") != 10:
        fail("KSQ001 smoke result must remain the 10-source smoke")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ001 smoke result must not license hidden-state work")
    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ001 smoke result missing summary")
    if summary.get("diagnostic_class") != "smoke_behavior_contrast_candidate":
        fail("KSQ001 smoke diagnostic class changed")
    if summary.get("behavior_ready") is not False:
        fail("KSQ001 smoke must not be behavior ready")
    if summary.get("behavior_candidate") is not True:
        fail("KSQ001 smoke must remain a behavior candidate")
    if summary.get("hidden_state_allowed") is not False:
        fail("KSQ001 smoke summary must not license hidden-state work")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ001 smoke structural gate must pass")
    if structural.get("record_count") != 120 or structural.get("source_count") != 10:
        fail("KSQ001 smoke must keep the 120-row, 10-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ001 smoke missing criteria")
    expected_criteria = {
        "source_local_artificial_control_passed": True,
        "semantic_prior_direct_control_passed": True,
        "answer_absent_null_passed": True,
        "primary_parseability_at_least_90p": True,
        "semantic_prior_competition_present": True,
        "conflict_neither_branch_above_90p": True,
        "candidate_and_output_margins_reported": True,
    }
    for key, expected_value in expected_criteria.items():
        if criteria.get(key) is not expected_value:
            fail(f"KSQ001 smoke criterion changed: {key}")
    if summary["selection"]["selected_template"] != "compact_question":
        fail("KSQ001 smoke selected template changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "source_local_artificial_lookup",
            "semantic_prior_direct_control",
            "semantic_prior_lure",
            "answer_absent_irrelevant_nulls",
        ]
    }
    expected_panel_counts = {
        "source_local_artificial_lookup": {"artificial_value": 10},
        "semantic_prior_direct_control": {"real_prior": 10},
        "semantic_prior_lure": {
            "artificial_value": 6,
            "real_prior": 1,
            "unknown": 2,
            "unparsed": 1,
        },
        "answer_absent_irrelevant_nulls": {"unknown": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ001 smoke panel label counts changed")
    return {
        "status": "smoke_behavior_contrast_candidate",
        "diagnostic_class": summary["diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
    }


def validate_ksq001_full_behavior_result_layer() -> dict[str, Any]:
    if not KSQ001_FULL_BEHAVIOR_RESULT_PATH.exists():
        fail(
            "missing KSQ001 full behavior result "
            f"{KSQ001_FULL_BEHAVIOR_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ001_FULL_BEHAVIOR_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ001 full behavior result schema_version must be 1")
    if result.get("candidate_id") != "ksq001_familiar_entity_prior_counterbalance":
        fail("KSQ001 full behavior candidate_id changed")
    if result.get("run_type") != "ksq001_familiar_entity_prior_counterbalance_behavior":
        fail("KSQ001 full behavior result must be a behavior run")
    if result.get("limit_sources") is not None:
        fail("KSQ001 full behavior result must use all sources")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ001 full behavior result must not license hidden-state work")
    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ001 full behavior result missing summary")
    if summary.get("diagnostic_class") != "familiar_entity_conflict_parseability_failed":
        fail("KSQ001 full behavior diagnostic class changed")
    if summary.get("behavior_ready") is not False:
        fail("KSQ001 full behavior must not be behavior ready")
    if summary.get("hidden_state_allowed") is not False:
        fail("KSQ001 full behavior summary must not license hidden-state work")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ001 full behavior structural gate must pass")
    if structural.get("record_count") != 480 or structural.get("source_count") != 40:
        fail("KSQ001 full behavior must keep the 480-row, 40-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ001 full behavior missing criteria")
    expected_criteria = {
        "full_source_count_is_40": True,
        "source_local_artificial_control_passed": True,
        "semantic_prior_direct_control_passed": True,
        "answer_absent_null_passed": True,
        "primary_parseability_at_least_90p": False,
        "primary_binary_rows_at_least_40": True,
        "semantic_prior_competition_present": True,
        "conflict_neither_branch_above_90p": True,
        "source_disjoint_holdout_mixture_passed": True,
        "candidate_and_output_margins_reported": True,
    }
    for key, expected_value in expected_criteria.items():
        if criteria.get(key) is not expected_value:
            fail(f"KSQ001 full behavior criterion changed: {key}")
    if summary["selection"]["selected_template"] != "compact_question":
        fail("KSQ001 full behavior selected template changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "source_local_artificial_lookup",
            "semantic_prior_direct_control",
            "semantic_prior_lure",
            "answer_absent_irrelevant_nulls",
        ]
    }
    expected_panel_counts = {
        "source_local_artificial_lookup": {"artificial_value": 40},
        "semantic_prior_direct_control": {
            "lure_value": 1,
            "real_prior": 38,
            "unparsed": 1,
        },
        "semantic_prior_lure": {
            "artificial_value": 18,
            "real_prior": 2,
            "unknown": 4,
            "unparsed": 16,
        },
        "answer_absent_irrelevant_nulls": {"unknown": 38, "unparsed": 2},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ001 full behavior panel label counts changed")
    return {
        "status": "full_behavior_failed_conflict_parseability",
        "diagnostic_class": summary["diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
    }


def validate_ksq002_first_run_result_layer() -> dict[str, Any]:
    if not KSQ002_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ002 first-run result "
            f"{KSQ002_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ002_FIRST_RUN_STATUS_PATH.exists():
        fail(
            "missing KSQ002 first-run status card "
            f"{KSQ002_FIRST_RUN_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ002_FIRST_RUN_PREREG_PATH.exists():
        fail(
            "missing KSQ002 first-run preregistration "
            f"{KSQ002_FIRST_RUN_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ002_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ002 first-run result schema_version must be 1")
    if result.get("candidate_id") != "ksq002_familiar_entity_source_rewrite_equivalence":
        fail("KSQ002 first-run result candidate_id changed")
    if result.get("run_type") != "ksq002_familiar_entity_source_rewrite_equivalence_structural_gate":
        fail("KSQ002 first-run result must be the structural gate result")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ002 first-run result must not license hidden-state work")
    structural = result.get("structural")
    if not isinstance(structural, dict):
        fail("KSQ002 first-run result missing structural payload")
    criteria = structural.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ002 first-run result missing structural criteria")
    failed_criteria = [key for key, value in criteria.items() if value is not True]
    if failed_criteria:
        fail(f"KSQ002 first-run structural criteria failed: {failed_criteria}")
    if structural.get("passed") is not True:
        fail("KSQ002 first-run structural gate must be passed")
    if structural.get("record_count") != 720:
        fail("KSQ002 first-run result must keep the full 720-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ002 first-run result must keep the full 40-source substrate")
    if structural.get("panel_counts") != {
        "baseline_source_value_lookup": 120,
        "neutral_rewrite_lookup": 120,
        "query_only_control": 120,
        "rewrite_output_geometry_audit": 120,
        "source_deletion": 120,
        "source_disjoint_rewrite_holdout": 120,
    }:
        fail("KSQ002 first-run panel counts changed")
    if structural.get("template_counts") != {
        "compact_rewrite": 240,
        "registry_question": 240,
        "sentence_rewrite": 240,
    }:
        fail("KSQ002 first-run template counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ002 first-run source split changed")
    return {
        "status": "structural_passed",
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": result["hidden_state_allowed"],
    }


def validate_ksq002_smoke_result_layer() -> dict[str, Any]:
    if not KSQ002_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ002 10-source smoke result "
            f"{KSQ002_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ002_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ002 smoke result schema_version must be 1")
    if result.get("candidate_id") != "ksq002_familiar_entity_source_rewrite_equivalence":
        fail("KSQ002 smoke result candidate_id changed")
    if result.get("run_type") != "ksq002_familiar_entity_source_rewrite_equivalence_behavior":
        fail("KSQ002 smoke result must be a behavior run")
    if result.get("limit_sources") != 10:
        fail("KSQ002 smoke result must remain the 10-source smoke")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ002 smoke result must not license hidden-state work")
    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ002 smoke result missing summary")
    if summary.get("diagnostic_class") != "smoke_source_rewrite_candidate":
        fail("KSQ002 smoke diagnostic class changed")
    if summary.get("behavior_ready") is not False:
        fail("KSQ002 smoke must not be behavior ready")
    if summary.get("behavior_candidate") is not True:
        fail("KSQ002 smoke must remain a behavior candidate")
    if summary.get("hidden_state_allowed") is not False:
        fail("KSQ002 smoke summary must not license hidden-state work")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ002 smoke structural gate must pass")
    if structural.get("record_count") != 180 or structural.get("source_count") != 10:
        fail("KSQ002 smoke must keep the 180-row, 10-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ002 smoke missing criteria")
    expected_criteria = {
        "selected_prompt_audit_passed": True,
        "baseline_source_value_lookup_passed": True,
        "neutral_rewrite_lookup_passed": True,
        "source_deletion_passed": True,
        "query_only_control_passed": True,
        "source_disjoint_rewrite_holdout_passed": True,
        "candidate_and_output_margins_reported": True,
    }
    for key, expected_value in expected_criteria.items():
        if criteria.get(key) is not expected_value:
            fail(f"KSQ002 smoke criterion changed: {key}")
    if summary["selection"]["selected_template"] != "registry_question":
        fail("KSQ002 smoke selected template changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "baseline_source_value_lookup",
            "neutral_rewrite_lookup",
            "source_deletion",
            "query_only_control",
            "source_disjoint_rewrite_holdout",
            "rewrite_output_geometry_audit",
        ]
    }
    expected_panel_counts = {
        "baseline_source_value_lookup": {"artificial_value": 10},
        "neutral_rewrite_lookup": {"artificial_value": 10},
        "source_deletion": {"unknown": 10},
        "query_only_control": {"unknown": 10},
        "source_disjoint_rewrite_holdout": {"artificial_value": 10},
        "rewrite_output_geometry_audit": {"artificial_value": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ002 smoke panel label counts changed")
    return {
        "status": "smoke_source_rewrite_candidate",
        "diagnostic_class": summary["diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
    }


def validate_ksq002_full_behavior_result_layer() -> dict[str, Any]:
    if not KSQ002_FULL_BEHAVIOR_RESULT_PATH.exists():
        fail(
            "missing KSQ002 full behavior result "
            f"{KSQ002_FULL_BEHAVIOR_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ002_FULL_BEHAVIOR_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ002 full behavior result schema_version must be 1")
    if result.get("candidate_id") != "ksq002_familiar_entity_source_rewrite_equivalence":
        fail("KSQ002 full behavior candidate_id changed")
    if result.get("run_type") != "ksq002_familiar_entity_source_rewrite_equivalence_behavior":
        fail("KSQ002 full behavior result must be a behavior run")
    if result.get("limit_sources") is not None:
        fail("KSQ002 full behavior result must use all sources")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ002 full behavior result must not license hidden-state work")
    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ002 full behavior result missing summary")
    if summary.get("diagnostic_class") != "source_rewrite_holdout_failed":
        fail("KSQ002 full behavior diagnostic class changed")
    if summary.get("behavior_ready") is not False:
        fail("KSQ002 full behavior must not be behavior ready")
    if summary.get("behavior_candidate") is not False:
        fail("KSQ002 full behavior must not remain a behavior candidate")
    if summary.get("hidden_state_allowed") is not False:
        fail("KSQ002 full behavior summary must not license hidden-state work")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ002 full behavior structural gate must pass")
    if structural.get("record_count") != 720 or structural.get("source_count") != 40:
        fail("KSQ002 full behavior must keep the 720-row, 40-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ002 full behavior missing criteria")
    expected_criteria = {
        "selected_prompt_audit_passed": True,
        "baseline_source_value_lookup_passed": True,
        "neutral_rewrite_lookup_passed": True,
        "source_deletion_passed": True,
        "query_only_control_passed": True,
        "source_disjoint_rewrite_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    for key, expected_value in expected_criteria.items():
        if criteria.get(key) is not expected_value:
            fail(f"KSQ002 full behavior criterion changed: {key}")
    if summary["selection"]["selected_template"] != "sentence_rewrite":
        fail("KSQ002 full behavior selected template changed")
    if summary["rewrite_delta_from_baseline"] != 0.07499999999999996:
        fail("KSQ002 full behavior rewrite delta changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "baseline_source_value_lookup",
            "neutral_rewrite_lookup",
            "source_deletion",
            "query_only_control",
            "source_disjoint_rewrite_holdout",
            "rewrite_output_geometry_audit",
        ]
    }
    expected_panel_counts = {
        "baseline_source_value_lookup": {"artificial_value": 39, "unknown": 1},
        "neutral_rewrite_lookup": {"artificial_value": 36, "unparsed": 4},
        "source_deletion": {"unknown": 40},
        "query_only_control": {"unknown": 40},
        "source_disjoint_rewrite_holdout": {"artificial_value": 36, "unparsed": 4},
        "rewrite_output_geometry_audit": {"artificial_value": 36, "unparsed": 4},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ002 full behavior panel label counts changed")
    rewrite_holdout = selected["rewrite_holdout"]
    if rewrite_holdout["label_counts"] != {"artificial_value": 14, "unparsed": 2}:
        fail("KSQ002 full behavior rewrite holdout label counts changed")
    if rewrite_holdout["parseable_rate"] != 0.875:
        fail("KSQ002 full behavior rewrite holdout parseability changed")
    if rewrite_holdout["artificial_value_rate"] != 0.875:
        fail("KSQ002 full behavior rewrite holdout artificial-value rate changed")
    return {
        "status": "full_behavior_failed_source_rewrite_holdout",
        "diagnostic_class": summary["diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "rewrite_delta_from_baseline": summary["rewrite_delta_from_baseline"],
        "rewrite_holdout_parseable_rate": rewrite_holdout["parseable_rate"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
    }


def validate_ksq003_first_run_result_layer() -> dict[str, Any]:
    if not KSQ003_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ003 first-run result "
            f"{KSQ003_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ003_FIRST_RUN_STATUS_PATH.exists():
        fail(
            "missing KSQ003 first-run status card "
            f"{KSQ003_FIRST_RUN_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ003_FIRST_RUN_PREREG_PATH.exists():
        fail(
            "missing KSQ003 first-run preregistration "
            f"{KSQ003_FIRST_RUN_PREREG_PATH.relative_to(ROOT)}"
        )

    result = load_json(KSQ003_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ003 first-run result schema_version must be 1")
    if result.get("candidate_id") != "ksq003_bridge_statusless_evidence_aggregation":
        fail("KSQ003 first-run result candidate_id changed")
    if result.get("run_type") != "ksq003_statusless_evidence_aggregation_structural_gate":
        fail("KSQ003 first-run result must be the structural gate result")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ003 first-run result must not license hidden-state work")

    structural = result.get("structural")
    if not isinstance(structural, dict):
        fail("KSQ003 first-run result missing structural payload")
    criteria = structural.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ003 first-run result missing structural criteria")
    failed_criteria = [key for key, value in criteria.items() if value is not True]
    if failed_criteria:
        fail(f"KSQ003 first-run structural criteria failed: {failed_criteria}")
    if structural.get("passed") is not True:
        fail("KSQ003 first-run structural gate must be passed")
    if structural.get("record_count") != 840:
        fail("KSQ003 first-run result must keep the full 840-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ003 first-run result must keep the full 40-source substrate")

    panel_counts = structural.get("panel_counts")
    expected_panel_counts = {
        "source_local_direct_control": 120,
        "learned_fact_direct_control": 120,
        "all_evidence_fit_conflict": 120,
        "one_evidence_mismatch_conflict": 120,
        "symbol_only_ablation": 120,
        "parity_only_ablation": 120,
        "answer_absent_and_side_null": 120,
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ003 first-run panel counts changed")
    template_counts = structural.get("template_counts")
    if template_counts != {
        "evidence_packet": 280,
        "field_observations": 280,
        "compact_fit": 280,
    }:
        fail("KSQ003 first-run template counts changed")
    split_source_counts = structural.get("split_source_counts")
    if split_source_counts != {"discovery": 24, "calibration": 8, "holdout": 8}:
        fail("KSQ003 first-run source split changed")
    if structural.get("primary_expected_label_counts") != {
        "local_number": 120,
        "atomic_number": 120,
    }:
        fail("KSQ003 first-run primary label balance changed")

    return {
        "status": "structural_passed",
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(panel_counts),
        "template_count": len(template_counts),
        "hidden_state_allowed": result["hidden_state_allowed"],
    }


def validate_ksq003_smoke_result_layer() -> dict[str, Any]:
    if not KSQ003_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ003 10-source smoke result "
            f"{KSQ003_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ003_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ003 smoke result schema_version must be 1")
    if result.get("candidate_id") != "ksq003_bridge_statusless_evidence_aggregation":
        fail("KSQ003 smoke result candidate_id changed")
    if result.get("run_type") != "ksq003_statusless_evidence_aggregation_behavior":
        fail("KSQ003 smoke result must be a behavior run")
    if result.get("limit_sources") != 10:
        fail("KSQ003 smoke result must remain the 10-source smoke")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ003 smoke result must not license hidden-state work")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ003 smoke result missing summary")
    if summary.get("diagnostic_class") != "smoke_only":
        fail("KSQ003 smoke diagnostic class changed")
    if summary.get("behavior_ready") is not False:
        fail("KSQ003 smoke must not be behavior ready")
    if summary.get("hidden_state_allowed") is not False:
        fail("KSQ003 smoke summary must not license hidden-state work")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ003 smoke structural gate must pass")
    if structural.get("record_count") != 210 or structural.get("source_count") != 10:
        fail("KSQ003 smoke must keep the 210-row, 10-source scope")

    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ003 smoke missing criteria")
    expected_criteria = {
        "source_local_direct_control_passed": True,
        "learned_fact_direct_control_passed": True,
        "answer_absent_null_passed": True,
        "all_evidence_fit_local_passed": True,
        "one_evidence_mismatch_atomic_passed": False,
        "single_feature_ablation_unknown_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    for key, expected_value in expected_criteria.items():
        if criteria.get(key) is not expected_value:
            fail(f"KSQ003 smoke criterion changed: {key}")

    selected = summary["selected_template_summary"]
    if summary["selection"]["selected_template"] != "compact_fit":
        fail("KSQ003 smoke selected template changed")
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "source_local_direct_control",
            "learned_fact_direct_control",
            "all_evidence_fit_conflict",
            "one_evidence_mismatch_conflict",
            "symbol_only_ablation",
            "parity_only_ablation",
            "answer_absent_and_side_null",
        ]
    }
    expected_panel_counts = {
        "source_local_direct_control": {"local_number": 10},
        "learned_fact_direct_control": {"atomic_number": 10},
        "all_evidence_fit_conflict": {"local_number": 10},
        "one_evidence_mismatch_conflict": {"local_number": 10},
        "symbol_only_ablation": {"local_number": 10},
        "parity_only_ablation": {"local_number": 10},
        "answer_absent_and_side_null": {"unknown": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ003 smoke panel label counts changed")

    return {
        "status": "smoke_failed_local_table_dominance",
        "diagnostic_class": summary["diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
    }


def validate_ksq004_first_run_result_layer() -> dict[str, Any]:
    if not KSQ004_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ004 first-run result "
            f"{KSQ004_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ004_FIRST_RUN_STATUS_PATH.exists():
        fail(
            "missing KSQ004 first-run status card "
            f"{KSQ004_FIRST_RUN_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ004_FIRST_RUN_PREREG_PATH.exists():
        fail(
            "missing KSQ004 first-run preregistration "
            f"{KSQ004_FIRST_RUN_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ004_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ004 first-run result schema_version must be 1")
    if result.get("candidate_id") != "ksq004_bridge_answer_interface_minimal_pairs":
        fail("KSQ004 first-run result candidate_id changed")
    if result.get("run_type") != "ksq004_bridge_answer_interface_minimal_pairs_structural_gate":
        fail("KSQ004 first-run result must be the structural gate result")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ004 first-run result must not license hidden-state work")

    structural = result.get("structural")
    if not isinstance(structural, dict):
        fail("KSQ004 first-run result missing structural payload")
    criteria = structural.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ004 first-run result missing structural criteria")
    failed_criteria = [key for key, value in criteria.items() if value is not True]
    if failed_criteria:
        fail(f"KSQ004 first-run structural criteria failed: {failed_criteria}")
    if structural.get("passed") is not True:
        fail("KSQ004 first-run structural gate must be passed")
    if structural.get("record_count") != 720:
        fail("KSQ004 first-run result must keep the full 720-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ004 first-run result must keep the full 40-source substrate")
    if structural.get("panel_counts") != {
        "local_learned_direct_controls": 160,
        "matched_minimal_pairs": 160,
        "minimal_pair_conflict": 160,
        "null_and_holdout": 80,
        "side_answer_leakage": 160,
    }:
        fail("KSQ004 first-run panel counts changed")
    if structural.get("template_counts") != {
        "compact_form": 360,
        "question_form": 360,
    }:
        fail("KSQ004 first-run template counts changed")
    if structural.get("subtype_counts") != {
        "answer_absent_null": 80,
        "atomic_branch_conflict": 80,
        "atomic_direct": 80,
        "atomic_side_leakage": 80,
        "local_branch_conflict": 80,
        "local_direct": 80,
        "local_side_leakage": 80,
        "matched_atomic_contract": 80,
        "matched_local_contract": 80,
    }:
        fail("KSQ004 first-run subtype counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ004 first-run source split changed")
    return {
        "status": "structural_passed",
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": result["hidden_state_allowed"],
    }


def validate_ksq004_smoke_result_layer() -> dict[str, Any]:
    if not KSQ004_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ004 10-source smoke result "
            f"{KSQ004_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ004_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ004 smoke result schema_version must be 1")
    if result.get("candidate_id") != "ksq004_bridge_answer_interface_minimal_pairs":
        fail("KSQ004 smoke result candidate_id changed")
    if result.get("run_type") != "ksq004_bridge_answer_interface_minimal_pairs_behavior":
        fail("KSQ004 smoke result must be a behavior run")
    if result.get("limit_sources") != 10:
        fail("KSQ004 smoke result must remain the 10-source smoke")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ004 smoke result must not license hidden-state work")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ004 smoke result missing summary")
    if summary.get("diagnostic_class") != "smoke_answer_interface_candidate":
        fail("KSQ004 smoke diagnostic class changed")
    if summary.get("behavior_ready") is not False:
        fail("KSQ004 smoke must not be behavior ready")
    if summary.get("behavior_candidate") is not True:
        fail("KSQ004 smoke must remain a behavior candidate")
    if summary.get("hidden_state_allowed") is not False:
        fail("KSQ004 smoke summary must not license hidden-state work")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ004 smoke structural gate must pass")
    if structural.get("record_count") != 180 or structural.get("source_count") != 10:
        fail("KSQ004 smoke must keep the 180-row, 10-source scope")

    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ004 smoke missing criteria")
    expected_criteria = {
        "selected_prompt_audit_passed": True,
        "answer_shape_balance_passed": True,
        "matched_minimal_pairs_passed": True,
        "local_learned_direct_controls_passed": True,
        "minimal_pair_conflict_passed": True,
        "side_answer_leakage_passed": True,
        "null_and_holdout_passed": True,
        "candidate_and_output_margins_reported": True,
    }
    for key, expected_value in expected_criteria.items():
        if criteria.get(key) is not expected_value:
            fail(f"KSQ004 smoke criterion changed: {key}")

    if summary["selection"]["selected_template"] != "question_form":
        fail("KSQ004 smoke selected template changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "matched_minimal_pairs",
            "local_learned_direct_controls",
            "minimal_pair_conflict",
            "side_answer_leakage",
            "null_and_holdout",
        ]
    }
    expected_panel_counts = {
        "matched_minimal_pairs": {"atomic_number": 10, "local_number": 10},
        "local_learned_direct_controls": {"atomic_number": 10, "local_number": 10},
        "minimal_pair_conflict": {"atomic_number": 9, "local_number": 11},
        "side_answer_leakage": {"atomic_number": 9, "local_number": 10, "other_number": 1},
        "null_and_holdout": {"unknown": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ004 smoke panel label counts changed")

    return {
        "status": "smoke_answer_interface_candidate",
        "diagnostic_class": summary["diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
    }


def validate_ksq004_full_behavior_result_layer() -> dict[str, Any]:
    if not KSQ004_FULL_BEHAVIOR_RESULT_PATH.exists():
        fail(
            "missing KSQ004 full behavior result "
            f"{KSQ004_FULL_BEHAVIOR_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ004_FULL_BEHAVIOR_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ004 full behavior result schema_version must be 1")
    if result.get("candidate_id") != "ksq004_bridge_answer_interface_minimal_pairs":
        fail("KSQ004 full behavior result candidate_id changed")
    if result.get("run_type") != "ksq004_bridge_answer_interface_minimal_pairs_behavior":
        fail("KSQ004 full behavior result must be a behavior run")
    if result.get("limit_sources") is not None:
        fail("KSQ004 full behavior result must use all sources")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ004 full behavior result must not license hidden-state work")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ004 full behavior result missing summary")
    if summary.get("diagnostic_class") != "bridge_minimal_pair_contrast_absent":
        fail("KSQ004 full behavior diagnostic class changed")
    if summary.get("behavior_ready") is not False:
        fail("KSQ004 full behavior must not be behavior ready")
    if summary.get("behavior_candidate") is not False:
        fail("KSQ004 full behavior must not be a behavior candidate")
    if summary.get("hidden_state_allowed") is not False:
        fail("KSQ004 full behavior summary must not license hidden-state work")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ004 full behavior structural gate must pass")
    if structural.get("record_count") != 720 or structural.get("source_count") != 40:
        fail("KSQ004 full behavior must keep the 720-row, 40-source scope")

    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ004 full behavior missing criteria")
    expected_criteria = {
        "selected_prompt_audit_passed": True,
        "answer_shape_balance_passed": True,
        "matched_minimal_pairs_passed": True,
        "local_learned_direct_controls_passed": True,
        "minimal_pair_conflict_passed": False,
        "side_answer_leakage_passed": True,
        "null_and_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    for key, expected_value in expected_criteria.items():
        if criteria.get(key) is not expected_value:
            fail(f"KSQ004 full behavior criterion changed: {key}")

    if summary["selection"]["selected_template"] != "compact_form":
        fail("KSQ004 full behavior selected template changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "matched_minimal_pairs",
            "local_learned_direct_controls",
            "minimal_pair_conflict",
            "side_answer_leakage",
            "null_and_holdout",
        ]
    }
    expected_panel_counts = {
        "matched_minimal_pairs": {"atomic_number": 40, "local_number": 40},
        "local_learned_direct_controls": {"atomic_number": 40, "local_number": 40},
        "minimal_pair_conflict": {"atomic_number": 9, "local_number": 70, "other_number": 1},
        "side_answer_leakage": {"atomic_number": 8, "local_number": 72},
        "null_and_holdout": {"unknown": 40},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ004 full behavior panel label counts changed")

    question_form = summary["by_template"]["question_form"]
    compact_form = summary["by_template"]["compact_form"]
    if question_form["primary_conflict"]["expected_correct_rate"] != 0.85:
        fail("KSQ004 question_form conflict rate changed")
    if compact_form["primary_conflict"]["expected_correct_rate"] != 0.6125:
        fail("KSQ004 compact_form conflict rate changed")
    if question_form["subtypes"]["atomic_branch_conflict"]["atomic_number_rate"] != 0.7:
        fail("KSQ004 question_form atomic-branch rate changed")
    if compact_form["subtypes"]["atomic_branch_conflict"]["atomic_number_rate"] != 0.225:
        fail("KSQ004 compact_form atomic-branch rate changed")
    if question_form["primary_conflict_holdout"]["expected_correct_rate"] != 0.875:
        fail("KSQ004 question_form holdout rate changed")
    if compact_form["primary_conflict_holdout"]["expected_correct_rate"] != 0.625:
        fail("KSQ004 compact_form holdout rate changed")

    return {
        "status": "full_failed_template_fragility",
        "diagnostic_class": summary["diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "question_form_conflict_expected_correct_rate": question_form["primary_conflict"]["expected_correct_rate"],
        "compact_form_conflict_expected_correct_rate": compact_form["primary_conflict"]["expected_correct_rate"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
    }


def validate_ksq005_first_run_result_layer() -> dict[str, Any]:
    if not KSQ005_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ005 first-run result "
            f"{KSQ005_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ005_FIRST_RUN_STATUS_PATH.exists():
        fail(
            "missing KSQ005 first-run status card "
            f"{KSQ005_FIRST_RUN_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ005_FIRST_RUN_PREREG_PATH.exists():
        fail(
            "missing KSQ005 first-run preregistration "
            f"{KSQ005_FIRST_RUN_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ005_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ005 first-run result schema_version must be 1")
    if result.get("candidate_id") != "ksq005_uncertainty_grounded_answerability":
        fail("KSQ005 first-run result candidate_id changed")
    if result.get("run_type") != "ksq005_uncertainty_grounded_answerability_structural_gate":
        fail("KSQ005 first-run result must be the structural gate result")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ005 first-run result must not license hidden-state work")

    structural = result.get("structural")
    if not isinstance(structural, dict):
        fail("KSQ005 first-run result missing structural payload")
    criteria = structural.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ005 first-run result missing structural criteria")
    failed_criteria = [key for key, value in criteria.items() if value is not True]
    if failed_criteria:
        fail(f"KSQ005 first-run structural criteria failed: {failed_criteria}")
    if structural.get("passed") is not True:
        fail("KSQ005 first-run structural gate must be passed")
    if structural.get("record_count") != 480:
        fail("KSQ005 first-run result must keep the full 480-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ005 first-run result must keep the full 40-source substrate")
    if structural.get("panel_counts") != {
        "contradicted_context_rows": 120,
        "known_factual_direct": 120,
        "unknown_nonce_rows": 120,
        "unsupported_context_rows": 120,
    }:
        fail("KSQ005 first-run panel counts changed")
    if structural.get("template_counts") != {
        "compact_field": 160,
        "plain_question": 160,
        "reference_note": 160,
    }:
        fail("KSQ005 first-run template counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ005 first-run source split changed")
    return {
        "status": "structural_passed",
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": result["hidden_state_allowed"],
    }


def validate_ksq005_smoke_result_layer() -> dict[str, Any]:
    if not KSQ005_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ005 10-source smoke result "
            f"{KSQ005_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ005_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ005 smoke result schema_version must be 1")
    if result.get("candidate_id") != "ksq005_uncertainty_grounded_answerability":
        fail("KSQ005 smoke result candidate_id changed")
    if result.get("run_type") != "ksq005_uncertainty_grounded_answerability_behavior":
        fail("KSQ005 smoke result must be a behavior run")
    if result.get("limit_sources") != 10:
        fail("KSQ005 smoke result must remain the 10-source smoke")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ005 smoke result must not license hidden-state work")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ005 smoke result missing summary")
    if summary.get("diagnostic_class") != "unknown_nonce_abstention_failed":
        fail("KSQ005 smoke diagnostic class changed")
    if summary.get("behavior_ready") is not False:
        fail("KSQ005 smoke must not be behavior ready")
    if summary.get("behavior_candidate") is not False:
        fail("KSQ005 smoke must not be a behavior candidate")
    if summary.get("hidden_state_allowed") is not False:
        fail("KSQ005 smoke summary must not license hidden-state work")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ005 smoke structural gate must pass")
    if structural.get("record_count") != 120 or structural.get("source_count") != 10:
        fail("KSQ005 smoke must keep the 120-row, 10-source scope")

    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ005 smoke missing criteria")
    expected_criteria = {
        "selected_prompt_audit_passed": True,
        "known_factual_direct_passed": True,
        "unknown_nonce_rows_passed": False,
        "unsupported_context_rows_passed": False,
        "contradicted_context_rows_passed": True,
        "candidate_and_output_margins_reported": True,
    }
    for key, expected_value in expected_criteria.items():
        if criteria.get(key) is not expected_value:
            fail(f"KSQ005 smoke criterion changed: {key}")

    if summary["selection"]["selected_template"] != "reference_note":
        fail("KSQ005 smoke selected template changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "known_factual_direct",
            "unknown_nonce_rows",
            "unsupported_context_rows",
            "contradicted_context_rows",
        ]
    }
    expected_panel_counts = {
        "known_factual_direct": {
            "known_correct": 8,
            "known_wrong_candidate": 1,
            "unparsed": 1,
        },
        "unknown_nonce_rows": {"abstain": 4, "unparsed": 6},
        "unsupported_context_rows": {
            "abstain": 4,
            "unparsed": 3,
            "unsupported_answer": 3,
        },
        "contradicted_context_rows": {
            "abstain": 3,
            "corrected": 5,
            "false_accept": 1,
            "wrong_answer": 1,
        },
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ005 smoke panel label counts changed")

    return {
        "status": "smoke_failed_unknown_and_unsupported_abstention",
        "diagnostic_class": summary["diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
    }


def validate_ksq006_first_run_result_layer() -> dict[str, Any]:
    if not KSQ006_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ006 first-run result "
            f"{KSQ006_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ006_FIRST_RUN_STATUS_PATH.exists():
        fail(
            "missing KSQ006 first-run status card "
            f"{KSQ006_FIRST_RUN_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ006_FIRST_RUN_PREREG_PATH.exists():
        fail(
            "missing KSQ006 first-run preregistration "
            f"{KSQ006_FIRST_RUN_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ006_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ006 first-run result schema_version must be 1")
    if result.get("candidate_id") != "ksq006_uncertainty_context_support_counterfactuals":
        fail("KSQ006 first-run result candidate_id changed")
    if result.get("run_type") != "ksq006_uncertainty_context_support_counterfactuals_structural_gate":
        fail("KSQ006 first-run result must be the structural gate result")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ006 first-run result must not license hidden-state work")

    structural = result.get("structural")
    if not isinstance(structural, dict):
        fail("KSQ006 first-run result missing structural payload")
    criteria = structural.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ006 first-run result missing structural criteria")
    failed_criteria = [key for key, value in criteria.items() if value is not True]
    if failed_criteria:
        fail(f"KSQ006 first-run structural criteria failed: {failed_criteria}")
    if structural.get("passed") is not True:
        fail("KSQ006 first-run structural gate must be passed")
    if structural.get("record_count") != 720:
        fail("KSQ006 first-run result must keep the full 720-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ006 first-run result must keep the full 40-source substrate")
    if structural.get("panel_counts") != {
        "claim_only_and_context_only_controls": 240,
        "contradicting_context_rows": 120,
        "insufficient_context_rows": 120,
        "irrelevant_context_rows": 120,
        "supported_context_rows": 120,
    }:
        fail("KSQ006 first-run panel counts changed")
    if structural.get("template_counts") != {
        "compact_record": 240,
        "field_form": 240,
        "note_question": 240,
    }:
        fail("KSQ006 first-run template counts changed")
    if structural.get("control_subtype_counts") != {
        "claim_only_control": 120,
        "context_only_control": 120,
    }:
        fail("KSQ006 first-run control subtype counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ006 first-run source split changed")
    return {
        "status": "structural_passed",
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": result["hidden_state_allowed"],
    }


def validate_ksq006_smoke_result_layer() -> dict[str, Any]:
    if not KSQ006_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ006 10-source smoke result "
            f"{KSQ006_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ006_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ006 smoke result schema_version must be 1")
    if result.get("candidate_id") != "ksq006_uncertainty_context_support_counterfactuals":
        fail("KSQ006 smoke result candidate_id changed")
    if result.get("run_type") != "ksq006_uncertainty_context_support_counterfactuals_behavior":
        fail("KSQ006 smoke result must be a behavior run")
    if result.get("limit_sources") != 10:
        fail("KSQ006 smoke result must remain the 10-source smoke")
    if result.get("hidden_state_allowed") is not False:
        fail("KSQ006 smoke result must not license hidden-state work")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ006 smoke result missing summary")
    if summary.get("diagnostic_class") != "insufficient_context_abstention_failed":
        fail("KSQ006 smoke diagnostic class changed")
    if summary.get("behavior_ready") is not False:
        fail("KSQ006 smoke must not be behavior ready")
    if summary.get("behavior_candidate") is not False:
        fail("KSQ006 smoke must not be a behavior candidate")
    if summary.get("hidden_state_allowed") is not False:
        fail("KSQ006 smoke summary must not license hidden-state work")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ006 smoke structural gate must pass")
    if structural.get("record_count") != 180 or structural.get("source_count") != 10:
        fail("KSQ006 smoke must keep the 180-row, 10-source scope")

    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ006 smoke missing criteria")
    expected_criteria = {
        "selected_prompt_audit_passed": True,
        "supported_context_rows_passed": True,
        "irrelevant_context_rows_passed": True,
        "contradicting_context_rows_passed": True,
        "insufficient_context_rows_passed": False,
        "claim_only_and_context_only_controls_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    for key, expected_value in expected_criteria.items():
        if criteria.get(key) is not expected_value:
            fail(f"KSQ006 smoke criterion changed: {key}")

    if summary["selection"]["selected_template"] != "field_form":
        fail("KSQ006 smoke selected template changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "supported_context_rows",
            "irrelevant_context_rows",
            "contradicting_context_rows",
            "insufficient_context_rows",
            "claim_only_and_context_only_controls",
        ]
    }
    expected_panel_counts = {
        "supported_context_rows": {"abstain": 1, "supported_answer": 9},
        "irrelevant_context_rows": {"abstain": 10},
        "contradicting_context_rows": {
            "contradiction_detected": 8,
            "false_accept": 1,
            "true_answer_despite_contradiction": 1,
        },
        "insufficient_context_rows": {"abstain": 5, "unsupported_answer": 5},
        "claim_only_and_context_only_controls": {
            "claim_only_reproduced_supported": 8,
            "context_only_reproduced_supported": 10,
            "control_abstain": 2,
        },
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ006 smoke panel label counts changed")
    subtype_counts = {
        subtype: selected["controls"][subtype]["label_counts"]
        for subtype in ["claim_only_control", "context_only_control"]
    }
    expected_subtype_counts = {
        "claim_only_control": {"claim_only_reproduced_supported": 8, "control_abstain": 2},
        "context_only_control": {"context_only_reproduced_supported": 10},
    }
    if subtype_counts != expected_subtype_counts:
        fail("KSQ006 smoke control subtype label counts changed")

    return {
        "status": "smoke_failed_insufficient_and_claim_context_controls",
        "diagnostic_class": summary["diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
    }


def validate_ksq007_first_run_result_layer() -> dict[str, Any]:
    if not KSQ007_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ007 first-run result "
            f"{KSQ007_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ007_STATUS_PATH.exists():
        fail(
            "missing KSQ007 status card "
            f"{KSQ007_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ007_PREREG_PATH.exists():
        fail(
            "missing KSQ007 preregistration "
            f"{KSQ007_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ007_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ007 first-run result schema_version must be 1")
    if result.get("card_id") != "KSQ007":
        fail("KSQ007 first-run card_id changed")
    if result.get("candidate_id") != "ksq007_nonce_evidence_answerability_calibrator":
        fail("KSQ007 first-run candidate_id changed")
    if result.get("run_type") != "ksq007_nonce_evidence_answerability_structural_gate":
        fail("KSQ007 first-run result must be the structural gate result")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ007 first-run result missing summary")
    if summary.get("diagnostic_class") != "structural_passed":
        fail("KSQ007 first-run diagnostic class changed")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ007 first-run structural gate must pass")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ007 first-run missing criteria")
    expected_criteria = {
        "structural_passed": True,
        "full_source_count_is_40": True,
        "source_disjoint_holdout": True,
        "hidden_state_license": False,
    }
    if criteria != expected_criteria:
        fail("KSQ007 first-run criteria changed")
    decision = summary.get("calibrator_decision")
    if not isinstance(decision, dict):
        fail("KSQ007 first-run missing calibrator decision")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ007 first-run must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ007 first-run must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ007 first-run must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ007 first-run must not license mechanism claims")

    if structural.get("record_count") != 840:
        fail("KSQ007 first-run result must keep the full 840-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ007 first-run result must keep the full 40-source substrate")
    if structural.get("panel_counts") != {
        "absent_evidence_rows": 120,
        "claim_only_control": 120,
        "conflicting_evidence_rows": 120,
        "exact_evidence_rows": 120,
        "mention_only_control": 120,
        "query_only_control": 120,
        "unrelated_entity_rows": 120,
    }:
        fail("KSQ007 first-run panel counts changed")
    if structural.get("template_counts") != {
        "compact_evidence": 280,
        "evidence_rows": 280,
        "ledger_form": 280,
    }:
        fail("KSQ007 first-run template counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ007 first-run source split changed")
    return {
        "status": "structural_passed",
        "diagnostic_class": summary["diagnostic_class"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq007_smoke_result_layer() -> dict[str, Any]:
    if not KSQ007_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ007 10-source smoke result "
            f"{KSQ007_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ007_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ007 smoke result schema_version must be 1")
    if result.get("card_id") != "KSQ007":
        fail("KSQ007 smoke card_id changed")
    if result.get("candidate_id") != "ksq007_nonce_evidence_answerability_calibrator":
        fail("KSQ007 smoke candidate_id changed")
    if result.get("run_type") != "ksq007_nonce_evidence_answerability_behavior":
        fail("KSQ007 smoke result must be a behavior run")
    if result.get("full_run") is not False:
        fail("KSQ007 smoke result must remain the 10-source smoke")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ007 smoke result missing summary")
    if summary.get("diagnostic_class") != "nonce_evidence_claim_or_mention_control_failed":
        fail("KSQ007 smoke diagnostic class changed")
    decision = summary.get("calibrator_decision")
    if not isinstance(decision, dict):
        fail("KSQ007 smoke missing calibrator decision")
    if decision.get("exported_diagnostic_class") != "NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY":
        fail("KSQ007 smoke exported diagnostic class changed")
    if decision.get("behavior_ready") is not False:
        fail("KSQ007 smoke must not be behavior ready")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ007 smoke must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ007 smoke must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ007 smoke must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ007 smoke must not license mechanism claims")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ007 smoke structural gate must pass")
    if structural.get("record_count") != 210 or structural.get("source_count") != 10:
        fail("KSQ007 smoke must keep the 210-row, 10-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ007 smoke missing criteria")
    expected_criteria = {
        "smoke_mode": True,
        "structural_passed": True,
        "full_source_count_is_40": False,
        "source_disjoint_holdout": True,
        "selected_prompt_audit_passed": True,
        "exact_evidence_rows_passed": True,
        "absent_evidence_rows_passed": True,
        "unrelated_entity_rows_passed": True,
        "conflicting_evidence_rows_passed": True,
        "claim_only_and_mention_only_controls_passed": False,
        "query_only_control_passed": True,
        "source_disjoint_answerability_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    if criteria != expected_criteria:
        fail("KSQ007 smoke criteria changed")
    if summary["selection"]["selected_template"] != "evidence_rows":
        fail("KSQ007 smoke selected template changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "exact_evidence_rows",
            "absent_evidence_rows",
            "unrelated_entity_rows",
            "conflicting_evidence_rows",
            "claim_only_control",
            "mention_only_control",
            "query_only_control",
        ]
    }
    expected_panel_counts = {
        "exact_evidence_rows": {"abstain": 1, "evidence_answer": 9},
        "absent_evidence_rows": {"abstain": 10},
        "unrelated_entity_rows": {"control_abstain": 10},
        "conflicting_evidence_rows": {"abstain": 9, "conflict_value_selected": 1},
        "claim_only_control": {"claim_only_reproduced": 6, "control_abstain": 4},
        "mention_only_control": {"control_abstain": 10},
        "query_only_control": {"control_abstain": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ007 smoke panel label counts changed")
    return {
        "status": "smoke_failed_claim_only_control",
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq007b_first_run_result_layer() -> dict[str, Any]:
    if not KSQ007B_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ007B first-run result "
            f"{KSQ007B_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ007B_STATUS_PATH.exists():
        fail(
            "missing KSQ007B status card "
            f"{KSQ007B_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ007B_PREREG_PATH.exists():
        fail(
            "missing KSQ007B preregistration "
            f"{KSQ007B_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ007B_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ007B first-run result schema_version must be 1")
    if result.get("card_id") != "KSQ007B":
        fail("KSQ007B first-run card_id changed")
    if result.get("parent_card_id") != "KSQ007":
        fail("KSQ007B first-run parent_card_id changed")
    if result.get("candidate_id") != "ksq007_claim_channel_boundary_audit":
        fail("KSQ007B first-run candidate_id changed")
    if result.get("run_type") != "ksq007_claim_channel_boundary_structural_gate":
        fail("KSQ007B first-run result must be the structural gate result")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ007B first-run result missing summary")
    if summary.get("diagnostic_class") != "structural_passed":
        fail("KSQ007B first-run diagnostic class changed")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ007B first-run structural gate must pass")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ007B first-run missing criteria")
    expected_criteria = {
        "structural_passed": True,
        "full_source_count_is_40": True,
        "source_disjoint_holdout": True,
        "hidden_state_license": False,
    }
    if criteria != expected_criteria:
        fail("KSQ007B first-run criteria changed")
    decision = summary.get("claim_boundary_decision")
    if not isinstance(decision, dict):
        fail("KSQ007B first-run missing claim-boundary decision")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ007B first-run must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ007B first-run must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ007B first-run must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ007B first-run must not license mechanism claims")

    if structural.get("record_count") != 1200:
        fail("KSQ007B first-run result must keep the full 1,200-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ007B first-run result must keep the full 40-source substrate")
    if structural.get("panel_counts") != {
        "bare_same_syntax": 120,
        "claim_prose": 120,
        "claim_same_syntax": 120,
        "exact_evidence_positive": 120,
        "mention_only": 120,
        "not_evidence_prefix": 120,
        "other_block_evidence_row": 120,
        "query_only": 120,
        "quoted_evidence_syntax": 120,
        "wrong_predicate_claim": 120,
    }:
        fail("KSQ007B first-run panel counts changed")
    if structural.get("template_counts") != {
        "counted_uncounted_sections": 400,
        "inline_rows": 400,
        "separated_blocks": 400,
    }:
        fail("KSQ007B first-run template counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ007B first-run source split changed")
    return {
        "status": "structural_passed",
        "diagnostic_class": summary["diagnostic_class"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq007b_smoke_result_layer() -> dict[str, Any]:
    if not KSQ007B_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ007B 10-source smoke result "
            f"{KSQ007B_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ007B_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ007B smoke result schema_version must be 1")
    if result.get("card_id") != "KSQ007B":
        fail("KSQ007B smoke card_id changed")
    if result.get("parent_card_id") != "KSQ007":
        fail("KSQ007B smoke parent_card_id changed")
    if result.get("candidate_id") != "ksq007_claim_channel_boundary_audit":
        fail("KSQ007B smoke candidate_id changed")
    if result.get("run_type") != "ksq007_claim_channel_boundary_behavior":
        fail("KSQ007B smoke result must be a behavior run")
    if result.get("full_run") is not False:
        fail("KSQ007B smoke result must remain the 10-source smoke")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ007B smoke result missing summary")
    if summary.get("diagnostic_class") != "answer_for_syntax_claim_leak":
        fail("KSQ007B smoke diagnostic class changed")
    decision = summary.get("claim_boundary_decision")
    if not isinstance(decision, dict):
        fail("KSQ007B smoke missing claim-boundary decision")
    if decision.get("exported_diagnostic_class") != "ANSWER_FOR_SYNTAX_CLAIM_LEAK":
        fail("KSQ007B smoke exported diagnostic class changed")
    if decision.get("behavior_ready") is not False:
        fail("KSQ007B smoke must not be behavior ready")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ007B smoke must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ007B smoke must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ007B smoke must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ007B smoke must not license mechanism claims")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ007B smoke structural gate must pass")
    if structural.get("record_count") != 300 or structural.get("source_count") != 10:
        fail("KSQ007B smoke must keep the 300-row, 10-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ007B smoke missing criteria")
    expected_criteria = {
        "smoke_mode": True,
        "structural_passed": True,
        "full_source_count_is_40": False,
        "source_disjoint_holdout": True,
        "selected_prompt_audit_passed": True,
        "exact_evidence_positive_passed": True,
        "claim_controls_passed": False,
        "syntax_controls_passed": False,
        "section_boundary_controls_passed": False,
        "mention_and_query_baselines_passed": True,
        "all_control_panels_passed": False,
        "source_disjoint_claim_boundary_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    if criteria != expected_criteria:
        fail("KSQ007B smoke criteria changed")
    if summary["selection"]["selected_template"] != "counted_uncounted_sections":
        fail("KSQ007B smoke selected template changed")
    if summary.get("failed_control_panels") != [
        "claim_prose",
        "bare_same_syntax",
        "other_block_evidence_row",
    ]:
        fail("KSQ007B smoke failed control panels changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "exact_evidence_positive",
            "claim_same_syntax",
            "claim_prose",
            "bare_same_syntax",
            "not_evidence_prefix",
            "other_block_evidence_row",
            "quoted_evidence_syntax",
            "wrong_predicate_claim",
            "mention_only",
            "query_only",
        ]
    }
    expected_panel_counts = {
        "exact_evidence_positive": {"abstain": 1, "evidence_answer": 9},
        "claim_same_syntax": {"claim_same_syntax_reproduced": 1, "control_abstain": 9},
        "claim_prose": {"claim_prose_reproduced": 4, "control_abstain": 6},
        "bare_same_syntax": {"bare_same_syntax_reproduced": 7, "control_abstain": 3},
        "not_evidence_prefix": {"control_abstain": 10},
        "other_block_evidence_row": {"control_abstain": 8, "other_block_evidence_reproduced": 2},
        "quoted_evidence_syntax": {"control_abstain": 10},
        "wrong_predicate_claim": {"control_abstain": 10},
        "mention_only": {"control_abstain": 10},
        "query_only": {"control_abstain": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ007B smoke panel label counts changed")
    return {
        "status": "smoke_failed_answer_for_syntax_claim_leak",
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq008_first_run_result_layer() -> dict[str, Any]:
    if not KSQ008_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ008 first-run result "
            f"{KSQ008_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ008_STATUS_PATH.exists():
        fail(
            "missing KSQ008 status card "
            f"{KSQ008_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ008_PREREG_PATH.exists():
        fail(
            "missing KSQ008 preregistration "
            f"{KSQ008_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ008_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ008 first-run result schema_version must be 1")
    if result.get("card_id") != "KSQ008":
        fail("KSQ008 first-run card_id changed")
    if result.get("parent_card_id") != "KSQ007B":
        fail("KSQ008 first-run parent_card_id changed")
    if result.get("candidate_id") != "ksq008_neutral_evidence_channel_repair":
        fail("KSQ008 first-run candidate_id changed")
    if result.get("run_type") != "ksq008_neutral_evidence_channel_repair_structural_gate":
        fail("KSQ008 first-run result must be the structural gate result")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ008 first-run result missing summary")
    if summary.get("diagnostic_class") != "structural_passed":
        fail("KSQ008 first-run diagnostic class changed")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ008 first-run structural gate must pass")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ008 first-run missing criteria")
    expected_criteria = {
        "structural_passed": True,
        "full_source_count_is_40": True,
        "source_disjoint_holdout": True,
        "hidden_state_license": False,
    }
    if criteria != expected_criteria:
        fail("KSQ008 first-run criteria changed")
    decision = summary.get("neutral_repair_decision")
    if not isinstance(decision, dict):
        fail("KSQ008 first-run missing neutral-repair decision")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ008 first-run must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ008 first-run must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ008 first-run must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ008 first-run must not license mechanism claims")

    if structural.get("record_count") != 1440:
        fail("KSQ008 first-run result must keep the full 1,440-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ008 first-run result must keep the full 40-source substrate")
    expected_panel_counts = {
        "absent_neutral_evidence": 120,
        "conflicting_neutral_evidence": 120,
        "counted_wrong_schema_answer_for": 120,
        "exact_neutral_evidence": 120,
        "forbidden_bare_answer_for": 120,
        "forbidden_claim_answer_for": 120,
        "forbidden_prose_claim": 120,
        "neutral_evidence_vs_forbidden_bare_alt": 120,
        "query_only_control": 120,
        "quoted_neutral_evidence": 120,
        "uncounted_neutral_evidence": 120,
        "unrelated_entity_neutral": 120,
    }
    if structural.get("panel_counts") != expected_panel_counts:
        fail("KSQ008 first-run panel counts changed")
    if structural.get("template_counts") != {
        "field_registry": 480,
        "neutral_table": 480,
        "tagged_records": 480,
    }:
        fail("KSQ008 first-run template counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ008 first-run source split changed")
    return {
        "status": "structural_passed",
        "diagnostic_class": summary["diagnostic_class"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq008_smoke_result_layer() -> dict[str, Any]:
    if not KSQ008_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ008 10-source smoke result "
            f"{KSQ008_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ008_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ008 smoke result schema_version must be 1")
    if result.get("card_id") != "KSQ008":
        fail("KSQ008 smoke card_id changed")
    if result.get("parent_card_id") != "KSQ007B":
        fail("KSQ008 smoke parent_card_id changed")
    if result.get("candidate_id") != "ksq008_neutral_evidence_channel_repair":
        fail("KSQ008 smoke candidate_id changed")
    if result.get("run_type") != "ksq008_neutral_evidence_channel_repair_behavior":
        fail("KSQ008 smoke result must be a behavior run")
    if result.get("full_run") is not False:
        fail("KSQ008 smoke result must remain the 10-source smoke")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ008 smoke result missing summary")
    if summary.get("diagnostic_class") != "neutral_evidence_positive_failed":
        fail("KSQ008 smoke diagnostic class changed")
    decision = summary.get("neutral_repair_decision")
    if not isinstance(decision, dict):
        fail("KSQ008 smoke missing neutral-repair decision")
    if decision.get("exported_diagnostic_class") != "NEUTRAL_EVIDENCE_POSITIVE_FAILED":
        fail("KSQ008 smoke exported diagnostic class changed")
    if decision.get("behavior_ready") is not False:
        fail("KSQ008 smoke must not be behavior ready")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ008 smoke must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ008 smoke must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ008 smoke must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ008 smoke must not license mechanism claims")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ008 smoke structural gate must pass")
    if structural.get("record_count") != 360 or structural.get("source_count") != 10:
        fail("KSQ008 smoke must keep the 360-row, 10-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ008 smoke missing criteria")
    expected_criteria = {
        "smoke_mode": True,
        "structural_passed": True,
        "full_source_count_is_40": False,
        "source_disjoint_holdout": True,
        "selected_prompt_audit_passed": True,
        "exact_neutral_evidence_passed": False,
        "neutral_evidence_vs_forbidden_bare_alt_passed": False,
        "answer_for_forbidden_controls_passed": False,
        "neutral_channel_locality_controls_passed": True,
        "abstain_controls_passed": False,
        "all_panels_passed": False,
        "source_disjoint_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    if criteria != expected_criteria:
        fail("KSQ008 smoke criteria changed")
    if summary["selection"]["selected_template"] != "field_registry":
        fail("KSQ008 smoke selected template changed")
    if summary.get("failed_panels") != [
        "exact_neutral_evidence",
        "conflicting_neutral_evidence",
        "counted_wrong_schema_answer_for",
        "neutral_evidence_vs_forbidden_bare_alt",
    ]:
        fail("KSQ008 smoke failed panels changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "exact_neutral_evidence",
            "absent_neutral_evidence",
            "unrelated_entity_neutral",
            "conflicting_neutral_evidence",
            "forbidden_bare_answer_for",
            "forbidden_claim_answer_for",
            "forbidden_prose_claim",
            "uncounted_neutral_evidence",
            "quoted_neutral_evidence",
            "counted_wrong_schema_answer_for",
            "neutral_evidence_vs_forbidden_bare_alt",
            "query_only_control",
        ]
    }
    expected_panel_counts = {
        "exact_neutral_evidence": {"abstain": 5, "evidence_answer": 5},
        "absent_neutral_evidence": {"control_abstain": 10},
        "unrelated_entity_neutral": {"control_abstain": 10},
        "conflicting_neutral_evidence": {"conflict_value_selected": 2, "control_abstain": 8},
        "forbidden_bare_answer_for": {"control_abstain": 10},
        "forbidden_claim_answer_for": {"control_abstain": 10},
        "forbidden_prose_claim": {"control_abstain": 10},
        "uncounted_neutral_evidence": {"control_abstain": 10},
        "quoted_neutral_evidence": {"control_abstain": 10},
        "counted_wrong_schema_answer_for": {
            "control_abstain": 5,
            "counted_wrong_schema_reproduced": 5,
        },
        "neutral_evidence_vs_forbidden_bare_alt": {
            "abstain": 8,
            "evidence_answer": 2,
        },
        "query_only_control": {"control_abstain": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ008 smoke panel label counts changed")
    return {
        "status": "smoke_failed_neutral_evidence_positive",
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq009_first_run_result_layer() -> dict[str, Any]:
    if not KSQ009_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ009 first-run result "
            f"{KSQ009_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ009_STATUS_PATH.exists():
        fail(
            "missing KSQ009 status card "
            f"{KSQ009_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ009_PREREG_PATH.exists():
        fail(
            "missing KSQ009 preregistration "
            f"{KSQ009_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ009_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ009 first-run result schema_version must be 1")
    if result.get("card_id") != "KSQ009":
        fail("KSQ009 first-run card_id changed")
    if result.get("parent_card_id") != "KSQ008":
        fail("KSQ009 first-run parent_card_id changed")
    if result.get("candidate_id") != "ksq009_schema_specific_value_lookup":
        fail("KSQ009 first-run candidate_id changed")
    if result.get("run_type") != "ksq009_schema_specific_value_lookup_structural_gate":
        fail("KSQ009 first-run result must be the structural gate result")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ009 first-run result missing summary")
    if summary.get("diagnostic_class") != "structural_passed":
        fail("KSQ009 first-run diagnostic class changed")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ009 first-run structural gate must pass")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ009 first-run missing criteria")
    expected_criteria = {
        "structural_passed": True,
        "full_source_count_is_40": True,
        "source_disjoint_holdout": True,
        "hidden_state_license": False,
    }
    if criteria != expected_criteria:
        fail("KSQ009 first-run criteria changed")
    decision = summary.get("schema_specific_decision")
    if not isinstance(decision, dict):
        fail("KSQ009 first-run missing schema-specific decision")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ009 first-run must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ009 first-run must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ009 first-run must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ009 first-run must not license mechanism claims")

    if structural.get("record_count") != 1200:
        fail("KSQ009 first-run result must keep the full 1,200-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ009 first-run result must keep the full 40-source substrate")
    expected_panel_counts = {
        "absent_allowed_value": 120,
        "allowed_value_vs_uncounted_answer_for_alt": 120,
        "conflicting_allowed_values": 120,
        "counted_wrong_schema_answer_for": 120,
        "exact_allowed_value": 120,
        "query_only_control": 120,
        "quoted_allowed_value": 120,
        "uncounted_allowed_value": 120,
        "uncounted_wrong_schema_answer_for": 120,
        "unrelated_entity_allowed_value": 120,
    }
    if structural.get("panel_counts") != expected_panel_counts:
        fail("KSQ009 first-run panel counts changed")
    if structural.get("template_counts") != {
        "csv_rows": 400,
        "kv_lines": 400,
        "pipe_rows": 400,
    }:
        fail("KSQ009 first-run template counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ009 first-run source split changed")
    return {
        "status": "structural_passed",
        "diagnostic_class": summary["diagnostic_class"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq009_smoke_result_layer() -> dict[str, Any]:
    if not KSQ009_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ009 10-source smoke result "
            f"{KSQ009_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ009_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ009 smoke result schema_version must be 1")
    if result.get("card_id") != "KSQ009":
        fail("KSQ009 smoke card_id changed")
    if result.get("parent_card_id") != "KSQ008":
        fail("KSQ009 smoke parent_card_id changed")
    if result.get("candidate_id") != "ksq009_schema_specific_value_lookup":
        fail("KSQ009 smoke candidate_id changed")
    if result.get("run_type") != "ksq009_schema_specific_value_lookup_behavior":
        fail("KSQ009 smoke result must be a behavior run")
    if result.get("full_run") is not False:
        fail("KSQ009 smoke result must remain the 10-source smoke")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ009 smoke result missing summary")
    if summary.get("diagnostic_class") != "schema_specific_positive_failed":
        fail("KSQ009 smoke diagnostic class changed")
    decision = summary.get("schema_specific_decision")
    if not isinstance(decision, dict):
        fail("KSQ009 smoke missing schema-specific decision")
    if decision.get("exported_diagnostic_class") != "SCHEMA_SPECIFIC_POSITIVE_FAILED":
        fail("KSQ009 smoke exported diagnostic class changed")
    if decision.get("behavior_ready") is not False:
        fail("KSQ009 smoke must not be behavior ready")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ009 smoke must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ009 smoke must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ009 smoke must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ009 smoke must not license mechanism claims")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ009 smoke structural gate must pass")
    if structural.get("record_count") != 300 or structural.get("source_count") != 10:
        fail("KSQ009 smoke must keep the 300-row, 10-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ009 smoke missing criteria")
    expected_criteria = {
        "smoke_mode": True,
        "structural_passed": True,
        "full_source_count_is_40": False,
        "source_disjoint_holdout": True,
        "selected_prompt_audit_passed": True,
        "exact_allowed_value_passed": False,
        "allowed_value_vs_uncounted_answer_for_alt_passed": False,
        "answer_for_schema_controls_passed": False,
        "allowed_row_locality_controls_passed": False,
        "abstain_controls_passed": False,
        "all_panels_passed": False,
        "source_disjoint_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    if criteria != expected_criteria:
        fail("KSQ009 smoke criteria changed")
    if summary["selection"]["selected_template"] != "kv_lines":
        fail("KSQ009 smoke selected template changed")
    if summary.get("failed_panels") != [
        "exact_allowed_value",
        "conflicting_allowed_values",
        "counted_wrong_schema_answer_for",
        "uncounted_wrong_schema_answer_for",
        "uncounted_allowed_value",
        "quoted_allowed_value",
        "allowed_value_vs_uncounted_answer_for_alt",
    ]:
        fail("KSQ009 smoke failed panels changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "exact_allowed_value",
            "absent_allowed_value",
            "unrelated_entity_allowed_value",
            "conflicting_allowed_values",
            "counted_wrong_schema_answer_for",
            "uncounted_wrong_schema_answer_for",
            "uncounted_allowed_value",
            "quoted_allowed_value",
            "allowed_value_vs_uncounted_answer_for_alt",
            "query_only_control",
        ]
    }
    expected_panel_counts = {
        "exact_allowed_value": {"abstain": 8, "evidence_answer": 2},
        "absent_allowed_value": {"control_abstain": 10},
        "unrelated_entity_allowed_value": {"control_abstain": 10},
        "conflicting_allowed_values": {"conflict_value_selected": 5, "control_abstain": 5},
        "counted_wrong_schema_answer_for": {
            "control_abstain": 1,
            "counted_answer_for_reproduced": 9,
        },
        "uncounted_wrong_schema_answer_for": {
            "control_abstain": 3,
            "uncounted_answer_for_reproduced": 7,
        },
        "uncounted_allowed_value": {
            "control_abstain": 8,
            "uncounted_allowed_reproduced": 2,
        },
        "quoted_allowed_value": {
            "control_abstain": 6,
            "quoted_allowed_reproduced": 4,
        },
        "allowed_value_vs_uncounted_answer_for_alt": {
            "answer_for_alt_overrode_allowed_value": 9,
            "unparsed": 1,
        },
        "query_only_control": {"control_abstain": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ009 smoke panel label counts changed")
    return {
        "status": "smoke_failed_schema_specific_positive",
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq010_first_run_result_layer() -> dict[str, Any]:
    if not KSQ010_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ010 first-run result "
            f"{KSQ010_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ010_STATUS_PATH.exists():
        fail(
            "missing KSQ010 status card "
            f"{KSQ010_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ010_PREREG_PATH.exists():
        fail(
            "missing KSQ010 preregistration "
            f"{KSQ010_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ010_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ010 first-run result schema_version must be 1")
    if result.get("card_id") != "KSQ010":
        fail("KSQ010 first-run card_id changed")
    if result.get("parent_card_id") != "KSQ009":
        fail("KSQ010 first-run parent_card_id changed")
    if result.get("candidate_id") != "ksq010_two_stage_codebook_value_lookup":
        fail("KSQ010 first-run candidate_id changed")
    if result.get("run_type") != "ksq010_two_stage_codebook_value_lookup_structural_gate":
        fail("KSQ010 first-run result must be the structural gate result")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ010 first-run result missing summary")
    if summary.get("diagnostic_class") != "structural_passed":
        fail("KSQ010 first-run diagnostic class changed")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ010 first-run structural gate must pass")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ010 first-run missing criteria")
    expected_criteria = {
        "structural_passed": True,
        "full_source_count_is_40": True,
        "source_disjoint_holdout": True,
        "hidden_state_license": False,
    }
    if criteria != expected_criteria:
        fail("KSQ010 first-run criteria changed")
    decision = summary.get("codebook_decision")
    if not isinstance(decision, dict):
        fail("KSQ010 first-run missing codebook decision")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ010 first-run must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ010 first-run must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ010 first-run must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ010 first-run must not license mechanism claims")

    if structural.get("record_count") != 1440:
        fail("KSQ010 first-run result must keep the full 1,440-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ010 first-run result must keep the full 40-source substrate")
    expected_panel_counts = {
        "bridge_vs_answer_for_alt": 120,
        "conflicting_code_values": 120,
        "conflicting_entity_codes": 120,
        "counted_answer_for_only": 120,
        "exact_codebook_bridge": 120,
        "missing_code_value": 120,
        "missing_entity_code": 120,
        "query_only_control": 120,
        "quoted_bridge_rows": 120,
        "uncounted_code_value": 120,
        "uncounted_entity_code": 120,
        "unrelated_entity_code": 120,
    }
    if structural.get("panel_counts") != expected_panel_counts:
        fail("KSQ010 first-run panel counts changed")
    if structural.get("template_counts") != {
        "block_lines": 480,
        "pipe_rows": 480,
        "tag_rows": 480,
    }:
        fail("KSQ010 first-run template counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ010 first-run source split changed")
    return {
        "status": "structural_passed",
        "diagnostic_class": summary["diagnostic_class"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq010_smoke_result_layer() -> dict[str, Any]:
    if not KSQ010_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ010 10-source smoke result "
            f"{KSQ010_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ010_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ010 smoke result schema_version must be 1")
    if result.get("card_id") != "KSQ010":
        fail("KSQ010 smoke card_id changed")
    if result.get("parent_card_id") != "KSQ009":
        fail("KSQ010 smoke parent_card_id changed")
    if result.get("candidate_id") != "ksq010_two_stage_codebook_value_lookup":
        fail("KSQ010 smoke candidate_id changed")
    if result.get("run_type") != "ksq010_two_stage_codebook_value_lookup_behavior":
        fail("KSQ010 smoke result must be a behavior run")
    if result.get("full_run") is not False:
        fail("KSQ010 smoke result must remain the 10-source smoke")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ010 smoke result missing summary")
    if summary.get("diagnostic_class") != "codebook_positive_failed":
        fail("KSQ010 smoke diagnostic class changed")
    decision = summary.get("codebook_decision")
    if not isinstance(decision, dict):
        fail("KSQ010 smoke missing codebook decision")
    if decision.get("exported_diagnostic_class") != "CODEBOOK_POSITIVE_FAILED":
        fail("KSQ010 smoke exported diagnostic class changed")
    if decision.get("behavior_ready") is not False:
        fail("KSQ010 smoke must not be behavior ready")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ010 smoke must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ010 smoke must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ010 smoke must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ010 smoke must not license mechanism claims")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ010 smoke structural gate must pass")
    if structural.get("record_count") != 360 or structural.get("source_count") != 10:
        fail("KSQ010 smoke must keep the 360-row, 10-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ010 smoke missing criteria")
    expected_criteria = {
        "smoke_mode": True,
        "structural_passed": True,
        "full_source_count_is_40": False,
        "source_disjoint_holdout": True,
        "selected_prompt_audit_passed": True,
        "exact_codebook_bridge_passed": False,
        "bridge_vs_answer_for_alt_passed": False,
        "answer_for_controls_passed": False,
        "codebook_locality_controls_passed": False,
        "abstain_controls_passed": False,
        "all_panels_passed": False,
        "source_disjoint_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    if criteria != expected_criteria:
        fail("KSQ010 smoke criteria changed")
    if summary["selection"]["selected_template"] != "tag_rows":
        fail("KSQ010 smoke selected template changed")
    if summary.get("failed_panels") != [
        "exact_codebook_bridge",
        "missing_code_value",
        "conflicting_entity_codes",
        "conflicting_code_values",
        "uncounted_entity_code",
        "uncounted_code_value",
        "quoted_bridge_rows",
        "counted_answer_for_only",
        "bridge_vs_answer_for_alt",
    ]:
        fail("KSQ010 smoke failed panels changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "exact_codebook_bridge",
            "missing_entity_code",
            "missing_code_value",
            "unrelated_entity_code",
            "conflicting_entity_codes",
            "conflicting_code_values",
            "uncounted_entity_code",
            "uncounted_code_value",
            "quoted_bridge_rows",
            "counted_answer_for_only",
            "bridge_vs_answer_for_alt",
            "query_only_control",
        ]
    }
    expected_panel_counts = {
        "exact_codebook_bridge": {"evidence_answer": 8, "unparsed": 2},
        "missing_entity_code": {"control_abstain": 9, "unparsed": 1},
        "missing_code_value": {"unparsed": 10},
        "unrelated_entity_code": {"control_abstain": 10},
        "conflicting_entity_codes": {"conflict_value_selected": 10},
        "conflicting_code_values": {"conflict_value_selected": 10},
        "uncounted_entity_code": {"unparsed": 10},
        "uncounted_code_value": {
            "uncounted_code_value_reproduced": 4,
            "unparsed": 6,
        },
        "quoted_bridge_rows": {
            "quoted_bridge_reproduced": 3,
            "unparsed": 7,
        },
        "counted_answer_for_only": {"counted_answer_for_reproduced": 10},
        "bridge_vs_answer_for_alt": {
            "answer_for_alt_overrode_bridge": 9,
            "unparsed": 1,
        },
        "query_only_control": {"control_abstain": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ010 smoke panel label counts changed")
    return {
        "status": "smoke_failed_codebook_positive",
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq011_first_run_result_layer() -> dict[str, Any]:
    if not KSQ011_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ011 first-run result "
            f"{KSQ011_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ011_STATUS_PATH.exists():
        fail(
            "missing KSQ011 status card "
            f"{KSQ011_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ011_PREREG_PATH.exists():
        fail(
            "missing KSQ011 preregistration "
            f"{KSQ011_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ011_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ011 first-run result schema_version must be 1")
    if result.get("card_id") != "KSQ011":
        fail("KSQ011 first-run card_id changed")
    if result.get("parent_card_id") != "KSQ010":
        fail("KSQ011 first-run parent_card_id changed")
    if result.get("candidate_id") != "ksq011_answer_for_syntax_ablation":
        fail("KSQ011 first-run candidate_id changed")
    if result.get("run_type") != "ksq011_answer_for_syntax_ablation_structural_gate":
        fail("KSQ011 first-run result must be the structural gate result")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ011 first-run result missing summary")
    if summary.get("diagnostic_class") != "structural_passed":
        fail("KSQ011 first-run diagnostic class changed")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ011 first-run structural gate must pass")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ011 first-run missing criteria")
    expected_criteria = {
        "structural_passed": True,
        "full_source_count_is_40": True,
        "source_disjoint_holdout": True,
        "hidden_state_license": False,
    }
    if criteria != expected_criteria:
        fail("KSQ011 first-run criteria changed")
    decision = summary.get("syntax_ablation_decision")
    if not isinstance(decision, dict):
        fail("KSQ011 first-run missing syntax-ablation decision")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ011 first-run must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ011 first-run must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ011 first-run must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ011 first-run must not license mechanism claims")

    if structural.get("record_count") != 480:
        fail("KSQ011 first-run result must keep the full 480-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ011 first-run result must keep the full 40-source substrate")
    expected_panel_counts = {
        "adversary_only_exact_answer_for": 40,
        "exact_bridge": 40,
        "other_answer_to_alt": 40,
        "other_bare_alt_mention": 40,
        "other_colon_answer_for_alt": 40,
        "other_entity_equals_alt": 40,
        "other_exact_answer_for_alt": 40,
        "other_prose_value_alt": 40,
        "other_quoted_answer_for_alt": 40,
        "other_spaced_answer_for_alt": 40,
        "other_value_for_alt": 40,
        "query_only_control": 40,
    }
    if structural.get("panel_counts") != expected_panel_counts:
        fail("KSQ011 first-run panel counts changed")
    if structural.get("template_counts") != {"tag_rows": 480}:
        fail("KSQ011 first-run template counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ011 first-run source split changed")
    return {
        "status": "structural_passed",
        "diagnostic_class": summary["diagnostic_class"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq011_smoke_result_layer() -> dict[str, Any]:
    if not KSQ011_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ011 10-source smoke result "
            f"{KSQ011_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ011_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ011 smoke result schema_version must be 1")
    if result.get("card_id") != "KSQ011":
        fail("KSQ011 smoke card_id changed")
    if result.get("parent_card_id") != "KSQ010":
        fail("KSQ011 smoke parent_card_id changed")
    if result.get("candidate_id") != "ksq011_answer_for_syntax_ablation":
        fail("KSQ011 smoke candidate_id changed")
    if result.get("run_type") != "ksq011_answer_for_syntax_ablation_behavior":
        fail("KSQ011 smoke result must be a behavior run")
    if result.get("full_run") is not False:
        fail("KSQ011 smoke result must remain the 10-source smoke")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ011 smoke result missing summary")
    if summary.get("diagnostic_class") != "function_assignment_answer_channel_dominance":
        fail("KSQ011 smoke diagnostic class changed")
    decision = summary.get("syntax_ablation_decision")
    if not isinstance(decision, dict):
        fail("KSQ011 smoke missing syntax-ablation decision")
    if decision.get("exported_diagnostic_class") != "FUNCTION_ASSIGNMENT_ANSWER_CHANNEL_DOMINANCE":
        fail("KSQ011 smoke exported diagnostic class changed")
    if decision.get("behavior_ready") is not False:
        fail("KSQ011 smoke must not be behavior ready")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ011 smoke must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ011 smoke must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ011 smoke must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ011 smoke must not license mechanism claims")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ011 smoke structural gate must pass")
    if structural.get("record_count") != 120 or structural.get("source_count") != 10:
        fail("KSQ011 smoke must keep the 120-row, 10-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ011 smoke missing criteria")
    expected_criteria = {
        "smoke_mode": True,
        "structural_passed": True,
        "full_source_count_is_40": False,
        "source_disjoint_holdout": True,
        "selected_prompt_audit_passed": True,
        "exact_bridge_passed": True,
        "exact_answer_for_variants_passed": False,
        "function_variants_passed": False,
        "nonfunction_variants_passed": False,
        "query_only_control_passed": True,
        "all_panels_passed": False,
        "source_disjoint_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    if criteria != expected_criteria:
        fail("KSQ011 smoke criteria changed")
    if summary["selection"]["selected_template"] != "tag_rows":
        fail("KSQ011 smoke selected template changed")
    if summary.get("failed_panels") != [
        "other_exact_answer_for_alt",
        "other_spaced_answer_for_alt",
        "other_colon_answer_for_alt",
        "other_answer_to_alt",
        "other_value_for_alt",
        "other_prose_value_alt",
        "other_quoted_answer_for_alt",
        "other_bare_alt_mention",
        "adversary_only_exact_answer_for",
    ]:
        fail("KSQ011 smoke failed panels changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "exact_bridge",
            "other_exact_answer_for_alt",
            "other_spaced_answer_for_alt",
            "other_colon_answer_for_alt",
            "other_answer_to_alt",
            "other_value_for_alt",
            "other_entity_equals_alt",
            "other_prose_value_alt",
            "other_quoted_answer_for_alt",
            "other_bare_alt_mention",
            "adversary_only_exact_answer_for",
            "query_only_control",
        ]
    }
    expected_panel_counts = {
        "exact_bridge": {"evidence_answer": 9, "unparsed": 1},
        "other_exact_answer_for_alt": {
            "adversary_alt_overrode_bridge": 9,
            "unparsed": 1,
        },
        "other_spaced_answer_for_alt": {"adversary_alt_overrode_bridge": 10},
        "other_colon_answer_for_alt": {
            "adversary_alt_overrode_bridge": 9,
            "evidence_answer": 1,
        },
        "other_answer_to_alt": {
            "adversary_alt_overrode_bridge": 9,
            "unparsed": 1,
        },
        "other_value_for_alt": {
            "adversary_alt_overrode_bridge": 8,
            "evidence_answer": 1,
            "unparsed": 1,
        },
        "other_entity_equals_alt": {
            "abstain": 1,
            "evidence_answer": 8,
            "unparsed": 1,
        },
        "other_prose_value_alt": {
            "adversary_alt_overrode_bridge": 7,
            "evidence_answer": 2,
            "unparsed": 1,
        },
        "other_quoted_answer_for_alt": {
            "adversary_alt_overrode_bridge": 9,
            "unparsed": 1,
        },
        "other_bare_alt_mention": {
            "evidence_answer": 8,
            "unparsed": 2,
        },
        "adversary_only_exact_answer_for": {
            "adversary_only_answer_for_reproduced": 9,
            "unparsed": 1,
        },
        "query_only_control": {"control_abstain": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ011 smoke panel label counts changed")
    return {
        "status": "smoke_failed_function_assignment_answer_channel",
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq012_first_run_result_layer() -> dict[str, Any]:
    if not KSQ012_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ012 first-run result "
            f"{KSQ012_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ012_STATUS_PATH.exists():
        fail(
            "missing KSQ012 status card "
            f"{KSQ012_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ012_PREREG_PATH.exists():
        fail(
            "missing KSQ012 preregistration "
            f"{KSQ012_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ012_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ012 first-run result schema_version must be 1")
    if result.get("card_id") != "KSQ012":
        fail("KSQ012 first-run card_id changed")
    if result.get("parent_card_id") != "KSQ011":
        fail("KSQ012 first-run parent_card_id changed")
    if result.get("candidate_id") != "ksq012_function_assignment_wrapper_repair":
        fail("KSQ012 first-run candidate_id changed")
    if result.get("run_type") != "ksq012_function_assignment_wrapper_repair_structural_gate":
        fail("KSQ012 first-run result must be the structural gate result")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ012 first-run result missing summary")
    if summary.get("diagnostic_class") != "structural_passed":
        fail("KSQ012 first-run diagnostic class changed")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ012 first-run structural gate must pass")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ012 first-run missing criteria")
    expected_criteria = {
        "structural_passed": True,
        "full_source_count_is_40": True,
        "source_disjoint_holdout": True,
        "hidden_state_license": False,
    }
    if criteria != expected_criteria:
        fail("KSQ012 first-run criteria changed")
    decision = summary.get("wrapper_repair_decision")
    if not isinstance(decision, dict):
        fail("KSQ012 first-run missing wrapper-repair decision")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ012 first-run must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ012 first-run must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ012 first-run must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ012 first-run must not license mechanism claims")

    if structural.get("record_count") != 480:
        fail("KSQ012 first-run result must keep the full 480-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ012 first-run result must keep the full 40-source substrate")
    expected_panel_counts = {
        "assignment_only_inactive_control": 40,
        "below_cut_answer_for_alt": 40,
        "comment_mark_answer_for_alt": 40,
        "detached_function_then_value_alt": 40,
        "exact_bridge": 40,
        "fenced_text_answer_for_alt": 40,
        "inactive_block_answer_for_alt": 40,
        "masked_function_value_bank_alt": 40,
        "query_only_control": 40,
        "raw_answer_for_alt": 40,
        "split_assignment_only_control": 40,
        "unrelated_entity_answer_for_alt": 40,
    }
    if structural.get("panel_counts") != expected_panel_counts:
        fail("KSQ012 first-run panel counts changed")
    if structural.get("template_counts") != {"tag_rows": 480}:
        fail("KSQ012 first-run template counts changed")
    if structural.get("split_source_counts") != {"calibration": 8, "discovery": 24, "holdout": 8}:
        fail("KSQ012 first-run source split changed")
    return {
        "status": "structural_passed",
        "diagnostic_class": summary["diagnostic_class"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq012_smoke_result_layer() -> dict[str, Any]:
    if not KSQ012_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ012 10-source smoke result "
            f"{KSQ012_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ012_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ012 smoke result schema_version must be 1")
    if result.get("card_id") != "KSQ012":
        fail("KSQ012 smoke card_id changed")
    if result.get("parent_card_id") != "KSQ011":
        fail("KSQ012 smoke parent_card_id changed")
    if result.get("candidate_id") != "ksq012_function_assignment_wrapper_repair":
        fail("KSQ012 smoke candidate_id changed")
    if result.get("run_type") != "ksq012_function_assignment_wrapper_repair_behavior":
        fail("KSQ012 smoke result must be a behavior run")
    if result.get("full_run") is not False:
        fail("KSQ012 smoke result must remain the 10-source smoke")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ012 smoke result missing summary")
    if summary.get("diagnostic_class") != "function_assignment_wrapper_control_and_repair_leak":
        fail("KSQ012 smoke diagnostic class changed")
    decision = summary.get("wrapper_repair_decision")
    if not isinstance(decision, dict):
        fail("KSQ012 smoke missing wrapper-repair decision")
    if decision.get("exported_diagnostic_class") != "FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK":
        fail("KSQ012 smoke exported diagnostic class changed")
    if decision.get("behavior_ready") is not False:
        fail("KSQ012 smoke must not be behavior ready")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ012 smoke must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ012 smoke must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ012 smoke must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ012 smoke must not license mechanism claims")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ012 smoke structural gate must pass")
    if structural.get("record_count") != 120 or structural.get("source_count") != 10:
        fail("KSQ012 smoke must keep the 120-row, 10-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ012 smoke missing criteria")
    expected_criteria = {
        "smoke_mode": True,
        "structural_passed": True,
        "full_source_count_is_40": False,
        "source_disjoint_holdout": True,
        "selected_prompt_audit_passed": True,
        "exact_bridge_passed": True,
        "raw_answer_channel_positive_control_passed": True,
        "repair_panels_passed": False,
        "wrapper_controls_passed": False,
        "all_panels_passed": False,
        "source_disjoint_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    if criteria != expected_criteria:
        fail("KSQ012 smoke criteria changed")
    if summary["selection"]["selected_template"] != "tag_rows":
        fail("KSQ012 smoke selected template changed")
    if summary.get("failed_panels") != [
        "inactive_block_answer_for_alt",
        "comment_mark_answer_for_alt",
        "fenced_text_answer_for_alt",
        "below_cut_answer_for_alt",
        "detached_function_then_value_alt",
        "masked_function_value_bank_alt",
        "unrelated_entity_answer_for_alt",
        "assignment_only_inactive_control",
    ]:
        fail("KSQ012 smoke failed panels changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "exact_bridge",
            "raw_answer_for_alt",
            "inactive_block_answer_for_alt",
            "comment_mark_answer_for_alt",
            "fenced_text_answer_for_alt",
            "below_cut_answer_for_alt",
            "detached_function_then_value_alt",
            "masked_function_value_bank_alt",
            "unrelated_entity_answer_for_alt",
            "assignment_only_inactive_control",
            "split_assignment_only_control",
            "query_only_control",
        ]
    }
    expected_panel_counts = {
        "exact_bridge": {"evidence_answer": 9, "unparsed": 1},
        "raw_answer_for_alt": {
            "raw_answer_channel_reproduced": 9,
            "unparsed": 1,
        },
        "inactive_block_answer_for_alt": {
            "adversary_alt_overrode_bridge": 9,
            "unparsed": 1,
        },
        "comment_mark_answer_for_alt": {
            "abstain": 1,
            "adversary_alt_overrode_bridge": 8,
            "unparsed": 1,
        },
        "fenced_text_answer_for_alt": {
            "abstain": 1,
            "adversary_alt_overrode_bridge": 9,
        },
        "below_cut_answer_for_alt": {
            "adversary_alt_overrode_bridge": 9,
            "unparsed": 1,
        },
        "detached_function_then_value_alt": {
            "abstain": 1,
            "adversary_alt_overrode_bridge": 7,
            "evidence_answer": 2,
        },
        "masked_function_value_bank_alt": {
            "abstain": 6,
            "evidence_answer": 3,
            "unparsed": 1,
        },
        "unrelated_entity_answer_for_alt": {
            "adversary_alt_overrode_bridge": 6,
            "evidence_answer": 4,
        },
        "assignment_only_inactive_control": {
            "control_abstain": 1,
            "control_reproduced_value": 8,
            "unparsed": 1,
        },
        "split_assignment_only_control": {"control_abstain": 10},
        "query_only_control": {"control_abstain": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ012 smoke panel label counts changed")
    return {
        "status": "smoke_failed_function_assignment_wrapper_control_and_repair_leak",
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq013_first_run_result_layer() -> dict[str, Any]:
    if not KSQ013_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ013 first-run result "
            f"{KSQ013_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ013_STATUS_PATH.exists():
        fail(
            "missing KSQ013 status card "
            f"{KSQ013_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ013_PREREG_PATH.exists():
        fail(
            "missing KSQ013 preregistration "
            f"{KSQ013_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ013_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ013 first-run result schema_version must be 1")
    if result.get("card_id") != "KSQ013":
        fail("KSQ013 first-run card_id changed")
    if result.get("parent_card_id") != "KSQ012":
        fail("KSQ013 first-run parent_card_id changed")
    if result.get("candidate_id") != "ksq013_nonfunction_representation_screen":
        fail("KSQ013 first-run candidate_id changed")
    if result.get("run_type") != "ksq013_nonfunction_representation_screen_structural_gate":
        fail("KSQ013 first-run result must be the structural gate result")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ013 first-run result missing summary")
    if summary.get("diagnostic_class") != "structural_passed":
        fail("KSQ013 first-run diagnostic class changed")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ013 first-run structural gate must pass")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ013 first-run missing criteria")
    expected_criteria = {
        "structural_passed": True,
        "full_source_count_is_40": True,
        "source_disjoint_holdout": True,
        "hidden_state_license": False,
    }
    if criteria != expected_criteria:
        fail("KSQ013 first-run criteria changed")
    decision = summary.get("nonfunction_representation_decision")
    if not isinstance(decision, dict):
        fail("KSQ013 first-run missing nonfunction representation decision")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ013 first-run must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ013 first-run must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ013 first-run must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ013 first-run must not license mechanism claims")

    if structural.get("record_count") != 480:
        fail("KSQ013 first-run result must keep the full 480-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ013 first-run result must keep the full 40-source substrate")
    expected_panel_counts = {
        "bare_alt_mention": 40,
        "decoy_entity_pair_alt": 40,
        "decoy_pair_only_control": 40,
        "entity_value_separate_alt": 40,
        "entity_value_separate_only_control": 40,
        "entity_value_slash_alt": 40,
        "exact_bridge": 40,
        "metadata_value_alt": 40,
        "query_only_control": 40,
        "raw_answer_for_alt": 40,
        "value_bank_alt": 40,
        "value_bank_only_control": 40,
    }
    if structural.get("panel_counts") != expected_panel_counts:
        fail("KSQ013 first-run panel counts changed")
    if structural.get("template_counts") != {"tag_rows": 480}:
        fail("KSQ013 first-run template counts changed")
    if structural.get("split_source_counts") != {
        "calibration": 8,
        "discovery": 24,
        "holdout": 8,
    }:
        fail("KSQ013 first-run source split changed")
    return {
        "status": "structural_passed",
        "diagnostic_class": summary["diagnostic_class"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq013_smoke_result_layer() -> dict[str, Any]:
    if not KSQ013_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ013 10-source smoke result "
            f"{KSQ013_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ013_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ013 smoke result schema_version must be 1")
    if result.get("card_id") != "KSQ013":
        fail("KSQ013 smoke card_id changed")
    if result.get("parent_card_id") != "KSQ012":
        fail("KSQ013 smoke parent_card_id changed")
    if result.get("candidate_id") != "ksq013_nonfunction_representation_screen":
        fail("KSQ013 smoke candidate_id changed")
    if result.get("run_type") != "ksq013_nonfunction_representation_screen_behavior":
        fail("KSQ013 smoke result must be a behavior run")
    if result.get("full_run") is not False:
        fail("KSQ013 smoke result must remain the 10-source smoke")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ013 smoke result missing summary")
    if summary.get("diagnostic_class") != "nonfunction_representation_control_and_repair_leak":
        fail("KSQ013 smoke diagnostic class changed")
    decision = summary.get("nonfunction_representation_decision")
    if not isinstance(decision, dict):
        fail("KSQ013 smoke missing nonfunction representation decision")
    if (
        decision.get("exported_diagnostic_class")
        != "NONFUNCTION_REPRESENTATION_CONTROL_AND_REPAIR_LEAK"
    ):
        fail("KSQ013 smoke exported diagnostic class changed")
    if decision.get("behavior_ready") is not False:
        fail("KSQ013 smoke must not be behavior ready")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ013 smoke must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ013 smoke must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ013 smoke must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ013 smoke must not license mechanism claims")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ013 smoke structural gate must pass")
    if structural.get("record_count") != 120 or structural.get("source_count") != 10:
        fail("KSQ013 smoke must keep the 120-row, 10-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ013 smoke missing criteria")
    expected_criteria = {
        "smoke_mode": True,
        "structural_passed": True,
        "full_source_count_is_40": False,
        "source_disjoint_holdout": True,
        "selected_prompt_audit_passed": True,
        "exact_bridge_passed": True,
        "raw_answer_channel_positive_control_passed": True,
        "nonfunction_panels_passed": False,
        "nonfunction_controls_passed": False,
        "all_panels_passed": False,
        "source_disjoint_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    if criteria != expected_criteria:
        fail("KSQ013 smoke criteria changed")
    if summary["selection"]["selected_template"] != "tag_rows":
        fail("KSQ013 smoke selected template changed")
    if summary.get("failed_panels") != [
        "value_bank_alt",
        "metadata_value_alt",
        "decoy_entity_pair_alt",
        "entity_value_separate_alt",
        "bare_alt_mention",
        "entity_value_separate_only_control",
    ]:
        fail("KSQ013 smoke failed panels changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "exact_bridge",
            "raw_answer_for_alt",
            "value_bank_alt",
            "metadata_value_alt",
            "decoy_entity_pair_alt",
            "entity_value_separate_alt",
            "entity_value_slash_alt",
            "bare_alt_mention",
            "value_bank_only_control",
            "decoy_pair_only_control",
            "entity_value_separate_only_control",
            "query_only_control",
        ]
    }
    expected_panel_counts = {
        "exact_bridge": {"evidence_answer": 9, "unparsed": 1},
        "raw_answer_for_alt": {
            "raw_answer_channel_reproduced": 9,
            "unparsed": 1,
        },
        "value_bank_alt": {"evidence_answer": 8, "unparsed": 2},
        "metadata_value_alt": {
            "abstain": 1,
            "evidence_answer": 7,
            "unparsed": 2,
        },
        "decoy_entity_pair_alt": {"evidence_answer": 8, "unparsed": 2},
        "entity_value_separate_alt": {
            "abstain": 1,
            "evidence_answer": 6,
            "unparsed": 3,
        },
        "entity_value_slash_alt": {"evidence_answer": 9, "unparsed": 1},
        "bare_alt_mention": {"evidence_answer": 8, "unparsed": 2},
        "value_bank_only_control": {"control_abstain": 10},
        "decoy_pair_only_control": {"control_abstain": 10},
        "entity_value_separate_only_control": {
            "control_abstain": 6,
            "control_reproduced_value": 4,
        },
        "query_only_control": {"control_abstain": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ013 smoke panel label counts changed")
    return {
        "status": "smoke_failed_nonfunction_representation_control_and_repair_leak",
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq014_first_run_result_layer() -> dict[str, Any]:
    if not KSQ014_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ014 first-run result "
            f"{KSQ014_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ014_STATUS_PATH.exists():
        fail(
            "missing KSQ014 status card "
            f"{KSQ014_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ014_PREREG_PATH.exists():
        fail(
            "missing KSQ014 preregistration "
            f"{KSQ014_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ014_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ014 first-run result schema_version must be 1")
    if result.get("card_id") != "KSQ014":
        fail("KSQ014 first-run card_id changed")
    if result.get("parent_card_id") != "KSQ013":
        fail("KSQ014 first-run parent_card_id changed")
    if result.get("candidate_id") != "ksq014_slash_locality_packet":
        fail("KSQ014 first-run candidate_id changed")
    if result.get("run_type") != "ksq014_slash_locality_packet_structural_gate":
        fail("KSQ014 first-run result must be the structural gate result")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ014 first-run result missing summary")
    if summary.get("diagnostic_class") != "structural_passed":
        fail("KSQ014 first-run diagnostic class changed")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ014 first-run structural gate must pass")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ014 first-run missing criteria")
    expected_criteria = {
        "structural_passed": True,
        "full_source_count_is_40": True,
        "source_disjoint_holdout": True,
        "hidden_state_license": False,
    }
    if criteria != expected_criteria:
        fail("KSQ014 first-run criteria changed")
    decision = summary.get("slash_locality_decision")
    if not isinstance(decision, dict):
        fail("KSQ014 first-run missing slash locality decision")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ014 first-run must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ014 first-run must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ014 first-run must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ014 first-run must not license mechanism claims")

    if structural.get("record_count") != 440:
        fail("KSQ014 first-run result must keep the full 440-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ014 first-run result must keep the full 40-source substrate")
    expected_panel_counts = {
        "bare_slash_entity_alt": 40,
        "bare_slash_entity_only_control": 40,
        "catalog_slash_decoy_alt": 40,
        "catalog_slash_decoy_only_control": 40,
        "catalog_slash_entity_alt": 40,
        "catalog_slash_entity_only_control": 40,
        "catalog_slash_reversed_entity_alt": 40,
        "catalog_slash_reversed_entity_only_control": 40,
        "exact_bridge": 40,
        "query_only_control": 40,
        "raw_answer_for_alt": 40,
    }
    if structural.get("panel_counts") != expected_panel_counts:
        fail("KSQ014 first-run panel counts changed")
    if structural.get("template_counts") != {"tag_rows": 440}:
        fail("KSQ014 first-run template counts changed")
    if structural.get("split_source_counts") != {
        "calibration": 8,
        "discovery": 24,
        "holdout": 8,
    }:
        fail("KSQ014 first-run source split changed")
    return {
        "status": "structural_passed",
        "diagnostic_class": summary["diagnostic_class"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq014_smoke_result_layer() -> dict[str, Any]:
    if not KSQ014_SMOKE_LIMIT10_RESULT_PATH.exists():
        fail(
            "missing KSQ014 10-source smoke result "
            f"{KSQ014_SMOKE_LIMIT10_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ014_SMOKE_LIMIT10_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ014 smoke result schema_version must be 1")
    if result.get("card_id") != "KSQ014":
        fail("KSQ014 smoke card_id changed")
    if result.get("parent_card_id") != "KSQ013":
        fail("KSQ014 smoke parent_card_id changed")
    if result.get("candidate_id") != "ksq014_slash_locality_packet":
        fail("KSQ014 smoke candidate_id changed")
    if result.get("run_type") != "ksq014_slash_locality_packet_behavior":
        fail("KSQ014 smoke result must be a behavior run")
    if result.get("full_run") is not False:
        fail("KSQ014 smoke result must remain the 10-source smoke")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ014 smoke result missing summary")
    if summary.get("diagnostic_class") != "slash_locality_bridge_loss":
        fail("KSQ014 smoke diagnostic class changed")
    decision = summary.get("slash_locality_decision")
    if not isinstance(decision, dict):
        fail("KSQ014 smoke missing slash locality decision")
    if decision.get("exported_diagnostic_class") != "SLASH_LOCALITY_BRIDGE_LOSS":
        fail("KSQ014 smoke exported diagnostic class changed")
    if decision.get("behavior_ready") is not False:
        fail("KSQ014 smoke must not be behavior ready")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ014 smoke must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ014 smoke must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ014 smoke must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ014 smoke must not license mechanism claims")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ014 smoke structural gate must pass")
    if structural.get("record_count") != 110 or structural.get("source_count") != 10:
        fail("KSQ014 smoke must keep the 110-row, 10-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ014 smoke missing criteria")
    expected_criteria = {
        "smoke_mode": True,
        "structural_passed": True,
        "full_source_count_is_40": False,
        "source_disjoint_holdout": True,
        "selected_prompt_audit_passed": True,
        "exact_bridge_passed": True,
        "raw_answer_channel_positive_control_passed": True,
        "slash_bridge_panels_passed": False,
        "slash_controls_passed": True,
        "all_panels_passed": False,
        "source_disjoint_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    if criteria != expected_criteria:
        fail("KSQ014 smoke criteria changed")
    if summary["selection"]["selected_template"] != "tag_rows":
        fail("KSQ014 smoke selected template changed")
    if summary.get("failed_panels") != ["bare_slash_entity_alt"]:
        fail("KSQ014 smoke failed panels changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "exact_bridge",
            "raw_answer_for_alt",
            "catalog_slash_entity_alt",
            "catalog_slash_decoy_alt",
            "bare_slash_entity_alt",
            "catalog_slash_reversed_entity_alt",
            "catalog_slash_entity_only_control",
            "catalog_slash_decoy_only_control",
            "bare_slash_entity_only_control",
            "catalog_slash_reversed_entity_only_control",
            "query_only_control",
        ]
    }
    expected_panel_counts = {
        "exact_bridge": {"evidence_answer": 9, "unparsed": 1},
        "raw_answer_for_alt": {
            "raw_answer_channel_reproduced": 9,
            "unparsed": 1,
        },
        "catalog_slash_entity_alt": {"evidence_answer": 9, "unparsed": 1},
        "catalog_slash_decoy_alt": {"abstain": 1, "evidence_answer": 9},
        "bare_slash_entity_alt": {"evidence_answer": 6, "unparsed": 4},
        "catalog_slash_reversed_entity_alt": {
            "evidence_answer": 9,
            "unparsed": 1,
        },
        "catalog_slash_entity_only_control": {"control_abstain": 10},
        "catalog_slash_decoy_only_control": {"control_abstain": 10},
        "bare_slash_entity_only_control": {"control_abstain": 10},
        "catalog_slash_reversed_entity_only_control": {"control_abstain": 10},
        "query_only_control": {"control_abstain": 10},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ014 smoke panel label counts changed")
    return {
        "status": "smoke_failed_bare_slash_bridge_loss",
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq015_first_run_result_layer() -> dict[str, Any]:
    if not KSQ015_FIRST_RUN_RESULT_PATH.exists():
        fail(
            "missing KSQ015 first-run result "
            f"{KSQ015_FIRST_RUN_RESULT_PATH.relative_to(ROOT)}"
        )
    if not KSQ015_STATUS_PATH.exists():
        fail(
            "missing KSQ015 status card "
            f"{KSQ015_STATUS_PATH.relative_to(ROOT)}"
        )
    if not KSQ015_PREREG_PATH.exists():
        fail(
            "missing KSQ015 preregistration "
            f"{KSQ015_PREREG_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ015_FIRST_RUN_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ015 first-run result schema_version must be 1")
    if result.get("card_id") != "KSQ015":
        fail("KSQ015 first-run card_id changed")
    if result.get("parent_card_id") != "KSQ014":
        fail("KSQ015 first-run parent_card_id changed")
    if result.get("candidate_id") != "ksq015_catalog_slash_full_source_packet":
        fail("KSQ015 first-run candidate_id changed")
    if result.get("run_type") != "ksq015_catalog_slash_full_source_structural_gate":
        fail("KSQ015 first-run result must be the structural gate result")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ015 first-run result missing summary")
    if summary.get("diagnostic_class") != "structural_passed":
        fail("KSQ015 first-run diagnostic class changed")
    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ015 first-run structural gate must pass")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ015 first-run missing criteria")
    expected_criteria = {
        "structural_passed": True,
        "full_source_count_is_40": True,
        "source_disjoint_holdout": True,
        "hidden_state_license": False,
    }
    if criteria != expected_criteria:
        fail("KSQ015 first-run criteria changed")
    decision = summary.get("catalog_slash_decision")
    if not isinstance(decision, dict):
        fail("KSQ015 first-run missing catalog slash decision")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ015 first-run must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ015 first-run must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ015 first-run must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ015 first-run must not license mechanism claims")

    if structural.get("record_count") != 360:
        fail("KSQ015 first-run result must keep the full 360-row substrate")
    if structural.get("source_count") != 40:
        fail("KSQ015 first-run result must keep the full 40-source substrate")
    expected_panel_counts = {
        "catalog_slash_decoy_alt": 40,
        "catalog_slash_decoy_only_control": 40,
        "catalog_slash_entity_alt": 40,
        "catalog_slash_entity_only_control": 40,
        "catalog_slash_reversed_entity_alt": 40,
        "catalog_slash_reversed_entity_only_control": 40,
        "exact_bridge": 40,
        "query_only_control": 40,
        "raw_answer_for_alt": 40,
    }
    if structural.get("panel_counts") != expected_panel_counts:
        fail("KSQ015 first-run panel counts changed")
    if structural.get("template_counts") != {"tag_rows": 360}:
        fail("KSQ015 first-run template counts changed")
    if structural.get("split_source_counts") != {
        "calibration": 8,
        "discovery": 24,
        "holdout": 8,
    }:
        fail("KSQ015 first-run source split changed")
    return {
        "status": "structural_passed",
        "diagnostic_class": summary["diagnostic_class"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "panel_count": len(structural["panel_counts"]),
        "template_count": len(structural["template_counts"]),
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def validate_ksq015_full_behavior_result_layer() -> dict[str, Any]:
    if not KSQ015_FULL_BEHAVIOR_RESULT_PATH.exists():
        fail(
            "missing KSQ015 full behavior result "
            f"{KSQ015_FULL_BEHAVIOR_RESULT_PATH.relative_to(ROOT)}"
        )
    result = load_json(KSQ015_FULL_BEHAVIOR_RESULT_PATH)
    if result.get("schema_version") != 1:
        fail("KSQ015 full behavior result schema_version must be 1")
    if result.get("card_id") != "KSQ015":
        fail("KSQ015 full behavior card_id changed")
    if result.get("parent_card_id") != "KSQ014":
        fail("KSQ015 full behavior parent_card_id changed")
    if result.get("candidate_id") != "ksq015_catalog_slash_full_source_packet":
        fail("KSQ015 full behavior candidate_id changed")
    if result.get("run_type") != "ksq015_catalog_slash_full_source_behavior":
        fail("KSQ015 full behavior result must be a behavior run")
    if result.get("full_run") is not True:
        fail("KSQ015 full behavior result must be full-source")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        fail("KSQ015 full behavior result missing summary")
    if summary.get("diagnostic_class") != "catalog_slash_bridge_positive_failed":
        fail("KSQ015 full behavior diagnostic class changed")
    decision = summary.get("catalog_slash_decision")
    if not isinstance(decision, dict):
        fail("KSQ015 full behavior missing catalog slash decision")
    if decision.get("exported_diagnostic_class") != "CATALOG_SLASH_BRIDGE_POSITIVE_FAILED":
        fail("KSQ015 full behavior exported diagnostic class changed")
    if decision.get("behavior_ready") is not False:
        fail("KSQ015 full behavior must not be behavior ready")
    if decision.get("signature_screen_allowed") is not False:
        fail("KSQ015 full behavior must not license signature screening")
    if decision.get("hidden_state_claim_allowed") is not False:
        fail("KSQ015 full behavior must not license hidden-state claims")
    if decision.get("intervention_allowed") is not False:
        fail("KSQ015 full behavior must not license interventions")
    if decision.get("mechanism_claim_allowed") is not False:
        fail("KSQ015 full behavior must not license mechanism claims")

    structural = summary.get("structural")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        fail("KSQ015 full behavior structural gate must pass")
    if structural.get("record_count") != 360 or structural.get("source_count") != 40:
        fail("KSQ015 full behavior must keep the 360-row, 40-source scope")
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        fail("KSQ015 full behavior missing criteria")
    expected_criteria = {
        "full_run_mode": True,
        "structural_passed": True,
        "full_source_count_is_40": True,
        "source_disjoint_holdout": True,
        "selected_prompt_audit_passed": True,
        "exact_bridge_passed": False,
        "raw_answer_channel_positive_control_passed": True,
        "catalog_slash_bridge_panels_passed": False,
        "catalog_slash_controls_passed": True,
        "all_panels_passed": False,
        "source_disjoint_holdout_passed": False,
        "candidate_and_output_margins_reported": True,
    }
    if criteria != expected_criteria:
        fail("KSQ015 full behavior criteria changed")
    if summary["selection"]["selected_template"] != "tag_rows":
        fail("KSQ015 full behavior selected template changed")
    if summary.get("failed_panels") != ["exact_bridge", "catalog_slash_reversed_entity_alt"]:
        fail("KSQ015 full behavior failed panels changed")
    selected = summary["selected_template_summary"]
    panel_counts = {
        panel_name: selected["panels"][panel_name]["label_counts"]
        for panel_name in [
            "exact_bridge",
            "raw_answer_for_alt",
            "catalog_slash_entity_alt",
            "catalog_slash_decoy_alt",
            "catalog_slash_reversed_entity_alt",
            "catalog_slash_entity_only_control",
            "catalog_slash_decoy_only_control",
            "catalog_slash_reversed_entity_only_control",
            "query_only_control",
        ]
    }
    expected_panel_counts = {
        "exact_bridge": {"evidence_answer": 35, "unparsed": 5},
        "raw_answer_for_alt": {
            "raw_answer_channel_reproduced": 39,
            "unparsed": 1,
        },
        "catalog_slash_entity_alt": {
            "abstain": 1,
            "evidence_answer": 36,
            "unparsed": 3,
        },
        "catalog_slash_decoy_alt": {
            "abstain": 3,
            "evidence_answer": 35,
            "unparsed": 2,
        },
        "catalog_slash_reversed_entity_alt": {
            "evidence_answer": 35,
            "unparsed": 5,
        },
        "catalog_slash_entity_only_control": {"control_abstain": 40},
        "catalog_slash_decoy_only_control": {"control_abstain": 40},
        "catalog_slash_reversed_entity_only_control": {"control_abstain": 40},
        "query_only_control": {"control_abstain": 40},
    }
    if panel_counts != expected_panel_counts:
        fail("KSQ015 full behavior panel label counts changed")
    holdout_counts = {
        panel_name: selected["holdout_panels"][panel_name]["label_counts"]
        for panel_name in [
            "exact_bridge",
            "raw_answer_for_alt",
            "catalog_slash_entity_alt",
            "catalog_slash_decoy_alt",
            "catalog_slash_reversed_entity_alt",
            "catalog_slash_entity_only_control",
            "catalog_slash_decoy_only_control",
            "catalog_slash_reversed_entity_only_control",
            "query_only_control",
        ]
    }
    expected_holdout_counts = {
        "exact_bridge": {"evidence_answer": 7, "unparsed": 1},
        "raw_answer_for_alt": {"raw_answer_channel_reproduced": 8},
        "catalog_slash_entity_alt": {"evidence_answer": 8},
        "catalog_slash_decoy_alt": {"abstain": 1, "evidence_answer": 7},
        "catalog_slash_reversed_entity_alt": {"evidence_answer": 7, "unparsed": 1},
        "catalog_slash_entity_only_control": {"control_abstain": 8},
        "catalog_slash_decoy_only_control": {"control_abstain": 8},
        "catalog_slash_reversed_entity_only_control": {"control_abstain": 8},
        "query_only_control": {"control_abstain": 8},
    }
    if holdout_counts != expected_holdout_counts:
        fail("KSQ015 full behavior holdout counts changed")
    return {
        "status": "full_behavior_failed_bridge_positive",
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "selected_template": summary["selection"]["selected_template"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "hidden_state_allowed": decision["hidden_state_claim_allowed"],
    }


def main() -> None:
    atlas = load_json(ATLAS_PATH)
    if atlas.get("schema_version") != 1:
        fail("schema_version must be 1")
    if not (ROOT / atlas["doctrine_ref"]).exists():
        fail(f"missing doctrine_ref {atlas['doctrine_ref']}")
    validate_rows(atlas)
    validate_artifact_assertions(atlas)
    artifact_summaries = validate_atlas_artifact_registry(atlas)
    validate_artifact_index(atlas)
    validate_comparison(atlas)
    hypothesis_status_counts = validate_hypotheses(atlas)
    law_audit_summary = validate_law_audit(atlas)
    next_queue_summary = validate_next_queue(atlas)
    smoke_diagnostics_summary = validate_smoke_diagnostics_layer()
    bridge_ladder_summary = validate_bridge_ladder_layer()
    mixture_law_summary = validate_mixture_law_layer()
    decision_frontier_summary = validate_decision_frontier_layer()
    route_disposition_summary = validate_route_disposition_layer()
    transfer_matrix_summary = validate_transfer_matrix_layer()
    reliability_matrix_summary = validate_reliability_matrix_layer()
    error_taxonomy_summary = validate_error_taxonomy_layer()
    gate_geometry_summary = validate_gate_geometry_layer()
    genome_snapshot = validate_genome_snapshot_layer()
    axis_interactions_summary = validate_axis_interactions_layer()
    coverage_gaps_summary = validate_coverage_gaps_layer()
    gap_closure_plan_summary = validate_gap_closure_plan_layer()
    offensive_doctrine_summary = validate_offensive_doctrine_layer()
    transfer_width_probe_summary = validate_transfer_width_probe_layer()
    singleton_stage_pack_summary = validate_singleton_stage_pack_layer()
    post_mc033_closeout_summary = validate_post_mc033_bridge_closeout_layer()
    mc005_reference_summary = validate_mc005_reference_specimen_layer()
    mc006_frontier_summary = validate_mc006_predecision_frontier_layer()
    compositional_genome_summary = validate_compositional_genome_audit_layer()
    family_matrix_summary = validate_family_matrix_layer()
    knowledge_ladder_summary = validate_knowledge_ladder_layer()
    knowledge_gap_plan_summary = validate_knowledge_gap_plan_layer()
    knowledge_admission_summary = validate_knowledge_substrate_admission_layer()
    knowledge_candidate_summary = validate_knowledge_candidate_queue_layer()
    knowledge_first_run_summary = validate_knowledge_first_run_pack_layer()
    knowledge_first_run_outcomes_summary = (
        validate_knowledge_first_run_outcomes_layer()
    )
    knowledge_failure_topology_summary = (
        validate_knowledge_failure_topology_layer()
    )
    knowledge_second_wave_summary = validate_knowledge_second_wave_outcomes_layer()
    knowledge_third_wave_summary = validate_knowledge_third_wave_outcomes_layer()
    knowledge_fourth_wave_summary = validate_knowledge_fourth_wave_outcomes_layer()
    knowledge_fifth_wave_summary = validate_knowledge_fifth_wave_outcomes_layer()
    knowledge_sixth_wave_summary = validate_knowledge_sixth_wave_outcomes_layer()
    knowledge_seventh_wave_summary = validate_knowledge_seventh_wave_outcomes_layer()
    knowledge_eighth_wave_summary = validate_knowledge_eighth_wave_outcomes_layer()
    knowledge_ninth_wave_summary = validate_knowledge_ninth_wave_outcomes_layer()
    knowledge_tenth_wave_summary = validate_knowledge_tenth_wave_outcomes_layer()
    knowledge_eleventh_wave_summary = validate_knowledge_eleventh_wave_outcomes_layer()
    ksq001_first_run_summary = validate_ksq001_first_run_result_layer()
    ksq001_smoke_summary = validate_ksq001_smoke_result_layer()
    ksq001_full_behavior_summary = validate_ksq001_full_behavior_result_layer()
    ksq002_first_run_summary = validate_ksq002_first_run_result_layer()
    ksq002_smoke_summary = validate_ksq002_smoke_result_layer()
    ksq002_full_behavior_summary = validate_ksq002_full_behavior_result_layer()
    ksq003_first_run_summary = validate_ksq003_first_run_result_layer()
    ksq003_smoke_summary = validate_ksq003_smoke_result_layer()
    ksq004_first_run_summary = validate_ksq004_first_run_result_layer()
    ksq004_smoke_summary = validate_ksq004_smoke_result_layer()
    ksq004_full_behavior_summary = validate_ksq004_full_behavior_result_layer()
    ksq005_first_run_summary = validate_ksq005_first_run_result_layer()
    ksq005_smoke_summary = validate_ksq005_smoke_result_layer()
    ksq006_first_run_summary = validate_ksq006_first_run_result_layer()
    ksq006_smoke_summary = validate_ksq006_smoke_result_layer()
    ksq007_first_run_summary = validate_ksq007_first_run_result_layer()
    ksq007_smoke_summary = validate_ksq007_smoke_result_layer()
    ksq007b_first_run_summary = validate_ksq007b_first_run_result_layer()
    ksq007b_smoke_summary = validate_ksq007b_smoke_result_layer()
    ksq008_first_run_summary = validate_ksq008_first_run_result_layer()
    ksq008_smoke_summary = validate_ksq008_smoke_result_layer()
    ksq009_first_run_summary = validate_ksq009_first_run_result_layer()
    ksq009_smoke_summary = validate_ksq009_smoke_result_layer()
    ksq010_first_run_summary = validate_ksq010_first_run_result_layer()
    ksq010_smoke_summary = validate_ksq010_smoke_result_layer()
    ksq011_first_run_summary = validate_ksq011_first_run_result_layer()
    ksq011_smoke_summary = validate_ksq011_smoke_result_layer()
    ksq012_first_run_summary = validate_ksq012_first_run_result_layer()
    ksq012_smoke_summary = validate_ksq012_smoke_result_layer()
    ksq013_first_run_summary = validate_ksq013_first_run_result_layer()
    ksq013_smoke_summary = validate_ksq013_smoke_result_layer()
    ksq014_first_run_summary = validate_ksq014_first_run_result_layer()
    ksq014_smoke_summary = validate_ksq014_smoke_result_layer()
    ksq015_first_run_summary = validate_ksq015_first_run_result_layer()
    ksq015_full_behavior_summary = validate_ksq015_full_behavior_result_layer()
    artifact_report = registry_report(artifact_summaries)

    verdict_counts: dict[str, int] = {}
    lead_time_counts: dict[str, int] = {}
    diagnostic_counts: dict[str, int] = {}
    for row in atlas["rows"]:
        verdict_counts[row["verdict"]["class"]] = (
            verdict_counts.get(row["verdict"]["class"], 0) + 1
        )
        lead_time_counts[row["lead_time"]["state"]] = (
            lead_time_counts.get(row["lead_time"]["state"], 0) + 1
        )
        for diagnostic in row["diagnostics"]:
            diagnostic_counts[diagnostic] = diagnostic_counts.get(diagnostic, 0) + 1

    print(f"atlas ok: {len(atlas['rows'])} rows")
    print("verdict_counts:", json.dumps(verdict_counts, sort_keys=True))
    print("lead_time_counts:", json.dumps(lead_time_counts, sort_keys=True))
    print("top_diagnostics:", json.dumps(diagnostic_counts, sort_keys=True))
    print(
        "hypothesis_status_counts:",
        json.dumps(hypothesis_status_counts, sort_keys=True),
    )
    print(
        "law_audit_level_counts:",
        json.dumps(law_audit_summary["audit_level_counts"], sort_keys=True),
    )
    print(
        "law_rows_without_support:",
        json.dumps(law_audit_summary["rows_without_law_support"]),
    )
    print(
        "next_queue_priority_counts:",
        json.dumps(next_queue_summary["priority_counts"], sort_keys=True),
    )
    print(
        "next_queue_top_ids:",
        json.dumps(next_queue_summary["top_queue_ids"]),
    )
    print(
        "next_queue_bridge_closure:",
        json.dumps(next_queue_summary["bridge_closure"], sort_keys=True),
    )
    print(
        "smoke_diagnostics_counts:",
        json.dumps(
            {
                "cards": smoke_diagnostics_summary["card_count"],
                "behavior_ready": smoke_diagnostics_summary["behavior_ready_count"],
                "signature_ready": smoke_diagnostics_summary["signature_ready_count"],
                "hidden_state_allowed": smoke_diagnostics_summary["hidden_state_allowed_count"],
            },
            sort_keys=True,
        ),
    )
    print(
        "bridge_ladder_counts:",
        json.dumps(
            {
                "rungs": bridge_ladder_summary["rung_count"],
                "behavior_ready": bridge_ladder_summary["behavior_ready_count"],
                "signature_ready": bridge_ladder_summary["signature_ready_count"],
                "hidden_state_allowed": bridge_ladder_summary["hidden_state_allowed_count"],
                "clean_unconfounded": bridge_ladder_summary["clean_unconfounded_bridge_count"],
            },
            sort_keys=True,
        ),
    )
    print(
        "mixture_law_counts:",
        json.dumps(
            {
                "rows": mixture_law_summary["row_count"],
                "promoted": mixture_law_summary["promoted_mechanism_count"],
                "bounded_internal_causal": mixture_law_summary["bounded_internal_causal_count"],
                "primary_blockers": mixture_law_summary["primary_blocker_counts"],
                "pressure_classes": mixture_law_summary["pressure_class_counts"],
            },
            sort_keys=True,
        ),
    )
    print(
        "decision_frontier_counts:",
        json.dumps(
            {
                "rows": decision_frontier_summary["row_count"],
                "classes": decision_frontier_summary["frontier_class_counts"],
                "monitor_only_rows": decision_frontier_summary["monitor_only_rows"],
                "predecision_causal_candidates": decision_frontier_summary[
                    "predecision_causal_candidate_count"
                ],
            },
            sort_keys=True,
        ),
    )
    print(
        "route_disposition_counts:",
        json.dumps(
            {
                "rows": route_disposition_summary["row_count"],
                "atlas_dispositions": route_disposition_summary[
                    "atlas_disposition_counts"
                ],
                "bridge_dispositions": route_disposition_summary[
                    "bridge_disposition_counts"
                ],
                "hidden_state_ready": route_disposition_summary[
                    "hidden_state_ready_route_count"
                ],
            },
            sort_keys=True,
        ),
    )
    print(
        "transfer_matrix_counts:",
        json.dumps(
            {
                "rows": transfer_matrix_summary["row_count"],
                "transfer_classes": transfer_matrix_summary[
                    "transfer_class_counts"
                ],
                "transfer_values": transfer_matrix_summary[
                    "transfer_value_counts"
                ],
                "transfer_ready_mechanisms": transfer_matrix_summary[
                    "transfer_ready_mechanism_count"
                ],
            },
            sort_keys=True,
        ),
    )
    print(
        "reliability_matrix_counts:",
        json.dumps(
            {
                "rows": reliability_matrix_summary["row_count"],
                "classes": reliability_matrix_summary["reliability_class_counts"],
                "missing_gates": reliability_matrix_summary["missing_gate_counts"],
                "full_reliability": reliability_matrix_summary[
                    "full_reliability_count"
                ],
                "bounded_reliability": reliability_matrix_summary[
                    "bounded_reliability_count"
                ],
            },
            sort_keys=True,
        ),
    )
    print(
        "error_taxonomy_counts:",
        json.dumps(
            {
                "smoke_cards": error_taxonomy_summary["smoke_card_count"],
                "bridge_rungs": error_taxonomy_summary["bridge_rung_count"],
                "mc028_other_number_count": error_taxonomy_summary[
                    "mc028_other_number_count"
                ],
                "hidden_state_allowed": error_taxonomy_summary[
                    "hidden_state_allowed_count"
                ],
            },
            sort_keys=True,
        ),
    )
    print(
        "gate_geometry_counts:",
        json.dumps(
            {
                "rows": gate_geometry_summary["row_count"],
                "terminal_stages": gate_geometry_summary["terminal_stage_counts"],
                "ordered_buckets": gate_geometry_summary["ordered_gate_buckets"],
                "bridge_terminal_stages": gate_geometry_summary[
                    "bridge_terminal_stage_counts"
                ],
            },
            sort_keys=True,
        ),
    )
    print(
        "genome_snapshot_counts:",
        json.dumps(
            {
                "rows": genome_snapshot["summary"]["row_count"],
                "artifacts": genome_snapshot["summary"]["artifact_count"],
                "verdicts": genome_snapshot["summary"]["verdict_counts"],
                "gate_buckets": genome_snapshot["gate_shape"][
                    "ordered_gate_buckets"
                ],
                "bridge_hidden_state_allowed": genome_snapshot["bridge_shape"][
                    "hidden_state_allowed_count"
                ],
                "top_queue_ids": genome_snapshot["summary"]["top_queue_ids"],
            },
            sort_keys=True,
        ),
    )
    print(
        "axis_interaction_counts:",
        json.dumps(
            {
                "rows": axis_interactions_summary["row_count"],
                "features": axis_interactions_summary["feature_count"],
                "predictive_rules": axis_interactions_summary[
                    "predictive_rule_count"
                ],
                "broad_or_supported_rules": axis_interactions_summary[
                    "broad_or_supported_rule_count"
                ],
                "mixed_predictors": axis_interactions_summary[
                    "mixed_predictor_count"
                ],
            },
            sort_keys=True,
        ),
    )
    print(
        "coverage_gap_counts:",
        json.dumps(
            {
                "gaps": coverage_gaps_summary["gap_count"],
                "severity": coverage_gaps_summary["severity_counts"],
                "types": coverage_gaps_summary["gap_type_counts"],
                "transfer_untested": coverage_gaps_summary[
                    "transfer_untested_count"
                ],
                "singleton_plus_sparse_features": coverage_gaps_summary[
                    "singleton_plus_sparse_feature_count"
                ],
                "bridge_hidden_state_allowed": coverage_gaps_summary[
                    "bridge_hidden_state_allowed_count"
                ],
            },
            sort_keys=True,
        ),
    )
    print(
        "gap_closure_plan_counts:",
        json.dumps(
            {
                "work_orders": gap_closure_plan_summary["work_order_count"],
                "covered_gaps": gap_closure_plan_summary["covered_gap_count"],
                "coverage_gaps": gap_closure_plan_summary["coverage_gap_count"],
                "critical_gap_coverage": gap_closure_plan_summary[
                    "critical_gap_coverage_count"
                ],
                "top_queue_coverage": gap_closure_plan_summary[
                    "top_queue_coverage_count"
                ],
                "urgency": gap_closure_plan_summary["urgency_counts"],
            },
            sort_keys=True,
        ),
    )
    print(
        "offensive_doctrine_counts:",
        json.dumps(
            {
                "branch_contracts": offensive_doctrine_summary[
                    "branch_contract_count"
                ],
                "source_work_orders": offensive_doctrine_summary[
                    "source_work_order_count"
                ],
                "critical_gaps": offensive_doctrine_summary["critical_gap_count"],
                "urgency": offensive_doctrine_summary["urgency_counts"],
                "track_types": offensive_doctrine_summary["track_type_counts"],
            },
            sort_keys=True,
        ),
    )
    print(
        "transfer_width_probe_counts:",
        json.dumps(transfer_width_probe_summary, sort_keys=True),
    )
    print(
        "singleton_stage_pack_counts:",
        json.dumps(singleton_stage_pack_summary, sort_keys=True),
    )
    print(
        "post_mc033_bridge_closeout_counts:",
        json.dumps(post_mc033_closeout_summary, sort_keys=True),
    )
    print(
        "mc005_reference_specimen_counts:",
        json.dumps(mc005_reference_summary, sort_keys=True),
    )
    print(
        "mc006_predecision_frontier_counts:",
        json.dumps(mc006_frontier_summary, sort_keys=True),
    )
    print(
        "compositional_genome_audit_counts:",
        json.dumps(compositional_genome_summary, sort_keys=True),
    )
    print(
        "family_matrix_counts:",
        json.dumps(family_matrix_summary, sort_keys=True),
    )
    print(
        "knowledge_ladder_counts:",
        json.dumps(knowledge_ladder_summary, sort_keys=True),
    )
    print(
        "knowledge_gap_plan_counts:",
        json.dumps(knowledge_gap_plan_summary, sort_keys=True),
    )
    print(
        "knowledge_substrate_admission_counts:",
        json.dumps(knowledge_admission_summary, sort_keys=True),
    )
    print(
        "knowledge_candidate_queue_counts:",
        json.dumps(knowledge_candidate_summary, sort_keys=True),
    )
    print(
        "knowledge_first_run_pack_counts:",
        json.dumps(knowledge_first_run_summary, sort_keys=True),
    )
    print(
        "knowledge_first_run_outcomes_counts:",
        json.dumps(knowledge_first_run_outcomes_summary, sort_keys=True),
    )
    print(
        "knowledge_failure_topology_counts:",
        json.dumps(knowledge_failure_topology_summary, sort_keys=True),
    )
    print(
        "knowledge_second_wave_outcomes_counts:",
        json.dumps(knowledge_second_wave_summary, sort_keys=True),
    )
    print(
        "knowledge_third_wave_outcomes_counts:",
        json.dumps(knowledge_third_wave_summary, sort_keys=True),
    )
    print(
        "knowledge_fourth_wave_outcomes_counts:",
        json.dumps(knowledge_fourth_wave_summary, sort_keys=True),
    )
    print(
        "knowledge_fifth_wave_outcomes_counts:",
        json.dumps(knowledge_fifth_wave_summary, sort_keys=True),
    )
    print(
        "knowledge_sixth_wave_outcomes_counts:",
        json.dumps(knowledge_sixth_wave_summary, sort_keys=True),
    )
    print(
        "knowledge_seventh_wave_outcomes_counts:",
        json.dumps(knowledge_seventh_wave_summary, sort_keys=True),
    )
    print(
        "knowledge_eighth_wave_outcomes_counts:",
        json.dumps(knowledge_eighth_wave_summary, sort_keys=True),
    )
    print(
        "knowledge_ninth_wave_outcomes_counts:",
        json.dumps(knowledge_ninth_wave_summary, sort_keys=True),
    )
    print(
        "knowledge_tenth_wave_outcomes_counts:",
        json.dumps(knowledge_tenth_wave_summary, sort_keys=True),
    )
    print(
        "knowledge_eleventh_wave_outcomes_counts:",
        json.dumps(knowledge_eleventh_wave_summary, sort_keys=True),
    )
    print(
        "ksq001_first_run_counts:",
        json.dumps(ksq001_first_run_summary, sort_keys=True),
    )
    print(
        "ksq001_smoke_counts:",
        json.dumps(ksq001_smoke_summary, sort_keys=True),
    )
    print(
        "ksq001_full_behavior_counts:",
        json.dumps(ksq001_full_behavior_summary, sort_keys=True),
    )
    print(
        "ksq002_first_run_counts:",
        json.dumps(ksq002_first_run_summary, sort_keys=True),
    )
    print(
        "ksq002_smoke_counts:",
        json.dumps(ksq002_smoke_summary, sort_keys=True),
    )
    print(
        "ksq002_full_behavior_counts:",
        json.dumps(ksq002_full_behavior_summary, sort_keys=True),
    )
    print(
        "ksq003_first_run_counts:",
        json.dumps(ksq003_first_run_summary, sort_keys=True),
    )
    print(
        "ksq003_smoke_counts:",
        json.dumps(ksq003_smoke_summary, sort_keys=True),
    )
    print(
        "ksq004_first_run_counts:",
        json.dumps(ksq004_first_run_summary, sort_keys=True),
    )
    print(
        "ksq004_smoke_counts:",
        json.dumps(ksq004_smoke_summary, sort_keys=True),
    )
    print(
        "ksq004_full_behavior_counts:",
        json.dumps(ksq004_full_behavior_summary, sort_keys=True),
    )
    print(
        "ksq005_first_run_counts:",
        json.dumps(ksq005_first_run_summary, sort_keys=True),
    )
    print(
        "ksq005_smoke_counts:",
        json.dumps(ksq005_smoke_summary, sort_keys=True),
    )
    print(
        "ksq006_first_run_counts:",
        json.dumps(ksq006_first_run_summary, sort_keys=True),
    )
    print(
        "ksq006_smoke_counts:",
        json.dumps(ksq006_smoke_summary, sort_keys=True),
    )
    print(
        "ksq007_first_run_counts:",
        json.dumps(ksq007_first_run_summary, sort_keys=True),
    )
    print(
        "ksq007_smoke_counts:",
        json.dumps(ksq007_smoke_summary, sort_keys=True),
    )
    print(
        "ksq007b_first_run_counts:",
        json.dumps(ksq007b_first_run_summary, sort_keys=True),
    )
    print(
        "ksq007b_smoke_counts:",
        json.dumps(ksq007b_smoke_summary, sort_keys=True),
    )
    print(
        "ksq008_first_run_counts:",
        json.dumps(ksq008_first_run_summary, sort_keys=True),
    )
    print(
        "ksq008_smoke_counts:",
        json.dumps(ksq008_smoke_summary, sort_keys=True),
    )
    print(
        "ksq009_first_run_counts:",
        json.dumps(ksq009_first_run_summary, sort_keys=True),
    )
    print(
        "ksq009_smoke_counts:",
        json.dumps(ksq009_smoke_summary, sort_keys=True),
    )
    print(
        "ksq010_first_run_counts:",
        json.dumps(ksq010_first_run_summary, sort_keys=True),
    )
    print(
        "ksq010_smoke_counts:",
        json.dumps(ksq010_smoke_summary, sort_keys=True),
    )
    print(
        "ksq011_first_run_counts:",
        json.dumps(ksq011_first_run_summary, sort_keys=True),
    )
    print(
        "ksq011_smoke_counts:",
        json.dumps(ksq011_smoke_summary, sort_keys=True),
    )
    print(
        "ksq012_first_run_counts:",
        json.dumps(ksq012_first_run_summary, sort_keys=True),
    )
    print(
        "ksq012_smoke_counts:",
        json.dumps(ksq012_smoke_summary, sort_keys=True),
    )
    print(
        "ksq013_first_run_counts:",
        json.dumps(ksq013_first_run_summary, sort_keys=True),
    )
    print(
        "ksq013_smoke_counts:",
        json.dumps(ksq013_smoke_summary, sort_keys=True),
    )
    print(
        "ksq014_first_run_counts:",
        json.dumps(ksq014_first_run_summary, sort_keys=True),
    )
    print(
        "ksq014_smoke_counts:",
        json.dumps(ksq014_smoke_summary, sort_keys=True),
    )
    print(
        "ksq015_first_run_counts:",
        json.dumps(ksq015_first_run_summary, sort_keys=True),
    )
    print(
        "ksq015_full_behavior_counts:",
        json.dumps(ksq015_full_behavior_summary, sort_keys=True),
    )
    print(
        "artifact_parser_counts:",
        json.dumps(artifact_report["parser_counts"], sort_keys=True),
    )
    print("linked_artifact_count:", artifact_report["artifact_count"])
    print("artifacts_with_metrics:", artifact_report["artifacts_with_metrics"])
    print(
        "family_claim_checks:",
        json.dumps(artifact_report["family_claim_checks"]),
    )


if __name__ == "__main__":
    main()
