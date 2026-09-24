"""Build first-run packets for the knowledge-substrate candidate queue.

The candidate queue names behavior-only substrate candidates. This layer turns
each candidate into a first-run preregistration packet with panels, thresholds,
artifact names, and fail-fast diagnostics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json
from control_surface_knowledge_candidate_queue import KNOWLEDGE_CANDIDATE_QUEUE_PATH
from control_surface_knowledge_substrate_admission import ADMISSION_GATE_ORDER


KNOWLEDGE_FIRST_RUN_PACK_PATH = (
    ROOT / "data" / "control_surface_knowledge_first_run_pack.json"
)
KNOWLEDGE_FIRST_RUN_PACK_REPORT_PATH = (
    ROOT / "research" / "46_CONTROL_SURFACE_KNOWLEDGE_FIRST_RUN_PACK.md"
)

PRIMARY_MODEL_ID = "Qwen/Qwen3-1.7B"
SECONDARY_MODEL_ID = "google/gemma-2-2b-it"

REQUIRED_PACKET_FIELDS = {
    "first_run_id",
    "candidate_id",
    "candidate_title",
    "level_id",
    "admission_class",
    "priority_class",
    "run_scope",
    "target_models",
    "artifact_paths",
    "freeze_before_run",
    "panel_specs",
    "baseline_specs",
    "admission_gate_bindings",
    "promotion_rule",
    "death_rule",
    "containment_rule",
    "export_rule",
    "hidden_state_license",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def by_key(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        item_key = item[key]
        if item_key in result:
            raise AssertionError(f"duplicate {key}: {item_key}")
        result[item_key] = item
    return result


def panel(
    panel_id: str,
    panel_type: str,
    purpose: str,
    minimum_rows: int,
    required_metrics: list[str],
    pass_rule: str,
    fail_exports: list[str],
) -> dict[str, Any]:
    return {
        "panel_id": panel_id,
        "panel_type": panel_type,
        "purpose": purpose,
        "minimum_rows": minimum_rows,
        "required_metrics": required_metrics,
        "pass_rule": pass_rule,
        "fail_exports": fail_exports,
    }


def common_baselines(candidate_id: str) -> list[dict[str, Any]]:
    return [
        {
            "baseline_id": f"{candidate_id}_prompt_format_baseline",
            "purpose": "Check whether prompt format, instruction tone, or answer schema predicts the target label.",
            "required_before_signature": True,
        },
        {
            "baseline_id": f"{candidate_id}_output_margin_baseline",
            "purpose": "Report next-token or generated-answer output margin separation before any hidden-state claim.",
            "required_before_signature": True,
        },
        {
            "baseline_id": f"{candidate_id}_candidate_score_baseline",
            "purpose": "Report candidate-score margin where a candidate set exists, or document why no candidate-score interface is used.",
            "required_before_signature": True,
        },
        {
            "baseline_id": f"{candidate_id}_shuffle_null",
            "purpose": "Run shuffled-label selection or equivalent null if a signature is later proposed.",
            "required_before_signature": True,
        },
    ]


def packet_artifact_paths(candidate_id: str) -> dict[str, str]:
    slug = candidate_id.upper()
    return {
        "prereg": f"research/prereg/{slug}_FIRST_RUN.md",
        "status_card": f"research/cards/{slug}_FIRST_RUN_STATUS.md",
        "result_json": f"results/cards/{slug}/{candidate_id}_first_run.json",
        "runner": f"code/{candidate_id}_first_run.py",
    }


def panel_library() -> dict[str, list[dict[str, Any]]]:
    return {
        "ksq001_familiar_entity_prior_counterbalance": [
            panel(
                "source_local_artificial_lookup",
                "direct_control",
                "Confirm artificial values are learnable from the local source without semantic-prior conflict.",
                40,
                ["parseable_rate", "artificial_value_rate", "source_disjoint_rate"],
                "Pass if parseable_rate >= 0.90 and artificial_value_rate >= 0.85 on non-holdout and >= 0.75 on source-disjoint holdout.",
                ["FAMILIAR_ENTITY_LOCAL_LOOKUP_FAILED"],
            ),
            panel(
                "semantic_prior_lure",
                "semantic_prior_control",
                "Measure whether real-world priors compete with artificial values under matched wording.",
                40,
                ["semantic_prior_rate", "artificial_value_rate", "label_balance"],
                "Pass as an admission signal only if both branches appear in matched conflict rows and neither branch exceeds 0.90 globally.",
                ["SEMANTIC_PRIOR_ABSENT", "LOCAL_COPY_DOMINATES"],
            ),
            panel(
                "answer_absent_irrelevant_nulls",
                "null",
                "Ensure the model does not emit artificial or prior values when the source is absent or irrelevant.",
                30,
                ["unknown_or_abstain_rate", "side_answer_rate", "parseable_rate"],
                "Pass if unknown_or_abstain_rate >= 0.80 and side_answer_rate <= 0.10.",
                ["ANSWER_ABSENT_NULL_FAILED", "SIDE_ANSWER_LEAKAGE"],
            ),
            panel(
                "source_disjoint_holdout",
                "holdout",
                "Check that behavior survives held-out familiar entities and artificial values.",
                16,
                ["holdout_parseable_rate", "holdout_expected_rate", "holdout_label_balance"],
                "Pass if holdout_parseable_rate >= 0.90, holdout_expected_rate >= 0.70, and both labels have at least 6 rows.",
                ["SOURCE_DISJOINT_HOLDOUT_FAILED"],
            ),
            panel(
                "candidate_output_margin_audit",
                "output_geometry",
                "Measure whether output or candidate margins explain the branch before hidden states.",
                40,
                ["candidate_margin_auc", "next_token_margin_auc", "answer_shape_auc"],
                "Pass only if no simple output/candidate/shape baseline reaches 0.80 AUC on holdout labels.",
                ["OUTPUT_MARGIN_CONFOUND", "ANSWER_SHAPE_CONFOUND"],
            ),
        ],
        "ksq002_familiar_entity_source_rewrite_equivalence": [
            panel(
                "baseline_source_value_lookup",
                "direct_control",
                "Establish the source-value lookup baseline before rewrite tests.",
                40,
                ["parseable_rate", "source_value_rate"],
                "Pass if parseable_rate >= 0.90 and source_value_rate >= 0.85.",
                ["SOURCE_VALUE_BASELINE_FAILED"],
            ),
            panel(
                "neutral_rewrite_lookup",
                "prompt_channel_locality",
                "Test whether neutral rewrites preserve source-value behavior.",
                40,
                ["rewrite_equivalence_rate", "parseable_rate", "delta_from_baseline"],
                "Pass if rewrite_equivalence_rate >= 0.80 and delta_from_baseline <= 0.15.",
                ["SOURCE_REWRITE_EQUIVALENCE_FAILED"],
            ),
            panel(
                "source_deletion",
                "null",
                "Delete the source value and confirm the behavior disappears.",
                30,
                ["source_value_rate", "unknown_or_other_rate"],
                "Pass if source_value_rate <= 0.10 after deletion.",
                ["SOURCE_DELETION_NOT_CIRCUIT"],
            ),
            panel(
                "query_only_control",
                "null",
                "Check whether entity/query text alone carries the answer.",
                30,
                ["source_value_rate", "query_proxy_rate"],
                "Pass if query_proxy_rate <= 0.10.",
                ["QUERY_ONLY_SOURCE_PROXY"],
            ),
            panel(
                "source_disjoint_rewrite_holdout",
                "holdout",
                "Hold out entities and rewrite forms together.",
                16,
                ["holdout_equivalence_rate", "holdout_parseable_rate"],
                "Pass if holdout_equivalence_rate >= 0.75 and holdout_parseable_rate >= 0.90.",
                ["REWRITE_HOLDOUT_FAILED"],
            ),
            panel(
                "rewrite_output_geometry_audit",
                "output_geometry",
                "Measure whether output or candidate geometry explains rewrite success.",
                40,
                ["candidate_margin_auc", "next_token_margin_auc", "rewrite_format_auc"],
                "Pass only if no candidate, output, or rewrite-format baseline reaches 0.80 AUC on held-out labels.",
                ["OUTPUT_MARGIN_CONFOUND", "PROMPT_FORMAT_CONFUND"],
            ),
        ],
        "ksq003_bridge_statusless_evidence_aggregation": [
            panel(
                "source_local_direct_control",
                "direct_control",
                "Verify local source answers are recoverable when no learned branch is requested.",
                40,
                ["local_answer_rate", "parseable_rate"],
                "Pass if parseable_rate >= 0.90 and local_answer_rate >= 0.85.",
                ["LOCAL_DIRECT_CONTROL_FAILED"],
            ),
            panel(
                "learned_fact_direct_control",
                "direct_control",
                "Verify learned facts remain available under the answer interface before conflict rows.",
                40,
                ["learned_answer_rate", "parseable_rate"],
                "Pass if parseable_rate >= 0.90 and learned_answer_rate >= 0.75.",
                ["LEARNED_FACT_DIRECT_CONTROL_FAILED"],
            ),
            panel(
                "multi_evidence_conflict",
                "conflict_mixture",
                "Test statusless evidence aggregation under local-versus-learned conflict.",
                80,
                ["local_branch_rate", "learned_branch_rate", "expected_branch_rate", "label_balance"],
                "Pass only if expected_branch_rate >= 0.70 and both local and learned branch labels have at least 24 rows.",
                ["LEARNED_BRANCH_TABLE_PRESSURE_COLLAPSE", "LOCAL_BRANCH_DOMINANCE"],
            ),
            panel(
                "evidence_ablation",
                "prompt_channel_locality",
                "Ablate individual evidence features to show no single visible cue carries the branch.",
                60,
                ["single_feature_auc", "ablation_delta", "expected_branch_rate"],
                "Pass only if no single visible feature reaches 0.80 AUC and branch behavior degrades predictably under multi-feature ablation.",
                ["STATUSLESS_EVIDENCE_VISIBLE_CHANNEL"],
            ),
            panel(
                "answer_absent_and_side_nulls",
                "null",
                "Check answer-absent, side-number, and irrelevant-source leakage.",
                40,
                ["unknown_rate", "side_answer_rate", "other_number_rate"],
                "Pass if unknown_rate >= 0.80 and side_answer_rate <= 0.10.",
                ["ANSWER_ABSENT_NULL_FAILED", "SIDE_NUMBER_LEAKAGE"],
            ),
            panel(
                "source_disjoint_bridge_holdout",
                "holdout",
                "Hold out entities, facts, and evidence combinations together.",
                24,
                ["holdout_expected_branch_rate", "holdout_label_balance"],
                "Pass if holdout_expected_branch_rate >= 0.65 and both branch labels have at least 8 holdout rows.",
                ["BRIDGE_SOURCE_DISJOINT_HOLDOUT_FAILED"],
            ),
            panel(
                "candidate_output_baselines",
                "output_geometry",
                "Report candidate and output geometry before any hidden-state work.",
                80,
                ["candidate_margin_auc", "next_token_margin_auc", "prompt_feature_auc"],
                "Pass only if simple output/candidate/prompt-feature baselines do not explain the branch on holdout.",
                ["CANDIDATE_SCORE_CONFOUND", "OUTPUT_MARGIN_CONFOUND"],
            ),
        ],
        "ksq004_bridge_answer_interface_minimal_pairs": [
            panel(
                "matched_minimal_pairs",
                "behavior_contract",
                "Create local and learned answers matched by type, length, frequency band, and parser shape.",
                60,
                ["answer_length_balance", "answer_type_balance", "parseable_rate"],
                "Pass if parser and answer-shape balance pass before behavior is inspected.",
                ["ANSWER_INTERFACE_BALANCE_FAILED"],
            ),
            panel(
                "local_learned_direct_controls",
                "direct_control",
                "Verify both answer branches work in isolation under the matched interface.",
                60,
                ["local_direct_rate", "learned_direct_rate", "parseable_rate"],
                "Pass if both direct rates >= 0.75 and parseable_rate >= 0.90.",
                ["DIRECT_CONTROL_FAILED"],
            ),
            panel(
                "minimal_pair_conflict",
                "conflict_mixture",
                "Test whether matched outputs preserve local-versus-learned conflict behavior.",
                80,
                ["expected_branch_rate", "local_branch_rate", "learned_branch_rate"],
                "Pass if expected_branch_rate >= 0.70 and both branch labels have at least 24 rows.",
                ["BRIDGE_MINIMAL_PAIR_CONTRAST_ABSENT"],
            ),
            panel(
                "side_answer_leakage",
                "side_effect",
                "Measure other-answer, side-number, and answer-token shortcut leakage.",
                50,
                ["side_answer_rate", "other_answer_rate", "answer_token_auc"],
                "Pass if side_answer_rate <= 0.10 and answer_token_auc < 0.80.",
                ["ANSWER_INTERFACE_BRANCH_SHORTCUT", "SIDE_ANSWER_LEAKAGE"],
            ),
            panel(
                "null_and_holdout",
                "null_holdout",
                "Combine answer-absent nulls with source-disjoint minimal-pair holdout.",
                40,
                ["null_unknown_rate", "holdout_expected_branch_rate"],
                "Pass if null_unknown_rate >= 0.80 and holdout_expected_branch_rate >= 0.65.",
                ["NULL_RELIABILITY_BOTTLENECK", "HOLDOUT_FAILED"],
            ),
            panel(
                "minimal_pair_output_geometry_audit",
                "output_geometry",
                "Measure whether matched-answer candidate or token geometry still predicts the branch.",
                80,
                ["candidate_margin_auc", "next_token_margin_auc", "answer_shape_auc"],
                "Pass only if candidate, next-token, and answer-shape baselines stay below 0.80 AUC on holdout labels.",
                ["CANDIDATE_SCORE_CONFOUND", "ANSWER_INTERFACE_BRANCH_SHORTCUT"],
            ),
        ],
        "ksq005_uncertainty_grounded_answerability": [
            panel(
                "known_factual_direct",
                "direct_control",
                "Verify answerable known facts are answered without visible known labels.",
                50,
                ["answer_rate", "correct_rate", "parseable_rate"],
                "Pass if correct_rate >= 0.80 and parseable_rate >= 0.90.",
                ["KNOWN_FACT_DIRECT_FAILED"],
            ),
            panel(
                "unknown_nonce_rows",
                "null",
                "Check whether unknown or nonce items produce abstention without pressure wording.",
                50,
                ["abstain_rate", "hallucination_rate", "label_shape_auc"],
                "Pass if abstain_rate >= 0.70 and hallucination_rate <= 0.20.",
                ["UNKNOWN_ABSTENTION_FAILED", "LABEL_GROUNDING_FAILURE"],
            ),
            panel(
                "unsupported_context_rows",
                "context_null",
                "Test contexts that do not support the requested answer.",
                50,
                ["abstain_rate", "unsupported_answer_rate", "requested_mode_auc"],
                "Pass if abstain_rate >= 0.65 and unsupported_answer_rate <= 0.20.",
                ["UNSUPPORTED_CONTEXT_LEAKAGE", "REQUESTED_MODE_CONFOUND"],
            ),
            panel(
                "contradicted_context_rows",
                "conflict_mixture",
                "Measure whether contradicted contexts cause correction, abstention, or false acceptance.",
                50,
                ["corrected_rate", "abstain_rate", "false_accept_rate"],
                "Pass if corrected_rate + abstain_rate >= 0.70 and false_accept_rate <= 0.20.",
                ["CONTRADICTION_FALSE_ACCEPTANCE"],
            ),
            panel(
                "source_disjoint_answerability_holdout",
                "holdout",
                "Hold out entities, facts, and support contexts together.",
                24,
                ["holdout_answerable_rate", "holdout_unanswerable_abstain_rate"],
                "Pass if answerable and unanswerable holdout panels each pass their branch floor.",
                ["ANSWERABILITY_HOLDOUT_FAILED"],
            ),
            panel(
                "requested_mode_output_baselines",
                "output_geometry",
                "Check requested-mode text, answer shape, and output margins before any signature work.",
                80,
                ["requested_mode_auc", "output_margin_auc", "answer_shape_auc"],
                "Pass only if no requested-mode, margin, or shape baseline reaches 0.80 AUC.",
                ["REQUESTED_MODE_CONFOUND", "OUTPUT_MARGIN_CONFOUND"],
            ),
        ],
        "ksq006_uncertainty_context_support_counterfactuals": [
            panel(
                "supported_context_rows",
                "direct_control",
                "Verify supported contexts produce the supported answer.",
                50,
                ["supported_answer_rate", "parseable_rate"],
                "Pass if supported_answer_rate >= 0.80 and parseable_rate >= 0.90.",
                ["SUPPORTED_CONTEXT_FAILED"],
            ),
            panel(
                "irrelevant_context_rows",
                "null",
                "Verify irrelevant contexts do not induce unsupported answers.",
                50,
                ["abstain_rate", "unsupported_answer_rate"],
                "Pass if abstain_rate >= 0.65 and unsupported_answer_rate <= 0.20.",
                ["IRRELEVANT_CONTEXT_LEAKAGE"],
            ),
            panel(
                "contradicting_context_rows",
                "conflict_mixture",
                "Test contradiction-sensitive answer or abstain behavior.",
                50,
                ["contradiction_detected_rate", "false_accept_rate"],
                "Pass if contradiction_detected_rate >= 0.65 and false_accept_rate <= 0.20.",
                ["CONTRADICTION_NOT_TRACKED"],
            ),
            panel(
                "insufficient_context_rows",
                "null",
                "Test underdetermined contexts separately from irrelevant contexts.",
                50,
                ["abstain_rate", "unsupported_answer_rate", "support_word_auc"],
                "Pass if abstain_rate >= 0.65 and support_word_auc < 0.80.",
                ["INSUFFICIENT_CONTEXT_CONFOUND"],
            ),
            panel(
                "claim_only_and_context_only_controls",
                "direct_control",
                "Separate claim priors from context wording.",
                50,
                ["claim_only_answer_rate", "context_only_answer_rate", "delta_from_supported"],
                "Pass if claim-only and context-only controls do not reproduce supported behavior.",
                ["CLAIM_ONLY_PRIOR_CONFUND", "CONTEXT_ONLY_PROMPT_CONFOUND"],
            ),
            panel(
                "support_counterfactual_holdout",
                "holdout",
                "Hold out claims and context sources together.",
                24,
                ["holdout_supported_rate", "holdout_unsupported_abstain_rate"],
                "Pass if holdout supported and unsupported panels each pass their branch floor.",
                ["CONTEXT_SUPPORT_HOLDOUT_FAILED"],
            ),
            panel(
                "requested_mode_output_baselines",
                "output_geometry",
                "Report requested-mode, support-word, output-margin, and answer-shape baselines.",
                80,
                ["requested_mode_auc", "support_word_auc", "output_margin_auc", "answer_shape_auc"],
                "Pass only if no simple baseline reaches 0.80 AUC on holdout labels.",
                ["CONTEXT_SUPPORT_PROMPT_CHANNEL", "OUTPUT_MARGIN_CONFOUND"],
            ),
        ],
    }


def build_first_run_packets(candidate_queue: dict[str, Any]) -> list[dict[str, Any]]:
    panels_by_candidate = panel_library()
    packets = []
    for candidate_item in candidate_queue["candidates"]:
        candidate_id = candidate_item["candidate_id"]
        if candidate_id not in panels_by_candidate:
            raise AssertionError(f"missing panels for {candidate_id}")
        first_run_id = f"{candidate_id}_first_run_v1"
        packet = {
            "first_run_id": first_run_id,
            "candidate_id": candidate_id,
            "candidate_title": candidate_item["title"],
            "level_id": candidate_item["level_id"],
            "admission_class": candidate_item["admission_class"],
            "priority_class": candidate_item["priority_class"],
            "priority_rank": candidate_item["priority_rank"],
            "run_scope": "behavior_substrate_admission_only",
            "target_models": {
                "primary": PRIMARY_MODEL_ID,
                "secondary_width_check": SECONDARY_MODEL_ID,
                "secondary_usage_rule": "Do not run the secondary target until the primary behavior packet passes or produces a diagnostic worth widening.",
            },
            "artifact_paths": packet_artifact_paths(candidate_id),
            "freeze_before_run": [
                "prompt templates",
                "parser/scorer",
                "row generation seed",
                "source-disjoint split manifest",
                "panel thresholds",
                "baseline interfaces",
                "allowed and forbidden claims",
            ],
            "panel_specs": panels_by_candidate[candidate_id],
            "baseline_specs": common_baselines(candidate_id),
            "admission_gate_bindings": candidate_item["gate_bindings"],
            "dumb_explanations": candidate_item["dumb_explanations"],
            "linked_next_queue_ids": candidate_item["linked_next_queue_ids"],
            "promotion_rule": candidate_item["promotion_rule"],
            "death_rule": candidate_item["death_rule"],
            "containment_rule": candidate_item["containment_rule"],
            "export_rule": candidate_item["export_rule"],
            "hidden_state_license": "forbidden_until_first_run_passes_admission",
            "allowed_output": "behavior_substrate_status_card_or_diagnostic_note",
            "forbidden_output": "hidden_state_signature_or_mechanism_claim",
        }
        missing = REQUIRED_PACKET_FIELDS - set(packet)
        if missing:
            raise AssertionError(f"{candidate_id}: missing fields {missing}")
        packets.append(packet)
    return packets


def build_summary(
    packets: list[dict[str, Any]],
    candidate_queue: dict[str, Any],
) -> dict[str, Any]:
    return {
        "packet_count": len(packets),
        "candidate_count": candidate_queue["summary"]["candidate_count"],
        "admission_gate_count": candidate_queue["summary"]["admission_gate_count"],
        "total_panel_count": sum(len(packet["panel_specs"]) for packet in packets),
        "total_baseline_count": sum(len(packet["baseline_specs"]) for packet in packets),
        "total_gate_binding_count": sum(
            len(packet["admission_gate_bindings"]) for packet in packets
        ),
        "hidden_state_packet_count": sum(
            1
            for packet in packets
            if packet["hidden_state_license"]
            != "forbidden_until_first_run_passes_admission"
        ),
        "packet_count_by_level": dict(
            sorted(Counter(packet["level_id"] for packet in packets).items())
        ),
        "packet_count_by_admission_class": dict(
            sorted(Counter(packet["admission_class"] for packet in packets).items())
        ),
        "priority_counts": dict(
            sorted(Counter(packet["priority_class"] for packet in packets).items())
        ),
        "primary_model": PRIMARY_MODEL_ID,
        "secondary_model": SECONDARY_MODEL_ID,
        "promoted_mechanism_count": candidate_queue["summary"][
            "promoted_mechanism_count"
        ],
    }


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    packets = payload["first_run_packets"]
    candidate_ids = sorted(
        candidate["candidate_id"]
        for candidate in payload["source_snapshot"]["candidate_queue"]
    )
    packet_candidate_ids = sorted(packet["candidate_id"] for packet in packets)
    missing_fields = {
        packet["first_run_id"]: sorted(REQUIRED_PACKET_FIELDS - set(packet))
        for packet in packets
    }
    missing_fields = {key: value for key, value in missing_fields.items() if value}
    gate_binding_failures = [
        packet["first_run_id"]
        for packet in packets
        if sorted(binding["gate_id"] for binding in packet["admission_gate_bindings"])
        != sorted(ADMISSION_GATE_ORDER)
    ]
    panel_failures = [
        packet["first_run_id"]
        for packet in packets
        if len(packet["panel_specs"]) < 5
        or not any("null" in panel["panel_type"] for panel in packet["panel_specs"])
        or not any("holdout" in panel["panel_type"] for panel in packet["panel_specs"])
        or not any(
            panel["panel_type"] == "output_geometry" for panel in packet["panel_specs"]
        )
    ]
    hidden_state_failures = [
        packet["first_run_id"]
        for packet in packets
        if packet["hidden_state_license"]
        != "forbidden_until_first_run_passes_admission"
        or packet["forbidden_output"] != "hidden_state_signature_or_mechanism_claim"
    ]
    artifact_failures = [
        packet["first_run_id"]
        for packet in packets
        if sorted(packet["artifact_paths"]) != ["prereg", "result_json", "runner", "status_card"]
    ]
    return [
        {
            "id": "covers_all_candidate_queue_items",
            "predicate": "packet candidate ids == candidate queue ids",
            "actual": {
                "packet_candidate_ids": packet_candidate_ids,
                "candidate_ids": candidate_ids,
            },
            "passed": packet_candidate_ids == candidate_ids,
            "why": "Every behavior-only candidate needs a run-ready packet.",
        },
        {
            "id": "required_fields_present",
            "predicate": "empty dict",
            "actual": missing_fields,
            "passed": not missing_fields,
            "why": "First-run packets need artifact paths, panels, baselines, gates, and decision rules.",
        },
        {
            "id": "all_packets_bind_all_admission_gates",
            "predicate": "empty list",
            "actual": gate_binding_failures,
            "passed": not gate_binding_failures
            and payload["summary"]["total_gate_binding_count"]
            == payload["summary"]["packet_count"] * len(ADMISSION_GATE_ORDER),
            "why": "First runs must inherit the admission protocol.",
        },
        {
            "id": "all_packets_have_null_holdout_and_output_geometry",
            "predicate": "empty list",
            "actual": panel_failures,
            "passed": not panel_failures,
            "why": "Every first run must test nulls, holdouts, and output geometry before signatures.",
        },
        {
            "id": "hidden_state_work_forbidden",
            "predicate": "empty list and hidden_state_packet_count == 0",
            "actual": {
                "packet_failures": hidden_state_failures,
                "hidden_state_packet_count": payload["summary"][
                    "hidden_state_packet_count"
                ],
            },
            "passed": not hidden_state_failures
            and payload["summary"]["hidden_state_packet_count"] == 0,
            "why": "These first runs are behavior-substrate admissions only.",
        },
        {
            "id": "artifact_paths_are_declared",
            "predicate": "empty list",
            "actual": artifact_failures,
            "passed": not artifact_failures,
            "why": "Each first run should already name its prereg, runner, result, and status card paths.",
        },
        {
            "id": "two_packets_per_admission_class",
            "predicate": "all class counts == 2",
            "actual": payload["summary"]["packet_count_by_admission_class"],
            "passed": all(
                count == 2
                for count in payload["summary"][
                    "packet_count_by_admission_class"
                ].values()
            ),
            "why": "The pack should preserve the primary and backup candidate per admission class.",
        },
        {
            "id": "does_not_claim_results",
            "predicate": "promoted_mechanism_count == 0 and forbidden claim names no result",
            "actual": {
                "promoted_mechanism_count": payload["summary"][
                    "promoted_mechanism_count"
                ],
                "forbidden_claim": payload["forbidden_claim"],
            },
            "passed": payload["summary"]["promoted_mechanism_count"] == 0
            and "does not report any run result" in payload["forbidden_claim"],
            "why": "A prereg pack is not evidence.",
        },
    ]


def build_control_surface_knowledge_first_run_pack() -> dict[str, Any]:
    candidate_queue = load_json(KNOWLEDGE_CANDIDATE_QUEUE_PATH)
    packets = build_first_run_packets(candidate_queue)
    summary = build_summary(packets, candidate_queue)
    payload = {
        "schema_version": 1,
        "updated_at": candidate_queue.get("updated_at"),
        "purpose": (
            "Turn every knowledge-substrate candidate into a behavior-only "
            "first-run packet with panels, thresholds, baselines, artifact paths, "
            "and decision rules."
        ),
        "sources": {
            "knowledge_candidate_queue": rel(KNOWLEDGE_CANDIDATE_QUEUE_PATH),
        },
        "source_snapshot": {
            "candidate_queue": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "level_id": candidate["level_id"],
                    "admission_class": candidate["admission_class"],
                    "priority_class": candidate["priority_class"],
                    "hidden_state_license": candidate["hidden_state_license"],
                }
                for candidate in candidate_queue["candidates"]
            ],
            "admission_gate_order": ADMISSION_GATE_ORDER,
        },
        "summary": summary,
        "first_run_packets": packets,
        "allowed_claim": (
            "The project now has run-ready behavior-substrate admission packets "
            "for all six knowledge-substrate candidates. Each packet names panels, "
            "thresholds, baselines, artifact paths, and fail-fast diagnostics."
        ),
        "forbidden_claim": (
            "This pack does not report any run result, does not promote a "
            "mechanism, and does not license hidden-state signatures, steering, "
            "editing, or surgery."
        ),
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_knowledge_first_run_pack(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("knowledge first-run pack schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"knowledge first-run pack source missing: {rel_path}")
    if payload["summary"]["packet_count"] != 6:
        raise AssertionError("knowledge first-run pack expected six packets")
    if payload["summary"]["total_gate_binding_count"] != 6 * len(ADMISSION_GATE_ORDER):
        raise AssertionError("knowledge first-run pack gate binding count mismatch")
    if payload["summary"]["hidden_state_packet_count"] != 0:
        raise AssertionError("knowledge first-run pack must not license hidden states")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"knowledge first-run pack checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Knowledge First-Run Pack",
        "",
        f"Source updated_at: {payload['updated_at']}",
        "",
        "Status: generated behavior-substrate first-run pack implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_knowledge_first_run_pack.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_first_run_pack.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_first_run_pack.py --write",
        "python code\\control_surface_knowledge_first_run_pack.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This pack converts the knowledge-candidate queue into run-ready",
        "behavior-substrate preregistrations. It deliberately stops before",
        "hidden-state signatures. The output of a first run can only be a",
        "behavior-substrate status card or a diagnostic note.",
        "",
        "## Generated Facts",
        "",
        f"- first-run packets: {summary['packet_count']};",
        f"- total panels: {summary['total_panel_count']};",
        f"- total baselines: {summary['total_baseline_count']};",
        f"- total admission-gate bindings: {summary['total_gate_binding_count']};",
        f"- hidden-state packets: {summary['hidden_state_packet_count']};",
        f"- primary model: `{summary['primary_model']}`;",
        f"- secondary width-check model: `{summary['secondary_model']}`;",
        f"- promoted mechanisms: {summary['promoted_mechanism_count']}.",
        "",
        "## Packet Table",
        "",
        "| Packet | Candidate | Level | Priority | Panels | Hidden-State License |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for packet in payload["first_run_packets"]:
        lines.append(
            f"| `{packet['first_run_id']}` | `{packet['candidate_id']}` | "
            f"`{packet['level_id']}` | `{packet['priority_class']}` | "
            f"{len(packet['panel_specs'])} | `{packet['hidden_state_license']}` |"
        )

    for packet in payload["first_run_packets"]:
        lines.extend(
            [
                "",
                f"## {packet['candidate_title']}",
                "",
                f"- first run id: `{packet['first_run_id']}`;",
                f"- candidate id: `{packet['candidate_id']}`;",
                f"- run scope: `{packet['run_scope']}`;",
                f"- primary model: `{packet['target_models']['primary']}`;",
                f"- secondary model: `{packet['target_models']['secondary_width_check']}`;",
                f"- artifact paths: `{format_value(packet['artifact_paths'])}`;",
                "",
                "Freeze before run:",
            ]
        )
        for item in packet["freeze_before_run"]:
            lines.append(f"- {item}")
        lines.extend(["", "Panels:"])
        for item in packet["panel_specs"]:
            lines.append(
                f"- `{item['panel_id']}` ({item['panel_type']}, n >= {item['minimum_rows']}): "
                f"{item['pass_rule']}"
            )
        lines.extend(
            [
                "",
                "Decision rules:",
                f"- `promotion_rule`: {packet['promotion_rule']}",
                f"- `death_rule`: {packet['death_rule']}",
                f"- `containment_rule`: {packet['containment_rule']}",
                f"- `export_rule`: {packet['export_rule']}",
            ]
        )

    lines.extend(
        [
            "",
            "## What This Proves",
            "",
            "It proves that the first behavior-only runs are now executable as",
            "predeclared packets. Every candidate has panels, thresholds,",
            "baselines, artifact paths, and decision rules.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not prove any candidate passes. It does not license",
            "hidden-state discovery or add mechanism evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_knowledge_first_run_pack(
    output_path: Path = KNOWLEDGE_FIRST_RUN_PACK_PATH,
    report_path: Path = KNOWLEDGE_FIRST_RUN_PACK_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_control_surface_knowledge_first_run_pack()
    validate_knowledge_first_run_pack(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write first-run artifacts")
    parser.add_argument("--json", action="store_true", help="print first-run JSON")
    args = parser.parse_args()

    payload = build_control_surface_knowledge_first_run_pack()
    validate_knowledge_first_run_pack(payload)

    if args.write:
        write_knowledge_first_run_pack()
        print(
            f"wrote {KNOWLEDGE_FIRST_RUN_PACK_PATH.relative_to(ROOT).as_posix()} and "
            f"{KNOWLEDGE_FIRST_RUN_PACK_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['packet_count']} first-run packets"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "knowledge first-run pack ok: "
        f"{payload['summary']['packet_count']} packets, "
        f"{payload['summary']['total_panel_count']} panels"
    )
    print(
        "packet_count_by_level:",
        json.dumps(payload["summary"]["packet_count_by_level"], sort_keys=True),
    )
    print("hidden_state_packets:", payload["summary"]["hidden_state_packet_count"])


if __name__ == "__main__":
    main()
