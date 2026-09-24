#!/usr/bin/env python
"""Build the control-surface knowledge second-wave outcome layer."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SECOND_WAVE_OUTCOMES_PATH = ROOT / "data" / "control_surface_knowledge_second_wave_outcomes.json"
SECOND_WAVE_REPORT_PATH = ROOT / "research" / "49_CONTROL_SURFACE_KNOWLEDGE_SECOND_WAVE_OUTCOMES.md"
TOPOLOGY_PATH = ROOT / "data" / "control_surface_knowledge_failure_topology.json"

KSQ002_REPAIR_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR"
    / "ksq002_source_rewrite_holdout_repair_full_behavior.json"
)
KSQ002_REPAIR_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR"
    / "ksq002_source_rewrite_holdout_repair_smoke_limit10.json"
)
KSQ002_REPAIR_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR"
    / "ksq002_source_rewrite_holdout_repair_first_run.json"
)
KSQ002_REPAIR_STATUS_PATH = ROOT / "research" / "cards" / "KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR_STATUS.md"
KSQ002_REPAIR_PREREG_PATH = ROOT / "research" / "prereg" / "KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR.md"
KSQ002_REPAIR_RUNNER_PATH = ROOT / "code" / "ksq002_source_rewrite_holdout_repair.py"

KSQ004_ADJUDICATION_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION"
    / "ksq004_template_invariance_adjudication_full_behavior.json"
)
KSQ004_ADJUDICATION_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION"
    / "ksq004_template_invariance_adjudication_smoke_limit10.json"
)
KSQ004_ADJUDICATION_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION"
    / "ksq004_template_invariance_adjudication_first_run.json"
)
KSQ004_ADJUDICATION_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION_STATUS.md"
)
KSQ004_ADJUDICATION_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION.md"
)
KSQ004_ADJUDICATION_RUNNER_PATH = ROOT / "code" / "ksq004_template_invariance_adjudication.py"

KSQ001_BOUND_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND"
    / "ksq001_familiar_prior_parseability_bound_full_behavior.json"
)
KSQ001_BOUND_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND"
    / "ksq001_familiar_prior_parseability_bound_smoke_limit10.json"
)
KSQ001_BOUND_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND"
    / "ksq001_familiar_prior_parseability_bound_first_run.json"
)
KSQ001_BOUND_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND_STATUS.md"
)
KSQ001_BOUND_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND.md"
)
KSQ001_BOUND_RUNNER_PATH = ROOT / "code" / "ksq001_familiar_prior_parseability_bound.py"

KSQ003_REDESIGN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN"
    / "ksq003_evidence_sufficiency_redesign_full_behavior.json"
)
KSQ003_REDESIGN_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN"
    / "ksq003_evidence_sufficiency_redesign_smoke_limit10.json"
)
KSQ003_REDESIGN_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN"
    / "ksq003_evidence_sufficiency_redesign_first_run.json"
)
KSQ003_REDESIGN_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN_STATUS.md"
)
KSQ003_REDESIGN_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN.md"
)
KSQ003_REDESIGN_RUNNER_PATH = ROOT / "code" / "ksq003_evidence_sufficiency_redesign.py"

KSQ005006_REDESIGN_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY"
    / "ksq005_006_relation_evidence_answerability_full_behavior.json"
)
KSQ005006_REDESIGN_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY"
    / "ksq005_006_relation_evidence_answerability_smoke_limit10.json"
)
KSQ005006_REDESIGN_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY"
    / "ksq005_006_relation_evidence_answerability_first_run.json"
)
KSQ005006_REDESIGN_STATUS_PATH = (
    ROOT
    / "research"
    / "cards"
    / "KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY_STATUS.md"
)
KSQ005006_REDESIGN_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY.md"
)
KSQ005006_REDESIGN_RUNNER_PATH = (
    ROOT / "code" / "ksq005_006_relation_evidence_answerability_redesign.py"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def panel_summary(summary: dict[str, Any], panel: str) -> dict[str, Any]:
    selected = summary["selected_template_summary"]
    item = selected["panels"][panel]
    return {
        "label_counts": item["label_counts"],
        "parseable_rate": item["parseable_rate"],
        "artificial_value_rate": item["artificial_value_rate"],
        "unknown_rate": item["unknown_rate"],
        "prior_or_lure_rate": item["prior_or_lure_rate"],
    }


def ksq001_parseability_bound_outcome(
    topology: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    work_orders = {
        work_order["work_order_id"]: work_order for work_order in topology["second_wave_work_orders"]
    }
    work_order = work_orders["bound_ksq001_familiar_prior_parseability"]
    summary = result["summary"]
    decision = result["repair_decision"]
    criteria = summary["criteria"]
    failed_criteria = [
        key for key, value in criteria.items() if key != "smoke_mode" and value is not True
    ]
    selected_primary = summary["selected_template_summary"]["primary"]
    return {
        "work_order_id": work_order["work_order_id"],
        "candidate_id": "ksq001_familiar_entity_prior_counterbalance",
        "repair_id": result["repair_id"],
        "priority": work_order["priority"],
        "track": work_order["track"],
        "status": "completed",
        "route_decision": decision["route_decision"],
        "repair_verdict": decision["repair_verdict"],
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "promotion_rule_passed": decision["promotion_rule_passed"],
        "kill_rule_triggered": decision["kill_rule_triggered"],
        "behavior_ready": decision["behavior_ready"],
        "signature_screen_allowed": decision["signature_screen_allowed"],
        "hidden_state_claim_allowed": decision["hidden_state_claim_allowed"],
        "intervention_allowed": decision["intervention_allowed"],
        "mechanism_claim_allowed": decision["mechanism_claim_allowed"],
        "failed_criteria": failed_criteria,
        "passing_templates": decision["passing_templates"],
        "collapse_mode": decision["collapse_mode"],
        "full_behavior": {
            "path": rel(KSQ001_BOUND_RESULT_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "selected_template": summary["selection"]["selected_template"],
            "candidate_and_output_margins_reported": criteria[
                "candidate_and_output_margins_reported"
            ],
            "parseability_delta_vs_first_run_compact": decision[
                "parseability_delta_vs_first_run_compact"
            ],
            "first_run_compact_primary_parseability": decision[
                "first_run_compact_primary_parseability"
            ],
            "selected_primary": {
                "label_counts": selected_primary["label_counts"],
                "parseable_rate": selected_primary["parseable_rate"],
                "artificial_value_rate": selected_primary["artificial_value_rate"],
                "prior_or_lure_rate": selected_primary["prior_or_lure_rate"],
                "unknown_rate": selected_primary["unknown_rate"],
                "unparsed_rate": selected_primary["unparsed_rate"],
            },
            "template_gates": decision["template_gates"],
        },
        "artifact_paths": {
            "runner": rel(KSQ001_BOUND_RUNNER_PATH),
            "prereg": rel(KSQ001_BOUND_PREREG_PATH),
            "status_card": rel(KSQ001_BOUND_STATUS_PATH),
            "structural_result": rel(KSQ001_BOUND_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ001_BOUND_SMOKE_PATH),
            "full_behavior_result": rel(KSQ001_BOUND_RESULT_PATH),
        },
        "allowed_claim": (
            "KSQ001 familiar-prior parseability repair was adjudicated with a "
            "compact replay and two softer answer-shape variants. The 10-source "
            "contrast candidate did not scale: the full selected compact replay "
            "kept the original 24/40 parseable conflict distribution and no "
            "template passed the full gate."
        ),
        "forbidden_claim": (
            "KSQ001 is not behavior-ready, does not license hidden-state search, "
            "and does not show a familiar-prior or learned-memory control surface."
        ),
        "insight": (
            "The familiar-prior mixture is scale-fragile and answer-shape "
            "sensitive. Softer answer formatting can improve parseability on "
            "some rows, but it removes the prior branch or pushes rows toward "
            "UNKNOWN rather than producing a clean behavior substrate."
        ),
    }


def ksq002_repair_outcome(topology: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    work_orders = {
        work_order["work_order_id"]: work_order for work_order in topology["second_wave_work_orders"]
    }
    work_order = work_orders["repair_ksq002_source_rewrite_holdout"]
    summary = result["summary"]
    decision = result["repair_decision"]
    selected = summary["selected_template_summary"]
    rewrite_holdout = selected["rewrite_holdout"]
    criteria = summary["criteria"]
    failed_criteria = [
        key for key, value in criteria.items() if key != "smoke_mode" and value is not True
    ]
    return {
        "work_order_id": work_order["work_order_id"],
        "candidate_id": "ksq002_familiar_entity_source_rewrite_equivalence",
        "repair_id": result["repair_id"],
        "priority": work_order["priority"],
        "track": work_order["track"],
        "status": "completed",
        "route_decision": decision["route_decision"],
        "repair_verdict": decision["repair_verdict"],
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "promotion_rule_passed": summary["behavior_ready"] is True,
        "kill_rule_triggered": decision["route_decision"] == "kill_ordinary_source_rewrite_repair",
        "behavior_ready": summary["behavior_ready"],
        "signature_screen_allowed": decision["signature_screen_allowed"],
        "hidden_state_claim_allowed": decision["hidden_state_claim_allowed"],
        "intervention_allowed": decision["intervention_allowed"],
        "mechanism_claim_allowed": decision["mechanism_claim_allowed"],
        "failed_criteria": failed_criteria,
        "full_behavior": {
            "path": rel(KSQ002_REPAIR_RESULT_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "selected_template": summary["selection"]["selected_template"],
            "rewrite_delta_from_baseline": summary["rewrite_delta_from_baseline"],
            "candidate_and_output_margins_reported": criteria[
                "candidate_and_output_margins_reported"
            ],
            "panels": {
                panel: panel_summary(summary, panel)
                for panel in [
                    "baseline_source_value_lookup",
                    "neutral_rewrite_lookup",
                    "source_deletion",
                    "query_only_control",
                    "source_disjoint_rewrite_holdout",
                    "rewrite_output_geometry_audit",
                ]
            },
            "source_disjoint_rewrite_holdout": {
                "rows": rewrite_holdout["rows"],
                "label_counts": rewrite_holdout["label_counts"],
                "parseable_rate": rewrite_holdout["parseable_rate"],
                "artificial_value_rate": rewrite_holdout["artificial_value_rate"],
            },
        },
        "artifact_paths": {
            "runner": rel(KSQ002_REPAIR_RUNNER_PATH),
            "prereg": rel(KSQ002_REPAIR_PREREG_PATH),
            "status_card": rel(KSQ002_REPAIR_STATUS_PATH),
            "structural_result": rel(KSQ002_REPAIR_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ002_REPAIR_SMOKE_PATH),
            "full_behavior_result": rel(KSQ002_REPAIR_RESULT_PATH),
        },
        "allowed_claim": (
            "The city-field answer-channel repair made neutral rewrite and "
            "source-disjoint rewrite holdout lookup perfect in the full run, "
            "but it degraded baseline lookup and broke answer-absent source "
            "deletion locality. Ordinary KSQ002 source-rewrite repair is killed."
        ),
        "forbidden_claim": (
            "The repair does not make KSQ002 behavior-ready, does not license "
            "hidden-state search, and does not show a source-channel mechanism."
        ),
        "insight": (
            "For this familiar-entity source-rewrite substrate, improving the "
            "answer channel can strengthen prompt-present rewrite behavior while "
            "weakening baseline and null behavior. The control surface is coupled "
            "to the answer interface, not a clean source-channel variable."
        ),
    }


def ksq003_evidence_sufficiency_outcome(
    topology: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    work_orders = {
        work_order["work_order_id"]: work_order for work_order in topology["second_wave_work_orders"]
    }
    work_order = work_orders["redesign_statusless_bridge_substrate"]
    summary = result["summary"]
    decision = result["redesign_decision"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    failed_criteria = [
        key for key, value in criteria.items() if key != "smoke_mode" and value is not True
    ]
    template_gates = {}
    for template, item in summary["by_template"].items():
        template_gates[template] = {
            "complete_identity_atomic_rate": item["panels"]["complete_identity_conflict"][
                "atomic_number_rate"
            ],
            "complete_identity_local_rate": item["panels"]["complete_identity_conflict"][
                "local_number_rate"
            ],
            "contradictory_identity_unknown_rate": item["panels"][
                "contradictory_identity_null"
            ]["unknown_rate"],
            "symbol_only_unknown_rate": item["panels"]["symbol_only_ablation"][
                "unknown_rate"
            ],
            "initial_only_unknown_rate": item["panels"]["initial_only_ablation"][
                "unknown_rate"
            ],
            "answer_absent_unknown_rate": item["panels"][
                "answer_absent_and_side_null"
            ]["unknown_rate"],
            "null_stress_unknown_rate": item["null_stress"]["unknown_rate"],
            "holdout_complete_identity_atomic_rate": item[
                "primary_conflict_holdout"
            ]["atomic_number_rate"],
        }
    return {
        "work_order_id": work_order["work_order_id"],
        "candidate_id": "ksq003_bridge_statusless_evidence_aggregation",
        "redesign_id": result["redesign_id"],
        "priority": work_order["priority"],
        "track": work_order["track"],
        "status": "completed",
        "route_decision": decision["route_decision"],
        "redesign_verdict": decision["redesign_verdict"],
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "promotion_rule_passed": decision["promotion_rule_passed"],
        "kill_rule_triggered": decision["kill_rule_triggered"],
        "behavior_ready": decision["behavior_ready"],
        "signature_screen_allowed": decision["signature_screen_allowed"],
        "hidden_state_claim_allowed": decision["hidden_state_claim_allowed"],
        "intervention_allowed": decision["intervention_allowed"],
        "mechanism_claim_allowed": decision["mechanism_claim_allowed"],
        "failed_criteria": failed_criteria,
        "full_behavior": {
            "path": rel(KSQ003_REDESIGN_RESULT_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "selected_template": summary["selection"]["selected_template"],
            "candidate_and_output_margins_reported": criteria[
                "candidate_and_output_margins_reported"
            ],
            "selected_primary_conflict": {
                "label_counts": selected["primary_conflict"]["label_counts"],
                "parseable_rate": selected["primary_conflict"]["parseable_rate"],
                "atomic_number_rate": selected["primary_conflict"][
                    "atomic_number_rate"
                ],
                "local_number_rate": selected["primary_conflict"][
                    "local_number_rate"
                ],
                "unknown_rate": selected["primary_conflict"]["unknown_rate"],
            },
            "selected_null_stress": {
                "label_counts": selected["null_stress"]["label_counts"],
                "parseable_rate": selected["null_stress"]["parseable_rate"],
                "unknown_rate": selected["null_stress"]["unknown_rate"],
                "local_number_rate": selected["null_stress"]["local_number_rate"],
                "atomic_number_rate": selected["null_stress"]["atomic_number_rate"],
            },
            "template_gates": template_gates,
        },
        "artifact_paths": {
            "runner": rel(KSQ003_REDESIGN_RUNNER_PATH),
            "prereg": rel(KSQ003_REDESIGN_PREREG_PATH),
            "status_card": rel(KSQ003_REDESIGN_STATUS_PATH),
            "structural_result": rel(KSQ003_REDESIGN_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ003_REDESIGN_SMOKE_PATH),
            "full_behavior_result": rel(KSQ003_REDESIGN_RESULT_PATH),
        },
        "allowed_claim": (
            "KSQ003 evidence-sufficiency redesign removed the original local-table "
            "dominance pattern but did not create a behavior-ready bridge. Direct "
            "local lookup, direct learned recall, answer-absent nulls, source split, "
            "prompt audit, and candidate/output reporting passed; complete-evidence "
            "learned routing and null-stress rejection did not pass together."
        ),
        "forbidden_claim": (
            "KSQ003 remains hidden-state-disallowed. The redesign does not license "
            "a signature screen, intervention, mechanism card, or internal "
            "knowledge-control claim."
        ),
        "insight": (
            "The second KSQ003 design changes the failure type. The original "
            "statusless evidence route collapsed mismatch and ablation rows to "
            "local lab numbers; the redesign mostly suppresses local intrusion, "
            "but exposes an evidence-sufficiency tradeoff. The identity-packet "
            "template gets 40/40 complete-evidence atomic answers while failing "
            "null stress almost completely. The compact template gets better "
            "UNKNOWN behavior but only 13/40 complete-evidence atomic answers. "
            "Statusless evidence can suppress the local table or activate learned "
            "recall, but this design cannot do both reliably."
        ),
    }


def ksq004_template_adjudication_outcome(
    topology: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    work_orders = {
        work_order["work_order_id"]: work_order for work_order in topology["second_wave_work_orders"]
    }
    work_order = work_orders["adjudicate_ksq004_template_invariance"]
    summary = result["summary"]
    decision = result["adjudication_decision"]
    failed_templates = [
        template
        for template, gate in decision["template_gates"].items()
        if gate["passed"] is not True
    ]
    return {
        "work_order_id": work_order["work_order_id"],
        "candidate_id": "ksq004_bridge_answer_interface_minimal_pairs",
        "adjudication_id": result["adjudication_id"],
        "priority": work_order["priority"],
        "track": work_order["track"],
        "status": "completed",
        "route_decision": decision["route_decision"],
        "adjudication_verdict": decision["adjudication_verdict"],
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "promotion_rule_passed": decision["behavior_ready"] is True,
        "kill_rule_triggered": decision["route_decision"]
        == "kill_same_family_answer_interface_repair",
        "behavior_ready": decision["behavior_ready"],
        "signature_screen_allowed": decision["signature_screen_allowed"],
        "hidden_state_claim_allowed": decision["hidden_state_claim_allowed"],
        "intervention_allowed": decision["intervention_allowed"],
        "mechanism_claim_allowed": decision["mechanism_claim_allowed"],
        "passing_templates": decision["passing_templates"],
        "non_question_passing_templates": decision["non_question_passing_templates"],
        "failed_templates": failed_templates,
        "full_behavior": {
            "path": rel(KSQ004_ADJUDICATION_RESULT_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "selected_template": summary["selection"]["selected_template"],
            "candidate_and_output_margins_reported": summary["criteria"][
                "candidate_and_output_margins_reported"
            ],
            "template_gates": decision["template_gates"],
        },
        "artifact_paths": {
            "runner": rel(KSQ004_ADJUDICATION_RUNNER_PATH),
            "prereg": rel(KSQ004_ADJUDICATION_PREREG_PATH),
            "status_card": rel(KSQ004_ADJUDICATION_STATUS_PATH),
            "structural_result": rel(KSQ004_ADJUDICATION_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ004_ADJUDICATION_SMOKE_PATH),
            "full_behavior_result": rel(KSQ004_ADJUDICATION_RESULT_PATH),
        },
        "allowed_claim": (
            "KSQ004 template-invariance adjudication found a behavior-ready "
            "bridge substrate across two predeclared templates: question_form "
            "and relation_key_form. neutral_sentence_form still fails conflict "
            "routing, so the result is a bounded behavior substrate, not a "
            "general bridge mechanism."
        ),
        "forbidden_claim": (
            "KSQ004 does not yet have a hidden-state signature, intervention, "
            "or mechanism card. The result only licenses a later signature "
            "screen under the admitted behavior substrate."
        ),
        "insight": (
            "The first-run template fragility was too coarse: compact underscore "
            "format collapsed expected-atomic rows, but a relation-key form with "
            "the same bare numeric answer channel generalized the bridge. The "
            "control surface is answer-interface and relation-format dependent."
        ),
    }


def ksq005006_relation_answerability_outcome(
    topology: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    work_orders = {
        work_order["work_order_id"]: work_order for work_order in topology["second_wave_work_orders"]
    }
    work_order = work_orders["redesign_real_uncertainty_answerability"]
    summary = result["summary"]
    decision = result["redesign_decision"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    failed_criteria = [
        key for key, value in criteria.items() if key != "smoke_mode" and value is not True
    ]
    template_gates = {}
    for template, item in summary["by_template"].items():
        template_gates[template] = {
            "known_correct_rate": item["panels"]["known_factual_direct"][
                "known_correct_rate"
            ],
            "supported_answer_rate": item["panels"]["supported_relation_rows"][
                "supported_answer_rate"
            ],
            "unknown_nonce_abstain_rate": item["panels"][
                "unknown_nonce_absent_rows"
            ]["abstain_rate"],
            "unsupported_abstain_rate": item["panels"]["unsupported_relation_rows"][
                "abstain_rate"
            ],
            "contradiction_abstain_rate": item["panels"][
                "contradictory_relation_rows"
            ]["abstain_rate"],
            "control_abstain_rate": item["controls"]["control_abstain_rate"],
            "control_reproduced_supported_rate": item["controls"][
                "control_reproduced_supported_rate"
            ],
        }
    return {
        "work_order_id": work_order["work_order_id"],
        "candidate_id": "ksq005_006_relation_evidence_answerability",
        "redesign_id": result["redesign_id"],
        "priority": work_order["priority"],
        "track": work_order["track"],
        "status": "completed",
        "route_decision": decision["route_decision"],
        "redesign_verdict": decision["redesign_verdict"],
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "promotion_rule_passed": decision["promotion_rule_passed"],
        "kill_rule_triggered": decision["kill_rule_triggered"],
        "behavior_ready": decision["behavior_ready"],
        "signature_screen_allowed": decision["signature_screen_allowed"],
        "hidden_state_claim_allowed": decision["hidden_state_claim_allowed"],
        "intervention_allowed": decision["intervention_allowed"],
        "mechanism_claim_allowed": decision["mechanism_claim_allowed"],
        "failed_criteria": failed_criteria,
        "full_behavior": {
            "path": rel(KSQ005006_REDESIGN_RESULT_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "selected_template": summary["selection"]["selected_template"],
            "candidate_and_output_margins_reported": criteria[
                "candidate_and_output_margins_reported"
            ],
            "panels": {
                panel: {
                    "label_counts": selected["panels"][panel]["label_counts"],
                    "parseable_rate": selected["panels"][panel]["parseable_rate"],
                    "known_correct_rate": selected["panels"][panel][
                        "known_correct_rate"
                    ],
                    "supported_answer_rate": selected["panels"][panel][
                        "supported_answer_rate"
                    ],
                    "abstain_rate": selected["panels"][panel]["abstain_rate"],
                    "control_abstain_rate": selected["panels"][panel][
                        "control_abstain_rate"
                    ],
                    "control_reproduced_supported_rate": selected["panels"][panel][
                        "control_reproduced_supported_rate"
                    ],
                    "unsupported_answer_rate": selected["panels"][panel][
                        "unsupported_answer_rate"
                    ],
                    "prior_or_true_answer_rate": selected["panels"][panel][
                        "prior_or_true_answer_rate"
                    ],
                }
                for panel in [
                    "known_factual_direct",
                    "supported_relation_rows",
                    "unknown_nonce_absent_rows",
                    "unsupported_relation_rows",
                    "contradictory_relation_rows",
                    "claim_only_control",
                    "mention_only_control",
                ]
            },
            "controls": {
                "label_counts": selected["controls"]["label_counts"],
                "control_abstain_rate": selected["controls"][
                    "control_abstain_rate"
                ],
                "control_reproduced_supported_rate": selected["controls"][
                    "control_reproduced_supported_rate"
                ],
            },
            "template_gates": template_gates,
        },
        "artifact_paths": {
            "runner": rel(KSQ005006_REDESIGN_RUNNER_PATH),
            "prereg": rel(KSQ005006_REDESIGN_PREREG_PATH),
            "status_card": rel(KSQ005006_REDESIGN_STATUS_PATH),
            "structural_result": rel(KSQ005006_REDESIGN_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ005006_REDESIGN_SMOKE_PATH),
            "full_behavior_result": rel(KSQ005006_REDESIGN_RESULT_PATH),
        },
        "allowed_claim": (
            "The KSQ005/KSQ006 relation-evidence redesign passed structural audit "
            "and preserved known/direct plus supported relation answers under the "
            "selected compact relation contract, but it failed unknown nonce, "
            "unsupported relation, contradiction, and claim/mention controls. The "
            "current real-uncertainty route is killed rather than admitted."
        ),
        "forbidden_claim": (
            "The redesign does not make KSQ005/KSQ006 behavior-ready, does not "
            "license hidden-state search, and does not show uncertainty, refusal, "
            "context-support, or knowledge-control mechanisms."
        ),
        "insight": (
            "A formal relation grammar separates two failure modes. The compact "
            "contract answers known facts and exact `REL capital_of` rows, but "
            "claim-only and mention-only controls still reproduce the capital on "
            "79/80 rows and contradiction rows choose the true capital on 38/40 "
            "rows. The stricter relation_rows template protects controls and "
            "abstains on unknown/unsupported/contradictory rows, but answers only "
            "11/40 supported relation rows. The model can use a visible relation "
            "grammar either as an answer trigger or as an abstention guard, but "
            "this route cannot make both roles stable together."
        ),
    }


def build_control_surface_knowledge_second_wave_outcomes() -> dict[str, Any]:
    topology = load_json(TOPOLOGY_PATH)
    ksq001_result = load_json(KSQ001_BOUND_RESULT_PATH)
    ksq002_result = load_json(KSQ002_REPAIR_RESULT_PATH)
    ksq003_result = load_json(KSQ003_REDESIGN_RESULT_PATH)
    ksq004_result = load_json(KSQ004_ADJUDICATION_RESULT_PATH)
    ksq005006_result = load_json(KSQ005006_REDESIGN_RESULT_PATH)
    rows = [
        ksq001_parseability_bound_outcome(topology, ksq001_result),
        ksq002_repair_outcome(topology, ksq002_result),
        ksq003_evidence_sufficiency_outcome(topology, ksq003_result),
        ksq004_template_adjudication_outcome(topology, ksq004_result),
        ksq005006_relation_answerability_outcome(topology, ksq005006_result),
    ]
    completed_work_order_ids = {row["work_order_id"] for row in rows}
    pending_work_orders = [
        work_order["work_order_id"]
        for work_order in topology["second_wave_work_orders"]
        if work_order["work_order_id"] not in completed_work_order_ids
    ]
    route_decisions = Counter(row["route_decision"] for row in rows)
    verdicts = Counter(
        row.get("repair_verdict")
        or row.get("redesign_verdict")
        or row.get("adjudication_verdict")
        for row in rows
    )
    exported = Counter(row["exported_diagnostic_class"] for row in rows)
    summary = {
        "outcome_count": len(rows),
        "completed_work_order_count": len(rows),
        "pending_work_order_count": len(pending_work_orders),
        "behavior_ready_count": sum(1 for row in rows if row["behavior_ready"] is True),
        "signature_screen_allowed_count": sum(
            1 for row in rows if row["signature_screen_allowed"] is True
        ),
        "hidden_state_claim_allowed_count": sum(
            1 for row in rows if row["hidden_state_claim_allowed"] is True
        ),
        "intervention_allowed_count": sum(
            1 for row in rows if row["intervention_allowed"] is True
        ),
        "killed_route_count": sum(
            1
            for row in rows
            if row["route_decision"]
            in {
                "closeout_familiar_prior_parseability_tradeoff",
                "kill_ordinary_familiar_prior_parseability_repair",
                "kill_ordinary_source_rewrite_repair",
                "kill_same_family_answer_interface_repair",
                "kill_current_uncertainty_route",
            }
        ),
        "route_decision_counts": dict(sorted(route_decisions.items())),
        "verdict_counts": dict(sorted(verdicts.items())),
        "exported_diagnostic_class_counts": dict(sorted(exported.items())),
        "pending_work_orders": pending_work_orders,
    }
    payload = {
        "schema_version": 1,
        "updated_at": time.strftime("%Y-%m-%d"),
        "purpose": (
            "Record second-wave outcomes against the generated knowledge failure "
            "topology so completed work orders become auditable route decisions."
        ),
        "sources": {
            "knowledge_failure_topology": rel(TOPOLOGY_PATH),
            "ksq001_parseability_bound_full_behavior": rel(KSQ001_BOUND_RESULT_PATH),
            "ksq002_repair_full_behavior": rel(KSQ002_REPAIR_RESULT_PATH),
            "ksq003_evidence_sufficiency_full_behavior": rel(
                KSQ003_REDESIGN_RESULT_PATH
            ),
            "ksq004_template_adjudication_full_behavior": rel(
                KSQ004_ADJUDICATION_RESULT_PATH
            ),
            "ksq005006_relation_answerability_full_behavior": rel(
                KSQ005006_REDESIGN_RESULT_PATH
            ),
        },
        "summary": summary,
        "outcome_rows": rows,
        "validation_checks": validation_checks(summary, rows, topology),
        "allowed_claim": (
            "All five second-wave KSQ work orders have been executed: KSQ001 "
            "familiar-prior parseability repair was closed as a scale-fragile "
            "tradeoff, KSQ002 ordinary source-rewrite repair was killed by "
            "locality regression, KSQ003 evidence-sufficiency redesign exposed "
            "a statusless evidence boundary without admitting hidden-state work, "
            "KSQ004 template-invariance adjudication admitted a bounded behavior "
            "substrate and a later signature screen, and KSQ005/KSQ006 relation-"
            "evidence answerability killed the current real-uncertainty route."
        ),
        "forbidden_claim": (
            "The second-wave layer does not promote any knowledge-control surface, "
            "hidden-state claim, intervention, or internal mechanism claim."
        ),
    }
    validate_knowledge_second_wave_outcomes(payload)
    return payload


def validation_checks(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    topology: dict[str, Any],
) -> list[dict[str, Any]]:
    work_order_ids = {
        work_order["work_order_id"] for work_order in topology["second_wave_work_orders"]
    }
    row_by_work_order = {row["work_order_id"]: row for row in rows}
    ksq001 = row_by_work_order["bound_ksq001_familiar_prior_parseability"]
    ksq002 = row_by_work_order["repair_ksq002_source_rewrite_holdout"]
    ksq003 = row_by_work_order["redesign_statusless_bridge_substrate"]
    ksq004 = row_by_work_order["adjudicate_ksq004_template_invariance"]
    ksq005006 = row_by_work_order["redesign_real_uncertainty_answerability"]
    paths_missing = [
        path
        for row in rows
        for path in row["artifact_paths"].values()
        if not (ROOT / path).exists()
    ]
    return [
        {
            "id": "executed_work_orders_exist_in_topology",
            "predicate": "completed work orders are subset of topology work orders",
            "actual": sorted(row_by_work_order),
            "passed": set(row_by_work_order).issubset(work_order_ids),
            "why": "Second-wave outcomes must close predeclared topology work orders.",
        },
        {
            "id": "all_second_wave_artifacts_exist",
            "predicate": "empty list",
            "actual": paths_missing,
            "passed": not paths_missing,
            "why": "The outcome row must point to concrete runner, prereg, status, and result artifacts.",
        },
        {
            "id": "ksq001_parseability_tradeoff_closed",
            "predicate": "full selected compact replay stays at 0.600 parseability and admits no behavior substrate",
            "actual": {
                "route_decision": ksq001["route_decision"],
                "behavior_ready": ksq001["behavior_ready"],
                "signature_screen_allowed": ksq001["signature_screen_allowed"],
                "selected_primary": ksq001["full_behavior"]["selected_primary"],
                "passing_templates": ksq001["passing_templates"],
            },
            "passed": ksq001["route_decision"]
            == "closeout_familiar_prior_parseability_tradeoff"
            and ksq001["behavior_ready"] is False
            and ksq001["signature_screen_allowed"] is False
            and ksq001["full_behavior"]["selected_primary"]["parseable_rate"] == 0.6
            and ksq001["full_behavior"]["selected_primary"]["label_counts"]
            == {
                "artificial_value": 18,
                "real_prior": 2,
                "unknown": 4,
                "unparsed": 16,
            }
            and ksq001["passing_templates"] == [],
            "why": "The KSQ001 second-wave result is a closeout of ordinary parseability repair, not a behavior admission.",
        },
        {
            "id": "repair_full_scope_preserved",
            "predicate": "source_count == 40 and record_count == 240",
            "actual": ksq002["full_behavior"],
            "passed": ksq002["full_behavior"]["source_count"] == 40
            and ksq002["full_behavior"]["record_count"] == 240,
            "why": "The repair must keep the full 40-source scope and all six panels for one fixed template.",
        },
        {
            "id": "holdout_fixed_but_locality_failed",
            "predicate": "holdout parseability == 1.0 and source_deletion_passed is failed",
            "actual": {
                "holdout_parseability": ksq002["full_behavior"][
                    "source_disjoint_rewrite_holdout"
                ]["parseable_rate"],
                "failed_criteria": ksq002["failed_criteria"],
            },
            "passed": ksq002["full_behavior"]["source_disjoint_rewrite_holdout"][
                "parseable_rate"
            ]
            == 1.0
            and "source_deletion_passed" in ksq002["failed_criteria"],
            "why": "The scientific result is the tradeoff: repair succeeds on the named holdout but fails locality.",
        },
        {
            "id": "ksq004_template_invariance_admitted",
            "predicate": "question_form and relation_key_form pass; neutral_sentence_form fails",
            "actual": ksq004["full_behavior"]["template_gates"],
            "passed": ksq004["behavior_ready"] is True
            and ksq004["signature_screen_allowed"] is True
            and ksq004["passing_templates"] == ["question_form", "relation_key_form"]
            and ksq004["full_behavior"]["template_gates"]["neutral_sentence_form"][
                "passed"
            ]
            is False,
            "why": "The KSQ004 adjudication must admit only the bounded two-template behavior substrate observed in the full result.",
        },
        {
            "id": "ksq003_evidence_sufficiency_boundary_recorded",
            "predicate": "full 40-source redesign suppresses local dominance but fails learned/null joint gate",
            "actual": {
                "route_decision": ksq003["route_decision"],
                "behavior_ready": ksq003["behavior_ready"],
                "selected_template": ksq003["full_behavior"]["selected_template"],
                "selected_primary": ksq003["full_behavior"][
                    "selected_primary_conflict"
                ],
                "selected_null_stress": ksq003["full_behavior"]["selected_null_stress"],
                "template_gates": ksq003["full_behavior"]["template_gates"],
            },
            "passed": ksq003["route_decision"]
            == "bound_evidence_sufficiency_without_hidden_state"
            and ksq003["behavior_ready"] is False
            and ksq003["signature_screen_allowed"] is False
            and ksq003["full_behavior"]["source_count"] == 40
            and ksq003["full_behavior"]["record_count"] == 840
            and ksq003["full_behavior"]["selected_template"] == "compact_identity"
            and ksq003["full_behavior"]["selected_primary_conflict"][
                "local_number_rate"
            ]
            == 0.0
            and ksq003["full_behavior"]["selected_primary_conflict"][
                "atomic_number_rate"
            ]
            == 0.325
            and ksq003["full_behavior"]["selected_null_stress"]["unknown_rate"]
            == 0.55
            and ksq003["full_behavior"]["template_gates"]["identity_packet"][
                "complete_identity_atomic_rate"
            ]
            == 1.0
            and ksq003["full_behavior"]["template_gates"]["identity_packet"][
                "null_stress_unknown_rate"
            ]
            == 0.008333333333333333,
            "why": "The KSQ003 redesign is a new sufficiency-boundary diagnostic, not a hidden-state-ready bridge and not a repeat of local-table dominance.",
        },
        {
            "id": "ksq005006_relation_answerability_route_killed",
            "predicate": "selected compact relation passes supported answers but fails unknown/control gates",
            "actual": {
                "route_decision": ksq005006["route_decision"],
                "behavior_ready": ksq005006["behavior_ready"],
                "selected_template": ksq005006["full_behavior"]["selected_template"],
                "template_gates": ksq005006["full_behavior"]["template_gates"],
                "controls": ksq005006["full_behavior"]["controls"],
            },
            "passed": ksq005006["route_decision"] == "kill_current_uncertainty_route"
            and ksq005006["behavior_ready"] is False
            and ksq005006["signature_screen_allowed"] is False
            and ksq005006["full_behavior"]["source_count"] == 40
            and ksq005006["full_behavior"]["record_count"] == 840
            and ksq005006["full_behavior"]["selected_template"] == "compact_relation"
            and ksq005006["full_behavior"]["template_gates"]["compact_relation"][
                "supported_answer_rate"
            ]
            == 1.0
            and ksq005006["full_behavior"]["template_gates"]["compact_relation"][
                "control_reproduced_supported_rate"
            ]
            == 0.9875
            and ksq005006["full_behavior"]["template_gates"]["relation_rows"][
                "control_abstain_rate"
            ]
            == 0.825
            and ksq005006["full_behavior"]["template_gates"]["relation_rows"][
                "supported_answer_rate"
            ]
            == 0.275,
            "why": "The real-uncertainty redesign must close as a relation-evidence tradeoff, not as a behavior substrate.",
        },
        {
            "id": "no_second_wave_hidden_or_intervention_claims",
            "predicate": "hidden_state_claim_allowed_count == 0 and intervention_allowed_count == 0",
            "actual": {
                "behavior_ready_count": summary["behavior_ready_count"],
                "signature_screen_allowed_count": summary["signature_screen_allowed_count"],
                "hidden_state_claim_allowed_count": summary[
                    "hidden_state_claim_allowed_count"
                ],
                "intervention_allowed_count": summary["intervention_allowed_count"],
            },
            "passed": summary["hidden_state_claim_allowed_count"] == 0
            and summary["intervention_allowed_count"] == 0,
            "why": "A behavior admission may license a later signature screen, but not a hidden-state claim or intervention.",
        },
    ]


def validate_knowledge_second_wave_outcomes(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("second-wave outcomes schema_version must be 1")
    rows = payload.get("outcome_rows")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("second-wave outcomes must contain KSQ001, KSQ002, KSQ003, KSQ004, and KSQ005/006 rows")
    row_by_work_order = {row["work_order_id"]: row for row in rows}
    expected_work_orders = {
        "bound_ksq001_familiar_prior_parseability",
        "repair_ksq002_source_rewrite_holdout",
        "redesign_statusless_bridge_substrate",
        "adjudicate_ksq004_template_invariance",
        "redesign_real_uncertainty_answerability",
    }
    if set(row_by_work_order) != expected_work_orders:
        raise ValueError("second-wave outcome work orders changed")
    ksq001 = row_by_work_order["bound_ksq001_familiar_prior_parseability"]
    ksq002 = row_by_work_order["repair_ksq002_source_rewrite_holdout"]
    ksq003 = row_by_work_order["redesign_statusless_bridge_substrate"]
    ksq004 = row_by_work_order["adjudicate_ksq004_template_invariance"]
    ksq005006 = row_by_work_order["redesign_real_uncertainty_answerability"]
    if ksq001.get("behavior_ready") is not False:
        raise ValueError("KSQ001 parseability bound must not be behavior-ready")
    if ksq001.get("signature_screen_allowed") is not False:
        raise ValueError("KSQ001 parseability bound must not allow a signature screen")
    if ksq001.get("route_decision") != "closeout_familiar_prior_parseability_tradeoff":
        raise ValueError("KSQ001 parseability bound route decision changed")
    if (
        ksq001.get("exported_diagnostic_class")
        != "FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF"
    ):
        raise ValueError("KSQ001 parseability bound diagnostic changed")
    ksq001_full = ksq001["full_behavior"]
    if ksq001_full["source_count"] != 40 or ksq001_full["record_count"] != 480:
        raise ValueError("KSQ001 parseability bound full behavior scope changed")
    if ksq001_full["selected_template"] != "compact_original_replay":
        raise ValueError("KSQ001 parseability bound selected template changed")
    if ksq001_full["parseability_delta_vs_first_run_compact"] != 0.0:
        raise ValueError("KSQ001 parseability delta changed")
    if ksq001_full["selected_primary"]["label_counts"] != {
        "artificial_value": 18,
        "real_prior": 2,
        "unknown": 4,
        "unparsed": 16,
    }:
        raise ValueError("KSQ001 selected-primary label counts changed")
    if ksq001_full["selected_primary"]["parseable_rate"] != 0.6:
        raise ValueError("KSQ001 selected-primary parseability changed")
    if ksq001.get("passing_templates") != []:
        raise ValueError("KSQ001 passing templates changed")
    gates001 = ksq001_full["template_gates"]
    if gates001["compact_original_replay"]["primary_parseable_rate"] != 0.6:
        raise ValueError("KSQ001 compact replay parseability changed")
    if gates001["compact_begin_city"]["primary_parseable_rate"] != 0.825:
        raise ValueError("KSQ001 begin-city parseability changed")
    if gates001["compact_city_name_first"]["primary_parseable_rate"] != 0.875:
        raise ValueError("KSQ001 city-name-first parseability changed")
    if ksq002.get("behavior_ready") is not False:
        raise ValueError("KSQ002 repair must not be behavior-ready")
    if ksq002.get("signature_screen_allowed") is not False:
        raise ValueError("KSQ002 repair must not allow a signature screen")
    if ksq002.get("route_decision") != "kill_ordinary_source_rewrite_repair":
        raise ValueError("KSQ002 repair route decision changed")
    if (
        ksq002.get("exported_diagnostic_class")
        != "SOURCE_REWRITE_REPAIR_LOCALITY_REGRESSION"
    ):
        raise ValueError("KSQ002 repair diagnostic changed")
    if set(ksq002.get("failed_criteria", [])) != {
        "baseline_source_value_lookup_passed",
        "neutral_rewrite_lookup_passed",
        "source_deletion_passed",
    }:
        raise ValueError("KSQ002 repair failed criteria changed")
    full = ksq002["full_behavior"]
    if full["source_count"] != 40 or full["record_count"] != 240:
        raise ValueError("KSQ002 repair full behavior scope changed")
    if full["source_disjoint_rewrite_holdout"]["label_counts"] != {"artificial_value": 16}:
        raise ValueError("KSQ002 repair holdout counts changed")
    if full["panels"]["source_deletion"]["label_counts"] != {"unknown": 10, "unparsed": 30}:
        raise ValueError("KSQ002 repair source-deletion boundary changed")
    if ksq003.get("behavior_ready") is not False:
        raise ValueError("KSQ003 evidence sufficiency redesign must not be behavior-ready")
    if ksq003.get("signature_screen_allowed") is not False:
        raise ValueError("KSQ003 evidence sufficiency redesign must not allow a signature screen")
    if ksq003.get("hidden_state_claim_allowed") is not False:
        raise ValueError("KSQ003 evidence sufficiency redesign must not claim hidden-state success")
    if ksq003.get("intervention_allowed") is not False:
        raise ValueError("KSQ003 evidence sufficiency redesign must not allow intervention")
    if ksq003.get("route_decision") != "bound_evidence_sufficiency_without_hidden_state":
        raise ValueError("KSQ003 evidence sufficiency route decision changed")
    if (
        ksq003.get("exported_diagnostic_class")
        != "STATUSLESS_EVIDENCE_SUFFICIENCY_BOUNDARY"
    ):
        raise ValueError("KSQ003 evidence sufficiency diagnostic changed")
    if set(ksq003.get("failed_criteria", [])) != {
        "complete_identity_conflict_atomic_passed",
        "contradictory_identity_unknown_passed",
        "single_feature_ablation_unknown_passed",
        "holdout_complete_identity_atomic_passed",
    }:
        raise ValueError("KSQ003 evidence sufficiency failed criteria changed")
    ksq003_full = ksq003["full_behavior"]
    if ksq003_full["source_count"] != 40 or ksq003_full["record_count"] != 840:
        raise ValueError("KSQ003 evidence sufficiency full behavior scope changed")
    if ksq003_full["selected_template"] != "compact_identity":
        raise ValueError("KSQ003 evidence sufficiency selected template changed")
    if ksq003_full["selected_primary_conflict"]["label_counts"] != {
        "atomic_number": 13,
        "unknown": 27,
    }:
        raise ValueError("KSQ003 selected primary conflict counts changed")
    if ksq003_full["selected_null_stress"]["label_counts"] != {
        "atomic_number": 41,
        "local_number": 5,
        "other_number": 5,
        "unknown": 66,
        "unparsed": 3,
    }:
        raise ValueError("KSQ003 selected null-stress counts changed")
    gates003 = ksq003_full["template_gates"]
    if gates003["identity_packet"]["complete_identity_atomic_rate"] != 1.0:
        raise ValueError("KSQ003 identity-packet learned branch changed")
    if gates003["identity_packet"]["null_stress_unknown_rate"] != 0.008333333333333333:
        raise ValueError("KSQ003 identity-packet null-stress boundary changed")
    if gates003["compact_identity"]["complete_identity_atomic_rate"] != 0.325:
        raise ValueError("KSQ003 compact learned branch changed")
    if gates003["compact_identity"]["null_stress_unknown_rate"] != 0.55:
        raise ValueError("KSQ003 compact null-stress boundary changed")
    if ksq004.get("behavior_ready") is not True:
        raise ValueError("KSQ004 adjudication must be behavior-ready")
    if ksq004.get("signature_screen_allowed") is not True:
        raise ValueError("KSQ004 adjudication must allow a later signature screen")
    if ksq004.get("hidden_state_claim_allowed") is not False:
        raise ValueError("KSQ004 adjudication must not claim hidden-state success")
    if ksq004.get("intervention_allowed") is not False:
        raise ValueError("KSQ004 adjudication must not allow intervention yet")
    if ksq004.get("route_decision") != "admit_behavior_substrate_only":
        raise ValueError("KSQ004 route decision changed")
    if ksq004.get("exported_diagnostic_class") != "TEMPLATE_INVARIANT_BRIDGE_BEHAVIOR":
        raise ValueError("KSQ004 exported diagnostic changed")
    if ksq004.get("passing_templates") != ["question_form", "relation_key_form"]:
        raise ValueError("KSQ004 passing templates changed")
    ksq004_full = ksq004["full_behavior"]
    if ksq004_full["source_count"] != 40 or ksq004_full["record_count"] != 1080:
        raise ValueError("KSQ004 adjudication full behavior scope changed")
    gates = ksq004_full["template_gates"]
    if gates["question_form"]["passed"] is not True:
        raise ValueError("KSQ004 question_form gate changed")
    if gates["relation_key_form"]["passed"] is not True:
        raise ValueError("KSQ004 relation_key_form gate changed")
    if gates["neutral_sentence_form"]["passed"] is not False:
        raise ValueError("KSQ004 neutral_sentence_form boundary changed")
    if gates["relation_key_form"]["conflict_expected_correct_rate"] != 0.975:
        raise ValueError("KSQ004 relation_key_form conflict rate changed")
    if gates["neutral_sentence_form"]["conflict_expected_correct_rate"] != 0.675:
        raise ValueError("KSQ004 neutral_sentence_form conflict rate changed")
    if ksq005006.get("behavior_ready") is not False:
        raise ValueError("KSQ005/006 relation answerability must not be behavior-ready")
    if ksq005006.get("signature_screen_allowed") is not False:
        raise ValueError("KSQ005/006 relation answerability must not allow a signature screen")
    if ksq005006.get("hidden_state_claim_allowed") is not False:
        raise ValueError("KSQ005/006 relation answerability must not claim hidden-state success")
    if ksq005006.get("intervention_allowed") is not False:
        raise ValueError("KSQ005/006 relation answerability must not allow intervention")
    if ksq005006.get("route_decision") != "kill_current_uncertainty_route":
        raise ValueError("KSQ005/006 relation answerability route decision changed")
    if (
        ksq005006.get("exported_diagnostic_class")
        != "RELATION_EVIDENCE_ANSWERABILITY_BOUNDARY"
    ):
        raise ValueError("KSQ005/006 relation answerability diagnostic changed")
    if set(ksq005006.get("failed_criteria", [])) != {
        "unknown_nonce_absent_rows_passed",
        "unsupported_relation_rows_passed",
        "contradictory_relation_rows_passed",
        "claim_only_and_mention_only_controls_passed",
        "source_disjoint_answerability_holdout_passed",
    }:
        raise ValueError("KSQ005/006 relation answerability failed criteria changed")
    ksq005006_full = ksq005006["full_behavior"]
    if ksq005006_full["source_count"] != 40 or ksq005006_full["record_count"] != 840:
        raise ValueError("KSQ005/006 relation answerability full behavior scope changed")
    if ksq005006_full["selected_template"] != "compact_relation":
        raise ValueError("KSQ005/006 selected template changed")
    gates005006 = ksq005006_full["template_gates"]
    if gates005006["compact_relation"]["known_correct_rate"] != 0.975:
        raise ValueError("KSQ005/006 compact known-direct rate changed")
    if gates005006["compact_relation"]["supported_answer_rate"] != 1.0:
        raise ValueError("KSQ005/006 compact supported rate changed")
    if gates005006["compact_relation"]["unknown_nonce_abstain_rate"] != 0.675:
        raise ValueError("KSQ005/006 compact unknown abstain rate changed")
    if gates005006["compact_relation"]["control_reproduced_supported_rate"] != 0.9875:
        raise ValueError("KSQ005/006 compact control reproduction changed")
    if gates005006["relation_rows"]["control_abstain_rate"] != 0.825:
        raise ValueError("KSQ005/006 relation_rows control abstention changed")
    if gates005006["relation_rows"]["supported_answer_rate"] != 0.275:
        raise ValueError("KSQ005/006 relation_rows supported rate changed")
    summary = payload["summary"]
    if summary["outcome_count"] != 5:
        raise ValueError("second-wave outcome count changed")
    if summary["completed_work_order_count"] != 5:
        raise ValueError("second-wave completed work order count changed")
    if summary["pending_work_order_count"] != 0:
        raise ValueError("second-wave pending work order count changed")
    if summary["behavior_ready_count"] != 1:
        raise ValueError("second-wave behavior-ready count changed")
    if summary["signature_screen_allowed_count"] != 1:
        raise ValueError("second-wave signature-screen count changed")
    if summary["killed_route_count"] != 3:
        raise ValueError("second-wave killed-route count changed")
    if summary["hidden_state_claim_allowed_count"] != 0:
        raise ValueError("second-wave hidden-state claim count changed")
    if summary["intervention_allowed_count"] != 0:
        raise ValueError("second-wave intervention count changed")
    failed_checks = [
        check["id"] for check in payload.get("validation_checks", []) if check.get("passed") is not True
    ]
    if failed_checks:
        raise ValueError(f"second-wave validation checks failed: {failed_checks}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = {row["work_order_id"]: row for row in payload["outcome_rows"]}
    ksq001 = rows["bound_ksq001_familiar_prior_parseability"]
    ksq002 = rows["repair_ksq002_source_rewrite_holdout"]
    ksq003 = rows["redesign_statusless_bridge_substrate"]
    ksq004 = rows["adjudicate_ksq004_template_invariance"]
    ksq005006 = rows["redesign_real_uncertainty_answerability"]
    ksq001_full = ksq001["full_behavior"]
    ksq002_full = ksq002["full_behavior"]
    ksq003_full = ksq003["full_behavior"]
    ksq004_full = ksq004["full_behavior"]
    ksq005006_full = ksq005006["full_behavior"]
    lines = [
        "# Control-Surface Knowledge Second-Wave Outcomes",
        "",
        "Status: generated second-wave outcome layer implemented and validated.",
        "",
        "Machine-readable source:",
        "",
        "> `data/control_surface_knowledge_second_wave_outcomes.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_second_wave_outcomes.py`",
        "",
        "Regenerate:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_second_wave_outcomes.py --write",
        "python code\\control_surface_knowledge_second_wave_outcomes.py",
        "```",
        "",
        "## Summary",
        "",
        f"- completed work orders: `{summary['completed_work_order_count']}`",
        f"- pending work orders: `{summary['pending_work_order_count']}`",
        f"- behavior-ready rows: `{summary['behavior_ready_count']}`",
        f"- signature-screen-allowed rows: `{summary['signature_screen_allowed_count']}`",
        f"- killed routes: `{summary['killed_route_count']}`",
        f"- hidden-state-claim rows: `{summary['hidden_state_claim_allowed_count']}`",
        f"- intervention-allowed rows: `{summary['intervention_allowed_count']}`",
        "",
        "## KSQ001 Parseability Bound Outcome",
        "",
        f"- work order: `{ksq001['work_order_id']}`",
        f"- repair verdict: `{ksq001['repair_verdict']}`",
        f"- route decision: `{ksq001['route_decision']}`",
        f"- exported diagnostic: `{ksq001['exported_diagnostic_class']}`",
        f"- behavior ready: `{str(ksq001['behavior_ready']).lower()}`",
        f"- signature screen allowed: `{str(ksq001['signature_screen_allowed']).lower()}`",
        "",
        "### KSQ001 Full-Run Evidence",
        "",
        f"- result: `{ksq001_full['path']}`",
        f"- sources: `{ksq001_full['source_count']}`",
        f"- records: `{ksq001_full['record_count']}`",
        f"- selected template: `{ksq001_full['selected_template']}`",
        f"- first-run compact parseability: `{ksq001_full['first_run_compact_primary_parseability']:.3f}`",
        f"- parseability delta: `{ksq001_full['parseability_delta_vs_first_run_compact']:.3f}`",
        f"- selected primary labels: `{json.dumps(ksq001_full['selected_primary']['label_counts'], sort_keys=True)}`",
        "",
        "| Template | Passed | Parseable | Artificial | Prior/Lure | UNKNOWN | Collapse |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for template, gate in ksq001_full["template_gates"].items():
        lines.append(
            f"| `{template}` | `{str(gate['passed']).lower()}` | "
            f"`{gate['primary_parseable_rate']:.3f}` | "
            f"`{gate['primary_artificial_rate']:.3f}` | "
            f"`{gate['primary_prior_or_lure_rate']:.3f}` | "
            f"`{gate['primary_unknown_rate']:.3f}` | `{gate['collapse_mode']}` |"
        )
    lines.extend(
        [
            "",
            "### KSQ001 Interpretation",
            "",
            ksq001["insight"],
            "",
            "The 10-source smoke candidate did not scale. The selected full",
            "compact replay exactly reproduces the original conflict parseability",
            "boundary: 24/40 parseable rows with 18 artificial-value answers, 2",
            "real-prior answers, 4 UNKNOWN answers, and 16 unparsed rows. Softer",
            "answer-shape variants raise parseability only by losing the prior",
            "branch or pushing rows toward UNKNOWN. Ordinary KSQ001 parseability",
            "repair is therefore closed rather than polished.",
            "",
            "## KSQ002 Repair Outcome",
            "",
            f"- work order: `{ksq002['work_order_id']}`",
            f"- repair verdict: `{ksq002['repair_verdict']}`",
            f"- route decision: `{ksq002['route_decision']}`",
            f"- exported diagnostic: `{ksq002['exported_diagnostic_class']}`",
            f"- behavior ready: `{str(ksq002['behavior_ready']).lower()}`",
            f"- signature screen allowed: `{str(ksq002['signature_screen_allowed']).lower()}`",
            "",
            "### KSQ002 Full-Run Evidence",
            "",
            f"- result: `{ksq002_full['path']}`",
            f"- sources: `{ksq002_full['source_count']}`",
            f"- records: `{ksq002_full['record_count']}`",
            f"- selected template: `{ksq002_full['selected_template']}`",
            f"- rewrite delta from baseline: `{ksq002_full['rewrite_delta_from_baseline']:.3f}`",
            "",
            "| Panel | Label Counts | Parseable | Artificial | UNKNOWN | Prior/Lure |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for panel_name, panel in ksq002_full["panels"].items():
        lines.append(
            f"| `{panel_name}` | `{json.dumps(panel['label_counts'], sort_keys=True)}` | "
            f"`{panel['parseable_rate']:.3f}` | `{panel['artificial_value_rate']:.3f}` | "
            f"`{panel['unknown_rate']:.3f}` | `{panel['prior_or_lure_rate']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "### KSQ002 Interpretation",
            "",
            ksq002["insight"],
            "",
            "The named holdout boundary was repaired: source-disjoint rewrite holdout",
            "went to 16/16 artificial-value answers. That did not make KSQ002",
            "behavior-ready because the same answer-channel repair reduced baseline",
            "lookup to 30/40 and source-deletion UNKNOWN to 10/40. The ordinary",
            "source-rewrite repair route is therefore killed rather than polished.",
            "",
            "## KSQ003 Evidence-Sufficiency Redesign Outcome",
            "",
            f"- work order: `{ksq003['work_order_id']}`",
            f"- redesign verdict: `{ksq003['redesign_verdict']}`",
            f"- route decision: `{ksq003['route_decision']}`",
            f"- exported diagnostic: `{ksq003['exported_diagnostic_class']}`",
            f"- behavior ready: `{str(ksq003['behavior_ready']).lower()}`",
            f"- signature screen allowed: `{str(ksq003['signature_screen_allowed']).lower()}`",
            "",
            "### KSQ003 Full-Run Evidence",
            "",
            f"- result: `{ksq003_full['path']}`",
            f"- sources: `{ksq003_full['source_count']}`",
            f"- records: `{ksq003_full['record_count']}`",
            f"- selected template: `{ksq003_full['selected_template']}`",
            f"- selected primary labels: `{json.dumps(ksq003_full['selected_primary_conflict']['label_counts'], sort_keys=True)}`",
            f"- selected null-stress labels: `{json.dumps(ksq003_full['selected_null_stress']['label_counts'], sort_keys=True)}`",
            "",
            "| Template | Complete Atomic | Complete Local | Contradiction UNKNOWN | Symbol UNKNOWN | Initial UNKNOWN | Answer-Absent UNKNOWN | Null-Stress UNKNOWN | Holdout Atomic |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for template, gate in ksq003_full["template_gates"].items():
        lines.append(
            f"| `{template}` | `{gate['complete_identity_atomic_rate']:.3f}` | "
            f"`{gate['complete_identity_local_rate']:.3f}` | "
            f"`{gate['contradictory_identity_unknown_rate']:.3f}` | "
            f"`{gate['symbol_only_unknown_rate']:.3f}` | "
            f"`{gate['initial_only_unknown_rate']:.3f}` | "
            f"`{gate['answer_absent_unknown_rate']:.3f}` | "
            f"`{gate['null_stress_unknown_rate']:.3f}` | "
            f"`{gate['holdout_complete_identity_atomic_rate']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "### KSQ003 Interpretation",
            "",
            ksq003["insight"],
            "",
            "The important result is not another local-table-dominance failure.",
            "`identity_packet` produces 40/40 complete-evidence atomic answers,",
            "but null-stress UNKNOWN is only 1/120 across contradiction and",
            "single-feature ablations. `compact_identity` improves null stress",
            "to 66/120 UNKNOWN and removes complete-conflict local answers, but",
            "complete-evidence atomic answers fall to 13/40. The route is bounded",
            "as an evidence-sufficiency boundary and remains hidden-state-disallowed.",
            "",
            "## KSQ004 Template-Invariance Outcome",
            "",
            f"- work order: `{ksq004['work_order_id']}`",
            f"- adjudication verdict: `{ksq004['adjudication_verdict']}`",
            f"- route decision: `{ksq004['route_decision']}`",
            f"- exported diagnostic: `{ksq004['exported_diagnostic_class']}`",
            f"- behavior ready: `{str(ksq004['behavior_ready']).lower()}`",
            f"- signature screen allowed: `{str(ksq004['signature_screen_allowed']).lower()}`",
            f"- passing templates: `{json.dumps(ksq004['passing_templates'])}`",
            "",
            "### KSQ004 Full-Run Evidence",
            "",
            f"- result: `{ksq004_full['path']}`",
            f"- sources: `{ksq004_full['source_count']}`",
            f"- records: `{ksq004_full['record_count']}`",
            f"- selected template: `{ksq004_full['selected_template']}`",
            "",
            "| Template | Passed | Conflict Expected | Atomic-Branch Atomic | Holdout Expected | Null UNKNOWN | Side Answer |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for template, gate in ksq004_full["template_gates"].items():
        lines.append(
            f"| `{template}` | `{str(gate['passed']).lower()}` | "
            f"`{gate['conflict_expected_correct_rate']:.3f}` | "
            f"`{gate['atomic_branch_atomic_rate']:.3f}` | "
            f"`{gate['holdout_expected_correct_rate']:.3f}` | "
            f"`{gate['null_unknown_rate']:.3f}` | `{gate['side_answer_rate']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "### KSQ004 Interpretation",
            "",
            ksq004["insight"],
            "",
            "`question_form` and `relation_key_form` pass the full behavior gate.",
            "`neutral_sentence_form` does not: conflict expected-correct is 0.675",
            "and expected-atomic conflict rows still collapse too often. This is a",
            "bounded behavior-substrate admission, not a mechanism card.",
            "",
            "## KSQ005/KSQ006 Relation-Evidence Answerability Outcome",
            "",
            f"- work order: `{ksq005006['work_order_id']}`",
            f"- redesign verdict: `{ksq005006['redesign_verdict']}`",
            f"- route decision: `{ksq005006['route_decision']}`",
            f"- exported diagnostic: `{ksq005006['exported_diagnostic_class']}`",
            f"- behavior ready: `{str(ksq005006['behavior_ready']).lower()}`",
            f"- signature screen allowed: `{str(ksq005006['signature_screen_allowed']).lower()}`",
            "",
            "### KSQ005/KSQ006 Full-Run Evidence",
            "",
            f"- result: `{ksq005006_full['path']}`",
            f"- sources: `{ksq005006_full['source_count']}`",
            f"- records: `{ksq005006_full['record_count']}`",
            f"- selected template: `{ksq005006_full['selected_template']}`",
            f"- combined controls: `{json.dumps(ksq005006_full['controls']['label_counts'], sort_keys=True)}`",
            "",
            "| Panel | Label Counts | Parseable | Known | Supported | Abstain | Unsupported | Prior/True | Control Abstain | Control Reproduced |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for panel_name, panel in ksq005006_full["panels"].items():
        lines.append(
            f"| `{panel_name}` | `{json.dumps(panel['label_counts'], sort_keys=True)}` | "
            f"`{panel['parseable_rate']:.3f}` | `{panel['known_correct_rate']:.3f}` | "
            f"`{panel['supported_answer_rate']:.3f}` | `{panel['abstain_rate']:.3f}` | "
            f"`{panel['unsupported_answer_rate']:.3f}` | `{panel['prior_or_true_answer_rate']:.3f}` | "
            f"`{panel['control_abstain_rate']:.3f}` | `{panel['control_reproduced_supported_rate']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "| Template | Known | Supported | Unknown Abstain | Unsupported Abstain | Contradiction Abstain | Control Abstain | Control Reproduced |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for template, gate in ksq005006_full["template_gates"].items():
        lines.append(
            f"| `{template}` | `{gate['known_correct_rate']:.3f}` | "
            f"`{gate['supported_answer_rate']:.3f}` | "
            f"`{gate['unknown_nonce_abstain_rate']:.3f}` | "
            f"`{gate['unsupported_abstain_rate']:.3f}` | "
            f"`{gate['contradiction_abstain_rate']:.3f}` | "
            f"`{gate['control_abstain_rate']:.3f}` | "
            f"`{gate['control_reproduced_supported_rate']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "### KSQ005/KSQ006 Interpretation",
            "",
            ksq005006["insight"],
            "",
            "`compact_relation` answers known/direct rows at 39/40 and exact",
            "`REL capital_of` supported rows at 40/40, but unknown nonce rows",
            "only abstain at 27/40, unsupported rows abstain at 28/40,",
            "contradiction rows choose the prior/true answer at 38/40, and",
            "claim/mention controls reproduce the supported capital at 79/80.",
            "`relation_rows` protects controls and abstention, but supported",
            "relation answering collapses to 11/40. The current real-uncertainty",
            "route is killed and no hidden-state screen is licensed.",
            "",
            "## Allowed Claim",
            "",
            payload["allowed_claim"],
            "",
            "## Forbidden Claim",
            "",
            payload["forbidden_claim"],
            "",
            "## Validation Checks",
            "",
            "| Check | Passed |",
            "| --- | --- |",
        ]
    )
    for check in payload["validation_checks"]:
        lines.append(f"| `{check['id']}` | `{str(check['passed']).lower()}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = build_control_surface_knowledge_second_wave_outcomes()
    if args.write:
        write_json(SECOND_WAVE_OUTCOMES_PATH, payload)
        write_report(SECOND_WAVE_REPORT_PATH, payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
