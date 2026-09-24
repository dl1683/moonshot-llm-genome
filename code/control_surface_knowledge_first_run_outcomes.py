"""Build the executed knowledge first-run outcome matrix.

The first-run pack names what should be tried. This layer records what happened
after all six KSQ behavior-substrate runs: which gate stopped each candidate,
which controls survived, which diagnostic was exported, and why hidden-state
work is still forbidden.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json
from control_surface_knowledge_first_run_pack import KNOWLEDGE_FIRST_RUN_PACK_PATH


KNOWLEDGE_FIRST_RUN_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_first_run_outcomes.json"
)
KNOWLEDGE_FIRST_RUN_OUTCOMES_REPORT_PATH = (
    ROOT / "research" / "47_CONTROL_SURFACE_KNOWLEDGE_FIRST_RUN_OUTCOMES.md"
)

OUTCOME_CONFIGS = [
    {
        "candidate_id": "ksq001_familiar_entity_prior_counterbalance",
        "card_dir": "KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE",
        "first_run_result": "ksq001_familiar_entity_prior_counterbalance_first_run.json",
        "smoke_result": "ksq001_familiar_entity_prior_counterbalance_smoke_limit10.json",
        "full_behavior_result": "ksq001_familiar_entity_prior_counterbalance_full_behavior.json",
        "status_card": "research/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN_STATUS.md",
        "prereg": "research/prereg/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN.md",
        "runner": "code/ksq001_familiar_entity_prior_counterbalance_first_run.py",
        "terminal_gate": "full_behavior_gate",
        "expected_diagnostic_class": "familiar_entity_conflict_parseability_failed",
        "exported_diagnostic_class": "FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF",
        "failure_axes": ["conflict_mixture_parseability"],
        "survived_controls": [
            "source-local artificial lookup",
            "direct real-capital prior recall",
            "answer-absent nulls",
            "source-disjoint mixture",
            "candidate/output margin reporting",
        ],
        "failed_gates": ["full conflict parseability"],
        "allowed_claim": (
            "Familiar priors and prompt-local artificial values can both be "
            "separately controlled, with weak conflict mixture."
        ),
        "forbidden_claim": (
            "KSQ001 is not a behavior-ready knowledge substrate and licenses no "
            "hidden-state work."
        ),
    },
    {
        "candidate_id": "ksq002_familiar_entity_source_rewrite_equivalence",
        "card_dir": "KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE",
        "first_run_result": "ksq002_familiar_entity_source_rewrite_equivalence_first_run.json",
        "smoke_result": "ksq002_familiar_entity_source_rewrite_equivalence_smoke_limit10.json",
        "full_behavior_result": "ksq002_familiar_entity_source_rewrite_equivalence_full_behavior.json",
        "status_card": "research/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN_STATUS.md",
        "prereg": "research/prereg/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN.md",
        "runner": "code/ksq002_familiar_entity_source_rewrite_equivalence_first_run.py",
        "terminal_gate": "full_behavior_gate",
        "expected_diagnostic_class": "source_rewrite_holdout_failed",
        "exported_diagnostic_class": "SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE",
        "failure_axes": ["source_disjoint_rewrite_parseability"],
        "survived_controls": [
            "baseline source-value lookup",
            "neutral rewrite lookup",
            "source deletion null",
            "query-only null",
            "candidate/output margin reporting",
        ],
        "failed_gates": ["full source-disjoint rewrite holdout parseability"],
        "allowed_claim": (
            "Source rewrite is mostly robust and source-local under the selected "
            "full template, with clean deletion/query-only controls."
        ),
        "forbidden_claim": (
            "KSQ002 is not source-rewrite invariant and licenses no internal "
            "source-channel mechanism claim."
        ),
    },
    {
        "candidate_id": "ksq003_bridge_statusless_evidence_aggregation",
        "card_dir": "KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION",
        "first_run_result": "ksq003_bridge_statusless_evidence_aggregation_first_run.json",
        "smoke_result": "ksq003_bridge_statusless_evidence_aggregation_smoke_limit10.json",
        "full_behavior_result": None,
        "status_card": "research/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN_STATUS.md",
        "prereg": "research/prereg/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN.md",
        "runner": "code/ksq003_bridge_statusless_evidence_aggregation_first_run.py",
        "terminal_gate": "smoke_behavior_gate",
        "expected_diagnostic_class": "smoke_only",
        "exported_diagnostic_class": "STATUSLESS_EVIDENCE_LOCAL_TABLE_DOMINANCE",
        "failure_axes": [
            "one_evidence_mismatch_local_table_dominance",
            "single_feature_ablation_local_table_dominance",
        ],
        "survived_controls": [
            "local-number direct control",
            "learned atomic-number direct control",
            "all-evidence-fit local routing",
            "answer-absent nulls",
        ],
        "failed_gates": [
            "mismatch evidence should route to atomic branch",
            "symbol/parity-only ablations should become unknown",
        ],
        "allowed_claim": (
            "Statusless symbol/parity evidence is insufficient under this "
            "contract even though direct learned recall and nulls are available."
        ),
        "forbidden_claim": (
            "KSQ003 is not a behavior-ready learned/local bridge and licenses no "
            "hidden-state work."
        ),
    },
    {
        "candidate_id": "ksq004_bridge_answer_interface_minimal_pairs",
        "card_dir": "KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS",
        "first_run_result": "ksq004_bridge_answer_interface_minimal_pairs_first_run.json",
        "smoke_result": "ksq004_bridge_answer_interface_minimal_pairs_smoke_limit10.json",
        "full_behavior_result": "ksq004_bridge_answer_interface_minimal_pairs_full_behavior.json",
        "status_card": "research/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN_STATUS.md",
        "prereg": "research/prereg/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN.md",
        "runner": "code/ksq004_bridge_answer_interface_minimal_pairs_first_run.py",
        "terminal_gate": "full_behavior_gate",
        "expected_diagnostic_class": "bridge_minimal_pair_contrast_absent",
        "exported_diagnostic_class": "ANSWER_INTERFACE_TEMPLATE_FRAGILITY",
        "failure_axes": [
            "compact_template_learned_branch_collapse",
            "answer_interface_not_sufficient_explanation",
        ],
        "survived_controls": [
            "matched answer interface smoke",
            "direct local and atomic controls",
            "side-answer leakage smoke",
            "answer-absent null smoke",
            "candidate/output margin reporting",
        ],
        "failed_gates": ["full compact-form minimal-pair conflict robustness"],
        "allowed_claim": (
            "Answer-interface matching alone does not explain away the bridge, "
            "because question_form mostly retains local-vs-learned contrast."
        ),
        "forbidden_claim": (
            "KSQ004 is not a behavior-ready answer-interface or learned-memory "
            "bridge substrate."
        ),
    },
    {
        "candidate_id": "ksq005_uncertainty_grounded_answerability",
        "card_dir": "KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY",
        "first_run_result": "ksq005_uncertainty_grounded_answerability_first_run.json",
        "smoke_result": "ksq005_uncertainty_grounded_answerability_smoke_limit10.json",
        "full_behavior_result": None,
        "status_card": "research/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN_STATUS.md",
        "prereg": "research/prereg/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN.md",
        "runner": "code/ksq005_uncertainty_grounded_answerability_first_run.py",
        "terminal_gate": "smoke_behavior_gate",
        "expected_diagnostic_class": "unknown_nonce_abstention_failed",
        "exported_diagnostic_class": "GROUNDED_ANSWERABILITY_UNKNOWN_UNSUPPORTED_FAILURE",
        "failure_axes": [
            "unknown_nonce_abstention_parseability",
            "unsupported_context_abstention_parseability",
        ],
        "survived_controls": [
            "known factual direct control",
            "contradicted familiar-context correction/abstention",
            "candidate/output margin reporting",
        ],
        "failed_gates": [
            "unknown nonce abstention",
            "unsupported context abstention",
        ],
        "allowed_claim": (
            "Correction of false familiar-context claims is easier than grounded "
            "abstention on unknown or unsupported entities."
        ),
        "forbidden_claim": (
            "KSQ005 is not an uncertainty, refusal, or answerability-control "
            "surface."
        ),
    },
    {
        "candidate_id": "ksq006_uncertainty_context_support_counterfactuals",
        "card_dir": "KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS",
        "first_run_result": "ksq006_uncertainty_context_support_counterfactuals_first_run.json",
        "smoke_result": "ksq006_uncertainty_context_support_counterfactuals_smoke_limit10.json",
        "full_behavior_result": None,
        "status_card": "research/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN_STATUS.md",
        "prereg": "research/prereg/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN.md",
        "runner": "code/ksq006_uncertainty_context_support_counterfactuals_first_run.py",
        "terminal_gate": "smoke_behavior_gate",
        "expected_diagnostic_class": "insufficient_context_abstention_failed",
        "exported_diagnostic_class": "CONTEXT_SUPPORT_CLAIM_OR_MENTION_CONTROL_FAILURE",
        "failure_axes": [
            "insufficient_context_abstention",
            "claim_or_context_only_answer_channel",
        ],
        "survived_controls": [
            "supported context",
            "irrelevant context abstention",
            "contradicting context detection",
            "candidate/output margin reporting",
        ],
        "failed_gates": [
            "insufficient context abstention",
            "claim/context-only control locality",
        ],
        "allowed_claim": (
            "Context support has a partial substrate, but relation-free claim or "
            "city mention controls still carry the answer channel."
        ),
        "forbidden_claim": (
            "KSQ006 is not a context-support, uncertainty, or correction-control "
            "surface."
        ),
    },
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def result_path(config: dict[str, Any], key: str) -> Path | None:
    filename = config.get(key)
    if filename is None:
        return None
    return ROOT / "results" / "cards" / config["card_dir"] / filename


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def by_candidate(packets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for packet in packets:
        candidate_id = packet["candidate_id"]
        if candidate_id in result:
            raise AssertionError(f"duplicate first-run packet {candidate_id}")
        result[candidate_id] = packet
    return result


def summarize_result(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    summary = payload.get("summary")
    structural = payload.get("structural")
    if isinstance(summary, dict):
        structural = summary.get("structural", structural)
    if not isinstance(structural, dict):
        raise AssertionError(f"{rel(path)} missing structural payload")
    selected_template = None
    if isinstance(summary, dict) and isinstance(summary.get("selection"), dict):
        selected_template = summary["selection"].get("selected_template")
    return {
        "path": rel(path),
        "schema_version": payload.get("schema_version"),
        "candidate_id": payload.get("candidate_id"),
        "run_type": payload.get("run_type"),
        "limit_sources": payload.get("limit_sources"),
        "hidden_state_allowed": payload.get("hidden_state_allowed"),
        "structural_passed": structural.get("passed"),
        "record_count": structural.get("record_count"),
        "source_count": structural.get("source_count"),
        "panel_counts": structural.get("panel_counts"),
        "template_counts": structural.get("template_counts"),
        "split_source_counts": structural.get("split_source_counts"),
        "diagnostic_class": summary.get("diagnostic_class") if isinstance(summary, dict) else None,
        "behavior_ready": summary.get("behavior_ready") if isinstance(summary, dict) else None,
        "behavior_candidate": summary.get("behavior_candidate") if isinstance(summary, dict) else None,
        "selected_template": selected_template,
        "criteria": summary.get("criteria") if isinstance(summary, dict) else None,
        "passed": summary.get("passed") if isinstance(summary, dict) else structural.get("passed"),
        "summary_hidden_state_allowed": (
            summary.get("hidden_state_allowed") if isinstance(summary, dict) else None
        ),
    }


def failed_criteria(summary: dict[str, Any]) -> list[str]:
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        return []
    return sorted(key for key, value in criteria.items() if value is False)


def build_outcome_rows(first_run_pack: dict[str, Any]) -> list[dict[str, Any]]:
    packets_by_candidate = by_candidate(first_run_pack["first_run_packets"])
    rows = []
    for config in OUTCOME_CONFIGS:
        candidate_id = config["candidate_id"]
        packet = packets_by_candidate[candidate_id]
        structural = summarize_result(result_path(config, "first_run_result"))
        smoke = summarize_result(result_path(config, "smoke_result"))
        full_path = result_path(config, "full_behavior_result")
        full = summarize_result(full_path) if full_path is not None else None
        final_result = full if full is not None else smoke
        final_hidden_state_allowed = any(
            value is True
            for value in [
                structural["hidden_state_allowed"],
                smoke["hidden_state_allowed"],
                smoke["summary_hidden_state_allowed"],
                final_result["hidden_state_allowed"],
                final_result["summary_hidden_state_allowed"],
            ]
        )
        row = {
            "candidate_id": candidate_id,
            "candidate_title": packet["candidate_title"],
            "level_id": packet["level_id"],
            "admission_class": packet["admission_class"],
            "priority_class": packet["priority_class"],
            "terminal_gate": config["terminal_gate"],
            "final_diagnostic_class": final_result["diagnostic_class"],
            "exported_diagnostic_class": config["exported_diagnostic_class"],
            "behavior_ready": final_result["behavior_ready"],
            "behavior_candidate": final_result["behavior_candidate"],
            "hidden_state_allowed": final_hidden_state_allowed,
            "structural": structural,
            "smoke": smoke,
            "full_behavior": full,
            "has_full_behavior_result": full is not None,
            "final_selected_template": final_result["selected_template"],
            "final_failed_criteria": failed_criteria(final_result),
            "failure_axes": config["failure_axes"],
            "survived_controls": config["survived_controls"],
            "failed_gates": config["failed_gates"],
            "artifact_paths": {
                "runner": config["runner"],
                "prereg": config["prereg"],
                "status_card": config["status_card"],
                "first_run_result": structural["path"],
                "smoke_result": smoke["path"],
                "full_behavior_result": full["path"] if full is not None else None,
            },
            "allowed_claim": config["allowed_claim"],
            "forbidden_claim": config["forbidden_claim"],
        }
        rows.append(row)
    return rows


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(row[key] for row in rows).items()))


def count_items(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row[key])
    return dict(sorted(counter.items()))


def build_summary(rows: list[dict[str, Any]], first_run_pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome_count": len(rows),
        "first_run_packet_count": first_run_pack["summary"]["packet_count"],
        "structural_passed_count": sum(1 for row in rows if row["structural"]["structural_passed"] is True),
        "smoke_result_count": sum(1 for row in rows if row["smoke"] is not None),
        "full_behavior_result_count": sum(1 for row in rows if row["has_full_behavior_result"]),
        "terminal_gate_counts": count_values(rows, "terminal_gate"),
        "level_counts": count_values(rows, "level_id"),
        "admission_class_counts": count_values(rows, "admission_class"),
        "diagnostic_class_counts": count_values(rows, "final_diagnostic_class"),
        "exported_diagnostic_class_counts": count_values(rows, "exported_diagnostic_class"),
        "failure_axis_counts": count_items(rows, "failure_axes"),
        "survived_control_counts": count_items(rows, "survived_controls"),
        "behavior_ready_count": sum(1 for row in rows if row["behavior_ready"] is True),
        "behavior_candidate_count": sum(1 for row in rows if row["behavior_candidate"] is True),
        "smoke_behavior_candidate_count": sum(1 for row in rows if row["smoke"]["behavior_candidate"] is True),
        "hidden_state_allowed_count": sum(1 for row in rows if row["hidden_state_allowed"] is True),
        "promotion_ready_count": sum(
            1
            for row in rows
            if row["behavior_ready"] is True or row["hidden_state_allowed"] is True
        ),
        "primary_model": first_run_pack["summary"]["primary_model"],
        "secondary_model": first_run_pack["summary"]["secondary_model"],
    }


def build_validation_checks(
    rows: list[dict[str, Any]], summary: dict[str, Any], first_run_pack: dict[str, Any]
) -> list[dict[str, Any]]:
    row_ids = sorted(row["candidate_id"] for row in rows)
    packet_ids = sorted(packet["candidate_id"] for packet in first_run_pack["first_run_packets"])
    result_path_failures = []
    for row in rows:
        for key, rel_path in row["artifact_paths"].items():
            if rel_path is None:
                continue
            if not (ROOT / rel_path).exists():
                result_path_failures.append([row["candidate_id"], key, rel_path])
    expected_diagnostic_failures = [
        [row["candidate_id"], row["final_diagnostic_class"]]
        for row, config in zip(rows, OUTCOME_CONFIGS)
        if row["final_diagnostic_class"] != config["expected_diagnostic_class"]
    ]
    structural_failures = [
        row["candidate_id"]
        for row in rows
        if row["structural"]["structural_passed"] is not True
        or row["structural"]["record_count"] <= 0
        or row["structural"]["source_count"] != 40
    ]
    missing_failure_axes = [
        row["candidate_id"]
        for row in rows
        if not row["failure_axes"] or not row["survived_controls"] or not row["failed_gates"]
    ]
    return [
        {
            "id": "covers_all_first_run_packets",
            "predicate": "outcome candidate ids == first-run packet candidate ids",
            "actual": {"outcome_candidate_ids": row_ids, "packet_candidate_ids": packet_ids},
            "passed": row_ids == packet_ids and summary["outcome_count"] == 6,
            "why": "The outcome matrix must cover every executed KSQ first-run packet exactly once.",
        },
        {
            "id": "all_artifact_paths_exist",
            "predicate": "empty list",
            "actual": result_path_failures,
            "passed": not result_path_failures,
            "why": "Each outcome row must point to concrete runner, prereg, status, and result artifacts.",
        },
        {
            "id": "all_structural_gates_passed",
            "predicate": "empty list and structural_passed_count == 6",
            "actual": {
                "structural_failures": structural_failures,
                "structural_passed_count": summary["structural_passed_count"],
            },
            "passed": not structural_failures and summary["structural_passed_count"] == 6,
            "why": "The first executed knowledge packet result is that structural construction is solved for all six candidates.",
        },
        {
            "id": "no_behavior_ready_or_hidden_state_allowed_rows",
            "predicate": "behavior_ready_count == 0 and hidden_state_allowed_count == 0",
            "actual": {
                "behavior_ready_count": summary["behavior_ready_count"],
                "hidden_state_allowed_count": summary["hidden_state_allowed_count"],
                "promotion_ready_count": summary["promotion_ready_count"],
            },
            "passed": summary["behavior_ready_count"] == 0
            and summary["hidden_state_allowed_count"] == 0
            and summary["promotion_ready_count"] == 0,
            "why": "The KSQ outcome layer must not accidentally promote hidden-state work.",
        },
        {
            "id": "expected_diagnostics_preserved",
            "predicate": "empty list",
            "actual": expected_diagnostic_failures,
            "passed": not expected_diagnostic_failures,
            "why": "The typed diagnostic classes are the point of the outcome matrix.",
        },
        {
            "id": "three_full_and_three_smoke_terminal_gates",
            "predicate": "full_behavior_gate == 3 and smoke_behavior_gate == 3",
            "actual": summary["terminal_gate_counts"],
            "passed": summary["terminal_gate_counts"]
            == {"full_behavior_gate": 3, "smoke_behavior_gate": 3},
            "why": "Three candidates reached full behavior before failure; three died at smoke behavior.",
        },
        {
            "id": "two_outcomes_per_knowledge_level",
            "predicate": "all level counts == 2",
            "actual": summary["level_counts"],
            "passed": all(count == 2 for count in summary["level_counts"].values()),
            "why": "The first-run pack executed two familiar, two bridge, and two uncertainty candidates.",
        },
        {
            "id": "diagnostic_boundaries_are_named",
            "predicate": "empty list",
            "actual": missing_failure_axes,
            "passed": not missing_failure_axes,
            "why": "Every failed candidate should export a reusable diagnostic boundary, not only a negative result.",
        },
    ]


def build_control_surface_knowledge_first_run_outcomes() -> dict[str, Any]:
    first_run_pack = load_json(KNOWLEDGE_FIRST_RUN_PACK_PATH)
    rows = build_outcome_rows(first_run_pack)
    summary = build_summary(rows, first_run_pack)
    payload = {
        "schema_version": 1,
        "updated_at": first_run_pack.get("updated_at"),
        "purpose": (
            "Turn the six executed KSQ first-run result artifacts into a "
            "comparable outcome matrix: structural status, terminal gate, "
            "surviving controls, failed gates, exported diagnostics, and hidden-"
            "state license boundary."
        ),
        "sources": {
            "knowledge_first_run_pack": rel(KNOWLEDGE_FIRST_RUN_PACK_PATH),
            **{
                f"{config['candidate_id']}_{key}": rel(path)
                for config in OUTCOME_CONFIGS
                for key, path in [
                    ("first_run_result", result_path(config, "first_run_result")),
                    ("smoke_result", result_path(config, "smoke_result")),
                    ("full_behavior_result", result_path(config, "full_behavior_result")),
                ]
                if path is not None
            },
        },
        "summary": summary,
        "outcome_rows": rows,
        "validation_checks": build_validation_checks(rows, summary, first_run_pack),
        "allowed_claim": (
            "All six knowledge first-run packets have been executed through at "
            "least smoke behavior; all six passed structural construction; none "
            "became behavior-ready or licensed hidden-state work. The reusable "
            "finding is the distribution of typed behavior-gate failures."
        ),
        "forbidden_claim": (
            "The KSQ outcome matrix does not promote a mechanism, does not claim "
            "a knowledge vector, and does not license probing, steering, editing, "
            "or surgery on any KSQ row."
        ),
    }
    return payload


def validate_knowledge_first_run_outcomes(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("knowledge first-run outcomes schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"knowledge first-run outcome source missing: {rel_path}")
    summary = payload["summary"]
    if summary["outcome_count"] != 6:
        raise AssertionError("knowledge first-run outcomes expected six rows")
    if summary["structural_passed_count"] != 6:
        raise AssertionError("knowledge first-run outcomes expected six structural passes")
    if summary["behavior_ready_count"] != 0:
        raise AssertionError("knowledge first-run outcomes must not be behavior ready")
    if summary["hidden_state_allowed_count"] != 0:
        raise AssertionError("knowledge first-run outcomes must not license hidden states")
    if summary["terminal_gate_counts"] != {
        "full_behavior_gate": 3,
        "smoke_behavior_gate": 3,
    }:
        raise AssertionError("knowledge first-run outcomes terminal gate counts changed")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"knowledge first-run outcome checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Knowledge First-Run Outcomes",
        "",
        f"Source updated_at: {payload['updated_at']}",
        "",
        "Status: generated executed-outcome matrix implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_knowledge_first_run_outcomes.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_first_run_outcomes.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_first_run_outcomes.py --write",
        "python code\\control_surface_knowledge_first_run_outcomes.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Summary",
        "",
        f"- outcomes: `{summary['outcome_count']}`",
        f"- structural passes: `{summary['structural_passed_count']}`",
        f"- full-behavior terminal gates: `{summary['terminal_gate_counts'].get('full_behavior_gate', 0)}`",
        f"- smoke-behavior terminal gates: `{summary['terminal_gate_counts'].get('smoke_behavior_gate', 0)}`",
        f"- behavior-ready rows: `{summary['behavior_ready_count']}`",
        f"- hidden-state-allowed rows: `{summary['hidden_state_allowed_count']}`",
        f"- exported diagnostics: `{format_value(summary['exported_diagnostic_class_counts'])}`",
        "",
        "## Outcome Matrix",
        "",
        "| Candidate | Level | Terminal Gate | Diagnostic | Selected Template | Failed Axes | Hidden State |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["outcome_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate_id']}`",
                    f"`{row['level_id']}`",
                    f"`{row['terminal_gate']}`",
                    f"`{row['exported_diagnostic_class']}`",
                    f"`{row['final_selected_template']}`",
                    format_value(row["failure_axes"]),
                    format_value(row["hidden_state_allowed"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Row Boundaries",
            "",
        ]
    )
    for row in payload["outcome_rows"]:
        lines.extend(
            [
                f"### `{row['candidate_id']}`",
                "",
                f"- final diagnostic: `{row['final_diagnostic_class']}`",
                f"- exported diagnostic: `{row['exported_diagnostic_class']}`",
                f"- terminal gate: `{row['terminal_gate']}`",
                f"- final selected template: `{row['final_selected_template']}`",
                f"- survived controls: `{format_value(row['survived_controls'])}`",
                f"- failed gates: `{format_value(row['failed_gates'])}`",
                f"- final failed criteria: `{format_value(row['final_failed_criteria'])}`",
                f"- allowed claim: {row['allowed_claim']}",
                f"- forbidden claim: {row['forbidden_claim']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Validation Checks",
            "",
            "| Check | Passed |",
            "| --- | --- |",
        ]
    )
    for check in payload["validation_checks"]:
        lines.append(f"| `{check['id']}` | `{str(check['passed']).lower()}` |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            payload["allowed_claim"],
            "",
            payload["forbidden_claim"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = build_control_surface_knowledge_first_run_outcomes()
    validate_knowledge_first_run_outcomes(payload)
    if args.write:
        write_json(KNOWLEDGE_FIRST_RUN_OUTCOMES_PATH, payload)
        KNOWLEDGE_FIRST_RUN_OUTCOMES_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_FIRST_RUN_OUTCOMES_REPORT_PATH.write_text(
            render_markdown(payload), encoding="utf-8", newline="\n"
        )
    print(
        json.dumps(
            {
                "outcome_count": payload["summary"]["outcome_count"],
                "structural_passed_count": payload["summary"]["structural_passed_count"],
                "terminal_gate_counts": payload["summary"]["terminal_gate_counts"],
                "behavior_ready_count": payload["summary"]["behavior_ready_count"],
                "hidden_state_allowed_count": payload["summary"]["hidden_state_allowed_count"],
                "exported_diagnostics": payload["summary"][
                    "exported_diagnostic_class_counts"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
