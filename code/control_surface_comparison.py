"""Build a cross-family comparison layer for the control-surface atlas.

The artifact index normalizes facts per result artifact. This module aggregates
those facts across atlas rows so the project can measure the genome-level
mixture: verdicts, lead-time states, intervention states, surface values,
failure classes, null-boundary signals, and artifact support.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from control_surface_artifacts import ARTIFACT_INDEX_PATH, ATLAS_PATH, ROOT, load_json


COMPARISON_PATH = ROOT / "data" / "control_surface_comparison.json"

STRONG_SURFACE_VALUES = {"high", "dominant", "behavior_only", "bounded", "failed"}
NULL_KEYWORDS = ("absent", "null", "unknown")
CRITERIA_PREFIXES = ("criteria.", "causal_criteria.", "mechanism_criteria.")


def count_values(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def ratio(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 6)


def collect_artifacts_by_row(artifact_index: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifact_index.get("artifacts", []):
        row_id = artifact.get("row_id")
        if row_id:
            by_row[row_id].append(artifact)
    return dict(by_row)


def artifact_metric_truths(artifacts: list[dict[str, Any]], prefix: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for artifact in artifacts:
        metrics = artifact.get("metrics", {})
        for key, value in metrics.items():
            if key.startswith(prefix) and value is True:
                counts[key] += 1
    return dict(sorted(counts.items()))


def artifact_metric_truths_any(
    artifacts: list[dict[str, Any]],
    prefixes: tuple[str, ...],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for artifact in artifacts:
        metrics = artifact.get("metrics", {})
        for key, value in metrics.items():
            if key.startswith(prefixes) and value is True:
                counts[key] += 1
    return dict(sorted(counts.items()))


def artifact_metric_failures(artifacts: list[dict[str, Any]], prefix: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for artifact in artifacts:
        metrics = artifact.get("metrics", {})
        for key, value in metrics.items():
            if key.startswith(prefix) and value is False:
                counts[key] += 1
    return dict(sorted(counts.items()))


def artifact_metric_failures_any(
    artifacts: list[dict[str, Any]],
    prefixes: tuple[str, ...],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for artifact in artifacts:
        metrics = artifact.get("metrics", {})
        for key, value in metrics.items():
            if key.startswith(prefixes) and value is False:
                counts[key] += 1
    return dict(sorted(counts.items()))


def null_boundary_summary(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    failed_null_criteria: Counter[str] = Counter()
    null_metric_values: dict[str, list[Any]] = defaultdict(list)
    artifacts_with_null_criteria = 0
    artifacts_with_failed_nulls = 0

    for artifact in artifacts:
        null_criteria = artifact.get("null_criteria", [])
        if null_criteria:
            artifacts_with_null_criteria += 1
        failed = [
            criterion
            for criterion in artifact.get("failed_criteria", [])
            if any(keyword in criterion.lower() for keyword in NULL_KEYWORDS)
        ]
        if failed:
            artifacts_with_failed_nulls += 1
        failed_null_criteria.update(failed)

        for key, value in artifact.get("metrics", {}).items():
            key_lower = key.lower()
            if any(keyword in key_lower for keyword in NULL_KEYWORDS):
                if isinstance(value, (str, int, float, bool)) or value is None:
                    null_metric_values[key].append(value)

    compact_metric_values: dict[str, Any] = {}
    for key, values in sorted(null_metric_values.items()):
        unique = []
        for value in values:
            if value not in unique:
                unique.append(value)
        compact_metric_values[key] = unique[:10]

    return {
        "artifacts_with_null_criteria": artifacts_with_null_criteria,
        "artifacts_with_failed_nulls": artifacts_with_failed_nulls,
        "failed_null_criteria": dict(sorted(failed_null_criteria.items())),
        "null_metric_values": compact_metric_values,
    }


def row_artifact_summary(row_id: str, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostic_counts = count_values(
        [
            artifact["diagnostic_class"]
            for artifact in artifacts
            if artifact.get("diagnostic_class")
        ]
    )
    failed_criteria = Counter()
    null_criteria = Counter()
    for artifact in artifacts:
        failed_criteria.update(artifact.get("failed_criteria", []))
        null_criteria.update(artifact.get("null_criteria", []))

    readiness_counts = {
        "behavior_ready_true": sum(1 for artifact in artifacts if artifact.get("behavior_ready") is True),
        "behavior_ready_false": sum(1 for artifact in artifacts if artifact.get("behavior_ready") is False),
        "signature_ready_true": sum(1 for artifact in artifacts if artifact.get("signature_ready") is True),
        "signature_ready_false": sum(1 for artifact in artifacts if artifact.get("signature_ready") is False),
        "intervention_ready_true": sum(1 for artifact in artifacts if artifact.get("intervention_ready") is True),
        "intervention_ready_false": sum(1 for artifact in artifacts if artifact.get("intervention_ready") is False),
    }

    return {
        "row_id": row_id,
        "artifact_count": len(artifacts),
        "diagnostic_counts": diagnostic_counts,
        "readiness_counts": readiness_counts,
        "failed_criteria": dict(sorted(failed_criteria.items())),
        "null_criteria": dict(sorted(null_criteria.items())),
        "true_criteria": artifact_metric_truths_any(artifacts, CRITERIA_PREFIXES),
        "false_criteria": artifact_metric_failures_any(artifacts, CRITERIA_PREFIXES),
    }


def collect_family_checks_by_row(artifact_index: dict[str, Any]) -> dict[str, list[str]]:
    by_row: dict[str, list[str]] = defaultdict(list)
    for check_name, row_id in artifact_index["report"].get(
        "family_claim_check_rows",
        {},
    ).items():
        by_row[row_id].append(check_name)
    return {row_id: sorted(checks) for row_id, checks in by_row.items()}


def audit_level(
    artifact_count: int,
    metric_artifact_count: int,
    family_check_count: int,
) -> str:
    if artifact_count == 0:
        return "prose_only"
    if metric_artifact_count == 0:
        return "artifact_linked_no_metrics"
    if family_check_count == 0:
        return "artifact_indexed_no_family_check"
    if metric_artifact_count < artifact_count:
        return "family_checked_partial_metrics"
    return "family_checked_with_metrics"


def row_claim_audit(
    row: dict[str, Any],
    artifacts: list[dict[str, Any]],
    family_checks: list[str],
) -> dict[str, Any]:
    metric_artifact_count = sum(1 for artifact in artifacts if artifact.get("metrics"))
    false_criteria_count = sum(
        1
        for artifact in artifacts
        for key, value in artifact.get("metrics", {}).items()
        if key.startswith(CRITERIA_PREFIXES) and value is False
    )
    failed_criteria_count = sum(
        len(artifact.get("failed_criteria", [])) for artifact in artifacts
    )
    null_criteria_count = sum(
        len(artifact.get("null_criteria", [])) for artifact in artifacts
    )
    baseline_field_count = sum(
        artifact.get("baseline_field_count", 0) for artifact in artifacts
    )
    intervention_field_count = sum(
        artifact.get("intervention_field_count", 0) for artifact in artifacts
    )
    readiness_flag_count = sum(
        1
        for artifact in artifacts
        for field in ["behavior_ready", "signature_ready", "intervention_ready"]
        if artifact.get(field) is not None
    )
    readiness_counts = {
        "behavior_ready_true": sum(1 for artifact in artifacts if artifact.get("behavior_ready") is True),
        "behavior_ready_false": sum(1 for artifact in artifacts if artifact.get("behavior_ready") is False),
        "signature_ready_true": sum(1 for artifact in artifacts if artifact.get("signature_ready") is True),
        "signature_ready_false": sum(1 for artifact in artifacts if artifact.get("signature_ready") is False),
        "intervention_ready_true": sum(1 for artifact in artifacts if artifact.get("intervention_ready") is True),
        "intervention_ready_false": sum(1 for artifact in artifacts if artifact.get("intervention_ready") is False),
    }
    level = audit_level(len(artifacts), metric_artifact_count, len(family_checks))

    gaps: list[str] = []
    if not artifacts:
        gaps.append("NO_RESULT_ARTIFACT")
    if artifacts and metric_artifact_count == 0:
        gaps.append("NO_EXTRACTED_METRICS")
    if not family_checks:
        gaps.append("NO_FAMILY_CLAIM_CHECK")
    if row["verdict"]["class"] != "diagnostic_note" and intervention_field_count == 0:
        gaps.append("NON_DIAGNOSTIC_WITHOUT_INTERVENTION_FIELDS")

    return {
        "row_id": row["id"],
        "verdict": row["verdict"]["class"],
        "lead_time": row["lead_time"]["state"],
        "intervention": row["intervention"]["state"],
        "artifact_count": len(artifacts),
        "metric_artifact_count": metric_artifact_count,
        "family_claim_checks": family_checks,
        "failed_criteria_count": failed_criteria_count,
        "false_criteria_count": false_criteria_count,
        "null_criteria_count": null_criteria_count,
        "baseline_field_count": baseline_field_count,
        "intervention_field_count": intervention_field_count,
        "readiness_flag_count": readiness_flag_count,
        "readiness_counts": readiness_counts,
        "audit_level": level,
        "gaps": gaps,
    }


def build_claim_audit(
    rows: list[dict[str, Any]],
    artifacts_by_row: dict[str, list[dict[str, Any]]],
    artifact_index: dict[str, Any],
) -> dict[str, Any]:
    checks_by_row = collect_family_checks_by_row(artifact_index)
    row_audits = [
        row_claim_audit(
            row,
            artifacts_by_row.get(row["id"], []),
            checks_by_row.get(row["id"], []),
        )
        for row in rows
    ]
    audit_level_counts = count_values([audit["audit_level"] for audit in row_audits])
    rows_without_artifacts = [
        audit["row_id"] for audit in row_audits if audit["artifact_count"] == 0
    ]
    rows_without_metrics = [
        audit["row_id"] for audit in row_audits if audit["metric_artifact_count"] == 0
    ]
    rows_without_family_checks = [
        audit["row_id"] for audit in row_audits if not audit["family_claim_checks"]
    ]
    rows_with_partial_metrics = [
        audit["row_id"]
        for audit in row_audits
        if 0 < audit["metric_artifact_count"] < audit["artifact_count"]
    ]
    rows_with_failed_or_false_criteria = [
        audit["row_id"]
        for audit in row_audits
        if audit["failed_criteria_count"] > 0 or audit["false_criteria_count"] > 0
    ]

    return {
        "summary": {
            "row_count": len(row_audits),
            "rows_with_artifacts": len(row_audits) - len(rows_without_artifacts),
            "rows_with_metrics": len(row_audits) - len(rows_without_metrics),
            "rows_with_family_claim_checks": (
                len(row_audits) - len(rows_without_family_checks)
            ),
            "rows_with_partial_metrics": len(rows_with_partial_metrics),
            "rows_with_failed_or_false_criteria": len(rows_with_failed_or_false_criteria),
            "audit_level_counts": audit_level_counts,
            "rows_without_artifacts": rows_without_artifacts,
            "rows_without_metrics": rows_without_metrics,
            "rows_without_family_checks": rows_without_family_checks,
            "rows_with_partial_metrics": rows_with_partial_metrics,
            "rows_with_failed_or_false_criteria": rows_with_failed_or_false_criteria,
        },
        "rows": row_audits,
    }


def row_claim_consistency(row: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    verdict = row["verdict"]["class"]
    intervention = row["intervention"]["state"]
    lead_time = row["lead_time"]["state"]
    diagnostics = set(row.get("diagnostics", []))
    mixture = row.get("mixture_profile", {})
    readiness = audit.get("readiness_counts", {})

    contradictions: list[str] = []
    checked: list[str] = []

    def require(condition: bool, code: str) -> None:
        checked.append(code)
        if not condition:
            contradictions.append(code)

    has_output_confounded_diagnostic = bool(
        diagnostics
        & {
            "OUTPUT_MARGIN_CONFUND",
            "GLOBAL_OUTPUT_CONFOUNDED_LEADTIME",
            "SOURCE_PATH_FINAL_MARGIN_SHADOW",
            "GREEDY_FINAL_MARGIN_SIGN_BARRIER",
            "DELAYED_INTERFACE_CANDIDATE_SCORE_VISIBLE",
        }
    )
    has_signature_failure_diagnostic = bool(
        diagnostics
        & {
            "SHUFFLED_SELECTION_OVERFIT",
            "SIGNATURE_NOT_CAUSAL",
            "CANDIDATE_DECOUPLED_SHUFFLE_OVERFIT",
            "LOCKED_COORDINATE_TRANSFER_FAILED",
            "EXPANDED_CANDIDATE_DECOUPLED_BANK_INSUFFICIENT",
            "TRANSFER_ROLE_REPAIR_BANK_INSUFFICIENT",
        }
    )
    has_null_failure_diagnostic = any("NULL" in diagnostic for diagnostic in diagnostics)

    if verdict == "promoted_mechanism_card":
        require(
            intervention not in {"not_allowed", "not_tested", "failed"},
            "PROMOTED_CARD_REQUIRES_ACTIVE_INTERVENTION_STATE",
        )
        require(
            audit["false_criteria_count"] == 0 and audit["failed_criteria_count"] == 0,
            "PROMOTED_CARD_REQUIRES_NO_FAILED_EXTRACTED_CRITERIA",
        )
        require(
            audit["intervention_field_count"] > 0,
            "PROMOTED_CARD_REQUIRES_INTERVENTION_FIELDS",
        )

    if verdict == "bounded_mechanism_card":
        require(
            intervention not in {"not_allowed", "not_tested", "failed"},
            "BOUNDED_CARD_REQUIRES_NONBLOCKED_INTERVENTION_STATE",
        )
        require(
            audit["intervention_field_count"] > 0,
            "BOUNDED_CARD_REQUIRES_INTERVENTION_FIELDS",
        )
        require(
            audit["false_criteria_count"] > 0
            or audit["failed_criteria_count"] > 0
            or audit["null_criteria_count"] > 0,
            "BOUNDED_CARD_REQUIRES_EXTRACTED_BOUNDARY",
        )

    if verdict == "failed_mechanism_card":
        require(
            intervention == "failed",
            "FAILED_CARD_REQUIRES_FAILED_INTERVENTION_STATE",
        )
        require(
            audit["intervention_field_count"] > 0
            or audit["false_criteria_count"] > 0
            or bool(audit["family_claim_checks"]),
            "FAILED_CARD_REQUIRES_FAILURE_EVIDENCE",
        )

    if intervention == "not_allowed":
        require(lead_time == "not_reached", "NOT_ALLOWED_REQUIRES_LEAD_TIME_NOT_REACHED")
        require(
            readiness.get("signature_ready_true", 0) == 0,
            "NOT_ALLOWED_REQUIRES_NO_SIGNATURE_READY_ARTIFACTS",
        )
        require(
            readiness.get("intervention_ready_true", 0) == 0,
            "NOT_ALLOWED_REQUIRES_NO_INTERVENTION_READY_ARTIFACTS",
        )

    if intervention == "not_tested":
        require(
            audit["intervention_field_count"] == 0,
            "NOT_TESTED_REQUIRES_NO_INTERVENTION_FIELDS",
        )

    if intervention == "failed":
        require(
            audit["intervention_field_count"] > 0,
            "FAILED_INTERVENTION_REQUIRES_INTERVENTION_FIELDS",
        )
        require(
            audit["false_criteria_count"] > 0
            or audit["failed_criteria_count"] > 0
            or bool(audit["family_claim_checks"]),
            "FAILED_INTERVENTION_REQUIRES_FAILURE_CRITERIA_OR_CHECK",
        )

    if intervention == "causal_dirty":
        require(
            audit["intervention_field_count"] > 0,
            "CAUSAL_DIRTY_REQUIRES_INTERVENTION_FIELDS",
        )
        require(
            audit["false_criteria_count"] > 0
            or audit["failed_criteria_count"] > 0
            or audit["null_criteria_count"] > 0,
            "CAUSAL_DIRTY_REQUIRES_BOUNDARY_CRITERIA",
        )

    if intervention == "behavior_control_only":
        require(
            audit["intervention_field_count"] > 0,
            "BEHAVIOR_CONTROL_REQUIRES_CONTROL_FIELDS",
        )

    if lead_time == "not_reached":
        require(
            intervention == "not_allowed",
            "LEAD_TIME_NOT_REACHED_REQUIRES_INTERVENTION_NOT_ALLOWED",
        )
        require(
            readiness.get("signature_ready_true", 0) == 0,
            "LEAD_TIME_NOT_REACHED_REQUIRES_NO_SIGNATURE_READY_ARTIFACTS",
        )

    if lead_time in {"lead_time_monitor_only", "lead_time_output_shadow"}:
        require(
            verdict != "promoted_mechanism_card",
            "MONITOR_OR_SHADOW_LEAD_TIME_FORBIDS_PROMOTION",
        )
        require(
            has_output_confounded_diagnostic
            or has_signature_failure_diagnostic
            or audit["false_criteria_count"] > 0
            or audit["failed_criteria_count"] > 0,
            "MONITOR_OR_SHADOW_LEAD_TIME_REQUIRES_CONFOUND_OR_FAILED_CRITERIA",
        )

    if lead_time == "zero_lead_time":
        require(
            readiness.get("signature_ready_true", 0) == 0,
            "ZERO_LEAD_TIME_REQUIRES_NO_SIGNATURE_READY_ARTIFACTS",
        )

    if "BEHAVIOR_SUBSTRATE_FAILED" in diagnostics:
        require(
            verdict == "diagnostic_note",
            "BEHAVIOR_SUBSTRATE_FAILED_REQUIRES_DIAGNOSTIC_VERDICT",
        )
        require(
            intervention == "not_allowed",
            "BEHAVIOR_SUBSTRATE_FAILED_REQUIRES_INTERVENTION_NOT_ALLOWED",
        )
        require(
            lead_time == "not_reached",
            "BEHAVIOR_SUBSTRATE_FAILED_REQUIRES_LEAD_TIME_NOT_REACHED",
        )

    if has_output_confounded_diagnostic:
        require(
            audit["baseline_field_count"] > 0,
            "OUTPUT_CONFOUND_DIAGNOSTIC_REQUIRES_BASELINE_FIELDS",
        )

    if "SIGNATURE_NOT_CAUSAL" in diagnostics:
        require(
            verdict != "promoted_mechanism_card",
            "SIGNATURE_NOT_CAUSAL_FORBIDS_PROMOTION",
        )

    if has_null_failure_diagnostic or mixture.get("null_locality") == "failed":
        require(
            audit["null_criteria_count"] > 0,
            "NULL_FAILURE_REQUIRES_EXTRACTED_NULL_CRITERIA",
        )

    if mixture.get("null_locality") == "failed":
        require(
            audit["false_criteria_count"] > 0 or audit["failed_criteria_count"] > 0,
            "NULL_LOCALITY_FAILED_REQUIRES_FAILED_CRITERIA",
        )

    return {
        "row_id": row["id"],
        "checked_count": len(checked),
        "contradiction_count": len(contradictions),
        "contradictions": contradictions,
    }


def build_claim_consistency(
    rows: list[dict[str, Any]],
    claim_audit: dict[str, Any],
) -> dict[str, Any]:
    audits_by_row = {audit["row_id"]: audit for audit in claim_audit["rows"]}
    row_consistency = [
        row_claim_consistency(row, audits_by_row[row["id"]])
        for row in rows
    ]
    rows_with_contradictions = [
        row["row_id"] for row in row_consistency if row["contradiction_count"] > 0
    ]
    contradictions = [
        {
            "row_id": row["row_id"],
            "code": code,
        }
        for row in row_consistency
        for code in row["contradictions"]
    ]
    return {
        "summary": {
            "row_count": len(row_consistency),
            "checked_condition_count": sum(row["checked_count"] for row in row_consistency),
            "contradiction_count": len(contradictions),
            "rows_with_contradictions": rows_with_contradictions,
            "contradictions": contradictions,
        },
        "rows": row_consistency,
    }


def build_control_surface_comparison(
    atlas: dict[str, Any],
    artifact_index: dict[str, Any],
) -> dict[str, Any]:
    rows = atlas["rows"]
    row_count = len(rows)
    artifacts_by_row = collect_artifacts_by_row(artifact_index)

    verdict_counts = count_values([row["verdict"]["class"] for row in rows])
    lead_time_counts = count_values([row["lead_time"]["state"] for row in rows])
    intervention_counts = count_values([row["intervention"]["state"] for row in rows])
    behavior_gate_counts = count_values([row["behavior_gate"] for row in rows])

    diagnostic_counts = Counter()
    for row in rows:
        diagnostic_counts.update(row["diagnostics"])

    mixture_axis_counts: dict[str, Any] = {}
    mixture_axes = sorted(rows[0]["mixture_profile"]) if rows else []
    for axis in mixture_axes:
        values = [row["mixture_profile"][axis] for row in rows]
        counts = count_values(values)
        strong_count = sum(1 for value in values if value in STRONG_SURFACE_VALUES)
        mixture_axis_counts[axis] = {
            "counts": counts,
            "strong_or_failed_count": strong_count,
            "strong_or_failed_ratio": ratio(strong_count, row_count),
        }

    artifact_rows = sorted(artifacts_by_row)
    row_summaries = [
        {
            "id": row["id"],
            "family": row["family"],
            "verdict": row["verdict"]["class"],
            "lead_time": row["lead_time"]["state"],
            "intervention": row["intervention"]["state"],
            "behavior_gate": row["behavior_gate"],
            "mixture_profile": row["mixture_profile"],
            "diagnostics": row["diagnostics"],
            "artifact_count": len(artifacts_by_row.get(row["id"], [])),
        }
        for row in rows
    ]

    all_artifacts = artifact_index.get("artifacts", [])
    artifact_diagnostic_counts = count_values(
        [
            artifact["diagnostic_class"]
            for artifact in all_artifacts
            if artifact.get("diagnostic_class")
        ]
    )
    claim_audit = build_claim_audit(rows, artifacts_by_row, artifact_index)

    return {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "atlas_ref": str(ATLAS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "artifact_index_ref": str(ARTIFACT_INDEX_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source": "code/control_surface_comparison.py",
        "purpose": (
            "Cross-family comparison of atlas rows and normalized artifact facts. "
            "This measures the distribution of control surfaces, not mechanism-card promotion."
        ),
        "report": {
            "row_count": row_count,
            "artifact_count": artifact_index["report"]["artifact_count"],
            "artifact_rows": artifact_rows,
            "rows_without_artifacts": [
                row["id"] for row in rows if row["id"] not in artifacts_by_row
            ],
            "verdict_counts": verdict_counts,
            "lead_time_counts": lead_time_counts,
            "intervention_counts": intervention_counts,
            "behavior_gate_counts": behavior_gate_counts,
            "row_diagnostic_counts": dict(sorted(diagnostic_counts.items())),
            "artifact_diagnostic_counts": artifact_diagnostic_counts,
        },
        "mixture_axis_counts": mixture_axis_counts,
        "null_boundary_summary": null_boundary_summary(all_artifacts),
        "claim_audit": claim_audit,
        "claim_consistency": build_claim_consistency(rows, claim_audit),
        "artifact_row_summaries": [
            row_artifact_summary(row_id, artifacts_by_row[row_id])
            for row_id in sorted(artifacts_by_row)
        ],
        "row_summaries": row_summaries,
        "current_genome_shape": {
            "promoted_mechanism_ratio": ratio(
                verdict_counts.get("promoted_mechanism_card", 0),
                row_count,
            ),
            "bounded_mechanism_ratio": ratio(
                verdict_counts.get("bounded_mechanism_card", 0),
                row_count,
            ),
            "diagnostic_or_failed_ratio": ratio(
                verdict_counts.get("diagnostic_note", 0)
                + verdict_counts.get("failed_mechanism_card", 0),
                row_count,
            ),
            "lead_time_monitor_or_shadow_ratio": ratio(
                lead_time_counts.get("lead_time_monitor_only", 0)
                + lead_time_counts.get("lead_time_output_shadow", 0),
                row_count,
            ),
            "intervention_not_allowed_or_failed_ratio": ratio(
                intervention_counts.get("not_allowed", 0)
                + intervention_counts.get("failed", 0),
                row_count,
            ),
            "output_margin_confound_row_ratio": ratio(
                diagnostic_counts.get("OUTPUT_MARGIN_CONFUND", 0),
                row_count,
            ),
            "behavior_substrate_failed_row_ratio": ratio(
                diagnostic_counts.get("BEHAVIOR_SUBSTRATE_FAILED", 0),
                row_count,
            ),
        },
    }


def write_comparison(
    atlas: dict[str, Any],
    artifact_index: dict[str, Any],
    output_path: Path = COMPARISON_PATH,
) -> dict[str, Any]:
    comparison = build_control_surface_comparison(atlas, artifact_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write data/control_surface_comparison.json")
    parser.add_argument("--json", action="store_true", help="print comparison JSON")
    args = parser.parse_args()

    atlas = load_json(ATLAS_PATH)
    artifact_index = load_json(ARTIFACT_INDEX_PATH)
    comparison = build_control_surface_comparison(atlas, artifact_index)

    if args.write:
        write_comparison(atlas, artifact_index)
        print(
            f"wrote {COMPARISON_PATH.relative_to(ROOT).as_posix()} "
            f"with {comparison['report']['row_count']} rows"
        )
        return
    if args.json:
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return

    print(f"comparison ok: {comparison['report']['row_count']} rows")
    print("verdict_counts:", json.dumps(comparison["report"]["verdict_counts"], sort_keys=True))
    print("lead_time_counts:", json.dumps(comparison["report"]["lead_time_counts"], sort_keys=True))
    print("intervention_counts:", json.dumps(comparison["report"]["intervention_counts"], sort_keys=True))
    print("current_genome_shape:", json.dumps(comparison["current_genome_shape"], sort_keys=True))
    print("claim_audit:", json.dumps(comparison["claim_audit"]["summary"], sort_keys=True))
    print(
        "claim_consistency:",
        json.dumps(comparison["claim_consistency"]["summary"], sort_keys=True),
    )


if __name__ == "__main__":
    main()
