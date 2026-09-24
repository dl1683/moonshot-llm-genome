"""Build the current coverage-gap map for the control-surface genome.

The genome snapshot and axis-interaction map state what we know. This layer
states what is still undercovered: missing mechanisms, singleton laws,
transfer gaps, model-family gaps, bridge closures, and intervention gaps. It
is the negative-space map for deciding what the atlas still cannot claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json


COVERAGE_GAPS_PATH = ROOT / "data" / "control_surface_coverage_gaps.json"
COVERAGE_GAPS_REPORT_PATH = ROOT / "research" / "37_CONTROL_SURFACE_COVERAGE_GAPS.md"

AXIS_INTERACTIONS_PATH = ROOT / "data" / "control_surface_axis_interactions.json"
BRIDGE_LADDER_PATH = ROOT / "data" / "control_surface_bridge_ladder.json"
GENOME_SNAPSHOT_PATH = ROOT / "data" / "control_surface_genome_snapshot.json"
NEXT_QUEUE_PATH = ROOT / "data" / "control_surface_next_experiment_queue.json"
RELIABILITY_MATRIX_PATH = ROOT / "data" / "control_surface_reliability_matrix.json"
TRANSFER_MATRIX_PATH = ROOT / "data" / "control_surface_transfer_matrix.json"


DOMAIN_PREFIXES = {
    "truth_agreement": ("mc001", "mc001b", "mc001g"),
    "abstention_known_unknown": ("mc002", "mc002b"),
    "delayed_copy": ("mc003",),
    "in_context_binding": ("mc004",),
    "synthetic_lookup": ("mc005",),
    "parametric_fact_override": ("mc006",),
    "knowledge_bridge": (
        "mc007",
        "mc008",
        "mc009",
        "mc010",
        "mc011",
        "mc012",
        "mc013",
        "mc014",
        "mc015",
        "mc016",
    ),
}

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "watch": 3}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def ratio(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 6)


def classify_domain(row_id: str) -> str:
    for domain, prefixes in DOMAIN_PREFIXES.items():
        if any(row_id.startswith(prefix) for prefix in prefixes):
            return domain
    return "other"


def row_ids_by_domain(atlas: dict[str, Any]) -> dict[str, list[str]]:
    domains: dict[str, list[str]] = {}
    for row in atlas["rows"]:
        domains.setdefault(classify_domain(row["id"]), []).append(row["id"])
    return {key: sorted(value) for key, value in sorted(domains.items())}


def model_rows(atlas: dict[str, Any]) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for row in atlas["rows"]:
        for model in row["models"]:
            rows.setdefault(model, []).append(row["id"])
    return {key: sorted(value) for key, value in sorted(rows.items())}


def queue_items_by_reason(queue: dict[str, Any], reason_code: str) -> list[str]:
    return [
        item["id"]
        for item in queue["queue"]
        if reason_code in item.get("reason_codes", [])
    ]


def make_gap(
    gap_id: str,
    severity: str,
    gap_type: str,
    title: str,
    evidence: dict[str, Any],
    why_it_matters: str,
    exit_condition: str,
    next_pressure: list[str],
    affected_rows: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": gap_id,
        "severity": severity,
        "gap_type": gap_type,
        "title": title,
        "evidence": evidence,
        "why_it_matters": why_it_matters,
        "exit_condition": exit_condition,
        "next_pressure": next_pressure,
        "affected_rows": sorted(affected_rows or []),
    }


def build_coverage_gaps(
    atlas: dict[str, Any],
    genome_snapshot: dict[str, Any],
    axis_interactions: dict[str, Any],
    bridge_ladder: dict[str, Any],
    next_queue: dict[str, Any],
    reliability_matrix: dict[str, Any],
    transfer_matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    row_count = len(atlas["rows"])
    gate_shape = genome_snapshot["gate_shape"]
    reliability_summary = reliability_matrix["summary"]
    transfer_summary = transfer_matrix["summary"]
    bridge_summary = bridge_ladder["summary"]
    axis_summary = axis_interactions["summary"]
    domains = row_ids_by_domain(atlas)
    models = model_rows(atlas)
    terminal_stage_counts = gate_shape["terminal_stage_counts"]
    singleton_terminal_stages = [
        stage for stage, count in terminal_stage_counts.items() if count == 1
    ]
    singleton_terminal_rows = [
        row_id
        for stage in singleton_terminal_stages
        for row_id in gate_shape["rows_by_terminal_stage"][stage]
    ]
    sparse_feature_count = (
        axis_summary["evidence_level_counts"].get("singleton", 0)
        + axis_summary["evidence_level_counts"].get("sparse", 0)
    )
    transfer_untested_rows = [
        entry["row_id"]
        for entry in transfer_matrix["transfer_entries"]
        if entry["transfer_value"] == "untested"
    ]
    gaps = [
        make_gap(
            "promoted_mechanism_absent",
            "critical",
            "mechanism_promotion",
            "No promoted mechanism exists in the current atlas.",
            {
                "promoted_mechanism_count": genome_snapshot["summary"][
                    "promoted_mechanism_count"
                ],
                "row_count": row_count,
            },
            "The project cannot yet claim a fully reliable control surface.",
            "At least one row clears signature, intervention, reliability, null/locality, robustness, and transfer gates.",
            next_queue["summary"]["top_queue_ids"],
        ),
        make_gap(
            "full_reliability_absent",
            "critical",
            "reliability",
            "No row clears the full reliability bar.",
            {
                "full_reliability_count": reliability_summary["full_reliability_count"],
                "missing_gate_counts": reliability_summary["missing_gate_counts"],
            },
            "Without a full reliability row, the atlas remains a diagnostic and bounded-mechanism map.",
            "Full reliability count becomes nonzero without contradicting missing-gate validation.",
            queue_items_by_reason(next_queue, "BEHAVIOR_SUBSTRATE_GATE")[:5],
        ),
        make_gap(
            "clean_intervention_absent",
            "critical",
            "intervention",
            "Every current row still misses clean predicted intervention.",
            {
                "clean_predicted_intervention_missing": reliability_summary[
                    "missing_gate_counts"
                ]["clean_predicted_intervention"],
                "intervention_test_missing": reliability_summary["missing_gate_counts"][
                    "intervention_test"
                ],
                "row_count": row_count,
            },
            "A signature atlas is not a control-surface atlas until interventions change behavior cleanly.",
            "A row passes a predicted intervention with documented nulls, locality, side effects, and holdout behavior.",
            queue_items_by_reason(next_queue, "INTERVENTION_RELEVANT")[:5],
        ),
        make_gap(
            "transfer_ready_mechanism_absent",
            "critical",
            "transfer",
            "There are zero transfer-ready mechanisms and most rows are transfer-untested.",
            {
                "transfer_ready_mechanism_count": transfer_summary[
                    "transfer_ready_mechanism_count"
                ],
                "transfer_value_counts": transfer_summary["transfer_value_counts"],
                "multi_model_row_count": transfer_summary["multi_model_row_count"],
            },
            "The current map cannot support broad small-model generalization claims.",
            "A mechanism-like row transfers with matched null panels, side rows, and reliability fields beyond bounded.",
            transfer_summary["transfer_gap_top_queue_ids"],
            transfer_untested_rows,
        ),
        make_gap(
            "bridge_hidden_state_not_licensed",
            "high",
            "bridge_substrate",
            "The bridge ladder still has no hidden-state-allowed or clean unconfounded rungs.",
            {
                "rung_count": bridge_summary["rung_count"],
                "smoke_rung_count": bridge_summary["smoke_rung_count"],
                "hidden_state_allowed_count": bridge_summary["hidden_state_allowed_count"],
                "clean_unconfounded_bridge_count": bridge_summary[
                    "clean_unconfounded_bridge_count"
                ],
                "recent_closed_rungs": next_queue["summary"]["bridge_closure"][
                    "recent_closed_rung_ids"
                ],
            },
            "The knowledge-like bridge is still behavior-substrate work, not mechanism work.",
            "A materially different bridge passes branch, null, local, side-number, prompt-channel, output/candidate, and source-disjoint gates together.",
            queue_items_by_reason(next_queue, "BRIDGE_ROUTE_NEEDED")[:5],
        ),
        make_gap(
            "singleton_terminal_stage_evidence",
            "high",
            "sample_size",
            "Key terminal stages are represented by singleton rows.",
            {
                "singleton_terminal_stages": singleton_terminal_stages,
                "singleton_terminal_rows": singleton_terminal_rows,
            },
            "Singleton stage boundaries can calibrate doctrine, but they cannot support broad terminal-stage laws.",
            "At least three materially distinct rows occupy prompt-channel locality, failed-intervention, and reliability-boundary stages.",
            next_queue["summary"]["top_queue_ids"],
            singleton_terminal_rows,
        ),
        make_gap(
            "axis_rules_sparse_or_singleton_heavy",
            "high",
            "predictive_law_support",
            "Most feature summaries are singleton or sparse evidence.",
            {
                "feature_count": axis_summary["feature_count"],
                "evidence_level_counts": axis_summary["evidence_level_counts"],
                "singleton_plus_sparse_count": sparse_feature_count,
            },
            "The axis-interaction map is useful, but many apparent rules are still small-n boundaries.",
            "Broad-or-supported pure rules increase while singleton/sparse evidence stops dominating feature summaries.",
            queue_items_by_reason(next_queue, "SINGLE_ROW_LAW_SUPPORT")[:5],
        ),
        make_gap(
            "model_family_coverage_qwen_dominant",
            "medium",
            "model_coverage",
            "The atlas is dominated by Qwen rows and has thin Gemma coverage.",
            {
                "model_rows": models,
                "model_row_counts": {model: len(rows) for model, rows in models.items()},
            },
            "A small-LLM genome map needs model-family boundaries, not only model-specific rows.",
            "Each major terminal-stage or mechanism-like claim has at least one materially comparable non-Qwen replication or failure.",
            queue_items_by_reason(next_queue, "TRANSFER_GAP")[:5],
        ),
        make_gap(
            "domain_coverage_knowledge_bridge_dominates_failures",
            "medium",
            "behavior_domain_coverage",
            "Knowledge-bridge rows dominate the behavior-substrate closure bucket.",
            {
                "domain_rows": domains,
                "domain_row_counts": {domain: len(rows) for domain, rows in domains.items()},
                "knowledge_bridge_rows": domains.get("knowledge_bridge", []),
            },
            "The bridge program is valuable, but the genome map also needs non-bridge behavior families at later gates.",
            "At least two non-bridge domains advance beyond singleton evidence at signature, intervention, or reliability stages.",
            next_queue["summary"]["top_queue_ids"],
            domains.get("knowledge_bridge", []),
        ),
        make_gap(
            "output_geometry_pressure_mixed",
            "medium",
            "predictive_law_support",
            "Output geometry is broad but not a single terminal-stage law.",
            {
                "mixed_predictor": [
                    item
                    for item in axis_interactions["mixed_predictors"]
                    if item["feature_key"]
                    == "primary_blocker:primary_blocker=output_geometry_shadow"
                ],
                "output_geometry_visible_count": genome_snapshot["mixture_shape"][
                    "pressure_class_counts"
                ]["output_geometry_visible"],
            },
            "Output geometry is a family-level pressure; the map needs sharper conditions for when it becomes shadow, monitor, or intervention failure.",
            "A future layer separates output-shadow, monitor-only, and failed-intervention cases with broader supported rules.",
            queue_items_by_reason(next_queue, "OUTPUT_GEOMETRY_CONTROL")[:5],
        ),
        make_gap(
            "mc005_singleton_internal_causal_reference",
            "medium",
            "mechanism_reference",
            "MC005 is the only bounded internal-causal reference specimen.",
            {
                "bounded_mechanism_count": genome_snapshot["summary"][
                    "bounded_mechanism_count"
                ],
                "bounded_rows": genome_snapshot["reliability_shape"][
                    "rows_by_reliability_class"
                ].get("bounded_reliability_reference", []),
            },
            "The project has one nearly real mechanism-like specimen, but no distribution of internal-causal surfaces.",
            "A second materially distinct internal-causal bounded or promoted row appears outside MC005.",
            queue_items_by_reason(next_queue, "INTERVENTION_RELEVANT")[:5],
            genome_snapshot["reliability_shape"]["rows_by_reliability_class"].get(
                "bounded_reliability_reference",
                [],
            ),
        ),
        make_gap(
            "mc006_knowledge_route_monitor_only",
            "medium",
            "knowledge_mechanism",
            "The main parametric-fact override branch is monitor-only and not mechanism-grade.",
            {
                "mc006_terminal_stage": genome_snapshot["gate_shape"][
                    "rows_by_terminal_stage"
                ]["signature_monitor_no_lever"],
                "predecision_causal_candidate_count": genome_snapshot["frontier_shape"][
                    "predecision_causal_candidate_count"
                ],
            },
            "This is the closest branch to the knowledge-genome ambition, but it still has no clean causal lever.",
            "A knowledge-like row beats output/candidate baselines and supports a clean intervention.",
            queue_items_by_reason(next_queue, "LEADTIME_FRONTIER")[:5],
            ["mc006_parametric_fact_override"],
        ),
    ]
    gaps.sort(
        key=lambda gap: (
            SEVERITY_ORDER[gap["severity"]],
            gap["gap_type"],
            gap["id"],
        )
    )
    return gaps


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    gap_ids = {gap["id"] for gap in payload["coverage_gaps"]}
    summary = payload["summary"]
    checks = [
        {
            "id": "coverage_gap_count_is_substantial",
            "actual": summary["gap_count"],
            "predicate": ">= 10",
            "passed": summary["gap_count"] >= 10,
            "why": "The coverage layer must expose multiple negative-space boundaries.",
        },
        {
            "id": "critical_mechanism_gaps_present",
            "actual": sorted(gap_ids),
            "predicate": "promoted, reliability, intervention, and transfer gaps exist",
            "passed": {
                "promoted_mechanism_absent",
                "full_reliability_absent",
                "clean_intervention_absent",
                "transfer_ready_mechanism_absent",
            }.issubset(gap_ids),
            "why": "The largest missing claims must stay explicit.",
        },
        {
            "id": "bridge_gap_preserves_zero_hidden_state_allowed",
            "actual": summary["bridge_hidden_state_allowed_count"],
            "predicate": "== 0",
            "passed": summary["bridge_hidden_state_allowed_count"] == 0,
            "why": "The coverage map must not imply bridge hidden-state readiness.",
        },
        {
            "id": "transfer_gap_preserves_untested_count",
            "actual": summary["transfer_untested_count"],
            "predicate": "== 14",
            "passed": summary["transfer_untested_count"] == 14,
            "why": "The current transfer gap is 14 untested rows.",
        },
        {
            "id": "axis_sparse_singleton_gap_present",
            "actual": summary["singleton_plus_sparse_feature_count"],
            "predicate": ">= 80",
            "passed": summary["singleton_plus_sparse_feature_count"] >= 80,
            "why": "Most feature-level regularities remain singleton or sparse.",
        },
        {
            "id": "coverage_gap_next_pressures_are_nonempty",
            "actual": [
                gap["id"]
                for gap in payload["coverage_gaps"]
                if not gap["next_pressure"]
            ],
            "predicate": "empty list",
            "passed": all(gap["next_pressure"] for gap in payload["coverage_gaps"]),
            "why": "Every gap should have at least one linked next pressure.",
        },
    ]
    return checks


def build_control_surface_coverage_gaps() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    genome_snapshot = load_json(GENOME_SNAPSHOT_PATH)
    axis_interactions = load_json(AXIS_INTERACTIONS_PATH)
    bridge_ladder = load_json(BRIDGE_LADDER_PATH)
    next_queue = load_json(NEXT_QUEUE_PATH)
    reliability_matrix = load_json(RELIABILITY_MATRIX_PATH)
    transfer_matrix = load_json(TRANSFER_MATRIX_PATH)
    gaps = build_coverage_gaps(
        atlas,
        genome_snapshot,
        axis_interactions,
        bridge_ladder,
        next_queue,
        reliability_matrix,
        transfer_matrix,
    )
    severity_counts = dict(sorted(Counter(gap["severity"] for gap in gaps).items()))
    gap_type_counts = dict(sorted(Counter(gap["gap_type"] for gap in gaps).items()))
    axis_summary = axis_interactions["summary"]
    payload = {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Expose the negative-space map for the current control-surface "
            "genome: what is still missing, undercovered, singleton-only, or "
            "not licensed for mechanism claims."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "axis_interactions": rel(AXIS_INTERACTIONS_PATH),
            "bridge_ladder": rel(BRIDGE_LADDER_PATH),
            "genome_snapshot": rel(GENOME_SNAPSHOT_PATH),
            "next_queue": rel(NEXT_QUEUE_PATH),
            "reliability_matrix": rel(RELIABILITY_MATRIX_PATH),
            "transfer_matrix": rel(TRANSFER_MATRIX_PATH),
        },
        "summary": {
            "row_count": len(atlas["rows"]),
            "gap_count": len(gaps),
            "severity_counts": severity_counts,
            "gap_type_counts": gap_type_counts,
            "transfer_untested_count": transfer_matrix["summary"]["transfer_value_counts"][
                "untested"
            ],
            "bridge_hidden_state_allowed_count": bridge_ladder["summary"][
                "hidden_state_allowed_count"
            ],
            "clean_unconfounded_bridge_count": bridge_ladder["summary"][
                "clean_unconfounded_bridge_count"
            ],
            "singleton_plus_sparse_feature_count": axis_summary[
                "evidence_level_counts"
            ].get("singleton", 0)
            + axis_summary["evidence_level_counts"].get("sparse", 0),
            "model_row_counts": {
                model: len(rows) for model, rows in model_rows(atlas).items()
            },
            "domain_row_counts": {
                domain: len(rows) for domain, rows in row_ids_by_domain(atlas).items()
            },
        },
        "coverage_gaps": gaps,
        "allowed_claim": (
            "The current map has clear negative-space boundaries: no promoted "
            "mechanism, no full reliability, no transfer-ready mechanism, no "
            "hidden-state-allowed bridge rung, and many singleton/sparse feature "
            "regularities."
        ),
        "forbidden_claim": (
            "Do not interpret the genome snapshot or axis interactions as a "
            "complete coverage map. This artifact records the missing coverage "
            "that must be reduced before broad small-LLM claims are justified."
        ),
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_coverage_gaps(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("coverage gaps schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"coverage gaps source missing: {rel_path}")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"coverage gaps checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Coverage Gaps",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated coverage-gap layer implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_coverage_gaps.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_coverage_gaps.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_coverage_gaps.py --write",
        "python code\\control_surface_coverage_gaps.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This layer is the negative-space map. It records which claims the",
        "current atlas still cannot support, which apparent laws are too sparse,",
        "and which next pressures are attached to those gaps.",
        "",
        "## Generated Facts",
        "",
        f"- rows: {summary['row_count']};",
        f"- coverage gaps: {summary['gap_count']};",
        f"- severity counts: `{format_value(summary['severity_counts'])}`;",
        f"- gap type counts: `{format_value(summary['gap_type_counts'])}`;",
        f"- transfer-untested rows: {summary['transfer_untested_count']};",
        f"- hidden-state-allowed bridge rungs: {summary['bridge_hidden_state_allowed_count']};",
        f"- clean unconfounded bridge rungs: {summary['clean_unconfounded_bridge_count']};",
        f"- singleton+sparse feature summaries: {summary['singleton_plus_sparse_feature_count']}.",
        "",
        "## Gap Ledger",
        "",
        "| Gap | Severity | Type | Evidence | Exit Condition |",
        "| --- | --- | --- | --- | --- |",
    ]
    for gap in payload["coverage_gaps"]:
        lines.append(
            f"| `{gap['id']}` | `{gap['severity']}` | `{gap['gap_type']}` | "
            f"`{format_value(gap['evidence'])}` | {gap['exit_condition']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The atlas is not merely incomplete in an abstract sense. Its missing",
            "coverage is structured: no promoted mechanisms, no full reliability,",
            "no transfer-ready mechanisms, no hidden-state-ready bridge, thin",
            "model-family coverage, singleton terminal-stage boundaries, and many",
            "singleton or sparse feature regularities.",
            "",
            "The immediate scientific pressure is therefore not to make a stronger",
            "claim from the current map. It is to reduce one of these gaps in a",
            "way that changes the generated snapshot, gate geometry, or axis",
            "interaction artifact.",
            "",
            "## What This Proves",
            "",
            "It proves that the current negative space is now explicit and tied to",
            "validated artifacts. Future experiments can be judged by whether",
            "they reduce a named gap rather than merely adding another row.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not prove that these gaps are exhaustive. It proves that these",
            "are the current validated gaps implied by the checked-in atlas stack.",
            "",
        ]
    )
    return "\n".join(lines)


def write_coverage_gaps(
    output_path: Path = COVERAGE_GAPS_PATH,
    report_path: Path = COVERAGE_GAPS_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_control_surface_coverage_gaps()
    validate_coverage_gaps(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write coverage-gap artifacts")
    parser.add_argument("--json", action="store_true", help="print coverage-gap JSON")
    args = parser.parse_args()

    payload = build_control_surface_coverage_gaps()
    validate_coverage_gaps(payload)

    if args.write:
        write_coverage_gaps()
        print(
            f"wrote {COVERAGE_GAPS_PATH.relative_to(ROOT).as_posix()} and "
            f"{COVERAGE_GAPS_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['gap_count']} gaps"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"coverage gaps ok: {payload['summary']['gap_count']} gaps")
    print("severity_counts:", json.dumps(payload["summary"]["severity_counts"], sort_keys=True))
    print("gap_type_counts:", json.dumps(payload["summary"]["gap_type_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
