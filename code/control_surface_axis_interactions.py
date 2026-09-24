"""Build predictive axis interactions for the control-surface atlas.

The genome snapshot states the current global shape. This layer asks which
features predict that shape: mixture axes, diagnostics, route dispositions,
frontier classes, and reliability classes. It keeps sample counts and purity
visible so small-n deterministic-looking rules do not become overclaimed laws.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json


AXIS_INTERACTIONS_PATH = ROOT / "data" / "control_surface_axis_interactions.json"
AXIS_INTERACTIONS_REPORT_PATH = ROOT / "research" / "36_CONTROL_SURFACE_AXIS_INTERACTIONS.md"

GENOME_SNAPSHOT_PATH = ROOT / "data" / "control_surface_genome_snapshot.json"
MIXTURE_LAW_PATH = ROOT / "data" / "control_surface_mixture_law.json"
GATE_GEOMETRY_PATH = ROOT / "data" / "control_surface_gate_geometry.json"
ROUTE_DISPOSITION_PATH = ROOT / "data" / "control_surface_route_disposition.json"
DECISION_FRONTIER_PATH = ROOT / "data" / "control_surface_decision_frontier.json"
RELIABILITY_MATRIX_PATH = ROOT / "data" / "control_surface_reliability_matrix.json"

FEATURE_SOURCE_ORDER = [
    "route_disposition",
    "primary_blocker",
    "frontier_class",
    "reliability_class",
    "mixture_axis",
    "pressure_class",
    "diagnostic",
]


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


def by_id(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {item[key]: item for item in items}


def count_values(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def evidence_level(count: int) -> str:
    if count >= 5:
        return "broad"
    if count >= 3:
        return "supported"
    if count >= 2:
        return "sparse"
    return "singleton"


def feature_key(source: str, name: str, value: str) -> str:
    return f"{source}:{name}={value}"


def build_row_feature_entries(
    atlas: dict[str, Any],
    mixture_law: dict[str, Any],
    gate_geometry: dict[str, Any],
    route_disposition: dict[str, Any],
    decision_frontier: dict[str, Any],
    reliability_matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    mixture_by_row = by_id(mixture_law["row_profiles"], "id")
    gate_by_row = by_id(gate_geometry["gate_entries"], "row_id")
    route_by_row = by_id(route_disposition["route_entries"], "row_id")
    frontier_by_row = by_id(decision_frontier["frontier_rows"], "id")
    reliability_by_row = by_id(reliability_matrix["reliability_entries"], "row_id")
    entries: list[dict[str, Any]] = []

    for row in atlas["rows"]:
        row_id = row["id"]
        mixture = mixture_by_row[row_id]
        gate = gate_by_row[row_id]
        route = route_by_row[row_id]
        frontier = frontier_by_row[row_id]
        reliability = reliability_by_row[row_id]
        features: list[dict[str, str]] = [
            {
                "source": "route_disposition",
                "name": "disposition",
                "value": route["disposition"],
                "feature_key": feature_key(
                    "route_disposition",
                    "disposition",
                    route["disposition"],
                ),
            },
            {
                "source": "primary_blocker",
                "name": "primary_blocker",
                "value": route["primary_blocker"],
                "feature_key": feature_key(
                    "primary_blocker",
                    "primary_blocker",
                    route["primary_blocker"],
                ),
            },
            {
                "source": "frontier_class",
                "name": "frontier_class",
                "value": frontier["frontier_class"],
                "feature_key": feature_key(
                    "frontier_class",
                    "frontier_class",
                    frontier["frontier_class"],
                ),
            },
            {
                "source": "reliability_class",
                "name": "reliability_class",
                "value": reliability["reliability_class"],
                "feature_key": feature_key(
                    "reliability_class",
                    "reliability_class",
                    reliability["reliability_class"],
                ),
            },
        ]
        for axis, value in sorted(row["mixture_profile"].items()):
            features.append(
                {
                    "source": "mixture_axis",
                    "name": axis,
                    "value": value,
                    "feature_key": feature_key("mixture_axis", axis, value),
                }
            )
        for pressure_class in mixture["pressure_classes"]:
            features.append(
                {
                    "source": "pressure_class",
                    "name": "pressure_class",
                    "value": pressure_class,
                    "feature_key": feature_key(
                        "pressure_class",
                        "pressure_class",
                        pressure_class,
                    ),
                }
            )
        for diagnostic in row["diagnostics"]:
            features.append(
                {
                    "source": "diagnostic",
                    "name": "diagnostic",
                    "value": diagnostic,
                    "feature_key": feature_key("diagnostic", "diagnostic", diagnostic),
                }
            )

        entries.append(
            {
                "row_id": row_id,
                "family": row["family"],
                "terminal_stage": gate["terminal_stage"],
                "route_disposition": route["disposition"],
                "primary_blocker": route["primary_blocker"],
                "frontier_class": frontier["frontier_class"],
                "reliability_class": reliability["reliability_class"],
                "verdict": row["verdict"]["class"],
                "lead_time_state": row["lead_time"]["state"],
                "intervention_state": row["intervention"]["state"],
                "pressure_classes": mixture["pressure_classes"],
                "mixture_profile": row["mixture_profile"],
                "diagnostics": row["diagnostics"],
                "features": features,
            }
        )
    return entries


def build_feature_summaries(row_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_rows: dict[str, list[dict[str, Any]]] = {}
    feature_meta: dict[str, dict[str, str]] = {}
    for row in row_entries:
        for feature in row["features"]:
            key = feature["feature_key"]
            feature_rows.setdefault(key, []).append(row)
            feature_meta[key] = {
                "source": feature["source"],
                "name": feature["name"],
                "value": feature["value"],
            }

    summaries: list[dict[str, Any]] = []
    for key, rows in feature_rows.items():
        terminal_counts = count_values([row["terminal_stage"] for row in rows])
        route_counts = count_values([row["route_disposition"] for row in rows])
        blocker_counts = count_values([row["primary_blocker"] for row in rows])
        count = len(rows)
        dominant_stage, dominant_stage_count = max(
            terminal_counts.items(),
            key=lambda item: (item[1], item[0]),
        )
        summaries.append(
            {
                "feature_key": key,
                **feature_meta[key],
                "row_count": count,
                "evidence_level": evidence_level(count),
                "terminal_stage_counts": terminal_counts,
                "route_disposition_counts": route_counts,
                "primary_blocker_counts": blocker_counts,
                "dominant_terminal_stage": dominant_stage,
                "dominant_terminal_stage_count": dominant_stage_count,
                "terminal_stage_purity": ratio(dominant_stage_count, count),
                "rows": sorted(row["row_id"] for row in rows),
            }
        )

    summaries.sort(
        key=lambda item: (
            FEATURE_SOURCE_ORDER.index(item["source"]),
            -item["row_count"],
            -item["terminal_stage_purity"],
            item["feature_key"],
        )
    )
    return summaries


def build_stage_feature_profile(
    row_entries: list[dict[str, Any]],
    feature_summaries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    row_count_by_stage = Counter(row["terminal_stage"] for row in row_entries)
    summaries_by_key = {summary["feature_key"]: summary for summary in feature_summaries}
    stage_features: dict[str, Counter[str]] = {}
    for row in row_entries:
        stage_counter = stage_features.setdefault(row["terminal_stage"], Counter())
        for feature in row["features"]:
            stage_counter[feature["feature_key"]] += 1

    profile: dict[str, dict[str, Any]] = {}
    for stage, counter in sorted(stage_features.items()):
        stage_count = row_count_by_stage[stage]
        top_features = []
        for key, count in counter.most_common():
            summary = summaries_by_key[key]
            if summary["source"] == "diagnostic" and summary["row_count"] == 1:
                continue
            top_features.append(
                {
                    "feature_key": key,
                    "source": summary["source"],
                    "name": summary["name"],
                    "value": summary["value"],
                    "stage_count": count,
                    "stage_coverage": ratio(count, stage_count),
                    "global_row_count": summary["row_count"],
                    "terminal_stage_purity": summary["terminal_stage_purity"],
                    "evidence_level": summary["evidence_level"],
                }
            )
            if len(top_features) == 10:
                break
        profile[stage] = {
            "row_count": stage_count,
            "top_features": top_features,
        }
    return profile


def build_predictive_rules(feature_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for summary in feature_summaries:
        if summary["row_count"] < 2:
            continue
        if summary["terminal_stage_purity"] < 1.0:
            continue
        if summary["source"] == "diagnostic" and summary["row_count"] < 3:
            continue
        rules.append(
            {
                "feature_key": summary["feature_key"],
                "source": summary["source"],
                "name": summary["name"],
                "value": summary["value"],
                "predicts_terminal_stage": summary["dominant_terminal_stage"],
                "row_count": summary["row_count"],
                "evidence_level": summary["evidence_level"],
                "purity": summary["terminal_stage_purity"],
                "rows": summary["rows"],
                "allowed_use": (
                    "Use as a current predictive regularity only at the stated "
                    "evidence level; do not generalize beyond current contracts."
                ),
            }
        )
    rules.sort(
        key=lambda rule: (
            FEATURE_SOURCE_ORDER.index(rule["source"]),
            -rule["row_count"],
            rule["feature_key"],
        )
    )
    return rules


def build_mixed_predictors(feature_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mixed = [
        summary
        for summary in feature_summaries
        if summary["row_count"] >= 3 and summary["terminal_stage_purity"] < 1.0
    ]
    mixed.sort(
        key=lambda item: (
            -item["row_count"],
            item["terminal_stage_purity"],
            item["feature_key"],
        )
    )
    return mixed


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    row_count = payload["summary"]["row_count"]
    feature_summaries = payload["feature_summaries"]
    route_rules = [
        rule
        for rule in payload["predictive_rules"]
        if rule["source"] == "route_disposition"
    ]
    checks = [
        {
            "id": "row_feature_entries_cover_snapshot_rows",
            "actual": {
                "row_feature_entries": len(payload["row_feature_entries"]),
                "row_count": row_count,
            },
            "predicate": "row_feature_entries == row_count",
            "passed": len(payload["row_feature_entries"]) == row_count,
            "why": "Every atlas row must be represented in the predictive feature map.",
        },
        {
            "id": "feature_summaries_nonempty",
            "actual": len(feature_summaries),
            "predicate": "> 0",
            "passed": len(feature_summaries) > 0,
            "why": "The interaction artifact must expose actual features.",
        },
        {
            "id": "route_disposition_rules_present",
            "actual": [rule["feature_key"] for rule in route_rules],
            "predicate": "closed-before-hidden-state and output-shadow route rules exist",
            "passed": {
                "route_disposition:disposition=closed_before_hidden_state",
                "route_disposition:disposition=output_shadow_diagnostic_baseline",
            }.issubset({rule["feature_key"] for rule in route_rules}),
            "why": "The broad route-disposition rules must preserve the main pre-signature and output-shadow regularities.",
        },
        {
            "id": "behavior_substrate_route_predicts_pre_signature",
            "actual": [
                rule
                for rule in payload["predictive_rules"]
                if rule["feature_key"]
                == "route_disposition:disposition=closed_before_hidden_state"
            ],
            "predicate": "single rule predicts pre_signature_behavior_substrate over 11 rows",
            "passed": any(
                rule["predicts_terminal_stage"] == "pre_signature_behavior_substrate"
                and rule["row_count"] == 11
                for rule in payload["predictive_rules"]
                if rule["feature_key"]
                == "route_disposition:disposition=closed_before_hidden_state"
            ),
            "why": "The largest current predictive law is behavior/bridge closure before hidden-state work.",
        },
        {
            "id": "output_geometry_is_mixed_not_single_rule",
            "actual": [
                summary
                for summary in payload["mixed_predictors"]
                if summary["feature_key"]
                == "primary_blocker:primary_blocker=output_geometry_shadow"
            ],
            "predicate": "output_geometry_shadow appears in mixed predictors",
            "passed": any(
                summary["feature_key"]
                == "primary_blocker:primary_blocker=output_geometry_shadow"
                for summary in payload["mixed_predictors"]
            ),
            "why": "Output geometry explains a family of failures, not one terminal stage.",
        },
        {
            "id": "singleton_rules_are_labeled_singleton",
            "actual": [
                summary["feature_key"]
                for summary in feature_summaries
                if summary["row_count"] == 1
                and summary["evidence_level"] != "singleton"
            ],
            "predicate": "empty list",
            "passed": all(
                summary["evidence_level"] == "singleton"
                for summary in feature_summaries
                if summary["row_count"] == 1
            ),
            "why": "Small-n deterministic features must not be promoted as broad laws.",
        },
    ]
    return checks


def build_control_surface_axis_interactions() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    genome_snapshot = load_json(GENOME_SNAPSHOT_PATH)
    mixture_law = load_json(MIXTURE_LAW_PATH)
    gate_geometry = load_json(GATE_GEOMETRY_PATH)
    route_disposition = load_json(ROUTE_DISPOSITION_PATH)
    decision_frontier = load_json(DECISION_FRONTIER_PATH)
    reliability_matrix = load_json(RELIABILITY_MATRIX_PATH)

    row_entries = build_row_feature_entries(
        atlas,
        mixture_law,
        gate_geometry,
        route_disposition,
        decision_frontier,
        reliability_matrix,
    )
    feature_summaries = build_feature_summaries(row_entries)
    stage_feature_profile = build_stage_feature_profile(row_entries, feature_summaries)
    predictive_rules = build_predictive_rules(feature_summaries)
    mixed_predictors = build_mixed_predictors(feature_summaries)
    payload = {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Expose which observed axes predict terminal claim stages in the "
            "current control-surface genome map, with row counts and purity so "
            "small-n patterns remain bounded."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "genome_snapshot": rel(GENOME_SNAPSHOT_PATH),
            "mixture_law": rel(MIXTURE_LAW_PATH),
            "gate_geometry": rel(GATE_GEOMETRY_PATH),
            "route_disposition": rel(ROUTE_DISPOSITION_PATH),
            "decision_frontier": rel(DECISION_FRONTIER_PATH),
            "reliability_matrix": rel(RELIABILITY_MATRIX_PATH),
        },
        "summary": {
            "row_count": len(row_entries),
            "feature_count": len(feature_summaries),
            "predictive_rule_count": len(predictive_rules),
            "mixed_predictor_count": len(mixed_predictors),
            "terminal_stage_counts": genome_snapshot["gate_shape"][
                "terminal_stage_counts"
            ],
            "feature_source_counts": count_values(
                [summary["source"] for summary in feature_summaries]
            ),
            "evidence_level_counts": count_values(
                [summary["evidence_level"] for summary in feature_summaries]
            ),
            "broad_or_supported_rule_count": sum(
                1
                for rule in predictive_rules
                if rule["evidence_level"] in {"broad", "supported"}
            ),
        },
        "row_feature_entries": row_entries,
        "feature_summaries": feature_summaries,
        "stage_feature_profile": stage_feature_profile,
        "predictive_rules": predictive_rules,
        "mixed_predictors": mixed_predictors,
        "allowed_claim": (
            "Current terminal-stage outcomes are strongly predicted by route "
            "disposition and behavior-substrate closure, while output geometry "
            "is a broad mixed predictor rather than one terminal-stage law."
        ),
        "forbidden_claim": (
            "Do not treat singleton or sparse pure features as general laws. "
            "This artifact reports current-contract predictive regularities, "
            "not universal mechanism laws."
        ),
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_axis_interactions(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("axis interactions schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"axis interactions source missing: {rel_path}")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"axis interactions checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Axis Interactions",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated predictive axis-interaction layer implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_axis_interactions.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_axis_interactions.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_axis_interactions.py --write",
        "python code\\control_surface_axis_interactions.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This layer asks which observed features predict where a control-surface",
        "claim dies. It is intentionally small-n aware: every feature has a row",
        "count, evidence level, dominant terminal stage, and purity.",
        "",
        "## Generated Facts",
        "",
        f"- rows: {summary['row_count']};",
        f"- feature summaries: {summary['feature_count']};",
        f"- pure predictive rules: {summary['predictive_rule_count']};",
        f"- broad or supported pure rules: {summary['broad_or_supported_rule_count']};",
        f"- mixed predictors: {summary['mixed_predictor_count']};",
        f"- feature source counts: `{format_value(summary['feature_source_counts'])}`;",
        f"- evidence-level counts: `{format_value(summary['evidence_level_counts'])}`.",
        "",
        "## Strong Pure Rules",
        "",
        "| Feature | Predicts | Rows | Evidence | Purity |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for rule in payload["predictive_rules"]:
        if rule["evidence_level"] not in {"broad", "supported"}:
            continue
        lines.append(
            f"| `{rule['feature_key']}` | `{rule['predicts_terminal_stage']}` | "
            f"{rule['row_count']} | `{rule['evidence_level']}` | "
            f"{format_value(rule['purity'])} |"
        )

    lines.extend(
        [
            "",
            "## Mixed Predictors",
            "",
            "| Feature | Rows | Dominant Stage | Purity | Stage Counts |",
            "| --- | ---: | --- | ---: | --- |",
        ]
    )
    for item in payload["mixed_predictors"]:
        lines.append(
            f"| `{item['feature_key']}` | {item['row_count']} | "
            f"`{item['dominant_terminal_stage']}` | "
            f"{format_value(item['terminal_stage_purity'])} | "
            f"`{format_value(item['terminal_stage_counts'])}` |"
        )

    lines.extend(
        [
            "",
            "## Stage Feature Profiles",
            "",
        ]
    )
    for stage, profile in payload["stage_feature_profile"].items():
        lines.extend(
            [
                f"### `{stage}`",
                "",
                f"Rows: {profile['row_count']}",
                "",
                "| Feature | Stage Coverage | Global Rows | Purity | Evidence |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for feature in profile["top_features"][:6]:
            lines.append(
                f"| `{feature['feature_key']}` | "
                f"{format_value(feature['stage_coverage'])} | "
                f"{feature['global_row_count']} | "
                f"{format_value(feature['terminal_stage_purity'])} | "
                f"`{feature['evidence_level']}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "The largest current predictive rule is not a hidden mechanism rule.",
            "It is route-level closure: `closed_before_hidden_state` predicts",
            "`pre_signature_behavior_substrate` across 11 rows. That is the",
            "project's strongest current law about where claims die.",
            "",
            "Output geometry is different. It is broad, but mixed: it appears",
            "across output-shadow signatures, monitor-only rows, and the failed",
            "intervention route. So output geometry is a family-level pressure,",
            "not a single terminal-stage rule.",
            "",
            "MC005 remains a singleton internal-causal reliability boundary.",
            "The layer marks singleton purity as singleton evidence, not as a",
            "general law.",
            "",
            "## What This Proves",
            "",
            "It proves that the current genome map has predictive regularities, and",
            "that their evidence strength can be made explicit. The strongest",
            "broad rule is pre-signature route closure, not a truth vector or",
            "knowledge vector.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not prove universal laws of small LLMs. It proves current-contract",
            "regularities over the validated 19-row atlas and bridge-derived stack.",
            "",
        ]
    )
    return "\n".join(lines)


def write_axis_interactions(
    output_path: Path = AXIS_INTERACTIONS_PATH,
    report_path: Path = AXIS_INTERACTIONS_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_control_surface_axis_interactions()
    validate_axis_interactions(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write axis interaction artifacts")
    parser.add_argument("--json", action="store_true", help="print axis interaction JSON")
    args = parser.parse_args()

    payload = build_control_surface_axis_interactions()
    validate_axis_interactions(payload)

    if args.write:
        write_axis_interactions()
        print(
            f"wrote {AXIS_INTERACTIONS_PATH.relative_to(ROOT).as_posix()} and "
            f"{AXIS_INTERACTIONS_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['feature_count']} features"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"axis interactions ok: {payload['summary']['row_count']} rows")
    print(
        "feature_source_counts:",
        json.dumps(payload["summary"]["feature_source_counts"], sort_keys=True),
    )
    print(
        "predictive_rule_count:",
        payload["summary"]["predictive_rule_count"],
    )
    print(
        "mixed_predictor_count:",
        payload["summary"]["mixed_predictor_count"],
    )


if __name__ == "__main__":
    main()
