"""Audit control-surface law hypotheses against the atlas.

The law hypothesis file is intentionally theory-shaped, but it should not be
allowed to drift into unsupported doctrine. This module checks whether each
hypothesis is backed by the rows and diagnostics it cites, then writes a compact
machine-readable audit for validation and comparison.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_comparison import COMPARISON_PATH


LAW_HYPOTHESES_PATH = ROOT / "data" / "control_surface_law_hypotheses.json"
LAW_AUDIT_PATH = ROOT / "data" / "control_surface_law_audit.json"

ALLOWED_STATUSES = {
    "strong_doctrine",
    "supported_pattern",
    "tentative_pattern",
}


def count_values(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_row_maps(atlas: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    rows_by_id = {row["id"]: row for row in atlas["rows"]}
    diagnostics_to_rows: dict[str, list[str]] = defaultdict(list)
    for row in atlas["rows"]:
        for diagnostic in row["diagnostics"]:
            diagnostics_to_rows[diagnostic].append(row["id"])
    return rows_by_id, {
        diagnostic: sorted(row_ids)
        for diagnostic, row_ids in sorted(diagnostics_to_rows.items())
    }


def row_state_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "verdict": count_values([row["verdict"]["class"] for row in rows]),
        "lead_time": count_values([row["lead_time"]["state"] for row in rows]),
        "intervention": count_values([row["intervention"]["state"] for row in rows]),
        "behavior_gate": count_values([row["behavior_gate"] for row in rows]),
    }


def mixture_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    if not rows:
        return {}
    axes = sorted(rows[0]["mixture_profile"])
    return {
        axis: count_values([row["mixture_profile"][axis] for row in rows])
        for axis in axes
    }


def hypothesis_audit(
    hypothesis: dict[str, Any],
    rows_by_id: dict[str, dict[str, Any]],
    diagnostics_to_rows: dict[str, list[str]],
) -> dict[str, Any]:
    evidence_rows = [rows_by_id[row_id] for row_id in hypothesis["evidence_rows"]]
    evidence_row_diagnostics: set[str] = set()
    for row in evidence_rows:
        evidence_row_diagnostics.update(row["diagnostics"])

    cited_diagnostics = set(hypothesis["evidence_diagnostics"])
    observed_diagnostics = set(diagnostics_to_rows)
    diagnostics_in_evidence_rows = sorted(cited_diagnostics & evidence_row_diagnostics)
    diagnostics_missing_from_evidence_rows = sorted(
        cited_diagnostics - evidence_row_diagnostics
    )
    diagnostics_not_observed = sorted(cited_diagnostics - observed_diagnostics)

    diagnostic_support_rows = {
        diagnostic: [
            row_id
            for row_id in diagnostics_to_rows.get(diagnostic, [])
            if row_id in hypothesis["evidence_rows"]
        ]
        for diagnostic in sorted(cited_diagnostics)
    }

    gaps: list[str] = []
    if hypothesis["status"] not in ALLOWED_STATUSES:
        gaps.append("UNKNOWN_HYPOTHESIS_STATUS")
    if diagnostics_missing_from_evidence_rows:
        gaps.append("CITED_DIAGNOSTIC_NOT_IN_EVIDENCE_ROWS")
    if diagnostics_not_observed:
        gaps.append("CITED_DIAGNOSTIC_NOT_OBSERVED")
    if hypothesis["status"] == "supported_pattern" and len(evidence_rows) < 2:
        gaps.append("SUPPORTED_PATTERN_SINGLE_ROW")
    if hypothesis["status"] == "strong_doctrine" and len(evidence_rows) < 3:
        gaps.append("STRONG_DOCTRINE_TOO_FEW_ROWS")
    if len(hypothesis["falsifiers"]) < 2:
        gaps.append("TOO_FEW_FALSIFIERS")
    if len(hypothesis["next_tests"]) < 2:
        gaps.append("TOO_FEW_NEXT_TESTS")

    if gaps:
        audit_level = "needs_review"
    elif hypothesis["status"] == "strong_doctrine":
        audit_level = "doctrine_evidence_consistent"
    elif hypothesis["status"] == "supported_pattern":
        audit_level = "pattern_evidence_consistent"
    else:
        audit_level = "tentative_evidence_consistent"

    return {
        "id": hypothesis["id"],
        "status": hypothesis["status"],
        "audit_level": audit_level,
        "gaps": gaps,
        "evidence_row_count": len(hypothesis["evidence_rows"]),
        "cited_diagnostic_count": len(hypothesis["evidence_diagnostics"]),
        "diagnostics_in_evidence_rows": diagnostics_in_evidence_rows,
        "diagnostics_missing_from_evidence_rows": diagnostics_missing_from_evidence_rows,
        "diagnostics_not_observed": diagnostics_not_observed,
        "diagnostic_support_rows": diagnostic_support_rows,
        "row_state_counts": row_state_counts(evidence_rows),
        "mixture_counts": mixture_counts(evidence_rows),
        "falsifier_count": len(hypothesis["falsifiers"]),
        "next_test_count": len(hypothesis["next_tests"]),
        "prediction_count": len(hypothesis["predicted_next_observations"]),
    }


def build_control_surface_law_audit(
    atlas: dict[str, Any],
    hypotheses_payload: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    rows_by_id, diagnostics_to_rows = build_row_maps(atlas)

    hypothesis_audits = [
        hypothesis_audit(hypothesis, rows_by_id, diagnostics_to_rows)
        for hypothesis in hypotheses_payload["hypotheses"]
    ]

    evidence_row_counts: Counter[str] = Counter()
    cited_diagnostic_counts: Counter[str] = Counter()
    for hypothesis in hypotheses_payload["hypotheses"]:
        evidence_row_counts.update(hypothesis["evidence_rows"])
        cited_diagnostic_counts.update(hypothesis["evidence_diagnostics"])

    row_ids = set(rows_by_id)
    rows_without_law_support = sorted(row_ids - set(evidence_row_counts))
    observed_diagnostics = set(diagnostics_to_rows)
    diagnostics_without_law_support = sorted(observed_diagnostics - set(cited_diagnostic_counts))
    unobserved_cited_diagnostics = sorted(set(cited_diagnostic_counts) - observed_diagnostics)
    hypotheses_with_gaps = sorted(
        audit["id"] for audit in hypothesis_audits if audit["gaps"]
    )

    return {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "atlas_ref": str(ATLAS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "hypotheses_ref": str(LAW_HYPOTHESES_PATH.relative_to(ROOT)).replace("\\", "/"),
        "comparison_ref": str(COMPARISON_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source": "code/control_surface_law_audit.py",
        "purpose": (
            "Audit whether law hypotheses cite currently observed atlas rows "
            "and diagnostics. This is a support audit, not proof that the laws "
            "are true."
        ),
        "summary": {
            "hypothesis_count": len(hypothesis_audits),
            "status_counts": count_values(
                [hypothesis["status"] for hypothesis in hypotheses_payload["hypotheses"]]
            ),
            "audit_level_counts": count_values(
                [audit["audit_level"] for audit in hypothesis_audits]
            ),
            "hypotheses_with_gaps": hypotheses_with_gaps,
            "row_count": len(rows_by_id),
            "rows_with_law_support": len(row_ids - set(rows_without_law_support)),
            "rows_without_law_support": rows_without_law_support,
            "observed_diagnostic_count": len(observed_diagnostics),
            "cited_diagnostic_count": len(cited_diagnostic_counts),
            "unobserved_cited_diagnostics": unobserved_cited_diagnostics,
            "observed_diagnostics_without_law_support": diagnostics_without_law_support,
            "comparison_shape_ref": comparison.get("current_genome_shape", {}),
        },
        "row_evidence_counts": dict(sorted(evidence_row_counts.items())),
        "diagnostic_evidence_counts": dict(sorted(cited_diagnostic_counts.items())),
        "hypotheses": hypothesis_audits,
    }


def write_law_audit(
    atlas: dict[str, Any],
    hypotheses_payload: dict[str, Any],
    comparison: dict[str, Any],
    output_path: Path = LAW_AUDIT_PATH,
) -> dict[str, Any]:
    audit = build_control_surface_law_audit(atlas, hypotheses_payload, comparison)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write data/control_surface_law_audit.json")
    parser.add_argument("--json", action="store_true", help="print law-audit JSON")
    args = parser.parse_args()

    atlas = load_json(ATLAS_PATH)
    hypotheses_payload = load_json(LAW_HYPOTHESES_PATH)
    comparison = load_json(COMPARISON_PATH)
    audit = build_control_surface_law_audit(atlas, hypotheses_payload, comparison)

    if args.write:
        write_law_audit(atlas, hypotheses_payload, comparison)
        print(
            f"wrote {LAW_AUDIT_PATH.relative_to(ROOT).as_posix()} "
            f"with {audit['summary']['hypothesis_count']} hypotheses"
        )
        return
    if args.json:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    print(f"law audit ok: {audit['summary']['hypothesis_count']} hypotheses")
    print("status_counts:", json.dumps(audit["summary"]["status_counts"], sort_keys=True))
    print(
        "audit_level_counts:",
        json.dumps(audit["summary"]["audit_level_counts"], sort_keys=True),
    )
    print(
        "rows_without_law_support:",
        json.dumps(audit["summary"]["rows_without_law_support"]),
    )
    print(
        "hypotheses_with_gaps:",
        json.dumps(audit["summary"]["hypotheses_with_gaps"]),
    )


if __name__ == "__main__":
    main()
