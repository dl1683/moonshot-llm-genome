"""Build a prioritized next-experiment queue from the law audit.

This is the operational counterpart to the law layer. The hypotheses say what
we think is happening; the queue says which falsification or widening test
should be run next and why.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_comparison import COMPARISON_PATH
from control_surface_law_audit import (
    LAW_AUDIT_PATH,
    LAW_HYPOTHESES_PATH,
)


NEXT_QUEUE_PATH = ROOT / "data" / "control_surface_next_experiment_queue.json"
BRIDGE_LADDER_PATH = ROOT / "data" / "control_surface_bridge_ladder.json"
ROUTE_DISPOSITION_PATH = ROOT / "data" / "control_surface_route_disposition.json"
ERROR_TAXONOMY_PATH = ROOT / "data" / "control_surface_error_taxonomy.json"


STATUS_BASE_PRIORITY = {
    "strong_doctrine": 55,
    "supported_pattern": 50,
    "tentative_pattern": 60,
}

REASON_WEIGHTS = {
    "TENTATIVE_PATTERN": 20,
    "SINGLE_ROW_LAW_SUPPORT": 25,
    "SUPPORTED_PATTERN_FALSIFICATION": 10,
    "STRONG_DOCTRINE_GUARDRAIL": 15,
    "BRIDGE_ROUTE_NEEDED": 20,
    "TRANSFER_GAP": 15,
    "INTERVENTION_RELEVANT": 15,
    "LEADTIME_FRONTIER": 10,
    "OUTPUT_GEOMETRY_CONTROL": 10,
    "BEHAVIOR_SUBSTRATE_GATE": 10,
    "ZERO_PROMOTED_MECHANISM_PRESSURE": 5,
}


def priority_class(score: int) -> str:
    if score >= 120:
        return "immediate"
    if score >= 100:
        return "high"
    if score >= 80:
        return "medium"
    return "watch"


def reason_codes(
    hypothesis_id: str,
    status: str,
    evidence_row_count: int,
    test_text: str,
    comparison_shape: dict[str, Any],
) -> list[str]:
    text = f"{hypothesis_id} {test_text}".lower()
    codes: list[str] = []
    if status == "tentative_pattern":
        codes.append("TENTATIVE_PATTERN")
    if status == "supported_pattern":
        codes.append("SUPPORTED_PATTERN_FALSIFICATION")
    if status == "strong_doctrine":
        codes.append("STRONG_DOCTRINE_GUARDRAIL")
    if evidence_row_count <= 1:
        codes.append("SINGLE_ROW_LAW_SUPPORT")
    if "bridge" in text or "mc009" in text or "post-mc009" in text:
        codes.append("BRIDGE_ROUTE_NEEDED")
    if "transfer" in text or "model families" in text or "model family" in text:
        codes.append("TRANSFER_GAP")
    if (
        "intervention" in text
        or "steer" in text
        or "control" in text
        or "causal" in text
    ):
        codes.append("INTERVENTION_RELEVANT")
    if "lead-time" in text or "pre-output" in text or "early" in text:
        codes.append("LEADTIME_FRONTIER")
    if "output" in text or "candidate" in text or "margin" in text:
        codes.append("OUTPUT_GEOMETRY_CONTROL")
    if (
        "behavior" in text
        or "substrate" in text
        or "parseability" in text
        or "null" in text
        or "holdout" in text
    ):
        codes.append("BEHAVIOR_SUBSTRATE_GATE")
    if (
        comparison_shape.get("promoted_mechanism_ratio") == 0
        and any(
            code in codes
            for code in [
                "BRIDGE_ROUTE_NEEDED",
                "INTERVENTION_RELEVANT",
                "LEADTIME_FRONTIER",
            ]
        )
    ):
        codes.append("ZERO_PROMOTED_MECHANISM_PRESSURE")
    return codes


def score_queue_item(status: str, codes: list[str]) -> int:
    score = STATUS_BASE_PRIORITY.get(status, 40)
    for code in codes:
        score += REASON_WEIGHTS[code]
    return score


def compact_falsifiers(falsifiers: list[str]) -> list[str]:
    return falsifiers[:2]


def action_type(test_text: str) -> str:
    text = test_text.lower()
    if text.startswith("do not") or "require " in text or "unless " in text:
        return "gate_rule"
    if text.startswith("run") or text.startswith("repeat") or "test" in text:
        return "experiment"
    return "operational_rule"


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def compact_metrics(rung: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "label": metric["label"],
            "value": metric["value"],
        }
        for metric in rung.get("headline_metrics", [])
    ]


def build_bridge_closure_context(
    bridge_ladder: dict[str, Any] | None,
    route_disposition: dict[str, Any] | None,
    error_taxonomy: dict[str, Any] | None,
) -> dict[str, Any]:
    if not bridge_ladder:
        return {
            "available": False,
            "constraints": [],
            "recent_closed_rungs": [],
        }

    rungs = bridge_ladder.get("rungs", [])
    recent_cards = {"MC030", "MC031", "MC032", "MC033"}
    recent_closed = [rung for rung in rungs if rung["card_id"] in recent_cards]
    bridge_counts = {}
    if route_disposition:
        bridge_counts = route_disposition.get("summary", {}).get(
            "bridge_disposition_counts",
            {},
        )
    error_summary = error_taxonomy.get("summary", {}) if error_taxonomy else {}

    constraints = [
        (
            "Do not reopen the MC028-MC030 operation-leak route through local "
            "prompt guards, example removal, query reordering, or answer-schema "
            "changes unless the branch, null, local, and side-number gates are "
            "all predeclared and pass together."
        ),
        (
            "Do not reopen the MC031 statusless checksum route unless invalid-source "
            "rows produce learned atomic/lure selections while direct controls, "
            "answer-absent nulls, source-disjoint holdout, and margin baselines "
            "are all reported."
        ),
        (
            "Do not reopen the MC032 cross-table consistency route unless mismatch "
            "rows produce learned atomic/lure selections without side-number "
            "copying while direct controls, answer-absent nulls, source-disjoint "
            "holdout, and margin baselines all pass together."
        ),
        (
            "Do not reopen the MC033 fact-claim route unless both match and "
            "mismatch branches pass together without local collapse, claimed-number "
            "copying, null damage, or output/candidate-margin explanation."
        ),
        (
            "Any new bridge must be materially different from source labels, "
            "row codes, query-operation handles, worked examples, constrained "
            "choices, numeric options, answer-interface sweeps, simple absence "
            "guards, and arithmetic checksum validity cues."
        ),
        (
            "A future bridge item remains behavior-substrate work until it also "
            "reports candidate/output baselines; hidden-state work is still "
            "forbidden for all smoke rungs."
        ),
    ]
    return {
        "available": True,
        "bridge_ladder_ref": str(BRIDGE_LADDER_PATH.relative_to(ROOT)).replace("\\", "/"),
        "route_disposition_ref": str(ROUTE_DISPOSITION_PATH.relative_to(ROOT)).replace("\\", "/"),
        "error_taxonomy_ref": str(ERROR_TAXONOMY_PATH.relative_to(ROOT)).replace("\\", "/"),
        "bridge_rung_count": bridge_ladder.get("summary", {}).get("rung_count"),
        "smoke_rung_count": bridge_ladder.get("summary", {}).get("smoke_rung_count"),
        "hidden_state_allowed_count": bridge_ladder.get("summary", {}).get(
            "hidden_state_allowed_count",
        ),
        "clean_unconfounded_bridge_count": bridge_ladder.get("summary", {}).get(
            "clean_unconfounded_bridge_count",
        ),
        "bridge_disposition_counts": bridge_counts,
        "error_taxonomy_summary": {
            "smoke_card_count": error_summary.get("smoke_card_count"),
            "bridge_rung_count": error_summary.get("bridge_rung_count"),
            "mc030_baseline_operation_atomic_rate": error_summary.get(
                "mc030_baseline_operation_atomic_rate",
            ),
            "mc030_baseline_answer_absent_unknown_rate": error_summary.get(
                "mc030_baseline_answer_absent_unknown_rate",
            ),
            "mc030_min_other_number_template_operation_atomic_rate": error_summary.get(
                "mc030_min_other_number_template_operation_atomic_rate",
            ),
            "mc031_invalid_checksum_atomic_or_lure_rate": error_summary.get(
                "mc031_invalid_checksum_atomic_or_lure_rate",
            ),
            "mc032_mismatch_conflict_atomic_or_lure_rate": error_summary.get(
                "mc032_mismatch_conflict_atomic_or_lure_rate",
            ),
            "mc032_mismatch_conflict_side_number_rate": error_summary.get(
                "mc032_mismatch_conflict_side_number_rate",
            ),
            "mc033_match_conflict_local_rate": error_summary.get(
                "mc033_match_conflict_local_rate",
            ),
            "mc033_mismatch_conflict_lure_rate": error_summary.get(
                "mc033_mismatch_conflict_lure_rate",
            ),
        },
        "closed_contract_axes": sorted(
            {
                rung["contract_axis"]
                for rung in rungs
                if not rung.get("hidden_state_allowed")
            }
        ),
        "recent_closed_rungs": [
            {
                "card_id": rung["card_id"],
                "title": rung["title"],
                "contract_axis": rung["contract_axis"],
                "behavior_outcome": rung["behavior_outcome"],
                "local_vs_learned_mixture": rung["local_vs_learned_mixture"],
                "dominant_failure": rung["dominant_failure"],
                "claim_boundary": rung["claim_boundary"],
                "headline_metrics": compact_metrics(rung),
            }
            for rung in recent_closed
        ],
        "constraints": constraints,
    }


def build_control_surface_next_queue(
    atlas: dict[str, Any],
    hypotheses_payload: dict[str, Any],
    law_audit: dict[str, Any],
    comparison: dict[str, Any],
    bridge_ladder: dict[str, Any] | None = None,
    route_disposition: dict[str, Any] | None = None,
    error_taxonomy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit_by_id = {audit["id"]: audit for audit in law_audit["hypotheses"]}
    comparison_shape = comparison.get("current_genome_shape", {})
    if bridge_ladder is None:
        bridge_ladder = maybe_load_json(BRIDGE_LADDER_PATH)
    if route_disposition is None:
        route_disposition = maybe_load_json(ROUTE_DISPOSITION_PATH)
    if error_taxonomy is None:
        error_taxonomy = maybe_load_json(ERROR_TAXONOMY_PATH)
    bridge_closure_context = build_bridge_closure_context(
        bridge_ladder,
        route_disposition,
        error_taxonomy,
    )
    queue_items: list[dict[str, Any]] = []

    for hypothesis in hypotheses_payload["hypotheses"]:
        audit = audit_by_id[hypothesis["id"]]
        for test_index, test_text in enumerate(hypothesis["next_tests"], start=1):
            codes = reason_codes(
                hypothesis_id=hypothesis["id"],
                status=hypothesis["status"],
                evidence_row_count=audit["evidence_row_count"],
                test_text=test_text,
                comparison_shape=comparison_shape,
            )
            score = score_queue_item(hypothesis["status"], codes)
            item_bridge_context = {}
            if "BRIDGE_ROUTE_NEEDED" in codes and bridge_closure_context["available"]:
                item_bridge_context = {
                    "recent_closed_rung_ids": [
                        rung["card_id"]
                        for rung in bridge_closure_context["recent_closed_rungs"]
                    ],
                    "active_constraints": bridge_closure_context["constraints"],
                }
            queue_items.append(
                {
                    "id": f"{hypothesis['id']}__next_test_{test_index}",
                    "hypothesis_id": hypothesis["id"],
                    "hypothesis_status": hypothesis["status"],
                    "audit_level": audit["audit_level"],
                    "priority_score": score,
                    "priority_class": priority_class(score),
                    "reason_codes": codes,
                    "action_type": action_type(test_text),
                    "next_test": test_text,
                    "expected_observations": hypothesis["predicted_next_observations"],
                    "falsifiers": compact_falsifiers(hypothesis["falsifiers"]),
                    "evidence_row_count": audit["evidence_row_count"],
                    "cited_diagnostic_count": audit["cited_diagnostic_count"],
                    "diagnostics_in_evidence_rows": audit[
                        "diagnostics_in_evidence_rows"
                    ],
                    "bridge_closure_context": item_bridge_context,
                }
            )

    queue_items.sort(
        key=lambda item: (
            -item["priority_score"],
            item["hypothesis_id"],
            item["id"],
        )
    )

    priority_counts = Counter(item["priority_class"] for item in queue_items)
    reason_counts = Counter(
        code for item in queue_items for code in item["reason_codes"]
    )

    top_items = queue_items[:5]
    immediate_or_high = [
        item for item in queue_items if item["priority_class"] in {"immediate", "high"}
    ]

    return {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "atlas_ref": str(ATLAS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "hypotheses_ref": str(LAW_HYPOTHESES_PATH.relative_to(ROOT)).replace("\\", "/"),
        "law_audit_ref": str(LAW_AUDIT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "comparison_ref": str(COMPARISON_PATH.relative_to(ROOT)).replace("\\", "/"),
        "bridge_ladder_ref": str(BRIDGE_LADDER_PATH.relative_to(ROOT)).replace("\\", "/"),
        "route_disposition_ref": str(ROUTE_DISPOSITION_PATH.relative_to(ROOT)).replace("\\", "/"),
        "error_taxonomy_ref": str(ERROR_TAXONOMY_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source": "code/control_surface_next_queue.py",
        "purpose": (
            "Prioritize next experiments from validated law hypotheses and "
            "current bridge-closure evidence. This queue compiles current "
            "next_tests, ranks them by evidence weakness, bridge pressure, "
            "transfer gaps, and intervention relevance, and attaches active "
            "constraints from killed bridge routes."
        ),
        "scoring": {
            "status_base_priority": STATUS_BASE_PRIORITY,
            "reason_weights": REASON_WEIGHTS,
            "priority_classes": {
                "immediate": "score >= 120",
                "high": "100 <= score < 120",
                "medium": "80 <= score < 100",
                "watch": "score < 80",
            },
        },
        "summary": {
            "queue_item_count": len(queue_items),
            "priority_counts": dict(sorted(priority_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "immediate_or_high_count": len(immediate_or_high),
            "top_queue_ids": [item["id"] for item in top_items],
            "top_hypotheses": [item["hypothesis_id"] for item in top_items],
            "comparison_shape_ref": comparison_shape,
            "bridge_closure": {
                "available": bridge_closure_context["available"],
                "bridge_rung_count": bridge_closure_context.get("bridge_rung_count"),
                "smoke_rung_count": bridge_closure_context.get("smoke_rung_count"),
                "hidden_state_allowed_count": bridge_closure_context.get(
                    "hidden_state_allowed_count",
                ),
                "clean_unconfounded_bridge_count": bridge_closure_context.get(
                    "clean_unconfounded_bridge_count",
                ),
                "recent_closed_rung_ids": [
                    rung["card_id"]
                    for rung in bridge_closure_context.get("recent_closed_rungs", [])
                ],
                "closed_contract_axis_count": len(
                    bridge_closure_context.get("closed_contract_axes", [])
                ),
            },
        },
        "bridge_closure_context": bridge_closure_context,
        "top_items": top_items,
        "queue": queue_items,
    }


def write_next_queue(
    atlas: dict[str, Any],
    hypotheses_payload: dict[str, Any],
    law_audit: dict[str, Any],
    comparison: dict[str, Any],
    output_path: Path = NEXT_QUEUE_PATH,
) -> dict[str, Any]:
    queue = build_control_surface_next_queue(
        atlas,
        hypotheses_payload,
        law_audit,
        comparison,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(queue, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return queue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write data/control_surface_next_experiment_queue.json")
    parser.add_argument("--json", action="store_true", help="print next-experiment queue JSON")
    args = parser.parse_args()

    atlas = load_json(ATLAS_PATH)
    hypotheses_payload = load_json(LAW_HYPOTHESES_PATH)
    law_audit = load_json(LAW_AUDIT_PATH)
    comparison = load_json(COMPARISON_PATH)
    queue = build_control_surface_next_queue(
        atlas,
        hypotheses_payload,
        law_audit,
        comparison,
    )

    if args.write:
        write_next_queue(atlas, hypotheses_payload, law_audit, comparison)
        print(
            f"wrote {NEXT_QUEUE_PATH.relative_to(ROOT).as_posix()} "
            f"with {queue['summary']['queue_item_count']} items"
        )
        return
    if args.json:
        print(json.dumps(queue, indent=2, sort_keys=True))
        return

    print(f"next queue ok: {queue['summary']['queue_item_count']} items")
    print("priority_counts:", json.dumps(queue["summary"]["priority_counts"], sort_keys=True))
    print("reason_counts:", json.dumps(queue["summary"]["reason_counts"], sort_keys=True))
    print("top_queue_ids:", json.dumps(queue["summary"]["top_queue_ids"]))


if __name__ == "__main__":
    main()
