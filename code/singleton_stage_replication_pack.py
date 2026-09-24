"""Build the singleton-stage replication pack for the control-surface atlas.

Axis interactions expose many singleton or sparse feature rules. This packet
turns the highest-risk singleton terminal stages into replication contracts:
what rows to add, which generated layer should move, and which result would
kill the tempting law.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json
from control_surface_axis_interactions import AXIS_INTERACTIONS_PATH
from control_surface_coverage_gaps import COVERAGE_GAPS_PATH
from control_surface_gate_geometry import GATE_GEOMETRY_PATH
from control_surface_offensive_doctrine import OFFENSIVE_DOCTRINE_PATH


SINGLETON_STAGE_PACK_PATH = ROOT / "data" / "singleton_stage_replication_pack.json"
SINGLETON_STAGE_PACK_REPORT_PATH = (
    ROOT / "research" / "prereg" / "SINGLETON_STAGE_REPLICATION_PACK.md"
)

WORK_ORDER_ID = "replicate_singleton_stage_laws"
TARGET_TERMINAL_STAGES = [
    "intervention_failed",
    "pre_signature_prompt_channel_locality",
    "reliability_null_boundary",
]
REQUIRED_REPLICATIONS_PER_STAGE = 2
TARGET_ROWS_BY_STAGE = {
    "intervention_failed": "mc001g_gemma_truth_agreement",
    "pre_signature_prompt_channel_locality": (
        "mc012_reliability_labeled_numeric_arbitration"
    ),
    "reliability_null_boundary": "mc005_associative_lookup",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def get_contract(doctrine: dict[str, Any]) -> dict[str, Any]:
    for contract in doctrine["branch_contracts"]:
        if contract["work_order_id"] == WORK_ORDER_ID:
            return contract
    raise AssertionError(f"missing offensive-doctrine contract {WORK_ORDER_ID}")


def feature_summaries_by_stage(
    axis_interactions: dict[str, Any],
    terminal_stage: str,
) -> list[dict[str, Any]]:
    return [
        summary
        for summary in axis_interactions["feature_summaries"]
        if summary.get("dominant_terminal_stage") == terminal_stage
        and summary.get("evidence_level") == "singleton"
    ]


def stage_entry(
    gate_geometry: dict[str, Any],
    terminal_stage: str,
) -> dict[str, Any]:
    entries = [
        entry
        for entry in gate_geometry["gate_entries"]
        if entry["terminal_stage"] == terminal_stage
    ]
    if len(entries) != 1:
        raise AssertionError(
            f"{terminal_stage} expected exactly one atlas row, got {len(entries)}"
        )
    return entries[0]


def replication_proposals_for_stage(terminal_stage: str) -> list[dict[str, Any]]:
    if terminal_stage == "pre_signature_prompt_channel_locality":
        return [
            {
                "proposal_id": "prompt_channel_locality_non_numeric_authority",
                "material_difference": (
                    "Non-numeric source-authority task with the visible channel "
                    "carried by source wording rather than trusted/untrusted labels."
                ),
                "target_stage_if_successful": terminal_stage,
                "minimum_evidence": [
                    "behavior contrast passes on source-disjoint holdout",
                    "direct local and learned controls pass",
                    "null rows pass",
                    "visible channel ablation or matching collapses the contrast",
                    "no hidden-state work is run after prompt-channel locality is shown",
                ],
                "promotion_rule": (
                    "Count as a replication only if the behavior contrast exists "
                    "and the prompt-channel locality control explains it."
                ),
                "kill_rule": (
                    "Do not count it for this stage if the contrast fails before "
                    "the prompt-channel control or if hidden-state work is needed "
                    "to diagnose the failure."
                ),
            },
            {
                "proposal_id": "prompt_channel_locality_answer_schema_authority",
                "material_difference": (
                    "Answer-schema or instruction-channel authority task where "
                    "the rule is visible through formatting rather than source status."
                ),
                "target_stage_if_successful": terminal_stage,
                "minimum_evidence": [
                    "same semantic task under at least two answer schemas",
                    "schema-visible positive control passes",
                    "schema-neutral or schema-matched control collapses the contrast",
                    "output/candidate baselines are reported",
                    "source-disjoint holdout remains balanced",
                ],
                "promotion_rule": (
                    "Count as a replication only if visible schema authority, "
                    "not hidden state, carries the behavior."
                ),
                "kill_rule": (
                    "Kill the prompt-channel-locality law if a schema-neutral "
                    "version preserves the contrast and licenses hidden-state work."
                ),
            },
        ]
    if terminal_stage == "intervention_failed":
        return [
            {
                "proposal_id": "leadtime_signature_intervention_failure",
                "material_difference": (
                    "A predecision monitor task outside truth/agreement where "
                    "the signature beats output, shuffle, and subgroup controls "
                    "before an intervention is attempted."
                ),
                "target_stage_if_successful": terminal_stage,
                "minimum_evidence": [
                    "behavior substrate passes",
                    "hidden signature survives holdout, shuffle, subgroup, same-stage output, and final-stage output controls",
                    "predeclared additive or patch intervention is attempted",
                    "intervention fails to move behavior in the predicted direction or creates side effects",
                    "failure is not reclassified as output-shadow or monitor-only",
                ],
                "promotion_rule": (
                    "Count as intervention_failed only if the signature gate "
                    "passes first and the predicted intervention then fails."
                ),
                "kill_rule": (
                    "Do not count output-confounded or shuffle-fragile signals as "
                    "intervention failures; those remain signature-stage failures."
                ),
            },
            {
                "proposal_id": "source_path_signature_intervention_failure",
                "material_difference": (
                    "A source-path or lookup-like task outside MC005 where a local "
                    "path signature passes but the first causal intervention fails."
                ),
                "target_stage_if_successful": terminal_stage,
                "minimum_evidence": [
                    "source-disjoint behavior substrate passes",
                    "source-path signature beats deletion, neutral rewrite, query-only, and output/candidate controls",
                    "local intervention is predeclared before observing effect size",
                    "primary direction fails or side rows/null rows fail",
                    "failed route is not patched repeatedly before classification",
                ],
                "promotion_rule": (
                    "Count as a replication only if a real signature reaches the "
                    "intervention gate and dies there."
                ),
                "kill_rule": (
                    "Kill the broad intervention-failed law if new rows die at "
                    "behavior substrate, prompt channel, or output geometry instead."
                ),
            },
        ]
    if terminal_stage == "reliability_null_boundary":
        return [
            {
                "proposal_id": "mc005_width_null_boundary_transfer",
                "material_difference": (
                    "Non-Qwen or prompt-family transfer of the MC005 bounded "
                    "reference, evaluated with null locality as a primary gate."
                ),
                "target_stage_if_successful": terminal_stage,
                "minimum_evidence": [
                    "answer-present primary effect reproduces",
                    "local or homologous intervention moves primary behavior",
                    "answer-absent null rows are stratified by margin",
                    "null flips or side effects persist after the planned transfer panel",
                    "transfer is bounded rather than promoted",
                ],
                "promotion_rule": (
                    "Count as a reliability-null-boundary replication only if "
                    "primary intervention works but null locality blocks promotion."
                ),
                "kill_rule": (
                    "Do not count failed primary transfer as a reliability boundary; "
                    "that is transfer-primary failure."
                ),
            },
            {
                "proposal_id": "new_synthetic_lookup_null_boundary",
                "material_difference": (
                    "A second synthetic or semi-synthetic source-value task with "
                    "a different answer interface and a predeclared local intervention."
                ),
                "target_stage_if_successful": terminal_stage,
                "minimum_evidence": [
                    "behavior table passes answer-present and answer-absent panels",
                    "internal/local path signature passes holdout and source controls",
                    "intervention moves answer-present rows in the predicted direction",
                    "answer-absent, side-row, robustness, or fluency gates remain bounded",
                    "ordinary repair budget is exhausted before classification",
                ],
                "promotion_rule": (
                    "Count as a replication only if the route reaches reliability "
                    "after a successful primary intervention."
                ),
                "kill_rule": (
                    "Kill the reliability-boundary law for this proposal if it dies "
                    "at signature, intervention, transfer, or output geometry instead."
                ),
            },
        ]
    raise AssertionError(f"unknown terminal stage {terminal_stage}")


def build_stage_targets(
    gate_geometry: dict[str, Any],
    axis_interactions: dict[str, Any],
) -> list[dict[str, Any]]:
    targets = []
    for stage in TARGET_TERMINAL_STAGES:
        entry = stage_entry(gate_geometry, stage)
        if entry["row_id"] != TARGET_ROWS_BY_STAGE[stage]:
            raise AssertionError(
                f"{stage} anchor drifted: expected {TARGET_ROWS_BY_STAGE[stage]}, "
                f"got {entry['row_id']}"
            )
        singleton_features = feature_summaries_by_stage(axis_interactions, stage)
        targets.append(
            {
                "terminal_stage": stage,
                "anchor_row_id": entry["row_id"],
                "anchor_title": entry.get("title", entry["family"]),
                "anchor_claim_boundary": entry.get(
                    "claim_boundary",
                    entry["claim_bar_action"],
                ),
                "current_count": 1,
                "required_new_rows": REQUIRED_REPLICATIONS_PER_STAGE,
                "singleton_feature_keys": [
                    feature["feature_key"] for feature in singleton_features
                ],
                "replication_proposals": replication_proposals_for_stage(stage),
                "stage_promotion_rule": (
                    "Promote the stage pattern only when at least three materially "
                    "distinct rows occupy the same terminal stage under the same "
                    "claimed predictor family."
                ),
                "stage_bound_rule": (
                    "Bound the pattern if the new rows replicate only inside one "
                    "behavior family, model family, or prompt contract."
                ),
                "stage_kill_rule": (
                    "Kill the broad law if two new rows under the same claimed "
                    "predictor land in different terminal stages."
                ),
            }
        )
    return targets


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    stage_targets = payload["stage_targets"]
    stages = [target["terminal_stage"] for target in stage_targets]
    proposal_counts = {
        target["terminal_stage"]: len(target["replication_proposals"])
        for target in stage_targets
    }
    missing_proposals = [
        stage
        for stage, count in proposal_counts.items()
        if count < REQUIRED_REPLICATIONS_PER_STAGE
    ]
    missing_rules = [
        proposal["proposal_id"]
        for target in stage_targets
        for proposal in target["replication_proposals"]
        if not (
            proposal["minimum_evidence"]
            and proposal["promotion_rule"]
            and proposal["kill_rule"]
        )
    ]
    checks = [
        {
            "id": "work_order_contract_matches_singleton_pack",
            "predicate": f"== {WORK_ORDER_ID}",
            "actual": payload["work_order"]["id"],
            "passed": payload["work_order"]["id"] == WORK_ORDER_ID,
            "why": "The pack must come from the offensive-doctrine law-replication contract.",
        },
        {
            "id": "target_terminal_stages_match_gap",
            "predicate": f"== {TARGET_TERMINAL_STAGES}",
            "actual": stages,
            "passed": stages == TARGET_TERMINAL_STAGES,
            "why": "The pack must target the current singleton terminal stages.",
        },
        {
            "id": "each_stage_has_two_replication_proposals",
            "predicate": "empty missing_proposals",
            "actual": proposal_counts,
            "passed": not missing_proposals,
            "why": "The exit condition requires at least two additional rows per singleton stage.",
        },
        {
            "id": "replication_proposals_have_decision_rules",
            "predicate": "empty list",
            "actual": missing_rules,
            "passed": not missing_rules,
            "why": "Every proposal must say what promotes it and what kills it.",
        },
        {
            "id": "no_law_promotion_claimed_by_pack",
            "predicate": "law_promotion_claimed == false",
            "actual": payload["claim_boundary"]["law_promotion_claimed"],
            "passed": payload["claim_boundary"]["law_promotion_claimed"] is False,
            "why": "This packet is a preregistration, not new law evidence.",
        },
        {
            "id": "singleton_gap_preserved",
            "predicate": "singleton_terminal_stage_count == 3",
            "actual": payload["source_snapshot"]["singleton_terminal_stage_count"],
            "passed": payload["source_snapshot"]["singleton_terminal_stage_count"] == 3,
            "why": "The pack must not pretend to have already reduced the singleton gap.",
        },
    ]
    return checks


def build_singleton_stage_replication_pack() -> dict[str, Any]:
    gate_geometry = load_json(GATE_GEOMETRY_PATH)
    axis_interactions = load_json(AXIS_INTERACTIONS_PATH)
    coverage_gaps = load_json(COVERAGE_GAPS_PATH)
    doctrine = load_json(OFFENSIVE_DOCTRINE_PATH)
    contract = get_contract(doctrine)
    gap_by_id = {gap["id"]: gap for gap in coverage_gaps["coverage_gaps"]}
    singleton_gap = gap_by_id["singleton_terminal_stage_evidence"]
    sparse_gap = gap_by_id["axis_rules_sparse_or_singleton_heavy"]
    stage_targets = build_stage_targets(gate_geometry, axis_interactions)
    payload = {
        "schema_version": 1,
        "updated_at": coverage_gaps.get("updated_at"),
        "purpose": (
            "Pre-register targeted row additions that can promote, bound, or "
            "kill singleton terminal-stage laws in the control-surface atlas."
        ),
        "sources": {
            "gate_geometry": rel(GATE_GEOMETRY_PATH),
            "axis_interactions": rel(AXIS_INTERACTIONS_PATH),
            "coverage_gaps": rel(COVERAGE_GAPS_PATH),
            "offensive_doctrine": rel(OFFENSIVE_DOCTRINE_PATH),
        },
        "work_order": {
            "id": contract["work_order_id"],
            "urgency": contract["urgency"],
            "track_type": contract["track_type"],
            "target_gap_ids": contract["target_gap_ids"],
            "first_artifact": contract["first_artifact"],
            "iteration_budget": contract["iteration_budget"],
            "promotion_rule": contract["promotion_rule"],
            "bound_rule": contract["bound_rule"],
            "kill_rule": contract["kill_rule"],
            "containment_rule": contract["containment_rule"],
            "export_rule": contract["export_rule"],
        },
        "source_snapshot": {
            "singleton_terminal_stage_count": len(
                singleton_gap["evidence"]["singleton_terminal_stages"]
            ),
            "singleton_terminal_rows": singleton_gap["evidence"][
                "singleton_terminal_rows"
            ],
            "singleton_terminal_stages": singleton_gap["evidence"][
                "singleton_terminal_stages"
            ],
            "feature_count": sparse_gap["evidence"]["feature_count"],
            "singleton_plus_sparse_count": sparse_gap["evidence"][
                "singleton_plus_sparse_count"
            ],
            "axis_evidence_level_counts": sparse_gap["evidence"][
                "evidence_level_counts"
            ],
        },
        "stage_targets": stage_targets,
        "summary": {
            "stage_target_count": len(stage_targets),
            "proposal_count": sum(
                len(target["replication_proposals"]) for target in stage_targets
            ),
            "proposal_counts_by_stage": dict(
                sorted(
                    Counter(
                        proposal["target_stage_if_successful"]
                        for target in stage_targets
                        for proposal in target["replication_proposals"]
                    ).items()
                )
            ),
            "target_terminal_stages": [target["terminal_stage"] for target in stage_targets],
        },
        "claim_boundary": {
            "law_promotion_claimed": False,
            "allowed_claim": (
                "This packet makes singleton terminal-stage law replication "
                "testable by naming anchor rows, required new rows, minimum "
                "evidence, and kill rules for each target stage."
            ),
            "forbidden_claim": (
                "This packet does not reduce the singleton count, does not "
                "promote any predictive law, and does not add mechanism evidence."
            ),
        },
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_singleton_stage_replication_pack(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("singleton stage pack schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"singleton stage pack source missing: {rel_path}")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"singleton stage pack checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    work_order = payload["work_order"]
    snapshot = payload["source_snapshot"]
    lines = [
        "# Singleton Stage Replication Pack",
        "",
        "Date: 2026-07-01",
        "",
        "Status: preregistered law-replication packet; no law promotion claimed.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/singleton_stage_replication_pack.json`",
        "",
        "Builder:",
        "",
        "> `code/singleton_stage_replication_pack.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\singleton_stage_replication_pack.py --write",
        "python code\\singleton_stage_replication_pack.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Target Gap",
        "",
        f"- work order: `{work_order['id']}`;",
        f"- urgency: `{work_order['urgency']}`;",
        f"- track: `{work_order['track_type']}`;",
        f"- target gaps: `{format_value(work_order['target_gap_ids'])}`;",
        f"- iteration budget: {work_order['iteration_budget']}.",
        "",
        "## Current Singleton Shape",
        "",
        f"- singleton terminal stages: `{format_value(snapshot['singleton_terminal_stages'])}`;",
        f"- singleton terminal rows: `{format_value(snapshot['singleton_terminal_rows'])}`;",
        f"- singleton+sparse feature summaries: {snapshot['singleton_plus_sparse_count']};",
        f"- axis evidence levels: `{format_value(snapshot['axis_evidence_level_counts'])}`.",
        "",
        "## Stage Targets",
        "",
        "| Stage | Anchor Row | Current Singleton Feature Keys | Required New Rows | Stage Kill Rule |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for target in payload["stage_targets"]:
        lines.append(
            f"| `{target['terminal_stage']}` | `{target['anchor_row_id']}` | "
            f"`{format_value(target['singleton_feature_keys'])}` | "
            f"{target['required_new_rows']} | {target['stage_kill_rule']} |"
        )

    lines.extend(["", "## Replication Proposals", ""])
    for target in payload["stage_targets"]:
        lines.extend(["", f"### `{target['terminal_stage']}`", ""])
        for proposal in target["replication_proposals"]:
            lines.extend(
                [
                    f"- proposal: `{proposal['proposal_id']}`;",
                    f"- material difference: {proposal['material_difference']}",
                    f"- minimum evidence: `{format_value(proposal['minimum_evidence'])}`;",
                    f"- promotion rule: {proposal['promotion_rule']}",
                    f"- kill rule: {proposal['kill_rule']}",
                ]
            )

    lines.extend(
        [
            "",
            "## Decision Rules",
            "",
            f"- promotion rule: {work_order['promotion_rule']}",
            f"- bound rule: {work_order['bound_rule']}",
            f"- kill rule: {work_order['kill_rule']}",
            f"- containment rule: {work_order['containment_rule']}",
            f"- export rule: {work_order['export_rule']}",
            "",
            "## Claim Boundary",
            "",
            payload["claim_boundary"]["allowed_claim"],
            "",
            payload["claim_boundary"]["forbidden_claim"],
            "",
            "## What This Proves",
            "",
            "It proves that singleton-stage law replication is now a concrete",
            "test plan rather than loose language. Each target stage has two",
            "materially distinct proposed rows and a predeclared failure mode.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not promote a law, close the singleton gap, or add a mechanism",
            "claim. The current singleton count remains the current singleton count",
            "until new rows land in the generated atlas.",
            "",
        ]
    )
    return "\n".join(lines)


def write_singleton_stage_replication_pack(
    output_path: Path = SINGLETON_STAGE_PACK_PATH,
    report_path: Path = SINGLETON_STAGE_PACK_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_singleton_stage_replication_pack()
    validate_singleton_stage_replication_pack(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write singleton-stage replication artifacts",
    )
    parser.add_argument("--json", action="store_true", help="print pack JSON")
    args = parser.parse_args()

    payload = build_singleton_stage_replication_pack()
    validate_singleton_stage_replication_pack(payload)

    if args.write:
        write_singleton_stage_replication_pack()
        print(
            f"wrote {SINGLETON_STAGE_PACK_PATH.relative_to(ROOT).as_posix()} and "
            f"{SINGLETON_STAGE_PACK_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['proposal_count']} proposals"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "singleton stage replication pack ok: "
        f"{payload['summary']['stage_target_count']} stages, "
        f"{payload['summary']['proposal_count']} proposals"
    )
    print(
        "target_terminal_stages:",
        json.dumps(payload["summary"]["target_terminal_stages"], sort_keys=True),
    )


if __name__ == "__main__":
    main()
