"""Build the gap-closure plan for the control-surface genome.

Coverage gaps say what the atlas cannot yet support. The next queue says
which tests are attractive. This layer turns both into decision-bound work
orders with promotion, bounded-claim, kill, containment, and export rules.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json


GAP_CLOSURE_PLAN_PATH = ROOT / "data" / "control_surface_gap_closure_plan.json"
GAP_CLOSURE_PLAN_REPORT_PATH = (
    ROOT / "research" / "38_CONTROL_SURFACE_GAP_CLOSURE_PLAN.md"
)

COVERAGE_GAPS_PATH = ROOT / "data" / "control_surface_coverage_gaps.json"
NEXT_QUEUE_PATH = ROOT / "data" / "control_surface_next_experiment_queue.json"
GENOME_SNAPSHOT_PATH = ROOT / "data" / "control_surface_genome_snapshot.json"
RELIABILITY_MATRIX_PATH = ROOT / "data" / "control_surface_reliability_matrix.json"
TRANSFER_MATRIX_PATH = ROOT / "data" / "control_surface_transfer_matrix.json"
BRIDGE_LADDER_PATH = ROOT / "data" / "control_surface_bridge_ladder.json"
AXIS_INTERACTIONS_PATH = ROOT / "data" / "control_surface_axis_interactions.json"

CRITICAL_GAP_IDS = {
    "clean_intervention_absent",
    "promoted_mechanism_absent",
    "full_reliability_absent",
    "transfer_ready_mechanism_absent",
}

REQUIRED_DECISION_FIELDS = {
    "promotion_rule",
    "bound_rule",
    "kill_rule",
    "containment_rule",
    "export_rule",
}

SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "watch": 3,
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def by_id(items: list[dict[str, Any]], key: str = "id") -> dict[str, dict[str, Any]]:
    return {item[key]: item for item in items}


def compact_queue_refs(
    queue_by_id: dict[str, dict[str, Any]],
    queue_ids: list[str],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for queue_id in queue_ids:
        item = queue_by_id[queue_id]
        refs.append(
            {
                "id": item["id"],
                "priority_class": item["priority_class"],
                "priority_score": item["priority_score"],
                "hypothesis_id": item["hypothesis_id"],
                "action_type": item["action_type"],
                "reason_codes": item["reason_codes"],
                "next_test": item["next_test"],
            }
        )
    return refs


def gap_refs(
    gaps_by_id: dict[str, dict[str, Any]],
    gap_ids: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "id": gap_id,
            "severity": gaps_by_id[gap_id]["severity"],
            "gap_type": gaps_by_id[gap_id]["gap_type"],
            "title": gaps_by_id[gap_id]["title"],
            "exit_condition": gaps_by_id[gap_id]["exit_condition"],
        }
        for gap_id in gap_ids
    ]


def make_work_order(
    order: int,
    work_id: str,
    title: str,
    urgency: str,
    track_type: str,
    primary_gap_ids: list[str],
    secondary_gap_ids: list[str],
    source_rows: list[str],
    queue_refs: list[dict[str, Any]],
    closure_question: str,
    required_evidence: list[str],
    decision_rules: dict[str, str],
    forbidden_moves: list[str],
    expected_atlas_change: str,
    first_artifact: str,
    iteration_budget: str,
    evidence_anchor: str,
) -> dict[str, Any]:
    missing_rules = REQUIRED_DECISION_FIELDS - set(decision_rules)
    if missing_rules:
        raise AssertionError(f"{work_id}: missing decision rules {missing_rules}")
    return {
        "order": order,
        "id": work_id,
        "title": title,
        "urgency": urgency,
        "track_type": track_type,
        "primary_gap_ids": primary_gap_ids,
        "secondary_gap_ids": secondary_gap_ids,
        "source_rows": source_rows,
        "queue_refs": queue_refs,
        "closure_question": closure_question,
        "required_evidence": required_evidence,
        "decision_rules": decision_rules,
        "forbidden_moves": forbidden_moves,
        "expected_atlas_change": expected_atlas_change,
        "first_artifact": first_artifact,
        "iteration_budget": iteration_budget,
        "evidence_anchor": evidence_anchor,
    }


def build_work_orders(
    coverage_gaps: dict[str, Any],
    next_queue: dict[str, Any],
    genome_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    gaps_by_id = by_id(coverage_gaps["coverage_gaps"])
    queue_by_id = by_id(next_queue["queue"])
    top_queue_ids = genome_snapshot["summary"]["top_queue_ids"]

    return [
        make_work_order(
            1,
            "close_mc005_reference_specimen",
            "Force a verdict on the MC005 internal-causal reference specimen.",
            "immediate",
            "deepening_closeout",
            [
                "promoted_mechanism_absent",
                "full_reliability_absent",
                "clean_intervention_absent",
            ],
            [
                "mc005_singleton_internal_causal_reference",
                "singleton_terminal_stage_evidence",
                "transfer_ready_mechanism_absent",
            ],
            ["mc005_associative_lookup"],
            compact_queue_refs(queue_by_id, [top_queue_ids[4]]),
            (
                "Is MC005 a promotable mechanism, a bounded mechanism with an "
                "intrinsic null boundary, or a failed write-replacement route?"
            ),
            [
                "Freeze high-margin lookup rows and answer-absent null rows before repair.",
                "Report lookup mediation, answer-absent flips, locality, fluency, side rows, and margin strata together.",
                "Compare write replacement against at least one materially different local intervention family.",
                "Record source-disjoint and layout/lexicon/pair-count holdouts without moving the goalposts.",
            ],
            {
                "promotion_rule": (
                    "Promote only if lookup mediation remains strong and "
                    "answer-absent null flips disappear or fall below a "
                    "predeclared trivial threshold across locality, side-row, "
                    "and holdout panels."
                ),
                "bound_rule": (
                    "Bound if source-value mediation remains high-margin and "
                    "localized but answer-absent low-margin rows continue to "
                    "flip after the planned repair attempts."
                ),
                "kill_rule": (
                    "Kill the write-replacement route if null flips or side "
                    "effects persist across two materially different local "
                    "intervention variants."
                ),
                "containment_rule": (
                    "The surviving claim may say source-value attention-write "
                    "mediation exists in the tested lookup contract; it may not "
                    "say full reliability, transfer, or general knowledge control."
                ),
                "export_rule": (
                    "Export NULL_ROW_LOW_MARGIN_FLIP and MODEL_SIZE_NULL_FRAGILITY "
                    "as named reliability diagnostics if promotion fails."
                ),
            },
            [
                "Do not add unlimited V-number repair attempts.",
                "Do not average null rows into primary lookup success.",
                "Do not treat coarse source deletion as circuit locality.",
            ],
            (
                "Either one atlas row moves to promoted, MC005 remains bounded "
                "with a sharper reliability boundary, or write replacement is "
                "closed as a failed intervention family."
            ),
            "research/prereg/MC005_WRITE_REPLACEMENT_CLOSEOUT.md",
            "Two serious repair variants, then promote, bound, or kill.",
            "MC005 is the sole bounded internal-causal specimen and the sole reliability-boundary row.",
        ),
        make_work_order(
            2,
            "close_post_mc033_bridge_substrate_family",
            "Close the post-MC033 bridge substrate family before any hidden-state work.",
            "immediate",
            "bridge_substrate",
            [
                "bridge_hidden_state_not_licensed",
                "clean_intervention_absent",
                "promoted_mechanism_absent",
            ],
            [
                "domain_coverage_knowledge_bridge_dominates_failures",
                "full_reliability_absent",
                "output_geometry_pressure_mixed",
            ],
            [
                "mc007_semi_synthetic_familiar_entity_lookup",
                "mc008_symbolic_fact_code_arbitration",
                "mc009_derived_code_arbitration",
                "mc010_two_hop_fact_code_arbitration",
                "mc011_atomic_number_code_arbitration",
                "mc012_reliability_labeled_numeric_arbitration",
                "mc013_status_channel_ablation_numeric_arbitration",
                "mc014_inferred_reliability_numeric_arbitration",
                "mc015_parity_gated_numeric_arbitration",
                "mc016_alphabet_gated_numeric_arbitration",
            ],
            compact_queue_refs(queue_by_id, top_queue_ids[:2]),
            (
                "Can a knowledge-like bridge pass direct controls, conflict "
                "mixture, nulls, source-disjoint holdout, prompt-channel "
                "locality, and output/candidate baselines after source labels, "
                "row codes, operation handles, examples, answer interfaces, "
                "absence guards, checksum reliability cues, cross-table "
                "consistency cues, and row-local fact-claim cues have failed?"
            ),
            [
                "Record MC031-MC033 as a same-family bridge closure sequence.",
                "Do not run hidden-state probes on MC031, MC032, or MC033.",
                "Require any future bridge to be materially outside visible status labels, row codes, operation handles, worked examples, answer schemas, absence guards, checksum, cross-table consistency, and row-local fact claims.",
                "If a future bridge is proposed, require MC012-level direct controls, conflict mixture, nulls, source-disjoint holdout, prompt-channel locality, and candidate/output baselines before hidden-state work.",
            ],
            {
                "promotion_rule": (
                    "Promote a future bridge substrate to hidden-state work only "
                    "if it is materially outside MC007-MC033 and branch, null, "
                    "local, side-number, parseability, prompt-channel, "
                    "source-disjoint, and output/candidate controls pass together."
                ),
                "bound_rule": (
                    "Bound the current route as a bridge diagnostic family: "
                    "direct controls and nulls can remain clean while learned "
                    "branch routing fails across statusless source-validity cues."
                ),
                "kill_rule": (
                    "Kill same-family repairs after MC033; do not add another "
                    "source-validity cue unless it changes the substrate class, "
                    "not just the wording of the reliability cue."
                ),
                "containment_rule": (
                    "The surviving claim is a behavior-contract diagnostic about "
                    "bridge failure modes; no hidden signature or mechanism claim "
                    "is licensed."
                ),
                "export_rule": (
                    "Export POST_MC032_BRIDGE_ROUTE_CLOSED, "
                    "FACT_CLAIM_MISMATCH_LOCAL_AND_CLAIM_LEAK, and "
                    "STATUSLESS_SOURCE_VALIDITY_LOCAL_DOMINANCE as diagnostics."
                ),
            },
            [
                "Do not reopen source labels, row codes, query-operation handles, worked examples, constrained choices, numeric options, answer-interface sweeps, absence guards, arithmetic checksum cues, simple cross-table consistency cues, or row-local fact-claim cues.",
                "Do not start probes on another behavior-substrate failure.",
                "Do not call a prompt-visible positive control a knowledge mechanism.",
            ],
            (
                "The bridge ladder records the MC031-MC033 statusless source-"
                "validity closure sequence and keeps hidden-state work forbidden "
                "until a materially new bridge substrate exists."
            ),
            "research/cards/MC033_FACT_CLAIM_BRIDGE_CLOSEOUT_STATUS.md",
            "Route closed after MC033 unless a new bridge substrate class is preregistered.",
            "The bridge ladder has 24 rungs, 0 hidden-state-allowed rungs, and recent MC030-MC033 closures.",
        ),
        make_work_order(
            3,
            "close_mc006_predecision_frontier",
            "Turn MC006 final-margin failure into a decision-timing result.",
            "high",
            "knowledge_frontier_closeout",
            [
                "clean_intervention_absent",
                "mc006_knowledge_route_monitor_only",
                "output_geometry_pressure_mixed",
            ],
            [
                "promoted_mechanism_absent",
                "axis_rules_sparse_or_singleton_heavy",
            ],
            ["mc006_parametric_fact_override"],
            compact_queue_refs(queue_by_id, [top_queue_ids[4]]),
            (
                "Is MC006 blocked because the current signatures are output "
                "shadows, or is there an earlier source/path window with "
                "control-surviving lead time?"
            ),
            [
                "Probe source-token, country-token, mapping-value, and post-context pre-question positions separately.",
                "Residualize or match candidate-score and next-token margins before signature claims.",
                "Report final-margin failure as a timing result, not merely a dead probe.",
                "Attempt intervention only after a same-stage and final-stage margin control is survived.",
            ],
            {
                "promotion_rule": (
                    "Promote to intervention only if an earlier signal predicts "
                    "the behavior on source-disjoint holdout while beating "
                    "same-stage output geometry, final candidate margin, and "
                    "shuffle controls."
                ),
                "bound_rule": (
                    "Bound if early signals exist but final-margin geometry "
                    "still explains the actionable decision at intervention time."
                ),
                "kill_rule": (
                    "Kill final-prompt-token probing for MC006 if another "
                    "margin-matched pass remains candidate-score or final-margin explained."
                ),
                "containment_rule": (
                    "The allowed claim becomes decision timing for capital-fact "
                    "override, not a truth or knowledge vector."
                ),
                "export_rule": (
                    "Export FINAL_STATE_OUTPUT_VISIBLE or PREDECISION_MONITOR_NO_LEVER "
                    "depending on where the failure lands."
                ),
            },
            [
                "Do not steer V15/V16-style final-position perfect AUCs.",
                "Do not treat final margin as a nuisance if it is the faithful downstream decision state.",
                "Do not use lonely AUCs without same-stage and final-stage baselines.",
            ],
            (
                "MC006 either gets a hidden-state-allowed predecision route or "
                "is closed as a monitor-only/final-output-visible knowledge branch."
            ),
            "research/prereg/MC006_PREDECISION_FRONTIER_CLOSEOUT.md",
            "One margin-matched early-position pass; no final-position reruns unless they change the control geometry.",
            "MC006 is the current knowledge-like monitor-only route with repeated final-margin and candidate-score blocks.",
        ),
        make_work_order(
            4,
            "run_width_transfer_probe",
            "Make transfer failure or success measured instead of assumed.",
            "immediate",
            "widening_probe",
            [
                "transfer_ready_mechanism_absent",
                "model_family_coverage_qwen_dominant",
                "full_reliability_absent",
            ],
            [
                "mc005_singleton_internal_causal_reference",
                "axis_rules_sparse_or_singleton_heavy",
            ],
            [
                "mc005_associative_lookup",
                "mc003_delayed_copy",
                "mc004_in_context_binding",
            ],
            compact_queue_refs(queue_by_id, top_queue_ids[2:4]),
            (
                "Do primary effects, null locality, side rows, and prompt "
                "robustness transfer together, or does transfer fail at "
                "reliability before primary effect?"
            ),
            [
                "Choose one bounded/reference row and one monitor/output-shadow row.",
                "Pre-register null panels and side rows before measuring primary effects.",
                "Run at least one materially comparable non-Qwen model-family boundary test.",
                "Update transfer fields only if null locality and widening evidence move together.",
            ],
            {
                "promotion_rule": (
                    "Promote a transfer claim only if primary effect, null "
                    "locality, side effects, prompt-contract robustness, and "
                    "holdouts all transfer."
                ),
                "bound_rule": (
                    "Bound if primary effects reproduce but null locality, side "
                    "rows, or prompt robustness fail."
                ),
                "kill_rule": (
                    "Kill the transfer route for a surface if two model-family "
                    "or prompt-family tests reproduce the same reliability "
                    "failure."
                ),
                "containment_rule": (
                    "A primary-effect replication is not a transferred mechanism "
                    "unless reliability and null fields move beyond bounded."
                ),
                "export_rule": (
                    "Export TRANSFER_PRIMARY_BEFORE_RELIABILITY or TRANSFER_PRIMARY_FAILED "
                    "as the widening diagnostic."
                ),
            },
            [
                "Do not celebrate primary effect replication without null panels.",
                "Do not mark transfer on rows whose atlas transfer and null_locality fields remain bounded or low.",
                "Do not use Qwen-only widening as small-model generalization.",
            ],
            (
                "The transfer matrix gains a real widening datum and either "
                "supports or falsifies the reliability-before-primary transfer law."
            ),
            "research/prereg/TRANSFER_WIDTH_PROBE_MC005_MC003_MC004.md",
            "One bounded/reference row plus one diagnostic row before more deepening.",
            "The current transfer matrix has 14 untested rows and 0 transfer-ready mechanisms.",
        ),
        make_work_order(
            5,
            "replicate_singleton_stage_laws",
            "Turn singleton terminal-stage laws into supported or killed laws.",
            "high",
            "law_replication",
            [
                "axis_rules_sparse_or_singleton_heavy",
                "singleton_terminal_stage_evidence",
                "output_geometry_pressure_mixed",
            ],
            [
                "domain_coverage_knowledge_bridge_dominates_failures",
                "model_family_coverage_qwen_dominant",
            ],
            [
                "mc001g_gemma_truth_agreement",
                "mc005_associative_lookup",
                "mc012_reliability_labeled_numeric_arbitration",
            ],
            compact_queue_refs(queue_by_id, [top_queue_ids[0], top_queue_ids[1]]),
            (
                "Which terminal-stage patterns are stable laws rather than "
                "single-row labels?"
            ),
            [
                "Add at least two materially distinct rows for prompt-channel locality, intervention failure, and reliability-null boundary stages.",
                "Require every new row to include the same terminal-stage features used by the axis-interaction builder.",
                "Keep output-shadow, monitor-only, and failed-intervention rows separated instead of merging them into one output-geometry story.",
            ],
            {
                "promotion_rule": (
                    "Promote a law only when a feature remains pure or nearly "
                    "pure across at least three materially distinct rows."
                ),
                "bound_rule": (
                    "Bound if the feature is predictive inside one behavior "
                    "family but mixed across families."
                ),
                "kill_rule": (
                    "Kill a proposed law if two new rows land in different "
                    "terminal stages under the same claimed predictor."
                ),
                "containment_rule": (
                    "Singleton features stay calibration labels, not predictive "
                    "laws."
                ),
                "export_rule": (
                    "Export the surviving or killed feature into the axis "
                    "interaction map with evidence_level preserved."
                ),
            },
            [
                "Do not write broad laws from singleton terminal stages.",
                "Do not collapse output-shadow, monitor-only, and failed-intervention cases.",
                "Do not hide sample size behind purity.",
            ],
            (
                "Axis interactions gain fewer singleton/sparse features or the "
                "project explicitly kills a tempting rule."
            ),
            "research/prereg/SINGLETON_STAGE_REPLICATION_PACK.md",
            "Three targeted row additions before new law language is allowed.",
            "Axis interactions currently include 82 singleton-or-sparse feature summaries.",
        ),
        make_work_order(
            6,
            "enforce_offensive_doctrine_harness",
            "Make every future branch pay rent against a named gap.",
            "high",
            "process_harness",
            [
                "promoted_mechanism_absent",
                "clean_intervention_absent",
                "full_reliability_absent",
                "transfer_ready_mechanism_absent",
            ],
            [
                "axis_rules_sparse_or_singleton_heavy",
                "bridge_hidden_state_not_licensed",
            ],
            [],
            compact_queue_refs(queue_by_id, top_queue_ids),
            (
                "Can the research loop stop accumulating attractive failures "
                "and instead force every line to promote, bound, kill, or export?"
            ),
            [
                "Require every new preregistration to name target coverage gaps.",
                "Require promote, bound, kill, containment, and export rules before execution.",
                "Require the validator to expose which generated layer would change after the run.",
                "Record typed failures as first-class atlas updates rather than chat-local judgment.",
            ],
            {
                "promotion_rule": (
                    "A branch continues only if it changes a named gap, atlas "
                    "field, reliability class, transfer class, or gate-geometry cell."
                ),
                "bound_rule": (
                    "A branch may remain active only with an explicit narrower "
                    "claim and a fixed remaining iteration budget."
                ),
                "kill_rule": (
                    "Kill any branch that repeats the same diagnostic class "
                    "after the predeclared repair budget."
                ),
                "containment_rule": (
                    "No result bypasses the mechanism-card gates by being "
                    "interesting, high-AUC, or aesthetically plausible."
                ),
                "export_rule": (
                    "Every closure emits one of promoted mechanism card, bounded "
                    "mechanism card, failed mechanism card, or diagnostic note."
                ),
            },
            [
                "Do not start work without a target gap and death rule.",
                "Do not keep half-alive branches open indefinitely.",
                "Do not substitute review prose for generated evidence.",
            ],
            (
                "Future work is judged by whether it changes generated map "
                "geometry, not by whether it adds another isolated story."
            ),
            "research/39_CONTROL_SURFACE_OFFENSIVE_DOCTRINE.md",
            "Applies to every new branch immediately.",
            "The atlas has a next queue and coverage gaps, but previously lacked a closure-decision layer.",
        ),
    ]


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    work_orders = payload["work_orders"]
    all_gap_ids = set(payload["coverage_gap_ids"])
    covered_gap_ids = {
        gap_id
        for order in work_orders
        for gap_id in order["primary_gap_ids"] + order["secondary_gap_ids"]
    }
    critical_missing = sorted(CRITICAL_GAP_IDS - covered_gap_ids)
    uncovered_gaps = sorted(all_gap_ids - covered_gap_ids)
    rule_failures = [
        order["id"]
        for order in work_orders
        if REQUIRED_DECISION_FIELDS - set(order["decision_rules"])
    ]
    empty_kill_rules = [
        order["id"]
        for order in work_orders
        if not order["decision_rules"].get("kill_rule")
    ]
    top_queue_ids = set(payload["top_queue_ids"])
    covered_queue_ids = {
        ref["id"] for order in work_orders for ref in order["queue_refs"]
    }
    uncovered_top_queue = sorted(top_queue_ids - covered_queue_ids)
    return [
        {
            "id": "all_critical_gaps_have_work_orders",
            "predicate": "empty list",
            "actual": critical_missing,
            "passed": not critical_missing,
            "why": "The plan must directly cover every critical gap.",
        },
        {
            "id": "all_coverage_gaps_are_covered",
            "predicate": "empty list",
            "actual": uncovered_gaps,
            "passed": not uncovered_gaps,
            "why": "A closure plan should route every current gap to at least one work order.",
        },
        {
            "id": "every_work_order_has_decision_rules",
            "predicate": "empty list",
            "actual": rule_failures,
            "passed": not rule_failures,
            "why": "Every line needs promote, bound, kill, containment, and export rules.",
        },
        {
            "id": "every_work_order_has_kill_rule",
            "predicate": "empty list",
            "actual": empty_kill_rules,
            "passed": not empty_kill_rules,
            "why": "The plan is meant to prevent indefinite low-intensity branch survival.",
        },
        {
            "id": "top_queue_items_are_covered",
            "predicate": "empty list",
            "actual": uncovered_top_queue,
            "passed": not uncovered_top_queue,
            "why": "The closure plan should absorb the current highest-priority next queue.",
        },
        {
            "id": "does_not_claim_completed_genome",
            "predicate": "promoted == 0 and hidden_state_bridge == 0",
            "actual": {
                "promoted_mechanism_count": payload["source_snapshot"][
                    "promoted_mechanism_count"
                ],
                "hidden_state_allowed_bridge_count": payload["source_snapshot"][
                    "hidden_state_allowed_bridge_count"
                ],
            },
            "passed": (
                payload["source_snapshot"]["promoted_mechanism_count"] == 0
                and payload["source_snapshot"]["hidden_state_allowed_bridge_count"] == 0
            ),
            "why": "This is a closure plan for missing claims, not a success declaration.",
        },
    ]


def build_control_surface_gap_closure_plan() -> dict[str, Any]:
    coverage_gaps = load_json(COVERAGE_GAPS_PATH)
    next_queue = load_json(NEXT_QUEUE_PATH)
    genome_snapshot = load_json(GENOME_SNAPSHOT_PATH)
    reliability_matrix = load_json(RELIABILITY_MATRIX_PATH)
    transfer_matrix = load_json(TRANSFER_MATRIX_PATH)
    bridge_ladder = load_json(BRIDGE_LADDER_PATH)
    axis_interactions = load_json(AXIS_INTERACTIONS_PATH)

    gaps_by_id = by_id(coverage_gaps["coverage_gaps"])
    work_orders = build_work_orders(coverage_gaps, next_queue, genome_snapshot)
    covered_gap_ids = sorted(
        {
            gap_id
            for order in work_orders
            for gap_id in order["primary_gap_ids"] + order["secondary_gap_ids"]
        }
    )
    urgency_counts = dict(sorted(Counter(order["urgency"] for order in work_orders).items()))
    track_type_counts = dict(
        sorted(Counter(order["track_type"] for order in work_orders).items())
    )
    primary_gap_coverage = dict(
        sorted(
            Counter(
                gap_id
                for order in work_orders
                for gap_id in order["primary_gap_ids"]
            ).items()
        )
    )
    ordered_gap_refs = gap_refs(gaps_by_id, sorted(gaps_by_id))
    payload = {
        "schema_version": 1,
        "updated_at": coverage_gaps.get("updated_at"),
        "purpose": (
            "Convert the validated coverage gaps and next queue into concrete "
            "closure work orders with promote, bound, kill, containment, and "
            "export rules."
        ),
        "sources": {
            "coverage_gaps": rel(COVERAGE_GAPS_PATH),
            "next_queue": rel(NEXT_QUEUE_PATH),
            "genome_snapshot": rel(GENOME_SNAPSHOT_PATH),
            "reliability_matrix": rel(RELIABILITY_MATRIX_PATH),
            "transfer_matrix": rel(TRANSFER_MATRIX_PATH),
            "bridge_ladder": rel(BRIDGE_LADDER_PATH),
            "axis_interactions": rel(AXIS_INTERACTIONS_PATH),
        },
        "source_snapshot": {
            "promoted_mechanism_count": genome_snapshot["summary"][
                "promoted_mechanism_count"
            ],
            "bounded_mechanism_count": genome_snapshot["summary"][
                "bounded_mechanism_count"
            ],
            "hidden_state_allowed_bridge_count": bridge_ladder["summary"][
                "hidden_state_allowed_count"
            ],
            "clean_unconfounded_bridge_count": bridge_ladder["summary"][
                "clean_unconfounded_bridge_count"
            ],
            "full_reliability_count": reliability_matrix["summary"][
                "full_reliability_count"
            ],
            "transfer_ready_mechanism_count": transfer_matrix["summary"][
                "transfer_ready_mechanism_count"
            ],
            "singleton_plus_sparse_feature_count": (
                axis_interactions["summary"]["evidence_level_counts"].get(
                    "singleton",
                    0,
                )
                + axis_interactions["summary"]["evidence_level_counts"].get(
                    "sparse",
                    0,
                )
            ),
        },
        "coverage_gap_ids": sorted(gaps_by_id),
        "coverage_gap_refs": ordered_gap_refs,
        "top_queue_ids": genome_snapshot["summary"]["top_queue_ids"],
        "summary": {
            "work_order_count": len(work_orders),
            "covered_gap_count": len(covered_gap_ids),
            "coverage_gap_count": len(gaps_by_id),
            "critical_gap_count": len(CRITICAL_GAP_IDS),
            "critical_gap_coverage_count": len(
                CRITICAL_GAP_IDS.intersection(covered_gap_ids)
            ),
            "urgency_counts": urgency_counts,
            "track_type_counts": track_type_counts,
            "primary_gap_coverage": primary_gap_coverage,
            "top_queue_coverage_count": len(
                {
                    ref["id"]
                    for order in work_orders
                    for ref in order["queue_refs"]
                }.intersection(genome_snapshot["summary"]["top_queue_ids"])
            ),
        },
        "operating_doctrine": [
            "Every new branch names a target coverage gap before execution.",
            "Every branch predeclares promotion, bounded-claim, kill, containment, and export rules.",
            "A typed failure is a valid output only if it changes the atlas, taxonomy, gap map, or closure plan.",
            "Deepening work must be interrupted by widening tests when a singleton boundary starts carrying broad language.",
            "Hidden-state work remains forbidden on bridge rows until behavior substrate, null, prompt-channel, holdout, and output/candidate gates pass together.",
        ],
        "work_orders": work_orders,
        "allowed_claim": (
            "The project now has a generated closure plan that routes every "
            "current coverage gap and top queue item to a decision-bound work order."
        ),
        "forbidden_claim": (
            "This does not close any gap by itself. It only specifies what "
            "evidence would close, bound, kill, or export each next line."
        ),
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_gap_closure_plan(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("gap closure plan schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"gap closure plan source missing: {rel_path}")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check["passed"]
    ]
    if failed_checks:
        raise AssertionError(f"gap closure plan checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    source_snapshot = payload["source_snapshot"]
    lines = [
        "# Control-Surface Gap Closure Plan",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated closure-plan layer implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_gap_closure_plan.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_gap_closure_plan.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_gap_closure_plan.py --write",
        "python code\\control_surface_gap_closure_plan.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This layer converts negative space into work orders. It does not add a",
        "new mechanism claim. It states how the current critical gaps can be",
        "closed, bounded, killed, or exported as reusable diagnostics.",
        "",
        "## Generated Facts",
        "",
        f"- work orders: {summary['work_order_count']};",
        f"- covered gaps: {summary['covered_gap_count']}/{summary['coverage_gap_count']};",
        f"- critical gap coverage: {summary['critical_gap_coverage_count']}/{summary['critical_gap_count']};",
        f"- urgency counts: `{format_value(summary['urgency_counts'])}`;",
        f"- track type counts: `{format_value(summary['track_type_counts'])}`;",
        f"- top queue coverage: {summary['top_queue_coverage_count']}/{len(payload['top_queue_ids'])};",
        f"- promoted mechanisms now: {source_snapshot['promoted_mechanism_count']};",
        f"- hidden-state-allowed bridge rungs now: {source_snapshot['hidden_state_allowed_bridge_count']};",
        f"- full-reliability rows now: {source_snapshot['full_reliability_count']}.",
        "",
        "## Operating Doctrine",
        "",
    ]
    for doctrine in payload["operating_doctrine"]:
        lines.append(f"- {doctrine}")

    lines.extend(
        [
            "",
            "## Work Orders",
            "",
            "| Order | Work Order | Urgency | Primary Gaps | Closure Question |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for order in payload["work_orders"]:
        lines.append(
            f"| {order['order']} | `{order['id']}` | `{order['urgency']}` | "
            f"`{', '.join(order['primary_gap_ids'])}` | {order['closure_question']} |"
        )

    for order in payload["work_orders"]:
        lines.extend(
            [
                "",
                f"## {order['order']}. {order['title']}",
                "",
                f"- track type: `{order['track_type']}`;",
                f"- source rows: `{format_value(order['source_rows'])}`;",
                f"- first artifact: `{order['first_artifact']}`;",
                f"- iteration budget: {order['iteration_budget']}",
                f"- evidence anchor: {order['evidence_anchor']}",
                f"- expected atlas change: {order['expected_atlas_change']}",
                "",
                "Required evidence:",
            ]
        )
        for item in order["required_evidence"]:
            lines.append(f"- {item}")
        lines.extend(["", "Decision rules:"])
        for key in [
            "promotion_rule",
            "bound_rule",
            "kill_rule",
            "containment_rule",
            "export_rule",
        ]:
            lines.append(f"- `{key}`: {order['decision_rules'][key]}")
        lines.extend(["", "Forbidden moves:"])
        for item in order["forbidden_moves"]:
            lines.append(f"- {item}")
        lines.extend(["", "Queue references:"])
        for ref in order["queue_refs"]:
            lines.append(
                f"- `{ref['id']}` ({ref['priority_class']}, score {ref['priority_score']}): "
                f"{ref['next_test']}"
            )

    lines.extend(
        [
            "",
            "## What This Proves",
            "",
            "It proves that the project now has an offensive complement to the",
            "defensive audit stack. Every current gap and every top queue item is",
            "attached to a decision rule, not just a suggestion.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not prove that any gap has closed. The current snapshot still",
            "has zero promoted mechanisms, zero full-reliability rows, zero",
            "transfer-ready mechanisms, and zero hidden-state-allowed bridge rungs.",
            "",
        ]
    )
    return "\n".join(lines)


def write_gap_closure_plan(
    output_path: Path = GAP_CLOSURE_PLAN_PATH,
    report_path: Path = GAP_CLOSURE_PLAN_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_control_surface_gap_closure_plan()
    validate_gap_closure_plan(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write gap-closure artifacts")
    parser.add_argument("--json", action="store_true", help="print gap-closure JSON")
    args = parser.parse_args()

    payload = build_control_surface_gap_closure_plan()
    validate_gap_closure_plan(payload)

    if args.write:
        write_gap_closure_plan()
        print(
            f"wrote {GAP_CLOSURE_PLAN_PATH.relative_to(ROOT).as_posix()} and "
            f"{GAP_CLOSURE_PLAN_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['work_order_count']} work orders"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"gap closure plan ok: {payload['summary']['work_order_count']} work orders")
    print("urgency_counts:", json.dumps(payload["summary"]["urgency_counts"], sort_keys=True))
    print(
        "critical_gap_coverage:",
        f"{payload['summary']['critical_gap_coverage_count']}/"
        f"{payload['summary']['critical_gap_count']}",
    )


if __name__ == "__main__":
    main()
