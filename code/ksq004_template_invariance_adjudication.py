#!/usr/bin/env python
"""KSQ004 template-invariance adjudication.

This second-wave runner closes the high-priority topology work order
`adjudicate_ksq004_template_invariance`. The first KSQ004 full run showed a
template split: `question_form` behaved like a mostly working bridge, while
`compact_form` collapsed expected-atomic conflict rows toward local-table
answers. This runner asks whether bridge behavior survives at least one
additional non-question template under the same bare numeric answer interface.

The result can promote only a behavior substrate and a later signature-screen
decision. It cannot claim an internal bridge, a causal intervention, or a
knowledge-control mechanism.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ksq004_bridge_answer_interface_minimal_pairs_first_run import (
    ATOMIC_SUBTYPES,
    BEHAVIOR_RUN_TYPE,
    CARD_ID,
    CANDIDATE_ID,
    LOCAL_SUBTYPES,
    MODEL_ID,
    NULL_SUBTYPES,
    PANELS,
    PRIMARY_CONFLICT_PANELS,
    SUBTYPES_BY_PANEL,
    base_sources,
    distractors,
    quiet_summary,
    response_suffix,
    score_records,
    structural_check,
    summarize,
    write_json,
)
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer


ADJUDICATION_ID = "ksq004_template_invariance_adjudication"
RUN_TYPE = "ksq004_template_invariance_adjudication_structural_gate"
ADJUDICATION_BEHAVIOR_RUN_TYPE = "ksq004_template_invariance_adjudication_behavior"
TEMPLATES = ("question_form", "neutral_sentence_form", "relation_key_form")
RESULT_DIR = Path("results/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq004_template_invariance_adjudication_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = RESULT_DIR / "ksq004_template_invariance_adjudication_smoke_limit10.json"
FULL_BEHAVIOR_RESULT_PATH = RESULT_DIR / "ksq004_template_invariance_adjudication_full_behavior.json"
STATUS_CARD = Path("research/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION_STATUS.md")
PREREG_PATH = Path("research/prereg/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION.md")


def render_local_table(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    include_query: bool,
    template: str,
) -> list[str]:
    rows = distractors(sources, source, 4)
    visible_rows = rows if include_query else [
        row for row in rows if row["source_id"] != source["source_id"]
    ]
    if template == "question_form":
        lines = [
            "A local lab table may define temporary element numbers.",
            "The requested relation decides which number to return.",
            "Local lab table:",
        ]
        for row in visible_rows:
            lines.append(f"{row['element']} has local lab number {row['local_number']}.")
        return lines
    if template == "neutral_sentence_form":
        lines = [
            "Task packet with local lab numbers.",
            "Use the requested relation to choose the returned number.",
            "Local lab table:",
        ]
        for row in visible_rows:
            lines.append(f"{row['element']} has local lab number {row['local_number']}.")
        return lines
    if template == "relation_key_form":
        lines = [
            "Task packet.",
            "local lab table:",
        ]
        for row in visible_rows:
            lines.append(f"{row['element']}: local lab number {row['local_number']}")
        return lines
    raise ValueError(f"unknown template {template}")


def append_query(lines: list[str], source: dict[str, Any], *, relation: str, template: str) -> None:
    if template == "question_form":
        lines.extend(
            [
                f"Requested relation: {relation}.",
                f"Query element: {source['element']}",
            ]
        )
    elif template == "neutral_sentence_form":
        lines.extend(
            [
                f"Requested relation is {relation}.",
                f"Lookup element is {source['element']}.",
            ]
        )
    elif template == "relation_key_form":
        lines.extend(
            [
                f"relation: {relation}",
                f"element: {source['element']}",
            ]
        )
    else:
        raise ValueError(f"unknown template {template}")


def no_table_lines(template: str) -> list[str]:
    if template == "question_form":
        return [
            "No local lab table is active for this item.",
            "Use standard chemistry for the requested relation.",
        ]
    if template == "neutral_sentence_form":
        return [
            "Task packet with no active local lab table.",
            "Use standard chemistry for the requested relation.",
        ]
    if template == "relation_key_form":
        return [
            "Task packet.",
            "local lab table: none",
        ]
    raise ValueError(f"unknown template {template}")


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    subtype: str,
    template: str,
) -> tuple[str, str]:
    if subtype in LOCAL_SUBTYPES:
        expected_label = "local_number"
    elif subtype in ATOMIC_SUBTYPES:
        expected_label = "atomic_number"
    elif subtype in NULL_SUBTYPES:
        expected_label = "unknown"
    else:
        raise ValueError(f"unknown subtype {subtype}")

    if panel == "matched_minimal_pairs":
        if subtype == "matched_local_contract":
            lines = render_local_table(sources, source, include_query=True, template=template)
            append_query(lines, source, relation="local table number", template=template)
        elif subtype == "matched_atomic_contract":
            lines = no_table_lines(template)
            append_query(lines, source, relation="standard atomic number", template=template)
        else:
            raise ValueError(f"bad subtype {subtype} for {panel}")
    elif panel == "local_learned_direct_controls":
        if subtype == "local_direct":
            lines = render_local_table(sources, source, include_query=True, template=template)
            append_query(lines, source, relation="local table number", template=template)
        elif subtype == "atomic_direct":
            lines = no_table_lines(template)
            append_query(lines, source, relation="standard atomic number", template=template)
        else:
            raise ValueError(f"bad subtype {subtype} for {panel}")
    elif panel == "minimal_pair_conflict":
        lines = render_local_table(sources, source, include_query=True, template=template)
        if subtype == "local_branch_conflict":
            append_query(lines, source, relation="local table number", template=template)
        elif subtype == "atomic_branch_conflict":
            append_query(lines, source, relation="standard atomic number", template=template)
        else:
            raise ValueError(f"bad subtype {subtype} for {panel}")
    elif panel == "side_answer_leakage":
        lines = render_local_table(sources, source, include_query=True, template=template)
        lines.extend(
            [
                f"Side note number: {source['lure_atomic_number']}.",
                "The side note number is not the requested relation.",
            ]
        )
        if subtype == "local_side_leakage":
            append_query(lines, source, relation="local table number", template=template)
        elif subtype == "atomic_side_leakage":
            append_query(lines, source, relation="standard atomic number", template=template)
        else:
            raise ValueError(f"bad subtype {subtype} for {panel}")
    elif panel == "null_and_holdout":
        if subtype != "answer_absent_null":
            raise ValueError(f"bad subtype {subtype} for {panel}")
        lines = render_local_table(sources, source, include_query=False, template=template)
        append_query(lines, source, relation="local table number", template=template)
        lines.append("If the query element is absent from the local lab table, return UNKNOWN.")
    else:
        raise ValueError(f"unknown panel {panel}")
    return "\n".join([*lines, *response_suffix()]), expected_label


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for panel_name in PANELS:
                for subtype in SUBTYPES_BY_PANEL[panel_name]:
                    prompt, expected_label = make_prompt(
                        sources,
                        source,
                        panel=panel_name,
                        subtype=subtype,
                        template=template,
                    )
                    records.append(
                        {
                            "id": f"{CARD_ID}T_{template}_{panel_name}_{subtype}_{source['source_id']}",
                            "card_id": CARD_ID,
                            "candidate_id": CANDIDATE_ID,
                            "adjudication_id": ADJUDICATION_ID,
                            "run_type": run_type,
                            "model_id": MODEL_ID,
                            "template": template,
                            "panel": panel_name,
                            "subtype": subtype,
                            "split": source["split"],
                            "source_id": source["source_id"],
                            "source_index": source["source_index"],
                            "element": source["element"],
                            "atomic_number": str(source["atomic_number"]),
                            "lure_atomic_number": str(source["lure_atomic_number"]),
                            "local_number": str(source["local_number"]),
                            "expected_label": expected_label,
                            "expected_local_answer": str(source["local_number"]),
                            "expected_real_answer": str(source["atomic_number"]),
                            "expected_null_answer": "UNKNOWN",
                            "side_answer": str(source["lure_atomic_number"]),
                            "prompt": prompt,
                            "candidate_answers": [
                                str(source["local_number"]),
                                str(source["atomic_number"]),
                                str(source["lure_atomic_number"]),
                                "UNKNOWN",
                            ],
                        }
                    )
    return records


def template_gate(
    template_summary: dict[str, Any],
    *,
    full_run: bool,
    score_candidates: bool,
) -> dict[str, Any]:
    panels = template_summary["panels"]
    subtypes = template_summary["subtypes"]
    matched = panels["matched_minimal_pairs"]
    conflict = template_summary["primary_conflict"]
    side = panels["side_answer_leakage"]
    null = panels["null_and_holdout"]
    holdout_conflict = template_summary["primary_conflict_holdout"]
    conflict_min_count = 24 if full_run else 6
    holdout_min_rate = 0.65 if full_run else 0.50
    conflict_expected_counts = conflict["expected_label_counts"]
    criteria = {
        "matched_minimal_pairs_passed": float(matched["parseable_rate"]) >= 0.90
        and float(matched["expected_correct_rate"]) >= 0.75,
        "local_learned_direct_controls_passed": float(
            subtypes["local_direct"]["parseable_rate"]
        )
        >= 0.90
        and float(subtypes["local_direct"]["local_number_rate"]) >= 0.75
        and float(subtypes["atomic_direct"]["parseable_rate"]) >= 0.90
        and float(subtypes["atomic_direct"]["atomic_number_rate"]) >= 0.75,
        "minimal_pair_conflict_passed": float(conflict["parseable_rate"]) >= 0.90
        and float(conflict["expected_correct_rate"]) >= 0.70
        and int(conflict_expected_counts.get("local_number", 0)) >= conflict_min_count
        and int(conflict_expected_counts.get("atomic_number", 0)) >= conflict_min_count,
        "side_answer_leakage_passed": float(side["parseable_rate"]) >= 0.90
        and float(side["side_answer_rate"]) <= 0.10,
        "null_and_holdout_passed": float(null["unknown_rate"]) >= 0.80
        and float(holdout_conflict["expected_correct_rate"]) >= holdout_min_rate,
        "candidate_and_output_margins_reported": bool(
            score_candidates
            and template_summary["primary_conflict_margin_audits"]["reported"]
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "conflict_expected_correct_rate": conflict["expected_correct_rate"],
        "conflict_parseable_rate": conflict["parseable_rate"],
        "atomic_branch_atomic_rate": subtypes["atomic_branch_conflict"][
            "atomic_number_rate"
        ],
        "local_branch_local_rate": subtypes["local_branch_conflict"][
            "local_number_rate"
        ],
        "holdout_expected_correct_rate": holdout_conflict["expected_correct_rate"],
        "null_unknown_rate": null["unknown_rate"],
        "side_answer_rate": side["side_answer_rate"],
        "conflict_label_counts": conflict["label_counts"],
    }


def add_adjudication_decision(
    summary: dict[str, Any],
    *,
    full_run: bool,
    score_candidates: bool,
) -> dict[str, Any]:
    template_gates = {
        template: template_gate(
            template_summary,
            full_run=full_run,
            score_candidates=score_candidates,
        )
        for template, template_summary in summary["by_template"].items()
    }
    passing_templates = [
        template for template, gate in template_gates.items() if gate["passed"]
    ]
    non_question_passing = [
        template for template in passing_templates if template != "question_form"
    ]
    compact_or_neutral = [
        template for template in TEMPLATES if template != "question_form"
    ]
    compact_atomic_rates = {
        template: template_gates[template]["atomic_branch_atomic_rate"]
        for template in compact_or_neutral
    }
    if len(passing_templates) >= 2 and non_question_passing:
        adjudication_verdict = "template_invariant_bridge_behavior"
        route_decision = "admit_behavior_substrate_only"
        exported_diagnostic = "TEMPLATE_INVARIANT_BRIDGE_BEHAVIOR"
        behavior_ready = True
        signature_screen_allowed = True
    elif template_gates.get("question_form", {}).get("passed") and not non_question_passing:
        adjudication_verdict = "question_form_only_bridge_behavior"
        route_decision = "kill_same_family_answer_interface_repair"
        exported_diagnostic = "ANSWER_INTERFACE_TEMPLATE_FRAGILITY"
        behavior_ready = False
        signature_screen_allowed = False
    elif any(rate < 0.50 for rate in compact_atomic_rates.values()):
        adjudication_verdict = "compact_neutral_atomic_branch_collapse"
        route_decision = "kill_same_family_answer_interface_repair"
        exported_diagnostic = "ANSWER_INTERFACE_TEMPLATE_FRAGILITY"
        behavior_ready = False
        signature_screen_allowed = False
    else:
        adjudication_verdict = "template_invariance_not_established"
        route_decision = "kill_same_family_answer_interface_repair"
        exported_diagnostic = "ANSWER_INTERFACE_TEMPLATE_FRAGILITY"
        behavior_ready = False
        signature_screen_allowed = False
    return {
        "adjudication_verdict": adjudication_verdict,
        "route_decision": route_decision,
        "exported_diagnostic_class": exported_diagnostic,
        "template_gates": template_gates,
        "passing_templates": passing_templates,
        "non_question_passing_templates": non_question_passing,
        "behavior_ready": behavior_ready,
        "signature_screen_allowed": signature_screen_allowed,
        "hidden_state_claim_allowed": False,
        "intervention_allowed": False,
        "mechanism_claim_allowed": False,
    }


def write_prereg(path: Path) -> None:
    lines = [
        "# KSQ004 Template-Invariance Adjudication",
        "",
        "Status: predeclared second-wave template-boundary adjudication.",
        "",
        "Runner:",
        "",
        "> `code/ksq004_template_invariance_adjudication.py`",
        "",
        "Full behavior result:",
        "",
        "> `results/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION/ksq004_template_invariance_adjudication_full_behavior.json`",
        "",
        "Status card:",
        "",
        "> `research/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION_STATUS.md`",
        "",
        "## Templates",
        "",
    ]
    for template in TEMPLATES:
        lines.append(f"- `{template}`")
    lines.extend(
        [
            "",
            "## Promotion Rule",
            "",
            "Admit bridge behavior only if at least two predeclared templates pass",
            "matched pairs, direct controls, conflict routing, side-answer leakage,",
            "null/holdout, and candidate/output margin gates. At least one passing",
            "template must be non-question-form.",
            "",
            "## Kill Rule",
            "",
            "Kill same-family answer-interface repair if `question_form` is the only",
            "passing surface or if compact/neutral templates keep collapsing",
            "expected-atomic rows to local answers.",
            "",
            "## Forbidden Claims",
            "",
            "- This adjudication is a mechanism card.",
            "- `question_form` alone is a robust bridge.",
            "- First-token numeric margins replace sequence candidate scoring.",
            "- A behavior pass permits hidden-state, intervention, or mechanism claims.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# KSQ004 Template-Invariance Adjudication Status",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Runner:",
        "",
        "> `code/ksq004_template_invariance_adjudication.py`",
        "",
        "Result:",
        "",
        f"> `{output_path.as_posix()}`",
        "",
    ]
    if "summary" not in result:
        structural = result["structural"]
        lines.extend(
            [
                "Status: structural_passed." if structural["passed"] else "Status: structural_failed.",
                "",
                "## Verdict",
                "",
                "The template-invariance substrate is constructed. Model-scored behavior is still required before any decision.",
                "",
                "## Structural Criteria",
                "",
                "| Criterion | Passed |",
                "| --- | --- |",
            ]
        )
        for key, value in structural["criteria"].items():
            lines.append(f"| `{key}` | `{str(value).lower()}` |")
    else:
        summary = result["summary"]
        decision = result["adjudication_decision"]
        lines.extend(
            [
                f"Status: {decision['adjudication_verdict']}.",
                "",
                "## Verdict",
                "",
                f"- route decision: `{decision['route_decision']}`",
                f"- exported diagnostic: `{decision['exported_diagnostic_class']}`",
                f"- behavior ready: `{str(decision['behavior_ready']).lower()}`",
                f"- signature screen allowed: `{str(decision['signature_screen_allowed']).lower()}`",
                f"- hidden-state claim allowed: `{str(decision['hidden_state_claim_allowed']).lower()}`",
                f"- intervention allowed: `{str(decision['intervention_allowed']).lower()}`",
                f"- passing templates: `{json.dumps(decision['passing_templates'])}`",
                "",
                "## Template Gates",
                "",
                "| Template | Passed | Conflict Expected | Atomic-Branch Atomic | Holdout Expected | Null UNKNOWN | Side Answer |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for template, gate in decision["template_gates"].items():
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
                "## Selected First-Run-Style Summary",
                "",
                f"- selected template: `{summary['selection']['selected_template']}`",
                f"- first-run-style diagnostic class: `{summary['diagnostic_class']}`",
                f"- first-run-style behavior ready: `{str(summary['behavior_ready']).lower()}`",
                "",
                "## Boundary",
                "",
                "The adjudication is behavior-only. It either admits a later signature",
                "screen or kills the same-family answer-interface repair route. It does",
                "not itself show an internal bridge, intervention, or mechanism.",
            ]
        )
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- KSQ004 adjudication is a mechanism card.",
            "- KSQ004 adjudication supports intervention.",
            "- KSQ004 adjudication found an internal bridge or knowledge-control surface.",
            "- A single passing template is template-invariant bridge behavior.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--score-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--score-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--prereg-path", type=Path, default=PREREG_PATH)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-prereg", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    run_type = ADJUDICATION_BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, TEMPLATES, run_type)
    structural = structural_check(records, TEMPLATES)

    if args.write_prereg:
        write_prereg(args.prereg_path)

    if not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "candidate_id": CANDIDATE_ID,
            "adjudication_id": ADJUDICATION_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for the KSQ004 template-invariance adjudication.",
            "templates": list(TEMPLATES),
            "panels": list(PANELS),
            "hidden_state_claim_allowed": False,
            "intervention_allowed": False,
            "structural": structural,
            "records": records,
        }
        if args.write_manifest:
            write_json(args.output_path, result)
            if args.write_status_card:
                write_status_card(args.status_card, result, args.output_path)
        print(
            json.dumps(
                {
                    "passed": structural["passed"],
                    "record_count": structural["record_count"],
                    "source_count": structural["source_count"],
                    "criteria": structural["criteria"],
                    "output_path": str(args.output_path) if args.write_manifest else None,
                    "status_card": str(args.status_card) if args.write_status_card else None,
                    "prereg": str(args.prereg_path) if args.write_prereg else None,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0 if structural["passed"] else 1

    if not structural["passed"]:
        print(
            json.dumps(
                {"passed": False, "diagnostic_class": "structural_invalid", "structural": structural},
                indent=2,
                ensure_ascii=True,
            )
        )
        return 1

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.local_files_only)
    outputs = score_records(
        records,
        tokenizer,
        model,
        args.max_new_tokens,
        args.score_candidates,
        verbose=not args.quiet,
    )
    full_run = args.limit_sources is None
    summary = summarize(records, outputs, TEMPLATES, full_run, args.score_candidates)
    adjudication_decision = add_adjudication_decision(
        summary,
        full_run=full_run,
        score_candidates=args.score_candidates,
    )
    result = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "candidate_id": CANDIDATE_ID,
        "adjudication_id": ADJUDICATION_ID,
        "run_type": ADJUDICATION_BEHAVIOR_RUN_TYPE,
        "model_id": args.model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": args.score_candidates,
        "limit_sources": args.limit_sources,
        "templates": list(TEMPLATES),
        "elapsed_s": time.time() - started,
        "purpose": "Behavior-only KSQ004 template-invariance adjudication run.",
        "summary": summary,
        "adjudication_decision": adjudication_decision,
        "outputs": outputs,
        "hidden_state_claim_allowed": False,
        "intervention_allowed": False,
    }
    if args.write_manifest:
        write_json(args.output_path, result)
        if args.write_status_card:
            write_status_card(args.status_card, result, args.output_path)

    payload = quiet_summary(summary, args.output_path if args.write_manifest else None)
    payload["adjudication_decision"] = adjudication_decision
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if (adjudication_decision["behavior_ready"] or not full_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
