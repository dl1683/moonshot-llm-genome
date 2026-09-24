#!/usr/bin/env python
"""MC028 bare-integer full-source boundary test.

MC027 showed that answer interfaces are behavior surfaces: among five answer
formats, bare integer was the only one to clear the 10-source operation
arbitration smoke. MC024, however, already showed that the same broad
few-shot/query-operation route can fail on full source-disjoint evaluation.

MC028 tests that boundary directly. It keeps only MC027's winning bare-integer
interface and runs the full 40-source source-disjoint set. This is not hidden-
state work and not a mechanism claim. If it passes behavior, the next step is
candidate/output margin auditing before any signature search.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mc027_answer_interface_sweep as mc027
from mc012_reliability_labeled_numeric_arbitration import MODEL_ID, base_sources
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer
from mc023_query_operation_numeric_arbitration import OPERATION_ASSIGNMENTS, OPERATION_CODES, RULE_ORDERS


CARD_ID = "MC028"
RUN_TYPE = "bare_integer_full_source_boundary_structural_gate"
BEHAVIOR_RUN_TYPE = "bare_integer_full_source_boundary_behavior"
RESULT_DIR = Path("results/cards/MC028")
STATUS_CARD = Path("research/cards/MC028_BARE_INTEGER_FULL_SOURCE_BOUNDARY_STATUS.md")
BARE_CONFIGS = (
    {
        "interface_variant": "bare_integer",
        "option_order": mc027.NO_OPTION_ORDER,
        "template": "bare_integer",
    },
)
TEMPLATES = ("bare_integer",)
PANELS = mc027.PANELS
PRIMARY_CONFLICT_PANELS = mc027.PRIMARY_CONFLICT_PANELS


def source_records(
    limit_sources: int | None = None,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    records = mc027.source_records(
        limit_sources,
        BARE_CONFIGS,
        OPERATION_ASSIGNMENTS,
        RULE_ORDERS,
        run_type,
    )
    for record in records:
        old_id = record["id"]
        record["id"] = old_id.replace("MC027", CARD_ID, 1)
        record["card_id"] = CARD_ID
        record["run_type"] = run_type
    return records


def structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    expected_count = len(source_ids) * len(BARE_CONFIGS) * len(OPERATION_ASSIGNMENTS) * len(RULE_ORDERS) * len(PANELS)
    split_by_source: dict[str, set[str]] = {}
    for row in records:
        split_by_source.setdefault(row["source_id"], set()).add(row["split"])
    split_rows = Counter(row["split"] for row in records)
    conflict_rows = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    expected_conflict = Counter(row["expected_label"] for row in conflict_rows)
    operation_conflict = Counter(row["query_operation"] for row in conflict_rows)
    audits = [mc027.prompt_audit(row) for row in records]
    status_rows = [row["id"] for row, audit in zip(records, audits, strict=True) if audit["prompt_has_status_lexeme"]]
    atomic_leaks = [
        row["id"]
        for row, audit in zip(records, audits, strict=True)
        if not audit["atomic_number_hidden_in_no_option_conflict"]
    ]
    null_leaks = [
        row["id"]
        for row, audit in zip(records, audits, strict=True)
        if not audit["answer_absent_omits_query_local_row"]
    ]
    candidate_collisions = [
        row["id"]
        for row in records
        if len({candidate["answer"] for candidate in row["candidate_answers"]}) != len(row["candidate_answers"])
    ]
    criteria = {
        "expected_row_count": len(records) == expected_count,
        "full_source_count_is_40": len(source_ids) == 40,
        "all_panels_present": set(PANELS) == {row["panel"] for row in records},
        "single_template_bare_integer": {row["template"] for row in records} == {"bare_integer"},
        "single_interface_bare_integer": {row["interface_variant"] for row in records} == {"bare_integer"},
        "all_assignments_present": set(OPERATION_ASSIGNMENTS) == {row["operation_assignment"] for row in records},
        "all_rule_orders_present": set(RULE_ORDERS) == {row["rule_order"] for row in records},
        "source_split_disjoint": all(len(splits) == 1 for splits in split_by_source.values()),
        "holdout_sources_present": any(row["split"] == "holdout" for row in records),
        "calibration_sources_present": any(row["split"] == "calibration" for row in records),
        "atomic_number_hidden_in_conflicts": not atomic_leaks,
        "answer_absent_omits_query_local_row": not null_leaks,
        "no_candidate_collisions": not candidate_collisions,
        "prompts_have_no_status_lexemes": not status_rows,
        "conflict_expected_balanced": expected_conflict["local_number"] == expected_conflict["atomic_number"],
        "conflict_operations_balanced": operation_conflict["ALPHA"] == operation_conflict["BETA"],
    }
    return {
        "passed": all(criteria.values()),
        "record_count": len(records),
        "source_count": len(source_ids),
        "expected_record_count": expected_count,
        "split_row_counts": dict(sorted(split_rows.items())),
        "criteria": criteria,
        "status_lexeme_rows": status_rows[:20],
        "atomic_leak_rows": atomic_leaks[:20],
        "null_leak_rows": null_leaks[:20],
        "candidate_collision_rows": candidate_collisions[:20],
        "conflict_expected_counts": dict(sorted(expected_conflict.items())),
        "conflict_operation_counts": dict(sorted(operation_conflict.items())),
    }


def classify(criteria: dict[str, Any], panels: dict[str, Any]) -> tuple[str, str]:
    if not criteria["structural_passed"]:
        return "structural_invalid", "structural_invalid"
    if not criteria["all_controls_passed"]:
        return "bare_integer_full_source_control_failed", "full_source_controls_failed"
    if not criteria["operation_local_conflict_at_least_85p"]:
        return "bare_integer_full_source_local_branch_failed", "full_source_local_branch_failed"
    if not criteria["operation_atomic_conflict_at_least_85p"]:
        if float(panels["operation_atomic_interface_conflict"]["other_number_rate"]) >= 0.10:
            return "bare_integer_full_source_atomic_other_number_leak", "full_source_atomic_other_number_leak"
        return "bare_integer_full_source_atomic_branch_failed", "full_source_atomic_branch_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "bare_integer_full_source_behavior_candidate_baselines_missing", "full_source_behavior_passed_without_margin_baselines"
    return "bare_integer_full_source_behavior_ready", "full_source_behavior_ready"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    structural = structural_check(records)
    by_template = mc027.config_summary(outputs, TEMPLATES)
    selected = by_template["bare_integer"]
    selected_rows = [row for row in outputs if row["template"] == "bare_integer"]
    panels = selected["panels"]
    prompt_audit_passed = all(
        row["answer_absent_omits_query_local_row"]
        and row["atomic_number_hidden_in_no_option_conflict"]
        for row in selected_rows
    )
    prompts_have_no_status_lexemes = all(not row["prompt_has_status_lexeme"] for row in selected_rows)
    criteria = {
        "smoke_mode": False,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"]
        and structural["criteria"]["holdout_sources_present"]
        and structural["criteria"]["calibration_sources_present"],
        "selected_prompt_audit_passed": prompt_audit_passed,
        "prompts_have_no_status_lexemes": prompts_have_no_status_lexemes,
        "familiar_lookup_at_least_90p": float(panels["familiar_interface_lookup"]["local_number_rate"]) >= 0.90,
        "atomic_control_at_least_85p": float(panels["atomic_interface_control"]["atomic_number_rate"]) >= 0.85,
        "answer_absent_unknown_at_least_90p": float(panels["answer_absent_interface_null"]["unknown_rate"]) >= 0.90,
        "operation_rule_absent_unknown_at_least_90p": float(panels["operation_rule_absent_interface_null"]["unknown_rate"]) >= 0.90,
        "operation_local_conflict_at_least_85p": float(panels["operation_local_interface_conflict"]["local_number_rate"]) >= 0.85,
        "operation_atomic_conflict_at_least_85p": float(panels["operation_atomic_interface_conflict"]["atomic_number_rate"]) >= 0.85,
        "operation_atomic_other_number_leak_below_10p": float(panels["operation_atomic_interface_conflict"]["other_number_rate"]) < 0.10,
        "all_panels_parseable_at_least_95p": min(float(panel["parseable_rate"]) for panel in panels.values()) >= 0.95,
        "candidate_and_output_margins_reported": False,
    }
    criteria["all_controls_passed"] = (
        criteria["familiar_lookup_at_least_90p"]
        and criteria["atomic_control_at_least_85p"]
        and criteria["answer_absent_unknown_at_least_90p"]
        and criteria["operation_rule_absent_unknown_at_least_90p"]
        and criteria["all_panels_parseable_at_least_95p"]
    )
    diagnostic_class, observed_pattern = classify(criteria, panels)
    behavior_gate_passed = diagnostic_class == "bare_integer_full_source_behavior_ready"
    behavior_candidate = diagnostic_class == "bare_integer_full_source_behavior_candidate_baselines_missing"
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": {
            "selected_template": "bare_integer",
            "selection_key": [
                min(
                    float(panels["familiar_interface_lookup"]["local_number_rate"]),
                    float(panels["atomic_interface_control"]["atomic_number_rate"]),
                    float(panels["answer_absent_interface_null"]["unknown_rate"]),
                    float(panels["operation_rule_absent_interface_null"]["unknown_rate"]),
                ),
                min(
                    float(panels["operation_local_interface_conflict"]["local_number_rate"]),
                    float(panels["operation_atomic_interface_conflict"]["atomic_number_rate"]),
                ),
                float(selected["primary_conflict"]["expected_correct_rate"]),
                min(float(panel["parseable_rate"]) for panel in panels.values()),
                0,
            ],
            "rule": [
                "bare-integer full-source boundary test",
                "no interface selection performed",
            ],
        },
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
        "observed_failure_pattern": observed_pattern,
        "passed": behavior_gate_passed,
        "behavior_ready": behavior_gate_passed,
        "behavior_candidate": behavior_candidate,
        "signature_ready": False,
        "intervention_ready": False,
        "diagnostic_class": diagnostic_class,
    }


def quiet_summary(summary: dict[str, Any], output_path: Path | None = None) -> dict[str, Any]:
    selected = summary["selected_template_summary"]
    payload = {
        "diagnostic_class": summary["diagnostic_class"],
        "observed_failure_pattern": summary["observed_failure_pattern"],
        "passed": summary["passed"],
        "behavior_ready": summary["behavior_ready"],
        "behavior_candidate": summary["behavior_candidate"],
        "signature_ready": summary["signature_ready"],
        "criteria": summary["criteria"],
        "selection": summary["selection"],
        "selected_controls": {
            panel: selected["panels"][panel]
            for panel in (
                "familiar_interface_lookup",
                "atomic_interface_control",
                "answer_absent_interface_null",
                "operation_rule_absent_interface_null",
            )
        },
        "selected_conflicts": {
            panel: selected["panels"][panel]
            for panel in PRIMARY_CONFLICT_PANELS
        },
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_behavior_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines = [
        "# MC028 Bare-Integer Full-Source Boundary Status",
        "",
        f"Status: {summary['diagnostic_class']}.",
        "",
        f"Observed pattern: `{summary['observed_failure_pattern']}`.",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## Artifact",
        "",
        "- runner:",
        "  `code/mc028_bare_integer_full_source_boundary.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
        "MC028 is a full-source boundary test for MC027's winning bare-integer",
        "interface. It does not establish a hidden signature, intervention, or",
        "mechanism card. If behavior passes, candidate/output margin baselines are",
        "still required before hidden-state work.",
        "",
        "## Gate Criteria",
        "",
        "| Criterion | Value |",
        "| --- | --- |",
    ]
    for key, value in criteria.items():
        rendered = f"{value:.3f}" if isinstance(value, float) else str(value).lower() if isinstance(value, bool) else str(value)
        lines.append(f"| `{key}` | `{rendered}` |")
    lines.extend(
        [
            "",
            "## Panels",
            "",
            "| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate | Other Rate | Expected Correct |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for panel in PANELS:
        item = selected["panels"][panel]
        lines.append(
            f"| `{panel}` | {item['rows']} | {item['parseable_rate']:.3f} | "
            f"{item['local_number_rate']:.3f} | {item['atomic_number_rate']:.3f} | "
            f"{item['unknown_rate']:.3f} | {item['other_number_rate']:.3f} | "
            f"{item['expected_correct_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This result can only decide whether the bare-integer answer interface",
            "survives full-source behavior gates. It does not license a probe or",
            "intervention until output/candidate baselines are added.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--score-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--artifact-prefix", default="mc028_bare_integer_full_source_boundary_structural")
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, run_type)
    structural = structural_check(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest and not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for full-source bare-integer operation arbitration boundary.",
            "templates": list(TEMPLATES),
            "interface_variants": ["bare_integer"],
            "operation_assignments": list(OPERATION_ASSIGNMENTS),
            "rule_orders": list(RULE_ORDERS),
            "operation_codes": list(OPERATION_CODES),
            "panels": list(PANELS),
            "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
            "structural": structural,
            "records": records,
        }
        output_path = args.output_dir / f"{args.artifact_prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "passed": structural["passed"],
                    "record_count": structural["record_count"],
                    "source_count": structural["source_count"],
                    "criteria": structural["criteria"],
                    "output_path": str(output_path),
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0 if structural["passed"] else 1

    if not args.score_model:
        print(json.dumps(structural, indent=2, ensure_ascii=True))
        return 0 if structural["passed"] else 1

    if not structural["passed"]:
        print(json.dumps({"passed": False, "diagnostic_class": "structural_invalid", "structural": structural}, indent=2))
        return 1

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.local_files_only)
    outputs = mc027.score_records(records, tokenizer, model, args.max_new_tokens, verbose=not args.quiet)
    summary = summarize(records, outputs)
    prefix = args.artifact_prefix
    if prefix == "mc028_bare_integer_full_source_boundary_structural":
        prefix = "mc028_bare_integer_full_source_boundary_behavior"
    output_path = args.output_dir / f"{prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    result = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "run_type": BEHAVIOR_RUN_TYPE,
        "model_id": args.model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": False,
        "limit_sources": args.limit_sources,
        "templates": list(TEMPLATES),
        "interface_variants": ["bare_integer"],
        "operation_assignments": list(OPERATION_ASSIGNMENTS),
        "rule_orders": list(RULE_ORDERS),
        "operation_codes": list(OPERATION_CODES),
        "elapsed_s": time.time() - started,
        "purpose": "Full-source bare-integer boundary test after MC027 interface sweep.",
        "panels": list(PANELS),
        "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
        "sources": base_sources(args.limit_sources),
        "structural": structural,
        "records": outputs,
        "summary": summary,
    }
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    if args.write_status_card:
        write_behavior_status_card(args.status_card, result, output_path)
    print(json.dumps(quiet_summary(summary, output_path), indent=2, ensure_ascii=True))
    return 0 if (summary["behavior_ready"] or summary["behavior_candidate"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
