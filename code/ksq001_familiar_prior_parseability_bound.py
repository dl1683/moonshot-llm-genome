#!/usr/bin/env python
"""KSQ001 familiar-prior parseability bound.

Second-wave adjudication for the KSQ001 work order:
`bound_ksq001_familiar_prior_parseability`.

The narrow question is whether the first-run conflict parseability failure can
be repaired by answer-shape pressure without collapsing the behavior to local
lookup, semantic-prior recall, or UNKNOWN. This remains behavior-only. A pass
can license a later signature screen, but not a hidden-state claim,
intervention, or mechanism claim.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ksq001_familiar_entity_prior_counterbalance_first_run import (
    CARD_ID,
    CANDIDATE_ID,
    MODEL_ID,
    PANELS,
    PRIMARY_PANEL,
    prompt_audit,
    render_registry,
    score_records,
    source_rows,
    structural_check,
    summarize,
    table_sources,
    write_json,
)
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer


REPAIR_ID = "ksq001_familiar_prior_parseability_bound"
WORK_ORDER_ID = "bound_ksq001_familiar_prior_parseability"
RUN_TYPE = "ksq001_familiar_prior_parseability_bound_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq001_familiar_prior_parseability_bound_behavior"
RESULT_DIR = Path("results/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq001_familiar_prior_parseability_bound_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = (
    RESULT_DIR / "ksq001_familiar_prior_parseability_bound_smoke_limit10.json"
)
FULL_BEHAVIOR_RESULT_PATH = (
    RESULT_DIR / "ksq001_familiar_prior_parseability_bound_full_behavior.json"
)
STATUS_CARD = Path("research/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND_STATUS.md")
PREREG_PATH = Path("research/prereg/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND.md")

TEMPLATES = (
    "compact_original_replay",
    "compact_begin_city",
    "compact_city_name_first",
)

FIRST_RUN_COMPACT_PRIMARY_PARSEABILITY = 0.60
FIRST_RUN_COMPACT_PRIMARY_LABEL_COUNTS = {
    "artificial_value": 18,
    "real_prior": 2,
    "unknown": 4,
    "unparsed": 16,
}


def response_suffix(template: str) -> list[str]:
    if template == "compact_original_replay":
        return [
            "Return only one city name, or UNKNOWN.",
            "Do not explain.",
            "Answer:",
        ]
    if template == "compact_begin_city":
        return [
            "Begin with one city name or UNKNOWN.",
            "Do not explain.",
            "Answer:",
        ]
    if template == "compact_city_name_first":
        return [
            "City name first. If no city is available, write UNKNOWN.",
            "Do not explain.",
            "Answer:",
        ]
    raise ValueError(f"unknown template {template}")


def prompt_for_panel(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
) -> tuple[str, str]:
    if panel == "source_local_artificial_lookup":
        rows = table_sources(sources, source, include_query=True)
        if template in TEMPLATES:
            lines = [
                "Temporary city registry:",
                *render_registry(rows, key_field="country"),
                f"Query country: {source['country']}",
            ]
        else:
            raise ValueError(f"unknown template {template}")
        expected_label = "artificial_value"
    elif panel == "semantic_prior_direct_control":
        if template in TEMPLATES:
            lines = [
                "Geography query:",
                f"Country: {source['country']}",
                "Capital:",
            ]
        else:
            raise ValueError(f"unknown template {template}")
        expected_label = "real_prior"
    elif panel == "semantic_prior_lure":
        if template in TEMPLATES:
            lines = [
                "Association task:",
                f"{source['country']} -> {source['artificial_value']}",
                f"Query: {source['country']}",
            ]
        else:
            raise ValueError(f"unknown template {template}")
        expected_label = "ambiguous"
    elif panel == "answer_absent_irrelevant_nulls":
        rows = table_sources(sources, source, include_query=False)
        if template in TEMPLATES:
            lines = [
                "Temporary city registry:",
                *render_registry(rows, key_field="country"),
                f"Query country: {source['country']}",
                "If absent: UNKNOWN",
            ]
        else:
            raise ValueError(f"unknown template {template}")
        expected_label = "unknown"
    else:
        raise ValueError(f"unknown panel {panel}")
    return "\n".join([*lines, *response_suffix(template)]), expected_label


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = source_rows(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for panel in PANELS:
                prompt, expected_label = prompt_for_panel(
                    sources,
                    source,
                    panel=panel,
                    template=template,
                )
                records.append(
                    {
                        "id": f"KSQ001_BOUND_{template}_{panel}_{source['source_id']}",
                        "card_id": CARD_ID,
                        "candidate_id": CANDIDATE_ID,
                        "repair_id": REPAIR_ID,
                        "work_order_id": WORK_ORDER_ID,
                        "run_type": run_type,
                        "model_id": MODEL_ID,
                        "template": template,
                        "panel": panel,
                        "split": source["split"],
                        "source_id": source["source_id"],
                        "source_index": source["source_index"],
                        "country": source["country"],
                        "synthetic_key": source["synthetic_key"],
                        "true_capital": source["true_capital"],
                        "override_capital": source["override_capital"],
                        "lure_capital": source["lure_capital"],
                        "artificial_value": source["artificial_value"],
                        "expected_label": expected_label,
                        "candidate_answers": [
                            source["artificial_value"],
                            source["true_capital"],
                            source["lure_capital"],
                            "UNKNOWN",
                        ],
                        "prompt": prompt,
                    }
                )
    return records


def branch_collapse(summary: dict[str, Any]) -> str | None:
    primary = summary["selected_template_summary"]["primary"]
    if float(primary["artificial_value_rate"]) > 0.90:
        return "local_artificial_lookup"
    if float(primary["prior_or_lure_rate"]) > 0.90:
        return "semantic_prior_recall"
    if float(primary["unknown_rate"]) > 0.90:
        return "unknown_abstention"
    return None


def template_gate(summary: dict[str, Any], template: str) -> dict[str, Any]:
    item = summary["by_template"][template]
    local = item["panels"]["source_local_artificial_lookup"]
    prior = item["panels"]["semantic_prior_direct_control"]
    null = item["panels"]["answer_absent_irrelevant_nulls"]
    primary = item["primary"]
    holdout = item["primary_holdout"]
    collapse = None
    if float(primary["artificial_value_rate"]) > 0.90:
        collapse = "local_artificial_lookup"
    elif float(primary["prior_or_lure_rate"]) > 0.90:
        collapse = "semantic_prior_recall"
    elif float(primary["unknown_rate"]) > 0.90:
        collapse = "unknown_abstention"
    criteria = {
        "source_local_artificial_control_passed": float(local["parseable_rate"]) >= 0.90
        and float(local["artificial_value_rate"]) >= 0.85,
        "semantic_prior_direct_control_passed": float(prior["parseable_rate"]) >= 0.90
        and float(prior["real_prior_rate"]) >= 0.75,
        "answer_absent_null_passed": float(null["parseable_rate"]) >= 0.90
        and float(null["unknown_rate"]) >= 0.80
        and float(null["artificial_value_rate"]) <= 0.10
        and float(null["prior_or_lure_rate"]) <= 0.10,
        "primary_parseability_passed": float(primary["parseable_rate"]) >= 0.90,
        "semantic_prior_competition_present": int(primary["artificial_value"]) > 0
        and int(primary["prior_or_lure"]) > 0,
        "no_branch_collapse": collapse is None,
        "holdout_mixture_present": int(holdout["artificial_value"]) > 0
        and int(holdout["prior_or_lure"]) > 0,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "collapse_mode": collapse,
        "primary_parseable_rate": primary["parseable_rate"],
        "primary_artificial_rate": primary["artificial_value_rate"],
        "primary_prior_or_lure_rate": primary["prior_or_lure_rate"],
        "primary_unknown_rate": primary["unknown_rate"],
        "primary_label_counts": primary["label_counts"],
        "holdout_label_counts": holdout["label_counts"],
        "local_artificial_rate": local["artificial_value_rate"],
        "prior_direct_real_prior_rate": prior["real_prior_rate"],
        "null_unknown_rate": null["unknown_rate"],
    }


def adjudication_decision(summary: dict[str, Any], *, full_run: bool) -> dict[str, Any]:
    selected = summary["selection"]["selected_template"]
    selected_gate = template_gate(summary, selected)
    gates = {template: template_gate(summary, template) for template in TEMPLATES}
    passing_templates = [template for template, gate in gates.items() if gate["passed"]]
    collapse = branch_collapse(summary)
    primary = summary["selected_template_summary"]["primary"]
    parseability_delta = (
        float(primary["parseable_rate"]) - FIRST_RUN_COMPACT_PRIMARY_PARSEABILITY
    )

    if not full_run:
        verdict = "smoke_parseability_bound_candidate"
        route_decision = "smoke_only_no_route_decision"
        exported = "FAMILIAR_PRIOR_PARSEABILITY_SMOKE_CANDIDATE"
        behavior_ready = False
        signature_screen_allowed = False
        kill_rule_triggered = False
    elif summary["behavior_ready"] and passing_templates:
        verdict = "familiar_prior_parseability_repaired"
        route_decision = "admit_behavior_substrate_only"
        exported = "FAMILIAR_PRIOR_BEHAVIOR_ADMITTED"
        behavior_ready = True
        signature_screen_allowed = True
        kill_rule_triggered = False
    elif selected_gate["criteria"]["primary_parseability_passed"] and collapse is not None:
        verdict = "parseability_repaired_by_branch_collapse"
        route_decision = "kill_ordinary_familiar_prior_parseability_repair"
        exported = "FAMILIAR_PRIOR_PARSEABILITY_BRANCH_COLLAPSE"
        behavior_ready = False
        signature_screen_allowed = False
        kill_rule_triggered = True
    elif not selected_gate["criteria"]["primary_parseability_passed"]:
        verdict = "parseability_repair_failed"
        route_decision = "closeout_familiar_prior_parseability_tradeoff"
        exported = "FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF"
        behavior_ready = False
        signature_screen_allowed = False
        kill_rule_triggered = True
    else:
        verdict = "repair_failed_controls_or_holdout"
        route_decision = "kill_ordinary_familiar_prior_parseability_repair"
        exported = "FAMILIAR_PRIOR_REPAIR_CONTROL_REGRESSION"
        behavior_ready = False
        signature_screen_allowed = False
        kill_rule_triggered = True

    return {
        "work_order_id": WORK_ORDER_ID,
        "repair_verdict": verdict,
        "route_decision": route_decision,
        "exported_diagnostic_class": exported,
        "promotion_rule_passed": behavior_ready,
        "kill_rule_triggered": kill_rule_triggered,
        "behavior_ready": behavior_ready,
        "signature_screen_allowed": signature_screen_allowed,
        "hidden_state_claim_allowed": False,
        "intervention_allowed": False,
        "mechanism_claim_allowed": False,
        "selected_template": selected,
        "selected_template_gate": selected_gate,
        "template_gates": gates,
        "passing_templates": passing_templates,
        "collapse_mode": collapse,
        "parseability_delta_vs_first_run_compact": parseability_delta,
        "first_run_compact_primary_parseability": FIRST_RUN_COMPACT_PRIMARY_PARSEABILITY,
        "first_run_compact_primary_label_counts": FIRST_RUN_COMPACT_PRIMARY_LABEL_COUNTS,
    }


def quiet_payload(summary: dict[str, Any], decision: dict[str, Any], output_path: Path) -> dict[str, Any]:
    return {
        "diagnostic_class": summary["diagnostic_class"],
        "repair_verdict": decision["repair_verdict"],
        "route_decision": decision["route_decision"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "behavior_ready": decision["behavior_ready"],
        "signature_screen_allowed": decision["signature_screen_allowed"],
        "hidden_state_claim_allowed": decision["hidden_state_claim_allowed"],
        "intervention_allowed": decision["intervention_allowed"],
        "selected_template": decision["selected_template"],
        "passing_templates": decision["passing_templates"],
        "collapse_mode": decision["collapse_mode"],
        "selected_primary": summary["selected_template_summary"]["primary"],
        "output_path": str(output_path),
    }


def write_prereg(path: Path) -> None:
    lines = [
        "# KSQ001 Familiar-Prior Parseability Bound",
        "",
        "Status: predeclared second-wave bounded repair-or-closeout.",
        "",
        "Runner:",
        "",
        "> `code/ksq001_familiar_prior_parseability_bound.py`",
        "",
        "Full behavior result:",
        "",
        "> `results/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND/ksq001_familiar_prior_parseability_bound_full_behavior.json`",
        "",
        "Status card:",
        "",
        "> `research/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND_STATUS.md`",
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
            "Admit behavior only if a full 40-source run reaches conflict",
            "parseability >= 0.90 while preserving local artificial lookup,",
            "semantic-prior direct recall, answer-absent nulls, conflict mixture,",
            "source-disjoint holdout mixture, prompt audit, and candidate/output",
            "margin reporting.",
            "",
            "## Kill Rule",
            "",
            "Kill ordinary KSQ001 repair if parseability improves only by",
            "collapsing to local lookup, semantic-prior recall, or UNKNOWN, or if",
            "direct controls/nulls regress.",
            "",
            "## Forbidden Claims",
            "",
            "- This repair is a mechanism card.",
            "- Parser repair alone is behavior repair.",
            "- A behavior pass is a hidden-state claim or intervention.",
            "- Familiar-prior pressure is a knowledge-control surface.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# KSQ001 Familiar-Prior Parseability Bound Status",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Runner:",
        "",
        "> `code/ksq001_familiar_prior_parseability_bound.py`",
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
                f"Status: {'structural_passed' if structural['passed'] else 'structural_failed'}.",
                "",
                "## Structural Criteria",
                "",
                "| Criterion | Passed |",
                "| --- | --- |",
            ]
        )
        for key, value in structural["criteria"].items():
            lines.append(f"| `{key}` | `{str(value).lower()}` |")
        lines.extend(
            [
                "",
                "Behavior scoring has not been claimed from this structural artifact.",
            ]
        )
    else:
        summary = result["summary"]
        decision = result["repair_decision"]
        lines.extend(
            [
                f"Status: {decision['repair_verdict']}.",
                "",
                "## Verdict",
                "",
                f"- route decision: `{decision['route_decision']}`",
                f"- exported diagnostic: `{decision['exported_diagnostic_class']}`",
                f"- behavior ready: `{str(decision['behavior_ready']).lower()}`",
                f"- signature screen allowed: `{str(decision['signature_screen_allowed']).lower()}`",
                f"- hidden-state claim allowed: `{str(decision['hidden_state_claim_allowed']).lower()}`",
                f"- intervention allowed: `{str(decision['intervention_allowed']).lower()}`",
                f"- selected template: `{decision['selected_template']}`",
                f"- passing templates: `{json.dumps(decision['passing_templates'])}`",
                f"- collapse mode: `{decision['collapse_mode']}`",
                "",
                "## Template Gates",
                "",
                "| Template | Passed | Parseable | Artificial | Prior/Lure | UNKNOWN | Collapse |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for template, gate in decision["template_gates"].items():
            lines.append(
                f"| `{template}` | `{str(gate['passed']).lower()}` | "
                f"`{gate['primary_parseable_rate']:.3f}` | "
                f"`{gate['primary_artificial_rate']:.3f}` | "
                f"`{gate['primary_prior_or_lure_rate']:.3f}` | "
                f"`{gate['primary_unknown_rate']:.3f}` | "
                f"`{gate['collapse_mode']}` |"
            )
        lines.extend(
            [
                "",
                "## Selected Primary",
                "",
                f"- label counts: `{json.dumps(summary['selected_template_summary']['primary']['label_counts'], sort_keys=True)}`",
                f"- first-run compact parseability: `{decision['first_run_compact_primary_parseability']:.3f}`",
                f"- parseability delta: `{decision['parseability_delta_vs_first_run_compact']:.3f}`",
                "",
                "## Boundary",
                "",
                "This is behavior-only. It can close or admit the KSQ001 behavior",
                "substrate, but it does not show an internal signature, causal",
                "intervention, or knowledge-control surface.",
            ]
        )
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- KSQ001 bound is a mechanism card.",
            "- KSQ001 bound supports intervention.",
            "- KSQ001 bound found an internal knowledge-control surface.",
            "- A parser or answer-shape repair is sufficient without branch accounting.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=10)
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
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, TEMPLATES, run_type)
    structural = structural_check(records, TEMPLATES)

    if args.write_prereg:
        write_prereg(args.prereg_path)

    if not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "candidate_id": CANDIDATE_ID,
            "repair_id": REPAIR_ID,
            "work_order_id": WORK_ORDER_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for the KSQ001 familiar-prior parseability bound.",
            "templates": list(TEMPLATES),
            "panels": list(PANELS),
            "primary_panel": PRIMARY_PANEL,
            "hidden_state_allowed": False,
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
                {
                    "passed": False,
                    "diagnostic_class": "structural_invalid",
                    "structural": structural,
                },
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
    decision = adjudication_decision(summary, full_run=full_run)
    result = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "candidate_id": CANDIDATE_ID,
        "repair_id": REPAIR_ID,
        "work_order_id": WORK_ORDER_ID,
        "run_type": BEHAVIOR_RUN_TYPE,
        "model_id": args.model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": args.score_candidates,
        "limit_sources": args.limit_sources,
        "templates": list(TEMPLATES),
        "elapsed_s": time.time() - started,
        "purpose": "Behavior-only KSQ001 familiar-prior parseability bound.",
        "panels": list(PANELS),
        "primary_panel": PRIMARY_PANEL,
        "sources": source_rows(args.limit_sources),
        "hidden_state_allowed": False,
        "structural": structural,
        "records": outputs,
        "summary": summary,
        "repair_decision": decision,
    }
    write_json(args.output_path, result)
    if args.write_status_card:
        write_status_card(args.status_card, result, args.output_path)
    print(json.dumps(quiet_payload(summary, decision, args.output_path), indent=2, ensure_ascii=True))
    return 0 if (summary["passed"] or not full_run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
