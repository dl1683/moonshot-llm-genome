#!/usr/bin/env python
"""KSQ002 source-rewrite holdout repair.

This is the single predeclared ordinary repair allowed by the generated
knowledge-failure topology for KSQ002. The first full behavior run narrowly
failed because the selected template produced a source-disjoint rewrite
holdout parseability rate of 0.875 against the 0.900 gate while deletion and
query-only controls remained clean.

The repair changes only the answer interface: each prompt asks for a city-field
answer and explicitly asks not to repeat the country. It does not drop the
source-disjoint holdout, deletion control, query-only control, prompt audit, or
candidate/output margin reporting. The result can promote only a behavior
substrate; it cannot claim a hidden signature, intervention, or mechanism.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ksq002_familiar_entity_source_rewrite_equivalence_first_run import (
    BEHAVIOR_RUN_TYPE,
    CARD_ID,
    CANDIDATE_ID,
    MODEL_ID,
    PANELS,
    REWRITE_PANELS,
    SOURCE_VALUE_PANELS,
    generate_answer,
    next_token_candidate_logits,
    prompt_audit,
    quiet_summary,
    response_suffix,
    source_rows,
    strict_parse,
    structural_check,
    summarize,
    table_sources,
    write_json,
)
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer


REPAIR_ID = "ksq002_source_rewrite_holdout_repair"
RUN_TYPE = "ksq002_source_rewrite_holdout_repair_structural_gate"
REPAIR_BEHAVIOR_RUN_TYPE = "ksq002_source_rewrite_holdout_repair_behavior"
TEMPLATES = ("city_field_rewrite",)
RESULT_DIR = Path("results/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq002_source_rewrite_holdout_repair_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = RESULT_DIR / "ksq002_source_rewrite_holdout_repair_smoke_limit10.json"
FULL_BEHAVIOR_RESULT_PATH = RESULT_DIR / "ksq002_source_rewrite_holdout_repair_full_behavior.json"
STATUS_CARD = Path("research/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR_STATUS.md")
PREREG_PATH = Path("research/prereg/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR.md")


def render_colon_rows(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["country: listed_city"]
    for row in rows:
        lines.append(f"{row['country']}: {row['artificial_value']}")
    return lines


def render_neutral_rows(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["Neutral city notes:"]
    for row in rows:
        lines.append(f"The listed city for {row['country']} is {row['artificial_value']}.")
    return lines


def answer_channel_lines(source: dict[str, Any], *, missing_allowed: bool) -> list[str]:
    lines = [
        f"Lookup country: {source['country']}",
        "Answer with the listed city name only; do not repeat the country.",
    ]
    if missing_allowed:
        lines.insert(1, "If the lookup country has no row, answer UNKNOWN.")
    return lines


def prompt_for_panel(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
) -> tuple[str, str]:
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template}")
    if panel == "baseline_source_value_lookup":
        rows = table_sources(sources, source, include_query=True)
        lines = [
            "Temporary city table for this task.",
            *render_colon_rows(rows),
            *answer_channel_lines(source, missing_allowed=False),
        ]
        expected_label = "artificial_value"
    elif panel in REWRITE_PANELS:
        rows = table_sources(sources, source, include_query=True)
        lines = [
            "The same temporary city table is written as neutral notes.",
            *render_neutral_rows(rows),
            *answer_channel_lines(source, missing_allowed=False),
        ]
        expected_label = "artificial_value"
    elif panel == "source_deletion":
        rows = table_sources(sources, source, include_query=False)
        lines = [
            "Temporary city table for this task.",
            *render_colon_rows(rows),
            *answer_channel_lines(source, missing_allowed=True),
        ]
        expected_label = "unknown"
    elif panel == "query_only_control":
        lines = [
            "Temporary city table for this task.",
            "(no rows provided)",
            *answer_channel_lines(source, missing_allowed=True),
        ]
        expected_label = "unknown"
    else:
        raise ValueError(f"unknown panel {panel}")
    return "\n".join([*lines, *response_suffix()]), expected_label


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
                        "id": f"{CARD_ID}R_{template}_{panel}_{source['source_id']}",
                        "card_id": CARD_ID,
                        "candidate_id": CANDIDATE_ID,
                        "repair_id": REPAIR_ID,
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


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    score_candidates: bool,
    *,
    verbose: bool,
) -> list[dict[str, Any]]:
    outputs = []
    for index, record in enumerate(records, start=1):
        generated = generate_answer(model, tokenizer, record["prompt"], max_new_tokens)
        parsed = strict_parse(record, generated["generated_text"])
        audit = prompt_audit(record)
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            **audit,
            "is_source_value_panel": record["panel"] in SOURCE_VALUE_PANELS,
            "is_rewrite_panel": record["panel"] in REWRITE_PANELS,
            "is_expected_correct": record["expected_label"] == parsed["selected_label"],
            "is_query_proxy": record["panel"] == "query_only_control"
            and parsed["selected_label"] == "artificial_value",
        }
        if score_candidates:
            output.update(next_token_candidate_logits(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
                f"panel={record['panel']} -> {output['selected_label']} "
                f"{str(output['selected_answer'])!r} generated={generated['generated_text']!r}"
            )
    return outputs


def add_repair_decision(summary: dict[str, Any]) -> dict[str, Any]:
    criteria = summary["criteria"]
    if summary["behavior_ready"]:
        repair_verdict = "source_rewrite_answer_channel_repair_behavior_ready"
        route_decision = "promote_behavior_substrate_only"
        exported_diagnostic = "SOURCE_REWRITE_BEHAVIOR_ADMITTED"
        signature_screen_allowed = True
    elif not criteria["source_deletion_passed"] or not criteria["query_only_control_passed"]:
        repair_verdict = "source_rewrite_repair_broke_locality_controls"
        route_decision = "kill_ordinary_source_rewrite_repair"
        exported_diagnostic = "SOURCE_REWRITE_REPAIR_LOCALITY_REGRESSION"
        signature_screen_allowed = False
    elif not criteria["source_disjoint_rewrite_holdout_passed"]:
        repair_verdict = "source_rewrite_holdout_parseability_persisted"
        route_decision = "kill_ordinary_source_rewrite_repair"
        exported_diagnostic = "SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE"
        signature_screen_allowed = False
    else:
        repair_verdict = f"source_rewrite_repair_failed_{summary['diagnostic_class']}"
        route_decision = "kill_ordinary_source_rewrite_repair"
        exported_diagnostic = "SOURCE_REWRITE_REPAIR_FAILED_OTHER_GATE"
        signature_screen_allowed = False
    return {
        "repair_verdict": repair_verdict,
        "route_decision": route_decision,
        "exported_diagnostic_class": exported_diagnostic,
        "signature_screen_allowed": signature_screen_allowed,
        "hidden_state_claim_allowed": False,
        "intervention_allowed": False,
        "mechanism_claim_allowed": False,
    }


def write_prereg(path: Path) -> None:
    lines = [
        "# KSQ002 Source-Rewrite Holdout Repair",
        "",
        "Status: predeclared single ordinary repair for the KSQ002 source-disjoint rewrite holdout boundary.",
        "",
        "Runner:",
        "",
        "> `code/ksq002_source_rewrite_holdout_repair.py`",
        "",
        "Full behavior result:",
        "",
        "> `results/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR/ksq002_source_rewrite_holdout_repair_full_behavior.json`",
        "",
        "Status card:",
        "",
        "> `research/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR_STATUS.md`",
        "",
        "## Repair",
        "",
        "Use one fixed template, `city_field_rewrite`, with the original KSQ002",
        "panels and source split. The only intended change is the answer channel:",
        "the prompt asks for the listed city name and asks the model not to repeat",
        "the country.",
        "",
        "## Promotion Rule",
        "",
        "Promote only to behavior-substrate admission if the full 40-source run",
        "passes baseline lookup, neutral rewrite, source deletion, query-only,",
        "source-disjoint rewrite holdout, prompt audit, and candidate/output",
        "margin-reporting gates.",
        "",
        "## Kill Rule",
        "",
        "Kill ordinary KSQ002 source-rewrite repair if this predeclared rerun still",
        "misses source-disjoint holdout parseability or breaks deletion/query-only",
        "controls.",
        "",
        "## Forbidden Claims",
        "",
        "- This repair is a mechanism card.",
        "- This repair proves a source-channel, knowledge vector, or internal control surface.",
        "- This repair supports intervention.",
        "- A behavior pass permits a hidden-state claim without a later signature and intervention audit.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# KSQ002 Source-Rewrite Holdout Repair Status",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Runner:",
        "",
        "> `code/ksq002_source_rewrite_holdout_repair.py`",
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
                "The repair substrate is constructed. Model-scored behavior is still required before any decision.",
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
                "## Counts",
                "",
                f"- records: `{structural['record_count']}`",
                f"- sources: `{structural['source_count']}`",
                f"- panels: `{json.dumps(structural['panel_counts'], sort_keys=True)}`",
                f"- templates: `{json.dumps(structural['template_counts'], sort_keys=True)}`",
                f"- split source counts: `{json.dumps(structural['split_source_counts'], sort_keys=True)}`",
            ]
        )
    else:
        summary = result["summary"]
        decision = result["repair_decision"]
        selected = summary["selected_template_summary"]
        rewrite_holdout = selected["rewrite_holdout"]
        lines.extend(
            [
                f"Status: {decision['repair_verdict']}.",
                "",
                "## Verdict",
                "",
                f"- route decision: `{decision['route_decision']}`",
                f"- exported diagnostic: `{decision['exported_diagnostic_class']}`",
                f"- behavior ready: `{str(summary['behavior_ready']).lower()}`",
                f"- signature screen allowed next: `{str(decision['signature_screen_allowed']).lower()}`",
                f"- hidden-state claim allowed: `{str(decision['hidden_state_claim_allowed']).lower()}`",
                f"- intervention allowed: `{str(decision['intervention_allowed']).lower()}`",
                "",
                "## Behavior Gates",
                "",
                "| Criterion | Passed |",
                "| --- | --- |",
            ]
        )
        for key, value in summary["criteria"].items():
            lines.append(f"| `{key}` | `{str(value).lower()}` |")
        lines.extend(
            [
                "",
                "## Selected Template",
                "",
                f"- selected template: `{summary['selection']['selected_template']}`",
                f"- rewrite delta from baseline: `{summary['rewrite_delta_from_baseline']:.3f}`",
                f"- source-disjoint rewrite holdout label counts: `{json.dumps(rewrite_holdout['label_counts'], sort_keys=True)}`",
                f"- source-disjoint rewrite holdout parseability: `{rewrite_holdout['parseable_rate']:.3f}`",
                f"- source-disjoint rewrite holdout artificial-value rate: `{rewrite_holdout['artificial_value_rate']:.3f}`",
                "",
                "| Panel | Label counts |",
                "| --- | --- |",
            ]
        )
        for panel_name in PANELS:
            labels = selected["panels"][panel_name]["label_counts"]
            lines.append(f"| `{panel_name}` | `{json.dumps(labels, sort_keys=True)}` |")
        lines.extend(
            [
                "",
                "## Boundary",
                "",
                "A pass here is only behavior-substrate admission for KSQ002. It means",
                "the ordinary answer-interface repair made the source-rewrite behavior",
                "stable enough to justify a later signature screen. It does not show",
                "an internal source-channel signature, a causal intervention, or a",
                "knowledge-control mechanism.",
            ]
        )
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- KSQ002 repair is a mechanism card.",
            "- KSQ002 repair supports intervention.",
            "- KSQ002 repair found an internal source-channel or knowledge-control surface.",
            "- A behavior pass is itself a hidden-state or causal result.",
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
    run_type = REPAIR_BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
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
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for the KSQ002 source-rewrite holdout answer-channel repair.",
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
    repair_decision = add_repair_decision(summary)
    result = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "candidate_id": CANDIDATE_ID,
        "repair_id": REPAIR_ID,
        "run_type": REPAIR_BEHAVIOR_RUN_TYPE,
        "model_id": args.model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": args.score_candidates,
        "limit_sources": args.limit_sources,
        "templates": list(TEMPLATES),
        "elapsed_s": time.time() - started,
        "purpose": "Behavior-only KSQ002 source-rewrite holdout answer-channel repair run.",
        "summary": summary,
        "repair_decision": repair_decision,
        "outputs": outputs,
        "hidden_state_claim_allowed": False,
        "intervention_allowed": False,
    }
    if args.write_manifest:
        write_json(args.output_path, result)
        if args.write_status_card:
            write_status_card(args.status_card, result, args.output_path)

    payload = quiet_summary(summary, args.output_path if args.write_manifest else None)
    payload["repair_decision"] = repair_decision
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if (summary["behavior_ready"] or not full_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
