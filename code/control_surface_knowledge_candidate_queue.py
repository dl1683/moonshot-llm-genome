"""Build the candidate queue for future knowledge substrates.

The admission protocol defines the gates. This layer proposes concrete
behavior-only substrate candidates for the levels that still need a new
substrate, and it keeps each candidate tied to the admission gates and to a
predeclared failure/export path.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json
from control_surface_knowledge_substrate_admission import (
    ADMISSION_GATE_ORDER,
    KNOWLEDGE_SUBSTRATE_ADMISSION_PATH,
)
from control_surface_next_queue import NEXT_QUEUE_PATH


KNOWLEDGE_CANDIDATE_QUEUE_PATH = (
    ROOT / "data" / "control_surface_knowledge_candidate_queue.json"
)
KNOWLEDGE_CANDIDATE_QUEUE_REPORT_PATH = (
    ROOT / "research" / "45_CONTROL_SURFACE_KNOWLEDGE_CANDIDATE_QUEUE.md"
)

REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "level_id",
    "admission_class",
    "priority_class",
    "first_run_type",
    "behavior_design",
    "dumb_explanations",
    "first_run_panels",
    "gate_bindings",
    "promotion_rule",
    "death_rule",
    "containment_rule",
    "export_rule",
    "hidden_state_license",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def by_key(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        item_key = item[key]
        if item_key in result:
            raise AssertionError(f"duplicate {key}: {item_key}")
        result[item_key] = item
    return result


def gate_bindings(extra: dict[str, str]) -> list[dict[str, str]]:
    bindings = []
    for gate_id in ADMISSION_GATE_ORDER:
        bindings.append(
            {
                "gate_id": gate_id,
                "candidate_requirement": extra.get(
                    gate_id,
                    "Use the admission-packet default for this gate and report the pass/fail evidence in the behavior status card.",
                ),
            }
        )
    return bindings


def candidate(
    candidate_id: str,
    level_id: str,
    admission_class: str,
    priority_class: str,
    priority_rank: int,
    title: str,
    behavior_design: str,
    why_this_candidate: str,
    first_run_panels: list[str],
    dumb_explanations: list[str],
    gate_overrides: dict[str, str],
    linked_queue_ids: list[str],
    promotion_rule: str,
    death_rule: str,
    containment_rule: str,
    export_rule: str,
) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate_id,
        "level_id": level_id,
        "admission_class": admission_class,
        "priority_class": priority_class,
        "priority_rank": priority_rank,
        "title": title,
        "first_run_type": "behavior_substrate_admission_only",
        "behavior_design": behavior_design,
        "why_this_candidate": why_this_candidate,
        "first_run_panels": first_run_panels,
        "dumb_explanations": dumb_explanations,
        "gate_bindings": gate_bindings(gate_overrides),
        "linked_next_queue_ids": linked_queue_ids,
        "promotion_rule": promotion_rule,
        "death_rule": death_rule,
        "containment_rule": containment_rule,
        "export_rule": export_rule,
        "hidden_state_license": "forbidden_until_candidate_passes_admission",
    }
    missing = REQUIRED_CANDIDATE_FIELDS - set(payload)
    if missing:
        raise AssertionError(f"{candidate_id}: missing fields {missing}")
    return payload


def build_candidate_specs() -> list[dict[str, Any]]:
    return [
        candidate(
            "ksq001_familiar_entity_prior_counterbalance",
            "level_2_semi_synthetic_familiar_entity",
            "new_familiar_entity_substrate",
            "high",
            1,
            "Counterbalance familiar-entity priors against artificial values.",
            (
                "Use familiar entity names, but assign artificial values from a "
                "non-city answer space. Pair each entity with semantic-prior, "
                "source-local, and answer-absent panels so real-world familiarity "
                "can be measured instead of smuggled in as the label."
            ),
            (
                "This is the smallest materially new MC007 successor because it "
                "keeps familiar names but removes city-answer proxying and explicit "
                "source-label authority."
            ),
            [
                "source-local artificial value lookup",
                "semantic-prior lure panel",
                "answer-absent and irrelevant-source nulls",
                "source-disjoint familiar-entity holdout",
                "candidate/output margin baselines",
            ],
            [
                "The artificial value is just easier to copy than the semantic prior.",
                "Entity familiarity is acting only as a prompt key, not as knowledge pressure.",
                "Prompt wording still tells the model to trust the local source.",
                "Output candidates or answer shape separate the labels.",
            ],
            {
                "material_novelty": "Must avoid explicit source labels and city answers from MC007.",
                "conflict_mixture": "Require both source-local and semantic-prior lure selections across matched rows.",
                "prompt_channel_locality": "Match or remove authority language across conditions.",
                "output_candidate_baselines": "Report artificial-value candidate margins and semantic-prior margins.",
            },
            [
                "familiar_entities_can_collapse_to_lookup_keys__next_test_1",
                "familiar_entities_can_collapse_to_lookup_keys__next_test_2",
            ],
            "Admit only if familiar priors measurably compete with local artificial values while prompt and output baselines fail to explain the split.",
            "Kill if behavior reduces to local copy, semantic prior recall, authority wording, or parse/answer-shape effects.",
            "If source-local lookup works without semantic competition, preserve it only as a familiar-key lookup diagnostic.",
            "Export SEMANTIC_PRIOR_INTERFERENCE or FAMILIAR_ENTITY_LOOKUP_KEY_COLLAPSE.",
        ),
        candidate(
            "ksq002_familiar_entity_source_rewrite_equivalence",
            "level_2_semi_synthetic_familiar_entity",
            "new_familiar_entity_substrate",
            "medium",
            2,
            "Test whether familiar-entity source lookup survives neutral rewrites.",
            (
                "Start from the cleanest familiar-entity lookup baseline and run "
                "neutral source rewrites, query-only variants, and source-deletion "
                "controls before any hidden-state or intervention work."
            ),
            (
                "The prior MC007 route was too entangled with source wording; this "
                "candidate decides whether a familiar-entity surface survives as a "
                "source-value behavior after visible wording is neutralized."
            ),
            [
                "baseline source-value lookup",
                "neutral rewrite lookup",
                "source deletion",
                "query-only control",
                "source-disjoint holdout",
            ],
            [
                "The source text is the whole mechanism.",
                "The query token or entity name alone carries the answer.",
                "Rewrite differences change parseability rather than behavior.",
                "The model follows instruction tone, not a source-value binding.",
            ],
            {
                "material_novelty": "Must change the source-channel test, not only the answer parser.",
                "direct_controls": "Include query-only and source-deletion controls.",
                "prompt_channel_locality": "Require neutral rewrite equivalence before considering the surface internal.",
            },
            [
                "coarse_source_ablation_overstates_circuit_locality__next_test_1",
                "coarse_source_ablation_overstates_circuit_locality__next_test_2",
            ],
            "Admit only if neutral rewrites preserve the behavior and source deletion/query-only controls fail to reproduce it.",
            "Kill if source deletion, query-only text, or prompt rewrite effects explain the behavior.",
            "Contain as a source-channel diagnostic if lookup works but rewrite equivalence fails.",
            "Export SOURCE_REWRITE_EQUIVALENCE_FAILED or QUERY_ONLY_SOURCE_PROXY.",
        ),
        candidate(
            "ksq003_bridge_statusless_evidence_aggregation",
            "level_3_symbolic_or_learned_memory_bridge",
            "new_bridge_substrate_class",
            "immediate",
            1,
            "Use statusless evidence aggregation instead of source-validity labels.",
            (
                "Build a local-versus-learned arbitration table where branch "
                "selection depends on multiple content facts that jointly imply "
                "source support, without trusted/untrusted labels, checksums, row "
                "codes, or direct fact-claim validity cues."
            ),
            (
                "The top queue asks for MC012-level direct controls and conflict "
                "mixture without visible source-status text. This is the most "
                "direct substrate candidate for that pressure."
            ),
            [
                "source-local direct control",
                "learned-fact direct control",
                "multi-evidence conflict rows",
                "evidence-ablation rows",
                "answer-absent nulls",
                "source-disjoint holdout",
                "candidate/output baselines",
            ],
            [
                "The evidence features are still visible labels in disguise.",
                "Prompt-local table dominance determines all rows.",
                "Learned-fact rows fail direct recall under table pressure.",
                "Candidate margins reveal the branch before hidden states are inspected.",
            ],
            {
                "material_novelty": "Must be outside MC007-MC033 source labels, checksums, consistency, and fact-claim cues.",
                "direct_controls": "Preserve MC012-level direct local and learned controls.",
                "conflict_mixture": "Require learned-fact branch rows to survive prompt-local table pressure.",
                "prompt_channel_locality": "Ablate each evidence feature and prove no single visible status channel carries the rule.",
            },
            [
                "authority_pressure_creates_contrast_before_clean_substrate__next_test_1",
                "behavior_substrate_first_or_everything_lies__next_test_2",
                "source_visible_lookup_localizes_more_than_parametric_override__next_test_3",
            ],
            "Admit only if direct controls, conflict mixture, nulls, prompt-channel audits, holdout, and output/candidate baselines pass together.",
            "Kill if the learned branch collapses under table pressure or the evidence features behave like visible status labels.",
            "Contain as a bridge diagnostic if it reveals a new typed failure before hidden-state work.",
            "Export STATUSLESS_EVIDENCE_VISIBLE_CHANNEL or LEARNED_BRANCH_TABLE_PRESSURE_COLLAPSE.",
        ),
        candidate(
            "ksq004_bridge_answer_interface_minimal_pairs",
            "level_3_symbolic_or_learned_memory_bridge",
            "new_bridge_substrate_class",
            "high",
            2,
            "Factor the bridge answer interface with minimal-pair outputs.",
            (
                "Use matched minimal-pair answer interfaces where local and learned "
                "branches share answer length, type, frequency band, and parser "
                "shape, then test whether the bridge failure persists without "
                "answer-token or numeric-option shortcuts."
            ),
            (
                "Many bridge rungs died through answer-interface and local-source "
                "salience artifacts. This candidate makes the answer interface the "
                "primary object under test before another route is attempted."
            ),
            [
                "matched local/learned minimal pairs",
                "answer-shape and frequency controls",
                "side-number leakage panel",
                "null rows",
                "source-disjoint holdout",
                "candidate/output baselines",
            ],
            [
                "The answer interface, not the branch rule, determines behavior.",
                "Numeric or token frequency shortcuts explain the branch.",
                "Side-number leakage substitutes for learned-fact arbitration.",
                "Balanced outputs make both branches fail rather than compete.",
            ],
            {
                "material_novelty": "Must change the answer interface class rather than adding another branch cue.",
                "parseability_and_label_balance": "Match answer shape, length, and parser reliability across branch labels.",
                "side_effect_and_leakage": "Report side-number and other-answer leakage as first-class outcomes.",
            },
            [
                "behavior_substrate_first_or_everything_lies__next_test_2",
                "null_reliability_bottleneck__next_test_2",
            ],
            "Admit only if matched answer interfaces preserve direct controls and produce a real local-versus-learned conflict mixture.",
            "Kill if balancing the interface removes the behavior contrast or exposes answer-token shortcuts.",
            "Contain as an answer-interface law candidate, not a knowledge mechanism.",
            "Export ANSWER_INTERFACE_BRANCH_SHORTCUT or BRIDGE_MINIMAL_PAIR_CONTRAST_ABSENT.",
        ),
        candidate(
            "ksq005_uncertainty_grounded_answerability",
            "level_5_real_abstention_uncertainty",
            "new_real_uncertainty_substrate",
            "high",
            1,
            "Build grounded answerability before refusal or uncertainty probing.",
            (
                "Create known, unknown, unsupported, and contradicted factual "
                "items from a frozen evidence source, then ask for generated short "
                "answers with abstention allowed but not requested by label text."
            ),
            (
                "MC002/MC002B failed because pressure prompts defined the behavior. "
                "This candidate moves label grounding outside the prompt before "
                "testing uncertainty."
            ),
            [
                "known factual direct rows",
                "unknown/nonce rows",
                "unsupported-context rows",
                "contradicted-context rows",
                "abstention nulls",
                "source-disjoint holdout",
                "requested-mode and output-margin baselines",
            ],
            [
                "Abstention is caused by the instruction style.",
                "Known/unknown labels leak through entity or answer shape.",
                "The model refuses because the prompt requests caution, not because evidence is absent.",
                "Output margins separate answerable and unanswerable rows before any hidden signature.",
            ],
            {
                "material_novelty": "Must use grounded labels outside MC002/MC002B pressure prompts.",
                "behavior_contract": "Do not put known/unknown/support labels in visible prompt text.",
                "null_rows": "Separate unknown, unsupported, contradicted, and irrelevant-context nulls.",
                "output_candidate_baselines": "Report answer versus abstain margins and requested-mode baselines.",
            },
            [
                "behavior_substrate_first_or_everything_lies__next_test_2",
                "final_state_output_geometry_dominance__next_test_2",
            ],
            "Admit only if answerability behavior survives prompt, requested-mode, label-balance, output-margin, null, and holdout gates.",
            "Kill if abstention follows caution wording, answer schema, entity familiarity, or output margin.",
            "Contain as a refusal-template or output-geometry diagnostic if controls explain it.",
            "Export LABEL_GROUNDING_FAILURE, REQUESTED_MODE_CONFOUND, or OUTPUT_MARGIN_CONFUND.",
        ),
        candidate(
            "ksq006_uncertainty_context_support_counterfactuals",
            "level_5_real_abstention_uncertainty",
            "new_real_uncertainty_substrate",
            "medium",
            2,
            "Use counterfactual context support rather than known/unknown prompts.",
            (
                "Construct factual claims with matched supporting, irrelevant, "
                "contradicting, and insufficient contexts. The target behavior is "
                "support-sensitive answer or abstain, not generic caution."
            ),
            (
                "This is the natural successor to MC002B, but with counterfactual "
                "context panels and prompt-mode controls specified before the run."
            ),
            [
                "supported-context rows",
                "irrelevant-context rows",
                "contradicting-context rows",
                "insufficient-context rows",
                "claim-only rows",
                "source-disjoint holdout",
                "requested-mode and output baselines",
            ],
            [
                "The model follows support words in the prompt.",
                "Contradiction rows are easier to parse than insufficient rows.",
                "Answer shape or length distinguishes supported and unsupported cases.",
                "The behavior is a refusal style, not evidence tracking.",
            ],
            {
                "material_novelty": "Must be a new counterfactual support table, not another pressure-prompt repair.",
                "direct_controls": "Include claim-only and context-only controls.",
                "prompt_channel_locality": "Match support-language cues across supported and unsupported rows.",
                "source_disjoint_holdout": "Hold out claims and evidence sources together.",
            },
            [
                "behavior_substrate_first_or_everything_lies__next_test_2",
                "null_reliability_bottleneck__next_test_1",
            ],
            "Admit only if support-sensitive behavior survives matched context controls and output/requested-mode baselines.",
            "Kill if support language, answer shape, or caution prompting explains the behavior.",
            "Contain as context-support behavior only until a control-surviving signature and intervention exist.",
            "Export CONTEXT_SUPPORT_PROMPT_CHANNEL or REFUSAL_TEMPLATE_LEAKAGE.",
        ),
    ]


def compact_queue_refs(next_queue: dict[str, Any], ids: list[str]) -> list[dict[str, Any]]:
    queue_by_id = by_key(next_queue["queue"], "id")
    refs = []
    for queue_id in ids:
        if queue_id not in queue_by_id:
            continue
        item = queue_by_id[queue_id]
        refs.append(
            {
                "id": item["id"],
                "priority_class": item["priority_class"],
                "priority_score": item["priority_score"],
                "hypothesis_id": item["hypothesis_id"],
                "action_type": item["action_type"],
                "next_test": item["next_test"],
            }
        )
    return refs


def build_candidates(
    admission: dict[str, Any],
    next_queue: dict[str, Any],
) -> list[dict[str, Any]]:
    packets_by_level = by_key(admission["admission_packets"], "level_id")
    candidates = []
    for spec in build_candidate_specs():
        packet = packets_by_level[spec["level_id"]]
        candidate_payload = {
            **spec,
            "admission_packet": {
                "level_id": packet["level_id"],
                "label": packet["label"],
                "admission_class": packet["admission_class"],
                "hidden_state_license": packet["hidden_state_license"],
                "admission_decision": packet["admission_decision"],
            },
            "linked_next_queue_refs": compact_queue_refs(
                next_queue, spec["linked_next_queue_ids"]
            ),
        }
        candidates.append(candidate_payload)
    return sorted(candidates, key=lambda item: (item["priority_rank"], item["candidate_id"]))


def build_summary(
    candidates: list[dict[str, Any]],
    admission: dict[str, Any],
    next_queue: dict[str, Any],
) -> dict[str, Any]:
    linked_queue_ids = sorted(
        {
            queue_id
            for candidate_item in candidates
            for queue_id in candidate_item["linked_next_queue_ids"]
        }
    )
    return {
        "candidate_count": len(candidates),
        "admission_packet_count": admission["summary"]["packet_count"],
        "admission_gate_count": admission["summary"]["unique_gate_count"],
        "total_gate_binding_count": sum(
            len(candidate_item["gate_bindings"]) for candidate_item in candidates
        ),
        "hidden_state_candidate_count": sum(
            1
            for candidate_item in candidates
            if candidate_item["hidden_state_license"]
            != "forbidden_until_candidate_passes_admission"
        ),
        "candidate_count_by_level": dict(
            sorted(Counter(item["level_id"] for item in candidates).items())
        ),
        "candidate_count_by_admission_class": dict(
            sorted(Counter(item["admission_class"] for item in candidates).items())
        ),
        "priority_counts": dict(
            sorted(Counter(item["priority_class"] for item in candidates).items())
        ),
        "linked_next_queue_id_count": len(linked_queue_ids),
        "linked_next_queue_ids": linked_queue_ids,
        "next_queue_item_count": next_queue["summary"]["queue_item_count"],
        "new_hidden_state_search_allowed_level_count": admission["summary"][
            "new_hidden_state_search_allowed_level_count"
        ],
        "promoted_mechanism_count": admission["summary"]["promoted_mechanism_count"],
    }


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload["candidates"]
    admission_level_ids = sorted(
        packet["level_id"] for packet in payload["source_snapshot"]["admission_packets"]
    )
    candidate_level_ids = sorted({item["level_id"] for item in candidates})
    missing_fields = {
        item["candidate_id"]: sorted(REQUIRED_CANDIDATE_FIELDS - set(item))
        for item in candidates
    }
    missing_fields = {key: value for key, value in missing_fields.items() if value}
    gate_failures = [
        item["candidate_id"]
        for item in candidates
        if sorted(binding["gate_id"] for binding in item["gate_bindings"])
        != sorted(ADMISSION_GATE_ORDER)
    ]
    hidden_state_failures = [
        item["candidate_id"]
        for item in candidates
        if item["hidden_state_license"] != "forbidden_until_candidate_passes_admission"
        or item["first_run_type"] != "behavior_substrate_admission_only"
    ]
    no_dumb_explanations = [
        item["candidate_id"] for item in candidates if len(item["dumb_explanations"]) < 3
    ]
    empty_rules = [
        item["candidate_id"]
        for item in candidates
        if not item["promotion_rule"]
        or not item["death_rule"]
        or not item["containment_rule"]
        or not item["export_rule"]
    ]
    return [
        {
            "id": "covers_all_admission_packets",
            "predicate": "candidate levels == admission-packet levels",
            "actual": {
                "candidate_level_ids": candidate_level_ids,
                "admission_level_ids": admission_level_ids,
            },
            "passed": candidate_level_ids == admission_level_ids,
            "why": "Every admission packet needs at least one concrete candidate substrate.",
        },
        {
            "id": "two_candidates_per_admission_class",
            "predicate": "all class counts == 2",
            "actual": payload["summary"]["candidate_count_by_admission_class"],
            "passed": all(
                count == 2
                for count in payload["summary"][
                    "candidate_count_by_admission_class"
                ].values()
            ),
            "why": "Each new-substrate level should have a primary and backup behavior-only candidate.",
        },
        {
            "id": "candidate_fields_are_complete",
            "predicate": "empty dict",
            "actual": missing_fields,
            "passed": not missing_fields,
            "why": "Candidate rows need enough data to be executed or killed.",
        },
        {
            "id": "all_candidates_bind_all_admission_gates",
            "predicate": "empty list",
            "actual": gate_failures,
            "passed": not gate_failures,
            "why": "The queue must not bypass the admission protocol.",
        },
        {
            "id": "hidden_state_work_remains_forbidden",
            "predicate": "empty list and count == 0",
            "actual": {
                "candidate_failures": hidden_state_failures,
                "hidden_state_candidate_count": payload["summary"][
                    "hidden_state_candidate_count"
                ],
                "new_hidden_state_search_allowed_level_count": payload["summary"][
                    "new_hidden_state_search_allowed_level_count"
                ],
            },
            "passed": not hidden_state_failures
            and payload["summary"]["hidden_state_candidate_count"] == 0
            and payload["summary"]["new_hidden_state_search_allowed_level_count"] == 0,
            "why": "These are behavior-substrate candidates, not hidden-state experiments.",
        },
        {
            "id": "dumb_explanations_are_first_class",
            "predicate": "empty list",
            "actual": no_dumb_explanations,
            "passed": not no_dumb_explanations,
            "why": "Each candidate should begin by trying to kill the attractive explanation.",
        },
        {
            "id": "all_candidates_have_decision_rules",
            "predicate": "empty list",
            "actual": empty_rules,
            "passed": not empty_rules,
            "why": "Each candidate needs promotion, death, containment, and export paths.",
        },
        {
            "id": "queue_links_existing_next_tests",
            "predicate": "at least four linked next queue ids",
            "actual": payload["summary"]["linked_next_queue_ids"],
            "passed": payload["summary"]["linked_next_queue_id_count"] >= 4,
            "why": "Candidate queue should operationalize the existing next experiment queue.",
        },
        {
            "id": "does_not_claim_mechanism_progress",
            "predicate": "promoted_mechanism_count == 0 and forbidden claim names no mechanism",
            "actual": {
                "promoted_mechanism_count": payload["summary"][
                    "promoted_mechanism_count"
                ],
                "forbidden_claim": payload["forbidden_claim"],
            },
            "passed": payload["summary"]["promoted_mechanism_count"] == 0
            and "does not promote" in payload["forbidden_claim"],
            "why": "A candidate queue is a planning artifact, not mechanism evidence.",
        },
    ]


def build_control_surface_knowledge_candidate_queue() -> dict[str, Any]:
    admission = load_json(KNOWLEDGE_SUBSTRATE_ADMISSION_PATH)
    next_queue = load_json(NEXT_QUEUE_PATH)
    candidates = build_candidates(admission, next_queue)
    summary = build_summary(candidates, admission, next_queue)
    payload = {
        "schema_version": 1,
        "updated_at": admission.get("updated_at"),
        "purpose": (
            "Turn the knowledge-substrate admission protocol into concrete "
            "behavior-only candidate substrates with dumb baselines, gate "
            "bindings, and kill/export rules."
        ),
        "sources": {
            "knowledge_substrate_admission": rel(KNOWLEDGE_SUBSTRATE_ADMISSION_PATH),
            "next_experiment_queue": rel(NEXT_QUEUE_PATH),
        },
        "source_snapshot": {
            "admission_packets": [
                {
                    "level_id": packet["level_id"],
                    "admission_class": packet["admission_class"],
                    "hidden_state_license": packet["hidden_state_license"],
                }
                for packet in admission["admission_packets"]
            ],
            "admission_gate_order": ADMISSION_GATE_ORDER,
            "next_queue_top_ids": next_queue["summary"]["top_queue_ids"],
        },
        "summary": summary,
        "candidates": candidates,
        "allowed_claim": (
            "The project now has a generated behavior-only candidate queue for "
            "the three knowledge levels that require new substrates. Each "
            "candidate binds to the admission gates and names dumb explanations "
            "to test before hidden-state work."
        ),
        "forbidden_claim": (
            "This queue does not promote any mechanism, does not license "
            "hidden-state search, and does not show that any proposed substrate "
            "will pass."
        ),
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_knowledge_candidate_queue(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("knowledge candidate queue schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"knowledge candidate queue source missing: {rel_path}")
    if payload["summary"]["candidate_count"] != 6:
        raise AssertionError("knowledge candidate queue expected six candidates")
    if payload["summary"]["admission_packet_count"] != 3:
        raise AssertionError("knowledge candidate queue expected three admission packets")
    if payload["summary"]["hidden_state_candidate_count"] != 0:
        raise AssertionError("knowledge candidate queue must not license hidden states")
    if payload["summary"]["total_gate_binding_count"] != 6 * len(ADMISSION_GATE_ORDER):
        raise AssertionError("knowledge candidate queue gate binding count mismatch")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"knowledge candidate queue checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Knowledge Candidate Queue",
        "",
        f"Source updated_at: {payload['updated_at']}",
        "",
        "Status: generated behavior-only candidate queue implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_knowledge_candidate_queue.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_candidate_queue.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_candidate_queue.py --write",
        "python code\\control_surface_knowledge_candidate_queue.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This queue is the operational layer below the admission protocol. It",
        "does not run hidden-state work. It specifies behavior-only substrate",
        "candidates and the dumb explanations that should kill them quickly if",
        "the apparent mechanism is prompt-, output-, parser-, or interface-born.",
        "",
        "## Generated Facts",
        "",
        f"- candidates: {summary['candidate_count']};",
        f"- admission packets covered: {summary['admission_packet_count']};",
        f"- admission gates per candidate: {summary['admission_gate_count']};",
        f"- total gate bindings: {summary['total_gate_binding_count']};",
        f"- hidden-state candidates: {summary['hidden_state_candidate_count']};",
        f"- linked next queue ids: {summary['linked_next_queue_id_count']};",
        f"- promoted mechanisms: {summary['promoted_mechanism_count']}.",
        "",
        "## Candidates",
        "",
        "| Rank | Candidate | Level | Priority | Hidden-State License |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for candidate_item in payload["candidates"]:
        lines.append(
            f"| {candidate_item['priority_rank']} | `{candidate_item['candidate_id']}` | "
            f"`{candidate_item['level_id']}` | `{candidate_item['priority_class']}` | "
            f"`{candidate_item['hidden_state_license']}` |"
        )

    for candidate_item in payload["candidates"]:
        lines.extend(
            [
                "",
                f"## {candidate_item['title']}",
                "",
                f"- candidate id: `{candidate_item['candidate_id']}`;",
                f"- level id: `{candidate_item['level_id']}`;",
                f"- admission class: `{candidate_item['admission_class']}`;",
                f"- first run type: `{candidate_item['first_run_type']}`;",
                f"- linked queue ids: `{format_value(candidate_item['linked_next_queue_ids'])}`;",
                "",
                "Behavior design:",
                "",
                candidate_item["behavior_design"],
                "",
                "Why this candidate:",
                "",
                candidate_item["why_this_candidate"],
                "",
                "First-run panels:",
            ]
        )
        for panel in candidate_item["first_run_panels"]:
            lines.append(f"- {panel}")
        lines.extend(["", "Dumb explanations to test first:"])
        for explanation in candidate_item["dumb_explanations"]:
            lines.append(f"- {explanation}")
        lines.extend(
            [
                "",
                "Decision rules:",
                f"- `promotion_rule`: {candidate_item['promotion_rule']}",
                f"- `death_rule`: {candidate_item['death_rule']}",
                f"- `containment_rule`: {candidate_item['containment_rule']}",
                f"- `export_rule`: {candidate_item['export_rule']}",
            ]
        )

    lines.extend(
        [
            "",
            "## What This Proves",
            "",
            "It proves that future knowledge-substrate work now has an executable",
            "queue shape: candidate, first-run behavior panels, admission-gate",
            "bindings, dumb baselines, and promote/kill/export rules.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not prove that any candidate will pass. It does not promote",
            "a mechanism or license hidden-state discovery.",
            "",
        ]
    )
    return "\n".join(lines)


def write_knowledge_candidate_queue(
    output_path: Path = KNOWLEDGE_CANDIDATE_QUEUE_PATH,
    report_path: Path = KNOWLEDGE_CANDIDATE_QUEUE_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_control_surface_knowledge_candidate_queue()
    validate_knowledge_candidate_queue(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write candidate queue artifacts")
    parser.add_argument("--json", action="store_true", help="print candidate queue JSON")
    args = parser.parse_args()

    payload = build_control_surface_knowledge_candidate_queue()
    validate_knowledge_candidate_queue(payload)

    if args.write:
        write_knowledge_candidate_queue()
        print(
            f"wrote {KNOWLEDGE_CANDIDATE_QUEUE_PATH.relative_to(ROOT).as_posix()} and "
            f"{KNOWLEDGE_CANDIDATE_QUEUE_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['candidate_count']} candidates"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "knowledge candidate queue ok: "
        f"{payload['summary']['candidate_count']} candidates, "
        f"{payload['summary']['total_gate_binding_count']} gate bindings"
    )
    print(
        "candidate_count_by_level:",
        json.dumps(payload["summary"]["candidate_count_by_level"], sort_keys=True),
    )
    print("hidden_state_candidates:", payload["summary"]["hidden_state_candidate_count"])


if __name__ == "__main__":
    main()
