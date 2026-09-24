# Control-Surface Knowledge Candidate Queue

Source updated_at: 2026-07-01

Status: generated behavior-only candidate queue implemented and validated.

Machine-readable artifact:

> `data/control_surface_knowledge_candidate_queue.json`

Builder:

> `code/control_surface_knowledge_candidate_queue.py`

Commands:

```powershell
python code\control_surface_knowledge_candidate_queue.py --write
python code\control_surface_knowledge_candidate_queue.py
python code\validate_control_surface_atlas.py
```

## Purpose

This queue is the operational layer below the admission protocol. It
does not run hidden-state work. It specifies behavior-only substrate
candidates and the dumb explanations that should kill them quickly if
the apparent mechanism is prompt-, output-, parser-, or interface-born.

## Generated Facts

- candidates: 6;
- admission packets covered: 3;
- admission gates per candidate: 11;
- total gate bindings: 66;
- hidden-state candidates: 0;
- linked next queue ids: 10;
- promoted mechanisms: 0.

## Candidates

| Rank | Candidate | Level | Priority | Hidden-State License |
| ---: | --- | --- | --- | --- |
| 1 | `ksq001_familiar_entity_prior_counterbalance` | `level_2_semi_synthetic_familiar_entity` | `high` | `forbidden_until_candidate_passes_admission` |
| 1 | `ksq003_bridge_statusless_evidence_aggregation` | `level_3_symbolic_or_learned_memory_bridge` | `immediate` | `forbidden_until_candidate_passes_admission` |
| 1 | `ksq005_uncertainty_grounded_answerability` | `level_5_real_abstention_uncertainty` | `high` | `forbidden_until_candidate_passes_admission` |
| 2 | `ksq002_familiar_entity_source_rewrite_equivalence` | `level_2_semi_synthetic_familiar_entity` | `medium` | `forbidden_until_candidate_passes_admission` |
| 2 | `ksq004_bridge_answer_interface_minimal_pairs` | `level_3_symbolic_or_learned_memory_bridge` | `high` | `forbidden_until_candidate_passes_admission` |
| 2 | `ksq006_uncertainty_context_support_counterfactuals` | `level_5_real_abstention_uncertainty` | `medium` | `forbidden_until_candidate_passes_admission` |

## Counterbalance familiar-entity priors against artificial values.

- candidate id: `ksq001_familiar_entity_prior_counterbalance`;
- level id: `level_2_semi_synthetic_familiar_entity`;
- admission class: `new_familiar_entity_substrate`;
- first run type: `behavior_substrate_admission_only`;
- linked queue ids: `["familiar_entities_can_collapse_to_lookup_keys__next_test_1", "familiar_entities_can_collapse_to_lookup_keys__next_test_2"]`;

Behavior design:

Use familiar entity names, but assign artificial values from a non-city answer space. Pair each entity with semantic-prior, source-local, and answer-absent panels so real-world familiarity can be measured instead of smuggled in as the label.

Why this candidate:

This is the smallest materially new MC007 successor because it keeps familiar names but removes city-answer proxying and explicit source-label authority.

First-run panels:
- source-local artificial value lookup
- semantic-prior lure panel
- answer-absent and irrelevant-source nulls
- source-disjoint familiar-entity holdout
- candidate/output margin baselines

Dumb explanations to test first:
- The artificial value is just easier to copy than the semantic prior.
- Entity familiarity is acting only as a prompt key, not as knowledge pressure.
- Prompt wording still tells the model to trust the local source.
- Output candidates or answer shape separate the labels.

Decision rules:
- `promotion_rule`: Admit only if familiar priors measurably compete with local artificial values while prompt and output baselines fail to explain the split.
- `death_rule`: Kill if behavior reduces to local copy, semantic prior recall, authority wording, or parse/answer-shape effects.
- `containment_rule`: If source-local lookup works without semantic competition, preserve it only as a familiar-key lookup diagnostic.
- `export_rule`: Export SEMANTIC_PRIOR_INTERFERENCE or FAMILIAR_ENTITY_LOOKUP_KEY_COLLAPSE.

## Use statusless evidence aggregation instead of source-validity labels.

- candidate id: `ksq003_bridge_statusless_evidence_aggregation`;
- level id: `level_3_symbolic_or_learned_memory_bridge`;
- admission class: `new_bridge_substrate_class`;
- first run type: `behavior_substrate_admission_only`;
- linked queue ids: `["authority_pressure_creates_contrast_before_clean_substrate__next_test_1", "behavior_substrate_first_or_everything_lies__next_test_2", "source_visible_lookup_localizes_more_than_parametric_override__next_test_3"]`;

Behavior design:

Build a local-versus-learned arbitration table where branch selection depends on multiple content facts that jointly imply source support, without trusted/untrusted labels, checksums, row codes, or direct fact-claim validity cues.

Why this candidate:

The top queue asks for MC012-level direct controls and conflict mixture without visible source-status text. This is the most direct substrate candidate for that pressure.

First-run panels:
- source-local direct control
- learned-fact direct control
- multi-evidence conflict rows
- evidence-ablation rows
- answer-absent nulls
- source-disjoint holdout
- candidate/output baselines

Dumb explanations to test first:
- The evidence features are still visible labels in disguise.
- Prompt-local table dominance determines all rows.
- Learned-fact rows fail direct recall under table pressure.
- Candidate margins reveal the branch before hidden states are inspected.

Decision rules:
- `promotion_rule`: Admit only if direct controls, conflict mixture, nulls, prompt-channel audits, holdout, and output/candidate baselines pass together.
- `death_rule`: Kill if the learned branch collapses under table pressure or the evidence features behave like visible status labels.
- `containment_rule`: Contain as a bridge diagnostic if it reveals a new typed failure before hidden-state work.
- `export_rule`: Export STATUSLESS_EVIDENCE_VISIBLE_CHANNEL or LEARNED_BRANCH_TABLE_PRESSURE_COLLAPSE.

## Build grounded answerability before refusal or uncertainty probing.

- candidate id: `ksq005_uncertainty_grounded_answerability`;
- level id: `level_5_real_abstention_uncertainty`;
- admission class: `new_real_uncertainty_substrate`;
- first run type: `behavior_substrate_admission_only`;
- linked queue ids: `["behavior_substrate_first_or_everything_lies__next_test_2", "final_state_output_geometry_dominance__next_test_2"]`;

Behavior design:

Create known, unknown, unsupported, and contradicted factual items from a frozen evidence source, then ask for generated short answers with abstention allowed but not requested by label text.

Why this candidate:

MC002/MC002B failed because pressure prompts defined the behavior. This candidate moves label grounding outside the prompt before testing uncertainty.

First-run panels:
- known factual direct rows
- unknown/nonce rows
- unsupported-context rows
- contradicted-context rows
- abstention nulls
- source-disjoint holdout
- requested-mode and output-margin baselines

Dumb explanations to test first:
- Abstention is caused by the instruction style.
- Known/unknown labels leak through entity or answer shape.
- The model refuses because the prompt requests caution, not because evidence is absent.
- Output margins separate answerable and unanswerable rows before any hidden signature.

Decision rules:
- `promotion_rule`: Admit only if answerability behavior survives prompt, requested-mode, label-balance, output-margin, null, and holdout gates.
- `death_rule`: Kill if abstention follows caution wording, answer schema, entity familiarity, or output margin.
- `containment_rule`: Contain as a refusal-template or output-geometry diagnostic if controls explain it.
- `export_rule`: Export LABEL_GROUNDING_FAILURE, REQUESTED_MODE_CONFOUND, or OUTPUT_MARGIN_CONFUND.

## Test whether familiar-entity source lookup survives neutral rewrites.

- candidate id: `ksq002_familiar_entity_source_rewrite_equivalence`;
- level id: `level_2_semi_synthetic_familiar_entity`;
- admission class: `new_familiar_entity_substrate`;
- first run type: `behavior_substrate_admission_only`;
- linked queue ids: `["coarse_source_ablation_overstates_circuit_locality__next_test_1", "coarse_source_ablation_overstates_circuit_locality__next_test_2"]`;

Behavior design:

Start from the cleanest familiar-entity lookup baseline and run neutral source rewrites, query-only variants, and source-deletion controls before any hidden-state or intervention work.

Why this candidate:

The prior MC007 route was too entangled with source wording; this candidate decides whether a familiar-entity surface survives as a source-value behavior after visible wording is neutralized.

First-run panels:
- baseline source-value lookup
- neutral rewrite lookup
- source deletion
- query-only control
- source-disjoint holdout

Dumb explanations to test first:
- The source text is the whole mechanism.
- The query token or entity name alone carries the answer.
- Rewrite differences change parseability rather than behavior.
- The model follows instruction tone, not a source-value binding.

Decision rules:
- `promotion_rule`: Admit only if neutral rewrites preserve the behavior and source deletion/query-only controls fail to reproduce it.
- `death_rule`: Kill if source deletion, query-only text, or prompt rewrite effects explain the behavior.
- `containment_rule`: Contain as a source-channel diagnostic if lookup works but rewrite equivalence fails.
- `export_rule`: Export SOURCE_REWRITE_EQUIVALENCE_FAILED or QUERY_ONLY_SOURCE_PROXY.

## Factor the bridge answer interface with minimal-pair outputs.

- candidate id: `ksq004_bridge_answer_interface_minimal_pairs`;
- level id: `level_3_symbolic_or_learned_memory_bridge`;
- admission class: `new_bridge_substrate_class`;
- first run type: `behavior_substrate_admission_only`;
- linked queue ids: `["behavior_substrate_first_or_everything_lies__next_test_2", "null_reliability_bottleneck__next_test_2"]`;

Behavior design:

Use matched minimal-pair answer interfaces where local and learned branches share answer length, type, frequency band, and parser shape, then test whether the bridge failure persists without answer-token or numeric-option shortcuts.

Why this candidate:

Many bridge rungs died through answer-interface and local-source salience artifacts. This candidate makes the answer interface the primary object under test before another route is attempted.

First-run panels:
- matched local/learned minimal pairs
- answer-shape and frequency controls
- side-number leakage panel
- null rows
- source-disjoint holdout
- candidate/output baselines

Dumb explanations to test first:
- The answer interface, not the branch rule, determines behavior.
- Numeric or token frequency shortcuts explain the branch.
- Side-number leakage substitutes for learned-fact arbitration.
- Balanced outputs make both branches fail rather than compete.

Decision rules:
- `promotion_rule`: Admit only if matched answer interfaces preserve direct controls and produce a real local-versus-learned conflict mixture.
- `death_rule`: Kill if balancing the interface removes the behavior contrast or exposes answer-token shortcuts.
- `containment_rule`: Contain as an answer-interface law candidate, not a knowledge mechanism.
- `export_rule`: Export ANSWER_INTERFACE_BRANCH_SHORTCUT or BRIDGE_MINIMAL_PAIR_CONTRAST_ABSENT.

## Use counterfactual context support rather than known/unknown prompts.

- candidate id: `ksq006_uncertainty_context_support_counterfactuals`;
- level id: `level_5_real_abstention_uncertainty`;
- admission class: `new_real_uncertainty_substrate`;
- first run type: `behavior_substrate_admission_only`;
- linked queue ids: `["behavior_substrate_first_or_everything_lies__next_test_2", "null_reliability_bottleneck__next_test_1"]`;

Behavior design:

Construct factual claims with matched supporting, irrelevant, contradicting, and insufficient contexts. The target behavior is support-sensitive answer or abstain, not generic caution.

Why this candidate:

This is the natural successor to MC002B, but with counterfactual context panels and prompt-mode controls specified before the run.

First-run panels:
- supported-context rows
- irrelevant-context rows
- contradicting-context rows
- insufficient-context rows
- claim-only rows
- source-disjoint holdout
- requested-mode and output baselines

Dumb explanations to test first:
- The model follows support words in the prompt.
- Contradiction rows are easier to parse than insufficient rows.
- Answer shape or length distinguishes supported and unsupported cases.
- The behavior is a refusal style, not evidence tracking.

Decision rules:
- `promotion_rule`: Admit only if support-sensitive behavior survives matched context controls and output/requested-mode baselines.
- `death_rule`: Kill if support language, answer shape, or caution prompting explains the behavior.
- `containment_rule`: Contain as context-support behavior only until a control-surviving signature and intervention exist.
- `export_rule`: Export CONTEXT_SUPPORT_PROMPT_CHANNEL or REFUSAL_TEMPLATE_LEAKAGE.

## What This Proves

It proves that future knowledge-substrate work now has an executable
queue shape: candidate, first-run behavior panels, admission-gate
bindings, dumb baselines, and promote/kill/export rules.

## What It Does Not Prove

It does not prove that any candidate will pass. It does not promote
a mechanism or license hidden-state discovery.
