# Control-Surface Atlas

Date: 2026-07-01

This document is the offensive complement to the project's defensive doctrine.

The defensive doctrine says:

> Do not promote weak behavior tables, output-visible probes, prompt-format
> classifiers, shuffled-selection accidents, dirty interventions, or null-broken
> stories into mechanism claims.

The offensive doctrine says:

> Measure the compositional law of small-model behavior: where each behavior
> lives across prompt contract, source tokens, output geometry, internal
> signatures, causal paths, null boundaries, and model scale.

The atlas is the main cumulative object. A mechanism card is one pixel. The
atlas is the image.

## Machine-Readable Atlas

The prose doctrine is backed by a structured atlas:

- data: `data/control_surface_atlas.json`
- law hypotheses: `data/control_surface_law_hypotheses.json`
- law audit: `data/control_surface_law_audit.json`
- next experiment queue: `data/control_surface_next_experiment_queue.json`
- artifact index: `data/control_surface_artifact_index.json`
- comparison: `data/control_surface_comparison.json`
- artifact registry: `code/control_surface_artifacts.py`
- comparison builder: `code/control_surface_comparison.py`
- law-audit builder: `code/control_surface_law_audit.py`
- next-queue builder: `code/control_surface_next_queue.py`
- validator: `code/validate_control_surface_atlas.py`
- validation command: `python code/validate_control_surface_atlas.py`

The validator checks row schema, controlled vocabularies, evidence-path
existence, law-hypothesis references, and hard result-artifact assertions for
the current MC005, MC006, MC007, MC008, MC009, MC010, MC011, MC012, MC013,
MC014, MC015, and MC016 boundary claims. It also runs
the artifact registry, which extracts linked result artifacts, fails direct
readiness contradictions between row states and artifact flags, and checks the
first forty-three family-level closure facts for MC001-MC016. It also verifies that
the checked-in artifact index matches the raw linked artifacts. The comparison
builder then aggregates the atlas and artifact index into cross-family verdict,
lead-time, intervention, mixture-axis, diagnostic, null-boundary, and artifact
coverage counts; validation fails if the checked-in comparison is stale, if any
row lacks full artifact/metric/family-check coverage, or if generic
claim-consistency rules find a row-level contradiction.
The law-audit builder then checks whether each law hypothesis is supported by
the atlas rows and diagnostics it cites; validation fails if the checked-in law
audit is stale, if a hypothesis cites unobserved evidence, or if any atlas row
has no law support.
The next-queue builder then compiles the validated law hypotheses into
prioritized next tests; validation fails if the checked-in queue is stale or has
no immediate/high-priority work.

Current validated snapshot:

- 18 family rows;
- 1 bounded mechanism card;
- 1 failed mechanism-card route;
- 17 diagnostic notes;
- 2 `lead_time_monitor_only` families;
- 6 rows with `OUTPUT_MARGIN_CONFUND`;
- 4 rows with `SIGNATURE_NOT_CAUSAL`.
- 1 row with `GLOBAL_MARGIN_SEPARATION_BLOCKS_MATCHING`;
- 1 row with `MARGIN_OVERLAP_TABLE_FAILED`;
- 1 row with `STRICT_FINAL_MARGIN_OVERLAP_ABSENT`;
- 1 row with `APPROXIMATE_PAIR_MATCHING_FAILED_MARGIN_BASELINES`;
- 1 row with `SOURCE_PATH_FINAL_MARGIN_SHADOW`;
- 1 row with `GREEDY_FINAL_MARGIN_SIGN_BARRIER`;
- 1 row with `MC005_BOUNDED_ATTENTION_WRITE_MEDIATION`;
- 1 row with `DELAYED_INTERFACE_CANDIDATE_SCORE_VISIBLE`;
- 1 row with `CANDIDATE_DECOUPLED_SHUFFLE_OVERFIT`;
- 1 row with `LOCKED_COORDINATE_TRANSFER_FAILED`;
- 1 row with `EXPANDED_CANDIDATE_DECOUPLED_BANK_INSUFFICIENT`;
- 1 row with `TRANSFER_ROLE_REPAIR_BANK_INSUFFICIENT`;
- 1 row with `DELAYED_CITY_ROUTE_CLOSED_MONITOR_ONLY`;
- 1 row with `MC007_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE`;
- 1 row with `SYMBOLIC_NULL_CONTROL_FAILED`;
- 1 row with `SYMBOLIC_CONFLICT_PARSEABILITY_FAILED`;
- 1 row with `SYMBOLIC_CONFLICT_CONTRAST_WEAK`;
- 1 row with `MC008_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE`;
- 1 row with `DERIVED_CODE_CONTROL_CONFLICT_TRADEOFF`;
- 1 row with `DERIVED_CODE_TYPED_SLOT_PROMPT_VISIBLE`;
- 1 row with `MC009_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE`;
- 1 row with `TWO_HOP_SYNTHETIC_LOOKUP_FAILED`;
- 1 row with `TWO_HOP_REAL_MEMORY_CONTROL_FAILED`;
- 1 row with `TWO_HOP_CONFLICT_CONTRAST_ABSENT`;
- 1 row with `MC010_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`;
- 3 rows with `NUMERIC_DIRECT_CONTROLS_PASSED`;
- 1 row with `NUMERIC_CONFLICT_CONTRAST_ABSENT`;
- 1 row with `MC011_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`;
- 1 row with `RELIABILITY_BEHAVIOR_CONTRAST_PASSED`;
- 1 row with `RELIABILITY_PROMPT_CHANNEL_VISIBLE`;
- 1 row with `MC012_BEHAVIOR_READY_PROMPT_VISIBLE_BRIDGE`;
- 1 row with `STATUS_CHANNEL_POSITIVE_CONTROL_REPRODUCED`;
- 1 row with `STATUS_CHANNEL_ABLATION_COLLAPSED_CONTRAST`;
- 1 row with `MC013_STATUS_ABLATION_FAILED_DIAGNOSTIC_BRIDGE`;
- 1 row with `CALIBRATION_STATUS_LABEL_ABSENT`;
- 1 row with `CALIBRATION_INFERENCE_CONFLICT_COLLAPSED`;
- 1 row with `MC014_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`;
- 1 row with `PARITY_GATE_LABELS_BALANCED`;
- 1 row with `PARITY_GATE_NOT_FOLLOWED`;
- 1 row with `MC015_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`;
- 9 falsifiable law hypotheses.
- 1 checked-in law audit validating hypothesis support against atlas rows and
  diagnostics.
- 53 unique atlas-linked result artifacts parsed by the registry.
- 45 parsed by the summary-schema parser and 7 parsed by the legacy-schema
  parser.
- 53 artifacts with extracted common/family metrics.
- all 19 atlas rows now have parsed result-artifact support.
- 41 enabled family-claim checks.
- 19/19 atlas rows with linked artifacts, extracted metrics, and family-level
  claim checks in the generated claim audit.
- 19 rows at `family_checked_with_metrics`.
- 0 rows at `family_checked_partial_metrics`.
- 120 generic row-claim consistency conditions checked.
- 0 claim-consistency contradictions.
- 9/9 law hypotheses with evidence-consistent audit levels.
- 19/19 atlas rows with law support.
- 0 unobserved diagnostics cited as law evidence.
- 19 generated next-queue items.
- 5 immediate next-queue items and 4 high-priority next-queue items.
- 1 checked-in compact artifact index validated against raw artifacts.
- 1 checked-in cross-family comparison validated against the atlas and artifact
  index.
- 0 promoted mechanism cards by current comparison.
- 0.052632 bounded-mechanism ratio.
- 0.947368 diagnostic-or-failed ratio.
- 0.315789 output-margin-confounded row ratio.
- 0.368421 behavior-substrate-failed row ratio.
- 0.736842 intervention not-allowed-or-failed ratio.

This is already a scientific result: the current map is dominated by
output-visible and diagnostic/boundary outcomes, not by clean local circuits.

## Reviewer Synthesis

The current experiments are saying something larger than "we have not found
enough clean circuits yet."

They are saying that small-model behavior is distributed across a mixture of
surfaces:

- visible prompt authority;
- output/candidate geometry;
- source-token dependence;
- broad residual state;
- localized attention/write paths;
- model-size-specific null fragility.

The mixture is the object. If capital-fact behavior is repeatedly
candidate-score-visible by the final token, that is not just a failed hidden
signature. It is evidence about when that behavior becomes answer-geometry. If
lookup mediation is exact on high-margin lookup rows but answer-absent rows
flip at low margins, that is not just a dirty null. It is a boundary condition
on the intervention. If early probes work until final output margin kills them,
that is not just another blocked promotion. It identifies the pre-decision
frontier.

This turns the project away from collecting passing cards and toward measuring
the law that determines why cards pass, bound, or die.

## Controls As Instruments

Controls should not be treated only as filters that delete claims.

Every control asks where the behavior is living:

- output-margin controls ask whether the label is already answer-geometry;
- prompt-format controls ask whether the probe is reading visible formatting;
- source-deletion and rewrite controls ask whether source removal is enough;
- shuffled-selection controls ask whether the search process overfit;
- null rows ask where the intervention boundary breaks;
- model-size tests ask whether the surface is architectural, scale-specific, or
  prompt-contract-specific.

When a result dies, the control that kills it becomes a measurement. The
project should preserve the typed death with the same care as a positive
effect.

## Asymptotic Conservatism

The project has a real risk: the control bar can rise monotonically until no
claim can pass, including claims that are true but structurally unable to beat
the wrong comparison.

The answer is not to lower the bar. The answer is to study the bar.

For example, a final candidate-score margin can be a confound for mechanism
promotion and a faithful downstream shadow of an upstream retrieval process.
Both can be true. If the project only records "candidate-score confounded," it
misses the temporal fact that the behavior may have become committed before
the final token. That is why lead-time must be first-class.

The right question becomes:

> At what stage does the behavior become output-visible, and is there any
> earlier stage where internal state predicts before same-stage output geometry
> does?

If the answer is no, the result is `zero_lead_time`, not an empty failure. If
the answer is yes but intervention fails, the result is `lead_time_monitor_only`.
These are atlas facts.

## North Star

The largest objective is not to find "the truth vector" or collect isolated
circuits.

The largest objective is:

> Build a predictive science of small-LLM control surfaces: given a behavior
> and prompt contract, predict which internal surfaces will appear, which
> baselines will explain them, which interventions will work, and which nulls
> will break.

This makes typed failure a first-class result. A result killed by
`candidate_score_confounded` is not merely dead. It tells us that, under that
prompt contract, the behavior has already become visible to candidate scoring.
A result killed by `null_row_low_margin_flip` tells us where the intervention
boundary lives. A result killed by `signature_not_causal` tells us that the
probe found a monitor, not a lever.

The genome is the distribution of these outcomes across behavior families.

## The Mixture

Every behavior should be measured as a mixture across these surfaces:

| Surface | Question |
| --- | --- |
| Prompt authority | Is the behavior mostly determined by visible instruction/source authority? |
| Prompt format | Does the signature classify format, requested mode, or rendered prompt shape? |
| Source-token dependence | Does removing/replacing source text reproduce the intervention? |
| Output geometry | Is the label already visible in next-token or candidate-score margins? |
| Lead-time internal signal | Is there a pre-output window where hidden state predicts better than same-position output margin? |
| Local internal path | Is the effect localized to a head, layer, slice, band, write, or broad aggregate? |
| Causal intervention | Does steering, masking, patching, replacement, or editing move behavior in the predicted direction? |
| Null locality | Does the intervention stay quiet on answer-absent, off-target, side-row, or no-reference rows? |
| Transfer | Does the surface survive model-size, model-family, layout, lexicon, task, or prompt-contract shifts? |

The project should stop treating broad/prompt-coupled/output-visible behavior
as disappointing residue. The ratio itself is the scientific object.

## Lead-Time Axis

Lead-time is now a required atlas axis.

Definition:

> Lead-time is the earliest token/layer stage where an internal signal predicts
> the eventual behavior better than same-position output geometry, before the
> final answer interface makes the label visible.

Each family should record:

- earliest tested stage;
- earliest supported hidden signal;
- same-stage output-margin AUC or effect;
- final-output/candidate-score AUC or effect;
- whether lead-time survived shuffled-label selection;
- whether lead-time survived subgroup and source-disjoint holdouts;
- whether any intervention from that stage worked.

Lead-time outcomes:

| Outcome | Meaning |
| --- | --- |
| `zero_lead_time` | No useful signal appears before the final output state. |
| `lead_time_output_shadow` | Signal appears early but is matched by same-stage output geometry. |
| `lead_time_monitor_only` | Signal beats same-stage output but fails intervention. |
| `lead_time_causal_dirty` | Intervention works but controls/nulls match or side effects break locality. |
| `lead_time_causal_clean` | Signal beats same-stage controls and intervention works with locality. |

MC006 V16/V17/V18/V19/V20/V21/V22/V23/V24/V25/V26/V27/V28 currently sits at `lead_time_monitor_only`: the
early signal reproduces, additive steering fails, the V14 holdout split has no
global-margin overlap, the expanded V19 prompt bank improves balance without
reaching strict overlap, and V20 shows strict final-margin overlap is absent in
that bank. V21 then shows that approximate 0.5z pair matching does not kill
candidate-score or final-output margin baselines. V22 maps the source/path and
later line-boundary lead-time curve, but the curve remains a final-margin
shadow rather than a promotion-ready signature. V23 then shows the stricter
point: under the current greedy binary generated-answer interface, final
next-token margin sign is the parsed-label boundary. V24 changes the answer
interface with delayed JSON city generation and breaks that first-token barrier,
but JSON-completion candidate scoring still beats the hidden monitor on holdout.
V25 finds a candidate-score-decoupled delayed-city template, then blocks hidden
promotion because shuffled-label selected searches also reach perfect holdout
AUC. V26 locks the V25 coordinate and shows it fails transfer to the only other
candidate-decoupled delayed-city template in the V24 bank. V27 expands the
delayed-city bank and shows source-role candidate decoupling is repairable, but
transfer-role coverage remains insufficient. V28 targets transfer-role repair
directly and finds one clean transfer template, but not the two required for a
source/transfer behavior bank. The delayed-city route is therefore closed as
monitor-only under the tested prompt families.

## Diagnostic Types

Use these reusable diagnostic classes across mechanism-card attempts. Keep the
lowercase artifact strings in JSON, but map them to these canonical atlas
types in reports.

| Canonical Type | Artifact Strings / Examples | Meaning |
| --- | --- | --- |
| `BEHAVIOR_SUBSTRATE_FAILED` | `real_mode_failed`, `parseability_failed`, `binary_volume_failed` | Labels do not yet mean what the mechanism claim needs them to mean. |
| `OUTPUT_MARGIN_CONFUND` | `output_margin_confounded`, final next-token AUC match | Hidden signal is no better than output logits. |
| `CANDIDATE_SCORE_CONFUND` | `candidate_score_confounded` | Hidden signal is no better than explicit candidate scoring. |
| `PROMPT_FORMAT_CONFUND` | prompt length/format baselines match | Probe classifies the rendered prompt family. |
| `REQUESTED_MODE_CONFUND` | `requested_mode_confounded` | Probe classifies explicit requested-mode text. |
| `SHUFFLED_SELECTION_OVERFIT` | shuffled-label p95 matches selected candidate | Layer/head/position selection overfits. |
| `SOURCE_DELETION_NOT_CIRCUIT` | input-mask effect matched by prompt rewrites | Source removal explains the apparent path effect. |
| `PROMPT_REWRITE_EQUIVALENCE` | rewrite arms match masks | Prompt-state recomputation explains the intervention. |
| `LOCALITY_FAILED` | wrong layer/head/slice matches or beats selected path | Intervention is not localized to the claimed path. |
| `MC005_BOUNDED_ATTENTION_WRITE_MEDIATION` | V29-V31 freeze layers-24-26 final-query attention-write mediation as bounded | Primary lookup mediation is exact on high-margin rows, but answer-absent null locality blocks full promotion. |
| `NULL_ROW_LOW_MARGIN_FLIP` | answer-absent low-margin flips | Null rows flip under intervention despite clean primary effect. |
| `MODEL_SIZE_NULL_FRAGILITY` | smaller model primary works but strict null fails | Primary behavior transfers but reliability does not. |
| `SIGNATURE_NOT_CAUSAL` | MC006 V17 `intervention_failed`; MC005 V27 row steering failure | Predictive signature does not become a control vector under tested intervention. |
| `SIDE_EFFECT_FAILED` | no-hint/locality/side-row corruption | Intervention changes off-target behavior. |
| `GLOBAL_OUTPUT_CONFOUNDED_LEADTIME` | V16 lead-time signal but final margin perfect | Earlier hidden signal exists, but final output geometry already exposes the label. |
| `GLOBAL_MARGIN_SEPARATION_BLOCKS_MATCHING` | V18 no holdout overlap in candidate-score or final-output margins | The row table cannot support a fair margin-matched hidden-signature claim. |
| `MARGIN_OVERLAP_TABLE_FAILED` | V19 expanded prompt bank has balanced labels but no strict margin overlap | Table construction moved toward a better substrate but still leaves labels output/candidate separated. |
| `STRICT_FINAL_MARGIN_OVERLAP_ABSENT` | V20 pooled V19 bank lacks final-output overlap | No matched-template or per-source subset can rescue strict overlap without new generated rows. |
| `APPROXIMATE_PAIR_MATCHING_FAILED_MARGIN_BASELINES` | V21 approximate 0.5z pairs still have perfect candidate/final margin pair accuracy | Near-pair matching is too weak to neutralize output/candidate ordering. |
| `SOURCE_PATH_FINAL_MARGIN_SHADOW` | V22 source/path curve has monitor structure but final margins still perfect | Source/path and line-boundary monitors exist, but the current row bank remains final-output/candidate visible. |
| `GREEDY_FINAL_MARGIN_SIGN_BARRIER` | V23 final next-token margin sign predicts every binary generated label | Strict final-margin overlap is malformed as a default gate for greedy first-token generated binary rows. |
| `DELAYED_INTERFACE_CANDIDATE_SCORE_VISIBLE` | V24 delayed JSON city interface breaks first-token sign barrier but JSON-completion candidate scoring reaches 1.000 holdout AUC | Fixing first-token answer format is insufficient when full-completion candidate geometry still exposes the label. |
| `CANDIDATE_DECOUPLED_SHUFFLE_OVERFIT` | V25 candidate-decoupled delayed-city template beats output/candidate controls but fails split-preserving shuffled-label selected-search p95 | Candidate-score decoupling is necessary but not sufficient; hidden selection must also beat null-selected layer/position searches. |
| `LOCKED_COORDINATE_TRANSFER_FAILED` | V26 locks V25 `after_mapping_line/layer_10`; source holdout AUC is 0.750 but transfer holdout AUC on `separate_task_weak` is 0.500 | A candidate-decoupled hidden coordinate is not reusable unless it transfers across candidate-decoupled prompt contracts. |
| `EXPANDED_CANDIDATE_DECOUPLED_BANK_INSUFFICIENT` | V27 finds 4 candidate-decoupled templates and 138 pooled binary rows, but only 1 predeclared transfer-role template passes | Larger MC006 banks need transfer-role prompt coverage, not just source-side candidate decoupling. |
| `TRANSFER_ROLE_REPAIR_BANK_INSUFFICIENT` | V28 finds one transfer-ready template with balanced holdout and JSON holdout AUC 0.1875, but the gate required at least two transfer-ready templates | Transfer-role candidate decoupling is possible but brittle; one template is not enough for hidden-state search or coordinate transfer. |
| `DELAYED_CITY_ROUTE_CLOSED_MONITOR_ONLY` | V14-V28 block the ordinary delayed-city promotion routes | Preserve MC006 as a monitor/confound map; stop repairing this route as if it were mechanism-promotion-ready. |
| `CONTRAST_ABSENT` | MC007 V1 source-value behavior pass with 0 real-prior/lure rows | Behavior is clean enough to map but lacks a label contrast for hidden-state probing. |
| `PROMPT_AUTHORITY_DIAL` | MC007 V2 numeric authority pressure | Prompt authority changes whether familiar entities behave like table keys or real-world facts. |
| `PROMPT_CONTRACT_PARSEABILITY` | MC007 V2/V3/V4 template-dependent strict-prefix parsing | Prompt contract changes whether generated behavior is measurable. |
| `SOURCE_DECLARATION_CITY_MISMATCH` | MC007 V4 source labels disagree with generated city fields | Explicit source declarations are not reliable proxies for generated-answer behavior. |
| `MC007_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE` | MC007 V1-V4 jointly close the first bridge route | Preserve V1 as source-value baseline and V2 as prior-pressure baseline; move to a materially different bridge family instead of prompt-only repair. |
| `SYMBOLIC_NULL_CONTROL_FAILED` | MC008 V1 selected `symbol_field`; answer-absent null was 40/40 parseable but only 30/40 `UNKNOWN` | Compact symbolic outputs can clean direct controls while still failing absence/locality behavior before intervention. |
| `SYMBOLIC_CONFLICT_PARSEABILITY_FAILED` | MC008 V1 primary conflict parseability was 193/240 = 80.4 percent | Short code outputs improve parsing but do not guarantee a hidden-state-ready conflict table. |
| `SYMBOLIC_CONFLICT_CONTRAST_WEAK` | MC008 V1 primary conflict rows had 176 artificial-code rows and only 8 real-symbol rows; V2 had 220 artificial-code rows and only 2 real-symbol rows | Authority pressure remained table-dominant under the first symbolic route. |
| `MC008_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE` | MC008 V1-V2 jointly close the first symbolic bridge route | Compact codes repair direct controls and V2 repairs nulls, but matching prompt-local task codes dominate generated conflict rows. |
| `TWO_HOP_SYNTHETIC_LOOKUP_FAILED` | MC010 selected `neutral_contract`; synthetic two-hop lookup produced only 23/40 task-code answers | Two-hop indirection did not even preserve the synthetic source path strongly enough to become a behavior substrate. |
| `TWO_HOP_REAL_MEMORY_CONTROL_FAILED` | MC010 real-world memory control produced only 12/40 real-symbol answers with 28/40 unparsed | The bridge failed both the prompt-local synthetic side and the learned-memory control side. |
| `TWO_HOP_CONFLICT_CONTRAST_ABSENT` | MC010 primary conflict produced 229 task-code rows and 0 real/lure-symbol rows | Removing direct source-value and row-position channels did not create a real-versus-task conflict mixture. |
| `MC010_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE` | MC010 full 800-row behavior gate failed before hidden-state work | Do not probe MC010; use it as evidence that ordinary two-hop symbolic repairs are still table-dominant. |
| `NUMERIC_DIRECT_CONTROLS_PASSED` | MC011 selected `neutral_numeric`; synthetic lookup, familiar lookup, real atomic-number recall, and answer-absent nulls were each 40/40 | Same-format numeric outputs can repair MC010's control-side failures. |
| `NUMERIC_CONFLICT_CONTRAST_ABSENT` | MC011 primary conflict produced 240 local-number rows and 0 atomic/lure-number rows | Answer-format repair is insufficient when prompt-local table authority determines every conflict row. |
| `MC011_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE` | MC011 full 800-row behavior gate failed before hidden-state work despite clean direct controls | Do not probe MC011; use it as evidence that the next bridge must change the source/evaluation contract, not only answer format. |
| `RELIABILITY_BEHAVIOR_CONTRAST_PASSED` | MC012 selected `compact_reliability`; direct controls and nulls were 40/40, trusted source rows were 40/40 local, and untrusted source rows were 39/40 atomic | A clean mixed local-versus-learned behavior table can be manufactured when source reliability is explicit. |
| `RELIABILITY_PROMPT_CHANNEL_VISIBLE` | MC012's contrast depends on visible trusted/untrusted source-status text | Behavior-ready is not signature-ready when the source channel itself carries the label. |
| `MC012_BEHAVIOR_READY_PROMPT_VISIBLE_BRIDGE` | MC012 full 560-row behavior gate passed but `prompt_channel_locality_gate_passed` was false | Do not probe MC012; use it as the prompt-visible positive control for future prompt-channel locality repairs. |
| `STATUS_CHANNEL_POSITIVE_CONTROL_REPRODUCED` | MC013 selected `compact_status_ablation`; statused trusted rows were 40/40 local and statused untrusted rows were 39/40 atomic | The MC012 contrast reproduces inside the MC013 runner when the visible status cue is present. |
| `STATUS_CHANNEL_ABLATION_COLLAPSED_CONTRAST` | MC013 matched ablation prompt pairs were text-identical, but the primary ablation conflict collapsed to 80/80 local rows and 0 atomic/lure rows | Removing the visible status channel destroys the learned-fact side under this prompt contract. |
| `MC013_STATUS_ABLATION_FAILED_DIAGNOSTIC_BRIDGE` | MC013 full 640-row behavior/locality gate failed before hidden-state work | Do not probe MC013; simple status-channel ablation is a killed bridge route, not a mechanism substrate. |
| `CALIBRATION_STATUS_LABEL_ABSENT` | MC014 primary prompts had no trusted/untrusted/reliable/unreliable/status lexemes and hid target/lure atomic numbers | The explicit status channel was removed structurally, not merely ignored in prose. |
| `CALIBRATION_INFERENCE_CONFLICT_COLLAPSED` | MC014 selected `calibration_rule`; calibration-consistent and calibration-inconsistent conflict rows both produced 40/40 local-number answers | Calibration evidence did not overcome prompt-local table dominance under this contract. |
| `MC014_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE` | MC014 full 560-row behavior gate failed before hidden-state work despite clean direct controls and nulls | Do not probe MC014; inferred source validity is a killed bridge route under this prompt contract. |
| `PARITY_GATE_LABELS_BALANCED` | MC015 structurally balanced expected local and expected atomic labels by split while hiding target/lure atomic numbers and removing source-status lexemes | Mixed-output claims must be checked against rule-aligned expected labels, not only output class counts. |
| `PARITY_GATE_NOT_FOLLOWED` | MC015 selected `parity_rule`; primary conflict rows were mixed but expected correctness was only 39/80, with expected-atomic rows selecting atomic only 12/40 | The learned atomic-number parity gate did not control generated answers under this prompt contract. |
| `MC015_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE` | MC015 full 840-row behavior gate failed before hidden-state work despite clean direct controls, nulls, prompt audit, and balanced expected labels | Do not probe MC015; rule-aligned learned-fact gating failed even though mixed local/atomic outputs appeared. |
| `ALPHABET_GATE_LABELS_BALANCED` | MC016 structurally balanced expected local and expected atomic labels by split while hiding target/lure atomic numbers and removing source-status lexemes | A visible non-status operational gate still needs rule-aligned expected-label evidence. |
| `ALPHABET_GATE_LOCAL_COLLAPSE` | MC016 selected `alphabet_rule`; primary conflict rows were 80/80 local-number answers despite 40 expected-atomic rows | The alphabet gate did not overcome prompt-local table dominance under this numeric contract. |
| `FEATURE_LABEL_GATE_DID_NOT_RESCUE` | MC016's explicit `feature_labeled_alphabet` template produced 61 local rows, 0 atomic rows, and 19 unparsed rows on primary conflict | Making the neutral first-letter feature visible did not reproduce MC012-style learned-atomic behavior. |
| `MC016_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE` | MC016 full 1120-row behavior gate failed before hidden-state work despite clean direct controls, nulls, prompt audit, and balanced expected labels | Do not probe MC016; visible non-status alphabet gating still collapsed expected-atomic rows to local answers. |

Typed failure is an exportable result. Every status card should name:

- promoted claim, if any;
- bounded surviving claim;
- killed claim;
- reusable diagnostic type.

## Verdict Classes

Every branch should end in one of four products:

| Product | Definition |
| --- | --- |
| Promoted mechanism card | Behavior, signature, intervention, and reliability gates pass. |
| Bounded mechanism card | Primary causal effect is real, but a documented boundary limits scope. |
| Failed mechanism card | The mechanism route is killed after preregistered controls or kill rules. |
| Diagnostic note | No mechanism claim, but a reusable failure mode or behavior fact is established. |

Positive and negative outcomes should have equal status if the claim boundary is
clear.

## Promotion, Death, Containment, Export

Before running a branch, write four rules:

| Rule | Question |
| --- | --- |
| Promotion | What exact evidence upgrades the claim? |
| Death | What exact evidence ends this intervention family? |
| Containment | If it half-works, what narrower claim survives? |
| Export | What diagnostic type or atlas cell becomes reusable? |

Example: MC005 write replacement.

- Promote if high-margin lookup mediation remains strong and answer-absent
  null flips vanish under predefined margin strata.
- Bound if lookup rows stay clean but low-margin answer-absent rows remain
  fragile after serious repair attempts.
- Kill write replacement if null flips persist across materially different
  write-replacement variants.
- Export `NULL_ROW_LOW_MARGIN_FLIP`.

Example: MC006 early capital-fact signature.

- Promote only if a pre-output signal beats same-position controls and a
  causal intervention changes behavior while preserving null/side rows, or if
  a margin-matched table shows the signal remains upstream of candidate/final
  output geometry.
- Bound as monitoring-only if it beats same-position output but fails
  intervention or if the row table has no fair margin-matching region.
- Kill simple additive steering after V17 unless a materially different dose or
  target is preregistered with a stronger rationale.
- Export `GLOBAL_OUTPUT_CONFOUNDED_LEADTIME`,
  `GLOBAL_MARGIN_SEPARATION_BLOCKS_MATCHING`, and `SIGNATURE_NOT_CAUSAL`.

## Depth/Width Rhythm

The project needs both deepening and widening.

Deepening asks:

> Where exactly is the surface, what intervention reaches it, and which null
> breaks it?

Widening asks:

> Does this surface survive a different model, task, prompt contract, layout,
> lexicon, or behavior level?

Rule:

> After a branch spends several iterations deepening one surface, the next
> serious iteration must either widen-test the finding or declare why widening
> is premature.

This prevents infinite polishing of a surface that cannot generalize. A failed
widening test is not a nuisance; it caps the claim.

## Cross-Family Atlas Table

This is the live atlas snapshot. It should be updated after every material
experiment.

| Family | Behavior Contract | Current Best Signal | Lead-Time State | Intervention State | Main Confound / Boundary | Transfer State | Current Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MC001 Qwen3-0.6B truth/agreement | Truth vs wrong-hint agreement under multiple prompt controls | Dense truth/agreement directions and broad source dependence | No clean early mechanism signal; answer/output and source-removal controls dominate | Raw steering and input/source masking move behavior, but mechanism routes failed | `SOURCE_DELETION_NOT_CIRCUIT`, `PROMPT_REWRITE_EQUIVALENCE`, output-margin confounds | Broad 0.6B route closed; move to larger/artifact-rich stacks | Failed mechanism / diagnostic control surface |
| MC001B Qwen3-1.7B truth/agreement | Escalated wrong-hint control surface | Raw h21 steering improves truth-following | Residualized/margin-clean signal failed | Raw dense steering works; residualized rescue failed | Output-margin / residualization failure | Larger model gives behavior control but not mechanism | Behavior control, mechanism blocked |
| MC001G Gemma truth/agreement | Gemma truth/agreement with repaired generated/pairwise interfaces | Dense signal exists; sparse features failed promotion | Matched generated and pairwise repairs still confounded by interface coverage/balance | Matched dense intervention and donor replacement failed controls | Answer-letter/order/interface confounds; sparse shuffle overfit | Gemma path useful as behavior-interface warning | Repaired-substrate / failed-intervention / sparse-promotion-failed |
| MC002 known/unknown | Known country vs unsupported nonce abstention/hallucination | No stable behavior substrate | Not reached | Not allowed | `BEHAVIOR_SUBSTRATE_FAILED`; pressure either too weak or damages locality | Base Gemma and IT variants failed different sides | Behavior-gated diagnostic |
| MC002B context support | Supported note vs unsupported near-neighbor | Improved clean supported/unsupported baseline | Not reached | Not allowed | Pressure contrast failed; lure/check prompts collapse behavior | Not yet widened | Behavior-gated diagnostic |
| MC003 delayed copy | Arbitrary delayed-copy code word under wrong-hint pressure | V2 behavior passed; final and early signatures tested | Early `after_wait_newline` signal appeared but failed shuffle/trace controls; V3 condition-balanced signature failed holdout | No supported intervention | Output margin, shuffled selection, condition-trace confound | Not yet widened | Behavior-supported, signature failed |
| MC004 in-context binding | Original note vs later update conflict | V2 behavior passed; final signature failed output margin | Pre-update `after_question` signal appeared but failed shuffle/subgroup controls | No supported intervention | Same-position output at final; lead-time shuffle/subgroup failure | Not yet widened | Behavior-supported, lead-time diagnostic failed promotion |
| MC005 synthetic lookup | In-prompt key/value lookup with `Response:` marker | Source-value path; layers 24-26 final-query attention writes; route frozen bounded after V29-V31 | Lookup is prompt-source mediated; lead-time less central than source-path mediation | Target source masking and write replacement exactly reproduce lookup effect on high-margin rows | `MC005_BOUNDED_ATTENTION_WRITE_MEDIATION`, rare answer-absent low-to-moderate-margin null flips | Qwen3-0.6B primary lookup replicated but strict null failed | Frozen bounded mechanism card |
| MC006 capital-fact override | Real capital vs fictional/task-local overwrite | V14 matched generated table; V16 lead-time signal; V19/V20/V21 table route blocked; V22 source/path curve mapped; V23 final-margin sign barrier diagnosed; V24 delayed interface repaired first-token barrier; V25 found a candidate-score-decoupled table but not a robust hidden selector; V26 locked that coordinate and failed transfer; V27 expanded the bank but failed transfer-role coverage; V28 directly repaired transfer-role templates but found only one ready transfer template; delayed-city route closed monitor-only | `lead_time_monitor_only`: source/path and later pre-output monitors exist, delayed JSON answers break the first-token city-margin barrier, V25 decouples one template from full-completion candidate scoring, V26 shows the locked coordinate does not transfer, V27 shows source-side candidate decoupling is easier than transfer-side coverage, V28 shows transfer-role decoupling is possible but not bank-stable, and the route is closed as monitor-only | V17 additive steering failed on holdout; no V22/V23/V24/V25/V26/V27/V28 intervention allowed | `GLOBAL_OUTPUT_CONFOUNDED_LEADTIME`, `GLOBAL_MARGIN_SEPARATION_BLOCKS_MATCHING`, `MARGIN_OVERLAP_TABLE_FAILED`, `STRICT_FINAL_MARGIN_OVERLAP_ABSENT`, `APPROXIMATE_PAIR_MATCHING_FAILED_MARGIN_BASELINES`, `SOURCE_PATH_FINAL_MARGIN_SHADOW`, `GREEDY_FINAL_MARGIN_SIGN_BARRIER`, `DELAYED_INTERFACE_CANDIDATE_SCORE_VISIBLE`, `CANDIDATE_DECOUPLED_SHUFFLE_OVERFIT`, `LOCKED_COORDINATE_TRANSFER_FAILED`, `EXPANDED_CANDIDATE_DECOUPLED_BANK_INSUFFICIENT`, `TRANSFER_ROLE_REPAIR_BANK_INSUFFICIENT`, `DELAYED_CITY_ROUTE_CLOSED_MONITOR_ONLY`, `SIGNATURE_NOT_CAUSAL` | Only Qwen3-1.7B tested on current V14-V28 path | Knowledge-like behavior supported; delayed-city route closed as monitor-only after selection nulls, transfer failure, expanded-bank insufficiency, and direct transfer-role repair failure |
| MC007 semi-synthetic familiar-entity lookup | Familiar countries and synthetic keys mapped to artificial task cities | V2 numeric authority dial creates the best partial real-prior/lure contrast; V3 answer-slot repair failed; V4 authority-source interface failed controls and source-city consistency; V1-V4 route closed diagnostic | Not reached; behavior table blocks probing and the current route is closed | Not allowed | `CONTRAST_ABSENT`, `PROMPT_AUTHORITY_DIAL`, `PROMPT_CONTRACT_PARSEABILITY`, `SOURCE_DECLARATION_CITY_MISMATCH`, `MC007_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE` | Only Qwen3-1.7B tested | First bridge route closed as diagnostic; new bridge family required |
| MC008 symbolic fact-code arbitration | Chemical element names mapped to prompt-local artificial codes versus learned real-world symbols | V1 made synthetic lookup and real-symbol recall clean but failed nulls and conflict balance; V2 repaired nulls and direct controls but primary conflict remained table-dominant | Not reached; behavior table blocks probing and the route is closed | Not allowed | `SYMBOLIC_NULL_CONTROL_FAILED`, `SYMBOLIC_CONFLICT_PARSEABILITY_FAILED`, `SYMBOLIC_CONFLICT_CONTRAST_WEAK`, `MC008_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE`, `PROMPT_AUTHORITY_DIAL`, `PROMPT_CONTRACT_PARSEABILITY` | Only Qwen3-1.7B tested | First symbolic bridge route closed as diagnostic |
| MC009 derived-code arbitration | Chemical element names mapped to row-position-derived task codes versus learned real-world symbols | Membership prompts preserve direct controls but fail conflict balance; typed slots create conflict balance but break controls and expose source channel in prompt | Not reached; behavior table blocks probing and the route is closed | Not allowed | `DERIVED_CODE_CONTROL_CONFLICT_TRADEOFF`, `DERIVED_CODE_TYPED_SLOT_PROMPT_VISIBLE`, `MC009_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE`, `PROMPT_AUTHORITY_DIAL`, `PROMPT_CONTRACT_PARSEABILITY` | Only Qwen3-1.7B tested | First derived-code bridge route closed as diagnostic |
| MC010 two-hop fact-code arbitration | Element names or synthetic keys map to nonce handles, and handles map to prompt-local task codes versus learned chemical symbols | Full generated behavior gate failed: synthetic lookup 23/40 task-code, familiar lookup 38/40 task-code, real memory 12/40 real-symbol, answer-absent null 36/40 UNKNOWN, primary conflict 229 task-code and 0 real/lure-symbol | Not reached; behavior table blocks probing and the route is closed | Not allowed | `TWO_HOP_SYNTHETIC_LOOKUP_FAILED`, `TWO_HOP_REAL_MEMORY_CONTROL_FAILED`, `TWO_HOP_CONFLICT_CONTRAST_ABSENT`, `MC010_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`, `PROMPT_AUTHORITY_DIAL` | Only Qwen3-1.7B tested | Two-hop bridge route closed as diagnostic; ordinary symbolic/two-hop repairs are not enough |
| MC011 atomic-number code arbitration | Synthetic keys or element names map to prompt-local local lab numbers versus learned real-world atomic numbers | Same-format numeric answer interface repaired all direct controls: synthetic lookup 40/40 local, familiar lookup 40/40 local, real atomic-number recall 40/40 atomic, null 40/40 UNKNOWN; primary conflict still collapsed to 240 local-number rows and 0 atomic/lure-number rows | Not reached; behavior table blocks probing and the route is closed | Not allowed | `NUMERIC_DIRECT_CONTROLS_PASSED`, `NUMERIC_CONFLICT_CONTRAST_ABSENT`, `MC011_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`, `PROMPT_AUTHORITY_DIAL` | Only Qwen3-1.7B tested | Numeric bridge route closed as diagnostic; answer-format repair is not enough |
| MC012 reliability-labeled numeric arbitration | Prompt-local local lab numbers compete with learned atomic numbers under explicit trusted/untrusted source-status labels | `compact_reliability` passed direct controls, nulls, source-disjoint conflict balance, and candidate/output margin reporting: primary conflict had 40 local rows, 39 atomic rows, and 1 other-number row | Not reached; prompt-channel locality blocks hidden-state probing | Not allowed | `RELIABILITY_BEHAVIOR_CONTRAST_PASSED`, `RELIABILITY_PROMPT_CHANNEL_VISIBLE`, `MC012_BEHAVIOR_READY_PROMPT_VISIBLE_BRIDGE`, `PROMPT_AUTHORITY_DIAL` | Only Qwen3-1.7B tested | First clean mixed bridge behavior table, but prompt-visible by construction |
| MC013 status-channel ablation numeric arbitration | MC012-style numeric arbitration with statused positive controls and text-identical matched ablation prompts | `compact_status_ablation` reproduced the statused contrast, with 40/40 trusted local rows and 39/40 untrusted atomic rows, but matched ablation collapsed to 80/80 local rows and 0 atomic/lure rows | Not reached; behavior/locality gate blocks probing and the route is closed | Not allowed | `STATUS_CHANNEL_POSITIVE_CONTROL_REPRODUCED`, `STATUS_CHANNEL_ABLATION_COLLAPSED_CONTRAST`, `MC013_STATUS_ABLATION_FAILED_DIAGNOSTIC_BRIDGE`, `PROMPT_AUTHORITY_DIAL` | Only Qwen3-1.7B tested | Status-channel ablation route closed as diagnostic; the next bridge must create contrast without first assigning it via visible status text |
| MC014 inferred-reliability numeric arbitration | Local lab numbers compete with learned atomic numbers; source validity must be inferred from calibration rows rather than status labels | `calibration_rule` kept direct controls and nulls clean at 40/40 and removed status lexemes, but calibration-consistent and calibration-inconsistent conflict rows both selected local numbers at 40/40 | Not reached; behavior gate blocks probing and the route is closed | Not allowed | `CALIBRATION_STATUS_LABEL_ABSENT`, `CALIBRATION_INFERENCE_CONFLICT_COLLAPSED`, `MC014_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`, `NUMERIC_DIRECT_CONTROLS_PASSED`, `PROMPT_AUTHORITY_DIAL` | Only Qwen3-1.7B tested | Calibration-inference route closed as diagnostic; removing status labels is insufficient while table-local authority dominates |
| MC015 parity-gated numeric arbitration | Local lab numbers compete with learned atomic numbers; the local table controls only when a hidden standard atomic-number parity condition matches the visible rule | `parity_rule` kept direct controls and nulls clean at 40/40, hid target/lure atomic numbers, removed status lexemes, and balanced expected labels by split, but primary conflict expected correctness was only 39/80 despite mixed outputs | Not reached; behavior rule-following gate blocks probing and the route is closed | Not allowed | `PARITY_GATE_LABELS_BALANCED`, `PARITY_GATE_NOT_FOLLOWED`, `MC015_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`, `NUMERIC_DIRECT_CONTROLS_PASSED`, `PROMPT_AUTHORITY_DIAL` | Only Qwen3-1.7B tested | Parity-gated bridge route closed as diagnostic; mixed local/atomic outputs are not enough without rule-aligned learned-fact gating |
| MC016 alphabet-gated numeric arbitration | Local lab numbers compete with learned atomic numbers; the local table controls only when a visible non-status first-letter rule says the queried element range controls | `alphabet_rule` kept direct controls and nulls clean at 40/40, hid target/lure atomic numbers, removed status lexemes, and balanced expected labels by split, but primary conflict collapsed to 80/80 local rows and expected-atomic rows selected atomic 0/40 | Not reached; behavior rule-following gate blocks probing and the route is closed | Not allowed | `ALPHABET_GATE_LABELS_BALANCED`, `ALPHABET_GATE_LOCAL_COLLAPSE`, `FEATURE_LABEL_GATE_DID_NOT_RESCUE`, `MC016_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`, `NUMERIC_DIRECT_CONTROLS_PASSED`, `PROMPT_AUTHORITY_DIAL` | Only Qwen3-1.7B tested | Alphabet-gated bridge route closed as diagnostic; visible non-status feature gating is still not enough under this prompt-local table contract |

## Current Track Priorities

### Track A: Use MC005 As The Bounded Calibration Specimen

MC005 is no longer an open repair queue. It is frozen as
`MC005_BOUNDED_ATTENTION_WRITE_MEDIATION`.

The calibrated claim is narrow:

- layers 24-26 final-query attention-write replacement exactly reproduces the
  high-margin lookup source-mask effect;
- strict answer-absent null locality fails on rare low-to-moderate-margin rows;
- Qwen3-0.6B primary lookup transfer works before strict null reliability.

Future MC005 work is justified only as a materially different intervention, a
matched transfer/null panel, or a comparative baseline for a new bridge family.

### Track B: MC006 Delayed-City Closed Unless Stress-Tested

MC006 should not chase stronger final-token hidden classifiers.

The next MC006 table should not broaden prompt search under the same greedy
binary first-token interface and hope strict final-margin overlap appears. V23
shows that final next-token margin sign is the parsed-label boundary under
that interface. V24 shows that changing the interface to delayed JSON city
generation breaks the first-token barrier but still leaves full-completion
candidate scoring stronger than the hidden monitor. V25 shows that a
candidate-score-decoupled delayed-city template exists, but the hidden selector
fails shuffled-label selected-search nulls. If probing continues, the signature
target should stay pre-output and source/path-specific and the hidden-selection
procedure must be preregistered, regularized, and transfer-tested. V26 shows
that simply freezing the V25 coordinate is not enough. V27 shows the next bank
construction bottleneck is transfer-role coverage, and V28 shows direct
transfer-role repair produces only one ready transfer template:

- source-token positions;
- country-token positions;
- fictional mapping value positions;
- post-context but pre-question positions;
- intermediate residual states before answer commitment;
- attention/write paths from country and fictional-code tokens into later
  decision positions.

Simple additive steering from the current V16 direction is closed by V17 unless
a new preregistration changes the intervention family materially. The current
V14 row table is also closed for margin-matched promotion by V18 because
candidate-score and final-output margins have no true/override holdout
overlap. V19 shows that simply broadening prompt pressure can repair balance
and create near-matched pairs, but still leaves strict final-margin overlap
unrepaired. V20 closes strict-overlap selection inside the V19 bank. V21 then
kills the bounded approximate-pair diagnostic that V20 still allowed:
approximate 0.5z pairs have perfect candidate/final margin-baseline ordering on
holdout. V22 maps the source/path-specific curve and shows monitor structure,
but also shows the same final-margin shadow. V23 shows that strict final-margin
overlap is the wrong default gate for the current greedy binary answer
interface. V24 changes that interface and shows the next boundary: the
first-token barrier breaks, but JSON-completion candidate scoring reaches 1.000
holdout AUC. V25 shows the V24 bank does contain a candidate-decoupled template,
but the selected hidden monitor fails shuffled-label selected-search p95. The
remaining MC006 routes are: broaden the candidate-decoupled target enough to
support transfer-tested hidden coordinates, or run a materially different
intervention family as a known-confounded stress test, not mechanism promotion.
The V25 coordinate itself should not be repaired further without new data. V27
adds a narrower table-construction target, and V28 consumes it. Ordinary
transfer-template repair should now stop unless the prompt family changes
materially. The remaining live options are a preregistered known-confounded
causal stress test, or a new behavior family. The delayed-city MC006 route
itself is now closed as monitor-only.

### Track C: Close MC008 V1-V2 And Pick A New Bridge

MC007 V1-V4 should not be treated as a prompt-repair queue.

The first bridge route is closed as `MC007_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE`:
V1 is the clean source-value baseline, V2 is the prior-pressure baseline, V3
kills answer-slot/target-note repair, and V4 kills explicit source labels as a
proxy for generated city behavior.

The next bridge attempt was preregistered as:

> `research/prereg/MC008_SYMBOLIC_FACT_CODE_ARBITRATION.md`

The first full MC008 behavior run is:

> `research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

The allowed V2 null/authority repair is:

> `research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_V2_NULL_AUTHORITY_REPAIR_STATUS.md`

The route closeout is:

> `research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_ROUTE_CLOSEOUT_STATUS.md`

MC008 changed the bridge from country/city generation to element/symbol-code
arbitration. V1's `symbol_field` template successfully made synthetic lookup
and real chemical-symbol recall clean at the generated-answer surface. That is
real progress over MC007's open-city interface.

V1 did not pass the behavior gate:

- answer-absent null rows were 40/40 parseable but only 30/40 `UNKNOWN`;
- primary conflict parseability was 193/240 = 80.4 percent;
- primary conflict rows were table-dominant: 176 artificial-code rows versus
  only 8 real-symbol rows and 0 lure-symbol rows;
- holdout real/lure balance was absent, so hidden-state work remains blocked.

V2 then repaired the null boundary:

- synthetic lookup stayed clean at 39/40 artificial-code rows;
- real-world symbol recall improved to 40/40 real-symbol rows;
- answer-absent null rows became 40/40 `UNKNOWN`;
- primary conflict parseability rose to 96.25 percent.

But V2 still failed the bridge:

- primary conflict rows were 220 artificial-code rows versus only
  2 real-symbol rows and 0 lure-symbol rows;
- authority-0 rows produced only 1/40 real-symbol answers;
- non-holdout and holdout real/lure balance both failed.

The first MC008 symbolic route is therefore closed as a diagnostic bridge, not
a probe substrate.

Any future bridge must preserve:

- no true-symbol prompt leaks;
- generated-answer scoring;
- source-disjoint holdouts;
- answer-absent null rows;
- real-world memory controls;
- synthetic lookup controls;
- at least 90 percent parseability;
- both artificial-code and real-symbol/lure labels on non-holdout and holdout.

The next attempt should not be another ordinary prompt rewrite around the same
matching element-code table. It must change the behavior family, answer
representation, conflict construction, source visibility, or evaluation
interface materially.

The closed derived-code bridge preregistration is:

> `research/prereg/MC009_DERIVED_CODE_ARBITRATION.md`

The route-closeout smoke status is:

> `research/cards/MC009_DERIVED_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

MC009 removed the direct `entity -> artificial code` row that dominated MC008.
The task-local answer is derived from row position under a generic row-code
rule. This keeps prompt-local source structure and compact generated answers,
but tests whether direct source-value binding was the reason MC008 collapsed to
table-code behavior.

MC009 is now an atlas diagnostic row. The membership smoke validates direct
controls but fails conflict balance; the typed-slot smoke creates balance but
breaks controls and is prompt-visible by construction. Hidden-state work is not
allowed from either artifact.

The current two-hop bridge preregistration and behavior status are:

> `research/prereg/MC010_TWO_HOP_FACT_CODE_ARBITRATION.md`

> `research/cards/MC010_TWO_HOP_FACT_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

MC010 removed both major channels that killed MC008 and MC009: direct
entity-to-code source rows and row-position-derived answer slots. The task-local
answer required a two-hop path from entity or synthetic key to nonce handle to
task code.

The full 800-row generated behavior gate still failed. The selected
`neutral_contract` template had 23/40 synthetic task-code rows, 38/40 familiar
entity task-code rows, 12/40 real-symbol rows, 36/40 answer-absent `UNKNOWN`
rows, and 229 primary task-code conflict rows versus 0 real/lure-symbol rows.

MC010 is therefore an atlas diagnostic row, not a hidden-state substrate.
Two-hop indirection is not enough; the next bridge must change the answer
interface, task construction, or source/evaluation contract materially.

The current numeric bridge preregistration and behavior status are:

> `research/prereg/MC011_ATOMIC_NUMBER_CODE_ARBITRATION.md`

> `research/cards/MC011_ATOMIC_NUMBER_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

MC011 changed the answer interface instead of only changing the prompt-local
source path. Both the prompt-local value and the learned real-world value were
integers: local lab numbers versus atomic numbers.

The full 800-row generated behavior gate selected `neutral_numeric`. Direct
controls were clean: synthetic numeric lookup, familiar-entity numeric lookup,
real atomic-number control, and answer-absent nulls were each 40/40. The bridge
still failed because the primary conflict was 240 local-number rows and 0
atomic/lure-number rows.

MC011 is therefore a sharper diagnostic than MC010 on one axis. It removes the
"bad answer interface" explanation for the direct-control side, but it leaves
prompt-local authority as the complete conflict-row explanation. Hidden-state
work remains forbidden.

The current MC012 reliability-labeled numeric bridge preregistration and
behavior status are:

> `research/prereg/MC012_RELIABILITY_LABELED_NUMERIC_ARBITRATION.md`

> `research/cards/MC012_RELIABILITY_LABELED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

MC012 changed the source/evaluation contract rather than the answer format.
The selected `compact_reliability` template passed all direct controls and
nulls at 40/40, produced 40/40 trusted-source local-number rows, and produced
39/40 untrusted-source atomic-number rows. Source-disjoint holdout balance and
candidate/output margin reporting passed.

MC012 is the first clean mixed local-versus-learned bridge behavior table in
this branch. It is also not signature-ready: the source-status text is visible
by design, and `prompt_channel_locality_gate_passed` is false. This makes MC012
the prompt-visible positive control for the next bridge, not a hidden-state
substrate.

The current MC013 status-channel ablation preregistration and behavior status
are:

> `research/prereg/MC013_STATUS_CHANNEL_ABLATION_NUMERIC_ARBITRATION.md`

> `research/cards/MC013_STATUS_CHANNEL_ABLATION_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

MC013 kept the numeric interface and source bank, reproduced the MC012-style
statused positive control, and then removed the source-status cue in
text-identical matched ablation prompts. The selected `compact_status_ablation`
template passed direct controls and nulls, produced 40/40 statused trusted
local-number rows and 39/40 statused untrusted atomic-number rows, but produced
80/80 local-number rows and 0 atomic/lure rows in the matched ablation
conflict.

MC013 closes the simple "just ablate the status text" repair route. The next
bridge must create local-versus-learned contrast without first assigning the
answer rule through visible trusted/untrusted source-status text.

The current MC014 inferred-reliability preregistration and behavior status are:

> `research/prereg/MC014_INFERRED_RELIABILITY_NUMERIC_ARBITRATION.md`

> `research/cards/MC014_INFERRED_RELIABILITY_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

MC014 removed the explicit status label and asked the model to infer whether a
source table was valid from calibration rows. The selected `calibration_rule`
template passed direct controls and nulls at 40/40, hid target and lure atomic
numbers from conflict prompts, and used no trusted/untrusted/reliable/
unreliable/status lexemes in primary prompts. But the conflict still collapsed:
calibration-consistent rows were 40/40 local and calibration-inconsistent rows
were also 40/40 local.

MC014 closes calibration-inferred source validity as a rescue route under this
prompt contract. The next bridge must weaken prompt-local table dominance or
change the source/evaluation contract materially, not just replace explicit
status words with calibration evidence.

The current MC015 parity-gated preregistration and behavior status are:

> `research/prereg/MC015_PARITY_GATED_NUMERIC_ARBITRATION.md`

> `research/cards/MC015_PARITY_GATED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

MC015 replaced source-validity inference with a learned factual gate. The
selected `parity_rule` template hid target and lure atomic numbers, had no
source-status lexemes, balanced expected local/atomic labels by split, and kept
synthetic lookup, familiar lookup, real atomic-number recall, and answer-absent
null controls clean at 40/40. But the gate itself failed: primary conflict
expected correctness was 39/80, expected-local rows selected local only 27/40,
and expected-atomic rows selected atomic only 12/40.

MC015 closes parity-gated numeric arbitration as a rescue route under this
prompt contract. The next bridge must prove rule-aligned learned-fact gating,
not merely parseable mixed local/atomic outputs.

The current MC016 alphabet-gated preregistration and behavior status are:

> `research/prereg/MC016_ALPHABET_GATED_NUMERIC_ARBITRATION.md`

> `research/cards/MC016_ALPHABET_GATED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

MC016 replaced the learned parity gate with a visible non-status alphabetic
gate. The selected `alphabet_rule` template hid target and lure atomic numbers,
had no source-status lexemes, balanced expected local/atomic labels by split,
and kept synthetic lookup, familiar lookup, real atomic-number recall, and
answer-absent null controls clean at 40/40. But the conflict still collapsed:
primary conflict rows were 80/80 local, expected-local rows selected local
40/40, and expected-atomic rows selected atomic 0/40. The stronger
`feature_labeled_alphabet` template did not rescue the atomic side; it produced
61 local rows, 0 atomic rows, and 19 unparsed rows.

MC016 closes visible non-status alphabet gating as a rescue route under this
prompt contract. The next bridge must change the source contract or answer
interface enough that expected-atomic rows survive prompt-local table pressure.

### Track D: Build The Atlas

Every new experiment should update the atlas table:

- behavior family;
- prompt contract;
- behavior-gate state;
- lead-time state;
- same-position output control;
- final-output/candidate-score control;
- intervention route;
- locality scale;
- null boundary;
- transfer state;
- promoted/bounded/failed/diagnostic verdict.

This is how MC001-MC016 become cumulative data instead of separate stories.

## Operating Loop

Before proposing an experiment:

1. State the exact claim.
2. State the easiest false explanation.
3. Build or verify the behavior table.
4. Refuse hidden-state work until labels are clean.
5. Compare against output, candidate, prompt, and null baselines.
6. Intervene only after the signature has a reason to be causally upstream.
7. Treat every side effect as data.
8. Write allowed and forbidden claims immediately after the run.
9. Decide: promote, bound, kill, or diagnose.
10. Add the result to the atlas.

This is the anti-self-deception instrument. The point is not to make every
beautiful story fail. The point is to learn the law governing why each story
passes, bounds, or dies.
