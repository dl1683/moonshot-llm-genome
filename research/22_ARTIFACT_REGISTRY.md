# Control-Surface Artifact Registry

Date: 2026-07-02

Status: artifact extraction layer implemented, wired into atlas validation,
extended with first family-specific claim checks, and supplemented with
explicit KSQ substrate validators through KSQ012 plus generated third- through
eighth-wave diagnostic outcome layers, now extended through KSQ015 and the
eleventh-wave catalog-slash full-source diagnostic layer.

## Purpose

The atlas should not depend only on hand-maintained prose. Every mechanism-card
attempt should increasingly reduce to checked relationships between:

- the family-level atlas row;
- the linked result artifacts;
- the claimed diagnostic class;
- pass/fail and readiness flags;
- selected templates, layers, positions, or intervention coordinates;
- baseline, null, and intervention controls.

This registry is the first infrastructure step toward that discipline.

## Implementation

Parser:

> `code/control_surface_artifacts.py`

Generated compact index:

> `data/control_surface_artifact_index.json`

Generated cross-family comparison:

> `data/control_surface_comparison.json`

Generated claim audit:

> `data/control_surface_comparison.json` -> `claim_audit`

Generated claim consistency audit:

> `data/control_surface_comparison.json` -> `claim_consistency`

Validator integration:

> `code/validate_control_surface_atlas.py`

Generated KSQ007 -> KSQ007B third-wave outcome layer:

> `data/control_surface_knowledge_third_wave_outcomes.json`

Third-wave outcome builder:

> `code/control_surface_knowledge_third_wave_outcomes.py`

Generated KSQ008 fourth-wave outcome layer:

> `data/control_surface_knowledge_fourth_wave_outcomes.json`

Fourth-wave outcome builder:

> `code/control_surface_knowledge_fourth_wave_outcomes.py`

Generated KSQ009 fifth-wave outcome layer:

> `data/control_surface_knowledge_fifth_wave_outcomes.json`

Fifth-wave outcome builder:

> `code/control_surface_knowledge_fifth_wave_outcomes.py`

Generated KSQ010 sixth-wave outcome layer:

> `data/control_surface_knowledge_sixth_wave_outcomes.json`

Sixth-wave outcome builder:

> `code/control_surface_knowledge_sixth_wave_outcomes.py`

Generated KSQ011 seventh-wave outcome layer:

> `data/control_surface_knowledge_seventh_wave_outcomes.json`

Seventh-wave outcome builder:

> `code/control_surface_knowledge_seventh_wave_outcomes.py`

Generated KSQ012 eighth-wave outcome layer:

> `data/control_surface_knowledge_eighth_wave_outcomes.json`

Eighth-wave outcome builder:

> `code/control_surface_knowledge_eighth_wave_outcomes.py`

Generated KSQ013 ninth-wave outcome layer:

> `data/control_surface_knowledge_ninth_wave_outcomes.json`

Ninth-wave outcome builder:

> `code/control_surface_knowledge_ninth_wave_outcomes.py`

Generated KSQ014 tenth-wave outcome layer:

> `data/control_surface_knowledge_tenth_wave_outcomes.json`

Tenth-wave outcome builder:

> `code/control_surface_knowledge_tenth_wave_outcomes.py`

Generated KSQ015 eleventh-wave outcome layer:

> `data/control_surface_knowledge_eleventh_wave_outcomes.json`

Eleventh-wave outcome builder:

> `code/control_surface_knowledge_eleventh_wave_outcomes.py`

Commands:

```powershell
python code\control_surface_artifacts.py
python code\control_surface_artifacts.py --json
python code\control_surface_artifacts.py --write-index
python code\control_surface_comparison.py --write
python code\control_surface_comparison.py
python code\control_surface_knowledge_third_wave_outcomes.py --write
python code\control_surface_knowledge_fourth_wave_outcomes.py --write
python code\control_surface_knowledge_fifth_wave_outcomes.py --write
python code\control_surface_knowledge_sixth_wave_outcomes.py --write
python code\control_surface_knowledge_seventh_wave_outcomes.py --write
python code\control_surface_knowledge_eighth_wave_outcomes.py --write
python code\control_surface_knowledge_ninth_wave_outcomes.py --write
python code\control_surface_knowledge_tenth_wave_outcomes.py --write
python code\control_surface_knowledge_eleventh_wave_outcomes.py --write
python code\validate_control_surface_atlas.py
```

Current registry output:

- 53 unique atlas-linked result artifacts;
- 46 parsed by the `summary_schema` parser;
- 7 parsed by the `legacy_schema` parser;
- 53 artifacts with extracted family/common metrics;
- 3 MC001 artifacts;
- 2 MC001B artifacts;
- 4 MC001G artifacts;
- 4 MC002 artifacts;
- 1 MC002B artifact;
- 3 MC003 artifacts;
- 3 MC004 artifacts;
- 1 MC005 artifact;
- 14 MC006 artifacts;
- 4 MC007 artifacts;
- 2 MC008 artifacts;
- 5 MC009 artifacts;
- 1 MC010 artifact;
- 1 MC011 artifact;
- 1 MC012 artifact;
- 1 MC013 artifact;
- 1 MC014 artifact;
- 1 MC015 artifact;
- 1 MC016 artifact.
- one compact checked-in artifact index generated from the raw linked
  artifacts.
- one checked-in cross-family comparison generated from the atlas and compact
  artifact index.
- one checked-in claim audit generated from the comparison layer.
- one checked-in claim-consistency audit generated from the comparison layer.

The atlas validator now runs the registry and reports:

- artifact parser counts;
- linked artifact count;
- artifacts with extracted metrics;
- row/artifact readiness contradictions.
- 43 enabled family-claim checks.
- whether the checked-in artifact index is current.
- whether the checked-in comparison has any row without artifacts, without
  metrics, without family-level claim checks, or with partial metric coverage.
- whether the checked-in comparison has any verdict, intervention-state,
  lead-time, null-locality, behavior-gate, output-confound, or signature-causal
  contradiction under the generic claim-consistency rules.

The validator also now checks the post-atlas KSQ substrate artifacts that are
not part of the atlas-linked registry:

- KSQ001 first-run, smoke, and full behavior;
- KSQ002 first-run, smoke, and full behavior;
- KSQ003 first-run and smoke;
- KSQ004 first-run, smoke, and full behavior;
- KSQ005 first-run and smoke;
- KSQ006 first-run and smoke;
- KSQ007 nonce-evidence answerability structural and smoke artifacts;
- KSQ007B claim-channel boundary structural and smoke artifacts.
- KSQ008 neutral-evidence channel repair structural and smoke artifacts.
- KSQ009 schema-specific value lookup structural and smoke artifacts.
- KSQ010 two-stage codebook value lookup structural and smoke artifacts.
- KSQ011 answer-for syntax ablation structural and smoke artifacts.
- KSQ012 function-assignment wrapper repair structural and smoke artifacts.
- KSQ013 nonfunction representation screen structural and smoke artifacts.
- KSQ014 slash locality packet structural and smoke artifacts.
- KSQ015 catalog slash full-source packet structural and full behavior
  artifacts.

These side checks pin schema, row counts, source splits, selected templates,
diagnostic classes, decision flags, and panel label counts. The point is to
make typed knowledge-substrate failures as audit-stable as atlas mechanism
rows, even before they are eligible for hidden-state or intervention work.

The generated third-wave outcome layer then joins KSQ007 and KSQ007B into one
checked diagnostic chain: removing real-world priors and value mentions did not
produce a behavior-ready real-uncertainty substrate, because bare
`answer_for(entity)=value` syntax remains answer-bearing even when marked as
outside the counted evidence contract. The validator treats this as a
staleness-checked boundary, not as a hidden-state or mechanism license.

The generated fourth-wave outcome layer then checks KSQ008 as the first direct
repair attempt against that boundary. The selected `field_registry` template
suppresses many forbidden channels, but exact neutral evidence answers only
`5/10`, counted wrong-schema `answer_for(entity)=value` rows reproduce `5/10`,
and neutral evidence versus a forbidden bare alternate answers only `2/10`.
The validator treats this as `NEUTRAL_EVIDENCE_POSITIVE_FAILED`, not as a
behavior-ready substrate.

The generated fifth-wave outcome layer then checks KSQ009 as a sharper
schema-specific repair. The selected `kv_lines` template answers exact ALLOW
rows only `2/10`, counted wrong-schema `answer_for(entity)=value` rows
reproduce `9/10`, uncounted wrong-schema `answer_for` rows reproduce `7/10`,
and an uncounted `answer_for` alternate overrides the ALLOW row `9/10`. The
validator treats this as `SCHEMA_SPECIFIC_POSITIVE_FAILED`, not as a
behavior-ready substrate. The practical boundary is that explicit row labels
are insufficient when the competing syntax is itself answer-bearing.

The generated sixth-wave outcome layer then checks KSQ010 as a two-stage
codebook repair. The selected `tag_rows` template gets closer on the positive
branch, answering exact entity-code/code-value bridges `8/10`, but it still
fails the parse gate, selects a value on both conflict panels `10/10`,
reproduces counted `answer_for(entity)=value` rows `10/10`, and lets
`answer_for` alternates override the codebook bridge `9/10`. The validator
treats this as `CODEBOOK_POSITIVE_FAILED`, not as a behavior-ready substrate.
The practical boundary is that positive lookup can improve while locality,
conflict abstention, and answer-like syntax competition remain fatal.

The generated seventh-wave outcome layer then checks KSQ011 as the targeted
syntax ablation for the KSQ010 answer_for competition. Holding the same
`tag_rows` bridge fixed, exact bridge lookup answers `9/10`, but function-like
alternate assignment forms dominate: exact `answer_for` overrides `9/10`,
spaced `answer_for` overrides `10/10`, colon `answer_for` overrides `9/10`,
`answer_to` overrides `9/10`, `value_for` overrides `8/10`, prose value notes
override `7/10`, and quoted `answer_for` overrides `9/10`. Plain entity
assignment and bare alternate mention mostly preserve the bridge at `8/10`
each. The validator treats this as
`FUNCTION_ASSIGNMENT_ANSWER_CHANNEL_DOMINANCE`, not as a behavior-ready
substrate or hidden-state license. The practical boundary is that the
answer_for failure is not one exact lexical artifact; function-like assignment
syntax is itself an answer-channel control surface separable from generic
value salience.

The generated eighth-wave outcome layer then checks KSQ012 as the first direct
wrapper repair against that function-assignment channel. The selected
`tag_rows` bridge is restored: exact bridge lookup answers `9/10`, and the raw
`answer_for` positive control reproduces the alternate `9/10`. The repair
fails anyway. Inactive-block `answer_for` overrides `9/10`, comment-mark
overrides `8/10`, fenced text overrides `9/10`, below-cut text overrides
`9/10`, detached function/value rows override `7/10`, unrelated-entity
`answer_for` still overrides `6/10`, and the assignment-only inactive control
leaks the alternate `8/10`. Split-assignment-only and query-only controls
abstain `10/10`, but split/masked bridge panels are only partial. The validator
treats this as `FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK`, not as a
behavior-ready substrate. The practical boundary is that visible function-like
assignment text is unsafe by default; wrapper prose is too weak a control
surface.

The generated ninth-wave outcome layer then checks KSQ013 as the first
materially nonfunction repair screen after the wrapper failure. The selected
`tag_rows` bridge again answers exact bridge rows `9/10`, and the raw
`answer_for` positive control reproduces the alternate `9/10`. Nonfunction text
reduces the catastrophic assignment-channel failure, but it does not pass:
value-bank and decoy-pair bridge panels answer the bridge `8/10` with `2/10`
unparsed, metadata answers `7/10`, separated entity/value answers `6/10`, bare
alternate mention answers `8/10`, and separated entity/value-only control leaks
the alternate `4/10`. The catalog slash bridge panel answers `9/10`, but KSQ013
does not include a matching slash-only no-bridge locality control. The
validator treats this as `NONFUNCTION_REPRESENTATION_CONTROL_AND_REPAIR_LEAK`,
not as a behavior-ready substrate. The practical boundary is that the next
useful experiment is a slash-only locality packet, not hidden-state work.

The generated tenth-wave outcome layer then checks KSQ014 as the missing
slash-only locality packet. Exact bridge and raw `answer_for` both pass at
`9/10`. Catalog slash entity and reversed bridge panels answer `9/10` with
`1/10` unparsed, catalog slash decoy answers `9/10` with `1/10` abstain, and
every matched slash-only no-bridge control abstains `10/10`. Bare slash is not
clean: the bare slash bridge panel answers only `6/10` with `4/10` unparsed.
The validator treats this as `SLASH_LOCALITY_BRIDGE_LOSS`, not as a
behavior-ready substrate. The practical boundary is now split: catalog-labeled
slash deserves a full-source behavior packet; bare slash does not.

The generated eleventh-wave outcome layer then checks KSQ015 as that
full-source catalog-slash packet. It removes bare slash and keeps the same
`tag_rows` contract across all 40 sources. The result is not behavior-ready:
exact bridge answers `35/40` with `5/40` unparsed code-token outputs, raw
`answer_for` remains active at `39/40`, catalog slash entity answers `36/40`,
catalog slash decoy answers `35/40`, catalog slash reversed answers `35/40`,
and every catalog slash-only no-bridge control abstains `40/40`. The validator
treats this as `CATALOG_SLASH_BRIDGE_POSITIVE_FAILED`, not as a hidden-state
or mechanism license. The practical boundary is that catalog slash locality is
not the failure; the counted bridge substrate itself is not full-source
reliable under this contract.

## Normalized Fields

The registry extracts:

- result path;
- atlas row id;
- parser name;
- card id;
- run type;
- model id;
- record count where exposed;
- diagnostic class;
- pass/fail flag;
- behavior-ready flag;
- signature-ready flag;
- intervention-ready flag;
- selected templates, layers, positions, coordinates, or doses;
- failed criteria;
- null-related criteria;
- baseline/control fields;
- intervention/locality/side-effect fields.
- family-specific metrics where schemas expose them.

Family-specific extracted metrics currently include:

- MC001 truth/agreement baseline, source-mask, token-matched rewrite, rewrite
  parity, and position-count metrics;
- MC001B raw and residual h21 probe metrics plus generation-arm metrics;
- MC001G generated/pairwise behavior summaries and layer-14 intervention arm
  summaries;
- MC002/MC002B behavior-substrate audit and label-count metrics;
- MC003/MC004 behavior summaries and hidden-signature criteria metrics;
- MC005 layers-24-26 write boundary, lookup target write reproduction, nonzero
  null flips, and non-simple margin-cutoff null boundary;
- MC006 pre-output hidden AUCs, candidate/final-output AUCs, shuffle-null
  status, additive-steering causal/mechanism criteria, best-dose steering
  effects, pair-matching failure criteria, source/path shadow criteria,
  delayed-city monitor-only criteria, candidate-decoupled shuffle-null
  criteria, transfer-ready counts, ready-template counts, and pooled binary
  rows;
- MC007 primary artificial-value behavior, authority-dial, parseability-repair,
  and authority-interface control failures;
- MC008 symbolic direct-control, null, conflict-parseability, and
  conflict-balance panel counts;
- MC009 derived-code membership, typed-slot, conflict/null/control, and
  prompt-visible tradeoff counts;
- MC010 two-hop direct-control failure and table-dominant conflict counts;
- MC011 numeric direct-control and conflict-collapse counts;
- MC012 reliability-labeled direct controls, conflict balance, and
  prompt-channel criteria;
- MC013 statused positive-control and matched-ablation collapse counts;
- MC014 inferred-reliability direct-control and calibration-collapse counts;
- MC015 parity-gated direct-control, expected-label balance, mixed-output, and
  rule-following failure counts;
- MC016 alphabet-gated direct-control, expected-label balance, feature-label,
  and local-collapse counts.

## Consistency Checks

The first row/artifact contradiction checks are intentionally conservative.
They fail only on direct readiness contradictions, such as:

- an atlas row saying intervention is `not_allowed` while a linked artifact says
  `signature_ready: true`;
- an atlas row saying intervention is `not_allowed` while a linked artifact says
  `intervention_ready: true`;
- an atlas row saying lead-time is `not_reached` while a linked artifact says
  `signature_ready: true`;
- a promoted mechanism-card row linking an artifact that says
  `intervention_ready: false`.

The old hard assertion checks still handle exact metric claims for selected
artifacts.

The family-specific checks now also fail validation unless linked
artifacts prove:

- `mc001_source_mask_and_rewrite_equivalence_evidence`;
- `mc001b_raw_signal_stronger_than_residual_signal`;
- `mc001g_layer14_intervention_no_holdout_class_movement`;
- `mc002_behavior_substrate_failure`;
- `mc002b_context_support_pressure_failure`;
- `mc003_shuffle_and_condition_trace_signature_failure`;
- `mc004_leadtime_shuffle_and_subgroup_failure`;
- `mc005_l24_26_attention_write_boundary`;
- `mc005_lookup_effect_with_nonzero_null_flips`;
- `mc005_null_boundary_not_simple_margin_cutoff`;
- `mc006_additive_steering_failure`;
- `mc006_v16_output_candidate_confounded`;
- `mc006_v21_pair_matching_margin_baseline_failure`;
- `mc006_v22_source_path_final_margin_shadow`;
- `mc006_v24_delayed_city_monitor_only`;
- `mc006_v25_candidate_decoupled_shuffle_overfit`;
- `mc006_v28_one_transfer_template_not_bank`;
- `mc006_shuffle_null_and_transfer_role_failure`;
- `mc007_contrast_absent_and_authority_controls_failed`;
- `mc007_v1_source_lookup_without_conflict`;
- `mc007_v2_authority_dial_parseability_failure`;
- `mc007_v3_parseability_repair_failure`;
- `mc007_v4_source_declaration_control_failure`;
- `mc008_direct_controls_clean_before_null_failure`;
- `mc008_null_repair_conflict_contrast_absent`;
- `mc008_v2_null_repaired_conflict_absent`;
- `mc009_membership_controls_clean_conflict_failed`;
- `mc009_membership_tradeoff_and_typed_slot_control_failure`;
- `mc009_typed_slot_balance_breaks_controls`;
- `mc010_two_hop_direct_controls_failed`;
- `mc010_two_hop_conflict_table_dominant`;
- `mc011_numeric_direct_controls_clean`;
- `mc011_numeric_conflict_table_dominant`;
- `mc012_reliability_behavior_contrast_passed`;
- `mc012_reliability_prompt_channel_blocks_signature`;
- `mc013_statused_positive_control_reproduced`;
- `mc013_status_channel_ablation_collapses_contrast`;
- `mc014_inferred_reliability_direct_controls_clean`;
- `mc014_inferred_reliability_conflict_collapsed`;
- `mc015_parity_gate_direct_controls_clean`;
- `mc015_parity_gate_mixed_outputs_wrong_rule`;
- `mc016_alphabet_gate_direct_controls_clean`;
- `mc016_alphabet_gate_local_collapse`.

## What This Proves

This does not prove any new mechanism claim.

It proves that the atlas is becoming less hand-wavy:

- linked artifacts are parseable through a common interface;
- a compact machine-readable fact index exists for downstream comparison;
- a generated comparison layer now converts that index into project-level
  verdict, lead-time, intervention, mixture, diagnostic, null-boundary, and
  artifact-coverage counts;
- readiness flags are extracted rather than manually remembered;
- contradictions between high-level row states and artifact flags now fail
  validation;
- central MC001-MC016 closure facts, including granular MC007-MC016 bridge-route
  deaths, are checked against linked artifacts;
- claim-audit validation now fails if any atlas row has no linked result
  artifact, no extracted metrics, no family-level claim check, or only partial
  metric coverage;
- claim-consistency validation now fails if the high-level verdict,
  intervention state, lead-time state, null-locality field, behavior-gate
  diagnostic, output-confound diagnostic, or signature-causality diagnostic
  contradicts normalized artifact evidence;
- future rows have a stable place to expose controls before prose is written.

## What It Does Not Yet Prove

The registry does not yet extract every family-specific metric.

It does not yet replace status cards, because many important claims are still
stored in nested schema-specific summaries. The compact index is an audit
surface, not an exhaustive replacement for raw artifacts. It now performs
generic row-claim consistency checks, but it still does not prove every
family-specific verdict boundary. It blocks direct contradictions, checks 43
family-level closure facts, and normalizes the evidence needed for stronger
checks.

## Next Expansion

The next extraction layer should deepen family-specific parsers for:

- MC005 transfer and side-effect strata beyond the current layers-24-26/null
  boundary checks;
- MC006 output/candidate-control and transfer checks beyond the current V16,
  V21, V22, V24, V25, and V28 closure checks;
- MC007/MC008/MC009/MC010/MC011/MC012/MC013/MC014/MC015/MC016 behavior-substrate gates beyond
  the current granular closeout checks, especially if a future bridge creates a
  real probe-ready substrate;
- future post-MC016 bridge runs.

The target is to make the validator fail when:

- a row says a behavior gate passed but the artifact criteria say it failed;
- a row says hidden-state work is allowed but output/candidate controls match
  the signature;
- a row says null locality is clean but null criteria or side-effect fields fail;
- a row says a route is closed but linked later artifacts reopen it without a
  new preregistered route;
- the checked-in compact index omits fields needed by stricter comparison and
  verdict-audit layers.

The endpoint is boring: every new experiment should exit with artifacts that
the registry can parse, compare, and audit before anyone writes a mechanism
story.
