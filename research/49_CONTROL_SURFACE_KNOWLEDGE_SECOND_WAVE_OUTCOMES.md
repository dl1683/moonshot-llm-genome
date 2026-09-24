# Control-Surface Knowledge Second-Wave Outcomes

Status: generated second-wave outcome layer implemented and validated.

Machine-readable source:

> `data/control_surface_knowledge_second_wave_outcomes.json`

Builder:

> `code/control_surface_knowledge_second_wave_outcomes.py`

Regenerate:

```powershell
python code\control_surface_knowledge_second_wave_outcomes.py --write
python code\control_surface_knowledge_second_wave_outcomes.py
```

## Summary

- completed work orders: `5`
- pending work orders: `0`
- behavior-ready rows: `1`
- signature-screen-allowed rows: `1`
- killed routes: `3`
- hidden-state-claim rows: `0`
- intervention-allowed rows: `0`

## KSQ001 Parseability Bound Outcome

- work order: `bound_ksq001_familiar_prior_parseability`
- repair verdict: `parseability_repair_failed`
- route decision: `closeout_familiar_prior_parseability_tradeoff`
- exported diagnostic: `FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF`
- behavior ready: `false`
- signature screen allowed: `false`

### KSQ001 Full-Run Evidence

- result: `results/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND/ksq001_familiar_prior_parseability_bound_full_behavior.json`
- sources: `40`
- records: `480`
- selected template: `compact_original_replay`
- first-run compact parseability: `0.600`
- parseability delta: `0.000`
- selected primary labels: `{"artificial_value": 18, "real_prior": 2, "unknown": 4, "unparsed": 16}`

| Template | Passed | Parseable | Artificial | Prior/Lure | UNKNOWN | Collapse |
| --- | --- | --- | --- | --- | --- | --- |
| `compact_original_replay` | `false` | `0.600` | `0.450` | `0.050` | `0.100` | `None` |
| `compact_begin_city` | `false` | `0.825` | `0.525` | `0.000` | `0.300` | `None` |
| `compact_city_name_first` | `false` | `0.875` | `0.075` | `0.000` | `0.800` | `None` |

### KSQ001 Interpretation

The familiar-prior mixture is scale-fragile and answer-shape sensitive. Softer answer formatting can improve parseability on some rows, but it removes the prior branch or pushes rows toward UNKNOWN rather than producing a clean behavior substrate.

The 10-source smoke candidate did not scale. The selected full
compact replay exactly reproduces the original conflict parseability
boundary: 24/40 parseable rows with 18 artificial-value answers, 2
real-prior answers, 4 UNKNOWN answers, and 16 unparsed rows. Softer
answer-shape variants raise parseability only by losing the prior
branch or pushing rows toward UNKNOWN. Ordinary KSQ001 parseability
repair is therefore closed rather than polished.

## KSQ002 Repair Outcome

- work order: `repair_ksq002_source_rewrite_holdout`
- repair verdict: `source_rewrite_repair_broke_locality_controls`
- route decision: `kill_ordinary_source_rewrite_repair`
- exported diagnostic: `SOURCE_REWRITE_REPAIR_LOCALITY_REGRESSION`
- behavior ready: `false`
- signature screen allowed: `false`

### KSQ002 Full-Run Evidence

- result: `results/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR/ksq002_source_rewrite_holdout_repair_full_behavior.json`
- sources: `40`
- records: `240`
- selected template: `city_field_rewrite`
- rewrite delta from baseline: `0.250`

| Panel | Label Counts | Parseable | Artificial | UNKNOWN | Prior/Lure |
| --- | --- | --- | --- | --- | --- |
| `baseline_source_value_lookup` | `{"artificial_value": 30, "unparsed": 10}` | `0.750` | `0.750` | `0.000` | `0.000` |
| `neutral_rewrite_lookup` | `{"artificial_value": 40}` | `1.000` | `1.000` | `0.000` | `0.000` |
| `source_deletion` | `{"unknown": 10, "unparsed": 30}` | `0.250` | `0.000` | `0.250` | `0.000` |
| `query_only_control` | `{"lure_value": 7, "real_prior": 13, "unparsed": 20}` | `0.500` | `0.000` | `0.000` | `0.500` |
| `source_disjoint_rewrite_holdout` | `{"artificial_value": 40}` | `1.000` | `1.000` | `0.000` | `0.000` |
| `rewrite_output_geometry_audit` | `{"artificial_value": 40}` | `1.000` | `1.000` | `0.000` | `0.000` |

### KSQ002 Interpretation

For this familiar-entity source-rewrite substrate, improving the answer channel can strengthen prompt-present rewrite behavior while weakening baseline and null behavior. The control surface is coupled to the answer interface, not a clean source-channel variable.

The named holdout boundary was repaired: source-disjoint rewrite holdout
went to 16/16 artificial-value answers. That did not make KSQ002
behavior-ready because the same answer-channel repair reduced baseline
lookup to 30/40 and source-deletion UNKNOWN to 10/40. The ordinary
source-rewrite repair route is therefore killed rather than polished.

## KSQ003 Evidence-Sufficiency Redesign Outcome

- work order: `redesign_statusless_bridge_substrate`
- redesign verdict: `evidence_sufficiency_learned_branch_failed`
- route decision: `bound_evidence_sufficiency_without_hidden_state`
- exported diagnostic: `STATUSLESS_EVIDENCE_SUFFICIENCY_BOUNDARY`
- behavior ready: `false`
- signature screen allowed: `false`

### KSQ003 Full-Run Evidence

- result: `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_full_behavior.json`
- sources: `40`
- records: `840`
- selected template: `compact_identity`
- selected primary labels: `{"atomic_number": 13, "unknown": 27}`
- selected null-stress labels: `{"atomic_number": 41, "local_number": 5, "other_number": 5, "unknown": 66, "unparsed": 3}`

| Template | Complete Atomic | Complete Local | Contradiction UNKNOWN | Symbol UNKNOWN | Initial UNKNOWN | Answer-Absent UNKNOWN | Null-Stress UNKNOWN | Holdout Atomic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `identity_packet` | `1.000` | `0.000` | `0.025` | `0.000` | `0.000` | `1.000` | `0.008` | `1.000` |
| `observation_gate` | `0.725` | `0.200` | `0.150` | `0.000` | `0.000` | `1.000` | `0.050` | `0.625` |
| `compact_identity` | `0.325` | `0.000` | `0.650` | `0.800` | `0.200` | `0.950` | `0.550` | `0.125` |

### KSQ003 Interpretation

The second KSQ003 design changes the failure type. The original statusless evidence route collapsed mismatch and ablation rows to local lab numbers; the redesign mostly suppresses local intrusion, but exposes an evidence-sufficiency tradeoff. The identity-packet template gets 40/40 complete-evidence atomic answers while failing null stress almost completely. The compact template gets better UNKNOWN behavior but only 13/40 complete-evidence atomic answers. Statusless evidence can suppress the local table or activate learned recall, but this design cannot do both reliably.

The important result is not another local-table-dominance failure.
`identity_packet` produces 40/40 complete-evidence atomic answers,
but null-stress UNKNOWN is only 1/120 across contradiction and
single-feature ablations. `compact_identity` improves null stress
to 66/120 UNKNOWN and removes complete-conflict local answers, but
complete-evidence atomic answers fall to 13/40. The route is bounded
as an evidence-sufficiency boundary and remains hidden-state-disallowed.

## KSQ004 Template-Invariance Outcome

- work order: `adjudicate_ksq004_template_invariance`
- adjudication verdict: `template_invariant_bridge_behavior`
- route decision: `admit_behavior_substrate_only`
- exported diagnostic: `TEMPLATE_INVARIANT_BRIDGE_BEHAVIOR`
- behavior ready: `true`
- signature screen allowed: `true`
- passing templates: `["question_form", "relation_key_form"]`

### KSQ004 Full-Run Evidence

- result: `results/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION/ksq004_template_invariance_adjudication_full_behavior.json`
- sources: `40`
- records: `1080`
- selected template: `relation_key_form`

| Template | Passed | Conflict Expected | Atomic-Branch Atomic | Holdout Expected | Null UNKNOWN | Side Answer |
| --- | --- | --- | --- | --- | --- | --- |
| `question_form` | `true` | `0.850` | `0.700` | `0.875` | `1.000` | `0.000` |
| `neutral_sentence_form` | `false` | `0.675` | `0.525` | `0.750` | `0.925` | `0.000` |
| `relation_key_form` | `true` | `0.975` | `0.950` | `0.875` | `1.000` | `0.000` |

### KSQ004 Interpretation

The first-run template fragility was too coarse: compact underscore format collapsed expected-atomic rows, but a relation-key form with the same bare numeric answer channel generalized the bridge. The control surface is answer-interface and relation-format dependent.

`question_form` and `relation_key_form` pass the full behavior gate.
`neutral_sentence_form` does not: conflict expected-correct is 0.675
and expected-atomic conflict rows still collapse too often. This is a
bounded behavior-substrate admission, not a mechanism card.

## KSQ005/KSQ006 Relation-Evidence Answerability Outcome

- work order: `redesign_real_uncertainty_answerability`
- redesign verdict: `relation_evidence_unknown_nonce_failed`
- route decision: `kill_current_uncertainty_route`
- exported diagnostic: `RELATION_EVIDENCE_ANSWERABILITY_BOUNDARY`
- behavior ready: `false`
- signature screen allowed: `false`

### KSQ005/KSQ006 Full-Run Evidence

- result: `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_full_behavior.json`
- sources: `40`
- records: `840`
- selected template: `compact_relation`
- combined controls: `{"claim_only_reproduced_supported": 40, "control_abstain": 1, "mention_only_reproduced_supported": 39}`

| Panel | Label Counts | Parseable | Known | Supported | Abstain | Unsupported | Prior/True | Control Abstain | Control Reproduced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `known_factual_direct` | `{"known_correct": 39, "known_wrong_candidate": 1}` | `1.000` | `0.975` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `supported_relation_rows` | `{"supported_answer": 40}` | `1.000` | `0.000` | `1.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `unknown_nonce_absent_rows` | `{"abstain": 27, "unparsed": 13}` | `0.675` | `0.000` | `0.000` | `0.675` | `0.000` | `0.000` | `0.000` | `0.000` |
| `unsupported_relation_rows` | `{"abstain": 28, "unparsed": 2, "unsupported_answer": 10}` | `0.950` | `0.000` | `0.000` | `0.700` | `0.250` | `0.000` | `0.000` | `0.000` |
| `contradictory_relation_rows` | `{"abstain": 2, "prior_or_true_answer": 38}` | `1.000` | `0.000` | `0.000` | `0.050` | `0.000` | `0.950` | `0.000` | `0.000` |
| `claim_only_control` | `{"claim_only_reproduced_supported": 40}` | `1.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `1.000` |
| `mention_only_control` | `{"control_abstain": 1, "mention_only_reproduced_supported": 39}` | `1.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.025` | `0.975` |

| Template | Known | Supported | Unknown Abstain | Unsupported Abstain | Contradiction Abstain | Control Abstain | Control Reproduced |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `relation_rows` | `0.975` | `0.275` | `1.000` | `1.000` | `1.000` | `0.825` | `0.175` |
| `ledger_form` | `0.975` | `1.000` | `0.375` | `0.375` | `0.125` | `0.037` | `0.963` |
| `compact_relation` | `0.975` | `1.000` | `0.675` | `0.700` | `0.050` | `0.013` | `0.988` |

### KSQ005/KSQ006 Interpretation

A formal relation grammar separates two failure modes. The compact contract answers known facts and exact `REL capital_of` rows, but claim-only and mention-only controls still reproduce the capital on 79/80 rows and contradiction rows choose the true capital on 38/40 rows. The stricter relation_rows template protects controls and abstains on unknown/unsupported/contradictory rows, but answers only 11/40 supported relation rows. The model can use a visible relation grammar either as an answer trigger or as an abstention guard, but this route cannot make both roles stable together.

`compact_relation` answers known/direct rows at 39/40 and exact
`REL capital_of` supported rows at 40/40, but unknown nonce rows
only abstain at 27/40, unsupported rows abstain at 28/40,
contradiction rows choose the prior/true answer at 38/40, and
claim/mention controls reproduce the supported capital at 79/80.
`relation_rows` protects controls and abstention, but supported
relation answering collapses to 11/40. The current real-uncertainty
route is killed and no hidden-state screen is licensed.

## Allowed Claim

All five second-wave KSQ work orders have been executed: KSQ001 familiar-prior parseability repair was closed as a scale-fragile tradeoff, KSQ002 ordinary source-rewrite repair was killed by locality regression, KSQ003 evidence-sufficiency redesign exposed a statusless evidence boundary without admitting hidden-state work, KSQ004 template-invariance adjudication admitted a bounded behavior substrate and a later signature screen, and KSQ005/KSQ006 relation-evidence answerability killed the current real-uncertainty route.

## Forbidden Claim

The second-wave layer does not promote any knowledge-control surface, hidden-state claim, intervention, or internal mechanism claim.

## Validation Checks

| Check | Passed |
| --- | --- |
| `executed_work_orders_exist_in_topology` | `true` |
| `all_second_wave_artifacts_exist` | `true` |
| `ksq001_parseability_tradeoff_closed` | `true` |
| `repair_full_scope_preserved` | `true` |
| `holdout_fixed_but_locality_failed` | `true` |
| `ksq004_template_invariance_admitted` | `true` |
| `ksq003_evidence_sufficiency_boundary_recorded` | `true` |
| `ksq005006_relation_answerability_route_killed` | `true` |
| `no_second_wave_hidden_or_intervention_claims` | `true` |