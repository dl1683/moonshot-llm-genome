# Control-Surface Next Experiment Queue

Date: 2026-07-01

Status: generated next-test queue implemented and validated.

This document interprets the machine-readable queue:

> `data/control_surface_next_experiment_queue.json`

Builder:

> `code/control_surface_next_queue.py`

Commands:

```powershell
python code\control_surface_next_queue.py --write
python code\control_surface_next_queue.py
python code\control_surface_smoke_diagnostics.py --write
python code\control_surface_smoke_diagnostics.py
python code\control_surface_bridge_ladder.py --write
python code\control_surface_bridge_ladder.py
python code\control_surface_mixture_law.py --write
python code\control_surface_mixture_law.py
python code\control_surface_decision_frontier.py --write
python code\control_surface_decision_frontier.py
python code\control_surface_route_disposition.py --write
python code\control_surface_route_disposition.py
python code\control_surface_transfer_matrix.py --write
python code\control_surface_transfer_matrix.py
python code\control_surface_reliability_matrix.py --write
python code\control_surface_reliability_matrix.py
python code\control_surface_error_taxonomy.py --write
python code\control_surface_error_taxonomy.py
python code\validate_control_surface_atlas.py
```

## Purpose

The atlas now has three increasingly strict layers:

- artifact evidence;
- law support;
- next-test pressure.

The mixture-law layer now sits beside this queue as the current map of what the
queue is trying to change: 11 behavior/bridge blockers, 6 output-geometry
shadows, 1 prompt-visible positive control, and 1 bounded internal-causal row.
The decision-frontier layer adds the timing baseline: 12 rows have no reached
frontier, 4 are output-visible or zero-lead, 2 are monitor-only, 1 is
causal-not-timing, and 0 are predecision causal candidates.
The route-disposition layer adds the branch-control baseline: 0
hidden-state-ready atlas routes, 11 closed-before-hidden-state rows, 3
output-shadow diagnostic baselines, MC005 bounded and frozen, MC006
monitor-only closed, and MC012 prompt-visible positive control.
The transfer-matrix layer adds the widening baseline: 0 transfer-ready
mechanisms, 14 transfer-untested rows, 2 low-transfer rows, 2 medium-transfer
rows, 1 failed-transfer row, MC005 as bounded transfer-fragile reference, and
MC006 as failed or bank-insufficient transfer route.
The reliability-matrix layer adds the promotion baseline: 0 full-reliability
mechanisms, 1 bounded reliability reference, 11 behavior/bridge-blocked rows, 3
output-shadow diagnostics, 2 monitor-only no-lever rows, 1 failed-intervention
route, and 1 prompt-visible positive control. It also records that every
current row still misses clean intervention, null/locality, robustness, and
transfer/widening gates.

The next-experiment queue is the third layer. It turns validated law hypotheses
into prioritized actions. It does not invent new claims. It compiles the
`next_tests` already attached to law hypotheses and ranks them by:

- tentative or single-row law support;
- bridge-route pressure;
- transfer gaps;
- intervention relevance;
- lead-time relevance;
- output/candidate-control relevance;
- behavior-substrate gate relevance;
- the current fact that the atlas has 0 promoted mechanism cards.

As of MC030, the generated queue also ingests the bridge-closure layers:
`data/control_surface_bridge_ladder.json`,
`data/control_surface_route_disposition.json`, and
`data/control_surface_error_taxonomy.json`. Bridge-related queue items now
carry active closure constraints from killed routes, so the next bridge cannot
quietly reopen the MC028-MC030 operation-leak family through another local
prompt guard, example-removal tweak, query-order edit, or answer-schema change
unless the branch, null, local, and side-number gates all pass together.

## Current Generated Facts

The current queue contains:

- 19 queue items;
- 5 `immediate` items;
- 4 `high` items;
- 9 `medium` items;
- 1 `watch` item;
- 9 immediate-or-high items.

Bridge-closure context:

- bridge-closure context available: `true`;
- 21 bridge rungs;
- 14 smoke rungs;
- 0 hidden-state-allowed bridge rungs;
- 0 clean unconfounded bridge rungs;
- 21 closed contract axes;
- recent closed rungs: `MC028`, `MC029`, `MC030`.

Reason-code counts:

- `ZERO_PROMOTED_MECHANISM_PRESSURE`: 12;
- `BEHAVIOR_SUBSTRATE_GATE`: 10;
- `BRIDGE_ROUTE_NEEDED`: 8;
- `SUPPORTED_PATTERN_FALSIFICATION`: 9;
- `TENTATIVE_PATTERN`: 8;
- `SINGLE_ROW_LAW_SUPPORT`: 6;
- `INTERVENTION_RELEVANT`: 4;
- `LEADTIME_FRONTIER`: 2;
- `OUTPUT_GEOMETRY_CONTROL`: 5;
- `TRANSFER_GAP`: 2;
- `STRONG_DOCTRINE_GUARDRAIL`: 2.

## Top Queue

| Rank | Priority | Action Type | Hypothesis | Next Test |
| ---: | --- | --- | --- | --- |
| 1 | `immediate` | `operational_rule` | `authority_pressure_creates_contrast_before_clean_substrate` | For the next bridge, preserve MC012-level direct controls and conflict mixture while making the answer rule independent of visible trusted/untrusted source-status text, proving that outputs follow the intended rule, showing that expected-atomic rows survive prompt-local table pressure, and beating neutral source-label order, definition-order, answer-option-order, local-source-salience, row-code prompt-local dominance, asymmetric conditional-arbitration, generic visible-route instability, semantic answer-source label, and rule-order-sensitivity controls. |
| 2 | `immediate` | `gate_rule` | `behavior_substrate_first_or_everything_lies` | For the next bridge, require MC012-level direct controls, conflict balance, nulls, parseability, source-disjoint holdout, candidate/output baselines, and an MC013/MC014/MC015/MC016-style prompt-channel and rule-following audit that does not collapse the learned-fact side. |
| 3 | `immediate` | `experiment` | `transfer_fails_at_reliability_before_primary_effect` | Run any future transfer test with matched null panels and side rows from the start. |
| 4 | `immediate` | `gate_rule` | `transfer_fails_at_reliability_before_primary_effect` | Do not mark a surface as transferred unless the atlas row's `null_locality` and transfer fields both move beyond bounded. |
| 5 | `immediate` | `gate_rule` | `coarse_source_ablation_overstates_circuit_locality` | Require source deletion, neutral rewrite, and query-only path comparisons in every source-token control-surface claim. |

## Interpretation

The queue is telling us the next scientific pressure is not "probe MC016."

The next pressure is:

> Preserve MC012-level direct controls, nulls, parseability, source-disjoint
> balance, and local-versus-learned mixture while making the answer rule
> independent of visible trusted/untrusted source-status text, proving that the
> generated answers follow the intended rule, and showing that expected-atomic
> rows survive prompt-local table pressure. After MC017/MC018, this must also
> beat neutral source-label order, source-definition order, answer-option
> order, and local-source-salience controls. After MC019, neutral row-local
> route codes are also insufficient unless expected-atomic rows follow the
> route rule under prompt-local table pressure. After MC020, the issue is not
> atomic recall availability: direct atomic instructions work with the queried
> local row present. The issue is asymmetric conditional arbitration when a
> rule points from a present local row to learned memory. After MC021, route
> codes are not a clean substrate even before learned memory: visible-visible
> route conflicts were still below behavior-gate quality. After MC022,
> semantic answer-source labels also fail as a clean substrate: nonlocal-first
> ordering can rescue visible-visible routing, but learned-memory routing still
> remains weak and local-source salience remains the default failure mode.
> After MC023, query-level operation handles are also insufficient unless both
> local and learned branches exceed gate thresholds: direct controls and nulls
> can be perfect while conflict branches stay below behavior-gate quality.
> After MC024, balanced worked examples are also insufficient as a full bridge:
> they repair prompt-local operation routing but the learned atomic operation
> branch still falls below gate on the full source-disjoint run.
> After MC025, constrained A/B/C candidate choices are also insufficient: they
> preserve local lookup and local operation routing but break direct atomic
> control, answer-absent nulls, and learned atomic conflict routing.
> After MC026, numeric option lists are also insufficient: they restore UNKNOWN
> nulls, but direct atomic control collapses into UNKNOWN and the learned atomic
> conflict branch still fails.
> After MC027, the answer interface itself is a measured behavior surface: bare
> integer is the only 10-source smoke survivor, while `ANSWER=...`, JSON, A/B/C
> choices, and numeric options break different gates.
> After MC028, that bare-integer survivor is not a full-source bridge repair:
> controls, nulls, and prompt-local routing stay clean across all 40 sources,
> but the learned atomic branch falls below gate through other-number leakage.
> After MC029, factorizing that leak is also not enough: `rules_only` improves
> the learned atomic branch to 0.875 and suppresses worked-example copying, but
> answer-absent UNKNOWN falls to 0.806; `query_before_examples` keeps nulls
> clean but amplifies worked-example copying to 54 rows. Factor movement is not
> behavior-substrate repair.
> After MC030, simple absence guards are also not enough: the unguarded
> `rules_only` baseline remains the best branch/null compromise, row-absence
> guard drops answer-absent UNKNOWN to 0.231, decision-order guard only reaches
> 0.425, and guarded query-last lowers other-number leakage to 0.056 only by
> collapsing operation-atomic atomic to 0.487.

The queue now has a closure feedback loop. The top bridge items still express
the law pressure to find a non-visible bridge, but they carry active
constraints saying the next route must be materially different from source
labels, row codes, query-operation handles, worked examples, constrained
choices, numeric options, answer-interface sweeps, and simple absence guards.
That makes the generated queue a claim-killing engine rather than a ranked
wish list.

The second pressure is transfer discipline:

> Future transfer claims must include matched null panels and side rows from the
> start, and transfer cannot be credited if only the primary effect moves.

The reliability pressure is stricter:

> A future experiment only improves the atlas if it moves a row to a better
> reliability class or makes a failure class sharper. A higher AUC, cleaner
> primary effect, or stronger prompt contrast is not progress unless the
> missing-gate list gets shorter under validation.

The third pressure is source-token discipline:

> Coarse source ablation is not enough. Source deletion, neutral rewrite, and
> query-only path comparisons must be part of any source-token control-surface
> claim.

## Current Instantiation

The top bridge pressure now has completed MC010, MC011, MC012, MC013, MC014,
MC015, MC016, and smoke-stage
MC017/MC018/MC019/MC020/MC021/MC022/MC023/MC024/MC025/MC026/MC027/MC028/MC029/MC030
diagnostics:

> `research/prereg/MC010_TWO_HOP_FACT_CODE_ARBITRATION.md`

> `code/mc010_two_hop_fact_code_arbitration.py`

> `research/cards/MC010_TWO_HOP_FACT_CODE_ARBITRATION_STRUCTURAL_STATUS.md`

> `research/cards/MC010_TWO_HOP_FACT_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

> `research/prereg/MC011_ATOMIC_NUMBER_CODE_ARBITRATION.md`

> `code/mc011_atomic_number_code_arbitration.py`

> `research/cards/MC011_ATOMIC_NUMBER_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

> `research/prereg/MC012_RELIABILITY_LABELED_NUMERIC_ARBITRATION.md`

> `code/mc012_reliability_labeled_numeric_arbitration.py`

> `research/cards/MC012_RELIABILITY_LABELED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `research/prereg/MC013_STATUS_CHANNEL_ABLATION_NUMERIC_ARBITRATION.md`

> `code/mc013_status_channel_ablation_numeric_arbitration.py`

> `research/cards/MC013_STATUS_CHANNEL_ABLATION_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `research/prereg/MC014_INFERRED_RELIABILITY_NUMERIC_ARBITRATION.md`

> `code/mc014_inferred_reliability_numeric_arbitration.py`

> `research/cards/MC014_INFERRED_RELIABILITY_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `research/prereg/MC015_PARITY_GATED_NUMERIC_ARBITRATION.md`

> `code/mc015_parity_gated_numeric_arbitration.py`

> `research/cards/MC015_PARITY_GATED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `research/prereg/MC016_ALPHABET_GATED_NUMERIC_ARBITRATION.md`

> `code/mc016_alphabet_gated_numeric_arbitration.py`

> `research/cards/MC016_ALPHABET_GATED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc017_selector_token_numeric_arbitration.py`

> `research/cards/MC017_SELECTOR_TOKEN_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc018_counterbalanced_selector_labels.py`

> `research/cards/MC018_COUNTERBALANCED_SELECTOR_LABELS_BEHAVIOR_STATUS.md`

> `code/mc019_row_code_numeric_arbitration.py`

> `research/cards/MC019_ROW_CODE_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc020_atomic_recall_table_pressure.py`

> `research/cards/MC020_ATOMIC_RECALL_TABLE_PRESSURE_BEHAVIOR_STATUS.md`

> `code/mc021_visible_vs_learned_arbitration.py`

> `research/cards/MC021_VISIBLE_VS_LEARNED_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc022_explicit_branch_name_arbitration.py`

> `research/cards/MC022_EXPLICIT_BRANCH_NAME_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc023_query_operation_numeric_arbitration.py`

> `research/cards/MC023_QUERY_OPERATION_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc024_fewshot_operation_numeric_arbitration.py`

> `research/cards/MC024_FEWSHOT_OPERATION_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc025_choice_interface_operation_arbitration.py`

> `research/cards/MC025_CHOICE_INTERFACE_OPERATION_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc026_numeric_option_operation_arbitration.py`

> `research/cards/MC026_NUMERIC_OPTION_OPERATION_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc027_answer_interface_sweep.py`

> `research/cards/MC027_ANSWER_INTERFACE_SWEEP_BEHAVIOR_STATUS.md`

> `code/mc028_bare_integer_full_source_boundary.py`

> `research/cards/MC028_BARE_INTEGER_FULL_SOURCE_BOUNDARY_STATUS.md`

> `code/mc029_operation_leak_factorial.py`

> `research/cards/MC029_OPERATION_LEAK_FACTORIAL_STATUS.md`

> `code/mc030_null_preserving_rules_repair.py`

> `research/cards/MC030_NULL_PRESERVING_RULES_REPAIR_STATUS.md`

MC010 uses a two-hop source path:

```text
element -> nonce handle -> task code
```

The structural scaffold passed on 800 records across 40 sources, 2 templates,
and 6 panels. It removes direct entity-to-code rows, hides true symbols from
conflict prompts, keeps one answer suffix across panels, and preserves
source-disjoint discovery/calibration/holdout splits.

The full generated behavior gate then failed. The selected `neutral_contract`
template had 23/40 synthetic two-hop task-code answers, 38/40 familiar
two-hop task-code answers, 12/40 real-symbol memory-control answers, 36/40
answer-absent `UNKNOWN` rows, and 229 primary task-code conflict rows versus
0 real/lure-symbol rows.

This closes MC010 as a diagnostic before hidden-state work. The next bridge
attempt must be materially different from ordinary symbolic-code, row-code, or
two-hop prompt repair.

MC011 then tested whether the issue was the answer interface rather than the
source path. It used the same integer answer type for prompt-local local lab
numbers and learned atomic numbers. The selected `neutral_numeric` template
passed all direct controls at 40/40 each, including real atomic-number recall
and answer-absent nulls, but primary conflict was 240 local-number rows and 0
atomic/lure-number rows.

This closes MC011 as a diagnostic before hidden-state work. The next bridge
attempt must be materially different from ordinary symbolic-code, row-code,
two-hop, or same-format numeric repair.

MC012 then tested whether the source/evaluation contract, not answer format,
could create the missing local-versus-learned mixture. It kept the same integer
answer interface and added explicit trusted/untrusted source-status labels. The
selected `compact_reliability` template passed the direct controls and nulls:
synthetic numeric lookup, familiar numeric lookup, real atomic-number recall,
trusted-source conflict, and answer-absent null rows were each 40/40, while
untrusted-source conflict selected the learned atomic number on 39/40 rows.
The primary conflict table had 40 local-number rows, 39 atomic/lure-number rows,
and one other-number row.

This is the first clean mixed bridge behavior table. It is still not a hidden
state substrate. The contrast is created by visible trusted/untrusted source
text, so `prompt_channel_locality_gate_passed` is false. MC012 therefore
exports a positive-control diagnostic: preserve this behavior shape, but do not
probe until the visible source-status channel is removed, matched, or ablated.

MC013 then tested the simplest version of that repair directly. It kept the
same numeric source bank and added text-identical matched ablation prompts. The
selected `compact_status_ablation` template reproduced the statused positive
control: 40/40 trusted rows selected the local number and 39/40 untrusted rows
selected the learned atomic number. Direct controls and nulls stayed clean.
But the matched ablation conflict collapsed to 80/80 local-number rows and 0
atomic/lure-number rows.

This closes simple status-channel ablation as a diagnostic route. The next
bridge must create the local-versus-learned mixture without first assigning the
answer rule through visible trusted/untrusted source-status text.

MC014 then tested an inferred-reliability repair. It removed explicit status
labels and required the model to check calibration rows against standard
chemistry before deciding whether the local table controlled. The selected
`calibration_rule` template kept direct controls and nulls clean at 40/40 and
primary prompts had no trusted/untrusted/reliable/unreliable/status lexemes.
But calibration-inconsistent conflict rows still selected local numbers on
40/40 rows, so the primary conflict collapsed to 80/80 local-number rows and
0 atomic/lure-number rows.

This closes calibration-inferred source validity as a diagnostic route under
the current prompt contract. The next bridge must weaken prompt-local table
dominance or change the source/evaluation contract materially, not just replace
the visible status label with calibration evidence.

MC015 then tested a learned-fact gate rather than a source-validity gate. It
hid the target and lure atomic numbers, removed visible source-status labels,
and made the local table control only when the queried element's standard
atomic-number parity matched the visible rule. The selected `parity_rule`
template kept direct controls and nulls clean at 40/40 each, and expected
local/atomic conflict labels were balanced by split. But the intended gate did
not control behavior: primary conflict expected correctness was 39/80, with
54 local-number rows, 24 atomic-number rows, and 2 other-number rows.
Expected-local rows selected local only 27/40, and expected-atomic rows
selected atomic only 12/40.

This closes parity-gated numeric arbitration as a diagnostic route. The next
bridge must not treat mixed local/atomic outputs as sufficient; it must prove
rule-aligned learned-fact gating before hidden-state work.

MC016 then tested whether a visible non-status operational gate could rescue
the bridge without using trusted/untrusted source-status text. It hid target
and lure atomic numbers, removed visible source-status labels, and made the
local table control according to the queried element's first-letter range. The
selected `alphabet_rule` template kept direct controls and nulls clean at 40/40
each, and expected local/atomic conflict labels were balanced by split. But the
primary conflict collapsed to 80/80 local-number rows and 0 atomic/lure rows.
Expected-local rows selected local 40/40; expected-atomic rows selected atomic
0/40. The stronger `feature_labeled_alphabet` template did not rescue the
atomic side, producing 61 local rows, 0 atomic rows, and 19 unparsed rows.

This closes visible non-status alphabet gating as a diagnostic route. The next
bridge must change the answer interface or source contract enough that
expected-atomic rows survive prompt-local table pressure.

MC017 then tested the simplest answer-interface repair by asking for source
tokens, `LOCAL` or `ATOMIC`, before any numeric answer. The reduced 10-source
smoke did not rescue the bridge: even the atomic-only control returned `LOCAL`
on 10/10 rows. This shows that source-token answers can themselves inject a
local/first-option prior.

MC018 replaced those words with neutral `A`/`B` source labels and
counterbalanced which source each label named, whether source definitions were
written A-first or B-first, and whether final answer options were written
A-first or B-first. The expanded 10-source structural gate passed at 960 rows.
The behavior smoke still failed: primary conflict rows were 160/160 parseable
and expected sources/choices were balanced, but source-rule correctness was
84/160, first-listed choices were 122/160, and local-source selections were
102/160. Atomic-only control improved to 66/80 atomic, so MC017's collapse was
partly answer-token-specific, but MC018 still blocks hidden-state work.

This means the next bridge must not merely use a nonnumeric answer interface.
It must demonstrate source-rule following after neutral label, definition
order, answer-option order, local-source salience, and null controls.

MC019 then removed the source-label answer interface and used neutral row-local
route codes while keeping the final answer numeric. The 10-source structural
gate passed at 420 rows. The selected smoke kept direct controls and nulls
clean: synthetic local lookup, familiar local lookup, real-world atomic
control, route-rule-absent UNKNOWN, and answer-absent UNKNOWN were all 20/20.
But route-code conflicts still failed, with 40/40 parseable primary rows and
only 21/40 expected-correct. The query-row repeated repair improved conflict
correctness to 27/40 and expected-local rows to 19/20, but expected-atomic rows
remained 8/20 and the answer-absent null slipped to 17/20.

This means the next bridge must not merely attach neutral row codes to the
local table. It must show that expected-atomic rows follow the operational rule
under prompt-local table pressure, not just that controls and nulls can be made
clean.

MC020 then isolated whether the MC019 expected-atomic failure was just local-row
interference. It was not. In the 10-source smoke, atomic recall was 20/20 with
no table, 18/20 with only distractor table rows, 18/20 with the queried local
row present, and 18/20 with the queried row repeated. The failure appeared only
under conditional route-rule arbitration: route-local reached 17/20 local, but
route-atomic collapsed to 15/20 local and only 2/20 atomic.

This means the next bridge should not optimize atomic recall availability. It
must use a materially different arbitration contract that does not collapse
when the rule points from a present local row to learned memory.

MC021 then tested whether the same route-code grammar works when both branches
are prompt-visible. It still did not pass. The selected 10-source smoke kept
direct controls and nulls clean: local 20/20, visible reference 20/20, atomic
18/20, and answer-absent UNKNOWN 18/20. But visible-visible route conflicts
were only 29/40 expected-correct, and visible-learned route conflicts were
20/40 expected-correct.

This means the next bridge should not rely on route codes as a clean
conditional substrate at all. It needs either a materially different
conditional format or a different behavior family.

MC022 then tested the narrowest route-code rescue by replacing opaque `P`/`Q`
codes with semantic answer-source labels: `LOCAL`, `REFERENCE`, and `ATOMIC`.
The selected 10-source smoke again kept direct controls and nulls clean: local
20/20, visible reference 20/20, atomic 18/20, and answer-absent UNKNOWN 19/20.
But semantic labels did not create a clean substrate. Visible conflicts were
30/40 expected-correct, learned conflicts were 24/40 expected-correct, and
ATOMIC-source rows selected learned atomic numbers only 5/20.

The useful split is rule-order sensitivity. When the nonlocal branch was
defined first, visible-visible routing reached 20/20 expected-correct; when
LOCAL was defined first, the same visible-visible contract collapsed to local
outputs. Defining ATOMIC first helped learned-memory routing only partially.
This means the next bridge must beat local-source salience and rule-order
sensitivity, not merely replace arbitrary route codes with more meaningful
source names.

MC023 then removed row-level source labels and tested query-level operation
handles. This cleaned the easy controls in the selected 10-source smoke:
synthetic local lookup, familiar local lookup, direct atomic control,
operation-rule-absent UNKNOWN, and answer-absent UNKNOWN were all 40/40, with
candidate/output margins reported and no status lexemes. But the conflict
branches stayed below gate quality: operation-local rows selected local only
31/40, and operation-atomic rows selected atomic only 28/40.

This means the next bridge should not treat query-level operation handles as a
solution unless both branches exceed behavior-gate thresholds on full
source-disjoint runs. MC023 is useful because it separates control cleanliness
from conflict adequacy: the prompt can pass direct controls and nulls while
still failing the actual local-versus-learned arbitration.

MC024 then tested the direct rescue by adding balanced worked examples to the
query operation rules. The smoke pass was strong enough to force a full run:
the compact worked-example template cleared all direct controls, nulls, and
both conflict thresholds on 10 sources. The 40-source run blocked promotion in
a more informative way. Controls and nulls stayed perfect at 160/160, and the
operation-local branch reached 159/160 local, but the operation-atomic branch
reached only 130/160 atomic and leaked 27/160 other-number answers.

This means the next bridge should not treat worked examples as a route to
hidden-state work unless the learned branch clears full-source gates. MC024 is
useful because it splits the failure: prompt-local operation routing is
repairable, but learned-memory operation routing remains the brittle edge.

MC025 then tested whether MC024's other-number leakage was mainly a free-form
integer answer-interface problem. It constrained the final answer to A/B/C
choices containing the local number, the atomic number, and UNKNOWN. The
structural gate passed, including balanced expected choices and prompt-visible
atomic-number options by design. The smoke behavior still failed before any
full run was justified: synthetic and familiar local lookup were 120/120,
operation-local choice conflict was 109/120 local, but direct atomic control
was only 44/120 atomic, answer-absent nulls were only 83/120 UNKNOWN, and
operation-atomic choice conflict was only 18/120 atomic.

This means the next bridge should not treat candidate choices as a neutral
answer-interface fix. Prompt-visible choices are themselves a control surface:
they can preserve prompt-local routing while damaging direct atomic recall and
null behavior.

MC026 then checked whether the A/B/C labels were the damaging part of MC025's
candidate-choice interface. It replaced letter choices with numeric options:
the local number, the atomic number, and UNKNOWN were all prompt-visible, and
the final answer was constrained to one of those option values. The structural
gate passed with balanced first options, no status lexemes, and explicit
tracking that atomic numbers are visible in conflict options by design. The
smoke behavior still failed: synthetic and familiar option lookup were 120/120
local, answer-absent and operation-rule-absent nulls were 120/120 UNKNOWN, and
operation-local conflict was 111/120 local, but direct atomic control selected
atomic 0/120 and UNKNOWN 116/120, and operation-atomic conflict selected atomic
only 19/120.

This means the next bridge should not treat numeric option lists as a neutral
answer-interface fix either. They repair one failure in MC025, the null rows,
but do it by creating a different behavior surface: learned atomic recall
becomes abstention. The control-surface lesson is that output scaffolds can
change the behavior substrate even when the allowed answer set is objectively
complete.

MC027 then treated answer format itself as the experimental object. It swept
bare integer, `ANSWER=...`, JSON, A/B/C choice, and numeric-option interfaces
over the same 10-source operation substrate. The structural gate passed over
2,160 rows with balanced operation assignments, rule orders, option orders, and
split coverage. The result was not "any formatting works"; it was the opposite.
Bare integer was the only interface that cleared smoke: direct atomic control
was 40/40 atomic, answer-absent and rule-absent nulls were 40/40 UNKNOWN,
operation-local conflict was 39/40 local, and operation-atomic conflict was
38/40 atomic. The other interfaces failed in different ways: `ANSWER=...`
damaged parseability and learned routing, JSON collapsed answer-absent nulls,
A/B/C choices kept weak learned routing, and numeric options reproduced atomic
abstention.

This means the next bridge should not add structured output as a cleanup step
unless the structure itself is the thing being tested. A schema can change the
behavior substrate. The bare-integer smoke result is useful, but it does not
erase MC024's full-source failure: the current safe claim is interface
dispersion, not bridge promotion.

MC028 then ran the necessary full-source boundary test for that bare-integer
survivor. It kept the same bare-integer answer interface and scaled back to all
40 sources. The structural gate passed at 960 rows with clean source-disjoint
coverage, no status lexemes, balanced assignments, and clean null rows. Behavior
kept the easy surfaces clean: familiar lookup was 160/160 local, direct atomic
control was 160/160 atomic, answer-absent and operation-rule-absent nulls were
160/160 UNKNOWN, and operation-local conflict was 159/160 local. The learned
branch still missed gate: operation-atomic conflict was 131/160 atomic, 28/160
other-number, and 1/160 local.

This closes the MC027 survivor as a bridge repair. The insight is not that
bare-integer formatting is useless; it is that removing output-schema
interference exposes the next limiting surface. Under full source-disjoint
pressure, the hard behavior is still learned atomic selection in the presence
of a prompt-local operation table, and the failure is now typed as
`FULL_SOURCE_OTHER_NUMBER_LEAK`.

MC029 then tested whether that leak could be split into repairable prompt
factors before any hidden-state work. It ran a 4,000-row full-source factorial
over baseline numeric examples, label examples without numbers, rules-only
prompting, query-before-example ordering, and query-row-last salience. No
variant passed the behavior gate. The important result is the pattern of
movement: `rules_only` improved operation-atomic atomic to 0.875 and reduced
other-number answers to 0.100, but answer-absent UNKNOWN fell to 0.806.
`query_before_examples` preserved answer-absent UNKNOWN at 1.000 but collapsed
operation-atomic atomic to 0.256 and produced 54 worked-example other-number
rows. `query_row_last` reduced other-number answers to 0.081 but left
operation-atomic atomic at 0.613 with a larger local-answer component.

This closes simple prompt-factor repair for the current operation-leak route
unless a materially different substrate changes the branch/null structure. A
prompt variant that only improves one axis while worsening another should be
recorded as a diagnostic, not treated as progress toward hidden-state probing.

MC030 then tested that exact local repair: explicit absence guards around the
rules-only prompt. The structural gate passed with 4,000 rows across all 40
sources after numeric list markers were removed because they leak early atomic
targets. Behavior did not repair. The unguarded `rules_only_baseline` stayed
best at 0.875 operation-atomic atomic and 0.806 answer-absent UNKNOWN.
`row_absence_guard_before_rules` kept operation-atomic atomic at 0.863 but
answer-absent UNKNOWN fell to 0.231. `decision_order_guard_after_query` kept
operation-atomic atomic at 0.869 but answer-absent UNKNOWN was 0.425.
`decision_order_guard_query_last` reduced other-number answers to 0.056, but
operation-atomic atomic collapsed to 0.487.

This closes simple absence-guard repair for the current operation-leak route.
The next bridge attempt should be materially different, or the current route
should be treated as a diagnostic family rather than kept alive through local
prompt guards.

## What This Proves

It proves that the project now has a generated bridge from law hypotheses to
next tests.

The queue is validated by `code/validate_control_surface_atlas.py`. Validation
fails if the queue is stale, if it has fewer items than hypotheses, if it has no
top queue ids, if no item reaches immediate/high priority, if bridge-closure
context is missing, if the bridge rung count is not 21, if any bridge rung is
hidden-state-allowed, if the recent closure set is not MC028-MC030, or if
bridge-route queue items do not carry active closure constraints.

## What It Does Not Prove

It does not prove that any top queue item will succeed.

It does not prove the next bridge route will succeed. MC010 is evidence that
structurally clean two-hop indirection is insufficient on its own. MC011 is
evidence that same-format numeric answers can repair direct controls while the
conflict table remains completely prompt-local. MC012 is evidence that explicit
source reliability can create the desired behavior mixture, but also that this
mixture is not mechanism-ready when the source-status channel is visible.
MC013 is evidence that simply removing that visible channel collapses the
learned-fact side under the current prompt contract. MC014 is evidence that
calibration-inferred source validity also collapses to prompt-local answers
under the current prompt contract. MC015 is evidence that balanced expected
labels and mixed local/atomic outputs are still insufficient when the learned
factual gate is not followed. MC016 is evidence that even a visible non-status
feature gate is insufficient when expected-atomic rows still collapse to local
table answers. MC017/MC018 are evidence that source-selector answer interfaces
are not automatically cleaner: token choice, source-definition order,
answer-option order, and local-source salience can dominate the intended source
rule before any hidden-state signature is worth testing. MC019 is evidence that
neutral row-local route codes can clean up direct controls and nulls while
still failing the expected-atomic side of the route rule. MC020 is evidence
that direct atomic recall survives local-table pressure; the specific hard
thing is asymmetric conditional arbitration toward learned memory. MC021 is
evidence that route-code arbitration is already unstable even when both
branches are prompt-visible, with learned-memory branches worse. MC022 is
evidence that semantic answer-source names do not by themselves repair the
substrate: visible-visible routing becomes rule-order-sensitive, and
learned-memory routing still collapses toward prompt-local answers.
MC023 is evidence that query-level operation handles can preserve direct
controls and nulls while both local and learned conflict branches remain below
gate quality.
MC024 is evidence that balanced worked examples can repair the local branch
while the learned atomic branch remains below gate quality on full
source-disjoint evaluation.
MC025 is evidence that constrained candidate choices are not a neutral repair:
they break atomic controls and answer-absent nulls before any hidden-state
claim is possible.
MC026 is evidence that numeric options are not a neutral repair either: they
restore null behavior, but direct atomic recall collapses into UNKNOWN and the
learned atomic branch remains weak.
MC027 is evidence that answer schemas are behavior surfaces rather than neutral
wrappers: only bare integer cleared the 10-source interface sweep, and the
structured interfaces failed controls, nulls, parseability, or learned routing.
MC028 is evidence that the bare-integer survivor still does not promote to a
full-source bridge: controls and nulls stay clean, but learned atomic conflict
rows leak other learned numbers below the behavior gate.
MC029 is evidence that factorized prompt edits can move the failure axes
without repairing the substrate: `rules_only` improves learned atomic routing
but breaks answer-absent null reliability, while `query_before_examples`
preserves nulls but amplifies worked-example copying.
MC030 is evidence that simple absence guards do not repair that tradeoff: they
worsen answer-absent nulls or reduce other-number leakage only by collapsing
the learned atomic branch.
The error taxonomy sharpens that into a next-test constraint: MC028's 28 wrong
operation-atomic rows are split across worked-example outputs, other bank
atomic facts, off-bank other numbers, bank local numbers, and prompt-local row
numbers, and MC029 shows that manipulating those factors separately still does
not yield a clean bridge, while MC030 closes simple absence-guard repair. The
next route should be materially different from the current operation-leak
prompt family.

It also does not prove that closure-aware queueing is itself a mechanism
advance. It is an experiment-selection guardrail: it prevents the project from
spending the next iteration on a route that the current bridge ladder, route
disposition, and error taxonomy have already closed.

It does not prove that scoring weights are final. The weights are an explicit
current policy so they can be argued with and changed, not hidden judgment.

## Next Use

Every future run should update the queue after updating:

- the atlas;
- the artifact index;
- the comparison;
- the law audit;
- the error taxonomy.

The question after each future result is:

> Did this result change the queue, or only add another example of a law we
> already understood?

That is how the project avoids wandering. The atlas says what happened; the
law audit says what the current theory is allowed to claim; the queue says what
the next experiment should try to kill.
