# Control-Surface Error Taxonomy

Date: 2026-07-01

Status: generated typed-failure taxonomy implemented and validated.

This document interprets:

> `data/control_surface_smoke_diagnostics.json`

> `data/control_surface_bridge_ladder.json`

> `data/control_surface_route_disposition.json`

Builder:

> `code/control_surface_error_taxonomy.py`

Command:

```powershell
python code\control_surface_error_taxonomy.py --write
python code\control_surface_error_taxonomy.py
python code\validate_control_surface_atlas.py
```

## Purpose

The point is to stop treating killed bridge attempts as prose-only
tombstones. Each failure is a measured datum about where behavior lives:
answer interface, prompt-local source salience, route-rule following,
learned-memory selection, null behavior, or output geometry.

## Current Facts

- smoke cards covered: `17`;
- smoke validation checks: `72`;
- bridge rungs covered: `24`;
- bridge rungs closed before hidden-state work: `23`;
- prompt-visible positive-control bridge rungs: `1`;
- hidden-state-allowed smoke cards: `0`;
- behavior-ready smoke cards: `0`.

## Top Failure Axes

| Axis | Count |
| --- | ---: |
| `direct_controls_and_nulls_clean` | 3 |
| `answer_absent_null_failed` | 2 |
| `local_source_salience` | 2 |
| `candidate_output_baselines_reported` | 2 |
| `answer_token_collapse` | 1 |
| `atomic_selector_control_failed` | 1 |
| `local_selector_prior` | 1 |
| `neutral_label_rule_following_failed` | 1 |
| `answer_option_order_bias` | 1 |
| `definition_order_bias` | 1 |
| `row_code_rule_following_failed` | 1 |
| `expected_atomic_route_weak` | 1 |

## MC028 Other-Number Leak

MC028 is the current sharpest example of why a failed behavior substrate
can still be informative. Bare integers preserve direct controls and
nulls at full-source scale, but the learned atomic operation branch
does not merely copy the prompt-local row. It leaks wrong numbers from
several different surfaces.

| Measure | Value |
| --- | ---: |
| operation-atomic rows | 160 |
| operation-atomic atomic rate | 0.819 |
| operation-atomic other-number count | 28 |
| operation-atomic other-number rate | 0.175 |
| operation-local local rate | 0.994 |
| direct atomic-control atomic rate | 1.000 |
| answer-absent UNKNOWN rate | 1.000 |
| rule-absent UNKNOWN rate | 1.000 |

Exclusive wrong-number buckets:

| Bucket | Rows |
| --- | ---: |
| `bank_atomic_number` | 8 |
| `bank_local_number` | 3 |
| `off_bank_other_number` | 6 |
| `prompt_local_row_number` | 2 |
| `worked_example_output` | 9 |

Overlap labels:

| Label | Rows |
| --- | ---: |
| `bank_atomic_number` | 17 |
| `bank_local_number` | 5 |
| `double_expected_atomic` | 2 |
| `expected_atomic_plus_10` | 2 |
| `off_bank_other_number` | 6 |
| `prompt_local_row_number` | 2 |
| `worked_example_output` | 9 |

## MC029 Factorized Leak Tradeoff

MC029 tested whether the MC028 other-number leak could be repaired by
separating numeric examples, label-only examples, rule-only prompts,
query-before-example ordering, and query-row-last salience. The result
is a sharper failure: individual factors move different error axes,
but no variant becomes a valid bridge substrate.

| Variant | Op-atomic atomic | Op-atomic other | Op-local local | Answer-absent UNKNOWN | Worked-example other rows | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_numeric_examples` | 0.713 | 0.244 | 1.000 | 0.988 | 7 | fail |
| `label_examples_no_numbers` | 0.800 | 0.175 | 0.950 | 0.725 | 0 | fail |
| `rules_only` | 0.875 | 0.100 | 0.994 | 0.806 | 0 | fail |
| `query_before_examples` | 0.256 | 0.450 | 0.981 | 1.000 | 54 | fail |
| `query_row_last` | 0.613 | 0.081 | 0.925 | 0.988 | 6 | fail |

MC029 shows factor movement, not substrate repair. Removing numeric examples improves the learned atomic branch enough to cross the atomic-rate threshold, but answer-absent null reliability collapses. Moving the query before examples preserves nulls while amplifying worked-example copying. Putting the query row last reduces other-number leakage but shifts errors into local-row selection.

## MC030 Absence-Guard Repair Failure

MC030 tested the narrowest repair implied by MC029: keep the rules-only
branch gain, but add explicit absence guards to restore answer-absent
null reliability. The repair failed. Guards either made answer-absent
rows worse or reduced other-number leakage by collapsing the learned
atomic branch.

| Variant | Op-atomic atomic | Op-atomic other | Op-local local | Rule-null UNKNOWN | Answer-absent UNKNOWN | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rules_only_baseline` | 0.875 | 0.100 | 0.994 | 1.000 | 0.806 | fail |
| `row_absence_guard_before_rules` | 0.863 | 0.081 | 0.900 | 0.275 | 0.231 | fail |
| `no_memory_for_absent_guard` | 0.787 | 0.087 | 1.000 | 0.013 | 0.431 | fail |
| `decision_order_guard_after_query` | 0.869 | 0.100 | 0.981 | 0.975 | 0.425 | fail |
| `decision_order_guard_query_last` | 0.487 | 0.056 | 0.906 | 1.000 | 0.425 | fail |

MC030 shows that simple absence guards do not repair the MC029 rules-only tradeoff. The baseline remains the best branch/null compromise. Row-absence and decision-order guards worsen answer-absent nulls, and the query-last guard reduces other-number leakage only by collapsing learned atomic routing.

## MC031 Statusless Checksum Collapse

MC031 tested whether a statusless arithmetic-checksum cue could replace
visible reliability labels and route invalid-source rows to learned
atomic facts. The controls and nulls stayed clean in smoke, but the
invalid-checksum branch still selected local numbers.

| Measure | Value |
| --- | ---: |
| selected template | `arithmetic_checksum` |
| synthetic lookup local rate | 1.000 |
| familiar lookup local rate | 1.000 |
| real atomic-control atomic rate | 1.000 |
| answer-absent UNKNOWN rate | 1.000 |
| valid-checksum local rate | 1.000 |
| invalid-checksum local rate | 1.000 |
| invalid-checksum atomic/lure rate | 0.000 |
| checksum-absent local rate | 1.000 |

MC031 isolates a statusless reliability-cue failure. Direct local lookup, familiar local lookup, direct atomic recall, valid-checksum local routing, and answer-absent nulls all survive smoke. The invalid checksum branch still selects local numbers on every selected row and never selects the atomic or lure value, so the bridge fails at source validity arbitration rather than recall or null behavior.

## MC032 Cross-Table Consistency Collapse

MC032 tested whether MC031's result was checksum-specific. It removed
arithmetic validity and used agreement between two neutral local tables
instead. The controls and nulls stayed clean, side-number leakage stayed
absent, and the mismatch branch still avoided learned atomic answers.

| Measure | Value |
| --- | ---: |
| selected template | `mirror_registry` |
| synthetic lookup local rate | 1.000 |
| familiar lookup local rate | 1.000 |
| real atomic-control atomic rate | 1.000 |
| answer-absent UNKNOWN rate | 1.000 |
| match-conflict local rate | 0.800 |
| mismatch-conflict local rate | 0.700 |
| mismatch-conflict atomic/lure rate | 0.000 |
| mismatch-conflict side-number rate | 0.000 |
| crosscheck-absent local rate | 1.000 |

MC032 removes the checksum-specific explanation for MC031. The cue is cross-table consistency rather than arithmetic validity, and direct local lookup, familiar local lookup, direct atomic recall, and answer-absent nulls all survive smoke. The mismatch branch still never selects the atomic or lure value; it mostly selects the primary local number, with no side-number copying. The failure is therefore broader local-table dominance under statusless source-validity pressure.

## MC033 Fact-Claim Closeout

MC033 tested the one repair pass allowed after MC032: replace checksum
and cross-table cues with a row-local standard-number claim checked
against learned atomic memory. Controls and nulls stayed clean, but
the branch rule did not become stable.

| Measure | Value |
| --- | ---: |
| selected template | `memory_comparison` |
| synthetic lookup local rate | 1.000 |
| familiar lookup local rate | 1.000 |
| real atomic-control atomic rate | 1.000 |
| answer-absent UNKNOWN rate | 1.000 |
| match-conflict local rate | 0.400 |
| match-conflict atomic rate | 0.500 |
| mismatch-conflict atomic rate | 0.100 |
| mismatch-conflict local rate | 0.500 |
| mismatch-conflict claimed-number/lure rate | 0.400 |
| fact-claim-absent local rate | 1.000 |

MC033 closes the post-MC032 repair path. It replaces checksum and cross-table cues with a row-local standard-number claim checked against learned atomic memory. Direct local lookup, familiar local lookup, direct atomic recall, and answer-absent nulls all survive smoke. The rule does not: match rows often return the atomic number instead of the local number, while mismatch rows mostly split between the local number and the wrong claimed number. The bridge failure is therefore not repaired by making the validity cue a learned-fact comparison.


## Interpretation

The MC028 failure is not one generic local-copy error. Wrong answers split across worked-example outputs, other bank atomic facts, bank local numbers, prompt-local numbers, and off-bank periodic-like numbers while direct atomic recall and null controls remain clean.

The immediate consequence is that the next bridge experiment should
not just try another answer schema. MC029 already manipulated worked
examples, query anchoring, and prompt-row salience separately; that
factorization moved the failure axes without repairing the substrate.
MC030 then tested the local absence-guard repair and found that this
also fails: the unguarded rules-only baseline remains the best
branch/null compromise, while the guards worsen null rows or collapse
the learned branch.
MC031 adds a separate statusless reliability-cue failure: the model can
obey direct controls and nulls while still refusing to use the checksum
cue to leave the local branch.
MC032 removes the checksum-specific objection and preserves the same
boundary: cross-table mismatch still fails to reach learned atomic facts,
and the error is not second-table side-number copying.
MC033 then removes the cross-table cue and uses a row-local learned-fact
claim; this still does not repair the bridge, because the match branch
does not reliably stay local and the mismatch branch leaks local and
claimed wrong numbers.
The current new failure families are wrong learned-number selection under
route pressure, branch/null tradeoff, statusless invalid-source local
collapse, statusless cross-table local collapse, and fact-claim branch
instability, distinct from generic parse failure and UNKNOWN abstention.

## Claim Boundary

Allowed: The post-atlas bridge program has a typed failure map. MC028's full-source bare-integer failure is a learned-branch other-number leak, not a direct atomic recall failure or null failure. MC029 shows that factorized prompt changes move the branch/null/error axes without producing a valid bridge substrate. MC030 shows that simple absence guards do not repair that tradeoff. MC031 shows that a statusless checksum reliability cue can keep controls and nulls clean while invalid-source rows still collapse to local answers. MC032 shows that replacing checksum validity with cross-table consistency does not repair the learned branch; the mismatch branch still collapses toward local numbers without side-number leakage. MC033 closes the post-MC032 repair path: a learned-fact claim cue keeps controls and nulls clean but fails both match and mismatch routing.

Forbidden: This taxonomy does not establish a hidden signature, intervention, or mechanism card, and it does not make MC028, MC029, MC030, MC031, MC032, or MC033 behavior-ready.
