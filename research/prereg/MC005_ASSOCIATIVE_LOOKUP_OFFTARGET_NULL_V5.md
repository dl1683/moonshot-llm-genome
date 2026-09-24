# MC005 Associative Lookup Off-Target Null V5 Preregistration

Date: 2026-06-30

## Question

MC005 V4 showed robust lookup-scope behavior for the fixed `late_20_26`
source-value mask, but the off-target null failed. V5 asks:

> Is the V4 off-target failure a broad source-mask side effect, or is it tied
> to same-grammar lookup prompts where another pair is still being queried?

## Fixed Intervention

V5 does not reselect a layer band.

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed intervention:

- layers 20-26;
- all attention heads;
- eager-attention source mask at the final answer position;
- source arms: primary irrelevant source value, secondary irrelevant source
  value, and random irrelevant source value.

## Null Suite

Planned scenarios:

| Scenario | Prompt surface | Role | Expected label |
| --- | --- | --- | --- |
| `same_grammar_query_other_pair` | key/value lookup list, query another pair | diagnostic repeat of V4 off-target | clean null or side effect |
| `reference_explicit_answer` | key/value reference list plus explicit answer instruction | primary repaired null | clean null |
| `sentence_reference_explicit_answer` | sentence-style reference notes plus explicit answer instruction | primary repaired null | clean null |
| `no_reference_note_explicit_answer` | ordinary note with irrelevant source words, no lookup query | primary repaired null | clean null |
| `nonlookup_marker_answer` | unrelated word list plus answer marker | primary repaired null | clean null |

The primary repaired-null scenarios score an answer word that is separately
specified outside the source values being masked. Masking any irrelevant source
value should not materially change the target-vs-distractor answer margin.

The same-grammar diagnostic is included because V4 failed there. If it remains
side-effectful while the primary repaired nulls pass, the correct conclusion is
not "fully reliable"; it is "out-of-grammar nulls are clean, but same-grammar
lookup side effects remain a boundary."

## Metrics

For each scenario:

- baseline target-minus-distractor margin;
- baseline clean rows;
- greedy next-token target/distractor/other counts;
- primary-source mask effect;
- secondary-source mask effect;
- random-source mask effect;
- absolute target-win change under each arm.

## Scenario Labels

Scenario label:

- `clean_null`: baseline clean rows at least 24/32, every source arm has
  absolute mean delta at most 0.50, and every source arm changes target-win
  count by at most one row.
- `weak_null`: baseline clean rows at least 24/32, every source arm has
  absolute mean delta at most 1.00, and every source arm changes target-win
  count by at most two rows.
- `side_effect`: baseline is clean but at least one source arm exceeds the
  `weak_null` limits.
- `invalid_baseline`: fewer than 24/32 baseline rows prefer the target answer.

## Pass Boundary

V5 passes the strict off-target-null suite only if every scenario, including the
same-grammar diagnostic, is `clean_null`.

V5 can also produce a weaker useful result:

- primary repaired-null pass: every scenario except
  `same_grammar_query_other_pair` is `clean_null`;
- same-grammar boundary remains: `same_grammar_query_other_pair` is
  `side_effect` or `weak_null`.

That weaker result would narrow the side-effect boundary but would still block
complete mechanism-card promotion.
