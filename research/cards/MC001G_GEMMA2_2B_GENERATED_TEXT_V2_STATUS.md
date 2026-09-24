# MC001G Gemma 2 2B Generated Text V2 Status

Status: complete. The generated-text V2 repair bank improved clean substrate
size, but still fails the preregistered reliability bar for hidden-state work.

Date: 2026-06-30

## Artifacts

- generation runner: `code/mc001_qwen3_smoke.py`
- audit: `code/mc001_gemma_generated_text_audit.py`
- variant: `gemma_generated_text_v2`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_GENERATED_TEXT_V2.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_generated_text_v2_raw_manifest.jsonl`
- manifest SHA256: `c3666424529e5f55cd72e497392de64644f20eb854f90926a025c0d37984acee`
- result: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_v2_smoke_gemma_generated_text_v2_20260630T144631.json`
- result SHA256: `3bee9c9bda4009d7e6083cdeca49f5e1bbef630ce444e24cccf45b131c15426c`
- audit result: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_v2_generation_audit_20260630T144711.json`
- audit SHA256: `9ac4866428caac295d64cf02b6cde344b9e8bd18125f26e39328d689e51f9739`

Configuration:

- model: `google/gemma-2-2b`
- render mode: `raw`
- max new tokens: `12`
- retained source groups: 333
- records: 2,331
- source bank: 237 inherited generated-text items plus 96 explicit free-text
  repair items
- clean item rule: `no_hint` and `correct_hint` must both parse as the correct
  answer text
- matching conditions: `wrong_disclaimed`, `wrong_unsure`
- strict match: condition plus correct/wrong answer shape plus correct/wrong
  answer length bins

## Question

Did the V2 generated-text repair bank fix the V1 answer-shape and coverage
failures enough to restart hidden-state signature discovery?

## Gate Verdict

No.

V2 repaired the clean-size floor and removed the V1 strict shape-dominance
failure. It still did not create a reliability-clean behavior substrate. The
primary weak-hint truth rows are too sparse, shape-matched rows miss the
preregistered volume floor, and strict holdout keys are not fully covered by
matching discovery keys.

Write this as:

> MC001G generated text V2 repaired clean coverage, but failed weak-hint truth,
> shape-matched volume, and exact strict holdout-key coverage.

## Structural Checks

The manifest passed the structural checks:

| Metric | Result |
| --- | ---: |
| retained source groups | 333 |
| records | 2,331 |
| records per condition | 333 |
| inherited V1 generated-text items | 237 |
| explicit repair items | 96 |
| displayed `Choices:` lines | 0 |
| displayed option lists | 0 |
| exact `A`/`B`/`C`/`D` candidate answers | 0 |
| blocked non-standalone stems | 0 |
| missing result records | 0 |
| extra result records | 0 |

Answer-shape inventory:

| Correct Answer Shape | Items |
| --- | ---: |
| number | 112 |
| single word | 197 |
| multi word | 24 |

## Behavior Summary

| Condition | Parseable | Truth | Agreement |
| --- | ---: | ---: | ---: |
| `no_hint` | 267/333 | 263 | 4 |
| `correct_hint` | 310/333 | 310 | 0 |
| `wrong_disclaimed` | 264/333 | 48 | 216 |
| `wrong_unsure` | 262/333 | 82 | 180 |
| `wrong_direct` | 278/333 | 21 | 257 |
| `wrong_high` | 311/333 | 1 | 310 |
| `anti_wrong` | 209/333 | 186 | 23 |

The behavior gradient remains real: the model usually answers correctly with
no hint or a correct hint, and it often follows direct or high-confidence wrong
hints. The reliability failure is narrower. The weak-hint truth class is still
too small after conservative grading, and the exact discovery/holdout support
needed for a causal mechanism test is incomplete.

## Audit Against Preregistration

| Metric | Result | Preregistered Bar | Verdict |
| --- | ---: | ---: | --- |
| clean items | 252 | at least 240 | pass |
| clean number-answer items | 103 | at least 70 | pass |
| clean single-word-answer items | 128 | at least 120 | pass |
| clean multi-word-answer items | 21 | at least 14 | pass |
| primary weak-hint truth rows | 123 | at least 150 | fail |
| primary weak-hint agreement rows | 295 | at least 180 | pass |
| shape-matched rows | 182 | at least 200 | fail |
| strict matched rows | 172 | at least 160 | pass |
| strict matched holdout rows | 58 | at least 50 | pass |
| min condition-label strict cell | 43 | at least 35 | pass |
| max single answer-shape share of either strict label | 58/86 | at most 70% | pass |
| every strict holdout key covered in discovery | no | yes | fail |

Clean item counts by answer shape:

| Correct Answer Shape | Clean Items |
| --- | ---: |
| number | 103 |
| single word | 128 |
| multi word | 21 |

Strict matched answer-shape label counts:

| Correct Answer Shape | Truth | Agreement |
| --- | ---: | ---: |
| number | 58 | 58 |
| single word | 23 | 23 |
| multi word | 5 | 5 |

Strict holdout keys missing discovery support:

- `wrong_disclaimed|multi_word|multi_word|char_7_12|char_7_12`
- `wrong_disclaimed|number|number|char_3_6|char_3_6`
- `wrong_unsure|multi_word|multi_word|char_7_12|char_7_12`
- `wrong_unsure|number|number|char_3_6|char_3_6`

Shape-level holdout coverage also failed for:

- `wrong_unsure|multi_word|multi_word`

## Diagnosis

V2 answered the easy part of the V1 failure. Adding single-word and multi-word
standalone facts increased clean items from 180 to 252 and raised strict
matched rows from 132 to 172. The strict matched subset is no longer over the
70 percent per-label answer-shape dominance guard.

It did not answer the hard part. Many added facts are clean under no-hint and
correct-hint prompts, but weak wrong hints still push too many rows into
agreement or unparseable outputs. The resulting truth class has only 123
primary weak-hint rows, below the 150-row floor, and the shape/length bins that
survive in holdout are not all represented in discovery.

That means a hidden-state signature discovered here could still be a
source-bank coverage artifact rather than a reliable truth-versus-user-agreement
control surface.

## Decision

Do not resume hidden-state discovery, sparse-feature discovery, path
localization, activation patching, or steering on the MC001G generated-text V2
substrate.

Close this branch as a behavior-interface negative result:

> generated text removed displayed choices, V2 repaired clean coverage, but the
> reliability controls still failed before any new mechanism test.

A future V3 would need to target weak-hint truth rows and exact strict
discovery/holdout coverage directly. Another broad intervention pass is not
justified by this substrate.
