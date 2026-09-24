# MC001G Gemma 2 2B Generated Text Status

Status: complete. Raw generated-answer text removed displayed choices entirely,
but the substrate still fails the preregistered reliability bar for hidden-state
work.

Date: 2026-06-30

## Artifacts

- generation runner: `code/mc001_qwen3_smoke.py`
- audit: `code/mc001_gemma_generated_text_audit.py`
- variant: `gemma_generated_text`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_GENERATED_TEXT.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_generated_text_raw_manifest.jsonl`
- manifest SHA256: `a5edf34ec51efcb8120161762b33beefc3d4d479b36b474c932b661b981083fa`
- result: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_smoke_gemma_generated_text_20260630T142558.json`
- result SHA256: `a3a572a82a3ab5e24a6bf87a687ba6ac45e33b5a987b14a93f6a6c23762d3aba`
- audit result: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_generation_audit_20260630T142611.json`
- audit SHA256: `21169e667ed9a72ea93b897957a64c9e066ff35369493dbef8c9f46bed83c23a`

Configuration:

- model: `google/gemma-2-2b`
- render mode: `raw`
- max new tokens: `12`
- retained source groups: 237
- records: 1,659
- source filter: removed exact `A`/`B`/`C`/`D` answer texts and
  option-dependent stems
- clean item rule: `no_hint` and `correct_hint` must both parse as the correct
  answer text
- matching conditions: `wrong_disclaimed`, `wrong_unsure`
- strict match: condition plus correct/wrong answer shape plus correct/wrong
  answer length bins

## Question

Can raw generated-answer text repair the MC001G behavior interface enough to
restart hidden-state discovery?

## Gate Verdict

No.

Generated-answer text is a better interface than the original free-generation
smoke and avoids both letter-token and displayed pair-order artifacts. It still
does not pass the preregistered reliability controls. The matched rows are too
number-dominated, weak wrong-hint truth is below the floor, and strict holdout
keys are not fully supported by discovery keys.

Write this as:

> MC001G generated text removed displayed choices and produced a real behavior
> gradient, but failed answer-shape balance and exact strict holdout support.

## Structural Checks

The manifest passed the structural checks:

| Metric | Result |
| --- | ---: |
| retained source groups | 237 |
| records | 1,659 |
| records per condition | 237 |
| displayed `Choices:` lines | 0 |
| displayed option lists | 0 |
| exact `A`/`B`/`C`/`D` candidate answers | 0 |
| blocked non-standalone stems | 0 |

Answer-shape inventory:

| Correct Answer Shape | Items |
| --- | ---: |
| number | 112 |
| single word | 112 |
| multi word | 13 |

## Behavior Summary

| Condition | Parseable | Truth | Agreement |
| --- | ---: | ---: | ---: |
| `no_hint` | 191/237 | 189 | 2 |
| `correct_hint` | 218/237 | 218 | 0 |
| `wrong_disclaimed` | 192/237 | 36 | 156 |
| `wrong_unsure` | 195/237 | 74 | 121 |
| `wrong_direct` | 206/237 | 16 | 190 |
| `wrong_high` | 225/237 | 0 | 225 |
| `anti_wrong` | 153/237 | 134 | 19 |

Compared with the original Gemma generation smoke, raw generated text is not an
answer-only interface collapse. The model knows many no-hint answers and shows
a strong pressure gradient. The reliability problem is narrower: after
conservative grading, the clean/matched behavior set is still not balanced or
covered enough for a mechanism test.

## Audit Against Preregistration

| Metric | Result | Preregistered Bar | Verdict |
| --- | ---: | ---: | --- |
| clean items | 180 | at least 180 | pass |
| clean number-answer items | 103 | at least 70 | pass |
| clean single-word-answer items | 67 | at least 70 | fail |
| clean multi-word-answer items | 10 | at least 6 | pass |
| primary weak-hint truth rows | 105 | at least 120 | fail |
| primary weak-hint agreement rows | 212 | at least 120 | pass |
| shape-matched rows | 144 | at least 160 | fail |
| strict matched rows | 132 | at least 120 | pass |
| strict matched holdout rows | 46 | at least 40 | pass |
| min condition-label strict cell | 31 | at least 25 | pass |
| max single answer-shape share of either strict label | 58/66 | at most 70% | fail |
| every strict holdout key covered in discovery | no | yes | fail |

Clean item counts by answer shape:

| Correct Answer Shape | Clean Items |
| --- | ---: |
| number | 103 |
| single word | 67 |
| multi word | 10 |

Strict matched answer-shape label counts:

| Correct Answer Shape | Truth | Agreement |
| --- | ---: | ---: |
| number | 58 | 58 |
| single word | 7 | 7 |
| multi word | 1 | 1 |

Strict holdout keys missing discovery support:

- `wrong_disclaimed|number|number|char_3_6|char_3_6`
- `wrong_disclaimed|single_word|single_word|char_1_2|char_1_2`
- `wrong_disclaimed|single_word|single_word|char_3_6|char_3_6`
- `wrong_unsure|number|number|char_3_6|char_3_6`
- `wrong_unsure|single_word|single_word|char_1_2|char_1_2`
- `wrong_unsure|single_word|single_word|char_3_6|char_3_6`

## Diagnosis

The generated-answer interface answers a different question than the
letter-choice and pairwise branches. It shows that base Gemma can express a
usable truth-versus-hint pressure gradient without displayed choices:

- no-hint truth is 189/237 under conservative grading;
- correct-hint truth is 218/237;
- direct wrong hints produce 190/237 agreement errors;
- high wrong hints produce 225/237 agreement errors.

But the weak-hint matched substrate is not reliability-clean. The useful
matched rows are dominated by number answers, while single-word and multi-word
rows are sparse after clean filtering. That means a hidden-state signature could
still be an answer-shape or arithmetic-substrate signal, not a general
truth-versus-user-agreement control surface.

## Decision

Do not resume hidden-state discovery, sparse-feature discovery, path
localization, activation patching, or steering on the MC001G generated-answer
text substrate.

Close MC001G as:

> repaired behavior, matched signatures, failed interventions,
> sparse-promotion failure, format-control failure, pairwise-interface failure,
> and generated-interface failure.

The next useful work should not be another immediate intervention. Either build
a generated-text V2 source bank that deliberately repairs single-word and
multi-word clean coverage plus strict holdout-key coverage, or move to a new
behavior family whose controls are not dominated by answer format.
