# MC-001 Qwen3-0.6B Controlled V12 Input-Mask Semantics Status

Status: complete.

V12 explains the large V9 source-token effect. V9-style input `attention_mask` zeroing is not behaving like a clean generation-query attention intervention. Literal prompt rewrites nearly reproduce the large effect, while V11-style generation-query masks stay much weaker.

The practical conclusion is future-facing: stop broad Qwen3-0.6B dense-direction and attention-localization searches for MC-001. The 0.6B result is a useful diagnostic/control surface and failed-mechanism artifact. V13 later completed the narrow layout/tokenization sanity check. A future mechanism-card attempt should move to a larger or more artifact-rich stack.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v12_input_mask_semantics.py`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v12_input_mask_semantics_manifest.jsonl`
- manifest SHA256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- smoke result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v12_input_mask_semantics_20260629T234828.json`
- full result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v12_input_mask_semantics_20260629T235517.json`
- elapsed: 363.6 seconds

Candidate counts:

- discovery: 96
- calibration: 96
- holdout: 96
- paraphrase holdout: 96
- validation rows: 288
- hard agreement-favored wrong-hint rows: 103
- side-effect rows: 108
- generation eval rows: 211

## Question

V11 answered the attention-band question: cumulative generation-query source masking is real, but much weaker than V9. V12 asked:

> Was V9's large source-token effect caused by source removal, prompt-state recomputation, positional/layout disruption, or direct generation-query attention?

## Design

V12 uses the V10/V11 eager-attention baseline and the same selection rule:

- validation rows only;
- hard rows are wrong-hint rows where baseline next-token margin favors the user's wrong hint;
- side rows are no-hint, correct-hint, and anti-wrong rows.

The arms compare three semantics:

1. `input_mask_*`: V9-style source positions are set to zero in the input `attention_mask` before generation.
2. `rewrite_*`: the unrendered prompt is rewritten before chat-template rendering.
3. `query_mask_*`: V11-style all-layer generation-query source masks preserve prompt processing.

Prompt rewrites:

- delete the full `User hint:` line;
- replace the full `User hint:` line with a neutral hidden-hint line;
- delete standalone target answer letters inside the hint line;
- replace standalone target answer letters inside the hint line with `X`.

Controls:

- answer-instruction positions matched to the full hint line;
- random content positions matched to the full hint line;
- answer-instruction positions matched to the hinted answer group;
- random content positions matched to the hinted answer group.

## Baseline

All 384 manifest rows:

| Parseable | Truth | User-agreement error | Other error |
| ---: | ---: | ---: | ---: |
| 384/384 | 177/384 | 156/384 | 51/384 |

Hard agreement-favored wrong-hint bin:

| Arm | Truth | Agreement | Other |
| --- | ---: | ---: | ---: |
| baseline | 0/103 | 99/103 | 4/103 |

## Full Result

Hard agreement-favored wrong-hint bin:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| `input_mask_hint_line` | 63/103 | 15/103 | 25/103 | 103/103 |
| `rewrite_neutralize_hint_line` | 60/103 | 20/103 | 23/103 | 103/103 |
| `rewrite_delete_hint_line` | 57/103 | 15/103 | 31/103 | 103/103 |
| `rewrite_placeholder_hint_answer_X` | 51/103 | 22/103 | 30/103 | 103/103 |
| `input_mask_hint_answer` | 49/103 | 23/103 | 31/103 | 103/103 |
| `rewrite_delete_hint_answer` | 48/103 | 26/103 | 29/103 | 103/103 |
| `input_mask_hint_non_answer` | 44/103 | 40/103 | 19/103 | 103/103 |
| `query_mask_hint_line_all` | 24/103 | 57/103 | 22/103 | 103/103 |
| `query_mask_hint_answer_all` | 22/103 | 57/103 | 24/103 | 103/103 |
| `input_mask_answer_instruction_matched_hint_answer` | 15/103 | 88/103 | 0/103 | 103/103 |
| `input_mask_random_matched_hint_line` | 6/103 | 83/103 | 13/103 | 102/103 |
| `input_mask_answer_instruction_matched_hint_line` | 4/103 | 96/103 | 3/103 | 103/103 |
| `input_mask_random_matched_hint_answer` | 3/103 | 96/103 | 4/103 | 103/103 |
| `query_mask_hint_non_answer_all` | 1/103 | 99/103 | 3/103 | 103/103 |
| baseline | 0/103 | 99/103 | 4/103 | 103/103 |

The key comparison is:

- full-line input masking: 63/103 hard-bin truth;
- full-line neutral rewrite: 60/103;
- full-line deletion rewrite: 57/103;
- full-line generation-query mask: 24/103.

For answer-token targeting:

- answer input masking: 49/103;
- answer placeholder rewrite: 51/103;
- answer deletion rewrite: 48/103;
- answer generation-query mask: 22/103.

This is the decisive V12 pattern. Literal prompt rewrites nearly match V9-style input masking. Generation-query attention masks do not.

## Side Effects

On the combined hard-plus-side 211-row eval set:

| Arm | Truth | Agreement | Other |
| --- | ---: | ---: | ---: |
| `input_mask_hint_line` | 135/211 | 24/211 | 52/211 |
| `rewrite_neutralize_hint_line` | 131/211 | 31/211 | 49/211 |
| `rewrite_delete_hint_line` | 126/211 | 24/211 | 61/211 |
| `rewrite_placeholder_hint_answer_X` | 119/211 | 33/211 | 59/211 |
| `input_mask_hint_answer` | 116/211 | 34/211 | 61/211 |
| `rewrite_delete_hint_answer` | 116/211 | 37/211 | 58/211 |
| baseline | 73/211 | 114/211 | 24/211 |

On no-hint plus correct-hint side rows, baseline was 53/72 truth. The primary source-removal arms reduce ordinary/correct-hint truth:

| Arm | Truth | Agreement | Other |
| --- | ---: | ---: | ---: |
| baseline | 53/72 | 7/72 | 12/72 |
| `input_mask_hint_line` | 48/72 | 6/72 | 18/72 |
| `rewrite_neutralize_hint_line` | 47/72 | 7/72 | 18/72 |
| `rewrite_delete_hint_line` | 46/72 | 6/72 | 20/72 |
| `input_mask_hint_answer` | 45/72 | 7/72 | 20/72 |

The source-removal behavior is useful causally, but it is broad and side-effectful. It is not a clean deployable intervention.

## Rewrite Counts

Rewrite application counts across the 211-row eval set:

| Rewrite arm | Applied rows | Mean replacements | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `rewrite_delete_hint_line` | 175 | 0.829 | 0 | 1 |
| `rewrite_neutralize_hint_line` | 175 | 0.829 | 0 | 1 |
| `rewrite_delete_hint_answer` | 175 | 1.123 | 0 | 2 |
| `rewrite_placeholder_hint_answer_X` | 175 | 1.123 | 0 | 2 |

The answer-letter rewrites can replace more than one standalone target-letter occurrence in a hint line. That is acceptable for this semantics audit because the goal is to test source-answer removal or neutralization, not to certify a minimal token-level circuit.

## Interpretation

V12 rules out the main productive but wrong next move: do not continue trying to recover V9 with broader generation-query attention masks on Qwen3-0.6B.

The large V9 effect is mostly explained by removing or neutralizing the prompt source content before the model forms its prompt state. The effect is not uniquely caused by padding/layout disruption, because literal rewrites nearly reproduce input masking. It is also not direct generation-query attention to the hint source tokens, because query-only masks are far weaker.

The effect is source-specific enough to matter: random and answer-instruction matched input-mask controls stay far below the primary hint-line and hint-answer arms. But it is not mechanism-card clean:

- the full hint line and the answer letter both matter;
- hint non-answer text alone still moves the hard bin to 44/103 truth;
- source-removal arms raise other errors and reduce ordinary/correct-hint side-row truth;
- the operation changes the prompt state rather than localizing a stable internal mechanism.

## Verdict

Do not promote V12 to a mechanism card.

V12 converts V9 from an open attention-localization target into a closed diagnostic/control result:

> Qwen3-0.6B wrong-hint agreement in MC-001 depends strongly on the user-hint source content. Coarse input-mask ablation and literal prompt rewrites can remove much of that dependence. Precise generation-query attention masks recover only a weak fraction of the effect, so the current 0.6B evidence does not identify a compact attention, residual-stream, or feature-level mechanism.

## Next Work

For Qwen3-0.6B MC-001, the broad search is closed:

- do not restart dense residual-stream steering;
- do not restart dense answer-prefix steering;
- do not rerun single-head, single-layer, or cumulative generation-query source-mask searches;
- do not treat V9 input masking as a localized attention mechanism.

Future work should either:

1. write the Qwen3-0.6B failed-mechanism and diagnostic/control-surface artifact from the completed v1-v13 series; or
2. escalate the mechanism-card attempt to Qwen3-1.7B or an artifact-rich Gemma stack.

The only narrow 0.6B follow-up left after V12 was a layout/tokenization parity sanity check with length-matched neutral placeholders. V13 completed that check and did not reopen the mechanism-card search.
