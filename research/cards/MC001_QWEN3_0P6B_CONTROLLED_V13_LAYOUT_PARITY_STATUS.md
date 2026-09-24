# MC-001 Qwen3-0.6B Controlled V13 Layout-Parity Status

Status: complete.

V13 closes the optional layout/tokenization caveat left after V12. Character-matched and rendered-token-count-matched neutral hint-line replacements do not move the result toward a new mechanism claim. They stay in the same broad source-removal/prompt-rewrite family and remain far above query-only attention masking.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v13_layout_parity.py`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v13_layout_parity_manifest.jsonl`
- manifest SHA256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- full result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v13_layout_parity_20260630T000706.json`
- elapsed: 185.0 seconds

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

Does a neutral replacement that preserves character length or rendered-token count behave like V12 neutral/deletion rewrites, or does it move closer to V9-style input masking?

## Design

V13 reuses the V10/V11/V12 eager-attention baseline and selection rule:

- validation rows only;
- hard rows are wrong-hint rows where baseline next-token margin favors the user's wrong hint;
- side rows are no-hint, correct-hint, and anti-wrong rows.

Arms:

- `baseline`;
- `input_mask_hint_line`, the V9-style coarse source-removal reference;
- `rewrite_neutralize_hint_line`, the V12 neutral-line reference;
- `rewrite_char_matched_neutral_hint_line`, a neutral `User hint:` replacement with exactly the same character length as the original hint line;
- `rewrite_token_matched_neutral_hint_line`, a neutral `User hint:` replacement chosen to make the rendered prompt token count exactly match the original;
- `rewrite_delete_hint_line`, the V12 line-deletion reference;
- `query_mask_hint_line_all`, the V11-style query-only source-mask reference.

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

| Arm | Truth | Agreement | Other |
| --- | ---: | ---: | ---: |
| `input_mask_hint_line` | 63/103 | 15/103 | 25/103 |
| `rewrite_neutralize_hint_line` | 60/103 | 20/103 | 23/103 |
| `rewrite_delete_hint_line` | 57/103 | 15/103 | 31/103 |
| `rewrite_char_matched_neutral_hint_line` | 52/103 | 20/103 | 31/103 |
| `rewrite_token_matched_neutral_hint_line` | 52/103 | 21/103 | 30/103 |
| `query_mask_hint_line_all` | 24/103 | 57/103 | 22/103 |
| baseline | 0/103 | 99/103 | 4/103 |

Combined hard-plus-side 211-row eval set:

| Arm | Truth | Agreement | Other |
| --- | ---: | ---: | ---: |
| `input_mask_hint_line` | 135/211 | 24/211 | 52/211 |
| `rewrite_neutralize_hint_line` | 131/211 | 31/211 | 49/211 |
| `rewrite_delete_hint_line` | 126/211 | 24/211 | 61/211 |
| `rewrite_char_matched_neutral_hint_line` | 119/211 | 30/211 | 62/211 |
| `rewrite_token_matched_neutral_hint_line` | 118/211 | 32/211 | 61/211 |
| `query_mask_hint_line_all` | 90/211 | 73/211 | 48/211 |
| baseline | 73/211 | 114/211 | 24/211 |

Side rows show the same broadness as V12. On no-hint plus correct-hint rows, baseline truth is 53/72. The matched neutral rewrites reduce it to 45/72. On anti-wrong rows, baseline truth is 20/36, char-matched neutral is 22/36, and token-matched neutral is 21/36.

## Parity Summary

Rewrite metadata across the 211-row eval set:

| Rewrite arm | Applied rows | Rendered token delta mean | Rendered token delta min/max | Hint-line char delta mean |
| --- | ---: | ---: | ---: | ---: |
| `rewrite_neutralize_hint_line` | 175 | -6.332 | -13 / 0 | -25.095 |
| `rewrite_delete_hint_line` | 175 | -16.284 | -25 / 0 | -78.175 |
| `rewrite_char_matched_neutral_hint_line` | 175 | +19.720 | 0 / +37 | 0.000 |
| `rewrite_token_matched_neutral_hint_line` | 175 | 0.000 | 0 / 0 | -39.801 |

The token-matched neutral arm exactly preserves rendered prompt token count on every rewritten row, yet it reaches only 52/103 hard-bin truth. Preserving token count does not recreate the stronger `input_mask_hint_line` result.

The char-matched neutral arm exactly preserves hint-line character count, but changes token count upward and also reaches 52/103. Preserving visible length does not reveal a hidden localized mechanism either.

## Interpretation

V13 does not overturn V12. It makes the closeout stronger:

- the big separation remains between prompt-source rewriting/removal and query-only attention masking;
- length matching and token-count matching do not recover a new mechanism-like effect;
- the matched neutral arms are still broad and side-effectful;
- the remaining difference between `input_mask_hint_line` at 63/103 and matched neutral rewrites at 52/103 is not enough to justify more broad 0.6B localization.

The lower matched-neutral result probably reflects replacement content semantics: the V12 neutral line explicitly said to solve independently, while V13 uses neutral filler to preserve layout. That is a prompt-source intervention, not a hidden circuit.

## Verdict

Do not promote V13 to a mechanism card.

V13 closes the last defensible Qwen3-0.6B MC-001 follow-up. The completed v1-v13 series supports this final 0.6B statement:

> Qwen3-0.6B has a reproducible wrong-hint control surface. Dense residual-stream directions, dense prefill, answer-prefix directions, single-head/layer source masking, cumulative generation-query source masking, input-mask semantics, and layout/tokenization parity checks do not identify a compact mechanism. The strongest source effect is prompt-source removal or recomputation, not a mechanism-card-ready internal circuit.

## Next Work

For MC-001, stop broad Qwen3-0.6B mechanism search.

Future work should either:

1. package the Qwen3-0.6B failed-mechanism and diagnostic/control-surface card from v1-v13; or
2. start the next mechanism-card attempt on Qwen3-1.7B or an artifact-rich Gemma stack.
