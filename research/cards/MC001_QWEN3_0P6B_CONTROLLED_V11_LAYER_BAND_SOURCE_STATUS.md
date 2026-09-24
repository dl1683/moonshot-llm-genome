# MC-001 Qwen3-0.6B Controlled V11 Layer-Band Source Status

Status: complete. Result: broad source-token dependence, still not a supported mechanism card.

V10 found only weak single-head and single-layer attention-source localization. V11 asked whether the source-token effect becomes strong when the same source mask is applied across cumulative all-head layer bands.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v11_layer_band_source.py`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v11_layer_band_source_manifest.jsonl`
- manifest SHA256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- smoke result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v11_layer_band_source_20260629T232137.json`
- full result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v11_layer_band_source_20260629T233533.json`
- elapsed time: `777.5` seconds

## Design

V11 uses the V10 eager-attention hook and installs it across multiple all-head layer bands. This is narrower than V9 and more precise:

- V9 zeroed source-token positions in the input attention mask. That treats the masked source tokens like padding throughout prompt processing.
- V10/V11 preserve the prompt tokens but block selected generation-query attention to selected source positions at selected layer/head paths.

Therefore V11 is not expected to exactly reproduce V9's coarse effect unless the decisive path is direct generation-query attention to the source tokens. The V9/V11 distinction is now central: the large V9 effect may come from prompt-processing or source-removal semantics, not just generation-query attention to already-processed hint tokens.

## Bands

Primary bands:

- all layers: L00-L27;
- early: L00-L06;
- middle: L07-L13;
- upper: L14-L20;
- late: L21-L27;
- focused late/core bands around V10's weak hits: L22, L21-L22, L17+L21+L22, L17-L22, L14-L22, L20-L23, L20-L27.

Primary source:

- `hint_answer`;

Diagnostic sources:

- `hint_line`;
- `hint_non_answer`;

Matched controls:

- `answer_instruction_matched_hint_answer`;
- `random_matched_hint_answer`.

## Baseline

The V11 eager-attention baseline produced 384 parseable generations:

| Rows | Truth | User Agreement | Other |
| --- | ---: | ---: | ---: |
| all rows | 177/384 = 46.1 percent | 156/384 = 40.6 percent | 51/384 = 13.3 percent |
| hard agreement-favored wrong-hint bin | 0/103 = 0.0 percent | 99/103 = 96.1 percent | 4/103 = 3.9 percent |
| no-hint plus correct-hint side rows | 53/72 = 73.6 percent | 7/72 = 9.7 percent | 12/72 = 16.7 percent |
| anti-wrong side rows | 20/36 = 55.6 percent | 8/36 = 22.2 percent | 8/36 = 22.2 percent |

## Hard-Bin Result

Primary hard-bin results:

| Arm | Hard Truth | Hard Agreement | Hard Other |
| --- | ---: | ---: | ---: |
| baseline | 0/103 | 99/103 | 4/103 |
| `hint_line__all_L00_L27` | 24/103 | 57/103 | 22/103 |
| `hint_answer__all_L00_L27` | 22/103 | 57/103 | 24/103 |
| `hint_line__band_L17_L22` | 21/103 | 53/103 | 29/103 |
| `hint_line__band_L14_L22` | 20/103 | 60/103 | 23/103 |
| `hint_answer__band_L17_L22` | 19/103 | 55/103 | 29/103 |
| `hint_answer__band_L14_L22` | 19/103 | 60/103 | 24/103 |
| `hint_line__late_L21_L27` | 12/103 | 73/103 | 18/103 |
| `hint_answer__top8_L20_L27` | 11/103 | 74/103 | 18/103 |
| `hint_answer__late_L21_L27` | 10/103 | 75/103 | 18/103 |
| `hint_answer__single_L22` | 5/103 | 87/103 | 11/103 |

Matched controls:

| Arm | Hard Truth | Hard Agreement | Hard Other |
| --- | ---: | ---: | ---: |
| `answer_instruction_matched_hint_answer__all_L00_L27` | 15/103 | 84/103 | 4/103 |
| `answer_instruction_matched_hint_answer__band_L17_L22` | 15/103 | 84/103 | 4/103 |
| `answer_instruction_matched_hint_answer__top8_L20_L27` | 9/103 | 90/103 | 4/103 |
| `answer_instruction_matched_hint_answer__late_L21_L27` | 7/103 | 92/103 | 4/103 |
| `random_matched_hint_answer__all_L00_L27` | 1/103 | 98/103 | 4/103 |
| `random_matched_hint_answer__band_L17_L22` | 1/103 | 98/103 | 4/103 |
| `random_matched_hint_answer__top8_L20_L27` | 0/103 | 99/103 | 4/103 |
| `hint_non_answer__all_L00_L27` | 1/103 | 99/103 | 3/103 |
| `hint_non_answer__band_L17_L22` | 0/103 | 99/103 | 4/103 |

## Side Effects

On no-hint plus correct-hint side rows, baseline was 53/72 truth. The broad source masks reduced this modestly:

| Arm | No/Correct Truth | No/Correct Agreement | No/Correct Other |
| --- | ---: | ---: | ---: |
| baseline | 53/72 | 7/72 | 12/72 |
| `hint_answer__all_L00_L27` | 48/72 | 8/72 | 16/72 |
| `hint_line__all_L00_L27` | 47/72 | 8/72 | 17/72 |
| `hint_answer__band_L17_L22` | 48/72 | 8/72 | 16/72 |
| `hint_line__band_L17_L22` | 48/72 | 8/72 | 16/72 |
| `answer_instruction_matched_hint_answer__all_L00_L27` | 50/72 | 7/72 | 15/72 |
| `random_matched_hint_answer__all_L00_L27` | 52/72 | 8/72 | 12/72 |

On anti-wrong rows, most target source masks were near baseline or slightly higher truth-following:

| Arm | Anti-Wrong Truth | Anti-Wrong Agreement | Anti-Wrong Other |
| --- | ---: | ---: | ---: |
| baseline | 20/36 | 8/36 | 8/36 |
| `hint_answer__all_L00_L27` | 21/36 | 7/36 | 8/36 |
| `hint_line__all_L00_L27` | 19/36 | 8/36 | 9/36 |
| `hint_answer__band_L17_L22` | 21/36 | 7/36 | 8/36 |
| `hint_line__band_L17_L22` | 19/36 | 9/36 | 8/36 |

## Interpretation

What V11 supports:

- cumulative source masking is materially stronger than V10 single-head/single-layer masking;
- the active path is concentrated in broad late-to-upper bands, especially L17-L22 and L14-L22;
- the effect is source-specific relative to random matched source masks;
- the full hint line is at least as active as the hinted answer letter alone.

What V11 rules against:

- no tested precise generation-query attention mask recovers most of V9's coarse source-token effect;
- the strongest V11 hard-bin truth result, `hint_line__all_L00_L27` at 24/103, is far below V9's coarse `hint_line` 63/104;
- the strongest precise `hint_answer` result, 22/103, is far below V9's coarse `hint_answer` 50/104;
- compact late bands recover much of the weak V11 all-layer effect, but the effect itself is not strong enough for a mechanism card;
- answer-instruction controls are not zero, so broad all-layer source masking is not a clean hint-answer mechanism even though random controls stay near baseline.

## Verdict

V11 does not support a mechanism-card promotion.

Current conclusion:

> Qwen3-0.6B wrong-hint agreement has a real source-token dependence, but the large V9 effect is not explained by direct generation-query attention to source tokens at any tested head, layer, or cumulative layer band. V11 closes cumulative attention-source masking as a weak, distributed control-only result.

The important result was the semantic split between V9 and V11. V9's coarse input-mask operation changed how source tokens participated in prompt processing. V11 left prompt processing intact and only blocked generation-query access. Since V11 only recovered a weak fraction of V9, V12 decomposed source-removal semantics rather than continuing broader attention-band searches.

## Handoff To V12

Do not restart dense residual-stream steering, dense prompt-prefill steering, answer-prefix dense steering, single-head source localization, or cumulative generation-query source masking.

The next 0.6B pass at V11 time was an input-mask semantics audit, now completed by V12:

1. compare V9-style input attention-mask zeroing against literal prompt deletion of the hinted answer and full hint line;
2. compare both against neutral or matched placeholder replacement;
3. keep the V11 generation-query all-layer mask as a reference arm;
4. keep answer-instruction and random matched controls;
5. decide whether the useful effect is source removal, prompt-state recomputation, positional/layout disruption, or direct generation-query attention.

V12 showed mostly broad prompt-state/source-removal dependence. The 0.6B MC-001 result should remain a diagnostic control-surface card, and the project should escalate to Qwen3-1.7B or an artifact-rich Gemma stack for a cleaner mechanism attempt unless a narrow layout/tokenization sanity check is needed.
