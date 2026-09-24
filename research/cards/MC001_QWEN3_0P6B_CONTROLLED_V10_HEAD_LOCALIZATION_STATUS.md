# MC-001 Qwen3-0.6B Controlled V10 Head-Localization Status

Status: complete. Result: weak localization only; not a supported mechanism card.

V9 showed that wrong-hint agreement depends causally on user-hint source tokens, especially the hinted answer letter. V10 asked whether that coarse source-token effect could be localized to a single attention head, layer, or small layer/head path.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v10_head_localization.py`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v10_head_localization_manifest.jsonl`
- manifest SHA256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- full result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v10_head_localization_20260629T231146.json`
- elapsed time: `625.0` seconds

## Design

V10 used Qwen3-0.6B with eager attention so that explicit source-token masks could be injected into selected attention heads.

The audit had two stages:

1. **Next-token screen:** block source positions for one layer/head at a time while measuring option logits on the agreement-favored wrong-hint bin.
2. **Generation validation:** rerun the strongest candidates on hard rows plus no-hint, correct-hint, and anti-wrong side-effect rows.

Primary source groups:

- `hint_answer`: the hinted answer letter inside the `User hint:` line.
- `hint_non_answer`: hint-line tokens other than the hinted answer letter.
- `hint_line`: the full `User hint:` line.

Matched controls:

- `answer_instruction_matched_hint_answer`: a matched number of source tokens from the final answer-instruction line.
- `random_matched_hint_answer`: a matched number of non-hint content tokens.

## Baseline

The V10 eager-attention baseline produced 384 parseable generations:

| Rows | Truth | User Agreement | Other |
| --- | ---: | ---: | ---: |
| all rows | 177/384 = 46.1 percent | 156/384 = 40.6 percent | 51/384 = 13.3 percent |
| hard agreement-favored wrong-hint bin | 0/103 = 0.0 percent | 99/103 = 96.1 percent | 4/103 = 3.9 percent |
| no-hint plus correct-hint side rows | 53/72 = 73.6 percent | 7/72 = 9.7 percent | 12/72 = 16.7 percent |
| anti-wrong side rows | 20/36 = 55.6 percent | 8/36 = 22.2 percent | 8/36 = 22.2 percent |

The hard bin has 103 rows in V10 rather than V9's 104 because V10 recomputed the baseline under the self-contained eager-attention implementation.

## Next-Token Screen

Top individual-head candidates on the hard bin:

| Source Mask | Layer | Head | Truth Labels | Agreement Labels | Other Labels | Mean Correct-Minus-Wrong Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hint_answer` | 22 | 9 | 6/103 | 84/103 | 13/103 | +1.805 |
| `hint_answer` | 17 | 1 | 1/103 | 94/103 | 8/103 | +1.100 |
| `hint_answer` | 21 | 13 | 1/103 | 96/103 | 6/103 | +0.381 |
| `hint_answer` | 21 | 1 | 1/103 | 96/103 | 6/103 | +0.376 |

Top all-head layer candidates:

| Source Mask | Layer | Truth Labels | Agreement Labels | Other Labels | Mean Correct-Minus-Wrong Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hint_answer` | 22 | 6/103 | 86/103 | 11/103 | +1.388 |
| `hint_line` | 22 | 6/103 | 86/103 | 11/103 | +1.351 |
| `hint_line` | 17 | 1/103 | 92/103 | 10/103 | +1.232 |
| `hint_answer` | 21 | 2/103 | 94/103 | 7/103 | +0.808 |

Matched control screen:

| Source Mask | Layer | Truth Labels | Agreement Labels | Other Labels | Mean Correct-Minus-Wrong Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `answer_instruction_matched_hint_answer` | 21 | 3/103 | 96/103 | 4/103 | +0.154 |
| `answer_instruction_matched_hint_answer` | 17 | 1/103 | 98/103 | 4/103 | +0.050 |
| `random_matched_hint_answer` | 2 | 0/103 | 97/103 | 6/103 | -0.004 |
| `random_matched_hint_answer` | 16 | 0/103 | 98/103 | 5/103 | +0.053 |

## Generation Validation

Best individual head:

| Arm | Hard Truth | Hard Agreement | Hard Other | No/Correct Truth | Anti-Wrong Truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0/103 | 99/103 | 4/103 | 53/72 | 20/36 |
| `hint_answer` L22 H09 | 7/103 | 84/103 | 12/103 | 49/72 | 21/36 |

Best layer-level source masks:

| Arm | Hard Truth | Hard Agreement | Hard Other | No/Correct Truth | Anti-Wrong Truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hint_answer` L22 all heads | 5/103 | 87/103 | 11/103 | 50/72 | 21/36 |
| `hint_line` L22 all heads | 5/103 | 86/103 | 12/103 | 50/72 | 20/36 |
| `hint_line` L17 all heads | 1/103 | 90/103 | 12/103 | 51/72 | 18/36 |
| `answer_instruction_matched_hint_answer` L21 all heads | 3/103 | 96/103 | 4/103 | 52/72 | 21/36 |
| `random_matched_hint_answer` L02 all heads | 0/103 | 99/103 | 4/103 | 53/72 | 20/36 |

The best candidate, `hint_answer` L22 H09, reduced hard-bin agreement from 99/103 to 84/103 but recovered truth on only 7/103 hard rows and increased hard-bin other errors from 4/103 to 12/103. It also reduced no/correct side-row truth from 53/72 to 49/72.

## Verdict

V10 does not support a mechanism-card promotion.

What V10 supports:

- the strongest localized signal is late, centered around layer 22;
- `hint_answer` masking is more active than random source masking;
- the effect is directionally consistent with V9's source-token result.

What V10 rules against:

- no single head recovers a substantial fraction of V9's coarse `hint_answer` effect;
- no single all-head layer recovers a substantial fraction of V9's coarse `hint_answer` or `hint_line` effect;
- the strongest head/layer effects trade agreement reduction for too much residual agreement and other-error increase;
- matched answer-instruction source masking is not zero, so a weak single-layer result is not precise enough to treat as a clean mechanism path.

Current conclusion:

> Qwen3-0.6B wrong-hint agreement depends strongly on the user-hint source tokens, but V10 finds that the attention-source effect is not concentrated in one head or one layer. The live mechanism candidate, if it exists on this model, is distributed across a layer band or requires a different path decomposition.

## Next Work At V10 Time

Do not restart dense residual-stream steering, dense prompt-prefill steering, or answer-prefix dense steering.

At V10 time, the next 0.6B pass was cumulative layer-band source masking:

1. reproduce the V9 coarse mask under the V10 eager-attention hook;
2. compare late, middle, early, and cumulative top-k layer bands for `hint_answer`;
3. keep `answer_instruction_matched_hint_answer` and `random_matched_hint_answer` controls on the same bands;
4. measure whether any compact band recovers most of the V9 hard-bin effect with bounded no/correct side effects.

V11 has now completed this follow-up. Cumulative generation-query source masking was real but too weak to explain V9's coarse input-mask result.

## V11 Handoff

V11 has now answered the cumulative layer-band question:

- status card: `research/cards/MC001_QWEN3_0P6B_CONTROLLED_V11_LAYER_BAND_SOURCE_STATUS.md`
- result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v11_layer_band_source_20260629T233533.json`
- strongest hard-bin result: `hint_line__all_L00_L27` at 24/103 truth and `hint_answer__all_L00_L27` at 22/103 truth;
- verdict: cumulative generation-query source masking is real but too weak to explain V9's coarse input-mask effect.

At V11 time, the next Qwen3-0.6B pass was to decompose V9 input-mask semantics rather than run another broader attention-band search. V12 and V13 have now completed that decomposition and closed broad Qwen3-0.6B MC-001 localization.
