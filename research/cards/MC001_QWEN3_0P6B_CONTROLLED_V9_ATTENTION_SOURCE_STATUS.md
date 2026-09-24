# MC-001 Qwen3-0.6B Controlled V9 Attention-Source Status

Status: attention-source ablation complete; source-token dependence supported; not a supported mechanism card.

V9 is the first positive path-local result after the dense residual-stream branches failed. Masking the user-hint source tokens sharply reduces wrong-hint agreement, and masking only the hinted answer letter also has a large effect. Matched answer-instruction and random source masks do not reproduce the result.

The result is still not a mechanism card. Source masking is a coarse causal ablation. It proves that the behavior depends on the hint source tokens, but it does not yet identify which attention heads, layers, value paths, MLP paths, or features carry the effect.

## Purpose

V8 closed dense answer-prefix steering. V9 asked a more local question:

> If attention to the user hint source tokens is blocked at generation time, does wrong-hint agreement fall more than under matched source-token controls?

The test used generation validation on the same controlled manifest. It did not train a new dense direction.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v9_attention_source_ablation.py`
- result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v9_attention_source_ablation_20260629T224234.json`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v9_attention_source_ablation_manifest.jsonl`
- manifest SHA-256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- model: `Qwen/Qwen3-0.6B`
- rows: same 384-row controlled v2-v8 manifest
- validation rows: 288
- elapsed runtime: 154.7 seconds

Mask arms:

- `baseline`: no source masking.
- `mask_hint_line`: zero attention to tokens on the `User hint:` line.
- `mask_hint_answer`: zero attention only to the hinted answer letter inside the hint line.
- `mask_question_matched_hint_line`: zero a matched number of question tokens.
- `mask_answer_instruction_matched_hint_line`: zero a matched number of final answer-instruction tokens.
- `mask_random_matched_hint_line`: zero a matched number of non-hint content tokens.

Average masked-token counts over the full 384 rows:

| Group | Mean | Max | Nonzero Rows |
| --- | ---: | ---: | ---: |
| hint line | 18.375 | 25 | 336 |
| hint answer | 1.125 | 2 | 336 |
| question matched to hint line | 17.911 | 25 | 336 |
| answer instruction matched to hint line | 13.875 | 16 | 336 |
| random matched to hint line | 18.375 | 25 | 336 |

## Baseline

Across all 384 rows, the unmasked baseline produced:

| Truth | Agreement | Other | Parseable |
| ---: | ---: | ---: | ---: |
| 182/384 = 47.4 percent | 156/384 = 40.6 percent | 46/384 = 12.0 percent | 384/384 |

Across the 288 validation rows:

| Filter | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| all validation | 135/288 = 46.9 percent | 117/288 = 40.6 percent | 36/288 = 12.5 percent | 288/288 |
| wrong-hint rows | 61/180 = 33.9 percent | 101/180 = 56.1 percent | 18/180 = 10.0 percent | 180/180 |
| no-hint plus correct-hint rows | 54/72 = 75.0 percent | 8/72 = 11.1 percent | 10/72 = 13.9 percent | 72/72 |

## Main Results

All validation rows:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline | 135/288 = 46.9 percent | 117/288 = 40.6 percent | 36/288 = 12.5 percent | 288/288 |
| mask hint line | 197/288 = 68.4 percent | 26/288 = 9.0 percent | 65/288 = 22.6 percent | 288/288 |
| mask hint answer | 181/288 = 62.8 percent | 35/288 = 12.2 percent | 72/288 = 25.0 percent | 288/288 |
| mask question matched to hint line | 64/288 = 22.2 percent | 148/288 = 51.4 percent | 76/288 = 26.4 percent | 288/288 |
| mask answer instruction matched to hint line | 140/288 = 48.6 percent | 120/288 = 41.7 percent | 28/288 = 9.7 percent | 288/288 |
| mask random matched to hint line | 66/287 = 23.0 percent | 132/287 = 46.0 percent | 89/287 = 31.0 percent | 287/288 |

Wrong-hint rows:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline | 61/180 = 33.9 percent | 101/180 = 56.1 percent | 18/180 = 10.0 percent | 180/180 |
| mask hint line | 124/180 = 68.9 percent | 16/180 = 8.9 percent | 40/180 = 22.2 percent | 180/180 |
| mask hint answer | 110/180 = 61.1 percent | 23/180 = 12.8 percent | 47/180 = 26.1 percent | 180/180 |
| mask question matched to hint line | 18/180 = 10.0 percent | 126/180 = 70.0 percent | 36/180 = 20.0 percent | 180/180 |
| mask answer instruction matched to hint line | 57/180 = 31.7 percent | 111/180 = 61.7 percent | 12/180 = 6.7 percent | 180/180 |
| mask random matched to hint line | 19/179 = 10.6 percent | 112/179 = 62.6 percent | 48/179 = 26.8 percent | 179/180 |

Agreement-favored wrong-hint rows:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0/104 = 0.0 percent | 101/104 = 97.1 percent | 3/104 = 2.9 percent | 104/104 |
| mask hint line | 63/104 = 60.6 percent | 16/104 = 15.4 percent | 25/104 = 24.0 percent | 104/104 |
| mask hint answer | 50/104 = 48.1 percent | 23/104 = 22.1 percent | 31/104 = 29.8 percent | 104/104 |
| mask question matched to hint line | 0/104 = 0.0 percent | 104/104 = 100.0 percent | 0/104 = 0.0 percent | 104/104 |
| mask answer instruction matched to hint line | 4/104 = 3.8 percent | 97/104 = 93.3 percent | 3/104 = 2.9 percent | 104/104 |
| mask random matched to hint line | 5/103 = 4.9 percent | 84/103 = 81.6 percent | 14/103 = 13.6 percent | 103/104 |

No-hint plus correct-hint side-effect rows:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline | 54/72 = 75.0 percent | 8/72 = 11.1 percent | 10/72 = 13.9 percent | 72/72 |
| mask hint line | 49/72 = 68.1 percent | 7/72 = 9.7 percent | 16/72 = 22.2 percent | 72/72 |
| mask hint answer | 47/72 = 65.3 percent | 8/72 = 11.1 percent | 17/72 = 23.6 percent | 72/72 |
| mask question matched to hint line | 37/72 = 51.4 percent | 13/72 = 18.1 percent | 22/72 = 30.6 percent | 72/72 |
| mask answer instruction matched to hint line | 57/72 = 79.2 percent | 5/72 = 6.9 percent | 10/72 = 13.9 percent | 72/72 |
| mask random matched to hint line | 39/72 = 54.2 percent | 9/72 = 12.5 percent | 24/72 = 33.3 percent | 72/72 |

## Split Robustness

Wrong-hint truth-following under hint-line masking stayed positive across all validation splits:

| Arm | Calibration | Holdout | Paraphrase Holdout |
| --- | ---: | ---: | ---: |
| baseline | 24/60 | 19/60 | 18/60 |
| mask hint line | 45/60 | 41/60 | 38/60 |
| mask hint answer | 43/60 | 34/60 | 33/60 |
| mask answer instruction matched to hint line | 21/60 | 19/60 | 17/60 |
| mask random matched to hint line | 3/60 | 10/60 | 6/60 |

The paraphrase split weakens the hint-answer-only arm but does not erase it.

## Interpretation

What V9 supports:

- wrong-hint agreement depends causally on attention to hint source tokens;
- the hinted answer letter carries much of the effect, because masking only that source token moves agreement-favored rows from 0/104 to 50/104 truth-following;
- the effect is not reproduced by same-size answer-instruction masking;
- the effect is not reproduced by same-size random content-token masking;
- the result survives calibration, holdout, and paraphrase holdout.

What V9 does not support yet:

- a named mechanism card;
- a head-local, layer-local, MLP-local, or feature-local explanation;
- a deployable control, because full hint-line and hint-answer masking increase `other_error`;
- treating question-matched masking as a clean null, because it is a destructive semantic ablation.

The agreement-favored bin is the decisive positive. Baseline truth-following is 0/104. Hint-line masking reaches 63/104 and hint-answer masking reaches 50/104. The matched answer-instruction mask reaches only 4/104, and the matched random mask reaches 5/103 parseable. That is a large source-token-specific causal effect.

The side-effect rows keep the result below mechanism-card standard. On no-hint plus correct-hint rows, hint-line masking drops truth from 54/72 to 49/72 and raises other errors from 10/72 to 16/72. Hint-answer masking drops truth to 47/72 and raises other errors to 17/72. This is much better evidence than dense steering, but not yet a precise intervention.

## Verdict

Do not write a supported mechanism card from V9.

The supported claim is narrower:

> On Qwen3-0.6B MC-001 prompts, wrong-hint agreement has a strong causal dependence on the user-hint source tokens, especially the hinted answer letter. This dependence is source-specific relative to matched answer-instruction and random source masks, but it is not yet localized to a circuit, head, layer, feature, or clean deployable intervention.

The mechanism claim remains open because:

- source masking is coarse;
- side effects are still material;
- the carrying path from hint source token to final answer token is unidentified;
- no attention-head, layer, MLP-path, or feature-level ablation has passed controls.

## V10 Handoff

The next local experiment described below has been run as V10.

V10 result:

- status card: `research/cards/MC001_QWEN3_0P6B_CONTROLLED_V10_HEAD_LOCALIZATION_STATUS.md`
- result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v10_head_localization_20260629T231146.json`
- verdict: single-head and single-layer source masking found only weak late localization. The best individual head, `hint_answer` layer 22 head 9, moved the V10 hard bin from 0/103 to 7/103 truth-following. The next 0.6B step at V10 time was cumulative layer-band source masking, now completed by V11.

## Next Decision At V9 Time

At V9 time, the decision was to continue Qwen3-0.6B for one narrower path-local iteration, not for another dense residual-stream sweep.

That local experiment was expected to:

1. use the agreement-favored wrong-hint bin as the primary target;
2. separate full hint-line source, hinted-answer source, and non-answer hint-word source;
3. localize the effect by layer and attention head during answer-token generation;
4. keep answer-instruction and random source masks as matched nulls;
5. retain no-hint, correct-hint, and anti-wrong side-effect rows;
6. promote only if a head/layer/path ablation preserves most of the hint-source effect with fewer side effects.

Escalate only after this head/layer path-local question is answered or the 0.6B substrate proves too coarse for localization.
