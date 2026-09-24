# MC-001 Qwen3-0.6B Controlled V7 Transport Status

Status: cross-layer option-logit transport audit complete; not a supported mechanism card.

V7 did not run another generation steering sweep. It measured, in single forward passes, how the same h14 truth-minus-agreement direction changes the correct-minus-wrong option-logit margin when injected at h7, h13, h14, and h20.

The key result is that h13 and h14 have almost the same raw-direction logit effect. This explains the V6 nearby-layer result and further rejects an h14-local mechanism.

## Purpose

V6 showed that raw h14 prompt-prefill steering is real but not h14-local, because applying the same h14 vector at h13 was equally or more active. V7 asked whether that is visible directly in the answer logits.

This is a next-token option-logit audit, not a generation validation. Each arm reports the next-token answer label and correct-minus-wrong logit margin after a single forward pass.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v7_transport.py`
- result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v7_transport_20260629T221738.json`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v7_transport_manifest.jsonl`
- manifest SHA-256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- model: `Qwen/Qwen3-0.6B`
- validation rows: 288
- elapsed runtime: 201.9 seconds

Direction:

- raw h14 kind: truth-minus-agreement at the last prompt token;
- hidden index: `14`;
- discovery train rows: 66;
- discovery positive rate: 57.6 percent;
- raw norm: 3.112;
- median hidden norm: 42.631.

Residualized direction:

- hidden index: `14`;
- nuisance features: intercept, baseline margin, condition, correct answer letter, wrong answer letter, baseline next-token answer;
- nuisance rank: 13;
- raw norm after residualization: 0.242.

## Main Logit Results

All validation rows:

| Arm | Truth | Agreement | Other | Positive Margin | Mean Delta Margin | Median Delta Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline logits | 135/288 | 117/288 | 36/288 | 162/288 | 0.000 | 0.000 |
| raw h14 at h7 `alpha=0.50` | 74/288 | 74/288 | 140/288 | 152/288 | -0.106 | -0.750 |
| raw h14 at h13 `alpha=0.50` | 155/288 | 82/288 | 51/288 | 189/288 | +0.668 | +0.375 |
| raw h14 at h14 `alpha=0.50` | 152/288 | 88/288 | 48/288 | 184/288 | +0.487 | +0.250 |
| raw h14 at h20 `alpha=0.50` | 136/288 | 111/288 | 41/288 | 165/288 | +0.009 | +0.125 |
| raw random at h13 `alpha=0.50` | 136/288 | 95/288 | 57/288 | 169/288 | +0.386 | +0.875 |
| raw random at h14 `alpha=0.50` | 144/288 | 93/288 | 51/288 | 171/288 | +0.776 | +1.250 |
| residual h14 at h13 `alpha=0.50` | 80/288 | 104/288 | 104/288 | 140/288 | -0.165 | -0.375 |
| residual h14 at h14 `alpha=0.50` | 120/288 | 123/288 | 45/288 | 145/288 | -0.319 | -0.563 |

Wrong-hint rows:

| Arm | Truth | Agreement | Other | Positive Margin | Mean Delta Margin | Median Delta Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline logits | 61/180 | 101/180 | 18/180 | 74/180 | 0.000 | 0.000 |
| raw h14 at h7 `alpha=0.50` | 45/180 | 46/180 | 89/180 | 85/180 | +2.338 | +3.813 |
| raw h14 at h13 `alpha=0.50` | 74/180 | 75/180 | 31/180 | 91/180 | +2.036 | +1.875 |
| raw h14 at h14 `alpha=0.50` | 72/180 | 81/180 | 27/180 | 87/180 | +1.767 | +1.375 |
| raw h14 at h20 `alpha=0.50` | 62/180 | 96/180 | 22/180 | 77/180 | +0.341 | +0.500 |
| raw random at h13 `alpha=0.50` | 61/180 | 91/180 | 28/180 | 77/180 | +0.933 | +1.313 |
| raw random at h14 `alpha=0.50` | 65/180 | 92/180 | 23/180 | 78/180 | +1.060 | +1.500 |

Agreement-favored wrong-hint hard bin:

| Arm | Truth | Agreement | Other | Positive Margin | Mean Delta Margin | Median Delta Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline logits | 0/104 | 101/104 | 3/104 | 0/104 | 0.000 | 0.000 |
| raw h14 at h7 `alpha=0.50` | 24/104 | 36/104 | 44/104 | 38/104 | +6.065 | +6.875 |
| raw h14 at h13 `alpha=0.50` | 14/104 | 75/104 | 15/104 | 15/104 | +4.435 | +4.250 |
| raw h14 at h14 `alpha=0.50` | 13/104 | 80/104 | 11/104 | 14/104 | +3.964 | +3.500 |
| raw h14 at h20 `alpha=0.50` | 3/104 | 95/104 | 6/104 | 3/104 | +0.793 | +1.000 |
| raw random at h13 `alpha=0.50` | 6/104 | 88/104 | 10/104 | 5/104 | +1.532 | +1.625 |
| raw random at h14 `alpha=0.50` | 6/104 | 90/104 | 8/104 | 6/104 | +1.525 | +1.875 |
| residual h14 at h13 `alpha=0.50` | 17/104 | 65/104 | 22/104 | 18/104 | +5.084 | +5.469 |
| residual h14 at h14 `alpha=0.50` | 1/104 | 98/104 | 5/104 | 0/104 | +1.671 | +1.688 |
| residual random at h13 `alpha=0.50` | 13/104 | 85/104 | 6/104 | 15/104 | +3.087 | +3.000 |

No-hint plus correct-hint side-effect rows:

| Arm | Truth | Agreement | Other | Positive Margin | Mean Delta Margin | Median Delta Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline logits | 54/72 | 8/72 | 10/72 | 63/72 | 0.000 | 0.000 |
| raw h14 at h7 `alpha=0.50` | 20/72 | 19/72 | 33/72 | 49/72 | -5.335 | -5.625 |
| raw h14 at h13 `alpha=0.50` | 56/72 | 3/72 | 13/72 | 66/72 | -2.679 | -3.063 |
| raw h14 at h14 `alpha=0.50` | 55/72 | 3/72 | 14/72 | 66/72 | -2.667 | -2.938 |
| raw h14 at h20 `alpha=0.50` | 54/72 | 7/72 | 11/72 | 63/72 | -0.715 | -0.750 |

## H13 Versus H14 Transport

| Filter | Pair | Delta Pearson | Mean Delta A | Mean Delta B | Mean Abs Delta Diff |
| --- | --- | ---: | ---: | ---: | ---: |
| all validation | raw h14 h13 vs h14 `alpha=0.50` | 0.986 | +0.668 | +0.487 | 0.556 |
| wrong-hint all | raw h14 h13 vs h14 `alpha=0.50` | 0.985 | +2.036 | +1.767 | 0.569 |
| hard bin | raw h14 h13 vs h14 `alpha=0.50` | 0.970 | +4.435 | +3.964 | 0.642 |
| hard bin | raw h14 h13 vs h14 `alpha=1.00` | 0.969 | +6.962 | +5.862 | 1.273 |
| hard bin | residual h14 h13 vs h14 `alpha=0.50` | 0.767 | +5.084 | +1.671 | 3.666 |

## Interpretation

What V7 supports:

- the V6 h13/h14 interchangeability is visible directly in option logits;
- raw h14 at h13 and raw h14 at h14 produce nearly the same per-row margin-shift pattern;
- h20 is much weaker, so the effect is not uniformly available at every later layer;
- wrong-token application remains near baseline;
- h7 can move hard-bin margins but does so with severe off-target disruption.

What V7 rejects:

- an h14-local mechanism;
- treating h7 hard-bin improvement as useful, because it collapses all-validation specificity and no/correct side-effect rows;
- treating the residualized h14 result as a clean hidden mechanism, because residual h14 is active at h13 and mostly fails at h14;
- another dense cross-layer direction sweep as the next Qwen3-0.6B step.

The h13/h14 raw-direction result is the decisive one. On the hard bin, h13 and h14 have delta Pearson 0.970 and mean deltas +4.435 versus +3.964. That is not locality. It is adjacent-layer residual-stream transport.

The h7 result is a warning, not a rescue. It produces 24/104 hard-bin truth labels, but all-validation `other_error` jumps to 140/288 and no/correct truth drops from 54/72 to 20/72. That is broad disruption.

## Verdict

Do not write a supported mechanism card from V7.

The supported claim is narrower:

> Qwen3-0.6B's MC-001 control surface is a residual-stream option-logit transport effect across nearby layers, not an h14-local truth-versus-agreement mechanism.

The mechanism claim fails because:

- output margin remains dominant from V2/V3;
- h13 and h14 carry nearly identical raw-direction logit effects;
- residualized h14 does not localize to h14;
- early-layer injection can move the hard bin only through large side effects;
- generation-level V6 and logit-level V7 now agree that dense steering is control-only.

## Next Decision

Close Qwen3-0.6B dense direction, dense prefill, and cross-layer transport for MC-001 mechanism-card purposes.

Retrospective after V9: the answer-prefix and attention-source routes have now been run. See [MC-001 Qwen3-0.6B Controlled V8 Answer-Prefix Status](MC001_QWEN3_0P6B_CONTROLLED_V8_ANSWER_PREFIX_STATUS.md) and [MC-001 Qwen3-0.6B Controlled V9 Attention-Source Status](MC001_QWEN3_0P6B_CONTROLLED_V9_ATTENTION_SOURCE_STATUS.md). V8 showed that a separately trained answer-prefix h14 dense direction does not produce a clean target effect. V9 showed that hint-source masking sharply reduces wrong-hint agreement, but the carrying head/layer/path is not yet localized.

At V7 time, the next Qwen3-0.6B pass was not another dense vector sweep. The 0.6B-compatible route then was:

1. head/layer attention-source ablation from hint tokens to answer tokens;
2. MLP-path or feature-level localization if attention-head localization fails.

Otherwise, escalate to Qwen3-1.7B or an artifact-rich Gemma stack with the Qwen3-0.6B result recorded as a control-only failed mechanism.
