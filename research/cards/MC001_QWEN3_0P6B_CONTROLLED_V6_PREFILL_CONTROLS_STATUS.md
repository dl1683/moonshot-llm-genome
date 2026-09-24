# MC-001 Qwen3-0.6B Controlled V6 Prefill Controls Status

Status: additive prompt-prefill control audit complete; not a supported mechanism card.

V6 ran the matched additive prefill controls that V5 required. It confirms that raw h14 prompt-prefill steering is a real control surface, not just random noise or a wrong-token artifact. It also blocks the h14-local mechanism claim because applying the same h14 vector at nearby h13 is as strong or stronger than applying it at h14.

## Purpose

V6 asked whether the V5 prompt-prefill effect could separate from additive nulls while preserving side-effect rows.

The validation set included:

- calibration, holdout, and paraphrase holdout rows;
- all wrong-hint rows plus no-hint/correct-hint side-effect rows;
- the agreement-favored output-margin hard bin used by V4 and V5;
- matched random, wrong-token, nearby-layer, and residualized controls.

The most important hard subset was unchanged:

- conditions: `wrong_marked_false`, `wrong_untrusted`, `wrong_unsure`, `wrong_direct`, `wrong_high`;
- baseline output-margin bin: `agreement_favored`;
- selected rows: 104;
- baseline truth-following: 0/104;
- baseline user-agreement error: 101/104;
- baseline other error: 3/104.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v6_prefill_controls.py`
- result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v6_prefill_controls_20260629T220249.json`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v6_prefill_controls_manifest.jsonl`
- manifest SHA-256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- model: `Qwen/Qwen3-0.6B`
- rows: same 384-row v2/v3/v4/v5 manifest
- validation rows: 288
- elapsed runtime: 1122.1 seconds

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

## Main Results

All validation rows:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline | 135/288 = 46.9 percent | 117/288 = 40.6 percent | 36/288 = 12.5 percent | 288/288 |
| prompt guard | 129/288 = 44.8 percent | 122/288 = 42.4 percent | 37/288 = 12.8 percent | 288/288 |
| raw h14 prefill `alpha=0.50` | 167/288 = 58.0 percent | 76/288 = 26.4 percent | 45/288 = 15.6 percent | 288/288 |
| raw h14 prefill `alpha=1.00` | 194/272 = 71.3 percent | 57/272 = 21.0 percent | 21/272 = 7.7 percent | 272/288 |
| raw h14 on h13 prefill `alpha=0.50` | 174/288 = 60.4 percent | 69/288 = 24.0 percent | 45/288 = 15.6 percent | 288/288 |
| raw h14 on h13 prefill `alpha=1.00` | 189/260 = 72.7 percent | 50/260 = 19.2 percent | 21/260 = 8.1 percent | 260/288 |
| raw h14 wrong-token prefill `alpha=0.50` | 133/288 = 46.2 percent | 116/288 = 40.3 percent | 39/288 = 13.5 percent | 288/288 |
| raw random prefill `alpha=0.50` | 124/288 = 43.1 percent | 108/288 = 37.5 percent | 56/288 = 19.4 percent | 288/288 |
| raw random prefill `alpha=1.00` | 107/288 = 37.2 percent | 85/288 = 29.5 percent | 96/288 = 33.3 percent | 288/288 |
| residual h14 prefill `alpha=0.50` | 132/288 = 45.8 percent | 112/288 = 38.9 percent | 44/288 = 15.3 percent | 288/288 |
| residual h14 on h13 prefill `alpha=0.50` | 176/287 = 61.3 percent | 90/287 = 31.4 percent | 21/287 = 7.3 percent | 287/288 |

Wrong-hint rows:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline | 61/180 = 33.9 percent | 101/180 = 56.1 percent | 18/180 = 10.0 percent | 180/180 |
| prompt guard | 58/180 = 32.2 percent | 103/180 = 57.2 percent | 19/180 = 10.6 percent | 180/180 |
| raw h14 prefill `alpha=0.50` | 86/180 = 47.8 percent | 73/180 = 40.6 percent | 21/180 = 11.7 percent | 180/180 |
| raw h14 prefill `alpha=1.00` | 106/169 = 62.7 percent | 52/169 = 30.8 percent | 11/169 = 6.5 percent | 169/180 |
| raw h14 on h13 prefill `alpha=0.50` | 91/180 = 50.6 percent | 65/180 = 36.1 percent | 24/180 = 13.3 percent | 180/180 |
| raw h14 on h13 prefill `alpha=1.00` | 109/160 = 68.1 percent | 42/160 = 26.3 percent | 9/160 = 5.6 percent | 160/180 |
| raw h14 wrong-token prefill `alpha=0.50` | 59/180 = 32.8 percent | 100/180 = 55.6 percent | 21/180 = 11.7 percent | 180/180 |
| raw random prefill `alpha=0.50` | 56/180 = 31.1 percent | 92/180 = 51.1 percent | 32/180 = 17.8 percent | 180/180 |
| residual h14 prefill `alpha=0.50` | 59/180 = 32.8 percent | 99/180 = 55.0 percent | 22/180 = 12.2 percent | 180/180 |
| residual h14 on h13 prefill `alpha=0.50` | 95/179 = 53.1 percent | 76/179 = 42.5 percent | 8/179 = 4.5 percent | 179/180 |

Agreement-favored wrong-hint hard bin:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0/104 = 0.0 percent | 101/104 = 97.1 percent | 3/104 = 2.9 percent | 104/104 |
| prompt guard | 2/104 = 1.9 percent | 100/104 = 96.2 percent | 2/104 = 1.9 percent | 104/104 |
| raw h14 prefill `alpha=0.50` | 21/104 = 20.2 percent | 72/104 = 69.2 percent | 11/104 = 10.6 percent | 104/104 |
| raw h14 prefill `alpha=1.00` | 41/95 = 43.2 percent | 51/95 = 53.7 percent | 3/95 = 3.2 percent | 95/104 |
| raw h14 on h13 prefill `alpha=0.50` | 25/104 = 24.0 percent | 65/104 = 62.5 percent | 14/104 = 13.5 percent | 104/104 |
| raw h14 on h13 prefill `alpha=1.00` | 42/86 = 48.8 percent | 39/86 = 45.3 percent | 5/86 = 5.8 percent | 86/104 |
| raw h14 wrong-token prefill `alpha=0.50` | 0/104 = 0.0 percent | 99/104 = 95.2 percent | 5/104 = 4.8 percent | 104/104 |
| raw random prefill `alpha=0.50` | 3/104 = 2.9 percent | 91/104 = 87.5 percent | 10/104 = 9.6 percent | 104/104 |
| raw random prefill `alpha=1.00` | 14/104 = 13.5 percent | 65/104 = 62.5 percent | 25/104 = 24.0 percent | 104/104 |
| residual h14 prefill `alpha=0.50` | 1/104 = 1.0 percent | 93/104 = 89.4 percent | 10/104 = 9.6 percent | 104/104 |
| residual h14 on h13 prefill `alpha=0.50` | 32/103 = 31.1 percent | 69/103 = 67.0 percent | 2/103 = 1.9 percent | 103/104 |

## Side-Effect Rows

No-hint plus correct-hint rows:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline | 54/72 = 75.0 percent | 8/72 = 11.1 percent | 10/72 = 13.9 percent | 72/72 |
| prompt guard | 55/72 = 76.4 percent | 7/72 = 9.7 percent | 10/72 = 13.9 percent | 72/72 |
| raw h14 prefill `alpha=0.50` | 55/72 = 76.4 percent | 2/72 = 2.8 percent | 15/72 = 20.8 percent | 72/72 |
| raw h14 prefill `alpha=1.00` | 58/69 = 84.1 percent | 4/69 = 5.8 percent | 7/69 = 10.1 percent | 69/72 |
| raw h14 on h13 prefill `alpha=0.50` | 57/72 = 79.2 percent | 1/72 = 1.4 percent | 14/72 = 19.4 percent | 72/72 |
| raw h14 on h13 prefill `alpha=1.00` | 54/68 = 79.4 percent | 5/68 = 7.4 percent | 9/68 = 13.2 percent | 68/72 |
| raw h14 wrong-token prefill `alpha=0.50` | 54/72 = 75.0 percent | 8/72 = 11.1 percent | 10/72 = 13.9 percent | 72/72 |
| raw random prefill `alpha=0.50` | 49/72 = 68.1 percent | 9/72 = 12.5 percent | 14/72 = 19.4 percent | 72/72 |
| raw random prefill `alpha=1.00` | 27/72 = 37.5 percent | 14/72 = 19.4 percent | 31/72 = 43.1 percent | 72/72 |
| residual h14 prefill `alpha=0.50` | 49/72 = 68.1 percent | 7/72 = 9.7 percent | 16/72 = 22.2 percent | 72/72 |
| residual h14 on h13 prefill `alpha=0.50` | 54/72 = 75.0 percent | 9/72 = 12.5 percent | 9/72 = 12.5 percent | 72/72 |

## Split Results

Wrong-hint rows by split:

| Arm | Calibration Truth | Holdout Truth | Paraphrase Truth |
| --- | ---: | ---: | ---: |
| baseline | 24/60 | 19/60 | 18/60 |
| raw h14 prefill `alpha=0.50` | 31/60 | 29/60 | 26/60 |
| raw h14 prefill `alpha=1.00` | 39/60 | 36/55 | 31/54 |
| raw h14 on h13 prefill `alpha=0.50` | 33/60 | 31/60 | 27/60 |
| raw random prefill `alpha=0.50` | 22/60 | 17/60 | 17/60 |
| residual h14 prefill `alpha=0.50` | 22/60 | 20/60 | 17/60 |
| residual h14 on h13 prefill `alpha=0.50` | 30/60 | 35/59 | 30/60 |

## Interpretation

What V6 supports:

- raw h14 prefill steering is a real additive control surface;
- the hard-bin effect is not reproduced by the wrong-token arm;
- the hard-bin effect is not reproduced by matched random `alpha=0.50`;
- same-layer residualized h14 fails, matching the V3 residual-signal failure;
- prompt-only guarding is near baseline or worse.

What V6 rejects:

- a clean h14-local mechanism claim;
- another dense final-token or prompt-prefill direction sweep as the next Qwen3-0.6B step;
- high-dose random steering as a meaningful null, because it creates large generic answer artifacts;
- treating parseability-losing high-dose raw steering as a clean mechanism intervention.

The decisive negative is nearby-layer interchangeability. Raw h14 at h14 moved the hard bin from 0/104 to 21/104 truth-following at `alpha=0.50`; raw h14 applied at h13 moved it to 25/104. At `alpha=1.00`, raw h14 at h14 moved it to 41/95 parseable truth-following; raw h14 at h13 moved it to 42/86. The nearby arm is not a weak null. It is at least as active as the target arm.

The residual result sharpens the same point. Residual h14 at h14 does not move the hard bin meaningfully, but residual h14 applied at h13 moves it to 32/103. That pattern is not consistent with a simple h14 truth-versus-agreement mechanism.

## Verdict

Do not write a supported mechanism card from V6.

The supported claim is narrower:

> Qwen3-0.6B has a reproducible prompt-prefill activation control surface for MC-001 wrong-hint behavior, but V6 shows that the surface is not h14-local and is not yet a clean hidden mechanism.

The mechanism claim fails because:

- output margin remains the dominant diagnostic baseline from V2/V3;
- same-layer residualized h14 does not preserve the effect;
- nearby h13 application of the h14 vector is equally or more active than h14 application;
- high-dose steering loses parseability on a nontrivial number of rows;
- the current evidence explains a control surface, not a localized causal mechanism.

## Next Decision

Retrospective after V13: the cross-layer transport, answer-prefix, attention-source, head/localization, layer-band, input-mask semantics, and layout-parity audits have now been run. See [MC-001 Qwen3-0.6B Diagnostic Control-Surface Card](MC001_QWEN3_0P6B_DIAGNOSTIC_CONTROL_SURFACE.md). V7 showed that the raw h14 direction has nearly identical option-logit effects at h13 and h14. V8 showed that a separately trained answer-prefix h14 direction fails the mechanism bar. V9 showed that hint-source tokens causally matter. V10/V11 failed to localize that source effect cleanly, and V12/V13 showed the large effect is mostly prompt-source removal or recomputation.

Close Qwen3-0.6B dense-prefill, dense transport, and dense answer-prefix steering for MC-001 mechanism-card purposes.

Do not run another broad dense direction sweep on this model. At V6 time, any further Qwen3-0.6B iteration needed to answer a narrower source-path question:

1. identify which attention heads or layers route hint-answer tokens to answer tokens;
2. separate hinted-answer source tokens from non-answer hint wording;
3. test attention or MLP-path ablation rather than residual-stream addition.

That source-path line was tested through V13 and closed as control-only. Escalate to Qwen3-1.7B or an artifact-rich Gemma stack after accepting the Qwen3-0.6B result as a control-only failed-mechanism artifact.
