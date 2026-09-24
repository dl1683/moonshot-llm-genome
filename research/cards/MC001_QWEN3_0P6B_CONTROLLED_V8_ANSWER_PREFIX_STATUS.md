# MC-001 Qwen3-0.6B Controlled V8 Answer-Prefix Status

Status: answer-prefix dense-signature audit complete; not a supported mechanism card.

V8 tested the last cheap dense 0.6B route left after V7: train the truth-minus-agreement direction on an answer-prefix hidden state rather than the final prompt token, then intervene only during the answer-prefix prefill.

The result is negative. Answer-prefix prompting changes the behavior baseline, but the separately trained h14 answer-prefix direction does not produce a clean target effect. Raw interventions mostly trade agreement errors for unparseable outputs or generic degradation, nearby h13 remains comparable, and residual/random controls match or beat the target arms.

## Purpose

V7 closed final-prompt residual-stream transport as a mechanism path. V8 asked whether a hidden state after a neutral assistant prefix, `Answer: `, exposes a cleaner control point closer to answer production.

The test used generation validation, not only option logits.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v8_answer_prefix.py`
- result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v8_answer_prefix_20260629T223039.json`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v8_answer_prefix_manifest.jsonl`
- manifest SHA-256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- model: `Qwen/Qwen3-0.6B`
- answer prefix: `Answer: `
- rows: same 384-row controlled v2-v7 manifest
- validation rows: 288
- elapsed runtime: 348.8 seconds

Direction:

- raw h14 kind: answer-prefix truth-minus-agreement at the last prefix token;
- hidden index: `14`;
- discovery train rows: 68;
- discovery positive rate: 44.1 percent;
- raw norm: 4.008;
- median hidden norm: 43.010.

Residualized direction:

- hidden index: `14`;
- nuisance features: intercept, baseline margin, condition, correct answer letter, wrong answer letter, baseline next-token answer;
- nuisance rank: 13;
- raw norm after residualization: 1.213.

## Baseline Shift

The neutral answer prefix changed the behavioral substrate before any intervention.

Across all 384 rows, answer-prefix baseline produced:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline answer prefix | 228/384 = 59.4 percent | 120/384 = 31.3 percent | 36/384 = 9.4 percent | 384/384 |

Across the 288 validation rows:

| Filter | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| all validation | 170/288 = 59.0 percent | 86/288 = 29.9 percent | 32/288 = 11.1 percent | 288/288 |
| wrong-hint rows | 83/180 = 46.1 percent | 80/180 = 44.4 percent | 17/180 = 9.4 percent | 180/180 |
| no-hint plus correct-hint rows | 58/72 = 80.6 percent | 3/72 = 4.2 percent | 11/72 = 15.3 percent | 72/72 |

This baseline is already more truth-following than the earlier no-prefix generation baseline, so V8 cannot be interpreted as a continuation of the same hard-bin substrate without its own controls.

## Main Results

All validation rows:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline answer prefix | 170/288 = 59.0 percent | 86/288 = 29.9 percent | 32/288 = 11.1 percent | 288/288 |
| raw h14 prefix at h14 `alpha=0.50` | 154/259 = 59.5 percent | 78/259 = 30.1 percent | 27/259 = 10.4 percent | 259/288 |
| raw h14 prefix at h13 `alpha=0.50` | 138/233 = 59.2 percent | 66/233 = 28.3 percent | 29/233 = 12.4 percent | 233/288 |
| raw h14 prefix at h14 `alpha=1.00` | 138/231 = 59.7 percent | 66/231 = 28.6 percent | 27/231 = 11.7 percent | 231/288 |
| raw h14 prefix at h13 `alpha=1.00` | 120/195 = 61.5 percent | 48/195 = 24.6 percent | 27/195 = 13.8 percent | 195/288 |
| raw h14 wrong-token `alpha=0.50` | 170/288 = 59.0 percent | 85/288 = 29.5 percent | 33/288 = 11.5 percent | 288/288 |
| raw random at h14 `alpha=0.50` | 164/288 = 56.9 percent | 91/288 = 31.6 percent | 33/288 = 11.5 percent | 288/288 |
| residual h14 at h14 `alpha=0.50` | 166/276 = 60.1 percent | 82/276 = 29.7 percent | 28/276 = 10.1 percent | 276/288 |
| residual h14 at h13 `alpha=0.50` | 166/276 = 60.1 percent | 80/276 = 29.0 percent | 30/276 = 10.9 percent | 276/288 |
| residual random at h14 `alpha=0.50` | 171/288 = 59.4 percent | 90/288 = 31.3 percent | 27/288 = 9.4 percent | 288/288 |

Wrong-hint rows:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline answer prefix | 83/180 = 46.1 percent | 80/180 = 44.4 percent | 17/180 = 9.4 percent | 180/180 |
| raw h14 prefix at h14 `alpha=0.50` | 70/153 = 45.8 percent | 69/153 = 45.1 percent | 14/153 = 9.2 percent | 153/180 |
| raw h14 prefix at h13 `alpha=0.50` | 69/143 = 48.3 percent | 59/143 = 41.3 percent | 15/143 = 10.5 percent | 143/180 |
| raw h14 prefix at h14 `alpha=1.00` | 65/136 = 47.8 percent | 57/136 = 41.9 percent | 14/136 = 10.3 percent | 136/180 |
| raw h14 prefix at h13 `alpha=1.00` | 58/114 = 50.9 percent | 42/114 = 36.8 percent | 14/114 = 12.3 percent | 114/180 |
| raw h14 wrong-token `alpha=0.50` | 83/180 = 46.1 percent | 79/180 = 43.9 percent | 18/180 = 10.0 percent | 180/180 |
| raw random at h14 `alpha=0.50` | 78/180 = 43.3 percent | 84/180 = 46.7 percent | 18/180 = 10.0 percent | 180/180 |
| residual h14 at h14 `alpha=0.50` | 79/168 = 47.0 percent | 73/168 = 43.5 percent | 16/168 = 9.5 percent | 168/180 |
| residual h14 at h13 `alpha=0.50` | 79/168 = 47.0 percent | 72/168 = 42.9 percent | 17/168 = 10.1 percent | 168/180 |
| residual random at h14 `alpha=0.50` | 82/180 = 45.6 percent | 85/180 = 47.2 percent | 13/180 = 7.2 percent | 180/180 |

Answer-prefix agreement-favored wrong-hint rows:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline answer prefix | 7/95 = 7.4 percent | 79/95 = 83.2 percent | 9/95 = 9.5 percent | 95/95 |
| raw h14 prefix at h14 `alpha=0.50` | 9/81 = 11.1 percent | 67/81 = 82.7 percent | 5/81 = 6.2 percent | 81/95 |
| raw h14 prefix at h13 `alpha=0.50` | 8/72 = 11.1 percent | 58/72 = 80.6 percent | 6/72 = 8.3 percent | 72/95 |
| raw h14 prefix at h14 `alpha=1.00` | 5/65 = 7.7 percent | 54/65 = 83.1 percent | 6/65 = 9.2 percent | 65/95 |
| raw h14 prefix at h13 `alpha=1.00` | 7/52 = 13.5 percent | 41/52 = 78.8 percent | 4/52 = 7.7 percent | 52/95 |
| raw h14 wrong-token `alpha=0.50` | 7/95 = 7.4 percent | 78/95 = 82.1 percent | 10/95 = 10.5 percent | 95/95 |
| raw random at h14 `alpha=0.50` | 6/95 = 6.3 percent | 80/95 = 84.2 percent | 9/95 = 9.5 percent | 95/95 |
| residual h14 at h14 `alpha=0.50` | 7/87 = 8.0 percent | 72/87 = 82.8 percent | 8/87 = 9.2 percent | 87/95 |
| residual h14 at h13 `alpha=0.50` | 7/87 = 8.0 percent | 71/87 = 81.6 percent | 9/87 = 10.3 percent | 87/95 |
| residual random at h14 `alpha=0.50` | 10/95 = 10.5 percent | 79/95 = 83.2 percent | 6/95 = 6.3 percent | 95/95 |

No-hint plus correct-hint side-effect rows:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline answer prefix | 58/72 = 80.6 percent | 3/72 = 4.2 percent | 11/72 = 15.3 percent | 72/72 |
| raw h14 prefix at h14 `alpha=0.50` | 56/70 = 80.0 percent | 5/70 = 7.1 percent | 9/70 = 12.9 percent | 70/72 |
| raw h14 prefix at h13 `alpha=0.50` | 45/58 = 77.6 percent | 3/58 = 5.2 percent | 10/58 = 17.2 percent | 58/72 |
| raw h14 prefix at h14 `alpha=1.00` | 48/62 = 77.4 percent | 5/62 = 8.1 percent | 9/62 = 14.5 percent | 62/72 |
| raw h14 prefix at h13 `alpha=1.00` | 42/54 = 77.8 percent | 3/54 = 5.6 percent | 9/54 = 16.7 percent | 54/72 |
| raw h14 wrong-token `alpha=0.50` | 58/72 = 80.6 percent | 3/72 = 4.2 percent | 11/72 = 15.3 percent | 72/72 |
| raw random at h14 `alpha=0.50` | 58/72 = 80.6 percent | 3/72 = 4.2 percent | 11/72 = 15.3 percent | 72/72 |
| residual h14 at h14 `alpha=0.50` | 59/72 = 81.9 percent | 5/72 = 6.9 percent | 8/72 = 11.1 percent | 72/72 |
| residual h14 at h13 `alpha=0.50` | 59/72 = 81.9 percent | 4/72 = 5.6 percent | 9/72 = 12.5 percent | 72/72 |
| residual random at h14 `alpha=0.50` | 60/72 = 83.3 percent | 2/72 = 2.8 percent | 10/72 = 13.9 percent | 72/72 |

## Interpretation

What V8 supports:

- answer-prefix prompting itself is a meaningful behavioral control on this substrate;
- wrong-token application is near baseline, so the implementation is position-sensitive;
- raw h14 answer-prefix directions can reduce agreement counts, but mostly by reducing parseability rather than increasing absolute truth-following;
- h13 and h14 answer-prefix effects remain too similar to support target-layer locality.

What V8 rejects:

- a clean answer-prefix h14 dense mechanism;
- treating parseability-normalized truth-rate gains as sufficient when absolute truth counts fall;
- treating the h13 arm as a weak nearby-layer null;
- treating residualized h14 as a rescue, because residual h14 and residual random controls are as good or better on the main metrics.

The agreement-favored answer-prefix bin is the decisive negative. Baseline truth-following is 7/95. The target raw h14 arm only reaches 9/81 parseable at `alpha=0.50`, while residual random reaches 10/95 with full parseability. The target arm does not beat controls with bounded side effects.

The all-wrong-hint result is also negative. Baseline is 83/180 truth-following with full parseability. Raw h14 at h14 `alpha=0.50` produces 70/153 parseable truth-following, and raw h14 at h13 produces 69/143. Higher dose increases parse loss. This is not a useful control surface.

## Verdict

Do not write a supported mechanism card from V8.

The supported claim is narrower:

> On Qwen3-0.6B MC-001 prompts, a neutral `Answer: ` prefix changes baseline truth-versus-agreement behavior, but a separately trained h14 answer-prefix dense direction does not produce a clean or local intervention.

The mechanism claim fails because:

- absolute truth-following does not improve on all wrong-hint rows;
- hard-bin target gains are tiny and matched by residual/random controls;
- h13 remains comparable to h14;
- raw interventions reduce parseability at useful doses;
- residualized h14 does not beat residual/random controls.

## Next Decision

Close dense answer-prefix steering on Qwen3-0.6B for MC-001 mechanism-card purposes.

At V8 time, the remaining Qwen3-0.6B-compatible work was no longer broad dense residual-stream addition. The next local experiment needed to be one of:

1. true path-local attribution from hint tokens to answer tokens;
2. attention-head or MLP-path ablation rather than residual-stream addition;
3. model escalation to Qwen3-1.7B or an artifact-rich Gemma stack after accepting this 0.6B control-only result.

## Retrospective

V8 closed the answer-prefix branch that V7 left open. Retrospective after V13: the next local experiment found strong hint-source dependence, later localization attempts did not find a compact head/layer/band mechanism, and input-mask plus layout-parity checks showed broad prompt-source removal or recomputation. The future path is escalation, not more broad Qwen3-0.6B dense steering.
