# MC-001 Qwen3-0.6B Controlled V3 Status

Status: lower-dose and residualized controlled pass complete; not a supported mechanism card.

The useful result is now sharper: raw h14 steering at `alpha=0.50` can move Qwen3-0.6B away from wrong-hint agreement without the severe parse collapse seen at higher dose, but the effect is not a clean hidden mechanism. Output margin remains the dominant diagnostic baseline, answer-letter artifacts remain visible in null controls, and the answer-letter/output-margin residualized h14 direction mostly loses the effect.

## Purpose

V3 asked whether the v2 failure was caused by dose and answer-token entanglement rather than by the absence of a usable hidden mechanism.

Changes from v2:

- kept the balanced v2 factual multiple-choice manifest;
- tested lower raw h14 doses: `0.10`, `0.25`, and `0.50`;
- built a residualized h14 direction after regressing out output margin, prompt condition, correct answer letter, wrong-hint letter, and baseline next-token answer;
- compared raw, residualized, nearby-layer, random matched, wrong-token, and prompt-guard arms;
- reported answer distributions by arm as a required control metric.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v3.py`
- result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v3_20260629T205927.json`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v3_manifest.jsonl`
- manifest SHA-256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- model: `Qwen/Qwen3-0.6B`
- rows: 384 total, 96 per split across discovery, calibration, holdout, and paraphrase holdout
- hidden index tested: `14`
- raw/residualized alphas tested: `0.0`, `0.10`, `0.25`, `0.50`
- max new tokens: `24`
- elapsed runtime: 855.3 seconds

## Baseline Behavior

Across all 384 manifest rows, baseline generation was fully parseable:

- truth-following: 182/384 = 47.4 percent
- user-agreement error: 156/384 = 40.6 percent
- other error: 46/384 = 12.0 percent

Wrong-hint rows remain a valid measurable substrate:

| Split | Baseline Wrong-Hint Truth | Agreement Error | Other Error | Parseable |
| --- | ---: | ---: | ---: | ---: |
| calibration | 24/60 = 40.0 percent | 34/60 = 56.7 percent | 2/60 = 3.3 percent | 60/60 |
| holdout | 19/60 = 31.7 percent | 34/60 = 56.7 percent | 7/60 = 11.7 percent | 60/60 |
| paraphrase holdout | 18/60 = 30.0 percent | 33/60 = 55.0 percent | 9/60 = 15.0 percent | 60/60 |

Candidate rows after filtering to truth-following versus user-agreement wrong-hint examples:

- discovery: 66
- calibration: 69
- holdout: 62
- paraphrase holdout: 59

Candidate margin-bin counts:

| Split | Agreement-Favored | Ambiguous | Truth-Favored |
| --- | ---: | ---: | ---: |
| discovery | 38 | 4 | 24 |
| calibration | 39 | 2 | 28 |
| holdout | 35 | 2 | 25 |
| paraphrase holdout | 35 | 1 | 23 |

## Signature Gate

The signature gate fails. Output margin remains perfect, and residualizing against output/answer-letter nuisance features removes most of the hidden-direction signal.

| Diagnostic | Calibration AUC | Holdout AUC | Paraphrase Holdout AUC |
| --- | ---: | ---: | ---: |
| output margin only | 1.000 | 1.000 | 1.000 |
| condition only | 0.873 | 0.928 | 0.921 |
| raw h14 scalar | 0.875 | 0.954 | 0.942 |
| residualized h14 scalar | 0.754 | 0.600 | 0.677 |
| margin + raw h14 | 0.991 | 0.997 | 0.994 |
| margin + residualized h14 | 0.998 | 1.000 | 1.000 |

Direction diagnostics:

| Direction | Raw Norm | Train Rows | Train Positive Rate | Notes |
| --- | ---: | ---: | ---: | --- |
| raw h14 truth-minus-agreement | 3.112 | 66 | 57.6 percent | predictive, but entangled with output/answer features |
| residualized h14 truth-minus-agreement | 0.242 | 66 | 57.6 percent | weak after nuisance removal |

Verdict on signature gate: fail.

## Intervention Gate

Raw h14 dose `0.50` is the strongest v3 arm, but it is still not clean enough for card promotion.

Wrong-hint truth-following by generation arm:

| Arm | Calibration | Holdout | Paraphrase Holdout | Main Control Note |
| --- | ---: | ---: | ---: | --- |
| baseline | 24/60 = 40.0 percent | 19/60 = 31.7 percent | 18/60 = 30.0 percent | reference |
| prompt guard | 23/60 = 38.3 percent | 18/60 = 30.0 percent | 17/60 = 28.3 percent | prompt-only does not solve it |
| raw h14 `alpha=0.10` | 25/60 = 41.7 percent | 19/60 = 31.7 percent | 19/60 = 31.7 percent | near baseline |
| raw h14 `alpha=0.25` | 30/60 = 50.0 percent | 22/60 = 36.7 percent | 19/60 = 31.7 percent | weak holdout/paraphrase |
| raw h14 `alpha=0.50` | 36/60 = 60.0 percent | 36/60 = 60.0 percent | 30/60 = 50.0 percent | real effect, not clean mechanism |
| raw random `alpha=0.50` | 18/60 = 30.0 percent | 14/60 = 23.3 percent | 12/60 = 20.0 percent | severe answer-token artifact |
| residual h14 `alpha=0.10` | 24/60 = 40.0 percent | 19/60 = 31.7 percent | 19/60 = 31.7 percent | near baseline |
| residual h14 `alpha=0.25` | 23/60 = 38.3 percent | 19/60 = 31.7 percent | 19/60 = 31.7 percent | near baseline |
| residual h14 `alpha=0.50` | 21/60 = 35.0 percent | 20/60 = 33.3 percent | 17/60 = 28.3 percent | no rescue |
| residual nearby `alpha=0.50` | 21/56 = 37.5 percent | 30/59 = 50.8 percent | 21/56 = 37.5 percent | active and partly unparseable |
| residual random `alpha=0.50` | 19/60 = 31.7 percent | 18/60 = 30.0 percent | 16/60 = 26.7 percent | answer-token artifact remains |
| residual wrong-token `alpha=0.50` | 20/60 = 33.3 percent | 19/60 = 31.7 percent | 19/60 = 31.7 percent | near baseline |

No-hint plus correct-hint truth-following:

| Arm | Calibration | Holdout | Paraphrase Holdout |
| --- | ---: | ---: | ---: |
| baseline | 19/24 = 79.2 percent | 18/24 = 75.0 percent | 17/24 = 70.8 percent |
| prompt guard | 19/24 = 79.2 percent | 18/24 = 75.0 percent | 18/24 = 75.0 percent |
| raw h14 `alpha=0.10` | 20/24 = 83.3 percent | 18/24 = 75.0 percent | 18/24 = 75.0 percent |
| raw h14 `alpha=0.25` | 20/24 = 83.3 percent | 18/24 = 75.0 percent | 18/24 = 75.0 percent |
| raw h14 `alpha=0.50` | 21/24 = 87.5 percent | 18/24 = 75.0 percent | 18/24 = 75.0 percent |
| raw random `alpha=0.50` | 14/24 = 58.3 percent | 11/24 = 45.8 percent | 8/24 = 33.3 percent |
| residual h14 `alpha=0.10` | 19/24 = 79.2 percent | 18/24 = 75.0 percent | 18/24 = 75.0 percent |
| residual h14 `alpha=0.25` | 18/24 = 75.0 percent | 18/24 = 75.0 percent | 18/24 = 75.0 percent |
| residual h14 `alpha=0.50` | 17/24 = 70.8 percent | 17/24 = 70.8 percent | 15/24 = 62.5 percent |
| residual nearby `alpha=0.50` | 17/23 = 73.9 percent | 16/24 = 66.7 percent | 15/22 = 68.2 percent |
| residual random `alpha=0.50` | 18/24 = 75.0 percent | 17/24 = 70.8 percent | 13/24 = 54.2 percent |
| residual wrong-token `alpha=0.50` | 19/24 = 79.2 percent | 18/24 = 75.0 percent | 17/24 = 70.8 percent |

Answer distributions by arm:

| Arm | A | B | C | D | Unparseable |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 120 | 75 | 48 | 45 | 0 |
| prompt guard | 123 | 73 | 48 | 44 | 0 |
| raw h14 `alpha=0.10` | 110 | 79 | 50 | 49 | 0 |
| raw h14 `alpha=0.25` | 98 | 83 | 59 | 48 | 0 |
| raw h14 `alpha=0.50` | 83 | 80 | 74 | 51 | 0 |
| raw random `alpha=0.50` | 205 | 49 | 16 | 18 | 0 |
| residual h14 `alpha=0.10` | 117 | 75 | 49 | 47 | 0 |
| residual h14 `alpha=0.25` | 114 | 75 | 51 | 48 | 0 |
| residual h14 `alpha=0.50` | 124 | 65 | 52 | 47 | 0 |
| residual nearby `alpha=0.50` | 144 | 42 | 57 | 32 | 13 |
| residual random `alpha=0.50` | 162 | 54 | 36 | 36 | 0 |
| residual wrong-token `alpha=0.50` | 119 | 77 | 46 | 46 | 0 |

Verdict on intervention gate: fail for mechanism-card promotion; useful as a reproducible control-surface result.

## Reliability Gate

What improved:

- raw h14 `alpha=0.50` raises wrong-hint truth-following from 61/180 to 102/180 overall;
- the effect appears on calibration, holdout, and paraphrase holdout;
- answer distribution is much less degenerate than the v2 `alpha=1.0` arm;
- no/correct rows are not catastrophically damaged at raw h14 `alpha=0.50`;
- wrong-token control is near-baseline, so the intervention still depends on the final-token timing.

What fails:

- output margin remains a perfect agreement-vs-truth diagnostic with AUC 1.0 on calibration, holdout, and paraphrase holdout;
- raw h14 is not a residual hidden signal under the nuisance controls tested here;
- residualized h14 loses the behavioral effect;
- raw random and residual random controls still create answer-token artifacts;
- residual nearby-layer control is active and partly unparseable;
- prompt-only guard remains near-baseline, so the practical control is not solved by instruction text;
- the raw effect is concentrated in a global activation direction that appears entangled with output-token geometry.

The hardest diagnostic is the agreement-favored output-margin bin. Baseline wrong-hint rows in that bin have 0/104 truth-following. Raw h14 `alpha=0.50` moves that to 30/104, so the intervention is not merely selecting already-truth-favored rows. But the same run shows the residualized direction failing and controls remaining active, so the correct conclusion is "causal surface, not supported mechanism."

## Verdict

Do not write a supported MC-001 mechanism card from v3.

Write the result as:

> A lower-dose Qwen3-0.6B pass found that raw h14 steering can causally reduce wrong-hint agreement, including in output-margin agreement-favored rows, but the result remains a failed-controls/control-surface finding because output margin dominates diagnostics, residualizing answer/output nuisances removes the hidden-direction effect, and matched/nearby controls remain active.

## Next Iteration

Do not run another broad dense-direction sweep as v4.

The next useful artifact is a diagnostic/control-surface card for Qwen3-0.6B plus one method shift if continuing:

1. condition future interventions inside the agreement-favored margin bin rather than averaging over all wrong-hint rows;
2. pair examples by correct answer, wrong answer, and output margin before estimating directions;
3. try activation patching or path-local interventions instead of one global dense direction;
4. keep answer distribution and margin-bin movement as required acceptance tables;
5. promote only if a target beats output margin diagnostically or proves a causal path that matched controls cannot reproduce.

If the next pass still cannot beat these controls, Qwen3-0.6B should be written up as a diagnostic/control-surface card and the mechanism-card search should move to a method with cleaner locality or to a larger comparison model.
