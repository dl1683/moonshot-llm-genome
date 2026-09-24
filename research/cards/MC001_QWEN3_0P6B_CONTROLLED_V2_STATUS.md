# MC-001 Qwen3-0.6B Controlled V2 Status

Status: output-margin-conditioned controlled pass complete; not a supported mechanism card.

The useful result is a sharper negative control result. Qwen3-0.6B still exposes a causal surface for reducing wrong-hint agreement, but the current dense direction does not survive the mechanism-card standard because output margin remains a perfect diagnostic baseline and matched/nearby controls reveal large generic answer-token side effects.

## Purpose

V2 asked whether the first controlled result still had hidden-state value after the answer-logit margin was promoted to a first-class baseline.

Changes from the first controlled pass:

- fresh balanced factual multiple-choice manifest;
- correct letters balanced across `A`, `B`, `C`, and `D` within each split;
- wrong-hint letters balanced across `A`, `B`, `C`, and `D` within each split;
- baseline answer-logit margin recorded for every row;
- residual probe tables with margin, condition, hidden direction, and combined baselines;
- prompt-only guard baseline included in generation validation;
- nearby-layer control replaces the earlier parse-collapsing wrong-layer null.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v2.py`
- result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v2_20260629T203703.json`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v2_manifest.jsonl`
- manifest SHA-256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- model: `Qwen/Qwen3-0.6B`
- rows: 384 total, 96 per split across discovery, calibration, holdout, and paraphrase holdout
- hidden indices tested: `7`, `14`
- calibration alphas tested: `0.0`, `0.5`, `1.0`
- max new tokens: `24`
- elapsed runtime: 1017.5 seconds

## Baseline Behavior

Across all 384 manifest rows, baseline generation was fully parseable:

- truth-following: 182/384 = 47.4 percent
- user-agreement error: 156/384 = 40.6 percent
- other error: 46/384 = 12.0 percent

Wrong-hint rows remain a valid substrate:

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

Most candidate rows were not ambiguous under the output margin:

| Split | Agreement-Favored | Ambiguous | Truth-Favored |
| --- | ---: | ---: | ---: |
| discovery | 38 | 4 | 24 |
| calibration | 39 | 2 | 28 |
| holdout | 35 | 2 | 25 |
| paraphrase holdout | 35 | 1 | 23 |

## Signature Gate

The signature gate fails again because answer-logit margin still dominates.

| Diagnostic | Calibration AUC | Holdout AUC | Paraphrase Holdout AUC |
| --- | ---: | ---: | ---: |
| margin only | 1.000 | 1.000 | 1.000 |
| condition only | 0.873 | 0.928 | 0.921 |
| margin + condition | 0.992 | 0.997 | 0.992 |
| h7 direction scalar | 0.910 | 0.925 | 0.938 |
| h14 direction scalar | 0.875 | 0.954 | 0.942 |
| margin + h7 direction | 0.996 | 0.995 | 0.993 |
| margin + h14 direction | 0.991 | 0.997 | 0.994 |
| margin + condition + h7 direction | 0.984 | 0.996 | 0.993 |
| margin + condition + h14 direction | 0.986 | 0.998 | 0.993 |

Hidden scalars are not useless diagnostics, but they do not beat the output baseline and do not add a clean residual signature under the current test. A mechanism claim would be overclaiming.

Verdict on signature gate: fail.

## Intervention Gate

Selected intervention:

- key: `h14_alpha1.0`
- hidden index: `14`
- alpha: `1.0`
- calibration selection score: `-0.1076`

The negative selection score matters. The selected cell was the best cell under the v2 score, but the score itself says the intervention is not clean on calibration.

Wrong-hint truth-following by generation arm:

| Arm | Calibration | Holdout | Paraphrase Holdout | Parseability Notes |
| --- | ---: | ---: | ---: | --- |
| baseline | 24/60 = 40.0 percent | 19/60 = 31.7 percent | 18/60 = 30.0 percent | 100 percent parseable |
| target h14 +1.0 | 23/58 = 39.7 percent | 36/60 = 60.0 percent | 31/55 = 56.4 percent | 277/288 overall parseable |
| nearby layer | 40/60 = 66.7 percent | 32/59 = 54.2 percent | 28/58 = 48.3 percent | 285/288 overall parseable |
| random matched norm | 15/60 = 25.0 percent | 15/60 = 25.0 percent | 15/60 = 25.0 percent | 100 percent parseable but dominated by other errors |
| wrong token | 23/60 = 38.3 percent | 19/60 = 31.7 percent | 18/60 = 30.0 percent | near-baseline |
| prompt guard | 23/60 = 38.3 percent | 18/60 = 30.0 percent | 17/60 = 28.3 percent | near-baseline |

The target improves holdout and paraphrase wrong-hint truth-following, but calibration does not improve. The nearby-layer control also produces large movement, so the target effect is not layer-local enough. The random matched direction produces a severe generic answer bias rather than a meaningful truth-vs-agreement effect.

No-hint plus correct-hint side effects:

| Arm | Calibration | Holdout | Paraphrase Holdout |
| --- | ---: | ---: | ---: |
| baseline | 19/24 = 79.2 percent | 18/24 = 75.0 percent | 17/24 = 70.8 percent |
| target h14 +1.0 | 14/23 = 60.9 percent | 19/24 = 79.2 percent | 18/22 = 81.8 percent |
| nearby layer | 14/24 = 58.3 percent | 13/24 = 54.2 percent | 15/24 = 62.5 percent |
| random matched norm | 6/24 = 25.0 percent | 6/24 = 25.0 percent | 6/24 = 25.0 percent |
| wrong token | 19/24 = 79.2 percent | 18/24 = 75.0 percent | 17/24 = 70.8 percent |
| prompt guard | 19/24 = 79.2 percent | 18/24 = 75.0 percent | 18/24 = 75.0 percent |

The target does not have a catastrophic no/correct side effect overall, but it has calibration degradation and unparseables. The nearby-layer and random controls are too active to count as clean nulls.

Verdict on intervention gate: fail for mechanism-card promotion; useful as control-surface evidence.

## Reliability Gate

Reliability failures:

- answer-logit margin still predicts agreement-vs-truth with AUC 1.0 on calibration, holdout, and paraphrase holdout;
- the selected intervention has a negative calibration selection score;
- nearby-layer control moves behavior strongly, so locality is weak;
- random matched direction collapses into a broad answer-token artifact: 284/288 generated answers were `A`;
- target intervention also has answer-token skew: 140/288 generated answers were `A`;
- target parseability falls to 277/288;
- calibration wrong-hint truth does not improve under the selected target;
- prompt-only guard does not solve the behavior and mostly tracks baseline.

What survives:

- Qwen3-0.6B remains cheap and informative for iterative controls;
- the factual-ladder substrate is measurable and parseable;
- wrong-token control remains near-baseline, so final-token timing matters;
- target h14 steering changes holdout/paraphrase behavior in the intended direction;
- the next failure mode is now concrete: dose and answer-letter artifact control, not broad prompt design.

## Verdict

Do not write a supported MC-001 mechanism card from v2.

Write the result as:

> An output-margin-conditioned Qwen3-0.6B pass found that h14 steering can reduce wrong-hint agreement on holdout and paraphrase-holdout rows, but the result remains a failed-controls/control-surface finding because output margin perfectly predicts the behavior and matched/nearby controls expose generic answer-token side effects.

## Next Iteration

Continue with Qwen3-0.6B for one narrower pass before escalating.

V3 should:

1. validate lower h14 doses, especially `alpha=0.5`, because calibration sweep gave the same wrong-hint truth count as `alpha=1.0` with much better no/correct accuracy;
2. add smaller alphas such as `0.1` and `0.25` to check whether the causal surface appears before answer-token collapse;
3. build an answer-letter and output-margin residualized direction, then compare it against the raw direction;
4. report parsed-answer distributions by arm as a required control metric;
5. keep prompt guard, wrong-token, nearby-layer, and matched-random controls;
6. promote only if the target beats the best clean null, preserves no/correct rows, and adds value beyond the margin baseline.

V3 has now run and did fail the margin/residual controls. See [MC-001 Qwen3-0.6B Controlled V3 Status](MC001_QWEN3_0P6B_CONTROLLED_V3_STATUS.md). The honest next artifact is a diagnostic/control-surface card plus a method shift, not another broad hidden-direction claim.
