# MC001B Qwen3-1.7B Residualized Status

Status: complete. Raw control surface replicated; residualized mechanism test failed.

Date: 2026-06-30

## Artifacts

- runner: `code/mc001_qwen3_controlled_v3.py`
- preregistration: `research/prereg/MC001B_QWEN3_1P7B_RESIDUALIZED.md`
- prior controlled status: `research/cards/MC001B_QWEN3_1P7B_CONTROLLED_STATUS.md`
- manifest: `data/cards/MC001B/mc001b_qwen3_1p7b_controlled_v3_manifest.jsonl`
- manifest SHA256: `afd457c4f87233504edab8464b6c37b8745a1093c09483179ef6313d63427cb4`
- result: `results/cards/MC001B/mc001b_qwen3_1p7b_controlled_v3_20260630T114242.json`
- model: `Qwen/Qwen3-1.7B`
- run type: `mc001b_qwen3_1p7b_controlled_v3_residual_dose`
- hidden index: `21`
- logit-sweep alphas: `0`, `0.1`, `0.25`, `0.5`
- records: 384
- elapsed: 319.2 seconds

## Question

Does the Qwen3-1.7B `h21` control surface survive after output margin, prompt condition, correct answer letter, wrong answer letter, and baseline next-token answer are treated as nuisance variables?

## Gate Verdict

No.

The raw `h21 alpha=0.5` intervention remains a strong behavior control surface, but the nuisance-residualized `h21` direction loses the effect. The margin baseline remains AUC 1.000 on calibration, holdout, and paraphrase holdout.

Write this as:

> Qwen3-1.7B has a reproducible dense late-layer control surface for reducing wrong-hint agreement, but the effect is still output-margin and prompt-policy dominated. It is not a supported mechanism card.

## Baseline Substrate

Baseline generation was fully parseable.

| Metric | Count |
| --- | ---: |
| all truth-following | 287/384 |
| all user-agreement error | 72/384 |
| all other error | 25/384 |
| wrong-hint truth-following | 157/240 |
| wrong-hint user-agreement error | 68/240 |
| wrong-hint other error | 15/240 |

Candidate rows after filtering to truth-following versus user-agreement wrong-hint examples:

| Split | Candidates | Agreement-Favored | Ambiguous | Truth-Favored |
| --- | ---: | ---: | ---: | ---: |
| discovery | 69 | 14 | 0 | 55 |
| calibration | 72 | 21 | 0 | 51 |
| holdout | 66 | 15 | 0 | 51 |
| paraphrase holdout | 62 | 19 | 1 | 42 |

The agreement-favored bin remains the hard substrate: baseline had 0/60 truth-following and 57/60 user-agreement errors in that bin.

## Signature Gate

| Signal | Calibration AUC | Holdout AUC | Paraphrase AUC | Verdict |
| --- | ---: | ---: | ---: | --- |
| margin only | 1.000 | 1.000 | 1.000 | still dominates |
| condition only | 0.853 | 0.910 | 0.842 | strong prompt baseline |
| raw `h21` scalar | 0.897 | 0.969 | 0.868 | real but not dominant |
| residualized `h21` scalar | 0.779 | 0.839 | 0.808 | weakened after nuisance removal |
| margin + residualized `h21` | 1.000 | 1.000 | 1.000 | margin explains the classifier |

Direction diagnostics:

| Direction | Raw Norm | Train Rows | Train Agreement Rate | Notes |
| --- | ---: | ---: | ---: | --- |
| raw `h21` truth-minus-agreement | 160.967 | 69 | 20.3 percent | strong dense signature |
| residualized `h21` truth-minus-agreement | 14.058 | 69 | 20.3 percent | much smaller after nuisance regression |

The residualized direction was built after regressing out:

- intercept;
- standardized correct-minus-wrong logit margin;
- prompt condition;
- correct answer letter;
- wrong answer letter;
- baseline next-token answer.

Signature verdict: fail for mechanism promotion.

## Intervention Gate

Generation validation used 288 rows across calibration, holdout, and paraphrase holdout.

| Arm | Parseable | Overall Truth | Wrong-Hint Truth | Wrong-Hint Agreement | No/Correct Truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 288/288 | 209/288 | 113/180 | 54/180 | 65/72 |
| prompt guard | 288/288 | 240/288 | 143/180 | 22/180 | 65/72 |
| raw `h21 alpha=0.10` | 288/288 | 220/288 | 123/180 | 44/180 | 66/72 |
| raw `h21 alpha=0.25` | 288/288 | 226/288 | 133/180 | 35/180 | 62/72 |
| raw `h21 alpha=0.50` | 288/288 | 244/288 | 154/180 | 15/180 | 59/72 |
| raw random `alpha=0.50` | 288/288 | 204/288 | 105/180 | 66/180 | 66/72 |
| residual `h21 alpha=0.10` | 288/288 | 210/288 | 114/180 | 55/180 | 66/72 |
| residual `h21 alpha=0.25` | 288/288 | 211/288 | 116/180 | 56/180 | 63/72 |
| residual `h21 alpha=0.50` | 288/288 | 194/288 | 103/180 | 61/180 | 61/72 |
| residual random `alpha=0.50` | 288/288 | 211/288 | 114/180 | 53/180 | 66/72 |
| residual nearby `alpha=0.50` | 286/288 | 188/286 parseable | 96/178 parseable | 63/178 parseable | 64/72 |
| residual wrong-token `alpha=0.50` | 288/288 | 209/288 | 113/180 | 54/180 | 65/72 |

Raw `h21 alpha=0.50` is behaviorally real:

- wrong-hint truth-following improves from 113/180 to 154/180;
- wrong-hint agreement errors drop from 54/180 to 15/180;
- parseability stays 288/288;
- raw random control is worse than baseline;
- wrong-token control is exactly baseline.

But the mechanism test fails:

- prompt guard alone reaches 143/180 wrong-hint truth and preserves no/correct rows at 65/72;
- raw `h21 alpha=0.50` improves another 11 wrong-hint rows over prompt guard, but lowers no/correct truth to 59/72;
- residualized `h21` does not improve wrong-hint truth at any tested dose;
- residualized `h21 alpha=0.50` is worse than baseline and introduces answer-distribution shift toward `C`/`D`.

## Reliability Gate

Held-out split behavior:

| Arm | Calibration Truth | Holdout Truth | Paraphrase Truth |
| --- | ---: | ---: | ---: |
| baseline | 73/96 | 73/96 | 63/96 |
| prompt guard | 84/96 | 83/96 | 73/96 |
| raw `h21 alpha=0.50` | 90/96 | 78/96 | 76/96 |
| residual `h21 alpha=0.50` | 78/96 | 59/96 | 57/96 |
| residual wrong-token `alpha=0.50` | 73/96 | 73/96 | 63/96 |

Wrong-hint split behavior:

| Arm | Calibration | Holdout | Paraphrase |
| --- | ---: | ---: | ---: |
| baseline | 40/60 | 40/60 | 33/60 |
| prompt guard | 51/60 | 50/60 | 42/60 |
| raw `h21 alpha=0.50` | 57/60 | 49/60 | 48/60 |
| residual `h21 alpha=0.50` | 42/60 | 31/60 | 30/60 |
| residual wrong-token `alpha=0.50` | 40/60 | 40/60 | 33/60 |

Agreement-favored margin-bin behavior:

| Arm | Truth | Agreement | Other |
| --- | ---: | ---: | ---: |
| baseline | 0/60 | 57/60 | 3/60 |
| prompt guard | 30/60 | 25/60 | 5/60 |
| raw `h21 alpha=0.50` | 41/60 | 16/60 | 3/60 |
| residual `h21 alpha=0.50` | 16/60 | 42/60 | 2/60 |

Answer distributions:

| Arm | A | B | C | D | Unparseable |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 76 | 97 | 57 | 58 | 0 |
| prompt guard | 86 | 93 | 54 | 55 | 0 |
| raw `h21 alpha=0.50` | 79 | 75 | 72 | 62 | 0 |
| residual `h21 alpha=0.50` | 62 | 59 | 82 | 85 | 0 |
| residual nearby `alpha=0.50` | 50 | 58 | 87 | 91 | 2 |
| residual wrong-token `alpha=0.50` | 76 | 97 | 57 | 58 | 0 |

Reliability verdict:

- raw `h21` passes as a practical intervention surface;
- residualized `h21` fails as a mechanism rescue;
- prompt-only control is strong enough that user-facing policy text must be treated as a serious baseline;
- output-margin AUC 1.000 remains the dominant signature;
- answer-distribution movement in residualized arms argues against promoting the residualized direction.

## Comparison To Prior MC001B Controlled Run

The prior controlled run selected `h21 alpha=0.5` and improved wrong-hint truth-following from 125/180 to 152/180 on the original controlled manifest.

This residualized pass reproduces the raw control story on a balanced manifest:

- baseline wrong-hint truth: 113/180;
- raw `h21 alpha=0.50` wrong-hint truth: 154/180;
- residualized `h21 alpha=0.50` wrong-hint truth: 103/180.

That is strong evidence for a reusable dense control surface, but strong evidence against the current dense direction as an explanation.

## Next Step

Do not run another broad dense residual-direction sweep for MC001B.

The next meaningful branch should be one of:

1. move MC-001 to Gemma with sparse-feature/path tooling;
2. run a much narrower causal-path test on Qwen3-1.7B that conditions on prompt guard and margin bin, not another global direction;
3. create a diagnostic control-surface card for Qwen3-1.7B and explicitly separate product-useful steering from mechanism evidence.
