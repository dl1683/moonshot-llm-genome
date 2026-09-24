# MC001B Qwen3-1.7B Controlled Status

Status: complete. Useful control surface found; mechanism not promoted.

Date: 2026-06-30

## Artifacts

- runner: `code/mc001_qwen3_controlled.py`
- preregistration: `research/prereg/MC001B_QWEN3_1P7B_DISCOVERY.md`
- prior smoke: `research/cards/MC001B_QWEN3_1P7B_SMOKE_STATUS.md`
- manifest: `data/cards/MC001B/mc001b_qwen3_1p7b_controlled_manifest.jsonl`
- manifest SHA256: `9b6434bec2d39febf921b00f1ab49d5c6fe87aa289637005085715495fb614d6`
- result: `results/cards/MC001B/mc001b_qwen3_1p7b_controlled_20260630T112935.json`
- model: `Qwen/Qwen3-1.7B`
- run type: `mc001b_qwen3_1p7b_controlled_discovery_calibration`
- hidden indices: `7`, `14`, `21`
- alphas: `-1`, `-0.5`, `0`, `0.5`, `1`
- records: 384
- elapsed: 429.2 seconds

## Question

Does Qwen3-1.7B turn the MC-001 truth-versus-wrong-hint behavior into a cleaner mechanism-card substrate than Qwen3-0.6B?

## Gate Verdict

No mechanism promotion.

The run found a real and useful additive control surface at `h21 alpha=0.5`, but the signature gate failed the preregistered standard because the output/logit margin baseline matched or beat the hidden probes.

This is a `control_surface_supported` / `mechanism_not_supported` result.

## Signature Gate

Probe target: predict user-agreement errors versus truth-following on controlled calibration, holdout, and paraphrase holdout rows.

| Signal | Calibration AUC | Holdout AUC | Paraphrase AUC | Verdict |
| --- | ---: | ---: | ---: | --- |
| prompt-condition baseline | 0.879 | 0.942 | 0.915 | strong surface baseline |
| next-token logit margin baseline | 1.000 | 1.000 | 1.000 | beats hidden probes |
| hidden layer 7 | 0.829 | 0.949 | 0.850 | real signal, not dominant |
| hidden layer 14 | 0.901 | 0.903 | 0.855 | real signal, not dominant |
| hidden layer 21 | 0.764 | 0.926 | 0.974 | strong held-out signal, calibration weaker |
| label-permuted hidden controls | 0.472-0.796 | 0.478-0.550 | 0.508-0.604 | mostly demoted |

Interpretation:

- hidden states carry behavior-relevant information;
- the signal is not a clean internal mechanism signature because the logit-margin baseline is essentially perfect;
- the selected `h21` direction is better treated as a late residual control handle than a localized explanation.

## Intervention Gate

Selected intervention:

| Field | Value |
| --- | --- |
| key | `h21_alpha0.5` |
| hidden index | `21` |
| alpha | `0.5` |
| calibration selection score | 0.892 |

Generation validation used 288 rows across calibration, holdout, and paraphrase holdout. The target intervention improved truth-following under wrong-hint pressure while preserving parseability.

| Arm | Parseable | Overall Truth | Wrong-Hint Truth | Wrong-Hint Agreement | No/Correct Truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 288/288 | 227/288 | 125/180 | 52/180 | 68/72 |
| random matched | 288/288 | 229/288 | 127/180 | 49/180 | 68/72 |
| sign flip | 288/288 | 198/288 | 99/180 | 77/180 | 67/72 |
| target `h21 alpha=0.5` | 288/288 | 251/288 | 152/180 | 24/180 | 65/72 |
| wrong layer | 175/288 | 59/175 parseable | 38/106 parseable | 23/106 parseable | 14/45 parseable |
| wrong token | 288/288 | 227/288 | 125/180 | 52/180 | 68/72 |

The target arm is behaviorally meaningful:

- wrong-hint truth-following improves from 125/180 to 152/180;
- wrong-hint user-agreement errors drop from 52/180 to 24/180;
- random-matched and wrong-token controls do not reproduce the effect;
- sign-flip moves in the predicted harmful direction;
- parseability remains 288/288 for the target arm.

## Reliability Gate

Held-out split behavior:

| Arm | Calibration Truth | Holdout Truth | Paraphrase Truth |
| --- | ---: | ---: | ---: |
| baseline | 66/96 | 81/96 | 80/96 |
| random matched | 68/96 | 81/96 | 80/96 |
| sign flip | 55/96 | 72/96 | 71/96 |
| target `h21 alpha=0.5` | 74/96 | 90/96 | 87/96 |
| wrong token | 66/96 | 81/96 | 80/96 |

Condition-local behavior:

| Condition | Baseline Truth | Target Truth | Change |
| --- | ---: | ---: | ---: |
| `wrong_high` | 6/36 | 22/36 | +16 |
| `wrong_direct` | 18/36 | 29/36 | +11 |
| `wrong_unsure` | 33/36 | 33/36 | 0 |
| `wrong_marked_false` | 34/36 | 34/36 | 0 |
| `wrong_untrusted` | 34/36 | 34/36 | 0 |
| `anti_wrong` | 34/36 | 34/36 | 0 |
| `no_hint` | 34/36 | 34/36 | 0 |
| `correct_hint` | 34/36 | 31/36 | -3 |

Reliability notes:

- the effect survives holdout and paraphrase holdout;
- it is concentrated where the model was behaviorally vulnerable: `wrong_high` and `wrong_direct`;
- target parseability is intact;
- the `correct_hint` condition loses 3 truth-following answers, so side effects are not zero;
- the wrong-layer control is not a specificity success because it destroys parseability and answer quality;
- the wrong-token control is a useful locality control because it leaves behavior at baseline.

## Preregistration Compliance

The preregistration said to promote only if at least one hidden probe beat the output/logit margin baseline. That did not happen.

The existing runner still executed the calibrated generation-validation arms after selecting the best calibration cell. Therefore the intervention result is valid as a diagnostic control-surface artifact, but it must not be upgraded into a supported mechanism card.

## Comparison To Qwen3-0.6B

Qwen3-1.7B improves the practical control story:

- it provides a cleaner target intervention with intact answer formatting;
- random and wrong-token controls do not explain the target effect;
- sign-flip behaves directionally as expected;
- the effect generalizes to held-out and paraphrased rows.

It does not solve the explanation problem:

- the best signature remains output/logit-margin dominated;
- hidden probes are real but not uniquely explanatory;
- the intervention likely acts through late residual answer-margin transport rather than an interpretable mechanism.

## Next Step

Do not run another broad dense-direction sweep on Qwen3-1.7B as if this were a mechanism card.

Useful next moves are narrower:

1. run a logit-conditioned or margin-residualized intervention pass on Qwen3-1.7B to test whether `h21 alpha=0.5` has any effect beyond direct answer-margin movement;
2. move MC-001 to an artifact-rich Gemma stack and use sparse-feature/path tools;
3. treat Qwen3-1.7B `h21 alpha=0.5` as the current best deployable diagnostic control surface, not as an explanation.
