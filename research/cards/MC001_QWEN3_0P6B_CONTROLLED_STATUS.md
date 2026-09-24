# MC-001 Qwen3-0.6B Controlled Status

Status: controlled pass complete; not a supported mechanism card.

The useful result is narrower: `Qwen/Qwen3-0.6B` has a controllable MC-001 surface on factual multiple-choice wrong-hint prompts, but the current evidence fails the final card standard because the hidden signature does not beat the output/logit baseline and the intervention is not robust enough under paraphrase and side-effect checks.

Follow-up: the output-margin-conditioned v2 pass is now recorded in [MC-001 Qwen3-0.6B Controlled V2 Status](MC001_QWEN3_0P6B_CONTROLLED_V2_STATUS.md). V2 keeps the no-card verdict and sharpens the next target: lower-dose h14 validation plus answer-letter/output-margin residualized directions.

## Purpose

Turn the smoke result into a controlled discovery/calibration run:

- fixed discovery, calibration, holdout, and paraphrase-holdout splits;
- structured option maps and answer parsing;
- hidden-state probes trained on discovery and evaluated on later splits;
- activation-direction intervention selected on calibration;
- matched nulls on calibration, holdout, and paraphrase holdout.

## Artifacts

- runner: `code/mc001_qwen3_controlled.py`
- result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_20260629T200930.json`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_manifest.jsonl`
- manifest SHA-256: `380b83823ca0bcd5da5e43ff2f6871901e92d2050de5e5a65f6cc8e2202e10ba`
- model: `Qwen/Qwen3-0.6B`
- rows: 384 total, 96 per split across discovery, calibration, holdout, and paraphrase holdout
- hidden indices tested: `5`, `7`, `14`
- intervention alphas tested: `-1.0`, `-0.5`, `0.0`, `0.5`, `1.0`
- max new tokens: `24`

Probe candidate counts after filtering to agreement-vs-truth wrong-hint rows:

- discovery: 67
- calibration: 69
- holdout: 54
- paraphrase holdout: 60

## Baseline Behavior

Across all 384 manifest rows, baseline generation was fully parseable:

- truth-following: 148/384 = 38.5 percent
- user-agreement error: 182/384 = 47.4 percent
- other error: 54/384 = 14.1 percent

On the generation-validation rows only, baseline wrong-hint truth-following was low:

| Split | Baseline Wrong-Hint Truth | Parseable |
| --- | ---: | ---: |
| calibration | 12/60 = 20.0 percent | 60/60 |
| holdout | 13/60 = 21.7 percent | 60/60 |
| paraphrase holdout | 13/60 = 21.7 percent | 60/60 |

This is a valid intervention substrate: wrong-hint compliance exists and is not saturated away.

## Signature Gate

The hidden probes are real diagnostics, but they do not pass the mechanism-card signature gate because the output/logit baseline is stronger.

| Diagnostic | Calibration AUC | Holdout AUC | Paraphrase Holdout AUC |
| --- | ---: | ---: | ---: |
| condition-only baseline | 0.853 | 0.890 | 0.890 |
| answer-logit margin baseline | 1.000 | 1.000 | 1.000 |
| hidden probe h5 | 0.841 | 0.820 | 0.761 |
| hidden probe h7 | 0.846 | 0.797 | 0.819 |
| hidden probe h14 | 0.843 | 0.887 | 0.919 |

Label-permutation hidden probes fell near chance or below on holdout/paraphrase, so the hidden probes are not pure noise. The failure is different: output-side evidence already explains the behavior too well. A hidden-state mechanism claim cannot ignore that.

Verdict on signature gate: fail for final-card purposes.

## Intervention Gate

Selected intervention:

- key: `h14_alpha1.0`
- hidden index: `14`
- alpha: `1.0`
- calibration selection score: `0.1083`

Wrong-hint truth-following by arm:

| Arm | Calibration | Holdout | Paraphrase Holdout | Parseability Notes |
| --- | ---: | ---: | ---: | --- |
| baseline | 12/60 = 20.0 percent | 13/60 = 21.7 percent | 13/60 = 21.7 percent | 100 percent parseable |
| target h14 +1.0 | 16/59 = 27.1 percent | 36/60 = 60.0 percent | 21/60 = 35.0 percent | 287/288 overall parseable |
| random matched norm | 11/60 = 18.3 percent | 17/60 = 28.3 percent | 14/60 = 23.3 percent | 100 percent parseable |
| sign flip | 15/60 = 25.0 percent | 15/60 = 25.0 percent | 15/60 = 25.0 percent | mostly other-error bias |
| wrong layer | 11/38 = 28.9 percent | 10/30 = 33.3 percent | 11/40 = 27.5 percent | parseability collapse |
| wrong token | 12/60 = 20.0 percent | 13/60 = 21.7 percent | 13/60 = 21.7 percent | identical to baseline summary |

The target intervention beats the clean parseable nulls on the standard holdout. The strongest standard-holdout effect is real: 21.7 percent baseline truth-following to 60.0 percent target truth-following on wrong-hint rows.

The effect is weaker on paraphrase holdout: 21.7 percent to 35.0 percent. That is directionally positive but not strong enough to call robust.

No-hint plus correct-hint side effects:

| Arm | Calibration | Holdout | Paraphrase Holdout |
| --- | ---: | ---: | ---: |
| baseline | 14/24 = 58.3 percent | 15/24 = 62.5 percent | 18/24 = 75.0 percent |
| target h14 +1.0 | 10/24 = 41.7 percent | 18/24 = 75.0 percent | 15/24 = 62.5 percent |
| random matched norm | 14/24 = 58.3 percent | 16/24 = 66.7 percent | 17/24 = 70.8 percent |
| wrong token | 15/24 = 62.5 percent | 14/24 = 58.3 percent | 18/24 = 75.0 percent |

The target improves no/correct rows on standard holdout but drops them on calibration and paraphrase. Overall no/correct truth-following drops from 47/72 baseline to 43/72 target. This violates the preregistered side-effect spirit even though the standard holdout alone looks good.

Verdict on intervention gate: promising but not card-complete.

## Reliability Gate

Reliability failures:

- the signature does not beat the output/logit baseline;
- paraphrase robustness is too weak;
- no/correct side effects are not clean;
- the selected direction likely carries answer-token or format bias as well as truth-vs-agreement signal;
- wrong-layer control destroys parseability, so it cannot serve as a clean efficacy null;
- the current factual-ladder set is still small and item-specific.

What survives:

- Qwen3-0.6B remains useful for fast iteration;
- wrong-hint behavior is measurable and controllable on this narrow substrate;
- h14 positive steering is a real causal handle on standard-form holdout behavior;
- wrong-token and random matched-norm controls do not reproduce the standard-holdout target effect;
- answer-logit margin is an extremely strong diagnostic and must become a first-class baseline, not an afterthought.

## Verdict

Do not write a supported MC-001 mechanism card from this run.

Write the result as:

> A controlled Qwen3-0.6B pass found a causal activation-direction surface for reducing wrong-hint compliance on standard factual multiple-choice holdout prompts, but the evidence remains diagnostic/control-surface evidence rather than a hidden-mechanism claim because output logits fully predict the behavior and the intervention is not paraphrase-robust enough.

## Next Iteration

Continue with Qwen3-0.6B because it is cheap and fast enough to run many controlled loops. Do not broaden back to arithmetic/code/logic yet.

The next runner should target the exact failure modes:

1. Stratify by answer-logit margin before intervention.
2. Ask whether hidden states add predictive or causal value after controlling for output margin.
3. Counterbalance answer letters more aggressively to reduce answer-token bias.
4. Add a prompt-only anti-sycophancy baseline to the same calibration/holdout table.
5. Separate no-hint and correct-hint side-effect gates instead of only pooled no/correct summaries.
6. Replace the wrong-layer null with a parseability-preserving nearby-layer and early-layer control.
7. Add a position-specific intervention at hint token, final question token, and answer-prefix token.
8. Select intervention strength without viewing holdout or paraphrase holdout.
9. Promote only if hidden evidence beats output-margin evidence or if the claim is explicitly reframed as output-margin-conditioned control.

V2 has now answered that the raw dense h14 direction is not enough. Escalate to Qwen3-1.7B or artifact-rich Gemma only after one narrower v3 pass tests lower doses and an answer-letter/output-margin residualized direction.
