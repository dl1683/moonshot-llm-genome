# MC-001 Qwen3-0.6B Controlled V4 Patch Status

Status: agreement-bin activation patching pass complete; not a supported mechanism card.

V4 tested the method shift required after v3: stop training broad global directions and instead patch same-question no-hint or correct-hint hidden states into the hard agreement-favored wrong-hint rows. This did not rescue the mechanism claim. Half-patching at h14 barely moved the hard rows, and full hidden-state replacement destroyed parseability across layers and donor controls.

Retrospective note after v5: the v4 hooks applied during every `generate()` forward pass, including one-token decode steps. The v4 donor-replacement failure remains evidence against simple same-question state transfer, but its full-replacement parse collapse should not be treated as a clean prompt-prefill-only locality result. V5 isolates prompt prefill separately in [MC-001 Qwen3-0.6B Controlled V5 Prefill Status](MC001_QWEN3_0P6B_CONTROLLED_V5_PREFILL_STATUS.md).

## Purpose

V4 asked whether the raw h14 steering effect from v3 could be localized by direct same-question activation patching.

The hard subset was:

- splits: calibration, holdout, paraphrase holdout;
- conditions: `wrong_marked_false`, `wrong_untrusted`, `wrong_unsure`, `wrong_direct`, `wrong_high`;
- baseline output-margin bin: `agreement_favored`;
- selected rows: 104.

Baseline on this subset was intentionally hard:

- truth-following: 0/104 = 0.0 percent;
- user-agreement error: 101/104 = 97.1 percent;
- other error: 3/104 = 2.9 percent;
- parseable: 104/104.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v4_patch.py`
- result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v4_patch_20260629T212411.json`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v4_patch_manifest.jsonl`
- manifest SHA-256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- model: `Qwen/Qwen3-0.6B`
- rows: same 384-row v2/v3 manifest
- hidden indices collected: `7`, `13`, `14`
- patch selection rows: 104 agreement-favored wrong-hint rows
- max new tokens: `24`
- elapsed runtime: 807.4 seconds

## Main Results

| Arm | Truth | Agreement | Other | Parseable | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0/104 = 0.0 percent | 101/104 = 97.1 percent | 3/104 = 2.9 percent | 104/104 | hard subset |
| prompt guard | 2/104 = 1.9 percent | 100/104 = 96.2 percent | 2/104 = 1.9 percent | 104/104 | near-baseline |
| same no-hint h14 mix 0.50 | 4/104 = 3.8 percent | 98/104 = 94.2 percent | 2/104 = 1.9 percent | 104/104 | weak |
| same correct-hint h14 mix 0.50 | 2/104 = 1.9 percent | 100/104 = 96.2 percent | 2/104 = 1.9 percent | 104/104 | weak |
| same no-hint h14 mix 1.00 | 0/0 parseable | 0/0 parseable | 0/0 parseable | 0/104 | destructive |
| same correct-hint h14 mix 1.00 | 0/0 parseable | 0/0 parseable | 0/0 parseable | 0/104 | destructive |
| same no-hint h7 mix 1.00 | 0/0 parseable | 0/0 parseable | 0/0 parseable | 0/104 | destructive |
| nearby same no-hint h13 mix 1.00 | 0/0 parseable | 0/0 parseable | 0/0 parseable | 0/104 | destructive |
| same-correct-letter other no-hint h14 mix 1.00 | 0/0 parseable | 0/0 parseable | 0/0 parseable | 0/104 | destructive |
| random other no-hint h14 mix 1.00 | 0/0 parseable | 0/0 parseable | 0/0 parseable | 0/104 | destructive |
| wrong-token same no-hint h14 mix 1.00 | 1/2 parseable | 1/2 parseable | 0/2 parseable | 2/104 | destructive |

## Split Results

The best non-destructive arm was `same_no_hint_h14_m0.50`:

| Split | Baseline Truth | h14 Half-Patch Truth | h14 Half-Patch Agreement | Parseable |
| --- | ---: | ---: | ---: | ---: |
| calibration | 0/34 = 0.0 percent | 1/34 = 2.9 percent | 33/34 = 97.1 percent | 34/34 |
| holdout | 0/35 = 0.0 percent | 1/35 = 2.9 percent | 33/35 = 94.3 percent | 35/35 |
| paraphrase holdout | 0/35 = 0.0 percent | 2/35 = 5.7 percent | 32/35 = 91.4 percent | 35/35 |

The best correct-hint donor half-patch was weaker:

| Split | Correct-Hint Half-Patch Truth | Correct-Hint Half-Patch Agreement | Parseable |
| --- | ---: | ---: | ---: |
| calibration | 0/34 = 0.0 percent | 34/34 = 100.0 percent | 34/34 |
| holdout | 0/35 = 0.0 percent | 34/35 = 97.1 percent | 35/35 |
| paraphrase holdout | 2/35 = 5.7 percent | 32/35 = 91.4 percent | 35/35 |

## Donor Quality

Same-question donor labels were imperfect, but not bad enough to explain the full failure:

| Donor Arm | Truth Donors | Agreement Donors | Other Donors |
| --- | ---: | ---: | ---: |
| same no-hint donor | 60/104 | 18/104 | 26/104 |
| same correct-hint donor | 78/104 | 20/104 | 6/104 |
| same-correct-letter other no-hint donor | 74/104 | 12/104 | 18/104 |
| random other no-hint donor | 69/104 | 15/104 | 20/104 |

Correct-hint donors were truth-following on 78/104 rows, but h14 half-patching from them only produced 2/104 truth-following outputs. The failure is therefore not just "bad donors."

## Interpretation

What v4 supports:

- all-step final-token hidden-state replacement is too destructive in Qwen3-0.6B for this task;
- same-question h14 half-patching is much weaker than the raw h14 direction from v3;
- the raw v3 direction is not explained by a simple "copy the no-hint/correct-hint state into the wrong-hint prompt" path at the final prompt token;
- prompt-only guard remains near-baseline on the hardest agreement-favored rows.

What v4 rejects:

- full activation replacement as a usable MC-001 intervention;
- same-question final-token patching as the missing clean mechanism behind raw h14 steering;
- the idea that the v3 raw effect can be localized by this simple donor-recipient patch.

## Verdict

Do not write a supported mechanism card from v4.

The v4 method shift is a negative result:

> On the 104 hardest agreement-favored wrong-hint rows, same-question h14 half-patching only moved truth-following from 0/104 to 4/104, while full hidden-state replacement across h7/h13/h14 and donor controls collapsed parseability. The raw h14 direction from v3 remains a real control surface, but simple final-token activation patching does not explain it.

## Next Decision

Do not run another final-token donor-replacement patch on Qwen3-0.6B.

The future path is one of:

1. close the Qwen3-0.6B MC-001 attempt as a control-only/failed-mechanism artifact;
2. try a genuinely different localization method, such as additive prompt-prefill controls, answer-prefix patching, or path-local patching rather than all-step final-token state replacement;
3. escalate to a larger or artifact-rich model only after accepting that Qwen3-0.6B has yielded a control surface but not a clean mechanism under current methods.
