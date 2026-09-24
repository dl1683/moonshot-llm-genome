# MC-001 Qwen3-0.6B Controlled V5 Prefill Status

Status: prompt-prefill intervention audit complete; not a supported mechanism card.

V5 corrected a methodological ambiguity in v4. The v4 hooks patched hidden states during every `generate()` forward pass, including one-token decode steps. V5 separately tested all-step additive steering, prompt-prefill-only additive steering, and prompt-prefill-only donor replacement on the same 104 hard agreement-favored wrong-hint rows.

The important positive result is that additive h14 steering still works when applied only during the prompt/prefix prefill pass. The important negative result is that donor replacement still does not transfer the effect, so the current evidence remains a control surface rather than a localized hidden mechanism.

## Purpose

V5 asked whether the v3 raw h14 effect was merely a repeated decode-step artifact.

The hard subset was unchanged from v4:

- splits: calibration, holdout, paraphrase holdout;
- conditions: `wrong_marked_false`, `wrong_untrusted`, `wrong_unsure`, `wrong_direct`, `wrong_high`;
- baseline output-margin bin: `agreement_favored`;
- selected rows: 104.

Baseline on this subset:

- truth-following: 0/104 = 0.0 percent;
- user-agreement error: 101/104 = 97.1 percent;
- other error: 3/104 = 2.9 percent;
- parseable: 104/104.

## Artifacts

- runner: `code/mc001_qwen3_controlled_v5_prefill_audit.py`
- result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v5_prefill_20260629T213701.json`
- manifest: `data/cards/MC001/mc001_qwen3_0p6b_controlled_v5_prefill_manifest.jsonl`
- manifest SHA-256: `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7`
- model: `Qwen/Qwen3-0.6B`
- rows: same 384-row v2/v3/v4 manifest
- selected validation rows: 104 agreement-favored wrong-hint rows
- hidden indices collected: `7`, `13`, `14`
- elapsed runtime: 379.4 seconds

Direction:

- kind: raw truth-minus-agreement at the last prompt token;
- hidden index: `14`;
- discovery train rows: 66;
- discovery positive rate: 57.6 percent;
- raw norm: 3.112;
- median hidden norm: 42.631.

## Main Results

| Arm | Truth | Agreement | Other | Parseable | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0/104 = 0.0 percent | 101/104 = 97.1 percent | 3/104 = 2.9 percent | 104/104 | hard subset |
| prompt guard | 2/104 = 1.9 percent | 100/104 = 96.2 percent | 2/104 = 1.9 percent | 104/104 | near-baseline |
| raw h14 all steps `alpha=0.50` | 30/104 = 28.8 percent | 61/104 = 58.7 percent | 13/104 = 12.5 percent | 104/104 | reproduces v3 hard-bin effect |
| raw h14 prefill `alpha=0.50` | 21/104 = 20.2 percent | 72/104 = 69.2 percent | 11/104 = 10.6 percent | 104/104 | survives prompt-prefill isolation |
| raw h14 prefill `alpha=1.00` | 41/95 = 43.2 percent | 51/95 = 53.7 percent | 3/95 = 3.2 percent | 95/104 | stronger but less parseable |
| same no-hint h14 prefill mix 0.50 | 1/104 = 1.0 percent | 100/104 = 96.2 percent | 3/104 = 2.9 percent | 104/104 | near-baseline |
| same no-hint h14 prefill mix 1.00 | 2/104 = 1.9 percent | 99/104 = 95.2 percent | 3/104 = 2.9 percent | 104/104 | near-baseline |
| same correct-hint h14 prefill mix 1.00 | 1/104 = 1.0 percent | 100/104 = 96.2 percent | 3/104 = 2.9 percent | 104/104 | near-baseline |
| nearby same no-hint h13 prefill mix 1.00 | 1/104 = 1.0 percent | 99/104 = 95.2 percent | 4/104 = 3.8 percent | 104/104 | near-baseline |
| random other no-hint h14 prefill mix 1.00 | 1/104 = 1.0 percent | 101/104 = 97.1 percent | 2/104 = 1.9 percent | 104/104 | near-baseline |
| same-correct-letter other no-hint h14 prefill mix 1.00 | 2/104 = 1.9 percent | 99/104 = 95.2 percent | 3/104 = 2.9 percent | 104/104 | near-baseline |
| wrong-token same no-hint h14 prefill mix 1.00 | 0/0 parseable | 0/0 parseable | 0/0 parseable | 0/104 | destructive |

## Split Results

Raw h14 prompt-prefill-only `alpha=0.50`:

| Split | Baseline Truth | Prefill Truth | Prefill Agreement | Prefill Other | Parseable |
| --- | ---: | ---: | ---: | ---: | ---: |
| calibration | 0/34 = 0.0 percent | 5/34 = 14.7 percent | 25/34 = 73.5 percent | 4/34 = 11.8 percent | 34/34 |
| holdout | 0/35 = 0.0 percent | 8/35 = 22.9 percent | 24/35 = 68.6 percent | 3/35 = 8.6 percent | 35/35 |
| paraphrase holdout | 0/35 = 0.0 percent | 8/35 = 22.9 percent | 23/35 = 65.7 percent | 4/35 = 11.4 percent | 35/35 |

Raw h14 prompt-prefill-only `alpha=1.00`:

| Split | Baseline Truth | Prefill Truth | Prefill Agreement | Prefill Other | Parseable |
| --- | ---: | ---: | ---: | ---: | ---: |
| calibration | 0/34 = 0.0 percent | 14/34 = 41.2 percent | 18/34 = 52.9 percent | 2/34 = 5.9 percent | 34/34 |
| holdout | 0/35 = 0.0 percent | 13/30 = 43.3 percent | 17/30 = 56.7 percent | 0/30 = 0.0 percent | 30/35 |
| paraphrase holdout | 0/35 = 0.0 percent | 14/31 = 45.2 percent | 16/31 = 51.6 percent | 1/31 = 3.2 percent | 31/35 |

## Interpretation

What v5 supports:

- the v3/v4 raw h14 additive effect is not only a repeated decode-step hook artifact;
- a prompt-prefill-only additive h14 intervention can move the hardest agreement-favored wrong-hint rows toward truth-following;
- donor replacement at the same prompt-prefill boundary remains near-baseline;
- wrong-token prefill replacement is destructive, so position handling remains a real side-effect risk.

What v5 rejects:

- closing Qwen3-0.6B solely because v4 full replacement was destructive;
- explaining the raw h14 effect as simple same-question no-hint or correct-hint state transfer;
- promoting the result to a mechanism card without matched additive prefill nulls, full no/correct side-effect validation, and output-margin/residual controls.

## Verdict

Do not write a supported mechanism card from v5.

The v5 result is a stronger control-surface result:

> On the 104 hardest agreement-favored wrong-hint rows, raw h14 steering applied only during prompt prefill moves truth-following from 0/104 to 21/104 at `alpha=0.50`, and to 41/95 parseable outputs at `alpha=1.00`.

The mechanism claim still fails because:

- output margin remains the dominant diagnostic baseline from v2/v3;
- residualized h14 did not preserve the effect in v3;
- donor replacement does not localize a path;
- v5 only tested the hard wrong-hint subset, not full no-hint and correct-hint side-effect rows;
- matched additive prefill controls remained missing in V5.

## Next Decision

Retrospective after V6: this required audit has now been run. See [MC-001 Qwen3-0.6B Controlled V6 Prefill Controls Status](MC001_QWEN3_0P6B_CONTROLLED_V6_PREFILL_CONTROLS_STATUS.md). V6 confirmed the additive control surface but closed dense-prefill steering as an h14-local mechanism path because nearby h13 application was equally or more active.

The required follow-up was one more Qwen3-0.6B iteration, limited to a narrow additive prefill control audit.

The V6 runner did:

1. apply raw h14 additive steering only during prompt prefill;
2. include matched random and nearby additive prefill controls;
3. include residualized additive prefill arms if runtime permits;
4. run on calibration, holdout, paraphrase holdout, plus no-hint and correct-hint side-effect rows;
5. preserve the output-margin bin analysis so the agreement-favored hard rows remain visible.

Because that pass confirmed a control surface but failed layer locality, Qwen3-0.6B dense-prefill steering is closed as a control-only failed-mechanism artifact. Retrospective after V13: the answer-prefix branch failed, the attention-source branch found strong hint-source dependence without circuit localization, and V12/V13 showed the large source result is mostly prompt-source removal or recomputation. The broad Qwen3-0.6B route is now closed.
