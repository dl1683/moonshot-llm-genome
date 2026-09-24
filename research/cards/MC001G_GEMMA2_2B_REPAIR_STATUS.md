# MC001G Gemma 2 2B Repair Gate Status

Status: complete. Behavior substrate repair passed; mechanism discovery is now justified.

Date: 2026-06-30

## Artifacts

- runner: `code/mc001_logit_smoke.py`
- manifest generator: `code/mc001_qwen3_smoke.py`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_REPAIR.md`
- prior status: `research/cards/MC001G_GEMMA2_2B_SMOKE_STATUS.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_manifest.jsonl`
- manifest SHA256: `cd0cf8f8d75ee0e8a64cd115e40df29bff97bc6e57ab45802050aea0bc8ccac1`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- model: `google/gemma-2-2b`
- run type: `mc001g_gemma2_2b_repair_logit_raw_smoke`
- variant: `gemma_repair`
- render mode: `raw`
- records: 448
- elapsed: 27.3 seconds

## Question

Can the failed first Gemma gate be repaired by using a larger balanced item bank,
multiline options, cleaner hint text, and base-model raw-logit scoring?

## Gate Verdict

Yes.

This is still not a mechanism result. It is a behavior-substrate result: the
base Gemma 2 2B stack now has enough clean examples to justify signature
discovery and later intervention tests.

## Aggregate Behavior

| Condition | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| `no_hint` | 42/64 | 8/64 | 14/64 | 64/64 |
| `correct_hint` | 64/64 | 0/64 | 0/64 | 64/64 |
| `wrong_disclaimed` | 25/64 | 38/64 | 1/64 | 64/64 |
| `wrong_unsure` | 20/64 | 44/64 | 0/64 | 64/64 |
| `wrong_direct` | 1/64 | 63/64 | 0/64 | 64/64 |
| `wrong_high` | 0/64 | 64/64 | 0/64 | 64/64 |
| `anti_wrong` | 49/64 | 0/64 | 15/64 | 64/64 |

Overall:

- truth-following: 201/448;
- user-agreement error: 217/448;
- other error: 30/448;
- wrong-hint rows: 46/256 truth, 209/256 agreement, 1/256 other.

## Clean Subset

Clean means truth-following under both `no_hint` and `correct_hint`.

The repair gate found 42 clean items.

| Correct Letter | Clean Items |
| --- | ---: |
| `A` | 16 |
| `B` | 10 |
| `C` | 8 |
| `D` | 8 |

Clean-subset pressure behavior:

| Condition | Truth | Agreement | Other |
| --- | ---: | ---: | ---: |
| `wrong_disclaimed` | 21/42 | 20/42 | 1/42 |
| `wrong_unsure` | 19/42 | 23/42 | 0/42 |
| `wrong_direct` | 1/42 | 41/42 | 0/42 |
| `wrong_high` | 0/42 | 42/42 | 0/42 |
| `anti_wrong` | 39/42 | 0/42 | 3/42 |

This passes the preregistered repair criteria:

- at least 24 clean items: 42;
- at least 4 clean items per correct letter: minimum 8;
- intermediate wrong-pressure conditions are not saturated;
- direct/high wrong hints produce substantial user agreement;
- anti-wrong mostly preserves truth-following;
- correct-hint rows are fully balanced across answer letters.

## Reliability Notes

The substrate is usable but not clean enough to skip controls:

- no-hint rows still show an `A` option bias: parsed `A` appears 34/64 times;
- clean items are answer-letter imbalanced: `A` has 16 clean items, `D` has 8;
- `wrong_direct` and `wrong_high` are nearly or fully saturated, so they are useful hard bins but not sufficient by themselves;
- `wrong_disclaimed` and `wrong_unsure` are the best intermediate pressure rows;
- `anti_wrong` has 3/42 other errors on the clean subset and 15/64 other errors overall.

The discovery run must stratify by correct answer letter and condition. Any
signature that only separates `A`-favored rows or direct/high prompt pressure is
not a mechanism.

## Next Step

Promote MC001G to signature discovery on the 42 clean items.

Requirements for the discovery pass:

- use only clean items for the primary split;
- preserve all answer letters in calibration and holdout;
- include output-margin, prompt-condition, correct-letter, wrong-letter, and baseline-answer controls;
- measure dense residual signatures before claiming sparse features;
- only install/use Gemma Scope features after the dense and margin baselines are saved;
- do not run an intervention unless a signature beats the obvious baselines.
