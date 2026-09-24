# MC001G Gemma 2 2B Smoke Status

Status: complete. Gate 1 did not pass for mechanism discovery.

Date: 2026-06-30

## Artifacts

- generation runner: `code/mc001_qwen3_smoke.py`
- logit runner: `code/mc001_logit_smoke.py`
- stack decision: `research/19_GEMMA_STACK_DECISION.md`
- generation preregistration: `research/prereg/MC001G_GEMMA2_2B_SMOKE.md`
- logit preregistration: `research/prereg/MC001G_GEMMA2_2B_LOGIT_SMOKE.md`
- base generation manifest: `data/cards/MC001G/mc001g_gemma2_2b_smoke_factual_ladder_manifest.jsonl`
- base generation manifest SHA256: `d4563d9556e0d39ef584f2d6a37fa61d2060290c84bdd6ce8bfed54fe169bace`
- IT generation manifest: `data/cards/MC001G/mc001g_gemma2_2b_it_smoke_factual_ladder_manifest.jsonl`
- IT generation manifest SHA256: `29eaa45173d6c2e3ebb27b78ba10a2b8a26e3f3d82acc407985b14fceeea4ffd`
- base logit manifest: `data/cards/MC001G/mc001g_gemma2_2b_logit_factual_ladder_manifest.jsonl`
- base logit manifest SHA256: `d4563d9556e0d39ef584f2d6a37fa61d2060290c84bdd6ce8bfed54fe169bace`
- base raw-logit manifest: `data/cards/MC001G/mc001g_gemma2_2b_logit_raw_factual_ladder_manifest.jsonl`
- base raw-logit manifest SHA256: `d4563d9556e0d39ef584f2d6a37fa61d2060290c84bdd6ce8bfed54fe169bace`

Results:

- base generation: `results/cards/MC001G/mc001g_gemma2_2b_smoke_factual_ladder_20260630T115439.json`
- IT generation: `results/cards/MC001G/mc001g_gemma2_2b_it_smoke_factual_ladder_20260630T115556.json`
- base logit, wrapped render: `results/cards/MC001G/mc001g_gemma2_2b_logit_factual_ladder_20260630T120051.json`
- base logit, raw render: `results/cards/MC001G/mc001g_gemma2_2b_raw_logit_raw_factual_ladder_20260630T120153.json`

## Question

Does Gemma provide a better substrate for the next MC-001 mechanism-card attempt,
especially the sparse-feature/path branch suggested by Gemma Scope?

## Gate Verdict

No, not yet.

Base `google/gemma-2-2b` is not promotable on this manifest. Free generation
fails the answer-only interface, and forced-choice logit scoring still has weak
no-hint accuracy plus strong option/prior effects.

The instruction-tuned fallback `google/gemma-2-2b-it` shows a real MC-001-like
pressure behavior, but its baseline and correct-hint accuracy are too weak for
hidden-state discovery. It is useful for prompt and item repair, not for a
sparse mechanism claim tied to Gemma Scope base-model features.

## Run Summary

| Run | Model | Mode | Overall Truth | Wrong-Hint Truth | Wrong-Hint Agreement | No-Hint Truth | Correct-Hint Truth | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| base generation | `google/gemma-2-2b` | free generation | 5/160 | 0/100 | 29/100 | 3/20 | 1/20 | fail: unparseable/interface |
| IT generation | `google/gemma-2-2b-it` | free generation | 68/160 | 37/100 | 51/100 | 11/20 | 6/20 | fail: dirty baseline |
| base logit wrapped | `google/gemma-2-2b` | forced-choice logit, wrapped render | 23/160 | 14/100 | 55/100 | 3/20 | 3/20 | fail: fixed option prior |
| base logit raw | `google/gemma-2-2b` | forced-choice logit, raw render | 46/160 | 21/100 | 67/100 | 7/20 | 11/20 | fail: baseline too weak |

Parseability:

- base generation: 58/160 parseable;
- IT generation: 160/160 parseable;
- logit runs: forced-choice by construction.

Elapsed times:

- base generation: 152.1 seconds;
- IT generation: 42.0 seconds;
- base logit wrapped: 15.7 seconds;
- base logit raw: 15.6 seconds.

## Clean-Subset Signal

The aggregate gate fails, but both the IT generation fallback and base raw-logit
control expose the same important clue: the behavior exists on a small clean
subset.

Clean subset means the item is truth-following under both `no_hint` and
`correct_hint`.

| Run | Clean Items | Marked False | Untrusted | Unsure | Direct | High | Anti-Wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IT generation | 6 | 6/6 truth | 6/6 truth | 5/6 truth, 1/6 agreement | 4/6 agreement | 6/6 agreement | 6/6 truth |
| base raw logit | 7 | 7/7 truth | 7/7 truth | 6/7 agreement | 7/7 agreement | 7/7 agreement | 7/7 truth |

This is not enough for a reliable signature/intervention split, but it is enough
to justify a targeted prompt and item-bank repair. The failure mode is not
"Gemma has no MC-001 behavior"; it is "the current manifest does not produce
enough clean Gemma examples."

## Failure Modes

Base generation:

- answer-only instruction is not reliable;
- parseability is only 58/160;
- wrong-hint rows have 0/100 truth-following.

Base logit, wrapped render:

- the generic `System/User/Assistant` wrapper is a poor base-model interface;
- no-hint and correct-hint truth are both only 3/20;
- outputs are dominated by an `A` option prior.

Base logit, raw render:

- raw rendering improves correct-hint truth to 11/20 and overall truth to 46/160;
- no-hint truth remains only 7/20;
- `wrong_direct` and `wrong_high` saturate at 20/20 user-agreement errors;
- the clean subset has only 7 items.

IT generation:

- parseability is perfect and pressure sensitivity is real;
- no-hint truth is only 11/20;
- correct-hint truth is only 6/20;
- clean subset has only 6 items;
- it is not a direct Gemma Scope sparse-feature substrate without transfer evidence.

## Decision

Do not install sparse tooling yet.

Do not start hidden-state discovery on MC001G yet.

The next Gemma step should be a repair gate:

1. remove prompt artifacts such as hint lines beginning with `A previous...`;
2. build a larger Gemma-targeted objective item bank with balanced correct and wrong letters;
3. require at least 24 clean items before discovery;
4. preserve base raw-logit scoring as the primary substrate test;
5. keep IT generation as a diagnostic fallback, not the mechanism substrate.

If the repair gate cannot produce enough clean base-Gemma examples, close the
Gemma Scope route for MC-001 and move to a narrower Qwen3-1.7B causal-path audit.
