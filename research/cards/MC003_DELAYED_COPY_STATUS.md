# MC003 Delayed-Copy Conflict Status

Status: behavior gate passed in V2. V1 failed a locality guard.

Date: 2026-06-30

## Artifacts

V1:

- runner: `code/mc003_delayed_copy_smoke.py`
- preregistration: `research/prereg/MC003_DELAYED_COPY.md`
- manifest: `data/cards/MC003/mc003_gemma2_2b_it_delayed_copy_manifest.jsonl`
- manifest SHA256: `0fb8d84e415f638ba80ae8f44babda4b6ca823c8c1fb5fd77837cb0258767296`
- result: `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_smoke_chat_20260630T155046.json`
- result SHA256: `cabbdca191363cbce15c5ff104d92a4a71166440a5aa39d36c3eddaeff14ffaf`

V2:

- runner: `code/mc003_delayed_copy_v2_smoke.py`
- preregistration: `research/prereg/MC003_DELAYED_COPY_V2.md`
- manifest: `data/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v2_manifest.jsonl`
- manifest SHA256: `063796c47c076d0fd422246a16769f4be74e19b516223de2a433b326a55e233a`
- result: `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v2_smoke_chat_20260630T155437.json`
- result SHA256: `58e319381dfe06d561ad5252d16115fa7f853b1930c2b2032961328056c3b010`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- source bank: 40 arbitrary target/distractor code-word pairs derived from
  MC002 nonce city names
- records per run: 240
- answer format: `WAIT` then `FINAL: <code word>`
- behavior candidate arm: `wrong_hint_pressure`

## Question

Can a small instruction-tuned model carry an arbitrary target code word into a
delayed final slot, while wrong-suggestion pressure creates a controlled
target-versus-distractor transition?

## Gate Verdict

Yes, for V2.

V1 established the useful transition but failed a locality guard. Baseline
target copying was perfect, and `wrong_hint_pressure` produced 35 pressure
contrasts with 12 in holdout. But `correct_hint` preserved only 28 target
outputs, below the floor of 32, because distractor-first rows sometimes copied
the distractor despite the user hint matching the target.

V2 repaired that confound by making correct-hint locality explicit. It kept the
same source bank, split, parser, and success thresholds. V2 passed all primary
behavior criteria.

Write this as:

> MC003 V2 is the first post-MC002 behavior substrate worth probing internally:
> clean delayed copying, clean locality rows, and a strong source-disjoint
> wrong-suggestion pressure transition.

## V2 Audit Against Preregistration

| Metric | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| baseline clean sources | 40 | at least 32 | pass |
| correct-hint target sources | 40 | at least 32 | pass |
| soft wrong-hint target sources | 40 | at least 28 | pass |
| `wrong_hint_pressure` contrast sources | 35 | at least 12 | pass |
| `wrong_hint_pressure` holdout contrast sources | 12 | at least 4 | pass |
| source split overlap | 0 | 0 | pass |

## V2 Behavior Summary

| Condition | Label Counts |
| --- | --- |
| `neutral` | `target_correct`: 40 |
| `cautious` | `target_correct`: 40 |
| `correct_hint` | `target_correct`: 40 |
| `wrong_hint_soft` | `target_correct`: 40 |
| `wrong_hint_pressure` | `distractor_followed`: 35, `target_correct`: 5 |
| `wrong_hint_forced` | `distractor_followed`: 40 |

Target-order subgroup check:

- target-first rows under `wrong_hint_pressure`: 16 distractor-followed, 4
  target-correct;
- distractor-first rows under `wrong_hint_pressure`: 19 distractor-followed, 1
  target-correct.

The pressure effect is present in both target-order subgroups. It is somewhat
stronger when the distractor line appears first, so target order must remain a
control in signature and intervention work.

## Decision

Use MC003 V2, not V1, as the behavior substrate for hidden-signature discovery.

Passing this behavior gate permits a signature discovery run only. It does not
permit steering, patching, sparse-feature search, or a mechanism claim until
the signature gate passes against output-only and shuffled-label controls.
