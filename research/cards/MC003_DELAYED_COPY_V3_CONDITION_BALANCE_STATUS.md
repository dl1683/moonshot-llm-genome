# MC003 Delayed-Copy V3 Condition-Balance Status

Status: behavior condition-balance gate passed; `wrong_hint_balanced` frozen as
the next signature condition.

Date: 2026-06-30

## Artifacts

- runner: `code/mc003_delayed_copy_v3_condition_balance.py`
- preregistration:
  `research/prereg/MC003_DELAYED_COPY_V3_CONDITION_BALANCE.md`
- manifest:
  `data/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v3_condition_balance_manifest.jsonl`
- result:
  `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v3_condition_balance_smoke_chat_20260630T161743.json`
- result SHA256:
  `3909fbd062f5e75b9c632640aa5710259e2120b636b7f654571e3d2b2da7c470`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- source bank: same 40 MC003 sources as V1/V2
- records: 320 rows, 40 per condition
- source split: same source-disjoint MC003 split
- prompt variant: `v3_condition_balance`

## Question

Can MC003 produce target-correct and distractor-followed outcomes within the
same wrong-hint condition, so a later hidden signature is not merely detecting
pressure versus non-pressure prompt state?

## Gate Verdict

Yes.

The behavior-only calibration found two qualifying same-condition tables:
`wrong_hint_balanced` and `wrong_hint_authority`. The preregistered selection
rule chooses the qualifying condition with the smallest absolute
target-versus-distractor count difference, breaking ties toward weaker
pressure. That freezes `wrong_hint_balanced` for the next signature run.

Write this as:

> MC003 V3 repaired the condition-confounding problem at the behavior-table
> level. It permits a condition-balanced signature attempt on
> `wrong_hint_balanced`, but does not itself support any hidden-state or
> intervention claim.

## Audit Against Preregistration

Locality guards:

| Guard | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| baseline clean sources | 40 | at least 32 | pass |
| correct-hint target sources | 40 | at least 32 | pass |
| source split overlap | 0 | 0 | pass |

Wrong-hint calibration:

| Condition | Target | Distractor | Discovery T/D | Holdout T/D | Candidate |
| --- | ---: | ---: | ---: | ---: | --- |
| `wrong_hint_guarded` | 40 | 0 | 27/0 | 13/0 | no |
| `wrong_hint_default` | 33 | 7 | 24/3 | 9/4 | no |
| `wrong_hint_balanced` | 19 | 21 | 12/15 | 7/6 | yes |
| `wrong_hint_pressure` | 5 | 35 | 4/23 | 1/12 | no |
| `wrong_hint_authority` | 23 | 17 | 17/10 | 6/7 | yes |

`wrong_hint_balanced` also passed the holdout target-order subgroup guard:

- target-first holdout: 5 target, 2 distractor;
- distractor-first holdout: 2 target, 4 distractor.

## Decision

Use `wrong_hint_balanced` as the only primary condition for the next MC003
signature preregistration.

Do not start intervention work from this behavior result. It is only a repaired
behavior table that closes the V2 condition-trace confound.
