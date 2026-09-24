# MC004 In-Context Binding V2 Status

Status: behavior condition-balance gate passed; `update_prefer_latest` frozen
for signature discovery.

Date: 2026-06-30

## Artifacts

- runner: `code/mc004_in_context_binding_v2_smoke.py`
- preregistration: `research/prereg/MC004_IN_CONTEXT_BINDING_V2.md`
- manifest: `data/cards/MC004/mc004_gemma2_2b_it_in_context_binding_v2_manifest.jsonl`
- result:
  `results/cards/MC004/mc004_gemma2_2b_it_in_context_binding_v2_smoke_chat_20260630T163531.json`
- result SHA256:
  `6f05d468dcf249397d122ef04e86ea025d1183ba4479ed8187e7b26e067d0bf7`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- source bank: same 40 nonce entity/code pairs as MC004 V1
- records: 320 rows, 40 per condition
- prompt variant: `v2_update_conflict`

## Question

Can an original-reference-note versus later-update conflict produce
target-correct and update-following outcomes within the same condition?

## Gate Verdict

Yes.

`update_prefer_latest` passed all preregistered behavior criteria and is frozen
for the next signature run. It produced 20 original-note answers, 19
update-following answers, and 1 other answer. The target/update labels were
present in discovery, holdout, and both holdout target-order subgroups.

## Audit Against Preregistration

Locality guards:

| Guard | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| baseline clean sources | 39 | at least 32 | pass |
| correct-hint target sources | 40 | at least 32 | pass |
| source split overlap | 0 | 0 | pass |

Update calibration:

| Condition | Target | Update | Other | Discovery T/U | Holdout T/U | Candidate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `update_unverified` | 37 | 2 | 1 | 26/1 | 11/1 | no |
| `update_ambiguous` | 34 | 5 | 1 | 25/2 | 9/3 | no |
| `update_prefer_latest` | 20 | 19 | 1 | 14/12 | 6/7 | yes |
| `update_authoritative` | 18 | 22 | 0 | 14/13 | 4/9 | no |
| `update_forced` | 8 | 32 | 0 | 7/20 | 1/12 | no |

`update_authoritative` had enough total and split labels but failed the
holdout target-order subgroup guard. `update_prefer_latest` is therefore the
only qualifying condition.

## Decision

Run one condition-balanced signature attempt using only target-correct and
update-following rows from `update_prefer_latest`.

Do not start intervention work from this behavior result alone.
