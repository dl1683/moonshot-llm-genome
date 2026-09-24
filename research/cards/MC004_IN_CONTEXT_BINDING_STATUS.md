# MC004 In-Context Binding Status

Status: V1 behavior gate failed; reference-note binding was too robust under
wrong-hint pressure to create a balanced signature table.

Date: 2026-06-30

## Artifacts

- runner: `code/mc004_in_context_binding_smoke.py`
- preregistration: `research/prereg/MC004_IN_CONTEXT_BINDING.md`
- manifest: `data/cards/MC004/mc004_gemma2_2b_it_in_context_binding_manifest.jsonl`
- result:
  `results/cards/MC004/mc004_gemma2_2b_it_in_context_binding_smoke_chat_20260630T163134.json`
- result SHA256:
  `2740f4829e6c9b94629fe1fff88b27eccc1cc158a3f88f06ce19da844c54d386`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- source bank: 40 nonce entity/code pairs derived from MC002 fake sources
- records: 320 rows, 40 per condition
- prompt variant: `v1_in_context_binding`

## Question

Can a nonce entity-to-code binding task produce target-correct and
distractor-followed outcomes within the same wrong-hint condition while
baseline and correct-hint locality remain clean?

## Gate Verdict

No.

The model read the in-context reference notes cleanly: `neutral`, `cautious`,
and `correct_hint` were all 40/40 target-correct. But the wrong-hint pressure
arms did not create a usable contrast. The strongest arms produced only 2/40
distractor-following rows.

## Audit Against Preregistration

Locality guards:

| Guard | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| baseline clean sources | 40 | at least 32 | pass |
| correct-hint target sources | 40 | at least 32 | pass |
| source split overlap | 0 | 0 | pass |

Wrong-hint calibration:

| Condition | Target | Distractor | Other | Candidate |
| --- | ---: | ---: | ---: | --- |
| `wrong_hint_guarded` | 39 | 0 | 1 | no |
| `wrong_hint_default` | 39 | 1 | 0 | no |
| `wrong_hint_balanced` | 37 | 0 | 3 | no |
| `wrong_hint_pressure` | 38 | 2 | 0 | no |
| `wrong_hint_authority` | 38 | 2 | 0 | no |

## Decision

Do not run MC004 V1 hidden-state discovery.

The behavior family remains interesting because the model can cleanly bind
nonce entities to in-context attributes. The next repair should increase the
conflict from a wrong user hint to an explicit later-update conflict and again
require same-condition target/distractor variation before any signature work.
