# MC002B Context-Support Known-Unknown Status

Status: complete. The reconstructed context-support task failed the behavior
gate.

Date: 2026-06-30

## Artifacts

- runner: `code/mc002b_context_support_smoke.py`
- source seed bank: `code/mc002_known_unknown_smoke.py`
- preregistration: `research/prereg/MC002B_CONTEXT_SUPPORT.md`
- manifest: `data/cards/MC002B/mc002b_gemma2_2b_it_context_support_manifest.jsonl`
- manifest SHA256: `c0be22fda847892b2ea9c4c22540bae632e76fb2087dc0e38eaba8d0c4419ef3`
- result: `results/cards/MC002B/mc002b_gemma2_2b_it_context_support_smoke_chat_20260630T154313.json`
- result SHA256: `4871cf0779883d140fab22ab5c9b260ac0eefb0b5ab437f57d744b2bd5a040a9`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- max new tokens: `12`
- sources: 40 context-supported nonce countries and 40 context-unsupported
  near-neighbor countries
- records: 400
- conditions: `neutral`, `cautious`, `similar_allowed`,
  `closest_required`, `lure_check`

## Question

Can an exact-support reference-note task create a cleaner known-versus-unknown
substrate than pretrained country knowledge?

## Gate Verdict

No.

The reconstructed baseline is materially cleaner than the original MC002
pressure variants: 39 supported sources were clean across `neutral` and
`cautious`, and 35 unsupported near-neighbor sources were clean across
`neutral` and `cautious`. But the pressure transition still failed. No pressure
condition became a mechanism-substrate candidate or a diagnostic prompt
candidate.

Write this as:

> MC002B proves exact in-context support is measurable, but it does not produce
> a reliable abstention-to-near-neighbor hallucination transition.

## Audit Against Preregistration

Baseline:

| Metric | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| clean context-supported sources | 39 | at least 32 | pass |
| clean context-unsupported sources | 35 | at least 32 | pass |
| source split overlap | 0 | 0 | pass |

Pressure arms:

| Condition | Unsupported contrast | Holdout contrast | Supported correct | Supported abstain | Candidate |
| --- | ---: | ---: | ---: | ---: | --- |
| `similar_allowed` | 1 | 0 | 40 | 0 | no |
| `closest_required` | 0 | 0 | 28 | 11 | no |
| `lure_check` | 1 | 0 | 1 | 39 | no |

## Behavior Summary

| Entity Type | Condition | Label Counts |
| --- | --- | --- |
| context supported | `neutral` | `supported_correct`: 40 |
| context supported | `cautious` | `supported_correct`: 39, `abstain`: 1 |
| context supported | `similar_allowed` | `supported_correct`: 40 |
| context supported | `closest_required` | `supported_correct`: 28, `abstain`: 11, `supported_other`: 1 |
| context supported | `lure_check` | `abstain`: 39, `supported_correct`: 1 |
| context unsupported | `neutral` | `abstain`: 35, `near_neighbor_hallucination`: 5 |
| context unsupported | `cautious` | `abstain`: 38, `near_neighbor_hallucination`: 2 |
| context unsupported | `similar_allowed` | `abstain`: 35, `near_neighbor_hallucination`: 5 |
| context unsupported | `closest_required` | `abstain`: 40 |
| context unsupported | `lure_check` | `abstain`: 36, `near_neighbor_hallucination`: 4 |

## Diagnosis

MC002B improves the clean measurement side but not the control surface side.
The model can read and answer exact reference-note facts, and it usually
abstains on unsupported near-neighbor names. That is a better known/unknown
labeling substrate than the original pretrained-country task.

The pressure arms do not move the intended unsupported clean rows. The
`similar_allowed` arm preserves supported answers but creates only one
same-source unsupported contrast. `closest_required` is worse: it creates zero
unsupported contrasts and damages supported locality, with 11 supported rows
abstaining and one supported row outputting the country name instead of the
capital. `lure_check` mostly turns into a broad abstention instruction.

This means the task still lacks the intervention dimension required before
hidden-state discovery. A probe could separate exact supported answers from
unsupported abstentions, but there is no reliable pressure-induced behavioral
transition to steer or patch.

## Decision

Do not start hidden-state signature discovery, steering, patching, sparse
feature search, or circuit localization on MC002B context-support outputs.

The known-unknown line has now produced:

- base-Gemma generation with pressure but no clean unknown baseline;
- base-Gemma chat rendering failure;
- base-Gemma answer-scoring failure;
- Gemma 2 2B IT clean baseline with no pressure contrast;
- Gemma 2 2B IT pressure calibration failure;
- MC002B exact-context reconstruction with clean baseline but no pressure
  transition.

The next mechanism-card attempt should not keep tuning MC002 wording. Either
retire MC002 as a behavior-gated line, or open a new card whose behavior has an
observable transition before hidden-state work begins.
