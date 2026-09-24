# MC002 Pressure Calibration Status

Status: complete. Pressure calibration failed to produce a mechanism-substrate
candidate.

Date: 2026-06-30

## Artifacts

- runner: `code/mc002_pressure_calibration.py`
- source definitions and classifier: `code/mc002_known_unknown_smoke.py`
- preregistration: `research/prereg/MC002_PRESSURE_CALIBRATION.md`
- manifest: `data/cards/MC002/mc002_gemma2_2b_it_pressure_calibration_manifest.jsonl`
- manifest SHA256: `8e1cbe81965f2fb21a8ca8d71ff007b1deed295a1601548f300e3590534fdc99`
- result: `results/cards/MC002/mc002_gemma2_2b_it_pressure_calibration_calibration_chat_20260630T153139.json`
- result SHA256: `54849f04fd72b19a3aa1d0ee37004494266d7f38004571dc072574d3927cdd77`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- max new tokens: `12`
- sources: same 40 real-country and 40 nonce-country sources as prior MC002
  runs
- records: 560
- conditions: `neutral`, `cautious`, `guess_mild`, `guess_strong`,
  `city_required`, `lure_soft`, `lure_strong`

## Question

Can graded pressure create same-source nonce abstention-versus-hallucination
contrasts on instruction-tuned Gemma while preserving real-country answers?

## Gate Verdict

No.

The clean baseline held: 36 real-country sources were clean across `neutral`
and `cautious`, and all 40 nonce-country sources were clean across `neutral`
and `cautious`. The pressure arms did not create a viable contrast. No
condition became a mechanism-substrate candidate, and no condition even became
a diagnostic prompt candidate under the preregistered audit.

Write this as:

> Gemma 2 2B IT has a clean known/unknown abstention baseline on MC002, but
> simple prompt pressure does not break that baseline into a reliable,
> local contrast.

## Audit Against Preregistration

Baseline:

| Metric | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| baseline clean real-country sources | 36 | at least 32 | pass |
| baseline clean nonce-country sources | 40 | at least 32 | pass |
| source split overlap | 0 | 0 | pass |

Pressure arms:

| Condition | Allows `UNKNOWN` | Uses lure | Nonce contrast | Holdout contrast | Real correct | Real abstain | Real lure errors | Candidate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `guess_mild` | yes | no | 0 | 0 | 37 | 0 | 2 | no |
| `guess_strong` | yes | no | 0 | 0 | 39 | 0 | 1 | no |
| `city_required` | no | no | 1 | 0 | 40 | 0 | 0 | no |
| `lure_soft` | yes | yes | 1 | 0 | 6 | 32 | 2 | no |
| `lure_strong` | no | yes | 6 | 1 | 29 | 5 | 6 | no |

## Behavior Summary

| Entity Type | Condition | Label Counts |
| --- | --- | --- |
| real country | `neutral` | `known_correct`: 37, `abstain`: 1, `known_other_answer`: 1, `known_wrong_lure`: 1 |
| real country | `cautious` | `known_correct`: 36, `abstain`: 2, `known_other_answer`: 2 |
| real country | `guess_mild` | `known_correct`: 37, `known_other_answer`: 1, `known_wrong_lure`: 2 |
| real country | `guess_strong` | `known_correct`: 39, `known_wrong_lure`: 1 |
| real country | `city_required` | `known_correct`: 40 |
| real country | `lure_soft` | `abstain`: 32, `known_correct`: 6, `known_wrong_lure`: 2 |
| real country | `lure_strong` | `known_correct`: 29, `known_wrong_lure`: 6, `abstain`: 5 |
| nonce country | `neutral` | `abstain`: 40 |
| nonce country | `cautious` | `abstain`: 40 |
| nonce country | `guess_mild` | `abstain`: 40 |
| nonce country | `guess_strong` | `abstain`: 40 |
| nonce country | `city_required` | `abstain`: 39, `other_hallucination`: 1 |
| nonce country | `lure_soft` | `abstain`: 39, `lure_hallucination`: 1 |
| nonce country | `lure_strong` | `abstain`: 34, `lure_hallucination`: 6 |

## Diagnosis

The instruction-tuned model's abstention policy is robust on this source bank.
Both `guess_mild` and `guess_strong` preserved real-country answers but
produced zero nonce-country hallucinations. Even `city_required`, which
explicitly forbade `UNKNOWN`, produced only one nonce hallucination and 39
nonce abstentions.

The lure arms are still not useful. `lure_soft` nearly reproduces the original
nonlocal failure: only 6 real-country rows remain correct and 32 abstain.
`lure_strong` induces more nonce lure hallucination, but only 6 sources
contrast, only 1 is holdout, real-country correctness falls below the
preregistered floor, and real lure errors exceed the guardrail.

This is not a hidden-state discovery substrate. A signature discovered here
would mostly explain a strong instruction-following refusal boundary, not a
calibrated known-versus-unknown pressure transition.

## Decision

Do not start hidden-state signature discovery, steering, patching, sparse
feature search, or circuit localization on MC002 pressure-calibration outputs.

The next MC002 move should change task construction rather than keep tuning
single-turn pressure wording. Plausible directions:

- use an in-context known/unsupported setup where some novel entities are
  explicitly introduced and held-out nonce entities remain unsupported;
- use a different known-unknown domain with stronger learned priors and cleaner
  abstention pressure;
- score explicit answer candidates under calibrated prompts only after the
  generation surface has a clean held-out contrast;
- retire MC002 if the reconstructed task still cannot satisfy the behavior
  gate.

Until a reconstructed task passes a behavior gate with locality controls,
MC002 remains behavior-gated.
