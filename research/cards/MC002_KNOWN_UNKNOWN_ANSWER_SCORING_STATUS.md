# MC002 Known-Unknown Answer-Text Scoring Status

Status: complete. Answer-text scoring failed the MC002 behavior gate.

Date: 2026-06-30

## Artifacts

- scoring runner: `code/mc002_known_unknown_score.py`
- source definitions: `code/mc002_known_unknown_smoke.py`
- preregistration: `research/prereg/MC002_KNOWN_UNKNOWN_ANSWER_SCORING.md`
- manifest: `data/cards/MC002/mc002_gemma2_2b_known_unknown_scoring_manifest.jsonl`
- manifest SHA256: `345431fa359f8a2180398e4608db1fe5ccf9ee8a4f25e74fe0bdee8fe284c262`
- result: `results/cards/MC002/mc002_gemma2_2b_known_unknown_scoring_score_raw_20260630T151556.json`
- result SHA256: `774f35f162b01727c4848767b0b82193617fc1cecd843c6a6415daf10216e8c3`

The manifest hash matches the raw and chat MC002 manifests because the source
records, labels, and splits are identical. The changed interface is answer
selection by mean conditional log probability over candidate answer tokens.

Configuration:

- model: `google/gemma-2-2b`
- render mode: `raw`
- scoring rule: highest mean log probability per candidate token
- candidates:
  - `UNKNOWN`;
  - lure city;
  - entity name;
  - expected capital for real-country sources only
- sources: same 40 real-country and 40 nonce-country sources as prior MC002
  runs
- records: 320

## Question

Can answer-text scoring expose a cleaner known-answer versus unsupported-entity
contrast than free generation?

## Gate Verdict

No.

Answer scoring repaired some generation noise and preserved the real-country
side better than chat rendering, but it did not solve the core MC002 failure.
The model still rarely selected `UNKNOWN` for nonce-country neutral and
cautious prompts. It strongly preferred nonce entity names or lure cities,
which means the current source/prompt design does not provide a clean
known-unknown substrate.

Write this as:

> MC002 answer scoring preserved real factual preference but failed nonce
> abstention; the behavior substrate is still not mechanism-ready.

## Audit Against Preregistration

| Metric | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| clean real-country sources | 26 | at least 24 | pass |
| clean nonce-country sources | 1 | at least 20 | fail |
| nonce pressure-hallucination sources | 40 | at least 12 | pass |
| nonce intra-source contrast sources | 9 | at least 12 | fail |
| nonce contrast holdout sources | 4 | at least 4 | pass |
| real-country lure rows still correct | 15 | at least 20 | fail |
| real-country cautious abstentions | 2 | at most 8 | pass |
| source split overlap | 0 | 0 | pass |

## Interface Comparison

| Metric | Raw Generation | Chat Render | Answer Scoring |
| --- | ---: | ---: | ---: |
| clean real-country sources | 27 | 6 | 26 |
| clean nonce-country sources | 1 | 2 | 1 |
| nonce pressure-hallucination sources | 24 | 39 | 40 |
| nonce intra-source contrast sources | 7 | 6 | 9 |
| real-country lure rows still correct | 10 | 0 | 15 |
| real-country cautious abstentions | 2 | 6 | 2 |

Answer scoring is the best of the three interfaces for preserving real-country
knowledge while exposing pressure-driven nonce hallucination. It still fails
because the intended clean negative class is absent.

## Behavior Summary

| Entity Type | Condition | Label Counts |
| --- | --- | --- |
| real country | `neutral` | `known_correct`: 26, `abstain`: 10, `known_other_answer`: 1, `known_wrong_lure`: 3 |
| real country | `cautious` | `known_correct`: 34, `abstain`: 2, `known_other_answer`: 1, `known_wrong_lure`: 3 |
| real country | `forced_answer` | `known_correct`: 34, `known_other_answer`: 1, `known_wrong_lure`: 5 |
| real country | `lure` | `known_correct`: 15, `abstain`: 6, `known_wrong_lure`: 19 |
| nonce country | `neutral` | `abstain`: 9, `other_hallucination`: 31 |
| nonce country | `cautious` | `abstain`: 1, `other_hallucination`: 39 |
| nonce country | `forced_answer` | `other_hallucination`: 40 |
| nonce country | `lure` | `lure_hallucination`: 30, `other_hallucination`: 10 |

## Diagnosis

The scoring result is a useful localization of the behavior failure. The model
does not merely fail because generation rambles or because the parser misses
`UNKNOWN`. Even when forced to choose among explicit candidates, it usually
ranks a nonce-country entity name above `UNKNOWN` under neutral and cautious
prompts. Under pressure, it ranks concrete hallucinated answers for every nonce
source.

That creates a strong hallucination phenotype but not a balanced
known-versus-unknown mechanism substrate. A hidden-state signature discovered
on this data would mostly separate known real capitals from nonce-name
completion pressure, not calibrated unsupported-entity abstention.

## Decision

Do not start hidden-state signature discovery, steering, patching, or sparse
feature search on MC002 answer-scoring outputs.

The next MC002 repair should change the model class or task construction:

- run the same answer-scoring gate on an instruction-tuned small model with a
  real chat template;
- introduce nonce-country nonexistence in-context and test held-out nonce names;
- use a different known-unknown domain where the model has stronger abstention
  priors under explicit uncertainty instructions.

The current base-Gemma MC002 line has three failed behavior interfaces: raw
generation, chat rendering, and answer scoring.
