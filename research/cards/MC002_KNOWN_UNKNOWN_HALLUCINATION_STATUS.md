# MC002 Known-Unknown Hallucination Status

Status: complete. The first known-vs-unknown hallucination substrate failed the
preregistered behavior gate.

Date: 2026-06-30

## Artifacts

- generation runner: `code/mc002_known_unknown_smoke.py`
- preregistration: `research/prereg/MC002_KNOWN_UNKNOWN_HALLUCINATION.md`
- manifest: `data/cards/MC002/mc002_gemma2_2b_known_unknown_manifest.jsonl`
- manifest SHA256: `345431fa359f8a2180398e4608db1fe5ccf9ee8a4f25e74fe0bdee8fe284c262`
- result: `results/cards/MC002/mc002_gemma2_2b_known_unknown_smoke_raw_20260630T150312.json`
- result SHA256: `30fbbb63fcac8e44381e213c3c4998d6bf0a319bfbb8ff1c3fc6e6ea94c3c93f`
- discarded diagnostic run: `results/cards/MC002/mc002_gemma2_2b_known_unknown_smoke_raw_20260630T150033.json`

The discarded run used the same manifest but an earlier classifier that checked
abstention before expected-answer matches on real-country rows and treated
repeated prompt markers as concrete behavior. It was used only to identify and
fix the classifier. The evidence-bearing run is the `20260630T150312` result.

Configuration:

- model: `google/gemma-2-2b`
- render mode: `raw`
- max new tokens: `12`
- sources: 40 real-country capital questions and 40 nonce-country capital
  questions
- conditions: `neutral`, `cautious`, `forced_answer`, `lure`
- records: 320
- split rule: deterministic source split, sorted by source ID, every third
  source assigned to holdout

## Question

Can the next behavior family produce a clean enough known-answer versus
unsupported-entity contrast to justify hidden-state signature discovery?

## Gate Verdict

No.

The run has useful raw behavior, but not a clean substrate. Real-country
answering is adequate, and lure prompts can induce fabricated nonce-country
capitals. The failure is that nonce-country neutral and cautious prompts do not
reliably abstain. Only 1 of 40 nonce sources was clean under both neutral and
cautious rows.

Write this as:

> MC002 found a hallucination-prone nonce-country interface, but failed the
> known-unknown abstention substrate gate.

## Structural Checks

The manifest passed the structural checks:

| Metric | Result |
| --- | ---: |
| sources | 80 |
| records | 320 |
| real-country sources | 40 |
| nonce-country sources | 40 |
| records per condition | 80 |
| source split overlap | 0 |
| duplicate record IDs | 0 |
| missing expected/lure answers | 0 |

## Behavior Summary

| Entity Type | Condition | Label Counts |
| --- | --- | --- |
| real country | `neutral` | `known_correct`: 29, `abstain`: 4, `known_other_answer`: 5, `known_wrong_lure`: 1, `unparseable`: 1 |
| real country | `cautious` | `known_correct`: 33, `abstain`: 2, `known_other_answer`: 1, `known_wrong_lure`: 2, `unparseable`: 2 |
| real country | `forced_answer` | `known_correct`: 35, `known_other_answer`: 1, `known_wrong_lure`: 1, `unparseable`: 3 |
| real country | `lure` | `known_correct`: 10, `known_wrong_lure`: 23, `unparseable`: 7 |
| nonce country | `neutral` | `abstain`: 11, `other_hallucination`: 28, `unparseable`: 1 |
| nonce country | `cautious` | `abstain`: 1, `other_hallucination`: 37, `unparseable`: 2 |
| nonce country | `forced_answer` | `abstain`: 27, `other_hallucination`: 7, `unparseable`: 6 |
| nonce country | `lure` | `abstain`: 11, `lure_hallucination`: 18, `other_hallucination`: 2, `unparseable`: 9 |

The surprising behavior is condition-dependent. The `forced_answer` prompt
caused more nonce abstention than the `cautious` prompt. The `lure` condition
worked as a pressure arm for some nonce sources, but the same lure condition
also corrupted real-country locality: only 10 of 40 real-country lure rows
remained correct, while 23 followed the wrong lure.

## Audit Against Preregistration

| Metric | Result | Preregistered Bar | Verdict |
| --- | ---: | ---: | --- |
| clean real-country sources | 27 | at least 24 | pass |
| clean nonce-country sources | 1 | at least 20 | fail |
| nonce pressure-hallucination sources | 24 | at least 12 | pass |
| nonce intra-source contrast sources | 7 | at least 12 | fail |
| nonce contrast holdout sources | 4 | at least 4 | pass |
| real-country lure rows still correct | 10 | at least 20 | fail |
| real-country cautious abstentions | 2 | at most 8 | pass |
| source split overlap | 0 | 0 | pass |

The gate failed three criteria:

- neutral/cautious nonce abstention is far too weak;
- there are too few same-source abstention-versus-hallucination contrasts;
- the lure condition is not local, because it often overrides real known
  capitals.

## Diagnosis

This is not a no-signal result. The model frequently treats nonce countries as
if they exist under neutral and cautious prompts, and the lure prompt often
pushes it to emit the suggested fake capital. That is a usable hallucination
phenotype.

It is not a mechanism substrate because the intended clean negative class is
missing. Without reliable abstention on unsupported nonce entities, a hidden
signature would likely separate prompt quirks, completion templates, or
entity-name continuation behavior rather than known-versus-unknown factual
status. The real-country lure failures also show that the pressure arm is not
local enough for a future intervention locality test.

## Decision

Do not start hidden-state signature discovery, activation steering, sparse
feature search, or patching on this MC002 substrate.

The next MC002 attempt should repair the behavior interface before touching
internals. Plausible repairs:

- use an instruction-tuned model or chat render for the known-unknown task;
- replace free generation with calibrated answer-text scoring between
  `UNKNOWN` and the candidate city;
- use a source bank where fake entities are explicitly introduced as candidate
  non-entities in-context, then test held-out nonce entities;
- separate pressure lures from real-country locality rows with matched
  prompt-only controls.

Until one of those passes the behavior gate, MC002 is a failed substrate, not a
mechanism-card branch.
