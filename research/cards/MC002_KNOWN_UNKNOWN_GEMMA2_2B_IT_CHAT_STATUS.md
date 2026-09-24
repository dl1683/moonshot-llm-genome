# MC002 Known-Unknown Gemma 2 2B IT Chat Status

Status: complete. Instruction-tuned Gemma 2 2B chat failed the MC002 behavior
gate by over-abstaining under pressure.

Date: 2026-06-30

## Artifacts

- generation runner: `code/mc002_known_unknown_smoke.py`
- preregistration: `research/prereg/MC002_KNOWN_UNKNOWN_GEMMA2_2B_IT_CHAT.md`
- manifest: `data/cards/MC002/mc002_gemma2_2b_it_known_unknown_chat_manifest.jsonl`
- manifest SHA256: `976e91aa79e58524a46f3357e04f3c849980638bd1d0d0cf3dbc3fcf9dca2aa1`
- result: `results/cards/MC002/mc002_gemma2_2b_it_known_unknown_chat_smoke_chat_20260630T152146.json`
- result SHA256: `2a2d1f058aefe966ba23e58a5f3a752974e53834d25982ecab66220f1b1995ea`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- max new tokens: `12`
- sources: same 40 real-country and 40 nonce-country sources as prior MC002
  runs
- records: 320
- success criteria: identical to the original MC002 behavior gate

## Question

Does an instruction-tuned small Gemma model repair MC002 by preserving real
country answers while abstaining on nonce countries and still hallucinating
under pressure?

## Gate Verdict

No.

Instruction tuning repaired the clean abstention side, but it removed the
pressure-induced hallucination contrast. All 40 nonce sources were clean under
neutral and cautious prompts, but only one nonce source hallucinated under a
pressure condition. The real-country lure locality also failed badly: only 6 of
40 real-country lure rows remained correct, while 32 abstained.

Write this as:

> Gemma 2 2B IT cleanly abstains on nonce countries, but the current MC002
> pressure arm is too weak and too nonlocal for a mechanism substrate.

## Audit Against Preregistration

| Metric | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| clean real-country sources | 36 | at least 24 | pass |
| clean nonce-country sources | 40 | at least 20 | pass |
| nonce pressure-hallucination sources | 1 | at least 12 | fail |
| nonce intra-source contrast sources | 1 | at least 12 | fail |
| nonce contrast holdout sources | 0 | at least 4 | fail |
| real-country lure rows still correct | 6 | at least 20 | fail |
| real-country cautious abstentions | 2 | at most 8 | pass |
| source split overlap | 0 | 0 | pass |

## Behavior Summary

| Entity Type | Condition | Label Counts |
| --- | --- | --- |
| real country | `neutral` | `known_correct`: 37, `abstain`: 1, `known_other_answer`: 1, `known_wrong_lure`: 1 |
| real country | `cautious` | `known_correct`: 36, `abstain`: 2, `known_other_answer`: 2 |
| real country | `forced_answer` | `known_correct`: 39, `known_wrong_lure`: 1 |
| real country | `lure` | `known_correct`: 6, `abstain`: 32, `known_wrong_lure`: 2 |
| nonce country | `neutral` | `abstain`: 40 |
| nonce country | `cautious` | `abstain`: 40 |
| nonce country | `forced_answer` | `abstain`: 40 |
| nonce country | `lure` | `abstain`: 39, `lure_hallucination`: 1 |

## Interface And Model Comparison

| Metric | Base Raw | Base Chat | Base Scoring | Gemma 2 2B IT Chat |
| --- | ---: | ---: | ---: | ---: |
| clean real-country sources | 27 | 6 | 26 | 36 |
| clean nonce-country sources | 1 | 2 | 1 | 40 |
| nonce pressure-hallucination sources | 24 | 39 | 40 | 1 |
| nonce intra-source contrast sources | 7 | 6 | 9 | 1 |
| real-country lure rows still correct | 10 | 0 | 15 | 6 |
| real-country cautious abstentions | 2 | 6 | 2 | 2 |

The instruction-tuned model is much better at following abstention instructions
than base Gemma. That is not enough for MC002. The mechanism substrate needs
both a clean negative class and pressure-induced failures on the same source
family. This run has the clean negative class but almost no pressure failures.

## Diagnosis

MC002 now has two opposite failure modes:

- base Gemma has pressure-induced nonce hallucination but lacks clean nonce
  abstention;
- instruction-tuned Gemma has clean nonce abstention but lacks pressure-induced
  nonce hallucination.

This means the current pressure arms are not calibrated. The `forced_answer`
prompt is too weak against an instruction-tuned refusal/uncertainty policy, and
the `lure` prompt is nonlocal because it causes many real-country rows to
abstain instead of preserving the known correct capital.

## Decision

Do not start hidden-state signature discovery, steering, patching, or sparse
feature search on the Gemma 2 2B IT MC002 chat outputs.

The next MC002 attempt should not change model class again before repairing the
pressure design. A useful V2 task should explicitly tune pressure strength:

- keep neutral/cautious nonce rows as the clean abstention baseline;
- add graded pressure arms that are strong enough to induce some nonce
  hallucination on the instruction-tuned model;
- include real-country locality rows for every pressure arm;
- preregister a lower but nonzero pressure-hallucination floor and a stronger
  real-country locality guard.

Until that calibrated pressure design passes, MC002 remains behavior-gated.
