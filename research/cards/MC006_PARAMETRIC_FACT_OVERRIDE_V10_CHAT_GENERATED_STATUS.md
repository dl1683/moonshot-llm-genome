# MC006 Parametric Fact Override V10 Chat-Generated Status

Status: chat-rendered generated-answer behavior substrate passed; hidden-state
signature discovery may proceed on this table, but no mechanism or intervention
claim is made.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V10_CHAT_GENERATED.md`
- runner:
  `code/mc006_parametric_fact_override_v10_chat_generated.py`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v10_chat_generated_20260630T232318.json`
- result SHA256:
  `7c3054588e876a16338be209666ea7245bead81c1a1fd1040e2ab114e522acb7`

## Verdict

MC006 V10 passes the preregistered generated-answer behavior-substrate gate.
Changing the rendering contract to Qwen3 chat template generation with a strict
one-city system instruction recovered both real-world capital recall and
task-local fictional-code following on the same prompt family.

The diagnostic class is:

```text
chat_generated_v10_substrate_passed
```

This is still only Gate 0 for the MC006 mechanism path. It proves a reliable
behavior table suitable for hidden-state signature discovery. It does not prove
an internal signature, a causal intervention, locality, or mechanism reliability.

## Structural Checks

All structural checks passed:

- tokenizer chat template available;
- rendering with `enable_thinking=False` succeeded;
- exactly 22 sources;
- exactly 5 original holdout sources;
- exactly 44 generated rows;
- two conditions per source;
- 22 rows per condition;
- 22 rows per requested mode;
- no duplicate record ids;
- no duplicate candidate answers;
- no true-capital prompt leaks.

## Behavior Results

| Criterion | Result |
| --- | --- |
| strict parseable rows at least 40/44 | pass: 44/44 |
| `mode_real` true answer at least 20/22 | pass: 20/22 |
| `mode_fictional` override answer at least 20/22 | pass: 22/22 |
| clean source-level contrasts at least 18/22 | pass: 20/22 |
| holdout clean contrasts at least 4/5 | pass: 5/5 |
| no true-capital prompt leaks | pass |

The two non-clean source contrasts were:

- Brazil: `mode_real` generated the lure city `Rio de Janeiro`, while
  `mode_fictional` generated the override `Recife`.
- Croatia: both modes generated the override `Zadar`.

No generated row was unparsed under the strict parser.

## Interpretation

V10 shows that the V8/V9 failure was not only a parser problem. The decisive
repair was a stronger output/rendering contract: chat rendering plus a
system-level one-city instruction restored the real-world side while preserving
fictional-code following.

MC006 should now be bounded as:

- V4: narrow prompt-bounded behavior substrate passed;
- V5: hidden signature present but output/prompt/null-confounded;
- V6: same-prompt-family two-source repair failed real-source reliability;
- V7: same-prompt-family mode-gated candidate scoring failed real-world mode;
- V8: same-prompt-family generated-answer repair failed parseability and
  real-world mode;
- V9: lenient parser improved V8 but still failed real-world mode and contrasts;
- V10: chat-rendered generated-answer behavior substrate passed.

The next MC006 step may start hidden-state signature discovery only on the V10
table, with requested-mode, output-text/logit margin, prompt length, source
split, shuffled-label, and prompt-rendering controls carried forward.
