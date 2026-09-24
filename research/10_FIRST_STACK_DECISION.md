# First Stack Decision

## Decision

Primary implementation target:

> `Qwen/Qwen3-0.6B`

Reason:

The user selected Qwen3-0.6B because it is the smallest Qwen3 model and should allow the most iteration cycles. This overrides the earlier Gemma-first design.

Artifact-rich comparison target:

> `google/gemma-3-4b-it`

Escalation target if Qwen3-0.6B is too weak:

> `Qwen/Qwen3-1.7B` or another small Qwen3 checkpoint, chosen only after the 0.6B smoke result.

## Why Qwen3-0.6B First

The first execution goal is not a final mechanism card. It is rapid iteration:

- check prompt formatting;
- check parseability;
- measure no-hint accuracy;
- measure wrong-hint compliance;
- verify activation access later;
- iterate cheaply before spending analysis time.

Qwen3-0.6B may be too small for a final claim, but that is exactly why it is useful first. If the behavior does not exist at 0.6B, we learn quickly. If it does exist, it becomes a cheap substrate for many rounds of controls.

## Tradeoff Versus Gemma

The earlier Gemma plan was artifact-rich because Gemma Scope 2 provides sparse autoencoders and transcoders. Qwen3-0.6B is iteration-rich instead.

This changes the method order:

1. behavior smoke;
2. output/logit baselines;
3. hidden-state probes and activation directions;
4. activation steering;
5. only then compare to artifact-rich Gemma or larger Qwen if needed.

Do not wait for SAE artifacts to begin. For Qwen3-0.6B, start with dense hidden states, probes, and directions.

## Tooling Direction

Use simple local tooling first:

- Hugging Face `transformers`;
- deterministic greedy generation;
- compact answer parsing;
- JSONL prompt manifest;
- JSON result artifact;
- later: hidden-state extraction via `output_hidden_states=True` or forward hooks.

## Promotion Rules

Continue with Qwen3-0.6B if:

- no-hint accuracy is high enough;
- wrong-hint compliance is measurable;
- completions are parseable;
- smoke runtime is low enough to iterate;
- behavior varies across pressure conditions.

Escalate from Qwen3-0.6B if:

- no-hint accuracy is too low for wrong-hint errors to mean anything;
- wrong-hint compliance is below 10 percent or above 90 percent after reasonable prompt pressure;
- answers are not parseable;
- hidden-state access is awkward enough to erase the iteration advantage.

Keep Gemma 3 4B-IT as a later comparison if sparse-feature artifacts become important.

## Smoke Result

Qwen3-0.6B passed the iteration gate on the MC-001 factual-ladder substrate:

- factual-ladder behavior smoke: `results/cards/MC001/mc001_qwen3_0p6b_smoke_factual_ladder_20260629T193212.json`;
- hidden probe: `results/cards/MC001/mc001_qwen3_0p6b_probe_factual_ladder_20260629T193342.json`;
- logit steering sweep: `results/cards/MC001/mc001_qwen3_0p6b_logit_steer_factual_ladder_20260629T194016.json`;
- generation steering validation: `results/cards/MC001/mc001_qwen3_0p6b_steer_factual_ladder_20260629T194456.json`;
- live summary: [MC-001 Qwen3-0.6B Smoke Status](cards/MC001_QWEN3_0P6B_SMOKE_STATUS.md).

Observed steering is promising but not clean: hidden index 14, alpha 1 reduced agreement errors on the eval split while introducing other errors and some format drift. Treat this as a discovery/calibration target, not a finished mechanism claim.

## Controlled Result

Qwen3-0.6B remains the right iteration substrate, but the first controlled pass did not produce a supported mechanism card.

- controlled runner: `code/mc001_qwen3_controlled.py`
- controlled result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_20260629T200930.json`
- controlled status: [MC-001 Qwen3-0.6B Controlled Status](cards/MC001_QWEN3_0P6B_CONTROLLED_STATUS.md)

The useful controlled finding is that hidden index 14 with alpha 1 improves standard-holdout wrong-hint truth-following from 13/60 to 36/60, while clean nulls do not reproduce that effect. The limiting finding is stronger: answer-logit margin predicts agreement-vs-truth with AUC 1.0 on calibration, holdout, and paraphrase holdout, so the current hidden signature does not beat the output baseline.

Next stack decision at V10 time:

- continue Qwen3-0.6B for one more controlled iteration;
- condition on or beat output-logit margin before claiming hidden mechanism value;
- escalate only if the smaller model cannot produce a residual hidden control signal beyond output logits.

## Controlled V2 Result

The output-margin-conditioned v2 pass also did not produce a supported mechanism card.

- controlled v2 runner: `code/mc001_qwen3_controlled_v2.py`
- controlled v2 result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v2_20260629T203703.json`
- controlled v2 status: [MC-001 Qwen3-0.6B Controlled V2 Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V2_STATUS.md)

V2 showed that the raw h14 alpha 1 intervention improves holdout and paraphrase wrong-hint truth-following, but output margin still reaches AUC 1.0 and matched/nearby controls expose answer-token side effects. The next stack decision is still Qwen3-0.6B, but only for a narrower v3 pass: lower doses and answer-letter/output-margin residualized directions. Escalate only after that residual-control question is answered.

## Controlled V3 Result

The lower-dose and residualized v3 pass answered the residual-control question and still did not produce a supported mechanism card.

- controlled v3 runner: `code/mc001_qwen3_controlled_v3.py`
- controlled v3 result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v3_20260629T205927.json`
- controlled v3 status: [MC-001 Qwen3-0.6B Controlled V3 Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V3_STATUS.md)

V3 showed a real raw control surface: h14 `alpha=0.50` raises wrong-hint truth-following from 24/60 to 36/60 on calibration, 19/60 to 36/60 on holdout, and 18/60 to 30/60 on paraphrase holdout. It also showed why that is still not a mechanism card: output margin remains AUC 1.0, residualized h14 loses most of the signal, random directions create answer-token artifacts, and nearby-layer controls stay active.

Next stack decision:

- keep Qwen3-0.6B long enough to write the diagnostic/control-surface card;
- do not run another broad global dense-direction sweep;
- if continuing the model, change method to output-margin-bin-conditioned, answer-letter-paired patching or path-local interventions;
- escalate to Qwen3-1.7B or Gemma only after the method-shift result says the 0.6B substrate cannot expose a cleaner residual mechanism.

## Controlled V4 Patch Result

The first method-shift patching pass also did not produce a supported mechanism card.

- controlled v4 patch runner: `code/mc001_qwen3_controlled_v4_patch.py`
- controlled v4 patch result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v4_patch_20260629T212411.json`
- controlled v4 patch status: [MC-001 Qwen3-0.6B Controlled V4 Patch Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V4_PATCH_STATUS.md)

V4 selected the 104 hardest agreement-favored wrong-hint rows, where baseline truth-following was 0/104. Same-question no-hint h14 half-patching moved truth-following only to 4/104, and correct-hint h14 half-patching only to 2/104. Full hidden-state replacement at h7, h13, and h14 collapsed parseability.

Next stack decision:

- Qwen3-0.6B has produced a real control surface but not a supported hidden mechanism;
- do not run another final-prompt-token donor-replacement patch;
- do not interpret v4 full-replacement collapse alone as final evidence, because v4 patched every generation forward pass;
- continue only with prompt-prefill-only additive controls or answer-prefix/path-local methods.

## Controlled V5 Prefill Result

The prompt-prefill audit also did not produce a supported mechanism card, but it kept Qwen3-0.6B alive for one narrower iteration.

- controlled v5 prefill runner: `code/mc001_qwen3_controlled_v5_prefill_audit.py`
- controlled v5 prefill result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v5_prefill_20260629T213701.json`
- controlled v5 prefill status: [MC-001 Qwen3-0.6B Controlled V5 Prefill Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V5_PREFILL_STATUS.md)

V5 selected the same 104 hardest agreement-favored wrong-hint rows, where baseline truth-following was 0/104. Raw h14 all-step `alpha=0.50` reproduced the v3 hard-bin effect at 30/104 truth-following. Raw h14 prompt-prefill-only `alpha=0.50` still moved truth-following to 21/104 with full parseability, and prompt-prefill-only `alpha=1.00` moved truth-following to 41/95 parseable outputs. Same-question donor prefill replacement remained near-baseline.

Next stack decision:

- run one more narrow additive prefill audit;
- require matched random and nearby additive prefill controls;
- require no-hint and correct-hint side-effect rows, not only the hard wrong-hint subset;
- close Qwen3-0.6B dense-prefill steering as control-only if that pass cannot separate raw h14 from additive nulls with bounded side effects.

## Controlled V6 Prefill Controls Result

The matched additive prompt-prefill controls closed the dense-prefill mechanism path for Qwen3-0.6B.

- controlled v6 prefill controls runner: `code/mc001_qwen3_controlled_v6_prefill_controls.py`
- controlled v6 prefill controls result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v6_prefill_controls_20260629T220249.json`
- controlled v6 prefill controls status: [MC-001 Qwen3-0.6B Controlled V6 Prefill Controls Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V6_PREFILL_CONTROLS_STATUS.md)

V6 confirmed the positive control surface: on the 104 hardest agreement-favored wrong-hint rows, raw h14 prompt-prefill `alpha=0.50` moved truth-following from 0/104 to 21/104, while wrong-token prefill stayed at 0/104 and matched random `alpha=0.50` reached only 3/104. On all wrong-hint rows, raw h14 prefill moved truth-following from 61/180 to 86/180.

V6 also supplied the decisive negative control. The same h14 vector applied at nearby h13 moved the hard bin to 25/104 at `alpha=0.50`, and to 42/86 parseable at `alpha=1.00`, matching or exceeding the h14 target arm. Same-layer residualized h14 stayed near baseline, while residual h14 applied at h13 remained active. That is not an h14-local mechanism.

Next stack decision:

- do not run another dense final-token or prompt-prefill direction sweep on Qwen3-0.6B for MC-001;
- keep the Qwen3-0.6B result as a control-only failed-mechanism artifact;
- run the cross-layer transport audit explaining h13/h14 interchangeability before any answer-prefix or path-local pass;
- otherwise escalate to Qwen3-1.7B or an artifact-rich Gemma stack after accepting the 0.6B dense-prefill failure.

## Controlled V7 Transport Result

The cross-layer logit-transport audit explained the V6 h13/h14 interchangeability and closed residual-stream dense transport as a mechanism path.

- controlled v7 transport runner: `code/mc001_qwen3_controlled_v7_transport.py`
- controlled v7 transport result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v7_transport_20260629T221738.json`
- controlled v7 transport status: [MC-001 Qwen3-0.6B Controlled V7 Transport Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V7_TRANSPORT_STATUS.md)

V7 measured next-token option logits rather than generation completions. On the hard agreement-favored wrong-hint bin, raw h14 at h13 and h14 produced nearly identical margin-shift patterns: delta Pearson 0.970, mean deltas +4.435 versus +3.964, and hard-bin truth labels 14/104 versus 13/104. At `alpha=1.00`, h13 again exceeded h14: 35/104 truth labels versus 22/104.

The early h7 injection produced a larger hard-bin logit change, but it was broad disruption rather than useful control: hard-bin truth rose to 24/104, while all-validation `other_error` rose to 140/288 and no/correct truth fell from 54/72 to 20/72. h20 was weak. Wrong-token application stayed near baseline.

Next stack decision:

- close Qwen3-0.6B dense direction, dense prefill, and cross-layer transport for MC-001 mechanism-card purposes;
- do not run another residual-stream dense vector sweep on this model;
- at V7 time, continue on Qwen3-0.6B only for answer-prefix signatures or true path-local attribution/ablation;
- otherwise escalate to Qwen3-1.7B or an artifact-rich Gemma stack.

## Controlled V8 Answer-Prefix Result

The answer-prefix dense-signature pass closed the remaining cheap dense 0.6B route.

- controlled v8 answer-prefix runner: `code/mc001_qwen3_controlled_v8_answer_prefix.py`
- controlled v8 answer-prefix result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v8_answer_prefix_20260629T223039.json`
- controlled v8 answer-prefix status: [MC-001 Qwen3-0.6B Controlled V8 Answer-Prefix Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V8_ANSWER_PREFIX_STATUS.md)

V8 added a neutral `Answer: ` prefix, trained a separate answer-prefix h14 direction, and intervened during answer-prefix prefill. The prefix itself shifted behavior: validation wrong-hint baseline truth-following rose to 83/180 with full parseability. The target intervention did not improve that. Raw h14 at h14 `alpha=0.50` produced 70/153 parseable wrong-hint truth-following, and raw h14 at h13 produced 69/143. On the answer-prefix agreement-favored bin, baseline was 7/95 truth, target raw h14 was only 9/81 parseable truth, and residual random was 10/95 with full parseability.

Next stack decision:

- close Qwen3-0.6B dense direction, dense prefill, cross-layer transport, and dense answer-prefix steering for MC-001 mechanism-card purposes;
- do not run another broad residual-stream vector sweep on this model;
- continue on Qwen3-0.6B only for true path-local attribution from hint tokens to answer tokens or attention/MLP-path ablation;
- otherwise escalate to Qwen3-1.7B or an artifact-rich Gemma stack with the 0.6B result recorded as control-only.

## Controlled V9 Attention-Source Result

The attention-source ablation pass produced the first positive path-local evidence, but still not a supported mechanism card.

- controlled v9 attention-source runner: `code/mc001_qwen3_controlled_v9_attention_source_ablation.py`
- controlled v9 attention-source result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v9_attention_source_ablation_20260629T224234.json`
- controlled v9 attention-source status: [MC-001 Qwen3-0.6B Controlled V9 Attention-Source Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V9_ATTENTION_SOURCE_STATUS.md)

V9 zeroed attention to source-token spans during generation. On all wrong-hint rows, hint-line masking moved truth-following from 61/180 to 124/180 and reduced agreement from 101/180 to 16/180. Masking only the hinted answer letter moved truth-following to 110/180. On the hardest agreement-favored wrong-hint bin, baseline truth-following was 0/104, hint-line masking reached 63/104, and hint-answer masking reached 50/104. Matched answer-instruction masking stayed near baseline at 4/104, and matched random source masking reached only 5/103 parseable truth-following.

The limitation is precision. Hint-line masking raised all-validation `other_error` to 65/288, and hint-answer masking raised it to 72/288. This is strong source-token causal evidence, not a localized mechanism. It does not identify the attention head, layer, value path, MLP path, or feature carrying the hint effect.

Next stack decision:

- keep Qwen3-0.6B for one narrower head/layer path-local ablation pass;
- use the agreement-favored wrong-hint bin as the primary target;
- localize hint-answer and non-answer hint-token influence during answer-token generation;
- keep answer-instruction and random source masks as matched nulls;
- escalate only after the head/layer localization question is answered or the 0.6B substrate proves too coarse.

## Controlled V10 Head-Localization Result

The single-head and single-layer localization pass did not produce a mechanism candidate.

- controlled v10 head-localization runner: `code/mc001_qwen3_controlled_v10_head_localization.py`
- controlled v10 head-localization result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v10_head_localization_20260629T231146.json`
- controlled v10 head-localization status: [MC-001 Qwen3-0.6B Controlled V10 Head-Localization Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V10_HEAD_LOCALIZATION_STATUS.md)

V10 used eager attention and targeted source-position masks to screen all 28 layers and 16 query heads. On the hard agreement-favored wrong-hint bin, the best individual head was `hint_answer` layer 22 head 9: baseline truth-following was 0/103 and the mask moved it to 7/103 while reducing agreement from 99/103 to 84/103 and increasing other errors from 4/103 to 12/103. The best all-head layer masks were similarly weak: `hint_answer` layer 22 reached 5/103 hard-bin truth and `hint_line` layer 22 reached 5/103.

Matched controls block promotion. `answer_instruction_matched_hint_answer` layer 21 all-head masking reached 3/103 hard-bin truth, while random source masks stayed at 0/103. The target signal is more real than random, but far too small to explain V9's coarse source-token effect.

Next stack decision:

- keep Qwen3-0.6B for one cumulative layer-band source-mask pass, now completed by V11;
- test whether late or top-k layer bands recover the V9 hint-source effect better than isolated layers/heads;
- keep answer-instruction and random source controls on the same bands;
- if the effect needs many layers or produces broad side effects, close Qwen3-0.6B attention-source localization as distributed control-only and escalate to Qwen3-1.7B or an artifact-rich Gemma stack.

## Controlled V11 Layer-Band Source Result

The cumulative layer-band source pass also did not produce a mechanism candidate.

- controlled v11 layer-band source runner: `code/mc001_qwen3_controlled_v11_layer_band_source.py`
- controlled v11 layer-band source result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v11_layer_band_source_20260629T233533.json`
- controlled v11 layer-band source status: [MC-001 Qwen3-0.6B Controlled V11 Layer-Band Source Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V11_LAYER_BAND_SOURCE_STATUS.md)

V11 used the V10 eager-attention hook and applied source masks across cumulative all-head layer bands. On the hard agreement-favored wrong-hint bin, baseline was again 0/103 truth, 99/103 agreement, and 4/103 other. The strongest precise generation-query mask was the full hint line across all layers: `hint_line__all_L00_L27` reached 24/103 truth, 57/103 agreement, and 22/103 other. The strongest hinted-answer mask, `hint_answer__all_L00_L27`, reached 22/103 truth, 57/103 agreement, and 24/103 other. Broad late bands recovered much of this weak precise effect: `hint_line__band_L17_L22` reached 21/103, and `hint_answer__band_L17_L22` reached 19/103.

Controls keep the result control-only. Random matched source masks stayed near baseline at 0-1/103 hard-bin truth, so the source-token effect is real. But answer-instruction matched masks were not zero: `answer_instruction_matched_hint_answer__all_L00_L27` and `band_L17_L22` both reached 15/103 hard-bin truth. More importantly, the strongest V11 precise masks are far below V9's coarse input-mask result, where hint-line masking reached 63/104 and hint-answer masking reached 50/104.

Next stack decision at V11 time:

- close cumulative generation-query source masking as a Qwen3-0.6B mechanism-card route;
- do not rerun dense residual-stream, dense prefill, answer-prefix dense, single-head, or cumulative layer-band source searches on the same semantics;
- run one input-mask semantics audit on Qwen3-0.6B to decompose the V9 effect into source deletion, prompt-state recomputation, placeholder replacement, positional/layout disruption, and direct generation-query attention, now completed by V12;
- if that audit shows only broad prompt-state/source-removal dependence, keep the 0.6B result as diagnostic/control-only and escalate to Qwen3-1.7B or artifact-rich Gemma for a cleaner mechanism attempt.

## Controlled V12 Input-Mask Semantics Result

The input-mask semantics audit closes the broad Qwen3-0.6B MC-001 route as diagnostic/control-only.

- controlled v12 input-mask semantics runner: `code/mc001_qwen3_controlled_v12_input_mask_semantics.py`
- controlled v12 input-mask semantics result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v12_input_mask_semantics_20260629T235517.json`
- controlled v12 input-mask semantics status: [MC-001 Qwen3-0.6B Controlled V12 Input-Mask Semantics Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V12_INPUT_MASK_SEMANTICS_STATUS.md)

V12 reproduced V9-style coarse input-mask behavior under the V10/V11 eager-attention baseline: on the hard agreement-favored wrong-hint bin, `input_mask_hint_line` moved truth-following from 0/103 to 63/103, and `input_mask_hint_answer` moved it to 49/103. Literal prompt rewrites nearly matched those effects. `rewrite_neutralize_hint_line` reached 60/103, `rewrite_delete_hint_line` reached 57/103, `rewrite_placeholder_hint_answer_X` reached 51/103, and `rewrite_delete_hint_answer` reached 48/103. By contrast, V11-style generation-query masks remained much weaker: `query_mask_hint_line_all` reached 24/103 and `query_mask_hint_answer_all` reached 22/103.

Controls keep this out of mechanism-card territory. Random and answer-instruction matched input masks stayed far below the primary source arms, so the result is source-specific. But hint non-answer text alone still reached 44/103, source-removal arms raised other errors, and no-hint/correct-hint side rows lost truth relative to baseline. The operation changes or removes prompt source content before prompt-state formation; it does not identify a compact residual, attention, or feature mechanism.

Next stack decision at V12 time:

- close broad Qwen3-0.6B MC-001 mechanism search as a useful failed-mechanism and diagnostic/control-surface result;
- write or maintain the Qwen3-0.6B diagnostic/control artifact from the completed v1-v13 series;
- use Qwen3-1.7B or an artifact-rich Gemma stack for the next mechanism-card attempt;
- V13 completed the length-matched neutral-placeholder sanity check for layout/tokenization parity.

## Controlled V13 Layout-Parity Result

The optional layout/tokenization sanity check is complete and does not reopen Qwen3-0.6B.

- controlled v13 layout-parity runner: `code/mc001_qwen3_controlled_v13_layout_parity.py`
- controlled v13 layout-parity result: `results/cards/MC001/mc001_qwen3_0p6b_controlled_v13_layout_parity_20260630T000706.json`
- controlled v13 layout-parity status: [MC-001 Qwen3-0.6B Controlled V13 Layout-Parity Status](cards/MC001_QWEN3_0P6B_CONTROLLED_V13_LAYOUT_PARITY_STATUS.md)

V13 compared the V12 neutral/deletion rewrites against neutral replacements matched for character length or rendered prompt token count. On the hard agreement-favored wrong-hint bin, `input_mask_hint_line` again reached 63/103 truth-following, the V12 references reached 60/103 for `rewrite_neutralize_hint_line` and 57/103 for `rewrite_delete_hint_line`, while both matched neutral variants reached 52/103. The query-only reference stayed at 24/103.

The token-matched neutral arm exactly preserved rendered prompt token count on every rewritten row, so token-count layout does not explain the main V9/V12 source-removal result. The matched neutral arms are still far above query-only masking, which preserves the V12 conclusion that direct generation-query attention is not the main effect. They are also broad and side-effectful, so they cannot support mechanism promotion.

Final Qwen3-0.6B stack decision for MC-001:

- stop broad Qwen3-0.6B mechanism search;
- treat the v1-v13 series as a failed-mechanism and diagnostic/control-surface artifact;
- move the next mechanism-card attempt to Qwen3-1.7B or an artifact-rich Gemma stack.

## Sources

- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Google DeepMind Gemma 3: https://deepmind.google/models/gemma/gemma-3/
- Google AI for Developers Gemma 3 model card: https://ai.google.dev/gemma/docs/core/model_card_3
- Gemma Scope docs: https://ai.google.dev/gemma/docs/gemma_scope
- Gemma Scope 2 4B-IT model card: https://huggingface.co/google/gemma-scope-2-4b-it
- Neuronpedia Gemma Scope 2 demo: https://www.neuronpedia.org/gemma-scope-2
