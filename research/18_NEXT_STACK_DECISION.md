# Next Stack Decision

Date: 2026-06-30

## Decision

Next MC-001 mechanism-card attempt:

> `MC001B` on `Qwen/Qwen3-1.7B`

Reason:

Qwen3-0.6B is closed as a diagnostic/control-only substrate. The next useful move is to ask whether the same truth-versus-wrong-hint behavior has a cleaner hidden signature and intervention surface in a slightly larger small LLM, while preserving the Qwen tokenizer/model-family continuity that makes comparisons interpretable.

## Current Environment Check

Verified on 2026-06-30:

- CUDA is available with one `NVIDIA GeForce RTX 5090 Laptop GPU`.
- `Qwen/Qwen3-1.7B` is present in the local Hugging Face cache.
- Hugging Face API reports `Qwen/Qwen3-1.7B` as ungated, public, `transformers`, `text-generation`.
- `google/gemma-3-4b-it` is also cached, but Hugging Face reports `gated: manual` and `pipeline_tag: image-text-to-text`, so it remains the artifact-rich comparison stack rather than the immediate smoke target.

## Why Not More Qwen3-0.6B

The Qwen3-0.6B v1-v13 series already closed the broad routes:

- dense final-token direction;
- residualized dense direction;
- prompt-prefill dense steering;
- cross-layer transport;
- answer-prefix dense steering;
- single-head and single-layer source masking;
- cumulative generation-query source masking;
- input-mask semantics;
- layout/tokenization parity.

The result was `control_without_explanation`, not `mechanism_supported`.

More 0.6B sweeps would mostly test the experimenter's patience, not the mechanism-card hypothesis.

## Why Qwen3-1.7B Before Gemma

Qwen3-1.7B is the lower-friction escalation:

- same family as the closed 0.6B run;
- same prompt/interface assumptions;
- same activation/probe tooling should work;
- local cache avoids setup ambiguity;
- ungated access avoids credential or license ambiguity;
- larger capacity may make truth-vs-agreement conflict less purely output-margin dominated.

Gemma remains important because Gemma Scope can support sparse-feature and path-level work. It should become the next comparison if Qwen3-1.7B repeats the same failure pattern or if a Qwen3-1.7B signature needs sparse-feature interpretation.

## Immediate Gate

Run only Gate 1 first:

> behavior smoke on the factual-ladder prompt family.

The smoke does not claim a mechanism. It answers whether Qwen3-1.7B has measurable wrong-hint compliance, parseable answers, and no-hint/correct-hint accuracy high enough to justify hidden-state discovery.

Preregistration:

- `research/prereg/MC001B_QWEN3_1P7B_SMOKE.md`

Planned artifacts:

- manifest: `data/cards/MC001B/mc001b_qwen3_1p7b_smoke_factual_ladder_manifest.jsonl`
- result: `results/cards/MC001B/mc001b_qwen3_1p7b_smoke_factual_ladder_<stamp>.json`
- status card: `research/cards/MC001B_QWEN3_1P7B_SMOKE_STATUS.md`

## Promotion Rule

Promote to hidden-state discovery only if the smoke shows:

- at least 95 percent parseability;
- no-hint plus correct-hint truth-following high enough that wrong-hint errors are interpretable;
- wrong-hint agreement neither below 10 percent nor above 90 percent overall;
- at least two wrong-hint pressure conditions with measurable disagreement between truth-following and user-agreement;
- no obvious template failure that would make the 0.6B manifest invalid for 1.7B.

If those fail, the next stack decision should be Gemma or a revised MC-001 prompt family, not hidden-state work.

## Gate 2 Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001B_QWEN3_1P7B_DISCOVERY.md`
- status: `research/cards/MC001B_QWEN3_1P7B_CONTROLLED_STATUS.md`
- result: `results/cards/MC001B/mc001b_qwen3_1p7b_controlled_20260630T112935.json`

Outcome:

- Qwen3-1.7B has a useful control surface at `h21 alpha=0.5`.
- Wrong-hint truth-following improved from 125/180 to 152/180 with 288/288 parseability.
- Holdout truth improved from 81/96 to 90/96 and paraphrase holdout truth improved from 80/96 to 87/96.
- Random-matched and wrong-token controls stayed near baseline; sign-flip moved in the harmful direction.
- The signature gate still failed mechanism promotion because the next-token logit-margin baseline reached AUC 1.000 on calibration, holdout, and paraphrase holdout.

Current decision:

Qwen3-1.7B should be treated as a better diagnostic control-surface substrate than Qwen3-0.6B, not as a solved mechanism-card substrate. The next broad mechanism attempt should either residualize away output-margin behavior on Qwen3-1.7B or move to an artifact-rich Gemma stack.

## Gate 2b Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001B_QWEN3_1P7B_RESIDUALIZED.md`
- status: `research/cards/MC001B_QWEN3_1P7B_RESIDUALIZED_STATUS.md`
- result: `results/cards/MC001B/mc001b_qwen3_1p7b_controlled_v3_20260630T114242.json`

Outcome:

- raw `h21 alpha=0.50` replicated the useful control surface on the balanced manifest: wrong-hint truth improved from 113/180 to 154/180.
- prompt guard alone reached 143/180 wrong-hint truth with no no/correct degradation, making prompt policy a strong surface baseline.
- residualized `h21` failed: `alpha=0.50` dropped wrong-hint truth to 103/180 and shifted answer distribution toward `C`/`D`.
- margin-only AUC remained 1.000 on calibration, holdout, and paraphrase holdout.

Current decision after Gate 2b:

No more broad dense residual-direction sweeps for MC001B. Qwen3-1.7B should be written as a diagnostic control-surface result. The next mechanism-card attempt should either use Gemma with sparse-feature/path tooling or a much narrower Qwen3-1.7B causal-path test conditioned on prompt guard and agreement-favored margin bins.

## Gemma Gate Update

Executed on 2026-06-30:

- stack decision: `research/19_GEMMA_STACK_DECISION.md`
- generation preregistration: `research/prereg/MC001G_GEMMA2_2B_SMOKE.md`
- logit preregistration: `research/prereg/MC001G_GEMMA2_2B_LOGIT_SMOKE.md`
- status: `research/cards/MC001G_GEMMA2_2B_SMOKE_STATUS.md`

Outcome:

- base `google/gemma-2-2b` generation failed the answer-only interface: 58/160 parseable and 5/160 truth-following.
- base forced-choice logit scoring with raw rendering improved behavior but still failed: no-hint truth 7/20, correct-hint truth 11/20, wrong-direct and wrong-high both saturated at 20/20 agreement.
- instruction-tuned `google/gemma-2-2b-it` was fully parseable and pressure-sensitive, but no-hint truth was 11/20 and correct-hint truth was 6/20.
- clean subsets showed the intended pressure slope, but only on 6-7 items.

Current decision after Gemma Gate 1:

Do not start sparse-feature/path discovery yet. The next Gemma move must be a
prompt/item-bank repair gate with enough clean base-Gemma examples for discovery
and holdout splits. If that fails, the Gemma Scope route should close for
MC-001 and the project should return to a narrow Qwen3-1.7B causal-path audit.

## Gemma Repair Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_REPAIR.md`
- status: `research/cards/MC001G_GEMMA2_2B_REPAIR_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`

Outcome:

- base Gemma repair passed the behavior-substrate gate with 42 clean items;
- minimum clean items per answer letter was 8;
- intermediate wrong-hint pressure was not saturated;
- direct/high wrong-hint pressure produced strong user agreement;
- anti-wrong mostly preserved truth-following.

Current decision:

The live MC-001 branch is now MC001G signature discovery on the repaired
base-Gemma clean subset. Qwen3-1.7B remains the fallback causal-path branch, not
the immediate next action.

## Gemma Dense Signature Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_DISCOVERY.md`
- status: `research/cards/MC001G_GEMMA2_2B_DISCOVERY_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_dense_signature_20260630T121520.json`

Outcome:

- dense signatures are measurable on MC001G;
- layer 16 truth-minus-agreement direction reached holdout AUC 0.857;
- layer 19 dense logistic probe reached holdout AUC 0.976;
- correct-minus-wrong option margin reached holdout AUC 1.000.

Current decision:

The live branch is now margin-conditioned or residualized MC001G discovery. Do
not promote a dense intervention until a signature survives the margin baseline.

## Gemma Residualized Dense Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_RESIDUALIZED_DISCOVERY.md`
- status: `research/cards/MC001G_GEMMA2_2B_RESIDUALIZED_DISCOVERY_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_residualized_dense_signature_20260630T122023.json`

Outcome:

- residualization against margin and answer-letter nuisances did not rescue the dense route;
- nuisance-feature bundle holdout AUC was 0.905;
- best residualized dense direction had holdout AUC 0.833 but weak discovery AUC 0.602;
- best residualized dense logistic probe had holdout AUC 0.690;
- current repaired rows have only 3 matched margin-overlap pairs.

Current decision:

The live branch is now a MC001G margin-overlap repair gate. More dense or sparse
feature discovery on the current rows would mostly rediscover option margin.

## Gemma Pre-Hint Margin Matched Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_PREHINT_MARGIN_DISCOVERY.md`
- status: `research/cards/MC001G_GEMMA2_2B_PREHINT_MARGIN_DISCOVERY_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_pre_hint_margin_matched_20260630T122717.json`

Outcome:

- same-row option margin was identified as a tautological forced-choice readout;
- non-tautological matching used item-level no-hint margin;
- matched set contained 24 truth rows and 24 agreement rows;
- best simple dense direction was layer 14 with holdout AUC 0.755;
- best dense logistic probe was layer 20 with holdout AUC 0.939;
- no-hint margin and no-hint bin baselines were weak or matched out.

Current decision:

The live MC001G branch is now a small layer-14 intervention gate on the pre-hint
margin matched substrate. This is the first Gemma branch that reaches the
signature-to-intervention boundary under a non-tautological margin control.

## Gemma Matched Intervention Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_INTERVENTION.md`
- status: `research/cards/MC001G_GEMMA2_2B_INTERVENTION_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_matched_layer14_intervention_20260630T123128.json`

Outcome:

- intended layer 14 positive steering failed at all tested doses;
- sign-flip and wrong-layer controls each matched or beat the intended effect;
- locality stayed mostly intact except one no-hint error at alpha `1.0`;
- MC001G does not yet satisfy the intervention gate.

Current decision:

The live branch is no longer dense layer-14 steering. Move to sparse-feature
discovery or path/source localization on the matched Gemma rows, carrying the
failed dense intervention as the baseline control.

## Gemma Activation Patch Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_ACTIVATION_PATCH.md`
- status: `research/cards/MC001G_GEMMA2_2B_ACTIVATION_PATCH_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_matched_activation_patch_20260630T123954.json`

Outcome:

- self replacement was a clean null;
- layer 14 truth-donor replacement moved 2/7 held-out agreement rows to truth;
- layer 14 agreement-donor replacement matched that 2/7 truth gain;
- both layer 14 donor arms caused 2/7 no-hint locality losses;
- layer 20 truth-donor replacement was broad disruption, not a selective intervention;
- wrong-layer 13 partially matched the intended effect and shared the locality loss.

Current decision:

MC001G remains below the mechanism-card bar. The branch has a repaired behavior
substrate and matched internal signatures, but both additive steering and
activation replacement failed reliability controls. The next useful move is
Gemma Scope sparse-feature discovery or a larger exact-bin matched holdout, not
another broad dense intervention.

## Gemma Sparse Feature Discovery Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_SPARSE_DISCOVERY.md`
- status: `research/cards/MC001G_GEMMA2_2B_SPARSE_DISCOVERY_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_gemma_scope_sparse_discovery_20260630T125743.json`

Outcome:

- official SAELens canonical residual SAEs were used after direct NPZ encoding
  failed sanity checks;
- layer 14 sparse sanity was acceptable: mean L0 86.77 and reconstruction
  cosine 0.907;
- layer 14 rank-1 sparse feature reached only 0.714 holdout AUC;
- layer 14 top-8 sparse signature reached only 0.694 holdout AUC;
- label-shuffle nulls found stronger layer 14 rank-1 holdout AUC, up to 0.816;
- layer 20 was weaker, with rank-1 holdout AUC 0.633.

Current decision:

Do not run feature-level sparse interventions on this matched set. MC001G now
has dense, activation-patching, and canonical sparse failures against the
mechanism-card gate. The next useful branch is data/match expansion: build a
larger repaired item bank with exact no-hint-margin bin coverage, then rerun
signature and intervention gates.

## Gemma Repair Expansion Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_REPAIR_EXPANSION.md`
- status: `research/cards/MC001G_GEMMA2_2B_REPAIR_EXPANSION_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_expanded_logit_raw_gemma_repair_expanded_20260630T130645.json`

Outcome:

- clean items increased from 42 to 76;
- clean letter counts passed the preregistered floor: `A=29`, `B=21`, `C=13`,
  `D=13`;
- matched rows increased from 48 to 74, but missed the 80-row floor;
- matched holdout increased from 14 to 32 rows;
- no-hint-margin holdout bin `3` still lacks matched discovery coverage.

Current decision:

The data expansion improved MC001G but did not make it intervention-ready. The
next useful work is targeted item-bank repair, not another final-token
intervention: add clean C/D items and matched discovery rows in bin `3`, then
rerun the matching audit.

## Gemma Targeted Repair Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_TARGETED_REPAIR.md`
- V2 preregistration: `research/prereg/MC001G_GEMMA2_2B_TARGETED_REPAIR_V2.md`
- status: `research/cards/MC001G_GEMMA2_2B_TARGETED_REPAIR_STATUS.md`
- targeted result: `results/cards/MC001G/mc001g_gemma2_2b_repair_targeted_logit_raw_gemma_repair_targeted_20260630T131750.json`
- targeted V2 result: `results/cards/MC001G/mc001g_gemma2_2b_repair_targeted_v2_logit_raw_gemma_repair_targeted_v2_20260630T132304.json`

Outcome:

- first targeted repair increased clean items to 109 and matched rows to 120;
- targeted V2 increased clean items to 130 and matched rows to 128;
- both targeted repairs still lacked exact matched discovery support for
  holdout no-hint-margin bin `3`;
- targeted V2 missed the D-clean floor, with `D=22` against a preregistered
  floor of 24;
- targeted V2 preserved a strong answer-letter confound: A-correct rows supplied
  45/64 matched user-agreement rows, while C/D-correct rows supplied only 8/64.

Current decision:

Stop unpermuted MC001G item-bank expansion. The next useful work is a
format-control repair, especially answer-position counterbalancing or
option-permutation controls, before any new dense, sparse, path, or
activation-replacement intervention.

## Gemma Format-Control Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_FORMAT_CONTROL.md`
- V2 preregistration: `research/prereg/MC001G_GEMMA2_2B_FORMAT_CONTROL_V2.md`
- status: `research/cards/MC001G_GEMMA2_2B_FORMAT_CONTROL_STATUS.md`
- format-control result: `results/cards/MC001G/mc001g_gemma2_2b_format_control_logit_raw_gemma_repair_permuted_20260630T133240.json`
- format-control audit: `results/cards/MC001G/mc001g_gemma2_2b_format_control_source_audit_20260630T133257.json`
- format-control V2 result: `results/cards/MC001G/mc001g_gemma2_2b_format_control_v2_logit_raw_gemma_repair_permuted_expanded_20260630T133648.json`
- format-control V2 audit: `results/cards/MC001G/mc001g_gemma2_2b_format_control_v2_source_audit_20260630T133701.json`

Outcome:

- option-position counterbalancing produced source groups with the correct
  answer appearing once at each of A, B, C, and D;
- the first format-control run reached 151 clean items and 98 strict
  source-disjoint matched rows, but failed the format-complete source floor and
  exact strict holdout bin-letter coverage;
- the expanded V2 run reached 270 clean items, 174 strict source-disjoint
  matched rows, and 54 strict holdout rows;
- V2 still failed the preregistered D-cell floor, with 7 strict D rows per
  label against a floor of 8;
- V2 also failed exact strict holdout bin-letter coverage, missing discovery
  support for `1|B`, `1|D`, and `3|C`.

Current decision after format control:

Close the current MC001G forced-choice letter substrate for mechanism work.
Counterbalancing improved the controls, but did not produce a reliability-clean
intervention substrate. The next MC001G-style attempt should change the
behavior interface itself: non-letter answers, pairwise choices, or
generated-answer grading before any new hidden-state discovery.

## Gemma Pairwise Text Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_PAIRWISE_TEXT.md`
- V2 preregistration: `research/prereg/MC001G_GEMMA2_2B_PAIRWISE_TEXT_V2.md`
- status: `research/cards/MC001G_GEMMA2_2B_PAIRWISE_TEXT_STATUS.md`
- pairwise manifest: `data/cards/MC001G/mc001g_gemma2_2b_pairwise_text_logit_raw_manifest.jsonl`
- pairwise result: `results/cards/MC001G/mc001g_gemma2_2b_pairwise_text_logit_raw_gemma_pairwise_text_20260630T135443.json`
- pairwise audit: `results/cards/MC001G/mc001g_gemma2_2b_pairwise_text_source_audit_20260630T135457.json`
- pairwise V2 manifest: `data/cards/MC001G/mc001g_gemma2_2b_pairwise_text_v2_logit_raw_manifest.jsonl`
- pairwise V2 result: `results/cards/MC001G/mc001g_gemma2_2b_pairwise_text_v2_logit_raw_gemma_pairwise_text_v2_20260630T140430.json`
- pairwise V2 audit: `results/cards/MC001G/mc001g_gemma2_2b_pairwise_text_v2_source_audit_20260630T140448.json`

Outcome:

- V1 used 125 retained sources and 250 pairwise answer-text items;
- V1 produced 226 clean items, 124 strict matched rows, and 42 strict holdout
  rows;
- V1 passed exact strict holdout bin-order coverage, but failed pair-order
  balance: `cw` contributed 18 rows per strict label against a floor of 20,
  and `wc` supplied 44/62 rows for each strict label;
- V2 used 253 retained sources and 506 pairwise answer-text items;
- V2 produced 464 clean items, 256 strict matched rows, and 92 strict holdout
  rows;
- V2 passed exact strict holdout bin-order coverage, but failed pair-order
  balance: `cw` contributed 34 rows per strict label against a floor of 40,
  and `wc` supplied 94/128 rows for each strict label.

Current decision after pairwise text:

Do not resume hidden-state discovery or intervention on the MC001G pairwise
answer-text interface. Pairwise scoring removed literal letter outputs and
fixed exact holdout coverage, but the displayed pair order remains too
confounded with the matched behavior set. The next MC001G attempt should use
generated-answer grading or answer-text scoring without displayed pair choices.

## Gemma Generated Text Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_GENERATED_TEXT.md`
- status: `research/cards/MC001G_GEMMA2_2B_GENERATED_TEXT_STATUS.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_generated_text_raw_manifest.jsonl`
- result: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_smoke_gemma_generated_text_20260630T142558.json`
- audit: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_generation_audit_20260630T142611.json`

Outcome:

- raw generated-answer text retained 237 standalone source questions and 1,659
  records;
- no-hint truth was 189/237, and correct-hint truth was 218/237;
- weak wrong-hint rows produced a real pressure gradient, but only 105 primary
  truth rows against a floor of 120;
- the run exactly hit the 180 clean-item floor, but clean single-word items were
  67 against a floor of 70;
- shape matching produced 144 rows against a floor of 160;
- strict matching produced 132 rows and 46 strict holdout rows, but 58/66 rows
  per strict label were number answers, exceeding the 70 percent shape-dominance
  guard;
- strict holdout key coverage failed for several single-word and longer-number
  keys.

Current decision after generated text:

Do not resume hidden-state discovery or intervention on the current generated
answer-text substrate. The interface removed displayed choices and exposed a
real behavior gradient, but the reliability-clean subset is still too
answer-shape dominated. The next useful work is a generated-text V2 item bank
that deliberately repairs single-word and multi-word coverage, or a new
behavior family whose controls are not dominated by answer format.

## Gemma Generated Text V2 Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_GENERATED_TEXT_V2.md`
- status: `research/cards/MC001G_GEMMA2_2B_GENERATED_TEXT_V2_STATUS.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_generated_text_v2_raw_manifest.jsonl`
- result: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_v2_smoke_gemma_generated_text_v2_20260630T144631.json`
- audit: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_v2_generation_audit_20260630T144711.json`

Outcome:

- V2 retained 333 standalone source questions and 2,331 raw generation records;
- the source bank combined 237 inherited generated-text items with 96 explicit
  free-text repair items;
- no-hint truth was 263/333, and correct-hint truth was 310/333;
- clean items improved from 180 to 252, with 103 number, 128 single-word, and
  21 multi-word clean items;
- primary weak-hint truth rows were 123 against a preregistered floor of 150,
  while primary weak-hint agreement rows were 295;
- shape matching produced 182 rows against a floor of 200;
- strict matching produced 172 rows and 58 strict holdout rows, and answer
  shape no longer exceeded the 70 percent dominance guard;
- exact strict holdout-key coverage still failed for four shape/length keys.

Current decision after generated text V2:

Do not resume hidden-state discovery or intervention on the MC001G
generated-text V2 substrate. The clean coverage repair worked, but the
reliability gate still failed before signature discovery. The next step should
not be another broad intervention pass; it should either directly repair
weak-hint truth plus strict key coverage or move to a different behavior family.
