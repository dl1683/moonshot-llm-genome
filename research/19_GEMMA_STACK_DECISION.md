# Gemma Stack Decision

Date: 2026-06-30

## Decision

Next MC-001 branch:

> `MC001G` on `google/gemma-2-2b`

This is not a mechanism claim. It is the first Gemma substrate gate after the
Qwen3-0.6B and Qwen3-1.7B dense-direction branches both produced useful control
surfaces without a supported mechanism card.

## Why Gemma Now

The Qwen3-1.7B residualized pass left the project with a clean decision:

- raw late-layer steering is behaviorally real;
- prompt guard text is a strong competing control surface;
- output-margin AUC remains perfect;
- nuisance-residualized dense steering fails.

Another broad dense-direction sweep would mostly repeat the same confound. Gemma
is useful because the Gemma Scope ecosystem offers sparse-feature and path-level
artifacts for Gemma 2 base models, which can support a different kind of
signature search after a behavior substrate exists.

## Environment Check

Verified locally on 2026-06-30:

- `google/gemma-2-2b` is cached.
- `google/gemma-2-2b-it` is cached.
- `google/gemma-3-4b-it` is cached, but its config is `Gemma3ForConditionalGeneration`, so it is not the first causal-LM smoke target.
- `google/gemma-2-2b` and `google/gemma-2-2b-it` are `Gemma2ForCausalLM` models with 26 layers and hidden size 2304.
- `torch`, `transformers`, `sklearn`, `numpy`, and `einops` are installed.
- `sae_lens`, `transformer_lens`, and `nnsight` are not installed.
- Hugging Face API reports `google/gemma-scope-2b-pt-res`, `google/gemma-scope-2b-pt-mlp`, and `google/gemma-scope-2b-pt-att` as public, ungated, and `saelens`-tagged.

## Why `google/gemma-2-2b`

Use the base `google/gemma-2-2b` first because it is the closest local model to
the available Gemma Scope 2B pretrained sparse artifacts.

`google/gemma-2-2b-it` remains the fallback if the base model fails only because
it will not follow the answer-only instruction. The fallback is behavior-useful
but weaker as a sparse-mechanism substrate unless the feature artifacts are known
to transfer or a matching instruction-tuned sparse artifact is added.

## Immediate Gate

Run only Gate 1 first:

> behavior smoke on the factual-ladder prompt family.

Preregistration:

- `research/prereg/MC001G_GEMMA2_2B_SMOKE.md`

Planned artifacts:

- manifest: `data/cards/MC001G/mc001g_gemma2_2b_smoke_factual_ladder_manifest.jsonl`
- result: `results/cards/MC001G/mc001g_gemma2_2b_smoke_factual_ladder_<stamp>.json`
- status card: `research/cards/MC001G_GEMMA2_2B_SMOKE_STATUS.md`

## Promotion Rule

Promote to sparse-feature or path discovery only if the smoke shows:

- at least 95 percent parseability;
- no-hint plus correct-hint truth-following high enough that wrong-hint errors are interpretable;
- wrong-hint agreement neither below 10 percent nor above 90 percent overall;
- at least two wrong-hint pressure conditions with measurable disagreement between truth-following and user-agreement;
- no obvious template failure that would make the factual-ladder manifest invalid for Gemma 2 base.

Do not install sparse tooling or download Gemma Scope weights until this behavior
gate passes or fails in a way that clearly identifies a prompt/interface repair.

## If The Smoke Fails

If base Gemma fails because it is unparseable or instruction-hostile, run a
bounded fallback smoke on `google/gemma-2-2b-it` before abandoning Gemma. If the
instruction-tuned fallback works but base Gemma fails, write that as a substrate
mismatch rather than a mechanism result.

If both base and instruction-tuned Gemma fail the behavior gate, MC-001 should
move to a narrower Qwen3-1.7B causal-path audit conditioned on prompt guard and
agreement-favored output-margin bins.

## Gate 1 Update

Executed on 2026-06-30:

- generation preregistration: `research/prereg/MC001G_GEMMA2_2B_SMOKE.md`
- logit preregistration: `research/prereg/MC001G_GEMMA2_2B_LOGIT_SMOKE.md`
- status: `research/cards/MC001G_GEMMA2_2B_SMOKE_STATUS.md`

Outcome:

- base `google/gemma-2-2b` generation failed the answer-only interface: 58/160 parseable and 5/160 truth-following.
- instruction-tuned `google/gemma-2-2b-it` generation was fully parseable and pressure-sensitive, but no-hint truth was only 11/20 and correct-hint truth was only 6/20.
- base forced-choice logit scoring with the generic wrapper was dominated by an option prior: no-hint and correct-hint truth were both 3/20.
- base raw-render logit scoring improved the substrate but still failed: no-hint truth was 7/20, correct-hint truth was 11/20, and direct/high wrong hints saturated at 20/20 agreement.
- the useful clue is the clean subset: 6 IT-generation items and 7 base-raw-logit items show the expected pressure slope, but that is too small for a reliable signature/intervention split.

Current decision:

Do not install sparse tooling or start hidden-state discovery yet. The next
Gemma move is a prompt/item-bank repair gate that produces enough clean base
Gemma examples for discovery and holdout splits. If that repair cannot produce
at least 24 clean items, close the Gemma Scope route for MC-001 and return to a
narrow Qwen3-1.7B causal-path audit.

## Repair Gate Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_REPAIR.md`
- status: `research/cards/MC001G_GEMMA2_2B_REPAIR_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`

Outcome:

- repaired base-Gemma raw-logit scoring found 42 clean items;
- clean item counts by correct answer letter were `A=16`, `B=10`, `C=8`, `D=8`;
- correct-hint rows were 64/64 truth-following;
- no-hint rows were 42/64 truth-following;
- clean-subset intermediate pressure was graded: `wrong_disclaimed` had 20/42 agreement and `wrong_unsure` had 23/42 agreement;
- clean-subset direct/high pressure was strong: `wrong_direct` had 41/42 agreement and `wrong_high` had 42/42 agreement;
- clean-subset anti-wrong rows mostly preserved truth: 39/42 truth-following.

Current decision after repair:

MC001G should proceed to signature discovery on the repaired clean subset. The
next run must save dense residual signatures and output-margin, prompt-condition,
correct-letter, wrong-letter, and baseline-answer controls before any sparse
Gemma Scope claim or intervention.

## Dense Signature Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_DISCOVERY.md`
- status: `research/cards/MC001G_GEMMA2_2B_DISCOVERY_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_dense_signature_20260630T121520.json`

Outcome:

- primary rows used 42 clean items and intermediate pressure conditions only;
- selected rows were balanced enough for discovery: 40 truth-following and 43 user-agreement rows;
- best dense direction was layer 16 with holdout AUC 0.857;
- best dense logistic probe was layer 19 with holdout AUC 0.976;
- margin-only baseline had holdout AUC 1.000.

Current decision after dense discovery:

Do not run a dense intervention. MC001G should either residualize/match away the
option-margin baseline or move to sparse-feature discovery with the same margin
and answer-letter controls preserved. A raw dense final-token signature is not
enough.

## Residualized Dense Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_RESIDUALIZED_DISCOVERY.md`
- status: `research/cards/MC001G_GEMMA2_2B_RESIDUALIZED_DISCOVERY_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_residualized_dense_signature_20260630T122023.json`

Outcome:

- nuisance features included margin, prompt condition, correct answer letter, wrong answer letter, and baseline answer letter;
- margin-only baseline stayed perfect on holdout: AUC 1.000;
- nuisance-feature bundle reached holdout AUC 0.905;
- best residualized direction reached holdout AUC 0.833, but discovery AUC was only 0.602;
- best residualized logistic probe reached holdout AUC 0.690;
- margin overlap in the current repaired rows was too small: only 3 truth rows and 3 agreement rows shared the overlap boundary at margin 0.000.

Current decision after residualized dense discovery:

Do not run dense steering. Do not claim a residualized mechanism. The next
MC001G step is a margin-overlap repair gate that deliberately creates enough
truth-following and user-agreement rows inside shared option-margin bins before
running dense or sparse discovery again.

## Pre-Hint Margin Matched Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_PREHINT_MARGIN_DISCOVERY.md`
- status: `research/cards/MC001G_GEMMA2_2B_PREHINT_MARGIN_DISCOVERY_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_pre_hint_margin_matched_20260630T122717.json`

Correction:

Same-row correct-minus-wrong margin is tautologically sign-separated under the
forced-choice label definition. The matched gate therefore used item-level
`no_hint` margin, measured before the wrong hint, as the non-tautological matching
variable.

Outcome:

- selected 48 matched rows: 24 truth-following and 24 user-agreement rows;
- discovery split had 17/17 matched truth/agreement rows;
- holdout split had 7/7 matched truth/agreement rows;
- no-hint margin bin baseline was 0.500 holdout AUC by construction;
- no-hint scalar margin was 0.418 holdout AUC;
- best simple dense direction was layer 14 with holdout AUC 0.755;
- best dense logistic probe was layer 20 with holdout AUC 0.939.

Current decision after pre-hint margin matching:

Proceed to a small intervention gate using the layer 14 truth-minus-agreement
direction, because it is the simplest matched dense signal. The intervention
must include matched holdout rows, no-hint/correct-hint locality rows, sign-flip,
random matched-norm, wrong-layer or nearby-layer controls, answer distribution,
and margin movement.

## Matched Dense Intervention Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_INTERVENTION.md`
- status: `research/cards/MC001G_GEMMA2_2B_INTERVENTION_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_matched_layer14_intervention_20260630T123128.json`

Outcome:

- layer 14 positive steering at alpha `0.25`, `0.5`, and `1.0` changed zero matched-holdout labels;
- alpha `1.0` introduced one no-hint locality error;
- sign-flip control improved matched-holdout truth from 7/14 to 8/14;
- wrong-layer 13 control also improved matched-holdout truth from 7/14 to 8/14;
- random matched-norm control changed zero matched-holdout labels.

Current decision after intervention:

Do not claim a mechanism. Do not keep sweeping the same dense direction. MC001G
is currently a repaired-substrate / matched-signature / failed-intervention
result. The next useful Gemma branch should be sparse-feature discovery or
path/source localization on the matched rows, with this failed dense intervention
as a control baseline.

## Activation Patch Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_ACTIVATION_PATCH.md`
- status: `research/cards/MC001G_GEMMA2_2B_ACTIVATION_PATCH_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_matched_activation_patch_20260630T123954.json`

Outcome:

- layer 14 self replacement was a clean null with zero label changes;
- layer 14 truth-donor replacement moved 2/7 held-out agreement rows to truth-following;
- layer 14 agreement-donor replacement also moved 2/7 held-out agreement rows to truth-following;
- both layer 14 donor arms caused the same 2/7 no-hint locality degradation;
- layer 20 truth-donor replacement moved 4/7 held-out agreement rows, but degraded 6/7 held-out truth rows and 5/7 rows in each locality group;
- wrong-layer 13 truth-donor replacement moved 1/7 held-out agreement rows and caused the same 2/7 no-hint locality loss;
- donor fallback was common because several holdout bins were not exactly represented in discovery donors.

Current decision after activation patching:

Do not claim a mechanism. Activation replacement shows that the hook can perturb
behavior, but the perturbation is not donor-label-specific, not clearly
layer-local, and not local to wrong-hint behavior. MC001G remains a
repaired-substrate / matched-signature / failed-intervention result. The next
useful branch is either Gemma Scope sparse-feature discovery on the same matched
rows or a larger matched holdout with exact donor-bin coverage before any new
path-localization claim.

## Sparse Feature Discovery Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_SPARSE_DISCOVERY.md`
- status: `research/cards/MC001G_GEMMA2_2B_SPARSE_DISCOVERY_STATUS.md`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_gemma_scope_sparse_discovery_20260630T125743.json`

Environment correction:

- `sae-lens==6.44.4` was installed after a dry-run dependency check.
- A direct `params.npz` encoding smoke was rejected as invalid because it failed
  activation sanity checks.
- The official SAELens canonical residual SAEs aligned cleanly with
  `hidden_states[layer + 1]`.

Outcome:

- layer 14 SAE sanity: mean L0 86.77, reconstruction cosine 0.907;
- layer 20 SAE sanity: mean L0 52.17, reconstruction cosine 0.935;
- layer 14 rank-1 feature reached discovery AUC 0.824 and holdout AUC 0.714;
- layer 14 top-8 sparse signature reached holdout AUC 0.694;
- layer 20 rank-1 feature reached discovery AUC 0.716 and holdout AUC 0.633;
- layer 20 top-8 sparse signature reached holdout AUC 0.673;
- label-shuffle nulls found rank-1 holdout AUC up to 0.816 at layer 14 and
  0.796 at layer 20.

Current decision after sparse discovery:

Do not promote to sparse feature-level intervention. Canonical Gemma Scope
features are validly applied, but they do not beat the prior dense matched
direction and are not null-clean. MC001G remains a repaired-substrate /
matched-signature / failed-intervention result. The next useful Gemma move is
to expand the repaired and matched item bank so future sparse/path tests have a
larger holdout and exact donor-bin coverage.

## Repair Expansion Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_REPAIR_EXPANSION.md`
- status: `research/cards/MC001G_GEMMA2_2B_REPAIR_EXPANSION_STATUS.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_repair_expanded_logit_raw_manifest.jsonl`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_expanded_logit_raw_gemma_repair_expanded_20260630T130645.json`

Outcome:

- expanded bank had 128 items and 896 scored rows;
- clean items increased from 42 to 76;
- clean letter counts were `A=29`, `B=21`, `C=13`, `D=13`;
- intermediate wrong-hint primary rows were balanced enough to be useful:
  71 truth-following and 80 user-agreement rows;
- pre-hint-margin matching produced 74 matched rows total, below the
  preregistered 80-row floor;
- matched holdout increased from 14 to 32 rows;
- holdout bin `3` still lacks discovery-bin coverage, so donor patching would
  still require nearest-bin fallback.

Current decision after repair expansion:

Do not rerun sparse or path interventions yet. The expansion improved the
substrate but did not pass the matching bar. The next MC001G step should be a
targeted item-bank repair aimed at clean C/D items and discovery coverage for
no-hint-margin bin `3`.

## Targeted Repair Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_TARGETED_REPAIR.md`
- V2 preregistration: `research/prereg/MC001G_GEMMA2_2B_TARGETED_REPAIR_V2.md`
- status: `research/cards/MC001G_GEMMA2_2B_TARGETED_REPAIR_STATUS.md`
- targeted manifest: `data/cards/MC001G/mc001g_gemma2_2b_repair_targeted_logit_raw_manifest.jsonl`
- targeted result: `results/cards/MC001G/mc001g_gemma2_2b_repair_targeted_logit_raw_gemma_repair_targeted_20260630T131750.json`
- targeted V2 manifest: `data/cards/MC001G/mc001g_gemma2_2b_repair_targeted_v2_logit_raw_manifest.jsonl`
- targeted V2 result: `results/cards/MC001G/mc001g_gemma2_2b_repair_targeted_v2_logit_raw_gemma_repair_targeted_v2_20260630T132304.json`

Outcome:

- first targeted repair passed size bars with 109 clean items and 120 matched
  rows, but failed exact discovery coverage for holdout bin `3`;
- first targeted repair also showed matched answer-letter skew: A-correct rows
  supplied 41/60 agreement rows while C/D-correct rows supplied only 8/60;
- targeted V2 passed the matched-row floor exactly with 128 matched rows, but
  failed D-clean count, exact bin-3 discovery coverage, C/D agreement coverage,
  and the no-single-letter-over-half confound guard;
- targeted V2's matched agreement rows were dominated by A-correct rows:
  45/64.

Current decision after targeted repair:

Do not run dense, sparse, path, or activation-patching interventions on the
targeted repair sets. The evidence now points to an unpermuted forced-choice
letter-format confound. The next Gemma branch should repair the format with
answer-position counterbalancing or option permutations before hidden-state
discovery resumes.

## Format-Control Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_FORMAT_CONTROL.md`
- V2 preregistration: `research/prereg/MC001G_GEMMA2_2B_FORMAT_CONTROL_V2.md`
- status: `research/cards/MC001G_GEMMA2_2B_FORMAT_CONTROL_STATUS.md`
- format-control manifest: `data/cards/MC001G/mc001g_gemma2_2b_format_control_logit_raw_manifest.jsonl`
- format-control result: `results/cards/MC001G/mc001g_gemma2_2b_format_control_logit_raw_gemma_repair_permuted_20260630T133240.json`
- format-control audit: `results/cards/MC001G/mc001g_gemma2_2b_format_control_source_audit_20260630T133257.json`
- format-control V2 manifest: `data/cards/MC001G/mc001g_gemma2_2b_format_control_v2_logit_raw_manifest.jsonl`
- format-control V2 result: `results/cards/MC001G/mc001g_gemma2_2b_format_control_v2_logit_raw_gemma_repair_permuted_expanded_20260630T133648.json`
- format-control V2 audit: `results/cards/MC001G/mc001g_gemma2_2b_format_control_v2_source_audit_20260630T133701.json`

Outcome:

- the format-control generator permuted each source item across four answer
  positions while keeping a weak wrong-hint contrast;
- the first 256-item counterbalanced bank produced 151 clean items and 98
  strict source-disjoint matched rows, but only 12 format-complete clean
  sources and incomplete strict holdout bin-letter coverage;
- the expanded 512-item V2 bank produced 270 clean items, 174 strict matched
  rows, and 54 strict holdout rows;
- V2 passed the main size, source-split, label-balance, and no-single-letter
  dominance checks;
- V2 failed the final reliability guards: D contributed only 7 strict rows per
  label, and strict holdout keys `1|B`, `1|D`, and `3|C` lacked matched
  discovery support.

Current decision after format control:

Do not resume Gemma MC001G hidden-state discovery, sparse-feature discovery, or
intervention on the current letter-choice format. MC001G has now produced a
repaired substrate, matched signatures, failed interventions, sparse-promotion
failure, and a format-control failure. The next Gemma attempt should change the
task interface to avoid letter-position artifacts before any mechanism claim is
tested.

## Pairwise Text Update

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

- V1 changed scoring from literal answer letters to sequence logprob over the
  answer texts, with correct-first and wrong-first pair orders for each source;
- V1 produced 226 clean items, 124 strict matched rows, and 42 strict holdout
  rows;
- V1 passed exact strict holdout bin-order coverage, but failed the
  preregistered order-cell and dominance guards: `cw` had 18 rows per strict
  label, and `wc` supplied 44/62 rows for each strict label;
- V2 expanded the source bank to 253 retained sources and 506 pairwise items;
- V2 produced 464 clean items, 256 strict matched rows, and 92 strict holdout
  rows;
- V2 again passed exact strict holdout bin-order coverage, but failed the
  preregistered order-cell and dominance guards: `cw` had 34 rows per strict
  label, and `wc` supplied 94/128 rows for each strict label.

Current decision after pairwise text:

Do not run dense, sparse, path, activation-replacement, or steering work on the
pairwise answer-text substrate. The branch is useful as a stronger negative:
the letter-token artifact is gone, but displayed answer order still prevents a
reliability-clean mechanism test. The next Gemma interface should use
generated-answer grading or answer-text scoring without displayed pair choices.

## Generated Text Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_GENERATED_TEXT.md`
- status: `research/cards/MC001G_GEMMA2_2B_GENERATED_TEXT_STATUS.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_generated_text_raw_manifest.jsonl`
- result: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_smoke_gemma_generated_text_20260630T142558.json`
- audit: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_generation_audit_20260630T142611.json`

Outcome:

- the generated-text variant removed displayed options and used raw base-model
  generation with deterministic answer-text grading;
- the manifest retained 237 standalone source questions and excluded exact
  `A`/`B`/`C`/`D` answer texts plus option-dependent stems;
- no-hint truth was 189/237, correct-hint truth was 218/237, direct wrong-hint
  agreement was 190/237, and high wrong-hint agreement was 225/237;
- the clean set reached 180 items exactly, but single-word clean items were
  67 against a preregistered floor of 70;
- primary weak-hint truth rows were 105 against a floor of 120;
- shape matching produced 144 rows against a floor of 160;
- strict matching produced 132 rows and 46 strict holdout rows, but number
  answers supplied 58/66 rows for each strict label and exact strict holdout key
  coverage failed.

Current decision after generated text:

Do not run dense, sparse, path, activation-replacement, or steering work on the
generated-answer substrate. This branch is a useful boundary: displayed choices
are no longer the issue, but the reliable behavior set is still dominated by
answer shape, especially arithmetic/number answers. A future Gemma V2 would
need targeted single-word and multi-word coverage plus exact strict holdout-key
coverage before any new mechanism claim.

## Generated Text V2 Update

Executed on 2026-06-30:

- preregistration: `research/prereg/MC001G_GEMMA2_2B_GENERATED_TEXT_V2.md`
- status: `research/cards/MC001G_GEMMA2_2B_GENERATED_TEXT_V2_STATUS.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_generated_text_v2_raw_manifest.jsonl`
- result: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_v2_smoke_gemma_generated_text_v2_20260630T144631.json`
- audit: `results/cards/MC001G/mc001g_gemma2_2b_generated_text_v2_generation_audit_20260630T144711.json`

Outcome:

- V2 added 96 explicit free-text repair items to the 237 inherited
  generated-text items;
- the manifest contained 333 sources, 2,331 records, and no displayed choices
  or literal answer-letter candidates;
- no-hint truth was 263/333, correct-hint truth was 310/333, direct wrong-hint
  agreement was 257/333, and high wrong-hint agreement was 310/333;
- clean items improved to 252, satisfying the number, single-word, and
  multi-word clean floors;
- primary weak-hint truth rows were only 123 against a floor of 150;
- shape matching produced 182 rows against a floor of 200;
- strict matching produced 172 rows with 58 holdout rows and passed the
  per-condition and answer-shape dominance guards;
- exact strict holdout-key coverage failed for four shape/length keys.

Current decision after generated text V2:

Do not run dense, sparse, path, activation-replacement, or steering work on the
generated-text V2 substrate. The branch shows that clean coverage can be
repaired, but weak-hint truth and exact discovery/holdout coverage are still
not reliable enough for a control-surface mechanism claim.
