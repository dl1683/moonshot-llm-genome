# MC-001 Qwen3-0.6B Implementation Closeout Review

Status: complete.

Date: 2026-06-30

This is the adversarial closeout review for the Qwen3-0.6B deep dive on `MC-001`. It reviews the implementation work after the no-code rebuild and decides whether more Qwen3-0.6B mechanism-search iterations are justified.

## Scope Under Review

Objective:

> Do an exceptionally deep dive with `Qwen/Qwen3-0.6B`, using the smallest Qwen3 model for maximum iteration count, until the project goals for this substrate are accomplished.

Project-specific goal:

- attempt the first mechanism card for truth-following versus user-agreement under incorrect hints;
- use Qwen3-0.6B as the rapid iteration substrate;
- keep thoughts and verdicts in markdown;
- do not promote weak results;
- stop Qwen3-0.6B only after the plausible dense, path-local, source-mask, input-mask, and layout/tokenization routes are exhausted or clearly fail the mechanism-card standard.

## Evidence Inventory

Primary final artifact:

- diagnostic/control-surface card: `research/cards/MC001_QWEN3_0P6B_DIAGNOSTIC_CONTROL_SURFACE.md`

Controlled run artifacts:

| Pass | Run type | Result | Records | Manifest SHA256 |
| --- | --- | --- | ---: | --- |
| v1 | `qwen3_0p6b_controlled_discovery_calibration` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_20260629T200930.json` | 384 | `380b83823ca0bcd5da5e43ff2f6871901e92d2050de5e5a65f6cc8e2202e10ba` |
| v2 | `qwen3_0p6b_controlled_v2_output_margin` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v2_20260629T203703.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |
| v3 | `qwen3_0p6b_controlled_v3_residual_dose` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v3_20260629T205927.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |
| v4 | `qwen3_0p6b_controlled_v4_agreement_bin_patch` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v4_patch_20260629T212411.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |
| v5 | `qwen3_0p6b_controlled_v5_prefill_audit` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v5_prefill_20260629T213701.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |
| v6 | `qwen3_0p6b_controlled_v6_prefill_controls` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v6_prefill_controls_20260629T220249.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |
| v7 | `qwen3_0p6b_controlled_v7_transport` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v7_transport_20260629T221738.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |
| v8 | `qwen3_0p6b_controlled_v8_answer_prefix` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v8_answer_prefix_20260629T223039.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |
| v9 | `qwen3_0p6b_controlled_v9_attention_source_ablation` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v9_attention_source_ablation_20260629T224234.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |
| v10 | `qwen3_0p6b_controlled_v10_head_localization` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v10_head_localization_20260629T231146.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |
| v11 | `qwen3_0p6b_controlled_v11_layer_band_source` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v11_layer_band_source_20260629T233533.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |
| v12 | `qwen3_0p6b_controlled_v12_input_mask_semantics` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v12_input_mask_semantics_20260629T235517.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |
| v13 | `qwen3_0p6b_controlled_v13_layout_parity` | `results/cards/MC001/mc001_qwen3_0p6b_controlled_v13_layout_parity_20260630T000706.json` | 384 | `911d3fafedfb15942ad82620c4fdd4666814bd33aab61c04ba7135aa2f3715e7` |

Review and standards files:

- mechanism-card contract: `research/05_MECHANISM_CARD_CONTRACT.md`
- artifact standards: `research/08_ARTIFACT_AND_BUDGET_STANDARDS.md`
- review rubric: `research/reviews/MC001_REVIEW_RUBRIC.md`
- preregistration: `research/prereg/MC001_SYCOPHANCY_TRUTH_CONFLICT.md`
- stack decision ledger: `research/10_FIRST_STACK_DECISION.md`
- open questions ledger: `research/15_OPEN_QUESTIONS_LEDGER.md`

Validation performed during closeout:

- Python compile validation passed for `code/mc001_qwen3_controlled_v12_input_mask_semantics.py` and `code/mc001_qwen3_controlled_v13_layout_parity.py`.
- JSON audit over v1-v13 result artifacts: all expected result files exist, all expected `run_type` values match, and each controlled result reports 384 records.
- Markdown path audit over the diagnostic card, closeout review, README, research README, and no-code closure review: all referenced local `.md`, `.py`, `.json`, and `.jsonl` paths resolve.
- Stale-action scan: no remaining live-language hits for old "next Qwen3-0.6B pass" or early controlled-status authority phrasing.

## Requirement Audit

| Requirement | Evidence | Verdict |
| --- | --- | --- |
| Use Qwen3-0.6B as the rapid iteration substrate | v1-v13 all use `Qwen/Qwen3-0.6B`; stack decision records why | satisfied |
| Keep thoughts and verdicts in markdown | status cards exist for smoke and v1-v13; final diagnostic card exists | satisfied |
| Test behavior validity | smoke, v1-v3, manifest-balanced factual ladder, parseability and no/correct rows | satisfied within factual multiple-choice scope |
| Test dense hidden signature route | v1-v3, v5-v8; output margin and controls block mechanism claim | exhausted for this substrate |
| Test intervention route | raw h14, residualized directions, dose, prefill, patching, answer-prefix variants | control surface found, mechanism not supported |
| Test nulls and confounds | prompt guard, output margin, random, nearby-layer, wrong-token, matched source masks, answer distribution | sufficient to block promotion |
| Test path/source route | v9 source ablation, v10 head/local layer localization, v11 cumulative layer bands | source dependence real but not localized |
| Decompose V9 input-mask semantics | v12 input masks, literal rewrites, query-only reference arms | satisfied |
| Close layout/tokenization caveat | v13 character-matched and token-count-matched neutral rewrites | satisfied |
| Preserve raw artifacts | JSON results and manifests under `results/cards/MC001/` and `data/cards/MC001/` | satisfied |
| Produce final card | diagnostic/control-surface card | satisfied |
| Avoid overclaiming | final verdict is control-only / failed hidden-mechanism claim | satisfied |

## Rubric Review

### 1. Behavior Validity

Verdict: pass within narrow scope.

The factual-ladder multiple-choice substrate gives known answers, wrong hints, parseable option labels, no-hint/correct-hint side rows, wrong-hint variants, and paraphrase holdout. The scope is narrow: objective multiple-choice only.

### 2. Signature Validity

Verdict: fail for mechanism support.

The raw h14 scalar is predictive, but output margin reaches AUC 1.0. Residualized h14 loses most of the useful signal. The hidden signature does not beat output-only monitoring.

### 3. Intervention Validity

Verdict: works as a control surface, not as a supported mechanism.

Raw h14 steering and prompt-prefill steering can move agreement-favored rows, but nearby-layer and residual controls break locality. Source masking and prompt rewriting move the behavior more strongly, but they act through broad prompt-source removal/recomputation.

### 4. Mechanistic Interpretation

Verdict: no supported mechanism.

The evidence supports several negative conclusions:

- dense residual directions behave like option-logit transport;
- answer-prefix dense directions do not rescue the claim;
- source tokens matter causally;
- single-head, single-layer, cumulative generation-query source masks do not recover the V9 effect;
- V12/V13 show the large source result is mostly source-removal/prompt-rewrite semantics, not a localized internal circuit.

### 5. Practical Value

Verdict: scientific/control-surface value only.

The internal interventions do not beat output-only monitoring as a deployable diagnostic. The source interventions are broad and side-effectful. The result is useful as a failed-mechanism card and as a warning about confounds, not as a controller.

### 6. Scope Discipline

Verdict: pass.

Allowed claim:

> Qwen3-0.6B has a reproducible control surface for MC-001 wrong-hint multiple-choice behavior, but the v1-v13 series does not identify a compact mechanism.

Disallowed claims:

- inner honesty;
- general truthfulness;
- universal sycophancy mechanism;
- hidden signature that beats output monitoring;
- clean h14 mechanism;
- deployable activation controller.

## Strongest Evidence

Positive evidence:

- v3: raw h14 `alpha=0.50` improves wrong-hint truth-following across calibration, holdout, and paraphrase holdout.
- v6: prompt-prefill raw h14 beats wrong-token and matched random additive controls on the hard bin.
- v9: hint-line source masking moves the agreement-favored hard bin from 0/104 to 63/104 truth-following; hinted-answer masking reaches 50/104.

Negative evidence:

- output margin reaches AUC 1.0;
- nearby h13 carries nearly the same raw h14 option-logit effect as h14;
- v8 answer-prefix dense direction fails against controls;
- v10 best individual head only reaches 7/103 hard-bin truth;
- v11 best all-layer generation-query hint-line mask reaches only 24/103;
- v12 literal rewrites nearly match input-mask behavior while query-only masks stay weak;
- v13 exact rendered-token-count matching reaches 52/103, not a new mechanism-like result.

## Strongest Objections

Objection: The project did not test every possible Qwen3-0.6B mechanism, such as MLP attribution, sparse feature discovery, or causal scrubbing.

Response: Correct. The closeout is not a theorem that no mechanism exists. It is a compute and project decision: the tested routes that were justified by the evidence either failed or became broad prompt-source semantics. The current substrate no longer offers a narrower object worth localizing.

Objection: Source masking found a real causal effect.

Response: Yes. That is why v10-v13 were run. The effect stayed broad and prompt-state-like after head, layer, cumulative attention, input-mask, rewrite, and layout/tokenization checks.

Objection: A larger Qwen3 model might behave differently.

Response: Yes. That is the recommended next mechanism-card stack. This review closes Qwen3-0.6B, not MC-001 globally.

## Final Decision

Reviewer decision:

- verdict: `control_without_explanation`
- strongest evidence: v3/v6/v9 show real control surfaces
- strongest objection: output-margin, locality, source-rewrite, and side-effect controls block mechanism interpretation
- missing control: no justified missing Qwen3-0.6B control remains before escalation; MLP/feature work should move to a more promising or artifact-rich stack
- recommended next action: package the Qwen3-0.6B failed-mechanism artifact and start the next mechanism-card attempt on Qwen3-1.7B or Gemma
- should this line receive more compute: no, not for broad Qwen3-0.6B mechanism search

## Closeout Verdict

The Qwen3-0.6B deep dive accomplished its project role.

It produced:

- a viable MC-001 factual-ladder substrate;
- a real but non-mechanistic control surface;
- a source-dependence result;
- multiple failed localization paths;
- a completed diagnostic/control-surface card;
- enough evidence to stop spending broad Qwen3-0.6B compute on MC-001.

The next work is not another Qwen3-0.6B sweep. It is either packaging or escalation to the next stack.
