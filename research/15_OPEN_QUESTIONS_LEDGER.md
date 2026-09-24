# Open Questions Ledger

This file records questions without letting them block the no-code rebuild.

## Resolved Decisions

### Decision: first target

Choose `MC-001` truth-versus-user-agreement conflict.

Reason:

It is objective, safety-relevant, measurable, and has clean nulls.

### Decision: first stack

Use `Qwen/Qwen3-0.6B` as the primary iteration target for MC-001.

Reason:

The user selected the smallest Qwen3 model to maximize iteration count. Gemma 3 4B-IT remains a later artifact-rich comparison if sparse-feature tooling becomes the bottleneck.

### Decision: first controlled pass verdict

Do not promote the current Qwen3-0.6B result to a supported mechanism card.

Reason:

The intervention has a real standard-holdout effect, but the hidden signature does not beat the output/logit baseline and paraphrase plus side-effect gates are not clean.

### Decision: output-margin-conditioned v2 verdict

Do not promote the Qwen3-0.6B v2 result to a supported mechanism card.

Reason:

Answer-logit margin still predicts agreement-vs-truth with AUC 1.0, the selected intervention has a negative calibration selection score, and nearby/random matched controls expose large answer-token side effects. V2 is useful failed-controls evidence, not a hidden-mechanism card.

### Decision: lower-dose residualized v3 verdict

Do not promote the Qwen3-0.6B v3 result to a supported mechanism card.

Reason:

Raw h14 `alpha=0.50` causally reduces wrong-hint agreement across calibration, holdout, and paraphrase holdout, but output margin still predicts agreement-vs-truth with AUC 1.0, the residualized h14 direction loses most of the effect, and matched/nearby controls remain active. V3 supports a diagnostic/control-surface artifact, not a hidden-mechanism card.

### Decision: prompt-prefill v5 verdict

Do not promote the Qwen3-0.6B v5 result to a supported mechanism card, but do run one more additive prefill control audit.

Reason:

V5 showed that raw h14 steering is not merely a repeated decode-step hook artifact: prompt-prefill-only h14 `alpha=0.50` moved the hardest agreement-favored rows from 0/104 to 21/104 truth-following, and `alpha=1.00` moved them to 41/95 parseable truth-following. But donor replacement remains near-baseline, output-margin and residual controls still block the mechanism claim, and v5 did not include full no-hint/correct-hint side-effect rows or matched additive prefill nulls.

### Decision: additive prefill controls v6 verdict

Do not promote the Qwen3-0.6B v6 result to a supported mechanism card. Close dense-prefill steering on Qwen3-0.6B as a control-only failed-mechanism path.

Reason:

V6 showed that raw h14 prompt-prefill steering beats the main additive nulls: on the 104 agreement-favored wrong-hint rows, raw h14 `alpha=0.50` moved truth-following from 0/104 to 21/104, while wrong-token prefill stayed at 0/104 and matched random `alpha=0.50` reached 3/104. But the same h14 vector applied at nearby h13 moved the hard bin to 25/104, and residual h14 applied at h13 also remained active. The intervention is real, but it is not h14-local and does not satisfy the mechanism-card locality standard.

### Decision: cross-layer transport v7 verdict

Do not promote the Qwen3-0.6B v7 result to a supported mechanism card. Close residual-stream dense transport on Qwen3-0.6B for MC-001.

Reason:

V7 showed that h13 and h14 have nearly identical raw h14 option-logit effects. On the 104 agreement-favored wrong-hint rows, raw h14 at h13 and h14 had delta Pearson 0.970 and mean correct-minus-wrong logit deltas +4.435 versus +3.964. The early h7 arm moved the hard bin but caused broad side effects, including all-validation `other_error` 140/288 and no/correct truth falling to 20/72. This supports residual-stream transport, not a local truth-versus-agreement mechanism.

### Decision: answer-prefix v8 verdict

Do not promote the Qwen3-0.6B v8 result to a supported mechanism card. Close dense answer-prefix steering on Qwen3-0.6B for MC-001.

Reason:

V8 showed that a neutral `Answer: ` prefix changes the behavior baseline, but the separately trained answer-prefix h14 direction does not produce a clean intervention. On wrong-hint rows, answer-prefix baseline was 83/180 truth-following with full parseability, while raw h14 at h14 `alpha=0.50` was 70/153 parseable truth-following. On the answer-prefix agreement-favored bin, baseline was 7/95 truth, target raw h14 was only 9/81 parseable truth, and residual random reached 10/95 with full parseability. This closes the dense answer-prefix branch.

### Decision: attention-source v9 verdict

Do not promote the Qwen3-0.6B v9 result to a supported mechanism card, but keep Qwen3-0.6B for one narrower head/layer path-local pass.

Reason:

V9 showed strong source-token dependence. On wrong-hint rows, hint-line masking moved truth-following from 61/180 to 124/180, while hint-answer masking moved it to 110/180. On the agreement-favored wrong-hint bin, baseline was 0/104 truth, hint-line masking reached 63/104, and hint-answer masking reached 50/104. Matched answer-instruction masking reached only 4/104, and matched random masking reached 5/103 parseable truth. The result is source-specific, but source masking is coarse and increases other errors, so it is not yet a mechanism card.

### Decision: head-localization v10 verdict

Do not promote the Qwen3-0.6B v10 result to a supported mechanism card. Single-head and single-layer attention-source localization is too weak.

Reason:

V10 found a late attention-source signal, but no compact head/layer mechanism. On the hard agreement-favored wrong-hint bin, the best individual head, `hint_answer` layer 22 head 9, moved truth-following from 0/103 to 7/103 while agreement remained 84/103 and other errors rose to 12/103. The best all-head layer masks were also small: `hint_answer` layer 22 and `hint_line` layer 22 each reached only 5/103 hard-bin truth. Matched answer-instruction masking reached 3/103, so weak single-layer effects are not precise enough to promote. At V10 time, the next 0.6B pass was cumulative layer-band source masking; V11 has now completed that follow-up.

### Decision: layer-band source v11 verdict

Do not promote the Qwen3-0.6B v11 result to a supported mechanism card. Cumulative generation-query source masking is real but still too weak and too broad.

Reason:

V11 showed a source-specific cumulative attention effect, but it recovered only a weak fraction of V9. On the hard agreement-favored wrong-hint bin, `hint_line__all_L00_L27` moved truth-following from 0/103 to 24/103, and `hint_answer__all_L00_L27` moved it to 22/103. Broad bands around L17-L22 recovered much of that weak V11 effect, but V9's coarse input-mask ablation had reached 63/104 for the hint line and 50/104 for the hinted answer. Random source controls stayed near baseline, while answer-instruction matched controls reached up to 15/103. The next 0.6B question is no longer "which attention band?" but "what did the V9 input-mask operation actually remove or disrupt?"

### Decision: input-mask semantics v12 verdict

Do not promote the Qwen3-0.6B v12 result to a supported mechanism card. Close broad Qwen3-0.6B MC-001 localization as diagnostic/control-only.

Reason:

V12 reproduced the large V9-style source effect and showed what it means. On the hard agreement-favored wrong-hint bin, `input_mask_hint_line` reached 63/103 truth-following and `input_mask_hint_answer` reached 49/103. Literal prompt rewrites nearly matched those effects: neutralizing the hint line reached 60/103, deleting the hint line reached 57/103, placeholdering the hinted answer reached 51/103, and deleting the hinted answer reached 48/103. Query-only source masks stayed much weaker at 24/103 for the hint line and 22/103 for the hinted answer. The result is source-specific but broad: hint non-answer text alone reached 44/103, source-removal arms raised other errors, and ordinary/correct-hint side rows degraded. V9 is therefore mostly source removal or prompt-state recomputation, not a compact generation-attention mechanism.

### Decision: layout-parity v13 verdict

Do not promote the Qwen3-0.6B v13 result to a supported mechanism card. Close the remaining Qwen3-0.6B MC-001 layout/tokenization caveat.

Reason:

V13 tested whether neutral replacements matched for character length or rendered prompt token count would move toward the V9-style input-mask ceiling. They did not. On the hard agreement-favored wrong-hint bin, `input_mask_hint_line` reached 63/103 truth-following, V12 reference rewrites reached 60/103 and 57/103, while both `rewrite_char_matched_neutral_hint_line` and `rewrite_token_matched_neutral_hint_line` reached 52/103. The token-matched arm preserved rendered prompt token count exactly on all rewritten rows. Query-only masking remained far lower at 24/103. The remaining difference is prompt-source/replacement semantics, not a mechanism-card-ready layout artifact.

### Decision: first product shape

Do not build a product first. Build mechanism cards and failed-card evidence first.

Reason:

The project has no right to productize hidden-state control until at least one mechanism card survives.

### Decision: MC002 known-unknown hallucination first gate

Do not start hidden-state work on the first MC002 known-unknown hallucination
substrate.

Reason:

The first Gemma 2 2B raw-generation gate changed behavior family from MC001 to
known-answer versus unsupported nonce-entity capitals, but failed the behavior
substrate controls. Real-country clean sources passed at 27 against a floor of
24, and nonce pressure hallucination was present at 24 against a floor of 12.
However, only 1 of 40 nonce-country sources abstained cleanly under both
neutral and cautious prompts, same-source nonce abstention-versus-hallucination
contrasts reached only 7 against a floor of 12, and real-country lure locality
failed with only 10 of 40 lure rows still correct. The next MC002 step must
repair the interface before any signature, steering, sparse-feature, or path
work.

### Decision: MC002 chat-render repair

Do not use base-Gemma chat-style rendering as the MC002 repair path.

Reason:

The chat-render repair held the same 80 sources, model, labels, split, and
success criteria fixed while changing only render mode from raw to chat. It
failed harder than raw rendering. Clean real-country sources dropped from 27 to
6, clean nonce-country sources improved only from 1 to 2, same-source nonce
contrasts dropped from 7 to 6, and real-country lure correctness dropped from
10 to 0. The run produced transcript-continuation and generic
misunderstanding artifacts, so hidden-state work would be confounded by render
format rather than known-versus-unknown behavior. The next MC002 repair should
use calibrated answer-text scoring or a true instruction-tuned model
comparison, not another base-model render tweak.

### Decision: MC002 answer-text scoring repair

Do not start hidden-state work on base-Gemma MC002 answer-scoring outputs.

Reason:

Answer-text scoring preserved real-country knowledge better than chat rendering
and exposed pressure-driven nonce hallucination on all 40 nonce sources, but it
still failed the behavior substrate. Clean real-country sources passed at 26,
but clean nonce-country sources remained 1 against a floor of 20, same-source
nonce contrasts reached only 9 against a floor of 12, and real-country lure
locality reached only 15 against a floor of 20. The failure is therefore not
just free-generation parsing. Even explicit candidate scoring usually prefers
nonce entity names or lure cities over `UNKNOWN`. The next MC002 attempt should
change model class or task construction before any signature, steering,
sparse-feature, or path work.

### Decision: MC002 instruction-tuned Gemma comparison

Do not start hidden-state work on Gemma 2 2B IT MC002 chat outputs.

Reason:

Instruction tuning reversed the base-model failure mode but did not satisfy the
behavior substrate. `google/gemma-2-2b-it` produced 36 clean real-country
sources and 40 clean nonce-country sources, but pressure-induced nonce
hallucination collapsed to 1 against a floor of 12, same-source nonce contrasts
collapsed to 1 against a floor of 12, and real-country lure correctness was
only 6 against a floor of 20. The model can abstain and answer known capitals,
but the current pressure arms are too weak and too nonlocal. The next MC002
attempt should calibrate pressure strength before any signature, steering,
sparse-feature, or path work.

### Decision: MC002 pressure calibration

Do not start hidden-state work on MC002 pressure-calibration outputs.

Reason:

The calibration preserved the instruction-tuned clean baseline but failed to
create a reliable pressure contrast. Baseline clean sources were 36 real and
40 nonce. `guess_mild` and `guess_strong` produced 0 nonce contrasts,
`city_required` produced only 1, `lure_soft` produced 1 while collapsing
real-country correctness to 6 and raising real abstention to 32, and
`lure_strong` produced only 6 nonce contrasts with 1 holdout contrast while
failing real-country locality controls. The next MC002 attempt should change
task construction, such as in-context supported versus unsupported entities or
a different known-unknown domain, before any signature, steering,
sparse-feature, or path work.

### Decision: MC002B context-support reconstruction

Do not start hidden-state work on MC002B context-support outputs.

Reason:

The exact-support reconstruction improved baseline measurement but still
failed the behavior substrate. `google/gemma-2-2b-it` produced 39 clean
context-supported sources and 35 clean context-unsupported sources, but
pressure did not induce a reliable same-source transition. `similar_allowed`
produced only 1 unsupported contrast, `closest_required` produced 0 while
dropping supported correctness to 28 and raising supported abstention to 11,
and `lure_check` produced only 1 contrast while collapsing supported
correctness to 1. The known-unknown line should be treated as behavior-gated
unless a qualitatively new task supplies an observable transition before
hidden-state discovery.

### Decision: MC003 delayed-copy behavior substrate

Use MC003 V2 as a behavior substrate for signature discovery.

Reason:

MC003 V1 found a strong wrong-suggestion transition but failed correct-hint
locality. V2 fixed that confound while holding the source bank, split, parser,
and success thresholds fixed. `google/gemma-2-2b-it` produced 40 clean
baseline sources, 40 correct-hint target rows, 40 soft-wrong-hint target rows,
and 35 `wrong_hint_pressure` source contrasts with 12 in holdout. This passes
the behavior gate only; it permits signature discovery but not intervention or
mechanism claims.

### Decision: MC003 delayed-copy signature

Do not start intervention work from the MC003 V2 final-prefix residual
direction.

Reason:

The layer-22 residual direction is a strong diagnostic signature but not a
mechanism-card-ready internal signature. It reached 0.994 source-disjoint
holdout AUC, exceeded the selected-layer shuffled-label p95 of 0.918, and held
across target-first and distractor-first holdout subgroups. However, the
first-token target-minus-distractor output-margin baseline at the same
`WAIT\nFINAL:` prefix reached 1.000 holdout AUC. The next MC003 signature
attempt must move earlier than the final prefix and beat output/trace controls
before steering, patching, sparse-feature, or circuit work begins.

### Decision: MC003 delayed-copy early signature

Do not start intervention work from the MC003 V2 early-position residual
direction.

Reason:

The early-position follow-up fixed the final-prefix output-margin problem but
failed two reliability controls. The selected `after_wait_newline` layer-14
direction reached 0.997 source-disjoint holdout AUC and beat the same-position
output-margin baseline of 0.913. However, the selected position/layer
shuffled-label null reached 0.998 p95 on holdout, and a simple
pressure-versus-non-pressure condition trace reached 0.991 holdout AUC. The
current MC003 signature table is therefore too condition-confounded for a
mechanism-signature claim. The next MC003 step, if any, must build
within-condition target-correct and distractor-followed variation before any
steering, patching, sparse-feature, or circuit work begins.

### Decision: MC003 delayed-copy V3 condition balance

Use `wrong_hint_balanced` as the frozen condition for one condition-balanced
signature attempt.

Reason:

MC003 V3 repaired the behavior-table confound that blocked the early-signature
run. Baseline and correct-hint locality remained clean at 40/40
target-correct. The `wrong_hint_balanced` condition produced 19
target-correct rows and 21 distractor-followed rows, with 12/15 in discovery
and 7/6 in holdout, and both holdout target-order subgroups contained both
labels. `wrong_hint_authority` also qualified at 23/17, but the preregistered
selection rule chose the smaller target-versus-distractor imbalance, so
`wrong_hint_balanced` is frozen before hidden-state inspection.

### Decision: MC003 delayed-copy V3 condition-balanced signature

Do not start intervention work from MC003 delayed-copy.

Reason:

The condition-balanced signature removed the V2 condition-trace confound and
the hidden signal failed on source-disjoint holdout. Global discovery selection
chose `after_wait` layer 25 with 0.994 discovery AUC, but holdout AUC was only
0.238. The same-position output-margin control reached 0.714 holdout AUC, the
selected shuffled-label p95 was 0.690, and target-order subgroup AUCs were
0.500 and 0.250. MC003 now has a real prompt-level behavior transition and a
condition-balanced behavior table, but no reliable internal signature on the
current 40-source delayed-copy setup.

### Decision: MC004 in-context binding V1

Do not start hidden-state work from MC004 V1.

Reason:

MC004 V1 cleanly tested nonce entity-to-code binding and passed locality:
neutral, cautious, and correct-hint rows were all 40/40 target-correct. But the
wrong-hint conditions were too weak to create a same-condition contrast.
`wrong_hint_pressure` and `wrong_hint_authority` produced only 2
distractor-following rows each, far below the 10/10 balance floor. The next
MC004 repair should change the pressure from a wrong user hint to an explicit
later-update conflict.

### Decision: MC004 in-context binding V2 behavior

Use `update_prefer_latest` as the frozen MC004 V2 condition for one
condition-balanced signature attempt.

Reason:

MC004 V2 repaired the behavior gate. Baseline clean sources were 39 and
correct-hint target sources were 40. `update_prefer_latest` produced 20
target-correct original-note answers, 19 update-following answers, and one
other row. Among target/update rows, discovery split was 14/12 and holdout
split was 6/7, with both holdout target-order subgroups containing both
labels. It was the only qualifying V2 condition.

### Decision: MC004 in-context binding V2 signature

Do not start intervention work from MC004 V2.

Reason:

The condition-balanced signature failed against holdout and output controls.
Global discovery selection chose `prompt_end` layer 20 with 1.000 discovery
AUC, but source-disjoint holdout AUC was only 0.738. Same-position
target-minus-update output margin reached 1.000 holdout AUC, the selected
shuffled-label p95 was 0.905, and target-first holdout subgroup AUC was only
0.500. MC004 now has a clean behavior table for in-context update conflict,
but no mechanism-card-ready hidden signature.

### Decision: MC004 in-context binding V2 lead-time

Do not start intervention work from the MC004 V2 lead-time direction.

Reason:

The lead-time audit found a pre-answer diagnostic signal but failed reliability
controls. Global discovery selection chose `after_question` layer 23, before
the later-update instruction, with 1.000 discovery AUC and 0.905
source-disjoint holdout AUC. Same-stage output margin was only 0.262, so the
hidden signal beat output at that stage. But the selected shuffled-label p95
was 0.929, and the target-first holdout subgroup reached only 0.667 AUC. The
signal is a clue for larger-bank update-conflict work, not a mechanism-card
signature.

### Decision: MC005 associative lookup source-edge

Do not promote MC005 to a compact mechanism card yet. Continue with a
layer-band/head-group localization audit.

Reason:

MC005 produced the strongest positive path result in the rebuild. Qwen3-1.7B
had 41/48 clean key/value lookup rows, with 28 discovery and 13 holdout clean
rows. The selected source-value attention signature, layer 16 head 14, reached
1.000 discovery AUC and 1.000 holdout AUC; it beat the shuffled-selection p95
of 0.911 and all source-order holdout subgroup AUCs were 1.000. Full-path
target source-value masking reduced holdout margin by -5.154 and flipped 11/13
target wins, while full-path distractor source-value masking increased margin
by +1.952 and flipped 0/13.

The locality control blocked mechanism-card promotion. The discovery-selected
causal layer was 25, where target source-value masking reduced holdout margin
by -0.606 and flipped 2/13 target wins, but the wrong-layer target source mask
was stronger at -0.913 and flipped 3/13. The selected attention head's own
target mask was directionally correct but small at -0.168 with 1/13 flips.
The result is therefore broad source-value path control plus a clean diagnostic
attention signature, not a localized layer/head mechanism.

### Decision: MC005 associative lookup band localization

Do not claim band-localized mechanism support from V2. Treat `late_20_26` and
`late_24_27` as strong diagnostics for the next preregistered run.

Reason:

The V2 band audit reproduced the fixed MC005 clean split: 41 clean rows, 28
discovery rows, and 13 holdout rows. Discovery target source-value masking
selected `all_layers`, with mean margin delta -5.071 and 25/28 target-win
loss. Because the preregistration said an `all_layers` selection is full-path
control only, the band-localization gate failed.

The late-band evidence is still strong. On holdout, `late_20_26` target
source-value masking reduced margin by -3.976 and flipped 8/13 target wins,
while `late_20_26` distractor source-value masking increased margin by +1.615
and random-value masking was essentially null at -0.005. Earlier target bands
were weak: `early_0_6` was 0.000, `mid_7_13` was -0.091, and
`signature_14_18` was -0.192. The next MC005 pass should preregister a
non-all-layer late-band selection rule, a larger row bank, prompt-layout
holdout, and head-group controls.

### Decision: MC005 associative lookup late-band V3

Treat MC005 as a narrow passed late-band control surface, but not yet a full
mechanism card.

Reason:

V3 excluded `all_layers` before selection, enlarged the row bank to 96 rows,
and added an arrow-layout holdout. The clean split passed with 81 clean rows:
40 discovery, 18 same-layout holdout, and 23 layout holdout. Discovery selected
`late_20_26`, where target source-value masking reduced margin by -4.345 and
flipped 29/40 target wins. On same-layout holdout, `late_20_26` target masking
reduced margin by -4.497 and flipped 10/18 target wins; distractor masking
increased margin by +1.733 and random-value masking was weak at -0.118. On
layout holdout, target masking reduced margin by -5.375 and flipped 17/23;
distractor masking increased margin by +3.196 and random-value masking was
weak at -0.071. Earlier bands stayed weak on both holdouts.

The effect remains broad over heads inside the band. Upper-half heads carry
more of the effect than lower-half heads, but all-head masking is substantially
stronger. The next MC005 step should be a reliability atlas: larger/disjoint
lexicon, more pair counts, longer contexts, generation-mode checks, off-target
side effects, and model-family replication.

### Decision: MC005 associative lookup reliability V4

Do not promote MC005 to a full mechanism card yet. Treat V4 as strong
lookup-scope reliability plus a failed off-target null.

Reason:

V4 fixed `late_20_26` and tested nine atlas scenarios: pair counts 3, 5, and
8; dash, arrow, and sentence layouts; a shifted holdout lexicon; greedy
next-token readout; and one off-target null. All 8/8 lookup scenarios worked.
Target source-value masking reduced the target-vs-distractor margin by -3.268
to -7.525 across lookup scenarios, flipped 14-26 target wins out of 32, and
usually removed greedy target outputs.

The atlas still failed because `offtarget_pair5_dash` was not a clean null.
Masking an unrelated source value moved the scored target margin by +0.713,
above the preregistered 0.50 null bound, and the matched random source mask
moved the margin by +0.994. This means the lookup path is robust over the
tested pair counts, layouts, shifted lexicon, and greedy readout, but source
masking has open side-effect boundaries. The next MC005 step should repair
off-target controls before model-family replication.

### Decision: MC005 associative lookup off-target null V5

Do not promote MC005 to a full mechanism card yet. Treat V5 as a repaired
primary-null pass plus a remaining same-grammar boundary.

Reason:

V5 fixed the `late_20_26` intervention and tested five off-target-null
scenarios. Four primary repaired nulls were clean: `reference_explicit_answer`,
`sentence_reference_explicit_answer`, `no_reference_note_explicit_answer`, and
`nonlookup_marker_answer` each had 32/32 baseline clean rows, zero target-win
loss under all source masks, and max absolute mean deltas between 0.098 and
0.295. This shows the late-band source mask is not a broad prompt-destroying
perturbation on explicit-answer, no-reference, or non-lookup prompts.

The strict suite still failed. The same-grammar diagnostic
`same_grammar_query_other_pair` had 29/32 baseline clean rows and stayed below
the weak-null bound, but it exceeded the strict 0.50 mean-delta bound on all
three source arms: +0.519, +0.533, and +0.619. The current boundary is therefore
same-grammar lookup sensitivity, not general off-target prompt collapse. The
next MC005 step should isolate same-grammar irrelevant-source effects with
source/key/punctuation/no-op mask controls before model-family replication.

### Decision: MC005 associative lookup same-grammar V6

Do not promote MC005 to a full mechanism card yet. Treat V6 as a source-line
null repair with an open query-label/position robustness issue.

Reason:

V6 decomposed same-grammar masking into irrelevant source value, paired key,
paired colon, non-source control value, and final query-label arms. The primary
seed-23 run passed all three scenarios: `query_other_pair_original`,
`query_other_pair_control_word`, and `answer_absent_reference_control` were all
`clean_null`, with 12/12 arms labeled clean.

The seed-17 robustness diagnostic sharpened the boundary. The two
query-other-pair scenarios stayed clean, and all source-line arms in
`answer_absent_reference_control` stayed clean or weak. But the final
query-label arm changed three target-win rows despite a small mean delta
(+0.059), so the scenario was `structural_boundary`. This means V5's
same-grammar source-value effect did not reproduce under source-line
decomposition, but query-label/position sensitivity remains open. The next
MC005 step should isolate final-label placement and row-flip sensitivity before
model-family replication.

### Decision: MC005 associative lookup query-label V7

Do not promote MC005 to a full mechanism card yet. Treat V7 as evidence that
source/control masks are stable, while final-position and prompt-surface
robustness remain open.

Reason:

V7 ran a 3-seed by 4-surface query-label sweep over the fixed `late_20_26`
intervention. All source/control arms were clean: there were zero non-clean
source value, source key, source colon, or non-source control-value arms across
the sweep. This strengthens the V6 conclusion that the V4/V5 side effect is not
best explained as a broad source-line masking problem.

The suite still failed. Six random-label scenarios were `invalid_baseline`,
showing that arbitrary final labels are not reliable answer surfaces. The
`generic_answer_label` surface had clean 32/32 baselines on all three seeds, but
its final-colon arm was weak every time, with mean deltas -0.641, -0.584, and
-0.627. V7 therefore moves the blocker to final-position punctuation and
prompt-surface reliability. The next MC005 step should keep the generic answer
surface, compare final colon against earlier matched colon and alternate answer
markers, and retain source/control arms as regression checks.

### Decision: MC005 associative lookup final-marker V8

Do not promote MC005 to a full mechanism card yet. Treat V8 as a
marker-specific reliability repair.

Reason:

V8 fixed the `late_20_26` intervention and tested four final answer markers
over seeds 17, 23, and 31. All structural controls stayed clean: source value,
non-source control value, and earlier neutral colon had zero non-clean arms.
This strengthens the conclusion that the remaining boundary is not source-line
or generic punctuation masking.

The full all-marker suite still failed, but two clean markers emerged.
`Response:` and `Output:` were `clean_null` on all three seeds. `Answer:` was
clean on seeds 17 and 23 but had a seed-31 final-colon weak arm with delta
-0.553. `Result:` was clean on seed 31 but had final-label weak arms on seeds
17 and 23 with deltas -0.547 and -0.576. The next MC005 pass should use
`Response:` or `Output:` as the answer marker and rerun a compact atlas that
combines lookup scenarios, source-line nulls, and off-target nulls under that
clean marker before model-family replication.

### Decision: MC005 associative lookup response-marker V9

Do not promote MC005 to a complete mechanism card yet. Treat V9 as a passed
compact `Response:` reliability atlas for Qwen3-1.7B.

Reason:

V9 fixed the `late_20_26` intervention and the `Response:` final answer marker,
then tested pair-count 5 and pair-count 8 lookup prompts plus off-target and
answer-absent nulls over seeds 17, 23, and 31. The atlas passed: all 6 lookup
scenarios were `works`, all 6 null scenarios were `clean_null`, and the suite
summary reported `passed: true`.

The lookup effects remained large and directionally specific. Across the six
lookup scenarios, target source-value masking reduced the target-vs-distractor
margin by roughly -9.729 to -10.746, while distractor source-value masking
moved the margin upward and random source-value masking stayed near zero.
The null arms stayed inside the strict clean bounds: same-grammar irrelevant
value/key/random masks were clean, and answer-absent source/control/earlier
colon/final-label/final-colon masks were clean.

This resolves the V7/V8 final-position blocker for the compact `Response:`
surface, but it still does not prove model-family generality, longer-context
robustness, or a single-head/single-layer circuit. The next MC005 step should
replicate under model-size or model-family variation, add longer pair-count
holdouts, and decompose heads/paths inside layers 20-26.

### Decision: MC005 associative lookup response-marker V10 size replication

Do not promote MC005 to a size-replicated mechanism card. Treat V10 as a
positive lookup replication on Qwen3-0.6B with a failed strict null gate.

Reason:

V10 fixed the V9 `Response:` atlas and changed only the model to
`Qwen/Qwen3-0.6B`. The lookup intervention replicated: all six lookup scenarios
were `works`, with target source-value mean deltas from -3.820 to -4.919 and
target-win losses from 14 to 21 rows. Distractor source-value arms moved
upward, and random source-value arms stayed near zero.

The preregistered suite still failed because not every null was `clean_null`.
Five of six null scenarios were clean, including all three off-target
same-grammar nulls. The seed-23 `answer_absent_response_null` was `weak_null`:
the non-source-control-value arm had only +0.039 mean delta but changed two
target-win rows, exceeding the strict one-row clean-null limit. This is a
small but preregistered reliability failure.

V10 therefore blocks a cross-size reliability claim. The next MC005 pass should
diagnose whether the Qwen3-0.6B weak null is a low-margin row-flip artifact, a
specific non-source-control-value issue, or a smaller-model marker/control
boundary.

### Decision: MC005 associative lookup response-marker V11 weak-null diagnostic

Treat the Qwen3-0.6B answer-absent null as a persistent reliability boundary,
not a small-row-count repair.

Reason:

V11 reran only the Qwen3-0.6B `answer_absent_response_null` under the fixed
`Response:` marker, using seeds 17, 23, 31, 37, and 41 with 128 rows per seed.
It saved row-level baseline and intervention margins for source value,
non-source control value, earlier neutral colon, final label, and final colon
arms.

The V10 seed-23 first-32 weak-null pattern reproduced. The two reproduced
non-source-control distractor-to-target flips were low-margin rows: one had
baseline margin 0.000 and the other had baseline margin -1.000. That confirms
the V10 failure was not a runner artifact and that the specific first-32 flips
were margin-fragile.

The larger row bank still failed. Only seed 17 was `clean_null`; seeds 23 and
37 were `weak_null`; seeds 31 and 41 were `side_effect`. The main 128-row
problem shifted from the exact V10 non-source-control arm to final-label and
final-colon target-win changes, with seed 31 final label losing three target
wins and seed 41 final colon changing four target-win rows. The diagnostic
class was `persistent_boundary`.

The next MC005 pass should test marker/control specificity on Qwen3-0.6B, not
rerun the same `Response:` answer-absent null as if more rows alone will repair
it. `Output:` is the obvious comparison because V8 found it clean on Qwen3-1.7B.

### Decision: MC005 associative lookup response-marker V12 0.6B marker specificity

Do not treat marker selection as a repair for the Qwen3-0.6B answer-absent
null. No tested marker was clean across five 128-row seeds.

Reason:

V12 compared `Response:`, `Output:`, `Answer:`, and `Result:` on the
Qwen3-0.6B answer-absent null using the V11 row-generation contract: seeds 17,
23, 31, 37, and 41; 128 rows per marker/seed; source value, non-source control
value, earlier neutral colon, final label, and final colon arms.

The `Response:` parity check reproduced the V11 boundary. `Output:` was the
best local repair on small slices: all five `Output:` first-32 scenarios were
`clean_null`. But the full 128-row `Output:` diagnostic still failed with two
`clean_null`, two `weak_null`, and one `side_effect` seed. Across all markers,
the full labels were 4 `clean_null`, 10 `weak_null`, and 6 `side_effect`.
Clean full-seed counts were `Response:` 1/5, `Output:` 2/5, `Answer:` 0/5, and
`Result:` 1/5. The diagnostic class was `no_clean_marker`.

MC005 therefore has a stable positive Qwen3-1.7B compact atlas and a stable
negative Qwen3-0.6B strict-reliability boundary. The next MC005 work should not
rerun marker selection on 0.6B. Either move forward on the Qwen3-1.7B positive
surface with longer-context/head-path work, or preregister a new smaller-model
null design as a repair attempt.

### Decision: MC005 associative lookup response-marker V13 long context

Do not claim complete longer-context reliability for MC005 on Qwen3-1.7B.
Treat V13 as positive lookup/off-target evidence through pair count 16 plus a
pair16 answer-absent final-label boundary.

Reason:

V13 fixed the Qwen3-1.7B `Response:` marker, the `late_20_26` intervention, and
the seeds 17, 23, and 31, then extended the V9 atlas to pair counts 12 and 16.
All six lookup scenarios were `works`, and all six off-target null scenarios
were `clean_null`. The pair-count 12 answer-absent null was also clean on all
three seeds.

The strict longer-context gate failed because seed 23
`answer_absent_pair16_response_null` was `weak_null`. The final-label arm had
mean delta -0.5039, just beyond the strict 0.50 clean bound, although it changed
zero target-win rows. This is a narrow margin-bound null failure rather than a
lookup failure or broad off-target collapse.

The next MC005 pass should diagnose the Qwen3-1.7B pair16 answer-absent
final-label boundary with more rows/seeds, row-level margins, and a pair-count
gradient between 12 and 16. Do not rerun the full lookup atlas until this null
boundary is understood.

### Decision: MC005 associative lookup response-marker V14 pair16 null diagnostic

Treat the V13 Qwen3-1.7B pair16 answer-absent weak null as sample-fragile under
the expanded diagnostic, not as a persistent Qwen3-1.7B longer-context null
failure.

Reason:

V14 fixed the Qwen3-1.7B `Response:` marker and `late_20_26` intervention, then
tested only answer-absent nulls at pair counts 12, 14, and 16. It used seeds 17,
23, 31, 37, and 41 with 128 rows per pair-count/seed. All 15 scenarios were
`clean_null`. Pair counts 12, 14, and 16 each had 5/5 clean seeds, no arm was
non-clean, and the diagnostic class was `all_clean_sample_fragile`.

The direct V13 failure did not reproduce. Seed 23 at pair count 16 had
final-label mean delta -0.3115 and zero target-win changes, versus V13's
32-row final-label mean delta -0.5039. V14 therefore repairs the Qwen3-1.7B
answer-absent pair16 boundary under this row-generation contract.

This does not repair the Qwen3-0.6B V10-V12 answer-absent reliability failure,
does not prove model-family generality, and does not localize MC005 to a
single head, layer, or minimal path. The next MC005 work should move to finer
head/path decomposition inside layers 20-26 while carrying V14's pair16
answer-absent null as a required holdout.

### Decision: MC005 associative lookup response-marker V15 fine localization

Do not promote MC005 to a compact localized mechanism. Treat V15 as evidence
that the Qwen3-1.7B intervention remains broad over the late band.

Reason:

V15 fixed the pair16 `Response:` lookup surface and screened preregistered
compact candidates on discovery seeds 17 and 23. The full layers-20-26 all-head
benchmark was not selectable and had discovery mean delta -8.8896 with 35
target-win losses. The selected compact candidate was `full_l20_26_upper_heads`
with discovery mean delta -3.5093 and five target-win losses.

On the seed-31 lookup holdout, the selected upper-head candidate remained
directional and source-specific: target source-value masking had mean delta
-3.3213 and six target-win losses, while distractor masking moved the margin
upward by +0.5752 and random masking was near zero at +0.0322. But the full
late-band target mask had mean delta -8.5283 and 21 target-win losses, so the
selected compact path recovered only 0.3894 of the full effect, below the
preregistered 0.60 effect-share floor.

The selected upper-head path preserved the V14 answer-absent null on seeds 37
and 41, so the failure is not a null/side-effect failure. The diagnostic class
was `full_band_required`. The next MC005 localization pass should not claim a
single layer, single layer-slice, or upper-head mechanism from V15. It should
either test finer head subsets inside the partial paths or use an attribution
method that can explain why the full late-band all-head intervention is much
larger than every preregistered compact candidate.

### Decision: MC005 associative lookup response-marker V16 path additivity

Treat the full Qwen3-1.7B late-band effect as mixed-superadditive, not as an
additive sum of independent compact head or layer pieces.

Reason:

V16 fixed the pair16 `Response:` lookup surface and scored 384 rows from seeds
17, 23, and 31. The full layers-20-26 all-head target mask had mean delta
-8.6110 and 112 target-win losses from a 381/384 target-win baseline.

Head partitions were not additive. Lower plus upper heads summed to -5.5086,
leaving a -3.1024 residual, or -36.0 percent of the full effect. Even plus odd
heads summed to -5.2404, leaving a -3.3706 residual, or -39.1 percent. Layer
partitions were also mostly superadditive. The three small slices
20-22/23-24/25-26 summed to -4.0540, leaving a -4.5570 residual, or -52.9
percent of the full effect.

The strongest new lead was `slice_l23_26_all`, which reached -6.2678 mean delta
and 51 target-win losses. The split `slice_l20_22_all` plus `slice_l23_26_all`
was near-additive, with only a -12.4 percent residual. That makes layers 23-26
the next localization target, but V16 does not itself promote that block because
it did not include answer-absent null holdouts for `slice_l23_26_all`.

The next MC005 pass should preregister `slice_l23_26_all` as a candidate and
test it on disjoint lookup and answer-absent null holdouts against the full
layers-20-26 benchmark.

### Decision: MC005 associative lookup response-marker V17 L23-26 localization

Treat `slice_l23_26_all` as the current supported Qwen3-1.7B localization
surface for MC005. Do not claim a single-head or single-layer mechanism.

Reason:

V17 tested `slice_l23_26_all` on fresh lookup seeds 43 and 47 with 96 rows per
seed, and answer-absent null seeds 53 and 59 with 96 rows per seed. The lookup
baseline was clean: 192/192 target wins and mean margin 10.5563.

The full layers-20-26 all-head benchmark had target mean delta -8.6820 and 51
target-win losses. The `slice_l23_26_all` candidate had target mean delta
-6.2664 and 23 target-win losses, recovering 0.7218 of the full benchmark
effect. Its source controls did not match the target effect: distractor source
masking moved the margin upward by +0.9113 with zero target-win loss, and random
source masking was near zero at +0.0127 with zero target-win loss.

The candidate also beat every smaller-slice control: `slice_l20_22_all` was
-1.3416 with zero target-win losses, `slice_l23_24_all` was -0.3818 with zero,
and `slice_l25_26_all` was -2.3957 with five. The answer-absent null holdout
was clean on seeds 53 and 59 across source value, non-source control value,
earlier colon, final label, and final colon arms. The diagnostic class was
`l23_26_localization_supported`.

The next MC005 localization pass should decompose layers 23-26 into smaller
sub-blocks and head partitions while preserving V13/V14/V17 lookup, source,
and answer-absent controls.

### Decision: MC005 associative lookup response-marker V18 L23-26 decomposition

Treat `slice_l24_26_all` as the current supported Qwen3-1.7B localization
surface for MC005. Do not claim a single-layer, single-head, or head-partition
mechanism.

Reason:

V18 used a split gate. Discovery lookup seeds 61 and 67 selected
`slice_l24_26_all` from 17 preregistered smaller layer/head candidates inside
layers 23-26. The selected path had discovery mean delta -5.7490 and 17
target-win losses, while the parent `slice_l23_26_all` had mean delta -6.3159
and 19 target-win losses.

On disjoint lookup holdout seeds 71 and 73, the parent `slice_l23_26_all` had
target mean delta -6.4159 and 33 target-win losses. The selected
`slice_l24_26_all` path had target mean delta -5.9295 and 23 target-win losses,
recovering 0.9242 of the parent effect. It remained the strongest selectable
holdout path; the next candidates were `slice_l23_25_all` at -3.1203 and
`slice_l24_25_all` at -3.0239. Coarse head partitions were weaker than the
selected three-layer block.

Source controls did not match the selected target effect: selected distractor
source masking moved the margin upward by +0.9631 with -2 target-win loss, and
selected random source masking was near zero at +0.0081 with zero target-win
loss. Answer-absent null seeds 79 and 83 were both `clean_null` across source
value, non-source control value, earlier colon, final label, and final colon
arms. All preregistered criteria passed with diagnostic class
`compact_l23_26_decomposition_supported`.

The next MC005 localization pass should decompose layers 24-26 into smaller
sub-blocks and layer-head intersections while preserving the V18
discovery/holdout split, source controls, and answer-absent null controls.

### Decision: MC005 associative lookup response-marker V19 L24-26 decomposition

Treat V19 as a failed finer-decomposition gate. Keep `slice_l24_26_all` as the
current supported Qwen3-1.7B localization surface for MC005.

Reason:

V19 used fresh split seeds. Discovery lookup seeds 89 and 97 selected
`slice_l24_25_all` from 23 preregistered smaller layer, head-partition, and
layer-head-intersection candidates inside layers 24-26. The selected path had
discovery mean delta -2.8882 and three target-win losses, while the parent
`slice_l24_26_all` had discovery mean delta -5.8120 and 14 target-win losses.

On disjoint lookup holdout seeds 101 and 103, the parent `slice_l24_26_all`
had target mean delta -5.9502 and 27 target-win losses. The selected
`slice_l24_25_all` path had target mean delta -2.9681 and nine target-win
losses, recovering only 0.4988 of the parent effect. It stayed rank-stable as
the strongest selectable holdout path, and it beat source controls: selected
distractor source masking moved the margin upward by +0.4928 with zero
target-win loss, while selected random source masking was near zero at -0.0023
with one target-win loss.

Answer-absent null seeds 107 and 109 were both `clean_null` across source
value, non-source control value, earlier colon, final label, and final colon
arms. The only failed preregistered criterion was effect share. The diagnostic
class was `l24_26_block_still_required`.

The next MC005 pass should stop trying to promote a smaller path directly and
instead diagnose the interaction inside layers 24-26: pairwise additivity,
leave-one-layer-out/omission tests, and whether layer 26 is necessary because
of a superadditive interaction rather than an independently strong path.

### Decision: MC005 associative lookup response-marker V20 L24-26 interaction

Treat `slice_l24_26_all` as a supported three-layer interaction block, not as a
set of independently strong layer components.

Reason:

V20 used fresh lookup seeds 113 and 127 with 128 rows per seed. The parent
`slice_l24_26_all` target source-value mask replicated with mean delta -5.9028
and 32 target-win losses. Parent source controls did not match it: distractor
source masking moved the margin upward by +1.0309 with -4 target-win loss, and
random source masking was near zero at -0.0060 with zero target-win loss.

Every leave-one-layer-out pair stayed below the 60 percent parent-effect
threshold: `slice_l24_25_all` recovered 0.5193 of the parent, the 24+26 pair
recovered 0.3751, and `slice_l25_26_all` recovered 0.3952. Single layers were
much weaker: layer 24 recovered 0.0998, layer 25 recovered 0.1559, and layer 26
recovered 0.1589 of the parent.

The single-layer sum was superadditive relative to parent: components summed
to -2.4469, leaving a -3.4559 residual, or -58.5 percent of the parent effect.
Every pair-plus-omitted-layer decomposition was also superadditive relative to
parent. Answer-absent null seeds 131 and 137 were both `clean_null` under the
parent path. The diagnostic class was
`l24_26_three_layer_interaction_supported`.

The next MC005 pass should keep layers 24-26 as the current interaction block
and test harder reliability axes for that block: layout and lexicon holdouts,
pair-count stress, row-level interaction structure, and continued answer-absent
null/source-control checks.

### Decision: MC005 associative lookup response-marker V21 L24-26 stress

Treat `slice_l24_26_all` as stress-supported across the tested layout,
lexicon, and pair-count axes. Do not expand the claim beyond the synthetic
associative lookup contract or beyond Qwen3-1.7B.

Reason:

V21 fixed the V20 parent path and tested six lookup stress scenarios: pair16
dash/colon base, pair16 arrow layout, pair16 sentence layout, pair16
dash/colon shifted lexicon, pair20 dash/colon, and pair20 arrow with shifted
lexicon. All six had valid baselines and all six were labeled `works`.

Target source-value mean deltas stayed large in every scenario:
`pair16_dash_base` -5.8174, `pair16_arrow_layout` -7.4199,
`pair16_sentence_layout` -6.6816, `pair16_dash_lexicon_shift` -5.3594,
`pair20_dash_pairstress` -5.9854, and `pair20_arrow_lexicon_stress` -7.3662.
The target-win losses were 6, 6, 9, 6, 6, and 4 respectively.

Source controls did not match the target effect. Distractor source-value
masking moved margins upward in every scenario, from +0.8164 to +1.3516.
Random source-value masking stayed near zero, from +0.0020 to +0.0361.
Fresh answer-absent null seeds 179 and 181 were both `clean_null`.

The diagnostic class was `l24_26_stress_supported`.

The next MC005 pass should either test row-level interaction structure for the
layers-24-26 block or use a new preregistered model-family/null design. It
should not attempt another smaller-path promotion unless the design directly
addresses the V19/V20 negative decomposition boundary.

### Decision: MC005 associative lookup response-marker V22 row interaction

Do not promote the layers-24-26 interaction to a broad row-level all-three
mechanism. Treat V22 as a negative refinement: the parent/control/null result
replicated, but the preregistered row-level all-three fraction failed.

Reason:

V22 used fresh lookup seeds 191 and 193 with 128 rows per seed. The parent
`slice_l24_26_all` target source-value mask replicated with mean delta -5.9562
and 33 target-win losses. Parent source controls did not match it: distractor
source masking moved the margin upward by +0.8573 with -2 target-win loss, and
random source masking stayed near zero at +0.0040 with zero target-win loss.

The aggregate decomposition remained similar to V20: `slice_l24_25_all` had
mean delta -3.0098, the 24+26 pair had -2.1232, and `slice_l25_26_all` had
-2.4711. Single layers remained much weaker: layer 24 was -0.4637, layer 25
was -0.9781, and layer 26 was -0.9690.

The row-level result was mixed. There were 253 parent-effect rows, 67
all-three margin rows, and 20 parent-flip pair-resistant rows. The median
single-layer-sum residual on parent-effect rows was -3.1875, so the parent
still looked superadditive on a typical parent-effect row. But all-three rows
were only 26.48 percent of parent-effect rows, below the preregistered 40
percent threshold. The median best leave-one-layer-out pair share was 0.7143,
which means most parent-effect rows still allowed some pair to recover at
least 60 percent of the parent negative-effect magnitude.

Fresh answer-absent null seeds 197 and 199 were both `clean_null`. The
diagnostic class was `mean_only_interaction`.

The next MC005 pass should stratify the row heterogeneity before another
intervention: query index, source position, margin band, token identity, and
best-pair winner are the first obvious cuts. The current allowed claim should
say aggregate interaction control surface with partial row-level all-three
evidence, not a broad row-level all-three mechanism.

### Decision: MC005 associative lookup response-marker V23 row heterogeneity

Treat source position and query index as the current leading explanation for
V22 row heterogeneity. Do not treat source position as causal yet.

Reason:

V23 was an offline diagnostic over the fixed V22 artifact. It used no new model
scoring. The source artifact was
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v22_row_interaction_20260630T201924.json`
with SHA256 `0b10c833d27a36666691973fe2dbde05de1a1ed74ef16220a1c43aa0299b627d`.

The diagnostic label was `best_pair_position_heterogeneity`, with
`target_source_position` as the dominant family. Eligible source-position
groups had a max-minus-min all-three fraction contrast of 0.4375. The earliest
target source position, token position 6, had 0/16 all-three rows. Several
mid/late positions reached 7/16 all-three rows: positions 36, 56, 71, and 76.
The query-index table mirrored this because the V22 prompt layout maps query
index to source position.

Baseline margin also showed a strong contrast, but it is secondary under the
preregistered priority order. `q1_low` baseline-margin rows had 9/64 all-three
rows and 18 parent-flip pair-resistant rows, while `q4_high` rows had 28/63
all-three rows and zero parent-flip pair-resistant rows. That means parent-only
behavior flips concentrate in low-margin rows, while all-three margin structure
is most common in high-margin rows.

Failure-reason accounting showed that all 186 non-all-three parent-effect rows
failed because at least one leave-one-layer-out pair recovered 60 percent or
more of the parent negative-effect magnitude. Single-layer residual weakness
was not the main blocker.

The next MC005 pass should causally test source position: hold key/value
identity and query identity fixed while moving the target pair between early
and mid/late source positions, then score parent, leave-one-layer-out pairs,
source controls, and answer-absent nulls.

### Decision: MC005 associative lookup response-marker V24 source-position causal

Treat source position as a directional contributor to the V22/V23 row split,
not as a sufficient single-factor causal explanation. Do not promote a
source-position mechanism claim.

Reason:

V24 causally moved the same target key/value/query identity between early and
mid/late source positions, then rescored the fixed parent `slice_l24_26_all`,
leave-one-layer-out pairs, source-value controls, and answer-absent null
holdouts. The result artifact was
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v24_source_position_causal_20260630T203903.json`
with SHA256 `78925ef6909a0b3c2008b7740553b429e4c107b2c1ae4d5fc5aba52ad49a2876`.

The parent/control/null checks passed. Early-position rows had baseline
127/128, parent mean delta -5.6353 with 26 target-win losses, clean distractor
and random controls, and an all-three fraction of 0.1575. Mid/late rows had
baseline 126/128, parent mean delta -6.3171 with 14 target-win losses, clean
distractor and random controls, and an all-three fraction of 0.3175. Median
best-pair share moved in the predicted direction from 0.7381 to 0.6769, and
the best-pair-share criterion passed.

But the preregistered source-position causality thresholds failed. The
mid/late minus early all-three fraction gain was about +0.1600, below the
required +0.20. The paired family net gain was +7, below the required +12. The
diagnostic class was `position_fraction_not_causal`.

The next MC005 pass should test a stronger factorial account instead of source
position alone: source position, baseline margin, distractor position, and
value identity should be varied or stratified together while preserving parent
source controls, answer-absent nulls, and the V21/V24 layout discipline.

### Decision: MC005 associative lookup response-marker V25 factorial row heterogeneity

Treat simple prompt-layout factorial explanations as failed. The current
row-level heterogeneity explanation is still incomplete, even though the
aggregate layers-24-26 parent surface remains strong and clean.

Reason:

V25 crossed target source position with distractor source position while
blocking target/distractor value identity by family and stratifying by observed
baseline margin. The result artifact was
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
with SHA256 `efdab6fad30de1762b8e92f7e46c7f4e1e160bcc988ad093c7d0bf9a5617b9cf`.

The core validity checks passed. Overall lookup baseline was 256/256. Every
factorial cell had 64/64 baseline target wins and 64 parent-effect rows. The
parent `slice_l24_26_all` target-value mask had mean delta -6.1912 and 27
target-win losses overall, while the distractor and random source controls did
not match the target effect. Answer-absent null seeds 251 and 257 were both
`clean_null`.

The factorial explanation failed because the cell table was nearly flat:

- `early_near`: 17/64 all-three rows, fraction 0.2656;
- `early_far`: 17/64 all-three rows, fraction 0.2656;
- `mid_late_near`: 20/64 all-three rows, fraction 0.3125;
- `mid_late_far`: 19/64 all-three rows, fraction 0.2969.

The best-minus-worst all-three fraction range was only 0.0469, far below the
0.25 threshold. Distractor relation modulated the source-position effect by
only 0.0156, below the 0.15 threshold. Baseline margin remained directional
but weak: the largest high-minus-low within-position contrast was 0.0864,
below the 0.20 threshold. The diagnostic class was
`factorial_cell_contrast_failed`.

The next MC005 pass should stop trying to explain row-level heterogeneity with
simple source/distractor layout factors. The useful next target is an internal
state predictor for which parent-effect rows become all-three rows: for
example, a pre-intervention residual, attention, or logit-margin signature that
predicts pair resistance inside the already reliable parent surface.

### Decision: MC005 associative lookup response-marker V26 internal row signature

Treat V26 as the first positive internal signature for MC005 row-level
heterogeneity. At V26 time, do not treat it as a mechanism card, because the
signature has not been causally intervened on.

Reason:

V26 used the V25 artifact as its source:
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
with SHA256 `efdab6fad30de1762b8e92f7e46c7f4e1e160bcc988ad093c7d0bf9a5617b9cf`.
The V26 result artifact was
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v26_internal_row_signature_20260630T210810.json`
with SHA256 `ba8b890de3ca0ed518aa6e3a4d963ee4838285798a7a6a4cbf9dbcf335bcaf03`.

The diagnostic predicted `all_three_margin_row` among V25 parent-effect rows.
Discovery seed 233 had 29 positives and 99 negatives. Holdout seed 239 had 44
positives and 84 negatives. The selected internal candidate was the
pre-intervention residual stream at layer 20 on the target source-value token:
`l20_target_value`.

The signature passed all preregistered gates:

- discovery AUC 0.9840 and holdout AUC 0.7933;
- best non-internal baseline was `baseline_margin`, holdout AUC 0.5586;
- shuffled-selection holdout AUC p95 was 0.6510;
- holdout subgroup AUCs were stable: early 0.7909, mid/late 0.7938, far
  distractor 0.7973, near distractor 0.7890.

This means the V22/V25 all-three row subset is not explained by simple prompt
layout or output-margin shortcuts alone. It is now tied to a measurable
internal state feature. That set up V27's intervention test: modulate or patch
the `l20_target_value` signature while preserving prompt text, then score
whether parent/pair row structure changes in the predicted direction under
source controls and answer-absent null holdouts.

### Decision: MC005 associative lookup response-marker V27 signature intervention

Treat V26 as predictive but not causal under the tested additive residual
intervention. Do not promote the row signature to a mechanism card.

Reason:

V27 used the V25 and V26 artifacts as sources:
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
with SHA256 `efdab6fad30de1762b8e92f7e46c7f4e1e160bcc988ad093c7d0bf9a5617b9cf`,
and
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v26_internal_row_signature_20260630T210810.json`
with SHA256 `ba8b890de3ca0ed518aa6e3a4d963ee4838285798a7a6a4cbf9dbcf335bcaf03`.
The V27 result artifact was
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v27_signature_intervention_20260630T212004.json`
with SHA256 `830914646f67105d5ee9532f0319adf6a8026ed0d17d758d7a425335c56bb605`.

The source signature reproduced: discovery AUC was 0.9840 and holdout AUC was
0.7933 for `l20_target_value`. But the primary `plus_target` residual
intervention did not move the row structure in the predicted direction.
No-intervention holdout rows had 44/128 all-three rows with median best-pair
share 0.6614. `plus_target` had 43/128 all-three rows with median best-pair
share 0.6700. `minus_target` had 48/128 rows, and random, final-colon, and
distractor-position controls each had 46/128 rows.

The parent source controls under `plus_target` passed: the target-value source
mask had mean delta -6.1689 and 11 target-win losses, while distractor and
random source controls had zero target-win losses. Residual answer-absent nulls
for seeds 251 and 257 were clean across `plus_source_value`,
`plus_final_colon`, and `random_source_value`.

The failed preregistered criteria were: `plus_target` all-three gain,
`plus_target` best-pair-share drop, and `minus_target` opposition. The
diagnostic class was `plus_target_no_row_effect`.

The next MC005 row-mechanism pass should not repeat simple additive residual
steering of the V26 direction. A future attempt needs a new preregistered
intervention family, such as donor patching, score-matched replacement,
attention/write-path intervention, or a nonlinear feature probe, with V27 as a
required negative control.

### Decision: MC005 associative lookup response-marker V28 donor replacement

Treat V28 as a failed non-additive intervention on the V26 row signature. Do
not promote donor activation replacement to a row-control mechanism.

Reason:

V28 used V25, V26, and V27 as sources. The V28 result artifact was
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v28_donor_replacement_20260630T213704.json`
with SHA256 `ab0b92f89c3cfcb384d8014bf15c3e2a9d269dfe4569f640d13c1bd183668c06`.

The source signature reproduced: discovery AUC was 0.9840 and holdout AUC was
0.7933. Positive donor replacement used 29 discovery all-three donors and
matched all 128 held-out rows on non-hidden layout/margin fields. But
`positive_target` damaged the surface it was supposed to modulate: baseline
target wins fell from 128/128 to 106/128, and parent-effect rows collapsed from
128 to 3. Its apparent all-three fraction gain, 0.3438 to 0.6667, is therefore
a denominator artifact, not a mechanism success.

Parent source controls failed under `positive_target`: the target-value source
mask was weak at mean delta -0.1729 with 2 target-win losses, while the
distractor source control changed 14 target-win rows in the opposite direction.
Replacement nulls also failed because `positive_final_colon` caused 11
target-win losses on seed 251 and 21 on seed 257.

The diagnostic class was `source_control_failed`, with null failure also
present. The current row-signature branch has now failed both additive residual
steering and matched donor activation replacement.

The next MC005 mechanism pass should either target the layers-24-26 write path
directly, or pause this row-signature branch and move to a fresh behavior family
with a cleaner intervention surface.

### Decision: MC005 associative lookup response-marker V29 attention write replacement

Treat V29 as a bounded positive write-path mediation result, not a full
mechanism card.

Reason:

V29 used the V25 artifact as source:
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
with SHA256 `efdab6fad30de1762b8e92f7e46c7f4e1e160bcc988ad093c7d0bf9a5617b9cf`.
The V29 result artifact was
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v29_attention_write_replacement_20260630T214645.json`
with SHA256 `c90bbd9a65fdd3866ef4e19e787d7c93b7faa693a89ddd0f843e58829097804b`.

The lookup mediation result was exact. Direct layers-24-26 target source
masking had mean delta -6.1978 and 11 target-win losses. Replacing only the
final-query self-attention output writes in layers 24-26 with the masked
counterfactual writes produced the same mean delta, -6.1978, and the same 11
target-win losses. Distractor and random write replacements matched the clean
direct controls: distractor moved margin upward by +0.8223 with zero losses,
and random moved +0.0239 with zero losses. Single-layer write replacements did
not recover the parent effect.

The strict reliability gate failed only on answer-absent seed 251:
`non_source_control_value` write replacement had mean delta +0.1406 and
`target_win_loss = -1`, a one-row target gain. Seed 257 was clean, and the
source-value/final-colon null arms were clean on seed 251. Because the
preregistered null rule required exact zero row changes, the diagnostic class
was `null_failed`.

V30 has now tested this boundary directly.

### Decision: MC005 associative lookup response-marker V30 write null sweep

Do not promote the layers-24-26 attention-write surface to a full mechanism
card.

Reason:

V30 used the V25 source artifact and V29 source artifact:
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
with SHA256 `efdab6fad30de1762b8e92f7e46c7f4e1e160bcc988ad093c7d0bf9a5617b9cf`,
and
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v29_attention_write_replacement_20260630T214645.json`
with SHA256 `c90bbd9a65fdd3866ef4e19e787d7c93b7faa693a89ddd0f843e58829097804b`.
The V30 result artifact was
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v30_write_null_sweep_20260630T215937.json`
with SHA256 `fab8f176ff5fd8a618529315bc547070555073ea9ab0c55e6db4c18683c13543`.

The V25 replay panel reproduced the V29 failure: seed 251
`non_source_control_value` write replacement changed one answer-absent row from
target-losing to target-winning (`target_win_loss = -1`) with mean delta
`+0.1406`. The fresh 8-seed, 1,024-row answer-absent sweep then found three
more strict row-change failures: seed 263 `non_source_control_value` had one
target-win gain, and seed 283 had one target-win loss under both
`source_value` and `non_source_control_value`. All fresh arms stayed inside the
absolute mean-delta tolerance of 0.25, so the failure is low-margin row sign
flips rather than broad aggregate drift.

The diagnostic class was `fresh_write_null_failed`. V29 remains exact lookup
mediation on its holdout rows, but V30 shows the answer-absent
write-replacement null boundary persists on fresh rows.

The next MC005 pass should either refine the write intervention to preserve
low-margin answer-absent null rows while retaining lookup mediation, or stop
trying to promote this surface and move to a different behavior family for the
first full mechanism card.

### Decision: MC005 associative lookup response-marker V31 margin boundary

Do not claim that the V29/V30 null boundary is fully explained by an absolute
baseline-margin cutoff of 0.5.

Reason:

V31 used V25, V29, and V30 as source artifacts and produced:
`results/cards/MC005/mc005_qwen3_1p7b_response_marker_v31_margin_boundary_20260630T220916.json`
with SHA256 `bd5e0e04bc6abd760256a7100e7c618b3fe8fe4fdfc4405f7808347f193947ff`.

The lookup target write effect reproduced exactly: 128/128 baseline target
wins, mean delta -6.1978, and 11 target-win losses. All 11 lookup loss rows had
baseline margin greater than 2.0, with minimum baseline margin 5.25.

The V31 fresh null panel found one additional target-win gain on seed 313
under `non_source_control_value`, with baseline margin 0.0000 and arm margin
+0.8750. All fresh null arms stayed inside the absolute mean-delta tolerance of
0.25.

However, V31 imported the V30 changed rows as preregistered. The original V30
replay row from seed 251 had absolute baseline margin 0.75, so the criterion
that all imported null flips have absolute baseline margin at most 0.5 failed.
The diagnostic class was `null_boundary_broad`.

The useful map is therefore:

- lookup write replacement has high-margin causal target effects;
- fresh null flips are rare and near the margin boundary;
- the strict 0.5 cutoff is too narrow because one replayed null flip sits at
  absolute margin 0.75;
- no mechanism-card promotion is allowed without changing the intervention or
  accepting this as a mapped failure mode.

### Decision: MC006 parametric fact override smoke

Do not start hidden-state discovery on the first MC006 parametric-fact versus
context-overwrite interface.

Reason:

MC006 opened a new behavior family after the MC005 write-surface boundary:
parametric country-capital facts versus task-local false context overwrites on
`Qwen/Qwen3-1.7B`. The result artifact was:
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_smoke_20260630T221737.json`
with SHA256 `75c8f1b4a230ffff92216cfd1a8ca71f27c816ff7ed9059fdc07a7f2dbaa9b9e`.

Structural checks passed: 40 sources, 200 rows, five conditions per source,
40 rows per condition, source splits valid, no duplicate row IDs, and no
duplicate candidate answers.

The behavior gate failed:

- `no_context`: true answer 26/40, below the 30/40 floor;
- `true_context`: true answer 40/40, passing;
- `irrelevant_context`: true answer 27/40, below the 30/40 floor;
- `task_override`: override answer 39/40, passing;
- `mistake_context`: true answer 0/40 and override answer 40/40, failing both
  the true-answer floor and the override-leak ceiling;
- clean source-level contrasts: 0/40.

The diagnostic class was `parametric_fact_failed`. The stricter interpretation
is that the interface is also a context-authority artifact: when a false
reference line is present, the model follows it even when the instruction says
the line may contain a mistake.

The next MC006 step should repair the behavior interface before any signature,
patching, sparse-feature, or path work. A repair should separate real-world and
task-local prompt modes more strongly, avoid highly salient alternate city
candidates, and add a stronger real-world-only control without reference-line
syntax.

### Decision: MC006 parametric fact override V2 repair

Do not start hidden-state discovery on the full MC006 V2 prompt suite.

MC006 V2 replaced the V1 false reference-line prompt with a false-claim audit
and a fictional codebook interface. The run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v2_repair_20260630T222815.json`
with SHA256 `7ae7ae5a1d1a9cc1a36d7ac9684432b913dfc2bd9ecfb1ec556de76f66ebca8b`.

The structural gate passed:

- 40 sources;
- 200 rows;
- five rows per source;
- 40 rows per condition;
- valid deterministic source splits;
- no duplicate row IDs or candidate answers;
- valid candidate token counts.

The behavior gate failed:

- `direct_real`: true answer 25/40, below the 32/40 floor;
- `true_fact`: true answer 40/40, passing;
- `false_claim_audit`: true answer 29/40, just below the 30/40 floor;
- `false_claim_audit`: override answer 8/40, at the allowed ceiling;
- `fictional_override`: override answer 40/40, passing;
- `real_after_fiction`: true answer 9/40, below the 30/40 floor;
- `real_after_fiction`: override answer 21/40, above the 8/40 ceiling;
- strict clean contrasts: 9/40 sources, below the 16-source floor;
- holdout strict clean contrasts: 1/8 sources, below the 6-source floor.

The diagnostic class was `parametric_fact_failed`. The stricter interpretation
is that the direct candidate-scoring interface still prefers salient non-capital
cities too often, and the fictional codebook rule contaminates later real-world
answering even when the prompt says to ignore the codebook for geography.

Useful lead, not a pass: 23/40 sources passed the narrower three-way contrast
`direct_real=true_answer`, `false_claim_audit=true_answer`, and
`fictional_override=override_answer`; 6 of those sources were holdout. A future
MC006 V3 may preregister a source-selected capital-fact substrate using that
lead, but it must explicitly document or replace the failed `real_after_fiction`
locality boundary.

### Decision: MC006 V3 source-selected paraphrase test

Do not start hidden-state discovery from MC006 V3.

MC006 V3 source-selected the 23 V2-clean three-way sources and tested
paraphrased direct-real, true-fact, false-claim, and fictional-code prompts. The
run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v3_source_selected_20260630T223516.json`
with SHA256 `d2483434bf26a89d09fe1c3223bc6d7914eb1945e9409d43aaf5a6fc35d1f4b1`.

Structural checks passed: 23 selected sources, 6 original holdout sources, 92
rows, four rows per source, 23 rows per condition, valid splits, no duplicate
IDs or candidate answers, and valid candidate token counts.

The real-world side passed:

- `direct_real_paraphrase`: true answer 23/23;
- `true_fact_paraphrase`: true answer 23/23;
- `false_claim_check`: true answer 22/23;
- `false_claim_check`: override answer 0/23.

The fictional-code side failed:

- `fictional_code_lookup`: override answer 18/23, below the 20/23 floor;
- clean source-level contrasts: 17/23, below the 18-source floor;
- original holdout clean contrasts: 5/6, passing.

The diagnostic class was `fictional_code_pressure_failed`. V3 shows that the
source-selected real-world side is stable under paraphrase, but the fictional
codebook pressure is prompt-sensitive.

### Decision: MC006 V4 source-selected hybrid substrate

MC006 V4 passed the source-selected behavior-substrate gate. Hidden-state
signature discovery may proceed only on the exact V4 source-selected,
prompt-bounded contract.

V4 made one controlled change from V3: it kept the 23 selected sources and the
V3 real-world prompts, but restored the stronger V2 fictional-codebook wording.
The run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v4_source_selected_hybrid_20260630T223734.json`
with SHA256 `b3d153dde317161e2c028c7ff23aeeafd32d00a8b8d4dbb21415841d1360f32e`.

Structural checks passed: 23 selected sources, 6 original holdout sources, 92
rows, four rows per source, 23 rows per condition, valid splits, no duplicate
IDs or candidate answers, and valid candidate token counts.

All behavior criteria passed:

- `direct_real_paraphrase`: true answer 23/23;
- `true_fact_paraphrase`: true answer 23/23;
- `false_claim_check`: true answer 22/23;
- `false_claim_check`: override answer 0/23;
- `fictional_code_lookup`: override answer 23/23;
- clean source-level contrasts: 22/23;
- original holdout clean contrasts: 5/6.

The diagnostic class was `source_selected_v4_substrate_passed`.

Allowed next step:

- run hidden-state signature discovery on the V4 behavior table;
- report source-split holdout, output/logit-margin baselines, and shuffled-label
  or row-label nulls;
- keep V1 reference-authority failure, V2 `real_after_fiction` contamination,
  and V3 weak-fictional-code prompt sensitivity as explicit reliability
  boundaries.

### Decision: MC006 V5 hidden-signature gate

Do not start intervention from the MC006 V5 hidden signature.

V5 tested hidden-state signature discovery on V4 clean source-level contrasts
only: 22 clean sources, 5 original holdout sources, and 88 rows. The run is
archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v5_signature_20260630T224601.json`
with SHA256 `97d322a974dc5101deac60795333edb3bcf14cf9866b08518db4eae966b5954a`.

Structural checks passed:

- 22 clean sources;
- 5 original holdout sources;
- 88 rows;
- 22 rows per V4 condition;
- both labels present in discovery and holdout;
- source splits valid;
- no duplicate row IDs.

The hidden-state probe found a strong separation:

- selected candidate: final-prompt-token `layer_0`;
- selected hidden discovery AUC: 1.000;
- selected hidden holdout AUC: 1.000;
- every holdout source had higher mean real-world score than fictional-code
  score.

But the controls matched or invalidated it:

- output-margin holdout AUC: 1.000;
- prompt-length holdout AUC: 1.000;
- prompt-format holdout AUC: 1.000;
- shuffled-label selection p95: 1.000.

The diagnostic class was `output_margin_confounded`. The stricter
interpretation is broader: V5 is output-margin confounded, prompt-format
confounded, and shuffle-selection confounded. The V4 behavior substrate remains
useful, but V5 does not provide a mechanism-grade internal signature.

Allowed next step:

- build a condition-balanced MC006 signature table where both labels occur
  inside the same prompt family; or
- explicitly control prompt length and prompt-format before hidden-state
  selection.

### Decision: MC006 V6 condition-balanced behavior repair

Do not start hidden-state discovery from MC006 V6.

V6 implemented the V5 repair as a behavior gate: 22 V4-clean sources, 5 original
holdout sources, 44 rows, two rows per source, one prompt family, and
counterbalanced query tags. The run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v6_condition_balanced_20260630T225327.json`
with SHA256 `3ee27a679a07b7759e7cfc8b7edcdbfb0b3113c77045bdd86abb88e51f9a7653`.

Structural checks passed:

- exactly 22 sources;
- exactly 5 original holdout sources;
- exactly 44 rows;
- two conditions per source;
- 22 rows per condition;
- valid source splits;
- no duplicate row IDs or candidate answers;
- valid candidate token counts;
- exact query-tag/expected-label balance:
  `A_true_answer=11`, `A_override_answer=11`, `B_true_answer=11`,
  `B_override_answer=11`.

The behavior gate failed:

- `balanced_real`: true answer 17/22, below the 20/22 floor;
- `balanced_fictional`: override answer 20/22, passing;
- clean source-level contrasts: 15/22, below the 18/22 floor;
- original holdout clean contrasts: 4/5, passing.

The diagnostic class was `real_query_failed`. V6 is a useful negative repair:
the prompt-family confound was structurally controlled, but real-world retrieval
inside the same two-source prompt family was too weak. MC006 remains
behavior-supported only under the narrow prompt-bounded V4 contract. The next
MC006 repair should recover same-prompt-family real-source answering before any
hidden-state signature or intervention work.

### Decision: MC006 V7 mode-gated behavior repair

Do not start hidden-state discovery from MC006 V7.

V7 implemented a second same-prompt-family repair: the prompt listed only the
fictional city code, then requested either `REAL_WORLD_CAPITAL` or
`FICTIONAL_CITY_CODE`. The true capital was not listed in the prompt. The run is
archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v7_mode_gated_20260630T230257.json`
with SHA256 `eac13cd76b248c58bb9526d4674345b98b42baa59699411af352d8f8fe741a08`.

Structural checks passed:

- exactly 22 sources;
- exactly 5 original holdout sources;
- exactly 44 rows;
- two conditions per source;
- 22 rows per condition;
- 22 rows per requested mode;
- valid source splits;
- no duplicate row IDs or candidate answers;
- valid candidate token counts;
- no true-capital prompt leaks.

The behavior gate failed:

- `mode_real`: true answer 8/22, below the 20/22 floor;
- `mode_real`: override answer 13/22 and lure answer 1/22;
- `mode_fictional`: override answer 22/22, passing;
- clean source-level contrasts: 8/22, below the 18/22 floor;
- original holdout clean contrasts: 1/5, below the 4/5 floor.

The diagnostic class was `real_mode_failed`. V7 is a sharper negative boundary
than V6: when the fictional city is present and the true capital is absent from
the prompt, this candidate-scored Qwen3-1.7B interface strongly prefers the
fictional city even under explicit real-world mode. The next MC006 repair should
change the scoring/rendering interface before any hidden-state signature work.

### Decision: MC006 V8 generated-answer behavior repair

Do not start hidden-state discovery from MC006 V8.

V8 kept the V7 same-prompt-family mode prompt but replaced candidate scoring
with deterministic greedy generation and a strict first-line city parser. The
run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v8_generated_mode_20260630T230955.json`
with SHA256 `f7b95d9a3d3ae841917f3f206eb88803b4f0bbd5088d4af08a4d16a889977ffb`.

Structural checks passed:

- exactly 22 sources;
- exactly 5 original holdout sources;
- exactly 44 rows;
- two conditions per source;
- 22 rows per condition;
- 22 rows per requested mode;
- valid source splits;
- no duplicate row IDs or candidate strings;
- no true-capital prompt leaks.

The behavior gate failed:

- strict parseability: 35/44, below the 40/44 floor;
- `mode_real`: true answer 6/22, below the 20/22 floor;
- `mode_real`: override answer 9/22 and unparsed 7/22;
- `mode_fictional`: override answer 20/22, passing;
- clean source-level contrasts: 6/22, below the 18/22 floor;
- original holdout clean contrasts: 1/5, below the 4/5 floor.

The diagnostic class was `parseability_failed`. The stricter interpretation is
that V8 also failed real-world mode and source contrasts. Several unparsed
real-world rows stated true facts in prose, but the prompt asked for only the
city name, and those rows do not form a reliable compact behavior substrate.
The next MC006 repair should change output-contract enforcement, chat rendering,
or the preregistered parser before any hidden-state signature work.

### Decision: MC006 V9 lenient parser audit

Do not start hidden-state discovery from MC006 V9.

V9 audited the frozen V8 generated outputs with a preregistered lenient parser:
full candidate strings could appear anywhere in the generated suffix, with the
unique earliest candidate mention used as the parsed answer. The run is
archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v9_lenient_parse_audit_20260630T231503.json`
with SHA256 `12ad8bc1d7e10bc4f171c940b41710775fe30b62605448a90670b34671d5def5`.

Structural checks passed:

- source run type was `parametric_fact_override_v8_generated_mode`;
- exactly 44 source records;
- exactly 22 sources;
- exactly 5 original holdout sources;
- two conditions per source;
- 22 rows per condition;
- no duplicate candidate answers;
- V8 reported no true-capital prompt leaks.

The lenient parser improved parseability but did not rescue the substrate:

- parseability improved from 35/44 strict to 41/44 lenient, passing;
- `mode_real` true answers improved from 6/22 to 11/22, still below the 20/22
  floor;
- `mode_fictional` override answers improved from 20/22 to 21/22, passing;
- clean source-level contrasts improved from 6/22 to 10/22, still below the
  18/22 floor;
- original holdout clean contrasts improved from 1/5 to 2/5, still below the
  4/5 floor.

The diagnostic class was `real_mode_failed`. V9 shows V8's strict parser was a
real but secondary blocker. The generated text contains recoverable factual
answers on some rows, but not enough same-prompt-family real-world behavior for
hidden-state discovery.

### Decision: MC006 V10 chat-generated behavior repair

MC006 V10 passes the generated-answer behavior substrate. Hidden-state
signature discovery may proceed only on the V10 table; intervention and
mechanism claims remain blocked until a later signature and causal intervention
gate pass.

V10 changed the rendering contract rather than the parser. It used the Qwen3
chat template with `enable_thinking=False` and a system-level instruction to
reply with exactly one city name. The run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v10_chat_generated_20260630T232318.json`
with SHA256 `7c3054588e876a16338be209666ea7245bead81c1a1fd1040e2ab114e522acb7`.

Structural checks passed:

- tokenizer chat template available;
- chat rendering succeeded;
- exactly 22 sources;
- exactly 5 original holdout sources;
- exactly 44 generated rows;
- two conditions per source;
- 22 rows per condition and requested mode;
- no duplicate candidate answers;
- no true-capital prompt leaks.

The behavior gate passed:

- strict parseability: 44/44;
- `mode_real` true answers: 20/22;
- `mode_fictional` override answers: 22/22;
- clean source-level contrasts: 20/22;
- original holdout clean contrasts: 5/5.

The diagnostic class was `chat_generated_v10_substrate_passed`. The two
non-clean source contrasts were Brazil, where real mode generated the lure
`Rio de Janeiro`, and Croatia, where real mode generated the override `Zadar`.
The next MC006 step should test hidden-state signature discovery on V10 with
requested-mode, output-text/logit margin, prompt length, source-split,
shuffled-label, and prompt-rendering controls.

### Decision: MC006 V11 chat-signature audit

Do not start intervention from the MC006 V11 hidden signature.

V11 restricted the V10 behavior table to clean source-level contrasts and tested
final rendered prompt-token hidden states. The run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v11_chat_signature_20260630T233253.json`
with SHA256 `d61853edd2bea85b2b628e6fcff4a7bbc4ee3ebdc5c66318b64de1bf41edb43c`.

Structural checks passed:

- exactly 20 clean sources;
- exactly 5 original holdout sources;
- exactly 40 rows;
- two conditions per source;
- 20 rows per condition;
- balanced binary labels;
- discovery and holdout each had both labels;
- no duplicate record ids.

The hidden signal was real but not promotable:

- selected hidden candidate: `layer_0` at final rendered prompt token;
- selected hidden holdout AUC: 1.000;
- holdout source-pair ordering: 5/5 passed;
- output-margin holdout AUC: 1.000;
- requested-mode holdout AUC: 1.000;
- prompt-length holdout AUC: 0.880;
- shuffled-label selection p95: 1.000.

The diagnostic class was `requested_mode_confounded`. The stricter
interpretation is broader: the V11 hidden signal is requested-mode confounded,
output-margin confounded, and shuffle-selection confounded. MC006 remains
behavior-supported on V10 but not signature-supported for intervention. The
next repair should make the critical labels vary under a matched requested-mode
surface, or otherwise beat requested-mode, output-margin, and shuffled-label
controls before intervention.

### Decision: MC006 V12 real-after-fiction signature diagnostic

Do not start intervention from the MC006 V12 matched-surface hidden signal.

V12 used the V2 `real_after_fiction` prompt surface, where every row says the
fictional codebook is not real-world geography and asks for the current
real-world capital. The run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v12_real_after_fiction_signature_20260630T234703.json`
with SHA256 `5594836b401c1d8e68f5b04dea7102135dcb029f99357fe3d15cc95c6b5eac56`.

Structural checks passed for the diagnostic table:

- 30 binary `real_after_fiction` rows;
- 10 excluded lure side rows;
- 9 `true_answer` rows;
- 21 `override_answer` rows;
- discovery: 6 true, 12 override;
- calibration: 2 true, 4 override;
- holdout: 1 true, 5 override;
- no duplicate record ids.

The hidden signal was again real but not promotable:

- selected hidden candidate: `layer_7` at final prompt token;
- selected hidden holdout AUC: 1.000;
- candidate-score margin holdout AUC: 1.000;
- next-token output-margin holdout AUC: 1.000;
- prompt-length holdout AUC: 0.800;
- shuffled-label selection p95: 1.000.

The diagnostic class was `holdout_balance_failed`. The stricter interpretation
is broader: V12 is also candidate-score confounded, output-margin confounded,
and shuffle-selection confounded. The matched surface removed the explicit
requested-mode label from V11, but the current row bank is too imbalanced and
too output-visible for intervention. The next MC006 repair should build a
larger matched-surface behavior table with multiple truth-following and
fictional-code-following holdout rows before another signature gate.

### Decision: MC006 V13 real-after-fiction generated behavior repair

Do not start hidden-state discovery from MC006 V13.

V13 repaired the V12 candidate-scoring flaw by generating answers directly
under six matched `real_after_fiction` templates. Each template contained a
fictional city mapping, asked for the current real-world capital, and did not
list the true capital. The run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v13_real_after_fiction_generated_20260630T235712.json`
with SHA256 `89c82723fe9e816757b96d78491f10a26f21a5ab8152bbec91dbd3d7008000a7`.

Structural checks passed:

- exactly 40 sources;
- exactly 6 templates;
- exactly 240 generated rows;
- exactly 40 rows per template;
- exactly 8 holdout rows per template;
- every source appeared once per template;
- no duplicate record ids;
- no duplicate candidate answers;
- no true-capital prompt leaks.

Discovery/calibration selection chose `fake_mapping_warning` with selection key
`[7, 21, -11, -2]`. The selected template had:

- strict parseable rows: 32/40;
- binary rows: 29/40;
- true answers: 20/40;
- override answers: 9/40;
- lure answers: 3/40;
- unparsed rows: 8/40;
- non-holdout true answers: 14;
- non-holdout override answers: 7;
- holdout true answers: 6;
- holdout override answers: 2;
- holdout side rows: 0.

The diagnostic class was `binary_volume_failed`. This is a near miss, not a
pass. V13 fixed holdout label balance and moved labels from candidate scoring
to generated text, but it missed the preregistered 30/40 binary-row floor by
one row. The next MC006 repair should improve strict output-shape compliance or
expand the matched prompt/source bank while preserving V13's holdout balance.

### Decision: MC006 V14 parser-normalized behavior repair

Use MC006 V14 as a behavior substrate for signature discovery only. Do not
claim a mechanism or start intervention from V14.

V14 reused the frozen V13 generated outputs and applied only strict NFKD
diacritic normalization before the same first-line prefix parser. The run is
archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json`
with SHA256 `7596e8c6869cfcb7413e9bbe8d9b6324b2ed3115021bbb1bf7091059cbac54c5`.

The parser delta was narrow:

- changed rows: 3/240;
- all changed rows were V13 `unparsed`;
- no candidate label changed to another candidate label;
- every changed row matched by `strict_first_line_prefix_nfkd`.

The selected template remained `fake_mapping_warning`. The selected table
passed:

- binary rows: 30/40;
- true answers: 21/40;
- override answers: 9/40;
- non-holdout true answers: 15;
- non-holdout override answers: 7;
- holdout true answers: 6;
- holdout override answers: 2;
- holdout side rows: 0.

The diagnostic class was `parser_normalized_generated_substrate_passed`. V14
finally supplies a matched generated-answer behavior table for the MC006
knowledge-like branch, but only as a behavior substrate. The next step must be
a hidden-signature diagnostic with candidate-score, output-margin, prompt/token,
parser-delta, shuffled-label, and side-row controls.

### Decision: MC006 V15 parser-normalized hidden-signature audit

Do not start intervention from the MC006 V15 hidden signature.

V15 tested the V14 selected binary table for a final-prompt-token hidden-state
signature. The run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v15_parser_normalized_signature_20260701T000911.json`
with SHA256 `25a68fadb36b5f60ec909f9e000c89adfb7779ba2ffeefea2b46bdfcb4e0b57c`.

Structural checks passed:

- 30 binary rows;
- 10 side rows;
- 21 true-answer rows;
- 9 override-answer rows;
- holdout: 6 true and 2 override;
- unique source ids;
- no duplicate record ids.

The hidden signal was real but not promotable:

- selected hidden candidate: `layer_16` at final prompt token;
- discovery AUC: 1.000;
- holdout AUC: 1.000;
- candidate-score margin holdout AUC: 1.000;
- next-token output-margin holdout AUC: 1.000;
- prompt-length holdout AUC: 0.583;
- final-token-id holdout AUC: 0.500;
- shuffled-label selection p95: 0.833.

The diagnostic class was `candidate_score_confounded`. The stricter
interpretation is broader: V15 is also output-margin confounded. MC006 is now
behavior-supported on a matched generated table, but still not
signature-supported for intervention. The next MC006 repair should construct a
table where labels are not perfectly visible to candidate scoring or
next-token output margin, or move to earlier/pre-output signature positions
that must beat the same baselines.

### Decision: MC006 V16 pre-output position signature audit

Do not start intervention from the MC006 V16 hidden signature. Treat it as a
positive lead-time diagnostic and a failed mechanism-promotion result.

V16 tested whether MC006's V15 failure was only an artifact of probing at the
final `Answer:` output interface. It reused the V14 selected binary table and
scored hidden states at `after_mapping_line`, `after_instruction_line`,
`after_question_line`, `after_return_line`, and `final_prompt_token`, with only
the first four positions selectable. The run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v16_pre_output_position_signature_20260701T002113.json`
with SHA256 `6c99826028de85d9b03b22cd8c467ccc42483e1b80ca4f1bbcab6fc67074eb31`.

Structural checks remained the same as V15:

- 30 binary rows;
- 10 side rows;
- 21 true-answer rows;
- 9 override-answer rows;
- holdout: 6 true and 2 override;
- unique source ids;
- no duplicate record ids.

The selected pre-output hidden candidate was `after_mapping_line/layer_4`:

- discovery AUC: 1.000;
- holdout AUC: 1.000;
- same-position next-token output-margin holdout AUC: 0.917;
- selected-position prefix-token-count holdout AUC: 0.583;
- selected-position token-id holdout AUC: 0.333;
- shuffled-label selection p95: 0.917.

The lead-time diagnostic criteria passed. The mechanism criteria failed:

- candidate-score margin holdout AUC: 1.000;
- final next-token output-margin holdout AUC: 1.000.

The diagnostic class was
`leadtime_signal_supported_but_output_global_confounded`. V16 shows that MC006
has an earlier internal trace of the true-versus-override outcome before the
final answer interface, but the completed prompt's output interface still
exposes the labels perfectly. The next MC006 step should either build a table
where labels are not final-output-visible or explicitly preregister a
monitoring-only/known-confounded causal stress test.

### Decision: MC006 V17 pre-output steering stress

Do not treat the MC006 V16 early direction as a simple additive control vector.

V17 ran the monitoring-only/known-confounded causal stress test proposed after
V16. It used the V14 behavior artifact and V16 lead-time signature artifact,
recomputed the `after_mapping_line/layer_4` direction on non-holdout primary
rows, and applied additive residual steering at standardized-signature score
doses 1.0, 2.0, and 4.0. The run is archived at
`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v17_pre_output_steering_stress_20260701T003620.json`
with SHA256 `d6033ff8a685292341818a95b33c1c19ae3f799afdb013fee8ade43a910e2437`.

Source and reproduction checks passed:

- V14 source hash matched V16;
- V16 selected `after_mapping_line/layer_4`;
- structural checks matched V15/V16;
- current-run baseline reproduced 8/8 frozen V14 holdout labels.

The causal stress gate failed on holdout primary rows:

- dose 1.0: plus margin delta +0.0469, minus margin delta +0.0625,
  selected/control gap +0.0078, generated-label changes 0;
- dose 2.0: plus +0.0625, minus -0.0156, selected/control gap -0.0234,
  generated-label changes 0;
- dose 4.0: plus +0.1250, minus -0.1406, selected/control gap -0.0547,
  generated-label changes 0.

The strongest directional dose, 4.0, still missed the preregistered +0.25 and
-0.25 margin floors, and a control arm had larger absolute effect. Side rows
did not show broad parse corruption: max selected-arm side-label changes were
1/10 against the 3/10 ceiling.

The diagnostic class was `intervention_failed`. MC006 is now
behavior-supported and lead-time-signature-supported, but additive residual
steering from that early signature failed. The next MC006 causal attempt should
not repeat this intervention family without a materially different
preregistration, such as non-additive activation patching, later-path transport,
or a less final-output-visible behavior table.

## Open But Non-Blocking Questions

### How exact should no-hint accuracy be before discovery?

Current answer:

High enough that wrong-hint errors are interpretable. The preregistration should set an exact value before implementation. A reasonable starting point is 80 percent no-hint accuracy on included task families.

Why non-blocking:

This can be fixed when constructing the manifest.

### Should MC-001 include chain-of-thought?

Current answer:

No. First card should force compact final answers. Chain-of-thought introduces faithfulness and policy complications.

Why non-blocking:

Explanations can become a later card.

### Should the first intervention be direction-based or feature-based?

Current answer:

Try directions first because they fail cheaply, but require SAE/transcoder comparison.

Why non-blocking:

Method order is specified in the method matrix.

### Should hidden signatures be measured before or after the hint?

Current answer:

Both. The most valuable early-warning claim requires signatures before the final answer. The mechanism claim may need hint-token and answer-prefix positions.

Why non-blocking:

Layer/position sweeps belong to future implementation.

### What if Qwen3-0.6B has no hidden value beyond output logits?

Current answer:

Write and maintain the diagnostic/control-surface card rather than a mechanism card. For MC-001, Qwen3-0.6B dense direction, dense prefill, cross-layer transport, dense answer-prefix steering, single-head/single-layer attention-source localization, cumulative generation-query source masking, broad input-mask localization, and layout/tokenization parity have failed the mechanism standard. V12 shows the large V9 input-mask result is mostly source removal or prompt-state recomputation. V13 closes the last 0.6B layout caveat. Future mechanism-card work should escalate to Qwen3-1.7B/Gemma or another artifact-rich stack.

Why non-blocking:

The baseline prevalence gate handles this.

### What if prompt-only instruction wins?

Current answer:

Then the practical control-surface claim fails. The hidden signature may remain diagnostic only.

Why non-blocking:

That is an intended falsification path.

### How should biology enter?

Current answer:

Only after a narrow LLM mechanism survives and makes an abstract non-LLM prediction.

Why non-blocking:

Biology is not part of MC-001.

## Questions That Would Change The Program

1. Do hidden signatures ever beat output/trace monitoring by meaningful lead time?
2. Are controllable signatures usually sparse features, dense directions, or path structures?
3. Do successful interventions survive paraphrase and task-family shift?
4. Are behavior mechanisms stable across instruction tuning?
5. Can a mechanism be preserved through distillation?
6. Can side-effect cost be predicted before intervention?
7. Can a reliability atlas become a deployable safety threshold?

## Future Decision Points

After MC-001 smoke:

- completed: continue sycophancy card on Qwen3-0.6B factual-ladder prompts.

After MC-001 discovery:

- completed: run output-margin-conditioned controlled v2;
- completed: run a narrower v3 pass with lower h14 doses and answer-letter/output-margin residualized directions;
- completed: write the Qwen3-0.6B diagnostic/control-surface card;
- completed: run method-shift v4 final-token patching inside the agreement-favored output-margin bin;
- completed: run v5 prompt-prefill audit to separate additive steering from decode-step artifacts;
- completed: run v6 additive prefill control audit with matched random/nearby nulls and no-hint/correct-hint side-effect rows;
- completed: run v7 cross-layer option-logit transport audit to explain h13/h14 interchangeability;
- completed: update the diagnostic/control-surface card with v7 and close residual-stream dense transport for mechanism-card purposes;
- completed: run v8 answer-prefix dense-signature audit and close dense answer-prefix steering;
- completed: run v9 attention-source ablation and identify strong hint-source dependence;
- completed: run v10 single-head/single-layer attention-source localization and find only weak late localization;
- completed: run v11 cumulative layer-band source masks and find only weak broad generation-query source dependence;
- completed: run v12 input-mask semantics audit and identify V9 as mostly source removal or prompt-state recomputation rather than direct generation-query attention;
- completed: run v13 length/tokenization parity sanity check and close the last broad 0.6B caveat;
- current: close the Qwen3-0.6B MC-001 route as a diagnostic/control-only artifact and choose the next mechanism-card stack.

After MC-001 holdout:

- supported mechanism card;
- failed-controls card;
- diagnostic-only card;
- artifact card;
- next target selection.
