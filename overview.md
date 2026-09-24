# Moonshot LLM Genome Overview

Date: 2026-07-02

This document is the consolidated working overview of the current
`moonshot-llm-genome` rebuild. It summarizes what we are trying to accomplish,
what experiments have been run, what the results mean, what we have learned,
what remains unproven, and how close the project is to its first fully reliable
mechanism-grade control surface.

The short version:

> We can find real, useful control surfaces in small language models. We have
> repeatedly moved model behavior in predicted directions. We have also learned
> that most attractive hidden-state stories collapse when output/logit,
> prompt-format, shuffled-label, locality, holdout, and null controls are made
> strict. The strongest current positive internal result is a narrow
> synthetic-associative-lookup control surface in Qwen3-1.7B, localized to an
> all-head layers-24-26 aggregate attention-write block with a mapped
> answer-absent null boundary. The more knowledge-like capital-fact line now
> has a matched generated-answer behavior substrate after V14. Its signature
> audits remain blocked for intervention: V11 was
> requested-mode/output/null-confounded, V12 was
> weak-holdout/candidate-score/output/null-confounded, V15 was
> candidate-score/output-confounded, and V16 found an earlier lead-time hidden
> signal that still failed mechanism promotion because final candidate-score
> and final next-token output margins were perfect. V17 then tested simple
> additive steering on that early signal and failed: holdout margins moved only
> weakly, generated labels did not change directionally, and controls matched
> or exceeded the small effects. V18 then reproduced the V16 early signal but
> showed that the V14 holdout table has no true/override overlap in
> candidate-score or final-output margins, so margin-matched promotion is
> blocked by row geometry. V19 expanded the generated prompt bank and selected
> a better balanced `separate_task_weak` table, but strict candidate/final
> margin overlap still failed. V20 then showed strict final-output overlap is
> absent even in the pooled V19 binary bank, leaving only bounded pair-matched
> diagnostics or new row generation on that route. V21 ran the bounded
> pair-matched diagnostic and killed that rescue route: the selected pre-output
> hidden direction overfit non-holdout pairs and reversed on holdout, while
> candidate-score and final-output margin baselines still perfectly ordered the
> holdout pairs. V22 then mapped source/path and line-boundary lead-time
> positions inside that same row bank. The curve found real monitor structure,
> especially at the queried country token and later pre-output line boundaries,
> but final candidate-score and final next-token output margins still perfectly
> ordered the holdout pairs. V23 then diagnosed the strict final-margin target
> itself: under the current greedy generated-answer interface, final
> true-minus-override next-token margin sign predicts the binary label. So
> strict final-margin overlap is the wrong default gate for that interface.
> V24 changed the answer interface with delayed JSON city generation and broke
> that first-token sign barrier, but JSON-completion candidate scoring still
> beat the hidden monitor on holdout. V25 then found a different delayed-city
> template where JSON-completion candidate scoring was not perfect, but the
> hidden result failed a split-preserving shuffled-label selection null. V26
> locked that V25 coordinate and tested transfer to the only other
> candidate-decoupled delayed-city template; it failed. V27 expanded the
> delayed-city bank to 640 rows with predeclared source/transfer template roles;
> it found enough source-side candidate-decoupled templates but insufficient
> transfer-role coverage. V28 targeted transfer-role repair directly with
> another 640-row transfer-only grid. It found one clean transfer template, but
> not the two required for a source/transfer behavior bank. The delayed-city
> MC006 route is now closed as monitor-only under the tested prompt families.

After MC016, the next bridge pressure moved from numeric answer selection to
source selection itself. MC017 showed that `LOCAL`/`ATOMIC` source-token answers
can collapse to `LOCAL` even on atomic-only controls. MC018 then removed those
words and counterbalanced neutral `A`/`B` labels, source-definition order, and
answer-option order. The expanded 10-source smoke remained structurally clean
but still failed behavior: primary conflict rows were fully parseable and
balanced by expected source/choice, yet source-rule correctness was only
84/160, first-listed choices won 122/160, and local-source selections won
102/160. The new insight is that answer-interface control is itself a surface:
presentation order and local-source salience can dominate a stated
source-selection rule before hidden-state mechanism work is even justified.

MC019 then removed the source-label answer interface entirely. It kept integer
answers but attached neutral route codes to local table rows and counterbalanced
the route-code rule order. The 10-source structural gate passed at 420 rows.
The selected generated smoke kept direct controls and nulls clean, including
20/20 route-rule-absent UNKNOWN rows and 20/20 answer-absent UNKNOWN rows, but
the route-code conflict still failed: 40/40 primary rows parsed, expected labels
and route codes were balanced, yet expected correctness was only 21/40. A
query-row-repeat repair improved conflict correctness to 27/40 and expected
local rows to 19/20, but expected-atomic rows stayed 8/20 and the null slipped
to 17/20. The new insight is that neutral row-local operational codes can clean
up controls and nulls without making learned atomic answers reliably beat
prompt-local table pressure.

MC020 then isolated whether MC019's expected-atomic failure was just local-row
interference. It was not. In the 10-source smoke, atomic recall remained strong
with no table, with distractor local rows, with the queried local row present,
and with the queried row repeated: 20/20, 18/20, 18/20, and 18/20 atomic,
respectively. The asymmetric failure appeared only when the same prompt had to
apply a conditional route rule. Route-local rows reached 17/20 local, but
route-atomic rows collapsed to 15/20 local and only 2/20 atomic. The new
insight is that the bridge failure is conditional source arbitration, not
ordinary atomic recall and not mere local-number interference.

MC021 then checked whether conditional routing itself works when both branches
are prompt-visible. It does not work reliably enough. The selected 10-source
smoke passed direct controls and nulls: local 20/20, visible reference 20/20,
atomic 18/20, and answer-absent UNKNOWN 18/20. But visible-visible route
conflicts reached only 29/40 expected-correct, while visible-learned route
conflicts fell to 20/40. The new insight is two-layered: conditional routing is
unstable even over prompt-visible numeric branches, and learned-memory branches
amplify that instability toward prompt-local outputs.

MC022 then replaced opaque `P`/`Q` route codes with semantic answer-source
labels: `LOCAL`, `REFERENCE`, and `ATOMIC`. The simple "route codes were too
arbitrary" rescue did not pass. The selected 10-source smoke again passed
controls and nulls: local 20/20, visible reference 20/20, atomic 18/20, and
answer-absent UNKNOWN 19/20. But visible conflicts were only 30/40
expected-correct, learned conflicts were only 24/40, and ATOMIC-source rows
selected learned atomic numbers only 5/20. The deeper insight is that branch
arbitration is sensitive to local-source salience and rule-definition order:
putting the nonlocal branch first rescued visible-visible rows, but it only
partly rescued learned-memory rows.

MC023 then removed row-level source labels and tested query-level operation
handles. This repaired the easy parts in the 10-source smoke: synthetic local
lookup, familiar local lookup, direct atomic control, operation-rule-absent
UNKNOWN, and answer-absent UNKNOWN were all 40/40 on the selected template,
with candidate/output margins reported and no status lexemes. But the conflict
branches still missed the behavior gate: operation-local rows selected local
31/40, and operation-atomic rows selected atomic 28/40. The new insight is
sharper than another prompt failure: query-level operations can clean direct
controls and nulls, but the local-versus-learned conflict still does not cross
the gate without visible source-status text.

MC024 then tested the direct rescue: keep query-level operation handles but add
balanced worked examples for the local and atomic operations. The 10-source
smoke looked promising: the compact worked-example template passed every
control and crossed both conflict thresholds. The full 40-source run killed the
promotion. Controls and nulls remained perfect, and the local branch improved
to 159/160 local, but the learned atomic branch fell to 130/160 atomic with
27/160 other-number outputs. The new insight is not "examples solve the
bridge"; it is that examples preferentially repair prompt-local routing while
learned-memory routing remains the fragile edge under full source-disjoint
evaluation.

MC025 then tested whether MC024's atomic-branch failure was mainly an open
integer answer-interface problem. It constrained answers to A/B/C choices that
included the local number, the atomic number, and UNKNOWN. That did not repair
the bridge. The 10-source smoke preserved local lookup and local operation
routing, but direct atomic control fell to 44/120, answer-absent UNKNOWN fell
to 83/120, and operation-atomic conflict selected the atomic choice only
18/120. The new insight is that prompt-visible candidate choices are not a
neutral output fix; they create their own atomic-control and null failures
while leaving learned-branch routing weak.

MC026 then asked whether the A/B/C labels themselves were the problem. It kept
the same local-number, atomic-number, and UNKNOWN candidate set but exposed the
answers as numeric options instead of letter choices. That repaired the nulls
but killed direct learned recall: synthetic and familiar option lookup were
120/120 local, answer-absent and operation-rule-absent nulls were 120/120
UNKNOWN, operation-local conflict was 111/120 local, but direct atomic control
was 0/120 atomic with 116/120 UNKNOWN, and operation-atomic conflict selected
atomic only 19/120. The new insight is sharper than "choices failed": visible
candidate lists are active behavior surfaces. Numeric options can turn learned
recall into abstention while preserving prompt-local routing and null behavior.

MC027 then stopped treating answer format as a nuisance variable and swept it
directly. On the same 10-source operation substrate, bare integer answers were
the only smoke survivor: familiar lookup was 40/40 local, direct atomic control
was 40/40 atomic, answer-absent and rule-absent nulls were 40/40 UNKNOWN,
operation-local conflict was 39/40 local, and operation-atomic conflict was
38/40 atomic. Every structured or option interface broke a different gate:
`ANSWER=` prefix failed parseability and learned routing, JSON collapsed nulls
and emitted wrong numbers, A/B/C choices kept weak atomic routing, and numeric
options reproduced atomic abstention. The new law is explicit: answer schemas
are behavior surfaces, not passive wrappers around one internal computation.

MC028 then tested whether that bare-integer survivor was a real bridge repair
or only a reduced-smoke boundary. The full-source bare-integer run passed its
structural gate with 960 rows across all 40 sources and kept the easy behavior
surfaces clean: familiar lookup was 160/160 local, direct atomic control was
160/160 atomic, answer-absent and rule-absent nulls were 160/160 UNKNOWN, and
operation-local conflict was 159/160 local. The failure was narrower and more
useful: operation-atomic conflict was only 131/160 atomic, with 28/160
other-number answers and 1/160 local answer. That closes the MC027 smoke
survivor before hidden-state work. Bare integer helped, but full-source learned
atomic routing still leaked other learned numbers under prompt-local table
pressure.

MC029 then factorized that leak instead of immediately trying hidden-state
work. It tested baseline numeric examples, label examples without numbers,
rules-only prompting, query-before-example ordering, and query-row-last
salience across 4,000 rows and all 40 sources. No variant passed the bridge
gate. The result is still useful because the failure axes moved separately:
`rules_only` improved the learned atomic branch to 0.875 atomic with only
0.100 other-number answers, but answer-absent UNKNOWN fell to 0.806.
`query_before_examples` preserved nulls but collapsed learned routing to 0.256
atomic and amplified worked-example copying to 54 rows. `query_row_last`
reduced other-number leakage to 0.081 but shifted the operation-atomic branch
toward local-row answers. The new lesson is that prompt factorization can move
the error surface without repairing the behavior substrate.

MC030 then tested the narrowest plausible repair of the MC029 result: keep the
rules-only branch gain, but add explicit absence guards to recover
answer-absent null reliability. The full-source structural gate passed with
4,000 rows across all 40 sources after the structural checker caught and
removed a real confound: numeric list markers in a decision-order prompt leak
target atomic numbers for early elements. The behavior result closed the simple
guard-repair path. The unguarded `rules_only_baseline` stayed best at 0.875
operation-atomic atomic and 0.806 answer-absent UNKNOWN. A direct row-absence
guard kept operation-atomic atomic at 0.863 but drove answer-absent UNKNOWN
down to 0.231. A decision-order guard after the query kept operation-atomic
atomic at 0.869 but answer-absent UNKNOWN was only 0.425. The guarded
query-last variant reduced other-number leakage to 0.056, but collapsed the
learned atomic branch to 0.487. The useful conclusion is now sharper: simple
absence guards do not repair the branch/null tradeoff; they move or worsen it.

MC031 then tested a materially different statusless reliability cue instead of
another operation-leak patch. It used arithmetic checksum validity to decide
whether a local source should be trusted, while hiding real and lure atomic
numbers from conflict prompts and removing trusted/untrusted/reliable status
lexemes. The full structural gate passed on 840 rows across 40 sources. The
10-source model smoke exposed a clean new failure mode: direct synthetic lookup,
familiar lookup, direct atomic control, answer-absent nulls, and valid-checksum
local rows passed, but invalid-checksum rows still selected local numbers on
every selected conflict and selected atomic/lure numbers 0/10 times. The useful
conclusion is that a statusless reliability cue can preserve controls and nulls
while failing source-validity arbitration itself.

MC032 then removed the checksum-specific objection. It used two neutral local
tables instead of arithmetic validity: if the query row matched across the
tables, the local number should control; if it mismatched, the learned atomic
number should control. The structural gate passed, and the 10-source model
smoke reported candidate/output baselines. Direct synthetic lookup, familiar
lookup, direct atomic recall, and answer-absent nulls were clean. The mismatch
branch still selected atomic/lure numbers 0/10 times; it selected the primary
local number 7/10 times, UNKNOWN 1/10, and other numbers 2/10, with side-number
copying at 0/10. That is a sharper insight than "checksum failed": a second
statusless source-validity cue also fails by local-table dominance, not by
basic atomic-recall failure, null failure, or side-table copying.

MC033 then tested the repair that would have made the MC031-MC032 story too
easy to dismiss: replace checksum and cross-table consistency with a row-local
standard-number claim checked against learned atomic memory. This made the cue
semantically fact-like rather than table-like. The structural gate passed on
840 rows across 40 sources, and the 10-source smoke kept direct synthetic
lookup, familiar lookup, direct atomic recall, answer-absent nulls, and
fact-claim-absent local rows clean. But the branch rule still failed. On match
rows, where the claim agreed with the learned atomic number and the expected
answer was local, the model returned local only 4/10 times and learned atomic
5/10 times. On mismatch rows, where the expected answer was learned atomic, it
returned atomic only 1/10 times, local 5/10 times, and the wrong claimed/lure
number 4/10 times. That closes the same-family bridge repair path after MC032:
the route is not merely "local dominance." It is branch instability under
statusless source-validity pressure, even when direct controls and nulls look
perfect.

We have not yet completed the full ambition:

> behavior -> internal signature -> intervention -> null baselines -> failure
> modes -> practical implication

We have made serious progress. We have mapped several behavior families,
isolated many false mechanism stories, and found one narrow internal causal
surface that is close enough to be worth continued refinement. But the project
does not yet have a clean, deployable, broad "knowledge genome" map for small
LLMs.

## The Ambition

The useful ambition is narrow and hard:

1. Map an internal signature.
2. Intervene on it.
3. Prove whether the intervention works, fails, or only looked real because the
   controls were weak.

The project is not trying to write another vague interpretability report. It is
trying to build mechanism cards where every claim survives a hostile audit.

The three gates are:

1. **Signature:** there is a measurable internal pattern tied to a concrete
   behavior.
2. **Intervention:** steering, editing, training, masking, replacement, or
   surgery changes behavior in the predicted direction.
3. **Reliability:** nulls, holdouts, locality, fluency, robustness, side
   effects, and failure modes are documented.

The "knowledge genome" phrase means the project is trying to discover the
practical internal grammar of small LLM behavior. We want to know which
activation directions, attention paths, residual features, token positions,
prompt surfaces, and source-token dependencies correspond to robust behaviors,
and which are only artifacts of easy baselines.

This is intentionally empirical. A result counts only when there is an
artifact, a preregistered or otherwise explicit gate, a result file, and a
status/verdict document that preserves the failure modes.

## The Current Answer

The current answer to "Can we find reliable control surfaces inside learned
language systems?" is:

Yes, for scoped behavior control.

Yes, for one narrow internal source-value control surface under a synthetic
associative lookup contract.

No, not yet for a broad, deployable, general mechanism of truth, honesty,
knowledge, or factual recall.

The most important distinction in the repo is:

- **Behavior control surface:** an intervention reliably changes model behavior
  under useful controls.
- **Mechanism control surface:** a measured internal signature predicts the
  behavior, the intervention targets that signature, and reliability checks show
  the effect is local, robust, and not matched by easier baselines.

Most experiments have found behavior control, not mechanism control. The
project has become valuable because it keeps separating those two instead of
blurring them.

## What We Are Doing

We are running mechanism-card attempts as numbered experiment families:

- `MC001`: truth versus user-agreement / wrong-hint behavior.
- `MC002`: known-answer versus unsupported nonce-entity hallucination.
- `MC002B`: support-in-reference versus unsupported near-neighbor entities.
- `MC003`: delayed-copy constrained generation.
- `MC004`: nonce in-context entity/code binding.
- `MC005`: synthetic associative lookup and source-value attention paths.
- `MC006`: real-world capital facts versus task-local false or fictional
  overwrites.
- `MC007`: semi-synthetic familiar-entity lookup as a bridge between prompt
  lookup and factual override.

The workflow is:

1. Build a behavior substrate.
2. Check that the behavior is reliable enough to study.
3. Search for hidden signatures.
4. Compare hidden signatures against output/logit and prompt baselines.
5. Intervene on the selected internal surface.
6. Stress-test locality and nulls.
7. Record whether the result works, fails, or was confounded.

The project has repeatedly learned that steps 2, 4, and 6 are where most
plausible results die.

## What Counts As Progress

Progress is not only a positive mechanism card.

Progress includes:

- finding a behavior substrate that survives holdout controls;
- showing a hidden signal exists but is explainable by output margin;
- showing a steering intervention moves behavior but lacks locality;
- showing a source mask works but is broad source removal rather than a circuit;
- showing a positive internal intervention works on lookup rows but causes rare
  null flips;
- mapping exactly which controls fail instead of pretending they passed.

By that standard, the project has made substantial progress. It has produced a
map of what does and does not work across several small-model behavior
families.

## The Main Lesson So Far

The main lesson is that small LLMs have many handles, but very few of those
handles are clean mechanism handles.

We can steer behavior.

We can mask source tokens and change answers.

We can find hidden directions with high AUC.

We can make donor replacements and attention-write replacements that reproduce
effects.

But when the controls are made strict, most results turn into one of these:

- output-margin measurement in disguise;
- prompt-format classification in disguise;
- requested-mode classification in disguise;
- source-token deletion or prompt-state recomputation;
- broad late-layer aggregate behavior rather than a compact circuit;
- sample-fragile null behavior;
- low-margin row flips under interventions that otherwise look clean;
- shuffled-label overfitting on small tables.

This is not a reason to stop. It is the genome map beginning to appear. The
shape of the failures tells us where the model is actually using prompt text,
source tokens, output margins, and late aggregate computation.

## How Close Are We?

We are not close to a broad knowledge genome.

We are close to a narrow, honest, first internal control-surface map in one
synthetic task family.

The current bounded mechanism card is MC005:

- behavior substrate: passed;
- internal signature: passed for source-value lookup and later row structure;
- intervention: passed for late-band source-value masking and exact
  attention-write mediation;
- reliability: strong but not complete, because answer-absent nulls still show
  rare row-level flips under write replacement;
- scope: narrow synthetic associative lookup on Qwen3-1.7B, not general
  knowledge.
- verdict: frozen bounded mechanism card, not a full promoted mechanism card;
- executable closeout: `code/mc005_write_replacement_closeout_audit.py`
  validates V27-V31 and returns `bounded_frozen_not_promoted`.

The current closest knowledge-like result is MC006:

- behavior substrate: finally passed in V10 under chat-rendered generated
  answers;
- hidden signature: failed in V11 because the perfect hidden separation was
  matched by output margin, explicit requested mode, and shuffled-label
  selection;
- matched generated surface: V13 fixed holdout balance but missed binary
  volume at 29/40 against a 30/40 floor;
- parser-normalized matched generated surface: passed in V14 with 30/40 binary
  rows and holdout 6 true / 2 override;
- matched generated signature: failed in V15 because candidate-score and
  next-token output margins matched the perfect hidden holdout AUC;
- pre-output matched generated signature: V16 found an `after_mapping_line`
  hidden signal that beat same-position output margin and null controls, but
  final candidate-score and final next-token output margins still matched it;
- pre-output additive steering: V17 failed to turn the V16 signal into
  holdout causal control;
- same-table margin-matched audit: V18 reproduced the V16 signal but found no
  true/override holdout overlap in candidate-score or final-output margins;
- overlapping-margin table search: V19 improved label balance and near-pair
  counts, but strict margin overlap still failed;
- strict-overlap selection audit: V20 showed the existing V19 bank cannot
  rescue strict final-margin overlap by subset selection;
- approximate pair-matched lead-time audit: V21 found 451 non-holdout and 28
  holdout approximate joint pairs, but candidate-score and final-output margins
  still achieved 1.000 holdout pair accuracy while the selected pre-output
  hidden direction fell to 0.286 holdout pair accuracy;
- source/path lead-time curve: V22 mapped eight token/line positions and found
  source/path and later pre-output monitor structure, but final candidate-score
  and final-output margins still achieved 1.000 holdout pair accuracy;
- final-margin sign-barrier audit: V23 showed final next-token margin sign
  predicted all V18/V19 binary generated labels, while candidate-score margin
  had overlap on V19 all-binary rows;
- delayed-city interface audit: V24 forced the city to appear after a JSON
  wrapper, selected `game_code_then_geo` with 31 binary rows and source-disjoint
  holdout coverage, broke the final city-token sign barrier, but failed
  promotion because JSON-completion candidate-score holdout AUC was 1.000 while
  the selected hidden monitor holdout AUC was 0.875;
- candidate-decoupled delayed-city audit: V25 selected `untrusted_note_real`
  with 37 binary rows and source-disjoint holdout coverage. JSON-completion
  candidate-score holdout AUC fell to 0.333 and the selected hidden monitor
  reached 1.000 holdout AUC, but split-preserving shuffled-label selection p95
  also reached 1.000, blocking signature promotion;
- locked-coordinate transfer audit: V26 froze the V25
  `after_mapping_line/layer_10` coordinate. It reproduced weakly on
  `untrusted_note_real` source holdout at 0.750 AUC, but fell to 0.500 holdout
  AUC on `separate_task_weak`, while a position-local output control reached
  0.833;
- expanded candidate-decoupled bank audit: V27 generated 640 delayed-city rows
  across 16 predeclared source/transfer templates. Four templates were
  candidate-decoupled and pooled holdout balance passed, but only one
  transfer-role template passed, so the bank is not hidden-state-ready;
- transfer-role repair audit: V28 generated 640 transfer-only delayed-city rows
  across 16 predeclared templates. One template,
  `transfer_sandbox_mapping_then_geo`, passed with 35 binary rows, 4 true / 4
  override holdout labels, and JSON candidate-score holdout AUC 0.1875, but
  the gate required at least two transfer-ready templates;
- formal delayed-city closeout: the V14-V28 chain is now closed as a
  monitor-only diagnostic branch, not a mechanism-promotion route;
- executable predecision-frontier closeout:
  `code/mc006_predecision_frontier_closeout_audit.py` validates V14-V28 and
  returns `monitor_only_closed`;
- intervention: not allowed yet.

So the honest answer is:

> We have one narrow internal causal surface under synthetic lookup, and one
> behavior-supported but signature/intervention-blocked capital-fact branch.
> We have not yet found a mechanism-grade internal control surface for learned
> factual knowledge.

## MC001: Truth Versus Agreement On Qwen3-0.6B

MC001 asked whether a small model has an internal control surface for choosing
truth over a misleading user hint.

The initial setup used factual multiple-choice prompts with wrong hints. The
model often followed the user's wrong hint instead of the true answer. This
gave a behavior surface: can we push the model toward truth?

The answer at the behavior level was yes.

Dense residual steering worked as behavior control. In one validation pass,
baseline truth-following was `135/288`, while raw h14 steering at `alpha=0.50`
raised it to `190/288`. On the hard agreement-favored wrong-hint bin, baseline
truth-following was `0/104`, and raw h14 steering moved it to `30/104`.

Prompt-prefill steering also worked. On the same hard bin:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0/104 | 101/104 | 3/104 | 104/104 |
| raw h14 prefill `alpha=0.50` | 21/104 | 72/104 | 11/104 | 104/104 |
| wrong-token prefill | 0/104 | 99/104 | 5/104 | 104/104 |
| matched random prefill | 3/104 | 91/104 | 10/104 | 104/104 |

Hint-source interventions worked even more strongly. In V9, on the hard bin:

| Arm | Truth | Agreement | Other | Parseable |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0/104 | 101/104 | 3/104 | 104/104 |
| mask hint line | 63/104 | 16/104 | 25/104 | 104/104 |
| mask hinted answer | 50/104 | 23/104 | 31/104 | 104/104 |
| matched answer-instruction mask | 4/104 | 97/104 | 3/104 | 104/104 |
| matched random mask | 5/104 | 84/104 | 14/104 | 103/104 |

That is a real source-specific control surface.

But the mechanism story failed.

The dense h14 direction did not beat the output/logit baseline. Residualization
removed the same-layer effect. Nearby-layer application remained active. A
transport audit showed h13 and h14 had nearly identical option-logit transport.
A separately trained answer-prefix direction did not rescue the dense
mechanism route.

The source-token story also failed as a localized attention mechanism. Later
audits did not find a single head, single layer, or cumulative generation-query
attention path that recovered the V9 effect. V12 decomposed the source effect:
full-line input masking reached `63/103` hard-bin truth for the hint line and
`49/103` for the hinted answer, while literal prompt rewrites nearly matched it
at `57-60/103` for the hint line and `48-51/103` for the answer. Query-only
masks stayed much weaker at `24/103` and `22/103`. V13 closed the
layout/tokenization caveat: character-matched and rendered-token-count-matched
neutral rewrites reached only `52/103`, not a compact mechanism-like result.

MC001 therefore teaches:

- small models can be steered toward truth in a scoped wrong-hint task;
- source-token dependence can be strong and source-specific;
- dense residual directions can move behavior;
- but the apparent hidden mechanism collapses into output margins, prompt-state
  recomputation, broad source removal, and transport effects.

Current MC001 label:

> failed-mechanism / diagnostic-control artifact

It is useful, but it is not a mechanism card.

## MC001B: Qwen3-1.7B Escalation

The Qwen3-1.7B branch asked whether a larger small model would give a cleaner
truth-versus-agreement mechanism.

It reproduced practical dense control. Raw `h21 alpha=0.50` improved
wrong-hint truth-following from `113/180` to `154/180`.

But the mechanism rescue failed. Residualized `h21 alpha=0.50` dropped
truth-following to `103/180`, and the margin-only AUC stayed at `1.000`.

Lesson:

> Scaling from Qwen3-0.6B to Qwen3-1.7B improved the practical steering
> surface, but not the mechanism claim. Output margin remained too strong a
> confound.

## MC001G: Gemma 2 2B Truth/Agreement Stack

The Gemma branch explored whether a more artifact-rich stack, including Gemma
Scope sparse features, could produce a better mechanism.

The first base-Gemma behavior gate failed. A repaired raw-logit behavior gate
then passed with 42 clean items, at least 8 clean items per answer letter, and
a clear pressure slope from intermediate wrong hints to direct/high wrong
hints.

Dense signature discovery found real hidden-state signal, but the same-row
margin was tautologically perfect under forced-choice labels.

A pre-hint margin matched discovery pass selected 24 truth and 24 agreement
rows matched on no-hint margin bins. The best simple direction reached `0.755`
holdout AUC.

The first matched dense intervention failed:

- positive layer-14 steering changed zero matched-holdout labels;
- sign-flip and wrong-layer controls each improved one row.

A stronger activation-replacement gate also failed controls:

- layer-14 truth-donor replacement moved `2/7` held-out agreement rows to truth;
- the agreement-donor control matched that effect;
- both caused `2/7` no-hint locality losses.

Canonical Gemma Scope sparse discovery failed promotion:

- the best layer-14 rank-1 sparse feature reached only `0.714` holdout AUC;
- it was below the dense direction;
- shuffle nulls found stronger holdout features.

Several repair passes improved behavior-table quality but found new confounds:

- expanded repair bank: 76 clean items and 32 matched holdout rows, but only 74
  matched rows total and missing discovery coverage for holdout bin 3;
- targeted repairs: 120 and 128 matched rows, but still missing exact discovery
  support for holdout bin 3;
- answer-letter confound: A-correct rows supplied 45/64 matched agreement rows,
  while C/D-correct rows supplied only 8/64;
- option-position counterbalancing improved controls;
- format-control V2 reached 174 strict source-disjoint matched rows, but failed
  the D-cell floor and exact strict holdout bin-letter coverage;
- pairwise answer-text scoring removed literal A/B/C/D outputs and produced 256
  strict source-disjoint/order-matched rows, but failed pair-order balance;
- raw generated-answer text removed displayed choices and produced 180 clean
  items plus 132 strict matched rows, but failed weak-hint truth, single-word
  clean coverage, answer-shape dominance, and strict holdout-key coverage;
- generated-text V2 repaired clean coverage to 252 items and strict matching to
  172 rows, but still failed primary weak-hint truth, shape-matched volume, and
  strict holdout-key coverage.

Current MC001G label:

> repaired-substrate / matched-signature / failed-intervention /
> sparse-promotion-failed / format-control-failed / pairwise-interface-failed /
> generated-interface-failed

Lesson:

> Better tooling and sparse features did not automatically yield a mechanism.
> The harder we controlled answer format, option position, pair order, and
> generated-answer shape, the more the original mechanism story dissolved.

## MC002: Known Answer Versus Unsupported Nonce Hallucination

MC002 tried a different behavior family: known-answer factual questions versus
unsupported nonce-country hallucinations.

The first Gemma 2 2B raw-generation gate failed before hidden-state work:

- real-country clean sources passed: `27 >= 24`;
- nonce pressure hallucination was present: `24 >= 12`;
- but only `1/40` nonce sources abstained cleanly under both neutral and
  cautious prompts;
- same-source nonce contrasts reached only `7 < 12`;
- real-country lure locality failed with only `10/40` lure rows still correct.

The first repair changed render mode from raw to chat-style prompts on the same
base Gemma model and same 80 sources. It failed harder:

- clean real-country sources dropped from 27 to 6;
- clean nonce-country sources rose only from 1 to 2;
- same-source nonce contrasts dropped from 7 to 6;
- real-country lure correctness fell from 10 to 0.

The answer-text scoring repair preserved real factual preference better than
chat rendering, but still failed:

- clean real-country sources: 26;
- clean nonce-country sources: 1;
- nonce pressure hallucination: 40/40;
- same-source nonce contrasts: 9;
- real-country lure correctness: 15.

Gemma 2 2B IT chat reversed the failure mode rather than solving it:

- clean real-country sources: 36;
- clean nonce-country sources: 40;
- nonce pressure hallucination: 1;
- same-source nonce contrasts: 1;
- real-country lure correctness: 6.

The pressure calibration kept the clean baseline but failed to create a usable
contrast:

- `guess_mild` and `guess_strong` produced zero nonce hallucinations;
- `city_required` produced only one;
- `lure_soft` reproduced the real-country locality failure;
- `lure_strong` reached only six nonce contrasts with one holdout contrast and
  failed real-country lure-error controls.

Lesson:

> Known/unknown behavior is not easy to turn into a stable mechanism substrate.
> Base Gemma either hallucinates too much, abstains too much, or damages factual
> locality. The behavior gate remains unresolved.

## MC002B: Support In Reference Notes

MC002B reconstructed known-versus-unknown as exact support in reference notes
versus unsupported near-neighbor names.

The baseline improved:

- 39 clean supported sources;
- 35 clean unsupported sources.

But pressure still failed:

- `similar_allowed` produced only one same-source unsupported contrast;
- `closest_required` produced zero;
- `closest_required` damaged supported locality;
- `lure_check` became broad abstention.

Lesson:

> Exact-support tasks can produce clean supported/unsupported baselines, but the
> pressure condition needed for hidden-state mechanism work is still too weak.

## MC003: Delayed Copy

MC003 moved to constrained-generation/planning with arbitrary delayed-copy code
words.

V1 found a strong pressure transition but failed correct-hint locality.

V2 repaired locality and passed the behavior gate:

- 40 clean baseline sources;
- 40 correct-hint target rows;
- 40 soft-wrong-hint target rows;
- 35 `wrong_hint_pressure` contrasts with 12 in holdout.

The first signature run found a strong residual direction:

- layer 22 residual direction;
- `0.994` holdout AUC;
- above shuffled-label nulls;
- stable across target-order subgroups.

But the first-token output-margin baseline reached `1.000` holdout AUC.

The early-position follow-up selected `after_wait_newline` layer 14:

- `0.997` holdout AUC;
- beat same-position output margin.

But it failed the selected shuffled-label p95 control and did not beat the
pressure/non-pressure condition-trace baseline.

V3 then repaired that behavior-table confound:

- `wrong_hint_balanced` produced 19 target-copying and 21 distractor-following
  rows inside one condition;
- both labels appeared in discovery, holdout, and holdout target-order
  subgroups.

Once the condition confound was removed, the hidden signature failed:

- selected `after_wait` layer 25;
- discovery AUC: `0.994`;
- holdout AUC: `0.238`;
- below output margin and null controls.

Current MC003 label:

> behavior-supported and failed-signature-only

Lesson:

> Strong hidden signatures can disappear when the behavior table is properly
> condition-balanced. Delayed copy is behavior-supported but not
> intervention-ready.

## MC004: In-Context Entity/Code Binding

MC004 tested nonce in-context entity/code binding.

V1 showed the model could bind cleanly:

- neutral rows: 40/40 target-correct;
- cautious rows: 40/40 target-correct;
- correct-hint rows: 40/40 target-correct.

But wrong-hint pressure was too weak, producing at most `2/40`
distractor-following rows.

V2 changed the pressure to an original-reference-note versus later-update
conflict and passed the behavior gate:

- `update_prefer_latest` produced 20 original-note answers;
- 19 update-following answers;
- one other row;
- both labels appeared in discovery, holdout, and holdout target-order
  subgroups.

The V2 signature failed:

- selected `prompt_end` layer 20 reached `1.000` discovery AUC;
- holdout AUC was only `0.738`;
- same-position output margin reached `1.000` holdout AUC.

The lead-time audit found a closer miss:

- pre-update `after_question` layer 23 reached `0.905` holdout AUC;
- same-stage output margin was only `0.262`;
- but shuffled-label p95 was `0.929`;
- target-order subgroup robustness was not clean.

Current MC004 label:

> behavior-supported and failed-signature result

Lesson:

> Pre-decision internal signals can appear before output margins, but shuffled
> selection and subgroup robustness are hard. A high AUC alone is not enough.

## MC005: Associative Lookup And Source-Value Attention

MC005 is the strongest positive line in the project.

It switched from broad truth/knowledge behavior to a narrower synthetic
associative lookup task on Qwen3-1.7B.

The task asks the model to read source key/value pairs and return the value for
a queried key. This is not general knowledge, but it is mechanistically useful:
the correct answer is inside the prompt, and source-token interventions can be
defined precisely.

### MC005 V1

V1 produced the first strong positive path result:

- 41/48 clean key/value lookup rows;
- selected source-value attention head: layer 16 head 14;
- discovery AUC: `1.000`;
- holdout AUC: `1.000`;
- full-path target source-value mask reduced holdout margin by `-5.154`;
- full-path target mask flipped 11/13 target wins;
- full-path distractor mask moved margin upward.

But selected-layer locality failed:

- layer 25 target masking reduced holdout margin by `-0.606`;
- the wrong-layer target mask was stronger at `-0.913`.

Verdict:

> broad source-value path control, not compact localized mechanism

### MC005 V2

V2 audited layer bands.

The preregistered selector chose `all_layers`, so the localization gate failed.
But late bands were strongly diagnostic:

- `late_20_26` target source-value masking reduced holdout margin by `-3.976`;
- it flipped 8/13 target wins;
- distractor masking increased margin by `+1.615`;
- random-value masking was near null.

Verdict:

> late-heavy source-value path inside a broader full-path dependency

### MC005 V3

V3 excluded `all_layers`, enlarged the row bank to 96 rows, and added an
arrow-layout holdout.

It passed:

- discovery selected `late_20_26`;
- same-layout holdout target masking: mean delta `-4.497`, 10/18 target-win
  flips;
- layout-holdout target masking: mean delta `-5.375`, 17/23 flips;
- distractor and random controls did not match.

This is the first passed internal control surface in the rebuild.

Scope remains narrow:

- synthetic associative lookup;
- Qwen3-1.7B;
- late-band aggregate, not single head or single layer.

### MC005 V4 to V9: Reliability Atlas

V4 expanded reliability:

- all 8 lookup atlas scenarios worked;
- pair counts 3/5/8 worked;
- dash, arrow, and sentence layouts worked;
- shifted lexicon worked;
- greedy next-token readout worked.

But one off-target null failed.

V5 repaired primary out-of-grammar nulls:

- explicit-answer, no-reference, and non-lookup prompts were clean;
- the same-grammar diagnostic remained only weak.

V6 decomposed same-grammar boundaries:

- irrelevant source value/key/colon/non-source value masks were clean or weak;
- one seed-17 final query-label control changed three target-win rows.

V7 ran a 3-seed query-label sweep:

- source/control arms stayed clean;
- final-position controls and prompt surfaces did not.

V8 found marker specificity:

- `Response:` and `Output:` were clean across seeds 17, 23, and 31;
- `Answer:` and `Result:` were less stable.

V9 fixed `Response:` and passed a compact atlas:

- 6/6 lookup scenarios worked;
- 6/6 source-line, off-target, and answer-absent null scenarios were clean;
- seeds 17, 23, and 31 all passed.

Verdict after V9:

> narrow late-band source-value control surface with a passed compact
> `Response:` reliability atlas

Still blocked:

- model-family generality;
- longer-context robustness;
- single-head/single-layer circuit claims.

### MC005 V10 to V14: Size And Longer-Context Reliability

V10 tested Qwen3-0.6B size replication.

Lookup replicated:

- all six lookup scenarios worked;
- target source-value mean deltas ranged from `-3.820` to `-4.919`;
- target-win losses ranged from 14 to 21 rows.

But strict reliability failed:

- one seed-23 answer-absent null was weak;
- mean delta was small, but two target-win rows changed.

V11 expanded the Qwen3-0.6B answer-absent diagnostic:

- five 128-row seeds;
- only 1/5 full seeds was clean;
- failures concentrated around final-label and final-colon controls.

V12 tested marker choice on Qwen3-0.6B:

- `Response:`, `Output:`, `Answer:`, and `Result:` were compared;
- no marker was clean across all five full seeds.

V13 returned to Qwen3-1.7B and extended to pair counts 12 and 16:

- lookup and off-target nulls worked through pair count 16;
- one pair16 answer-absent final-label weak null remained.

V14 reran that answer-absent boundary:

- pair counts 12/14/16;
- five seeds;
- 128 rows per pair-count/seed;
- all 15 scenarios were clean.

Verdict:

> Qwen3-1.7B longer-context answer-absent reliability looks much stronger than
> Qwen3-0.6B. The 0.6B boundary is real and should not be hand-waved away.

### MC005 V15 to V21: Localization To Layers 24-26

V15 tried compact head/layer localization inside the late band.

Selected candidate:

- `full_l20_26_upper_heads`;
- directional and null-clean;
- recovered only `38.9%` of the full late-band holdout effect.

Verdict:

> full band required

V16 decomposed the full-band effect:

- full layers 20-26 all-head masking: mean delta `-8.6110`, 112 target-win
  losses;
- lower plus upper heads were superadditive;
- small layer slices were superadditive;
- `slice_l23_26_all` became the next lead.

V17 confirmed `slice_l23_26_all`:

- fresh lookup/null seeds;
- recovered `72.2%` of full-band effect;
- beat smaller slices and source controls;
- answer-absent nulls stayed clean.

V18 split layers 23-26:

- selected `slice_l24_26_all`;
- recovered `92.4%` of parent layers-23-26 effect;
- beat source controls;
- answer-absent nulls stayed clean.

V19 tried to split layers 24-26:

- selected `slice_l24_25_all`;
- source-specific, rank-stable, null-clean;
- recovered only `49.9%` of parent effect.

V20 confirmed layers 24-26 as a three-layer interaction:

- all leave-one-layer-out pairs recovered less than 60%;
- single-layer sum was strongly superadditive;
- answer-absent nulls stayed clean.

V21 stress-tested fixed layers 24-26:

- dash/colon, arrow, sentence layouts;
- shifted lexicon;
- pair count 20;
- fresh answer-absent null seeds;
- all six lookup stress scenarios worked.

Verdict:

> the current supported internal intervention surface is an all-head
> layers-24-26 aggregate interaction block under synthetic associative lookup

### MC005 V22 to V25: Row-Level Heterogeneity

V22 asked whether the layers-24-26 interaction is visible row by row.

Parent/control/null replication passed, but the stronger row-level all-three
promotion gate failed:

- 67 all-three margin rows;
- 67 was only `26.48%` of parent-effect rows;
- threshold was 40%.

V23 stratified row heterogeneity:

- source position and query index were strongest;
- baseline margin was secondary;
- some positions had near-zero all-three rows;
- some mid/late positions reached 43.75%.

V24 causally moved source position:

- all-three fraction rose directionally from `0.1575` to `0.3175`;
- median best-pair share fell;
- but preregistered thresholds failed.

V25 crossed target source position with distractor position and margin:

- parent/control/null validity passed;
- factorial cell contrast was nearly flat;
- best-minus-worst all-three range was only `0.0469` against a `0.25`
  threshold.

Verdict:

> row heterogeneity is real, but simple source-position or layout-factor
> explanations are insufficient

### MC005 V26 to V28: Internal Row Signature And Failed Interventions

V26 found an internal row signature for all-three row structure:

- selected `l20_target_value`;
- discovery AUC: `0.9840`;
- holdout AUC: `0.7933`;
- best non-internal baseline holdout AUC: `0.5586`;
- shuffled-selection null p95: `0.6510`;
- subgroup AUCs were stable at `0.7890-0.7973`.

This is a real internal signature.

V27 tested additive residual steering of that signature:

- source signature reproduced;
- baseline `none`: 44/128 all-three rows;
- `plus_target`: 43/128 all-three rows;
- median best-pair share worsened;
- controls matched or exceeded the primary row-structure change;
- parent source controls and answer-absent residual nulls passed.

Verdict:

> clean negative causal test

V28 tested donor activation replacement:

- source signature reproduced again;
- positive target donor replacement reduced baseline target wins from 128/128
  to 106/128;
- parent-effect rows collapsed from 128 to 3;
- parent source controls failed;
- final-colon donor replacement broke answer-absent nulls.

Verdict:

> donor replacement disrupted the parent surface and failed controls

Lesson:

> A real internal signature is not automatically a causal steering handle.

### MC005 V29 to V31: Attention-Write Mediation And Null Boundary

V29 targeted the supported parent surface directly by replacing final-query
self-attention output writes in layers 24-26.

This produced the sharpest mechanism localization so far:

- direct target source masking and write replacement both had mean delta
  `-6.1978`;
- both had 11 target-win losses;
- distractor write replacement moved margin upward with zero losses;
- random write replacement was near zero;
- single-layer write replacement did not recover the parent effect.

But the strict reliability gate failed:

- seed 251 `non_source_control_value` write replacement had one target-win gain
  despite small mean delta.

V30 replayed and expanded the null boundary:

- the seed-251 gain reproduced;
- fresh 8-seed, 1,024-row sweep found three more strict row-change failures;
- all fresh arms stayed within mean-delta tolerance.

So the failure is not broad margin drift. It is low-margin answer-absent row
sign flips under write replacement.

V31 tested a tight baseline-margin explanation:

- lookup target write effect reproduced exactly;
- all 11 lookup target losses had baseline margin greater than 2.0;
- a new fresh null flip occurred at margin 0.0;
- but the imported V30 replay flip had absolute margin 0.75, so the
  preregistered 0.5 cutoff failed.

Current MC005 verdict:

> exact lookup mediation by layers-24-26 final-query attention writes, with a
> mapped failure mode: high-margin lookup mediation and rare low-to-moderate
> margin answer-absent null flips. Not a full mechanism card.

## MC006: Parametric Fact Override

MC006 is the current knowledge-like branch. It asks whether we can separate
real learned capital facts from task-local false or fictional overwrites.

This is closer to the "knowledge genome" ambition than MC005, but it has been
harder.

### MC006 V1

The first Qwen3-1.7B candidate-scoring smoke failed the behavior substrate.

Results:

- true supporting context: 40/40 correct;
- explicit task-local overwrite: 39/40 override;
- no-context truth: 26/40;
- irrelevant-context truth: 27/40;
- `mistake_context` followed the false reference line on 40/40 rows.

Interpretation:

> The interface measured reference-line authority more than a clean
> parametric-fact/context-overwrite switch.

No hidden-state work should start from V1.

### MC006 V2

V2 replaced the false reference line with a false-claim audit and a fictional
codebook prompt.

It still failed the full substrate:

- direct real-world truth: 25/40;
- true verified context: 40/40;
- false-claim audit truth: 29/40;
- false-claim audit override: 8/40;
- fictional override: 40/40;
- `real_after_fiction` truth: 9/40;
- `real_after_fiction` override: 21/40.

A narrower lead appeared:

- 23/40 sources were clean across direct real, false-claim audit, and
  fictional override;
- 6 of those were original holdout sources.

But V2 itself failed.

Important learning from V2:

> The model can follow a fictional codebook very easily, but real-world
> answering after seeing a fictional rule is fragile. The `real_after_fiction`
> boundary is a real knowledge/control stress point.

### MC006 V3

V3 source-selected the 23 V2-clean sources and tested prompt paraphrases.

The real-world side held:

- direct real paraphrase: 23/23 true;
- true-fact paraphrase: 23/23 true;
- false-claim check: 22/23 true with zero override.

The run failed because the weakened fictional-code prompt selected override on
only 18/23 rows and clean contrasts reached only 17/23.

Verdict:

> real-world paraphrase held, but fictional-code pressure was too weak

### MC006 V4

V4 kept the V3 selected source bank and real-world prompts, but restored the
stronger V2 fictional-codebook wording.

It passed a narrow behavior substrate:

- direct real paraphrase: 23/23 true;
- true-fact paraphrase: 23/23 true;
- false-claim check: 22/23 true with zero override;
- fictional code lookup: 23/23 override;
- clean contrasts: 22/23;
- original holdout clean contrasts: 5/6.

This allowed hidden-state discovery only inside the V4 source-selected,
prompt-bounded contract.

### MC006 V5

V5 tested hidden-state discovery on V4 clean rows.

It found a perfect hidden classifier:

- 22 clean sources;
- 5 original holdout sources;
- 88 rows;
- selected final-prompt-token layer-0 direction;
- holdout AUC: `1.000`;
- all 5 holdout sources had real-world mean score greater than fictional-code
  score.

But controls matched it:

- output-margin holdout AUC: `1.000`;
- prompt-length holdout AUC: `1.000`;
- prompt-format holdout AUC: `1.000`;
- shuffled-label selection p95: `1.000`.

Verdict:

> hidden signal present, but output/prompt/null-confounded

No intervention should start from V5.

### MC006 V6

V6 tried to repair the prompt-family confound structurally.

It put both labels inside one counterbalanced two-source prompt family with
exact query-tag and expected-label balance.

Structural controls passed, but behavior failed:

- `balanced_real` true: 17/22;
- `balanced_fictional` override: 20/22;
- clean source contrasts: 15/22;
- holdout clean contrasts: 4/5.

Diagnostic:

> `real_query_failed`

Verdict:

> same-prompt-family repair failed before hidden-state discovery

### MC006 V7

V7 removed true-answer prompt leakage and used explicit modes:

- `REAL_WORLD_CAPITAL`;
- `FICTIONAL_CITY_CODE`.

Structural checks passed, including no true-capital prompt leaks.

Behavior failed hard:

- `mode_real` true: 8/22;
- `mode_real` override: 13/22;
- `mode_real` lure: 1/22;
- `mode_fictional` override: 22/22;
- clean contrasts: 8/22;
- holdout clean contrasts: 1/5.

Verdict:

> the exposed fictional city line dominated real-world mode under
> candidate-scored evaluation

### MC006 V8

V8 removed hidden candidate scoring and tested greedy generated answers under
the same mode-gated prompt.

Structural checks passed, including no true-capital prompt leaks.

Behavior failed:

- strict parseability: 35/44;
- `mode_real`: 6 true, 9 override, 7 unparsed;
- `mode_fictional`: 20 override, 2 unparsed;
- clean contrasts: 6/22;
- holdout clean contrasts: 1/5.

Diagnostic:

> `parseability_failed`

Lesson:

> Candidate scoring was not the only blocker. Raw generation added format
> failures and still did not recover same-prompt-family real-world behavior.

### MC006 V9

V9 audited V8 with a lenient parser.

The parser accepted full candidate mentions anywhere in the generated suffix.

Parseability improved:

- strict V8 parseability: 35/44;
- lenient V9 parseability: 41/44.

But behavior still failed:

- `mode_real` true: 11/22;
- `mode_fictional` override: 21/22;
- clean contrasts: 10/22;
- holdout clean contrasts: 2/5.

Diagnostic:

> `real_mode_failed`

Lesson:

> The strict parser was a real but secondary blocker. Parser rescue alone did
> not recover a behavior substrate.

### MC006 V10

V10 changed the rendering contract instead of the parser.

It used:

- Qwen3 chat-template generation;
- `enable_thinking=False`;
- a strict system instruction to return exactly one city name;
- no true-capital prompt leakage.

V10 passed:

- strict parseability: 44/44;
- `mode_real` true: 20/22;
- `mode_fictional` override: 22/22;
- clean contrasts: 20/22;
- holdout clean contrasts: 5/5.

Two source contrasts remained non-clean:

- Brazil: real mode generated the lure `Rio de Janeiro`, fictional mode
  generated `Recife`;
- Croatia: both modes generated the override `Zadar`.

Diagnostic:

> `chat_generated_v10_substrate_passed`

This is the best current knowledge-like behavior substrate.

It does not prove an internal signature or intervention. It only permits
hidden-state signature discovery on the V10 table.

### MC006 V11

V11 tested V10 for a hidden-state signature.

Primary rows:

- V10 clean source-level contrasts only;
- 20 clean sources;
- 5 original holdout sources;
- 40 balanced rows.

It found a perfect hidden classifier:

- selected candidate: `layer_0` at final rendered prompt token;
- selected hidden holdout AUC: `1.000`;
- holdout source-pair ordering: 5/5 passed.

But controls matched it:

- output-margin holdout AUC: `1.000`;
- requested-mode holdout AUC: `1.000`;
- prompt-length holdout AUC: `0.880`;
- shuffled-label selection p95: `1.000`.

Diagnostic:

> `requested_mode_confounded`

Interpretation:

> The V10 table has a strong internal separation between real-world mode and
> fictional-code mode. That separation is not mechanism-grade because the
> requested mode is explicitly in the prompt, output logits already separate
> the rows perfectly, and shuffled layer selection can also reach perfect
> holdout AUC on this small table.

No intervention should start from V11.

### The Next MC006 Lead

The next MC006 repair needed to make the critical labels vary under a matched
requested-mode or matched prompt surface. The first diagnostic used the older
V2 `real_after_fiction` boundary:

- every prompt says a fictional codebook is not real-world geography;
- every prompt asks for the current real-world national capital;
- the model sometimes answers the true capital and sometimes follows the
  fictional codebook.

For `real_after_fiction`, excluding lure rows:

- 30 binary rows remain;
- 9 true-answer rows;
- 21 override-answer rows;
- discovery: 6 true, 12 override;
- calibration: 2 true, 4 override;
- holdout: 1 true, 5 override.

This is imbalanced, especially in holdout, but it is the right kind of matched
surface: the label is not directly the requested-mode string. It can support a
diagnostic signature audit, though not a strong promotion claim unless the
holdout balance is repaired.

### MC006 V12

V12 ran that matched-surface diagnostic.

It found another perfect hidden classifier:

- selected candidate: `layer_7` at final prompt token;
- selected hidden discovery AUC: `1.000`;
- selected hidden holdout AUC: `1.000`.

But promotion failed:

- holdout balance was only 1 true-answer row and 5 override-answer rows;
- candidate-score margin holdout AUC was `1.000`;
- next-token output-margin holdout AUC was `1.000`;
- prompt-length holdout AUC was `0.800`;
- shuffled-label selection p95 was `1.000`.

Diagnostic:

> `holdout_balance_failed`

Interpretation:

> V12 removed the explicit requested-mode confound from V11, but did not produce
> a mechanism-grade signature. The hidden signal is matched by candidate
> scoring, next-token output margin, and shuffled-label selection, and the
> source-disjoint holdout has too little truth-following support.

No intervention should start from V12. The next MC006 repair must build a
larger matched-surface table with multiple truth-following and
fictional-code-following holdout rows before another signature gate.

### MC006 V13

V13 repaired the V12 table construction problem directly. It stopped using
candidate-scored labels and instead generated answers under six matched
`real_after_fiction` prompt templates. Every prompt contained a fictional city
mapping, asked for the current real-world capital, and did not list the true
capital. The run used all 40 V2 sources and preserved the V2 split:

- 24 discovery sources;
- 8 calibration sources;
- 8 holdout sources;
- 6 templates;
- 240 generated rows.

The template selector used only discovery and calibration rows. It chose the
template with the best balance between true-answer and override-answer rows,
then highest binary volume, then fewest side rows. Holdout rows were not used
for selection.

The selected template was:

> `fake_mapping_warning`

Structural checks all passed:

- exactly 40 sources;
- exactly 6 templates;
- exactly 240 rows;
- exactly 40 rows per template;
- exactly 8 holdout rows per template;
- every source appeared once per template;
- no duplicate record ids;
- no duplicate candidate answers;
- no true-capital prompt leaks.

The selected template results were:

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

The selected table passed the balance gates:

- non-holdout true floor: `14 >= 6`;
- non-holdout override floor: `7 >= 6`;
- holdout true floor: `6 >= 2`;
- holdout override floor: `2 >= 2`;
- holdout side-row ceiling: `0 <= 4`.

But it failed the binary-volume gate:

> `29/40 < 30/40`

Diagnostic:

> `binary_volume_failed`

This is an informative near miss. V13 shows that a matched generated prompt
surface can produce both truth-following and fictional-code-following rows on
source-disjoint holdout. It also shows why the gate should stay strict: the
selected template still had 11 side rows, mostly unparsed explanatory answers
or lure answers. A hidden-state audit on this table would inherit a brittle
behavior substrate.

No intervention should start from V13. The next MC006 repair should improve the
generated-answer output contract or expand the matched prompt/source bank until
the selected table clears binary volume while preserving the V13 holdout
balance.

### MC006 V14

V14 tested whether V13's one-row binary-volume failure was a narrow parser
normalization problem.

It did not rerun the model. It reused the frozen V13 generated output artifact
and changed only the parser:

- Unicode NFKD decomposition;
- removal of combining marks;
- lowercase and whitespace collapse;
- the same strict first-line prefix boundary;
- no substring rescue;
- no explanatory-prefix rescue;
- no alias list.

The parser delta was narrow. Only three rows changed:

- `mc006_v13_brazil_memory_check`: `Brasilia` with diacritic, unparsed -> true;
- `mc006_v13_colombia_fake_mapping_warning`: `Bogota` with diacritic,
  unparsed -> true;
- `mc006_v13_colombia_memory_check`: `Bogota` with diacritic,
  unparsed -> true.

All changed rows came from V13 `unparsed`; no candidate label changed into a
different candidate label.

Template selection was recomputed under normalized parsing. It still selected:

> `fake_mapping_warning`

The selected template passed:

- binary rows: 30/40;
- true answers: 21/40;
- override answers: 9/40;
- side rows: 10/40;
- non-holdout true answers: 15;
- non-holdout override answers: 7;
- holdout true answers: 6;
- holdout override answers: 2;
- holdout side rows: 0;
- prompt leaks: 0.

Diagnostic:

> `parser_normalized_generated_substrate_passed`

Interpretation:

> MC006 finally has a matched generated-answer behavior substrate for this
> capital-fact versus fictional-code contamination task, but only under a
> narrow accent-normalized strict parser. This permits a hidden-signature
> diagnostic. It does not permit intervention.

### MC006 V15

V15 tested the V14 table for a hidden-state signature.

Primary rows were the 30 selected-template binary rows:

- 21 true-answer rows;
- 9 override-answer rows;
- holdout: 6 true, 2 override;
- side rows excluded but documented: 3 lure, 7 unparsed.

It found a perfect hidden classifier:

- selected hidden candidate: `layer_16` at final prompt token;
- discovery AUC: `1.000`;
- holdout AUC: `1.000`.

But the easy baselines matched it:

- candidate-score margin holdout AUC: `1.000`;
- next-token output-margin holdout AUC: `1.000`.

Other baselines were weaker:

- prompt length holdout AUC: `0.583`;
- final prompt token id holdout AUC: `0.500`;
- generated first-token id holdout AUC: `0.833`;
- parser-normalization delta holdout AUC: `0.500`;
- shuffled-label selection p95: `0.833`.

Diagnostic:

> `candidate_score_confounded`

The stricter interpretation is:

> V15 is candidate-score confounded and next-token output-margin confounded.

No intervention should start from V15. The behavior substrate is now good enough
to study, but the current final-prompt-token hidden direction does not beat
what the model's output interface already exposes.

### MC006 V16

V16 tested whether the V15 failure was only a final-answer-interface problem.
Instead of probing only the final `Answer:` token, it extracted hidden states
from earlier line-boundary positions in the same V14 selected table:

- `after_mapping_line`;
- `after_instruction_line`;
- `after_question_line`;
- `after_return_line`;
- `final_prompt_token` as a reference only.

Only the four pre-output positions were selectable. Token positions were mapped
with the fast-tokenizer full-prompt offset map and no special tokens, and each
full prompt was run once without padding.

Primary rows stayed the same as V15:

- 30 selected-template binary rows;
- 21 true-answer rows;
- 9 override-answer rows;
- holdout: 6 true, 2 override;
- side rows excluded but documented: 3 lure, 7 unparsed.

V16 found an earlier hidden separator:

- selected pre-output candidate: `after_mapping_line/layer_4`;
- discovery AUC: `1.000`;
- holdout AUC: `1.000`.

That selected candidate beat the same-position controls:

- same-position next-token output-margin holdout AUC: `0.917`;
- prefix token count holdout AUC: `0.583`;
- position token id holdout AUC: `0.333`;
- shuffled-label selection p95: `0.917`.

This is the first MC006 result that supports a narrow lead-time diagnostic: at
the fake-mapping-line boundary, before the explicit instruction, question,
return-format line, or `Answer:` marker, the model already carries a hidden
true-versus-override trace that is stronger than the same-position output-logit
view.

But the global output-interface controls still blocked mechanism promotion:

- candidate-score margin holdout AUC: `1.000`;
- final next-token output-margin holdout AUC: `1.000`;
- final-position reference hidden candidate: `final_prompt_token/layer_16`
  with discovery and holdout AUC `1.000`.

Diagnostic:

> `leadtime_signal_supported_but_output_global_confounded`

Interpretation:

> V16 is useful as a monitoring/lead-time result, not as an intervention-ready
> mechanism signature. It shows that the behavior label is present internally
> before the final answer interface, but the completed prompt's output
> interface already exposes the same label perfectly.

No intervention should start from V16 unless the next preregistration explicitly
treats this as a monitoring-only or known-confounded causal stress test.

### MC006 V17

V17 ran that known-confounded causal stress test.

It asked:

> If we add or subtract the V16 `after_mapping_line/layer_4` direction at the
> selected early prompt token, does the final answer move in the predicted
> direction, and do simple controls match the effect?

Source checks passed:

- V14 was the behavior substrate;
- V16 was the lead-time signature source;
- V16 selected `after_mapping_line/layer_4`;
- V16's source hash matched V14;
- V17 structural checks matched V15/V16.

The current-run baseline reproduced the frozen V14 holdout labels:

- 8/8 holdout primary rows matched;
- holdout label mix remained 6 true and 2 override.

V17 fit the direction on non-holdout primary rows and ran a dose sweep:

- signature-score shifts: `1.0`, `2.0`, `4.0`;
- selected arms: `plus_selected`, `minus_selected`;
- controls: `random_selected`, `plus_wrong_position`,
  `plus_wrong_layer`;
- readouts: strict NFKD generated-answer parse and final true-minus-override
  first-token margin.

Holdout result:

| Dose | Plus Delta | Minus Delta | Selected Abs Effect | Max Control Abs Effect | Predicted Label Changes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | `+0.0469` | `+0.0625` | `0.0547` | `0.0469` | `0` |
| 2.0 | `+0.0625` | `-0.0156` | `0.0391` | `0.0625` | `0` |
| 4.0 | `+0.1250` | `-0.1406` | `0.1328` | `0.1875` | `0` |

The strongest directional dose was `4.0`, but it still failed the preregistered
causal stress gate:

- `plus_selected` did not reach the `+0.25` margin floor;
- `minus_selected` did not reach the `-0.25` margin floor;
- max control effect exceeded the selected-arm average absolute effect;
- no holdout generated label changed in the predicted direction.

Side rows did not show broad parse corruption:

- max selected-arm side-label changes: 1/10;
- preregistered ceiling: 3/10.

Diagnostic:

> `intervention_failed`

Interpretation:

> V17 does not show that the V16 early signature was fake. It shows that the
> V16 early direction is not a simple additive residual control vector under
> this intervention family. MC006 now has behavior support and a lead-time
> signature, but the first causal route from that signature failed.

### MC006 V18

V18 tested the advisor/reviewer concern directly:

> Is the V16 pre-output signal still useful if we try to match or residualize
> away candidate-score and final-output margins?

V18 used the same V14 matched generated-answer table and re-extracted row-level
hidden scores because V16 had not stored per-row projection scores.

The V16 early signal reproduced:

- selected candidate: `after_mapping_line/layer_4`;
- discovery AUC: `1.000`;
- holdout AUC: `1.000`.

The easier global baselines also reproduced:

- candidate-score margin holdout AUC: `1.000`;
- final next-token output-margin holdout AUC: `1.000`;
- same-position next-token output-margin holdout AUC: `0.917`.

The decisive result was the margin-overlap audit:

- candidate-score holdout true/override overlap: false;
- candidate-score separation gap: `0.5976`;
- final-output holdout true/override overlap: false;
- final-output separation gap: `0.6027`;
- matched true/override holdout pairs at z <= 0.5: `0` for both margins.

Residualizing the hidden score against the global margins erased the holdout
signal:

- hidden residual after candidate-score margin: holdout AUC `0.333`;
- hidden residual after final-output margin: holdout AUC `0.333`;
- hidden residual after both: holdout AUC `0.333`.

Diagnostic:

> `global_margin_separation_blocks_matching`

Interpretation:

> V18 does not show that the V16 lead-time signal is fake. It shows that the
> current V14 row set cannot adjudicate a margin-matched MC006 mechanism claim:
> the behavior label is already separated in global output/candidate geometry
> on holdout, with no matching region. At that point the route had to build a
> new overlapping-margin table or switch to source/path-specific lead-time
> audits.

### MC006 V19

V19 tested the first version of that new table-construction route.

It expanded MC006 from one V14 prompt contract to a 400-row generated prompt
bank:

- 40 sources;
- 10 real-after-fiction prompt variants;
- strict NFKD first-line parsing;
- candidate mean-logprob margins;
- final next-token true-minus-override margins;
- non-holdout template selection.

The selected template was:

> `separate_task_weak`

Selected-template behavior:

- total rows: 40;
- binary rows: 35;
- side rows: 5;
- non-holdout true / override: 11 / 16;
- holdout true / override: 4 / 4;
- holdout side rows: 0;
- prompt leaks: 0.

This is a real improvement over the V18 table geometry. But strict overlap
still failed.

Holdout candidate-score z-ranges:

- true rows: 0.2882 to 0.7235;
- override rows: -0.5745 to 0.0311;
- overlap: false;
- separation gap: 0.2571;
- matched pairs at z <= 0.5: 4.

Holdout final-output z-ranges:

- true rows: 0.1146 to 0.8253;
- override rows: -0.5752 to -0.1363;
- overlap: false;
- separation gap: 0.2508;
- matched pairs at z <= 0.5: 4.

Diagnostic:

> `non_holdout_candidate_margin_overlap_failed`

Interpretation:

> V19 shows that MC006 can move from "no matching region" to "near-matched
> pairs with balanced holdout labels," but it still has not produced a strict
> overlapping-margin behavior substrate. A V19 hidden-state run would have to
> be labeled as an approximate pair-matched diagnostic, not a mechanism-grade
> signature attempt.

### MC006 V20

V20 audited whether V19's existing prompt bank could be rescued by offline
selection.

It was an upper-bound audit:

- source: V19 400-row prompt bank;
- binary rows: 257;
- no new model generation;
- pooled all binary rows across templates;
- checked every single template;
- recomputed candidate-score and final next-token overlap;
- counted joint candidate/final near-pairs.

The result was decisive for strict overlap:

- pooled non-holdout candidate overlap: true;
- pooled non-holdout final-output overlap: false;
- pooled holdout candidate overlap: false;
- pooled holdout final-output overlap: false;
- any single-template full strict-overlap gate: false;
- any template non-holdout final-output overlap: false;
- any template holdout final-output overlap: false.

Pooled holdout final-output margin:

- true rows: 42;
- override rows: 15;
- true z-range: -0.4833 to 1.6183;
- override z-range: -1.5239 to -0.5445;
- z separation gap: 0.0612.

V20 did find near-pairs:

- pooled holdout joint candidate/final pairs at z <= 0.5: 28;
- selected-template holdout joint pairs at z <= 0.5: 3.

Diagnostic:

> `strict_final_margin_overlap_absent`

Interpretation:

> V20 closes strict-overlap selection inside the V19 bank. At that point,
> approximate pair-matched lead-time diagnostics were the only bounded use of
> the existing rows, while strict mechanism-grade margin-overlap promotion
> required new generated rows or a different source/path route.

### MC006 V21

V21 ran the bounded diagnostic that V20 allowed: approximate pair matching on
the existing V19 pooled binary row bank.

It used joint true/override pairs where both candidate-score-margin and
final-margin z-deltas were <= 0.5.

Pair volume was sufficient for a diagnostic:

- non-holdout joint pairs: 451;
- non-holdout participants: 63;
- holdout joint pairs: 28;
- holdout participants: 14.

The hidden result did not survive:

- selected candidate: `after_mapping_line/layer_14`;
- non-holdout hidden pair accuracy: 0.9180;
- holdout hidden pair accuracy: 0.2857;
- holdout hidden AUC: 0.5603.

The easy margin baselines also did not die:

- candidate-score margin holdout pair accuracy: 1.0000;
- V19 final next-token margin holdout pair accuracy: 1.0000;
- final-prompt next-token margin holdout pair accuracy: 1.0000.

Diagnostic:

> `approximate_pair_matching_failed_margin_baselines`

Interpretation:

> V21 closes the existing V19/V20 approximate pair-matching route. Near-pairs
> at z <= 0.5 are not a strong enough control when the candidate-score and
> final-output margins still rank every holdout pair in the behavior-label
> direction.

### MC006 V22

V22 ran the source/path-specific audit that V21 left open. It did not try to
promote another scalar separator on the same row bank. It mapped where the
true/override label becomes visible across token positions:

- `mapping_country_token`;
- `mapping_value_token`;
- `question_country_token`;
- `after_mapping_line`;
- `after_instruction_line`;
- `after_question_line`;
- `after_return_line`;
- `final_prompt_token`.

Position mapping succeeded for all 257 pooled V19 binary rows.

The source/path-selected signal was strongest at the queried country token:

- best pair-selected candidate: `question_country_token/layer_20`;
- holdout AUC: 0.800;
- holdout pair accuracy: 0.607;
- best AUC-selected candidate: `question_country_token/layer_22`;
- holdout AUC: 0.832;
- holdout pair accuracy: 0.714;
- same-position output holdout pair accuracy: 0.679.

The full curve found stronger later pre-output monitors:

- `after_instruction_line/layer_27`: holdout pair accuracy 0.750;
- `after_question_line/layer_22`: holdout pair accuracy 0.893;
- `after_return_line/layer_11`: holdout pair accuracy 0.786.

But global margins still dominated:

- candidate-score margin holdout pair accuracy: 1.0000;
- V19 final next-token margin holdout pair accuracy: 1.0000;
- final-prompt next-token margin holdout pair accuracy: 1.0000.

Diagnostic:

> `source_path_final_margin_shadow`

Interpretation:

> V22 supports the claim that MC006 has source/path and line-boundary
> lead-time monitor structure. It does not support a mechanism signature or
> intervention. The stronger result is a law-shaped one: for this current row
> bank, source/path evidence appears before the final answer interface, but the
> final candidate/output geometry still perfectly orders the behavior labels.

## What We Have Learned Across Families

### 1. Behavior Substrates Are The Real First Gate

Many plausible mechanism projects fail before hidden-state work is justified.

MC002 and MC002B show that known/unknown behavior is hard to construct. MC006
V1, V2, V3, V6, V7, V8, and V9 show that even a simple capital-fact task can
fail because the prompt interface measures reference authority, fictional-rule
salience, parseability, or requested-mode text rather than a clean internal
knowledge switch.

The project should never start probing hidden states until the behavior table
is clean enough that labels mean what we think they mean.

### 2. Output Margins Are Brutal Baselines

Output/logit baselines repeatedly match or beat hidden directions:

- MC001 dense truth/agreement steering;
- MC001B Qwen3-1.7B residualized rescue;
- MC003 delayed-copy first signature;
- MC004 binding signature;
- MC006 V5 prompt-bounded signature;
- MC006 V11 chat-mode signature;
- MC006 V12 matched real-after-fiction signature;
- MC006 V15 final-prompt-token matched generated signature;
- MC006 V16 pre-output signature at the global output-interface level.
- MC006 V18 same-table margin-matched lead-time audit.
- MC006 V19 overlapping-margin table search.
- MC006 V20 strict-overlap selection audit.
- MC006 V21 approximate pair-matched lead-time audit.
- MC006 V22 source/path lead-time curve.
- MC006 V23 final-margin sign-barrier audit.
- MC006 V24 delayed-city interface audit.

If a hidden direction does not beat output margin, it may still be a useful
diagnostic signal, but it is not mechanism-grade.

### 3. Prompt Format Can Produce Perfect Hidden Signatures

MC006 V5 and V11 are the clearest examples.

In V5, the hidden direction separated real-world prompts from fictional-code
prompts perfectly. But prompt length and prompt format also separated them
perfectly.

In V11, the hidden direction separated V10 real-world mode from fictional-code
mode perfectly. But the requested-mode string was explicit in the prompt and
also reached perfect holdout AUC.

Lesson:

> A perfect hidden classifier can be a perfect prompt classifier.

### 4. Shuffled-Label Selection Is Necessary

Small tables with many layer candidates can overfit.

Several experiments found high hidden AUCs that failed because shuffled-label
selection p95 was too high:

- MC003 early-position signature;
- MC004 lead-time audit;
- MC006 V5;
- MC006 V11.
- MC006 V12.

The project should keep shuffled-selection nulls as mandatory for any hidden
signature gate.

### 5. Interventions Can Work But Still Fail Reliability

MC005 shows this clearly.

Late-band source-value masking works. Layers 24-26 final-query attention-write
replacement exactly reproduces the lookup target source-mask effect.

But the strict answer-absent null boundary remains imperfect: rare row-level
flips happen under write replacement.

That is the difference between "we found a causal surface" and "we have a full
mechanism card."

### 6. Synthetic Tasks Are Easier To Mechanize Than Knowledge Tasks

MC005 is synthetic associative lookup, and it has the strongest internal
surface.

MC006 is closer to learned factual knowledge, and it is much harder:

- reference-line authority dominates;
- fictional codebooks are sticky;
- real-world mode can collapse when false city names are visible;
- chat rendering repairs behavior;
- hidden signatures still confound with explicit mode text.

This does not mean MC005 is less valuable. It means MC005 is a controlled
mechanistic sandbox, while MC006 is closer to the actual knowledge problem.

### 7. Small-Model Generalization Is Not Free

MC005 replicated lookup behavior on Qwen3-0.6B, but strict null reliability
failed.

This matters because the objective is small LLMs. A surface that works on
Qwen3-1.7B may not be reliable on Qwen3-0.6B, even if the primary behavior
appears to replicate.

### 8. The Genome Is Multi-Level

The experiments suggest several levels of "control surface":

- prompt text and requested modes;
- answer-option/output margins;
- source-token presence;
- source-value attention paths;
- late-layer aggregate interactions;
- residual row signatures;
- final-query attention writes;
- model-size-specific null behavior.

The genome is not a single magic vector. It is a layered map of surfaces, many
of which are useful but not mechanism-grade.

### 9. The Mixture Is Now A Measured Object

The strongest current genome-scale result is not a new mechanism card. It is a
distribution:

- 11/19 atlas rows are primarily blocked by behavior-substrate or bridge
  construction failures;
- 6/19 are primarily output-geometry shadows;
- 1/19 is a prompt-visible positive control;
- 1/19 is a bounded internal-causal surface with a null boundary.

The non-exclusive pressure classes are equally sharp:

- prompt-contract-visible: 19/19 rows;
- output-geometry-visible: 14/19 rows;
- source-or-prompt-token-dependent: 13/19 rows;
- internal-monitor-present: 5/19 rows;
- internal-causal-surface: 1/19 rows.

That is the current control-surface mixture law. It says the project should not
optimize for "the next pretty vector." It should measure how each behavior
family distributes across prompt contracts, output geometry, source tokens,
internal monitors, bounded causal surfaces, null boundaries, and transfer
failures.

The generated compositional-genome audit now makes that insight validator-
backed. It treats the mixture itself as the current genome-scale object:
19/19 rows are prompt-contract-visible, 14/19 are output-geometry-visible,
13/19 are source-or-prompt-token-dependent, 11/19 are behavior/bridge-
substrate-blocked, 5/19 have internal monitors, 1/19 has a bounded internal
causal surface, and 0/19 are promoted, full-reliability, or transfer-ready
mechanisms. This does not lower the mechanism-card bar. It makes the bar's
shape part of the result.

The generated family matrix now makes the cross-family table explicit. Every
one of the 19 atlas rows has a joined prompt/output/source/internal/frontier/
reliability/transfer/gate row, so future experiments can be evaluated by which
cell they change rather than by whether they add another isolated story.

The generated knowledge ladder now makes the knowledge-specific gap explicit:
level 1 synthetic lookup has the only bounded reference; level 2 semi-synthetic
familiar-entity lookup is behavior-substrate-blocked; level 3 symbolic and
learned-memory bridge rows remain hidden-state-disallowed; level 4 capital-fact
override is monitor-only with no lever; and level 5 real abstention/uncertainty
is not mechanism-ready. The auxiliary truth/agreement, delayed-copy, and
in-context-binding rows remain diagnostics, not knowledge-ladder victories.

### 10. Lead-Time Is A Frontier, Not A Trophy

The current decision-frontier artifact makes commitment timing explicit:

- frontier not reached: 12/19 rows;
- output-visible at or before the frontier: 4/19 rows;
- predecision monitor with no lever: 2/19 rows;
- causal surface not primarily about timing: 1/19 row;
- predecision causal candidates: 0/19 rows.

The two monitor-only rows are MC004 and MC006. That matters: early hidden
signals can exist before same-position output geometry fully catches up, but
the current atlas has not turned any of those signals into a reliable
intervention. Lead-time is useful only when it survives the same audit as every
other mechanism claim.

### 11. Route Closure Is Now Operational

The route-disposition ledger makes experiment death explicit:

- closed before hidden-state work: 11/19 rows;
- output-shadow diagnostic baseline: 3/19 rows;
- failed intervention or mechanism route: 1/19 row;
- monitor-only closed: 1/19 row;
- monitor-only conditional revisit: 1/19 row;
- prompt-visible positive control: 1/19 row;
- bounded frozen mechanism: 1/19 row.

There are 0 hidden-state-ready atlas routes. That is not a failure of the
project; it is the current honest state of the map. A route can only reopen by
satisfying its promotion rule, not by accumulating adjacent prompt tweaks.

### 12. Transfer Is Mostly Unproven, Fragile, Or Failed

The transfer matrix makes widening explicit:

- transfer-ready mechanisms: 0/19 rows;
- untested transfer value: 14/19 rows;
- low transfer value: 2/19 rows;
- medium transfer value: 2/19 rows;
- failed transfer value: 1/19 row.

The strongest row, MC005, is only a bounded transfer-fragile reference because
null and model-size boundaries remain first-class. MC006 is the failed-transfer
exemplar: locked-coordinate transfer, expanded transfer-bank construction, and
transfer-role repair did not create a transfer-ready route. Cross-model or
medium evidence is not transfer success until the mechanism gates transfer too.

### 13. Reliability Failure Is Now A Measured Distribution

The reliability matrix turns the three-gate mechanism-card standard into a
row-level audit:

- full-reliability mechanisms: 0/19 rows;
- bounded reliability references: 1/19 rows;
- behavior/bridge blocked: 11/19 rows;
- output-shadow diagnostics: 3/19 rows;
- monitor-only no-lever rows: 2/19 rows;
- failed-intervention route: 1/19 row;
- prompt-visible positive control: 1/19 row.

The missing-gate counts are the sharper insight:

- clean predicted intervention is missing in 19/19 rows;
- null/locality cleanliness is missing in 19/19 rows;
- robustness and side-effect clearance are missing in 19/19 rows;
- transfer or widening is missing in 19/19 rows;
- control-surviving signature is missing in 18/19 rows;
- local internal path is missing in 18/19 rows;
- behavior substrate is missing in 11/19 rows.

That is not a complaint that the project is too strict. It is the current
control-surface genome in reliability coordinates. A future experiment only
improves the map if it moves a row to a stronger reliability class or makes a
failure class sharper.

## What We Have Accomplished

We have accomplished:

1. A strict project doctrine: every mechanism claim must pass signature,
   intervention, and reliability gates.
2. Multiple behavior substrates across different task families.
3. Multiple real behavior-control surfaces.
4. A failed-mechanism closeout for Qwen3-0.6B truth/agreement steering.
5. A clear map of output-margin and prompt-format confounds.
6. A strong synthetic associative lookup internal control surface on
   Qwen3-1.7B.
7. A narrowed MC005 localization from broad full-path dependence to all-head
   layers 24-26.
8. Exact lookup mediation by layers-24-26 final-query attention writes.
9. A mapped MC005 reliability boundary: rare answer-absent null flips under
   write replacement.
10. A repaired MC006 generated-answer behavior substrate for capital facts in
    V10.
11. A clear MC006 V11 failure: hidden separation exists but is
    requested-mode/output/null-confounded.
12. A clear MC006 V12 failure: matched-surface hidden separation exists but is
    weak-holdout/candidate-score/output/null-confounded.
13. A clear MC006 V13 near miss: generated matched-surface labels fix holdout
    balance but fail binary volume by one row.
14. A V14 behavior-substrate pass: the V13 miss was repaired by a narrow
    accent-normalized strict parser.
15. A clear MC006 V15 failure: the V14 matched generated table has a perfect
    hidden separator, but candidate-score and next-token output margins match
    it.
16. A narrow MC006 V16 lead-time result: an `after_mapping_line/layer_4`
    hidden signal beats same-position output margin and null controls, but
    final candidate-score and final output margins still match it.
17. A clear MC006 V17 causal negative: additive residual steering on the V16
    early direction failed holdout margin, generated-label, and control-gap
    gates.
18. A clear MC006 V18 table-geometry negative: the V16 lead-time signal
    reproduced, but V14 holdout true/override rows had no candidate-score or
    final-output margin overlap, and residualized hidden holdout AUC fell to
    0.333.
19. A clear MC006 V19 near miss: an expanded 400-row prompt bank selected
    `separate_task_weak` with 35/40 binary rows and 4/4 holdout labels, but
    strict candidate and final-output margin overlap still failed.
20. A clear MC006 V20 upper-bound negative: even the pooled V19 binary bank has
    no strict final-output overlap.
21. A clear MC006 V21 approximate-matching negative: V19/V20 near-pairs exist,
    but candidate-score and final-output margins still perfectly order holdout
    pairs, while the selected pre-output hidden direction reverses on holdout.
22. A clear MC006 V22 source/path lead-time map: queried-country and later
    pre-output line-boundary monitors exist, but final candidate-score and
    final-output margins still perfectly order holdout pairs.
23. A clear MC006 V23 output-interface diagnostic: final next-token margin sign
    predicts every V18/V19 binary generated label, so strict final-margin
    overlap is malformed as the default gate for the current greedy interface.
24. A clear MC006 V24 answer-interface diagnostic: delayed JSON city generation
    breaks the V23 first-token sign barrier, but JSON-completion candidate
    scoring still beats the selected hidden monitor on holdout.
25. A clear MC006 V25 candidate-decoupled negative: the V24 bank contains a
    behavior-passing delayed-city template where JSON-completion candidate
    scoring is not perfect, but the hidden monitor fails shuffled-label
    selected-search nulls.
26. A clear MC006 V26 locked-coordinate transfer negative: the V25 coordinate
    reproduces weakly on its source template but fails to transfer to the only
    other candidate-decoupled delayed-city template in the V24 bank.
27. A clear MC006 V27 expanded-bank negative: a 640-row delayed-city bank found
    four candidate-decoupled templates and enough pooled holdout volume, but
    failed the predeclared transfer-role coverage gate.
28. A clear MC006 V28 transfer-role repair negative: a transfer-only 640-row
    repair grid found one candidate-decoupled transfer template, but failed the
    predeclared requirement of at least two transfer-ready templates and 160
    combined pooled binary rows.
29. A formal MC006 delayed-city closeout: the V14-V28 route is now classified
    as `DELAYED_CITY_ROUTE_CLOSED_MONITOR_ONLY`, preserving the monitor and
    confound map while blocking further ordinary repairs as promotion routes.
    This is now backed by `code/mc006_predecision_frontier_closeout_audit.py`,
    which validates the behavior substrate, predecision monitors,
    output/candidate blockers, steering failure, shuffle null failure, and
    transfer-bank insufficiency before returning `monitor_only_closed`.
    Generated audit layer:
    `code/mc006_predecision_frontier_audit.py` writes
    `data/mc006_predecision_frontier_audit.json` and
    `research/cards/MC006_PREDECISION_FRONTIER_AUDIT.md`. It turns the
    delayed-city closeout into a validator-backed knowledge-frontier boundary:
    MC006 is a `diagnostic_note` with route status `monitor_only_closed`,
    frontier class `predecision_monitor_no_lever`, terminal stage
    `signature_monitor_no_lever`, reliability class
    `not_reliable_monitor_only_no_lever`, and transfer class
    `transfer_failed_or_bank_insufficient`. It validates the positive timing
    result and the failure together: V14 has 30 selected binary rows with both
    labels on holdout, V16 has an `after_mapping_line/layer_4` monitor with
    1.0 holdout AUC that beats same-position output, but that monitor does not
    beat candidate score or final next-token output, V17 steering fails, V25
    fails shuffled-label selected-search nulls, and V26-V28 close the transfer
    bank. The allowed claim is now decision timing; the forbidden claim is a
    knowledge vector, steering vector, or mechanism card.
30. A formal MC007 first-bridge closeout: the V1-V4 route is now classified as
    `MC007_ROUTE_CLOSED_DIAGNOSTIC_BRIDGE`, preserving V1 as the source-value
    baseline and V2 as the prior-pressure baseline while blocking ordinary
    prompt-only repairs as promotion routes.
31. A formal MC005 bounded closeout: the V29-V31 write-replacement route is now
    classified as `MC005_BOUNDED_ATTENTION_WRITE_MEDIATION`, preserving exact
    high-margin lookup mediation while making rare answer-absent null flips the
    explicit boundary. This is now backed by
    `code/mc005_write_replacement_closeout_audit.py`, which validates V27-V31,
    compares the write route against the failed additive-residual/donor
    intervention family, and emits a bounded/frozen route verdict.
32. A closed MC008 symbolic fact-code arbitration diagnostic: V1 showed compact
    generated element-code answers repair synthetic lookup and real-symbol
    controls but fail nulls and conflict balance; V2 repaired the nulls while
    preserving controls, but conflict rows stayed table-dominant, closing the
    first symbolic bridge route before hidden-state work.
33. A preregistered MC009 derived-code bridge that removes direct
    `entity -> artificial code` source-value binding and tests whether a
    prompt-local code derived from row position can create a cleaner
    task-versus-memory conflict substrate.
34. A runnable MC009 behavior harness plus smoke-boundary result: structural
    audits pass, `membership_authority_split` is the best initial template,
    and a 10-source smoke has clean synthetic ordinal lookup, real-symbol
    memory, and answer-absent null controls. The bridge is still blocked
    because primary conflict parseability is 47/60 and real/lure conflict
    balance fails.
35. A closed MC009 derived-code diagnostic: the non-default `typed_slot_v2`
    repair created conflict balance, but only by breaking synthetic and null
    controls and exposing the answer channel in the prompt. The first
    derived-code bridge route is therefore closed before hidden-state work.
36. A closed MC010 two-hop fact-code bridge diagnostic: two-hop indirection
    removed the obvious direct entity-to-code row and row-position answer slot,
    but the full generated table still failed synthetic lookup, real-symbol
    memory control, and real/lure conflict balance before hidden-state work.
37. A closed MC011 numeric answer-interface bridge diagnostic: same-format
    integer answers repaired the direct controls, with 40/40 synthetic lookup,
    40/40 familiar lookup, 40/40 atomic-number recall, and 40/40 null rows, but
    primary conflict collapsed to 240/240 prompt-local lab numbers and 0
    atomic/lure-number rows.
38. A behavior-ready but prompt-visible MC012 bridge diagnostic: explicit
    trusted/untrusted source-status labels produced the first clean mixed
    local-versus-learned numeric behavior table, with 40 local conflict rows,
    39 atomic/lure rows, one other row, clean direct controls, clean nulls, and
    source-disjoint balance. Because the source-status channel is visible by
    construction, MC012 is not signature-ready and no hidden-state work is
    allowed from it.
39. A closed MC013 status-channel ablation diagnostic: the statused positive
    control reproduced MC012 with 40/40 trusted local rows and 39/40 untrusted
    atomic rows, but text-identical matched ablation collapsed to 80/80 local
    rows and 0 atomic/lure rows. The simple "remove the source-status text"
    repair route is therefore killed before hidden-state work.
40. A closed MC014 inferred-reliability diagnostic: explicit source-status
    labels were removed and the model had to infer source validity from
    calibration rows. Direct controls and nulls stayed 40/40 clean and primary
    prompts had no status lexemes, but calibration-inconsistent rows still
    selected local numbers on 40/40 rows, collapsing the primary conflict to
    80/80 local and 0 atomic/lure rows.
41. A closed MC015 parity-gated diagnostic: explicit source-status labels were
    absent, target and lure atomic numbers were hidden, and the expected local
    versus atomic labels were balanced by split. Direct controls and nulls
    stayed 40/40 clean, and the primary conflict produced mixed outputs, but
    the learned parity gate was not followed: expected correctness was 39/80,
    expected-local rows selected local only 27/40, and expected-atomic rows
    selected atomic only 12/40.
42. A closed MC016 alphabet-gated diagnostic: explicit source-status labels
    were absent, target and lure atomic numbers were hidden, and the expected
    local versus atomic labels were balanced by split. Direct controls and
    nulls stayed 40/40 clean, but the selected `alphabet_rule` conflict
    collapsed to 80/80 local-number rows and 0 atomic/lure rows; expected-local
    rows selected local 40/40 and expected-atomic rows selected atomic 0/40.
    The stronger `feature_labeled_alphabet` template also failed to rescue the
    atomic side, producing 61 local rows, 0 atomic rows, and 19 unparsed rows.
43. A smoke-stage MC017/MC018 answer-interface diagnostic: replacing numeric
    answers with `LOCAL`/`ATOMIC` source tokens did not rescue MC016, because
    MC017 collapsed to `LOCAL` even on atomic-only controls. Replacing those
    words with neutral counterbalanced `A`/`B` labels partially repaired the
    atomic control but still failed the behavior gate: MC018's expanded
    10-source smoke had 160/160 parseable primary conflict rows and balanced
    expected sources/choices, but only 84/160 source-rule-correct rows, a
    122/160 first-listed-choice rate, and a 102/160 local-source selection
    rate. The new diagnostic is not an atlas row yet, but it names the next
    failure class: option/definition-order and local-source salience can mimic
    or overwhelm source-rule following.
44. A smoke-stage MC019 row-code numeric diagnostic: replacing source-label
    answers with neutral route codes and integer answers repaired the direct
    controls and nulls but not the route conflict. The selected 10-source smoke
    had 20/20 synthetic local lookup, 20/20 familiar local lookup, 20/20
    real-world atomic control, 20/20 route-rule-absent UNKNOWN, and 20/20
    answer-absent UNKNOWN, while primary route-code conflict stayed at 21/40
    expected-correct. Repeating the queried row improved conflict correctness
    to 27/40 and expected-local rows to 19/20, but expected-atomic rows stayed
    8/20 and the null slipped to 17/20. The new diagnostic is not an atlas row
    yet, but it shows that neutral row-local codes are not enough to make
    learned atomic answers survive prompt-local table pressure.
45. A smoke-stage MC020 atomic table-pressure diagnostic: direct atomic recall
    survived local-table pressure, so MC019's failure is not ordinary recall
    suppression. The selected 10-source smoke had 20/20 no-table atomic recall,
    18/20 distractor-table atomic recall, 18/20 query-row atomic recall, and
    18/20 repeated-query-row atomic recall. But route-local reached 17/20 local
    while route-atomic collapsed to 15/20 local and only 2/20 atomic. The new
    diagnostic is not an atlas row yet, but it isolates the failure to
    asymmetric conditional source arbitration.
46. A smoke-stage MC021 visible-versus-learned arbitration diagnostic:
    conditional routing was weak even when both branches were prompt-visible,
    and learned-memory branches were weaker. The selected 10-source smoke
    passed direct controls and nulls: local 20/20, visible reference 20/20,
    atomic 18/20, and answer-absent UNKNOWN 18/20. But visible-visible route
    conflicts were only 29/40 expected-correct, and visible-learned route
    conflicts were 20/40 expected-correct. The new diagnostic is not an atlas
    row yet, but it shows route-code arbitration itself is not a clean behavior
    substrate.
47. A smoke-stage MC022 explicit branch-name arbitration diagnostic: replacing
    opaque route codes with semantic `LOCAL`/`REFERENCE`/`ATOMIC` source labels
    did not create a clean substrate. The selected 10-source smoke passed
    direct controls and nulls: local 20/20, visible reference 20/20, atomic
    18/20, and answer-absent UNKNOWN 19/20. But visible conflicts were only
    30/40 expected-correct, learned conflicts were only 24/40, and ATOMIC rows
    selected learned atomic numbers only 5/20. The rule-order split sharpened
    the diagnosis: putting the nonlocal branch first rescued visible-visible
    rows, but only partly rescued learned-memory rows. The new diagnostic is
    not an atlas row yet, but it shows that explicit semantic branch names
    still leave local-source salience and learned-memory arbitration as
    behavior-substrate blockers.
48. A smoke-stage MC023 query-operation arbitration diagnostic: removing
    row-level source labels and using counterbalanced query operation handles
    preserved direct controls and nulls but still did not clear the conflict
    gate. The selected 10-source smoke had 40/40 synthetic local lookup, 40/40
    familiar local lookup, 40/40 direct atomic control, 40/40
    operation-rule-absent UNKNOWN, and 40/40 answer-absent UNKNOWN. But
    operation-local conflict selected local only 31/40, and operation-atomic
    conflict selected atomic only 28/40. The new diagnostic is not an atlas row
    yet, but it shows that query-level operation handles are not enough unless
    both local and learned branches cross behavior-gate thresholds.
49. A full-source MC024 few-shot query-operation arbitration diagnostic:
    adding balanced worked examples to query-level operation handles created a
    useful smoke-to-full failure. The 10-source smoke passed all controls and
    both conflict thresholds, but the full 40-source run blocked promotion:
    synthetic lookup, familiar lookup, atomic control, operation-rule-absent
    UNKNOWN, and answer-absent UNKNOWN were all 160/160; operation-local
    conflict selected local 159/160; operation-atomic conflict selected atomic
    only 130/160 and leaked 27/160 other-number answers. The diagnostic is not
    an atlas row, but it shows that worked examples repair prompt-local routing
    more readily than learned atomic routing.
50. A smoke-stage MC025 constrained-choice operation arbitration diagnostic:
    constraining the answer to A/B/C options did not repair MC024. The
    structural gate passed with balanced expected choices and prompt-visible
    atomic-number options by design. The 10-source smoke preserved synthetic
    and familiar local lookup at 120/120 each and operation-local conflict at
    109/120, but direct atomic control selected atomic only 44/120,
    answer-absent nulls selected UNKNOWN only 83/120, and operation-atomic
    conflict selected atomic only 18/120. The diagnostic is not an atlas row;
    it shows that prompt-visible choice constraints introduce their own
    control/null failure surface instead of neutrally repairing learned routing.
51. A smoke-stage MC026 numeric-option operation arbitration diagnostic:
    replacing A/B/C choices with literal numeric options did not repair MC025.
    The structural gate passed with balanced first options, no status lexemes,
    and prompt-visible atomic-number options by design. The 10-source smoke
    preserved synthetic and familiar option lookup at 120/120 each, repaired
    answer-absent and operation-rule-absent UNKNOWN nulls to 120/120 each, and
    kept operation-local conflict at 111/120 local. But direct atomic control
    selected atomic 0/120, selected UNKNOWN 116/120, and left 4/120 unparsed;
    operation-atomic conflict selected atomic only 19/120. The diagnostic is
    not an atlas row; it shows that numeric option lists can preserve nulls
    while converting learned recall into abstention, so prompt-visible option
    lists are not neutral output scaffolds.
52. A smoke-stage MC027 answer-interface sweep diagnostic:
    sweeping bare integer, `ANSWER=...`, JSON, A/B/C choice, and numeric-option
    interfaces on one operation substrate made the interface law measurable.
    The structural gate passed with 2,160 rows over 10 sources, balanced
    operation assignments, rule orders, option orders, split balance, and no
    status lexemes. The selected bare-integer interface cleared the 10-source
    smoke: familiar lookup 40/40 local, direct atomic control 40/40 atomic,
    answer-absent and rule-absent nulls 40/40 UNKNOWN, operation-local conflict
    39/40 local, and operation-atomic conflict 38/40 atomic. But this is still
    smoke-only and inherits MC024's full-source boundary. The diagnostic shows
    that structured answer schemas are not neutral: `ANSWER=` prefix, JSON,
    A/B/C choices, and numeric options each break controls, nulls, parseability,
    or learned routing in different ways.
53. A full-source MC028 bare-integer boundary diagnostic:
    carrying MC027's only smoke-surviving interface to all 40 sources did not
    promote the bridge. The structural gate passed with 960 rows, full
    source-disjoint coverage, balanced operation assignments and rule orders,
    clean null rows, and no status lexemes. Behavior kept the direct controls
    and nulls clean: familiar lookup 160/160 local, direct atomic control
    160/160 atomic, answer-absent and rule-absent nulls 160/160 UNKNOWN, and
    operation-local conflict 159/160 local. The learned branch still failed:
    operation-atomic conflict was 131/160 atomic, 28/160 other-number, and
    1/160 local. This closes the MC027 smoke survivor as a full-source boundary
    before hidden-state work and names the failure mode:
    `FULL_SOURCE_OTHER_NUMBER_LEAK`.
54. A full-source MC029 operation-leak factorial diagnostic:
    factorizing the MC028 leak across baseline numeric examples,
    label-examples-without-numbers, rules-only prompting, query-before-example
    ordering, and query-row-last salience did not repair the bridge, but it
    split the failure cleanly. The structural gate passed with 4,000 rows
    across all 40 sources. No variant passed the behavior gate. Baseline
    numeric examples reached only 0.713 operation-atomic atomic with 0.244
    other-number answers. `rules_only` reached 0.875 operation-atomic atomic
    and lowered other-number answers to 0.100, but answer-absent UNKNOWN fell
    to 0.806. `query_before_examples` preserved answer-absent nulls but
    collapsed operation-atomic atomic to 0.256 and amplified worked-example
    copying to 54 rows. `query_row_last` reduced other-number answers to 0.081
    but kept operation-atomic atomic at only 0.613. The diagnostic class is
    `FACTORIAL_BRANCH_NULL_TRADEOFF`: prompt factors can move branch, null, and
    example-copying errors separately without creating a valid substrate.
55. A full-source MC030 null-preserving rules-repair diagnostic:
    testing explicit absence guards on the MC029 rules-only prompt did not
    repair the bridge. The structural gate passed with 4,000 rows across all 40
    sources after removing numeric decision-list markers that leaked early
    atomic-number targets. No variant passed the behavior gate. The unguarded
    `rules_only_baseline` remained the best branch/null compromise at 0.875
    operation-atomic atomic and 0.806 answer-absent UNKNOWN. The row-absence
    guard kept operation-atomic atomic at 0.863 but dropped answer-absent
    UNKNOWN to 0.231. The decision-order guard after query kept operation-atomic
    atomic at 0.869 but answer-absent UNKNOWN was only 0.425. The guarded
    query-last variant lowered other-number answers to 0.056 but collapsed
    operation-atomic atomic to 0.487. The diagnostic class is
    `RULES_ONLY_BRANCH_NULL_TRADEOFF_PERSISTS`: simple absence guards are now a
    closed repair path for this substrate.
56. A first artifact-registry layer for the control-surface atlas: linked
    MC001-MC016 result artifacts are now parsed into normalized pass/fail,
    diagnostic, readiness, selected-coordinate, baseline/control, null, and
    intervention fields, and atlas validation fails direct row/artifact
    readiness contradictions plus 43 central family-level closure checks.
57. A compact machine-readable artifact index:
    `data/control_surface_artifact_index.json`, generated from the raw linked
    result artifacts and validated against them so downstream comparison tools
    do not have to reparse every nested result JSON. The registry now covers
    53 linked artifacts across all 19 atlas rows, including legacy MC001-MC004
    artifacts, and extracts metrics from all 53.
58. A generated cross-family comparison layer:
    `data/control_surface_comparison.json`, built by
    `code/control_surface_comparison.py` and validated against the atlas plus
    artifact index. The current comparison measures the project-level genome
    shape directly: 19 atlas rows, 0 promoted mechanism cards, 1 bounded
    mechanism card, 17 diagnostic notes, 1 failed mechanism-card route,
    0.947368 diagnostic-or-failed ratio, 0.315789 output-margin-confounded row
    ratio, 0.368421 behavior-substrate-failed row ratio, and 0.736842
    intervention not-allowed-or-failed ratio.
59. A generated claim-audit layer inside
    `data/control_surface_comparison.json`: all 19 atlas rows now have linked
    result artifacts, extracted metrics, and family-level claim checks. The
    current audit has all 19 rows at `family_checked_with_metrics`, 0 rows at
    `family_checked_partial_metrics`, three checked MC005 closure claims, and
    eight checked MC006 closure claims, plus granular MC007-MC016 bridge checks.
    The new checks split MC005 into layers-24-26 write mediation, nonzero null
    flips, and non-simple margin-cutoff null boundary; they split MC006 into
    V16 output/candidate confound, V17 additive-steering failure, V21
    pair-matching failure, V22 source/path shadow, V24 delayed-city monitor-only
    boundary, V25 candidate-decoupled shuffle overfit, V28
    one-template-not-bank transfer result, and the older combined
    shuffle/transfer failure. They also split the bridge routes into MC007 V1
    source lookup without conflict, V2 authority-dial parseability failure, V3
    parseability-repair failure, V4 source-declaration control failure, MC008
    direct-controls-before-null and null-repaired/conflict-absent boundaries,
    MC009 membership-control versus typed-slot-control tradeoffs, and MC010
    two-hop direct-control and table-dominant conflict failures, MC011
    numeric direct-control success and numeric conflict-collapse failure, and
    MC012 reliability-labeled contrast success plus prompt-channel blockage,
    and MC013 statused positive-control reproduction plus matched-ablation
    contrast collapse, MC014 direct-control cleanliness plus
    calibration-inference conflict collapse, and MC015 direct-control
    cleanliness plus parity-gate rule-following failure, and MC016
    direct-control cleanliness plus alphabet-gate local collapse.
60. A generated claim-consistency layer inside
    `data/control_surface_comparison.json`: the comparison now checks 120
    generic row-level consistency conditions across verdict class,
    intervention state, lead-time state, null-locality status, behavior-gate
    diagnostics, output-confound diagnostics, and signature-causality
    diagnostics. The current audit has 0 contradictions, and atlas validation
    fails if contradictions appear.
61. A generated smoke-diagnostic layer for MC017-MC033:
    `data/control_surface_smoke_diagnostics.json`, built by
    `code/control_surface_smoke_diagnostics.py` and documented in
    `research/26_CONTROL_SURFACE_SMOKE_DIAGNOSTICS.md`. It keeps the
    post-atlas bridge smoke runs out of the mechanism-card atlas while still
    making their typed failures auditable. The current layer covers 17 smoke
    cards, all 17 structural gates passed, 0 behavior-ready cards, 0
    signature-ready cards, 0 hidden-state-allowed cards, and 72 validation
    checks. Atlas validation now fails if this smoke ledger is stale or if its
    core failure assertions stop holding. MC026 adds the key option-interface
    diagnostic: numeric options preserve nulls but make direct atomic recall
    abstain and leave the learned atomic branch weak. MC027 adds the interface
    sweep: bare integer is the only 10-source smoke survivor, while structured
    schemas and option interfaces break different gates. MC028 closes that
    smoke survivor at full-source scale: controls, nulls, and local routing
    stay clean, but the learned atomic operation branch leaks other numbers.
    MC029 factorizes that leak and shows branch, null, local-row, and
    worked-example-copying errors move independently. MC030 shows that simple
    absence guards do not repair the resulting rules-only branch/null tradeoff.
    MC031 shows that a statusless checksum reliability cue preserves controls
    and nulls while invalid-checksum rows still collapse to local answers.
    MC032 shows that replacing checksum validity with cross-table consistency
    preserves controls and nulls but still fails the learned atomic branch.
    MC033 shows that replacing cross-table consistency with a row-local
    learned-fact claim keeps controls and nulls clean while match rows and
    mismatch rows both fail stable routing.
62. A generated bridge-ladder layer for MC010-MC033:
    `data/control_surface_bridge_ladder.json`, built by
    `code/control_surface_bridge_ladder.py` and documented in
    `research/27_CONTROL_SURFACE_BRIDGE_LADDER.md`. It combines validated
    MC010-MC016 atlas rows with MC017-MC033 smoke diagnostics without promoting
    smoke runs into atlas rows. The current ladder has 24 rungs, 7 atlas rungs,
    17 smoke rungs, 1 behavior-ready rung, 0 signature-ready rungs, 0
    hidden-state-allowed rungs, and 0 clean unconfounded bridge rungs. The
    generated claim is now explicit: the only clean local-versus-learned
    contrast in MC010-MC033 is MC012's prompt-visible status-label positive
    control; MC024 shows that worked examples repair local routing but not the
    learned atomic branch, MC025 shows that constrained choices introduce their
    own atomic-control and null failures, and MC026 shows that numeric options
    preserve nulls by pushing direct atomic recall into UNKNOWN, while MC027
    shows that answer schemas themselves are behavior surfaces, MC028 shows
    the bare-integer survivor still fails full-source learned-branch routing,
    and MC029 shows that factorized prompt edits move failure axes without
    producing a clean bridge, while MC030 closes simple absence-guard repair,
    MC031 closes the first statusless checksum reliability cue as local
    collapse on invalid-source rows, and MC032 shows that cross-table
    consistency repeats the statusless local-dominance boundary, and MC033
    closes the same-family fact-claim route as branch instability rather than
    clean learned-fact arbitration.
63. A generated mixture-law layer:
    `data/control_surface_mixture_law.json`, built by
    `code/control_surface_mixture_law.py` and documented in
    `research/28_CONTROL_SURFACE_MIXTURE_LAW.md`. It turns the atlas into a
    measured distribution of where behavior currently lives. The current
    primary blockers are 11 behavior-substrate-or-bridge-blocked rows, 6
    output-geometry-shadow rows, 1 prompt-visible positive control, and 1
    bounded internal-causal row. The current pressure classes are explicit:
    prompt-contract-visible in 19/19 rows, output-geometry-visible in 14/19,
    source-or-prompt-token-dependent in 13/19, internal-monitor-present in
    5/19, internal-causal in 1/19, and clean unconfounded bridge rungs in
    0/19.
64. A generated decision-frontier layer:
    `data/control_surface_decision_frontier.json`, built by
    `code/control_surface_decision_frontier.py` and documented in
    `research/29_CONTROL_SURFACE_DECISION_FRONTIER.md`. It makes lead-time a
    first-class measured axis. The current frontier map has 12 rows where the
    timing frontier is not reached, 4 rows where the decision is output-visible
    at or before the frontier, 2 predecision monitor-only rows, 1 bounded
    causal surface that is not a timing-frontier result, and 0 predecision
    causal candidates. The monitor-only rows are MC004 and MC006.
65. A generated route-disposition layer:
    `data/control_surface_route_disposition.json`, built by
    `code/control_surface_route_disposition.py` and documented in
    `research/30_CONTROL_SURFACE_ROUTE_DISPOSITION.md`. It is the current
    claim-killing ledger. The atlas dispositions are 11 closed-before-hidden
    routes, 3 output-shadow diagnostic baselines, 1 failed
    intervention/mechanism route, 1 monitor-only closed route, 1 conditional
    monitor-only revisit route, 1 prompt-visible positive control, and 1
    bounded frozen mechanism. It records 0 hidden-state-ready atlas routes and
    0 bridge rungs that allow hidden-state work; bridge dispositions are 23
    rungs closed before hidden-state work and 1 prompt-visible positive
    control.
66. A generated transfer matrix:
    `data/control_surface_transfer_matrix.json`, built by
    `code/control_surface_transfer_matrix.py` and documented in
    `research/31_CONTROL_SURFACE_TRANSFER_MATRIX.md`. It makes widening and
    transfer claims explicit. The current atlas has 0 transfer-ready
    mechanisms, 14 untested transfer rows, 2 low-transfer rows, 2
    medium-transfer rows, and 1 failed transfer row. The matrix classifies
    MC005 as a bounded transfer-fragile reference specimen and MC006 as a
    failed or bank-insufficient transfer route.
67. A generated reliability matrix:
    `data/control_surface_reliability_matrix.json`, built by
    `code/control_surface_reliability_matrix.py` and documented in
    `research/32_CONTROL_SURFACE_RELIABILITY_MATRIX.md`. It makes the promotion
    gate executable: 0 full-reliability mechanisms, 1 bounded reliability
    reference, 11 behavior/bridge-blocked rows, 3 output-shadow diagnostics, 2
    monitor-only no-lever rows, 1 failed-intervention route, and 1
    prompt-visible positive control.
68. A generated error-taxonomy layer:
    `data/control_surface_error_taxonomy.json`, built by
    `code/control_surface_error_taxonomy.py` and documented in
    `research/33_CONTROL_SURFACE_ERROR_TAXONOMY.md`. It makes killed bridge
    attempts first-class data rather than prose-only tombstones. The current
    taxonomy covers 17 smoke cards, 24 bridge rungs, 0 hidden-state-allowed
    smoke cards, and validates MC028's full-source failure as a learned-branch
    other-number leak: 28/160 operation-atomic rows were wrong numbers while
    direct atomic control, answer-absent nulls, and rule-absent nulls stayed
    160/160 clean. The wrong-number buckets are 9 worked-example outputs, 8
    other bank atomic numbers, 6 off-bank other numbers, 3 bank local numbers,
    and 2 prompt-local row numbers. This sharpens the next experiment pressure:
    manipulate worked examples, query anchoring, and prompt-row salience
    separately before any hidden-state probe. It also validates MC029 as a
    factorized branch/null/example-leak tradeoff: `rules_only` improves
    operation-atomic atomic to 0.875 while answer-absent UNKNOWN falls to
    0.806, and `query_before_examples` preserves nulls while amplifying
    worked-example copying to 54 rows. It also validates MC030 as a failed
    absence-guard repair: row-absence and decision-order guards worsen nulls,
    and query-last reduces other-number leakage only by collapsing the learned
    atomic branch. It also validates MC031 as a statusless reliability-cue
    failure: direct controls and nulls are clean, but invalid-checksum rows
    select local numbers 1.000 of the time and atomic/lure values 0.000 of the
    time. It also validates MC032 as a statusless cross-table failure: direct
    controls and nulls stay clean, mismatch rows select atomic/lure values
    0.000 of the time, and side-number copying remains 0.000. It also
    validates MC033 as a fact-claim closeout failure: direct controls and nulls
    stay clean, match rows often return learned atomic numbers when local is
    expected, and mismatch rows split between local answers and the wrong
    claimed number instead of selecting the learned atomic answer.
69. A generated gate-geometry layer:
    `data/control_surface_gate_geometry.json`, built by
    `code/control_surface_gate_geometry.py` and documented in
    `research/34_CONTROL_SURFACE_GATE_GEOMETRY.md`. It turns the mechanism-card
    bar itself into a measured artifact. The current row funnel is 11/19
    behavior-or-bridge substrate closures, 1/19 prompt-channel locality
    closure, 3/19 output-geometry signature shadows, 2/19 monitor-only
    no-lever rows, 1/19 failed intervention route, 1/19 bounded reliability
    specimen, and 0/19 promoted mechanisms. The bridge funnel is 23/24
    behavior-substrate closures and 1/24 prompt-visible positive control. This
    directly encodes the reviewer insight that killed controls are structured
    evidence about the bar's shape, not just tombstones.
70. A generated genome-snapshot layer:
    `data/control_surface_genome_snapshot.json`, built by
    `code/control_surface_genome_snapshot.py` and documented in
    `research/35_CONTROL_SURFACE_GENOME_SNAPSHOT.md`. It fuses the atlas,
    artifact index, comparison, law audit, next queue, smoke diagnostics,
    bridge ladder, mixture law, decision frontier, route disposition, transfer
    matrix, reliability matrix, error taxonomy, and gate geometry into one
    compact current-state object. It records the current global claim state:
    19 atlas rows, 53 linked result artifacts, 1 bounded mechanism card, 18
    diagnostic-or-failed rows, 0 promoted mechanisms, 12 rows blocked before
    signature work, 5 signature-stage blocks, 1 failed intervention route, 1
    bounded reliability specimen, 0 transfer-ready mechanisms, 0
    hidden-state-allowed bridge rungs, and MC030-MC033 as the current bridge
    closure set.
71. A generated axis-interactions layer:
    `data/control_surface_axis_interactions.json`, built by
    `code/control_surface_axis_interactions.py` and documented in
    `research/36_CONTROL_SURFACE_AXIS_INTERACTIONS.md`. It turns the genome
    snapshot into a small-n-aware predictive map: 123 feature summaries, 14
    pure predictive rules, 10 broad-or-supported pure rules, and 31 mixed
    predictors. The strongest broad rule is not a hidden mechanism rule:
    `closed_before_hidden_state` predicts
    `pre_signature_behavior_substrate` across 11 rows. Output geometry is
    broad but mixed, spanning output-shadow signatures, monitor-only rows, and
    the failed-intervention route rather than one terminal stage.
72. A generated coverage-gaps layer:
    `data/control_surface_coverage_gaps.json`, built by
    `code/control_surface_coverage_gaps.py` and documented in
    `research/37_CONTROL_SURFACE_COVERAGE_GAPS.md`. It makes the map's
    negative space explicit: 12 named gaps, 4 critical gaps, 14
    transfer-untested rows, 0 hidden-state-allowed bridge rungs, 0 clean
    unconfounded bridge rungs, and 82 singleton-or-sparse feature summaries.
    The critical gaps are no promoted mechanism, no full reliability, no clean
    predicted intervention, and no transfer-ready mechanism.
73. A generated gap-closure-plan layer:
    `data/control_surface_gap_closure_plan.json`, built by
    `code/control_surface_gap_closure_plan.py` and documented in
    `research/38_CONTROL_SURFACE_GAP_CLOSURE_PLAN.md`. It turns the coverage
    gaps and next queue into 6 decision-bound work orders covering 12/12
    current gaps, 4/4 critical gaps, and 5/5 top queue items. Each work order
    carries promotion, bounded-claim, kill, containment, and export rules, so
    future branches have to close, bound, die, or become a reusable diagnostic
    instead of surviving as attractive but unresolved stories.
74. A generated offensive-doctrine harness:
    `data/control_surface_offensive_doctrine.json`, built by
    `code/control_surface_offensive_doctrine.py` and documented in
    `research/39_CONTROL_SURFACE_OFFENSIVE_DOCTRINE.md`. It converts the six
    gap-closure work orders into branch-intake contracts. Each future branch
    must name target gaps, expected generated-layer movement, minimum evidence,
    promotion, bound, kill, containment, and export rules, and one of four
    closeout classes before it can count as progress. The atlas validator now
    fails if this harness is missing or stale, so future insight has to move
    map geometry or kill a claim class rather than remain reviewer prose.
75. A generated transfer-width probe packet:
    `data/transfer_width_probe_mc005_mc003_mc004.json`, built by
    `code/transfer_width_probe_mc005_mc003_mc004.py` and documented in
    `research/prereg/TRANSFER_WIDTH_PROBE_MC005_MC003_MC004.md`. It is the
    first branch contract emitted under the offensive doctrine. It targets the
    immediate `run_width_transfer_probe` work order, uses MC005 as the bounded
    reference surface, MC003 as the output-shadow diagnostic baseline, and
    MC004 as the predecision-monitor diagnostic baseline. It predeclares Gemma
    as the first non-Qwen target family and requires six panels before any
    transfer language can improve: primary lookup effect, answer-absent null
    locality, side rows, prompt robustness, delayed-copy output-shadow
    baseline, and in-context-binding monitor-only baseline. It claims no
    transfer result; it makes transfer failure or success measurable.
76. A generated singleton-stage replication pack:
    `data/singleton_stage_replication_pack.json`, built by
    `code/singleton_stage_replication_pack.py` and documented in
    `research/prereg/SINGLETON_STAGE_REPLICATION_PACK.md`. It targets the
    high-priority law-replication work order and the three singleton terminal
    stages currently blocking stronger predictive laws:
    `intervention_failed`, `pre_signature_prompt_channel_locality`, and
    `reliability_null_boundary`. It anchors those stages to MC001G, MC012, and
    MC005, respectively, then predeclares two materially distinct replication
    proposals per stage. It claims no new law; it makes the next law update
    conditional on new rows landing in the same terminal stage rather than on
    attractive singleton purity.
77. A preregistered MC031 statusless reliability bridge:
    `research/prereg/MC031_STATUSLESS_RELIABILITY_BRIDGE.md`, implemented in
    `code/mc031_statusless_reliability_bridge.py`, with smoke status in
    `research/cards/MC031_STATUSLESS_RELIABILITY_BRIDGE_STATUS.md`. Its full
    structural gate passes on 840 records across 40 sources: source-disjoint
    splits, balanced primary labels, hidden atomic/lure numbers in conflict
    prompts, clean answer-absent null prompts, parseable candidates, no
    candidate collisions, one answer suffix, and no trusted/untrusted/reliable
    source-status lexemes in primary prompts. A 10-source model smoke then
    exposes the likely failure mode: direct controls and nulls pass, valid
    checksum rows select local numbers, but invalid-checksum rows also select
    local numbers 10/10 times. This is not a full-source behavior verdict and
    not an atlas row; it is a concrete continuation of the bridge-substrate
    closure plan and keeps hidden-state work forbidden on this route.
78. A preregistered and smoke-tested MC032 post-checksum cross-table bridge:
    `research/prereg/MC032_POST_CHECKSUM_BRIDGE.md`, implemented in
    `code/mc032_post_checksum_bridge.py`, with status in
    `research/cards/MC032_POST_CHECKSUM_BRIDGE_STATUS.md`. Its structural gate
    passes on 840 records across 40 sources. The 10-source model smoke keeps
    direct local lookup, familiar local lookup, direct atomic recall, and
    answer-absent nulls clean, but the mismatch branch selects atomic/lure
    numbers 0/10 times and mostly collapses to the primary local row. The side
    number from the second table is never selected. This converts MC031 from a
    possible checksum-specific failure into a broader statusless source-validity
    local-dominance result.
79. A preregistered and smoke-tested MC033 fact-claim bridge closeout:
    `research/prereg/MC033_FACT_CLAIM_BRIDGE_CLOSEOUT.md`, implemented in
    `code/mc033_fact_claim_bridge_closeout.py`, with status in
    `research/cards/MC033_FACT_CLAIM_BRIDGE_CLOSEOUT_STATUS.md`. Its structural
    gate passes on 840 records across 40 sources. The 10-source model smoke
    keeps synthetic lookup, familiar lookup, direct atomic recall, answer-absent
    nulls, and fact-claim-absent local rows clean, but the fact-claim rule fails
    on both branches: match rows return local only 4/10 times and learned atomic
    5/10 times, while mismatch rows return learned atomic only 1/10 times,
    local 5/10 times, and the wrong claimed/lure number 4/10 times. This closes
    the same-family bridge route after MC033 unless a materially new substrate
    class is preregistered.
80. A generated post-MC033 bridge-substrate closeout audit:
    `code/post_mc033_bridge_closeout_audit.py` writes
    `data/post_mc033_bridge_closeout_audit.json` and
    `research/cards/POST_MC033_BRIDGE_SUBSTRATE_CLOSEOUT_STATUS.md`. It turns
    the reviewer insight about typed failures into a validator-backed research
    boundary: MC031-MC033 are recorded as a same-family statusless
    source-validity closure sequence, MC030-MC033 remain the current closed
    bridge set, the bridge ladder still has 24 rungs and 17 smoke rungs, and
    there are still 0 hidden-state-allowed bridge rungs and 0 clean
    unconfounded bridge candidates. The audit validates the decisive failure
    metrics directly: MC031 invalid-checksum rows select local 1.000 and
    atomic/lure 0.000, MC032 mismatch rows select atomic/lure 0.000 with no
    side-number copying, and MC033 fact-claim rows fail both match and mismatch
    routing. The allowed claim is now precise: this is a diagnostic family
    boundary showing that clean direct controls and nulls can coexist with
    failed learned/local bridge routing. The forbidden claim is equally
    explicit: no hidden signature, intervention, mechanism card, or general
    knowledge-control surface is licensed by this closeout.
81. A generated MC005 reference-specimen audit:
    `code/mc005_reference_specimen_audit.py` writes
    `data/mc005_reference_specimen_audit.json` and
    `research/cards/MC005_REFERENCE_SPECIMEN_AUDIT.md`. It freezes MC005 as the
    atlas' calibrated bounded positive control: verdict
    `bounded_mechanism_card`, route status `bounded_frozen_not_promoted`,
    terminal stage `reliability_null_boundary`, reliability class
    `bounded_reliability_reference`, and transfer class
    `bounded_transfer_fragile_reference`. The audit validates the positive
    surface directly: V29 attention-write replacement has 1.0 delta recovery
    and 1.0 target-win-loss recovery versus direct source masking, with source
    controls passing. It also validates the boundary directly: strict
    answer-absent nulls are not clean, V31 has 5 combined null flips, and the
    strict 0.5 absolute-margin explanation fails. This matters because it keeps
    the project honest about the strongest result: MC005 proves that a local
    internal causal surface can be real and still not be reliable enough for
    full mechanism-card promotion.
82. A generated MC006 predecision-frontier audit:
    `code/mc006_predecision_frontier_audit.py` writes
    `data/mc006_predecision_frontier_audit.json` and
    `research/cards/MC006_PREDECISION_FRONTIER_AUDIT.md`. It freezes MC006 as
    a knowledge-like monitor-only route rather than a mechanism card: verdict
    `diagnostic_note`, route status `monitor_only_closed`, frontier class
    `predecision_monitor_no_lever`, reliability class
    `not_reliable_monitor_only_no_lever`, and transfer class
    `transfer_failed_or_bank_insufficient`. The audit validates the real
    positive result: V14 provides a matched generated behavior substrate and
    V16 provides an `after_mapping_line` layer-4 monitor with 1.0 holdout AUC
    that beats same-position output. It also validates the boundary:
    candidate/final-output geometry still blocks promotion, V17 steering fails,
    V25 fails shuffled-label selected-search nulls, V26 fails locked-coordinate
    transfer, and V27-V28 fail to build a transfer-ready bank.
83. A generated compositional-genome audit:
    `code/control_surface_compositional_genome_audit.py` writes
    `data/control_surface_compositional_genome_audit.json` and
    `research/40_CONTROL_SURFACE_COMPOSITIONAL_GENOME_AUDIT.md`. It turns the
    strongest reviewer insight into a validator-backed artifact: the current
    genome-level object is the measured distribution across prompt contracts,
    output geometry, source/prompt-token dependence, behavior/bridge substrate
    blockage, internal monitors, bounded causal surfaces, reliability, and
    transfer. It validates 19/19 prompt-contract-visible rows, 14/19
    output-geometry-visible rows, 13/19 source-or-prompt-token-dependent rows,
    11/19 behavior-or-bridge-substrate-blocked rows, 5/19 internal-monitor
    rows, 1/19 internal-causal row, 0/19 promoted mechanisms, 0/19
    full-reliability mechanisms, 0/19 transfer-ready mechanisms, and 0 clean
    unconfounded bridge substrates. It anchors the mixture to MC005 as the
    bounded reference, MC006 as the knowledge-like monitor-only boundary, the
    post-MC033 bridge closeout as the substrate death condition, and MC004/MC006
    as the current monitor-only lead-time frontier.
84. A generated control-surface family matrix:
    `code/control_surface_family_matrix.py` writes
    `data/control_surface_family_matrix.json` and
    `research/41_CONTROL_SURFACE_FAMILY_MATRIX.md`. It is the compact
    one-row-per-family table the reviewer asked for in operational form:
    row id, behavior domain, models, behavior gate, verdict, primary blocker,
    terminal stage, route disposition, frontier class, lead-time state,
    intervention state, reliability class, transfer class/value, prompt/output/
    source/internal/null/transfer axes, failed gates, claim-bar action, and next
    decision. It validates that all 19 atlas rows are joined exactly once,
    that the joined counts match the mixture law, decision frontier,
    reliability matrix, transfer matrix, and gate geometry, that MC005 remains
    the only bounded reference, that MC004/MC006 remain the only monitor-only
    rows, and that there are still 0 promotion-ready rows.
85. A generated knowledge-ladder coverage map:
    `code/control_surface_knowledge_ladder.py` writes
    `data/control_surface_knowledge_ladder.json` and
    `research/42_CONTROL_SURFACE_KNOWLEDGE_LADDER.md`. It separates the
    knowledge-specific ladder from auxiliary diagnostics: level 1 synthetic
    lookup is MC005 and is bounded, level 2 semi-synthetic familiar-entity
    lookup is MC007 and behavior-blocked, level 3 symbolic/learned-memory
    bridge rows are MC008-MC016 and remain hidden-state-disallowed alongside the
    24-rung bridge ladder, level 4 parametric-fact override is MC006 and
    monitor-only, and level 5 real abstention/uncertainty is MC002/MC002B and
    not mechanism-ready. It validates that the ladder rows plus auxiliary rows
    partition all 19 family-matrix rows, that there is 1 bounded-reference
    level, 1 monitor-only level, 0 promoted levels, and 0 real
    abstention/uncertainty mechanism-ready levels.
86. A generated knowledge-gap plan:
    `code/control_surface_knowledge_gap_plan.py` writes
    `data/control_surface_knowledge_gap_plan.json` and
    `research/43_CONTROL_SURFACE_KNOWLEDGE_GAP_PLAN.md`. It turns the five
    ladder levels into a missing-evidence ledger rather than another review
    narrative: 15 missing-evidence items, four linked existing work orders,
    three levels requiring new behavior substrates or future work orders, two
    levels requiring reliability or predecision closure, zero newly licensed
    hidden-state-search levels, and zero promoted knowledge levels. Its current
    judgment is that MC005 should be closed or widened, MC006 should be treated
    as a monitor-only predecision frontier unless it beats final output/candidate
    geometry, the bridge route remains killed after MC033 unless a materially new
    substrate appears, and real uncertainty remains below the behavior gate.
87. A generated knowledge-substrate admission protocol:
    `code/control_surface_knowledge_substrate_admission.py` writes
    `data/control_surface_knowledge_substrate_admission.json` and
    `research/44_CONTROL_SURFACE_KNOWLEDGE_SUBSTRATE_ADMISSION.md`. It turns the
    "new behavior substrate required" finding into a front-door packet for the
    three affected levels: semi-synthetic familiar entities, symbolic/learned-
    memory bridge, and real abstention/uncertainty. Each packet has 11 admission
    gates: material novelty, behavior contract, parseability and label balance,
    direct controls, conflict mixture, null rows, source-disjoint holdout,
    prompt-channel locality, output/candidate baselines, side-effect/leakage
    checks, and split freeze. It validates that no new hidden-state search is
    licensed, the bridge route remains killed unless a proposal is outside
    MC007-MC033, and real uncertainty proposals require grounded labels and
    abstention controls before any signature work.
88. A generated knowledge-candidate queue:
    `code/control_surface_knowledge_candidate_queue.py` writes
    `data/control_surface_knowledge_candidate_queue.json` and
    `research/45_CONTROL_SURFACE_KNOWLEDGE_CANDIDATE_QUEUE.md`. It turns the
    admission protocol into six behavior-only substrate candidates: two for
    familiar-entity priors, two for symbolic/learned-memory bridge routing, and
    two for real uncertainty or context support. Each candidate has first-run
    behavior panels, dumb explanations to kill the attractive story early, all
    11 admission-gate bindings, next-queue links, and promote/death/
    containment/export rules. It validates 66 total gate bindings, zero
    hidden-state candidates, and zero promoted mechanisms.
89. A generated knowledge first-run pack:
    `code/control_surface_knowledge_first_run_pack.py` writes
    `data/control_surface_knowledge_first_run_pack.json` and
    `research/46_CONTROL_SURFACE_KNOWLEDGE_FIRST_RUN_PACK.md`. It converts all
    six behavior-only knowledge-substrate candidates into first-run packets with
    frozen-before-run fields, primary and secondary model targets, prereg/runner/
    result/status-card paths, panel thresholds, baseline checks, all 11
    admission-gate bindings, and promote/death/containment/export rules. It
    validates 6 packets, 37 behavior panels, 24 baseline checks, 66 gate
    bindings, 0 hidden-state packets, and 0 promoted mechanisms.
90. A first executed knowledge-candidate structural gate plus behavior smoke:
    `code/ksq003_bridge_statusless_evidence_aggregation_first_run.py` writes
    `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_first_run.json`,
    `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_smoke_limit10.json`,
    `research/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN_STATUS.md`,
    and
    `research/prereg/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN.md`.
    It turns the top first-run packet,
    `ksq003_bridge_statusless_evidence_aggregation`, into an executable
    behavior-substrate admission gate. The full structural substrate has 840
    rows, 40 sources, 7 panels, 3 templates, discovery/calibration/holdout
    source splits, balanced primary local-vs-atomic branch labels, and
    hidden-state license set to false. It passed after the prompt audit caught
    an accidental numeric channel: `Evidence note 1/2` created false
    atomic-number leakage for low atomic-number elements, so the runner now uses
    nonnumeric evidence labels. A contained 10-source behavior smoke then
    produced the real first KSQ003 boundary: in the selected `compact_fit`
    template, local direct control passed (`10/10` local), learned atomic direct
    control passed (`10/10` atomic), all-evidence-fit conflict routing passed
    (`10/10` local), and answer-absent nulls passed (`10/10` unknown), but
    one-evidence-mismatch conflicts returned local numbers (`10/10` local), and
    both symbol-only and parity-only ablations returned local numbers (`10/10`
    local each). KSQ003 is therefore not behavior-ready. The insight is that
    statusless symbol/parity evidence does not yet overcome local-table
    dominance, even when direct learned-memory recall and null behavior are
    separately available.
91. A first executed semi-synthetic familiar-entity prior-counterbalance run:
    `code/ksq001_familiar_entity_prior_counterbalance_first_run.py` writes
    `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_first_run.json`,
    `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_smoke_limit10.json`,
    `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_full_behavior.json`,
    `research/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN_STATUS.md`,
    and
    `research/prereg/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN.md`.
    It turns the Level 2 familiar-entity candidate into an executable
    behavior-substrate admission gate. The structural substrate has 480 rows,
    40 country entities, 4 panels, 3 templates, discovery/calibration/holdout
    source splits, no candidate collisions, clean prompt audit, and hidden-state
    license set to false. The 10-source smoke was a behavior-contrast candidate:
    selected `compact_question`, local artificial lookup was `10/10`, direct
    real-capital prior recall was `10/10`, nulls were `10/10` unknown, and the
    familiar-entity conflict showed `6/10` artificial, `1/10` real prior,
    `2/10` unknown, and `1/10` unparsed. The full 40-source behavior run did
    not promote: direct controls and nulls survived (`40/40` artificial local
    control, `38/40` real prior direct control with one lure and one unparsed,
    `38/40` unknown nulls with two unparsed), but the selected conflict template
    had only `24/40` parseable rows (`18/40` artificial, `2/40` real prior,
    `4/40` unknown, `16/40` unparsed). The learned boundary is different from
    KSQ003: familiar priors can compete weakly with prompt-local artificial
    values, but the mixture/parseability tradeoff blocks a behavior-ready
    substrate at full scale.
92. A first executed familiar-entity source-rewrite equivalence run:
    `code/ksq002_familiar_entity_source_rewrite_equivalence_first_run.py`
    writes
    `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_first_run.json`,
    `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_smoke_limit10.json`,
    `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_full_behavior.json`,
    `research/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN_STATUS.md`,
    and
    `research/prereg/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN.md`.
    It turns the source-rewrite equivalence packet into an executable
    behavior-substrate admission gate, still before hidden-state work. The
    structural substrate has 720 rows, 40 country entities, 6 panels, 3
    templates, discovery/calibration/source-disjoint holdout splits, no
    candidate collisions, clean prompt audit, and hidden-state license set to
    false. The 10-source smoke selected `registry_question` and passed all
    smoke behavior criteria: baseline lookup, neutral rewrite lookup, source
    deletion, query-only control, source-disjoint rewrite holdout, and margin
    reporting all passed. The full 40-source run selected `sentence_rewrite`
    and sharpened the boundary: baseline source lookup returned `39/40`
    artificial values; neutral rewrite returned `36/40` artificial values and
    `4/40` unparsed; source deletion and query-only controls were both `40/40`
    unknown. But the full gate still failed because the source-disjoint rewrite
    holdout was only `14/16` parseable/artificial with `2/16` unparsed,
    producing `0.875` parseability and artificial-value rates against the
    predeclared `0.900` threshold. The learned boundary is not that source
    rewrite is useless; it is that source-channel behavior can be mostly robust
    and source-local while still missing the strict holdout parseability needed
    before hidden-state work.
93. A first executed real-uncertainty precursor run:
    `code/ksq005_uncertainty_grounded_answerability_first_run.py` writes
    `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_first_run.json`,
    `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_smoke_limit10.json`,
    `research/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN_STATUS.md`,
    and
    `research/prereg/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN.md`.
    It turns the Level 5 grounded-answerability candidate into an executable
    behavior-substrate admission gate, still before hidden-state work. The
    structural substrate has 480 rows, 40 real country sources, 40 nonce
    country controls, 4 panels, 3 templates, discovery/calibration/source-
    disjoint holdout splits, no candidate collisions, clean prompt audit, a
    shared answer schema, and hidden-state license set to false. The 10-source
    smoke selected `reference_note`. It passed known factual direct control at
    the threshold (`8/10` known correct, `1/10` wrong candidate, `1/10`
    unparsed) and passed contradicted familiar-context correction/abstention
    (`5/10` corrected, `3/10` abstain, `1/10` false accept, `1/10` wrong
    answer). It failed the answerability gate because unknown nonce rows were
    only `4/10` abstain with `6/10` unparsed, and unsupported context rows were
    only `4/10` abstain with `3/10` unsupported prompt-city answers and `3/10`
    unparsed. The learned boundary is important: apparent correction of false
    familiar-context claims is easier than grounded abstention on unknown or
    unsupported entities. That means a "correction" behavior can look
    promising while the underlying answerability substrate is still too weak
    for uncertainty-control or refusal-control claims.
94. A second executed real-uncertainty/context-support precursor run:
    `code/ksq006_uncertainty_context_support_counterfactuals_first_run.py`
    writes
    `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_first_run.json`,
    `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_smoke_limit10.json`,
    `research/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN_STATUS.md`,
    and
    `research/prereg/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN.md`.
    It turns the Level 5 context-support counterfactual candidate into an
    executable behavior-substrate admission gate, still before hidden-state
    work. The structural substrate has 720 rows, 40 real country sources, 5
    panels, 3 templates, 2 control subtypes, discovery/calibration/source-
    disjoint holdout splits, no candidate collisions, no support-word prompt
    channel, clean prompt audit, and hidden-state license set to false. The
    10-source smoke selected `field_form`. It passed supported context rows
    (`9/10` supported answer, `1/10` abstain), irrelevant context rows
    (`10/10` abstain), and contradicting context rows (`8/10` contradiction
    detected, `1/10` false accept, `1/10` true answer despite contradiction).
    It failed the context-support behavior gate because insufficient context
    rows split `5/10` abstain and `5/10` unsupported answers, while the
    claim/context-only controls reproduced supported behavior on `18/20` rows:
    claim-only controls returned the supported answer `8/10` times and
    context-only city-mention controls returned it `10/10` times. The learned
    boundary is that context support can look strong on supported,
    irrelevant, and contradicting rows while relation-free claim or city
    mention controls still carry the answer channel. That is a diagnostic
    context-support failure, not uncertainty control.
95. A first executed bridge answer-interface minimal-pairs full behavior run:
    `code/ksq004_bridge_answer_interface_minimal_pairs_first_run.py` writes
    `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_first_run.json`,
    `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_smoke_limit10.json`,
    `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_full_behavior.json`,
    `research/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN_STATUS.md`,
    and
    `research/prereg/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN.md`.
    It turns the bridge answer-interface shortcut concern into an executable
    behavior-substrate admission gate, still before hidden-state work. The
    structural substrate has 720 rows, 40 element sources, 5 panels, 2
    templates, 9 subtypes, discovery/calibration/source-disjoint holdout
    splits, balanced local/atomic expected labels, no candidate collisions,
    clean prompt audit, and hidden-state license set to false. The 10-source
    smoke selected `question_form` and passed all smoke behavior criteria:
    conflict rows were `19/20` expected-correct, side-answer selections were
    `0/20`, answer-absent nulls were `10/10` UNKNOWN, and candidate/output
    margins were reported. The full run did not promote. The canonical
    selected `compact_form` failed with `bridge_minimal_pair_contrast_absent`:
    conflict rows were only `49/80` expected-correct, with `70/80` local
    selections and only `9/80` atomic selections; the atomic conflict subtype
    selected atomic only `9/40` times. The sharper insight is template
    fragility: `question_form` retained much more of the bridge (`0.850`
    conflict expected-correct, `0.700` atomic-branch atomic rate, `0.875`
    holdout expected-correct), while `compact_form` collapsed toward
    prompt-local lookup (`0.6125`, `0.225`, `0.625` respectively). First-token
    next-token numeric margins were degenerate (`0.0` local-minus-atomic gaps
    in the summaries), so candidate sequence-logprob is the meaningful output
    geometry baseline here.
96. A generated executed-KSQ outcome matrix:
    `code/control_surface_knowledge_first_run_outcomes.py` writes
    `data/control_surface_knowledge_first_run_outcomes.json` and
    `research/47_CONTROL_SURFACE_KNOWLEDGE_FIRST_RUN_OUTCOMES.md`. It turns the
    six executed knowledge first-run packets into one comparable table with
    candidate id, ladder level, terminal gate, final diagnostic, exported
    diagnostic class, selected template, failure axes, survived controls,
    failed gates, artifact paths, and hidden-state license. The matrix freezes
    the current distribution: all six KSQ structural gates passed; three
    candidates stopped at full behavior; three stopped at smoke behavior; zero
    are behavior-ready; zero allow hidden-state work; and the six exported
    diagnostics are
    `FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF`,
    `SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE`,
    `STATUSLESS_EVIDENCE_LOCAL_TABLE_DOMINANCE`,
    `ANSWER_INTERFACE_TEMPLATE_FRAGILITY`,
    `GROUNDED_ANSWERABILITY_UNKNOWN_UNSUPPORTED_FAILURE`, and
    `CONTEXT_SUPPORT_CLAIM_OR_MENTION_CONTROL_FAILURE`. This is the first
    machine-readable distribution of why the new knowledge-substrate candidates
    fail, rather than another prose pile of failed candidates.
97. A generated KSQ failure topology and second-wave plan:
    `code/control_surface_knowledge_failure_topology.py` writes
    `data/control_surface_knowledge_failure_topology.json` and
    `research/48_CONTROL_SURFACE_KNOWLEDGE_FAILURE_TOPOLOGY.md`. It reads the
    executed KSQ outcome matrix and separates the first wave into four
    topology nodes: structural construction solved, full-behavior parseability
    near-misses, bridge template/local-table boundary, and real-uncertainty
    answer-channel failures. It then records five second-wave work orders:
    immediate KSQ002 source-rewrite holdout repair, high-priority KSQ004
    template-invariance adjudication, medium KSQ001 parseability repair or
    closeout, medium KSQ003 material bridge redesign, and medium KSQ005/KSQ006
    real-uncertainty answerability redesign. Each work order includes required
    evidence, promotion, kill, containment, export rules, and explicit
    forbidden moves. The practical effect is that the project now knows which
    KSQ branch is a narrow repair candidate and which branches need material
    redesign before more probing.
98. A completed KSQ002 second-wave source-rewrite holdout repair:
    `code/ksq002_source_rewrite_holdout_repair.py` writes
    `results/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR/ksq002_source_rewrite_holdout_repair_first_run.json`,
    `results/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR/ksq002_source_rewrite_holdout_repair_smoke_limit10.json`,
    `results/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR/ksq002_source_rewrite_holdout_repair_full_behavior.json`,
    `research/prereg/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR.md`, and
    `research/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR_STATUS.md`.
    `code/control_surface_knowledge_second_wave_outcomes.py` then writes
    `data/control_surface_knowledge_second_wave_outcomes.json` and
    `research/49_CONTROL_SURFACE_KNOWLEDGE_SECOND_WAVE_OUTCOMES.md`. The
    repair fixed the named source-disjoint rewrite holdout boundary
    (`16/16` artificial-value rows) but degraded baseline lookup (`30/40`)
    and source-deletion UNKNOWN (`10/40`). The result kills ordinary KSQ002
    source-rewrite repair and exports
    `SOURCE_REWRITE_REPAIR_LOCALITY_REGRESSION` instead of promoting a
    behavior substrate.
99. A completed KSQ004 second-wave template-invariance adjudication:
    `code/ksq004_template_invariance_adjudication.py` writes
    `results/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION/ksq004_template_invariance_adjudication_first_run.json`,
    `results/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION/ksq004_template_invariance_adjudication_smoke_limit10.json`,
    `results/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION/ksq004_template_invariance_adjudication_full_behavior.json`,
    `research/prereg/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION.md`, and
    `research/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION_STATUS.md`.
    The full adjudication ran 1,080 rows across 40 sources, three templates,
    and the original KSQ004 matched-pair, direct-control, conflict, side-
    leakage, null, and holdout panels. `question_form` passed, `relation_key_form`
    passed, and `neutral_sentence_form` failed. The result exports
    `TEMPLATE_INVARIANT_BRIDGE_BEHAVIOR`, admits a bounded behavior substrate,
    and licenses a later signature screen, while keeping hidden-state claims,
    interventions, and mechanism claims closed.
100. A completed KSQ001 second-wave familiar-prior parseability bound:
    `code/ksq001_familiar_prior_parseability_bound.py` writes
    `results/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND/ksq001_familiar_prior_parseability_bound_first_run.json`,
    `results/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND/ksq001_familiar_prior_parseability_bound_smoke_limit10.json`,
    `results/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND/ksq001_familiar_prior_parseability_bound_full_behavior.json`,
    `research/prereg/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND.md`, and
    `research/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND_STATUS.md`.
    The 10-source compact replay looked like a behavior-contrast candidate,
    but the full 40-source run reproduced the original boundary exactly:
    `compact_original_replay` selected, conflict parseability `0.600`, label
    counts `18` artificial-value, `2` real-prior, `4` UNKNOWN, and `16`
    unparsed rows. Softer answer-shape variants raised parseability only while
    losing the prior branch or moving toward UNKNOWN. The result closes
    ordinary KSQ001 parseability repair and keeps hidden-state claims,
    interventions, and mechanism claims closed.
101. A completed KSQ003 second-wave evidence-sufficiency redesign:
    `code/ksq003_evidence_sufficiency_redesign.py` writes
    `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_first_run.json`,
    `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_smoke_limit10.json`,
    `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_full_behavior.json`,
    `research/prereg/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN.md`, and
    `research/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN_STATUS.md`.
    The full 40-source run changed the failure type. Direct local lookup was
    `40/40` local, direct learned atomic recall was `40/40` atomic, and
    answer-absent nulls were `38/40` UNKNOWN under the selected
    `compact_identity` template. But complete identity conflict produced only
    `13/40` atomic answers and `27/40` UNKNOWN, while null stress produced
    `66/120` UNKNOWN and `41/120` atomic answers. Across templates,
    `identity_packet` got `40/40` complete-evidence atomic answers but only
    `1/120` null-stress UNKNOWN; `compact_identity` improved null stress but
    lost the learned branch. The result exports
    `STATUSLESS_EVIDENCE_SUFFICIENCY_BOUNDARY`: this is not another simple
    local-table-dominance result, but it is still not behavior-ready and still
    licenses no signature screen, hidden-state claim, intervention, or
    mechanism claim.
102. A completed KSQ005/KSQ006 second-wave relation-evidence answerability
    redesign:
    `code/ksq005_006_relation_evidence_answerability_redesign.py` writes
    `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_first_run.json`,
    `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_smoke_limit10.json`,
    `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_full_behavior.json`,
    `research/prereg/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY.md`, and
    `research/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY_STATUS.md`.
    The structural run passed with 840 rows, 40 sources, seven panels, three
    templates, clean prompt audit, source-disjoint split, and clean relation
    grammar construction. The full 40-source behavior run selected
    `compact_relation`: known/direct answers reached `39/40` and exact
    `REL capital_of` supported rows reached `40/40`, but unknown nonce rows
    only abstained `27/40`, unsupported rows only abstained `28/40`,
    contradiction rows chose the prior/true capital `38/40`, and claim/mention
    controls reproduced the supported capital on `79/80` rows. The stricter
    `relation_rows` template protected controls and abstention better, but
    supported relation answering collapsed to `11/40`. The result exports
    `RELATION_EVIDENCE_ANSWERABILITY_BOUNDARY`, kills the current
    real-uncertainty route, and licenses no signature screen, hidden-state
    claim, intervention, or mechanism claim.
103. A post-KSQ005/KSQ006 nonce-evidence answerability calibrator:
    `code/ksq007_nonce_evidence_answerability_calibrator.py` writes
    `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_first_run.json`,
    `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_smoke_limit10.json`,
    `research/prereg/KSQ007_NONCE_EVIDENCE_ANSWERABILITY.md`, and
    `research/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY_STATUS.md`. The
    structural gate passed with 840 rows, 40 nonce sources, seven panels,
    three templates, source-disjoint splits, prompt audit, no support-word or
    status prompt channel, no duplicate rows, and no candidate collisions. The
    10-source smoke selected `evidence_rows`: exact evidence answered `9/10`,
    absent evidence abstained `10/10`, unrelated-entity rows abstained `10/10`,
    conflicting rows abstained `9/10`, mention-only and query-only controls
    abstained `10/10`, but claim-only controls reproduced the value `6/10`.
    The result exports `NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY`: removing
    familiar capitals and real city values helps most answerability branches,
    but claim text still acts too much like evidence. No signature screen,
    hidden-state claim, intervention, mechanism claim, uncertainty claim, or
    factual-correction claim is licensed.
104. A KSQ007B claim-channel boundary audit:
    `code/ksq007_claim_channel_boundary_audit.py` writes
    `results/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY/ksq007_claim_channel_boundary_first_run.json`,
    `results/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY/ksq007_claim_channel_boundary_smoke_limit10.json`,
    `research/prereg/KSQ007_CLAIM_CHANNEL_BOUNDARY.md`, and
    `research/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY_STATUS.md`. The structural
    gate passed with 1,200 rows, 40 nonce sources, 10 panels, three templates,
    source-disjoint splits, prompt audit, no support-word or status prompt
    channel, no duplicate rows, no candidate collisions, and no hidden-state
    license. The 10-source smoke selected `counted_uncounted_sections`: exact
    evidence answered `9/10`, literal `CLAIM answer_for(entity)=value`
    abstained `9/10`, `NOT_EVIDENCE`, quoted evidence syntax, wrong-predicate
    claim, mention-only, and query-only controls abstained `10/10`; but prose
    claims reproduced `4/10`, bare `answer_for(entity)=value` reproduced
    `7/10`, and evidence-looking rows outside the counted block reproduced
    `2/10`. The exported diagnostic is `ANSWER_FOR_SYNTAX_CLAIM_LEAK`: the
    primary leak is not simply familiar priors, the word `CLAIM`, or value
    mention. It is the slot-binding syntax itself becoming answer-bearing. No
    signature screen, hidden-state claim, intervention, mechanism claim,
    uncertainty claim, or factual-correction claim is licensed.
105. A hardened validator path for the newest knowledge-substrate diagnostics:
    `code/validate_control_surface_atlas.py` now explicitly checks KSQ007 and
    KSQ007B structural and smoke artifacts. The checks pin schema version,
    candidate IDs, run types, row counts, source splits, selected templates,
    decision flags, diagnostic classes, exported diagnostics, failed criteria,
    and selected-panel label counts. This turns `NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY`
    and `ANSWER_FOR_SYNTAX_CLAIM_LEAK` from prose claims into machine-checked
    boundary facts.
106. A generated knowledge third-wave outcome layer:
    `code/control_surface_knowledge_third_wave_outcomes.py` writes
    `data/control_surface_knowledge_third_wave_outcomes.json` and
    `research/50_CONTROL_SURFACE_KNOWLEDGE_THIRD_WAVE_OUTCOMES.md`. It records
    the post-second-wave KSQ007 -> KSQ007B diagnostic chain as two completed
    outcomes, 2,040 structural rows, 510 smoke rows, zero behavior-ready
    outcomes, zero signature-screen licenses, zero hidden-state licenses, zero
    intervention licenses, and zero mechanism claims. The generated conclusion
    is that real-world prior pressure is not the whole answerability problem:
    the remaining boundary is answer-bearing slot-binding syntax, especially
    bare `answer_for(entity)=value` assertions.
107. A KSQ008 neutral-evidence channel repair and generated fourth-wave outcome
    layer: `code/ksq008_neutral_evidence_channel_repair.py` builds a 1,440-row
    structural substrate and a 360-row, 10-source smoke over 12 panels and 3
    templates. The selected `field_registry` template suppresses forbidden
    bare answer_for, claim answer_for, prose claim, uncounted neutral, quoted
    neutral, absent, unrelated, and query-only controls at `10/10`, but exact
    neutral evidence answers only `5/10`, conflicting neutral evidence selects
    a value `2/10`, counted wrong-schema `answer_for(entity)=value` rows
    reproduce `5/10`, and neutral evidence versus a forbidden bare alternate
    answers only `2/10`. The exported diagnostic is
    `NEUTRAL_EVIDENCE_POSITIVE_FAILED`: the first neutral-channel repair does
    not produce a behavior-ready substrate. The generated fourth-wave layer
    `code/control_surface_knowledge_fourth_wave_outcomes.py` writes
    `data/control_surface_knowledge_fourth_wave_outcomes.json` and
    `research/51_CONTROL_SURFACE_KNOWLEDGE_FOURTH_WAVE_OUTCOMES.md`, with zero
    behavior-ready, signature-screen, hidden-state, intervention, or mechanism
    licenses.
108. A KSQ009 schema-specific value-lookup repair and generated fifth-wave
    outcome layer: `code/ksq009_schema_specific_value_lookup.py` shows that
    explicit schema labels do not beat answer-bearing `answer_for` syntax. The
    selected `kv_lines` template answers exact ALLOW rows only `2/10`, counted
    wrong-schema `answer_for(entity)=value` rows reproduce `9/10`, uncounted
    wrong-schema `answer_for` rows reproduce `7/10`, and uncounted
    `answer_for` alternates override ALLOW rows `9/10`. The fifth-wave layer
    records `SCHEMA_SPECIFIC_POSITIVE_FAILED`, with no behavior, hidden-state,
    intervention, or mechanism license.
109. A KSQ010 two-stage codebook repair and generated sixth-wave outcome
    layer: `code/ksq010_two_stage_codebook_value_lookup.py` improves the
    positive branch but does not repair locality or answer-like syntax
    competition. Exact bridges answer `8/10`, but conflict panels select a
    value `10/10`, counted `answer_for(entity)=value` rows reproduce `10/10`,
    and an `answer_for` alternate overrides the codebook bridge `9/10`. The
    sixth-wave layer records `CODEBOOK_POSITIVE_FAILED`.
110. A KSQ011 answer-for syntax ablation and generated seventh-wave outcome
    layer: `code/ksq011_answer_for_syntax_ablation.py` shows the problem is
    not one exact string. Exact bridge lookup answers `9/10`, but exact
    `answer_for`, spaced `answer_for`, colon `answer_for`, `answer_to`,
    `value_for`, prose value notes, and quoted `answer_for` override `7/10` to
    `10/10`. Plain entity assignment and bare alternate mention mostly
    preserve the bridge at `8/10`. The seventh-wave layer records
    `FUNCTION_ASSIGNMENT_ANSWER_CHANNEL_DOMINANCE`.
111. A KSQ012 function-assignment wrapper repair and generated eighth-wave
    outcome layer: `code/ksq012_function_assignment_wrapper_repair.py` restores
    the bridge baseline and raw positive control, but wrappers do not
    quarantine function-like assignment text. Exact bridge and raw
    `answer_for` both pass at `9/10`; inactive, comment, fenced, below-cut,
    detached, unrelated-entity, and assignment-only inactive surfaces leak
    `6/10` to `9/10`. The eighth-wave layer records
    `FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK`.
112. A KSQ013 nonfunction representation screen and generated ninth-wave
    outcome layer: `code/ksq013_nonfunction_representation_screen.py` removes
    function-call and assignment syntax from the proposed repair surfaces.
    Exact bridge and raw `answer_for` still pass at `9/10`. Nonfunction forms
    are less catastrophic than wrappers but still fail: value bank and decoy
    pair bridge panels answer `8/10`, metadata `7/10`, separated entity/value
    `6/10`, bare alternate mention `8/10`, and separated entity/value-only
    control leaks `4/10`. Catalog slash text is the one constructive signal at
    `9/10`, but KSQ013 lacks a slash-only no-bridge locality control. The
    ninth-wave layer records `NONFUNCTION_REPRESENTATION_CONTROL_AND_REPAIR_LEAK`.
113. A KSQ014 slash-locality packet and generated tenth-wave outcome layer:
    `code/ksq014_slash_locality_packet.py` tests the missing control from
    KSQ013. Exact bridge and raw `answer_for` both pass at `9/10`. Catalog
    slash entity and reversed variants answer the bridge `9/10` with `1/10`
    unparsed, catalog slash decoy answers `9/10` with `1/10` abstain, and all
    matched slash-only controls abstain `10/10`. Bare slash fails as a repair
    surface: the bare slash bridge panel answers only `6/10` with `4/10`
    unparsed. The tenth-wave layer records `SLASH_LOCALITY_BRIDGE_LOSS`: the
    useful result is a split between catalog-labeled slash, which deserves a
    full-source behavior packet, and bare slash, which does not.
114. A KSQ015 catalog-slash full-source packet and generated eleventh-wave
    outcome layer: `code/ksq015_catalog_slash_full_source_packet.py` removes
    bare slash and tests the KSQ014 catalog-labeled slash survivor across all
    40 sources. The full-source run does not promote. Exact bridge answers
    only `35/40` with `5/40` unparsed code-token outputs, raw `answer_for`
    remains active at `39/40`, catalog slash entity answers `36/40`, catalog
    slash decoy answers `35/40`, catalog slash reversed answers `35/40`, and
    every catalog slash-only no-bridge control abstains `40/40`. The
    eleventh-wave layer records `CATALOG_SLASH_BRIDGE_POSITIVE_FAILED`: the
    catalog slash locality controls are clean, but the counted bridge
    substrate itself is not full-source reliable under this contract.

## What We Have Not Accomplished

We have not accomplished:

1. A broad mechanism card for truth, honesty, factuality, or knowledge.
2. A deployable control surface.
3. A hidden-state monitor that generally beats output/logit monitoring.
4. A single-head or single-layer circuit for MC005.
5. A complete cross-model reliability atlas.
6. A clean intervention on the MC005 V26 row signature.
7. A clean donor-replacement intervention for row-level all-three structure.
8. A fully reliable write-replacement mechanism card for MC005.
9. A mechanism-grade MC006 knowledge signature.
10. Any MC006 intervention justified by a passing hidden signature.
11. A knowledge-like signature that beats both pre-output and final-output
    controls.
12. A prompt-channel-local bridge family that preserves artificial-versus-real
    contrast, parseability, null behavior, and source-disjoint holdout balance
    without a visible source-status label carrying the answer rule.
13. A causal MC006 intervention from the V16 early signature.
14. A behavior-ready neutral-evidence, wrapper, or nonfunction representation
    repair for the KSQ007B/KSQ011/KSQ012 answer-channel boundary.
15. A full-source catalog-slash behavior substrate after KSQ015: the run was
    executed and failed because the exact bridge positive control did not
    clear full-source or holdout reliability.
14. A same-table MC006 margin-matched mechanism claim on the V14 row set.
15. A strict overlapping-margin MC006 generated-answer table.
16. A MC006 table-selection route that rescues strict final-margin overlap
    inside the current V19 bank.
17. An approximate pair-matched MC006 route that neutralizes candidate-score or
    final-output margin baselines inside the current V19 bank.
18. A source/path MC006 route that beats final candidate-score or final-output
    baselines inside the current V19/V20/V21/V22 bank.
19. A same-interface MC006 row-generation route that produces strict
    final-margin overlap under greedy binary first-token generation.
20. A delayed-city MC006 answer interface that beats full-completion candidate
    scoring on holdout.
21. A candidate-score-decoupled MC006 hidden signature that beats
    shuffled-label selected-search nulls.
22. A locked-coordinate MC006 hidden signature that transfers across
    candidate-decoupled delayed-city templates.
23. A transfer-ready expanded MC006 candidate-decoupled delayed-city bank.
24. A prompt-channel-local numeric bridge that preserves MC012-level
    local-versus-learned contrast after MC013-style source-status ablation.
25. A source-selector bridge that beats neutral label order, source-definition
    order, answer-option order, local-source salience, and answer-absent null
    controls.
26. A row-code numeric bridge that preserves clean controls and nulls while
    making expected-atomic rows follow the route rule under prompt-local table
    pressure.
27. A conditional source-arbitration bridge where the same kind of rule can
    select prompt-local values or learned atomic facts symmetrically.
28. A route-code behavior substrate that reliably routes even between two
    prompt-visible numeric branches.
29. An explicit branch-name behavior substrate that defeats local-source
    salience, rule-definition-order sensitivity, and learned-memory branch
    collapse under prompt-local table pressure.
30. A few-shot query-operation bridge that keeps both prompt-local and learned
    atomic conflict branches above gate on full source-disjoint evaluation.
31. A constrained-choice answer interface that preserves direct atomic
    controls, answer-absent nulls, and learned atomic conflict routing.
32. A numeric-option answer interface that preserves direct atomic controls,
    clean nulls, and learned atomic conflict routing at the same time.
33. A structured answer schema, such as `ANSWER=...`, JSON, A/B/C choices, or
    numeric options, that preserves the same behavior as the bare-integer
    10-source smoke without breaking controls, nulls, parseability, or learned
    routing.
34. A bare-integer full-source bridge that keeps the learned atomic branch
    above gate without other-number leakage while preserving clean direct
    controls and nulls.
35. A factorized operation-leak bridge that keeps learned atomic routing,
    answer-absent nulls, prompt-local routing, and low other-number leakage
    above gate at the same time. MC029 moved those axes separately but did not
    repair them together.
36. A simple absence-guard repair for the MC029 rules-only branch/null tradeoff.
    MC030 shows that direct guards either worsen null rows or collapse learned
    atomic routing.
37. A transfer-stable mixture law across new model families, new behavior
    families, or fresh bridge ladders. The current mixture law is a measured
    current-atlas distribution, not yet a predictive universal law.
38. A lead-time causal lever. The current decision-frontier artifact records
    MC004 and MC006 as monitor-only and records 0 predecision causal
    candidates.
39. A transfer-ready mechanism. The current transfer matrix records 0
    transfer-ready mechanisms; medium or cross-model evidence remains
    diagnostic until signature, intervention, null, locality, side-effect, and
    output/candidate controls survive on the widened target.
40. A full-reliability mechanism. The current reliability matrix records 0
    rows that clear behavior, signature, intervention, null/locality, local
    path, robustness/side-effect, and transfer gates simultaneously.
41. A KSQ003 behavior-ready statusless evidence-aggregation bridge. The
    structural substrate passed, the first smoke showed local-table dominance
    on mismatch and single-feature ablation branches, and the second-wave
    evidence-sufficiency redesign removed most local-table intrusion only by
    exposing a new learned-branch/null-stress tradeoff.
42. A KSQ001 behavior-ready familiar-entity prior-counterbalance substrate.
    Direct local lookup, direct semantic-prior recall, nulls, candidate margins,
    and holdout survived, but the first-run full conflict template failed
    parseability, and the second-wave answer-shape repair replayed the same
    full-scale boundary rather than repairing it.
43. A KSQ002 behavior-ready source-rewrite equivalence substrate. Baseline
    lookup, neutral rewrite, source deletion, query-only controls, and margins
    mostly survived, but the full source-disjoint rewrite holdout missed the
    predeclared parseability gate.
44. A KSQ005 behavior-ready grounded-answerability substrate. The structural
    substrate passed and contradicted familiar facts often corrected or
    abstained, but unknown nonce rows and unsupported context rows failed
    abstention and parseability in the 10-source smoke; the second-wave
    relation-evidence redesign also failed unknown, unsupported,
    contradiction, and claim/mention controls.
45. A KSQ006 behavior-ready context-support counterfactual substrate. The
    structural substrate passed and supported/irrelevant/contradicting rows
    passed in the 10-source smoke, but insufficient rows and claim/context-only
    controls failed; the combined KSQ005/KSQ006 redesign showed that formal
    relation evidence can act as an answer trigger or an abstention guard, but
    not both under this route.
46. A KSQ004 hidden-state signature, intervention, or mechanism claim. The
    second-wave template-invariance adjudication admits a bounded behavior
    substrate, but it has not probed internals, steered the behavior, or shown
    an internal knowledge-control surface.
47. Any behavior-ready KSQ first-run substrate. The generated KSQ outcome
    matrix records six structural passes, but zero behavior-ready rows and
    zero hidden-state licenses.
48. A behavior-ready real-uncertainty KSQ substrate after the completed
    second-wave program. KSQ001, KSQ002, KSQ003, KSQ004, and KSQ005/KSQ006
    are now closed, bounded, or admitted behavior-only, but no uncertainty,
    context-support, refusal, hidden-state, intervention, or mechanism claim is
    licensed.
49. A behavior-ready nonce-evidence answerability calibrator. KSQ007 removes
    familiar capital and real-city pressure and makes most answerability
    branches work under `evidence_rows`, but claim-only rows still reproduce
    the value on `6/10` smoke rows, so it remains a boundary diagnostic rather
    than a substrate admission.
50. A repaired claim-channel boundary for nonce evidence. KSQ007B shows that
    counted/uncounted sections can suppress several control channels, but bare
    `answer_for(entity)=value` still reproduces the value on `7/10` smoke
    rows and prose claims reproduce on `4/10`, so claim-channel control remains
    behavior-not-ready and hidden-state work remains forbidden.

## Current Claim Boundaries

Allowed claims:

- Qwen3-0.6B truth/agreement prompts have reproducible behavior control
  surfaces, but the dense and source-local interpretations do not support a
  mechanism card.
- Qwen3-1.7B synthetic associative lookup has a narrow source-value internal
  control surface.
- The strongest MC005 surface is currently an all-head layers-24-26 aggregate
  interaction block.
- Layers 24-26 final-query attention-write replacement exactly reproduces the
  lookup target source-mask effect on lookup rows.
- The MC005 write surface is not fully reliable because answer-absent null rows
  can flip.
- MC005 is a frozen bounded mechanism card for Qwen3-1.7B synthetic lookup
  under the tested `Response:` contract, not a full promoted mechanism card.
- MC006 V10 passed a chat-rendered generated-answer behavior substrate for
  capital facts versus fictional code.
- MC006 V11 found a hidden separation on that V10 table, but it is not
  mechanism-grade because output margin, requested mode, and shuffled-label
  controls match it.
- MC006 V12 found a hidden separation on the matched `real_after_fiction`
  surface, but it is not mechanism-grade because holdout balance is weak and
  candidate-score, output-margin, and shuffled-label controls match it.
- MC006 V13 generated a matched `real_after_fiction` table where holdout had
  both labels, but it failed the behavior gate at 29/40 binary rows.
- MC006 V14 passed a matched generated `real_after_fiction` behavior table
  under a narrow accent-normalized strict parser.
- MC006 V15 found a perfect hidden separation on V14, but it is not
  mechanism-grade because candidate-score and next-token output margins match
  it.
- MC006 V16 found a pre-output `after_mapping_line/layer_4` hidden separation
  on V14 that beats same-position output margin, selected-position token
  controls, and shuffled-label p95, but it is still not mechanism-grade because
  candidate-score and final next-token output margins match it.
- MC006 V17 tested simple additive steering on that V16 direction and failed:
  holdout margins moved only weakly, controls matched or exceeded the effect,
  and no holdout generated label changed directionally.
- MC006 V18 reproduced the V16 early signal but showed the V14 holdout split
  has no true/override overlap in candidate-score or final-output margins, so
  same-table margin-matched promotion is blocked.
- MC006 V19 improved the generated-answer table with 35/40 binary rows and 4/4
  holdout true/override labels under `separate_task_weak`, but strict
  candidate and final-output margin overlap still failed.
- MC006 V20 showed the existing V19 binary bank has no strict final-output
  overlap even when pooled across templates.
- MC006 V21 showed the existing V19/V20 approximate pairs still have perfect
  candidate-score and final-output margin baseline ordering on holdout, so
  approximate matching does not rescue MC006.
- MC006 V22 showed source/path and later pre-output lead-time monitors exist
  inside the same bank, but final candidate-score and final-output margins
  still perfectly order holdout pairs.
- MC006 V23 showed strict final-output margin overlap is a malformed default
  gate for the current greedy generated-answer interface: final margin sign
  predicted 257/257 V19 binary labels, while candidate-score margin had
  overlap on V19 all-binary rows.
- MC006 V24 changed the answer interface to delayed JSON city generation. The
  selected table had 31 binary rows, 10 true rows, 21 override rows, and 2 true
  / 4 override holdout rows. Final city-token sign accuracy fell to 0.516 with
  raw overlap, but JSON-completion candidate-score holdout AUC reached 1.000,
  beating the selected hidden monitor at 0.875 holdout AUC.
- MC006 V25 found a candidate-score-decoupled delayed-city table inside the V24
  bank. On `untrusted_note_real`, JSON-completion candidate-score holdout AUC
  was 0.333 while the selected `after_mapping_line/layer_10` hidden monitor
  reached 1.000 holdout AUC, but shuffled-label selected-search p95 also
  reached 1.000, so this is selection-null-confounded rather than
  mechanism-grade.
- MC006 V26 locked the V25 coordinate and tested transfer from
  `untrusted_note_real` to `separate_task_weak`. Source holdout AUC was 0.750,
  but transfer holdout AUC was 0.500, below the 0.833 position-local
  next-token-margin control and the 0.833 train-label shuffle p95.
- MC006 V27 expanded to 640 delayed-city rows across 16 predeclared
  source/transfer templates. Four templates were candidate-decoupled, pooled
  binary rows reached 138, and pooled holdout labels were 14 true / 14 override,
  but only one transfer-role template passed.
- MC006 V28 targeted the transfer side with 640 rows across 16 predeclared
  transfer-role templates. It found `transfer_sandbox_mapping_then_geo` with 35
  binary rows, 4 true / 4 override holdout labels, and JSON candidate-score
  holdout AUC 0.1875, but only one transfer-ready template passed.
- KSQ001 establishes a clean semi-synthetic familiar-entity structural
  substrate and shows that local artificial lookup, direct real-capital prior
  recall, and answer-absent nulls can all pass together.
- KSQ001 does not yet pass full behavior admission: the only selected template
  with local/prior conflict mixture fails primary parseability at full scale.
- KSQ001 second-wave parseability repair is closed negatively: the compact
  replay reproduces the original `0.600` full-scale conflict parseability with
  `18` artificial-value, `2` real-prior, `4` UNKNOWN, and `16` unparsed rows,
  while softer answer-shape variants improve parseability only by losing the
  prior branch or shifting toward UNKNOWN.
- KSQ002 establishes a clean source-rewrite structural substrate and shows that
  neutral rewrite, source deletion, and query-only controls can be audited
  together under a familiar-entity source-value contract.
- KSQ002 does not yet pass full behavior admission: the selected full template
  preserves most rewritten source values, but source-disjoint rewrite holdout
  parseability reaches only `0.875` against the `0.900` gate.
- KSQ002 ordinary source-rewrite repair has now been tried once and killed:
  the `city_field_rewrite` answer-channel repair made neutral rewrite and
  source-disjoint rewrite holdout perfect, but reduced baseline lookup to
  `30/40` and source-deletion UNKNOWN to `10/40`.
- KSQ003 establishes a clean statusless evidence-aggregation structural
  substrate, but its behavior smoke fails because mismatch and incomplete
  evidence branches collapse to local-table outputs.
- KSQ003 second-wave evidence-sufficiency redesign changes the failure mode:
  local-table dominance mostly disappears, but complete-evidence learned
  routing and contradiction/single-feature UNKNOWN behavior do not pass
  together. The route is bounded as
  `STATUSLESS_EVIDENCE_SUFFICIENCY_BOUNDARY`, not promoted.
- KSQ005 establishes a clean grounded-answerability structural substrate and
  shows an asymmetric behavior boundary: known facts and contradicted familiar
  facts can pass while unknown nonce and unsupported-context abstention fail.
- KSQ006 establishes a clean context-support counterfactual structural
  substrate and shows a sharper counterfactual boundary: supported,
  irrelevant, and contradicting rows can pass while insufficient rows and
  claim/context-only controls still reproduce answer channels.
- KSQ005/KSQ006 second-wave relation-evidence answerability exports
  `RELATION_EVIDENCE_ANSWERABILITY_BOUNDARY`: `compact_relation` answers known
  and exact supported rows but fails unknown, unsupported, contradiction, and
  claim/mention controls, while stricter `relation_rows` protects controls and
  abstention but loses supported answering.
- KSQ007 nonce-evidence answerability exports
  `NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY`: removing familiar capital priors and
  real city values repairs absent, unrelated, contradiction, mention-only, and
  query-only behavior under the selected `evidence_rows` template, but
  claim-only text still reproduces the nonce value on `6/10` smoke rows.
- KSQ007B claim-channel boundary exports `ANSWER_FOR_SYNTAX_CLAIM_LEAK`:
  counted/uncounted sections suppress literal claim labels, `NOT_EVIDENCE`,
  quotes, wrong predicates, mention-only, and query-only baselines, but bare
  `answer_for(entity)=value` reproduces `7/10` and prose claims reproduce
  `4/10`. The remaining failure is primarily slot-binding syntax becoming
  answer-bearing, not the word `CLAIM` or simple value mention.
- KSQ008 neutral-evidence repair exports `NEUTRAL_EVIDENCE_POSITIVE_FAILED`:
  neutral row labels can suppress several forbidden controls, but exact
  neutral evidence answers only `5/10` and wrong-schema answer_for rows still
  reproduce values.
- KSQ009 schema-specific value lookup exports `SCHEMA_SPECIFIC_POSITIVE_FAILED`:
  explicit ALLOW/BLOCK row labels are not enough when competing
  `answer_for(entity)=value` syntax remains answer-bearing.
- KSQ010 two-stage codebook lookup exports `CODEBOOK_POSITIVE_FAILED`: the
  bridge improves positive lookup, but conflict locality and answer_for
  alternates still fail.
- KSQ011 answer-for syntax ablation exports
  `FUNCTION_ASSIGNMENT_ANSWER_CHANNEL_DOMINANCE`: the failure is a broader
  function-like assignment answer channel, not one exact spelling and not
  generic value salience alone.
- KSQ012 wrapper repair exports
  `FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK`: wrapper prose,
  comments, fences, cut markers, and inactive labels are weak controls for
  visible function-like assignment text.
- KSQ013 nonfunction representation screen exports
  `NONFUNCTION_REPRESENTATION_CONTROL_AND_REPAIR_LEAK`: nonfunction value
  mentions reduce function-assignment dominance but still split bridge
  preservation, parseability, and no-bridge locality. Catalog slash text is a
  promising smoke signal only, not a promoted surface.
- KSQ014 slash locality packet exports `SLASH_LOCALITY_BRIDGE_LOSS`: catalog
  slash variants pass matched no-bridge locality smoke, but bare slash fails
  bridge preservation through parse loss. Catalog-labeled slash is a candidate
  for a full-source behavior-only packet, not a hidden-state license.
- KSQ015 catalog slash full-source packet exports
  `CATALOG_SLASH_BRIDGE_POSITIVE_FAILED`: removing bare slash keeps all
  catalog slash-only controls clean at `40/40`, but exact bridge answers only
  `35/40` with `5/40` unparsed code-token outputs, so catalog slash cannot
  promote while the counted bridge substrate itself is unreliable.
- KSQ004 establishes a clean matched answer-interface structural substrate and
  shows that answer-interface matching is not a complete shortcut explanation:
  `question_form` retains a mostly working local-vs-learned bridge, but full
  first-run behavior is not robust because `compact_form` collapses the
  learned branch toward prompt-local table lookup.
- KSQ004 second-wave template-invariance adjudication admits a bounded bridge
  behavior substrate: `question_form` and `relation_key_form` pass all
  predeclared full-behavior gates, while `neutral_sentence_form` fails conflict
  routing. This licenses a later signature screen only, not a hidden-state
  claim, intervention, or mechanism card.
- The executed KSQ outcome matrix establishes the current knowledge-substrate
  distribution: six structural passes, three full-behavior terminal failures,
  three smoke-behavior terminal failures, six exported diagnostics, zero
  behavior-ready rows, and zero hidden-state licenses.
- The KSQ second-wave outcome layer establishes a split frontier: KSQ001's
  ordinary familiar-prior parseability repair is closed as a scale-fragile
  tradeoff, KSQ002's ordinary source-rewrite repair route is killed by
  locality regression, KSQ003's material bridge redesign is bounded by an
  evidence-sufficiency tradeoff, and KSQ004's relation-format bridge is
  behavior-ready but still mechanism-empty, while KSQ005/KSQ006's
  real-uncertainty relation-evidence route is killed. The second-wave program
  is completed as five work orders, zero pending rows, one behavior-ready row,
  one signature-screen license, three killed/closed routes, and zero
  hidden-state/intervention licenses.

Disallowed claims:

- the model has an honesty vector;
- h14 is a truth-versus-agreement mechanism;
- hint-source masking identified a compact attention circuit;
- MC005 generalizes beyond synthetic associative lookup;
- MC005 is a single-head or single-layer circuit;
- MC005 row-level heterogeneity is explained by source position alone;
- MC005 row-level heterogeneity is explained by simple prompt-layout factors;
- the V26 internal row signature is already causal;
- additive residual steering of `l20_target_value` controls row structure;
- donor replacement of `l20_target_value` controls row structure;
- MC005 V29 is already a full mechanism card;
- MC005 V31 fully explains the write-replacement null boundary;
- MC005 should keep being repaired as if full promotion is one small tweak
  away;
- MC006 V10 has already found a hidden internal signature;
- MC006 V11 supports intervention;
- MC006 V11 hidden monitoring beats output/logit or requested-mode monitoring;
- MC006 V12 supports intervention;
- MC006 V12 hidden monitoring beats candidate-score or output-margin
  monitoring;
- MC006 V13 supports hidden-state discovery or intervention;
- MC006 V13 passed the matched generated behavior-substrate gate;
- MC006 V14 supports intervention;
- MC006 V15 supports intervention;
- MC006 V15 hidden monitoring beats candidate-score or output-margin
  monitoring;
- MC006 V16 supports intervention;
- MC006 V16 is a mechanism-grade knowledge signature;
- MC006 V16 hidden monitoring beats final candidate-score or final
  output-margin monitoring;
- MC006 V17 rescues MC006 intervention;
- the V16 `after_mapping_line/layer_4` direction is a simple additive control
  vector;
- MC006 V17 produced directional holdout generated-label control;
- MC006 V18 rescues MC006 as a mechanism claim;
- the current V14 row set can adjudicate margin-matched MC006 causality;
- residualized MC006 hidden scores survive candidate-score and final-output
  controls on holdout;
- MC006 V19 supports hidden-state probing or intervention;
- near-pair matching at z <= 0.5 is equivalent to strict margin overlap;
- pooling V19 templates solves MC006 without prompt-template controls;
- MC006 V20 rescues strict margin-overlap selection inside V19;
- matched-template or per-source selection can make the existing V19 bank
  mechanism-ready;
- MC006 V21 supports a hidden signature or intervention;
- approximate V19/V20 pair matching neutralizes candidate-score or final-output
  baselines;
- MC006 V22 supports a hidden signature or intervention;
- source/path probing inside the current MC006 bank neutralizes final
  candidate-score or final-output baselines;
- MC006 V23 supports a hidden signature or intervention;
- MC006 V24 supports a hidden signature or intervention;
- delayed JSON city answers remove output/candidate visibility from MC006;
- MC006 V25 supports a hidden signature or intervention;
- candidate-score decoupling alone is enough to promote an MC006 signature;
- `after_mapping_line/layer_10` is a reliable MC006 knowledge-control surface;
- MC006 V26 supports a hidden signature or intervention;
- the V25 coordinate transfers across candidate-decoupled delayed-city
  templates;
- MC006 V27 supports hidden-state probing or intervention;
- the V27 expanded bank is transfer-ready;
- MC006 V28 supports hidden-state probing or intervention;
- one transfer-ready template is enough to repair the MC006 source/transfer
  behavior bank;
- ordinary delayed-city transfer-template repair remains a live mechanism
  promotion route;
- more prompt variants under the same greedy binary first-token interface are
  likely to rescue strict final-margin overlap;
- final-output controls can be ignored because V23 diagnosed a sign barrier;
- KSQ001 is a behavior-ready knowledge substrate;
- KSQ001 licenses hidden-state probing or intervention;
- KSQ002 is a behavior-ready source-rewrite or source-channel mechanism
  substrate;
- KSQ002 licenses hidden-state probing or intervention;
- KSQ002 proves source-rewrite invariance rather than a source-disjoint
  holdout parseability boundary;
- KSQ003 is a behavior-ready bridge substrate;
- KSQ003 licenses hidden-state probing or intervention;
- KSQ005 is a behavior-ready grounded-answerability or uncertainty substrate;
- KSQ005 licenses hidden-state probing or intervention;
- KSQ005 demonstrates a real uncertainty, refusal, or correction-control
  surface;
- KSQ006 is a behavior-ready context-support or uncertainty substrate;
- KSQ006 licenses hidden-state probing or intervention;
- KSQ006 demonstrates a real context-support, uncertainty, refusal, or
  correction-control surface;
- KSQ005/KSQ006 relation-evidence answerability is behavior-ready;
- KSQ005/KSQ006 relation-evidence answerability licenses hidden-state probing
  or intervention;
- KSQ005/KSQ006 relation-evidence answerability proves uncertainty,
  context-support, refusal, correction, or knowledge-control behavior;
- KSQ007 is a behavior-ready answerability, uncertainty, refusal, factuality,
  or knowledge-control substrate;
- KSQ007 licenses hidden-state probing or intervention;
- KSQ007 proves that removing real-world priors is enough to solve
  answerability controls;
- KSQ007B repairs KSQ007 into a behavior-ready answerability substrate;
- KSQ007B licenses hidden-state probing, intervention, or a mechanism claim;
- KSQ007B proves that section labels alone solve claim-channel leakage;
- KSQ008 neutral-evidence repair is behavior-ready;
- KSQ008 licenses hidden-state probing, intervention, or a mechanism claim;
- KSQ009 schema-specific value lookup is behavior-ready;
- KSQ009 licenses hidden-state probing, intervention, or a mechanism claim;
- KSQ010 two-stage codebook value lookup is behavior-ready;
- KSQ010 licenses hidden-state probing, intervention, or a mechanism claim;
- KSQ011 answer-for syntax ablation repairs the answer-channel boundary;
- KSQ011 licenses hidden-state probing, intervention, or a mechanism claim;
- KSQ012 wrapper repair quarantines function-like assignment syntax;
- KSQ012 licenses hidden-state probing, intervention, or a mechanism claim;
- KSQ013 proves nonfunction value representations are safe repair surfaces;
- KSQ013 promotes catalog slash text without a slash-only locality control;
- KSQ013 licenses hidden-state probing, intervention, or a mechanism claim;
- KSQ014 is behavior-ready;
- KSQ014 proves bare slash notation is a clean repair surface;
- KSQ014 promotes catalog slash notation without a full-source behavior run;
- KSQ014 licenses hidden-state probing, intervention, or a mechanism claim;
- KSQ015 is behavior-ready;
- KSQ015 proves catalog slash notation is a reliable repair surface;
- KSQ015 licenses hidden-state probing, intervention, or a mechanism claim;
- KSQ015's clean catalog slash-only controls overcome the failed exact bridge
  positive control;
- KSQ004 is a learned-memory bridge, internal mechanism, or knowledge-control
  substrate;
- KSQ004 licenses hidden-state claims, intervention, or mechanism promotion;
- KSQ004 proves that answer-interface matching explains or rules out the
  bridge; the first-run result was template-fragility evidence, and the
  second-wave result is behavior-only relation-format evidence, not a promoted
  mechanism route;
- the executed KSQ outcome matrix licenses hidden-state probing or
  intervention on any KSQ row;
- six passed KSQ structural gates imply a behavior-ready knowledge substrate;
- the KSQ outcome matrix proves a knowledge vector or knowledge-control
  surface;
- the current control surface is deployable.

## Current Bounded Mechanism Card

The current bounded mechanism card is MC005, not MC006.

MC005 has:

- clear behavior;
- source-value internal signature;
- causal intervention;
- repeated holdouts;
- layout and lexicon stress;
- pair-count stress;
- source/distractor/random controls;
- null diagnostics;
- localization to layers 24-26;
- exact final-query attention-write mediation.

Its boundary:

- strict answer-absent write-replacement null locality is not fully reliable.

The closeout status is:

> `research/cards/MC005_ASSOCIATIVE_LOOKUP_BOUNDED_MECHANISM_CLOSEOUT_STATUS.md`

The executable write-route closeout is:

> `research/cards/MC005_WRITE_REPLACEMENT_CLOSEOUT_STATUS.md`

MC005 should stay frozen as `MC005_BOUNDED_ATTENTION_WRITE_MEDIATION` unless
future work is a materially different intervention, a matched transfer/null
panel, or a comparative baseline for a new bridge family.

## Current Knowledge-Like Diagnostic Chain

The current knowledge-like chain is MC006 V14/V16/V17/V18/V19/V20/V21/V22/V23/V24/V25/V26/V27/V28.
It is not a mechanism-card candidate. It is a closed monitor-only diagnostic
route showing how a capital-fact override behavior can remain behavior-supported
and internally monitorable while each ordinary promotion route fails.

The executable predecision-frontier closeout is:

> `research/cards/MC006_PREDECISION_FRONTIER_CLOSEOUT_STATUS.md`

V10 gives the behavior substrate:

- same chat-rendered prompt family;
- no true-capital prompt leakage;
- real-world mode works;
- fictional-code mode works;
- all 5 holdout sources are clean.

V11 blocks intervention:

- hidden direction is perfect;
- output margin is perfect;
- requested-mode baseline is perfect;
- shuffled-label selection p95 is perfect.

V12 tested a better matched surface:

- every row says to ignore the fictional codebook;
- every row asks for the real-world capital;
- the hidden direction is perfect;
- but holdout balance is weak;
- candidate scoring, output margin, and shuffled-label selection are also
  perfect.

V13 tested the missing generated-answer matched-surface repair:

- all 40 V2 sources;
- six matched templates;
- strict generated-answer parsing;
- discovery/calibration-only template selection;
- selected template `fake_mapping_warning`;
- holdout balance passed with 6 true and 2 override rows;
- binary volume failed at 29/40 against a 30/40 floor.

V14 repaired the behavior substrate narrowly:

- strict NFKD parser normalization changed only three rows;
- no candidate label changed into another candidate label;
- selected template remained `fake_mapping_warning`;
- binary volume passed at 30/40;
- holdout balance remained 6 true and 2 override.

V15 then tested hidden monitoring:

- selected `layer_16` final-prompt-token direction;
- discovery and holdout AUC were both 1.000;
- candidate-score and next-token output margins were also 1.000 on holdout;
- diagnostic class: `candidate_score_confounded`.

V16 moved the signature earlier:

- selectable pre-output positions were the line boundaries before final
  `Answer:`;
- selected candidate: `after_mapping_line/layer_4`;
- discovery and holdout AUC were both 1.000;
- same-position output-margin holdout AUC was 0.917;
- shuffled-label selection p95 was 0.917;
- candidate-score and final next-token output margins were still 1.000 on
  holdout;
- diagnostic class:
  `leadtime_signal_supported_but_output_global_confounded`.

V17 then tested the first causal route from that signal:

- intervention: additive residual steering at `after_mapping_line/layer_4`;
- doses: 1.0, 2.0, 4.0 standardized-signature score shifts;
- baseline reproduction: 8/8 holdout labels matched V14;
- strongest directional dose: 4.0;
- `plus_selected` holdout margin delta: +0.1250;
- `minus_selected` holdout margin delta: -0.1406;
- max control absolute effect: 0.1875;
- holdout generated-label changes in the predicted direction: 0;
- diagnostic class: `intervention_failed`.

V18 then tested the same-table margin-matching route:

- selected early signal reproduced: `after_mapping_line/layer_4`, holdout AUC
  1.000;
- candidate-score margin holdout AUC: 1.000;
- final next-token output-margin holdout AUC: 1.000;
- candidate-score holdout margin overlap: false;
- final-output holdout margin overlap: false;
- matched true/override holdout pairs at z <= 0.5: 0 for both margins;
- hidden residual after both global margins: holdout AUC 0.333;
- diagnostic class: `global_margin_separation_blocks_matching`.

V19 then tested an expanded overlapping-margin behavior table:

- prompt bank: 400 rows, 40 sources, 10 prompt variants;
- selected template: `separate_task_weak`;
- binary rows: 35/40;
- non-holdout true / override: 11 / 16;
- holdout true / override: 4 / 4;
- holdout side rows: 0;
- holdout candidate-score margin overlap: false;
- holdout final-output margin overlap: false;
- holdout candidate pairs at z <= 0.5: 4;
- holdout final-output pairs at z <= 0.5: 4;
- diagnostic class: `non_holdout_candidate_margin_overlap_failed`.

V20 then audited strict-overlap selection inside the V19 bank:

- binary rows in pooled bank: 257;
- pooled non-holdout candidate overlap: true;
- pooled non-holdout final-output overlap: false;
- pooled holdout candidate overlap: false;
- pooled holdout final-output overlap: false;
- any single-template full strict-overlap gate: false;
- any template final-output overlap: false;
- pooled holdout joint candidate/final pairs at z <= 0.5: 28;
- selected-template holdout joint pairs at z <= 0.5: 3;
- diagnostic class: `strict_final_margin_overlap_absent`.

V21 then audited the bounded approximate-pair route:

- non-holdout joint pairs at z <= 0.5: 451;
- holdout joint pairs at z <= 0.5: 28;
- selected hidden candidate: `after_mapping_line/layer_14`;
- selected hidden holdout pair accuracy: 0.2857;
- candidate-score margin holdout pair accuracy: 1.0000;
- V19 final next-token margin holdout pair accuracy: 1.0000;
- final-prompt next-token margin holdout pair accuracy: 1.0000;
- diagnostic class: `approximate_pair_matching_failed_margin_baselines`.

V22 then mapped the source/path lead-time curve inside the same row bank:

- source/path positions: `mapping_country_token`, `mapping_value_token`,
  `question_country_token`;
- later pre-output positions: `after_mapping_line`, `after_instruction_line`,
  `after_question_line`, `after_return_line`;
- final position: `final_prompt_token`;
- mapped rows: 257/257;
- non-holdout joint pairs at z <= 0.5: 451;
- holdout joint pairs at z <= 0.5: 28;
- best source/path pair candidate: `question_country_token/layer_20`, holdout
  AUC 0.800, holdout pair accuracy 0.607;
- best source/path AUC candidate: `question_country_token/layer_22`, holdout
  AUC 0.832, holdout pair accuracy 0.714;
- stronger later pre-output monitors appeared at `after_instruction_line`,
  `after_question_line`, and `after_return_line`;
- candidate-score margin holdout pair accuracy: 1.0000;
- V19 final next-token margin holdout pair accuracy: 1.0000;
- final-prompt next-token margin holdout pair accuracy: 1.0000;
- diagnostic class: `source_path_final_margin_shadow`.

V23 then audited the final-margin gate itself:

- V18 / V14 selected binary rows: 30;
- V18 final next-token sign prediction accuracy: 1.000;
- V18 final raw overlap: false;
- V18 final raw gap: 1.500;
- V19 all binary rows: 257;
- V19 final next-token sign prediction accuracy: 1.000;
- V19 final raw overlap: false;
- V19 final raw gap: 0.375;
- V19 selected-template final sign prediction accuracy: 1.000;
- V19 all-binary candidate-score sign prediction accuracy: 0.953;
- V19 all-binary candidate-score raw overlap: true;
- V14 selected first-token alignment: 30/30;
- V19 selected-template first-token alignment: 35/35;
- V19 all-binary first-token alignment: 255/257;
- diagnostic class: `greedy_final_margin_sign_barrier`.

V24 then changed the answer interface:

- answer format: delayed JSON city value;
- source rows: 400;
- selected template: `game_code_then_geo`;
- selected rows: 40;
- binary rows: 31;
- binary label counts: 10 true / 21 override;
- source-disjoint holdout labels: 2 true / 4 override;
- selected-city first-token rate: 0.000;
- final city-token margin sign accuracy: 0.516;
- final city-token raw overlap: true;
- city candidate-score sign accuracy: 0.645;
- JSON candidate-score sign accuracy: 0.742;
- best hidden monitor: `after_mapping_line/layer_0`, discovery AUC 1.000,
  holdout AUC 0.875;
- JSON-completion candidate-score holdout AUC: 1.000;
- diagnostic class: `delayed_city_interface_decouples_first_token_margin`.

V24 repaired the narrow V23 interface failure, but did not produce a
mechanism-grade signature. The next knowledge-like step should not repeat simple
additive steering on the V16 direction, search for another scalar classifier on
the same V14 table, broaden prompt generation under the same greedy binary
first-token interface, or steer from the V24 hidden monitor. The existing
V19/V20/V21/V22/V23/V24 route is now closed for strict-overlap selection,
approximate pair-matching rescue, source/path final-margin rescue,
same-interface strict final-overlap row generation, and delayed-city
first-token repair as a promotion route.

V25 then selected a candidate-score-decoupled delayed-city template:

- source bank: V24 delayed-city generated rows;
- selected template: `untrusted_note_real`;
- binary rows: 37;
- binary label counts: 24 true / 13 override;
- source-disjoint holdout labels: 4 true / 3 override;
- JSON-completion candidate-score holdout AUC: 0.333;
- final next-token city margin holdout AUC: 0.417 in the hidden-screen audit;
- city candidate-score margin holdout AUC: 0.250 in the hidden-screen audit;
- best hidden monitor: `after_mapping_line/layer_10`, discovery AUC 1.000,
  holdout AUC 1.000;
- shuffled-label selected-search p95: discovery AUC 1.000, holdout AUC 1.000;
- diagnostic class: `candidate_decoupled_hidden_shuffle_overfit`.

V25 is the right kind of negative. It shows that V24's candidate-score blocker
was not universal across the delayed-city prompt bank. MC006 can produce a
candidate-score-decoupled delayed-city behavior table. But it also shows that a
small candidate-decoupled table plus flexible layer/position selection can still
manufacture perfect hidden holdout AUC under shuffled labels. The next MC006
route must preregister or regularize hidden selection, or broaden the
candidate-decoupled table, before any intervention is justified.

V26 then tested the obvious regularization:

- locked coordinate: V25 `after_mapping_line/layer_10`;
- source template: `untrusted_note_real`;
- transfer template: `separate_task_weak`;
- source train AUC: 1.000 on 30 non-holdout rows;
- source holdout AUC: 0.750 on 7 rows;
- transfer holdout AUC: 0.500 on 5 rows;
- best transfer control: `after_mapping_line` next-token city margin at 0.833
  holdout AUC;
- fixed-coordinate train-label shuffle p95: 0.833;
- diagnostic class: `locked_coordinate_transfer_failed`.

V26 kills the simple "freeze the V25 coordinate" repair route. The coordinate is
not a reusable MC006 knowledge surface across the candidate-decoupled delayed
templates already present in the V24 bank. The next MC006 route must either
build a materially larger candidate-decoupled bank or move to a different
intervention family; more work on this coordinate is not justified.

V27 then tested the larger-bank route:

- generated rows: 640;
- sources: 40;
- predeclared source-role templates: 8;
- predeclared transfer-role templates: 8;
- candidate-decoupled templates: 4;
- source-ready templates: 3;
- transfer-ready templates: 1;
- pooled binary rows across ready templates: 138;
- pooled non-holdout labels: 59 true / 51 override;
- pooled holdout labels: 14 true / 14 override;
- diagnostic class: `expanded_candidate_decoupled_bank_insufficient`.

Ready templates:

- `source_untrusted_note_real_v24`;
- `source_unverified_note_check`;
- `source_mistaken_source`;
- `transfer_separate_task_weak_v24`.

V27 shows the current prompt family can repair the source side of the
candidate-decoupled bank but not the transfer side. This means MC006 should not
probe hidden states on V27.

V28 then tested that narrower transfer-role repair:

- generated rows: 640;
- sources: 40;
- predeclared transfer-role templates: 16;
- transfer-ready templates: 1;
- selected ready transfer template: `transfer_sandbox_mapping_then_geo`;
- ready transfer binary rows: 35;
- ready transfer holdout labels: 4 true / 4 override;
- ready transfer JSON candidate-score holdout AUC: 0.1875;
- ready transfer final-city holdout AUC: 0.3125;
- combined V27+V28 ready templates: 4;
- combined pooled binary rows: 143;
- combined pooled holdout labels: 16 true / 15 override;
- diagnostic class: `transfer_role_repair_bank_insufficient`.

V28 shows that transfer-role candidate decoupling is possible, but brittle. One
predeclared transfer prompt passed, while the bank gate required at least two.
The current delayed-city MC006 branch should therefore not proceed to hidden
states. It has now failed flexible hidden selection, locked-coordinate transfer,
expanded source/transfer bank construction, and direct transfer-role repair.
It is now formally closed as `DELAYED_CITY_ROUTE_CLOSED_MONITOR_ONLY`.

Future MC006 work must either be explicitly framed as a known-confounded causal
stress test, with no mechanism-promotion language, or change the behavior
family/prompt contract materially enough that V14-V28 are no longer the route
being repaired.

## Practical Research Doctrine Going Forward

The project should keep doing the following:

1. Start with behavior gates.
2. Refuse hidden-state work on weak behavior substrates.
3. Always include output/logit baselines.
4. Always include prompt-format or requested-mode baselines when labels align
   with visible prompt text.
5. Always include shuffled-label selection nulls when selecting among layers,
   heads, features, or positions.
6. Separate discovery, calibration, and holdout splits.
7. Preserve source-disjoint holdouts when possible.
8. Treat intervention side effects as first-class results.
9. Record failure modes rather than hiding them.
10. Avoid promoting synthetic lookup results into general knowledge claims.
11. Avoid calling behavior control a mechanism.
12. Prefer narrow honest mechanism cards over broad impressive claims.

## What The Reviewer Comments Changed

The reviewer comments are not being treated as a review appendix. They changed
the operating theory of the project.

The old temptation was:

> Find a clean mechanism card, then collect more cards until the genome appears.

The stronger operating theory is:

> Measure the law that determines why mechanism cards pass, fail, or become
> bounded.

That changes what counts as progress. A passing mechanism card is still
valuable, but it is no longer the only prestigious artifact. A killed result is
also valuable when it has a typed death: output-margin confound,
candidate-score confound, prompt-format confound, prompt-authority collapse,
shuffle overfit, null-row fragility, source-deletion ambiguity, transfer
failure, or model-size fragility. The repeated death pattern is part of the
genome.

The main reviewer-derived insights now embedded in the research spine are:

1. The mixture is the object. The project should estimate how much of a
   behavior lives in prompt authority, source tokens, output geometry,
   candidate-score geometry, internal paths, null margins, and model-scale
   quirks. A single circuit is one row in that mixture, not the whole prize.
2. Controls are instruments, not only obstacles. When final output margin kills
   a hidden-state result, that may be a real finding about when the behavior
   became decided, not merely a failed probe.
3. Lead-time is a first-class axis. The project should ask whether an internal
   signal appears before the output interface exposes the label, whether that
   lead survives nulls and shuffled-label selection, and whether it transfers.
4. Typed failures must be reusable. `OUTPUT_MARGIN_CONFUND`,
   `CANDIDATE_SCORE_CONFUND`, `SHUFFLED_SELECTION_OVERFIT`,
   `NULL_ROW_LOW_MARGIN_FLIP`, and similar labels should become diagnostic
   classes in the atlas, not prose footnotes.
5. Depth needs width tests. A route can be deepened only for a bounded number
   of serious repairs before it must either widen, become bounded, or die.
6. Promotion and death should both be preregistered. Each branch should state
   what upgrades the claim, what kills the route, what narrower claim survives,
   and what diagnostic lesson is exported.
7. The cross-family table is the deliverable. The atlas should make every
   behavior family comparable by behavior gate, lead-time, output/candidate
   visibility, locality, intervention, null reliability, transfer, and verdict.

This is now implemented, not just stated: `code/control_surface_comparison.py`
generates `data/control_surface_comparison.json`, and atlas validation fails if
that cross-family comparison is stale.

This is why MC009 was preregistered instead of immediately searching for a new
hidden classifier. MC008 showed compact symbolic answers can repair direct
controls and nulls, but the direct `entity -> artificial code` row still made
the conflict table overwhelmingly prompt-local. MC009 falsified the simplest
repair: removing the printed source-value pair and deriving the task answer
from row position did not produce a hidden-state-ready bridge. The membership
contract preserved controls while failing conflict balance; typed slots created
balance while breaking controls and making the answer channel prompt-visible.

## The Deep Pattern

The deepest pattern so far is:

> Small LLMs expose many internal and prompt-level surfaces that can steer
> behavior, but the surfaces most responsible for robust behavior are often
> broad, distributed, prompt-coupled, and output-visible.

This does not make interpretability hopeless. It changes the target.

Instead of searching for one "truth vector" or one "capital-fact circuit," the
project is discovering a layered control map:

- when behavior is prompt-authority driven;
- when behavior is output-margin visible;
- when source tokens dominate;
- when source removal and prompt rewriting produce the same effect;
- when late attention writes mediate lookup;
- when row-level structure is predictable but not steerable;
- when null rows are fragile because margins are low.

That is the knowledge genome beginning to take shape.

## Control-Surface Atlas

The next-level artifact is the control-surface atlas:

> `research/20_CONTROL_SURFACE_ATLAS.md`

It now has a machine-readable companion:

> `data/control_surface_atlas.json`

and a validator:

> `python code/validate_control_surface_atlas.py`

The current law-and-gap layer is:

> `research/21_CONTROL_SURFACE_LAWS_AND_GAPS.md`

The law-audit layer is:

> `research/24_CONTROL_SURFACE_LAW_AUDIT.md`

> `code/control_surface_law_audit.py`

> `data/control_surface_law_audit.json`

The next-experiment queue layer is:

> `research/25_CONTROL_SURFACE_NEXT_EXPERIMENT_QUEUE.md`

> `code/control_surface_next_queue.py`

> `data/control_surface_next_experiment_queue.json`

The smoke-diagnostic layer is:

> `research/26_CONTROL_SURFACE_SMOKE_DIAGNOSTICS.md`

> `code/control_surface_smoke_diagnostics.py`

> `data/control_surface_smoke_diagnostics.json`

The bridge-ladder layer is:

> `research/27_CONTROL_SURFACE_BRIDGE_LADDER.md`

> `code/control_surface_bridge_ladder.py`

> `data/control_surface_bridge_ladder.json`

The mixture-law layer is:

> `research/28_CONTROL_SURFACE_MIXTURE_LAW.md`

> `code/control_surface_mixture_law.py`

> `data/control_surface_mixture_law.json`

The compositional-genome audit layer is:

> `research/40_CONTROL_SURFACE_COMPOSITIONAL_GENOME_AUDIT.md`

> `code/control_surface_compositional_genome_audit.py`

> `data/control_surface_compositional_genome_audit.json`

The family-matrix layer is:

> `research/41_CONTROL_SURFACE_FAMILY_MATRIX.md`

> `code/control_surface_family_matrix.py`

> `data/control_surface_family_matrix.json`

The knowledge-ladder layer is:

> `research/42_CONTROL_SURFACE_KNOWLEDGE_LADDER.md`

> `code/control_surface_knowledge_ladder.py`

> `data/control_surface_knowledge_ladder.json`

The knowledge-gap-plan layer is:

> `research/43_CONTROL_SURFACE_KNOWLEDGE_GAP_PLAN.md`

> `code/control_surface_knowledge_gap_plan.py`

> `data/control_surface_knowledge_gap_plan.json`

The knowledge-substrate-admission layer is:

> `research/44_CONTROL_SURFACE_KNOWLEDGE_SUBSTRATE_ADMISSION.md`

> `code/control_surface_knowledge_substrate_admission.py`

> `data/control_surface_knowledge_substrate_admission.json`

The knowledge-candidate-queue layer is:

> `research/45_CONTROL_SURFACE_KNOWLEDGE_CANDIDATE_QUEUE.md`

> `code/control_surface_knowledge_candidate_queue.py`

> `data/control_surface_knowledge_candidate_queue.json`

The knowledge-first-run-pack layer is:

> `research/46_CONTROL_SURFACE_KNOWLEDGE_FIRST_RUN_PACK.md`

> `code/control_surface_knowledge_first_run_pack.py`

> `data/control_surface_knowledge_first_run_pack.json`

The executed knowledge-first-run-outcomes layer is:

> `research/47_CONTROL_SURFACE_KNOWLEDGE_FIRST_RUN_OUTCOMES.md`

> `code/control_surface_knowledge_first_run_outcomes.py`

> `data/control_surface_knowledge_first_run_outcomes.json`

The knowledge-failure-topology layer is:

> `research/48_CONTROL_SURFACE_KNOWLEDGE_FAILURE_TOPOLOGY.md`

> `code/control_surface_knowledge_failure_topology.py`

> `data/control_surface_knowledge_failure_topology.json`

The knowledge-second-wave-outcomes layer is:

> `research/49_CONTROL_SURFACE_KNOWLEDGE_SECOND_WAVE_OUTCOMES.md`

> `code/control_surface_knowledge_second_wave_outcomes.py`

> `data/control_surface_knowledge_second_wave_outcomes.json`

The knowledge-third-wave-outcomes layer is:

> `research/50_CONTROL_SURFACE_KNOWLEDGE_THIRD_WAVE_OUTCOMES.md`

> `code/control_surface_knowledge_third_wave_outcomes.py`

> `data/control_surface_knowledge_third_wave_outcomes.json`

The knowledge-fourth-wave-outcomes layer is:

> `research/51_CONTROL_SURFACE_KNOWLEDGE_FOURTH_WAVE_OUTCOMES.md`

> `code/control_surface_knowledge_fourth_wave_outcomes.py`

> `data/control_surface_knowledge_fourth_wave_outcomes.json`

The knowledge-fifth-wave-outcomes layer is:

> `research/52_CONTROL_SURFACE_KNOWLEDGE_FIFTH_WAVE_OUTCOMES.md`

> `code/control_surface_knowledge_fifth_wave_outcomes.py`

> `data/control_surface_knowledge_fifth_wave_outcomes.json`

The knowledge-sixth-wave-outcomes layer is:

> `research/53_CONTROL_SURFACE_KNOWLEDGE_SIXTH_WAVE_OUTCOMES.md`

> `code/control_surface_knowledge_sixth_wave_outcomes.py`

> `data/control_surface_knowledge_sixth_wave_outcomes.json`

The knowledge-seventh-wave-outcomes layer is:

> `research/54_CONTROL_SURFACE_KNOWLEDGE_SEVENTH_WAVE_OUTCOMES.md`

> `code/control_surface_knowledge_seventh_wave_outcomes.py`

> `data/control_surface_knowledge_seventh_wave_outcomes.json`

The knowledge-eighth-wave-outcomes layer is:

> `research/55_CONTROL_SURFACE_KNOWLEDGE_EIGHTH_WAVE_OUTCOMES.md`

> `code/control_surface_knowledge_eighth_wave_outcomes.py`

> `data/control_surface_knowledge_eighth_wave_outcomes.json`

The knowledge-ninth-wave-outcomes layer is:

> `research/56_CONTROL_SURFACE_KNOWLEDGE_NINTH_WAVE_OUTCOMES.md`

> `code/control_surface_knowledge_ninth_wave_outcomes.py`

> `data/control_surface_knowledge_ninth_wave_outcomes.json`

The knowledge-tenth-wave-outcomes layer is:

> `research/57_CONTROL_SURFACE_KNOWLEDGE_TENTH_WAVE_OUTCOMES.md`

> `code/control_surface_knowledge_tenth_wave_outcomes.py`

> `data/control_surface_knowledge_tenth_wave_outcomes.json`

The knowledge-eleventh-wave-outcomes layer is:

> `research/58_CONTROL_SURFACE_KNOWLEDGE_ELEVENTH_WAVE_OUTCOMES.md`

> `code/control_surface_knowledge_eleventh_wave_outcomes.py`

> `data/control_surface_knowledge_eleventh_wave_outcomes.json`

The KSQ008 neutral-evidence channel repair is:

> `research/cards/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR_STATUS.md`

> `research/prereg/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR.md`

> `code/ksq008_neutral_evidence_channel_repair.py`

> `results/cards/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR/ksq008_neutral_evidence_channel_repair_first_run.json`

> `results/cards/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR/ksq008_neutral_evidence_channel_repair_smoke_limit10.json`

The KSQ009 schema-specific value lookup repair is:

> `research/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP_STATUS.md`

> `research/prereg/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP.md`

> `code/ksq009_schema_specific_value_lookup.py`

> `results/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP/ksq009_schema_specific_value_lookup_first_run.json`

> `results/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP/ksq009_schema_specific_value_lookup_smoke_limit10.json`

The KSQ010 two-stage codebook value lookup repair is:

> `research/cards/KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP_STATUS.md`

> `research/prereg/KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP.md`

> `code/ksq010_two_stage_codebook_value_lookup.py`

> `results/cards/KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP/ksq010_two_stage_codebook_value_lookup_first_run.json`

> `results/cards/KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP/ksq010_two_stage_codebook_value_lookup_smoke_limit10.json`

The KSQ011 answer-for syntax ablation is:

> `research/cards/KSQ011_ANSWER_FOR_SYNTAX_ABLATION_STATUS.md`

> `research/prereg/KSQ011_ANSWER_FOR_SYNTAX_ABLATION.md`

> `code/ksq011_answer_for_syntax_ablation.py`

> `results/cards/KSQ011_ANSWER_FOR_SYNTAX_ABLATION/ksq011_answer_for_syntax_ablation_first_run.json`

> `results/cards/KSQ011_ANSWER_FOR_SYNTAX_ABLATION/ksq011_answer_for_syntax_ablation_smoke_limit10.json`

The KSQ012 function-assignment wrapper repair is:

> `research/cards/KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR_STATUS.md`

> `research/prereg/KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR.md`

> `code/ksq012_function_assignment_wrapper_repair.py`

> `results/cards/KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR/ksq012_function_assignment_wrapper_repair_first_run.json`

> `results/cards/KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR/ksq012_function_assignment_wrapper_repair_smoke_limit10.json`

The KSQ013 nonfunction representation screen is:

> `research/cards/KSQ013_NONFUNCTION_REPRESENTATION_SCREEN_STATUS.md`

> `research/prereg/KSQ013_NONFUNCTION_REPRESENTATION_SCREEN.md`

> `code/ksq013_nonfunction_representation_screen.py`

> `results/cards/KSQ013_NONFUNCTION_REPRESENTATION_SCREEN/ksq013_nonfunction_representation_screen_first_run.json`

> `results/cards/KSQ013_NONFUNCTION_REPRESENTATION_SCREEN/ksq013_nonfunction_representation_screen_smoke_limit10.json`

The KSQ014 slash locality packet is:

> `research/cards/KSQ014_SLASH_LOCALITY_PACKET_STATUS.md`

> `research/prereg/KSQ014_SLASH_LOCALITY_PACKET.md`

> `code/ksq014_slash_locality_packet.py`

> `results/cards/KSQ014_SLASH_LOCALITY_PACKET/ksq014_slash_locality_packet_first_run.json`

> `results/cards/KSQ014_SLASH_LOCALITY_PACKET/ksq014_slash_locality_packet_smoke_limit10.json`

The KSQ015 catalog slash full-source packet is:

> `research/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET_STATUS.md`

> `research/prereg/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET.md`

> `code/ksq015_catalog_slash_full_source_packet.py`

> `results/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET/ksq015_catalog_slash_full_source_first_run.json`

> `results/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET/ksq015_catalog_slash_full_source_full_behavior.json`

The first executed KSQ001 familiar-entity prior-counterbalance gate is:

> `research/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN_STATUS.md`

> `research/prereg/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN.md`

> `code/ksq001_familiar_entity_prior_counterbalance_first_run.py`

> `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_first_run.json`

> `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_smoke_limit10.json`

> `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_full_behavior.json`

The executed KSQ001 familiar-prior parseability bound is:

> `research/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND_STATUS.md`

> `research/prereg/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND.md`

> `code/ksq001_familiar_prior_parseability_bound.py`

> `results/cards/KSQ001_FAMILIAR_PRIOR_PARSEABILITY_BOUND/ksq001_familiar_prior_parseability_bound_full_behavior.json`

The first executed KSQ002 familiar-entity source-rewrite equivalence structural,
smoke, and full behavior gate is:

> `research/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN_STATUS.md`

> `research/prereg/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN.md`

> `code/ksq002_familiar_entity_source_rewrite_equivalence_first_run.py`

> `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_first_run.json`

> `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_smoke_limit10.json`

> `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_full_behavior.json`

The executed KSQ002 source-rewrite holdout repair is:

> `research/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR_STATUS.md`

> `research/prereg/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR.md`

> `code/ksq002_source_rewrite_holdout_repair.py`

> `results/cards/KSQ002_SOURCE_REWRITE_HOLDOUT_REPAIR/ksq002_source_rewrite_holdout_repair_full_behavior.json`

The first executed KSQ003 knowledge-candidate structural and smoke gate is:

> `research/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN_STATUS.md`

> `research/prereg/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN.md`

> `code/ksq003_bridge_statusless_evidence_aggregation_first_run.py`

> `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_first_run.json`

> `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_smoke_limit10.json`

The executed KSQ003 evidence-sufficiency redesign is:

> `research/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN_STATUS.md`

> `research/prereg/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN.md`

> `code/ksq003_evidence_sufficiency_redesign.py`

> `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_first_run.json`

> `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_smoke_limit10.json`

> `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_full_behavior.json`

The first executed KSQ004 bridge answer-interface minimal-pairs structural,
smoke, and full behavior gate is:

> `research/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN_STATUS.md`

> `research/prereg/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN.md`

> `code/ksq004_bridge_answer_interface_minimal_pairs_first_run.py`

> `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_first_run.json`

> `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_smoke_limit10.json`

> `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_full_behavior.json`

The executed KSQ004 template-invariance adjudication is:

> `research/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION_STATUS.md`

> `research/prereg/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION.md`

> `code/ksq004_template_invariance_adjudication.py`

> `results/cards/KSQ004_TEMPLATE_INVARIANCE_ADJUDICATION/ksq004_template_invariance_adjudication_full_behavior.json`

The first executed KSQ005 grounded-answerability structural and smoke gate is:

> `research/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN_STATUS.md`

> `research/prereg/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN.md`

> `code/ksq005_uncertainty_grounded_answerability_first_run.py`

> `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_first_run.json`

> `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_smoke_limit10.json`

The first executed KSQ006 context-support counterfactual structural and smoke
gate is:

> `research/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN_STATUS.md`

> `research/prereg/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN.md`

> `code/ksq006_uncertainty_context_support_counterfactuals_first_run.py`

> `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_first_run.json`

> `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_smoke_limit10.json`

The executed KSQ005/KSQ006 relation-evidence answerability redesign is:

> `research/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY_STATUS.md`

> `research/prereg/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY.md`

> `code/ksq005_006_relation_evidence_answerability_redesign.py`

> `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_first_run.json`

> `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_smoke_limit10.json`

> `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_full_behavior.json`

The executed KSQ007 nonce-evidence answerability calibrator is:

> `research/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY_STATUS.md`

> `research/prereg/KSQ007_NONCE_EVIDENCE_ANSWERABILITY.md`

> `code/ksq007_nonce_evidence_answerability_calibrator.py`

> `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_first_run.json`

> `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_smoke_limit10.json`

The executed KSQ007B claim-channel boundary audit is:

> `research/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY_STATUS.md`

> `research/prereg/KSQ007_CLAIM_CHANNEL_BOUNDARY.md`

> `code/ksq007_claim_channel_boundary_audit.py`

> `results/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY/ksq007_claim_channel_boundary_first_run.json`

> `results/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY/ksq007_claim_channel_boundary_smoke_limit10.json`

The decision-frontier layer is:

> `research/29_CONTROL_SURFACE_DECISION_FRONTIER.md`

> `code/control_surface_decision_frontier.py`

> `data/control_surface_decision_frontier.json`

The route-disposition layer is:

> `research/30_CONTROL_SURFACE_ROUTE_DISPOSITION.md`

> `code/control_surface_route_disposition.py`

> `data/control_surface_route_disposition.json`

The transfer-matrix layer is:

> `research/31_CONTROL_SURFACE_TRANSFER_MATRIX.md`

> `code/control_surface_transfer_matrix.py`

> `data/control_surface_transfer_matrix.json`

The reliability-matrix layer is:

> `research/32_CONTROL_SURFACE_RELIABILITY_MATRIX.md`

> `code/control_surface_reliability_matrix.py`

> `data/control_surface_reliability_matrix.json`

The error-taxonomy layer is:

> `research/33_CONTROL_SURFACE_ERROR_TAXONOMY.md`

> `code/control_surface_error_taxonomy.py`

> `data/control_surface_error_taxonomy.json`

The gate-geometry layer is:

> `research/34_CONTROL_SURFACE_GATE_GEOMETRY.md`

> `code/control_surface_gate_geometry.py`

> `data/control_surface_gate_geometry.json`

The genome-snapshot layer is:

> `research/35_CONTROL_SURFACE_GENOME_SNAPSHOT.md`

> `code/control_surface_genome_snapshot.py`

> `data/control_surface_genome_snapshot.json`

The axis-interactions layer is:

> `research/36_CONTROL_SURFACE_AXIS_INTERACTIONS.md`

> `code/control_surface_axis_interactions.py`

> `data/control_surface_axis_interactions.json`

The coverage-gaps layer is:

> `research/37_CONTROL_SURFACE_COVERAGE_GAPS.md`

> `code/control_surface_coverage_gaps.py`

> `data/control_surface_coverage_gaps.json`

The gap-closure-plan layer is:

> `research/38_CONTROL_SURFACE_GAP_CLOSURE_PLAN.md`

> `code/control_surface_gap_closure_plan.py`

> `data/control_surface_gap_closure_plan.json`

The artifact-registry layer is:

> `research/22_ARTIFACT_REGISTRY.md`

> `code/control_surface_artifacts.py`

> `data/control_surface_artifact_index.json`

The cross-family comparison layer is:

> `research/23_CONTROL_SURFACE_COMPARISON.md`

> `code/control_surface_comparison.py`

> `data/control_surface_comparison.json`

The MC005 bounded mechanism closeout is:

> `research/cards/MC005_ASSOCIATIVE_LOOKUP_BOUNDED_MECHANISM_CLOSEOUT_STATUS.md`

The first bridge behavior result is:

> `research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_STATUS.md`

The first bridge prior-pressure result is:

> `research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V2_AUTHORITY_DIAL_STATUS.md`

The first bridge parse-repair result is:

> `research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V3_PARSE_REPAIR_STATUS.md`

The first bridge authority-interface result is:

> `research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_V4_AUTHORITY_INTERFACE_STATUS.md`

The first bridge route closeout is:

> `research/cards/MC007_SEMI_SYNTHETIC_FAMILIAR_ENTITY_LOOKUP_ROUTE_CLOSEOUT_STATUS.md`

The symbolic bridge preregistration, repair, and route closeout are:

> `research/prereg/MC008_SYMBOLIC_FACT_CODE_ARBITRATION.md`

> `research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

> `research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_V2_NULL_AUTHORITY_REPAIR_STATUS.md`

> `research/cards/MC008_SYMBOLIC_FACT_CODE_ARBITRATION_ROUTE_CLOSEOUT_STATUS.md`

The closed derived-code bridge preregistration and smoke status are:

> `research/prereg/MC009_DERIVED_CODE_ARBITRATION.md`

> `research/cards/MC009_DERIVED_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

The runnable MC009 behavior harness is:

> `code/mc009_derived_code_arbitration.py`

The current MC010 two-hop bridge preregistration, structural status, behavior
status, and scaffold are:

> `research/prereg/MC010_TWO_HOP_FACT_CODE_ARBITRATION.md`

> `research/cards/MC010_TWO_HOP_FACT_CODE_ARBITRATION_STRUCTURAL_STATUS.md`

> `research/cards/MC010_TWO_HOP_FACT_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc010_two_hop_fact_code_arbitration.py`

The current MC011 numeric bridge preregistration, behavior status, and scaffold
are:

> `research/prereg/MC011_ATOMIC_NUMBER_CODE_ARBITRATION.md`

> `research/cards/MC011_ATOMIC_NUMBER_CODE_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc011_atomic_number_code_arbitration.py`

The current MC012 reliability-labeled numeric bridge preregistration, behavior
status, and scaffold are:

> `research/prereg/MC012_RELIABILITY_LABELED_NUMERIC_ARBITRATION.md`

> `research/cards/MC012_RELIABILITY_LABELED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc012_reliability_labeled_numeric_arbitration.py`

The current MC013 status-channel ablation preregistration, behavior status, and
scaffold are:

> `research/prereg/MC013_STATUS_CHANNEL_ABLATION_NUMERIC_ARBITRATION.md`

> `research/cards/MC013_STATUS_CHANNEL_ABLATION_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc013_status_channel_ablation_numeric_arbitration.py`

The current MC014 inferred-reliability numeric bridge preregistration,
behavior status, and scaffold are:

> `research/prereg/MC014_INFERRED_RELIABILITY_NUMERIC_ARBITRATION.md`

> `research/cards/MC014_INFERRED_RELIABILITY_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc014_inferred_reliability_numeric_arbitration.py`

The current MC015 parity-gated numeric bridge preregistration, behavior status,
and scaffold are:

> `research/prereg/MC015_PARITY_GATED_NUMERIC_ARBITRATION.md`

> `research/cards/MC015_PARITY_GATED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc015_parity_gated_numeric_arbitration.py`

The current MC016 alphabet-gated numeric bridge preregistration, behavior
status, and scaffold are:

> `research/prereg/MC016_ALPHABET_GATED_NUMERIC_ARBITRATION.md`

> `research/cards/MC016_ALPHABET_GATED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md`

> `code/mc016_alphabet_gated_numeric_arbitration.py`

The atlas reframes the project around the compositional law of behavior rather
than the next passing mechanism card. It treats typed failures as primary data,
lead-time as a measured axis, and controls as instruments that locate where a
behavior lives: prompt authority, source tokens, output geometry, internal
paths, null boundaries, or model-scale fragility.

The artifact registry makes that reframing less dependent on prose. It parses
53 unique result artifacts across all 19 atlas rows, extracts metrics from all 53,
and extracts readiness flags, diagnostics, failed criteria, selected
templates/coordinates, baseline/control fields, null fields, and
intervention/locality fields before a row is allowed to pass validation. The
hard family checks now cover MC001 source-mask/rewrite equivalence, MC001B
raw-versus-residual signal, MC001G intervention non-movement, MC002 behavior
failure, MC002B pressure failure, MC003 shuffle/condition-trace signature
failure, MC004 lead-time shuffle/subgroup failure, MC005 layers-24-26 write
boundary, MC005 lookup/null boundary, MC005 non-simple margin cutoff, MC006
V16 output/candidate confound, MC006 additive-steering failure, MC006 V21/V22
margin-shadow boundaries, MC006 V24/V25/V28 delayed-city route boundaries,
MC006 shuffle/transfer failure, and granular MC007-MC016 bridge-route deaths:
MC007 V1 source lookup without conflict, MC007 V2 authority-dial parseability
failure, MC007 V3 parseability-repair failure, MC007 V4 source-declaration
control failure, MC008 direct-controls-before-null and
null-repaired/conflict-absent boundaries, MC009 membership-control versus
typed-slot-control tradeoffs, MC010 two-hop direct-control and table-dominant
conflict failures, MC011 numeric direct-control success plus complete numeric
conflict collapse, MC012 reliability-labeled contrast success plus
prompt-channel blockage, MC013 statused positive-control reproduction plus
matched-ablation contrast collapse, MC014 direct-control cleanliness plus
calibration-inference conflict collapse, and MC015 direct-control cleanliness
plus parity-gate rule-following failure, and MC016 direct-control cleanliness
plus alphabet-gate local collapse.
The compact index freezes those extracted facts into a smaller machine-readable
surface for downstream atlas comparison.

The comparison layer now performs that downstream aggregation. It reports that
the current atlas has 19 rows, 0 promoted mechanism cards, 1 bounded mechanism
card, 17 diagnostic notes, 1 failed mechanism-card route, and a
0.947368 diagnostic-or-failed ratio. The strongest measured pressures are not
clean local paths: prompt authority is high or dominant in every row, output
geometry is high or dominant in 14 of 19 rows, behavior substrate failure
appears in 7 of 19 rows, and intervention is not allowed or failed in 14 of 19
rows.
That ratio is the reviewer insight made concrete.

The same comparison now carries a claim audit: validation fails if any row lacks
linked result artifacts, extracted metrics, or a family-level claim check, and
it now also fails if any row falls back to partial metric coverage. The current
audit passes those hard gaps for all 19 rows, with no partial-metric row left
in the atlas.

It also carries a claim-consistency audit: validation fails if a row's high-level
verdict, intervention state, lead-time state, null-locality field,
behavior-gate diagnostic, output-confound diagnostic, or signature-causality
diagnostic contradicts the normalized artifact evidence. The current
consistency audit checks 120 conditions and finds 0 contradictions.

The law-audit layer now checks the theory layer itself. It validates 9 law
hypotheses against cited atlas rows and observed diagnostics, covers all 19
atlas rows, cites all 60 currently observed diagnostics at least once, and has
0 hypothesis evidence gaps. It also forced two useful corrections:
`REQUESTED_MODE_CONFUND` was removed from current law evidence because it is
not observed in the atlas, and
`coarse_source_ablation_overstates_circuit_locality` was downgraded from
supported to tentative because current evidence is single-row.

The next-experiment queue now turns those laws into ranked action pressure. It
contains 19 queue items: 5 immediate, 4 high-priority, 9 medium-priority, and 1
watch item.
It is now bridge-closure-aware: it ingests the bridge ladder, route
disposition, and error taxonomy layers, records 24 bridge rungs, 17 smoke
rungs, 0 hidden-state-allowed bridge rungs, 0 clean unconfounded bridge rungs,
24 closed contract axes, and recent closed rungs MC030, MC031, MC032, and
MC033. Any bridge-route item now carries active constraints from those closures
rather than treating local prompt guards, example removal, query ordering,
answer schema edits, arithmetic checksum validity cues, cross-table consistency
cues, or row-local fact-claim cues as fresh routes.
The top pressures are building a bridge where the answer rule is independent of
visible trusted/untrusted source-status text and where outputs follow the
intended rule, the full behavior-substrate gate
for any future bridge, matched transfer/null panels, a stricter transfer-
promotion rule, and source-token locality controls.

MC010 now closes the first two-hop bridge attempt as a diagnostic. It uses a
two-hop source path, `element -> nonce handle -> task code`, so the task code is
available through local source structure without direct entity-to-code rows and
without MC009's row-position answer slot. The full structural manifest passed
on 800 records across 40 sources, 2 templates, and 6 panels. The full generated
behavior run then failed before hidden-state work: selected `neutral_contract`
had 23/40 synthetic task-code rows, 38/40 familiar task-code rows, 12/40
real-symbol rows, 36/40 answer-absent `UNKNOWN` rows, and 229 primary task-code
conflict rows versus 0 real/lure-symbol rows, with output/candidate baselines
reported.

MC011 now closes the first numeric answer-interface bridge attempt as a
diagnostic. It uses prompt-local local lab numbers versus learned real-world
atomic numbers, so both sides answer with short integers. The full generated
behavior run selected `neutral_numeric`, reported output/candidate baselines,
and passed all direct controls: 40/40 synthetic numeric lookup, 40/40 familiar
element numeric lookup, 40/40 real atomic-number recall, and 40/40 answer-absent
`UNKNOWN` rows. The bridge still failed before hidden-state work because primary
conflict was 240 local-number rows versus 0 atomic/lure-number rows.

MC012 now supplies the first behavior-ready mixed bridge table, but it is a
prompt-visible positive control rather than a probe substrate. It keeps the
integer answer interface and changes the source/evaluation contract with
explicit trusted/untrusted source-status labels. The selected
`compact_reliability` template passed synthetic lookup, familiar lookup, real
atomic-number recall, trusted-source conflict, and answer-absent null controls
at 40/40, selected learned atomic numbers for 39/40 untrusted-source conflict
rows, and produced a primary conflict table with 40 local-number rows,
39 atomic/lure-number rows, and one other-number row. Because that contrast is
carried by visible source-status text, `prompt_channel_locality_gate_passed` is
false: MC012 is behavior-ready, not signature-ready.

MC013 now closes the simple source-status ablation repair. It keeps MC012's
numeric source bank, adds statused positive controls, and makes the two matched
ablation prompts text-identical. The selected `compact_status_ablation`
template passed direct controls and nulls, reproduced the statused positive
control with 40/40 trusted local rows and 39/40 untrusted atomic rows, but the
matched ablation conflict collapsed to 80/80 local-number rows and 0
atomic/lure-number rows. MC013 is therefore a diagnostic boundary: removing the
visible source-status channel does not preserve the learned-fact side under the
current prompt contract.

MC014 now closes the first inferred-reliability repair. It removes explicit
source-status labels and asks the model to check calibration rows against
standard chemistry before deciding whether the local table controls. The
selected `calibration_rule` template keeps direct controls and nulls clean at
40/40 and primary prompts contain no status lexemes, but calibration-inconsistent
conflict still produces 40/40 local-number answers. The route therefore fails
before hidden-state work: calibration evidence alone did not overcome
prompt-local table dominance.

MC015 now closes the first learned-parity gate repair. It removes explicit
source-status labels, hides target and lure atomic numbers, and makes the local
table control depend on whether the queried element's standard atomic-number
parity matches the visible rule. The selected `parity_rule` template keeps
direct controls and nulls clean at 40/40 and balances expected local/atomic
labels by split, but the generated answers do not follow the intended gate:
primary conflict expected correctness is 39/80, expected-local rows select
local only 27/40, and expected-atomic rows select atomic only 12/40. The route
therefore fails before hidden-state work: mixed local/atomic outputs are not
enough unless they track the intended learned-fact rule.

MC016 now closes the first visible non-status alphabet-gate repair. It keeps
target and lure atomic numbers hidden, removes source-status labels, and makes
the local table control depend on the queried element's first-letter range
rather than a trusted/untrusted source cue. The selected `alphabet_rule`
template keeps direct controls and nulls clean at 40/40 and balances expected
local/atomic labels by split, but the conflict collapses completely to local
answers: primary conflict rows are 80/80 local, expected-local rows select local
40/40, and expected-atomic rows select atomic 0/40. The explicit
`feature_labeled_alphabet` variant also fails to rescue the atomic side, with
61 local rows, 0 atomic rows, and 19 unparsed rows. The route therefore fails
before hidden-state work: visible non-status feature gates are not enough under
this prompt-local table contract.

MC017 and MC018 then tested whether MC016's failure was merely the numeric
answer interface. MC017 asked the model to answer with source tokens
`LOCAL`/`ATOMIC` instead of numbers. The reduced 10-source smoke preserved clean
numeric direct controls, but the selector interface collapsed: even the
atomic-only control prompt, which explicitly said no local lab table was active
and standard chemistry was the active source, returned `LOCAL` on 10/10 rows.
That killed MC017 as a behavior substrate and exposed a new dumb explanation:
the answer options themselves can create a local/first-option prior.

MC018 replaced `LOCAL`/`ATOMIC` answers with neutral `A`/`B` labels and
counterbalanced three axes: whether `A` names the local or standard-chemistry
source, whether source definitions are written A-first or B-first, and whether
the final answer list is written A-first or B-first. The expanded 10-source
smoke passed the structural audit at 960 rows: expected local/atomic sources
and expected A/B choices were balanced, source splits were disjoint, target and
lure atomic numbers were hidden, and primary prompts had no status lexemes. The
behavior still failed: selected primary conflict rows were 160/160 parseable,
but source-rule correctness was only 84/160 = 0.525. Primary conflict choices
were almost A/B balanced overall, 88 A versus 72 B, but first-listed choices
won 122/160 = 0.7625 and local-source selections won 102/160 = 0.6375. The
atomic selector control improved to 66/80 atomic, proving MC017's total
`LOCAL` collapse was partly an answer-token artifact, but it still missed the
90% control gate and the answer-absent null returned `UNKNOWN` only 26/80
times. MC018 is therefore not probe-ready. Its value is sharper: the bridge now
has a named answer-interface failure mode where presentation order and
local-source salience dominate the intended source-selection rule even after
neutral labels and counterbalancing.

MC019 then tested a different repair: do not ask the model to choose a source
label at all. Instead, attach a neutral route code to each local table row,
counterbalance the route-rule definition order, and ask for the final integer
or `UNKNOWN`. The 10-source structural gate passed at 420 rows. The selected
`compact_route_column` smoke kept every direct control and null clean: synthetic
local lookup 20/20, familiar-entity local lookup 20/20, real-world atomic
control 20/20, route-rule-absent UNKNOWN 20/20, and answer-absent UNKNOWN
20/20. But the primary route-code conflict still failed: rows were 40/40
parseable, labels and route codes were balanced, but expected correctness was
21/40, expected-local rows selected local only 13/20, and expected-atomic rows
selected atomic only 8/20. A repair template that repeated the queried row
near the rule improved the conflict to 27/40 and expected-local rows to 19/20,
but expected-atomic rows stayed 8/20 and the answer-absent null slipped to
17/20. MC019 therefore becomes a cleaner prompt-contract diagnostic than
MC017/MC018: row-local non-status codes can repair controls and nulls, but they
still do not make learned atomic answers reliably survive prompt-local table
pressure.

MC020 then tested the simplest explanation for MC019: maybe the queried local
number suppresses learned atomic recall whenever it appears beside the element.
That explanation failed. The selected `plain_table` smoke showed atomic recall
was 20/20 with no table, 18/20 with only distractor local rows, 18/20 with the
queried local row present, and 18/20 with the queried row repeated. Local lookup
was also clean at 20/20, and answer-absent null was 18/20. The only sharp
failure was conditional arbitration: route-local reached 17/20 local, but
route-atomic produced 15/20 local and only 2/20 atomic. MC020 therefore names
the next boundary: direct instructions can retrieve learned atomic numbers
under local-table pressure, but route-rule arbitration is asymmetric and tends
to resolve toward the prompt-local table when the rule should select learned
memory.

MC021 then asked whether route-code arbitration works if both branches are
prompt-visible numbers. This matters because a visible-visible route pass plus
a visible-learned route failure would have isolated the boundary to learned
memory selection. Instead, MC021 found a broader behavior-substrate problem.
The selected `plain_branch_table` smoke passed all controls: local-number
control 20/20, visible-reference-number control 20/20, atomic control 18/20,
and answer-absent null 18/20. But visible-visible route conflicts were only
29/40 expected-correct, and visible-learned route conflicts were 20/40
expected-correct. Learned-memory routing was worse, but generic route-code
following was already too weak to serve as a probe substrate. MC021 therefore
turns the next bridge target away from route-code prompts as such: the project
needs a materially different conditional format or a different behavior family.

MC022 then tested the cleanest immediate rescue: keep integer answers, replace
opaque route codes with semantic answer-source labels, and counterbalance
whether the local or nonlocal source rule is defined first. The selected
`explicit_source_column` smoke passed direct controls and nulls: local-number
control 20/20, visible-reference control 20/20, atomic control 18/20, and
answer-absent null 19/20. The structural audit passed at 320 rows: expected
labels and answer-source labels were balanced, splits were source-disjoint,
target atomic numbers were hidden, candidate answers were parseable and
collision-free, and prompts had no status lexemes. The conflict still failed.
Visible-visible conflicts were 30/40 expected-correct, learned conflicts were
24/40 expected-correct, and the ATOMIC branch selected learned atomic numbers
only 5/20. The rule-order split is the important new evidence: defining the
nonlocal visible branch first made visible-visible routing 20/20, while
defining LOCAL first collapsed the same rows to local outputs; defining ATOMIC
first helped learned-memory routing only partially. MC022 therefore closes the
"opaque codes were the whole problem" rescue. The branch-arbitration surface is
not just about code names; it is shaped by local-source salience,
rule-definition order, and whether the selected branch is prompt-visible or
learned memory.

The central insight is:

> The genome is not a list of clean circuits. It is the measured distribution
> of where behaviors become determined and where attempts to control them
> break.

## Bottom Line

The project has not completed the full knowledge genome of small LLMs.

It has accomplished something more concrete than a vague survey:

- it built several behavior substrates;
- it found real control levers;
- it killed multiple tempting false mechanism stories;
- it localized one narrow synthetic lookup surface to a late-layer aggregate
  interaction and final-query attention-write mediation;
- it mapped the reliability boundary that still blocks full promotion;
- it repaired a knowledge-like capital-fact behavior substrate;
- it showed why hidden signatures on that substrate are not good enough;
- it found a near-passing generated matched-surface capital-fact table and
  documented the exact binary-volume failure;
- it repaired that binary-volume failure with a narrow parser-normalization
  audit;
- it showed that the resulting hidden signature is still output/candidate-score
  confounded;
- it found an earlier MC006 lead-time hidden signal that beats same-position
  output margin but remains globally output-confounded;
- it tested simple additive steering on that early signal and found a clean
  negative causal result;
- it showed the original MC006 table cannot support same-table margin matching,
  then improved table balance with V19 while still failing strict
  overlapping-margin promotion;
- it mapped the MC006 source/path lead-time curve and showed that real
  source/path and line-boundary monitors remain final-margin-shadowed;
- it diagnosed the MC006 greedy final-margin sign barrier, showing that the
  current output-interface gate must change before strict final-overlap
  promotion is meaningful;
- it tested that interface change with delayed JSON city generation and found
  that the first-token barrier breaks, but JSON-completion candidate scoring
  still blocks a hidden-signature claim;
- it found a candidate-score-decoupled delayed-city template, then showed the
  hidden monitor still fails shuffled-label selected-search nulls;
- it locked that candidate-decoupled hidden coordinate and showed it fails
  transfer to the other candidate-decoupled delayed-city template;
- it expanded the delayed-city bank and showed the current prompt family repairs
  source-role candidate decoupling better than transfer-role candidate
  decoupling;
- it opened MC007 as the MC005-to-MC006 bridge, showing that familiar-country
  names can collapse into prompt-local lookup keys under terse table authority,
  that explicit authority pressure can create partial prior/lure contrast, and
  that answer-slot parse repair and explicit source/city interfaces do not
  produce a hidden-state-ready behavior table;
- it closed the current MC007 V1-V4 route as a diagnostic bridge, preserving
  V1 as the source-value baseline and V2 as the prior-pressure baseline while
  blocking ordinary prompt-only repairs as mechanism-promotion routes.
- it closed MC008 V1-V2 as the first symbolic fact-code bridge diagnostic;
- it closed MC009 as the first derived-code bridge diagnostic: row-position
  derivation did not rescue the bridge, and typed answer slots proved that
  apparent conflict balance can be manufactured by breaking controls and
  exposing the source channel.
- it closed MC010 as the first two-hop fact-code bridge diagnostic: two-hop
  indirection removed the obvious direct and row-position channels, but the
  full generated behavior table still failed synthetic lookup, real-symbol
  control, and real/lure conflict balance before hidden-state work.
- it closed MC011 as the first numeric answer-interface bridge diagnostic:
  same-format integer answers repaired direct controls, but the full generated
  behavior table still produced 240 local-number conflict rows and 0
  atomic/lure-number conflict rows before hidden-state work.
- it added MC012 as the first clean mixed local-versus-learned bridge behavior
  table, but only as a prompt-visible diagnostic: explicit reliability labels
  create the contrast, so hidden-state work remains blocked.
- it added MC013 as the matched source-status ablation test: the statused
  positive control reproduced MC012, but text-identical ablation collapsed to
  80/80 local-number rows and 0 atomic/lure-number rows.
- it added MC014 as the inferred-reliability calibration test: explicit status
  labels were absent and direct controls stayed clean, but calibration-
  inconsistent rows still collapsed to local-number answers.
- it added MC015 as the parity-gated learned-fact test: target/lure atomic
  numbers and status labels were absent, direct controls stayed clean, and
  expected labels were balanced, but the generated answers did not follow the
  intended parity rule.
- it added MC016 as the alphabet-gated non-status feature test: target/lure
  atomic numbers and status labels were absent, direct controls stayed clean,
  and expected labels were balanced, but expected-atomic rows still collapsed
  to local table answers.
- it added a parser registry that extracts 53 linked result artifacts across
  all 19 atlas rows into a common audit surface and fails direct row/artifact
  readiness contradictions plus 43 family-level closure checks during atlas
  validation.
- it added a checked-in compact artifact index so extracted facts can be
  consumed without reparsing every raw artifact.
- it added a checked-in cross-family comparison layer that turns atlas rows and
  artifact facts into genome-shape ratios, including the current 0 promoted,
  1 bounded, 17 diagnostic, and 1 failed-mechanism-route verdict distribution.
- it added a checked-in law-audit layer that keeps the predictive theory layer
  evidence-bounded against current rows and diagnostics.
- it added a checked-in next-experiment queue that turns law hypotheses into
  prioritized falsification and widening tests; it now ingests bridge-ladder,
  route-disposition, and error-taxonomy closure context so bridge-route items
  carry active MC030-MC033 constraints instead of reopening killed local
  guard/order/schema/checksum/cross-table/fact-claim variants.
- it turned the top bridge-queue pressure into completed MC010, MC011, MC012,
  MC013, MC014, MC015, and MC016 diagnostic rows, plus smoke-stage MC017,
  MC018, MC019, MC020, MC021, MC022, MC023, MC024, MC025, MC026, MC027, and
  MC028 full-source boundary diagnostics, plus MC029 factorized operation-leak
  diagnostics, MC030 null-preserving guard diagnostics, MC031 statusless
  checksum diagnostics, MC032 cross-table consistency diagnostics, and MC033
  fact-claim closeout diagnostics, rather than pending structural scaffolds.
- it added a checked-in claim-audit layer that fails validation if any atlas row
  lacks result artifacts, extracted metrics, family-level claim checks, or full
  row-level metric coverage.
- it added a checked-in claim-consistency layer that fails validation if generic
  verdict, intervention, lead-time, null-locality, behavior-gate,
  output-confound, or signature-causality claims contradict artifact evidence.
- it added a checked-in smoke-diagnostic layer for MC017-MC033 that keeps
  reduced bridge smoke failures machine-readable without promoting them into
  atlas rows; validation now checks 17 smoke cards, 0 hidden-state-allowed
  cards, and 72 failure assertions.
- it added a checked-in bridge-ladder layer for MC010-MC033 that fuses
  validated atlas rows and smoke diagnostics into one cumulative map; validation
  now checks 24 rungs, 1 prompt-visible positive control, 0 unconfounded bridge
  rungs, and 0 hidden-state-allowed bridge rungs.
- it added a checked-in error-taxonomy layer for MC017-MC033 that turns killed
  bridge smoke runs into typed failure data; validation now checks 17 smoke
  cards, 24 bridge rungs, 0 hidden-state-allowed smoke cards, the MC028
  other-number leak bucketization, the MC029 branch/null/example-leak tradeoff,
  the MC030 absence-guard repair failure, and the MC031 checksum invalid-branch
  local collapse, plus the MC032 cross-table mismatch local collapse and the
  MC033 fact-claim match/mismatch branch failure.
- it added a checked-in gate-geometry layer that turns the mechanism-card bar
  itself into measured data; validation now checks the 19-row funnel
  partition, the 23-rung bridge funnel, MC005 as the sole reliability boundary,
  MC012 as the prompt-channel boundary, MC001G as the failed-intervention
  boundary, and 0 promoted mechanisms.
- it added a checked-in genome-snapshot layer that fuses the generated stack
  into one compact current-state object; validation now checks 19 atlas rows,
  53 linked artifacts, 0 promoted mechanisms, 0 hidden-state-allowed bridge
  rungs, MC005 as the only bounded reference, MC030-MC033 as recent bridge
  closures, and paired global allowed/forbidden claims.
- it added a checked-in axis-interactions layer that turns the snapshot into
  predictive regularities; validation now checks 123 feature summaries, 14
  pure predictive rules, the 11-row closed-before-hidden-state law, output
  geometry as a mixed predictor rather than a single terminal-stage rule, and
  singleton features labeled as singleton evidence.
- it added a checked-in coverage-gaps layer that turns the map's negative
  space into named work items; validation now checks 12 gaps, 4 critical gaps,
  14 transfer-untested rows, 0 hidden-state-allowed bridge rungs, at least 80
  singleton/sparse feature summaries, and nonempty next pressures for every
  gap.
- it added a checked-in gap-closure-plan layer that routes every current gap
  and every top queue item into a work order; validation now checks 6 work
  orders, 12/12 gap coverage, 4/4 critical-gap coverage, 5/5 top-queue
  coverage, decision rules on every work order, and preservation of the fact
  that there are still 0 promoted mechanisms and 0 hidden-state-allowed bridge
  rungs.
- it added a checked-in post-MC033 bridge-substrate closeout audit that turns
  the same-family statusless source-validity failures into an explicit
  death condition; validation now checks MC031-MC033 as the closure sequence,
  MC030-MC033 as the recent closed bridge set, 24 bridge rungs, 17 smoke rungs,
  0 hidden-state-allowed bridge rungs, 0 clean unconfounded bridge candidates,
  and the claim boundary that no hidden signature, intervention, mechanism
  card, or general knowledge-control surface is licensed.
- it added a checked-in MC005 reference-specimen audit that makes the project's
  strongest positive control validator-backed; validation now checks MC005's
  bounded verdict, frozen route status, reliability-bound terminal stage,
  exact V29 lookup write mediation, reproduced V31 null boundary, fragile
  transfer class, and the rule that same-route repair is not allowed.
- it added a checked-in MC006 predecision-frontier audit that makes the
  knowledge-like timing boundary validator-backed; validation now checks V14
  behavior-substrate passage, V16 predecision monitor support, final/candidate
  geometry blockers, V17 steering failure, V25 shuffle-null failure, V26-V28
  transfer-bank closure, monitor-only route status, and the rule that ordinary
  delayed-city prompt repair is not allowed as a promotion route.
- it added a checked-in compositional-genome audit that makes the mixture law
  itself the current genome-scale object; validation now checks 19/19
  prompt-contract-visible rows, 14/19 output-geometry-visible rows, 13/19
  source-or-prompt-token-dependent rows, 11/19 behavior/bridge-substrate-
  blocked rows, 5/19 internal-monitor rows, 1/19 internal-causal row, 0
  promoted mechanisms, 0 full-reliability mechanisms, 0 transfer-ready
  mechanisms, and 0 clean unconfounded bridge substrates, while explicitly
  forbidding general truth-vector or knowledge-control claims.
- it added a checked-in family matrix that turns the atlas into a single
  cross-family table; validation now checks 19 joined rows, 25 declared
  columns, preservation of mixture/frontier/reliability/transfer/gate counts,
  MC005 as the sole bounded reference, MC004/MC006 as monitor-only rows, and 0
  promotion-ready rows.
- it added a checked-in knowledge ladder that separates synthetic lookup,
  semi-synthetic familiar-entity lookup, symbolic/learned-memory bridges,
  parametric-fact override, and real abstention/uncertainty; validation now
  checks 5 levels, 14 ladder rows, 5 auxiliary diagnostic rows, MC005 as the
  bounded level-1 reference, MC006 as the monitor-only level-4 boundary, 0
  promoted levels, 0 hidden-state-allowed bridge rungs, and 0 real
  abstention/uncertainty mechanism-ready levels.
- it executed the KSQ002 familiar-entity source-rewrite equivalence first run:
  the 720-row structural substrate passed, the 10-source smoke was a candidate
  under `registry_question`, and the full run exposed a narrow but decisive
  source-disjoint holdout boundary. `sentence_rewrite` preserved most rewritten
  source values (`36/40` artificial rows) and deletion/query-only controls were
  clean (`40/40` unknown each), but source-disjoint rewrite holdout
  parseability reached only `14/16` (`0.875`) against the `0.900` gate. This
  says source-channel rewrite behavior can be mostly local and still fail the
  reliability condition needed before hidden-state work.
- it executed the KSQ004 bridge answer-interface minimal-pairs first run: the
  720-row structural substrate passed, the 10-source smoke was a candidate
  under `question_form`, and the full run exposed template fragility rather
  than a behavior-ready bridge. `question_form` retained a mostly working
  local-vs-learned branch (`0.850` conflict expected-correct), while
  `compact_form` collapsed expected-atomic conflict rows toward prompt-local
  table answers (`0.6125` conflict expected-correct, `0.225` atomic-branch
  atomic rate). This says answer-interface matching is not a sufficient
  shortcut explanation, but prompt-template form is itself a control surface.
- it executed the KSQ005 grounded-answerability first run for the real
  abstention/uncertainty level: the 480-row structural substrate passed, but
  the 10-source smoke failed before hidden-state work because unknown nonce
  rows and unsupported context rows did not abstain or parse reliably enough,
  even though known direct facts and contradicted familiar facts passed their
  branch floors.
- it executed the KSQ006 context-support counterfactual first run for the real
  abstention/uncertainty level: the 720-row structural substrate passed, but
  the 10-source smoke failed before hidden-state work because insufficient
  context rows and claim/context-only controls reproduced answer channels,
  even though supported, irrelevant, and contradicting context rows passed
  their branch floors.
- it added a generated KSQ first-run outcome matrix that converts the six
  executed knowledge-substrate runs into a comparable diagnostic distribution:
  6 structural passes, 3 full-behavior terminal failures, 3 smoke terminal
  failures, 6 exported diagnostics, 0 behavior-ready rows, and 0 hidden-state
  licenses.
- it added a generated KSQ failure topology that turns that distribution into
  second-wave decisions: repair KSQ002 first, adjudicate KSQ004 template
  invariance next, bound-or-close KSQ001, and redesign the statusless bridge
  plus real-uncertainty substrates before any hidden-state work.
- it executed the KSQ001 second-wave parseability bound and closed ordinary
  familiar-prior repair: the compact replay preserved the original full-scale
  `0.600` parseability boundary, while softer answer-shape variants raised
  parseability only by losing the prior branch or moving toward UNKNOWN. The
  new result keeps the diagnostic `FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF`
  and licenses no signature screen, hidden-state claim, intervention, or
  mechanism.
- it executed the immediate KSQ002 second-wave repair and killed the ordinary
  source-rewrite repair route: `city_field_rewrite` repaired the named
  source-disjoint holdout boundary to `16/16`, but baseline lookup fell to
  `30/40` and source-deletion UNKNOWN fell to `10/40`. The new diagnostic is
  `SOURCE_REWRITE_REPAIR_LOCALITY_REGRESSION`, and no signature screen,
  hidden-state claim, or intervention is licensed.
- it executed the KSQ003 second-wave evidence-sufficiency redesign and bounded
  the statusless bridge route without promoting it: `identity_packet` produced
  `40/40` complete-evidence atomic answers but failed null stress almost
  completely, while `compact_identity` improved null stress to `66/120`
  UNKNOWN and removed complete-conflict local answers but dropped
  complete-evidence atomic answers to `13/40`. The new diagnostic is
  `STATUSLESS_EVIDENCE_SUFFICIENCY_BOUNDARY`, and no signature screen,
  hidden-state claim, or intervention is licensed.
- it executed the high-priority KSQ004 second-wave template-invariance
  adjudication and admitted a bounded behavior substrate: `question_form`
  passed, `relation_key_form` passed, and `neutral_sentence_form` failed.
  This converts the first-run template-fragility result into a sharper
  relation-format boundary: answer-interface matching alone is not enough, but
  the bridge can survive a non-question relation-key form. The new diagnostic
  is `TEMPLATE_INVARIANT_BRIDGE_BEHAVIOR`; it licenses one later signature
  screen and still licenses no hidden-state claim, intervention, or mechanism.
- it executed the KSQ005/KSQ006 second-wave relation-evidence answerability
  redesign and killed the current real-uncertainty route: `compact_relation`
  answered known/direct rows at `39/40` and exact `REL capital_of` supported
  rows at `40/40`, but unknown nonce rows only abstained `27/40`,
  unsupported rows only abstained `28/40`, contradiction rows selected the
  prior/true capital `38/40`, and claim/mention controls reproduced the
  supported capital on `79/80` rows. The stricter `relation_rows` template
  protected abstention and controls better, but supported answering collapsed
  to `11/40`. The diagnostic is
  `RELATION_EVIDENCE_ANSWERABILITY_BOUNDARY`; no signature screen, hidden-state
  claim, intervention, or mechanism is licensed.
- it executed the KSQ007 nonce-evidence answerability calibrator and split the
  real-uncertainty failure into a sharper axis: under `evidence_rows`, exact
  nonce evidence answered `9/10`, absent evidence abstained `10/10`,
  unrelated-entity rows abstained `10/10`, conflicting rows abstained `9/10`,
  and mention/query-only controls abstained `10/10`, but claim-only controls
  still reproduced the nonce value `6/10`. This exports
  `NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY`: real capital priors are not the
  only problem; claim text itself can become an answer channel. No signature
  screen, hidden-state claim, intervention, or mechanism is licensed.
- it executed the KSQ007B claim-channel boundary audit and sharpened that
  failure: under `counted_uncounted_sections`, exact evidence answered `9/10`,
  literal `CLAIM answer_for(entity)=value` abstained `9/10`, `NOT_EVIDENCE`,
  quoted evidence syntax, wrong-predicate claims, mention-only, and query-only
  controls abstained `10/10`, but prose claims reproduced `4/10`, bare
  `answer_for(entity)=value` reproduced `7/10`, and evidence-looking rows
  outside the counted block reproduced `2/10`. This exports
  `ANSWER_FOR_SYNTAX_CLAIM_LEAK`: the live bottleneck is slot-binding syntax
  becoming answer-bearing, not merely familiar priors, value mention, or the
  word `CLAIM`. No signature screen, hidden-state claim, intervention, or
  mechanism is licensed.
- it hardened `code/validate_control_surface_atlas.py` so KSQ007 and KSQ007B
  cannot silently drift: the validator now checks their structural and smoke
  JSONs, expected diagnostics, decision flags, selected templates, failed
  criteria, and panel label counts. This makes the two newest knowledge
  substrate failures auditable boundary facts, not just narrative additions.
- it generated the knowledge third-wave outcomes layer, which links KSQ007 and
  KSQ007B into one diagnostic chain: removing real-world priors repairs many
  answerability branches but leaves claim-only leakage, and claim-channel
  splitting localizes the remaining problem to answer-bearing slot-binding
  syntax. The layer records two completed outcomes, 2,040 structural rows, 510
  smoke rows, and zero behavior-ready, hidden-state, intervention, or mechanism
  licenses.
- it executed KSQ008 as the first neutral-evidence repair against the
  `ANSWER_FOR_SYNTAX_CLAIM_LEAK` boundary. The 1,440-row structural substrate
  passed, and the 360-row smoke selected `field_registry`, but exact neutral
  evidence answered only `5/10`, counted wrong-schema `answer_for(entity)=value`
  rows reproduced `5/10`, conflicts selected a value `2/10`, and neutral
  evidence versus a forbidden bare alternate answered only `2/10`. The result
  exports `NEUTRAL_EVIDENCE_POSITIVE_FAILED`: neutral evidence grammar alone
  suppresses many forbidden channels but does not yet produce a behavior-ready
  answerability substrate.
- it generated the knowledge fourth-wave outcomes layer, which records KSQ008
  as one completed failed repair with 1,440 structural rows, 360 smoke rows,
  and zero behavior-ready, signature-screen, hidden-state, intervention, or
  mechanism licenses.
- it executed KSQ009 as a schema-specific value-lookup repair against the
  KSQ008 boundary. The 1,200-row structural substrate passed, and the 300-row
  smoke selected `kv_lines`, but exact `ALLOW` rows answered only `2/10`,
  counted wrong-schema `answer_for(entity)=value` rows reproduced `9/10`,
  uncounted wrong-schema `answer_for` rows reproduced `7/10`, uncounted
  `answer_for` alternates overrode explicit `ALLOW` rows `9/10`, and quoted
  `ALLOW` rows still reproduced `4/10`. The result exports
  `SCHEMA_SPECIFIC_POSITIVE_FAILED`: schema labels alone are not enough when a
  competing string form has become an answer-bearing micro-language.
- it generated the knowledge fifth-wave outcomes layer, which records KSQ009
  as one completed failed repair with 1,200 structural rows, 300 smoke rows,
  and zero behavior-ready, signature-screen, hidden-state, intervention, or
  mechanism licenses. The practical next constraint is no longer merely
  "make evidence neutral" or "declare a schema"; the next packet must separate
  target extraction and counted evidence from answer-like syntax strongly
  enough that the model cannot satisfy the prompt by following the old
  `answer_for` surface.
- it executed KSQ010 as that separation test: a two-stage codebook in which a
  counted entity-code row must be joined to a counted code-value row, so the
  output value is not directly attached to the entity. The 1,440-row
  structural substrate passed, and the 360-row smoke selected `tag_rows`. This
  improved positive lookup relative to KSQ009: exact bridges answered `8/10`.
  But it still missed the parse gate with `2/10` unparsed, missing code-value
  rows returned code-like unparsed strings `10/10`, both conflict panels
  selected a value `10/10`, counted `answer_for(entity)=value` rows reproduced
  `10/10`, and an `answer_for` alternate overrode the codebook bridge `9/10`.
  The result exports `CODEBOOK_POSITIVE_FAILED`: separating target extraction
  and value lookup helps the positive branch, but does not yet repair locality,
  conflict abstention, or answer-like syntax competition.
- it generated the knowledge sixth-wave outcomes layer, which records KSQ010
  as one completed failed repair with 1,440 structural rows, 360 smoke rows,
  and zero behavior-ready, signature-screen, hidden-state, intervention, or
  mechanism licenses. The new map cell is more precise: positive lookup and
  answer-channel suppression are now separable axes. A future packet must stop
  using positive-answer rate as the main admission signal unless conflict,
  locality, and answer_for-adversary panels pass beside it.
- it executed KSQ011 as the targeted syntax ablation for the KSQ010
  `answer_for` competition. Holding the same selected `tag_rows` codebook
  bridge fixed, exact bridge lookup now answers `9/10`, so the positive branch
  is not the core blocker in this packet. The failure is sharper:
  function-like alternate assignment forms dominate. Exact `answer_for`
  overrides `9/10`, spaced `answer_for` overrides `10/10`, colon `answer_for`
  overrides `9/10`, `answer_to` overrides `9/10`, `value_for` overrides
  `8/10`, prose value notes override `7/10`, and quoted `answer_for` overrides
  `9/10`. By contrast, plain entity assignment and bare alternate mention
  mostly preserve the bridge at `8/10` each, and query-only abstains `10/10`.
  The result exports `FUNCTION_ASSIGNMENT_ANSWER_CHANNEL_DOMINANCE`: the
  boundary is not one exact `answer_for` spelling, but a broader function-like
  assignment answer channel separable from generic value salience.
- it generated the knowledge seventh-wave outcomes layer, which records KSQ011
  as one completed diagnostic with 480 structural rows, 120 smoke rows, and
  zero behavior-ready, signature-screen, hidden-state, intervention, or
  mechanism licenses. The new map cell is that answer interfaces are behavior
  surfaces: a prompt-visible assignment micro-language can beat a cleaner
  two-stage lookup bridge even when nonfunction value mentions mostly do not.
  The next repair must remove, mask, section, or delay function-like assignment
  text itself; escaping a single exact syntax is the wrong lesson.
- it executed KSQ012 as the first direct wrapper repair against that
  function-assignment channel. This repaired the design flaw from the first
  attempted wrapper prompt and restored the baseline: exact bridge lookup
  answers `9/10`, and raw `answer_for` reproduces the alternate `9/10`, so the
  packet is testing the right pressure. The repair still fails. Inactive-block
  `answer_for` overrides `9/10`, comment-mark wrapper overrides `8/10`, fenced
  text overrides `9/10`, below-cut text overrides `9/10`, detached
  function/value rows override `7/10`, unrelated-entity `answer_for` still
  overrides `6/10`, and assignment-only inactive control leaks the alternate
  `8/10`. Split-assignment-only and query-only controls abstain `10/10`, but
  masked and split bridge panels are partial rather than behavior-ready. The
  result exports `FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK`.
- it generated the knowledge eighth-wave outcomes layer, which records KSQ012
  as one completed diagnostic with 480 structural rows, 120 smoke rows, and
  zero behavior-ready, signature-screen, hidden-state, intervention, or
  mechanism licenses. The new map cell is sharper than "function syntax is
  dangerous": wrapper prose, comments, fences, cut markers, and inactive labels
  are weak controls, and visible function-like assignment text can leak even
  without a valid counted bridge. Future designs should treat that syntax as
  unsafe by default and move to materially nonfunction representations unless
  split/masked forms pass both bridge-preservation and no-bridge-abstention
  gates on full-source holdout.
- it executed KSQ013 as the first materially nonfunction representation screen
  after the wrapper failure. The packet removed function-call and assignment
  syntax from proposed repair panels while retaining raw `answer_for` only as
  a positive-control pressure check. Exact bridge lookup still answers `9/10`,
  and raw `answer_for` still reproduces the alternate `9/10`. The result is
  useful because it is not another wrapper failure: value-bank and decoy-pair
  bridge panels answer `8/10`, metadata answers `7/10`, separated entity/value
  answers `6/10`, bare alternate mention answers `8/10`, value-bank-only,
  decoy-pair-only, and query-only controls abstain `10/10`, but separated
  entity/value-only control leaks the alternate `4/10`. Catalog slash text is
  the single bridge panel matching exact bridge at `9/10`, but KSQ013 lacks a
  slash-only no-bridge control, so it cannot promote.
- it generated the knowledge ninth-wave outcomes layer, which records KSQ013
  as one completed diagnostic with 480 structural rows, 120 smoke rows, and
  zero behavior-ready, signature-screen, hidden-state, intervention, or
  mechanism licenses. The new map cell is that "nonfunction" is not one
  category. Value-bank, metadata, decoy-pair, separated-mention, slash-note,
  and bare-mention surfaces have different behavior, parse, and locality
  profiles. The next sharp experiment is a slash-only locality packet, not a
  hidden-state screen.
- it executed KSQ014 as that slash-only locality packet. The first version
  exposed a real prompt-contract issue: adding slash-specific warning text
  caused exact bridge rows to emit code IDs, so the prompt wording was repaired
  back to the KSQ013 non-counted-row instruction before interpretation. The
  repaired smoke restored exact bridge to `9/10` and raw `answer_for` to
  `9/10`. Catalog slash entity, decoy, and reversed variants passed the smoke
  bridge/locality gates, all matched slash-only controls abstained `10/10`,
  and bare slash failed at `6/10` bridge answer with `4/10` unparsed.
- it generated the knowledge tenth-wave outcomes layer, which records KSQ014
  as one completed diagnostic with 440 structural rows, 110 smoke rows, and
  zero behavior-ready, signature-screen, hidden-state, intervention, or
  mechanism licenses. The new map cell is that catalog labels can stabilize a
  slash representation enough for full-source behavior testing, while bare
  slash notation is parse-fragile under the same counted-codebook contract.
- it executed KSQ015 as that full-source catalog-slash-only packet. The
  result did not promote: exact bridge answered only `35/40` with `5/40`
  unparsed code-token outputs, raw `answer_for` remained active at `39/40`,
  catalog slash entity answered `36/40`, catalog slash decoy answered `35/40`,
  catalog slash reversed answered `35/40`, and all catalog slash-only
  no-bridge controls abstained `40/40`.
- it generated the knowledge eleventh-wave outcomes layer, which records
  KSQ015 as one completed diagnostic with 360 structural rows, 360
  full-behavior rows, and zero behavior-ready, signature-screen, hidden-state,
  intervention, or mechanism licenses. The new map cell is that catalog slash
  locality is clean in no-bridge controls, but the underlying counted bridge
  contract fails full-source positive-control reliability.

The current state is best described as:

> The strongest current result is the compositional map itself: prompt
> contracts, output geometry, source/prompt-token pressure, bridge-substrate
> failures, internal monitors, bounded causal surfaces, reliability boundaries,
> and transfer gaps now form a measured distribution rather than a pile of
> anecdotes. The project has not found a general truth vector or knowledge
> vector; it has built a machine that makes those claims pass through typed
> failure, null, locality, output, and transfer gates before they are believed.
>
> We have a rigorous partial map of small-LLM control surfaces, with one narrow
> internal causal surface close to mechanism-card status and one knowledge-like
> behavior branch that now has a matched generated table, a near-overlap
> prompt-bank table, a source/path lead-time curve, a final-margin sign-barrier
> diagnosis, a delayed-city interface repair that breaks the first-token
> barrier, and a candidate-decoupled delayed-city target that fails selection
> nulls, plus a locked-coordinate transfer test that fails across
> candidate-decoupled templates, plus an expanded bank that is source-ready but
> transfer-insufficient, but still lacks a hidden signature that beats all
> output/candidate, shuffled-label, and transfer controls or a causal
> intervention that works from its current early signals.
> We also have bridge branches, MC007-MC033, that have become diagnostic maps
> of prompt-authority parseability failure,
> source-declaration/city-answer mismatch, table-dominant symbolic conflicts,
> row-position-derived prompt-local dominance, typed-slot prompt-visibility,
> two-hop direct-control failure, numeric conflict collapse, reliability-label
> prompt-channel visibility, matched source-status ablation collapse,
> calibration-inference collapse, parity-gate rule-following failure, and
> alphabet-gate local collapse, plus smoke-stage evidence that source-selector
> interfaces can be dominated by answer-token, definition-order,
> answer-option-order, and local-source-salience biases, and that row-local
> route codes can clean up controls and nulls without making expected-atomic
> rows follow the route rule, plus evidence that direct atomic recall survives
> local-table pressure while conditional source arbitration remains asymmetric,
> that route-code arbitration is weak even between prompt-visible branches, and
> that explicit semantic source labels still leave local-source salience,
> rule-order, and learned-memory branch selection as behavior-substrate blockers
> and that few-shot query-operation examples repair prompt-local routing more
> readily than learned atomic routing under full source-disjoint evaluation,
> and that prompt-visible A/B/C candidate choices introduce atomic-control and
> answer-absent-null failures rather than neutrally repairing the output
> interface, and that numeric option lists preserve null behavior by converting
> direct learned recall into UNKNOWN rather than rescuing learned routing,
> and that answer schemas themselves are behavior surfaces: bare integer can
> clear the 10-source smoke while prefix, JSON, choice, and option interfaces
> break controls, nulls, parseability, or learned routing, and the bare-integer
> smoke survivor closes at full-source scale through learned-branch
> other-number leakage. The error taxonomy now shows that this leakage is not
> one generic local-copy failure: among the 28 wrong operation-atomic rows, 9
> copy worked-example outputs, 8 are other bank atomic numbers, 6 are off-bank
> other numbers, 3 are bank local numbers, and 2 are prompt-local row numbers,
> and MC029 shows that factorizing numeric examples, label-only examples,
> rules-only prompts, query-before-example ordering, and query-row-last salience
> moves the failure axes without repairing the substrate: `rules_only` reaches
> 0.875 operation-atomic atomic but drops answer-absent UNKNOWN to 0.806, while
> `query_before_examples` preserves nulls but amplifies worked-example copying
> to 54 rows, and MC030 shows that simple absence guards do not repair that
> tradeoff: the row-absence guard drives answer-absent UNKNOWN to 0.231, the
> decision-order guard only reaches 0.425, and query-last lowers other-number
> leakage to 0.056 only by collapsing operation-atomic atomic to 0.487. Those
> are diagnostic pressure points rather than probe substrates. MC031 then
> tests a materially different statusless arithmetic-checksum route; its
> structural gate passes, but the 10-source smoke still collapses
> invalid-checksum rows to local-table answers rather than learned atomic
> answers. MC032 removes the checksum-specific objection by replacing checksum
> validity with cross-table consistency; the mismatch branch still selects
> atomic/lure numbers 0/10 times, mostly chooses the primary local row, and
> never copies the second-table side number. MC033 removes the cross-table
> objection by replacing table consistency with a row-local standard-number
> claim checked against learned atomic memory; direct controls and nulls stay
> clean, but match rows return learned atomic numbers too often and mismatch
> rows split between local answers and the wrong claimed number instead of
> selecting the learned atomic answer.
> Those bridge routes are now closed, bounded, or
> smoke-blocked as
> diagnostics; the next bridge attempt must create MC012-level contrast while
> making the answer rule independent of visible trusted/untrusted source-status
> text and arithmetic checksum validity cues, proving that outputs follow the
> intended rule, showing that expected-atomic rows survive prompt-local table
> pressure, defeating neutral label/order controls, and beating row-code
> prompt-local dominance, asymmetric conditional arbitration, checksum collapse,
> cross-table local dominance, and fact-claim branch instability with a generic
> route-stable substrate before any hidden-state work. We are
> not done, but we now know much more precisely what would count as done.
