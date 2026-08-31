# HANDLE Substrate Proposal

Date: 2026-08-31

Status: **NO-GO as currently specified** (Codex direction review, 2026-08-31).
Architecturally new but not yet a discriminating mechanism hypothesis. The
dumbest explanation — an ordinary finite-state key-value store with an
answer-sufficient code in an explicit slot — is not defeated by the current
design. Conditional reconsideration after a HANDLE-0 admission packet
demonstrates that the task requires private persistent state, defeats
output-visible explanations, and distinguishes typed slots from a symbolic
cache or generic recurrent controller. See "Codex ruling" section below.

## What HANDLE Is

HANDLE (Handle-Addressable Neural Dynamics with Lawful Execution) is a
co-trained module architecture that explicitly commits to typed state
operations. It is a **materially new substrate class** relative to MC007–MC033
(which were all prompt-based behavioral bridges on frozen models).

Core design:
- Frozen base model (e.g., Qwen3-0.6B-Base) with KV cache discarded after
  every chunk.
- A small (<15M parameter) trainable module with eight persistent entity slots.
- Typed operations: READ, WRITE, COMPOSE, TRANSPORT.
- The module must pass all three state gates (see §11 of the math agenda)
  before any hidden-state work is licensed.

## Why It Fits Here

The POST_MC033 bridge closeout requires a "materially new substrate class" to
reopen hidden-state work. HANDLE is materially outside the closed family
because:

1. **Not prompt-based.** MC007–MC033 attempted behavioral bridges through prompt
   engineering on frozen models. HANDLE co-trains a module that must learn
   explicit state operations.
2. **Explicit commitments.** The module's operations are typed and predeclared —
   not discovered post hoc from activations.
3. **Three-gate contract.** A HANDLE state must be present (measurable),
   addressable (intervention changes behavior), AND composable (two states
   combine predictably). This is strictly stronger than the existing
   signature → intervention → reliability contract.

## What It Does NOT Inherit

- No hidden-state license from LSR. HANDLE enters at its own behavioral
  positive control.
- No prior-art claim. The LSR program found no native latent mathematics;
  HANDLE is a constructive response, not a continuation.
- No claim that existing latent spaces contain undiscovered structure.

## LSR Constraints That Apply (from §11 math agenda)

1. **Transport-earned quotients.** If HANDLE claims two slot states are
   equivalent, the equivalence must be demonstrated by a held-out intervention,
   not by representation similarity.
2. **Collision witnesses for absence.** Claiming a slot lacks a property requires
   two inputs with the same slot state but different downstream behavior.
3. **Cheapest-mechanism nulls.** Before attributing behavior to a HANDLE slot,
   run the cheapest direct alternative (e.g., does the behavior follow from
   the frozen model's existing KV-cache state without the module?).
4. **Many operational latent spaces.** Specify which (actions, observations,
   horizon) triple the HANDLE mechanism operates in.

## Admission Requirements (per POST_MC033)

Before any hidden-state work on HANDLE, the behavioral positive control must:
- Show the module learns to READ and WRITE state correctly on training
  entities/templates.
- Show held-out generalization (new entities, new templates, new slot
  assignments).
- Pass branch, null, local, side-number, parseability, prompt-channel,
  source-disjoint, and output/candidate controls together.
- Include the cheapest-mechanism null: frozen model alone, with no module,
  on the same task.

## Kill Conditions

- If eight slots reduce to a lookup table (identity mapping from input to
  output), HANDLE is not a state mechanism.
- If the module's learned operations reduce to linear systems, automata,
  coalgebra, causal abstraction, or semiseparable matrices, stop — the
  state is standard, not new.
- If the behavioral positive control fails (training entities don't learn
  typed operations), close without hidden-state work.

## Next Steps

**HANDLE-0 admission packet completed:** see
[60_HANDLE_0_AFFINE_LEDGER_ADMISSION.md](60_HANDLE_0_AFFINE_LEDGER_ADMISSION.md).

The Affine Ledger task (four-register machine over GF(5), 10,000 states, 13.29
zero-error bits, exactly balanced renderer crossing) satisfies the admission
requirements. Conditional admission under the narrower claim: HANDLE may learn
a causally factorized state mechanism more reliably than an equally budgeted
unstructured recurrent controller. Behavioral accuracy alone is a control result.

Remaining gates before training:
1. Implement the reference symbolic cache/interpreter (CPU, <200 lines Python).
2. Verify kill conditions 1-10 from the admission packet pass.
3. Write preregistration to `research/prereg/`.
4. Only then: implementation and behavioral positive control.

## Codex ruling (2026-08-31)

**NO-GO as currently specified.** Architecturally outside MC007–MC033 (yes),
but not hidden-state-ready. Key objections:

1. **Output-geometry shadow risk.** If slots are trained against the answer,
   the likely result is output geometry relocated into a named bottleneck —
   causal but not a state mechanism of independent interest.
2. **Dumbest explanation not defeated.** An ordinary finite-state key-value
   store whose training loss writes an answer-sufficient code into a slot,
   with the decoder reading that code, explains every planned observation.
3. **Kill condition is misguided.** Demanding "new mathematical vocabulary"
   imports the failed LSR ambition. A reliable standard mechanism would
   satisfy this project's question.
4. **Task problem, not substrate problem.** The 24 bridge failures are a
   correlated family of prompt-local vs. learned-memory arbitration tasks.
   HANDLE will repeat the same failure if trained on another source-validity
   or numeric-arbitration behavior.

**Conditional path forward (HANDLE-0 admission packet):**
1. Define the operational space (actions, observations, horizon).
2. Choose a behavior that provably requires private cross-chunk state.
3. Construct renderer-crossed counterfactual data (same final chunk, different
   histories, counterbalanced output tokens).
4. Specify competing hypotheses: symbolic cache, direct answer register,
   generic recurrence, typed compositional slots.
5. Demonstrate that planned observations can distinguish those hypotheses.
6. Preregister behavior, nulls, holdouts, failure criteria, and later
   intervention before training.

**If HANDLE survives that audit:** use four slots, <1M-parameter controller,
procedural cross-chunk task. Behavioral performance only first. Slot mechanism
card is a separate, later gate.

## Source

Deposited from `Latent-Space-Reasoning/` (closed 2026-08-31, 50 audited
experiments). The constructive insight: the reason the LSR program found no
native latent mathematics is that no current architecture has
fixed-dimensional, non-linear-algebraic, compositional, accessible state.
HANDLE is the proposal to build one and test whether it requires new
mathematical vocabulary.
