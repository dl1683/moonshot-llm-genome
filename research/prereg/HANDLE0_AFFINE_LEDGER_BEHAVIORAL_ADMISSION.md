# HANDLE-0 Affine Ledger Behavioral Admission

Status: preregistered, pending execution.

Design source: Codex design review (2026-08-31), deposited from
Latent-Space-Reasoning program closure. Full spec in
`research/60_HANDLE_0_AFFINE_LEDGER_ADMISSION.md`.

Runner: `code/handle0_affine_ledger.py` (reference interpreter),
`code/handle0_data_generator.py` (trajectory generator).

## Scope

This is a **behavioral admission** preregistration, not a mechanism card.
It establishes that the Affine Ledger task is well-defined, requires private
cross-chunk state, and defeats output-visible explanations. It does NOT
establish that HANDLE's typed slots are necessary — only that the task is
suitable for later mechanism investigation.

The admissible claim after passing: HANDLE may learn a causally factorized,
cache-like state mechanism more reliably than an equally budgeted unstructured
recurrent controller. Behavioral accuracy alone is a control result.

## Task

Four-register affine machine over GF(5):

- V0, V1: value registers in Z_5
- M0, M1: affine bijection registers (a, b) where a in {1,2,3,4}, b in Z_5
- 10,000 distinct states, 13.29 zero-error bits
- 6 operations: SETV, SETM, TRANSPORT, COMPOSE, SWAPV, SWAPM
- 14 late-bound queries (READ, EVAL, APPLY, APPLY_COMPOSED)
- KV cache discarded between every chunk
- Renderer (logical -> output token) disclosed only in final chunk

## Model and architecture

- Base: Qwen3-0.6B-Base (frozen, KV cache discarded after every chunk)
- Controller: <1M parameters, four persistent slots
- Competitors: symbolic cache, direct answer register, single unrestricted
  learned vector, matched-parameter generic RNN

## Primary metric

Behavioral accuracy: fraction of test trajectories where the model's output
token matches the reference interpreter's answer under the given renderer.

## Success criterion

HANDLE behavioral accuracy >= 90% on held-out trajectories with held-out
entity assignments, command sequences, renderers, and query types.

## Failure criterion

If HANDLE behavioral accuracy < 50% on the training distribution after
convergence, close the line. The task is too hard for the architecture at
this parameter budget.

## Required controls

### Positive controls

1. **Symbolic cache ceiling:** hand-coded key-value interpreter scores 100%
   on every trajectory by construction. If it does not, the generator is wrong.
   Do not train.

2. **Full-history frozen model ceiling:** frozen Qwen3-0.6B with full KV
   history (no resets) as a capability ceiling.

### Negative controls

3. **No-state baseline:** frozen model with KV reset and no persistent module.
   Must score ~20% (chance). If above 25%, there is leakage.

4. **Final-chunk-only baseline:** model sees only the final chunk. Must score
   ~20%. If above 25%, the renderer or query leaks information.

5. **Current-chunk classifier:** logistic regression on last-chunk tokens.
   Must score ~20%.

### Competitor controls

6. **Direct answer register:** single 5-way learned register (no slots, no
   typed operations). Expected: fails late-bound query family because it
   cannot prepare the right answer without knowing the query.

7. **Single unrestricted learned vector:** continuous vector of matched
   dimensionality to HANDLE's total slot space. Can in principle encode all
   10,000 states. If it matches HANDLE, typed slots add no behavioral value.

8. **Matched-parameter generic RNN:** GRU or similar with same parameter
   count as HANDLE controller, no explicit slot structure. If it matches
   HANDLE on systematic holdouts, HANDLE typing has no demonstrated benefit.

### Generalization holdouts

9. **Held-out entities:** novel nonce entity names not seen in training.
10. **Held-out renderers:** permutations not seen in training.
11. **Held-out query types:** at least 2 of the 14 queries withheld.
12. **Held-out command sequences:** template sequences not seen in training.
13. **Held-out trajectory lengths:** if trained on 4-8 updates, test on 9-12.

## Counterfactual groups

For every scored final-query string, matched histories producing all 5 logical
answers with identical:
- history length
- command-template sequence
- final pre-query command
- renderer
- query
- intermediate outputs (all ACK)

Only earlier command arguments vary. This is the private-state witness.

## Renderer balance

For every logical value, every output token appears with equal frequency:
I(logical answer; output token) = 0 marginally. Verified by the reference
interpreter (24 per cell across 120 permutations).

## Pre-training kill conditions

Do not begin model training if any of these hold:

1. Final-chunk leakage (>20% from final chunk alone)
2. Missing collisions in counterfactual groups
3. Renderer imbalance
4. <10,000 distinct state signatures
5. Answer precomputable before final reset
6. COMPOSE/TRANSPORT removable without changing the task
7. Reference interpreter scores <100%
8. HANDLE receives information unavailable to competitors
9. No hybrid-patch predictions preregistered
10. Claim scope exceeds "factorized vs. unstructured"

All 10 verified PASS by `code/handle0_affine_ledger.py`.

## Later mechanism card (separately gated)

If behavioral admission passes, the mechanism card requires:

- Slot decoding: preregistered readout of logical value and entity binding
  at a fixed pre-query time, beating prompt-only and output-margin baselines
- Donor-state intervention: overwrite one slot, predict all 14 query answers
  from the hybrid-state oracle
- Dependency-cone locality: only queries dependent on the patched slot change
- Permutation equivariance: V0<->V1 and M0<->M1 swaps permute answers
- Composition-law consistency: patched composed maps agree with sequential
  application
- Matched intervention gap: slot patch beats whole-vector, matched-norm, and
  coordinate-block patches

This is NOT part of the current preregistration. It requires a separate gate.

## Verdict criteria

- **ADMIT:** HANDLE >= 90% behavioral accuracy, all controls pass, at least
  one competitor (answer register or RNN) performs measurably worse on
  systematic holdouts. Proceed to mechanism card preregistration.
- **DIAGNOSTIC ONLY:** HANDLE matches competitors. Typed slots provide no
  behavioral advantage. The task is valid but HANDLE architecture is not
  justified.
- **FAIL:** HANDLE < 50% or controls fail. Close the HANDLE line.

## Preregistration date

2026-08-31.

## Experiment owner

Devansh (devansh@iqidis.ai), with Codex architectural authority.
