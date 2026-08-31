# HANDLE-0: Affine Ledger Admission Packet

Date: 2026-08-31

Status: **Conditional admission as a private-state identifiability task.**

Source: Codex design review (session `01a05754-b1aa-7163-aa9a-ad3aba5955f7`),
deposited from Latent-Space-Reasoning program closure. Verified: 10,000
behaviorally distinct states, 13.29 zero-error bits, exactly balanced renderer
crossing.

Scope: Proceed only under the narrower claim that HANDLE may learn a causally
factorized, cache-like state mechanism more reliably than an equally budgeted
unstructured recurrent controller. Behavioral accuracy alone is a control
result. If the desired claim is "this task proves HANDLE is necessary rather
than a symbolic cache," the claim is unidentifiable and must be killed now.

Prerequisite: [59_HANDLE_SUBSTRATE_PROPOSAL.md](59_HANDLE_SUBSTRATE_PROPOSAL.md)
(NO-GO ruling adopted).

---

## 1. The task

Maintain four private, typed registers over arithmetic modulo 5:

- `V0, V1`: value registers in Z_5.
- `M0, M1`: transformation registers containing affine bijections
  M(x) = ax + b (mod 5), where a in {1,2,3,4} and b in Z_5.

This gives 5^2 x 20^2 = 10,000 possible machine states, requiring 13.29 bits
for zero-error identification.

### Command grammar

Each chunk contains exactly one command:

```text
SETV V0 3
SETM M1 2 4
TRANSPORT V0 THROUGH M1
COMPOSE M1 AFTER M0
SWAPV
SWAPM
```

Semantics:

- `SETV Vi c`: V_i <- c
- `SETM Mi a b`: M_i(x) <- ax + b
- `TRANSPORT Vi THROUGH Mj`: V_i <- M_j(V_i)
- `COMPOSE Mi AFTER Mj`: M_i <- M_i . M_j
- `SWAPV`: exchange V0, V1
- `SWAPM`: exchange M0, M1

Every nonfinal chunk must produce the same literal output: `ACK`.

The ACK is not supplied to the following chunk. The base model's KV cache and
input transcript are discarded after every chunk. Only the candidate
persistent-state mechanism survives.

Use one initialization chunk, 4-12 update chunks, and one final query. Every
scored trajectory must contain at least one COMPOSE and one TRANSPORT.

### Late-bound query family

The final query is selected only after the history is complete:

```text
READ V0
READ V1
EVAL M0 AT 0
EVAL M0 AT 1
EVAL M1 AT 0
EVAL M1 AT 1
APPLY M0 TO V0
APPLY M0 TO V1
APPLY M1 TO V0
APPLY M1 TO V1
APPLY M0 AFTER M1 TO V0
APPLY M0 AFTER M1 TO V1
APPLY M1 AFTER M0 TO V0
APPLY M1 AFTER M0 TO V1
```

The first six queries identify all 10,000 states: the value queries recover
V0, V1, while evaluation at 0 and 1 uniquely recovers each affine map. The
remaining queries test composition rather than mere storage.

Therefore the mechanism cannot know which answer to prepare before the last
reset.

### Required counterfactual groups

For every exact final query string, generate matched histories producing all
five logical answers. Match:

- history length;
- command-template sequence;
- final pre-query command;
- renderer;
- query;
- intermediate outputs.

Only earlier command arguments vary.

Example histories:

```text
History A init: V0=0 V1=4 M0=(2,1) M1=(3,0)
History B init: V0=1 V1=4 M0=(2,1) M1=(3,0)

Both:
TRANSPORT V0 THROUGH M0
COMPOSE M1 AFTER M0
```

After composition, M1(x) = x + 3. History A ends with V0 = 1; history B with
V0 = 3.

Both receive the identical final chunk:

```text
RENDER 0=D 1=A 2=E 3=B 4=C
QUERY APPLY M1 TO V0
ANSWER
```

History A's logical result is 4, so its answer is C. History B's result is 1,
so its answer is A.

This is the central private-state witness: identical final observations require
different answers.

---

## 2. Renderer crossing

The output codebook is disclosed only in the final chunk. It is not known when
persistent state is written.

Use five tokenizer-audited, equal-length, single-token outputs, provisionally
A-E. Before training, verify that they have identical token counts in the
frozen model's tokenizer.

For every logical result l:

- cross it with every output token;
- use all 120 permutations in the formal audit, or balanced five-permutation
  Latin blocks in materialized datasets;
- independently balance the order of codebook clauses.

Required audit table (each cell must contain equal count N):

| Logical value | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| 0 | N | N | N | N | N |
| 1 | N | N | N | N | N |
| 2 | N | N | N | N | N |
| 3 | N | N | N | N | N |
| 4 | N | N | N | N | N |

Consequences:

- I(logical answer; output token) = 0 marginally.
- Query-only, renderer-only, last-chunk-only, and ACK-history baselines have
  exactly 20% Bayes accuracy.
- Persistent state must encode query-relevant logical state, not a
  predetermined output token.
- A renderer change must change the output without changing the pre-query
  persistent state.

---

## 3. Competing hypotheses

| Hypothesis | Predicted result | What success means | What failure means |
|---|---|---|---|
| Hand-coded symbolic cache | 100% on every trajectory, renderer, composition, and horizon | The task is coherent and ordinary explicit persistent state is sufficient. This is an oracle, not evidence for HANDLE. | Generator, reference interpreter, or harness is wrong. Do not train. |
| Direct answer register | A genuinely answer-specific five-way register cannot solve the late-bound query family | If it nevertheless succeeds, it either received leaked query/output information or encoded more than one answer. | Supports the claim that one prepared answer is insufficient, but not that slots are required. |
| Single unrestricted learned vector | Can encode all 10,000 states; no theoretical behavioral failure is guaranteed | Private recurrence suffices. Behavior alone does not establish typed slots. | A HANDLE advantage would suggest an inductive-bias benefit, subject to optimization and seed controls. |
| Matched-parameter generic RNN | Computationally capable of solving the task | If it matches HANDLE on systematic holdouts and causal tests, explicit HANDLE typing has not added demonstrated value. | If HANDLE passes consistently while it fails, that supports a slot-typing/generalization advantage -- not logical necessity. |
| HANDLE | Should solve all queries, lengths, renderers, and held-out recombinations | Behavioral success admits the later mechanism card. It does not by itself prove slot use. | Failure on the symbolic positive-control task kills HANDLE. |

Important boundary: an unrestricted continuous "answer register" has far more
than 13.29 bits of capacity. The formal state-count result only rules out a
register committed to one of five final answers. It does not rule out a vector
that encodes the whole machine state.

---

## 4. Distinguishing observations

The no-training deliverable should produce three machine-auditable artifacts:

### 4a. state_query_matrix

Enumerate all 10,000 states and their 14-query answer vectors. Verify that the
six separating queries produce 10,000 distinct signatures.

### 4b. renderer_matrix

Verify exact balance of logical values against output tokens and exact 20%
conditional baselines.

### 4c. counterfactual_patch_oracle

For every state, slot, and matched donor, calculate the answer vector for the
hybrid state: s[V0 <- V0'], s[V1 <- V1'], s[M0 <- M0'], s[M1 <- M1'].

This oracle defines the later causal test:

- Patch one HANDLE slot from a donor history.
- Run all 14 final queries.
- Require the complete answer vector to equal the symbolic hybrid-state
  prediction.
- Require queries outside that slot's dependency cone to remain unchanged.
- Repeat under every renderer.

### Preregistered measurements

- **Hybrid exact-vector accuracy:** all 14 answers match the hybrid oracle.
- **Dependency-cone precision and recall:** only symbolically dependent queries
  change.
- **Renderer invariance:** the logical hybrid result is constant across
  renderers.
- **Permutation equivariance:** renaming V0<->V1 or M0<->M1 permutes states
  and answers as specified.
- **Composition-law error:** patched/composed maps agree with sequential
  application on every x in Z_5.
- **Matched intervention gap:** slot patching must beat whole-vector donor
  patches, matched-norm random patches, and arbitrary coordinate-block patches.

### Expected interpretation

- A direct answer code should not generate correct novel hybrid answer vectors.
- A monolithic RNN may behave correctly but lack a stable factorwise causal
  patch.
- HANDLE should expose four independently patchable factors.
- A symbolic cache will also pass perfectly.

That last point is fundamental: no input/output dataset can distinguish HANDLE
from a hand-coded cache implementing the same state algebra. The cache is the
mechanistic gold standard. HANDLE can only demonstrate that it learned a
cache-like typed mechanism and that explicit typing improves learning or
systematic generalization over the matched recurrent control.

---

## 5. Paper-stage kill conditions

No model training is licensed if any condition holds:

1. **Final-chunk leakage:** answer accuracy above 20% is possible from the
   final chunk, renderer, history length, command templates, last command, or
   ACK transcript without private state.
2. **Missing collisions:** any scored final-query family lacks exact-query
   groups containing all five logical answers.
3. **Renderer imbalance:** logical values are associated with output tokens,
   token lengths, legend positions, or tokenizer segmentation.
4. **Insufficient state separation:** the registered query family produces
   fewer than 10,000 distinct state signatures.
5. **Answer-precomputation:** the final query or renderer is known before the
   final reset, allowing the mechanism to store only the output token.
6. **Fake composition:** removing COMPOSE and TRANSPORT, or replacing them with
   direct writes, leaves an equivalent task.
7. **Broken symbolic ceiling:** the reference cache/interpreter scores below
   100%.
8. **Unfair competitors:** HANDLE receives parsed operations, slot addresses,
   extra state width, or query information unavailable to the cache and RNN
   controls.
9. **No unique causal prediction:** the preregistration contains only
   behavioral accuracy, with no hybrid-patch, locality, or equivariance tests.
10. **Claim overreach:** the intended conclusion remains that HANDLE beats or
    is necessary relative to the symbolic cache. The task cannot establish that
    conclusion.

---

## 6. Verdict

**Conditional admission as a private-state identifiability task.** Proceed only
under the narrower claim that HANDLE may learn a causally factorized,
cache-like state mechanism more reliably than an equally budgeted unstructured
recurrent controller. Behavioral accuracy alone is a control result.

This satisfies the design-freeze requirement that failure be unambiguous and
carries the behavior, null, output-only, holdout, failure, and
later-intervention obligations from the
[experiment roadmap](03_EXPERIMENT_ROADMAP.md) and
[mechanism-card contract](05_MECHANISM_CARD_CONTRACT.md).

---

## Formal verification (CPU, pre-training)

Codex verified with Python:

```
states = 10,000  (5^2 x 20^2)
distinct_separating_signatures = 10,000  (all states distinguishable)
zero_error_bits = 13.29
renderer_counts_per_logical = [[1,1,1,1,1], [1,1,1,1,1], ...]  (exactly balanced)
```

No repository files were modified by this design review.
