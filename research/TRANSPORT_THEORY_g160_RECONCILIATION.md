# Reconciling g158 PASS-direction with g160 PILOT_KILL

**Status.** Theoretical note authored 2026-04-27 during g158c canonical run. Captures an observation that may refine the transport theory's scope of applicability. **Not** a claim — a hypothesis to be tested.

## The apparent tension

The transport theory has now been stress-tested on two dimensions:

| Test | Setup | Verdict |
|---|---|---|
| **g158 PILOT** (input-side prediction) | Vary context length L at fixed architecture; minimal_3L_noMLP vs baseline_6L+MLP, both trained from scratch | DIRECTIONAL_SUPPORT: rho(L, Delta)=+1.00, Delta_256=+4.10pp, Delta_32=-0.24pp (sign inversion) |
| **g160 PILOT** (design-rule prediction) | Fix matched inference FLOPs (4.03 GFLOP); transport_heavy 6L noMLP h512 vs local_heavy 4L MLP h384 ffn1024, both **distilled** from same Qwen3-1.7B teacher | PILOT_KILL: C3 gap=-0.34pp, CtQ_90 ratio=1.00 |

These appear contradictory at first read: if transport demand is the control variable (g158), why doesn't allocating more capacity to transport beat allocating more to local processing (g160)?

## Hypothesis: the theory is scoped to training-from-scratch, not to distillation

The two experiments differ on a critical axis the prereg did not isolate:

- **g158 trains both arms from scratch on natural C4.** Each arm must DISCOVER prefix-information transport from data alone. Absence of MLP forces the architecture to rely on attention + residual stream for transport, and the inductive bias becomes the dominant determinant of capability at long L.
- **g160 distills both arms from a frozen Qwen3-1.7B teacher.** The teacher's logits already encode prefix-transport-rich representations. The student doesn't need to discover transport; it just needs sufficient capacity to **mimic** the teacher's logit distribution.

Under this lens:
- Distillation injects a strong supervised signal that **compensates** for missing transport inductive bias. A student with MLP layers can use them as soft "shortcut" feature extractors that reproduce teacher logits via local processing, even though the teacher itself uses transport-heavy attention.
- The architecture-prior advantage of MLP-free designs is therefore a property of **what the architecture must DISCOVER** under self-supervised training pressure, not a property of **what it can REPRESENT** at convergence.
- Consequence: at distillation, the local_heavy arm catches up because it has enough capacity to mimic the teacher's outputs; the transport-heavy arm has no advantage because the teacher's signal removes the need to discover transport.

## Falsifiable predictions of this scope-refinement

1. **At smaller distillation budgets** (e.g., 100 steps instead of 4000), the transport_heavy arm should win because the local_heavy arm hasn't yet exploited teacher signal to compensate.
2. **At larger distillation temperatures** (T >> 1, softening teacher logits to remove transport-specific information), the architecture-prior inversion should re-emerge.
3. **At pure-CE supervised training matched-FLOPs** (no teacher), the architecture-prior advantage should hold on the same FLOP-matched design pair as g160 — this is the prediction g160 was trying to make but tested under distillation instead.

## Connection to g158c verdict outcomes

- If **g158c PASS_canonical**: the input-side prediction holds under training-from-scratch, supporting the scope-refinement (theory is alive in the from-scratch regime).
- If **g158c WEAK_canonical**: the from-scratch prediction is also softer than expected, weakening the scope-refinement.
- If **g158c PILOT_FRAGILE**: the from-scratch prediction also fails canonical, the scope-refinement is unfalsifiable (we'd be defending a theory only on PILOT data).

## Implication for next experiments

Conditional on g158c PASS_canonical, the highest-leverage **mechanism** experiment that distinguishes "discovery-time-only" from "convergence" interpretations is:

- **g162 from-scratch matched-FLOPs design comparison** (parallel to g160 but **without distillation**): same student configurations as g160, train from scratch on C4 at matched 4.03 GFLOP each, compare C3_macro. Predicted by the scope-refinement: transport_heavy beats local_heavy (the inverse of g160's null).

This is a single ~3hr run that directly tests the scope refinement. Codex direction review is asked in the post-g158c design consult to adjudicate whether this is the right Path A move or whether a causal head-ablation is higher leverage.

## Honest caveats

1. This is post-hoc interpretation, not a pre-registered prediction. The g160 prereg did not separate distillation regime from from-scratch regime as a critical variable; the rationale was "matched FLOPs is the manifesto-aligned cash-out and KD is how you cash it out."
2. The scope-refinement reduces the theory's claim. Originally the principle was supposed to be a model-selection law; under this refinement it becomes "a model-selection law at training-from-scratch, but distillation washes it out." That's a weaker claim — but a more honest one.
3. If g158c also fails, this note becomes irrelevant and the theory dies in both regimes.
