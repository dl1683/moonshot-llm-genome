# MC005 Associative Lookup Bounded Mechanism Closeout Status

Status: bounded mechanism card; full promotion blocked by null locality.

Date: 2026-07-01

## Diagnostic

Artifact diagnostic:

```text
mc005_bounded_attention_write_mediation
```

Canonical atlas diagnostic:

```text
MC005_BOUNDED_ATTENTION_WRITE_MEDIATION
```

## Verdict

MC005 is now frozen as a bounded mechanism card for Qwen3-1.7B associative
lookup under the tested `Response:` prompt contract.

The bounded positive claim is:

> Layers 24-26 final-query self-attention output writes mediate the tested
> source-value lookup effect on high-margin lookup rows.

The boundary is:

> The same write-replacement route is not strictly local on answer-absent null
> rows. It can flip rare low-to-moderate-margin target-versus-distractor null
> comparisons while leaving aggregate mean deltas clean.

This is not a promoted full mechanism card. It is also not a failed result. It
is the project's first calibrated bounded internal control surface.

## Gate Judgment

| Gate | Judgment | Evidence |
| --- | --- | --- |
| Signature | Passed for the lookup surface | source-value attention/path work and layers 24-26 localization through V21; V26 gives a separate row-heterogeneity signature but that route stays predictive-only |
| Intervention | Passed for the parent lookup surface | V29 attention-write replacement exactly reproduced the direct layers-24-26 target source-mask effect on lookup holdout rows |
| Reliability | Bounded, not full | lookup/off-target controls are strong, but V29-V31 reproduce rare answer-absent null-row flips under write replacement |

The mechanism card is therefore bounded:

```text
bounded_mechanism_card
```

not:

```text
promoted_mechanism_card
```

## Evidence Chain

MC005 progressed through three separable routes.

### Route 1: Source-Value Lookup Surface

The early route established that prompt-visible lookup can be behaviorally and
internally localized better than broad truth/agreement or factual-override
tasks.

Key results:

- V3 selected a late-band source-value masking surface.
- V9 passed a compact `Response:` atlas across lookup, source-line,
  off-target, and answer-absent null scenarios.
- V17 localized the parent surface to layers 23-26.
- V18 narrowed the useful parent to layers 24-26, recovering 92.4 percent of
  the parent effect on disjoint holdout rows.
- V20 showed layers 24-26 should remain a block because the single-layer and
  leave-one-layer-out decompositions did not recover the parent.
- V21 kept layers 24-26 fixed and passed layout, shifted-lexicon, pair-count,
  source-control, and fresh answer-absent null stress.

This route supports a real source-value lookup surface.

### Route 2: Row-Level All-Three Structure

The row-heterogeneity route is not the bounded mechanism card.

Key results:

- V22 found row-level all-three structure but missed the preregistered
  promotion threshold.
- V23 showed target source position and baseline margin explain part of the
  heterogeneity.
- V24 and V25 failed simple causal explanations based on source position and
  prompt-layout factorials.
- V26 found a predictive internal row signature.
- V27 additive residual steering failed to control the row split.
- V28 donor replacement failed and broke controls/nulls.

This route exports:

```text
SIGNATURE_NOT_CAUSAL
```

It should not be confused with the V29-V31 parent lookup mediation result.

### Route 3: Attention-Write Mediation

V29 returned to the supported parent lookup surface and tested whether the
layers 24-26 effect is specifically mediated by final-query attention output
writes.

The result was exact on the lookup holdout:

| Comparison | Mean Delta | Target-Win Loss |
| --- | ---: | ---: |
| Direct target source mask | -6.1978 | 11 |
| Target attention-write replacement | -6.1978 | 11 |

Source controls stayed aligned:

- distractor write replacement: +0.8223 mean delta, 0 target-win losses;
- random write replacement: +0.0239 mean delta, 0 target-win losses;
- single-layer write replacement did not recover the parent effect.

This is the strongest causal/localization result in the project.

## Null Boundary

V29 did not pass full reliability because one answer-absent null arm produced a
one-row target-win gain.

V30 showed the failure was not only sample-fragile:

- the V29 seed-251 `non_source_control_value` gain reproduced;
- an 8-seed fresh answer-absent sweep found three additional strict row-change
  failures;
- all fresh arms stayed inside the absolute mean-delta tolerance.

V31 tested whether a preregistered absolute baseline-margin cutoff of 0.5 fully
explained the boundary. It failed:

- the new fresh null flip was at baseline margin 0.0;
- the imported V30 replay flip had absolute baseline margin 0.75;
- lookup target-win losses were all high-margin, with minimum baseline margin
  5.25;
- all fresh null arms remained aggregate-mean clean.

The right boundary is therefore:

```text
rare low-to-moderate-margin answer-absent null flips
```

not:

```text
fully reliable null locality
```

and not:

```text
the 0.5 absolute margin cutoff explains everything
```

## Allowed Claims

- MC005 has a bounded internal mechanism for source-visible associative lookup
  under the tested Qwen3-1.7B `Response:` contract.
- The bounded mechanism is an all-head layers 24-26 aggregate, not a single
  head or single layer.
- Final-query self-attention output write replacement in layers 24-26 exactly
  reproduces the direct target source-mask lookup effect on the tested holdout.
- Lookup target effects are high-margin and separated from the observed
  answer-absent null flips.
- The answer-absent null boundary is real enough to block full promotion.

## Forbidden Claims

- MC005 is a full promoted mechanism card.
- MC005 generalizes beyond synthetic associative lookup.
- MC005 is a general knowledge or factual-recall mechanism.
- MC005 is a single-head or single-layer circuit.
- The V26 row-heterogeneity signature is causally controllable.
- The write-replacement null boundary is fully explained by a 0.5 absolute
  baseline-margin cutoff.
- The source-value surface transfers cleanly to smaller models with strict null
  reliability.

## Exported Rule

MC005 is the calibration specimen for the atlas.

Future mechanism-card attempts should be compared against it on two axes:

1. Can they produce a primary internal intervention as clean as the MC005
   high-margin lookup mediation?
2. Can they do better than MC005 on null locality, transfer, and side effects?

The lesson is not that bounded results are failures. The lesson is that a
mechanism card must state the boundary as carefully as the working surface.

## Next Decision

Do not keep repairing this exact write-replacement route as if full promotion
is one small tweak away.

Future MC005 work is justified only if it is one of:

- a materially different intervention family with a preregistered reason to
  suppress answer-absent null flips while preserving the lookup effect;
- a matched transfer panel that treats null reliability as a first-class gate;
- a comparative baseline for a new bridge family.

Otherwise, MC005 should remain frozen as:

```text
bounded_mechanism_card
```
