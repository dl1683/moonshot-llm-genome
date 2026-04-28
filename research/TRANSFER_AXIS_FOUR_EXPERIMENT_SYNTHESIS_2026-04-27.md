# Four-experiment synthesis: continuous optimization constraint as active ingredient

**Date.** 2026-04-27 evening, after g169 FAIL.
**Status.** Pattern locked across 4 distinct experiments. Awaiting g167 (KD logits) for the 5th data point.

## The pattern

| Experiment | Mechanism | Schedule | Outcome | Magnitude |
|---|---|---|---:|---:|
| g165 anchor_lam0.01_constant | Frobenius weight anchor | constant λ=0.01 | **PASS** | +1.088 nats |
| g165 anchor_lam0.0013_constant | Frobenius weight anchor | constant λ=0.0013 | **PASS** | +0.717 nats |
| g165 anchor_lam0.00013_constant | Frobenius weight anchor | constant λ=0.00013 | WEAK | +0.274 nats |
| g165 decay arms (12 of 12) | Frobenius weight anchor | step / linear / exp decay | **FAIL** | mostly ~0 |
| g168 identity / raw_copy | Zero-step weight transplant | none (zero training) | **FAIL** | -0.023 |
| g168 permutation_only | Aligned zero-step transplant | none | **FAIL** | +0.003 |
| g168 norm_refit | Aligned zero-step transplant | none | **FAIL** | -0.013 (zero-step) / +0.438 (step=50) |
| g169 scaffold_step | Activation-level scaffold | hard cutoff at step 50 | **FAIL** | -0.388 |
| g169 scaffold_linear | Activation-level scaffold | linear decay to 0 by step 250 | **FAIL** | -0.207 |
| g169 scaffold_exponential | Activation-level scaffold | exp decay τ=50 | **FAIL** | -0.119 |

## What works (n=3 out of 21 anchored arms tested)

ONLY mechanism: **continuous Frobenius weight anchor at λ ≥ 1e-4**, applied throughout 500 steps of SGD training. Persistence is monotone in anchor strength. Stronger anchor → larger persistent advantage.

## What fails (n=18 out of 21)

- ANY decay schedule, regardless of mechanism (weight or activation)
- ANY zero-step weight injection, regardless of alignment (raw, permutation, norm-refit, combined)
- Hard cutoff (alpha=1 then 0)
- Linear / exponential gradual decay
- Attention-only anchor with hard-cut (g165's `anchor_attn_only_lam1.3e-3_hardcut` arm: -0.012)

## The active ingredient hypothesis

**Continuous optimization constraint** is the active ingredient — NOT the donor weights themselves, NOT the activation-level scaffolding, NOT initialization choice, NOT alignment.

Specifically: the constraint must be applied **at every gradient step**, with magnitude that does NOT decay to zero. The constraint must shape the loss landscape that SGD navigates, throughout training. If the constraint is removed (decay to 0) or never applied (zero-step), the recipient drifts to its own scratch-equivalent solution.

This is consistent with a **basin-of-attraction interpretation**: the donor occupies a particular region of weight space. The Frobenius anchor adds a quadratic bowl centered on the donor's weights to the loss landscape. SGD with this modified landscape converges to a compromise point near the donor's basin. Without the anchor, SGD finds the same basin scratch finds. The anchor doesn't "transfer knowledge" — it constrains where SGD can converge.

## Falsifiable prediction

If the basin-of-attraction interpretation is correct, then:
- ANY mechanism that adds a continuous quadratic-style bowl on the donor's weight position should produce a similar persistent advantage.
- Mechanisms that don't constrain weight positions (KD logits, attention routing, RL signal) are operating on a different axis — they may or may not transfer capability.
- g167 (KD canonical) is testing one such axis — KD operates on output logit distribution, not weight positions. If g167 PASSes, KD transfers capability via a fundamentally different mechanism than g165's anchor.

## Implications for §0 capability transfer

The project's stated end goal is "efficient transfer of trained capabilities... without retraining the recipient."

**Hard finding from g168**: zero-step weight injection (no retraining) is empirically dead, even with full alignment. The "without retraining" reading of §0 cannot be satisfied via weight-space mechanisms.

**Soft finding from g165**: persistent capability transfer IS achievable (+1.088 nats) but requires "retraining-with-anchor" — recipient still trains, but its training is constrained by the donor.

So §0 should be reframed: **"efficient transfer of trained capability via cheap recipient-side training, where the donor anchors the recipient's training landscape rather than initializing it."**

This is a meaningful but more honest framing than "without retraining."

## What to test next

1. **g167 (running)**: KD logits — does output-distribution constraint give similar persistence? If yes, two distinct transfer mechanisms validated.
2. **g170 (next)**: transport-gated KD — does weighting KD by g158c's transport-demand variable amplify the effect?
3. **g171**: attention-routing KD — does transferring attention maps (not weights, not full logits) work?
4. **Constant-anchor strength sweep at higher λ**: does the persistence keep growing? Or saturate?
5. **Cross-architecture variant**: can g165's anchor be applied between Qwen3 donor and a DIFFERENT-architecture recipient (e.g., Llama-arch)? The basin-of-attraction theory predicts this fails (no shared weight space).

## Provenance

- g165 results: `results/genome_165_annealed_donor.json`
- g168 results: `results/genome_168_rebasin_zero_step_transplant.json`
- g169 results: `results/genome_169_scaffold_swap_distillation.json`
- Codex transfer-axis rethink: `codex_outputs/transfer_axis_rethink_20260427T200000.md`
- Codex cycle 36 direction review (framing as "continuous-anchor persistence law"): `codex_outputs/heartbeats/cycle36_direction_review_20260427T210000.md`
