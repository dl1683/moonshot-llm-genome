# Provisional theory revision (2026-04-26, written before g157b complete)

**Status:** PROVISIONAL — written while g157b is still running. Will be finalized after the canonical 3-seed verdict (g157c) if PILOT survives, or recanted if PILOT changes.

## Observed pattern (g157 v2 + g157b in flight, single seed)

| Arm × Condition | eta (CE_local − CE_prefix) | Sign |
|---|---:|---|
| natural-baseline | -0.45 | NEGATIVE |
| natural-minimal | -0.41 | NEGATIVE |
| shuffled-baseline | +0.18 | POSITIVE |
| shuffled-minimal (pending) | TBD | TBD |

The theory's prediction was: eta > 0 on natural-minimal (transport gap unsolved), eta ≤ 0 on shuffled (no prefix info to use).

**Observation: opposite signs.** Natural arms have eta < 0; shuffled arms have eta > 0.

## Reinterpretation

**Hypothesis:** the theory's "transport gap" framing was right but the η > δ^mlp criterion measures the WRONG thing.

What we actually observe is consistent with:
- **Natural training** successfully transports prefix info into h_t. q_local extracts it from h_t alone. q_prefix has nothing extra to add → eta < 0.
- **Shuffled training** fails to transport much (shuffled is too hard to fit cleanly). q_prefix can extract un-transported info from embed(prefix) → eta > 0.

This is NOT "minimal arm has more transport gap"; it's "well-trained models close the gap, ill-trained ones don't." Tautological — distinguishing transformers from random init.

## What this means for the architecture-prior thesis

**The empirical g156 PASS_TRANSPORT stands.** Δ_nat=+0.56pp, Δ_shuf=−0.20pp, C=+0.76pp. Minimal architecture WINS on natural and LOSES on shuffled at 200M, 3 seeds.

**The proposed mechanism (η > δ^mlp transport budget criterion) is not supported.** The probe paradigm cannot show eta_nat_minimal > 0 because well-trained models close the transport gap regardless of architecture.

**Status of post_g156 program:**
- g158 (context-length inversion) — still meaningful, tests an INPUT-SIDE prediction (transport demand control variable)
- g159 (cross-class lesion) — still meaningful, tests architecture-class extension (no internal-quantity needed)
- g160 (transport-guided student) — still meaningful, tests model-selection rule (cashout)
- g161 (RWKV training) — still meaningful, tests architecture-class extension at training time
- **g157 series (η/δ probe) — likely DEAD as a primitive.** The criterion needs reformulation.

## Possible reformulations

1. **Cross-arm criterion:** eta_minimal − eta_baseline at matched compute. Predicts: minimal arm's eta is closer to 0 than baseline's because minimal has more transport demand at any layer. Tested on natural-only. But this is a contrast, not an absolute internal quantity.

2. **Layer-wise transport curve:** plot eta_l from layer 0 to n_layers−1. Theory predicts eta_l decreases with depth (info gets transported as we go). Test: does the slope differ between minimal and baseline?

3. **Counterfactual probe:** train q_prefix to predict TRANSPORTED-info-direction (e.g. PCA of h_t along training direction) rather than next token. Direct measurement of how much prefix info is in h_t.

These are all speculative. The conservative stance: g157 series produces NULL / inconclusive data; the theory's empirical predictions (g156, g158, g159) stand independently; the η/δ criterion as locked in PILOT does NOT discriminate.

## Decision: if g157b shuffled-minimal eta > 0

If shuf-minimal eta > 0 (matching shuf-baseline), then ALL natural arms have eta < 0 and ALL shuffled arms have eta > 0 — pattern is by-condition, not by-arm. The theory's mechanism is rejected.

**Action:**
1. Mark g157 family experiments as KILL_MECHANISM in CLAIM_EVIDENCE_MAP.
2. Update derivation doc to reflect theory's mechanism rejection.
3. Skip g157c (3-seed canonical) — PILOT direction is wrong-signed, not just below threshold.
4. Skip g157d (probe-budget expansion) — same reason.
5. Pivot to: launch g158 (context-length inversion, independent of η/δ) immediately when GPU frees.
6. Re-evaluate the post-g156 program. Without internal-mechanism validation, the §0.1 ceiling is ~6/10 (empirical chain only). Distillation track (g160 → g155) becomes the manifesto cash-out path.

## Decision: if g157b shuffled-minimal eta ≤ 0

Then the pattern is by-arm not by-condition: natural-* has eta < 0; shuffled has mixed signs. Worth investigating further with g157d (probe-budget expansion).

## Critical: this revision is PROVISIONAL

g157b might yet show:
- shuffled-minimal eta near 0 → ambiguous, run g157d
- shuffled-minimal eta far positive → confirms by-condition pattern, mechanism dies
- some other surprise

When g157b actually finishes, integrate_g157b.py will compute both criteria; this doc gets updated accordingly.
