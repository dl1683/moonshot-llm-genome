# Theory revision (2026-04-26 FINAL at PILOT scale)

**Status:** FINAL at PILOT scale — g157b returned KILL_157b at 22:55 with eta-only contrast = **-0.51 nats** (wrong sign). Both probe variants (same-layer in v2, embedding-layer in b) reject the η > δ^mlp criterion. The mechanism is empirically falsified across two independent probe paradigms.

Multi-seed canonical verdict (g157c) is NOT being run because the PILOT direction is wrong-signed, not just below threshold. Re-running with 3 seeds will not recover a wrong-signed pilot result. Code/preregs for g157c, g157d, g157 v3 retained for the audit trail; archived not run.

## Observed pattern (g157 v2 + g157b FINAL, single seed)

| Arm × Condition | eta (CE_local − CE_prefix) | Sign |
|---|---:|---|
| natural-baseline | -0.45 | NEGATIVE |
| natural-minimal | -0.39 | NEGATIVE |
| shuffled-baseline | +0.18 | POSITIVE |
| shuffled-minimal | +0.11 | POSITIVE |

**Pattern is BY-CONDITION (natural<0, shuffled>0), NOT BY-ARM.** Confirmed across both probe variants (same-layer prefix in v2, embedding-layer prefix in b).

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

## Realized outcome: by-condition pattern confirmed (mechanism rejected)

shuf-minimal eta = +0.11, exactly as the by-condition hypothesis predicted. ALL natural arms have eta < 0 and ALL shuffled arms have eta > 0. The theory's mechanism (η > δ^mlp would discriminate transport-heavy from local-heavy ARCHITECTURES) is rejected.

**Actions taken:**
1. ✅ g157 family REJECTED in CLAIM_EVIDENCE_MAP (P12 → R7)
2. ✅ Derivation doc references updated (this file is canonical)
3. ✅ Skipped g157c, g157d, g157 v3 — wrong-signed PILOT result not recoverable by re-running
4. ✅ Launched g158 (context-length inversion) at 22:55 immediately upon KILL_157b verdict
5. **Active:** post-g156 program continues with g158 → g159 → g160 → g161. The breakthrough-axis claim re-frames as "empirical architecture-prior + cross-axis falsifying evidence" rather than "first-principles derivation with internal-mechanism validation."

## §0.1 score implications

**Current: 6/10** (empirical g156 PASS + g152 PARTIAL; mechanism candidate rejected).

**Projected paths to 7/10:**
- g158 PASS_INVERSION (context-length is control variable) → 6.5/10
- g158 + g159 PASS (cross-class extension) → 7/10
- g158 + g159 + g160 PASS (manifesto cash-out: matched-cost transport-heavy student wins) → 7-8/10

**Without internal-mechanism validation, ceiling is ~7-8/10.** A "first-principles derivation that big labs can't publish" was the path to 9-10/10; this is now closed via the η/δ probe paradigm.

## Open theoretical question

The empirical g156 PASS_TRANSPORT (Δ_nat=+0.56pp, Δ_shuf=−0.20pp, C=+0.76pp at 200M, 3 seeds) is REAL. It needs SOME mechanism explanation. The η > δ^mlp criterion was the Codex-identified candidate; that candidate is rejected. Open candidates:

1. **Rate-distortion / Zipfian** (Codex's Candidate 3): static MLP codebook vs dynamic attention codebook over heavy-tailed contexts. NOT yet tested.
2. **Operator-energy** (Codex's Candidate 1): cross-token Jacobian decomposition. Linearized shadow of transport-info. NOT yet tested.
3. **Better idea**: the natural-vs-shuffled sign flip (eta_nat<0, eta_shuf>0) IS information about the model's transport behavior — but it measures the END STATE of training, not the training-time bottleneck. Could a trajectory-level probe (eta_l vs training step) detect transport gap during training rather than at convergence? UNTESTED.

These are research directions, not currently locked in any prereg. If g158/g159/g160 also fail to validate a derivation route, we pivot to: empirical chain only, distillation product (g155) as manifesto cash-out.
