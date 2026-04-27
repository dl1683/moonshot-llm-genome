# Prefix-Information Transport Principle

**Status:** Derivation skeleton (not yet theorem-grade). Falsifiable.
**Origin:** Codex first-principles consult, 2026-04-26.
**Source:** `codex_outputs/first_principles_derivation.md`

## g152 long-horizon crossover (2026-04-26): theory's ATTENUATION prediction observed

The transport theory predicts MLP parameters are wasted only until the transport gap closes (η_ℓ > δ_ℓ^mlp). At larger compute, the transport gap should close, so the architecture-prior advantage should SHRINK. g152's 3-seed long-horizon crossover at 200M gave exactly this trajectory:

| (base_step, min_step) | C4 gap | OOD gap |
|---|---:|---:|
| (4000, 8000)   | +0.54pp | +1.03pp |
| (8000, 16000)  | +1.60pp | +1.70pp ← peak |
| (16000, 32000) | +0.69pp | +0.96pp |
| (25000, 50000) | +0.27pp | +0.45pp ← final |

The peak is at modest compute and the gap monotonically attenuates after, while remaining strictly positive. This is **consistent** with transport-gap-closes-with-compute. **It is consistency evidence, not discrimination** — the same trajectory is also compatible with a banal "smaller transport-heavy arm is more compute-efficient early; the larger MLP arm catches up toward parity later" reading. So g152 alone does NOT validate the theory.

**Statistical caveat (Codex 2026-04-26):** the final-checkpoint 3-seed paired-gap 95% CIs *include zero* (C4: [-0.42, +0.95]pp; OOD: [-0.06, +0.97]pp). The "small but persistent advantage" framing is at the noise floor. Power-law extrapolation projects the gap to ~0 by 1B-scale compute and below noise floor by 7B+. Honest framing: the long-run gap goes to practical zero.

g156 is the causal test along an ORTHOGONAL axis (data structure rather than compute) that can DISCRIMINATE between transport-control-variable and early-budget-artifact. g158 extends along context length. The triangulation of all three axes (compute, data order, context length) is what makes the theory load-bearing — no single trajectory does. g152 alone strengthens the case for running g156; it does not strengthen the thesis on its own.

## g156 prefix-destruction killer test (2026-04-26): PASS_TRANSPORT ★★★

The orthogonal-axis discrimination test the theory required. 3 seeds × 2 conditions × 2 arms = 12 cells at 200M, arm-specific best LRs from g151.

| condition | baseline top-1 | minimal top-1 | Δ |
|---|---:|---:|---:|
| natural c4 | 18.38% | 18.94% | **+0.560pp** |
| token_shuffled | 4.51% | 4.31% | **−0.197pp** |

Contrast: **C = Δ_nat − Δ_shuf = +0.757pp**. All three PASS_TRANSPORT thresholds cleared (≥0.5, ≤0.1, ≥0.4). Per-seed std small (~0.05pp baseline, ~0.15pp minimal) so signal dominates noise here unlike g152's final-checkpoint borderline.

**Combined with g152's compute-axis attenuation, this is now cross-axis falsifiable evidence.** The architecture-prior win exists in natural condition, attenuates as compute closes the transport gap (g152), and INVERTS when ordered prefix information is destroyed (g156). Two orthogonal control axes, both behaving as the theory predicts.

The Codex §0.1 publishability score moves from 4/10 (narrow regime-specific empirical effect) to 6/10 (serious theory lead with cross-axis falsifiable evidence). Closing the gap to a "measured design law" requires the locked post-g156 program (g157 η/δ probe, g158 context-length inversion, g159 cross-class lesion, g160 transport-guided student, g161 RWKV training).

## Empirical record this is meant to derive

In this repo's matched-budget protocol, a smaller **MLP-free attention+residual** model beats a larger attention+MLP baseline across 30M → 100M → 200M, and the win survives arm-specific tuning (`g138`, `g141`, `g146`, `g147`, `g148`, `g151`).

Codex audit correction: the repo's "matched-FLOPs" framing is operationally matched-by-step-count at different param counts (baseline ~209M @ 4000 steps, minimal ~81M @ 8000 steps at 200M scale). The honest claim is *not* "equal-params + equal-FLOPs" but "this matched-budget protocol favors MLP-free attention+residual."

## The principle

**The dominant bottleneck in shallow-to-mid autoregressive LMs is transporting useful prefix information into the current-token state — not locally synthesizing extra tokenwise nonlinear features after that transport.**

- Width `d` is the channel size for prefix information.
- Residuals preserve transported information across layers.
- Attention strictly increases mutual information between current-token state and prefix: `I(H_t^{ℓ+1}; X_{<t}) > I(H_t^ℓ; X_{<t})` is achievable only via attention.
- Token-local MLP sublayers `H_t^{ℓ+1} = H_t^ℓ + g_φ(H_t^ℓ)` are deterministic functions of the same token; by data processing they cannot increase prefix information at that token. They can only re-encode what attention has already transported.

Until the transport gap is closed, MLP parameters are *worse spent* than additional attention/width/residual parameters.

## Formal core

Let `H_t^ℓ` be the hidden state at token t, layer ℓ.

**MLP sublayer:** `H̃_t^{ℓ+1} = H_t^ℓ + g_φ(H_t^ℓ)` — deterministic function of `H_t^ℓ`, hence

```
I(H̃_t^{ℓ+1}; X_{<t}) ≤ I(H_t^ℓ; X_{<t})    (data processing)
```

The MLP sublayer cannot CREATE prefix information at token t.

**Attention sublayer:** `H_t^{ℓ+1} = H_t^ℓ + Σ_{j≤t} α_{tj}(H^ℓ) U H_j^ℓ` — depends on `H_{<t}^ℓ`, hence can *strictly increase* `I(H_t^{ℓ+1}; X_{<t})`.

The next-token CE objective satisfies `L_t ≥ H(X_{t+1} | H_t^L)`, so reducing CE requires raising `I(X_{t+1}; H_t^L)`. When the task is context-dominated, that reduces to raising `I(H_t^L; X_{<t})`. Only attention can do that.

## Budget criterion

Define:
- `η_ℓ := I(X_{t+1}; X_{<t} | H_t^ℓ)` — remaining transport gap after layer ℓ
- `δ_ℓ^mlp := inf_{f ∈ G_local} [ H(X_{t+1} | H_t^ℓ) − H(X_{t+1} | f(H_t^ℓ)) ]` — best gain from any token-local nonlinear decoder

The next unit of budget should go to **attention** iff
```
η_ℓ > δ_ℓ^mlp.
```

The empirical g138-g151 results imply that in this regime `η_ℓ` is still the larger quantity, so MLP parameters are a worse spend than additional attention transport.

## Why this is the breakthrough-axis answer

Per CLAUDE.md §0.1, the distinctive moves big labs cannot publish are first-principles derivations of phenomena that contradict the "bigger model = better" product story. This principle:

1. **Is mechanistically right-shaped** — explains *why* the architecture-prior is localized to attention + width + residuals (g138 finding) without hand-waving.
2. **Is brutally falsifiable** — destroying ordered prefix information (token-shuffled C4) should collapse the minimal win.
3. **Is product-conflict for big labs** — saying "MLP parameters are waste" undercuts their "more parameters in deeper FFN stacks" sales pitch.
4. **Subsumes Candidate 1** (operator-energy / cross-token Jacobian) as its linearized shadow.
5. **Predicts inversion regime** — when transport gap is small (very short context, context-destroyed data, much-deeper-saturated models), MLP budget should overtake attention budget.

## Falsifying experiment (`genome_156_prefix_destruction_200m`)

See `research/prereg/genome_156_prefix_destruction_200m_2026-04-26.md`.

Two stimulus conditions on the same baseline / minimal arms from g147/g151:
- `natural`: standard `c4_clean_v1`
- `token_shuffled`: per-sequence permutation destroying prefix order while preserving token marginals

Predicted:
- `Δ_nat` (minimal − baseline) ≥ +0.5pp (replication of g147 win)
- `Δ_shuf` ≤ +0.1pp (transport gap destroyed → win collapses)
- Support statistic `C := Δ_nat − Δ_shuf` ≥ +0.4pp.

If `Δ_shuf` stays within 0.2pp of `Δ_nat`, the transport theory is badly damaged.

## What this is NOT

This is NOT yet a theorem. The missing piece is a measurement primitive that estimates `η_ℓ` and `δ_ℓ^mlp` directly from a trained model. Until that primitive exists, the honest framing is **"derivation skeleton, falsifiable, predicts the right inversion regime."**

Future work after the destruction experiment:
1. Build the layerwise mutual-information probe (`I(H_t^ℓ; X_{<t})` estimator over text)
2. Test the prediction `η_ℓ > δ_ℓ^mlp` directly on g141/g147 checkpoints
3. Extend across architectures (Mamba, RWKV) — does the principle predict their relative MLP-vs-attention allocation?

## Related candidates that DID NOT survive

Codex's audit rejected:
- **Candidate 2 (statistical mechanics / spin-glass)**: superlinear MLP sample-complexity claim is unsupported by the cited Mei/Misiakiewicz/Montanari literature.
- **Candidate 4 (spectral invariant `√eff_rank · α`)**: already established epiphenomenal, not causal (g135). Diagnostic constraint, not mechanism.

Kept as secondary:
- **Candidate 1 (operator-energy / Jacobian decomposition)**: linearized shadow of Candidate 5; gives a clean budget-criterion `prefer_attn iff E_cross/E_local ≳ 3m/4d` (≈ 1.7-2.0 for our configs).
- **Candidate 3 (rate-distortion / Zipfian codebook)**: complementary explanatory prior — static MLP codebook vs dynamic attention codebook over heavy-tailed contexts. Subsumed by transport once the η/δ probe exists.
