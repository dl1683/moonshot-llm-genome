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

The peak is at modest compute and the gap monotonically attenuates after, while remaining strictly positive. This is consistent with transport-gap-closes-with-compute. It does NOT prove the theory (a single trajectory is not causal evidence), but it falsifies the alternative "transport doesn't matter at all and the win is artifactual" — the artifactual reading would predict either crossover (baseline overtakes) or a flat gap, neither observed.

g156 is the causal test of the same prediction along a DIFFERENT axis (data structure rather than compute). g158 would extend it along context length. The triangulation of these three axes (compute, data order, context length) is what makes the theory load-bearing.

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
