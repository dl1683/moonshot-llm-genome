# Session 2026-04-25 — Findings Summary

**Internal research notes, not a paper.** Consolidates 15 experiments completed in one session.

## The narrative arc

The session began with the surgery research thread (genome_119) and ended with cross-architecture universality of a training-dynamics phenomenon (genome_133). Two sub-arcs:

1. **Surgery is dead** (genome_119–125): six consecutive KILL results across two architectures (Pythia, Qwen3) and several transfer strategies (component copy, compound copy, scale calibration, layerwise FM, activation Procrustes, frozen-attn glue-train). The holism barrier is real and unbreakable by any naive zero-step weight-transfer strategy.

2. **The trained-spectrum invariant is a real training-dynamics phenomenon** (genome_126–133): population scaling, scale-invariant trajectory within Pythia, predictive utility within trajectory, architecture-universal U-shape across Pythia ≠ Llama.

## What we know now

### Surgery (genome_119–125)

- **All weight-subset transplants hurt** at zero steps. Best is `all_attn` at +0.6–0.9% gap closure, near-noise.
- **Compound surgery is worse than single-component.** Adding donor norms causes catastrophic damage (-77% with embed+attn+ln+zero_mlp on Qwen3).
- **Scale calibration via gamma = donor_rms / recipient_rms is catastrophic.** Extreme gamma values amplify activations.
- **Architecture-prior is unexpectedly strong** (genome_125): training only the embedding + LM head + RMSNorm gammas (26% of params) on a random-init Qwen3 with frozen random attention/MLP achieves 42.66% gap closure in 100 steps. Donor attention copy adds only +0.86 pp on top of this.

### Trained-spectrum invariant (genome_126–133)

- **Universal asymptotic value** ~4.243 (`sqrt(eff_rank) * alpha`) reached by all 11 fully-trained capable text LMs we measured (excluding GPT-Neo-125m as undertrained outlier). Population CV = 8.2% across 11 systems (looser than the original genome_088 N=5 CV of 5.1%).
- **Universal trajectory shape** within Pythia: random-init=9.6, mode-collapse-minimum=2.7-2.9 at step ~512, recovery to 4.0-4.2 by step 143000. Pythia-160m and Pythia-410m share IDENTICAL trajectory landmarks (min step, first-crossing, recovery point).
- **Capacity-dependent rate of traversal**: Pythia-1.4b reaches the same trajectory landmarks 4× earlier (min at step 128 vs step 512 for smaller sizes), and reaches lower mode-collapse minimum (2.56 vs 2.77-2.88).
- **Within-trajectory NLL prediction**: |inv − 4.243| predicts NLL across 16 Pythia checkpoints with Pearson r = 0.89.
- **Cross-architecture NLL prediction fails** (r = 0.18, N=9): the invariant tracks geometric convergence, not raw NLL across arch boundaries. Other factors (training-data quantity, capacity, optimization) dominate cross-arch.
- **Trajectory is architecture-universal**: trained a 30M-param Llama from scratch (RoPE + RMSNorm + SwiGLU + tied embed) and observed identical U-shape: random=6.83, mode-collapse-min=1.03 at step 128, recovery to 4.58 by step 4000. Mode-collapse is deeper and earlier in smaller Llama.

## Quick post-session analysis (no new experiment)

Re-analyzed g132's 9 cross-arch trained-LM data points:

| Predictor | Pearson r with NLL |
|---|---|
| `sqrt(er)*alpha` (universal invariant) | −0.17 (no signal) |
| `er` (eff_rank) | −0.51 |
| `log(er)` | −0.36 |
| **`alpha` (decay exponent)** | **+0.65** |
| `alpha^2` | +0.69 |
| 2-feature linear `(er, alpha)` | R² = 0.43 |

**Insight:** the universal invariant 4.243 is a FIXED POINT (all capable trained LMs converge to it), but the actual cross-arch NLL signal lives mostly in `alpha` alone (steeper decay = worse NLL). Within Pythia, alpha tracks NLL monotonically: 160m (α=0.77, NLL=3.51) > 410m (α=0.71, NLL=3.09) > 1.4b (α=0.65, NLL=2.83).

This refines the framing: capable models converge to a 1-D universal manifold (parametrized by the invariant), and within that manifold, lower α is "more capable." The trajectory observed in g127–g133 is a path that slides along this manifold during training.

The remaining 57% of cross-arch NLL variance (R²=0.43 fit) is non-spectral: training data, model capacity, optimizer settings, etc.

## What we don't know

1. **Why 4.243?** The constant 18 = (3√2)² is empirical. No first-principles derivation yet. Codex's Y-direction recommendation was to attempt this; the empirical work has now constrained the target enough that derivation has a fighting chance.

2. **How does mode-collapse depth scale?** Llama 30M dips to 1.03; Pythia 160M dips to 2.88; Pythia 1.4B dips to 2.56. Smaller-or-bigger axis is not clean. Need more data points.

3. **Is this a transformer-specific phenomenon, or general?** SSM/RNN architectures (Mamba, RWKV) untested. Windows + mamba-ssm kernel issues block direct tests; would need workaround.

4. **Does the invariant predict capability metrics (HellaSwag, ARC, MMLU)?** We tested NLL prediction (within-arch r=0.89, cross-arch r=0.18). Capability benchmarks not yet tested.

5. **Can the trajectory be SHORTCUT via initialization?** If we initialize weights such that the initial spectrum already sits at 4.243, does training proceed faster? genome_093 showed adding the invariant as auxiliary loss doesn't help (spectrum is symptom not cause). Initialization is different — untested.

## Connection to moonshot goal

The session's surgery work (genome_119–125) establishes a HARD NEGATIVE: zero-step weight-subset transfer doesn't work. The spectrum invariant work (genome_126–133) establishes a POSITIVE: there is universal geometric structure in trained models, characterized by a fixed point and a trajectory.

These two findings together suggest a refined moonshot path: capability transfer doesn't happen via weight surgery, but capability MIGHT be characterized by trajectory position. If we can identify which AXIS of the geometry carries capability (vs which axes are universal/decorative), targeted intervention along that axis becomes plausible.

The invariant `sqrt(er)*alpha = 4.243` is a SCALAR projection of a high-dimensional spectrum. It's universal — every trained capable text LM converges to it. So this scalar can't be the capability axis (it's the same for all). The capability axis must be ORTHOGONAL to this universal scalar — perhaps in the EIGENVECTOR DIRECTIONS, not the eigenvalue magnitudes.

This is a hypothesis worth testing: do trained models converge to the same eigenvalue spectrum but DIFFERENT eigenvector content? If yes, capability is in eigenvector content; transfer becomes a directional alignment problem.

## Code/data state

- All 15 experiments have ledger entries, EXPERIMENTS.md entries, results JSON, code committed and pushed.
- `WIKI.md` reflects every change.
- 14 commits on `main` today.
- Repo is clean (no stale files except some `*_run.log` files that are gitignored).

## Open candidates for next session

(In rough priority order, no commitment.)

A. **Eigenvector-content vs eigenvalue-spectrum analysis.** If invariant is universal but capability differs, capability lives in eigenvector direction.

B. **Mamba/RWKV trajectory** if mamba-ssm Windows issue can be worked around.

C. **Variational derivation of constant 18** — Codex Y direction. Most theoretical.

D. **Geometry-aware initialization** — does spectrum-prior init shortcut the trajectory? Risky given genome_093.

E. **Predicting capability benchmarks** (HellaSwag/ARC) instead of NLL — does the invariant work as a capability proxy?

F. **Larger Llama from scratch + longer training** to verify trajectory at scale.
