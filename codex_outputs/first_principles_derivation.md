# Architecture-Prior Derivation Report

Local evidence audited from [WIKI.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:835>), [WIKI.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:893>), [WIKI.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:984>), [WIKI.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:1004>), [WIKI.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:1024>), [WIKI.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:1083>), [code/genome_146_matched_flops_bigdata_100m.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_146_matched_flops_bigdata_100m.py:1>), [code/genome_147_matched_flops_200m.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_147_matched_flops_200m.py:1>), and [research/derivations/trained_spectrum_invariant.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/derivations/trained_spectrum_invariant.md:1>).

## 0. Audit First

The repo does **not** currently support the exact sentence “equal parameter count and equal training FLOPs.”

The local record is:

- `30M-scale`: baseline `29.93M` vs minimal `21.08M`, matched steps, not equal params.
- `100M-scale`: baseline `123.57M` vs minimal `52.78M`, `4000` vs `8000` steps, called “matched-FLOPs” in the repo but operationally matched by near-equal wall-clock.
- `200M-scale`: baseline `209.32M` vs minimal `80.86M`, `4000` vs `8000` steps, same caveat.
- `g151`: best-vs-best LR basin still favors minimal by `+0.65pp` C4 top-1 and `+0.52pp` OOD.

So the empirically supported claim is:

> In this repo’s matched-budget protocol, a **smaller MLP-free attention+residual model** beats a **larger attention+MLP baseline** across `30M -> 100M -> 200M`, and the win survives arm-specific tuning.

That is still strong enough to theorize about.

For the actual Llama blocks used here, the per-layer parameter split is

\[
P_{\text{att}} = 4d^2,\qquad P_{\text{mlp}} = 3dm.
\]

Hence the MLP consumes

\[
\frac{P_{\text{mlp}}}{P_{\text{att}}}=\frac{3m}{4d}
\]

which is `2.0` at the `30M/100M` configs and `1.6875` at `200M`. In other words: the MLP is eating about `63%–67%` of block parameters while not obviously being the winning compute allocation.

## 1. Candidate 1: Information-Theoretic / Effective-Rank

**A. Assumptions.**

Assume the useful layerwise computation for next-token prediction is mostly cross-token: token `t` needs context-dependent information from `1..t-1`. Assume the dominant object is the block Jacobian, not scalar nonlinearity count.

**B. Derivation.**

Write one residual block on token matrix \(X\in \mathbb R^{n\times d}\).

MLP:
\[
M(X)=\sigma(XW_1)W_2.
\]

Attention value path:
\[
A(X)=S(X)\,X\,U,\qquad U=W_VW_O,\qquad S(X)=\text{softmax}(QK^\top/\sqrt d).
\]

Linearize.

For the MLP, on a fixed activation region,
\[
d\,\mathrm{vec}(M)=\mathrm{diag}(B_1,\dots,B_n)\,d\,\mathrm{vec}(X),
\]
with \(B_i=W_2^\top D_i W_1^\top\), so \(\mathrm{rank}(B_i)\le \min(d,m)\). Crucially:
\[
\frac{\partial M_i}{\partial X_j}=0\quad (i\neq j).
\]

The operator is block-diagonal in sequence space.

For attention, even before differentiating the scores,
\[
d\,\mathrm{vec}(A)\approx (S\otimes U)\,d\,\mathrm{vec}(X)+\text{score terms},
\]
so
\[
\frac{\partial A_i}{\partial X_j}\approx s_{ij}U\neq 0.
\]

The operator is dense in sequence space. The point is not raw matrix rank alone; the point is **off-diagonal operator energy**.

Define the Bayes-optimal layer operator \(T^\star\) and decompose its Jacobian energy:
\[
E_{\text{cross}}=\mathbb E\sum_{i\neq j}\left\|\frac{\partial T^\star_i}{\partial X_j}\right\|_F^2,\qquad
E_{\text{local}}=\mathbb E\sum_i\left\|\frac{\partial T^\star_i}{\partial X_i}\right\|_F^2.
\]

MLP budget only attacks \(E_{\text{local}}\). Attention budget attacks both. A budget-first criterion is therefore

\[
\text{prefer attention over MLP if}\qquad
\frac{E_{\text{cross}}}{E_{\text{local}}}\gtrsim \frac{P_{\text{mlp}}}{P_{\text{att}}}=\frac{3m}{4d}.
\]

In this repo that threshold is about `1.7–2.0`. If the layerwise target operator is even moderately more cross-token than token-local, attention is the right place to spend parameters.

This is a real derivation skeleton. It is **not yet a theorem**, because \(E_{\text{cross}}\) is not measured.

**C. Failure / inversion regime.**

This story fails when \(E_{\text{local}}\) dominates: very short contexts, token-local transforms, continuous regression heads, or regimes where attention has already saturated the cross-token operator and the remaining error is decoder-local.

**D. Killer experiment.**

Destroy cross-token structure while preserving local token statistics. If the minimal win survives unchanged, this candidate dies.

## 2. Candidate 2: Statistical Mechanics / Spin-Glass

**A. Assumptions.**

The stated version assumes SwiGLU/GELU MLPs incur a superlinear width sample-complexity penalty while attention+residual belongs to a better complexity class.

**B. Derivation.**

I do **not** think this derivation exists from the cited literature.

What the relevant Mei/Misiakiewicz/Montanari papers actually give is:

- mean-field / PDE and kernel limits for two-layer nets, not transformer block comparisons;
- architecture-aligned invariance can reduce sample complexity by a factor \(d^\alpha\), but in invariant random-feature / kernel settings, not self-attention vs SwiGLU inside an autoregressive LM.

So the honest implication is:

\[
\text{“inductive bias matters”} \neq \text{“MLP sample complexity is superlinear while attention is better.”}
\]

The leap from shallow mean-field theory to “MLPs are the wrong compute allocation inside Llama blocks” is unsupported.

**C. Failure / inversion regime.**

Already failed conceptually. It is too loose to predict a clean inversion boundary.

**D. Killer experiment.**

A width sweep at fixed depth, fixed sequence length, fixed budget, and with/without MLP. If “superlinear MLP sample complexity” were real, the excess-risk penalty of adding MLP should worsen rapidly with width. Right now this is conjecture, not derivation.

**Verdict:** reject as current first-principles route.

## 3. Candidate 3: Rate-Distortion / Compression / Zipf

**A. Assumptions.**

Assume natural language exposes a heavy-tailed catalogue of context classes \(c\), with \(p(c)\) near-Zipf. Assume the model must allocate finite “codebook mass” either into fixed weights or into runtime context retrieval.

**B. Derivation.**

A static MLP is a fixed codebook. Suppose it can resolve only the top \(M\) context classes cleanly. Then unresolved tail distortion is lower-bounded by tail mass:

\[
D_{\text{mlp}}(M)\gtrsim \sum_{r>M} p_r \,\delta_r.
\]

For Zipf \(p_r\propto r^{-1}\), if \(\delta_r\) is bounded below on unresolved contexts, then

\[
\sum_{r>M} p_r \approx \frac{H_K-H_M}{H_K}\approx \frac{\log(K/M)}{\log K}.
\]

So tail distortion falls only logarithmically unless \(M\) scales aggressively.

Attention uses a **dynamic codebook**:
\[
A_t=\sum_{j<t}\alpha_{tj}v_j.
\]
The codebook entries \(v_j\) are in the activations, not the weights. Its error depends on retrieval error, not on having memorized each rare context in parameters.

This is a good explanation of why “activation memory” can beat “weight memory” on heavy-tailed language. But it still does **not** close a theorem, because the map from “context class” to actual LM residual error is unmeasured.

**C. Failure / inversion regime.**

It should fail on distributions with weak context reuse, short contexts, synthetic IID text, or tasks whose error is not dominated by long-tail context classes.

**D. Killer experiment.**

Train on token-shuffled C4. If the minimal advantage persists after the heavy-tailed contextual catalogue is destroyed, this candidate is badly weakened.

**Verdict:** promising explanatory prior; not yet the derivation.

## 4. Candidate 4: Geometric / Spectral / \( \sqrt{\mathrm{eff\_rank}}\alpha \)

**A. Assumptions.**

Assume the trained-spectrum invariant is causal for capability and that MLPs perturb it away from the useful manifold.

**B. Derivation.**

The local record itself says not to do this. The spectral invariant is empirically strong, but the repo also records:

- the specific `3√2` value is probe-dependent;
- the shifted-power-law derivation failed;
- the aux-loss that pushed the spectrum toward the teacher did **not** improve capability.

So even if one writes
\[
\mathrm{eff\_rank}\cdot \alpha^2 \approx C,
\]
that is a diagnostic constraint on spectra, not a demonstrated mechanism for the MLP-free win.

**C. Failure / inversion regime.**

Already failed as a causal explanation in this repo’s own evidence.

**D. Killer experiment.**

Compare training trajectories of baseline vs minimal and ask whether “closer to invariant” predicts the winner. Even if yes, it remains correlational unless intervention on the invariant changes the architecture ordering. The repo’s current aux-loss nulls already hurt this route.

**Verdict:** reject as primary explanation of `g138 -> g151`.

## 5. Candidate 5: Better Idea — Prefix-Information Transport Principle

This is the one to commit to.

**A. Assumptions.**

Assume the dominant bottleneck in shallow-to-mid autoregressive LMs is **transporting useful prefix information into the current-token state**, not locally synthesizing extra tokenwise nonlinear features after that transport.

Assume width \(d\) is the channel size and residuals preserve transported information.

**B. Derivation.**

Let \(H_t^\ell\) be the hidden state at token \(t\), layer \(\ell\).

A token-local MLP sublayer is
\[
\widetilde H_t^{\ell+1}=H_t^\ell + g_\phi(H_t^\ell).
\]

This is a deterministic function of \(H_t^\ell\). Therefore, by data processing,
\[
I(\widetilde H_t^{\ell+1};X_{<t}) \le I(H_t^\ell;X_{<t}).
\]

So an MLP sublayer can **re-encode** prefix information already present at token \(t\), but it cannot create new prefix information at that token.

An attention sublayer is
\[
H_t^{\ell+1}=H_t^\ell+\sum_{j\le t}\alpha_{tj}(H^\ell)\,U H_j^\ell.
\]

Because it depends on \(H_{<t}^\ell\), it can strictly increase
\[
I(H_t^{\ell+1};X_{<t}).
\]

Now the per-token cross-entropy obeys
\[
\mathcal L_t \ge H(X_{t+1}\mid H_t^L).
\]

To lower \(\mathcal L_t\), the model must raise \(I(X_{t+1};H_t^L)\). When the task is context-dominated, that first requires raising \(I(H_t^L;X_{<t})\). Only attention does that. Hence:

- **attention** spends parameters on moving prefix information into the current-token state;
- **width** gives that state enough channel capacity to carry it;
- **residuals** keep previously transported information from being overwritten;
- **MLPs** only improve the local decoding of already transported information.

This gives a clean budget criterion. Define

\[
\eta_\ell := I(X_{t+1};X_{<t}\mid H_t^\ell)
\]

as the remaining transport gap after layer \(\ell\), and

\[
\delta_\ell^{\text{mlp}} := \inf_{f\in\mathcal G_{\text{local}}}
\Big(H(X_{t+1}\mid H_t^\ell)-H(X_{t+1}\mid f(H_t^\ell))\Big)
\]

as the best possible gain from a token-local nonlinear decoder.

Then the next unit of budget should go to attention iff

\[
\eta_\ell > \delta_\ell^{\text{mlp}}.
\]

That is the first-principles statement. The current empirical results imply that, in this budget regime, \(\eta_\ell\) is still larger than \(\delta_\ell^{\text{mlp}}\), so MLP parameters are a worse use of budget than more attention-transport budget.

This subsumes Candidate 1. Candidate 1 is the linearized Jacobian shadow of this information-transport statement.

What is still missing is a new primitive that estimates \(\eta_\ell\) and \(\delta_\ell^{\text{mlp}}\) directly.

**C. Failure / inversion regime.**

This theory predicts inversion when the transport bottleneck is gone:

- very short context or context-destroyed data, where \(I(X_{t+1};X_{<t}\mid X_t)\) is small;
- much deeper / larger models where attention has already saturated the prefix channel;
- tasks whose remaining error is token-local synthesis rather than retrieval.

This is the cleanest predicted failure boundary of any candidate.

**D. Killer experiment.**

Destroy ordered prefix information while preserving token marginals. The minimal win should collapse. If it does not, this theory is wrong.

## Ranking

Scores are `1–10`, higher is better. “Big-lab resistant” means: hard for a scaling lab to publish without undercutting its own product story.

| Candidate | Elegance | Testability | Big-lab resistant | Verdict |
|---|---:|---:|---:|---|
| 1. Rank / operator-energy | 7 | 8 | 5 | Good shadow theory, not final |
| 2. Stat-mech / spin-glass | 3 | 4 | 2 | Reject |
| 3. Rate-distortion / Zipf | 6 | 7 | 7 | Keep as secondary route |
| 4. Spectral invariant | 4 | 8 | 3 | Reject as causal explanation |
| 5. Prefix-information transport | 9 | 9 | 8 | Commit |

## What I Would Actually Commit To

Commit to **Candidate 5**.

Not as “the theorem is solved,” but as the only route here that is both mechanistically right-shaped and brutally falsifiable.

The core claim should be:

> In an autoregressive LM under a tight budget, the scarce resource is not token-local nonlinearity. It is prefix-information transport into the current-token channel. Attention, width, and residuals directly buy that transport. The MLP does not. Therefore MLP parameters are waste until the transport gap has been mostly closed.

That is the paradigm-shift direction. It says *why* attention-only can beat a larger attention+MLP baseline, not merely that it does.

## Pre-Registered 200M Test

**Experiment name:** `genome_155_prefix_information_destruction_200m`

**Hypothesis.** The `g147/g151` minimal win is caused by superior budget allocation to prefix-information transport. Therefore destroying ordered prefix information should eliminate the win.

**System.**

- Same `200M` family as `g147/g151`.
- Baseline: `14L+MLP`, `d=1024`, `ffn=2304`, `4000` steps.
- Minimal: `7L noMLP`, `d=1024`, `8000` steps.
- Use the arm-specific best LRs from `g151`: baseline `2e-4`, minimal `3e-4`.
- Warmup `200` steps.
- `SEQ_LEN=256`, `N_TRAIN=32768`, `BATCH=8`, seeds `{42,7,13}`.

**Stimulus conditions.**

- `Natural C4`: standard `c4_clean_v1`.
- `Token-shuffled C4`: tokenize each sequence once, then apply one fixed random permutation per sequence to destroy order while preserving token multiset and length.

**Primary metric.**

\[
\Delta_{\text{nat}} = \text{top1}_{\text{min,nat}}-\text{top1}_{\text{base,nat}},\qquad
\Delta_{\text{shuf}} = \text{top1}_{\text{min,shuf}}-\text{top1}_{\text{base,shuf}}.
\]

**Predictions.**

- On natural C4: replicate the positive gap, `Δ_nat >= +0.5pp`.
- On shuffled C4: gap collapses, `Δ_shuf <= +0.1pp`, and likely reverses.
- Support statistic:
\[
C := \Delta_{\text{nat}}-\Delta_{\text{shuf}}.
\]
Require `C >= 0.4pp`.

**Kill condition.**

If `Δ_shuf` stays within `0.2pp` of `Δ_nat`, the transport theory is badly damaged. Then the win is not coming from ordered-context transport.

**Why this is the right test.**

It directly attacks the only unique thing attention buys: ordered prefix information. It does not hide behind spectra, probes, or post-hoc fits.

**Compute envelope.**

- VRAM: same as `g147/g151`, about `11 GB` peak. Safe.
- RAM: tokenized pool plus shuffled copy, well below `56 GB`.
- Wall-clock: `2 conditions x 2 arms x 3 seeds`; comfortably under `2 hours` on the current machine.
- Disk: trivial.
- Checkpointing: not necessary.

## Bottom Line

None of Candidates `1–4` currently gives a sound derivation. Candidate `5` is the only one I would defend in front of an audit.

It is still one step short of theorem-grade. The missing primitive is a direct estimate of:

- layerwise prefix-transport gain \( \eta_\ell \),
- layerwise local-decode gain \( \delta_\ell^{\text{mlp}} \).

Until that primitive exists, the honest claim is:

> We have a derivation skeleton, not a finished derivation.

That is still enough to choose the next experiment correctly.

## External sources

- Mei, Misiakiewicz, Montanari, “Mean-field theory of two-layers neural networks: dimension-free bounds and kernel limit” (COLT 2019): https://proceedings.mlr.press/v99/mei19a.html
- Mei, Misiakiewicz, Montanari, “Learning with invariances in random features and kernel models” (COLT 2021): https://proceedings.mlr.press/v134/mei21a.html
- Mei, Montanari, Nguyen, “A mean field view of the landscape of two-layer neural networks” (PNAS 2018): https://pmc.ncbi.nlm.nih.gov/articles/PMC6099898/
- Shannon, “Prediction and Entropy of Printed English” (1951): https://www.nokia.com/bell-labs/publications-and-media/publications/prediction-and-entropy-of-printed-english/
- Piantadosi, “Zipf’s word frequency law in natural language” (2014 review): https://pmc.ncbi.nlm.nih.gov/articles/PMC4176592/