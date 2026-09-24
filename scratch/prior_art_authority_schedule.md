# Prior-art check: T003-B "authority schedule" hypothesis (e014b)

Date: 2026-09-24. Scope: (1) residual-stream norm growth across depth, (2) the
stream-growth -> early-layer-lesion-criticality link, (3) training schemes that
normalize/rescale the residual stream, (4) verdict on the planned causal test
(retrain with constant-norm stream at block inputs; check whether the lesion
map flattens).

---

## Stream-norm growth

Well documented. The observation is established prior art; our E011b numbers
([0.67, 5.66, 6.14, 5.12, 5.20, 5.62]) are a 6-layer, char-level instance of
a known phenomenon.

1. **TurnTrout (Q. Pope), "Residual stream norms grow exponentially over the
   forward pass" (LessWrong / Alignment Forum, May 2023).**
   https://turntrout.com/residual-stream-norms-grow-exponentially-over-the-forward-pass
   Canonical finding: residual-stream L2 norm grows ~exponentially with depth
   across many models/prompts (GPT-2-XL: ~1.045x per layer, layers ~5-41; small
   models superexponential; GPT-Neo/Pythia *decrease* in late layers via
   canceling writes). Favored explanation: blocks only see LN(x) — you cannot
   cleanly write -v without knowing the LN scale — so layers *overshadow*
   rather than delete, each writing ~4.5% larger. Notes explicitly that LN
   makes the network's function invariant to the stream norm (only direction
   matters), and (Appendix 1) that per-layer writes must exceed (g-1) x stream
   norm to sustain growth rate g. **Crucially for us: no causal interventions
   performed or proposed** — purely observational (weight-norm analysis
   supports growth being baked into W_OV / MLP weights).
2. **Admin — Liu et al., "Understanding the Difficulty of Training
   Transformers" (ACL 2020), arXiv:2004.08249.**
   https://arxiv.org/abs/2004.08249
   Shows pre-LN hidden-state norms grow across depth and at initialization the
   network output is dominated by the residual contributions of the last
   layers; proposes adaptive initialization to make effective layer influence
   uniform. This is the earliest "norm growth is an architectural property
   that skews per-layer influence" argument we found — but the skew they
   target is at INIT and about training stability, not trained-model lesion
   maps.
3. **Xiong et al., "On Layer Normalization in the Transformer Architecture"
   (ICML 2020), arXiv:2002.04745.**
   https://arxiv.org/abs/2002.04745
   Pre-LN hidden-state norm grows gradually with depth at init (post-LN does
   not); motivates why pre-LN trains without warmup. The pre/post divergence in
   stream-norm behavior is standard textbook material now.
4. **Peri-LN — Kim et al., "Peri-LN: Revisiting Normalization Layer in the
   Transformer" (ICML 2025), arXiv:2502.05730.**
   https://arxiv.org/abs/2502.05730
   Analyzes pre-LN "unbounded variance growth" across depth and hidden-state
   redundancy; adds a normalization on module outputs (a hybrid between pre-
   and post-LN). Training-stability focus; no lesion/criticality readout.
5. **Adjacent interp context:** nostalgebraist's "logit lens" (2020,
   https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6rs/interpreting-gpt-the-logit-lens)
   — intermediate residual states must be final-LN'd before unembedding, i.e.
   the community already treats the stream's magnitude as uninformative and
   its direction as the content; Anthropic's "A Mathematical Framework for
   Transformer Circuits" (2021, https://transformer-circuits.pub/2021/framework/index.html)
   frames the residual stream as a bandwidth-limited shared communication
   channel; "Privileged Bases in the Transformer Residual Stream" (2023,
   https://transformer-circuits.pub/2023/privileged-bases/index.html) discusses
   LN as the main basis-dependent op.

**Gap confirmed:** everyone documents the growth and hand-waves about
"overshadowing"; nobody we found quantifies the per-layer **write/stream
ratio** as a bound on *angular* influence, and nobody connects it to ablation
damage. Note also a regime difference worth stating in the paper: TurnTrout's
sustained-exponential models keep ‖w‖/‖x‖ roughly *constant* with depth
(writes scale up with the stream), whereas our net jumps 8.4x at block 0 then
**plateaus** — in the plateau regime relative authority necessarily *falls*
with depth. Our "authority schedule" claim is specific to jump-then-plateau
streams, which is a less-studied profile.

## Depth-importance observations

Front-loaded lesion maps and "later layers are dispensable" are extensively
observed — but always explained functionally (early layers do lexical/syntactic
aggregation) or via representation similarity, never via stream-norm geometry.

1. **Gromov et al., "The Unreasonable Ineffectiveness of the Deeper Layers"
   (2024), arXiv:2403.17887.**
   https://arxiv.org/abs/2403.17887
   Deleting up to ~half the *deeper* layers of Llama-2/Mistral causes minimal
   degradation (after QLoRA "healing"); layer criticality estimated by angular
   similarity between block inputs/outputs. Closest LLM-scale analogue of our
   L0-heavy lesion map. They also try "frozen deep layers during training" —
   a *training-time* experiment on the same axis, but with freeze (not stream
   renorm) and without isolating the norm-geometry mechanism.
2. **ShortGPT — Men et al. (2024), arXiv:2403.03853.**
   https://arxiv.org/abs/2403.03853
   "Block Influence" score; many LLM layers have negligible roles. Observational.
3. **Nepal et al., "Layer Importance for Mathematical Reasoning is Forged in
   Pre-Training and Invariant after Post-Training" (2025).**
   https://arxiv.org/abs/2508.02023 (layer-wise ablation showing a few critical
   layers whose importance survives post-training — relevant to our
   learning-vs-architecture question, but they never test architecture either).
4. **BERT-era layer pruning:** "To Filter Prune, or to Layer Prune, That Is the
   Question" (ECCV 2020, https://openaccess.thecvf.com/content_ECCV_2020/papers/Ameya_Prabhu_ECCV_2020_paper.pdf);
   Rasa's BERT pruning writeup (2019) noting last layers pruned first;
   Michel et al., "Are Sixteen Heads Really Better than One?" (NeurIPS 2019,
   https://arxiv.org/abs/1905.10650). Consensus profile: early layers hardest
   to prune, late layers redundant.
5. **LLRD (layer-wise LR decay)** — e.g., Sun et al. 2019 and
   https://arxiv.org/abs/2212.06138 — discounts LR for lower layers because
   "early layers learn general features": a *learned-importance* framing that
   our hypothesis directly competes with; notably NVIDIA's
   "Pretraining BERT with Layer-wise Adaptive Learning Rates" found decay made
   little difference in some settings.
6. **Gradient/entropy asymmetries with depth:** NormFormer (Shleifer et al.
   2021, https://arxiv.org/abs/2110.09456) — in pre-LN, early-layer gradients
   are larger and early attention entropy too high; σReparam (Zhai et al.
   ICML 2023, https://arxiv.org/abs/2303.06296) — attention entropy collapses
   (sharpens) with training/depth. Related facts, different mechanism.
7. **Ziming Liu, "Depth 1 — Understanding Pre-LN and Post-LN" (blog, Jan
   2026), https://kindxiaoming.github.io** — argues pre-norm induces
   representation collapse (later layers contribute little) while post-norm
   avoids it but vanishes gradients. Conceptually the nearest neighbor to
   T003-B's architecture-vs-learning split; no norm-ratio quantification, no
   renorm-training lesion experiment.

**Gap confirmed:** the specific claim "stream norm growth geometrically caps
the angular influence of later blocks, so front-loaded lesion maps are partly
architectural" was not found anywhere as a stated, tested mechanism. All
prior front-loading explanations are functional/similarity-based or
initialization-time.

## Residual-normalization training schemes

Many schemes rescale *branches* (per-block scalars) or normalize *hidden
states* — i.e., the architecture family e014b will build exists — but every
one was evaluated on convergence speed/stability/final loss, **never on
depth-wise ablation/lesion profiles**.

1. **nGPT — Loshchilov et al., "Normalized Transformer with Representation
   Learning on the Hypersphere" (NVIDIA, 2024), arXiv:2410.01131.**
   https://arxiv.org/abs/2410.01131
   **The closest prior scheme:** ALL hidden states are unit-normalized every
   block (stream lives on a hypersphere); updates are "normalized coordinate
   descent" with learned per-layer coefficients. Reports 4-20x faster
   convergence. This literally trains with a constant-norm residual stream —
   but the paper analyzes optimization, not component criticality. e014b's
   question (does flattening the stream flatten the LESION MAP?) is unasked.
2. **Post-LN itself** (original Transformer, and every pre-vs-post study):
   x_{l+1} = LN(x_l + F(x_l)) renormalizes the stream after every add, i.e.
   constant-norm block inputs. e014b's hook is architecturally a post-LN-like
   variant. Comparisons exist (Xiong 2020; Admin 2020; Peri-LN 2025;
   HybridNorm, NeurIPS, https://arxiv.org/abs/2505.14183) — all metrics are
   gradients/loss/stability, none measure layer-criticality profiles.
3. **Branch-scaling family** (normalize/scale the *write*, not the stream):
   ReZero (Bachlechner et al. 2020, https://arxiv.org/abs/2003.04887) —
   x + alpha*F(x), alpha init 0; Fixup (Zhang et al. 2019,
   https://arxiv.org/abs/1901.09321); SkipInit (De & Smith, NeurIPS 2020,
   https://arxiv.org/abs/2001.07261 — shows BN's benefit is damping residual
   branches; a learned scalar suffices); LayerScale/CaiT (Touvron et al. 2021,
   https://arxiv.org/abs/2103.17239 — per-channel learnable write scales);
   DeepNorm/DeepNet (Wang et al. 2022, https://arxiv.org/abs/2203.00555 —
   depth-dependent residual-branch scaling to 1000 layers). Note: LayerScale
   models *learn* a per-layer write schedule — a trained analogue of our
   "authority schedule" knob, again never probed with lesion maps.
4. **Value Residual Learning — Zhou et al. (2024), arXiv:2410.17897**
   (https://arxiv.org/abs/2410.17897) — adds value-path residuals to fight
   attention concentration in deeper layers; adjacent phenomenon
   (attention sharpens with depth, cf. σReparam), different intervention.
5. **Admin** (again) — re-initialization rather than renormalization, but
   explicitly designed to equalize per-layer influence on the output; if e014b
   flattens the lesion map, Admin is the init-time sibling of the result.

## Verdict + closest prior work

**Verdict: PARTIALLY-KNOWN.**

- The **observation** (stream-norm growth; front-loaded lesion maps) is
  thoroughly documented prior art on both halves.
- The **mechanistic link** (LN scale-invariance => angular influence capped by
  ‖w‖/‖x‖ => front-loading is partly architecture) is *half*-anticipated:
  TurnTrout states the LN-invariance and overshadowing logic and proves writes
  must scale with the stream; Ziming Liu's blog and the Admin/Peri-LN line
  argue pre-LN's norm growth harms later-layer usefulness. But no source we
  found states or tests "ablation damage per layer is set by the write/stream
  ratio."
- The **causal experiment** (retrain with the stream renormalized to constant
  norm at block inputs, then compare depth-wise lesion maps against a
  matched pre-LN baseline) is **not done anywhere we can find**. nGPT already
  builds constant-norm-stream transformers (for speed) and post-LN is exactly
  that architecture — so the *machinery* exists, but nobody has ever pointed a
  lesion map at it. Prediction P2/P3 and the controlled comparison are novel.
- **Refutation risk from prior art is low but nonzero:** Gromov's frozen-layer
  training and Nepal's pre/post-training invariance hint importance profiles
  may be sticky (could survive renorm — which would support the *learning*
  reading, our registered falsifier). Also flag: constant-norm streams at
  block inputs make e014b behaviorally post-LN-like; expect the usual post-LN
  training-stability caveats (warmup; see Xiong 2020, Peri-LN 2025), and cite
  TurnTrout's counterexamples (Pythia/Neo nets whose stream norm *shrinks*),
  which bound how universal the "authority schedule" claim can be.

Closest six:

| # | Work | URL | What it covers |
|---|------|-----|----------------|
| 1 | TurnTrout, "Residual stream norms grow exponentially over the forward pass" (LW/AF 2023) | https://turntrout.com/residual-stream-norms-grow-exponentially-over-the-forward-pass | Stream-norm growth + LN overshadowing; scale-invariance; no intervention |
| 2 | Liu et al., Admin (ACL 2020) | https://arxiv.org/abs/2004.08249 | Pre-LN norm growth skews per-layer influence at init; init fix |
| 3 | Xiong et al. (ICML 2020) | https://arxiv.org/abs/2002.04745 | Pre-LN hidden-state norm growth; pre-vs-post analysis |
| 4 | Gromov et al., "Unreasonable Ineffectiveness of the Deeper Layers" (2024) | https://arxiv.org/abs/2403.17887 | Late-layer prunability + angular similarity + freeze-training |
| 5 | Loshchilov et al., nGPT (2024) | https://arxiv.org/abs/2410.01131 | Trains with unit-norm stream every block (for speed, no lesion map) |
| 6 | Kim et al., Peri-LN (ICML 2025) | https://arxiv.org/abs/2502.05730 | Pre-LN unbounded variance growth / hidden-state redundancy; hybrid norm |

(Honorable mentions: ShortGPT arXiv:2403.03853; NormFormer arXiv:2110.09456;
ReZero arXiv:2003.04887; SkipInit arXiv:2001.07261; LayerScale/CaiT
arXiv:2103.17239; DeepNet arXiv:2203.00555; σReparam arXiv:2303.06296; Nepal
et al. arXiv:2508.02023; Ziming Liu's pre/post-LN blog, kindxiaoming.github.io.)

Design notes for e014b taken from this check: (a) include a post-LN arm as a
second renormalized architecture so the result isn't read as "just post-LN";
(b) verify the renorm model actually reaches comparable loss (nGPT suggests it
may even train faster); (c) also log the learned write/stream ratio profile of
the renormed net — if writes re-inflate relative to the (now constant) stream,
that itself measures how strongly training wants an authority schedule.
