**Kill Verdict**

I would not let the current project claim “Neural Genome transfer” or “universal training-health geometry.” The defensible claim is much narrower: **Qwen-family trained embed/lm_head interface priors help Qwen-family training under a continuous anchor.** Current §0.1 score: **3.8/10**.

I verified the raw JSONs.

**What Dies**

- `g181a`: full anchor `+0.999` nats, embed/lm_head-only `+0.483`, no-embed/lm_head `-0.439`. The no-embed vs embed gap is `-0.923` nats with CI `[-1.055, -0.835]`.
- `g181b`: embed/lm_head-only persists to 5000 steps: `+0.513` nats, CI `[+0.486, +0.531]`.
- `g183`: **incomplete**. Only scratch and trained_anchor are present. Scratch mean `6.4559`, trained_anchor `6.0664`, gap `+0.3895`. No PPMI/SVD result yet, so it cannot support the corpus-derived-init rescue.
- `g180b`: cross-tokenizer forecast fails. Geometry worsens MSE by `-39.4%`; BERT `-42.9%`, T5 `-96.4%`, GPT-2 `+44.0%`.
- `g182`: raw JSON top-level verdict says `WEAK PASS`, but the checks are failures. LOAO R² is catastrophic, roughly `-11` to `-19`. The only surviving signal is pairwise within-architecture delta: `R²=0.518`, `corr=0.720`, `n=24`, tiny label std `0.005`.
- `g186`: kills continuous Forecast/Diagnostic. Geometry pooled R² `0.022`; `arm_mean` R² `0.936`; `alpha_quad` R² `0.774`; per-arch R² Qwen3 `-0.098`, GPT-2 `-8.73`; conditioned permutation `p=1.0`.

**Answers To The Attacks**

- Is `+0.39` to `+0.51` meaningful? Yes as an optimization effect, no as a Neural Genome claim. It is not literal step-0 warm-starting in g181a/g181b, because initial NLL is identical across scratch and anchor arms. But it is functionally a **continuous trained-codebook prior**. That is much less interesting than transfer of internal structure.

- Could block-anchor harm be gradient interference? Yes, and that is the best explanation. The no-embed arm anchors ~440M non-interface params into a random recipient without the matching lexical interface. Equalizing initial total anchor-gradient L2 does not equalize layerwise curvature, optimizer burden, or representational compatibility. Also full_anchor beats embed-only, so “blocks harm” is not generally true. The true claim is “blocks without matching interface harm.”

- Is cross-architecture failure surprising? Not really. g173/g180b/g182/g186 mostly confirm the obvious: different tokenizers, decoders, and block families do not share a plug-compatible coordinate system. The surprising result would have been robust cross-family success.

- Is the diagnostic pivot novel? Weakly at best. HELM already normalizes broad evaluation infrastructure, Pythia exists specifically for studying training dynamics across many checkpoints, RepE centers population representations for monitoring/control, and 2026 work like Geometric Canary directly uses representational stability for steerability/drift diagnostics. Sources: [HELM](https://github.com/stanford-crfm/helm), [Pythia](https://proceedings.mlr.press/v202/biderman23a.html), [Representation Engineering](https://arxiv.org/abs/2310.01405), [Geometric Canary](https://arxiv.org/abs/2604.17698). The repo’s differentiator is falsification discipline, not current diagnostic performance.

**What Would Move It To 7+/10**

1. Finish `g183` and require PPMI/SVD to recover at least 50% of trained_anchor, 3/3 seeds, while beating frequency/random/spectrum/covariance controls by `>=0.10` nats.
2. Run an exact decomposition: init-only vs anchor-only vs init+anchor, trained vs PPMI vs random/frequency/spectral, same lambda, same norm, same compute.
3. Show cross-tokenizer interface transfer with a corpus/tokenizer bridge, not Qwen-reference features. Target: recover `>=20-30%` of the Qwen anchor effect on held-out tokenizers/architectures.
4. For diagnostics, stop scoring predictors and score decisions. A prospective policy must beat early loss, trajectory loss, gradient stats, Shesha, arm_mean, and alpha priors on regret per FLOP.
5. The real 7+ path is still electricity-grade: same task quality at materially lower kWh or compute under a locked wall-meter protocol.

Bottom line: **the project has a real interface-prior phenomenon, but the broad Neural Genome and forecast claims are currently overclaimed.** The honest next headline is “trained lexical/codebook priors may be replaceable by corpus-derived priors,” not “geometry predicts training health.”

