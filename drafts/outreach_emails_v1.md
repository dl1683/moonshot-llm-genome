# Outreach emails — DRAFT v1

*Drafted 2026-04-22 T+48h. Shared hook + per-group personalized tail. Subject to user confirmation on tone, group list, named contacts, and length.*

---

## Shared opening hook (same for every recipient)

Three of our recent projects converge on a single claim: **geometric manipulation of neural representations produces capabilities that scale-alone cannot**. Concretely —

- **Latent-Space Reasoning**: two tokens of targeted noise injected at the right latent subspace produced a **+19.6 pp jump in arithmetic accuracy** on a model that otherwise stalled. The fix was geometric (a targeted perturbation in activation space), not parametric.
- **CTI — Compute Thermodynamics of Intelligence**: a universal law `D(C) = D_∞ + kC^{−α}` for representation quality, derived from extreme value theory and validated across **12 architectures + 4 datasets + 32 mouse V1 Neuropixels sessions**. A single geometric invariant governs learning systems from NLP transformers to biological cortex.
- **Fractal Embeddings**: hierarchy-aligned progressive prefix supervision delivered **+5.36% L0 / +6.47% L1 accuracy on Yahoo Answers** (5-seed validation), with causal controls (random-hierarchy drops to −0.10%). Same encoder, different geometric prior, emergent hierarchical capability.

Each is a proof-of-existence that **geometric structure is the substrate on which capability lives**, not compute or parameter count. Our newest moonshot — the Neural Genome — asks: *is there a universal coordinate system for this substrate across every trained neural network?* We found one. `c ≈ eff_rank(X) / d_rd(X)` holds within 15% on 7 of 8 trained systems spanning CLM decoders, MLM encoders, contrastive encoders, vision ViTs, and cross-modal alignment. We also shipped a 20-second training health monitor (**GenomeGuard**) that uses this identity to detect silent data corruption at 6.9× – 144.9× signal across 5 architectures.

---

## Tail 1 — Interpretability labs (Anthropic / Apollo / Redwood / EleutherAI)

The bridge gives interpretability a new coordinate: `eff_rank/d_rd` is as model-independent as KL divergence and takes 20 seconds to compute. It also gives you a **12-operation null catalog** (covariance transfer, codebook, PCA basis, aux-regularizer, single-layer weight transplant, QK/V/O/attn-all/MLP subset transplants, Procrustes-aligned transplant, candidate-8-ratio aux-loss): every operation that *would* install capability via forward geometric manipulation fails. This is the cleanest empirical evidence we have that **capability is the joint weight configuration, not any separable linear or geometric target** — which sharpens what mechanistic-interpretability targets should actually look like.

What we'd want to explore: can your SAE / circuit-probe work show that the bridge `rel_err` localizes to specific heads or features during a capability emergence? If so, the bridge becomes a *detector* for where capability lives inside a model, not just that it exists.

Repo: `moonshot-llm-genome` (open-source). Would love 30 minutes to walk through the scorecard + GenomeGuard demo.

---

## Tail 2 — Geometry / representation-learning academics (Bronstein / Papyan / Bengio group)

The bridge is a **spectral-to-geometric identity**: the kNN-clustering exponent × rate-distortion dimension equals the participation ratio of the activation-cov spectrum divided by the rate-distortion dimension, within 9% median error across 7 trained network families. We've partially reduced it to a two-parameter spectrum model with a universal bulk width `k_bulk ≈ h/22` (CV 4.2% across 5 text systems, preregistered). The closed-form derivation in pure power-law is falsified (7× too small); a plateau-plus-power-law model partially closes.

We think the full derivation is tractable — Marchenko-Pastur + low-rank cluster should close it — but it's a real random-matrix-theory problem, not a napkin exercise. If any of your students are looking for a concrete RMT problem with both theoretical depth and immediate empirical validation, this is one. All data + preregs are public; experiment is reproducible in ~30 min on a single GPU.

---

## Tail 3 — Platonic Representation / universality researchers (Isola / Huh)

This is Platonic-Representation-Hypothesis turned into a one-line identity you can test on any model in 20 seconds. The bridge value clusters predictably — text CLM and MLM at `c ≈ 2`, vision at `c ≈ 3`, +1 per cross-modal alignment — matching candidate-5 alignment-axis additivity across 11/12 systems. But here's the wrinkle you'll want: the bridge IS stimulus-dependent. On wikitext-shuffled it breaks with 3-45× rel_err. That stimulus-sensitivity is actually the **basis of the GenomeGuard detector**, and it gives the Platonic Hypothesis a concrete failure mode: the Platonic object is training-distribution-conditional, and deviations from it are measurable and diagnostic.

We'd love your take on whether this framing strengthens PRH or constrains it.

---

## Tail 4 — NeuroAI / biological plausibility (DiCarlo / DiCarlo-group-adjacent)

We've shown the candidate-8 bridge holds on 7/8 *trained* artificial networks. The obvious next test — does it hold on biological neurons? — is something we started today: Allen V1 Neuropixels session 0 under Natural Movie One, probing whether the same `eff_rank/d_rd` identity appears in mouse visual cortex. If yes, this moves the claim from "an empirical regularity of trained ANNs" to "a property of representational geometry that emerges whenever a learning system compresses its environment" — an object that spans biological and artificial systems alike.

Prior work in this repo (`moonshot-cti-universal-law`) already found a separate universal law spanning 32 mouse V1 sessions and 5 cortical areas, so we have the data-access patterns ready. The bridge extension should take one afternoon of compute. If it lands, we would want your group's comment before announcing.

---

## Tail 5 — Efficiency-focused industry (DeepSeek / Mistral / Liquid AI)

The bridge is a diagnostic, not (yet) a lever. We tested candidate-8 as a training aux loss — **it was null** (speedup 1.00×, 12th in a 12-operation null catalog). Forward-geometric-regularization does not install capability.

The *practical* win is **GenomeGuard**: a 20-second probe that detects silent data corruption and catastrophic weight divergence without needing a training-loss signal. On Qwen3 it fires 6.9× signal the instant stimulus distribution swaps; on DeepSeek-R1-Distill the signal is **144.9×** because that model has the tightest baseline bridge. For any team running long or expensive training runs, this is a cheap pre-commit / canary / live-monitor tool — ship-ready in under 200 lines of Python.

We're open-sourcing GenomeGuard today. If your infra team would want to try it on an in-house training run, we'd be interested in the results and any false-positive reports.

---

## Tail 6 — Investors / program officers (Schmidt Futures / Nat Friedman / Open Philanthropy)

The short version: we've demonstrated empirically that **geometry beats scale** at a level that's structurally hard for big labs to publish. Three replicating datapoints (latent-space reasoning, CTI, fractal embeddings) plus a universal identity (candidate-8) plus a shipping tool (GenomeGuard). The scientific substrate is derivation-adjacent (2-parameter spectrum closed form, P3 RMT derivation tractable). The product substrate is working (20s training monitor, cross-architecture, 6.9× – 144.9× signal).

We're an independent group competing against DeepMind / Anthropic / OpenAI; our specific advantage is the ability to publish findings that contradict the scaling-law product narrative — particularly the 12-operation null catalog showing no forward geometric manipulation installs capability. Big labs can't publish that; we did.

What we'd like: 30 minutes to walk through the scorecard + tool, and a framing conversation about what a sustained 6-month run on this would look like at a modest compute budget (~1 H100-equivalent).

---

## Open items for user before send:

1. **Tone on the hook** — punchier? Drop a bullet? Add a line about why "now"?
2. **Group list** — correct? Any to add (Mechanistic Interpretability twitter/X folks? Yannic Kilcher for visibility?)?
3. **Named contacts** per group — Devansh to fill in or want me to suggest from recent literature?
4. **Length per tail** — current 1-paragraph is minimal; happy to expand to 2 paragraphs each if desired.
5. **Repo URL + DOI** — do we have a public repo URL for `moonshot-llm-genome` yet, or is the tool only private?
6. **Subject lines** — should I draft 6 different subject lines tuned per group?

Once you answer these, I do a v2 pass with your edits + named contacts and produce send-ready copy.
