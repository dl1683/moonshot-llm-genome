# Grafting — Geometry-First Initialization for Efficient Training

## The Goal

Train neural networks dramatically faster by initializing them with geometric structure borrowed from pretrained models — bypassing the attractor-finding phase of gradient descent entirely.

Target: **orders-of-magnitude reduction in training steps to convergence**, not incremental improvement.

---

## The Core Idea

Every trained text neural network converges to the same geometric attractor, regardless of architecture:

- **Spectral shape**: `sqrt(eff_rank) * alpha ≈ 4.27` (CV 5.65% across 5 architectures, 13.5σ from random-init)
- **Direction identity**: top-30 activation-space eigenvectors overlap 0.65 across architectures (17× random baseline)
- **The attractor is a dynamical fixed point**: gradient descent moves every randomly-initialized network toward it, always, regardless of starting point

Standard training wastes most of its budget *finding* this attractor from a random starting point. The attractor is known in advance.

**Grafting** initializes a new model's weights so that its activations *already* sit in the attractor basin — correct spectral shape, correct top-30 directions, borrowed analytically from any pretrained reference model on similar data.

Gradient descent then only needs to learn *content* (what lives inside the correct geometric framework), not *geometry* (what the framework is).

---

## Why This Is Different From What Has Failed

15 null-ops catalog (in parent project) established: pushing spectrum shape toward the attractor with an *aux loss during training* does not install capability. Geometry matching fills directions with optimization noise.

Grafting is fundamentally different:
- **Null-ops**: aux-loss *during* training, gradient descent starts from random init and is nudged geometrically — fails
- **Grafting**: pre-initialization *before* training, gradient descent starts *inside* the basin — untested, theoretically motivated

The 2000-step layerwise feature matching experiment (genome_087) proved capability IS recoverable when a teacher provides dense geometric supervision. Grafting asks: can we compute that initialization analytically rather than iteratively?

---

## The Inverse Problem

**Given**: target top-30 left singular vectors U_target (extracted from any trained reference model on C4/natural text)
**Given**: target spectral shape (broken power-law, specific alpha and eff_rank)
**Find**: initial weight matrices W for a new model such that its activations on natural text already satisfy:
- SVD top-30 directions ≈ U_target
- `sqrt(eff_rank) * alpha ≈ 4.27`

If this inverse problem is tractable — even approximately — grafting gives a shortcut past the attractor-finding phase of training.

---

## Success Criteria

A grafting experiment succeeds if a grafted-initialized model reaches the same perplexity as a randomly-initialized model in **significantly fewer gradient steps**.

Minimum meaningful result: **10× fewer steps**.
Moonshot target: **10,000× fewer steps** (i.e., grafted model needs only ~14 gradient steps where baseline needs 143,000).

---

## Relation to Parent Project (Neural Genome)

Grafting uses the Neural Genome's measurement infrastructure (extractor, spectral invariant, direction identity probe) as its foundation. It is the first concrete *application* of the atlas:

- **Atlas**: measures and maps the geometric structure of trained models
- **Grafting**: uses that map to navigate a new model to the trained region directly

Results feed back: any grafting success becomes evidence that the atlas coordinates are causally relevant, not just correlational — which is the hardest open question in the parent project.

---

## Scope

- **In scope**: initialization strategies, inverse problem solutions, convergence acceleration, cross-architecture geometry transfer
- **Out of scope**: biology, weight-space editing after convergence, standard knowledge distillation (probability matching), anything that doesn't directly accelerate training

---

## Files in This Directory

| File | Purpose |
|---|---|
| `OBJECTIVE.md` | This file — the goal, the mechanism, the success criteria |
| `code/` | Experiment scripts (`grafting_*.py`) |
| `results/` | Experiment results and figures |
| `research/` | Theoretical derivations, preregistrations |
