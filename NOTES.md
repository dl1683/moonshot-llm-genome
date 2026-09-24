# Lab Notebook

Append-only. Newest entries at the top. Format per experiment:

```
## E### — name (date)
WHAT WE DID / WHAT WE SAW / WHAT'S NEXT
```

---

## E003 — forgetting selectivity frontier (2026-09-24) — RUNNING

WHAT WE DID: T002 discriminator suite, launched detached: (1) gradient cosine
A↔B vs A↔French(Les Misérables), (2) fine LR ascent sweep trajectories
(ΔA,ΔB) at 1e-6…3e-5, (3) implant-French-then-unlearn arm (selectivity for
dissimilar content), (4) fluency-vs-content CE probes (A-unique/B-unique
lines). Results land in runs/e003/ — next heartbeat harvests and THINKING.md
gets the T002 resolution.

## E011a — write norms vs lesion damage (2026-09-24) — DONE

WHAT WE DID: measured mean residual write norms (per token) of every
attention/MLP block on val batches; compared to E001 lesion damage (T001
discriminator 1, zero training).

WHAT WE SAW:
- **H4 (write-norm confound) REFUTED.** Attention write norms are NOT
  monotone in depth ([2.72, 2.49, 3.31, 2.73, 2.39, 1.93] — layer 2 writes
  the most), yet damage still falls monotonically. Damage per unit write:
  attn [0.88, 0.70, 0.32, 0.14, 0.08, 0.02] — an 11× efficiency gradient.
  The front-loading is information architecture, not geometry.
- MLP write norms RISE with depth (1.82 → 5.64); MLP-5 writes the largest
  residual in the net yet ablation costs only +0.59 nats. Late MLPs write
  large, dispensable content — new open anomaly (for whom/what is it
  writing?). MLP-0 damage-per-write (0.95) is 5-10× any other MLP.
- Registered prediction "write norms will not decline monotonically" was
  CONFIRMED — first register-then-run success of the discipline.

WHAT'S NEXT: T001's remaining discriminators: mean-replace ablations and
LN-only recalibration (H2, off-manifold artifact) — e011 proper. And the new
anomaly: what does MLP-5 write? (logit-lens on its output direction space).

---

## 2026-09-24 — The pivot (context entry)

After three programs (neural genome transplants, LLM control surfaces,
HANDLE), we got too ambitious and drifted from the original spirit: da
Vinci-style dissection of neural networks for its own sake. Today the repo was
reset to a clean lab (tombstone commit `106aeff` preserves everything prior)
and the mission narrowed to the original one: small nets (1–10M params), many
experiments, curiosity first, graphs for everything, play.

The three questions opening the program: (1) what does a lesion map of a
freshly trained tiny transformer look like? (2) can we make a network forget
one part of its training data without wrecking the rest? (3) everything after
that is whatever the first two cuts turn up.

---

## E002 — forgetting pilot: naive vs anchored unlearning (2026-09-24) — DONE

WHAT WE DID: split the corpus positionally (A = first half of Shakespeare, B =
second half). From the E001 checkpoint, unlearned A with two arms: (1) naive
gradient ascent on A only; (2) anchored ascent (retain loss on B, weight 1.0).
AdamW lr 2e-5, 400 steps, batch 32×256. Tracked held-out CE on A and B;
generation probes before/after (runs/e002/probes.txt).

WHAT WE SAW:
- **Naive ascent is a bomb, not a scalpel.** ΔA +26.1, ΔB +25.9 nats — the
  entire model is destroyed (final CE ≈ random guessing over 65 chars).
- **Anti-selectivity:** to raise A by just +1 nat, B had ALREADY risen +1.51
  nats. Collateral damage runs ahead of the target damage.
- **The retain anchor only slows the destruction** (ΔA +6.4, ΔB +5.5 at equal
  steps; B damage at A+1 nat still +1.04). It does not create selectivity at
  this dose.
- Honesty reflex: the intervention changes behavior (massively) but with zero
  targeting value at this scale of step size.

WHAT'S NEXT: map the *selectivity frontier*: sweep ascent LR × steps × retain
weight; unlearn a small subset (one play) instead of half the corpus; try
Fisher/EWC-style parameter anchor instead of replay anchor; try weight-space
surgery (rank parameters by gradient overlap between A and B). → e003.

## E001 — first cut: lesion map (2026-09-24) — DONE

WHAT WE DID: trained a 2.74M-param char GPT (6 layers, 6 heads, 192 dim,
block 256) on Tiny Shakespeare (val loss 1.622 after ~2000 steps / 200 s; val
bottomed ~1.55 at step 1500 then overfit slightly). Then zeroed every
attention block, every MLP block, and each of the 36 heads individually;
measured deterministic val-loss delta per lesion (runs/e001/lesion_map.png).

WHAT WE SAW:
- **MLP-0 is the keystone organ:** zeroing it costs +4.08 nats — the single
  most damaging lesion, worse than any attention ablation.
- **Attention is strictly front-loaded:** L0 +2.40, L1 +1.74, L2 +1.06,
  L3 +0.38, L4 +0.19, **L5 +0.03 — the last attention layer is almost dead
  weight** in this model.
- **MLP damage grows with depth** after L0: L1 +0.15 → L4 +0.60, L5 +0.59.
- **16/48 components are ~dispensable** (Δ<0.02) — mostly late-layer heads.
- Net shape: early attention + layer-0 MLP carry the load; the top of the net
  is attention-light, MLP-heavy, and partly vestigial.

WHAT'S NEXT: what does MLP-0 actually store (char/unigram statistics? test by
probing / ablate-then-finetune recovery cost)? Why is attn-5 dispensable —
vestigial or quietly specialized (punctuation/newline)? Damage ≠ necessity:
how cheaply can the net re-learn around a lesion (zero-finetune recovery)?
Does this shape hold at 1M/10M/30M params (→ e004 ladder)?
