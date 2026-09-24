# Lab Notebook

Append-only. Newest entries at the top. Format per experiment:

```
## E### — name (date)
WHAT WE DID / WHAT WE SAW / WHAT'S NEXT
```

---

## V001 — token journey visualization (2026-09-24) — DONE

WHAT WE DID: first artifact of the standing VISUALIZER thread: one forward
pass per prompt, three panels — logit lens through depth (final LN+head
applied to the last token's stream after emb and each block), PCA-2D token
trajectory with write arrows, per-layer angular authority (1−cos and
write/stream). runs/v001/token_journey.png.

WHAT WE SAW:
- **New observable: decision depth.** "…torches to burn " is decided at L3
  (top-1 `t`, p 0.69) and L5 HALVES its confidence (0.69→0.33); "To be, or
  not to " only surfaces the correct `b` at L4. Predictions form at
  different depths per token — T004 written with hypotheses D1-D3 and three
  registered predictions.
- Authority panel makes T003 visible: L0 write/stream ≈ 8.5 vs ≈ 1 later;
  angular displacement 0.7 (L0) vs 0.2-0.3 (later).

WHAT'S NEXT: e012 redesigned → measure decision depth over ~2000 val
positions; correlate with next-token entropy and L4/L5 ablation damage
(T004 discriminators). Viz polish backlog: arrowheads on trajectory, L0 bar
headroom in panel 3 (v001.1).

## E011b — L0 redundancy sweep + orthogonal-innovation control (2026-09-24) — DONE

WHAT WE DID: eval-only discriminators from the critique harvest: (1) all 63
L0 head-subset lesions; (2) same-norm random replacements of every attention/
MLP write (orthogonal-innovation control); (3) residual-stream norm profile at
block inputs. runs/e011b/redundancy_ortho.png.

WHAT WE SAW:
- **L0 heads are a cooperative ensemble with graceful degradation:** singles
  mean +0.05 nats (one slightly negative), yet all-6 = +2.40. Sum of singles
  0.317 vs joint 2.403 = 7.6× superadditivity. Keeping 1 of 6 heads still
  leaves 94% of full-ablation damage. Hook implementation validated exactly
  (all-6 head-zero 2.403 ≈ block-zero 2.401).
- **Same-norm random writes hurt more than zeroing** everywhere — attn L0
  +3.66 vs +2.40; MLP L1 0.82 vs 0.15 (5.4×). Ratios exceed the √2
  perturbation-scale prediction for MLPs and late attention → downstream is
  calibrated to write direction, not just magnitude.
- **Stream norm:** 0.67 at block 0 → 8.4× jump → plateau ~5.5. Write/stream
  ratio falls ~12× L0→L5.
- Emerging mechanism (see THINKING T003): the residual stream's norm growth
  may SCHEDULE each block's angular authority — front-loaded lesion maps
  could be partly architecture, not learning.

WHAT'S NEXT: e011c matched-perturbation control (60°-rotated writes) settles
geometry-vs-content per component; e014b stream-renorm training tests the
authority-schedule hypothesis directly. e003b (corrected ascent instruments)
still queued.

## E003 — forgetting selectivity frontier (2026-09-24) — DONE

WHAT WE DID: T002 discriminator suite: (1) gradient cosines (A↔B vs within-half
vs A↔French), (2) LR ascent sweep 1e-6…3e-5 with (ΔA,ΔB) trajectories, (3)
implant-French (400 steps, 5e-4) then ascend-on-French arm, (4) fluency-vs-
content CE probes (A-unique/B-unique lines vs generic) after mild ascent.
runs/e003/selectivity_frontier.png.

WHAT WE SAW (all three registered predictions resolved):
- **P1 REFUTED:** cos(A,B) = 0.345 ≈ within-half baselines (0.363/0.390);
  French is 2.4× lower (0.144). Same-corpus halves are NOT gradient-parallel.
- **P2 CONFIRMED:** no LR reaches ΔA ≥ 1 with ΔB ≤ 0.1. Even lr 1e-6 (the
  gentle walk along grad_A itself) gives ΔA +0.29 / ΔB +0.27 at step 200 —
  perfectly anti-selective.
- **P3 REFUTED:** unlearning French also destroyed Shakespeare
  (dissimilar_selective = false). Content distance does not rescue ascent.
- Fluency probes: after "mild" ascent (1e-5×200) everything ≈ 24 nats
  (random): A-unique 1.26→24.2, B-unique 1.35→24.0, generic 1.63→23.7.
- **Unified story forming:** first-order ascent — any dose, any content —
  destroys the shared fluency substrate first and content memories only with
  it. Gradient geometry (P1) shows content IS distinguishable in gradient
  space; the failure is that ascent trajectories do not follow it.

WHAT'S NEXT: T002 final resolution (thinking, not running): read the early
trajectory steps — was there ANY transient selectivity window (French rising
faster in the first 25 steps) before collapse? If yes: early-stopped ascent +
fluency-anchored objective is the repair hypothesis. If no: first-order
methods are structurally dead here; next family = weight-targeted surgery
(ascend only low-overlap weights) or second-order directions. Also reconcile
with the independent critique (scratch/critique_T001_T002.md) when it lands.

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
