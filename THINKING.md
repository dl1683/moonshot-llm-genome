# Thinking Journal — the 80%

The lab's operating ratio is 80% thinking / 20% doing. Every result gets an
entry here BEFORE the next experiment that builds on it: what it could mean
(multiple hypotheses), how the hypotheses differ, which cheap observation
would discriminate them, and a registered prediction so we can't retrofit.
New experiments are gated on this file: if the latest result has no
interpretation entry, the next heartbeat thinks instead of runs.

---

## T001 — What does the E001 lesion map actually show? (2026-09-24)

**Observed:** attention damage strictly monotone with depth (L0 +2.40 → L5
+0.03 nats); MLP-0 keystone (+4.08); MLP damage rising with depth (+0.15 →
+0.60); 16/48 components near-dispensable; model slightly overfit.

**Hypothesis 1 — the da Vinci reading (early layers do the work):** early
attention performs the actual context aggregation for a char-level task;
late attention genuinely contributes little.

**H2 — off-manifold ablation artifact:** zeroing a block's residual write
pushes downstream LayerNorms off their calibrated statistics. Damage measures
*distribution shift*, not *information content*. A "harmless" block might be
quietly important while a loud lesion is just miscalibration.

**H3 — redundancy, not vestigiality:** late layers may duplicate each other.
Single-block ablations under-measure a block whose function is also carried
by its neighbors (cf. the old lab's superadditive three-layer block).

**H4 — residual-stream scale confound:** in pre-LN residual nets, if block
write norms shrink with depth, then zeroing late blocks changes the stream
less *by construction*. "Front-loaded importance" could be "front-loaded
writes" — a geometry fact wearing an anatomy costume.

**H5 — under-training:** at ~2k steps late layers may not yet have
specialized; importance might migrate up with longer training.

**Discriminating observations (cheap → expensive):**
1. Measure per-block residual write norms (no training). If write norm tracks
   damage, H4 gains weight; if not, H4 weakens.
2. Mean-replace instead of zero (keep block's mean activation): if damage
   collapses, H2 (miscalibration) explains much of the lesion map.
3. Ablate-then-recalibrate: freeze everything, fine-tune ONLY LayerNorm
   affine params for ~200 steps after each lesion. If damage shrinks a lot,
   the lesion map overstated importance (H2); what remains is closer to true
   information content.
4. Cumulative ablations L4+L5, L3–L5: superadditive damage ⇒ H3.
5. Lesion maps at 500 / 2000 / 8000 steps: importance migrating ⇒ H5.

**Registered predictions (written before running):** write norms will NOT
decline monotonically with depth (they usually grow or stay flat in trained
residual nets), so H4 will NOT fully explain the front-loading; LN-only
recalibration will recover a meaningful fraction (≥30%) of MLP-0's damage,
meaning the +4.08 headline overstates true information content.

**Design consequence:** e011 (MLP-0 anatomy) must include the mean-replace
and LN-recalibrate controls or it will rediscover H2 the hard way.

---

## T002 — Why was unlearning anti-selective? (2026-09-24)

**Observed:** naive ascent on half A destroyed the model (ΔA +26, ΔB +26
nats ≈ random). Worse: by the time A rose +1 nat, B had ALREADY risen +1.51 —
collateral ran ahead of target. Retain anchor slowed but did not rescue.

**H1 — dose pathology:** AdamW on −CE at lr 2e-5 explodes; the model passes
through a regime where everything degrades before A-specific structure fails.

**H2 — structural non-separability (the deep one):** A and B are halves of
the SAME corpus — same style, same vocabulary, same char statistics. Their
gradients are nearly parallel, so ANY weight motion that damages A-knowledge
damages B-knowledge first (shared 'fluency' substrate fails before
content-specific memory). Selective weight-level forgetting of same-
distribution material may be impossible in principle at this scale.

**H3 — wrong measurement axis:** CE mixes general fluency with content
memory. The model may have lost fluency everywhere while both memories are
intact-but-unreadable; or A-memory gone and we can't tell through the
fluency smoke.

**H4 — wrong instrument, not wrong target:** uniform ascent steps are the
blunt tool; weights differ in A-specificity. Targeting only low-A/B-gradient-
overlap weights might find selectivity that uniform stepping can't.

**Discriminating observations:**
1. Gradient cosine between A-batches and B-batches (no training, minutes):
   cos ≳ 0.8 ⇒ H2 is structural and no LR sweep will fix it; cos ≪ 1 ⇒ H1/H4
   remain live.
2. Sweep lr 1e-6…1e-4 × steps; plot the (ΔA, ΔB) trajectory. An operating
   point with ΔA ≥ 1 and ΔB ≤ 0.1 would falsify H2 for this setup.
3. Dissimilar-content unlearning (Shakespeare vs French/code splice): if
   selectivity appears ONLY there, H2 is confirmed — selectivity is a
   property of content distance, not method.
4. Separate fluency from content: after mild ascent, probe A-specific names
   (chars/names unique to A) vs generic continuation quality (H3).
5. Targeted ascent on low-overlap weights only (H4).

**Registered predictions:** grad cosine A↔B will be ≥ 0.85 (H2 structural
for same-corpus halves); no lr in the sweep achieves ΔA ≥ 1 with ΔB ≤ 0.1;
selectivity WILL appear for the dissimilar splice. If these hold, the real
research question shifts from "how to unlearn" to "what is the content-
distance dependence of achievable selectivity" — a curve, not a method.

**Design consequence:** e003 is redesigned around these discriminators
(gradient cosine + dissimilar-content arm + fluency/content split + fine LR
sweep), not a blind hyperparameter grid.

---

*Next thinking obligations: e003 results (the selectivity curve), e011 (MLP-0
must run mean-replace + LN-recalibrate controls), write-norm profile (T001
discriminator 1 — cheapest, should run first).*
