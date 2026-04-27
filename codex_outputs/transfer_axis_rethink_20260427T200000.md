Grounded in `CLAUDE §0/§0.1`, the four ★ WIKI findings, the corrected `n=7` washout audit, and the post-`g158c` tree, I would demote “more weight anchoring variants” as the center of gravity. The repo now says three things clearly: raw weight copy dies, raw activation matching dies by basis mismatch, and donor signal is real but SGD erases it. That points to transferring function, routing, or aligned coordinates, not just parameter proximity.

**New Techniques**
1. **Full-stack re-basin + norm-refit zero-step transplant**: Fit per-layer permutation/orthogonal alignment plus RMSNorm refit on a calibration set, then copy donor blocks into the aligned coordinates with zero gradient steps. Better than anchor-decay because it attacks the exact failure mode from `g121-g124` (basis + norm mismatch) instead of regularizing the wrong coordinates. Compute: `<=2h`, `<=20 GB`. PASS: across 3 seeds, mean zero-step C4 NLL gain `>= +0.8 nat` vs random-init, 95% CI excludes 0, and it beats the old `all_attn` zero-step effect by `>=5x`. Score on PASS: **8.3/10**. Relation: **orthogonal**.

2. **Functional scaffold distillation (“ScaffoldSwap”)**: During early training, selected donor attention outputs are mixed into the recipient forward path with `alpha(t) -> 0`, while the recipient learns CE plus imitation; donor path is fully gone by the end. Better than anchor-decay because it transfers live computation, not static weights, so the student learns a working function before the scaffold disappears. Compute: `500` steps, 3 seeds, two models live, `~3-4h`, `14-18 GB`. PASS: after scaffold removal, recipient-alone final C4 NLL advantage `>= +0.5 nat` vs scratch with 95% CI > 0. Score: **8.0/10**. Relation: **orthogonal**.

3. **Transport-gated token KD**: Standard top-k logit KD, but heavily weight only long-context / high-disagreement / high-transport-demand tokens; CE everywhere else. Better than anchor-decay because it uses `g158c`’s locked control variable and `g154`’s validated KD pipe, while avoiding teacher spend on easy local tokens. Compute: cached teacher logits + `1k-2k` student steps, `<=4h`, `<=10 GB`. PASS: beats CE-only by `>= +1.0pp` at `L=256` and beats ungated KD by `>= +0.4pp`, CI > 0 across 3 seeds. Score: **7.8/10**. Relation: **orthogonal / complementary to g158c**.

4. **Attention-routing KD (“RouteKD”)**: Transfer donor attention maps or top-k token-to-token edges at transport-heavy layers, not donor weights or raw activations. Better than anchor-decay because `all_attn` was the only mildly positive surgery component, suggesting routing is closer to the transferable object than parameter values. Compute: cache selected attention maps, `500-1000` steps, `<=4h`, `10-14 GB`. PASS: beats CE-only by `>= +0.3 nat` and logit-only KD by `>= +0.1 nat` at `L=256`, with no regression at `L=32`. Score: **7.6/10**. Relation: **complementary**.

5. **Spectral scaffold transfer**: Initialize each recipient layer inside the donor’s low-rank singular subspace, but randomize coefficients / complement; optionally freeze the subspace briefly, then unfreeze. Better than anchor-decay because it preserves donor geometry while allowing adaptation instead of forcing exact donor weights. Compute: offline truncated SVD + `500` train steps, `~3h`, `<=12 GB`. PASS: at least one rank yields final NLL advantage `>= +0.3 nat` vs scratch, CI > 0. Score: **7.1/10**. Relation: **complementary**.

6. **Gradient-sketch replay**: Transfer donor Fisher / RMS gradient sketches or projected update directions from a small calibration set, then train recipient normally. Better than anchor-decay because it transfers “how to move” rather than “where to sit,” which matters if SGD is what destroys donor priors. Compute: sketch + `200-500` steps, `<=3h`, `<=12 GB`. PASS: `CtQ_75 >= 1.5x` faster than scratch and final NLL `>= +0.2 nat`, CI > 0. Score: **6.9/10**. Relation: **near-orthogonal; update-space cousin of g166**.

**Ranking By PASS Ceiling**
1. Re-basin + norm-refit zero-step transplant — **8.3**
2. Functional scaffold distillation — **8.0**
3. Transport-gated token KD — **7.8**
4. Attention-routing KD — **7.6**
5. `g165` annealed-donor / decay-anchor — **7.3**
6. Spectral scaffold transfer — **7.1**
7. Gradient-sketch replay — **6.9**
8. `g162` transport-arm capacity sweep — **6.8**
9. `g166` optimizer-state + decay-anchor — **6.4**

Two honest notes. First, probability-weighted, `g165` is still the safest active bet because it is already grounded by the corrected `n=7` washout pattern. Second, if the wall-power meter appears, `g155` still jumps above this whole non-hardware queue on manifesto cash-out.

**Priority Shift**
Fire **immediately in parallel with `g165`**: **full-stack re-basin + norm-refit zero-step transplant**. It is cheap, fully orthogonal, and it tests the last serious loophole in the killed zero-step surgery story: alignment, not anchoring.

Fire **after `g165` finishes**: **functional scaffold distillation**. If `g165` fails and ScaffoldSwap passes, you get a sharp statement: donor signal persists when transferred as temporary computation, not as weight proximity. That is much more informative than `g166`, and more ceiling-bearing than `g162`.

Net: I would demote `g166` further, keep `g162` as theory-tightening rather than discovery-first, and open a new top lane around **aligned zero-step transfer** plus **functional scaffold transfer**.