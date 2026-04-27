`Score` below means expected `§0.1` uplift on `PASS`, not total project score. I excluded the obvious `g155 + wall meter` path per your constraint.

**A. Experimental Ideas**
1. **Conditional last-7 atlas patch.** Hypothesis: `g084` failed because the atlas used unconditional teacher means; conditioning the last-7 patch on early-layer scaffold/task/position (`g112`, `g083`, `g084`) will restore context-sensitive capability, not just unigram prior. Missed because the atlas branch was scope-corrected, then effectively retired before the `g112` scaffold result was folded back in. Compute: `~1–2h, <8GB`. Parallel with `g158c`: `not really` (brief GPU eval). Score: `+1.8/10`. §0 link: closest remaining zero-step transfer handle.

2. **Receiver-first transfer on a scaffold recipient.** Hypothesis: the missing ingredient in `g117–g124` was a receiver, not a donor object; create the receiver first with the `g125` scaffold recipe (interfaces/gammas only), then inject donor object. Missed because `g125` was diverted into the architecture-prior chain instead of being treated as a transfer precondition. Compute: `~2–3h, <12GB`. Parallel: `no`. Score: `+1.6/10`. §0 link: near-direct transfer with tiny recipient adaptation.

3. **Full-stack permutation `Re-Basin` on a partially trained recipient.** Hypothesis: `g124` killed only `T0` rotation, not norm-aware permutation alignment; a partially trained recipient plus layerwise permutation/gamma re-permutation can make donor weights readable. Missed because the PC1 null was over-generalized to the whole alignment path. Compute: `~3–4h, <16GB`. Parallel: `no`. Score: `+1.5/10`. §0 link: direct transformation-based transfer, exactly the post-`g119` theoretical recommendation.

4. **Task/content-direction surgery, not structural-PC1 surgery.** Hypothesis: the transferred object in `g117/118` was the wrong one; use `g112` task scaffold / deeper semantic directions rather than the sentence-boundary axis from `g116b/e`. Missed because the catastrophic PC1 effect dominated attention and prematurely stood in for “capability direction.” Compute: `~1–3h, <8GB`. Parallel: `no`. Score: `+1.2/10`. §0 link: selective capability transfer is still transfer.

5. **Conditional cross-family last-7 atlas transfer.** Hypothesis: if conditioned atlas works at all, test it across families, not just cross-size within Qwen (`g082`); same-h or projected cross-family transfer is a much stronger zero-step statement. Missed because atlas was shelved after the unigram-prior correction. Compute: `~1–2h, <10GB`. Parallel: `no`. Score: `+1.1/10`. §0 link: direct trained-model-to-other-model transfer generality.

6. **Annealed donor-attn / mean-shift warm start.** Hypothesis: donor priors help only very early, then become liabilities (`grafting_007–009`, `g125`); inject them for the first `N` steps, then decay them away. Missed because fixed-persistence failures were treated as exhaustion of the whole scheduling dimension. Compute: `~1–2h, <10GB`. Parallel: `no`. Score: `+0.9/10`. §0 link: tests whether “cheap partial transfer” exists even if strict zero-step does not.

**B. No-New-Compute Analysis Ideas**
1. **Conditional-atlas mismatch audit.** Reanalyze `g078–084` plus `g112`: do early-layer clusters/task bins predict the donor-minus-lesion residual the last-7 patch should add? Compute: `CPU-only`. Parallel: `yes`. Score: `+1.2/10`. §0 link: validates or kills the best zero-step revival before burning GPU.

2. **Early-help / no-persistence meta-analysis.** Combine `grafting_005–009`, `g125`, `g134`, `g137`, `g089` to locate the maturity window where donor signal helps before washout. Compute: `CPU-only`. Parallel: `yes`. Score: `+1.0/10`. §0 link: directly designs the receiver-first / annealed-transfer experiments.

3. **Eigenvector-content audit.** Use `genome_097/099`, `spectrum_dump_analysis`, and `g126–133` to test whether capability variation lives in left-singular subspace content rather than the universal eigenvalue law. Compute: `CPU-only`. Parallel: `yes`. Score: `+1.0/10`. §0 link: identifies what object should actually be transferred.

4. **All-attn residue meta-analysis.** Pool `g120–125` to test whether the `+0.6–0.9%` `all_attn` signal is a real transferable residue rather than noise. Compute: `CPU-only`. Parallel: `yes`. Score: `+0.7/10`. §0 link: narrows the only weight subset with repeated positive sign.

**C. Dormant Paths Worth Reviving**
1. `research/FINAL_10_MOONSHOTS.md` and `research/MOONSHOT_CANDIDATES.md` are absent; using `WIKI`, preregs, and `EXPERIMENTS` as the dormant-path record.

2. **Neural-genome atlas path.** Retired too broadly. `g083/084` killed only the unconditional atlas as a practical surgery tool, not conditional/gated/cross-family variants.

3. **Transformation path.** Proper permutation `Re-Basin`, donor-init + structured noise, and inference-time RSA-style transfer were all recommended after `g123/124` and never truly tested.

4. **Receiver-first / delayed-transfer path.** The repo has multiple signs that a receiver must exist before transfer (`g117`, `g125`, `g087/089`, grafting), but no experiment actually targeted that regime.

5. **`g159b` and `g161`.** These were archived for leverage/ops reasons, not cleanly falsified. They matter less than §0 transfer work, but they are unresolved, not dead.

**D. Fire Right Now**
Run the **conditional-atlas mismatch audit** now, in parallel with `g158c`.

Reason: it is `CPU-only`, it directly tests the strongest zero-step transfer revival, and it tells you whether the atlas branch was truly dead or merely mis-specified. If the audit says early-layer scaffold/task state predicts the needed last-7 correction, the next GPU slot should go to **conditional last-7 atlas on partial lesions**, not another architecture-prior refinement.