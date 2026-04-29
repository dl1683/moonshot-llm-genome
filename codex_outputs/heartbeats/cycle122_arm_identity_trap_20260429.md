1. Yes. If labels are mostly arm-level and C-prime decodes arm at `R2=0.986`, LOAO mainly tests whether the scratch-vs-KD geometry offset transfers across architecture. With within-KD `R2=-0.52` and label std `0.002`, it is not testing quality sensitivity inside the arm.

2. Meaningful LOAO-PASS: cross-architecture prediction beats:
   - arm-only baseline,
   - architecture-only baseline,
   - early-loss/NLL baseline,
   - arm+dumb-metadata baseline.

   Quantitatively: evaluate on arm-demeaned labels, e.g. `y_resid = y - mean(y | arch, arm)`. A real pass needs positive held-out `R2_resid`, stable Spearman within each arm, and significance under within-arm label permutation. A trivial pass gives high raw R2 but near-zero or negative residual R2.

3. Negative KD labels make trivial passing easier, not more meaningful. If KD uniformly hurts, then “detect KD” becomes “predict worse NLL.” That is a valid intervention diagnosis, but not evidence that geometry predicts training health variance. It says: the geometry can identify the harmful training recipe.

4. This narrows the verdict toward: “C-prime is an intervention/arm detector” and away from “C-prime is a general health forecaster.” P3 falsification plus 2-19 pooled-SD cross-architecture gaps makes universal invariant language weak. The live path is only saved if completed data show residual or severity prediction beyond arm identity.

5. Run these on 48 cells:
   - Arm-demeaned LOAO: predict `y - arm_mean`.
   - Within-arm LOAO separately for scratch and KD.
   - Arm-only vs C-prime vs arm+C-prime model comparison.
   - Permute labels within arm; raw LOAO must beat this.
   - Pairwise delta test: does `Δfeatures(KD-scratch)` predict `ΔNLL(KD-scratch)` across matched seeds?
   - Classifier sanity check: report cross-arch arm AUC separately from quality R2.
   - Partial regression: quality ~ C-prime controlling for `arch`, `arm`, `arch×arm`.
   - Calibration plots by arm: predicted vs actual inside each arm.

Bottom line: LOAO only matters if it survives arm control. Otherwise it is a KD detector wearing a forecasting badge.

