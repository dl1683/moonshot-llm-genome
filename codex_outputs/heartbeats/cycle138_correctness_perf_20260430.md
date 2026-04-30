FAIL verdict is correct. I independently recomputed the labels, held-out-seed CV, fold R2s, `arm_mean`, and MSE reduction from the JSON; they match `dose_analysis`.

**Findings**

- SEV 3: The conditioned permutation is very strict, but not misleading for this FAIL. It shuffles geometry within `(arch, alpha)`, so it tests seed-level geometry value after preserving arm identity, not a full Freedman-Lane incremental test over `arm_mean`. Since pooled R2, best-baseline, alpha-only, per-arch, and raw permutation all fail anyway, this cannot be producing an artificial FAIL.

- SEV 2: D5 alpha-decodability is in-sample, so treat it as descriptive only. It is not used in the verdict.

- SEV 1: `alpha_1_only_flag` wording is slightly misleading: it can fire when non-1 doses fail even if alpha=1 also fails. No verdict impact.

No issues found in the requested critical checks: `delta_NLL = scratch - dose` is correct; standardization is train-fold-only; leave-two-seeds-out holds out seed IDs across both architectures; Ridge alpha grid matches prereg; `arm_mean` is fair and cross-validated from train seeds only.

Performance is fine: 48 rows x 8 geometry features is negligible. The 1000 permutation x 3-fold RidgeCV loop is computationally trivial and not memory-risky. No SEV >= 7 findings.

