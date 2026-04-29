# Cycle 110 — Codex Review: route3_predictions diagnostics (lines 2024-2300)

**Model:** GPT-5.5, medium reasoning effort
**Date:** 2026-04-29 ~18:30 UTC
**Session:** 019dda73 (route3 review)

## Scope

Reviewed `route3_predictions()` function + R1/R3/D1/D2/P6/P7b diagnostics for data leakage and methodological issues.

## Verdict: CLEAN (no changes needed)

1. **P1 full-dataset KMeans+ARI** — OK as descriptive diagnostic; not a predictive/generalization claim.
2. **P2 all-data Ridge refit** — OK for descriptive coefficient ranking; caveat: avoid treating weights as stable causal importances under correlated features.
3. **D1/D2 LOO implementations** — FIXED in cycle 109. Scaling is train-only, Ridge alpha CV is inside each LOO train fold, D1 KMeans is fit only on train-fold `X_tr_s`.
4. **P6 LOO polynomial** — OK: scaling is train-only, polynomial expansion is fit on train fold and transformed on test fold.
5. **R1/R3/P7b** — R1 descriptive-only (fine). R3 permutation is one-sided and unstratified — minor caveat: possible arch/arm composition confounding. P7b is descriptive CV convergence (fine).

## Also reviewed this cycle

- **frozen_eval_main (lines 1603-1885):** ALL PASS — Ridge trained only on g182 data, Falcon uses native tokenizer, no refit during evaluation, bootstrap/permutation use only Falcon test data. Caveat: cached teacher_texts trusted without provenance validation (minor).

## Net

All g182 analysis code is Codex-verified clean. Ready for data.
