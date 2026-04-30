# Pre-registration: genome_185v2 Dose-Selection Optimization (Compute-Savings Demonstration)

**STATUS:** DRAFT pre-staged 2026-04-30 cycle 133. **LOCKS only after g186 PASS verdict.** If g186 FAIL, this prereg is ARCHIVED.

- Date: 2026-04-30
- Trigger: Codex §B cycle 128 recommended formalizing "intervention susceptibility" — geometry estimates d(final_NLL)/d(alpha) — and demonstrating compute savings by selecting the optimal KD dose at 3% of training instead of running all doses.
- Supersedes: `genome_185_prospective_triage_2026-04-29.md` (original binary triage design; deprecated because g182 FAILED, no frozen Ridge available).

---

## Motivation

g186 (if PASS) shows that early geometry delta predicts the full KD dose-response curve. g185v2 tests whether that prediction is USEFUL: can a frozen geometry-based dose-selector at 3% of training save compute by picking the right KD dose early, without running all 5 doses to completion?

This is the "electricity-grade efficiency" demonstration the manifesto demands: geometry converts scientific understanding into a concrete compute-savings tool.

## Hypothesis

**H1 (dose-selection saves compute):** A frozen geometry-based dose-selector (Ridge from g186, applied without refitting) selects a KD dose at 3% of training that achieves >= 85% of the oracle's final NLL quality, while requiring only ~20% of the compute needed to train all 5 doses to completion (i.e., >= 80% compute savings vs brute-force).

**H0 (dose-selection not actionable):** The geometry predictor's errors at 3% cause either (a) systematic selection of suboptimal doses (quality retention < 70% vs oracle), or (b) the geometry-selected dose is no better than the alpha-only heuristic (always pick alpha=1.0 or the population mean from g186).

## Design

### Phase 1: Freeze dose-selector from g186

1. Load g186 result JSON (48 delta rows with features + delta_NLL labels).
2. Use the trained Ridge model from g186 (8 manifold features, trained on all 48 rows with optimal Ridge alpha from g186's CV).
3. Define the dose-selection rule:
   - For a new (arch, seed): train all 5 doses to the feature step (3% of training).
   - Compute delta_geometry for each nonzero dose vs the scratch run.
   - Apply frozen Ridge to predict delta_NLL for each nonzero dose.
   - **SELECT** the dose with the highest predicted delta_NLL (most predicted benefit).
   - If ALL predicted delta_NLLs are negative, select alpha=0.0 (scratch — KD predicted harmful at all doses).
4. Save Ridge coefficients + feature means/scales + selection rule as frozen artifact.

### Phase 2: Run fresh prospective cells

- **Architectures:** Qwen3-arch + GPT-2-arch (same as g186).
- **Seeds:** 6 fresh seeds `[6, 7, 8, 9, 10, 11]` — NO overlap with g186 seeds (0-5).
- **Doses:** same 5 alpha levels as g186: `[0.0, 0.3, 0.7, 1.0, 2.0]`.
- **Total cells:** 2 arch x 6 seeds x 5 doses = 60 cells.
- **Training:** identical protocol to g186 (1200 steps, same hyperparameters, same feature step).

### Phase 3: Dose selection at 3% (irreversible decision)

At the feature step (3% of training), for each (arch, seed) pair:
1. Extract 8 manifold features for all 5 dose cells.
2. Compute delta_geometry for each nonzero dose vs the scratch cell.
3. Apply frozen Ridge to predict delta_NLL for each nonzero dose.
4. Record the predicted-best dose (the selection decision).
5. Record all predictions for all doses (for diagnostic analysis).

### Phase 4: Counterfactual comparison (all cells train to completion)

**Critical design choice:** ALL 60 cells train to completion regardless of the selection decision. The dose selection is recorded but not acted on during training. This allows counterfactual analysis.

### Phase 5: Score four policies

**Policy 1 (Brute-Force):** Train all 5 doses to completion, pick the actually-best dose per (arch, seed). Total FLOPs = 5 * F_full per seed. Quality = oracle (by definition, since you see all results).

**Policy 2 (Geometry-Select):** Train all 5 doses to 3%, then only the geometry-predicted-best dose to completion. Total FLOPs per seed = 5 * F_3pct + 1 * F_97pct. Quality = final_NLL of the geometry-selected dose.

**Policy 3 (Alpha-Heuristic):** Always select alpha=1.0 (the g182-era default). Total FLOPs per seed = 1 * F_3pct + 1 * F_97pct (don't even probe other doses). Quality = final_NLL of the alpha=1.0 cell.

**Policy 4 (Population-Mean):** Select the dose that had the highest mean delta_NLL in g186 training data (no geometry, just population prior). Total FLOPs per seed = 1 * F_3pct + 1 * F_97pct. Quality = final_NLL of the population-selected dose.

### Metrics

1. **Compute savings vs brute-force:** `(FLOPs_brute - FLOPs_geometry) / FLOPs_brute * 100`
   - Expected: ~80% (brute-force trains 5 doses; geometry probes 5 at 3% then trains 1).
2. **Quality retention:** `mean_NLL_geometry / mean_NLL_oracle` (closer to 1.0 = better).
3. **Dose-selection accuracy:** fraction of (arch, seed) pairs where geometry selects the actually-best dose.
4. **Regret:** `mean(NLL_oracle - NLL_geometry_selected)` — how much worse geometry's pick is.
5. **Geometry vs alpha-heuristic quality:** paired comparison of final NLLs.
6. **Geometry vs population-mean quality:** paired comparison of final NLLs.

## PASS criteria (all must hold)

1. Compute savings >= 75% vs brute-force (geometry selects 1 of 5 doses, plus 3% probing overhead).
2. Quality retention >= 85% vs oracle (geometry doesn't consistently pick bad doses).
3. Dose-selection accuracy >= 40% (better than random 25% across 4 nonzero doses).
4. Geometry-select beats alpha-heuristic on mean final NLL (paired, p <= 0.10 one-sided).
5. Geometry-select beats population-mean on mean final NLL (paired, p <= 0.10 one-sided).

## WEAK PASS criteria

- Quality retention >= 75% vs oracle, but dose-selection accuracy < 40%.
- OR: geometry ties alpha-heuristic but beats population-mean.

## FAIL criteria

- Quality retention < 70% vs oracle (geometry systematically picks bad doses).
- Dose-selection accuracy <= 25% (no better than random).
- Alpha-heuristic ties or beats geometry on quality AND compute.

## Confound analyses

### C1: Architecture asymmetry
Report dose-selection accuracy and quality retention separately per architecture. If geometry works for Qwen3 but not GPT-2 (or vice versa), the claim narrows to within-family dose selection.

### C2: Dose diversity in g186 training
If g186's Ridge was trained on data where one dose dominated (e.g., alpha=1.0 always best), the selector may just be encoding "always pick alpha=1.0." **Diagnostic:** check if geometry-select agrees with alpha-heuristic on >= 80% of selections. If so, geometry is not adding value over the simple heuristic.

### C3: Seed generalization
The frozen Ridge was trained on g186 seeds 0-5. g185v2 uses seeds 6-11. If the geometry signal is seed-specific (overfitting to g186's random initializations), the selector will fail. This is the primary test — same as g186's held-out-seed design.

## What a null result means

If H0 holds: geometry features predict the dose-response curve statistically (g186 PASS) but the signal-to-noise ratio at the individual-seed level is too low for actionable dose selection. The finding narrows to "population-level diagnostic" — knowing that geometry tracks KD response in aggregate, but not reliably enough to save compute on any single run. The manifesto's efficiency claim would require either more features, more training before selection, or an ensemble approach.

## Connection to manifesto

This is the experiment that converts geometry-as-science into geometry-as-engineering. If PASS: "we saved 80% of dose-search compute by measuring geometry at 3% of training." That is the electricity-grade efficiency headline. Combined with g186 PASS: "geometry predicts KD dose-response, and that prediction saves real compute."

## COMPUTE.md compliance

- [x] Peak VRAM <= 22 GB (same as g186: ~6 GB per cell)
- [x] System RAM <= 56 GB
- [x] Wall-clock <= 4h (60 cells at ~90 sec each = ~1.5h; plus analysis)
- [x] No cloud compute required
- [x] Save/resume: same per-cell atomic writes as g186
- [x] Windows/CUDA rules: num_workers=0, pin_memory=False, n_jobs=1

## Implementation artifacts

- `code/genome_185v2_dose_selection.py` (can fork from g186 with frozen Ridge + selection logic)
- `results/genome_185v2_dose_selection.json`
- Frozen Ridge artifact: embedded in result JSON or separate `results/genome_186_frozen_ridge.json`
