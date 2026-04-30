# Pre-registration: genome_185 Prospective Training Triage (Compute-Savings Demonstration)

**STATUS:** DRAFT pre-staged 2026-04-29 cycle 114. **NEEDS REDESIGN** (cycle 129): g182 FAILED — no frozen Ridge available. If g186 PASS, redesign as **dose-selection optimization**: use g186 dose-response Ridge to predict optimal KD dose at 3% of training, then demonstrate compute savings from not running all 5 doses (Codex §B cycle 128 recommendation: formalize "intervention susceptibility" — geometry estimates d(final_NLL)/d(alpha)).

- Date: 2026-04-29
- Trigger: Codex cycle 111 scored this 9.0/10 as the highest-leverage follow-up to g182 PASS — converts geometry prediction into electricity savings.

---

## Motivation

g182 tests whether geometry features PREDICT training outcome. g185 tests whether that prediction is USEFUL: can an automated triage policy at 3% of training save compute by killing doomed runs without harming good ones? This is the "electricity-grade efficiency" demonstration the manifesto demands.

## Hypothesis

**H1 (triage saves compute):** A frozen geometry-based triage policy (trained on g182 cells, applied without refitting) saves >= 30% of total compute (measured in FLOPs) compared to always-continue, while retaining >= 90% of the quality (measured by mean final NLL of surviving runs) relative to the oracle policy (continue only runs that would have been in the top 50%).

**H0 (triage is not actionable):** The geometry predictor's errors at 3% cause either (a) excessive false kills (good runs stopped), losing >= 10% quality, or (b) excessive false continues (bad runs not stopped), saving < 15% compute. Either failure means the prediction, even if statistically valid, is not practically useful.

## Design

### Phase 1: Freeze triage thresholds from g182

1. Load g182 result JSON (cells with features + normalized labels)
2. Use the trained Model C' Ridge (8 manifold features) from g182 LOAO analysis
3. Define triage thresholds on the Ridge's predicted label:
   - **KILL threshold (T_kill):** predicted_label < P25 of g182 training labels (bottom quartile predictions are killed)
   - **CONTINUE threshold (T_continue):** predicted_label >= P25
   - Note: thresholds are fixed from g182 training data, NOT tuned on g185 data
4. Save thresholds + Ridge coefficients as frozen artifact

### Phase 2: Run fresh prospective cells

- **Architectures:** Qwen3 + GPT-2 (same as g182, but fresh seeds NOT in g182's seed set)
- **Arms:** scratch_ce, seq_kd_full (same as g182 stage 1)
- **Seeds:** 12 fresh seeds (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23) — NO overlap with g182 seeds (0-11)
- **Total cells:** 2 arch x 2 arms x 12 seeds = 48 cells
- **Training:** identical protocol to g182 (3600 steps, same hyperparameters)

### Phase 3: Triage at 3% (irreversible decision)

At step 108 (3% of training), for each cell:
1. Extract 8 manifold features (same pipeline as g182)
2. Apply frozen Ridge to predict label
3. Make irreversible triage decision:
   - If predicted_label < T_kill: **STOP this run** (do not train further)
   - If predicted_label >= T_continue: **CONTINUE to completion** (train full 3600 steps)
4. Record the triage decision, predicted label, and features
5. For stopped runs: still record the final NLL at step 108 (for counterfactual analysis)

### Phase 4: Counterfactual comparison (all runs complete to 3600 steps)

**Critical design choice:** ALL 48 cells train to completion regardless of triage decision. The triage decision is recorded but not acted on during training. This allows counterfactual analysis:
- What would have happened if we had stopped the killed runs?
- What quality and compute would each policy achieve?

### Phase 5: Score three policies

**Policy 1 (Always-Continue):** All 48 runs complete. Total FLOPs = 48 * F_full. Quality = mean(final_NLL across all 48).

**Policy 2 (Geometry Triage):** Only runs with predicted_label >= T_kill continue. Total FLOPs = N_continued * F_full + N_killed * F_3pct. Quality = mean(final_NLL across continued runs only).

**Policy 3 (Oracle):** Continue only runs whose actual label is in top 50%. Total FLOPs = 24 * F_full + 24 * F_3pct. Quality = mean(final_NLL across oracle-selected runs).

### Metrics

1. **Compute savings:** (FLOPs_always_continue - FLOPs_triage) / FLOPs_always_continue * 100
2. **Quality retention:** mean_NLL_triage / mean_NLL_oracle (higher is better, 1.0 = oracle-quality)
3. **False kill rate:** fraction of triage-killed runs that oracle would have continued
4. **False continue rate:** fraction of triage-continued runs that oracle would have killed
5. **Quality-per-FLOP ratio:** (1/mean_NLL_triage) / FLOPs_triage, normalized to always-continue = 1.0

## PASS criteria (all must hold)

1. Compute savings >= 30% vs always-continue (geometry triage stops at least ~30% of runs)
2. Quality retention >= 90% vs oracle (triage doesn't kill too many good runs)
3. False kill rate <= 15% (at most 15% of killed runs were actually good)
4. Quality-per-FLOP ratio > 1.2x vs always-continue (triage genuinely more efficient)

## FAIL criteria

- Compute savings < 15% (geometry can't distinguish good from bad well enough to stop anything)
- Quality retention < 80% vs oracle (too many false kills)
- False kill rate > 25% (dangerously unreliable)

## Compute envelope (COMPUTE.md section 9)

- [x] Peak VRAM <= 22 GB (same as g182: ~6 GB per cell)
- [x] System RAM <= 56 GB
- [x] Wall-clock <= 4h per 12-cell batch (same as g182; 4 batches of 12 = ~16h total)
- [x] No cloud compute required
- NOTE: 48 cells at ~20 min each = ~16h. Can batch in groups of 12 with checkpoint resume.

## Confound analyses (pre-registered, Codex adversarial cycle 115)

### C1: Arm-identity leakage
The frozen Ridge was trained on g182's labeled cells (seq_kd_full only, scratch excluded as denominator). g185 applies it to BOTH scratch and kd cells. If C' manifold features encode arm identity (whether the run uses teacher KD), the Ridge may be a KD-benefit detector rather than a general training-health predictor. **Diagnostic:** run arm_identity_diagnostics() from g182 on the g185 cells. If C' features decode arm identity at AUROC > 0.75, the triage claim is weakened to "geometry predicts KD benefit" (still useful but narrower). **Additional analysis:** compare triage decisions within-arm — does the Ridge differentiate good vs. bad scratch runs and good vs. bad kd runs separately?

### C2: Scratch cell handling
Scratch cells have label = 0 by definition (self-referenced denominator). The triage threshold T_kill = P25 of g182 training labels (all non-scratch). If most labels are positive (kd helps), scratch cells will systematically fall below T_kill and be killed. **Diagnostic:** report triage decisions stratified by arm. If all scratch cells are killed and all kd cells are continued, triage is trivially equivalent to arm-identity decoding.

### C3: Teacher-corpus compatibility as confound
Qwen3-0.6B teacher texts may create a ceiling/floor: Qwen3 recipients benefit more from teacher text than GPT-2. If so, triage is detecting teacher/recipient tokenizer compatibility, not universal training health. **Diagnostic:** report C' predictions stratified by architecture. If Qwen3 cells are systematically predicted higher than GPT-2 cells, the signal is architecture-specific, not geometry-driven.

## What a null result means

If H0 holds: geometry features predict training outcome with statistical significance (g182 PASS) but the signal-to-noise ratio at the individual-run level is too low for actionable triage decisions. The finding narrows to "population-level diagnostic" (like BMI predicting health outcomes — true statistically, useless for individual patients). The manifesto's "electricity-grade efficiency" claim would require either (a) better features, (b) more training time before triage, or (c) an ensemble/Bayesian approach instead of point prediction.

## Connection to manifesto

This is the experiment that converts geometry-as-science into geometry-as-engineering. If PASS: "we saved 30%+ compute by measuring geometry at 3% of training." That is the electricity-grade efficiency headline the manifesto demands. If g182 + g184 + g185 all PASS: geometry predicts training health across architectures and saves real compute — the strongest possible vindication of Intelligence = Geometry.
