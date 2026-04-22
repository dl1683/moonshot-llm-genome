# Pre-registration: Geometry→Efficiency decision rule, kNN-k10 power-law R² ↔ NLL under weight quantization

**Status:** STAGED at draft commit, to be LOCKED once Qwen3-1.7B is confirmed loadable and C4 stimulus hash is pinned.

**Date:** 2026-04-21.

**Authors:** Dev (CMC / AI Moonshots).

**Primary deliverable** per strategic-adversarial Codex verdict 2026-04-21-2047 (`.codex/outputs/strategic_2026-04-21-2047.md`): one tight pre-registered demo that promotes the Geometry→Efficiency finding (genome_018 + genome_023) from descriptive to decision-making.

---

## 1. Research question

Does `R² of the C(X, k) = c_0 · k^p` power-law fit, measured on pooled hidden-state activations at FP16 and Q8, **predict** whether aggressive Q4 compression will substantially degrade next-token loss — on a system not yet in the geom-efficiency sweep?

## 2. Pre-specified decision rule (locked here)

For a trained causal LM `M` at a specified mid-depth, let:

- `R²_FP16` = R² of the least-squares fit of `log C(k)` vs `log k` across `k ∈ {3, 5, 8, 12, 18, 27, 40, 60, 90, 130}` on a pooled 500-stimulus C4-clean batch at seed 42, with weights at FP16.
- `R²_Q8` = same, with bitsandbytes 8-bit quantization (`load_in_8bit=True`).
- `ΔR²_Q8` = `R²_Q8 − R²_FP16` (expected negative or zero).
- `NLL_FP16, NLL_Q4` = next-token cross-entropy per token on the same batch at FP16 and bitsandbytes 4-bit NF4 (`bnb_4bit_quant_type="nf4"`).
- `ΔNLL_Q4% = 100 · (NLL_Q4 − NLL_FP16) / NLL_FP16`.

**Rule.**

> If `ΔR²_Q8 ≤ −0.003`, then we predict `ΔNLL_Q4% ≥ 2.0%`.

**Justification of the thresholds.** Observed cross-system deltas in `results/gate2/geom_efficiency*.json`:

| System | `ΔR²_Q8` | `ΔNLL_Q4 %` |
|---|---:|---:|
| Qwen3-0.6B (genome_018) | −0.004 | +3.7 |
| RWKV-4-169M (genome_023b) | −0.003 | +7.3 |
| DeepSeek-R1-Distill-Qwen-1.5B (genome_023 DeepSeek-3q) | −0.003 | +2.1 |

All three have `ΔR²_Q8 ≤ −0.003` and all three have `ΔNLL_Q4 ≥ 2%`. The rule would therefore have **predicted correctly on 3/3 training data**. This pre-registration tests whether the rule survives on a **held-out model** not used to set the threshold.

## 3. Held-out test system

- **System:** `Qwen/Qwen3-1.7B` — 1.7B-param Qwen3-family transformer CLM.
- **Rationale:** Same architecture family as Qwen3-0.6B (which was in the training set), but 2.8× parameters. Not in any previous geom-efficiency experiment. If the rule is a Qwen3-specific constant, it trivially transfers; if it's an architectural coincidence, it may fail. Qwen3-1.7B is in `Projects/models/MODEL_DIRECTORY.md`.
- **Depth:** mid-depth per CLAUDE.md sentinel convention (`layer = num_hidden_layers // 2`).
- **Stimuli:** 500 C4-clean samples at seed 42 (same bank as all three training-set systems, hash-pinned in the locked prereg below).
- **k-grid:** `{3, 5, 8, 12, 18, 27, 40, 60, 90, 130}`.
- **Quant backend:** bitsandbytes, `load_in_8bit=True` and NF4 `load_in_4bit=True`, same as genome_018/023.

## 4. Pre-registered outcome categories (one of these four must be reported)

1. **Rule VALIDATED** — `ΔR²_Q8 ≤ −0.003` AND `ΔNLL_Q4% ≥ 2.0%`. The rule predicts correctly.
2. **Rule FALSE NEGATIVE** — `ΔR²_Q8 > −0.003` (rule says "geometry not drifted") AND `ΔNLL_Q4% ≥ 2.0%` (capability nevertheless broke). Rule missed a real break.
3. **Rule FALSE POSITIVE** — `ΔR²_Q8 ≤ −0.003` AND `ΔNLL_Q4% < 2.0%`. Rule cried wolf.
4. **Rule VACUOUSLY PASSES** — `ΔR²_Q8 > −0.003` AND `ΔNLL_Q4% < 2.0%`. Both signals absent; rule coincidentally correct.

Only outcomes **1** and **4** constitute correct predictions. Outcomes **2** and **3** falsify the rule.

## 5. Sample size and statistical power

Single pass at n=500, seed 42. Per genome_018/023, NLL measurements are stable to ~0.001 at this sample size (n_tokens ≈ 64,000), and `C(k)` values are stable to ~0.003 per k (cross-seed CV ≈ 5% on text), so the `ΔR²_Q8` and `ΔNLL_Q4%` differentials dominate estimator noise by an order of magnitude. A single-seed test is adequate for the point prediction.

## 6. Stop conditions / abort

- If Qwen3-1.7B fails to load at any of FP16/Q8/Q4 on the local envelope (22 GB VRAM), abort and substitute with next-available system in the registry.
- If bitsandbytes NF4 produces NaN activations (seen once at Q4 on Falcon-H1), abort and report as environmental failure, not rule failure.

## 7. Reporting commitment

Ledger entry `genome_035_geom_eff_decision_rule_qwen3_1p7b` with fields:

- `rule_verdict ∈ {VALIDATED, FALSE_NEGATIVE, FALSE_POSITIVE, VACUOUSLY_PASSES}`
- Raw `ΔR²_Q8`, `ΔNLL_Q4%`.
- The 10-k-grid `C(k)` values at all three quant levels.
- Whether the prediction would also be correct on a tighter threshold (robustness check).

**Paper §5.5 will be updated to report the blind-test outcome regardless of sign.**

## 8. Compute envelope compliance (CLAUDE.md §1.5)

- **VRAM:** Qwen3-1.7B at FP16 ≈ 3.4 GB; with bnb quantization less. Well under 22 GB.
- **Wall clock:** ~15 min for 3 quant levels × extract + NLL. Well under 4 h.
- **Network:** model weights cached from prior Qwen3 downloads; zero new bandwidth expected.

## 9. LOCK procedure

This file is LOCKED by the commit that adds it to the repository, with `git_commit` recorded in the ledger entry. Any change after LOCK invalidates the pre-registration; a new file dated and suffixed `_v2` is required.

Locked-at-commit: (to be filled by the commit that finalizes this draft)

---

**Pre-registered by:** Dev (devansh@svam.com), 2026-04-21.
