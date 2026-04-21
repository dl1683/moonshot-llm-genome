# UNIVERSALITY LEVELS

*The three-tier framework every claim uses. Inherited from the CTI moonshot.*

---

## Why this framework exists

"Universal" is a word that collapses under pressure. A correlation that holds for three models is called universal. A constant that differs across architecture families is called "not universal" — even if its functional form is preserved. Without a shared framework, Codex and Claude end up arguing about vocabulary instead of evidence.

CTI solved this by splitting universality into three nested levels. The Neural Genome inherits the same split. Every claim made in this project — in `EXPERIMENTS.md`, in pre-registrations, in the ledger, in papers — must be tagged with a level.

---

## The three levels

### Level 1 — Functional-Form Universality

**Definition.** A geometric property depends on system variables (scale, training compute, task diversity, depth, width) through the *same functional shape* across every class of neural network tested.

**What "shape" means.** A parametric form: `f(x; θ) = θ_0 + θ_1 · g(x)` where `g` is a named nonlinearity (log, power, sigmoid, EVT form, Gumbel race, etc.). The shape is fixed; the parameters `θ` can vary.

**Example claim (hypothetical).** "Intrinsic dimension as a function of layer depth follows `ID(l) = θ_0 + θ_1 · exp(θ_2 · l/L)` across Transformers, SSMs, hybrids, JEPAs, and diffusion models."

**Evidence required to claim Level 1:**

1. Functional form derived from first principles (information theory, geometry, statistical mechanics, extreme value theory) *before* fitting.
2. Fit tested on ≥5 system classes from the bestiary.
3. Fit R² ≥ 0.9 within each class.
4. Leave-one-class-out cross-validation shows the form generalizes; residual structure is not class-specific.
5. Causal test: intervening on the input variable (scale, training compute) produces the predicted change in the geometric property. Not just correlation.
6. Biological validation: the same form fits neural recordings from the Allen Neuropixels dataset at the Cross-System Auditor's discretion.

**What Level 1 is not.** It is not "the same number everywhere." If α = 1.48 for autoregressive LLMs and α = 0.63 for ViTs, that is a Level-2 family constant, not a Level-1 failure. The *form* `logit(q_norm) = α · κ_nearest − β · log(K−1) + C` is Level 1. The *α value* is Level 2.

**Why this matters.** Level-1 universality is the paradigm-shift claim. It says the same mathematical object governs representation across all learning systems. That is the axiom of the project.

### Level 2 — Family Constants

**Definition.** The coefficients of a Level-1 shape are universal *within* a family (all autoregressive LLMs cluster around one slope; all diffusion models cluster around another). Different families have different constants; those constants are few, predictable, and explain the bulk of cross-family variance.

**Example claim.** "The slope of `ID(l) vs. l/L` is θ_1 = 1.48 ± 0.03 for autoregressive LLMs (n=12 architectures), θ_1 = 0.63 ± 0.05 for ViTs (n=4), θ_1 = 4.4 ± 0.8 for CNNs (n=3). CV within family < 10%."

**Evidence required to claim Level 2:**

1. Level 1 already established (or being established in parallel).
2. Coefficient of variation within family ≤ 10%.
3. Family distinction is structural, not arbitrary: the family boundary corresponds to a named architectural choice (autoregressive vs. masked vs. diffusion; attention vs. state-space vs. convolution).
4. At least 3 members per family — no "family of 1."

**What Level 2 is not.** It is not "the law holds within transformers only." That is not a family constant; that is scope limitation. Level 2 requires that multiple families each exhibit internally universal constants — the family structure is the finding.

**Practical utility of Level 2.** Predictive. Given a new model's family, we can predict its geometric fingerprint from a small number of probe measurements. CTI demonstrated this: 4 probe measurements reduce prediction MAE by 86%.

### Level 3 — Task/Data Intercepts

**Definition.** Additive offsets (intercepts, baseline difficulty terms) vary by task, dataset, or training distribution. These are parameters we fit per-task; their variation is expected.

**Example claim.** "The `C_dataset` intercept in the CTI law ranges from −2.1 (easy tasks) to +1.4 (hard tasks) with a smooth monotonic relationship to task entropy."

**Evidence required to claim Level 3:**

1. Task or dataset variable is specified (entropy, number of classes, domain).
2. Intercept variation is explained by at most one or two task-descriptor variables — not by an ad-hoc per-task free parameter.

**Level 3 is not failure.** Expected. Task specificity is compatible with — and often a consequence of — Level-1 universality.

---

## The decision tree

When you produce a measurement, tag it with a level using this tree:

1. **Is there a derivation for why this shape should hold, from first principles?**
   - No → Phase-2 atlas observation. Record in ledger; do not claim any universality level yet.
   - Yes → go to 2.

2. **Does the shape fit across ≥5 system classes with R² ≥ 0.9 within each?**
   - No → Potential Level 2 only. Check within one family.
   - Yes → go to 3.

3. **Do coefficients vary across families?**
   - No (coefficients are the same across all classes) → **Level 1 with a universal constant.** Rare. Celebrate carefully; double-check the measurement.
   - Yes, but each family is internally tight (CV ≤ 10%) → **Level 1 form + Level 2 family constants.** This is the expected CTI-style outcome.
   - Yes, and families are not tight → Level-1 form did not survive. Demote to Phase-2 observation; consider whether the primitive needs refinement.

4. **Are there additive per-task offsets that vary smoothly with task descriptors?**
   - Yes → **Level 3 intercepts.** Expected and fine.
   - No, offsets are scattered → the primitive may not be capturing what we think it is; reinvestigate.

5. **Has a causal test been run?**
   - No → claim is observational. Do not publish as a Level-1 law.
   - Yes, intervention confirms predicted direction → claim is causal. Proceed.
   - Yes, intervention contradicts predicted direction → the functional form is wrong or the interpretation is wrong. Reinvestigate.

6. **Has biological validation been attempted?**
   - No → claim is "trained-neural-network-level universal."
   - Yes, and form holds on neural recordings → claim is "learning-system-level universal." This is the strongest version.
   - Yes, and form does not hold on biology → claim stays at trained-network level; document the boundary explicitly.

---

## Anti-patterns

| Anti-pattern | What's wrong | Fix |
|---|---|---|
| "We found a universal law" | Level unspecified | Tag the level |
| "It works on 3 Transformers" | n < 3 system classes | Test ≥5 classes before claiming Level 1 |
| Fit R² = 0.87 across 10 models | Within-family variance too high | Check Level 2; group by family |
| Derivation added post-hoc | Curve-fit dressed as law | Require derivation pre-registered before fitting |
| Different slope per model = "failure" | Misunderstanding Level 2 | If CV within family ≤ 10%, it's a success — Level 2 |
| "Universal within transformers" | That is scope, not a level | Label as "scope: autoregressive LLMs" and investigate other families separately |

---

## Labeling convention

Every ledger entry and every pre-registration must carry:

```
universality_level_claimed: 1 | 2 | 3 | null
```

`null` means "Phase-2 atlas observation, no universality claim yet." Most early entries should be `null`. Only after pattern and derivation should entries carry a level.

Codex's Cross-System Auditor persona (CLAUDE.md §7.4) enforces this labeling at every gate.
