# x13 — FLAT TAX vs WEALTH TAX (W047's follow-up tension)

**Status**: COMPLETE — verdict **MIXED**
**Envelope**: CPU-only desk cell (no torch import, threads 1 (<= 2 per the
dispatch), 0 GPU calls, 0 envelope-log writes, no NOTES/THINKING/QUEUE/STATE
edits; every timestamp datetime.now(UTC)).
**Registration**: bars frozen VERBATIM in the script docstring, birth-committed
before compute (git head at start f009585c3c90350f31e7d13c6f7a547934b80687).

## The tension

x9 fitted exponentials and called the one-T ladder a RATE ladder (exponential
margins imply per-step damage proportional to current gap — a WEALTH tax).
x12 measured the damage-strength Spearman ~ 0 for every battery — damage
independent of remaining strength — a FLAT tax (linear margin decline). Both
cannot be right; x9's family set NEVER INCLUDED LINEAR, and over a limited
state range exp and linear are nearly indistinguishable by R2. The missing
member decides.

## Gates (all machine-checked)

- G_SHA: all 8 e238 dumps' sha256 match x9's committed G_SHA table AND x12's
  independent re-verification; runs/x9/metrics.json hashed fresh matches x12's
  committed record.
- G_MEANP: float64 softmax mean_p reproduces x9's committed discharge curves
  to 0.00e+00 (tol 1e-09).
- G_REPRO: this cell's EXP refit reproduces x9's 8 committed EXP rows to
  max |dtheta| = 0.00e+00,
  max |dnll| = 0.00e+00.

## The 8-fit dAICc table (both families k=2, so dAICc = 2*dNLL exactly)

| fit | NLL_LIN | NLL_EXP | AICc_LIN | AICc_EXP | dAICc | winner | max mean-curve gap |
|---|---|---|---|---|---|---|---|
| ctrl/w1 | 30.385 | 30.378 | 65.04 | 65.02 | **-0.01** | inconclusive | 0.014 |
| ctrl/w2 | 23.557 | 23.541 | 51.48 | 51.45 | **-0.03** | inconclusive | 0.013 |
| fact/w1 | 45.452 | 45.460 | 95.06 | 95.08 | **+0.02** | inconclusive | 0.018 |
| fact/w2 | 36.267 | 36.310 | 76.75 | 76.83 | **+0.09** | inconclusive | 0.018 |
| near/w1 | 6.812 | 6.803 | 18.96 | 18.94 | **-0.02** | inconclusive | 0.064 |
| near/w2 | 5.215 | 5.211 | 16.43 | 16.42 | **-0.01** | inconclusive | 0.096 |
| tmpl/w1 | 38.270 | 38.293 | 80.70 | 80.75 | **+0.05** | inconclusive | 0.037 |
| tmpl/w2 | 33.716 | 33.544 | 71.65 | 71.31 | **-0.34** | inconclusive | 0.031 |

LIN wins (dAICc >= 2): **0/8**; EXP wins (dAICc <= -2): **0/8**;
inconclusive: 8/8.

## The consistency check (hybrid diagnostic, cross-cell)

| battery | x12 rho (margin) | bridge rho (p) | LIN-implied rho | EXP-implied rho | x12 closer to |
|---|---|---|---|---|---|
| ctrl | +0.059 | -0.310 | +0.602 | +0.944 | LIN |
| fact | +0.086 | -0.271 | +0.724 | +0.973 | LIN |
| near | +0.220 | +0.488 | +0.908 | +0.959 | LIN |
| tmpl | +0.360 | -0.125 | +0.386 | +0.891 | LIN |

Arithmetic: pure EXP (c=0) forces damage = lam*strength -> rho = +1; pure LIN
makes damage constant within a probe. Consistent-with-LIN (all batteries
closer to LIN-implied): **True**.

## Discrimination power (the design statement)

Noiseless-template scan on the measured state range: D = 1 - g(80) of the TRUE
template; D* = the decline depth at which the design separates the families
at dAICc >= 2 (systematic-signal threshold; probe dispersion not modeled —
the bootstrap co-report carries the noise side).

| fit | D* (truth=LIN) | D* (truth=EXP) | D obs (LIN fit) | D obs (EXP fit) |
|---|---|---|---|---|
| ctrl/w1 | 0.977 | >0.98 | 0.399 | 0.388 |
| ctrl/w2 | 0.978 | >0.98 | 0.382 | 0.374 |
| fact/w1 | 0.914 | 0.845 | 0.446 | 0.429 |
| fact/w2 | 0.916 | 0.848 | 0.455 | 0.436 |
| near/w1 | >0.98 | >0.98 | 1.381 | 0.902 |
| near/w2 | >0.98 | >0.98 | 3.152 | 0.976 |
| tmpl/w1 | 0.894 | 0.820 | 0.579 | 0.549 |
| tmpl/w2 | 0.898 | 0.825 | 0.504 | 0.489 |

Medians: D* truth=LIN **0.9152343750000003**,
D* truth=EXP **0.8350585937500002**,
observed D (LIN) 0.479,
observed D (EXP) 0.462.

## Bootstrap dAICc (probe resampling, B=200, seed 20261007; never a bar)

| fit | median | 95% CI | frac >= 2 | frac <= -2 |
|---|---|---|---|---|
| ctrl/w1 | -0.01 | [-0.07, +0.01] | 0.00 | 0.00 |
| ctrl/w2 | -0.03 | [-0.10, -0.01] | 0.00 | 0.00 |
| fact/w1 | +0.03 | [-0.14, +0.40] | 0.00 | 0.00 |
| fact/w2 | +0.09 | [-0.03, +0.32] | 0.00 | 0.00 |
| near/w1 | -0.02 | [-0.05, +0.02] | 0.00 | 0.00 |
| near/w2 | -0.01 | [-0.04, +0.02] | 0.00 | 0.00 |
| tmpl/w1 | +0.05 | [-0.27, +0.19] | 0.00 | 0.00 |
| tmpl/w2 | -0.34 | [-0.84, -0.10] | 0.00 | 0.00 |

## Verdict clause (frozen routing)

split or inconclusive: n_lin=0, n_exp=0, inconclusive=8, consistent-with-LIN=True — the table verbatim + the discrimination-power statement below

## Honesty

n=1 organism; near n=3 (coarse); x12's measured rho is in MARGIN currency
while the implied rhos are in x9's p currency (the bridge row shows the
currency gap on the same transitions); the discrimination template is
noiseless (systematic-signal threshold, not a power analysis); the LIN
survival hinge max(0, 1-slope*t) never activates at the fitted slopes; equal
k means dAICc is pure likelihood ratio; no bar shopping — routing frozen in
the birth commit.
