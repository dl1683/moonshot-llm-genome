# x12 — THE RUNAWAY DISCRIMINATOR (MIXED)

**The registered question (P-x12a, T256):** x9's stretched-exponential fit
gave the FACT battery beta = 1.31 (w1 1.2672 / w2 1.3458; bootstrap CIs
[1.137, 2.894] / [1.148, 2.357] — the only battery excluding 1). Is that
acceleration PREDATORY (per-probe damage runs away as strength falls,
H-DRAIN-RUNAWAY) or SUPERPOSITION (a bi-exponential mixture of fast and
slow probes imitating acceleration, H-SHAPE-MISMATCH)?

## (i) The family table (fact battery; x9's data + machinery, BIEXP added)

| wash | family | k | theta | NLL | AICc | lam_eff | bound_hit |
|---|---|---|---|---|---|---|---|
| w1 | EXP | 2 | 0.0070137, 0 | 45.4600 | **95.0759** | 0.007014 | True |
| w1 | STR | 3 | 0.0083561, 1.2672, 0 | 45.3940 | **97.1037** | 0.007504 | True |
| w1 | BIEXP | 4 | 0.015437, 0.0070137, 0, 0 | 45.4600 | **99.4534** | 0.007014 | True |
| w1 | oneT | 4 | - | 45.5574 | **99.6482** | None | False |
| w2 | EXP | 2 | 0.0071514, 0 | 36.3099 | **76.8303** | 0.007151 | True |
| w2 | STR | 3 | 0.00877, 1.3458, 0 | 36.2281 | **78.8848** | 0.007759 | True |
| w2 | BIEXP | 4 | 0.0045322, 0.0071514, 0, 0 | 36.3099 | **81.3471** | 0.007151 | True |
| w2 | oneT | 3 | - | 36.5878 | **79.6041** | None | False |

**dA := AICc_BIEXP - AICc_STR** (positive = stretched wins): w1
**+2.350**, w2 **+2.462** -> A-channel =
**STR-wins** (bar |dA| >= 2 on both washes).

Bootstrap stability (B=200 probe-level resamples, single-start polish,
co-report never a bar; das := AICc_STR - AICc_BIEXP, negative = stretched
wins by that margin): w1 median -2.36 (95%
[-3.02, -2.26]; stretched
wins by >= 2 in 100% of
resamples, BIEXP wins by >= 2 in
0%); w2 median
-2.49 (95% [-2.77,
-2.33]; stretched
100% / BIEXP
0%).
The EXP-vs-STR context: x9's committed fact EXP AICc w1
95.08 / w2 76.83
(this cell reproduced them, G_REPRO).

## (ii) The derivative test (e228's committed per-probe margins)

damage = -d(gap)/d(state); strength = from-state margin_raw; pooled over
both wash lineages (independent from the shared t0).

| battery | n probes | rho pooled | rho w1 | rho w2 | pairs | negative margins |
|---|---|---|---|---|---|---|
| ctrl | 12 | +0.0595 (p=5.9e-01) | +0.0274 | +0.1084 | 84 | 0 |
| fact | 20 | +0.0857 (p=3.1e-01) | +0.0320 | +0.1710 | 140 | 0 |
| near | 3 | +0.2197 (p=3.4e-01) | -0.0070 | +0.5000 | 21 | 0 |
| tmpl | 19 | +0.3602 (p=2.1e-05) | +0.2582 | +0.5112 | 133 | 0 |

**PRIMARY (fact): rho = +0.0857** (p =
3.14e-01); margin_sigma variant
-0.0049; bar rho <= -0.5 ->
B-channel does not fire.

Arithmetic disclosure: under per-probe pure exponential decay damage =
lam x strength -> rho -> +1, and a cross-probe MIXTURE of exponentials
also keeps rho > 0 — the -0.5 bar is deep in genuine per-probe
acceleration territory. That is why the derivative test is the
runaway-specific channel and the family test the mixture-specific one.

## Verdict: MIXED

Clause trace: BIEXP wins both washes = False;
STR wins both washes = True; rho <= -0.5 =
False; no |dA| >= 2 signal on either wash =
False.


## Gates

- G_SHA: the 8 e238 npz dumps sha16-match x9's committed G_SHA table;
  runs/x9/metrics.json (10a8ff05c3f62310), runs/e228/journal.json
  (e1f476405746ad70), runs/e228/metrics.json (da0fb7af1a3d105e)
  fresh-bound (recorded for future cells).
- G_JOURNAL: 8 states, probe sets aligned by name at every state, all
  margins finite.
- G_REPRO: this cell's EXP/STR refits reproduce x9's committed fact fits
  (nll <= 1e-6, AICc <= 1e-4, theta <= 1e-3) and the one-T refit
  reproduces x9's per-state T within 5e-3 — the machinery is x9's before
  any BIEXP number is believed.

## Disclosures

1. The BIEXP family is NEW (k = 4); its w bound at 0 or 1 degenerates to
   EXP — flagged as bound_hit, never hidden.
2. The derivative test pools both wash lineages; the shared t0 from-state
   double-counts across them (disclosed at birth); per-wash co-reports
   shown; the margin_raw field is primary per the dispatch, margin_sigma
   the sensitivity.
3. The bootstrap uses single-start polish from full-data thetas (not the
   full multistart) — a stability co-report, never a bar.
4. CPU-only: threads 1, torch never imported, no envelope-log writes, no
   NOTES/THINKING/QUEUE/STATE edits; all timestamps datetime.now(UTC).
