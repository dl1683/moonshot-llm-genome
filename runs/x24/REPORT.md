# x24 — THE PRIOR-FRAGILITY CENSUS (NAME-SPECIFIC-STRUCTURE)

**The question (T285):** is a name's lift under a fixed full-dose generic
displacement a generic function of its prior (or anchoring), or
name-specific structure? **The displacement:** the K10K write's complement
at full dose (x15's construction, bit-equal to its committed carrier,
probed from the carrier's own model dict — ONE applied state for every
name read). Its committed effect on two names: ZEPHYRA 0.0007308
("unmoved" in absolute terms), TAVIREN 0.0272878 (lifted) — both
reproduced in-session before any clause counted.

**The panel:** 11 names, 11 distinct initial slots — the anchor ZEPHYRA
(the displacement's parent write's own target name; a disclosed confound),
TAVIREN (committed reference), 3 FRESH e293-bank names (QELVARO / BUVONDI
/ NYSTORA — count-0, no relation to the write), the scrambled control
VIRETAN (TAVIREN's letters, initial V), and 5 corpus names with stated
train-split counts (MAMILLIUS 13, LEONTES 125, KING 556, ELIZABETH 105,
FLORIZEL 45). **Two reading sites:** the host's 60 g0 install contexts
(the committed bank) + a fresh NEUTRAL bank (60 host-free corpus
capitals, seed 24001).

## THE LIFT-vs-PRIOR CENSUS (read = p(name[0]), mean over 60 contexts)

| site | name | class | count | base prior | comp read | LIFT | dlogit | gain | fullK lift | fullT lift | argmax |
|---|---|---|---|---|---|---|---|---|---|---|---|
| host_g0 | **ZEPHYRA** | anchor_formed_host | 0 | 1.338e-05 | 0.0007308 | **54.60x** | +4.00 | 0.0007174 | 19773.49x | 14.07x | 0.00 |
| host_g0 | **TAVIREN** | synthetic_count0 | 0 | 0.002304 | 0.02729 | **11.84x** | +2.50 | 0.02498 | 1.63x | 124.05x | 0.02 |
| host_g0 | **QELVARO** | synthetic_count0_fresh_bank | 0 | 0.002591 | 0.001293 | **0.50x** | -0.70 | -1.297e-03 | 3.05x | 0.80x | 0.00 |
| host_g0 | **BUVONDI** | synthetic_count0_fresh_bank | 0 | 0.00909 | 0.02868 | **3.15x** | +1.17 | 0.01959 | 0.67x | 1.28x | 0.00 |
| host_g0 | **NYSTORA** | synthetic_count0_fresh_bank | 0 | 0.002826 | 0.005357 | **1.90x** | +0.64 | 0.002531 | 1.13x | 0.98x | 0.00 |
| host_g0 | **VIRETAN** | scrambled_control | 0 | 0.005194 | 0.002289 | **0.44x** | -0.82 | -2.904e-03 | 2.72x | 1.45x | 0.00 |
| host_g0 | **MAMILLIUS** | corpus_rare_seen | 13 | 0.1141 | 0.187 | **1.64x** | +0.58 | 0.07292 | 0.56x | 0.57x | 0.13 |
| host_g0 | **LEONTES** | corpus_mid_proper | 125 | 0.0307 | 0.01615 | **0.53x** | -0.66 | -1.454e-02 | 0.63x | 0.98x | 0.00 |
| host_g0 | **KING** | corpus_common_token | 556 | 0.006559 | 0.008406 | **1.28x** | +0.25 | 0.001846 | 0.82x | 0.60x | 0.00 |
| host_g0 | **ELIZABETH** | corpus_host_high | 105 | 0.5886 | 0.3466 | **0.59x** | -0.99 | -2.420e-01 | 0.59x | 0.59x | 0.55 |
| host_g0 | **FLORIZEL** | corpus_host_high | 45 | 0.02307 | 0.00864 | **0.37x** | -1.00 | -1.443e-02 | 4.53x | 3.08x | 0.00 |
| neutral | **ZEPHYRA** | anchor_formed_host | 0 | 2.020e-05 | 0.0005764 | **28.53x** | +3.35 | 0.0005562 | 322.56x | 4.11x | 0.00 |
| neutral | **TAVIREN** | synthetic_count0 | 0 | 0.07263 | 0.1128 | **1.55x** | +0.48 | 0.04016 | 1.01x | 1.17x | 0.23 |
| neutral | **QELVARO** | synthetic_count0_fresh_bank | 0 | 0.005315 | 0.0005221 | **0.10x** | -2.33 | -4.793e-03 | 1.95x | 2.14x | 0.00 |
| neutral | **BUVONDI** | synthetic_count0_fresh_bank | 0 | 0.05663 | 0.04729 | **0.84x** | -0.19 | -9.342e-03 | 0.92x | 0.97x | 0.03 |
| neutral | **NYSTORA** | synthetic_count0_fresh_bank | 0 | 0.01334 | 0.02307 | **1.73x** | +0.56 | 0.009731 | 0.58x | 0.64x | 0.02 |
| neutral | **VIRETAN** | scrambled_control | 0 | 0.003731 | 0.001518 | **0.41x** | -0.90 | -2.213e-03 | 1.65x | 1.10x | 0.00 |
| neutral | **MAMILLIUS** | corpus_rare_seen | 13 | 0.03945 | 0.02961 | **0.75x** | -0.30 | -9.844e-03 | 0.92x | 0.85x | 0.02 |
| neutral | **LEONTES** | corpus_mid_proper | 125 | 0.07351 | 0.03674 | **0.50x** | -0.73 | -3.677e-02 | 0.96x | 0.96x | 0.03 |
| neutral | **KING** | corpus_common_token | 556 | 0.005376 | 0.004823 | **0.90x** | -0.11 | -5.529e-04 | 1.13x | 0.81x | 0.00 |
| neutral | **ELIZABETH** | corpus_host_high | 105 | 0.1321 | 0.04094 | **0.31x** | -1.27 | -9.115e-02 | 0.94x | 0.92x | 0.05 |
| neutral | **FLORIZEL** | corpus_host_high | 45 | 0.01676 | 0.01233 | **0.74x** | -0.31 | -4.430e-03 | 1.77x | 1.79x | 0.00 |

## THE CLAUSES (frozen at birth)

| site | cohort-A spread (bar <= 3x) | anchor lift | cohort-A median | clause B (<= 0.2x med) | interval | clause C | all three |
|---|---|---|---|---|---|---|---|
| host_g0 | 26.86x (FAIL) | 54.60x | 1.77x | FAIL | [0.44x, 54.60x] | FAIL | FAIL |
| neutral | 17.61x (FAIL) | 28.53x | 0.79x | FAIL | [0.10x, 28.53x] | PASS | FAIL |

## Verdict: NAME-SPECIFIC-STRUCTURE

some marginal names lift >> others — TAVIREN vs VIRETAN at host_g0 (priors 0.0023/0.00519, lifts 11.8x/0.4x, 26.9x spread among same-prior names) — fragility is structure, not prior; the census continues with name-geometry probes; necessity stays weakened.

**P-x24a scoring (registered pre-compute, never shopped):**
Register P-x24a BEFORE compute. Lab guess: ONE-GENERIC-CURVE. -> **MISS — the verdict is NAME-SPECIFIC-STRUCTURE**.
Executor: MISS — the executor predicted GAP; the verdict is NAME-SPECIFIC-STRUCTURE.

**Riders (never bars):** Spearman(log-prior, log-lift) — host
-0.418, neutral
-0.209; the anchor's
formed-fraction (p_comp/p_fullK on ZEPHYRA) — host
0.0028, neutral 0.0884; the
x10-prior rider census (comp < 10x AND fullK > 10x) — host:
(none); neutral: (none). ce_r — base
1.616, comp
2.060, fullK 1.592, fullT
1.585; top-1 p on host — base
0.667, comp
0.516. No
organism collapsed; every read is a name-level fact.

## Gates: 16/16 PASS

G_ENDPOINTS (10 md5 binds incl. the displacement carrier + the
fullT carrier + the e293 bank source; 12 committed
references read from artifacts, never retyped), G_FLATBASIS, G_PANEL
(+G_NAMEFREE; e311's convention ported; initials distinct; counts exact),
G_BATTERY (host-g0 splice bank verbatim, 19/41, [60,130]), G_NEUTRAL
(seed 24001; all upper-next; host-free), G_BASEPRIOR (both committed
priors in [0.5x, 2x]; every synthetic <= 0.05), G_PANELPRIOR (e293's
0.004 band, co-report), G_ROOM (D/S bit-bound), G_ORTH, G_REGEN (the
displacement bit-equal to x15's carrier, every key), G_DOSEMATCH (1e-12),
G_INROOM (energy share <= 1e-12), G_FULLREAD (fullK|Z |d|
0.0e+00; fullT|T |d| 3.0e-08),
G_DIAG (comp|Z x1.0000; comp|T x1.0000),
G_ONESTATE.

Disclosures: the read currency is the name-INITIAL char (the committed
KK/KT currency); lift = ratio for every clause (the dispatch's own
measure) — the committed values (anchor 54.6x vs TAVIREN 11.8x at host)
strained clause B pre-compute and that strain was registered at birth;
the anchor is the displacement's parent write's own name (confound
disclosed; the fresh bank is the clean cohort); cohort membership frozen
by construction, priors measured per site; n=1 per cell (one lineage,
one session — the standing lottery note); bars + P-x24a frozen at birth
before any compute.
