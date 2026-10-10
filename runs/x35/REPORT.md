# x35 — THE START-STATE CROSS

Run: 2026-10-10T08:58:01Z (FULL). ONE gen (32401) through BOTH loading conventions at matched everything else (K10K held room bit-bound, ZEPHYRA, 400 Dmix steps, the rig's own lr path; the streams converged identically).


## THE ADJUDICATION (the frozen bars)

- norm_ratio (B root-start / A base-start, census convention vs base) = **2.9326**
- in-own-room: A 0.9449 vs B 0.6319 (abs diff 0.3129)
- base arm pruned/localized (norm <= 12.0 AND in-room >= 0.85): True
- **VERDICT: START-SHAPES** (START-SHAPES iff ratio >= 2.0 AND B in-room <= 0.75 AND base arm pruned; GEN-SHAPES iff ratio <= 1.3 AND in-room diff <= 0.05; residue = CROSS-SPLIT)
- P-x35a (lean START-SHAPES, moderate + the mechanism rider): HIT

## THE TEXTURE TABLES (both arms, the full axes)

| arm | s1 | to-prior | to-standing | to-own-net0 | s100 | post g0 | write norm (vs base) | in-own-room | gm12 | gp12 | CE_R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A BASE-START | 1.3463e-05 | 1.0059e+00x | 1.8077e-05x | 1.0059e+00x | 0.3025 | 0.251815 | 9.2164 | 0.9449 | 0.0934 | 0.1031 | 1.5958 |
| B ROOT-START | 7.4490e-01 | 5.5656e+04x | 1.0002e+00x | 1.0002e+00x | 0.4926 | 0.534977 | 27.0284 | 0.6319 | 0.7019 | 0.6688 | 1.6250 |

## THE NET0 CLASSES (the standing rule)

- ARM_A_BASE_START: BASE-FORMED-START (prior-level; the canon lineage's convention) — theta0 fp64-md5 `b8cd47e63dcc16dae07122bb703623e3` (x32's canon row theta0_md5 (the fingerprint-verified base convention))
- ARM_B_ROOT_START: ROOT-FORMED-START (standing read; the census convention) — theta0 fp64-md5 `4c962fb6ee8da02f45011fc568ed4657` (x32's e324/32401 row theta0_md5 (the fingerprint-verified root convention))

## THE REPLICATION CONTROL (arm B vs e324 GEN32401)

- s1 0.7449015379 vs committed 0.7449015975 (rel 8.00e-08)
- post g0 0.534977 vs 0.534977 (rel 2.23e-07)
- write norm 27.0284 vs 27.0284 (rel 9.93e-11)
- in-own-room 0.6319 vs 0.6319 (abs 1.90e-09)

## THE SUBSTRATE DECOMPOSITION (report-only, registered at birth)

- ||root - base|| = 21.5602 (in-room 0.0601) — the two net0s' own separation
- ARM B vs-base norm 27.0284 = substrate carry + write-from-root 17.5182 (in-room of the write-from-root 0.9783)
- ARM A's net0 IS the base — its vs-base norm 9.2164 IS its own write

## GATES

25/25 gate classes PASS.
