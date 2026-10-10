# X40 — THE FIRST-STEP SURVIVAL MECHANISM (why annealed reads survive the first projected step)

* VERDICT: **MIXED** — multiple mechanisms fire: BROAD-SUPPORT + VACCINATION — the decomposition reported
* fired: ['BROAD-SUPPORT', 'VACCINATION']; P-x40a (my registered read): HIT (guess: MIXED (BROAD-SUPPORT + VACCINATION), weakly (concurring with the lab lean); the dispatch's lab lean: MIXED (broad-support + vaccination), weakly)
* gates: 22/22 PASS

## 1. The per-context distribution at s1 (the survival axis's anatomy)

| arm | battery | t0 mean | t0 alive | s1 mean (mine) | s1 alive | s1 mean (committed) |
|---|---|---|---|---|---|---|
| VARIED | g-12 | 0.6881 | 1.000 | 0.061403 | 0.467 | None |
| VARIED | g0 | 0.7520 | 0.983 | 0.115421 | 0.400 | 0.11542 |
| VARIED | g+12 | 0.7601 | 1.000 | 0.070101 | 0.417 | None |
| FIXED | g-12 | 0.1311 | 0.583 | 0.009395 | 0.033 | None |
| FIXED | g0 | 0.7558 | 1.000 | 0.097188 | 0.383 | 0.097188 |
| FIXED | g+12 | 0.1842 | 0.750 | 0.015492 | 0.100 | None |
| FRESH | g-12 | 0.1744 | 0.817 | 0.004182 | 0.000 | None |
| FRESH | g0 | 0.2859 | 0.883 | 0.004079 | 0.000 | 0.004079 |
| FRESH | g+12 | 0.2206 | 0.833 | 0.003993 | 0.000 | None |

## 2. The dose desk (the vaccination check)

| arm | path length | endpoint | max roam | per-step mean | per-step max | floor (2x write) |
|---|---|---|---|---|---|---|
| VARIED | 102.66 | 16.778 | 16.778 | 0.342 | 1.654 | 17.899 |
| FIXED | 102.71 | 16.802 | 16.802 | 0.342 | 1.654 | 17.899 |

* x38's committed ladder (md5-bound): 1x 9.1788 / 2x 18.3577 / 4x 36.7154 L2 — GAUSS-BREAKS-AT-4x; the 2x rung is the floor's value (the dispatch's own '~2x the write's norm').
* e311's write norm 8.9496 -> the floor 17.8993; the committed ENDPOINT doses (varied 16.779 / fixed 16.801) sit ~6-7% BELOW it — the path length is the clause's carrier (frozen at birth; see P-x40a refinement 2).

## 3. The sensitivity curves (the basin proxy)

| arm | curve | 0.25R | 0.5R | 1R |
|---|---|---|---|---|
| VARIED | kill_ray | 0.7950 | 0.4531 | 0.1535 |
| VARIED | gaussian | 0.9991 | 0.9982 | 0.9963 |
| FIXED | kill_ray | 0.7669 | 0.4223 | 0.1286 |
| FIXED | gaussian | 1.0007 | 1.0013 | 1.0025 |
| FRESH | kill_ray | 0.0629 | 0.0290 | 0.0143 |
| FRESH | gaussian | 0.9837 | 0.9674 | 0.9350 |

## The clauses (the frozen composite)

```json
{
 "broad_support_clause1_material": {
  "annealed_min_g0": 0.38333332538604736,
  "fresh_g0": 0.0,
  "margin": 0.1,
  "fires": true
 },
 "broad_support_clause2_offset": {
  "alive": {
   "VARIED": {
    "g-12": 0.46666666865348816,
    "g0": 0.4000000059604645,
    "g+12": 0.4166666567325592
   },
   "FIXED": {
    "g-12": 0.03333333507180214,
    "g0": 0.38333332538604736,
    "g+12": 0.10000000149011612
   },
   "FRESH": {
    "g-12": 0.0,
    "g0": 0.0,
    "g+12": 0.0
   }
  },
  "fires": true
 },
 "vaccination_clause1_dose": {
  "path_lengths": {
   "VARIED": 102.65931212902069,
   "FIXED": 102.70670491456985
  },
  "noise_floor_2x_write": 17.89925812031448,
  "endpoints": {
   "VARIED": 16.778236389160156,
   "FIXED": 16.802047729492188
  },
  "max_roam": {
   "VARIED": 16.778236389160156,
   "FIXED": 16.802047729492188
  },
  "per_step_max": {
   "VARIED": 1.6543740034103394,
   "FIXED": 1.6543701887130737
  },
  "fires": true
 },
 "vaccination_clause2_flatter": {
  "kill_ray_retention_0.5R": {
   "VARIED": 0.45309007658371536,
   "FIXED": 0.42232331484753177,
   "FRESH": 0.029018515039387145
  },
  "kill_ray_retention_0.25R": {
   "VARIED": 0.7950214481644655,
   "FIXED": 0.7668645378537778,
   "FRESH": 0.06289055083828203
  },
  "gaussian_retention_0.5R": {
   "VARIED": 0.9982217815824004,
   "FIXED": 1.0012932726399018,
   "FRESH": 0.9674267464352332
  },
  "gaussian_retention_1R": {
   "VARIED": 0.9963108744944316,
   "FIXED": 1.0024547573433917,
   "FRESH": 0.9350063352433765
  },
  "fires": true,
  "direction_at_0.25R": true
 }
}
```

## Provenance
* birth commit: fc0d67d; final head: 2e23c8e67c3fc38134527fc65c98391b8e180ecd
* every artifact md5-bound (G_MD5: 16 binds); the s1 events reproduce e341's committed first-step input bit-exact + the three s1 classes (G_S1EVENT); CPU-only (threads 4); timestamps UTC only
