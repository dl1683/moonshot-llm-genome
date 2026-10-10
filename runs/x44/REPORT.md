# X44 — R*-AT-THE-OFFSETS (is x38's flip a slot property or a state-at-context property?)

* VERDICT: **R*-UNIFORM** — the geometries' flip levels agree within ~2x (max state ratio nan) AND the nine cells' constraints are pooled-consistent (max lower 0.0442 < min upper 0.05274) — the flip is a slot property; the two-channel law's parameter is context-independent
* P-x44a (my registered read): MISSED — guess R*-TRACKS-THE-SUPPORT-TAIL, weakly (concurring with the lab lean); the dispatch's lean: R*-TRACKS-THE-SUPPORT-TAIL, weakly
* direction: None
* gates: 16/16 PASS

## 1. The flip cells (the primary band form; the dispatch's per-context question)

| state | geo | pre mean (alive) | post mean (alive) | L/C/A | L_max | C_min | R_hat | viol | floor | satup |
|---|---|---|---|---|---|---|---|---|---|---|
| VARIED | g-12 | 0.6881 (1.00) | 0.1435 (0.93) | 0/59/1 | None | 0.06542 | None | 0 | 0 | 2 |
| VARIED | g0 | 0.7520 (0.98) | 0.0915 (0.70) | 1/59/0 | 0.03931 | 0.05274 | 0.04553 | 0 | 0 | 3 |
| VARIED | g+12 | 0.7601 (1.00) | 0.0941 (0.87) | 0/60/0 | None | 0.12824 | None | 0 | 0 | 0 |
| FIXED | g-12 | 0.1311 (0.58) | 0.0457 (0.32) | 21/33/6 | 0.08585 | 0.05865 | None | 7 | 0 | 1 |
| FIXED | g0 | 0.7558 (1.00) | 0.2072 (0.73) | 0/60/0 | None | 0.06592 | None | 0 | 0 | 1 |
| FIXED | g+12 | 0.1842 (0.75) | 0.0496 (0.37) | 13/42/5 | 0.0442 | 0.05316 | 0.04847 | 0 | 0 | 2 |
| FRESH | g-12 | 0.1744 (0.82) | 0.1252 (0.88) | 15/37/8 | 0.22209 | 0.05395 | None | 20 | 0 | 7 |
| FRESH | g0 | 0.2859 (0.88) | 0.1509 (0.93) | 9/49/2 | 0.09292 | 0.05011 | None | 9 | 0 | 6 |
| FRESH | g+12 | 0.2206 (0.83) | 0.1419 (0.92) | 10/44/6 | 0.27237 | 0.05361 | None | 22 | 0 | 7 |

## 2. The geometry comparison (the frozen composite)

```json
{
 "state_ratios": {},
 "pooled_brackets": {
  "g-12": {
   "pooled_L_max": null,
   "pooled_C_min": null,
   "n_two_sided": 0,
   "R_hat": null
  },
  "g0": {
   "pooled_L_max": 0.039309922605752945,
   "pooled_C_min": 0.05274265632033348,
   "n_two_sided": 1,
   "R_hat": 0.045533611079884004
  },
  "g+12": {
   "pooled_L_max": 0.04419664666056633,
   "pooled_C_min": 0.05316195636987686,
   "n_two_sided": 1,
   "R_hat": 0.04847246848948268
  }
 },
 "pooled_bounds": {
  "max_lower": 0.04419664666056633,
  "min_upper": 0.05274265632033348
 },
 "pooled_consistent": true,
 "primary_exists": false,
 "secondary_exists": true,
 "tracks_fire": false,
 "tracks_source": null,
 "uniform_fire": true
}
```

## 3. The riders (descriptive, never bars)

### The margin map (the dispatch's closing claim, countable)
```json
{
 "VARIED": {
  "g-12": {
   "R_hat_used": null
  },
  "g0": {
   "R_hat_used": 0.045533611079884004,
   "n_above": 59,
   "n_below": 1,
   "p_alive_post_given_pre_above": 0.6949152542372882,
   "p_alive_post_given_pre_atbelow": 1.0
  },
  "g+12": {
   "R_hat_used": 0.04847246848948268,
   "n_above": 60,
   "n_below": 0,
   "p_alive_post_given_pre_above": 0.8666666666666667,
   "p_alive_post_given_pre_atbelow": null
  }
 },
 "FIXED": {
  "g-12": {
   "R_hat_used": null
  },
  "g0": {
   "R_hat_used": 0.045533611079884004,
   "n_above": 60,
   "n_below": 0,
   "p_alive_post_given_pre_above": 0.7333333333333333,
   "p_alive_post_given_pre_atbelow": null
  },
  "g+12": {
   "R_hat_used": 0.04847246848948268,
   "n_above": 45,
   "n_below": 15,
   "p_alive_post_given_pre_above": 0.4222222222222222,
   "p_alive_post_given_pre_atbelow": 0.2
  }
 },
 "FRESH": {
  "g-12": {
   "R_hat_used": null
  },
  "g0": {
   "R_hat_used": 0.045533611079884004,
   "n_above": 54,
   "n_below": 6,
   "p_alive_post_given_pre_above": 0.9259259259259259,
   "p_alive_post_given_pre_atbelow": 1.0
  },
  "g+12": {
   "R_hat_used": 0.04847246848948268,
   "n_above": 50,
   "n_below": 10,
   "p_alive_post_given_pre_above": 0.94,
   "p_alive_post_given_pre_atbelow": 0.8
  }
 }
}
```

### The sensitivity form (parameter-free; never adjudicates)
```json
{
 "state_ratios": {
  "VARIED": {
   "R_hats": {
    "g-12": null,
    "g0": 0.15849703083290068,
    "g+12": null
   },
   "ratio_max_over_min": null
  },
  "FIXED": {
   "R_hats": {
    "g-12": null,
    "g0": null,
    "g+12": null
   },
   "ratio_max_over_min": null
  },
  "FRESH": {
   "R_hats": {
    "g-12": null,
    "g0": 0.13791473554467357,
    "g+12": null
   },
   "ratio_max_over_min": null
  }
 },
 "verdict_agrees": true
}
```

### The paired elicitation correlations (same 60 host positions across geometries)
```json
{
 "VARIED": {
  "g-12|g0": 0.32645383757181856,
  "g0|g+12": 0.4595522544294192,
  "g-12|g+12": 0.8782171704110276
 },
 "FIXED": {
  "g-12|g0": 0.606837072381079,
  "g0|g+12": 0.6383906132751799,
  "g-12|g+12": 0.9180048456039056
 },
 "FRESH": {
  "g-12|g0": 0.9720452877514917,
  "g0|g+12": 0.9791284880678133,
  "g-12|g+12": 0.9722471582157748
 }
}
```

### ce_r per state (bare -> comp) + noise-zone lifts

* VARIED: CE_R 1.6682 -> 2.0893; noise-zone lifts g-12 0/g0 0/g+12 0
* FIXED: CE_R 1.6598 -> 2.0364; noise-zone lifts g-12 0/g0 0/g+12 0
* FRESH: CE_R 1.5853 -> 2.1486; noise-zone lifts g-12 0/g0 0/g+12 0

## 4. The references (runtime-read from the md5-bound records)

* x38's committed host-g0 bracket: (1.338e-05, 0.5302] (verdict SHARP-FLIP)
* x42's committed interior window: (0.02172, 0.5052]
* x40's committed s1 support table (the reference this cell re-reads): {"VARIED": {"g-12": 0.46666666865348816, "g0": 0.4000000059604645, "g+12": 0.4166666567325592}, "FIXED": {"g-12": 0.03333333507180214, "g0": 0.38333332538604736, "g+12": 0.10000000149011612}, "FRESH": {"g-12": 0.0, "g0": 0.0, "g+12": 0.0}}

## Provenance
* birth commit: 1bd264c67153b0ff17314a7c8372eb813f26b535; final head: None
* 19 md5 binds; the carrier certified end-to-end (model == base + delta bit-equal + x24's 22 host-g0 panel cells bit-exact through this cell's own reader); the six committed t0 reads + three ce_r reproduce; CPU-only (threads 4); timestamps UTC only; no NOTES/THINKING/QUEUE/STATE edits (the heartbeat folds)
