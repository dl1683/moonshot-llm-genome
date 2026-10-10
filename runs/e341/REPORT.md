# E341 — THE VARIED-CONTEXT ANNEALING (the chamber's missing ingredient)

* VERDICT: **AMOUNT-NOT-TYPE** — both annealing arms' s1 survive (varied 0.1154; fixed 0.0972) — the gradient was about STEP COUNT, not context variety (the fixed control at matched steps bought the same survival); the recipe re-words: ANNEAL (any contexts at this dose) + COMMIT(0.7)
* P-e341a (my registered read): MISSED (guess: SHAPING-TRANSFERS, WEAKLY (concurring with the lab lean; AMOUNT-NOT-TYPE the registered runner-up); the dispatch's lab lean: SHAPING-TRANSFERS, weakly)
* gates: 22/22 PASS

## The protocol port (the disclosure)

The annealing is the cons's own construction, ported exactly:
e109/e113's finetune_arm == `g1_anchored_ball.py`'s `consolidate` — the jittered install pool (jitters (-8,-4,0,4,8) x the 60 install contexts = 300 windows, the TAVIREN name at column PRE+j), cons_anchor = anchor_full[:16], per step ix(16) pool draws (name-masked) + aj(8) originals + rj(8) random, token-level union CE, AdamW (0.9,0.95) wd 0.1 const lr 1e-3, clip 1.0, 300 steps, seed 10901. The FIXED control = the SAME arithmetic with the j=0 pool alone (60 windows — the controller's own fixed-context convention) at matched steps and matched name-token dose.

THE QUOTED CONSTRUCTION (substring-verified at run time):

> `lab/g1_anchored_ball.py`: `e113's finetune_arm VERBATIM (recipe/seed/batch composition): batch` [VERIFIED] — *(the cons's own docstring line (the arithmetic being ported))*
> `lab/g1_anchored_ball.py`: `16 install windows from the jittered pool + 16 anchors (8 paired` [VERIFIED] — *(the batch composition (the pool half + the anchor half))*
> `lab/g1_anchored_ball.py`: `constant lr 1e-3 AdamW (0.9,0.95) wd 0.1 clip 1.0; 300 steps,` [VERIFIED] — *(the optimizer/step-count constants)*
> `lab/g1_anchored_ball.py`: `gen = torch.Generator().manual_seed(CONS_SEED)` [VERIFIED] — *(the cons's own generator line (seed 10901))*
> `lab/g1_anchored_ball.py`: `ix = torch.randint(n_pool, (16,), generator=gen)` [VERIFIED] — *(the pool draw (the VARIED axis's carrier: ix over the jittered pool))*
> `lab/g1_anchored_ball.py`: `aj = torch.randint(n_anc, (8,), generator=gen)` [VERIFIED] — *(the paired-originals draw)*
> `lab/g1_anchored_ball.py`: `rj = torch.randint(len(train_ids) - BLOCK - 1, (8,), generator=gen)` [VERIFIED] — *(the random-corpus draw)*
> `lab/g1_anchored_ball.py`: `loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())` [VERIFIED] — *(the masked union CE (the loss form))*
> `lab/g1_anchored_ball.py`: `JITTERS = (-8, -4, 0, 4, 8)       # e109/e113's registered jitter set` [VERIFIED] — *(THE VARIED-CONTEXT AXIS's exact value set (the registered jitters))*
> `lab/g1_anchored_ball.py`: `pool_a_x = torch.cat([jit_x[j] for j in JITTERS])          # (300, 256)` [VERIFIED] — *(the varied pool's construction (5 jitters x 60 install windows))*
> `lab/g1_anchored_ball.py`: `cons_anchor = anchor_full[:16]      # e113: first-16-install original bank` [VERIFIED] — *(the anchor bank (e113's own))*
> `lab/g1e_cons_redraw.py`: `e113 jitter replay VERBATIM (jitters {-8,-4,0,+4,+8} pool` [VERIFIED] — *(the era's own description of the cons protocol (the g1e port))*

The commit placement (the load-bearing birth decision): the primary legs are ANNEAL -> COMMIT(0.7) -> WASH — every class the bars cite is a committed-wash observation, and every bare wash in the record dies at s1 (e335's bare cons-shaped root included). The dispatch's literal bare reading is honored as the secondary panel below.

## The classes (runtime-read, md5-bound)

| class | s1 read | retention | source |
|---|---|---|---|
| fresh (unshaped, committed) | 0.004079 | 0.0448 | e338 COMMIT |
| controller-annealed (fixed ctx, committed) | 0.003655 | 0.1303 | e340 COMMIT |
| cons-shaped (committed) | first-panel dips 0.7460.. | band [0.9572, 1.0879] | the W1 legs |
| bare cons-shaped root | 0.0087 | 0.0003 | e335 cut2 |
| sand (taught) | < 0.05 | <= 0.01 | e322 |

## The annealings (the read rise)

### ARM (a) VARIED — 300 steps, seed 10901, pool 300 windows

| step | read p(T)@g0 | gm12 p(T) | host p(Z) | CE_R |
|---|---|---|---|---|
| 0 | 0.285851 | 0.1744 | 0.0002 | 1.5853 |
| 25 | 0.684342 | 0.7199 | 0.0000 | 1.8480 |
| 50 | 0.803378 | 0.8721 | 0.0001 | 1.8099 |
| 75 | 0.381863 | 0.6450 | 0.0001 | 1.7312 |
| 100 | 0.453776 | 0.7767 | 0.0001 | 1.7261 |
| 125 | 0.308544 | 0.8214 | 0.0002 | 1.7396 |
| 150 | 0.430883 | 0.6936 | 0.0001 | 1.7420 |
| 175 | 0.576084 | 0.8129 | 0.0001 | 1.7320 |
| 200 | 0.692121 | 0.8150 | 0.0000 | 1.7100 |
| 225 | 0.735593 | 0.8652 | 0.0001 | 1.6847 |
| 250 | 0.771332 | 0.7700 | 0.0000 | 1.7158 |
| 275 | 0.575567 | 0.7309 | 0.0000 | 1.6974 |
| 300 | 0.751972 | 0.6881 | 0.0001 | 1.6682 |

### ARM (b) FIXED — 300 steps, seed 10901, pool 60 windows

| step | read p(T)@g0 | gm12 p(T) | host p(Z) | CE_R |
|---|---|---|---|---|
| 0 | 0.285851 | 0.1744 | 0.0002 | 1.5853 |
| 25 | 0.607696 | 0.5400 | 0.0000 | 1.8580 |
| 50 | 0.865117 | 0.6462 | 0.0000 | 1.7888 |
| 75 | 0.743312 | 0.4944 | 0.0001 | 1.7447 |
| 100 | 0.831252 | 0.7168 | 0.0001 | 1.7108 |
| 125 | 0.784200 | 0.6278 | 0.0001 | 1.7248 |
| 150 | 0.711672 | 0.5291 | 0.0001 | 1.7338 |
| 175 | 0.678328 | 0.3727 | 0.0002 | 1.7078 |
| 200 | 0.835176 | 0.5096 | 0.0000 | 1.6948 |
| 225 | 0.872356 | 0.3836 | 0.0000 | 1.6695 |
| 250 | 0.761047 | 0.4279 | 0.0001 | 1.6942 |
| 275 | 0.851431 | 0.3159 | 0.0001 | 1.6975 |
| 300 | 0.755755 | 0.1311 | 0.0000 | 1.6598 |

## The washes (e322's 100-step unbiased wash, seed 10902, bit-identical draws across all five legs)

### varied_committed  (t0 0.751972; s1 0.11542; retention 0.2437; ALIVE at s1)

| step | read p(T)@g0 | gm12 p(T) | host p(Z) | CE_R | |d| raw | note |
|---|---|---|---|---|---|---|
| 0 | 0.751972 | 0.6881 | 0.0001 | 1.6682 | 0.0000 | t0 |
| 1 | 0.115420 | 0.0614 | 0.0000 | 1.6635 | 1.6544 |  |
| 5 | 0.296019 | 0.3392 | 0.0000 | 1.6201 | 1.3538 |  |
| 10 | 0.116204 | 0.1353 | 0.0000 | 1.6209 | 1.2768 |  |
| 25 | 0.192631 | 0.2148 | 0.0000 | 1.6173 | 1.2733 |  |
| 50 | 0.191449 | 0.1922 | 0.0000 | 1.6332 | 1.2742 |  |
| 100 | 0.183254 | 0.1866 | 0.0000 | 1.6159 | 1.2473 |  |

### fixed_committed  (t0 0.755755; s1 0.097188; retention 0.3200; ALIVE at s1)

| step | read p(T)@g0 | gm12 p(T) | host p(Z) | CE_R | |d| raw | note |
|---|---|---|---|---|---|---|
| 0 | 0.755755 | 0.1311 | 0.0000 | 1.6598 | 0.0000 | t0 |
| 1 | 0.097188 | 0.0094 | 0.0000 | 1.6707 | 1.6543 |  |
| 5 | 0.398230 | 0.0274 | 0.0000 | 1.6211 | 1.3473 |  |
| 10 | 0.181277 | 0.0099 | 0.0000 | 1.6150 | 1.2690 |  |
| 25 | 0.196074 | 0.0106 | 0.0000 | 1.6236 | 1.2744 |  |
| 50 | 0.212613 | 0.0079 | 0.0000 | 1.6254 | 1.2691 |  |
| 100 | 0.241859 | 0.0101 | 0.0000 | 1.6160 | 1.2451 |  |

### twin_committed  (t0 0.285851; s1 0.004079; retention 0.0448; DEAD at s1)

| step | read p(T)@g0 | gm12 p(T) | host p(Z) | CE_R | |d| raw | note |
|---|---|---|---|---|---|---|
| 0 | 0.285851 | 0.1744 | 0.0002 | 1.5853 | 0.0000 | t0 |
| 1 | 0.004079 | 0.0042 | 0.0001 | 1.8997 | 1.6543 | **< 0.05** |
| 5 | 0.010868 | 0.0098 | 0.0000 | 1.6188 | 1.1977 | **< 0.05** |
| 10 | 0.008039 | 0.0065 | 0.0000 | 1.6141 | 1.0901 | **< 0.05** |
| 25 | 0.015989 | 0.0103 | 0.0001 | 1.6172 | 1.0706 | **< 0.05** |
| 50 | 0.017449 | 0.0121 | 0.0001 | 1.6252 | 1.0698 | **< 0.05** |
| 100 | 0.012817 | 0.0080 | 0.0001 | 1.6274 | 1.0616 | **< 0.05** |

### varied_bare  (t0 0.751972; s1 0.014514; retention 0.0006; DEAD at s1)

| step | read p(T)@g0 | gm12 p(T) | host p(Z) | CE_R | |d| raw | note |
|---|---|---|---|---|---|---|
| 0 | 0.751972 | 0.6881 | 0.0001 | 1.6682 | 0.0000 | t0 |
| 1 | 0.014514 | 0.0124 | 0.0000 | 1.9921 | 1.6544 | **< 0.05** |
| 5 | 0.016876 | 0.0151 | 0.0000 | 1.7660 | 3.8055 | **< 0.05** |
| 10 | 0.023947 | 0.0205 | 0.0000 | 1.7591 | 4.9025 | **< 0.05** |
| 25 | 0.008006 | 0.0156 | 0.0000 | 1.6945 | 6.8960 | **< 0.05** |
| 50 | 0.002317 | 0.0125 | 0.0000 | 1.6945 | 8.9030 | **< 0.05** |
| 100 | 0.000448 | 0.0034 | 0.0000 | 1.6684 | 11.3071 | **< 0.05** |

### fixed_bare  (t0 0.755755; s1 0.013586; retention 0.0016; DEAD at s1)

| step | read p(T)@g0 | gm12 p(T) | host p(Z) | CE_R | |d| raw | note |
|---|---|---|---|---|---|---|
| 0 | 0.755755 | 0.1311 | 0.0000 | 1.6598 | 0.0000 | t0 |
| 1 | 0.013586 | 0.0037 | 0.0000 | 2.1183 | 1.6543 | **< 0.05** |
| 5 | 0.076935 | 0.0184 | 0.0000 | 1.7832 | 3.7943 |  |
| 10 | 0.036127 | 0.0084 | 0.0000 | 1.7441 | 4.9264 | **< 0.05** |
| 25 | 0.019521 | 0.0047 | 0.0000 | 1.6805 | 6.9853 | **< 0.05** |
| 50 | 0.004411 | 0.0041 | 0.0000 | 1.6924 | 8.9734 | **< 0.05** |
| 100 | 0.001210 | 0.0028 | 0.0000 | 1.6847 | 11.3825 | **< 0.05** |

## The match gates

* G_DRAWS/G_INPUTS: all five washes bit-identical (100 steps)
* G_PIN: max raw displacement at panels 1.6544 <= R + 1.5 = 2.2
* G_REPL: the TWIN+COMMIT arm replicates e338's committed class (s1 0.004079 vs e338's 0.004079; retention 0.0448 vs 0.0448)

## Provenance
* birth commit: 3b19023; final head: a32f0c6116ae370cb272d9cf8dbf3fec8dcce61c
* every artifact md5-bound (see metrics.gates: subject, harness, the cons rigs, the class records); timestamps UTC only; envelope polls to runs/_envelope_log.jsonl tagged e341:*
