# x10 — THE DEAD RUNG'S FILL (FILLS-ANYWAY)

**The registered question (P-x10a, T253):** x6's fill law says a quiet
write fills ~73% of whatever room it is granted (10k: 7,342/10,000;
237k: 174,283/237,123). e272 relocated the expression edge to (1k,2k]:
the 1k write is DEAD (post g0 0.000435). Does the dead 1k write STILL
fill ~73% of its 1,000-dim room — or does expression failure leave a
different geometric signature?

## The fill curve (x6's conventions; each rung in its own room)

| rung | k | post g0 (cited) | inside share | in-room m50/m90/m99 | FILL m99/k |
|---|---|---|---|---|---|
| 1k (DEAD, e261) | 1,000 | 0.000435 | 0.8082 | 119/431/735 | **0.7350** |
| 2k (transitional, e272) | 2,000 | 0.026616 | 0.8264 | 247/893/1466 | **0.7330** |
| 5k (alive-ish, e272) | 5,000 | 0.127096 | 0.8546 | 615/2231/3676 | **0.7352** |
| 10k (alive, x6's object) | 10,000 | 0.264648 | 0.8914 | 1243/4428/7342 | **0.7342** |
| 237k (natural width, x6) | 237,123 | 0.384364 | 0.9614 | 29330/105413/174283 | **0.7350** |

**Co-reports (never bars):** K1KM — e272's kept-matched DEAD 1k arm
(lr x3.7306, post 0.0012): fill 0.7270,
inside 0.6329 (the dead fill under a 3.7x
dose change, same room). K10KR — e272's fresh-room 10k replicate
(post 0.2097): fill 0.7330, inside
0.8921 (the room lottery priced on the
fill law itself).

## Verdict: FILLS-ANYWAY

Clause trace: fill_1k = 0.7350 in [0.55, 0.90] =
True; fill_1k < 0.40 = False; top50_1k
(0.2933) >= 2 x top50_10k (0.0496) =
True (ratio 5.92x);
the [0.40, 1.00-eps] upper clause is vacuous (m99 <= k by
construction); 2k trend-break = False.

The 10k and 237k reads were RECOMPUTED by this cell and reproduce x6's
committed metrics exactly (G_X6REPRO: inside shares 0.8914359886/
0.9613593175, m's integer-exact) — the
pipeline is x6's before any new number is believed.

## Gates

- G_ENDPOINTS: e001/e261_K10K/e261_K237K/e261_rooms md5-bound to x6's
  committed binds; e261/e264/e272 metrics md5-bound to e272's G_PARENTS.
- G_DISPL (content bind for the un-md5'd vehicles): every arm's fp64
  inside_share matches the parent cell's committed in_own_room NORM
  ratio squared within 2e-3 (max abs diff
  6.42e-12; e261 K237K
  matches to 4.37e-13).
- G_ROOMS: all 7 room entries bit-verified vs fresh seed reconstruction;
  e272's K1KM room re-gated bit-equal to the committed K1K room.
- G_FLATBASIS: key order == TinyGPT parameters, no buffers, N=2,739,072.

## Disclosures

1. e261_K1K_inst_resume.pt and the four e272 inst_resumes carry NO
   committed md5 anywhere in the record (searched runs/*/metrics.json).
   This cell binds them by fresh md5 (recorded in metrics G_ENDPOINTS,
   citable by future cells) + step==400 + the G_DISPL content match +
   the room bit-verification.
2. The dead rung read is e261's committed K1K arm; e272's K1KM
   (dose-matched) is a co-report, never a bar. The 10k expression cite
   is e264's committed rung value on the same net x6 read
   (e261_K10K_inst_resume s400); the 237k cite is e261's K237K arm
   (post 0.384364).
3. The in-room spectra are stored decimated (~240 log-spaced cumulative
   points) in metrics.json; the m's/top50/fill are computed on the full
   spectra, never the decimation.
4. CPU-only: threads 4/4, no envelope-log writes, no GPU code path.
