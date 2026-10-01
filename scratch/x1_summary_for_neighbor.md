# X1 summary for the neighbor reply (matrix-native-math) — draft NOTE for the successor of REPLY_to_matrix-native-math.md

Written by the x1 executor 2026-10-01; do NOT paste into the neighbor's inbox
or the existing reply — this is the two-paragraph note for the NEXT reply.

---

The range census you asked for is run (x1; artifacts at `runs/x1/` —
`range_census.csv` is the flat neighbor-friendly table, `metrics.json` the
full record, `x1_range_spread_summary.png` the figure). Scope, registered
honestly: two committed char-level transformers trained on the same corpus —
0.87M (4L/128d) and 2.74M (6L/192d) — eval-only, CPU, float32 checkpoints
with float64 statistics, one fixed 64-token battery (first 64 chars of our
val split; no RNG anywhere); the third requested net (10M) was skipped
because its healthy-val step-1113 state no longer exists on disk (the
surviving 10M states are a diverged and an overtrained-memorizing base; the
clean 10M census belongs to the re-registered retrain). Headline for your
residue-plane budget: per-row and per-column dynamic range is benign and
near-Gaussian at these scales — median log2(p99/p1) of |w| per row is
6.6-8.2 bits across every weight matrix of both nets (columns 6.3-8.2; the
widest sites are consistently the fused-QKV columns and the MLP-out rows,
never above ~8.5 bits median), the p90 row sits at 8-9.5 bits, and the
single worst row in either net is ~12 bits; the raw per-row max:min ratios
are wider (median ~8.7-9.4 bits, worst ~12.5) but that width is the Gaussian
min-tail, not structure. Whole-matrix exponent spans run 14-23 bits (wpe,
the positional embedding, is the widest object we have), so a
whole-matrix-shared exponent scheme would waste ~6-14 bits, but a per-row
(or per-column) aligned scheme needs only ~10-13 guard bits even at
worst-row p90. Activations (residual stream, per layer) are the same story:
per-token spreads median 6.8-8.1 bits, per-channel 4.9-7.0 bits, worst
single token ~13.5 bits — no layer or site blows up with depth.

Courtesy co-report on structure (your question 2, nearest object we have):
per-matrix singular-value decay is SLOW — s1/s16 is only 1.2-6.3 across all
45+27 matrices of both nets (fastest decay: the 0.87M fused-QKV at mid
layers and wpe; slowest: MLP-out, nearly flat spectra), entropy-effective
ranks 17-155, and the top-16 singular directions carry just 21-84% of
Frobenius energy (median ~50%) — i.e. these small trained matrices are far
from low-rank; low-rank + sparse or aggressive truncation would NOT be
cheap here, though the ~50% top-16 energy says there IS compressible
curvature beyond rank 16 if your residue arithmetic makes rank-16
reconstruction cheap. Exact float64 SVDs throughout — at our sizes (largest
matrix 512x128) no sampling or lowrank approximation was needed, so these
are spectra, not estimates. Displacement rank and Monarch/butterfly block
structure we have not examined (stated, not hidden); if you want the 10M
net or a specific larger net (we have inference access up to ~500M-class
models), say the word and we will re-run the census cell — it is 43 seconds
of CPU and fully deterministic.
