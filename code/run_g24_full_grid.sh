#!/usr/bin/env bash
# Full Gate-2 G2.4 grid on 3 text systems x 3 sentinel depths x 5 lambda x 3
# schemes. Per prereg genome_knn_k10_causal_2026-04-21.md §5, the verdict
# requires effect size >5% AND monotonic AND topk-specific on >=2/3 systems.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/gate2/g24_full_grid.log
mkdir -p results/gate2
{
  echo "=== G2.4 FULL GRID START ==="
  for sys in qwen3-0.6b rwkv-4-169m deepseek-r1-distill-qwen-1.5b; do
    for depth in 0 1 2; do
      echo ""
      echo "--- system=$sys depth-idx=$depth ---"
      python code/genome_causal_probe.py --system "$sys" -n 500 --seed 42 \
        --depth-index "$depth" --schemes topk random pca \
        --lam 0.0 0.25 0.5 0.75 1.0 || {
        echo "FAIL system=$sys depth=$depth (continuing)"
      }
    done
  done

  echo ""
  echo "=== G2.4 GRID ANALYSIS ==="
  python - <<'PY'
import json, glob
from pathlib import Path
results = {}
for p in glob.glob("results/gate2/causal_*_depth*_n500_seed42.json"):
    d = json.load(open(p))
    key = (d["system_key"], d["depth_index"])
    results[key] = d

# Per-system summary
systems = sorted({s for (s,_) in results.keys()})
print(f"systems tested: {systems}")
print()
print(f"{'system':30s} {'depth':>6s} {'baseline':>10s} {'topk_lam1_rel':>14s} {'random_lam1_rel':>16s} {'pca_lam1_rel':>14s} {'monotonic':>10s}")
for sys_ in systems:
    for depth in (0, 1, 2):
        if (sys_, depth) not in results:
            continue
        d = results[(sys_, depth)]
        base = d["results"]["baseline"]["loss"]
        topk1 = d["results"].get("topk|lam=1.0", {}).get("rel_delta", 0)
        rand1 = d["results"].get("random|lam=1.0", {}).get("rel_delta", 0)
        pca1 = d["results"].get("pca|lam=1.0", {}).get("rel_delta", 0)
        # Monotonicity: topk across lams
        topk_seq = []
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            k = f"topk|lam={lam}" if lam > 0 else "baseline"
            topk_seq.append(d["results"].get(k, d["results"]["baseline"]).get("loss", float("nan")))
        monotonic = all(topk_seq[i+1] >= topk_seq[i] - 1e-4 for i in range(4))
        print(f"  {sys_:28s} {depth:6d} {base:10.4f} {topk1*100:13.1f}% {rand1*100:15.1f}% {pca1*100:13.1f}% {'YES' if monotonic else 'no':>10s}")

# Pass criteria: topk >5% at lam=1.0 AND monotonic AND topk > random AND topk > pca
passed_systems = set()
for sys_ in systems:
    pass_count_depth = 0
    for depth in (0, 1, 2):
        if (sys_, depth) not in results:
            continue
        d = results[(sys_, depth)]
        topk1 = d["results"].get("topk|lam=1.0", {}).get("rel_delta", 0)
        rand1 = d["results"].get("random|lam=1.0", {}).get("rel_delta", 0)
        pca1 = d["results"].get("pca|lam=1.0", {}).get("rel_delta", 0)
        topk_seq = [d["results"].get(f"topk|lam={lam}" if lam > 0 else "baseline",
                    d["results"]["baseline"])["loss"] for lam in (0.0, 0.25, 0.5, 0.75, 1.0)]
        monotonic = all(topk_seq[i+1] >= topk_seq[i] - 1e-4 for i in range(4))
        passed = topk1 > 0.05 and monotonic and topk1 > rand1 and topk1 > pca1
        if passed:
            pass_count_depth += 1
    if pass_count_depth >= 2:  # at least 2/3 depths pass per system
        passed_systems.add(sys_)

print()
print(f"G2.4 verdict per system (>=2/3 depths pass):")
for sys_ in systems:
    print(f"  {sys_:30s}: {'PASS' if sys_ in passed_systems else 'FAIL'}")
print()
print(f"Prereg primary criterion: 2/3 of (Qwen3, RWKV, DINOv2) pass.")
print(f"Current grid: 3 text systems (Qwen3, RWKV, DeepSeek). DINOv2 is vision — different loss target.")
print(f"Text-systems-only verdict: {len(passed_systems)}/{len(systems)} systems pass = "
      f"{'PROVISIONAL G2.4 PASS' if len(passed_systems) >= 2 else 'G2.4 FAIL'}")
PY
  echo ""
  echo "=== G2.4 GRID COMPLETE ==="
} >> "$LOG" 2>&1
