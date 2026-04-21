#!/usr/bin/env bash
# Chained step 2 of the autonomous mission: wait for batch2_pipeline.log to
# produce its STEP 4 "=== PIPELINE COMPLETE ===" marker, then immediately
# run the Gate-2 G2.4 causal-ablation smoke test on Qwen3 (50 stimuli, all
# 3 schemes × 5 λ values at middle sentinel depth). Populates the G2.4
# prereg §11 last checkbox, enabling the STAGED→LOCKED transition.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/gate2/g24_smoke_pipeline.log
mkdir -p results/gate2
{
  echo "=== STEP A: wait for batch2_pipeline STEP 4 complete ==="
  until grep -q "PIPELINE COMPLETE" results/cross_arch/batch2_pipeline.log 2>/dev/null; do
    sleep 20
  done
  echo "Batch-2 pipeline finished. Sleeping 5s to ensure GPU freed..."
  sleep 5

  echo ""
  echo "=== STEP B: G2.4 causal-ablation smoke test on Qwen3 (n=50) ==="
  python code/genome_causal_probe.py --system qwen3-0.6b -n 50 --seed 42 \
    --depth-index 1 --schemes topk random pca \
    --lam 0.0 0.25 0.5 0.75 1.0

  echo ""
  echo "=== STEP C: interpret G2.4 smoke result ==="
  python - <<'PY'
import json
from pathlib import Path
# Find the most recent G2.4 smoke output
p = Path("results/gate2/causal_qwen3-0.6b_depth1_n50_seed42.json")
if not p.exists():
    print("ERROR: smoke output not found")
    exit(1)
d = json.load(open(p))
print(f"Qwen3 G2.4 smoke (n=50, depth-idx=1):")
baseline = d["results"]["baseline"]["loss"]
print(f"  baseline NLL: {baseline:.4f}")
print(f"{'scheme':<8s} {'lam':>6s} {'loss':>8s} {'delta':>10s} {'rel':>8s}")
for key, v in d["results"].items():
    if key == "baseline" or not isinstance(v, dict):
        continue
    if "loss" not in v:
        continue
    if "|lam=" in key:
        scheme, lam_str = key.split("|lam=")
        lam = float(lam_str)
        loss = v["loss"]
        delta = v.get("delta", loss - baseline)
        rel = v.get("rel_delta", delta / baseline if baseline else 0)
        print(f"  {scheme:<6s} {lam:6.2f} {loss:8.4f} {delta:+10.4f} {rel:+8.2%}")

# Verdict logic per prereg §5:
# Primary: topk lam=1 delta > 5% of baseline (we'll see if smoke is in that range)
# Monotonicity: topk loss non-decreasing in lam
import math
topk = {}
for key, v in d["results"].items():
    if isinstance(v, dict) and key.startswith("topk|lam="):
        lam = float(key.split("|lam=")[1])
        topk[lam] = v["loss"]
# Expect topk monotonic up
print()
print(f"Monotonicity of topk: {sorted(topk.items())}")
prev = -1
monotonic = True
for lam in sorted(topk):
    if topk[lam] < prev - 1e-4:
        monotonic = False; break
    prev = topk[lam]
print(f"topk monotonic non-decreasing: {monotonic}")
PY

  echo ""
  echo "=== G2.4 SMOKE DONE ==="
} >> "$LOG" 2>&1
