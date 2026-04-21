#!/usr/bin/env bash
# Queue CLIP seed 42 re-run after batch2 pipeline completes — the seed-42
# CLIP run failed before the n_hidden_layers fix landed. This script waits
# for the main pipeline to complete, then runs just CLIP seed 42 and
# recomputes the 8-class G1.3 verdict with the full 3 seeds of CLIP data.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/cross_arch/clip_seed42_retry.log
{
  # Wait for main pipeline to finish (look for STEP 4 completion line)
  until grep -q "PIPELINE COMPLETE" results/cross_arch/batch2_pipeline.log 2>/dev/null; do
    sleep 20
  done
  echo "=== main pipeline done; retrying CLIP seed 42 ==="
  sleep 5
  python code/genome_cross_arch.py -n 2000 --c4 --seed 42 \
    --systems clip-vit-b32-image

  echo ""
  echo "=== Recomputing 8-class G1.3 verdict with full CLIP data ==="
  python - <<'PY'
import json, sys
sys.path.insert(0, "code")
from genome_stim_resample import _group_rows, evaluate_g13_sensitivity
rows = []
for seed in (42, 123, 456):
    with open(f"results/cross_arch/atlas_rows_n2000_c4_seed{seed}.json") as f:
        rows.extend(json.load(f)["rows"])
    for new_sys in ("deepseek-r1-distill-qwen-1.5b","bert-base-uncased",
                    "minilm-l6-contrastive","clip-vit-b32-image"):
        p = f"results/cross_arch/atlas_rows_n2000_c4_seed{seed}_only_{new_sys}.json"
        try:
            with open(p) as f:
                rows.extend(json.load(f)["rows"])
        except FileNotFoundError:
            print(f"WARN: still missing {p}")
grouped = _group_rows(rows)
verdicts = evaluate_g13_sensitivity(grouped)
out_path = "results/gate1/stim_resample_n2000_8class_full.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "experiment_id": "genome_011_8class_batch2_full",
        "seeds": [42, 123, 456],
        "n_sentences": 2000,
        "g13_verdicts": {f"{s}||{p}||{e}": v for (s,p,e), v in verdicts.items()},
    }, f, indent=2, default=float)

systems = sorted({k[0] for k in verdicts.keys()})
print(f"=== 8-CLASS G1.3 VERDICT (full CLIP data) ===")
print(f"systems: {systems}")
print()
print("kNN-k10 per system at delta=0.05 / 0.10 / 0.20:")
for (s,p,e), v in sorted(verdicts.items()):
    if p == "knn_clustering" and e == "knn_k10":
        sw = v["sensitivity_sweep"]
        print(f"  {s:30s}  {sw['delta_0.05']['status']:5s}/{sw['delta_0.1']['status']:5s}/{sw['delta_0.2']['status']:5s}  "
              f"max_stat={v['max_stat']:.4f}  mgn@0.10={0.10*v['median_abs_f']:.4f}")
PY
  echo ""
  echo "=== CLIP RETRY COMPLETE ==="
} >> "$LOG" 2>&1
