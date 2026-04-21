#!/usr/bin/env bash
# Auto-chained pipeline: waits for Falcon seed-456 to finish, computes the
# 3-seed n=4000 Falcon verdict (likely clean pass → 5/5 Level-1 threshold
# for kNN-k10), then immediately runs the Batch-2 encoder/contrastive
# sweep (BERT + MiniLM + CLIP-image) at n=2000 × 3 seeds, then computes
# the combined 8-class G1.3 verdict.
#
# Purpose: remove the manual "wait for heartbeat → check → launch next"
# loop. Everything runs end-to-end without supervision.
set -euo pipefail

cd "$(dirname "$0")/.."
LOG=results/cross_arch/batch2_pipeline.log

{
  echo "=== STEP 1: wait for Falcon seed-456 n=4000 ==="
  # Poll for the seed-456 output file.
  until [ -f results/cross_arch/atlas_rows_n4000_c4_seed456_only_falcon-h1-0.5b.json ]; do
    sleep 30
  done
  echo "Falcon seed-456 file present. Waiting 10s to ensure write is complete..."
  sleep 10

  echo ""
  echo "=== STEP 2: compute 3-seed Falcon n=4000 verdict ==="
  python - <<'PY'
import json, sys
sys.path.insert(0, "code")
from genome_stim_resample import _group_rows, evaluate_g13_sensitivity
rows = []
for seed in (42, 123, 456):
    with open(f"results/cross_arch/atlas_rows_n4000_c4_seed{seed}_only_falcon-h1-0.5b.json") as f:
        rows.extend(json.load(f)["rows"])
grouped = _group_rows(rows)
verdicts = evaluate_g13_sensitivity(grouped)
with open("results/gate1/stim_resample_n4000_seeds42_123_456_falcon.json", "w", encoding="utf-8") as f:
    json.dump({
        "experiment_id": "genome_010_falcon_n4000",
        "n_sentences": 4000,
        "seeds": [42, 123, 456],
        "g13_verdicts": {f"{s}||{p}||{e}": v for (s,p,e), v in verdicts.items()},
    }, f, indent=2, default=float)
k10 = verdicts[("falcon-h1-0.5b","knn_clustering","knn_k10")]
sw = k10["sensitivity_sweep"]
print(f"Falcon kNN-k10 n=4000 3-seed: max_stat={k10['max_stat']:.4f} margin={0.10*k10['median_abs_f']:.4f}")
print(f"  delta[0.05/0.10/0.20]: {sw['delta_0.05']['status']}/{sw['delta_0.1']['status']}/{sw['delta_0.2']['status']}")
PY

  echo ""
  echo "=== STEP 3: run Batch-2 G1.3 sweep (BERT + MiniLM + CLIP) 3 seeds ==="
  for seed in 42 123 456; do
    for sys in bert-base-uncased minilm-l6-contrastive clip-vit-b32-image; do
      echo "--- seed=$seed system=$sys ---"
      python code/genome_cross_arch.py -n 2000 --c4 --seed "$seed" --systems "$sys" || {
        echo "FAIL seed=$seed system=$sys (continuing)"
      }
    done
  done

  echo ""
  echo "=== STEP 4: compute 8-class combined G1.3 verdict ==="
  python - <<'PY'
import json, sys
sys.path.insert(0, "code")
from genome_stim_resample import _group_rows, evaluate_g13_sensitivity
rows = []
for seed in (42, 123, 456):
    with open(f"results/cross_arch/atlas_rows_n2000_c4_seed{seed}.json") as f:
        rows.extend(json.load(f)["rows"])
    # Add DeepSeek
    p = f"results/cross_arch/atlas_rows_n2000_c4_seed{seed}_only_deepseek-r1-distill-qwen-1.5b.json"
    with open(p) as f:
        rows.extend(json.load(f)["rows"])
    # Add Batch-2 systems
    for new_sys in ("bert-base-uncased","minilm-l6-contrastive","clip-vit-b32-image"):
        p = f"results/cross_arch/atlas_rows_n2000_c4_seed{seed}_only_{new_sys}.json"
        try:
            with open(p) as f:
                rows.extend(json.load(f)["rows"])
        except FileNotFoundError:
            print(f"WARN: missing {p} — {new_sys} at seed {seed} did not produce output")
grouped = _group_rows(rows)
verdicts = evaluate_g13_sensitivity(grouped)
with open("results/gate1/stim_resample_n2000_8class.json", "w", encoding="utf-8") as f:
    json.dump({
        "experiment_id": "genome_011_8class_batch2",
        "n_sentences": 2000,
        "seeds": [42, 123, 456],
        "g13_verdicts": {f"{s}||{p}||{e}": v for (s,p,e), v in verdicts.items()},
    }, f, indent=2, default=float)

systems = sorted({k[0] for k in verdicts.keys()})
print(f"=== 8-CLASS G1.3 VERDICT ===")
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
  echo "=== PIPELINE COMPLETE ==="
} >> "$LOG" 2>&1
