#!/usr/bin/env bash
# Extended k-sweep on 5 Batch-1 systems × 3 seeds at n=2000.
# Unblocks G2.3 hierarchical-fit test (prereg §8 requires k ∈ {5, 10, 20, 30}).
# New measurement covers k ∈ {3, 5, 10, 20, 30} — strict superset of prior
# per-system atlases (which had k ∈ {5, 10} only). Overwrite is safe: the
# new data enriches without loss; the original full-bestiary atlases at
# `atlas_rows_n2000_c4_seed{seed}.json` use no --systems filter and are
# untouched.
#
# On completion runs genome_hierarchical_fit.py against the enriched data.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/gate2/extended_ksweep.log
mkdir -p results/gate2
{
  echo "=== EXTENDED k-SWEEP START ==="
  for seed in 42 123 456; do
    for sys in qwen3-0.6b rwkv-4-169m deepseek-r1-distill-qwen-1.5b falcon-h1-0.5b dinov2-small; do
      echo ""
      echo "--- seed=$seed system=$sys ---"
      python code/genome_cross_arch.py -n 2000 --c4 --seed "$seed" --systems "$sys" || {
        echo "FAIL seed=$seed system=$sys (continuing)"
      }
    done
  done

  echo ""
  echo "=== HIERARCHICAL FIT ON EXTENDED k-SWEEP ==="
  python code/genome_hierarchical_fit.py --full
  echo ""
  echo "=== EXTENDED SWEEP COMPLETE ==="
} >> "$LOG" 2>&1
