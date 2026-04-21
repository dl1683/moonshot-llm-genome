"""Aggregate biology kNN-k10 measurements across sessions.

Parses results/gate2/biology_10session_v2.log for per-session kNN values,
reports:
  - per-session table
  - mean +/- SE across sessions
  - pass count at delta=0.10 vs DINOv2 reference band [0.30, 0.35]
  - pass count at delta=0.05
  - cross-session CV
  - Gate-2 G2.5 prereg criterion (>=60% pass at delta=0.10) status
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_LOG = _ROOT / "results/gate2/biology_10session_v2.log"
_DINOv2_REF = (0.30, 0.35)


def parse_sessions(log_path: Path) -> list[dict]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # One SESSION block per header
    blocks = re.split(r"=== SESSION (\d+) ===", text)
    # blocks = [pre, idx0, body0, idx1, body1, ...]
    sessions = []
    for i in range(1, len(blocks), 2):
        idx = int(blocks[i])
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        m = re.search(
            r"biology kNN-k10 = (\d+\.\d+) \(SE (\d+\.\d+), n=(\d+) stimuli, d=(\d+) neurons\)",
            body,
        )
        if m:
            sessions.append({
                "session_idx": idx,
                "kNN_k10": float(m.group(1)),
                "SE": float(m.group(2)),
                "n_stimuli": int(m.group(3)),
                "n_neurons": int(m.group(4)),
                "status": "completed",
            })
        else:
            sessions.append({
                "session_idx": idx,
                "status": "pending_or_failed",
            })
    return sessions


def main():
    sessions = parse_sessions(_LOG)
    completed = [s for s in sessions if s["status"] == "completed"]
    print(f"Found {len(sessions)} session blocks ({len(completed)} completed).")
    print(f"\n{'idx':>4s} {'kNN-k10':>10s} {'SE':>8s} {'in_DINOv2_±0.10':>18s} {'in_DINOv2_±0.05':>18s}")
    vals = []
    passes_10 = 0
    passes_05 = 0
    for s in completed:
        v = s["kNN_k10"]
        vals.append(v)
        lo10, hi10 = _DINOv2_REF[0] - 0.10, _DINOv2_REF[1] + 0.10
        lo05, hi05 = _DINOv2_REF[0] - 0.05, _DINOv2_REF[1] + 0.05
        p10 = "YES" if lo10 <= v <= hi10 else "no"
        p05 = "YES" if lo05 <= v <= hi05 else "no"
        if p10 == "YES":
            passes_10 += 1
        if p05 == "YES":
            passes_05 += 1
        print(f"  {s['session_idx']:2d} {v:10.4f} {s['SE']:8.4f} {p10:>18s} {p05:>18s}")

    if vals:
        arr = np.array(vals)
        mean = arr.mean()
        sd = arr.std(ddof=1) if arr.size > 1 else 0.0
        se_mean = sd / np.sqrt(arr.size) if arr.size > 0 else 0.0
        pr_10 = passes_10 / len(vals) * 100
        pr_05 = passes_05 / len(vals) * 100
        print(f"\n  mean  = {mean:.4f}")
        print(f"  SD    = {sd:.4f}")
        print(f"  SE of mean = {se_mean:.4f}")
        print(f"  CV%   = {100 * sd / abs(mean):.1f}%")
        print(f"  range = [{arr.min():.4f}, {arr.max():.4f}]")
        print(f"\n  pass @ delta=0.10 (lenient): {passes_10}/{len(vals)} = {pr_10:.0f}%")
        print(f"  pass @ delta=0.05 (strict):  {passes_05}/{len(vals)} = {pr_05:.0f}%")
        print(f"\n  G2.5 prereg criterion: >=60% at delta=0.10 "
              f"{'PASS' if pr_10 >= 60 else 'NOT-YET-MET'} (at {len(vals)} sessions)")

        out = {
            "n_sessions_completed": len(vals),
            "n_sessions_total": len(sessions),
            "DINOv2_reference_band": list(_DINOv2_REF),
            "per_session": [{"idx": s["session_idx"],
                             "kNN_k10": s["kNN_k10"], "SE": s["SE"]}
                            for s in completed],
            "summary": {
                "mean": float(mean),
                "SD": float(sd),
                "SE_of_mean": float(se_mean),
                "CV_pct": float(100 * sd / abs(mean)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "pass_rate_delta_0.10": pr_10,
                "pass_rate_delta_0.05": pr_05,
                "g2_5_criterion_60pct_at_delta_0.10": bool(pr_10 >= 60),
            },
        }
        out_path = _ROOT / "results/gate2/biology_10session_aggregate.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
