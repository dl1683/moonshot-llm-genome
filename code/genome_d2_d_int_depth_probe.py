"""Probe: is d_2/d_int ratio depth-stable within a single architecture?

Framework-A-falsification pilot (genome_024) incidentally discovered that
d_2/d_int ~= 0.58 +/- 0.008 is cross-architecture stable (CV 1.4%) at
mid-depth across Qwen3/RWKV/DeepSeek. This probe tests whether the ratio
is also DEPTH-STABLE within a single architecture — a stronger claim.

Runs on Qwen3-0.6B at 5 depths (0.18, 0.37, 0.52, 0.66, 0.81) × seed 42
× n=1000 C4-clean. Reports d_int (TwoNN), d_2 (GP), ratio d_2/d_int
per depth.

If ratio is depth-stable (CV < 5%), it graduates to a legitimate
cross-architecture + cross-depth invariant worth its own atlas slot.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from genome_fractal_dim_probe import grassberger_procaccia_d2  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import twonn_id  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


def main():
    t0 = time.time()
    hf_id = "Qwen/Qwen3-0.6B"
    system_key = "qwen3-0.6b"

    print(f"[{time.time()-t0:.1f}s] loading {hf_id} fp16...")
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    target_fracs = [0.18, 0.37, 0.52, 0.66, 0.81]
    depth_indices = [int(round(f * (n_layers - 1))) for f in target_fracs]
    print(f"[{time.time()-t0:.1f}s] {n_layers} layers, probing depth indices {depth_indices}")

    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 1000:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} C4 stimuli")

    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=depth_indices, pooling="seq_mean",
        device="cuda", system_key=system_key, class_id=1,
        quantization="fp16",
        stimulus_version="c4_clean.v1.seed42.n1000",
        seed=42, batch_size=16, max_length=256,
    )
    print(f"[{time.time()-t0:.1f}s] extraction complete")

    rows = []
    for lyr in traj.layers:
        X = lyr.X.astype(np.float32)
        d_int = float(twonn_id(X).value)
        gp = grassberger_procaccia_d2(X)
        d_2 = float(gp["d_2"])
        ratio = d_2 / d_int if d_int > 0 else float("nan")
        rows.append({
            "depth_idx": int(lyr.k_index),
            "depth_normalized": float(lyr.k_normalized),
            "n_stimuli": int(X.shape[0]),
            "hidden_dim": int(X.shape[1]),
            "d_int_twonn": d_int,
            "d_2_gp": d_2,
            "gp_fit_R2": float(gp["R2_fit"]),
            "ratio_d2_over_dint": ratio,
        })
        print(f"  depth {lyr.k_normalized:.2f} (layer {lyr.k_index}): "
              f"d_int={d_int:.2f}  d_2={d_2:.2f}  ratio={ratio:.3f}")

    ratios = np.array([r["ratio_d2_over_dint"] for r in rows])
    summary = {
        "mean_ratio": float(ratios.mean()),
        "std_ratio": float(ratios.std(ddof=1)),
        "cv_pct": float(100 * ratios.std(ddof=1) / abs(ratios.mean())),
        "min_ratio": float(ratios.min()),
        "max_ratio": float(ratios.max()),
    }
    print(f"\n=== SUMMARY: Qwen3 d_2/d_int across 5 depths ===")
    print(f"  mean = {summary['mean_ratio']:.3f}")
    print(f"  std  = {summary['std_ratio']:.3f}")
    print(f"  CV%  = {summary['cv_pct']:.2f}%")
    print(f"  range = [{summary['min_ratio']:.3f}, {summary['max_ratio']:.3f}]")
    verdict = "DEPTH_STABLE" if summary["cv_pct"] < 5.0 else "DEPTH_VARIABLE"
    print(f"  verdict = {verdict}")

    out = {
        "purpose": "d_2/d_int ratio depth stability within Qwen3-0.6B",
        "system": system_key, "hf_id": hf_id, "seed": 42, "n": 1000,
        "target_depth_fracs": target_fracs,
        "per_depth": rows,
        "summary": summary,
        "verdict": verdict,
        "wall_clock_s": time.time() - t0,
    }
    out_path = _ROOT / "results/gate2/d2_dint_depth_qwen3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
