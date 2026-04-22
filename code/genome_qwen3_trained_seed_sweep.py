"""Trained Qwen3-0.6B at 3 stimulus-resample seeds — symmetric check on the
1.1x trained-p-spread claim in Figure 4.

Previous data (paper Table 7): Qwen3 mid-depth p=0.156 at seed 42 n=2000.
This probe runs at seeds 42/123/456 n=1000 (quick) to report per-seed p
and confirm the trained side is tight (should be CV < 5%).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def fit_power_law(ks, Cs):
    lks = np.log(np.asarray(ks, dtype=float))
    lcs = np.log(np.asarray(Cs, dtype=float))
    p, log_c0 = np.polyfit(lks, lcs, 1)
    pred = p * lks + log_c0
    ss_res = float(np.sum((lcs - pred) ** 2))
    ss_tot = float(np.sum((lcs - lcs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(p), float(np.exp(log_c0)), r2


def main():
    hf_id = "Qwen/Qwen3-0.6B"
    print(f"Loading {hf_id} trained fp16 once...")
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2

    results = []
    for stim_seed in (42, 123, 456):
        t0 = time.time()
        print(f"\n=== stim_seed={stim_seed} ===")
        sents = []
        for rec in c4_clean_v1(seed=stim_seed, n_samples=5000):
            sents.append(rec["text"])
            if len(sents) >= 1000:
                break

        traj = extract_trajectory(
            model=sys_obj.model, tokenizer=sys_obj.tokenizer,
            texts=sents, layer_indices=[mid], pooling="seq_mean",
            device="cuda", system_key="qwen3-0.6b", class_id=1,
            quantization="fp16",
            stimulus_version=f"c4_clean.v1.seed{stim_seed}.n1000",
            seed=stim_seed, batch_size=16, max_length=256,
        )
        import numpy as np
        X = traj.layers[0].X.astype(np.float32)
        Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
        p, c0, r2 = fit_power_law(K_GRID, Cs)
        print(f"  p={p:.3f}  c_0={c0:.3f}  R^2={r2:.4f}")
        results.append({"stim_seed": stim_seed, "p": p, "c_0": c0, "R2": r2,
                        "C_values": Cs, "elapsed_s": time.time() - t0})

    ps = np.array([r["p"] for r in results])
    print(f"\n=== SUMMARY: Qwen3 TRAINED 3 stim-seeds ===")
    print(f"  p = {ps.mean():.3f} +/- {ps.std(ddof=1):.3f} "
          f"(range [{ps.min():.3f}, {ps.max():.3f}], CV {100*ps.std(ddof=1)/abs(ps.mean()):.1f}%)")

    sys_obj.unload()
    out = {"system": "qwen3-0.6b", "hf_id": hf_id,
           "per_stim_seed": results,
           "p_mean": float(ps.mean()),
           "p_std": float(ps.std(ddof=1)),
           "p_cv_pct": float(100 * ps.std(ddof=1) / abs(ps.mean()))}
    out_path = _ROOT / "results/gate2/qwen3_trained_seed_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
