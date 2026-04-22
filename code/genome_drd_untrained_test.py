"""Is d_rd itself the training-converged quantity? Test on untrained twins.

genome_036 showed p = 2/d_rd on 3 trained text systems with d_rd in a narrow
band ~11-14. The cleaner theoretical story would be: training converges d_rd
to a shared target, and the power-law p = 2/d_rd is a mechanical consequence.

This test computes d_rd on the random-init twins we already know (from
genome_028) to have wildly different p values. If untrained d_rd is also
wildly different (matching the untrained p = 2/d_rd), that supports the
'd_rd is the primary converged quantity' story.

If untrained d_rd is NEAR the trained d_rd band but untrained p is wildly
different, then d_rd is NOT the primary quantity and the relation breaks
on random init.
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
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def run_one(hf_id: str, system_key: str, untrained: bool):
    tag = "UNTRAINED" if untrained else "TRAINED"
    print(f"\n=== {system_key} [{tag}] ===")
    t0 = time.time()
    sys_obj = load_system(hf_id, quant="fp16", untrained=untrained, device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2

    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 1000:
            break

    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=system_key, class_id=0,
        quantization="fp16",
        stimulus_version="c4_clean.v1.seed42.n1000",
        seed=42, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p_emp, c0_emp, r2_ck = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    d_rd = rd["d_rd"]
    p_pred = 2.0 / d_rd if d_rd > 0 else float("nan")
    rel_err = abs(p_pred - p_emp) / abs(p_emp) if abs(p_emp) > 1e-6 else float("inf")

    print(f"  p_emp={p_emp:.3f}  Ck_R2={r2_ck:.3f}  d_rd={d_rd:.2f}  p_pred=2/d_rd={p_pred:.3f}  rel_err={rel_err:.3f}")
    sys_obj.unload()
    torch.cuda.empty_cache()
    return {"system_key": system_key, "untrained": untrained,
            "p_emp": p_emp, "Ck_R2": r2_ck, "d_rd": d_rd,
            "p_pred": p_pred, "rel_err": rel_err, "n": int(X.shape[0])}


def main():
    # Only RWKV and DeepSeek — Qwen3 untrained doesn't produce a power law
    # (R^2 < 0.04 in genome_028/029), so p_emp is meaningless.
    targets = [
        ("RWKV/rwkv-4-169m-pile", "rwkv-4-169m"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
         "deepseek-r1-distill-qwen-1.5b"),
    ]
    results = []
    for hf_id, sk in targets:
        for untrained in (False, True):
            try:
                results.append(run_one(hf_id, sk, untrained=untrained))
            except Exception as e:
                import traceback; traceback.print_exc()
                results.append({"system_key": sk, "untrained": untrained,
                                "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY: d_rd trained vs untrained ===")
    print(f"{'system':>32s} {'tag':>10s} {'p_emp':>8s} {'d_rd':>8s} {'p=2/d_rd':>10s} {'rel_err':>8s}")
    for r in results:
        if "error" in r:
            print(f"  {r['system_key']:>32s}  ERROR")
            continue
        tag = "untrained" if r["untrained"] else "trained"
        print(f"  {r['system_key']:>32s} {tag:>10s} {r['p_emp']:8.3f} "
              f"{r['d_rd']:8.2f} {r['p_pred']:10.3f} {r['rel_err']:8.3f}")

    # Per-system deltas
    per_system = {}
    for r in results:
        if "error" in r:
            continue
        per_system.setdefault(r["system_key"], {})[r["untrained"]] = r
    print("\nTrained vs untrained d_rd shift:")
    for sk, pair in per_system.items():
        if True in pair and False in pair:
            t = pair[False]; u = pair[True]
            print(f"  {sk}: d_rd trained={t['d_rd']:.2f}  untrained={u['d_rd']:.2f}  "
                  f"delta={u['d_rd']-t['d_rd']:+.2f}")
            print(f"  {sk}: p    trained={t['p_emp']:.3f}  untrained={u['p_emp']:.3f}  "
                  f"rule rel_err untrained={u['rel_err']:.3f}")

    out = {"purpose": "d_rd trained vs untrained — is d_rd the converged quantity?",
           "per_run": results}
    out_path = _ROOT / "results/gate2/drd_untrained_test.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
