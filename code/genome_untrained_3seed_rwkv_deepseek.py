"""Multi-seed untrained robustness for RWKV + DeepSeek.

genome_028 showed random-init RWKV at torch_seed=default gives p=0.355 and
random-init DeepSeek gives p=0.192. Qwen3 already multi-seed-verified
(genome_029). This extends multi-seed to the two systems that DO produce
a power law on random init — check that their exponents (very different
from each other and from trained 0.17) are seed-stable.

3 torch seeds × 2 systems × n=1000 C4-clean seed-42 stimuli.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
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


def run_one(hf_id: str, system_key: str, torch_seed: int, class_id: int = 1):
    print(f"\n=== {system_key} torch_seed={torch_seed} ===")
    t0 = time.time()
    torch.manual_seed(torch_seed)
    config = AutoConfig.from_pretrained(hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    # RWKV needs fp32 init on CPU to avoid geqrf_cpu+Half
    if "rwkv" in hf_id.lower():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
        model = model.to(dtype=torch.float16, device="cuda").eval()
    else:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        model = model.to("cuda").eval()
    print(f"[{time.time()-t0:.1f}s] model ready")

    n_layers = (getattr(config, "num_hidden_layers", None)
                or getattr(config, "n_layer", None))
    mid = n_layers // 2

    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 1000:
            break

    traj = extract_trajectory(
        model=model, tokenizer=tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=system_key, class_id=class_id,
        quantization="fp16",
        stimulus_version="c4_clean.v1.seed42.n1000",
        seed=42, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, c0, r2 = fit_power_law(K_GRID, Cs)
    print(f"  p={p:.3f}  c_0={c0:.3f}  R^2={r2:.4f}")
    del model
    torch.cuda.empty_cache()
    return {
        "hf_id": hf_id, "system_key": system_key, "torch_seed": torch_seed,
        "k_grid": K_GRID, "C_values": Cs,
        "p_slope": p, "c_0": c0, "R2": r2,
        "elapsed_s": time.time() - t0,
    }


def main():
    targets = [
        ("RWKV/rwkv-4-169m-pile", "rwkv-4-169m", 3),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
         "deepseek-r1-distill-qwen-1.5b", 2),
    ]
    results = []
    for hf_id, sk, cid in targets:
        for ts in (42, 123, 456):
            results.append(run_one(hf_id, sk, ts, class_id=cid))

    print("\n=== SUMMARY: UNTRAINED 3-seed robustness ===")
    print(f"{'system':>32s} {'seed':>6s} {'p':>8s} {'c_0':>8s} {'R^2':>8s}")
    per_system = {}
    for r in results:
        per_system.setdefault(r["system_key"], []).append(r)
        print(f"  {r['system_key']:>32s} {r['torch_seed']:6d} "
              f"{r['p_slope']:8.3f} {r['c_0']:8.3f} {r['R2']:8.4f}")

    print("\nPer-system stability:")
    summary = {}
    for sk, rs in per_system.items():
        ps = np.array([r["p_slope"] for r in rs])
        r2s = np.array([r["R2"] for r in rs])
        print(f"  {sk}: p mean={ps.mean():.3f} std={ps.std(ddof=1):.3f} "
              f"range=[{ps.min():.3f}, {ps.max():.3f}]  "
              f"R^2 mean={r2s.mean():.3f}")
        summary[sk] = {
            "p_mean": float(ps.mean()),
            "p_std": float(ps.std(ddof=1)),
            "p_min": float(ps.min()),
            "p_max": float(ps.max()),
            "R2_mean": float(r2s.mean()),
        }

    out = {"purpose": "3-seed robustness of RWKV + DeepSeek untrained power law",
           "per_run": results,
           "per_system_summary": summary}
    out_path = _ROOT / "results/gate2/untrained_3seed_rwkv_deepseek.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
