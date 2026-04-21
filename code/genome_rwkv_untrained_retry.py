"""Workaround for RWKV-4-169M untrained init geqrf_cpu fp16 failure.

genome_untrained_power_law.py failed on RWKV untrained because RWKV-4's
from_config init calls a QR decomposition that has no fp16-on-CPU
implementation. Workaround: initialize at fp32 on CPU, then cast+move to
fp16-on-GPU before the forward pass.

Produces the missing UNTRAINED row for RWKV and appends it to
results/gate2/untrained_power_law.json.
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


def main():
    t0 = time.time()
    hf_id = "RWKV/rwkv-4-169m-pile"
    print(f"[{time.time()-t0:.1f}s] Loading RWKV-4-169M config + tokenizer...")
    config = AutoConfig.from_pretrained(hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    print(f"[{time.time()-t0:.1f}s] Init random-weight RWKV at fp32 on CPU...")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    print(f"[{time.time()-t0:.1f}s] cast to fp16 and move to cuda...")
    model = model.to(dtype=torch.float16, device="cuda").eval()
    print(f"[{time.time()-t0:.1f}s] model ready")

    n_layers = getattr(config, "num_hidden_layers", None) or config.n_layer
    mid = n_layers // 2

    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 1000:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} C4 stimuli")

    traj = extract_trajectory(
        model=model, tokenizer=tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key="rwkv-4-169m", class_id=3,
        quantization="fp16",
        stimulus_version="c4_clean.v1.seed42.n1000",
        seed=42, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    print(f"[{time.time()-t0:.1f}s] cloud {X.shape}")

    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, c0, r2 = fit_power_law(K_GRID, Cs)
    print(f"  UNTRAINED RWKV: p={p:.3f}  c_0={c0:.3f}  R^2={r2:.4f}")
    print(f"  C values: {[round(c, 3) for c in Cs]}")

    # Append to existing untrained_power_law.json
    out_path = _ROOT / "results/gate2/untrained_power_law.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else {"per_cell": []}
    row = {
        "system_key": "rwkv-4-169m", "hf_id": hf_id, "untrained": True,
        "depth_idx": mid, "n_stimuli": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "k_grid": K_GRID, "C_values": Cs,
        "p_slope": p, "c_0": c0, "R2": r2,
        "elapsed_s": time.time() - t0,
        "workaround_note": "initialized at fp32 on CPU, then cast to fp16 on GPU to avoid geqrf_cpu+half NotImplementedError during RWKV from_config",
    }
    # Replace any existing untrained RWKV row with error
    cells = [c for c in existing.get("per_cell", [])
             if not (c.get("system_key") == "rwkv-4-169m" and c.get("untrained") is True)]
    cells.append(row)
    existing["per_cell"] = cells
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"[{time.time()-t0:.1f}s] merged into {out_path}")


if __name__ == "__main__":
    main()
