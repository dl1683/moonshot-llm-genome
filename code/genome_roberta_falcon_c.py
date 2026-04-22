"""Candidate-5 extension: RoBERTa (another MLM) + Falcon-H1 (hybrid text).

genome_052 showed BERT MLM c=2.65 (outlier vs candidate-5 prediction c=2).
RoBERTa is the cleanest disambiguation: if it also lands at ~2.65, candidate-5
needs MLM-specific base (CLM_base=2, MLM_base~2.65). If ~2, BERT was an
anomaly and candidate-5 survives unchanged.

Falcon-H1-0.5B (hybrid transformer+Mamba, CLM): predicts c~=2 per the
single-modality text rule. Extends the scorecard.
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


def measure(hf_id, sk, class_id, n=1000, seed=42):
    print(f"\n=== {sk} ===")
    t0 = time.time()
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2
    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=sk, class_id=class_id,
        quantization="fp16",
        stimulus_version=f"c4_clean.v1.seed{seed}.n{n}", seed=seed,
        batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, _, _ = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    print(f"  p={p:.3f}  d_rd={rd['d_rd']:.2f}  c={c:.2f}  t={time.time()-t0:.1f}s")
    sys_obj.unload(); torch.cuda.empty_cache()
    return {"system": sk, "p": p, "d_rd": rd["d_rd"], "c": c}


def main():
    targets = [
        ("FacebookAI/roberta-base", "roberta-base", 7),
        ("tiiuae/Falcon-H1-0.5B-Instruct", "falcon-h1-0.5b", 4),
    ]
    results = []
    for hf, sk, cid in targets:
        try:
            results.append(measure(hf, sk, cid))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"system": sk, "error": str(e)})

    print("\n=== CANDIDATE-5 EXTENSION SCORECARD ===")
    print("  base_text_CLM = 2, base_text_MLM = ??  (disambiguation)")
    for r in results:
        if "error" in r:
            print(f"  {r['system']}: ERROR: {r['error']}")
            continue
        rel_err_2 = abs(r["c"] - 2.0) / 2.0
        pass_tag = "FIT to c=2" if rel_err_2 < 0.15 else f"MISS (rel_err {rel_err_2:.2f})"
        print(f"  {r['system']}: c={r['c']:.2f}  {pass_tag}")

    out = {"purpose": "Candidate-5 extension: RoBERTa + Falcon-H1",
           "per_system": results}
    out_path = _ROOT / "results/gate2/roberta_falcon_c.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
