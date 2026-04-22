"""Weight-space mid-layer transplant: one trained block into otherwise-untrained model.

If a single trained transformer block inserted into an untrained stack produces
ANY capability recovery, capability is partially layer-localized. If null,
capability is distributed across many layers' weights simultaneously.

Protocol (DeepSeek-R1-Distill-Qwen-1.5B):
  1. Load trained model's state_dict; extract all weights whose keys match
     "model.layers.<mid>." prefix — this is the mid block's parameters.
  2. Load untrained twin. Overwrite its mid block's parameters with the
     trained mid block's parameters. All other layers stay untrained.
  3. Measure NLL + mid-depth geometry on 500 C4 stimuli.

Baselines for comparison:
  - untrained (all random)         -> NLL ~ 12.2
  - trained (all trained)          -> NLL ~ 4.3
  - mid-block transplant only      -> ?

Expected outcomes:
  - NLL ~= untrained -> capability not localized in mid block alone
  - NLL substantially < untrained -> partial localization; ledger as positive finding
  - NLL approaches trained -> strong localization; paradigm-shift direction
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
from genome_geometry_transfusion import measure_nll, fit_power_law  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def main():
    hf_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    sk = "deepseek-r1-distill-qwen-1.5b"
    seed = 42
    n = 500
    t0 = time.time()

    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} stim")

    # Load trained for mid-block weights
    print(f"[{time.time()-t0:.1f}s] loading TRAINED to extract mid block weights...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_t.n_hidden_layers()
    mid = n_layers // 2
    prefix = f"model.layers.{mid}."
    mid_block_weights = {
        k: v.detach().clone() for k, v in sys_t.model.state_dict().items()
        if k.startswith(prefix)
    }
    print(f"[{time.time()-t0:.1f}s] captured {len(mid_block_weights)} mid-block parameters")
    nll_trained, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    print(f"  TRAINED full NLL = {nll_trained:.4f}")
    sys_t.unload(); torch.cuda.empty_cache()

    # Load untrained
    print(f"[{time.time()-t0:.1f}s] loading UNTRAINED...")
    sys_u = load_system(hf_id, quant="fp16", untrained=True, device="cuda")
    nll_untrained, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
    print(f"  UNTRAINED NLL = {nll_untrained:.4f}")

    # Transplant
    print(f"[{time.time()-t0:.1f}s] transplanting mid block (layer {mid})...")
    sd = sys_u.model.state_dict()
    n_overwritten = 0
    for k in list(sd.keys()):
        if k.startswith(prefix) and k in mid_block_weights:
            if sd[k].shape == mid_block_weights[k].shape:
                sd[k] = mid_block_weights[k].to(sd[k].dtype)
                n_overwritten += 1
    sys_u.model.load_state_dict(sd, strict=False)
    print(f"  overwrote {n_overwritten} parameters")

    nll_trans, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
    traj_t = extract_trajectory(
        model=sys_u.model, tokenizer=sys_u.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=sk + "_mid_transplant", class_id=2,
        quantization="fp16",
        stimulus_version=f"mid_transplant.seed{seed}.n{n}", seed=seed,
        batch_size=16, max_length=256,
    )
    X_t = traj_t.layers[0].X.astype(np.float32)
    Cs_t = [float(knn_clustering_coefficient(X_t, k=k).value) for k in K_GRID]
    p_t, _, _ = fit_power_law(K_GRID, Cs_t)
    rd_t = rate_distortion_dim(X_t)
    c_t = p_t * rd_t["d_rd"]
    sys_u.unload(); torch.cuda.empty_cache()

    drop_rel = (nll_untrained - nll_trans) / nll_untrained
    fraction_toward_trained = (nll_untrained - nll_trans) / (nll_untrained - nll_trained) if (nll_untrained - nll_trained) > 0 else 0.0
    print(f"\n=== LAYER TRANSPLANT RESULT ===")
    print(f"  trained:        NLL={nll_trained:.3f}")
    print(f"  untrained:      NLL={nll_untrained:.3f}")
    print(f"  mid-transplant: NLL={nll_trans:.3f}  p={p_t:.3f}  d_rd={rd_t['d_rd']:.2f}  c={c_t:.2f}")
    print(f"  NLL rel drop (untrained->transplant) = {100*drop_rel:+.1f}%")
    print(f"  fraction of full gap closed = {fraction_toward_trained:.2f}")

    if fraction_toward_trained > 0.20:
        verdict = "LAYER_LOCALIZED_partial — mid block transplant closes >=20 percent of capability gap"
    elif fraction_toward_trained > 0.05:
        verdict = "WEAK_LAYER_SIGNAL — mid block transplant closes 5-20 percent"
    else:
        verdict = "NOT_LAYER_LOCALIZED — mid block transplant closes <5 percent, capability distributed"
    print(f"  verdict: {verdict}")

    out = {"purpose": "Mid-layer weight transplant — is capability layer-localized?",
           "trained_NLL": nll_trained, "untrained_NLL": nll_untrained,
           "transplant_NLL": nll_trans, "transplant_p": p_t,
           "transplant_d_rd": rd_t["d_rd"], "transplant_c": c_t,
           "fraction_gap_closed": fraction_toward_trained,
           "verdict": verdict, "n_params_overwritten": n_overwritten}
    out_path = _ROOT / "results/gate2/layer_transplant.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
