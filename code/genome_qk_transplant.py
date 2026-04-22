"""Attention Q/K-only transplant: test if capability lives in attention ROUTING.

Orthogonal-compiler intervention per strategic verdict 2026-04-22-0047
mandate ("every cycle: one derivation + one prediction + one Compiler
intervention"). All 5 previously-tested transfusion operators are null:

  1. covariance whiten+recolor    (genome_042) — 2nd-moment
  2. K-means codebook snap        (genome_043) — piecewise-constant
  3. PCA basis projection         (genome_046) — linear projection
  4. d_rd aux-loss regularizer    (genome_048) — from-scratch training
  5. Single-layer weight transpl. (genome_049) — all weights, one layer

genome_056 / genome_057 localize c to training-specific JOINT inter-dim
structure with a power-law singular-spectrum signature (alpha=0.86).
Attention patterns ARE quintessential joint structure: Q @ K^T defines
which dimensions at which positions talk to which other dimensions at
other positions. None of the 5 prior ops target Q/K directly.

This probe grafts Q and K projection matrices (ONLY) from TRAINED Qwen3
into the UNTRAINED twin at every attention layer. V, O, MLP, embedding,
layer-norm all stay at untrained init. Measures NLL + mid-depth c.

PREDICTIONS:
  - If capability lives in attention routing (joint structure hypothesis):
    NLL drops substantially toward trained; c moves toward trained 1.89.
  - If capability lives in value processing / MLP: QK transplant is null;
    capability requires V, O, and MLP transfer alongside Q/K.
  - If capability requires globally coordinated Q/K/V/O/MLP: still null.

CONTRAST probes (sanity): V-only transplant and MLP-only transplant.
If ALL are null: capability is a joint weight-structure property; if one
subset is substantially non-null, capability localizes there.
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


PATTERNS = {
    "qk_only": ("self_attn.q_proj", "self_attn.k_proj"),
    "v_only": ("self_attn.v_proj",),
    "o_only": ("self_attn.o_proj",),
    "attn_all": ("self_attn.q_proj", "self_attn.k_proj",
                 "self_attn.v_proj", "self_attn.o_proj"),
    "mlp_only": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
}


def capture_matching(state_dict, substrings):
    return {k: v.detach().clone() for k, v in state_dict.items()
            if any(sub in k for sub in substrings)}


def graft(target_model, trained_weights):
    sd = target_model.state_dict()
    n = 0
    for k in list(sd.keys()):
        if k in trained_weights and sd[k].shape == trained_weights[k].shape:
            sd[k] = trained_weights[k].to(sd[k].dtype)
            n += 1
    target_model.load_state_dict(sd, strict=False)
    return n


def measure_intervention(sk, class_id, hf_id, sents, seed, sd_trained,
                          sd_untrained_init, pattern_name):
    substrings = PATTERNS[pattern_name]
    # Build fresh untrained, then graft the chosen subset.
    print(f"\n-- pattern={pattern_name}  substrings={substrings}")
    sys_u = load_system(hf_id, quant="fp16", untrained=True, device="cuda")
    # Reset to fresh random init (reload known state_dict)
    sys_u.model.load_state_dict(sd_untrained_init, strict=True)
    trained_subset = capture_matching(sd_trained, substrings)
    n_over = graft(sys_u.model, trained_subset)
    print(f"  grafted {n_over} parameters (out of {len(trained_subset)} available)")
    nll, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
    mid = sys_u.n_hidden_layers() // 2
    traj = extract_trajectory(
        model=sys_u.model, tokenizer=sys_u.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=f"{sk}_graft_{pattern_name}",
        class_id=class_id, quantization="fp16",
        stimulus_version=f"qk_transplant.{pattern_name}.seed{seed}",
        seed=seed, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, _, r2 = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    sys_u.unload()
    torch.cuda.empty_cache()
    return {"pattern": pattern_name, "n_over": n_over, "nll": float(nll),
            "p": float(p), "d_rd": float(rd["d_rd"]), "c": float(c),
            "Ck_R2": float(r2)}


def main():
    hf_id = "Qwen/Qwen3-0.6B"
    sk = "qwen3-0.6b"
    seed = 42
    n = 500
    t0 = time.time()

    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} stim")

    # Capture trained state_dict
    print(f"[{time.time()-t0:.1f}s] loading TRAINED to capture state_dict...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd_trained = {k: v.detach().clone() for k, v in sys_t.model.state_dict().items()}
    nll_trained, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    print(f"  TRAINED NLL = {nll_trained:.3f}")
    sys_t.unload(); torch.cuda.empty_cache()

    # Capture untrained-init state_dict (pinned so every graft starts from
    # the same random init — reload_state_dict call is the reset hook).
    print(f"[{time.time()-t0:.1f}s] loading UNTRAINED to capture init state_dict...")
    sys_u = load_system(hf_id, quant="fp16", untrained=True, device="cuda")
    sd_untrained_init = {k: v.detach().clone()
                         for k, v in sys_u.model.state_dict().items()}
    nll_untrained, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
    print(f"  UNTRAINED NLL = {nll_untrained:.3f}")
    sys_u.unload(); torch.cuda.empty_cache()

    # Run each intervention fresh (re-loading untrained model each time keeps
    # graft isolated — one graft does not leak into the next).
    results = []
    for pattern in ["qk_only", "v_only", "o_only", "attn_all", "mlp_only"]:
        try:
            r = measure_intervention(sk, 1, hf_id, sents, seed,
                                     sd_trained, sd_untrained_init, pattern)
            frac = ((nll_untrained - r["nll"])
                    / max(nll_untrained - nll_trained, 1e-6))
            r["fraction_gap_closed"] = float(frac)
            print(f"  {pattern}: NLL={r['nll']:.3f}  c={r['c']:.2f}  "
                  f"fraction_gap_closed={frac:+.3f}")
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"pattern": pattern, "error": str(e)})

    # Verdict
    qk = next((r for r in results if r.get("pattern") == "qk_only"
               and "error" not in r), None)
    if qk is not None:
        f = qk["fraction_gap_closed"]
        if f > 0.20:
            verdict = ("ROUTING_CARRIES_CAPABILITY — QK-only transplant "
                       f"closes {100*f:.0f}pct of capability gap. Attention "
                       "ROUTING (joint structure) transfers capability "
                       "without value / MLP transfer.")
        elif f > 0.05:
            verdict = ("ROUTING_PARTIAL — QK-only transplant closes "
                       f"{100*f:.0f}pct of gap. Weak routing signal.")
        else:
            verdict = ("ROUTING_NULL — QK-only transplant is null; "
                       "attention routing alone does not install capability. "
                       "Joint structure must live in more than Q/K routing.")
    else:
        verdict = "QK_RUN_FAILED"
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Orthogonal-compiler probe: does capability live in attention Q/K routing?",
           "model": sk,
           "nll_trained": float(nll_trained),
           "nll_untrained": float(nll_untrained),
           "per_pattern": results,
           "verdict": verdict,
           "seed": seed, "n_stim": n}
    out_path = _ROOT / "results/gate2/qk_transplant.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
