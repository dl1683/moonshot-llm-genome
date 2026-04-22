"""Codex Move 2: Procrustes-aligned weight transplant.

Per Codex strategic verdict 2026-04-22-T43h (DeepMind-publishability 9.1/10).

Hypothesis. The 10 null forward-transfer ops (genome_042..059) all transplant
trained weights into an untrained twin whose hidden-state basis is a
random rotation away from the trained basis. Even when the trained weights
are copied byte-for-byte, upstream untrained weights produce activations in
a rotated frame that the transplanted block cannot interpret. Maybe
capability appears to be non-transferable because the block BOUNDARIES are
misaligned, not because the weights themselves carry hidden joint-structure
we cannot capture.

Test. At each layer boundary where we transplant, first FIT the orthogonal
rotation R such that R @ X_untrained_l ≈ X_trained_l on a common C4 stimulus
bank (orthogonal Procrustes). Conjugate the transplanted weights through R:
the transplanted down-weights absorb R on the input side, up-weights of the
next block absorb R^T on the output side. This keeps residual stream
consistency.

Conditions. Start with 1 trained mid-block (genome_049 was null at 0.00
fraction gap closed). Then 2 trained blocks (mid-1, mid). If capability
emerges after alignment that didn't without alignment, basis-mismatch WAS
the blocker.

Kill condition (Codex): if best fraction_gap_closed < 0.05 after 2-block
aligned transplant (with alignment R² >= 0.8), conclude "not just basis
mismatch; capability is genuinely distributed/joint". This is a strong
falsification of the "maybe we just need the right rotation" hope.

Prereg: this script implements the experiment; result commit locks the
first-ever Procrustes-aligned compiler test.
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
from genome_geometry_transfusion import measure_nll  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


def fit_procrustes(X_src, X_tgt):
    """Fit R such that X_src @ R approximately equals X_tgt in Frobenius norm.

    Orthogonal Procrustes: R = U @ V^T from SVD of X_src^T @ X_tgt.
    Returns R of shape (h_src, h_tgt) -- when h_src==h_tgt this is square
    orthogonal.
    """
    # center
    mu_s = X_src.mean(axis=0, keepdims=True)
    mu_t = X_tgt.mean(axis=0, keepdims=True)
    A = X_src - mu_s
    B = X_tgt - mu_t
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    # quality: R^2 of aligned vs target
    A_aligned = A @ R
    ss_tot = (B ** 2).sum()
    ss_res = ((A_aligned - B) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return R, float(r2)


def extract_layer_boundary_states(sys_obj, sents, layer_idx, device="cuda"):
    """Return (n, h) mean-pooled activations AT the output of layer_idx."""
    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[layer_idx], pooling="seq_mean",
        device=device, system_key="align_probe", class_id=1,
        quantization="fp16",
        stimulus_version="align_probe",
        seed=42, batch_size=16, max_length=256,
    )
    return traj.layers[0].X.astype(np.float32)


def main():
    hf_id = "Qwen/Qwen3-0.6B"
    seed = 42
    n = 500
    t0 = time.time()

    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} stim")

    # Step 1: extract TRAINED activations at (mid-1) and (mid) output (block boundary).
    print(f"[{time.time()-t0:.1f}s] extracting TRAINED boundary activations...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_t.n_hidden_layers()
    mid = n_layers // 2
    assert mid - 1 >= 0
    X_t_pre  = extract_layer_boundary_states(sys_t, sents, mid - 1)
    X_t_mid  = extract_layer_boundary_states(sys_t, sents, mid)
    sd_trained = {k: v.detach().clone() for k, v in sys_t.model.state_dict().items()}
    nll_trained, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    print(f"  TRAINED NLL={nll_trained:.3f}, X_t_pre {X_t_pre.shape}, X_t_mid {X_t_mid.shape}")
    sys_t.unload(); torch.cuda.empty_cache()

    # Step 2: capture UNTRAINED init state_dict + its boundary activations.
    print(f"[{time.time()-t0:.1f}s] extracting UNTRAINED boundary activations...")
    sys_u = load_system(hf_id, quant="fp16", untrained=True, device="cuda")
    sd_untrained = {k: v.detach().clone() for k, v in sys_u.model.state_dict().items()}
    X_u_pre  = extract_layer_boundary_states(sys_u, sents, mid - 1)
    X_u_mid  = extract_layer_boundary_states(sys_u, sents, mid)
    nll_untrained, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
    print(f"  UNTRAINED NLL={nll_untrained:.3f}")
    sys_u.unload(); torch.cuda.empty_cache()

    # Step 3: Fit orthogonal Procrustes at the pre-mid boundary
    # R_pre takes untrained pre-mid state -> trained pre-mid state
    R_pre, r2_pre = fit_procrustes(X_u_pre, X_t_pre)
    R_post, r2_post = fit_procrustes(X_u_mid, X_t_mid)
    print(f"[{time.time()-t0:.1f}s] Procrustes R_pre R2={r2_pre:.3f}, R_post R2={r2_post:.3f}")

    results = []

    def run_intervention(condition, graft_layers, use_align):
        print(f"\n-- condition={condition}  graft_layers={graft_layers}  align={use_align}")
        sys_v = load_system(hf_id, quant="fp16", untrained=True, device="cuda")
        sys_v.model.load_state_dict(sd_untrained, strict=True)
        sd = sys_v.model.state_dict()
        n_over = 0
        R_pre_t = torch.from_numpy(R_pre).to(dtype=torch.float32)
        R_post_t = torch.from_numpy(R_post).to(dtype=torch.float32)
        for li in graft_layers:
            prefix = f"model.layers.{li}."
            for k in list(sd.keys()):
                if k.startswith(prefix) and k in sd_trained:
                    W = sd_trained[k].to(dtype=torch.float32).cpu().clone()
                    if use_align:
                        # For input-projecting weights (q/k/v/gate/up), apply
                        # R_pre on the RIGHT (input basis). For down/out projs,
                        # apply R_post^T on the LEFT (output basis).
                        if any(s in k for s in ["q_proj", "k_proj", "v_proj",
                                                 "gate_proj", "up_proj"]):
                            if W.ndim == 2 and W.shape[1] == R_pre_t.shape[0]:
                                W = W @ R_pre_t
                        elif any(s in k for s in ["o_proj", "down_proj"]):
                            if W.ndim == 2 and W.shape[0] == R_post_t.shape[0]:
                                W = R_post_t.T @ W
                    if sd[k].shape == W.shape:
                        sd[k] = W.to(dtype=sd[k].dtype)
                        n_over += 1
        sys_v.model.load_state_dict(sd, strict=False)
        print(f"  grafted {n_over} tensors")
        nll, _ = measure_nll(sys_v.model, sys_v.tokenizer, sents)
        fgc = (nll_untrained - nll) / max(nll_untrained - nll_trained, 1e-6)
        print(f"  {condition}: NLL={nll:.3f}  fraction_gap_closed={fgc:+.3f}")
        sys_v.unload(); torch.cuda.empty_cache()
        return {"condition": condition, "graft_layers": graft_layers,
                "use_align": use_align, "nll": float(nll),
                "fraction_gap_closed": float(fgc), "n_tensors": n_over}

    # Control: single-block transplant without alignment (genome_049 replica).
    results.append(run_intervention("1block_noalign", [mid], use_align=False))
    # Main: single-block transplant WITH Procrustes alignment.
    results.append(run_intervention("1block_align", [mid], use_align=True))
    # Extended: two-block aligned transplant.
    results.append(run_intervention("2block_align", [mid - 1, mid], use_align=True))

    print("\n=== PROCRUSTES-ALIGNED TRANSPLANT ===")
    print(f"  trained NLL={nll_trained:.3f}, untrained NLL={nll_untrained:.3f}")
    print(f"  Procrustes alignment R2: pre={r2_pre:.3f}, post={r2_post:.3f}")
    for r in results:
        print(f"  {r['condition']}: NLL={r['nll']:.3f}  "
              f"gap_closed={r['fraction_gap_closed']:+.3f}")

    best = max(r["fraction_gap_closed"] for r in results)
    if r2_pre >= 0.8 and r2_post >= 0.8:
        if best >= 0.05:
            verdict = (f"BASIS_ALIGNMENT_HELPS — best gap_closed={best:.3f} "
                       "after alignment. Basis mismatch was at least partially "
                       "blocking weight transfer.")
        else:
            verdict = (f"CAPABILITY_GENUINELY_DISTRIBUTED — alignment R2 "
                       f"{r2_pre:.2f}/{r2_post:.2f} is high but best "
                       f"gap_closed={best:.3f} < 0.05. Kill condition fires: "
                       "not just basis mismatch; capability is genuinely "
                       "distributed/joint.")
    else:
        verdict = (f"ALIGNMENT_POOR — R2 {r2_pre:.2f}/{r2_post:.2f} below 0.8; "
                   "Procrustes is not a faithful basis map here. Kill condition "
                   "does not cleanly discriminate. Pivot to learned alignment "
                   "(linear probe + cross-entropy) before concluding.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Procrustes-aligned weight-transplant (Codex move 2)",
           "trained_NLL": float(nll_trained),
           "untrained_NLL": float(nll_untrained),
           "procrustes_R2_pre": r2_pre,
           "procrustes_R2_post": r2_post,
           "per_condition": results,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/align_then_transplant.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
