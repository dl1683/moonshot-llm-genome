"""Adversarial #6: is there a sensitivity-side (Fisher / gradient) invariant?

Codex: 'activation covariance can look universal while the input-output map
differs; the relevant spectra in IB/NTK/mean-field/edge-of-chaos arguments
are Jacobians/Fisher/NTK kernels.'

Test: compute per-layer per-parameter-group Fisher diagonal on Qwen3-0.6B
over 128 C4 sentences. Fisher_diag[p] = E[(∂ log p / ∂ p)^2] at the LM
output. Sort diagonal entries, treat as a spectrum, compute analog
sqrt(er)*alpha.

If this 'Fisher invariant' produces a tight constant across systems
(comparable to activation-side 4.3), the universality is DEEPER than just
activations - it's in the loss-landscape curvature. If it's wildly
different / high-CV, the activation universality doesn't extend to the
sensitivity side.

Budget: forward+backward on 128 sentences, ~90s per system. 3 systems ~5 min.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent

SYSTEMS = [
    ("qwen3-0.6b",                   "Qwen/Qwen3-0.6B"),
    ("deepseek-r1-distill-qwen-1.5b","deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("bert-base-uncased",            "bert-base-uncased"),
]


def compute_fisher_diag_per_parameter(model, tokenizer, texts, n_max=128,
                                       batch=1, max_len=128, device="cuda"):
    """Diagonal of empirical Fisher = E_x[ (grad log p(x))^2 ] per parameter.
    Accumulate squared gradients of next-token NLL across batches.
    Returns 1D numpy array of per-parameter squared-gradient means.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    # Accumulators same shape as model params
    accum = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    n_done = 0
    for i in range(0, min(len(texts), n_max), batch):
        chunk = texts[i:i+batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len).to(device)
        model.zero_grad(set_to_none=True)
        # For causal LM, shift labels
        input_ids = enc["input_ids"]
        labels = input_ids.clone()
        # Mask pad tokens in labels
        if tokenizer.pad_token_id is not None:
            labels = labels.masked_fill(input_ids == tokenizer.pad_token_id, -100)
        try:
            out = model(**enc, labels=labels)
            loss = out.loss
        except Exception:
            # MLM-style models: use LM head with MLM loss surrogate
            out = model(**enc, output_hidden_states=False)
            logits = out.logits.float()
            if labels.shape == logits.shape[:2]:
                lbl = labels[:, 1:]
                lg = logits[:, :-1, :]
                loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lbl.reshape(-1),
                                        ignore_index=-100)
            else:
                continue
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                accum[n] += p.grad.detach() ** 2
        n_done += 1
    # Average
    fisher_flat = []
    for n, acc in accum.items():
        fisher_flat.append((acc / max(n_done, 1)).flatten().cpu().numpy())
    return np.concatenate(fisher_flat).astype(np.float64)


def stats_from_fisher(fisher_diag, subsample=50000):
    """Treat Fisher diagonal as a 'spectrum' via sorted values. Take top-k subsample
    to keep SVD-equivalent computation cheap (we only need eff_rank + tail slope).
    """
    v = np.sort(fisher_diag)[::-1]  # descending
    if subsample and len(v) > subsample:
        v = v[:subsample]
    total = v.sum()
    er = float(total ** 2 / (v * v).sum()) if total > 0 else 0.0
    h = len(v)
    r = np.arange(1, h + 1)
    lo, hi = max(1, int(h * 0.05)), int(h * 0.5)
    # Fit log v vs log r
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(v[lo:hi] + 1e-30), 1)
    alpha_on_v = float(-slope)  # v acts like sigma^2; so alpha_on_sigma = alpha_on_v / 2
    alpha_sigma = alpha_on_v / 2.0
    return {"fisher_eff_rank": er,
            "fisher_alpha_on_v": alpha_on_v,
            "fisher_alpha_sigma": alpha_sigma,
            "sqrt_er_alpha": float(np.sqrt(er) * alpha_sigma),
            "total_params_subsampled": h,
            "total_fisher_mass": float(total)}


def main():
    t0 = time.time()
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=2000):
        sents.append(rec["text"])
        if len(sents) >= 128:
            break

    rows = []
    for label, hf_id in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {label} =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
        # Cast to fp32 for stable Fisher computation
        sys_obj.model.to(dtype=torch.float32)
        try:
            fisher = compute_fisher_diag_per_parameter(
                sys_obj.model, sys_obj.tokenizer, sents, n_max=64,
            )
        except Exception as e:
            print(f"  FAIL Fisher: {e}")
            sys_obj.unload(); torch.cuda.empty_cache(); continue
        sys_obj.unload(); torch.cuda.empty_cache()
        st = stats_from_fisher(fisher)
        st.update({"label": label})
        rows.append(st)
        print(f"  params={st['total_params_subsampled']}  "
              f"fisher_eff_rank={st['fisher_eff_rank']:.2f}  "
              f"alpha_sigma={st['fisher_alpha_sigma']:.3f}  "
              f"sqrt(er)*a={st['sqrt_er_alpha']:.3f}")

    print("\n\n=== FISHER-SIDE INVARIANT ===")
    invs = [r["sqrt_er_alpha"] for r in rows]
    ers = [r["fisher_eff_rank"] for r in rows]
    if invs:
        m, s = float(np.mean(invs)), float(np.std(invs))
        cv = 100*s/m if m else 0
        print(f"  sqrt(er_fisher)*alpha  mean={m:.3f}  std={s:.3f}  CV={cv:.2f}%")
        em, es_ = float(np.mean(ers)), float(np.std(ers))
        ecv = 100*es_/em if em else 0
        print(f"  fisher_eff_rank        mean={em:.2f}  std={es_:.2f}  CV={ecv:.2f}%")
        print(f"  activation-side baseline:  ~4.3, CV 5%")

    out = {"rows": rows}
    out_path = _ROOT / "results/gate2/fisher_invariant.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
