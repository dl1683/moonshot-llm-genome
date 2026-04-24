"""Genome_109: Codex round-2 blind spot #1 - is fractional depth the right axis?

Codex: 'Normalized depth = layer_index/(n_layers-1) assumes halfway through
BERT-12 matches halfway through Qwen-28. Test: re-index layers by functional
progress (logit-lens ΔNLL) and see if the tight band survives.'

Plan: for each of 3 causal LMs (Qwen3, DeepSeek, Qwen3-1.7B), compute:
 1. Per-layer logit-lens NLL on C4 (project mid-layer hidden state through
    final LM head, compute next-token NLL). This gives a FUNCTIONAL depth axis.
 2. Per-layer sqrt(er)*alpha as before.
Plot sqrt(er)*alpha vs LOGIT-LENS-NLL (functional axis) instead of vs
fractional depth. If the curves ALSO overlay in functional-axis space, the
band universality holds. If not, fractional-depth axis was arbitrary.
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
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent

SYSTEMS = [
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    ("deepseek-r1-distill-qwen-1.5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("qwen3-1.7b", "Qwen/Qwen3-1.7B"),
]


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(max(X.shape[0] - 1, 1))
    return s.astype(np.float64)


def stats(s):
    s2 = s ** 2
    er = float(s2.sum() ** 2 / (s2 ** 2).sum()) if s2.sum() > 0 else 0.0
    h = len(s)
    r = np.arange(1, h + 1)
    lo, hi = max(1, int(h * 0.05)), int(h * 0.5)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    alpha = float(-slope)
    return float(np.sqrt(er) * alpha)


@torch.no_grad()
def logit_lens_nll_per_layer(model, tok, texts, batch=4, max_len=128, device="cuda"):
    """Compute per-layer logit-lens NLL on next-token prediction.
    For each layer L, project hidden state through the LM head, compute NLL.
    Returns array of NLLs, one per layer.
    """
    model.eval()
    # Find LM head — varies by arch
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        lm_head = getattr(model, "embed_out", None)  # Pythia
    if lm_head is None:
        raise RuntimeError("Cannot find LM head")

    # Find final layernorm
    final_ln = None
    for name in ("model.norm", "model.final_layer_norm", "gpt_neox.final_layer_norm",
                 "transformer.ln_f"):
        parts = name.split(".")
        obj = model
        try:
            for p in parts: obj = getattr(obj, p)
            final_ln = obj
            break
        except AttributeError:
            continue

    n_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else None
    if n_layers is None:
        raise RuntimeError("Cannot find num_hidden_layers")

    nll_by_layer = np.zeros(n_layers + 1)  # 0..n_layers (hidden_states has n_layers+1)
    count_by_layer = np.zeros(n_layers + 1)

    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                   max_length=max_len).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        out = model(**enc, output_hidden_states=True)
        # Next-token label: shift
        labels = input_ids[:, 1:].clone()  # (b, s-1)
        # Mask padding + first token
        valid = attention_mask[:, 1:].bool()
        for L, h in enumerate(out.hidden_states):
            h = h.float()[:, :-1, :]  # match shifted
            if final_ln is not None:
                h_norm = final_ln(h.to(next(final_ln.parameters()).dtype)).float()
            else:
                h_norm = h
            logits = lm_head(h_norm.to(next(lm_head.parameters()).dtype)).float()
            # NLL on valid positions
            logp = torch.nn.functional.log_softmax(logits, dim=-1)
            gathered = torch.gather(logp, 2, labels.unsqueeze(-1)).squeeze(-1)  # (b, s-1)
            nll_vals = -gathered[valid]
            nll_by_layer[L] += float(nll_vals.sum().item())
            count_by_layer[L] += int(valid.sum().item())
    nll_by_layer = nll_by_layer / np.maximum(count_by_layer, 1)
    return nll_by_layer[1:]  # layer 0 is input embeddings; skip, return n_layers values


@torch.no_grad()
def extract_per_layer_invariant(model, tok, texts, batch=16, max_len=256, device="cuda"):
    n_layers = model.config.num_hidden_layers
    invs = np.zeros(n_layers)
    for L in range(n_layers):
        acts = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_len).to(device)
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[L+1].float()  # +1 to skip input embeddings
            mask = enc["attention_mask"].float().unsqueeze(-1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            acts.append(pooled.cpu().numpy())
        X = np.concatenate(acts, axis=0).astype(np.float32)
        s = spectrum(X)
        invs[L] = stats(s)
    return invs


def main():
    t0 = time.time()
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=2000):
        sents.append(rec["text"])
        if len(sents) >= 400:
            break

    per_system = {}
    for label, hf_id in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {label} =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
        try:
            invs = extract_per_layer_invariant(sys_obj.model, sys_obj.tokenizer, sents)
            nlls = logit_lens_nll_per_layer(sys_obj.model, sys_obj.tokenizer, sents)
        except Exception as e:
            print(f"  FAIL compute: {e}")
            sys_obj.unload(); torch.cuda.empty_cache(); continue
        sys_obj.unload(); torch.cuda.empty_cache()
        per_system[label] = {"invs": invs.tolist(), "nlls": nlls.tolist(),
                              "n_layers": len(invs)}
        n_layers = len(invs)
        print(f"  n_layers={n_layers}")
        print(f"  invariant per layer (first/mid/last): {invs[0]:.2f}, "
              f"{invs[n_layers//2]:.2f}, {invs[-1]:.2f}")
        print(f"  logit-lens NLL     (first/mid/last): {nlls[0]:.2f}, "
              f"{nlls[n_layers//2]:.2f}, {nlls[-1]:.2f}")

    # Analysis: bin by functional depth = (nll_0 - nll_L) / (nll_0 - nll_final)
    # i.e. normalize NLL descent to [0, 1], then bin invariants per functional-depth bin
    print("\n=== CV across systems per FUNCTIONAL-DEPTH bin ===")
    n_bins = 10
    bin_stats = [[] for _ in range(n_bins)]
    for label, data in per_system.items():
        invs = np.array(data["invs"])
        nlls = np.array(data["nlls"])
        # Normalize NLL to functional-depth [0, 1]: 0 = first layer (high NLL), 1 = last (low NLL)
        nll_lo, nll_hi = nlls.max(), nlls.min()
        denom = (nll_lo - nll_hi) if nll_lo != nll_hi else 1.0
        func_depth = (nll_lo - nlls) / denom  # 0=first, 1=last
        for f, inv in zip(func_depth, invs):
            b = min(int(f * n_bins), n_bins - 1)
            bin_stats[b].append(inv)

    print(f"  {'func_depth_bin':>15s} {'mean':>7s} {'CV%':>6s} {'N':>3s}")
    for b in range(n_bins):
        vals = bin_stats[b]
        if len(vals) < 2: continue
        m, s = float(np.mean(vals)), float(np.std(vals))
        cv = 100*s/m if m else 0
        lo, hi = b/n_bins, (b+1)/n_bins
        mark = "  <-- TIGHT" if 0 < cv < 12 else ("  <-- LOOSE" if cv > 25 else "")
        print(f"  [{lo:.1f},{hi:.1f}]       {m:>7.3f} {cv:>6.2f}% {len(vals):>3d}{mark}")

    out = {"per_system": per_system}
    out_path = _ROOT / "results/gate2/functional_depth.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
