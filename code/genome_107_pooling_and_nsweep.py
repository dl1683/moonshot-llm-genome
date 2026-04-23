"""Genome_107: Codex round-2 blind spots #2 (pooling) and #4 (n-sweep).

Blind spot #2: 'seq_mean pooling may be the whole story. Scrambled word-order
preserving the effect screams bag-of-words/unigram.' Test token-level (no
averaging) and last-token pooling.

Blind spot #4: 'n_stimuli=800 rank-caps spectra and may concentrate
generically.' Test n=800 vs n=3200.

Probe: Qwen3-0.6B + DeepSeek + BERT at mid-depth, 4 pooling modes x 2 n values.
If seq_mean at n=800 is special (much tighter than other (pool, n)), the claim
is artifactual.
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
    ("bert-base-uncased", "bert-base-uncased"),
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
    return {"eff_rank": er, "alpha": alpha,
            "sqrt_er_alpha": float(np.sqrt(er) * alpha)}


@torch.no_grad()
def extract_with_pooling(model, tok, texts, layer_idx, pooling, batch=8, max_len=256,
                          device="cuda"):
    """Extract mid-depth activations under different pooling rules.
    pooling ∈ {'seq_mean', 'last_token', 'no_pool', 'cls_first'}
    """
    model.eval()
    acts = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                   max_length=max_len).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer_idx].float()  # (b, s, d)
        mask = enc["attention_mask"].float()
        if pooling == "seq_mean":
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        elif pooling == "last_token":
            # Find last non-pad position per row
            lens = mask.sum(1).long() - 1  # (b,)
            pooled = h[torch.arange(h.shape[0]), lens]
        elif pooling == "cls_first":
            pooled = h[:, 0]  # first token
        elif pooling == "no_pool":
            # Flatten all tokens (subject to mask)
            valid = mask.bool()  # (b, s)
            flat = h[valid]  # (sum_valid, d)
            # subsample to keep reasonable size
            if flat.shape[0] > 256:
                idx = torch.randperm(flat.shape[0], device=device)[:256]
                flat = flat[idx]
            acts.append(flat.cpu().numpy())
            continue
        else:
            raise ValueError(pooling)
        acts.append(pooled.cpu().numpy())
    X = np.concatenate(acts, axis=0).astype(np.float32)
    return X


def main():
    t0 = time.time()
    # Load enough sentences for n=3200
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=8000):
        sents.append(rec["text"])
        if len(sents) >= 3200:
            break

    subset_800 = sents[:800]
    subset_3200 = sents[:3200]
    pools = ["seq_mean", "last_token", "cls_first", "no_pool"]

    rows = []
    for label, hf_id in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {label} =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
        mid = max(1, sys_obj.n_hidden_layers() // 2)
        for n_val, sset in [(800, subset_800), (3200, subset_3200)]:
            for pool in pools:
                try:
                    X = extract_with_pooling(sys_obj.model, sys_obj.tokenizer,
                                              sset, mid, pool)
                except Exception as e:
                    print(f"  FAIL {pool},n={n_val}: {e}"); continue
                s = spectrum(X)
                st = stats(s)
                row = {"system": label, "pooling": pool, "n": n_val,
                        "X_shape": list(X.shape), **st}
                rows.append(row)
                print(f"  {pool:12s} n={n_val:5d}  X={X.shape}  "
                      f"sqrt(er)*a={st['sqrt_er_alpha']:.3f}")
        sys_obj.unload(); torch.cuda.empty_cache()

    print(f"\n\n=== CROSS-SYSTEM CV per (pool, n) ===")
    print(f"  {'pool':12s} {'n':>5s}  {'mean':>7s} {'std':>7s} {'CV%':>6s}")
    pool_n_keys = sorted(set((r["pooling"], r["n"]) for r in rows))
    for pool, n_val in pool_n_keys:
        invs = [r["sqrt_er_alpha"] for r in rows if r["pooling"] == pool and r["n"] == n_val]
        if len(invs) < 2: continue
        m, s = float(np.mean(invs)), float(np.std(invs))
        cv = 100*s/m if m else 0
        mark = "  <-- TIGHT" if 0 < cv < 10 else ("  <-- LOOSE" if cv > 20 else "")
        print(f"  {pool:12s} {n_val:>5d}  {m:>7.3f} {s:>7.3f} {cv:>6.2f}%{mark}")

    out = {"rows": rows}
    out_path = _ROOT / "results/gate2/pooling_nsweep.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
