"""
genome_116c_multilayer_decode.py

Does PC1 change character as depth increases across early layers?
genome_116b showed layer-5 PC1 is sentence-boundary/DC axis.
Are layers 8 and 11 also structural, or do they transition to semantic?

Fast decode: top/bottom token types for PC1 at layers 2, 5, 8, 11.
~8 minutes on RTX 5090.

Results: results/genome_116c_multilayer_decode.json
"""

import json
import pathlib
import time
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT    = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED     = 42

N_FIT   = 200
N_PROBE = 300
SEQ_LEN = 64
BATCH   = 8
PROBE_LAYERS = [2, 5, 8, 11]
TOP_K   = 20


def load_wikitext_split(n, offset, seed=SEED):
    ds  = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    out, count = [], 0
    for idx in perm:
        t = ds[int(idx)]["text"].strip()
        if len(t) < 60:
            continue
        if count >= offset:
            out.append(t[:300])
        count += 1
        if len(out) >= n:
            break
    return out


def tokenize(texts, tok):
    enc = tok(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=SEQ_LEN)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def fit_pc1(model, tok, fit_texts, layer_idx):
    pooled = []
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        pooled.append(h.detach().float().mean(dim=1).cpu())
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    for i in range(0, len(fit_texts), BATCH):
        ids, mask = tokenize(fit_texts[i:i+BATCH], tok)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
    handle.remove()
    acts = torch.cat(pooled, dim=0).numpy()
    pca = PCA(n_components=5)
    pca.fit(acts)
    pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-8)
    return pc1, float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])


def decode_layer(model, tok, probe_texts, pc1, layer_idx):
    per_tok_acts, seq_ids_storage = [], []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        per_tok_acts.append(h.detach().float().cpu())

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    for i in range(0, len(probe_texts), BATCH):
        ids, mask = tokenize(probe_texts[i:i+BATCH], tok)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        seq_ids_storage.append(ids.cpu())
    handle.remove()

    pc1_t = torch.tensor(pc1, dtype=torch.float32)
    token_proj_map = defaultdict(list)
    proj_stats = []

    for acts_b, ids_b in zip(per_tok_acts, seq_ids_storage):
        B, T, D = acts_b.shape
        projs = (acts_b @ pc1_t).numpy()
        for b in range(B):
            for t in range(T):
                tid = ids_b[b, t].item()
                if tid == tok.pad_token_id:
                    continue
                p = float(projs[b, t])
                token_proj_map[tid].append(p)
                proj_stats.append(p)

    proj_arr = np.array(proj_stats)
    token_mean = {
        tok.decode([tid]).strip(): float(np.mean(vals))
        for tid, vals in token_proj_map.items()
        if len(vals) >= 3
    }
    top  = sorted(token_mean.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    bot  = sorted(token_mean.items(), key=lambda x: x[1])[:TOP_K]

    def safe(s):
        return s.encode("ascii", errors="replace").decode("ascii")

    return {
        "proj_mean": float(proj_arr.mean()),
        "proj_std":  float(proj_arr.std()),
        "proj_min":  float(proj_arr.min()),
        "proj_max":  float(proj_arr.max()),
        "frac_positive": float((proj_arr > 0).mean()),
        "top_tokens":  [{"token": safe(t), "mean_proj": p} for t, p in top],
        "bot_tokens":  [{"token": safe(t), "mean_proj": p} for t, p in bot],
    }


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(DEVICE).eval()

    fit_texts   = load_wikitext_split(N_FIT, offset=0)
    probe_texts = load_wikitext_split(N_PROBE, offset=N_FIT)

    results = {}
    for layer_idx in PROBE_LAYERS:
        print(f"\n=== Layer {layer_idx} ===")
        pc1, var1, var2 = fit_pc1(model, tok, fit_texts, layer_idx)
        print(f"  PC1 var={var1:.3f}, PC2 var={var2:.3f}")
        dec = decode_layer(model, tok, probe_texts, pc1, layer_idx)
        print(f"  proj: mean={dec['proj_mean']:.1f} std={dec['proj_std']:.1f} "
              f"min={dec['proj_min']:.2f} frac_pos={dec['frac_positive']:.3f}")
        print(f"  TOP: {[t['token'] for t in dec['top_tokens'][:8]]}")
        print(f"  BOT: {[t['token'] for t in dec['bot_tokens'][:8]]}")
        results[layer_idx] = {"var_pc1": var1, "var_pc2": var2, **dec}

    out = {"model": MODEL_ID, "probe_layers": PROBE_LAYERS,
           "n_fit": N_FIT, "n_probe": N_PROBE,
           "results_by_layer": results, "elapsed_s": time.time() - t0}
    out_path = RESULTS / "genome_116c_multilayer_decode.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
