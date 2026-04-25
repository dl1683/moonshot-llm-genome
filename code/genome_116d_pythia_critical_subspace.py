"""
genome_116d_pythia_critical_subspace.py

Cross-architecture replication: does Pythia-160M show the same power-law
critical subspace concentration as Qwen3-0.6B (genome_114)?

Qwen3 result: top PCA direction at layer 14 causes +5.83 nats (+138%) when
ablated. ratio_k1=2.38 (73% of total k=20 damage in k=1). Power-law exp=0.108.
PC1 is a sentence-boundary/DC axis (genome_116b), identical across layers 2-11.

If Pythia-160M shows the same concentration structure:
- Critical subspace concentration is architecture-universal (not Qwen3-specific)
- The sentence-boundary prior is universal across training objectives/architectures
- Surgery transplant is a viable research direction across models

Protocol:
1. Load Pythia-160M (12 decoder layers, d=768)
2. Layer sweep: ablate top-1 PCA direction at layers {2,4,6,8,10} to find max-damage layer
3. At strongest layer: ablate top-k for k=1..10, compare vs random-k controls
4. Report: ratio_k1, power_law_exponent, vs Qwen3 baseline

Pass: ratio_k1 >= 2.0 AND layer_top1_delta >= 1.0 nats (same thresholds as genome_113)
Kill: ratio_k1 < 1.5 AND delta < 0.5 nats

~10 minutes on RTX 5090.
Results: results/genome_116d_pythia_critical_subspace.json
"""

import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT    = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "EleutherAI/pythia-160m"
SEED     = 42

N_FIT   = 200
N_EVAL  = 100
SEQ_LEN = 64
BATCH   = 8
N_BOOT  = 500
TOP_K   = 10

PASS_DELTA   = 1.0
PASS_RATIO   = 2.0


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


def masked_mean_pool(h, mask):
    m = mask.float().unsqueeze(-1)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)


def extract_pca_directions(model, tok, texts, layer_idx, n_components=20):
    raw_acts, masks = [], []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        raw_acts.append(h.detach().float().cpu())

    handle = model.gpt_neox.layers[layer_idx].register_forward_hook(hook_fn)
    for i in range(0, len(texts), BATCH):
        ids, mask = tokenize(texts[i:i+BATCH], tok)
        masks.append(mask.cpu())
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
    handle.remove()

    pooled = torch.cat(
        [masked_mean_pool(h, m) for h, m in zip(raw_acts, masks)], dim=0
    ).numpy()
    pca = PCA(n_components=n_components)
    pca.fit(pooled)
    return pca.components_, pca.explained_variance_ratio_


def logits_to_nll(logits, input_ids, attention_mask=None):
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
        ignore_index=-100,
    ).item()


def make_ablation_hook(directions):
    """Project out all rows of directions (shape: k x D) from activations."""
    dirs_t = torch.tensor(np.array(directions), dtype=torch.bfloat16, device=DEVICE)
    # orthonormalize
    dirs_t = dirs_t / (dirs_t.norm(dim=1, keepdim=True) + 1e-8)

    def hook_fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        for d in dirs_t:
            proj = (h @ d).unsqueeze(-1) * d
            h = h - proj
        return (h,) + out[1:] if is_tuple else h

    return hook_fn


def measure_nll_per_seq(model, tok, texts, hook_fns_by_layer=None):
    handles = []
    if hook_fns_by_layer:
        for li, fn in hook_fns_by_layer.items():
            handles.append(model.gpt_neox.layers[li].register_forward_hook(fn))

    per_seq = []
    for i in range(0, len(texts), BATCH):
        ids, mask = tokenize(texts[i:i+BATCH], tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        for j in range(ids.shape[0]):
            per_seq.append(logits_to_nll(out.logits[j:j+1], ids[j:j+1], mask[j:j+1]))

    for h in handles:
        h.remove()
    return np.array(per_seq)


def bootstrap_delta(clean_nlls, ablated_nlls, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    deltas = ablated_nlls - clean_nlls
    boots = [rng.choice(deltas, size=len(deltas), replace=True).mean()
             for _ in range(n_boot)]
    arr = np.array(boots)
    return {
        "mean":   float(deltas.mean()),
        "ci_lo":  float(np.percentile(arr, 2.5)),
        "ci_hi":  float(np.percentile(arr, 97.5)),
    }


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    n_layers = model.config.num_hidden_layers
    d_model  = model.config.hidden_size
    print(f"  {MODEL_ID}: {n_layers} layers, d={d_model}")

    fit_texts  = load_wikitext_split(N_FIT, offset=0)
    eval_texts = load_wikitext_split(N_EVAL, offset=N_FIT)

    print("Measuring clean NLL...")
    clean_per_seq = measure_nll_per_seq(model, tok, eval_texts)
    nll_clean = float(clean_per_seq.mean())
    print(f"  NLL_clean = {nll_clean:.4f}")

    # -----------------------------------------------------------------------
    # Step 1: Layer sweep — find strongest ablation layer
    # -----------------------------------------------------------------------
    sweep_layers = list(range(1, n_layers, max(1, n_layers // 6)))
    print(f"\n=== Layer sweep {sweep_layers} ===")
    sweep_results = {}
    best_layer, best_delta = 0, 0.0

    for li in sweep_layers:
        dirs, vars_ratio = extract_pca_directions(model, tok, fit_texts, li, n_components=5)
        hook = make_ablation_hook([dirs[0]])
        abl_per_seq = measure_nll_per_seq(model, tok, eval_texts, {li: hook})
        stat = bootstrap_delta(clean_per_seq, abl_per_seq)
        print(f"  layer {li:2d}  delta={stat['mean']:.3f}  CI=[{stat['ci_lo']:.3f},{stat['ci_hi']:.3f}]  var={vars_ratio[0]:.3f}")
        sweep_results[li] = {**stat, "var_pc1": float(vars_ratio[0])}
        if stat["mean"] > best_delta:
            best_delta, best_layer = stat["mean"], li

    print(f"\nBest layer: {best_layer} (delta={best_delta:.3f} nats)")

    # -----------------------------------------------------------------------
    # Step 2: Power-law curve at best layer
    # -----------------------------------------------------------------------
    print(f"\n=== Power-law curve at layer {best_layer} ===")
    dirs_best, vars_best = extract_pca_directions(model, tok, fit_texts, best_layer, n_components=20)

    topk_results = []
    rand_k1_deltas = []

    # Top-k ablations
    for k in range(1, TOP_K + 1):
        hook = make_ablation_hook(dirs_best[:k])
        abl = measure_nll_per_seq(model, tok, eval_texts, {best_layer: hook})
        stat = bootstrap_delta(clean_per_seq, abl)
        topk_results.append({"k": k, **stat})
        print(f"  top-{k:2d}  delta={stat['mean']:.3f}  CI=[{stat['ci_lo']:.3f},{stat['ci_hi']:.3f}]")

    # Random-k=1 controls (n=5)
    for _ in range(5):
        rand_dir = rng.standard_normal(d_model)
        rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-8)
        hook = make_ablation_hook([rand_dir])
        abl = measure_nll_per_seq(model, tok, eval_texts, {best_layer: hook})
        stat = bootstrap_delta(clean_per_seq, abl)
        rand_k1_deltas.append(stat["mean"])

    rand_k1_mean = float(np.mean(rand_k1_deltas))
    delta_k1 = topk_results[0]["mean"]
    delta_k10 = topk_results[-1]["mean"]
    ratio_k1 = delta_k1 / (abs(rand_k1_mean) + 1e-6)

    print(f"\n  delta_k1={delta_k1:.3f}, delta_k10={delta_k10:.3f}")
    print(f"  rand_k1_mean={rand_k1_mean:.4f}")
    print(f"  ratio_k1={ratio_k1:.2f}")

    # Fit power-law to top-k curve
    ks = np.arange(1, TOP_K + 1, dtype=float)
    deltas = np.array([r["mean"] for r in topk_results])
    log_ks = np.log(ks)
    log_deltas = np.log(np.maximum(deltas, 1e-6))
    coeffs = np.polyfit(log_ks, log_deltas, 1)
    power_law_exp = float(coeffs[0])
    print(f"  power_law_exponent={power_law_exp:.3f}")

    layer_pass = (delta_k1 >= PASS_DELTA and ratio_k1 >= PASS_RATIO)
    print(f"\n  PASS={layer_pass} (delta>={PASS_DELTA}, ratio>={PASS_RATIO})")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_fit": N_FIT,
        "n_eval": N_EVAL,
        "nll_clean": nll_clean,
        "layer_sweep": sweep_results,
        "best_layer": best_layer,
        "best_layer_delta": best_delta,
        "topk_curve": topk_results,
        "rand_k1_mean": rand_k1_mean,
        "rand_k1_deltas": rand_k1_deltas,
        "ratio_k1": ratio_k1,
        "power_law_exponent": power_law_exp,
        "layer_pass": layer_pass,
        "elapsed_s": time.time() - t0,
        "qwen3_comparison": {
            "ratio_k1": 2.38,
            "power_law_exp": 0.108,
            "best_layer": 14,
            "delta_k1": 5.83,
        },
    }
    out_path = RESULTS / "genome_116d_pythia_critical_subspace.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
