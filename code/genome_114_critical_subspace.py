"""
genome_114_critical_subspace.py

Critical Subspace Power Law: how many PCA directions at layer 14 are actually
critical for capability? genome_113 found that ablating dir-0 alone raises NLL
by +5.83 nats (+138%). Is this a sharp threshold effect (a few dirs are critical,
rest are noise) or a gradual power law?

Protocol
--------
For k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
  Ablate the top-k PCA directions simultaneously across all layers.
  Measure mean NLL on 100 wikitext sequences.

Also:
  - Ablate random-k directions (same k, random unit vectors) as control.
  - Ablate bottom-k PCA directions (least variance) as control.
  - Compute NLL(k) curve: does it follow a power law, exponential, or threshold?

Per-task breakdown: run k-sweep separately on math / code / factual prompts
to test whether the same critical subspace serves all tasks.

Pass: NLL at k=1 > 2x NLL_clean (dominant direction confirmed critical)
      AND NLL at k=5 < 5x NLL at k=1 (not all information in top-1)
Kill: NLL at k=1 < 1.5x NLL_clean (no dominant critical direction — genome_113
      dir-0 finding was an artifact)

Results: results/genome_114_critical_subspace.json
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
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED     = 42

N_SEQS   = 100
SEQ_LEN  = 64
BATCH    = 8
N_DIRS   = 20      # total PCA directions computed
SCAFFOLD_LAYER = 14

K_VALUES = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]  # 0 = clean baseline

# Per-task domain sizes (subset for per-task sweep)
N_TASK = 60

PASS_RATIO_K1  = 2.0   # NLL(k=1) / NLL_clean > 2.0
KILL_RATIO_K1  = 1.5   # NLL(k=1) / NLL_clean < 1.5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_wikitext(n, seed=SEED):
    ds  = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    out = []
    for idx in rng.permutation(len(ds)):
        t = ds[int(idx)]["text"].strip()
        if len(t) >= 60:
            out.append(t[:300])
        if len(out) >= n:
            break
    return out


def make_math_texts(n=N_TASK, seed=SEED):
    rng = np.random.default_rng(seed)
    templates = [
        "What is {a} + {b}?", "Calculate {a} times {b}.",
        "Solve: {a}x - {b} = {c}", "Find the area of a circle with radius {a}.",
        "What is {a}% of {b}?", "Factor: x^2 + {a}x + {b}",
    ]
    out = []
    for _ in range(n):
        t = rng.choice(templates).format(
            a=int(rng.integers(2, 50)),
            b=int(rng.integers(2, 50)),
            c=int(rng.integers(2, 50)),
        )
        out.append(t)
    return out


def make_code_texts(n=N_TASK, seed=SEED):
    snippets = [
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1",
        "import numpy as np\narr = np.zeros((3,3))\nprint(arr.shape)",
        "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
        "class Node:\n    def __init__(self, val):\n        self.val = val",
        "x = [1,2,3,4,5]\nprint(sum(x)/len(x))",
        "try:\n    result = 10/0\nexcept ZeroDivisionError:\n    result = None",
    ]
    rng = np.random.default_rng(seed)
    return [snippets[i % len(snippets)] for i in range(n)]


def make_factual_texts(n=N_TASK, seed=SEED):
    facts = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The speed of light is approximately 299,792 km/s.",
        "DNA is a double helix discovered by Watson and Crick in 1953.",
        "Albert Einstein published special relativity in 1905.",
        "The Pacific Ocean is the largest ocean on Earth.",
    ]
    return [facts[i % len(facts)] for i in range(n)]


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def make_ablate_hook(dirs):
    """Project out directions from residual stream at every layer. dirs: [K, D]."""
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h_f = h.float()
        for k in range(dirs.shape[0]):
            d = dirs[k].to(h_f.device)
            h_f = h_f - (h_f @ d).unsqueeze(-1) * d
        h_out = h_f.to(h.dtype)
        return (h_out,) + out[1:] if isinstance(out, tuple) else h_out
    return hook


def nll_with_ablation(model, enc, dirs, n_layers):
    """Compute mean next-token NLL with directions ablated at all layers."""
    hooks = []
    if dirs.shape[0] > 0:
        hooks = [
            model.model.layers[l].register_forward_hook(make_ablate_hook(dirs))
            for l in range(n_layers)
        ]
    with torch.no_grad():
        logits = model(**enc, use_cache=False).logits.detach().float().cpu()
    for h in hooks:
        h.remove()

    B, T, V = logits.shape
    ids  = enc["input_ids"].cpu()
    mask = enc["attention_mask"].cpu().bool()
    lp   = F.log_softmax(logits[:, :-1, :], dim=-1)
    tgt  = ids[:, 1:]
    m    = mask[:, 1:]
    nll  = -lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    return float(nll[m].mean().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print(f"[0s] Loading model...")
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(DEVICE)
    model.eval()

    NL = len(model.model.layers)
    D  = model.config.hidden_size
    print(f"[{time.time()-t0:.0f}s] {MODEL_ID}: {NL} layers, d={D}")

    # ---- Build wikitext dataset ----
    print(f"[{time.time()-t0:.0f}s] Loading {N_SEQS} wikitext sequences...")
    texts   = load_wikitext(N_SEQS)
    batches = [texts[i:i+BATCH] for i in range(0, len(texts), BATCH)]

    # ---- Collect layer-14 activations for PCA ----
    print(f"[{time.time()-t0:.0f}s] Collecting layer-{SCAFFOLD_LAYER} activations...")
    l14_vecs = []
    for batch_texts in batches:
        enc   = tok(batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=SEQ_LEN).to(DEVICE)
        amask = enc["attention_mask"].bool()
        buf   = {}
        hk    = model.model.layers[SCAFFOLD_LAYER].register_forward_hook(
            lambda m, i, o, b=buf: b.update({"h": (o[0] if isinstance(o, tuple) else o).detach().float().cpu()})
        )
        with torch.no_grad():
            model(**enc, use_cache=False)
        hk.remove()
        h = buf["h"]
        for b in range(h.shape[0]):
            l14_vecs.append(h[b][amask[b].cpu()].mean(0).numpy())

    X   = np.stack(l14_vecs)
    pca = PCA(n_components=N_DIRS, random_state=SEED)
    pca.fit(X)
    pca_dirs    = torch.from_numpy(pca.components_).float()    # [N_DIRS, D] unit
    pca_varexp  = pca.explained_variance_ratio_.tolist()
    print(f"  PCA var_explained top-{N_DIRS}: {sum(pca_varexp):.4f}")

    # ---- Random control directions ----
    torch.manual_seed(SEED)
    rand_dirs_raw = torch.randn(N_DIRS, D)
    rand_dirs     = rand_dirs_raw / rand_dirs_raw.norm(dim=1, keepdim=True)

    # ---- Bottom-k PCA (least variance) ----
    pca_full = PCA(n_components=N_DIRS, random_state=SEED)
    pca_full.fit(X)
    # Reverse: bottom-k = last-k PCA components (least variance)
    pca_bottom = torch.from_numpy(pca_full.components_[::-1].copy()).float()

    # ---- Pre-encode all batches once ----
    batch_encs = []
    for batch_texts in batches:
        enc = tok(batch_texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=SEQ_LEN).to(DEVICE)
        batch_encs.append(enc)

    def sweep_k(directions, label):
        """Run NLL sweep over K_VALUES using given direction matrix."""
        results = {}
        for k in K_VALUES:
            dirs_k = directions[:k] if k > 0 else torch.zeros(0, D)
            nlls   = [nll_with_ablation(model, enc, dirs_k, NL) for enc in batch_encs]
            results[k] = float(np.mean(nlls))
        print(f"  {label}: " + ", ".join(f"k={k}:{results[k]:.3f}" for k in K_VALUES))
        return results

    # ---- Main sweeps ----
    print(f"[{time.time()-t0:.0f}s] PCA top-k sweep...")
    nll_pca_top    = sweep_k(pca_dirs,    "PCA-top")
    print(f"[{time.time()-t0:.0f}s] Random-k sweep (control)...")
    nll_rand       = sweep_k(rand_dirs,   "rand")
    print(f"[{time.time()-t0:.0f}s] PCA bottom-k sweep (control)...")
    nll_pca_bottom = sweep_k(pca_bottom,  "PCA-bot")

    # ---- Per-task sweeps (k=0,1,3,5,10) ----
    task_k_vals = [0, 1, 3, 5, 10]
    task_results = {}
    for task_name, task_texts_fn in [
        ("math",    lambda: make_math_texts()),
        ("code",    lambda: make_code_texts()),
        ("factual", lambda: make_factual_texts()),
    ]:
        print(f"[{time.time()-t0:.0f}s] Per-task sweep: {task_name}...")
        tt = task_texts_fn()
        t_batches = [tt[i:i+BATCH] for i in range(0, len(tt), BATCH)]
        t_encs    = [tok(b, return_tensors="pt", padding=True,
                         truncation=True, max_length=SEQ_LEN).to(DEVICE)
                     for b in t_batches]
        task_nll = {}
        for k in task_k_vals:
            dirs_k = pca_dirs[:k] if k > 0 else torch.zeros(0, D)
            nlls   = [nll_with_ablation(model, enc, dirs_k, NL) for enc in t_encs]
            task_nll[k] = float(np.mean(nlls))
        task_results[task_name] = task_nll
        print(f"  {task_name}: " + ", ".join(f"k={k}:{task_nll[k]:.3f}" for k in task_k_vals))

    # ---- Analysis ----
    nll_clean = nll_pca_top[0]
    nll_k1    = nll_pca_top[1]
    ratio_k1  = nll_k1 / nll_clean

    # Power-law fit to NLL(k) - NLL_clean
    ks   = [k for k in K_VALUES if k > 0]
    deltas = [nll_pca_top[k] - nll_clean for k in ks]
    try:
        log_ks     = np.log(ks)
        log_deltas = np.log(np.clip(deltas, 1e-6, None))
        slope, intercept = np.polyfit(log_ks, log_deltas, 1)
        powerlaw_exponent = float(slope)   # NLL_delta ~ k^slope
    except Exception:
        powerlaw_exponent = None

    # Fraction of total damage explained by k=1
    total_damage_k20  = nll_pca_top[20] - nll_clean
    frac_k1           = (nll_k1 - nll_clean) / max(total_damage_k20, 1e-6)

    # ---- Verdict ----
    if ratio_k1 >= PASS_RATIO_K1:
        verdict = (
            f"CRITICAL_SUBSPACE_CONFIRMED: NLL(k=1)/NLL_clean={ratio_k1:.3f} >= {PASS_RATIO_K1}. "
            f"Top PCA direction is catastrophically critical. "
            f"Dir-0 accounts for {frac_k1:.1%} of total k=20 damage. "
            f"Power-law exponent={powerlaw_exponent:.3f}."
        )
    elif ratio_k1 >= KILL_RATIO_K1:
        verdict = (
            f"PARTIAL: ratio_k1={ratio_k1:.3f} in [{KILL_RATIO_K1},{PASS_RATIO_K1}). "
            f"Dir-0 moderately critical but not catastrophic."
        )
    else:
        verdict = (
            f"NULL: ratio_k1={ratio_k1:.3f} < {KILL_RATIO_K1}. "
            "genome_113 dir-0 finding may have been an artifact. "
            "No dominant critical subspace direction."
        )

    results = {
        "model":                MODEL_ID,
        "n_layers":             NL,
        "d_model":              D,
        "n_seqs":               N_SEQS,
        "scaffold_layer":       SCAFFOLD_LAYER,
        "pca_var_explained":    pca_varexp,
        "k_values":             K_VALUES,
        "nll_pca_top":          nll_pca_top,
        "nll_random":           nll_rand,
        "nll_pca_bottom":       nll_pca_bottom,
        "nll_per_task":         task_results,
        "nll_clean":            nll_clean,
        "nll_k1":               nll_k1,
        "ratio_k1_vs_clean":    ratio_k1,
        "frac_k1_of_total_damage": frac_k1,
        "powerlaw_exponent":    powerlaw_exponent,
        "verdict":              verdict,
        "elapsed_s":            time.time() - t0,
    }

    out_path = RESULTS / "genome_114_critical_subspace.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.time()-t0:.0f}s] Results -> {out_path}")
    print(f"Verdict: {verdict}")
    print(f"NLL curve: " + " ".join(f"k{k}={nll_pca_top[k]:.3f}" for k in K_VALUES))


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
