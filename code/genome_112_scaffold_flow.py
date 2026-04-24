"""
genome_112_scaffold_flow.py

Scaffold-and-Flow Fields: does Qwen3-0.6B route distinct task types through
different paths in a shared latent scaffold?

Hypothesis (Codex mental model 1): capability lives in how hidden-state mass
occupies and flows through a shared scaffold of directions. Different capabilities
= different occupancy/flow patterns through the same set of shared directions.
The model's own attractor geometry defines the scaffold; tasks are distinguished
by which regions they visit and how activation mass transitions between them.

Protocol
--------
1. Load top-30 shared eigendirections from genome_099 (cross-arch PCA).
   If not available, compute on-the-fly via PCA of layer-14 activations.
2. Collect 400 diverse contexts: 100 wikitext + 100 math + 100 code + 100 factual.
3. For each context, record the layer-14 residual stream projected onto the top-30
   shared directions -> "scaffold coordinates" vector of shape [T, 30].
   Use mean-pool over T to get a per-sequence scaffold coordinate [30].
4. Compute 2-class separability for each task pair in scaffold space:
   - LDA (linear discriminant): Fisher ratio = between_scatter / within_scatter
   - Pairwise cosine distance distributions (mean, std across task pairs)
5. For each layer l in [0, n_layers-1]:
   - Project layer-l states onto same scaffold directions
   - Compute inter-task centroid distance matrix (4x4) and within-task spread
   - Measure separability = mean(inter_centroid) / mean(sqrt(within_scatter))
6. Test whether separability peaks at a specific depth (mid-layer "flow node").

Pass:  max_separability > 2.0 AND layer at peak is in range [6, 22] (mid-depth)
Kill:  max_separability < 1.2 across all layers -> flow fields are homogeneous

Also compute: trajectory divergence = how early in depth do task-specific paths
diverge? (first layer where inter-task distance > 2*within-task distance)

Results: results/genome_112_scaffold_flow.json
"""

import json
import pathlib
import time

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

SEQ_LEN     = 64
BATCH       = 8
N_SCAFFOLD  = 30   # top-k scaffold directions to project onto
SCAFFOLD_LAYER = 14  # layer used to compute scaffold directions
N_WIKI  = 100
N_MATH  = 100
N_CODE  = 100
N_FACT  = 100

PASS_SEP = 2.0
KILL_SEP = 1.2


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
            out.append(("wiki", t[:300]))
        if len(out) >= n:
            break
    return out


def make_math_texts(n=N_MATH, seed=SEED):
    rng = np.random.default_rng(seed)
    templates = [
        "Solve: {a} + {b} * {c} = ?",
        "What is the derivative of x^{a} + {b}x?",
        "Find the area of a circle with radius {a}.",
        "Calculate {a} / {b} rounded to 2 decimal places.",
        "If x = {a} and y = {b}, what is 2x + 3y?",
        "Factorize: x^2 + {a}x + {b}",
        "Solve for x: {a}x - {b} = {c}",
        "What is {a}% of {b}?",
    ]
    out = []
    for i in range(n):
        t = templates[i % len(templates)].format(
            a=int(rng.integers(2, 20)),
            b=int(rng.integers(2, 20)),
            c=int(rng.integers(2, 20)),
        )
        out.append(("math", t))
    return out


def make_code_texts(n=N_CODE, seed=SEED):
    snippets = [
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2",
        "for i in range(len(data)):\n    if data[i] > threshold:\n        result.append(transform(data[i]))",
        "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None",
        "import numpy as np\ndef normalize(x):\n    return (x - x.mean()) / (x.std() + 1e-8)",
        "SELECT user_id, COUNT(*) as visits\nFROM events\nWHERE timestamp > '2024-01-01'\nGROUP BY user_id",
        "const fetchData = async (url) => {\n    const response = await fetch(url);\n    return response.json();\n};",
        "fn main() {\n    let v: Vec<i32> = (0..10).map(|x| x*x).collect();\n    println!(\"{:?}\", v);\n}",
        "try:\n    result = json.loads(response.text)\nexcept json.JSONDecodeError as e:\n    logger.error(f'Parse error: {e}')",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]",
        "git log --oneline --graph --all | head -20",
    ]
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = snippets[i % len(snippets)]
        out.append(("code", s))
    return out


def make_factual_texts(n=N_FACT, seed=SEED):
    facts = [
        "The speed of light is approximately 299,792 kilometers per second.",
        "The capital of France is Paris, which was founded around 250 BC.",
        "DNA is a double helix structure discovered by Watson and Crick in 1953.",
        "The Roman Empire fell in 476 AD when Romulus Augustulus was deposed.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "Shakespeare wrote approximately 37 plays and 154 sonnets.",
        "The Pacific Ocean covers about 165 million square kilometers.",
        "Penicillin was discovered by Alexander Fleming in 1928.",
        "The human brain contains approximately 86 billion neurons.",
        "Albert Einstein published his theory of special relativity in 1905.",
        "The Great Wall of China stretches approximately 21,196 kilometers.",
        "Bitcoin was created by Satoshi Nakamoto in 2008.",
    ]
    out = []
    for i in range(n):
        out.append(("fact", facts[i % len(facts)]))
    return out


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def extract_all_layers(model, enc, n_layers):
    """Return dict[layer_idx -> [B, T, D] float32 cpu tensor]."""
    storage = {}

    def make_hook(l):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            storage[l] = h.detach().float().cpu()
        return hook

    hooks = [model.model.layers[l].register_forward_hook(make_hook(l))
             for l in range(n_layers)]
    with torch.no_grad():
        model(**enc, use_cache=False)
    for h in hooks:
        h.remove()
    return storage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print(f"[0s] Loading model...")
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    model.eval()

    NL = len(model.model.layers)
    D  = model.config.hidden_size
    print(f"[{time.time()-t0:.0f}s] {MODEL_ID}: {NL} layers, d={D}")

    # ---- Build dataset ----
    print(f"[{time.time()-t0:.0f}s] Loading texts...")
    all_texts = (
        load_wikitext(N_WIKI) +
        make_math_texts() +
        make_code_texts() +
        make_factual_texts()
    )
    np.random.seed(SEED)
    np.random.shuffle(all_texts)
    labels = [x[0] for x in all_texts]   # domain labels
    texts  = [x[1] for x in all_texts]
    N      = len(texts)
    domain2idx = {"wiki": 0, "math": 1, "code": 2, "fact": 3}
    label_ids  = np.array([domain2idx[l] for l in labels])

    # ---- Forward pass: collect all layers ----
    print(f"[{time.time()-t0:.0f}s] Running forward passes ({N} contexts, batch={BATCH})...")
    # all_states[l] = list of [T, D] tensors (one per sequence)
    all_states = {l: [] for l in range(NL)}

    batches = [texts[i:i+BATCH] for i in range(0, N, BATCH)]
    for bi, batch_texts in enumerate(batches):
        enc = tok(batch_texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=SEQ_LEN).to(DEVICE)
        amask = enc["attention_mask"].bool()
        storage = extract_all_layers(model, enc, NL)
        for l in range(NL):
            h = storage[l]  # [B, T, D]
            for b in range(h.shape[0]):
                valid_tokens = h[b][amask[b].cpu()]  # [T_valid, D]
                all_states[l].append(valid_tokens.mean(0).numpy())  # [D]
        if (bi+1) % 10 == 0:
            print(f"  [{time.time()-t0:.0f}s] batch {bi+1}/{len(batches)}")

    # ---- Compute scaffold PCA from SCAFFOLD_LAYER ----
    print(f"[{time.time()-t0:.0f}s] Computing scaffold directions (PCA, layer={SCAFFOLD_LAYER})...")
    X_scaffold = np.stack(all_states[SCAFFOLD_LAYER])  # [N, D]
    pca = PCA(n_components=N_SCAFFOLD, random_state=SEED)
    pca.fit(X_scaffold)
    scaffold_dirs  = pca.components_  # [N_SCAFFOLD, D]
    explained_var  = pca.explained_variance_ratio_

    # ---- Project all layers onto scaffold ----
    # coords[l] = [N, N_SCAFFOLD]
    print(f"[{time.time()-t0:.0f}s] Projecting all layers onto scaffold...")
    coords = {}
    for l in range(NL):
        X_l = np.stack(all_states[l])           # [N, D]
        X_l_centered = X_l - X_l.mean(axis=0)
        coords[l] = X_l_centered @ scaffold_dirs.T   # [N, N_SCAFFOLD]

    # ---- Compute separability per layer ----
    domains = ["wiki", "math", "code", "fact"]
    sep_per_layer = []
    diverge_layer = None

    for l in range(NL):
        C = coords[l]  # [N, N_SCAFFOLD]
        centroids = {}
        withinvar = {}
        for dom_name, dom_id in domain2idx.items():
            mask = label_ids == dom_id
            C_dom = C[mask]
            centroids[dom_name] = C_dom.mean(axis=0)
            withinvar[dom_name] = np.mean(np.sum((C_dom - centroids[dom_name])**2, axis=1))

        # inter-centroid distances (mean over pairs)
        pairs = [(d1, d2) for i, d1 in enumerate(domains) for d2 in domains[i+1:]]
        inter_dists = [np.linalg.norm(centroids[d1] - centroids[d2]) for d1, d2 in pairs]
        mean_inter  = np.mean(inter_dists)
        mean_within = np.mean([np.sqrt(withinvar[d]) for d in domains])
        sep = mean_inter / (mean_within + 1e-8)
        sep_per_layer.append(float(sep))

        if diverge_layer is None and sep > 2.0:
            diverge_layer = l

    max_sep       = float(max(sep_per_layer))
    peak_layer    = int(np.argmax(sep_per_layer))

    # ---- Compute per-domain purity in scaffold space at peak layer ----
    # Use simple nearest-centroid classification
    C_peak = coords[peak_layer]
    centroid_mat = np.stack([np.mean(C_peak[label_ids == d], axis=0)
                             for d in range(4)])  # [4, N_SCAFFOLD]
    dists_to_cent = np.stack([
        np.linalg.norm(C_peak - centroid_mat[d], axis=1) for d in range(4)
    ], axis=1)  # [N, 4]
    pred_labels = np.argmin(dists_to_cent, axis=1)
    nn_accuracy = float((pred_labels == label_ids).mean())

    # ---- Variance explained ----
    total_var_explained = float(sum(explained_var))

    # ---- Verdict ----
    if max_sep > PASS_SEP and PASS_SEP > 0 and 6 <= peak_layer <= 22:
        verdict = (
            f"SCAFFOLD_FLOW_CONFIRMED: max_separability={max_sep:.3f} at layer={peak_layer}. "
            f"Task types follow distinct paths through scaffold space."
        )
    elif max_sep > KILL_SEP:
        verdict = (
            f"PARTIAL: max_sep={max_sep:.3f} in [{KILL_SEP},{PASS_SEP}). "
            f"Some task separation in scaffold space but weak."
        )
    else:
        verdict = (
            f"NULL: max_sep={max_sep:.3f} < {KILL_SEP}. "
            "No systematic task-specific flow paths. Scaffold-and-Flow model falsified for Qwen3-0.6B."
        )

    results = {
        "model":                   MODEL_ID,
        "n_layers":                NL,
        "d_model":                 D,
        "n_contexts":              N,
        "n_scaffold_dirs":         N_SCAFFOLD,
        "scaffold_layer":          SCAFFOLD_LAYER,
        "explained_variance_top30": total_var_explained,
        "sep_per_layer":           sep_per_layer,
        "max_separability":        max_sep,
        "peak_layer":              peak_layer,
        "diverge_layer":           diverge_layer,
        "nn_accuracy_at_peak":     nn_accuracy,
        "verdict":                 verdict,
        "elapsed_s":               time.time() - t0,
    }

    out_path = RESULTS / "genome_112_scaffold_flow.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.time()-t0:.0f}s] Results -> {out_path}")
    print(f"Verdict: {verdict}")
    print(f"max_sep={max_sep:.3f} at layer={peak_layer}, nn_acc={nn_accuracy:.3f}, diverge_layer={diverge_layer}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
