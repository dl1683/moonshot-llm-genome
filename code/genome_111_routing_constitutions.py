"""
genome_111_routing_constitutions.py

Routing Constitutions: does Qwen3-0.6B route computation through distinct
attention-head coalitions depending on internal state (not just input domain)?

Hypothesis (Codex mental model 2): capability is a stable routing regime --
a repeatable coalition of attention heads that wins control in specific
internal states. The model's 'constitution' determines which heads are active
in which states. Different capabilities = different routing regimes.

Protocol
--------
1. Collect 500 diverse contexts (wikitext + math + code + factual).
2. Run forward pass with output_attentions=True.
3. Record layer-14 residual stream (per-sequence mean as state fingerprint).
4. K-means cluster into K=8 clusters.
5. For each cluster: compute mean attention entropy per head across all layers.
   entropy[h] = -sum(A[h] * log(A[h] + 1e-8)) averaged over query positions.
   High entropy = diffuse/broadcast head. Low entropy = focused/routing head.
6. Build per-cluster head-entropy profile (vector of length n_heads * n_layers).
7. Compute pairwise JS-divergence between cluster profiles (normalize to dist).
8. Check: do clusters show distinct head coalitions?

Pass:  >30% of cluster pairs have JS-div > 0.30 (distinct routing regimes)
Kill:  mean JS-div across all pairs < 0.10 (uniform -- no routing constitution)

Also check: do K-means clusters correlate with input domain (domain purity)?

Results: results/genome_111_routing_constitutions.json
"""

import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT    = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED     = 42

SEQ_LEN      = 64
BATCH        = 4    # smaller batch: output_attentions=True is memory-intensive
K_CLUSTERS   = 8
CLUSTER_LAYER = 14  # mid-depth layer for state fingerprinting

N_WIKI  = 200
N_MATH  = 100
N_CODE  = 100
N_FACT  = 100

PASS_JS = 0.30
KILL_JS = 0.10


# ---------------------------------------------------------------------------
# Diverse context generation
# ---------------------------------------------------------------------------

def load_wikitext(n=N_WIKI, seed=SEED):
    ds  = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    out = []
    for idx in rng.permutation(len(ds)):
        t = ds[int(idx)]["text"].strip()
        if len(t) >= 60:
            out.append(t[:256])
        if len(out) >= n:
            break
    return out, ["wiki"] * len(out)


def make_math_prompts(n=N_MATH, seed=SEED):
    rng = np.random.default_rng(seed)
    templates = [
        "What is {a} + {b}?",
        "Calculate {a} times {b}.",
        "If you have {a} items and add {b} more, how many do you have?",
        "Solve: {a} - {b} = ?",
        "{a} divided by {b} equals?",
        "A store has {a} products. {b} are sold. How many remain?",
    ]
    prompts = []
    for _ in range(n):
        t = rng.choice(templates)
        a = int(rng.integers(2, 200))
        b = int(rng.integers(2, 100))
        prompts.append(t.format(a=a, b=b))
    return prompts, ["math"] * n


def make_code_prompts(n=N_CODE, seed=SEED):
    snippets = [
        "def add(a, b):\n    return a + b\n\nresult = add(3, 4)",
        "import numpy as np\narr = np.zeros((3, 3))\nprint(arr.shape)",
        "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
        "class Animal:\n    def __init__(self, name):\n        self.name = name",
        "x = [1, 2, 3, 4, 5]\nprint(sum(x) / len(x))",
        "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)",
        "with open('file.txt', 'r') as f:\n    data = f.read()",
        "d = {'key': 'value', 'num': 42}\nprint(d.get('num', 0))",
        "import torch\nx = torch.tensor([1.0, 2.0, 3.0])\nprint(x.mean())",
        "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    result = None",
    ]
    rng   = np.random.default_rng(seed)
    out   = [snippets[i % len(snippets)] for i in rng.permutation(n)]
    return out, ["code"] * n


def make_factual_prompts(n=N_FACT, seed=SEED):
    facts = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The speed of light is approximately 299,792 kilometers per second.",
        "The Great Wall of China was built over many centuries.",
        "Albert Einstein developed the theory of relativity.",
        "DNA stands for deoxyribonucleic acid.",
        "The human body has 206 bones.",
        "Shakespeare wrote Hamlet in approximately 1600.",
        "The Amazon river is the largest river by discharge volume.",
        "Jupiter is the largest planet in our solar system.",
        "The Eiffel Tower is located in Paris, France.",
        "Gold has the chemical symbol Au.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Photosynthesis converts sunlight into chemical energy.",
        "The Pythagorean theorem states that a squared plus b squared equals c squared.",
    ]
    rng = np.random.default_rng(seed)
    out = [facts[i % len(facts)] for i in rng.permutation(n)]
    return out, ["fact"] * n


# ---------------------------------------------------------------------------
# JS divergence
# ---------------------------------------------------------------------------

def js_divergence(p, q, eps=1e-8):
    """Jensen-Shannon divergence between two probability vectors."""
    p = np.array(p, dtype=np.float64) + eps
    q = np.array(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # ---- Build diverse context set ----
    print(f"[0s] Building contexts...")
    wiki_texts, wiki_labels = load_wikitext()
    math_texts, math_labels = make_math_prompts()
    code_texts, code_labels = make_code_prompts()
    fact_texts, fact_labels = make_factual_prompts()

    all_texts  = wiki_texts  + math_texts  + code_texts  + fact_texts
    all_labels = wiki_labels + math_labels + code_labels + fact_labels
    N = len(all_texts)
    print(f"  {N} total contexts: {N_WIKI} wiki + {N_MATH} math + {N_CODE} code + {N_FACT} fact")

    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    NL  = len(model.model.layers)
    D   = model.config.hidden_size
    NH  = model.config.num_attention_heads

    print(f"[{time.time()-t0:.0f}s] {MODEL_ID}: {NL} layers, d={D}, n_heads={NH}")

    # ---- Forward passes: collect layer-14 states + attention entropies ----
    batches = [all_texts[i:i + BATCH] for i in range(0, N, BATCH)]
    NB      = len(batches)

    layer14_means   = []  # [N, D] float32 -- state fingerprints for clustering
    # attn_entropies[l][seq_idx] = mean entropy per head for that sequence/layer
    # Shape after collection: [N, NL, NH]
    attn_entropies_all = np.zeros((N, NL, NH), dtype=np.float32)
    seq_counter = 0

    for bi, batch in enumerate(batches):
        enc    = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=SEQ_LEN).to(DEVICE)
        amask  = enc["attention_mask"].bool()  # [B, T]
        B_     = amask.shape[0]

        # Capture layer-14 hidden state
        l14_out = {}
        def make_l14_hook(storage):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                storage["h"] = h.detach().float().cpu()
            return hook
        h14_hook = model.model.layers[CLUSTER_LAYER].register_forward_hook(
            make_l14_hook(l14_out)
        )

        with torch.no_grad():
            out = model(**enc, output_attentions=True, use_cache=False)

        h14_hook.remove()

        # Layer-14 mean over valid tokens per sequence
        h14 = l14_out["h"]  # [B, T, D]
        for b in range(B_):
            valid_mask = amask[b].cpu()
            mean_h = h14[b][valid_mask].mean(dim=0).numpy()  # [D]
            layer14_means.append(mean_h)

        # Attention entropies: out.attentions is a tuple of [B, NH, T, T]
        if out.attentions is not None:
            for l_idx, attn_w in enumerate(out.attentions):
                # attn_w: [B, NH, T, T] float32
                attn_np = attn_w.float().cpu().numpy()
                for b in range(B_):
                    valid_T = int(amask[b].cpu().sum().item())
                    # Entropy per head: mean over query positions of H(row)
                    # Use only valid query positions
                    A = attn_np[b, :, :valid_T, :valid_T]  # [NH, valid_T, valid_T]
                    A = np.clip(A, 1e-8, 1.0)
                    H = -(A * np.log(A)).sum(axis=-1)  # [NH, valid_T]
                    mean_H = H.mean(axis=-1)            # [NH]
                    sidx = seq_counter + b
                    if sidx < N:
                        attn_entropies_all[sidx, l_idx, :] = mean_H.astype(np.float32)

        seq_counter += B_
        if (bi + 1) % 20 == 0 or bi == NB - 1:
            print(f"  [{time.time()-t0:.0f}s] batch {bi+1}/{NB}")

    layer14_means = np.stack(layer14_means[:N], axis=0)  # [N, D]
    print(f"[{time.time()-t0:.0f}s] Forward passes done. layer14_means: {layer14_means.shape}")

    # ---- K-means clustering on layer-14 states ----
    print(f"[{time.time()-t0:.0f}s] K-means K={K_CLUSTERS}...")
    km = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=10)
    cluster_ids = km.fit_predict(layer14_means)

    sil_score = float(silhouette_score(layer14_means, cluster_ids, sample_size=min(N, 500)))
    print(f"  Silhouette score: {sil_score:.4f}")

    # Cluster size distribution
    cluster_sizes = [int((cluster_ids == k).sum()) for k in range(K_CLUSTERS)]

    # Domain purity per cluster
    label_to_int = {"wiki": 0, "math": 1, "code": 2, "fact": 3}
    label_ints   = np.array([label_to_int[l] for l in all_labels])
    cluster_purity = []
    cluster_domain_dist = []
    for k in range(K_CLUSTERS):
        mask = cluster_ids == k
        if mask.sum() == 0:
            cluster_purity.append(0.0)
            cluster_domain_dist.append([0.0, 0.0, 0.0, 0.0])
            continue
        domain_counts = np.bincount(label_ints[mask], minlength=4)
        purity = float(domain_counts.max() / domain_counts.sum())
        dist   = (domain_counts / domain_counts.sum()).tolist()
        cluster_purity.append(purity)
        cluster_domain_dist.append(dist)
    mean_purity = float(np.mean(cluster_purity))

    # ---- Per-cluster attention entropy profiles ----
    # Flatten to [N, NL*NH] for per-cluster mean
    attn_flat = attn_entropies_all.reshape(N, NL * NH)  # [N, NL*NH]

    cluster_profiles = []  # K x (NL*NH)
    for k in range(K_CLUSTERS):
        mask = cluster_ids == k
        if mask.sum() == 0:
            cluster_profiles.append(np.zeros(NL * NH))
        else:
            cluster_profiles.append(attn_flat[mask].mean(axis=0))

    # ---- Pairwise JS-divergence between cluster profiles ----
    # Normalize each profile to a probability distribution (softmax over heads)
    def profile_to_dist(profile):
        e = np.exp(profile - profile.max())
        return e / e.sum()

    js_matrix = np.zeros((K_CLUSTERS, K_CLUSTERS))
    for i in range(K_CLUSTERS):
        for j in range(i + 1, K_CLUSTERS):
            p = profile_to_dist(cluster_profiles[i])
            q = profile_to_dist(cluster_profiles[j])
            jsd = js_divergence(p, q)
            js_matrix[i, j] = jsd
            js_matrix[j, i] = jsd

    # Upper triangle only for statistics
    upper = js_matrix[np.triu_indices(K_CLUSTERS, k=1)]
    mean_js    = float(upper.mean())
    max_js     = float(upper.max())
    frac_above = float((upper > PASS_JS).mean())
    print(f"  JS-div: mean={mean_js:.4f}, max={max_js:.4f}, frac>{PASS_JS:.2f}={frac_above:.3f}")

    # ---- Per-layer analysis: which layers show most routing diversity? ----
    # For each layer, compute mean JS-div between cluster head-entropy distributions
    layer_routing_diversity = []
    for l in range(NL):
        layer_profiles = []
        for k in range(K_CLUSTERS):
            mask = cluster_ids == k
            if mask.sum() == 0:
                layer_profiles.append(np.zeros(NH))
            else:
                layer_profiles.append(attn_entropies_all[mask, l, :].mean(axis=0))
        js_vals = []
        for i in range(K_CLUSTERS):
            for j in range(i + 1, K_CLUSTERS):
                p = profile_to_dist(layer_profiles[i])
                q = profile_to_dist(layer_profiles[j])
                js_vals.append(js_divergence(p, q))
        layer_routing_diversity.append(float(np.mean(js_vals)))

    most_diverse_layer = int(np.argmax(layer_routing_diversity))

    # ---- Verdict ----
    if frac_above >= 0.30:
        verdict = (
            f"ROUTING_CONSTITUTION_CONFIRMED: {frac_above:.1%} of cluster pairs "
            f"show JS-div > {PASS_JS:.2f}. Mean JS={mean_js:.4f}. "
            f"Most diverse routing layer: {most_diverse_layer}."
        )
    elif mean_js >= KILL_JS:
        verdict = (
            f"PARTIAL: mean JS-div {mean_js:.4f} in [{KILL_JS:.2f},{PASS_JS:.2f}). "
            f"Some routing differentiation but below pass threshold."
        )
    else:
        verdict = (
            f"NULL: mean JS-div {mean_js:.4f} < {KILL_JS:.2f}. "
            "Head coalitions are uniform across internal states. "
            "Routing constitution model falsified for Qwen3-0.6B."
        )

    results = {
        "model":                  MODEL_ID,
        "n_layers":               NL,
        "d_model":                D,
        "n_heads":                NH,
        "cluster_layer":          CLUSTER_LAYER,
        "k_clusters":             K_CLUSTERS,
        "n_contexts":             N,
        "context_breakdown":      {"wiki": N_WIKI, "math": N_MATH,
                                   "code": N_CODE, "fact": N_FACT},
        "silhouette_score":       sil_score,
        "cluster_sizes":          cluster_sizes,
        "cluster_purity":         cluster_purity,
        "cluster_domain_dist":    cluster_domain_dist,
        "mean_domain_purity":     mean_purity,
        "js_matrix":              js_matrix.tolist(),
        "mean_js_divergence":     mean_js,
        "max_js_divergence":      max_js,
        "frac_pairs_above_pass":  frac_above,
        "layer_routing_diversity": layer_routing_diversity,
        "most_diverse_layer":     most_diverse_layer,
        "verdict":                verdict,
        "elapsed_s":              time.time() - t0,
    }

    out_path = RESULTS / "genome_111_routing_constitutions.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.time()-t0:.0f}s] Results -> {out_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
