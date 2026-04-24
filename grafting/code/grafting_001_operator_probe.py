"""
grafting_001_operator_probe.py

Kill test: is the layer-wise transition operator T_l shared across trained text models?

Codex hypothesis (session 2026-04-24): trained models share not just the spectrum
of activations but the transition law z_{l+1} approx T_l(z_l, context), where z_l
are activations projected onto the shared semantic basis B_l. If T_l is shared,
we can compile it directly (no gradient descent) and install it into a new model.

If T_l is NOT shared across models -> direction dies here.

Method:
1. Load Qwen3-0.6B, DeepSeek-R1-Distill-1.5B, BERT-base-uncased on n C4 samples
2. Extract all hidden states per layer (seq_mean pooled)
3. Compute shared basis B_l: average top-30 left singular vectors across models
   (aligned via Procrustes to a reference model before averaging)
4. For each model, project activations: z_l = B_l.T @ h_l  (shape: n x 30)
5. Fit T_l: least-squares z_{l+1} = z_l @ T_l.T  (30 x 30 operator)
6. Compare T_l across model pairs: Frobenius distance after Procrustes alignment,
   cosine similarity of flattened operators
7. Compare to shuffled-model baseline (T_l of random model pairs)

Kill criterion: mean pairwise T_l similarity <= 2x shuffled baseline -> direction dead.
Pass criterion: mean pairwise T_l similarity >= 5x shuffled baseline -> operator is shared.

Hardware: RTX 5090, 22GB VRAM, Windows, num_workers=0, pin_memory=False
Expected runtime: ~20 min for n=2000
"""

import json
import pathlib
import sys
import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset

ROOT = pathlib.Path(__file__).parent.parent.parent
RESULTS = pathlib.Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_SAMPLES = 2000
N_BASIS = 30
SEQ_LEN = 128
DEPTH_BAND = (0.35, 0.75)

MODELS = {
    "qwen3": ("Qwen/Qwen3-0.6B", "causal"),
    "deepseek": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "causal"),
    "bert": ("google-bert/bert-base-uncased", "encoder"),
}


def load_texts(n, seed=42):
    """Load from locally cached wikitext-103 — no network required."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n * 3, len(ds)), replace=False)
    texts = []
    for i in indices:
        t = ds[int(i)]["text"].strip()
        if len(t) >= 80:
            texts.append(t[:512])
        if len(texts) >= n:
            break
    return texts


def extract_hidden_states(model_name, model_type, texts, seq_len, device):
    """Return dict: layer_idx -> (n, hidden) numpy array (seq_mean pooled)."""
    print(f"  Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, output_hidden_states=True,
            trust_remote_code=True
        ).to(device).eval()
    else:
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16, output_hidden_states=True,
            trust_remote_code=True
        ).to(device).eval()

    n_layers = model.config.num_hidden_layers
    all_states = {i: [] for i in range(n_layers + 1)}

    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=seq_len).to(device)
            out = model(**enc)
            hs = out.hidden_states  # tuple: (n_layers+1) x (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            for layer_idx, h in enumerate(hs):
                pooled = (h * mask).sum(1) / mask.sum(1)
                all_states[layer_idx].append(pooled.float().cpu().numpy())

    del model
    torch.cuda.empty_cache()

    return {k: np.concatenate(v, axis=0) for k, v in all_states.items()}, n_layers


def to_stimulus_space(X, n_basis):
    """Project (n_samples, hidden) activations to (n_samples, n_basis) stimulus space.

    Works in left-singular-vector space so dimensionality is n_samples x n_basis,
    identical across all models regardless of their hidden_size.
    Scale by singular values so magnitude information is preserved.
    """
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_basis] * S[:n_basis]


def align_to_reference(z, z_ref):
    """Procrustes-align z to z_ref (both n x k). Returns aligned z."""
    R, _ = orthogonal_procrustes(z, z_ref)
    return z @ R


def fit_transition_operator(z_l, z_l1):
    """Fit T: z_{l+1} approx z_l @ T.T via least squares. Returns T (n_basis x n_basis)."""
    T, _, _, _ = np.linalg.lstsq(z_l, z_l1, rcond=None)
    return T.T


def operator_similarity(T1, T2):
    """Cosine similarity of flattened operators after Procrustes alignment."""
    R, _ = orthogonal_procrustes(T1, T2)
    T1_aligned = T1 @ R
    flat1 = T1_aligned.ravel()
    flat2 = T2.ravel()
    cos = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2) + 1e-9)
    return float(cos)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading wikitext texts (cached)...")
    texts = load_texts(N_SAMPLES, SEED)
    print(f"  {len(texts)} texts loaded")

    states_per_model = {}
    for key, (model_name, model_type) in MODELS.items():
        print(f"\nExtracting: {key} ({model_name})")
        states, n_layers = extract_hidden_states(model_name, model_type, texts, SEQ_LEN, DEVICE)
        states_per_model[key] = (states, n_layers)
        print(f"  {n_layers} layers, hidden states extracted")

    results = {"n_samples": N_SAMPLES, "n_basis": N_BASIS, "depth_band": DEPTH_BAND,
               "models": list(MODELS.keys()), "layers": {}}

    reference_key = "qwen3"
    model_keys = list(MODELS.keys())

    n_layers_ref = states_per_model[reference_key][1]
    band_layers = [i for i in range(n_layers_ref)
                   if DEPTH_BAND[0] <= i / n_layers_ref <= DEPTH_BAND[1]]
    print(f"\nDepth band {DEPTH_BAND}: {len(band_layers)} layers in reference model")

    layer_sims = []

    for layer_idx in band_layers:
        print(f"\nLayer {layer_idx}/{n_layers_ref} (depth {layer_idx/n_layers_ref:.2f})")

        # Get reference z_l for alignment
        ref_states, _ = states_per_model[reference_key]
        z_ref = to_stimulus_space(ref_states[layer_idx], N_BASIS)

        operators = {}
        for key, (states, n_layers) in states_per_model.items():
            scaled_idx = int(layer_idx / n_layers_ref * n_layers)
            scaled_idx_next = min(scaled_idx + 1, n_layers)

            # Work in stimulus space — same dimensionality for all models
            z_l = to_stimulus_space(states[scaled_idx], N_BASIS)
            z_l1 = to_stimulus_space(states[scaled_idx_next], N_BASIS)

            # Align to reference so operators are comparable
            if key != reference_key:
                z_l = align_to_reference(z_l, z_ref)
                z_l_next_ref = to_stimulus_space(ref_states[scaled_idx_next], N_BASIS)
                z_l1 = align_to_reference(z_l1, z_l_next_ref)

            T = fit_transition_operator(z_l, z_l1)
            operators[key] = T

        pairs = []
        for i, k1 in enumerate(model_keys):
            for k2 in model_keys[i + 1:]:
                sim = operator_similarity(operators[k1], operators[k2])
                pairs.append({"pair": f"{k1}_vs_{k2}", "similarity": sim})
                print(f"  {k1} vs {k2}: T_l similarity = {sim:.4f}")

        mean_sim = np.mean([p["similarity"] for p in pairs])
        print(f"  Mean pairwise similarity: {mean_sim:.4f}")

        results["layers"][str(layer_idx)] = {
            "depth": layer_idx / n_layers_ref,
            "pairs": pairs,
            "mean_similarity": mean_sim,
        }
        layer_sims.append(mean_sim)

    shuffled_sims = []
    print("\nComputing shuffled baseline...")
    for layer_idx in band_layers[:3]:
        ref_states, _ = states_per_model[reference_key]
        z_ref = to_stimulus_space(ref_states[layer_idx], N_BASIS)
        operators = {}
        for key, (states, n_layers) in states_per_model.items():
            scaled_idx = int(layer_idx / n_layers_ref * n_layers)
            scaled_idx_next = min(scaled_idx + 1, n_layers)
            z_l = to_stimulus_space(states[scaled_idx], N_BASIS)
            z_l1 = to_stimulus_space(states[scaled_idx_next], N_BASIS)
            if key != reference_key:
                z_l = align_to_reference(z_l, z_ref)
                z_l_next_ref = to_stimulus_space(ref_states[scaled_idx_next], N_BASIS)
                z_l1 = align_to_reference(z_l1, z_l_next_ref)
            idx = np.random.permutation(len(z_l))
            z_l = z_l[idx]
            T = fit_transition_operator(z_l, z_l1)
            operators[key] = T
        for i, k1 in enumerate(model_keys):
            for k2 in model_keys[i + 1:]:
                shuffled_sims.append(operator_similarity(operators[k1], operators[k2]))

    shuffled_mean = float(np.mean(shuffled_sims))
    trained_mean = float(np.mean(layer_sims))
    ratio = trained_mean / (shuffled_mean + 1e-9)

    results["summary"] = {
        "mean_trained_sim": trained_mean,
        "mean_shuffled_sim": shuffled_mean,
        "ratio": ratio,
        "n_band_layers": len(band_layers),
    }

    if ratio >= 5.0:
        verdict = "PASS: transition operator is shared (>= 5x shuffled)"
    elif ratio >= 2.0:
        verdict = "WEAK: operator partially shared (2-5x shuffled)"
    else:
        verdict = "KILL: operator NOT shared (<= 2x shuffled) -- direction dead"

    results["verdict"] = verdict
    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"Trained mean sim: {trained_mean:.4f} | Shuffled: {shuffled_mean:.4f} | Ratio: {ratio:.1f}x")
    print(f"{'='*60}")

    out_path = RESULTS / "grafting_001_operator_probe.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
