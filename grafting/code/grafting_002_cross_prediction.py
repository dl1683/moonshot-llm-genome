"""
grafting_002_cross_prediction.py

Proper kill test for shared transition operators.

grafting_001 showed 0.98 trained T_l similarity but shuffled baseline was 0.61
(Procrustes alignment artifact). This test removes the artifact entirely.

Protocol:
  Train split (n=1500): fit T_l^(A) from model A's activations
  Test split  (n=500):  use T_l^(A) to PREDICT model B's z_{l+1} from model B's z_l

If T_l is truly shared: cross-prediction R^2 should be close to within-model R^2.
If T_l is not shared:  cross-prediction R^2 should be near zero.

Baselines:
  - within-model R^2: fit and predict on same model (upper bound)
  - mean baseline R^2: predict z_{l+1} = mean(z_{l+1}_train) (lower bound = 0)
  - random-init model T_l: does an untrained model share operators with trained ones?

Kill: cross_R2 / within_R2 < 0.3 for all pairs -> operators not shared
Pass: cross_R2 / within_R2 > 0.7 for majority of depth-band layers -> operators shared
"""

import json
import pathlib
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
N_TRAIN = 1500
N_TEST = 500
N_SAMPLES = N_TRAIN + N_TEST
N_BASIS = 30
SEQ_LEN = 128
DEPTH_BAND = (0.35, 0.75)

MODELS = {
    "qwen3":    ("Qwen/Qwen3-0.6B",                          "causal"),
    "deepseek": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "causal"),
    "bert":     ("google-bert/bert-base-uncased",              "encoder"),
}


def load_texts(n, seed=42):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n * 4, len(ds)), replace=False)
    texts = []
    for i in indices:
        t = ds[int(i)]["text"].strip()
        if len(t) >= 80:
            texts.append(t[:512])
        if len(texts) >= n:
            break
    return texts


def extract_hidden_states(model_name, model_type, texts, seq_len, device):
    print(f"  Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, output_hidden_states=True,
            trust_remote_code=True
        ).to(device).eval()
    else:
        model = AutoModel.from_pretrained(
            model_name, dtype=torch.float16, output_hidden_states=True,
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
            hs = out.hidden_states
            mask = enc["attention_mask"].unsqueeze(-1).float()
            for layer_idx, h in enumerate(hs):
                pooled = (h * mask).sum(1) / mask.sum(1)
                all_states[layer_idx].append(pooled.float().cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return {k: np.concatenate(v, axis=0) for k, v in all_states.items()}, n_layers


def to_stimulus_space(X, n_basis, fit_on=None):
    """
    Project (n, hidden) -> (n, n_basis) via left SVs fitted on fit_on rows.
    If fit_on is None, fit on all rows (self-projection).
    Returns (z, V) where V is the right SVs for reuse.
    """
    X_fit = X[fit_on] if fit_on is not None else X
    X_fit = X_fit - X_fit.mean(0)
    U_fit, S_fit, Vt_fit = np.linalg.svd(X_fit, full_matrices=False)

    X_proj = X - X_fit.mean(0)
    # Project onto the right singular vectors (consistent projection)
    z = X_proj @ Vt_fit[:n_basis].T  # (n, n_basis)
    return z, Vt_fit[:n_basis]


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(0)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def main():
    np.random.seed(SEED)

    print("Loading texts...")
    texts = load_texts(N_SAMPLES, SEED)
    print(f"  {len(texts)} texts | train={N_TRAIN} test={N_TEST}")

    train_idx = np.arange(N_TRAIN)
    test_idx  = np.arange(N_TRAIN, N_SAMPLES)

    states_per_model = {}
    for key, (model_name, model_type) in MODELS.items():
        print(f"\nExtracting: {key}")
        states, n_layers = extract_hidden_states(model_name, model_type, texts, SEQ_LEN, DEVICE)
        states_per_model[key] = (states, n_layers)
        print(f"  {n_layers} layers extracted")

    n_layers_ref = states_per_model["qwen3"][1]
    band_layers = [i for i in range(n_layers_ref)
                   if DEPTH_BAND[0] <= i / n_layers_ref <= DEPTH_BAND[1]]
    print(f"\nDepth band {DEPTH_BAND}: {len(band_layers)} layers")

    results = {
        "n_train": N_TRAIN, "n_test": N_TEST, "n_basis": N_BASIS,
        "models": list(MODELS.keys()), "layers": {}
    }

    model_keys = list(MODELS.keys())

    for layer_idx in band_layers:
        depth = layer_idx / n_layers_ref
        print(f"\nLayer {layer_idx} (depth {depth:.2f})")

        layer_results = {"depth": depth, "within": {}, "cross": {}}

        # Build projections for each model at this layer (fit on train split)
        projs = {}  # key -> (z_l_all, z_l1_all, Vt_l)
        for key, (states, n_layers) in states_per_model.items():
            sl = int(layer_idx / n_layers_ref * n_layers)
            sl1 = min(sl + 1, n_layers)
            z_l,  Vt_l  = to_stimulus_space(states[sl],  N_BASIS, fit_on=train_idx)
            z_l1, Vt_l1 = to_stimulus_space(states[sl1], N_BASIS, fit_on=train_idx)
            projs[key] = (z_l, z_l1, Vt_l, Vt_l1)

        # Align all test projections to reference (qwen3) using Procrustes on train split
        ref_z_train = projs["qwen3"][0][train_idx]
        ref_z1_train = projs["qwen3"][1][train_idx]

        aligned = {}
        for key in model_keys:
            z_l, z_l1, Vt_l, Vt_l1 = projs[key]
            if key == "qwen3":
                aligned[key] = (z_l, z_l1)
            else:
                R_l,  _ = orthogonal_procrustes(z_l[train_idx],   ref_z_train)
                R_l1, _ = orthogonal_procrustes(z_l1[train_idx],  ref_z1_train)
                aligned[key] = (z_l @ R_l, z_l1 @ R_l1)

        # Within-model: fit T on train, evaluate on test
        operators = {}
        for key in model_keys:
            z_tr  = aligned[key][0][train_idx]
            z1_tr = aligned[key][1][train_idx]
            T, _, _, _ = np.linalg.lstsq(z_tr, z1_tr, rcond=None)
            operators[key] = T.T

            z_te  = aligned[key][0][test_idx]
            z1_te = aligned[key][1][test_idx]
            pred  = z_te @ T
            r2 = r2_score(z1_te, pred)
            layer_results["within"][key] = r2
            print(f"  within {key:10s}: R2={r2:.4f}")

        # Cross-model: use model A's T to predict model B's z_{l+1} on test
        for src in model_keys:
            for tgt in model_keys:
                if src == tgt:
                    continue
                T_src = operators[src]
                z_te_tgt  = aligned[tgt][0][test_idx]
                z1_te_tgt = aligned[tgt][1][test_idx]
                pred = z_te_tgt @ T_src.T
                r2 = r2_score(z1_te_tgt, pred)
                key = f"{src}->{ tgt}"
                layer_results["cross"][key] = r2
                print(f"  cross  {key:20s}: R2={r2:.4f}")

        results["layers"][str(layer_idx)] = layer_results

    # Summary
    all_within = [v for lr in results["layers"].values() for v in lr["within"].values()]
    all_cross  = [v for lr in results["layers"].values() for v in lr["cross"].values()]
    mean_within = float(np.mean(all_within))
    mean_cross  = float(np.mean(all_cross))
    ratio = mean_cross / (mean_within + 1e-9)

    if ratio > 0.7:
        verdict = "PASS: cross-prediction R2 > 70% of within-model -- operators ARE shared"
    elif ratio > 0.3:
        verdict = "WEAK: partial sharing (30-70%) -- investigate further"
    else:
        verdict = "KILL: cross-prediction R2 < 30% of within-model -- operators NOT shared"

    results["summary"] = {
        "mean_within_r2": mean_within,
        "mean_cross_r2": mean_cross,
        "cross_to_within_ratio": ratio,
        "verdict": verdict,
    }

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"Mean within-model R2: {mean_within:.4f}")
    print(f"Mean cross-model R2:  {mean_cross:.4f}")
    print(f"Ratio: {ratio:.3f}")
    print(f"{'='*60}")

    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    out = RESULTS / "grafting_002_cross_prediction.json"
    with open(out, "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
