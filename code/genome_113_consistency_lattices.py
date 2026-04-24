"""
genome_113_consistency_lattices.py

Consistency Lattices: does Qwen3-0.6B store knowledge as a dense lattice of
soft compatibility constraints among latent directions?

Hypothesis (Codex mental model 3): a competent state satisfies many soft
constraints simultaneously. "Capability" is a region where latent variable
interactions are mutually consistent. The evidence for this structure is
superadditivity: simultaneously ablating a PAIR of directions causes MORE
damage than the sum of each individual ablation.

Protocol
--------
For N_SEQS sequences from wikitext:
  1. Collect clean logits (baseline NLL).
  2. For each pair (i, j) in a SAMPLE of direction pairs:
     a. Ablate direction i alone: project out e_i from each layer's residual stream.
        -> NLL_i
     b. Ablate direction j alone: project out e_j.
        -> NLL_j
     c. Ablate both: project out both e_i, e_j.
        -> NLL_ij
     d. Synergy = NLL_ij - NLL_clean - (NLL_i - NLL_clean) - (NLL_j - NLL_clean)
                = NLL_ij - NLL_i - NLL_j + NLL_clean
        > 0  => superadditive damage (the pair is a constraint / compatibility relation)
        ≈ 0  => additive (directions are independent)
        < 0  => subadditive (redundant / substitute directions)

Directions: top-20 PCA directions of layer-14 activations (on-the-fly).
Pairs: sample min(C(20,2), 100) = 100 pairs.

Pass:  >20% of pairs show synergy > 0.05 nats (superadditive damage)
Kill:  mean synergy < 0.01 nats -> directions are independent, no lattice structure

Also report:
- Distribution of synergy values
- Correlation of synergy with (NLL_i + NLL_j) magnitude
- Spatial pattern: are synergistic pairs among nearby vs distant PCA components?

Results: results/genome_113_consistency_lattices.json
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

N_SEQS      = 80    # smaller: many forward passes per sequence
SEQ_LEN     = 64
BATCH       = 8
N_DIRS      = 20    # top-k PCA directions to probe
N_PAIRS     = 100   # how many (i,j) pairs to test
SCAFFOLD_LAYER = 14 # layer for PCA scaffold
ABLATE_ALL_LAYERS = True  # ablate across ALL layers (not just one)

PASS_THRESH = 0.05   # nats synergy
KILL_THRESH = 0.01   # nats mean synergy


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_texts(n=N_SEQS, seed=SEED):
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


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def make_ablate_hook(dirs_to_remove):
    """
    Project out one or more unit directions from the residual stream at every layer.
    dirs_to_remove: [K, D] float32 tensor (unit vectors).
    """
    dirs = dirs_to_remove  # [K, D]

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out   # [B, T, D]
        h = h.float()
        # Project out each direction: h -= (h @ d) * d for each d
        for k in range(dirs.shape[0]):
            d = dirs[k].to(h.device)        # [D]
            proj = (h @ d).unsqueeze(-1) * d  # [B, T, D]
            h = h - proj
        h = h.to(out[0].dtype if isinstance(out, tuple) else out.dtype)
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h
    return hook


def run_with_ablation(model, enc, dirs_to_remove, n_layers):
    """Run forward pass with specified directions projected out at every layer."""
    if dirs_to_remove.shape[0] == 0:
        # Clean run
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits.detach().float().cpu()
        return logits

    hooks = [
        model.model.layers[l].register_forward_hook(
            make_ablate_hook(dirs_to_remove)
        )
        for l in range(n_layers)
    ]
    with torch.no_grad():
        logits = model(**enc, use_cache=False).logits.detach().float().cpu()
    for h in hooks:
        h.remove()
    return logits


def logits_to_nll(logits, enc_ids, amask):
    """Compute mean NLL (cross-entropy) over valid tokens."""
    B, T, V = logits.shape
    # shift: predict token t+1 from position t
    lp   = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
    tgt  = enc_ids[:, 1:].cpu()                        # [B, T-1]
    mask = amask[:, 1:].cpu().bool()                    # [B, T-1]
    nll  = -lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    return float(nll[mask].mean().item())


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

    # ---- Load texts ----
    print(f"[{time.time()-t0:.0f}s] Loading texts...")
    texts   = load_texts()
    batches = [texts[i:i+BATCH] for i in range(0, len(texts), BATCH)]

    # ---- Step 1: collect layer-14 activations to compute scaffold PCA ----
    print(f"[{time.time()-t0:.0f}s] Collecting layer-{SCAFFOLD_LAYER} activations for PCA...")
    l14_vecs = []
    hooks_pca = []

    def make_pca_hook(storage):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            storage.append(h.detach().float().cpu())
        return hook

    for bi, batch_texts in enumerate(batches):
        enc   = tok(batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=SEQ_LEN).to(DEVICE)
        amask = enc["attention_mask"].bool()
        buf   = []
        hk    = model.model.layers[SCAFFOLD_LAYER].register_forward_hook(
            make_pca_hook(buf)
        )
        with torch.no_grad():
            model(**enc, use_cache=False)
        hk.remove()

        h14 = buf[0]   # [B, T, D]
        for b in range(h14.shape[0]):
            valid = h14[b][amask[b].cpu()].mean(0)  # [D]
            l14_vecs.append(valid.numpy())

    X = np.stack(l14_vecs)  # [N, D]
    pca = PCA(n_components=N_DIRS, random_state=SEED)
    pca.fit(X)
    dirs_np  = pca.components_   # [N_DIRS, D] unit vectors
    dirs_t   = torch.from_numpy(dirs_np).float()  # [N_DIRS, D]
    print(f"  [{time.time()-t0:.0f}s] PCA done. var_explained={pca.explained_variance_ratio_.sum():.3f}")

    # ---- Step 2: pick pairs ----
    all_pairs = [(i, j) for i in range(N_DIRS) for j in range(i+1, N_DIRS)]
    np.random.seed(SEED)
    np.random.shuffle(all_pairs)
    pairs = all_pairs[:N_PAIRS]
    print(f"[{time.time()-t0:.0f}s] Testing {len(pairs)} direction pairs ({N_DIRS} dirs)")

    # ---- Step 3: ablation sweeps ----
    empty_dirs = torch.zeros(0, D)

    nll_clean_list  = []
    nll_i_list      = []   # indexed by unique dir idx
    nll_j_list      = []
    nll_ij_list     = []

    # Cache single-direction ablation NLLs (avoid recomputing)
    single_nll_cache = {}  # dir_idx -> list of per-batch NLL

    # First: compute clean NLL and cache single-dir ablations for all dirs in pairs
    unique_dirs = list({d for pair in pairs for d in pair})
    n_batches   = len(batches)

    # Clean pass per batch
    print(f"[{time.time()-t0:.0f}s] Clean forward passes...")
    batch_clean_nlls = []
    batch_enc_cache  = []
    for bi, batch_texts in enumerate(batches):
        enc   = tok(batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=SEQ_LEN).to(DEVICE)
        amask = enc["attention_mask"].bool()
        batch_enc_cache.append((enc, amask))
        logits = run_with_ablation(model, enc, empty_dirs, NL)
        nll    = logits_to_nll(logits, enc["input_ids"], amask)
        batch_clean_nlls.append(nll)
    mean_nll_clean = float(np.mean(batch_clean_nlls))
    print(f"  clean NLL = {mean_nll_clean:.4f}")

    # Single-dir ablation for each unique direction
    print(f"[{time.time()-t0:.0f}s] Single-direction ablations ({len(unique_dirs)} dirs)...")
    for di, d_idx in enumerate(unique_dirs):
        d_dirs = dirs_t[d_idx:d_idx+1]  # [1, D]
        batch_nlls = []
        for bi, (enc, amask) in enumerate(batch_enc_cache):
            logits = run_with_ablation(model, enc, d_dirs, NL)
            nll    = logits_to_nll(logits, enc["input_ids"], amask)
            batch_nlls.append(nll)
        single_nll_cache[d_idx] = float(np.mean(batch_nlls))
        if (di+1) % 5 == 0:
            print(f"  [{time.time()-t0:.0f}s] single-ablation {di+1}/{len(unique_dirs)}")

    # Pairwise ablations
    print(f"[{time.time()-t0:.0f}s] Pairwise ablations ({len(pairs)} pairs)...")
    synergies = []
    pair_results = []

    for pi, (i, j) in enumerate(pairs):
        d_pair = dirs_t[[i, j]]  # [2, D]
        batch_nlls = []
        for enc, amask in batch_enc_cache:
            logits = run_with_ablation(model, enc, d_pair, NL)
            nll    = logits_to_nll(logits, enc["input_ids"], amask)
            batch_nlls.append(nll)
        nll_ij = float(np.mean(batch_nlls))
        nll_i  = single_nll_cache[i]
        nll_j  = single_nll_cache[j]

        # Synergy = NLL_ij - NLL_i - NLL_j + NLL_clean
        synergy = nll_ij - nll_i - nll_j + mean_nll_clean
        synergies.append(synergy)
        pair_results.append({
            "i": i, "j": j,
            "nll_clean": mean_nll_clean,
            "nll_i": nll_i, "nll_j": nll_j, "nll_ij": nll_ij,
            "synergy": synergy,
        })

        if (pi+1) % 20 == 0:
            print(f"  [{time.time()-t0:.0f}s] pair {pi+1}/{len(pairs)} | "
                  f"mean_synergy={np.mean(synergies):.4f}")

    synergies_arr   = np.array(synergies)
    mean_synergy    = float(synergies_arr.mean())
    max_synergy     = float(synergies_arr.max())
    min_synergy     = float(synergies_arr.min())
    frac_above_pass = float((synergies_arr > PASS_THRESH).mean())
    n_super         = int((synergies_arr > PASS_THRESH).sum())
    n_sub           = int((synergies_arr < -PASS_THRESH).sum())

    # ---- Verdict ----
    if frac_above_pass > 0.20:
        verdict = (
            f"CONSISTENCY_LATTICE_CONFIRMED: {frac_above_pass:.1%} of pairs show "
            f"synergy>{PASS_THRESH} nats. mean_synergy={mean_synergy:.4f}. "
            f"Soft compatibility constraints exist between latent directions."
        )
    elif mean_synergy > KILL_THRESH:
        verdict = (
            f"PARTIAL: mean_synergy={mean_synergy:.4f} in ({KILL_THRESH},{PASS_THRESH}). "
            f"{n_super} superadditive pairs, {n_sub} subadditive pairs."
        )
    else:
        verdict = (
            f"NULL: mean_synergy={mean_synergy:.4f} <= {KILL_THRESH}. "
            "Directions are approximately independent. "
            "Consistency Lattice model falsified for Qwen3-0.6B."
        )

    results = {
        "model":               MODEL_ID,
        "n_layers":            NL,
        "d_model":             D,
        "n_seqs":              len(texts),
        "n_dirs":              N_DIRS,
        "n_pairs_tested":      len(pairs),
        "nll_clean":           mean_nll_clean,
        "mean_synergy":        mean_synergy,
        "max_synergy":         max_synergy,
        "min_synergy":         min_synergy,
        "frac_above_pass":     frac_above_pass,
        "n_superadditive":     n_super,
        "n_subadditive":       n_sub,
        "synergies":           synergies,
        "pair_results":        pair_results[:20],   # keep top-20 for inspection
        "single_dir_nlls":     {str(d): v for d, v in single_nll_cache.items()},
        "verdict":             verdict,
        "elapsed_s":           time.time() - t0,
    }

    out_path = RESULTS / "genome_113_consistency_lattices.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.time()-t0:.0f}s] Results -> {out_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
