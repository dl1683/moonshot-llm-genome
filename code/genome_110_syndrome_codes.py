"""
genome_110_syndrome_codes.py

Syndrome Codes: does the model systematically repair controlled
hidden-state corruptions as they propagate forward through layers?

Hypothesis (Codex mental model 4): capability may be organized like an
error-correcting code. When the residual stream drifts off a valid
codeword, downstream layers detect and reconstruct the correct state
from distributed redundancies.

Protocol
--------
For each injection layer l_inj in [0, n_layers-1]:
  For each epsilon in EPSILONS:
    For each random unit direction v in DIRECTIONS:
      - Run forward pass with delta = eps*v added at l_inj output.
      - At each l_meas > l_inj, compute:
            repair_frac = 1 - mean_token_norm(corr[l_meas] - clean[l_meas]) / eps
        (0 = no repair, 1 = perfect repair, <0 = amplification)
      - Compute KL(clean_logits || corr_logits) at final layer.
- Average repair_frac over sequences and directions.

Pass:  >= 3 (l_inj, l_meas) pairs with mean repair_frac > 0.50
Kill:  max repair_frac < 0.20 across all pairs -> syndrome code model falsified

Results: results/genome_110_syndrome_codes.json
"""

import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED     = 42

N_SEQS   = 100
SEQ_LEN  = 64
BATCH    = 8
N_DIRS   = 5
EPSILONS = [0.1, 0.5, 1.0, 2.0]

PASS_THRESH = 0.50
KILL_THRESH = 0.20


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
# Hook helpers
# ---------------------------------------------------------------------------

def get_h(layer_out):
    """Extract hidden-state tensor from a decoder-layer forward output."""
    return layer_out[0] if isinstance(layer_out, tuple) else layer_out


def set_h(layer_out, new_h):
    """Return layer output with hidden states replaced by new_h."""
    if isinstance(layer_out, tuple):
        return (new_h,) + layer_out[1:]
    return new_h


def make_clean_hook(storage, layer_idx):
    def hook(module, inp, out):
        storage[layer_idx] = get_h(out).detach().float().cpu()
    return hook


def make_inject_hook(eps, v):
    def hook(module, inp, out):
        h = get_h(out)
        return set_h(out, h + eps * v.to(h.dtype))
    return hook


def make_record_hook(storage, layer_idx):
    def hook(module, inp, out):
        storage[layer_idx] = get_h(out).detach().float().cpu()
    return hook


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print(f"[0s] Loading texts...")
    texts = load_texts()
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    model.eval()

    NL = len(model.model.layers)
    D  = model.config.hidden_size
    NE = len(EPSILONS)

    print(f"[{time.time()-t0:.0f}s] {MODEL_ID}: {NL} layers, d={D}")
    print(f"  {N_SEQS} seqs | {N_DIRS} dirs | epsilons={EPSILONS} | batch={BATCH}")
    print(f"  Total forward passes: ~{len(range(0, N_SEQS, BATCH)) * (1 + NL*NE*N_DIRS)}")

    # Pre-generate fixed unit directions (same across all batches for fair averaging)
    torch.manual_seed(SEED)
    dirs = []
    for _ in range(N_DIRS):
        v = torch.randn(1, 1, D, device=DEVICE, dtype=torch.float32)
        dirs.append((v / v.norm()).to(torch.bfloat16))

    batches = [texts[i:i + BATCH] for i in range(0, len(texts), BATCH)]
    NB = len(batches)

    # Accumulators: sum over (batch, dir) pairs
    # R_acc[l_inj, l_meas, eps_idx] = sum of repair_fracs
    R_acc  = np.zeros((NL, NL, NE), dtype=np.float64)
    KL_acc = np.zeros((NL, NE),     dtype=np.float64)

    for bi, batch in enumerate(batches):
        enc   = tok(batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=SEQ_LEN).to(DEVICE)
        amask = enc["attention_mask"].bool()   # [B, T]

        # ---- Clean forward pass (once per batch) ----
        ch   = {}
        chks = [
            model.model.layers[l].register_forward_hook(make_clean_hook(ch, l))
            for l in range(NL)
        ]
        with torch.no_grad():
            clean_logits = model(**enc, use_cache=False).logits.detach().float().cpu()
        for h in chks:
            h.remove()

        # ---- Corrupted passes ----
        for l_inj in range(NL):
            for ei, eps in enumerate(EPSILONS):
                for d_vec in dirs:
                    crh  = {}
                    hks2 = [
                        model.model.layers[l_inj].register_forward_hook(
                            make_inject_hook(eps, d_vec)
                        )
                    ]
                    for l_m in range(l_inj + 1, NL):
                        hks2.append(
                            model.model.layers[l_m].register_forward_hook(
                                make_record_hook(crh, l_m)
                            )
                        )

                    with torch.no_grad():
                        corr_logits = model(**enc, use_cache=False).logits.detach().float().cpu()
                    for h in hks2:
                        h.remove()

                    # Repair fraction at each subsequent layer
                    for l_m in range(l_inj + 1, NL):
                        if l_m not in crh:
                            continue
                        delta     = crh[l_m] - ch[l_m]             # [B, T, D]
                        tok_norms = delta.norm(dim=-1)               # [B, T]
                        mean_norm = tok_norms[amask.cpu()].mean().item()
                        R_acc[l_inj, l_m, ei] += 1.0 - mean_norm / eps

                    # KL divergence at final logits
                    B, T, V = clean_logits.shape
                    mf      = amask.cpu().view(-1)
                    p_c  = F.softmax(clean_logits.view(-1, V)[mf], dim=-1).clamp(min=1e-8)
                    p_cr = F.softmax(corr_logits.view(-1, V)[mf], dim=-1).clamp(min=1e-8)
                    kl   = (p_c * (p_c.log() - p_cr.log())).sum(-1).mean().item()
                    KL_acc[l_inj, ei] += kl

        elapsed = time.time() - t0
        eta     = elapsed / (bi + 1) * (NB - bi - 1)
        print(f"  [{elapsed:.0f}s] batch {bi+1}/{NB} | ETA {eta:.0f}s")

    # Normalize by (NB * N_DIRS)
    denom   = NB * N_DIRS
    R_mean  = R_acc  / denom   # [NL, NL, NE]
    KL_mean = KL_acc / denom   # [NL, NE]

    # ---- Analysis ----
    R_eps_mean = R_mean.mean(axis=2)    # [NL, NL] averaged over epsilons
    max_repair = float(R_eps_mean.max())
    max_idx    = np.unravel_index(R_eps_mean.argmax(), R_eps_mean.shape)
    n_above_50 = int((R_eps_mean > PASS_THRESH).sum())
    n_above_20 = int((R_eps_mean > KILL_THRESH).sum())

    # Mean repair by distance (how many layers after injection)
    repair_by_dist = []
    for dist in range(1, NL):
        vals = [R_eps_mean[l, l + dist] for l in range(NL - dist)]
        repair_by_dist.append(float(np.mean(vals)))

    # Repair profile at each injection layer (mean over l_meas, eps)
    repair_per_inject = R_eps_mean.mean(axis=1).tolist()

    # Verdict
    if n_above_50 >= 3:
        verdict = (
            f"SYNDROME_CODE_CONFIRMED: {n_above_50} pairs show >{PASS_THRESH:.0%} repair. "
            f"Max={max_repair:.3f} at l_inj={max_idx[0]}, l_meas={max_idx[1]}"
        )
    elif max_repair >= KILL_THRESH:
        verdict = (
            f"PARTIAL: max repair {max_repair:.3f} in [{KILL_THRESH:.0%},{PASS_THRESH:.0%}). "
            f"{n_above_20} pairs above {KILL_THRESH:.0%}."
        )
    else:
        verdict = (
            f"NULL: max repair {max_repair:.3f} < {KILL_THRESH:.0%}. "
            "No systematic repair. Syndrome code model falsified for Qwen3-0.6B."
        )

    results = {
        "model":                   MODEL_ID,
        "n_layers":                NL,
        "d_model":                 D,
        "n_seqs":                  N_SEQS,
        "n_directions":            N_DIRS,
        "epsilons":                EPSILONS,
        "repair_matrix_eps_mean":  R_eps_mean.tolist(),
        "repair_matrix_per_eps": {
            str(eps): R_mean[:, :, ei].tolist()
            for ei, eps in enumerate(EPSILONS)
        },
        "kl_per_linj_per_eps": {
            str(eps): KL_mean[:, ei].tolist()
            for ei, eps in enumerate(EPSILONS)
        },
        "repair_by_distance":      repair_by_dist,
        "repair_per_inject_layer": repair_per_inject,
        "max_repair":              max_repair,
        "max_repair_l_inj":        int(max_idx[0]),
        "max_repair_l_meas":       int(max_idx[1]),
        "n_pairs_above_50pct":     n_above_50,
        "n_pairs_above_20pct":     n_above_20,
        "verdict":                 verdict,
        "elapsed_s":               time.time() - t0,
    }

    out_path = RESULTS / "genome_110_syndrome_codes.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.time()-t0:.0f}s] Results -> {out_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
