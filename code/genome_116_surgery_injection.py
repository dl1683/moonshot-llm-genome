"""
genome_116_surgery_injection.py

END GOAL: efficient capability transfer from a trained model into an untrained
model without retraining.

This is the FIRST SURGERY EXPERIMENT. genome_115 confirmed that a single
layer-local PCA direction causes catastrophic capability loss when ablated
(layer 5: 4.46 nats, 906x vs random, 57x vs PC2). This experiment tests
the reverse: can INJECTING that direction into a lesioned/untrained recipient
RESTORE capability?

Protocol (Codex to confirm before running — see TODO markers)
-------------------------------------------------------------
Donor:     healthy Qwen/Qwen3-0.6B (trained)
Recipient: [TODO: Codex to specify — lesioned Qwen3-0.6B vs random-init]

Step 1. Extract critical direction:
  - Run donor on fit-split texts (n=200)
  - Fit PCA on layer-5 activations
  - dir_critical = PC1 (the direction with 906x ablation ratio)
  - Extract mean activation of donor at each layer (mu_donor[l])

Step 2. Prepare recipient:
  - [TODO: Codex option A] Lesioned: load Qwen3-0.6B, hook that projects out
    dir_critical at layer 5 (and optionally layers 2-11)
  - [TODO: Codex option B] Random-init: torch.manual_seed(42) random init

Step 3. Injection conditions (Codex to specify subset):
  a. NO_INJECTION: recipient as-is (baseline for lesioned, or random-init NLL)
  b. INJECT_HOOK: add fixed activation hook at layer 5 that adds
     alpha * (donor_mean[5] projected onto dir_critical) to every token
  c. INJECT_SUBSPACE: add fixed hook at layers {2,5,8,11} (all early-layer
     critical layers from genome_115) that adds the per-layer donor direction
  d. INJECT_WEIGHT: directly modify MLP/attention weights at layer 5 to
     encode the critical direction (weight-space surgery, not hook-space)

Step 4. Measure:
  - NLL on eval-split (n=100, disjoint from fit-split), ZERO gradient steps
  - Bootstrap CIs (n=500)
  - Report: NLL_recipient, NLL_donor, NLL_after_injection, gap_closed_%

Pass: injection recovers >= 20% of the lesion gap (NLL_recipient - NLL_donor)
      at ZERO gradient steps with CI excluding zero
Kill: injection recovers < 5% of gap — direction cannot be transplanted

This is the decisive experiment toward the end goal.
Results: results/genome_116_surgery_injection.json
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

N_FIT    = 200
N_EVAL   = 100
SEQ_LEN  = 64
BATCH    = 8

# Layer 5: strongest local critical direction (906x vs random, 4.46 nats)
SURGERY_LAYER = 5
# All early-layer critical layers from genome_115 (pass threshold)
CRITICAL_LAYERS = [2, 5, 8, 11]

N_BOOT   = 500

# TODO: Codex to specify exact recipient type and injection conditions
# RECIPIENT_TYPE = "lesioned"  # or "random_init"
# INJECTION_LAYERS = [5]  # or CRITICAL_LAYERS


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_trained():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    return model, tok


def load_random_init(tok):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_config(cfg).to(DEVICE)
    model = model.to(torch.bfloat16).eval()
    return model


def tokenize(texts, tok):
    enc = tok(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=SEQ_LEN)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


# ---------------------------------------------------------------------------
# Extract critical direction from donor at SURGERY_LAYER
# ---------------------------------------------------------------------------

def extract_critical_direction(donor, tok, fit_texts, layer_idx):
    """Returns unit-normed PC1 direction (d_model,) from layer_idx."""
    pooled = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        pooled.append(h.detach().float().mean(dim=1).cpu())

    handle = donor.model.layers[layer_idx].register_forward_hook(hook_fn)
    for i in range(0, len(fit_texts), BATCH):
        ids, mask = tokenize(fit_texts[i:i+BATCH], tok)
        with torch.no_grad():
            donor(input_ids=ids, attention_mask=mask)
    handle.remove()

    acts = torch.cat(pooled, dim=0).numpy()
    pca = PCA(n_components=5)
    pca.fit(acts)
    dir_pc1 = pca.components_[0]
    return dir_pc1 / (np.linalg.norm(dir_pc1) + 1e-8), float(pca.explained_variance_ratio_[0])


def extract_per_layer_directions(donor, tok, fit_texts, layer_idxs):
    """Returns dict {layer_idx: unit-normed PC1} for multiple layers."""
    dirs = {}
    for li in layer_idxs:
        d, var = extract_critical_direction(donor, tok, fit_texts, li)
        dirs[li] = (d, var)
        print(f"  layer {li}: PC1 var={var:.3f}")
    return dirs


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def make_lesion_hook(direction):
    """Project out direction from activations (lesion)."""
    dir_t = torch.tensor(direction, dtype=torch.bfloat16, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)

    def hook_fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        proj = (h @ dir_t).unsqueeze(-1) * dir_t
        h_new = h - proj
        return (h_new,) + out[1:] if is_tuple else h_new

    return hook_fn


def make_injection_hook(direction, scale=1.0):
    """Add scale * direction to every token's activation (injection)."""
    dir_t = torch.tensor(direction, dtype=torch.bfloat16, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)
    scale_t = torch.tensor(scale, dtype=torch.bfloat16, device=DEVICE)

    def hook_fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        h_new = h + scale_t * dir_t
        return (h_new,) + out[1:] if is_tuple else h_new

    return hook_fn


# ---------------------------------------------------------------------------
# NLL measurement
# ---------------------------------------------------------------------------

def logits_to_nll(logits, input_ids):
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    ).item()


def measure_nll_per_seq(model, tok, texts, hook_fns_by_layer=None):
    handles = []
    if hook_fns_by_layer:
        for li, fn in hook_fns_by_layer.items():
            handles.append(model.model.layers[li].register_forward_hook(fn))

    per_seq = []
    for i in range(0, len(texts), BATCH):
        ids, mask = tokenize(texts[i:i+BATCH], tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        for j in range(ids.shape[0]):
            per_seq.append(logits_to_nll(out.logits[j:j+1], ids[j:j+1]))

    for h in handles:
        h.remove()
    return np.array(per_seq)


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    return {
        "mean": float(np.mean(values)),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def gap_closed(nll_recipient, nll_donor, nll_after):
    gap = nll_recipient - nll_donor
    if abs(gap) < 1e-6:
        return 0.0
    return float((nll_recipient - nll_after) / gap * 100)


# ---------------------------------------------------------------------------
# Main — AWAITING CODEX PROTOCOL SPEC
# ---------------------------------------------------------------------------

def main():
    # NOTE: This scaffold is ready but the exact protocol (recipient type,
    # injection conditions, layer subset) is pending Codex review of genome_115.
    # Run after Codex specifies the protocol in the genome_116 Codex review.
    print("genome_116 scaffold ready. Awaiting Codex protocol specification.")
    print("See TODO markers in this file for Codex to fill in.")


if __name__ == "__main__":
    main()
