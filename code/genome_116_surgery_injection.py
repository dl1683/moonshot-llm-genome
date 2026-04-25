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

Step 3. Injection conditions:
  a. NO_INJECTION: recipient as-is (baseline)
  b. LESION_ONLY: project out dir_critical at surgery layers (defines gap)
  c. TRANSPLANT_HOOK: composite hook = project-out then inject donor mean proj
     (make_transplant_hook: h -> h - (h·d)d + donor_mean_proj * d)
  d. INJECT_ONLY: add donor mean proj without first projecting out
     (tests injection on unlesioned model as upper bound)

  Correctness notes (fixed from scaffold):
  - Injection uses donor_mean_proj (extracted from fit split) not fixed alpha
  - make_transplant_hook is the correct inverse: removes recipient component,
    injects donor mean — avoids non-inverse composition bug
  - logits_to_nll masks pad labels via attention_mask -> ignore_index=-100
  - extract_critical_direction uses masked mean pooling (no pad contamination)
  - eval split starts at offset 700 (past genome_116b probe range 200-699)

Step 4. Measure:
  - NLL on eval-split (n=100, offset=700, disjoint from fit+116b probe), ZERO gradient steps
  - Paired bootstrap CIs (n=500)
  - Report: NLL_donor, NLL_lesion, NLL_transplant, gap_closed_%

Pass: transplant recovers >= 20% of lesion gap at ZERO gradient steps with CI_lo > 0
Kill: < 5% gap recovery — direction cannot be transplanted

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

# Eval offset: past genome_115 fit (0-199) + genome_116b probe (200-699) = 700
EVAL_OFFSET = 700


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

def masked_mean_pool(h_raw, mask):
    """Attention-mask-weighted mean pool. h_raw: (B,T,D), mask: (B,T) -> (B,D)."""
    m = mask.float().unsqueeze(-1)          # (B, T, 1)
    return (h_raw * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)


def extract_critical_direction(donor, tok, fit_texts, layer_idx):
    """Returns unit-normed PC1 direction (d_model,) from layer_idx."""
    raw_acts, masks = [], []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        raw_acts.append(h.detach().float().cpu())

    handle = donor.model.layers[layer_idx].register_forward_hook(hook_fn)
    for i in range(0, len(fit_texts), BATCH):
        ids, mask = tokenize(fit_texts[i:i+BATCH], tok)
        masks.append(mask.cpu())
        with torch.no_grad():
            donor(input_ids=ids, attention_mask=mask)
    handle.remove()

    pooled = torch.cat(
        [masked_mean_pool(h, m) for h, m in zip(raw_acts, masks)], dim=0
    ).numpy()
    pca = PCA(n_components=5)
    pca.fit(pooled)
    dir_pc1 = pca.components_[0]
    return dir_pc1 / (np.linalg.norm(dir_pc1) + 1e-8), float(pca.explained_variance_ratio_[0])


def extract_per_layer_directions(donor, tok, fit_texts, layer_idxs):
    """Returns dict {layer_idx: (unit-normed PC1, var)} for multiple layers."""
    dirs = {}
    for li in layer_idxs:
        d, var = extract_critical_direction(donor, tok, fit_texts, li)
        dirs[li] = (d, var)
        print(f"  layer {li}: PC1 var={var:.3f}")
    return dirs


def extract_donor_mean_proj(donor, tok, fit_texts, direction, layer_idx):
    """Compute mean per-token projection of donor activations onto direction."""
    dir_t = torch.tensor(direction, dtype=torch.float32)
    proj_vals = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out   # (B, T, D)
        proj_vals.append((h.detach().float().cpu() @ dir_t))  # (B, T)

    handle = donor.model.layers[layer_idx].register_forward_hook(hook_fn)
    masks_list = []
    for i in range(0, len(fit_texts), BATCH):
        ids, mask = tokenize(fit_texts[i:i+BATCH], tok)
        masks_list.append(mask.cpu())
        with torch.no_grad():
            donor(input_ids=ids, attention_mask=mask)
    handle.remove()

    all_projs = []
    for p, m in zip(proj_vals, masks_list):
        all_projs.extend(p[m.bool()].tolist())
    return float(np.mean(all_projs))


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


def make_injection_hook(direction, donor_mean_proj):
    """Add donor_mean_proj * direction to every token (sets mean component to donor's)."""
    dir_t = torch.tensor(direction, dtype=torch.bfloat16, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)
    scale_t = torch.tensor(donor_mean_proj, dtype=torch.bfloat16, device=DEVICE)

    def hook_fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        h_new = h + scale_t * dir_t
        return (h_new,) + out[1:] if is_tuple else h_new

    return hook_fn


def make_transplant_hook(direction, donor_mean_proj):
    """Composite: project out recipient's component then inject donor mean.
    Equivalent to: h -> h - (h·d)d + donor_mean_proj * d
    Correct inverse of lesion for the mean-level capability we measured.
    """
    dir_t = torch.tensor(direction, dtype=torch.bfloat16, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)
    scale_t = torch.tensor(donor_mean_proj, dtype=torch.bfloat16, device=DEVICE)

    def hook_fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        proj = (h @ dir_t).unsqueeze(-1) * dir_t   # remove recipient component
        h_new = h - proj + scale_t * dir_t          # inject donor mean
        return (h_new,) + out[1:] if is_tuple else h_new

    return hook_fn


# ---------------------------------------------------------------------------
# NLL measurement
# ---------------------------------------------------------------------------

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


def measure_nll_per_seq(model, tok, texts, hook_fns_by_layer=None):
    """hook_fns_by_layer: dict {layer_idx: hook_fn | list[hook_fn]}"""
    handles = []
    if hook_fns_by_layer:
        for li, fn in hook_fns_by_layer.items():
            fns = fn if isinstance(fn, list) else [fn]
            for f in fns:
                handles.append(model.model.layers[li].register_forward_hook(f))

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
