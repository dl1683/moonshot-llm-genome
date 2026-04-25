"""
genome_117_cross_model_surgery.py

DECISIVE CROSS-MODEL SURGERY TEST.

genome_116 validated the surgery machinery (100% on same-model).
genome_116d+e confirmed: critical direction is architecture-universal (sentence-
boundary axis, sign-flipped, same tokens in Qwen3 and Pythia).

This experiment tests GENUINE capability transfer: donor model != recipient.

Protocol specified by Codex (see C:/tmp/codex_117_protocol.txt).
Scaffold is ready; main() filled after Codex review.

Design space:
  (A) Qwen3 trained donor -> random-init Qwen3 recipient (inject only)
      - Same d_model: injection is directly applicable
      - Random-init has no sentence-boundary structure at layer 5
      - Test: does injecting donor PC1 coefficients improve random-init NLL?
  (B) Qwen3 trained donor -> partially trained Qwen3 recipient (inject only)
      - Recipient has some structure; more realistic surgery target
  (C) Mean-coefficient injection: use donor's mean proj (scalar) instead of
      per-token coefficients — "knowledge-free" surgery scenario

Key question for (A)/(B):
  Does injecting the donor's per-token PC1 values at layer 5 move a model
  that doesn't know the sentence-boundary structure toward lower NLL?
  If yes: the critical direction IS transferable at zero gradient steps.
  If no: downstream layers need to be co-trained to use the injected info.

Pass: gap_closed >= 20% with CI_lo > 0 (direction IS transferable)
Kill: gap_closed < 5% (direction is model-specific, not transferable)

Results: results/genome_117_cross_model_surgery.json
"""

import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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
N_BOOT   = 500

SURGERY_LAYER = 5
CRITICAL_LAYERS = [2, 5, 8, 11]


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
# Model loading
# ---------------------------------------------------------------------------

def load_trained(tok=None):
    if tok is None:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    return model, tok


def load_random_init(tok):
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).to(DEVICE).eval()
    return model


def tokenize(texts, tok):
    enc = tok(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=SEQ_LEN)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------

def extract_critical_direction(donor, tok, fit_texts, layer_idx):
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
    direction = pca.components_[0]
    direction /= (np.linalg.norm(direction) + 1e-8)
    return direction, float(pca.explained_variance_ratio_[0])


def collect_donor_coeff_batches(donor, tok, texts, directions_by_layer):
    """Collect per-token donor coefficients batch-by-batch."""
    coeff_batches = {li: [] for li in directions_by_layer}
    dir_tensors = {}
    for li, direction in directions_by_layer.items():
        d = torch.tensor(direction, dtype=torch.float32, device=DEVICE)
        dir_tensors[li] = d / (d.norm() + 1e-8)

    def make_capture_hook(li):
        d = dir_tensors[li]
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            coeff_batches[li].append(torch.matmul(h.detach().float(), d).cpu())
        return hook_fn

    handles = [donor.model.layers[li].register_forward_hook(make_capture_hook(li))
               for li in directions_by_layer]
    mask_list = []
    for i in range(0, len(texts), BATCH):
        ids, mask = tokenize(texts[i:i+BATCH], tok)
        mask_list.append(mask.cpu().float())
        with torch.no_grad():
            donor(input_ids=ids, attention_mask=mask)
    for h in handles:
        h.remove()

    # Mask out padding
    for li in coeff_batches:
        coeff_batches[li] = [c * m for c, m in zip(coeff_batches[li], mask_list)]
    return coeff_batches


def compute_donor_mean_proj(donor, tok, fit_texts, direction, layer_idx):
    """Scalar: mean per-token projection of donor activations onto direction."""
    dir_t = torch.tensor(direction, dtype=torch.float32, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)
    proj_vals, mask_list = [], []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        proj_vals.append(torch.matmul(h.detach().float(), dir_t).cpu())

    handle = donor.model.layers[layer_idx].register_forward_hook(hook_fn)
    for i in range(0, len(fit_texts), BATCH):
        ids, mask = tokenize(fit_texts[i:i+BATCH], tok)
        mask_list.append(mask.cpu().bool())
        with torch.no_grad():
            donor(input_ids=ids, attention_mask=mask)
    handle.remove()

    all_vals = []
    for p, m in zip(proj_vals, mask_list):
        all_vals.extend(p[m].tolist())
    return float(np.mean(all_vals))


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def make_inject_hook(direction, donor_mean_proj):
    """Add donor_mean_proj * direction to every token (constant offset)."""
    dir_t = torch.tensor(direction, dtype=torch.bfloat16, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)
    scale = torch.tensor(donor_mean_proj, dtype=torch.bfloat16, device=DEVICE)

    def hook_fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        h_new = h + scale * dir_t
        return (h_new,) + out[1:] if is_tuple else h_new
    return hook_fn


def make_replace_hook(direction, donor_coeff_batch):
    """Replace recipient's PC1 component with donor's per-token coefficients."""
    dir_t = torch.tensor(direction, dtype=torch.float32, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)
    dir_view = dir_t.view(1, 1, -1)
    donor_coeff_cpu = donor_coeff_batch.detach().cpu().float()

    def hook_fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        donor_c = donor_coeff_cpu.to(device=h.device, dtype=torch.float32).unsqueeze(-1)
        h_float = h.float()
        recip_c = torch.matmul(h_float, dir_t).unsqueeze(-1)
        h_new = (h_float - recip_c * dir_view + donor_c * dir_view).to(h.dtype)
        return (h_new,) + out[1:] if is_tuple else h_new
    return hook_fn


def make_batch_replace_factories(directions_by_layer, donor_coeff_batches, layers):
    factories = {}
    for li in layers:
        d = directions_by_layer[li]
        def factory(batch_idx, li=li, d=d):
            return make_replace_hook(d, donor_coeff_batches[li][batch_idx])
        factories[li] = factory
    return factories


# ---------------------------------------------------------------------------
# NLL measurement
# ---------------------------------------------------------------------------

def logits_to_nll(logits, input_ids, attention_mask):
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous().clone()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels[shift_mask == 0] = -100
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
        ignore_index=-100,
    ).item()


def measure_nll_per_seq(model, tok, texts, hook_fns_by_layer=None,
                        batch_hook_factories_by_layer=None):
    if hook_fns_by_layer and batch_hook_factories_by_layer:
        raise ValueError("Use static OR batch hook factories, not both.")

    static_handles = []
    if hook_fns_by_layer:
        for li, fn in hook_fns_by_layer.items():
            static_handles.append(model.model.layers[li].register_forward_hook(fn))

    per_seq, batch_idx = [], 0
    for i in range(0, len(texts), BATCH):
        batch_handles = []
        if batch_hook_factories_by_layer:
            for li, factory in batch_hook_factories_by_layer.items():
                batch_handles.append(
                    model.model.layers[li].register_forward_hook(factory(batch_idx))
                )
        ids, mask = tokenize(texts[i:i+BATCH], tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        for j in range(ids.shape[0]):
            per_seq.append(logits_to_nll(out.logits[j:j+1], ids[j:j+1], mask[j:j+1]))
        for h in batch_handles:
            h.remove()
        batch_idx += 1

    for h in static_handles:
        h.remove()
    return np.array(per_seq)


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(np.mean(values)),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


def gap_closed_pct(nll_baseline, nll_donor, nll_after):
    gap = nll_baseline - nll_donor
    if abs(gap) < 1e-6:
        return 0.0
    return float((nll_baseline - nll_after) / gap * 100.0)


def bootstrap_gap_closed(baseline_nlls, donor_nlls, repaired_nlls, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(baseline_nlls)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(gap_closed_pct(
            baseline_nlls[idx].mean(), donor_nlls[idx].mean(), repaired_nlls[idx].mean()
        ))
    point = gap_closed_pct(baseline_nlls.mean(), donor_nlls.mean(), repaired_nlls.mean())
    return {"mean": float(point),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


# ---------------------------------------------------------------------------
# Main — Codex protocol to be filled in from C:/tmp/codex_117_protocol.txt
# ---------------------------------------------------------------------------

def main():
    # TODO: implement once Codex specifies protocol
    print("genome_117 scaffold ready. Awaiting Codex cross-model surgery protocol.")
    print("Check C:/tmp/codex_117_protocol.txt for Codex output.")


if __name__ == "__main__":
    main()
