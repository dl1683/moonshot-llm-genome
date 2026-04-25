"""
genome_115_local_subspace_disambiguation.py

genome_114 showed a catastrophic effect when ablating the top PCA direction
from layer 14 across ALL 28 layers simultaneously. But PCA was also fit on
layer-14 activations and applied everywhere. Codex flagged this as a potential
protocol artifact: the all-layer global hook may be doing the damage, not the
layer-14 direction per se.

This experiment disambiguates: is there a LAYER-LOCAL critical direction?

Protocol
--------
Fit split (n=200) and eval split (n=100) are disjoint wikitext draws.

For each probe layer in [2, 5, 8, 11, 14, 17, 20, 23, 26]:
  1. Fit PCA on fit-split activations AT THAT LAYER (layer-native directions).
  2. Evaluate 5 conditions on eval split:
     a. clean          - no ablation
     b. local_top1     - ablate PC1 only at this layer
     c. local_pc2      - ablate PC2 only at this layer (same-layer control)
     d. local_random   - ablate a matched random unit vector only at this layer
     e. all_layers_top1 - ablate PC1 (from this layer's PCA) at ALL layers
                          (genome_114 protocol replicate for direct comparison)
  3. Bootstrap (500 resamples) DELTA_NLL for each condition.

Pass: some probe layer shows
  - DELTA_NLL(local_top1) >= 1.0 nat
  - local_top1 / local_random >= 5x
  - local_top1 / local_pc2   >= 3x
  - bootstrap 95% CI for local_top1 effect excludes zero

Kill: ALL probe layers show local_top1 / local_random < 2x, while
      all_layers_top1 remains catastrophic — genome_114 was protocol artifact.

End goal: identifying WHICH layers contain layer-local critical directions
is the prerequisite for any surgical capability injection into an untrained model.
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

N_FIT    = 200    # sequences for PCA fitting (disjoint from eval)
N_EVAL   = 100    # sequences for NLL evaluation
SEQ_LEN  = 64
BATCH    = 8

PROBE_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26]

N_BOOT   = 500    # bootstrap resamples

PASS_DELTA_NLL   = 1.0   # local_top1 DELTA_NLL must exceed this
PASS_VS_RANDOM   = 5.0   # local_top1 / local_random ratio
PASS_VS_PC2      = 3.0   # local_top1 / local_pc2 ratio


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_wikitext_split(n, offset, seed=SEED):
    ds  = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    out = []
    count = 0
    for idx in perm:
        t = ds[int(idx)]["text"].strip()
        if len(t) < 60:
            continue
        if count >= offset and count < offset + n:
            out.append(t[:300])
        count += 1
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    return model, tok


def tokenize(texts, tok, seq_len=SEQ_LEN):
    enc = tok(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=seq_len)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


# ---------------------------------------------------------------------------
# Activation extraction (no ablation) — collect layer activations
# ---------------------------------------------------------------------------

def extract_activations(model, tok, texts, layer_idx):
    """Return shape (N_seqs, d_model) mean-pooled activations at layer_idx."""
    acts = []
    handles = []

    def hook_fn(module, inp, out):
        h = out[0]   # (B, T, D)
        acts.append(h.detach().float().cpu())

    h = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    handles.append(h)

    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        ids, mask = tokenize(batch, tok)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)

    for h in handles:
        h.remove()

    all_acts = torch.cat(acts, dim=0)   # (N, T, D)
    mask_cpu = torch.ones(all_acts.shape[:2], dtype=torch.bool)
    pooled = (all_acts * mask_cpu.unsqueeze(-1)).sum(1) / mask_cpu.sum(1, keepdim=True)
    return pooled.numpy()   # (N, D)


# ---------------------------------------------------------------------------
# NLL measurement with optional ablation hook
# ---------------------------------------------------------------------------

def logits_to_nll(logits, input_ids):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
        ignore_index=-100,
    )
    return loss.item()


def make_local_hook(direction, layer_idx):
    """Hook that projects out `direction` only at `layer_idx`."""
    dir_t = torch.tensor(direction, dtype=torch.bfloat16, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)

    def hook_fn(module, inp, out):
        h = out[0]
        proj = (h @ dir_t).unsqueeze(-1) * dir_t
        h_new = h - proj
        if isinstance(out, tuple):
            return (h_new,) + out[1:]
        return h_new

    return hook_fn


def make_alllayer_hook(direction):
    """Hook that projects out `direction` at every layer it's registered on."""
    dir_t = torch.tensor(direction, dtype=torch.bfloat16, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)

    def hook_fn(module, inp, out):
        h = out[0]
        proj = (h @ dir_t).unsqueeze(-1) * dir_t
        h_new = h - proj
        if isinstance(out, tuple):
            return (h_new,) + out[1:]
        return h_new

    return hook_fn


def measure_nll(model, tok, texts, hook_fns_by_layer=None):
    """Measure mean NLL. hook_fns_by_layer: dict {layer_idx: hook_fn} or None."""
    handles = []
    if hook_fns_by_layer:
        for li, fn in hook_fns_by_layer.items():
            h = model.model.layers[li].register_forward_hook(fn)
            handles.append(h)

    nlls = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        ids, mask = tokenize(batch, tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        nll = logits_to_nll(out.logits, ids)
        nlls.append(nll)

    for h in handles:
        h.remove()

    return float(np.mean(nlls))


def measure_nll_per_seq(model, tok, texts, hook_fns_by_layer=None):
    """Return per-sequence NLL for bootstrap."""
    handles = []
    if hook_fns_by_layer:
        for li, fn in hook_fns_by_layer.items():
            h = model.model.layers[li].register_forward_hook(fn)
            handles.append(h)

    per_seq = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        ids, mask = tokenize(batch, tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        # per-sequence NLL
        for j in range(ids.shape[0]):
            nll_j = logits_to_nll(out.logits[j:j+1], ids[j:j+1])
            per_seq.append(nll_j)

    for h in handles:
        h.remove()

    return np.array(per_seq)


def bootstrap_delta(clean_nlls, ablated_nlls, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    deltas = ablated_nlls - clean_nlls
    boot = []
    n = len(deltas)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(deltas[idx].mean())
    arr = np.array(boot)
    return {
        "mean": float(deltas.mean()),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("Loading model...")
    model, tok = load_model()
    n_layers = model.config.num_hidden_layers
    d_model  = model.config.hidden_size
    print(f"  {MODEL_ID}: {n_layers} layers, d={d_model}")

    print(f"Loading data (fit={N_FIT}, eval={N_EVAL})...")
    fit_texts  = load_wikitext_split(N_FIT, offset=0)
    eval_texts = load_wikitext_split(N_EVAL, offset=N_FIT)
    print(f"  fit={len(fit_texts)}, eval={len(eval_texts)}")

    # Clean per-sequence NLL on eval
    print("Measuring clean NLL on eval split...")
    clean_per_seq = measure_nll_per_seq(model, tok, eval_texts)
    nll_clean = float(clean_per_seq.mean())
    print(f"  NLL_clean = {nll_clean:.4f}")

    results_by_layer = {}
    any_pass = False

    for layer_idx in PROBE_LAYERS:
        print(f"\n--- Layer {layer_idx} ---")

        # Fit PCA on fit split at this layer
        print(f"  Extracting fit-split activations at layer {layer_idx}...")
        fit_acts = extract_activations(model, tok, fit_texts, layer_idx)  # (N_FIT, D)
        pca = PCA(n_components=10)
        pca.fit(fit_acts)
        dir_top1 = pca.components_[0]   # top PC
        dir_pc2  = pca.components_[1]   # second PC
        # random matched unit vector
        rand_vec = rng.standard_normal(d_model)
        rand_vec = rand_vec / (np.linalg.norm(rand_vec) + 1e-8)

        var_top1 = float(pca.explained_variance_ratio_[0])
        var_pc2  = float(pca.explained_variance_ratio_[1])
        print(f"  PC1 var={var_top1:.3f}, PC2 var={var_pc2:.3f}")

        cond_results = {}

        # Condition b: local top-1 ablation
        local_top1_per_seq = measure_nll_per_seq(
            model, tok, eval_texts,
            hook_fns_by_layer={layer_idx: make_local_hook(dir_top1, layer_idx)}
        )
        delta_top1 = bootstrap_delta(clean_per_seq, local_top1_per_seq)
        cond_results["local_top1"] = delta_top1
        print(f"  local_top1  DELTA_NLL={delta_top1['mean']:.4f}  CI=[{delta_top1['ci_lo']:.4f},{delta_top1['ci_hi']:.4f}]")

        # Condition c: local PC-2 ablation
        local_pc2_per_seq = measure_nll_per_seq(
            model, tok, eval_texts,
            hook_fns_by_layer={layer_idx: make_local_hook(dir_pc2, layer_idx)}
        )
        delta_pc2 = bootstrap_delta(clean_per_seq, local_pc2_per_seq)
        cond_results["local_pc2"] = delta_pc2
        print(f"  local_pc2   DELTA_NLL={delta_pc2['mean']:.4f}  CI=[{delta_pc2['ci_lo']:.4f},{delta_pc2['ci_hi']:.4f}]")

        # Condition d: local random ablation
        local_rand_per_seq = measure_nll_per_seq(
            model, tok, eval_texts,
            hook_fns_by_layer={layer_idx: make_local_hook(rand_vec, layer_idx)}
        )
        delta_rand = bootstrap_delta(clean_per_seq, local_rand_per_seq)
        cond_results["local_random"] = delta_rand
        print(f"  local_rand  DELTA_NLL={delta_rand['mean']:.4f}  CI=[{delta_rand['ci_lo']:.4f},{delta_rand['ci_hi']:.4f}]")

        # Condition e: all-layers top-1 ablation (genome_114 replicate)
        all_layer_hooks = {li: make_alllayer_hook(dir_top1) for li in range(n_layers)}
        alllayer_per_seq = measure_nll_per_seq(model, tok, eval_texts, all_layer_hooks)
        delta_all = bootstrap_delta(clean_per_seq, alllayer_per_seq)
        cond_results["all_layers_top1"] = delta_all
        print(f"  all_layers  DELTA_NLL={delta_all['mean']:.4f}  CI=[{delta_all['ci_lo']:.4f},{delta_all['ci_hi']:.4f}]")

        # Ratios
        eps = 1e-6
        ratio_vs_rand = delta_top1["mean"] / (abs(delta_rand["mean"]) + eps)
        ratio_vs_pc2  = delta_top1["mean"] / (abs(delta_pc2["mean"]) + eps)

        layer_pass = (
            delta_top1["mean"] >= PASS_DELTA_NLL and
            ratio_vs_rand >= PASS_VS_RANDOM and
            ratio_vs_pc2  >= PASS_VS_PC2 and
            delta_top1["ci_lo"] > 0
        )
        if layer_pass:
            any_pass = True
        print(f"  ratio_vs_rand={ratio_vs_rand:.2f}, ratio_vs_pc2={ratio_vs_pc2:.2f}, layer_pass={layer_pass}")

        results_by_layer[layer_idx] = {
            "var_top1": var_top1,
            "var_pc2":  var_pc2,
            "conditions": cond_results,
            "ratio_vs_random": ratio_vs_rand,
            "ratio_vs_pc2":    ratio_vs_pc2,
            "layer_pass":      layer_pass,
        }

    # Overall verdict
    if any_pass:
        passing = [l for l, r in results_by_layer.items() if r["layer_pass"]]
        verdict = f"LAYER_LOCAL_CONFIRMED: {len(passing)} layers pass ({passing}). dir-0 effect is real and layer-local, not just a global hook artifact."
    else:
        # Check if all-layers still catastrophic
        max_all = max(r["conditions"]["all_layers_top1"]["mean"] for r in results_by_layer.values())
        max_local = max(r["conditions"]["local_top1"]["mean"] for r in results_by_layer.values())
        if max_all > 2.0 and max_local < PASS_DELTA_NLL:
            verdict = f"PROTOCOL_ARTIFACT: all-layers max={max_all:.3f} but no local layer passes ({max_local:.3f} < {PASS_DELTA_NLL}). genome_114 effect is primarily a global-hook artifact."
        else:
            verdict = f"INCONCLUSIVE: max_local_delta={max_local:.3f}, max_all_delta={max_all:.3f}. Thresholds not met cleanly in either direction."

    print(f"\nVerdict: {verdict}")

    out = {
        "model": MODEL_ID,
        "n_fit": N_FIT,
        "n_eval": N_EVAL,
        "nll_clean": nll_clean,
        "probe_layers": PROBE_LAYERS,
        "n_boot": N_BOOT,
        "pass_criteria": {
            "delta_nll_threshold": PASS_DELTA_NLL,
            "ratio_vs_random": PASS_VS_RANDOM,
            "ratio_vs_pc2": PASS_VS_PC2,
        },
        "results_by_layer": results_by_layer,
        "any_layer_pass": any_pass,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }

    out_path = RESULTS / "genome_115_local_subspace_disambiguation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
