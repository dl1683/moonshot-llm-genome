"""
genome_116_surgery_injection.py

END GOAL: efficient capability transfer from a trained model into an untrained
model without retraining.

Locked protocol for the first surgery experiment:
  - Recipient FIRST: a lesioned pretrained Qwen3-0.6B clone, not random-init.
    Reason: donor PC1 is basis-aligned here; raw donor PC1 is basis-misaligned
    in a random-init recipient, so random-init is a phase-2 experiment.
  - Primary condition: layer 5 only. genome_115 showed layer 5 is the strongest
    local critical direction (4.46 nats, 906x vs random).
  - Secondary condition: exact same replacement at layers [2, 5, 8, 11] as a
    comparison, not the first thing to trust.
  - Injection is the actual inverse of the lesion at a given layer:
        h_lesioned = h - (h.d)d
        h_replaced = h_lesioned + c_donor d
                   = h - (h.d)d + c_donor d
    where c_donor is the donor projection coefficient for EACH token on the
    eval batch. This replaces the along-d component instead of adding a fixed
    alpha*d offset.
  - Lesion+inject at the same layer is implemented as ONE composite hook per
    layer. A dict with one hook per layer is therefore sufficient.

Pass criteria for this stage:
  - PASS: layer-5 exact replacement closes >= 90% of the lesion gap
  - PARTIAL: closes >= 20%
  - KILL: closes < 5%

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

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 42

N_FIT = 200
N_EVAL = 100
SEQ_LEN = 64
BATCH = 8

SURGERY_LAYER = 5
CRITICAL_LAYERS = [2, 5, 8, 11]

PRIMARY_RECIPIENT = "lesioned_pretrained_qwen3"
PRIMARY_LAYERS = [SURGERY_LAYER]
SECONDARY_LAYERS = CRITICAL_LAYERS

N_BOOT = 500

PASS_GAP_CLOSED_PCT = 90.0
PARTIAL_GAP_CLOSED_PCT = 20.0
KILL_GAP_CLOSED_PCT = 5.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_wikitext_split(n, offset, seed=SEED):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    out = []
    count = 0
    for idx in perm:
        text = ds[int(idx)]["text"].strip()
        if len(text) < 60:
            continue
        if count >= offset:
            out.append(text[:300])
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


def tokenize(texts, tok):
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN,
    )
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


# ---------------------------------------------------------------------------
# Direction fitting and donor coefficient extraction
# ---------------------------------------------------------------------------

def extract_critical_direction(donor, tok, fit_texts, layer_idx):
    """Fit sequence-mean PC1 at one layer and return unit direction + variance."""
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
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction, float(pca.explained_variance_ratio_[0])


def extract_per_layer_directions(donor, tok, fit_texts, layer_idxs):
    out = {}
    for layer_idx in layer_idxs:
        direction, var_pc1 = extract_critical_direction(donor, tok, fit_texts, layer_idx)
        out[layer_idx] = {"direction": direction, "var_pc1": var_pc1}
        print(f"  layer {layer_idx}: PC1 var={var_pc1:.3f}")
    return out


def collect_donor_coeff_batches(donor, tok, texts, directions_by_layer):
    """
    For each requested layer, collect the donor's per-token coefficient c_donor
    = h.d on the eval texts, batched exactly as the recipient evaluation will run.
    """
    coeff_batches = {layer_idx: [] for layer_idx in directions_by_layer}
    dir_tensors = {}
    for layer_idx, direction in directions_by_layer.items():
        dir_t = torch.tensor(direction, dtype=torch.float32, device=DEVICE)
        dir_tensors[layer_idx] = dir_t / (dir_t.norm() + 1e-8)

    def make_capture_hook(layer_idx):
        dir_t = dir_tensors[layer_idx]

        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            coeff = torch.matmul(h.detach().float(), dir_t).cpu()
            coeff_batches[layer_idx].append(coeff)

        return hook_fn

    handles = []
    for layer_idx in directions_by_layer:
        handles.append(
            donor.model.layers[layer_idx].register_forward_hook(
                make_capture_hook(layer_idx)
            )
        )

    for i in range(0, len(texts), BATCH):
        ids, mask = tokenize(texts[i:i+BATCH], tok)
        with torch.no_grad():
            donor(input_ids=ids, attention_mask=mask)
        mask_cpu = mask.cpu().float()
        for layer_idx in coeff_batches:
            coeff_batches[layer_idx][-1] = coeff_batches[layer_idx][-1] * mask_cpu

    for handle in handles:
        handle.remove()

    return coeff_batches


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def make_lesion_hook(direction):
    """Project out the along-direction component at one layer."""
    dir_t = torch.tensor(direction, dtype=torch.float32, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)
    dir_view = dir_t.view(1, 1, -1)

    def hook_fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        h_float = h.float()
        coeff = torch.matmul(h_float, dir_t).unsqueeze(-1)
        h_new = (h_float - coeff * dir_view).to(h.dtype)
        return (h_new,) + out[1:] if is_tuple else h_new

    return hook_fn


def make_replace_hook(direction, donor_coeff_batch):
    """
    Composite hook for lesion+inject at the same layer:
      1. remove recipient coeff (h.d)d
      2. add donor coeff c_donor d
    """
    dir_t = torch.tensor(direction, dtype=torch.float32, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)
    dir_view = dir_t.view(1, 1, -1)
    donor_coeff_cpu = donor_coeff_batch.detach().cpu().float()

    def hook_fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        donor_coeff = donor_coeff_cpu.to(device=h.device, dtype=torch.float32)
        if donor_coeff.shape != h.shape[:2]:
            raise ValueError(
                f"donor coeff shape {tuple(donor_coeff.shape)} != hidden shape {tuple(h.shape[:2])}"
            )
        h_float = h.float()
        recip_coeff = torch.matmul(h_float, dir_t).unsqueeze(-1)
        donor_coeff = donor_coeff.unsqueeze(-1)
        h_new = (h_float - recip_coeff * dir_view + donor_coeff * dir_view).to(h.dtype)
        return (h_new,) + out[1:] if is_tuple else h_new

    return hook_fn


def make_static_lesion_map(directions_by_layer, layers):
    return {layer_idx: make_lesion_hook(directions_by_layer[layer_idx]) for layer_idx in layers}


def make_batch_replace_factories(directions_by_layer, donor_coeff_batches, layers):
    factories = {}
    for layer_idx in layers:
        direction = directions_by_layer[layer_idx]

        def factory(batch_idx, layer_idx=layer_idx, direction=direction):
            return make_replace_hook(direction, donor_coeff_batches[layer_idx][batch_idx])

        factories[layer_idx] = factory
    return factories


# ---------------------------------------------------------------------------
# NLL measurement
# ---------------------------------------------------------------------------

def logits_to_nll(logits, input_ids, attention_mask):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous().clone()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels[shift_mask == 0] = -100
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
        ignore_index=-100,
    ).item()


def measure_nll_per_seq(
    model,
    tok,
    texts,
    hook_fns_by_layer=None,
    batch_hook_factories_by_layer=None,
):
    if hook_fns_by_layer and batch_hook_factories_by_layer:
        raise ValueError("Use static hooks OR batch hook factories, not both.")

    static_handles = []
    if hook_fns_by_layer:
        for layer_idx, hook_fn in hook_fns_by_layer.items():
            static_handles.append(
                model.model.layers[layer_idx].register_forward_hook(hook_fn)
            )

    per_seq = []
    batch_idx = 0
    for i in range(0, len(texts), BATCH):
        batch_handles = []
        if batch_hook_factories_by_layer:
            for layer_idx, factory in batch_hook_factories_by_layer.items():
                batch_handles.append(
                    model.model.layers[layer_idx].register_forward_hook(factory(batch_idx))
                )

        ids, mask = tokenize(texts[i:i+BATCH], tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        for j in range(ids.shape[0]):
            per_seq.append(logits_to_nll(out.logits[j:j+1], ids[j:j+1], mask[j:j+1]))

        for handle in batch_handles:
            handle.remove()
        batch_idx += 1

    for handle in static_handles:
        handle.remove()

    return np.array(per_seq)


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    boots = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ]
    return {
        "mean": float(np.mean(values)),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def gap_closed(nll_recipient, nll_donor, nll_after):
    gap = nll_recipient - nll_donor
    if abs(gap) < 1e-6:
        return 0.0
    return float((nll_recipient - nll_after) / gap * 100.0)


def bootstrap_gap_closed(clean_nlls, lesion_nlls, repaired_nlls, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(clean_nlls)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(
            gap_closed(
                float(lesion_nlls[idx].mean()),
                float(clean_nlls[idx].mean()),
                float(repaired_nlls[idx].mean()),
            )
        )
    point = gap_closed(
        float(lesion_nlls.mean()),
        float(clean_nlls.mean()),
        float(repaired_nlls.mean()),
    )
    return {
        "mean": float(point),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def evaluate_condition(
    model,
    tok,
    texts,
    clean_per_seq,
    lesion_per_seq=None,
    hook_fns_by_layer=None,
    batch_hook_factories_by_layer=None,
):
    per_seq = measure_nll_per_seq(
        model,
        tok,
        texts,
        hook_fns_by_layer=hook_fns_by_layer,
        batch_hook_factories_by_layer=batch_hook_factories_by_layer,
    )
    result = {"nll": bootstrap_mean_ci(per_seq)}
    if lesion_per_seq is not None:
        result["gap_closed_pct"] = bootstrap_gap_closed(clean_per_seq, lesion_per_seq, per_seq)
    return per_seq, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("Loading model...")
    model, tok = load_trained()
    print(f"  recipient_first = {PRIMARY_RECIPIENT}")
    print(f"  primary_layers  = {PRIMARY_LAYERS}")
    print(f"  secondary_layers= {SECONDARY_LAYERS}")

    print(f"Loading data (fit={N_FIT}, eval={N_EVAL})...")
    fit_texts = load_wikitext_split(N_FIT, offset=0)
    eval_texts = load_wikitext_split(N_EVAL, offset=N_FIT)
    print(f"  fit={len(fit_texts)}, eval={len(eval_texts)}")

    print("Measuring clean donor/recipient NLL...")
    clean_per_seq = measure_nll_per_seq(model, tok, eval_texts)
    clean_stats = bootstrap_mean_ci(clean_per_seq)
    print(f"  clean NLL = {clean_stats['mean']:.4f}")

    direction_layers = sorted(set(SECONDARY_LAYERS))
    print("Fitting donor PC1 directions...")
    direction_info = extract_per_layer_directions(model, tok, fit_texts, direction_layers)
    directions_by_layer = {
        layer_idx: info["direction"] for layer_idx, info in direction_info.items()
    }

    print("Collecting donor per-token coefficients on eval split...")
    donor_coeff_batches = collect_donor_coeff_batches(
        model, tok, eval_texts, directions_by_layer
    )

    print("\nEvaluating primary lesion: layer 5 only...")
    lesion_l5_per_seq, lesion_l5 = evaluate_condition(
        model,
        tok,
        eval_texts,
        clean_per_seq,
        hook_fns_by_layer=make_static_lesion_map(directions_by_layer, PRIMARY_LAYERS),
    )
    print(f"  lesion_l5 NLL = {lesion_l5['nll']['mean']:.4f}")

    print("Evaluating primary exact replacement: layer 5 only...")
    replace_l5_per_seq, replace_l5 = evaluate_condition(
        model,
        tok,
        eval_texts,
        clean_per_seq,
        lesion_per_seq=lesion_l5_per_seq,
        batch_hook_factories_by_layer=make_batch_replace_factories(
            directions_by_layer, donor_coeff_batches, PRIMARY_LAYERS
        ),
    )
    print(
        f"  replace_l5 NLL = {replace_l5['nll']['mean']:.4f}  "
        f"gap_closed = {replace_l5['gap_closed_pct']['mean']:.2f}%"
    )

    print("\nEvaluating secondary lesion: layers [2, 5, 8, 11]...")
    lesion_early4_per_seq, lesion_early4 = evaluate_condition(
        model,
        tok,
        eval_texts,
        clean_per_seq,
        hook_fns_by_layer=make_static_lesion_map(directions_by_layer, SECONDARY_LAYERS),
    )
    print(f"  lesion_early4 NLL = {lesion_early4['nll']['mean']:.4f}")

    print("Evaluating secondary exact replacement: layers [2, 5, 8, 11]...")
    replace_early4_per_seq, replace_early4 = evaluate_condition(
        model,
        tok,
        eval_texts,
        clean_per_seq,
        lesion_per_seq=lesion_early4_per_seq,
        batch_hook_factories_by_layer=make_batch_replace_factories(
            directions_by_layer, donor_coeff_batches, SECONDARY_LAYERS
        ),
    )
    print(
        f"  replace_early4 NLL = {replace_early4['nll']['mean']:.4f}  "
        f"gap_closed = {replace_early4['gap_closed_pct']['mean']:.2f}%"
    )

    primary_gap = replace_l5["gap_closed_pct"]["mean"]
    primary_ci_lo = replace_l5["gap_closed_pct"]["ci_lo"]
    secondary_gap = replace_early4["gap_closed_pct"]["mean"]

    if primary_gap >= PASS_GAP_CLOSED_PCT:
        verdict = (
            f"PASS: layer-5 exact coefficient replacement closes {primary_gap:.1f}% "
            "of the lesion gap. The correct first surgery protocol is a lesioned "
            "pretrained Qwen recipient with a composite hook at layer 5."
        )
    elif primary_gap >= PARTIAL_GAP_CLOSED_PCT and primary_ci_lo > 0.0:
        verdict = (
            f"PARTIAL: layer-5 exact coefficient replacement closes {primary_gap:.1f}% "
            "of the lesion gap. The hook algebra is directionally correct, but not "
            "yet near-exact."
        )
    elif primary_gap < KILL_GAP_CLOSED_PCT:
        verdict = (
            f"KILL: layer-5 exact coefficient replacement closes only {primary_gap:.1f}% "
            "of the lesion gap. If the exact inverse cannot recover the lesion, PC1 "
            "alone is not a viable surgery handle."
        )
    else:
        verdict = (
            f"INCONCLUSIVE: layer-5 exact coefficient replacement closes {primary_gap:.1f}% "
            f"(CI lo {primary_ci_lo:.1f}%). Re-run before escalating to random-init."
        )

    print("\n=== SURGERY SUMMARY ===")
    print(f"  clean NLL:          {clean_stats['mean']:.4f}")
    print(f"  lesion_l5 NLL:      {lesion_l5['nll']['mean']:.4f}")
    print(
        f"  replace_l5 NLL:     {replace_l5['nll']['mean']:.4f}  "
        f"gap_closed={primary_gap:.2f}%"
    )
    print(f"  lesion_early4 NLL:  {lesion_early4['nll']['mean']:.4f}")
    print(
        f"  replace_early4 NLL: {replace_early4['nll']['mean']:.4f}  "
        f"gap_closed={secondary_gap:.2f}%"
    )
    print(f"  verdict: {verdict}")

    out = {
        "model": MODEL_ID,
        "recipient_type_first": PRIMARY_RECIPIENT,
        "random_init_first": False,
        "n_fit": N_FIT,
        "n_eval": N_EVAL,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH,
        "primary_layers": PRIMARY_LAYERS,
        "secondary_layers": SECONDARY_LAYERS,
        "pass_criteria": {
            "pass_gap_closed_pct": PASS_GAP_CLOSED_PCT,
            "partial_gap_closed_pct": PARTIAL_GAP_CLOSED_PCT,
            "kill_gap_closed_pct": KILL_GAP_CLOSED_PCT,
        },
        "pc1_by_layer": {
            str(layer_idx): {"var_pc1": info["var_pc1"]}
            for layer_idx, info in direction_info.items()
        },
        "conditions": {
            "clean": {"layers": [], "nll": clean_stats},
            "lesion_l5": {"layers": PRIMARY_LAYERS, **lesion_l5},
            "replace_l5_exact": {"layers": PRIMARY_LAYERS, **replace_l5},
            "lesion_early4": {"layers": SECONDARY_LAYERS, **lesion_early4},
            "replace_early4_exact": {"layers": SECONDARY_LAYERS, **replace_early4},
        },
        "recommended_protocol": {
            "recipient_type": PRIMARY_RECIPIENT,
            "inject_layers": PRIMARY_LAYERS,
            "why": (
                "Layer 5 is the strongest local causal axis, shares basis with the donor, "
                "and avoids adding approximation error from weaker early layers."
            ),
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }

    out_path = RESULTS / "genome_116_surgery_injection.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
