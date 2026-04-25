"""
genome_117_cross_model_surgery.py

DECISIVE CROSS-MODEL SURGERY TEST.

genome_116 validated the surgery machinery (100% on same-model).
genome_116d+e confirmed: critical direction is architecture-universal (sentence-
boundary axis, sign-flipped, same tokens in Qwen3 and Pythia).

This experiment tests GENUINE capability transfer: donor model != recipient.

Locked genome_117 protocol:
  - Recipient FIRST: random-init Qwen3-0.6B, not partially-trained Qwen3 and
    not Pythia-lesioned.
  - Reason: the project end goal is trained -> untrained transfer at zero
    gradient steps, and same-architecture Qwen avoids the token-alignment and
    hidden-size confounds that a Pythia recipient would introduce.
  - Primary condition: exact per-token coefficient replacement at layer 5.
  - Secondary condition: exact per-token coefficient replacement at layers
    [2, 5, 8, 11] to test whether the structural scaffold is distributed.
  - Diagnostic condition: donor mean-coefficient injection at layer 5 using a
    constant offset along the donor PC1 direction.

Key question:
  Does injecting donor PC1 structure into a random-init twin close any
  meaningful fraction of the donor-recipient NLL gap at zero gradient steps?
  If yes: this is the cleanest direct hit on the moonshot end goal so far.
  If no: PC1 alone is a real causal handle, but insufficient without trained
  downstream readers.

Pass: exact layer-5 replacement closes >= 20% of the donor-recipient gap with
      CI_lo > 0
Partial: best exact condition closes >= 5% with CI_lo > 0
Kill: both exact conditions close < 5% of the gap

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

PRIMARY_RECIPIENT = "random_init_qwen3"
PRIMARY_LAYERS = [SURGERY_LAYER]
SECONDARY_LAYERS = CRITICAL_LAYERS

PASS_GAP_CLOSED_PCT = 20.0
PARTIAL_GAP_CLOSED_PCT = 5.0
KILL_GAP_CLOSED_PCT = 5.0


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
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_condition(model, tok, texts, recipient_clean_per_seq, donor_clean_per_seq,
                       hook_fns_by_layer=None, batch_hook_factories_by_layer=None):
    per_seq = measure_nll_per_seq(
        model,
        tok,
        texts,
        hook_fns_by_layer=hook_fns_by_layer,
        batch_hook_factories_by_layer=batch_hook_factories_by_layer,
    )
    return per_seq, {
        "nll": bootstrap_mean_ci(per_seq),
        "gap_closed_pct": bootstrap_gap_closed(
            recipient_clean_per_seq,
            donor_clean_per_seq,
            per_seq,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("Loading donor + random-init recipient...")
    donor, tok = load_trained()
    recipient = load_random_init(tok)
    print(f"  donor_model     = {MODEL_ID}")
    print(f"  recipient_first = {PRIMARY_RECIPIENT}")
    print(f"  primary_layers  = {PRIMARY_LAYERS}")
    print(f"  secondary_layers= {SECONDARY_LAYERS}")

    print(f"Loading data (fit={N_FIT}, eval={N_EVAL})...")
    fit_texts = load_wikitext_split(N_FIT, offset=0)
    eval_texts = load_wikitext_split(N_EVAL, offset=N_FIT)
    print(f"  fit={len(fit_texts)}, eval={len(eval_texts)}")

    print("Measuring donor clean NLL...")
    donor_clean_per_seq = measure_nll_per_seq(donor, tok, eval_texts)
    donor_clean_stats = bootstrap_mean_ci(donor_clean_per_seq)
    print(f"  donor clean NLL = {donor_clean_stats['mean']:.4f}")

    print("Measuring random-init recipient clean NLL...")
    recipient_clean_per_seq = measure_nll_per_seq(recipient, tok, eval_texts)
    recipient_clean_stats = bootstrap_mean_ci(recipient_clean_per_seq)
    print(f"  recipient clean NLL = {recipient_clean_stats['mean']:.4f}")

    direction_layers = sorted(set(SECONDARY_LAYERS))
    print("Fitting donor PC1 directions...")
    direction_info = {}
    directions_by_layer = {}
    for layer_idx in direction_layers:
        direction, var_pc1 = extract_critical_direction(donor, tok, fit_texts, layer_idx)
        direction_info[layer_idx] = {"var_pc1": var_pc1}
        directions_by_layer[layer_idx] = direction
        print(f"  layer {layer_idx}: PC1 var={var_pc1:.3f}")

    print("Collecting donor per-token coefficients on eval split...")
    donor_coeff_batches = collect_donor_coeff_batches(
        donor, tok, eval_texts, directions_by_layer
    )

    print("Computing donor mean coefficient at layer 5...")
    donor_mean_proj = compute_donor_mean_proj(
        donor, tok, fit_texts, directions_by_layer[SURGERY_LAYER], SURGERY_LAYER
    )
    print(f"  donor mean proj (l5) = {donor_mean_proj:.4f}")

    print("\nEvaluating diagnostic mean injection: layer 5...")
    inject_l5_per_seq, inject_l5 = evaluate_condition(
        recipient,
        tok,
        eval_texts,
        recipient_clean_per_seq,
        donor_clean_per_seq,
        hook_fns_by_layer={
            SURGERY_LAYER: make_inject_hook(
                directions_by_layer[SURGERY_LAYER], donor_mean_proj
            )
        },
    )
    print(
        f"  inject_l5_mean NLL = {inject_l5['nll']['mean']:.4f}  "
        f"gap_closed = {inject_l5['gap_closed_pct']['mean']:.2f}%"
    )

    print("\nEvaluating primary exact replacement: layer 5...")
    replace_l5_per_seq, replace_l5 = evaluate_condition(
        recipient,
        tok,
        eval_texts,
        recipient_clean_per_seq,
        donor_clean_per_seq,
        batch_hook_factories_by_layer=make_batch_replace_factories(
            directions_by_layer, donor_coeff_batches, PRIMARY_LAYERS
        ),
    )
    print(
        f"  replace_l5_exact NLL = {replace_l5['nll']['mean']:.4f}  "
        f"gap_closed = {replace_l5['gap_closed_pct']['mean']:.2f}%"
    )

    print("\nEvaluating secondary exact replacement: layers [2, 5, 8, 11]...")
    replace_early4_per_seq, replace_early4 = evaluate_condition(
        recipient,
        tok,
        eval_texts,
        recipient_clean_per_seq,
        donor_clean_per_seq,
        batch_hook_factories_by_layer=make_batch_replace_factories(
            directions_by_layer, donor_coeff_batches, SECONDARY_LAYERS
        ),
    )
    print(
        f"  replace_early4_exact NLL = {replace_early4['nll']['mean']:.4f}  "
        f"gap_closed = {replace_early4['gap_closed_pct']['mean']:.2f}%"
    )

    primary_gap = replace_l5["gap_closed_pct"]["mean"]
    primary_ci_lo = replace_l5["gap_closed_pct"]["ci_lo"]
    secondary_gap = replace_early4["gap_closed_pct"]["mean"]
    secondary_ci_lo = replace_early4["gap_closed_pct"]["ci_lo"]
    mean_gap = inject_l5["gap_closed_pct"]["mean"]
    best_exact_gap = max(primary_gap, secondary_gap)

    if primary_gap >= PASS_GAP_CLOSED_PCT and primary_ci_lo > 0.0:
        verdict = (
            f"PASS: random-init layer-5 exact replacement closes {primary_gap:.1f}% "
            "of the donor-recipient gap. A trained donor can directly improve an "
            "untrained twin at zero gradient steps."
        )
    elif secondary_gap >= PASS_GAP_CLOSED_PCT and secondary_ci_lo > 0.0:
        verdict = (
            f"PASS: early-4 exact replacement closes {secondary_gap:.1f}% of the "
            "donor-recipient gap. Transfer is real, but distributed scaffold "
            "injection outperforms single-layer surgery for a random-init twin."
        )
    elif primary_gap >= PARTIAL_GAP_CLOSED_PCT and primary_ci_lo > 0.0:
        verdict = (
            f"PARTIAL: layer-5 exact replacement closes {primary_gap:.1f}% of the "
            "donor-recipient gap. The critical direction transfers some useful "
            "structure, but not enough to stand alone."
        )
    elif secondary_gap >= PARTIAL_GAP_CLOSED_PCT and secondary_ci_lo > 0.0:
        verdict = (
            f"PARTIAL: early-4 exact replacement closes {secondary_gap:.1f}% of the "
            "donor-recipient gap. Multi-layer scaffold helps, but transfer remains "
            "limited."
        )
    elif best_exact_gap < KILL_GAP_CLOSED_PCT:
        verdict = (
            f"KILL: best exact condition closes only {best_exact_gap:.1f}% of the "
            "donor-recipient gap. Sentence-boundary PC1 alone is not sufficient "
            "for zero-step transfusion into a random-init twin."
        )
    else:
        verdict = (
            f"INCONCLUSIVE: layer-5={primary_gap:.1f}% (CI lo {primary_ci_lo:.1f}%), "
            f"early-4={secondary_gap:.1f}% (CI lo {secondary_ci_lo:.1f}%). "
            "Re-run before escalating to cross-architecture recipients."
        )

    print("\n=== CROSS-MODEL SURGERY SUMMARY ===")
    print(f"  donor clean NLL:        {donor_clean_stats['mean']:.4f}")
    print(f"  recipient clean NLL:    {recipient_clean_stats['mean']:.4f}")
    print(
        f"  inject_l5_mean NLL:     {inject_l5['nll']['mean']:.4f}  "
        f"gap_closed={mean_gap:.2f}%"
    )
    print(
        f"  replace_l5_exact NLL:   {replace_l5['nll']['mean']:.4f}  "
        f"gap_closed={primary_gap:.2f}%"
    )
    print(
        f"  replace_early4_exact:   {replace_early4['nll']['mean']:.4f}  "
        f"gap_closed={secondary_gap:.2f}%"
    )
    print(f"  verdict: {verdict}")

    out = {
        "donor_model": MODEL_ID,
        "recipient_type_first": PRIMARY_RECIPIENT,
        "random_init_first": True,
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
            "donor_clean": {"layers": [], "nll": donor_clean_stats},
            "recipient_clean": {"layers": [], "nll": recipient_clean_stats},
            "inject_l5_mean": {"layers": PRIMARY_LAYERS, **inject_l5},
            "replace_l5_exact": {"layers": PRIMARY_LAYERS, **replace_l5},
            "replace_early4_exact": {"layers": SECONDARY_LAYERS, **replace_early4},
        },
        "recommended_protocol": {
            "recipient_type": PRIMARY_RECIPIENT,
            "inject_layers": PRIMARY_LAYERS,
            "primary_mode": "per_token_exact_replacement",
            "secondary_mode": "early4_exact_replacement",
            "why": (
                "Random-init same-architecture transfer is the cleanest direct test "
                "of the moonshot end goal. It keeps tokenizer and hidden-size "
                "alignment exact, so any failure is informative rather than a "
                "cross-architecture bookkeeping confound."
            ),
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }

    out_path = RESULTS / "genome_117_cross_model_surgery.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
