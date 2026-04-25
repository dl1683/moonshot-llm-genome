"""
genome_118_checkpoint_surgery.py

TRAINING-THRESHOLD SURGERY TEST — Pythia-160M checkpoint series.

genome_117 KILL established: PC1 injection into a random-init twin closes 0%
of the donor-recipient gap because random downstream weights cannot read from
the injected direction. Trained readout alignment is required.

This experiment measures WHEN during training the recipient becomes aligned
enough for surgery to work.

Protocol:
  - Donor: Pythia-160M step-143000 (fully trained)
  - Recipients: Pythia-160M at log-spaced checkpoints
      [step0, step1, step8, step64, step512, step4000, step32000, step143000]
  - Surgery: exact per-token PC1 replacement at layer 3 (sentence-boundary axis)
  - Measure: gap_closed_% = (NLL_recip - NLL_after) / (NLL_recip - NLL_donor) * 100
    where NLL_donor is always the fully-trained donor (fixed ceiling)

Key question:
  Is there a training threshold after which PC1 injection begins to transfer
  meaningful capability? If the curve rises from 0% (step-0) to some positive
  value at intermediate checkpoints, that threshold is the minimum training
  budget for surgery to work.

Pass: any checkpoint closes >=10% gap with CI_lo > 0
Partial: any checkpoint closes >=3% gap with CI_lo > 0
Kill: all checkpoints <=3% gap (PC1 surgery never works on same-arch partially-trained)

Results: results/genome_118_checkpoint_surgery.json
"""

import json
import pathlib
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT    = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID  = "EleutherAI/pythia-160m"
SEED      = 42

N_FIT     = 200
N_EVAL    = 100
SEQ_LEN   = 64
BATCH     = 8
N_BOOT    = 500

SURGERY_LAYER = 3  # Layer 3: sentence-boundary axis confirmed by genome_116e

# Log-spaced checkpoints across training
CHECKPOINTS = [
    "step0",
    "step1",
    "step8",
    "step64",
    "step512",
    "step4000",
    "step32000",
    "step143000",
]

PASS_GAP_CLOSED_PCT    = 10.0
PARTIAL_GAP_CLOSED_PCT = 3.0
KILL_GAP_CLOSED_PCT    = 3.0


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

def load_pythia(revision):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    return model, tok


def tokenize(texts, tok):
    enc = tok(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=SEQ_LEN)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


# ---------------------------------------------------------------------------
# Direction extraction (Pythia uses gpt_neox.layers[i])
# ---------------------------------------------------------------------------

def extract_pc1(model, tok, fit_texts, layer_idx):
    pooled = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        pooled.append(h.detach().float().mean(dim=1).cpu())

    handle = model.gpt_neox.layers[layer_idx].register_forward_hook(hook_fn)
    for i in range(0, len(fit_texts), BATCH):
        ids, mask = tokenize(fit_texts[i:i+BATCH], tok)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
    handle.remove()

    acts = torch.cat(pooled, dim=0).numpy()
    pca  = PCA(n_components=5)
    pca.fit(acts)
    pc1  = pca.components_[0]
    pc1 /= (np.linalg.norm(pc1) + 1e-8)
    return pc1, float(pca.explained_variance_ratio_[0])


def collect_donor_coeff_batches(donor, tok, texts, direction, layer_idx):
    dir_t = torch.tensor(direction, dtype=torch.float32, device=DEVICE)
    dir_t = dir_t / (dir_t.norm() + 1e-8)
    coeff_batches = []
    mask_list     = []

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        coeff_batches.append(torch.matmul(h.detach().float(), dir_t).cpu())

    handle = donor.gpt_neox.layers[layer_idx].register_forward_hook(hook_fn)
    for i in range(0, len(texts), BATCH):
        ids, mask = tokenize(texts[i:i+BATCH], tok)
        mask_list.append(mask.cpu().float())
        with torch.no_grad():
            donor(input_ids=ids, attention_mask=mask)
    handle.remove()

    coeff_batches = [c * m for c, m in zip(coeff_batches, mask_list)]
    return coeff_batches


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def make_replace_hook(direction, donor_coeff_batch):
    dir_t    = torch.tensor(direction, dtype=torch.float32, device=DEVICE)
    dir_t    = dir_t / (dir_t.norm() + 1e-8)
    dir_view = dir_t.view(1, 1, -1)
    donor_c  = donor_coeff_batch.detach().cpu().float()

    def hook_fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        dc = donor_c.to(device=h.device, dtype=torch.float32).unsqueeze(-1)
        h_float = h.float()
        recip_c = torch.matmul(h_float, dir_t).unsqueeze(-1)
        h_new   = (h_float - recip_c * dir_view + dc * dir_view).to(h.dtype)
        return (h_new,) + out[1:] if is_tuple else h_new
    return hook_fn


# ---------------------------------------------------------------------------
# NLL measurement
# ---------------------------------------------------------------------------

def logits_to_nll(logits, input_ids, attention_mask):
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous().clone()
    shift_mask   = attention_mask[:, 1:].contiguous()
    shift_labels[shift_mask == 0] = -100
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
        ignore_index=-100,
    ).item()


def measure_nll_per_seq(model, tok, texts, batch_hook_factories=None):
    per_seq = []
    for batch_idx, i in enumerate(range(0, len(texts), BATCH)):
        batch_handles = []
        if batch_hook_factories:
            batch_handles.append(
                model.gpt_neox.layers[SURGERY_LAYER].register_forward_hook(
                    batch_hook_factories(batch_idx)
                )
            )
        ids, mask = tokenize(texts[i:i+BATCH], tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        for j in range(ids.shape[0]):
            per_seq.append(
                logits_to_nll(out.logits[j:j+1], ids[j:j+1], mask[j:j+1])
            )
        for h in batch_handles:
            h.remove()
    return np.array(per_seq)


def bootstrap_gap_closed(baseline_nlls, donor_nlls, repaired_nlls,
                          n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n   = len(baseline_nlls)

    def point(b, d, r):
        gap = b.mean() - d.mean()
        if abs(gap) < 1e-6:
            return 0.0
        return float((b.mean() - r.mean()) / gap * 100.0)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(point(baseline_nlls[idx], donor_nlls[idx], repaired_nlls[idx]))
    pt = point(baseline_nlls, donor_nlls, repaired_nlls)
    return {
        "mean":   float(pt),
        "ci_lo":  float(np.percentile(boots, 2.5)),
        "ci_hi":  float(np.percentile(boots, 97.5)),
    }


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=SEED):
    rng   = np.random.default_rng(seed)
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    return {
        "mean":  float(np.mean(values)),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"genome_118: checkpoint surgery sweep on {MODEL_ID}")
    print(f"  surgery_layer={SURGERY_LAYER}, n_fit={N_FIT}, n_eval={N_EVAL}")
    print(f"  checkpoints={CHECKPOINTS}")

    print("\nLoading data...")
    fit_texts  = load_wikitext_split(N_FIT, offset=0)
    eval_texts = load_wikitext_split(N_EVAL, offset=N_FIT)
    print(f"  fit={len(fit_texts)}, eval={len(eval_texts)}")

    print("\nLoading donor (step-143000, fully trained)...")
    donor, tok = load_pythia("step143000")
    print("Fitting donor PC1...")
    pc1, var_pc1 = extract_pc1(donor, tok, fit_texts, SURGERY_LAYER)
    print(f"  donor PC1 var={var_pc1:.3f}")

    print("Measuring donor clean NLL...")
    donor_nlls = measure_nll_per_seq(donor, tok, eval_texts)
    donor_stats = bootstrap_mean_ci(donor_nlls)
    print(f"  donor NLL={donor_stats['mean']:.4f}")

    print("Collecting donor per-token coefficients on eval split...")
    donor_coeff_batches = collect_donor_coeff_batches(
        donor, tok, eval_texts, pc1, SURGERY_LAYER
    )

    del donor
    torch.cuda.empty_cache()

    results_by_checkpoint = {}

    for ckpt in CHECKPOINTS:
        print(f"\n--- Recipient: {ckpt} ---")
        t_ckpt = time.time()

        recip, _ = load_pythia(ckpt)

        recip_nlls  = measure_nll_per_seq(recip, tok, eval_texts)
        recip_stats = bootstrap_mean_ci(recip_nlls)
        print(f"  recipient NLL={recip_stats['mean']:.4f}")

        def make_factory(idx):
            def factory(batch_idx):
                return make_replace_hook(pc1, donor_coeff_batches[batch_idx])
            return factory

        surgery_nlls = measure_nll_per_seq(
            recip, tok, eval_texts,
            batch_hook_factories=make_factory(None)
        )
        surgery_stats = bootstrap_mean_ci(surgery_nlls)
        gap = bootstrap_gap_closed(recip_nlls, donor_nlls, surgery_nlls)

        print(
            f"  surgery NLL={surgery_stats['mean']:.4f}  "
            f"gap_closed={gap['mean']:.2f}% [CI {gap['ci_lo']:.2f}%, {gap['ci_hi']:.2f}%]"
        )

        results_by_checkpoint[ckpt] = {
            "recipient_nll":  recip_stats,
            "surgery_nll":    surgery_stats,
            "gap_closed_pct": gap,
            "elapsed_s":      time.time() - t_ckpt,
        }

        del recip
        torch.cuda.empty_cache()

    # Determine verdict
    best_gap     = max(r["gap_closed_pct"]["mean"] for r in results_by_checkpoint.values())
    best_ci_lo   = max(r["gap_closed_pct"]["ci_lo"] for r in results_by_checkpoint.values())
    best_ckpt    = max(results_by_checkpoint,
                       key=lambda k: results_by_checkpoint[k]["gap_closed_pct"]["mean"])

    if best_gap >= PASS_GAP_CLOSED_PCT and best_ci_lo > 0:
        verdict = (
            f"PASS: {best_ckpt} closes {best_gap:.1f}% of the donor-recipient gap "
            f"(CI_lo={best_ci_lo:.1f}%). Training threshold for surgery identified."
        )
    elif best_gap >= PARTIAL_GAP_CLOSED_PCT and best_ci_lo > 0:
        verdict = (
            f"PARTIAL: {best_ckpt} closes {best_gap:.1f}% of the donor-recipient gap "
            f"(CI_lo={best_ci_lo:.1f}%). Weak signal above noise floor."
        )
    else:
        verdict = (
            f"KILL: best checkpoint {best_ckpt} closes only {best_gap:.1f}% "
            "(CI_lo<=0%). PC1 surgery does not work even for partially-trained "
            "same-arch recipients."
        )

    print("\n=== CHECKPOINT SURGERY SUMMARY ===")
    print(f"  donor NLL:     {donor_stats['mean']:.4f}")
    for ckpt, r in results_by_checkpoint.items():
        print(
            f"  {ckpt:12s}: recip={r['recipient_nll']['mean']:.4f}  "
            f"surgery={r['surgery_nll']['mean']:.4f}  "
            f"gap_closed={r['gap_closed_pct']['mean']:.2f}%"
        )
    print(f"  verdict: {verdict}")

    out = {
        "model":          MODEL_ID,
        "surgery_layer":  SURGERY_LAYER,
        "checkpoints":    CHECKPOINTS,
        "n_fit":          N_FIT,
        "n_eval":         N_EVAL,
        "donor_revision": "step143000",
        "donor_pc1_var":  var_pc1,
        "donor_nll":      donor_stats,
        "pass_criteria": {
            "pass_gap_closed_pct":    PASS_GAP_CLOSED_PCT,
            "partial_gap_closed_pct": PARTIAL_GAP_CLOSED_PCT,
            "kill_gap_closed_pct":    KILL_GAP_CLOSED_PCT,
        },
        "results_by_checkpoint": {
            k: {
                "recipient_nll_mean":    v["recipient_nll"]["mean"],
                "recipient_nll_ci":      v["recipient_nll"],
                "surgery_nll_mean":      v["surgery_nll"]["mean"],
                "surgery_nll_ci":        v["surgery_nll"],
                "gap_closed_pct":        v["gap_closed_pct"],
                "elapsed_s":             v["elapsed_s"],
            }
            for k, v in results_by_checkpoint.items()
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }

    out_path = RESULTS / "genome_118_checkpoint_surgery.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
