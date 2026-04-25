"""
genome_119_weight_component_surgery.py

WEIGHT-COMPONENT ISOLATION SURGERY.

genome_117/118 established: PC1 activation injection fails at all training stages.
The bottleneck is that random downstream weights cannot decode the injected signal.

This experiment tests which INDIVIDUAL weight components carry the most capability,
by copying one component at a time from trained Pythia-160M into a random-init twin.

Components tested (smallest to largest, in bits of the total param count):
  1. embed_only:     copy token embeddings (gpt_neox.embed_in)
  2. lm_head_only:   copy output projection (embed_out, tied in Pythia)
  3. embed+head:     copy both (in Pythia, embed_in == embed_out, so same as embed_only)
  4. layer0_mlp:     copy MLP weights in layer 0 only
  5. early_mlp:      copy MLP weights in layers 0-3 (early-layer critical subspace region)
  6. all_mlp:        copy ALL MLP weights across all 12 layers
  7. all_attn:       copy ALL attention weights across all 12 layers

Key question:
  Which minimal weight component, when copied from donor to random-init recipient,
  closes the most capability gap? Is there a component that achieves >20% with <10%
  of total parameters copied?

Pass:   any single component closes >=20% gap with CI_lo > 0
Partial: any component closes >=5% gap with CI_lo > 0
Kill:   all components <= 5% gap (capability is not localizable to any single component)

Results: results/genome_119_weight_component_surgery.json
"""

import json
import pathlib
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

ROOT    = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "EleutherAI/pythia-160m"
SEED     = 42

N_EVAL  = 200
SEQ_LEN = 64
BATCH   = 8
N_BOOT  = 500


# ---------------------------------------------------------------------------
# Data & tokenization
# ---------------------------------------------------------------------------

def load_wikitext(n, offset, seed=SEED):
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


def tokenize(texts, tok):
    enc = tok(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=SEQ_LEN)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


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
    return AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).to(DEVICE).eval()


def copy_weights(donor, recipient, component):
    """Copy named weight component from donor into a COPY of recipient."""
    recip_copy = copy.deepcopy(recipient)
    sd_donor   = donor.state_dict()
    sd_recip   = recip_copy.state_dict()

    def _copy_if_match(prefix):
        copied, total_params = 0, 0
        for k in sd_donor:
            if k.startswith(prefix):
                sd_recip[k] = sd_donor[k].clone()
                copied     += sd_donor[k].numel()
            total_params += sd_donor[k].numel()
        recip_copy.load_state_dict(sd_recip)
        return copied

    if component == "embed_only":
        n = _copy_if_match("gpt_neox.embed_in")
    elif component == "lm_head_only":
        n = _copy_if_match("embed_out")
    elif component == "embed_and_head":
        _copy_if_match("gpt_neox.embed_in")
        n = _copy_if_match("embed_out")
    elif component == "layer0_mlp":
        n = _copy_if_match("gpt_neox.layers.0.mlp")
    elif component == "early_mlp":
        n = 0
        for i in range(4):
            n += _copy_if_match(f"gpt_neox.layers.{i}.mlp")
    elif component == "all_mlp":
        n = 0
        for i in range(12):
            n += _copy_if_match(f"gpt_neox.layers.{i}.mlp")
    elif component == "all_attn":
        n = 0
        for i in range(12):
            n += _copy_if_match(f"gpt_neox.layers.{i}.attention")
    elif component == "all_layers":
        n = _copy_if_match("gpt_neox.layers")
    else:
        raise ValueError(f"Unknown component: {component}")

    total = sum(p.numel() for p in donor.parameters())
    return recip_copy, n, total


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


def measure_nll(model, tok, texts):
    per_seq = []
    for i in range(0, len(texts), BATCH):
        ids, mask = tokenize(texts[i:i+BATCH], tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        for j in range(ids.shape[0]):
            per_seq.append(
                logits_to_nll(out.logits[j:j+1], ids[j:j+1], mask[j:j+1])
            )
    return np.array(per_seq)


def bootstrap_gap_closed(recip_nlls, donor_nlls, surgery_nlls,
                          n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n   = len(recip_nlls)

    def point(r, d, s):
        gap = r.mean() - d.mean()
        if abs(gap) < 1e-6:
            return 0.0
        return float((r.mean() - s.mean()) / gap * 100.0)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(point(recip_nlls[idx], donor_nlls[idx], surgery_nlls[idx]))
    pt = point(recip_nlls, donor_nlls, surgery_nlls)
    return {
        "mean":  float(pt),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def bsci(arr, n_boot=N_BOOT, seed=SEED):
    rng   = np.random.default_rng(seed)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(arr.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COMPONENTS = [
    "embed_only",
    "lm_head_only",
    "layer0_mlp",
    "early_mlp",
    "all_mlp",
    "all_attn",
    "all_layers",
]

PASS_GAP    = 20.0
PARTIAL_GAP = 5.0
KILL_GAP    = 5.0


def main():
    t0 = time.time()
    print(f"genome_119: weight-component isolation surgery on {MODEL_ID}")

    eval_texts = load_wikitext(N_EVAL, offset=0)
    print(f"  eval={len(eval_texts)}")

    print("Loading donor (fully trained)...")
    donor, tok = load_trained()
    donor_nlls = measure_nll(donor, tok, eval_texts)
    donor_stats = bsci(donor_nlls)
    print(f"  donor NLL={donor_stats['mean']:.4f}")

    print("Loading random-init recipient...")
    recipient   = load_random_init(tok)
    recip_nlls  = measure_nll(recipient, tok, eval_texts)
    recip_stats = bsci(recip_nlls)
    print(f"  recipient NLL={recip_stats['mean']:.4f}")
    print(f"  gap={recip_stats['mean'] - donor_stats['mean']:.4f} nats")

    results_by_component = {}

    for comp in COMPONENTS:
        print(f"\n--- Component: {comp} ---")
        model_copy, n_copied, total = copy_weights(donor, recipient, comp)
        pct_params = n_copied / total * 100

        surgery_nlls = measure_nll(model_copy, tok, eval_texts)
        surgery_stats = bsci(surgery_nlls)
        gap = bootstrap_gap_closed(recip_nlls, donor_nlls, surgery_nlls)

        print(
            f"  params_copied={pct_params:.1f}%  "
            f"surgery_NLL={surgery_stats['mean']:.4f}  "
            f"gap_closed={gap['mean']:.2f}%  [CI {gap['ci_lo']:.2f}%, {gap['ci_hi']:.2f}%]"
        )

        results_by_component[comp] = {
            "params_copied_pct": pct_params,
            "surgery_nll":       surgery_stats,
            "gap_closed_pct":    gap,
        }

        del model_copy
        torch.cuda.empty_cache()

    del donor, recipient
    torch.cuda.empty_cache()

    best_gap    = max(r["gap_closed_pct"]["mean"] for r in results_by_component.values())
    best_ci_lo  = max(r["gap_closed_pct"]["ci_lo"] for r in results_by_component.values())
    best_comp   = max(results_by_component,
                      key=lambda k: results_by_component[k]["gap_closed_pct"]["mean"])

    if best_gap >= PASS_GAP and best_ci_lo > 0:
        verdict = (
            f"PASS: {best_comp} closes {best_gap:.1f}% of the gap "
            f"(CI_lo={best_ci_lo:.1f}%, params={results_by_component[best_comp]['params_copied_pct']:.1f}%). "
            "Capability is localizable."
        )
    elif best_gap >= PARTIAL_GAP and best_ci_lo > 0:
        verdict = (
            f"PARTIAL: {best_comp} closes {best_gap:.1f}% of the gap "
            f"(CI_lo={best_ci_lo:.1f}%). Weak but real signal."
        )
    else:
        verdict = (
            f"KILL: best component {best_comp} closes only {best_gap:.1f}%. "
            "Capability is fully distributed — no single component transfers."
        )

    print("\n=== WEIGHT COMPONENT SURGERY SUMMARY ===")
    print(f"  donor NLL:     {donor_stats['mean']:.4f}")
    print(f"  recipient NLL: {recip_stats['mean']:.4f}")
    for comp, r in results_by_component.items():
        print(
            f"  {comp:20s}: {r['params_copied_pct']:5.1f}% params  "
            f"NLL={r['surgery_nll']['mean']:.4f}  "
            f"gap_closed={r['gap_closed_pct']['mean']:.2f}%"
        )
    print(f"  verdict: {verdict}")

    out = {
        "model":              MODEL_ID,
        "n_eval":             N_EVAL,
        "components_tested":  COMPONENTS,
        "donor_nll":          donor_stats,
        "recipient_nll":      recip_stats,
        "pass_criteria": {
            "pass_gap_pct":    PASS_GAP,
            "partial_gap_pct": PARTIAL_GAP,
            "kill_gap_pct":    KILL_GAP,
        },
        "results_by_component": {
            k: {
                "params_copied_pct":    v["params_copied_pct"],
                "surgery_nll_mean":     v["surgery_nll"]["mean"],
                "surgery_nll_ci":       v["surgery_nll"],
                "gap_closed_pct":       v["gap_closed_pct"],
            }
            for k, v in results_by_component.items()
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }

    out_path = RESULTS / "genome_119_weight_component_surgery.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
