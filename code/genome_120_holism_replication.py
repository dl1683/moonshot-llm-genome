"""
genome_120_holism_replication.py

HOLISM BARRIER CROSS-ARCHITECTURE REPLICATION.

genome_119 established on Pythia-160M: all weight component transfers hurt;
capability is holistic. Codex verdict: replicate on a second architecture
to confirm the holism barrier generalizes (required for any publication).

This experiment repeats the genome_119 protocol on Qwen3-0.6B:
  - Donor: trained Qwen3-0.6B (d=1024, 28 layers)
  - Recipient: random-init Qwen3-0.6B (same architecture)
  - 7 component conditions: embed_only, lm_head_only, layer0_mlp,
    early_mlp (layers 0-3), all_mlp, all_attn, all_layers

Key differences from genome_119:
  - Qwen3 uses model.model.layers[i] (not gpt_neox.layers)
  - Embed: model.model.embed_tokens (not gpt_neox.embed_in)
  - LM head: model.lm_head (not embed_out)
  - d_model=1024, n_layers=28, intermediate_size=2048

If KILL across both architectures: holism barrier is cross-architecture.
If PASS on Qwen3 but KILL on Pythia: the effect is architecture-specific.

Pass:   any component closes >=20% gap with CI_lo > 0
Partial: any component closes >=5% gap with CI_lo > 0
Kill:   all components <= 5% gap

Results: results/genome_120_holism_replication.json
"""

import copy
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT    = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED     = 42

N_EVAL  = 200
SEQ_LEN = 64
BATCH   = 8
N_BOOT  = 500

PASS_GAP    = 20.0
PARTIAL_GAP = 5.0
KILL_GAP    = 5.0

COMPONENTS = [
    "embed_only",
    "lm_head_only",
    "layer0_mlp",
    "early_mlp",
    "all_mlp",
    "all_attn",
    "all_layers",
]


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


# ---------------------------------------------------------------------------
# Weight copying
# ---------------------------------------------------------------------------

def copy_weights(donor, recipient, component):
    recip_copy = copy.deepcopy(recipient)
    sd_donor   = donor.state_dict()
    sd_recip   = recip_copy.state_dict()
    n_copied   = 0

    def _copy(prefix):
        nonlocal n_copied
        for k in sd_donor:
            if k.startswith(prefix):
                sd_recip[k] = sd_donor[k].clone()
                n_copied   += sd_donor[k].numel()
        recip_copy.load_state_dict(sd_recip)

    if component == "embed_only":
        _copy("model.embed_tokens")
    elif component == "lm_head_only":
        _copy("lm_head")
    elif component == "embed_and_head":
        _copy("model.embed_tokens")
        _copy("lm_head")
    elif component == "layer0_mlp":
        _copy("model.layers.0.mlp")
    elif component == "early_mlp":
        for i in range(4):
            _copy(f"model.layers.{i}.mlp")
    elif component == "all_mlp":
        for i in range(28):
            _copy(f"model.layers.{i}.mlp")
    elif component == "all_attn":
        for i in range(28):
            _copy(f"model.layers.{i}.self_attn")
    elif component == "all_layers":
        _copy("model.layers")
    else:
        raise ValueError(f"Unknown component: {component}")

    total = sum(p.numel() for p in donor.parameters())
    return recip_copy, n_copied, total


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


def bsci(arr, n_boot=N_BOOT, seed=SEED):
    rng   = np.random.default_rng(seed)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(arr.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


def bootstrap_gap_closed(recip, donor, surgery, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n   = len(recip)

    def pt(r, d, s):
        gap = r.mean() - d.mean()
        return 0.0 if abs(gap) < 1e-6 else float((r.mean() - s.mean()) / gap * 100.0)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(pt(recip[idx], donor[idx], surgery[idx]))
    return {"mean": float(pt(recip, donor, surgery)),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"genome_120: holism replication on {MODEL_ID} (cross-arch from genome_119)")

    eval_texts = load_wikitext(N_EVAL, offset=0)
    print(f"  eval={len(eval_texts)}")

    print("Loading donor (trained Qwen3-0.6B)...")
    donor, tok = load_trained()
    donor_nlls  = measure_nll(donor, tok, eval_texts)
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
        pct = n_copied / total * 100

        surg_nlls = measure_nll(model_copy, tok, eval_texts)
        surg_stats = bsci(surg_nlls)
        gap = bootstrap_gap_closed(recip_nlls, donor_nlls, surg_nlls)

        print(
            f"  params={pct:.1f}%  NLL={surg_stats['mean']:.4f}  "
            f"gap_closed={gap['mean']:.2f}%  [CI {gap['ci_lo']:.2f}%, {gap['ci_hi']:.2f}%]"
        )

        results_by_component[comp] = {
            "params_copied_pct": pct,
            "surgery_nll":       surg_stats,
            "gap_closed_pct":    gap,
        }

        del model_copy
        torch.cuda.empty_cache()

    del donor, recipient
    torch.cuda.empty_cache()

    best_gap   = max(r["gap_closed_pct"]["mean"] for r in results_by_component.values())
    best_ci_lo = max(r["gap_closed_pct"]["ci_lo"] for r in results_by_component.values())
    best_comp  = max(results_by_component,
                     key=lambda k: results_by_component[k]["gap_closed_pct"]["mean"])

    if best_gap >= PASS_GAP and best_ci_lo > 0:
        verdict = (
            f"PASS: {best_comp} closes {best_gap:.1f}% gap (CI_lo={best_ci_lo:.1f}%). "
            "Holism barrier does NOT generalize to Qwen3."
        )
    elif best_gap >= PARTIAL_GAP and best_ci_lo > 0:
        verdict = (
            f"PARTIAL: {best_comp} closes {best_gap:.1f}% gap (CI_lo={best_ci_lo:.1f}%). "
            "Weak signal on Qwen3."
        )
    else:
        verdict = (
            f"KILL: best {best_comp} closes {best_gap:.1f}%. "
            "Holism barrier confirmed on Qwen3 — cross-architecture generalization established."
        )

    print("\n=== HOLISM REPLICATION SUMMARY (Qwen3-0.6B) ===")
    print(f"  donor NLL:     {donor_stats['mean']:.4f}")
    print(f"  recipient NLL: {recip_stats['mean']:.4f}")
    for comp, r in results_by_component.items():
        print(
            f"  {comp:20s}: {r['params_copied_pct']:5.1f}%  "
            f"NLL={r['surgery_nll']['mean']:.4f}  "
            f"gap_closed={r['gap_closed_pct']['mean']:.2f}%"
        )
    print(f"  verdict: {verdict}")

    out = {
        "model":              MODEL_ID,
        "replication_of":     "genome_119 (Pythia-160M)",
        "n_eval":             N_EVAL,
        "components_tested":  COMPONENTS,
        "donor_nll":          donor_stats,
        "recipient_nll":      recip_stats,
        "pass_criteria":      {"pass_gap_pct": PASS_GAP, "partial_gap_pct": PARTIAL_GAP,
                               "kill_gap_pct": KILL_GAP},
        "results_by_component": {
            k: {"params_copied_pct": v["params_copied_pct"],
                "surgery_nll_mean":  v["surgery_nll"]["mean"],
                "surgery_nll_ci":    v["surgery_nll"],
                "gap_closed_pct":    v["gap_closed_pct"]}
            for k, v in results_by_component.items()
        },
        "verdict":   verdict,
        "elapsed_s": time.time() - t0,
    }

    out_path = RESULTS / "genome_120_holism_replication.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
