"""
genome_122_scale_calibrated_transfer.py

SCALE-CALIBRATED TRANSFER: UNTESTED ARM + NORM CALIBRATION.

genome_121 revealed the norm catastrophe: adding donor layer norms to a
random-init recipient causes -52 to -77 pp degradation. The mechanism:
donor norms assume donor activation statistics; in the wrong context
they amplify rather than normalize.

genome_121 also left one key arm untested: embed_attn_zero_mlp
(donor embed + donor attn, random norms at gamma=1, zero MLP).
This removes all three interference sources:
  - random MLP interference (zeroed)
  - donor norm scale mismatch (left at gamma=1 init)
  - random LM head (tied with donor embed in Qwen3)

This experiment tests:
  1. embed_attn_zero_mlp — the untested combination
  2. embed_attn_calib_zero_mlp — same but with calibrated norms
     (gamma adjusted so transplanted model output scale matches donor)
  3. all_attn_zero_mlp — attn-only without embed (control)
  4. embed_attn — genome_121 anchor (-1.59%)
  5. all_attn — genome_121 anchor (+0.89%)
  6. full_exact — 100% positive control

Norm calibration protocol:
  - Run N_CALIB sequences through donor, collect per-RMSNorm layer
    mean RMS of pre-norm activations
  - Run same sequences through transplanted model (embed+attn+gamma=1)
  - For each norm: gamma_new = donor_rms / transplant_rms
    (so post-norm output scale matches donor's)

Pass:    embed_attn_zero_mlp OR calib variant closes >=10% gap, CI_lo > 0
Partial: any arm closes >=5% gap, CI_lo > 0
Kill:    all non-full arms close <5% gap

Results: results/genome_122_scale_calibrated_transfer.json
"""

import copy
import json
import pathlib
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
BASE_SEED = 42
SEEDS = [7, 13, 42]

N_EVAL = 200
N_CALIB = 200
SEQ_LEN = 64
BATCH = 8
N_BOOT = 500

PASS_GAP = 10.0
PARTIAL_GAP = 5.0


def load_wikitext(n, offset, seed=BASE_SEED):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    out, count = [], 0
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


def tokenize(texts, tok):
    enc = tok(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=SEQ_LEN)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def load_trained():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    return model, tok


def load_random_init(seed):
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).to(DEVICE).eval()


def apply_embed_attn(donor, recipient, zero_mlp=False):
    m = copy.deepcopy(recipient)
    sd_d = donor.state_dict()
    sd_m = m.state_dict()
    n_copied, n_zeroed = 0, 0
    for k in sd_d:
        if k in ("model.embed_tokens.weight", "lm_head.weight") or ".self_attn." in k:
            sd_m[k] = sd_d[k].clone()
            n_copied += sd_d[k].numel()
        elif zero_mlp and ".mlp." in k:
            sd_m[k] = torch.zeros_like(sd_m[k])
            n_zeroed += sd_m[k].numel()
    m.load_state_dict(sd_m, strict=True)
    total = sum(p.numel() for p in donor.parameters())
    return m, n_copied, n_zeroed, total


def apply_all_attn(donor, recipient, zero_mlp=False):
    m = copy.deepcopy(recipient)
    sd_d = donor.state_dict()
    sd_m = m.state_dict()
    n_copied, n_zeroed = 0, 0
    for k in sd_d:
        if ".self_attn." in k:
            sd_m[k] = sd_d[k].clone()
            n_copied += sd_d[k].numel()
        elif zero_mlp and ".mlp." in k:
            sd_m[k] = torch.zeros_like(sd_m[k])
            n_zeroed += sd_m[k].numel()
    m.load_state_dict(sd_m, strict=True)
    total = sum(p.numel() for p in donor.parameters())
    return m, n_copied, n_zeroed, total


def apply_full_exact(donor, recipient):
    m = copy.deepcopy(recipient)
    m.load_state_dict(donor.state_dict(), strict=True)
    total = sum(p.numel() for p in donor.parameters())
    return m, total, 0, total


def collect_norm_rms(model, tok, texts):
    """Collect mean RMS of pre-norm activations for each RMSNorm layer."""
    rms_by_layer = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0].float()
            rms = x.pow(2).mean(-1).sqrt().mean().item()
            rms_by_layer[name].append(rms)
        return hook

    for name, module in model.named_modules():
        if "layernorm" in name.lower() or "norm" in name.lower():
            if hasattr(module, "weight") and module.weight is not None:
                if len(module.weight.shape) == 1:
                    h = module.register_forward_hook(make_hook(name))
                    hooks.append(h)

    model.eval()
    for i in range(0, len(texts), BATCH):
        ids, mask = tokenize(texts[i:i + BATCH], tok)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)

    for h in hooks:
        h.remove()

    return {k: float(np.mean(v)) for k, v in rms_by_layer.items()}


def calibrate_norms(transplanted, donor_rms, transplant_rms):
    """Adjust RMSNorm gammas so transplanted output scale matches donor."""
    m = copy.deepcopy(transplanted)
    with torch.no_grad():
        for name, module in m.named_modules():
            if name in donor_rms and name in transplant_rms:
                d_rms = donor_rms[name]
                t_rms = transplant_rms[name]
                if t_rms > 1e-8 and hasattr(module, "weight"):
                    ratio = d_rms / t_rms
                    module.weight.mul_(ratio)
    return m


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


def measure_nll(model, tok, texts):
    per_seq = []
    for i in range(0, len(texts), BATCH):
        ids, mask = tokenize(texts[i:i + BATCH], tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        for j in range(ids.shape[0]):
            per_seq.append(
                logits_to_nll(out.logits[j:j+1], ids[j:j+1], mask[j:j+1])
            )
    return np.array(per_seq)


def bsci(arr, n_boot=N_BOOT, seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(arr.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


def bootstrap_gap(recip, donor, surgery, n_boot=N_BOOT, seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    n = len(recip)

    def pt(r, d, s):
        gap = r.mean() - d.mean()
        return 0.0 if abs(gap) < 1e-6 else float((r.mean() - s.mean()) / gap * 100)

    boots = [pt(recip[rng.integers(0, n, n)], donor[rng.integers(0, n, n)],
                surgery[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return {"mean": float(pt(recip, donor, surgery)),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


def main():
    t0 = time.time()
    print(f"genome_122: scale-calibrated transfer on {MODEL_ID}")
    print(f"  seeds={SEEDS}  n_eval={N_EVAL}  n_calib={N_CALIB}")

    eval_texts = load_wikitext(N_EVAL, offset=0)
    calib_texts = load_wikitext(N_CALIB, offset=N_EVAL)
    print(f"  eval={len(eval_texts)}  calib={len(calib_texts)}")

    print("Loading donor...")
    donor, tok = load_trained()
    donor_nlls = measure_nll(donor, tok, eval_texts)
    donor_stats = bsci(donor_nlls)
    print(f"  donor NLL={donor_stats['mean']:.4f}")

    print("Collecting donor norm RMS stats for calibration...")
    donor_rms = collect_norm_rms(donor, tok, calib_texts)
    print(f"  collected {len(donor_rms)} norm layers")

    arm_names = [
        "embed_attn",
        "embed_attn_zero_mlp",
        "embed_attn_calib_zero_mlp",
        "all_attn",
        "all_attn_zero_mlp",
        "full_exact",
    ]

    pooled_recip, pooled_donor = [], []
    pooled_by_arm = {n: [] for n in arm_names}
    arm_meta = {}
    per_seed = []

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        recipient = load_random_init(seed)
        recip_nlls = measure_nll(recipient, tok, eval_texts)
        recip_stats = bsci(recip_nlls, seed=seed)
        print(f"  recipient NLL={recip_stats['mean']:.4f}")
        pooled_recip.append(recip_nlls)
        pooled_donor.append(donor_nlls)

        seed_results = {"seed": seed, "recipient_nll": recip_stats, "arms": {}}

        for arm_name in arm_names:
            if arm_name == "embed_attn":
                m, nc, nz, tot = apply_embed_attn(donor, recipient, zero_mlp=False)
            elif arm_name == "embed_attn_zero_mlp":
                m, nc, nz, tot = apply_embed_attn(donor, recipient, zero_mlp=True)
            elif arm_name == "embed_attn_calib_zero_mlp":
                base, nc, nz, tot = apply_embed_attn(donor, recipient, zero_mlp=True)
                print(f"    calibrating norms...")
                transplant_rms = collect_norm_rms(base, tok, calib_texts)
                m = calibrate_norms(base, donor_rms, transplant_rms)
                del base
            elif arm_name == "all_attn":
                m, nc, nz, tot = apply_all_attn(donor, recipient, zero_mlp=False)
            elif arm_name == "all_attn_zero_mlp":
                m, nc, nz, tot = apply_all_attn(donor, recipient, zero_mlp=True)
            elif arm_name == "full_exact":
                m, nc, nz, tot = apply_full_exact(donor, recipient)

            surg_nlls = measure_nll(m, tok, eval_texts)
            surg_stats = bsci(surg_nlls, seed=seed)
            gap = bootstrap_gap(recip_nlls, donor_nlls, surg_nlls, seed=seed)
            cp_pct = nc / tot * 100
            zr_pct = nz / tot * 100
            print(f"  {arm_name:30s} cp={cp_pct:5.1f}% zr={zr_pct:5.1f}%  "
                  f"NLL={surg_stats['mean']:.4f}  gap={gap['mean']:.2f}%  "
                  f"[CI {gap['ci_lo']:.2f}%, {gap['ci_hi']:.2f}%]")

            pooled_by_arm[arm_name].append(surg_nlls)
            if arm_name not in arm_meta:
                arm_meta[arm_name] = {"params_copied_pct": cp_pct,
                                      "params_zeroed_pct": zr_pct}
            seed_results["arms"][arm_name] = {
                "surgery_nll": surg_stats, "gap_closed_pct": gap}
            del m
            torch.cuda.empty_cache()

        per_seed.append(seed_results)
        del recipient
        torch.cuda.empty_cache()

    # Pool across seeds
    pr = np.concatenate(pooled_recip)
    pd = np.concatenate(pooled_donor)
    agg = {}
    for name in arm_names:
        ps = np.concatenate(pooled_by_arm[name])
        agg[name] = {
            **arm_meta[name],
            "surgery_nll": bsci(ps),
            "gap_closed_pct": bootstrap_gap(pr, pd, ps),
        }

    # Verdict
    best_name = max(
        (n for n in arm_names if n != "full_exact"),
        key=lambda n: agg[n]["gap_closed_pct"]["mean"]
    )
    best = agg[best_name]["gap_closed_pct"]
    full_err = abs(agg["full_exact"]["surgery_nll"]["mean"] - donor_stats["mean"])

    if best["mean"] >= PASS_GAP and best["ci_lo"] > 0:
        verdict = (f"PASS: {best_name} closes {best['mean']:.1f}% of gap "
                   f"[CI {best['ci_lo']:.1f}%, {best['ci_hi']:.1f}%]. "
                   "Scale-calibrated transfer partially breaks holism barrier.")
    elif best["mean"] >= PARTIAL_GAP and best["ci_lo"] > 0:
        verdict = (f"PARTIAL: {best_name} closes {best['mean']:.1f}% of gap "
                   f"[CI {best['ci_lo']:.1f}%, {best['ci_hi']:.1f}%]. "
                   "Weak signal — barrier still holds at practical level.")
    else:
        verdict = (f"KILL: best arm ({best_name}) closes {best['mean']:.1f}%. "
                   "Scale calibration does not break the holism barrier. "
                   "Pivot to curriculum learning.")

    print("\n=== GENOME 122 SUMMARY ===")
    print(f"  donor NLL:     {donor_stats['mean']:.4f}")
    for name, res in agg.items():
        print(f"  {name:30s} cp={res['params_copied_pct']:5.1f}% "
              f"zr={res['params_zeroed_pct']:5.1f}%  "
              f"NLL={res['surgery_nll']['mean']:.4f}  "
              f"gap={res['gap_closed_pct']['mean']:.2f}%")
    print(f"  full_exact error vs donor: {full_err:.4f} nats")
    print(f"  verdict: {verdict}")

    out = {
        "model": MODEL_ID,
        "genome": 122,
        "name": "scale_calibrated_transfer",
        "n_eval": N_EVAL,
        "n_calib": N_CALIB,
        "seeds": SEEDS,
        "donor_nll": donor_stats,
        "criteria": {"pass_gap_pct": PASS_GAP, "partial_gap_pct": PARTIAL_GAP},
        "aggregate_results": {
            name: {
                "params_copied_pct": res["params_copied_pct"],
                "params_zeroed_pct": res["params_zeroed_pct"],
                "surgery_nll_mean": res["surgery_nll"]["mean"],
                "surgery_nll_ci": res["surgery_nll"],
                "gap_closed_pct": res["gap_closed_pct"],
            }
            for name, res in agg.items()
        },
        "per_seed": per_seed,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }

    out_path = RESULTS / "genome_122_scale_calibrated_transfer.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
