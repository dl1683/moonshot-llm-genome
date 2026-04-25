"""
genome_125_frozen_attn_glue_train.py

INTERFACE-ONLY RESCUE of the only surviving donor submodule.

Codex verdict on genome_125:
"all_attn is the ONLY consistent positive signal across genome_120-124.
Norm-copy catastrophes say the bottleneck is INTERFACE CALIBRATION, not
lack of donor computation. Copy donor all_attn into random-init Qwen3,
FREEZE it, and train ONLY the glue for 100 steps."

Glue = embed_tokens (tied with lm_head) + RMSNorm gammas (input_layernorm +
post_attention_layernorm + final model.norm).

Arms:
  - frozen_attn_glue:    Recipient = random-init + donor all_attn (frozen).
                         Train embed/lm_head + all RMSNorm gammas (trainable).
  - matched_param_ctrl:  Recipient = random-init (no donor copy).
                         Train embed/lm_head + all RMSNorm gammas (trainable).
                         Same trainable parameter count as frozen_attn_glue.
                         Frozen attn weights remain at random init.
  - full_train_ctrl:     Recipient = random-init. Train ALL parameters
                         (matches genome_123 baseline).

Pass:    frozen_attn_glue closes >= 20% of gap by step 100, beats matched_param by >= 5pp.
Partial: frozen_attn_glue closes >= 10% of gap by step 100.
Kill:    frozen_attn_glue <= matched_param at step 100.

If KILL, surgery is dead. Codex-stated honest conclusion.

Results: results/genome_125_frozen_attn_glue_train.json
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

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 42

N_EVAL = 200
SEQ_LEN = 64
BATCH_TRAIN = 4
BATCH_EVAL = 8
N_BOOT = 500
MAX_STEPS = 100
EVAL_STEPS = [10, 25, 50, 100]
LR = 3e-4

PASS_GAP = 20.0
PARTIAL_GAP = 10.0
GLUE_DELTA_PASS = 5.0


def load_wikitext(n, offset, seed=SEED):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
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


def load_trained():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(DEVICE).eval(), tok


def load_random_init(seed):
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).to(DEVICE)


def copy_donor_attn(donor, recipient):
    """Copy all self_attn weights from donor to recipient."""
    sd_d = donor.state_dict()
    sd_r = recipient.state_dict()
    n_copied = 0
    for k in sd_d:
        if ".self_attn." in k:
            sd_r[k] = sd_d[k].clone()
            n_copied += sd_d[k].numel()
    recipient.load_state_dict(sd_r, strict=True)
    return n_copied


def is_glue_param(name):
    """Glue = embedding (tied with LM_head) + all RMSNorm weights."""
    if "embed_tokens" in name or name == "lm_head.weight":
        return True
    if "layernorm" in name.lower() or name == "model.norm.weight":
        return True
    return False


def is_attn_param(name):
    return ".self_attn." in name


def setup_arm_grads(model, arm):
    """Configure requires_grad based on arm type."""
    n_train, n_total = 0, 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if arm == "frozen_attn_glue":
            # Train only glue; freeze attn AND mlp.
            p.requires_grad = is_glue_param(name)
        elif arm == "matched_param_ctrl":
            # Train only glue (same as frozen_attn_glue but no donor copy).
            p.requires_grad = is_glue_param(name)
        elif arm == "full_train_ctrl":
            # Train everything.
            p.requires_grad = True
        else:
            raise ValueError(f"unknown arm {arm}")
        if p.requires_grad:
            n_train += p.numel()
    return n_train, n_total


def measure_nll_arr(model, tok, texts):
    model.eval()
    per_seq = []
    for i in range(0, len(texts), BATCH_EVAL):
        ids, mask = tokenize(texts[i:i + BATCH_EVAL], tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        for j in range(ids.shape[0]):
            lj = out.logits[j:j+1, :-1]
            lbl = ids[j:j+1, 1:].clone()
            mk = mask[j:j+1, 1:]
            lbl[mk == 0] = -100
            per_seq.append(
                F.cross_entropy(lj.view(-1, lj.size(-1)),
                                lbl.view(-1), ignore_index=-100).item()
            )
    return np.array(per_seq)


def bsci(arr, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(arr.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


def gap_closed(recip_nll, donor_nll, current_nll):
    if abs(recip_nll - donor_nll) < 1e-6:
        return 0.0
    return (recip_nll - current_nll) / (recip_nll - donor_nll) * 100.0


def train_arm(model, tok, train_texts, eval_texts, recip_nll, donor_nll,
              arm_name):
    n_train, n_total = setup_arm_grads(model, arm_name)
    print(f"  [{arm_name}] trainable params: {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.1f}%)")

    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=LR)
    model.train()
    rng = np.random.default_rng(SEED)
    nll_curve = {}
    t_arm = time.time()

    # Step 0 evaluation
    arr = measure_nll_arr(model, tok, eval_texts)
    stats = bsci(arr)
    gap = gap_closed(recip_nll, donor_nll, stats["mean"])
    nll_curve[0] = {**stats, "gap_closed_pct": gap}
    print(f"  [{arm_name}] step=  0  NLL={stats['mean']:.4f}  gap={gap:.2f}%")
    model.train()

    for step in range(1, MAX_STEPS + 1):
        idx = rng.integers(0, len(train_texts), size=BATCH_TRAIN)
        batch = [train_texts[i] for i in idx]
        ids, mask = tokenize(batch, tok)
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits
        shift = logits[:, :-1].contiguous()
        labels = ids[:, 1:].contiguous().clone()
        labels[mask[:, 1:] == 0] = -100
        loss = F.cross_entropy(
            shift.view(-1, shift.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()

        if step in EVAL_STEPS:
            arr = measure_nll_arr(model, tok, eval_texts)
            stats = bsci(arr)
            gap = gap_closed(recip_nll, donor_nll, stats["mean"])
            nll_curve[step] = {**stats, "gap_closed_pct": gap}
            print(f"  [{arm_name}] step={step:3d}  NLL={stats['mean']:.4f}  "
                  f"gap={gap:.2f}%  ({time.time()-t_arm:.0f}s)")
            model.train()

    return nll_curve, n_train, n_total


def main():
    t0 = time.time()
    print(f"genome_125: frozen-attn glue-train on {MODEL_ID}")

    train_texts = load_wikitext(5000, offset=N_EVAL)
    eval_texts = load_wikitext(N_EVAL, offset=0)
    print(f"  train={len(train_texts)}  eval={len(eval_texts)}")

    print("Loading donor...")
    donor, tok = load_trained()
    donor_arr = measure_nll_arr(donor, tok, eval_texts)
    donor_nll = bsci(donor_arr)
    print(f"  donor NLL={donor_nll['mean']:.4f}")

    print("Measuring random-init NLL...")
    rand = load_random_init(SEED)
    rand_arr = measure_nll_arr(rand, tok, eval_texts)
    rand_nll = bsci(rand_arr)
    del rand
    torch.cuda.empty_cache()
    print(f"  random-init NLL={rand_nll['mean']:.4f}")

    gap = rand_nll["mean"] - donor_nll["mean"]
    print(f"  gap={gap:.4f}  PASS_GAP_target_NLL={rand_nll['mean'] - 0.20*gap:.4f}")

    arm_results = {}

    # Arm 1: frozen_attn_glue
    print("\n--- Arm: frozen_attn_glue (donor all_attn copied + frozen, train glue) ---")
    m1 = load_random_init(SEED)
    n_copied = copy_donor_attn(donor, m1)
    print(f"  copied {n_copied:,} attn parameters from donor")
    curve1, n_train1, n_total1 = train_arm(
        m1, tok, train_texts, eval_texts,
        rand_nll["mean"], donor_nll["mean"],
        "frozen_attn_glue"
    )
    arm_results["frozen_attn_glue"] = {
        "trainable_params": n_train1, "total_params": n_total1,
        "donor_attn_copied": True, "nll_curve": curve1,
    }
    del m1
    torch.cuda.empty_cache()

    # Arm 2: matched_param_ctrl
    print("\n--- Arm: matched_param_ctrl (random-init, train glue only) ---")
    m2 = load_random_init(SEED)
    curve2, n_train2, n_total2 = train_arm(
        m2, tok, train_texts, eval_texts,
        rand_nll["mean"], donor_nll["mean"],
        "matched_param_ctrl"
    )
    arm_results["matched_param_ctrl"] = {
        "trainable_params": n_train2, "total_params": n_total2,
        "donor_attn_copied": False, "nll_curve": curve2,
    }
    del m2
    torch.cuda.empty_cache()

    # Arm 3: full_train_ctrl (sanity / baseline matching genome_123)
    print("\n--- Arm: full_train_ctrl (random-init, full unfreeze) ---")
    m3 = load_random_init(SEED)
    curve3, n_train3, n_total3 = train_arm(
        m3, tok, train_texts, eval_texts,
        rand_nll["mean"], donor_nll["mean"],
        "full_train_ctrl"
    )
    arm_results["full_train_ctrl"] = {
        "trainable_params": n_train3, "total_params": n_total3,
        "donor_attn_copied": False, "nll_curve": curve3,
    }
    del m3
    torch.cuda.empty_cache()

    # Verdict
    fa_step100 = arm_results["frozen_attn_glue"]["nll_curve"][100]
    mp_step100 = arm_results["matched_param_ctrl"]["nll_curve"][100]
    ft_step100 = arm_results["full_train_ctrl"]["nll_curve"][100]

    fa_gap = fa_step100["gap_closed_pct"]
    mp_gap = mp_step100["gap_closed_pct"]
    delta = fa_gap - mp_gap

    if fa_gap >= PASS_GAP and delta >= GLUE_DELTA_PASS:
        verdict = (f"PASS: frozen_attn_glue closes {fa_gap:.1f}% of gap at step 100, "
                   f"beats matched_param control by {delta:.1f} pp. "
                   f"Donor attention provides genuine transferable computation when interfaces are trained.")
    elif fa_gap >= PARTIAL_GAP and delta > 0:
        verdict = (f"PARTIAL: frozen_attn_glue closes {fa_gap:.1f}% (>= {PARTIAL_GAP}%) "
                   f"but only beats matched_param by {delta:.1f} pp. Weak signal.")
    elif fa_gap <= mp_gap:
        verdict = (f"KILL: frozen_attn_glue ({fa_gap:.1f}%) <= matched_param ({mp_gap:.1f}%). "
                   f"Surgery is dead. Capability is globally co-adapted, not recoverable by basis tricks. "
                   f"Honest conclusion per Codex: STOP weight-surgery experiments.")
    else:
        verdict = (f"KILL (weak): frozen_attn_glue ({fa_gap:.1f}%) > matched_param ({mp_gap:.1f}%) "
                   f"but below 10% partial threshold.")

    print("\n=== GENOME 125 SUMMARY ===")
    print(f"  donor NLL:     {donor_nll['mean']:.4f}")
    print(f"  random-init:   {rand_nll['mean']:.4f}")
    print(f"  frozen_attn_glue   step 100:  NLL={fa_step100['mean']:.4f}  gap={fa_gap:.2f}%")
    print(f"  matched_param_ctrl step 100:  NLL={mp_step100['mean']:.4f}  gap={mp_gap:.2f}%")
    print(f"  full_train_ctrl    step 100:  NLL={ft_step100['mean']:.4f}  gap={ft_step100['gap_closed_pct']:.2f}%")
    print(f"  delta (fa - mp):                          {delta:+.2f} pp")
    print(f"  verdict: {verdict}")

    out = {
        "model": MODEL_ID, "genome": 125, "name": "frozen_attn_glue_train",
        "donor_nll": donor_nll, "random_init_nll": rand_nll,
        "max_steps": MAX_STEPS, "lr": LR, "eval_steps": EVAL_STEPS,
        "criteria": {"pass_gap_pct": PASS_GAP, "partial_gap_pct": PARTIAL_GAP,
                     "glue_delta_pass_pp": GLUE_DELTA_PASS},
        "arm_results": {
            n: {**{k: v for k, v in r.items() if k != "nll_curve"},
                "nll_curve": {str(k): v for k, v in r["nll_curve"].items()}}
            for n, r in arm_results.items()
        },
        "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = RESULTS / "genome_125_frozen_attn_glue_train.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
