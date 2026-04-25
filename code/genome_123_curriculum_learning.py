"""
genome_123_curriculum_learning.py

GENOME-GUIDED CURRICULUM LEARNING.

Zero-step surgery is fully exhausted (genome_119-122). All weight-copy
strategies fail due to the holism barrier: scale mismatch + readout
alignment + distributed co-adaptation.

The pivot: use the DONOR'S geometry as a training signal during gradient-
based learning. If donor geometric invariants can guide the recipient to
converge faster to the universal attractor, we demonstrate that the genome
has practical value even when direct transplant fails.

Protocol: train random-init Qwen3-0.6B from scratch on wikitext with two arms:

  Arm A (baseline):
    CE loss on wikitext tokens (standard pretraining).

  Arm B (genome-guided):
    CE loss + layerwise activation matching loss:
    L = CE + gamma * sum_l MSE(recipient_hidden_l, donor_hidden_l)
    The donor activations are computed on the SAME batch with no_grad.
    gamma swept: [0.01, 0.1, 1.0] — three sub-arms.

Metric: NLL on held-out wikitext at steps [50, 100, 200, 500, 1000].
CtQ_75: gradient steps to reach NLL <= donor_NLL + 0.25 * initial_gap
        (i.e., close 75% of the initial random-init to donor gap).

Pass:    any genome-guided arm reaches CtQ_75 with >=2x speedup vs baseline
Partial: >=1.5x speedup at CtQ_75 or lower NLL at step 1000
Kill:    all genome-guided arms match or are worse than baseline at all checkpoints

Results: results/genome_123_curriculum_learning.json
"""

import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 42

N_EVAL = 200
SEQ_LEN = 64
BATCH_TRAIN = 4
BATCH_EVAL = 8
N_BOOT = 500
MAX_STEPS = 1000
EVAL_STEPS = [50, 100, 200, 500, 1000]
LR = 3e-4
GAMMAS = [0.01, 0.1, 1.0]


def load_wikitext_texts(n, offset=0, seed=SEED):
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


def tokenize(texts, tok, device=DEVICE):
    enc = tok(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=SEQ_LEN)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def load_trained(tok=None):
    if tok is None:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    return model, tok


def load_random_init(seed=SEED):
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).to(DEVICE)


def ce_loss(model, ids, mask):
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits
    shift = logits[:, :-1].contiguous()
    labels = ids[:, 1:].contiguous().clone()
    labels[mask[:, 1:] == 0] = -100
    return F.cross_entropy(
        shift.view(-1, shift.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


def get_hidden_states(model, ids, mask):
    """Return list of per-layer hidden states (no grad)."""
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask,
                    output_hidden_states=True)
    return out.hidden_states  # tuple: (embed, layer0, ..., layerN)


def fm_loss(student_hiddens, teacher_hiddens, mask):
    """Layerwise MSE matching loss, masked to non-padding tokens."""
    total = torch.tensor(0.0, device=DEVICE)
    m = mask.unsqueeze(-1).float()  # (B, T, 1)
    n_layers = min(len(student_hiddens), len(teacher_hiddens))
    for l in range(n_layers):
        s = student_hiddens[l].float() * m
        t = teacher_hiddens[l].float().detach() * m
        total = total + F.mse_loss(s, t, reduction="mean")
    return total / max(n_layers, 1)


def measure_nll_arr(model, tok, texts):
    model.eval()
    per_seq = []
    for i in range(0, len(texts), BATCH_EVAL):
        ids, mask = tokenize(texts[i:i + BATCH_EVAL], tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits
        for j in range(ids.shape[0]):
            lj = logits[j:j+1, :-1]
            lbl = ids[j:j+1, 1:].clone()
            mk = mask[j:j+1, 1:]
            lbl[mk == 0] = -100
            per_seq.append(
                F.cross_entropy(lj.view(-1, lj.size(-1)),
                                lbl.view(-1), ignore_index=-100).item()
            )
    model.train()
    return np.array(per_seq)


def bsci(arr, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(arr.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


def train_arm(model, donor, tok, train_texts, eval_texts,
              gamma=0.0, arm_name="baseline"):
    """Train for MAX_STEPS, evaluate at EVAL_STEPS. Returns nll_curve."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    rng = np.random.default_rng(SEED)
    nll_curve = {}
    step = 0
    t_arm = time.time()

    while step < MAX_STEPS:
        # sample a batch
        idx = rng.integers(0, len(train_texts), size=BATCH_TRAIN)
        batch = [train_texts[i] for i in idx]
        ids, mask = tokenize(batch, tok)

        opt.zero_grad()
        loss = ce_loss(model, ids, mask)

        if gamma > 0:
            student_h = model(input_ids=ids, attention_mask=mask,
                              output_hidden_states=True).hidden_states
            teacher_h = get_hidden_states(donor, ids, mask)
            loss = loss + gamma * fm_loss(student_h, teacher_h, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step in EVAL_STEPS:
            arr = measure_nll_arr(model, tok, eval_texts)
            stats = bsci(arr)
            nll_curve[step] = stats
            print(f"  [{arm_name}] step={step:5d}  NLL={stats['mean']:.4f}  "
                  f"({time.time()-t_arm:.0f}s)")
            model.train()

    return nll_curve


def find_ctq(nll_curve, target_nll):
    """Return first step where NLL <= target, or None."""
    for step in sorted(nll_curve.keys()):
        if nll_curve[step]["mean"] <= target_nll:
            return step
    return None


def main():
    t0 = time.time()
    print(f"genome_123: genome-guided curriculum on {MODEL_ID}")

    train_texts = load_wikitext_texts(5000, offset=N_EVAL)
    eval_texts = load_wikitext_texts(N_EVAL, offset=0)
    print(f"  train={len(train_texts)}  eval={len(eval_texts)}")

    print("Loading donor...")
    donor, tok = load_trained()
    donor_arr = measure_nll_arr(donor, tok, eval_texts)
    donor_nll = bsci(donor_arr)
    donor.eval()
    print(f"  donor NLL={donor_nll['mean']:.4f}")

    print("Measuring random-init NLL...")
    rand_init = load_random_init(SEED)
    rand_arr = measure_nll_arr(rand_init, tok, eval_texts)
    rand_nll = bsci(rand_arr)
    del rand_init
    torch.cuda.empty_cache()
    print(f"  random-init NLL={rand_nll['mean']:.4f}")

    gap = rand_nll["mean"] - donor_nll["mean"]
    target_75 = donor_nll["mean"] + 0.25 * gap
    print(f"  gap={gap:.4f}  CtQ_75 target NLL={target_75:.4f}")

    arm_results = {}

    # Arm A: baseline CE only
    print("\n--- Arm A: baseline (CE only) ---")
    model_a = load_random_init(SEED)
    curve_a = train_arm(model_a, donor, tok, train_texts, eval_texts,
                        gamma=0.0, arm_name="baseline")
    ctq_a = find_ctq(curve_a, target_75)
    arm_results["baseline"] = {"gamma": 0.0, "nll_curve": curve_a,
                                "ctq_75_step": ctq_a}
    del model_a
    torch.cuda.empty_cache()

    # Arms B1, B2, B3: genome-guided with gamma sweep
    for gamma in GAMMAS:
        arm_name = f"genome_guided_g{gamma:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        print(f"\n--- Arm {arm_name} (CE + FM loss gamma={gamma}) ---")
        model_b = load_random_init(SEED)
        curve_b = train_arm(model_b, donor, tok, train_texts, eval_texts,
                            gamma=gamma, arm_name=arm_name)
        ctq_b = find_ctq(curve_b, target_75)
        arm_results[arm_name] = {"gamma": gamma, "nll_curve": curve_b,
                                  "ctq_75_step": ctq_b}
        del model_b
        torch.cuda.empty_cache()

    # Verdict
    baseline_ctq = arm_results["baseline"]["ctq_75_step"]
    best_guided_name = min(
        (k for k in arm_results if k != "baseline"),
        key=lambda k: (arm_results[k]["ctq_75_step"] or MAX_STEPS + 1)
    )
    best_guided_ctq = arm_results[best_guided_name]["ctq_75_step"]

    # Also check NLL at step 1000
    baseline_nll_1000 = arm_results["baseline"]["nll_curve"].get(1000, {}).get("mean", float("inf"))
    best_nll_1000 = min(
        arm_results[k]["nll_curve"].get(1000, {}).get("mean", float("inf"))
        for k in arm_results if k != "baseline"
    )

    if baseline_ctq and best_guided_ctq and best_guided_ctq <= baseline_ctq / 2:
        verdict = (f"PASS: {best_guided_name} reaches CtQ_75 at step {best_guided_ctq} "
                   f"vs baseline step {baseline_ctq} "
                   f"({baseline_ctq/best_guided_ctq:.1f}x speedup). "
                   "Genome-guided curriculum accelerates capability acquisition.")
    elif (baseline_ctq and best_guided_ctq and best_guided_ctq <= baseline_ctq / 1.5) \
            or (best_nll_1000 < baseline_nll_1000 - 0.1):
        speedup = (baseline_ctq / best_guided_ctq) if (baseline_ctq and best_guided_ctq) else 1.0
        verdict = (f"PARTIAL: {best_guided_name} shows {speedup:.1f}x CtQ_75 speedup "
                   f"or NLL improvement at step 1000 "
                   f"(guided {best_nll_1000:.4f} vs baseline {baseline_nll_1000:.4f}). "
                   "Weak positive signal from donor geometry.")
    else:
        verdict = (f"KILL: genome-guided activation matching does not accelerate training. "
                   f"Best guided arm: {best_guided_name}, CtQ_75={best_guided_ctq}. "
                   f"Baseline CtQ_75={baseline_ctq}. "
                   f"NLL at step 1000: guided={best_nll_1000:.4f} baseline={baseline_nll_1000:.4f}.")

    print("\n=== GENOME 123 SUMMARY ===")
    print(f"  donor NLL:     {donor_nll['mean']:.4f}")
    print(f"  random-init:   {rand_nll['mean']:.4f}  gap={gap:.4f}")
    print(f"  CtQ_75 target: {target_75:.4f}")
    for name, res in arm_results.items():
        nll_1000 = res["nll_curve"].get(1000, {}).get("mean", float("nan"))
        print(f"  {name:35s}: CtQ_75={res['ctq_75_step']}  NLL@1000={nll_1000:.4f}")
    print(f"  verdict: {verdict}")

    out = {
        "model": MODEL_ID,
        "genome": 123,
        "name": "curriculum_learning",
        "n_eval": N_EVAL,
        "n_train": len(train_texts),
        "max_steps": MAX_STEPS,
        "lr": LR,
        "eval_steps": EVAL_STEPS,
        "gammas": GAMMAS,
        "donor_nll": donor_nll,
        "random_init_nll": rand_nll,
        "initial_gap_nats": float(gap),
        "ctq_75_target_nll": float(target_75),
        "arm_results": {
            name: {
                "gamma": res["gamma"],
                "ctq_75_step": res["ctq_75_step"],
                "nll_curve": {str(k): v for k, v in res["nll_curve"].items()},
            }
            for name, res in arm_results.items()
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }

    out_path = RESULTS / "genome_123_curriculum_learning.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
