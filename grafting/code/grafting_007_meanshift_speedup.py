"""
grafting_007_meanshift_speedup.py

Mean-shift speedup test: does adding donor-derived per-layer mean-shift bias
provide CtQ_75 CE training speedup vs lesion baseline?

Background
----------
capability_patch_generalize.json showed that adding a FIXED per-layer bias
bias[l] = mean(h_donor[l]) - mean(h_lesion[l])
achieves 61% gap closure with ZERO fitting and ZERO trainable parameters.

This experiment tests whether that free 61% head-start translates into
CtQ_75 CE training speedup when we then unfreeze the full backbone.

Protocol
--------
Compile step:
  Load donor + lesion simultaneously. Stream n_compile texts.
  Accumulate token-level sum and count per layer.
  Compute bias[l] = (sum_donor[l] / n_tokens) - (sum_lesion[l] / n_tokens).

Arm A (lesion baseline):
  Fresh model load, all down_proj zeroed. No hooks. Full backbone unfrozen.
  N_STEPS CE training steps.

Arm B (mean-shift):
  Fresh model load, all down_proj zeroed.
  Fixed (non-trainable) forward hook at each decoder layer: out += bias[l].
  Full backbone unfrozen.
  N_STEPS CE training steps.

Primary metric: CtQ_75 speedup = steps_A / steps_B.
Project gate (grafting/OBJECTIVE.md): >= 10x.
"""

import json
import pathlib
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

ROOT    = pathlib.Path(__file__).parent.parent.parent
RESULTS = pathlib.Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED     = 42

N_COMPILE = 512
SEQ_LEN   = 128
COMPILE_BATCH = 8

N_EVAL    = 300
N_TRAIN   = 2000
N_STEPS   = 300
LR        = 1e-4
LOG_EVERY = 25
TRAIN_BATCH = 8
EVAL_BATCH  = 8


# ── data ──────────────────────────────────────────────────────────────────────

def load_all_texts(n_total=3200, seed=SEED):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds))
    texts = []
    for i in indices:
        t = ds[int(i)]["text"].strip()
        if len(t) >= 80:
            texts.append(t[:512])
        if len(texts) >= n_total:
            break
    return texts


# ── metrics ───────────────────────────────────────────────────────────────────

def measure_nll(model, tokenizer, texts, max_len=SEQ_LEN, batch=EVAL_BATCH):
    model.eval()
    total_nll, total_toks = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(DEVICE)
            out    = model(**enc)
            logits = out.logits[:, :-1].float()
            labels = enc["input_ids"][:, 1:].clone()
            labels[enc["attention_mask"][:, 1:] == 0] = -100
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1), ignore_index=-100, reduction="sum"
            )
            total_nll  += loss.item()
            total_toks += (labels != -100).sum().item()
    return total_nll / max(total_toks, 1)


def steps_to_nll(trajectory, target):
    for step in sorted(trajectory):
        if trajectory[step] <= target:
            return step
    return None


# ── compile: per-layer mean-shift bias ────────────────────────────────────────

def compile_meanshift(donor, lesion, tokenizer, compile_texts, n_layers, d_model, t0):
    sum_donor  = [np.zeros(d_model, dtype=np.float64) for _ in range(n_layers + 1)]
    sum_lesion = [np.zeros(d_model, dtype=np.float64) for _ in range(n_layers + 1)]
    n_tokens   = 0

    donor.eval(); lesion.eval()

    with torch.no_grad():
        for i in range(0, len(compile_texts), COMPILE_BATCH):
            chunk = compile_texts[i:i + COMPILE_BATCH]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=SEQ_LEN).to(DEVICE)
            mask = enc["attention_mask"].bool()  # (B, T)

            out_d = donor(**enc, output_hidden_states=True)
            out_l = lesion(**enc, output_hidden_states=True)

            # hidden_states[0] = embedding, [1..n_layers] = decoder outputs
            for l_idx in range(n_layers + 1):
                hd = out_d.hidden_states[l_idx].float()  # (B, T, d)
                hl = out_l.hidden_states[l_idx].float()
                # Mean over valid tokens
                for b in range(hd.shape[0]):
                    valid = mask[b]              # (T,) bool
                    sum_donor[l_idx]  += hd[b][valid].cpu().numpy().sum(axis=0)
                    sum_lesion[l_idx] += hl[b][valid].cpu().numpy().sum(axis=0)
                    if l_idx == 0:
                        n_tokens += valid.sum().item()

    # bias[l] = mean_donor[l] - mean_lesion[l], computed for decoder layers 1..n_layers
    biases = []
    for l in range(1, n_layers + 1):
        b = (sum_donor[l] - sum_lesion[l]) / max(n_tokens, 1)
        biases.append(b.astype(np.float32))
        if l % 7 == 1 or l == n_layers:
            norm = float(np.linalg.norm(b))
            print(f"  L{l-1:02d}: bias norm = {norm:.4f}")

    print(f"  [{time.time()-t0:.1f}s] Mean-shift compiled from {n_tokens} tokens")
    return biases   # list of length n_layers, each (d_model,)


# ── hooks ─────────────────────────────────────────────────────────────────────

def attach_meanshift_hooks(model, biases, n_layers):
    hooks = []

    def make_hook(l, bias_vec):
        bias_t = torch.tensor(bias_vec, dtype=torch.float32, device=DEVICE)

        def forward_hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
                orig_dtype = h.dtype
                new_h = (h.float() + bias_t).to(orig_dtype)
                return (new_h,) + out[1:]
            else:
                orig_dtype = out.dtype
                return (out.float() + bias_t).to(orig_dtype)

        return forward_hook

    for l in range(n_layers):
        h = model.model.layers[l].register_forward_hook(make_hook(l, biases[l]))
        hooks.append(h)
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── training arm ──────────────────────────────────────────────────────────────

def run_arm(model, tokenizer, train_texts, eval_texts, n_steps, t0, arm_name,
            biases=None):
    # Full backbone unfreeze
    for p in model.parameters():
        p.requires_grad_(True)

    hooks = []
    if biases is not None:
        hooks = attach_meanshift_hooks(model, biases, len(biases))

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    trajectory = {}
    text_idx = 0
    n_texts  = len(train_texts)

    for step in range(n_steps + 1):
        if step % LOG_EVERY == 0:
            nll = measure_nll(model, tokenizer, eval_texts)
            trajectory[step] = float(nll)
            print(f"  [{arm_name}] step {step:4d} | NLL {nll:.4f} | {time.time()-t0:.0f}s")
            model.train()

        if step == n_steps:
            break

        chunk = []
        for _ in range(TRAIN_BATCH):
            chunk.append(train_texts[text_idx % n_texts])
            text_idx += 1

        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=SEQ_LEN).to(DEVICE)
        model.train()
        out    = model(**enc)
        logits = out.logits[:, :-1].float()
        labels = enc["input_ids"][:, 1:].clone()
        labels[enc["attention_mask"][:, 1:] == 0] = -100
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1), ignore_index=-100
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    remove_hooks(hooks)
    return trajectory


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print(f"[{t0-t0:.1f}s] Loading texts (n_total=3200)...")
    all_texts     = load_all_texts(n_total=3200)
    compile_texts = all_texts[:N_COMPILE]
    eval_texts    = all_texts[N_COMPILE:N_COMPILE + N_EVAL]
    train_texts   = all_texts[N_COMPILE + N_EVAL:N_COMPILE + N_EVAL + N_TRAIN]
    print(f"  compile={len(compile_texts)} eval={len(eval_texts)} train={len(train_texts)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- COMPILE: compute per-layer mean-shift bias ----
    print(f"\n[{time.time()-t0:.1f}s] Loading donor + lesion for compile step...")
    donor = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    n_layers = donor.config.num_hidden_layers
    d_model  = donor.config.hidden_size
    print(f"  n_layers={n_layers}, d_model={d_model}")

    nll_donor = measure_nll(donor, tok, eval_texts)
    print(f"  Donor NLL: {nll_donor:.4f}")

    lesion_c = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    with torch.no_grad():
        for l in range(n_layers):
            lesion_c.model.layers[l].mlp.down_proj.weight.zero_()
    nll_lesion = measure_nll(lesion_c, tok, eval_texts)
    print(f"  Lesion NLL: {nll_lesion:.4f}")

    print(f"\n[{time.time()-t0:.1f}s] Compiling mean-shift biases...")
    biases = compile_meanshift(donor, lesion_c, tok, compile_texts, n_layers, d_model, t0)

    del donor, lesion_c
    torch.cuda.empty_cache()
    print(f"  [{time.time()-t0:.1f}s] Compile done. GPU cleared.")

    gap       = nll_lesion - nll_donor
    target_50 = nll_donor + 0.50 * gap
    target_75 = nll_donor + 0.25 * gap
    target_90 = nll_donor + 0.10 * gap
    print(f"\n  Gap: {gap:.4f} nats | target_75={target_75:.4f}")

    # ================================================================
    # ARM A: lesion baseline, no mean-shift
    # ================================================================
    print(f"\n{'='*60}")
    print(f"ARM A: lesion baseline (no mean-shift)")
    print(f"{'='*60}")
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    with torch.no_grad():
        for l in range(n_layers):
            model_a.model.layers[l].mlp.down_proj.weight.zero_()
    traj_a = run_arm(model_a, tok, train_texts, eval_texts, N_STEPS, t0, "arm_a",
                     biases=None)
    del model_a
    torch.cuda.empty_cache()

    # ================================================================
    # ARM B: lesion + fixed mean-shift hooks
    # ================================================================
    print(f"\n{'='*60}")
    print(f"ARM B: lesion + fixed mean-shift bias per layer")
    print(f"{'='*60}")
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    with torch.no_grad():
        for l in range(n_layers):
            model_b.model.layers[l].mlp.down_proj.weight.zero_()

    # Verify: measure NLL before training with bias applied
    hooks_verify = attach_meanshift_hooks(model_b, biases, n_layers)
    nll_b_step0 = measure_nll(model_b, tok, eval_texts)
    remove_hooks(hooks_verify)
    print(f"  Arm B NLL at step 0 (with bias): {nll_b_step0:.4f}")
    fg0 = (nll_lesion - nll_b_step0) / gap
    print(f"  Fraction gap closed at step 0: {fg0:.3f}")

    traj_b = run_arm(model_b, tok, train_texts, eval_texts, N_STEPS, t0, "arm_b",
                     biases=biases)
    del model_b
    torch.cuda.empty_cache()

    # ================================================================
    # Speedup analysis
    # ================================================================
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"  Donor NLL:     {nll_donor:.4f}")
    print(f"  Lesion NLL:    {nll_lesion:.4f}")
    print(f"  Gap:           {gap:.4f} nats")
    print(f"  Arm B NLL@0:   {nll_b_step0:.4f}  (fg_closed={fg0:.3f})")
    print(f"{'='*60}")
    print(f"  {'Step':>5}  {'Arm A (base)':>14}  {'Arm B (shift)':>14}  {'Advantage':>10}")
    for step in sorted(traj_a):
        na = traj_a[step]
        nb = traj_b.get(step, float('nan'))
        print(f"  {step:5d}  {na:14.4f}  {nb:14.4f}  {na - nb:+10.4f}")
    print(f"{'='*60}")

    metrics = {}
    for name, target in [("CtQ_50", target_50), ("CtQ_75", target_75), ("CtQ_90", target_90)]:
        sa = steps_to_nll(traj_a, target)
        sb = steps_to_nll(traj_b, target)
        if sb == 0:
            sp = "inf"
        elif sa is None or sb is None:
            sp = None
        else:
            sp = sa / sb
        metrics[name] = {"target_nll": float(target), "steps_a": sa, "steps_b": sb,
                         "speedup": sp}
        label = f"{sp:.2f}x" if isinstance(sp, float) else str(sp)
        print(f"  {name} (NLL={target:.2f}): arm_a={sa} arm_b={sb} speedup={label}")

    sp75 = metrics.get("CtQ_75", {}).get("speedup")
    if sp75 == "inf":
        verdict = f"PASS: Arm B already at target_75 at step 0 — mean-shift init is sufficient"
    elif isinstance(sp75, float) and sp75 >= 10.0:
        verdict = f"PASS: CtQ_75 speedup {sp75:.1f}x >= 10x — mean-shift provides significant training acceleration"
    elif isinstance(sp75, float) and sp75 >= 2.0:
        verdict = f"PARTIAL: CtQ_75 speedup {sp75:.1f}x (2x-10x) — some acceleration, below project gate"
    elif sp75 is None:
        verdict = f"KILL: neither arm reaches target_75 within {N_STEPS} steps"
    else:
        verdict = f"KILL: CtQ_75 speedup {sp75:.2f}x < 2x — mean-shift bias does not accelerate training"

    print(f"\n  VERDICT: {verdict}")

    elapsed = float(time.time() - t0)
    out = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_compile": N_COMPILE,
        "n_steps": N_STEPS,
        "lr": LR,
        "nll_donor": float(nll_donor),
        "nll_lesion_base": float(nll_lesion),
        "nll_arm_b_step0": float(nll_b_step0),
        "fraction_gap_closed_step0": float(fg0),
        "gap_nats": float(gap),
        "targets": {"50pct": float(target_50), "75pct": float(target_75), "90pct": float(target_90)},
        "trajectory_arm_a": {str(k): v for k, v in traj_a.items()},
        "trajectory_arm_b": {str(k): v for k, v in traj_b.items()},
        "speedup_metrics": {k: {**v, "speedup": str(v["speedup"]) if v["speedup"] is not None else None}
                            for k, v in metrics.items()},
        "verdict": verdict,
        "elapsed_s": elapsed,
    }

    out_path = RESULTS / "grafting_007_meanshift_speedup.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}  (elapsed {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
