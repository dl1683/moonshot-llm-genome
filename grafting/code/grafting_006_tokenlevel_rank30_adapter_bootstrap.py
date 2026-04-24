"""
grafting_006_tokenlevel_rank30_adapter_bootstrap.py

Corrected speedup test: token-level rank-30 adapter bootstrap vs zero-init adapters.

Fixes all grafting_005 methodological flaws identified by Codex:
  1. Fresh model load for EACH arm (no contamination from shared training)
  2. Frozen backbone: only adapter params {A_l, B_l, g_l} are trained
  3. Token-level (not sentence-pooled) statistics for adapter fitting
  4. Arm A = zero-init adapters (fair baseline, same architecture as Arm B)
  5. Disjoint compile/eval/train splits

Protocol
--------
Compile step (~10 min):
  Load donor (bfloat16) + lesion (bfloat16) simultaneously on GPU.
  For n_compile texts, collect token-level (h_l^lesion, h_{l+1}^donor - h_{l+1}^lesion) pairs.
  Accumulate XtX[l] and XtY[l] per layer (streaming sufficient stats, avoids OOM).
  Solve ridge per layer: W_l = (XtX[l] + lambda*I)^{-1} XtY[l]
  Top-30 SVD: A_l = U[:,:30] * sqrt(S[:30]), B_l = V[:,:30] * sqrt(S[:30])
  Both donor + lesion then unloaded.

Arm A (zero-init adapters):
  Fresh model load, zero down_proj, zero adapters. Freeze backbone.
  500 CE steps training adapters only.

Arm B (token-fitted adapters):
  Fresh model load, zero down_proj, adapters initialized from SVD solution. Freeze backbone.
  500 CE steps training adapters only.

Primary metric: CtQ_75 speedup = steps_A / steps_B to reach 75% gap closure
Project gate (grafting/OBJECTIVE.md): >= 10x speedup
"""

import json
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn
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

RANK      = 30
N_COMPILE = 512      # texts for token-level stats (token-level >> sentence-pooled)
RIDGE_LAM = 1.0      # regularization for token-level solve (1024x1024 system)
SEQ_LEN   = 128
COMPILE_BATCH = 8

N_EVAL    = 300
N_TRAIN   = 2000
N_STEPS   = 500
LR        = 1e-3     # higher LR: only adapter params (~1.7M) training
LOG_EVERY = 50
TRAIN_BATCH = 8
EVAL_BATCH  = 8


# ── data ──────────────────────────────────────────────────────────────────────

def load_all_texts(n_total=4000, seed=SEED):
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


# ── compile: token-level adapter fitting ──────────────────────────────────────

def compile_adapters(donor, lesion, tokenizer, compile_texts, n_layers, d_model,
                     ridge_lam, t0):
    XtX = [np.zeros((d_model, d_model), dtype=np.float64) for _ in range(n_layers)]
    XtY = [np.zeros((d_model, d_model), dtype=np.float64) for _ in range(n_layers)]

    donor.eval(); lesion.eval()
    n_tokens_total = 0

    with torch.no_grad():
        for i in range(0, len(compile_texts), COMPILE_BATCH):
            chunk = compile_texts[i:i + COMPILE_BATCH]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=SEQ_LEN).to(DEVICE)
            mask = enc["attention_mask"].bool().cpu().numpy()  # (B, T)

            out_d = donor(**enc, output_hidden_states=True)
            out_l = lesion(**enc, output_hidden_states=True)

            h_donor  = [h.float().cpu().numpy() for h in out_d.hidden_states]  # list[(B,T,d)]
            h_lesion = [h.float().cpu().numpy() for h in out_l.hidden_states]

            for l in range(n_layers):
                X_bt = h_lesion[l]      # (B, T, d) — adapter input
                Y_bt = h_donor[l + 1] - h_lesion[l + 1]   # (B, T, d) — residual target

                for b in range(len(chunk)):
                    valid = mask[b]             # (T,) bool
                    X = X_bt[b][valid]          # (n_valid, d)
                    Y = Y_bt[b][valid]          # (n_valid, d)
                    n_tokens_total += len(X)
                    XtX[l] += X.T @ X
                    XtY[l] += X.T @ Y

    print(f"  [{time.time()-t0:.1f}s] Token-level stats: {n_tokens_total} tokens across "
          f"{len(compile_texts)} texts, {n_layers} layers")

    # Per-layer ridge solve + rank-30 SVD
    A_fits = []  # list[(d, RANK)]
    B_fits = []
    residuals = []
    for l in range(n_layers):
        A_reg = XtX[l] + ridge_lam * np.eye(d_model)
        W = np.linalg.solve(A_reg, XtY[l])  # (d, d) full-rank solution
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        sqrt_S = np.sqrt(np.maximum(S[:RANK], 0))
        A_l = U[:, :RANK] * sqrt_S           # (d, RANK)
        B_l = Vt[:RANK].T * sqrt_S           # (d, RANK)
        A_fits.append(A_l)
        B_fits.append(B_l)
        if l % 7 == 0:
            residuals.append(float(S[0] / (S.sum() + 1e-10)))
            print(f"  L{l:02d}: top eigenvalue fraction = {residuals[-1]:.4f}")

    return A_fits, B_fits


# ── adapter module ─────────────────────────────────────────────────────────────

class RankAdapters(nn.Module):
    def __init__(self, d_model, rank, n_layers, init_A=None, init_B=None):
        super().__init__()
        self.rank = rank
        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        self.g = nn.ParameterList()
        for l in range(n_layers):
            if init_A is not None:
                self.A.append(nn.Parameter(
                    torch.tensor(init_A[l], dtype=torch.float32).to(DEVICE)))
                self.B.append(nn.Parameter(
                    torch.tensor(init_B[l], dtype=torch.float32).to(DEVICE)))
            else:
                self.A.append(nn.Parameter(torch.zeros(d_model, rank, device=DEVICE)))
                self.B.append(nn.Parameter(torch.zeros(d_model, rank, device=DEVICE)))
            self.g.append(nn.Parameter(torch.zeros(1, device=DEVICE)))


def attach_adapters(model, adapters, n_layers):
    hooks = []

    def make_hook(l):
        def forward_hook(module, inp, out):
            h_l = inp[0].float()                        # (B, T, d)
            h_l1 = out[0].float()                       # (B, T, d)
            A = adapters.A[l].to(h_l.device)
            B = adapters.B[l].to(h_l.device)
            g = adapters.g[l].to(h_l.device)
            delta = (h_l @ B) @ A.T * g                # (B, T, d)
            new_h = (h_l1 + delta).to(out[0].dtype)
            return (new_h,) + out[1:]
        return forward_hook

    for l in range(n_layers):
        h = model.model.layers[l].register_forward_hook(make_hook(l))
        hooks.append(h)
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── training arm ──────────────────────────────────────────────────────────────

def run_arm(model, adapters, tokenizer, train_texts, eval_texts, n_steps, t0, arm_name):
    # Freeze ALL model params
    for p in model.parameters():
        p.requires_grad_(False)
    # Only train adapter params
    for p in adapters.parameters():
        p.requires_grad_(True)

    hooks = attach_adapters(model, adapters, len(adapters.A))
    optimizer = AdamW(adapters.parameters(), lr=LR, weight_decay=0.01)

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
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), 1.0)
        optimizer.step()

    remove_hooks(hooks)
    return trajectory


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print(f"[{t0-t0:.1f}s] Loading texts (n_total=4000)...")
    all_texts    = load_all_texts(n_total=4000)
    compile_texts = all_texts[:N_COMPILE]
    eval_texts    = all_texts[N_COMPILE:N_COMPILE + N_EVAL]
    train_texts   = all_texts[N_COMPILE + N_EVAL:N_COMPILE + N_EVAL + N_TRAIN]
    print(f"  compile={len(compile_texts)} eval={len(eval_texts)} train={len(train_texts)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- COMPILE: token-level adapter fitting ----
    print(f"\n[{time.time()-t0:.1f}s] Loading donor + lesion for compile step...")
    donor = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    n_layers = donor.config.num_hidden_layers
    d_model  = donor.config.hidden_size
    print(f"  n_layers={n_layers}, d_model={d_model}")

    # Measure donor NLL (reference)
    nll_donor = measure_nll(donor, tok, eval_texts)
    print(f"  Donor NLL: {nll_donor:.4f}")

    # Create lesion: zero all down_proj weights
    lesion = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    with torch.no_grad():
        for l in range(n_layers):
            lesion.model.layers[l].mlp.down_proj.weight.zero_()
    nll_lesion = measure_nll(lesion, tok, eval_texts)
    print(f"  Lesion NLL: {nll_lesion:.4f}")

    print(f"\n[{time.time()-t0:.1f}s] Compiling token-level rank-{RANK} adapters...")
    A_fits, B_fits = compile_adapters(
        donor, lesion, tok, compile_texts, n_layers, d_model, RIDGE_LAM, t0)

    # Free both compile models
    del donor, lesion
    torch.cuda.empty_cache()
    print(f"  [{time.time()-t0:.1f}s] Compile done. GPU cleared.")

    gap      = nll_lesion - nll_donor
    target_50 = nll_donor + 0.50 * gap
    target_75 = nll_donor + 0.25 * gap
    target_90 = nll_donor + 0.10 * gap
    print(f"\n  Gap: {gap:.4f} nats | target_75={target_75:.4f}")

    # ================================================================
    # ARM A: zero-init adapters (fair baseline)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"ARM A: zero-init rank-{RANK} adapters")
    print(f"{'='*60}")
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    with torch.no_grad():
        for l in range(n_layers):
            model_a.model.layers[l].mlp.down_proj.weight.zero_()

    adapters_a = RankAdapters(d_model, RANK, n_layers, init_A=None, init_B=None)
    traj_a = run_arm(model_a, adapters_a, tok, train_texts, eval_texts, N_STEPS, t0, "arm_a")
    del model_a, adapters_a
    torch.cuda.empty_cache()

    # ================================================================
    # ARM B: token-fitted adapters (the compile-step solution)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"ARM B: token-fitted rank-{RANK} adapters")
    print(f"{'='*60}")
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    with torch.no_grad():
        for l in range(n_layers):
            model_b.model.layers[l].mlp.down_proj.weight.zero_()

    adapters_b = RankAdapters(d_model, RANK, n_layers, init_A=A_fits, init_B=B_fits)
    traj_b = run_arm(model_b, adapters_b, tok, train_texts, eval_texts, N_STEPS, t0, "arm_b")
    del model_b, adapters_b
    torch.cuda.empty_cache()

    # ================================================================
    # Speedup analysis
    # ================================================================
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"  Donor NLL:  {nll_donor:.4f}")
    print(f"  Lesion NLL: {nll_lesion:.4f}")
    print(f"  Gap:        {gap:.4f} nats")
    print(f"{'='*60}")
    print(f"  {'Step':>5}  {'Arm A (zero)':>14}  {'Arm B (fitted)':>14}  {'Advantage':>10}")
    for step in sorted(traj_a):
        na = traj_a[step]
        nb = traj_b.get(step, float('nan'))
        print(f"  {step:5d}  {na:14.4f}  {nb:14.4f}  {na - nb:+10.4f}")
    print(f"{'='*60}")

    metrics = {}
    for name, target in [("CtQ_50", target_50), ("CtQ_75", target_75), ("CtQ_90", target_90)]:
        sa = steps_to_nll(traj_a, target)
        sb = steps_to_nll(traj_b, target)
        if sa is None and sb is None:
            sp = None
        elif sb == 0:
            sp = "inf"
        elif sa is None:
            sp = None
        else:
            sp = sa / sb if sb else "inf"
        metrics[name] = {"target_nll": float(target), "steps_a": sa, "steps_b": sb,
                         "speedup": sp}
        label = f"{sp:.2f}x" if isinstance(sp, float) else str(sp)
        print(f"  {name} (NLL={target:.2f}): arm_a={sa} arm_b={sb} speedup={label}")

    sp75 = metrics.get("CtQ_75", {}).get("speedup")
    if isinstance(sp75, float) and sp75 >= 10.0:
        verdict = f"PASS: CtQ_75 speedup {sp75:.1f}x >= 10x — token-level adapters provide significant training acceleration"
    elif isinstance(sp75, float) and sp75 >= 2.0:
        verdict = f"PARTIAL: CtQ_75 speedup {sp75:.1f}x (2x-10x) — some acceleration, below project gate"
    elif sp75 == "inf":
        verdict = "PASS: arm_b already at target_75 at step 0"
    elif sp75 is None:
        verdict = f"KILL: neither arm reaches target_75 within {N_STEPS} steps"
    else:
        verdict = f"KILL: CtQ_75 speedup {sp75:.2f}x < 2x — no meaningful acceleration"

    print(f"\n  VERDICT: {verdict}")

    out = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "d_model": d_model,
        "rank": RANK,
        "n_compile": N_COMPILE,
        "ridge_lambda": RIDGE_LAM,
        "n_steps": N_STEPS,
        "lr": LR,
        "nll_donor": float(nll_donor),
        "nll_lesion_base": float(nll_lesion),
        "gap_nats": float(gap),
        "targets": {"50pct": float(target_50), "75pct": float(target_75), "90pct": float(target_90)},
        "trajectory_arm_a": {str(k): v for k, v in traj_a.items()},
        "trajectory_arm_b": {str(k): v for k, v in traj_b.items()},
        "speedup_metrics": {k: {**v, "speedup": str(v["speedup"]) if v["speedup"] is not None else None}
                            for k, v in metrics.items()},
        "verdict": verdict,
        "elapsed_s": float(time.time() - t0),
    }

    out_path = RESULTS / "grafting_006_tokenlevel_rank30_adapter_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
