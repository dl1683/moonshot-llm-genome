"""
grafting_008_trainable_meanshift_persistence.py

Trainable mean-shift persistence test: does donor-initialized trainable bias
survive CE training when given a carrier that can adapt?

Background
----------
grafting_007 showed: fixed mean-shift gives 58% zero-step gap closure (CtQ_50=inf)
but becomes a liability after step 25 — backbone compensates for fixed offset, making
arm_b WORSE than arm_a from step 50 onward. Bottleneck is PERSISTENCE, not step-0 effect.

grafting_006 showed: good basis (token-fitted rank-30 adapters) with frozen backbone
gives CtQ_75 speedup=1.0x. Closed-loop mismatch + adapter-only training fails.

The untested quadrant: "good function + trainable carrier."
This experiment tests whether donor-initialized biases with an adaptive carrier
and an anchor penalty can retain the step-0 geometry advantage through CE.

Protocol
--------
Compile step: compute per-layer mean-shift bias (same as grafting_007).

Arm A (zero-init trainable bias):
  Trainable bias per layer (b[l] ∈ R^d_model, init=0). Protected warmup:
  steps 0-10: bias params only; steps 11+: full backbone unfreeze.

Arm B (donor-init fixed bias):
  Same as grafting_007 arm_b: fixed non-trainable bias from donor means.
  Full backbone unfreeze from step 0. Reference for comparison.

Arm C (donor-init trainable bias + anchor):
  Trainable bias per layer, init from donor_mean - lesion_mean.
  Protected warmup: steps 0-10: bias params only; steps 11+: full backbone.
  Anchor penalty: lambda(t) * ||b[l] - b0[l]||^2 decaying through step 50.

Primary metric: CtQ_75 speedup of arm_c vs arm_a.
Secondary: retention ratio, bias cosine similarity to donor init over time.
Project gate: >= 10x. Strong partial: arm_c beats arm_a AND arm_b, retention >= 0.7.
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

N_COMPILE = 512
SEQ_LEN   = 128
COMPILE_BATCH = 8

N_EVAL    = 300
N_TRAIN   = 2000
N_STEPS   = 150          # shorter run: fine-grained logging catches speedup early
WARMUP_STEPS = 10        # bias-only warmup before backbone unfreeze
ANCHOR_DECAY = 50        # anchor lambda decays to 0 by this step
ANCHOR_LAMBDA0 = 1.0     # initial anchor coefficient

LR_BACKBONE = 1e-4
LR_BIAS     = 1e-3       # bias trains faster (small param count)
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


def log_interval(step):
    if step <= 20:
        return 1
    elif step <= 100:
        return 5
    else:
        return 25


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
            mask = enc["attention_mask"].bool()

            out_d = donor(**enc, output_hidden_states=True)
            out_l = lesion(**enc, output_hidden_states=True)

            for l_idx in range(n_layers + 1):
                hd = out_d.hidden_states[l_idx].float()
                hl = out_l.hidden_states[l_idx].float()
                for b in range(hd.shape[0]):
                    valid = mask[b]
                    sum_donor[l_idx]  += hd[b][valid].cpu().numpy().sum(axis=0)
                    sum_lesion[l_idx] += hl[b][valid].cpu().numpy().sum(axis=0)
                    if l_idx == 0:
                        n_tokens += valid.sum().item()

    biases = []
    for l in range(1, n_layers + 1):
        b = (sum_donor[l] - sum_lesion[l]) / max(n_tokens, 1)
        biases.append(b.astype(np.float32))

    print(f"  [{time.time()-t0:.1f}s] Mean-shift compiled from {n_tokens} tokens")
    return biases


# ── trainable bias module ─────────────────────────────────────────────────────

class LayerBiases(nn.Module):
    def __init__(self, d_model, n_layers, init_biases=None):
        super().__init__()
        self.biases = nn.ParameterList()
        for l in range(n_layers):
            if init_biases is not None:
                b_init = torch.tensor(init_biases[l], dtype=torch.float32, device=DEVICE)
            else:
                b_init = torch.zeros(d_model, dtype=torch.float32, device=DEVICE)
            self.biases.append(nn.Parameter(b_init))

    def init_values(self):
        return [b.detach().float().cpu().numpy() for b in self.biases]


def attach_bias_hooks(model, layer_biases, n_layers, fixed=False):
    hooks = []

    def make_hook(l, bias_param):
        def forward_hook(module, inp, out):
            bias = bias_param if fixed else bias_param
            bias_t = bias.to(dtype=torch.float32, device=DEVICE)
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
        bias_ref = layer_biases[l] if fixed else layer_biases.biases[l]
        h = model.model.layers[l].register_forward_hook(make_hook(l, bias_ref))
        hooks.append(h)
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── retention metric ──────────────────────────────────────────────────────────

def bias_cosine_sim(biases_module, init_biases):
    sims = []
    for l, b in enumerate(biases_module.biases):
        b_curr = b.detach().float().cpu().numpy()
        b_init = init_biases[l]
        cos = float(np.dot(b_curr, b_init) / (np.linalg.norm(b_curr) * np.linalg.norm(b_init) + 1e-10))
        sims.append(cos)
    return float(np.mean(sims))


# ── training arm ──────────────────────────────────────────────────────────────

def run_arm(model, tokenizer, train_texts, eval_texts, n_steps, t0, arm_name,
            fixed_biases=None, trainable_biases=None, use_anchor=False,
            anchor_init=None):
    """
    arm_name: 'arm_a', 'arm_b', 'arm_c'
    fixed_biases: list of numpy arrays for fixed hook (arm_b)
    trainable_biases: LayerBiases module (arm_a, arm_c)
    use_anchor: whether to apply decaying anchor penalty (arm_c)
    anchor_init: initial bias values for anchor (arm_c)
    """
    n_layers = len(model.model.layers)
    hooks = []
    bias_cosine_log = {}

    if fixed_biases is not None:
        # arm_b: fixed non-trainable hooks
        bias_tensors = [torch.tensor(b, dtype=torch.float32, device=DEVICE)
                        for b in fixed_biases]
        for p in model.parameters():
            p.requires_grad_(True)
        hooks = attach_bias_hooks(model, bias_tensors, n_layers, fixed=True)
        optimizer = AdamW(model.parameters(), lr=LR_BACKBONE, weight_decay=0.01)
    else:
        # arm_a or arm_c: trainable biases
        for p in model.parameters():
            p.requires_grad_(False)   # start frozen; unfreeze at WARMUP_STEPS
        for p in trainable_biases.parameters():
            p.requires_grad_(True)
        hooks = attach_bias_hooks(model, trainable_biases, n_layers, fixed=False)
        # Create optimizer (backbone params added after warmup)
        optimizer = AdamW(trainable_biases.parameters(), lr=LR_BIAS, weight_decay=0.01)

    trajectory = {}
    text_idx = 0
    n_texts  = len(train_texts)
    backbone_unfrozen = fixed_biases is not None  # arm_b starts unfrozen

    # Initial NLL
    nll0 = measure_nll(model, tokenizer, eval_texts)
    trajectory[0] = float(nll0)
    fg0 = (17.83 - nll0) / 13.99  # approximate, for logging
    print(f"  [{arm_name}] step  0 | NLL {nll0:.4f} | {time.time()-t0:.0f}s")
    if trainable_biases is not None and arm_name != 'arm_a':
        bias_cosine_log[0] = bias_cosine_sim(trainable_biases, anchor_init or [])
    model.train()

    for step in range(1, n_steps + 1):
        # Unfreeze backbone after warmup
        if trainable_biases is not None and step == WARMUP_STEPS + 1 and not backbone_unfrozen:
            backbone_unfrozen = True
            for p in model.parameters():
                p.requires_grad_(True)
            optimizer = AdamW([
                {'params': model.parameters(), 'lr': LR_BACKBONE},
                {'params': trainable_biases.parameters(), 'lr': LR_BIAS},
            ], weight_decay=0.01)
            print(f"  [{arm_name}] step {step:3d} | backbone unfrozen (NLL {trajectory.get(step-1, '?'):.4f})")

        # Training step
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
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1), ignore_index=-100
        )

        # Anchor penalty for arm_c
        loss = ce_loss
        if use_anchor and anchor_init is not None and step <= ANCHOR_DECAY:
            lam = ANCHOR_LAMBDA0 * max(0.0, 1.0 - step / ANCHOR_DECAY)
            anchor_pen = sum(
                (trainable_biases.biases[l] - torch.tensor(
                    anchor_init[l], dtype=torch.float32, device=DEVICE)).pow(2).sum()
                for l in range(n_layers)
            )
            loss = ce_loss + lam * anchor_pen

        optimizer.zero_grad()
        loss.backward()
        if backbone_unfrozen:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if trainable_biases is not None:
            torch.nn.utils.clip_grad_norm_(trainable_biases.parameters(), 1.0)
        optimizer.step()

        # Log according to fine-grained schedule
        interval = log_interval(step)
        if step % interval == 0:
            nll = measure_nll(model, tokenizer, eval_texts)
            trajectory[step] = float(nll)
            anchor_str = f" | anchor_lam={ANCHOR_LAMBDA0 * max(0., 1-step/ANCHOR_DECAY):.3f}" if use_anchor else ""
            print(f"  [{arm_name}] step {step:3d} | NLL {nll:.4f} | {time.time()-t0:.0f}s{anchor_str}")
            model.train()
            if trainable_biases is not None and arm_name == 'arm_c' and anchor_init:
                bias_cosine_log[step] = bias_cosine_sim(trainable_biases, anchor_init)

    remove_hooks(hooks)
    return trajectory, bias_cosine_log


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print(f"[{0:.1f}s] Loading texts (n_total=3200)...")
    all_texts     = load_all_texts(n_total=3200)
    compile_texts = all_texts[:N_COMPILE]
    eval_texts    = all_texts[N_COMPILE:N_COMPILE + N_EVAL]
    train_texts   = all_texts[N_COMPILE + N_EVAL:N_COMPILE + N_EVAL + N_TRAIN]
    print(f"  compile={len(compile_texts)} eval={len(eval_texts)} train={len(train_texts)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- COMPILE ----
    print(f"\n[{time.time()-t0:.1f}s] Loading donor + lesion for compile step...")
    donor = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    n_layers = donor.config.num_hidden_layers
    d_model  = donor.config.hidden_size

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
    biases_np = compile_meanshift(donor, lesion_c, tok, compile_texts, n_layers, d_model, t0)

    del donor, lesion_c
    torch.cuda.empty_cache()
    print(f"  Compile done. GPU cleared.")

    gap       = nll_lesion - nll_donor
    target_50 = nll_donor + 0.50 * gap
    target_75 = nll_donor + 0.25 * gap
    target_90 = nll_donor + 0.10 * gap
    print(f"\n  Gap: {gap:.4f} nats | target_75={target_75:.4f}")

    # ================================================================
    # ARM A: zero-init trainable biases + protected warmup
    # ================================================================
    print(f"\n{'='*60}")
    print(f"ARM A: zero-init trainable biases + protected warmup")
    print(f"{'='*60}")
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    with torch.no_grad():
        for l in range(n_layers):
            model_a.model.layers[l].mlp.down_proj.weight.zero_()
    biases_a = LayerBiases(d_model, n_layers, init_biases=None)
    traj_a, _ = run_arm(model_a, tok, train_texts, eval_texts, N_STEPS, t0, "arm_a",
                         trainable_biases=biases_a)
    del model_a, biases_a
    torch.cuda.empty_cache()

    # ================================================================
    # ARM B: donor-init fixed biases + full unfreeze (grafting_007 arm_b reference)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"ARM B: donor-init fixed biases + full unfreeze")
    print(f"{'='*60}")
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    with torch.no_grad():
        for l in range(n_layers):
            model_b.model.layers[l].mlp.down_proj.weight.zero_()
    traj_b, _ = run_arm(model_b, tok, train_texts, eval_texts, N_STEPS, t0, "arm_b",
                         fixed_biases=biases_np)
    del model_b
    torch.cuda.empty_cache()

    # ================================================================
    # ARM C: donor-init trainable biases + anchor + protected warmup
    # ================================================================
    print(f"\n{'='*60}")
    print(f"ARM C: donor-init trainable biases + anchor + protected warmup")
    print(f"{'='*60}")
    model_c = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    with torch.no_grad():
        for l in range(n_layers):
            model_c.model.layers[l].mlp.down_proj.weight.zero_()
    biases_c = LayerBiases(d_model, n_layers, init_biases=biases_np)
    anchor_init = biases_c.init_values()
    traj_c, cosine_log_c = run_arm(model_c, tok, train_texts, eval_texts, N_STEPS, t0, "arm_c",
                                    trainable_biases=biases_c, use_anchor=True,
                                    anchor_init=anchor_init)
    del model_c, biases_c
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
    print(f"  {'Step':>5}  {'Arm A':>10}  {'Arm B':>10}  {'Arm C':>10}")
    all_steps = sorted(set(list(traj_a.keys()) + list(traj_b.keys()) + list(traj_c.keys())))
    for step in all_steps:
        na = traj_a.get(step, float('nan'))
        nb = traj_b.get(step, float('nan'))
        nc = traj_c.get(step, float('nan'))
        print(f"  {step:5d}  {na:10.4f}  {nb:10.4f}  {nc:10.4f}")
    print(f"{'='*60}")

    metrics = {}
    for metric_name, traj_vs in [("CtQ_50", target_50), ("CtQ_75", target_75), ("CtQ_90", target_90)]:
        sa = steps_to_nll(traj_a, traj_vs)
        sb = steps_to_nll(traj_b, traj_vs)
        sc = steps_to_nll(traj_c, traj_vs)

        def speedup_str(sa, sx):
            if sx == 0:
                return "inf"
            elif sa is None or sx is None:
                return None
            else:
                return round(sa / sx, 2)

        sac = speedup_str(sa, sc)
        metrics[metric_name] = {
            "target_nll": float(traj_vs),
            "steps_a": sa, "steps_b": sb, "steps_c": sc,
            "speedup_c_vs_a": sac
        }
        print(f"  {metric_name}: arm_a={sa} arm_b={sb} arm_c={sc} speedup_c_vs_a={sac}")

    # Retention ratio: fg_closed at step 10 / fg_closed at step 0 for arm_c
    nll_c0 = traj_c.get(0, nll_lesion)
    nll_c10 = traj_c.get(10, traj_c.get(9, traj_c.get(8, nll_c0)))
    fg0 = (nll_lesion - nll_c0) / max(gap, 1e-6)
    fg10 = (nll_lesion - nll_c10) / max(gap, 1e-6)
    retention_ratio = fg10 / fg0 if fg0 > 0.01 else float('nan')
    print(f"\n  Retention ratio (fg@10 / fg@0): {retention_ratio:.3f}")

    sp75 = metrics.get("CtQ_75", {}).get("speedup_c_vs_a")
    if sp75 == "inf" or (isinstance(sp75, (int, float)) and sp75 >= 10.0):
        verdict = f"PASS: CtQ_75 arm_c speedup {sp75}x >= 10x — trainable mean-shift persists"
    elif isinstance(sp75, (int, float)) and sp75 >= 2.0:
        verdict = f"PARTIAL: CtQ_75 arm_c speedup {sp75:.1f}x — some persistence, below project gate"
    elif sp75 is None:
        verdict = f"KILL: arm_c does not reach target_75 within {N_STEPS} steps"
    else:
        verdict = f"KILL: CtQ_75 arm_c speedup {sp75}x < 2x — trainable bias does not help"

    print(f"\n  VERDICT: {verdict}")

    elapsed = float(time.time() - t0)
    out = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_compile": N_COMPILE,
        "n_steps": N_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "anchor_decay": ANCHOR_DECAY,
        "anchor_lambda0": ANCHOR_LAMBDA0,
        "lr_backbone": LR_BACKBONE,
        "lr_bias": LR_BIAS,
        "nll_donor": float(nll_donor),
        "nll_lesion_base": float(nll_lesion),
        "gap_nats": float(gap),
        "targets": {"50pct": float(target_50), "75pct": float(target_75), "90pct": float(target_90)},
        "trajectory_arm_a": {str(k): v for k, v in traj_a.items()},
        "trajectory_arm_b": {str(k): v for k, v in traj_b.items()},
        "trajectory_arm_c": {str(k): v for k, v in traj_c.items()},
        "bias_cosine_log_arm_c": {str(k): v for k, v in cosine_log_c.items()},
        "retention_ratio_c": retention_ratio,
        "speedup_metrics": metrics,
        "verdict": verdict,
        "elapsed_s": elapsed,
    }

    out_path = RESULTS / "grafting_008_trainable_meanshift_persistence.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}  (elapsed {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
