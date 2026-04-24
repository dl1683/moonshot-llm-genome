"""
grafting_005_ce_training_speedup.py

Does the 55% zero-step capability recovery from grafting_004 (best: n=1500, lambda=0.1)
translate into faster CE training convergence vs a lesioned baseline?

grafting_003/004 finding: analytical MLP transplant recovers ~55-60% of capability
at zero gradient steps, but is capped by mean-pooling information loss. The open
question: does starting from NLL~10 (grafted) converge to donor NLL faster than
starting from NLL~18 (lesioned)?

Protocol
--------
Compile step (n=1500, lambda=0.1 Ridge, ~30s):
  - Collect donor hidden states + lesion intermediates
  - Solve Ridge normal equations for all 28 layers
  - Install as down_proj.weight -> grafted init (NLL~10.0)

Training arms (300 CE steps each, sequential):
  Arm "lesion":  all down_proj=0,              NLL_0 ~ 17.86
  Arm "grafted": best Ridge weights installed,  NLL_0 ~ 10.08
  Both: identical pretrained attention/embed/LN; full unfreeze; same LR/data/schedule

Key metrics
-----------
  NLL at steps 0/25/50/100/150/200/300 for each arm
  CtQ_50: steps to reach 50% closure of (initial_lesion_NLL - donor_NLL) gap
          = NLL 10.87 nats. Grafted model starts there already -> CtQ_50_grafted=0.
  CtQ_75: steps to 75% closure = NLL 7.37 nats (requires training for both arms)
  Speedup = steps_lesion / steps_grafted to reach NLL_target_75

Pass: CtQ_75 speedup >= 2x (grafted reaches 7.37 NLL in half the steps)
Kill: < 1.5x (55% init provides no meaningful training acceleration)
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
SEED     = 42
MODEL_ID = "Qwen/Qwen3-0.6B"

# Compile step
N_COMPILE = 1500
RIDGE_LAM = 0.1
SEQ_LEN   = 128
BATCH     = 16

# Training
N_TRAIN_CE  = 2000   # CE training texts (different slice from compile set)
TRAIN_BATCH = 8
N_STEPS     = 300
LR          = 1e-4
LOG_EVERY   = 25
EVAL_BATCH  = 8
N_EVAL      = 300

# Gap metric targets (computed from lesion NLL ~17.86, donor NLL ~3.87)
# These are set dynamically based on measured NLLs


def load_texts_at_offset(offset, n, seed=42):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min((n + offset) * 4, len(ds)), replace=False)
    texts, skipped = [], 0
    for i in indices:
        t = ds[int(i)]["text"].strip()
        if len(t) >= 80:
            if skipped < offset:
                skipped += 1
            else:
                texts.append(t[:512])
        if len(texts) >= n:
            break
    return texts


def measure_nll(model, tokenizer, texts, max_len=SEQ_LEN, batch=EVAL_BATCH):
    model.eval()
    total_nll, total_toks = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(DEVICE)
            out   = model(**enc)
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


def collect_hidden(model, tokenizer, texts, n_layers):
    hidden = {l: [] for l in range(n_layers + 1)}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=SEQ_LEN).to(DEVICE)
            out  = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            for l, h in enumerate(out.hidden_states):
                pooled = (h.float() * mask).sum(1) / mask.sum(1)
                hidden[l].append(pooled.cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in hidden.items()}


def collect_mlp_intermediates(model, tokenizer, texts, n_layers):
    mlp_interm = {l: [] for l in range(n_layers)}
    hooks = []

    def make_pre_hook(layer_idx):
        def pre_hook(mod, inp):
            x = inp[0].float().mean(dim=1).detach().cpu().numpy()
            mlp_interm[layer_idx].append(x)
        return pre_hook

    for l in range(n_layers):
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
            make_pre_hook(l)))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=SEQ_LEN).to(DEVICE)
            model(**enc)

    for h in hooks:
        h.remove()
    return {l: np.concatenate(v, axis=0) for l, v in mlp_interm.items()}


def ridge_solve(h_donor, h_lesion, mlp_interm, n_layers, n_train, lam):
    solutions = {}
    for L in range(n_layers):
        r = (h_donor[L+1][:n_train] - h_lesion[L+1][:n_train]).astype(np.float64)
        f = mlp_interm[L][:n_train].astype(np.float64)
        FtF = f.T @ f
        FtR = f.T @ r
        A   = FtF + lam * np.eye(f.shape[1])
        sol = np.linalg.solve(A, FtR)  # (d_interm, d_hidden)
        solutions[L] = sol
    return solutions


def install_solutions(model, solutions, n_layers):
    with torch.no_grad():
        for L in range(n_layers):
            w = torch.tensor(solutions[L].T, dtype=torch.float16).to(DEVICE)
            model.model.layers[L].mlp.down_proj.weight.copy_(w)


def zero_mlp(model, n_layers):
    with torch.no_grad():
        for L in range(n_layers):
            model.model.layers[L].mlp.down_proj.weight.zero_()


def train_arm(model, tokenizer, train_texts, eval_texts, n_steps, t0, arm_name):
    """Train arm for n_steps CE steps. Return trajectory {step: nll}."""
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()

    trajectory = {}
    text_idx   = 0
    n_texts    = len(train_texts)

    for step in range(n_steps + 1):
        if step % LOG_EVERY == 0:
            nll = measure_nll(model, tokenizer, eval_texts)
            trajectory[step] = float(nll)
            print(f"  [{arm_name}] step {step:4d} | NLL {nll:.4f} | {time.time()-t0:.0f}s")
            model.train()

        if step == n_steps:
            break

        # Get next batch
        chunk = []
        while len(chunk) < TRAIN_BATCH:
            chunk.append(train_texts[text_idx % n_texts])
            text_idx += 1

        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=SEQ_LEN).to(DEVICE)
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

    return trajectory


def steps_to_nll(trajectory, target_nll):
    """First step where NLL <= target_nll, or None if never reached."""
    for step in sorted(trajectory.keys()):
        if trajectory[step] <= target_nll:
            return step
    return None


def main():
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading texts...")
    compile_texts = load_texts_at_offset(0,    N_COMPILE,  SEED)
    eval_texts    = load_texts_at_offset(5000, N_EVAL,     SEED)
    train_texts   = load_texts_at_offset(6000, N_TRAIN_CE, SEED)
    print(f"  compile={len(compile_texts)} eval={len(eval_texts)} train={len(train_texts)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- Compile step: collect donor hidden states ----
    print(f"\n[{time.time()-t0:.1f}s] Loading donor for compile step...")
    donor = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    n_layers = donor.config.num_hidden_layers

    nll_donor = measure_nll(donor, tok, eval_texts)
    print(f"  Donor NLL (eval): {nll_donor:.4f}")

    print(f"[{time.time()-t0:.1f}s] Collecting donor hidden states (n={N_COMPILE})...")
    h_donor = collect_hidden(donor, tok, compile_texts, n_layers)
    del donor
    torch.cuda.empty_cache()

    # ---- Load recipient, measure lesion NLL, collect intermediates ----
    print(f"\n[{time.time()-t0:.1f}s] Loading recipient, creating lesion...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    zero_mlp(model, n_layers)

    nll_lesion = measure_nll(model, tok, eval_texts)
    print(f"  Lesion NLL (eval): {nll_lesion:.4f}")

    print(f"[{time.time()-t0:.1f}s] Collecting lesion intermediates for Ridge solve...")
    h_lesion   = collect_hidden(model, tok, compile_texts, n_layers)
    mlp_interm = collect_mlp_intermediates(model, tok, compile_texts, n_layers)

    # ---- Ridge solve ----
    print(f"\n[{time.time()-t0:.1f}s] Ridge solve (n={N_COMPILE}, lambda={RIDGE_LAM})...")
    solutions  = ridge_solve(h_donor, h_lesion, mlp_interm, n_layers, N_COMPILE, RIDGE_LAM)
    del h_donor, h_lesion, mlp_interm

    # ---- Gap targets ----
    gap = nll_lesion - nll_donor
    target_50 = nll_donor + 0.50 * gap
    target_75 = nll_donor + 0.25 * gap  # 75% gap closure = 25% remaining
    target_90 = nll_donor + 0.10 * gap
    print(f"  Gap: {gap:.4f} nats | target_50={target_50:.4f} target_75={target_75:.4f} target_90={target_90:.4f}")

    # ================================================================
    # ARM A: LESION baseline (down_proj = 0)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"[{time.time()-t0:.1f}s] ARM A: lesion baseline (NLL_0 = {nll_lesion:.4f})")
    print(f"{'='*60}")
    zero_mlp(model, n_layers)
    traj_lesion = train_arm(model, tok, train_texts, eval_texts, N_STEPS, t0, "lesion")

    # ================================================================
    # ARM B: GRAFTED (Ridge n=1500, lambda=0.1 weights)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"[{time.time()-t0:.1f}s] ARM B: grafted init")
    print(f"{'='*60}")
    zero_mlp(model, n_layers)
    install_solutions(model, solutions, n_layers)
    nll_grafted_0 = measure_nll(model, tok, eval_texts)
    print(f"  Grafted NLL_0: {nll_grafted_0:.4f}")
    traj_grafted = train_arm(model, tok, train_texts, eval_texts, N_STEPS, t0, "grafted")

    # ================================================================
    # Analysis
    # ================================================================
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"  Donor NLL (ceiling):  {nll_donor:.4f}")
    print(f"  Lesion NLL (arm A 0): {nll_lesion:.4f}")
    print(f"  Grafted NLL (arm B 0): {nll_grafted_0:.4f}")
    print(f"{'='*60}")
    print(f"  {'Step':>5}  {'Lesion':>8}  {'Grafted':>8}  {'Advantage':>10}")
    for step in sorted(traj_lesion.keys()):
        nl = traj_lesion[step]
        ng = traj_grafted.get(step, float('nan'))
        adv = nl - ng
        print(f"  {step:5d}  {nl:8.4f}  {ng:8.4f}  {adv:+10.4f}")
    print(f"{'='*60}")

    # Compute speedup metrics
    def speedup(traj_a, traj_b, target):
        sa = steps_to_nll(traj_a, target)
        sb = steps_to_nll(traj_b, target)
        if sa is None and sb is None: return None, None, None
        if sb == 0: return "inf", sa, 0
        if sa is None: return None, None, sb
        return sa / sb, sa, sb

    metrics = {}
    for name, target in [("CtQ_50", target_50), ("CtQ_75", target_75), ("CtQ_90", target_90)]:
        sp, steps_a, steps_b = speedup(traj_lesion, traj_grafted, target)
        metrics[name] = {"target_nll": float(target), "steps_lesion": steps_a,
                         "steps_grafted": steps_b, "speedup": sp}
        label = f"{sp:.2f}x" if isinstance(sp, float) else str(sp)
        print(f"  {name} (NLL={target:.2f}): lesion={steps_a} grafted={steps_b} speedup={label}")

    # Verdict
    sp75 = metrics.get("CtQ_75", {}).get("speedup")
    if isinstance(sp75, float) and sp75 >= 2.0:
        verdict = f"PASS: CtQ_75 speedup {sp75:.2f}x >= 2x — grafted init accelerates training"
    elif isinstance(sp75, (float, int)) and sp75 >= 1.5:
        verdict = f"PARTIAL: CtQ_75 speedup {sp75:.2f}x (1.5x-2x)"
    elif sp75 == "inf":
        verdict = "PASS: grafted already at target_75, lesion never reaches it"
    elif sp75 is None:
        verdict = f"KILL: neither arm reaches target_75 within {N_STEPS} steps"
    else:
        verdict = f"KILL: CtQ_75 speedup {sp75:.2f}x < 1.5x"

    print(f"\n  VERDICT: {verdict}")

    summary = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "n_compile": N_COMPILE,
        "ridge_lambda": RIDGE_LAM,
        "n_steps": N_STEPS,
        "lr": LR,
        "nll_donor": float(nll_donor),
        "nll_lesion_arm_a_step0": float(nll_lesion),
        "nll_grafted_arm_b_step0": float(nll_grafted_0),
        "gap_nats": float(gap),
        "targets": {"50pct": float(target_50), "75pct": float(target_75), "90pct": float(target_90)},
        "trajectory_lesion": {str(k): v for k, v in traj_lesion.items()},
        "trajectory_grafted": {str(k): v for k, v in traj_grafted.items()},
        "speedup_metrics": {k: {
            **v, "speedup": str(v["speedup"]) if v["speedup"] is not None else None
        } for k, v in metrics.items()},
        "verdict": verdict,
        "elapsed_s": float(time.time() - t0),
    }

    out = RESULTS / "grafting_005_ce_training_speedup.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
