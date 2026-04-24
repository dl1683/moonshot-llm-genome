"""
grafting_009_weightspace_seed.py

Rank-1 weight-space seed vs zero-init baseline.

Background
----------
grafting_006-008 showed all output-space priors (hooks, adapters, trainable biases)
are nullified within 25-50 CE steps.  Root cause: CE gradient adapts backbone to
cancel the static output-space prior (bias cosine stayed 0.9999 throughout g_008
because the anchor froze the bias, and the backbone simply compensated).

grafting_009: bake the mean-shift INTO the weight matrix so there is no conflict.
- No hooks. No adapters. No warmup. Just a different W_init.
- Rank-1 seed (Codex-corrected formulation):
    mu_inner[l]      = E_lesion[gate*up activation]   (forward_pre_hook on lesion)
    mu_out_target[l] = W_down_donor[l] @ mu_inner[l]  (analytical — consistent distribution)
    W_seed[l]        = outer(mu_out_target, mu_inner) / (||mu_inner||^2 + ridge)
    ridge[l]         = 1e-4 * E[||x||^2]  (prevents blow-up when mu_inner is small)
  Property: W_seed[l] @ mu_inner[l] ≈ mu_out_target[l]  (exact to ridge term)
- Unlike bias hook, gradient updates W freely from the rank-1 starting point — no cancellation.

Protocol
--------
Arm A (baseline): lesion (all down_proj=0), full unfreeze from step 0, plain CE.
Arm B (seed):     lesion + rank-1 seed in down_proj.weight, full unfreeze, plain CE.
Arm C (oracle):   lesion + full donor down_proj weights copied — trivially restores donor.
                  Unscored; sanity-check only.

Primary metric: CtQ_75 speedup arm_b vs arm_a.  Gate: >= 10x.  Kill: < 2x.
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

N_COMPILE     = 1024
SEQ_LEN       = 128
COMPILE_BATCH = 8

N_EVAL      = 300
N_TRAIN     = 2000
N_STEPS     = 150
LR          = 1e-4
TRAIN_BATCH = 8
EVAL_BATCH  = 8
GRAD_CLIP   = 1.0


# -- data ------------------------------------------------------------------

def load_all_texts(n_total=3500, seed=SEED):
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


# -- metrics ---------------------------------------------------------------

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
    if step <= 20:    return 1
    elif step <= 100: return 5
    else:             return 25


# -- compile ---------------------------------------------------------------

def compile_rank1_seeds(donor, lesion, tokenizer, compile_texts,
                        n_layers, d_model, d_inner, t0):
    """
    Codex-corrected formulation:
      mu_inner[l]      = E_lesion[gate*up]            (forward_pre_hook, lesion only)
      mu_out_target[l] = W_down_donor[l] @ mu_inner   (analytical — consistent distribution)
      ridge[l]         = 1e-4 * E[||x||^2]
      W_seed[l]        = outer(mu_out_target, mu_inner) / (||mu_inner||^2 + ridge)

    signal_frac[l] = ||mu_inner||^2 / E[||x||^2]
    If signal_frac < 1e-4, layer is degenerate; set to zero.
    """
    sum_inner    = [np.zeros(d_inner, dtype=np.float64) for _ in range(n_layers)]
    sum_inner_sq = [0.0 for _ in range(n_layers)]
    n_tokens     = 0
    cur_mask     = [None]

    hooks = []
    for l in range(n_layers):
        def make_pre_hook(l_idx):
            def hook(module, inp):
                mask = cur_mask[0]
                if mask is None:
                    return
                x = inp[0]  # gate*up (down_proj input), shape [B, T, d_inner]
                for b in range(x.shape[0]):
                    valid = mask[b]
                    tokens = x[b][valid].detach().float().cpu().numpy()
                    sum_inner[l_idx]    += tokens.sum(axis=0)
                    sum_inner_sq[l_idx] += float((tokens ** 2).sum())
            return hook
        hooks.append(
            lesion.model.layers[l].mlp.down_proj.register_forward_pre_hook(make_pre_hook(l))
        )

    lesion.eval()
    with torch.no_grad():
        for i in range(0, len(compile_texts), COMPILE_BATCH):
            chunk = compile_texts[i:i + COMPILE_BATCH]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=SEQ_LEN).to(DEVICE)
            mask = enc["attention_mask"].bool()
            cur_mask[0] = mask
            n_tokens += int(mask.sum().item())
            lesion(**enc)

    for h in hooks:
        h.remove()

    # Extract donor down_proj weights (CPU float32 for stable matmul)
    donor_weights = []
    for l in range(n_layers):
        W_d = donor.model.layers[l].mlp.down_proj.weight.detach().float().cpu().numpy()
        donor_weights.append(W_d)  # shape [d_model, d_inner]

    seeds        = []
    signal_fracs = []
    frob_norms   = []

    for l in range(n_layers):
        mu_inner    = sum_inner[l] / max(n_tokens, 1)
        avg_sq_norm = sum_inner_sq[l] / max(n_tokens, 1)
        norm_sq     = float(np.dot(mu_inner, mu_inner))
        ridge       = 1e-4 * avg_sq_norm
        signal_frac = norm_sq / max(avg_sq_norm, 1e-30)
        signal_fracs.append(float(signal_frac))

        if signal_frac < 1e-4:
            W_seed = np.zeros((d_model, d_inner), dtype=np.float32)
        else:
            mu_out_target = donor_weights[l] @ mu_inner   # [d_model]
            W_seed = (
                np.outer(mu_out_target, mu_inner) / (norm_sq + ridge)
            ).astype(np.float32)

        seeds.append(W_seed)
        frob_norms.append(float(np.linalg.norm(W_seed)))

    degenerate = sum(1 for sf in signal_fracs if sf < 1e-4)
    print(f"  [{time.time()-t0:.1f}s] Rank-1 seeds compiled from {n_tokens} tokens")
    print(f"  signal_frac range: [{min(signal_fracs):.4e}, {max(signal_fracs):.4e}]"
          f"  ({degenerate} degenerate layers)")
    print(f"  W_seed Frobenius range: [{min(frob_norms):.3f}, {max(frob_norms):.3f}]")
    return seeds, signal_fracs


# -- model helpers ---------------------------------------------------------

def load_lesion(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    ).to(DEVICE)
    with torch.no_grad():
        for l in range(len(model.model.layers)):
            model.model.layers[l].mlp.down_proj.weight.zero_()
    return model


def install_seeds(model, seeds):
    with torch.no_grad():
        for l, W_seed in enumerate(seeds):
            W = torch.tensor(
                W_seed,
                dtype=model.model.layers[l].mlp.down_proj.weight.dtype,
                device=DEVICE,
            )
            model.model.layers[l].mlp.down_proj.weight.copy_(W)


def install_donor_weights(model, donor):
    """Copy full donor down_proj weights into model (oracle arm_c)."""
    with torch.no_grad():
        for l in range(len(model.model.layers)):
            W_d = donor.model.layers[l].mlp.down_proj.weight
            model.model.layers[l].mlp.down_proj.weight.copy_(W_d)


# -- training arm ----------------------------------------------------------

def run_arm(model, tokenizer, train_texts, eval_texts, n_steps, t0, arm_name):
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    text_idx = 0
    n_texts  = len(train_texts)
    trajectory = {}

    nll0 = measure_nll(model, tokenizer, eval_texts)
    trajectory[0] = float(nll0)
    print(f"  [{arm_name}] step  0 | NLL {nll0:.4f} | {time.time()-t0:.0f}s")
    model.train()

    for step in range(1, n_steps + 1):
        batch = []
        for _ in range(TRAIN_BATCH):
            batch.append(train_texts[text_idx % n_texts])
            text_idx += 1

        enc = tokenizer(batch, return_tensors="pt", padding=True,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        interval = log_interval(step)
        if step % interval == 0:
            nll = measure_nll(model, tokenizer, eval_texts)
            trajectory[step] = float(nll)
            print(f"  [{arm_name}] step {step:3d} | NLL {nll:.4f} | {time.time()-t0:.0f}s")
            model.train()

    return trajectory


# -- main ------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"[{0.0:.1f}s] Loading texts (n_total=3500)...")
    all_texts     = load_all_texts(n_total=3500)
    compile_texts = all_texts[:N_COMPILE]
    eval_texts    = all_texts[N_COMPILE:N_COMPILE + N_EVAL]
    train_texts   = all_texts[N_COMPILE + N_EVAL:N_COMPILE + N_EVAL + N_TRAIN]
    print(f"  compile={len(compile_texts)} eval={len(eval_texts)} train={len(train_texts)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"\n[{time.time()-t0:.1f}s] Loading donor + lesion for compile step...")
    donor  = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    ).to(DEVICE)
    lesion_compile = load_lesion(tok)

    nll_donor  = measure_nll(donor,          tok, eval_texts)
    nll_lesion = measure_nll(lesion_compile, tok, eval_texts)
    print(f"  Donor NLL: {nll_donor:.4f}")
    print(f"  Lesion NLL: {nll_lesion:.4f}")

    n_layers = len(donor.model.layers)
    d_model  = donor.config.hidden_size
    d_inner  = donor.model.layers[0].mlp.down_proj.in_features
    print(f"  n_layers={n_layers} d_model={d_model} d_inner={d_inner}")

    print(f"\n[{time.time()-t0:.1f}s] Compiling rank-1 seeds...")
    seeds, signal_fracs = compile_rank1_seeds(
        donor, lesion_compile, tok, compile_texts, n_layers, d_model, d_inner, t0
    )

    # Quick NLL check: install seed into a throw-away lesion to confirm step-0 effect
    print(f"\n[{time.time()-t0:.1f}s] Verifying arm B step-0 NLL...")
    check = load_lesion(tok)
    install_seeds(check, seeds)
    nll_b0  = measure_nll(check, tok, eval_texts)
    gap_nats = nll_lesion - nll_donor
    frac_b0 = (nll_lesion - nll_b0) / gap_nats
    print(f"  arm_b step-0 NLL: {nll_b0:.4f}  gap_closed: {frac_b0:.4f}")
    del check
    torch.cuda.empty_cache()

    # Quick NLL check for oracle arm_c
    print(f"\n[{time.time()-t0:.1f}s] Verifying arm C (oracle) step-0 NLL...")
    check_c = load_lesion(tok)
    install_donor_weights(check_c, donor)
    nll_c0  = measure_nll(check_c, tok, eval_texts)
    frac_c0 = (nll_lesion - nll_c0) / gap_nats
    print(f"  arm_c step-0 NLL: {nll_c0:.4f}  gap_closed: {frac_c0:.4f}")
    del check_c
    torch.cuda.empty_cache()

    del lesion_compile
    torch.cuda.empty_cache()

    tgt50 = nll_donor + 0.50 * gap_nats
    tgt75 = nll_donor + 0.25 * gap_nats
    tgt90 = nll_donor + 0.10 * gap_nats
    print(f"\n  Gap: {gap_nats:.4f} nats | target_75={tgt75:.4f}")

    # ── ARM A: zero-init lesion baseline ──────────────────────────────────
    print(f"\n{'='*60}")
    print("ARM A: zero-init down_proj (lesion baseline)")
    print('='*60)
    model_a = load_lesion(tok)
    traj_a  = run_arm(model_a, tok, train_texts, eval_texts, N_STEPS, t0, "arm_a")
    del model_a
    torch.cuda.empty_cache()

    # ── ARM B: rank-1 weight seed ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ARM B: rank-1 weight seed (W_donor @ mu_inner x mu_inner / ||mu_inner||^2)")
    print('='*60)
    model_b = load_lesion(tok)
    install_seeds(model_b, seeds)
    traj_b  = run_arm(model_b, tok, train_texts, eval_texts, N_STEPS, t0, "arm_b")
    del model_b
    torch.cuda.empty_cache()

    # ── ARM C: full donor copy (oracle, unscored) ──────────────────────────
    print(f"\n{'='*60}")
    print("ARM C: full donor down_proj copy (oracle sanity — expected trivial win)")
    print('='*60)
    model_c = load_lesion(tok)
    install_donor_weights(model_c, donor)
    traj_c  = run_arm(model_c, tok, train_texts, eval_texts, N_STEPS, t0, "arm_c")
    del model_c, donor
    torch.cuda.empty_cache()

    # ── Speedup metrics ───────────────────────────────────────────────────
    steps_a_50 = steps_to_nll(traj_a, tgt50)
    steps_b_50 = steps_to_nll(traj_b, tgt50)
    steps_a_75 = steps_to_nll(traj_a, tgt75)
    steps_b_75 = steps_to_nll(traj_b, tgt75)
    steps_a_90 = steps_to_nll(traj_a, tgt90)
    steps_b_90 = steps_to_nll(traj_b, tgt90)

    def speedup(sa, sb):
        if sa is None or sb is None:
            return None
        if sb == 0:
            return "inf"
        return round(sa / sb, 2)

    sp75 = speedup(steps_a_75, steps_b_75)

    if sp75 is None:
        verdict = "INCONCLUSIVE: CtQ_75 not reached by at least one arm within N_STEPS"
    elif sp75 == "inf" or (isinstance(sp75, float) and sp75 >= 10.0):
        verdict = f"PASS: CtQ_75 speedup {sp75}x >= 10x -- rank-1 weight seed accelerates training"
    elif isinstance(sp75, float) and sp75 >= 2.0:
        verdict = f"PARTIAL: CtQ_75 speedup {sp75}x in [2x, 10x) -- meaningful but below gate"
    else:
        verdict = f"KILL: CtQ_75 speedup {sp75}x < 2x -- rank-1 seed does not accelerate training"

    results = {
        "model":                     MODEL_ID,
        "n_layers":                  n_layers,
        "d_model":                   d_model,
        "d_inner":                   d_inner,
        "n_compile":                 N_COMPILE,
        "n_steps":                   N_STEPS,
        "lr":                        LR,
        "grad_clip":                 GRAD_CLIP,
        "nll_donor":                 float(nll_donor),
        "nll_lesion_base":           float(nll_lesion),
        "nll_arm_b_step0":           float(nll_b0),
        "nll_arm_c_step0_oracle":    float(nll_c0),
        "fraction_gap_closed_b0":    float(frac_b0),
        "fraction_gap_closed_c0":    float(frac_c0),
        "gap_nats":                  float(gap_nats),
        "signal_fracs_per_layer":    signal_fracs,
        "n_degenerate_layers":       sum(1 for sf in signal_fracs if sf < 1e-4),
        "targets":                   {"50pct": float(tgt50), "75pct": float(tgt75), "90pct": float(tgt90)},
        "trajectory_arm_a":          {str(k): v for k, v in traj_a.items()},
        "trajectory_arm_b":          {str(k): v for k, v in traj_b.items()},
        "trajectory_arm_c_oracle":   {str(k): v for k, v in traj_c.items()},
        "speedup_metrics": {
            "CtQ_50": {
                "target_nll": float(tgt50),
                "steps_a":    steps_a_50,
                "steps_b":    steps_b_50,
                "speedup_b_vs_a": str(speedup(steps_a_50, steps_b_50)),
            },
            "CtQ_75": {
                "target_nll": float(tgt75),
                "steps_a":    steps_a_75,
                "steps_b":    steps_b_75,
                "speedup_b_vs_a": str(sp75),
            },
            "CtQ_90": {
                "target_nll": float(tgt90),
                "steps_a":    steps_a_90,
                "steps_b":    steps_b_90,
                "speedup_b_vs_a": str(speedup(steps_a_90, steps_b_90)),
            },
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }

    out_path = RESULTS / "grafting_009_weightspace_seed.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.time()-t0:.0f}s] Results -> {out_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
