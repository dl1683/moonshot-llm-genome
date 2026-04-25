"""
genome_134_glue_only_trajectory.py

GLUE-ONLY TRAJECTORY TEST.

g125 PARTIAL: training only embed/lm_head + RMSNorm gammas (26% of params,
frozen random attention/MLP) achieves 42.66% gap closure in 100 steps on
Qwen3-0.6B. This bypassed the holism barrier in a surprising way.

g127-133: trained-spectrum invariant trajectory is universal across Pythia
sizes AND Llama-from-scratch.

Open question: does the trajectory hold under GLUE-ONLY training (not full
unfreeze)? If yes, the trajectory is even more robust — it emerges in the
RESIDUAL STREAM regardless of which weights are being updated, as long as
SOME training is happening on the interface layers.

Protocol:
  - Random init Qwen3-0.6B (g125 setup)
  - Freeze attention + MLP. Trainable: embed_tokens (tied with lm_head) + all RMSNorm.
  - Train CE loss for 100 steps on c4_clean_v1.
  - Measure sqrt(er)*alpha at steps [0, 5, 10, 25, 50, 100].

Pre-stated PASS:
  - Initial random sqrt(er)*alpha > 6
  - Some checkpoint dips below 3.5 (mode collapse)
  - Final step approaches target (within 30%)

If PASS: trajectory is interface-only-train robust. Suggests the spectral
trajectory is a property of the residual stream's response to gradient,
not specific to which weights are being optimized.

If KILL: trajectory requires full-unfreeze training; glue-only is special.

Results: results/genome_134_glue_only_trajectory.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent
TARGET = float(np.sqrt(18))
SEED = 42

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_STEPS = 100
EVAL_AT = [0, 5, 10, 25, 50, 100]
LR = 3e-4
BATCH_SIZE = 4
SEQ_LEN = 256


def is_glue_param(name):
    if "embed_tokens" in name or name == "lm_head.weight":
        return True
    if "layernorm" in name.lower() or name == "model.norm.weight":
        return True
    return False


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(max(X.shape[0] - 1, 1))
    return s.astype(np.float64)


def fit_power_tail(s, lo=0.05, hi=0.5):
    r = np.arange(1, len(s) + 1)
    a = max(1, int(len(s) * lo))
    b = int(len(s) * hi)
    lr = np.log(r[a:b])
    ls = np.log(s[a:b] + 1e-12)
    slope, _ = np.polyfit(lr, ls, 1)
    return float(-slope)


def eff_rank(s):
    s2 = s ** 2
    tot = s2.sum()
    return float(tot ** 2 / (s2 ** 2).sum()) if tot > 0 else 0.0


def measure_invariant(model, tok, calib_texts):
    n_layers = model.config.num_hidden_layers
    mid = max(1, n_layers // 2)
    traj = extract_trajectory(
        model=model, tokenizer=tok,
        texts=calib_texts, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key="qwen3-glue-only", class_id=1,
        quantization="fp16",
        stimulus_version="c4_clean.v1.seed42.n800",
        seed=42, batch_size=8, max_length=SEQ_LEN,
    )
    X = traj.layers[0].X.astype(np.float32)
    s = spectrum(X)
    a = fit_power_tail(s)
    er = eff_rank(s)
    return {"alpha": a, "eff_rank": er, "sqrt_er_alpha": float(np.sqrt(er) * a)}


def main():
    t0 = time.time()
    print("genome_134: glue-only trajectory on random-init Qwen3-0.6B")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading c4_clean_v1 stimuli (calib n=400, train n=400)...")
    all_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=800):
        all_texts.append(rec["text"])
    calib_texts = all_texts[:400]
    train_texts = all_texts[400:]
    print(f"  calib N={len(calib_texts)}, train N={len(train_texts)}")

    print("Building random-init Qwen3-0.6B...")
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).to("cuda")

    # Freeze attn + MLP, train only glue
    n_train, n_total = 0, 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        p.requires_grad = is_glue_param(name)
        if p.requires_grad:
            n_train += p.numel()
    print(f"  trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")

    # Pre-tokenize training
    enc = tok(train_texts, padding=True, truncation=True,
               max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc["input_ids"]
    train_mask = enc["attention_mask"]
    print(f"  train tensor shape: {train_ids.shape}")

    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=LR)
    rng = np.random.default_rng(SEED)
    rows = []

    # Step 0
    print(f"\n[{time.time()-t0:.1f}s] === step 0 (random init) ===")
    model.eval()
    r = measure_invariant(model, tok, calib_texts)
    rows.append({"step": 0, **r})
    print(f"  step= 0  alpha={r['alpha']:.3f}  er={r['eff_rank']:7.2f}  sqrt(er)*alpha={r['sqrt_er_alpha']:.3f}")

    step = 0
    next_idx = 1
    model.train()
    while step < MAX_STEPS:
        idx = rng.integers(0, train_ids.size(0), size=BATCH_SIZE)
        ids = train_ids[idx].to("cuda")
        mask = train_mask[idx].to("cuda")
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits
        sl = logits[:, :-1].contiguous()
        lbl = ids[:, 1:].contiguous().clone()
        sm = mask[:, 1:].contiguous()
        lbl[sm == 0] = -100
        loss = F.cross_entropy(
            sl.view(-1, sl.size(-1)), lbl.view(-1), ignore_index=-100
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        step += 1

        if next_idx < len(EVAL_AT) and step == EVAL_AT[next_idx]:
            print(f"\n[{time.time()-t0:.1f}s] === step {step} ===")
            model.eval()
            r = measure_invariant(model, tok, calib_texts)
            rows.append({"step": step, "loss": float(loss.item()), **r})
            print(f"  step={step:3d}  loss={loss.item():.3f}  alpha={r['alpha']:.3f}  "
                  f"er={r['eff_rank']:7.2f}  sqrt(er)*alpha={r['sqrt_er_alpha']:.3f}")
            model.train()
            next_idx += 1

    # Verdict
    invs = [r["sqrt_er_alpha"] for r in rows]
    steps = [r["step"] for r in rows]
    min_idx = int(np.argmin(invs))
    final_dev = abs(invs[-1] - TARGET) / TARGET * 100

    init_high = invs[0] > 6.0
    has_dip = any(v < 3.5 for v in invs)
    final_close = final_dev < 30.0

    print(f"\n=== TRAJECTORY (target {TARGET:.3f}) ===")
    for r in rows:
        mark = " (above)" if r["sqrt_er_alpha"] >= TARGET else " (below)"
        if r["step"] == steps[min_idx]:
            mark += " [MIN]"
        print(f"  step {r['step']:4d}: sqrt(er)*alpha={r['sqrt_er_alpha']:.3f}  er={r['eff_rank']:7.2f}{mark}")

    if init_high and has_dip and final_close:
        verdict = (f"PASS: glue-only training reproduces U-shape trajectory. "
                   f"random={invs[0]:.2f}>6, min={invs[min_idx]:.2f}<3.5, "
                   f"final={invs[-1]:.2f} (dev {final_dev:.1f}%<30%). "
                   f"Trajectory is interface-only-train robust.")
    elif init_high and has_dip:
        verdict = (f"PARTIAL: U-shape present but final not within 30% of target. "
                   f"random={invs[0]:.2f}, min={invs[min_idx]:.2f}, final={invs[-1]:.2f}, "
                   f"dev={final_dev:.1f}%.")
    elif init_high and final_close:
        verdict = (f"PARTIAL: random + final converged but no clear mode-collapse dip. "
                   f"random={invs[0]:.2f}, min={invs[min_idx]:.2f}, final={invs[-1]:.2f}.")
    else:
        verdict = (f"KILL: glue-only training does NOT trace the trajectory. "
                   f"random={invs[0]:.2f}, min={invs[min_idx]:.2f}, final={invs[-1]:.2f}, "
                   f"dev={final_dev:.1f}%.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 134, "name": "glue_only_trajectory",
        "model": MODEL_ID, "max_steps": MAX_STEPS, "lr": LR,
        "trainable_pct": 100*n_train/n_total,
        "target": TARGET, "rows": rows,
        "summary": {
            "min_step": steps[min_idx], "min_value": invs[min_idx],
            "final_value": invs[-1], "final_dev_pct": final_dev,
        },
        "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_134_glue_only_trajectory.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
