"""
genome_153_mlp_depth_factorial.py

CODEX MECHANISM TEST — 2x2 FACTORIAL (depth × MLP).

g149+g150: minimal_7L (no MLP, 200M) collapses harder than baseline_full
(14L+MLP) at lr=1e-3 even with warmup. Codex mechanism conjecture:
the issue is "active residual-branch density" — minimal has 7 attn
branches vs baseline's 28 (14 attn + 14 MLP). Same global LR -> larger
effective per-branch update -> narrower stable LR window.

This experiment isolates depth from MLP via 2x2 factorial:
  Arms (200M-class, hidden=1024, ffn=2304):
    A) 14L + MLP        (~209M, 28 branches)
    B) 14L no-MLP       (~ 88M, 14 branches)
    C) 7L  + MLP        (~110M, 14 branches — branch-match to B)
    D) 7L  no-MLP       (~ 81M, 7 branches — minimal)

Match optimizer steps (NOT FLOPs) — this is an optimization-geometry
test, not an efficiency test. All 4 arms get same step count (4000).

LR sweep: {2e-4, 3e-4, 4e-4, 6e-4, 8e-4, 1e-3} (6 LRs).
Single seed (42) per cell to keep cost manageable: 4 arms × 6 LRs = 24 runs.

Per cell, log final loss + C4 top-1 + warmup-ok (last 1000-step loss
non-increasing) flag.

Prediction if branch-density mechanism is right:
  Stability order: 14L+MLP > 14L no-MLP ≈ 7L+MLP > 7L no-MLP
  Key discriminator: 7L+MLP — if mostly rescues collapse, branch-count
  is the story.

Pre-stated criteria:
  PASS_BRANCH: 7L+MLP s critical LR is closer to 14L+MLP than to 7L no-MLP.
              Branch density explains stability.
  PASS_M1: all no-MLP arms (regardless of depth) collapse at high LR
            similarly. MLP is special, not depth/branch count.
  AMBIGUOUS: mixed signal.

LR warmup over 200 steps for all cells.

Compute: 24 runs × ~5 min = ~2 hours.

Results: results/genome_153_mlp_depth_factorial.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent

SEQ_LEN = 256
BATCH_SIZE = 8
SEED = 42
N_C4_EVAL = 200
N_TRAIN = 32768
TRAIN_STEPS = 4000
LR_WARMUP_STEPS = 200
LR_GRID = [2e-4, 3e-4, 4e-4, 6e-4, 8e-4, 1e-3]


class ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def make_llama(vocab_size, hidden, layers, heads, ffn, no_mlp=False, seed=42):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        intermediate_size=ffn,
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg).to("cuda").to(torch.bfloat16)
    if no_mlp:
        for layer in model.model.layers:
            layer.mlp = ZeroMLP()
    return model


def measure_full(model, eval_ids, eval_mask):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    correct_top1 = 0
    with torch.no_grad():
        for i in range(0, eval_ids.size(0), BATCH_SIZE):
            ids = eval_ids[i:i+BATCH_SIZE].to("cuda")
            mask = eval_mask[i:i+BATCH_SIZE].to("cuda")
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = out.logits
            sl = logits[:, :-1].contiguous()
            lbl = ids[:, 1:].contiguous().clone()
            sm = mask[:, 1:].contiguous()
            valid = (sm != 0)
            lbl_for_loss = lbl.clone()
            lbl_for_loss[~valid] = -100
            loss = F.cross_entropy(
                sl.view(-1, sl.size(-1)), lbl_for_loss.view(-1),
                ignore_index=-100, reduction="sum",
            )
            n = valid.sum().item()
            total_loss += loss.item()
            total_tokens += n
            preds = sl.argmax(dim=-1)
            correct_top1 += ((preds == lbl) & valid).sum().item()
    model.train()
    return {"nll": total_loss / max(total_tokens, 1),
            "top1_acc": correct_top1 / max(total_tokens, 1)}


def warmup_lr(step, target_lr, warmup_steps):
    if step < warmup_steps:
        return target_lr * (step + 1) / warmup_steps
    return target_lr


def train_cell(arm_name, lr_target, model, train_ids, train_mask):
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name} lr={lr_target}: params={n_total/1e6:.1f}M warmup={LR_WARMUP_STEPS}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr_target, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(SEED)
    t = time.time()
    model.train()
    n_train = train_ids.size(0)
    nan_seen = False
    losses = []
    for step in range(1, TRAIN_STEPS + 1):
        current_lr = warmup_lr(step, lr_target, LR_WARMUP_STEPS)
        for g in opt.param_groups:
            g['lr'] = current_lr
        idx = rng.integers(0, n_train, size=BATCH_SIZE)
        ids = train_ids[idx].to("cuda")
        mask = train_mask[idx].to("cuda")
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits
        sl = logits[:, :-1].contiguous()
        lbl = ids[:, 1:].contiguous().clone()
        sm = mask[:, 1:].contiguous()
        lbl[sm == 0] = -100
        loss = F.cross_entropy(
            sl.view(-1, sl.size(-1)), lbl.view(-1), ignore_index=-100
        )
        if not torch.isfinite(loss):
            nan_seen = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            losses.append(float(loss.item()))
    final_loss = losses[-1] if losses else float('nan')
    # last-1000-step trend: are last 2 logged points (step 3500, 4000) lower than earlier?
    if len(losses) >= 4:
        late_increasing = (losses[-1] > losses[-3] + 0.3)
    else:
        late_increasing = False
    return n_total, time.time() - t, nan_seen, final_loss, losses, late_increasing


def main():
    t0 = time.time()
    print("genome_153: 2x2 factorial (depth x MLP) LR-stability test (Codex mechanism)")
    print(f"  LR grid: {LR_GRID}, 4 arms x 6 LRs = 24 runs")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print(f"Loading {N_TRAIN} c4...")
    pool_texts = []
    target_n = N_TRAIN + N_C4_EVAL
    for rec in c4_clean_v1(seed=42, n_samples=target_n):
        pool_texts.append(rec["text"])
        if len(pool_texts) >= target_n:
            break
    train_texts = pool_texts[:N_TRAIN]
    eval_texts = pool_texts[N_TRAIN:N_TRAIN + N_C4_EVAL]

    enc_t = tok(train_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc_t["input_ids"]; train_mask = enc_t["attention_mask"]
    enc_e = tok(eval_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    eval_ids = enc_e["input_ids"]; eval_mask = enc_e["attention_mask"]

    arms = [
        ("14L_MLP",     dict(hidden=1024, layers=14, heads=16, ffn=2304, no_mlp=False)),
        ("14L_noMLP",   dict(hidden=1024, layers=14, heads=16, ffn=2304, no_mlp=True)),
        ("7L_MLP",      dict(hidden=1024, layers=7,  heads=16, ffn=2304, no_mlp=False)),
        ("7L_noMLP",    dict(hidden=1024, layers=7,  heads=16, ffn=2304, no_mlp=True)),
    ]

    results = {}
    for arm_name, kw in arms:
        results[arm_name] = {"per_lr": {}, "params_M": None}
        for lr in LR_GRID:
            print(f"\n=== {arm_name} lr={lr:.0e} ===")
            try:
                model = make_llama(actual_vocab, seed=SEED, **kw)
            except Exception as e:
                print(f"  build fail: {e}"); continue
            n_total, elapsed, nan_seen, final_loss, losses, late_increasing = train_cell(
                arm_name, lr, model, train_ids, train_mask,
            )
            print(f"  {elapsed:.0f}s, final_loss={final_loss:.3f}, nan={nan_seen}, late_inc={late_increasing}")
            if nan_seen:
                metrics = {"nll": float("nan"), "top1_acc": float("nan")}
            else:
                metrics = measure_full(model, eval_ids, eval_mask)
                print(f"    C4 NLL={metrics['nll']:.4f} top1={100*metrics['top1_acc']:.2f}%")
            results[arm_name]["per_lr"][lr] = {
                "metrics": metrics, "nan_seen": nan_seen,
                "final_loss": final_loss, "late_increasing": late_increasing,
                "wallclock_s": elapsed,
            }
            results[arm_name]["params_M"] = n_total / 1e6
            del model; torch.cuda.empty_cache()

    # Critical-LR analysis: highest LR where training is stable
    print(f"\n=== CRITICAL-LR ANALYSIS ===")
    crit_lr = {}
    for arm_name, _ in arms:
        per_lr = results[arm_name]["per_lr"]
        # Stable cell: not NaN, not late-increasing, top1 >= 0.10 (way above random)
        stable_lrs = [lr for lr in LR_GRID
                      if (lr in per_lr and not per_lr[lr]["nan_seen"]
                          and not per_lr[lr]["late_increasing"]
                          and per_lr[lr]["metrics"].get("top1_acc", 0) >= 0.10)]
        crit_lr[arm_name] = max(stable_lrs) if stable_lrs else None
        # Best LR by top-1
        valid = {lr: r for lr, r in per_lr.items() if not r["nan_seen"]}
        if valid:
            best_lr = max(valid, key=lambda lr: valid[lr]["metrics"]["top1_acc"])
            print(f"  {arm_name:12s}: critical_lr={crit_lr[arm_name]}  best_lr={best_lr} (top1={100*valid[best_lr]['metrics']['top1_acc']:.2f}%)")
        else:
            print(f"  {arm_name:12s}: all cells failed")

    # Verdict: branch-density vs MLP-as-special
    a_crit = crit_lr["14L_MLP"]
    b_crit = crit_lr["14L_noMLP"]
    c_crit = crit_lr["7L_MLP"]
    d_crit = crit_lr["7L_noMLP"]

    print(f"\n  14L+MLP   crit_lr={a_crit}")
    print(f"  14L noMLP crit_lr={b_crit}")
    print(f"  7L  +MLP  crit_lr={c_crit}")
    print(f"  7L  noMLP crit_lr={d_crit}")

    if a_crit and b_crit and c_crit and d_crit:
        # Branch-density prediction: A > B ~ C > D (more branches = more stable)
        # MLP-as-special prediction: (A, C) > (B, D) regardless of depth
        if a_crit > c_crit and c_crit >= b_crit and b_crit > d_crit:
            verdict = (f"PASS_BRANCH: stability orders by branch count: 14L+MLP({a_crit}) > "
                       f"7L+MLP({c_crit}) ~ 14L noMLP({b_crit}) > 7L noMLP({d_crit}). "
                       f"Branch density is the mechanism.")
        elif (a_crit >= c_crit and c_crit > b_crit and c_crit > d_crit and a_crit > b_crit):
            verdict = (f"PASS_M1_PARTIAL: MLP improves stability at both depths. "
                       f"Order: 14L+MLP > 7L+MLP > 14L noMLP > 7L noMLP.")
        else:
            verdict = (f"AMBIGUOUS/MIXED: critical LRs do not cleanly fit either prediction. "
                       f"{a_crit} {b_crit} {c_crit} {d_crit}")
    else:
        verdict = (f"INCOMPLETE: some arms have no stable LR. "
                   f"14L+MLP={a_crit}, 14L noMLP={b_crit}, 7L+MLP={c_crit}, 7L noMLP={d_crit}")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 153, "name": "mlp_depth_factorial",
        "config": {"lr_grid": LR_GRID, "warmup_steps": LR_WARMUP_STEPS,
                    "train_steps": TRAIN_STEPS, "n_train_pool": N_TRAIN, "seed": SEED},
        "results": {arm: {"params_M": v["params_M"],
                          "per_lr": {str(k): vv for k, vv in v["per_lr"].items()}}
                    for arm, v in results.items()},
        "critical_lr": crit_lr,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_153_mlp_depth_factorial.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
