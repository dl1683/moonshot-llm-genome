"""
genome_149_hp_robustness_200m.py

CODEX AA2 — HYPERPARAMETER ROBUSTNESS SWEEP AT 200M.

After g141/g146/g147/g148 chain established the architecture-prior efficiency
win is scale-monotonic AND capability-grade, the remaining objection is
"this might be a 3e-4 / wd=0.1 specific result." This experiment closes
that loophole.

Compact grid (asks: "does minimal still win across reasonable HP?"):
  Arms: baseline_200M_4k vs minimal_7L_200M_8k (same as g147)
  LR sweep: {1e-4, 3e-4 (default), 1e-3}
  Weight decay: 0.1 (fixed default)
  Batch size: 8 (fixed default)
  Seeds: [42] (single seed per cell — sweep is HP not seed)

Total: 2 arms × 3 LRs × 1 seed = 6 runs × ~5 min = ~30-40 min.

Pre-stated criteria:
  PASS: minimal still beats baseline by >=0.3pp top-1 on EITHER C4 in-dist
        OR OOD across all 3 LR cells. The win is HP-robust.
  PARTIAL: minimal wins at >=2/3 LR cells. Mostly robust.
  KILL: minimal loses at >=2/3 LR cells. Win was HP-specific to 3e-4.

Compute: ~30-40 min. Run AFTER user resumes from hibernate.

Results: results/genome_149_hp_robustness_200m.json
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
N_C4_EVAL = 200
N_OOD_EVAL = 200
N_TRAIN = 32768
SEED = 42  # single seed for HP sweep
LR_GRID = [1e-4, 3e-4, 1e-3]
BASELINE_STEPS = 4000
MINIMAL_STEPS = 8000


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
    correct_top1, correct_top5 = 0, 0
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
            top5 = sl.topk(5, dim=-1).indices
            correct_top5 += ((top5 == lbl.unsqueeze(-1)).any(dim=-1) & valid).sum().item()
    model.train()
    return {"nll": total_loss / max(total_tokens, 1),
            "top1_acc": correct_top1 / max(total_tokens, 1),
            "top5_acc": correct_top5 / max(total_tokens, 1)}


def load_wikitext_eval(n=N_OOD_EVAL, seed=12345):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    out = []
    for idx in perm:
        text = ds[int(idx)]["text"].strip()
        if len(text) < 200:
            continue
        out.append(text[:1500])
        if len(out) >= n:
            break
    return out


def train_arm(arm_name, lr, model, train_ids, train_mask, n_steps):
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name} (lr={lr}): total_params={n_total/1e6:.2f}M  steps={n_steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(SEED)
    t_arm = time.time()
    model.train()
    log_every = 1000
    n_train = train_ids.size(0)
    nan_seen = False
    for step in range(1, n_steps + 1):
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
        if step % log_every == 0:
            print(f"    step={step:5d} loss={loss.item():.3f} ({time.time()-t_arm:.0f}s)")
    return n_total, time.time() - t_arm, nan_seen


def main():
    t0 = time.time()
    print("genome_149: HP robustness sweep at 200M (Codex AA2)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print(f"Loading {N_TRAIN} c4 train + eval stimuli...")
    pool_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=N_TRAIN + N_C4_EVAL):
        pool_texts.append(rec["text"])
    train_texts = pool_texts[:N_TRAIN]
    c4_eval_texts = pool_texts[N_TRAIN:N_TRAIN + N_C4_EVAL]
    ood_eval_texts = load_wikitext_eval()

    enc_t = tok(train_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc_t["input_ids"]; train_mask = enc_t["attention_mask"]
    enc_c4 = tok(c4_eval_texts, padding=True, truncation=True,
                  max_length=SEQ_LEN, return_tensors="pt")
    c4_eval_ids = enc_c4["input_ids"]; c4_eval_mask = enc_c4["attention_mask"]
    enc_ood = tok(ood_eval_texts, padding=True, truncation=True,
                   max_length=SEQ_LEN, return_tensors="pt")
    ood_eval_ids = enc_ood["input_ids"]; ood_eval_mask = enc_ood["attention_mask"]

    arms = [
        ("baseline_200M_4k",   dict(hidden=1024, layers=14, heads=16, ffn=2304, no_mlp=False), BASELINE_STEPS),
        ("minimal_7L_200M_8k", dict(hidden=1024, layers=7,  heads=16, ffn=2304, no_mlp=True),  MINIMAL_STEPS),
    ]

    results = {}

    for arm_name, kw, n_steps in arms:
        results[arm_name] = {"per_lr": {}, "params_M": None}
        for lr in LR_GRID:
            print(f"\n=== {arm_name}  lr={lr} ===")
            try:
                model = make_llama(actual_vocab, seed=SEED, **kw)
            except Exception as e:
                print(f"  build fail: {e}"); continue
            n_total, elapsed, nan_seen = train_arm(
                arm_name, lr, model, train_ids, train_mask, n_steps,
            )
            print(f"  trained in {elapsed:.0f}s, evaluating... (nan_seen={nan_seen})")
            if nan_seen:
                c4 = {"nll": float("nan"), "top1_acc": float("nan"), "top5_acc": float("nan")}
                ood = {"nll": float("nan"), "top1_acc": float("nan"), "top5_acc": float("nan")}
            else:
                c4 = measure_full(model, c4_eval_ids, c4_eval_mask)
                ood = measure_full(model, ood_eval_ids, ood_eval_mask)
                print(f"    C4:  NLL={c4['nll']:.4f} top1={100*c4['top1_acc']:.2f}%")
                print(f"    OOD: NLL={ood['nll']:.4f} top1={100*ood['top1_acc']:.2f}%")
            results[arm_name]["per_lr"][lr] = {
                "c4": c4, "ood": ood, "wallclock_s": elapsed, "nan_seen": nan_seen,
            }
            results[arm_name]["params_M"] = n_total / 1e6
            del model; torch.cuda.empty_cache()

    # Analysis
    print(f"\n=== ANALYSIS ===")
    sweep = []
    for lr in LR_GRID:
        bf = results["baseline_200M_4k"]["per_lr"].get(lr)
        m7 = results["minimal_7L_200M_8k"]["per_lr"].get(lr)
        if bf is None or m7 is None:
            continue
        if bf.get("nan_seen") or m7.get("nan_seen"):
            sweep.append({"lr": lr, "skipped": True, "reason": "nan in training"})
            continue
        c4_top1_gap_pp = (m7["c4"]["top1_acc"] - bf["c4"]["top1_acc"]) * 100
        ood_top1_gap_pp = (m7["ood"]["top1_acc"] - bf["ood"]["top1_acc"]) * 100
        c4_nll_gap = bf["c4"]["nll"] - m7["c4"]["nll"]
        ood_nll_gap = bf["ood"]["nll"] - m7["ood"]["nll"]
        cell_pass = (c4_top1_gap_pp >= 0.3 or ood_top1_gap_pp >= 0.3)
        sweep.append({
            "lr": lr,
            "baseline_c4_top1": bf["c4"]["top1_acc"],
            "baseline_ood_top1": bf["ood"]["top1_acc"],
            "minimal_c4_top1": m7["c4"]["top1_acc"],
            "minimal_ood_top1": m7["ood"]["top1_acc"],
            "c4_top1_gap_pp": c4_top1_gap_pp,
            "ood_top1_gap_pp": ood_top1_gap_pp,
            "c4_nll_gap_minimal_better": c4_nll_gap,
            "ood_nll_gap_minimal_better": ood_nll_gap,
            "minimal_wins_cell": cell_pass,
        })
        print(f"  lr={lr:.0e}  C4 top1: {100*bf['c4']['top1_acc']:.2f} -> {100*m7['c4']['top1_acc']:.2f} (gap {c4_top1_gap_pp:+.2f}pp)  "
              f"OOD: {100*bf['ood']['top1_acc']:.2f} -> {100*m7['ood']['top1_acc']:.2f} ({ood_top1_gap_pp:+.2f}pp)  "
              f"win={cell_pass}")

    n_wins = sum(1 for c in sweep if c.get("minimal_wins_cell"))
    n_cells = sum(1 for c in sweep if not c.get("skipped"))

    if n_cells == 0:
        verdict = "FAIL: all cells skipped due to NaN"
    elif n_wins == n_cells:
        verdict = (f"PASS: minimal wins all {n_wins}/{n_cells} HP cells. "
                   f"Architecture-prior win is HP-ROBUST, not 3e-4 specific.")
    elif n_wins >= n_cells * 2 / 3:
        verdict = (f"PARTIAL: minimal wins {n_wins}/{n_cells} HP cells. Mostly robust.")
    else:
        verdict = (f"KILL: minimal wins only {n_wins}/{n_cells} HP cells. Win was HP-specific.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 149, "name": "hp_robustness_200m",
        "config": {"baseline_steps": BASELINE_STEPS, "minimal_steps": MINIMAL_STEPS,
                    "n_train_pool": N_TRAIN, "lr_grid": LR_GRID, "seed": SEED,
                    "batch": BATCH_SIZE},
        "results": {arm: {"params_M": v["params_M"],
                          "per_lr": {str(k): vv for k, vv in v["per_lr"].items()}}
                    for arm, v in results.items()},
        "sweep": sweep, "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_149_hp_robustness_200m.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
