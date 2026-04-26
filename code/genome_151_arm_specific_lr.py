"""
genome_151_arm_specific_lr.py

CODEX MECHANISTIC INSIGHT (parallel consult during g150):

g149 showed minimal improved +1.0pp going lr=1e-4 -> 3e-4 vs baseline's
+0.3pp, suggesting they may have DIFFERENT optimal LRs. Theoretical
support: removing MLP changes Hessian curvature, gradient noise, and
residual Jacobians — different stability window is plausible.

If minimal's optimum is higher than baseline's (3e-4), our previous
matched-LR results UNDERESTIMATE the architecture-prior advantage.

PROTOCOL — arm-specific LR sweep at 200M:

Centered around 3e-4 default, both arms get the same fine grid:
  lr ∈ {2e-4, 3e-4, 4e-4, 6e-4}

  baseline_200M_4k × 4 LRs = 4 runs
  minimal_7L_200M_8k × 4 LRs = 4 runs

  Single seed (42, matching g149 sweep style).

  All with linear LR warmup (200 steps) so high-LR cells don't break.

Pre-stated criteria:
  PASS: minimal s best LR cell beats baseline's best LR cell by >=0.3pp
        top-1 on BOTH C4 and OOD. The architecture-prior advantage is
        UNDERESTIMATED at matched-LR — minimal has its own (likely
        higher) optimum that produces a bigger win.
  PARTIAL: minimal's best matches or slightly exceeds baseline's best.
  KILL: baseline's best beats minimal's best — even with arm-specific
        tuning, minimal loses.

Compute: 8 runs × ~5 min = ~40 min.

If PASS: this is a STRONGER claim than g141/g146/g147. The architecture-
prior advantage we measured was a lower bound; the true advantage is
larger when each arm uses its own optimal LR.

Results: results/genome_151_arm_specific_lr.json
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
N_OOD_EVAL = 200
N_TRAIN = 32768
BASELINE_STEPS = 4000
MINIMAL_STEPS = 8000
LR_WARMUP_STEPS = 200
LR_GRID = [2e-4, 3e-4, 4e-4, 6e-4]


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


def warmup_lr(step, target_lr, warmup_steps):
    if step < warmup_steps:
        return target_lr * (step + 1) / warmup_steps
    return target_lr


def train_arm(arm_name, lr_target, model, train_ids, train_mask, n_steps):
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name} (lr={lr_target}): params={n_total/1e6:.2f}M steps={n_steps} warmup={LR_WARMUP_STEPS}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr_target, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(SEED)
    t_arm = time.time()
    model.train()
    n_train = train_ids.size(0)
    nan_seen = False
    log_every = 1000
    for step in range(1, n_steps + 1):
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
        if step % log_every == 0:
            print(f"    step={step:5d} loss={loss.item():.3f} lr={current_lr:.2e} ({time.time()-t_arm:.0f}s)")
    return n_total, time.time() - t_arm, nan_seen


def main():
    t0 = time.time()
    print("genome_151: arm-specific LR sweep at 200M (Codex mechanistic insight)")
    print(f"  LR_GRID={LR_GRID}, both arms each LR with warmup")

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
        ("baseline_200M_4k", dict(hidden=1024, layers=14, heads=16, ffn=2304, no_mlp=False), BASELINE_STEPS),
        ("minimal_7L_200M_8k", dict(hidden=1024, layers=7, heads=16, ffn=2304, no_mlp=True), MINIMAL_STEPS),
    ]

    results = {}
    for arm_name, kw, n_steps in arms:
        results[arm_name] = {"per_lr": {}, "params_M": None}
        for lr in LR_GRID:
            print(f"\n=== {arm_name} lr={lr} ===")
            try:
                model = make_llama(actual_vocab, seed=SEED, **kw)
            except Exception as e:
                print(f"  build fail: {e}"); continue
            n_total, elapsed, nan_seen = train_arm(arm_name, lr, model, train_ids, train_mask, n_steps)
            print(f"  trained in {elapsed:.0f}s nan={nan_seen}")
            if nan_seen:
                c4 = {"nll": float("nan"), "top1_acc": float("nan"), "top5_acc": float("nan")}
                ood = {"nll": float("nan"), "top1_acc": float("nan"), "top5_acc": float("nan")}
            else:
                c4 = measure_full(model, c4_eval_ids, c4_eval_mask)
                ood = measure_full(model, ood_eval_ids, ood_eval_mask)
                print(f"    C4 top1={100*c4['top1_acc']:.2f}%  OOD top1={100*ood['top1_acc']:.2f}%")
            results[arm_name]["per_lr"][lr] = {"c4": c4, "ood": ood, "wallclock_s": elapsed, "nan_seen": nan_seen}
            results[arm_name]["params_M"] = n_total / 1e6
            del model; torch.cuda.empty_cache()

    # Find each arm's best LR (highest C4 top-1)
    print(f"\n=== ANALYSIS ===")
    arm_best = {}
    for arm_name in [a[0] for a in arms]:
        per_lr = results[arm_name]["per_lr"]
        valid = {lr: r for lr, r in per_lr.items() if not r.get("nan_seen") and np.isfinite(r["c4"]["top1_acc"])}
        if not valid:
            arm_best[arm_name] = None
            continue
        best_lr = max(valid, key=lambda lr: valid[lr]["c4"]["top1_acc"])
        arm_best[arm_name] = {"lr": best_lr, "c4_top1": valid[best_lr]["c4"]["top1_acc"],
                                "ood_top1": valid[best_lr]["ood"]["top1_acc"],
                                "c4_nll": valid[best_lr]["c4"]["nll"],
                                "ood_nll": valid[best_lr]["ood"]["nll"]}
        print(f"  {arm_name}: best lr={best_lr}, C4 top1={100*arm_best[arm_name]['c4_top1']:.2f}%, OOD top1={100*arm_best[arm_name]['ood_top1']:.2f}%")
        # Show full sweep
        for lr in sorted(valid):
            print(f"    lr={lr:.0e}: C4 {100*valid[lr]['c4']['top1_acc']:.2f}% OOD {100*valid[lr]['ood']['top1_acc']:.2f}%")

    if arm_best.get("baseline_200M_4k") and arm_best.get("minimal_7L_200M_8k"):
        bf = arm_best["baseline_200M_4k"]
        m7 = arm_best["minimal_7L_200M_8k"]
        c4_gap_pp = (m7["c4_top1"] - bf["c4_top1"]) * 100
        ood_gap_pp = (m7["ood_top1"] - bf["ood_top1"]) * 100
        print(f"\n  best-vs-best C4 gap: {c4_gap_pp:+.2f}pp")
        print(f"  best-vs-best OOD gap: {ood_gap_pp:+.2f}pp")
        print(f"  baseline best LR: {bf['lr']}, minimal best LR: {m7['lr']}")
        if c4_gap_pp >= 0.3 and ood_gap_pp >= 0.3:
            verdict = (f"PASS: minimal best (lr={m7['lr']}) beats baseline best (lr={bf['lr']}) "
                       f"by C4 +{c4_gap_pp:.2f}pp, OOD +{ood_gap_pp:.2f}pp. The architecture-prior "
                       f"advantage is UNDERESTIMATED at matched-LR. arm-specific tuning gives "
                       f"a bigger win.")
        elif c4_gap_pp >= 0.0 and ood_gap_pp >= 0.0:
            verdict = (f"PARTIAL: minimal best matches/slightly beats baseline best. "
                       f"C4 {c4_gap_pp:+.2f}pp, OOD {ood_gap_pp:+.2f}pp.")
        else:
            verdict = (f"KILL: baseline best beats minimal best. C4 {c4_gap_pp:+.2f}pp, "
                       f"OOD {ood_gap_pp:+.2f}pp. Even with arm-specific LR, minimal loses.")
    else:
        verdict = "FAIL: arms did not complete"

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 151, "name": "arm_specific_lr",
        "config": {"lr_grid": LR_GRID, "warmup_steps": LR_WARMUP_STEPS,
                    "baseline_steps": BASELINE_STEPS, "minimal_steps": MINIMAL_STEPS,
                    "n_train_pool": N_TRAIN, "seed": SEED},
        "results": {arm: {"params_M": v["params_M"],
                          "per_lr": {str(k): vv for k, vv in v["per_lr"].items()}}
                    for arm, v in results.items()},
        "arm_best": arm_best, "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_151_arm_specific_lr.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
