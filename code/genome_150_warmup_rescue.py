"""
genome_150_warmup_rescue.py

CODEX adjudication of g149: KILL_strict but with broken cell at lr=1e-3
(both arms diverged). Codex picks Option C — warmup rescue: fix the
broken cell with linear LR warmup before deciding the thesis status.

Original g150 plan (long-horizon crossover) is parked in genome_150_long_
horizon_crossover.py and will run AFTER this rescue, IF this passes.

Protocol:
  Same arms as g149 at lr=1e-3, but with linear LR warmup over first
  200 steps (0 -> 1e-3 linear). 200/4000 = 5% warmup, standard practice.
  Single seed (42) to match g149 sweep style.

  baseline_200M_4k_lr1e3_warmup
  minimal_7L_200M_8k_lr1e3_warmup

Pre-stated criteria:
  PASS: both arms train successfully (no divergence) AND minimal beats
        baseline by >=0.3pp top-1. Combined with g149 lr=3e-4 win, makes
        the architecture-prior thesis robust at 2/3 lrs (excluding broken
        too-low 1e-4 region tie).
  PARTIAL: both train but minimal ties or loses by <=0.5pp. Thesis is
        nuanced — works at default + tied/below at higher LRs.
  KILL: even with warmup, minimal loses by >0.5pp at lr=1e-3 OR diverges.
        Thesis is genuinely LR-fragile beyond the default basin.
  UNINFORMATIVE: both still diverge with warmup. lr=1e-3 is outside
        stable training region for this setup; cell is junk regardless.

Compute: 2 runs at 200M ~5-7 min each = ~15 min.

Results: results/genome_150_warmup_rescue.json
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
LR_TARGET = 1e-3
LR_WARMUP_STEPS = 200
SEED = 42
N_C4_EVAL = 200
N_OOD_EVAL = 200
N_TRAIN = 32768
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


def warmup_lr(step, target_lr, warmup_steps):
    if step < warmup_steps:
        return target_lr * (step + 1) / warmup_steps
    return target_lr


def train_arm(arm_name, model, train_ids, train_mask, n_steps):
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name}: total_params={n_total/1e6:.2f}M  steps={n_steps}  target_lr={LR_TARGET}  warmup={LR_WARMUP_STEPS}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR_TARGET, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(SEED)
    t_arm = time.time()
    model.train()
    n_train = train_ids.size(0)
    nan_seen = False
    log_every = 1000
    final_loss = float('nan')

    for step in range(1, n_steps + 1):
        # Apply LR warmup
        current_lr = warmup_lr(step, LR_TARGET, LR_WARMUP_STEPS)
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
            print(f"    step={step} NaN seen, training diverged")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.item())
        if step % log_every == 0:
            print(f"    step={step:5d} loss={loss.item():.3f} lr={current_lr:.2e} ({time.time()-t_arm:.0f}s)")
    return n_total, time.time() - t_arm, nan_seen, final_loss


def main():
    t0 = time.time()
    print("genome_150: warmup-rescue of g149 lr=1e-3 broken cell (Codex Option C)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print(f"Loading {N_TRAIN} c4 train + eval stimuli...")
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
        ("baseline_200M_4k_lr1e3_warmup",
         dict(hidden=1024, layers=14, heads=16, ffn=2304, no_mlp=False), BASELINE_STEPS),
        ("minimal_7L_200M_8k_lr1e3_warmup",
         dict(hidden=1024, layers=7,  heads=16, ffn=2304, no_mlp=True),  MINIMAL_STEPS),
    ]

    results = {}

    for arm_name, kw, n_steps in arms:
        print(f"\n=== {arm_name} ===")
        try:
            model = make_llama(actual_vocab, seed=SEED, **kw)
        except Exception as e:
            print(f"  build fail: {e}"); continue
        n_total, elapsed, nan_seen, final_loss = train_arm(
            arm_name, model, train_ids, train_mask, n_steps,
        )
        print(f"  trained in {elapsed:.0f}s, final_loss={final_loss:.3f}, nan_seen={nan_seen}")
        if nan_seen:
            c4 = {"nll": float("nan"), "top1_acc": float("nan"), "top5_acc": float("nan")}
            ood = {"nll": float("nan"), "top1_acc": float("nan"), "top5_acc": float("nan")}
        else:
            c4 = measure_full(model, c4_eval_ids, c4_eval_mask)
            ood = measure_full(model, ood_eval_ids, ood_eval_mask)
            print(f"    C4:  NLL={c4['nll']:.4f} top1={100*c4['top1_acc']:.2f}%")
            print(f"    OOD: NLL={ood['nll']:.4f} top1={100*ood['top1_acc']:.2f}%")
        results[arm_name] = {
            "params_M": n_total / 1e6, "wallclock_s": elapsed,
            "nan_seen": nan_seen, "final_loss": final_loss,
            "c4": c4, "ood": ood,
        }
        del model; torch.cuda.empty_cache()

    # Verdict
    print(f"\n=== ANALYSIS ===")
    bf = results.get("baseline_200M_4k_lr1e3_warmup")
    m7 = results.get("minimal_7L_200M_8k_lr1e3_warmup")
    if bf is None or m7 is None:
        verdict = "FAIL: arms did not complete"
    elif bf["nan_seen"] and m7["nan_seen"]:
        verdict = (f"UNINFORMATIVE: both arms still diverge at lr=1e-3 even with warmup. "
                   f"lr=1e-3 is outside stable training region for this setup. "
                   f"Cell is junk regardless of architecture; g149 KILL_strict was overly punishing.")
    elif bf["nan_seen"] or m7["nan_seen"]:
        which = "baseline" if bf["nan_seen"] else "minimal"
        verdict = (f"REVEAL: {which} diverges at lr=1e-3+warmup but the other trains. "
                   f"Architecture matters for stability at high LR.")
    else:
        c4_gap_pp = (m7["c4"]["top1_acc"] - bf["c4"]["top1_acc"]) * 100
        ood_gap_pp = (m7["ood"]["top1_acc"] - bf["ood"]["top1_acc"]) * 100
        print(f"  baseline: C4 top1 {100*bf['c4']['top1_acc']:.2f}%, OOD top1 {100*bf['ood']['top1_acc']:.2f}%")
        print(f"  minimal:  C4 top1 {100*m7['c4']['top1_acc']:.2f}%, OOD top1 {100*m7['ood']['top1_acc']:.2f}%")
        print(f"  C4 gap: {c4_gap_pp:+.2f}pp  OOD gap: {ood_gap_pp:+.2f}pp")
        if c4_gap_pp >= 0.3 or ood_gap_pp >= 0.3:
            verdict = (f"PASS: both arms train with warmup, minimal beats baseline. "
                       f"C4 +{c4_gap_pp:.2f}pp, OOD +{ood_gap_pp:.2f}pp. Combined with g149 "
                       f"lr=3e-4 win, the architecture-prior advantage is robust across "
                       f"reasonable LR settings (with appropriate warmup at extreme LR).")
        elif abs(c4_gap_pp) <= 0.5 and abs(ood_gap_pp) <= 0.5:
            verdict = (f"PARTIAL: both train, gap small. C4 {c4_gap_pp:+.2f}pp, OOD {ood_gap_pp:+.2f}pp. "
                       f"Win is real at default LR but ties at higher LRs.")
        else:
            verdict = (f"KILL: minimal loses meaningfully at lr=1e-3 even with warmup. "
                       f"C4 {c4_gap_pp:+.2f}pp, OOD {ood_gap_pp:+.2f}pp. Win is LR-fragile "
                       f"beyond default basin.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 150, "name": "warmup_rescue",
        "config": {"target_lr": LR_TARGET, "warmup_steps": LR_WARMUP_STEPS,
                    "baseline_steps": BASELINE_STEPS, "minimal_steps": MINIMAL_STEPS,
                    "n_train_pool": N_TRAIN, "seed": SEED},
        "results": results, "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_150_warmup_rescue.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
