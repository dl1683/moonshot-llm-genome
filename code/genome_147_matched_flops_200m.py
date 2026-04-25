"""
genome_147_matched_flops_200m.py

CODEX Y1 — SCALE EXTENSION TO 200M.

g146 PARTIAL/CLEAN: at 100M, minimal_6L (8000 steps, 53M) beats baseline
(4000 steps, 124M) at matched FLOPs by ~0.8pp top-1 across all metrics
when N_TRAIN=32k.

Now: does the architecture-prior efficiency win persist at 200M? If yes,
the thesis is structural, not local. Same protocol as g146:

Arms (Llama family, N_TRAIN=32k):
  - baseline_200M:    14L+MLP, hidden=1024, ffn=2304 (~209M params)
  - minimal_7L_200M:  7L no-MLP, hidden=1024, ffn=2304 (~80M params)

Matched-FLOPs comparison: baseline 4000 steps, minimal 8000 steps.
Wallclock will be ~similar (baseline ~400s, minimal ~370s).

Pre-stated criteria:
  PASS: minimal_7L (8000 steps) beats baseline (4000 steps) by >=0.5pp
        top-1 on BOTH C4 and OOD across 3 seeds.
        => Architecture-prior efficiency win is SCALE-MONOTONIC (30M -> 100M -> 200M).
  PARTIAL: minimal still better but gap 0.2-0.5pp.
  NEUTRAL: tie within 0.2pp — win disappears at 200M.
  REVERSE: baseline wins clean — win is scale-bounded.

Compute: 3 seeds x 2 arms. baseline ~400s/run, minimal ~370s/run.
Total ~40-45 min on RTX 5090.

Results: results/genome_147_matched_flops_200m.json
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
LR = 3e-4
SEEDS = [42, 7, 13]
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


def train_arm(arm_name, seed, model, train_ids, train_mask, n_steps):
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name} (seed {seed}): total_params={n_total/1e6:.2f}M  steps={n_steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(seed)
    t_arm = time.time()
    model.train()
    log_every = 1000
    n_train = train_ids.size(0)
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"    step={step:5d} loss={loss.item():.3f} ({time.time()-t_arm:.0f}s)")
    return n_total, time.time() - t_arm


def main():
    t0 = time.time()
    print("genome_147: matched-FLOPs at 200M scale (Codex Y1)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print(f"Loading {N_TRAIN} c4 + {N_C4_EVAL} eval + {N_OOD_EVAL} OOD stimuli...")
    pool_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=N_TRAIN + N_C4_EVAL):
        pool_texts.append(rec["text"])
    train_texts = pool_texts[:N_TRAIN]
    c4_eval_texts = pool_texts[N_TRAIN:N_TRAIN + N_C4_EVAL]
    ood_eval_texts = load_wikitext_eval()

    print(f"Tokenizing {N_TRAIN} train sequences...")
    enc_t = tok(train_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc_t["input_ids"]; train_mask = enc_t["attention_mask"]
    enc_c4 = tok(c4_eval_texts, padding=True, truncation=True,
                  max_length=SEQ_LEN, return_tensors="pt")
    c4_eval_ids = enc_c4["input_ids"]; c4_eval_mask = enc_c4["attention_mask"]
    enc_ood = tok(ood_eval_texts, padding=True, truncation=True,
                   max_length=SEQ_LEN, return_tensors="pt")
    ood_eval_ids = enc_ood["input_ids"]; ood_eval_mask = enc_ood["attention_mask"]
    print(f"  train: {train_ids.shape}")

    arms = [
        ("baseline_200M_4k",   dict(hidden=1024, layers=14, heads=16, ffn=2304, no_mlp=False), BASELINE_STEPS),
        ("minimal_7L_200M_8k", dict(hidden=1024, layers=7,  heads=16, ffn=2304, no_mlp=True),  MINIMAL_STEPS),
    ]

    results = {a[0]: {"per_seed": {}, "params_M": None, "wallclock_s": []} for a in arms}

    for arm_name, kw, n_steps in arms:
        for seed in SEEDS:
            print(f"\n=== {arm_name} seed {seed} ===")
            try:
                model = make_llama(actual_vocab, seed=seed, **kw)
            except Exception as e:
                print(f"  build fail: {e}"); continue
            n_total, elapsed = train_arm(arm_name, seed, model, train_ids, train_mask, n_steps)
            print(f"  trained in {elapsed:.0f}s, evaluating...")
            c4_metrics = measure_full(model, c4_eval_ids, c4_eval_mask)
            ood_metrics = measure_full(model, ood_eval_ids, ood_eval_mask)
            print(f"    C4:  NLL={c4_metrics['nll']:.4f} top1={100*c4_metrics['top1_acc']:.2f}% top5={100*c4_metrics['top5_acc']:.2f}%")
            print(f"    OOD: NLL={ood_metrics['nll']:.4f} top1={100*ood_metrics['top1_acc']:.2f}% top5={100*ood_metrics['top5_acc']:.2f}%")
            results[arm_name]["per_seed"][seed] = {"c4": c4_metrics, "ood": ood_metrics}
            results[arm_name]["params_M"] = n_total / 1e6
            results[arm_name]["wallclock_s"].append(elapsed)
            del model; torch.cuda.empty_cache()

    # Analysis
    print(f"\n=== ANALYSIS ===")
    summary = {}
    for arm_name in [a[0] for a in arms]:
        per_seed = results[arm_name]["per_seed"]
        if not per_seed:
            continue
        c4_top1 = [per_seed[s]["c4"]["top1_acc"] for s in SEEDS if s in per_seed]
        ood_top1 = [per_seed[s]["ood"]["top1_acc"] for s in SEEDS if s in per_seed]
        c4_nll = [per_seed[s]["c4"]["nll"] for s in SEEDS if s in per_seed]
        ood_nll = [per_seed[s]["ood"]["nll"] for s in SEEDS if s in per_seed]
        summary[arm_name] = {
            "c4_nll_mean": float(np.mean(c4_nll)),
            "c4_top1_mean": float(np.mean(c4_top1)),
            "c4_top1_std": float(np.std(c4_top1)),
            "ood_nll_mean": float(np.mean(ood_nll)),
            "ood_top1_mean": float(np.mean(ood_top1)),
            "ood_top1_std": float(np.std(ood_top1)),
            "params_M": results[arm_name]["params_M"],
            "wallclock_s_mean": float(np.mean(results[arm_name]["wallclock_s"])),
        }
        s = summary[arm_name]
        print(f"  {arm_name:24s}  C4_NLL={s['c4_nll_mean']:.4f} top1={100*s['c4_top1_mean']:.2f}%+/-{100*s['c4_top1_std']:.3f}")
        print(f"  {' '*24}  OOD_NLL={s['ood_nll_mean']:.4f} top1={100*s['ood_top1_mean']:.2f}%+/-{100*s['ood_top1_std']:.3f}")
        print(f"  {' '*24}  params={s['params_M']:.2f}M  time={s['wallclock_s_mean']:.0f}s")

    if "baseline_200M_4k" in summary and "minimal_7L_200M_8k" in summary:
        bf = summary["baseline_200M_4k"]
        m7 = summary["minimal_7L_200M_8k"]
        c4_nll_gap = m7["c4_nll_mean"] - bf["c4_nll_mean"]
        ood_nll_gap = m7["ood_nll_mean"] - bf["ood_nll_mean"]
        c4_top1_gap_pp = (m7["c4_top1_mean"] - bf["c4_top1_mean"]) * 100
        ood_top1_gap_pp = (m7["ood_top1_mean"] - bf["ood_top1_mean"]) * 100
        time_pct = 100 * m7["wallclock_s_mean"] / bf["wallclock_s_mean"]
        max_std = max(m7["c4_top1_std"], m7["ood_top1_std"]) * 100

        print(f"\n  GAPS (minimal - baseline, positive = minimal better):")
        print(f"    C4 NLL: {-c4_nll_gap:+.4f}")
        print(f"    C4 top1 gap: {c4_top1_gap_pp:+.3f}pp")
        print(f"    OOD NLL: {-ood_nll_gap:+.4f}")
        print(f"    OOD top1 gap: {ood_top1_gap_pp:+.3f}pp")
        print(f"    minimal time {time_pct:.0f}% of baseline")

        if (c4_top1_gap_pp >= 0.5 and ood_top1_gap_pp >= 0.5 and max_std <= 0.5):
            verdict = (f"PASS_200M: minimal_7L still beats baseline at 200M. "
                       f"C4 +{c4_top1_gap_pp:.2f}pp, OOD +{ood_top1_gap_pp:.2f}pp at matched FLOPs. "
                       f"Architecture-prior efficiency win is scale-MONOTONIC (30M -> 100M -> 200M).")
        elif (c4_top1_gap_pp >= 0.2 or ood_top1_gap_pp >= 0.2):
            verdict = (f"PARTIAL_200M: minimal slightly better at 200M, gap 0.2-0.5pp. "
                       f"C4 +{c4_top1_gap_pp:.2f}pp, OOD +{ood_top1_gap_pp:.2f}pp.")
        elif (abs(c4_top1_gap_pp) <= 0.2 and abs(ood_top1_gap_pp) <= 0.2):
            verdict = (f"NEUTRAL_200M: tie at 200M. Architecture-prior win disappears at this scale.")
        else:
            verdict = (f"REVERSE_200M: baseline now wins at 200M. "
                       f"Architecture-prior advantage is SCALE-BOUNDED.")
    else:
        verdict = "FAIL: arms did not complete"

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 147, "name": "matched_flops_200m",
        "config": {"baseline_steps": BASELINE_STEPS, "minimal_steps": MINIMAL_STEPS,
                    "n_train_pool": N_TRAIN, "lr": LR, "batch": BATCH_SIZE, "seeds": SEEDS,
                    "scale_target": "200M"},
        "results": results, "summary": summary, "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_147_matched_flops_200m.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
