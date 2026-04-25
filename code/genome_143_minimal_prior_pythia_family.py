"""
genome_143_minimal_prior_pythia_family.py

CODEX U3 — CROSS-FAMILY VALIDATION OF MINIMAL-PRIOR EFFICIENCY.

g138-g142 established the win on Llama (30M, 4000 steps). Easy dismissal:
"Llama-specific trick." This experiment closes that loophole by running
the same protocol on Pythia GPT-NeoX architecture family.

Pythia differs from Llama in:
  - Positional encoding: learned absolute (Pythia) vs RoPE (Llama)
  - Normalization: LayerNorm (Pythia) vs RMSNorm (Llama)
  - Activation: GELU (Pythia) vs SwiGLU (Llama)
  - Bias terms: present (Pythia) vs none (Llama)
  - Attention: separate Q/K/V projections (Pythia GPT-NeoX dense_attn) vs
    fused QKV (Llama)

Arms (3 seeds each):
  - pythia_baseline_full: 6L + MLP, hidden=384 (parallel to baseline_full)
  - pythia_minimal_3L:    3L + no MLP, hidden=384 (parallel to minimal_3L)

Same protocol as g141:
  4000 train steps, lr=3e-4, batch=8, c4_clean_v1 train + C4 in-dist eval
  + WikiText-103 OOD eval. Metrics: NLL, top-1, top-5.

Pre-stated criteria:
  PASS: pythia_minimal_3L matches pythia_baseline_full within 1pp top-1
        AND 0.05 NLL on BOTH C4 and OOD with std <=0.5pp.
        => Architecture-prior efficiency win is FAMILY-INDEPENDENT.
  PARTIAL: gap 1-3pp top-1.
  KILL: gap >3pp — Llama-specific architectural detail (e.g., RoPE+RMSNorm)
        was carrying the prior.

Compute: 2 arms × 3 seeds × 4000 steps + capability eval ≈ 12-15 min.

Results: results/genome_143_minimal_prior_pythia_family.json
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
TRAIN_STEPS = 4000
SEEDS = [42, 7, 13]
N_C4_EVAL = 200
N_OOD_EVAL = 200
N_TRAIN = 4000


class ZeroMLP(nn.Module):
    def forward(self, *args, **kwargs):
        # Pythia MLP signature is forward(hidden_states); Llama is forward(x).
        # Both call positionally. Take first arg.
        x = args[0]
        return torch.zeros_like(x)


def make_pythia_neox(vocab_size, hidden, layers, heads, ffn, no_mlp=False, seed=42):
    """Build a GPT-NeoX-style model (matches Pythia architecture family)."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    cfg = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=ffn,
        max_position_embeddings=SEQ_LEN + 64,
        layer_norm_eps=1e-5,
        tie_word_embeddings=True,
        use_parallel_residual=True,  # Pythia default
        rotary_pct=0.25,  # Pythia default
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    model = GPTNeoXForCausalLM(cfg).to("cuda").to(torch.bfloat16)
    if no_mlp:
        for layer in model.gpt_neox.layers:
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


def train_arm(arm_name, seed, model, train_ids, train_mask):
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name} (seed {seed}): total_params={n_total/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(seed)
    t_arm = time.time()
    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        idx = rng.integers(0, train_ids.size(0), size=BATCH_SIZE)
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
    return n_total, time.time() - t_arm


def main():
    t0 = time.time()
    print("genome_143: cross-family validation on Pythia GPT-NeoX (Codex U3)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print("Loading c4 + ood stimuli...")
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
        ("pythia_baseline_full", dict(hidden=384, layers=6, heads=6, ffn=1024, no_mlp=False)),
        ("pythia_minimal_3L",    dict(hidden=384, layers=3, heads=6, ffn=1024, no_mlp=True)),
    ]

    results = {a[0]: {"per_seed": {}, "params_M": None, "wallclock_s": []} for a in arms}

    for arm_name, kw in arms:
        for seed in SEEDS:
            print(f"\n=== {arm_name} seed {seed} ===")
            try:
                model = make_pythia_neox(actual_vocab, seed=seed, **kw)
            except Exception as e:
                print(f"  build fail: {e}"); continue
            n_total, elapsed = train_arm(arm_name, seed, model, train_ids, train_mask)
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

    if "pythia_baseline_full" in summary and "pythia_minimal_3L" in summary:
        bf = summary["pythia_baseline_full"]
        m3 = summary["pythia_minimal_3L"]
        c4_nll_gap = m3["c4_nll_mean"] - bf["c4_nll_mean"]
        ood_nll_gap = m3["ood_nll_mean"] - bf["ood_nll_mean"]
        c4_top1_gap_pp = (bf["c4_top1_mean"] - m3["c4_top1_mean"]) * 100
        ood_top1_gap_pp = (bf["ood_top1_mean"] - m3["ood_top1_mean"]) * 100
        params_pct = 100 * m3["params_M"] / bf["params_M"]
        time_pct = 100 * m3["wallclock_s_mean"] / bf["wallclock_s_mean"]
        max_std = max(m3["c4_top1_std"], m3["ood_top1_std"]) * 100

        print(f"\n  GAPS (baseline - minimal):")
        print(f"    C4 NLL gap {c4_nll_gap:+.4f}  OOD NLL gap {ood_nll_gap:+.4f}")
        print(f"    C4 top1 gap {c4_top1_gap_pp:+.3f}pp  OOD top1 gap {ood_top1_gap_pp:+.3f}pp")
        print(f"    params {params_pct:.0f}%  time {time_pct:.0f}%  max_std {max_std:.3f}pp")

        if (c4_nll_gap <= 0.05 and abs(ood_nll_gap) <= 0.05 and
            abs(c4_top1_gap_pp) <= 1.0 and abs(ood_top1_gap_pp) <= 1.0 and
            max_std <= 0.5):
            verdict = (f"PASS: pythia_minimal_3L matches pythia_baseline within 1pp top-1 / 0.05 NLL "
                       f"on BOTH C4 and OOD at {params_pct:.0f}% params, {time_pct:.0f}% time. "
                       f"Architecture-prior efficiency win is FAMILY-INDEPENDENT (Llama + GPT-NeoX).")
        elif (abs(c4_top1_gap_pp) <= 3.0 and abs(ood_top1_gap_pp) <= 3.0):
            verdict = (f"PARTIAL: gap 1-3pp top-1 ({c4_top1_gap_pp:+.2f}/{ood_top1_gap_pp:+.2f}). "
                       f"Cross-family efficiency win exists but tighter than Llama.")
        else:
            verdict = (f"KILL: top-1 gap > 3pp on at least one metric. "
                       f"Llama-specific architectural detail was carrying the prior; "
                       f"the win does NOT generalize to Pythia GPT-NeoX family.")
    else:
        verdict = "FAIL: arms didn t complete"

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 143, "name": "minimal_prior_pythia_family",
        "config": {"train_steps": TRAIN_STEPS, "lr": LR, "batch": BATCH_SIZE,
                    "seeds": SEEDS, "architecture_family": "GPT-NeoX (Pythia)"},
        "results": results, "summary": summary, "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_143_minimal_prior_pythia_family.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
