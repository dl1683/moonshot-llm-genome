"""
genome_141_minimal_prior_capability.py

CODEX T3 — capability validation of the g140 minimal-prior efficiency win.

g140 PASS: minimal_3L matches baseline NLL within 0.019 nats and is BETTER
OOD by 1.88% perplexity at 70% params, 63% wallclock, 3-seed robust.

Open objection: matched NLL is not matched CAPABILITY. NLL averages over
all tokens; capability is about correctly predicting specific high-stakes
tokens. T3 tests this: does the minimal architecture preserve top-1 and
top-5 next-token accuracy on OOD data?

Protocol: 2 arms x 3 seeds x 4000 steps (same as g140). Eval at end:
  - C4 in-distribution: NLL, top-1 acc, top-5 acc
  - WikiText-103 OOD: NLL, top-1 acc, top-5 acc

Pre-stated criteria:
  PASS: minimal_3L mean(top-1 acc) within 1 percentage point of baseline
        on BOTH C4 and OOD AND mean(top-5 acc) within 1.5 pp on both AND
        per-seed std (top-1) <= 0.5 pp.
        This means: 30% cheaper architecture preserves capability as
        measured by both perplexity and discrete prediction accuracy.
  PARTIAL: gap 1-3 pp on top-1 with reasonable variance.
  KILL: gap > 3 pp top-1, OR std too high — minimal architecture saves
        compute by sacrificing capability, not just compute redundancy.

If PASS: g140's efficiency claim extends to capability. Strongest possible
defendable result without scale-up.

Compute: ~15 min on RTX 5090.

Results: results/genome_141_minimal_prior_capability.json
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
    """Return NLL, top-1 acc, top-5 acc over valid next-token positions."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    correct_top1, correct_top5 = 0, 0
    with torch.no_grad():
        for i in range(0, eval_ids.size(0), BATCH_SIZE):
            ids = eval_ids[i:i+BATCH_SIZE].to("cuda")
            mask = eval_mask[i:i+BATCH_SIZE].to("cuda")
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = out.logits  # (B, T, V)
            sl = logits[:, :-1].contiguous()  # (B, T-1, V)
            lbl = ids[:, 1:].contiguous().clone()  # (B, T-1)
            sm = mask[:, 1:].contiguous()  # (B, T-1)
            valid = (sm != 0)

            # NLL
            lbl_for_loss = lbl.clone()
            lbl_for_loss[~valid] = -100
            loss = F.cross_entropy(
                sl.view(-1, sl.size(-1)), lbl_for_loss.view(-1),
                ignore_index=-100, reduction="sum",
            )
            n = valid.sum().item()
            total_loss += loss.item()
            total_tokens += n

            # Top-1
            preds = sl.argmax(dim=-1)  # (B, T-1)
            correct_top1 += ((preds == lbl) & valid).sum().item()

            # Top-5
            top5 = sl.topk(5, dim=-1).indices  # (B, T-1, 5)
            correct_top5 += ((top5 == lbl.unsqueeze(-1)).any(dim=-1) & valid).sum().item()

    model.train()
    return {
        "nll": total_loss / max(total_tokens, 1),
        "top1_acc": correct_top1 / max(total_tokens, 1),
        "top5_acc": correct_top5 / max(total_tokens, 1),
        "n_tokens": total_tokens,
    }


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

    elapsed = time.time() - t_arm
    return n_total, elapsed


def main():
    t0 = time.time()
    print("genome_141: minimal-prior CAPABILITY validation (Codex T3)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print("Loading c4_clean_v1 train+eval stimuli...")
    pool_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=N_TRAIN + N_C4_EVAL):
        pool_texts.append(rec["text"])
    train_texts = pool_texts[:N_TRAIN]
    c4_eval_texts = pool_texts[N_TRAIN:N_TRAIN + N_C4_EVAL]

    print("Loading wikitext-103 OOD eval...")
    ood_eval_texts = load_wikitext_eval(n=N_OOD_EVAL)
    print(f"  C4 train={len(train_texts)}, C4 eval={len(c4_eval_texts)}, OOD={len(ood_eval_texts)}")

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
        ("baseline_full", dict(hidden=384, layers=6, heads=6, ffn=1024, no_mlp=False)),
        ("minimal_3L",    dict(hidden=384, layers=3, heads=6, ffn=1024, no_mlp=True)),
    ]

    results = {arm[0]: {"per_seed": {}, "params_M": None, "wallclock_s": []} for arm in arms}

    for arm_name, kw in arms:
        for seed in SEEDS:
            print(f"\n=== {arm_name} seed {seed} ===")
            model = make_llama(actual_vocab, seed=seed, **kw)
            n_total, elapsed = train_arm(arm_name, seed, model, train_ids, train_mask)
            print(f"  trained in {elapsed:.0f}s, evaluating...")

            c4_metrics = measure_full(model, c4_eval_ids, c4_eval_mask)
            ood_metrics = measure_full(model, ood_eval_ids, ood_eval_mask)
            print(f"    C4:  NLL={c4_metrics['nll']:.4f}  top1={100*c4_metrics['top1_acc']:.2f}%  top5={100*c4_metrics['top5_acc']:.2f}%")
            print(f"    OOD: NLL={ood_metrics['nll']:.4f}  top1={100*ood_metrics['top1_acc']:.2f}%  top5={100*ood_metrics['top5_acc']:.2f}%")

            results[arm_name]["per_seed"][seed] = {
                "c4": c4_metrics, "ood": ood_metrics,
            }
            results[arm_name]["params_M"] = n_total / 1e6
            results[arm_name]["wallclock_s"].append(elapsed)
            del model
            torch.cuda.empty_cache()

    # === ANALYSIS ===
    print(f"\n=== ANALYSIS ===")
    summary = {}
    for arm_name in ["baseline_full", "minimal_3L"]:
        per_seed = results[arm_name]["per_seed"]
        c4_top1 = [per_seed[s]["c4"]["top1_acc"] for s in SEEDS]
        c4_top5 = [per_seed[s]["c4"]["top5_acc"] for s in SEEDS]
        c4_nll = [per_seed[s]["c4"]["nll"] for s in SEEDS]
        ood_top1 = [per_seed[s]["ood"]["top1_acc"] for s in SEEDS]
        ood_top5 = [per_seed[s]["ood"]["top5_acc"] for s in SEEDS]
        ood_nll = [per_seed[s]["ood"]["nll"] for s in SEEDS]
        summary[arm_name] = {
            "c4_top1_mean": float(np.mean(c4_top1)),
            "c4_top1_std": float(np.std(c4_top1)),
            "c4_top5_mean": float(np.mean(c4_top5)),
            "c4_nll_mean": float(np.mean(c4_nll)),
            "ood_top1_mean": float(np.mean(ood_top1)),
            "ood_top1_std": float(np.std(ood_top1)),
            "ood_top5_mean": float(np.mean(ood_top5)),
            "ood_nll_mean": float(np.mean(ood_nll)),
            "ood_ppl_mean": float(np.exp(np.mean(ood_nll))),
            "params_M": results[arm_name]["params_M"],
            "wallclock_s_mean": float(np.mean(results[arm_name]["wallclock_s"])),
        }
        s = summary[arm_name]
        print(f"  {arm_name:18s}")
        print(f"    C4:  NLL={s['c4_nll_mean']:.4f}  top1={100*s['c4_top1_mean']:.2f}%+/-{100*s['c4_top1_std']:.3f}  top5={100*s['c4_top5_mean']:.2f}%")
        print(f"    OOD: NLL={s['ood_nll_mean']:.4f}  top1={100*s['ood_top1_mean']:.2f}%+/-{100*s['ood_top1_std']:.3f}  top5={100*s['ood_top5_mean']:.2f}%")
        print(f"    params={s['params_M']:.2f}M, time={s['wallclock_s_mean']:.0f}s")

    bf = summary["baseline_full"]
    m3 = summary["minimal_3L"]
    c4_top1_gap_pp = (bf["c4_top1_mean"] - m3["c4_top1_mean"]) * 100
    c4_top5_gap_pp = (bf["c4_top5_mean"] - m3["c4_top5_mean"]) * 100
    ood_top1_gap_pp = (bf["ood_top1_mean"] - m3["ood_top1_mean"]) * 100
    ood_top5_gap_pp = (bf["ood_top5_mean"] - m3["ood_top5_mean"]) * 100

    print(f"\n  GAPS (baseline - minimal_3L, pp):")
    print(f"    C4 top1:  {c4_top1_gap_pp:+.3f} pp   C4 top5:  {c4_top5_gap_pp:+.3f} pp")
    print(f"    OOD top1: {ood_top1_gap_pp:+.3f} pp   OOD top5: {ood_top5_gap_pp:+.3f} pp")

    max_top1_std = max(m3["c4_top1_std"], m3["ood_top1_std"]) * 100

    if (abs(c4_top1_gap_pp) <= 1.0 and abs(ood_top1_gap_pp) <= 1.0 and
        abs(c4_top5_gap_pp) <= 1.5 and abs(ood_top5_gap_pp) <= 1.5 and
        max_top1_std <= 0.5):
        verdict = (f"PASS: minimal_3L matches capability within 1pp top-1 ({c4_top1_gap_pp:+.2f}/"
                   f"{ood_top1_gap_pp:+.2f}) and 1.5pp top-5 across BOTH C4 and OOD, "
                   f"std {max_top1_std:.2f}pp. Efficiency win extends to CAPABILITY, not just NLL.")
    elif (abs(c4_top1_gap_pp) <= 3.0 and abs(ood_top1_gap_pp) <= 3.0):
        verdict = (f"PARTIAL: top-1 gap 1-3pp ({c4_top1_gap_pp:+.2f}/{ood_top1_gap_pp:+.2f}). "
                   f"Real but not full parity.")
    else:
        verdict = (f"KILL: top-1 gap > 3pp on at least one metric. Minimal architecture "
                   f"saves compute by sacrificing capability, not redundancy.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 141, "name": "minimal_prior_capability",
        "config": {"train_steps": TRAIN_STEPS, "lr": LR, "batch": BATCH_SIZE,
                    "seeds": SEEDS, "n_c4_eval": N_C4_EVAL, "n_ood_eval": N_OOD_EVAL},
        "results": results,
        "summary": summary,
        "deltas_pp": {
            "c4_top1": c4_top1_gap_pp, "c4_top5": c4_top5_gap_pp,
            "ood_top1": ood_top1_gap_pp, "ood_top5": ood_top5_gap_pp,
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_141_minimal_prior_capability.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
