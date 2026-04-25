"""
genome_140_minimal_prior_multiseed_ood.py

CODEX S1 — MULTI-SEED REPRODUCIBILITY + OOD CAPABILITY for the g139 win.

genome_139 single-seed PASS: minimal_3L (3L, no MLP, hidden=384) matches
baseline NLL within 0.027 nats at 70% params, 68% wallclock. Could be
lucky single-seed effect or only-on-distribution.

This experiment lifts that to a defendable claim:
  - 3 matched seeds (42, 7, 13)
  - Primary: C4 val NLL gap (in-distribution)
  - Secondary: WikiText-103 perplexity gap (OOD distribution)
  - Report mean, std, wallclock per arm

Pre-stated criteria (locked):
  PASS: minimal_3L mean(C4 NLL) gap <= 0.05 nats vs baseline_full AND
        mean(wikitext PPL) gap within 5% of baseline AND
        params <= 75% of baseline.
  PARTIAL: gap <= 0.10 on both metrics, params reduction holds.
  KILL: gap > 0.10 on either metric, OR per-seed std > 0.10
        (single-seed g139 result was a fluke).

Compute: 2 arms x 3 seeds x 4000 steps = ~10-15 min.

Results: results/genome_140_minimal_prior_multiseed_ood.json
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
EVAL_AT = [0, 32, 128, 512, 1000, 2000, 4000]
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


def measure_eval_nll(model, eval_ids, eval_mask):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, eval_ids.size(0), BATCH_SIZE):
            ids = eval_ids[i:i+BATCH_SIZE].to("cuda")
            mask = eval_mask[i:i+BATCH_SIZE].to("cuda")
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = out.logits
            sl = logits[:, :-1].contiguous()
            lbl = ids[:, 1:].contiguous().clone()
            sm = mask[:, 1:].contiguous()
            lbl[sm == 0] = -100
            loss = F.cross_entropy(
                sl.view(-1, sl.size(-1)), lbl.view(-1),
                ignore_index=-100, reduction="sum",
            )
            n = (sm != 0).sum().item()
            total_loss += loss.item()
            total_tokens += n
    model.train()
    return total_loss / max(total_tokens, 1)


def load_wikitext_eval(tok, n=N_OOD_EVAL, seed=12345):
    """Load wikitext-103 as OOD eval set (different from C4 training data)."""
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


def train_arm(arm_name, model, train_ids, train_mask, c4_eval_ids, c4_eval_mask,
              ood_eval_ids, ood_eval_mask, seed):
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name} (seed {seed}): total_params={n_total/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(seed)
    rows = []
    t_arm = time.time()

    nll0 = measure_eval_nll(model, c4_eval_ids, c4_eval_mask)
    rows.append({"step": 0, "c4_nll": nll0})
    print(f"    step=0  C4_NLL={nll0:.3f}")
    model.train()
    next_idx = 1

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
        if next_idx < len(EVAL_AT) and step == EVAL_AT[next_idx]:
            nll = measure_eval_nll(model, c4_eval_ids, c4_eval_mask)
            rows.append({"step": step, "c4_nll": nll, "loss": float(loss.item())})
            print(f"    step={step:5d}  C4_NLL={nll:.3f}  loss={loss.item():.3f}  ({time.time()-t_arm:.0f}s)")
            model.train()
            next_idx += 1

    # Final OOD eval
    ood_nll = measure_eval_nll(model, ood_eval_ids, ood_eval_mask)
    elapsed = time.time() - t_arm
    return rows, n_total, elapsed, ood_nll


def main():
    t0 = time.time()
    print("genome_140: minimal-prior multi-seed + OOD reproducibility (Codex S1)")

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
    ood_eval_texts = load_wikitext_eval(tok, n=N_OOD_EVAL)
    print(f"  C4 train={len(train_texts)}, C4 eval={len(c4_eval_texts)}, OOD eval={len(ood_eval_texts)}")

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
        ("baseline_full",  dict(hidden=384, layers=6, heads=6, ffn=1024, no_mlp=False)),
        ("minimal_3L",     dict(hidden=384, layers=3, heads=6, ffn=1024, no_mlp=True)),
    ]

    results = {arm[0]: {"per_seed": {}, "params_M": None, "wallclock_s": []} for arm in arms}

    for arm_name, kw in arms:
        for seed in SEEDS:
            print(f"\n=== {arm_name} seed {seed} ===")
            model = make_llama(actual_vocab, seed=seed, **kw)
            rows, n_total, elapsed, ood_nll = train_arm(
                arm_name, model, train_ids, train_mask,
                c4_eval_ids, c4_eval_mask, ood_eval_ids, ood_eval_mask, seed,
            )
            print(f"    OOD wikitext NLL={ood_nll:.3f}  PPL={float(np.exp(ood_nll)):.2f}")
            results[arm_name]["per_seed"][seed] = {
                "rows": rows, "ood_nll": ood_nll,
                "ood_ppl": float(np.exp(ood_nll)),
                "final_c4_nll": rows[-1]["c4_nll"],
            }
            results[arm_name]["params_M"] = n_total / 1e6
            results[arm_name]["wallclock_s"].append(elapsed)
            del model
            torch.cuda.empty_cache()

    # === ANALYSIS ===
    print(f"\n=== ANALYSIS ===")
    summary = {}
    for arm_name in ["baseline_full", "minimal_3L"]:
        c4_finals = [results[arm_name]["per_seed"][s]["final_c4_nll"] for s in SEEDS]
        ood_nlls = [results[arm_name]["per_seed"][s]["ood_nll"] for s in SEEDS]
        ood_ppls = [results[arm_name]["per_seed"][s]["ood_ppl"] for s in SEEDS]
        summary[arm_name] = {
            "c4_final_per_seed": [float(x) for x in c4_finals],
            "c4_final_mean": float(np.mean(c4_finals)),
            "c4_final_std": float(np.std(c4_finals)),
            "ood_nll_per_seed": [float(x) for x in ood_nlls],
            "ood_nll_mean": float(np.mean(ood_nlls)),
            "ood_ppl_mean": float(np.mean(ood_ppls)),
            "params_M": results[arm_name]["params_M"],
            "wallclock_mean_s": float(np.mean(results[arm_name]["wallclock_s"])),
        }
        s = summary[arm_name]
        print(f"  {arm_name:18s}  C4_final={s['c4_final_mean']:.4f} +/- {s['c4_final_std']:.4f}")
        print(f"  {' '*18}  OOD_NLL={s['ood_nll_mean']:.4f}  OOD_PPL={s['ood_ppl_mean']:.2f}")
        print(f"  {' '*18}  params={s['params_M']:.2f}M, time={s['wallclock_mean_s']:.0f}s")

    bf = summary["baseline_full"]
    m3 = summary["minimal_3L"]
    c4_gap = m3["c4_final_mean"] - bf["c4_final_mean"]
    ood_gap = m3["ood_nll_mean"] - bf["ood_nll_mean"]
    ood_ppl_gap_pct = (m3["ood_ppl_mean"] - bf["ood_ppl_mean"]) / bf["ood_ppl_mean"] * 100
    params_ratio = m3["params_M"] / bf["params_M"]
    time_ratio = m3["wallclock_mean_s"] / bf["wallclock_mean_s"]

    print(f"\n  C4 NLL gap (minimal_3L - baseline): {c4_gap:+.4f}")
    print(f"  OOD NLL gap:  {ood_gap:+.4f}")
    print(f"  OOD PPL gap (%):                    {ood_ppl_gap_pct:+.2f}%")
    print(f"  params ratio: {params_ratio:.3f}")
    print(f"  time ratio:   {time_ratio:.3f}")
    print(f"  baseline std (C4): {bf['c4_final_std']:.4f}, minimal_3L std: {m3['c4_final_std']:.4f}")

    # Verdict
    if (c4_gap <= 0.05 and abs(ood_ppl_gap_pct) <= 5 and
        params_ratio <= 0.75 and m3["c4_final_std"] <= 0.10):
        verdict = (f"PASS: minimal_3L matches baseline within 0.05 nats C4 ({c4_gap:+.4f}) AND "
                   f"within 5% OOD PPL ({ood_ppl_gap_pct:+.2f}%) at {100*params_ratio:.0f}% params, "
                   f"per-seed std {m3['c4_final_std']:.3f}<=0.10. ROBUST efficiency claim.")
    elif c4_gap <= 0.10 and abs(ood_ppl_gap_pct) <= 10 and params_ratio <= 0.85:
        verdict = (f"PARTIAL: minimal_3L within 0.10 on both metrics. C4 gap {c4_gap:+.4f}, "
                   f"OOD PPL gap {ood_ppl_gap_pct:+.2f}%. Real but below tightest threshold.")
    else:
        verdict = (f"KILL: C4 gap {c4_gap:+.4f} OR OOD PPL gap {ood_ppl_gap_pct:+.2f}% > thresholds, "
                   f"OR std too high. g139 single-seed result not robust.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 140, "name": "minimal_prior_multiseed_ood",
        "config": {"train_steps": TRAIN_STEPS, "lr": LR, "batch": BATCH_SIZE,
                    "seeds": SEEDS, "n_c4_eval": N_C4_EVAL, "n_ood_eval": N_OOD_EVAL},
        "results": results,
        "summary": summary,
        "deltas": {
            "c4_gap_nats": c4_gap,
            "ood_nll_gap_nats": ood_gap,
            "ood_ppl_gap_pct": ood_ppl_gap_pct,
            "params_ratio": params_ratio,
            "time_ratio": time_ratio,
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_140_minimal_prior_multiseed_ood.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
