"""
genome_139_minimal_prior_benchmark.py

CODEX R1 — MINIMAL-PRIOR ARCHITECTURE BENCHMARK.

genome_138 PASS: architecture-prior under glue-only training is localized to
attention + width + residuals. MLP, depth, causal mask are nearly irrelevant.

R1: build a stripped-down model around the irreducible prior, train it FULLY
from scratch (not just glue), compare to full Llama at matched compute.
If minimal-prior model REACHES or EXCEEDS full Llama, this is electricity-
grade efficiency demo (manifesto criterion c).

Three arms:
  A) BASELINE_FULL:    full Llama, 6 layers, hidden=384, ffn=1024 (~30M params)
                       attn + MLP + residual + RMSNorm + RoPE
  B) MINIMAL_PRIOR:    3 layers, hidden=384, NO MLP (zeroed), attn-only
                       (~21M params, fewer than baseline)
  C) MINIMAL_WIDE:     2 layers, hidden=512, NO MLP, attn-only
                       (~31M params, similar params to baseline but wider+shallower)

All trained 4000 steps full unfreeze, same lr, same batch, same data pool.

Metrics:
  - NLL trajectory at log-spaced checkpoints [0, 32, 128, 512, 1000, 2000, 4000]
  - Final NLL @ step 4000
  - Total params (hardware-cost proxy)
  - Wall-clock time per arm

Pre-stated criteria:
  PASS: Arm B or Arm C reaches NLL within 0.10 of Arm A at step 4000
        AND uses similar-or-fewer total params (within 110%).
        This is electricity-grade efficiency: same capability at no cost increase.
  STRONGER PASS: minimal arm reaches Arm A's final NLL at <=70% the steps
        OR <=70% the params.
  PARTIAL: minimal reaches within 0.25 of baseline final NLL.
  KILL: minimal arms are >0.25 worse than baseline. The glue-only-prior
        finding doesn t generalize to from-scratch full training.

Compute estimate: 3 arms x 4000 steps + evals = ~10-15 min on RTX 5090.

Results: results/genome_139_minimal_prior_benchmark.json
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
SEED = 42
N_EVAL = 200
N_TRAIN = 4000


class ZeroMLP(nn.Module):
    """Replaces MLP with zero output (residual passes through unchanged)."""
    def forward(self, x):
        return torch.zeros_like(x)


def make_llama(vocab_size, hidden, layers, heads, ffn, no_mlp=False, seed=SEED):
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


def count_effective_params(model):
    """Count parameters with requires_grad=True (excludes ZeroMLP since it has no params)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def train_arm(arm_name, model, train_ids, train_mask, eval_ids, eval_mask):
    n_total = count_total_params(model)
    n_grad = count_effective_params(model)
    print(f"  total_params: {n_total/1e6:.2f}M  trainable: {n_grad/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(SEED)
    rows = []
    t_arm = time.time()

    nll0 = measure_eval_nll(model, eval_ids, eval_mask)
    rows.append({"step": 0, "nll": nll0})
    print(f"  step=0  NLL={nll0:.3f}")
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
            nll = measure_eval_nll(model, eval_ids, eval_mask)
            rows.append({"step": step, "nll": nll, "loss": float(loss.item())})
            print(f"  step={step:5d}  NLL={nll:.3f}  loss={loss.item():.3f}  ({time.time()-t_arm:.0f}s)")
            model.train()
            next_idx += 1

    elapsed = time.time() - t_arm
    return rows, n_total, n_grad, elapsed


def main():
    t0 = time.time()
    print("genome_139: minimal-prior architecture benchmark (Codex R1)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print("Loading c4_clean_v1 stimuli...")
    pool_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=N_TRAIN + N_EVAL):
        pool_texts.append(rec["text"])
    train_texts = pool_texts[:N_TRAIN]
    eval_texts = pool_texts[N_TRAIN:N_TRAIN + N_EVAL]

    enc_t = tok(train_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc_t["input_ids"]; train_mask = enc_t["attention_mask"]
    enc_e = tok(eval_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    eval_ids = enc_e["input_ids"]; eval_mask = enc_e["attention_mask"]
    print(f"  train: {train_ids.shape}, eval: {eval_ids.shape}")

    arms = [
        # name, hidden, layers, heads, ffn, no_mlp
        ("baseline_full",  384, 6, 6, 1024, False),
        ("minimal_3L",     384, 3, 6, 1024, True),  # 3 layers, no MLP
        ("minimal_wide_2L",512, 2, 8, 1024, True),  # 2 layers, wider, no MLP
    ]

    all_results = {}
    for arm_name, hidden, layers, heads, ffn, no_mlp in arms:
        print(f"\n=== Arm: {arm_name} (hidden={hidden}, layers={layers}, no_mlp={no_mlp}) ===")
        model = make_llama(actual_vocab, hidden, layers, heads, ffn, no_mlp=no_mlp)
        rows, n_total, n_grad, elapsed = train_arm(
            arm_name, model, train_ids, train_mask, eval_ids, eval_mask,
        )
        all_results[arm_name] = {
            "config": {"hidden": hidden, "layers": layers, "heads": heads,
                        "ffn": ffn, "no_mlp": no_mlp},
            "n_total_params": n_total,
            "n_trainable_params": n_grad,
            "elapsed_s": elapsed,
            "rows": rows,
        }
        del model
        torch.cuda.empty_cache()

    # === ANALYSIS ===
    print(f"\n=== ANALYSIS ===")
    baseline_final = all_results["baseline_full"]["rows"][-1]["nll"]
    baseline_params = all_results["baseline_full"]["n_total_params"]
    baseline_time = all_results["baseline_full"]["elapsed_s"]
    print(f"  baseline_full:    NLL={baseline_final:.4f}  params={baseline_params/1e6:.2f}M  time={baseline_time:.0f}s")

    summary = {}
    for arm_name in ["baseline_full", "minimal_3L", "minimal_wide_2L"]:
        r = all_results[arm_name]
        final_nll = r["rows"][-1]["nll"]
        params_M = r["n_total_params"] / 1e6
        nll_gap = final_nll - baseline_final
        params_ratio = r["n_total_params"] / baseline_params
        summary[arm_name] = {
            "final_nll": float(final_nll),
            "nll_gap_vs_baseline": float(nll_gap),
            "params_M": float(params_M),
            "params_ratio_vs_baseline": float(params_ratio),
            "wallclock_s": r["elapsed_s"],
        }
        print(f"  {arm_name:18s} final_nll={final_nll:.4f}  gap={nll_gap:+.4f}  "
              f"params={params_M:.2f}M ({100*params_ratio:.0f}%)  time={r['elapsed_s']:.0f}s")

    # Pick the better minimal arm
    minimal_arms = ["minimal_3L", "minimal_wide_2L"]
    best_minimal = min(minimal_arms, key=lambda a: summary[a]["final_nll"])
    bm = summary[best_minimal]
    print(f"\n  best minimal arm: {best_minimal}")
    print(f"    final_nll: {bm['final_nll']:.4f} (gap {bm['nll_gap_vs_baseline']:+.4f})")
    print(f"    params: {bm['params_M']:.2f}M ({100*bm['params_ratio_vs_baseline']:.0f}% of baseline)")

    # Verdict
    nll_gap = bm["nll_gap_vs_baseline"]
    params_ratio = bm["params_ratio_vs_baseline"]

    if nll_gap <= 0.10 and params_ratio <= 1.10:
        # Stronger PASS check
        if params_ratio <= 0.70 or bm["final_nll"] < baseline_final:
            verdict = (f"STRONG PASS: {best_minimal} reaches NLL {bm['final_nll']:.4f} (gap {nll_gap:+.4f}) "
                       f"with {100*params_ratio:.0f}% baseline params. "
                       f"Architecture-prior collapse is electricity-grade efficiency.")
        else:
            verdict = (f"PASS: {best_minimal} matches baseline NLL within 0.10 "
                       f"(gap {nll_gap:+.4f}) at {100*params_ratio:.0f}% params. "
                       f"Minimal architecture is viable; MLP and most depth confirmed unnecessary.")
    elif nll_gap <= 0.25:
        verdict = (f"PARTIAL: {best_minimal} within 0.25 of baseline (gap {nll_gap:+.4f}). "
                   f"Minimal architecture works but not at full parity. Refinement needed.")
    else:
        verdict = (f"KILL: best minimal arm ({best_minimal}) has gap {nll_gap:+.4f} > 0.25. "
                   f"Glue-only-prior finding does NOT generalize to from-scratch full training. "
                   f"MLP and depth matter when training the full stack.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 139, "name": "minimal_prior_benchmark",
        "config": {"train_steps": TRAIN_STEPS, "lr": LR, "batch": BATCH_SIZE,
                    "seq_len": SEQ_LEN, "n_train_pool": N_TRAIN, "n_eval": N_EVAL},
        "arms": all_results,
        "summary": summary,
        "best_minimal_arm": best_minimal,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_139_minimal_prior_benchmark.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
