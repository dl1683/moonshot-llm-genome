"""
genome_152_long_horizon_crossover.py

CODEX (parallel adversarial + prestage consults aligned):
  Strongest remaining attack on architecture-prior thesis is "short-horizon
  compute-optimality masquerading as architectural superiority." Both arms
  are still in immature pretraining at 4k/8k steps (HellaSwag 25% confirms it).

  The clean falsifier: train both at 5-10x current compute; if baseline
  late-crosses and wins, the thesis collapses to "smaller is better in
  low-budget regime" rather than "MLP/depth are wasted compute."

PROTOCOL (PASS-branch of Codex genome_152 prestage):
  Same arms as g147:
    baseline_200M:  14L+MLP, hidden=1024, ffn=2304 (~209M)
    minimal_7L_200M: 7L no-MLP, hidden=1024, ffn=2304 (~81M)

  Matched FLOPs: baseline_steps : minimal_steps = 1 : 2

  Long-horizon: baseline 25k steps, minimal 50k steps (~6x more compute
  than g147's 4k/8k). At 200M, baseline 25k = ~33 min/seed, minimal 50k =
  ~33 min/seed. 3 seeds × 2 arms = 6 runs × 33 min = ~3.3 hours.

  N_TRAIN=131072 (4x larger than g146/147's 32k) so neither arm overfits
  even at 50k steps. 50k × batch 8 = 400k samples / 131k pool = ~3 epochs OK.

  Evaluate at matched-compute checkpoints: (4k,8k), (8k,16k), (16k,32k),
  (25k,50k). At each checkpoint compute C4 NLL + top-1, OOD NLL + top-1.

PRE-STATED CRITERIA (locked):
  PASS: minimal never behind by >0.3pp top-1 at any checkpoint AND
        >=0.3pp ahead by final checkpoint, on BOTH C4 and OOD.
        => Architecture-prior thesis survives the strongest attack.
  PARTIAL: final tie band [-0.3pp, +0.3pp] — gap closes but doesn't reverse.
  KILL: baseline late-crosses and wins by >0.3pp at final.
        => Win was short-horizon, thesis downgraded to low-budget regime.

Compute: ~3.3 hours. Largest experiment of the project.

Results: results/genome_152_long_horizon_crossover.json
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
N_TRAIN = 131072  # 4x larger than g146/147

BASELINE_STEPS = 25000
MINIMAL_STEPS = 50000  # matched FLOPs (2x more steps for half-cost model)

# Matched-compute checkpoint pairs: (baseline_step, minimal_step)
# baseline 4000 ≈ minimal 8000 (matched FLOPs)
# Evaluate at 4 progressively longer compute budgets
CKPT_PAIRS = [
    (4000, 8000),
    (8000, 16000),
    (16000, 32000),
    (25000, 50000),
]


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


def train_arm_with_checkpoints(arm_name, seed, model, train_ids, train_mask,
                                 c4_ids, c4_mask, ood_ids, ood_mask,
                                 n_steps, ckpt_steps):
    """Train, evaluate at each ckpt_step. Returns list of {step, c4, ood}."""
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name} (seed {seed}): total_params={n_total/1e6:.2f}M  steps={n_steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(seed)
    t_arm = time.time()
    model.train()
    n_train = train_ids.size(0)
    ckpts_done = []
    ckpt_set = set(ckpt_steps)
    log_every = 2500

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

        if step in ckpt_set:
            c4 = measure_full(model, c4_ids, c4_mask)
            ood = measure_full(model, ood_ids, ood_mask)
            elapsed = time.time() - t_arm
            ckpts_done.append({
                "step": step, "c4": c4, "ood": ood,
                "wallclock_so_far_s": elapsed,
            })
            print(f"    step={step:6d}  C4_top1={100*c4['top1_acc']:.2f}%  "
                  f"OOD_top1={100*ood['top1_acc']:.2f}%  ({elapsed:.0f}s)")

        if step % log_every == 0 and step not in ckpt_set:
            print(f"    step={step:6d} loss={loss.item():.3f} ({time.time()-t_arm:.0f}s)")

    return ckpts_done, n_total, time.time() - t_arm


def main():
    t0 = time.time()
    print("genome_152: long-horizon crossover test at 200M (Codex AA1)")
    print(f"  N_TRAIN={N_TRAIN}, baseline_steps={BASELINE_STEPS}, minimal_steps={MINIMAL_STEPS}")
    print(f"  checkpoints (baseline,minimal): {CKPT_PAIRS}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print(f"\nLoading {N_TRAIN} c4 train + eval stimuli...")
    pool_texts = []
    target_n = N_TRAIN + N_C4_EVAL
    for rec in c4_clean_v1(seed=42, n_samples=target_n):
        pool_texts.append(rec["text"])
        if len(pool_texts) >= target_n:
            break
    train_texts = pool_texts[:N_TRAIN]
    c4_eval_texts = pool_texts[N_TRAIN:N_TRAIN + N_C4_EVAL]
    ood_eval_texts = load_wikitext_eval()

    print(f"Tokenizing {N_TRAIN} train sequences (this takes ~5 min)...")
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

    baseline_ckpts = [p[0] for p in CKPT_PAIRS]
    minimal_ckpts = [p[1] for p in CKPT_PAIRS]

    arms = [
        ("baseline_200M_25k",   dict(hidden=1024, layers=14, heads=16, ffn=2304, no_mlp=False), BASELINE_STEPS, baseline_ckpts),
        ("minimal_7L_200M_50k", dict(hidden=1024, layers=7,  heads=16, ffn=2304, no_mlp=True),  MINIMAL_STEPS,  minimal_ckpts),
    ]

    results = {a[0]: {"per_seed": {}, "params_M": None, "wallclock_s": []} for a in arms}

    for arm_name, kw, n_steps, ckpts in arms:
        for seed in SEEDS:
            print(f"\n=== {arm_name} seed {seed} ===")
            try:
                model = make_llama(actual_vocab, seed=seed, **kw)
            except Exception as e:
                print(f"  build fail: {e}"); continue
            ckpt_data, n_total, elapsed = train_arm_with_checkpoints(
                arm_name, seed, model, train_ids, train_mask,
                c4_eval_ids, c4_eval_mask, ood_eval_ids, ood_eval_mask,
                n_steps, ckpts,
            )
            results[arm_name]["per_seed"][seed] = ckpt_data
            results[arm_name]["params_M"] = n_total / 1e6
            results[arm_name]["wallclock_s"].append(elapsed)
            del model; torch.cuda.empty_cache()

    # Analysis: gap at each matched-compute checkpoint pair
    print(f"\n=== ANALYSIS ===")
    crossover_data = []
    for (b_step, m_step) in CKPT_PAIRS:
        bf_per_seed = []
        m7_per_seed = []
        for seed in SEEDS:
            bf_seed = results["baseline_200M_25k"]["per_seed"].get(seed, [])
            m7_seed = results["minimal_7L_200M_50k"]["per_seed"].get(seed, [])
            bf_at = next((c for c in bf_seed if c["step"] == b_step), None)
            m7_at = next((c for c in m7_seed if c["step"] == m_step), None)
            if bf_at is None or m7_at is None:
                continue
            bf_per_seed.append(bf_at)
            m7_per_seed.append(m7_at)
        if not bf_per_seed:
            continue
        bf_c4_top1 = float(np.mean([c["c4"]["top1_acc"] for c in bf_per_seed]))
        m7_c4_top1 = float(np.mean([c["c4"]["top1_acc"] for c in m7_per_seed]))
        bf_ood_top1 = float(np.mean([c["ood"]["top1_acc"] for c in bf_per_seed]))
        m7_ood_top1 = float(np.mean([c["ood"]["top1_acc"] for c in m7_per_seed]))
        c4_gap_pp = (m7_c4_top1 - bf_c4_top1) * 100
        ood_gap_pp = (m7_ood_top1 - bf_ood_top1) * 100
        crossover_data.append({
            "baseline_step": b_step, "minimal_step": m_step,
            "baseline_c4_top1": bf_c4_top1,
            "minimal_c4_top1": m7_c4_top1,
            "baseline_ood_top1": bf_ood_top1,
            "minimal_ood_top1": m7_ood_top1,
            "c4_gap_pp": c4_gap_pp,
            "ood_gap_pp": ood_gap_pp,
        })
        print(f"  ckpt ({b_step:5d},{m_step:5d}):  C4 baseline {100*bf_c4_top1:.2f}% -> minimal {100*m7_c4_top1:.2f}% (gap {c4_gap_pp:+.2f}pp)  "
              f"OOD baseline {100*bf_ood_top1:.2f}% -> minimal {100*m7_ood_top1:.2f}% ({ood_gap_pp:+.2f}pp)")

    if not crossover_data:
        verdict = "FAIL: no checkpoints completed"
    else:
        # PASS: minimal never behind by >0.3pp at any ckpt AND >=0.3pp ahead at final on both metrics
        any_minimal_behind = any(
            (c["c4_gap_pp"] < -0.3 or c["ood_gap_pp"] < -0.3) for c in crossover_data
        )
        final = crossover_data[-1]
        final_strong = (final["c4_gap_pp"] >= 0.3 and final["ood_gap_pp"] >= 0.3)
        final_in_band = (-0.3 <= final["c4_gap_pp"] <= 0.3 and -0.3 <= final["ood_gap_pp"] <= 0.3)
        late_crossover = (final["c4_gap_pp"] < -0.3 or final["ood_gap_pp"] < -0.3)

        if final_strong and not any_minimal_behind:
            verdict = (f"PASS: minimal beats baseline at final checkpoint by C4 +{final['c4_gap_pp']:.2f}pp, "
                       f"OOD +{final['ood_gap_pp']:.2f}pp, and never falls behind >0.3pp at any earlier "
                       f"checkpoint. Architecture-prior thesis SURVIVES the strongest attack.")
        elif final_in_band:
            verdict = (f"PARTIAL: final gap in [-0.3, +0.3]pp band (C4 {final['c4_gap_pp']:+.2f}, "
                       f"OOD {final['ood_gap_pp']:+.2f}). Gap closes but no reversal — win was real "
                       f"but bounded.")
        elif late_crossover:
            verdict = (f"KILL: baseline late-crosses. C4 {final['c4_gap_pp']:+.2f}pp, OOD {final['ood_gap_pp']:+.2f}pp "
                       f"at final ckpt. Architecture-prior win was a SHORT-HORIZON ARTIFACT. "
                       f"Thesis downgraded to low-budget regime claim.")
        else:
            verdict = (f"AMBIGUOUS: C4 {final['c4_gap_pp']:+.2f}pp, OOD {final['ood_gap_pp']:+.2f}pp at final. "
                       f"Mixed signal across metrics.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 152, "name": "long_horizon_crossover",
        "config": {"baseline_steps": BASELINE_STEPS, "minimal_steps": MINIMAL_STEPS,
                    "n_train_pool": N_TRAIN, "lr": LR, "batch": BATCH_SIZE, "seeds": SEEDS,
                    "ckpt_pairs": CKPT_PAIRS},
        "results": {arm: {"params_M": v["params_M"],
                          "per_seed": {str(s): v["per_seed"][s] for s in v["per_seed"]},
                          "wallclock_s": v["wallclock_s"]}
                    for arm, v in results.items()},
        "crossover_analysis": crossover_data,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_152_long_horizon_crossover.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
