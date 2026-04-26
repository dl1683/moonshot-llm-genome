"""
genome_148_hellaswag_capability.py

CODEX Z3 — DOWNSTREAM CAPABILITY VALIDATION.

g141/g146/g147 chain: minimal architecture beats baseline at matched FLOPs
across 30M -> 100M -> 200M, ~0.8pp top-1 edge consistent. Biggest remaining
attack: "this is next-token efficiency, not real capability."

This experiment tests behavioral parity on HellaSwag-style multiple choice:
for each context, 4 candidate continuations, measure log-likelihood under
each, accuracy = how often the model picks the gold continuation.

Protocol: same arms as g147 at 200M scale, retrained (no saved checkpoints).
Reuses g147 protocol exactly: 3 seeds, matched FLOPs, N_TRAIN=32k.

Arms:
  - baseline_200M_4k:    14L+MLP, hidden=1024, ffn=2304 (~209M, 4000 steps)
  - minimal_7L_200M_8k:   7L no-MLP, hidden=1024, ffn=2304 (~80M, 8000 steps)

Eval: HellaSwag validation set, 500 examples (subset for speed).
For each: 4 candidate endings, score each by mean per-token NLL of ending
given context, pick lowest. Accuracy = correct picks / total.

Pre-stated criteria:
  PASS: minimal accuracy >= baseline accuracy − 1pp on HellaSwag across
        3 seeds (i.e. no significant capability degradation).
        Combined with g147 +0.79pp top-1 win, makes capability claim solid.
  PARTIAL: minimal accuracy 1-3pp below baseline.
  KILL: minimal accuracy >3pp below baseline. Win was perplexity-only,
        capability collapses on downstream task.

NOTE: at 200M params trained only 8000 steps on c4_clean, both models
will be near random (25%) on HellaSwag. The COMPARISON between arms is
the signal, not absolute accuracy.

Compute: same as g147 (~35 min training) + HellaSwag eval (~5 min) = ~40 min.

Results: results/genome_148_hellaswag_capability.json
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
N_HELLASWAG = 500
N_TRAIN = 32768
BASELINE_STEPS = 4000
MINIMAL_STEPS = 8000
HS_BATCH = 4  # batch for HellaSwag scoring (4 candidates per question)


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


def load_hellaswag(n=N_HELLASWAG):
    """Load HellaSwag validation. Each example: ctx + 4 endings + label."""
    from datasets import load_dataset
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation")
    except Exception:
        ds = load_dataset("hellaswag", split="validation")
    examples = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        ctx = (ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")).strip()
        if not ctx:
            ctx = ex.get("activity_label", "") + ": " + ex.get("ctx", "")
        endings = ex.get("endings", [])
        try:
            label = int(ex.get("label", -1))
        except (ValueError, TypeError):
            label = -1
        if label < 0 or len(endings) < 4:
            continue
        examples.append({"ctx": ctx, "endings": endings[:4], "label": label})
    return examples


def score_completion(model, tok, ctx_ids, ctx_mask, end_ids, end_mask):
    """Compute mean per-token NLL of end tokens given ctx.
       ctx_ids: (B, T_ctx), end_ids: (B, T_end). Concatenates and runs.
       Returns: tensor of shape (B,) — mean NLL of ending tokens."""
    B = ctx_ids.size(0)
    full_ids = torch.cat([ctx_ids, end_ids], dim=1)
    full_mask = torch.cat([ctx_mask, end_mask], dim=1)
    with torch.no_grad():
        out = model(input_ids=full_ids.to("cuda"), attention_mask=full_mask.to("cuda"),
                    use_cache=False)
    logits = out.logits  # (B, T_ctx+T_end, V)
    # Score the end tokens: predict end_ids[t] from full_ids[T_ctx + t - 1]
    T_ctx = ctx_ids.size(1)
    # Shifted: logits at positions [T_ctx-1 ... T_ctx+T_end-2] predict end_ids[0..T_end-1]
    pred_logits = logits[:, T_ctx - 1: T_ctx - 1 + end_ids.size(1)].float()  # (B, T_end, V)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    end_ids_cuda = end_ids.to("cuda")
    end_mask_cuda = end_mask.to("cuda")
    gathered = log_probs.gather(2, end_ids_cuda.unsqueeze(-1)).squeeze(-1)  # (B, T_end)
    mask = end_mask_cuda.float()
    sum_log_prob = (gathered * mask).sum(dim=1)
    n_tokens = mask.sum(dim=1).clamp(min=1)
    mean_nll = -sum_log_prob / n_tokens
    return mean_nll.cpu()


def hellaswag_accuracy(model, tok, examples):
    """Compute accuracy on HellaSwag. For each example, score 4 endings."""
    model.eval()
    correct = 0
    for i, ex in enumerate(examples):
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = ex["label"]
        # Tokenize all 4 separately, pad
        ctx_enc = tok([ctx] * 4, padding=True, truncation=True,
                      max_length=128, return_tensors="pt")
        end_enc = tok(endings, padding=True, truncation=True,
                      max_length=64, return_tensors="pt", add_special_tokens=False)
        nlls = score_completion(model, tok,
                                  ctx_enc["input_ids"], ctx_enc["attention_mask"],
                                  end_enc["input_ids"], end_enc["attention_mask"])
        pred = int(nlls.argmin())  # lowest NLL = highest probability
        if pred == label:
            correct += 1
    model.train()
    return correct / len(examples)


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
    print("genome_148: HellaSwag downstream capability test (Codex Z3)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print(f"Loading {N_TRAIN} c4 train + 200 c4 eval stimuli...")
    pool_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=N_TRAIN + 200):
        pool_texts.append(rec["text"])
    train_texts = pool_texts[:N_TRAIN]
    c4_eval_texts = pool_texts[N_TRAIN:N_TRAIN + 200]

    print(f"Loading {N_HELLASWAG} HellaSwag examples...")
    hs_examples = load_hellaswag(n=N_HELLASWAG)
    print(f"  loaded {len(hs_examples)} HS examples")

    print(f"Tokenizing...")
    enc_t = tok(train_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc_t["input_ids"]; train_mask = enc_t["attention_mask"]
    enc_c4 = tok(c4_eval_texts, padding=True, truncation=True,
                  max_length=SEQ_LEN, return_tensors="pt")
    c4_eval_ids = enc_c4["input_ids"]; c4_eval_mask = enc_c4["attention_mask"]
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
            c4_nll = measure_eval_nll(model, c4_eval_ids, c4_eval_mask)
            print(f"    C4 NLL: {c4_nll:.4f}")
            print(f"    HellaSwag accuracy ({len(hs_examples)} examples)...")
            hs_acc = hellaswag_accuracy(model, tok, hs_examples)
            print(f"    HellaSwag acc = {100*hs_acc:.2f}%  (random = 25%)")
            results[arm_name]["per_seed"][seed] = {
                "c4_nll": c4_nll, "hellaswag_acc": hs_acc,
            }
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
        c4_nll = [per_seed[s]["c4_nll"] for s in SEEDS if s in per_seed]
        hs_accs = [per_seed[s]["hellaswag_acc"] for s in SEEDS if s in per_seed]
        summary[arm_name] = {
            "c4_nll_mean": float(np.mean(c4_nll)),
            "hs_acc_mean": float(np.mean(hs_accs)),
            "hs_acc_std": float(np.std(hs_accs)),
            "params_M": results[arm_name]["params_M"],
            "wallclock_s_mean": float(np.mean(results[arm_name]["wallclock_s"])),
        }
        s = summary[arm_name]
        print(f"  {arm_name:24s}  C4_NLL={s['c4_nll_mean']:.4f}  "
              f"HellaSwag={100*s['hs_acc_mean']:.2f}%+/-{100*s['hs_acc_std']:.2f}  "
              f"params={s['params_M']:.2f}M  time={s['wallclock_s_mean']:.0f}s")

    if "baseline_200M_4k" in summary and "minimal_7L_200M_8k" in summary:
        bf = summary["baseline_200M_4k"]
        m7 = summary["minimal_7L_200M_8k"]
        hs_gap_pp = (m7["hs_acc_mean"] - bf["hs_acc_mean"]) * 100
        c4_gap = m7["c4_nll_mean"] - bf["c4_nll_mean"]
        print(f"\n  HellaSwag gap (minimal - baseline): {hs_gap_pp:+.3f}pp")
        print(f"  C4 NLL gap: {-c4_gap:+.4f} (positive = minimal better)")

        if hs_gap_pp >= -1.0:
            verdict = (f"PASS: HellaSwag gap {hs_gap_pp:+.2f}pp (>=−1pp). Minimal preserves "
                       f"or matches downstream capability at 200M. Combined with g147 +0.79pp "
                       f"NLL top-1 win, the architecture-prior efficiency claim is now "
                       f"capability-grade.")
        elif hs_gap_pp >= -3.0:
            verdict = (f"PARTIAL: HellaSwag gap {hs_gap_pp:+.2f}pp (1-3pp loss). Some downstream "
                       f"capability degradation. Win is real on perplexity, narrower on capability.")
        else:
            verdict = (f"KILL: HellaSwag gap {hs_gap_pp:+.2f}pp (>3pp loss). Minimal saves "
                       f"compute by sacrificing actual capability. NLL win does not transfer.")
    else:
        verdict = "FAIL: arms did not complete"

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 148, "name": "hellaswag_capability",
        "config": {"baseline_steps": BASELINE_STEPS, "minimal_steps": MINIMAL_STEPS,
                    "n_train_pool": N_TRAIN, "n_hellaswag": len(hs_examples),
                    "lr": LR, "batch": BATCH_SIZE, "seeds": SEEDS},
        "results": results, "summary": summary, "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_148_hellaswag_capability.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
