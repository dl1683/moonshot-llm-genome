"""
genome_142_push_efficiency_boundary.py

CODEX U1 — PUSH THE EFFICIENCY BOUNDARY.

After g138-141 established that minimal_3L (3 layers, no MLP, attention only,
hidden=384) matches baseline_full (6L+MLP) at 30% less params/compute with
slightly better OOD generalization, this experiment locates the FAILURE
BOUNDARY by testing increasingly minimal architectures.

Arms (same training protocol as g139-141: 4000 steps full-unfreeze, lr=3e-4):
  - baseline_full        : 6L + MLP, hidden=384  (~30M)  [reference]
  - minimal_3L           : 3L no_MLP, hidden=384 (~21M)  [g139-141 anchor]
  - minimal_2L           : 2L no_MLP, hidden=384 (~20M)  [NEW]
  - minimal_1L           : 1L no_MLP, hidden=384 (~19M)  [NEW]

Same 3 seeds (42, 7, 13), same eval (C4 + WikiText-103 OOD), same metrics
(NLL, top-1 acc, top-5 acc).

Pre-stated criteria:
  PASS_2L: minimal_2L matches baseline within 1pp top-1, 0.05 NLL on BOTH
           C4 and OOD with std <=0.5pp. Headline: 50%+ efficiency.
  PARTIAL_2L: 2L within 2pp top-1.
  KILL_2L: 2L drops capability >2pp — 3L is the boundary.

  Same criteria for 1L. We expect at least 1L to fail.

Compute: 4 arms x 3 seeds x 4000 steps + capability eval = ~25-35 min.

Results: results/genome_142_push_efficiency_boundary.json
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
    print("genome_142: push efficiency boundary (Codex U1)")

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
        ("baseline_full", dict(hidden=384, layers=6, heads=6, ffn=1024, no_mlp=False)),
        ("minimal_3L",    dict(hidden=384, layers=3, heads=6, ffn=1024, no_mlp=True)),
        ("minimal_2L",    dict(hidden=384, layers=2, heads=6, ffn=1024, no_mlp=True)),
        ("minimal_1L",    dict(hidden=384, layers=1, heads=6, ffn=1024, no_mlp=True)),
    ]

    results = {a[0]: {"per_seed": {}, "params_M": None, "wallclock_s": []} for a in arms}

    for arm_name, kw in arms:
        for seed in SEEDS:
            print(f"\n=== {arm_name} seed {seed} ===")
            model = make_llama(actual_vocab, seed=seed, **kw)
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
        c4_top1 = [per_seed[s]["c4"]["top1_acc"] for s in SEEDS]
        ood_top1 = [per_seed[s]["ood"]["top1_acc"] for s in SEEDS]
        c4_nll = [per_seed[s]["c4"]["nll"] for s in SEEDS]
        ood_nll = [per_seed[s]["ood"]["nll"] for s in SEEDS]
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

    bf = summary["baseline_full"]
    print(f"\n  baseline reference: C4_NLL={bf['c4_nll_mean']:.4f} OOD_NLL={bf['ood_nll_mean']:.4f}")
    print(f"  baseline top-1: C4={100*bf['c4_top1_mean']:.2f}% OOD={100*bf['ood_top1_mean']:.2f}%")

    boundaries = {}
    for arm_name in ["minimal_3L", "minimal_2L", "minimal_1L"]:
        a = summary[arm_name]
        c4_nll_gap = a["c4_nll_mean"] - bf["c4_nll_mean"]
        ood_nll_gap = a["ood_nll_mean"] - bf["ood_nll_mean"]
        c4_top1_gap_pp = (bf["c4_top1_mean"] - a["c4_top1_mean"]) * 100
        ood_top1_gap_pp = (bf["ood_top1_mean"] - a["ood_top1_mean"]) * 100
        params_pct = 100 * a["params_M"] / bf["params_M"]
        time_pct = 100 * a["wallclock_s_mean"] / bf["wallclock_s_mean"]
        max_std_pp = max(a["c4_top1_std"], a["ood_top1_std"]) * 100

        if (c4_nll_gap <= 0.05 and abs(ood_nll_gap) <= 0.05 and
            abs(c4_top1_gap_pp) <= 1.0 and abs(ood_top1_gap_pp) <= 1.0 and
            max_std_pp <= 0.5):
            verdict = "PASS"
        elif (abs(c4_top1_gap_pp) <= 2.0 and abs(ood_top1_gap_pp) <= 2.0):
            verdict = "PARTIAL"
        else:
            verdict = "KILL"

        boundaries[arm_name] = {
            "verdict": verdict,
            "c4_nll_gap": c4_nll_gap,
            "ood_nll_gap": ood_nll_gap,
            "c4_top1_gap_pp": c4_top1_gap_pp,
            "ood_top1_gap_pp": ood_top1_gap_pp,
            "params_pct": params_pct,
            "time_pct": time_pct,
            "max_top1_std_pp": max_std_pp,
        }
        print(f"\n  {arm_name:13s} [{verdict}]:")
        print(f"    C4_NLL gap {c4_nll_gap:+.4f}  OOD_NLL gap {ood_nll_gap:+.4f}")
        print(f"    C4 top1 gap {c4_top1_gap_pp:+.3f}pp  OOD top1 gap {ood_top1_gap_pp:+.3f}pp")
        print(f"    params {params_pct:.0f}%  time {time_pct:.0f}%  max_std {max_std_pp:.3f}pp")

    # Final headline
    if boundaries["minimal_2L"]["verdict"] == "PASS":
        if boundaries["minimal_1L"]["verdict"] == "PASS":
            headline = (f"PASS_1L+: even 1-layer attention-only minimal matches baseline. "
                        f"Boundary not yet hit. Architecture-prior is incredibly compressible.")
        else:
            headline = (f"PASS_2L: minimal_2L matches baseline at "
                        f"{boundaries['minimal_2L']['params_pct']:.0f}% params, "
                        f"{boundaries['minimal_2L']['time_pct']:.0f}% time. "
                        f"50%+ efficiency confirmed; minimal_1L verdict={boundaries['minimal_1L']['verdict']}.")
    elif boundaries["minimal_2L"]["verdict"] == "PARTIAL":
        headline = (f"PARTIAL_2L: minimal_2L within 2pp top-1, refinement of efficiency frontier. "
                    f"Boundary near 2L.")
    else:
        headline = (f"KILL_2L: minimal_2L drops capability >2pp. 3L is the boundary. "
                    f"g141 result is at the boundary already.")
    print(f"\n  HEADLINE: {headline}")

    out = {
        "genome": 142, "name": "push_efficiency_boundary",
        "config": {"train_steps": TRAIN_STEPS, "lr": LR, "batch": BATCH_SIZE,
                    "seeds": SEEDS},
        "results": results, "summary": summary, "boundaries": boundaries,
        "headline": headline,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_142_push_efficiency_boundary.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
