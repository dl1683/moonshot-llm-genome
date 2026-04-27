"""
genome_158_context_length_inversion.py

POST-g156-PASS THEORY-PREDICTION TEST: context-length inversion sweep.

Pre-reg LOCKED: research/prereg/genome_158_context_length_inversion_2026-04-26.md
Theory: research/derivations/prefix_information_transport.md
Program: research/programs/post_g156_pass_program.md §g158

Tests the theory's sharpest unique prediction: architecture-prior advantage
is monotone in transport demand. As context shrinks, transport demand
shrinks, the minimal-arm advantage shrinks, eventually inverts.

Two arms (30M-class):
  baseline_6L+MLP   hidden=384, ffn=1024, ~30M params
  minimal_3L_noMLP  hidden=384, ZeroMLP,  ~21M params

Four context lengths: L in {32, 64, 128, 256}.
Three seeds: {42, 7, 13}.
Token-budget matched: N_TRAIN_L = N_TRAIN_256 * 256/L (so total token-FLOPs
roughly comparable across L).

Pre-stated criteria:
  PASS_INVERSION:
    Spearman rho(L, delta_L_c4)  >= +0.8
    AND Spearman rho(L, delta_L_ood) >= +0.8
    AND delta_32_c4  <= -0.2pp
    AND delta_256_c4 >= +0.5pp
    AND sign of delta_L matches across both eval sets at every L
  PARTIAL_INVERSION: monotone-increasing with delta_256 >= +0.3pp AND
    rho >= +0.6 in both, but no clean sign flip at L=32
  KILL_INVERSION: rho < +0.3 in either eval set, OR no decay toward zero

Results: results/genome_158_context_length_inversion.json
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

BATCH_SIZE = 8
SEEDS = [42, 7, 13]
N_C4_EVAL = 200
N_OOD_EVAL = 200
N_TRAIN_256 = 32768  # token-budget match: scale up at shorter L
CONTEXT_LENGTHS = [32, 64, 128, 256]
LR_WARMUP_STEPS = 200
LR_GRID = [2e-4, 3e-4, 4e-4]
LR_SELECT_L = 128  # depth at which to choose arm-specific LR


class ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def make_llama(vocab_size, hidden, layers, heads, ffn, no_mlp=False, max_pos=320, seed=42):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        intermediate_size=ffn,
        max_position_embeddings=max_pos,
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


def measure(model, eval_ids, eval_mask):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    correct_top1 = 0
    with torch.no_grad():
        for i in range(0, eval_ids.size(0), BATCH_SIZE):
            ids = eval_ids[i:i+BATCH_SIZE].to("cuda")
            mask = eval_mask[i:i+BATCH_SIZE].to("cuda")
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = out.logits
            sl = logits[:, :-1].contiguous()
            lbl = ids[:, 1:].clone()
            sm = mask[:, 1:]
            valid = (sm != 0)
            lbl_for_loss = lbl.clone()
            lbl_for_loss[~valid] = -100
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl_for_loss.reshape(-1),
                                    ignore_index=-100, reduction="sum")
            n = valid.sum().item()
            total_loss += loss.item()
            total_tokens += n
            preds = sl.argmax(dim=-1)
            correct_top1 += ((preds == lbl) & valid).sum().item()
    model.train()
    return {"nll": total_loss / max(total_tokens, 1),
            "top1_acc": correct_top1 / max(total_tokens, 1)}


def warmup_lr(step, target_lr, warmup_steps):
    if step < warmup_steps:
        return target_lr * (step + 1) / warmup_steps
    return target_lr


def train_arm(arm_name, lr, model, train_ids, train_mask, n_steps, seed):
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name} L={train_ids.size(1)} seed={seed} lr={lr}: params={n_total/1e6:.2f}M steps={n_steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(seed)
    t_arm = time.time()
    model.train()
    n_train = train_ids.size(0)
    nan_seen = False
    for step in range(1, n_steps + 1):
        cur_lr = warmup_lr(step, lr, LR_WARMUP_STEPS)
        for g in opt.param_groups:
            g['lr'] = cur_lr
        idx = rng.integers(0, n_train, size=BATCH_SIZE)
        ids = train_ids[idx].to("cuda")
        mask = train_mask[idx].to("cuda")
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits
        sl = logits[:, :-1].contiguous()
        lbl = ids[:, 1:].clone()
        sm = mask[:, 1:]
        lbl[sm == 0] = -100
        loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            nan_seen = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"    step={step:5d} loss={loss.item():.3f} ({time.time()-t_arm:.0f}s)")
    return n_total, time.time() - t_arm, nan_seen


def tokenize_at_L(tok, texts, L):
    enc = tok(texts, padding="max_length", truncation=True, max_length=L, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def select_lr_per_arm(arm_name, kw, max_pos, vocab, train_ids, train_mask, val_ids, val_mask, n_steps_select):
    """Run small LR sweep on a separate validation bank, pick best."""
    print(f"\n  -- LR selection for {arm_name} at L=128 --")
    best_lr = None
    best_top1 = -1
    for lr in LR_GRID:
        model = make_llama(vocab, max_pos=max_pos, seed=42, **kw)
        train_arm(arm_name + f"_lrsel_{lr}", lr, model, train_ids, train_mask, n_steps_select, seed=42)
        m = measure(model, val_ids, val_mask)
        print(f"    lr={lr}: val_top1={100*m['top1_acc']:.2f}%")
        if m["top1_acc"] > best_top1:
            best_top1 = m["top1_acc"]
            best_lr = lr
        del model
        torch.cuda.empty_cache()
    print(f"  selected lr={best_lr} for {arm_name} (val_top1={100*best_top1:.2f}%)")
    return best_lr


def main():
    t0 = time.time()
    print("genome_158: context-length inversion sweep")
    print(f"  L in {CONTEXT_LENGTHS}, seeds {SEEDS}, arms baseline_6L+MLP vs minimal_3L_noMLP")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    vocab = len(tok)

    # Load enough text for the longest L training + eval + lr-select val + ood eval
    target_n = N_TRAIN_256 + N_C4_EVAL + 1024  # 1024 for LR selection val bank
    print(f"\nLoading {target_n} c4 sequences...")
    pool = []
    for rec in c4_clean_v1(seed=77, n_samples=target_n):
        pool.append(rec["text"])
        if len(pool) >= target_n:
            break

    # OOD eval from wikitext-103 VALIDATION split (per audit-fix A1)
    from datasets import load_dataset
    ds_ood = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    ood_texts = []
    rng = np.random.default_rng(12345)
    perm = rng.permutation(len(ds_ood))
    for idx in perm:
        text = ds_ood[int(idx)]["text"].strip()
        if len(text) < 200:
            continue
        ood_texts.append(text[:1500])
        if len(ood_texts) >= N_OOD_EVAL:
            break

    # Pre-tokenize at every L
    print("Pre-tokenizing at all context lengths...")
    train_texts_all = pool[:N_TRAIN_256]
    eval_texts_c4 = pool[N_TRAIN_256:N_TRAIN_256 + N_C4_EVAL]
    val_lr_texts = pool[N_TRAIN_256 + N_C4_EVAL:N_TRAIN_256 + N_C4_EVAL + 1024]

    arms = {
        "baseline_6L+MLP":  dict(hidden=384, layers=6, heads=6, ffn=1024, no_mlp=False),
        "minimal_3L_noMLP": dict(hidden=384, layers=3, heads=6, ffn=1024, no_mlp=True),
    }

    # Step budget per L: scale to keep total token-FLOPs comparable.
    # At L=256, base budget = 4000 steps × 8 batch × 256 = ~8.4M tokens.
    # At L=L', steps_L = 4000 * 256 / L  (gives same total tokens).
    def steps_at_L(L):
        return int(4000 * 256 / L)

    # 1) LR selection per arm at L=128 with reduced steps for speed
    print("\n=== LR SELECTION at L=128 ===")
    train_ids_128, train_mask_128 = tokenize_at_L(tok, train_texts_all, 128)
    val_ids_128, val_mask_128 = tokenize_at_L(tok, val_lr_texts, 128)
    arm_lr = {}
    for arm_name, kw in arms.items():
        arm_lr[arm_name] = select_lr_per_arm(
            arm_name, kw, max_pos=128 + 64, vocab=vocab,
            train_ids=train_ids_128, train_mask=train_mask_128,
            val_ids=val_ids_128, val_mask=val_mask_128,
            n_steps_select=2000,
        )

    # 2) Main sweep
    results = {}  # results[L][arm][seed] = metrics
    for L in CONTEXT_LENGTHS:
        results[L] = {}
        train_ids, train_mask = tokenize_at_L(tok, train_texts_all, L)
        eval_ids_c4, eval_mask_c4 = tokenize_at_L(tok, eval_texts_c4, L)
        eval_ids_ood, eval_mask_ood = tokenize_at_L(tok, ood_texts, L)
        n_steps = steps_at_L(L)
        max_pos = L + 64
        print(f"\n=== L={L}, n_steps={n_steps} ===")
        for arm_name, kw in arms.items():
            results[L][arm_name] = {}
            lr = arm_lr[arm_name]
            for seed in SEEDS:
                print(f"  -- {arm_name} L={L} seed={seed} --")
                model = make_llama(vocab, max_pos=max_pos, seed=seed, **kw)
                _, elapsed, nan_seen = train_arm(arm_name, lr, model, train_ids, train_mask, n_steps, seed)
                if nan_seen:
                    metrics = {"c4": {"top1_acc": float("nan")}, "ood": {"top1_acc": float("nan")}, "nan_seen": True}
                else:
                    metrics = {
                        "c4": measure(model, eval_ids_c4, eval_mask_c4),
                        "ood": measure(model, eval_ids_ood, eval_mask_ood),
                        "nan_seen": False,
                    }
                metrics["wallclock_s"] = elapsed
                results[L][arm_name][seed] = metrics
                print(f"    c4 top1={100*metrics['c4']['top1_acc']:.2f}%  ood top1={100*metrics['ood']['top1_acc']:.2f}%")
                del model
                torch.cuda.empty_cache()

    # 3) Analysis
    print(f"\n=== ANALYSIS ===")
    delta_per_L = {}
    for L in CONTEXT_LENGTHS:
        nat_b = [results[L]["baseline_6L+MLP"][s]["c4"]["top1_acc"] for s in SEEDS
                 if not results[L]["baseline_6L+MLP"][s]["nan_seen"]]
        nat_m = [results[L]["minimal_3L_noMLP"][s]["c4"]["top1_acc"] for s in SEEDS
                 if not results[L]["minimal_3L_noMLP"][s]["nan_seen"]]
        ood_b = [results[L]["baseline_6L+MLP"][s]["ood"]["top1_acc"] for s in SEEDS
                 if not results[L]["baseline_6L+MLP"][s]["nan_seen"]]
        ood_m = [results[L]["minimal_3L_noMLP"][s]["ood"]["top1_acc"] for s in SEEDS
                 if not results[L]["minimal_3L_noMLP"][s]["nan_seen"]]
        if not nat_b or not nat_m:
            continue
        d_c4 = (np.mean(nat_m) - np.mean(nat_b)) * 100
        d_ood = (np.mean(ood_m) - np.mean(ood_b)) * 100
        delta_per_L[L] = {"c4": d_c4, "ood": d_ood,
                          "n_b": len(nat_b), "n_m": len(nat_m)}
        print(f"  L={L:3d}: delta_c4={d_c4:+.2f}pp  delta_ood={d_ood:+.2f}pp")

    Ls_arr = np.array([L for L in CONTEXT_LENGTHS if L in delta_per_L])
    d_c4_arr = np.array([delta_per_L[L]["c4"] for L in Ls_arr])
    d_ood_arr = np.array([delta_per_L[L]["ood"] for L in Ls_arr])

    from scipy.stats import spearmanr
    rho_c4, _ = spearmanr(Ls_arr, d_c4_arr) if len(Ls_arr) >= 3 else (float("nan"), None)
    rho_ood, _ = spearmanr(Ls_arr, d_ood_arr) if len(Ls_arr) >= 3 else (float("nan"), None)
    print(f"\n  Spearman rho(L, delta_c4)  = {rho_c4:+.3f}")
    print(f"  Spearman rho(L, delta_ood) = {rho_ood:+.3f}")

    d32_c4 = delta_per_L.get(32, {}).get("c4", float("nan"))
    d256_c4 = delta_per_L.get(256, {}).get("c4", float("nan"))

    # Sign-pattern check
    sign_consistent = all(
        np.sign(delta_per_L[L]["c4"]) == np.sign(delta_per_L[L]["ood"])
        for L in delta_per_L
    )

    if (rho_c4 >= 0.8 and rho_ood >= 0.8 and d32_c4 <= -0.2 and d256_c4 >= 0.5
        and sign_consistent):
        verdict = (f"PASS_INVERSION: rho_c4={rho_c4:+.2f}, rho_ood={rho_ood:+.2f}, "
                   f"delta_32_c4={d32_c4:+.2f}pp, delta_256_c4={d256_c4:+.2f}pp, signs consistent. "
                   f"Theory's monotone-attenuation prediction VALIDATED.")
    elif (rho_c4 >= 0.6 and rho_ood >= 0.6 and d256_c4 >= 0.3):
        verdict = (f"PARTIAL_INVERSION: rho_c4={rho_c4:+.2f}, rho_ood={rho_ood:+.2f}, "
                   f"delta_256_c4={d256_c4:+.2f}pp. Direction supported, no clean sign flip at L=32.")
    elif rho_c4 < 0.3 or rho_ood < 0.3:
        verdict = (f"KILL_INVERSION: rho_c4={rho_c4:+.2f} or rho_ood={rho_ood:+.2f} below 0.3. "
                   f"No monotone-in-L pattern; theory loses inversion axis.")
    else:
        verdict = (f"AMBIGUOUS: rho_c4={rho_c4:+.2f}, rho_ood={rho_ood:+.2f}, "
                   f"delta_32={d32_c4:+.2f}pp, delta_256={d256_c4:+.2f}pp.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 158, "name": "context_length_inversion",
        "config": {"context_lengths": CONTEXT_LENGTHS, "seeds": SEEDS,
                    "warmup_steps": LR_WARMUP_STEPS, "n_train_256": N_TRAIN_256,
                    "lr_grid": LR_GRID, "lr_select_L": LR_SELECT_L,
                    "arm_lr": arm_lr},
        "results": {str(L): results[L] for L in results},
        "delta_per_L": {str(L): v for L, v in delta_per_L.items()},
        "spearman_rho_c4": float(rho_c4) if rho_c4 == rho_c4 else None,
        "spearman_rho_ood": float(rho_ood) if rho_ood == rho_ood else None,
        "delta_32_c4": float(d32_c4) if d32_c4 == d32_c4 else None,
        "delta_256_c4": float(d256_c4) if d256_c4 == d256_c4 else None,
        "sign_consistent": bool(sign_consistent),
        "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_158_context_length_inversion.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
