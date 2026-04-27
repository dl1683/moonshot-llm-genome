"""
genome_161_run.py — full training experiment for g161.

Imports the model + helpers from genome_161_rwkv_training_extension.py
(Codex-maintained module) and adds the locked-prereg training loop.

Conditional on g159 PASS/PARTIAL per
research/programs/post_g156_pass_program.md §g161.

NOTE: Codex pre-flight (codex_outputs/g161_rwkv_implementation.md) flagged
that the pure-PyTorch WKV scan is too slow for the 4-hr envelope on
RTX 5090 Laptop. This runner expects a fused WKV kernel OR a reduced-
scope launch. Hard-aborts via microbenchmark before main loop if the
projected runtime exceeds COMPUTE.md envelope.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

# Import model + helpers from the Codex-maintained module
from genome_161_rwkv_training_extension import (
    RWKV4ForCausalLM, build_rwkv_arm, flop_match_ratio,
    BASELINE_LAYERS, TRANSPORT_LAYERS, HIDDEN_SIZE, CHANNEL_MIX_HIDDEN, VOCAB_SIZE,
)
from stimulus_banks import c4_clean_v1

ROOT = _THIS_DIR.parent

SEQ_LEN = 256
BATCH_SIZE = 8
SEEDS = [42, 7, 13]
# Per Codex pre-flight: locked prereg requires 1024/512 eval banks
N_C4_EVAL = 1024
N_OOD_EVAL = 512
N_TRAIN = 32768
TRAIN_STEPS = 4000
LR_WARMUP_STEPS = 200
SHUFFLE_SEED = 42
LR_GRID = [2e-4, 3e-4, 4e-4]
HARD_ABORT_HOURS = 3.5
MICRO_BENCH_STEPS = 30


def shuffle_token_rows(ids, mask, shuffle_seed=SHUFFLE_SEED):
    rng = np.random.default_rng(shuffle_seed)
    out = ids.clone()
    for r in range(ids.shape[0]):
        valid_pos = (mask[r] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        if len(valid_pos) <= 1:
            continue
        perm = rng.permutation(len(valid_pos))
        out[r, valid_pos] = ids[r, valid_pos[perm]]
    return out


def warmup_lr(step, target_lr, warmup_steps):
    if step < warmup_steps:
        return target_lr * (step + 1) / warmup_steps
    return target_lr


def measure(model, eval_ids, eval_mask, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    with torch.no_grad():
        for i in range(0, eval_ids.size(0), BATCH_SIZE):
            ids = eval_ids[i:i+BATCH_SIZE].to(device)
            mask = eval_mask[i:i+BATCH_SIZE].to(device)
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = out.logits.float() if hasattr(out, "logits") else out["logits"].float()
            sl = logits[:, :-1].contiguous()
            lbl = ids[:, 1:].clone()
            sm = mask[:, 1:]
            valid = (sm != 0)
            lbl[~valid] = -100
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1),
                                    ignore_index=-100, reduction="sum")
            total_loss += loss.item()
            total_tokens += valid.sum().item()
            preds = sl.argmax(dim=-1)
            correct_top1 += ((preds == lbl) & valid).sum().item()
    model.train()
    return {"nll": total_loss / max(total_tokens, 1),
            "top1_acc": correct_top1 / max(total_tokens, 1)}


def train_arm(arm_name, lr, model, train_ids, train_mask, n_steps, seed, device="cuda"):
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name} seed={seed} lr={lr}: params={n_total/1e6:.2f}M steps={n_steps}")
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
        ids = train_ids[idx].to(device)
        mask = train_mask[idx].to(device)
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits.float() if hasattr(out, "logits") else out["logits"].float()
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


def microbenchmark():
    """Estimate total runtime; abort if over envelope."""
    print("Microbenchmark to project total runtime...")
    device = "cuda"
    fake_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    fake_mask = torch.ones_like(fake_ids)
    times = {}
    for arm_name in ["baseline_rwkv", "transport_heavy"]:
        model = build_rwkv_arm(arm_name, seed=42, device=device, dtype=torch.bfloat16)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        # Warmup
        for _ in range(3):
            out = model(input_ids=fake_ids, attention_mask=fake_mask, use_cache=False)
            loss = F.cross_entropy(out.logits.float().reshape(-1, VOCAB_SIZE),
                                    fake_ids.reshape(-1))
            loss.backward()
            opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(MICRO_BENCH_STEPS):
            out = model(input_ids=fake_ids, attention_mask=fake_mask, use_cache=False)
            loss = F.cross_entropy(out.logits.float().reshape(-1, VOCAB_SIZE),
                                    fake_ids.reshape(-1))
            loss.backward()
            opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        per_step = (time.time() - t0) / MICRO_BENCH_STEPS
        times[arm_name] = per_step
        print(f"  {arm_name}: {1000*per_step:.1f} ms/step")
        del model, opt
        torch.cuda.empty_cache()
    # 12 cells = 2 arms x 2 conds x 3 seeds, each TRAIN_STEPS = 4000.
    # Per Codex pre-flight: was undercounting by 2x. Correct formula:
    # for each arm, time = per_step[arm] * TRAIN_STEPS * (#seeds * #conditions)
    n_cells_per_arm = len(SEEDS) * 2  # natural + shuffled
    total_seconds = sum(times[arm] * TRAIN_STEPS * n_cells_per_arm for arm in times)
    total_hours = total_seconds / 3600
    print(f"  PROJECTED full run: {total_hours:.2f} hr")
    if total_hours > HARD_ABORT_HOURS:
        raise RuntimeError(f"Projected {total_hours:.2f} hr > envelope {HARD_ABORT_HOURS} hr; "
                           f"per Codex pre-flight, RWKV pure-PyTorch scan is too slow on RTX 5090. "
                           f"Need a fused WKV kernel before running g161.")
    return total_hours


def main():
    t0 = time.time()
    print("genome_161 run: RWKV transport extension")
    flop_diff = flop_match_ratio() * 100
    print(f"  forward FLOP match: {flop_diff:.2f}%")
    if flop_diff > 2.0:
        raise RuntimeError(f"FLOP match violated: {flop_diff:.2f}%")

    projected_hr = microbenchmark()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # Per Codex pre-flight Sev-8: must use c4 VALIDATION split for eval, not train stream.
    print(f"\nLoading {N_TRAIN} c4 train sequences (seed=161; train stream)...")
    train_pool = []
    for rec in c4_clean_v1(seed=161, n_samples=N_TRAIN):
        train_pool.append(rec["text"])
        if len(train_pool) >= N_TRAIN:
            break
    enc_t = tok(train_pool, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    train_ids_nat = enc_t["input_ids"]
    train_mask = enc_t["attention_mask"]

    from datasets import load_dataset
    print(f"Loading {N_C4_EVAL} c4 VALIDATION sequences for eval...")
    val_pool = []
    try:
        ds_c4_val = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        for ex in ds_c4_val:
            t = ex["text"]
            if len(t) > 200:
                val_pool.append(t)
            if len(val_pool) >= N_C4_EVAL:
                break
    except Exception as e:
        print(f"  c4 streaming failed: {e}; trying file fallback")
        ds_c4_val = load_dataset("allenai/c4", "en", split="validation",
                                   data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"})
        for ex in ds_c4_val:
            val_pool.append(ex["text"])
            if len(val_pool) >= N_C4_EVAL:
                break
    enc_e = tok(val_pool[:N_C4_EVAL], padding="max_length", truncation=True,
                  max_length=SEQ_LEN, return_tensors="pt")
    eval_ids_nat = enc_e["input_ids"]
    eval_mask = enc_e["attention_mask"]
    train_ids_shuf = shuffle_token_rows(train_ids_nat, train_mask, SHUFFLE_SEED)
    eval_ids_shuf = shuffle_token_rows(eval_ids_nat, eval_mask, SHUFFLE_SEED + 1)

    from datasets import load_dataset
    ds_ood = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    ood_texts = []
    rng_ood = np.random.default_rng(12345)
    for idx in rng_ood.permutation(len(ds_ood)):
        t = ds_ood[int(idx)]["text"].strip()
        if len(t) > 200:
            ood_texts.append(t[:1500])
        if len(ood_texts) >= N_OOD_EVAL:
            break
    enc_ood = tok(ood_texts, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    ood_ids_nat = enc_ood["input_ids"]
    ood_mask = enc_ood["attention_mask"]
    ood_ids_shuf = shuffle_token_rows(ood_ids_nat, ood_mask, SHUFFLE_SEED + 2)

    arms = ["baseline_rwkv", "transport_heavy"]
    conditions = [
        ("natural", train_ids_nat, train_mask, eval_ids_nat, eval_mask, ood_ids_nat, ood_mask),
        ("token_shuffled", train_ids_shuf, train_mask, eval_ids_shuf, eval_mask, ood_ids_shuf, ood_mask),
    ]
    # Per Codex pre-flight: arm-specific LR selection on a SEPARATE val bank.
    # Use a small slice of validation for selection, train budget = 500 steps.
    print("\n=== LR SELECTION (per arm, separate val bank) ===")
    LR_GRID_RWKV = [2e-4, 3e-4, 4e-4]
    LR_SELECT_STEPS = 500
    # Per cycle 9 code review Sev-8: previous fallback `val_pool[:256]` leaked
    # eval data into LR selection (same bank as eval_ids_nat). Force-load a
    # SEPARATE 256-sequence slice AFTER the eval bank, both on streaming and
    # file-fallback paths. Raise if not enough validation data.
    sel_val_pool = []
    needed = N_C4_EVAL + 256 + 50
    try:
        ds_c4_val2 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        for ex in ds_c4_val2:
            t = ex["text"]
            if len(t) > 200:
                sel_val_pool.append(t)
            if len(sel_val_pool) >= needed:
                break
    except Exception as e:
        print(f"  c4 streaming for LR-sel failed: {e}; trying file fallback")
        ds_c4_val2 = load_dataset("allenai/c4", "en", split="validation",
                                    data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"})
        for ex in ds_c4_val2:
            t = ex["text"]
            if len(t) > 200:
                sel_val_pool.append(t)
            if len(sel_val_pool) >= needed:
                break
    if len(sel_val_pool) < N_C4_EVAL + 256:
        raise RuntimeError(f"Not enough c4 validation for LR sel: {len(sel_val_pool)} < {N_C4_EVAL + 256}")
    # Always take the slice AFTER the eval bank — never overlap
    sel_texts = sel_val_pool[N_C4_EVAL:N_C4_EVAL + 256]
    enc_sel = tok(sel_texts, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    sel_ids, sel_mask_t = enc_sel["input_ids"], enc_sel["attention_mask"]

    arm_lr = {}
    for arm_name in arms:
        best_lr = None
        best_top1 = -1
        for lr_try in LR_GRID_RWKV:
            print(f"  selecting LR for {arm_name} at lr={lr_try}...")
            mdl = build_rwkv_arm(arm_name, seed=42, device="cuda", dtype=torch.bfloat16)
            _, _, ns = train_arm(arm_name + f"_lrsel_{lr_try}", lr_try, mdl,
                                   train_ids_nat, train_mask, LR_SELECT_STEPS, seed=42)
            if not ns:
                m = measure(mdl, sel_ids, sel_mask_t)
                if m["top1_acc"] > best_top1:
                    best_top1 = m["top1_acc"]
                    best_lr = lr_try
                print(f"    lr={lr_try}: val_top1={100*m['top1_acc']:.2f}%")
            del mdl
            torch.cuda.empty_cache()
        if best_lr is None:
            raise RuntimeError(f"LR selection failed for {arm_name} (all NaN)")
        print(f"  {arm_name}: chosen lr={best_lr} (val_top1={100*best_top1:.2f}%)")
        arm_lr[arm_name] = best_lr

    results = {}
    for cond_name, t_ids, t_mask, e_ids, e_mask, o_ids, o_mask in conditions:
        results[cond_name] = {}
        for arm_name in arms:
            results[cond_name][arm_name] = {}
            lr = arm_lr[arm_name]
            for seed in SEEDS:
                print(f"\n=== cond={cond_name} arm={arm_name} seed={seed} ===")
                model = build_rwkv_arm(arm_name, seed=seed, device="cuda", dtype=torch.bfloat16)
                n_total, elapsed, nan_seen = train_arm(arm_name, lr, model, t_ids, t_mask, TRAIN_STEPS, seed)
                metrics = {"nan_seen": nan_seen, "wallclock_s": elapsed, "params_M": n_total / 1e6}
                if not nan_seen:
                    metrics["c4"] = measure(model, e_ids, e_mask)
                    metrics["ood"] = measure(model, o_ids, o_mask)
                    print(f"    c4 top1={100*metrics['c4']['top1_acc']:.2f}%  "
                          f"ood top1={100*metrics['ood']['top1_acc']:.2f}%")
                else:
                    metrics["c4"] = {"top1_acc": float("nan"), "nll": float("nan")}
                    metrics["ood"] = {"top1_acc": float("nan"), "nll": float("nan")}
                results[cond_name][arm_name][seed] = metrics
                del model
                torch.cuda.empty_cache()

    # Per Codex pre-flight Sev-8: completeness guard before verdict
    required_n = len(SEEDS)
    incomplete = []
    for cond in ["natural", "token_shuffled"]:
        for arm in ["baseline_rwkv", "transport_heavy"]:
            n_valid = sum(1 for s in SEEDS
                           if not results[cond][arm].get(s, {}).get("nan_seen", True))
            if n_valid != required_n:
                incomplete.append((cond, arm, n_valid))
    if incomplete:
        raise RuntimeError(f"g161 incomplete: {incomplete}; cannot emit verdict")

    print(f"\n=== ANALYSIS ===")
    deltas = {}
    for cond_name in ["natural", "token_shuffled"]:
        b_c4 = [results[cond_name]["baseline_rwkv"][s]["c4"]["top1_acc"] for s in SEEDS
                if not results[cond_name]["baseline_rwkv"][s]["nan_seen"]]
        t_c4 = [results[cond_name]["transport_heavy"][s]["c4"]["top1_acc"] for s in SEEDS
                if not results[cond_name]["transport_heavy"][s]["nan_seen"]]
        b_ood = [results[cond_name]["baseline_rwkv"][s]["ood"]["top1_acc"] for s in SEEDS
                 if not results[cond_name]["baseline_rwkv"][s]["nan_seen"]]
        t_ood = [results[cond_name]["transport_heavy"][s]["ood"]["top1_acc"] for s in SEEDS
                 if not results[cond_name]["transport_heavy"][s]["nan_seen"]]
        d_c4 = (np.mean(t_c4) - np.mean(b_c4)) * 100 if b_c4 and t_c4 else float("nan")
        d_ood = (np.mean(t_ood) - np.mean(b_ood)) * 100 if b_ood and t_ood else float("nan")
        deltas[cond_name] = {"c4": d_c4, "ood": d_ood}
        print(f"  {cond_name}: delta_c4={d_c4:+.2f}pp, delta_ood={d_ood:+.2f}pp")

    nat_c4 = deltas["natural"]["c4"]
    shuf_c4 = deltas["token_shuffled"]["c4"]
    nat_ood = deltas["natural"]["ood"]
    shuf_ood = deltas["token_shuffled"]["ood"]
    contrast_c4 = nat_c4 - shuf_c4
    contrast_ood = nat_ood - shuf_ood

    if (nat_c4 >= 0.3 and nat_ood >= 0.3 and shuf_c4 <= 0.1 and shuf_ood <= 0.1
        and contrast_c4 >= 0.3 and contrast_ood >= 0.3):
        verdict = (f"PASS_RWKV: contrast_c4={contrast_c4:+.2f}pp / ood={contrast_ood:+.2f}pp.")
    elif nat_c4 >= 0.2 and contrast_c4 >= 0.2:
        verdict = (f"PARTIAL_RWKV: contrast_c4={contrast_c4:+.2f}pp.")
    else:
        verdict = (f"KILL_RWKV: contrast_c4={contrast_c4:+.2f}pp / ood={contrast_ood:+.2f}pp.")
    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 161, "name": "rwkv_training_extension",
        "config": {"baseline_layers": BASELINE_LAYERS, "transport_layers": TRANSPORT_LAYERS,
                    "hidden_size": HIDDEN_SIZE, "channel_mix_hidden": CHANNEL_MIX_HIDDEN,
                    "seeds": SEEDS, "shuffle_seed": SHUFFLE_SEED, "n_train": N_TRAIN,
                    "train_steps": TRAIN_STEPS, "arm_lr": arm_lr,
                    "flop_diff_pct": float(flop_diff), "projected_hours": projected_hr},
        "results": results, "deltas": deltas,
        "contrast_c4": float(contrast_c4), "contrast_ood": float(contrast_ood),
        "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_161_rwkv_training_extension.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
