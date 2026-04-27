"""
genome_156_prefix_destruction_200m.py

CODEX FIRST-PRINCIPLES KILLER EXPERIMENT — PREFIX-INFORMATION TRANSPORT.

Pre-reg LOCKED at research/prereg/genome_156_prefix_destruction_200m_2026-04-26.md
Theory: research/derivations/prefix_information_transport.md

Falsifies (or supports) the Prefix-Information Transport Principle: the
architecture-prior win in g138-g151 is caused by superior allocation of
parameters to prefix-information transport (attention + width + residuals)
rather than to token-local nonlinear synthesis (MLP). If this is right,
destroying ordered prefix information should collapse the win.

Two stimulus conditions:
  natural:        standard c4_clean_v1
  token_shuffled: per-sequence permutation destroying prefix order
                  while preserving token marginals

Two arms (200M-class, arm-specific best LRs from g151):
  baseline_200M_4k: 14L + MLP, lr=2e-4, 4000 steps
  minimal_7L_200M_8k: 7L noMLP, lr=3e-4, 8000 steps

3 seeds: {42, 7, 13}.

Pre-stated criteria:
  PASS_TRANSPORT: delta_nat >= +0.5pp AND delta_shuf <= +0.1pp AND C := delta_nat - delta_shuf >= +0.4pp
  PARTIAL_TRANSPORT: delta_nat >= +0.3pp AND delta_shuf <= +0.2pp AND C >= +0.2pp
  KILL_TRANSPORT: |delta_nat - delta_shuf| <= 0.2pp (theory dies)

Compute: 12 runs (2 conditions x 2 arms x 3 seeds), ~1hr on RTX 5090.

Results: results/genome_156_prefix_destruction_200m.json
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
SEEDS = [42, 7, 13]
N_C4_EVAL = 200
N_TRAIN = 32768
BASELINE_STEPS = 4000
MINIMAL_STEPS = 8000
LR_WARMUP_STEPS = 200
SHUFFLE_SEED = 42

ARM_LR = {
    "baseline_200M_4k": 2e-4,
    "minimal_7L_200M_8k": 3e-4,
}


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


def shuffle_token_rows(ids, mask, shuffle_seed=SHUFFLE_SEED):
    """Per-row permutation destroying prefix order, preserving token multiset.
    Permutes only valid (mask==1) positions; pads stay at their positions.
    Same shuffle seed used for all training seeds → frozen shuffled corpus."""
    rng = np.random.default_rng(shuffle_seed)
    out_ids = ids.clone()
    n_rows, seq = ids.shape
    for r in range(n_rows):
        valid_pos = (mask[r] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        if len(valid_pos) <= 1:
            continue
        perm = rng.permutation(len(valid_pos))
        out_ids[r, valid_pos] = ids[r, valid_pos[perm]]
    return out_ids


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


def warmup_lr(step, target_lr, warmup_steps):
    if step < warmup_steps:
        return target_lr * (step + 1) / warmup_steps
    return target_lr


def train_arm(arm_name, lr_target, model, train_ids, train_mask, n_steps, seed):
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name} seed={seed} lr={lr_target}: params={n_total/1e6:.2f}M steps={n_steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr_target, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(seed)
    t_arm = time.time()
    model.train()
    n_train = train_ids.size(0)
    nan_seen = False
    log_every = 1000
    for step in range(1, n_steps + 1):
        current_lr = warmup_lr(step, lr_target, LR_WARMUP_STEPS)
        for g in opt.param_groups:
            g['lr'] = current_lr
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
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lbl.view(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            nan_seen = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"    step={step:5d} loss={loss.item():.3f} ({time.time()-t_arm:.0f}s)")
    return n_total, time.time() - t_arm, nan_seen


def main():
    t0 = time.time()
    print("genome_156: prefix-destruction killer test (Codex first-principles)")
    print(f"  conditions=[natural, token_shuffled], shuffle_seed={SHUFFLE_SEED}")
    print(f"  seeds={SEEDS}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print(f"\nLoading {N_TRAIN}+{N_C4_EVAL} c4 sequences...")
    pool_texts = []
    target_n = N_TRAIN + N_C4_EVAL
    for rec in c4_clean_v1(seed=42, n_samples=target_n):
        pool_texts.append(rec["text"])
        if len(pool_texts) >= target_n:
            break
    train_texts = pool_texts[:N_TRAIN]
    c4_eval_texts = pool_texts[N_TRAIN:N_TRAIN + N_C4_EVAL]

    print("Tokenizing train + eval...")
    enc_t = tok(train_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    train_ids_nat = enc_t["input_ids"]; train_mask = enc_t["attention_mask"]
    enc_e = tok(c4_eval_texts, padding=True, truncation=True,
                  max_length=SEQ_LEN, return_tensors="pt")
    eval_ids_nat = enc_e["input_ids"]; eval_mask = enc_e["attention_mask"]

    print("Building shuffled corpus (frozen at shuffle_seed=42)...")
    train_ids_shuf = shuffle_token_rows(train_ids_nat, train_mask, SHUFFLE_SEED)
    # Eval uses SHUFFLE_SEED+1 to avoid identical permutation across train/eval
    # (same-row identity is irrelevant since eval texts != train texts, but +1
    # documents intent and avoids the appearance of train/eval coupling).
    eval_ids_shuf = shuffle_token_rows(eval_ids_nat, eval_mask, SHUFFLE_SEED + 1)
    print(f"  train shapes: nat={tuple(train_ids_nat.shape)} shuf={tuple(train_ids_shuf.shape)}")

    # Save shuffled corpus per Codex bug-audit findings §6 (prereg artifact plan).
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"c4_shuffled_seed{SHUFFLE_SEED}_pythia_n{N_TRAIN}.pt"
    torch.save({
        "train_ids_shuf": train_ids_shuf, "train_mask": train_mask,
        "eval_ids_shuf": eval_ids_shuf, "eval_mask": eval_mask,
        "shuffle_seed_train": SHUFFLE_SEED, "shuffle_seed_eval": SHUFFLE_SEED + 1,
    }, cache_path)
    print(f"  saved shuffled corpus -> {cache_path.relative_to(ROOT)}")

    # Pre-flight integrity audit (Codex bug-audit checklist items 4 & 5):
    # token-multiset preservation per row + global frequency equality
    print("Running pre-flight shuffle audit...")
    n_audit = min(100, train_ids_nat.shape[0])
    for r in range(n_audit):
        valid_pos = (train_mask[r] == 1).nonzero(as_tuple=True)[0]
        nat_multiset = sorted(train_ids_nat[r, valid_pos].tolist())
        shuf_multiset = sorted(train_ids_shuf[r, valid_pos].tolist())
        assert nat_multiset == shuf_multiset, f"shuffle multiset mismatch at row {r}"
    nat_hist = torch.bincount(train_ids_nat[train_mask == 1], minlength=actual_vocab)
    shuf_hist = torch.bincount(train_ids_shuf[train_mask == 1], minlength=actual_vocab)
    assert torch.equal(nat_hist, shuf_hist), "global token frequency mismatch"
    print(f"  audit passed: per-row multiset + global frequency equal across {n_audit}/{train_ids_nat.shape[0]} rows")

    arms = [
        ("baseline_200M_4k", dict(hidden=1024, layers=14, heads=16, ffn=2304, no_mlp=False), BASELINE_STEPS),
        ("minimal_7L_200M_8k", dict(hidden=1024, layers=7, heads=16, ffn=2304, no_mlp=True), MINIMAL_STEPS),
    ]
    conditions = [
        ("natural", train_ids_nat, train_mask, eval_ids_nat, eval_mask),
        ("token_shuffled", train_ids_shuf, train_mask, eval_ids_shuf, eval_mask),
    ]

    # Final checkpoints saved for downstream g157 layerwise probe (hygiene per
    # post_g156_experimental_program.md). Prereg locks hypothesis/criteria/thresholds,
    # not artifact set; saving final weights does not change protocol.
    ckpt_dir = ROOT / "results" / "genome_156_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = {}  # results[condition][arm][seed] = {nll, top1_acc, top5_acc, nan_seen, wallclock_s}
    for cond_name, t_ids, t_mask, e_ids, e_mask in conditions:
        results[cond_name] = {}
        for arm_name, kw, n_steps in arms:
            results[cond_name][arm_name] = {}
            lr = ARM_LR[arm_name]
            for seed in SEEDS:
                print(f"\n=== cond={cond_name} arm={arm_name} seed={seed} ===")
                model = make_llama(actual_vocab, seed=seed, **kw)
                n_total, elapsed, nan_seen = train_arm(arm_name, lr, model, t_ids, t_mask, n_steps, seed)
                if nan_seen:
                    metrics = {"nll": float("nan"), "top1_acc": float("nan"), "top5_acc": float("nan")}
                else:
                    metrics = measure_full(model, e_ids, e_mask)
                    print(f"    eval: top1={100*metrics['top1_acc']:.2f}% nll={metrics['nll']:.3f}")
                metrics["nan_seen"] = nan_seen
                metrics["wallclock_s"] = elapsed
                metrics["params_M"] = n_total / 1e6
                # Save final checkpoint for g157 downstream probe
                ckpt_path = ckpt_dir / f"{cond_name}__{arm_name}__seed{seed}.pt"
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "cond": cond_name, "arm": arm_name, "seed": seed,
                            "config": {**kw, "lr": lr, "n_steps": n_steps,
                                        "vocab_size": actual_vocab},
                            "metrics": metrics},
                            ckpt_path)
                metrics["checkpoint_path"] = str(ckpt_path.relative_to(ROOT))
                results[cond_name][arm_name][seed] = metrics
                del model; torch.cuda.empty_cache()

    # Analysis
    print(f"\n=== ANALYSIS ===")
    summary = {}
    for cond_name in [c[0] for c in conditions]:
        summary[cond_name] = {}
        for arm_name in [a[0] for a in arms]:
            tops = [results[cond_name][arm_name][s]["top1_acc"]
                    for s in SEEDS
                    if not results[cond_name][arm_name][s]["nan_seen"]
                    and np.isfinite(results[cond_name][arm_name][s]["top1_acc"])]
            if not tops:
                summary[cond_name][arm_name] = {"top1_mean": float("nan"), "top1_std": float("nan"), "n": 0}
            else:
                summary[cond_name][arm_name] = {
                    "top1_mean": float(np.mean(tops)),
                    "top1_std": float(np.std(tops)),
                    "n": len(tops),
                }

    # Codex bug-audit Severity-8 fix: do not emit a verdict if any cell is incomplete.
    required_n = len(SEEDS)
    incomplete = [
        (cond_name, arm_name, summary[cond_name][arm_name]["n"])
        for cond_name in summary
        for arm_name in summary[cond_name]
        if summary[cond_name][arm_name]["n"] != required_n
    ]
    if incomplete:
        raise RuntimeError(f"Incomplete g156 run; missing valid seeds: {incomplete}")

    delta_nat = (summary["natural"]["minimal_7L_200M_8k"]["top1_mean"]
                 - summary["natural"]["baseline_200M_4k"]["top1_mean"]) * 100
    delta_shuf = (summary["token_shuffled"]["minimal_7L_200M_8k"]["top1_mean"]
                  - summary["token_shuffled"]["baseline_200M_4k"]["top1_mean"]) * 100
    C = delta_nat - delta_shuf
    # Codex bug-audit Severity-7 fix: at 200M, per-seed C4-gap std ~ 0.35pp,
    # so n=3 yields SE_mean ~ 0.20pp on each delta_. If |C| or any delta_ lands within
    # 0.2pp of a threshold, the result is at noise floor — flag for promotion
    # to 5 seeds before treating as decisive.
    pp_se_estimate = 0.20
    near_threshold = (abs(C) < 0.30 or abs(delta_nat - 0.5) < pp_se_estimate
                      or abs(delta_nat - 0.3) < pp_se_estimate
                      or abs(delta_shuf - 0.1) < pp_se_estimate
                      or abs(delta_shuf - 0.2) < pp_se_estimate)

    print(f"  natural:        baseline {100*summary['natural']['baseline_200M_4k']['top1_mean']:.2f}% "
          f"vs minimal {100*summary['natural']['minimal_7L_200M_8k']['top1_mean']:.2f}% "
          f"(delta_nat = {delta_nat:+.2f}pp)")
    print(f"  token_shuffled: baseline {100*summary['token_shuffled']['baseline_200M_4k']['top1_mean']:.2f}% "
          f"vs minimal {100*summary['token_shuffled']['minimal_7L_200M_8k']['top1_mean']:.2f}% "
          f"(delta_shuf = {delta_shuf:+.2f}pp)")
    print(f"  C = delta_nat - delta_shuf = {C:+.2f}pp")

    if delta_nat >= 0.5 and delta_shuf <= 0.1 and C >= 0.4:
        verdict = (f"PASS_TRANSPORT: delta_nat={delta_nat:+.2f}pp >= +0.5pp AND "
                   f"delta_shuf={delta_shuf:+.2f}pp <= +0.1pp AND C={C:+.2f}pp >= +0.4pp. "
                   f"Prefix-Information Transport theory SUPPORTED.")
    elif delta_nat >= 0.3 and delta_shuf <= 0.2 and C >= 0.2:
        verdict = (f"PARTIAL_TRANSPORT: delta_nat={delta_nat:+.2f}pp delta_shuf={delta_shuf:+.2f}pp "
                   f"C={C:+.2f}pp. Direction consistent with theory but signal weaker than expected.")
    elif abs(delta_nat - delta_shuf) <= 0.2:
        verdict = (f"KILL_TRANSPORT: |delta_nat - delta_shuf|={abs(delta_nat - delta_shuf):.2f}pp <= 0.2pp. "
                   f"Architecture-prior win does NOT depend on prefix order. Transport theory FALSIFIED.")
    else:
        verdict = (f"AMBIGUOUS: delta_nat={delta_nat:+.2f}pp delta_shuf={delta_shuf:+.2f}pp C={C:+.2f}pp.")

    if near_threshold:
        verdict = "PROVISIONAL_" + verdict.split(":", 1)[0] + (
            f" — RESULT IS NOISE-FLOOR-ADJACENT (Codex bug-audit Severity-7). "
            f"|C|={abs(C):.2f}pp, SE_mean per delta_ ≈ {pp_se_estimate:.2f}pp at n=3 seeds. "
            f"Promote to 5 seeds before treating as decisive: "
            + verdict.split(":", 1)[1]
        )

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 156, "name": "prefix_destruction_200m",
        "config": {"seeds": SEEDS, "shuffle_seed_train": SHUFFLE_SEED,
                    "shuffle_seed_eval": SHUFFLE_SEED + 1,
                    "warmup_steps": LR_WARMUP_STEPS, "n_train": N_TRAIN,
                    "baseline_steps": BASELINE_STEPS, "minimal_steps": MINIMAL_STEPS,
                    "arm_lr": ARM_LR},
        "results": results, "summary": summary,
        "delta_nat_pp": delta_nat, "delta_shuf_pp": delta_shuf, "C_pp": C,
        "near_threshold": near_threshold,
        "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_156_prefix_destruction_200m.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
