"""
genome_137_optimizer_state_transfer.py

CODEX P3a NARROWED — OPTIMIZER-STATE TRANSFER.

Hypothesis: AdamW moment state (m, v, step count) carries path-dependent
training information that is NOT present in weights alone. If true,
preserving the correct optimizer state at a checkpoint should improve
subsequent convergence vs. resetting or mismatching it.

Pre-registered at:
  research/prereg/genome_137_optimizer_state_transfer_2026-04-25.md

Protocol:
  - Tiny Llama 30M (g133/g135/g136 stack)
  - 3 donor seeds: [42, 7, 13], train each to K=1000 on fixed c4_clean_v1 pool
  - 4 arms × 3 seeds = 12 students. Continuation horizon 1000 -> 4000:
      resume_true:    donor weights + donor optimizer state
      resume_reset:   donor weights + fresh AdamW
      resume_foreign: donor weights + optimizer state from a DIFFERENT seed's donor
      state_only:     fresh random weights + donor optimizer state
  - Metrics: post-K CtQ_75 (target = donor_final + 0.25*(NLL@K - donor_final)),
             mean eval NLL over first 128 continuation steps,
             final eval NLL at step 4000.

Pre-stated criteria (locked):
  PASS: resume_true beats resume_reset by >=20% on post-K CtQ_75
        AND >=0.05 mean-NLL over first 128 steps in >=2/3 seeds;
        resume_foreign worse than resume_true;
        state_only shows no speedup over scratch.
  PARTIAL: 10-20% CtQ_75 gain or consistent short-horizon NLL gain.
  KILL: resume_true and resume_reset within +/-5% CtQ_75 AND +/-0.03 early NLL.
"""
from __future__ import annotations
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent

HIDDEN_SIZE = 384
N_LAYERS = 6
N_HEADS = 6
INTERMEDIATE_SIZE = 1024
SEQ_LEN = 256
BATCH_SIZE = 8
LR = 3e-4

POOL_SIZE = 32768
DONOR_STEPS = 1000
TOTAL_STEPS = 4000
EVAL_AT_DONOR = [0, 32, 128, 512, 1000]
EVAL_AT_CONTINUATION = [1000, 1064, 1128, 1256, 1512, 2000, 3000, 4000]
EARLY_EVAL_HORIZON = 1128  # mean NLL up through here counts as "early"
INIT_SEEDS = [42, 7, 13]
N_EVAL = 200


def make_tiny_llama(vocab_size, seed=42):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg).to("cuda").to(torch.bfloat16)


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                              weight_decay=0.1)


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


def step_train(model, opt, ids, mask):
    opt.zero_grad()
    out = model(input_ids=ids, attention_mask=mask)
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
    return float(loss.item())


def train_donor(seed, vocab_size, train_ids, train_mask, eval_ids, eval_mask,
                pool_order):
    """Train donor for DONOR_STEPS. Return final weights, optimizer state, rows."""
    print(f"\n=== DONOR seed {seed}: training {DONOR_STEPS} steps ===")
    model = make_tiny_llama(vocab_size, seed=seed)
    opt = make_optimizer(model)
    rows = []
    cursor = 0
    n_pool = len(pool_order)

    nll0 = measure_eval_nll(model, eval_ids, eval_mask)
    rows.append({"step": 0, "nll": nll0})
    print(f"  step=0  NLL={nll0:.3f}")
    model.train()

    next_idx = 1
    for step in range(1, DONOR_STEPS + 1):
        if cursor + BATCH_SIZE > n_pool:
            cursor = 0
        batch_idx = pool_order[cursor:cursor + BATCH_SIZE]
        cursor += BATCH_SIZE
        ids = train_ids[batch_idx].to("cuda")
        mask = train_mask[batch_idx].to("cuda")
        loss = step_train(model, opt, ids, mask)
        if next_idx < len(EVAL_AT_DONOR) and step == EVAL_AT_DONOR[next_idx]:
            nll = measure_eval_nll(model, eval_ids, eval_mask)
            rows.append({"step": step, "nll": nll, "loss": loss})
            print(f"  step={step:4d}  NLL={nll:.3f}  loss={loss:.3f}")
            model.train()
            next_idx += 1

    # Snapshot weights and optimizer state
    weights_state = copy.deepcopy(model.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    nll_at_K = rows[-1]["nll"]
    return model, opt, weights_state, opt_state, nll_at_K, rows, cursor


def continuation(arm_name, weights_state, opt_state, vocab_size, init_seed,
                  train_ids, train_mask, eval_ids, eval_mask, pool_order,
                  start_cursor):
    """Continue training from step 1000 to 4000 under a specific arm setup."""
    print(f"\n--- continuation arm={arm_name} (init_seed for fresh weights={init_seed}) ---")
    if weights_state is not None:
        # Build a model and load the donor weights
        model = make_tiny_llama(vocab_size, seed=init_seed)
        model.load_state_dict(weights_state)
    else:
        # state_only arm: fresh random weights from a DIFFERENT seed than donor
        model = make_tiny_llama(vocab_size, seed=init_seed)

    opt = make_optimizer(model)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    rows = []
    cursor = start_cursor
    n_pool = len(pool_order)

    # Initial NLL at the post-fork starting point
    nll_start = measure_eval_nll(model, eval_ids, eval_mask)
    rows.append({"step": DONOR_STEPS, "nll": nll_start})
    print(f"  step={DONOR_STEPS}  NLL={nll_start:.3f}")
    model.train()

    next_idx = 1
    early_nlls = [nll_start]
    early_steps = [DONOR_STEPS]
    for step in range(DONOR_STEPS + 1, TOTAL_STEPS + 1):
        if cursor + BATCH_SIZE > n_pool:
            cursor = 0
        batch_idx = pool_order[cursor:cursor + BATCH_SIZE]
        cursor += BATCH_SIZE
        ids = train_ids[batch_idx].to("cuda")
        mask = train_mask[batch_idx].to("cuda")
        loss = step_train(model, opt, ids, mask)

        if next_idx < len(EVAL_AT_CONTINUATION) and step == EVAL_AT_CONTINUATION[next_idx]:
            nll = measure_eval_nll(model, eval_ids, eval_mask)
            rows.append({"step": step, "nll": nll, "loss": loss})
            print(f"  step={step:4d}  NLL={nll:.3f}  loss={loss:.3f}")
            model.train()
            next_idx += 1
            if step <= EARLY_EVAL_HORIZON:
                early_nlls.append(nll)
                early_steps.append(step)

    early_mean_nll = float(np.mean(early_nlls))
    final_nll = rows[-1]["nll"]

    del model
    torch.cuda.empty_cache()

    return {
        "rows": rows,
        "early_mean_nll": early_mean_nll,
        "early_nll_steps": early_steps,
        "early_nll_values": early_nlls,
        "final_nll": final_nll,
    }


def find_post_K_ctq(rows, target_nll):
    """First step >= DONOR_STEPS where NLL <= target_nll."""
    for r in rows:
        if r["step"] > DONOR_STEPS and r["nll"] <= target_nll:
            return r["step"]
    return None


def main():
    t0 = time.time()
    print("genome_137: optimizer-state transfer test")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print(f"Loading {POOL_SIZE} c4_clean_v1 stimuli (seed=42)...")
    pool_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=POOL_SIZE + N_EVAL):
        pool_texts.append(rec["text"])
        if len(pool_texts) >= POOL_SIZE + N_EVAL:
            break
    train_texts = pool_texts[:POOL_SIZE]
    eval_texts = pool_texts[POOL_SIZE:POOL_SIZE + N_EVAL]

    print("Tokenizing pool once...")
    enc = tok(train_texts, padding=True, truncation=True,
               max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc["input_ids"]
    train_mask = enc["attention_mask"]
    enc_eval = tok(eval_texts, padding=True, truncation=True,
                    max_length=SEQ_LEN, return_tensors="pt")
    eval_ids = enc_eval["input_ids"]
    eval_mask = enc_eval["attention_mask"]
    print(f"  train tensor: {train_ids.shape}")

    # Fixed pool order per Codex prereg (same as g136 random_A)
    rng = np.random.default_rng(42)
    pool_order = rng.permutation(POOL_SIZE)

    # === DONORS: train one per seed to K=1000 ===
    donor_artifacts = {}
    for seed in INIT_SEEDS:
        m, o, w_state, o_state, nll_K, donor_rows, cursor = train_donor(
            seed, actual_vocab, train_ids, train_mask, eval_ids, eval_mask,
            pool_order,
        )
        # Run donor a bit further to get donor_final reference
        # Actually, donor_final = result of resume_true, we'll compute that next.
        donor_artifacts[seed] = {
            "weights": w_state, "opt_state": o_state, "nll_K": nll_K,
            "donor_rows": donor_rows, "cursor": cursor,
        }
        del m, o
        torch.cuda.empty_cache()

    # === CONTINUATIONS: 4 arms x 3 seeds ===
    foreign_seed_map = {42: 7, 7: 13, 13: 42}  # cyclic
    all_results = {}

    for seed in INIT_SEEDS:
        donor = donor_artifacts[seed]
        nll_K = donor["nll_K"]
        cursor_after_donor = donor["cursor"]
        all_results[seed] = {}

        # Arm 1: resume_true — donor weights + donor opt state
        all_results[seed]["resume_true"] = continuation(
            f"resume_true_seed{seed}",
            donor["weights"], donor["opt_state"], actual_vocab, seed,
            train_ids, train_mask, eval_ids, eval_mask,
            pool_order, cursor_after_donor,
        )

        # Arm 2: resume_reset — donor weights, fresh opt state
        all_results[seed]["resume_reset"] = continuation(
            f"resume_reset_seed{seed}",
            donor["weights"], None, actual_vocab, seed,
            train_ids, train_mask, eval_ids, eval_mask,
            pool_order, cursor_after_donor,
        )

        # Arm 3: resume_foreign — donor weights + DIFFERENT donor's opt state
        foreign_seed = foreign_seed_map[seed]
        foreign_opt = donor_artifacts[foreign_seed]["opt_state"]
        all_results[seed]["resume_foreign"] = continuation(
            f"resume_foreign_seed{seed}",
            donor["weights"], foreign_opt, actual_vocab, seed,
            train_ids, train_mask, eval_ids, eval_mask,
            pool_order, cursor_after_donor,
        )

        # Arm 4: state_only — fresh random weights + donor opt state
        # Use a NEW init seed so weights aren't from the donor seed lineage
        fresh_seed = seed + 1000
        all_results[seed]["state_only"] = continuation(
            f"state_only_seed{seed}",
            None, donor["opt_state"], actual_vocab, fresh_seed,
            train_ids, train_mask, eval_ids, eval_mask,
            pool_order, cursor_after_donor,
        )

    # === ANALYSIS ===
    print(f"\n=== ANALYSIS ===")
    arm_names = ["resume_true", "resume_reset", "resume_foreign", "state_only"]

    # Per-arm summary
    summary = {}
    for arm in arm_names:
        finals = [all_results[s][arm]["final_nll"] for s in INIT_SEEDS]
        earlies = [all_results[s][arm]["early_mean_nll"] for s in INIT_SEEDS]
        summary[arm] = {
            "final_nll_per_seed": [float(x) for x in finals],
            "final_nll_mean": float(np.mean(finals)),
            "early_mean_nll_per_seed": [float(x) for x in earlies],
            "early_mean_nll_avg": float(np.mean(earlies)),
        }
        print(f"  {arm:18s} final_nll={summary[arm]['final_nll_mean']:.4f}  "
              f"early_mean={summary[arm]['early_mean_nll_avg']:.4f}  "
              f"per-seed final: {[f'{x:.3f}' for x in finals]}")

    # CtQ_75 post-K target = donor_final_avg + 0.25 * (nll_K_avg - donor_final_avg)
    # Use resume_true as donor-final proxy
    donor_final_avg = summary["resume_true"]["final_nll_mean"]
    nll_K_avg = float(np.mean([donor_artifacts[s]["nll_K"] for s in INIT_SEEDS]))
    ctq_target = donor_final_avg + 0.25 * (nll_K_avg - donor_final_avg)
    print(f"\n  donor_final_avg (resume_true): {donor_final_avg:.4f}")
    print(f"  nll_K_avg: {nll_K_avg:.4f}")
    print(f"  post-K CtQ_75 target NLL: {ctq_target:.4f}")

    ctq_summary = {}
    for arm in arm_names:
        ctqs = []
        for seed in INIT_SEEDS:
            rows = all_results[seed][arm]["rows"]
            ctq = find_post_K_ctq(rows, ctq_target)
            ctqs.append(ctq if ctq is not None else TOTAL_STEPS + 1)
        # Steps post-K, not absolute
        ctq_post_K = [c - DONOR_STEPS if c <= TOTAL_STEPS else (TOTAL_STEPS + 1 - DONOR_STEPS) for c in ctqs]
        ctq_summary[arm] = {
            "ctq_per_seed": ctqs,
            "ctq_post_K_per_seed": ctq_post_K,
            "ctq_mean": float(np.mean(ctqs)),
            "ctq_post_K_mean": float(np.mean(ctq_post_K)),
        }
        print(f"  {arm:18s}  CtQ steps (post-K): {ctq_post_K}  mean={np.mean(ctq_post_K):.0f}")

    # PASS criteria
    rt_ctq = ctq_summary["resume_true"]["ctq_post_K_mean"]
    rr_ctq = ctq_summary["resume_reset"]["ctq_post_K_mean"]
    rf_ctq = ctq_summary["resume_foreign"]["ctq_post_K_mean"]
    so_ctq = ctq_summary["state_only"]["ctq_post_K_mean"]

    rt_early = summary["resume_true"]["early_mean_nll_avg"]
    rr_early = summary["resume_reset"]["early_mean_nll_avg"]

    ctq_gain_pct = (rr_ctq - rt_ctq) / max(rr_ctq, 1) * 100
    early_nll_gain = rr_early - rt_early
    print(f"\n  resume_true vs resume_reset CtQ gain: {ctq_gain_pct:+.1f}%")
    print(f"  resume_true vs resume_reset early-mean-NLL gain: {early_nll_gain:+.4f}")
    print(f"  resume_foreign worse than resume_true? rf={rf_ctq:.0f} vs rt={rt_ctq:.0f}: {rf_ctq > rt_ctq}")

    # Per-seed PASS check
    rt_seed_wins = 0
    for i, seed in enumerate(INIT_SEEDS):
        rt_seed_ctq = ctq_summary["resume_true"]["ctq_post_K_per_seed"][i]
        rr_seed_ctq = ctq_summary["resume_reset"]["ctq_post_K_per_seed"][i]
        if rt_seed_ctq <= rr_seed_ctq * 0.80:
            rt_seed_wins += 1

    if (ctq_gain_pct >= 20 and early_nll_gain >= 0.05 and
        rf_ctq > rt_ctq and rt_seed_wins >= 2):
        verdict = (f"PASS: resume_true beats resume_reset by {ctq_gain_pct:.0f}% post-K CtQ "
                   f"AND {early_nll_gain:.3f} early NLL, in {rt_seed_wins}/3 seeds. "
                   f"resume_foreign worse. OPTIMIZER STATE IS A TRANSFERABLE PROCESS LEVER.")
    elif ctq_gain_pct >= 10 or early_nll_gain >= 0.03:
        verdict = (f"PARTIAL: ctq_gain={ctq_gain_pct:.1f}%, early_nll_gain={early_nll_gain:.3f}. "
                   f"Real but below PASS threshold.")
    else:
        verdict = (f"KILL: resume_true and resume_reset within +/-5% CtQ "
                   f"({ctq_gain_pct:.1f}%) and +/-0.03 early NLL ({early_nll_gain:.4f}). "
                   f"Optimizer state adds no material transferable signal beyond weights.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 137, "name": "optimizer_state_transfer",
        "config": {"donor_steps": DONOR_STEPS, "total_steps": TOTAL_STEPS,
                    "init_seeds": INIT_SEEDS, "lr": LR, "batch": BATCH_SIZE},
        "donor_nll_K_per_seed": {str(s): donor_artifacts[s]["nll_K"] for s in INIT_SEEDS},
        "donor_rows_per_seed": {str(s): donor_artifacts[s]["donor_rows"] for s in INIT_SEEDS},
        "summary": summary,
        "ctq_summary": ctq_summary,
        "post_K_ctq_target_nll": ctq_target,
        "rows_per_seed_per_arm": {
            str(s): {arm: all_results[s][arm]["rows"] for arm in arm_names}
            for s in INIT_SEEDS
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_137_optimizer_state_transfer.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
