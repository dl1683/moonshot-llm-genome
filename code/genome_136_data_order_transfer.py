"""
genome_136_data_order_transfer.py

CODEX P3c VERDICT after spectrum thread closed (g135 KILL).

Question: is DATA ORDERING a transferable process object that affects
capability, when architecture / init / optimizer / data multiset are
held fixed?

If YES: the first transferable process-level lever for capability.
If NO: data ordering is irrelevant; narrow P3a to gradient/optimizer state.

Protocol:
  - Tiny Llama 30M (same as g133/g135 stack)
  - Fixed pool of 32768 sequences from c4_clean_v1(seed=42)
  - Tokenize once, reuse exact same pool across all arms
  - Warm a DONOR model on random_A order for 512 steps; freeze; score
    every sequence by donor token-NLL (the difficulty score)
  - 4 student arms, 3 init seeds each:
      A) random_A (control 1)
      B) random_B (control 2 — different shuffle, same data)
      C) easy_to_hard (sorted ascending by donor NLL)
      D) hard_to_easy (sorted descending)
  - Each student trains 4000 steps from IDENTICAL random init (seed)
  - Metric: CtQ_75 and final validation NLL on held-out 200 sequences

Pre-stated criteria:
  PASS: easy_to_hard beats BOTH random_A and random_B controls by
        >=20% CtQ_75 AND >=0.15 final-NLL in >=2/3 seeds, AND
        hard_to_easy is worse than easy_to_hard.
  PARTIAL: >=10% CtQ_75 gain OR >=0.08 final-NLL gain (not both).
  KILL: all orders within ±5% CtQ_75 AND ±0.05 final NLL — data
        ordering is not a hidden lever; narrow P3a to optimizer/grad.

Compute estimate: 4 arms × 3 seeds × 4000 steps + donor warmup + scoring
≈ 30-45 min on RTX 5090.

Results: results/genome_136_data_order_transfer.json
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
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent

# Llama config (same as g133/g135)
HIDDEN_SIZE = 384
N_LAYERS = 6
N_HEADS = 6
INTERMEDIATE_SIZE = 1024
SEQ_LEN = 256
BATCH_SIZE = 8
LR = 3e-4

# Experiment config
POOL_SIZE = 32768  # total sequences in the fixed pool
DONOR_WARMUP_STEPS = 512
STUDENT_STEPS = 4000
EVAL_AT = [0, 32, 128, 512, 1000, 2000, 4000]
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


def measure_per_seq_nll(model, ids, mask):
    """Compute per-sequence NLL (mean over valid tokens). Returns 1D array."""
    model.eval()
    nlls = np.zeros(ids.size(0), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, ids.size(0), BATCH_SIZE):
            b_ids = ids[i:i+BATCH_SIZE].to("cuda")
            b_mask = mask[i:i+BATCH_SIZE].to("cuda")
            out = model(input_ids=b_ids, attention_mask=b_mask, use_cache=False)
            logits = out.logits
            sl = logits[:, :-1].contiguous()
            lbl = b_ids[:, 1:].contiguous().clone()
            sm = b_mask[:, 1:].contiguous()
            lbl[sm == 0] = -100
            # per-sequence loss
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            flat_loss = loss_fn(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1))
            seq_loss = flat_loss.view(b_ids.size(0), -1)
            seq_mask = (lbl != -100).float()
            seq_loss_sum = (seq_loss * seq_mask).sum(dim=1)
            seq_token_count = seq_mask.sum(dim=1).clamp(min=1)
            nll_per_seq = (seq_loss_sum / seq_token_count).cpu().float().numpy()
            nlls[i:i+b_ids.size(0)] = nll_per_seq
    model.train()
    return nlls


def measure_eval_nll(model, ids, mask):
    """Mean NLL over a small eval set."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.size(0), BATCH_SIZE):
            b_ids = ids[i:i+BATCH_SIZE].to("cuda")
            b_mask = mask[i:i+BATCH_SIZE].to("cuda")
            out = model(input_ids=b_ids, attention_mask=b_mask, use_cache=False)
            logits = out.logits
            sl = logits[:, :-1].contiguous()
            lbl = b_ids[:, 1:].contiguous().clone()
            sm = b_mask[:, 1:].contiguous()
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


def train_student(arm_name, init_seed, order_indices, train_ids, train_mask,
                  eval_ids, eval_mask, vocab_size):
    """Train a fresh student following the given order."""
    print(f"\n--- {arm_name} (seed {init_seed}) ---")
    model = make_tiny_llama(vocab_size, seed=init_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rows = []
    t_arm = time.time()

    # Step 0
    nll0 = measure_eval_nll(model, eval_ids, eval_mask)
    rows.append({"step": 0, "nll": nll0})
    print(f"  step=0  NLL={nll0:.3f}")
    model.train()

    # Cycle through order_indices, sampling batches in order
    cursor = 0
    n_seqs = len(order_indices)
    next_eval_idx = 1
    step = 0
    while step < STUDENT_STEPS:
        # Take next BATCH_SIZE indices, wrap if needed
        if cursor + BATCH_SIZE > n_seqs:
            cursor = 0
        batch_idx = order_indices[cursor:cursor + BATCH_SIZE]
        cursor += BATCH_SIZE
        ids = train_ids[batch_idx].to("cuda")
        mask = train_mask[batch_idx].to("cuda")
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
        step += 1

        if next_eval_idx < len(EVAL_AT) and step == EVAL_AT[next_eval_idx]:
            nll = measure_eval_nll(model, eval_ids, eval_mask)
            rows.append({"step": step, "nll": nll, "loss": float(loss.item())})
            print(f"  step={step:5d}  NLL={nll:.3f}  loss={loss.item():.3f}  "
                  f"({time.time()-t_arm:.0f}s)")
            model.train()
            next_eval_idx += 1

    del model
    torch.cuda.empty_cache()
    return rows


def find_ctq(rows, target_nll):
    for r in rows:
        if r["nll"] <= target_nll:
            return r["step"]
    return None


def main():
    t0 = time.time()
    print("genome_136: data-ordering transfer test")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    # Load fixed pool of POOL_SIZE sequences
    print(f"Loading {POOL_SIZE} c4_clean_v1 stimuli (seed=42)...")
    pool_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=POOL_SIZE + N_EVAL):
        pool_texts.append(rec["text"])
        if len(pool_texts) >= POOL_SIZE + N_EVAL:
            break
    train_texts = pool_texts[:POOL_SIZE]
    eval_texts = pool_texts[POOL_SIZE:POOL_SIZE + N_EVAL]
    print(f"  train pool: {len(train_texts)}, eval: {len(eval_texts)}")

    print("Tokenizing pool once (this takes a moment)...")
    enc = tok(train_texts, padding=True, truncation=True,
               max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc["input_ids"]
    train_mask = enc["attention_mask"]
    enc_eval = tok(eval_texts, padding=True, truncation=True,
                    max_length=SEQ_LEN, return_tensors="pt")
    eval_ids = enc_eval["input_ids"]
    eval_mask = enc_eval["attention_mask"]
    print(f"  train tensor: {train_ids.shape}")

    # === DONOR WARMUP ===
    print(f"\n=== DONOR WARMUP ({DONOR_WARMUP_STEPS} steps on random_A order) ===")
    rng_donor = np.random.default_rng(42)
    random_a_order = rng_donor.permutation(POOL_SIZE)
    donor = make_tiny_llama(actual_vocab, seed=42)
    opt = torch.optim.AdamW(donor.parameters(), lr=LR, betas=(0.9, 0.95),
                             weight_decay=0.1)
    cursor = 0
    for step in range(DONOR_WARMUP_STEPS):
        if cursor + BATCH_SIZE > POOL_SIZE:
            cursor = 0
        batch_idx = random_a_order[cursor:cursor + BATCH_SIZE]
        cursor += BATCH_SIZE
        ids = train_ids[batch_idx].to("cuda")
        mask = train_mask[batch_idx].to("cuda")
        opt.zero_grad()
        out = donor(input_ids=ids, attention_mask=mask)
        logits = out.logits
        sl = logits[:, :-1].contiguous()
        lbl = ids[:, 1:].contiguous().clone()
        sm = mask[:, 1:].contiguous()
        lbl[sm == 0] = -100
        loss = F.cross_entropy(
            sl.view(-1, sl.size(-1)), lbl.view(-1), ignore_index=-100
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(donor.parameters(), 1.0)
        opt.step()
    donor_eval_nll = measure_eval_nll(donor, eval_ids, eval_mask)
    print(f"  donor warmed, eval NLL = {donor_eval_nll:.3f}")

    # === SCORE EVERY SEQUENCE BY DONOR NLL ===
    print(f"Scoring all {POOL_SIZE} sequences by donor NLL...")
    seq_difficulty = measure_per_seq_nll(donor, train_ids, train_mask)
    print(f"  difficulty range: [{seq_difficulty.min():.3f}, {seq_difficulty.max():.3f}], "
          f"mean={seq_difficulty.mean():.3f}")
    del donor
    torch.cuda.empty_cache()

    # === BUILD ORDERS ===
    rng_b = np.random.default_rng(123)
    orders = {
        "random_A": random_a_order,
        "random_B": rng_b.permutation(POOL_SIZE),
        "easy_to_hard": np.argsort(seq_difficulty),
        "hard_to_easy": np.argsort(-seq_difficulty),
    }

    # === STUDENT ARMS ===
    all_results = {}
    for arm_name, order in orders.items():
        all_results[arm_name] = {"by_seed": {}, "rows_per_seed": []}
        for seed in INIT_SEEDS:
            rows = train_student(
                f"{arm_name}", seed, order, train_ids, train_mask,
                eval_ids, eval_mask, actual_vocab,
            )
            all_results[arm_name]["by_seed"][seed] = rows
            all_results[arm_name]["rows_per_seed"].append(rows)

    # === ANALYSIS ===
    # Average final NLL per arm across seeds
    print(f"\n=== ANALYSIS ===")
    arm_summary = {}
    for arm_name in orders:
        finals = [rows[-1]["nll"] for rows in all_results[arm_name]["rows_per_seed"]]
        mean_final = float(np.mean(finals))
        std_final = float(np.std(finals))
        arm_summary[arm_name] = {
            "final_nll_mean": mean_final, "final_nll_std": std_final,
            "final_nll_per_seed": [float(x) for x in finals],
        }
        print(f"  {arm_name:14s}  final NLL = {mean_final:.4f} +/- {std_final:.4f}  per-seed: {[f'{x:.3f}' for x in finals]}")

    # CtQ_75: target = step0 - 0.75 * (step0 - final_random_A)
    step0_nll = all_results["random_A"]["rows_per_seed"][0][0]["nll"]
    rand_a_final = arm_summary["random_A"]["final_nll_mean"]
    initial_gap = step0_nll - rand_a_final
    target_nll = step0_nll - 0.75 * initial_gap
    print(f"\n  step0 NLL: {step0_nll:.3f}, random_A final: {rand_a_final:.3f}")
    print(f"  CtQ_75 target NLL: {target_nll:.3f}")

    ctq_summary = {}
    for arm_name in orders:
        ctqs = []
        for rows in all_results[arm_name]["rows_per_seed"]:
            ctq = find_ctq(rows, target_nll)
            ctqs.append(ctq if ctq is not None else STUDENT_STEPS + 1)
        ctq_summary[arm_name] = {"per_seed": ctqs, "mean": float(np.mean(ctqs))}
        print(f"  {arm_name:14s}  CtQ_75: {ctqs}  mean={np.mean(ctqs):.0f}")

    # Verdict
    e2h_ctq = ctq_summary["easy_to_hard"]["mean"]
    h2e_ctq = ctq_summary["hard_to_easy"]["mean"]
    rA_ctq = ctq_summary["random_A"]["mean"]
    rB_ctq = ctq_summary["random_B"]["mean"]
    rand_ctq = (rA_ctq + rB_ctq) / 2

    e2h_final = arm_summary["easy_to_hard"]["final_nll_mean"]
    h2e_final = arm_summary["hard_to_easy"]["final_nll_mean"]
    rand_final = (arm_summary["random_A"]["final_nll_mean"] + arm_summary["random_B"]["final_nll_mean"]) / 2

    ctq_gain_pct = (rand_ctq - e2h_ctq) / max(rand_ctq, 1) * 100
    nll_gain = rand_final - e2h_final

    print(f"\n  easy_to_hard CtQ gain vs random: {ctq_gain_pct:+.1f}%")
    print(f"  easy_to_hard NLL gain vs random: {nll_gain:+.4f}")
    print(f"  hard_to_easy CtQ: {h2e_ctq:.0f}, easy_to_hard CtQ: {e2h_ctq:.0f}")

    # PASS: e2h beats BOTH controls by >=20% CtQ AND >=0.15 NLL in >=2/3 seeds AND h2e < e2h
    e2h_seed_wins = sum(1 for i in range(len(INIT_SEEDS))
                        if (ctq_summary["easy_to_hard"]["per_seed"][i] <= rA_ctq * 0.80 and
                            ctq_summary["easy_to_hard"]["per_seed"][i] <= rB_ctq * 0.80))
    e2h_better_than_h2e = e2h_ctq < h2e_ctq

    if (ctq_gain_pct >= 20 and nll_gain >= 0.15 and e2h_better_than_h2e and e2h_seed_wins >= 2):
        verdict = (f"PASS: easy_to_hard beats random by {ctq_gain_pct:.0f}% CtQ AND "
                   f"{nll_gain:.3f} final NLL, h2e<e2h, {e2h_seed_wins}/3 seeds. "
                   f"DATA ORDERING IS A TRANSFERABLE PROCESS LEVER.")
    elif ctq_gain_pct >= 10 or nll_gain >= 0.08:
        verdict = (f"PARTIAL: ctq_gain={ctq_gain_pct:.1f}%, nll_gain={nll_gain:.3f}. "
                   f"Real but below PASS threshold.")
    else:
        verdict = (f"KILL: all orders within +/-5% (ctq_gain={ctq_gain_pct:.1f}%, "
                   f"nll_gain={nll_gain:.3f}). Data ordering is not the hidden lever. "
                   f"Narrow P3a to optimizer/gradient state.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 136, "name": "data_order_transfer",
        "config": {
            "pool_size": POOL_SIZE, "donor_warmup_steps": DONOR_WARMUP_STEPS,
            "student_steps": STUDENT_STEPS, "init_seeds": INIT_SEEDS,
            "lr": LR, "batch": BATCH_SIZE,
        },
        "donor_eval_nll": donor_eval_nll,
        "difficulty_range": [float(seq_difficulty.min()), float(seq_difficulty.max())],
        "arm_summary": arm_summary,
        "ctq_summary": ctq_summary,
        "rows_per_arm_per_seed": {
            arm: {str(s): rows for s, rows in d["by_seed"].items()}
            for arm, d in all_results.items()
        },
        "target_nll_ctq75": target_nll,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_136_data_order_transfer.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
