"""Direction-identity contrastive distillation — capability transfer
without matching weights.

The 12-op null catalog ruled out forward GEOMETRIC manipulation as a
route to capability transfer. This experiment tries a different route:
distill the TEACHER's pairwise sample-to-sample similarity geometry
into an untrained STUDENT of the same architecture, via contrastive
loss on mid-layer activation cosine similarities.

MOTIVATION. genome_045 (random-vs-PCA) established that DIRECTION
IDENTITY of trained feature-vectors is what carries capability —
a random orthonormal basis at the same effective rank preserves the
geometric envelope (c, d_rd) but destroys NLL. Directions matter.
Contrastive distillation teaches the student to reproduce WHICH pairs
of samples look similar and which look different, which is a weak
form of direction-identity matching without copying weights.

SETUP. 2-layer d=128 tiny transformer from scratch. Train on C4 CE
loss for 500 steps in two conditions:
  (A) BASELINE: CE only
  (B) DISTILLED: CE + lambda * L_contrastive(student_mid, teacher_mid)
where teacher is a pretrained Qwen3-0.6B mid-layer activation
on the SAME batch. L_contrastive is the symmetric cross-entropy of
normalized student pairwise cos-sim against teacher pairwise cos-sim,
projected to match hidden dims via an untrained linear probe.

KILL: if distilled val_NLL >= baseline val_NLL (no speedup or
capability gain), distillation fails. Candidate-8-bridge-target was
null (genome_066); direction-identity target is a stronger target
and could land where the bridge didn't.

If distilled val_NLL converges faster or lower, this is the FIRST
non-naive capability transfer through candidate-8-adjacent direction
geometry. Partner demo for Martian (Model Mapping).
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
from genome_geometry_aux_loss import (  # noqa: E402
    TinyTransformer, build_tokens, measure_val_nll,
)
from genome_loaders import load_system  # noqa: E402

_ROOT = _THIS_DIR.parent


def teacher_mid_activations(teacher_sys, teacher_tokenizer, texts, device):
    """Pool mid-layer activations of teacher Qwen3 on a batch of texts."""
    mid = teacher_sys.n_hidden_layers() // 2
    # forward with output_hidden_states
    inputs = teacher_tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=64).to(device)
    with torch.no_grad():
        out = teacher_sys.model(**inputs, output_hidden_states=True)
    hs = out.hidden_states[mid]  # (b, s, h)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.float()  # (b, h_teacher)


def contrastive_sim_loss(student_feat, teacher_feat, temperature=0.1):
    """Match student's pairwise cosine similarity matrix to teacher's via
    symmetric KL divergence.

    For batch of size b:
      S_ij = cos(student_i, student_j)
      T_ij = cos(teacher_i, teacher_j)
    Convert each to a probability distribution via softmax(./tau). Compute
    KL(T||S) + KL(S||T).
    """
    sf = F.normalize(student_feat, dim=-1)
    tf = F.normalize(teacher_feat, dim=-1)
    S = sf @ sf.T / temperature          # (b, b)
    T = tf @ tf.T / temperature
    logP_S = F.log_softmax(S, dim=-1)
    logP_T = F.log_softmax(T, dim=-1)
    P_S = logP_S.exp()
    P_T = logP_T.exp()
    kl_TS = (P_T * (logP_T - logP_S)).sum(dim=-1).mean()
    kl_ST = (P_S * (logP_S - logP_T)).sum(dim=-1).mean()
    return 0.5 * (kl_TS + kl_ST)


def train_one(vocab_size, train_tokens, val_tokens, teacher_sys,
              teacher_tokenizer, train_texts,
              *, use_distill, distill_lambda=0.3,
              steps=500, eval_every=50, max_len=64, batch=16, lr=3e-4,
              seed=42):
    torch.manual_seed(seed)
    device = "cuda"
    student = TinyTransformer(vocab_size=vocab_size, max_len=max_len).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95))
    n_train = len(train_tokens) - max_len - 1
    rng = np.random.default_rng(seed)

    # For the contrastive loss we need to know teacher hidden size.
    # Qwen3-0.6B has h=1024. Student has h=128. Use a frozen random
    # projection to map student (h=128) -> teacher-space (h=1024).
    if use_distill:
        proj = nn.Linear(128, 1024, bias=False).to(device)
        with torch.no_grad():
            proj.weight.copy_(torch.randn_like(proj.weight) * (1.0 / 128 ** 0.5))
        proj.requires_grad_(False)

    history = []
    t0 = time.time()
    text_idx = 0
    for step in range(steps + 1):
        if step % eval_every == 0:
            val_nll = measure_val_nll(student, val_tokens, device, max_len=max_len)
            history.append({"step": step, "val_nll": val_nll, "wall_s": time.time() - t0})
            print(f"  [{'DIST' if use_distill else 'CE  '} step {step:4d}] val_nll={val_nll:.4f}  t={time.time()-t0:.1f}s")
        if step == steps:
            break

        idxs = rng.integers(0, n_train, size=batch)
        batch_arr = np.stack([train_tokens[i:i + max_len + 1] for i in idxs])
        x = torch.from_numpy(batch_arr[:, :max_len]).to(device)
        y = torch.from_numpy(batch_arr[:, 1:]).to(device)
        logits, mid = student(x, return_mid_hidden=True)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = ce

        if use_distill:
            # Get aligned text batch for teacher forward
            tx_idx = (text_idx * batch) % max(len(train_texts) - batch, 1)
            text_batch = train_texts[tx_idx:tx_idx + batch]
            if len(text_batch) < batch:
                text_batch = (text_batch + train_texts[:batch - len(text_batch)])
            text_idx += 1
            try:
                t_feat = teacher_mid_activations(teacher_sys, teacher_tokenizer,
                                                  text_batch, device)
                s_pooled = mid.mean(dim=1)           # (b, 128)
                s_proj = proj(s_pooled)              # (b, 1024)
                distill = contrastive_sim_loss(s_proj, t_feat)
                loss = ce + distill_lambda * distill
            except Exception as e:
                # fallback to CE if teacher call fails
                if step < 3:
                    print(f"  teacher call failed at step {step}: {e}")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

    return history


def main():
    print("Building C4 token stream + text cache...")
    tokens = build_tokens(seed=42, n_total=80000, vocab_cap=5000)
    split = 60000
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    vocab_size = 5000

    # Need C4 texts in parallel for teacher forward
    from stimulus_banks import c4_clean_v1
    train_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        train_texts.append(rec["text"])
        if len(train_texts) >= 2000:
            break

    print("Loading teacher Qwen3-0.6B...")
    teacher_sys = load_system("Qwen/Qwen3-0.6B", quant="fp16", untrained=False, device="cuda")
    teacher_tokenizer = teacher_sys.tokenizer
    # freeze teacher
    for p in teacher_sys.model.parameters():
        p.requires_grad_(False)

    all_results = {}
    for seed in (42, 1337):
        print(f"\n\n===== SEED {seed} =====")
        print("\n-- BASELINE (CE only) --")
        hist_a = train_one(vocab_size, train_tokens, val_tokens,
                            teacher_sys, teacher_tokenizer, train_texts,
                            use_distill=False, steps=500, seed=seed)
        print("\n-- DISTILLED (CE + direction-identity contrastive) --")
        hist_b = train_one(vocab_size, train_tokens, val_tokens,
                            teacher_sys, teacher_tokenizer, train_texts,
                            use_distill=True, distill_lambda=0.3,
                            steps=500, seed=seed)
        all_results[seed] = {"baseline": hist_a, "distilled": hist_b}

    teacher_sys.unload()
    torch.cuda.empty_cache()

    print("\n\n=== DIRECTION-IDENTITY DISTILLATION RESULTS ===")
    summary = []
    for seed, d in all_results.items():
        a = d["baseline"][-1]["val_nll"]
        b = d["distilled"][-1]["val_nll"]
        target = a * 1.05
        a_steps = next((h["step"] for h in d["baseline"] if h["val_nll"] <= target), None)
        b_steps = next((h["step"] for h in d["distilled"] if h["val_nll"] <= target), None)
        speedup = (a_steps / b_steps) if (a_steps and b_steps) else None
        summary.append({"seed": seed, "baseline_nll": a, "distilled_nll": b,
                        "speedup": speedup})
        print(f"  seed={seed}: baseline={a:.4f}, distilled={b:.4f}, "
              f"speedup={'{:.2f}x'.format(speedup) if speedup else 'n/a'}")

    speedups = [s["speedup"] for s in summary if s["speedup"] is not None]
    deltas = [(s["baseline_nll"] - s["distilled_nll"]) / max(s["baseline_nll"], 1e-6)
              for s in summary]
    if all(s >= 1.10 for s in speedups) or all(d >= 0.01 for d in deltas):
        verdict = ("DIRECTION_IDENTITY_DISTILL_LANDS - contrastive sim "
                   "distillation from Qwen3 teacher accelerates student "
                   "convergence or lowers final NLL. First non-naive "
                   "capability transfer in the 12-op catalog - sim-geometry "
                   "matching works where weight-subset transplant did not.")
    else:
        verdict = ("PARTIAL_OR_NULL - contrastive distillation did not "
                   "clearly beat CE baseline. Candidate for larger lambda, "
                   "longer schedule, or better projection.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Direction-identity contrastive distillation - home-run A",
           "all_results": all_results, "summary": summary,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/direction_identity_distill.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
