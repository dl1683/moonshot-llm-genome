"""
genome_154_distillation_smoke.py

CODEX D1+D2 SMOKE TEST — DISTILLATION PIPELINE VALIDATION.

Strategic background: the architecture-prior thread (g138-g151) shows
minimal_3L matches/beats baseline at matched compute. The high-leverage
extension (per Codex outreach analysis) is a DISTILLATION PRODUCT: train
minimal_300M-1B student from strong teacher, ship as edge inference
server with superior quality-per-joule.

This experiment validates the full distillation pipeline at small scale
before committing to production 7B-teacher run:

  Teacher: Qwen3-0.6B (well-trained, ~600M params, fits in VRAM)
  Student: minimal_3L_30M (3 layers, no MLP, hidden=384, our validated minimal)
  Data: 4096 c4_clean_v1 sequences (smoke-test scale)
  Training: 4000 steps, batch=8, single seed

Two student arms:
  Arm A: from-scratch CE training (control, matches g141 setup)
  Arm B: KD from Qwen3-0.6B teacher (CE + KL on logits, mixed at γ=0.5)

The teacher tokenizer (Qwen3) and student tokenizer (Pythia) are
DIFFERENT. We use Qwen3 tokenizer for both arms (student must use
teacher's vocab for KD). This is a smoke test, not optimal — just
validates pipeline mechanics.

Pre-stated criteria:
  PASS: KD student beats from-scratch by >=0.3pp top-1 on C4 eval.
        Distillation pipeline works; ready to scale to bigger teacher.
  PARTIAL: KD student matches or slightly improves over from-scratch.
  KILL: KD student does NOT improve over from-scratch — protocol broken.

Compute: teacher logit cache (~5 min) + 2 student arms × ~5 min = ~15 min.

If PASS: g155 = production distillation with stronger teacher (Pythia-2.8b
or Qwen3-1.7B) and bigger student (minimal_6L_100M or 200M).

Results: results/genome_154_distillation_smoke.json
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

TEACHER_HF = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
BATCH_SIZE = 8
LR = 3e-4
TRAIN_STEPS = 4000
LR_WARMUP_STEPS = 200
SEED = 42
N_TRAIN = 4096  # smoke test
N_C4_EVAL = 200
KD_TEMP = 2.0
KD_GAMMA = 0.5  # mix: loss = (1-gamma)*CE + gamma*KL


class ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def make_minimal_student(vocab_size, seed=SEED):
    """3L Llama, no MLP, hidden=384 — our validated minimal architecture."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        num_hidden_layers=3,
        num_attention_heads=6,
        num_key_value_heads=6,
        intermediate_size=1024,
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg).to("cuda").to(torch.bfloat16)
    for layer in model.model.layers:
        layer.mlp = ZeroMLP()
    return model


def warmup_lr(step, target_lr, warmup_steps):
    if step < warmup_steps:
        return target_lr * (step + 1) / warmup_steps
    return target_lr


def precompute_teacher_logits(teacher, tok, train_ids, train_mask, top_k=64):
    """Run teacher on all training sequences, save top-k logits per token.
    Top-k is enough for KL distillation and saves disk space. Returns:
      teacher_topk_indices: (N, T, top_k) int64
      teacher_topk_logits:  (N, T, top_k) bfloat16
    """
    print(f"  Precomputing teacher top-{top_k} logits over {train_ids.shape[0]} sequences...")
    teacher.eval()
    n = train_ids.shape[0]
    T = train_ids.shape[1]
    topk_idx = torch.zeros(n, T, top_k, dtype=torch.int64)
    topk_logits = torch.zeros(n, T, top_k, dtype=torch.bfloat16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            ids = train_ids[i:i+BATCH_SIZE].to("cuda")
            mask = train_mask[i:i+BATCH_SIZE].to("cuda")
            out = teacher(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = out.logits  # (B, T, V)
            top_v, top_i = logits.topk(top_k, dim=-1)
            topk_idx[i:i+ids.size(0)] = top_i.cpu()
            topk_logits[i:i+ids.size(0)] = top_v.to(torch.bfloat16).cpu()
            if i % (BATCH_SIZE * 50) == 0:
                pct = 100 * i / n
                print(f"    {i}/{n} ({pct:.0f}%) {time.time()-t0:.0f}s")
    print(f"  done in {time.time()-t0:.0f}s")
    return topk_idx, topk_logits


def kd_loss(student_logits, teacher_topk_idx, teacher_topk_logits, mask, T_temp):
    """KL divergence on top-k positions, with temperature."""
    # student_logits: (B, T-1, V) for next-token prediction
    # teacher_topk_idx, _logits: (B, T, K) — full T including final
    # Align: teacher_topk_idx[:, :-1] corresponds to student_logits at T-1
    teacher_idx = teacher_topk_idx[:, :-1].to("cuda")
    teacher_lg = teacher_topk_logits[:, :-1].to("cuda").float()
    K = teacher_idx.size(-1)
    # Gather student logits at the same top-k positions as teacher
    student_at_topk = student_logits.gather(2, teacher_idx)  # (B, T-1, K)
    # softmax with temperature
    s_log_softmax = F.log_softmax(student_at_topk / T_temp, dim=-1)
    t_softmax = F.softmax(teacher_lg / T_temp, dim=-1)
    # KL(teacher || student) = sum t * (log_t - log_s)
    kl = (t_softmax * (t_softmax.clamp_min(1e-10).log() - s_log_softmax)).sum(dim=-1)  # (B, T-1)
    # mask out padding
    m = mask[:, 1:].to(kl.device).float()
    kl = (kl * m).sum() / m.sum().clamp(min=1)
    # temperature scaling correction
    return kl * (T_temp ** 2)


def measure_eval(model, eval_ids, eval_mask):
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
    model.train()
    return {"nll": total_loss / max(total_tokens, 1),
            "top1_acc": correct_top1 / max(total_tokens, 1)}


def train_arm(arm_name, model, train_ids, train_mask, eval_ids, eval_mask,
                teacher_topk_idx=None, teacher_topk_logits=None):
    """Train student arm. If teacher_topk_idx is None: pure CE. Else: CE + KD."""
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name}: params={n_total/1e6:.2f}M kd={teacher_topk_idx is not None}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(SEED)
    t = time.time()
    model.train()
    n_train = train_ids.size(0)
    losses = []
    log_every = 1000
    for step in range(1, TRAIN_STEPS + 1):
        current_lr = warmup_lr(step, LR, LR_WARMUP_STEPS)
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
        ce_loss = F.cross_entropy(
            sl.view(-1, sl.size(-1)), lbl.view(-1), ignore_index=-100
        )
        if teacher_topk_idx is not None:
            t_idx = teacher_topk_idx[idx]
            t_lg = teacher_topk_logits[idx]
            kd = kd_loss(sl, t_idx, t_lg, mask, KD_TEMP)
            loss = (1 - KD_GAMMA) * ce_loss + KD_GAMMA * kd
        else:
            loss = ce_loss
        if not torch.isfinite(loss):
            print(f"    step={step} NaN")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            losses.append((step, float(ce_loss.item())))
            print(f"    step={step:5d} ce={ce_loss.item():.3f} ({time.time()-t:.0f}s)")
    metrics = measure_eval(model, eval_ids, eval_mask)
    print(f"    eval: NLL={metrics['nll']:.4f} top1={100*metrics['top1_acc']:.2f}%")
    return n_total, time.time() - t, metrics


def main():
    t0 = time.time()
    print(f"genome_154: distillation smoke test (teacher={TEACHER_HF}, student=minimal_3L_30M)")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\nLoading teacher tokenizer {TEACHER_HF}...")
    tok = AutoTokenizer.from_pretrained(TEACHER_HF)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    actual_vocab = len(tok)
    print(f"  vocab size: {actual_vocab}")

    print(f"\nLoading {N_TRAIN} c4 train + {N_C4_EVAL} eval...")
    pool_texts = []
    target = N_TRAIN + N_C4_EVAL
    for rec in c4_clean_v1(seed=42, n_samples=target):
        pool_texts.append(rec["text"])
        if len(pool_texts) >= target:
            break
    train_texts = pool_texts[:N_TRAIN]
    eval_texts = pool_texts[N_TRAIN:N_TRAIN + N_C4_EVAL]

    enc_t = tok(train_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc_t["input_ids"]; train_mask = enc_t["attention_mask"]
    enc_e = tok(eval_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    eval_ids = enc_e["input_ids"]; eval_mask = enc_e["attention_mask"]
    print(f"  train: {train_ids.shape}")

    print(f"\nLoading teacher {TEACHER_HF}...")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_HF, torch_dtype=torch.float16
    ).to("cuda").eval()
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  teacher params: {teacher_params/1e6:.1f}M")

    teacher_topk_idx, teacher_topk_logits = precompute_teacher_logits(
        teacher, tok, train_ids, train_mask, top_k=64,
    )
    del teacher
    torch.cuda.empty_cache()

    # === ARM A: from-scratch CE ===
    print(f"\n=== Arm A: from-scratch CE ===")
    student_a = make_minimal_student(actual_vocab, seed=SEED)
    n_a, t_a, m_a = train_arm("arm_A_scratch", student_a, train_ids, train_mask,
                                 eval_ids, eval_mask)
    del student_a
    torch.cuda.empty_cache()

    # === ARM B: KD ===
    print(f"\n=== Arm B: KD (gamma={KD_GAMMA}, T={KD_TEMP}) ===")
    student_b = make_minimal_student(actual_vocab, seed=SEED)
    n_b, t_b, m_b = train_arm("arm_B_kd", student_b, train_ids, train_mask,
                                 eval_ids, eval_mask,
                                 teacher_topk_idx=teacher_topk_idx,
                                 teacher_topk_logits=teacher_topk_logits)
    del student_b
    torch.cuda.empty_cache()

    # Analysis
    print(f"\n=== ANALYSIS ===")
    c4_top1_gap_pp = (m_b["top1_acc"] - m_a["top1_acc"]) * 100
    nll_gap = m_a["nll"] - m_b["nll"]
    print(f"  Arm A (scratch): NLL={m_a['nll']:.4f} top1={100*m_a['top1_acc']:.2f}%")
    print(f"  Arm B (KD):      NLL={m_b['nll']:.4f} top1={100*m_b['top1_acc']:.2f}%")
    print(f"  KD top1 gap: {c4_top1_gap_pp:+.3f}pp")
    print(f"  KD NLL gap (positive = KD better): {nll_gap:+.4f}")

    if c4_top1_gap_pp >= 0.3:
        verdict = (f"PASS: KD student beats scratch by {c4_top1_gap_pp:+.2f}pp top1 + "
                   f"{nll_gap:+.3f} NLL. Distillation pipeline works. Ready to scale "
                   f"to stronger teacher and bigger student.")
    elif c4_top1_gap_pp >= 0:
        verdict = (f"PARTIAL: KD matches scratch ({c4_top1_gap_pp:+.2f}pp). "
                   f"Pipeline runs but signal weak — likely teacher too small or "
                   f"data scale too tiny. Try stronger teacher.")
    else:
        verdict = (f"KILL: KD student worse than scratch ({c4_top1_gap_pp:+.2f}pp). "
                   f"Pipeline mechanics broken — debug before scaling.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 154, "name": "distillation_smoke",
        "config": {"teacher": TEACHER_HF, "student": "minimal_3L_30M",
                    "n_train": N_TRAIN, "train_steps": TRAIN_STEPS,
                    "kd_temp": KD_TEMP, "kd_gamma": KD_GAMMA,
                    "lr": LR, "batch": BATCH_SIZE, "seed": SEED},
        "teacher_params_M": teacher_params / 1e6,
        "arm_A_scratch": {"params_M": n_a / 1e6, "wallclock_s": t_a, "metrics": m_a},
        "arm_B_kd": {"params_M": n_b / 1e6, "wallclock_s": t_b, "metrics": m_b},
        "deltas": {"top1_gap_pp": c4_top1_gap_pp, "nll_gap": nll_gap},
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_154_distillation_smoke.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
