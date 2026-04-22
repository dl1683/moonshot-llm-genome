"""Gradient-distillation surgery on last-7 layers.

After genome_083/084 scope-corrected the mean-shift atlas to "unigram
prior restorer only," the next question is: can a small AMOUNT of
gradient-based surgery on the last-7 layers restore coherent generation
from a fully-lesioned Qwen3-0.6B?

PROTOCOL
 - Teacher: pretrained Qwen3-0.6B, frozen.
 - Student: Qwen3-0.6B with ALL 28 transformer layers lesioned.
 - Unfreeze: only layers 21..27 (last 7). Middle/early stay lesioned.
 - Distillation loss: KL(student_logits || teacher_logits) on C4 text.
 - 200 steps at lr=1e-4, batch=4.
 - Measure: val_nll + coherent generation at {0, 50, 100, 200} steps.

PASS: if val_nll drops substantially AND completions become coherent
(not "directly directly..."), we have a cheap surgery primitive.

KILL: if val_nll stays high or completions stay degenerate after 200
steps, gradient distillation on last-7 alone is insufficient and we'd
need wider unfreezing (more layers) or different loss class.

Partner angle: this mirrors the "patch a partially-corrupted checkpoint"
use case - except we're using a maximal lesion to stress the surgery.
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
from genome_atlas_qualitative_samples import PROMPTS, generate  # noqa: E402
from genome_capability_patch_k48 import lesion_midblock  # noqa: E402
from genome_geometry_transfusion import measure_nll  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


def safe_print(label, prompt, completion):
    sp = prompt.encode("ascii", "backslashreplace").decode("ascii")
    sc = completion.encode("ascii", "backslashreplace").decode("ascii")
    print(f"  [{label}] {sp!r} -> {sc!r}")


def main():
    hf_id = "Qwen/Qwen3-0.6B"
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 400:
            break
    train_texts = sents[:300]
    val_texts = sents[300:400]

    t0 = time.time()

    # Teacher (frozen)
    print(f"[{time.time()-t0:.1f}s] TEACHER...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    for p in sys_t.model.parameters():
        p.requires_grad_(False)
    nll_teacher, _ = measure_nll(sys_t.model, sys_t.tokenizer, val_texts)
    n_layers = sys_t.n_hidden_layers()
    print(f"  teacher NLL = {nll_teacher:.3f}  n_layers = {n_layers}")

    # Student: fully lesioned. Cast to fp32 so gradient updates are stable.
    print(f"[{time.time()-t0:.1f}s] STUDENT (all 28 layers lesioned, fp32)...")
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    for L in range(n_layers):
        lesion_midblock(sd, f"model.layers.{L}.")
    sys_s.model.load_state_dict(sd, strict=False)
    sys_s.model.to(dtype=torch.float32)  # cast whole model to fp32 for training
    nll_lesion, _ = measure_nll(sys_s.model, sys_s.tokenizer, val_texts)
    print(f"  student lesion NLL = {nll_lesion:.3f}")

    # Freeze everything except last-7 layers
    last_7 = set(range(n_layers - 7, n_layers))
    trainable_params = []
    for name, p in sys_s.model.named_parameters():
        # e.g., model.layers.21.self_attn.q_proj.weight
        unfreeze = False
        parts = name.split(".")
        if parts[:2] == ["model", "layers"]:
            try:
                L = int(parts[2])
                if L in last_7:
                    unfreeze = True
            except ValueError:
                pass
        if unfreeze:
            p.requires_grad_(True)
            trainable_params.append(p)
        else:
            p.requires_grad_(False)
    n_trainable = sum(p.numel() for p in trainable_params)
    total = sum(p.numel() for p in sys_s.model.parameters())
    print(f"  trainable params: {n_trainable} / {total} ({100*n_trainable/total:.1f}pct)")

    opt = torch.optim.AdamW(trainable_params, lr=1e-4, betas=(0.9, 0.95))

    # Gen sample BEFORE any training
    print(f"\n[{time.time()-t0:.1f}s] generating pre-training samples...")
    pretrain_out = {}
    for p in PROMPTS:
        pretrain_out[p] = generate(sys_s.model, sys_s.tokenizer, p)
        safe_print("PRE", p, pretrain_out[p])

    # Distillation loop
    print(f"\n[{time.time()-t0:.1f}s] starting distillation (200 steps)...")
    log = [{"step": 0, "val_nll": float(nll_lesion), "wall_s": 0.0}]
    batch_texts = train_texts
    max_len = 64
    batch_size = 4

    for step in range(1, 201):
        # Sample batch
        idx_start = ((step - 1) * batch_size) % max(len(batch_texts) - batch_size, 1)
        chunk = batch_texts[idx_start:idx_start + batch_size]
        if len(chunk) < batch_size:
            chunk = chunk + batch_texts[:batch_size - len(chunk)]

        enc = sys_s.tokenizer(chunk, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_len).to("cuda")
        with torch.no_grad():
            teacher_logits = sys_t.model(**enc).logits.float()

        student_logits = sys_s.model(**enc).logits.float()

        # KL divergence per-token
        T = 1.0
        p_t = F.log_softmax(teacher_logits / T, dim=-1)
        p_s = F.log_softmax(student_logits / T, dim=-1)
        # Mask padding
        mask = enc["attention_mask"].unsqueeze(-1).float()
        kl = F.kl_div(p_s, p_t, reduction="none", log_target=True).sum(-1, keepdim=True)
        loss = (kl * mask).sum() / mask.sum().clamp(min=1)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()

        if step in (10, 25, 50, 100, 200):
            nll_now, _ = measure_nll(sys_s.model, sys_s.tokenizer, val_texts)
            log.append({"step": step, "val_nll": float(nll_now),
                        "train_kl": float(loss.item()),
                        "wall_s": time.time() - t0})
            print(f"  [step {step:4d}] val_nll = {nll_now:.3f}  "
                  f"train_kl = {loss.item():.3f}  t = {time.time()-t0:.1f}s")

    # Post-training completions
    print(f"\n[{time.time()-t0:.1f}s] generating post-training samples...")
    post_out = {}
    for p in PROMPTS:
        post_out[p] = generate(sys_s.model, sys_s.tokenizer, p)
        safe_print("POST", p, post_out[p])

    nll_post = log[-1]["val_nll"]
    gap = nll_lesion - nll_teacher
    fg = (nll_lesion - nll_post) / max(gap, 1e-6)

    print(f"\n=== LAST-7 DISTILLATION SURGERY ===")
    print(f"  teacher NLL:      {nll_teacher:.3f}")
    print(f"  lesion NLL:       {nll_lesion:.3f}")
    print(f"  post-distill NLL: {nll_post:.3f}  fg_closed = {fg:+.3f}")
    print(f"  trainable params: {n_trainable} ({100*n_trainable/total:.1f}pct of full)")

    # Crude coherence check: is the post-training output all-repetitive?
    repetitive_post = sum(1 for p, c in post_out.items()
                          if len(set(c.split())) <= 3)
    print(f"  repetitive completions: {repetitive_post}/{len(PROMPTS)}")

    out = {"teacher_nll": float(nll_teacher),
           "lesion_nll": float(nll_lesion),
           "post_distill_nll": float(nll_post),
           "fraction_gap_closed": float(fg),
           "n_trainable_params": int(n_trainable),
           "trainable_fraction": float(n_trainable / total),
           "pre_training_completions": {p: c for p, c in pretrain_out.items()},
           "post_training_completions": {p: c for p, c in post_out.items()},
           "log": log,
           "repetitive_completions_count": int(repetitive_post)}
    out_path = _ROOT / "results/gate2/last7_distill_surgery.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
