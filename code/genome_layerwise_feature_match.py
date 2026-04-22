"""Layer-wise feature-matching distillation.

genome_083 (static atlas), genome_085 (output-KL distill on last-7):
both recovered unigram prior but failed to restore coherent generation.
Converging diagnosis: sparse output-level constraints cannot reconstruct
context-conditional hidden-state trajectory.

This probe tests the stronger supervision class: at every layer, match
student's hidden activation to teacher's hidden activation (MSE loss on
full hidden states, not just output logits). We unfreeze ALL layers of
the lesioned student and supervise every layer's output.

If coherent generation returns, feature-matching is the key. If still
degenerate, the three-way convergence (atlas / last-7-KL / layerwise-
FM) establishes a strong publishable negative claim about the limits of
sparse capability transfer.

Params: 100M+ total trainable (full student unfrozen), 200 steps, lr
3e-5 (lower, because supervising intermediate features is more powerful
signal and we want stable updates).
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
    print(f"  teacher NLL={nll_teacher:.3f}  n_layers={n_layers}")

    # Student: all layers lesioned, cast to fp32
    print(f"[{time.time()-t0:.1f}s] STUDENT (all lesioned, fp32)...")
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    for L in range(n_layers):
        lesion_midblock(sd, f"model.layers.{L}.")
    sys_s.model.load_state_dict(sd, strict=False)
    sys_s.model.to(dtype=torch.float32)
    nll_lesion, _ = measure_nll(sys_s.model, sys_s.tokenizer, val_texts)
    print(f"  lesion NLL={nll_lesion:.3f}")

    # Unfreeze everything in student
    for p in sys_s.model.parameters():
        p.requires_grad_(True)
    trainable = sum(p.numel() for p in sys_s.model.parameters())
    print(f"  all trainable: {trainable/1e6:.0f}M params")

    opt = torch.optim.AdamW(sys_s.model.parameters(), lr=3e-5, betas=(0.9, 0.95))

    # Pre-train generation baseline
    print(f"\n[{time.time()-t0:.1f}s] pre-train generation...")
    pre = {}
    for p in PROMPTS:
        pre[p] = generate(sys_s.model, sys_s.tokenizer, p)
        safe_print("PRE", p, pre[p])

    # Training loop: layer-wise feature matching
    print(f"\n[{time.time()-t0:.1f}s] distillation (full layer-wise FM, 200 steps)...")
    log = [{"step": 0, "val_nll": float(nll_lesion)}]
    batch_size = 2
    max_len = 64

    for step in range(1, 201):
        idx = ((step - 1) * batch_size) % max(len(train_texts) - batch_size, 1)
        chunk = train_texts[idx:idx + batch_size]
        if len(chunk) < batch_size:
            chunk = chunk + train_texts[:batch_size - len(chunk)]

        enc = sys_s.tokenizer(chunk, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_len).to("cuda")
        with torch.no_grad():
            t_out = sys_t.model(**enc, output_hidden_states=True)
            t_hidden = [h.float() for h in t_out.hidden_states]  # list of (b,s,h)
            t_logits = t_out.logits.float()

        s_out = sys_s.model(**enc, output_hidden_states=True)
        s_hidden = s_out.hidden_states
        s_logits = s_out.logits

        # Layer-wise MSE on hidden states
        mask = enc["attention_mask"].unsqueeze(-1).float()
        fm_loss = 0.0
        for th, sh in zip(t_hidden, s_hidden):
            diff = (sh - th) * mask
            fm_loss = fm_loss + (diff ** 2).sum() / max(mask.sum() * sh.shape[-1], 1)
        fm_loss = fm_loss / len(t_hidden)

        # Plus output KL
        logT_t = F.log_softmax(t_logits, dim=-1)
        logT_s = F.log_softmax(s_logits, dim=-1)
        kl = F.kl_div(logT_s, logT_t, reduction="none", log_target=True).sum(-1, keepdim=True)
        kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1)

        loss = fm_loss + 0.5 * kl_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sys_s.model.parameters(), 1.0)
        opt.step()

        if step in (10, 25, 50, 100, 200):
            nll_now, _ = measure_nll(sys_s.model, sys_s.tokenizer, val_texts)
            log.append({"step": step, "val_nll": float(nll_now),
                        "fm_loss": float(fm_loss.item()),
                        "kl_loss": float(kl_loss.item()),
                        "wall_s": time.time() - t0})
            print(f"  [step {step:4d}] val_nll={nll_now:.3f}  "
                  f"fm={fm_loss.item():.4f}  kl={kl_loss.item():.3f}  "
                  f"t={time.time()-t0:.1f}s")

    # Post gen
    print(f"\n[{time.time()-t0:.1f}s] post-train generation...")
    post = {}
    for p in PROMPTS:
        post[p] = generate(sys_s.model, sys_s.tokenizer, p)
        safe_print("POST", p, post[p])

    repetitive = sum(1 for p, c in post.items() if len(set(c.split())) <= 3)
    fg = (nll_lesion - log[-1]["val_nll"]) / max(nll_lesion - nll_teacher, 1e-6)
    print(f"\n=== LAYERWISE FM DISTILLATION ===")
    print(f"  teacher:      NLL={nll_teacher:.3f}")
    print(f"  lesion:       NLL={nll_lesion:.3f}")
    print(f"  post-FM:      NLL={log[-1]['val_nll']:.3f}  fg={fg:+.3f}")
    print(f"  repetitive:   {repetitive}/{len(PROMPTS)}")

    out = {"teacher_nll": float(nll_teacher),
           "lesion_nll": float(nll_lesion),
           "post_nll": float(log[-1]["val_nll"]),
           "fg_closed": float(fg),
           "repetitive_completions": int(repetitive),
           "pre": pre, "post": post, "log": log}
    out_path = _ROOT / "results/gate2/layerwise_feature_match.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
