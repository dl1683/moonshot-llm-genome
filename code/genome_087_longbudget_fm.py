"""Long-budget layerwise feature-matching — is the three-wall ceiling
training-budget-limited or structural?

genome_078-084 (atlas): 5/5 repetitive at 49-58pct NLL
genome_085 (output-KL last-7, 200 steps):     5/5 rep at 66pct NLL
genome_086 (layerwise-FM full-unfreeze, 200 steps): 5/5 rep at 65pct NLL

All three hit the same coherence wall at 200 training steps. Does
running the strongest supervision class (layer-wise FM + output KL +
full unfreeze) for 10x more steps break the wall?

If YES — the wall is a budget artifact, capability CAN be reconstructed
with enough gradient supervision, and the honest bound is how-cheaply.

If NO — the wall is structural, and the three-wall result hardens into
a bona fide negative capability-transfer claim (cannot retrain from
scratch with teacher supervision in a small budget either).

Budget: 2000 steps (10x genome_086), lr 1e-4 (3x), batch 4, max_len 96.
Previous 200-step run took 55s; 2000 at larger batch should fit under
20 minutes. Well inside COMPUTE.md envelope.

Checkpoint NLL + generation at {200, 500, 1000, 1500, 2000}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
        if len(sents) >= 1200:
            break
    train_texts = sents[:1000]
    val_texts = sents[1000:1100]
    t0 = time.time()

    print(f"[{time.time()-t0:.1f}s] TEACHER...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    for p in sys_t.model.parameters():
        p.requires_grad_(False)
    nll_teacher, _ = measure_nll(sys_t.model, sys_t.tokenizer, val_texts)
    n_layers = sys_t.n_hidden_layers()
    print(f"  teacher NLL={nll_teacher:.3f}  n_layers={n_layers}")

    print(f"[{time.time()-t0:.1f}s] STUDENT (all lesioned, fp32)...")
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    for L in range(n_layers):
        lesion_midblock(sd, f"model.layers.{L}.")
    sys_s.model.load_state_dict(sd, strict=False)
    sys_s.model.to(dtype=torch.float32)
    nll_lesion, _ = measure_nll(sys_s.model, sys_s.tokenizer, val_texts)
    print(f"  lesion NLL={nll_lesion:.3f}")

    for p in sys_s.model.parameters():
        p.requires_grad_(True)
    trainable = sum(p.numel() for p in sys_s.model.parameters())
    print(f"  trainable: {trainable/1e6:.0f}M")

    opt = torch.optim.AdamW(sys_s.model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000, eta_min=1e-5)

    print(f"\n[{time.time()-t0:.1f}s] pre-train generation...")
    pre = {}
    for p in PROMPTS:
        pre[p] = generate(sys_s.model, sys_s.tokenizer, p)
        safe_print("PRE", p, pre[p])

    print(f"\n[{time.time()-t0:.1f}s] distillation (2000 steps, lr=1e-4)...")
    log = [{"step": 0, "val_nll": float(nll_lesion)}]
    gen_snapshots = {}
    batch_size = 4
    max_len = 96
    checkpoints = {200, 500, 1000, 1500, 2000}

    for step in range(1, 2001):
        idx = ((step - 1) * batch_size) % max(len(train_texts) - batch_size, 1)
        chunk = train_texts[idx:idx + batch_size]
        if len(chunk) < batch_size:
            chunk = chunk + train_texts[:batch_size - len(chunk)]

        enc = sys_s.tokenizer(chunk, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_len).to("cuda")
        with torch.no_grad():
            t_out = sys_t.model(**enc, output_hidden_states=True)
            t_hidden = [h.float() for h in t_out.hidden_states]
            t_logits = t_out.logits.float()

        s_out = sys_s.model(**enc, output_hidden_states=True)
        s_hidden = s_out.hidden_states
        s_logits = s_out.logits

        mask = enc["attention_mask"].unsqueeze(-1).float()
        fm_loss = 0.0
        for th, sh in zip(t_hidden, s_hidden):
            diff = (sh - th) * mask
            fm_loss = fm_loss + (diff ** 2).sum() / max(mask.sum() * sh.shape[-1], 1)
        fm_loss = fm_loss / len(t_hidden)

        logT_t = F.log_softmax(t_logits, dim=-1)
        logT_s = F.log_softmax(s_logits, dim=-1)
        kl = F.kl_div(logT_s, logT_t, reduction="none", log_target=True).sum(-1, keepdim=True)
        kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1)

        # Weight KL more heavily so output coherence gets real pressure
        loss = 0.01 * fm_loss + 1.0 * kl_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sys_s.model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step in checkpoints:
            nll_now, _ = measure_nll(sys_s.model, sys_s.tokenizer, val_texts)
            log.append({"step": step, "val_nll": float(nll_now),
                        "fm_loss": float(fm_loss.item()),
                        "kl_loss": float(kl_loss.item()),
                        "wall_s": time.time() - t0})
            print(f"  [step {step:4d}] val_nll={nll_now:.3f}  "
                  f"fm={fm_loss.item():.4f}  kl={kl_loss.item():.3f}  "
                  f"t={time.time()-t0:.1f}s")
            snap = {}
            for p in PROMPTS:
                snap[p] = generate(sys_s.model, sys_s.tokenizer, p)
                safe_print(f"S{step}", p, snap[p])
            gen_snapshots[step] = snap

    post = gen_snapshots[2000]
    repetitive = sum(1 for p, c in post.items() if len(set(c.split())) <= 3)
    fg = (nll_lesion - log[-1]["val_nll"]) / max(nll_lesion - nll_teacher, 1e-6)
    print(f"\n=== LONG-BUDGET LAYERWISE FM ===")
    print(f"  teacher:  NLL={nll_teacher:.3f}")
    print(f"  lesion:   NLL={nll_lesion:.3f}")
    print(f"  post 2k:  NLL={log[-1]['val_nll']:.3f}  fg={fg:+.3f}")
    print(f"  repetitive @ 2000: {repetitive}/{len(PROMPTS)}")
    rep_curve = {s: sum(1 for p, c in snap.items() if len(set(c.split())) <= 3)
                 for s, snap in gen_snapshots.items()}
    print(f"  rep curve: {rep_curve}")

    out = {"teacher_nll": float(nll_teacher),
           "lesion_nll": float(nll_lesion),
           "post_nll": float(log[-1]["val_nll"]),
           "fg_closed": float(fg),
           "repetitive_completions_final": int(repetitive),
           "repetitive_curve": rep_curve,
           "pre": pre,
           "gen_snapshots": gen_snapshots,
           "log": log}
    out_path = _ROOT / "results/gate2/longbudget_fm.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
