"""Does the sqrt(eff_rank)*alpha invariant track capability recovery?

genome_088 validated sqrt(eff_rank)*alpha ≈ 3*sqrt(2) on trained ML
(N=5, CV 5.1%, 5.5σ separation from shuffled/Gaussian baseline at 5.47).

genome_087 showed capability recovery from catastrophic lesion is a
phase transition at ~1500 steps of layer-wise FM + KL.

If the invariant is capability-COUPLED, it should evolve from the
lesion value (~5.5, iid-like) to the trained attractor (~4.25) during
the phase transition — specifically, the crossover should correlate
with the rep-count drop between step 1000 and 1500.

This experiment repeats genome_087 but tracks:
  - sqrt(eff_rank)*alpha on mid-depth student activations at each
    checkpoint (0, 200, 500, 1000, 1500, 2000)
  - val_nll, repetition-count, fm_loss, kl_loss as before

Output: trajectory table + plot — value of sqrt(er)*alpha vs step vs
rep-count. If invariant falls toward 4.24 alongside coherence emergence,
we have a measurable capability-tracking signal.

Budget: ~8-9 min wall (1 SVD per checkpoint ≈ 2s overhead, negligible
vs the ~500s training cost).
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


@torch.no_grad()
def measure_invariant(model, tokenizer, texts, layer_idx, batch=16, max_len=256, device="cuda"):
    """Extract mid-depth activations, compute sqrt(eff_rank)*alpha and er*alpha^2."""
    model.eval()
    acts = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer_idx].float()  # (b, s, h)
        mask = enc["attention_mask"].float().unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        acts.append(pooled.cpu().numpy())
    X = np.concatenate(acts, axis=0).astype(np.float64)
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(max(X.shape[0] - 1, 1))
    s2 = s ** 2
    er = float(s2.sum() ** 2 / (s2 ** 2).sum()) if s2.sum() > 0 else 0.0
    r = np.arange(1, len(s) + 1)
    lo, hi = max(1, int(len(s) * 0.05)), int(len(s) * 0.5)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    alpha = float(-slope)
    return {
        "eff_rank": er, "alpha": alpha,
        "sqrt_er_alpha": float(np.sqrt(er) * alpha),
        "er_alpha2": float(er * alpha ** 2),
    }


def main():
    hf_id = "Qwen/Qwen3-0.6B"
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 1400:
            break
    train_texts = sents[:1000]
    val_texts = sents[1000:1100]
    probe_texts = sents[1100:1300]
    t0 = time.time()

    print(f"[{time.time()-t0:.1f}s] TEACHER...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    for p in sys_t.model.parameters():
        p.requires_grad_(False)
    nll_teacher, _ = measure_nll(sys_t.model, sys_t.tokenizer, val_texts)
    n_layers = sys_t.n_hidden_layers()
    mid = max(1, n_layers // 2)
    inv_teacher = measure_invariant(sys_t.model, sys_t.tokenizer, probe_texts, mid)
    print(f"  teacher NLL={nll_teacher:.3f}  mid-layer={mid}  inv={inv_teacher}")

    print(f"[{time.time()-t0:.1f}s] STUDENT (all lesioned, fp32)...")
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    for L in range(n_layers):
        lesion_midblock(sd, f"model.layers.{L}.")
    sys_s.model.load_state_dict(sd, strict=False)
    sys_s.model.to(dtype=torch.float32)
    nll_lesion, _ = measure_nll(sys_s.model, sys_s.tokenizer, val_texts)
    inv_lesion = measure_invariant(sys_s.model, sys_s.tokenizer, probe_texts, mid)
    print(f"  lesion NLL={nll_lesion:.3f}  inv={inv_lesion}")

    for p in sys_s.model.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(sys_s.model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000, eta_min=1e-5)

    log = [{"step": 0, "val_nll": float(nll_lesion), **inv_lesion, "rep": 5}]
    checkpoints = {200, 500, 1000, 1500, 2000}
    batch_size = 4
    max_len = 96

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
        loss = 0.01 * fm_loss + 1.0 * kl_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sys_s.model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step in checkpoints:
            nll_now, _ = measure_nll(sys_s.model, sys_s.tokenizer, val_texts)
            inv = measure_invariant(sys_s.model, sys_s.tokenizer, probe_texts, mid)
            gen = {p: generate(sys_s.model, sys_s.tokenizer, p) for p in PROMPTS}
            rep = sum(1 for p, c in gen.items() if len(set(c.split())) <= 3)
            log.append({"step": step, "val_nll": float(nll_now), **inv,
                        "rep": int(rep), "fm_loss": float(fm_loss.item()),
                        "kl_loss": float(kl_loss.item()),
                        "wall_s": time.time() - t0})
            print(f"\n  [step {step:4d}] val_nll={nll_now:.3f}  "
                  f"eff_rank={inv['eff_rank']:.2f}  alpha={inv['alpha']:.3f}  "
                  f"sqrt(er)*alpha={inv['sqrt_er_alpha']:.3f}  "
                  f"er*alpha^2={inv['er_alpha2']:.3f}  rep={rep}/5  t={time.time()-t0:.1f}s")
            for p, c in gen.items():
                safe_print(f"S{step}", p, c)

    print(f"\n=== INVARIANT TRAJECTORY ===")
    print(f"  teacher: sqrt(er)*alpha = {inv_teacher['sqrt_er_alpha']:.3f}  (3√2 = 4.243)")
    print(f"  trajectory:")
    for r in log:
        print(f"    step {r['step']:4d}  NLL={r['val_nll']:6.3f}  "
              f"sqrt(er)*α={r.get('sqrt_er_alpha',0):.3f}  rep={r.get('rep',-1)}/5")

    out = {
        "teacher_nll": float(nll_teacher),
        "teacher_invariant": inv_teacher,
        "lesion_nll": float(nll_lesion),
        "lesion_invariant": inv_lesion,
        "trajectory": log,
        "ref_3sqrt2": float(np.sqrt(18)),
    }
    out_path = _ROOT / "results/gate2/invariant_trajectory.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
