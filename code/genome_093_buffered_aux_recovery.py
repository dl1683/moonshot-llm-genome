"""Buffered geometric aux loss — stronger formulation of genome_090.

genome_090 used batch-level eff_rank matching at batch=4, where
both teacher and student are naturally bounded at er≈3. The aux
loss magnitude was tiny (0.01) and did not meaningfully influence
training.

This variant maintains a ROLLING BUFFER of the last K=64 pooled
mid-depth activations from the student. Computes eff_rank on the
buffer (effective dimension up to 64). Targets buffer eff_rank at
matched teacher buffer eff_rank. At K=64 the invariant has real
leverage — student under mode-collapse will show buffer-er ≈ 1-3,
while healthy model shows 15+.

Compare against same-seed γ=0 control.

Budget: ~17 min.
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
    model.eval()
    acts = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer_idx].float()
        mask = enc["attention_mask"].float().unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        acts.append(pooled.cpu().numpy())
    X = np.concatenate(acts, axis=0).astype(np.float64)
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(max(X.shape[0] - 1, 1))
    s2 = s ** 2
    er = float(s2.sum()**2 / (s2**2).sum()) if s2.sum() > 0 else 0.0
    r = np.arange(1, len(s) + 1)
    lo, hi = max(1, int(len(s) * 0.05)), int(len(s) * 0.5)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    alpha = float(-slope)
    return {"eff_rank": er, "alpha": alpha,
            "sqrt_er_alpha": float(np.sqrt(er) * alpha),
            "er_alpha2": float(er * alpha ** 2)}


def eff_rank_buffered(buf):
    """buf is (K, h) tensor. Compute eff_rank differentiably.
    Uses trace(S)^2 / trace(S^2) via pairwise inner products.
    """
    x = buf - buf.mean(dim=0, keepdim=True)
    K = x.shape[0]
    denom = max(K - 1, 1)
    xtx = x @ x.T  # (K, K)
    trS = xtx.diagonal().sum() / denom
    trS2 = (xtx * xtx).sum() / (denom ** 2)
    er = trS ** 2 / trS2.clamp(min=1e-8)
    return er


def train_one(cond_name, gamma, t0, sys_t, mid, n_layers,
               train_texts, val_texts, probe_texts, hf_id, checkpoints,
               buffer_K=64, er_target=16.0):
    print(f"\n========== RUN: {cond_name} (gamma={gamma}) ==========\n")
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
    batch_size = 4
    max_len = 96
    buffer = []  # list of torch tensors (batch_size, h), each a single-step's pooled acts

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

        # Buffered aux: accumulate last K/batch_size steps of pooled acts
        mid_h_s = s_hidden[mid]
        pooled_s = (mid_h_s * mask).sum(1) / mask.sum(1).clamp(min=1)  # (b, h)
        buffer.append(pooled_s)
        max_buf_steps = buffer_K // batch_size
        if len(buffer) > max_buf_steps:
            buffer = buffer[-max_buf_steps:]

        if gamma > 0 and len(buffer) >= max_buf_steps:
            buf_tensor = torch.cat(buffer, dim=0)  # (K, h)
            er_s = eff_rank_buffered(buf_tensor)
            aux_loss = (er_s - er_target) ** 2
        else:
            er_s = torch.tensor(0.0, device="cuda")
            aux_loss = torch.tensor(0.0, device="cuda")

        loss = 0.01 * fm_loss + 1.0 * kl_loss + gamma * aux_loss

        opt.zero_grad()
        loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(sys_s.model.parameters(), 1.0)
        opt.step()
        sched.step()
        # Detach buffer after backward so it doesn't accumulate graph
        buffer = [b.detach() for b in buffer]

        if step in checkpoints:
            nll_now, _ = measure_nll(sys_s.model, sys_s.tokenizer, val_texts)
            inv = measure_invariant(sys_s.model, sys_s.tokenizer, probe_texts, mid)
            gen = {p: generate(sys_s.model, sys_s.tokenizer, p) for p in PROMPTS}
            rep = sum(1 for p, c in gen.items() if len(set(c.split())) <= 3)
            log.append({"step": step, "val_nll": float(nll_now), **inv,
                        "rep": int(rep), "fm_loss": float(fm_loss.item()),
                        "kl_loss": float(kl_loss.item()),
                        "aux_loss": float(aux_loss.item()),
                        "er_buffer": float(er_s.item()),
                        "wall_s": time.time() - t0})
            print(f"\n  [{cond_name} step {step:4d}] val_nll={nll_now:.3f}  "
                  f"eff_rank={inv['eff_rank']:.2f}  alpha={inv['alpha']:.3f}  "
                  f"sqrt(er)*a={inv['sqrt_er_alpha']:.3f}  aux={aux_loss.item():.2f}  "
                  f"er_buf={er_s.item():.2f}  rep={rep}/5  t={time.time()-t0:.1f}s")
            for p, c in gen.items():
                safe_print(f"{cond_name[:3]}{step}", p, c)
    sys_s.unload(); torch.cuda.empty_cache()
    return {"lesion_nll": float(nll_lesion), "lesion_invariant": inv_lesion, "log": log}


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

    checkpoints = {200, 500, 1000, 1500, 2000}
    results = {}

    results["control"] = train_one(
        "control_gamma0", 0.0, t0, sys_t, mid, n_layers,
        train_texts, val_texts, probe_texts, hf_id, checkpoints)

    results["aux"] = train_one(
        "aux_buffered", 1e-2, t0, sys_t, mid, n_layers,
        train_texts, val_texts, probe_texts, hf_id, checkpoints,
        buffer_K=64, er_target=16.0)

    out = {"teacher_nll": float(nll_teacher),
           "teacher_invariant": inv_teacher,
           **results}
    out_path = _ROOT / "results/gate2/buffered_aux_recovery.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
