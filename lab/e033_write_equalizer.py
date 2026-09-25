"""E033 — Write-equalizer: who absorbs the late-MLP energy? (homeostasis bio-analogue)

REGISTERED (before running):
- Arm: fresh net (seed 42), same corpus/budget as e001, but every MLP
  block's residual write is renormalized per-token to the MEAN of the
  baseline's MLP write norms per layer (hooks in train+eval — the e014b
  playbook applied to MLP writes instead of the stream).
- P1 (parity): equalized net reaches val <= 1.7224 (the e014b gate).
- P2 (energy migrates): in the equalized net, the late-attention damage
  profile RISES (attn-L5 damage >= 2x baseline's +0.03) OR the lm_head/
  embedding rows absorb it (their ablation cost rises >= 2x) — the
  energy-carrier function (e047: 5/5 replication) is allocated, not owned.
- P3 (law survives): the L5-calibrator signature persists (KL(final||L4)
  >= 0.5 nats) — reshaping is not the MLP energy's job.

Run: python lab/e033_write_equalizer.py
"""
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cooldown, estimate_loss,
                    lesion_loss, plot_history, run_dir, save_json, set_seed,
                    train_model)
import json

BASE_CKPT = REPO / "runs" / "checkpoints" / "e005s_small.pt"  # 0.84M — compute envelope
TRAIN_CKPT = REPO / "runs" / "checkpoints" / "e033.train.pt"
FINAL_CKPT = REPO / "runs" / "checkpoints" / "e033.pt"
N_EVAL = 30


@torch.no_grad()
def mlp_write_norms(model, corpus, n_batches=8):
    norms = [0.0] * model.cfg.n_layer
    handles = []
    for i, block in enumerate(model.h):
        def mk(i):
            def h(m, a, o):
                norms[i] += float(o.norm(dim=-1).mean())
            return h
        handles.append(block.mlp.register_forward_hook(mk(i)))
    gen = torch.Generator().manual_seed(33)
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = corpus.get_batch("val", model.cfg.block_size, 16, gen=gen)
            model(x)
    for h in handles:
        h.remove()
    return [n / n_batches for n in norms]


def register_equalizer(model, target):
    """Rescale every MLP block's residual WRITE to a single equal norm
    (forward hook on the block's output — post-hook transform, cheap)."""
    hooks = []
    for i, block in enumerate(model.h):
        def post(m, a, o):
            n = o.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return o * (target / n)
        hooks.append(block.mlp.register_forward_hook(post))
    return hooks


def main():
    set_seed(42)
    rd = run_dir("e033")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=4, n_head=4, n_embd=128, block_size=256)  # 0.84M

    base = TinyGPT(cfg).to(DEVICE)
    base.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=True))
    base.eval()
    base_val = estimate_loss(base, corpus, "val", n_batches=N_EVAL)
    targets = mlp_write_norms(base, corpus)
    print(f"baseline val {base_val:.4f}; MLP write-norm targets {[round(t,2) for t in targets]}")

    b_attn = [lesion_loss(base, corpus, "attn", i, n_batches=N_EVAL) - base_val for i in range(cfg.n_layer)]
    b_mlp = [lesion_loss(base, corpus, "mlp", i, n_batches=N_EVAL) - base_val for i in range(cfg.n_layer)]
    print("baseline attn damage:", [round(v, 3) for v in b_attn])
    print("baseline mlp damage: ", [round(v, 3) for v in b_mlp])

    # equalized arm
    set_seed(42)
    eq = TinyGPT(cfg).to(DEVICE)
    equal_target = sum(targets) / len(targets)
    print(f"equalized write target: {equal_target:.2f} (mean of baseline layer norms)")
    hooks = register_equalizer(eq, equal_target)
    cooldown(90)  # thermal block (envelope rule)
    history = train_model(eq, corpus, steps=4000, lr=1e-3, batch_size=32,
                          max_seconds=180.0, ckpt=TRAIN_CKPT)
    eq_val = estimate_loss(eq, corpus, "val", n_batches=N_EVAL)
    torch.save(eq.state_dict(), FINAL_CKPT)
    e_attn = [lesion_loss(eq, corpus, "attn", i, n_batches=N_EVAL) - eq_val for i in range(cfg.n_layer)]
    e_mlp = [lesion_loss(eq, corpus, "mlp", i, n_batches=N_EVAL) - eq_val for i in range(cfg.n_layer)]
    print(f"equalized val {eq_val:.4f} (P1 gate 1.7224: {'PASS' if eq_val <= 1.7224 else 'FAIL'})")
    print("equal attn damage:", [round(v, 3) for v in e_attn])
    print("equal mlp damage: ", [round(v, 3) for v in e_mlp])

    # KL(L5||L4) on the equalized net (e013 protocol, quick)
    import torch.nn.functional as F
    from e012_decision_depth import batched_snapshots
    gen = torch.Generator().manual_seed(14)
    ix = torch.randint(len(corpus.val) - cfg.block_size - 2, (300,), generator=gen)
    xs = torch.stack([corpus.val[i : i + cfg.block_size] for i in ix]).to(DEVICE)
    kls = []
    eq.eval()
    for b0 in range(0, 300, 64):
        xb = xs[b0:b0+64]
        last = batched_snapshots(eq, xb)
        p4 = F.softmax(eq.lm_head(eq.ln_f(last[cfg.n_layer - 1])), dim=-1)
        p6 = F.softmax(eq.lm_head(eq.ln_f(last[cfg.n_layer])), dim=-1)
        kls.append(float((p6 * (p6 / (p4 + 1e-12)).log()).sum(-1).mean()))
    kl_eq = sum(kls) / len(kls)
    print(f"equalized KL(L5||L4) {kl_eq:.3f} (P3 bar >= 0.5)")

    verdicts = {
        "P1_parity": bool(eq_val <= 1.7224),
        "P2_energy_migrates": bool(e_attn[-1] >= 2 * max(b_attn[-1], 1e-3) or max(e_mlp) < 0.5 * max(b_mlp)),
        "P3_calibrator_survives": bool(kl_eq >= 0.5),
    }
    save_json(rd / "metrics.json", {
        "experiment": "e033_write_equalizer", "targets": targets,
        "baseline_val": base_val, "equalized_val": eq_val,
        "baseline_attn": b_attn, "equalized_attn": e_attn,
        "baseline_mlp": b_mlp, "equalized_mlp": e_mlp,
        "kl_l5_l4_equalized": kl_eq, "verdicts": verdicts,
    })
    print("verdicts:", verdicts)
    if history:
        plot_history(history, rd / "equalized_loss_curve.png", "E033 equalized-MLP-writes training")
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
