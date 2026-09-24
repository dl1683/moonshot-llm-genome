"""E014b — Stream-renorm training: the decisive authority-schedule test (T003 P2).

Design per scratch/e014b_design.md:
- Baseline arm = the E001 checkpoint (no retraining; profile recomputed with
  paired eval batches).
- Renorm arm = fresh model, seed 42, trained identically BUT with a forward
  pre-hook pinning every block-input stream to c = 5.6 per token (median
  baseline block-input norm), active in train AND eval. No ln_f renorm
  (LN(ax)=LN(x) makes it a no-op — the readout cannot break).
- Parity gate: renorm final val <= 1.7224 (= 1.6224 + 0.10). A large gap is
  itself pre-registered evidence that stream growth is optimization-load-
  bearing.
- Re-inflation observable: rho_i = (|w|_renorm / c) / (|w|_base / |x|_base,i);
  mean rho >= 1.3 over L1-L5 => optimization WANTS an authority schedule.
- P2 CONFIRMED iff spread D'_0 - D'_5 <= 1.18 AND D'_5 >= +0.10 AND
  D'_0 < +2.00. REFUTED iff parity passed AND D'_0 >= 2.00 AND D'_5 <= 0.05.
  Bootstrap 2000-resample CI on D'_5 (must exclude the baseline 0.034).
- Control arm a': baseline model evaluated WITH renorm hooks (eval-only) —
  if that barely changes CE, the geometry channel is inert (anomaly).

Run: python lab/e014b_stream_renorm.py
"""
import matplotlib.pyplot as plt
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, estimate_loss, lesion_loss,
                    plot_history, run_dir, save_json, set_seed, train_model)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
TRAIN_CKPT = REPO / "runs" / "checkpoints" / "e014b.train.pt"
RENORM_CKPT = REPO / "runs" / "checkpoints" / "e014b.pt"
C_RENORM = 5.6
SEED = 42
N_EVAL = 30


def register_renorm(model, c=C_RENORM):
    hooks = []
    for block in model.h:
        def pre(m, args, _c=c):
            x = args[0]
            n = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return (x * (_c / n),)
        hooks.append(block.register_forward_pre_hook(pre))
    return hooks


@torch.no_grad()
def write_stream_profile(model, corpus, n_batches=8):
    """Mean per-block write norms (attn+mlp) and block-input norms on val."""
    stats = {"attn": [0.0] * model.cfg.n_layer, "mlp": [0.0] * model.cfg.n_layer,
             "in": [0.0] * model.cfg.n_layer}
    cnt = [0] * model.cfg.n_layer
    handles = []
    for i, block in enumerate(model.h):
        def mk(i, kind):
            def h(m, a, o):
                stats[kind][i] += float(o.norm(dim=-1).mean())
            return h
        def pre(i):
            def h(m, a):
                stats["in"][i] += float(a[0].norm(dim=-1).mean())
            return h
        handles.append(block.attn.register_forward_hook(mk(i, "attn")))
        handles.append(block.mlp.register_forward_hook(mk(i, "mlp")))
        handles.append(block.register_forward_pre_hook(pre(i)))
    gen = torch.Generator().manual_seed(77)
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = corpus.get_batch("val", model.cfg.block_size, 16, gen=gen)
            model(x)
            for i in range(model.cfg.n_layer):
                cnt[i] += 1
    for h in handles:
        h.remove()
    for k in ("attn", "mlp", "in"):
        stats[k] = [round(v / max(cnt[0], 1), 4) for v in stats[k]]
    return stats


def lesion_profile(model, corpus, n_eval=N_EVAL):
    attn = [lesion_loss(model, corpus, "attn", i, n_batches=n_eval) for i in range(model.cfg.n_layer)]
    mlp = [lesion_loss(model, corpus, "mlp", i, n_batches=n_eval) for i in range(model.cfg.n_layer)]
    return attn, mlp


def main():
    set_seed(SEED)
    rd = run_dir("e014b")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)

    # ---- arm a: baseline (reuse E001 weights) ----
    base = TinyGPT(cfg).to(DEVICE)
    base.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    base_val = estimate_loss(base, corpus, "val", n_batches=30)
    base_prof = write_stream_profile(base, corpus)
    print(f"baseline val {base_val:.4f}; write/stream profile {base_prof}")
    b_attn, b_mlp = lesion_profile(base, corpus)
    b_attn = [round(v - base_val, 4) for v in b_attn]
    b_mlp = [round(v - base_val, 4) for v in b_mlp]
    print("baseline attn damage:", b_attn)

    # ---- arm a': eval-only renorm control on baseline weights ----
    hooks = register_renorm(base)
    renorm_eval_ce = estimate_loss(base, corpus, "val", n_batches=30)
    for h in hooks:
        h.remove()
    print(f"baseline + eval-only renorm CE: {renorm_eval_ce:.4f} "
          f"(delta {renorm_eval_ce - base_val:+.4f})")

    # ---- arm b: renorm training ----
    set_seed(SEED)
    rn = TinyGPT(cfg).to(DEVICE)
    hooks = register_renorm(rn)
    history = train_model(rn, corpus, steps=4000, lr=1e-3, batch_size=64,
                          max_seconds=252.0, ckpt=TRAIN_CKPT)
    rn_val = estimate_loss(rn, corpus, "val", n_batches=30)
    rn_prof = write_stream_profile(rn, corpus)
    torch.save(rn.state_dict(), RENORM_CKPT)
    parity_pass = rn_val <= base_val + 0.10
    print(f"renorm arm final val {rn_val:.4f} (parity {'PASS' if parity_pass else 'FAIL'}, "
          f"gate {base_val + 0.10:.4f}); profile {rn_prof}")

    r_attn, r_mlp = lesion_profile(rn, corpus)   # renorm hooks still active
    for h in hooks:
        h.remove()
    r_attn = [round(v - rn_val, 4) for v in r_attn]
    r_mlp = [round(v - rn_val, 4) for v in r_mlp]
    print("renorm attn damage:", r_attn)
    print("renorm mlp damage:", r_mlp)

    # re-inflation rho_i for L1..L5
    rho = []
    for i in range(1, cfg.n_layer):
        w_rn = rn_prof["attn"][i] + rn_prof["mlp"][i]
        w_b = base_prof["attn"][i] + base_prof["mlp"][i]
        x_b = base_prof["in"][i]
        rho.append(round((w_rn / C_RENORM) / (w_b / x_b), 3))
    reinflated = sum(rho) / len(rho) >= 1.3
    print(f"re-inflation rho L1-L5: {rho} (mean {sum(rho)/len(rho):.3f}, criterion >= 1.3: {reinflated})")

    spread = r_attn[0] - r_attn[-1]
    confirmed = (spread <= 1.18) and (r_attn[-1] >= 0.10) and (r_attn[0] < 2.00)
    refuted = parity_pass and (r_attn[0] >= 2.00) and (r_attn[-1] <= 0.05)
    print(f"P2: spread {spread:.3f}, D'_5 {r_attn[-1]:+.3f}, D'_0 {r_attn[0]:+.3f} "
          f"-> {'CONFIRMED' if confirmed else 'REFUTED' if refuted else 'AMBIGUOUS'}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    xs = range(cfg.n_layer)
    axes[0].plot(xs, b_attn, "o-", label="baseline (E001 weights)", color="crimson")
    axes[0].plot(xs, r_attn, "s-", label=f"renorm-trained (val {rn_val:.2f})", color="steelblue")
    axes[0].set_title("attention zero-ablation damage"); axes[0].legend(fontsize=8)
    axes[1].plot(xs, b_mlp, "o-", color="darkorange", label="baseline")
    axes[1].plot(xs, r_mlp, "s-", color="steelblue", label="renorm")
    axes[1].set_title("MLP zero-ablation damage"); axes[1].legend(fontsize=8)
    ws_b = [(base_prof['attn'][i] + base_prof['mlp'][i]) / base_prof['in'][i] for i in xs]
    ws_r = [(rn_prof['attn'][i] + rn_prof['mlp'][i]) / C_RENORM for i in xs]
    axes[2].bar([i - 0.2 for i in xs], ws_b, 0.4, label="baseline write/stream", color="gray")
    axes[2].bar([i + 0.2 for i in xs], ws_r, 0.4, label="renorm write/c", color="steelblue")
    axes[2].set_title("write/stream ratios (+ re-inflation check)"); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("layer"); ax.set_xticks(list(xs)); ax.set_xticklabels([f"L{i}" for i in xs])
    axes[0].set_ylabel("Δ val CE (nats)")
    fig.suptitle("E014b — does renormalized training flatten the lesion map? (T003 P2)")
    fig.tight_layout(); fig.savefig(rd / "stream_renorm.png", dpi=140); plt.close(fig)
    if history:
        plot_history(history, rd / "renorm_loss_curve.png", "E014b renorm-arm training")

    save_json(rd / "metrics.json", {
        "experiment": "e014b_stream_renorm", "c_renorm": C_RENORM, "seed": SEED,
        "baseline_val": base_val, "renorm_val": rn_val, "parity_pass": parity_pass,
        "eval_only_renorm_control_ce": renorm_eval_ce,
        "baseline_damage": {"attn": b_attn, "mlp": b_mlp},
        "renorm_damage": {"attn": r_attn, "mlp": r_mlp},
        "baseline_profile": base_prof, "renorm_profile": rn_prof,
        "reinflation_rho_L1_L5": rho, "reinflation_criterion_mean_ge_1.3": reinflated,
        "P2_spread": spread,
        "P2_confirmed": confirmed, "P2_refuted": refuted,
    })
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
