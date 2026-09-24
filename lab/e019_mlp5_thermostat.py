"""E019 — MLP-5 thermostat: energy vs direction in the long-zombie row (REGISTERED).

Context (E011a/E011c): MLP-5 writes the LARGEST residual in the net (5.64/
token) yet costs only +0.59 to zero; a 60° rotation (same perturbation norm,
destroyed direction) costs just +0.14. Hypothesis: MLP-5 is an ENERGY CARRIER
— the stream needs its write MAGNITUDE, barely its direction.

Instrument: hook-scale MLP-5's write by alpha in {0, 0.5, 1, 2} (alpha=1 =
identity hook, sanity) and ROTATE it 60° at alpha=1 (matched-perturbation
control, e011c machinery reused). Fixed val batches (400 blocks, seed 606).
Measured: mean CE, mean output entropy (nats), mean top-1 prob, mean top-5
prob mass.

Energy-carrier predictions: CE degrades gracefully with alpha (norm channel);
entropy responds monotonically to alpha; rotation ~ free at matched energy.

REGISTERED RULE (as received, verbatim): "if CE(alpha=0.5) and CE(alpha=2)
are both within +0.15 of baseline while rotation(alpha=1) costs >2x
zero-ablation, MLP-5 is confirmed as a distribution-shaper whose ENERGY
matters more than its direction."

FLAG BEFORE RUNNING (deviation note): the ">2x zero-ablation" inequality as
written is INVERTED relative to the e011c energy-carrier finding it cites
(rotate +0.14 vs zero +0.59 — rotation is ~4x CHEAPER than zeroing; if energy
mattered more than direction, rotation must be CHEAP, not >2x the zero cost).
We therefore evaluate BOTH readings and report each:
  R1 (as written):  rotate60_damage > 2 x zero_damage.
  R2 (e011c-consistent energy-presence reading): zero_damage > 2 x rotate60_damage.
The "energy matters more than direction" confirmation is the CONJUNCTION of
graceful alpha-scaling with whichever rotation reading the data supports.

Run: python lab/e019_mlp5_thermostat.py   (requires E001 checkpoint)
"""
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict, run_dir,
                    save_json, set_seed)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
N_EVAL_BLOCKS = 400
EVAL_BS = 64
EVAL_SEED = 606
ALPHAS = [0.0, 0.5, 1.0, 2.0]
GRACEFUL_TOL = 0.15       # registered: |dCE| within +0.15 of baseline
RATIO_BAR = 2.0


def rotated_like(w: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """w' at exactly 60 deg from w per token: ||w'-w|| == ||w||  (e011c)."""
    r = torch.randn(w.shape, generator=gen, device=w.device, dtype=w.dtype)
    wn = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    r = r - (r * w).sum(-1, keepdim=True) * w / (wn * wn)
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return 0.5 * w + (3 ** 0.5 / 2) * r * wn


@torch.no_grad()
def eval_stats(model: TinyGPT, x: torch.Tensor, y: torch.Tensor) -> dict:
    """Mean CE / entropy / top1 / top5-mass over all positions of fixed blocks."""
    model.eval()
    ce_s, ent_s, p1_s, p5_s, n = 0.0, 0.0, 0.0, 0.0, 0
    for i in range(0, len(x), EVAL_BS):
        xb, yb = x[i: i + EVAL_BS], y[i: i + EVAL_BS]
        logits, _ = model(xb, yb)
        logp = F.log_softmax(logits.float(), dim=-1)
        p = logp.exp()
        ce = F.nll_loss(logp.view(-1, logp.size(-1)), yb.view(-1), reduction="sum")
        ent = -(p * logp.clamp_min(-1e9)).sum(-1).sum()
        p1 = p.max(-1).values.sum()
        p5 = p.topk(5, dim=-1).values.sum(-1).sum()
        k = yb.numel()
        ce_s += float(ce.item()); ent_s += float(ent.item())
        p1_s += float(p1.item()); p5_s += float(p5.item()); n += k
    model.train()
    return {"ce": ce_s / n, "entropy": ent_s / n, "top1_prob": p1_s / n, "top5_mass": p5_s / n}


@torch.no_grad()
def run_condition(model: TinyGPT, x, y, kind: str, alpha: float = 1.0) -> dict:
    """kind in {identity, scale, rotate}."""
    mod = model.h[5].mlp
    handle = None
    try:
        if kind == "scale":
            def hook(module, args, out, a=alpha):
                return out * a
            handle = mod.register_forward_hook(hook)
        elif kind == "rotate":
            g = torch.Generator(device=DEVICE).manual_seed(9090)

            def hook(module, args, out, _g=g):
                return rotated_like(out.float(), _g).to(out.dtype)
            handle = mod.register_forward_hook(hook)
        return eval_stats(model, x, y)
    finally:
        if handle is not None:
            handle.remove()


def main():
    assert E001_CKPT.exists(), "run e001 first"
    set_seed(19)
    rd = run_dir("e019")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))

    # fixed val batches
    gen = torch.Generator().manual_seed(EVAL_SEED)
    ix = torch.randint(len(corpus.val) - cfg.block_size - 1, (N_EVAL_BLOCKS,), generator=gen)
    x = torch.stack([corpus.val[i: i + cfg.block_size] for i in ix]).to(DEVICE)
    y = torch.stack([corpus.val[i + 1: i + 1 + cfg.block_size] for i in ix]).to(DEVICE)

    # rotation sanity on a real MLP-5 write tensor (e011c check, reused)
    with torch.no_grad():
        probe_in = torch.randn(4, 8, cfg.n_embd, device=DEVICE)
        probe_out = model.h[5].mlp(probe_in)
        g0 = torch.Generator(device=DEVICE).manual_seed(0)
        wp = rotated_like(probe_out, g0)
        ratio = float(((wp - probe_out).norm(dim=-1) / probe_out.norm(dim=-1)).mean().item())
    print(f"rotation sanity ||w'-w||/||w|| = {ratio:.4f} (expect 1.0)", flush=True)

    conditions = {"baseline": run_condition(model, x, y, "identity")}
    print("baseline:", {k: round(v, 4) for k, v in conditions["baseline"].items()}, flush=True)
    hook_identity = run_condition(model, x, y, "scale", alpha=1.0)
    conditions["alpha=1(identity-hook)"] = hook_identity
    for a in ALPHAS:
        if a == 1.0:
            continue
        conditions[f"alpha={a}"] = run_condition(model, x, y, "scale", alpha=a)
        print(f"alpha={a}:", {k: round(v, 4) for k, v in conditions[f'alpha={a}'].items()}, flush=True)
    conditions["rotate60@alpha=1"] = run_condition(model, x, y, "rotate")
    print("rotate60:", {k: round(v, 4) for k, v in conditions["rotate60@alpha=1"].items()}, flush=True)

    base = conditions["baseline"]
    dmg = {k: v["ce"] - base["ce"] for k, v in conditions.items() if k != "baseline"}
    zero_d = dmg["alpha=0.0"]
    rot_d = dmg["rotate60@alpha=1"]
    graceful = {f"alpha={a}": dmg[f"alpha={a}"] <= GRACEFUL_TOL for a in ALPHAS if a != 1.0}
    R1_as_written = rot_d > RATIO_BAR * zero_d
    R2_e011c_consistent = zero_d > RATIO_BAR * rot_d
    ent_monotone = all(conditions[f"alpha={a}"]["entropy"] <= conditions[f"alpha={b}"]["entropy"] + 1e-4
                       for a, b in zip([0.0, 0.5, 1.0], [0.5, 1.0, 2.0]))
    verdict = {
        "hook_sanity_identity_matches_baseline": abs(hook_identity["ce"] - base["ce"]) < 1e-4,
        "graceful_alpha_within_+0.15": graceful,
        "rotate_over_zero": rot_d / max(zero_d, 1e-6),
        "R1_as_written_rotate_>2x_zero": R1_as_written,
        "R2_e011c_consistent_zero_>2x_rotate": R2_e011c_consistent,
        "entropy_monotone_in_alpha": ent_monotone,
        "energy_carrier_confirmed (graceful AND R2)": all(graceful.values()) and R2_e011c_consistent,
        "as_written_rule_confirmed (graceful AND R1)": all(graceful.values()) and R1_as_written,
    }
    print("\ndamages (dCE):", {k: round(v, 4) for k, v in dmg.items()})
    print("verdicts:", verdict)

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    labels = ["alpha=0", "alpha=0.5", "alpha=1", "alpha=2", "rotate60"]
    keys = ["alpha=0.0", "alpha=0.5", "alpha=1(identity-hook)", "alpha=2.0", "rotate60@alpha=1"]
    colors = ["steelblue"] * 4 + ["darkorange"]
    axes[0].bar(labels, [dmg[k] for k in keys], color=colors)
    axes[0].axhline(GRACEFUL_TOL, color="seagreen", ls=":", lw=1, label="+0.15 graceful bar")
    axes[0].axhline(zero_d, color="gray", ls="--", lw=0.8, label="zero-ablation damage")
    axes[0].set_ylabel("d val CE (nats)"); axes[0].set_title("MLP-5 thermostat: CE damage")
    axes[0].tick_params(axis="x", labelrotation=20); axes[0].legend(fontsize=7)
    for metric, mk in (("entropy", "o-"), ("top1_prob", "s-"), ("top5_mass", "^-")):
        axes[1].plot(ALPHAS, [conditions[k][metric] for k in keys[:4]], mk,
                     label=f"{metric} (alpha sweep)")
        axes[1].plot([2.4], [conditions["rotate60@alpha=1"][metric]], mk[0],
                     color="darkorange", markersize=9)
    axes[1].set_xlabel("alpha (orange marker at x=2.4 = rotate60)"); axes[1].set_ylabel("value")
    axes[1].set_title("distribution shape vs thermostat setting"); axes[1].legend(fontsize=7)
    fig.suptitle("E019 MLP-5 thermostat — energy carrier or distribution shaper?")
    fig.tight_layout()
    fig.savefig(rd / "thermostat.png", dpi=140)
    plt.close(fig)

    save_json(rd / "metrics.json", {
        "experiment": "e019_mlp5_thermostat",
        "n_eval_blocks": N_EVAL_BLOCKS, "eval_seed": EVAL_SEED,
        "rotation_sanity_ratio": ratio,
        "conditions": conditions, "damage_dce": dmg,
        "verdicts": verdict,
        "registered_rule_verbatim": "CE(a=0.5) and CE(a=2) within +0.15 of baseline AND rotate(a=1) "
                                     "costs >2x zero-ablation -> energy>direction",
        "deviation_note": "the >2x inequality is evaluated both as written (R1) and in its "
                          "e011c-consistent form zero>2x-rotate (R2); see module docstring",
        "config": cfg_dict(cfg),
    })
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
