"""E003b — Targeted / projected ascent: can smarter instruments forget selectively?

Question (T002 H4, the live hypothesis): naive ascent is anti-selective (r ~ 1.0
at every dose, E002/E003) but gradient space DOES separate content (cos A-B
0.345 vs 0.144 French). Do targeted (masked) or projected instruments exploit
that separation?

CORRECTED MEASUREMENT (Review-1 critique, mandatory):
  target     = CE on a FIXED sample of TRAIN-A text (corpus 0-45%; memorization
               readout; 400 fixed 256-blocks, seeded). NOT val_a — that set was
               mislabeled B-side text (critique #1).
  collateral = CE on val_B (corpus 95-100%, held out) AND generic val.
  selectivity r(t) = d_target/d_collateral at matched eval steps.

ARMS (all from the E001 checkpoint; AdamW betas (0.9,0.95), grad-clip 1.0,
batch 32x256, 300 steps, eval every 20 — optimizer identical to e003's
ascent_track so the naive arms are true replication anchors):
  1. naive-1e-6 / naive-1e-5 : plain gradient ascent on A (anchors).
  2. projected-1e-5 : step along g_A minus its projection onto the mean
     B-gradient direction u_B (mean over 4 B-batches, UNIT-normalized; u_B
     recomputed every 10 steps). Projection applied to the raw gradient
     before Adam's per-coordinate normalization. Variant
     "projected-rawscale-1e-5" = run-1 accident kept for the record (see
     FIX LOG below).
  3. masked-1e-5 : rank weights by specificity s_i = |mean g_A,i| /
     (|mean g_B,i| + eps) over 8 A-batches and 8 B-batches; ascent only on the
     top-10% mask (grad zeroed elsewhere).
  4. masked+projected-1e-5 : project, then mask (bonus arm).

DEVIATIONS from the task sheet (noted per protocol):
  - B-side gradient SOURCE for projection/masking = TRAIN-B (corpus 45-90%),
    matching E003's gradient-cosine definition of "B"; collateral EVALUATION
    is on held-out val_B (95-100%) + generic val, per the corrected rules.
    (Train-B is what the model shares with A; val_B is pure held-out fluency.)
  - projected/masked/combined run at lr 1e-5 only (gentle, matched to the
    naive anchor); eps = 1e-12 in the specificity score.
  - The mask is computed ONCE from the base checkpoint and held fixed for the
    whole run (re-ranking each step was not specified and would blur the
    "wrong instrument vs wrong direction" question).
  - r(t) verdicts use val_B as primary collateral; r vs generic val recorded
    as secondary. Generic val = fixed 400-block sample of the full val split
    (90-100%, which contains val_B — a fluency readout, overlap noted).

FIX LOG (run 2): the first run's projected arm subtracted (g.u)u with the
UNNORMALIZED mean B-gradient u (||u|| = 0.84), i.e. it removed |u|^2 ~ 0.70x
of the true B-component (under-removal, no accidental retain term: residual
motion stayed cos +0.26 along +grad_B). That accidental variant is kept and
re-run as arm "projected-rawscale-1e-5". The registered instrument
("projected-1e-5") now uses a UNIT B-direction — verified cos(u_B,
projected gradient) = 1.1e-7 at the base checkpoint. Base geometry:
cos(mean g_A, mean g_B) = 0.674 at the 4-batch-mean level (vs 0.345
batch-level in E003).

Registered verdicts (T002): projected r >= 1.5 -> instrument works; masked
r >= 2 -> works; both fail (r <= 1.2) -> claim 5 hardens to "first-order
selective forgetting impossible at this granularity"; next family is
second-order or weight-surgery unlearning.

Run: python lab/e003b_targeted_ascent.py   (requires E001 checkpoint)
"""
import copy

import matplotlib.pyplot as plt
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict, run_dir,
                    save_json, set_seed)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
SEED = 27182
STEPS = 300
EVAL_EVERY = 20
BATCH = 32
N_EVAL_BLOCKS = 400          # fixed eval blocks per set
EVAL_BS = 64
B_DIR_BATCHES = 4            # batches for the mean B-gradient direction
B_DIR_REFRESH = 10           # recompute u_B every N steps
MASK_A_BATCHES, MASK_B_BATCHES = 8, 8
MASK_FRAC = 0.10
EPS = 1e-12
R_TARGET_LEVEL = 0.5         # nat of target damage for the operating-point r
R_MIN_DT = 0.05              # minimum d_target before an r(t) point counts


# ---------------------------------------------------------------- helpers

def batch_from(src: torch.Tensor, block: int, bs: int, gen: torch.Generator):
    ix = torch.randint(len(src) - block - 1, (bs,), generator=gen)
    x = torch.stack([src[i: i + block] for i in ix]).to(DEVICE)
    y = torch.stack([src[i + 1: i + 1 + block] for i in ix]).to(DEVICE)
    return x, y


def fixed_blocks(src: torch.Tensor, block: int, n: int, seed: int):
    """One fixed, seeded sample of n blocks — the memorization/collateral readout."""
    gen = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(src) - block - 1, (n,), generator=gen)
    x = torch.stack([src[i: i + block] for i in ix])
    y = torch.stack([src[i + 1: i + 1 + block] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def ce_fixed(model: TinyGPT, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), EVAL_BS):
        xb, yb = x[i: i + EVAL_BS], y[i: i + EVAL_BS]
        _, loss = model(xb, yb)
        tot += float(loss.item()) * len(xb)
        n += len(xb)
    model.train()
    return tot / max(n, 1)


def zero_grads(model: TinyGPT) -> None:
    for p in model.parameters():
        p.grad = None


def flat_grad(model: TinyGPT) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def split_flat(flat: torch.Tensor, model: TinyGPT) -> list[torch.Tensor]:
    """Split a flat vector into per-parameter views (order of model.parameters())."""
    out, pos = [], 0
    for p in model.parameters():
        n = p.numel()
        out.append(flat[pos: pos + n].view_as(p).clone())
        pos += n
    assert pos == flat.numel()
    return out


def mean_grad(model: TinyGPT, src: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """Flat mean loss-gradient over k batches drawn from src."""
    acc = None
    for _ in range(k):
        zero_grads(model)
        x, y = batch_from(src, model.cfg.block_size, BATCH, gen)
        _, loss = model(x, y)
        loss.backward()
        g = flat_grad(model)
        acc = g.clone() if acc is None else acc + g
    zero_grads(model)
    return acc / k


def project_out(model: TinyGPT, u_params: list[torch.Tensor]) -> float:
    """Remove from the current gradient its component along u (UNIT norm).
    Returns the removed cosine (dot with unit u)."""
    dot = torch.zeros((), device=DEVICE)
    for p, up in zip(model.parameters(), u_params):
        if p.grad is not None:
            dot = dot + (p.grad * up).sum()
    for p, up in zip(model.parameters(), u_params):
        if p.grad is not None:
            p.grad.add_(-dot * up)
    return float(dot.item())


def b_direction(m: TinyGPT, train_b, gen: torch.Generator, unit: bool) -> list[torch.Tensor]:
    """Mean B-gradient direction as per-param tensors; unit=True normalizes
    (the registered projection); unit=False keeps the raw mean gradient scale
    (run-1 accident: removes |u|^2 ~ 0.7x of the true component)."""
    flat = mean_grad(m, train_b, B_DIR_BATCHES, gen)
    if unit:
        flat = flat / (flat.norm() + 1e-12)
    return split_flat(flat, m)


def apply_mask(model: TinyGPT, mask_params: list[torch.Tensor]) -> None:
    for p, mp in zip(model.parameters(), mask_params):
        if p.grad is not None:
            p.grad.mul_(mp)


# ---------------------------------------------------------------- arms

def run_arm(base: TinyGPT, train_a, train_b, evals: dict, lr: float, mode: str, tag: str):
    """mode in {naive, projected, masked, combined}."""
    m = copy.deepcopy(base)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(SEED)

    u_params = None
    if mode in ("projected", "combined"):
        u_params = b_direction(m, train_b, gen, unit=True)
    elif mode == "projected-raw":
        u_params = b_direction(m, train_b, gen, unit=False)

    mask_params = None
    mask_diag = {}
    if mode in ("masked", "combined"):
        gmask = torch.Generator().manual_seed(SEED + 1)  # independent of training draws
        ga = mean_grad(m, train_a, MASK_A_BATCHES, gmask)
        gb = mean_grad(m, train_b, MASK_B_BATCHES, gmask)
        s = ga.abs() / (gb.abs() + EPS)
        thr = torch.quantile(s, 1.0 - MASK_FRAC)
        flat_mask = (s >= thr).float()
        mask_params = split_flat(flat_mask, m)
        mask_diag = {
            "n_selected": int(flat_mask.sum().item()),
            "n_total": int(flat_mask.numel()),
            "threshold": float(thr.item()),
            "gA_mass_in_mask": float((ga.abs() * flat_mask).sum().item() / ga.abs().sum().item()),
            "gB_mass_in_mask": float((gb.abs() * flat_mask).sum().item() / gb.abs().sum().item()),
            "cos_mean_gA_mean_gB": float(torch.nn.functional.cosine_similarity(ga, gb, dim=0).item()),
        }
        print(f"  [{tag}] mask: {mask_diag['n_selected']}/{mask_diag['n_total']} weights "
              f"({100 * mask_diag['gA_mass_in_mask']:.1f}% of |g_A| mass, "
              f"{100 * mask_diag['gB_mass_in_mask']:.1f}% of |g_B| mass)", flush=True)

    traj, removed = [], []
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            point = {"step": step}
            for name, (x, y) in evals.items():
                point[name] = ce_fixed(m, x, y)
            traj.append(point)
            print(f"  [{tag}] s{step:3d} " + " ".join(f"{k}={v:.4f}" for k, v in point.items() if k != "step"),
                  flush=True)
        if step == STEPS:
            break
        if mode in ("projected", "combined", "projected-raw") and step % B_DIR_REFRESH == 0:
            u_params = b_direction(m, train_b, gen, unit=(mode != "projected-raw"))
        zero_grads(m)
        xa, ya = batch_from(train_a, m.cfg.block_size, BATCH, gen)
        _, loss_a = m(xa, ya)
        (-loss_a).backward()                      # ascent on A
        if mode in ("projected", "combined", "projected-raw"):
            removed.append(project_out(m, u_params))  # cosine along u_B removed
        if mode in ("masked", "combined"):
            apply_mask(m, mask_params)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    diag = dict(mask_diag)
    if removed:
        diag["mean_removed_cos"] = sum(removed) / len(removed)
        diag["mean_removed_cos_last50"] = sum(removed[-50:]) / len(removed[-50:])
    return m, traj, diag


def r_trajectory(traj: list[dict], t_key: str, c_key: str, base: dict):
    """r(t) = d_target/d_collateral at matched steps (None where undefined)."""
    out = []
    for p in traj:
        dt, dc = p[t_key] - base[t_key], p[c_key] - base[c_key]
        if p["step"] == 0:
            out.append(None)
        elif dt >= R_MIN_DT and dc > 1e-4:
            out.append(dt / dc)
        else:
            out.append(None)
    return out


def peak_r(r_traj):
    vals = [r for r in r_traj if r is not None and r < 1e3]
    return max(vals) if vals else None


def r_at_target(traj, t_key, c_key, base, level=R_TARGET_LEVEL):
    """r interpolated at d_target == level (first crossing)."""
    prev = None
    for p in traj:
        dt, dc = p[t_key] - base[t_key], p[c_key] - base[c_key]
        if prev is not None and prev[0] < level <= dt:
            f = (level - prev[0]) / max(dt - prev[0], 1e-9)
            dc_interp = prev[1] + f * (dc - prev[1])
            return level / dc_interp if dc_interp > 1e-6 else None
        prev = (dt, dc)
    return None


# ---------------------------------------------------------------- main

def main():
    assert E001_CKPT.exists(), "run e001 first (need runs/checkpoints/e001.pt)"
    set_seed(SEED)
    rd = run_dir("e003b")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    base = TinyGPT(cfg).to(DEVICE)
    base.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))

    train_a = corpus.slice("train", 0.0, 0.5)   # corpus 0-45%  (A: to forget)
    train_b = corpus.slice("train", 0.5, 1.0)   # corpus 45-90% (B: gradient source)
    val_b = corpus.slice("val", 0.5, 1.0)       # corpus 95-100% (collateral readout)

    evals = {
        "target": fixed_blocks(train_a, cfg.block_size, N_EVAL_BLOCKS, seed=101),
        "val_B": fixed_blocks(val_b, cfg.block_size, N_EVAL_BLOCKS, seed=202),
        "generic": fixed_blocks(corpus.val, cfg.block_size, N_EVAL_BLOCKS, seed=303),
    }
    base_ce = {name: ce_fixed(base, x, y) for name, (x, y) in evals.items()}
    print("baseline CE:", {k: round(v, 4) for k, v in base_ce.items()},
          "(target = TRAIN-A memorization readout; val_B/generic = collateral)", flush=True)

    arms = [
        ("naive-1e-6", 1e-6, "naive"),
        ("naive-1e-5", 1e-5, "naive"),
        ("projected-1e-5", 1e-5, "projected"),
        ("projected-rawscale-1e-5", 1e-5, "projected-raw"),
        ("masked-1e-5", 1e-5, "masked"),
        ("masked+projected-1e-5", 1e-5, "combined"),
    ]

    results = {}
    for tag, lr, mode in arms:
        print(f"arm: {tag}")
        _, traj, diag = run_arm(base, train_a, train_b, evals, lr, mode, tag)
        results[tag] = {
            "lr": lr, "mode": mode, "traj": traj, "diag": diag,
            "r_vs_valB": r_trajectory(traj, "target", "val_B", base_ce),
            "r_vs_generic": r_trajectory(traj, "target", "generic", base_ce),
            "peak_r_valB": peak_r(r_trajectory(traj, "target", "val_B", base_ce)),
            "peak_r_generic": peak_r(r_trajectory(traj, "target", "generic", base_ce)),
            "r_at_dT0.5_valB": r_at_target(traj, "target", "val_B", base_ce),
            "final_deltas": {k: traj[-1][k] - base_ce[k] for k in ("target", "val_B", "generic")},
        }

    # ---- verdicts (registered thresholds) ----
    pk = {t: results[t]["peak_r_valB"] for t in results}
    verdicts = {
        "projected_works_r_ge_1.5": pk["projected-1e-5"] is not None and pk["projected-1e-5"] >= 1.5,
        "masked_works_r_ge_2.0": pk["masked-1e-5"] is not None and pk["masked-1e-5"] >= 2.0,
        "both_fail_r_le_1.2": all(pk[t] is not None and pk[t] <= 1.2
                                  for t in ("projected-1e-5", "masked-1e-5")),
        "claim5_hardens_next_family_second_order_or_weight_surgery":
            all(pk[t] is not None and pk[t] <= 1.2
                for t in ("projected-1e-5", "masked-1e-5", "masked+projected-1e-5")),
    }
    print("\n=== peak r (target vs val_B collateral) ===")
    for t, v in pk.items():
        print(f"  {t:24s} peak r = {v if v is None else round(v, 3)}   "
              f"(r at dTarget=0.5: {results[t]['r_at_dT0.5_valB']})")
    print("verdicts:", verdicts)

    # ---- graphs ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"naive-1e-6": "tab:gray", "naive-1e-5": "black", "projected-1e-5": "crimson",
              "projected-rawscale-1e-5": "salmon", "masked-1e-5": "seagreen",
              "masked+projected-1e-5": "purple"}
    for tag, res in results.items():
        st = [p["step"] for p, r in zip(res["traj"], res["r_vs_valB"]) if r is not None]
        rv = [r for r in res["r_vs_valB"] if r is not None]
        axes[0].plot(st, rv, "o-", ms=3, color=colors[tag], label=tag)
        axes[1].plot([p["val_B"] - base_ce["val_B"] for p in res["traj"]],
                     [p["target"] - base_ce["target"] for p in res["traj"]],
                     "o-", ms=3, color=colors[tag], label=tag)
    axes[0].axhline(1.0, color="k", ls="--", lw=0.8, label="anti-selective r=1")
    axes[0].axhline(1.5, color="crimson", ls=":", lw=1, label="projected threshold 1.5")
    axes[0].axhline(2.0, color="seagreen", ls=":", lw=1, label="masked threshold 2.0")
    axes[0].set_xlabel("ascent step"); axes[0].set_ylabel("r(t) = Δtarget/Δval_B")
    axes[0].set_title("selectivity r(t) — corrected target (TRAIN-A CE)")
    axes[0].legend(fontsize=7)
    lim = max(1.0, max((r for res in results.values() for r in res["r_vs_valB"] if r is not None), default=1.0))
    axes[0].set_ylim(0, min(lim * 1.15, 6.0))
    axes[1].plot([0, max(p["val_B"] - base_ce["val_B"] for res in results.values() for p in res["traj"])],
                 [0, max(p["val_B"] - base_ce["val_B"] for res in results.values() for p in res["traj"])],
                 "k--", lw=0.8)
    axes[1].axvline(0.1, color="g", ls=":", lw=1); axes[1].axhline(1.0, color="g", ls=":", lw=1)
    axes[1].set_xlabel("Δ val_B CE (collateral, nats)"); axes[1].set_ylabel("Δ TRAIN-A CE (target, nats)")
    axes[1].set_title("parametric damage: target vs collateral")
    axes[1].legend(fontsize=7)
    fig.suptitle("E003b targeted ascent — projected/masked instruments vs naive (corrected measurement)")
    fig.tight_layout()
    fig.savefig(rd / "selectivity.png", dpi=140)
    plt.close(fig)

    save_json(rd / "metrics.json", {
        "experiment": "e003b_targeted_ascent", "seed": SEED, "steps": STEPS,
        "eval_every": EVAL_EVERY, "batch": BATCH, "n_eval_blocks": N_EVAL_BLOCKS,
        "measurement": {
            "target": "TRAIN-A (corpus 0-45%) CE, 400 fixed 256-blocks (memorization readout)",
            "collateral": "val_B (95-100%) + generic val (90-100%), 400 fixed blocks each",
            "r": "d_target/d_collateral at matched steps; peak over points with d_target>=0.05",
        },
        "baseline_ce": base_ce,
        "b_gradient_source": "train-B (corpus 45-90%), 4-batch mean, refreshed every 10 steps",
        "mask_rule": "s_i=|mean g_A|/(|mean g_B|+1e-12), 8+8 batches at base ckpt, top 10%, fixed",
        "arms": results,
        "peak_r_valB": pk,
        "verdicts": verdicts,
        "registered_thresholds": {"projected": 1.5, "masked": 2.0, "both_fail": 1.2},
        "config": cfg_dict(cfg),
    })
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
