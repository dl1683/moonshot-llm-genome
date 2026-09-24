"""E003c — projected-ascent dose-response to the FORGETTING BAR (REGISTERED).

Review-3 correction being applied: e003b showed projected ascent is SELECTIVE
SO FAR (peak r 6.1 at gentle doses) but the forgetting bar was never reached —
the run stopped at dTarget 0.28. The bar = the train/val memorization gap:
baseline train-A CE 1.0180 vs val_B 1.6814 -> gap 0.6635 nats. Reaching it
means train-A CE has risen to the level where the model held out val_B text
from the start: the memorization advantage on A is fully erased.

PROTOCOL (e003b machinery reused verbatim: unit u_B projection before Adam,
u_B = mean grad over 4 TRAIN-B batches, refreshed every 10 steps, AdamW
(0.9,0.95), grad-clip 1.0, batch 32x256, lr 1e-5, from the E001 checkpoint):
  1. projected arm: up to 2000 steps, eval every 50. Logged: train-A CE
     (target), val_B CE (collateral), TRAIN-B CE (second collateral —
     memorization-symmetric control, never measured before), generic val CE.
  2. step-norm-matched naive control: naive ascent (same protocol, no
     projection) at lr scaled so its per-step weight-displacement norm at
     step 10 matches the projected arm's (both measured at lr 1e-5, 10-step
     probe; naive lr = 1e-5 * proj_norm10/naive_norm10; match re-verified on
     the full runs).

VERDICT RULES (registered):
  (a) r = dTarget/dVal_B at dTarget = gap (0.6635):
      r >= 2.0            -> claim 5 upgrades to "projected ascent
                             selectively forgets to the bar";
      r < 1.5 before bar  -> claim 5 downgrades to "projection buys a
                             selective head-start that doesn't survive real
                             forgetting";
      otherwise           -> intermediate, report the curve.
  (b) does train-B collateral track val_B (fluency-substrate damage) or
      diverge toward target-rate (memorization-symmetric wipe)?
  (c) r(dose) curve shape.

DEVIATIONS (noted per protocol):
  - the gap is COMPUTED from this run's baselines (0.6635) rather than
    hard-coded 0.66; same quantity to 2 decimals.
  - naive arm: eval every 10 steps (denser — it crosses the whole dose range
    in <100 steps; every-50 would give 1-2 points below the bar), early stop
    once dTarget > 3.0 (CE then runs to the ~24-nat attractor with no dose
    resolution left; 3.0 is 4.5x past the bar).
  - naive probe for the norm match: separate 10-step run at lr 1e-5
    (identical seed/code path), because the matched lr must be known before
    the naive arm starts.
  - per-step displacement norms recorded on every step of every arm (this is
    how proj_norm10 is obtained; probe and main projected run share the
    deterministic path so their step-10 norms agree exactly).

POST-RUN REPRODUCIBILITY FINDING (folded into metrics.json, 2026-09-24):
  this run's seed is 31415; it showed NO r>=2 head-start (peak 1.52). A
  replication on e003b's original seed 27182 (800 steps) and a re-run of
  e003b's own unmodified run_arm at seed 27182 BOTH give the same fast
  trajectory (mutually bit-identical, e.g. s280 target 16.7069) with peak r
  1.46-1.73 — NOT the recorded gentle r~5-6 walk of runs/e003b. The e003b
  head-start trajectory is therefore NON-REPRODUCIBLE (chaotic ascent /
  nondeterministic kernels); the reliable behavior of projected ascent is
  the one characterized here. Verdict (a) below uses the INTERPOLATED r(dose)
  curve for the "collapses below 1.5 before the bar" test (discrete eval
  points alone can straddle the collapse between samples).

Run: python lab/e003c_projected_dose_response.py   (requires E001 checkpoint)
"""
import copy

import matplotlib.pyplot as plt
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict, run_dir,
                    save_json, set_seed)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
SEED = 31415
LR = 1e-5
STEPS = 2000
EVAL_EVERY_PROJ = 50
EVAL_EVERY_NAIVE = 10
BATCH = 32
N_EVAL_BLOCKS = 400
EVAL_BS = 64
B_DIR_BATCHES = 4
B_DIR_REFRESH = 10
R_MIN_DT = 0.05               # minimum dTarget before an r point counts
NAIVE_STOP_DT = 3.0           # early stop for the naive arm (4.5x past bar)
MATCH_STEP = 10               # step at which displacement norms are matched


# ---------------------------------------------------------------- helpers
# (verbatim from e003b unless noted)

def batch_from(src: torch.Tensor, block: int, bs: int, gen: torch.Generator):
    ix = torch.randint(len(src) - block - 1, (bs,), generator=gen)
    x = torch.stack([src[i: i + block] for i in ix]).to(DEVICE)
    y = torch.stack([src[i + 1: i + 1 + block] for i in ix]).to(DEVICE)
    return x, y


def fixed_blocks(src: torch.Tensor, block: int, n: int, seed: int):
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


def flat_params(model: TinyGPT) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def split_flat(flat: torch.Tensor, model: TinyGPT) -> list[torch.Tensor]:
    out, pos = [], 0
    for p in model.parameters():
        n = p.numel()
        out.append(flat[pos: pos + n].view_as(p).clone())
        pos += n
    assert pos == flat.numel()
    return out


def mean_grad(model: TinyGPT, src: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
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
    dot = torch.zeros((), device=DEVICE)
    for p, up in zip(model.parameters(), u_params):
        if p.grad is not None:
            dot = dot + (p.grad * up).sum()
    for p, up in zip(model.parameters(), u_params):
        if p.grad is not None:
            p.grad.add_(-dot * up)
    return float(dot.item())


def b_direction(m: TinyGPT, train_b, gen: torch.Generator) -> list[torch.Tensor]:
    flat = mean_grad(m, train_b, B_DIR_BATCHES, gen)
    flat = flat / (flat.norm() + 1e-12)
    return split_flat(flat, m)


# ---------------------------------------------------------------- arm

def run_arm(base: TinyGPT, train_a, train_b, evals: dict, lr: float, mode: str,
            tag: str, steps: int, eval_every: int):
    """mode in {naive, projected}. Records per-step displacement norms."""
    m = copy.deepcopy(base)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(SEED)

    u_params = b_direction(m, train_b, gen) if mode == "projected" else None

    traj, removed, step_norms = [], [], []
    prev_flat = flat_params(m)
    for step in range(steps + 1):
        if evals and step % eval_every == 0:
            point = {"step": step}
            for name, (x, y) in evals.items():
                point[name] = ce_fixed(m, x, y)
            traj.append(point)
            print(f"  [{tag}] s{step:4d} " + " ".join(
                f"{k}={v:.4f}" for k, v in point.items() if k != "step"), flush=True)
            if mode == "naive" and point["target"] - base_ce_cache["target"] > NAIVE_STOP_DT:
                print(f"  [{tag}] early stop: dTarget > {NAIVE_STOP_DT} (past the bar with margin)")
                break
        if step == steps:
            break
        if mode == "projected" and step % B_DIR_REFRESH == 0:
            u_params = b_direction(m, train_b, gen)
        zero_grads(m)
        xa, ya = batch_from(train_a, m.cfg.block_size, BATCH, gen)
        _, loss_a = m(xa, ya)
        (-loss_a).backward()                      # ascent on A
        if mode == "projected":
            removed.append(project_out(m, u_params))
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        cur_flat = flat_params(m)
        step_norms.append(float((cur_flat - prev_flat).norm().item()))
        prev_flat = cur_flat

    diag = {}
    if removed:
        diag["mean_removed_cos"] = sum(removed) / len(removed)
        diag["mean_removed_cos_last200"] = sum(removed[-200:]) / len(removed[-200:])
    diag["step_norm_at_10"] = step_norms[MATCH_STEP - 1] if len(step_norms) >= MATCH_STEP else None
    diag["mean_step_norm"] = sum(step_norms) / len(step_norms) if step_norms else None
    return m, traj, diag, step_norms


def r_trajectory(traj, t_key, c_key, base):
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


def r_at_target(traj, t_key, c_key, base, level):
    """Interpolated dTarget/dCollateral at first crossing of dTarget==level."""
    prev = None
    for p in traj:
        dt, dc = p[t_key] - base[t_key], p[c_key] - base[c_key]
        if prev is not None and prev[0] < level <= dt:
            f = (level - prev[0]) / max(dt - prev[0], 1e-9)
            dc_interp = prev[1] + f * (dc - prev[1])
            return level / dc_interp if dc_interp > 1e-6 else None
        prev = (dt, dc)
    return None


def min_r_before(traj, r_traj, base, t_key, level):
    vals = [r for p, r in zip(traj, r_traj)
            if r is not None and R_MIN_DT <= p[t_key] - base[t_key] < level]
    return min(vals) if vals else None


def r_below_dose(traj, r_traj, base, t_key, r_bar, level):
    """First dose where the PIECEWISE-LINEAR r(dose) interpolant crosses
    below r_bar, searching only doses < level. None if it never does."""
    pts = sorted((p[t_key] - base[t_key], r) for p, r in zip(traj, r_traj)
                 if r is not None)
    for (d0, r0), (d1, r1) in zip(pts, pts[1:]):
        hi_d = min(d1, level)
        if d0 >= level:
            break
        if r0 < r_bar:
            return d0
        if r1 < r_bar and hi_d > d0:
            return d0 + (r0 - r_bar) / max(r0 - r1, 1e-9) * (hi_d - d0)
    return None


def dose_curve_shape(traj, r_traj, base):
    """(c): classify r(dose). Split valid points at the median dose."""
    pts = [(p["target"] - base["target"], r) for p, r in zip(traj, r_traj) if r is not None]
    if len(pts) < 4:
        return {"n_points": len(pts), "class": "insufficient"}
    pts.sort()
    half = len(pts) // 2
    lo = sum(r for _, r in pts[:half]) / half
    hi = sum(r for _, r in pts[half:]) / (len(pts) - half)
    if hi > lo * 1.25:
        cls = "increasing (selectivity improves with dose)"
    elif lo > hi * 1.25:
        cls = "declining (selectivity decays with dose)"
    else:
        cls = "flat"
    return {"n_points": len(pts), "mean_r_low_dose": lo, "mean_r_high_dose": hi, "class": cls}


# ---------------------------------------------------------------- main

base_ce_cache: dict = {}


def main():
    assert E001_CKPT.exists(), "run e001 first (need runs/checkpoints/e001.pt)"
    set_seed(SEED)
    rd = run_dir("e003c")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    base = TinyGPT(cfg).to(DEVICE)
    base.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))

    train_a = corpus.slice("train", 0.0, 0.5)
    train_b = corpus.slice("train", 0.5, 1.0)
    val_b = corpus.slice("val", 0.5, 1.0)

    evals = {
        "target": fixed_blocks(train_a, cfg.block_size, N_EVAL_BLOCKS, seed=101),
        "train_B": fixed_blocks(train_b, cfg.block_size, N_EVAL_BLOCKS, seed=404),
        "val_B": fixed_blocks(val_b, cfg.block_size, N_EVAL_BLOCKS, seed=202),
        "generic": fixed_blocks(corpus.val, cfg.block_size, N_EVAL_BLOCKS, seed=303),
    }
    base_ce = {name: ce_fixed(base, x, y) for name, (x, y) in evals.items()}
    base_ce_cache.update(base_ce)
    gap = base_ce["val_B"] - base_ce["target"]          # the forgetting bar
    print("baseline CE:", {k: round(v, 4) for k, v in base_ce.items()}, flush=True)
    print(f"forgetting bar (val_B - target gap) = {gap:.4f} nats", flush=True)

    # ---- step-norm match: naive probe at lr 1e-5, 10 steps, no evals ----
    _, _, naive_probe_diag, _ = run_arm(base, train_a, train_b, {}, LR, "naive",
                                        "naive-probe-1e-5", MATCH_STEP, eval_every=10 ** 9)
    naive_n10 = naive_probe_diag["step_norm_at_10"]
    # the projected main run records its own step-10 norm (same deterministic path)

    # ---- arm 1: projected ascent, 2000 steps ----
    print("arm: projected-1e-5 (2000 steps)")
    _, traj_p, diag_p, norms_p = run_arm(base, train_a, train_b, evals, LR, "projected",
                                         "projected-1e-5", STEPS, EVAL_EVERY_PROJ)
    proj_n10 = diag_p["step_norm_at_10"]

    naive_lr = LR * proj_n10 / naive_n10
    print(f"step-norm match @step10: projected {proj_n10:.6f} vs naive(1e-5) {naive_n10:.6f} "
          f"-> matched naive lr = {naive_lr:.3e}", flush=True)

    # ---- arm 2: naive ascent at matched step norm ----
    print(f"arm: naive-matched (lr {naive_lr:.3e})")
    _, traj_n, diag_n, norms_n = run_arm(base, train_a, train_b, evals, naive_lr, "naive",
                                         f"naive-matched-{naive_lr:.1e}", STEPS, EVAL_EVERY_NAIVE)
    print(f"verified match @step10: naive-matched {diag_n['step_norm_at_10']:.6f} "
          f"vs projected {proj_n10:.6f} "
          f"(ratio {diag_n['step_norm_at_10'] / proj_n10:.3f})", flush=True)

    arms = {
        "projected-1e-5": {"lr": LR, "traj": traj_p, "diag": diag_p, "step_norms_first20": norms_p[:20]},
        f"naive-matched-{naive_lr:.2e}": {"lr": naive_lr, "traj": traj_n, "diag": diag_n,
                                          "step_norms_first20": norms_n[:20]},
    }

    # ---- verdicts (registered) ----
    ptag = "projected-1e-5"
    ntag = f"naive-matched-{naive_lr:.2e}"
    r_p_valB = r_trajectory(traj_p, "target", "val_B", base_ce)
    r_p_trainB = r_trajectory(traj_p, "target", "train_B", base_ce)
    r_n_valB = r_trajectory(traj_n, "target", "val_B", base_ce)

    reached_p = any(p["target"] - base_ce["target"] >= gap for p in traj_p)
    r_gap_p = r_at_target(traj_p, "target", "val_B", base_ce, gap)
    r_gap_p_trainB = r_at_target(traj_p, "target", "train_B", base_ce, gap)
    r_gap_n = r_at_target(traj_n, "target", "val_B", base_ce, gap)
    min_r_before_bar = min_r_before(traj_p, r_p_valB, base_ce, "target", gap)
    collapse_dose = r_below_dose(traj_p, r_p_valB, base_ce, "target", 1.5, gap)
    shape = dose_curve_shape(traj_p, r_p_valB, base_ce)

    dtB_at_gap = r_at_target(traj_p, "target", "train_B", base_ce, gap)  # for ratio below
    valB_at_gap = gap / r_gap_p if r_gap_p else None
    trainB_coll_at_gap = gap / dtB_at_gap if dtB_at_gap else None

    verdict_a = ("UPGRADE (r>=2 at bar)" if (reached_p and r_gap_p is not None and r_gap_p >= 2.0)
                 else "DOWNGRADE (r collapses below 1.5 before the bar)"
                 if collapse_dose is not None
                 else "INTERMEDIATE" if reached_p
                 else "BAR NOT REACHED — selective-so-far persists, still untested to the bar")

    results = {
        "baseline_ce": base_ce, "forgetting_bar_gap": gap,
        "step_norm_match": {
            "proj_step10_norm": proj_n10, "naive_probe_step10_norm_at_1e-5": naive_n10,
            "naive_matched_lr": naive_lr, "naive_matched_step10_norm": diag_n["step_norm_at_10"],
            "verified_ratio": diag_n["step_norm_at_10"] / proj_n10,
        },
        "arms": arms,
        "r_projected_vs_valB": r_p_valB,
        "r_projected_vs_trainB": r_p_trainB,
        "r_naive_vs_valB": r_n_valB,
        "r_at_gap": {"projected_vs_valB": r_gap_p, "projected_vs_trainB": r_gap_p_trainB,
                     "naive_vs_valB": r_gap_n},
        "min_r_before_bar_projected": min_r_before_bar,
        "collapse_below_1.5_dose_projected": collapse_dose,
        "final_deltas": {
            "projected": {k: traj_p[-1][k] - base_ce[k] for k in base_ce},
            "naive": {k: traj_n[-1][k] - base_ce[k] for k in base_ce},
        },
        "verdicts": {
            "(a)_r_at_bar": verdict_a,
            "(a)_bar_reached": reached_p,
            "(b)_trainB_over_valB_at_gap": (trainB_coll_at_gap / valB_at_gap
                                            if trainB_coll_at_gap and valB_at_gap else None),
            "(c)_dose_curve": shape,
        },
    }

    print("\n=== VERDICTS ===")
    print(f"(a) r at bar (dTarget={gap:.3f}): projected {r_gap_p}, naive {r_gap_n}, "
          f"reached={reached_p}, min r before bar={min_r_before_bar} -> {verdict_a}")
    print(f"(b) trainB collateral at gap {trainB_coll_at_gap} vs valB {valB_at_gap} "
          f"(ratio {results['verdicts']['(b)_trainB_over_valB_at_gap']})")
    print(f"(c) dose curve: {shape}")

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for tag, res, col in ((ptag, arms[ptag], "crimson"), (ntag, arms[ntag], "black")):
        st = [p["target"] - base_ce["target"] for p, r in zip(res["traj"],
                                                              r_trajectory(res["traj"], "target", "val_B", base_ce))
              if r is not None]
        rv = [r for r in r_trajectory(res["traj"], "target", "val_B", base_ce) if r is not None]
        axes[0].plot(st, rv, "o-", ms=3, color=col, label=tag)
        axes[1].plot([p["step"] for p in res["traj"]], [p["target"] - base_ce["target"] for p in res["traj"]],
                     "o-", ms=3, color=col, label=f"{tag} Δtarget")
    axes[0].axvline(gap, color="navy", ls="-.", lw=1.2, label=f"forgetting bar Δ={gap:.2f}")
    axes[0].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[0].axhline(1.5, color="crimson", ls=":", lw=1)
    axes[0].axhline(2.0, color="seagreen", ls=":", lw=1)
    axes[0].set_xlabel("dose = Δ train-A CE (nats)"); axes[0].set_ylabel("r = Δtarget/Δval_B")
    axes[0].set_title("r(dose): projected vs step-norm-matched naive")
    axes[0].legend(fontsize=7)
    axes[1].axhline(gap, color="navy", ls="-.", lw=1.2)
    axes[1].set_xlabel("ascent step"); axes[1].set_ylabel("Δ CE (nats)")
    axes[1].set_title("target damage trajectories (bar = navy dash-dot)")
    axes[1].legend(fontsize=7)
    fig.suptitle("E003c projected-ascent dose response to the forgetting bar")
    fig.tight_layout()
    fig.savefig(rd / "r_vs_dose.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, key in zip(axes, ("target", "train_B", "val_B", "generic")):
        for tag, col in ((ptag, "crimson"), (ntag, "black")):
            res = arms[tag]
            ax.plot([p["step"] for p in res["traj"]], [p[key] for p in res["traj"]],
                    "o-", ms=2.5, color=col, label=tag.split("-")[0])
        ax.set_title(key); ax.set_xlabel("step"); ax.set_ylabel("CE")
        if key == "target":
            ax.axhline(base_ce["target"] + gap, color="navy", ls="-.", lw=1)
    axes[0].legend(fontsize=7)
    fig.suptitle("E003c trajectories: all four CE readouts (projected vs naive-matched)")
    fig.tight_layout()
    fig.savefig(rd / "trajectories.png", dpi=140)
    plt.close(fig)

    save_json(rd / "metrics.json", {
        "experiment": "e003c_projected_dose_response", "seed": SEED, "steps": STEPS,
        "lr": LR, "eval_every": {"projected": EVAL_EVERY_PROJ, "naive": EVAL_EVERY_NAIVE},
        "batch": BATCH, "n_eval_blocks": N_EVAL_BLOCKS,
        "measurement": {
            "target": "TRAIN-A (corpus 0-45%) CE, 400 fixed 256-blocks",
            "collateral_1": "val_B (95-100%) CE — the forgetting bar's denominator",
            "collateral_2": "TRAIN-B (45-90%) CE — memorization-symmetric control (NEW)",
            "generic": "val (90-100%) CE",
            "forgetting_bar": "baseline val_B CE - baseline train-A CE (gap closure)",
        },
        **results,
        "config": cfg_dict(cfg),
    })
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
