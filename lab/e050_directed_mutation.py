"""E050 — Directed-mutation lineage control: REACHABILITY vs VISIBILITY.

Pre-registered in THINKING.md T024 (follow-up section) as the control that
disambiguates e040's P2 verdict ("weakly-evolvable / effectively frozen"):
P2 conflated "selection can't see the basis" (VISIBILITY-LIMITED) with
"random mutation can't reach it" (REACHABILITY-LIMITED). e050 changes ONE
thing relative to e040 — WHERE mutation acts — and re-runs the lineage.

Protocol (e040 machinery verbatim; 2 generations, smaller):
  cfg    = e005s SMALL 4L/4H/128d/256 (840,704 params, asserted).
  REF    = e040_ref.pt REUSED (frozen donor, seed 4304); its mlp-L2/L3 organs
           are THE graft for every assay. No retrain.
  W      = e040_w.pt REUSED (wildtype, seed 42 init) — gen-0 member, no retrain.
  gen-0  = W + 3 DIRECTED mutants: init42 + eps applied ONLY to the
           stream-facing MLP matrices (h.{i}.mlp.0.weight = W_in,
           h.{i}.mlp.2.weight = W_out, every block), per-element
           sigma = 0.005 = e040's EFFECTIVE sigma_mut (G7 never escalated
           there). LN weights/biases and everything else are NOT mutated.
           Draw seeds 5021/5022/5023.
  gen-1  = 3 children of gen-0's best two eligible members by R (two from the
           top parent, one from the second), parent init + fresh directed eps,
           seeds 5031/5032/5033. ONE selection event total.
  Every child trains FROM SCRATCH, step-matched: exactly 4000 steps, lr 1e-3,
  batch 32, cosine, 180s CAP. Same batch order for everyone (corpus seed 1337).

Trait / assay (identical instrument to e040, CPU-only inside cooldowns):
  D_i = mean over {L2,L3} of dCE(host_i <- REF mlp-Lsite), 30 fixed val
  batches, paired bootstrap CI;  A_is = dCE(zero own mlp-Lsite);
  R_i = mean D_is/A_is (the selected scalar). Eligibility gates: parity
  val CE <= val(W)+0.05 AND A_is in [0.5x,2.0x] A_Ws. G1 C0 self-transplant,
  G2 determinism, G5 step==4000, G6 thermal discipline, all as e040.
  dW-alignment cos(dW_member, dW_REF) per graft organ reported per generation.

REGISTERED VERDICTS (T024; response = ONE selection event, gen-1 vs gen-0):
  resp_D = (D_g0 - D_g1) / D_g0.
  REACHABILITY-LIMITED : resp_D >= 25% AND contrast CI excludes 0 AND gates
                         held -> random global noise couldn't reach the basis;
                         directed mutation can. e040's FROZEN flips.
  VISIBILITY-LIMITED   : resp_D < 10% (or CI includes 0) AND alignment shift
                         < +0.05 -> even reachable, stream-facing mutations are
                         not selected on. FROZEN holds at its strongest.
  MIDDLE               : anything else -> quantified, no binary claim.
  Degenerate routes (parity-broken / organ-devaluing) reported as-is.

LN-CONFOUND CHECK (registered in T024): per member, L2 distance of the
trained LN parameters (ln1/ln2 weight+bias) to REF's trained LNs — is graft
damage tracking host-side LN calibration rather than basis geometry?
Reported: dist over all 4 blocks (primary) and blocks {2,3} ln2 only (the
direct graft entry points); Pearson + Spearman vs D across all members.

Scope: n=1 donor (e040's REF), 3 founders, 1 selection event, sigma matched
to e040's effective 0.005. e040's own first-event response was -3.5%
(D 2.6252->2.5329) and -5.6% over two events; both quoted in metrics for the
apples-to-apples read. No NOTES/THINKING/QUEUE/STATE edits, no git commit.

Run: python lab/e050_directed_mutation.py
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sps

from common import (DEVICE, GPU_IDLE_TEMP_TARGET, REPO, Cfg, CharCorpus,
                    TinyGPT, cfg_dict, cooldown, gpu_ok, gpu_status, lesion,
                    now_iso, run_dir, save_json, set_seed, train_model)

# ---------------------------------------------------------------- protocol
CORPUS_PATH = REPO / "data" / "input.txt"
CKPT_DIR = REPO / "runs" / "checkpoints"
E040_METRICS = REPO / "runs" / "e040" / "metrics.json"

SEED_W, SEED_REF = 42, 4304
G0_MUT_SEEDS = (5021, 5022, 5023)
G1_SEEDS = (5031, 5032, 5033)
SIGMA = 0.005                    # = e040's effective sigma_mut (G7 never fired)
SITES = (2, 3)                   # graft sites (mlp-L2, mlp-L3)
STEPS, LR, BS, CAP_S = 4000, 1e-3, 32, 180.0
N_EVAL, N_BOOT = 30, 2000
PARITY_SLACK = 0.05
BAND_LO, BAND_HI = 0.5, 2.0
REF_BAND = (1.50, 1.66)
STOP_RULE_S = 800.0              # skip a child launch past this (min viable:
WALL_VIOLATION_S = 150.0         #   gen-0 + >=2 gen-1 children)
COOLDOWN_S, COOLDOWN_S_HOT, HOT_TEMP = 60.0, 90.0, 70.0

# THE ONE CHANGE: mutation touches ONLY the stream-facing MLP matrices.
DIRECTED_KEYS = tuple(f"h.{i}.mlp.{j}.weight" for i in range(4) for j in (0, 2))
DIRECTED_SET = frozenset(DIRECTED_KEYS)

T0 = time.time()
THERMAL: list[dict] = []
SELECTION: list[dict] = []
STATUS = {"parked": False, "wall_violated": False, "aborted": None}
CFG: Cfg | None = None
CORPUS: CharCorpus | None = None
EVAL_NET: TinyGPT | None = None
REF_SD: dict | None = None
REF_DW: dict | None = None
REF_RAW: dict | None = None
REF_REC: dict = {}
MEMBERS: dict[int, list[dict]] = {0: [], 1: []}


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():6.1f}s] {msg}", flush=True)


def mean(xs) -> float:
    return float(sum(xs) / len(xs))


# ---------------------------------------------------------------- surgery (e028/e040 verbatim, CPU)
def mlp_keys(i: int) -> list[str]:
    return [f"h.{i}.mlp.0.weight", f"h.{i}.mlp.0.bias",
            f"h.{i}.mlp.2.weight", f"h.{i}.mlp.2.bias"]


def snapshot(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def transplant(model, donor_sd: dict, keys: list[str]) -> None:
    sd = model.state_dict()
    for k in keys:
        assert sd[k].shape == donor_sd[k].shape, k
        sd[k] = donor_sd[k].detach().clone().to(sd[k].dtype)
    model.load_state_dict(sd, strict=True)


def assert_identical(model, snap: dict, name: str) -> None:
    now = model.state_dict()
    bad = [k for k in snap if not torch.equal(now[k], snap[k])]
    if bad:
        raise SystemExit(f"HOST {name} NOT BITWISE-RESTORED after a cell: {bad[:5]}")


@torch.no_grad()
def per_batch_losses_cpu(model, corpus, n_batches: int = N_EVAL) -> list[float]:
    model.eval()
    cfg = model.cfg
    gen = torch.Generator().manual_seed(corpus.seed)
    src = corpus.val
    losses = []
    for _ in range(n_batches):
        ix = torch.randint(len(src) - cfg.block_size - 1, (16,), generator=gen)
        x = torch.stack([src[i: i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1: i + 1 + cfg.block_size] for i in ix])
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return losses


def bootstrap_ci(diffs, seed: int, n: int = N_BOOT):
    rng = np.random.default_rng(seed)
    a = np.asarray(diffs, dtype=np.float64)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    vals = a[idx].mean(axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


def dw_vec(sd: dict, init: dict, keys: list[str]) -> torch.Tensor:
    return torch.cat([(sd[k].float() - init[k].float()).reshape(-1) for k in keys])


def raw_vec(sd: dict, keys: list[str]) -> torch.Tensor:
    return torch.cat([sd[k].float().reshape(-1) for k in keys])


# ---------------------------------------------------------------- LN confound vectors
def ln_vec(sd: dict, blocks=(0, 1, 2, 3), parts=("ln1", "ln2")) -> torch.Tensor:
    keys = [f"h.{b}.{p}.{q}" for b in blocks for p in parts
            for q in ("weight", "bias")]
    return torch.cat([sd[k].float().reshape(-1) for k in keys])


LN_REF_ALL = None      # filled in main after REF load
LN_REF_SITE = None


def ln_distances(sd: dict) -> dict:
    return {"ln_all": float((ln_vec(sd) - LN_REF_ALL).norm()),
            "ln_site_l2l3_ln2": float(
                (ln_vec(sd, blocks=(2, 3), parts=("ln2",)) - LN_REF_SITE).norm())}


# ---------------------------------------------------------------- thermal / GPU discipline (G6)
def thermal_event(event: str, phase: str) -> dict:
    s = gpu_status()
    rec = {"t_s": round(elapsed(), 1), "event": event, "phase": phase,
           "temp_c": s["temp"], "util": s["util"], "power_w": s["power"]}
    THERMAL.append(rec)
    return rec


def cooldown_with(work, phase: str) -> None:
    t0 = time.time()
    temp = thermal_event("cooldown_start", phase)["temp_c"]
    window = COOLDOWN_S_HOT if temp > HOT_TEMP else COOLDOWN_S
    log(f"[thermal] cooldown window {window:.0f}s (temp {temp:.0f}C) — CPU assay runs inside")
    if work is not None:
        work()
    rem = window - (time.time() - t0)
    if rem > 0:
        cooldown(rem)
    thermal_event("cooldown_end", phase)


def gpu_gate(phase: str) -> bool:
    if gpu_ok():
        thermal_event("launch", phase)
        return True
    log(f"[gpu_guard] HOLD at {phase} — waiting for {GPU_IDLE_TEMP_TARGET}C (max 300s)")
    t0 = time.time()
    while time.time() - t0 < 300:
        time.sleep(15)
        if gpu_status()["temp"] <= GPU_IDLE_TEMP_TARGET:
            break
    if gpu_ok():
        thermal_event("launch_retry_ok", phase)
        return True
    thermal_event("launch_retry_fail", phase)
    STATUS["parked"] = True
    return False


# ---------------------------------------------------------------- members
def make_init(seed: int) -> dict:
    set_seed(seed)
    return snapshot(TinyGPT(CFG))          # CPU construction: bitwise-stable


def directed_mutant_init(base_sd: dict, seed: int, sigma: float) -> dict:
    """THE ONE CHANGE vs e040: eps lands ONLY on the stream-facing MLP
    matrices (W_in/W_out of every block). LNs, attention, embeddings, biases:
    untouched."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in base_sd.items():
        if k in DIRECTED_SET:
            out[k] = v + torch.randn(v.shape, generator=g) * sigma
        else:
            out[k] = v.clone()
    return out


def load_member(name: str, gen: int, prov: str, ckpt: str, init_sd: dict) -> dict:
    final = CKPT_DIR / ckpt
    assert final.exists(), f"reuse ckpt missing: {final}"
    sd = torch.load(final, map_location="cpu", weights_only=True)
    return {"name": name, "gen": gen, "provenance": prov, "ckpt": final.name,
            "steps": 4000, "wall_s": None, "excluded": False, "fresh": False,
            "trained_sd": sd, "init_sd": init_sd}


def train_member(name: str, init_sd: dict, gen: int, prov: str) -> dict | None:
    final = CKPT_DIR / f"{name}.pt"
    traint = CKPT_DIR / f"{name}.train.pt"
    rec = {"name": name, "gen": gen, "provenance": prov, "ckpt": final.name,
           "steps": None, "wall_s": None, "excluded": False, "fresh": False}
    if final.exists():                      # resumable rerun skip
        rec["trained_sd"] = torch.load(final, map_location="cpu", weights_only=True)
        rec["steps"] = STEPS
        log(f"{name}: final ckpt found — skip training (resumable rerun)")
        return rec
    if not gpu_gate(name):
        return None
    model = TinyGPT(CFG)
    model.load_state_dict(init_sd)
    model.to(DEVICE)
    t0 = time.time()
    hist = train_model(model, CORPUS, steps=STEPS, lr=LR, batch_size=BS,
                       max_seconds=CAP_S, ckpt=traint)
    wall = time.time() - t0
    rec["wall_s"] = round(wall, 1)
    rec["fresh"] = True
    rec["steps"] = hist[-1]["step"] if hist else 0
    if wall > WALL_VIOLATION_S:
        STATUS["wall_violated"] = True
        log(f"{name}: WALL {wall:.0f}s > {WALL_VIOLATION_S:.0f}s — envelope violated, "
            "will finish this cohort's assay and stop")
    if rec["steps"] < STEPS:                # G5: excluded, never extended
        rec["excluded"] = True
        log(f"{name}: EXCLUDED (stopped at step {rec['steps']} < {STEPS})")
        return rec
    rec["trained_sd"] = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    torch.save(rec["trained_sd"], final)
    del model
    return rec


def assay_member(rec: dict, run_c0: bool, cohort: str) -> None:
    net = EVAL_NET
    net.load_state_dict(rec["trained_sd"])
    snap = snapshot(net)
    base1 = per_batch_losses_cpu(net, CORPUS)
    base2 = per_batch_losses_cpu(net, CORPUS)
    rec["g2_determinism"] = bool(base1 == base2)      # G2
    if not rec["g2_determinism"]:
        raise SystemExit(f"G2 GATE FAILURE: base CE not reproducible for {rec['name']}")
    base = base1
    rec["base_ce"] = mean(base)
    rec["gates"] = {}

    if run_c0:                                         # G1 (once per cohort)
        keys = mlp_keys(SITES[0])
        transplant(net, snap, keys)
        losses = per_batch_losses_cpu(net, CORPUS)
        net.load_state_dict(snap)
        assert_identical(net, snap, rec["name"])
        exact = losses == base
        rec["gates"]["c0_self_transplant_zero"] = bool(
            exact and mean(losses) - mean(base) == 0.0)
        if not rec["gates"]["c0_self_transplant_zero"]:
            raise SystemExit(f"G1 GATE FAILURE: C0 self-transplant changed CE for {rec['name']}")
        log(f"  C0 gate ({cohort}, {rec['name']}): dCE exactly 0.0, bitwise restored")

    rec["graft"], rec["ablate"] = {}, {}
    for si, site in enumerate(SITES):
        keys = mlp_keys(site)
        transplant(net, REF_SD, keys)                  # member <- REF organ
        losses = per_batch_losses_cpu(net, CORPUS)
        net.load_state_dict(snap)
        assert_identical(net, snap, rec["name"])
        diffs = [a - b for a, b in zip(losses, base)]
        lo, hi = bootstrap_ci(diffs, seed=21400 + si)
        rec["graft"][site] = {"dce": mean(diffs), "ci95": [lo, hi], "per_batch": diffs}
        with lesion(net, "mlp", site):                 # zero own organ
            abl = per_batch_losses_cpu(net, CORPUS)
        adiff = [a - b for a, b in zip(abl, base)]
        assert_identical(net, snap, rec["name"])
        rec["ablate"][site] = {"dce": mean(adiff), "per_batch": adiff}
        log(f"  {rec['name']:9s} <-REF mlp-L{site}: dCE {mean(diffs):+.4f} "
            f"(CI {lo:+.3f},{hi:+.3f}) | ablate {mean(adiff):+.4f}")

    rec["dw_cos"], rec["raw_cos"] = {}, {}
    for site in SITES:
        keys = mlp_keys(site)
        rec["dw_cos"][site] = cos(dw_vec(rec["trained_sd"], rec["init_sd"], keys),
                                  REF_DW[site])
        rec["raw_cos"][site] = cos(raw_vec(rec["trained_sd"], keys), REF_RAW[site])
    rec["D"] = mean([rec["graft"][s]["dce"] for s in SITES])
    rec["A"] = {s: rec["ablate"][s]["dce"] for s in SITES}
    rec["R_sites"] = {s: (rec["graft"][s]["dce"] / rec["ablate"][s]["dce"]
                          if abs(rec["ablate"][s]["dce"]) > 1e-9 else None)
                      for s in SITES}
    rec["R_mean"] = mean([v for v in rec["R_sites"].values() if v is not None])
    rec["align"] = mean([rec["dw_cos"][s] for s in SITES])
    rec["ln_dist"] = ln_distances(rec["trained_sd"])
    log(f"  {rec['name']:9s} D {rec['D']:.4f} | R {rec['R_mean']:.3f} | "
        f"align(dW vs REF) {rec['align']:+.4f} | base CE {rec['base_ce']:.4f} | "
        f"LN dist {rec['ln_dist']['ln_all']:.3f}")


def train_and_assay(name: str, init_sd: dict, gen: int, prov: str,
                    first: bool = False, pre_work=None) -> dict | None:
    rec = train_member(name, init_sd, gen, prov)
    if rec is None:
        return None
    rec["init_sd"] = init_sd
    if rec["excluded"]:
        cooldown_with(None, name)
        return rec

    def work():
        if pre_work is not None:
            pre_work()
            pre_work.done = True
        assay_member(rec, run_c0=first, cohort=f"gen-{gen}")

    if rec["fresh"]:
        cooldown_with(work, name)
    else:
        work()
    return rec


def assign_eligibility(recs: list[dict], w_rec: dict) -> None:
    val_w = w_rec["base_ce"]
    for r in recs:
        if r.get("excluded") or "base_ce" not in r:
            continue
        r["parity_pass"] = bool(r["base_ce"] <= val_w + PARITY_SLACK)
        r["band_pass"] = bool(all(
            BAND_LO * w_rec["A"][s] <= r["A"][s] <= BAND_HI * w_rec["A"][s]
            for s in SITES))
        r["eligible"] = bool(r["parity_pass"] and r["band_pass"])


def included(recs: list[dict]) -> list[dict]:
    return [r for r in recs if not r.get("excluded") and "D" in r]


def cohort_stats(recs: list[dict]) -> dict:
    inc = included(recs)
    if not inc:
        return {"n": 0}
    D = np.array([np.mean([r["graft"][s]["per_batch"] for s in SITES], axis=0)
                  for r in inc])
    Dmean = D.mean(axis=0)
    ci = bootstrap_ci(list(Dmean), seed=21600 + inc[0]["gen"])
    return {"n": len(inc), "D_mean": float(Dmean.mean()), "D_ci95": ci,
            "R_mean": float(np.mean([r["R_mean"] for r in inc])),
            "align_mean": float(np.mean([r["align"] for r in inc])),
            "val_mean": float(np.mean([r["base_ce"] for r in inc])),
            "members": [r["name"] for r in inc]}


def gen_contrast(recs_a: list[dict], recs_b: list[dict], seed: int) -> dict:
    """Paired (by val batch) bootstrap of mean_D(gen b) - mean_D(gen a); R too."""
    ia, ib = included(recs_a), included(recs_b)
    if not ia or not ib:
        return {}
    Da = np.array([[[r["graft"][s]["per_batch"] for s in SITES]] for r in ia]).squeeze(1)
    Aa = np.array([[[r["ablate"][s]["per_batch"] for s in SITES]] for r in ia]).squeeze(1)
    Db = np.array([[[r["graft"][s]["per_batch"] for s in SITES]] for r in ib]).squeeze(1)
    Ab = np.array([[[r["ablate"][s]["per_batch"] for s in SITES]] for r in ib]).squeeze(1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, Da.shape[2], size=(N_BOOT, Da.shape[2]))
    dD, dR = [], []
    for row in idx:
        dD.append(float(Db[:, :, row].mean() - Da[:, :, row].mean()))
        ra = np.mean(np.divide(Da[:, :, row].mean(axis=2), Aa[:, :, row].mean(axis=2)))
        rb = np.mean(np.divide(Db[:, :, row].mean(axis=2), Ab[:, :, row].mean(axis=2)))
        dR.append(float(rb - ra))
    dD, dR = np.array(dD), np.array(dR)
    return {"D_diff": float(dD.mean()),
            "D_ci95": [float(np.percentile(dD, 2.5)), float(np.percentile(dD, 97.5))],
            "D_ci_excludes_0": bool(np.percentile(dD, 2.5) > 0
                                    or np.percentile(dD, 97.5) < 0),
            "R_diff": float(dR.mean()),
            "R_ci95": [float(np.percentile(dR, 2.5)), float(np.percentile(dR, 97.5))],
            "R_ci_excludes_0": bool(np.percentile(dR, 2.5) > 0
                                    or np.percentile(dR, 97.5) < 0),
            "n_boot_valid": int(N_BOOT)}


# ---------------------------------------------------------------- figure
def figures(rd, stats: dict) -> None:
    gens = [g for g in sorted(stats) if stats[g].get("n")]
    if not gens:
        return
    w = next((r for r in MEMBERS[0] if "base_ce" in r), None)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for gi, g in enumerate(gens):
        recs = included(MEMBERS[g])
        axes[0].scatter([gi] * len(recs), [r["D"] for r in recs], color="#444",
                        s=18, zorder=3)
        axes[1].scatter([gi] * len(recs), [r["R_mean"] for r in recs], color="#444",
                        s=18, zorder=3)
        axes[2].scatter([gi] * len(recs), [r["align"] for r in recs], color="#2c6fbb",
                        s=18, zorder=3)
        axes[3].scatter([gi] * len(recs), [r["base_ce"] for r in recs], color="#444",
                        s=18, zorder=3)
        lo, hi = stats[g]["D_ci95"]
        axes[0].errorbar(gi, stats[g]["D_mean"],
                         yerr=[[stats[g]["D_mean"] - lo], [hi - stats[g]["D_mean"]]],
                         color="#c0392b", capsize=5, lw=2, zorder=4)
        axes[1].plot(gi, stats[g]["R_mean"], "o", color="#c0392b", ms=9, zorder=4)
        axes[2].plot(gi, stats[g]["align_mean"], "o", color="#c0392b", ms=9, zorder=4)
        axes[3].plot(gi, stats[g]["val_mean"], "o", color="#c0392b", ms=9, zorder=4)
    axes[0].set_ylabel("D = graft damage (member<-REF mlp L2/L3)")
    axes[0].set_title("Graft damage (red = cohort mean, CI95)")
    axes[1].set_ylabel("R = D / own-ablation")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_title("Selection index R")
    axes[2].set_ylabel("cos(dW_member, dW_REF), mean of sites")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].set_title("dW-alignment to donor")
    axes[3].set_ylabel("val CE (30 fixed batches)")
    axes[3].set_title("Parity panel")
    if w is not None:
        axes[3].axhline(w["base_ce"] + PARITY_SLACK, color="k", ls="--", lw=1,
                        label=f"parity = val(W)+{PARITY_SLACK}")
        axes[3].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(len(gens)), [f"gen-{g}" for g in gens])
        ax.set_xlabel("generation")
    fig.suptitle("E050 directed-mutation lineage — mutation touches ONLY MLP W_in/W_out "
                 f"(sigma {SIGMA}); REF/W reused from e040; one selection event")
    fig.tight_layout()
    fig.savefig(rd / "lineage_trait.png", dpi=140)
    plt.close(fig)

    # LN-confound panel: per-member D vs LN distance to REF
    recs = [r for g in MEMBERS for r in MEMBERS[g] if "D" in r]
    if len(recs) >= 3:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        xs = [r["ln_dist"]["ln_all"] for r in recs]
        ys = [r["D"] for r in recs]
        for r, x, y in zip(recs, xs, ys):
            ax.scatter(x, y, s=30,
                       color="#c0392b" if r["gen"] == 0 else "#2c6fbb")
            ax.annotate(r["name"].replace("e050_", ""), (x, y), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        r_p = float(np.corrcoef(xs, ys)[0, 1])
        b = np.polyfit(xs, ys, 1)
        xr = np.linspace(min(xs), max(xs), 50)
        ax.plot(xr, np.polyval(b, xr), "k--", lw=1)
        ax.set_xlabel("LN distance to REF: ||ln1/ln2 w+b (all blocks) member - REF||2")
        ax.set_ylabel("D = graft damage (L2/L3 mean)")
        ax.set_title(f"LN-confound check: Pearson r = {r_p:+.3f} (n={len(recs)}); "
                     "red = gen-0, blue = gen-1")
        fig.tight_layout()
        fig.savefig(rd / "ln_confound.png", dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------- outputs / verdicts
def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("init_sd", "trained_sd")}


def ln_confound_block() -> dict:
    recs = [r for g in MEMBERS for r in MEMBERS[g] if "D" in r]
    out = {"n_members": len(recs),
           "members": [r["name"] for r in recs],
           "note": "trained-LN (ln1/ln2 w+b) L2 distance to trained REF vs graft damage D"}
    if len(recs) >= 3:
        for key, label in (("ln_all", "all blocks ln1+ln2 w+b"),
                           ("ln_site_l2l3_ln2", "blocks 2,3 ln2 w+b (graft entry)")):
            xs = np.array([r["ln_dist"][key] for r in recs])
            ys = np.array([r["D"] for r in recs])
            pr, pp = sps.pearsonr(xs, ys)
            sr, sp = sps.spearmanr(xs, ys)
            out[label] = {"pearson_r": float(pr), "pearson_p": float(pp),
                          "spearman_rho": float(sr), "spearman_p": float(sp)}
        xs = np.array([r["align"] for r in recs])
        ys = np.array([r["D"] for r in recs])
        pr, pp = sps.pearsonr(xs, ys)
        out["dw_align_vs_D"] = {"pearson_r": float(pr), "pearson_p": float(pp)}
    return out


def e040_comparison() -> dict:
    try:
        m = json.loads(E040_METRICS.read_text(encoding="utf-8"))
        gs = m["generation_stats"]
        vc = m["verdicts"]["conditions"]
        return {
            "source": "runs/e040/metrics.json",
            "D_by_gen": {g: gs[g]["D_mean"] for g in ("0", "1", "2")},
            "R_by_gen": {g: gs[g]["R_mean"] for g in ("0", "1", "2")},
            "align_by_gen": {g: gs[g]["align_mean"] for g in ("0", "1", "2")},
            "first_event_resp_D": 1 - gs["1"]["D_mean"] / gs["0"]["D_mean"],
            "two_event_resp_D": 1 - gs["2"]["D_mean"] / gs["0"]["D_mean"],
            "e040_verdict": m["verdicts"]["verdict"],
            "e040_D_drop_frac": vc["D_drop_frac"],
        }
    except Exception as e:  # comparison is context, never fatal
        return {"error": str(e)}


def finish(rd, partial: bool):
    stats = {g: cohort_stats(recs) for g, recs in MEMBERS.items()}
    contrast = gen_contrast(MEMBERS[0], MEMBERS[1], seed=21800)
    lnc = ln_confound_block()

    # ---------------- verdicts (T024 registered bars, ONE selection event) ----
    s0, s1 = stats.get(0, {}), stats.get(1, {})
    v = {"partial": partial, "conditions": {}}
    w_rec = next((r for r in MEMBERS[0] if r["name"] == "e050_w" and "base_ce" in r), None)
    if s0.get("n") and s1.get("n") and w_rec is not None:
        D0, D1, R0, R1 = s0["D_mean"], s1["D_mean"], s0["R_mean"], s1["R_mean"]
        a0, a1 = s0["align_mean"], s1["align_mean"]
        inc0m, inc1m = included(MEMBERS[0]), included(MEMBERS[1])
        val_w = w_rec["base_ce"]
        gates_g1 = all(r["eligible"] for r in inc1m)
        gates_g0 = all(r["eligible"] for r in inc0m)
        resp_D = (D0 - D1) / D0
        resp_R = (R0 - R1) / R0
        align_shift = a1 - a0
        ci_excl = contrast.get("D_ci_excludes_0")
        c = v["conditions"]
        c.update({
            "D_gen0": D0, "D_gen1": D1, "D_ratio": D1 / D0,
            "R_gen0": R0, "R_gen1": R1, "R_ratio": R1 / R0,
            "align_gen0": a0, "align_gen1": a1, "align_shift": align_shift,
            "resp_D": resp_D, "resp_R": resp_R,
            "contrast_ci_excludes_0": ci_excl,
            "gates_gen0_all_eligible": gates_g0,
            "gates_gen1_all_eligible": gates_g1,
        })
        reachability = bool(resp_D >= 0.25 and ci_excl and gates_g1 and gates_g0)
        visibility = bool((resp_D < 0.10 or not ci_excl) and align_shift < 0.05)
        band_fail_g1 = [r["name"] for r in inc1m if not r["band_pass"]]
        parity_broken = bool(resp_D >= 0.25 and
                             mean([r["base_ce"] for r in inc1m]) > val_w + PARITY_SLACK)
        organ_devaluing = bool(resp_D >= 0.25 and band_fail_g1
                               and not (R1 <= 0.75 * R0))
        all_recs = [r for g in MEMBERS for r in MEMBERS[g]]
        instrument_fail = bool(
            not all(r.get("g2_determinism", True) for r in all_recs if "D" in r)
            or not all(r.get("gates", {}).get("c0_self_transplant_zero", True)
                       for r in all_recs if r.get("gates")))
        v.update({"reachability_limited": reachability,
                  "visibility_limited": visibility,
                  "middle": bool(not reachability and not visibility),
                  "parity_broken_route": parity_broken,
                  "organ_devaluing_route": organ_devaluing,
                  "band_fail_gen1": band_fail_g1,
                  "instrument_failure": instrument_fail})
        if instrument_fail:
            v["verdict"] = "INSTRUMENT-FAILURE (G1/G2) — nothing interpreted"
        elif parity_broken or organ_devaluing:
            v["verdict"] = ("GATE-FAILURE ROUTE ("
                            + ("parity-broken" if parity_broken else "organ-devaluing")
                            + ") — reported as-is, distinct from the registered verdicts")
        elif reachability:
            v["verdict"] = "REACHABILITY-LIMITED (directed mutation flips FROZEN: resp >= 25%)"
        elif visibility:
            v["verdict"] = "VISIBILITY-LIMITED (FROZEN at its strongest: resp < 10%)"
        else:
            v["verdict"] = (f"MIDDLE (resp_D {resp_D * 100:+.1f}% — between the 10% and "
                            "25% bars; quantified, no binary flip)")
        log("=" * 78)
        log(f"VERDICT: {v['verdict']}")
        log(f"  D {D0:.4f} -> {D1:.4f} ({resp_D * 100:+.1f}% after ONE selection event; "
            f"e040 random-mutation event-1 was -3.5%, two events -5.6%)")
        log(f"  R {R0:.3f} -> {R1:.3f} ({resp_R * 100:+.1f}%) | align {a0:+.4f} -> "
            f"{a1:+.4f} (shift {align_shift:+.4f}) | CI excl 0: {ci_excl} | "
            f"gates g0/g1: {gates_g0}/{gates_g1}")
        if "all blocks ln1+ln2 w+b" in lnc:
            b = lnc["all blocks ln1+ln2 w+b"]
            log(f"  LN-confound: Pearson r(D, LN-dist) {b['pearson_r']:+.3f} "
                f"(p {b['pearson_p']:.3f}), Spearman rho {b['spearman_rho']:+.3f} "
                f"(p {b['spearman_p']:.3f}), n={lnc['n_members']}")
    else:
        v["verdict"] = "PARTIAL (incomplete cohort — stop-rule); directional read only"
        log("VERDICT: partial — incomplete cohort")

    figures(rd, stats)
    ref_block = {"reused_from": "e040_ref.pt (seed 4304; frozen donor)",
                 "val_ce": REF_REC.get("base_ce"),
                 "g3_band": list(REF_BAND),
                 "g3_pass": bool(REF_BAND[0] <= REF_REC.get("base_ce", 0)
                                 <= REF_BAND[1])} if REF_REC.get("base_ce") else \
        {"reused_from": "e040_ref.pt", "val_ce": None}
    n_directed = sum(v.numel() for k, v in
                     torch.load(CKPT_DIR / "e040_w.pt", map_location="cpu",
                                weights_only=True).items() if k in DIRECTED_SET)
    n_weight_all = sum(v.numel() for k, v in
                       torch.load(CKPT_DIR / "e040_w.pt", map_location="cpu",
                                  weights_only=True).items() if k.endswith(".weight"))
    out = {
        "experiment": "e050_directed_mutation",
        "date": now_iso(),
        "smoke": False,
        "status": "parked_for_heartbeat" if STATUS["parked"] else
                  ("aborted_g5" if STATUS["aborted"] else "complete"),
        "partial": bool(partial or STATUS["aborted"]),
        "device": {"train": DEVICE, "assays": "cpu (GPU untouched during assays)"},
        "config": cfg_dict(CFG), "params": 840_704,
        "protocol": {
            "steps": STEPS, "lr": LR, "batch_size": BS, "cap_s": CAP_S,
            "sigma_directed": SIGMA,
            "sigma_note": ("per-element sigma matched to e040's EFFECTIVE sigma_mut "
                           "(0.005; G7 never escalated)"),
            "mutation_targets": list(DIRECTED_KEYS),
            "mutation_target_note": ("THE ONE CHANGE vs e040: eps lands only on MLP "
                                     "W_in/W_out (stream-facing matrices); LN w/b, "
                                     "attention, embeddings, biases untouched"),
            "directed_param_count": int(n_directed),
            "all_weight_param_count": int(n_weight_all),
            "energy_fraction_vs_e040_total_eps": float(n_directed / n_weight_all),
            "sites": list(SITES), "n_eval_batches": N_EVAL, "n_boot": N_BOOT,
            "parity_slack": PARITY_SLACK, "band": [BAND_LO, BAND_HI],
            "seed_w": SEED_W, "seed_ref": SEED_REF,
            "reuse": {"ref": "e040_ref.pt", "wildtype": "e040_w.pt"},
            "g0_mut_seeds": list(G0_MUT_SEEDS), "g1_seeds": list(G1_SEEDS),
            "generations": 2, "selection_events": 1,
        },
        "ref": ref_block,
        "gates": {
            "g5_excluded": [r["name"] for g in MEMBERS for r in MEMBERS[g]
                            if r.get("excluded")],
            "g6_thermal_log": THERMAL},
        "members": {str(g): [strip(r) for r in MEMBERS[g]] for g in MEMBERS},
        "generation_stats": {str(g): stats[g] for g in stats},
        "gen1_vs_gen0_contrast": contrast,
        "selection_events": SELECTION,
        "ln_confound": lnc,
        "e040_comparison": e040_comparison(),
        "verdicts": v,
        "timing_s": {"total": round(elapsed(), 1)},
    }
    save_json(rd / "metrics.json", out)
    log(f"outputs: {rd} (metrics.json, lineage_trait.png, ln_confound.png)")
    return out


def park(rd):
    STATUS["parked"] = True
    log("PARKED for the heartbeat (GPU gate failed twice; ckpts make rerun resumable)")
    return finish(rd, partial=True)


def abort(rd, reason: str):
    STATUS["aborted"] = reason
    log(f"ABORT: {reason}")
    return finish(rd, partial=True)


# ---------------------------------------------------------------- main
def main():
    global CFG, CORPUS, EVAL_NET, REF_SD, REF_DW, REF_RAW, LN_REF_ALL, LN_REF_SITE
    rd = run_dir("e050")
    log("E050 directed-mutation lineage control — REACHABILITY vs VISIBILITY "
        "(T024 pre-registered); mutation touches ONLY MLP W_in/W_out, sigma 0.005")
    CORPUS = CharCorpus(CORPUS_PATH, seed=1337)
    CFG = Cfg(vocab=CORPUS.vocab_size, n_layer=4, n_head=4, n_embd=128, block_size=256)
    EVAL_NET = TinyGPT(CFG)
    n_params = EVAL_NET.num_params()
    assert n_params == 840_704 and n_params <= 1_000_000, n_params
    log(f"cfg e005s SMALL verbatim: 4L/4H/128d — {n_params} params <= 1M OK")

    # ---- REF + W: reuse e040 finals (no retrain) ---------------------------
    ref_init = make_init(SEED_REF)
    REF_SD = torch.load(CKPT_DIR / "e040_ref.pt", map_location="cpu", weights_only=True)
    REF_DW = {s: dw_vec(REF_SD, ref_init, mlp_keys(s)) for s in SITES}
    REF_RAW = {s: raw_vec(REF_SD, mlp_keys(s)) for s in SITES}
    LN_REF_ALL = ln_vec(REF_SD)
    LN_REF_SITE = ln_vec(REF_SD, blocks=(2, 3), parts=("ln2",))

    w_init = make_init(SEED_W)

    def ref_assay():
        net = EVAL_NET
        net.load_state_dict(REF_SD)
        base = per_batch_losses_cpu(net, CORPUS)
        REF_REC["base_ce"] = mean(base)
        g3 = REF_BAND[0] <= REF_REC["base_ce"] <= REF_BAND[1]
        log(f"  REF (reused) base CE {REF_REC['base_ce']:.4f} (G3 band {REF_BAND}: "
            f"{'PASS' if g3 else 'FLAG'})")

    # ---- gen-0: W (reuse) + 3 directed mutants -----------------------------
    w_rec = load_member("e050_w", 0, "wildtype REUSED from e040_w.pt (set_seed(42) init)",
                        "e040_w.pt", w_init)
    gen0_specs = [(f"e050_m{i}", s) for i, s in enumerate(G0_MUT_SEEDS, start=1)]

    def w_assay():
        assay_member(w_rec, run_c0=True, cohort="gen-0")   # G1 on the wildtype

    # Train m1 first; REF + W + m1 assays run inside its cooldown window.
    for i, (name, seed) in enumerate(gen0_specs):
        init_sd = directed_mutant_init(w_init, seed, SIGMA)
        prov = (f"init42 + DIRECTED eps(seed {seed}, sigma {SIGMA}, MLP W_in/W_out only)")
        pre = None
        if i == 0:
            def pre():          # noqa: F811 — intentional single use
                ref_assay()
                w_assay()
        rec = train_and_assay(name, init_sd, 0, prov, first=False, pre_work=pre)
        if rec is None:
            MEMBERS[0].append(w_rec)                        # keep what we have
            return park(rd)
        if i == 0 and "base_ce" not in w_rec:               # m1 loaded from ckpt
            ref_assay()
            w_assay()
        if i == 0:
            MEMBERS[0].append(w_rec)
        MEMBERS[0].append(rec)
    if "base_ce" not in w_rec:
        return abort(rd, "wildtype assay never ran")
    if sum(1 for r in MEMBERS[0] if r.get("excluded")) > 1:
        return abort(rd, "gen-0: >1 member excluded (G5)")
    if STATUS["wall_violated"]:
        return finish(rd, partial=True)

    assign_eligibility(MEMBERS[0], w_rec)
    stats0 = cohort_stats(MEMBERS[0])
    log(f"gen-0: D mean {stats0['D_mean']:.4f} (CI {stats0['D_ci95'][0]:+.3f},"
        f"{stats0['D_ci95'][1]:+.3f}) R mean {stats0['R_mean']:.3f} "
        f"align {stats0['align_mean']:+.4f} val(W) {w_rec['base_ce']:.4f}")
    muts = [r for r in MEMBERS[0] if r["name"].startswith("e050_m") and "D" in r]
    if muts:
        spread = max(r["D"] for r in muts) - min(r["D"] for r in muts)
        hw = mean([(r["graft"][s]["ci95"][1] - r["graft"][s]["ci95"][0]) / 2
                   for r in muts for s in SITES])
        log(f"variance report (no escalation ladder in e050; sigma registered as "
            f"matched to e040): mutant spread {spread:.4f} vs 3x CI half-width "
            f"{3 * hw:.4f}")
        SELECTION.append({"gen": 0, "variance_report": {
            "spread": spread, "ci_half_width": hw, "threshold_3x_hw": 3 * hw,
            "escalated": False,
            "note": "e050 has no registered escalation; sigma stays 0.005"}})

    # ---- selection: best two eligible by R ----------------------------------
    ranked = sorted(included(MEMBERS[0]), key=lambda r: r["R_mean"])
    elig = [r for r in ranked if r["eligible"]]
    pool = elig if len(elig) >= 2 else ranked
    p1 = pool[0]
    p2 = pool[1] if len(pool) > 1 else None
    bottleneck = len(elig) < 2
    SELECTION.append({"gen": 1, "ranking": [(r["name"], round(r["R_mean"], 4),
                                             bool(r["eligible"])) for r in ranked],
                      "parent1": p1["name"],
                      "parent2": p2["name"] if p2 else None,
                      "bottleneck": bottleneck, "sigma": SIGMA})
    log(f"selection gen-0->1: parents {p1['name']} + "
        f"{p2['name'] if p2 else '(none)'} (bottleneck={bottleneck}); sigma {SIGMA}")

    # ---- gen-1: 3 children (2 from top parent, 1 from second) ---------------
    p2name = p2["name"] if p2 else p1["name"]
    specs = [("e050_g1a", p1, G1_SEEDS[0]),
             ("e050_g1b", p1, G1_SEEDS[1]),
             ("e050_g1c", (p2 or p1), G1_SEEDS[2])]
    for i, (name, parent, seed) in enumerate(specs):
        if elapsed() > STOP_RULE_S:
            log(f"stop-rule: skipping {name} (elapsed {elapsed():.0f}s "
                f"> {STOP_RULE_S:.0f}s)")
            SELECTION.append({"gen": 1, "skipped": name,
                              "reason": f"elapsed>{STOP_RULE_S:.0f}s"})
            continue
        init_sd = directed_mutant_init(parent["init_sd"], seed, SIGMA)
        prov = (f"child of {parent['name']} (DIRECTED eps seed {seed}, sigma "
                f"{SIGMA}, MLP W_in/W_out only)")
        rec = train_and_assay(name, init_sd, 1, prov, first=(i == 0))
        if rec is None:
            return park(rd)
        MEMBERS[1].append(rec)
    if sum(1 for r in MEMBERS[1] if r.get("excluded")) > 1:
        return abort(rd, "gen-1: >1 member excluded (G5)")
    assign_eligibility(MEMBERS[1], w_rec)
    stats1 = cohort_stats(MEMBERS[1])
    if stats1.get("n"):
        log(f"gen-1: D mean {stats1['D_mean']:.4f} R mean {stats1['R_mean']:.3f} "
            f"align {stats1['align_mean']:+.4f}")
    return finish(rd, partial=bool(STATUS["wall_violated"] or not stats1.get("n")))


if __name__ == "__main__":
    main()
