"""E040 — Graft-evolution lineage: is init-anchoring EVOLVABLE?

First EVOLUTION-thread experiment. Implements scratch/e040_design.md exactly
(registered in THINKING.md T022). Genotype = the init; phenotype = the trained
net; selection = graft compatibility against a fixed REF donor.

Lineage (all 0.84M = e005s SMALL cfg 4L/4H/128d/256, params asserted 840,704):
  REF  = fresh init set_seed(4304), trained, FROZEN; its mlp-L2/L3 organs are
         THE standard graft for every assay in every generation.
  gen-0 = W (exact set_seed(42) init) + M1..M3 (init42 + eps, sigma_mut=0.005
         = 25% of the 0.02 init std, draw seeds 4021/4022/4023).
  gen-1 = 3 children of the two best eligible gen-0 members by R
         (two from top, one from second; seeds 4031/4032/4033).
  gen-2 = same rule from gen-1 (seeds 4041/4042/4043).
Every member trains FROM SCRATCH, step-matched: exactly 4000 steps, lr 1e-3,
batch 32, cosine, 180s CAP (never extended; >1 exclusion per cohort = abort).

Trait (all assays CPU-only, inside cooldown windows):
  D_i  = mean over sites {2,3} of dCE(host_i <- REF mlp-Lsite), 30 fixed
         val batches, paired bootstrap CI (n=2000).
  A_is = ablate ref: dCE(host_i, zero own mlp-Lsite), same batches.
  R_i  = mean over sites of D_is / A_is   <- the selected scalar.
  Eligibility: parity val CE_i <= val(W)+0.05 AND A_is in [0.5x, 2.0x] A_Ws.
  dW-alignment: cos(dW_member, dW_REF) per graft organ (dW = trained - OWN
  init; raw cos(W_member, W_REF) reported secondary).
  Reverse-graft control (REF <- winner organs) on the gen-2 winner only.

REGISTERED VERDICTS (T022, frozen):
  P1 EVOLVABLE          mean D_g2 <= 0.75*mean D_g0 AND mean R_g2 <= 0.75*
                        mean R_g0 AND paired-bootstrap CI of the g2-vs-g0
                        difference excludes 0 AND every contributing member
                        inside both gates.
  P2 FROZEN             |D_g2 - D_g0| < 10% of D_g0 or CI includes 0, AND
                        alignment shift < +0.05.
  P3 DEGENERATION-ROUTE alignment rises >= +0.05 g0->g2 while P1 fails.
  Gate-failure routes: parity-broken / organ-devaluing / instrument-failure
  (reported as-is, distinct from P1-P3).

Gates: G0 val(W_b32) within 0.03 of e005s_small 1.5581 | G1 C0 self-transplant
dCE exactly 0.0 per cohort | G2 base-CE determinism double-run | G3 REF val in
[1.50, 1.66] | G4 organ band | G5 step==4000 | G6 gpu_ok before every launch,
serial-only, cooldowns logged with temp | G7 gen-0 variance gate (spread of D
over M1..M3 < 3x mean CI half-width -> sigma x2 = 0.010 from gen-1, logged).

Stop-rules: skip a child launch if elapsed > 1140s (min viable = REF + gen-0 +
>=2 gen-1 children); a single training wall > 150s = envelope violated ->
finish the current cohort's assay and stop; gpu_ok() fail -> wait for 65C,
retry once, else park for the heartbeat (ckpts e040_*.train.pt make every
member resumable; finals e040_*.pt skip retraining on rerun).

Deviations / slack choices (lab discipline, none change the registered bars):
- Mutation eps is drawn for every state_dict key ending in ".weight" (the
  memo's "ALL weights"; includes LN weights whose init is 1.0 — a 0.5%
  relative nudge, noted). Biases (init 0) are not mutated.
- Inits are constructed on CPU (set_seed -> CPU TinyGPT) so every genotype
  snapshot is bitwise-stable across runs; training moves the model to GPU.
- Cohort means and the P1/P2/P3 bars are computed over INCLUDED members
  (step==4000); gen-0 baseline = W + included mutants.
- The P1 CI is computed on the D difference (primary instrument, batch-paired
  resampling shared across generations); the R difference CI is reported
  alongside with invalid resamples dropped.
- G0/G3 failures are flagged prominently, not fatal (sanity gates; the abort
  conditions in the memo are G5 exclusions and instrument gates G1/G2).
- The 150s wall rule is measured around train_model (includes ckpt resume).
- No NOTES/THINKING/QUEUE/STATE edits, no git commit (per instructions).

Run: python lab/e040_graft_evolution.py
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (DEVICE, GPU_IDLE_TEMP_TARGET, REPO, Cfg, CharCorpus,
                    TinyGPT, cfg_dict, cooldown, gpu_ok, gpu_status, lesion,
                    now_iso, run_dir, save_json, set_seed, train_model)

# ---------------------------------------------------------------- protocol
CORPUS_PATH = REPO / "data" / "input.txt"
CKPT_DIR = REPO / "runs" / "checkpoints"

SEED_W, SEED_REF = 42, 4304
G0_MUT_SEEDS = (4021, 4022, 4023)
MUT_SEEDS = {1: (4031, 4032, 4033), 2: (4041, 4042, 4043)}
SIGMA_MUT = 0.005           # 25% of the 0.02 init std
SIGMA_ESCALATED = 0.010     # G7 ladder x2
SITES = (2, 3)              # graft sites (mlp-L2, mlp-L3)
STEPS, LR, BS, CAP_S = 4000, 1e-3, 32, 180.0
N_EVAL, N_BOOT = 30, 2000
PARITY_SLACK = 0.05
BAND_LO, BAND_HI = 0.5, 2.0
E005S_ANCHOR, G0_TOL = 1.5581, 0.03
REF_BAND = (1.50, 1.66)
STOP_RULE_S = 1140.0
WALL_VIOLATION_S = 150.0
COOLDOWN_S, COOLDOWN_S_HOT, HOT_TEMP = 60.0, 90.0, 70.0

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
REF_REC: dict | None = None
MEMBERS: dict[int, list[dict]] = {0: [], 1: [], 2: []}
SIGMA_BY_GEN = {1: SIGMA_MUT, 2: SIGMA_MUT}


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():6.1f}s] {msg}", flush=True)


def mean(xs) -> float:
    return float(sum(xs) / len(xs))


# ---------------------------------------------------------------- surgery (e028 verbatim, CPU)
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
    """e028/e029 fixed 30-batch protocol, CPU-only (assays never touch GPU)."""
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


# ---------------------------------------------------------------- thermal / GPU discipline (G6)
def thermal_event(event: str, phase: str) -> dict:
    s = gpu_status()
    rec = {"t_s": round(elapsed(), 1), "event": event, "phase": phase,
           "temp_c": s["temp"], "util": s["util"], "power_w": s["power"]}
    THERMAL.append(rec)
    return rec


def cooldown_with(work, phase: str) -> None:
    """Cooldown window (>=60s; 90s if hot) with the CPU assay run INSIDE it."""
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
    """G6: gpu_ok before every launch; on fail wait for idle temp, retry once."""
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


def mutated_init(base_sd: dict, seed: int, sigma: float) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in base_sd.items():
        if k.endswith(".weight"):
            out[k] = v + torch.randn(v.shape, generator=g) * sigma
        else:
            out[k] = v.clone()
    return out


def train_member(name: str, init_sd: dict, gen: int, prov: str) -> dict | None:
    """Train one lineage member (GPU, guarded, resumable). None = parked."""
    final = CKPT_DIR / f"{name}.pt"
    traint = CKPT_DIR / f"{name}.train.pt"
    rec = {"name": name, "gen": gen, "provenance": prov, "ckpt": final.name,
           "steps": None, "wall_s": None, "excluded": False, "fresh": False}
    if final.exists():                      # rerun/resume skip
        rec["trained_sd"] = torch.load(final, map_location="cpu", weights_only=True)
        rec["steps"] = 4000
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
    """Full CPU assay of one member (runs inside its cooldown window)."""
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
        transplant(net, snap, keys)                    # own tissue through surgery
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
        lo, hi = bootstrap_ci(diffs, seed=20400 + si)
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
    log(f"  {rec['name']:9s} D {rec['D']:.4f} | R {rec['R_mean']:.3f} | "
        f"align(dW vs REF) {rec['align']:+.4f} | base CE {rec['base_ce']:.4f}")


def train_and_assay(name: str, init_sd: dict, gen: int, prov: str,
                    first: bool = False) -> dict | None:
    rec = train_member(name, init_sd, gen, prov)
    if rec is None:
        return None
    rec["init_sd"] = init_sd
    if rec["excluded"]:
        cooldown_with(None, name)         # thermal discipline still holds
        return rec

    def work():
        assay_member(rec, run_c0=first, cohort=f"gen-{gen}")
    if rec["fresh"]:
        cooldown_with(work, name)         # assay inside the cooldown window
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


def select_parents(recs: list[dict]):
    """Rank by R ascending over eligible; bottleneck if <2 eligible."""
    ranked = sorted(included(recs), key=lambda r: r["R_mean"])
    elig = [r for r in ranked if r["eligible"]]
    if len(elig) >= 2:
        return elig[0], elig[1], False, ranked
    pool = elig if elig else ranked
    return pool[0], None, True, ranked


def cohort_stats(recs: list[dict]) -> dict:
    inc = included(recs)
    if not inc:
        return {"n": 0}
    D = np.array([np.mean([r["graft"][s]["per_batch"] for s in SITES], axis=0)
                  for r in inc])                       # (n_members, 30)
    Dmean = D.mean(axis=0)
    ci = bootstrap_ci(list(Dmean), seed=20600 + inc[0]["gen"])
    return {"n": len(inc), "D_mean": float(Dmean.mean()), "D_ci95": ci,
            "R_mean": float(np.mean([r["R_mean"] for r in inc])),
            "align_mean": float(np.mean([r["align"] for r in inc])),
            "val_mean": float(np.mean([r["base_ce"] for r in inc])),
            "members": [r["name"] for r in inc]}


def _site_arrays(recs: list[dict]):
    """Per-member per-site per-batch arrays for graft and ablate dCE."""
    Dsm = np.array([[[r["graft"][s]["per_batch"] for s in SITES]] for r in recs]
                   ).squeeze(1)                       # (n, 2, 30)
    Asm = np.array([[[r["ablate"][s]["per_batch"] for s in SITES]] for r in recs]
                   ).squeeze(1)
    return Dsm, Asm


def gen_contrast(recs_a: list[dict], recs_b: list[dict], seed: int) -> dict:
    """Paired (by val batch) bootstrap of mean_D(g b) - mean_D(g a); same for R."""
    ia, ib = included(recs_a), included(recs_b)
    if not ia or not ib:
        return {}
    Da, Aa = _site_arrays(ia)
    Db, Ab = _site_arrays(ib)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, Da.shape[2], size=(N_BOOT, Da.shape[2]))
    dD, dR, n_valid = [], [], 0
    for row in idx:
        dD.append(float(Db[:, :, row].mean() - Da[:, :, row].mean()))
        ra = np.mean(np.divide(Da[:, :, row].mean(axis=2), Aa[:, :, row].mean(axis=2)))
        rb = np.mean(np.divide(Db[:, :, row].mean(axis=2), Ab[:, :, row].mean(axis=2)))
        dR.append(float(rb - ra))
        n_valid += 1
    dD, dR = np.array(dD), np.array(dR)
    return {"D_diff": float(dD.mean()),
            "D_ci95": [float(np.percentile(dD, 2.5)), float(np.percentile(dD, 97.5))],
            "D_ci_excludes_0": bool(np.percentile(dD, 2.5) > 0
                                    or np.percentile(dD, 97.5) < 0),
            "R_diff": float(dR.mean()),
            "R_ci95": [float(np.percentile(dR, 2.5)), float(np.percentile(dR, 97.5))],
            "R_ci_excludes_0": bool(np.percentile(dR, 2.5) > 0
                                    or np.percentile(dR, 97.5) < 0),
            "n_boot_valid": n_valid}


# ---------------------------------------------------------------- figures
def figures(rd, stats: dict) -> None:
    gens = [g for g in sorted(stats) if stats[g].get("n")]
    if not gens:
        return
    w = next((r for r in MEMBERS[0] if "base_ce" in r), None)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for gi, g in enumerate(gens):
        recs = included(MEMBERS[g])
        axes[0].scatter([gi] * len(recs), [r["D"] for r in recs], color="#444",
                        s=18, zorder=3)
        axes[1].scatter([gi] * len(recs), [r["R_mean"] for r in recs], color="#444",
                        s=18, zorder=3)
        axes[2].scatter([gi] * len(recs), [r["base_ce"] for r in recs], color="#444",
                        s=18, zorder=3)
        lo, hi = stats[g]["D_ci95"]
        axes[0].errorbar(gi, stats[g]["D_mean"],
                         yerr=[[stats[g]["D_mean"] - lo], [hi - stats[g]["D_mean"]]],
                         color="#c0392b", capsize=5, lw=2, zorder=4)
        axes[1].plot(gi, stats[g]["R_mean"], "o", color="#c0392b", ms=9, zorder=4)
        axes[2].plot(gi, stats[g]["val_mean"], "o", color="#c0392b", ms=9, zorder=4)
    axes[0].set_ylabel("D = graft damage (nats; member<-REF mlp L2/L3)")
    axes[0].set_title("Graft damage by generation (red = cohort mean, CI95 bar)")
    axes[1].set_ylabel("R = D / own-ablation")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_title("Selection index R by generation (dashed = ablation-equivalent)")
    axes[2].set_ylabel("val CE (30 fixed batches)")
    axes[2].set_title("Parity panel (host quality)")
    if w is not None:
        axes[2].axhline(w["base_ce"] + PARITY_SLACK, color="k", ls="--", lw=1,
                        label=f"parity gate = val(W)+{PARITY_SLACK}")
        axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(len(gens)), [f"gen-{g}" for g in gens])
        ax.set_xlabel("generation")
    fig.suptitle("E040 graft-evolution lineage — damage D, selected scalar R, parity "
                 f"(points = members; n per gen: "
                 + ", ".join(f"g{g}:{stats[g]['n']}" for g in gens) + ")")
    fig.tight_layout()
    fig.savefig(rd / "lineage_trait.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for gi, g in enumerate(gens):
        recs = included(MEMBERS[g])
        for site, ax in zip(SITES, axes[:2]):
            ax.scatter([gi] * len(recs), [r["dw_cos"][site] for r in recs],
                       color="#2c6fbb", s=18, zorder=3)
            ax.plot(gi, float(np.mean([r["dw_cos"][site] for r in recs])), "o",
                    color="#c0392b", ms=9, zorder=4)
    for site, ax in zip(SITES, axes[:2]):
        ax.set_xticks(range(len(gens)), [f"gen-{g}" for g in gens])
        ax.set_ylabel(f"cos(dW_member, dW_REF) — mlp-L{site}")
        ax.set_title(f"dW-alignment at graft site L{site} (red = cohort mean)")
        ax.axhline(0.0, color="k", lw=0.8)
    for gi, g in enumerate(gens):
        recs = included(MEMBERS[g])
        axes[2].scatter([gi] * len(recs), [r["raw_cos"][SITES[0]] for r in recs],
                        color="#2c6fbb", marker="o", s=18,
                        label="L2" if gi == 0 else None)
        axes[2].scatter([gi] * len(recs), [r["raw_cos"][SITES[1]] for r in recs],
                        color="#b8860b", marker="s", s=18,
                        label="L3" if gi == 0 else None)
    axes[2].set_xticks(range(len(gens)), [f"gen-{g}" for g in gens])
    axes[2].set_ylabel("raw cos(W_member, W_REF)")
    axes[2].set_title("secondary: raw weight-space alignment at graft organs")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].legend(fontsize=8)
    fig.suptitle("E040 dW-alignment drift under selection "
                 "(genotype inits anchored to the seed-42 family)")
    fig.tight_layout()
    fig.savefig(rd / "dw_alignment.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- outputs / verdicts
def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("init_sd", "trained_sd")}


def finish(rd, partial: bool):
    global REF_REC
    stats = {g: cohort_stats(recs) for g, recs in MEMBERS.items()}
    contrast = gen_contrast(MEMBERS[0], MEMBERS[2], seed=20800)

    # reverse-graft control on the gen-2 winner (REF <- winner organs)
    reverse = None
    inc2 = included(MEMBERS[2])
    if inc2 and REF_REC is not None:
        winner = min(inc2, key=lambda r: r["R_mean"])
        net = EVAL_NET
        net.load_state_dict(REF_REC["trained_sd"])
        snap = snapshot(net)
        base = per_batch_losses_cpu(net, CORPUS)
        reverse = {"winner": winner["name"], "cells": {}}
        for site in SITES:
            transplant(net, winner["trained_sd"], mlp_keys(site))
            losses = per_batch_losses_cpu(net, CORPUS)
            net.load_state_dict(snap)
            assert_identical(net, snap, "ref(reverse)")
            diffs = [a - b for a, b in zip(losses, base)]
            lo, hi = bootstrap_ci(diffs, seed=20900 + site)
            reverse["cells"][site] = {
                "dce": mean(diffs), "ci95": [lo, hi],
                "r_vs_ref_ablate": mean(diffs) / REF_REC["ablate"][site]["dce"]}
        log(f"reverse-graft REF <- {winner['name']}: " + " | ".join(
            f"L{s} dCE {reverse['cells'][s]['dce']:+.3f} "
            f"(R {reverse['cells'][s]['r_vs_ref_ablate']:.2f})" for s in SITES))

    # ---------------- verdicts (T022 frozen bars) ----------------
    s0, s2 = stats.get(0, {}), stats.get(2, {})
    v = {"partial": partial, "conditions": {}}
    w_rec = next((r for r in MEMBERS[0] if r["name"] == "e040_w" and "base_ce" in r), None)
    if s0.get("n") and s2.get("n") and w_rec is not None:
        D0, D2, R0, R2 = s0["D_mean"], s2["D_mean"], s0["R_mean"], s2["R_mean"]
        a0, a2 = s0["align_mean"], s2["align_mean"]
        inc0m, inc2m = included(MEMBERS[0]), included(MEMBERS[2])
        val_w = w_rec["base_ce"]
        gates_gen2 = all(r["eligible"] for r in inc2m)
        gates_gen0 = all(r["eligible"] for r in inc0m)
        d_drop = D0 - D2
        align_shift = a2 - a0
        c = v["conditions"]
        c.update({
            "D_gen0": D0, "D_gen2": D2, "D_ratio": D2 / D0 if D0 else None,
            "R_gen0": R0, "R_gen2": R2, "R_ratio": R2 / R0 if R0 else None,
            "align_gen0": a0, "align_gen2": a2, "align_shift": align_shift,
            "D_drop_frac": d_drop / D0 if D0 else None,
            "contrast_ci_excludes_0": contrast.get("D_ci_excludes_0"),
            "gates_gen2_all_eligible": gates_gen2,
            "gates_gen0_all_eligible": gates_gen0,
        })
        p1 = bool(D2 <= 0.75 * D0 and R2 <= 0.75 * R0
                  and contrast.get("D_ci_excludes_0") and gates_gen2 and gates_gen0)
        p2 = bool((abs(d_drop) < 0.10 * D0 or not contrast.get("D_ci_excludes_0"))
                  and align_shift < 0.05)
        p3 = bool(align_shift >= 0.05 and not p1)
        band_fail_gen2 = [r["name"] for r in inc2m if not r["band_pass"]]
        parity_broken = bool(d_drop >= 0.25 * D0 and
                             mean([r["base_ce"] for r in inc2m]) > val_w + PARITY_SLACK)
        organ_devaluing = bool(d_drop >= 0.25 * D0 and band_fail_gen2
                               and not (R2 <= 0.75 * R0))
        all_recs = [r for g in MEMBERS for r in MEMBERS[g]]
        instrument_fail = bool(
            not all(r.get("g2_determinism", True) for r in all_recs if "D" in r)
            or not all(r.get("gates", {}).get("c0_self_transplant_zero", True)
                       for r in all_recs if r.get("gates")))
        v.update({"P1_evolvable": p1, "P2_frozen": p2, "P3_degeneration_route": p3,
                  "parity_broken_route": parity_broken,
                  "organ_devaluing_route": organ_devaluing,
                  "band_fail_gen2": band_fail_gen2,
                  "instrument_failure": instrument_fail})
        if instrument_fail:
            v["verdict"] = "INSTRUMENT-FAILURE (G1/G2) — nothing interpreted"
        elif p1:
            v["verdict"] = "P1-EVOLVABLE"
        elif p3:
            v["verdict"] = "P3-DEGENERATION-ROUTE"
        elif p2:
            v["verdict"] = "P2-FROZEN"
        else:
            v["verdict"] = ("NO-REGISTERED-VERDICT (response present but below the "
                            "P1 bar; see conditions)")
        log("=" * 78)
        log(f"VERDICT: {v['verdict']}  | D {D0:.4f}->{D2:.4f} "
            f"({(D2 / D0 - 1) * 100:+.1f}%) | R {R0:.3f}->{R2:.3f} "
            f"({(R2 / R0 - 1) * 100:+.1f}%) | align {a0:+.4f}->{a2:+.4f} "
            f"(shift {align_shift:+.4f}) | CI excl 0: "
            f"{contrast.get('D_ci_excludes_0')} | gates g2/g0: "
            f"{gates_gen2}/{gates_gen0}")
    else:
        v["verdict"] = "PARTIAL (no complete gen-2 cohort — stop-rule); " \
                       "P2 direction only, with caveat"
        log("VERDICT: partial — no complete gen-2 cohort; directional read in metrics")

    figures(rd, stats)
    ref_block = None
    if REF_REC is not None:
        ref_block = {"val_ce": REF_REC["base_ce"], "g3_band": list(REF_BAND),
                     "g3_pass": bool(REF_BAND[0] <= REF_REC["base_ce"] <= REF_BAND[1]),
                     "ablate": {str(s): REF_REC["ablate"][s]["dce"] for s in SITES}}
    g0_block = None
    if w_rec is not None:
        g0_block = {"val_w_b32": w_rec["base_ce"], "anchor": E005S_ANCHOR,
                    "tol": G0_TOL,
                    "pass": bool(abs(w_rec["base_ce"] - E005S_ANCHOR) <= G0_TOL)}
    out = {
        "experiment": "e040_graft_evolution",
        "date": now_iso(),
        "smoke": False,
        "status": "parked_for_heartbeat" if STATUS["parked"] else
                  ("aborted_g5" if STATUS["aborted"] else "complete"),
        "partial": bool(partial or STATUS["aborted"]),
        "device": {"train": DEVICE, "assays": "cpu (GPU untouched during assays)"},
        "config": cfg_dict(CFG), "params": 840_704,
        "protocol": {"steps": STEPS, "lr": LR, "batch_size": BS, "cap_s": CAP_S,
                     "sigma_mut": SIGMA_MUT, "sigma_escalated": SIGMA_ESCALATED,
                     "sites": list(SITES), "n_eval_batches": N_EVAL,
                     "n_boot": N_BOOT, "parity_slack": PARITY_SLACK,
                     "band": [BAND_LO, BAND_HI], "seed_w": SEED_W,
                     "seed_ref": SEED_REF},
        "ref": ref_block,
        "gates": {"g0_anchor": g0_block,
                  "g5_excluded": [r["name"] for g in MEMBERS for r in MEMBERS[g]
                                  if r.get("excluded")],
                  "g6_thermal_log": THERMAL},
        "sigma_by_gen": SIGMA_BY_GEN,
        "members": {str(g): [strip(r) for r in MEMBERS[g]] for g in MEMBERS},
        "generation_stats": {str(g): stats[g] for g in stats},
        "gen2_vs_gen0_contrast": contrast,
        "selection_events": SELECTION,
        "reverse_graft_control": reverse,
        "verdicts": v,
        "timing_s": {"total": round(elapsed(), 1)},
    }
    save_json(rd / "metrics.json", out)
    log(f"outputs: {rd} (metrics.json, lineage_trait.png, dw_alignment.png)")
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
    global CFG, CORPUS, EVAL_NET, REF_SD, REF_DW, REF_RAW, REF_REC
    rd = run_dir("e040")
    log("E040 graft-evolution lineage — genotype=init, phenotype=trained net, "
        "selection=graft compatibility vs fixed REF (seed 4304)")
    CORPUS = CharCorpus(CORPUS_PATH, seed=1337)       # identical batch order for all
    CFG = Cfg(vocab=CORPUS.vocab_size, n_layer=4, n_head=4, n_embd=128, block_size=256)
    EVAL_NET = TinyGPT(CFG)                            # CPU assay engine
    n_params = EVAL_NET.num_params()
    assert n_params == 840_704 and n_params <= 1_000_000, n_params
    log(f"cfg e005s SMALL verbatim: 4L/4H/128d — {n_params} params <= 1M OK")

    # ---- REF (frozen donor) ------------------------------------------------
    ref_init = make_init(SEED_REF)
    ref = train_member("e040_ref", ref_init, -1, "set_seed(4304) fresh init; frozen donor")
    if ref is None:
        return park(rd)
    REF_REC = {"name": "ref", "gen": -1, "init_sd": ref_init,
               "trained_sd": ref["trained_sd"], "steps": ref["steps"]}
    REF_SD = ref["trained_sd"]

    def ref_assay():
        net = EVAL_NET
        net.load_state_dict(REF_SD)
        base1 = per_batch_losses_cpu(net, CORPUS)
        base2 = per_batch_losses_cpu(net, CORPUS)
        REF_REC["g2_determinism"] = bool(base1 == base2)
        REF_REC["base_ce"] = mean(base1)
        snap = snapshot(net)
        REF_REC["ablate"] = {}
        for site in SITES:                              # reverse-graft R context
            with lesion(net, "mlp", site):
                abl = per_batch_losses_cpu(net, CORPUS)
            REF_REC["ablate"][site] = {
                "dce": mean([a - b for a, b in zip(abl, base1)]),
                "per_batch": [a - b for a, b in zip(abl, base1)]}
        assert_identical(net, snap, "ref")
        g3 = REF_BAND[0] <= REF_REC["base_ce"] <= REF_BAND[1]
        log(f"  REF base CE {REF_REC['base_ce']:.4f} (G3 band {REF_BAND}: "
            f"{'PASS' if g3 else 'FLAG'})")

    if ref["fresh"]:
        cooldown_with(ref_assay, "e040_ref")
    else:
        ref_assay()
    REF_DW = {s: dw_vec(REF_SD, ref_init, mlp_keys(s)) for s in SITES}
    REF_RAW = {s: raw_vec(REF_SD, mlp_keys(s)) for s in SITES}

    # ---- gen-0: W + M1..M3 -------------------------------------------------
    gen0_specs = [("e040_w", make_init(SEED_W), "wildtype set_seed(42) exact init")]
    for i, s in enumerate(G0_MUT_SEEDS, start=1):
        gen0_specs.append((f"e040_m{i}", None,
                           f"init42 + eps(seed {s}, sigma {SIGMA_MUT})"))
    for i, (name, init_sd, prov) in enumerate(gen0_specs):
        if init_sd is None:                             # mutants: init42 + eps
            init_sd = mutated_init(MEMBERS[0][0]["init_sd"], G0_MUT_SEEDS[i - 1],
                                   SIGMA_MUT)
        rec = train_and_assay(name, init_sd, 0, prov, first=(i == 0))
        if rec is None:
            return park(rd)
        MEMBERS[0].append(rec)
    if sum(1 for r in MEMBERS[0] if r["excluded"]) > 1:
        return abort(rd, "gen-0: >1 member excluded (G5)")
    if STATUS["wall_violated"]:
        return finish(rd, partial=True)

    w_rec = next(r for r in MEMBERS[0] if r["name"] == "e040_w")
    assign_eligibility(MEMBERS[0], w_rec)
    stats0 = cohort_stats(MEMBERS[0])
    log(f"gen-0: D mean {stats0['D_mean']:.4f} (CI {stats0['D_ci95'][0]:+.3f},"
        f"{stats0['D_ci95'][1]:+.3f}) R mean {stats0['R_mean']:.3f} "
        f"align {stats0['align_mean']:+.4f} val(W) {w_rec['base_ce']:.4f}")
    log(f"G0 anchor: val(W_b32) {w_rec['base_ce']:.4f} vs e005s_small {E005S_ANCHOR} "
        f"(tol {G0_TOL}): "
        f"{'PASS' if abs(w_rec['base_ce'] - E005S_ANCHOR) <= G0_TOL else 'FLAG'}")

    # ---- G7 variance gate -> sigma escalation ------------------------------
    muts = [r for r in MEMBERS[0] if r["name"].startswith("e040_m") and "D" in r]
    spread = max(r["D"] for r in muts) - min(r["D"] for r in muts)
    hw = mean([(r["graft"][s]["ci95"][1] - r["graft"][s]["ci95"][0]) / 2
               for r in muts for s in SITES])
    escalated = bool(spread < 3 * hw)
    if escalated:
        SIGMA_BY_GEN[1] = SIGMA_ESCALATED
        SIGMA_BY_GEN[2] = SIGMA_ESCALATED
        log(f"G7: gen-0 mutant spread {spread:.4f} < 3x CI half-width {3 * hw:.4f} "
            f"-> sigma escalated to {SIGMA_ESCALATED} from gen-1 (P1/P2 carry the caveat)")
    else:
        log(f"G7: gen-0 mutant spread {spread:.4f} >= 3x CI half-width {3 * hw:.4f} "
            f"-> sigma stays {SIGMA_MUT}")
    SELECTION.append({"gen": 0, "g7_variance_gate": {
        "gen0_mutant_spread": spread, "ci_half_width": hw,
        "threshold_3x_hw": 3 * hw, "escalated": escalated,
        "sigma_from_gen1": SIGMA_BY_GEN[1]}})

    # ---- gen-1 / gen-2 -----------------------------------------------------
    def run_generation(gen: int, parent_recs: list[dict]) -> None:
        p1, p2, bottleneck, ranked = select_parents(parent_recs)
        sigma = SIGMA_BY_GEN[gen]
        SELECTION.append({"gen": gen, "ranking": [(r["name"], round(r["R_mean"], 4),
                                                   bool(r["eligible"])) for r in ranked],
                          "parent1": p1["name"],
                          "parent2": p2["name"] if p2 else None,
                          "bottleneck": bottleneck, "sigma": sigma})
        log(f"selection gen-{gen - 1}->{gen}: parents {p1['name']} + "
            f"{p2['name'] if p2 else '(none)'} (bottleneck={bottleneck}); "
            f"sigma {sigma}")
        seeds = MUT_SEEDS[gen]
        p2name = p2["name"] if p2 else p1["name"]
        specs = [(f"e040_g{gen}a", p1["init_sd"], seeds[0],
                  f"child of {p1['name']} (eps seed {seeds[0]}, sigma {sigma})"),
                 (f"e040_g{gen}b", p1["init_sd"], seeds[1],
                  f"child of {p1['name']} (eps seed {seeds[1]}, sigma {sigma})"),
                 (f"e040_g{gen}c", (p2 or p1)["init_sd"], seeds[2],
                  f"child of {p2name} (eps seed {seeds[2]}, sigma {sigma})")]
        for i, (name, pinit, seed, prov) in enumerate(specs):
            if elapsed() > STOP_RULE_S:
                log(f"stop-rule: skipping {name} (elapsed {elapsed():.0f}s "
                    f"> {STOP_RULE_S:.0f}s)")
                SELECTION.append({"gen": gen, "skipped": name,
                                  "reason": f"elapsed>{STOP_RULE_S:.0f}s"})
                continue
            rec = train_and_assay(name, mutated_init(pinit, seed, sigma), gen, prov,
                                  first=(i == 0))
            if rec is None:
                return
            MEMBERS[gen].append(rec)

    run_generation(1, MEMBERS[0])
    if STATUS["parked"]:
        return park(rd)
    if sum(1 for r in MEMBERS[1] if r.get("excluded")) > 1:
        return abort(rd, "gen-1: >1 member excluded (G5)")
    assign_eligibility(MEMBERS[1], w_rec)
    if STATUS["wall_violated"] or len(included(MEMBERS[1])) < 2:
        return finish(rd, partial=True)
    stats1 = cohort_stats(MEMBERS[1])
    if stats1.get("n"):
        log(f"gen-1: D mean {stats1['D_mean']:.4f} R mean {stats1['R_mean']:.3f} "
            f"align {stats1['align_mean']:+.4f}")

    run_generation(2, MEMBERS[1])
    if STATUS["parked"]:
        return park(rd)
    if sum(1 for r in MEMBERS[2] if r.get("excluded")) > 1:
        return abort(rd, "gen-2: >1 member excluded (G5)")
    assign_eligibility(MEMBERS[2], w_rec)
    stats2 = cohort_stats(MEMBERS[2])
    if stats2.get("n"):
        log(f"gen-2: D mean {stats2['D_mean']:.4f} R mean {stats2['R_mean']:.3f} "
            f"align {stats2['align_mean']:+.4f}")
    return finish(rd, partial=bool(STATUS["wall_violated"] or not stats2.get("n")))


if __name__ == "__main__":
    main()
