"""E063 — LOAD HOMEOSTASIS: is own-organ load A a defended setpoint?

REGISTERED VERDICT (frozen before running; primary rungs on the 2.7M cohort
where the clean factorial exists, 0.84M cohort supportive):

Rung definitions (mean over a cohort's MLP sites of |ΔA|, A = own-mlp
ablation dCE on 15 fixed val batches, e052 protocol):
  D_order = |A(e041_bdo) - A(e001)|      same init(42), DIFFERENT data order
                                          (batch seed 1337 vs 7777)
  D_init  = |A(e028_b43) - A(e001)|      DIFFERENT init (seed 43), same order
  D_noise = mean bootstrap se of A       instrument floor
  D_exp   = max over e048 arms of |A(arm) - A(e001)|   continued-training drift

  H-EMERGENT  if D_init <= max(2*D_noise, 0.10 nats)
              (independent inits converge to the same A profile — a universal
              load allocation; data order then cannot matter much either)
  H-FREE      elif D_order >= 0.75*D_init
              (retraining with a different data order moves A most of the way
              to a fresh init — the organ-reliance trait is plastic)
  H-SETPOINT  elif D_order <= 0.5*D_init AND D_exp <= 0.5*D_init
              (A is init-determined and defended: order/re-exposure perturb it
              far less than re-initialization does)
  else        MIXED (report which leg failed)

0.84M supportive test (same thresholds on analogous rungs): family =
{w, m1..m3, g1a..g2c} (10 nets, init42+eps lineage, IDENTICAL batch order);
within-family excess spread (variance above the bootstrap noise floor) vs
across-init distance |A(e040_ref, seed 4304) - A(family centroid)|.
family_excess <= 0.5*across_init supports the setpoint reading;
family_excess >= across_init supports free drift.

CHECKPOINT REALITY (differs from the task brief — verified before running):
  0.84M = 4L/4H/128d (840,704 params): the 11 e040 lineage ckpts
  (ref=seed4304, w=seed42, m*/g* = init42 + sigma-0.005 eps), e005s_small
  (seed42, same 4000-step protocol as w — a free REPLICATE rung), e033
  (seed42 + write-equalizer constraint — an intervention probe, excluded
  from family stats).
  e041_bdo and ALL e048 ckpts are 2.7M = 6L/6H/192d (trainstate dicts, model
  under ['model']): BDO = init42/order7777; e048_repro/dose/direct400/800 =
  e001-host continued 400/1600 (install Dmix) and 400/800 (direct spliced
  corpus) steps. There are no 400/800/4000-step dose checkpoints of the
  0.84M net; exposure is read on the 2.7M arms instead.

Run: python lab/e063_load_homeostasis.py   (CPU-only; CUDA masked pre-torch)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # strict CPU-only

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sstats

import common

common.DEVICE = "cpu"
from common import (REPO, Cfg, CharCorpus, TinyGPT, lesion, run_dir,
                    save_json, set_seed)

CORPUS_PATH = REPO / "data" / "input.txt"
CKPT = REPO / "runs" / "checkpoints"
N_EVAL = 15                                        # e052 fixed-batch protocol
N_BOOT = 2000
T0 = time.time()

CFG_A = dict(n_layer=4, n_head=4, n_embd=128)      # 0.84M family
CFG_B = dict(n_layer=6, n_head=6, n_embd=192)      # 2.7M family

# name -> (cohort, group, load_mode, note)
COHORT_A = {                                       # plain state dicts
    "e040_ref":       ("init_ref", "seed-4304 fresh init (frozen donor)"),
    "e040_w":         ("family",   "seed-42 exact init (wildtype)"),
    "e005s_small":    ("replicate", "seed-42, same protocol as w (e005s)"),
    "e040_m1":        ("family",   "init42 + eps(4021)"),
    "e040_m2":        ("family",   "init42 + eps(4022)"),
    "e040_m3":        ("family",   "init42 + eps(4023)"),
    "e040_g1a":       ("family",   "gen-1 from m1"),
    "e040_g1b":       ("family",   "gen-1 from m1"),
    "e040_g1c":       ("family",   "gen-1 from m2"),
    "e040_g2a":       ("family",   "gen-2 from g1c"),
    "e040_g2b":       ("family",   "gen-2 from g1c"),
    "e040_g2c":       ("family",   "gen-2 from g1a"),
    "e033":           ("probe",    "write-equalizer net (e005s_small base + "
                                   "constrained training; needs its eval hook)"),
}
# e033's equalizer is a forward hook active in train AND eval (e033 source);
# the saved net is BROKEN without it (hookless base CE ~3.9). Measuring A
# with the hook pinned = the net's functioning organ load. Uniform write
# target = mean of the baseline per-layer norms stored in runs/e033.
EQ_TARGET = 1.637113148021873
COHORT_B = {                                       # e001/e041/e028 plain; e048 trainstates
    "e001":           ("anchor",   "B: init42, batch order 1337"),
    "e041_bdo":       ("order",    "init42 bitwise, batch order 7777"),
    "e028_b43":       ("init",     "seed-43 fresh init, batch order 1337"),
    "e048_repro":     ("exposure", "B + 400 install(Dmix) steps"),
    "e048_dose":      ("exposure", "B + 1600 install(Dmix) steps"),
    "e048_direct400": ("exposure", "B + 400 direct(spliced) steps"),
    "e048_direct800": ("exposure", "B + 800 direct(spliced) steps"),
}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def load_sd(name: str) -> dict:
    obj = torch.load(CKPT / f"{name}.pt", map_location="cpu", weights_only=False)
    return obj["model"] if "model" in obj else obj


@torch.no_grad()
def per_batch_losses_cpu(model, corpus, n_batches: int = N_EVAL) -> list[float]:
    """e052/e028/e040 fixed-batch protocol (RNG seeded by corpus.seed=1337)."""
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


def name_seed(name: str) -> int:
    return 30630 + sum(ord(c) * (i + 1) for i, c in enumerate(name))


def boot_se(diffs: list[float], seed: int) -> float:
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs)
    means = rng.choice(arr, size=(N_BOOT, arr.size), replace=True).mean(axis=1)
    return float(means.std(ddof=1))


def profile_dist(a: dict[str, float], b: dict[str, float]) -> float:
    """Mean over shared sites of |ΔA| — the registered rung metric."""
    ks = sorted(set(a) & set(b))
    return float(np.mean([abs(a[k] - b[k]) for k in ks]))


def register_equalizer(model, target: float = EQ_TARGET):
    """e033's constraint verbatim: rescale every MLP residual write to `target`."""
    hooks = []
    for block in model.h:
        def post(m, a, o, t=target):
            n = o.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return o * (t / n)
        hooks.append(block.mlp.register_forward_hook(post))
    return hooks


def measure_net(name: str, group: str, note: str, cohort: str, cfg, corpus):
    net = TinyGPT(cfg)
    net.load_state_dict(load_sd(name))
    eq_hooks = register_equalizer(net) if name == "e033" else []
    base = per_batch_losses_cpu(net, corpus)
    A, se, per_batch = {}, {}, {}
    for s in range(cfg.n_layer):
        with lesion(net, "mlp", s):
            abl = per_batch_losses_cpu(net, corpus)
        diffs = [a - b for a, b in zip(abl, base)]
        A[s] = float(np.mean(diffs))
        se[s] = boot_se(diffs, seed=name_seed(name) + s)
        per_batch[s] = diffs
    for h in eq_hooks:
        h.remove()
    return {"cohort": cohort, "group": group, "note": note,
            "base_ce": float(np.mean(base)), "A": A, "se": se,
            "per_batch": per_batch}


def main():
    rd = run_dir("e063")
    log("E063 load homeostasis — own-organ load A across 20 checkpoints "
        "(CPU-only)")
    corpus = CharCorpus(CORPUS_PATH, seed=1337)

    # ---- measure A at every MLP site for every checkpoint (cached) ---------
    cache_path = rd / "cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) \
        if cache_path.exists() else {}
    if "e033" in cache and "hook_active" not in cache["e033"]:
        cache.pop("e033")          # invalidate hookless-e033 measurements
    nets: dict[str, dict] = {}
    for cohort, cfgkw in (("A", CFG_A), ("B", CFG_B)):
        table = COHORT_A if cohort == "A" else COHORT_B
        cfg = Cfg(vocab=corpus.vocab_size, block_size=256, **cfgkw)
        for name, (group, note) in table.items():
            if name in cache:
                rec = cache[name]
                for fld in ("A", "se", "per_batch"):
                    rec[fld] = {int(k): v for k, v in rec[fld].items()}
                nets[name] = rec
                log(f"  {name:15s} [{cohort}/{group:9s}] cached | base CE "
                    f"{nets[name]['base_ce']:.4f} | "
                    + " ".join(f"A@L{s} {nets[name]['A'][s]:+.3f}"
                               for s in range(cfg.n_layer)))
                continue
            nets[name] = measure_net(name, group, note, cohort, cfg, corpus)
            if name == "e033":
                nets[name]["hook_active"] = True
            log(f"  {name:15s} [{cohort}/{group:9s}] base CE "
                f"{nets[name]['base_ce']:.4f} | "
                + " ".join(f"A@L{s} {nets[name]['A'][s]:+.3f}"
                           for s in range(cfg.n_layer)))
        cache_path.write_text(json.dumps(nets, indent=1, default=float),
                              encoding="utf-8")

    # ---- instrument gates ----------------------------------------------------
    w_sd = load_sd("e040_w")
    small_sd = load_sd("e005s_small")
    w_small_bitwise = all(torch.equal(w_sd[k], small_sd[k]) for k in w_sd)
    gates = {
        "ref_base_ce_15batch": nets["e040_ref"]["base_ce"],
        "ref_base_ce_e052_reference": 1.5350,
        "B_base_ce_15batch": nets["e001"]["base_ce"],
        "B_base_ce_e029_30batch_reference": 1.622391,
        "w_vs_e005s_small_bitwise_identical": bool(w_small_bitwise),
        "e033_hooked_base_ce_15batch": nets["e033"]["base_ce"],
        "e033_hooked_base_ce_reference_e033": 1.5147,
        "n_checkpoints": len(nets),
    }
    log(f"gates: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in gates.items()})}")

    def A_of(name: str) -> dict[str, float]:
        return {str(k): v for k, v in nets[name]["A"].items()}

    # ---- cohort A (0.84M) rungs ----------------------------------------------
    famA = [n for n, m in nets.items() if m["group"] == "family"]
    sitesA = [str(s) for s in range(4)]
    fam_prof = {s: float(np.mean([nets[n]["A"][int(s)] for n in famA])) for s in sitesA}
    fam_std = {s: float(np.std([nets[n]["A"][int(s)] for n in famA], ddof=1))
               for s in sitesA}
    noise_A = float(np.mean([se for n in famA for se in nets[n]["se"].values()]))
    fam_excess = float(np.sqrt(max(
        np.mean([v ** 2 for v in fam_std.values()]) - noise_A ** 2, 0.0)))
    across_init_A = profile_dist(A_of("e040_ref"), fam_prof)
    replicate_A = profile_dist(A_of("e005s_small"), A_of("e040_w"))
    mutant_spread = {s: float(np.std([nets[n]["A"][int(s)] for n in
                                      ["e040_m1", "e040_m2", "e040_m3"]], ddof=1))
                     for s in sitesA}
    probe_e033 = profile_dist(A_of("e033"), A_of("e005s_small"))  # its baseline
    # profile SHAPE stability (allocation ratios), w vs ref and B vs B43/BDO
    shape_w_ref = sstats.pearsonr([nets["e040_w"]["A"][s] for s in range(4)],
                                  [nets["e040_ref"]["A"][s] for s in range(4)])[0]

    cohA = {
        "family_members": famA,
        "family_centroid": fam_prof, "family_std": fam_std,
        "mutant_spread_m123": mutant_spread,
        "rungs_mean_over_4_sites": {
            "D_noise": noise_A,
            "D_replicate_same_init_same_order(w_vs_e005s)": replicate_A,
            "D_family_excess_spread": fam_excess,
            "D_across_init(ref4304_vs_family)": across_init_A,
            "D_probe_equalizer(e033_vs_w)": probe_e033,
        },
        "supportive_test": {
            "family_excess": fam_excess, "across_init": across_init_A,
            "ratio": fam_excess / across_init_A if across_init_A > 0 else float("inf"),
            "setpoint_supports": bool(fam_excess <= 0.5 * across_init_A),
            "free_supports": bool(fam_excess >= across_init_A),
        },
        "shape_pearson_w_vs_ref": float(shape_w_ref),
    }
    log(f"cohort A rungs: noise {noise_A:.4f} | replicate {replicate_A:.4f} | "
        f"family-excess {fam_excess:.4f} | across-init {across_init_A:.4f} | "
        f"equalizer probe {probe_e033:.4f}")

    # ---- cohort B (2.7M) rungs — PRIMARY --------------------------------------
    order = profile_dist(A_of("e041_bdo"), A_of("e001"))
    init = profile_dist(A_of("e028_b43"), A_of("e001"))
    exp_arms = {n: profile_dist(A_of(n), A_of("e001")) for n in nets
                if nets[n]["group"] == "exposure"}
    noise_B = float(np.mean([se for n, m in nets.items()
                             if m["cohort"] == "B" for se in m["se"].values()]))
    exp_max = max(exp_arms.values())
    shape_B_BDO = sstats.pearsonr([nets["e001"]["A"][s] for s in range(6)],
                                  [nets["e041_bdo"]["A"][s] for s in range(6)])[0]
    shape_B_B43 = sstats.pearsonr([nets["e001"]["A"][s] for s in range(6)],
                                  [nets["e028_b43"]["A"][s] for s in range(6)])[0]
    cohB = {
        "anchor": "e001 (B: init42, order 1337)",
        "rungs_mean_over_6_sites": {
            "D_noise": noise_B, "D_order_bdo": order, "D_init_b43": init,
            "D_exp_max": exp_max, **{f"D_exp_{k}": v for k, v in exp_arms.items()},
        },
        "ratios": {"order_over_init": order / init if init > 0 else float("inf"),
                   "exp_over_init": exp_max / init if init > 0 else float("inf")},
        "shape_pearson_B_vs_BDO": float(shape_B_BDO),
        "shape_pearson_B_vs_B43": float(shape_B_B43),
    }
    log(f"cohort B rungs: noise {noise_B:.4f} | ORDER {order:.4f} | INIT "
        f"{init:.4f} | exp-max {exp_max:.4f} | shape r(B,BDO) "
        f"{shape_B_BDO:+.3f} r(B,B43) {shape_B_B43:+.3f}")

    # ---- registered verdict ----------------------------------------------------
    emergent_bar = max(2 * noise_B, 0.10)
    if init <= emergent_bar:
        verdict = ("H-EMERGENT — different inits converge to the same A profile "
                   f"(D_init {init:.3f} <= bar {emergent_bar:.3f}); load "
                   "allocation is universal, not init-determined")
    elif order >= 0.75 * init:
        verdict = ("H-FREE — same init + different data order moves A "
                   f"{order / init * 100:.0f}% of the way to a fresh init "
                   f"(D_order {order:.3f} >= 0.75*D_init {init:.3f}); the "
                   "organ-reliance trait is plastic, data-order-determined")
    elif order <= 0.5 * init and exp_max <= 0.5 * init:
        verdict = ("H-SETPOINT — A is init-determined and defended: order "
                   f"perturbs it {order / init * 100:.0f}% of the re-init "
                   f"distance (D_order {order:.3f} vs D_init {init:.3f}) and "
                   f"continued-training drift stays at {exp_max / init * 100:.0f}%"
                   " (D_exp " + f"{exp_max:.3f})")
    else:
        legs = []
        if order > 0.5 * init:
            legs.append(f"order leg failed (D_order {order:.3f} > 0.5*D_init "
                        f"{0.5 * init:.3f})")
        if exp_max > 0.5 * init:
            legs.append(f"exposure leg failed (D_exp {exp_max:.3f} > 0.5*D_init "
                        f"{0.5 * init:.3f})")
        verdict = (f"MIXED — not setpoint, not fully free: D_order {order:.3f} vs "
                   f"D_init {init:.3f} (ratio {order / init:.2f}); "
                   + "; ".join(legs))
    log("=" * 78)
    log(f"VERDICT: {verdict}")

    # ---- outputs ----------------------------------------------------------------
    metrics = {
        "experiment": "e063_load_homeostasis",
        "date": common.now_iso(),
        "status": "complete",
        "device": "cpu-only (CUDA_VISIBLE_DEVICES empty; common.DEVICE=cpu)",
        "question": ("does own-organ load A converge to a defended setpoint, "
                     "drift freely with data order/training, or emerge "
                     "regardless of init?"),
        "protocol": {"A": "dCE of zeroing own mlp organ, mean over 15 fixed "
                          "val batches (e052 protocol, corpus seed 1337)",
                     "rung": "mean over cohort MLP sites of |ΔA|",
                     "n_eval_batches": N_EVAL, "n_boot": N_BOOT,
                     "checkpoint_reality": (
                         "brief assumed BDO+e048 were 0.84M; verified they are "
                         "2.7M/6L trainstates (model under ['model']); no "
                         "0.84M dose ckpts exist — exposure read on 2.7M arms"),
                     "registered_verdict_rules": {
                         "H-emergent": "D_init <= max(2*D_noise, 0.10)",
                         "H-free": "D_order >= 0.75*D_init",
                         "H-setpoint": "D_order <= 0.5*D_init AND D_exp <= 0.5*D_init"}},
        "instrument_gates": gates,
        "checkpoints": {n: {"cohort": m["cohort"], "group": m["group"],
                            "note": m["note"], "base_ce": m["base_ce"],
                            "A_by_site": A_of(n), "bootstrap_se": {
                                str(k): v for k, v in m["se"].items()}}
                        for n, m in nets.items()},
        "cohort_A_0.84M": cohA,
        "cohort_B_2.7M_primary": cohB,
        "verdict": verdict,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", metrics)

    # ---- A-by-checkpoint figure --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    xs = np.arange(4)
    fam_vals = np.array([[nets[n]["A"][s] for s in range(4)] for n in famA])
    ax.fill_between(xs, fam_vals.min(0), fam_vals.max(0), color="#bbbbbb",
                    alpha=0.5, label="seed-42 family band (10 nets)")
    for i, n in enumerate(famA):
        ax.plot(xs, fam_vals[i], color="#888", lw=0.6, zorder=2)
    ax.plot(xs, [nets["e040_w"]["A"][s] for s in range(4)], color="#2c6fbb",
            lw=2.4, marker="o", label="w (seed42)", zorder=4)
    ax.plot(xs, [nets["e005s_small"]["A"][s] for s in range(4)], color="#2aa198",
            lw=1.8, marker="s", ls="--", label="e005s_small (replicate)", zorder=4)
    ax.plot(xs, [nets["e040_ref"]["A"][s] for s in range(4)], color="#c0392b",
            lw=2.0, marker="^", ls=":", label="ref (seed 4304)", zorder=4)
    ax.plot(xs, [nets["e033"]["A"][s] for s in range(4)], color="#7b3fa0",
            lw=1.8, marker="d", ls="-.", label="e033 (equalizer, hook active)", zorder=4)
    ax.set_xticks(xs); ax.set_xlabel("MLP site (layer)"); ax.set_ylabel("A = own-ablation dCE (nats)")
    ax.set_title("A) 0.84M cohort — A profiles at 4 MLP sites")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    xs6 = np.arange(6)
    styles = {"e001": ("#2c6fbb", "-", "o", 2.4, "B (init42, order 1337)"),
              "e041_bdo": ("#e67e22", "-", "s", 2.0, "BDO (init42, order 7777) — ORDER rung"),
              "e028_b43": ("#c0392b", ":", "^", 2.0, "B43 (init43) — INIT rung"),
              "e048_repro": ("#2aa198", "--", "v", 1.2, "B+400 install"),
              "e048_dose": ("#2aa198", "--", "D", 1.2, "B+1600 install"),
              "e048_direct400": ("#7b3fa0", "-.", "P", 1.2, "B+400 direct"),
              "e048_direct800": ("#7b3fa0", "-.", "X", 1.2, "B+800 direct")}
    for n, (c, ls, mk, lw, lab) in styles.items():
        ax.plot(xs6, [nets[n]["A"][s] for s in range(6)], color=c, ls=ls,
                marker=mk, lw=lw, label=lab)
    ax.set_xticks(xs6); ax.set_xlabel("MLP site (layer)"); ax.set_ylabel("A (nats)")
    ax.set_title("B) 2.7M cohort — A profiles at 6 MLP sites (PRIMARY rungs)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    labels_A = ["noise", "replicate\n(same init+order)", "family\nexcess",
                "across init\n(ref 4304)", "equalizer\nprobe (e033)"]
    vals_A = [noise_A, replicate_A, fam_excess, across_init_A, probe_e033]
    labels_B = ["noise", "ORDER\n(BDO)", "INIT\n(B43)", "exposure\n(max)"]
    vals_B = [noise_B, order, init, exp_max]
    x0 = np.arange(len(labels_A))
    ax.bar(x0 - 0.2, vals_A, width=0.4, color="#2c6fbb", label="0.84M cohort")
    ax.bar([len(labels_A) + 0.5 + i for i in range(len(labels_B))], vals_B,
           width=0.4, color="#e67e22", label="2.7M cohort (primary)")
    ax.set_xticks(list(x0 - 0.2) + [len(labels_A) + 0.5 + i for i in range(len(labels_B))])
    ax.set_xticklabels(labels_A + labels_B, fontsize=7)
    for xx, vv in zip(list(x0 - 0.2) + [len(labels_A) + 0.5 + i for i in range(len(labels_B))],
                      vals_A + vals_B):
        ax.text(xx, vv + 0.01, f"{vv:.3f}", ha="center", fontsize=7)
    ax.set_ylabel("mean |ΔA| over sites (nats)")
    ax.set_title("C) drift rungs vs re-init distance")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 1]
    steps = [400, 800]; direct = [exp_arms["e048_direct400"], exp_arms["e048_direct800"]]
    steps2 = [400, 1600]; install = [exp_arms["e048_repro"], exp_arms["e048_dose"]]
    ax.plot([0] + steps, [0] + direct, color="#7b3fa0", marker="P", label="direct arm (spliced corpus)")
    ax.plot([0] + steps2, [0] + install, color="#2aa198", marker="v", label="install arm (Dmix)")
    ax.axhline(order, color="#e67e22", ls="--", lw=1.2, label=f"D_order {order:.3f}")
    ax.axhline(init, color="#c0392b", ls=":", lw=1.6, label=f"D_init {init:.3f}")
    ax.axhline(noise_B, color="#666", ls="-.", lw=1, label=f"noise {noise_B:.3f}")
    ax.set_xlabel("added training steps on top of B"); ax.set_ylabel("|ΔA| from B (nats)")
    ax.set_title("D) exposure drift vs the order/init rungs")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle("E063 — load homeostasis: is own-organ load A a defended setpoint?\n"
                 f"VERDICT: {verdict.split(' — ')[0]}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(rd / "A_by_checkpoint.png", dpi=140)
    plt.close(fig)
    log(f"outputs: {rd} (metrics.json, A_by_checkpoint.png)")
    return metrics


if __name__ == "__main__":
    main()
