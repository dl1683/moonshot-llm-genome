"""E064 — the T029 registered stress test: does the R-ladder peak track the
host's causal gate, or mid-stack idiosyncrasy? (CPU-only, no training.)

T029 (downgraded to SUGGESTIVE by the interpreter audit) registered this
exact experiment: E058 found B's R-ladder peaking at L3 (gate L3) and B43's
at L4 (gate L4) — but the two gates are adjacent and near mid-stack, so
"peak AT the gate" was a third, unregistered hypothesis. The clean
discriminator is a host whose measured causal gate sits FAR from mid-stack.

Causal-gate anchors (published runs/e012d/metrics.json nets, frozen):
  B    seed42 base   causal_mode=3  (hist [214,32,172,324,308,250])
  B43  seed43 base   causal_mode=4  ([239,36,253,299,335,225])
  R    seed42 renorm causal_mode=4  ([213,11,158,241,368,336])
  R43  seed43 renorm causal_mode=5  ([222,11,112,229,309,423]) <- gate FAR
                                    from mid-stack (L2/L3) — THE host.

Design:
  Free pre-step (registered): paired per-batch bootstrap on E058's B/B43
  R-ladders (runs/e058/metrics.json graft_per_batch, 15 batches, ablation
  means fixed — per-batch ablations were not stored) — are the original
  peaks even stable?
  Main: hosts R (e014b.pt) and R43 (e029_r43.pt), both renorm models
  evaluated WITH renorm hooks (e014b/e029 protocol: hooks active in train
  AND eval); donor B (e001.pt, seed 42) site-matched MLP organs grafted
  into all 6 sites, plus own-ablation A per site; R-ladder = D/A per site
  (e058 scale-B machinery, same 15 fixed batches).

Instrument gates:
  - renorm liveness assert (e029 §3: block-input norms == 5.6 +- 1e-3)
  - C0 self-transplant bitwise / dCE exactly 0.0 on both hosts
  - base CE vs e029's 30-batch references (tol 0.05; mine are 15-batch)
  - R<-B graft per-batch at L0/L3/L5 vs runs/e029 'R<-regime|L{s}|mlp'
    first 15 batches (GPU-eval'd there: 2e-3 tolerance, e058 gate-B2 rule)
  - R ablation L0/L3/L5 vs e029 in-run ablate_refs (30-batch GPU refs,
    tol 0.1)
  - hosts bitwise-restored after every cell

REGISTERED VERDICT (frozen in THINKING.md T029 before running):
  R43 (gate L5) is the primary read:
    R43 R-ladder peak at L5            -> unification SURVIVES the stress
                                          test (peak tracks the gate)
    R43 R-ladder peak at mid-stack
      (L3 or L4) with gate at L5       -> KILLED by shared-method bias
                                          (the peak tracks mid-stack
                                          idiosyncrasy, not the gate)
    anything else (L0/L1/L2)           -> ambiguous
  R (gate L4) is the secondary read: peak at L4 supports; elsewhere
  weakens (reported, does not override the primary).

Run: python lab/e064_gate_stress.py   (CPU-only; CUDA masked before torch)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # strict CPU (e053 computing)

import json
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import common

common.DEVICE = "cpu"                              # everything on CPU
from common import (REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json)

# ---------------------------------------------------------------- protocol
CORPUS_PATH = REPO / "data" / "input.txt"
CKPT_DIR = REPO / "runs" / "checkpoints"
E058_METRICS = REPO / "runs" / "e058" / "metrics.json"
E029_METRICS = REPO / "runs" / "e029" / "metrics.json"

CKPT = {"B": CKPT_DIR / "e001.pt",       # donor: seed42 base
        "R": CKPT_DIR / "e014b.pt",      # host : seed42 renorm
        "R43": CKPT_DIR / "e029_r43.pt"} # host : seed43 renorm (gate L5)
SITES = (0, 1, 2, 3, 4, 5)
N_EVAL = 15                                        # e058 scale-B protocol
C_RENORM = 5.6
N_BOOT = 2000
SEED_BOOT_PRE, SEED_BOOT_MAIN = 2064, 2065
BASE_REF_30 = {"R": 1.6099902311960856, "R43": 1.559572164217631}  # e029, hooks

GATE = {"B": 3, "B43": 4, "R": 4, "R43": 5}        # e012d causal modes
MIDSTACK = (3, 4)                                  # on a 6L net

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def mean(xs) -> float:
    return float(sum(xs) / len(xs))


# ------------------------------------------------ surgery units (e028/e058 verbatim)
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


# ------------------------------------------------ renorm hooks (e014b/e029 verbatim)
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
def renorm_liveness_assert(model, corpus) -> list[float]:
    hs = register_renorm(model)
    probes, handles = [], []
    for block in model.h:
        def mk():
            def pre(m, args):
                probes.append(float(args[0].norm(dim=-1).mean()))
                return None
            return pre
        handles.append(block.register_forward_pre_hook(mk()))  # AFTER renorm hooks
    x, _ = corpus.get_batch("val", model.cfg.block_size, 16,
                             gen=torch.Generator().manual_seed(7))
    model(x)
    for h in handles:
        h.remove()
    for h in hs:
        h.remove()
    if len(probes) != model.cfg.n_layer or not all(
            abs(n - C_RENORM) <= 1e-3 for n in probes):
        raise SystemExit(f"RENORM LIVENESS ASSERT FAILED: block-input norms "
                         f"{probes} != {C_RENORM} +- 1e-3 — R/R43 evals would "
                         f"be off-manifold")
    return probes


# ------------------------------------------------ fixed-batch eval (e058 verbatim)
@torch.no_grad()
def per_batch_losses_cpu(model, corpus, n_batches: int = N_EVAL) -> list[float]:
    """e028/e040/e052/e058 fixed-batch protocol (RNG stream seeded corpus.seed)."""
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


def eval_with_renorm(model, corpus, renorm: bool) -> list[float]:
    hs = register_renorm(model) if renorm else []
    losses = per_batch_losses_cpu(model, corpus)
    for h in hs:
        h.remove()
    return losses


@torch.no_grad()
def ablate_with_renorm(model, corpus, site: int, renorm: bool) -> list[float]:
    """Own-ablation of the site's MLP with renorm hooks active (e029 protocol)."""
    model.eval()
    hs = register_renorm(model) if renorm else []

    def zero_out(module, args, out):
        return torch.zeros_like(out)

    hh = model.h[site].mlp.register_forward_hook(zero_out)
    cfg = model.cfg
    gen = torch.Generator().manual_seed(corpus.seed)
    src = corpus.val
    losses = []
    for _ in range(N_EVAL):
        ix = torch.randint(len(src) - cfg.block_size - 1, (16,), generator=gen)
        x = torch.stack([src[i: i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1: i + 1 + cfg.block_size] for i in ix])
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    hh.remove()
    for h in hs:
        h.remove()
    model.train()
    return losses


# ------------------------------------------------ paired bootstrap (e011c/e028 discipline)
def bootstrap_ladder(per_batch: dict[int, list[float]], ablate: dict[int, float],
                     seed: int, n: int = N_BOOT) -> dict:
    """Paired resample of the 15 shared eval batches; recompute the R-ladder
    (D_boot/A fixed — e058 stored only ablation means) and its argmax."""
    rng = random.Random(seed)
    nb = len(next(iter(per_batch.values())))
    peaks, r_dists = [], {s: [] for s in per_batch}
    d_peaks = []
    for _ in range(n):
        idx = [rng.randrange(nb) for _ in range(nb)]
        r_lad, d_lad = {}, {}
        for s, diffs in per_batch.items():
            d = mean([diffs[i] for i in idx])
            d_lad[s] = d
            r_lad[s] = d / ablate[s]
            r_dists[s].append(r_lad[s])
        peaks.append(max(r_lad, key=r_lad.get))
        d_peaks.append(max(d_lad, key=d_lad.get))

    def dist(ps) -> dict[str, float]:
        return {f"L{s}": ps.count(s) / len(ps) for s in sorted(per_batch)}

    def ci95(vals) -> list[float]:
        v = sorted(vals)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]

    modal = max(set(peaks), key=peaks.count)
    covered = sorted({p for p in peaks if peaks.count(p) / len(peaks) >= 0.025})
    return {"n_boot": n, "n_batches": nb, "R_peak_modal": modal,
            "R_peak_dist": dist(peaks), "R_peak_ci95_sites": covered,
            "D_peak_modal": max(set(d_peaks), key=d_peaks.count),
            "D_peak_dist": dist(d_peaks),
            "R_per_site_ci95": {f"L{s}": ci95(r_dists[s]) for s in sorted(per_batch)}}


# ---------------------------------------------------------------- pre-step
def prestep_bootstrap_e058() -> dict:
    """Free pre-step (registered): are E058's B/B43 R-ladder peaks stable
    under paired per-batch resampling?"""
    e058 = json.loads(E058_METRICS.read_text(encoding="utf-8"))
    out = {}
    for host in ("B", "B43"):
        donor = "B43" if host == "B" else "B"
        pb = {s: e058["scaleB"]["graft_per_batch"][f"{host}<-{donor}|L{s}"]
              for s in SITES}
        ab = {s: e058["scaleB"]["ablate"][host][str(s)] for s in SITES}
        point_R = {s: mean(pb[s]) / ab[s] for s in SITES}
        peak = max(point_R, key=point_R.get)
        boot = bootstrap_ladder(pb, ab, SEED_BOOT_PRE)
        out[host] = {"gate": GATE[host], "point_R": {f"L{s}": point_R[s]
                                                     for s in SITES},
                     "point_peak": peak,
                     "e058_reported_peak": e058["scaleB_per_direction"][host]
                                           ["R_ladder_peak_site"],
                     "bootstrap": boot}
        log(f"pre-step {host} (gate L{GATE[host]}): point peak L{peak} | boot "
            f"modal L{boot['R_peak_modal']} dist "
            + " ".join(f"L{s}:{p:.2f}" for s, p in boot["R_peak_dist"].items()))
    return out


# ---------------------------------------------------------------- main
def main():
    rd = run_dir("e064")
    log("E064 gate stress test (T029 registration): R/R43 R-ladders with donor B "
        "— does the peak track the causal gate or mid-stack? (CPU-only)")

    # ---- free pre-step FIRST (registered) ----
    prestep = prestep_bootstrap_e058()

    corpus = CharCorpus(CORPUS_PATH, seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192,
              block_size=256)
    net = TinyGPT(cfg)
    assert net.num_params() == 2_739_072, net.num_params()
    sds = {k: torch.load(p, map_location="cpu", weights_only=True)
           for k, p in CKPT.items()}
    log(f"2.7M cfg ({net.num_params()} params); B(donor) + R, R43(hosts) loaded")

    gates = {"renorm_liveness": {}, "self_transplant_zero": {},
             "base_ce": {}, "base_ce_vs_e029": {}}
    results = {}

    for host in ("R", "R43"):
        net.load_state_dict(sds[host])
        snap = snapshot(net)
        gates["renorm_liveness"][host] = renorm_liveness_assert(net, corpus)
        # C0: self-transplant must be bitwise / dCE exactly 0 (hooks active)
        transplant(net, snap, mlp_keys(0))
        self_losses = eval_with_renorm(net, corpus, renorm=True)
        assert_identical(net, snap, f"{host}-C0")
        base = eval_with_renorm(net, corpus, renorm=True)
        gates["self_transplant_zero"][host] = bool(
            self_losses == base and mean(self_losses) - mean(base) == 0.0)
        gates["base_ce"][host] = mean(base)
        gates["base_ce_vs_e029"][host] = {
            "mine_15batch": mean(base), "e029_ref_30batch": BASE_REF_30[host],
            "abs_diff": abs(mean(base) - BASE_REF_30[host]),
            "pass_0.05": abs(mean(base) - BASE_REF_30[host]) <= 0.05}
        log(f"host {host}: base CE(hooks,15b) {mean(base):.6f} "
            f"[e029 30b ref {BASE_REF_30[host]:.6f}] | liveness ok | C0 "
            f"exact-zero {gates['self_transplant_zero'][host]}")

        graft_pb, ablate = {}, {}
        for site in SITES:
            transplant(net, sds["B"], mlp_keys(site))          # host <- B organ
            losses = eval_with_renorm(net, corpus, renorm=True)
            net.load_state_dict(snap)
            assert_identical(net, snap, f"{host}@graft-L{site}")
            graft_pb[site] = [a - b for a, b in zip(losses, base)]
            abl = ablate_with_renorm(net, corpus, site, renorm=True)
            assert_identical(net, snap, f"{host}@ablate-L{site}")
            ablate[site] = mean([a - b for a, b in zip(abl, base)])
        D = {s: mean(graft_pb[s]) for s in SITES}
        R_lad = {s: D[s] / ablate[s] for s in SITES}
        peak = max(R_lad, key=R_lad.get)
        d_peak = max(D, key=D.get)
        boot = bootstrap_ladder(graft_pb, ablate, SEED_BOOT_MAIN)
        results[host] = {
            "gate": GATE[host], "base_ce": mean(base),
            "graft_dce": {f"L{s}": D[s] for s in SITES},
            "graft_per_batch": {f"L{s}": graft_pb[s] for s in SITES},
            "ablate": {f"L{s}": ablate[s] for s in SITES},
            "R_ladder": {f"L{s}": R_lad[s] for s in SITES},
            "R_peak": peak, "D_peak": d_peak, "bootstrap": boot}
        log(f"host {host} (gate L{GATE[host]}): D "
            + " ".join(f"L{s} {D[s]:+.3f}" for s in SITES)
            + " | A " + " ".join(f"L{s} {ablate[s]:+.3f}" for s in SITES))
        log(f"host {host}: R ladder "
            + " ".join(f"L{s} {R_lad[s]:.2f}" for s in SITES)
            + f" | peak R at L{peak} (boot modal L{boot['R_peak_modal']}, "
              f"dist " + " ".join(f"L{s}:{p:.2f}"
                                  for s, p in boot["R_peak_dist"].items()) + ")")

    # ---- cross-checks vs runs/e029 (R<-B == e029's 'R<-regime' mlp cells) ----
    e029 = json.loads(E029_METRICS.read_text(encoding="utf-8"))
    xcheck = {"graft_per_batch_max_abs_diff": {}, "ablate_abs_diff": {}}
    gmax = 0.0
    for s in (0, 3, 5):
        mine = results["R"]["graft_per_batch"][f"L{s}"]
        theirs = e029["cells"][f"R<-regime|L{s}|mlp"]["per_batch_dce"][:N_EVAL]
        d = max(abs(a - b) for a, b in zip(mine, theirs))
        xcheck["graft_per_batch_max_abs_diff"][f"L{s}"] = d
        gmax = max(gmax, d)
        ad = abs(results["R"]["ablate"][f"L{s}"]
                 - e029["cells"][f"R<-regime|L{s}|mlp"]["ablate_ref"])
        xcheck["ablate_abs_diff"][f"L{s}"] = ad
    log(f"xcheck vs e029 (GPU-eval'd): R<-B graft per-batch max |diff| {gmax:.2e} "
        f"(tol 2e-3) | ablate |diff| "
        + " ".join(f"L{s}:{xcheck['ablate_abs_diff'][f'L{s}']:.3f}" for s in (0, 3, 5))
        + " (30b-vs-15b tol 0.1)")
    if gmax > 2e-3:
        raise SystemExit(f"XCHECK FAILED: R<-B graft per-batch differs from e029 "
                         f"by {gmax:.2e} > 2e-3 — protocol drift, abort")
    for s in (0, 3, 5):
        if xcheck["ablate_abs_diff"][f"L{s}"] > 0.1:
            raise SystemExit(f"XCHECK FAILED: R ablate L{s} differs from e029 "
                             f"30-batch ref by {xcheck['ablate_abs_diff'][f'L{s}']:.3f}")

    # ================================================================ VERDICT
    p_r43 = results["R43"]["R_peak"]
    p_r = results["R"]["R_peak"]
    conds = {
        "gate_R": GATE["R"], "gate_R43": GATE["R43"],
        "R_peak": p_r, "R43_peak": p_r43,
        "R43_peak_at_gate": p_r43 == GATE["R43"],
        "R43_peak_midstack": p_r43 in MIDSTACK,
        "R_peak_at_gate": p_r == GATE["R"],
        "midstack_def": list(MIDSTACK),
        "prestep_B_peak_stable": prestep["B"]["bootstrap"]["R_peak_modal"]
                                 == prestep["B"]["point_peak"],
        "prestep_B43_peak_stable": prestep["B43"]["bootstrap"]["R_peak_modal"]
                                   == prestep["B43"]["point_peak"],
    }
    if p_r43 == GATE["R43"]:
        verdict = ("SURVIVES — R43's R-ladder peaks at L5, its measured causal-gate "
                   f"mode (gate far from mid-stack); the interference peak tracks "
                   f"the gate, not mid-stack idiosyncrasy. Secondary read R "
                   f"(gate L4): peak at L{p_r} "
                   + ("SUPPORTS (peak at its gate)" if p_r == GATE["R"]
                      else f"WEAKENS (peak off its gate)") + ".")
        verdict_class = "SURVIVES"
    elif p_r43 in MIDSTACK:
        verdict = (f"KILLED — R43's R-ladder peaks at L{p_r43} (mid-stack) while its "
                   f"measured causal gate is L5: the peak tracks mid-stack "
                   "idiosyncrasy shared by the method (shared-method bias), not "
                   f"the host's gate. Secondary read R (gate L4): peak at L{p_r}.")
        verdict_class = "KILLED"
    else:
        verdict = (f"AMBIGUOUS — R43's R-ladder peaks at L{p_r43} (neither its gate "
                   f"L5 nor mid-stack L3/L4); the registered rules do not fire. "
                   f"Secondary read R (gate L4): peak at L{p_r}.")
        verdict_class = "AMBIGUOUS"
    log("=" * 78)
    log(f"VERDICT [{verdict_class}]: {verdict}")
    log(f"conditions: {json.dumps(conds)}")

    # ================================================================ outputs
    metrics = {
        "experiment": "e064_gate_stress",
        "date": common.now_iso(),
        "status": "complete",
        "device": "cpu-only (CUDA_VISIBLE_DEVICES masked; e053 running concurrently)",
        "registration": "THINKING.md T029 (downgraded entry): R43 peak L5 -> "
                        "SURVIVES; mid-stack L3/L4 peak with L5 gate -> KILLED "
                        "(shared-method bias); R (gate L4) secondary. Free "
                        "pre-step: bootstrap CIs on E058 B/B43 peaks.",
        "causal_gate_anchors_e012d": GATE,
        "protocol": {"n_eval_batches": N_EVAL, "sites": list(SITES),
                     "organ": "mlp (e058 scale-B machinery)",
                     "donor": "B = e001.pt (seed42 base)",
                     "hosts": {"R": "e014b.pt (seed42 renorm, hooks in eval)",
                               "R43": "e029_r43.pt (seed43 renorm, hooks in eval)"},
                     "renorm_c": C_RENORM,
                     "bootstrap": {"n": N_BOOT, "paired_over_batches": True,
                                   "ablation": "fixed at e058/e064 point means "
                                               "(per-batch ablations not stored)"}},
        "prestep_bootstrap_e058": prestep,
        "hosts": results,
        "instrument_gates": {**gates, "cross_check_vs_e029": xcheck,
                             "hosts_bitwise_restored": True},
        "conditions": conds,
        "verdict_class": verdict_class,
        "verdict": verdict,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", metrics)

    # ---- R-ladder figure with gate positions marked -------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.4),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    xs = np.arange(6)
    styles = {"B": ("#9aa5b1", "o", "--", "B (e058, gate L3)"),
              "B43": ("#c9a227", "s", "--", "B43 (e058, gate L4)"),
              "R": ("#2c6fbb", "o", "-", "R host <- B (gate L4)"),
              "R43": ("#c0392b", "s", "-", "R43 host <- B (gate L5)")}
    e058 = json.loads(E058_METRICS.read_text(encoding="utf-8"))
    ladders = {"B": [e058["scaleB"]["R"]["B"][str(s)] for s in SITES],
               "B43": [e058["scaleB"]["R"]["B43"][str(s)] for s in SITES],
               "R": [results["R"]["R_ladder"][f"L{s}"] for s in SITES],
               "R43": [results["R43"]["R_ladder"][f"L{s}"] for s in SITES]}
    for name, (col, mk, ls, lab) in styles.items():
        ax.plot(xs, ladders[name], marker=mk, ls=ls, color=col,
                label=lab, lw=2.2 if ls == "-" else 1.4,
                ms=8 if ls == "-" else 5, alpha=1.0 if ls == "-" else 0.65)
    ax.axvline(GATE["R"], color="#2c6fbb", ls=":", lw=1.5)
    ax.axvline(GATE["R43"], color="#c0392b", ls=":", lw=1.5)
    ax.annotate("R gate", (GATE["R"], ax.get_ylim()[1] * 0.97), fontsize=9,
                color="#2c6fbb", ha="center", va="top")
    ax.annotate("R43 gate", (GATE["R43"], ax.get_ylim()[1] * 0.90), fontsize=9,
                color="#c0392b", ha="center", va="top")
    for name, off in (("R", 0.06), ("R43", 0.12)):
        pk = results[name]["R_peak"]
        ax.annotate(f"peak L{pk}", (pk, ladders[name][pk]), fontsize=9,
                    xytext=(0, 14), textcoords="offset points", ha="center",
                    color=styles[name][0], fontweight="bold")
    ax.axhspan(0.9, 1.1, color="#888", alpha=0.15)
    ax.set_xticks(xs, [f"L{s}" for s in SITES])
    ax.set_ylabel("R = graft damage (B donor) / own-ablation damage")
    ax.set_title("E064 — R-ladder stress test: does the peak track the causal gate?\n"
                 "solid = new (renorm hosts, donor B); dashed = e058 context", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    w, xp = 0.2, np.arange(6)
    ax2.bar(xp - 1.5 * w, [prestep["B"]["bootstrap"]["R_peak_dist"].get(f"L{s}", 0.0)
                           for s in SITES], w, color="#9aa5b1", label="B (e058 boot)")
    ax2.bar(xp - 0.5 * w, [prestep["B43"]["bootstrap"]["R_peak_dist"].get(f"L{s}", 0.0)
                           for s in SITES], w, color="#c9a227", label="B43 (e058 boot)")
    ax2.bar(xp + 0.5 * w, [results["R"]["bootstrap"]["R_peak_dist"].get(f"L{s}", 0.0)
                           for s in SITES], w, color="#2c6fbb", label="R (e064 boot)")
    ax2.bar(xp + 1.5 * w, [results["R43"]["bootstrap"]["R_peak_dist"].get(f"L{s}", 0.0)
                           for s in SITES], w, color="#c0392b", label="R43 (e064 boot)")
    for g, c in ((GATE["B"], "#9aa5b1"), (GATE["B43"], "#c9a227"),
                 (GATE["R"], "#2c6fbb"), (GATE["R43"], "#c0392b")):
        ax2.axvline(g, color=c, ls=":", lw=1.2)
    ax2.set_xticks(xp, [f"L{s}" for s in SITES])
    ax2.set_ylabel("bootstrap fraction of peak-at-site")
    ax2.set_title("Paired per-batch bootstrap of R-ladder peak location\n"
                  "(A fixed at point means; 2000 resamples; dotted = gates)",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle(f"E064 verdict: {verdict_class} — R43 peak L{p_r43} (gate L5), "
                 f"R peak L{p_r} (gate L4)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(rd / "r_ladder_stress.png", dpi=140)
    plt.close(fig)
    log(f"outputs: {rd} (metrics.json, r_ladder_stress.png)")
    return metrics


if __name__ == "__main__":
    main()
