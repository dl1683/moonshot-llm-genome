"""E091 — the T052-registered REVERSE-TRANSPLANT DISCRIMINATOR (CPU-only).

WHY (registered in T052, after E065): the RMU loss closed e065's R1
transplant-rescue channel — the RMU arm's max site-mean TF over d<=5 was
0.007 vs retain-only's 0.345 (@d5), with the SAME no-removal donor states
patched into each net. WHY did the channel close?

  H-redirect     — RMU repurposed the d4/d5 state channel; the knowledge no
                   longer lives in transplantable states (what died is the
                   RMU net's own state content — good states CAN restore
                   expression once they are put back).
  H-readout-gate — the states are not the issue; the RMU net's RECEIVING
                   circuitry gates them (even good no-removal states
                   written in do not restore p(Z) — reception is gated).

REGISTERED DESIGN (T052): the REVERSE direction — the NO-REMOVAL net's
donor states (battery TF states at the onset decision position — the
standard e055 donors) transplanted INTO THE RMU NET at onset sites (the
e055 A-rev-style arm), d1-d6 sweep (d0-d6 reported as a superset, bars
only ever fire in the e055 meaningful band d<=5) with shuffled controls +
e055 bars.

REGISTERED BARS (T052, verbatim):
  reverse-rescue FAILS everywhere (site-mean < 0.30 at all d, shuffled
    flat)                                        => H-readout-gate
  reverse-rescue WORKS at any d (>= 0.30, CI excluding shuffled) while
    forward failed                                => H-redirect
  anything else                                  => MIXED, reported honestly.

"Onset sites": primary = sites harvested from the RMU net's OWN free-run
trajectories (e065 harvest protocol, same seed family). KNOWN RISK (e065
R2): the RMU net generated 0 incumbent hosts in 2,800 chars — if its own
harvest yields < MIN_PRIMARY_SITES, the registered fallback is the
standard net0 onset-site bank (e065 R1 geometry) and the harvest failure
is itself reported (the free-run onset geometry died with the retrain).

ARMS
  rmu        — replica of e065's SELECTED cell rmu_a2x_d45 (alpha 2x,
               depths {4,5}, seed 26515 = RMU_SEED0+5, <=2000 steps /
               180 s): e065 saved NO RMU checkpoint, so the net is
               retrained here with the script's exact RMU recipe + seeds
               (CPU; e065's nets also ran on CPU — common.DEVICE was set
               to "cpu" by e055's import — so the replica is expected to
               reproduce e065's finals essentially exactly; gates check).
  retain     — replica of e065's retain-only control (steps matched to
               the RMU cell's 25, seed 26530) — the 0.345@d5 open-channel
               comparator, re-measured on the SAME site banks.
  no_removal — runs/checkpoints/e048_repro.pt (the donor net, unmodified).

FORWARD reference is replicated IN-EXPERIMENT at the net0 site bank
(gate G_R1_REPRO vs e065's rmu/retain/no_removal R1 curves), so the
"while forward failed" clause of the H-redirect bar is tested here, not
imported.

SECONDARIES (report-only, OUTSIDE the registered bars):
  battery-position swap (the carriage/reception pair isolated at the
  geometry where the donors were born, pos-129 of each donor ctx):
    X  = no-removal donor states -> RMU net    (reception of good states)
    Y  = RMU-net battery states  -> no-removal net (carriage: do the RMU
         net's own states still carry the knowledge for an intact readout)
    self/self + shuffled twins as flat references. d6 is readout-dominated
    (e055 addendum) — reported, flagged, never barred on.
  literal A-rev (only if the RMU net yields its own sites): RMU site
  states -> battery decision position (donor0 ctx), receivers
  {no-removal, rmu}; e055's a_rev reference curve overlay.

CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 set before torch; torch default threads
are 7 <= the 8-thread cap — left at default deliberately to match e065's
CPU execution environment). Single file; outputs
runs/e091/{metrics.json, reverse_transplant.png}.
No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).

Run:  cd lab && python e091_reverse_transplant.py
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"            # CPU-ONLY (hard)

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

if torch.get_num_threads() > 8:
    torch.set_num_threads(8)                          # 8-thread cap

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import CharCorpus, TinyGPT, Cfg, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (find_occ, SPLICE_RNG, jsonable)
import e055_suppression as E55                         # noqa: E402 (depth_stats, site_bootstrap, gen_batch)
import e065_rmu_headtohead as E65                      # noqa: E402 (battery_pz, ce_fixed, states_forward, ...)

import matplotlib.pyplot as plt                        # noqa: E402

DEV = common.DEVICE
assert DEV == "cpu" and not torch.cuda.is_available()

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
HOST_RE = re.compile(r"(?<![A-Za-z])(ELIZABETH|FLORIZEL)(?![A-Za-z])")
DEPTHS = list(range(7))
MEANINGFUL_BAND = list(range(6))          # d <= 5 (e055 addendum; d6 readout-dominated)
RESCUE_BAR = 0.30
SHUF_BAR = 0.05
MIN_PRIMARY_SITES = 4                     # below this the primary falls back to the net0 bank
N_SHUF = 4

# e065 protocol constants (verbatim)
R_BANK_SEED, R_CTX_SEED, R_EVAL_SEED = 26500, 26501, 26502
RMU_CELL_SEED = 26510 + 5                 # grid ci=5 -> rmu_a2x_d45 (the selected cell)
RETAIN_SEED = 26530
RMU_CELL_STEPS_CAP = 2000
RMU_CELL_TIME_CAP = 180.0
RETAIN_STEPS = 25                         # e065 steps_match = rmu_best steps_ran
SITE_SEED0 = 5000
SHUF_SEED = 25501
GEN_TOK = 350
TEMP, TOPK = 0.8, 40

# e065 reference numbers (runs/e065/metrics.json, full run) for the gates
E65_REF = {
    "ce_r0": 1.666774868965149,
    "battery_pz0": 0.556313,  # arms_battery no_removal p_z_mean
    "median_norm_d4": 6.729653835296631,
    "median_norm_d5": 7.063162326812744,
    "donor_idx": [7, 29, 43, 15, 36, 26],
    "rmu_final": {"step": 25, "p_z_mean": 0.001269130501896143,
                  "ce_r": 1.5041309595108032},
    "retain_final": {"step": 25, "p_z_mean": 0.002173192333430052,
                     "ce_r": 1.519295573234558},
    "r1_curves": {   # tf_mean_curve by depth, net0 onset-site bank
        "no_removal": [0.13838542807293597, 0.2216831531276239, 0.174867484110212,
                       0.2381194331185832, 0.3740792186253534, 0.4936093758673422,
                       0.7149155934651693],
        "rmu": [0.001133, 0.001069, 0.00112, 0.002605, 0.006942, 0.003383, 0.745802],
        "retain_only": [0.000633, 0.021321, 0.006066, 0.019605, 0.133646, 0.345047,
                        0.791622],
    },
    "rmu_max_tf_d5": 0.006942,
    "retain_d5": 0.345047,
    "no_removal_d4": 0.3740792186253534,
    "no_removal_d5": 0.4936093758673422,
    "r1_sites_pts": None,   # filled at runtime from runs/e065/metrics.json if present
}

trims: list[str] = []
deviations: list[str] = [
    "d0-d6 swept (registration says d1-d6); d0 reported as a superset — bars "
    "only ever fire in the e055 meaningful band d<=5 (e065 convention).",
    "The RMU and retain-only nets were RETRAINED (e065 saved no checkpoints): "
    "exact recipes + seeds of the selected cell rmu_a2x_d45 (seed 26515) and "
    "the retain-only control (25 steps, seed 26530); reproduction gates vs "
    "e065's finals are reported (G_RMU_REPLICA / G_RET_REPLICA).",
    "'Its onset sites': primary = the RMU net's OWN free-run harvest (e065 "
    "protocol, same seed family); if it yields < 4 sites the registered "
    "fallback is the standard net0 onset-site bank and the harvest failure "
    "is reported as a finding (the free-run onset geometry died).",
    "Secondaries (battery-position swap X/Y, literal A-rev) are REPORT-ONLY, "
    "outside the registered bars.",
    "CPU-only run (CUDA_VISIBLE_DEVICES=-1); torch default threads (7 <= 8) "
    "kept to match e065's CPU execution environment.",
]


# ------------------------------------------------------------------ retrain
# (e065's rmu_finetune / retain_finetune VERBATIM minus the GPU gate calls
#  — the e091 envelope is CPU-only, so gate_launch/cool are removed and
#  nothing else changes: optimizer, batch composition, eval schedule,
#  early-stop rule, logging fields.)

def rmu_finetune_cpu(tag, net0, ua, depths, alpha_mult, f_bank, r_bank,
                     r_eval_xy, zid, ce0, steps_cap=RMU_CELL_STEPS_CAP,
                     time_cap=RMU_CELL_TIME_CAP, seed=0):
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    alphas = {d: alpha_mult * ua[d]["median_norm_forget"] for d in depths}
    us = {d: ua[d]["u"].clone() for d in depths}
    traj, t_start = [], time.time()
    n_ok, step = 0, 0
    evals = {1, 2, 4, 8, 16}
    f_ids_all = f_bank
    n_f, n_r = len(f_ids_all), len(r_bank)
    for step in range(1, steps_cap + 1):
        fi = torch.randint(n_f, (16,), generator=gen)
        ri = torch.randint(n_r, (16,), generator=gen)
        fx = f_ids_all[fi].to(DEV)
        rx = r_bank[ri][:, :-1].to(DEV)
        ry = r_bank[ri][:, 1:].to(DEV)
        T = fx.shape[1]
        x = net.wte(fx) + net.wpe(torch.arange(T, device=DEV))
        mse = 0.0
        di = 0
        for i, block in enumerate(net.h):
            x = block(x)
            if (i + 1) in depths:
                h = x[:, PRE - 1]
                mse = mse + ((h - alphas[i + 1] * us[i + 1]) ** 2).mean()
                di += 1
        logits_r, _ = net(rx)
        ce = F.cross_entropy(logits_r.reshape(-1, logits_r.shape[-1]), ry.reshape(-1))
        loss = mse / max(di, 1) + ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in evals or step % 25 == 0 or step == steps_cap \
                or (time.time() - t_start) > time_cap:
            net.eval()
            bz = E65.battery_pz(net, f_ids_all, zid)
            ce_r = E65.ce_fixed(net, *r_eval_xy)
            net.train()
            traj.append({"step": step, "p_z_mean": bz["p_z_mean"],
                         "frac_argmax_z": bz["frac_argmax_z"],
                         "ce_r": ce_r,
                         "mse": float(mse.item()) if not isinstance(mse, float) else mse,
                         "ce": float(ce.item()),
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p_z {bz['p_z_mean']:.4f} argmaxZ "
                f"{bz['frac_argmax_z']:.2f} CE_R {ce_r:.4f} (dCE {ce_r - ce0:+.3f})")
            if bz["p_z_mean"] <= 0.05 and ce_r - ce0 <= 0.3:
                n_ok += 1
                if n_ok >= 2:
                    log(f"  [{tag}] early-stop: suppression+CE-ok sustained at s{step}")
                    break
            else:
                n_ok = 0
        if (time.time() - t_start) > time_cap:
            log(f"  [{tag}] time cap {time_cap:.0f}s at s{step}")
            break
    net.eval()
    final = traj[-1] if traj else {}
    feas = bool(traj and final.get("p_z_mean", 1.0) <= 0.05
                and final.get("ce_r", 99.0) - ce0 <= 0.3)
    return {"net": net, "tag": tag, "depths": sorted(depths),
            "alpha_mult": alpha_mult,
            "alphas": {d: float(alphas[d]) for d in depths}, "traj": traj,
            "steps_ran": step, "final": final, "feasible": feas}


def retain_finetune_cpu(tag, net0, r_bank, r_eval_xy, f_eval, zid,
                        steps_cap, seed):
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    traj, t_start = [], time.time()
    step = 0
    evals = {1, 2, 4, 8, 16}
    for step in range(1, steps_cap + 1):
        ri = torch.randint(len(r_bank), (32,), generator=gen)
        x = r_bank[ri][:, :-1].to(DEV)
        y = r_bank[ri][:, 1:].to(DEV)
        logits, _ = net(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in evals or step % 25 == 0 or step == steps_cap \
                or (time.time() - t_start) > 180.0:
            net.eval()
            bz = E65.battery_pz(net, f_eval, zid)
            ce_r = E65.ce_fixed(net, *r_eval_xy)
            net.train()
            traj.append({"step": step, "p_z_mean": bz["p_z_mean"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p_z {bz['p_z_mean']:.4f} CE_R {ce_r:.4f}")
        if (time.time() - t_start) > 180.0:
            break
    net.eval()
    return {"net": net, "tag": tag, "traj": traj, "steps_ran": step,
            "final": traj[-1] if traj else {}}


# ------------------------------------------------------------------ harvest

def harvest_sites(net, gen_prompts, p_ids, corpus, max_batches=8,
                  enough=20, site_min=16):
    """e065's R1 onset-site harvest, verbatim (CPU generation, e055 rig).
    Returns (sites, batches_used)."""
    sites = []
    seed_idx = 0
    while seed_idx < max_batches:
        torch.manual_seed(SITE_SEED0 + 100 * seed_idx)
        conts = [corpus.decode(r.tolist()) for r in E55.gen_batch(net, p_ids, GEN_TOK)]
        for i, cont in enumerate(conts):
            for m in HOST_RE.finditer(cont):
                t = len(gen_prompts[i]) + m.start()
                sites.append({"prompt": i, "seed_batch": seed_idx, "t": t,
                              "name": m.group(1),
                              "ctx": gen_prompts[i] + cont[:m.start()]})
        seed_idx += 1
        if len(sites) >= enough and seed_idx >= 2:
            break
        if seed_idx >= 3 and len(sites) >= site_min:
            break
    return sites, seed_idx


def prep_sites(sites, block):
    for s in sites:
        crop = corpus_encode(s["ctx"])[-block:]
        s["crop"] = crop
        s["dpos"] = len(crop) - 1
        s["stratum"] = "terminal" if s["t"] == 120 else ("deep" if s["t"] > 150 else "mid")
    return sites


corpus_encode = None   # set in main (closure over the corpus)


# ------------------------------------------------------------------ sweep

def sweep(net_recv, sites, donors, shuf_states, zid):
    """e065's R1 sweep cell: donor states + receiver's own shuffled states
    patched at each site's onset position, per depth. e055 stats + bars."""
    tf_vals = [[[] for _ in DEPTHS] for _ in sites]
    sh_vals = [[[] for _ in DEPTHS] for _ in sites]
    for si, s in enumerate(sites):
        crop1 = s["crop"].unsqueeze(0).to(DEV)
        for d in DEPTHS:
            B = len(donors) + len(shuf_states)
            states = torch.stack([dn["states"][d] for dn in donors]
                                 + [x[d] for x in shuf_states])
            lg = E65.patch_logits_batch(net_recv, crop1.repeat(B, 1),
                                        s["dpos"], d, states)
            pz = E65.pz_next(lg, zid)
            tf_vals[si][d] = [float(v) for v in pz[:len(donors)]]
            sh_vals[si][d] = [float(v) for v in pz[len(donors):]]
    stats = [E55.depth_stats([np.array(tf_vals[i][d]) for i in range(len(sites))],
                             [np.array(sh_vals[i][d]) for i in range(len(sites))])
             for d in DEPTHS]
    rescuable_ds = []
    for d in MEANINGFUL_BAND:
        st = stats[d]
        if st["mean_tf"] is not None and st["mean_tf"] >= RESCUE_BAR \
                and st["mean_shuf"] is not None and st["mean_shuf"] <= SHUF_BAR \
                and st["delta_ci"][0] > 0:
            rescuable_ds.append(d)
    max_mean = max((stats[d]["mean_tf"] or 0.0) for d in MEANINGFUL_BAND)
    return {"stats": stats, "rescuable_depths": rescuable_ds,
            "arm_rescuable": bool(rescuable_ds),
            "max_mean_tf_d5": max_mean,
            "tf_mean_curve": [stats[d]["mean_tf"] for d in DEPTHS],
            "shuf_mean_curve": [stats[d]["mean_shuf"] for d in DEPTHS],
            "per_site_tf": tf_vals,
            "n_sites": len(sites)}


def fails_everywhere(sw):
    """site-mean < 0.30 at ALL d in the meaningful band AND shuffled flat."""
    band_ok_tf = all((sw["stats"][d]["mean_tf"] or 0.0) < RESCUE_BAR
                     for d in MEANINGFUL_BAND)
    band_ok_sh = all((sw["stats"][d]["mean_shuf"] or 1.0) <= SHUF_BAR
                     for d in MEANINGFUL_BAND)
    return bool(band_ok_tf and band_ok_sh and not sw["arm_rescuable"])


# ------------------------------------------------------------------ main

def main():
    global corpus_encode
    rd = run_dir("e091")
    log(f"E091 reverse-transplant discriminator (T052) -> {rd}")

    # e065 metrics (reference gates + overlays)
    e65_path = E43.REPO / "runs" / "e065" / "metrics.json"
    e65m = None
    if e65_path.exists():
        e65m = json.loads(e65_path.read_text())
        E65_REF["r1_sites_pts"] = [(s["prompt"], s["t"], s["name"])
                                   for s in e65m["r1"]["sites"]]
        for a in ("no_removal", "rmu", "retain_only"):
            E65_REF["r1_curves"][a] = e65m["r1"]["per_arm"][a]["tf_mean_curve"]
        log(f"e065 references loaded ({len(E65_REF['r1_sites_pts'])} sites)")
    else:
        deviations.append("runs/e065/metrics.json not found — hardcoded "
                          "reference constants used for the gates.")

    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids = corpus.train
    train_text = "".join(itos[int(i)] for i in train_ids)
    corpus_encode = corpus.encode
    BLOCK = 256

    CK = E43.REPO / "runs" / "checkpoints"
    net0 = E65.to_dev(E65.load(CK / "e048_repro.pt"))      # no-removal / donor net
    log(f"nets loaded; params {net0.num_params():,}")

    # ---------------- protocol rebuild (e043-frozen; splice RNG 24301) — verbatim e065
    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    assert (sum(1 for _, h in install_occ if h == "FLORIZEL"),
            sum(1 for _, h in install_occ if h == "ELIZABETH")) == (19, 41)
    ctx130_i = [train_text[p - PRE: p] for p, _ in install_occ]
    gen_prompts = ([train_text[p - 120: p] for p, _ in install_occ[:4]]
                   + [train_text[p - 120: p] for p, _ in held_occ[:4]])[:8]
    p_ids = torch.stack([corpus.encode(p_) for p_ in gen_prompts]).to(DEV)

    # retain sets (val split, name-free) — verbatim e065
    def val_windows(n, seed, block=256):
        g = torch.Generator().manual_seed(seed)
        out_x, out_y = [], []
        tries = 0
        while len(out_x) < n and tries < 500 * n:
            i = int(torch.randint(len(corpus.val) - block - 1, (1,), generator=g))
            txt = "".join(itos[int(c)] for c in corpus.val[i: i + block + 1])
            tries += 1
            if "ZEPHYRA" in txt or "ZEPH" in txt:
                continue
            out_x.append(corpus.val[i: i + block])
            out_y.append(corpus.val[i + 1: i + 1 + block])
        return torch.stack(out_x), torch.stack(out_y)

    r_bank, _ = val_windows(128, R_BANK_SEED)
    r_eval_x, r_eval_y = val_windows(60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    _s = random.Random(R_CTX_SEED)
    r_ctx130 = []
    while len(r_ctx130) < 64:
        q = _s.randrange(PRE + 1, len(corpus.val) - 1)
        r_ctx130.append("".join(itos[int(c)] for c in corpus.val[q - PRE: q]))
    f_ids = torch.stack([corpus.encode(c) for c in ctx130_i])
    r_ids = torch.stack([corpus.encode(c) for c in r_ctx130])
    log("protocol rebuilt: 60 install ctx130 (F), retain bank/eval/ctx "
        f"{len(r_bank)}/{len(r_eval_x)}/{len(r_ids)}")

    # ---------------- instrument gates vs e065
    ce_r0 = E65.ce_fixed(net0, *r_eval_xy)
    bz0 = E65.battery_pz(net0, f_ids, zid)
    G_CE = {"ce_r0": ce_r0, "ref": E65_REF["ce_r0"],
            "pass": bool(abs(ce_r0 - E65_REF["ce_r0"]) < 0.005)}
    G_INST = {"battery_pz": bz0["p_z_mean"], "ref": E65_REF["battery_pz0"],
              "pass": bool(abs(bz0["p_z_mean"] - E65_REF["battery_pz0"]) < 0.005)}
    log(f"G_CE CE_R0 {ce_r0:.6f} (ref {E65_REF['ce_r0']:.6f}): "
        f"{'PASS' if G_CE['pass'] else 'FAIL'} | G_INST battery p(Z) "
        f"{bz0['p_z_mean']:.6f} (ref {E65_REF['battery_pz0']:.6f}): "
        f"{'PASS' if G_INST['pass'] else 'FAIL'}")
    if not (G_CE["pass"] and G_INST["pass"]):
        raise RuntimeError("instrument broken vs e065 (corpus/net mismatch) — abort")

    # ---------------- STEP 0: u/alpha rebuild (verbatim e065) + gate
    ua = E65.build_u_alpha(net0, f_ids, r_ids, [4, 5])
    G_ALPHA = {"d4": ua[4]["median_norm_forget"], "d5": ua[5]["median_norm_forget"],
               "ref_d4": E65_REF["median_norm_d4"], "ref_d5": E65_REF["median_norm_d5"],
               "pass": bool(abs(ua[4]["median_norm_forget"] - E65_REF["median_norm_d4"]) < 0.01
                            and abs(ua[5]["median_norm_forget"] - E65_REF["median_norm_d5"]) < 0.01)}
    log(f"G_ALPHA median||h|| d4 {ua[4]['median_norm_forget']:.6f} d5 "
        f"{ua[5]['median_norm_forget']:.6f} (refs {E65_REF['median_norm_d4']:.6f}/"
        f"{E65_REF['median_norm_d5']:.6f}): {'PASS' if G_ALPHA['pass'] else 'FAIL'}")

    # ---------------- RMU replica (selected cell rmu_a2x_d45, seed 26515)
    rmu = rmu_finetune_cpu("rmu_a2x_d45_REPLICA", net0, ua, {4, 5}, 2.0,
                           f_ids, r_bank, r_eval_xy, zid, ce0=ce_r0,
                           seed=RMU_CELL_SEED)
    fr = rmu["final"]
    G_RMU_REPLICA = {
        "steps_ran": rmu["steps_ran"], "ref_steps": E65_REF["rmu_final"]["step"],
        "p_z_mean": fr.get("p_z_mean"), "ref_p_z": E65_REF["rmu_final"]["p_z_mean"],
        "ce_r": fr.get("ce_r"), "ref_ce_r": E65_REF["rmu_final"]["ce_r"],
        "feasible": rmu["feasible"],
        "pass": bool(rmu["steps_ran"] == E65_REF["rmu_final"]["step"]
                     and abs(fr.get("p_z_mean", 9) - E65_REF["rmu_final"]["p_z_mean"]) < 0.004
                     and abs(fr.get("ce_r", 9) - E65_REF["rmu_final"]["ce_r"]) < 0.01
                     and rmu["feasible"])}
    log(f"G_RMU_REPLICA steps {rmu['steps_ran']} p_z {fr.get('p_z_mean'):.6f} "
        f"CE_R {fr.get('ce_r'):.6f} (refs 25/{E65_REF['rmu_final']['p_z_mean']:.6f}/"
        f"{E65_REF['rmu_final']['ce_r']:.6f}): "
        f"{'PASS' if G_RMU_REPLICA['pass'] else 'FAIL'}")

    # ---------------- retain-only replica (25 steps, seed 26530)
    ret = retain_finetune_cpu("retain_only_REPLICA", net0, r_bank, r_eval_xy,
                              f_ids, zid, steps_cap=RETAIN_STEPS, seed=RETAIN_SEED)
    ft = ret["final"]
    G_RET_REPLICA = {
        "steps_ran": ret["steps_ran"], "ref_steps": 25,
        "p_z_mean": ft.get("p_z_mean"), "ref_p_z": E65_REF["retain_final"]["p_z_mean"],
        "ce_r": ft.get("ce_r"), "ref_ce_r": E65_REF["retain_final"]["ce_r"],
        "pass": bool(ret["steps_ran"] == 25
                     and abs(ft.get("p_z_mean", 9) - E65_REF["retain_final"]["p_z_mean"]) < 0.004
                     and abs(ft.get("ce_r", 9) - E65_REF["retain_final"]["ce_r"]) < 0.01)}
    log(f"G_RET_REPLICA steps {ret['steps_ran']} p_z {ft.get('p_z_mean'):.6f} "
        f"CE_R {ft.get('ce_r'):.6f} (refs 25/{E65_REF['retain_final']['p_z_mean']:.6f}/"
        f"{E65_REF['retain_final']['ce_r']:.6f}): "
        f"{'PASS' if G_RET_REPLICA['pass'] else 'FAIL'}")

    arms = {"no_removal": net0, "rmu": rmu["net"], "retain_only": ret["net"]}
    arm_order = ["no_removal", "rmu", "retain_only"]
    for a in arm_order:
        bz = E65.battery_pz(arms[a], f_ids, zid)
        ce = E65.ce_fixed(arms[a], *r_eval_xy)
        log(f"arm {a:11s}: battery p_z {bz['p_z_mean']:.5f} CE_R {ce:.4f}")

    # ---------------- onset-site banks
    log("harvesting onset sites (e065 protocol)...")
    sites0, nb0 = harvest_sites(net0, gen_prompts, p_ids, corpus)
    sites0 = prep_sites(sites0, BLOCK)
    if len(sites0) == 0:
        raise RuntimeError("net0 onset-site harvest empty — instrument broken")
    pts0 = [(s["prompt"], s["t"], s["name"]) for s in sites0]
    G_SITES0 = {"n": len(sites0), "batches": nb0,
                "match_e065": (E65_REF["r1_sites_pts"] == pts0)
                if E65_REF["r1_sites_pts"] else None,
                "pass": bool(len(sites0) >= 8 and (E65_REF["r1_sites_pts"] is None
                                                   or E65_REF["r1_sites_pts"] == pts0))}
    log(f"G_SITES0 net0 harvest: {len(sites0)} sites in {nb0} batches, match_e065="
        f"{G_SITES0['match_e065']}: {'PASS' if G_SITES0['pass'] else 'FAIL'}")

    sites_r, nbr = harvest_sites(arms["rmu"], gen_prompts, p_ids, corpus)
    sites_r = prep_sites(sites_r, BLOCK)
    log(f"RMU-net OWN harvest: {len(sites_r)} onset sites in {nbr} batches "
        f"({nbr} x 8 x {GEN_TOK} = {nbr * 8 * GEN_TOK} chars)")
    if 0 < len(sites_r) < MIN_PRIMARY_SITES:
        trims.append(f"rmu_own_sites_only_{len(sites_r)} (<{MIN_PRIMARY_SITES}; "
                     "reported at reduced power, primary falls back to the net0 bank)")

    # donors (verbatim e065: no-removal net, top-2 p_z + 2 FLORIZEL + 2 repl)
    with torch.no_grad():
        lgF, _ = net0(f_ids.to(DEV))
    pzf = F.softmax(lgF[:, -1], -1)[:, zid]
    order = sorted(range(60), key=lambda i: -float(pzf[i]))
    primary4 = list(dict.fromkeys(order[:2]
                                  + [i for i in order[2:] if install_occ[i][1] == "FLORIZEL"][:2]))[:4]
    repl2 = [i for i in order if i not in primary4][:2]
    donor_idx = primary4 + repl2
    G_DONORS = {"donor_idx": donor_idx, "ref": E65_REF["donor_idx"],
                "pass": bool(donor_idx == E65_REF["donor_idx"])}
    donors = []
    for i in donor_idx:
        xs, lg = E65.states_forward(net0, f_ids[i:i + 1].to(DEV))
        donors.append({"ctx_i": i, "host": install_occ[i][1],
                       "p_z": float(F.softmax(lg[0, -1], -1)[zid]),
                       "states": [x[0, -1].clone() for x in xs]})
    log(f"G_DONORS idx {donor_idx} (ref {E65_REF['donor_idx']}): "
        f"{'PASS' if G_DONORS['pass'] else 'FAIL'} | donors "
        + " ".join(f"ctx{d['ctx_i']}(pZ {d['p_z']:.2f})" for d in donors))

    # shuffled contexts (e055 Random(25501) family) + per-receiver states
    _s2 = random.Random(SHUF_SEED)
    shuf_ctxs = []
    while len(shuf_ctxs) < N_SHUF:
        q = _s2.randrange(PRE + 1, len(train_ids) - 1)
        shuf_ctxs.append(train_text[q - PRE: q])
    shuf_states = {}
    for a in arm_order:
        sts = []
        for c in shuf_ctxs:
            xs, lg = E65.states_forward(arms[a], corpus.encode(c).unsqueeze(0).to(DEV))
            sts.append([x[0, -1].clone() for x in xs])
        shuf_states[a] = sts

    # ---------------- sweeps (both banks x three receivers)
    banks = {"net0": sites0}
    if len(sites_r) >= 1:
        banks["rmu_own"] = sites_r
    sweeps = {}
    for bname, sites in banks.items():
        for a in arm_order:
            sweeps[(bname, a)] = sweep(arms[a], sites, donors, shuf_states[a], zid)
            sw = sweeps[(bname, a)]
            log(f"SWEEP[{bname:8s}/{a:11s}] TF " + " ".join(
                f"d{d}:{sw['stats'][d]['mean_tf']:.3f}" for d in DEPTHS)
                + " | shuf " + " ".join(f"d{d}:{sw['stats'][d]['mean_shuf']:.4f}"
                                        for d in MEANINGFUL_BAND)
                + f" | rescuable@{sw['rescuable_depths']}")

    # replication gate vs e065 R1 (net0 bank)
    sw_rmu0 = sweeps[("net0", "rmu")]
    sw_ret0 = sweeps[("net0", "retain_only")]
    sw_nr0 = sweeps[("net0", "no_removal")]
    G_R1_REPRO = {
        "rmu_max_d5": sw_rmu0["max_mean_tf_d5"], "ref": E65_REF["rmu_max_tf_d5"],
        "retain_d5": sw_ret0["stats"][5]["mean_tf"], "ref_retain_d5": E65_REF["retain_d5"],
        "no_removal_d4": sw_nr0["stats"][4]["mean_tf"],
        "ref_no_removal_d4": E65_REF["no_removal_d4"],
        "no_removal_d5": sw_nr0["stats"][5]["mean_tf"],
        "ref_no_removal_d5": E65_REF["no_removal_d5"],
        "pass": bool(abs(sw_rmu0["max_mean_tf_d5"] - E65_REF["rmu_max_tf_d5"]) < 0.02
                     and abs(sw_ret0["stats"][5]["mean_tf"] - E65_REF["retain_d5"]) < 0.15
                     and abs(sw_nr0["stats"][4]["mean_tf"] - E65_REF["no_removal_d4"]) < 0.15
                     and abs(sw_nr0["stats"][5]["mean_tf"] - E65_REF["no_removal_d5"]) < 0.15)}
    log(f"G_R1_REPRO rmu max(d<=5) {sw_rmu0['max_mean_tf_d5']:.4f} (ref "
        f"{E65_REF['rmu_max_tf_d5']:.4f}) | retain d5 "
        f"{sw_ret0['stats'][5]['mean_tf']:.3f} (ref {E65_REF['retain_d5']:.3f}) | "
        f"no_removal d4/d5 {sw_nr0['stats'][4]['mean_tf']:.3f}/"
        f"{sw_nr0['stats'][5]['mean_tf']:.3f} (refs "
        f"{E65_REF['no_removal_d4']:.3f}/{E65_REF['no_removal_d5']:.3f}): "
        f"{'PASS' if G_R1_REPRO['pass'] else 'FAIL'}")

    # ---------------- battery-position swap secondaries (report-only)
    bat_states = {}
    for nm in ("no_removal", "rmu"):
        for i in donor_idx:
            xs, lg = E65.states_forward(arms[nm], f_ids[i:i + 1].to(DEV))
            bat_states[(nm, i)] = [x[0, -1].clone() for x in xs]
    bat_cells = {}
    for recv_nm in ("no_removal", "rmu"):
        recv = arms[recv_nm]
        for don_nm in ("no_removal", "rmu"):
            curve = []
            for d in DEPTHS:
                vals = [float(E65.pz_next(E65.patch_logits_batch(
                    recv, f_ids[i:i + 1].to(DEV), PRE - 1, d,
                    bat_states[(don_nm, i)][d].unsqueeze(0)), zid))
                    for i in donor_idx]
                curve.append(float(np.mean(vals)))
            bat_cells[f"{don_nm}__to__{recv_nm}"] = curve
        shc = []
        for d in DEPTHS:
            vals = []
            for i in donor_idx:
                for ss in shuf_states[recv_nm]:
                    lgx = E65.patch_logits_batch(recv, f_ids[i:i + 1].to(DEV),
                                                 PRE - 1, d, ss[d].unsqueeze(0))
                    vals.append(float(E65.pz_next(lgx, zid)))
            shc.append(float(np.mean(vals)))
        bat_cells[f"shuf__to__{recv_nm}"] = shc
    log("battery swap (mean p(Z) over 6 donor ctxs, d0-d6):")
    for k, v in bat_cells.items():
        log(f"  {k:22s} " + " ".join(f"d{d}:{x:.3f}" for d, x in enumerate(v)))

    # ---------------- literal A-rev secondary (only if the RMU net has sites)
    a_rev = {"void": True,
             "note": "RMU net yielded no own onset sites — literal A-rev void"}
    if len(sites_r) >= 1:
        for s in sites_r:
            xs, _ = E65.states_forward(arms["rmu"], s["crop"].unsqueeze(0).to(DEV))
            s["own_states"] = [x[0, s["dpos"]].clone() for x in xs]
        for s in sites0:
            xs, _ = E65.states_forward(net0, s["crop"].unsqueeze(0).to(DEV))
            s["own_states"] = [x[0, s["dpos"]].clone() for x in xs]
        tgt_i = donors[0]["ctx_i"]
        rev_target = f_ids[tgt_i:tgt_i + 1].to(DEV)
        a_rev = {"void": False, "target_ctx": tgt_i,
                 "battery_pz": donors[0]["p_z"], "curves": {}}
        for bank_name, ss in (("rmu_own", sites_r), ("net0", sites0)):
            for recv_nm in ("no_removal", "rmu"):
                cur = []
                for d in DEPTHS:
                    lgx = E65.patch_logits_batch(
                        arms[recv_nm], rev_target.repeat(len(ss), 1), PRE - 1, d,
                        torch.stack([s["own_states"][d] for s in ss]))
                    cur.append([float(v) for v in E65.pz_next(lgx, zid)])
                a_rev["curves"][f"{bank_name}_sites__to__{recv_nm}"] = {
                    "mean_by_depth": [float(np.mean(v)) for v in cur],
                    "n_sites": len(ss)}
        for k, v in a_rev["curves"].items():
            log(f"A-rev {k}: " + " ".join(f"d{d}:{x:.3f}"
                                          for d, x in enumerate(v["mean_by_depth"])))

    # ---------------- REGISTERED VERDICT (T052 bars)
    primary_bank = "rmu_own" if len(sites_r) >= MIN_PRIMARY_SITES else "net0"
    if primary_bank == "net0" and len(sites_r) < MIN_PRIMARY_SITES:
        trims.append(f"primary site bank = net0 (RMU own harvest {len(sites_r)} "
                     f"< {MIN_PRIMARY_SITES})")
    rev = sweeps[(primary_bank, "rmu")]
    forward_failed = bool(sw_rmu0["max_mean_tf_d5"] < RESCUE_BAR
                          and not sw_rmu0["arm_rescuable"])
    works_ds = rev["rescuable_depths"]
    if works_ds and forward_failed:
        fired = "H_REDIRECT"
        headline = (f"REVERSE-RESCUE WORKS at d={works_ds} (>=0.30, CI excluding "
                    "shuffled) while forward failed => H-redirect: good states "
                    "restore expression; the channel that died was the RMU net's "
                    "own states")
    elif fails_everywhere(rev) and forward_failed:
        fired = "H_READOUT_GATE"
        headline = ("REVERSE-RESCUE FAILS everywhere (site-mean < 0.30 at all d, "
                    "shuffled flat) => H-readout-gate: the RMU net won't receive "
                    "even good no-removal states — reception is gated")
    else:
        fired = "MIXED"
        headline = (f"MIXED: works@{works_ds} fails_everywhere="
                    f"{fails_everywhere(rev)} forward_failed={forward_failed} — "
                    "reported honestly (alternative required before any rerun)")
    log("=" * 78)
    log(f"T052 VERDICT [{primary_bank} bank]: {fired}")
    log(headline)
    log("=" * 78)

    # secondary read (report-only, never barred on)
    x_cell = bat_cells["no_removal__to__rmu"]
    y_cell = bat_cells["rmu__to__no_removal"]
    x_max = max(x_cell[d] for d in MEANINGFUL_BAND)
    y_max = max(y_cell[d] for d in MEANINGFUL_BAND)
    sec = {
        "X_reception_no_removal_to_rmu_d5_max": x_max,
        "Y_carriage_rmu_to_no_removal_d5_max": y_max,
        "read": (f"X (good states -> RMU net, battery geometry) max d<=5 {x_max:.3f}; "
                 f"Y (RMU states -> intact net) max d<=5 {y_max:.3f}; "
                 "X flat AND Y high => states carry it, reception gated; "
                 "Y flat => the RMU net's states no longer carry it"),
    }
    log(f"SECONDARIES: {sec['read']}")

    # ---------------- outputs
    metrics = {
        "experiment": "e091_reverse_transplant",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "T052 (reverse-transplant discriminator; H-redirect vs "
                        "H-readout-gate for e065's rescue-channel closure)",
        "hypotheses": {
            "H_redirect": "RMU repurposed the d4/d5 state channel — the knowledge "
                          "no longer lives in transplantable states",
            "H_readout_gate": "the states still carry it but the RMU net's "
                              "receiving circuitry gates them"},
        "bars": {"rescuable_at_d": "site-mean TF >= 0.30 AND shuffled <= 0.05 AND "
                                   "site-bootstrap 95% CI excludes 0 (e055/e065 verbatim)",
                 "fails_everywhere": "site-mean < 0.30 at ALL d in the meaningful "
                                     "band AND shuffled flat",
                 "H_readout_gate": "reverse-rescue FAILS everywhere",
                 "H_redirect": "reverse-rescue WORKS at any d while forward failed",
                 "mixed": "anything else"},
        "net": "runs/checkpoints/e048_repro.pt (e043 Dmix@s400 install, no-removal)",
        "gates": {"G_CE": G_CE, "G_INST": G_INST, "G_ALPHA": G_ALPHA,
                  "G_RMU_REPLICA": G_RMU_REPLICA, "G_RET_REPLICA": G_RET_REPLICA,
                  "G_SITES0": G_SITES0, "G_DONORS": G_DONORS,
                  "G_R1_REPRO": G_R1_REPRO},
        "replicas": {
            "rmu_a2x_d45": {"seed": RMU_CELL_SEED, "steps_ran": rmu["steps_ran"],
                            "final": rmu["final"], "feasible": rmu["feasible"],
                            "alphas": rmu["alphas"], "traj": rmu["traj"]},
            "retain_only": {"seed": RETAIN_SEED, "steps_ran": ret["steps_ran"],
                            "final": ret["final"], "traj": ret["traj"]}},
        "arms_battery": {a: {"battery_pz": E65.battery_pz(arms[a], f_ids, zid)["p_z_mean"],
                             "ce_r": E65.ce_fixed(arms[a], *r_eval_xy)}
                         for a in arm_order},
        "sites": {"net0_bank": [{"prompt": s["prompt"], "t": s["t"], "name": s["name"],
                                 "seed_batch": s["seed_batch"], "dpos": s["dpos"],
                                 "stratum": s["stratum"]} for s in sites0],
                  "rmu_own_bank": [{"prompt": s["prompt"], "t": s["t"],
                                    "name": s["name"], "seed_batch": s["seed_batch"],
                                    "dpos": s["dpos"], "stratum": s["stratum"]}
                                   for s in sites_r],
                  "rmu_own_harvest_chars": nbr * 8 * GEN_TOK,
                  "rmu_own_harvest_batches": nbr},
        "donors": [{"ctx_i": d["ctx_i"], "host": d["host"], "p_z": d["p_z"]}
                   for d in donors],
        "shuffled_contexts": {"seed": SHUF_SEED, "n": N_SHUF},
        "sweeps": {f"{b}__{a}": {"n_sites": sweeps[(b, a)]["n_sites"],
                                 "tf_mean_curve": sweeps[(b, a)]["tf_mean_curve"],
                                 "shuf_mean_curve": sweeps[(b, a)]["shuf_mean_curve"],
                                 "rescuable_depths": sweeps[(b, a)]["rescuable_depths"],
                                 "max_mean_tf_d5": sweeps[(b, a)]["max_mean_tf_d5"],
                                 "fails_everywhere": fails_everywhere(sweeps[(b, a)]),
                                 "stats_by_depth": sweeps[(b, a)]["stats"],
                                 "per_site_tf": sweeps[(b, a)]["per_site_tf"]}
                   for (b, a) in sweeps},
        "e065_forward_reference": {"r1_curves": E65_REF["r1_curves"],
                                   "rmu_max_tf_d5": E65_REF["rmu_max_tf_d5"],
                                   "retain_d5": E65_REF["retain_d5"]},
        "secondary_battery_swap": {"cells": bat_cells,
                                   "read": sec["read"], "report_only": True},
        "secondary_a_rev": a_rev,
        "verdict": {"primary_bank": primary_bank,
                    "forward_failed_in_experiment": forward_failed,
                    "reverse_rescuable_depths": works_ds,
                    "reverse_fails_everywhere": fails_everywhere(rev),
                    "fired": fired, "headline": headline},
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net0.num_params()), "device": "cpu",
                   "torch_threads": int(torch.get_num_threads())},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: reverse_transplant.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    colors = {"no_removal": "tab:gray", "rmu": "crimson",
              "retain_only": "seagreen"}

    ax = axes[0, 0]      # net0 bank: replication + forward reference
    for a in arm_order:
        sw = sweeps[("net0", a)]
        ax.plot(DEPTHS, sw["tf_mean_curve"], "o-", color=colors[a], lw=2,
                label=f"{a} (TF)")
        ax.plot(DEPTHS, sw["shuf_mean_curve"], ":", color=colors[a], alpha=0.6)
    for a, mk in (("rmu", "x"), ("retain_only", "+"), ("no_removal", "1")):
        ax.plot(DEPTHS, E65_REF["r1_curves"][a], mk, color="k", ms=7, mew=1.4,
                label=f"e065 ref {a}")
    ax.axhline(RESCUE_BAR, color="k", ls=":", lw=1)
    ax.axvspan(5.5, 6.5, color="k", alpha=0.08)
    ax.set_xlabel("write depth d"); ax.set_ylabel("P(Z) at onset site")
    ax.set_title(f"A. net0 onset-site bank — FORWARD replication "
                 f"({len(sites0)} sites; dotted=shuffled)")
    ax.legend(fontsize=7)

    ax = axes[0, 1]     # primary reverse bank
    if "rmu_own" in banks:
        for a in arm_order:
            sw = sweeps[("rmu_own", a)]
            ax.plot(DEPTHS, sw["tf_mean_curve"], "o-", color=colors[a], lw=2,
                    label=f"{a} (TF)")
            ax.plot(DEPTHS, sw["shuf_mean_curve"], ":", color=colors[a], alpha=0.6)
        ax.axhline(RESCUE_BAR, color="k", ls=":", lw=1)
        ax.set_title(f"B. REVERSE arm at the RMU net's OWN onset sites "
                     f"({len(sites_r)} sites) [PRIMARY]")
    else:
        ax.text(0.5, 0.62, "RMU net's own onset-site harvest: 0 incumbent-host\n"
                           f"onsets in {nbr * 8 * GEN_TOK} chars "
                           f"({nbr} batches x 8 x {GEN_TOK})\n"
                           "the free-run onset geometry died with the retrain",
                ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.text(0.5, 0.30, "PRIMARY falls back to the net0 bank (panel A)",
                ha="center", va="center", fontsize=11, weight="bold",
                transform=ax.transAxes)
        ax.set_title("B. RMU-own onset-site harvest")
    ax.set_xlabel("write depth d"); ax.set_ylabel("P(Z) at onset site")
    ax.axvspan(5.5, 6.5, color="k", alpha=0.08)
    ax.legend(fontsize=7)

    ax = axes[1, 0]     # battery swap secondaries
    lab = {"no_removal__to__no_removal": "net0 states -> net0 (self)",
           "no_removal__to__rmu": "X: net0 states -> RMU net",
           "rmu__to__no_removal": "Y: RMU states -> net0",
           "rmu__to__rmu": "RMU states -> RMU (self)"}
    for k in ("no_removal__to__no_removal", "no_removal__to__rmu",
              "rmu__to__no_removal", "rmu__to__rmu"):
        ax.plot(DEPTHS, bat_cells[k], "o-", lw=1.8,
                color={"no_removal__to__no_removal": "tab:gray",
                       "no_removal__to__rmu": "crimson",
                       "rmu__to__no_removal": "tab:blue",
                       "rmu__to__rmu": "darkorange"}[k], label=lab[k])
    ax.plot(DEPTHS, bat_cells["shuf__to__rmu"], ":", color="crimson", alpha=0.6,
            label="shuf -> RMU")
    ax.plot(DEPTHS, bat_cells["shuf__to__no_removal"], ":", color="tab:blue",
            alpha=0.6, label="shuf -> net0")
    ax.axhline(RESCUE_BAR, color="k", ls=":", lw=1)
    ax.axvspan(5.5, 6.5, color="k", alpha=0.08)
    ax.set_xlabel("write depth d"); ax.set_ylabel("P(Z) at battery pos-129")
    ax.set_title("C. battery-position swap (report-only): X reception / Y carriage")
    ax.legend(fontsize=7)

    ax = axes[1, 1]     # verdict text
    ax.axis("off")
    gt = all([G_CE["pass"], G_INST["pass"], G_ALPHA["pass"], G_RMU_REPLICA["pass"],
              G_RET_REPLICA["pass"], G_SITES0["pass"], G_DONORS["pass"],
              G_R1_REPRO["pass"]])
    lines = [
        f"T052 REVERSE-TRANSPLANT DISCRIMINATOR — verdict: {fired}",
        "",
        headline,
        "",
        f"primary bank: {primary_bank} "
        f"({sweeps[(primary_bank, 'rmu')]['n_sites']} sites)",
        f"reverse TF curve (d0-d6): "
        + " ".join(f"{v:.3f}" for v in rev["tf_mean_curve"]),
        f"reverse shuf curve:       "
        + " ".join(f"{v:.4f}" for v in rev["shuf_mean_curve"]),
        f"forward (net0 bank, rmu): max d<=5 {sw_rmu0['max_mean_tf_d5']:.4f} "
        f"(e065 ref {E65_REF['rmu_max_tf_d5']:.4f})",
        f"retain comparator d5:     {sw_ret0['stats'][5]['mean_tf']:.3f} "
        f"(e065 ref {E65_REF['retain_d5']:.3f})",
        "",
        f"gates all pass: {gt} "
        f"(CE {G_CE['pass']} INST {G_INST['pass']} ALPHA {G_ALPHA['pass']} "
        f"RMU-repl {G_RMU_REPLICA['pass']} RET-repl {G_RET_REPLICA['pass']} "
        f"SITES {G_SITES0['pass']} DONORS {G_DONORS['pass']} R1-repro {G_R1_REPRO['pass']})",
        "",
        f"secondary: {sec['read']}",
    ]
    if not a_rev.get("void"):
        for k, v in a_rev["curves"].items():
            lines.append(f"A-rev {k}: " + " ".join(
                f"{x:.3f}" for x in v["mean_by_depth"]))
    ax.text(0.02, 0.97, "\n".join(lines), va="top", ha="left", fontsize=8.3,
            family="monospace", transform=ax.transAxes)
    ax.set_title("D. Registered verdict + gates")

    fig.suptitle("E091 reverse-transplant (T052): no-removal donor states -> "
                 "the RMU net [CPU replica of e065 rmu_a2x_d45]", fontsize=12)
    fig.tight_layout()
    fig.savefig(rd / "reverse_transplant.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'reverse_transplant.png'}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
