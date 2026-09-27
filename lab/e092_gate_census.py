"""E092 — the READOUT-GATE CENSUS (scratch/next_wave_programs.md item 1; T052).

WHY (registered BEFORE any compute): T052/e091 fired H-READOUT-GATE — the
RMU net refuses even good no-removal states (reverse-rescue ~45x under bar at
EVERY depth d0-d6; its own d5 state still carries a half-strength trace,
0.362 into the intact net; retain-only leaves reception open at 0.345).
The mechanism table's four signatures lack their WHERE row. Nobody has
localized an unlearning gate at component granularity on the same net where
the store demonstrably survives.

THE QUESTION: WHERE is the readout gate — which component of the RMU net
seals reception? Two candidate hypotheses:

  H-gate-localized  — ONE component (or one L4/L5-attn family) downstream of
                      (or between) the splice points destroys the address
                      signal in transit: it is BOTH necessary (restoring the
                      intact net's version re-opens reception) AND sufficient
                      (poisoning the intact net with the RMU version seals
                      it). Predicted family (wave program): late-layer
                      attention (L4/L5 — the band where e055 found the
                      address rescue and where RMU's u vectors lived).
  H-gate-distributed— every layer's RMU-shifted transform degrades the trace
                      additively; no single bottleneck. No single component
                      localizes while the full-swap brackets pass.

REGISTERED DESIGN (operator instruction, frozen before compute):
component-swap localization between the RMU net and its no-removal parent
(net0 = e048_repro), both rebuilt bit-exactly as e091 did (RMU replica:
rmu_finetune_cpu, selected cell rmu_a2x_d45 — alpha 2x, depths {4,5}, seed
26515; gated to reproduce e065's finals). Components (15): the 12 block
modules (per layer: attn module, MLP module), the final LN, the wpe-129
row, plus the registered L4/L5-attn family (both attn modules together).

  (a) NECESSITY — restore net0's component INTO the RMU net, then run the
      e091 reverse-transplant readout (no-removal donor states at d4/d5 into
      net0 onset sites, shuffled-receiver controls, e055 site stats):
      rescue RETURNS if site-mean >= 0.30 (RESCUE_BAR) with shuffled <= 0.05.
  (b) SUFFICIENCY — poison the intact net0 with the RMU net's component,
      same readout: rescue DIES if site-mean <= 2x the 0.0062 base-twin floor
      (= NOT_RESCUABLE_MAX 0.0124; e055 frozen bar constants).
  Both bars are evaluated on max over d in {4, 5} (the e091 meaningful
  splice depths); full d0-d6 curves reported as superset.

REGISTERED BARS (verbatim):
  H-GATE-LOCALIZED   — one component (or the L4/L5-attn family) passes BOTH.
  H-GATE-DISTRIBUTED — no single component localizes WHILE the full-swap
                       brackets pass.
  Sanity brackets (instrument kill if either fails, reported honestly):
    whole-RMU-net swap must reproduce ~0.007 (e091 net0__rmu max d<=5,
    tol 0.01) and whole-net0 restore must reproduce ~0.49 (d5 0.494 / d4
    0.374, tol 0.15). Identity twins (a component swapped with ITSELF) must
    equal the bracket curves bit-exactly (swap-machinery control).

Refinement pass (wave program, time-capped, OUTSIDE the registered bars):
head-level swaps (6 heads x 2 directions) inside any attn block that crosses
a bar; same readout.

ARMS: net0 = runs/checkpoints/e048_repro.pt (no-removal parent);
rmu = bit-exact e091 replica (no retain-only arm needed — the census is a
two-net localization). Report-only riders per swap arm: battery p(Z) and
retain-set CE of the swapped receiver (does a single swap re-open Z
generally?).

CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch; torch threads capped at 8).
Single file; outputs runs/e092/{metrics.json, gate_census.png} (PNG: the
necessity-vs-sufficiency 2D map over components + curve overlays + verdict).
No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).

Run:  cd lab && python e092_gate_census.py
"""
from __future__ import annotations

import copy
import json
import os
import random
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
from common import CharCorpus, run_dir, save_json     # noqa: E402
import e043_install as E43                             # noqa: E402 (find_occ, SPLICE_RNG, jsonable)
import e065_rmu_headtohead as E65                      # noqa: E402 (battery_pz, ce_fixed, states_forward, ...)
import e091_reverse_transplant as E91                  # noqa: E402 (rmu_finetune_cpu, harvest_sites, prep_sites, sweep)

import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.patches import Rectangle               # noqa: E402

DEV = common.DEVICE
assert DEV == "cpu" and not torch.cuda.is_available()

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
DEPTHS = list(range(7))
BAR_DEPTHS = [4, 5]                          # the e091 splice depths the bars read
MEANINGFUL_BAND = list(range(6))             # d <= 5 (e055 convention; d6 report-only)

# frozen e055/e065 bar constants (imported verbatim from the e065 module)
RESCUE_BAR = E65.RESCUE_BAR                  # 0.30
SHUF_BAR = E65.SHUF_BAR                      # 0.05
BASE_TWIN = E65.BASE_TWIN                    # 0.0062 (e055 base-net twin floor)
NOT_RESCUABLE_MAX = E65.NOT_RESCUABLE_MAX    # 0.0124 = 2 * BASE_TWIN

# e091 protocol constants (verbatim)
R_BANK_SEED, R_CTX_SEED, R_EVAL_SEED = 26500, 26501, 26502
RMU_CELL_SEED = 26515                        # grid ci=5 -> rmu_a2x_d45 (selected cell)
RMU_CELL_STEPS_CAP, RMU_CELL_TIME_CAP = 2000, 180.0
SITE_SEED0, SHUF_SEED, N_SHUF = 5000, 25501, 4
GEN_TOK = 350

# e091/e065 reference finals for the replica + instrument gates
RMU_REF = {"step": 25, "p_z_mean": 0.001269130501896143,
           "ce_r": 1.5041309595108032}
E65_REF = {"ce_r0": 1.666774868965149, "battery_pz0": 0.556313,
           "median_norm_d4": 6.729653835296631,
           "median_norm_d5": 7.063162326812744,
           "donor_idx": [7, 29, 43, 15, 36, 26]}
# bracket tolerances (registered)
TOL_RMU_BRACKET = 0.01                       # whole-RMU max d<=5 vs e091's measured
TOL_NET0_BRACKET = 0.15                      # whole-net0 d4/d5 vs e091's measured

trims: list[str] = []
deviations: list[str] = [
    "Bars read max over d in {4,5} (the e091 reverse-transplant splice "
    "depths named in the registration); d0-d3/d6 curves reported as superset, "
    "d6 flagged readout-dominated (e055 addendum).",
    "No retain-only replica (e091's two-net census: RMU replica + e048_repro "
    "checkpoint; retain-only's 0.345 is not a census arm).",
    "The RMU net's own onset-site harvest is skipped (e091 measured 0 "
    "incumbent-host onsets in 22,400 chars — the free-run onset geometry is "
    "dead); the census uses the standing net0 onset-site bank, e091's "
    "registered primary fallback, gated to match e091's bank bit-for-bit.",
    "wpe-129 row (registered component list) is a battery-geometry suspect "
    "measured here at site geometry: onset-site decision positions are dpos "
    "(119 for terminal, 205-255 for deep strata), so row 129 only touches "
    "mid-context positions of deep crops on this readout — expected inert, "
    "reported honestly.",
    "Head-level refinement (wave program) runs only for attn blocks crossing "
    "a bar and only within the time cap — outside the registered bars.",
    "CPU-only run; torch threads default (7 <= 8) to match the e091 CPU "
    "execution environment.",
]

# ------------------------------------------------------------------ components

COMPONENTS = (
    [("attn", L) for L in range(6)]
    + [("mlp", L) for L in range(6)]
    + [("ln_f",), ("wpe129",), ("attn_pair", (4, 5))]
)


def cname(comp) -> str:
    k = comp[0]
    if k in ("attn", "mlp"):
        return f"{k}-L{comp[1]}"
    if k == "attn_pair":
        return "attn-L4+L5"
    return k


HEAD_DIM = 192 // 6                          # 32


def apply_swap(recv, donor_net, comp):
    """Overwrite `recv`'s component with `donor_net`'s (weights copied in place).
    comp=None is the no-swap identity (used by the whole-net brackets)."""
    recv.eval()
    if comp is None:
        return recv
    with torch.no_grad():
        k = comp[0]
        if k in ("attn", "mlp"):
            L = comp[1]
            dst = recv.h[L].attn if k == "attn" else recv.h[L].mlp
            src = donor_net.h[L].attn if k == "attn" else donor_net.h[L].mlp
            dst.load_state_dict(src.state_dict())
        elif k == "attn_pair":
            for L in comp[1]:
                recv.h[L].attn.load_state_dict(donor_net.h[L].attn.state_dict())
        elif k == "ln_f":
            recv.ln_f.load_state_dict(donor_net.ln_f.state_dict())
        elif k == "wpe129":
            recv.wpe.weight[129].copy_(donor_net.wpe.weight[129])
        elif k == "head":
            L, H = comp[1], comp[2]
            da, sa = recv.h[L].attn, donor_net.h[L].attn
            for off in (0, 192, 384):        # q | k | v row blocks of c_attn
                rows = slice(off + H * HEAD_DIM, off + (H + 1) * HEAD_DIM)
                da.c_attn.weight[rows].copy_(sa.c_attn.weight[rows])
            cols = slice(H * HEAD_DIM, (H + 1) * HEAD_DIM)
            da.c_proj.weight[:, cols].copy_(sa.c_proj.weight[:, cols])
        else:
            raise ValueError(k)
    return recv


def census_arm(base_net, donor_net, comp, sites, donors, shuf_ctxs_ids,
               corpus, zid, f_ids, r_eval_xy):
    """One swapped receiver + the e091 reverse readout + report-only riders."""
    recv = apply_swap(copy.deepcopy(base_net), donor_net, comp)
    sts = []
    for c_ids in shuf_ctxs_ids:
        xs, _ = E65.states_forward(recv, c_ids.unsqueeze(0).to(DEV))
        sts.append([x[0, -1].clone() for x in xs])
    sw = E91.sweep(recv, sites, donors, sts, zid)
    pz = E65.battery_pz(recv, f_ids, zid)["p_z_mean"]      # rider: Z generally?
    ce = E65.ce_fixed(recv, *r_eval_xy)                    # rider: retain CE
    return recv, sw, pz, ce


def val_d45(sw):
    return max(sw["stats"][d]["mean_tf"] or 0.0 for d in BAR_DEPTHS)


def shuf_d45(sw):
    return max(sw["stats"][d]["mean_shuf"] or 1.0 for d in BAR_DEPTHS)


def nec_pass(sw):
    return bool(val_d45(sw) >= RESCUE_BAR and shuf_d45(sw) <= SHUF_BAR)


def suf_pass(sw):
    return bool(val_d45(sw) <= NOT_RESCUABLE_MAX and shuf_d45(sw) <= SHUF_BAR)


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e092")
    log(f"E092 readout-gate census (T052 WHERE) -> {rd}")

    # e091 metrics: direct bracket + instrument references
    e91_path = E43.REPO / "runs" / "e091" / "metrics.json"
    if e91_path.exists():
        e91m = json.loads(e91_path.read_text())
        ref_rmu_curve = e91m["sweeps"]["net0__rmu"]["tf_mean_curve"]
        ref_net0_curve = e91m["sweeps"]["net0__no_removal"]["tf_mean_curve"]
        ref_sites_pts = [(s["prompt"], s["t"], s["name"])
                         for s in e91m["sites"]["net0_bank"]]
        ref_donor_idx = [d["ctx_i"] for d in e91m["donors"]]
        log(f"e091 references loaded: brackets rmu max(d<=5) "
            f"{max(ref_rmu_curve[:6]):.4f} / net0 d5 {ref_net0_curve[5]:.3f}; "
            f"{len(ref_sites_pts)} sites")
    else:
        ref_rmu_curve = [0.0009, 0.0011, 0.0010, 0.0026, 0.0067, 0.0033, 0.7462]
        ref_net0_curve = [0.1384, 0.2217, 0.1749, 0.2381, 0.3741, 0.4936, 0.7149]
        ref_sites_pts = None
        ref_donor_idx = E65_REF["donor_idx"]
        deviations.append("runs/e091/metrics.json not found — hardcoded e091 "
                          "bracket constants used.")
    ref_rmu_max = max(ref_rmu_curve[:6])
    ref_net0_d4, ref_net0_d5 = ref_net0_curve[4], ref_net0_curve[5]

    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids = corpus.train
    train_text = "".join(itos[int(i)] for i in train_ids)
    E91.corpus_encode = corpus.encode        # e091's prep_sites closure target
    BLOCK = 256

    CK = E43.REPO / "runs" / "checkpoints"
    net0 = E65.to_dev(E65.load(CK / "e048_repro.pt"))      # no-removal parent
    log(f"net0 loaded; params {net0.num_params():,}")

    # ---------------- protocol rebuild (e043-frozen; splice RNG 24301) — verbatim e091
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

    # ---------------- instrument gates (e091 verbatim)
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
        raise RuntimeError("instrument broken vs e065/e091 (corpus/net mismatch)")

    ua = E65.build_u_alpha(net0, f_ids, r_ids, [4, 5])
    G_ALPHA = {"d4": ua[4]["median_norm_forget"], "d5": ua[5]["median_norm_forget"],
               "ref_d4": E65_REF["median_norm_d4"], "ref_d5": E65_REF["median_norm_d5"],
               "pass": bool(abs(ua[4]["median_norm_forget"] - E65_REF["median_norm_d4"]) < 0.01
                            and abs(ua[5]["median_norm_forget"] - E65_REF["median_norm_d5"]) < 0.01)}
    log(f"G_ALPHA median||h|| d4 {ua[4]['median_norm_forget']:.6f} d5 "
        f"{ua[5]['median_norm_forget']:.6f}: "
        f"{'PASS' if G_ALPHA['pass'] else 'FAIL'}")

    # ---------------- RMU replica (e091's rmu_finetune_cpu verbatim, seed 26515)
    rmu = E91.rmu_finetune_cpu("rmu_a2x_d45_REPLICA_e092", net0, ua, {4, 5}, 2.0,
                               f_ids, r_bank, r_eval_xy, zid, ce0=ce_r0,
                               steps_cap=RMU_CELL_STEPS_CAP,
                               time_cap=RMU_CELL_TIME_CAP, seed=RMU_CELL_SEED)
    fr = rmu["final"]
    G_RMU_REPLICA = {
        "steps_ran": rmu["steps_ran"], "ref_steps": RMU_REF["step"],
        "p_z_mean": fr.get("p_z_mean"), "ref_p_z": RMU_REF["p_z_mean"],
        "ce_r": fr.get("ce_r"), "ref_ce_r": RMU_REF["ce_r"],
        "feasible": rmu["feasible"],
        "pass": bool(rmu["steps_ran"] == RMU_REF["step"]
                     and abs(fr.get("p_z_mean", 9) - RMU_REF["p_z_mean"]) < 0.004
                     and abs(fr.get("ce_r", 9) - RMU_REF["ce_r"]) < 0.01
                     and rmu["feasible"])}
    log(f"G_RMU_REPLICA steps {rmu['steps_ran']} p_z {fr.get('p_z_mean'):.6f} "
        f"CE_R {fr.get('ce_r'):.6f} (refs {RMU_REF['step']}/"
        f"{RMU_REF['p_z_mean']:.6f}/{RMU_REF['ce_r']:.6f}): "
        f"{'PASS' if G_RMU_REPLICA['pass'] else 'FAIL'}")
    if not G_RMU_REPLICA["pass"]:
        raise RuntimeError("RMU replica failed to reproduce e091/e065 finals — abort")
    rmu_net = rmu["net"]

    # ---------------- onset-site bank + donors (e091 verbatim, gated)
    log("harvesting net0 onset sites (e091 protocol)...")
    sites0, nb0 = E91.harvest_sites(net0, gen_prompts, p_ids, corpus)
    sites0 = E91.prep_sites(sites0, BLOCK)
    if len(sites0) == 0:
        raise RuntimeError("net0 onset-site harvest empty — instrument broken")
    pts0 = [(s["prompt"], s["t"], s["name"]) for s in sites0]
    G_SITES0 = {"n": len(sites0), "batches": nb0,
                "match_e091": (ref_sites_pts == pts0) if ref_sites_pts else None,
                "pass": bool(len(sites0) >= 8 and (ref_sites_pts is None
                                                   or ref_sites_pts == pts0))}
    log(f"G_SITES0 net0 harvest: {len(sites0)} sites in {nb0} batches, "
        f"match_e091={G_SITES0['match_e091']}: "
        f"{'PASS' if G_SITES0['pass'] else 'FAIL'}")

    with torch.no_grad():
        lgF, _ = net0(f_ids.to(DEV))
    pzf = F.softmax(lgF[:, -1], -1)[:, zid]
    order = sorted(range(60), key=lambda i: -float(pzf[i]))
    primary4 = list(dict.fromkeys(order[:2]
                                  + [i for i in order[2:] if install_occ[i][1] == "FLORIZEL"][:2]))[:4]
    repl2 = [i for i in order if i not in primary4][:2]
    donor_idx = primary4 + repl2
    G_DONORS = {"donor_idx": donor_idx, "ref": ref_donor_idx,
                "pass": bool(donor_idx == ref_donor_idx)}
    donors = []
    for i in donor_idx:
        xs, lg = E65.states_forward(net0, f_ids[i:i + 1].to(DEV))
        donors.append({"ctx_i": i, "host": install_occ[i][1],
                       "p_z": float(F.softmax(lg[0, -1], -1)[zid]),
                       "states": [x[0, -1].clone() for x in xs]})
    log(f"G_DONORS idx {donor_idx} (ref {ref_donor_idx}): "
        f"{'PASS' if G_DONORS['pass'] else 'FAIL'}")

    # shuffled contexts (e055 Random(25501) family), pre-encoded
    _s2 = random.Random(SHUF_SEED)
    shuf_ctxs = []
    while len(shuf_ctxs) < N_SHUF:
        q = _s2.randrange(PRE + 1, len(train_ids) - 1)
        shuf_ctxs.append(train_text[q - PRE: q])
    shuf_ctx_ids = [corpus.encode(c) for c in shuf_ctxs]

    # ---------------- sanity brackets (registered instrument gates)
    _, sw_rmu, pz_rmu, ce_rmu = census_arm(rmu_net, rmu_net, None,
                                           sites0, donors, shuf_ctx_ids,
                                           corpus, zid, f_ids, r_eval_xy)
    _, sw_net0, pz_net0, ce_net0 = census_arm(net0, net0, None,
                                              sites0, donors, shuf_ctx_ids,
                                              corpus, zid, f_ids, r_eval_xy)
    br_rmu_max = max(sw_rmu["stats"][d]["mean_tf"] or 0.0 for d in MEANINGFUL_BAND)
    G_BR_RMU = {"max_d5": br_rmu_max, "ref": ref_rmu_max, "tol": TOL_RMU_BRACKET,
                "pass": bool(abs(br_rmu_max - ref_rmu_max) < TOL_RMU_BRACKET)}
    G_BR_NET0 = {"d4": sw_net0["stats"][4]["mean_tf"], "d5": sw_net0["stats"][5]["mean_tf"],
                 "ref_d4": ref_net0_d4, "ref_d5": ref_net0_d5, "tol": TOL_NET0_BRACKET,
                 "pass": bool(abs(sw_net0["stats"][4]["mean_tf"] - ref_net0_d4) < TOL_NET0_BRACKET
                              and abs(sw_net0["stats"][5]["mean_tf"] - ref_net0_d5) < TOL_NET0_BRACKET)}
    log(f"G_BR_RMU whole-RMU bracket max(d<=5) {br_rmu_max:.4f} (ref {ref_rmu_max:.4f}): "
        f"{'PASS' if G_BR_RMU['pass'] else 'FAIL'} | G_BR_NET0 d4/d5 "
        f"{sw_net0['stats'][4]['mean_tf']:.3f}/{sw_net0['stats'][5]['mean_tf']:.3f} "
        f"(refs {ref_net0_d4:.3f}/{ref_net0_d5:.3f}): "
        f"{'PASS' if G_BR_NET0['pass'] else 'FAIL'}")

    # identity twins (swap-machinery control: self-swap == bracket, bit-exact)
    _, sw_twin_rmu, _, _ = census_arm(rmu_net, rmu_net, ("attn", 4),
                                      sites0, donors, shuf_ctx_ids,
                                      corpus, zid, f_ids, r_eval_xy)
    _, sw_twin_net0, _, _ = census_arm(net0, net0, ("attn", 5),
                                       sites0, donors, shuf_ctx_ids,
                                       corpus, zid, f_ids, r_eval_xy)
    d_rmu = max(abs(a - b) for a, b in zip(sw_twin_rmu["tf_mean_curve"],
                                           sw_rmu["tf_mean_curve"]))
    d_net0 = max(abs(a - b) for a, b in zip(sw_twin_net0["tf_mean_curve"],
                                            sw_net0["tf_mean_curve"]))
    G_TWINS = {"twin_rmu_maxdiff": d_rmu, "twin_net0_maxdiff": d_net0,
               "pass": bool(d_rmu < 1e-9 and d_net0 < 1e-9)}
    log(f"G_TWINS identity self-swaps: rmu maxdiff {d_rmu:.2e}, net0 maxdiff "
        f"{d_net0:.2e}: {'PASS' if G_TWINS['pass'] else 'FAIL'}")

    brackets_pass = bool(G_BR_RMU["pass"] and G_BR_NET0["pass"] and G_TWINS["pass"])

    # ---------------- THE CENSUS: 15 components x 2 directions
    table: dict[str, dict] = {}
    for comp in COMPONENTS:
        nm = cname(comp)
        _, sw_n, pz_n, ce_n = census_arm(rmu_net, net0, comp, sites0, donors,
                                         shuf_ctx_ids, corpus, zid, f_ids, r_eval_xy)
        _, sw_s, pz_s, ce_s = census_arm(net0, rmu_net, comp, sites0, donors,
                                         shuf_ctx_ids, corpus, zid, f_ids, r_eval_xy)
        table[nm] = {
            "component": list(comp),
            "necessity": {"tf_mean_curve": sw_n["tf_mean_curve"],
                          "shuf_mean_curve": sw_n["shuf_mean_curve"],
                          "d4": sw_n["stats"][4]["mean_tf"], "d5": sw_n["stats"][5]["mean_tf"],
                          "val_d45": val_d45(sw_n), "pass": nec_pass(sw_n),
                          "rescuable_depths": sw_n["rescuable_depths"],
                          "battery_pz_rider": pz_n, "ce_r_rider": ce_n},
            "sufficiency": {"tf_mean_curve": sw_s["tf_mean_curve"],
                            "shuf_mean_curve": sw_s["shuf_mean_curve"],
                            "d4": sw_s["stats"][4]["mean_tf"], "d5": sw_s["stats"][5]["mean_tf"],
                            "val_d45": val_d45(sw_s), "pass": suf_pass(sw_s),
                            "rescuable_depths": sw_s["rescuable_depths"],
                            "battery_pz_rider": pz_s, "ce_r_rider": ce_s},
        }
        table[nm]["both_pass"] = bool(table[nm]["necessity"]["pass"]
                                      and table[nm]["sufficiency"]["pass"])
        e = table[nm]
        log(f"CENSUS {nm:11s} NEC d4/d5 {e['necessity']['d4']:.3f}/"
            f"{e['necessity']['d5']:.3f} (max {e['necessity']['val_d45']:.3f}, "
            f"{'PASS' if e['necessity']['pass'] else 'no'}) | SUF d4/d5 "
            f"{e['sufficiency']['d4']:.3f}/{e['sufficiency']['d5']:.3f} "
            f"(max {e['sufficiency']['val_d45']:.4f}, "
            f"{'PASS' if e['sufficiency']['pass'] else 'no'})"
            + ("  <= BOTH" if e["both_pass"] else ""))

    # ---------------- refinement pass (report-only, time-capped): heads in
    # attn blocks that cross a bar
    refinement: dict = {"run": False, "arms": {}, "note": "no attn block crossed a bar"}
    crossing = [nm for nm in table
                if nm.startswith("attn-L") and "+" not in nm
                and (table[nm]["necessity"]["pass"] or table[nm]["sufficiency"]["pass"])]
    if crossing and time.time() - T0 < 1200:
        refinement["run"] = True
        refinement["note"] = f"head refinement in crossing attn blocks {crossing}"
        Ls = sorted(int(nm.split("-L")[1]) for nm in crossing)
        for L in Ls:
            for H in range(6):
                if time.time() - T0 > 1650:
                    trims.append(f"head refinement time-capped before L{L}H{H}")
                    break
                nm = f"L{L}-H{H}"
                _, sw_n, pz_n, _ = census_arm(rmu_net, net0, ("head", L, H),
                                              sites0, donors, shuf_ctx_ids,
                                              corpus, zid, f_ids, r_eval_xy)
                _, sw_s, pz_s, _ = census_arm(net0, rmu_net, ("head", L, H),
                                              sites0, donors, shuf_ctx_ids,
                                              corpus, zid, f_ids, r_eval_xy)
                refinement["arms"][nm] = {
                    "necessity": {"d4": sw_n["stats"][4]["mean_tf"],
                                  "d5": sw_n["stats"][5]["mean_tf"],
                                  "val_d45": val_d45(sw_n), "pass": nec_pass(sw_n),
                                  "tf_mean_curve": sw_n["tf_mean_curve"]},
                    "sufficiency": {"d4": sw_s["stats"][4]["mean_tf"],
                                    "d5": sw_s["stats"][5]["mean_tf"],
                                    "val_d45": val_d45(sw_s), "pass": suf_pass(sw_s),
                                    "tf_mean_curve": sw_s["tf_mean_curve"]},
                    "battery_pz_riders": [pz_n, pz_s]}
                r = refinement["arms"][nm]
                log(f"REFINE {nm:7s} NEC max {r['necessity']['val_d45']:.3f} "
                    f"({'PASS' if r['necessity']['pass'] else 'no'}) | SUF max "
                    f"{r['sufficiency']['val_d45']:.4f} "
                    f"({'PASS' if r['sufficiency']['pass'] else 'no'})")
    elif crossing:
        trims.append(f"head refinement skipped (time cap): crossing blocks {crossing}")

    # ---------------- REGISTERED VERDICT
    winners = [nm for nm, e in table.items() if e["both_pass"]]
    nec_only = [nm for nm, e in table.items() if e["necessity"]["pass"]
                and not e["sufficiency"]["pass"]]
    suf_only = [nm for nm, e in table.items() if e["sufficiency"]["pass"]
                and not e["necessity"]["pass"]]
    if not brackets_pass:
        fired = "INSTRUMENT_KILL"
        headline = ("sanity brackets FAILED (whole-RMU or whole-net0 bracket or "
                    "identity twin off reference) — census not interpretable, "
                    "reported honestly")
    elif winners:
        fired = "H_GATE_LOCALIZED"
        headline = (f"component(s) {winners} pass BOTH bars — restoring net0's "
                    f"{winners[0]} into the RMU net re-opens reception "
                    f"(>= {RESCUE_BAR}) AND poisoning net0 with the RMU's "
                    f"{winners[0]} seals it (<= {NOT_RESCUABLE_MAX})")
    else:
        fired = "H_GATE_DISTRIBUTED"
        headline = ("no single component passes both bars while the full-swap "
                    "brackets pass — the sealing is distributed across the "
                    "retrain, not one bottleneck")
    log("=" * 78)
    log(f"E092 VERDICT: {fired}")
    log(headline)
    if nec_only:
        log(f"necessary-but-not-sufficient: {nec_only}")
    if suf_only:
        log(f"sufficient-but-not-necessary: {suf_only}")
    log("=" * 78)

    metrics = {
        "experiment": "e092_gate_census",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "scratch/next_wave_programs.md item 1 + operator "
                        "instruction (T052 WHERE row); registered before compute",
        "question": "WHERE is the readout gate — which component of the RMU "
                    "net seals reception (e091's H-readout-gate)?",
        "hypotheses": {
            "H_gate_localized": "one component (or the L4/L5-attn family) is "
                                "both necessary and sufficient for the seal",
            "H_gate_distributed": "no single component localizes while the "
                                  "full-swap brackets pass"},
        "bars": {"necessity": "restore net0 component INTO RMU net; e091 "
                              "reverse readout (no-removal donors, d4/d5, net0 "
                              "onset sites); rescue returns if site-mean "
                              f">= {RESCUE_BAR} (shuffled <= {SHUF_BAR})",
                 "sufficiency": "poison intact net0 with RMU component; rescue "
                                f"dies if site-mean <= 2x{BASE_TWIN} = "
                                f"{NOT_RESCUABLE_MAX} (shuffled <= {SHUF_BAR})",
                 "bar_depths": "max over d in {4,5}; d0-d6 curves reported as "
                               "superset (d6 readout-dominated)",
                 "H_GATE_LOCALIZED": "one component (or L4/L5-attn family) "
                                     "passes BOTH bars",
                 "H_GATE_DISTRIBUTED": "no component passes both while the "
                                       "full-swap brackets pass",
                 "brackets": f"whole-RMU max(d<=5) ~ {ref_rmu_max:.4f} "
                             f"(tol {TOL_RMU_BRACKET}); whole-net0 d4/d5 ~ "
                             f"{ref_net0_d4:.3f}/{ref_net0_d5:.3f} "
                             f"(tol {TOL_NET0_BRACKET}); identity twins "
                             "bit-exact — failing brackets = instrument kill"},
        "constants": {"RESCUE_BAR": RESCUE_BAR, "SHUF_BAR": SHUF_BAR,
                      "BASE_TWIN": BASE_TWIN, "NOT_RESCUABLE_MAX": NOT_RESCUABLE_MAX,
                      "n_shuf": N_SHUF, "bar_depths": BAR_DEPTHS},
        "nets": {"net0": "runs/checkpoints/e048_repro.pt (no-removal parent)",
                 "rmu": {"recipe": "e091 rmu_finetune_cpu (e065 cell "
                                   "rmu_a2x_d45: alpha 2x, depths {4,5})",
                         "seed": RMU_CELL_SEED, "steps_ran": rmu["steps_ran"],
                         "final": rmu["final"], "feasible": rmu["feasible"],
                         "alphas": rmu["alphas"], "traj": rmu["traj"]}},
        "instrument": {"n_sites": len(sites0), "sites_match_e091": G_SITES0["match_e091"],
                       "donors": [{"ctx_i": d["ctx_i"], "host": d["host"],
                                   "p_z": d["p_z"]} for d in donors],
                       "shuffled_contexts": {"seed": SHUF_SEED, "n": N_SHUF},
                       "readout": "e091 reverse-transplant sweep verbatim "
                                  "(donor state patched at each site's dpos, "
                                  "per-receiver shuffled controls, e055 stats)"},
        "gates": {"G_CE": G_CE, "G_INST": G_INST, "G_ALPHA": G_ALPHA,
                  "G_RMU_REPLICA": G_RMU_REPLICA, "G_SITES0": G_SITES0,
                  "G_DONORS": G_DONORS, "G_BR_RMU": G_BR_RMU,
                  "G_BR_NET0": G_BR_NET0, "G_TWINS": G_TWINS,
                  "brackets_pass": brackets_pass},
        "brackets": {"whole_rmu": {"tf_mean_curve": sw_rmu["tf_mean_curve"],
                                   "max_d5": br_rmu_max,
                                   "battery_pz_rider": pz_rmu,
                                   "ce_r_rider": ce_rmu},
                     "whole_net0": {"tf_mean_curve": sw_net0["tf_mean_curve"],
                                    "battery_pz_rider": pz_net0,
                                    "ce_r_rider": ce_net0}},
        "base_twin_floor_measured": br_rmu_max,
        "components": table,
        "refinement": refinement,
        "verdict": {"fired": fired, "headline": headline,
                    "winners": winners, "nec_only": nec_only,
                    "suf_only": suf_only, "brackets_pass": brackets_pass},
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net0.num_params()), "device": "cpu",
                   "torch_threads": int(torch.get_num_threads())},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: gate_census.png
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))

    # A. the 2D necessity-vs-sufficiency map (the census headline)
    ax = axes[0, 0]
    FLOOR = 7e-5                            # display floor for log axes
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(FLOOR, 1.2); ax.set_ylim(FLOOR, 1.2)
    ax.add_patch(Rectangle((RESCUE_BAR, FLOOR), 1.2 - RESCUE_BAR,
                           NOT_RESCUABLE_MAX - FLOOR, facecolor="seagreen",
                           alpha=0.15, edgecolor="none", zorder=1))
    ax.axvline(RESCUE_BAR, color="k", ls=":", lw=1.2)
    ax.axhline(NOT_RESCUABLE_MAX, color="k", ls=":", lw=1.2)
    for nm, e in table.items():
        if nm == "attn-L4+L5":
            c = "tab:red"
        elif nm.startswith("attn"):
            c = "tab:blue"
        elif nm.startswith("mlp"):
            c = "tab:orange"
        elif nm == "ln_f":
            c = "tab:green"
        else:
            c = "tab:purple"
        x = max(e["necessity"]["val_d45"], 1e-4)
        y = max(e["sufficiency"]["val_d45"], 1e-4)
        ax.scatter(x, y, s=150 if e["both_pass"] else 46,
                   marker="*" if e["both_pass"] else "o", color=c,
                   edgecolor="k", linewidth=0.5, zorder=3)
        ax.annotate(nm, (x, y), textcoords="offset points", xytext=(5, 4),
                    fontsize=7.2)
    ax.scatter(max(br_rmu_max, 1e-4), max(val_d45(sw_net0), 1e-4), marker="X",
               s=120, color="crimson", edgecolor="k", zorder=4)
    ax.annotate("whole-RMU (sealed null)", (br_rmu_max, val_d45(sw_net0)),
                textcoords="offset points", xytext=(6, -2), fontsize=7.5,
                color="crimson")
    ax.scatter(val_d45(sw_net0), max(br_rmu_max, 1e-4), marker="X", s=120,
               color="0.35", edgecolor="k", zorder=4)
    ax.annotate("whole-net0 (open null)", (val_d45(sw_net0), br_rmu_max),
                textcoords="offset points", xytext=(6, -2), fontsize=7.5,
                color="0.35")
    ax.set_xlabel(f"NECESSITY: restore net0 comp -> RMU net\n"
                  f"max site-mean p(Z) over d4/d5 (bar >= {RESCUE_BAR})")
    ax.set_ylabel(f"SUFFICIENCY: RMU comp -> net0\n"
                  f"max site-mean p(Z) over d4/d5 (bar <= {NOT_RESCUABLE_MAX})")
    ax.set_title(f"A. Gate census 2D map — {fired}\n"
                 "(green shading = localized quadrant: necessary AND sufficient)")

    # B. necessity curves (restore into RMU net)
    ax = axes[0, 1]
    for nm, e in table.items():
        hl = e["necessity"]["pass"] or e["both_pass"]
        ax.plot(DEPTHS, e["necessity"]["tf_mean_curve"], "o-" if hl else "-",
                lw=1.9 if hl else 0.8, alpha=0.95 if hl else 0.45,
                color="tab:red" if nm == "attn-L4+L5" else "tab:blue",
                label=nm if hl else None, ms=4)
    ax.plot(DEPTHS, sw_rmu["tf_mean_curve"], "o-", color="crimson", lw=2.6, ms=5,
            label="whole-RMU (sealed)")
    ax.plot(DEPTHS, sw_net0["tf_mean_curve"], "o-", color="0.2", lw=2.6, ms=5,
            label="whole-net0 (open)")
    ax.axhline(RESCUE_BAR, color="k", ls=":", lw=1)
    ax.axvspan(5.5, 6.5, color="k", alpha=0.08)
    ax.set_xlabel("write depth d"); ax.set_ylabel("site-mean P(Z) at onset")
    ax.set_title(f"B. NECESSITY — net0 component restored INTO the RMU net "
                 f"({len(sites0)} sites; d6 shaded)")
    ax.legend(fontsize=6.8, ncol=2)

    # C. sufficiency curves (poison into net0)
    ax = axes[1, 0]
    for nm, e in table.items():
        hl = e["sufficiency"]["pass"] or e["both_pass"]
        ax.plot(DEPTHS, e["sufficiency"]["tf_mean_curve"], "o-" if hl else "-",
                lw=1.9 if hl else 0.8, alpha=0.95 if hl else 0.45,
                color="tab:red" if nm == "attn-L4+L5" else "tab:blue",
                label=nm if hl else None, ms=4)
    ax.plot(DEPTHS, sw_net0["tf_mean_curve"], "o-", color="0.2", lw=2.6, ms=5,
            label="whole-net0 (open)")
    ax.plot(DEPTHS, sw_rmu["tf_mean_curve"], "o-", color="crimson", lw=2.6, ms=5,
            label="whole-RMU (sealed)")
    ax.axhline(NOT_RESCUABLE_MAX, color="k", ls=":", lw=1)
    ax.axvspan(5.5, 6.5, color="k", alpha=0.08)
    ax.set_xlabel("write depth d"); ax.set_ylabel("site-mean P(Z) at onset")
    ax.set_title(f"C. SUFFICIENCY — RMU component poisoned INTO net0 "
                 f"(bar <= {NOT_RESCUABLE_MAX:.4f})")
    ax.legend(fontsize=6.8, ncol=2)

    # D. verdict + component table
    ax = axes[1, 1]
    ax.axis("off")
    gt = all([G_CE["pass"], G_INST["pass"], G_ALPHA["pass"], G_RMU_REPLICA["pass"],
              G_SITES0["pass"], G_DONORS["pass"], G_BR_RMU["pass"],
              G_BR_NET0["pass"], G_TWINS["pass"]])
    lines = [
        f"E092 READOUT-GATE CENSUS — verdict: {fired}",
        "",
        headline,
        "",
        f"{'component':12s} {'NEC d4':>7s} {'NEC d5':>7s} {'NECmax':>7s} {'pass':>5s}"
        f" | {'SUF d4':>7s} {'SUF d5':>7s} {'SUFmax':>7s} {'pass':>5s}  BOTH",
        "-" * 88,
    ]
    for nm, e in table.items():
        n, s = e["necessity"], e["sufficiency"]
        lines.append(
            f"{nm:12s} {n['d4']:7.3f} {n['d5']:7.3f} {n['val_d45']:7.3f} "
            f"{('YES' if n['pass'] else '.'):>5s} | {s['d4']:7.3f} {s['d5']:7.3f} "
            f"{s['val_d45']:7.4f} {('YES' if s['pass'] else '.'):>5s}  "
            f"{'**BOTH**' if e['both_pass'] else ''}")
    lines += [
        "-" * 88,
        f"brackets: whole-RMU max(d<=5) {br_rmu_max:.4f} (ref {ref_rmu_max:.4f}) "
        f"{G_BR_RMU['pass']} | whole-net0 d4/d5 "
        f"{sw_net0['stats'][4]['mean_tf']:.3f}/{sw_net0['stats'][5]['mean_tf']:.3f} "
        f"(refs {ref_net0_d4:.3f}/{ref_net0_d5:.3f}) {G_BR_NET0['pass']} | twins {G_TWINS['pass']}",
        f"gates all pass: {gt} (CE {G_CE['pass']} INST {G_INST['pass']} ALPHA "
        f"{G_ALPHA['pass']} RMU-repl {G_RMU_REPLICA['pass']} SITES {G_SITES0['pass']} "
        f"DONORS {G_DONORS['pass']})",
        f"necessary-only: {nec_only or '-'} | sufficient-only: {suf_only or '-'}",
        f"riders: whole-RMU battery p(Z) {pz_rmu:.4f} CE_R {ce_rmu:.3f}; "
        f"whole-net0 battery p(Z) {pz_net0:.4f} CE_R {ce_net0:.3f}",
        f"refinement: {refinement['note']}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=7.4,
            family="monospace", transform=ax.transAxes)
    ax.set_title("D. Component table + registered verdict")

    fig.suptitle("E092 readout-gate census (T052 WHERE): component swaps between "
                 "the RMU net and e048_repro — necessity x sufficiency [CPU]",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(rd / "gate_census.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'gate_census.png'}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
