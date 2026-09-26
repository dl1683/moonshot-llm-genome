"""E082 — P1 phase-2: the CROSS-SEED row-129 transplant ("a one-row organ").

Registration: scratch/day5_programs.md section 1 (written as "e081" there;
the e081 slot was taken by the RIF probe, so the task assigns e082 — outputs
under runs/e082/). Frozen before any run; quoted bars below are verbatim.

DESIGN (registered):
  Host  = B43 (runs/checkpoints/e028_b43.pt, seed 43, 6L/6H/192, block 256).
  Donor = e048_repro.pt (seed-42 Dmix@s400 install, the exact net behind
          T042/T043/T068; same cfg, same corpus, 192-d wpe rows both sides).
  GATE 0 (the only training touch): e043's exposure protocol VERBATIM on B43
          — same windows, same splice RNG (24301 -> install-60/held-30), same
          name targets, same LR/step count (lr 1e-3, 100 steps, total 100,
          16 name + 48 anchor windows of which 32 random, wd 0.1, clip 1.0,
          house cosine warmup 100), exposure generator seed 24331 = E43's
          own b43-donor seed (protocol-identity crosscheck vs
          e043_donor_b43.pt is therefore possible). Resumable ckpt at
          runs/checkpoints/e082_b43_install.pt. Pass bars: install-60 p(Z)
          in the e043 gate family (100-step installs measure ~0.32 battery
          pZ; informativeness floor 0.20 = e067's replica gate) AND R1i NLL
          <= 4.5 (e043's G6). The registration's 2x-steps-to-bar patience
          window is implemented as: if the verbatim 100-step run fails, ONE
          fresh same-seed 200-step trajectory (total 200) is tried; if that
          also fails -> GATE 0 FAIL -> PARK (registered kill; no ad-hoc
          tuning).
  GATE 1 (eval-only, CPU): single-row wpe census on the B43 install x
          install-60 (held-30 recorded): rows {0, 1, 122..135, 250..255},
          row <- mean(all 256 host rows) + zero arm for the top rows.
          Pass bar (frozen): "129 in the top-3 rows by p(Z) drop" (the
          bimodal 0/129 structure replicates). If the host's own top
          decision row r* != 129, every arm below retargets to r* (the
          registration's explicit branch).
  MAIN (six registered arms + one task-named overlay, eval-only, CPU,
          batteries install-60 + held-30, readouts p(Z) + KL + argmax-Z):
    A0 host-intact (reference)
    A1 host-own-row destruction (zero + mean, in place) — seed-43 knife-edge
    A2 DONOR OVERWRITE: wpe[r*]_host <- wpe[129]_donor  (THE primary arm)
    A3 donor adjacent-slot add: wpe[r*+1] <- wpe[129]_donor (host row intact)
        — the e068 shifted-window analogue, cross-seed: the unshifted
        battery never feeds row r*+1 (causally dead; asserted), so the
        informative readout is the k=1-shifted battery (content at 1..r*+1,
        readout at r*+1); the k=1 nocopy collapse is recorded as reference.
    A4 sham surgery: delete host row r*, re-copy host row r* (bitwise
        roundtrip; must equal A0 exactly — surgery-damage control; the
        ~0.13-0.15 no-copy plateau it "calibrates" is measured at the k=10
        rebound leg).
    A5 donor-family mean control: wpe[r*] <- mean(donor's 256 wpe rows)
    A6 donor PAIR (overlay; the task's "pair-with-row0" = the e068
       PAIR-COPY analogue, cross-seed): wpe[0] <- donor row 0 AND
       wpe[r*] <- donor row 129.
  CROSSMATCH overlay (T046 bridge): e062's probe_streams VERBATIM (2 fixed
          val batches x 16 x 256, seed 1337), host-install vs donor, per
          depth d0..d6. The 0.4459 rule is applied at mean(d2,d3) (the
          depths it was calibrated on; primary reading of "the SAME
          graft-input depths") and reported at d0 (the literal input depth
          of a wpe-row graft) — both flagged. Position-129-restricted
          cosine per depth + direct row-cosine(donor 129, host r*),
          report-only.
  FREE RIDER (T043's open discriminator, zero marginal cost): k=10
          shifted battery (the e068/e078 rebound geometry), arms
          {nocopy (the plateau), hostcopy139 (in-seed rebind reference),
           donorcopy139 (cross-seed at the rebound)}, readout p(Z) + Z's
          RANK at k+129. Plateau-with-rank-2 => sub-argmax residue (b);
          plateau-with-rank>5 => content-row co-adaptation (a).

REGISTERED BARS (frozen, verbatim from day5_programs.md):
  Primary:   A2 p(Z) < 0.30 on install-60 => BASIS-PRIVATE; >= 0.30 =>
             PORTABLE ORGAN (0.30 = half the 0.60-0.71 in-seed rebind band).
  Secondary: the crossmatch verdict matches direction.
  Tertiary:  A1 drops p(Z) by >= half (knife-edge replicates at seed 43).
  Kills:     GATE 0 fail -> park. GATE 1 no concentration -> P1 phase-2
             halts. A2 ~= plateau (donor indistinguishable from sham) ->
             closes negative at n=1 pair, STOP (do not iterate donors).

DEVIATIONS / DECISIONS (all eval-side; GATE 0 protocol itself verbatim):
  1. Numbering: day5_programs.md calls this proposal e081; task assigns
     e082 (slot e081 already used). Outputs under runs/e082/.
  2. The task's "install fine-tune <= 180s, batch 32" note cannot override
     "VERBATIM": the e043 exposure batch is 64 windows (16 name + 48
     anchor). Kept verbatim; measured runtime is recorded (GPU: well under
     the 180s cap).
  3. A6 (donor pair) added as the task's named 6th arm ("pair-with-row0");
     the registration's own sixth arm A0 (host-intact) is kept as the
     reference, so seven arm cells run in total.
  4. Thermal discipline: gpu_ok() DOUBLE-POLLED (2 s apart, both must pass)
     before the training launch; bounded wait (<= 20 min, 30 s polls) then
     CPU fallback (flagged); cooldown(120) after any training; every eval
     CPU-side.

Outputs: runs/e082/{metrics.json, cross_seed_transplant.png}.
Run: python lab/e082_cross_seed_transplant.py   (E082_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

import common
from common import (REPO, Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,
                    run_dir, save_json, set_seed)
import e043_install as E43
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SMOKE = os.environ.get("E082_SMOKE") == "1"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
HOST_CK = REPO / "runs" / "checkpoints" / "e028_b43.pt"
DONOR_CK = REPO / "runs" / "checkpoints" / "e048_repro.pt"
E043_B43_DONOR_CK = REPO / "runs" / "checkpoints" / "e043_donor_b43.pt"
INSTALL_CK = REPO / "runs" / "checkpoints" / (
    "e082_b43_install_smoke.pt" if SMOKE else "e082_b43_install.pt")

INSTALL_GEN = 24331            # = E43.GEN_DONOR["b43"]: the e043 b43-donor recipe
INSTALL_STEPS = 100 if not SMOKE else 8
INSTALL_TOTAL = INSTALL_STEPS
INSTALL_EVAL_AT = ({25, 50, 100} if not SMOKE else {4, 8})
PATIENCE_STEPS = 200 if not SMOKE else 16   # the 2x window (fresh trajectory)
GATE0_PZ_FLOOR = 0.20          # e067 replica-gate informativeness floor
GATE0_R1I_MAX = 4.5            # e043 G6
PORTABLE_BAR = 0.30            # frozen primary bar (day5_programs.md)
CROSSMATCH_RULE = 0.4459       # T046/e062 calibrated threshold
PLATEAU_EPS = 0.03             # "A2 ~= plateau" tolerance for the kill check
K_REBOUND = 10                 # e068/e078 rebound geometry
K_ADJ = 1                      # A3's adjacent-slot readout shift
CENSUS_ROWS = sorted(set([0, 1] + list(range(122, 136)) + list(range(250, 256))))
GPU_WAIT_MAX_S = 1200
GPU_POLL_S = 30

REGISTERED = {
    "organizing_question": ("does the seed-42 install's wpe-129 address row "
                            "TRANSPLANT into a seed-43 host's own install "
                            "— a one-row organ, or basis-private code?"),
    "gate0": ("e043 exposure verbatim on B43 (100 steps, seed 24331, 16+32 "
              "anchor mix); pass: install60 p(Z) >= 0.20 AND R1i NLL <= 4.5; "
              "2x-steps patience = one fresh 200-step trajectory; else PARK"),
    "gate1": ("census rows {0,1,122-135,250-255}, row<-mean (+zero top rows) "
              "on install-60; pass: 129 in top-3 by p(Z) drop; r* retarget "
              "branch if the host binds a different row"),
    "arms": {"A0": "host-intact reference",
             "A1": "own-row destroy (zero + mean, in place)",
             "A2": "DONOR OVERWRITE wpe[r*] <- donor wpe[129] (primary)",
             "A3": "adjacent-slot add wpe[r*+1] <- donor wpe[129] "
                   "(readout at k=1 shift)",
             "A4": "sham: delete host row, re-copy host row (bitwise A0)",
             "A5": "wpe[r*] <- mean(donor wpe rows)",
             "A6": "donor PAIR: wpe[0] <- donor 0 AND wpe[r*] <- donor 129 "
                   "(task-named pair-with-row0 overlay)"},
    "bars": {"primary": "A2 install60 p(Z) >= 0.30 => PORTABLE; < 0.30 with "
                        "own-GATE-1 passed => BASIS-PRIVATE",
             "secondary": "crossmatch (0.4459 rule at d2/d3) matches the "
                          "transplant outcome direction",
             "tertiary": "A1 drops install60 p(Z) by >= half (knife-edge "
                         "replicates at seed 43)"},
    "kills": {"gate0": "install fails within 2x steps-to-bar -> PARK "
                       "(protocol fragility across seeds, T015)",
              "gate1": "no single-row concentration at seed 43 -> P1 "
                       "phase-2 halts; seed-42 scope flag on claim B",
              "a2_sham": "A2 ~= plateau (donor indistinguishable from "
                         "sham/no-copy) -> closes negative at n=1 pair, STOP"},
    "free_rider": ("k=10 rebound geometry: Z's rank at the plateau/rebound — "
                   "rank-2 plateau => sub-argmax residue (b); rank>5 => "
                   "content-row co-adaptation (a)"),
}
log("REGISTERED: " + " | ".join(f"{k}: {v}" for k, v in REGISTERED.items()))


# ------------------------------------------------------------------ helpers

def gpu_ok_double(poll_gap_s: float = 2.0) -> bool:
    """Double-poll: both readings must clear the gate (task requirement)."""
    if not gpu_ok():
        return False
    time.sleep(poll_gap_s)
    return gpu_ok()


def wait_for_gpu() -> tuple[bool, list[dict]]:
    polls = []
    t0 = time.time()
    while time.time() - t0 < GPU_WAIT_MAX_S:
        s = common.gpu_status()
        polls.append(s)
        if gpu_ok_double():
            return True, polls
        log(f"[gpu_guard] HOLD (double-poll failed): {s} — waiting "
            f"{GPU_POLL_S}s ({time.time() - t0:.0f}/{GPU_WAIT_MAX_S}s)")
        time.sleep(GPU_POLL_S)
    return False, polls


def load_ckpt_model(path: Path, dev: str) -> TinyGPT:
    m = TinyGPT(Cfg()).to(dev)
    st = torch.load(path, map_location=dev, weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery(m: TinyGPT, ids: torch.Tensor, zid: int, bs: int = 30):
    """e067/e068/e071/e078 battery verbatim + full probs + Z-rank."""
    pzs, prs = [], []
    for i in range(0, ids.shape[0], bs):
        lg, _ = m(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, zid]) for k in range(pr.shape[0])]
        prs.append(pr)
    pr = torch.cat(prs)
    pz = np.array(pzs, dtype=np.float64)
    amax = (pr.argmax(-1) == zid)
    # Z's rank: 1 + count of strictly-greater probs (tie-tolerant)
    zrank = (pr > pr[:, zid:zid + 1]).sum(-1) + 1
    return pz, pr, np.array([int(v) for v in amax]), np.array(
        [int(v) for v in zrank])


def cell(pz, pr, amax, zrank, base_pr=None) -> dict:
    c = {
        "mean_pz": float(pz.mean()), "median_pz": float(np.median(pz)),
        "std_pz": float(pz.std()), "frac_pz_ge_0.5": float((pz >= 0.5).mean()),
        "frac_argmax_z": float(amax.mean()),
        "zrank_mean": float(zrank.mean()),
        "zrank_frac_eq2": float((zrank == 2).mean()),
        "zrank_frac_3to5": float(((zrank >= 3) & (zrank <= 5)).mean()),
        "zrank_frac_gt5": float((zrank > 5).mean()),
        "pz_per_ctx": pz.tolist(),
    }
    if base_pr is not None:
        P = (base_pr + 1e-12).numpy() if torch.is_tensor(base_pr) else base_pr + 1e-12
        Q = (pr + 1e-12).numpy() if torch.is_tensor(pr) else pr + 1e-12
        P = P / P.sum(1, keepdims=True)
        Q = Q / Q.sum(1, keepdims=True)
        c["kl_base_pert"] = float((P * np.log(P / Q)).sum(1).mean())
    return c


def kl_rows_np(base_pr, pr) -> float:
    P = (base_pr + 1e-12).numpy() if torch.is_tensor(base_pr) else base_pr + 1e-12
    Q = (pr + 1e-12).numpy() if torch.is_tensor(pr) else pr + 1e-12
    P = P / P.sum(1, keepdims=True)
    Q = Q / Q.sum(1, keepdims=True)
    return float((P * np.log(P / Q)).sum(1).mean())


@torch.no_grad()
def probe_streams(model: TinyGPT, corpus: CharCorpus, n_batches: int = 2):
    """e062's probe_streams VERBATIM (token-normalized, per-depth)."""
    model.eval()
    cfg = model.cfg
    gen = torch.Generator().manual_seed(corpus.seed)
    src = corpus.val
    ix = torch.randint(len(src) - cfg.block_size - 1, (16 * n_batches,),
                       generator=gen)
    acc = None
    for b in range(n_batches):
        x = torch.stack([src[i:i + cfg.block_size]
                         for i in ix[b * 16:(b + 1) * 16]])
        pos = torch.arange(cfg.block_size)
        s = model.wte(x) + model.wpe(pos)
        outs = [s]
        for block in model.h:
            s = block(s)
            outs.append(s)
        nrm = [t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8) for t in outs]
        flat = [t.reshape(-1, t.shape[-1]) for t in nrm]
        acc = flat if acc is None else \
            [torch.cat([a, f], dim=0) for a, f in zip(acc, flat)]
    model.train()
    return acc


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)) or x is None or isinstance(x, (int, str)):
        return bool(x) if isinstance(x, np.bool_) else x
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return "inf" if math.isinf(v) else ("nan" if math.isnan(v) else v)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, torch.Tensor):
        return jsonable(x.tolist())
    return str(x)


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e082_smoke" if SMOKE else "e082")
    set_seed(E43.SEED)
    log(f"E082 cross-seed row-129 transplant (smoke={SMOKE})")

    # ---- protocol rebuild (verbatim e055/e066/e066b/e067/e068/e078)
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi = corpus.stoi
    train_ids = corpus.train
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    zid = stoi["Z"]
    assert not E43.find_occ(train_text, NAME)

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + E43.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"
    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)

    def build_win(p, host):
        return torch.cat([train_ids[p - PRE: p], name_ids,
                          train_ids[p + len(host): p + len(host) + E43.POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    win_h = torch.stack([build_win(p, h) for p, h in held_occ])
    log(f"protocol rebuilt: install60 {mix}, held30 "
        f"(corpus 1337, SPLICE_RNG {E43.SPLICE_RNG})")

    # batteries (CPU): unshifted 130-char contexts, readout at 129
    ids_unshifted = {
        "install60": torch.stack([corpus.encode(train_text[p - PRE:p])
                                  for p, _ in install_occ]),
        "held30": torch.stack([corpus.encode(train_text[p - PRE:p])
                               for p, _ in held_occ]),
    }
    # shifted batteries (k real preceding chars; content verified)
    ids_shifted = {}
    for k in (K_ADJ, K_REBOUND):
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - k:p] for p, _ in occ]
            old = [train_text[p - PRE:p] for p, _ in occ]
            for c_new, c_old in zip(cs, old):
                assert len(c_new) == PRE + k and c_new[k:] == c_old
            ids_shifted[(k, tag)] = torch.stack([corpus.encode(c) for c in cs])

    # ================================================================ GATE 0
    log("GATE 0: e043 exposure VERBATIM on B43 (the only training touch)")
    gpu_ok_now, polls = wait_for_gpu()
    dev = "cuda" if gpu_ok_now else "cpu"
    common.DEVICE = dev          # E43.exposure builds random-anchor batches
    E43.DEVICE = dev             # E43.eval_seq moves its slices to DEVICE
    log(f"gpu double-poll {'PASS' if gpu_ok_now else 'FAIL'} after "
        f"{len(polls)} poll(s); install device = {dev}")

    inst_x = win_i.clone().to(dev)
    anchor_full = torch.stack([train_ids[p - PRE: p - PRE + E43.BLOCK]
                               for p, _ in install_occ]).to(dev)
    inst_mask = torch.zeros(60, E43.BLOCK - 1, dtype=torch.bool, device=dev)
    inst_mask[:, PRE - 1:PRE - 1 + len(NAME)] = True
    bat_i_seq = win_i[:, :PRE + len(NAME)]        # R1i teacher-forced battery

    net = load_ckpt_model(HOST_CK, dev)
    log(f"B43 host loaded: params {net.num_params():,} "
        f"row129 norm {float(net.wpe.weight.data[129].norm()):.3f}")

    def gate_readout(m, step):
        pz, _, amax, _ = battery(m, ids_unshifted["install60"].to(dev), zid)
        pzh, _, _, _ = battery(m, ids_unshifted["held30"].to(dev), zid)
        r1i = E43.eval_seq(m, bat_i_seq.to(dev), len(NAME), PRE - 1)
        rec = {"step": step, "install60_pz": float(pz.mean()),
               "held30_pz": float(pzh.mean()), "r1i_nll": r1i["nll"],
               "r1i_acc": r1i["acc"],
               "onset_acc": r1i["per_pos_acc"][0],
               "pos36_acc": sum(r1i["per_pos_acc"][3:7]) / 4}
        log(f"  [install s{step:3d}] install60 p(Z) {rec['install60_pz']:.4f} "
            f"| held30 {rec['held30_pz']:.4f} | R1i {rec['r1i_nll']:.3f}/"
            f"{rec['r1i_acc']:.3f} onset {rec['onset_acc']:.2f}")
        return rec

    gen = torch.Generator().manual_seed(INSTALL_GEN)
    traj = E43.exposure(net, inst_x, inst_mask, anchor_full,
                        steps=INSTALL_STEPS, total=INSTALL_TOTAL, gen=gen,
                        tag="e082_b43_install", ckpt=INSTALL_CK,
                        eval_at=set(INSTALL_EVAL_AT),
                        on_eval=lambda m, s: gate_readout(m, s),
                        mix_random=E43.MIX_RANDOM, train_ids=train_ids,
                        log=log)
    t_install_s = time.time() - T0
    last = traj[-1] if traj else gate_readout(net, INSTALL_STEPS)
    gate0 = {
        "device": dev, "gpu_double_poll_pass": bool(gpu_ok_now),
        "n_gpu_polls": len(polls), "steps": INSTALL_STEPS,
        "seed": INSTALL_GEN, "ckpt": str(INSTALL_CK.name),
        "traj": traj, "install_runtime_s": round(t_install_s, 1),
        "pz_floor": GATE0_PZ_FLOOR, "r1i_max": GATE0_R1I_MAX,
        "patience_branch": None,
    }

    def gate0_pass(rec):
        return bool(rec["install60_pz"] >= GATE0_PZ_FLOOR
                    and rec["r1i_nll"] <= GATE0_R1I_MAX)

    if not gate0_pass(last) and not SMOKE:
        log(f"GATE 0 not met at s{INSTALL_STEPS} (pZ {last['install60_pz']:.4f}, "
            f"R1i {last['r1i_nll']:.3f}) — trying the registered 2x window: "
            f"one fresh {PATIENCE_STEPS}-step trajectory (same seed)")
        if dev == "cuda":
            cooldown(120)
        if not gpu_ok_double():
            log("gpu gate closed for the patience branch — CPU fallback")
        net2 = load_ckpt_model(HOST_CK, dev)
        gen2 = torch.Generator().manual_seed(INSTALL_GEN)
        traj2 = E43.exposure(net2, inst_x, inst_mask, anchor_full,
                             steps=PATIENCE_STEPS, total=PATIENCE_STEPS,
                             gen=gen2, tag="e082_b43_install_2x", ckpt=None,
                             eval_at={INSTALL_STEPS, PATIENCE_STEPS},
                             on_eval=lambda m, s: gate_readout(m, s),
                             mix_random=E43.MIX_RANDOM, train_ids=train_ids,
                             log=log)
        gate0["patience_branch"] = {
            "steps": PATIENCE_STEPS, "traj": traj2,
            "pass": gate0_pass(traj2[-1]) if traj2 else False}
        if gate0["patience_branch"]["pass"]:
            net, last = net2, traj2[-1]
    if dev == "cuda":
        cooldown(120)             # thermal discipline after any training
    gate0["final"] = {k: last[k] for k in ("step", "install60_pz", "held30_pz",
                                           "r1i_nll", "r1i_acc", "onset_acc",
                                           "pos36_acc")}
    gate0["pass"] = gate0_pass(last)
    log(f"GATE 0: {'PASS' if gate0['pass'] else 'FAIL'} — "
        f"install60 p(Z) {last['install60_pz']:.4f} (floor {GATE0_PZ_FLOOR}), "
        f"R1i {last['r1i_nll']:.3f} (<= {GATE0_R1I_MAX}), "
        f"held30 {last['held30_pz']:.4f}")

    # protocol-identity crosscheck vs the existing e043 b43 donor (report-only)
    host_install = load_ckpt_model(
        INSTALL_CK if Path(INSTALL_CK).exists() else HOST_CK, "cpu")
    xcheck = None
    if E043_B43_DONOR_CK.exists() and not SMOKE:
        ref = load_ckpt_model(E043_B43_DONOR_CK, "cpu")
        xcheck = {
            "wpe_max_absdiff": float((host_install.wpe.weight.data
                                      - ref.wpe.weight.data).abs().max()),
            "all_params_max_absdiff": float(max(
                (a - b).abs().max() for a, b in zip(
                    host_install.state_dict().values(),
                    ref.state_dict().values()))),
            "note": ("same recipe (seed 24331, 100 steps); nonzero diffs = "
                     "device/kernel nondeterminism between runs, report-only"),
        }
        log(f"protocol-identity vs e043_donor_b43: wpe max|d| "
            f"{xcheck['wpe_max_absdiff']:.2e}, all-param max|d| "
            f"{xcheck['all_params_max_absdiff']:.2e} (report-only)")

    if not gate0["pass"]:
        gate0["kill"] = ("GATE 0 FAIL — install did not take within 2x "
                         "e043's steps-to-bar on seed 43. REGISTERED KILL: "
                         "PARK (protocol fragility across seeds, T015); do "
                         "not ad-hoc tune. Partial metrics written; arms "
                         "NOT run.")
        out = {"experiment": "e082_cross_seed_transplant",
               "date": common.now_iso(), "smoke": SMOKE,
               "registered": REGISTERED, "protocol": {
                   "corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                   "install_mix": mix, "n_held": len(held_occ)},
               "gate0": gate0, "protocol_identity_vs_e043_donor_b43": xcheck,
               "gate1": None, "arms": None, "crossmatch": None,
               "rebound_free_rider": None, "bars": None,
               "verdict": gate0["kill"],
               "timing_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", jsonable(out))
        log("GATE 0 FAIL — STOPPED (registered kill). Partial metrics written.")
        return 0

    # ---- everything below is eval-only, CPU
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    host_install = host_install.cpu()
    donor = load_ckpt_model(DONOR_CK, "cpu")
    w_h = host_install.wpe.weight.data.clone()      # host install rows
    w_d = donor.wpe.weight.data.clone()             # donor rows
    log(f"donor {DONOR_CK.name}: params {donor.num_params():,} "
        f"row129 norm {float(w_d[129].norm()):.3f}")

    # refs (A0)
    ref = {}
    base_prs = {}
    for tag, ids in ids_unshifted.items():
        pz, pr, amax, zr = battery(host_install, ids, zid)
        ref[tag] = cell(pz, pr, amax, zr)
        base_prs[tag] = pr
        log(f"[A0 ref] unshifted {tag}: p(Z) {ref[tag]['mean_pz']:.5f} "
            f"(argmax-Z {ref[tag]['frac_argmax_z']:.2f})")

    # ================================================================ GATE 1
    log("GATE 1: seed-43 address census (rows <- mean; + zero for top rows)")
    w = host_install.wpe.weight.data
    mean_row_h = w_h.mean(0)
    census = {"rows": CENSUS_ROWS, "arm_mean": {}, "arm_zero": {}}
    for r in CENSUS_ROWS:
        w.copy_(w_h)
        w[r] = mean_row_h
        row = {}
        for tag, ids in ids_unshifted.items():
            pz, _, _, _ = battery(host_install, ids, zid)
            row[tag] = ref[tag]["mean_pz"] - float(pz.mean())
        census["arm_mean"][r] = row
        w.copy_(w_h)
    drops_i = {r: census["arm_mean"][r]["install60"] for r in CENSUS_ROWS}
    top3 = sorted(CENSUS_ROWS, key=lambda r: -drops_i[r])[:3]
    for r in top3:
        w.copy_(w_h)
        w[r] = 0.0
        pz, _, _, _ = battery(host_install, ids_unshifted["install60"], zid)
        census["arm_zero"][r] = ref["install60"]["mean_pz"] - float(pz.mean())
        w.copy_(w_h)
    ranked_rows = sorted(CENSUS_ROWS, key=lambda r: -drops_i[r])
    r0_rank = ranked_rows.index(0) + 1 if 0 in ranked_rows else None
    r129_rank = ranked_rows.index(129) + 1 if 129 in ranked_rows else None
    r_star = max((r for r in CENSUS_ROWS if r != 0), key=lambda r: drops_i[r])
    census.update({
        "drops_install60": drops_i, "top3_by_drop": top3,
        "row0_rank": r0_rank, "row129_rank": r129_rank,
        "r_star_excl_row0": int(r_star),
        "retargeted": bool(r_star != 129),
        "pass": bool(129 in top3),
    })
    log(f"[census] install60 drops: " + " ".join(
        f"{r}:{drops_i[r]:.3f}" for r in ranked_rows[:6]))
    log(f"[census] top3 {top3} | row0 rank {r0_rank} | row129 rank "
        f"{r129_rank} | r* (excl row0) {r_star} "
        f"{'(RETARGET)' if r_star != 129 else ''} | zero-arm "
        + " ".join(f"{r}:{census['arm_zero'][r]:.3f}" for r in top3))
    if not census["pass"]:
        log("GATE 1 FAIL: no single-row concentration at 129 — registered "
            "kill (P1 phase-2 halts; seed-42 scope flag). Writing partial "
            "metrics and stopping.")
        out = {"experiment": "e082_cross_seed_transplant",
               "date": common.now_iso(), "smoke": SMOKE,
               "registered": REGISTERED, "protocol": {
                   "corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                   "install_mix": mix, "n_held": len(held_occ)},
               "gate0": gate0, "refs_A0": ref,
               "protocol_identity_vs_e043_donor_b43": xcheck,
               "gate1": census, "arms": None, "crossmatch": None,
               "rebound_free_rider": None,
               "verdict": ("GATE 1 FAIL — seed-43 install shows no 129 "
                           "concentration: the single-row address is an "
                           "in-seed coincidence; P1 phase-2 HALTS, claim-B "
                           "gets an explicit seed-42 scope flag."),
               "timing_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", jsonable(out))
        return 0

    r_star = 129 if not census["retargeted"] else int(r_star)

    # ================================================================ crossmatch
    log("CROSSMATCH overlay: e062 probe_streams (host install vs donor)")
    hs = probe_streams(host_install, corpus)
    ds = probe_streams(donor, corpus)
    per_depth, pos129 = {}, {}
    for d in range(len(hs)):
        per_depth[d] = float((hs[d] * ds[d]).sum(-1).mean())
    # position-129-restricted cosine (windows are 256 tokens: pos 129 exists)
    @torch.no_grad()
    def pos129_streams(model):
        cfg = model.cfg
        g = torch.Generator().manual_seed(corpus.seed)
        src = corpus.val
        ix = torch.randint(len(src) - cfg.block_size - 1, (32,), generator=g)
        x = torch.stack([src[i:i + cfg.block_size] for i in ix])
        pos = torch.arange(cfg.block_size)
        s = model.wte(x) + model.wpe(pos)
        outs = [s[:, 129]]
        for block in model.h:
            s = block(s)
            outs.append(s[:, 129])
        nrm = [t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8) for t in outs]
        return nrm
    hp = pos129_streams(host_install)
    dp = pos129_streams(donor)
    for d in range(len(hp)):
        pos129[d] = float((hp[d] * dp[d]).sum(-1).mean())
    cos_gi = 0.5 * (per_depth[2] + per_depth[3])
    cos_d0 = per_depth[0]
    row_cos = float(F.cosine_similarity(w_d[129], w_h[r_star], dim=0))
    crossmatch = {
        "threshold": CROSSMATCH_RULE,
        "per_depth_cos_all_positions": per_depth,
        "per_depth_cos_pos129": pos129,
        "cos_graftinput_d23": cos_gi,
        "rule_admit_d23": bool(cos_gi >= CROSSMATCH_RULE),
        "cos_d0_wpe_graft_input": cos_d0,
        "rule_admit_d0": bool(cos_d0 >= CROSSMATCH_RULE),
        "row_cos_donor129_host_rstar": row_cos,
        "norms": {"donor_row129": float(w_d[129].norm()),
                  "host_row_rstar": float(w_h[r_star].norm()),
                  "host_mean_row": float(mean_row_h.norm()),
                  "donor_mean_row": float(w_d.mean(0).norm())},
        "note": ("primary rule reading: mean(d2,d3) = the depths the 0.4459 "
                 "threshold was calibrated on; d0 = the literal graft-input "
                 "depth of a wpe-row graft (instrument extension, flagged)"),
    }
    log(f"[crossmatch] cos d0..d6: "
        + " ".join(f"d{d}:{per_depth[d]:.3f}" for d in sorted(per_depth)))
    log(f"[crossmatch] cos_graftinput(d2,d3) {cos_gi:.4f} -> "
        f"{'ADMIT' if cos_gi >= CROSSMATCH_RULE else 'REJECT'} (rule "
        f"{CROSSMATCH_RULE}) | d0 {cos_d0:.4f} -> "
        f"{'ADMIT' if cos_d0 >= CROSSMATCH_RULE else 'REJECT'} | pos-129 "
        f"cos d2/d3 {pos129[2]:.3f}/{pos129[3]:.3f} | row-cos "
        f"donor129~host{r_star} {row_cos:+.3f}")

    # ================================================================ MAIN arms
    log(f"MAIN: seven arm cells (r* = {r_star}); batteries install-60 + "
        f"held-30, CPU, in-place surgery + bitwise restore")

    def run_arm(name, surgery):
        w.copy_(w_h)
        surgery(w)
        out = {}
        for tag, ids in ids_unshifted.items():
            pz, pr, amax, zr = battery(host_install, ids, zid)
            out[tag] = cell(pz, pr, amax, zr, base_pr=base_prs[tag])
        w.copy_(w_h)
        r60, r30 = out["install60"], out["held30"]
        log(f"[{name:34s}] install60 p(Z) {r60['mean_pz']:.4f} "
            f"(argmax-Z {r60['frac_argmax_z']:.2f}, KL {r60['kl_base_pert']:.3f}) "
            f"| held30 {r30['mean_pz']:.4f} (KL {r30['kl_base_pert']:.3f})")
        return out

    donor_row = w_d[129].clone()          # the donor's address row (seed 42)
    donor_row0 = w_d[0].clone()
    donor_mean_row = w_d.mean(0)

    arms = {}
    arms["A0_host_intact"] = {t: dict(ref[t]) for t in ref}
    arms["A1_own_destroy_zero"] = run_arm(
        "A1 own-row destroy (zero)",
        lambda w_, r=r_star: w_.__setitem__(r, torch.zeros_like(w_[r])))
    arms["A1_own_destroy_mean"] = run_arm(
        "A1 own-row destroy (mean-host-rows)",
        lambda w_, r=r_star: w_.__setitem__(r, mean_row_h))
    arms["A2_donor_overwrite"] = run_arm(
        "A2 DONOR OVERWRITE (primary)",
        lambda w_, r=r_star: w_.__setitem__(r, donor_row))
    arms["A3_donor_adjacent_add"] = run_arm(
        "A3 donor adjacent-slot add (unshifted; causally dead)",
        lambda w_, r=r_star: w_.__setitem__(r + 1, donor_row))
    # A4 sham: delete host row, re-copy host row (bitwise roundtrip)
    def _sham(w_, r=r_star):
        w_[r] = torch.zeros_like(w_[r])
        w_[r] = w_h[r]
    arms["A4_sham_recopy"] = run_arm("A4 sham delete+recopy", _sham)
    arms["A5_donor_mean_control"] = run_arm(
        "A5 donor-family mean control",
        lambda w_, r=r_star: w_.__setitem__(r, donor_mean_row))
    def _pair(w_, r=r_star):
        w_[0] = donor_row0
        w_[r] = donor_row
    arms["A6_donor_pair_row0_plus_rstar"] = run_arm(
        "A6 donor PAIR (row0 + address)", _pair)

    # A4 must be bitwise A0
    sham_ok = all(
        arms["A4_sham_recopy"][t]["pz_per_ctx"] == arms["A0_host_intact"][t]["pz_per_ctx"]
        for t in ids_unshifted)
    # A3 unshifted must be bitwise A0 (row r*+1 never fed: causal)
    a3_dead_ok = all(
        arms["A3_donor_adjacent_add"][t]["pz_per_ctx"] == arms["A0_host_intact"][t]["pz_per_ctx"]
        for t in ids_unshifted)
    log(f"[controls] A4 sham == A0 bitwise: {sham_ok} | A3 unshifted == A0 "
        f"bitwise (slot {r_star + 1} causally dead): {a3_dead_ok}")

    # A3 informative readout: k=1-shifted battery (readout at r*+1 = 130)
    a3_shift = {}
    nocopy_k1 = {}
    k1_base_prs = {}
    for tag in ("install60", "held30"):
        ids = ids_shifted[(K_ADJ, tag)]
        w.copy_(w_h)
        pz, pr, amax, zr = battery(host_install, ids, zid)
        k1_base_prs[tag] = pr
        nocopy_k1[tag] = cell(pz, pr, amax, zr)
        w.copy_(w_h)
        w[r_star + K_ADJ] = donor_row
        pz, pr, amax, zr = battery(host_install, ids, zid)
        a3_shift[tag] = cell(pz, pr, amax, zr, base_pr=k1_base_prs[tag])
        w.copy_(w_h)
        log(f"[A3 k={K_ADJ} shifted] {tag}: donor@{r_star + K_ADJ} p(Z) "
            f"{a3_shift[tag]['mean_pz']:.4f} vs nocopy "
            f"{nocopy_k1[tag]['mean_pz']:.4f}")
    # host's own row copied to the k=1 destination (in-seed rebind reference)
    hostcopy_k1 = {}
    for tag in ("install60", "held30"):
        ids = ids_shifted[(K_ADJ, tag)]
        w.copy_(w_h)
        w[r_star + K_ADJ] = w_h[r_star]
        pz, pr, amax, zr = battery(host_install, ids, zid)
        hostcopy_k1[tag] = cell(pz, pr, amax, zr, base_pr=k1_base_prs[tag])
        w.copy_(w_h)
        log(f"[A3 k={K_ADJ} in-seed ref] {tag}: host-row copy p(Z) "
            f"{hostcopy_k1[tag]['mean_pz']:.4f}")

    # restore check
    assert torch.equal(host_install.wpe.weight.data, w_h), "wpe not restored"
    pz_chk, _, _, _ = battery(host_install, ids_unshifted["install60"], zid)
    assert np.allclose(pz_chk, ref["install60"]["pz_per_ctx"], atol=0, rtol=0)
    log("restore checks passed: wpe bitwise restored, A0 reproduced")

    # ================================================================ free rider
    log(f"FREE RIDER: k={K_REBOUND} rebound geometry — p(Z) + Z's rank")
    rebound = {}
    for arm_name, surg in (
            ("nocopy_plateau", lambda: None),
            ("hostcopy139_inseed_rebind", lambda: w.__setitem__(
                K_REBOUND + r_star, w_h[r_star])),
            ("donorcopy139_crossseed", lambda: w.__setitem__(
                K_REBOUND + r_star, donor_row))):
        w.copy_(w_h)
        surg()
        out = {}
        for tag in ("install60", "held30"):
            ids = ids_shifted[(K_REBOUND, tag)]
            pz, pr, amax, zr = battery(host_install, ids, zid)
            out[tag] = cell(pz, pr, amax, zr)
        w.copy_(w_h)
        r60 = out["install60"]
        log(f"[rebound k={K_REBOUND} {arm_name:26s}] install60 p(Z) "
            f"{r60['mean_pz']:.4f} | Z-rank mean {r60['zrank_mean']:.2f} "
            f"frac rank2 {r60['zrank_frac_eq2']:.2f} frac 3-5 "
            f"{r60['zrank_frac_3to5']:.2f} frac >5 {r60['zrank_frac_gt5']:.2f}")
        rebound[arm_name] = out

    # ================================================================ bars
    a2 = arms["A2_donor_overwrite"]["install60"]["mean_pz"]
    a0v = ref["install60"]["mean_pz"]
    a1_drop_frac = 1 - min(
        arms["A1_own_destroy_zero"]["install60"]["mean_pz"],
        arms["A1_own_destroy_mean"]["install60"]["mean_pz"]) / max(a0v, 1e-9)
    plateau = rebound["nocopy_plateau"]["install60"]["mean_pz"]
    portable = bool(a2 >= PORTABLE_BAR)
    basis_private = bool(a2 < PORTABLE_BAR and census["pass"])
    a2_approx_plateau = bool(a2 <= plateau + PLATEAU_EPS)
    outcome_works = portable
    rule_admit = crossmatch["rule_admit_d23"]
    direction_match = bool(rule_admit == outcome_works)
    if portable:
        verdict = ("PORTABLE ORGAN — the address row is a transplantable "
                   "one-row organ ACROSS seeds: donor wpe-129 overwritten "
                   "into the seed-43 host's own decision row restores "
                   f"install-60 p(Z) to {a2:.3f} >= {PORTABLE_BAR} "
                   f"(host base {a0v:.3f}).")
    elif basis_private:
        verdict = ("BASIS-PRIVATE — row portability dies at the seed "
                   "boundary: donor overwrite leaves install-60 p(Z) at "
                   f"{a2:.3f} < {PORTABLE_BAR} while the host's own GATE-1 "
                   "concentration passed (T043's n=2 was within-family "
                   "luck).")
    else:
        verdict = ("INDETERMINATE — own-GATE-1 failed so the registered "
                   "primary dichotomy does not adjudicate (see kill).")
    if a2_approx_plateau and not portable:
        verdict += (" KILL FIRES: A2 ~= the no-copy plateau "
                    f"({a2:.3f} vs plateau {plateau:.3f}) — donor "
                    "indistinguishable from sham; the one-row-organ "
                    "question closes negative at n=1 pair (register the P2 "
                    "implication: even the minimal graft inherits init "
                    "privacy; STOP — do not iterate donors).")
    fr = rebound["nocopy_plateau"]["install60"]
    if fr["mean_pz"] > 0.02:
        if fr["zrank_frac_eq2"] >= 0.5:
            fr_call = "(b) sub-argmax residue — plateau-with-rank-2"
        elif fr["zrank_frac_gt5"] >= 0.5:
            fr_call = "(a) content-row co-adaptation — plateau-with-rank>5"
        else:
            fr_call = "mixed (no majority class)"
    else:
        fr_call = "plateau ~ 0 — rank uninformative"
    bars = {
        "primary": {"a2_install60_pz": a2, "bar": PORTABLE_BAR,
                    "portable": portable, "basis_private": basis_private,
                    "host_base_a0": a0v, "a2_over_a0": a2 / max(a0v, 1e-9)},
        "secondary_crossmatch": {
            "rule_admit_d23": rule_admit, "outcome_works": outcome_works,
            "direction_match": direction_match,
            "divergence_note": None if direction_match else (
                "INFORMATIVE divergence (registered): instrument-vs-"
                "mechanism separation at the minimal graft"),
            "rule_admit_d0": crossmatch["rule_admit_d0"]},
        "tertiary_knife_edge": {
            "a1_drop_fraction": a1_drop_frac, "ge_half": bool(a1_drop_frac >= 0.5),
            "a1_zero_pz": arms["A1_own_destroy_zero"]["install60"]["mean_pz"],
            "a1_mean_pz": arms["A1_own_destroy_mean"]["install60"]["mean_pz"]},
        "a2_vs_plateau": {"plateau_k10_nocopy": plateau,
                          "a2": a2, "a2_approx_plateau": a2_approx_plateau,
                          "eps": PLATEAU_EPS},
        "free_rider_zrank": {
            "plateau_cell": {k: fr[k] for k in
                             ("mean_pz", "zrank_mean", "zrank_frac_eq2",
                              "zrank_frac_3to5", "zrank_frac_gt5")},
            "call": fr_call},
        "controls": {"a4_sham_bitwise_a0": bool(sham_ok),
                     "a3_unshifted_bitwise_a0": bool(a3_dead_ok)},
    }
    log("=" * 78)
    log(f"BARS: A2 {a2:.4f} vs {PORTABLE_BAR} -> portable {portable}, "
        f"basis-private {basis_private} | plateau(k10) {plateau:.4f} "
        f"| A1 drop {a1_drop_frac * 100:.0f}% (>=50% knife-edge "
        f"{bars['tertiary_knife_edge']['ge_half']}) | crossmatch d23 "
        f"{'ADMIT' if rule_admit else 'REJECT'} vs outcome "
        f"{'works' if outcome_works else 'fails'} -> direction match "
        f"{direction_match} | free-rider {fr_call}")
    log(f"VERDICT: {verdict}")

    # ================================================================ outputs
    out = {
        "experiment": "e082_cross_seed_transplant",
        "date": common.now_iso(), "smoke": SMOKE,
        "registered": REGISTERED,
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "n_held": len(held_occ),
                     "host_ckpt": HOST_CK.name, "donor_ckpt": DONOR_CK.name,
                     "donor_note": ("e048_repro = seed-42 Dmix@s400 install "
                                    "(e068 refs: install60 0.5563, held30 "
                                    "0.4287 — reproduced bit-exact in this "
                                    "run's pre-flight)"),
                     "r_star": int(r_star), "retargeted": census["retargeted"]},
        "gate0": gate0,
        "protocol_identity_vs_e043_donor_b43": xcheck,
        "gate1": census,
        "refs_A0": ref,
        "crossmatch": crossmatch,
        "arms": arms,
        "a3_k1_shifted": {"donor_adjacent": a3_shift, "nocopy_k1": nocopy_k1,
                          "hostcopy_inseed_ref": hostcopy_k1,
                          "note": (f"readout at {r_star + K_ADJ}; unshifted "
                                   "A3 is bitwise A0 (slot causally dead)")},
        "rebound_free_rider_k10": rebound,
        "bars": bars,
        "verdict": verdict,
        "timing_s": round(time.time() - T0, 1),
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block": 256, "params": int(host_install.num_params())},
    }
    save_json(rd / "metrics.json", jsonable(out))

    # ---------------------------------------------------------------- figure
    arm_keys = ["A0_host_intact", "A1_own_destroy_zero", "A1_own_destroy_mean",
                "A2_donor_overwrite", "A5_donor_mean_control",
                "A6_donor_pair_row0_plus_rstar"]
    arm_lbl = ["A0\nintact", "A1 zero", "A1 mean", "A2 DONOR\nOVERWRITE",
               "A5 donor\nmean row", "A6 donor\npair(0,r*)"]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.6))

    # (0,0) census
    ax = axes[0, 0]
    rows_plt = CENSUS_ROWS
    xs = np.arange(len(rows_plt))
    ax.bar(xs - 0.2, [census["arm_mean"][r]["install60"] for r in rows_plt],
           0.4, color="steelblue", label="install-60", edgecolor="k", lw=0.3)
    ax.bar(xs + 0.2, [census["arm_mean"][r]["held30"] for r in rows_plt],
           0.4, color="lightsteelblue", label="held-30", edgecolor="k", lw=0.3)
    for i, r in enumerate(rows_plt):
        if r == 129:
            ax.axvline(i, color="crimson", ls=":", lw=1.2)
        if r == 0:
            ax.axvline(i, color="k", ls=":", lw=0.8)
    zr_ = sorted(census["arm_zero"])
    ax.scatter([rows_plt.index(r) for r in zr_],
               [census["arm_zero"][r] for r in zr_], marker="x", color="k",
               s=28, zorder=4, label="zero arm (top rows)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r) for r in rows_plt], fontsize=6.5, rotation=90)
    ax.set_ylabel("drop in battery p(Z)  (row <- mean)")
    ax.set_title(f"GATE 1 census on B43 install (base p(Z) {a0v:.3f}) — "
                 f"{'PASS' if census['pass'] else 'FAIL'}\n"
                 f"top3 {top3}; row129 rank {r129_rank}; r*={r_star}"
                 + (" RETARGET" if census["retargeted"] else ""), fontsize=9)
    ax.legend(fontsize=7)

    # (0,1) arms
    ax = axes[0, 1]
    xs = np.arange(len(arm_keys))
    for j, (tag, col) in enumerate((("install60", "crimson"),
                                    ("held30", "steelblue"))):
        vals = [arms[k][tag]["mean_pz"] for k in arm_keys]
        ax.bar(xs + (j - 0.5) * 0.38, vals, 0.38, color=col, label=tag,
               edgecolor="k", lw=0.4)
        for x, v in zip(xs + (j - 0.5) * 0.38, vals):
            ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.5,
                    rotation=90, va="bottom")
    ax.axhline(PORTABLE_BAR, color="seagreen", ls="--", lw=1.4,
               label=f"PORTABLE bar {PORTABLE_BAR}")
    ax.axhline(plateau, color="gray", ls=":", lw=1.2,
               label=f"k10 no-copy plateau {plateau:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(arm_lbl, fontsize=7)
    ax.set_ylabel("mean p(Z) at readout")
    top = max(max(arms[k]["install60"]["mean_pz"] for k in arm_keys), a0v)
    ax.set_ylim(0, max(0.5, top * 1.3))
    ax.set_title(f"MAIN arms (r*={r_star}) — A2 {a2:.3f} -> "
                 f"{'PORTABLE' if portable else 'BASIS-PRIVATE'}", fontsize=9)
    ax.legend(fontsize=7)

    # (0,2) crossmatch
    ax = axes[0, 2]
    ds_ = sorted(per_depth)
    ax.plot(ds_, [per_depth[d] for d in ds_], "o-", color="seagreen",
            label="cos all positions")
    ax.plot(ds_, [pos129[d] for d in ds_], "s--", color="darkorange", ms=4,
            label="cos @ position 129")
    ax.axhline(CROSSMATCH_RULE, color="crimson", ls="--", lw=1.2,
               label=f"rule {CROSSMATCH_RULE}")
    for d in (2, 3):
        ax.axvline(d, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("residual-stream depth (d2,d3 = calibrated graft-input)")
    ax.set_ylabel("host-install vs donor stream cosine")
    ax.set_title(f"CROSSMATCH: cos(d2,d3) {cos_gi:.4f} -> "
                 f"{'ADMIT' if rule_admit else 'REJECT'} | outcome "
                 f"{'works' if outcome_works else 'fails'} -> match "
                 f"{direction_match}\nrow-cos donor129~host{r_star} "
                 f"{row_cos:+.3f}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # (1,0) KL lens
    ax = axes[1, 0]
    kl_keys = arm_keys[1:]
    vals = [arms[k]["install60"]["kl_base_pert"] for k in kl_keys]
    xs = np.arange(len(kl_keys))
    cols = ["gray", "gray", "crimson", "darkorange", "purple"]
    ax.bar(xs, vals, 0.55, color=cols, edgecolor="k", lw=0.4)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhspan(0.2, 0.5, color="seagreen", alpha=0.15, label="row-129 band (0.2-0.5)")
    ax.axhspan(2.6, 7.2, color="crimson", alpha=0.10, label="row-0 band (2.6-7.2)")
    ax.set_xticks(xs)
    ax.set_xticklabels([l.replace("\n", " ") for l in arm_lbl[1:]], fontsize=6.5)
    ax.set_ylabel("KL(base || arm), install-60 readout")
    ax.set_title("KL lens: surgical vs distribution-wide destruction", fontsize=9)
    ax.legend(fontsize=7)

    # (1,1) A3 + rebound
    ax = axes[1, 1]
    labels = ["k=1 nocopy\n(collapse ref)", "k=1 host-copy\n(in-seed rebind)",
              "k=1 donor@" + str(r_star + 1), f"k={K_REBOUND} nocopy\n(plateau)",
              f"k={K_REBOUND} host-copy\n(rebound)", f"k={K_REBOUND} donor"]
    vals = [nocopy_k1["install60"]["mean_pz"],
            hostcopy_k1["install60"]["mean_pz"],
            a3_shift["install60"]["mean_pz"],
            rebound["nocopy_plateau"]["install60"]["mean_pz"],
            rebound["hostcopy139_inseed_rebind"]["install60"]["mean_pz"],
            rebound["donorcopy139_crossseed"]["install60"]["mean_pz"]]
    xs = np.arange(len(labels))
    cols = ["lightgray", "steelblue", "darkorange", "lightgray", "steelblue",
            "darkorange"]
    ax.bar(xs, vals, 0.55, color=cols, edgecolor="k", lw=0.4)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("mean p(Z) at shifted readout")
    ax.set_title(f"adjacent-slot (k={K_ADJ}) + rebound (k={K_REBOUND}) legs — "
                 f"free rider: {fr_call}", fontsize=9)

    # (1,2) verdict panel
    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        f"E082 CROSS-SEED ROW-{r_star} TRANSPLANT  (host B43 seed-43 install, "
        f"donor e048_repro seed-42)",
        "",
        f"GATE 0 install: {'PASS' if gate0['pass'] else 'FAIL'}  "
        f"(install60 p(Z) {gate0['final']['install60_pz']:.4f}; R1i "
        f"{gate0['final']['r1i_nll']:.3f}; held30 {gate0['final']['held30_pz']:.4f})",
        f"GATE 1 census : {'PASS' if census['pass'] else 'FAIL'}  "
        f"(top3 {top3}, row129 rank {r129_rank}, r*={r_star})",
        "",
        f"A2 donor overwrite install60 p(Z): {a2:.4f}  "
        f"(bar {PORTABLE_BAR}; host base {a0v:.4f}; ratio {a2 / max(a0v, 1e-9):.2f})",
        f"A1 own-row destroy: zero "
        f"{arms['A1_own_destroy_zero']['install60']['mean_pz']:.4f} / mean "
        f"{arms['A1_own_destroy_mean']['install60']['mean_pz']:.4f} "
        f"(drop {a1_drop_frac * 100:.0f}%)",
        f"k10 no-copy plateau {plateau:.4f} | A2 ~= plateau: {a2_approx_plateau}",
        f"crossmatch cos(d2,d3) {cos_gi:.4f} -> "
        f"{'ADMIT' if rule_admit else 'REJECT'}; direction match "
        f"{direction_match}",
        f"free rider (T043): {fr_call} (rank2 frac "
        f"{fr['zrank_frac_eq2']:.2f}, >5 frac {fr['zrank_frac_gt5']:.2f})",
        "",
        "VERDICT: " + verdict,
    ]
    ax.text(0.02, 0.97, "\n".join(lines), va="top", ha="left", fontsize=8.4,
            family="monospace", transform=ax.transAxes,
            bbox=dict(facecolor="#f7f7f7", edgecolor="gray"))

    fig.suptitle("E082 — cross-seed row-129 transplant: a one-row organ? "
                 f"(host seed-43 B43 install <- donor seed-42 row "
                 f"{r_star})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "cross_seed_transplant.png", dpi=140)
    log(f"outputs: {rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
