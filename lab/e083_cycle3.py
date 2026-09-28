"""E083 — CANALIZATION CYCLE-3: T037 construct 2's still-unrun registered prediction.

Registration (T037 "SYNTHESIS", construct 2 CANALIZATION, verbatim):
  "REGISTERED PREDICTION (falsifiable): a third erase/re-learn cycle is
   SLOWER and more surgical-proof than the second — monotone closure,
   never oscillation."
Fresh frame (W003): the scar (e044/e044b) is the index's trace in the
geometry — canalization as sclerosis of the fast store. New context from
e115 (dimmer switch): the address brake can SIGN-FLIP (deleting the
address raised expression 0.785->0.917 on the replayed e109 net) — so the
surgical-proofness arm is reported as retention ratio AND absolutes.

TEST (task-frozen): three full erase->re-learn cycles on the B43 INSTALL
line. runs/checkpoints/e082_b43_install.pt is cycle-0 post-install (B43 +
e043 exposure verbatim, seed 24331, install60 p(Z) 0.320, R1i 0.191).
  ERASE    = e044b's D2 row-reset verbatim: wte[Z]=0, lm_head[Z]=0
             (Z = ZEPHYRA's first char; the install's address rows),
             confinement gate (2*192 elems, Z rows only) + dCE gate
             (<= 0.01) per cycle + off-bar-with-margin gate (deviation 7).
  RE-LEARN = e044b's Dmix battery verbatim: batch 32 = 8 name + 24 corpus
             (8 paired anchors + 16 random), AdamW lr 1e-3 (0.9,0.95)
             wd 0.1 clip 1.0, cosine total 1000 (house warmup 100),
             token-weighted union CE; FULL 300-step battery per cycle; <=
             180 s of TRAINING compute per re-learn; bar = spliced
             install-60 NLL <= 1.0 & acc >= 0.8 (e044b's tasking bar),
             scanned from the dense eval grid; the at-bar state is
             SNAPSHOTTED for the proofness strike; cycles chain from
             their END states (e044b-verbatim).
  RECORD per cycle: steps-to-bar; regrown-row cos to the cycle-0 original
             (cos_wte primary, cos_lm secondary, 25% norm-validity guard
             as metadata); surgical-proofness = install-60 p(Z) after the
             FIXED re-ablation (e082's A1 address-row strike: wpe[129]<-0
             primary, wpe[129]<-mean(all rows) secondary) at bar, plus
             intact p(Z) and the retention ratio.

REGISTERED BARS (task-frozen):
  cycle3/cycle2 steps ratio >= 1 AND regrown cos >= 0.6  => MONOTONE
    CLOSURE (canalization stands — the groove deepens).
  ratio < 0.8 OR cos < 0.5 => CANALIZATION FALSIFIED (strips the
    read-policy program's canal prior).
  oscillation anywhere (a cycle FASTER than its predecessor) => W001's
    dreaming — report honestly.

DEVIATIONS / DECISIONS (all documented, none touching the registered bars):
  1. FULL 300-step re-learn battery per cycle (e044b-verbatim: steps-to-
     bar scanned from the trajectory; the scar cos is read at cycle END);
     the at-bar state is snapshotted mid-run for the proofness strike. An
     earlier implementation early-stopped at bar and chained at-bar
     states; its cycle-3 erase went VACUOUS (state still at bar after the
     full D2 reset) because at-bar states on this line have ~1% row
     regrowth — the bar is met by the completion machinery before the
     address rows refill. That artifact is recorded in the run log and
     the design reverted to the full battery before adjudication. A
     GENUINE vacuous erase under the full-battery protocol (lesion-
     tolerant tasking) is recorded as steps_to_bar=0 + erase_vacuous and
     reported honestly, distinct from W001 dreaming.
  2. The exposure generator seed is IDENTICAL across cycles (24401,
     e044's paired-draw family): cycle-to-cycle differences are state-
     only, never data-draw noise.
  3. The 180 s cap counts TRAINING compute only (task: "each re-learn
     <=180s"); ALL evals run CPU-side on state_dict snapshots (the GPU
     idles during evals — natural thermal spacing).
  4. Eval grid densified early (every step 1..40, then 50..300) for
     steps-ratio resolution; e044b's sparser grid [1,2,3,4,6,8,12,...]
     is the ancestry.
  5. Thermal: gpu_ok() DOUBLE-POLLED (2 s apart) before each fine-tune,
     bounded wait (<= 20 min) then flagged CPU fallback; cooldown(120)
     after each fine-tune (task requirement).
  6. Surgical-proofness instrument = e082's A1 strike on wpe[129]
     (GATE-1 r* = 129, not retargeted), fixed across cycles.
  7. Erase blowup gate RE-BASED (smoke findings, fixed before the full
     run): e044b's natural-name NLL >= 4.0 does not transfer to the
     install line (the 6 continuation positions survive the Z-row
     strike); and a fixed off-bar margin would confound the registered
     signal (canalization = the erase knocks the state off LESS). The
     gate is strictly-off-bar (fails NLL<=1.0 & acc>=0.8). e044b's 4.0
     recorded as reference.

Outputs: runs/e083/{metrics.json, cycle3.png}   (smoke: runs/e083_smoke/)
Run: python lab/e083_cycle3.py   (E083_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import copy
import math
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import common
from common import (REPO, Cfg, CharCorpus, TinyGPT, cfg_dict, cosine_lr,
                    cooldown, gpu_ok, gpu_status, run_dir, save_json,
                    set_seed)

SMOKE = os.environ.get("E083_SMOKE") == "1"

INSTALL_CK = REPO / "runs" / "checkpoints" / "e082_b43_install.pt"
NAME = "ZEPHYRA"
HOSTS = ["FLORIZEL", "ELIZABETH"]
PRE, BLOCK, POST_CAP = 130, 256, 119          # e043 frozen install geometry
SPLICE_RNG = 24301                            # e043 frozen splice set
SEED = 24403                                  # e044 protocol seed family
GEN_EXP = 24401                               # e044's paired-draw seed, ALL cycles
LR = 1e-3
NAME_BS, CORP_BS, MIX_RANDOM = 8, 24, 16      # e044b compact batch 32
TOTAL_SCHED = 1000                            # e044b cosine total (warmup 100)
N_CYCLES = 3
BAR_NLL, BAR_ACC = 1.0, 0.8                   # e044b's tasking bar
# Erase gate, RE-BASED for the install line (smoke findings, fixed before
# the full run): e044b's natural-name G3 (NLL >= 4.0) does NOT transfer —
# on the install line the 6 continuation positions survive the Z-row
# strike (position-bound completion machinery), so D2 lands at NLL ~2.7.
# A fixed-magnitude off-bar margin would be WRONG here: if later cycles
# canalize (rely LESS on the Z rows), the erase knocks them off LESS —
# the gate must not kill the run exactly when the registered finding
# appears. The honest gate is STRICTLY OFF-BAR (fails NLL<=1.0 & acc>=0.8)
# so steps-to-bar >= 1 is meaningful; margins recorded, not gated.
ERASE_NLL_REF_E044B = 4.0
RETENTION_PZ_FLOOR = 0.05   # retention ratio meaningful only above this
DCE_GATE = 0.01                               # e044b G2
COS_MIN_NORM_FRAC = 0.25                      # e044b Review-6 validity guard
TRAIN_CAP_S = 180.0                           # per re-learn, TRAINING compute
EXPOSE_STEPS = 300
COOLDOWN_S = 120.0
ADDR_ROW = 129                                # e082 GATE-1 r* (not retargeted)
CE_SEED, N_CE_BLOCKS = 202, 30                # e044b CE instrument
GPU_WAIT_MAX_S, GPU_POLL_S = 1200.0, 30.0     # e082's bounded GPU wait

if SMOKE:
    EXPOSE_STEPS, TRAIN_CAP_S, COOLDOWN_S = 8, 60.0, 5.0
    EVAL_STEPS = list(range(1, 9))
else:
    EVAL_STEPS = list(range(1, 41)) + [50, 60, 75, 100, 150, 200, 250, 300]
EVAL_SET = set(EVAL_STEPS)
SPARSE_AT = {4, 8, 12, 16, 20, 25, 30, 35, 40, 50, 60, 75, 100, 150, 200,
             250, 300}                        # held/CE/p(Z)-intact readouts

REGISTERED = {
    "prediction": ("a third erase/re-learn cycle is SLOWER and more "
                   "surgical-proof than the second — monotone closure, "
                   "never oscillation (T037 construct 2)"),
    "protocol": ("3x (D2 row-reset of the install's Z rows -> e044b Dmix "
                 "re-learn to bar NLL<=1.0 & acc>=0.8) on the e082 B43 "
                 "install; record steps-to-bar, regrown-row cos to the "
                 "cycle-0 original, p(Z) after the fixed wpe[129] strike "
                 "at bar"),
    "bars": {
        "monotone_closure": "steps3/steps2 >= 1 AND cos_wte(cycle3) >= 0.6",
        "canalization_falsified": "steps3/steps2 < 0.8 OR cos_wte(cycle3) < 0.5",
        "oscillation": "any cycle strictly faster than its predecessor "
                       "=> W001 dreaming, report honestly",
        "censored": "a cycle that never reaches bar inside the 300-step/"
                    "180 s envelope has steps_to_bar=None; the ratio "
                    "clause cannot fire and the outcome is reported as "
                    "censored-on-steps, cos clause still read",
    },
    "e115_context": ("dimmer switch: the address brake sign-flips (deleting "
                     "the address RAISED expression on the replayed e109 "
                     "net) — proofness reported as retention ratio AND "
                     "absolutes"),
    "proofness_rider": ("at-bar p(Z) sits at the install line's onset wall "
                        "(~1e-3; the bar is carried by the 6 completion "
                        "positions), so the SAME fixed strike's dNLL on "
                        "the spliced install-60 battery is recorded "
                        "alongside — smaller/negative = more surgical-"
                        "proof; added after cycle 1 of the first full-run "
                        "attempt, before adjudication"),
}

E082_REFS = {"a0_install60_pz": 0.3198219204942385,
             "a1_zero_install60_pz": 0.1945319922020038,
             "a1_mean_install60_pz": 0.2237501868357261,
             "a0_held30_pz": 0.2577103716166069,
             "gate0_r1i_nll": 0.19129371643066406}
E044B_REF = {"cos_wte_a": 0.7276514172554016, "steps_a": 35, "steps_b": 12,
             "steps_b2": 25, "regrow_wte_a": 0.454943406615193,
             "note": "e044b scar arm on the NATURAL-name line (JULIET on "
                     "D2-erased B43); e083 is the INSTALL line (ZEPHYRA)"}

# ------------------------------------------------------------------ helpers

def find_occ(text: str, word: str) -> list[int]:
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])")
    return [m.start() for m in pat.finditer(text)]


def fixed_blocks(src, block, n, seed):
    gen = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(src) - block - 1, (n,), generator=gen)
    x = torch.stack([src[i: i + block] for i in ix])
    y = torch.stack([src[i + 1: i + 1 + block] for i in ix])
    return x, y


def gpu_ok_double(poll_gap_s: float = 2.0) -> bool:
    if not gpu_ok():
        return False
    time.sleep(poll_gap_s)
    return gpu_ok()


def wait_for_gpu(log) -> tuple[bool, list[dict]]:
    polls = []
    t0 = time.time()
    while time.time() - t0 < GPU_WAIT_MAX_S:
        s = gpu_status()
        polls.append(s)
        if gpu_ok_double():
            return True, polls
        log(f"[gpu_guard] HOLD (double-poll failed): {s} — waiting "
            f"{GPU_POLL_S:.0f}s ({time.time() - t0:.0f}/{GPU_WAIT_MAX_S:.0f}s)")
        time.sleep(GPU_POLL_S)
    return False, polls


def load_ckpt_model(path: Path, dev: str) -> TinyGPT:
    m = TinyGPT(Cfg()).to(dev)
    st = torch.load(path, map_location=dev, weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


def mk_cpu_net(sd: dict) -> TinyGPT:
    m = TinyGPT(Cfg())
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def eval_bat(net, seq, L, lo, chunk=128):
    """e044b eval_bat verbatim (CPU net)."""
    net.eval()
    x, y = seq[:, :-1], seq[:, 1:]
    nlls, accs = [], []
    for i in range(0, len(x), chunk):
        xc, yc = x[i: i + chunk], y[i: i + chunk]
        logits, _ = net(xc)
        lg = logits[:, lo: lo + L, :]
        tg = yc[:, lo: lo + L]
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                              reduction="none").view(-1, L)
        nlls.append(nll)
        accs.append((lg.argmax(-1) == tg).float())
    nll_m, acc_m = torch.cat(nlls), torch.cat(accs)
    return {"n": int(nll_m.shape[0]), "nll": float(nll_m.mean().item()),
            "acc": float(acc_m.mean().item())}


@torch.no_grad()
def ce_val(net, x, y, bs=64):
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        xb, yb = x[i: i + bs], y[i: i + bs]
        _, loss = net(xb, yb)
        tot += float(loss.item()) * len(xb)
        n += len(xb)
    return tot / max(n, 1)


@torch.no_grad()
def pz_battery(net, ids, zid, bs=30) -> float:
    """e082's battery p(Z at last position), mean over contexts (CPU)."""
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i: i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(v) for v in pr[:, zid]]
    return float(np.mean(pzs))


def row_probe(sd, zid, orig_w, orig_l, orig_wpe129):
    w, l = sd["wte.weight"][zid], sd["lm_head.weight"][zid]
    w129 = sd["wpe.weight"][ADDR_ROW]
    return {"wte_norm": float(w.norm()), "lm_norm": float(l.norm()),
            "cos_wte": float(F.cosine_similarity(w, orig_w, dim=0)),
            "cos_lm": float(F.cosine_similarity(l, orig_l, dim=0)),
            "wpe129_norm": float(w129.norm()),
            "cos_wpe129": float(F.cosine_similarity(w129, orig_wpe129, dim=0))}


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


# ------------------------------------------------------------------ exposure

def relearn(net, dev, inst_x, inst_mask, anchor, train_ids, *, steps, gen,
            log, on_eval):
    """e044b's Dmix exposure VERBATIM in optimizer/loss/batch composition
    AND run length (the FULL step battery; steps-to-bar is scanned from the
    dense eval grid, the at-bar state is SNAPSHOTTED for the proofness
    strike, and the cycle chains from its end state). Deviations: explicit
    device, TRAINING-compute-only time cap, CPU-snapshot eval callback."""
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    net.train()
    n_inst, n_anc = inst_x.shape[0], anchor.shape[0]
    t_all0 = time.time()
    t_train = 0.0
    stopped = None
    traj = []
    bar_step = None
    at_bar_sd = None
    last_sd = None
    for step in range(1, steps + 1):
        f = cosine_lr(step - 1, TOTAL_SCHED)
        for g in opt.param_groups:
            g["lr"] = LR * f
        ts = time.time()
        ix = torch.randint(n_inst, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (CORP_BS - MIX_RANDOM,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (MIX_RANDOM,),
                           generator=gen)
        corp = torch.cat([anchor[aj],
                          torch.stack([train_ids[s: s + BLOCK]
                                       for s in rj]).to(dev)], 0)
        nw = inst_x[ix]
        x = torch.cat([nw[:, :-1], corp[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], corp[:, 1:]], 0)
        m = torch.zeros(NAME_BS + CORP_BS, x.shape[1], dtype=torch.bool,
                        device=dev)
        m[:NAME_BS] = inst_mask[ix]
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:NAME_BS][m[:NAME_BS]]
        cm = nll[NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - ts
        if step in EVAL_SET:
            last_sd = {k: v.detach().cpu().clone()
                       for k, v in net.state_dict().items()}
            rec = on_eval(last_sd, step)
            traj.append(rec)
            if rec["bar"] and bar_step is None:
                bar_step = step
                at_bar_sd = {k: v.clone() for k, v in last_sd.items()}
        if t_train > TRAIN_CAP_S:
            stopped = ("train_cap", step)
            break
    net.eval()
    return {"traj": traj, "bar_step": bar_step, "stopped": stopped,
            "planned_steps": steps, "train_s": round(t_train, 1),
            "wall_s": round(time.time() - t_all0, 1),
            "final_sd": last_sd, "at_bar_sd": at_bar_sd, "device": dev}


# ------------------------------------------------------------------ main

def main():
    T0 = time.time()
    stamp = lambda: f"[{time.time() - T0:7.1f}s]"
    log = lambda m: print(f"{stamp()} {m}", flush=True)
    set_seed(SEED)
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    rd = run_dir("e083_smoke" if SMOKE else "e083")
    log(f"E083 canalization cycle-3 (smoke={SMOKE}) — T037 construct 2's "
        f"registered prediction, W003 frame")

    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192,
              block_size=256)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    assert find_occ(train_text, NAME) == []

    # ---- e043 frozen install set (data-side identical to e044b/e082)
    host_occ = []
    for host in HOSTS:
        for p in find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"
    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)
    L = len(NAME)

    def build_win(p, host):
        return torch.cat([train_ids[p - PRE: p], name_ids,
                          train_ids[p + len(host): p + len(host) + POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    win_h = torch.stack([build_win(p, h) for p, h in held_occ])
    anchor_full = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                               for p, _ in install_occ])
    inst_mask = torch.zeros(60, BLOCK - 1, dtype=torch.bool)
    inst_mask[:, PRE - 1: PRE - 1 + L] = True
    bat_i, bat_h = win_i[:, :PRE + L], win_h[:, :PRE + L]
    pz_i_ids = torch.stack([corpus.encode(train_text[p - PRE: p])
                            for p, _ in install_occ])
    pz_h_ids = torch.stack([corpus.encode(train_text[p - PRE: p])
                            for p, _ in held_occ])
    vx, vy = fixed_blocks(val_ids, BLOCK, N_CE_BLOCKS, CE_SEED)
    log(f"install set rebuilt: install60 {mix} / held30 "
        f"(SPLICE_RNG {SPLICE_RNG}); Z row id {zid}; address row {ADDR_ROW}")

    # ---- cycle-0 anchor (CPU)
    net0 = load_ckpt_model(INSTALL_CK, "cpu")
    sd0 = {k: v.clone() for k, v in net0.state_dict().items()}
    orig_w = sd0["wte.weight"][zid].clone()
    orig_l = sd0["lm_head.weight"][zid].clone()
    orig_wpe129 = sd0["wpe.weight"][ADDR_ROW].clone()
    ow, ol = float(orig_w.norm()), float(orig_l.norm())

    def full_readout(sd, tag):
        m = mk_cpu_net(sd)
        out = {"tag": tag,
               "spliced_i": eval_bat(m, bat_i, L, PRE - 1),
               "spliced_h": eval_bat(m, bat_h, L, PRE - 1),
               "ce": ce_val(m, vx, vy),
               "pz_i": pz_battery(m, pz_i_ids, zid),
               "pz_h": pz_battery(m, pz_h_ids, zid),
               "rows": row_probe(sd, zid, orig_w, orig_l, orig_wpe129)}
        w129 = sd["wpe.weight"][ADDR_ROW].clone()
        for kind in ("zero", "mean"):
            sd_s = {k: v.clone() for k, v in sd.items()}
            sd_s["wpe.weight"][ADDR_ROW] = (torch.zeros_like(w129) if kind == "zero"
                                            else sd["wpe.weight"].mean(0))
            ms = mk_cpu_net(sd_s)
            out[f"pz_i_strike_{kind}"] = pz_battery(ms, pz_i_ids, zid)
            out[f"pz_h_strike_{kind}"] = pz_battery(ms, pz_h_ids, zid)
            # battery-damage rider (added after the first full-run cycle
            # showed at-bar p(Z) pinned at the onset wall ~1e-3 — the
            # install line's bar is carried by the 6 completion positions,
            # so p(Z) alone cannot grade proofness at bar; the SAME fixed
            # strike's damage to the tasking battery can)
            sb = eval_bat(ms, bat_i, L, PRE - 1)
            out[f"spliced_i_strike_{kind}"] = sb
            out[f"strike_dNLL_{kind}"] = sb["nll"] - out["spliced_i"]["nll"]
            out[f"strike_dAcc_{kind}"] = sb["acc"] - out["spliced_i"]["acc"]
        for kind in ("zero", "mean"):
            out[f"retention_{kind}"] = (out[f"pz_i_strike_{kind}"]
                                        / max(out["pz_i"], 1e-9))
        out["retention_valid"] = bool(out["pz_i"] >= RETENTION_PZ_FLOOR)
        out["regrow_frac_wte"] = out["rows"]["wte_norm"] / ow
        out["regrow_frac_lm"] = out["rows"]["lm_norm"] / ol
        out["cos_guard_wte_pass"] = bool(out["regrow_frac_wte"]
                                         >= COS_MIN_NORM_FRAC)
        return out

    log("cycle-0 anchor (e082 install, CPU)")
    anchor = full_readout(sd0, "cycle0")
    xcheck = {
        "a0_install60_pz": {"this_run": anchor["pz_i"],
                            "e082_ref": E082_REFS["a0_install60_pz"],
                            "diff": abs(anchor["pz_i"] - E082_REFS["a0_install60_pz"])},
        "a1_zero_strike_pz": {"this_run": anchor["pz_i_strike_zero"],
                              "e082_ref": E082_REFS["a1_zero_install60_pz"],
                              "diff": abs(anchor["pz_i_strike_zero"]
                                          - E082_REFS["a1_zero_install60_pz"])},
        "a1_mean_strike_pz": {"this_run": anchor["pz_i_strike_mean"],
                              "e082_ref": E082_REFS["a1_mean_install60_pz"],
                              "diff": abs(anchor["pz_i_strike_mean"]
                                          - E082_REFS["a1_mean_install60_pz"])},
        "r1i_nll": {"this_run": anchor["spliced_i"]["nll"],
                    "e082_ref": E082_REFS["gate0_r1i_nll"],
                    "diff": abs(anchor["spliced_i"]["nll"]
                                - E082_REFS["gate0_r1i_nll"])},
        "tol": 2e-3,
    }
    xcheck["pass"] = bool(all(v["diff"] <= xcheck["tol"]
                              for k, v in xcheck.items()
                              if isinstance(v, dict)))
    log(f"crosscheck vs e082 (tol {xcheck['tol']}): "
        f"pz {anchor['pz_i']:.6f}/{E082_REFS['a0_install60_pz']:.6f}, "
        f"strike0 {anchor['pz_i_strike_zero']:.6f}/"
        f"{E082_REFS['a1_zero_install60_pz']:.6f}, "
        f"R1i {anchor['spliced_i']['nll']:.4f}/"
        f"{E082_REFS['gate0_r1i_nll']:.4f} -> {xcheck['pass']}")
    log(f"anchor: pz {anchor['pz_i']:.4f} | strike zero "
        f"{anchor['pz_i_strike_zero']:.4f} (retention "
        f"{anchor['retention_zero']:.3f}) | mean "
        f"{anchor['pz_i_strike_mean']:.4f} ({anchor['retention_mean']:.3f}) "
        f"| CE {anchor['ce']:.4f}")

    # ---- three cycles
    cycles = []
    cur_sd = {k: v.clone() for k, v in sd0.items()}
    ce_pre = anchor["ce"]
    for c in range(1, N_CYCLES + 1):
        if c > 1:
            cooldown(COOLDOWN_S)
        gpu_now, polls = wait_for_gpu(log)
        dev = "cuda" if gpu_now else "cpu"
        gpu_msg = "PASS" if gpu_now else "FAIL -> CPU fallback"
        log(f"cycle {c}: gpu double-poll {gpu_msg} after {len(polls)} "
            f"poll(s); training on {dev}")
        gpu_before = gpu_status()

        # ---- ERASE (e044b D2 row-reset verbatim + G2/G3-analog gates)
        er_sd = {k: v.clone() for k, v in cur_sd.items()}
        er_sd["wte.weight"][zid] = 0.0
        er_sd["lm_head.weight"][zid] = 0.0
        n_diff, confined = 0, True
        for key in ("wte.weight", "lm_head.weight"):
            d = er_sd[key] != cur_sd[key]
            nd = int(d.sum().item())
            n_diff += nd
            rows = torch.nonzero(d)[:, 0].unique()
            if not (len(rows) == 1 and int(rows[0]) == zid
                    and nd == cfg.n_embd):
                confined = False
        others = all(torch.equal(er_sd[k], cur_sd[k])
                     for k in er_sd
                     if k not in ("wte.weight", "lm_head.weight"))
        netE = mk_cpu_net(er_sd)
        r_er = eval_bat(netE, bat_i, L, PRE - 1)
        ce_er = ce_val(netE, vx, vy)
        G2 = {"n_elements_changed": n_diff, "expected": 2 * cfg.n_embd,
              "confined_to_Z_rows": confined,
              "others_bit_identical": bool(others),
              "dce": ce_er - ce_pre, "dce_gate": DCE_GATE,
              "pass": bool(n_diff == 2 * cfg.n_embd and confined and others
                           and abs(ce_er - ce_pre) <= DCE_GATE)}
        G3 = {"nll": r_er["nll"], "acc": r_er["acc"],
              "off_bar": bool(r_er["nll"] > BAR_NLL
                              or r_er["acc"] < BAR_ACC),
              "e044b_natural_line_ref": ERASE_NLL_REF_E044B,
              "note": ("install-line re-base: strictly-off-bar gate "
                       "(smoke findings — D2 lands ~2.7 here, not >= 4.0 "
                       "as on the natural-name line; and a fixed margin "
                       "would confound the canalization signal itself)"),
              "pass": bool(r_er["nll"] > BAR_NLL
                           or r_er["acc"] < BAR_ACC)}
        log(f"cycle {c} erase: {n_diff} elems, dCE {ce_er - ce_pre:+.5f}, "
            f"spliced {r_er['nll']:.3f}/{r_er['acc']:.3f} -> "
            f"G2 {G2['pass']} G3 {G3['pass']}")
        assert G2["pass"], f"erase confinement FAILED: {G2}"
        if not G3["pass"]:
            # Genuine vacuity handler (the early-stop artifact route was
            # removed): the FULL D2 row-reset leaves the state AT bar —
            # the tasking is lesion-tolerant w.r.t. the address rows.
            # Record steps_to_bar=0, no re-learn, report honestly.
            log(f"cycle {c} ERASE VACUOUS: state still AT bar after D2 "
                f"({r_er['nll']:.3f}/{r_er['acc']:.3f}) — lesion-tolerant "
                f"tasking; recording steps_to_bar=0, no re-learn")
            ab = full_readout(cur_sd, f"cycle{c}_vacuous")
            cycles.append({
                "cycle": c, "device": dev,
                "gpu": {"double_poll_pass": bool(gpu_now),
                        "n_polls": len(polls), "before": gpu_before},
                "erase": {"G2_confined": G2, "G3_offbar": G3,
                          "post_erase_spliced": r_er,
                          "post_erase_ce": ce_er},
                "relearn": None, "erase_vacuous": True,
                "steps_to_bar": 0, "censored": False,
                "at_bar": ab, "final": ab})
            continue

        # ---- RE-LEARN (e044b's FULL step battery; bar scanned from grid)
        net = TinyGPT(cfg).to(dev)
        net.load_state_dict({k: v.to(dev) for k, v in er_sd.items()})

        def on_eval(sd_cpu, step, c=c):
            rec = {"step": step, "cycle": c,
                   "spliced_i": eval_bat(mk_cpu_net(sd_cpu), bat_i, L, PRE - 1),
                   "rows": row_probe(sd_cpu, zid, orig_w, orig_l, orig_wpe129)}
            if step in SPARSE_AT:
                m_ = mk_cpu_net(sd_cpu)
                rec["spliced_h"] = eval_bat(m_, bat_h, L, PRE - 1)
                rec["ce"] = ce_val(m_, vx, vy)
                rec["pz_i"] = pz_battery(m_, pz_i_ids, zid)
            sp = rec["spliced_i"]
            rec["bar"] = bool(sp["nll"] <= BAR_NLL and sp["acc"] >= BAR_ACC)
            log(f"  [c{c} s{step:3d}] spliced {sp['nll']:6.3f}/{sp['acc']:.3f}"
                + (f" ce {rec['ce']:.4f} pz {rec['pz_i']:.3f}"
                   if "ce" in rec else "")
                + f" |wte| {rec['rows']['wte_norm']:.3f}"
                  f" cos {rec['rows']['cos_wte']:+.3f}"
                + ("  BAR" if rec["bar"] else ""))
            return rec

        gen = torch.Generator().manual_seed(GEN_EXP)   # paired draws across cycles
        run = relearn(net, dev, win_i.to(dev), inst_mask.to(dev),
                      anchor_full.to(dev), train_ids, steps=EXPOSE_STEPS,
                      gen=gen, log=log, on_eval=on_eval)
        bar_step = run["bar_step"]
        final_sd = run["final_sd"]
        censored = bar_step is None
        if final_sd is None:                # no eval ever ran (cannot happen)
            final_sd = {k: v.detach().cpu().clone()
                        for k, v in net.state_dict().items()}
        at_bar_sd = run["at_bar_sd"] if run["at_bar_sd"] is not None \
            else final_sd
        gpu_after = gpu_status()
        log(f"cycle {c} re-learn done: bar at "
            f"{'step ' + str(bar_step) if bar_step else 'NOT REACHED (censored)'}"
            f", ran full {run['planned_steps']}-step battery, "
            f"train {run['train_s']}s / wall {run['wall_s']}s"
            f"{', stopped ' + str(run['stopped']) if run['stopped'] else ''}; "
            f"gpu {gpu_before['temp']:.0f}C->{gpu_after['temp']:.0f}C")

        # ---- AT-BAR + FINAL measurements (CPU)
        at_bar = full_readout(at_bar_sd, f"cycle{c}_at_bar"
                              if run["at_bar_sd"] is not None
                              else f"cycle{c}_final_as_bar")
        final = full_readout(final_sd, f"cycle{c}_final")
        ce_pre = final["ce"]
        cur_sd = {k: v.clone() for k, v in final_sd.items()}
        cyc = {"cycle": c, "device": dev,
               "gpu": {"double_poll_pass": bool(gpu_now),
                       "n_polls": len(polls), "before": gpu_before,
                       "after": gpu_after},
               "erase": {"G2_confined": G2, "G3_offbar": G3,
                         "post_erase_spliced": r_er, "post_erase_ce": ce_er},
               "relearn": {"run": {k: v for k, v in run.items()
                                   if k not in ("final_sd", "at_bar_sd")},
                           "traj": run["traj"]},
               "steps_to_bar": bar_step, "censored": censored,
               "erase_vacuous": False,
               "at_bar": at_bar, "final": final}
        cycles.append(cyc)
        log(f"cycle {c} at-bar: pz {at_bar['pz_i']:.4f} | strike zero "
            f"{at_bar['pz_i_strike_zero']:.4f} (retention "
            f"{at_bar['retention_zero']:.3f}, valid "
            f"{at_bar['retention_valid']}) | strike dNLL "
            f"{at_bar['strike_dNLL_zero']:+.3f} | cos_wte@bar "
            f"{at_bar['rows']['cos_wte']:+.3f}")
        log(f"cycle {c} final(s{run['planned_steps'] if not run['stopped'] else run['stopped'][1]}): "
            f"cos_wte {final['rows']['cos_wte']:+.3f} (regrow "
            f"{final['regrow_frac_wte']:.2f}, guard "
            f"{final['cos_guard_wte_pass']}) cos_lm "
            f"{final['rows']['cos_lm']:+.3f} | pz {final['pz_i']:.4f} | "
            f"CE {final['ce']:.4f}")
        del net
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---- adjudication (registered bars)
    s = [c["steps_to_bar"] for c in cycles]
    ratio32 = (s[2] / s[1]) if (s[2] is not None and s[1] not in (None, 0)) \
        else None
    ratio21 = (s[1] / s[0]) if (s[1] is not None and s[0] not in (None, 0)) \
        else None
    cos3 = cycles[2]["final"]["rows"]["cos_wte"]     # e044b headline form
    cos2 = cycles[1]["final"]["rows"]["cos_wte"]
    cos3_bar = cycles[2]["at_bar"]["rows"]["cos_wte"]
    cos2_bar = cycles[1]["at_bar"]["rows"]["cos_wte"]
    ret = [c["at_bar"]["retention_zero"] for c in cycles]
    dnlls = [c["at_bar"]["strike_dNLL_zero"] for c in cycles]
    any_vacuous = bool(any(c.get("erase_vacuous") for c in cycles))
    oscillation = bool(
        (s[1] is not None and s[0] is not None and s[1] < s[0])
        or (s[2] is not None and s[1] is not None and s[2] < s[1]))
    monotone_closure = bool(ratio32 is not None and ratio32 >= 1.0
                            and cos3 >= 0.6)
    falsified = bool((ratio32 is not None and ratio32 < 0.8) or cos3 < 0.5)
    if SMOKE:
        verdict = "SMOKE — not adjudicated"
    elif monotone_closure:
        verdict = ("MONOTONE CLOSURE — canalization stands (the groove "
                   "deepens): cycle3/cycle2 steps ratio >= 1 AND regrown "
                   "cos >= 0.6")
    elif falsified:
        verdict = ("CANALIZATION FALSIFIED — strips the read-policy "
                   "program's canal prior (ratio < 0.8 OR cos < 0.5)")
    else:
        verdict = ("MIXED — neither registered bar fires; texture "
                   "reported honestly")
    if oscillation and not SMOKE:
        verdict += (" | W001 OSCILLATION: a cycle was FASTER than its "
                    "predecessor — dreaming flag, report honestly")
    if any_vacuous and not SMOKE:
        verdict += (" | ERASE-VACUOUS cycle(s): steps clause degenerate "
                    "(tasking lesion-tolerant to the address rows) — "
                    "distinct from W001 dreaming, reported honestly")
    if any(c["censored"] for c in cycles) and not SMOKE:
        verdict += " | CENSORED-ON-STEPS noted (a cycle missed the bar)"
    adjudication = {
        "steps_to_bar": {"cycle1": s[0], "cycle2": s[1], "cycle3": s[2]},
        "ratio_cycle3_over_cycle2": ratio32,
        "ratio_cycle2_over_cycle1": ratio21,
        "cos_wte_cycle3_final": cos3, "cos_wte_cycle2_final": cos2,
        "cos_wte_cycle3_at_bar": cos3_bar, "cos_wte_cycle2_at_bar": cos2_bar,
        "cos_form_note": ("primary = END-of-cycle cos (e044b's headline "
                          "form: the scar cos is read after the full "
                          "re-learn battery); at-bar cos recorded alongside "
                          "(on this line the bar is met before the row "
                          "refills, so at-bar cos is systematically low)"),
        "retention_zero": {"cycle1": ret[0], "cycle2": ret[1],
                           "cycle3": ret[2], "cycle0_anchor":
                           anchor["retention_zero"]},
        "strike_dNLL_zero": {"cycle0_anchor": anchor["strike_dNLL_zero"],
                             "cycle1": dnlls[0], "cycle2": dnlls[1],
                             "cycle3": dnlls[2],
                             "note": ("battery-damage rider: the SAME fixed "
                                      "wpe[129]=0 strike's dNLL on the "
                                      "spliced install-60 battery at bar; "
                                      "SMALLER (or more negative) = more "
                                      "surgical-proof; added because at-bar "
                                      "p(Z) sits at the onset wall")},
        "battery_damage_monotone_decreasing": bool(
            dnlls[0] >= dnlls[1] >= dnlls[2]),
        "retention_valid": {"cycle0": anchor["retention_valid"],
                            "cycle1": cycles[0]["at_bar"]["retention_valid"],
                            "cycle2": cycles[1]["at_bar"]["retention_valid"],
                            "cycle3": cycles[2]["at_bar"]["retention_valid"],
                            "pz_floor": RETENTION_PZ_FLOOR},
        "proofness_monotone_c1_le_c2_le_c3": bool(ret[0] <= ret[1] <= ret[2]),
        "oscillation_w001": oscillation,
        "monotone_closure": monotone_closure,
        "canalization_falsified": falsified,
        "verdict": verdict,
    }
    log("=" * 74)
    log("THREE-CYCLE TABLE  (cos/regrow/CE at cycle END; pz/strike at BAR)")
    log(f"{'cycle':>5} {'steps':>6} {'cos@fin':>8} {'regrow':>7} "
        f"{'cos@bar':>8} {'pz@bar':>7} {'pz|zero':>7} {'retention':>9} "
        f"{'dNLL|zero':>9} {'CE@fin':>7}")
    log(f"{'0':>5} {'-':>6} {'-':>8} {'1.00':>7} {'-':>8} "
        f"{anchor['pz_i']:>7.4f} {anchor['pz_i_strike_zero']:>7.4f} "
        f"{anchor['retention_zero']:>9.3f} "
        f"{anchor['strike_dNLL_zero']:>9.3f} {anchor['ce']:>7.4f}")
    for c in cycles:
        ab, fn = c["at_bar"], c["final"]
        log(f"{c['cycle']:>5} {str(c['steps_to_bar']):>6} "
            f"{fn['rows']['cos_wte']:>8.3f} {fn['regrow_frac_wte']:>7.2f} "
            f"{ab['rows']['cos_wte']:>8.3f} "
            f"{ab['pz_i']:>7.4f} {ab['pz_i_strike_zero']:>7.4f} "
            f"{ab['retention_zero']:>9.3f} {ab['strike_dNLL_zero']:>9.3f} "
            f"{fn['ce']:>7.4f}")
    log(f"ratio c3/c2 {ratio32 if ratio32 is not None else 'None'} | "
        f"cos3(final) {cos3:+.3f} (cos@bar {cos3_bar:+.3f}) | retention "
        f"c1->c3 {ret[0]:.3f}->{ret[1]:.3f}->{ret[2]:.3f} | strike dNLL "
        f"c1->c3 {dnlls[0]:+.3f}->{dnlls[1]:+.3f}->{dnlls[2]:+.3f} | "
        f"oscillation {oscillation} | vacuous {any_vacuous}")
    log(f"REGISTERED VERDICT: {verdict}")

    # ---- outputs
    metrics = {
        "experiment": "e083_cycle3",
        "date": common.now_iso(),
        "smoke": SMOKE,
        "registered": REGISTERED,
        "protocol": {
            "install_ckpt": INSTALL_CK.name,
            "install_note": ("e082 gate0: B43 + e043 exposure verbatim "
                             "(seed 24331, 100 steps) — cycle-0 post-install"),
            "erase": "e044b D2 row-reset: wte[Z]=0, lm_head[Z]=0",
            "relearn": ("e044b Dmix: batch 8+24 (8 anchors + 16 random), "
                        "AdamW lr 1e-3 (0.9,0.95) wd 0.1 clip 1.0, cosine "
                        "1000/wu 100, token-weighted union CE, 300-step cap, "
                        "180s TRAINING-compute cap, full battery, at-bar "
                        "snapshot for the strike"),
            "bar": {"nll": BAR_NLL, "acc": BAR_ACC},
            "gen_exp_seed_per_cycle": GEN_EXP,
            "splice_rng": SPLICE_RNG, "install_mix": mix,
            "address_row": ADDR_ROW,
            "surgical_strike": ("e082 A1: wpe[129]<-0 (primary) / "
                                "<-mean(all 256 rows) (secondary), at bar"),
            "cos_min_norm_frac": COS_MIN_NORM_FRAC,
            "eval_steps": EVAL_STEPS,
        },
        "crosscheck_vs_e082": xcheck,
        "cycle0_anchor": anchor,
        "cycles": cycles,
        "adjudication": adjudication,
        "refs": {"e082": E082_REFS, "e044b": E044B_REF},
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": cfg_dict(cfg),
    }
    save_json(rd / "metrics.json", jsonable(metrics))

    # ---- plot: steps & cos & proofness vs cycle index (+ trajectories)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    cols = ["tab:blue", "tab:orange", "tab:green"]
    ax = axes[0, 0]
    for c, col in zip(cycles, cols):
        tr = c["relearn"]["traj"]
        xs = [0] + [r["step"] for r in tr]
        ys = [c["erase"]["post_erase_spliced"]["nll"]] + \
            [r["spliced_i"]["nll"] for r in tr]
        ax.plot(xs, ys, "o-", ms=3, color=col, label=f"cycle {c['cycle']}")
    ax.axhline(BAR_NLL, color="purple", ls=":", lw=1.2, label="bar NLL 1.0")
    ax.set_xlabel("re-learn step"); ax.set_ylabel("spliced install-60 NLL")
    ax.set_title("re-learn trajectories (post-erase NLL at step 0)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = axes[0, 1]
    xs = [1, 2, 3]
    vs = [c["steps_to_bar"] if c["steps_to_bar"] is not None else 0
          for c in cycles]
    bars_ = ax.bar(xs, vs, 0.55, color=cols, edgecolor="k", lw=0.5)
    for x, c, v in zip(xs, cycles, vs):
        ax.text(x, v + 0.4, str(c["steps_to_bar"]), ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(["cycle 1", "cycle 2", "cycle 3"])
    ax.set_ylabel("steps-to-bar")
    ax.set_title(f"steps-to-bar vs cycle — ratio c3/c2 "
                 f"{ratio32 if ratio32 is not None else float('nan'):.2f} "
                 f"(registered: >=1 closure, <0.8 falsified)")
    ax.grid(alpha=0.25, axis="y")
    ax = axes[1, 0]
    ax.plot(xs, [c["final"]["rows"]["cos_wte"] for c in cycles], "o-",
            color="crimson", label="cos_wte @cycle end (primary bar)")
    ax.plot(xs, [c["final"]["rows"]["cos_lm"] for c in cycles], "s--",
            color="gray", ms=4, label="cos_lm @cycle end (secondary)")
    ax.plot(xs, [c["at_bar"]["rows"]["cos_wte"] for c in cycles], "x:",
            color="pink", ms=5, label="cos_wte @bar (row unrefilled there)")
    ax.plot(xs, [c["final"]["rows"]["cos_wpe129"] for c in cycles], "^:",
            color="seagreen", ms=4, label="cos wpe[129] @end (address drift)")
    ax.axhline(0.6, color="seagreen", ls="--", lw=1.2, label="closure bar 0.6")
    ax.axhline(0.5, color="crimson", ls=":", lw=1.2, label="falsify bar 0.5")
    ax.set_xticks(xs); ax.set_xticklabels(["cycle 1", "cycle 2", "cycle 3"])
    ax.set_ylabel("cos(regrown row, cycle-0 original)")
    ax.set_title(f"regrown-row cos vs cycle — cycle3 cos_wte(final) "
                 f"{cos3:+.3f} (regrow "
                 f"{cycles[2]['final']['regrow_frac_wte']:.2f})")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
    ax = axes[1, 1]
    xs4 = np.arange(4)
    intact = [anchor["pz_i"]] + [c["at_bar"]["pz_i"] for c in cycles]
    struck = [anchor["pz_i_strike_zero"]] + \
        [c["at_bar"]["pz_i_strike_zero"] for c in cycles]
    struck_m = [anchor["pz_i_strike_mean"]] + \
        [c["at_bar"]["pz_i_strike_mean"] for c in cycles]
    ax.bar(xs4 - 0.25, intact, 0.25, color="lightsteelblue",
           edgecolor="k", lw=0.4, label="p(Z) intact")
    ax.bar(xs4, struck, 0.25, color="crimson", edgecolor="k", lw=0.4,
           label="p(Z) after wpe[129]=0 (fixed strike)")
    ax.bar(xs4 + 0.25, struck_m, 0.25, color="darkorange", edgecolor="k",
           lw=0.4, label="p(Z) after wpe[129]=mean")
    for x, v in zip(xs4, [anchor["retention_zero"]]
                    + [c["at_bar"]["retention_zero"] for c in cycles]):
        ax.text(x, 0.012, f"ret {v:.2f}", ha="center", fontsize=8)
    for x, v in zip(xs4, [anchor["strike_dNLL_zero"]]
                    + [c["at_bar"]["strike_dNLL_zero"] for c in cycles]):
        ax.text(x, 0.030, f"dNLL {v:+.2f}", ha="center", fontsize=7.5,
                color="darkred")
    ax.set_xticks(xs4)
    ax.set_xticklabels(["cycle 0\n(e082 install)", "cycle 1", "cycle 2",
                        "cycle 3"])
    ax.set_ylabel("install-60 mean p(Z) at readout")
    ax.set_title("surgical-proofness at bar (higher retention = more "
                 "surgical-proof)")
    ax.legend(fontsize=7.5)
    fig.suptitle("E083 — canalization cycle-3 (T037 construct 2): three "
                 "erase->re-learn cycles on the B43 install line — "
                 f"{verdict[:96]}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "cycle3.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
