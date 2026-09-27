"""E109 — the W003-registered CLS CONSOLIDATION head-to-head (REGISTERED).

WHY (W003, verbatim spirit): the lab's coordinate law (T043: "row 129 is the
address") says the installed fact dies with its address row (wpe-129).
Complementary learning systems (McClelland/McNaughton/Nadel) says enough
replay at VARIED positions consolidates content into the slow
position-independent store, surviving index loss. This is the head-to-head
between our own law and fifty years of memory theory, in one 2.7M char-LM.

BASE (registered): the e048_repro install-family machinery —
lab/e043_install.py protocol rebuilt VERBATIM on the runs/checkpoints/
e048_repro.pt checkpoint line, exactly as e065/e091 did (corpus seed 1337,
E43.SPLICE_RNG 24301 splice, install-60 / held-30, PRE=130, the 60 ctx130
onset battery). Gate G_INST: that battery's p(Z) must reproduce 0.556313
(|diff| < 0.005, e091 convention).

ARMS (registered):
  (a) CONSOLIDATION — fine-tune (<=180 s, lr 1e-3, batch 32) on install
      exposures at JITTERED positions: the same 60 name-windows' content
      placed at position offsets J = {-8, -4, 0, +4, +8} (pool of 300
      windows; the name onset decision / address read sits at wpe row
      129+j). The replay that CLS predicts transfers content to the
      position-independent store.
  (b) POSITION-LOCKED CONTROL — equal-step, equal-batch, equal-lr fine-tune
      on the ORIGINAL fixed-position (offset-0) exposures only. Controls
      for extra training per se.
  (c) BASELINE — no extra training (the installed net itself).

Fine-tune recipe (registered): batch 32 = 16 install windows (uniform from
the arm's pool) + 16 anchor windows (8 paired original host windows (the
e043 16-paired bank, e065 anchor) + 8 random train-corpus windows — the
e043 Dmix anchor composition scaled to batch 32); loss = e043's token-level
union CE over the 16x7 name-char targets and all 16x255 anchor tokens;
AdamW (0.9, 0.95) wd 0.1, CONSTANT lr 1e-3 (e065 house fine-tune
convention), clip 1.0; 300 steps (e044 relearn cap precedent) or the 180 s
wall cap, whichever first; seeds 10901 (a) / 10902 (b). In-loop CPU evals
every 25 steps: battery p(Z) at the ORIGINAL geometry + CE_R on 60
name-free val windows (e065 R_EVAL_SEED 26502 battery).

DELETION (registered): D129 — the D2-style subtractive row-zero from the
e042/e065 machinery applied to the address row: wpe.weight[129] := 0 in ALL
THREE arms, confinement-gated (exactly 192 elements changed, only that row,
everything else bit-identical). Batteries measured at the ORIGINAL geometry
(offset 0, readout at row 129) and at the JITTERED geometries (-8, -4, +4,
+8; readout at rows 121/125/133/137). Jittered battery construction =
e068's left-extension verbatim: ctx = train_text[p-PRE-j : p], readout p(Z)
at the new last position.

PRE-REGISTERED CONTINGENCY (written before compute, from e067's published
zero_arm on this exact net+battery): zeroing wpe row 129 alone drops
battery p(Z) by 0.3415 -> residual ~0.2148, i.e. arm (c) is EXPECTED NOT
to collapse to <= 0.05 under D129, so the operator bar-1 clause "(c)
collapses (<=0.05)" is expected unfireable under the literal registered
deletion. Therefore:
  - the operator bars are still evaluated verbatim under D129;
  - a pre-registered DELTA-RESCUE read adjudicates when they cannot fire:
    rescue_X = max over geometries of [post_D129(X) - post_D129(c)];
    rescue >= +0.15 (the bar gap 0.20 - 0.05) = a survival signal
    substantially ABOVE the no-extra-training residual:
      rescue_a >= +0.15 and rescue_b < +0.15 -> CONSOLIDATION-SIGNAL;
      both >= +0.15 -> TRAINING-MASS-SIGNAL; both < +0.15 -> NO-RESCUE
      (one-row-law-consistent); else MIXED;
  - a SECONDARY report-only deletion D0129 (wpe rows {0, 129} both zeroed —
    the full-index loss under which e067 predicts (c) ~ 0.011 <= 0.05) is
    measured identically as the collapse-calibrated variant. NOTE: row 0 is
    generic scaffolding (T043) present at position 0 of EVERY window, so
    D0129 is expected to suppress all arms; it is reported, never the
    primary bar.
  - STRICT-CLS read (report-only): post-D129 p(Z) of arm (a) at geometry 0
    specifically — expression THROUGH the deleted coordinate, vs expression
    at other (trained) coordinates which may be mere re-addressing.

REGISTERED BARS (operator, verbatim; adjudicated per deletion in this
order — (b) surviving alongside (a) converts a would-be consolidation win
into the training-mass verdict, per the operator's bar-3 clause):
  c collapses (max_g post(g) <= 0.05):
    a survives (max_g >= 0.20) AND b survives          -> TRAINING-MASS EFFECT
    a survives AND NOT b survives                      -> CONSOLIDATION WINS
    else (a collapses like c and b adds nothing)       -> ONE-ROW LAW STANDS
  c does NOT collapse                                  -> registered bars
    cannot fire as written; report honestly and read
    the DELTA-RESCUE adjudication above.
Operationalizations: "collapses like (c)" = max_g [post_X(g)-post_c(g)] <=
+0.05; "b adds nothing" = max_g [post_b(g)-post_c(g)] <= +0.05.

Secondaries (report-only, never barred on): held-30 battery at every
geometry; wpe row probes (norm + cos-to-original for rows {0, 121, 125,
129, 133, 137}) per arm — address regrowth at jittered rows, e044-style.

COMPUTE ENVELOPE: GPU-PERMITTED but thermally gated — every fine-tune
launch gates on gpu_ok() (util <= 85% / temp <= 80C / mem <= 85%) on two
polls 10 s apart; cooldown(120) between the two fine-tunes; each fine-tune
<= 180 s wall including its in-loop evals; batch 32. ALL EVALS CPU-SIDE
(the fine-tuned state dict is pulled to a CPU eval net).

Honesty notes: (i) D129 zeroes the row GLOBALLY — jittered batteries with
j > 0 contain row 129 as a CONTEXT row (zeroed), j < 0 batteries never
touch it; (ii) training windows are full 256-token windows, so every
training step of every arm saw wpe row 0 — no arm trained without it;
(iii) arm (a)'s jittered replay re-installs addresses at rows
121/125/133/137, so post-D129 expression at those geometries can be
re-addressing rather than position-independent consolidation — the
STRICT-CLS geometry-0 read exists to separate them.

Outputs: runs/e109/{metrics.json, consolidation.png}
(PNG: pre/post-deletion battery p(Z) per arm x geometry).
No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).

Run:  cd lab && python e109_consolidation.py
"""
from __future__ import annotations

import copy
import random
import re
import time

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # GPU for the fine-tunes only

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

import common                                         # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,     # noqa: E402
                    gpu_status, run_dir, save_json)
import e043_install as E43                            # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib.pyplot as plt                       # noqa: E402

SMOKE = os.environ.get("E109_SMOKE") == "1"
DEV = common.DEVICE                                   # cuda (fine-tunes)
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
INSTALLED_CK = "e048_repro.pt"

JITTERS = (-8, -4, 0, 4, 8)       # the registered jitter set (includes 0)
GEO_ORDER = [-8, -4, 0, 4, 8]     # display order (original geometry in middle)

# fine-tune envelope (registered)
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
FT_TIME_CAP = 180.0
EVAL_EVERY = 25 if not SMOKE else 2
NAME_BS = 16                      # install windows per step
ANCH_BS = 16                      # anchor windows per step (8 paired + 8 random)
CONS_SEED, LOCK_SEED = 10901, 10902

# batteries / guards
R_EVAL_SEED = 26502               # e065's CE_R eval-bank seed (verbatim)
G_INST_REF = 0.556313             # e065 arms_battery no_removal p_z_mean
G_INST_TOL = 0.005                # e091 convention

# registered bars
BAR_SURVIVE = 0.20
BAR_COLLAPSE = 0.05
DELTA_RESCUE = 0.15               # = 0.20 - 0.05 (the bar gap)

OPERATOR_BARS = {
    "bar1": "(a) post-deletion p(Z) >= 0.20 at any geometry while (c) collapses "
            "(<=0.05) => CONSOLIDATION WINS",
    "bar2": "(a) collapses like (c) and (b) adds nothing => ONE-ROW LAW STANDS",
    "bar3": "(a) and (b) both survive => TRAINING-MASS EFFECT",
}

trims: list[str] = []
deviations: list[str] = [
    "Constant lr 1e-3 (e065 house fine-tune convention) instead of e043's cosine — "
    "these are fine-tunes of an installed net, not fresh installs; schedule decay "
    "would confound the training-mass comparison.",
    "Anchor composition scaled to batch 32: 8 paired originals (of the e043 16) + "
    "8 random train-corpus windows per step (e043 Dmix was 16 paired + 32 random at "
    "batch 64); token-level union loss kept verbatim.",
    "300 steps per fine-tune (e044 relearn cap precedent), 180 s wall cap, evals "
    "every 25 steps (in-loop, CPU-side).",
    "Pre-registered contingency (e067 zero_arm): arm (c) post-D129 expected ~0.215 "
    "> 0.05, so the operator bar-1 '(c) collapses' clause is expected unfireable "
    "under the literal D129 deletion; a pre-registered DELTA-RESCUE read "
    "(+0.15 = the bar gap) adjudicates instead, and a report-only D0129 "
    "(rows {0,129}) secondary calibrates full collapse.",
    "held-30 battery, wpe row probes, and the STRICT-CLS geometry-0 read are "
    "report-only secondaries.",
]


# ------------------------------------------------------------------ gpu guard

def gate_launch(tag: str) -> dict:
    """Hard gate (e065 verbatim): launch only when gpu_ok() (util<=85%,
    temp<=80C, mem<=85%) holds on two polls 10 s apart."""
    while True:
        if gpu_ok():
            time.sleep(10)
            s2 = gpu_status()
            if gpu_ok():
                log(f"[gpu] launch '{tag}' ok (util {s2['util']:.0f}% temp "
                    f"{s2['temp']:.0f}C mem {s2['mem_used']:.0f}/"
                    f"{s2['mem_total']:.0f}MB)")
                return s2
        time.sleep(10)


# ------------------------------------------------------------------ instruments

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30,
                 keep_per_ctx=False) -> dict:
    """e068-style battery on CPU: p(Z) at the last position over contexts."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    out = {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
           "std_pz": float(p.std()),
           "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
           "frac_argmax_z": amax / ids.shape[0]}
    if keep_per_ctx:
        out["pz_per_ctx"] = [float(v) for v in p.tolist()]
    return out


@torch.no_grad()
def ce_fixed_cpu(net: TinyGPT, x, y, bs=64) -> float:
    """e065 ce_fixed (CPU)."""
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    """e065 val_windows verbatim: name-free val-split windows."""
    g = torch.Generator().manual_seed(seed)
    out_x, out_y = [], []
    tries = 0
    while len(out_x) < n and tries < 500 * n:
        i = int(torch.randint(len(val_ids) - block - 1, (1,), generator=g))
        txt = val_text[i: i + block + 1]
        tries += 1
        if "ZEPHYRA" in txt or "ZEPH" in txt:
            continue
        out_x.append(val_ids[i: i + block])
        out_y.append(val_ids[i + 1: i + 1 + block])
    return torch.stack(out_x), torch.stack(out_y)


def deleted_wpe(sd: dict, rows: tuple[int, ...]) -> tuple[dict, dict]:
    """D2-style subtractive row-zero on wpe rows; confinement gate (e065
    G_SURG convention: exact element count, row confinement, everything
    else bit-identical)."""
    out = {k: v.clone() for k, v in sd.items()}
    for r in rows:
        out["wpe.weight"][r] = 0.0
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    gate = {"rows": list(rows), "n_elements_changed": n,
            "expected": len(rows) * sd["wpe.weight"].shape[1],
            "changed_rows": changed_rows,
            "confined": bool(changed_rows == sorted(rows)),
            "others_bit_identical": bool(others),
            "pass": bool(n == len(rows) * sd["wpe.weight"].shape[1]
                         and changed_rows == sorted(rows) and others)}
    return out, gate


# ------------------------------------------------------------------ fine-tune

def finetune_arm(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                 pool_mask: torch.Tensor, anchor: torch.Tensor,
                 train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                 seed: int):
    """Registered fine-tune: batch 32 = 16 install windows from pool + 16
    anchors (8 paired + 8 random); e043 token-level union CE; constant lr
    1e-3 AdamW (0.9,0.95) wd 0.1 clip 1.0; <=300 steps / <=180 s. In-loop
    CPU evals every 25 steps (original-geometry battery + CE_R)."""
    gate_launch(tag)
    net = copy.deepcopy(net0).to(DEV)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    traj, t_start = [], time.time()
    step = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, FT_STEPS + 1):
        ix = torch.randint(n_pool, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                           generator=gen)
        nw = pool_x[ix].to(DEV)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])], 0).to(DEV)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(NAME_BS + ANCH_BS, x.shape[1], dtype=torch.bool, device=DEV)
        m[:NAME_BS] = pool_mask[ix].to(DEV)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              reduction="none").view(x.shape[0], x.shape[1])
        nm = nll[:NAME_BS][m[:NAME_BS]]
        cm = nll[NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == FT_STEPS or \
                (time.time() - t_start) > FT_TIME_CAP:
            evl.load_state_dict({k: v.detach().cpu().clone()
                                 for k, v in net.state_dict().items()})
            evl.eval()
            bz = battery_cell(evl, f_eval_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "p_z_mean": bz["mean_pz"],
                         "frac_argmax_z": bz["frac_argmax_z"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p_z {bz['mean_pz']:.4f} argmaxZ "
                f"{bz['frac_argmax_z']:.2f} CE_R {ce_r:.4f} "
                f"({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            break
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e109_smoke" if SMOKE else "e109")
    log(f"E109 CLS CONSOLIDATION head-to-head (W003; smoke={SMOKE}) -> {rd}")

    # ---------------- protocol rebuild (e065/e091 verbatim)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30 (SPLICE_RNG {E43.SPLICE_RNG})")

    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)

    # ---------------- jittered install windows (training pools) + batteries
    # window at offset j: pre = train_ids[p-PRE-j : p] (len 130+j), then the
    # name, then post = host continuation (len 119-j) -> exactly 256 tokens.
    # Name targets (y-space) at columns [PRE-1+j, PRE-1+j+7).
    jit_x, jit_mask = {}, {}
    for j in JITTERS:
        wins = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            if len(w) != BLOCK:
                raise RuntimeError(f"window len {len(w)} != {BLOCK} at offset {j}")
            wins.append(w)
        jit_x[j] = torch.stack(wins)
        m = torch.zeros(len(wins), BLOCK - 1, dtype=torch.bool)
        m[:, PRE - 1 + j: PRE - 1 + j + len(NAME)] = True
        jit_mask[j] = m
    pool_a_x = torch.cat([jit_x[j] for j in JITTERS])          # (300, 256)
    pool_a_mask = torch.cat([jit_mask[j] for j in JITTERS])
    pool_b_x, pool_b_mask = jit_x[0], jit_mask[0]              # offset-0 only
    log(f"jitter pools: arm-a {tuple(pool_a_x.shape)} (offsets {list(JITTERS)}), "
        f"arm-b {tuple(pool_b_x.shape)}; anchor bank 16 paired originals")

    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])       # e065 anchor bank

    # batteries per geometry: ctx = train_text[p-PRE-j : p], readout at the
    # last position (row 129+j). install60 primary, held30 secondary.
    bat_ids = {}
    for j in GEO_ORDER:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    f_eval_ids = bat_ids[(0, "install60")]                     # the G_INST battery

    # CE_R eval bank (e065 verbatim, seed 26502)
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- base net + instrument gate
    net0 = load_cpu(E43.REPO / "runs" / "checkpoints" / INSTALLED_CK)
    evl = copy.deepcopy(net0)
    bz0 = battery_cell(evl, f_eval_ids, zid, keep_per_ctx=True)
    G_INST = {"battery_pz": bz0["mean_pz"], "ref": G_INST_REF,
              "tol": G_INST_TOL,
              "pass": bool(abs(bz0["mean_pz"] - G_INST_REF) < G_INST_TOL)}
    log(f"G_INST installed battery p(Z) {bz0['mean_pz']:.6f} "
        f"(ref {G_INST_REF}): {'PASS' if G_INST['pass'] else 'FAIL'}")
    if not G_INST["pass"]:
        raise RuntimeError("instrument broken vs e065/e091 (net/battery mismatch)")
    ce_r0 = ce_fixed_cpu(evl, *r_eval_xy)
    log(f"CE_R (60 name-free val windows): {ce_r0:.4f}")

    # ---------------- arm definitions + fine-tunes (GPU, thermally gated)
    arm_order = ["a_consolidation", "b_position_locked", "c_baseline"]
    arms = {"c_baseline": {"sd": {k: v.clone() for k, v in
                                  net0.state_dict().items()},
                           "traj": [], "steps_ran": 0, "seed": None,
                           "desc": "no extra training (the installed net)"}}

    log("ARM A (consolidation): fine-tune on jittered-position exposures "
        f"{list(JITTERS)} (<=180s, lr 1e-3, batch 32)")
    a = finetune_arm("a_consolidation", net0, pool_a_x, pool_a_mask, anchor,
                     train_ids, r_eval_xy, f_eval_ids, zid, CONS_SEED)
    a["desc"] = f"jittered replay {list(JITTERS)}, {a['steps_ran']} steps"
    arms["a_consolidation"] = a

    log("[thermal] cooldown(120) between fine-tunes")
    cooldown(120.0)

    log("ARM B (position-locked control): equal budget, offset-0 exposures only")
    b = finetune_arm("b_position_locked", net0, pool_b_x, pool_b_mask, anchor,
                     train_ids, r_eval_xy, f_eval_ids, zid, LOCK_SEED)
    b["desc"] = f"position-locked replay (offset 0 only), {b['steps_ran']} steps"
    arms["b_position_locked"] = b

    # ---------------- measurement phase (ALL CPU): arm x deletion x geometry
    wpe_probes = {}
    for an in arm_order:
        sd = arms[an]["sd"]
        w0 = sd["wpe.weight"]
        orig = net0.state_dict()["wpe.weight"]
        rows = {}
        for r in (0, 121, 125, 129, 133, 137):
            rows[str(r)] = {"norm": float(w0[r].norm()),
                            "cos_to_e048": float(F.cosine_similarity(
                                w0[r], orig[r], dim=0))}
        wpe_probes[an] = rows
    log("wpe row probes (norm/cos vs e048_repro): "
        + " | ".join(f"{an}: r129 {wpe_probes[an]['129']['norm']:.2f}/"
                     f"{wpe_probes[an]['129']['cos_to_e048']:+.2f}, "
                     f"r133 {wpe_probes[an]['133']['norm']:.2f}/"
                     f"{wpe_probes[an]['133']['cos_to_e048']:+.2f}"
                     for an in arm_order))

    deletions = {"none": (), "d129": (129,), "d0129": (0, 129)}
    table = {}                       # (arm, deletion, geometry, battery) -> cell
    gates_surg = {}
    evl = copy.deepcopy(net0)
    for dl_name, rows in deletions.items():
        for an in arm_order:
            sd_del, gate = deleted_wpe(arms[an]["sd"], rows)
            gates_surg[f"{an}__{dl_name}"] = gate
            if not gate["pass"]:
                raise RuntimeError(f"deletion gate FAILED {an}/{dl_name}: {gate}")
            evl.load_state_dict(sd_del)
            for j in GEO_ORDER:
                for bt in ("install60", "held30"):
                    table[(an, dl_name, j, bt)] = battery_cell(
                        evl, bat_ids[(j, bt)], zid,
                        keep_per_ctx=(bt == "install60"))
        log(f"deletion {dl_name:5s} measured (rows {list(rows) or '—'}): "
            + " | ".join(f"{an} max_g install60 "
                         f"{max(table[(an, dl_name, g, 'install60')]['mean_pz'] for g in GEO_ORDER):.3f}"
                         for an in arm_order))

    # ---------------- adjudication (registered)
    def post(an, dl, g, bt="install60"):
        return table[(an, dl, g, bt)]["mean_pz"]

    def adjudicate(dl):
        c_max = max(post("c_baseline", dl, g) for g in GEO_ORDER)
        a_max = max(post("a_consolidation", dl, g) for g in GEO_ORDER)
        b_max = max(post("b_position_locked", dl, g) for g in GEO_ORDER)
        resc_a = max(post("a_consolidation", dl, g) - post("c_baseline", dl, g)
                     for g in GEO_ORDER)
        resc_b = max(post("b_position_locked", dl, g) - post("c_baseline", dl, g)
                     for g in GEO_ORDER)
        c_collapses = c_max <= BAR_COLLAPSE
        if c_collapses:
            if a_max >= BAR_SURVIVE and b_max >= BAR_SURVIVE:
                fired = "TRAINING_MASS_EFFECT"
            elif a_max >= BAR_SURVIVE:
                fired = "CONSOLIDATION_WINS"
            else:
                fired = "ONE_ROW_LAW_STANDS"
            delta_branch = None
        else:
            fired = "REGISTERED_BARS_UNFIREABLE_c_did_not_collapse"
            if resc_a >= DELTA_RESCUE and resc_b >= DELTA_RESCUE:
                delta_branch = "TRAINING_MASS_SIGNAL"
            elif resc_a >= DELTA_RESCUE:
                delta_branch = "CONSOLIDATION_SIGNAL"
            elif resc_a < DELTA_RESCUE and resc_b < DELTA_RESCUE:
                delta_branch = "NO_RESCUE_BEYOND_BASELINE"
            else:
                delta_branch = "MIXED"
        best_g_a = max(GEO_ORDER, key=lambda g: post("a_consolidation", dl, g))
        return {"c_max": c_max, "a_max": a_max, "b_max": b_max,
                "rescue_a": resc_a, "rescue_b": resc_b,
                "c_collapses": bool(c_collapses), "fired": fired,
                "delta_branch": delta_branch,
                "a_best_geometry": best_g_a,
                "a_post_at_best": post("a_consolidation", dl, best_g_a),
                "c_post_at_best": post("c_baseline", dl, best_g_a)}

    verdicts = {dl: adjudicate(dl) for dl in ("d129", "d0129")}
    v = verdicts["d129"]
    strict_cls = {
        "a_post_d129_geom0": post("a_consolidation", "d129", 0),
        "b_post_d129_geom0": post("b_position_locked", "d129", 0),
        "c_post_d129_geom0": post("c_baseline", "d129", 0),
        "note": "expression THROUGH the deleted coordinate (true CLS "
                "consolidation) vs expression at other trained coordinates "
                "(possible re-addressing)",
    }
    headline = (f"D129 (registered deletion): {v['fired']}"
                + (f" -> delta-read {v['delta_branch']}"
                   if v["delta_branch"] else "")
                + f" | D0129 (secondary): {verdicts['d0129']['fired']}")
    log("=" * 78)
    log(f"E109 VERDICT: {headline}")
    log(f"  D129: c_max {v['c_max']:.3f} a_max {v['a_max']:.3f} (at geom "
        f"{v['a_best_geometry']:+d}) b_max {v['b_max']:.3f} | rescue_a "
        f"{v['rescue_a']:+.3f} rescue_b {v['rescue_b']:+.3f}")
    log(f"  STRICT-CLS (geometry 0, post-D129): a {strict_cls['a_post_d129_geom0']:.3f} "
        f"b {strict_cls['b_post_d129_geom0']:.3f} c {strict_cls['c_post_d129_geom0']:.3f}")
    log("=" * 78)

    # ---------------- arm table (install60) printout
    log("ARM TABLE — battery p(Z) (install-60), rows = arm x deletion, "
        "cols = geometry:")
    hdr = f"{'arm/deletion':24s}" + "".join(f"{g:+7d}" for g in GEO_ORDER)
    log(hdr)
    for dl in ("none", "d129", "d0129"):
        for an in arm_order:
            row = f"{an + '/' + dl:24s}" + "".join(
                f"{post(an, dl, g):7.3f}" for g in GEO_ORDER)
            log(row)

    # ---------------- outputs
    metrics = {
        "experiment": "e109_consolidation",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "W003 (CLS systems-consolidation head-to-head) + "
                        "operator task registration (bars verbatim below); "
                        "docstring written before compute",
        "operator_registered_bars": OPERATOR_BARS,
        "adjudication_rule": {
            "survive": f"max over geometries post-deletion p(Z) >= {BAR_SURVIVE}",
            "collapse": f"max over geometries <= {BAR_COLLAPSE}",
            "collapses_like_c / adds_nothing": "max_g [post_X(g)-post_c(g)] <= +0.05",
            "order": "c collapses: (a&b survive -> TRAINING-MASS) elif (a "
                     "survives -> CONSOLIDATION WINS) else ONE-ROW LAW STANDS; "
                     "c does not collapse: bars unfireable, pre-registered "
                     f"delta-rescue read (>= +{DELTA_RESCUE}) adjudicates",
        },
        "net": f"runs/checkpoints/{INSTALLED_CK} (e043 Dmix@s400 install line)",
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "jitters": list(JITTERS),
                     "jitter_construction": "e068 left-extension/retraction: pre "
                                            "= train_ids[p-PRE-j:p] (len 130+j), "
                                            "post = host continuation (119-j); "
                                            "name targets y-cols [129+j,136+j)",
                     "battery_construction": "ctx = train_text[p-PRE-j:p], "
                                             "readout p(Z) at last position "
                                             "(wpe row 129+j)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_INST": G_INST,
                  "G_SURG": gates_surg,
                  "G_CE_R0": ce_r0},
        "fine_tune": {"lr": FT_LR, "steps": FT_STEPS, "time_cap_s": FT_TIME_CAP,
                      "batch": f"{NAME_BS} install + {ANCH_BS} anchor "
                               f"({ANCH_BS // 2} paired + {ANCH_BS // 2} random)",
                      "loss": "e043 token-level union CE",
                      "optimizer": "AdamW (0.9,0.95) wd 0.1 clip 1.0 constant lr",
                      "seeds": {"a": CONS_SEED, "b": LOCK_SEED}},
        "arms": {an: {"desc": arms[an]["desc"], "steps_ran": arms[an]["steps_ran"],
                      "seed": arms[an]["seed"], "traj": arms[an]["traj"]}
                 for an in arm_order},
        "wpe_row_probes": wpe_probes,
        "battery_table": {f"{an}__{dl}__g{g:+d}__{bt}": table[(an, dl, g, bt)]
                          for an in arm_order for dl in deletions
                          for g in GEO_ORDER for bt in ("install60", "held30")},
        "verdicts": {"d129_primary": verdicts["d129"],
                     "d0129_secondary_report_only": verdicts["d0129"],
                     "strict_cls_geom0": strict_cls,
                     "headline": headline},
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net0.num_params()),
                   "finetune_device": str(DEV), "eval_device": "cpu",
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: consolidation.png
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    colors = {"a_consolidation": "crimson", "b_position_locked": "steelblue",
              "c_baseline": "tab:gray"}
    arm_lbl = {"a_consolidation": "(a) CONSOLIDATION\n(jittered replay)",
               "b_position_locked": "(b) POSITION-LOCKED\n(equal steps, offset 0)",
               "c_baseline": "(c) BASELINE\n(no extra training)"}
    panels = [("none", "PRE-deletion battery p(Z)", axes[0, 0]),
              ("d129", "POST-D129 (registered: wpe row 129 := 0)", axes[0, 1]),
              ("d0129", "POST-D0129 (secondary: rows {0,129} := 0)", axes[1, 0])]
    xs = np.arange(len(GEO_ORDER))
    bw = 0.26
    for dl, ttl, ax in panels:
        for k, an in enumerate(arm_order):
            vals = [post(an, dl, g) for g in GEO_ORDER]
            ax.bar(xs + (k - 1) * bw, vals, bw, color=colors[an],
                   label=arm_lbl[an], edgecolor="k", linewidth=0.4)
            for x, vv in zip(xs + (k - 1) * bw, vals):
                ax.text(x, vv + 0.006, f"{vv:.3f}", ha="center", fontsize=6.6,
                        rotation=90, va="bottom")
        ax.axhline(BAR_SURVIVE, color="seagreen", ls="--", lw=1.1,
                   label=f"survive bar {BAR_SURVIVE}")
        ax.axhline(BAR_COLLAPSE, color="gray", ls=":", lw=1.1,
                   label=f"collapse bar {BAR_COLLAPSE}")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{g:+d}\n(row {129 + g})" for g in GEO_ORDER],
                           fontsize=8)
        ax.set_ylabel("battery p(Z) at onset (install-60)")
        top = max(post(an, dl, g) for an in arm_order for g in GEO_ORDER)
        ax.set_ylim(0, max(0.85, top * 1.25))
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=6.8)

    ax = axes[1, 1]
    for an in ("a_consolidation", "b_position_locked"):
        tr = arms[an]["traj"]
        if tr:
            ax.plot([t["step"] for t in tr], [t["p_z_mean"] for t in tr], "o-",
                    color=colors[an], ms=4, label=f"{an} battery p(Z) @orig")
    ax.axhline(G_INST_REF, color="k", ls="-.", lw=0.9,
               label=f"e048_repro ref {G_INST_REF:.3f}")
    ax.set_xlabel("fine-tune step")
    ax.set_ylabel("battery p(Z) (original geometry)")
    ax2 = ax.twinx()
    for an, sty in (("a_consolidation", ":"), ("b_position_locked", "-.")):
        tr = arms[an]["traj"]
        if tr:
            ax2.plot([t["step"] for t in tr], [t["ce_r"] for t in tr], sty,
                     color=colors[an], alpha=0.7,
                     label=f"{an} CE_R")
    ax2.axhline(ce_r0, color="k", lw=0.5)
    ax2.set_ylabel("CE_R (nats)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.8)
    vtxt = (f"D129: {verdicts['d129']['fired']}"
            + (f"\ndelta-read: {verdicts['d129']['delta_branch']}"
               if verdicts["d129"]["delta_branch"] else "")
            + f"\nD0129 (sec): {verdicts['d0129']['fired']}"
            + f"\nSTRICT-CLS geom0 post-D129: a "
              f"{strict_cls['a_post_d129_geom0']:.3f} / b "
              f"{strict_cls['b_post_d129_geom0']:.3f} / c "
              f"{strict_cls['c_post_d129_geom0']:.3f}")
    ax.set_title("fine-tune trajectories + verdict", fontsize=10)
    ax.text(0.02, 0.02, vtxt, transform=ax.transAxes, fontsize=7.4, va="bottom",
            family="monospace",
            bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))

    fig.suptitle(f"E109 — W003 CLS consolidation head-to-head: {headline}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "consolidation.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'consolidation.png'}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
