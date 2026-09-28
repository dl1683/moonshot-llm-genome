"""E113 — ALL-ADDRESSES DELETION: body-stored or address-migrated? (REGISTERED)

WHY (T064 AGENT CORRECTION, direct follow-through): e109's consolidated net
(arm a: 300-step jittered replay, seed 10901) grew NEW address rows
121/125/133/137 (norms 0.25-0.29 vs baseline 0.18-0.20, cos-to-e048
0.76-0.85) on top of the original row 129 — a re-addressing component.
e109's D0129 secondary (rows {0,129} := 0) killed everything (0.001-0.043),
but that deletion CONFLATES address loss with window-scaffold loss because
wpe row 0 is the generic load-bearing anchor of EVERY window (T043/e071).
The open question after the correction: after jittered-replay consolidation,
is the fact stored in ITS addresses (old 129 + new 121/125/133/137) or in
the body/content rows (everything except those five)?

DESIGN (registered): rebuild arm (a) of e109 EXACTLY — same recipe, same
seed 10901, 300 steps, jittered replay {-8,-4,0,+4,+8}, batch 16 install +
16 anchor (8 paired + 8 random), e043 token-level union CE, AdamW (0.9,0.95)
wd 0.1 constant lr 1e-3 clip 1.0 — on CPU. REPRODUCTION GATE (G_REPRO): the
post-none install-60 battery table must reproduce e109's (tol 0.05/cell):
  g-8 0.9924 | g-4 0.9496 | g+0 0.7761 | g+4 0.9849 | g+8 0.9855
(the operator's 0.99/0.95/0.78/0.99/0.99 table at 2 dp).

Then THREE deletion arms on the consolidated net (subtracted row-zero,
D2-style, confinement-gated; row 0 NEVER touched):
  (i)   D-ALL-ITS-ADDRESSES: zero rows {121,125,129,133,137}
  (ii)  D-NEW-ONLY:          zero rows {121,125,133,137} (row 129 intact)
  (iii) D129:                zero row {129} (replicates e109's 0.909 control)

Battery p(Z) at all five geometries (-8,-4,0,+4,+8; readout rows
121/125/129/133/137) + held-30 secondary.

REGISTERED BARS (operator, verbatim):
  (i) survives (>= 0.20 at most geometries)            => BODY-STORED
      (true content consolidation — the fact left the address system
       entirely; CLS fully right at this scale after all);
  (i) collapses (<= 0.05) while (iii) stays high       => ADDRESS-MIGRATED
      (the fact merely moved house — five rows now carry it; consolidation
       is re-indexing);
  intermediate (0.05-0.20)                              => report texture.
Operationalizations: "survives at most geometries" = post-(i) p(Z) >= 0.20
at >= 3 of 5 geometries (install-60 primary); "collapses" = max over
geometries <= 0.05; "(iii) stays high" = max_g post-(iii) >= 0.20.
Held-30 mirrors are report-only.

wpe-probe norms of rows {121,125,129,133,137} pre- and post-each-deletion
are reported (deleted rows are zero by construction and nothing retrains
after a deletion — eval-time deletions cannot grow rows; the "does deleting
addresses grow yet more?" dynamics question needs a post-deletion replay
and is NOT registered here — noted for the queue).

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import;
torch.set_num_threads(8)); single run target <= 30 min. e109's 180 s wall
cap was a GPU thermal guard — raised to 1500 s so the registered 300 steps
complete on CPU; GPU gate_launch/cooldown machinery is void without a GPU.

Outputs: runs/e113/{metrics.json, all_addresses.png}.
No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).

Run:  cd lab && python e113_all_addresses.py        (E113_SMOKE=1 for smoke)
"""
from __future__ import annotations

import copy
import random
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (operator)

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(8)                              # 8 threads max (operator)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E113_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
INSTALLED_CK = "e048_repro.pt"

JITTERS = (-8, -4, 0, 4, 8)       # e109's registered jitter set
GEO_ORDER = [-8, -4, 0, 4, 8]     # display order (original geometry in middle)
ADDR_ROWS = (121, 125, 129, 133, 137)   # 129 + the grown rows (T064)

# fine-tune envelope (registered — e109 arm (a) verbatim except CPU/caps)
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
FT_TIME_CAP = 1500.0              # CPU safety cap (e109's 180 s was GPU thermal)
EVAL_EVERY = 25 if not SMOKE else 2
NAME_BS = 16                      # install windows per step
ANCH_BS = 16                      # anchor windows per step (8 paired + 8 random)
CONS_SEED = 10901                 # e109 arm (a) seed VERBATIM

# batteries / guards
R_EVAL_SEED = 26502               # e065's CE_R eval-bank seed (verbatim)
G_INST_REF = 0.556313             # e065 arms_battery no_removal p_z_mean
G_INST_TOL = 0.005                # e091 convention

# e109 arm (a) reference cells (runs/e109/metrics.json, full precision) —
# the reproduction gate for this rebuild
E109_REF_NONE = {-8: 0.9924036860466003, -4: 0.9496323466300964,
                 0: 0.776076078414917, 4: 0.9848979115486145,
                 8: 0.9854525923728943}
E109_REF_D129_G0 = 0.90878826379776       # the "0.909 control"
G_REPRO_TOL = 0.05                        # per-cell tolerance

# registered bars
BAR_SURVIVE = 0.20
BAR_COLLAPSE = 0.05

REGISTERED_BARS = {
    "body_stored": "arm (i) D-all-addresses: p(Z) >= 0.20 at most geometries "
                   "(>= 3 of 5, install-60) => BODY-STORED",
    "address_migrated": "arm (i) collapses (max_g <= 0.05) while arm (iii) "
                        "D129 stays high (max_g >= 0.20) => ADDRESS-MIGRATED",
    "intermediate": "arm (i) lands 0.05-0.20 (or mixed across geometries) "
                    "=> report texture",
    "operationalizations": "survives = >= 3/5 geometries >= 0.20; collapses "
                           "= max over geometries <= 0.05; (iii) high = "
                           "max_g >= 0.20; held-30 mirrors report-only",
}

DELETIONS = {                     # row 0 is NEVER touched in any arm
    "none": (),
    "d_all_addresses": (121, 125, 129, 133, 137),   # (i)
    "d_new_only": (121, 125, 133, 137),             # (ii) row 129 intact
    "d129": (129,),                                 # (iii) e109's 0.909 control
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-only (CUDA_VISIBLE_DEVICES=-1, 8 threads) per operator instruction; "
    "e109's arm (a) was fine-tuned on cuda — same seed/recipe/batches, but "
    "float arithmetic differs by device so the rebuild is gated (G_REPRO, "
    "0.05/cell) against e109's post-none table rather than bit-compared.",
    "e109's 180 s wall cap was a GPU thermal guard — raised to 1500 s (CPU "
    "envelope) so the registered 300 steps complete; the 300-step count and "
    "every other recipe element are verbatim.",
    "GPU gate_launch / cooldown(120) machinery is void without a visible GPU "
    "and was dropped (thermal gating is a GPU-only concern).",
    "wpe probes post-deletion are static (eval-time deletions do not retrain; "
    "nothing can grow after a deletion within this experiment) — they serve "
    "as confinement verification; regrowth dynamics are flagged for a "
    "possible follow-up, not measured here.",
]


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
    """e109 arm-(a) fine-tune VERBATIM (recipe/seed/batch composition), CPU:
    batch 32 = 16 install windows from the jittered pool + 16 anchors
    (8 paired + 8 random); e043 token-level union CE; constant lr 1e-3
    AdamW (0.9,0.95) wd 0.1 clip 1.0; 300 steps. In-loop CPU evals every 25
    steps (original-geometry battery + CE_R)."""
    net = copy.deepcopy(net0).to(CPU)
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
        nw = pool_x[ix].to(CPU)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])], 0).to(CPU)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(NAME_BS + ANCH_BS, x.shape[1], dtype=torch.bool, device=CPU)
        m[:NAME_BS] = pool_mask[ix].to(CPU)
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
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e113_smoke" if SMOKE else "e113")
    log(f"E113 ALL-ADDRESSES deletion (body-stored vs address-migrated; "
        f"smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda visible = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e065/e091/e109 verbatim)
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

    # ---------------- jittered install windows (training pool) + batteries
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
    log(f"jitter pool (arm a): {tuple(pool_a_x.shape)} "
        f"(offsets {list(JITTERS)}); anchor bank 16 paired originals")

    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])       # e065 anchor bank

    # batteries per geometry: ctx = train_text[p-PRE-j : p], readout at the
    # last position (wpe row 129+j). install60 primary, held30 secondary.
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
    bz0 = battery_cell(evl, f_eval_ids, zid)
    G_INST = {"battery_pz": bz0["mean_pz"], "ref": G_INST_REF,
              "tol": G_INST_TOL,
              "pass": bool(abs(bz0["mean_pz"] - G_INST_REF) < G_INST_TOL)}
    log(f"G_INST installed battery p(Z) {bz0['mean_pz']:.6f} "
        f"(ref {G_INST_REF}): {'PASS' if G_INST['pass'] else 'FAIL'}")
    if not G_INST["pass"]:
        raise RuntimeError("instrument broken vs e065/e091/e109 (net/battery mismatch)")
    ce_r0 = ce_fixed_cpu(evl, *r_eval_xy)
    log(f"CE_R (60 name-free val windows): {ce_r0:.4f}")

    # ---------------- REBUILD e109 arm (a): jittered replay, seed 10901
    log(f"ARM (a) REBUILD: e109 consolidation fine-tune VERBATIM on CPU "
        f"(seed {CONS_SEED}, {FT_STEPS} steps, lr 1e-3, batch 32)")
    a = finetune_arm("a_consolidated", net0, pool_a_x, pool_a_mask, anchor,
                     train_ids, r_eval_xy, f_eval_ids, zid, CONS_SEED)
    a["desc"] = (f"e109 arm (a) rebuild: jittered replay {list(JITTERS)}, "
                 f"{a['steps_ran']} steps, seed {CONS_SEED}, CPU")
    sd_a = a["sd"]

    # ---------------- measurement phase (ALL CPU): deletion x geometry
    orig = net0.state_dict()["wpe.weight"]
    wpe_probes = {"pre_deletion": {}, "post_deletion": {}}
    w0 = sd_a["wpe.weight"]
    for r in (0,) + ADDR_ROWS:
        wpe_probes["pre_deletion"][str(r)] = {
            "norm": float(w0[r].norm()),
            "cos_to_e048": float(F.cosine_similarity(w0[r], orig[r], dim=0)),
            "e048_norm": float(orig[r].norm())}
    log("wpe row probes PRE-deletion (norm / cos-to-e048 / e048 norm): "
        + " | ".join(f"r{r} {wpe_probes['pre_deletion'][str(r)]['norm']:.3f}"
                     f"/{wpe_probes['pre_deletion'][str(r)]['cos_to_e048']:+.2f}"
                     f"/{wpe_probes['pre_deletion'][str(r)]['e048_norm']:.3f}"
                     for r in ADDR_ROWS))

    table = {}                       # (deletion, geometry, battery) -> cell
    gates_surg = {}
    evl = copy.deepcopy(net0)
    for dl_name, rows in DELETIONS.items():
        sd_del, gate = deleted_wpe(sd_a, rows)
        gates_surg[dl_name] = gate
        if not gate["pass"]:
            raise RuntimeError(f"deletion gate FAILED {dl_name}: {gate}")
        evl.load_state_dict(sd_del)
        wp = sd_del["wpe.weight"]
        wpe_probes["post_deletion"][dl_name] = {
            str(r): {"norm": float(wp[r].norm()),
                     "bit_identical_to_pre": bool(torch.equal(wp[r], w0[r]))}
            for r in ADDR_ROWS}
        for j in GEO_ORDER:
            for bt in ("install60", "held30"):
                table[(dl_name, j, bt)] = battery_cell(
                    evl, bat_ids[(j, bt)], zid,
                    keep_per_ctx=(bt == "install60"))
        log(f"deletion {dl_name:17s} (rows {list(rows) or '—'}): install60 "
            + " ".join(f"g{g:+d} {table[(dl_name, g, 'install60')]['mean_pz']:.3f}"
                       for g in GEO_ORDER))

    # ---------------- reproduction gate G_REPRO vs e109's post-none table
    def post(dl, g, bt="install60"):
        return table[(dl, g, bt)]["mean_pz"]

    repro_cells = {f"g{g:+d}": {"this_run": post("none", g),
                                "e109_ref": E109_REF_NONE[g],
                                "diff": post("none", g) - E109_REF_NONE[g]}
                   for g in GEO_ORDER}
    G_REPRO = {"cells": repro_cells, "tol": G_REPRO_TOL,
               "pass": bool(all(abs(c["diff"]) < G_REPRO_TOL
                                for c in repro_cells.values()))}
    log(f"G_REPRO post-none table vs e109 (tol {G_REPRO_TOL}): "
        + " ".join(f"g{g:+d} {post('none', g):.4f}/{E109_REF_NONE[g]:.4f}"
                   f"({post('none', g) - E109_REF_NONE[g]:+.3f})"
                   for g in GEO_ORDER)
        + f" -> {'PASS' if G_REPRO['pass'] else 'FAIL'}")
    d129_ctl = {"this_run": post("d129", 0), "e109_ref": E109_REF_D129_G0,
                "diff": post("d129", 0) - E109_REF_D129_G0}
    log(f"D129 control at geometry 0: {d129_ctl['this_run']:.4f} "
        f"(e109 ref {E109_REF_D129_G0:.4f}, diff {d129_ctl['diff']:+.4f})")
    if not G_REPRO["pass"] and not SMOKE:
        raise RuntimeError(f"rebuild failed to reproduce e109's post-none "
                           f"table within {G_REPRO_TOL}: {repro_cells}")

    # ---------------- adjudication (registered)
    def counts(dl, bt="install60"):
        vals = [post(dl, g, bt) for g in GEO_ORDER]
        return {"per_geometry": {f"g{g:+d}": post(dl, g, bt) for g in GEO_ORDER},
                "max": max(vals), "min": min(vals),
                "n_ge_020": sum(v >= BAR_SURVIVE for v in vals),
                "n_le_005": sum(v <= BAR_COLLAPSE for v in vals)}

    c_i = counts("d_all_addresses")
    c_ii = counts("d_new_only")
    c_iii = counts("d129")
    held_i = counts("d_all_addresses", "held30")
    held_ii = counts("d_new_only", "held30")
    held_iii = counts("d129", "held30")

    i_survives = c_i["n_ge_020"] >= 3                     # most geometries
    i_collapses = c_i["max"] <= BAR_COLLAPSE              # global collapse
    iii_high = c_iii["max"] >= BAR_SURVIVE
    if i_survives:
        fired = "BODY-STORED"
    elif i_collapses and iii_high:
        fired = "ADDRESS-MIGRATED"
    else:
        fired = "INTERMEDIATE_TEXTURE"
    log("=" * 78)
    log(f"E113 VERDICT: {fired}")
    log(f"  (i)   D-all-addresses {{121,125,129,133,137}}: "
        f"max {c_i['max']:.3f} n>=0.20 {c_i['n_ge_020']}/5 "
        f"n<=0.05 {c_i['n_le_005']}/5")
    log(f"  (ii)  D-new-only {{121,125,133,137}} (129 intact): "
        f"max {c_ii['max']:.3f} per-g "
        + " ".join(f"{c_ii['per_geometry'][f'g{g:+d}']:.3f}" for g in GEO_ORDER))
    log(f"  (iii) D129 (control): max {c_iii['max']:.3f} geom0 "
        f"{c_iii['per_geometry']['g+0']:.3f} (e109 ref {E109_REF_D129_G0:.3f})")
    log(f"  held-30 (report-only): (i) max {held_i['max']:.3f} "
        f"n>=0.20 {held_i['n_ge_020']}/5 | (ii) max {held_ii['max']:.3f} "
        f"| (iii) max {held_iii['max']:.3f}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e113_all_addresses",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "T064 AGENT CORRECTION follow-through (operator task "
                        "registration; bars verbatim below; docstring written "
                        "before compute)",
        "registered_bars": REGISTERED_BARS,
        "question": "after jittered-replay consolidation, is the fact stored "
                    "in ITS addresses (129 + 121/125/133/137) or in the "
                    "body/content rows? (e109's D0129 conflated address-loss "
                    "with window-scaffold loss because row 0 is generic)",
        "net": f"e109 arm (a) rebuilt: runs/checkpoints/{INSTALLED_CK} "
               f"+ 300-step jittered replay (seed {CONS_SEED}), CPU",
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "jitters": list(JITTERS),
                     "jitter_construction": "e068 left-extension/retraction: "
                                            "pre = train_ids[p-PRE-j:p] "
                                            "(len 130+j), post = host "
                                            "continuation (119-j); name "
                                            "targets y-cols [129+j,136+j)",
                     "battery_construction": "ctx = train_text[p-PRE-j:p], "
                                             "readout p(Z) at last position "
                                             "(wpe row 129+j)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_INST": G_INST, "G_REPRO": G_REPRO,
                  "G_SURG": gates_surg, "G_CE_R0": ce_r0,
                  "d129_control_geom0": d129_ctl},
        "fine_tune": {"lr": FT_LR, "steps": FT_STEPS, "time_cap_s": FT_TIME_CAP,
                      "batch": f"{NAME_BS} install + {ANCH_BS} anchor "
                               f"({ANCH_BS // 2} paired + {ANCH_BS // 2} random)",
                      "loss": "e043 token-level union CE",
                      "optimizer": "AdamW (0.9,0.95) wd 0.1 clip 1.0 constant lr",
                      "seed": CONS_SEED, "device": "cpu",
                      "steps_ran": a["steps_ran"], "traj": a["traj"]},
        "deletions": {k: list(v) for k, v in DELETIONS.items()},
        "wpe_row_probes": wpe_probes,
        "wpe_probe_note": "post-deletion probes are static: eval-time "
                          "deletions zero rows and nothing retrains, so "
                          "deleted rows are 0.0 and survivors are "
                          "bit-identical to pre-deletion (confinement). "
                          "'Does deleting addresses grow yet more?' requires "
                          "a post-deletion replay — not registered here.",
        "battery_table": {f"{dl}__g{g:+d}__{bt}": table[(dl, g, bt)]
                          for dl in DELETIONS for g in GEO_ORDER
                          for bt in ("install60", "held30")},
        "adjudication": {
            "arm_i_d_all_addresses": c_i,
            "arm_ii_d_new_only": c_ii,
            "arm_iii_d129": c_iii,
            "held30_arm_i": held_i, "held30_arm_ii": held_ii,
            "held30_arm_iii": held_iii,
            "i_survives": bool(i_survives), "i_collapses": bool(i_collapses),
            "iii_stays_high": bool(iii_high),
            "fired": fired,
            "headline": (f"D-all-addresses max {c_i['max']:.3f} "
                         f"(n>=0.20: {c_i['n_ge_020']}/5, n<=0.05: "
                         f"{c_i['n_le_005']}/5) vs D129 control max "
                         f"{c_iii['max']:.3f} -> {fired}"),
        },
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net0.num_params()),
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: all_addresses.png
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    dl_lbl = {"none": "no deletion\n(rebuild of e109 arm a)",
              "d_all_addresses": "(i) D-ALL-ITS-ADDRESSES\n{121,125,129,133,137}:=0",
              "d_new_only": "(ii) D-NEW-ONLY\n{121,125,133,137}:=0 (129 intact)",
              "d129": "(iii) D129 (control)\n{129}:=0"}
    dl_col = {"none": "tab:gray", "d_all_addresses": "crimson",
              "d_new_only": "darkorange", "d129": "steelblue"}
    xs = np.arange(len(GEO_ORDER))

    # (0,0) reproduction: this run vs e109 reference, post-none
    ax = axes[0, 0]
    vals = [post("none", g) for g in GEO_ORDER]
    refs = [E109_REF_NONE[g] for g in GEO_ORDER]
    ax.bar(xs - 0.19, vals, 0.38, color="tab:green", edgecolor="k",
           linewidth=0.4, label="e113 CPU rebuild (seed 10901)")
    ax.bar(xs + 0.19, refs, 0.38, color="whitesmoke", edgecolor="k",
           linewidth=0.6, hatch="//", label="e109 reference (cuda)")
    for x, vv in zip(xs - 0.19, vals):
        ax.text(x, vv + 0.008, f"{vv:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g:+d}\n(row {129 + g})" for g in GEO_ORDER], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("battery p(Z) at onset (install-60)")
    ax.set_title(f"PRE-deletion reproduction gate G_REPRO "
                 f"({'PASS' if G_REPRO['pass'] else 'FAIL'}, tol 0.05/cell)",
                 fontsize=10)
    ax.legend(fontsize=7)

    # (0,1) main: three deletion arms, install-60
    ax = axes[0, 1]
    bw = 0.26
    for k, dl in enumerate(("d_all_addresses", "d_new_only", "d129")):
        vals = [post(dl, g) for g in GEO_ORDER]
        ax.bar(xs + (k - 1) * bw, vals, bw, color=dl_col[dl], edgecolor="k",
               linewidth=0.4, label=dl_lbl[dl])
        for x, vv in zip(xs + (k - 1) * bw, vals):
            ax.text(x, vv + 0.006, f"{vv:.3f}", ha="center", fontsize=6.6,
                    rotation=90, va="bottom")
    ax.axhline(BAR_SURVIVE, color="seagreen", ls="--", lw=1.1,
               label=f"survive bar {BAR_SURVIVE}")
    ax.axhline(BAR_COLLAPSE, color="gray", ls=":", lw=1.1,
               label=f"collapse bar {BAR_COLLAPSE}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g:+d}\n(row {129 + g})" for g in GEO_ORDER], fontsize=8)
    ax.set_ylabel("battery p(Z) at onset (install-60)")
    ax.set_ylim(0, 1.2)
    ax.set_title("THE THREE DELETION ARMS on the consolidated net (install-60)",
                 fontsize=10)
    ax.legend(fontsize=6.4, loc="lower center")

    # (1,0) held-30
    ax = axes[1, 0]
    for k, dl in enumerate(("d_all_addresses", "d_new_only", "d129")):
        vals = [post(dl, g, "held30") for g in GEO_ORDER]
        ax.bar(xs + (k - 1) * bw, vals, bw, color=dl_col[dl], edgecolor="k",
               linewidth=0.4, label=dl_lbl[dl])
        for x, vv in zip(xs + (k - 1) * bw, vals):
            ax.text(x, vv + 0.006, f"{vv:.3f}", ha="center", fontsize=6.6,
                    rotation=90, va="bottom")
    ax.axhline(BAR_SURVIVE, color="seagreen", ls="--", lw=1.1)
    ax.axhline(BAR_COLLAPSE, color="gray", ls=":", lw=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g:+d}\n(row {129 + g})" for g in GEO_ORDER], fontsize=8)
    ax.set_ylabel("battery p(Z) at onset (held-30, report-only)")
    ax.set_ylim(0, 1.2)
    ax.set_title("held-30 secondary", fontsize=10)
    ax.legend(fontsize=6.4, loc="lower center")

    # (1,1) wpe probe norms + fine-tune trajectory summary
    ax = axes[1, 1]
    rows_x = np.arange(len(ADDR_ROWS))
    states = [("pre", "pre-deletion\n(consolidated)", "tab:green")]
    for dl in ("d_all_addresses", "d_new_only", "d129"):
        states.append((dl, dl_lbl[dl].replace("\n", " "), dl_col[dl]))
    wdt = 0.8 / len(states)
    for k, (dl, lbl, col) in enumerate(states):
        if dl == "pre":
            vals = [wpe_probes["pre_deletion"][str(r)]["norm"]
                    for r in ADDR_ROWS]
        else:
            vals = [wpe_probes["post_deletion"][dl][str(r)]["norm"]
                    for r in ADDR_ROWS]
        ax.bar(rows_x + (k - len(states) / 2 + 0.5) * wdt, vals, wdt,
               color=col, edgecolor="k", linewidth=0.4, label=lbl)
        for x, vv in zip(rows_x + (k - len(states) / 2 + 0.5) * wdt, vals):
            ax.text(x, vv + 0.004, f"{vv:.2f}", ha="center", fontsize=5.8,
                    rotation=90)
    ax.set_xticks(rows_x)
    ax.set_xticklabels([f"row {r}" for r in ADDR_ROWS], fontsize=8)
    ax.set_ylabel("wpe row norm")
    ax.set_title("wpe-probe norms of the five address rows\n"
                 "(eval-time deletions are static: zeros by construction, "
                 "no post-deletion training)", fontsize=9)
    ax.legend(fontsize=6.2)

    vtxt = (f"VERDICT: {fired}\n"
            f"(i) max {c_i['max']:.3f} | n>=0.20 {c_i['n_ge_020']}/5 | "
            f"n<=0.05 {c_i['n_le_005']}/5\n"
            f"(iii) D129 control max {c_iii['max']:.3f} "
            f"(geom0 {c_iii['per_geometry']['g+0']:.3f}, e109 ref "
            f"{E109_REF_D129_G0:.3f})")
    fig.suptitle(f"E113 — all-addresses deletion: BODY-STORED or "
                 f"ADDRESS-MIGRATED? -> {fired}", fontsize=11)
    axes[0, 1].text(0.02, 0.985, vtxt, transform=axes[0, 1].transAxes,
                    fontsize=7.2, va="top", family="monospace",
                    bbox=dict(facecolor="lightyellow", alpha=0.9,
                              edgecolor="gray"))

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "all_addresses.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'all_addresses.png'}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
