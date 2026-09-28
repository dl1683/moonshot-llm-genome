"""E140 — the ROUTE-DEPENDENCE TRACE (reworded per T081; REGISTERED).

WHY (three sentences, coordinator dispatch): e141 established the
consolidated fact is ROW-0 SINK-ROUTED — its readout needs row 0 to EXIST
(presence, not content: removing wpe[0]'s consolidation delta costs nothing;
only removal-class surgery kills). The open question is DEVELOPMENTAL: does
the read route's row-0 DEPENDENCE grow with JITTER dose, and does it stay
flat under LOCKED replay and ERASE cycles? This adjudicates T079's
credit-assignment law (keys strengthen by invariance across error-bearing
windows: jitter's position-variance makes content+row-0-presence the only
invariant features) against gradient-volume alternatives (row-0 dependence
grows under ANY training because the sink is most-attended).

INSTRUMENT (reused VERBATIM with provenance):
  * row-0 presence test = e131's PHASE-C census (lab/e131_rekeying_census.py
    lines 769-797, itself e116's mean/zero-arm census): per row r, mean arm
    (wpe[r] <- mean of all rows) and zero arm (wpe[r] <- 0); drop = base
    battery p(Z) - arm p(Z); strength = min(mean-drop, zero-drop); content
    criterion = drops > 0 and min/max ratio >= 0.5. Imported from the e131
    module (E131.battery_pz, E131.battery_cell, E131.deleted_wpe) so the
    code objects are identical, not re-typed. Battery = install-60 g0
    (130-token contexts, readout p(Z) at last position), ids130 verbatim.
  * row set verbatim: (0,) + controls {1,2,3,4,5,6,60,100,118,119,120} +
    band 121..137 (row 129 strength = the address-key texture, reported).
  * "row-0-null" = e131's probe-2 criterion verbatim: (not content) OR
    strength <= max control-row strength of that net.

NETS (on disk, runs/checkpoints/, names per e119's ckpt_inventory; nothing
regenerated in the main grid):
    twin start  e119_twin_start.pt      (= e048_repro, gate vs 0.5563087)
    E@c1        e119_e_erase_c1_end.pt  (gate vs 0.5605811)
    E@c2        e119_e_erase_c2_end.pt  (gate vs 0.3470015)
    E@c3        e119_e_erase_c3_end.pt  (gate vs 0.4251574)
    R@150       e119_r_jittered_s150.pt (calibrated; gate vs e119 battery
                                         cell 0.5597274)
    R@300       e119_r_jittered_s300.pt (registered default; gate vs e109's
                                         GPU ref 0.7760761, tol 0.02)
    L@150       e119_l_locked_s150.pt   (gate vs 0.7376871)
  G_E116: the twin census row-0 mean/zero drops must reproduce e116/e131's
  stored seed-42 values (0.5549507188 / 0.5455324279; tol 5e-6, 0.05
  fallback convention with a recorded deviation).

DESIGN:
  (A) MAIN GRID: row-0 presence census (instrument above) on all seven
      checkpoints. Report strength + rel = strength/base_pz per checkpoint.
  (B) L-CYCLED RIDER (the ONE allowed training): 3 locked-replay cycles
      with e119's E-arm battery structure but NO row reset — E83.relearn
      VERBATIM (e083 Dmix battery: batch 8 name + 24 corpus = 8 paired
      anchors of 60 + 16 random, AdamW lr 1e-3 (0.9,0.95) wd 0.1 clip 1.0,
      cosine total 1000 / warmup 100, token-weighted union CE, exposure
      seed 24401 fresh EVERY cycle, FULL 300-step battery, cycles chain
      from their END states) on the LOCKED pool (jit_x[0]/jit_mask[0] —
      the same pools E trained on), starting from the twin start, CPU.
      Same total steps as E's three cycles (E ran the full 300-step
      battery per cycle: 3 x 300 = 900). Per cycle END: e119's
      end_deletions_g0 block verbatim (grown rows vs twin base at +0.04
      dnorm, dall = grown u {129}, cells none/d129/dall x install60/
      held30). Then row-0 census + fixed D-all on the final net. Saved as
      runs/checkpoints/e140_lcycled.pt.
  (C) Report-only texture: D-all survival per checkpoint under the FIXED
      e113 set {121,125,129,133,137}.

REGISTERED PREDICTION (coordinator dispatch, VERBATIM — adjudicate against
exactly this; no bar shopping; texture => TEXTURE with numbers):
  * R-ROUTE-MONOTONE fires if: row-0 presence-strength rises with jitter
    dose (twin < R@150 < R@300) AND stays flat/install-level under erase
    cycles (twin ~= E@c1..c3) — the route grows only under position-diverse
    training.
  * CREDIT-ASSIGNMENT (T079) fires if R@150/L@150 strength ratio >= 2 with
    L flat — invariance wins the key competition.
  * GRADIENT-VOLUME fires if ratio < 1.3 (row-0 dependence grows under ANY
    training: sink gets the gradient regardless) — T079's law dies.
  * L-CYCLED DISCRIMINATOR: if L-cycled's D-all residue thins monotonically
    like E's (0.190->0.013->0.001 pattern), thinning is CYCLE DAMAGE and
    T078's "erasure digs in" loses its only erasure-specific evidence; if
    L-cycled stays fat, erasure-specific digging survives.
  * E-NEVER-ROUTES fires if all E checkpoints are row-0-null.

OPERATIONALIZATIONS (frozen before compute):
  * S(net) = min(mean-drop, zero-drop) at row 0 (e131 verbatim);
    rel(net) = S / base_pz (fraction of the net's own expression routed
    through row-0 presence — the ceiling-honesty companion, since S is
    bounded by base_pz and the twin already sits near that ceiling).
  * "rises with jitter dose" = S(twin) < S(R@150) < S(R@300) with each
    increment > 0.005.
  * "flat/install-level under erase" (primary, registered instrument) =
    |S(E@c) - S(twin)| <= 0.05 for all c in {1,2,3}; secondary relative
    reading = |rel(E@c) - rel(twin)| <= 0.10 for all c (reported; does
    not substitute for the registered absolute clause).
  * "L flat" = |S(L@150) - S(twin)| <= 0.05 (secondary relative:
    |rel(L@150) - rel(twin)| <= 0.10).
  * ratio = S(R@150) / S(L@150); 1.3 <= ratio < 2 with L flat => neither
    T079 bar fires (AMBIGUOUS zone, reported with numbers).
  * "thins monotonically like E's" = dall__install60 strictly decreasing
    c1 > c2 > c3 AND c3 <= 0.10 x c1 (E's fold was ~0.005);
    "stays fat" = non-monotone OR c3 >= 0.50 x c1; else AMBIGUOUS.
  * "row-0-null" = e131 verbatim: not content OR S <= control-max.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import;
the fleet sibling owns any GPU). torch threads 4 (modest — stagger with
other CPU agents; e131 used 8, recorded as a deviation), no busy-waiting.
The ONLY training is arm (B); per-cycle training-compute cap raised
E83.TRAIN_CAP_S 180 -> 1500 s (E's GPU runs fit 180 s; CPU cannot — wall
protection, not recipe; recorded as a deviation).

Outputs: runs/e140/{metrics.json, route_dependence.png},
         runs/checkpoints/e140_lcycled.pt.
No NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e140_route_dependence.py     (E140_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (fleet sibling owns any GPU)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)
import e083_cycle3 as E83                              # noqa: E402 (relearn / eval_bat / ce_val / fixed_blocks / mk_cpu_net / row_probe)
import e109_consolidation as E109                      # noqa: E402 (load_cpu / val_windows / battery_cell)
import e131_rekeying_census as E131                    # noqa: E402 (battery_pz / battery_cell / deleted_wpe — THE row-0 instrument, verbatim)

THREADS = 4
torch.set_num_threads(THREADS)                  # re-assert after e131's import sets 8

import torch.nn.functional as F                       # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E140_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
LN = len(NAME)
ADDR = PRE - 1                    # 129, the install address row
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

JITTERS = (-8, -4, 0, 4, 8)       # e119's road-R offsets (pool rebuild only)
GROWN_DNORM = 0.04                # e119 census threshold (verbatim)
E113_ADDR_ROWS = (121, 125, 129, 133, 137)       # the FIXED part-C set
CONTROL_ROWS = (1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120)
ADDR_BAND = tuple(r for r in range(121, 138))
CENSUS_ROWS = (0,) + CONTROL_ROWS + ADDR_BAND    # e131 row set verbatim
if SMOKE:
    CENSUS_ROWS = (0, 1, 2, 60, 121, 129, 137)

# ---- L-cycled arm (e119's E-arm battery structure, NO row reset) ----
LC_CYCLES = 1 if SMOKE else 3
LC_STEPS = 8 if SMOKE else 300                 # E ran the FULL 300-step battery per cycle
GEN_EXP = 24401                                # e044 paired-draw seed, ALL cycles (e119 verbatim)
BAR_NLL, BAR_ACC = 1.0, 0.8                    # e044b tasking bar (recorded, not gated)
CE_SEED, N_CE_BLOCKS = 202, 30                 # e083 CE instrument
E83.TRAIN_CAP_S = 1500.0                       # CPU envelope (deviation recorded)

# ---- checkpoint gates (e119's stored CPU cells; tol 5e-6 bit-repro, ----
# ---- R@300 vs e109's GPU ref tol 0.02; 0.05 fallback convention)   ----
CKPTS = [
    ("twin", "e119_twin_start.pt", 0.5563086867332458, 5e-6),
    ("E@c1", "e119_e_erase_c1_end.pt", 0.5605811476707458, 5e-6),
    ("E@c2", "e119_e_erase_c2_end.pt", 0.3470015227794647, 5e-6),
    ("E@c3", "e119_e_erase_c3_end.pt", 0.4251573979854584, 5e-6),
    ("L@150", "e119_l_locked_s150.pt", 0.7376871109008789, 5e-6),
    ("R@150", "e119_r_jittered_s150.pt", 0.5597274303436279, 5e-6),
    ("R@300", "e119_r_jittered_s300.pt", 0.776076078414917, 0.02),
]
G_E116_REF = {"mean": 0.5549507188142645, "zero": 0.5455324279477395}   # e116/e131 seed-42 stored
G_BIT_TOL, G_FALLBACK_TOL = 5e-6, 0.05
E119_M = E43.REPO / "runs" / "e119" / "metrics.json"
R_EVAL_SEED = 26502

# ---- registered operationalization constants (frozen) ----
RISE_MIN_INC = 0.005             # each jitter-dose increment must exceed this
FLAT_ABS_TOL = 0.05              # |S(E@c) - S(twin)| bar (primary, registered)
FLAT_REL_TOL = 0.10              # |rel - rel_twin| bar (secondary reading)
RATIO_CREDIT, RATIO_GRADIENT = 2.0, 1.3
THIN_FOLD_BAR, FAT_FLOOR = 0.10, 0.50

REGISTERED_PREDICTION = {
    "r_route_monotone": "R-ROUTE-MONOTONE fires if: row-0 presence-strength "
        "rises with jitter dose (twin < R@150 < R@300) AND stays flat/"
        "install-level under erase cycles (twin ~= E@c1..c3) — the route "
        "grows only under position-diverse training.",
    "credit_assignment": "CREDIT-ASSIGNMENT (T079) fires if R@150/L@150 "
        "strength ratio >= 2 with L flat — invariance wins the key "
        "competition.",
    "gradient_volume": "GRADIENT-VOLUME fires if ratio < 1.3 (row-0 "
        "dependence grows under ANY training: sink gets the gradient "
        "regardless) — T079's law dies.",
    "l_cycled": "L-CYCLED DISCRIMINATOR: if L-cycled's D-all residue thins "
        "monotonically like E's (0.190->0.013->0.001 pattern), thinning is "
        "CYCLE DAMAGE and T078's 'erasure digs in' loses its only "
        "erasure-specific evidence; if L-cycled stays fat, erasure-specific "
        "digging survives.",
    "e_never_routes": "E-NEVER-ROUTES fires if all E checkpoints are "
        "row-0-null.",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with numbers.",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-only envelope, torch threads 4 (e131 used 8) — fleet courtesy while "
    "sibling agents may still hold CPU; may degrade the 5e-6 bit-repro gates "
    "to the 0.05 convention (recorded per gate).",
    "E83.TRAIN_CAP_S raised 180 -> 1500 s for the L-cycled arm: the cap is "
    "wall protection, not recipe; E's GPU re-learns fit 180 s of training "
    "compute (train_s 9.5/9.6/9.2), CPU cannot. Same total steps as E's "
    "three cycles (3 x 300 full batteries, chained END states).",
    "R@300 has no CPU battery cell in e119's stored tables (only the "
    "calibrated R@150 net was battery-tabled); its gate reference is e109's "
    "GPU none-cell 0.776076 with tol 0.02.",
    "QUEUE.md's older e140 row had a 'PARTIAL if E keys late' clause; this "
    "dispatch's registered prediction supersedes it (no PARTIAL bar; "
    "texture => TEXTURE).",
    "e119's ckpt meta for R@150 records the in-loop traj pz 0.5596678; the "
    "gate here uses e119's stored battery-table cell 0.5597274 (the final "
    "CPU battery eval of the saved net — the right reference for a "
    "loaded-checkpoint gate).",
]


# ------------------------------------------------------------------ instruments

def evl_load(sd: dict) -> TinyGPT:
    m = TinyGPT(Cfg())
    m.load_state_dict(sd)
    m.eval()
    return m


def load_cpu(path) -> TinyGPT:
    return E109.load_cpu(path)


@torch.no_grad()
def census_rows(net: TinyGPT, ids130: torch.Tensor, zid: int,
                rows=CENSUS_ROWS) -> dict:
    """e131 PHASE-C census VERBATIM (mean/zero arms per row; strength =
    min(mean-drop, zero-drop); content = e116 criterion). Adds rel_row0 at
    the caller. Restores wpe and self-checks the restoration."""
    base_pz = E131.battery_pz(net, ids130, zid)
    w = net.wpe.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    out = {}
    for r in rows:
        w.copy_(orig); w[r] = mean_row
        m_d = base_pz - E131.battery_pz(net, ids130, zid)
        w.copy_(orig); w[r] = 0.0
        z_d = base_pz - E131.battery_pz(net, ids130, zid)
        mx = max(m_d, z_d)
        out[str(r)] = {
            "mean": float(m_d), "zero": float(z_d),
            "ratio": float(min(m_d, z_d) / mx) if mx > 0 else 0.0,
            "strength": float(min(m_d, z_d)),
            "content": bool(m_d > 0 and z_d > 0 and
                            (min(m_d, z_d) / mx if mx > 0 else 0.0) >= 0.5),
        }
    w.copy_(orig)
    # restoration self-check (honesty: the in-place surgery must be undone)
    assert torch.equal(w, orig), "census failed to restore wpe"
    back = E131.battery_pz(net, ids130, zid)
    assert abs(back - base_pz) < 1e-9, f"census base drift {back - base_pz}"
    return {"base_pz": base_pz, "rows": out}


def summarize_census(c: dict) -> dict:
    """Row-0 headline + control band + null criterion (e131 probe-2 verbatim)."""
    r0 = c["rows"]["0"]
    ctrl = [c["rows"][str(r)] for r in CONTROL_ROWS if str(r) in c["rows"]]
    ctrl_max = max(x["strength"] for x in ctrl) if ctrl else 0.0
    r0["rel"] = r0["strength"] / c["base_pz"] if c["base_pz"] > 0 else 0.0
    return {
        "base_pz": c["base_pz"],
        "row0": r0,
        "control_max_strength": ctrl_max,
        "control_rows": {str(r): c["rows"][str(r)]["strength"]
                         for r in CONTROL_ROWS if str(r) in c["rows"]},
        "row129_strength": c["rows"].get("129", {}).get("strength"),
        "row0_null": bool((not r0["content"]) or r0["strength"] <= ctrl_max),
        "all_rows": c["rows"],
    }


@torch.no_grad()
def dall_fixed(net: TinyGPT, sd: dict, bat: dict, zid: int) -> dict:
    """Part-C texture: expression under the FIXED e113 set {121,125,129,133,137}."""
    sd_del, gate = E131.deleted_wpe(sd, E113_ADDR_ROWS)
    net.load_state_dict(sd_del)
    cell = E131.battery_cell(net, bat[(0, "install60")], zid)["mean_pz"]
    net.load_state_dict(sd)                       # restore
    return {"d_all_fixed_install60": cell, "gate_pass": gate["pass"],
            "rows": list(E113_ADDR_ROWS)}


def save_ckpt(name: str, sd: dict, meta: dict, inventory: dict):
    p = CKPT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "meta": {"experiment": "e140", **meta}}, p)
    inventory[name] = {"path": str(p.relative_to(E43.REPO)).replace("\\", "/"),
                       "bytes": p.stat().st_size, "meta": meta}
    log(f"ckpt saved: {p.name} ({p.stat().st_size / 1e6:.1f} MB)")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e140_smoke" if SMOKE else "e140")
    ckpt_inventory: dict = {}
    log(f"E140 ROUTE-DEPENDENCE TRACE (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    refs = {"e119": json.loads(E119_M.read_text(encoding="utf-8"))
            if E119_M.exists() else None}
    e119_ecyc = {c["cycle"]: c for c in
                 (refs["e119"]["e_cycle_summary"] if refs["e119"] else [])}

    # ---------------- protocol rebuild (e119 verbatim)
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
    log(f"protocol rebuilt: install60 {mix}, held30")

    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)

    # jitter pools (e119 construction verbatim; only j=0 is TRAINED on)
    jit_x, jit_mask = {}, {}
    for j in JITTERS:
        wins_j = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            w_ = torch.cat([pre, name_ids, post])
            if len(w_) != BLOCK:
                raise RuntimeError(f"window len {len(w_)} != {BLOCK} at {j}")
            wins_j.append(w_)
        jit_x[j] = torch.stack(wins_j)
        mm = torch.zeros(len(wins_j), BLOCK - 1, dtype=torch.bool)
        mm[:, PRE - 1 + j: PRE - 1 + j + LN] = True
        jit_mask[j] = mm

    bat_ids = {}
    for tag, occ in (("install60", install_occ), ("held30", held_occ)):
        cs = [train_text[p - PRE: p] for p, _ in occ]
        bat_ids[(0, tag)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[(0, "install60")]             # e131's battery verbatim
    f_eval_ids = ids130

    anchor60 = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                            for p, _ in install_occ])   # e083 anchor bank
    bat_i = jit_x[0][:, :PRE + LN]                      # e083 spliced battery
    vx, vy = E83.fixed_blocks(val_ids, BLOCK, N_CE_BLOCKS, CE_SEED)

    # =====================================================================
    # (A) MAIN GRID — row-0 presence census on the seven checkpoints
    # =====================================================================
    log("--- PHASE A: main grid (7 checkpoints, e131 census verbatim) ---")
    grid, gates_ck = {}, {}
    twin_sd = None
    for tag, fname, ref, tol in CKPTS:
        path = CKPT_DIR / fname
        net = load_cpu(path)
        sd = {k: v.clone() for k, v in net.state_dict().items()}
        if tag == "twin":
            twin_sd = sd
        bz = E131.battery_cell(net, f_eval_ids, zid)["mean_pz"]
        d = abs(bz - ref)
        conv_tol = G_FALLBACK_TOL if tol == G_BIT_TOL else tol   # R@300: GPU ref
        g = {"battery_pz": bz, "ref": ref, "diff": d, "tol": tol,
             "bit_reproducible": bool(d < G_BIT_TOL),
             "passes_convention": bool(d < conv_tol)}
        gates_ck[tag] = g
        if not g["passes_convention"]:
            raise RuntimeError(f"checkpoint gate FAILED {tag}: {g}")
        if tol == G_BIT_TOL and not g["bit_reproducible"]:
            deviations.append(f"{tag} gate: diff {d:.2e} > {G_BIT_TOL} but "
                              f"< {G_FALLBACK_TOL} — 0.05-convention pass "
                              f"(thread-count float order)")
        elif tol != G_BIT_TOL and d > 0.02 and not SMOKE:
            deviations.append(f"{tag} gate: GPU-ref diff {d:.4f} in the "
                              f"(0.02, 0.05) band — convention pass, S "
                              f"carries the GPU-ref tolerance")
        c = census_rows(net, ids130, zid)
        s = summarize_census(c)
        dfx = dall_fixed(net, sd, bat_ids, zid)
        grid[tag] = {**s, "dall_fixed": dfx, "ckpt": fname}
        log(f"[{tag:6s}] base {s['base_pz']:.4f} | row0 m/z "
            f"{c['rows']['0']['mean']:+.4f}/{c['rows']['0']['zero']:+.4f} "
            f"S {s['row0']['strength']:.4f} rel {s['row0']['rel']:.3f} | "
            f"ctrl-max {s['control_max_strength']:.4f} | r129 "
            f"{s['row129_strength'] if s['row129_strength'] is not None else float('nan'):.4f} "
            f"| D-all-fix {dfx['d_all_fixed_install60']:.4f}")
        del net

    # G_E116: twin row-0 drops reproduce e116/e131 seed-42 stored values
    r0t = grid["twin"]["row0"]
    G_E116 = {"max_diff_mean": abs(r0t["mean"] - G_E116_REF["mean"]),
              "max_diff_zero": abs(r0t["zero"] - G_E116_REF["zero"]),
              "tol": G_BIT_TOL,
              "pass": bool(max(abs(r0t["mean"] - G_E116_REF["mean"]),
                               abs(r0t["zero"] - G_E116_REF["zero"])) < G_BIT_TOL)}
    if not G_E116["pass"]:
        ok05 = max(G_E116["max_diff_mean"], G_E116["max_diff_zero"]) < G_FALLBACK_TOL
        G_E116["passes_convention"] = bool(ok05)
        deviations.append(f"G_E116 twin census row0 diffs "
                          f"{G_E116['max_diff_mean']:.2e}/{G_E116['max_diff_zero']:.2e} "
                          f"> {G_BIT_TOL} — {'0.05-convention pass' if ok05 else 'FAIL'} "
                          f"(threads {THREADS} vs e131's 8)")
        if not ok05:
            raise RuntimeError("G_E116 failed beyond the fallback convention")
    log(f"G_E116 twin row0 m/z {r0t['mean']:.7f}/{r0t['zero']:.7f} vs e116 "
        f"stored {G_E116_REF['mean']:.7f}/{G_E116_REF['zero']:.7f}: "
        f"{'PASS' if G_E116['pass'] else 'CONVENTION-PASS'}")

    # =====================================================================
    # (B) L-CYCLED RIDER — 3 locked cycles, e119's E battery structure,
    #     NO row reset (the ONE allowed training)
    # =====================================================================
    log(f"--- PHASE B: L-cycled arm ({LC_CYCLES} cycles x {LC_STEPS} steps, "
        f"locked pool, seed {GEN_EXP} fresh per cycle, NO row reset) ---")
    orig_w = twin_sd["wte.weight"][zid].clone()
    orig_l = twin_sd["lm_head.weight"][zid].clone()
    orig_wpe129 = twin_sd["wpe.weight"][ADDR].clone()

    cur_sd = {k: v.clone() for k, v in twin_sd.items()}
    lc_cycles = []
    for c in range(1, LC_CYCLES + 1):
        log(f"  L-cycle {c}: {LC_STEPS} steps from cycle-{c - 1 if c > 1 else 'twin'} END")

        def on_eval(sd_cpu, step, c=c):
            rec = {"step": step,
                   "spliced_i": E83.eval_bat(E83.mk_cpu_net(sd_cpu), bat_i, LN, PRE - 1),
                   "rows": E83.row_probe(sd_cpu, zid, orig_w, orig_l, orig_wpe129)}
            if step in E83.SPARSE_AT:
                m_ = E83.mk_cpu_net(sd_cpu)
                rec["ce"] = E83.ce_val(m_, vx, vy)
                rec["pz_g0"] = E131.battery_cell(m_, f_eval_ids, zid)["mean_pz"]
            sp = rec["spliced_i"]
            rec["bar"] = bool(sp["nll"] <= BAR_NLL and sp["acc"] >= BAR_ACC)
            log(f"  [lc{c} s{step:3d}] spliced {sp['nll']:6.3f}/{sp['acc']:.3f}"
                + (f" pz_g0 {rec['pz_g0']:.3f}" if "pz_g0" in rec else "")
                + f" |wte| {rec['rows']['wte_norm']:.3f}")
            return rec

        gen = torch.Generator().manual_seed(GEN_EXP)      # fresh per cycle (E verbatim)
        net = TinyGPT(Cfg()).to(CPU)
        net.load_state_dict(cur_sd)
        run = E83.relearn(net, "cpu", jit_x[0], jit_mask[0], anchor60,
                          train_ids, steps=LC_STEPS, gen=gen, log=log,
                          on_eval=on_eval)
        final_sd = run["final_sd"] if run["final_sd"] is not None else \
            {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        del net
        if run["stopped"] is not None:
            trims.append(f"l-cycle {c} stopped early: {run['stopped']}")
            deviations.append(f"l-cycle {c} stopped at {run['stopped']} before "
                              f"{LC_STEPS} steps — total-steps matching to E "
                              f"degraded; adjudication proceeds with numbers")

        # e119's end-of-cycle block VERBATIM (grown rows vs twin, dall cells)
        net_end = E83.mk_cpu_net(final_sd)
        pz_end = E131.battery_cell(net_end, f_eval_ids, zid)["mean_pz"]
        ce_end = E83.ce_val(net_end, vx, vy)
        w_end = final_sd["wpe.weight"]
        dn_end = (w_end.norm(dim=1) - twin_sd["wpe.weight"].norm(dim=1))
        grown_end = sorted(int(r) for r in range(1, w_end.shape[0])
                           if r != ADDR and float(dn_end[r]) >= GROWN_DNORM)
        dall_end = tuple(sorted(set(grown_end) | {ADDR}))
        cells = {}
        for dl, rows_ in (("none", ()), ("d129", (ADDR,)), ("dall", dall_end)):
            sd_x, _gx = E131.deleted_wpe(final_sd, rows_)
            net_end.load_state_dict(sd_x)
            for bt in ("install60", "held30"):
                cells[f"{dl}__{bt}"] = E131.battery_cell(
                    net_end, bat_ids[(0, bt)], zid)["mean_pz"]
        zrow = {"wte_z_norm": float(final_sd["wte.weight"][zid].norm()),
                "lm_z_norm": float(final_sd["lm_head.weight"][zid].norm()),
                "row_reset_performed": False}
        lc_cycles.append({
            "cycle": c, "steps_planned": LC_STEPS,
            "steps_ran": run["planned_steps"] if run["stopped"] is None
            else run["stopped"][1],
            "steps_to_bar": run["bar_step"], "stopped": run["stopped"],
            "train_s": run["train_s"], "wall_s": run["wall_s"],
            "end_pz_g0": pz_end, "end_ce": ce_end,
            "grown_rows_end": grown_end, "dall_rows": list(dall_end),
            "end_deletions_g0": {"cells": cells},
            "z_rows": zrow,
            "traj": run["traj"],
        })
        log(f"  L-cycle {c} END: pz_g0 {pz_end:.4f} ce {ce_end:.4f} | dall "
            f"inst60 {cells['dall__install60']:.4f} held30 "
            f"{cells['dall__held30']:.4f} | grown {len(grown_end)} rows | "
            f"wte|Z| {zrow['wte_z_norm']:.3f}")
        cur_sd = {k: v.clone() for k, v in final_sd.items()}

    save_ckpt("e140_lcycled.pt", cur_sd,
              {"recipe": "3 locked-replay cycles, e119 E-arm battery "
                         "structure verbatim (E83.relearn, seed 24401 fresh "
                         "per cycle, 300-step batteries, chained ENDs), NO "
                         "row reset; base = e119_twin_start",
               "cycles": LC_CYCLES, "steps_per_cycle": LC_STEPS},
              ckpt_inventory)

    net_lc = evl_load(cur_sd)
    c_lc = census_rows(net_lc, ids130, zid)
    grid_lc = summarize_census(c_lc)
    grid_lc["dall_fixed"] = dall_fixed(net_lc, cur_sd, bat_ids, zid)
    grid_lc["ckpt"] = "e140_lcycled.pt"
    log(f"[L-cyc ] base {grid_lc['base_pz']:.4f} | row0 S "
        f"{grid_lc['row0']['strength']:.4f} rel {grid_lc['row0']['rel']:.3f} "
        f"| ctrl-max {grid_lc['control_max_strength']:.4f} | D-all-fix "
        f"{grid_lc['dall_fixed']['d_all_fixed_install60']:.4f}")
    del net_lc

    # =====================================================================
    # ADJUDICATION (registered bars, verbatim clauses)
    # =====================================================================
    S = {t: grid[t]["row0"]["strength"] for t, _, _, _ in CKPTS}
    REL = {t: grid[t]["row0"]["rel"] for t, _, _, _ in CKPTS}
    base = {t: grid[t]["base_pz"] for t, _, _, _ in CKPTS}

    inc1 = S["R@150"] - S["twin"]
    inc2 = S["R@300"] - S["R@150"]
    rise = bool(S["twin"] < S["R@150"] < S["R@300"]
                and inc1 > RISE_MIN_INC and inc2 > RISE_MIN_INC)
    flat_abs = bool(all(abs(S[f"E@c{c}"] - S["twin"]) <= FLAT_ABS_TOL
                        for c in (1, 2, 3)))
    flat_rel = bool(all(abs(REL[f"E@c{c}"] - REL["twin"]) <= FLAT_REL_TOL
                        for c in (1, 2, 3)))
    r_route_monotone = bool(rise and flat_abs)

    ratio = S["R@150"] / S["L@150"] if S["L@150"] > 0 else float("inf")
    l_flat_abs = bool(abs(S["L@150"] - S["twin"]) <= FLAT_ABS_TOL)
    l_flat_rel = bool(abs(REL["L@150"] - REL["twin"]) <= FLAT_REL_TOL)
    credit_assignment = bool(ratio >= RATIO_CREDIT and l_flat_abs)
    gradient_volume = bool(ratio < RATIO_GRADIENT)

    e_nulls = {f"E@c{c}": grid[f"E@c{c}"]["row0_null"] for c in (1, 2, 3)}
    e_never_routes = bool(all(e_nulls.values()))

    lc_dall = [cy["end_deletions_g0"]["cells"]["dall__install60"]
               for cy in lc_cycles]
    e_dall = [e119_ecyc[c]["end_deletions_g0"]["cells"]["dall__install60"]
              for c in sorted(e119_ecyc)] if len(e119_ecyc) == 3 else lc_dall
    if len(lc_dall) == 3:
        mono = bool(lc_dall[0] > lc_dall[1] > lc_dall[2])
        thins_like_e = bool(mono and lc_dall[2] <= THIN_FOLD_BAR * lc_dall[0])
        stays_fat = bool((not mono) or lc_dall[2] >= FAT_FLOOR * lc_dall[0])
    else:
        mono = thins_like_e = stays_fat = None    # smoke / partial: report only

    bars = {
        "r_route_monotone": {
            "fired": r_route_monotone,
            "rise_clause": {"pass": rise, "S_twin": S["twin"],
                            "S_R150": S["R@150"], "S_R300": S["R@300"],
                            "inc_R150": inc1, "inc_R300": inc2,
                            "min_increment": RISE_MIN_INC},
            "flat_clause_abs": {"pass": flat_abs, "tol": FLAT_ABS_TOL,
                                "S_E": {f"E@c{c}": S[f"E@c{c}"]
                                        for c in (1, 2, 3)}},
            "flat_clause_rel_secondary": {"pass": flat_rel, "tol": FLAT_REL_TOL,
                                          "rel_twin": REL["twin"],
                                          "rel_E": {f"E@c{c}": REL[f"E@c{c}"]
                                                    for c in (1, 2, 3)}},
            "note": "registered bar uses the absolute e131 strength; the "
                    "relative clause is the ceiling-honesty companion "
                    "(S is bounded by base_pz; the twin starts near "
                    "ceiling), reported, never substituted",
        },
        "credit_assignment_T079": {
            "fired": credit_assignment, "ratio_R150_over_L150": ratio,
            "bar_ratio": RATIO_CREDIT, "l_flat_abs": l_flat_abs,
            "l_flat_rel_secondary": l_flat_rel,
            "S_R150": S["R@150"], "S_L150": S["L@150"], "S_twin": S["twin"]},
        "gradient_volume": {
            "fired": gradient_volume, "ratio": ratio, "bar": RATIO_GRADIENT,
            "note": "fires iff ratio < 1.3 — row-0 dependence grows under "
                    "any training (sink gets the gradient regardless)"},
        "e_never_routes": {"fired": e_never_routes, "nulls": e_nulls,
                           "criterion": "e131 verbatim: not content OR "
                                        "strength <= control-max"},
        "l_cycled_discriminator": {
            "fired_thins_like_e": thins_like_e, "fired_stays_fat": stays_fat,
            "monotone": mono, "lc_dall_install60": lc_dall,
            "e_dall_install60_ref": e_dall,
            "fold_c3_over_c1": (lc_dall[2] / lc_dall[0]
                                if len(lc_dall) == 3 and lc_dall[0] > 0
                                else None),
            "e_fold_ref": (e_dall[2] / e_dall[0] if len(e_dall) == 3
                           and e_dall[0] > 0 else None),
            "bars": {"thins_like_e": f"strictly decreasing AND c3 <= "
                                     f"{THIN_FOLD_BAR} x c1",
                     "stays_fat": f"non-monotone OR c3 >= {FAT_FLOOR} x c1"},
            "verdict": ("CYCLE-DAMAGE (T078 'erasure digs in' loses its "
                        "only erasure-specific evidence)" if thins_like_e
                        else "ERASURE-SPECIFIC DIGGING SURVIVES" if stays_fat
                        else "AMBIGUOUS with numbers")},
        "ambiguous_zone": bool(RATIO_GRADIENT <= ratio < RATIO_CREDIT
                               and not credit_assignment),
    }

    # verdict headline (which bars fired; texture otherwise)
    fired = [k for k in ("r_route_monotone", "credit_assignment_T079",
                         "gradient_volume", "e_never_routes")
             if bars[k]["fired"]]
    if fired:
        verdict = "FIRED: " + ", ".join(fired)
    elif bars["ambiguous_zone"]:
        verdict = "TEXTURE (T079 ratio in the ambiguous zone 1.3-2; no bar)"
    else:
        verdict = "TEXTURE (no registered bar fired cleanly; numbers below)"

    log("=" * 78)
    log(f"E140 VERDICT: {verdict}")
    log(f"  S: " + " ".join(f"{t} {S[t]:.4f}" for t, _, _, _ in CKPTS)
        + f" | L-cycled {S.get('L-cycled', grid_lc['row0']['strength']):.4f}")
    log(f"  rel: " + " ".join(f"{t} {REL[t]:.3f}" for t, _, _, _ in CKPTS))
    log(f"  ratio R@150/L@150 = {ratio:.3f} (credit >= {RATIO_CREDIT}, "
        f"gradient < {RATIO_GRADIENT}) | L flat abs {l_flat_abs} rel "
        f"{l_flat_rel}")
    log(f"  rise {rise} (inc {inc1:+.4f}/{inc2:+.4f}) | flat abs {flat_abs} "
        f"rel {flat_rel} | E-nulls {e_nulls}")
    if len(lc_dall) == 3:
        log(f"  L-cycled dall: " + " -> ".join(f"{v:.4f}" for v in lc_dall)
            + f" (E ref " + " -> ".join(f"{v:.4f}" for v in e_dall) + ")"
            + f" -> {bars['l_cycled_discriminator']['verdict']}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e140_route_dependence",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("coordinator dispatch (T081-reworded e140 + T079 "
                         "adjudication + R44 L-cycled rider); bars and "
                         "operationalizations frozen in the module docstring "
                         "before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does the read route's row-0 PRESENCE-dependence grow "
                     "with jitter dose, and stay flat under locked replay "
                     "and erase cycles? (adjudicates T079 credit-assignment "
                     "vs gradient-volume; L-cycled arm adjudicates T078's "
                     "cycle-damage confound)"),
        "nets": {"main_grid": [f"runs/checkpoints/{f} (loaded, gated)"
                               for _, f, _, _ in CKPTS],
                 "lcycled": "runs/checkpoints/e140_lcycled.pt (this run, the "
                            "ONE allowed training)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_CKPT": gates_ck, "G_E116": G_E116,
                  "G_SURG_note": "every deletion via e131.deleted_wpe with "
                                 "its confinement gate; gates all passed "
                                 "(asserts would have raised)"},
        "main_grid": {t: {k: v for k, v in grid[t].items() if k != "all_rows"}
                      for t, _, _, _ in CKPTS},
        "main_grid_all_census_rows": {t: grid[t]["all_rows"]
                                      for t, _, _, _ in CKPTS},
        "strengths": {"S": S, "rel": REL, "base_pz": base,
                      "lcycled_final": {
                          "S": grid_lc["row0"]["strength"],
                          "rel": grid_lc["row0"]["rel"],
                          "base_pz": grid_lc["base_pz"],
                          "row0": grid_lc["row0"],
                          "control_max": grid_lc["control_max_strength"],
                          "dall_fixed": grid_lc["dall_fixed"],
                          "row0_null": grid_lc["row0_null"]}},
        "lcycled_arm": {"cycles": [{k: v for k, v in cy.items() if k != "traj"}
                                   for cy in lc_cycles],
                        "traj": {str(cy["cycle"]): cy["traj"]
                                 for cy in lc_cycles},
                        "recipe": "e119 E-arm battery structure VERBATIM "
                                  "(E83.relearn, locked pool jit_x[0], "
                                  "anchor60, seed 24401 fresh per cycle, "
                                  "300-step batteries chained END-to-END) "
                                  "minus the D2 row reset",
                        "row_reset_performed": False},
        "dall_texture_C": {"fixed_set": list(E113_ADDR_ROWS),
                           "per_checkpoint": {t: grid[t]["dall_fixed"]
                                              for t, _, _, _ in CKPTS},
                           "lcycled_final": grid_lc["dall_fixed"],
                           "note": "report-only"},
        "bars": bars,
        "adjudication": {"fired": fired, "verdict": verdict,
                         "headline": (f"S(0) twin {S['twin']:.3f} / R@150 "
                                      f"{S['R@150']:.3f} / R@300 "
                                      f"{S['R@300']:.3f} (rel {REL['twin']:.2f}"
                                      f"/{REL['R@150']:.2f}/{REL['R@300']:.2f}); "
                                      f"E@c1-3 " +
                                      "/".join(f"{S[f'E@c{c}']:.3f}"
                                               for c in (1, 2, 3)) +
                                      f"; L@150 {S['L@150']:.3f}; ratio "
                                      f"{ratio:.2f}; L-cycled dall " +
                                      (">".join(f"{v:.3f}" for v in lc_dall)
                                       if lc_dall else "n/a"))},
        "honesty_reflex": [
            "SINGLE-LINEAGE caveat: all arms descend from the ONE twin "
            "install (e119's degenerate twin reading) — route differences "
            "are route-caused within this lineage, but the lineage's "
            "install-specific idiosyncrasies (row-0 sink anatomy at "
            "install) are not sampled; e116's per-seed spread is the only "
            "cross-seed evidence.",
            "INSTRUMENT SENSITIVITY (presence-vs-norm): the e131 census "
            "measures REMOVAL-class necessity — e141 showed what kills is "
            "removal (presence), not content or direction, so S conflates "
            "'the route keys on row-0 presence' with 'row-0 is the "
            "load-bearing sink for ANY readout'. The twin already sits "
            "near the ceiling (S/base ~= "
            f"{REL['twin']:.2f}), so absolute growth under training "
            "largely tracks expression growth (base_pz), not route "
            "change; rel is the ceiling-honesty companion (reported, "
            "never substituted for the registered absolute bars).",
            "R@300's gate reference is a GPU-produced cell (tol 0.02), "
            "so its S carries a ~1e-2-scale numeric tolerance the other "
            "checkpoints do not.",
        ],
        "trims": trims,
        "deviations": deviations,
        "ckpt_inventory": {"saved": ckpt_inventory,
                           "external_used": [f"runs/checkpoints/{f}"
                                             for _, f, _, _ in CKPTS],
                           "note": "*.pt gitignored — on-disk persistence"},
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072, "device": "cpu",
                   "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "route_dependence.png", grid, grid_lc, S, REL, base, bars,
         verdict, lc_dall, e_dall, ratio)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'route_dependence.png'}, "
        f"ckpt runs/checkpoints/e140_lcycled.pt")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def plot(path, grid, grid_lc, S, REL, base, bars, verdict, lc_dall, e_dall,
         ratio):
    order = ["twin", "E@c1", "E@c2", "E@c3", "L@150", "R@150", "R@300"]
    xs = np.arange(len(order))
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.0))

    # (0,0) row-0 strength vs checkpoint + base ceiling + control band
    ax = axes[0, 0]
    sv = [S[t] for t in order]
    bv = [base[t] for t in order]
    ax.bar(xs, sv, 0.62, color=["tab:gray", "darkorange", "darkorange",
                                "darkorange", "steelblue", "crimson",
                                "crimson"],
           edgecolor="k", lw=0.5, label="row-0 presence-strength S")
    ax.plot(xs, bv, "D--", color="k", ms=5, lw=1.1, label="base p(Z) (the S ceiling)")
    cmax = max(grid[t]["control_max_strength"] for t in order)
    ax.axhline(cmax, color="gray", ls=":", lw=1.2,
               label=f"control-band max {cmax:.4f}")
    for x, t in zip(xs, order):
        ax.text(x, sv[x - 0] + 0.012, f"{sv[x]:.3f}\nrel {REL[t]:.2f}",
                ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=8.5)
    ax.set_ylabel("strength = min(mean-drop, zero-drop) at row 0")
    ax.set_ylim(0, 1.12)
    rise = bars["r_route_monotone"]["rise_clause"]
    flat = bars["r_route_monotone"]["flat_clause_abs"]
    ax.set_title("(A) MAIN GRID: row-0 presence-dependence per checkpoint\n"
                 f"rise clause {'PASS' if rise['pass'] else 'FAIL'} "
                 f"(inc {rise['inc_R150']:+.4f}/{rise['inc_R300']:+.4f} > "
                 f"{rise['min_increment']}) | flat(E) "
                 f"{'PASS' if flat['pass'] else 'FAIL'} "
                 f"(abs tol {flat['tol']}; rel clause "
                 f"{bars['r_route_monotone']['flat_clause_rel_secondary']['pass']})",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    # (0,1) T079 adjudication: R@150 vs L@150 + ratio bar
    ax = axes[0, 1]
    ax.bar([0], [S["R@150"]], 0.5, color="crimson", edgecolor="k",
           label="S(R@150) jittered")
    ax.bar([1], [S["L@150"]], 0.5, color="steelblue", edgecolor="k",
           label="S(L@150) locked")
    ax.bar([2], [ratio], 0.5, color="seagreen", edgecolor="k",
           label=f"ratio R/L = {ratio:.2f}")
    ax.axhline(2.0, color="seagreen", ls="--", lw=1.3,
               label="CREDIT-ASSIGNMENT bar (ratio >= 2)")
    ax.axhline(1.3, color="crimson", ls="--", lw=1.3,
               label="GRADIENT-VOLUME bar (ratio < 1.3)")
    for x, v, t in ((0, S["R@150"], "R@150"), (1, S["L@150"], "L@150")):
        ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.text(2, ratio + 0.05, f"{ratio:.2f}", ha="center", fontsize=8)
    ca, gv = bars["credit_assignment_T079"], bars["gradient_volume"]
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["S(R@150)", "S(L@150)", "ratio R/L"], fontsize=8.5)
    ax.set_ylim(0, 2.4)
    ax.set_title("(T079) credit-assignment vs gradient-volume\n"
                 f"CREDIT {'FIRES' if ca['fired'] else 'no'} (L flat abs "
                 f"{ca['l_flat_abs']}, rel {ca['l_flat_rel_secondary']}) | "
                 f"GRADIENT-VOLUME {'FIRES' if gv['fired'] else 'no'}",
                 fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")

    # (0,2) part C: fixed D-all survival per checkpoint
    ax = axes[0, 2]
    dv = [grid[t]["dall_fixed"]["d_all_fixed_install60"] for t in order]
    dv.append(grid_lc["dall_fixed"]["d_all_fixed_install60"])
    xs2 = np.arange(len(order) + 1)
    ax.bar(xs2, dv, 0.62,
           color=["tab:gray", "darkorange", "darkorange", "darkorange",
                  "steelblue", "crimson", "crimson", "mediumseagreen"],
           edgecolor="k", lw=0.5)
    ax.axhline(0.20, color="seagreen", ls="--", lw=1.1, label="survive bar 0.20")
    for x, v in zip(xs2, dv):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xs2)
    ax.set_xticklabels(order + ["L-cyc"], fontsize=8, rotation=20)
    ax.set_ylabel("p(Z) under D-all{121,125,129,133,137}")
    ax.set_ylim(0, 1.05)
    ax.set_title("(C) TEXTURE: fixed D-all (e113 set) survival",
                 fontsize=9.5)
    ax.legend(fontsize=7.5)

    # (1,0) L-cycled discriminator: per-cycle dall (log scale)
    ax = axes[1, 0]
    if len(lc_dall) == 3:
        cx = np.arange(3)
        ax.bar(cx - 0.19, e_dall, 0.36, color="darkorange", edgecolor="k",
               label="E (e119 stored): erase + relearn")
        ax.bar(cx + 0.19, lc_dall, 0.36, color="mediumseagreen",
               edgecolor="k", label="L-cycled (this run): NO row reset")
        for x, v in zip(cx - 0.19, e_dall):
            ax.text(x, v * 1.15, f"{v:.4f}", ha="center", fontsize=7.5)
        for x, v in zip(cx + 0.19, lc_dall):
            ax.text(x, v * 1.15, f"{v:.4f}", ha="center", fontsize=7.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.0)
        ax.set_xticks(cx)
        ax.set_xticklabels(["cycle 1 END", "cycle 2 END", "cycle 3 END"],
                           fontsize=9)
        d = bars["l_cycled_discriminator"]
        ax.set_title("(B) L-CYCLED DISCRIMINATOR: D-all residue per cycle "
                     "(install-60, log)\n"
                     + d["verdict"] +
                     f" (fold c3/c1 = {d['fold_c3_over_c1'] if d['fold_c3_over_c1'] is None else round(d['fold_c3_over_c1'], 4)}, "
                     f"E ref {d['e_fold_ref'] if d['e_fold_ref'] is None else round(d['e_fold_ref'], 4)})",
                     fontsize=8.5)
        ax.legend(fontsize=7.5)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"L-cycled dall per cycle: {[round(v, 4) for v in lc_dall]}",
                ha="center", fontsize=10)

    # (1,1) L-cycled comparison: final row-0 strength vs lineage endpoints
    ax = axes[1, 1]
    comp = ["twin", "E@c3", "L@150", "L-cycled"]
    svs = [S["twin"], S["E@c3"], S["L@150"], grid_lc["row0"]["strength"]]
    bvs = [base["twin"], base["E@c3"], base["L@150"], grid_lc["base_pz"]]
    xc = np.arange(len(comp))
    ax.bar(xc, svs, 0.55, color=["tab:gray", "darkorange", "steelblue",
                                 "mediumseagreen"], edgecolor="k", lw=0.5)
    ax.plot(xc, bvs, "D--", color="k", ms=5, lw=1.1, label="base p(Z)")
    for x, v, t in zip(xc, svs, comp):
        r = REL[t] if t in REL else grid_lc["row0"]["rel"]
        ax.text(x, v + 0.015, f"{v:.3f}\nrel {r:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(xc)
    ax.set_xticklabels(comp, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("row-0 presence-strength S")
    ax.set_title("(B) L-cycled final net: row-0 presence vs lineage endpoints\n"
                 f"900 locked steps, no erase (3 x 300, e119 E battery "
                 f"structure)", fontsize=9)
    ax.legend(fontsize=7.5)

    # (1,2) verdict panel
    ax = axes[1, 2]
    ax.axis("off")
    flat = bars["r_route_monotone"]["flat_clause_abs"]
    flr = bars["r_route_monotone"]["flat_clause_rel_secondary"]
    lc = bars["l_cycled_discriminator"]
    lines = [
        "VERDICT:",
        f"  {verdict}",
        "",
        "DECIDING NUMBERS:",
        f"  S: " + " ".join(f"{t}={S[t]:.4f}" for t in order)
        + f" L-cyc={grid_lc['row0']['strength']:.4f}",
        f"  rel: " + " ".join(f"{t}={REL[t]:.2f}" for t in order)
        + f" L-cyc={grid_lc['row0']['rel']:.2f}",
        f"  rise(twin<R150<R300, inc>0.005): "
        f"{bars['r_route_monotone']['rise_clause']['pass']}"
        f"  flat(E,abs 0.05): {flat['pass']}  flat(E,rel 0.10): {flr['pass']}",
        f"  R-ROUTE-MONOTONE: {bars['r_route_monotone']['fired']}",
        f"  ratio R@150/L@150 = {ratio:.3f} -> CREDIT "
        f"{bars['credit_assignment_T079']['fired']} | GRADIENT-VOLUME "
        f"{bars['gradient_volume']['fired']}",
        f"  E-NEVER-ROUTES: {bars['e_never_routes']['fired']} "
        f"(nulls {bars['e_never_routes']['nulls']})",
        f"  L-cycled dall c1>c2>c3: "
        + (f"{lc_dall[0]:.4f}>{lc_dall[1]:.4f}>{lc_dall[2]:.4f} "
           f"-> {lc['verdict']}" if len(lc_dall) == 3 else str(lc_dall)),
        "",
        "HONESTY:",
        "  single lineage (e119 twin); S bounded by base_pz —",
        "  twin starts near ceiling; rel is the companion read;",
        "  R@300 gate is a GPU ref (tol 0.02).",
    ]
    for i, tx in enumerate(lines):
        ax.text(0.02, 0.97 - i * 0.052, tx, fontsize=7.4, va="top",
                family="monospace",
                bbox=None if i else dict(facecolor="lightyellow",
                                         alpha=0.9, edgecolor="gray"))

    fig.suptitle("E140 — the route-dependence trace: row-0 presence across "
                 "jitter dose / erase cycles / locked replay", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
