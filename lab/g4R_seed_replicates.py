"""G4R — THE COMPASS + KNIFE SEED REPLICATES (R55's replication debt on g4's
two positives; dispatch brief 2026-09-29; bars = g4's registered P1/P4
clauses VERBATIM, scratch/g4_design.md sec 6 — no bar shopping).

CONTEXT (T127 + runs/g4/metrics.json): g4's dual-address net fired two
positives at n=1 install seed (10901):
  (1) COMPASS-IS-POSITIONAL — the NEAR arm built P-site content at rows
      5-13 (strength +0.2879 vs 2x shared-control max 0.0000) while the
      A-floor was never recruited (A-arm strength -0.0004; P1 HELD).
  (2) THE N2-CLASS HEAD-KNIFE — the per-census top-2 {L1H2, L3H0}
      flat-CE-killed the jitter (w8) fact: 98.8% drop at +0.179 CE
      (P4's b_n2 clause TRUE; the P4 whole was FALSIFIED elsewhere —
      D-A-all killed the jitter fact at +0.44 CE, a bound that stands
      and is NOT re-litigated here).
R55's rule (REVIEWS.md Review 55): "every g-positive is n=1 single seed
single lineage, while the e-series licensed nouns only at n>=3 — the
minting standard must match"; the owed debt: "seed replicates on every
g-positive (the >=3 rule before law-grade)".

THIS FILE pays it for the two g4 positives: 2 new install draws per
claim's cell —
  COMPASS cells: install seeds 10903 / 10904 -> install + NEAR arm
  KNIFE cells:   install seeds 10905 / 10906 -> install + w8 arm
EVERYTHING ELSE IS g4 VERBATIM: both pretrained roots are LOADED from
runs/checkpoints/g4_{dual,ctrl}_base.pt (no root retraining; the control
root is provenance only — both claims are dual-net claims), the corpus/
splice/pools/batteries rebuild bit-exactly (G_SPLICE gate), and every
model/instrument/recipe is lab/g4_dual_address.py's, IMPORTED and called
unmodified (DualGPT, finetune_arm, row_census, slot_census,
fact_read_slots, head_drops, the lesion knife). The downstream arm seeds
are g4's own (NEAR 10902, w8 10901): ONLY the install seed varies — this
is install-draw replication on a shared root/lineage, and the honesty
reflex says so.

REGISTERED BARS (frozen in this header before compute):
  COMPASS-REPLICATES fires iff BOTH compass draws place on P —
    near_site_pos (e116 criterion AND strength >= 2x shared-control max,
    e143 convention) AND A inert (A-arm strength < 0.5 x P-site strength)
    — the placement claim at n=3 (g4 = draw 1). Any miss = HONEST BOUND.
  KNIFE-REPLICATES fires iff BOTH knife draws' per-census N2 (top-2 by
    the g-12 head census, the e160 escalation convention; identities may
    differ per draw) flat-CE-kills the jitter fact — drop >= 60% at
    CE delta <= +0.35 — the surgery claim at n=3 (g4 = draw 1). Any
    miss = HONEST BOUND. Co-reports, never bars: N3/E4 escalation cells,
    g4's literal headset {L1H2, L3H0} transferred to each new net, and
    head-identity drift.
  Either claim failing = the honest bound, numbers reported.

CO-REPORTED TEXTURE (never bars): per-draw install gates (p(Z)@g0 vs
g4's failed-and-recorded [0.35, 0.75] band), install/w8 carrier
classify (g4's bars verbatim), NEAR row-0 vs this draw's install
baseline, NEAR novel-geometry mean (site-bound, the 0.10 bar as
texture), D-all at g0.

COMPUTE ENVELOPE: 8 short GPU trainings (300-step installs/arms, the g4
recipe's own ~10-25 s each) with gpu-gated launches + 60 s thermal
cooldowns, NO concurrent GPU; every census CPU-side on state-dict
snapshots (8 threads). SMOKE env G4R_SMOKE=1 runs the 8-step shakedown
(nothing adjudicated).

Outputs: runs/g4R/{metrics.json (flushed after every cell),
seed_replicates.png}; checkpoints runs/checkpoints/g4R_*.pt (gitignored).
No NOTES/THINKING/QUEUE/STATE/REVIEWS edits; single commit, no push.

Run:  cd lab && python g4R_seed_replicates.py    (G4R_SMOKE=1 shakedown)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # GPU lane for trainings

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
torch.set_num_threads(8)                              # e143 convention

import common                                          # noqa: E402
from common import (CharCorpus, cooldown, run_dir,     # noqa: E402
                    save_json)
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

# THE MACHINERY, VERBATIM: every model/instrument/recipe below is g4's own.
import g4_dual_address as G4                           # noqa: E402
from g4_dual_address import (                          # noqa: E402
    A_A_RIDES, A_INERT_FRAC, A_P_ALIVE, ARM_SEED, BLOCK, D_ALL, FLAT_CE_BAR,
    HEADS_EXPR_BAR, INSTALL_SEED, INSTALL_PZ_LO, INSTALL_PZ_HI, NAME,
    NEAR_ADDR_ROW, NEAR_CONT, NEAR_PRE, NEAR_ROWS, NEAR_SITE_ROWS,
    NEAR_FAR_SEED, NEAR_Z_XCOL, NOVEL0_BAR, NOVEL_GEO_COMPASS, POST_CAP,
    PRE, R_EVAL_SEED, READ_BAND, SHARED_CTR, SLOT_SEED, SURG_KILL,
    battery_cell, battery_pz, ce_fixed_cpu, deleted_wpe, evl_load,
    fact_read_slots, finetune_arm, head_ablate, head_drops, offset_grid,
    read_fact_at, row_census, slot_census, slot_usage, val_windows,
)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("G4R_SMOKE") == "1"
CPU = torch.device("cpu")
DEV = G4.DEV                       # g4's own lane decision (gpu_ok at import)

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

CKPT_DIR = E43.REPO / "runs" / "checkpoints"
HOSTS = ["FLORIZEL", "ELIZABETH"]
CORPUS_SEED = 1337
G4_N2_HEADSET = [(1, 2), (3, 0)]    # g4's literal knife {L1H2, L3H0}

# ---- the ONLY thing that changes vs g4: fresh install draws ---------------------
COMPASS_INSTALL_SEEDS = (10903, 10904)     # 2 new draws -> install + NEAR arm
KNIFE_INSTALL_SEEDS = (10905, 10906)       # 2 new draws -> install + w8 arm
# downstream arm seeds are g4's VERBATIM (mission: only the install seed changes)
NEAR_SEED = NEAR_FAR_SEED                  # 10902 (e143 L_SEED convention)
W8_SEED = ARM_SEED                         # 10901 (e147 ladder convention)
STEPS = 8 if SMOKE else 300                # g4's FT_STEPS, made explicit
COOLDOWN_S = 60.0                          # g4's dispatch envelope

CLAIM_BARS = {
    "compass": "FIRES iff both compass draws pass near_site_pos (e116 "
               "criterion AND strength >= 2x shared-control max) AND A "
               "inert (A-arm < 0.5 x P-site); n=3 with g4's draw; any miss "
               "= HONEST BOUND.",
    "knife": "FIRES iff both knife draws' per-census N2 (top-2 by the g-12 "
             "head census, e160 convention) kills the jitter fact (drop >= "
             "0.60 at CE delta <= +0.35); n=3 with g4's draw; any miss = "
             "HONEST BOUND.",
    "no_bar_shopping": "g4's registered clauses verbatim; texture stays "
                       "texture.",
}

deviations: list[str] = [
    "Only the INSTALL seed varies (10903-10906 vs g4's 10901); roots, "
    "corpus, protocol, pools, batteries and downstream arm seeds (NEAR "
    "10902, w8 10901) are g4's VERBATIM — this is install-draw replication "
    "on a shared root: n=3 on the install draw, n=1 on the root/lineage "
    "(the honesty reflex carries the bound).",
    "The knife's head-set is selected PER-CENSUS per draw (e160 escalation "
    "convention, as in g4's stage 5); g4's literal headset {L1H2, L3H0} is "
    "co-reported as a transfer cell, never adjudicated.",
    "g4's registered install in-band gate [0.35, 0.75] FAILED-and-was-"
    "recorded in g4 (dual 0.7521); g4R records the same fields as texture "
    "and proceeds identically — no new gate, no gate shopping.",
    "FAR arm, wash, locked-net surgery and the table-surgery cells are NOT "
    "replicated (not part of the two claims; g4's P4 FALSIFIED verdict — "
    "D-A-all killed the jitter fact at +0.44 CE — stands on its own).",
    "NEAR row-0 is reported against this draw's OWN install baseline "
    "(g4's midpoint construction degenerated to the baseline: its routed "
    "reference sat below it); the clause is texture either way.",
    "GPU float nondeterminism precedent (e119): fresh GPU trainings carry "
    "no bit-repro expectations; gates are internal plus the registered "
    "program gates.",
    "Smoke mode trims to 8-step trainings; nothing adjudicated.",
]

trims: list[str] = []
M: dict = {}


def flush(rd: Path):
    M["timing"] = {"total_s": round(time.time() - T0, 1)}
    save_json(rd / "metrics.json", E43.jsonable(M))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"g4R_{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g4R", **meta}}, path)
    log(f"[ckpt] saved {path.name}")


def classify(A_Pv, A_Av, expr):
    """g4 main()'s classify VERBATIM (carrier bars verbatim)."""
    if A_Pv is None:
        return "N/A (control has no A floor)"
    if A_Av is not None and A_Av >= A_A_RIDES and A_Pv < A_P_ALIVE:
        return "A"
    if A_Pv >= A_P_ALIVE and (A_Av is None or A_Av < A_A_RIDES):
        return "P"
    if expr >= HEADS_EXPR_BAR and A_Pv < A_P_ALIVE and \
            (A_Av is None or A_Av < A_A_RIDES):
        return "HEADS"
    return "UNRESOLVED"


def head_tag(tag: str):
    return int(tag[1:tag.index("H")]), int(tag[tag.index("H") + 1:])


def knife_cell(net, prim_ids, r_eval_xy, zid, heads, none_prim, ce_arm):
    """One knife cell — g4 stage-5's lesion-context arithmetic VERBATIM:
    ablate the head SET, read the fact battery + CE under the lesion."""
    ctxs = [common.lesion(net, "head", l, h) for (l, h) in heads]
    for c_ in ctxs:
        c_.__enter__()
    try:
        net.eval()
        pz = battery_pz(net, prim_ids, zid)
        ce_k = ce_fixed_cpu(net, *r_eval_xy)
    finally:
        for c_ in reversed(ctxs):
            c_.__exit__(None, None, None)
    drop = 1.0 - pz / max(none_prim, 1e-12)
    return {"heads": [f"L{l}H{h}" for (l, h) in heads], "pz_prim": pz,
            "ce_r": ce_k, "ce_delta": ce_k - ce_arm, "drop": drop,
            "flat_ce_kill": bool(drop >= SURG_KILL
                                 and (ce_k - ce_arm) <= FLAT_CE_BAR)}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g4R_smoke" if SMOKE else "g4R")
    log(f"G4R THE COMPASS + KNIFE SEED REPLICATES (smoke={SMOKE}) -> {rd}")
    log(f"compute: train device {DEV} (g4 import lane), cpu threads "
        f"{torch.get_num_threads()}")

    # ---- g4 reference (the n=1 draw; pulled from runs/g4/metrics.json) ----
    g4_metrics_path = E43.REPO / "runs" / "g4" / "metrics.json"
    g4m = json.loads(g4_metrics_path.read_text(encoding="utf-8"))
    dn = g4m["stage3"]["dual_near"]
    G4_REF = {
        "install_seed": g4m["seeds"]["install"],
        "dual_near": {"site_strength": dn["site_strength"],
                      "control_max": dn["control_max"],
                      "A_arm_strength": dn["A_arm_strength"],
                      "row0": dn["row0"], "site_pos": dn["site_pos"]},
        "P1_clauses": g4m["stage3"]["P1"]["clauses"],
        "P1_verdict": g4m["stage3"]["P1"]["verdict"],
        "near_census_rows": dn["census"]["rows"],
        "knife_N2": g4m["stage5"]["jitter"]["knife"]["cells"]["N2"],
        "knife_ranking": g4m["stage5"]["jitter"]["knife"]["ranking"][:4],
        "P4_b_n2_kills_jitter": g4m["stage5"]["P4"]["b_n2_kills_jitter"],
        "install_dual": {"pz_g0": g4m["stage1"]["dual"]["pz_g0"],
                         "A_P": g4m["stage1"]["dual"]["A_P"],
                         "A_A": g4m["stage1"]["dual"]["A_A"],
                         "carrier": g4m["stage1"]["carrier_observed_dual"]},
        "dual_root_val_ce": g4m["stage0"]["gates"]["G_CE"]["dual_val_ce"],
    }
    assert G4_REF["knife_N2"]["heads"] == ["L1H2", "L3H0"], G4_REF["knife_N2"]

    M.update({
        "experiment": "g4R_seed_replicates",
        "date": common.now_iso(),
        "design": "dispatch brief 2026-09-29; bars = g4's registered P1/P4 "
                  "clauses verbatim (scratch/g4_design.md sec 6)",
        "claims": "COMPASS placement (P1) + N2 head-knife (P4b) seed "
                  "replicates — R55's >=3 rule",
        "claim_bars": CLAIM_BARS,
        "smoke": SMOKE,
        "provenance": {
            "machinery": "lab/g4_dual_address.py imported VERBATIM (models, "
                         "instruments, recipes); nothing re-implemented "
                         "except this file's driver",
            "design_lineage": "e043 install / e113 pools / e131 census / "
                              "e133+e160 knife / e143 compass / e147 w8 "
                              "ladder — all via the g4 import",
            "g4_reference_metrics": str(g4_metrics_path.relative_to(E43.REPO))
                                    .replace("\\", "/"),
            "g4_reference": "inlined below under 'g4_reference'",
        },
        "seeds": {"corpus": CORPUS_SEED, "splice_rng": E43.SPLICE_RNG,
                  "compass_installs": list(COMPASS_INSTALL_SEEDS),
                  "knife_installs": list(KNIFE_INSTALL_SEEDS),
                  "near_arm": NEAR_SEED, "w8_arm": W8_SEED,
                  "r_eval": R_EVAL_SEED, "slot_usage": SLOT_SEED,
                  "g4_install_for_reference": INSTALL_SEED},
        "deviations": deviations, "trims": trims,
        "g4_reference": G4_REF,
    })
    flush(rd)

    # ================= protocol rebuild (g4 main() VERBATIM subset) =======
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=CORPUS_SEED)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    assert train_text.count("ZEPHYRA") == 0 and val_text.count("ZEPHYRA") == 0

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(
        mix == {"FLORIZEL": 19, "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    name_ids = corpus.encode(NAME)
    L = len(NAME)
    log(f"protocol rebuilt: install60 {mix}, held30 "
        f"(SPLICE_RNG {E43.SPLICE_RNG}) — bit-exact with g4")

    # install pool (e043 windows, offset 0) + anchor bank (g4 verbatim)
    wins = []
    for p, h in install_occ:
        w = torch.cat([train_ids[p - PRE: p], name_ids,
                       train_ids[p + len(h): p + len(h) + POST_CAP]])
        assert len(w) == BLOCK
        wins.append(w)
    pool0_x = torch.stack(wins)
    pool0_mask = torch.zeros(len(wins), BLOCK - 1, dtype=torch.bool)
    pool0_mask[:, PRE - 1: PRE - 1 + L] = True
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])

    # w8 pool (g4's ladder pool, grid offset_grid(8), VERBATIM)
    grid = offset_grid(8)
    ws_, ms_ = [], []
    for j in grid:
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            win = torch.cat([pre, name_ids, post])
            assert len(win) == BLOCK and torch.equal(
                win[PRE + j: PRE + j + L], name_ids)
            ws_.append(win)
            m = torch.zeros(BLOCK - 1, dtype=torch.bool)
            m[PRE - 1 + j: PRE - 1 + j + L] = True
            ms_.append(m)
    pool_w8_x, pool_w8_mask = torch.stack(ws_), torch.stack(ms_)

    # compass NEAR pool (e143 truncation, g4 verbatim)
    near_wins = []
    for p, h in install_occ:
        pre = train_ids[p - NEAR_PRE: p]
        post = train_ids[p + len(h): p + len(h) + NEAR_CONT]
        w = torch.cat([pre, name_ids, post])
        assert len(w) == BLOCK
        near_wins.append(w)
    pool_near_x = torch.stack(near_wins)
    pool_near_mask = torch.zeros(len(near_wins), BLOCK - 1, dtype=torch.bool)
    pool_near_mask[:, NEAR_ADDR_ROW: NEAR_ADDR_ROW + L] = True

    # batteries (e068 construction; the js g4R reads)
    bat_ids = {}
    for j in sorted(set(NOVEL_GEO_COMPASS) | {0}):
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[(0, "install60")]
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    M["protocol"] = {"corpus_seed": CORPUS_SEED, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "w8_grid": list(grid),
                     "batteries": "ctx = train_text[p-PRE-j:p], p(Z) at last "
                                  "position (wpe row 129+j)"}
    M["gates"] = {"G_SPLICE": G_SPLICE}
    flush(rd)

    # ================= roots: LOADED from disk (no retraining) ===========
    ck_dual = CKPT_DIR / "g4_dual_base.pt"
    ck_ctrl = CKPT_DIR / "g4_ctrl_base.pt"
    dual_ckpt = torch.load(ck_dual, map_location="cpu")
    ctrl_ckpt = torch.load(ck_ctrl, map_location="cpu")
    sd_dual_base = dual_ckpt["model"]
    netD = evl_load(sd_dual_base, dual=True).to(DEV)   # common.DEVICE lane
    ce_root = common.estimate_loss(netD, corpus, "val", n_batches=10)
    del netD
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    g4_root_ce = G4_REF["dual_root_val_ce"]
    G_ROOT = {"root": "runs/checkpoints/g4_dual_base.pt (LOADED, g4's "
                      "pretrained dual root — no retraining)",
              "sha256_16": sha256_of(ck_dual),
              "meta": dual_ckpt["meta"],
              "val_ce_this_load": ce_root, "g4_recorded_val_ce": g4_root_ce,
              "diff": ce_root - g4_root_ce, "bar": 0.05,
              "pass": bool(abs(ce_root - g4_root_ce) <= 0.05)}
    ctrl_meta = {"root": "runs/checkpoints/g4_ctrl_base.pt (provenance "
                         "only — both g4R claims are dual-net claims)",
                 "sha256_16": sha256_of(ck_ctrl),
                 "meta": ctrl_ckpt["meta"]}
    log(f"roots loaded: dual val CE {ce_root:.4f} vs g4's {g4_root_ce:.4f} "
        f"(diff {ce_root - g4_root_ce:+.4f}, bar 0.05): "
        f"{'PASS' if G_ROOT['pass'] else 'FAIL'}")
    assert G_ROOT["pass"], "root identity drift (not g4's dual root?)"
    M["roots"] = {"dual": G_ROOT, "ctrl": ctrl_meta}
    M["gates"]["G_ROOT"] = G_ROOT
    flush(rd)

    rep = {"compass": {}, "knife": {}}

    def install_texture(sd: dict) -> dict:
        """The g4 stage-1 readouts, minimal set (texture + carrier)."""
        netI = evl_load(sd, dual=True)
        pz_g0 = battery_cell(netI, ids130, zid)["mean_pz"]
        pz_held = battery_cell(netI, bat_ids[(0, "held30")], zid)["mean_pz"]
        cen = row_census(netI, (0, 129),
                         lambda n: battery_pz(n, ids130, zid))
        mass_rb, usage_rb = slot_usage(netI, train_ids, READ_BAND,
                                       2000 if not SMOKE else 200, SLOT_SEED)
        sc = slot_census(netI, usage_rb["top3_slots"],
                         lambda n: battery_pz(n, ids130, zid))
        carrier = classify(cen["rows"]["129"]["strength"], sc["strength"],
                           pz_g0)
        out = {"pz_g0": pz_g0, "pz_held_g0": pz_held,
               "ce_r": ce_fixed_cpu(netI, *r_eval_xy),
               "A_P": cen["rows"]["129"]["strength"],
               "row0": cen["rows"]["0"]["strength"],
               "census_rows_0_129": cen["rows"],
               "slot_usage_read_band": usage_rb, "A_A": sc["strength"],
               "carrier_observed": carrier,
               "in_band_gate": {"lo": INSTALL_PZ_LO, "hi": INSTALL_PZ_HI,
                                "pass": bool(INSTALL_PZ_LO <= pz_g0
                                             <= INSTALL_PZ_HI),
                                "note": "record-only (g4's own dual install "
                                        "sat at 0.7521, out of band)"}}
        del netI
        return out

    # ================= COMPASS DRAWS (install -> NEAR -> census) =========
    for i, seed in enumerate(COMPASS_INSTALL_SEEDS, 1):
        tag = f"compass_r{i}"
        log("=" * 78)
        log(f"COMPASS DRAW r{i} (install seed {seed})")
        net0 = evl_load(sd_dual_base, dual=True)
        res = finetune_arm(f"{tag}_install", net0, pool0_x, pool0_mask,
                           anchor, train_ids, r_eval_xy, ids130, zid, seed,
                           steps=STEPS)
        sd_i = res["sd"]
        save_ckpt(f"{tag}_install", sd_i,
                  {"desc": f"g4 dual root + {STEPS}-step masked install "
                           f"(offset 0, seed {seed}) — g4R compass draw {i}",
                   "steps": res["steps_ran"], "seed": seed,
                   "base": "runs/checkpoints/g4_dual_base.pt"})
        inst = install_texture(sd_i)
        log(f"install[{tag}]: p(Z)@g0 {inst['pz_g0']:.4f} held "
            f"{inst['pz_held_g0']:.4f} CE_R {inst['ce_r']:.4f} | A_P "
            f"{inst['A_P']:+.4f} A_A {inst['A_A']:+.4f} carrier "
            f"{inst['carrier_observed']}")
        del net0
        cooldown(COOLDOWN_S)

        net0c = evl_load(sd_i, dual=True)
        res_n = finetune_arm(f"{tag}_near", net0c, pool_near_x,
                             pool_near_mask, anchor, train_ids, r_eval_xy,
                             pool_near_x[:, :NEAR_PRE], zid, NEAR_SEED,
                             steps=STEPS)
        sd_n = res_n["sd"]
        save_ckpt(f"{tag}_near", sd_n,
                  {"desc": f"compass draw {i} installed root + {STEPS}-step "
                           f"locked NEAR replay (x-cols 6..12, seed "
                           f"{NEAR_SEED})", "steps": res_n["steps_ran"],
                   "seed": NEAR_SEED,
                   "base": f"runs/checkpoints/g4R_{tag}_install.pt"})
        net = evl_load(sd_n, dual=True)

        def own_read(n):
            return read_fact_at(n, pool_near_x, name_ids, zid, NEAR_ADDR_ROW,
                                NEAR_Z_XCOL)["pz_onset_mean"]

        cen = row_census(net, NEAR_ROWS, own_read)
        ctrl_max = max(cen["rows"][str(r)]["strength"] for r in SHARED_CTR)
        site_str = max(cen["rows"][str(r)]["strength"]
                       for r in NEAR_SITE_ROWS)
        site_pos = any(cen["rows"][str(r)]["content"] and
                       cen["rows"][str(r)]["strength"] >=
                       2.0 * max(ctrl_max, 0.0) for r in NEAR_SITE_ROWS)
        ks, freq = fact_read_slots(net, pool_near_x[:, :NEAR_Z_XCOL + 1],
                                   NEAR_ADDR_ROW)
        asc = slot_census(net, ks, own_read)
        a_arm = asc["strength"]
        a_inert = bool(a_arm < A_INERT_FRAC * max(site_str, 1e-9))
        novel = {g: battery_pz(net, bat_ids[(g, "install60")], zid)
                 for g in NOVEL_GEO_COMPASS}
        novel_mean = float(np.mean(list(novel.values())))
        sd_d, gtd = deleted_wpe(sd_n, D_ALL)
        assert gtd["pass"]
        net.load_state_dict(sd_d)
        dall_g0 = battery_pz(net, ids130, zid)
        net.load_state_dict(sd_n)
        del sd_d
        row0 = cen["rows"]["0"]["strength"]
        rep["compass"][f"r{i}"] = {
            "install_seed": seed, "near_seed": NEAR_SEED,
            "install": inst, "near_traj": res_n["traj"],
            "census": cen, "control_max": ctrl_max,
            "site_rows": list(NEAR_SITE_ROWS), "site_strength": site_str,
            "site_pos": bool(site_pos), "A_arm_strength": a_arm,
            "A_arm_census": asc, "A_arm_slots": ks,
            "A_arm_slot_freq": freq, "A_inert": a_inert,
            "row0": row0, "row0_at_install_baseline": bool(
                row0 <= inst["row0"]),
            "novel_geos": novel, "novel_mean": novel_mean,
            "novel_site_bound_texture": bool(novel_mean <= NOVEL0_BAR),
            "D_all_g0": dall_g0,
            "ce_r": ce_fixed_cpu(net, *r_eval_xy),
            "PASS": bool(site_pos and a_inert),
        }
        c = rep["compass"][f"r{i}"]
        log(f"compass[{tag}]: site {site_str:+.4f} (2x-ctrl "
            f"{2 * ctrl_max:.4f}) site_pos {site_pos} | A-arm "
            f"{a_arm:+.4f} (inert bar {0.5 * max(site_str, 1e-9):.4f}) "
            f"A_inert {a_inert} | row0 {row0:+.4f} (base "
            f"{inst['row0']:+.4f}) novel-mean {novel_mean:.4f} | "
            f"PASS {c['PASS']}")
        del net, net0c
        cooldown(COOLDOWN_S)
        M["replicates"] = rep
        flush(rd)

    # ================= KNIFE DRAWS (install -> w8 -> knife) ==============
    for i, seed in enumerate(KNIFE_INSTALL_SEEDS, 1):
        tag = f"knife_r{i}"
        log("=" * 78)
        log(f"KNIFE DRAW r{i} (install seed {seed})")
        net0 = evl_load(sd_dual_base, dual=True)
        res = finetune_arm(f"{tag}_install", net0, pool0_x, pool0_mask,
                           anchor, train_ids, r_eval_xy, ids130, zid, seed,
                           steps=STEPS)
        sd_i = res["sd"]
        save_ckpt(f"{tag}_install", sd_i,
                  {"desc": f"g4 dual root + {STEPS}-step masked install "
                           f"(offset 0, seed {seed}) — g4R knife draw {i}",
                   "steps": res["steps_ran"], "seed": seed,
                   "base": "runs/checkpoints/g4_dual_base.pt"})
        inst = install_texture(sd_i)
        log(f"install[{tag}]: p(Z)@g0 {inst['pz_g0']:.4f} CE_R "
            f"{inst['ce_r']:.4f} | A_P {inst['A_P']:+.4f} A_A "
            f"{inst['A_A']:+.4f} carrier {inst['carrier_observed']}")
        del net0
        cooldown(COOLDOWN_S)

        net0w = evl_load(sd_i, dual=True)
        res_w = finetune_arm(f"{tag}_w8", net0w, pool_w8_x, pool_w8_mask,
                             anchor, train_ids, r_eval_xy, ids130, zid,
                             W8_SEED, steps=STEPS)
        sd_w = res_w["sd"]
        save_ckpt(f"{tag}_w8", sd_w,
                  {"desc": f"knife draw {i} installed root + {STEPS}-step "
                           f"jitter w=8 replay (grid {list(grid)}, seed "
                           f"{W8_SEED})", "steps": res_w["steps_ran"],
                   "seed": W8_SEED,
                   "base": f"runs/checkpoints/g4R_{tag}_install.pt"})
        net = evl_load(sd_w, dual=True)
        prim_ids = bat_ids[(-12, "install60")]
        none_prim = battery_pz(net, prim_ids, zid)
        ce_arm = ce_fixed_cpu(net, *r_eval_xy)
        # carrier texture (g4 stage-2 readouts, minimal set)
        cen_w = row_census(net, (0, 129),
                           lambda n: battery_pz(n, ids130, zid))
        mass_rb, usage_rb = slot_usage(net, train_ids, READ_BAND,
                                       2000 if not SMOKE else 200, SLOT_SEED)
        sc_w = slot_census(net, usage_rb["top3_slots"],
                           lambda n: battery_pz(n, ids130, zid))
        carrier_w8 = classify(cen_w["rows"]["129"]["strength"],
                              sc_w["strength"], none_prim)
        # the knife: per-census top heads -> N2 (bar) + N3/E4 + g4-headset
        hc = head_drops(net, prim_ids, zid)
        ranked = sorted(hc["drops"].items(), key=lambda kv: -kv[1])
        top_heads = [head_tag(t) for t, _ in ranked[:4]]
        cells = {}
        for (l, h) in top_heads[:2]:
            pz = head_ablate(net, prim_ids, zid, [(l, h)])
            cells[f"L{l}H{h}"] = {"pz_prim": pz,
                                  "drop": 1.0 - pz / max(none_prim, 1e-12)}
        for k, n in (("N2", 2), ("N3", 3), ("E4", 4)):
            cells[k] = knife_cell(net, prim_ids, r_eval_xy, zid,
                                  top_heads[:n], none_prim, ce_arm)
            log(f"knife[{tag}/{k}]: {cells[k]['heads']} drop "
                f"{100 * cells[k]['drop']:.1f}% dCE "
                f"{cells[k]['ce_delta']:+.4f} flat-CE kill "
                f"{cells[k]['flat_ce_kill']}")
        cells["g4_headset_transfer"] = knife_cell(
            net, prim_ids, r_eval_xy, zid, G4_N2_HEADSET, none_prim, ce_arm)
        g4t = cells["g4_headset_transfer"]
        log(f"knife[{tag}/g4-headset]: {g4t['heads']} drop "
            f"{100 * g4t['drop']:.1f}% dCE {g4t['ce_delta']:+.4f} kill "
            f"{g4t['flat_ce_kill']}")
        rep["knife"][f"r{i}"] = {
            "install_seed": seed, "w8_seed": W8_SEED, "install": inst,
            "w8_traj": res_w["traj"], "pz_gm12": none_prim,
            "ce_arm": ce_arm, "A_P_w8": cen_w["rows"]["129"]["strength"],
            "row0_w8": cen_w["rows"]["0"]["strength"],
            "A_A_w8": sc_w["strength"],
            "slot_usage_read_band": usage_rb, "carrier_w8": carrier_w8,
            "head_census": hc, "ranking_top8": [t for t, _ in ranked[:8]],
            "knife_cells": cells,
            "PASS": bool(cells["N2"]["flat_ce_kill"]),
        }
        log(f"knife[{tag}]: expression g-12 {none_prim:.4f} carrier "
            f"{carrier_w8} | per-census N2 {cells['N2']['heads']} | PASS "
            f"{rep['knife'][f'r{i}']['PASS']}")
        del net, net0w
        cooldown(COOLDOWN_S)
        M["replicates"] = rep
        flush(rd)

    # ================= ADJUDICATION ======================================
    compass_pass = {k: v["PASS"] for k, v in rep["compass"].items()}
    knife_pass = {k: v["PASS"] for k, v in rep["knife"].items()}
    compass_fires = all(compass_pass.values())
    knife_fires = all(knife_pass.values())
    verdicts = {
        "compass": {
            "bars": CLAIM_BARS["compass"], "per_draw": compass_pass,
            "g4_draw1": {"site_strength": G4_REF["dual_near"]["site_strength"],
                         "A_arm": G4_REF["dual_near"]["A_arm_strength"],
                         "site_pos": G4_REF["dual_near"]["site_pos"],
                         "P1_verdict": G4_REF["P1_verdict"]},
            "fires": compass_fires,
            "verdict": (
                "COMPASS-REPLICATES FIRE — the placement claim stands at "
                "n=3 on the install-draw axis (g4 + 2 fresh installs, all "
                "placing on P with the A floor inert)"
                if compass_fires else
                "HONEST BOUND — a compass draw missed the registered "
                "placement bar; COMPASS-IS-POSITIONAL stays n<3 and keeps "
                "its n=1 scope clause (numbers in replicates)"),
        },
        "knife": {
            "bars": CLAIM_BARS["knife"], "per_draw": knife_pass,
            "g4_draw1": G4_REF["knife_N2"],
            "fires": knife_fires,
            "verdict": (
                "KNIFE-REPLICATES FIRE — the N2-class flat-CE head-knife "
                "stands at n=3 on the install-draw axis (per-census top-2 "
                "killed the jitter fact in every draw)"
                if knife_fires else
                "HONEST BOUND — a knife draw's per-census N2 missed the "
                "flat-CE kill; the head-knife keeps its n=1 scope clause "
                "(escalation cells co-reported in replicates)"),
        },
    }
    verdicts["honesty_reflex"] = [
        "REPLICATED AXIS: the INSTALL draw (fresh 300-step installs from "
        "the SAME pretrained dual root). NOT varied: the root/lineage (n=1 "
        "— seed replication of the roots is a different, unrun cell), the "
        "corpus, the protocol, the downstream arm seeds, and (no wash "
        "cell here) any wash seed. R55's >=3 is satisfied on the install "
        "axis only; the lineage bound (T113) still applies.",
        "The roots were LOADED from g4's checkpoints (sha256 recorded; val "
        "CE re-verified against g4's recorded value, |diff| <= 0.05) — no "
        "root retraining, so g4R measures install+arm variance, not "
        "pretraining variance.",
        f"Device: trainings on {DEV} (gpu-gated per launch, 60 s thermal "
        "cooldowns, no concurrent GPU); every census CPU-side on "
        "state-dict snapshots — g4's own compute conventions.",
        "The knife's head identities are selected per-census per draw "
        "(e160); identity drift is REPORTED, not adjudicated — the claim "
        "is the N2-class kill, not the specific heads.",
        "g4's in-run negative controls are not re-run here (spine, "
        "control root, FAR arm); the compass draws carry their own "
        "shared-control band (2x-max bar) inside every census.",
    ]
    M["verdicts"] = verdicts
    log(f"ADJUDICATION: compass fires={compass_fires} "
        f"({compass_pass}) | knife fires={knife_fires} ({knife_pass})")
    flush(rd)

    if not SMOKE:
        plot_all(rd)
    flush(rd)
    log(f"outputs: {rd}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

def plot_all(rd: Path):
    g4r = M["g4_reference"]
    comp = M["replicates"]["compass"]
    knf = M["replicates"]["knife"]
    labels_c = ["g4\n(seed 10901)"] + [f"r{i}\n(seed {s})" for i, s in
                                       enumerate(COMPASS_INSTALL_SEEDS, 1)]
    labels_k = ["g4\n(seed 10901)"] + [f"r{i}\n(seed {s})" for i, s in
                                       enumerate(KNIFE_INSTALL_SEEDS, 1)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (0,0) NEAR census spectra
    ax = axes[0, 0]
    specs = [("g4 (draw 1)", g4r["near_census_rows"], "dimgray")] + \
        [(f"compass r{i}", comp[f"r{i}"]["census"]["rows"],
          ("crimson" if comp[f"r{i}"]["PASS"] else "darkorange"))
         for i in (1, 2)]
    for lbl, rows, col in specs:
        xs = sorted(int(r) for r in rows)
        ax.plot(xs, [rows[str(r)]["strength"] for r in xs], "o-", ms=3,
                color=col, label=lbl)
    ax.axvspan(5, 13, color="crimson", alpha=0.07, label="NEAR site 5-13")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("wpe row"); ax.set_ylabel("census strength")
    ax.set_title("(i) NEAR site-content census — the P floor carries it",
                 fontsize=10)
    ax.legend(fontsize=7)

    # (0,1) placement bars: site vs A-arm
    ax = axes[0, 1]
    site_v = [g4r["dual_near"]["site_strength"]] + \
        [comp[f"r{i}"]["site_strength"] for i in (1, 2)]
    aarm_v = [g4r["dual_near"]["A_arm_strength"] or 0.0] + \
        [comp[f"r{i}"]["A_arm_strength"] for i in (1, 2)]
    bar2 = [0.5 * s for s in site_v]         # the inert bar itself
    w = 0.27
    ax.bar(np.arange(3) - w, site_v, w, color="crimson", label="P-site str")
    ax.bar(np.arange(3), aarm_v, w, color="steelblue", label="A-arm str")
    ax.bar(np.arange(3) + w, bar2, w, color="none", edgecolor="navy",
           ls="--", label="inert bar (0.5x P-site)")
    for i in range(3):
        ax.annotate(f"2xctrl={2 * ([g4r['dual_near']['control_max']] + [comp[f'r{j}']['control_max'] for j in (1, 2)])[i]:.4f}",
                    (i - w, site_v[i]), fontsize=6, ha="center",
                    va="bottom" if site_v[i] >= 0 else "top")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels_c, fontsize=8)
    ax.set_title(f"(ii) placement: {M['verdicts']['compass']['verdict']}"
                 .split(" — ")[0] + " — "
                 + ("FIRES" if M["verdicts"]["compass"]["fires"]
                    else "BOUND"), fontsize=10)
    ax.legend(fontsize=7)

    # (1,0) knife drops
    ax = axes[1, 0]
    n2_v = [g4r["knife_N2"]["drop"]] + \
        [knf[f"r{i}"]["knife_cells"]["N2"]["drop"] for i in (1, 2)]
    tr_v = [np.nan] + [knf[f"r{i}"]["knife_cells"]["g4_headset_transfer"]
                       ["drop"] for i in (1, 2)]
    n3_v = [np.nan] + [knf[f"r{i}"]["knife_cells"]["N3"]["drop"]
                       for i in (1, 2)]
    ax.bar(np.arange(3), [100 * v for v in n2_v], 0.45, color="crimson",
           label="per-census N2 kill")
    ax.bar(np.arange(3) + 0.5, [100 * v if v == v else 0 for v in n3_v],
           0.45, color="darkorange", alpha=0.6, label="N3 (escalation)")
    ax.bar(np.arange(3) - 0.5, [100 * v if v == v else 0 for v in tr_v],
           0.45, color="steelblue", alpha=0.6, hatch="//",
           label="g4 headset {L1H2,L3H0} transfer")
    ax.axhline(100 * SURG_KILL, ls="--", color="navy", lw=1,
               label="kill bar 60%")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels_k, fontsize=8)
    ax.set_ylabel("% fact drop at g-12")
    ids = [g4r["knife_N2"]["heads"]] + \
        [knf[f"r{i}"]["knife_cells"]["N2"]["heads"] for i in (1, 2)]
    for i, hs in enumerate(ids):
        ax.annotate(",".join(hs), (i, 100 * n2_v[i]), fontsize=6,
                    ha="center", va="bottom")
    ax.set_title("(iii) the N2-class head-knife on the jitter fact",
                 fontsize=10)
    ax.legend(fontsize=7)

    # (1,1) verdict text
    ax = axes[1, 1]
    ax.axis("off")
    ces = [g4r["knife_N2"]["ce_delta"]] + \
        [knf[f"r{i}"]["knife_cells"]["N2"]["ce_delta"] for i in (1, 2)]
    lines = [
        "G4R — THE COMPASS + KNIFE SEED REPLICATES (install draws "
        "10903-10906; roots reused)",
        "",
        f"COMPASS: {'FIRES at n=3' if M['verdicts']['compass']['fires'] else 'HONEST BOUND'}",
        "  P-site str:  g4 "
        f"{g4r['dual_near']['site_strength']:+.4f} | r1 "
        f"{comp['r1']['site_strength']:+.4f} | r2 "
        f"{comp['r2']['site_strength']:+.4f}",
        "  A-arm str:   g4 "
        f"{(g4r['dual_near']['A_arm_strength'] or 0.0):+.4f} | r1 "
        f"{comp['r1']['A_arm_strength']:+.4f} | r2 "
        f"{comp['r2']['A_arm_strength']:+.4f}",
        f"  site_pos: r1 {comp['r1']['site_pos']} r2 "
        f"{comp['r2']['site_pos']} | A_inert: r1 "
        f"{comp['r1']['A_inert']} r2 {comp['r2']['A_inert']}",
        "",
        f"KNIFE: {'FIRES at n=3' if M['verdicts']['knife']['fires'] else 'HONEST BOUND'}",
        "  N2 kill %:   g4 "
        f"{100 * g4r['knife_N2']['drop']:.1f} | r1 "
        f"{100 * knf['r1']['knife_cells']['N2']['drop']:.1f} | r2 "
        f"{100 * knf['r2']['knife_cells']['N2']['drop']:.1f}",
        f"  N2 dCE:      g4 {g4r['knife_N2']['ce_delta']:+.3f} | r1 "
        f"{ces[1]:+.3f} | r2 {ces[2]:+.3f} (bar <= +0.35)",
        "  N2 heads:    g4 "
        f"{','.join(g4r['knife_N2']['heads'])} | r1 "
        f"{','.join(knf['r1']['knife_cells']['N2']['heads'])} | r2 "
        f"{','.join(knf['r2']['knife_cells']['N2']['heads'])}",
        "",
        "n=3 on the INSTALL draw only; root/lineage n=1 (R55 clause); "
        "censuses CPU; trainings GPU-gated.",
    ]
    for i, tx in enumerate(lines):
        ax.text(0.02, 0.97 - i * 0.062, tx, fontsize=8.6, va="top",
                family="monospace",
                bbox=dict(facecolor="lightyellow", alpha=0.9,
                          edgecolor="gray")
                if tx.startswith(("COMPASS", "KNIFE")) else None)
    fig.suptitle("G4R — compass + knife seed replicates (R55's >=3 rule; "
                 "g4 draw 1 + two fresh install draws per claim)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "seed_replicates.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
