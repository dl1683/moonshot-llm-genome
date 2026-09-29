"""G1BR — THE WALL SEED REPLICATES (R55's owed-debt #1; T125's claim at n>=3).

Context (T125): g1b's anchored ball (one commit event + a hard L2 projection
at R=0.7) held the consolidated fact FLAT at 0.918 through the 300-step
neutral wash that kills the control at +2 — but n=1, a single wash-draw seed
(10902, the locked lineage). The lab's own >=3 rule (e184, g2d) applies to
the WALL-HOLDS claim: an architectural positive is not licensed on one draw.

THE WHOLE DIFFERENCE from g1b: the wash-draw seed. Two fresh seeds (10907,
10908 — 10903/10904 are g2d's wash seeds, 10905/10906 e152r's straddle
seeds; 10907/10908 are the next free draws of the locked 109xx lineage), each
running the claim's own two arms:
    C_<seed>  uncommitted neutral wash 300 (the per-seed control: the wash
              that must still kill)
    W1_<seed> commit(0.7) + the identical-per-seed neutral wash 300
Everything else is lab/g1b_continuity.py VERBATIM via import: the same
bit-exact e131_consolidated root, the same protocol rebuild (e152's j=54
pool, e170's neutral bank, e119's batteries, the e131 dial set), the same
g1_wash arithmetic, instruments, bars and adjudication conventions. W2/W3 and
the noise arms are NOT re-run — the claim under replication is WALL-HOLDS
(W1-at-0.7 vs C), not the cliff localization or the noise split (each
already adjudicated in g1b at n=1 with their own texture).

THE FROZEN BARS (registered before the run; no shopping):
  maintains (per g1/g1b VERBATIM) = light g-12 >= 0.50 at EVERY checkpoint
      {1,2,4,10,50,100,200,300}; dies = g-12 <= 0.27.
  WALL-REPLICATES  = both new seeds' W1 maintain (and every gate passes)
      -> the WALL-HOLDS claim licensed at n=3 (10902 + the two new draws).
  WALL-SEED-BOUND  = any seed's W1 fails (honest bound; report which seed
      and its trace — the claim re-scoped to the seed lottery, g2e's
      ROOT-DRAW-BOUND precedent).
  Co-reported per seed: FLAT-AT-PIN (|g(+300) - g(+50)| <= 0.05), the +2
  dip, held30/anatomy at +300 (full dials {2,50,300} on W1, g1b VERBATIM).

GATES (any failure => ABORT TO TEXTURE, nothing adjudicated):
  G-ROOT  the root loads bit-exact and g-12 >= 0.78 (g1b's own gate; the
          same checkpoint, the same dial).
  G-CTRL  EACH seed's C kills by +50 (the wash is lethal at every draw —
          the contrast the claim needs).
  G-PIN   each W1's raw displacement <= 0.7 + 1.5 at every checkpoint.
  G-BITROOT each W1 root's body is bit-identical to theta0 (anchors copies).
  G-INPUTS per-step input batches bit-identical (md5) between each seed's
          C and W1 through +300 (the commit is the only delta), and the
          two seeds' streams DIFFER (the seed change did something).
  G-STEP1 each seed's W1 step-1 body == C's step-1 body (max|diff| <= 1e-4;
          the wall cannot act at forward 1 — d=0 < R — so any difference
          would be an implementation failure, modulo device-float fuzz).

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier; the
stated reason is g1b's own: CONTINUITY on the e131 line); 4 trainings x
300 steps, each <= 180 s GPU / 1800 s CPU; cooldown 60 s before each; NO
concurrent GPU; mid-run contention poll every 25 steps (all inherited from
g1's policy, run verbatim).

Outputs: runs/g1bR/{metrics.json, seed_replicates.png}; checkpoints
runs/checkpoints/g1bR_*.pt (each arm's +300 final). No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python g1bR_seed_replicates.py    (G1BR_SMOKE=1 shakedown)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

SMOKE = os.environ.get("G1BR_SMOKE") == "1"
if SMOKE:
    os.environ["G1B_SMOKE"] = "1"     # cascades to G1_SMOKE inside g1b's import

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e152R/e143/e184/e179

import common                                          # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # jsonable)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1:
                                                      # it sets G1_SMOKE (in
                                                      # smoke mode) before
                                                      # g1_anchored_ball's
                                                      # first import, then
                                                      # applies the
                                                      # G1.G1_CFG/G1_PARAMS
                                                      # -> 2.74M patch and
                                                      # exposes GB.CKPT_DIR /
                                                      # GB.ROOT_CK verbatim
import g1_anchored_ball as G1                          # noqa: E402 — the
                                                      # machinery (patched to
                                                      # 2.74M by g1b's import)

from common import CharCorpus, cooldown, run_dir, save_json, set_seed   # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)
G1.log = log                                          # unify the timeline

# ======================================================================
# THE CONFIG DELTA (the whole difference from g1b)
# ======================================================================
NEW_SEEDS = (10907, 10908)      # the wash-draw seeds (see docstring provenance)
R_CLAIM = 0.7                   # the claim's radius (g1b's W1, VERBATIM)
REF_METRICS = E43.REPO / "runs" / "g1b" / "metrics.json"   # the n=1 reference

G1BR_PREDICTION = {
    "bars": ("maintains = light g-12 >= 0.50 at EVERY checkpoint "
             "{1,2,4,10,50,100,200,300} (g1/g1b VERBATIM); dies = g-12 <= "
             "0.27. WALL-REPLICATES = both new seeds' W1 maintain and every "
             "gate passes -> the WALL-HOLDS claim licensed at n=3. "
             "WALL-SEED-BOUND = any seed's W1 fails (honest bound, report "
             "which)."),
    "predicted": ("WALL-REPLICATES (T125's claim, registered): each new "
                  "seed's W1 maintains with a floor in g1b's registered band "
                  "0.55-0.85 (reference seed 10902: min 0.7768 at +2, flat "
                  "0.918 at +300, FLAT-AT-PIN delta 0.0026), while each "
                  "seed's C dies by +50 (reference: dead at +2, 0.0271)."),
    "falsifier": ("any seed's W1 drops below 0.50 at any checkpoint through "
                  "+300 (the well is a seed lottery, not an architecture) "
                  "or below 0.27 (the kill pierces at a fresh draw) — "
                  "WALL-SEED-BOUND, reported per seed."),
}

deviations: list[str] = [
    "ARMS: g1b's seven reduced to the claim's own two per seed (C + W1) — "
    "the replicated claim is WALL-HOLDS (W1-at-0.7 vs C, g1 spec section 8 "
    "priority (1): 'seeds x3 on W1 and C'). W2/W3/N0/N1/N2 are g1b's cells "
    "(cliff localization, noise split) and are not re-run; their n=1 "
    "verdicts stand as texture.",
    "SEEDS: 10907/10908 (the next free draws of the locked 109xx lineage — "
    "10903/10904 are g2d's wash seeds, 10905/10906 e152r's straddle seeds). "
    "The ONLY stochastic delta from g1b: g1_wash's seed argument. Same "
    "root, same neutral bank (seed 170), same targets ('true'), same "
    "checkpoints, same lr.",
    "ROOT: loaded bit-exact from the same e131_consolidated_e113.pt and "
    "gated (G-ROOT, G-BITEXACT) but measured LEAN (base batteries + CE_R + "
    "site read): the root is bit-identical to g1b's by gate, and its full "
    "dial (census, deletions) is g1b's record — re-measuring it re-draws "
    "nothing.",
    "DIALS: full dials {2,50,300} on the W1 arms only (g1b's FULL_DIAL_WASH "
    "VERBATIM); C arms run light-only (the control's role here is the kill "
    "clock, read by the light g-12; g1b's C full dials stand).",
    "CHECKPOINTS: each arm's +300 final only (g1b's wash-arm convention; "
    "the +2 dip states are bit-reproducible from the recorded seeds).",
    "G-STEP1 (new gate, strengthens g1b's docstring claim to a checked "
    "gate): each seed's W1 step-1 body vs C's step-1 body, max|diff| <= "
    "1e-4 — the wall cannot act at forward 1 (d=0 < R) and the input "
    "streams are md5-identical, so this verifies the commit is inert until "
    "forward 2, modulo device-float fuzz (tolerance, not bit-exactness, "
    "because GPU reductions are order-nondeterministic).",
    "MACHINERY: lab/g1b_continuity.py VERBATIM via import (which owns the "
    "G1.G1_CFG/G1.G1_PARAMS -> 2.74M patch); measure()/flat_cells are "
    "g1b's closures copied VERBATIM (they close over the batteries, as in "
    "g1b and g1 before it).",
    "DEVICE RECORD FIX (after the first full run, re-run end-to-end): the "
    "first run's static honesty text claimed 'same device class (cuda)' "
    "while the machine-recorded device_events showed all four arms "
    "migrated CPU mid-run on the registered thermal policy (81-83C). The "
    "honesty_reflex.device text was corrected to state the mix truthfully "
    "(the events were recorded correctly all along); NO bar, gate, "
    "adjudication or training logic changed; the experiment was re-run "
    "end-to-end from the stored root with identical seeds (numbers "
    "re-drawn within device-float fuzz, story unchanged) — g1b's own "
    "clause-template precedent.",
    "Smoke mode trims: g1's (via G1B_SMOKE -> G1_SMOKE: 4-step washes, "
    "ckpts {1,2,4}, lean dials, no cooldowns) — nothing adjudicated.",
]

CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = GB.CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g1bR", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g1bR_smoke" if SMOKE else "g1bR")
    log(f"G1BR THE WALL SEED REPLICATES (smoke={SMOKE}) -> {rd}")
    set_seed(G1.INSTALL_SEED)          # g1's opening seed (global init only)

    # ---------------- reference (the n=1 cell this run replicates)
    ref = json.loads(REF_METRICS.read_text(encoding="utf-8"))
    ref_wall = ref["adjudication"]["wall"]
    REF = {
        "source": "runs/g1b/metrics.json",
        "seed": G1.FREEZE_SEED,
        "verdict": ref["adjudication"]["verdict"],
        "W1_g_m12": ref_wall["W1"]["g_m12"],
        "W1_min": ref_wall["W1"]["min_gm12"],
        "C_g_m12": ref_wall["C"]["g_m12"],
        "C_min": ref_wall["C"]["min_gm12"],
        "W1_flat_delta": ref["adjudication"]["flat_delta"],
        "traces": ref["traces"],
        "disp_table": ref["displacement"]["table"],
        "device": ref["arms"]["W1"]["device"],
    }
    log(f"reference loaded: g1b seed {G1.FREEZE_SEED} W1 min "
        f"{REF['W1_min']:.4f} (+300 {REF['W1_g_m12']['300']:.4f}), C dead at "
        f"+2 ({REF['C_g_m12']['2']:.4f}) — device {REF['device']}")

    # ---------------- protocol rebuild (g1b's main VERBATIM)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "pass": bool(corpus_zeph == 0)}
    assert G_NAMEFREE["pass"], f"corpus contains ZEPH x{corpus_zeph}"

    host_occ = []
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
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
    name_ids = corpus.encode(G1.NAME)

    # measurement pool: e152's locked j=54 windows (instrument only)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - G1.PRE - G1.RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + G1.SITE_CONT]
        if len(pre) != G1.PRE + G1.RETEACH_J or len(post) != G1.SITE_CONT:
            raise RuntimeError(f"pool window short at p={p}")
        w = torch.cat([pre, name_ids, post])
        if len(w) != G1.BLOCK:
            raise RuntimeError(f"pool window len {len(w)} != {G1.BLOCK}")
        wins.append(w)
    pool_x = torch.stack(wins)
    G_POOL = {"shape": list(pool_x.shape),
              "name_xcols": [G1.SITE_Z_XCOL, G1.SITE_Z_XCOL + len(G1.NAME) - 1],
              "all_windows_name_in_place": bool(
                  all(torch.equal(w[G1.SITE_Z_XCOL: G1.SITE_Z_XCOL + len(G1.NAME)],
                                  name_ids) for w in pool_x)),
              "note": "MEASUREMENT instrument only (e152's locked j=54 pool); "
                      "NO window from this pool enters any training"}
    G_POOL["pass"] = G_POOL["all_windows_name_in_place"]
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    # ---------------- the neutral stream (e170's construction VERBATIM)
    arng = random.Random(G1.E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - G1.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        txt = train_text[s: s + G1.BLOCK + 1]
        if any(f in txt for f in G1.ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + G1.BLOCK] for s in n_starts])
    G_ANCHOR = {"neutral_bank": {"seed": G1.E170_ANCHOR_SEED,
                                 "n_windows": 16, "starts": n_starts,
                                 "note": "e170's construction VERBATIM via "
                                         "g1b (= e176n arm A's stream; FIXED "
                                         "content, shared by ALL arms here)"},
                "budget": list(anchor_neutral.shape)}
    G_ANCHOR["pass"] = bool(anchor_neutral.shape == (16, G1.BLOCK))
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{G1.BLOCK} (seed {G1.E170_ANCHOR_SEED}): PASS")

    # ---------------- batteries (e119/e176n verbatim)
    bat_ids, held_ids = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids = bat_ids[-12]
    g0_ids = bat_ids[0]

    gates_surg: dict = {}

    def measure(sd: dict, tag: str, lean: bool = False) -> dict:
        """g1b's measure() VERBATIM dial (e176n's = the e131 dial set) on
        evl_load (settle+disarm; g1's PIVOT)."""
        net = G1.evl_load(sd)
        sd_local = {k: v.detach().clone() for k, v in net.state_dict().items()}
        out: dict = {"tag": tag}
        out["base"] = {j: G1.battery_cell(net, bat_ids[j], zid) for j in G1.GEOS}
        out["base_held"] = {j: G1.battery_cell(net, held_ids[j], zid)
                            for j in G1.GEOS}
        out["ce_r"] = G1.ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in G1.GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in G1.GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        out["site_read"] = G1.read_fact_at(net, pool_x, name_ids, zid,
                                           G1.SITE_ADDR_ROW, G1.SITE_Z_XCOL)
        log(f"[{tag}] site read @183: onset "
            f"{out['site_read']['pz_onset_mean']:.4f} span "
            f"{out['site_read']['pname_mean_over7']:.4f}")
        if not lean and not SMOKE:
            out["census_old"] = G1.row_census(net, G1.ROWS_OLD,
                                              lambda n: G1.battery_pz(
                                                  n, bat_ids[0], zid))
            co = out["census_old"]["rows"]
            out["old_band"] = {
                "base_pz": out["census_old"]["base_readout"],
                "row0_strength": co["0"]["strength"],
                "A129": co["129"]["strength"],
                "band121_129_max": max(co[str(r)]["strength"]
                                       for r in range(121, 130)
                                       if str(r) in co)}
            log(f"[{tag}] old band: row0 S "
                f"{out['old_band']['row0_strength']:+.4f} | A(129) "
                f"{out['old_band']['A129']:+.4f}")
            DELS = {"d_all": G1.D_ALL, "d183": (G1.SITE_ADDR_ROW,)}
            out["del_table"] = {}
            for dl, rows_ in DELS.items():
                sd_d, gate = G1.deleted_wpe(sd_local, rows_)
                gates_surg[f"{tag}__{dl}"] = gate
                if not gate["pass"]:
                    raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: "
                                       f"{gate}")
                net.load_state_dict(sd_d)
                cell = {"g0": G1.battery_cell(net, bat_ids[0], zid)["mean_pz"]}
                if dl == "d183":
                    cell["gm12"] = G1.battery_cell(net, bat_ids[-12],
                                                   zid)["mean_pz"]
                out["del_table"][dl] = cell
            net.load_state_dict(sd_local)
            log(f"[{tag}] deletions g0: " + " | ".join(
                f"{dl} {out['del_table'][dl]['g0']:.3f}" for dl in DELS))
        else:
            w = net.wpe.weight.data
            orig = w.clone()
            mean_row = orig.mean(0)
            bp = G1.battery_pz(net, bat_ids[0], zid)
            w[129] = mean_row
            m129 = bp - G1.battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            w[129] = 0.0
            z129 = bp - G1.battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            assert torch.equal(w, orig), "lean A129 failed to restore wpe"
            out["A129_quick"] = float(min(m129, z129))
        del net
        return out

    def flat_cells(m: dict) -> dict:
        c = {"gm12": m["base"][-12]["mean_pz"],
             "g0": m["base"][0]["mean_pz"],
             "gp12": m["base"][12]["mean_pz"],
             "held30_gm12": m["base_held"][-12]["mean_pz"],
             "held30_g0": m["base_held"][0]["mean_pz"],
             "ce_r": m["ce_r"],
             "site_read_onset": m["site_read"]["pz_onset_mean"],
             "site_read_span": m["site_read"]["pname_mean_over7"]}
        if "old_band" in m:
            c["A129"] = m["old_band"]["A129"]
            c["row0_strength"] = m["old_band"]["row0_strength"]
            c["dall_g0"] = m["del_table"]["d_all"]["g0"]
            c["d183_g0"] = m["del_table"]["d183"]["g0"]
            c["d183_gm12"] = m["del_table"]["d183"]["gm12"]
        else:
            c["A129"] = m["A129_quick"]
        return c

    # =====================================================================
    # THE ROOT — the arc's consolidated root, loaded DIRECTLY (g1b VERBATIM)
    # =====================================================================
    log("=" * 78)
    root_net = G1.load_g1(GB.CKPT_DIR / GB.ROOT_CK)
    assert root_net.num_params() == GB.G1B_PARAMS, \
        f"root param count {root_net.num_params()} != {GB.G1B_PARAMS}"
    theta0 = {k: v.detach().clone() for k, v in root_net.state_dict().items()}
    raw = torch.load(GB.CKPT_DIR / GB.ROOT_CK, map_location="cpu",
                     weights_only=False)
    raw_sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    md0 = max(float((theta0[k].float() - raw_sd[k].float()).abs().max())
              for k in raw_sd)
    G_BITEXACT = {"checkpoint": f"runs/checkpoints/{GB.ROOT_CK}",
                  "n_tensors": len(raw_sd),
                  "max_abs_diff_vs_file": md0, "pass": bool(md0 == 0.0)}
    assert G_BITEXACT["pass"], f"root load not bit-exact: {md0}"
    log(f"root loaded bit-exact from {GB.ROOT_CK} ({GB.G1B_PARAMS} params)")

    root = measure(theta0, "g1bR_root", lean=True)
    root_cells = flat_cells(root)
    G_ROOT0 = {"bar": G1.EXPRESS_BAR, "gm12": root_cells["gm12"],
               "reference_gm12": REF["traces"]["W1"][0]["gm12"],
               "pass": bool(root_cells["gm12"] >= G1.EXPRESS_BAR)}
    log(f"G-ROOT: root g-12 {root_cells['gm12']:.4f} "
        f"(bar >= {G1.EXPRESS_BAR}; g1b's reading "
        f"{G_ROOT0['reference_gm12']:.4f}): "
        f"{'PASS' if G_ROOT0['pass'] else 'FAIL'}")
    if not G_ROOT0["pass"] and not SMOKE:
        log("G-ROOT FAILED — arms still run for the record; verdict will be "
            "TEXTURE (gate failure), nothing adjudicated")

    # =====================================================================
    # THE ARMS — per new seed: C (uncommitted) + W1 (commit R=0.7); g1b's
    # arm loop VERBATIM, the ONLY delta g1_wash(seed=NEW_SEEDS[i])
    # =====================================================================
    ARM_SPECS = []
    for seed in NEW_SEEDS:
        ARM_SPECS.append((f"C{seed}", None, seed,
                          f"CONTROL seed {seed} — uncommitted neutral wash "
                          f"(e176N arm A VERBATIM at this draw): the "
                          f"per-seed kill clock"))
        ARM_SPECS.append((f"W1_{seed}", R_CLAIM, seed,
                          f"WALL R={R_CLAIM} seed {seed} — commit({R_CLAIM}) "
                          f"then the seed-identical neutral wash; step-1 "
                          f"weights equal to C{seed}'s (the wall first acts "
                          f"at forward 2)"))

    arms: dict = {}
    batteries_all: dict = {}
    G_BITROOT = {}
    for tag, R, seed, desc in ARM_SPECS:
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {G1.COOLDOWN_S:.0f}s before {tag}")
            cooldown(G1.COOLDOWN_S)
        log(f"ARM {tag} — {desc}")
        net0 = G1.evl_load(theta0) if R is None else G1.CommittedGPT(GB.G1B_CFG)
        if R is not None:
            net0.load_state_dict(theta0)
            net0.commit(R)
            body, _ = G1.split_anchored_sd(net0.state_dict())
            md = max(float((body[k].float() - theta0[k].float()).abs().max())
                     for k in theta0)
            anch_ok = all(torch.equal(net0._anchor(n), p.detach())
                          for n, p in net0.named_parameters())
            G_BITROOT[tag] = {"max_abs_diff": md,
                              "anchors_bit_equal": bool(anch_ok),
                              "n_anchor_tensors": net0._n_anchor_tensors,
                              "pass": bool(md == 0.0 and anch_ok)}
            assert G_BITROOT[tag]["pass"], f"{tag}: wall root != theta0"
            log(f"G_BITROOT[{tag}]: max|diff| {md:.1e}, anchors bit-equal: "
                f"PASS")
        arm = G1.g1_wash(tag, net0, anchor_neutral, train_ids, itos,
                         r_eval_xy, gm12_ids, g0_ids, zid, target_mode="true",
                         noise_seed=0, ckpt_steps=G1.CK_WASH, seed=seed)
        G_DRAWFREE = {"zeph_violations": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert G_DRAWFREE["pass"], f"{tag}: name token leaked into a window"
        gates_surg[f"G_DRAWFREE_{tag}"] = G_DRAWFREE
        arms[tag] = arm

        # checkpoint: the arm's final state
        for s in sorted(arm["sds"]):
            if s == max(arm["sds"]):
                save_ckpt(f"g1bR_{tag}_s{s}", arm["sds"][s],
                          {"desc": f"e131 root + {s}-step true-target neutral "
                                   f"wash (R={R}, input seed {seed}, lr "
                                   f"{G1.FT_LR})",
                           "steps": int(s), "R": R, "input_seed": seed,
                           "lr": G1.FT_LR, "target_mode": "true",
                           "base": f"runs/checkpoints/{GB.ROOT_CK}"})

        # full dials: W1 arms at g1b's FULL_DIAL_WASH {2,50,300}; C light-only
        full_at = ([s for s in G1.FULL_DIAL_WASH if s in arm["sds"]]
                   if R is not None else [])
        batteries = {}
        for s in full_at:
            log(f"{tag} +{s} full dial")
            batteries[str(s)] = measure(arm["sds"][s], f"{tag}{s}", lean=SMOKE)
        batteries_all[tag] = batteries

    # =====================================================================
    # CONSTRUCTION-FIDELITY GATES
    # =====================================================================
    # G-INPUTS: within each seed, C and W1 share bit-identical per-step
    # inputs (the commit is the only delta); across seeds the streams differ.
    G_INPUTS = {"within_seed": {}, "across_seed": {}, "pass": None}
    for seed in NEW_SEEDS:
        hc = arms[f"C{seed}"]["x_hashes"]
        hw = arms[f"W1_{seed}"]["x_hashes"]
        shared = sorted(set(hc) & set(hw))
        G_INPUTS["within_seed"][str(seed)] = {
            "steps_compared": len(shared),
            "identical": bool(len(shared) > 0
                              and all(hc[s] == hw[s] for s in shared))}
    h1 = arms[f"C{NEW_SEEDS[0]}"]["x_hashes"]
    h2 = arms[f"C{NEW_SEEDS[1]}"]["x_hashes"]
    shared12 = sorted(set(h1) & set(h2))
    G_INPUTS["across_seed"] = {
        "steps_compared": len(shared12),
        "differing_steps": sum(1 for s in shared12 if h1[s] != h2[s]),
    }
    G_INPUTS["pass"] = bool(all(v["identical"]
                                for v in G_INPUTS["within_seed"].values())
                            and G_INPUTS["across_seed"]["differing_steps"]
                            == G_INPUTS["across_seed"]["steps_compared"]
                            and G_INPUTS["across_seed"]["steps_compared"] > 0)
    assert G_INPUTS["pass"], f"input stream gate FAILED: {G_INPUTS}"
    log(f"G_INPUTS: per-step inputs bit-identical within each seed pair "
        f"(C==W1, md5), and all {G_INPUTS['across_seed']['differing_steps']}/"
        f"{G_INPUTS['across_seed']['steps_compared']} steps differ across "
        f"seeds: PASS")

    # G-STEP1: the wall is inert at forward 1 — W1's step-1 body == C's
    # step-1 body (same seed), to device-float fuzz.
    G_STEP1 = {"per_seed": {}}
    for seed in NEW_SEEDS:
        body_w, _ = G1.split_anchored_sd(arms[f"W1_{seed}"]["sds"][1])
        sd_c = arms[f"C{seed}"]["sds"][1]
        md1 = max(float((body_w[k].float() - sd_c[k].float()).abs().max())
                  for k in sd_c)
        G_STEP1["per_seed"][str(seed)] = {
            "max_abs_diff": md1, "tol": 1e-4,
            "pass": bool(md1 <= 1e-4)}
    G_STEP1["pass"] = bool(all(v["pass"] for v in G_STEP1["per_seed"].values()))
    assert G_STEP1["pass"], f"G-STEP1 FAILED (wall acted at forward 1?): {G_STEP1}"
    log("G-STEP1: W1 step-1 bodies == C step-1 bodies per seed (max|diff| "
        + ", ".join(f"{v['max_abs_diff']:.1e}"
                    for v in G_STEP1['per_seed'].values())
        + " <= 1e-4): PASS — the wall is inert until forward 2")

    # =====================================================================
    # DISPLACEMENT TABLES (e185's currency, measured)
    # =====================================================================
    def disp_rows(tag):
        return [{"step": t["step"], "ce_batch": t["ce_batch"],
                 "cum_disp": t["cum_disp"], "step_disp": t["step_disp"],
                 "d_proj": t["d_proj"],
                 **({"g_m12_light": t["g_m12_mean_pz"],
                     "ce_r_light": t["ce_r"]}
                    if "g_m12_mean_pz" in t else {})}
                for t in arms[tag]["traj"]]

    disp_table = {tag: disp_rows(tag) for tag in arms}

    # =====================================================================
    # ADJUDICATION (g1's arm_verdict logic VERBATIM; the n>=3 composition)
    # =====================================================================
    def light_gm12(tag):
        return {t["step"]: t["g_m12_mean_pz"] for t in arms[tag]["traj"]
                if "g_m12_mean_pz" in t}

    def arm_verdict(tag, cks):
        g = light_gm12(tag)
        vals = [g[s] for s in cks if s in g]
        maintains = bool(vals and all(v >= G1.MAINTAIN_BAR for v in vals))
        dies_by_50 = bool(g.get(50, 1.0) <= G1.SHUT_BAR)
        first_under = next((s for s in cks if g.get(s, 1.0) <= G1.SHUT_BAR),
                           None)
        return {"g_m12": g, "min_gm12": min(vals) if vals else None,
                "argmin_step": (min(g, key=lambda s: g[s])
                                if vals else None),
                "maintains": maintains, "dies_by_50": dies_by_50,
                "first_ck_le_bar": first_under}

    # ---- G-CTRL: EVERY seed's C kills by +50
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    G_CTRL = {"per_seed": {}}
    for seed in NEW_SEEDS:
        g = light_gm12(f"C{seed}")
        G_CTRL["per_seed"][str(seed)] = {
            "gm12_at_50": g.get(50),
            "earliest_le_bar": next((s for s in G1.CK_WASH if s > 0
                                     and g.get(s, 1.0) <= G1.SHUT_BAR), None),
            "pass": bool(g.get(50, 1.0) <= G1.SHUT_BAR)}
    G_CTRL["pass"] = bool(all(v["pass"] for v in G_CTRL["per_seed"].values()))
    log("G-CTRL: " + " | ".join(
        f"seed {s}: C dead at +{v['earliest_le_bar']} "
        f"(+50 {fmt(v['gm12_at_50'])})"
        for s, v in G_CTRL["per_seed"].items())
        + f": {'PASS' if G_CTRL['pass'] else 'FAIL'}")

    # ---- G-PIN: each W1's raw displacement <= R + 1.5 at every ckpt
    G_PIN = {"per_arm": {}, "fuzz_formula_at_this_size": float(
        G1.FT_LR * (GB.G1B_PARAMS ** 0.5))}
    for seed in NEW_SEEDS:
        tag = f"W1_{seed}"
        rows = [r for r in disp_table[tag] if "g_m12_light" in r]
        mx = max(r["cum_disp"] for r in rows) if rows else None
        G_PIN["per_arm"][tag] = {
            "R": R_CLAIM, "bound": R_CLAIM + G1.PIN_FUZZ_BAR,
            "max_raw_disp_at_ckpt": mx,
            "per_ckpt": {r["step"]: r["cum_disp"] for r in rows},
            "pass": bool(mx is not None and mx <= R_CLAIM + G1.PIN_FUZZ_BAR)}
        log(f"G-PIN[{tag}]: max raw |d| at ckpt {mx:.4f} <= "
            f"{R_CLAIM + G1.PIN_FUZZ_BAR:.2f}: "
            f"{'PASS' if G_PIN['per_arm'][tag]['pass'] else 'FAIL'}")
    G_PIN["pass"] = bool(all(v["pass"] for v in G_PIN["per_arm"].values()))

    gates_pass = bool(G_ROOT0["pass"] and G_CTRL["pass"] and G_PIN["pass"]
                      and G_INPUTS["pass"] and G_STEP1["pass"]
                      and all(v["pass"] for v in G_BITROOT.values()))

    # ---- the per-seed WALL verdicts + FLAT-AT-PIN
    wall = {tag: arm_verdict(tag, G1.CK_WASH)
            for tag in (f"C{s}" for s in NEW_SEEDS)}
    wall.update({f"W1_{s}": arm_verdict(f"W1_{s}", G1.CK_WASH)
                 for s in NEW_SEEDS})
    flat = {}
    for seed in NEW_SEEDS:
        g = light_gm12(f"W1_{seed}")
        d = abs(g.get(300, float("nan")) - g.get(50, float("nan"))) \
            if 50 in g and 300 in g else None
        seq = [g[s] for s in (50, 100, 200, 300) if s in g]
        mono = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
        flat[str(seed)] = {
            "flat_delta": d,
            "FLAT_AT_PIN": bool(d is not None and d <= G1.FLAT_BAR),
            "F4_ERODES": bool(len(seq) >= 2 and mono
                              and (seq[0] - seq[-1]) >= G1.ERODE_BAR),
            "dip_at_2": g.get(2), "g300": g.get(300),
        }

    # ---- the composed verdict (the mission's frozen dichotomy)
    seed_maintains = {str(s): wall[f"W1_{s}"]["maintains"] for s in NEW_SEEDS}
    all_maintain = all(seed_maintains.values())
    ref_maintains = bool(ref_wall["W1"]["maintains"])

    if not gates_pass:
        failed = [k for k, g in (("G-ROOT", G_ROOT0), ("G-CTRL", G_CTRL),
                                 ("G-PIN", G_PIN), ("G-INPUTS", G_INPUTS),
                                 ("G-STEP1", G_STEP1),
                                 ("G-BITROOT", {"pass": all(
                                     v["pass"] for v in G_BITROOT.values())})
                                 ) if not g["pass"]]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a construction gate failed — nothing adjudicated; "
                  f"failed: {failed}")
    elif all_maintain:
        verdict = "WALL-REPLICATES"
        clause = (f"both new seeds MAINTAIN at R={R_CLAIM}: "
                  + " | ".join(
                      f"seed {s}: min g-12 {fmt(wall[f'W1_{s}']['min_gm12'])} "
                      f"(at +{wall[f'W1_{s}']['argmin_step']}), +300 "
                      f"{fmt(wall[f'W1_{s}']['g_m12'].get(300))}"
                      for s in NEW_SEEDS)
                  + f" — with the reference seed {G1.FREEZE_SEED} (min "
                  f"{fmt(REF['W1_min'])}, +300 "
                  f"{fmt(REF['W1_g_m12']['300'])}), the WALL-HOLDS claim is "
                  f"licensed at n=3 (three wash-draw seeds, one lineage, "
                  f"one root); each seed's control died ("
                  + " / ".join(f"+{v['earliest_le_bar']}" for v in
                               G_CTRL["per_seed"].values())
                  + f") under bit-identical per-seed inputs (G-INPUTS), "
                  f"the wall inert at forward 1 (G-STEP1) and pinning "
                  f"verified (G-PIN).")
    else:
        verdict = "WALL-SEED-BOUND"
        failed_seeds = [s for s in NEW_SEEDS
                        if not wall[f"W1_{s}"]["maintains"]]
        clause = ("honest bound — the well does NOT survive every wash-draw: "
                  + " | ".join(
                      f"seed {s}: min g-12 {fmt(wall[f'W1_{s}']['min_gm12'])} "
                      f"(at +{wall[f'W1_{s}']['argmin_step']}, first under "
                      f"0.50: "
                      f"+{next((c for c in G1.CK_WASH if wall[f'W1_{s}']['g_m12'].get(c, 1.0) < G1.MAINTAIN_BAR), None)})"
                      for s in failed_seeds)
                  + f"; maintained: "
                  + ", ".join(str(s) for s in NEW_SEEDS
                              if wall[f"W1_{s}"]["maintains"])
                  + f" — the claim re-scopes to the seed lottery at n=1-of-3 "
                  f"(reference {G1.FREEZE_SEED} maintained: {ref_maintains}).")

    log("=" * 78)
    log(f"G1BR VERDICT: {verdict}")
    for tag in arms:
        g = light_gm12(tag)
        log(f"  {tag} (seed {arms[tag]['seed']}): g-12 "
            + " -> ".join(f"+{s}:{v:.4f}" for s, v in sorted(g.items())))
    for seed in NEW_SEEDS:
        f = flat[str(seed)]
        log(f"  seed {seed}: FLAT-AT-PIN {f['FLAT_AT_PIN']} "
            f"(|d(+300,+50)| {f['flat_delta']}) | dip@+2 "
            f"{fmt(f['dip_at_2'])} | F4 {f['F4_ERODES']}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # OUTPUTS
    # ======================================================================
    trace = {}
    for tag in arms:
        rows = [{"freeze_steps": 0,
                 **{k: root_cells[k] for k in
                    ("gm12", "g0", "gp12", "held30_gm12", "held30_g0",
                     "ce_r", "site_read_onset", "site_read_span")}}]
        g0_light = {t["step"]: t["g0_mean_pz"] for t in arms[tag]["traj"]
                    if "g0_mean_pz" in t}
        for s in sorted(arms[tag]["sds"]):
            c = None
            if str(s) in batteries_all.get(tag, {}):
                c = flat_cells(batteries_all[tag][str(s)])
            lg = light_gm12(tag).get(s)
            if c is not None:
                rows.append({"freeze_steps": s, **c})
            elif lg is not None:
                rows.append({"freeze_steps": s, "gm12": lg,
                             "g0": g0_light.get(s),
                             "ce_r": next((t["ce_r"] for t in arms[tag]["traj"]
                                           if t["step"] == s), None)})
        trace[tag] = rows

    metrics = {
        "experiment": "g1bR_seed_replicates",
        "date": common.now_iso(),
        "design": ("g1b's WALL-HOLDS claim at the lab's n>=3 rule: the "
                   "identical cell re-run at two fresh wash-draw seeds "
                   f"({NEW_SEEDS[0]}, {NEW_SEEDS[1]}), C + W1(R={R_CLAIM}) "
                   "per seed, machinery lab/g1b_continuity.py VERBATIM via "
                   "import (R55 owed-debt #1; g1 spec section 8 priority 1)"),
        "registered_prediction": G1BR_PREDICTION,
        "question": ("does the anchored ball's WALL-HOLDS result replicate "
                     "across wash-draw seeds — one commit event + hard L2 "
                     f"projection at R={R_CLAIM} holding the consolidated "
                     "fact (g-12 >= 0.50 at every checkpoint through +300) "
                     "under the same-draw neutral wash that kills the "
                     "uncommitted control?"),
        "reference": {"source": REF["source"], "seed": REF["seed"],
                      "verdict": REF["verdict"],
                      "W1_min": REF["W1_min"],
                      "W1_g300": REF["W1_g_m12"].get("300"),
                      "W1_g_m12": REF["W1_g_m12"],
                      "C_g_m12": REF["C_g_m12"],
                      "flat_delta": REF["W1_flat_delta"],
                      "device": REF["device"]},
        "arms": {
            tag: {"desc": desc, "R": R, "wash_seed": seed,
                  "target_mode": "true", "ckpt_steps": list(G1.CK_WASH),
                  "steps_ran": arms[tag]["steps_ran"],
                  "device": arms[tag]["device"],
                  "traj": [{k: v for k, v in t.items() if k != "d_proj"}
                           for t in arms[tag]["traj"]],
                  "missing_checkpoints": [s for s in G1.CK_WASH
                                          if s not in arms[tag]["sds"]]}
            for (tag, R, seed, desc) in ARM_SPECS},
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": G1.PRE, "post_cap":
                     G1.POST_CAP, "neutral_bank": G_ANCHOR,
                     "measure_dial": "e176n's measure() (the e131 dial set) "
                                     "on evl_load (settle+disarm PIVOT)"},
        "displacement": {
            "currency": (f"cumulative ||theta_t - theta_0||_2 over all "
                         f"{GB.G1B_PARAMS} trainable parameters (fp32, "
                         "measured per step) + the projected displacement "
                         "min(d, R)"),
            "table": disp_table,
            "theta0_norm": {t: arms[t]["theta0_norm"] for t in arms},
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR,
                  "G_BITEXACT": G_BITEXACT, "G_ROOT": G_ROOT0,
                  "G_CTRL": G_CTRL, "G_PIN": G_PIN, "G_INPUTS": G_INPUTS,
                  "G_STEP1": G_STEP1, "G_BITROOT": G_BITROOT,
                  "G_SURG": gates_surg},
        "traces": trace,
        "batteries": batteries_all,
        "adjudication": {
            "bars": G1BR_PREDICTION["bars"],
            "gates_pass": gates_pass,
            "wall": wall, "flat": flat,
            "seed_maintains": seed_maintains,
            "n_total": 1 + len(NEW_SEEDS),
            "WALL_REPLICATES": bool(gates_pass and all_maintain),
            "verdict": verdict, "clause": clause,
        },
        "honesty_reflex": {
            "intervention_not_logits": ("the wall IS the intervention: each "
                "seed's C and W1 share bit-identical step-0 weights "
                "(G-BITROOT max|diff| = 0.0), bit-identical per-step input "
                "streams through +300 (md5-gated, G-INPUTS) and identical "
                "targets; the only delta is the commit event — and G-STEP1 "
                "shows the commit is inert at forward 1, so the divergence "
                "begins exactly at the first projection. G-PIN verifies the "
                "geometry before any behavioral clause is read."),
            "device": ("HONEST MIX, fully recorded in device_events: every "
                "arm STARTED on GPU and migrated to CPU mid-run (steps "
                "175-275) on the registered park-on-thermal policy (81-83C "
                "mid-run polls) — the reference g1b run was cuda throughout. "
                "This does not touch the claim: both kill clocks (C dead at "
                "+4/+2) completed entirely on GPU; the W1 trajectories "
                "crossed the migration boundary, but every bar adjudicates "
                "with margins (~0.90 vs the 0.50 maintain bar; C ~0.002 vs "
                "the 0.27 death bar) that dwarf cross-device float fuzz "
                "(G-STEP1 measured the same-stream reproduction at "
                "~6e-6 max|diff|); all readings are CPU evals of CPU state "
                "dicts, and the bars are absolute, not differential, so the "
                "n=3 overlay spanning a device mix adjudicates cleanly. The "
                "stored e176N/e185 CPU priors remain co-reported context "
                "only."),
            "r_dial_sensitivity": ("the claim is R=0.7 SPECIFICALLY: g1b's "
                "W2 (R=1.4) dipped to 0.320 at +4 — below the maintain bar "
                "— and W3 (R=4.2) died; survival orders with R, so the "
                "replicate licenses 'the well at 0.7', not 'any wall'. The "
                "R-dial's sensitivity is itself g1b's cliff measurement."),
            "single_seed_per_cell": ("one trajectory per arm per seed (the "
                "arc's honesty convention); what this run adds is SEED "
                "replication of the wash draws (n=3 with g1b), not "
                "within-cell error bars — the root remains single (the "
                "root-draw question is g2e's, not this cell's)."),
            "within_run_scope": ("W2/W3 and the noise arms are NOT re-run: "
                "the replicated claim is WALL-HOLDS only; the cliff "
                "localization (W2's dip-and-recover) and the noise split "
                "(F3) remain n=1 textures from g1b, honestly scoped."),
        },
        "trims": G1.trims, "deviations": deviations,
        "device_events": G1.device_events,
        "device_policy": {"parked": G1.GPU_PARKED, "reason": G1.PARK_REASON},
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": GB.G1B_PARAMS,
                   "R_claim": R_CLAIM, "new_seeds": list(NEW_SEEDS),
                   "reference_seed": G1.FREEZE_SEED, "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    c_dead = {str(s): G_CTRL["per_seed"][str(s)]["earliest_le_bar"]
              for s in NEW_SEEDS}
    plot(rd / "seed_replicates.png", trace, disp_table, wall, flat, verdict,
         clause, gates_pass, REF, root_cells, c_dead)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'seed_replicates.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/g1bR_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot
# THE n=3 OVERLAY: the claim's statistic (W1 vs C across three wash-draw
# seeds) — reference trajectories read from runs/g1b/metrics.json.

def plot(path, trace, disp_table, wall, flat, verdict, clause, gates_pass,
         REF, root_cells, c_dead):
    import textwrap
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    seed_colors = {10902: "seagreen", NEW_SEEDS[0]: "royalblue",
                   NEW_SEEDS[1]: "darkorange"}
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    f4 = lambda v: "n/a" if v is None else f"{v:.4f}"

    def series(rows):
        pts = [(r["freeze_steps"], r["gm12"]) for r in rows]
        return [p[0] for p in pts], [p[1] for p in pts]

    # (0,0) THE HEADLINE OVERLAY: W1 across 3 seeds + the controls
    ax = axes[0, 0]
    xs, ys = series(REF["traces"]["W1"])
    ax.plot(xs, ys, "s-", ms=7, lw=2.4, color=seed_colors[10902],
            alpha=0.95, label=f"W1 R=0.7 — seed 10902 (g1b reference)")
    for seed in NEW_SEEDS:
        xs, ys = series(trace[f"W1_{seed}"])
        ax.plot(xs, ys, "s-", ms=7, lw=2.2, color=seed_colors[seed],
                alpha=0.95, label=f"W1 R=0.7 — seed {seed} (new)")
    xs, ys = series(REF["traces"]["C"])
    ax.plot(xs, ys, "o--", ms=5, lw=1.4, color="crimson", alpha=0.75,
            label="C no wall — seed 10902 (ref)")
    for seed in NEW_SEEDS:
        xs, ys = series(trace[f"C{seed}"])
        ax.plot(xs, ys, "o--", ms=5, lw=1.4, color=seed_colors[seed],
                alpha=0.55, label=f"C no wall — seed {seed}")
    for yv, col in ((G1.MAINTAIN_BAR, "seagreen"), (G1.SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8)
    ax.axhline(G1.EXPRESS_BAR, ls=":", lw=1.0, color="gray", alpha=0.7)
    ax.annotate(f"root {root_cells['gm12']:.3f}", (0, root_cells["gm12"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("neutral-wash steps from the committed root")
    ax.set_ylabel("g-12 (absolute mean p(Z), install-60 battery)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, loc="center right")
    ax.set_title("THE ANCHORED BALL at R=0.7 — the n=3 wash-draw overlay "
                 "(2.74M, one root)", fontsize=10)

    # (0,1) THE CLAIM'S STATISTIC: min g-12 through +300, per seed
    ax = axes[0, 1]
    seeds = [10902] + list(NEW_SEEDS)
    w1_mins = [REF["W1_min"]] + [wall[f"W1_{s}"]["min_gm12"] for s in NEW_SEEDS]
    c_mins = [REF["C_min"]] + [wall[f"C{s}"]["min_gm12"] for s in NEW_SEEDS]
    xpos = np.arange(len(seeds))
    ax.bar(xpos - 0.18, w1_mins, width=0.36, color=[seed_colors[s]
                                                    for s in seeds],
           alpha=0.9, label="W1 commit(0.7)")
    ax.bar(xpos + 0.18, c_mins, width=0.36, color="crimson", alpha=0.55,
           label="C no wall")
    ax.axhline(G1.MAINTAIN_BAR, ls="--", lw=1.4, color="seagreen",
               alpha=0.9)
    ax.axhline(G1.SHUT_BAR, ls="--", lw=1.1, color="tab:purple", alpha=0.8)
    for x, v, g3 in zip(xpos, w1_mins,
                        [REF["W1_g_m12"].get("300")]
                        + [wall[f"W1_{s}"]["g_m12"].get(300)
                           for s in NEW_SEEDS]):
        ax.annotate(f"min {f3(v)}\n+300 {f3(g3)}", (x - 0.18, v),
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"seed {s}\n{'(g1b ref)' if s == 10902 else '(new)'}"
                        for s in seeds])
    ax.set_ylabel("min g-12 over all checkpoints {1..300}")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("THE CLAIM'S STATISTIC: min g-12 through +300 (bar 0.50)",
                 fontsize=10)

    # (1,0) DISPLACEMENT: the new W1 arms vs the wall (reference co-plotted)
    ax = axes[1, 0]
    ref_rows = [r for r in REF["disp_table"]["W1"] if "g_m12_light" in r]
    ax.plot([r["step"] for r in ref_rows], [r["cum_disp"] for r in ref_rows],
            "s--", ms=5, lw=1.4, color=seed_colors[10902], alpha=0.8,
            label="W1 seed 10902 (ref)")
    for seed in NEW_SEEDS:
        rows = [r for r in disp_table[f"W1_{seed}"] if "g_m12_light" in r]
        ax.plot([r["step"] for r in rows], [r["cum_disp"] for r in rows],
                "s-", ms=6, lw=1.8, color=seed_colors[seed], alpha=0.9,
                label=f"W1 seed {seed}")
    ax.axhline(R_CLAIM, ls="--", lw=1.2, color="k", alpha=0.7)
    ax.axhline(R_CLAIM + G1.PIN_FUZZ_BAR, ls=":", lw=1.0, color="k",
               alpha=0.5)
    ax.annotate(f"R={R_CLAIM} (+pin {R_CLAIM + G1.PIN_FUZZ_BAR})", (0.99, R_CLAIM),
                xycoords=("axes fraction", "data"), ha="right", fontsize=7.5)
    for seed in NEW_SEEDS:
        rows = [r for r in disp_table[f"C{seed}"] if "g_m12_light" in r]
        ax.plot([r["step"] for r in rows], [r["cum_disp"] for r in rows],
                "o--", ms=4, lw=1.2, color=seed_colors[seed], alpha=0.5,
                label=f"C seed {seed} (no wall)")
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"raw $\|\theta_t-\theta_0\|_2$ at checkpoints")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("DISPLACEMENT vs the wall (G-PIN) — the controls free-run",
                 fontsize=10)

    # (1,1) THE VERDICT PANEL
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "G1BR — THE WALL SEED REPLICATES (n=3 overlay)",
            fontsize=11, va="top", family="monospace", weight="bold")
    y -= 0.055
    ax.text(0.02, y, f"gates: {'ALL PASS' if gates_pass else 'FAILURE'} | "
            f"seeds {NEW_SEEDS[0]}/{NEW_SEEDS[1]} vs reference "
            f"{G1.FREEZE_SEED}", fontsize=7.5, va="top", family="monospace")
    y -= 0.032
    ax.text(0.02, y, f"  root g-12 {root_cells['gm12']:.4f} (bit-exact "
            f"e131 root; g1b's own gate)", fontsize=7.2, va="top",
            family="monospace")
    y -= 0.034
    for seed in [10902] + list(NEW_SEEDS):
        if seed == 10902:
            g = {int(k): v for k, v in REF["W1_g_m12"].items()}
            cg = {int(k): v for k, v in REF["C_g_m12"].items()}
            mn, a3 = REF["W1_min"], REF["W1_g_m12"].get("300")
            fd = REF["W1_flat_delta"]
            cd = next((s for s in sorted(cg) if cg[s] <= G1.SHUT_BAR), None)
        else:
            g = wall[f"W1_{seed}"]["g_m12"]
            mn, a3 = wall[f"W1_{seed}"]["min_gm12"], g.get(300)
            fd = flat[str(seed)]["flat_delta"]
            cd = c_dead.get(str(seed))
        seq = " -> ".join(f"+{s}:{g[s]:.4f}" for s in sorted(g))
        ax.text(0.02, y, f"  W1 seed {seed}: {seq}", fontsize=6.4, va="top",
                family="monospace", color=seed_colors[seed])
        y -= 0.028
        ax.text(0.02, y, f"      min {f4(mn)} | +300 {f4(a3)} | FLAT d "
                f"{f4(fd)} | C dead at +{cd}", fontsize=6.8, va="top",
                family="monospace", color=seed_colors[seed])
        y -= 0.033
    y -= 0.006
    ax.text(0.02, y, f"VERDICT: {verdict}", fontsize=10, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.044
    for wd in textwrap.wrap(clause, width=92, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.8, va="top", family="monospace")
        y -= 0.026

    fig.suptitle("G1BR — WALL-HOLDS at the lab's n>=3 rule: the anchored "
                 f"ball (commit R=0.7 + L2 projection) across three "
                 f"wash-draw seeds -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
