"""G2B — THE ONSET-ONLY MONITOR (g2's named fix; T122 / QUEUE row g2b).

THE DELTA (exactly one channel of the organ changes): g2's read monitor
averaged p(true name char) over the 7x8 masked name positions, so the gate
watched a mean dominated by the name's SELF-CORRELATION channel (Z->E, E->P,
... — which the wash spares at 0.65+) instead of the ctx->onset read (which
it kills). g2b's monitor reads ONLY the onset position — the FIRST masked
column of each cue window, p(Z | preceding context), 8 values instead of 56:

    g2:   float(ptrue[mask[idx]].mean())
    g2b:  onset = mask[idx] & (mask[idx].cumsum(1) == 1)
          float(ptrue[onset].mean())

The organ is otherwise VERBATIM: same cue pool (loaded from g2's own root
checkpoint, runs/checkpoints/g2_root.pt — registered buffers, write-once),
same CADENCE/K_MON/MON_OFFSETS rotation, same THETA_OPEN=0.5, same
REFRACTORY=24, same error-replay event (e174 arm B), same wash (e176n arm
A), same seed 10902. Implemented as a G2Net subclass whose monitor() is the
one changed method; g2.run_cell builds the organ through its module global,
so g2b patches G2.G2Net = G2BNet and reuses g2's ENTIRE cell machinery.

REGISTERED BARS (QUEUE row g2b, VERBATIM): "Bars: MAINTAINS (>=0.5 at +300
with the gate firing in the 5-25 band — the organ vindicated) / GATE-HYPER
(over-firing; the sawtooth rides every cycle) / STILL-SILENT (the onset
channel also decays too slowly under wash)". Adjudication frozen here:
  - MAINTAINS   : ruler@+300 >= 0.5 AND 5 <= n_events <= 25 AND the gate
                  closes between events at least once (not every
                  post-refractory check fires).
  - GATE-HYPER  : every post-refractory monitor check fires (all_fired —
                  g2's F3 texture) — the sawtooth rides every cycle;
                  whether it maintains anyway is co-reported.
  - STILL-SILENT: n_events < 5 (the gate never opens enough; if the
                  onset trace hovers >= 0.5 while the ruler dies, that is
                  the slow-decay texture named in the bar).
  Mid-cycle dips below the bars do not un-maintain (e179's convention); the
  late-grid mean/min over {100,200,300} and the +50 value are co-reported
  for phase honesty. No bar shopping; every sub-boolean reported.

REGISTERED PREDICTION (before running; from the benchmark probe at the
root, 4 CPU threads): root onset-only monitor = 0.609 (7-pos was 0.941) —
above THETA_OPEN at install end, so the gate starts CLOSED. The wash kills
the onset read fast (g2's base site-onset 0.005 by +50; its 7-pos monitor's
step-76 dip to 0.48 with self-corr channels at 0.65+ implies onset ~0.0x by
then), so the gate should open at the first or second legal check (step
24-44) and then sawtooth: 8-12 events, spacing refractory-bounded 24-40,
post-event +24 ruler in g2's single-event band (0.4-0.7). The +300 ruler
rides cycle phase — a coin flip vs the 0.5 bar; the late-grid mean is the
honest texture. DISCRIMINATING OBSERVATION (MAINTAINS vs GATE-HYPER): the
post-refractory monitor values — if the onset read recovers >= 0.5 after
most events (re-entry sticks) the gate closes and the rhythm is healthy;
if it stays < 0.5 at every first check, the gate is hyper and the sawtooth
rides every cycle.

CONSTRAINTS (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 set before the
g2 import), torch threads 4 (LOW), a stagger sleep between trainings. g2's
checkpoints and machinery are reused — the root is NOT rebuilt; cells rerun
from it because the event schedule changes with the monitor.

Outputs: runs/g2b/{metrics.json, onset_monitor.png}. No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python g2b_onset_monitor.py    (G2B_SMOKE=1 shakedown)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # CPU-ONLY (before g2 import;
                                               # g2's setdefault cannot clobber)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                # noqa: E402
import torch                                      # noqa: E402
import torch.nn.functional as F                    # noqa: E402

import g2_rehearsal_organ as G2                   # noqa: E402 — the organ
import common                                      # noqa: E402
from common import CharCorpus, run_dir, save_json  # noqa: E402

torch.set_num_threads(4)                          # LOW threads (dispatch)

import matplotlib                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import matplotlib.gridspec as gridspec            # noqa: E402

SMOKE = os.environ.get("G2B_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- frozen constants (all inherited from g2; restated for the record) --------
FREEZE_SEED = G2.FREEZE_SEED                      # 10902 — the locked lineage
RULER_GEO = 0                                      # g2_root.pt meta (frozen)
RULER_KEY = {-12: "gm12", 0: "g0", 12: "gp12"}[RULER_GEO]
SHUT_BAR, MAINTAIN_BAR = G2.SHUT_BAR, G2.MAINTAIN_BAR   # 0.27 / 0.50
ECON_MIN, ECON_MAX = G2.ECON_MIN, G2.ECON_MAX           # 5 / 25
CKPT_STEPS: tuple[int, ...] = (1, 2, 4, 10, 25, 50, 100, 200, 300) \
    if not SMOKE else (1, 2, 4, 36)
N_STEPS = CKPT_STEPS[-1]
THREADS = 4
STAGGER_S = 20.0 if not SMOKE else 2.0            # the dispatch's stagger
G2B_CAP_S = 600.0                                 # see deviations (CPU-only)
G2_METRICS = G2.E43.REPO / "runs" / "g2" / "metrics.json"

REGISTERED_BARS = {
    "maintains": "MAINTAINS (>=0.5 at +300 with the gate firing in the 5-25 "
                 "band — the organ vindicated)",
    "gate_hyper": "GATE-HYPER (over-firing; the sawtooth rides every cycle)",
    "still_silent": "STILL-SILENT (the onset channel also decays too slowly "
                    "under wash)",
    "source": "QUEUE.md row g2b, VERBATIM; operationalized in this file's "
              "docstring (frozen before the run; no bar shopping).",
}

deviations: list[str] = [
    "THE DELTA: monitor channel only — g2 averaged p(true name char) over "
    "the 7x8 masked positions; g2b reads the FIRST masked column per window "
    "(mask & cumsum==1; p(Z|ctx), 8 values). Everything else verbatim, via "
    "the G2Net subclass + module-global patch (g2.run_cell unchanged).",
    "ROOT REUSE (the honest-reflex point): the root is g2's own "
    "g2_root.pt (cuda install + cpu consolidation, ruler g0 0.7106) — loaded, "
    "not rebuilt; the cue pool on disk is bit-verified against the freshly "
    "rebuilt e113 pool (G_POOL), and the root dials are re-measured and "
    "checked against g2's stored cells (G_ROOT2).",
    "CELL-BASE rerun on CPU: g2's base cell ran on CUDA; g2b reruns it "
    "CPU-only (same seed -> same RNG batch stream) so the contrast gate is "
    "device-consistent within g2b; g2's stored CUDA base trace is embedded "
    "as the shape-level cross-check.",
    "TRAIN CAP: G2B_CAP_S=600 s (g2's own cap was 1800 s; the lab's 180-s "
    "single-run rule is GPU-era — at the mandated 4 CPU threads a 300-step "
    "cell measures ~217 s train-only, so the cap cannot hold; recorded, "
    "bounded at 600 s).",
    "DIAL TRIM: the +300 anatomy dial keeps geos/held30/site-read/"
    "deletions/A129-quick and DROPS the row census (anatomy bars are not in "
    "g2b's registered bars; the census is the slow CPU piece).",
    "Stagger: 20 s sleep between trainings (dispatch's CPU stagger), "
    "cooldowns skipped (CPU-only; g2's 60 s thermal cooldowns are GPU-era).",
    "Single seed (10902), single lineage, n=1 per cell — point estimates "
    "until replicated.",
    "Smoke mode trims: 36-step cells, checkpoints {1,2,4,36}, no dials, "
    "2 s stagger — nothing adjudicated.",
]


# ------------------------------------------------------------------ the delta
G2_ORIG_NET = G2.G2Net        # kept for the 7-pos root co-report


class G2BNet(G2.G2Net):
    """g2's organ with the ONSET-ONLY monitor (the named fix). The one
    changed line is flagged; every other arithmetic step is g2's monitor
    VERBATIM (same window rotation, same softmax read)."""

    @torch.no_grad()
    def monitor(self, c: int, dev=None) -> float:
        dev = dev or next(self.body.parameters()).device
        pool = self.cue_pool.to(dev).long()
        mask = self.cue_mask.to(dev)
        blocks = {j: G2.JITTERS.index(j) for j in G2.MON_OFFSETS}
        idx = []
        for k in range(G2.K_MON):
            b = blocks[G2.MON_OFFSETS[k % len(G2.MON_OFFSETS)]] * 60
            idx.append(b + ((c * G2.K_MON + k) % 60))
        idx = torch.tensor(idx, device=dev)
        w = pool[idx]
        x, y = w[:, :-1], w[:, 1:]
        logits, _ = self.body(x)
        pr = F.softmax(logits, -1)
        ptrue = pr.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        m = mask[idx]
        onset = m & (m.cumsum(1) == 1)     # THE FIX: 1 of the 7 name
                                           # positions — p(Z | ctx), before
                                           # the name's self-correlation
        return float(ptrue[onset].mean())


G2.G2Net = G2BNet                 # g2.run_cell builds the organ via this
                                  # module global — the whole cell machinery
                                  # (event branch, G_REPLAY checks, post-event
                                  # bookkeeping) is reused unchanged.


# ------------------------------------------------------------------ protocol
# Rebuild of the measurement protocol (g2.main's construction arithmetic
# VERBATIM, trimmed to what g2b needs): corpus, splice, jitter pools,
# batteries, val windows, e170's neutral anchor bank.

def rebuild_protocol() -> dict:
    corpus = CharCorpus(G2.E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH")
    assert corpus_zeph == 0, f"corpus contains ZEPH x{corpus_zeph}"

    host_occ = []
    for host in G2.HOSTS:
        for p in G2.E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G2.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(G2.E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    name_ids = corpus.encode(G2.NAME)
    L = len(G2.NAME)

    def offset_pool(j: int):
        wins = []
        for p, h in install_occ:
            pre = train_ids[p - G2.PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + G2.POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            assert len(w) == G2.BLOCK
            wins.append(w)
        px = torch.stack(wins)
        pm = torch.zeros(len(wins), G2.BLOCK - 1, dtype=torch.bool)
        pm[:, G2.PRE - 1 + j: G2.PRE - 1 + j + L] = True
        return px, pm

    jit_pools = {j: offset_pool(j) for j in G2.JITTERS}
    jit_pool_x = torch.cat([jit_pools[j][0] for j in G2.JITTERS])
    jit_pool_mask = torch.cat([jit_pools[j][1] for j in G2.JITTERS])
    pool_183_x, _ = offset_pool(G2.RETEACH_J)

    bat_ids, held_bat = {}, {}
    for j in G2.GEOS:
        cs = [train_text[p - G2.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G2.PRE - j: p] for p, _ in held_occ]
        held_bat[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = G2.val_windows(val_ids, val_text, 60, G2.R_EVAL_SEED)

    arng = _random.Random(G2.E170_ANCHOR_SEED)
    hi_start = len(train_ids) - G2.BLOCK - 2
    n_starts = []
    while len(n_starts) < 16:
        s = arng.randrange(hi_start)
        if any(f in train_text[s: s + G2.BLOCK + 1]
               for f in G2.ANCHOR_FORBIDDEN):
            continue
        n_starts.append(s)
    anchor_neutral = torch.stack([train_ids[s: s + G2.BLOCK]
                                   for s in n_starts])
    return {"corpus": corpus, "itos": itos, "zid": zid,
            "train_ids": train_ids, "install_occ": install_occ,
            "jit_pool_x": jit_pool_x, "jit_pool_mask": jit_pool_mask,
            "pool_183_x": pool_183_x, "bat_ids": bat_ids,
            "held_bat": held_bat, "r_eval_xy": (r_eval_x, r_eval_y),
            "anchor_neutral": anchor_neutral, "n_starts": n_starts}


# ------------------------------------------------------------------ dials
# g2.main's nested measure()/flat_cells(), copied to module scope and
# trimmed (census dropped — see deviations). Instruments themselves are
# g2's module-level functions, reused.

def measure(sd: dict, tag: str, P: dict) -> dict:
    net = G2.evl_load(sd)
    zid = P["zid"]
    out: dict = {"tag": tag}
    out["base"] = {j: G2.battery_cell(net, P["bat_ids"][j], zid)
                   for j in G2.GEOS}
    out["base_held"] = {j: G2.battery_cell(net, P["held_bat"][j], zid)
                        for j in G2.GEOS}
    out["ce_r"] = G2.ce_fixed_cpu(net, *P["r_eval_xy"])
    out["site_read"] = G2.read_fact_at(net, P["pool_183_x"],
                                       P["corpus"].encode(G2.NAME), zid,
                                       G2.SITE_ADDR_ROW, G2.SITE_Z_XCOL)
    log(f"[{tag}] base: " + " ".join(
        f"g{j:+d} {out['base'][j]['mean_pz']:.4f}" for j in G2.GEOS)
        + " | held30 g0 " + f"{out['base_held'][0]['mean_pz']:.4f}"
        + f" | CE_R {out['ce_r']:.4f} | onset "
        f"{out['site_read']['pz_onset_mean']:.4f}")
    DELS = {"d_all": G2.D_ALL, "d183": (G2.SITE_ADDR_ROW,)}
    out["del_table"] = {}
    for dl, rows_ in DELS.items():
        sd_d, gate = G2.deleted_wpe(sd, rows_)
        assert gate["pass"], f"deletion gate FAILED {tag}/{dl}"
        net.load_state_dict(sd_d)
        cell = {"g0": G2.battery_cell(net, P["bat_ids"][0], zid)["mean_pz"]}
        if dl == "d183":
            cell["gm12"] = G2.battery_cell(net, P["bat_ids"][-12],
                                           zid)["mean_pz"]
        out["del_table"][dl] = cell
        net.load_state_dict(sd)
    w = net.wpe.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    bp = G2.battery_pz(net, P["bat_ids"][0], zid)
    w[129] = mean_row
    m129 = bp - G2.battery_pz(net, P["bat_ids"][0], zid)
    w.copy_(orig)
    w[129] = 0.0
    z129 = bp - G2.battery_pz(net, P["bat_ids"][0], zid)
    w.copy_(orig)
    assert torch.equal(w, orig), "A129 quick failed to restore wpe"
    out["A129_quick"] = float(min(m129, z129))
    del net
    return out


def flat_cells(m: dict) -> dict:
    return {"gm12": m["base"][-12]["mean_pz"],
            "g0": m["base"][0]["mean_pz"],
            "gp12": m["base"][12]["mean_pz"],
            "held30_gm12": m["base_held"][-12]["mean_pz"],
            "held30_g0": m["base_held"][0]["mean_pz"],
            "ce_r": m["ce_r"],
            "site_read_onset": m["site_read"]["pz_onset_mean"],
            "site_read_span": m["site_read"]["pname_mean_over7"],
            "A129": m["A129_quick"],
            "dall_g0": m["del_table"]["d_all"]["g0"],
            "d183_g0": m["del_table"]["d183"]["g0"],
            "d183_gm12": m["del_table"]["d183"]["gm12"]}


# ------------------------------------------------------------------ checkpoints
CKPT_INVENTORY: dict = {}


def save_g2b_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = G2.CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g2b", **meta}}, path)
    CKPT_INVENTORY[name] = {
        "path": str(path.relative_to(G2.E43.REPO)).replace("\\", "/"),
        **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g2b_smoke" if SMOKE else "g2b")
    common.DEVICE = "cpu"
    G2.TRAIN_CAP_S = G2B_CAP_S          # read by G2.run_cell at call time
    log(f"G2B THE ONSET-ONLY MONITOR (g2's named fix; CPU-only, threads "
        f"{torch.get_num_threads()}, cuda avail "
        f"{torch.cuda.is_available()}) -> {rd}")

    # ---- g2's stored references (never delete runs/) ----------------------
    g2m = json.loads(G2_METRICS.read_text(encoding="utf-8"))
    g2_root_cells = g2m["root"]["cells"]
    g2_anchor_starts = g2m["gates"]["G_ANCHOR"]["starts"]
    g2_base_traj = {r["step"]: r["g0"] for r in g2m["cells"]["base"]["traj"]}

    # ---- protocol rebuild + the root checkpoint (REUSE) -------------------
    P = rebuild_protocol()
    log(f"protocol rebuilt (install60/held30 splice, pools, batteries, "
        f"neutral bank 16) — corpus ZEPH 0, bank starts match g2: "
        f"{P['n_starts'] == g2_anchor_starts}")

    ck = torch.load(G2.CKPT_DIR / ("smoke_g2_root.pt" if SMOKE
                                   else "g2_root.pt"),
                    map_location="cpu", weights_only=False)
    ck_sd = ck["model"]
    root_sd = G2.organ_body_sd(ck_sd)
    cue_pool16, cue_mask = ck_sd["cue_pool"], ck_sd["cue_mask"]
    cue_pool = cue_pool16.long()
    root_meta = ck["meta"]

    # G_POOL: the registered cue pool on disk == the freshly rebuilt e113 pool
    G_POOL = {"form": "bit",
              "pool_bit_identical": bool(torch.equal(cue_pool,
                                                     P["jit_pool_x"])),
              "mask_bit_identical": bool(torch.equal(cue_mask,
                                                     P["jit_pool_mask"])),
              "shape": list(cue_pool.shape)}
    G_POOL["pass"] = bool(G_POOL["pool_bit_identical"]
                          and G_POOL["mask_bit_identical"])
    assert G_POOL["pass"], f"G_POOL FAILED: {G_POOL}"
    log("G_POOL: checkpoint cue pool bit-identical to the rebuilt e113 "
        "jitter pool: PASS")

    # G_ROOT2: the reused root reproduces g2's stored CPU dials
    re_cells = {j: G2.battery_cell(G2.evl_load(root_sd), P["bat_ids"][j],
                                   P["zid"])["mean_pz"] for j in G2.GEOS}
    re_ce = G2.ce_fixed_cpu(G2.evl_load(root_sd), *P["r_eval_xy"])
    diffs = {"gm12": re_cells[-12] - g2_root_cells["gm12"],
             "g0": re_cells[0] - g2_root_cells["g0"],
             "gp12": re_cells[12] - g2_root_cells["gp12"],
             "ce_r": re_ce - g2_root_cells["ce_r"]}
    G_ROOT2 = {"form": "reuse-verify (threads 4 vs g2's 8)",
               "cells_reloaded": {f"g{j:+d}": re_cells[j] for j in G2.GEOS},
               "ce_r_reloaded": re_ce,
               "g2_stored": {k: g2_root_cells[k]
                             for k in ("gm12", "g0", "gp12", "ce_r")},
               "diffs": diffs,
               "max_abs_diff": max(abs(v) for v in diffs.values()),
               "bit_tol": G2.G_BIT_TOL, "tol": G2.G_FALLBACK_TOL}
    G_ROOT2["bit"] = bool(G_ROOT2["max_abs_diff"] < G2.G_BIT_TOL)
    G_ROOT2["pass"] = bool(G_ROOT2["max_abs_diff"] < G2.G_FALLBACK_TOL)
    log(f"G_ROOT2 (root reuse vs g2's stored dials): max|diff| "
        f"{G_ROOT2['max_abs_diff']:.2e}: "
        + ("PASS" if G_ROOT2["pass"] else "FAIL")
        + (" (bit)" if G_ROOT2["bit"] else ""))
    if not SMOKE:
        assert G_ROOT2["pass"], "G_ROOT2 FAILED — root checkpoint drifted"

    # the root's two monitor channels (the delta, measured)
    root_mon_onset = G2BNet(G2.evl_load(root_sd), cue_pool, cue_mask)\
        .monitor(0, dev=CPU)
    root_mon_7pos = G2_ORIG_NET(G2.evl_load(root_sd), cue_pool, cue_mask)\
        .monitor(0, dev=CPU)
    log(f"root monitors: onset-only {root_mon_onset:.4f} vs 7-pos "
        f"{root_mon_7pos:.4f} (g2 stored {g2m['organ']['root_monitor']:.4f}); "
        f"THETA_OPEN {G2.THETA_OPEN} — gate starts "
        f"{'CLOSED' if root_mon_onset >= G2.THETA_OPEN else 'OPEN'}")

    # ---- the two cells (base = device-consistent contrast; g2 = the fix) ---
    cells_out: dict = {}
    for mode in ("base", "g2"):
        log("=" * 78)
        log(f"stagger {STAGGER_S:.0f}s (CPU-only dispatch)")
        time.sleep(STAGGER_S)
        desc = {"base": "gate hard-disabled — the CPU contrast identity",
                "g2": "gate live, ONSET-ONLY monitor, all three locks on "
                      "(THE FIX)"}[mode]
        log(f"CELL {mode.upper()} (g2b): {desc} — {N_STEPS} steps, seed "
            f"{FREEZE_SEED}, checkpoints +{list(CKPT_STEPS)}")
        cells_out[mode] = G2.run_cell(
            mode, root_sd, cue_pool, cue_mask, P["anchor_neutral"],
            P["train_ids"], P["itos"], P["r_eval_xy"], P["bat_ids"],
            P["zid"], FREEZE_SEED, CKPT_STEPS)
        save_g2b_ckpt(
            f"g2b_{mode}", cells_out[mode]["final_sd"],
            {"desc": f"g2 root (reused) + {N_STEPS}-step g2b cell ({desc}), "
                     f"onset-only monitor, seed {FREEZE_SEED}, "
                     f"{cells_out[mode]['n_events']} events",
             "mode": mode, "steps": int(cells_out[mode]["steps_ran"]),
             "n_events": int(cells_out[mode]["n_events"]),
             "realized_r": cells_out[mode]["realized_r"],
             "seed": FREEZE_SEED, "monitor": "onset-only (1 of 7)"})
        if mode == "g2" and 50 in cells_out[mode]["sds"]:
            save_g2b_ckpt("g2b_g2_s50", cells_out[mode]["sds"][50],
                          {"desc": "g2b g2 cell body at +50 (phase analysis)",
                           "mode": mode, "steps": 50})

    # gates
    G_NAMEFREE = {"corpus_zeph_count": 0,
                  "cell_zeph_violations": {m: cells_out[m]["zeph_violations"]
                                           for m in cells_out},
                  "pass": bool(all(cells_out[m]["zeph_violations"] == 0
                                   for m in cells_out))}
    assert G_NAMEFREE["pass"], "name token leaked into a wash/replay window"
    G_REPLAY = {m: {"n_events": cells_out[m]["n_events"],
                    "checks": cells_out[m]["replay_checks"],
                    "pass": bool(
                        cells_out[m]["n_events"] == 0 or
                        (cells_out[m]["replay_checks"]["mask7_ok"]
                         == cells_out[m]["n_events"]
                         and cells_out[m]["replay_checks"]["anchors_8_8_ok"]
                         == cells_out[m]["n_events"]))}
                for m in cells_out}
    G_STEP = {m: {"steps_ran": cells_out[m]["steps_ran"],
                  "expected": N_STEPS, "batch": 32,
                  "pass": bool(cells_out[m]["steps_ran"] == N_STEPS)}
              for m in cells_out}
    # device-consistency: g2b's CPU base must track g2's stored CUDA base
    b_cpu = {r["step"]: r[RULER_KEY] for r in cells_out["base"]["traj"]}
    common_steps = sorted(set(b_cpu) & set(g2_base_traj))
    G_DEV = {"form": "shape (CPU vs g2's CUDA base; same RNG stream)",
             "max_abs_diff": max((abs(b_cpu[s] - g2_base_traj[s])
                                  for s in common_steps), default=0.0),
             "both_dead_at_50": bool(b_cpu.get(50, 1.0) <= SHUT_BAR
                                     and g2_base_traj.get(50, 1.0)
                                     <= SHUT_BAR)}
    log(f"G_NAMEFREE: {'PASS' if G_NAMEFREE['pass'] else 'FAIL'}")
    log("G_REPLAY: " + " ".join(f"{m}={G_REPLAY[m]['pass']}" for m in G_REPLAY)
        + f" -> {'PASS' if all(v['pass'] for v in G_REPLAY.values()) else 'FAIL'}")
    log("G_STEP_PARITY: " + " ".join(f"{m}={G_STEP[m]['steps_ran']}"
                                     for m in G_STEP)
        + f" -> {'PASS' if all(v['pass'] for v in G_STEP.values()) else 'FAIL'}")
    hard = {"G_POOL": G_POOL["pass"], "G_ROOT2": G_ROOT2["pass"],
            "G_NAMEFREE": G_NAMEFREE["pass"],
            "G_REPLAY": all(v["pass"] for v in G_REPLAY.values()),
            "G_STEP": all(v["pass"] for v in G_STEP.values())}
    bad = [k for k, v in hard.items() if not v]
    if bad and not SMOKE:
        raise RuntimeError(f"gate(s) FAILED: {bad}")

    # ---- +300 anatomy dial for the g2 cell (trimmed; see deviations) -------
    dials300 = None
    if not SMOKE and 300 in cells_out["g2"]["sds"]:
        dials300 = measure(cells_out["g2"]["sds"][300], "g2b300", P)

    # =====================================================================
    # ADJUDICATION (registered bars; no shopping)
    # =====================================================================
    ruler_at = {m: {r["step"]: r[RULER_KEY] for r in cells_out[m]["traj"]}
                for m in cells_out}
    g2c = cells_out["g2"]
    trace = g2c["monitor_trace"]
    n_ev = g2c["n_events"]
    spac = g2c["event_spacings"]
    late_steps = [s for s in (100, 200, 300) if s in ruler_at["g2"]]
    late_vals = [ruler_at["g2"][s] for s in late_steps]
    late_mean = float(np.mean(late_vals)) if late_vals else float("nan")

    contrast = {"base_ruler_50": ruler_at["base"].get(50), "bar": SHUT_BAR,
                "g2_stored_cuda_base_50": g2_base_traj.get(50),
                "pass": bool(ruler_at["base"].get(50, 1.0) <= SHUT_BAR)}
    in_band = bool(ECON_MIN <= n_ev <= ECON_MAX)
    all_fired = bool(len(trace) > 0 and all(t["fired"] for t in trace))
    gate_closed_between = bool(len(trace) > 0
                               and any(not t["fired"] for t in trace))
    maintain = bool(ruler_at["g2"].get(300, 0.0) >= MAINTAIN_BAR)
    stickiness = [e["ruler_cells_at_plus_refractory"][RULER_KEY]
                  for e in g2c["event_log"]
                  if e.get("ruler_cells_at_plus_refractory") is not None]
    frac_stuck = float(np.mean([s >= MAINTAIN_BAR for s in stickiness])) \
        if stickiness else None
    onset_trace_vals = [t["monitor"] for t in trace]
    post_first = [e.get("post_refractory_monitor") for e in g2c["event_log"]]
    frac_post_recovered = float(np.mean([v is not None and v >= G2.THETA_OPEN
                                         for v in post_first])) if post_first \
        else None

    bars = {
        "MAINTAINS": bool(contrast["pass"] and maintain and in_band
                          and gate_closed_between),
        "GATE_HYPER": bool(contrast["pass"] and all_fired),
        "STILL_SILENT": bool(n_ev < ECON_MIN),
        "maintain_ruler_300": ruler_at["g2"].get(300),
        "maintain_ruler_50_coreport": ruler_at["g2"].get(50),
        "late_grid_mean_100_200_300": late_mean,
        "late_grid_min": float(min(late_vals)) if late_vals else None,
        "n_events": n_ev, "in_band_5_25": in_band,
        "all_checks_fired": all_fired,
        "gate_closed_between_events": gate_closed_between,
        "event_spacings": spac,
        "spacing_min_median_max": [float(min(spac)), float(np.median(spac)),
                                   float(max(spac))] if spac else None,
        "frac_spacing_in_20_45": float(np.mean(
            [G2.SPACING_BAND[0] <= s <= G2.SPACING_BAND[1]
             for s in spac])) if spac else None,
        "frac_events_ruler_ge_0.5_at_plus24": frac_stuck,
        "frac_events_post_refractory_monitor_ge_theta": frac_post_recovered,
        "realized_r": g2c["realized_r"],
        "contrast_gate": contrast,
    }

    if not contrast["pass"]:
        verdict = "NO-CONTRAST (base did not die by +50 — record and stop)"
        clause = (f"CELL-BASE ruler at +50 was {contrast['base_ruler_50']} "
                  f"(bar {SHUT_BAR}) — the wash law did not reproduce on "
                  f"CPU; no adjudication.")
    elif bars["GATE_HYPER"]:
        verdict = "GATE-HYPER"
        clause = (f"the onset-only gate OVER-FIRES: every post-refractory "
                  f"check fired ({n_ev} events in {N_STEPS} steps, spacings "
                  f"{bars['spacing_min_median_max']}) — the sawtooth rides "
                  f"every cycle; it {'MAINTAINS anyway ' if maintain else 'does NOT maintain '}"
                  f"(ruler@+300 {ruler_at['g2'].get(300)}, late-grid mean "
                  f"{late_mean:.3f}). The sensor now sees the death but "
                  f"never sees the re-entry: theta={G2.THETA_OPEN} sits "
                  f"above the post-event onset read. The engine is "
                  f"vindicated further if maintain holds; the fix's fix is "
                  f"a hysteresis or a lower theta, not a new channel.")
    elif bars["STILL_SILENT"]:
        hover = [v for v in onset_trace_vals if v >= G2.THETA_OPEN]
        clause = (f"the gate stayed near-silent ({n_ev} events < {ECON_MIN}): "
                  f"the onset channel also decays too slowly under wash "
                  f"({len(hover)}/{len(onset_trace_vals)} checks at or above "
                  f"theta {G2.THETA_OPEN} while the ruler died to "
                  f"{ruler_at['g2'].get(300)}) — the registered STILL-SILENT "
                  f"bar. The sensor gap is NOT the self-correlation channel "
                  f"alone; the ctx->onset read itself is wash-spared at this "
                  f"cadence.")
        verdict = "STILL-SILENT"
    elif bars["MAINTAINS"]:
        verdict = "MAINTAINED (organ vindicated)"
        clause = (f"with the monitor on the onset channel (1 of 7), the gate "
                  f"fired {n_ev} times (in the 5-25 band; spacings "
                  f"{bars['spacing_min_median_max']}, "
                  f"{bars['frac_spacing_in_20_45']:.0%} in 20-45) and g2b "
                  f"MAINTAINED: ruler {ruler_at['g2'].get(300)} at +300 "
                  f"(bar {MAINTAIN_BAR}; +50 co-report "
                  f"{ruler_at['g2'].get(50)}, late-grid mean "
                  f"{late_mean:.3f}); "
                  f"{frac_stuck:.0%} of events had ruler >= 0.5 at +24. THE "
                  f"ORGAN IS VINDICATED: the resurrection economy is "
                  f"architecturally sufficient once the organ watches its "
                  f"own ctx->onset read — T122's sensor fix was the missing "
                  f"piece.")
    else:
        verdict = "MAINTAIN-FAILED (firing in band but ruler below bar)"
        clause = (f"the gate fired in band ({n_ev} events) but the ruler did "
                  f"not maintain at +300 ({ruler_at['g2'].get(300)} < "
                  f"{MAINTAIN_BAR}; late-grid mean {late_mean:.3f}, min "
                  f"{bars['late_grid_min']}) — not one of the three "
                  f"registered bars; texture recorded (dips/stickiness: "
                  f"{frac_stuck}).")
    log("=" * 78)
    log(f"G2B VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  bars: " + " ".join(f"{k}={bars[k]}" for k in
                               ("MAINTAINS", "GATE_HYPER", "STILL_SILENT",
                                "n_events", "in_band_5_25",
                                "all_checks_fired")))
    log("=" * 78)

    # =====================================================================
    # PLOT — A: ruler sawtooth; B: the onset monitor trace (the sensor);
    # C: per-event pre/post monitor (the re-entry test)
    # =====================================================================
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1.0])
    axA = fig.add_subplot(gs[0, :])
    colors = {"base": "#777777", "g2": "#d62728"}
    for m in ("base", "g2"):
        xs = sorted(ruler_at[m])
        axA.plot(xs, [ruler_at[m][s] for s in xs], "-o", ms=3,
                 color=colors[m], lw=1.8, zorder=3,
                 label={"base": "CELL-BASE (gate off, CPU)",
                        "g2": "CELL-G2B (onset-only monitor)"}[m])
    for e in g2c["event_log"]:
        axA.axvline(e["step"], color="#d62728", ls="-", lw=0.8, alpha=0.45,
                    zorder=1)
    axA.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8, alpha=0.6)
    axA.axhline(SHUT_BAR, color="k", ls=":", lw=0.8, alpha=0.6)
    axA.text(302, MAINTAIN_BAR, " maintain 0.5", va="bottom", fontsize=8)
    axA.text(302, SHUT_BAR, " die 0.27", va="bottom", fontsize=8)
    axA.axhline(g2_root_cells[RULER_KEY], color="#2ca02c", lw=0.8, alpha=0.5)
    axA.set_xlabel("wash step")
    axA.set_ylabel(f"ruler g{RULER_GEO:+d} mean p(Z)")
    axA.set_title(f"G2B THE ONSET-ONLY MONITOR — ruler vs step (root "
                  f"{g2_root_cells[RULER_KEY]:.3f} reused from g2; "
                  f"vlines = gate events; verdict: {verdict})")
    axA.legend(loc="center right", fontsize=8)
    axA.set_xlim(0, 315)

    axB = fig.add_subplot(gs[1, 0])
    xs = [t["step"] for t in trace]
    ys = [t["monitor"] for t in trace]
    fired = [t["fired"] for t in trace]
    axB.plot(xs, ys, "-", color="#1f77b4", lw=1.0, alpha=0.6, zorder=2)
    axB.scatter([x for x, f in zip(xs, fired) if f],
                [y for y, f in zip(ys, fired) if f], s=18, color="#d62728",
                zorder=3, label="fired (event)")
    axB.scatter([x for x, f in zip(xs, fired) if not f],
                [y for y, f in zip(ys, fired) if not f], s=14,
                color="#7f7f7f", zorder=3, label="no fire")
    axB.axhline(G2.THETA_OPEN, color="k", ls="--", lw=0.8)
    axB.text(302, G2.THETA_OPEN, " theta 0.5", va="bottom", fontsize=8)
    axB.axhline(root_mon_onset, color="#2ca02c", lw=0.8, alpha=0.5)
    axB.text(302, root_mon_onset, f" root {root_mon_onset:.2f}",
             va="bottom", fontsize=8, color="#2ca02c")
    axB.set_xlabel("wash step")
    axB.set_ylabel("onset-only monitor p(Z|ctx)")
    axB.set_title("THE SENSOR — onset channel (1 of 7) per check")
    axB.legend(fontsize=8, loc="lower left")

    axC = fig.add_subplot(gs[1, 1])
    evs = [e for e in g2c["event_log"]
           if e.get("pre_event_monitor") is not None]
    if evs:
        steps_ev = [e["step"] for e in evs]
        pre = [e["pre_event_monitor"] for e in evs]
        post = [e["post_refractory_monitor"] for e in evs]
        axC.plot(steps_ev, pre, "o-", color="#d62728", label="pre-event")
        axC.plot(steps_ev, post, "s--", color="#1f77b4",
                 label="post-refractory (+24)")
        axC.axhline(G2.THETA_OPEN, color="k", ls="--", lw=0.8)
        axC.set_xlabel("event step")
        axC.set_ylabel("onset monitor")
        axC.set_title("per-event pre/post monitor (the re-entry test; "
                      f"{frac_post_recovered if frac_post_recovered is not None else float('nan'):.0%} "
                      "recover >= theta)")
        axC.legend(fontsize=8)
        if spac:
            axC.text(0.02, 0.02,
                     f"spacings {bars['spacing_min_median_max']} "
                     f"({bars['frac_spacing_in_20_45']:.0%} in 20-45)",
                     transform=axC.transAxes, fontsize=8, color="#444444")
    else:
        axC.text(0.5, 0.5, "no events", ha="center", va="center")
        axC.set_title("per-event pre/post monitor")

    fig.tight_layout()
    png = rd / "onset_monitor.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    log(f"[plot] {png}")

    # =====================================================================
    # metrics.json
    # =====================================================================
    def strip_cell(mrec: dict) -> dict:
        return {"mode": mrec["mode"], "steps_ran": mrec["steps_ran"],
                "n_events": mrec["n_events"], "n_checks": mrec["n_checks"],
                "realized_r": mrec["realized_r"],
                "event_spacings": mrec["event_spacings"],
                "traj": mrec["traj"],
                "monitor_trace": mrec["monitor_trace"],
                "event_log": mrec["event_log"],
                "devices": {"initial": mrec["initial_device"],
                            "final": mrec["final_device"]},
                "replay_checks": mrec["replay_checks"]}

    metrics = {
        "experiment": "g2b",
        "date": common.now_iso(),
        "purpose": "THE ONSET-ONLY MONITOR — g2's named fix (T122): the "
                   "read monitor watches p(Z|ctx->onset) only (1 of the 7 "
                   "name positions) instead of the self-correlation-"
                   "dominated 7-position mean; the organ is otherwise "
                   "verbatim, root and cue pool REUSED from g2's "
                   "checkpoints.",
        "delta_vs_g2": {
            "g2_monitor": "mean p(true name char) over the 7x8 masked name "
                          "positions (self-correlation-dominated; wash "
                          "spares it at 0.65+)",
            "g2b_monitor": "mean p(true name char) at the FIRST masked "
                           "column of each of the K_MON=8 cue windows "
                           "(mask & cumsum==1) — p(Z|ctx), the channel the "
                           "wash kills",
            "arithmetic": "onset = mask & (mask.cumsum(1) == 1); "
                          "float(ptrue[onset].mean())  [g2: "
                          "float(ptrue[mask].mean())]",
            "everything_else": "VERBATIM g2 (cue pool, rotation, THETA_OPEN "
                               "0.5, REFRACTORY 24, CADENCE 4, e174-arm-B "
                               "replay, e176n-arm-A wash, seed 10902) via "
                               "the G2Net subclass + module-global patch",
        },
        "smoke": SMOKE,
        "threads": torch.get_num_threads(),
        "cpu_only": True,
        "cfg": g2m["cfg"],
        "organ": {**g2m["organ"],
                  "monitor_channel": "onset-only (1 of 7)",
                  "root_monitor_onset": root_mon_onset,
                  "root_monitor_7pos_recheck": root_mon_7pos},
        "compute": {"cpu_only": True, "cuda_visible_devices": "-1",
                    "threads": THREADS, "stagger_s": STAGGER_S,
                    "train_cap_s": G2B_CAP_S},
        "root": {"source": "runs/checkpoints/g2_root.pt (REUSED — not "
                           "rebuilt)",
                 "meta": root_meta,
                 "cells_stored": {k: g2_root_cells[k]
                                  for k in ("gm12", "g0", "gp12", "ce_r")},
                 "gates": {"G_POOL": G_POOL, "G_ROOT2": G_ROOT2}},
        "cells": {m: strip_cell(cells_out[m]) for m in cells_out},
        "dials": {"g2b_g2_300": flat_cells(dials300) if dials300 else None,
                  "root": {k: g2_root_cells.get(k) for k in
                           ("held30_g0", "site_read_onset",
                            "site_read_span", "A129", "row0_strength",
                            "dall_g0")}},
        "gates": {"G_NAMEFREE": G_NAMEFREE, "G_REPLAY": G_REPLAY,
                  "G_STEP_PARITY": G_STEP, "G_DEV": G_DEV},
        "registered_bars": REGISTERED_BARS,
        "registered_prediction": {
            "pre_run": "root onset monitor 0.609 > theta (gate starts "
                       "closed); wash kills the onset read -> first event "
                       "step 24-44; 8-12 events, spacing 24-40; post-event "
                       "+24 ruler 0.4-0.7; +300 ruler phase-riding "
                       "(coin-flip vs 0.5); late-grid mean is the honest "
                       "texture.",
            "discriminating_observation": "post-refractory onset monitor: "
                                          ">= theta after most events -> "
                                          "gate closes (healthy); < theta at "
                                          "every first check -> GATE-HYPER.",
        },
        "adjudication": {"ruler": {"geo": RULER_GEO, "key": RULER_KEY,
                                   "root_value": g2_root_cells[RULER_KEY],
                                   "die_bar": SHUT_BAR,
                                   "maintain_bar": MAINTAIN_BAR},
                         "bars": bars, "verdict": verdict, "clause": clause},
        "honesty": [
            "CHECKPOINT REUSE: the root (and its cue pool) is g2's, built "
            "cuda-install + cpu-consolidation; g2b adds no new training "
            "before the cells. The cells rerun CPU-only — g2b's own "
            "CELL-BASE is the device-consistent contrast; g2's CUDA base "
            "trace is the embedded cross-check (G_DEV).",
            "SINGLE SEED (10902), single lineage, n=1 per cell — point "
            "estimates until replicated (e179's own clause).",
            "THREAD TEXTURE: 4 threads vs g2's 8 can reorder float "
            "reductions; G_ROOT2 reports the bit flag (5e-6) and the 0.05 "
            "fallback.",
            "The +300 endpoint rides cycle phase (e179's sawtooth "
            "convention): the late-grid mean/min over {100,200,300} and "
            "the +50 value are co-reported.",
        ],
        "checkpoints": CKPT_INVENTORY,
        "deviations": deviations,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", G2.E43.jsonable(metrics))
    log(f"[done] metrics -> {rd / 'metrics.json'} "
        f"({metrics['timing_s']:.0f}s total); verdict: {verdict}")


if __name__ == "__main__":
    main()
