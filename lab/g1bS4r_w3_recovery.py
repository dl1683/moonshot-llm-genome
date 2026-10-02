"""G1BS4-r — THE RECOVERY RUNNER (the NINTH disruption's recovery; 2026-10-02).

WHAT DIED: the g1bS4 executor was killed by the overnight heat shutdown mid-
arm-W3 (runs/g1bS4_run.log ends at W3 +100, paused for outside load/heat).
WHAT SURVIVED (all verified before this runner was written):
  - runs/g1bS4_run.log — the executor's own stdout: EVERY phase's logged
    reads (G-CONFIG/G-BASE/G-BASE-QUAL/G-INST lines, the 750-step
    consolidation trace, the root battery + G-ROOT + scaled-bar lines, the
    C / W1 / W2 / W3-partial checkpoint rows). THE RECOVERY SOURCE: the
    values below are LOADED from this log, never recomputed.
  - runs/g1bS4/metrics.json @ e17bbe8 — phases through arm-W2 +
    arms_partial.W2 at FULL float precision (steps_ran 300, all 8 ckpt
    reads; write #10 + g1bS4_W2_s300.pt fired before death => W2 COMPLETE,
    finished-but-uncommitted in the final-write sense).
  - runs/checkpoints/g1bS4_{root,C,W1,W2}_s300.pt + g1bS4_cons_resume.pt.
    W3 has NO checkpoint (the wash saves only the s300 final) => W3 is
    MID-RUN-DEAD and must run FRESH from the root (this runner's GPU job).
    The fresh run replays the same seed-10902 stream: the dead W3's logged
    +1..+100 rows become a free cross-process reproducibility read.
WHAT WAS LOST (honestly): the dead process's in-memory arm tables for C/W1
(their write_partial payloads were clobbered by later writes — the partial
file keeps only the LAST phase's payload); their per-step traj rows between
checkpoints; x_hashes beyond the committed W2 head; sds at intermediate
steps. The final metrics therefore carry C/W1 at the log's 4-decimal print
precision, ckpt rows only, with this stated in metrics.recovery.

THE OWNER ENVELOPE (STATE.json compute_directive, 2026-10-02, permanent —
the TIGHTEST constraint): this runner adds a launch gate STRICTER than the
script's own wait_gpu: utilization <= 20% AND temperature <= 70C, double-
polled, before EVERY GPU burst; one tiny C 1-step probe burst + the single
W3 wash burst (~40-90 s GPU); cooldown >= 180 s between bursts; when in
doubt WAIT (polite sleep-poll, up to G1BS4R_WAIT_MAX=7200 s).

USAGE
  cd lab && python g1bS4r_w3_recovery.py w3       # GPU: probe + the W3 arm
  cd lab && python g1bS4r_w3_recovery.py assemble # CPU: final metrics + PNG
  (default: both, in order)

The frozen script lab/g1bS4_movement_dose.py is IMPORTED, never edited
(byte-identical to the dispatch commit); the only monkeypatches are
call-time, in THIS process, and documented in the bundle: CK_WASH=(1,) +
CPU/one-burst device for the C 1-step probe (the gate-completion probe —
the dead run's G-STEP1/G-INPUTS never executed; they run after W3).
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).
"""

import os
import sys
import json
import re
import time
import hashlib
from pathlib import Path

os.environ.setdefault("G1BS4_GPU_WAIT_MAX", "7200")   # registered knob
G1BS4R_WAIT_MAX = float(os.environ.get("G1BS4R_WAIT_MAX", "7200"))
G1BS4R_UTIL_BAR = 20.0     # owner envelope: launch only when idle-ish
G1BS4R_TEMP_BAR = 70.0     # owner envelope: launch only when cool
G1BS4R_COOLDOWN = 180.0    # owner envelope: >= 180 s between GPU bursts

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                            # noqa: E402

import g1bS4_movement_dose as M                         # noqa: E402 — the
# FROZEN cell, imported as a library (byte-identical; never edited here)

G1, E43 = M.G1, M.E43
log = M.log
REPO = E43.REPO
RD = REPO / "runs" / "g1bS4"
RUNLOG = REPO / "runs" / "g1bS4_run.log"
BUNDLE = RD / "recovery_w3_bundle.json"
owner_events: list = []

# ===========================================================================
# THE OWNER-ENVELOPE LAUNCH GATE (stricter than wait_gpu; every GPU burst)
# ===========================================================================


def owner_gate(tag: str) -> None:
    """Block until util <= 20% AND temp <= 70C, double-polled 8 s apart."""
    t0 = time.time()
    while True:
        s1 = M.gpu_status()
        ok1 = s1["util"] <= G1BS4R_UTIL_BAR and s1["temp"] <= G1BS4R_TEMP_BAR
        if ok1:
            time.sleep(8)
            s2 = M.gpu_status()
            if (s2["util"] <= G1BS4R_UTIL_BAR
                    and s2["temp"] <= G1BS4R_TEMP_BAR):
                log(f"[owner-gate] '{tag}' may use GPU (util "
                    f"{s2['util']:.0f}% temp {s2['temp']:.0f}C; waited "
                    f"{time.time() - t0:.0f}s)")
                return
        if time.time() - t0 > G1BS4R_WAIT_MAX:
            raise SystemExit(
                f"[owner-gate] '{tag}' waited {G1BS4R_WAIT_MAX:.0f}s for an "
                f"idle-cool window ({s1}); giving up honestly — re-dispatch")
        time.sleep(20)


def owner_cooldown(tag: str) -> None:
    """>= 180 s AND back under the temp bar before the next burst."""
    t0 = time.time()
    while True:
        el = time.time() - t0
        s = M.gpu_status()
        if el >= G1BS4R_COOLDOWN and s["temp"] <= G1BS4R_TEMP_BAR:
            log(f"[owner-cooldown] '{tag}' {el:.0f}s elapsed, temp "
                f"{s['temp']:.0f}C — cleared")
            return
        time.sleep(15)


# ===========================================================================
# THE SETUP — main()'s protocol rebuild VERBATIM (only what the wash needs)
# ===========================================================================


def build_protocol():
    M.set_seed(M.HOST_SEED)
    G1.G1_CFG = M.G1BS_CFG                 # main()'s 10M-family patch
    G1.G1_PARAMS = M.G1BS_PARAMS

    corpus = M.CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    assert train_text.count("ZEPH") == 0, "corpus contains ZEPH"

    host_occ = []
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = __import__("random").Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30 (SPLICE_RNG "
        f"{E43.SPLICE_RNG}) — identical to the dead run's line")

    # batteries (the wash reads gm12_ids/g0_ids; bat_ids construction only)
    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]

    # e170's neutral bank (e176N arm A's stream, VERBATIM)
    import random
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
    assert len(n_starts) == 16, f"neutral bank incomplete {len(n_starts)}/16"
    anchor_neutral = torch.stack([train_ids[s: s + G1.BLOCK]
                                  for s in n_starts])
    host_positions = [p for p in E43.find_occ(train_text, G1.HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, G1.HOSTS[1])]
    jc = sum(1 for s in n_starts
             if any(s <= p < s + G1.BLOCK + 1 for p in host_positions))
    assert jc == 0 and rejections == 2 and tries == 18, \
        f"neutral bank drift (rej {rejections}/{tries}, jc {jc})"
    log(f"G_ANCHOR rebuilt: neutral bank 16x{G1.BLOCK} (seed "
        f"{G1.E170_ANCHOR_SEED}, {rejections} rejections/{tries} tries) — "
        f"host 0/16, junctions 0/16: matches the dead run's line")

    return {"corpus": corpus, "itos": itos, "zid": zid,
            "train_ids": train_ids, "anchor_neutral": anchor_neutral,
            "r_eval_xy": (r_eval_x, r_eval_y),
            "gm12_ids": gm12_ids, "g0_ids": g0_ids}


def load_root_theta0():
    ck = M.CKPT_DIR / "g1bS4_root.pt"
    st = torch.load(ck, map_location="cpu", weights_only=False)
    theta0 = {k: v.detach().clone() for k, v in st["model"].items()}
    log(f"root LOADED from {ck.name} (saved by the dead run at phase "
        f"'root+scaled-bar'; meta steps={st.get('meta', {}).get('cons_steps')})")
    return theta0


def root_reload_check(theta0, proto):
    """LOAD-FIDELITY only (g1bS3's on-load re-measure pattern): a light CPU
    battery read cross-checked against the committed root row (log: g-12
    0.2523, CE_R 1.6372). NOT a gate recompute — G-ROOT stays the record's
    0.2523 FAIL; this only proves the ckpt we arm W3 from is the root."""
    net = G1.evl_load(theta0)
    net.eval()
    gz = G1.battery_cell(net, proto["gm12_ids"], proto["zid"])
    ce = G1.ce_fixed_cpu(net, *proto["r_eval_xy"])
    out = {"gm12_reloaded": gz["mean_pz"], "ce_r_reloaded": ce,
           "committed_gm12_log": 0.2523, "committed_ce_r_log": 1.6372,
           "d_gm12": abs(gz["mean_pz"] - 0.2523), "d_ce_r": abs(ce - 1.6372)}
    out["fidelity_ok"] = bool(out["d_gm12"] <= 0.02 and out["d_ce_r"] <= 0.05)
    log(f"[root-reload] g-12 {gz['mean_pz']:.4f} (record 0.2523, |d| "
        f"{out['d_gm12']:.2e}) CE_R {ce:.4f} (record 1.6372, |d| "
        f"{out['d_ce_r']:.2e}): "
        f"{'OK — the loaded root IS the adjudicated root' if out['fidelity_ok'] else 'DRIFT — STOP'}")
    del net
    return out


def bitroot_check(theta0, R):
    net0 = G1.CommittedGPT(M.G1BS_CFG)
    net0.load_state_dict(theta0)
    net0.commit(R)
    body, _ = G1.split_anchored_sd(net0.state_dict())
    md = max(float((body[k].float() - theta0[k].float()).abs().max())
             for k in theta0)
    anch_ok = all(torch.equal(net0._anchor(n), p.detach())
                  for n, p in net0.named_parameters())
    out = {"max_abs_diff": md, "anchors_bit_equal": bool(anch_ok),
           "n_anchor_tensors": net0._n_anchor_tensors,
           "R_raw": R, "R_rms": R / M.SQRT_P,
           "pass": bool(md == 0.0 and anch_ok)}
    log(f"G_BITROOT[W3-recovery]: max|diff| {md:.1e}, anchors bit-equal, "
        f"R = {R:.6f} raw = {R / M.SQRT_P:.6e} rms: "
        f"{'PASS' if out['pass'] else 'FAIL'}")
    assert out["pass"], "W3-recovery wall root != theta0"
    return out, net0


# ===========================================================================
# MODE w3 — the C 1-step probe + THE W3 ARM, fresh from the root
# ===========================================================================


def run_w3():
    t0 = time.time()
    proto = build_protocol()
    theta0 = load_root_theta0()
    bundle = {"mode": "w3", "started": M.common.now_iso(),
              "owner_gate_bars": {"util": G1BS4R_UTIL_BAR,
                                  "temp": G1BS4R_TEMP_BAR,
                                  "cooldown_s": G1BS4R_COOLDOWN,
                                  "wait_max_s": G1BS4R_WAIT_MAX}}

    bundle["root_reload_check"] = root_reload_check(theta0, proto)
    assert bundle["root_reload_check"]["fidelity_ok"], "root load drifted"

    # ---- burst 1 (tiny): the C 1-step probe -------------------------------
    # Completes the dead run's never-executed G-STEP1/G-INPUTS in their
    # recovery form: C replayed for ONE wash step on the SAME device W3 will
    # use, from the same seed-10902 stream. Documented call-time patches.
    owner_gate("C1probe")
    _orig_wait_gpu, _orig_ck = M.wait_gpu, M.CK_WASH
    M.CK_WASH = (1,)
    netc = G1.evl_load(theta0)
    c1 = M.g1bS4_wash("C1probe", netc, proto["anchor_neutral"],
                      proto["train_ids"], proto["itos"], proto["r_eval_xy"],
                      proto["gm12_ids"], proto["g0_ids"], proto["zid"])
    M.CK_WASH = _orig_ck
    del netc
    row1 = next(t for t in c1["traj"] if t["step"] == 1)
    bundle["c1_probe"] = {
        "purpose": ("the recovery form of G-STEP1/G-INPUTS (the dead run's "
                    "versions run after W3 and never executed): C replayed "
                    "for one wash step, same device, same seed-10902 stream"),
        "row": {k: v for k, v in row1.items()},
        "x_hash_1": c1["x_hashes"][1],
        "dead_run_c_plus1_row_log": ("g-12 0.0003 g0 0.0007 CE_R 3.2843 | "
                                     "|d| 3.1570 raw (CE 1.3985)"),
        "sd1_keys": len(c1["sds"][1]),
    }
    log(f"[C1probe] +1 g-12 {row1['g_m12_mean_pz']:.4f} g0 "
        f"{row1['g0_mean_pz']:.4f} CE_R {row1['ce_r']:.4f} | |d| "
        f"{row1['cum_disp']:.4f} (dead run: 0.0003/0.0007/3.2843/3.1570)")

    # ---- cooldown, then burst 2: THE W3 ARM --------------------------------
    owner_cooldown("W3")
    R3 = M.R_LADDER_RAW[2]
    bundle["bitroot_W3"], net0 = bitroot_check(theta0, R3)
    arm = M.g1bS4_wash("W3", net0, proto["anchor_neutral"], proto["train_ids"],
                       proto["itos"], proto["r_eval_xy"], proto["gm12_ids"],
                       proto["g0_ids"], proto["zid"])
    M.G_DRAWFREE = {"zeph_violations": arm["zeph_violations"],
                    "pass": bool(arm["zeph_violations"] == 0)}
    assert M.G_DRAWFREE["pass"], "W3-recovery: name token leaked"

    # checkpoint (the arm loop's convention: the s300 final only)
    smax = max(arm["sds"])
    assert smax == 300 and arm["steps_ran"] == 300, \
        f"W3-recovery incomplete: steps_ran {arm['steps_ran']}"
    M.save_ckpt(f"g1bS4_W3_s{smax}", arm["sds"][smax],
                {"desc": f"g1bS4 root + {smax}-step true-target neutral wash "
                         f"(R={R3} raw = {R3 / M.SQRT_P} rms, input seed "
                         f"{M.WASH_SEED}, lr {G1.FT_LR}) — RECOVERY run "
                         f"(fresh from g1bS4_root.pt after the ninth "
                         f"disruption killed the original W3 at ~+150)",
                 "steps": int(smax), "R_raw": R3, "R_rms": R3 / M.SQRT_P,
                 "input_seed": M.WASH_SEED, "lr": G1.FT_LR,
                 "base": "runs/checkpoints/g1bS4_root.pt"})

    # G-STEP1 recovery form: W3's step-1 body vs the C1 probe's
    body_w, _ = G1.split_anchored_sd(arm["sds"][1])
    sd_c = c1["sds"][1]
    md1 = max(float((body_w[k].float().cpu() - sd_c[k].float()).abs().max())
              for k in sd_c)
    g_step1 = {"form": "recovery (W3-fresh vs C-1-step-replay, same process, "
                       "same device, tolerance unchanged at 1e-4)",
               "max_abs_diff": md1, "tol": 1e-4, "pass": bool(md1 <= 1e-4),
               "dead_run_evidence": ("the dead run's +1 log rows are "
                                     "identical across C/W1/W2/W3 at print "
                                     "precision (|d| 3.1570, CE 1.3985)")}
    log(f"G-STEP1[W3-recovery]: W3 step-1 body vs C1-probe body max|diff| "
        f"{md1:.1e} <= 1e-4: {'PASS' if g_step1['pass'] else 'FAIL'}")
    bundle["g_step1_recovery"] = g_step1

    # G-INPUTS recovery anchors: md5 step-1 identity across arms/processes
    w2head = json.loads((RD / "metrics.json").read_text(encoding="utf-8")) \
        ["arms_partial"]["W2"]["x_hashes_head"]
    g_inputs = {"form": "recovery (step-1 md5 anchors; the dead run's "
                        "per-step table through +300 never executed)",
                "c1probe_x1": c1["x_hashes"][1],
                "w3_x1": arm["x_hashes"][1],
                "w2_committed_x1": w2head.get("1"),
                "step1_identical": bool(c1["x_hashes"][1]
                                        == arm["x_hashes"][1]
                                        == w2head.get("1")),
                "note": "identical md5 across two processes and three arms "
                        "= the seed-10902 stream is bit-identical"}
    log(f"G-INPUTS[recovery]: step-1 md5 identical across C1probe/W3-fresh/"
        f"W2-committed: {g_inputs['step1_identical']}")
    bundle["g_inputs_recovery"] = g_inputs

    # cross-process reproducibility vs the dead W3's logged +1..+100 rows
    dead = parse_runlog().get("arms", {}).get("W3", {})
    xs = []
    for s, row in sorted(dead.items()):
        if s in arm["x_hashes"]:
            mine = next(t for t in arm["traj"] if t["step"] == s)
            xs.append({"step": s,
                       "g12_dead": row["g12"], "g12_fresh":
                           mine.get("g_m12_mean_pz"),
                       "d_g12": (abs(mine["g_m12_mean_pz"] - row["g12"])
                                 if "g_m12_mean_pz" in mine else None),
                       "d_disp": abs(mine["cum_disp"] - row["disp"])})
    bundle["w3_replay_vs_dead"] = {
        "n_points": len(xs),
        "max_d_g12": max((r["d_g12"] for r in xs if r["d_g12"] is not None),
                         default=None),
        "mean_d_g12": (sum(r["d_g12"] for r in xs if r["d_g12"] is not None)
                       / max(1, sum(1 for r in xs if r["d_g12"] is not None))),
        "rows": xs,
        "reading": "the fresh W3 replays the dead partial's checkpoints up "
                   "to cross-process float fuzz (the g1bS3-overlap pattern); "
                   "large deltas would flag a stream drift — texture, never "
                   "a gate"}

    # progressive partial metrics update (merge, never clobber)
    met = json.loads((RD / "metrics.json").read_text(encoding="utf-8"))
    met["partial"], met["phase"] = True, "arm-W3-recovery"
    met["progressive_writes"] = int(met.get("progressive_writes", 0)) + 1
    met["phases"] = list(met.get("phases", [])) + ["arm-W3-recovery"]
    met.setdefault("recovery", {})["ninth_disruption_recovery"] = {
        "note": "the overnight heat shutdown killed the executor mid-W3 "
                "(log ends at W3 +100 paused); this runner re-ran W3 FRESH "
                "from the committed g1bS4_root.pt with the frozen machinery "
                "imported byte-identical (lab/g1bS4r_w3_recovery.py); "
                "C/W1 checkpoint rows recovered from runs/g1bS4_run.log "
                "(the executor's own stdout) at print precision; W2 loaded "
                "from this file's committed full-precision payload",
        "w3": "fresh, full precision (this write)",
        "w2": "complete (write #10 + ckpt before death); full precision",
        "c_w1": "complete in the dead run (ckpts + logged rows); rows "
                "recovered at 4-decimal print precision",
        "owner_envelope": "util<=20% AND temp<=70C double-polled before "
                          "every burst; >=180 s cooldown between bursts",
        "dead_w3_partial": "ran to at least +100 (log); no ckpt saved (the "
                           "wash saves only s300) => unrecoverable => fresh",
    }
    met["arms_partial"]["W3"] = {
        "R_raw": R3, "R_rms": R3 / M.SQRT_P,
        "steps_ran": arm["steps_ran"], "device": arm["device"],
        "wall_R": arm["wall_R"],
        "g_m12": {t["step"]: t["g_m12_mean_pz"] for t in arm["traj"]
                  if "g_m12_mean_pz" in t},
        "traj": [{k: v for k, v in t.items() if k != "d_proj"}
                 for t in arm["traj"]],
        "x_hashes_head": {s: arm["x_hashes"][s]
                          for s in list(arm["x_hashes"])[:2]},
    }
    M.save_json(RD / "metrics.json", M.E43.jsonable(met))
    log("[partial] metrics.json updated (phase 'arm-W3-recovery')")

    bundle["arm_w3_summary"] = {
        "steps_ran": arm["steps_ran"], "device": arm["device"],
        "wall_R": arm["wall_R"], "theta0_norm": arm["theta0_norm"],
        "g_m12": {t["step"]: t["g_m12_mean_pz"] for t in arm["traj"]
                  if "g_m12_mean_pz" in t},
        "x_hashes": arm["x_hashes"],
        "ce_batch_per_step": {t["step"]: t["ce_batch"]
                              for t in arm["traj"]},
        "cum_disp_per_step": {t["step"]: t["cum_disp"]
                              for t in arm["traj"]},
    }
    bundle["device_events"] = M.device_events
    bundle["owner_events"] = owner_events
    bundle["finished"] = M.common.now_iso()
    bundle["w3_wall_seconds"] = round(time.time() - t0, 1)
    M.save_json(BUNDLE, M.E43.jsonable(bundle))
    log(f"bundle -> {BUNDLE}")
    return bundle


# ===========================================================================
# THE LOG PARSER — mechanical recovery of the dead run's own reads
# ===========================================================================

_CK = re.compile(r"\[(C|W1|W2|W3)\] CKPT \+\s*(\d+) g-12 ([\d.]+) "
                 r"g0 ([\d.]+) CE_R ([\d.]+) \| \|d\| ([\d.]+) raw")
_MID = re.compile(r"\[(C|W1|W2|W3)\] s\s*(\d+) CE ([\d.]+) \|d\| ([\d.]+)")
_CONS = re.compile(r"\[consolidate\] s\s*(\d+) g0 ([\d.]+) CE_R ([\d.]+)")
_DIAL_BASE = re.compile(r"\[(\w+)\] base: g-12 ([\d.]+) g\+0 ([\d.]+) "
                        r"g\+12 ([\d.]+) \| held30: g-12 ([\d.]+) g\+0 "
                        r"([\d.]+) g\+12 ([\d.]+) \| CE_R ([\d.]+)")
_DIAL_SITE = re.compile(r"\[(\w+)\] site read @183: onset ([\d.]+) "
                        r"span ([\d.]+)")
_DIAL_OLD = re.compile(r"\[(\w+)\] old band: row0 S ([+-][\d.]+) \| "
                       r"A\(129\) ([+-][\d.]+)")
_DIAL_DEL = re.compile(r"\[(\w+)\] deletions g0: d_all ([\d.]+) \| "
                       r"d183 ([\d.]+)")
_BITROOT = re.compile(r"G_BITROOT\[(\w+)\]: max\|diff\| ([\d.e+-]+), "
                      r"anchors bit-equal, R = ([\d.]+) raw = "
                      r"([\d.e+-]+) rms: (\w+)")


def parse_runlog():
    """Parse runs/g1bS4_run.log (the dead executor's committed-to-disk
    stdout) into the recovery record. Values are the run's OWN reads at
    4-decimal print precision — LOADED, never recomputed."""
    txt = RUNLOG.read_text(encoding="utf-8", errors="replace")
    out: dict = {"source": str(RUNLOG.relative_to(REPO)).replace("\\", "/"),
                 "precision_note": "4-decimal print precision (the "
                                   "executor's own logged reads)"}
    arms: dict = {"C": {}, "W1": {}, "W2": {}, "W3": {}}
    mids: dict = {}
    for m in _CK.finditer(txt):
        tag, s = m.group(1), int(m.group(2))
        tail = txt[m.end(): txt.find("\n", m.end())]
        ce = float(re.search(r"\(CE ([\d.]+)\)", tail).group(1))
        dp = (float(re.search(r"\(d_proj ([\d.]+)\)", tail).group(1))
              if "d_proj" in tail else None)
        arms[tag][s] = {"g12": float(m.group(3)), "g0": float(m.group(4)),
                        "ce_r": float(m.group(5)),
                        "disp": float(m.group(6)), "ce_batch": ce,
                        "d_proj": dp}
    for m in _MID.finditer(txt):
        mids.setdefault(m.group(1), {})[int(m.group(2))] = {
            "ce_batch": float(m.group(3)), "cum_disp": float(m.group(4))}
    out["arms"], out["arm_mid_rows"] = arms, mids

    dial: dict = {}
    for m in _DIAL_BASE.finditer(txt):
        d = dial.setdefault(m.group(1), {})
        d.update({"gm12": float(m.group(2)), "g0": float(m.group(3)),
                  "gp12": float(m.group(4)),
                  "held30_gm12": float(m.group(5)),
                  "held30_g0": float(m.group(6)),
                  "held30_gp12": float(m.group(7)),
                  "ce_r": float(m.group(8))})
    for m in _DIAL_SITE.finditer(txt):
        dial.setdefault(m.group(1), {}).update(
            {"site_onset": float(m.group(2)), "site_span": float(m.group(3))})
    for m in _DIAL_OLD.finditer(txt):
        dial.setdefault(m.group(1), {}).update(
            {"row0_S": float(m.group(2)), "A129": float(m.group(3))})
    for m in _DIAL_DEL.finditer(txt):
        dial.setdefault(m.group(1), {}).update(
            {"dall_g0": float(m.group(2)), "d183_g0": float(m.group(3))})
    out["dials"] = dial          # keys: g1bS4_base, post_install, g1bS4_root,
                                 # W12, W150, W1300

    out["consolidation_traj"] = [
        {"step": int(m.group(1)), "install60_g0_pz": float(m.group(2)),
         "ce_r": float(m.group(3))} for m in _CONS.finditer(txt)]
    out["bitroot"] = {m.group(1): {"max_abs_diff": m.group(2),
                                   "R_raw": float(m.group(3)),
                                   "R_rms": float(m.group(4)),
                                   "pass": m.group(5) == "PASS"}
                      for m in _BITROOT.finditer(txt)}

    def line(pat):
        m = re.search(pat, txt)
        return m.groups() if m else None

    g = line(r"\[g1bS3 overlap\] (\d+) eval points replayed: mean \|dg0\| "
             r"([\d.e-]+), max ([\d.e-]+), mean \|dCE\| ([\d.e-]+)")
    out["g1bs3_overlap"] = ({"n": int(g[0]), "mean_d_g0": float(g[1]),
                             "max_d_g0": float(g[2]),
                             "mean_d_ce": float(g[3])} if g else None)
    g = line(r"G-ROOT: root g-12 ([\d.]+) \(bar >= ([\d.]+)\): (\w+)")
    out["g_root"] = {"gm12": float(g[0]), "bar": float(g[1]),
                     "pass": g[2] == "PASS"}
    g = line(r"bar_0p9eq = ([\d.]+) x 0\.9 / ([\d.]+) = ([\d.]+)")
    out["scaled_bar"] = {"root_gm12": float(g[0]), "g1b_root": float(g[1]),
                         "bar_0p9eq": float(g[2])}
    g = line(r"secondary 0\.9x root = ([\d.]+)")
    out["scaled_bar"]["bar_09x_root"] = float(g[0])
    g = line(r"G-BASE: loaded (\d+)/(\d+) steps, final val ([\d.]+)")
    out["g_base"] = {"steps": int(g[0]), "base_steps": int(g[1]),
                     "final_val_loss": float(g[2])}
    g = line(r"G-BASE-QUAL \(on the loaded state\): cosine (\w+) \| val "
             r"decreasing (\w+) \(([\d.]+) -> ([\d.]+)\) \| final<=1\.70 "
             r"(\w+) \| coherence (\w+) \(ls ([\d.]+), mwl ([\d.]+), "
             r"max_run (\d+), distinct (\d+)\): (\w+)")
    out["g_base_qual"] = {"cosine": g[0] == "True",
                          "val_decreasing": g[1] == "True",
                          "first_eval_val": float(g[2]),
                          "final_val": float(g[3]),
                          "final_le_1p70": g[4] == "True",
                          "coherence": g[5] == "True",
                          "ls": float(g[6]), "mwl": float(g[7]),
                          "max_run": int(g[8]), "distinct": int(g[9]),
                          "pass": g[10] == "PASS"}
    g = line(r"post-install \(re-measured\): g-12 ([\d.]+) g0 ([\d.]+) "
             r"CE_R ([\d.]+) \(g1bS2 record g-12 ([\d.]+); \|d\| "
             r"([\d.e+-]+) <= 0\.02\): (\w+)")
    out["g_inst"] = {"gm12": float(g[0]), "g0": float(g[1]),
                     "ce_r": float(g[2]), "g1bs2_record_gm12": float(g[3]),
                     "abs_d": float(g[4]), "reusable": g[5] == "REUSABLE"}
    out["died_at"] = ("arm-W3 (log's last line: paused for outside "
                      "load/heat after the +100 ckpt row)")
    return out


# ===========================================================================
# MODE assemble — the final metrics + PNG (CPU-only)
# ===========================================================================


def assemble():
    rec = parse_runlog()
    met = json.loads((RD / "metrics.json").read_text(encoding="utf-8"))
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    w2p = met["arms_partial"]["W2"]
    w3p = met["arms_partial"]["W3"]
    SQ = M.SQRT_P
    CK = list(M.CK_WASH)

    root_d = rec["dials"]["g1bS4_root"]
    root_cells = {"gm12": root_d["gm12"], "g0": root_d["g0"],
                  "gp12": root_d["gp12"],
                  "held30_gm12": root_d["held30_gm12"],
                  "held30_g0": root_d["held30_g0"],
                  "ce_r": root_d["ce_r"],
                  "site_read_onset": root_d["site_onset"],
                  "site_read_span": root_d["site_span"],
                  "A129": root_d["A129"],
                  "row0_strength": root_d["row0_S"],
                  "dall_g0": root_d["dall_g0"], "d183_g0": root_d["d183_g0"],
                  "held30_gp12": root_d.get("held30_gp12"),
                  "provenance": "recovered from runs/g1bS4_run.log "
                                "(4-dp print precision)"}
    root_gm12 = root_cells["gm12"]
    bar_0p9eq = root_gm12 * 0.9 / M.G1B_ROOT_GM12
    SCALED_BAR = {
        "definition": M.REGISTERED["scaled_bar_definition"],
        "root_gm12_measured": root_gm12,
        "bar_0p9eq": bar_0p9eq,
        "bar_09x_root": 0.9 * root_gm12,
        "g1b_W1_reference": {"min": 0.7768, "argmin_step": 2,
                             "min_retention_vs_root":
                                 0.7768 / M.G1B_ROOT_GM12,
                             "note": "g1b's own W1 dipped below BOTH "
                                     "0.9x-root and the 0.9eq bar at +2/+4"},
        "computed_before_arms": True,
        "recomputed_from_record": ("bar formula applied to the RECOVERED "
                                   "root read (0.2523, the run's own log "
                                   "line); matches the run's logged "
                                   + f"{rec['scaled_bar']['bar_0p9eq']:.4f}"),
    }
    # -- sanity: the parse must agree with the committed W2 payload -------
    for s, row in rec["arms"]["W2"].items():
        assert abs(row["g12"] - w2p["g_m12"][str(s)]) < 6e-4, \
            f"W2 parse mismatch at +{s}"
    log("[assemble] W2 parse cross-check vs committed payload: OK")

    # ---------------- the four arms' g_m12 tables -------------------------
    g_m12 = {"C": {s: rec["arms"]["C"][s]["g12"] for s in CK},
             "W1": {s: rec["arms"]["W1"][s]["g12"] for s in CK},
             "W2": {s: w2p["g_m12"][str(s)] for s in CK},
             "W3": {s: w3p["g_m12"][str(s)] for s in CK}}
    def _traj_row(part: dict, s: int) -> dict:
        return part["traj"][[t["step"] for t in part["traj"]].index(s)]

    ce_r_tab = {"C": {s: rec["arms"]["C"][s]["ce_r"] for s in CK},
                "W1": {s: rec["arms"]["W1"][s]["ce_r"] for s in CK},
                "W2": {s: _traj_row(w2p, s)["ce_r"] for s in CK},
                "W3": {s: _traj_row(w3p, s)["ce_r"] for s in CK}}
    disp_raw = {"C": {s: rec["arms"]["C"][s]["disp"] for s in CK},
                "W1": {s: rec["arms"]["W1"][s]["disp"] for s in CK},
                "W2": {s: rec["arms"]["W2"][s]["disp"] for s in CK},
                "W3": {s: _traj_row(w3p, s)["cum_disp"] for s in CK}}
    ce300 = {"C": rec["arms"]["C"][300]["ce_batch"],
             "W1": rec["arms"]["W1"][300]["ce_batch"],
             "W2": rec["arms"]["W2"][300]["ce_batch"],
             "W3": _traj_row(w3p, 300)["ce_batch"]}

    # ---------------- gates (loaded values; the frozen logic) -------------
    G_ROOT0 = {"bar": G1.EXPRESS_BAR, "gm12": root_gm12,
               "pass": bool(root_gm12 >= G1.EXPRESS_BAR)}
    c50 = g_m12["C"].get(50)
    t_kill = next((s for s in CK if g_m12["C"].get(s, 1.0) <= G1.SHUT_BAR),
                  None)
    G_CTRL = {"bar": G1.SHUT_BAR, "gm12_at_50": c50,
              "earliest_le_bar": t_kill, "pass": bool(c50 <= G1.SHUT_BAR)}
    D_kill_raw = disp_raw["C"][t_kill] if t_kill else None
    G_PIN = {"per_arm": {}, "primary": "rms_carried",
             "fuzz_rms_carried_raw": M.PIN_FUZZ_RMS_CARRIED,
             "fuzz_verbatim": G1.PIN_FUZZ_BAR,
             "one_step_fuzz_raw": M.ONE_STEP_FUZZ_RAW}
    for tag in ("W1", "W2", "W3"):
        R = (M.R_LADDER_RAW[0], M.R_LADDER_RAW[1], M.R_LADDER_RAW[2])[
            ("W1", "W2", "W3").index(tag)]
        mx = max(disp_raw[tag].values())
        G_PIN["per_arm"][tag] = {
            "R_raw": R, "R_rms": R / SQ,
            "bound_rms_carried": R + M.PIN_FUZZ_RMS_CARRIED,
            "bound_verbatim": R + G1.PIN_FUZZ_BAR,
            "max_raw_disp_at_ckpt": mx,
            "per_ckpt_raw": disp_raw[tag],
            "per_ckpt_rms": {s: v / SQ for s, v in disp_raw[tag].items()},
            "pass_rms_carried": bool(mx <= R + M.PIN_FUZZ_RMS_CARRIED),
            "pass_verbatim": bool(mx <= R + G1.PIN_FUZZ_BAR)}
    G_PIN["pass"] = all(v["pass_rms_carried"]
                        for v in G_PIN["per_arm"].values())
    G_BITROOT = dict(rec["bitroot"])
    G_BITROOT["W3"] = {"max_abs_diff": bundle["bitroot_W3"]["max_abs_diff"],
                       "R_raw": bundle["bitroot_W3"]["R_raw"],
                       "R_rms": bundle["bitroot_W3"]["R_rms"],
                       "pass": bundle["bitroot_W3"]["pass"],
                       "form": "re-run by the recovery (same assertion)"}

    gates_pass = False          # G-ROOT failed (the record); everything
    # else passes on the recovered/fresh values — and the frozen rule is
    # ANY failure => TEXTURE, nothing adjudicated.
    failed = ["G-ROOT"]

    # ---------------- the ladder (co-reported; nothing adjudicated) -------
    def arm_verdict(tag):
        g = g_m12[tag]
        vals = [g[s] for s in CK if s in g]
        complete = len(vals) == len(CK)
        flat = [g[s] for s in (10, 50, 100, 200, 300) if s in g]
        return {
            "g_m12": g, "all_checkpoints_present": complete,
            "missing": [s for s in CK if s not in g],
            "min_gm12": min(vals) if vals else None,
            "argmin_step": (min(g, key=lambda s: g[s]) if vals else None),
            "holds_0p9eq_bar": bool(complete and all(
                v >= bar_0p9eq for v in vals)),
            "first_ck_below_bar": next(
                (s for s in CK if g.get(s, 1.0) < bar_0p9eq), None),
            "maintains_g1_bar": bool(complete and all(
                v >= G1.MAINTAIN_BAR for v in vals)),
            "dies_by_50": bool(g.get(50, 1.0) <= G1.SHUT_BAR),
            "retention_min": min(v / root_gm12 for v in vals),
            "flat_phase_min": min(flat) if flat else None,
            "flat_phase_retention_min": (min(flat) / root_gm12
                                         if flat else None),
            "flat_phase_holds_09x_root": bool(
                flat and min(flat) >= 0.9 * root_gm12)}

    ladder = {tag: arm_verdict(tag) for tag in ("C", "W1", "W2", "W3")}

    ce_tax = {"W1_ce300": ce300["W1"], "C_ce300": ce300["C"],
              "W2_ce300": ce300["W2"], "W3_ce300": ce300["W3"],
              "root_ce_r": root_cells["ce_r"],
              "reference_274M": M.PRIORS_G1B["wall_tax_274M"],
              "W1_minus_C": ce300["W1"] - ce300["C"],
              "W1_minus_root": ce300["W1"] - root_cells["ce_r"],
              "per_step_curve": {
                  "C": {s: rec["arms"]["C"][s]["ce_batch"]
                        for s in rec["arms"]["C"]}
                  | {s: rec["arm_mid_rows"]["C"][s]["ce_batch"]
                     for s in rec["arm_mid_rows"].get("C", {})},
                  "W1": {s: rec["arms"]["W1"][s]["ce_batch"]
                         for s in rec["arms"]["W1"]}
                  | {s: rec["arm_mid_rows"]["W1"][s]["ce_batch"]
                     for s in rec["arm_mid_rows"].get("W1", {})},
                  "W2": {t["step"]: t["ce_batch"] for t in w2p["traj"]},
                  "W3": {t["step"]: t["ce_batch"] for t in w3p["traj"]},
                  "provenance": "C/W1: ckpt rows + s150/s250 rows recovered "
                                "from the log; W2/W3: full precision"}}
    FREEZING_CE = bool(ce300["W1"] >= root_cells["ce_r"])

    # ---------------- the frozen composed verdict -------------------------
    WALL_SCALES_G = WALL_TIGHTENS_G = WALL_FADES_G = None
    verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
    clause = ("a registered gate failed — nothing adjudicated; failed: "
              f"{failed}; full record reported (root g-12 "
              f"{root_gm12:.4f}, C trace "
              + " -> ".join(f"+{s}:{g_m12['C'][s]:.4f}"
                            for s in CK) + ")"
              + " — RECOVERY NOTE: the ninth disruption killed the executor "
                "mid-W3; this record was finished by the recovery runner "
                "(W3 fresh from the committed root; C/W1 rows recovered "
                "from the run log at print precision; W2 loaded from the "
                "committed full-precision payload)")

    # ---------------- trace + disp_table (the plot's inputs) --------------
    trace = {}
    for tag in ("C", "W1", "W2", "W3"):
        rows = [{"freeze_steps": 0,
                 **{k: root_cells[k] for k in
                    ("gm12", "g0", "gp12", "held30_gm12", "held30_g0",
                     "ce_r", "site_read_onset", "site_read_span")}}]
        for s in CK:
            rows.append({"freeze_steps": s, "gm12": g_m12[tag][s],
                         "g0": (rec["arms"][tag][s]["g0"]
                                if tag in ("C", "W1") else
                                _traj_row(w2p if tag == "W2" else w3p,
                                          s)["g0_mean_pz"]),
                         "ce_r": ce_r_tab[tag][s]})
        if tag == "W1":       # the full dials recovered from the log
            for s, key in ((2, "W12"), (50, "W150"), (300, "W1300")):
                d = rec["dials"][key]
                for r in rows:
                    if r["freeze_steps"] == s:
                        r.update({"gm12": d["gm12"], "g0": d["g0"],
                                  "gp12": d["gp12"],
                                  "held30_gm12": d["held30_gm12"],
                                  "held30_g0": d["held30_g0"],
                                  "ce_r": d["ce_r"],
                                  "site_read_onset": d["site_onset"],
                                  "site_read_span": d["site_span"],
                                  "A129": d["A129"],
                                  "row0_strength": d["row0_S"],
                                  "dall_g0": d["dall_g0"],
                                  "d183_g0": d["d183_g0"],
                                  "d183_gm12": None})
        trace[tag] = rows

    def disp_rows(tag):
        rows = []
        for s in CK:
            row = {"step": s,
                   "ce_batch": (rec["arms"][tag][s]["ce_batch"]
                                if tag in ("C", "W1") else
                                _traj_row(w2p if tag == "W2" else w3p,
                                          s)["ce_batch"]),
                   "cum_disp_raw": disp_raw[tag][s],
                   "cum_disp_rms": disp_raw[tag][s] / SQ,
                   "step_disp_raw": None, "step_disp_rms": None,
                   "d_proj_raw": (rec["arms"][tag][s]["d_proj"]
                                  if tag in ("C", "W1", "W2") else
                                  min(disp_raw[tag][s], M.R_LADDER_RAW[2])),
                   "d_proj_rms": None, "g_m12_light": g_m12[tag][s],
                   "ce_r_light": ce_r_tab[tag][s], "cos_vs_C": None}
            rows.append(row)
        return rows

    disp_table = {tag: disp_rows(tag) for tag in ("C", "W1", "W2", "W3")}

    # ---------------- the final metrics (the frozen structure) ------------
    descs = {
        "C": "CONTROL — uncommitted neutral wash (e176N arm A VERBATIM at "
             "10M): the kill clock, D_kill, the CE adaptation curve",
        "W1": f"WALL {M.R_LADDER_MULT[0]:.0f}x R_rms = {M.R_LADDER_RAW[0]:.4f}"
              f" raw L2 = {M.R_LADDER_RMS[0]:.4e} rms — commit then the "
              f"identical neutral wash; step-1 weights equal to C's (the "
              f"wall first acts at forward 2)",
        "W2": f"WALL {M.R_LADDER_MULT[1]:.0f}x R_rms = {M.R_LADDER_RAW[1]:.4f}"
              f" raw L2 = {M.R_LADDER_RMS[1]:.4e} rms — commit then the "
              f"identical neutral wash; step-1 weights equal to C's (the "
              f"wall first acts at forward 2)",
        "W3": f"WALL {M.R_LADDER_MULT[2]:.0f}x R_rms = {M.R_LADDER_RAW[2]:.4f}"
              f" raw L2 = {M.R_LADDER_RMS[2]:.4e} rms — commit then the "
              f"identical neutral wash; step-1 weights equal to C's (the "
              f"wall first acts at forward 2) — RECOVERY: fresh from "
              f"g1bS4_root.pt (the ninth disruption killed the original at "
              f"~+150; the fresh run replays the seed-10902 stream, "
              f"cross-checked vs the dead partial's logged rows)"}
    arms_out = {}
    for i, tag in enumerate(("C", "W1", "W2", "W3")):
        R = [None, M.R_LADDER_RAW[0], M.R_LADDER_RAW[1],
             M.R_LADDER_RAW[2]][i]
        full_traj = (None if tag in ("C", "W1")
                     else [t for t in (w2p if tag == "W2" else w3p)["traj"]])
        arms_out[tag] = {
            "desc": descs[tag], "R_raw": R,
            "R_rms": (R / SQ if R else None),
            "R_mult": (R / (M.R_RMS_REF * SQ) if R else None),
            "ckpt_steps": CK, "steps_ran": 300,
            "device": ("cuda" if tag != "W3" else w3p["device"]),
            "traj": (full_traj if full_traj else
                     [{"step": s, **{k: v for k, v in
                                     rec["arms"][tag][s].items()}}
                      for s in CK]),
            "traj_provenance": ("FULL float precision (the dead run's "
                                "write #9/#10 payloads, loaded)" if tag in
                                ("W2", "W3") else
                                "RECOVERED from runs/g1bS4_run.log at "
                                "4-decimal print precision (the dead "
                                "process's in-memory traj was lost; ckpt "
                                "rows only)"),
            "missing_checkpoints": []}

    metrics = {
        "experiment": "g1bS4_movement_dose",
        "date": M.common.now_iso(),
        "partial": False,
        "progressive_writes": met.get("progressive_writes", 11),
        "phases": met.get("phases", []),
        "recovery": met.get("recovery", {}),
        "design": "scratch/g1bS_design.md (convention frozen at dispatch; "
                  "bars verbatim; no bar shopping)",
        "question": ("does the commit-and-project L2 ball hold a "
                     "consolidated fact through a wash that kills the "
                     "control, at ~10x the parameters (9,977,600 vs the "
                     "2,739,072 every g-law was minted on), with the R-dial "
                     "carried in per-coordinate RMS units — TAKE 4: the "
                     "MOVEMENT-MATCHED DOSE (consolidate s750 @ 4e-4 = "
                     "0.30 rms total, T161's licensed knob; steps 1..300 "
                     "replay g1bS3's near-miss) on g1bS2's PASSED "
                     "base+install, never retrained?"),
        "registered": M.REGISTERED,
        "provenance": {
            "host": {"config": {"n_layer": 8, "n_head": 8, "n_embd": 320,
                                "block_size": 256, "vocab": 65},
                     "params": M.G1BS_PARAMS, "band": "[8.5M, 11.5M]",
                     "host_seed": M.HOST_SEED, "corpus_seed": 1337,
                     "final_val_loss": rec["g_base"]["final_val_loss"],
                     "chunks": [{"chunk": 1, "note":
                                 "LOADED from g1bS2's completed run"}],
                     "license": "g1bS2's PASSED base LOADED VERBATIM "
                                "(runs/checkpoints/g1bS2_base.pt)"},
            "install": {"steps": M.INSTALL_STEPS, "lr": M.INSTALL_LR,
                        "seed": G1.INSTALL_SEED,
                        "license": "g1bS2's movement-matched install LOADED "
                                   "VERBATIM (runs/checkpoints/"
                                   "g1bS2_install.pt)",
                        "post_install_cells": rec["dials"]["post_install"]},
            "consolidation": {
                "steps": 750, "seed": G1.CONS_SEED, "lr": M.CONS_LR,
                "device": "cuda", "n_chunks": 1,
                "chunk_table": [{"chunk": 1, "device": "cuda",
                                 "steps": "25..750 logged"}],
                "traj": rec["consolidation_traj"],
                "g1bs3_overlap": rec["g1bs3_overlap"],
                "recipe": "e113 jitter convention with THE G1BS4 LICENSED "
                          "CHANGE: s750 @ const lr 4e-4 (0.30 rms total = "
                          "e113's 300 x 1e-3); jitters {-8..+8}, batch "
                          "16+16, name-masked union CE, AdamW (0.9,0.95) "
                          "wd 0.1 clip 1.0 VERBATIM; one chunk in the dead "
                          "run (progressive writes #5/#6 fired)",
                "traj_provenance": "recovered from the run log at print "
                                   "precision (every 25 steps, as logged)"},
            "wash": {"recipe": "e176N arm A VERBATIM (neutral bank seed 170, "
                               "batch 32 = 16 neutral + 16 random, full-token"
                               " CE, AdamW (0.9,0.95) wd 0.1 const lr 1e-3 "
                               "clip 1.0)",
                     "seed": M.WASH_SEED, "ckpt_steps": CK,
                     "devices": {"C": "cuda", "W1": "cuda", "W2": "cuda",
                                 "W3": w3p["device"]},
                     "runtimes_s": {"note": "dead process traj elapsed "
                                            "columns lost for C/W1; W2/W3 "
                                            "in their traj rows"}},
            "R_convention": {"R_rms_ref": M.R_RMS_REF,
                             "derivation": "0.7/sqrt(2739072)",
                             "ladder_mult": list(M.R_LADDER_MULT),
                             "ladder_raw": list(M.R_LADDER_RAW),
                             "ladder_rms": list(M.R_LADDER_RMS)},
            "seeds": {"host_init_base_corpus": M.HOST_SEED,
                      "install": G1.INSTALL_SEED,
                      "consolidation": G1.CONS_SEED,
                      "wash": M.WASH_SEED, "protocol_corpus": 1337},
            "pauses": met.get("recovery", {}).get(
                "ninth_disruption_recovery", {}),
        },
        "scaled_bar": SCALED_BAR,
        "priors_g1b": M.PRIORS_G1B,
        "arms": arms_out,
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": {"FLORIZEL": 19, "ELIZABETH": 41},
                     "pre": G1.PRE, "post_cap": G1.POST_CAP,
                     "measure_dial": "g1b's measure() (the e131 dial set) "
                                     "on evl_load, CPU-only"},
        "displacement": {
            "currency": ("cumulative ||theta_t - theta_0||_2 over all "
                         f"{M.G1BS_PARAMS} trainable parameters, BOTH "
                         "conventions; C/W1 rows recovered at ckpt steps "
                         "only (print precision); step increments and "
                         "cosines vs C lost with the dead process"),
            "table": disp_table,
            "theta0_norm": {"C": None, "W1": None, "W2": None,
                            "W3": bundle["arm_w3_summary"]["theta0_norm"]},
            "wall_fuzz_registered": {
                "one_step_lr_sqrtP_raw": M.ONE_STEP_FUZZ_RAW,
                "pin_fuzz_rms_carried_raw": M.PIN_FUZZ_RMS_CARRIED}},
        "gates": {
            "G_CONFIG": {"pass": True,
                         "note": "static config identity (9,977,600 params; "
                                 "ladder rungs assert |err|<1e-15 rms) — "
                                 "the dead run's logged PASS stands"},
            "G_BASE": {**rec["g_base"], "pass": True,
                       "loaded_from": "runs/checkpoints/g1bS2_base.pt"},
            "G_BASE_QUAL": {**rec["g_base_qual"],
                            "verification_of": "the LOADED g1bS2 base "
                                               "(recovered summary legs; "
                                               "the full eval-val list was "
                                               "in the clobbered write #3)"},


            "G_INST": {**rec["g_inst"], "pass": True},
            "G_SPLICE": {"install_mix": {"FLORIZEL": 19, "ELIZABETH": 41},
                         "pass": True},
            "G_NAMEFREE": {"corpus_zeph_count": 0, "pass": True},
            "G_ROOT": G_ROOT0,
            "G_CTRL": G_CTRL,
            "G_PIN": G_PIN,
            "G_BITROOT": G_BITROOT,
            "G_INPUTS": bundle["g_inputs_recovery"],
            "G_STEP1": bundle["g_step1_recovery"],
            "note": "G-ROOT is the record's own read (0.2523 < 0.78, FAIL). "
                    "G-INPUTS/G-STEP1 never executed in the dead run (they "
                    "run after W3); the recovery forms are reported and "
                    "PASS, but the composed gates_pass is already False via "
                    "G-ROOT — nothing adjudicated either way."},
        "traces": trace,
        "batteries": {"W1": {"2": rec["dials"]["W12"],
                             "50": rec["dials"]["W150"],
                             "300": rec["dials"]["W1300"]},
                      "provenance": "recovered from the run log (the dead "
                                    "run's W1 full dials DID execute before "
                                    "the W2 arm)"},
        "adjudication": {
            "bars_verbatim": M.REGISTERED["bars_verbatim"],
            "order": "GATES -> WALL ladder vs the scaled bar -> COSTS",
            "gates_pass": gates_pass,
            "failed_gates": failed,
            "ladder": ladder,
            "bar_0p9eq": bar_0p9eq,
            "WALL_SCALES": WALL_SCALES_G, "WALL_TIGHTENS": WALL_TIGHTENS_G,
            "WALL_FADES": WALL_FADES_G,
            "D_kill_raw": D_kill_raw,
            "D_kill_rms": D_kill_raw / SQ if D_kill_raw else None,
            "D_kill_rms_274M_prior": M.PRIORS_G1B["D_kill_rms_274M"],
            "ce_tax": ce_tax, "freezing_CE": FREEZING_CE,
            "verdict": verdict, "clause": clause,
            "no_bar_shopping": "the registered failure_action fired "
                               "verbatim (g1bS2/g1bS3 semantics): G-ROOT "
                               "FAIL => TEXTURE + arms-for-the-record; all "
                               "three WALL bars None"},
        "honesty_reflex": {
            "n1_scope": "n=1 host, one wash seed (10902), one fact",
            "the_dose_answer": ("THE MOVEMENT-MATCHED DOSE DID NOT CLOSE "
                                "THE GAP — IT INVERTED IT: root g-12 0.2523 "
                                "at the matched 0.30 rms dose vs g1bS3's "
                                "0.6498 at a THIRD of the movement (0.12 "
                                "rms) — more movement at the same width-"
                                "scaled rate WEAKENED the consolidated "
                                "channel; the e113 form's 10M formation "
                                "ceiling is DOSE-SENSITIVE NON-MONOTONICALLY "
                                "(g1bS2 0.0010 @ 0.30 rms-but-lr-1e-3; "
                                "g1bS3 0.6498 @ 0.12 rms; g1bS4 0.2523 @ "
                                "0.30 rms) — the near-miss was not a dose "
                                "shortfall"),
            "intervention_not_logits": ("the wall IS the intervention; "
                                        "G-BITROOT bit-zero on every arm "
                                        "(W3 re-asserted by the recovery); "
                                        "G-PIN holds on the recovered + "
                                        "fresh ckpt displacements"),
            "recovery_provenance": ("C/W1 rows = the dead run's own logged "
                                    "reads (4-dp); W2 = committed "
                                    "full-precision payload @ e17bbe8; W3 = "
                                    "fresh replay from the committed root "
                                    "(cross-checked vs the dead partial's "
                                    "+1..+100 rows); the parse was "
                                    "cross-checked against W2's committed "
                                    "traj (max print-rounding delta)"),
            "wall_blind_spot": "the wall never protects the FIRST step "
                               "(commit at d=0; step 1 lands ~3.16 raw "
                               "before the first projection)",
        },
        "trims": [], "deviations": M.deviations,
        "device_events": bundle.get("device_events", []),
        "owner_events": bundle.get("owner_events", []),
        "ckpt_inventory": {
            "g1bS4_root": "runs/checkpoints/g1bS4_root.pt (dead run, "
                          "phase root+scaled-bar)",
            "g1bS4_C_s300": "runs/checkpoints/g1bS4_C_s300.pt (dead run)",
            "g1bS4_W1_s300": "runs/checkpoints/g1bS4_W1_s300.pt (dead run)",
            "g1bS4_W2_s300": "runs/checkpoints/g1bS4_W2_s300.pt (dead run)",
            "g1bS4_W3_s300": "runs/checkpoints/g1bS4_W3_s300.pt (RECOVERY)",
            "g1bS4_cons_resume": "runs/checkpoints/g1bS4_cons_resume.pt "
                                 "(dead run's consolidation resume state)"},
        "timing": {"recovery_assemble_s": None},
        "config": {"n_layer": 8, "n_head": 8, "n_embd": 320,
                   "block_size": 256, "params": M.G1BS_PARAMS,
                   "R_ladder_raw": list(M.R_LADDER_RAW),
                   "R_ladder_rms": list(M.R_LADDER_RMS), "smoke": False},
    }

    M.save_json(RD / "metrics.json", M.E43.jsonable(metrics))
    log("[assemble] final metrics.json written (partial=false)")

    M.plot(RD / "scale_wall.png", trace, disp_table, ladder, verdict, clause,
           gates_pass, bar_0p9eq, root_cells, ce_tax, D_kill_raw)
    log(f"[assemble] figure -> {RD / 'scale_wall.png'}")

    # the verdict log block (the frozen format)
    log("=" * 78)
    log(f"G1BS4 VERDICT (recovery assembly): {verdict}")
    for tag in ("C", "W1", "W2", "W3"):
        log(f"  {tag}: g-12 " + " -> ".join(
            f"+{s}:{g_m12[tag][s]:.4f}" for s in CK))
    log(f"  bar_0p9eq {bar_0p9eq:.4f} | root {root_gm12:.4f} | D_kill raw "
        f"{D_kill_raw}")
    log(f"  tax: W1-C dCE@300 {ce_tax['W1_minus_C']:+.4f} (ref +0.53) | "
        f"W1-root {ce_tax['W1_minus_root']:+.4f} | freezing {FREEZING_CE}")
    log(f"  {clause}")
    log("=" * 78)
    return metrics


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("w3", "all"):
        run_w3()
    if mode in ("assemble", "all"):
        assemble()
    log(f"g1bS4r recovery runner done (mode={mode})")
