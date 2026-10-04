"""E235 — THE SECOND 10M ROOT'S LIFT (T211's registered follow-up (a),
verbatim).

WHY (T211's registration, THINKING.md T211, prediction (a)): e227 found
the few-shot exemplar lift PRESENT at 2.74M (+0.185) and ABSENT at 10M
(-0.019, the 2-shot Z-form gating 0/30) — but the 10M root e227 read was
g1bS7's redrawn consolidation (cons seed 10903, the "momentum-variant" in
the NOTES shorthand; runs/e227/metrics.json scale10). Is the absence SCALE
or CONSOLIDATION VARIANT? The discriminating root: g1bS5's PEAK root
(runs/checkpoints/g1bS5_root_m020.pt — the ORIGINAL-draw consolidation
(cons seed 10901) at the same movement 0.2 rms; committed ruler root_gm12
0.7676599621772766, runtime-read from runs/g1bS5/metrics.json). If the
lift appears there, the absence was the variant; if absent on BOTH 10M
roots, it is scale (in this instrument form).

THE CELL (dispatch letter, eval-only, CPU, matched conventions — the
battery/lift/wash instruments are lab/e227_faculty_at_home.py's, imported
VERBATIM, not retyped):
  (1) the t=0 exemplar-lift battery VERBATIM from e227's conventions on
      g1bS5's peak root: the MIRABEL untrained-name lift read (mis2M_all
      − snip0M_all, the ungated full held-30 instrument per e227's 10M
      addendum); the mis2 Z/M split (mis2Z vs mis2M, with tmpl2/snip0Z
      co-reads); the 2-shot Z-form gate count at t=0 (e182's gate
      verbatim, per-organism screen).
  (2) the same reads on g1bS7's root REPRODUCED as the control — must
      match e227's committed numbers (runs/e227/metrics.json scale10:
      lift_all −0.018882, gates 0/30, ruler 0.935076, ce_r 1.677454, …);
      THE GATE — if it does not reproduce, STOP AND REPORT.
  (3) a short wash continuation on the g1bS5 root at e227's lineage dose
      (lr 1e-3, the locked seed-10902 stream — the arm where the 2.74M
      lift SURVIVED and returned post-kill; runs/e227/journal_L10902:
      lift {t0 +0.185, +2 −0.050, +10 +0.314, +50 +0.154}), grid
      {t0, +2, +10, +50} — CPU steps on 10M are slow, the grid kept
      minimal and stated — does the lift survive the wash on the second
      root as it did at 2.74M? (DESCRIPTIVE: the dispatch registers no
      wash bar; the trajectory is read verbatim against the 2.74M shape.)

REGISTERED BARS (frozen VERBATIM from the dispatch BEFORE any compute;
adjudicate against exactly this; no bar shopping):
  - VARIANT-NOT-SCALE — "the lift at t=0 on g1bS5's peak root >= +0.05
    (and/or the 2-shot Z-form gates >= 10/30) where g1bS7's reproduced
    read stays absent — the 10M absence was the consolidation variant;
    the faculty is present at 10M; the dissociation readjusts to a
    consolidation axis"
  - SCALE-CONFIRMED — "the lift absent on BOTH 10M roots (reproduced
    control intact) — the absence is scale in this instrument form;
    T211's table stands"
  - PARTIAL — "anything between — the reads verbatim (t0 lift, gate
    counts, the wash trajectory), no narrative inflation"

REGISTERED PREDICTION (T211's own): VARIANT-NOT-SCALE.

OPERATIONALIZATIONS (frozen before compute; the bars are qualitative so
every numeric form below is a REGISTERED ADDITION, flagged for the
report; they fix the clauses, they do not move the bars):
  * "the lift at t=0" := the UNGATED full held-30 lift (p_M 2-shot
    MIRABEL minus p_M 0-shot, identical carriers) — e227's 10M addendum
    instrument (gating the lift on the Z-read censors exactly the
    regime where it is informative; the gated set is EMPTY on g1bS7).
  * "gates" := e182's gate VERBATIM ((argmax==Z & p>=0.8) | (rank<5 &
    p>=0.5)) counted over the 30 2-shot Z-form windows at t=0
    (per-organism screen, e227's discipline).
  * "stays absent" (the g1bS7 control) := the reproduced reads match
    e227's committed scale10 values (G_CTRL below) — lift_all
    non-positive, gates 0/30.
  * "absent" (the SCALE-CONFIRMED leg) := lift_all <= 0.0 (non-positive —
    the same shape as g1bS7's −0.019) AND gates < 10/30. A POSITIVE but
    sub-bar lift (0 < lift < +0.05) with sub-bar gates is PARTIAL (a
    whisper: neither absence nor the bar).
  * ADJUDICATION PARTITION: VARIANT-NOT-SCALE := (lift5 >= +0.05) OR
    (gates5 >= 10/30), control PASS; SCALE-CONFIRMED := (lift5 <= 0.0
    AND gates5 < 10/30), control PASS; PARTIAL := everything else
    (0 < lift5 < +0.05 with gates5 < 10/30, or any honest in-between
    texture — reads verbatim). G_CTRL FAIL -> STOP, no adjudication.
  * The wash trajectory carries NO bar (descriptive, read verbatim
    against e227's committed 2.74M L10902 shape).

VERIFICATION GATES (registered tolerances, frozen before compute):
  * G_ROOT5 — runs/checkpoints/g1bS5_root_m020.pt loads bit-exact
    (max|diff| 0.0 vs file; params 9,977,600) and the t=0 ruler
    (install-60 g-12 battery_cell) reproduces g1bS5's committed
    root_gm12 0.7676599621772766 (runtime-read from
    runs/g1bS5/metrics.json ckpt_inventory) within 5e-6 (CPU fp32,
    8 threads — the e223 hard-bound reference convention).
  * G_CTRL (THE gate) — g1bS7's reproduced reads match e227's committed
    runs/e227/metrics.json scale10 block within 5e-6 on every quantity
    (ruler 0.935076117515564; near 0.6394560010482867; ctrl
    0.8930203579366207; tmpl2/mis2Z/mis2M/snip0Z/snip0M _all means;
    lift_all −0.018882022045242287; ce_r 1.677453637123108) with the
    2-shot Z-form gate count == 0 exactly and the checkpoint md5
    1464bba48b574bc6a4472413f2b8a824. FAIL -> STOP AND REPORT.
  * G_CORP — ZEPH-free train text; install mix FLORIZEL 19 / ELIZABETH
    41 after the SPLICE_RNG shuffle (the g1b gate).
  * G_ANCHOR — the e170 neutral bank identity (16x256, 0 host content,
    0 junctions) — the wash's provenance tie.
  * G_BATT — deterministic battery windows; ctrl windows wash-disjoint;
    committed instruments (fact/near) read whole.
  * G_CPU — GPU PARKED before any wash call (e233 owns the GPU lane);
    CPU load checked and logged at start (e234 is the CPU neighbor —
    load-polite: one wash, 8 threads, no parallelism).

HONESTY REFLEX (before believing any trace):
  * Does the fact channel alone predict the lift? NO — the lift is an
    internal 2-shot-minus-0-shot difference on IDENTICAL carriers
    (MIRABEL: 7 chars, 0 corpus occurrences); the two roots' fact
    channels differ (g1bS5 ruler 0.768 vs g1bS7's 0.935 — the roots'
    own difference, the very confound being priced; co-reported).
  * Does intervening change it? The exemplar-content swap is the
    in-prompt intervention (mis2 vs tmpl2); the consolidation draw IS
    the treatment (10901 vs 10903 at the same movement/lr protocol).
  * n=1 per root; the wash is n=1 draw — the LOCKED lineage stream
    10902, the same stream e227 read at both scales (stream-matched).
  * The form bundle (length + exemplars + separators) is priced by
    snip0 (length-matched); e227's disclosure carries unchanged.
  * Nothing guaranteed; no bar shopping.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (e233 owns the GPU lane; e234 is
CPU-heavy — load checked, threads 8 = the e227 convention the control's
bit-exactness needs); probes CPU fp32; the wash CPU fp32 under g1_wash's
own TRAIN_CAP_CPU = 1800 s guard; grid {0,+2,+10,+50} minimal per the
dispatch.

PROVENANCE: the organism + battery + lift + wash instruments are
lab/e227_faculty_at_home.py VERBATIM via import (build_batteries /
probe_items / gate_pass / battery constants; G1.g1_wash / battery_cell /
evl_load / load_g1 / val_windows via g1_anchored_ball), itself the
e182c2/g1b/e113 lineage; the roots are g1bS5's and g1bS7's committed
checkpoints (g1bS2's licensed base+install + e113 jitter consolidation
at movement 0.2 rms — cons seed 10901 (the original draw) vs 10903 (the
redraw); never retrained). Builds on: e227/T211 (the cross-scale
dissociation + this registration), g1bS5/g1bS7 (the two 10M roots),
g1b (the wash stream). NEW: the second 10M root's t=0 lift + gate
census, the g1bS7 control reproduction gate, the g1bS5 lineage-dose wash
lift trajectory at {0,+2,+10,+50}.

Outputs: runs/e235/{metrics.json (PROGRESSIVE), journal_5L10902.json,
e235_second_10m_lift.png} + runs/e235_run.log. No NOTES/THINKING/QUEUE/
STATE edits (dispatch).

Run:  cd lab && python e235_second_10m_lift.py   (E235_SMOKE=1: 2-step
      wash, grid {1,2}, own smoke dir, nothing adjudicated)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402
import e227_faculty_at_home as E227                    # noqa: E402 — EVERYTHING
import g1_anchored_ball as G1                          # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

# e227's import sets torch threads to 8 (the lab convention the control's
# bit-exactness needs) and G1.G1_CFG to the 2.74M family; we take the 10M
# cfg from e227's own registered constants.
G1.G1_CFG = E227.SCALE10_CFG
G1.G1_PARAMS = E227.SCALE10_PARAMS

SMOKE = os.environ.get("E235_SMOKE") == "1"
NAME = "E235_SECOND_10M_LIFT"
DIRNAME = "e235_smoke" if SMOKE else "e235"

T0 = time.time()
LOG_PATH = common.REPO / "runs" / (DIRNAME + "_run.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ------------------------------------------------------- frozen constants
ROOT5_CK = "g1bS5_root_m020.pt"        # the discriminating root (cons 10901)
ROOT7_CK = "g1bS7_root_m020f.pt"       # the control root (cons 10903)
G_ROOT_TOL = 5e-6                      # the e223/e227 ruler tolerance
CTRL_TOL = 5e-6                        # the control reproduction tolerance

# the wash arm (registered BEFORE compute): e227's lineage dose, the locked
# stream; the minimal grid the dispatch states ({t0,+2,+10,+50})
WASH_TAG, WASH_LR, WASH_SEED = "5L10902", 1e-3, 10902
WASH_CKS: tuple[int, ...] = (2, 10, 50) if not SMOKE else (1, 2)

# bar constants (frozen; registered additions — flagged in the docstring)
LIFT_BAR = 0.05                        # the VARIANT lift leg (>= +0.05)
GATE_BAR = 10                          # the VARIANT gate leg (>= 10/30)
LIFT_ABSENT = 0.0                      # the SCALE leg (lift <= 0.0)

FORM_KEYS = ("tmpl2", "mis2Z", "mis2M", "snip0Z", "snip0M")
BAT_KEYS = E227.BAT_KEYS if hasattr(E227, "BAT_KEYS") else [
    "fact", "near", "tmpl2", "mis2Z", "mis2M", "snip0Z", "snip0M", "ctrl"]

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "VARIANT-NOT-SCALE": "the lift at t=0 on g1bS5's peak root >= +0.05 "
            "(and/or the 2-shot Z-form gates >= 10/30) where g1bS7's "
            "reproduced read stays absent — the 10M absence was the "
            "consolidation variant; the faculty is present at 10M; the "
            "dissociation readjusts to a consolidation axis",
        "SCALE-CONFIRMED": "the lift absent on BOTH 10M roots (reproduced "
            "control intact) — the absence is scale in this instrument "
            "form; T211's table stands",
        "PARTIAL": "anything between — the reads verbatim (t0 lift, gate "
            "counts, the wash trajectory), no narrative inflation",
    },
    "registered_prediction": "VARIANT-NOT-SCALE (T211's own, THINKING.md "
        "T211 prediction (a))",
    "operationalizations": "see the script docstring (the ungated _all lift "
        "as the instrument; the e182 gate count over 30; 'absent' := "
        "lift <= 0.0 AND gates < 10; the partition VARIANT/SCALE/PARTIAL; "
        "the wash descriptive) — all REGISTERED ADDITIONS (the dispatch "
        "bars are qualitative)",
    "registration": "bars + partition + arm + constants frozen in this file "
        "and committed BEFORE any compute; no bar shopping",
}

deviations: list[str] = []


# --------------------------------------------------------------- utilities

def md5_of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def cpu_load_pct() -> int | None:
    """Best-effort load read (the load-politeness check; non-fatal)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).LoadPercentage"],
            capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip())
    except Exception:                                     # noqa: BLE001
        return None


def load_root(ck_name: str):
    """e227's G_ROOT10 load pattern VERBATIM: load, bit-diff vs file, md5."""
    ck = common.REPO / "runs" / "checkpoints" / ck_name
    net = G1.load_g1(ck)
    assert net.num_params() == E227.SCALE10_PARAMS, \
        f"{ck_name}: params {net.num_params()} != {E227.SCALE10_PARAMS}"
    theta = {k: v.detach().clone() for k, v in net.state_dict().items()}
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    raw_sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    md = max(float((theta[k].float() - raw_sd[k].float()).abs().max())
             for k in raw_sd)
    return net, theta, {"checkpoint": f"runs/checkpoints/{ck_name}",
                        "meta": raw.get("meta"), "md5": md5_of(ck),
                        "max_abs_diff_vs_file": md,
                        "params": net.num_params()}


def organism_battery(net, B: dict, ctrl_c_recs=None):
    """The t=0 battery on one organism, e227's 10M sequence VERBATIM:
    (a) the UNGATED full-30 form reads first (the addendum instrument),
    (b) then the per-organism gate screen (kept set may be empty),
    (c) then the gated-set reads + committed instruments + ctrl + ce_r."""
    zid = B["zid"]
    out = {}
    # (a) ungated full-set form reads — THE lift instrument
    full = {bk: list(B[bk]["ids"]) for bk in FORM_KEYS}
    for bk in FORM_KEYS:
        per = E227.probe_items(net, full[bk], B[bk]["ans"])
        out[bk + "_all"] = {"mean_p": float(np.mean([r["p"] for r in per])),
                            "per": per}
    # (b) the per-organism screen (e182's gate, e227's discipline)
    tmpl0 = out["tmpl2_all"]["per"]
    kept = [i for i, r in enumerate(tmpl0) if E227.gate_pass(r, zid)]
    if len(kept) > E227.CAP_ITEMS:
        kept = sorted(kept, key=lambda i: -tmpl0[i]["p"])[:E227.CAP_ITEMS]
    screen = {"tmpl2": {"n_probed": len(tmpl0), "n_passed": len(kept),
                        "reduced_flag": bool(len(kept) < E227.FLOOR_ITEMS),
                        "p0_range": [float(min(r["p"] for r in tmpl0)),
                                     float(max(r["p"] for r in tmpl0))]}}
    # (c) the gated-set reads (kept items, like-for-like across forms —
    # the same kept carrier indices in every form, e227's discipline)
    for bk in FORM_KEYS:
        ids_kept = [full[bk][i] for i in kept]
        per = E227.probe_items(net, ids_kept, B[bk]["ans"]) if ids_kept else []
        out[bk] = {"mean_p": (float(np.mean([r["p"] for r in per])
                               if per else float("nan"))),
                   "per": per, "kept_idx": kept}
    # committed instruments (read whole, no gate) — probe path (e227's t0)
    for k in ("fact", "near"):
        per = E227.probe_items(net, B[k]["ids"], B[k]["ans"])
        out[k] = {"mean_p": float(np.mean([r["p"] for r in per])), "per": per}
    # ctrl gating (per-item against its OWN answer) then the kept-set read
    kept_ctrl = [(c, r) for c, r in ctrl_c_recs
                 if E227.gate_pass(r, c["ans_id"])]
    bysub: dict[str, list] = {}
    for c, r in kept_ctrl:
        bysub.setdefault(c["sub"], []).append((c, r))
    kept_ctrl = []
    for sub in ("colon", "word", "cname", "sent"):
        kept_ctrl += bysub.get(sub, [])[:E227.CTRL_SUB_CAP]
    if len(kept_ctrl) > E227.CTRL_CAP:
        kept_ctrl = kept_ctrl[:E227.CTRL_CAP]
    per_ctrl = []
    for c, _ in kept_ctrl:
        r2 = E227.probe_items(net, [c["ids"]], c["ans_id"])[0]
        per_ctrl.append(r2)
    out["ctrl"] = {"mean_p": float(np.mean([r["p"] for r in per_ctrl])
                                   if per_ctrl else float("nan")),
                   "per": per_ctrl,
                   "items": [c["fact"] for c, _ in kept_ctrl],
                   "subs": [c["sub"] for c, _ in kept_ctrl]}
    screen["ctrl"] = {"n_passed": len(kept_ctrl),
                      "by_sub": {s: sum(1 for c, _ in kept_ctrl
                                        if c["sub"] == s)
                                 for s in ("colon", "word", "cname", "sent")}}
    return out, screen


# ------------------------------------------------------------------ main

def main():
    rd = run_dir(DIRNAME)
    log(f"E235 THE SECOND 10M ROOT'S LIFT (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": NAME,
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "question": ("e227: exemplar lift +0.185 at 2.74M, absent at 10M "
                     "(-0.019 on g1bS7's root). Is the 10M absence SCALE "
                     "or CONSOLIDATION VARIANT? The discriminating root: "
                     "g1bS5's peak (cons 10901, the original draw). If the "
                     "lift appears there -> variant; absent on both -> "
                     "scale (in this instrument form)"),
        "registration": ("bars frozen VERBATIM from the dispatch "
                         "(VARIANT-NOT-SCALE / SCALE-CONFIRMED / PARTIAL); "
                         "all numeric operationalizations are REGISTERED "
                         "ADDITIONS (flagged); arm + constants committed "
                         "before any compute; no bar shopping"),
        "registered_prediction": REGISTERED_PREDICTION,
        "builds_on": ["e227/T211 (the cross-scale dissociation + this "
                      "registration; EVERY instrument imported from "
                      "lab/e227_faculty_at_home.py)",
                      "g1bS5 (the peak root, cons 10901)",
                      "g1bS7 (the control root, cons 10903)",
                      "g1b (the wash stream, seed 10902)"],
        "whats_new": ["the second 10M root's t=0 lift + gate census",
                      "the g1bS7 control reproduction gate (the "
                      "instrument tie)",
                      "the g1bS5 lineage-dose wash lift trajectory at "
                      "{0,+2,+10,+50} (CPU)"],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    # ---------------- P0 runtime-bound references (never transcribed)
    m227 = json.loads((common.REPO / "runs" / "e227" / "metrics.json")
                      .read_text(encoding="utf-8"))
    s10 = m227["scale10"]
    mg5 = json.loads((common.REPO / "runs" / "g1bS5" / "metrics.json")
                     .read_text(encoding="utf-8"))
    mg7 = json.loads((common.REPO / "runs" / "g1bS7" / "metrics.json")
                     .read_text(encoding="utf-8"))
    ref5 = {"root_gm12": mg5["ckpt_inventory"]["g1bS5_root_m020"]
            ["root_gm12"],
            "cons_seed": mg5["ckpt_inventory"]["g1bS5_root_m020"]
            ["cons_seed"],
            "movement_rms": mg5["ckpt_inventory"]["g1bS5_root_m020"]
            ["movement_rms"],
            "source": "runs/g1bS5/metrics.json "
                      "ckpt_inventory.g1bS5_root_m020"}
    ref7 = {"root_gm12": mg7["ckpt_inventory"]["g1bS7_root_m020f"]
            ["root_gm12"],
            "cons_seed": mg7["ckpt_inventory"]["g1bS7_root_m020f"]
            ["cons_seed"],
            "movement_rms": mg7["ckpt_inventory"]["g1bS7_root_m020f"]
            ["movement_rms"],
            "source": "runs/g1bS7/metrics.json "
                      "ckpt_inventory.g1bS7_root_m020f"}
    ref_ctrl = {  # e227's committed scale10 block — THE control target
        "ruler": s10["root"]["gates"]["t0_ruler"],
        "md5": s10["root"]["gates"]["md5"],
        "near": s10["t0_reads"]["near"],
        "ctrl": s10["t0_reads"]["ctrl"],
        "all_reads": s10["t0_reads_all_ungated"],
        "lift_all": s10["t0_fewshot_lift_all"],
        "gates": s10["screening"]["tmpl2"]["n_passed"],
        "gates_probed": s10["screening"]["tmpl2"]["n_probed"],
        "ce_r": s10["root"]["ce_r_t0"],
        "source": "runs/e227/metrics.json scale10",
    }
    # the 2.74M lineage reference (the arm where the lift survived)
    j274 = json.loads((common.REPO / "runs" / "e227" / "journal_L10902.json")
                      .read_text(encoding="utf-8"))
    ref274_lift = {str(s["step"]):
                   s["reads"]["mis2M"]["mean_p"]
                   - s["reads"]["snip0M"]["mean_p"]
                   for s in j274["states"]}
    metrics["references"] = {
        "g1bS5_root": ref5, "g1bS7_root": ref7, "e227_scale10": ref_ctrl,
        "wash_274_L10902_lift": ref274_lift,
        "note_274": "runs/e227/journal_L10902.json — the 2.74M lineage arm "
                    "(lr 1e-3, seed 10902): lift {t0 +0.185, +2 −0.050, "
                    "+10 +0.314, +50 +0.154} — the survival shape this "
                    "cell's wash leg is read against",
    }
    log(f"refs: g1bS5 root_gm12 {ref5['root_gm12']:.6f} (cons "
        f"{ref5['cons_seed']}, mov {ref5['movement_rms']}) | g1bS7 "
        f"{ref7['root_gm12']:.6f} (cons {ref7['cons_seed']}) | e227 "
        f"lift_all {ref_ctrl['lift_all']:+.6f}, gates "
        f"{ref_ctrl['gates']}/{ref_ctrl['gates_probed']}")
    write_metrics("PARTIAL: references bound")

    # load-politeness (the dispatch's check)
    lp = cpu_load_pct()
    metrics["load_politeness"] = {
        "cpu_load_pct_at_start": lp,
        "threads": torch.get_num_threads(),
        "gpu": "PARKED before the wash (e233 owns the GPU lane; e235 is "
               "CPU-only by dispatch)",
        "neighbors": "e233 (GPU lane, ARM-C under thermal guard per "
                     "STATE.json) + e234 (CPU-heavy) — one wash, no "
                     "parallelism",
    }
    log(f"load check: CPU {lp}% — proceeding (CPU-only, 8 threads)")

    # ---------------- P1 the corpus + protocol rebuild (e227 P1 VERBATIM)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    assert train_text.count("ZEPH") == 0
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
    G_CORP = {"corpus_zeph": train_text.count("ZEPH"), "install_mix": mix,
              "n_install": len(install_occ), "n_held": len(held_occ),
              "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41}
                           and len(held_occ) == 30)}
    metrics["gates"] = {"G_CORP": G_CORP}
    assert G_CORP["pass"], f"protocol drift {mix}"
    log(f"G_CORP: install60 {mix}, held30: PASS")

    # ---------------- P2 the batteries (e227's build_batteries, imported)
    B = E227.build_batteries(corpus, train_text, val_text, install_occ,
                             held_occ, zid)[0]
    B["zid"] = zid
    ctrl_c = B["ctrl"]["cands"]

    # ---------------- P3 cell (1): g1bS5's peak root, the t=0 battery
    net5, theta5, prov5 = load_root(ROOT5_CK)
    gm12_ids = torch.stack(B["fact"]["ids"])
    ruler5 = G1.battery_cell(net5, gm12_ids, zid)
    G_ROOT5 = {**prov5, "ruler_ref": ref5["root_gm12"], "tol": G_ROOT_TOL,
               "t0_ruler": float(ruler5["mean_pz"]),
               "source": ref5["source"]}
    G_ROOT5["pass"] = bool(prov5["max_abs_diff_vs_file"] == 0.0
                           and abs(G_ROOT5["t0_ruler"]
                                   - ref5["root_gm12"]) <= G_ROOT_TOL)
    metrics["gates"]["G_ROOT5"] = G_ROOT5
    log(f"G_ROOT5: t0 ruler {G_ROOT5['t0_ruler']:.10f} vs ref "
        f"{ref5['root_gm12']:.10f} (dp "
        f"{abs(G_ROOT5['t0_ruler'] - ref5['root_gm12']):.2e}) md5 "
        f"{prov5['md5'][:12]} -> "
        f"{'PASS' if G_ROOT5['pass'] else 'FAIL'}")
    assert G_ROOT5["pass"] or SMOKE, "G_ROOT5 FAILED"

    # per-organism ctrl candidate records (the gate input; e227's sequence)
    ctrl_c_recs5 = [(c, E227.probe_items(net5, [c["ids"]], c["ans_id"])[0])
                    for c in ctrl_c]
    t0_5, screen5 = organism_battery(net5, B, ctrl_c_recs5)
    r_eval_xy = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    ce5 = G1.ce_fixed_cpu(net5, *r_eval_xy)
    lift5_all = (t0_5["mis2M_all"]["mean_p"]
                 - t0_5["snip0M_all"]["mean_p"])
    gates5 = screen5["tmpl2"]["n_passed"]
    metrics["g1bS5"] = {
        "root": {"provenance": prov5, "ruler": G_ROOT5, "ce_r_t0": ce5,
                 "ref": ref5},
        "screening": screen5,
        "t0_reads": {k: (t0_5[k]["mean_p"] if k in t0_5 else None)
                     for k in BAT_KEYS},
        "t0_reads_all_ungated": {bk: t0_5[bk + "_all"]["mean_p"]
                                 for bk in FORM_KEYS},
        "t0_fewshot_lift_all": lift5_all,
        "t0_fewshot_lift_gated": (t0_5["mis2M"]["mean_p"]
                                  - t0_5["snip0M"]["mean_p"]
                                  if t0_5["mis2M"]["per"] else None),
        "gate_count_2shot_Z": gates5,
        "mis2_split": {"pZ_under_MIRABEL": t0_5["mis2Z_all"]["mean_p"],
                       "pM_following": t0_5["mis2M_all"]["mean_p"],
                       "tmpl2Z": t0_5["tmpl2_all"]["mean_p"],
                       "snip0Z": t0_5["snip0Z_all"]["mean_p"]},
    }
    log(f"g1bS5 t=0: fact {t0_5['fact']['mean_p']:.4f} | near "
        f"{t0_5['near']['mean_p']:.4f} | tmpl2_all "
        f"{t0_5['tmpl2_all']['mean_p']:.4f} | mis2Z_all "
        f"{t0_5['mis2Z_all']['mean_p']:.4f} | mis2M_all "
        f"{t0_5['mis2M_all']['mean_p']:.4f} | snip0Z_all "
        f"{t0_5['snip0Z_all']['mean_p']:.4f} | snip0M_all "
        f"{t0_5['snip0M_all']['mean_p']:.4f} | ctrl "
        f"{t0_5['ctrl']['mean_p']:.4f} | gates {gates5}/30 | "
        f"LIFT_all {lift5_all:+.4f} | CE_R {ce5:.4f}")
    write_metrics("PARTIAL: g1bS5 t=0 battery done; control pending")

    # ---------------- P4 cell (2): the g1bS7 control reproduction (GATE)
    net7, theta7, prov7 = load_root(ROOT7_CK)
    ruler7 = G1.battery_cell(net7, gm12_ids, zid)
    ctrl_c_recs7 = [(c, E227.probe_items(net7, [c["ids"]], c["ans_id"])[0])
                    for c in ctrl_c]
    t0_7, screen7 = organism_battery(net7, B, ctrl_c_recs7)
    ce7 = G1.ce_fixed_cpu(net7, *r_eval_xy)
    lift7_all = (t0_7["mis2M_all"]["mean_p"]
                 - t0_7["snip0M_all"]["mean_p"])
    gates7 = screen7["tmpl2"]["n_passed"]
    cmp_rows = [
        ("ruler_battery_cell", float(ruler7["mean_pz"]), ref_ctrl["ruler"]),
        ("near", t0_7["near"]["mean_p"], ref_ctrl["near"]),
        ("ctrl", t0_7["ctrl"]["mean_p"], ref_ctrl["ctrl"]),
        ("tmpl2_all", t0_7["tmpl2_all"]["mean_p"],
         ref_ctrl["all_reads"]["tmpl2"]),
        ("mis2Z_all", t0_7["mis2Z_all"]["mean_p"],
         ref_ctrl["all_reads"]["mis2Z"]),
        ("mis2M_all", t0_7["mis2M_all"]["mean_p"],
         ref_ctrl["all_reads"]["mis2M"]),
        ("snip0Z_all", t0_7["snip0Z_all"]["mean_p"],
         ref_ctrl["all_reads"]["snip0Z"]),
        ("snip0M_all", t0_7["snip0M_all"]["mean_p"],
         ref_ctrl["all_reads"]["snip0M"]),
        ("lift_all", lift7_all, ref_ctrl["lift_all"]),
        ("ce_r", ce7, ref_ctrl["ce_r"]),
    ]
    G_CTRL = {"what": "g1bS7 reproduced vs e227's committed scale10 "
                      "(runs/e227/metrics.json)",
              "tol": CTRL_TOL,
              "md5_got": prov7["md5"], "md5_ref": ref_ctrl["md5"],
              "md5_match": prov7["md5"] == ref_ctrl["md5"],
              "bit_exact_load": prov7["max_abs_diff_vs_file"] == 0.0,
              "gates_got": gates7, "gates_ref": ref_ctrl["gates"],
              "gates_match": gates7 == ref_ctrl["gates"],
              "rows": [{"read": n, "got": g, "ref": r,
                        "dp": abs(g - r),
                        "within_tol": abs(g - r) <= CTRL_TOL}
                       for n, g, r in cmp_rows]}
    G_CTRL["max_dp"] = max(r["dp"] for r in G_CTRL["rows"])
    G_CTRL["pass"] = bool(G_CTRL["md5_match"] and G_CTRL["bit_exact_load"]
                          and G_CTRL["gates_match"]
                          and G_CTRL["max_dp"] <= CTRL_TOL)
    metrics["gates"]["G_CTRL"] = G_CTRL
    metrics["g1bS7_control"] = {
        "root": {"provenance": prov7,
                 "t0_ruler": float(ruler7["mean_pz"]), "ce_r_t0": ce7,
                 "ref": ref7},
        "screening": screen7,
        "t0_reads": {k: t0_7[k]["mean_p"] for k in BAT_KEYS},
        "t0_reads_all_ungated": {bk: t0_7[bk + "_all"]["mean_p"]
                                 for bk in FORM_KEYS},
        "t0_fewshot_lift_all": lift7_all,
        "gate_count_2shot_Z": gates7,
    }
    log(f"G_CTRL: max dp {G_CTRL['max_dp']:.2e} (tol {CTRL_TOL:g}), "
        f"md5 {'match' if G_CTRL['md5_match'] else 'MISMATCH'}, gates "
        f"{gates7} vs {ref_ctrl['gates']} -> "
        f"{'PASS' if G_CTRL['pass'] else 'FAIL'}")
    for r in G_CTRL["rows"]:
        log(f"  {r['read']:16s} got {r['got']:+.10f} ref {r['ref']:+.10f} "
            f"dp {r['dp']:.2e}")
    write_metrics("PARTIAL: g1bS7 control reproduced; "
                  + ("GATE PASS" if G_CTRL["pass"] else "GATE FAIL"))
    if not G_CTRL["pass"]:
        if not SMOKE:
            metrics["STOPPED"] = ("G_CTRL FAILED — the g1bS7 control does "
                                  "NOT reproduce e227's committed numbers; "
                                  "stopped per the dispatch (no wash, no "
                                  "adjudication); report and do not shop")
            write_metrics("STOPPED: G_CTRL FAILED (report; no adjudication)")
            log("G_CTRL FAILED -> STOP (dispatch: stop and report)")
            return 1
        deviations.append("SMOKE: G_CTRL failed in smoke (not adjudicated)")

    # ---------------- P5 cell (3): the wash on g1bS5 (CPU, lineage dose)
    G1.GPU_PARKED = True
    G1.PARK_REASON = ("E235: CPU-only dispatch (e233 owns the GPU lane; "
                      "e234 the CPU lane — load-polite)")
    metrics["gates"]["G_CPU"] = {
        "gpu_parked_before_wash": True,
        "threads": torch.get_num_threads(),
        "cpu_load_pct_at_start": lp,
        "pass": True,
    }
    # the neutral anchor bank (e170's construction VERBATIM — e227's P4)
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
    assert len(n_starts) == 16
    anchor_neutral = torch.stack([train_ids[s: s + G1.BLOCK]
                                  for s in n_starts])
    host_positions = [p for p in E43.find_occ(train_text, G1.HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, G1.HOSTS[1])]
    jc = sum(1 for s in n_starts
             if any(s <= p < s + G1.BLOCK + 1 for p in host_positions))
    whost = sum(1 for s in n_starts
                if any(f in train_text[s: s + G1.BLOCK + 1]
                       for f in G1.HOSTS))
    G_ANCH = {"seed": G1.E170_ANCHOR_SEED, "n": 16, "starts": n_starts,
              "tries": tries, "rejections": rejections,
              "windows_with_host_content": whost, "junctions_covered": jc,
              "pass": bool(whost == 0 and jc == 0
                           and anchor_neutral.shape == (16, G1.BLOCK))}
    metrics["gates"]["G_ANCHOR"] = G_ANCH
    assert G_ANCH["pass"], f"G_ANCHOR FAILED: {G_ANCH}"
    log(f"G_ANCHOR: neutral bank 16x{G1.BLOCK} ({rejections} rejections/"
        f"{tries} tries), host 0/16, junctions 0/16: PASS")

    g0_ids = torch.stack([corpus.encode(train_text[p - G1.PRE: p])
                          for p, _ in install_occ])
    kept5 = t0_5["tmpl2"]["kept_idx"]
    B5 = {bk: {"ids": [B[bk]["ids"][i] for i in kept5],
               "ans": B[bk]["ans"]} for bk in FORM_KEYS}
    # e227's ctrl capping (document order, per-sub cap, pooled cap)
    keptc5 = [(c, r_) for c, r_ in ctrl_c_recs5
              if E227.gate_pass(r_, c["ans_id"])]
    _byb: dict[str, list] = {}
    for c, r_ in keptc5:
        _byb.setdefault(c["sub"], []).append((c, r_))
    keptc5 = []
    for sub in ("colon", "word", "cname", "sent"):
        keptc5 += _byb.get(sub, [])[:E227.CTRL_SUB_CAP]
    if len(keptc5) > E227.CTRL_CAP:
        keptc5 = keptc5[:E227.CTRL_CAP]
    ctrl_ids5 = [c["ids"] for c, _ in keptc5]
    ctrl_ans5 = [c["ans_id"] for c, _ in keptc5]

    def read_state(sd: dict) -> dict:
        """e227's read_state10 VERBATIM shape: gated keys + _all co-reports
        + ctrl + ce_r (the instrument, not retyped — the probe path)."""
        net = G1.evl_load(sd)
        out = {}
        for bk in FORM_KEYS:
            ids_g = B5[bk]["ids"]
            per = (E227.probe_items(net, ids_g, B5[bk]["ans"])
                   if ids_g else [])
            out[bk] = {"mean_p": (float(np.mean([r["p"] for r in per])
                                   if per else float("nan")))}
            per_a = E227.probe_items(net, list(B[bk]["ids"]), B[bk]["ans"])
            out[bk + "_all"] = {
                "mean_p": float(np.mean([r["p"] for r in per_a]))}
        for k in ("fact", "near"):
            per = E227.probe_items(net, B[k]["ids"], B[k]["ans"])
            out[k] = {"mean_p": float(np.mean([r["p"] for r in per]))}
        per = [E227.probe_items(net, [i_], a_)[0]
               for i_, a_ in zip(ctrl_ids5, ctrl_ans5)]
        out["ctrl"] = {"mean_p": float(np.mean([r["p"] for r in per])
                                        if per else float("nan"))}
        out["ce_r"] = G1.ce_fixed_cpu(net, *r_eval_xy)
        return out

    jp = rd / f"journal_{WASH_TAG}.json"
    net0_wash = G1.evl_load(theta5)
    log(f"WASH {WASH_TAG}: lr {WASH_LR:g} seed {WASH_SEED} ckpts "
        f"{list(WASH_CKS)} (g1_wash VERBATIM, CPU PARKED, batch 32)")
    t_w = time.time()
    res = G1.g1_wash(WASH_TAG, net0_wash, anchor_neutral, train_ids, itos,
                     r_eval_xy, gm12_ids, g0_ids, zid,
                     target_mode="true", ckpt_steps=WASH_CKS,
                     lr=WASH_LR, seed=WASH_SEED)
    log(f"wash ran {res['steps_ran']} steps on {res.get('device')} in "
        f"{time.time() - t_w:.0f}s (zeph violations "
        f"{res['zeph_violations']})")
    states = [{"step": 0,
               "reads": {**{k: {"mean_p": t0_5[k]["mean_p"]}
                           for k in ("fact", "near", "ctrl")},
                        **{bk: {"mean_p": t0_5[bk]["mean_p"]}
                           for bk in FORM_KEYS},
                        **{bk + "_all": {"mean_p": t0_5[bk + "_all"]["mean_p"]}
                           for bk in FORM_KEYS}},
               "ce_r": ce5}]
    for s_ in sorted(res["sds"]):
        rd_ = read_state(res["sds"][s_])
        states.append({"step": s_,
                       "reads": {k: v for k, v in rd_.items()
                                 if k != "ce_r"},
                       "ce_r": rd_["ce_r"],
                       "g1_wash_row": {kk: next(
                           (r[kk] for r in res["traj"] if r["step"] == s_),
                           None) for kk in ("g_m12_mean_pz", "g0_mean_pz",
                                            "ce_r", "cum_disp")}})
        log(f"  {WASH_TAG} +{s_:4d}: fact {rd_['fact']['mean_p']:.4f} "
            f"near {rd_['near']['mean_p']:.4f} mis2M_all "
            f"{rd_['mis2M_all']['mean_p']:.4f} snip0M_all "
            f"{rd_['snip0M_all']['mean_p']:.4f} LIFT_all "
            f"{rd_['mis2M_all']['mean_p'] - rd_['snip0M_all']['mean_p']:+.4f}"
            f" | CE_R {rd_['ce_r']:.4f}")
        jp.write_text(json.dumps({"tag": WASH_TAG, "lr": WASH_LR,
                                  "seed": WASH_SEED, "states": states},
                                 indent=1, default=float),
                      encoding="utf-8")
        metrics["wash_arm"] = {
            "tag": WASH_TAG, "lr": WASH_LR, "seed": WASH_SEED,
            "device": res.get("device"), "steps_ran": res["steps_ran"],
            "grid_registered": list(WASH_CKS),
            "states": [s["step"] for s in states],
            "reads": {str(s["step"]):
                      {k: v["mean_p"] for k, v in s["reads"].items()}
                      for s in states},
            "lift_all": {str(s["step"]):
                         (s["reads"]["mis2M_all"]["mean_p"]
                          - s["reads"]["snip0M_all"]["mean_p"])
                         for s in states},
            "ce_r": {str(s["step"]): s["ce_r"] for s in states},
            "zeph_violations": res["zeph_violations"],
        }
        write_metrics(f"PARTIAL: wash through +{s_}")
    del res

    # ---------------- P6 adjudication (frozen) + the wash read
    lift_traj = {str(s["step"]): (s["reads"]["mis2M_all"]["mean_p"]
                                  - s["reads"]["snip0M_all"]["mean_p"])
                 for s in states}
    wash_read = {
        "lift_all_trajectory": lift_traj,
        "reference_274_L10902": ref274_lift,
        "survival_read": ("descriptive (registered): the lift's trajectory "
                          "verbatim vs the 2.74M shape {t0 +0.185, dip "
                          "−0.050 at +2, return +0.314 at +10, +0.154 at "
                          "+50}; no wash bar"),
        "max_lift_post": (max(v for k, v in lift_traj.items() if k != "0")
                          if any(k != "0" for k in lift_traj) else None),
        "lift_ge_t0_any_post": any(v >= lift_traj["0"]
                                   for k, v in lift_traj.items() if k != "0"),
    }
    variant = bool(lift5_all >= LIFT_BAR or gates5 >= GATE_BAR)
    scale = bool((not variant) and lift5_all <= LIFT_ABSENT
                 and gates5 < GATE_BAR)
    partial = bool((not variant) and (not scale))
    ctrl_absent = bool(G_CTRL["pass"] and lift7_all < LIFT_BAR
                       and gates7 < GATE_BAR)   # "stays absent" clause
    if not G_CTRL["pass"]:
        verdict, clause = "STOPPED (G_CTRL FAILED)", (
            "the control did not reproduce; stopped per dispatch")
    elif SMOKE:
        verdict, clause = "SMOKE (nothing adjudicated)", "smoke run"
    elif variant and ctrl_absent:
        verdict = "VARIANT-NOT-SCALE"
        clause = (f"the lift at t=0 on g1bS5's peak root {lift5_all:+.4f} "
                  f"(bar >= +{LIFT_BAR}) and/or the 2-shot Z-form gates "
                  f"{gates5}/30 (bar >= {GATE_BAR}/30) where g1bS7's "
                  f"reproduced read stays absent ({lift7_all:+.4f}, "
                  f"{gates7}/30) — the 10M absence was the consolidation "
                  "variant; the faculty is present at 10M; the "
                  "dissociation readjusts to a consolidation axis")
    elif scale:
        verdict = "SCALE-CONFIRMED"
        clause = (f"the lift absent on BOTH 10M roots (g1bS5 "
                  f"{lift5_all:+.4f}, gates {gates5}/30; g1bS7 reproduced "
                  f"{lift7_all:+.4f}, gates {gates7}/30; control intact "
                  f"max dp {G_CTRL['max_dp']:.1e}) — the absence is scale "
                  "in this instrument form; T211's table stands")
    else:
        verdict = "PARTIAL"
        clause = (f"anything between — the reads verbatim: g1bS5 lift "
                  f"{lift5_all:+.4f} (0 < lift < +{LIFT_BAR}), gates "
                  f"{gates5}/30; g1bS7 reproduced {lift7_all:+.4f}, "
                  f"{gates7}/30; the wash trajectory "
                  f"{ {k: round(v, 4) for k, v in lift_traj.items()} } — "
                  "no narrative inflation")
    adj = {
        "bars": REGISTERED_PREDICTION["bars_verbatim"],
        "bar_constants": {"LIFT_BAR": LIFT_BAR, "GATE_BAR": GATE_BAR,
                          "LIFT_ABSENT": LIFT_ABSENT, "ctrl_tol": CTRL_TOL},
        "reads": {"lift5_t0_all": lift5_all, "gates5": gates5,
                  "lift7_t0_all_reproduced": lift7_all,
                  "gates7": gates7,
                  "control_pass": G_CTRL["pass"],
                  "ctrl_stays_absent": ctrl_absent,
                  "registered_prediction": "VARIANT-NOT-SCALE"},
        "verdict": verdict, "clause": clause,
        "wash_read": wash_read,
    }
    metrics["adjudication"] = adj
    log("=" * 78)
    log(f"E235 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  wash lift_all trajectory: "
        f"{ {k: round(v, 4) for k, v in lift_traj.items()} } (2.74M ref: "
        f"{ {k: round(v, 4) for k, v in ref274_lift.items()} })")
    write_metrics("PARTIAL: adjudicated; plot pending")

    # ---------------- P7 the figure
    pngs = []
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0))
    cols = {"fact": "tab:red", "near": "tab:green", "tmpl2_all": "tab:purple",
            "mis2Z_all": "violet", "mis2M_all": "tab:orange",
            "snip0Z_all": "gray", "snip0M_all": "pink", "ctrl": "tab:blue"}

    ax = axes[0, 0]
    steps5 = [s["step"] for s in states]
    ax.plot(steps5, [lift_traj[str(s)] for s in steps5], "o-", ms=7,
            lw=2.0, color="tab:cyan",
            label=f"g1bS5 peak root wash lift (lr {WASH_LR:g}, seed "
                  f"{WASH_SEED})")
    steps274 = sorted(int(k) for k in ref274_lift)
    ax.plot(steps274, [ref274_lift[str(s)] for s in steps274], "s--", ms=5,
            lw=1.4, color="tab:green", alpha=0.8,
            label="2.74M L10902 (e227 committed — the survival shape)")
    ax.axhline(LIFT_BAR, color="darkred", ls="--", lw=1.2,
               label=f"the VARIANT bar lift >= +{LIFT_BAR}")
    ax.axhline(0.0, color="gray", ls=":", lw=0.9)
    ax.scatter([0], [lift7_all], marker="x", s=90, color="tab:red",
               zorder=5, label=f"g1bS7 reproduced t0 ({lift7_all:+.4f})")
    ax.set_xlabel("wash steps (the lineage dose)")
    ax.set_ylabel("exemplar lift (p_M 2shot − p_M 0shot, ungated)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="best")
    ax.set_title("the lift's wash trajectory: the second 10M root vs 2.74M",
                 fontsize=10)

    ax = axes[0, 1]
    names = ["2.74M (e227)", "10M g1bS5\n(this cell)", "10M g1bS7\n(control)"]
    vals = [m227["t0_fewshot_lift"]["lift"], lift5_all, lift7_all]
    bars_ = ax.bar(names, vals, color=["tab:green", "tab:cyan", "tab:red"],
                   alpha=0.85)
    for b, v, g in zip(bars_, vals,
                       [m227["screening"]["tmpl2"]["n_passed"], gates5,
                        gates7]):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.012 if v >= 0 else -0.03),
                f"{v:+.4f}\ngates {g}/30", ha="center", fontsize=8.5,
                family="monospace")
    ax.axhline(LIFT_BAR, color="darkred", ls="--", lw=1.2,
               label=f"the VARIANT bar +{LIFT_BAR}")
    ax.axhline(0.0, color="gray", ls=":", lw=0.9)
    ax.set_ylabel("t=0 exemplar lift (ungated _all)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    ax.set_title("the t=0 lift across organisms (both 10M roots shown)",
                 fontsize=10)

    ax = axes[1, 0]
    st0 = states[0]["reads"]
    for k in ("fact", "near", "tmpl2_all", "mis2Z_all", "mis2M_all",
              "snip0Z_all", "snip0M_all", "ctrl"):
        r0 = st0[k]["mean_p"]
        ax.plot(steps5, [s["reads"][k]["mean_p"] / r0 for s in states],
                "o-", ms=5, lw=1.8, color=cols[k], label=k)
    axr = ax.twinx()
    axr.plot(steps5, [s["ce_r"] for s in states], "s:", ms=4, lw=1.2,
             color="seagreen", alpha=0.8)
    axr.set_ylabel("CE_R (dotted, right)", fontsize=8, color="seagreen")
    ax.set_xlabel(f"wash steps (g1bS5: lr {WASH_LR:g}, seed {WASH_SEED})")
    ax.set_ylabel("retention R(s)/R(0)")
    ax.set_ylim(-0.05, 3.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    ax.set_title("the g1bS5 wash context (retention; _all ungated forms)",
                 fontsize=10)

    ax = axes[1, 1]
    ax.axis("off")
    y = 0.98
    ax.text(0.02, y, f"VERDICT: {verdict}", fontsize=11, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.05
    for wd in textwrap.wrap(clause, width=96, break_long_words=False):
        ax.text(0.02, y, wd, fontsize=7.2, va="top", family="monospace")
        y -= 0.022
    y -= 0.012
    ax.text(0.02, y, "gates:", fontsize=8.5, va="top", family="monospace",
            weight="bold")
    y -= 0.024
    for gname, gv in metrics["gates"].items():
        pv = gv.get("pass")
        ax.text(0.02, y, f"  {gname:10s} "
                f"{'PASS' if pv else ('FAIL' if pv is False else 'n/a')}",
                fontsize=7.4, va="top", family="monospace")
        y -= 0.02
    y -= 0.01
    ax.text(0.02, y, f"roots: g1bS5 cons {ref5['cons_seed']} (ruler "
            f"{ref5['root_gm12']:.4f}) vs g1bS7 cons {ref7['cons_seed']} "
            f"(ruler {ref7['root_gm12']:.4f}); same movement "
            f"{ref5['movement_rms']} rms, same protocol — the DRAW is the "
            f"treatment", fontsize=7.2, va="top", family="monospace")
    y -= 0.026
    ax.text(0.02, y, f"wash lift_all: "
            f"{ {k: round(v, 4) for k, v in lift_traj.items()} } vs 2.74M "
            f"{ {k: round(v, 4) for k, v in ref274_lift.items()} }",
            fontsize=7.2, va="top", family="monospace")

    fig.suptitle("E235 — THE SECOND 10M ROOT'S LIFT: is the 10M absence "
                 "scale or consolidation variant?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p1 = rd / "e235_second_10m_lift.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    pngs.append(p1)

    metrics["honesty_reflex"] = {
        "n_counts": ("n=1 organism per root; the wash n=1 draw (the LOCKED "
                     "lineage stream 10902 — the same stream e227 read at "
                     "both scales, stream-matched); batteries: fact 60 / "
                     "near 30 / forms 30 ungated / ctrl val-carried"),
        "root_difference": ("the two roots differ in fact strength (g1bS5 "
                            "ruler 0.768 vs g1bS7 0.935) — the roots' own "
                            "difference at the same movement/lr protocol; "
                            "the consolidation DRAW is the treatment; "
                            "co-reported, not controlled"),
        "lift_instrument": ("an internal 2-shot-minus-0-shot difference on "
                            "IDENTICAL carriers (MIRABEL, 0 corpus "
                            "occurrences) — the fact channel does not "
                            "predict it; the rulers are co-reports"),
        "form_bundle": ("tmpl2 changes length + exemplars + separators "
                        "together; snip0 is the length-matched control "
                        "(e227's disclosure carries)"),
        "cpu_only": ("GPU PARKED before the wash (e233 owns the GPU lane); "
                     "probes + wash CPU fp32, 8 threads; load logged "
                     f"({lp}% at start; e234 the CPU neighbor)"),
        "nothing_guaranteed": ("a control PASS proves the instrument, not "
                               "the finding; the openness is the point"),
    }
    metrics["compute"] = {
        "wash": f"{WASH_TAG}: lr {WASH_LR:g} seed {WASH_SEED} steps "
                f"{max(WASH_CKS)} (CPU, g1_wash VERBATIM)",
        "instrument": "E227.build_batteries / E227.probe_items / "
                      "E227.gate_pass + G1.g1_wash / G1.battery_cell / "
                      "G1.evl_load — imported VERBATIM (do not retype)",
        "run_log": str(LOG_PATH),
    }
    metrics["trims"] = list(G1.trims) if hasattr(G1, "trims") else []
    metrics["deviations"] = deviations
    write_metrics("DONE" if not SMOKE else "SMOKE DONE")
    log(f"outputs: {rd / 'metrics.json'}, {pngs}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
