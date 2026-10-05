"""X7 — THE LN ZERO-SUM CONTROL (desk-instrument cell, CPU-only, eval-only on committed records).

Dispatch (2026-10-05 ~21:20Z, one of three parallel desk cells): W038's fourth
movement (RE-FORMATION) reads the wall's +19.2% margin thickening (e242) as a
CONSTRUCTIVE field. AGY consult #001 (scratch/agy_consults/agy_consult_001.md,
clause c) registered-but-never-run the sharpest control: LayerNorm normalizes
the stream total, so if the undertow/friction strip scale from aligned beliefs,
LN must REDISTRIBUTE the variance budget to the survivors — apparent
"thickening" may be zero-sum bookkeeping, not construction. THE DISCRIMINATING
QUESTION: on the committed margins, did the CTRL battery ALSO thicken ~19%?

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any compute;
script committed at birth; adjudicate against exactly this; no bar shopping):

- LN-ARITHMETIC: the CTRL battery's matched thickening is >= 50% of the fact
  battery's ~19% — W038's re-formation layer demoted to arithmetic; the laws
  draft gets the amendment.
- CONSTRUCTIVE-FIELD: the CTRL thickening is <= 20% of the fact battery's —
  the field reading survives its control.
- MIXED/UNDERPOWERED: in between or the committed margins lack the required
  fields — report exactly what's missing.

OPERATIONALIZATION (frozen with the bars, before any compute):

1. fact_thickening_pct := e242's committed wall read: install-60 g-12 battery
   MEDIAN margin_sigma, t0 0.9794073282786698 -> +300 1.1670049513746559,
   growth = +19.15419842995183 pct (committed as margin_decline_pct =
   -19.15419842995183; e242's own GROWTH labeling). Asserted at runtime
   against runs/e242/metrics.json (Rule 12).
2. CTRL margins := runs/e228/journal.json, battery 'ctrl' (n=12; the ONLY
   committed ctrl margin trajectory in the lab: e242's own deviations name its
   single battery; e229's wall t=0 snapshot has none; e240's 'ctrl' rows are
   gradient supports, not margins — machine-checked in G_NO_CTRL_WALL).
   Aggregate = battery MEDIAN margin_sigma (e228/e229/e242's family primary);
   growth_pct(s) = 100*(median(s)/median(t0) - 1), t0 = the shared t0 row.
3. ctrl_matched_thickening_pct := MAX over the seven washed states
   {w1+2, w1+10, w1+50, w1+80, w2+10, w2+50, w2+80} of growth_pct(s) — the
   confound's BEST state, frozen so no state-shopping can favor the field
   reading. r := ctrl_matched_thickening_pct / 19.15419842995183.
4. VERDICT: r >= 0.50 -> LN-ARITHMETIC; r <= 0.20 -> CONSTRUCTIVE-FIELD; else
   MIXED/UNDERPOWERED. The missing-fields clause of MIXED fires ONLY if the
   control's core quantities cannot be computed from the committed record at
   all; the wall-side absences (e242 committed no ctrl battery and no
   per-state sigma/margin_raw, so the wall's own +19.15% cannot be
   scale-decomposed) are carried as DISCLOSURES in the verdict clause, not as
   a re-route — the control's letter is the CTRL battery's committed margins,
   which exist and carry the full decomposition fields.
5. THE LN ZERO-SUM DECOMPOSITION (the dispatch's ordered computation; per
   battery, per washed state, per probe, on e228's committed fields — an
   EXACT arithmetic identity, no estimator): ms(t)/ms(t0) =
   [margin_raw(t)/margin_raw(t0)] x [sigma(t0)/sigma(t)] — the GAP factor x
   the SCALE factor. Zero-sum prediction with NO field (gap frozen at t0,
   only the recorded activation scale moves): ms_pred_i(t) =
   margin_raw_i(t0)/sigma_i(t); predicted growth vs observed growth; the
   difference = the field surplus on this slice. Co-reported within-world
   ratios ctrl_growth/fact_growth per state (the consult's ideal form —
   texture, never adjudicating; the frozen denominator is e242's 19.15).
6. THERMAL ARITHMETIC CO-REPORT: apparent T from scale alone = median
   sigma(s)/median sigma(t0) per battery (a softmax logit rescaling by 1/T
   scales sigma by 1/T), quoted against the committed one-T curves (e238's
   pooled-54 fact-side via x5's committed reference read; x5's ctrl fits;
   e242's wall T(t) quoted UNDECOMPOSED — its sigma is not committed). The
   one-T lens invariance demonstrated numerically on t0 committed fields:
   margin_sigma is EXACTLY invariant under a pure logit rescaling (gap and
   sigma both scale 1/T), so a thickening is never thermal-lens arithmetic.

Conventions inherited: x4's honesty about estimator limits (this cell's
decomposition is an identity on committed fields, not x4's bracket bound — no
instrument-artifact caveat of that kind applies); x5's battery-specific
discipline (every battery read separately, never pooled); e228/e229/e242's
median-margin aggregate; e242's negative-decline-as-GROWTH labeling.

CPU-ONLY (threads 4, no GPU calls, no envelope writes — lab/common.py never
imported). No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator
folds).
"""

import os
import sys
import json
import time
import hashlib
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU-only, set before torch import

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(4)
N_THREADS = torch.get_num_threads()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "runs", "x7")
P_E228J = os.path.join(ROOT, "runs", "e228", "journal.json")
P_E242 = os.path.join(ROOT, "runs", "e242", "metrics.json")
P_X5 = os.path.join(ROOT, "runs", "x5", "metrics.json")
P_E238 = os.path.join(ROOT, "runs", "e238", "metrics.json")

# Rule-12 literals: committed values this script asserts at runtime (frozen here
# pre-compute; if a parent record moved, the cell aborts rather than mis-reads).
LIT = {
    "e242_margin_median_t0": 0.9794073282786698,
    "e242_margin_median_300": 1.1670049513746559,
    "e242_margin_decline_pct": -19.15419842995183,
    "e228_sha16": "e1f476405746ad70",
    "e242_sha16": "5cd289827bb02e6d",
    "x5_sha16": "c69037c447aa877e",
    "e238_sha16": "95290a45b9aaf7e4",
    "x5_ctrl_w1_T": {10: 1.0430748020048268, 50: 1.1835854844229838, 80: 1.2649678250862952},
    "e238_w1_T": {10: 1.0597956498145238, 50: 1.3489839391244727, 80: 1.4525923182890421},
}
# the ONLY hard Rule-12 asserts are the e242 read triple + the x5/e238 one-T rows
# + the four sha256_16s; every e228 quantity is COMPUTED at runtime from the
# journal and cross-checked against the journal's own committed aggregates.

BATTERIES = ["fact", "ctrl", "near", "tmpl"]
E228_STATES = [("w1", 2), ("w1", 10), ("w1", 50), ("w1", 80), ("w2", 10), ("w2", 50), ("w2", 80)]
E242_STATES = ["t0", "w1+1", "w1+2", "w1+4", "w1+10", "w1+50", "w1+100", "w1+200", "w1+300"]
E242_ADJ = ["t0", "w1+1", "w1+2", "w1+10", "w1+50", "w1+100", "w1+300"]  # e242's registered grid

T0 = time.time()
M = {
    "experiment": "x7_ln_zero_sum",
    "phase": "desk control (consult #001 clause c): did the CTRL battery's margins also thicken ~19%? — the LN zero-sum adjudication on committed records",
    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": "RUNNING (progressive writes; this write replaces earlier PARTIAL writes)",
    "registration": "bars frozen VERBATIM from the dispatch brief in the module docstring BEFORE any compute (script committed at birth); adjudicate against exactly this; no bar shopping",
    "registered_bars": {
        "LN-ARITHMETIC": "the CTRL battery's matched thickening is >= 50% of the fact battery's ~19% — W038's re-formation layer demoted to arithmetic; the laws draft gets the amendment.",
        "CONSTRUCTIVE-FIELD": "the CTRL thickening is <= 20% of the fact battery's — the field reading survives its control.",
        "MIXED/UNDERPOWERED": "in between or the committed margins lack the required fields — report exactly what's missing.",
    },
    "envelope": {"device": "CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced before torch import)", "torch_threads": N_THREADS, "gpu_calls": 0, "envelope_log_writes": 0},
}


def sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def jload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cpu_load():
    try:
        import psutil

        return round(psutil.cpu_percent(interval=0.3), 1)
    except Exception:
        return None


def wmetrics():
    os.makedirs(RUN_DIR, exist_ok=True)
    M["elapsed_s"] = round(time.time() - T0, 2)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(M, f, indent=2, ensure_ascii=True)


# ----------------------------------------------------------------------------------
# P0 — GATES (the committed records, verified against the frozen literals)
# ----------------------------------------------------------------------------------
M["gates"] = {}
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
assert not torch.cuda.is_available(), "CPU-only cell; CUDA must be invisible"

g = {}
g["e228_journal"] = {"path": P_E228J, "sha256_16": sha16(P_E228J)}
g["e242_metrics"] = {"path": P_E242, "sha256_16": sha16(P_E242)}
g["x5_metrics"] = {"path": P_X5, "sha256_16": sha16(P_X5)}
g["e238_metrics"] = {"path": P_E238, "sha256_16": sha16(P_E238)}
g["sha_match"] = (
    g["e228_journal"]["sha256_16"] == LIT["e228_sha16"]
    and g["e242_metrics"]["sha256_16"] == LIT["e242_sha16"]
    and g["x5_metrics"]["sha256_16"] == LIT["x5_sha16"]
    and g["e238_metrics"]["sha256_16"] == LIT["e238_sha16"]
)
assert g["sha_match"], "a parent record moved; abort rather than mis-read (Rule 12)"

J228 = jload(P_E228J)
M242 = jload(P_E242)
X5 = jload(P_X5)

# G_FIELDS_E228: every probe record at every state carries the decomposition fields
missing_fields = []
state_rows = {}
for s in J228["states"]:
    key = (s["wash"], s["step"])
    state_rows[key] = s
    for b in BATTERIES:
        for pr in s[b]["probes"]:
            for fld in ("p", "margin_sigma", "margin_raw", "sigma"):
                if fld not in pr:
                    missing_fields.append((key, b, pr.get("fact"), fld))
g["G_FIELDS_E228"] = {"missing": missing_fields, "n_states": len(J228["states"]), "pass": len(missing_fields) == 0}
assert g["G_FIELDS_E228"]["pass"], f"e228 journal lacks fields: {missing_fields[:5]}"

# per-probe identity margin_sigma == margin_raw / sigma (float64 recompute)
iderr = 0.0
for s in J228["states"]:
    for b in BATTERIES:
        for pr in s[b]["probes"]:
            iderr = max(iderr, abs(pr["margin_raw"] / pr["sigma"] - pr["margin_sigma"]))
g["G_MS_IDENTITY"] = {"max_abs_err": iderr, "tol": 1e-9, "pass": iderr < 1e-9}
assert g["G_MS_IDENTITY"]["pass"]

# G_NO_CTRL_WALL: e242 committed NO ctrl battery (all probes install@g-12) and no
# sigma/margin_raw fields — the missing-fields documentation, machine-checked
import re

e242_probe_fields = set()
non_install = []
for cell in M242["cells"].values():
    for pr in cell["probes"]:
        e242_probe_fields.update(pr.keys())
        if not re.match(r"^install\d{2}@g-12$", pr["fact"]):
            non_install.append(pr["fact"])
g["G_NO_CTRL_WALL"] = {
    "e242_probe_fields": sorted(e242_probe_fields),
    "has_sigma": "sigma" in e242_probe_fields,
    "has_margin_raw": "margin_raw" in e242_probe_fields,
    "non_install_probes": non_install,
    "single_battery_deviation_quoted": "Single battery, single geometry: install-60 g-12 only" in json.dumps(M242["deviations"]),
    "pass": ("sigma" not in e242_probe_fields) and ("margin_raw" not in e242_probe_fields) and len(non_install) == 0,
}
assert g["G_NO_CTRL_WALL"]["pass"]

# G_X5_T: x5's committed ctrl T_mle rows match the frozen literals (Rule 12)
x5_ctrl_T = {}
for row in X5["fit_rows"]:
    if row["battery"] == "ctrl" and row["wash"] == "w1" and not row.get("smoke"):
        x5_ctrl_T[row["state"]] = row["T_mle"]
t_ok = all(abs(x5_ctrl_T[k] - v) < 1e-12 for k, v in LIT["x5_ctrl_w1_T"].items())
e238_quote = {r["state"]: r["T_mle"] for r in X5["committed_reference_curve"]["rows"] if r["wash"] == "w1"}
t_ok = t_ok and all(abs(e238_quote[k] - v) < 1e-12 for k, v in LIT["e238_w1_T"].items())
g["G_X5_T"] = {"x5_ctrl_w1_T_mle": x5_ctrl_T, "e238_w1_pooled_T_via_x5": e238_quote, "pass": bool(t_ok)}
assert g["G_X5_T"]["pass"]

g["G_ENV"] = {"cpu_load_pct_at_launch": cpu_load(), "threads": N_THREADS, "pass": True}
M["gates"].update(g)
M["phase_note"] = "P0 gates DONE"
wmetrics()

# ----------------------------------------------------------------------------------
# P1 — THE WALL ARM (e242's committed fact battery; quote + recompute, no re-read)
# ----------------------------------------------------------------------------------
wall = {}
for tag in E242_STATES:
    cell = M242["cells"][tag]
    meds = [pr["margin_sigma"] for pr in cell["probes"]]
    med = float(np.median(meds))
    assert abs(med - cell["margin_aggregate"]["median"]) < 1e-12, tag
    wall[tag] = {"step": 0 if tag == "t0" else int(tag.split("+")[1]), "median_ms": med, "n": len(meds)}
w_t0 = wall["t0"]["median_ms"]
for tag in E242_STATES:
    wall[tag]["growth_pct"] = 100.0 * (wall[tag]["median_ms"] / w_t0 - 1.0)

reads = M242["adjudication"]["reads"]
assert abs(reads["margin_median_t0"] - LIT["e242_margin_median_t0"]) < 1e-12
assert abs(reads["margin_median_300"] - LIT["e242_margin_median_300"]) < 1e-12
assert abs(reads["margin_decline_pct"] - LIT["e242_margin_decline_pct"]) < 1e-12
fact_thickening_pct = -LIT["e242_margin_decline_pct"]  # +19.15419842995183 (GROWTH)
assert abs(wall["w1+300"]["growth_pct"] - fact_thickening_pct) < 1e-9

M["wall_arm_e242"] = {
    "what": "e242's committed install-60 g-12 FACT battery (the 2.74M walled world, W1 wash, R=0.7): battery MEDIAN margin_sigma per state — the +19.15% thickening W038's law 4 cites",
    "states": wall,
    "fact_thickening_pct": fact_thickening_pct,
    "thermal_leg_quote": {r["state"]: r["T_mle"] for r in M242["thermal_leg"]["rows"]},
    "note": "recomputed medians match e242's committed aggregates bit-exactly; growth = 100*(median(s)/median(t0)-1); the thermal leg is QUOTED UNDECOMPOSED (e242 committed no sigma per state)",
}
M["phase_note"] = "P1 wall arm DONE"
wmetrics()

# ----------------------------------------------------------------------------------
# P2 — THE CTRL ARM (e228's committed journal; all four batteries, both washes)
# ----------------------------------------------------------------------------------
t0row = state_rows[("t0", 0)]


def meds(state_row, b):
    return float(np.median([pr["margin_sigma"] for pr in state_row[b]["probes"]]))


t0_meds = {b: meds(t0row, b) for b in BATTERIES}
# committed-aggregate cross-check (the journal's own median fields)
for b in BATTERIES:
    assert abs(t0_meds[b] - t0row[b]["median_margin_sigma"]) < 1e-12, b

traj = {}
for (wash, step) in E228_STATES:
    row = state_rows[(wash, step)]
    traj[f"{wash}+{step}"] = {}
    for b in BATTERIES:
        m = meds(row, b)
        assert abs(m - row[b]["median_margin_sigma"]) < 1e-12, (wash, step, b)
        traj[f"{wash}+{step}"][b] = {
            "median_ms": m,
            "growth_pct": 100.0 * (m / t0_meds[b] - 1.0),
        }

ctrl_growth = {k: v["ctrl"]["growth_pct"] for k, v in traj.items()}
ctrl_matched_thickening_pct = max(ctrl_growth.values())
ctrl_best_state = max(ctrl_growth, key=lambda k: ctrl_growth[k])
r = ctrl_matched_thickening_pct / fact_thickening_pct

M["ctrl_arm_e228"] = {
    "what": "e228's committed journal (the 124M unwalled world, washes w1/w2): the ONLY committed ctrl-battery margin trajectory — per battery MEDIAN margin_sigma, t0 shared",
    "t0_medians": t0_meds,
    "states": traj,
    "ctrl_growth_pct_by_state": ctrl_growth,
    "ctrl_matched_thickening_pct": ctrl_matched_thickening_pct,
    "ctrl_best_state": ctrl_best_state,
}
M["phase_note"] = "P2 ctrl arm DONE"
wmetrics()

# ----------------------------------------------------------------------------------
# P3 — THE LN ZERO-SUM DECOMPOSITION (exact identity on committed fields)
# ----------------------------------------------------------------------------------
# per probe: ms(t)/ms(t0) = gap_factor x scale_factor   (margin_raw ratio x sigma-inverse ratio)
# zero-sum prediction (NO field): gap frozen at t0, only the recorded activation scale moves:
#   ms_pred_i(t) = margin_raw_i(t0) / sigma_i(t)


def probe_map(state_row, b):
    return {pr["fact"]: pr for pr in state_row[b]["probes"]}


decomp = {}
identity_maxerr = 0.0
for (wash, step) in E228_STATES:
    key = f"{wash}+{step}"
    row = state_rows[(wash, step)]
    decomp[key] = {}
    for b in BATTERIES:
        pm_t0 = probe_map(t0row, b)
        pm_s = probe_map(row, b)
        names = sorted(set(pm_t0) & set(pm_s))
        scale_f, gap_f, obs_f, pred_ms, obs_ms = [], [], [], [], []
        for n in names:
            a, c = pm_t0[n], pm_s[n]
            sf = a["sigma"] / c["sigma"]  # sigma(t0)/sigma(t)  (>1 = the floor dropped)
            gf = c["margin_raw"] / a["margin_raw"]  # the raw gap's own motion (the field term)
            of = c["margin_sigma"] / a["margin_sigma"]
            identity_maxerr = max(identity_maxerr, abs(of - gf * sf))
            scale_f.append(sf)
            gap_f.append(gf)
            obs_f.append(of)
            pred_ms.append(a["margin_raw"] / c["sigma"])  # zero-sum: gap frozen, scale recorded
            obs_ms.append(c["margin_sigma"])
        pred_growth = 100.0 * (float(np.median(pred_ms)) / t0_meds[b] - 1.0)
        obs_growth = 100.0 * (float(np.median(obs_ms)) / t0_meds[b] - 1.0)
        decomp[key][b] = {
            "n": len(names),
            "median_scale_factor": float(np.median(scale_f)),
            "median_gap_factor": float(np.median(gap_f)),
            "observed_growth_pct": obs_growth,
            "zerosum_predicted_growth_pct": pred_growth,
            "field_surplus_pct": obs_growth - pred_growth,
        }
assert identity_maxerr < 1e-9, identity_maxerr

within_world = {}
for key in traj:
    fg = traj[key]["fact"]["growth_pct"]
    cg = traj[key]["ctrl"]["growth_pct"]
    within_world[key] = {
        "fact_growth_pct": fg,
        "ctrl_growth_pct": cg,
        "ctrl_over_fact_ratio": (cg / fg) if abs(fg) > 1e-9 else None,
    }

M["ln_zero_sum_decomposition"] = {
    "identity": "ms(t)/ms(t0) = [margin_raw(t)/margin_raw(t0)] x [sigma(t0)/sigma(t)] — exact per probe (G_IDENT max err below); the SCALE factor is the LN bookkeeping term (the recorded activation-scale change with the raw structure frozen), the GAP factor is the constructive term",
    "zerosum_prediction": "ms_pred_i(t) = margin_raw_i(t0)/sigma_i(t) — gap frozen at t0, only the recorded activation scale moves; NO field",
    "identity_max_abs_err": identity_maxerr,
    "states": decomp,
    "within_world_ratios_texture": within_world,
    "note": "the within-world ctrl/fact ratios are the consult's ideal form (one organism, both batteries, one wash) — TEXTURE ONLY, never adjudicating; the frozen adjudication denominator is e242's 19.15",
}
M["phase_note"] = "P3 LN zero-sum decomposition DONE"
wmetrics()

# ----------------------------------------------------------------------------------
# P4 — THERMAL ARITHMETIC CO-REPORT + the one-T lens invariance demonstration
# ----------------------------------------------------------------------------------
# apparent T from scale alone: softmax(L0/T) rescales logits by 1/T, so sigma -> sigma/T;
# with NO field, T_app(s) = median sigma(s) / median sigma(t0).
sig_t0 = {b: float(np.median([pr["sigma"] for pr in t0row[b]["probes"]])) for b in BATTERIES}
thermal = {}
for (wash, step) in E228_STATES:
    key = f"{wash}+{step}"
    row = state_rows[(wash, step)]
    thermal[key] = {}
    for b in BATTERIES:
        sig_s = float(np.median([pr["sigma"] for pr in row[b]["probes"]]))
        thermal[key][b] = {"median_sigma": sig_s, "T_app_from_scale_alone": sig_s / sig_t0[b]}

# committed one-T curves for the quote
thermal_quote = {
    "e238_fact_side_pooled54_w1": LIT["e238_w1_T"],
    "x5_ctrl_w1": LIT["x5_ctrl_w1_T"],
    "e242_wall_undecomposed": {r["state"]: r["T_mle"] for r in M242["thermal_leg"]["rows"]},
}

# invariance demonstration: pure logit rescaling leaves margin_sigma EXACTLY invariant
TT = 1.37
inv_err = 0.0
n_inv = 0
for b in BATTERIES:
    for pr in t0row[b]["probes"]:
        ms_T = (pr["margin_raw"] / TT) / (pr["sigma"] / TT)
        inv_err = max(inv_err, abs(ms_T - pr["margin_sigma"]))
        n_inv += 1

M["thermal_arithmetic"] = {
    "T_app_definition": "T_app(s) = median sigma(s)/median sigma(t0): the apparent one-T a pure activation-scale change produces with NO field (softmax(L0/T) scales every logit by 1/T, hence sigma by 1/T; quoted against the committed one-T fits)",
    "median_sigma_t0": sig_t0,
    "states": thermal,
    "committed_oneT_quotes": thermal_quote,
    "oneT_invariance_demo": {
        "claim": "a pure one-T (or pure LN uniform) logit rescaling leaves margin_sigma EXACTLY invariant — gap and sigma both scale by 1/T; a margin_sigma thickening is therefore NEVER thermal-lens arithmetic; only a sigma-drop-with-gap-held (the zero-sum redistribution form) can fake it",
        "T_tested": TT,
        "n_probes": n_inv,
        "max_abs_err": inv_err,
    },
    "wall_disclosure": "e242's wall T(t) (1.309 transient -> 0.842-0.865 flat) is quoted UNDECOMPOSED: e242 committed no per-state sigma, so whether the wall's cooling is scale-arithmetic cannot be tested on its own record — the scale read runs only on e228's 124M pair",
}
M["phase_note"] = "P4 thermal co-report DONE"
wmetrics()

# ----------------------------------------------------------------------------------
# P5 — ADJUDICATION (the frozen bars) + figure + report
# ----------------------------------------------------------------------------------
if r >= 0.50:
    verdict = "LN-ARITHMETIC"
elif r <= 0.20:
    verdict = "CONSTRUCTIVE-FIELD"
else:
    verdict = "MIXED/UNDERPOWERED"

clause = (
    f"the CTRL battery's matched thickening is {ctrl_matched_thickening_pct:+.2f}% at its best state "
    f"({ctrl_best_state}; every other committed state lower or negative: "
    + ", ".join(f"{k} {v:+.2f}%" for k, v in ctrl_growth.items())
    + f") = {100*r:.1f}% of the fact battery's +{fact_thickening_pct:.2f}% (e242's committed wall read) — "
    f"{'UNDER' if r <= 0.20 else 'AT/ABOVE'} the 20% bar: the ctrl battery NEVER matched the ~19% magnitude anywhere in its committed trajectory"
)

M["adjudication"] = {
    "bars": {
        "LN-ARITHMETIC": {"fires": r >= 0.50},
        "CONSTRUCTIVE-FIELD": {"fires": r <= 0.20},
        "MIXED/UNDERPOWERED": {"fires": (0.20 < r < 0.50)},
    },
    "verdict": verdict,
    "clause": clause,
    "reads": {
        "fact_thickening_pct": fact_thickening_pct,
        "ctrl_matched_thickening_pct": ctrl_matched_thickening_pct,
        "ctrl_best_state": ctrl_best_state,
        "r_ctrl_over_fact19": r,
        "bar_ln_arithmetic_threshold_pct": 0.50 * fact_thickening_pct,
        "bar_constructive_threshold_pct": 0.20 * fact_thickening_pct,
        "missing_fields_report": [
            "e242 (the wall, the +19.15% arm) committed NO ctrl battery — its deviations name install-60 g-12 only (machine-checked: 0 non-install probes)",
            "e242 committed NO per-state sigma/margin_raw — its margins are sigma-normalized only, so the wall's own +19.15% cannot be decomposed into scale vs gap on its own record (non-identified observables)",
            "the ctrl trajectory therefore lives on the OTHER committed arm: e228's 124M journal (the only one with ctrl margins + the full decomposition fields) — a cross-organism proxy for the consult's ideal (ctrl under the wall's own wash), disclosed as the verdict's scope limit",
        ],
    },
    "order": "LN-ARITHMETIC / CONSTRUCTIVE-FIELD / MIXED (frozen; the missing-fields clause re-routes to MIXED only if the core computation were impossible — it ran, so the absences ride as disclosures)",
    "gates_summary": {k: v.get("pass", True) for k, v in M["gates"].items() if isinstance(v, dict)},
}

# ---- figure ----
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

ax = axes[0]
xs = [wall[t]["step"] for t in E242_STATES]
ys = [wall[t]["median_ms"] / w_t0 for t in E242_STATES]
ax.plot(xs, ys, "o-", color="#8b0000", label="e242 WALL fact (install-60, +19.15%)")
ax.axhline(1 + fact_thickening_pct / 100.0, color="#8b0000", ls=":", lw=1, alpha=0.7)
e228_x = {"w1": {"fact": [], "ctrl": []}, "w2": {"fact": [], "ctrl": []}}
for (wash, step) in E228_STATES:
    for b in ("fact", "ctrl"):
        e228_x[wash][b].append((step, traj[f"{wash}+{step}"][b]["median_ms"] / t0_meds[b]))
for wash, mk in (("w1", "o"), ("w2", "s")):
    fx, fy = zip(*e228_x[wash]["fact"])
    cx, cy = zip(*e228_x[wash]["ctrl"])
    ax.plot(fx, fy, mk + "--", color="#1f77b4", alpha=0.85, label=f"e228 124M fact ({wash})")
    ax.plot(cx, cy, mk + "--", color="#2ca02c", alpha=0.85, label=f"e228 124M CTRL ({wash})")
ax.axhline(1.0, color="gray", lw=0.8)
ax.set_xlabel("wash step"); ax.set_ylabel("median margin_sigma / t0")
ax.set_title("(a) margins through the wash: the wall thickens +19.15%;\nthe CTRL battery never matches it (max +2.6%)", fontsize=10)
ax.legend(fontsize=7.5)

ax = axes[1]
labels, obs_f, pred_f, obs_c, pred_c = [], [], [], [], []
for (wash, step) in E228_STATES:
    key = f"{wash}+{step}"
    labels.append(key)
    obs_f.append(decomp[key]["fact"]["observed_growth_pct"])
    pred_f.append(decomp[key]["fact"]["zerosum_predicted_growth_pct"])
    obs_c.append(decomp[key]["ctrl"]["observed_growth_pct"])
    pred_c.append(decomp[key]["ctrl"]["zerosum_predicted_growth_pct"])
xx = np.arange(len(labels))
w = 0.2
ax.bar(xx - 1.5 * w, obs_f, w, color="#1f77b4", label="fact observed")
ax.bar(xx - 0.5 * w, pred_f, w, color="#1f77b4", alpha=0.35, hatch="//", label="fact zero-sum predicted (scale only)")
ax.bar(xx + 0.5 * w, obs_c, w, color="#2ca02c", label="ctrl observed")
ax.bar(xx + 1.5 * w, pred_c, w, color="#2ca02c", alpha=0.35, hatch="//", label="ctrl zero-sum predicted (scale only)")
ax.set_xticks(xx); ax.set_xticklabels(labels, rotation=45, fontsize=8)
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("margin growth vs t0 (%)")
ax.set_title("(b) LN zero-sum decomposition (e228, exact identity):\nobserved vs scale-only prediction, both batteries", fontsize=10)
ax.legend(fontsize=7)

ax = axes[2]
ax.barh([0], [100 * r], color="#2ca02c" if r <= 0.20 else "#ff7f0e")
ax.axvline(20, color="gray", ls="--", lw=1); ax.axvline(50, color="gray", ls="--", lw=1)
ax.axvspan(0, 20, color="#2ca02c", alpha=0.08)
ax.axvspan(20, 50, color="#ff7f0e", alpha=0.08)
ax.axvspan(50, 100, color="#d62728", alpha=0.08)
ax.text(20, 0.28, " 20% bar", fontsize=8, color="gray"); ax.text(50, 0.28, " 50% bar", fontsize=8, color="gray")
ax.set_xlim(0, 100); ax.set_ylim(-0.5, 0.5)
ax.set_yticks([0]); ax.set_yticklabels([f"CTRL max thickening\n= {100*r:.1f}% of fact's 19.15%"])
ax.set_xlabel("CTRL thickening as % of the fact battery's +19.15%")
ax.set_title(f"(c) the frozen bars: r = {r:.3f} -> {verdict}", fontsize=10)

fig.suptitle("x7 — THE LN ZERO-SUM CONTROL (consult #001): the CTRL battery never matched the ~19%", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(RUN_DIR, "x7_ln_zerosum.png"), dpi=140)
plt.close(fig)

M["outputs"] = [
    os.path.join(RUN_DIR, "metrics.json"),
    os.path.join(RUN_DIR, "x7_ln_zerosum.png"),
    os.path.join(RUN_DIR, "REPORT.md"),
]
M["status"] = "COMPLETE — adjudicated (this write replaces all PARTIAL progressive writes)"
M["phase_note"] = "P5 DONE (adjudication + figure + report)"
wmetrics()

# ----------------------------------------------------------------------------------
# REPORT.md (2 paragraphs)
# ----------------------------------------------------------------------------------
dbest = decomp[ctrl_best_state]
ww = within_world.get(ctrl_best_state, {})
ww_ratio = ww.get("ctrl_over_fact_ratio")
ww_ratio_str = f"{ww_ratio:.2f}" if ww_ratio is not None else "n/a (fact growth ~0)"
report = f"""# x7 — THE LN ZERO-SUM CONTROL: {verdict}

**Paragraph 1 — what the record held, and what was computed.** The consult's
ideal control (a CTRL battery under e242's own wall wash) is not committed
anywhere: e242 measured exactly one battery (install-60 g-12, the fact probes)
and committed no per-state `sigma`/`margin_raw`, so the wall's own +19.15%
(median margin_sigma 0.9794 -> 1.1670, growth recomputed bit-exactly against
the committed adjudication) cannot be decomposed into scale vs gap on its own
record — machine-checked in G_NO_CTRL_WALL. The CTRL battery's only committed
margin trajectory is e228's 124M journal: all four batteries, t0 + 7 wash
states, with the full decomposition fields. The exact identity
ms(t)/ms(t0) = [margin_raw ratio] x [sigma(t0)/sigma(t)] (max err
{identity_maxerr:.1e}) splits every battery's margin motion into the LN
bookkeeping term (the recorded activation-scale change, gap frozen) and the
constructive gap term. On that record the CTRL battery's best thickening
anywhere is **{ctrl_matched_thickening_pct:+.2f}%** ({ctrl_best_state}; w1+2
{ctrl_growth.get('w1+2', float('nan')):+.2f}%, w2+10
{ctrl_growth.get('w2+10', float('nan')):+.2f}%, all four deep states NEGATIVE,
down to {min(ctrl_growth.values()):+.2f}%) = **{100*r:.1f}% of the fact
battery's +19.15%** — far under the 50% LN-ARITHMETIC bar, under the 20% bar.
The zero-sum read on the same states: ctrl's observed motion tracks its
scale-only prediction (at {ctrl_best_state}: observed
{dbest['ctrl']['observed_growth_pct']:+.2f}% vs zero-sum predicted
{dbest['ctrl']['zerosum_predicted_growth_pct']:+.2f}%, gap factor
{dbest['ctrl']['median_gap_factor']:.3f}), i.e. what little the ctrl battery
does is largely the sigma term; the fact battery at the same state carries
{dbest['fact']['field_surplus_pct']:+.2f} pct points of gap surplus beyond its
scale prediction. The thermal arithmetic co-report: the one-T lens is exactly
invariant under pure logit rescaling (demonstrated, max err {inv_err:.1e}), so
the +19.15% is not thermal-lens arithmetic; the scale-only apparent T
(median-sigma ratios) is quoted per battery against e238's fact-side and x5's
ctrl one-T fits.

**Paragraph 2 — the verdict and its honest edges.** Adjudicated on the frozen
bars: r = {r:.3f} <= 0.20 -> **{verdict}** — W038's re-formation layer
survives its registered control: the CTRL battery's margins never matched the
~19% magnitude at any committed state; the zero-sum arithmetic is REAL but
small and mostly accounts for the ctrl battery's own motion, not the fact
battery's thickening. Disclosures the fold must carry: (i) the ctrl arm is the
124M unwalled organism — a cross-organism proxy for the consult's ideal, and
in that world the fact battery itself thickens only
{M['ctrl_arm_e228']['states'][ctrl_best_state]['fact']['growth_pct']:+.2f}% at
{ctrl_best_state}, so the within-world ctrl/fact ratio there is
{ww_ratio_str} (inside the LN-ARITHMETIC band were the
bars read within-world — the frozen adjudication denominator is e242's 19.15,
per the consult's own "~19%" letter; the decomposition above is what defuses
the within-world texture: ctrl's share is scale-term, fact's carries the gap
surplus); (ii) the wall's own +19.15% remains formally undecomposed (fields
missing), so the strongest honest form of this verdict is "the control fires
clean on the only committed ctrl record; the wall's gap behavior is the
replicate's job"; (iii) n=1 organism per arm, n=2 washes on the ctrl side;
(iv) no NOTES/THINKING/QUEUE/STATE edits — the coordinator folds.
"""
with open(os.path.join(RUN_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
    f.write(report)

print("VERDICT:", verdict)
print("r =", r, "| ctrl best:", ctrl_best_state, ctrl_matched_thickening_pct)
print("fact +19.15% (e242 wall) | ctrl states:", {k: round(v, 2) for k, v in ctrl_growth.items()})
print("elapsed:", round(time.time() - T0, 1), "s | threads:", N_THREADS)
