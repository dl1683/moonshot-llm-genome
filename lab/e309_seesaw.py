# -*- coding: utf-8 -*-
"""
e309 — THE CORPSE SEESAW (desk cell, CPU-only, clean start after a prior
executor died on a rate limit before artifacts).

THE QUESTION (dispatch letter, verbatim in spirit and letter):
  e179's law says one cons replay resurrects the dead; e299/x14 say
  subtraction resurrects the dead. Run both routes on the same corpse
  classes from the COMMITTED records: do they revive the SAME memory — or
  does surgery revive the original while the cons BUILDS A NEW ONE?
  Revival vs replacement.

FROZEN BARS (verbatim from the dispatch letter; adjudicated against
exactly this; no bar shopping):
  TWO-DOORS-ONE-MEMORY — the routes reconcile in value and geometry.
  TWO-MEMORIES — surgery revives the original write's read; the cons
    builds its standard construction — corpse-independent landing;
    e179's 'resurrection' and x14's 'resurrection' are different events
    sharing a name.
  UNITS-BARRED — the records cannot reconcile; native tables verbatim.
  MIXED — anything else, no inflation.

OPERATIONALIZATIONS (frozen at birth, BEFORE any compute):
  This cell is a DESK ADJUDICATION over COMMITTED records — no states
  are loaded, no training runs, no probes fire. Every number is EXTRACTED
  PROGRAMMATICALLY from the md5-bound parent metrics files and VALUE-BOUND
  against the design-time literal snapshot in FROZEN below (a mismatch
  HALTS). The two routes:
    ROUTE S (surgery) := e299's committed Arm-A table (x14's subtraction,
      ported and instrument-checked by e299's G_PORTX14 |d| 0.0): the
      recovered fraction of the loaded post-g0 baseline 0.26464763283729553,
      per corpse, 7 corpses in 3 kill classes.
    ROUTE C (cons) := e281's committed five-point landing curve (one cons
      stream, one session, seed 10901) + the parent sessions' landing
      records it anchors to (e268/e270): the landing read in the ROOT-g0
      battery, the family band [0.65, 0.82] hard.
  CLASS MAP (frozen): free-AdamW := {e283-CONCURRENT, e285-TWIN} on the
    surgery side, {K10KD = e268's 10k concurrent, K100KD = e270's 100k
    concurrent} on the cons side — SAME CLASS, DIFFERENT STATES (no corpse
    state has both routes in the committed record; disclosed, the table's
    rows are class rows). orthogonal-sgdm := the 4 orthogonal corpses
    (surgery side ONLY — no committed cons datum exists for ANY
    orthogonal-class corpse; the gap is a finding, disclosed verbatim).
    orthogonal+in-room-maint := e288 (surgery only, same gap). The cons
    controls NOINST (no write at all) / R10 / K1KM are width-poverty
    deaths, reported as controls, never as orthogonal-class rows.
  UNITS WORK (frozen method): the surgery route's outputs live in the
    POST-g0 battery (fractions of 0.2646); the cons route's outputs live
    in the ROOT-g0 battery. The committed record's only matched-state
    dual readings are e264's serial curve (hardbound in e281's G_PARENTS):
    post/root at widths 1k/10k/100k. The conversion is worked THREE ways
    and each assumption DISCLOSED: (1) the inventory test — does ANY
    committed record read either route's product in the other's battery?
    (2) the proportionality test — is post/root constant across widths?
    (3) the pseudo-conversion — landings divided by the alive root anchors
    (e264's rungs), reported as NON-BINDING context with its assumptions
    named, because the zero-point control has no denominator.
  GEOMETRY (frozen sources): e299's surviving-mass reads (in-room mass
    ratio, cos of the surviving component vs the write's in-room
    direction, per corpse) for the surgery side; e268/e270's committed
    displacement-load ledgers (in_own_room of the install and of the
    cons-landing root states, serial-alive-seeded vs concurrent-corpse-
    seeded) for the cons side; e281's own record carries seed displacement
    L2 only (its own-room reads were vs the R10 room only — its
    disclosure, carried). e179's founding law record cited with its
    battery named (family-1 g-12 ruler; cross-era units uncertified).
  THE SEESAW TEST (frozen decision rule, code in adjudicate()):
    fire TWO-MEMORIES iff (i) the cons landing is corpse-independent BY
    ITS OWN RECORD (e281 verdict FLAT AND the no-write zero point lands
    in the hard band — a landing from a state with no memory cannot be a
    revival of one) AND (ii) the surgery's product is the original
    write's read (e299's construction identity theta0 + P(delta) with ID
    closure 0.0 AND the corpse-dependent mass-tracked class split).
    fire TWO-DOORS-ONE-MEMORY only if a cross-battery reconciliation
    datum EXISTS and agrees. fire UNITS-BARRED only if the identity
    discriminators are missing from the record (each leg must then be
    read natively and the question stays open). MIXED otherwise.

PREDICTIONS (registered pre-compute, this desk):
  P-SEESAW: TWO-MEMORIES fires — the zero-point control decides it
    unit-free within e281's own battery; the surgery leg is original by
    construction; the VALUE leg alone is units-barred.
  P-GEOM: the parent records' landing occupancies do NOT rank by the
    write's presence (corpse-seeded >= alive-seeded in_own_room in both
    e268 and e270) — the landing geometry is the cons's own signature.
  P-GAP: the orthogonal class has NO cons datum anywhere in the
    committed record — the cons column there is empty by record.
  P-CONV: the pseudo-conversion reads 'full' (0.92/1.11 of the alive
    root anchors) yet cannot discriminate — the zero point has no
    denominator; a construction and a revival convert identically.

ENVELOPE: CPU-ONLY desk cell. Threads <= 4 (OMP/OPENBLAS/MKL/NUMEXPR
pinned before numpy import). NO torch import, NO cuda tensor, NO GPU op,
NO training, NO state loads, NO envelope-log writes. Timestamps:
datetime.now(UTC) ONLY. Progressive metrics.json writes per phase.
No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import math
import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "e309"
RUN.mkdir(parents=True, exist_ok=True)

E309_ID = "e309_seesaw"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def md5file(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------
# FROZEN design-time snapshot (recorded at birth, BEFORE compute; every
# extracted value is checked against this table; a mismatch HALTS)
# ----------------------------------------------------------------------------
PARENTS = {
    "e299": {"path": "runs/e299/metrics.json", "md5": "715a59caea1a1615b5a132b2b963a0db"},
    "e281": {"path": "runs/e281/metrics.json", "md5": "3f57291095368d112b6dbb1e5ab30288"},
    "e268": {"path": "runs/e268/metrics.json", "md5": "c1149229b7f0191943a7b8eb0442b494"},
    "e270": {"path": "runs/e270/metrics.json", "md5": "006ae12e5ee7ff262db2d1c19d95c0f1"},
    "e179": {"path": "runs/e179/metrics.json", "md5": "b3d91e3567e2c5eb238da015f3c2585b"},
    "x14":  {"path": "runs/x14/metrics.json",  "md5": "4970e27ae8c315df8762e5c3499a2be5"},
}

FROZEN = {
    # --- e299: the surgery route (post-g0 battery; baseline hardbound) ---
    "baseline_post_g0": 0.26464763283729553,
    "x14_arm_A_g0": 0.03915366902947426,
    "x14_port_abs_diff": 0.0,
    "e299_verdict": "ALWAYS-RESUSCITABLE + MASS-TRACKS",
    "e299_rho_fraction_vs_mass": 0.8829187134416479,
    "surgery_fractions": {
        "e283-CONCURRENT": 0.14794641693827582,
        "e285-TWIN": 0.15296437788263895,
        "e285-SANCTUARY": 0.9999996621660047,
        "e290-0.1X": 0.9999996621660047,
        "e290-0.02X": 1.0000006756679907,
        "e290-0.004X": 0.9999987612753504,
        "e288-NAME-FIXED-TWIN": 1.0478814373845522,
    },
    "surgery_mass_ratios": {
        "e283-CONCURRENT": 0.8434573564882518,
        "e285-TWIN": 0.8440658727037936,
        "e285-SANCTUARY": 0.999999997823999,
        "e290-0.1X": 0.9999999988512346,
        "e290-0.02X": 0.9999999967568527,
        "e290-0.004X": 0.9999999948868795,
        "e288-NAME-FIXED-TWIN": 1.0002659403575358,
    },
    "surgery_factors": {
        "e283-CONCURRENT": 4328.215777016314,
        "e285-TWIN": 10390.853270529175,
    },
    "surgery_cos_surviving": {
        "e283-CONCURRENT": 0.9930729893654585,
        "e285-TWIN": 0.9932813027642661,
    },
    "surgery_id_closure": 0.0,  # every corpse: construction_id_closure_fp64
    # --- e281: the cons route (root-g0 battery; band hard) ---
    "e281_verdict": "FLAT",
    "band": [0.65, 0.82],
    "landings": {
        "NOINST": 0.6508122086524963,
        "R10": 0.693087637424469,
        "K1KM": 0.6881048083305359,
        "K10KD": 0.7117618918418884,
        "K100KD": 0.8103598356246948,
    },
    "seed_post_g0": {
        "NOINST": 1.3383959412749391e-05,
        "R10": 4.028632974950597e-05,
        "K1KM": 0.0012453667586669326,
        "K10KD": 4.004325455753133e-05,
        "K100KD": 0.010897441767156124,
    },
    "seed_displacement_l2": {
        "NOINST": 0.0, "R10": 2.762690179770432, "K1KM": 15.033078227618146,
        "K10KD": 37.210529791827966, "K100KD": 40.80605278026411,
    },
    # e264 matched-state dual readings (hardbound inside e281's G_PARENTS)
    "e264_curve_post": {"1000": 0.00043458465370349586,
                        "10000": 0.26464763283729553,
                        "100000": 0.43598243594169617},
    "e264_curve_root": {"1000": 0.5768678784370422,
                        "10000": 0.7707884907722473,
                        "100000": 0.7276116609573364},
    # --- e268/e270: the cons-side geometry (in_own_room displacement fracs) ---
    "e268": {"SERIAL_install_in_room": 0.944158883562913,
             "SERIAL_root_in_room": 0.41335725848418325,
             "SERIAL_root_g0": 0.764111340045929,
             "CONCURRENT_install_in_room": 0.6695546684696058,
             "CONCURRENT_root_in_room": 0.5841513782723378,
             "CONCURRENT_root_g0": 0.7118977308273315},
    "e270": {"SERIAL_install_in_room": 0.9814438757998657,
             "SERIAL_root_in_room": 0.6483002018978558,
             "SERIAL_root_g0": 0.6884167194366455,
             "CONCURRENT_install_in_room": 0.7681272067523449,
             "CONCURRENT_root_in_room": 0.6896646631755264,
             "CONCURRENT_root_g0": 0.8102989196777344},
    # --- e179: the founding law (family-1 g-12 ruler; cross-era battery) ---
    "e179_resurrection_gm12_at50": 0.6861528158187866,
    "e179_wash_control_gm12_at50": 0.022121647372841835,
    # --- room chance occupancy (K10K) ---
    "k10k_chance_occupancy": 0.0036508715360530864,
}

SURGERY_CLASS = {
    "e283-CONCURRENT": "free-adamw",
    "e285-TWIN": "free-adamw",
    "e285-SANCTUARY": "orthogonal-sgdm",
    "e290-0.1X": "orthogonal-sgdm",
    "e290-0.02X": "orthogonal-sgdm",
    "e290-0.004X": "orthogonal-sgdm",
    "e288-NAME-FIXED-TWIN": "orthogonal+in-room-maint",
}
CONS_CLASS = {  # the cons arms' corpse classes (width-poverty rows are controls)
    "NOINST": "no-write-control",
    "R10": "width-poverty(dead write, rank-10)",
    "K1KM": "width-poverty(dead write, 1k dose-comp)",
    "K10KD": "free-adamw-class(10k concurrent kill, e268)",
    "K100KD": "free-adamw-class(100k concurrent kill, e270)",
}

MET = {
    "experiment": "e309_seesaw",
    "phase": ("THE CORPSE SEESAW — e179's law (one cons replay resurrects the "
              "dead) vs e299/x14's subtraction resurrection, on the same corpse "
              "classes from the COMMITTED records: do the two routes revive the "
              "SAME memory, or does surgery revive the original write while the "
              "cons BUILDS ITS OWN? Revival vs replacement"),
    "date": now(),
    "status": "BIRTH (bars frozen; no compute yet)",
    "smoke": False,
    "envelope": {
        "device": "CPU-ONLY desk cell — threads 4 (OMP/OPENBLAS/MKL/NUMEXPR "
                  "pinned); NO torch import, NO cuda tensor, NO GPU op, NO "
                  "training, NO state loads, NO envelope-log writes",
        "trainings": "NONE — a desk adjudication over md5-bound committed "
                     "records; nothing moves",
        "timestamps": "datetime.now(UTC) only",
    },
    "registration": {
        "bars_verbatim": {
            "TWO-DOORS-ONE-MEMORY": "the routes reconcile in value and geometry",
            "TWO-MEMORIES": ("surgery revives the original write's read; the cons "
                             "builds its standard construction — corpse-independent "
                             "landing; e179's 'resurrection' and x14's 'resurrection' "
                             "are different events sharing a name"),
            "UNITS-BARRED": "the records cannot reconcile; native tables verbatim",
            "MIXED": "anything else, no inflation",
        },
        "decision_rule": ("TWO-MEMORIES iff cons landing corpse-independent by its "
                          "own record (e281 FLAT AND zero point in hard band) AND "
                          "surgery product original (construction identity + "
                          "mass-tracked corpse-dependent split); TWO-DOORS only if "
                          "a cross-battery reconciliation datum exists and agrees; "
                          "UNITS-BARRED only if the identity discriminators are "
                          "missing; MIXED otherwise"),
        "source_letter": "the e309 dispatch letter (R70's re-dispatch, commit c92c302)",
    },
    "predictions_pre_compute": {
        "P-SEESAW": "TWO-MEMORIES fires — the zero-point control decides it "
                    "unit-free within e281's own battery; the surgery leg is "
                    "original by construction; the VALUE leg alone is units-barred",
        "P-GEOM": "the parent records' landing occupancies do NOT rank by the "
                  "write's presence (corpse-seeded >= alive-seeded in_own_room in "
                  "both e268 and e270) — the landing geometry is the cons's own",
        "P-GAP": "the orthogonal class has NO cons datum anywhere in the committed "
                 "record — the cons column there is empty by record",
        "P-CONV": "the pseudo-conversion reads 'full' (0.92/1.11 of the alive root "
                  "anchors) yet cannot discriminate — the zero point has no "
                  "denominator; a construction and a revival convert identically",
    },
    "deviations": [
        "THE SNAPSHOT CORRECTION (post-birth, pre-adjudication, the gate's own "
        "catch): the design-time frozen literal e270.CONCURRENT_root_in_room was "
        "transcribed ...5344 for the file's verbatim ...5264 (a 2-digit slip); "
        "G_LITERAL halted the first run on it BEFORE any table, unit work, or "
        "adjudication executed; the literal was corrected verbatim from "
        "runs/e270/metrics.json and the run restarted clean. No bar, gate form, "
        "extraction path, or adjudication logic was touched — the bind did "
        "exactly what it exists to do.",
        "INSTRUMENT SHAKEOUT (disclosed): the first execution attempts halted on "
        "extraction-path misses (e299's hardbound lives under G_PARENTS; e179's "
        "cells live under cells.r1_32) and two report-format defects — all fixed "
        "before the adjudicating run; the adjudication, bars, and frozen rule are "
        "exactly the birth-committed forms.",
        "HEAD MOVED BETWEEN BIRTH AND RUN: other fleet cells committed after "
        "this cell's birth commit 81f6511 and before its adjudicating run — "
        "both hashes recorded in provenance; the registration is the birth "
        "commit.",
        "CPU-ONLY desk cell (dispatch): threads 4 pinned, no torch import, no "
        "cuda tensor, no GPU op, no training, no state loads, no envelope-log "
        "writes; datetime.now(UTC) only.",
        "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).",
    ],
}


def write_met(status):
    MET["status"] = status
    MET["date_updated"] = now()
    (RUN / "metrics.json").write_text(
        json.dumps(MET, indent=2, ensure_ascii=True), encoding="utf-8")


def halt(gate, msg):
    MET.setdefault("gates", {})[gate] = {"pass": False, "error": msg}
    write_met("HALTED at %s — %s" % (gate, msg))
    raise SystemExit("HALT: %s: %s" % (gate, msg))


def check_val(gate, name, mine, frozen, tol=0.0):
    if isinstance(frozen, float) and isinstance(mine, (int, float)) and not isinstance(mine, bool):
        ok = abs(mine - frozen) <= tol
        diff = abs(mine - frozen)
    else:
        ok = mine == frozen
        diff = 0.0 if ok else float("inf")
    if not ok:
        halt(gate, "value bind FAILED for %s: mine=%r frozen=%r" % (name, mine, frozen))
    return {"name": name, "mine": mine, "frozen": frozen, "abs_diff": diff}


# ============================================================================
# P0 — load the parents, md5-bind, extract, value-bind
# ============================================================================
P = {}
g_parents = {}
for key, spec in PARENTS.items():
    p = ROOT / spec["path"]
    live = md5file(p)
    rec = {"path": str(p), "md5": live, "bound_md5": spec["md5"], "pass": live == spec["md5"]}
    g_parents[key] = rec
    if not rec["pass"]:
        halt("G_PARENTS", "%s md5 drift: %s != %s" % (key, live, spec["md5"]))
    P[key] = json.loads(p.read_text(encoding="utf-8"))

m299, m281, m268, m270, m179 = P["e299"], P["e281"], P["e268"], P["e270"], P["e179"]

# --- extract + bind: e299 surgery route ---
binds = []
base = m299["gates"]["G_PARENTS"]["hardbound"]["fact_baseline_g0"]
binds.append(check_val("G_LITERAL", "baseline_post_g0", base, FROZEN["baseline_post_g0"]))
binds.append(check_val("G_LITERAL", "x14_arm_A_g0",
                       m299["gates"]["G_PARENTS"]["hardbound"]["x14_arm_A_g0"], FROZEN["x14_arm_A_g0"]))
binds.append(check_val("G_LITERAL", "x14_port_abs_diff",
                       m299["gates"]["G_PORTX14"]["arm_A_g0"]["abs_diff"], FROZEN["x14_port_abs_diff"]))
binds.append(check_val("G_LITERAL", "e299_verdict",
                       m299["adjudication"]["verdict"], FROZEN["e299_verdict"]))
binds.append(check_val("G_LITERAL", "e299_rho",
                       m299["curve"]["correlations"]["primary_spearman_fraction_vs_mass"]["rho"],
                       FROZEN["e299_rho_fraction_vs_mass"]))

surgery = {}
for rec in m299["surgery"]["records"]:
    k = rec["key"]
    surgery[k] = {
        "class": SURGERY_CLASS[k],
        "fraction": rec["resurrection"]["recovered_fraction_of_baseline"],
        "factor": rec["resurrection"]["factor_over_dead"],
        "mass_ratio": rec["write_mass"]["mass_ratio_vs_t0"],
        "cos_surviving": rec["write_mass"]["cos_surviving_vs_write_in_room"],
        "drift_in_room_frac": rec["kill_depth"]["drift_in_room_frac"],
        "id_closure": rec["arm_A_orthogonal_subtraction"]["construction_id_closure_fp64"],
        "arm_A_g0": rec["arm_A_orthogonal_subtraction"]["read"]["g0"],
    }
for k, v in FROZEN["surgery_fractions"].items():
    binds.append(check_val("G_LITERAL", "fraction[%s]" % k, surgery[k]["fraction"], v))
for k, v in FROZEN["surgery_mass_ratios"].items():
    binds.append(check_val("G_LITERAL", "mass_ratio[%s]" % k, surgery[k]["mass_ratio"], v))
for k, v in FROZEN["surgery_factors"].items():
    binds.append(check_val("G_LITERAL", "factor[%s]" % k, surgery[k]["factor"], v))
for k, v in FROZEN["surgery_cos_surviving"].items():
    binds.append(check_val("G_LITERAL", "cos[%s]" % k, surgery[k]["cos_surviving"], v))
for k in surgery:
    binds.append(check_val("G_LITERAL", "id_closure[%s]" % k, surgery[k]["id_closure"],
                           FROZEN["surgery_id_closure"]))

# --- extract + bind: e281 cons route ---
binds.append(check_val("G_LITERAL", "e281_verdict", m281["adjudication"]["verdict"],
                       FROZEN["e281_verdict"]))
lc = m281["adjudication"]["reads"]["landing_curve"]
cons = {}
for arm, row in lc.items():
    cons[arm] = {
        "class": CONS_CLASS[arm],
        "width": row["width"],
        "post_g0_seed": row["post_g0_seed"],
        "landing": row["root_g0_landing"],
        "disp_l2": row["seed_displacement_l2"],
        "in_band": row["in_band"],
    }
for arm, v in FROZEN["landings"].items():
    binds.append(check_val("G_LITERAL", "landing[%s]" % arm, cons[arm]["landing"], v))
for arm, v in FROZEN["seed_post_g0"].items():
    binds.append(check_val("G_LITERAL", "seed_post[%s]" % arm, cons[arm]["post_g0_seed"], v))
for arm, v in FROZEN["seed_displacement_l2"].items():
    binds.append(check_val("G_LITERAL", "seed_disp[%s]" % arm, cons[arm]["disp_l2"], v))
e264_post = {int(k): v for k, v in
             m281["gates"]["G_PARENTS"]["hardbound"]["e264_curve_post"].items()}
e264_root = {int(k): v for k, v in
             m281["gates"]["G_PARENTS"]["hardbound"]["e264_curve_root"].items()}
for k, v in FROZEN["e264_curve_post"].items():
    binds.append(check_val("G_LITERAL", "e264_post[%s]" % k, e264_post[int(k)], v))
for k, v in FROZEN["e264_curve_root"].items():
    binds.append(check_val("G_LITERAL", "e264_root[%s]" % k, e264_root[int(k)], v))

# --- extract + bind: e268/e270 geometry + e179 law ---
for run, m in (("e268", m268), ("e270", m270)):
    arms = m["arms"]
    for arm, tag in (("SERIAL", "SERIAL"), ("CONCURRENT", "CONCURRENT")):
        binds.append(check_val("G_LITERAL", "%s_%s_install_in_room" % (run, tag),
                               arms[arm]["install"]["displacement_loads"]["in_own_room"],
                               FROZEN[run]["%s_install_in_room" % tag]))
        binds.append(check_val("G_LITERAL", "%s_%s_root_in_room" % (run, tag),
                               arms[arm]["root"]["displacement_loads"]["in_own_room"],
                               FROZEN[run]["%s_root_in_room" % tag]))
        binds.append(check_val("G_LITERAL", "%s_%s_root_g0" % (run, tag),
                               arms[arm]["root"]["g0"], FROZEN[run]["%s_root_g0" % tag]))
binds.append(check_val("G_LITERAL", "e179_resurrection_gm12_at50",
                       m179["cells"]["r1_32"]["cells_full"]["50"]["gm12"],
                       FROZEN["e179_resurrection_gm12_at50"]))
binds.append(check_val("G_LITERAL", "e179_wash_control_gm12_at50",
                       m179["cells"]["r0_wash"]["cells_full"]["50"]["gm12"],
                       FROZEN["e179_wash_control_gm12_at50"]))

MET["gates"] = {
    "G_PARENTS": {"per_parent": g_parents, "pass": True},
    "G_LITERAL": {
        "form": "every extracted number value-bound bit-exact against the "
                "design-time frozen snapshot (a drift HALTS)",
        "n_binds": len(binds), "max_abs_diff": max(b["abs_diff"] for b in binds),
        "pass": True,
    },
}
MET["builds_on"] = [
    "T263/x14 + e299 (the subtraction surgery: Arm A resurrects the dead to a "
    "mass-tracked fraction of the ORIGINAL baseline; e299's G_PORTX14 |d| 0.0 "
    "instrument check; the 7-corpse 3-class age-0 table this cell's surgery leg IS)",
    "W045/e281 (the rehearsal dose-response: FLAT — the cons teaches from "
    "ANYTHING; the no-write zero point lands 0.6508 in the hard band; the "
    "five-point one-session curve this cell's cons leg IS)",
    "T246/e268 + T248/e270 (the parent cons landings: 0.7119/0.8103 root g0; "
    "their displacement-load ledgers — the landing states' in_own_room geometry)",
    "e179 (the founding law: one replay at +32 resurrects the +2-dead fact, "
    "ruler gm12 0.6862 at +50 — family-1 g-12 battery, cross-era units uncertified)",
    "e264 (the matched-state dual readings: post and root g0 at widths "
    "1k/10k/100k — the units wall's only committed evidence, hardbound in e281)",
]
MET["whats_new"] = [
    "THE TWO ROUTES ON ONE TABLE FOR THE FIRST TIME: the surgery leg and the "
    "cons leg adjudicated against each other per corpse CLASS — the revival-vs-"
    "replacement question e299 and e281 each beggared alone",
    "THE UNITS WALL, WORKED: the inventory test (no cross-battery datum exists "
    "anywhere in the committed record), the proportionality test (post/root "
    "varies 3 orders across widths — no conversion constant), and the "
    "pseudo-conversion with its assumptions named — the value leg's "
    "UNITS-BARRED status established from the record, not assumed",
    "THE SEESAW DISCRIMINATOR: the identity question shown to be decidable "
    "WITHIN batteries (the zero-point control; the construction identity) even "
    "while the value question is barred across them",
]
write_met("P0 DONE (parents md5-bound, %d literal binds bit-exact) — no compute beyond extraction" % len(binds))

# ============================================================================
# P1 — (a) the two-route table per corpse class
# ============================================================================
free_keys = [k for k in surgery if SURGERY_CLASS[k] == "free-adamw"]
orth_keys = [k for k in surgery if SURGERY_CLASS[k] == "orthogonal-sgdm"]
maint_keys = [k for k in surgery if SURGERY_CLASS[k] == "orthogonal+in-room-maint"]

def _mean(xs):
    return float(np.mean(xs))

free_frac = _mean([surgery[k]["fraction"] for k in free_keys])
orth_frac = _mean([surgery[k]["fraction"] for k in orth_keys])
all_fracs = [surgery[k]["fraction"] for k in surgery]
surgery_range = max(all_fracs) / min(all_fracs)
surgery_class_split = orth_frac / free_frac

cons_land = {a: cons[a]["landing"] for a in cons}
cons_range = max(cons_land.values()) / min(cons_land.values())
seeded = ["R10", "K1KM", "K10KD", "K100KD"]
seeded_mean = _mean([cons_land[a] for a in seeded])
zero_gap = cons_land["NOINST"] / seeded_mean

two_route = {
    "form": "(a) the two routes per corpse class — native units per battery; "
            "the class rows are CLASS rows (no single corpse state has both "
            "routes in the committed record — disclosed)",
    "units": {
        "route_S_surgery": "recovered fraction of the loaded POST-g0 baseline "
                           "%.10f (e299's convention)" % FROZEN["baseline_post_g0"],
        "route_C_cons": "the landing read in the ROOT-g0 battery, family hard "
                        "band [0.65, 0.82] (e281's convention)",
        "common_unit_exists": False,
    },
    "rows": [
        {
            "corpse_class": "free-AdamW (the concurrent/interleave kills)",
            "route_S_surgery": {
                "corpses": free_keys,
                "fractions_of_post_baseline": {k: surgery[k]["fraction"] for k in free_keys},
                "factors_over_dead": {k: surgery[k]["factor"] for k in free_keys},
                "class_mean_fraction": free_frac,
            },
            "route_C_cons": {
                "corpses": ["K10KD = e268's 10k concurrent post (4.004e-05)",
                            "K100KD = e270's 100k concurrent post (0.0109)"],
                "landings_root_g0": {"K10KD": cons_land["K10KD"],
                                     "K100KD": cons_land["K100KD"]},
                "in_band": True,
                "note": "same CLASS, DIFFERENT STATES than the surgery column — "
                        "the record's only overlap is at class level",
            },
        },
        {
            "corpse_class": "orthogonal-sgdm (the room-preserving kills)",
            "route_S_surgery": {
                "corpses": orth_keys,
                "fractions_of_post_baseline": {k: surgery[k]["fraction"] for k in orth_keys},
                "class_mean_fraction": orth_frac,
            },
            "route_C_cons": None,
            "gap": "NO COMMITTED CONS DATUM — no orthogonal-class corpse has "
                   "ever been cons-taught in the committed record (e281's arms "
                   "are the width ladder + concurrent kills; e179 is family-1, "
                   "pre-rooms); the column is empty BY RECORD, disclosed",
        },
        {
            "corpse_class": "orthogonal+in-room-maint (e288's third class)",
            "route_S_surgery": {
                "corpses": maint_keys,
                "fractions_of_post_baseline": {k: surgery[k]["fraction"] for k in maint_keys},
            },
            "route_C_cons": None,
            "gap": "NO COMMITTED CONS DATUM (same record gap as above)",
        },
        {
            "corpse_class": "controls (not corpses — the cons route's own floor)",
            "route_S_surgery": None,
            "route_C_cons": {
                "NOINST_no_write_at_all": {"landing": cons_land["NOINST"],
                                           "seed_disp_l2": 0.0, "in_band": True},
                "R10_rank10_dead": {"landing": cons_land["R10"]},
                "K1KM_1k_dead": {"landing": cons_land["K1KM"]},
            },
        },
    ],
    "seesaw_contrast": {
        "surgery_output_range_over_corpses": surgery_range,
        "surgery_class_split_orth_over_free": surgery_class_split,
        "cons_output_range_incl_no_write": cons_range,
        "cons_zero_point_over_seeded_mean": zero_gap,
        "reading": "the surgery's product is a function of the corpse (7.08x "
                   "across corpses, 6.6x across classes); the cons's product "
                   "varies 1.24x across a width sweep that includes NO WRITE "
                   "AT ALL — and the no-write control reaches %.1f%% of the "
                   "seeded mean" % (100 * zero_gap),
    },
}
MET["two_route_table"] = two_route
write_met("P1 DONE (the two-route table, native units)")

# ============================================================================
# P2 — the units work (inventory, proportionality, pseudo-conversion)
# ============================================================================
# (1) inventory test: does any committed record read either route's product
#     in the other's battery?
inv = {
    "cons_products_read_in_post_battery": False,
    "evidence": "e281's landing_curve records root_g0_landing only; e268/e270's "
                "root blocks record the root battery (g0/gm12) + displacement "
                "loads — no post-g0 (write-battery) read of ANY cons landing "
                "state exists in the committed record",
    "surgery_products_read_in_root_battery": False,
    "evidence_2": "e299's arm_A reads are post-battery only (g0/gm12/gp12/ce_r "
                  "of the write battery); no root-g0 read of any theta0+P(delta) "
                  "state exists in the committed record",
    "e179_cross_era": "e179's law is recorded in the family-1 g-12 ruler "
                      "(install-60 battery) — a THIRD battery, pre-rooms; its "
                      "0.6862 is numerically inside the modern band [0.65,0.82] "
                      "but that coincidence is uncertified across eras — texture "
                      "only, never a conversion input",
}
# (2) proportionality test
ratios = {w: e264_post[w] / e264_root[w] for w in sorted(e264_post)}
prop = {
    "post_over_root_at_matched_widths": ratios,
    "span_orders": math.log10(max(ratios.values()) / min(ratios.values())),
    "constant_of_proportionality": None,
    "reading": "post/root runs 7.5e-4 -> 0.343 -> 0.599 across 1k/10k/100k — "
               "the batteries are NON-LINEARLY related; no single conversion "
               "factor exists even at fixed family; a width-matched conversion "
               "is an assumption, not a measurement",
}
# (3) the pseudo-conversion (non-binding context, assumptions named)
conv = {
    "K10KD_over_alive_root_anchor_10k": cons_land["K10KD"] / e264_root[10000],
    "K100KD_over_alive_root_anchor_100k": cons_land["K100KD"] / e264_root[100000],
    "assumptions": [
        "A1: the denominator is e264's SERIAL rung alive root read — a "
        "different fact instance's alive anchor (the concurrent arms' own "
        "alive root anchors were never recorded)",
        "A2: a 'revived write' should express in the root battery at its "
        "alive-anchor level — no committed cross-battery test of this exists",
        "A3: battery proportionality at matched width — contradicted across "
        "widths by the proportionality test (A3 is the load-bearing "
        "assumption and it is unbound)",
    ],
    "why_it_cannot_discriminate": "the zero-point control has NO denominator "
                                  "(no write exists to anchor) — the pseudo-"
                                  "conversion cannot even be stated for the one "
                                  "arm that decides the identity question; a "
                                  "construction and a revival convert identically",
    "status": "NON-BINDING CONTEXT — the native tables are the deliverable",
}
MET["units_work"] = {
    "form": "(b, units) the conversion worked three ways, assumptions disclosed",
    "inventory_test": inv,
    "proportionality_test": prop,
    "pseudo_conversion": conv,
    "verdict_on_the_value_leg": "UNITS-BARRED — the routes' outputs cannot be "
                                "compared in value from the committed record; "
                                "both tables stand natively per battery",
}
write_met("P2 DONE (the units wall: inventory + proportionality + pseudo-conversion)")

# ============================================================================
# P3 — (b) the geometry from committed records
# ============================================================================
geometry = {
    "form": "(b, geometry) the two routes' products in the write's own room — "
            "all numbers committed records, no state loaded",
    "route_S_surgery": {
        "per_corpse": {k: {"surviving_in_room_mass_ratio": surgery[k]["mass_ratio"],
                           "cos_surviving_vs_write_in_room": surgery[k]["cos_surviving"],
                           "drift_in_room_frac": surgery[k]["drift_in_room_frac"]}
                       for k in surgery},
        "construction_identity": "Arm A == theta0 + P(delta) with ID closure "
                                 "0.0 (fp64) on every corpse — the product IS "
                                 "the original write plus the corpse's surviving "
                                 "in-room drift; the read is the ORIGINAL "
                                 "write's read to a mass-tracked fraction "
                                 "(e299 rho +0.883)",
        "reading": "free class: mass 0.843/0.844 survives in-room at cos 0.993 "
                   "to the write's direction, read restores to 0.148/0.153; "
                   "orthogonal: mass ~1.000 survives at cos ~1.0, read restores "
                   "to ~1.0000 — the surgery reads out WHAT THE CORPSE KEPT",
    },
    "route_C_cons": {
        "sources": "e268/e270's committed displacement ledgers (the parent "
                   "sessions of e281's K10KD/K100KD anchors); e281's own record "
                   "carries seed displacement L2 only — its own-room reads were "
                   "vs the R10 room only (its disclosure, carried)",
        "in_own_room": {
            "the_write_itself_serial_installs": {"e268": FROZEN["e268"]["SERIAL_install_in_room"],
                                                 "e270": FROZEN["e270"]["SERIAL_install_in_room"]},
            "free_class_corpse_installs_concurrent": {"e268": FROZEN["e268"]["CONCURRENT_install_in_room"],
                                                      "e270": FROZEN["e270"]["CONCURRENT_install_in_room"]},
            "cons_landings_from_ALIVE_seeds": {"e268": FROZEN["e268"]["SERIAL_root_in_room"],
                                               "e270": FROZEN["e270"]["SERIAL_root_in_room"]},
            "cons_landings_from_CORPSE_seeds": {"e268": FROZEN["e268"]["CONCURRENT_root_in_room"],
                                                "e270": FROZEN["e270"]["CONCURRENT_root_in_room"]},
            "chance_occupancy_k10k": FROZEN["k10k_chance_occupancy"],
        },
        "ranking_fact": "the landing states do NOT rank by the write's "
                        "presence: corpse-seeded landings sit MORE in-room than "
                        "alive-seeded ones in BOTH records (0.584 > 0.413; "
                        "0.690 > 0.648) — if the landing were the write revived, "
                        "the alive-seeded landing (starting from the full "
                        "94.4%-in-room write) should be the more write-like; it "
                        "is the less. The landing geometry is the cons's own "
                        "signature on any substrate",
        "the_zero_point": "NOINST (no write, no room, displacement 0.0) lands "
                          "0.6508 in-band — room occupancy is not necessary "
                          "for the landing read at all",
    },
    "e179_law_record": {
        "event": "r=1/32: the first replay at +32 resurrects the +2-dead fact, "
                 "ruler gm12 %.4f at +50 (the wash control's corpse reads %.4f "
                 "at the same milestone — a %.0fx behavioral jump)"
                 % (FROZEN["e179_resurrection_gm12_at50"],
                    FROZEN["e179_wash_control_gm12_at50"],
                    FROZEN["e179_resurrection_gm12_at50"] / FROZEN["e179_wash_control_gm12_at50"]),
        "battery": "family-1 g-12 ruler (install-60 battery) — cross-era, "
                   "cross-battery vs both modern routes",
        "scope_note": "e179's corpse was TWO STEPS dead under wash (a fresh "
                      "kill), not a t400 corpse; e281's zero point is the "
                      "modern, controlled form of the same question — and it "
                      "lands the same band with NO corpse at all",
    },
}
MET["geometry"] = geometry
write_met("P3 DONE (geometry from committed records)")

# ============================================================================
# P4 — (c) THE SEESAW VERDICT (frozen decision rule)
# ============================================================================
band_lo, band_hi = FROZEN["band"]
zero_in_band = band_lo <= cons_land["NOINST"] <= band_hi
all_five_in_band = all(band_lo <= v <= band_hi for v in cons_land.values())
cons_flat_committed = m281["adjudication"]["verdict"] == "FLAT"
surgery_identity = all(surgery[k]["id_closure"] == 0.0 for k in surgery)
surgery_corpse_dependent = surgery_class_split > 2.0 and \
    m299["curve"]["correlations"]["primary_spearman_fraction_vs_mass"]["rho"] >= 0.8
cross_battery_datum = inv["cons_products_read_in_post_battery"] or \
    inv["surgery_products_read_in_root_battery"]

reads = {
    "cons_zero_point_in_band": zero_in_band,
    "cons_all_five_in_band": all_five_in_band,
    "cons_verdict_FLAT_committed": cons_flat_committed,
    "surgery_construction_identity_all_corpses": surgery_identity,
    "surgery_class_split_orth_over_free": surgery_class_split,
    "surgery_mass_tracks_rho": FROZEN["e299_rho_fraction_vs_mass"],
    "cross_battery_reconciliation_datum_exists": cross_battery_datum,
    "value_leg": "UNITS-BARRED (P2)",
    "geometry_ranking_fact": geometry["route_C_cons"]["ranking_fact"],
}

# frozen rule
if cross_battery_datum:
    verdict = "TWO-DOORS-ONE-MEMORY (requires the datum to agree — it does not exist; unreachable)"
    branch = "no cross-battery datum exists; this bar cannot fire from the record"
elif zero_in_band and cons_flat_committed and surgery_identity and surgery_corpse_dependent:
    verdict = "TWO-MEMORIES"
    branch = ("both identity discriminators are WITHIN-battery: (i) the cons "
              "landing is corpse-independent by its own record — e281's FLAT "
              "verdict with the NO-WRITE zero point landing 0.6508 inside the "
              "hard band (a landing from a state with no memory cannot be a "
              "revival of one; every corpse lands in the same band, and the "
              "parent geometry ranks corpse-seeded landings ABOVE alive-seeded "
              "in-room); (ii) the surgery's product is the original write's "
              "read by construction (theta0 + P(delta), ID closure 0.0) and by "
              "behavior (7.08x corpse-dependent, mass-tracked rho +0.883) — "
              "the value leg is UNITS-BARRED and the verdict never needed it")
elif not (zero_in_band and cons_flat_committed and surgery_identity and surgery_corpse_dependent):
    verdict = "UNITS-BARRED"
    branch = "identity discriminators missing from the record; native tables verbatim"
else:
    verdict = "MIXED"
    branch = "named branches reported verbatim"

# the founding-law reconciliation (the dispatch's e179-vs-x14 clause)
founding = {
    "e179_resurrection": "one cons replay at +32 restores a +2-step-dead fact's "
                         "ruler to 0.6862 (family-1 g-12 battery) — the event "
                         "the law names",
    "x14_e299_resurrection": "subtraction restores the t400 corpse's WRITE read "
                             "4328x to 14.8% of ITS OWN baseline (post battery) "
                             "— mass-tracked, corpse-dependent",
    "clause": "different events sharing a name: e179's is the cons's standard "
              "construction arriving on a fresh substrate (e281's zero point "
              "is the controlled form: the same landing with NO memory "
              "present); x14/e299's is the original write's read recovered "
              "from the corpse's own surviving mass. The seesaw does not "
              "tilt: the two 'resurrections' were never the same operation",
}

MET["adjudication"] = {
    "bars_verbatim": MET["registration"]["bars_verbatim"],
    "reads": reads,
    "verdict": verdict,
    "branch": branch,
    "clause": ("THE CORPSE SEESAW SETTLES TWO-MEMORIES: the surgery restores "
               "the ORIGINAL write's read (free 0.148/0.153 vs orthogonal "
               "1.0000 of the post-g0 baseline — a function of the corpse, "
               "mass-tracked); the cons's landing is its OWN standard "
               "construction (0.65-0.81 regardless of the corpse — INCLUDING "
               "NO CORPSE AT ALL: the zero point lands in-band, and the "
               "landing geometry ranks corpse-seeded above alive-seeded "
               "in-room). e179's 'resurrection' and x14's 'resurrection' are "
               "different events sharing a name. THE VALUE LEG IS UNITS-BARRED "
               "(post-g0 vs root-g0; no cross-battery datum; matched-width "
               "anchors non-proportional by 3 orders) — the two-route table "
               "stands natively per battery, and the identity verdict is "
               "decided within batteries, where the records are clean"),
    "founding_law_reconciliation": founding,
    "disclosed_gaps": [
        "the orthogonal class carries NO cons datum (no orthogonal-class "
        "corpse has ever been cons-taught in the committed record) — the "
        "cons column is empty BY RECORD at exactly the class where surgery "
        "restores 1.0000",
        "the class rows match CLASS, not STATES: the surgery leg's free-class "
        "corpses (e283/e285) and the cons leg's free-class corpses (e268/e270 "
        "concurrent posts) are different states from the same kill class — "
        "no single corpse has both routes on record",
        "e179's law is cross-era (family-1, g-12 ruler, pre-rooms): cited as "
        "the founding event, never as a convertible datum",
    ],
}
write_met("P4 DONE (the seesaw verdict: %s)" % verdict)

# ============================================================================
# P5 — figure + report + honesty + provenance
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
fig.suptitle("e309 THE CORPSE SEESAW — verdict: %s (desk cell over committed records)"
             % verdict, fontsize=14, fontweight="bold")

# Panel 1: the surgery route (post battery)
ax = axes[0, 0]
order = free_keys + orth_keys + maint_keys
colors = {"free-adamw": "#c0392b", "orthogonal-sgdm": "#2471a3",
          "orthogonal+in-room-maint": "#7d3c98"}
xs = np.arange(len(order))
vals = [surgery[k]["fraction"] for k in order]
ax.bar(xs, vals, color=[colors[SURGERY_CLASS[k]] for k in order], edgecolor="black", linewidth=0.6)
ax.axhline(1.0, color="green", ls="--", lw=1.2)
ax.axhline(0.0, color="black", lw=0.8)
for x, v in zip(xs, vals):
    ax.text(x, v + 0.02 if v < 0.9 else v - 0.09, "%.4f" % v if v < 0.9 else "%.4f" % v,
            ha="center", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels([k.replace("-", "\n") for k in order], fontsize=7.5)
ax.set_ylabel("recovered fraction of POST-g0 baseline 0.2646")
ax.set_title("ROUTE S — subtraction surgery (e299/x14)\nthe product is a function of the CORPSE (7.08x range)", fontsize=10)
ax.set_ylim(0, 1.18)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
ax.legend(handles, ["free-AdamW (~0.15)", "orthogonal (~1.00)", "orth+maint (1.05)"],
          fontsize=8, loc="lower right")

# Panel 2: the cons route (root battery)
ax = axes[0, 1]
w = [cons[a]["width"] for a in ["NOINST", "R10", "K1KM", "K10KD", "K100KD"]]
lv = [cons_land[a] for a in ["NOINST", "R10", "K1KM", "K10KD", "K100KD"]]
ax.axhspan(band_lo, band_hi, color="#d5f4e6", alpha=0.8, label="landing band [0.65, 0.82]")
ax.plot([max(x, 0.5) for x in w[1:]], lv[1:], "o-", color="#2471a3", ms=6, label="seeded arms (dead writes)")
ax.plot([0.5], [lv[0]], "*", color="#c0392b", ms=22, label="NOINST — NO WRITE AT ALL")
ax.annotate("zero point lands IN-BAND\n(no memory to revive)", xy=(0.5, lv[0]),
            xytext=(3, 0.535), fontsize=9, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b"))
for x, y, a in zip([max(v, 0.5) for v in w], lv, ["NOINST", "R10", "K1KM", "K10KD", "K100KD"]):
    ax.text(x, y + 0.008, "%s\n%.4f" % (a, y), ha="center", fontsize=7.5)
ax.set_xscale("log")
ax.set_xlabel("seed width (log) — the corpse axis degenerates to NONE at w=0")
ax.set_ylabel("landing read, ROOT-g0 battery")
ax.set_title("ROUTE C — the cons re-teach (e281)\nthe product is corpse-INDEPENDENT by its own FLAT record (1.24x range)", fontsize=10)
ax.set_ylim(0.52, 0.86)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(0.3, 3e5)

# Panel 3: the units wall
ax = axes[1, 0]
ws = sorted(ratios)
ax.plot(ws, [ratios[x] for x in ws], "s-", color="#7f8c8d", ms=8)
ax.set_xscale("log"); ax.set_yscale("log")
for x in ws:
    ax.annotate("post/root = %.2e" % ratios[x], xy=(x, ratios[x]),
                xytext=(x * 1.3, ratios[x] * 1.7), fontsize=8)
ax.set_xlabel("write width (the e264 serial rungs — the only matched-state dual readings)")
ax.set_ylabel("post-g0 / root-g0 at the SAME state")
ax.set_title("THE UNITS WALL — no constant of proportionality\n(the batteries are non-linearly related: 3 orders across widths)", fontsize=10)

# Panel 4: the geometry
ax = axes[1, 1]
cats = ["the write\n(serial installs)", "free-class corpse\ninstalls",
        "cons landing\nfrom ALIVE seed", "cons landing\nfrom CORPSE seed"]
e268v = [FROZEN["e268"]["SERIAL_install_in_room"], FROZEN["e268"]["CONCURRENT_install_in_room"],
         FROZEN["e268"]["SERIAL_root_in_room"], FROZEN["e268"]["CONCURRENT_root_in_room"]]
e270v = [FROZEN["e270"]["SERIAL_install_in_room"], FROZEN["e270"]["CONCURRENT_install_in_room"],
         FROZEN["e270"]["SERIAL_root_in_room"], FROZEN["e270"]["CONCURRENT_root_in_room"]]
xg = np.arange(len(cats))
ax.bar(xg - 0.18, e268v, 0.34, label="e268 (10k)", color="#2471a3", edgecolor="black", lw=0.5)
ax.bar(xg + 0.18, e270v, 0.34, label="e270 (100k)", color="#5dade2", edgecolor="black", lw=0.5)
ax.axhline(FROZEN["k10k_chance_occupancy"], color="red", ls=":", lw=1.5)
ax.text(3.35, 0.05, "chance occupancy\nk/N = 0.0037", color="red", fontsize=8)
for x, v in zip(xg - 0.18, e268v):
    ax.text(x, v + 0.015, "%.3f" % v, ha="center", fontsize=7.5)
for x, v in zip(xg + 0.18, e270v):
    ax.text(x, v + 0.015, "%.3f" % v, ha="center", fontsize=7.5)
ax.set_xticks(xg); ax.set_xticklabels(cats, fontsize=8)
ax.set_ylabel("displacement in_own_room (fraction of L2)")
ax.set_title("THE GEOMETRY — landing states do NOT rank by the write's presence\n(corpse-seeded MORE in-room than alive-seeded, both records)", fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(0, 1.12)

fig.text(0.5, 0.008,
         "ROUTE S reads the POST-g0 battery (fractions of 0.2646); ROUTE C reads the ROOT-g0 battery (band units). "
         "THE PANELS SHARE NO Y-AXIS — the value leg is UNITS-BARRED; each route discriminates within its own battery.",
         ha="center", fontsize=8.5, style="italic", color="#555555")
fig.tight_layout(rect=[0, 0.02, 1, 0.965])
fig.savefig(RUN / "e309_seesaw.png", dpi=140)
plt.close(fig)

# --- REPORT.md ---
def fmt(v, n=4):
    return ("%." + str(n) + "f") % v

lines = []
lines.append("# E309 — THE CORPSE SEESAW (desk cell over committed records)")
lines.append("")
lines.append("**VERDICT: %s**" % verdict)
lines.append("")
lines.append("e179's law says one cons replay resurrects the dead; e299/x14 say subtraction "
             "resurrects the dead. Both routes, one table, same corpse classes: **the surgery "
             "restores the ORIGINAL write's read; the cons builds its own standard construction "
             "on any substrate** — e179's 'resurrection' and x14's 'resurrection' are different "
             "events sharing a name. The value comparison between the routes is UNITS-BARRED "
             "(different batteries); the identity verdict is decided within batteries.")
lines.append("")
lines.append("## (a) The two-route table, native units per battery")
lines.append("")
lines.append("| corpse class | ROUTE S — surgery (e299/x14): fraction of post-g0 baseline %.4f | ROUTE C — cons (e281 + parents): landing, root-g0, band [0.65, 0.82] |" % FROZEN["baseline_post_g0"])
lines.append("|---|---|---|")
row0 = ("| **free-AdamW** (concurrent kills) | e283 **%.4f** (%.0fx over dead), e285-TWIN **%.4f** (%.0fx); class mean %.4f | K10KD (e268 10k concurrent) **%.4f**, K100KD (e270 100k concurrent) **%.4f** — in band; same CLASS, different STATES |"
        % (surgery["e283-CONCURRENT"]["fraction"], surgery["e283-CONCURRENT"]["factor"],
           surgery["e285-TWIN"]["fraction"], surgery["e285-TWIN"]["factor"], free_frac,
           cons_land["K10KD"], cons_land["K100KD"]))
row1 = ("| **orthogonal-sgdm** (4 corpses) | %.4f / %.4f / %.7f / %.7f — **1.0000** restored | **NO COMMITTED CONS DATUM** (the gap: no orthogonal corpse ever cons-taught on record) |"
        % (surgery["e285-SANCTUARY"]["fraction"], surgery["e290-0.1X"]["fraction"],
           surgery["e290-0.02X"]["fraction"], surgery["e290-0.004X"]["fraction"]))
row2 = ("| **orthogonal + in-room-maint** (e288) | **%.4f** | NO COMMITTED CONS DATUM |"
        % surgery["e288-NAME-FIXED-TWIN"]["fraction"])
row3 = ("| controls (not corpses) | — | NOINST (**no write at all**) **%.4f** in-band; R10 %.4f; K1KM %.4f |"
        % (cons_land["NOINST"], cons_land["R10"], cons_land["K1KM"]))
lines += [row0, row1, row2, row3]
lines.append("")
lines.append("**The seesaw contrast:** the surgery's output is a function of the corpse — %.2fx range "
             "across corpses, %.1fx between classes; the cons's output varies %.2fx across a width sweep "
             "that includes NO WRITE AT ALL, and the no-write control reaches %.0f%% of the seeded mean."
             % (surgery_range, surgery_class_split, cons_range, 100 * zero_gap))
lines.append("")
lines.append("## (b) The units disclosure — why the value leg is UNITS-BARRED")
lines.append("")
lines.append("1. **Inventory test:** no committed record reads ANY cons landing state in the post-g0 "
             "(write) battery, and no committed record reads ANY surgery product (theta0+P(delta)) in "
             "the root battery. The routes have literally never been measured in a common unit.")
lines.append("2. **Proportionality test:** the only matched-state dual readings on record (e264's serial "
             "rungs) give post/root = %.2e / %.2f / %.2f at widths 1k/10k/100k — three orders of "
             "variation; there is no conversion constant, and width-matching is an assumption, not a "
             "measurement." % (ratios[1000], ratios[10000], ratios[100000]))
lines.append("3. **The pseudo-conversion (non-binding context):** landing / alive-root-anchor = "
             "%.4f (K10KD) and %.4f (K100KD) — 'full-looking' numbers under assumptions A1-A3 "
             "(named in metrics.units_work). It cannot discriminate: the zero-point control has no "
             "denominator, and a construction converts identically to a revival."
             % (conv["K10KD_over_alive_root_anchor_10k"], conv["K100KD_over_alive_root_anchor_100k"]))
lines.append("")
lines.append("Both tables therefore stand NATIVELY — the UNITS-BARRED branch's native-table clause, "
             "applied to the value leg. The identity question never needed the conversion.")
lines.append("")
lines.append("## (b) The geometry")
lines.append("")
lines.append("- **Surgery side (e299):** free-class corpses keep %.3f/%.3f of the write's in-room mass at "
             "cos %.4f/%.4f to the write's direction, and the surgery restores %.3f/%.3f of the original "
             "read; orthogonal corpses keep ~1.000 of the mass at cos ~1.0, and the surgery restores "
             "~1.0000. The product is theta0 + P(delta) (ID closure 0.0 on every corpse) — it reads out "
             "WHAT THE CORPSE KEPT, mass-tracked (rho +0.883)."
             % (surgery["e283-CONCURRENT"]["mass_ratio"], surgery["e285-TWIN"]["mass_ratio"],
                surgery["e283-CONCURRENT"]["cos_surviving"], surgery["e285-TWIN"]["cos_surviving"],
                surgery["e283-CONCURRENT"]["fraction"], surgery["e285-TWIN"]["fraction"]))
lines.append("- **Cons side (e268/e270, the parents of the anchors):** the write itself sits %.3f/%.3f "
             "in-room; the corpse-seeded landings sit %.3f/%.3f; the ALIVE-seeded landings sit %.3f/%.3f. "
             "**The landing states do not rank by the write's presence** — corpse-seeded landings are MORE "
             "in-room than alive-seeded in both records. If the landing were the write revived, the "
             "alive-seeded landing (starting from the full 94%%-in-room write) should be the more "
             "write-like; it is the less. The landing geometry is the cons's own signature."
             % (FROZEN["e268"]["SERIAL_install_in_room"], FROZEN["e270"]["SERIAL_install_in_room"],
                FROZEN["e268"]["CONCURRENT_root_in_room"], FROZEN["e270"]["CONCURRENT_root_in_room"],
                FROZEN["e268"]["SERIAL_root_in_room"], FROZEN["e270"]["SERIAL_root_in_room"]))
lines.append("- **The zero point:** NOINST has no write, no room, displacement 0.0 — and lands %.4f "
             "in-band. Room occupancy is not necessary for the landing read at all." % cons_land["NOINST"])
lines.append("")
lines.append("## (c) The seesaw verdict")
lines.append("")
lines.append(branch[0].upper() + branch[1:] if branch else branch)
lines.append("")
lines.append("**The founding-law reconciliation:** e179's resurrection (one replay at +32 restores a "
             "+2-step-dead fact's ruler to %.4f, family-1 g-12 battery) is the cons's standard "
             "construction arriving on a fresh substrate — e281's zero point is its controlled form: "
             "the same landing with NO memory present. x14/e299's resurrection is the original write's "
             "read recovered from the corpse's own surviving mass. Different events sharing a name; "
             "the seesaw does not tilt."
             % FROZEN["e179_resurrection_gm12_at50"])
lines.append("")
lines.append("## Disclosed gaps and caveats")
lines.append("")
lines.append("- The **orthogonal class has no cons datum anywhere in the committed record** — the cons "
             "column is empty BY RECORD at exactly the class where surgery restores 1.0000. Filling it "
             "(a cons run from an e290/e285-SANCTUARY corpse) is the obvious successor cell.")
lines.append("- The class rows match CLASS, not STATES: the free-class surgery corpses (e283/e285) and "
             "the free-class cons corpses (e268/e270 concurrent posts) are different states of the same "
             "kill class; no single corpse has both routes on record.")
lines.append("- e179's law is cross-era (family-1, g-12 ruler, pre-rooms) — cited as the founding event, "
             "never as a convertible datum. Its 0.6862 is numerically inside the modern band; that "
             "coincidence is uncertified across eras (texture only).")
lines.append("- e281's own record carries seed displacement L2 only; its own-room reads were vs the R10 "
             "room only (its own disclosure). The landing-occupancy numbers are the PARENT sessions' "
             "committed ledgers — the closest committed geometry.")
lines.append("- n=1 per corpse per route, one session each, one lineage; this cell loads no states and "
             "runs no compute beyond extraction — every number is a committed record, value-bound "
             "bit-exact at birth (%d binds, max |d| %.1e)." % (len(binds), max(b["abs_diff"] for b in binds)))
lines.append("- No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).")
lines.append("")
lines.append("## Provenance")
lines.append("")
provs = []
try:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                          capture_output=True, text=True, check=True).stdout.strip()
    provs.append(("git_head_at_start", head))
except Exception as e:
    provs.append(("git_head_at_start", "unavailable: %s" % e))
MET["provenance"] = {
    "script": str(ROOT / "lab" / "e309_seesaw.py"),
    "birth_commit": "81f65113b22c7814919e9865b74c4249a3aac927 "
                    "(the script's bars-frozen registration commit)",
    "git_head_at_start": provs[0][1],
    "head_note": "HEAD at run start is NOT the birth commit — other fleet cells "
                 "committed between this cell's birth and its run; both hashes "
                 "recorded, no ambiguity (the registration is the birth commit)",
    "machinery": "NONE imported — a pure desk adjudication (json/hashlib/numpy/"
                 "matplotlib only); no torch, no cuda tensor, no state loads",
    "records_read": {k: {"path": v["path"], "md5": v["md5"]} for k, v in PARENTS.items()},
    "eval": {"device": "cpu", "threads": 4, "cuda_tensors_created": 0,
             "datetime": "datetime.now(UTC) only"},
    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                 "matplotlib": matplotlib.__version__},
}
for k, v in provs:
    lines.append("- **%s**: `%s`" % (k, v))
lines.append("- **birth_commit** (the bars-frozen registration): `81f65113b22c7814919e9865b74c4249a3aac927` — HEAD moved past it before the run (other fleet cells committed in between); the registration is the birth commit, the run started at the HEAD above")
lines.append("- parents md5-bound: " + ", ".join("%s (%s)" % (k, v["md5"][:8]) for k, v in PARENTS.items()))
lines.append("- outputs: `runs/e309/metrics.json`, `runs/e309/e309_seesaw.png`, this report")
lines.append("")
(RUN / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

MET["honesty"] = {
    "records_not_recomputations": "this cell loads no states and replays no "
                                  "history — every number is extracted from an "
                                  "md5-bound committed record and value-bound "
                                  "bit-exact against the design-time snapshot",
    "the_units_honesty": "the two routes' outputs live in different batteries "
                         "with no cross-battery datum and no conversion constant; "
                         "the value comparison is UNITS-BARRED and is REPORTED "
                         "NATIVELY — the pseudo-conversion is disclosed as "
                         "non-binding context with its three assumptions named",
    "the_identity_honesty": "the TWO-MEMORIES verdict rests on within-battery "
                            "discriminators only: the zero-point control (a "
                            "landing from a no-memory state cannot be a revival "
                            "of one) and the surgery's construction identity "
                            "(theta0 + P(delta)); neither requires the barred "
                            "conversion",
    "the_gap_honesty": "the orthogonal class's empty cons column is a RECORD "
                       "GAP, not a negative result — no orthogonal corpse has "
                       "ever been cons-taught; the successor cell is named",
    "n_and_scope": "n=1 per corpse per route, one session each, one lineage; "
                   "the class rows match classes, not states; nothing guaranteed",
}
MET["outputs"] = [str(RUN / "metrics.json"), str(RUN / "e309_seesaw.png"),
                  str(RUN / "REPORT.md")]
MET["phase_note"] = "P5 DONE (figure + report + honesty + provenance)"
write_met("COMPLETE — adjudicated (%s)" % verdict)
print("e309 DONE — verdict:", verdict)
print("surgery class split orth/free: %.2fx | cons range incl no-write: %.2fx | zero/seeded-mean: %.3f"
      % (surgery_class_split, cons_range, zero_gap))
