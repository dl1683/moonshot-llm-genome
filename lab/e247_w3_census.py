"""E247 — T222's registered follow-up: THE W3 FLIP CENSUS + THE LAYER-ORDER
MAP + THE WALKER READ (eval-only, CPU, threads 4; load-polite — e240 is CPU
live, the GPU lane untouched).

WHY (T222, verbatim registration): "REGISTERED (e247): the w3 flip census
fattens both sides (the non-P-FIRST side is n=4); the walker question at
finer states (how many transient mis-dials precede collapses between journal
states?); prediction: the order->mode map holds at w3 (non-P-FIRST ->
mis-dial; P-FIRST -> collapse-or-other) and the walkers are common (the
mis-dial is a STAGE most collapses pass through, visible only at fine
grids)." e245 found THE TWO DEATH MODES ARE THE TWO DEATH ORDERS on
w1/w2 (non-P-FIRST 4/4 mis-dial; collapses 7/7 P-FIRST) with the thin side
at n=4 — this cell tests the map on the THIRD certified wash (w3, e217's
fresh draw, seed 21703) and prices the walker texture at the finest
available state grid.

THE CELL (frozen from the dispatch brief):
  (1) THE 54 PROBES' MARGINS AND ARGMAXES AT W3'S STATES — w3's archived
      state grid (LISTED in metrics): journal steps {0, 10, 50, 80}; the
      committed checkpoints runs/checkpoints/e217_fresh_s{10,50,80}.pt
      (+ e217_fresh_latest.pt resumable; NO +2 state was archived for w3 —
      the w3 grid is w2-shaped, disclosed). Margins/argmaxes for w3 DO NOT
      EXIST in e228's journal (t0 + w1{2,10,50,80} + w2{10,50,80} only) —
      they are COMPUTED FRESH here with e228's margin_pass module-imported
      VERBATIM (the instrument: (top1-top2 logit)/std(vocab logits) at the
      answer position, same forward as the p re-probe), on the batteries
      module-imported VERBATIM from e182c_forgetting_control/e182c2_template
      — DISCLOSED as the dispatch requires. t0 is certified by reproducing
      e228's committed t0 journal record (G_T0REPRO); every w3 state by the
      re-probe convention vs e217's committed wash3_states p records
      (G_STATES, tol 0.005, e228/e226's convention; expected ~0 — e217's
      probes were CPU fp32 too).
  (2) THE W3 FLIP CENSUS — every (probe) record whose w3+80 argmax differs
      from t0's (the +80-argmax-differs convention, e243 verbatim;
      double-sourced: my fresh top1 vs e217's committed answer-rank field).
      Each flip classified MIS-DIAL / FREQUENCY-COLLAPSE / OTHER by e243's
      rules IMPORTED VERBATIM (module import of e243_mode_selector's
      CLASS_RULES_VERBATIM + thresholds; first match wins; the z convention
      e241/e243 verbatim: cos on L2-normalized float64 pristine-wte rows,
      1000-id uniform null, ONE global default_rng(20261004); DRAW ORDER
      FROZEN: records in battery order fact/ctrl/near/tmpl x probe grid
      order, runner-up null first then target null per record). The t0
      runner-up identity from e228's committed t0 journal top2_id (gate:
      == the e238 npz t0 rank-2, e243's G_RU reproduced); the target's t0
      rank from the sha-gated e238 t0 logit dump.
  (3) THE LAYER-ORDER MAP — each w3 flip joined to its DEATH ORDER
      (P-FIRST / MARGIN-FIRST / TOGETHER / NEITHER): e230's classification
      EXTENDED to w3 from the fresh margins + e217's committed p records
      (the operationalization verbatim: margin event = first state with
      argmax margin_sigma < 0.05 [T204's flip zone, strictly below,
      absorbing]; p event = first state with p < 0.5*p_t0 [strictly below,
      absorbing]; classes per e230's probe-level rules). The extension code
      is VALIDATED by reproducing e230's committed w1/w2 probe-level
      classes EXACTLY (G_E230, 108 records — e232's cross-check
      convention). MAP READ + bars below.
  (4) THE WALKER READ (co-report, NO bar) — at the finest available state
      grid per wash (w1 {2,10,50,80}; w2 {10,50,80}; w3 {10,50,80}), how
      many records transiently mis-dial before collapsing (the Egypt
      pattern: w1 Egypt->Cairo visited ' Alexandria' at +50 then ' the' at
      +80). Operationalized, frozen: a flip record PASSED THROUGH A
      MIS-DIAL iff some state STRICTLY BEFORE +80 has argmax == the t0
      runner-up; the WALKER RATE = the fraction of FREQUENCY-COLLAPSE flip
      records that passed through a mis-dial, pooled over the three washes
      (per-wash co-reported). Co-reports, never adjudicating: the looser
      excursion read (any strictly-before state whose argmax is neither
      the answer nor the final target); the full argmax trajectory table.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any
compute; the registration commit of this file precedes any compute;
adjudicate against exactly this; no bar shopping; the bars adjudicate on
whatever n exists, disclosed):

  ORDER-MAP-HOLDS — "w3's flips follow the order->mode map (every
    non-P-FIRST flip mis-dials; the P-FIRST flips collapse or other; zero
    violations) AND the pooled n on the thin side grows — the unification
    replicates on a third wash"
  ORDER-MAP-BREAKS — "any violation — the map is w1/w2 texture; T222's
    unification bounded"
  MIXED — "the tables verbatim, no inflation"

OPERATIONALIZATIONS FROZEN BEFORE COMPUTE (they fix the clauses; they do
not move the bars):
  * flip := (probe) record with top1(w3,+80) != top1(t0) — endpoint read;
    transient flips that recover by +80 are not flips (e243's convention);
    n disclosed.
  * non-P-FIRST := death-order class in {MARGIN-FIRST, TOGETHER, NEITHER}
    (e245's "the commitment or the pair dying first" + the no-event class,
    disclosed if it occurs). thin side := the non-P-FIRST side (e245:
    n=4 = 3 TOGETHER + 1 MARGIN-FIRST). "The pooled n on the thin side
    grows" := w3 contributes >= 1 non-P-FIRST flip record (pooled n > 4).
  * THE MAP, TWO READINGS — frozen BEFORE compute, both reported, and the
    DISCLOSURE that motivates the choice: e245's committed w1/w2 table
    ALREADY contains 5 P-FIRST->MIS-DIAL records (US->dollar w1/w2,
    Boston w1, Atlanta w1/w2), so the dispatch's literal parenthetical
    ("the P-FIRST flips collapse or other") is STRICTER than the map as
    committed in e245/T222 (whose exact form is the ASYMMETRIC
    conjunction: non-P-FIRST => mis-dial 4/4 AND collapse => P-FIRST 7/7
    — P-FIRST mis-dials are allowed members of the committed map). The
    LITERAL reading ADJUDICATES (the frozen letter, the conservative
    test, e245's adjudicating-denominator precedent); the committed
    asymmetric map co-reports (disclosed, NOT adjudicating). Both
    readings' violation counts are reported verbatim per clause.
  * LITERAL violation := (a) any w3 non-P-FIRST flip whose classification
    != MIS-DIAL, OR (b) any w3 P-FIRST flip whose classification not in
    {FREQUENCY-COLLAPSE, OTHER}. COMMITTED-MAP violation := (a) OR (c)
    any w3 FREQUENCY-COLLAPSE flip whose death-order class != P-FIRST.
  * ladder (gated on G_T0REPRO/G_BATT/G_CORPUS/G_STATES/G_E230/G_WTE/
    G_IDS/G_ENV): any LITERAL violation -> ORDER-MAP-BREAKS; else if the
    thin side grew -> ORDER-MAP-HOLDS; else MIXED (zero violations but
    the thin side did not grow — the empty-stratum form, e243's
    pre-registered empty-stratum rule; if w3 has ZERO flips the verdict is
    MIXED with the empty census disclosed). Verification-gate failure ->
    tables reported, NO bar read (e228's precedent).
  * pooled table := e245's 19 committed records VERBATIM (read at
    runtime) + w3's records; pooled counts recomputed from that table.
  * walker population := the 19 committed flip records + w3's flip
    records; trajectories from e228's committed journal argmax records
    (w1/w2) and this cell's fresh w3 argmaxes; RU identity per probe from
    e228's committed t0 top2_id.
  * medians, where quoted: numpy median (linear interpolation at even n).

INPUTS (all COMMITTED, sha-gated at runtime, never re-derived):
  runs/e217/metrics.json (+journal.json) — w3's committed per-probe p
    records (wash3_states), the w3 provenance (seed 21703, e182c2
    machinery verbatim, GPU fp32 wash / CPU fp32 probes) and state grid.
  runs/checkpoints/e217_fresh_s{10,50,80}.pt — w3's committed states
    (sizes cross-checked vs e226's committed inventory; certified by the
    re-probe below — "bit-exact loads; the certified stream").
  runs/e228/journal.json — the margins+argmax journal (t0 shared record
    reproduced here; w1/w2 trajectories for the walker read).
  runs/e214/journal.json — the w1/w2 p journal (the e230-extension
    validation's p-side).
  runs/e230/metrics.json — e230's committed death-order probe-level
    records (the validation target; w1/w2 classes for the pooled map are
    e245's committed join, itself read not re-derived).
  runs/e238/logits_t0.npz (+journal sha) — the t0 full-logit dump (the
    t0 runner-up gate + the target's t0 rank).
  runs/e243/metrics.json — e243's committed 19-record table (z-join
    identity checks on overlapping records).
  runs/e245/metrics.json — e245's committed 19-record join (the pooled
    table's w1/w2 side, verbatim).
  the pristine shared-124M wte — tensor-only from the committed HF
    snapshot safetensors (openai-community/gpt2 @ 607a30d783dfa663caf39
    e06633721c8d4cfcd7e), sha-gated vs e241's committed wte sha.

COMPUTE ENVELOPE: CPU-only (no GPU calls — e237's lane untouched), torch
threads 4 (e228's deterministic path), load checks before launch and
between states, one state = one eval burst; ~54 probes x 4 state reads +
3 checkpoint loads; minutes. PROGRESSIVE metrics + resumable journal after
every state (the standing disruption rule). Nothing here is guaranteed.

PROVENANCE: the margin instrument is e228's margin_pass module-imported
VERBATIM; the batteries are e182c_forgetting_control/e182c2_template
module-imported VERBATIM (via e228); the classification rules/thresholds/
z-convention are e243's module-imported VERBATIM; the journal loaders and
the census/order-join conventions are e245's module-imported; the
death-order operationalization is e230's committed wording re-implemented
(the e230 script is a side-effecting script — NOT importable — so its
constants are QUOTED (FLIP_ZONE=0.05, P_HALF=0.5, e230 lines 149-150) and
the re-implementation is PROVEN equivalent by G_E230's exact reproduction
of e230's committed 108 records). Builds on: T222/e245 (the registration),
T221/e243 + T219/e241 (the mode taxonomy), T212/e232 + T209/e230 (the
death-order taxonomy), T207/e228 (the margin instrument), T192/e217 (the
third wash), T187/e214 + e182c/e182c2/e182 (the archive). NEW: w3's
margins/argmaxes (never read before); the w3 flip census; the order->mode
map tested on a third wash; the walker read at the finest available grid.

Run:  cd lab && python e247_w3_census.py   (E247_SMOKE=1: t0 + w3+10 only,
      own smoke dir, nothing adjudicated)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # pinned revision, local cache
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

# ---- module imports of the instruments/conventions (VERBATIM, never retyped)
import e228_margin_landscape as e228                # sets env, threads 4
import e182c_forgetting_control as e1               # noqa: F401 — via e228 too
import e182c2_template as e2                        # noqa: F401
import e243_mode_selector as e243m                  # classification rules VERBATIM
import e245_death_depth as e245m                    # journal loaders + census conventions

import torch                                            # noqa: E402

import common                                            # noqa: E402
from common import now_iso, run_dir, save_json            # noqa: E402

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

margin_pass = e228.margin_pass        # THE instrument, module-imported verbatim

SMOKE = os.environ.get("E247_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e247_smoke" if SMOKE else "e247"

T0 = time.time()
LOG_PATH = common.REPO / "runs" / (NAME + "_run.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- the frozen cell constants (registered BEFORE compute) --------------------
W3_STATES: tuple[int, ...] = (10,) if SMOKE else (10, 50, 80)
W3_CK = {s: common.REPO / "runs" / "checkpoints" / f"e217_fresh_s{s}.pt"
         for s in W3_STATES}
W3_LATEST = common.REPO / "runs" / "checkpoints" / "e217_fresh_latest.pt"
BATTERIES: tuple[str, ...] = ("fact", "ctrl", "near", "tmpl")
W1_W2_TRAJ = {"w1": (2, 10, 50, 80), "w2": (10, 50, 80)}   # e228 journal grids
W3_TRAJ = (10, 50, 80)

E217_M = common.REPO / "runs" / "e217" / "metrics.json"
E217_J = common.REPO / "runs" / "e217" / "journal.json"
E228_J = common.REPO / "runs" / "e228" / "journal.json"
E214_J = common.REPO / "runs" / "e214" / "journal.json"
E230_M = common.REPO / "runs" / "e230" / "metrics.json"
E238_J = common.REPO / "runs" / "e238" / "journal.json"
E238_NPZ_T0 = common.REPO / "runs" / "e238" / "logits_t0.npz"
E243_M = common.REPO / "runs" / "e243" / "metrics.json"
E245_M = common.REPO / "runs" / "e245" / "metrics.json"
E226_M = common.REPO / "runs" / "e226" / "metrics.json"

# e230's committed constants, QUOTED (the e230 script side-effects at import;
# G_E230 proves the equivalence by exact reproduction):
FLIP_ZONE = 0.05   # e230 line 149: T204's flip zone (strictly below, absorbing)
P_HALF = 0.5       # e230 line 150: p < P_HALF * p_t0 (strictly below, absorbing)

# registered tolerances (frozen)
TOL_T0_P = 1e-6        # my t0 p vs e228's committed t0 journal record
TOL_T0_MARGIN = 1e-4   # my t0 margin_sigma vs e228's committed record
TOL_STATE_DP = 0.005   # w3-state re-probe vs e217's committed p (e228/e226 conv)
TOL_PROBE_DP = 0.010   # battery t0 vs committed records (e214's TOL_PROBE_DP)

BARS_VERBATIM = {
    "ORDER-MAP-HOLDS": (
        "w3's flips follow the order->mode map (every non-P-FIRST flip "
        "mis-dials; the P-FIRST flips collapse or other; zero violations) "
        "AND the pooled n on the thin side grows — the unification "
        "replicates on a third wash"),
    "ORDER-MAP-BREAKS": (
        "any violation — the map is w1/w2 texture; T222's unification "
        "bounded"),
    "MIXED": "the tables verbatim, no inflation",
}
PREDICTION_T222_VERBATIM = (
    "the order->mode map holds at w3 (non-P-FIRST -> mis-dial; P-FIRST -> "
    "collapse-or-other) and the walkers are common (the mis-dial is a STAGE "
    "most collapses pass through, visible only at fine grids)")

trims: list[str] = []
deviations: list[str] = [
    "EVAL-ONLY on the committed checkpoints (no wash run here); CPU-only "
    "per the dispatch (threads 4, load-polite — e240 CPU live, GPU lane "
    "untouched).",
    "w3's margins/argmaxes DO NOT exist in e228's journal — COMPUTED FRESH "
    "with the module-imported instrument (DISCLOSED per the dispatch); t0 "
    "and every w3 state are certified against committed records (G_T0REPRO, "
    "G_STATES).",
    "e230's script side-effects at import (it re-runs its desk pass and "
    "rewrites runs/e230/metrics.json) — NOT importable; its FLIP_ZONE/P_HALF "
    "constants are QUOTED and the re-implemented extension is PROVEN "
    "equivalent by G_E230's exact reproduction of e230's committed 108 "
    "probe-level records on w1/w2.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch); the draft NOTES entry "
    "is delivered in the cell's report only.",
    "Smoke mode: t0 + the w3 +10 state only, own smoke dir, nothing "
    "adjudicated or verified.",
    "POST-FIRST-RUN REPORTING ADDITIONS, disclosed (deterministic rerun; no "
    "bar, gate, table or verdict touched): two co-report summary fields "
    "(map_read.committed_map_reading_CO_REPORT_not_adjudicating and "
    "walker_read.early_arrival_mis_dials_CO_REPORT) added so the two map "
    "readings and the mis-dial early-arrival texture are machine-readable — "
    "every number in them was already in the first run's tables.",
    "e243 z-join note, disclosed: 10 w3 flips overlap e243's table (same "
    "probe, same RU token — identity guaranteed by G_RU); the RU z values "
    "agree within null-draw noise on 9/10 (|dz| <= 0.33); iPhone's RU "
    "('Apple', z ~ 9.4) reads |dz| = 1.33 — the largest null-draw excursion, "
    "co-reported in the table's e243_join field; no gate or bar reads on dz.",
]


def sha16_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(common.REPO),
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return "unavailable"


def cpu_load_check(tag: str) -> dict:
    try:
        import psutil
        pct = float(psutil.cpu_percent(interval=1.0))
        rec = {"tag": tag, "cpu_percent": pct, "tool": "psutil"}
    except Exception as e:                                # noqa: BLE001
        pct, rec = None, {"tag": tag, "cpu_percent": None,
                          "tool": f"unavailable ({e})"}
    log(f"  [load:{tag}] cpu {pct if pct is None else round(pct, 1)}%")
    return rec


# ---------------------------------------------------------------- e230 extension
def e230_classify(seq: list[int], margins: list[float], ps: list[float]) -> dict:
    """e230's per-probe death-order classification, operationalization
    verbatim (margin event: first state with margin_sigma < 0.05, strictly
    below, absorbing; p event: first state with p < 0.5*p_t0, strictly
    below, absorbing; classes MARGIN-FIRST / P-FIRST / TOGETHER / NEITHER).
    Validated by G_E230 on w1/w2 before it is applied to w3."""
    half = P_HALF * ps[0]
    m_cross = m_depth = p_cross = None
    for i, (m, p) in enumerate(zip(margins, ps)):
        if m_cross is None and m < FLIP_ZONE:
            m_cross, m_depth = i, m
        if p_cross is None and p < half:
            p_cross = i
    last = len(seq) - 1
    if m_cross is not None and (p_cross is None or p_cross > m_cross):
        cls = "MARGIN-FIRST"
        censored = p_cross is None
        lead = None if censored else p_cross - m_cross
        lead_lb = last - m_cross if censored else lead
    elif p_cross is not None and (m_cross is None or m_cross > p_cross):
        cls, lead, lead_lb, censored = "P-FIRST", None, None, False
    elif m_cross is not None and p_cross is not None and m_cross == p_cross:
        cls, lead, lead_lb, censored = "TOGETHER", 0, 0, False
    else:
        cls, lead, lead_lb, censored = "NEITHER", None, None, False
    return {
        "class": cls, "m_cross_idx": m_cross,
        "m_cross_step": seq[m_cross] if m_cross is not None else None,
        "m_depth_at_cross": m_depth, "p_cross_idx": p_cross,
        "p_cross_step": seq[p_cross] if p_cross is not None else None,
        "lead": lead, "lead_lower_bound": lead_lb, "censored": censored,
    }


# ------------------------------------------------------------------ main
def main() -> int:
    rd = run_dir(NAME)
    jp = rd / "journal.json"
    head_at_start = git_head()
    log(f"E247 — THE W3 FLIP CENSUS + THE LAYER-ORDER MAP + THE WALKER READ "
        f"(smoke={SMOKE}) -> {rd} (head at start {head_at_start[:10]})")

    metrics: dict = {
        "experiment": "e247_w3_census",
        "phase": ("eval-only CPU (threads 4) on the committed three-wash 124M "
                  "archive's w3 leg + desk census; T222's registered "
                  "follow-up"),
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": ("bars + operationalizations frozen VERBATIM from the "
                         "dispatch brief in this file's docstring; the "
                         "registration commit precedes any compute; "
                         "adjudicate against exactly this; no bar shopping"),
        "question": ("does e245's order->mode map (the two death modes ARE "
                     "the two death orders) replicate on the THIRD certified "
                     "wash (w3) — and do the collapses pass through a "
                     "mis-dial stage at the finest available state grid?"),
        "builds_on": [
            "T222 / e245 (the registration — the map this cell tests; the "
            "pooled table's w1/w2 side, verbatim)",
            "T221 / e243 + T219 / e241 (the MIS-DIAL / FREQUENCY-COLLAPSE / "
            "OTHER taxonomy — rules + z convention imported VERBATIM)",
            "T212 / e232 + T209 / e230 (the death-order taxonomy — extended "
            "to w3; validated by exact reproduction on w1/w2)",
            "T207 / e228 (the margin instrument — margin_pass module-imported "
            "VERBATIM; the t0 shared record reproduced)",
            "T192 / e217 (the third certified wash — seed 21703, the "
            "committed checkpoints this cell reads)",
            "T187 / e214 + T149 / e182c + T183 / e182c2 + T123 / e182 (the "
            "archive underneath)",
        ],
        "registered_bars_verbatim": BARS_VERBATIM,
        "registered_prediction_T222_verbatim": PREDICTION_T222_VERBATIM,
        "smoke": SMOKE,
    }

    def write_metrics(status: str) -> None:
        metrics["status"] = status
        metrics["updated"] = now_iso()
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    gates: dict[str, dict] = {}
    load_checks: list[dict] = []

    # ------------------------------------------------------ P1 the records
    for p in (E217_M, E217_J, E228_J, E214_J, E230_M, E238_J, E238_NPZ_T0,
              E243_M, E245_M, E226_M, e1.E182_METRICS):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e217m = json.loads(E217_M.read_text(encoding="utf-8"))
    e217j = json.loads(E217_J.read_text(encoding="utf-8"))
    e228j = json.loads(E228_J.read_text(encoding="utf-8"))
    e214j = json.loads(E214_J.read_text(encoding="utf-8"))
    e230m = json.loads(E230_M.read_text(encoding="utf-8"))
    e238j = json.loads(E238_J.read_text(encoding="utf-8"))
    e243r = json.loads(E243_M.read_text(encoding="utf-8"))
    e245r = json.loads(E245_M.read_text(encoding="utf-8"))
    e226m = json.loads(E226_M.read_text(encoding="utf-8"))
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))

    # w3's committed per-probe p records + state grid (e226's convention)
    w3_states = {int(r["step"]): r for r in e217m["wash3_states"]}
    w3_journal_steps = sorted(int(s["step"]) for s in e217j["states"])
    metrics["w3_state_grid"] = {
        "e217_journal_steps": w3_journal_steps,
        "e217_metrics_wash3_states": sorted(w3_states),
        "committed_checkpoints": {
            str(s): {"path": str(W3_CK[s]),
                     "size_bytes": W3_CK[s].stat().st_size,
                     "e226_committed_size_bytes":
                         e226m["inventory"]["states"]["w3"][str(s)]["size_bytes"],
                     "size_match": bool(
                         W3_CK[s].stat().st_size
                         == e226m["inventory"]["states"]["w3"][str(s)]["size_bytes"]),
                     "mtime": time.strftime(
                         "%Y-%m-%dT%H:%M:%SZ",
                         time.gmtime(W3_CK[s].stat().st_mtime))}
            for s in W3_STATES},
        "resumable_latest": {"path": str(W3_LATEST), "exists": W3_LATEST.exists()},
        "finest_available_w3_states": list(W3_TRAJ),
        "no_plus2_state": "w3 archived NO +2 state (the w1-only curve point) — "
                          "the w3 grid is w2-shaped; the walker read's "
                          "finest w3 grid is {10, 50, 80}, disclosed",
        "provenance": e217m["compute"]["wash3"],
    }
    gates["G_W3FILES"] = {
        "all_exist": all(W3_CK[s].exists() for s in W3_STATES),
        "sizes_match_e226_inventory": all(
            v["size_match"] for v in
            metrics["w3_state_grid"]["committed_checkpoints"].values()),
        "journal_steps_cover_checkpoints":
            set(W3_STATES).issubset(set(w3_states)),
        "pass": bool(all(W3_CK[s].exists() for s in W3_STATES)
                     and all(v["size_match"] for v in
                             metrics["w3_state_grid"]["committed_checkpoints"]
                             .values())
                     and set(W3_STATES).issubset(set(w3_states))),
        "desc": "w3's committed checkpoints exist with e226's committed "
                "sizes; the journal covers their steps; bit-exactness is "
                "certified by the G_STATES re-probe vs e217's committed "
                "records (the certified stream: seed 21703, e182c2 "
                "machinery verbatim)",
    }

    # the e228/e214 journals via e245's loader
    j228 = e245m.load_journal(E228_J)
    j214 = e245m.load_journal(E214_J)

    # e238 t0 npz sha gate + grid
    gates["G_NPZ"] = {
        "got": sha16_of(E238_NPZ_T0),
        "committed": e238j["logit_states"]["t0"]["npz_sha256_16"],
        "pass": sha16_of(E238_NPZ_T0)
        == e238j["logit_states"]["t0"]["npz_sha256_16"],
        "desc": "the t0 full-logit dump sha-gated vs e238's journal (the t0 "
                "runner-up gate + the targets' t0 ranks)",
    }
    npz = np.load(E238_NPZ_T0, allow_pickle=True)
    names54 = [str(x) for x in npz["names"]]
    batts54 = [str(x) for x in npz["battery"]]
    ans54 = npz["ans_ids"].astype(int)
    L0 = npz["logits"]

    # e245's committed table (the pooled map's w1/w2 side) + sha sources
    e245_table = e245r["table"]
    pooled_w12_counts: dict[tuple[str, str], int] = {}
    for r in e245_table:
        pooled_w12_counts[(r["e230_death_order_class"], r["classification"])] = \
            pooled_w12_counts.get((r["e230_death_order_class"],
                                   r["classification"]), 0) + 1
    w12_literal_violations = [
        f"{r['wash']} {r['probe']}: {r['e230_death_order_class']} -> "
        f"{r['classification']}"
        for r in e245_table
        if (r["e230_death_order_class"] != "P-FIRST"
            and r["classification"] != "MIS-DIAL")
        or (r["e230_death_order_class"] == "P-FIRST"
            and r["classification"] not in ("FREQUENCY-COLLAPSE", "OTHER"))]
    w12_thin_n = sum(1 for r in e245_table
                     if r["e230_death_order_class"] != "P-FIRST")
    metrics["w1w2_baseline_disclosed"] = {
        "e245_table_n": len(e245_table),
        "order_x_mode_counts": {f"{k[0]}|{k[1]}": v
                                for k, v in sorted(pooled_w12_counts.items())},
        "thin_side_nonPF_n": w12_thin_n,
        "committed_map_form": ("e245/T222's committed map is the ASYMMETRIC "
                               "conjunction: non-P-FIRST => mis-dial (4/4) "
                               "AND collapse => P-FIRST (7/7); P-FIRST "
                               "mis-dials are allowed members"),
        "literal_reading_violations_in_w1w2": w12_literal_violations,
        "note": ("the dispatch's literal parenthetical ('the P-FIRST flips "
                 "collapse or other') already fails 5x on w1/w2's committed "
                 "table — frozen disclosure BEFORE compute: the LITERAL "
                 "reading adjudicates (the frozen letter); the committed "
                 "asymmetric map co-reports"),
    }

    metrics["sources"] = {k: {"path": str(v), "sha256_16": sha16_of(v)}
                          for k, v in {
        "e217_metrics": E217_M, "e217_journal": E217_J,
        "e228_journal": E228_J, "e214_journal": E214_J,
        "e230_metrics": E230_M, "e238_journal": E238_J,
        "e238_logits_t0": E238_NPZ_T0, "e243_metrics": E243_M,
        "e245_metrics": E245_M, "e226_metrics": E226_M,
        "e182_metrics": e1.E182_METRICS}.items()}
    write_metrics("PARTIAL: records read; w3 state grid listed")

    # ------------------------------------------ P2 the organism + batteries
    load_checks.append(cpu_load_check("launch"))
    tok, net0, org_meta = e1.load_organism()
    metrics["organism"] = org_meta
    gates["G_SIZE"] = {"params": org_meta["params"], "ceiling": e1.SIZE_CEILING,
                       "reason": e1.SIZE_REASON,
                       "pass": bool(org_meta["params"] <= e1.SIZE_CEILING)}
    assert gates["G_SIZE"]["pass"], f"size envelope exceeded: {gates['G_SIZE']}"

    # the frozen corpus (G_CORPUS — e228's convention, feeds the batteries'
    # contamination scans only; no wash is run here)
    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    banned = sorted({s.lower() for rel in e1.POOLS for s, _ in e1.POOLS[rel]}
                    | {a.lower() for rel in e1.POOLS
                       for _, a in e1.POOLS[rel]}
                    | set(e1.BANNED_EXTRA))
    assert banned == e182["gates"]["G_STR"]["banned"], \
        "banned list diverged from e182's record"
    cand, _dropped = e1.build_candidates(tok)
    base_cand = e1.probe_battery(net0, cand)
    for r, b in zip(cand, base_cand["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(cand)
    battery = [r for r in cand if r["kept"]]
    answer_ids = {r["fact"]: r["ans_id"] for r in battery}
    train_ids, _bank_xy, filtered, G_STR, _G_TOK, corpus_stats = \
        e1.build_wash_corpus(tok, text, banned, answer_ids)
    e_corp = e182["corpus"]
    gates["G_CORPUS"] = {
        "lines_total": [G_STR["lines_total"], e182["gates"]["G_STR"]["lines_total"]],
        "chars_after": [corpus_stats["chars_after"], e_corp["chars_after"]],
        "tokens_after": [corpus_stats["tokens_after"], e_corp["tokens_after"]],
        "train_tokens": [corpus_stats["train_tokens"], e_corp["train_tokens"]],
        "pass": bool(
            G_STR["lines_total"] == e182["gates"]["G_STR"]["lines_total"]
            and corpus_stats["chars_after"] == e_corp["chars_after"]
            and corpus_stats["tokens_after"] == e_corp["tokens_after"]
            and corpus_stats["train_tokens"] == e_corp["train_tokens"]),
        "note": "the corpus is INHERITED FROZEN (e228's convention) — it "
                "feeds the batteries' contamination scans only",
    }
    log(f"G_CORPUS: {'PASS' if gates['G_CORPUS']['pass'] else 'FAIL'}")

    # the four batteries, VERBATIM (e228's sequence)
    ccand, _cd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, banned, e1.CTRL_POOLS, e1.CTRL_TMPL)
    for r, b in zip(ccand, e1.probe_battery(net0, ccand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ccand)
    cbattery = [r for r in ccand if r["kept"]]
    ncand, _nd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, banned,
        {"near": e1.NEAR_POOL}, {"near": e1.NEAR_TMPL})
    for r, b in zip(ncand, e1.probe_battery(net0, ncand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ncand)
    nbattery = [r for r in ncand if r["kept"]]
    tcand, _td = e2.build_tmpl_candidates(tok, filtered.lower(), train_ids,
                                          banned)
    for r, b in zip(tcand, e1.probe_battery(net0, tcand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(tcand)
    tbattery = [r for r in tcand if r["kept"]]
    bats = {"fact": battery, "ctrl": cbattery,
            "near": nbattery, "tmpl": tbattery}
    log("batteries rebuilt VERBATIM: " + " | ".join(
        f"{b} n={len(bats[b])}" for b in BATTERIES))

    # ------------------------- P3 the margin pass (t0 reproduction + w3 fresh)
    t0_228 = next(s for s in e228j["states"] if s["wash"] == "t0")
    mjour: list[dict] = []
    if jp.exists():
        try:
            mjour = json.loads(jp.read_text(encoding="utf-8"))["states"]
            log(f"journal: {len(mjour)} margin records restored")
        except Exception as e:                              # noqa: BLE001
            log(f"journal unreadable ({e}); recomputing")
            mjour = []
    mdone = {(r["wash"], r["step"]) for r in mjour}

    def margin_all(evl) -> dict:
        return {b: margin_pass(evl, bats[b]) for b in BATTERIES}

    # t0 — ONE read of the shared pristine organism; GATE vs e228's committed
    if ("t0", 0) not in mdone:
        load_checks.append(cpu_load_check("t0"))
        cur = margin_all(net0)
        mjour.append({"wash": "t0", "step": 0, **cur})
        mdone.add(("t0", 0))
        mjour.sort(key=lambda r: (r["wash"], r["step"]))
        jp.write_text(json.dumps({"states": mjour}, indent=1),
                      encoding="utf-8")
    t0m = next(r for r in mjour if r["wash"] == "t0")

    if not SMOKE:
        # G_BATT (e228's convention, on the margin pass's own t0 record) +
        # G_T0REPRO: my fresh t0 must reproduce e217's AND e228's committed
        # t0 records — the instrument + batteries certified before any w3 read
        gb = {}
        dps, dms, tid_mm, t2id_mm = [], [], 0, 0
        for b in BATTERIES:
            mine = {r["fact"]: r for r in t0m[b]["probes"]}
            ref217 = {f: v["p"] for f, v in w3_states[0][b]["probes"].items()}
            ref228 = {r["fact"]: r for r in t0_228[b]["probes"]}
            gb[b] = {
                "n": len(bats[b]),
                "set_equal_committed": bool(set(mine) == set(ref217)),
                "max_dp_vs_e217_t0": max(abs(mine[f]["p"] - v)
                                         for f, v in ref217.items()),
                "max_dp_vs_e228_t0": max(abs(mine[f]["p"] - v["p"])
                                         for f, v in ref228.items()),
            }
            for r in t0m[b]["probes"]:
                v = ref228[r["fact"]]
                dps.append(abs(r["p"] - v["p"]))
                if r["margin_sigma"] is not None and v["margin_sigma"] is not None:
                    dms.append(abs(r["margin_sigma"] - v["margin_sigma"]))
                tid_mm += int(r["top1_id"] != v["top1_id"])
                t2id_mm += int(r["top2_id"] != v["top2_id"])
        gates["G_BATT"] = {
            **gb, "tol_per_probe_dp": TOL_PROBE_DP,
            "note": "the four batteries are the phase-1/phase-2 pools "
                    "VERBATIM (module import); my t0 margin-pass p must "
                    "reproduce BOTH e217's committed w3-lineage t0 AND "
                    "e228's committed t0 journal record",
            "pass": bool(all(gb[b]["set_equal_committed"]
                             and gb[b]["max_dp_vs_e217_t0"] <= TOL_PROBE_DP
                             and gb[b]["max_dp_vs_e228_t0"] <= TOL_PROBE_DP
                             for b in BATTERIES)),
        }
        gates["G_T0REPRO"] = {
            "max_dp_p": max(dps), "tol_p": TOL_T0_P,
            "max_d_margin_sigma": max(dms), "tol_margin": TOL_T0_MARGIN,
            "top1_id_mismatches": tid_mm, "top2_id_mismatches": t2id_mm,
            "pass": bool(max(dps) <= TOL_T0_P and max(dms) <= TOL_T0_MARGIN
                         and tid_mm == 0 and t2id_mm == 0),
            "desc": "my fresh t0 margin pass (module-imported instrument, "
                    "same machine/CPU fp32 path) reproduces e228's committed "
                    "t0 journal record (p, margin_sigma, top1/top2) — the "
                    "instrument + batteries certified before any w3 read",
        }
        metrics["gates"] = gates
        log(f"G_BATT: {'PASS' if gates['G_BATT']['pass'] else 'FAIL'} "
            + " | ".join(f"{b}: dp217 {gb[b]['max_dp_vs_e217_t0']:.2e} "
                         f"dp228 {gb[b]['max_dp_vs_e228_t0']:.2e}"
                         for b in BATTERIES))
        log(f"G_T0REPRO: {'PASS' if gates['G_T0REPRO']['pass'] else 'FAIL'} "
            f"(dp {max(dps):.2e}, dm {max(dms):.2e}, t1mm {tid_mm}, "
            f"t2mm {t2id_mm})")
    write_metrics("PARTIAL: organism + batteries rebuilt; t0 certified")

    # w3 states — the fresh margins + argmaxes (DISCLOSED: not in e228)
    for s_ in W3_STATES:
        if ("w3", s_) in mdone:
            continue
        load_checks.append(cpu_load_check(f"w3s{s_}"))
        sd = torch.load(W3_CK[s_], map_location=CPU,
                        weights_only=False)["model"]
        evl = copy.deepcopy(net0)
        evl.load_state_dict(sd)
        cur = margin_all(evl)
        del evl, sd
        mjour.append({"wash": "w3", "step": s_, **cur})
        mdone.add(("w3", s_))
        mjour.sort(key=lambda r: (r["wash"], r["step"]))
        jp.write_text(json.dumps({"states": mjour}, indent=1),
                      encoding="utf-8")
        log(f"  W3 STATE +{s_}: " + " | ".join(
            f"{b} m {cur[b]['mean_margin_sigma']:.3f}s (argmax-ans "
            f"{cur[b]['frac_argmax_answer']:.2f}) p "
            f"{cur[b]['mean_p']:.4f} (e217 {w3_states[s_][b]['mean_p']:.4f})"
            for b in BATTERIES))
        write_metrics(f"PARTIAL: w3 margins read through +{s_}")
    mst = {(r["wash"], r["step"]): r for r in mjour}

    # G_STATES: the p re-probe on every loaded w3 state vs e217's committed
    if not SMOKE:
        all_dps, rank_mm = [], 0
        for s_ in W3_STATES:
            rec = mst[("w3", s_)]
            for b in BATTERIES:
                ref = w3_states[s_][b]["probes"]
                mine = {r["fact"]: r for r in rec[b]["probes"]}
                all_dps.append(max(abs(mine[f]["p"] - v["p"])
                                   for f, v in ref.items()))
                # double-source the argmax: my top1==answer iff e217 rank==0
                for f, v in ref.items():
                    rank_mm += int((int(mine[f]["top1_id"])
                                    == int(answer_ids_b(bats, b)[f]))
                                   != (int(v["rank"]) == 0))
        gates["G_STATES"] = {
            "reprobe_max_dp": max(all_dps), "n_reprobe_checks": len(all_dps),
            "tol_reprobe_dp": TOL_STATE_DP,
            "argmax_rank_double_source_mismatches": rank_mm,
            "pass": bool(max(all_dps) <= TOL_STATE_DP and rank_mm == 0),
            "desc": "every battery's p re-probed on every LOADED w3 state vs "
                    "e217's committed wash3_states records (expected ~0 — "
                    "e217's probes were CPU fp32); the argmax side "
                    "double-sourced: my top1==answer iff e217's committed "
                    "answer-rank==0, all 54 probes x all states",
        }
        log(f"G_STATES: {'PASS' if gates['G_STATES']['pass'] else 'FAIL'} "
            f"(max dp {max(all_dps):.2e}, rank mismatches {rank_mm})")
    gates["G_ENV"] = {
        "cpu_only": True, "torch_threads": torch.get_num_threads(),
        "gpu_calls": 0, "load_checks": len(load_checks),
        "burst": "one state = one eval burst",
        "pass": bool(torch.get_num_threads() <= 8),
    }
    metrics["gates"] = gates
    write_metrics("PARTIAL: margin pass complete (t0 + w3 states)")
    if SMOKE:
        log("SMOKE DONE (nothing adjudicated)")
        metrics["status"] = "SMOKE DONE"
        metrics["all_gates_pass"] = None
        save_json(rd / "metrics.json", metrics)
        return 0

    # ============================ P4 THE CENSUS DESK (no more model loads) ====
    # the wte + tokenizer (e243's conventions, module-imported constants)
    from safetensors import safe_open
    hub_snap = (Path.home() / ".cache" / "huggingface" / "hub"
                / f"models--{e243m.MODEL_REPO.replace('/', '--')}" / "snapshots"
                / e243m.MODEL_REV)
    snap_file = next(hub_snap.glob("*safetensors"))
    with safe_open(str(snap_file), framework="numpy") as f:
        wte = f.get_tensor("wte.weight").astype(np.float64)
    V = int(wte.shape[0])
    wte_sha = hashlib.sha256(
        np.ascontiguousarray(wte.astype(np.float32)).tobytes()
    ).hexdigest()[:16]
    gates["G_WTE"] = {
        "got": wte_sha, "committed": e243m.E241_WTE_SHA16,
        "pass": bool(wte_sha == e243m.E241_WTE_SHA16
                     and wte.shape == (50257, 768)),
        "desc": "pristine shared-124M wte read tensor-only from the committed "
                "HF snapshot; sha16(fp32 bytes) == e241's committed wte sha",
    }
    enc_the = tok.encode(e243m.THE_STR)
    the_id = int(enc_the[0]) if len(enc_the) == 1 else -1
    gates["G_IDS"] = {
        "pass": bool(the_id == e243m.THE_ID_EXPECTED),
        "the_encodes_to": enc_the,
        "desc": f"tokenizer encodes '{e243m.THE_STR}' to id "
                f"{e243m.THE_ID_EXPECTED} (the frequency token; "
                "e243's tokenizer-verified convention)",
    }
    Wn = wte / np.linalg.norm(wte, axis=1, keepdims=True)

    # the t0 per-probe reference table (from e228's committed t0 + the npz)
    t0_top = {}   # (battery, probe) -> committed t0 record
    for b in BATTERIES:
        for r in t0_228[b]["probes"]:
            t0_top[(b, r["fact"])] = r
    # RU gate: npz rank-2 == e228 t0 top2_id (all 54) — e243's G_RU reproduced
    ru_mism = []
    ru_of = {}
    ranks0 = {}
    for i, nm in enumerate(names54):
        order = np.argsort(-L0[i].astype(np.float64), kind="stable")
        rk = {int(t): j + 1 for j, t in enumerate(order)}
        ranks0[i] = rk
        r2 = int(next(t for t, r in rk.items() if r == 2))
        b = batts54[i]
        ru_of[(b, nm)] = r2
        if r2 != int(t0_top[(b, nm)]["top2_id"]):
            ru_mism.append((nm, r2, t0_top[(b, nm)]["top2_id"]))
    gates["G_RU"] = {
        "pass": not ru_mism, "n_mismatches": len(ru_mism),
        "desc": "npz-derived t0 rank-2 == e228's committed t0 top2_id for all "
                "54 probes (e243's G_RU reproduced — the runner-up gate)",
    }
    gates["G_ANSWER_T0"] = {
        "pass": all(int(t0_top[(batts54[i], names54[i])]["top1_id"])
                    == int(ans54[i]) for i in range(54)),
        "desc": "the answer was the t0 argmax for all 54 probes (e243's "
                "G_ANSWER_T0 reproduced — every flip is a commitment death)",
    }

    # ------------------------------------------------ the w3 flip population
    w3_80 = mst[("w3", 80)]
    flips = []
    for i, nm in enumerate(names54):
        b = batts54[i]
        t0_id = int(t0_top[(b, nm)]["top1_id"])
        mine = {r["fact"]: r for r in w3_80[b]["probes"]}[nm]
        if int(mine["top1_id"]) != t0_id:
            flips.append({"battery": b, "probe": nm, "row": i})
    log(f"w3 flip population (+80-argmax-differs): {len(flips)} records "
        + " ".join(f"[{f['battery']} {f['probe']}]" for f in flips))

    # ------------------------------------------------------- the e230 G_E230
    # validation: reproduce e230's committed w1/w2 classes EXACTLY, then w3
    def journal_series(wash: str, steps, bat: str, probe: str):
        """(seq, margins, ps) for one probe-wash lineage; margins from e228's
        committed journal (w1/w2) or THIS CELL's fresh journal (w3, t0 read:
        the fresh t0 record — one session, the deterministic-in-session
        argument, gated equal to e228's by G_T0REPRO); p from e214 (w1/w2) or
        e217 (w3), all committed records."""
        seq = [0] + list(steps)
        if wash == "w3":
            margins = [next(r for r in mst[("t0", 0)][bat]["probes"]
                            if r["fact"] == probe)["margin_sigma"]]
            margins += [next(r for r in mst[("w3", s)][bat]["probes"]
                             if r["fact"] == probe)["margin_sigma"]
                        for s in steps]
            ps = [w3_states[0][bat]["probes"][probe]["p"]]
            ps += [w3_states[s][bat]["probes"][probe]["p"] for s in steps]
        else:
            traj = j228[(bat, probe)]
            margins = [traj["t0"]["margin_sigma"]]
            margins += [traj[f"{wash}_{s}"]["margin_sigma"] for s in steps]
            pj = j214[(bat, probe)]
            ps = [pj["t0"]["p"]] + [pj[f"{wash}_{s}"]["p"] for s in steps]
        return seq, margins, ps

    e230_repro = {"n": 0, "class_mismatches": []}
    for b in BATTERIES:
        for w in ("w1", "w2"):
            committed = e230m["read1_population"]["probe_level"][b][w]
            for crec in committed:
                seq, ms, ps = journal_series(w, W1_W2_TRAJ[w], b, crec["probe"])
                mine = e230_classify(seq, ms, ps)
                e230_repro["n"] += 1
                if mine["class"] != crec["class"]:
                    e230_repro["class_mismatches"].append(
                        {"probe": crec["probe"], "wash": w,
                         "mine": mine["class"], "committed": crec["class"]})
    gates["G_E230"] = {
        "n_records": e230_repro["n"],
        "class_mismatches": e230_repro["class_mismatches"],
        "pass": not e230_repro["class_mismatches"],
        "desc": "this cell's re-implementation of e230's death-order "
                "classification reproduces e230's committed w1/w2 "
                "probe-level classes EXACTLY (108 records) — the extension "
                "to w3 runs on validated code (e232's cross-check "
                "convention)",
    }
    log(f"G_E230: {'PASS' if gates['G_E230']['pass'] else 'FAIL'} "
        f"({e230_repro['n']} records, "
        f"{len(e230_repro['class_mismatches'])} mismatches)")

    # w3 death-order classes for ALL 54 probes (order census co-report)
    w3_orders = {}
    for b in BATTERIES:
        for r in t0_228[b]["probes"]:
            nm = r["fact"]
            seq, ms, ps = journal_series("w3", W3_TRAJ, b, nm)
            w3_orders[(b, nm)] = e230_classify(seq, ms, ps)
    order_counts: dict[str, int] = {}
    for v in w3_orders.values():
        order_counts[v["class"]] = order_counts.get(v["class"], 0) + 1
    log("w3 death-order census (all 54): " + str(order_counts))

    # ---------------------------------------- per-flip reads: z + classification
    rng = np.random.default_rng(e243m.NULL_SEED)
    e243_idx = {(r["battery"], r["probe"]): r for r in e243r["table"]}
    table = []
    for f in flips:  # frozen order: battery fact/ctrl/near/tmpl x grid order
        b, nm, i = f["battery"], f["probe"], f["row"]
        ans_id = int(ans54[i])
        ru_id = int(ru_of[(b, nm)])
        rec80 = {r["fact"]: r for r in w3_80[b]["probes"]}[nm]
        tgt_id = int(rec80["top1_id"])
        l0 = L0[i]
        rank_of = ranks0[i]
        rank_tgt = rank_of[tgt_id]
        gap_ans_ru = float(l0[ans_id]) - float(l0[ru_id])
        gap_ans_tgt = float(l0[ans_id]) - float(l0[tgt_id])

        # z reads (e241/e243 convention; RU null first, then target null)
        e_ans = Wn[ans_id]
        zrec = {}
        for role, tid in [("ru", ru_id), ("tgt", tgt_id)]:
            sim = float(np.dot(Wn[tid], e_ans))
            allowed = np.array([x for x in range(V) if x not in (ans_id, tid)])
            ids = rng.choice(allowed, size=e243m.N_NULL, replace=False)
            sn = Wn[ids] @ e_ans
            mu, sd = float(sn.mean()), float(sn.std(ddof=1))
            zrec[role] = {"token": tok.decode([tid]), "id": tid,
                          "cos_sim": round(sim, 4),
                          "null_mean": round(mu, 4), "null_std": round(sd, 4),
                          "z": round((sim - mu) / sd, 3)}
        z_ru, z_tgt = zrec["ru"]["z"], zrec["tgt"]["z"]

        # classification — e243's rules, module-imported VERBATIM
        if tgt_id == ru_id and z_ru >= e243m.Z_LOCAL:
            cls = "MIS-DIAL"
        elif (tgt_id == the_id
              or (rank_tgt >= e243m.RANK_FREQ and z_tgt < e243m.Z_NEG)):
            cls = "FREQUENCY-COLLAPSE"
        else:
            cls = "OTHER"

        # the e230 death-order class (validated extension)
        oc = w3_orders[(b, nm)]

        # the trajectory + first crossing + walker reads (w3 lineage)
        traj = []
        for s in W3_TRAJ:
            rr = next(r for r in mst[("w3", s)][b]["probes"] if r["fact"] == nm)
            traj.append({"step": s, "top1_id": int(rr["top1_id"]),
                         "token": tok.decode([int(rr["top1_id"])]),
                         "p": w3_states[s][b]["probes"][nm]["p"],
                         "margin_sigma": rr["margin_sigma"]})
        t0_id = int(t0_top[(b, nm)]["top1_id"])
        first_step = next((t["step"] for t in traj
                           if t["top1_id"] != t0_id), None)
        p_t0 = w3_states[0][b]["probes"][nm]["p"]
        p_80 = w3_states[80][b]["probes"][nm]["p"]
        passed_ru = any(t["top1_id"] == ru_id for t in traj[:-1])
        excursion = any(t["top1_id"] not in (t0_id, tgt_id) for t in traj[:-1])

        j243 = e243_idx.get((b, nm))  # same probe under w1/w2 in e243's table
        j243_note = None
        if j243 is not None:
            j243_note = {
                "e243_washes": [r["wash"] for r in e243r["table"]
                                if r["battery"] == b and r["probe"] == nm],
                "dz_RU_vs_e243": round(z_ru - j243["runner_up_z"], 3),
                "note": "same probe, same RU; null draws differ by record "
                        "order (e243's convention note) — identity check |dz| "
                        "< 1",
            }

        table.append({
            "wash": "w3", "battery": b, "probe": nm,
            "dead_answer_id": ans_id,
            "dead_answer_token": tok.decode([ans_id]),
            "runner_up_id": ru_id, "runner_up": zrec["ru"]["token"],
            "runner_up_z": z_ru, "runner_up_cos_sim": zrec["ru"]["cos_sim"],
            "target_id": tgt_id, "target": zrec["tgt"]["token"],
            "target_rank_t0": rank_tgt,
            "target_cos_sim": zrec["tgt"]["cos_sim"], "target_z": z_tgt,
            "target_is_the": bool(tgt_id == the_id),
            "target_is_runner_up": bool(tgt_id == ru_id),
            "logit_gap_ans_minus_ru_t0": round(gap_ans_ru, 4),
            "logit_gap_ans_minus_tgt_t0": round(gap_ans_tgt, 4),
            "classification": cls,
            "classification_rules_verbatim": e243m.CLASS_RULES_VERBATIM,
            "death_order_class": oc["class"],
            "death_order_detail": {k: v for k, v in oc.items() if k != "class"},
            "p_t0": round(p_t0, 6), "p_80": round(p_80, 6),
            "halved_endpoint": bool(p_80 < 0.5 * p_t0),
            "margin_sigma_80_new_commitment": rec80["margin_sigma"],
            "argmax_trajectory": traj,
            "first_cross_step": first_step,
            "first_cross_is_target_80": bool(
                first_step is not None
                and next(t["top1_id"] for t in traj if t["step"] == first_step)
                == tgt_id),
            "passed_through_RU_before_80": passed_ru,
            "transient_excursion_before_80": excursion,
            "e243_join": j243_note,
        })
        log(f"w3 {b} {nm}: order={oc['class']} mode={cls} tgt="
            f"'{zrec['tgt']['token']}' (rank {rank_tgt}, z {z_tgt:+.2f}) "
            f"RU='{zrec['ru']['token']}' (z {z_ru:+.2f}) walker={passed_ru}")

    n_mis = sum(1 for r in table if r["classification"] == "MIS-DIAL")
    n_frq = sum(1 for r in table if r["classification"] == "FREQUENCY-COLLAPSE")
    n_oth = sum(1 for r in table if r["classification"] == "OTHER")

    # --------------------------------------------- the map: joins + violations
    def is_pf(r):
        return r["death_order_class"] == "P-FIRST"

    literal_viol = []
    committed_viol = []
    for r in table:
        if not is_pf(r) and r["classification"] != "MIS-DIAL":
            literal_viol.append(
                f"w3 {r['probe']}: {r['death_order_class']} -> "
                f"{r['classification']} (clause a: non-P-FIRST must "
                f"mis-dial)")
        if is_pf(r) and r["classification"] not in ("FREQUENCY-COLLAPSE",
                                                    "OTHER"):
            literal_viol.append(
                f"w3 {r['probe']}: {r['death_order_class']} -> "
                f"{r['classification']} (clause b: P-FIRST must "
                f"collapse-or-other)")
        if r["classification"] == "FREQUENCY-COLLAPSE" and not is_pf(r):
            committed_viol.append(
                f"w3 {r['probe']}: {r['death_order_class']} -> "
                f"{r['classification']} (clause c: collapse must be "
                f"P-FIRST)")
    w3_thin_n = sum(1 for r in table if not is_pf(r))
    thin_side_grew = w3_thin_n >= 1
    pooled_thin = w12_thin_n + w3_thin_n

    w3_map_counts: dict[tuple[str, str], int] = {}
    for r in table:
        k = (r["death_order_class"], r["classification"])
        w3_map_counts[k] = w3_map_counts.get(k, 0) + 1
    pooled_counts = dict(pooled_w12_counts)
    for k, v in w3_map_counts.items():
        pooled_counts[k] = pooled_counts.get(k, 0) + v

    map_read = {
        "w3_counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(w3_map_counts.items())},
        "pooled_counts_w1w2w3": {f"{k[0]}|{k[1]}": v
                                 for k, v in sorted(pooled_counts.items())},
        "literal_violations": literal_viol,
        "committed_map_violations_c_only": committed_viol,
        "committed_map_violations_full": (
            [v for v in literal_viol if "clause a" in v] + committed_viol),
        "thin_side": {
            "w1w2_nonPF_n": w12_thin_n,
            "w3_nonPF_n": w3_thin_n,
            "pooled_nonPF_n": pooled_thin,
            "grew": thin_side_grew,
            "w3_nonPF_members": [f"{r['probe']} ({r['death_order_class']} -> "
                                 f"{r['classification']})"
                                 for r in table if not is_pf(r)],
        },
        "n_disclosed": {
            "w3_flips": len(table),
            "w3_P-FIRST": sum(1 for r in table if is_pf(r)),
            "w3_nonP-FIRST": w3_thin_n,
            "w3_MIS-DIAL": n_mis, "w3_FREQ-COLLAPSE": n_frq,
            "w3_OTHER": n_oth,
            "w3_order_census_all54": order_counts,
            "w1w2_flips": len(e245_table),
        },
        "readings_disclosure": {
            "literal_adjudicates": ("(a) every non-P-FIRST flip mis-dials AND "
                                    "(b) every P-FIRST flip is "
                                    "collapse-or-other — the dispatch's frozen "
                                    "letter (the conservative test)"),
            "committed_map_co_report": ("e245/T222's asymmetric map: (a) + "
                                        "(c) collapse => P-FIRST; P-FIRST "
                                        "mis-dials are allowed members — the "
                                        "w1/w2 table already carries 5 under "
                                        "the literal clause b"),
            "w1w2_literal_violations_count": len(w12_literal_violations),
        },
        "committed_map_reading_CO_REPORT_not_adjudicating": {
            "clause_a_nonPF_all_mis_dial": all(
                r["classification"] == "MIS-DIAL"
                for r in table if not is_pf(r)),
            "clause_c_all_collapses_P_FIRST": all(
                is_pf(r) for r in table
                if r["classification"] == "FREQUENCY-COLLAPSE"),
            "violations_w3": committed_viol,
            "thin_side_grew": thin_side_grew,
            "holds_on_w3": bool(not committed_viol) and thin_side_grew,
            "pooled_check_w1w2w3": {
                "nonPF_all_mis_dial": all(
                    (r["e230_death_order_class"] == "P-FIRST")
                    or (r["classification"] == "MIS-DIAL")
                    for r in e245_table) and all(
                    r["classification"] == "MIS-DIAL"
                    for r in table if not is_pf(r)),
                "all_collapses_P_FIRST": all(
                    (r["e230_death_order_class"] == "P-FIRST")
                    for r in e245_table
                    if r["classification"] == "FREQUENCY-COLLAPSE") and all(
                    is_pf(r) for r in table
                    if r["classification"] == "FREQUENCY-COLLAPSE"),
                "pooled_nonPF_n": pooled_thin,
            },
            "note": ("e245/T222's committed unification, evaluated verbatim on "
                     "w3 and pooled — a CO-REPORT; the LITERAL bar adjudicates "
                     "the verdict (frozen before compute)"),
        },
    }

    # ------------------------------------------------------- the walker read
    walker_rows = []
    for r in e245_table:  # the 19 committed records; trajectories from e228
        b, nm, w = r["battery"], r["probe"], r["wash"]
        traj228 = j228[(b, nm)]
        t0_id = int(traj228["t0"]["top1_id"])
        ru_id = int(traj228["t0"]["top2_id"])
        tgt_id = int(r["target_id_80"])
        traj = [{"step": s, "top1_id": int(traj228[f"{w}_{s}"]["top1_id"]),
                 "token": tok.decode([int(traj228[f"{w}_{s}"]["top1_id"])]),
                 "p": traj228[f"{w}_{s}"]["p"],
                 "margin_sigma": traj228[f"{w}_{s}"]["margin_sigma"]}
                for s in W1_W2_TRAJ[w]]
        walker_rows.append({
            "wash": w, "battery": b, "probe": nm,
            "classification": r["classification"],
            "target_80": r["target_80"], "target_id_80": tgt_id,
            "death_order_class": r["e230_death_order_class"],
            "t0_top1_id": t0_id, "runner_up_id": ru_id,
            "runner_up": tok.decode([ru_id]),
            "argmax_trajectory": traj,
            "passed_through_RU_before_80":
                any(t["top1_id"] == ru_id for t in traj[:-1]),
            "transient_excursion_before_80":
                any(t["top1_id"] not in (t0_id, tgt_id) for t in traj[:-1]),
            "source": "e228 journal (committed)",
        })
    for r in table:  # w3 (fresh)
        walker_rows.append({
            "wash": "w3", "battery": r["battery"], "probe": r["probe"],
            "classification": r["classification"],
            "target_80": r["target"], "target_id_80": r["target_id"],
            "death_order_class": r["death_order_class"],
            "t0_top1_id": int(t0_top[(r["battery"], r["probe"])]["top1_id"]),
            "runner_up_id": r["runner_up_id"], "runner_up": r["runner_up"],
            "argmax_trajectory": r["argmax_trajectory"],
            "passed_through_RU_before_80": r["passed_through_RU_before_80"],
            "transient_excursion_before_80": r["transient_excursion_before_80"],
            "source": "this cell's fresh w3 pass (G_STATES-certified)",
        })

    def _rate(rows):
        coll = [r for r in rows if r["classification"] == "FREQUENCY-COLLAPSE"]
        k = sum(1 for r in coll if r["passed_through_RU_before_80"])
        exc = sum(1 for r in coll if r["transient_excursion_before_80"])
        return {"n_collapses": len(coll),
                "of_which_passed_through_RU": k,
                "walker_rate": round(k / len(coll), 4) if coll else None,
                "of_which_transient_excursion": exc,
                "excursion_rate": round(exc / len(coll), 4) if coll else None}

    walker_read = {
        "definition_frozen": ("a flip record PASSED THROUGH A MIS-DIAL iff "
                              "some state STRICTLY BEFORE +80 of its own "
                              "wash's grid has argmax == the t0 runner-up; "
                              "walker rate = fraction of FREQUENCY-COLLAPSE "
                              "flip records that passed through a mis-dial "
                              "(co-report, NO bar); grids: w1 {2,10,50,80}, "
                              "w2 {10,50,80}, w3 {10,50,80}"),
        "pooled": _rate(walker_rows),
        "per_wash": {w: _rate([r for r in walker_rows if r["wash"] == w])
                     for w in ("w1", "w2", "w3")},
        "the_walkers": [f"{r['wash']} {r['probe']} ({r['classification']}) "
                        f"RU='{r['runner_up']}' -> '{r['target_80']}'"
                        for r in walker_rows
                        if r["passed_through_RU_before_80"]],
        "mis_dials_that_visited_RU_at_80_only": [
            f"{r['wash']} {r['probe']}" for r in walker_rows
            if r["classification"] == "MIS-DIAL"
            and not r["passed_through_RU_before_80"]],
        "early_arrival_mis_dials_CO_REPORT": {
            "definition": "MIS-DIAL records whose RU was already the argmax "
                          "at a state STRICTLY BEFORE +80 (the mis-dial as "
                          "an EARLY, ABSORBING slide — arrived and stayed, "
                          "vs the collapse's one-way walk past it)",
            "pooled_n": sum(1 for r in walker_rows
                            if r["classification"] == "MIS-DIAL"
                            and r["passed_through_RU_before_80"]),
            "mis_dial_total_pooled": sum(1 for r in walker_rows
                                         if r["classification"] == "MIS-DIAL"),
            "members": [f"{r['wash']} {r['probe']} (RU reached at +"
                        + str(next(t["step"] for t in r["argmax_trajectory"]
                                   if t["top1_id"] == r["runner_up_id"])) + ")"
                        for r in walker_rows
                        if r["classification"] == "MIS-DIAL"
                        and r["passed_through_RU_before_80"]]},
        "note": ("the finer-state read between journal states remains "
                 "invisible (the coarseness e245 disclosed); a mis-dial "
                 "AT +80 that is the record's own final target is the "
                 "mis-dial itself, not a stage — hence strictly-before-80; "
                 "T222's registered 'the walkers are common' is read "
                 "against the collapse-side walker rate (pooled "
                 "of_which_passed_through_RU / n_collapses)"),
    }

    # -------------------------------------------------------- adjudication
    gates_ok = bool(gates["G_W3FILES"]["pass"] and gates["G_CORPUS"]["pass"]
                    and gates["G_BATT"]["pass"] and gates["G_T0REPRO"]["pass"]
                    and gates["G_STATES"]["pass"] and gates["G_E230"]["pass"]
                    and gates["G_RU"]["pass"] and gates["G_ANSWER_T0"]["pass"]
                    and gates["G_NPZ"]["pass"] and gates["G_WTE"]["pass"]
                    and gates["G_IDS"]["pass"] and gates["G_ENV"]["pass"])

    if not gates_ok:
        verdict = "VERIFICATION-FAILED (tables reported; no bar read)"
        clause = ("verification gates failed: " + ", ".join(
            k for k, v in gates.items() if not v.get("pass", True)))
    elif literal_viol:
        verdict = "ORDER-MAP-BREAKS"
        clause = ("any violation — the map is w1/w2 texture; T222's "
                  "unification bounded (violations: "
                  + "; ".join(literal_viol)
                  + f"); thin side w3 n={w3_thin_n}, pooled {pooled_thin}; "
                  "the committed asymmetric map's own violations co-reported "
                  "in map_read")
    elif thin_side_grew:
        verdict = "ORDER-MAP-HOLDS"
        clause = (f"w3's {len(table)} flips follow the order->mode map (every "
                  f"non-P-FIRST flip mis-dials; the P-FIRST flips collapse or "
                  f"other; zero violations) AND the pooled n on the thin side "
                  f"grows ({w12_thin_n} -> {pooled_thin}) — the unification "
                  "replicates on a third wash")
    else:
        verdict = "MIXED"
        clause = ("zero violations but the thin side did not grow "
                  f"(w3 non-P-FIRST n={w3_thin_n} of {len(table)} flips; "
                  f"pooled {pooled_thin}) — the empty-stratum form (e243's "
                  "pre-registered rule); the tables verbatim, no inflation")

    metrics["table"] = table
    metrics["map_read"] = map_read
    metrics["walker_read"] = walker_read
    metrics["walker_rows"] = walker_rows
    metrics["w3_order_census_all54"] = {
        "counts": order_counts,
        "probe_level": {f"{b}|{nm}": v for (b, nm), v in w3_orders.items()},
        "note": "e230's classification extended to w3 (validated by G_E230); "
                "the map joins only the FLIP records to their order class",
    }
    metrics["adjudication"] = {
        "verdict": verdict, "clause": clause,
        "bar_fired_verbatim": BARS_VERBATIM.get(
            verdict if verdict in BARS_VERBATIM else "MIXED", None),
        "ladder": ("any LITERAL violation -> ORDER-MAP-BREAKS; else thin side "
                   "grew -> ORDER-MAP-HOLDS; else MIXED (empty-stratum form); "
                   "gated on all verification gates (e228's no-bar-read "
                   "precedent on gate failure)"),
        "gates_summary": {k: bool(v.get("pass", True)) for k, v in gates.items()},
    }
    metrics["compute"] = {
        "envelope": f"CPU-only eval+desk (threads {torch.get_num_threads()}); "
                    f"{len(load_checks)} load checks; one state = one eval "
                    "burst; no GPU calls",
        "load_checks": load_checks,
        "state_archive": [str(W3_CK[s]) for s in W3_STATES],
        "model_loads": 1 + len(W3_STATES),
    }
    metrics["honesty_reflex"] = (
        "KNOWN CONFOUNDS, disclosed not corrected: (1) THE MAP'S TWO "
        "READINGS — the dispatch's literal clause b ('P-FIRST flips collapse "
        "or other') is stricter than e245/T222's committed asymmetric map "
        "(non-P-FIRST => mis-dial AND collapse => P-FIRST); w1/w2's own "
        "committed table carries 5 P-FIRST->MIS-DIAL records, so clause b "
        "can fire on w3 without touching the committed map — both readings "
        "reported per clause; the literal reading adjudicates (frozen letter, "
        "no shopping). (2) w3's margins/argmaxes are COMPUTED FRESH (they do "
        "not exist in e228's journal) — t0 reproduces e228's committed "
        "record and every w3 state re-probes e217's committed p (expected "
        "~0; e217's probes were CPU fp32); the w3 WEIGHTS are GPU fp32 "
        "origin (the archive's device asymmetry, inherited, disclosed). "
        "(3) The w3 grid has no +2 state — the finest w3 grid is {10,50,80}; "
        "transient mis-dials BETWEEN states remain invisible (e245's "
        "disclosure carries); the walker rate is a lower bound. (4) e243's "
        "confounds carry VERBATIM: the 'the' frequency geometry (its "
        "anti-neighbor z is frequency, not semantics); the z-null draw noise "
        "(|dz| < 1 identity checks on e243 joins); 6 probes flip under more "
        "than one wash sharing one t0/RU read. (5) n disclosed per stratum "
        "(the w1/w2 thin side was n=4; w3's counts in map_read.n_disclosed); "
        "the bars adjudicate on whatever n exists; the death-order classes "
        "are grid-coarse (TOGETHER = same grid cell). (6) three washes share "
        "corpus/optimizer/lr — independence is over draw streams only (the "
        "archive's only independence class). (7) ONE organism (the shared "
        "pristine 124M). Nothing here is guaranteed.")
    metrics["trims"] = trims
    metrics["deviations"] = deviations
    metrics["all_gates_pass"] = gates_ok
    write_metrics("PARTIAL: census + map + walker computed — figures next")
    log("=" * 78)
    log(f"E247 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  w3 flips {len(table)} (MIS-DIAL {n_mis} / FREQ {n_frq} / "
        f"OTHER {n_oth}); order census {order_counts}")
    log(f"  walker rate (pooled, collapses): "
        f"{walker_read['pooled']['walker_rate']}")

    # ------------------------------------------------------------- figures
    fig = make_map_figure(rd, verdict, clause, map_read, gates)
    fig2 = make_walker_figure(rd, walker_rows, walker_read)
    metrics["figures"] = [str(fig), str(fig2)]
    metrics["provenance"] = {
        "git_head_at_start": head_at_start,
        "script": str(Path(__file__).resolve()),
        "script_sha256_16": sha16_of(Path(__file__).resolve()),
        "instrument": "e228.margin_pass module-imported VERBATIM",
        "batteries": "e182c_forgetting_control + e182c2_template "
                     "module-imported VERBATIM (via e228's import pattern)",
        "classification_rules": "e243_mode_selector module-imported VERBATIM "
                                "(CLASS_RULES_VERBATIM + thresholds + z "
                                "convention)",
        "death_order": "e230's committed operationalization (FLIP_ZONE 0.05, "
                       "P_HALF 0.5 — quoted, not imported: e230 side-effects "
                       "at import); validated by G_E230's exact reproduction",
        "journal_loaders": "e245_death_depth.load_journal module-imported",
        "versions": {"torch": torch.__version__,
                     "transformers": org_meta["transformers_version"],
                     "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "threads": torch.get_num_threads(), "device": "cpu fp32",
    }
    write_metrics("DONE")
    log(f"outputs: {rd / 'metrics.json'}, {fig}, {fig2}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


def answer_ids_b(bats, b):
    return {r["fact"]: r["ans_id"] for r in bats[b]}


# ------------------------------------------------------------------ figures
def make_map_figure(rd, verdict, clause, map_read, gates):
    """THE ORDER-MODE MAP: the order x mode matrix (w3 + pooled), the thin
    side's growth, and the verdict."""
    modes = ["MIS-DIAL", "FREQUENCY-COLLAPSE", "OTHER"]
    rows = ["P-FIRST", "non-P-FIRST"]
    w3c = map_read["w3_counts"]
    pc = map_read["pooled_counts_w1w2w3"]
    w3m = np.array([[w3c.get(f"{r}|{m}", 0) for m in modes] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4),
                             gridspec_kw={"width_ratios": [1.5, 1, 1.3]})

    ax = axes[0]
    ax.imshow(w3m, cmap="Blues", vmin=0, vmax=max(1, w3m.max()))
    for i, r in enumerate(rows):
        for j, m in enumerate(modes):
            w3n = w3c.get(f"{r}|{m}", 0)
            pn = pc.get(f"{r}|{m}", 0)
            viol_cell = (r == "non-P-FIRST" and m != "MIS-DIAL") or \
                        (r == "P-FIRST" and m == "MIS-DIAL")
            ax.text(j, i, f"w3: {w3n}\npooled: {pn}", ha="center",
                    va="center", fontsize=12, fontweight="bold",
                    color="white" if w3n > w3m.max() * 0.6 else "black")
            if viol_cell:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           fill=False, edgecolor="red",
                                           lw=3))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["mis-dial", "frequency\ncollapse", "other"])
    ax.set_yticks(range(2))
    ax.set_yticklabels(["P-FIRST\n(belief first)", "non-P-FIRST\n(commitment/"
                        "pair first\n/ neither)"])
    ax.set_title("THE LAYER-ORDER MAP on w3 (red outline = the literal "
                 "bar's violation cells)", fontsize=10)

    ax = axes[1]
    ts = map_read["thin_side"]
    bars = ax.bar([0, 1, 2],
                  [ts["w1w2_nonPF_n"], ts["w3_nonPF_n"], ts["pooled_nonPF_n"]],
                  color=["#8e9aaf", "#cb769e", "#1f6f43"], width=0.6)
    for x, v in zip([0, 1, 2],
                    [ts["w1w2_nonPF_n"], ts["w3_nonPF_n"],
                     ts["pooled_nonPF_n"]]):
        ax.text(x, v + 0.05, str(v), ha="center", fontsize=12,
                fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["w1/w2\n(e245 committed)", "w3\n(this cell)",
                        "pooled"], fontsize=9)
    ax.set_ylabel("non-P-FIRST flip records")
    ax.set_title(f"THE THIN SIDE — grew: {ts['grew']}", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    ax.axis("off")
    ax.text(0.02, 0.97, f"E247 VERDICT: {verdict}", fontsize=12.5,
            fontweight="bold", va="top", color="darkred", wrap=True)
    import textwrap
    y = 0.86
    for ln in textwrap.wrap(clause, width=58)[:12]:
        ax.text(0.02, y, ln, fontsize=7.6, va="top", wrap=True)
        y -= 0.045
    y -= 0.02
    ax.text(0.02, y, "gates: " + "  ".join(
        f"{k}={'P' if v.get('pass') else 'F'}" for k, v in gates.items()),
        fontsize=6.4, va="top", family="monospace", wrap=True)

    fig.suptitle("E247 — THE W3 FLIP CENSUS + THE LAYER-ORDER MAP: do the "
                 "two death modes stay the two death orders on a third wash?"
                 "\ncommitted map (e245/T222): non-P-FIRST => mis-dial AND "
                 "collapse => P-FIRST (P-FIRST mis-dials allowed); the "
                 "LITERAL bar adjudicates", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    png = rd / "e247_order_mode_map.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


def make_walker_figure(rd, walker_rows, walker_read):
    """THE WALKER TRAJECTORIES: every flip record's argmax trajectory at the
    finest available grid, role-coded; walkers highlighted; the rates."""
    role_color = {"ANSWER": "#1f8b4c", "RUNNER-UP": "#1f6fb2",
                  "FREQ": "#b23a48", "TARGET": "#b8860b", "OTHER": "#8a8a8a"}

    def role(row, t):
        if t["top1_id"] == row["t0_top1_id"]:
            return "ANSWER"
        if t["top1_id"] == row["runner_up_id"]:
            return "RUNNER-UP"
        if t["top1_id"] == 262:
            return "FREQ"
        if t["top1_id"] == row["target_id_80"]:
            return "TARGET"
        return "OTHER"

    order = sorted(walker_rows, key=lambda r: (
        {"w1": 0, "w2": 1, "w3": 2}[r["wash"]],
        {"FREQUENCY-COLLAPSE": 0, "MIS-DIAL": 1, "OTHER": 2}[r["classification"]],
        r["probe"]))
    n_lanes = max(1, len(order))
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(16.5, 0.42 * n_lanes + 3),
                                  gridspec_kw={"width_ratios": [3, 1]})
    for y, row in enumerate(order):
        walker = row["passed_through_RU_before_80"]
        for x, t in enumerate(row["argmax_trajectory"]):
            r = role(row, t)
            ax.scatter(x, y, s=200 if walker else 120, marker="o",
                       color=role_color[r], edgecolor="k",
                       linewidth=1.6 if walker else 0.6, zorder=3)
            ax.annotate(t["token"].strip()[:10], (x, y),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=6.0, ha="center", alpha=0.9)
        ax.annotate("", xy=(len(row["argmax_trajectory"]) - 0.55, y),
                    xytext=(-0.45, y),
                    arrowprops=dict(arrowstyle="->", color="#555",
                                    lw=0.9 if walker else 0.5, alpha=0.7,
                                    linestyle="-" if walker else "--"))
        label = (f"{row['wash']} {row['probe'].split('->')[0][:20]}"
                 f" [{row['classification'][:6]}|{row['death_order_class'][:5]}]"
                 + (" WALKER" if walker else ""))
        ax.text(-0.35, y, label, ha="right", va="center", fontsize=6.8,
                fontweight="bold" if walker else "normal",
                color="#b23a48" if walker else "black")
    ax.set_xlim(-0.6, len(W3_TRAJ) + 0.1)
    ax.set_ylim(-0.8, n_lanes - 0.2)
    ax.invert_yaxis()
    ax.set_xticks(range(4))
    ax.set_xticklabels(["1st", "2nd", "3rd", "4th"], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("state ordinal on the wash's own grid (w1: +2,+10,+50,+80; "
                  "w2/w3: +10,+50,+80)")
    ax.set_title("THE WALKER READ — every flip's argmax trajectory, "
                 "role-coded (green=answer, blue=runner-up, red='the', "
                 "gold=final target, gray=other); WALKER = visited the "
                 "runner-up strictly before +80", fontsize=9.5)

    axr.axis("off")
    y = 0.95
    axr.text(0.02, y, "walker rate among collapses\n(strictly-before-80 RU "
             "visit; NO bar)", fontsize=9, va="top", fontweight="bold")
    y -= 0.14
    for k in ("pooled", "w1", "w2", "w3"):
        r = walker_read[k] if k == "pooled" else walker_read["per_wash"][k]
        rate = r["walker_rate"]
        axr.text(0.02, y,
                 f"{k}: {r['of_which_passed_through_RU']}/"
                 f"{r['n_collapses']}"
                 + (f" = {rate:.0%}" if rate is not None else ""),
                 fontsize=9, va="top", family="monospace")
        y -= 0.075
    y -= 0.03
    axr.text(0.02, y, "excursion read (looser):\nany strictly-before state "
             "off the answer-final axis", fontsize=8, va="top")
    y -= 0.115
    for k in ("pooled", "w1", "w2", "w3"):
        r = walker_read[k] if k == "pooled" else walker_read["per_wash"][k]
        rate = r["excursion_rate"]
        axr.text(0.02, y,
                 f"{k}: {r['of_which_transient_excursion']}/"
                 f"{r['n_collapses']}"
                 + (f" = {rate:.0%}" if rate is not None else ""),
                 fontsize=8, va="top", family="monospace")
        y -= 0.065

    fig.suptitle("E247 — the walker trajectories: is the mis-dial a STAGE on "
                 "the way to collapse?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0.26)
    png = rd / "e247_walker_trajectories.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


if __name__ == "__main__":
    sys.exit(main())
