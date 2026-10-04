"""E229 — W032's THE WALL'S CURRENCY: DECISIONS OR GRADIENTS (eval-only, CPU).

THE QUESTION (W032's composition, quoted): e225's GRADED join showed the
flat phase does NOT track the gradient-level currency (the SNR multiple)
— the cons pair g1e/g1f sits at near-equal multiples (0.60/0.56) with
opposite retentions (0.615/0.932). W032 registered the flip candidate:
does the flat phase track a DECISION-level currency instead — the root's
margin aggregate?

W032's REGISTERED PREDICTION (honored VERBATIM, no retrofit): "within the
cons pair, the retention order (g1f 0.932 > g1e 0.615) is matched by the
margin-aggregate order of their fact batteries; an anti-match or a
scatter kills the composition and leaves the flat phase owning a third
currency still unnamed (the honest branch)."

HONESTY OWED UP FRONT: e228 (T207) just showed margins do NOT order the
124M EROSION order — this cell asks a DIFFERENT question, the WALL ROOTS'
flat-phase retention, where the currency is unknown. Both facts ride the
same page; neither is collapsed into the other. The e208 distinction
RESTATED (T207/W031's convention): e208's margin object was the FACT-EDGE
over the wash band — that instrument died its honest scope death; THIS
cell's object is the NEXT-TOKEN ARGMAX MARGIN in sigma units vs
arithmetic noise ((top1-top2)/std of vocab logits at the answer position)
— a different ruler that shares only the beloved word "margin". Not a
resurrection of e208.

REGISTERED BARS (frozen from W032 verbatim, in this docstring BEFORE any
compute; adjudicate against exactly this; no bar shopping):
  - DECISIONS-OWN-THE-FLAT-PHASE — "retention tracks the margin
    aggregate where it failed to track the multiple: the overall Spearman
    vs the aggregate clears 0.714 (the e225 line) AND the cons pair
    separates in y AND in the aggregate in matching order — the wall's
    currency = decisions; W032's prediction confirmed"
  - GRADIENTS-KEEP-THE-FLAT-PHASE — "the aggregate does no better than
    the multiple (Spearman <= e225's 0.607, or the cons pair matches in
    NEITHER), or anti-matches — the flat phase's currency stays
    gradient-side or unnamed; the honest branch"
  - PARTIAL — "anything between — both tables verbatim, no narrative
    inflation"

OPERATIONALIZATIONS (frozen BEFORE compute; they fix the clauses, they do
not move the bars):
  * THE CELL: for each of the seven e225 roots (J1 e131 consolidated /
    J2 g1c fresh / J3 g1d half-expressed / J4 g1e cons-1 / J5 g1f cons-2 /
    J6 the 10M take5 root 0.7677 / J7 the 10M take6 root 0.9351; loading
    conventions = lab/e225_one_currency.py VERBATIM by MODULE IMPORT —
    its ROSTER, load_body, battery_cell, GEOS/PRE constants), compute the
    FACT battery's per-probe argmax margins in sigma at the ROOT state
    (t=0, pre-wash): margin_sigma = (top1 logit - top2 logit)/
    std(vocab logits) at the last position (the answer position; Z), the
    organism's OWN decision margin whether or not the argmax is Z —
    e228's instrument (lab/e228_margin_landscape.py margin_pass) PORTED
    BY MODULE IMPORT, arithmetic untouched, through a thin adapter shim
    (TinyGPT's (logits, loss) tuple -> the .logits interface); never
    retyped.
  * THE FACT BATTERY: the install-60 g-12 ruler battery (60 windows,
    SPLICE_RNG 24301, the family's own battery) — the SAME windows whose
    collective p(Z) defines each root's committed root_gm12 and e225's
    kill-D walks. The 10M roots read the SAME battery content per the
    g1bS conventions (e225's own committed reading: g1bS5/G_SPLICE's own
    convention, content-identical install-60 across scales — the scale
    axis; NO battery dialect is mixed across scales; the g0/g+12
    geometries are computed as texture co-reports only, never
    adjudicated).
  * THE AGGREGATES (frozen NOW, before compute): primary = the fact
    battery's MEDIAN margin in sigma (e228's own median statistic,
    module-imported); co-report = the 25th percentile (numpy linear
    interpolation) — the thin tail. Both recorded for every root.
  * THE JOIN: the seven (margin-aggregate, flat-phase retention) pairs —
    Spearman overall (e228's spearman, module-imported; the e225 line
    0.714 = the one-sided 5% critical value at n=7; e225's observed
    multiple-rho 0.607 = the does-no-better line); and W032's sharpest
    cell adjudicated EXPLICITLY: cons_separates_y := ret(g1f) > ret(g1e)
    (desk-fixed true: 0.9317 > 0.6154); cons_match := agg(g1f) > agg(g1e)
    (strict). Co-report the same join against the multiple (x from
    e225, read at runtime from runs/e225/metrics.json, hard-bound) on
    the same page — the question is RELATIVE tracking, so both scatters
    side by side.
  * BAR CLAUSES FIXED: DECISIONS-OWN-THE-FLAT-PHASE := (rho_agg >= 0.714)
    AND cons_match. GRADIENTS-KEEP-THE-FLAT-PHASE := (rho_agg <= 0.607)
    OR (NOT cons_match) OR (rho_agg <= -0.714, the anti-match clause —
    subsumed by the 0.607 clause, stated for the record). PARTIAL :=
    otherwise (the gap 0.607 < rho_agg < 0.714 WITH cons_match). The
    first two are mutually exclusive by construction; composite order
    DECISIONS / GRADIENTS / PARTIAL.
  * CO-REPORTS (never adjudicated): Spearman(p25-aggregate, retention);
    the flat-median retention flavor join; Spearman(agg, multiple) (is
    the aggregate just the multiple in disguise?); Spearman(agg,
    root_gm12) + Spearman(root_gm12, retention) (is the aggregate just
    fact strength? — the margin-vs-p analog of e228's coupling reflex);
    Kendall tau-b beside every rho; leave-one-out rho; per-root
    frac_argmax_z; the fraction of probes under T204's ~0.05 sigma flip
    zone.

CHECKS (the dispatch's letter): the roots' provenance — each must
bit-reproduce its committed cell's root reads (e225's G_ROOT gates
ported verbatim: param count + the CPU battery read vs the committed
root_gm12, tol 5e-3; the protocol gates G_NAMEFREE/G_SPLICE/G_BATTERY/
G_ANCHOR ported from e225's P0a); margins deterministic (T204: same code
path, same device -> bit-exact) so n=1 evals suffice, eval provenance
recorded (batch shape 1 x L disclosed — batch-shape re-rounding sits at
the ~3e-7 texture floor, far under every margin adjudicated here; the
thin-tail co-report prices where that caveat bites); nothing guaranteed.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced before torch;
the GPU lane belongs to e227 — never claimed), torch threads 4,
load-checks recorded, one root = one eval burst (60 probes x 3
geometries), progressive metrics.json writes after every phase,
n=1 per root (deterministic), decisive; minutes.

Outputs: runs/e229/{metrics.json (PROGRESSIVE), e229_wall_currency.png}.
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python e229_wall_currency.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e225/e229 convention)
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # e228's offline convention (imported)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                     # noqa: E402 — the p25 convention
import torch                                            # noqa: E402

import common                                           # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,           # noqa: E402
                    run_dir, save_json)

import e043_install as E43                              # noqa: E402 — REPO, find_occ, SPLICE_RNG, jsonable
import e225_one_currency as E225                        # noqa: E402 — THE roster + loads + gates (VERBATIM)
import e228_margin_landscape as E228                    # noqa: E402 — THE margin instrument (VERBATIM)

torch.set_num_threads(4)                                # the dispatch envelope (e225/e228's own)

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e229 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------------ frozen constants
RUNS = E43.REPO / "runs"
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E225_METRICS = RUNS / "e225" / "metrics.json"
G_READ_TOL = 5e-3          # e225's G_ROOT tolerance (CPU read vs committed CUDA root cells)
RHO_DECISIONS_LINE = 0.714  # the e225 one-sided 5% Spearman critical line at n=7
RHO_E225_MULTIPLE = 0.6071428571428571  # e225's COMMITTED observed multiple-rho (the no-better line)
T204_FLIP_SIGMA = 0.05     # T204's batch-shape flip threshold (co-report dial)

# e225's committed joined table, HARD-BOUND (every number at its path; Rule 12).
# Read at runtime from runs/e225/metrics.json and asserted against these literals.
E225_TABLE = {
    "J1": {"ret": 0.9979692766675983, "retmed": 1.0027290489458747, "mult": 3.7206885127780804},
    "J2": {"ret": 1.0263911969648605, "retmed": 1.037891435414073, "mult": 1.2161096471128718},
    "J3": {"ret": 1.3266428034732003, "retmed": 1.3448413635063021, "mult": 1.3789356098383065},
    "J4": {"ret": 0.6153917590189493, "retmed": 0.6979575531343495, "mult": 0.5995813346448845},
    "J5": {"ret": 0.9317064573129114, "retmed": 0.9446859198099864, "mult": 0.5617260108399429},
    "J6": {"ret": 0.8949634049389881, "retmed": 0.9646209309477912, "mult": 0.5433506780298486},
    "J7": {"ret": 0.7897980962997516, "retmed": 0.832540454235837, "mult": 1.0231918306111651},
}
ROOT_TAG = {"J1": "e131", "J2": "g1c", "J3": "g1d", "J4": "g1e", "J5": "g1f",
            "J6": "take5", "J7": "take6"}

REGISTERED_BARS = {
    "DECISIONS_OWN_THE_FLAT_PHASE": 'DECISIONS-OWN-THE-FLAT-PHASE — "retention '
        'tracks the margin aggregate where it failed to track the multiple: the '
        'overall Spearman vs the aggregate clears 0.714 (the e225 line) AND the '
        'cons pair separates in y AND in the aggregate in matching order — the '
        'wall\'s currency = decisions; W032\'s prediction confirmed"',
    "GRADIENTS_KEEP_THE_FLAT_PHASE": 'GRADIENTS-KEEP-THE-FLAT-PHASE — "the '
        'aggregate does no better than the multiple (Spearman <= e225\'s 0.607, '
        'or the cons pair matches in NEITHER), or anti-matches — the flat '
        'phase\'s currency stays gradient-side or unnamed; the honest branch"',
    "PARTIAL": 'PARTIAL — "anything between — both tables verbatim, no '
        'narrative inflation"',
    "W032_registered_prediction_verbatim": "within the cons pair, the retention "
        "order (g1f 0.932 > g1e 0.615) is matched by the margin-aggregate order "
        "of their fact batteries; an anti-match or a scatter kills the "
        "composition and leaves the flat phase owning a third currency still "
        "unnamed (the honest branch).",
    "registration": "bars frozen from W032 verbatim in the module docstring "
                    "BEFORE any compute (script committed pre-run); adjudicate "
                    "against exactly this; no bar shopping.",
    "clause_fixes": "DECISIONS := (rho_agg >= 0.714) AND cons_match; GRADIENTS "
                    ":= (rho_agg <= 0.607) OR (NOT cons_match) OR (rho_agg <= "
                    "-0.714, subsumed); PARTIAL := otherwise (the gap 0.607 < "
                    "rho_agg < 0.714 WITH cons_match); cons_separates_y := "
                    "ret(g1f) > ret(g1e) [desk-fixed true]; cons_match := "
                    "agg(g1f) > agg(g1e) [strict, rank-level].",
}

deviations: list[str] = [
    "EVAL-ONLY on the seven committed pristine roots (no wash, no training; "
    "the GPU lane untouched — CPU-only by dispatch, e227 owns the GPU).",
    "The margin x-side is computed FRESH here (no committed margins exist at "
    "these roots); the y-side (retentions) and the comparison x-side "
    "(multiples) are e225's COMMITTED joined table, read at runtime and "
    "hard-bound to 1e-12 (G_PARENTS).",
    "e228's margin_pass is MODULE-IMPORTED and driven through a thin adapter "
    "(TinyGPT -> .logits namespace); its arithmetic is untouched. Importing "
    "e228 opens its runs/e228_run.log in append mode as a module side effect "
    "— nothing is written to it by this cell (own stdout log).",
    "The battery protocol rebuild (P0a) re-runs e225's own construction from "
    "its module constants (E43.SPLICE_RNG, corpus.encode) — same code path, "
    "same seeds -> bit-identical windows, gated by G_SPLICE/G_BATTERY/"
    "G_ANCHOR against e225's committed values.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "POST-FIRST-RUN REPORTING FIX (numbers untouched): the first pass's "
    "generated GRADIENTS clause omitted the cons-pair cell's outcome (which "
    "MATCHED); the clause generator now ALWAYS states the pair read in that "
    "branch, and two honesty entries (w032_pair_outcome, "
    "aggregate_is_fact_strength) were added. A second reporting fix: "
    "gates_summary mis-stamped G_ROOT FAIL (per-root records carry the pass "
    "flag; the summary now aggregates ALL-roots-pass). Deterministic re-runs "
    "(T204): every number identical; the bars and all computed values "
    "unchanged; disclosed here rather than silently patched.",
]


# ------------------------------------------------------------------ the shim
class _E228NetShim:
    """Adapter only — TinyGPT(idx) -> (logits, loss) becomes the transformers-
    style net(input_ids=...) -> .logits object e228's margin_pass expects.
    NO arithmetic lives here; the margin computation stays e228's, imported."""

    def __init__(self, net: TinyGPT):
        self.net = net

    def eval(self):                     # e228's margin_pass calls net.eval()
        self.net.eval()
        return self

    def __call__(self, input_ids=None):
        lg, _ = self.net(input_ids)
        return SimpleNamespace(logits=lg)


def cpu_load_probe() -> int | None:
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).LoadPercentage"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


def git_head() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(E43.REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unavailable"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e229")
    metrics: dict = {
        "experiment": "e229_wall_currency",
        "phase": "W032's discriminating observation — the wall's flat-phase "
                 "currency: decisions (margin aggregate) or gradients (the "
                 "multiple)?",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": False,
        "envelope": {
            "device": "CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced; GPU never "
                      "claimed — e227 owns the GPU lane)",
            "torch_threads": torch.get_num_threads(),
            "bursts": "one root = one eval burst (60 probes x 3 geometries)",
            "n_evals": "n=1 per root (margins deterministic in-session, T204)",
            "cpu_load_pct_at_launch": cpu_load_probe(),
        },
        "deviations": deviations,
        "builds_on": [
            "W032 (the composition card this cell IS the registered "
            "discriminating observation of)",
            "T207 / e228 (the argmax-margin instrument, module-imported; and "
            "the honesty note: margins do NOT order the 124M erosion order — "
            "a different question)",
            "T206 + UPDATE / e225 (the graded join, the cons pair, the seven "
            "roots, the retentions and multiples — the y-side and the "
            "comparison x-side)",
            "T204 / x3 (determinism + the arithmetic floor + the ~0.05 sigma "
            "batch-shape flip zone)",
            "T186/T188/T194/T170/T178 (the wall family's committed flat "
            "phases behind e225's table)",
        ],
        "whats_new": [
            "the per-probe argmax margin in sigma at the seven pristine wall "
            "roots (the fact battery's decision landscape, t=0)",
            "the margin-aggregate join vs the committed flat-phase retention "
            "— W032's decisions-or-gradients flip, with the cons pair as the "
            "sharpest cell",
            "the side-by-side page: retention-vs-aggregate and "
            "retention-vs-multiple on the same axes for direct comparison",
        ],
    }

    def write_partial(note: str):
        metrics["date"] = common.now_iso()
        metrics["phase_note"] = note
        save_json(rd / "metrics.json", E43.jsonable(metrics))
        log(f"WROTE partial metrics ({note})")

    log(f"E229 — W032's THE WALL'S CURRENCY (decisions or gradients) -> {rd}")

    # ================= P0a: the battery protocol, e225's P0a VERBATIM ========
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    corpus_zeph = train_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "pass": bool(corpus_zeph == 0)}
    assert G_NAMEFREE["pass"], f"corpus contains ZEPH x{corpus_zeph}"

    host_occ = []
    for host in E225.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + E225.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix,
                "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    bat_ids = {}
    for j in E225.GEOS:
        cs = [train_text[p - E225.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in E225.GEOS},
        "pass": bool(list(bat_ids[-12].shape) == [60, E225.PRE - 12]
                     and list(bat_ids[0].shape) == [60, E225.PRE]
                     and list(bat_ids[12].shape) == [60, E225.PRE + 12]),
        "note": "install-60 battery at ctx offsets {-12,0,+12} — e225's own "
                "construction re-run from its module constants; the g-12 "
                "ruler is THE fact battery (the family's own); the 10M "
                "roots read the SAME content (g1bS5/G_SPLICE's own "
                "convention, e225's committed reading) — no dialect mixed "
                "across scales",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"

    arng = _random.Random(E225.E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - E225.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        txt = train_text[s: s + E225.BLOCK + 1]
        if any(f in txt for f in E225.ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    G_ANCHOR = {"n_windows": 16, "tries": tries,
                "bank_starts_match_e185_stored":
                    bool(n_starts == E225.E170_BANK_STARTS)}
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"

    ruler_ids = bat_ids[E225.RULER_J]
    metrics["gates"] = {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                        "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR}
    log("P0a: protocol gates PASS (e225's P0a ported; splice 19+41; "
        "battery shapes; e170 bank bit-match)")

    # ================= P0b: G_PARENTS — e225's committed table hard-bound =====
    if not E225_METRICS.exists():
        raise RuntimeError(f"missing parent table: {E225_METRICS}")
    e225m = json.loads(E225_METRICS.read_text(encoding="utf-8"))
    jt = {r["id"]: r for r in e225m["joined_table"]}
    binds, ok = {}, True
    for jid, lit in E225_TABLE.items():
        r = jt[jid]
        d_ret = abs(r["retention_flat_min"] - lit["ret"])
        d_med = abs(r["retention_flat_median"] - lit["retmed"])
        d_mul = abs(r["multiple"] - lit["mult"])
        binds[jid] = {"d_ret": d_ret, "d_retmed": d_med, "d_mult": d_mul}
        ok &= (d_ret < 1e-12 and d_med < 1e-12 and d_mul < 1e-12)
    rho_e225 = e225m["adjudication"]["rank_layer"]["spearman"]
    G_PARENTS = {
        "e225_metrics": {"path": str(E225_METRICS),
                         "md5": md5of(E225_METRICS),
                         "sha256_16": sha256_of(E225_METRICS)},
        "hardbound": binds,
        "e225_committed_multiple_rho": rho_e225,
        "assert_rho_literal": abs(rho_e225 - RHO_E225_MULTIPLE) < 1e-12,
        "e225_verdict": e225m["adjudication"]["verdict"],
        "e228_verdict_composed": json.loads(
            (RUNS / "e228" / "metrics.json").read_text(encoding="utf-8")
        )["adjudication"]["bars"]["verdict"],
        "pass": bool(ok and abs(rho_e225 - RHO_E225_MULTIPLE) < 1e-12),
        "note": "e225's joined table (the y-side AND the comparison x-side) "
                "read at runtime and asserted against the literals frozen in "
                "this script pre-compute; e225's committed rho 0.6071428... "
                "= the does-no-better line; the 0.714 line = the one-sided 5% "
                "Spearman critical value at n=7 (e225's own table)",
    }
    assert G_PARENTS["pass"], f"e225 table bind failed: {binds}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — e225 md5 {G_PARENTS['e225_metrics']['md5']}; "
        f"7 rows bound to 1e-12; e225 rho {rho_e225:.4f}")
    write_partial("P0 protocol + parent gates PASSED")

    # the margin battery: e228's probe dicts over the ruler windows
    def probe_battery(ids: torch.Tensor, geo: int) -> list[dict]:
        return [{"ids": ids[i: i + 1],
                 "fact": f"install{i:02d}@g{geo:+d}",
                 "relation": f"install60_g{geo:+d}",
                 "ans_id": zid}
                for i in range(ids.shape[0])]

    # ================= P1: the seven root cells ==============================
    cells = {}
    g_root_rec = {}
    for spec in E225.ROSTER:
        rid = spec["id"]
        cfg = Cfg(**spec["cfg"]) if spec["cfg"] else Cfg()
        net, sd, meta = E225.load_body(CKPT_DIR / spec["ckpt"], cfg)
        n_par = sum(p.numel() for p in net.parameters())
        theta0 = E225.flat_params(net)
        root_read = E225.battery_cell(net, ruler_ids, zid)["mean_pz"]
        rdev = abs(root_read - spec["committed_root_gm12"])
        G_ROOT = {
            "checkpoint": f"runs/checkpoints/{spec['ckpt']}",
            "n_params": n_par, "expected_params": spec["n_params"],
            "battery_read_measured": root_read,
            "battery_read_committed": spec["committed_root_gm12"],
            "abs_diff": rdev, "tol": G_READ_TOL,
            "flat_md5": hashlib.md5(theta0.numpy().tobytes()).hexdigest(),
            "pass": bool(n_par == spec["n_params"] and rdev < G_READ_TOL),
        }
        log(f"G_ROOT[{rid}] ({ROOT_TAG[rid]}): {n_par} params; battery "
            f"{root_read:.6f} vs committed {spec['committed_root_gm12']:.6f} "
            f"(|d| {rdev:.1e}): "
            + ("PASS" if G_ROOT["pass"] else "FAIL"))
        if not G_ROOT["pass"]:
            raise RuntimeError(f"root gate FAILED at {rid}")
        g_root_rec[rid] = G_ROOT

        # THE MARGIN PASS — e228's instrument, module-imported, on the shim
        shim = _E228NetShim(net)
        cell = {"id": rid, "tag": ROOT_TAG[rid], "scale": spec["scale"],
                "organism": spec["organism"], "ckpt": spec["ckpt"],
                "G_ROOT": G_ROOT}
        for geo in E225.GEOS:                    # g-12 = THE fact battery
            rec = E228.margin_pass(shim, probe_battery(bat_ids[geo], geo))
            ms = [r["margin_sigma"] for r in rec["probes"]]
            entry = {
                "n": len(ms),
                "mean": rec["mean_margin_sigma"],
                "median": rec["median_margin_sigma"],
                "p25": float(np.percentile(ms, 25)),
                "min": rec["min_margin_sigma"],
                "frac_below_T204_flip_zone":
                    float(sum(1 for m in ms if m < T204_FLIP_SIGMA) / len(ms)),
                "frac_argmax_z": float(sum(
                    1 for r in rec["probes"] if r["top1_id"] == zid) / len(ms)),
                "mean_pz": rec["mean_p"],
                "probes": [{"fact": r["fact"], "p": r["p"],
                            "margin_sigma": r["margin_sigma"],
                            "top1_id": r["top1_id"], "top2_id": r["top2_id"],
                            "argmax_is_z": bool(r["top1_id"] == zid)}
                           for r in rec["probes"]],
            }
            if geo == E225.RULER_J:
                cell["fact_battery_g-12"] = entry       # THE adjudicated battery
            else:
                cell[f"texture_g{geo:+d}"] = {          # texture co-reports
                    "median": entry["median"], "p25": entry["p25"],
                    "mean": entry["mean"], "min": entry["min"]}
        cells[rid] = cell
        metrics["cells"] = cells
        a = cell["fact_battery_g-12"]
        log(f"  margins[{ROOT_TAG[rid]}]: median {a['median']:.4f}s p25 "
            f"{a['p25']:.4f}s mean {a['mean']:.4f}s min {a['min']:.4f}s "
            f"(argmax-Z {a['frac_argmax_z']:.2f}; <0.05s "
            f"{a['frac_below_T204_flip_zone']:.2f}) | texture g0 "
            f"{cell['texture_g+0']['median']:.4f}s g+12 "
            f"{cell['texture_g+12']['median']:.4f}s")
        write_partial(f"P1[{rid}] root gate + margins done")
        del net, sd, meta, theta0, shim

    metrics["gates"]["G_ROOT"] = {k: {"pass": v["pass"],
                                      "abs_diff": v["abs_diff"],
                                      "flat_md5": v["flat_md5"]}
                                  for k, v in g_root_rec.items()}
    metrics["gates"]["G_ENV"] = {
        "cpu_only": True, "torch_threads": torch.get_num_threads(),
        "gpu_calls": 0, "pass": bool(torch.get_num_threads() <= 4)}
    write_partial("P1 COMPLETE (all seven roots gated + margin-passed)")

    # ================= P2: THE JOIN ==========================================
    join_rows = []
    for spec in E225.ROSTER:
        rid = spec["id"]
        a = cells[rid]["fact_battery_g-12"]
        join_rows.append({
            "id": rid, "tag": ROOT_TAG[rid], "scale": spec["scale"],
            "aggregate_median": a["median"],
            "aggregate_p25": a["p25"],
            "retention": jt[rid]["retention_flat_min"],
            "retention_flat_median_flavor": jt[rid]["retention_flat_median"],
            "multiple_e225": jt[rid]["multiple"],
            "root_gm12": spec["committed_root_gm12"],
            "frac_argmax_z": a["frac_argmax_z"],
        })
    metrics["join_table"] = join_rows

    xs_agg = [r["aggregate_median"] for r in join_rows]
    xs_p25 = [r["aggregate_p25"] for r in join_rows]
    xs_mul = [r["multiple_e225"] for r in join_rows]
    xs_root = [r["root_gm12"] for r in join_rows]
    ys = [r["retention"] for r in join_rows]
    ys_med = [r["retention_flat_median_flavor"] for r in join_rows]

    rho_agg, ties_a = E228.spearman(xs_agg, ys)
    tau_agg = E228.kendall_tau_b(xs_agg, ys)
    rho_p25, _ = E228.spearman(xs_p25, ys)
    rho_mul, _ = E228.spearman(xs_mul, ys)          # must reproduce 0.6071
    rho_mul_med, _ = E228.spearman(xs_mul, ys_med)
    rho_agg_med, _ = E228.spearman(xs_agg, ys_med)
    rho_agg_mul, _ = E228.spearman(xs_agg, xs_mul)  # aggregate-vs-multiple
    rho_agg_root, _ = E228.spearman(xs_agg, xs_root)  # aggregate-vs-fact-strength
    rho_root_ret, _ = E228.spearman(xs_root, ys)
    loo = {}
    for i, r in enumerate(join_rows):
        xs_ = [v for j, v in enumerate(xs_agg) if j != i]
        ys_ = [v for j, v in enumerate(ys) if j != i]
        loo[r["id"]] = E228.spearman(xs_, ys_)[0]

    # W032's sharpest cell — the cons pair, adjudicated EXPLICITLY
    j4 = next(r for r in join_rows if r["id"] == "J4")   # g1e
    j5 = next(r for r in join_rows if r["id"] == "J5")   # g1f
    cons_separates_y = bool(j5["retention"] > j4["retention"])
    cons_match = bool(j5["aggregate_median"] > j4["aggregate_median"])
    cons_gap = j5["aggregate_median"] - j4["aggregate_median"]
    cons_ret_gap = j5["retention"] - j4["retention"]
    cons_p25_match = bool(j5["aggregate_p25"] > j4["aggregate_p25"])
    cons_cell = {
        "pair": "J4 = g1e (cons-1) vs J5 = g1f (cons-2)",
        "retention_g1e": j4["retention"], "retention_g1f": j5["retention"],
        "aggregate_median_g1e": j4["aggregate_median"],
        "aggregate_median_g1f": j5["aggregate_median"],
        "aggregate_p25_g1e": j4["aggregate_p25"],
        "aggregate_p25_g1f": j5["aggregate_p25"],
        "multiple_g1e": j4["multiple_e225"], "multiple_g1f": j5["multiple_e225"],
        "cons_separates_y": cons_separates_y,
        "cons_match_median": cons_match,
        "cons_match_p25": cons_p25_match,
        "aggregate_gap_median": cons_gap,
        "retention_gap": cons_ret_gap,
        "w032_prediction_verbatim":
            REGISTERED_BARS["W032_registered_prediction_verbatim"],
        "w032_pair_read": ("MATCH" if cons_match else
                           ("ANTI-MATCH" if j5["aggregate_median"]
                            < j4["aggregate_median"] else "TIE")),
        "note": "W032's registered prediction adjudicated at its own cell: "
                "MATCH = the aggregate orders the cons pair as the retention "
                "does (g1f > g1e); ANTI-MATCH/TIE = the honest branch fires "
                "(the flat phase's third currency stays unnamed); the p25 "
                "co-report rides beside, never adjudicating",
    }
    metrics["join_statistics"] = {
        "primary": {"spearman_aggregate_vs_retention": rho_agg,
                    "kendall_tau_b": tau_agg, "n": 7,
                    "ties": ties_a,
                    "decisions_line": RHO_DECISIONS_LINE,
                    "e225_multiple_rho_line": RHO_E225_MULTIPLE},
        "cons_pair_cell": cons_cell,
        "co_reports_never_adjudicated": {
            "spearman_p25_vs_retention": rho_p25,
            "spearman_multiple_vs_retention_recomputed": rho_mul,
            "spearman_multiple_vs_retention_flatmedian_flavor": rho_mul_med,
            "spearman_aggregate_vs_retention_flatmedian_flavor": rho_agg_med,
            "spearman_aggregate_vs_multiple": rho_agg_mul,
            "spearman_aggregate_vs_root_gm12": rho_agg_root,
            "spearman_root_gm12_vs_retention": rho_root_ret,
            "leave_one_out_spearman": loo,
            "note": "rho(agg, multiple) prices whether the aggregate is the "
                    "multiple in disguise; rho(agg, root_gm12) prices whether "
                    "it is just fact strength (e228's margin-vs-p coupling "
                    "reflex, ported to the join level); the median-flavor "
                    "retention joins guard the y-convention",
        },
    }
    log("P2 THE JOIN: " + "; ".join(
        f"{r['tag']} agg {r['aggregate_median']:.4f}s ret {r['retention']:.3f}"
        for r in join_rows))
    log(f"  rho(agg, ret) {rho_agg:+.4f} (tau {tau_agg:+.3f}); rho(mult, ret) "
        f"{rho_mul:+.4f} [e225 committed {RHO_E225_MULTIPLE:+.4f}]; cons pair "
        f"y {cons_separates_y} / agg-match {cons_match} "
        f"(gap {cons_gap:+.4f}s)")
    write_partial("P2 the join computed (adjudication next)")

    # ================= P3: the adjudication (frozen bars) ====================
    decisions_fires = bool(rho_agg >= RHO_DECISIONS_LINE and cons_match)
    gradients_fires = bool(rho_agg <= RHO_E225_MULTIPLE or not cons_match)
    partial_fires = not decisions_fires and not gradients_fires
    assert not (decisions_fires and gradients_fires), "bars must be exclusive"

    if decisions_fires:
        verdict = "DECISIONS-OWN-THE-FLAT-PHASE"
        clause = (f"retention tracks the margin aggregate where it failed to "
                  f"track the multiple: the overall Spearman vs the aggregate "
                  f"{rho_agg:+.3f} clears {RHO_DECISIONS_LINE} (the e225 "
                  f"line) AND the cons pair separates in y "
                  f"({cons_separates_y}; g1f {j5['retention']:.3f} > g1e "
                  f"{j4['retention']:.3f}) AND in the aggregate in matching "
                  f"order (g1f {j5['aggregate_median']:.4f}s > g1e "
                  f"{j4['aggregate_median']:.4f}s, gap {cons_gap:+.4f}s) — "
                  f"the wall's currency = decisions; W032's prediction "
                  f"confirmed (the multiple's rho stays {rho_mul:+.3f})")
    elif gradients_fires:
        why = []
        if rho_agg <= RHO_E225_MULTIPLE:
            why.append(f"the aggregate does no better than the multiple "
                       f"(rho {rho_agg:+.3f} <= e225's {RHO_E225_MULTIPLE:.3f})")
        if not cons_match:
            why.append(f"the cons pair does NOT match in the aggregate "
                       f"(g1e {j4['aggregate_median']:.4f}s vs g1f "
                       f"{j5['aggregate_median']:.4f}s — "
                       f"{cons_cell['w032_pair_read']}, against the retention "
                       f"order g1f {j5['retention']:.3f} > g1e "
                       f"{j4['retention']:.3f})")
        if rho_agg <= -RHO_DECISIONS_LINE:
            why.append("the join is significantly anti-monotone")
        # the pair cell is ALWAYS stated in this branch (honesty: the frozen
        # rho clause adjudicates, but W032's registered prediction LIVES at
        # the pair — its outcome is reported either way, never omitted)
        pair_txt = (f"the cons-pair cell itself reads "
                    f"{cons_cell['w032_pair_read']} (aggregate g1f "
                    f"{j5['aggregate_median']:.4f}s vs g1e "
                    f"{j4['aggregate_median']:.4f}s, gap {cons_gap:+.4f}s, "
                    f"against the retention order g1f {j5['retention']:.3f} "
                    f"> g1e {j4['retention']:.3f}) — W032's pair-level "
                    f"prediction "
                    + ("CONFIRMED even as the OVERALL join scatters "
                       "(a pair is a pair; the currency claim dies at n=7)"
                       if cons_match else "NOT confirmed"))
        verdict = "GRADIENTS-KEEP-THE-FLAT-PHASE"
        clause = ("; ".join(why) + "; " + pair_txt
                  + " — the flat phase's currency stays gradient-side or "
                  "unnamed; the honest branch (the frozen clauses "
                  "adjudicated as written; no bar shopping)")
    else:
        verdict = "PARTIAL"
        clause = (f"anything between — the aggregate clears the multiple's "
                  f"rho ({rho_agg:+.3f} > {RHO_E225_MULTIPLE:.3f}) and the "
                  f"cons pair matches ({cons_match}) but the overall Spearman "
                  f"sits under the {RHO_DECISIONS_LINE} line — both tables "
                  f"verbatim, no narrative inflation")

    gates_summary = {}
    for g, v in metrics["gates"].items():
        if g == "G_ROOT":      # per-root records — the gate is ALL roots pass
            gates_summary[g] = bool(v) and all(
                rr.get("pass") for rr in v.values())
        else:
            gates_summary[g] = bool(v.get("pass"))
    metrics["adjudication"] = {
        "bars": {"DECISIONS_OWN_THE_FLAT_PHASE": {"fires": decisions_fires},
                 "GRADIENTS_KEEP_THE_FLAT_PHASE": {"fires": gradients_fires},
                 "PARTIAL": {"fires": partial_fires}},
        "clause_fixes_applied": REGISTERED_BARS["clause_fixes"],
        "verdict": verdict, "clause": clause,
        "composite_order": "DECISIONS-OWN-THE-FLAT-PHASE / "
                           "GRADIENTS-KEEP-THE-FLAT-PHASE / PARTIAL (frozen "
                           "before compute; the first two mutually exclusive "
                           "by construction)",
        "gates_summary": gates_summary,
    }
    log("=" * 78)
    log(f"E229 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)
    write_partial("P3 adjudicated")

    # ================= P4: the figure — both scatters side by side ===========
    fig, axes = plt.subplots(1, 3, figsize=(20.5, 7.6),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})

    def scatter(ax, xs, xlabel, rho, tau, title):
        for r, x in zip(join_rows, xs):
            is10 = r["scale"] == "10M"
            ax.plot(x, r["retention"],
                    "*" if is10 else "o", ms=17 if is10 else 10,
                    color="darkorange" if is10 else "black", mec="k",
                    zorder=6, alpha=0.9)
            ax.annotate(r["tag"], (x, r["retention"]),
                        textcoords="offset points", xytext=(9, 5),
                        fontsize=10, weight="bold")
        # the cons pair highlighted: a ring + the retention-order arrow
        for r, x, col in ((j4, xs[join_rows.index(j4)], "crimson"),
                          (j5, xs[join_rows.index(j5)], "crimson")):
            ax.plot(x, r["retention"], "o", ms=26, mfc="none", mec=col,
                    mew=2.2, zorder=5)
        ax.annotate("", xy=(xs[join_rows.index(j5)],
                            j5["retention"] + 0.012),
                    xytext=(xs[join_rows.index(j4)], j4["retention"]),
                    arrowprops=dict(arrowstyle="->", color="crimson",
                                    lw=1.4, ls="--", alpha=0.8))
        ax.annotate("the cons pair\ng1e -> g1f (retention order)",
                    (xs[join_rows.index(j4)], j4["retention"] - 0.055),
                    fontsize=8.5, color="crimson")
        ax.axhline(1.0, ls=":", lw=1.0, color="gray")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("flat-phase retention (e225 committed, flat-min)")
        ax.set_ylim(0.52, 1.42)
        ax.grid(alpha=0.25)
        ax.set_title(title, fontsize=9.5)

    scatter(axes[0], xs_agg,
            "MARGIN AGGREGATE — fact battery median argmax margin (sigma)",
            rho_agg, tau_agg,
            f"DECISIONS' JOIN — retention vs the margin aggregate\n"
            f"Spearman {rho_agg:+.3f} (tau {tau_agg:+.2f}) vs the "
            f"{RHO_DECISIONS_LINE} line — the registered flip")
    scatter(axes[1], xs_mul,
            "GRADIENTS' JOIN — edge-multiple (e225 committed)",
            rho_mul, None,
            f"the multiple's join (e225's x, same seven rows)\n"
            f"Spearman {rho_mul:+.3f} (e225 committed "
            f"{RHO_E225_MULTIPLE:+.3f}) — the currency that FAILED")

    # the verdict + join table panel
    ax = axes[2]
    ax.axis("off")
    y = 0.97
    ax.text(0.03, y, f"E229 VERDICT: {verdict}", fontsize=11.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.055
    import textwrap
    for wd in textwrap.wrap(clause, width=64, break_long_words=False)[:9]:
        ax.text(0.03, y, wd, fontsize=7.0, va="top", family="monospace")
        y -= 0.024
    y -= 0.02
    ax.text(0.03, y, "root   agg_med(s)  agg_p25(s)  mult   ret", fontsize=7.4,
            va="top", family="monospace", weight="bold")
    y -= 0.026
    for r in sorted(join_rows, key=lambda r: r["aggregate_median"]):
        ax.text(0.03, y,
                f"{r['tag']:6s} {r['aggregate_median']:10.4f} "
                f"{r['aggregate_p25']:10.4f} {r['multiple_e225']:6.3f} "
                f"{r['retention']:6.3f}",
                fontsize=7.4, va="top", family="monospace")
        y -= 0.024
    y -= 0.015
    ax.text(0.03, y, f"cons pair: y g1f {j5['retention']:.3f} > g1e "
            f"{j4['retention']:.3f} | agg g1f "
            f"{j5['aggregate_median']:.4f} vs g1e "
            f"{j4['aggregate_median']:.4f} -> "
            f"{cons_cell['w032_pair_read']}", fontsize=7.2, va="top",
            family="monospace", color="crimson")
    y -= 0.035
    ax.text(0.03, y, "GATES: " + "  ".join(
        f"{g}={'PASS' if v else 'FAIL'}"
        for g, v in metrics["adjudication"]["gates_summary"].items()),
        fontsize=7.2, va="top", family="monospace")

    fig.suptitle("E229 — W032's THE WALL'S CURRENCY: DECISIONS OR GRADIENTS — "
                 "the flat-phase retention vs the margin aggregate (left) and "
                 "the multiple (right), cons pair highlighted",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = rd / "e229_wall_currency.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)

    # ================= P5: honesty + provenance + close ======================
    metrics["honesty"] = {
        "e208_distinction": "e208's margin object was the FACT-EDGE over the "
                            "wash band (an instrument that died its honest "
                            "scope death); THIS cell's object is the "
                            "NEXT-TOKEN ARGMAX MARGIN in sigma vs arithmetic "
                            "noise — a different ruler; do not conflate; not "
                            "a resurrection",
        "e228_composition_note": "e228 (T207) showed margins do NOT order "
                                 "the 124M erosion order — a DIFFERENT "
                                 "question (per-probe death order under one "
                                 "wash) from this one (the WALL ROOTS' "
                                 "flat-phase retention across organisms); "
                                 "both facts ride the page; neither collapses "
                                 "into the other",
        "determinism": "margins are deterministic in-session (T204: same code "
                       "path, same device -> bit-exact), so n=1 evals "
                       "suffice; the eval batch shape (1 x L, single-probe "
                       "forwards) is disclosed — batch-shape re-rounding sits "
                       "at the ~3e-7 texture floor, far under every aggregate "
                       "adjudicated here; the per-root frac below T204's "
                       "~0.05 sigma flip zone is co-reported",
        "aggregate_vs_its_rivals": f"the aggregate's own rivals are priced in "
                                   f"co-reports: rho(agg, multiple) "
                                   f"{rho_agg_mul:+.3f} (is it the multiple "
                                   f"in disguise?) and rho(agg, root_gm12) "
                                   f"{rho_agg_root:+.3f} (is it just fact "
                                   f"strength — e228's margin-vs-p reflex at "
                                   f"the join level); the root-gm12 join "
                                   f"itself: {rho_root_ret:+.3f}",
        "n_and_scope": "n=7 organisms (5 x 2.74M + 2 x 10M), n=1 margin read "
                       "each, ONE battery (install-60 g-12; g0/g+12 texture "
                       "only); the y-side is desk-fixed committed data "
                       "(e225's GRADED table, with J1's multiple "
                       "mixed-instrument provenance and J3's "
                       "observed-unadjudicated stamp carried on their rows); "
                       "the cons pair is n=2 — its cell is W032's sharpest "
                       "instrument but a pair is a pair",
        "w032_pair_outcome": ("W032's registered prediction CONFIRMED at its "
                             "own cell (aggregate g1f "
                             f"{j5['aggregate_median']:.4f}s > g1e "
                             f"{j4['aggregate_median']:.4f}s, "
                             "matching the retention order; the p25 flavor "
                             "matches too) — while the OVERALL join is "
                             f"negative ({rho_agg:+.3f}): the pair-level "
                             "confirmation is a residue the next card owns, "
                             "not a currency; LOO max "
                             f"{max(loo.values()):+.3f} "
                             f"(dropping {max(loo, key=loo.get)}) — no row's "
                             "removal brings the aggregate join near the "
                             "0.714 line"),
        "aggregate_is_fact_strength": f"rho(aggregate, root_gm12) "
                                      f"{rho_agg_root:+.3f}: at the pristine "
                                      f"roots the battery's MEDIAN decision "
                                      f"margin is essentially the root's fact "
                                      f"strength (mean p(Z)) in rank — the "
                                      f"margin aggregate is NOT a new axis "
                                      f"here but the old fact-strength axis "
                                      f"in decision clothes (rho(agg, "
                                      f"multiple) {rho_agg_mul:+.3f}: not the "
                                      f"multiple in disguise); and the flat "
                                      f"phase tracks fact strength no better "
                                      f"(rho(root_gm12, retention) "
                                      f"{rho_root_ret:+.3f}) — the breaker "
                                      f"rows are the formation lottery's "
                                      f"exotic members (g1d the half-"
                                      f"expressed 1.33x record at the table's "
                                      f"lowest aggregate; take6 the strong "
                                      f"root whose flat phase falls)",
        "cross_scale_rulers": "content-identical install-60 batteries read by "
                              "different hosts (6L/192d vs 8L/320d) — the "
                              "scale axis (e225's own disclosed caveat, "
                              "T182's instrument-shadow class); no battery "
                              "dialect is mixed across scales",
        "nothing_guaranteed": "the aggregates could have landed anywhere; "
                              "the observed outcome is recorded verbatim "
                              "against the frozen bars; no bar shopping",
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "e225_metrics": G_PARENTS["e225_metrics"],
        "checkpoints": {rid: {"file": f"runs/checkpoints/{spec['ckpt']}",
                              "flat_md5": g_root_rec[rid]["flat_md5"],
                              "n_params": g_root_rec[rid]["n_params"],
                              "battery_read": g_root_rec[rid]
                              ["battery_read_measured"]}
                        for spec, rid in zip(E225.ROSTER, g_root_rec)},
        "machinery": {
            "margin_instrument": "lab/e228_margin_landscape.py margin_pass "
                                 "MODULE-IMPORTED (adapter shim only; "
                                 "arithmetic untouched): margin_sigma = "
                                 "(top1-top2 logit)/std(vocab logits) at the "
                                 "answer position, torch std unbiased",
            "loads_and_gates": "lab/e225_one_currency.py MODULE-IMPORTED: "
                               "ROSTER, load_body, battery_cell, flat_params, "
                               "P0a protocol constants (G_NAMEFREE/G_SPLICE/"
                               "G_BATTERY/G_ANCHOR re-run; G_ROOT per root at "
                               "tol 5e-3)",
            "battery": "install-60 g-12 ruler (SPLICE_RNG 24301; 60 windows; "
                       "the family's own battery; same content at 10M per "
                       "the g1bS conventions)",
        },
        "eval": {"device": "cpu fp32", "threads": torch.get_num_threads(),
                 "batch_shape": "1 x L per probe (e228's margin_pass shape)",
                 "n_forwards": "7 roots x 3 geometries x 60 probes = 1260"},
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(rd / "metrics.json"), str(png)]
    write_partial("P5 DONE (honesty + provenance + figure)")
    log(f"outputs: {rd / 'metrics.json'}, {png}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
