"""E141 — the SINK-KEY MECHANISM BATTERY (R44 critic attacks 1/4/7 + ideator
e141 merged; REGISTERED).

WHY (T077 SECOND AMENDMENT, verbatim context): "THE CRACK WAS ALREADY IN THE
CENSUS, UNREAD. Row 0's consolidation delta is the SECOND-SMALLEST of all 256
wpe rows (delta_norm 0.0557 vs band median 0.1265) and its projection on the
fact axis (0.0124) sits AT the band median (0.0108): consolidation wrote
essentially nothing fact-specific INTO wpe[0]. The 0.545->0.732 strengthening
therefore lives in READOUT WEIGHTS keyed to whatever row 0 already was.
'Re-keyed to row 0' is DOWNGRADED to 'row-0-ROUTED': row 0's necessity may be
sink-ROLE necessity (the pivot every readout routes through), not a written
key." e131 proved row 0 NECESSARY (d_r0 -97%); this battery adjudicates
WHY: ROLE-ROUTED (sink-role the pivot readouts route through) vs WRITTEN-KEY
(consolidation's delta in wpe[0] is the key), plus presence-vs-content and
scaffold hardening.

NETS (on disk, gated; eval-only — no training, no fine-tunes):
  * CONSOLIDATED: runs/checkpoints/e131_consolidated_e113.pt (gate: install-60
    g0 battery p(Z) = 0.7850371599197388, CE_R = 1.663516640663147; e131
    bit-reproducible cell).
  * INSTALL-PHASE: runs/checkpoints/e048_repro.pt (source of install wpe[0];
    gate G_E048: g0 battery p(Z) = 0.5563086867332458).
  * R@150 / R@300: runs/checkpoints/e119_r_jittered_s150.pt (calibrated
    road-R arm; gates g0 = 0.5597274303436279, g-12 = 0.8130165338516235 —
    the novel-geometry generalization cell W011 missed) and
    e119_r_jittered_s300.pt (registered default; gate g0 = 0.7753651738166809
    from e119's calibration dial_default cell).

REGISTERED PREDICTION (QUEUE.md e141 row, VERBATIM — no bar shopping):
  "On e131_consolidated + e048_repro + e119 R@150/R@300: (1) INSTALL-RESTORE
  SURGERY — swap consolidated wpe[0] <- install wpe[0] with t-interpolation
  dose curve (WRITTEN-KEY: fact dies with delta removed while CE flat;
  ROLE-ROUTED: survives t=1, dies only under direction-scramble);
  (2) PRESENCE-VS-CONTENT — scramble first 1-16 tokens of eval contexts vs
  mid-context matched control (PRESENCE-KEY <20% cost; CONTENT-KEY >=50%
  beyond control); (3) SCAFFOLD HARDENING — delete rows 2-6 individually +
  norm-matched random, CE_R + expression each (only-row-0-wrecks => sink
  uniqueness); (4) d_r0 at NOVEL geometry g-12 on R@150/R@300 (sink-keyed
  dies / content-keyed survives); (5) GATE-VS-SOURCE — graded interpolation
  of wpe[0] consolidated->install, expression vs r (SOURCE-GRADED R^2>=0.8;
  GATE-THRESHOLD >=80% retained to r* then collapse)."

OPERATIONALIZATIONS (fixed before compute; coordinator dispatch + lab
conventions):
  * probe-1 curve: wpe[0](t) = t*install + (1-t)*consolidated, t in
    {0, 0.25, 0.5, 0.75, 1.0}; expression = install-60 g0 battery mean p(Z);
    "dies" = expr <= 0.5 x expr(0) (e131 collapse convention); "survives" =
    expr >= 0.8 x expr(0) (e131 survive convention); "CE flat" =
    |CE_R(t) - CE_R(0)| <= 0.35 (a quarter of row-0's own zero-arm CE cost
    3.0662 - 1.6635 = 1.4026 on this net).
  * direction-scramble arm: wpe[0] <- wpe[0][perm] with a seeded fixed
    permutation of the 192 embedding dims (norm-preserving, presence-
    preserving, direction-destroying); run at t=0 (consolidated row) and
    t=1 (install row). ROLE-ROUTED requires the t=0 scramble to kill
    (expr <= 0.5 x expr(0)).
  * probe-2 doses {1,2,4,8,16}; per-context seeded permutations, the SAME
    permutation applied at the front window [0,k) and the mid window
    [64, 64+k) (matched count, matched permutation); excess_pct =
    ((base - front) - (base - mid)) / base; bar read at k=16 (largest dose),
    full curve reported.
  * probe-3 "wrecks" = min(expr_zero, expr_mean) <= 0.5 x base_expr;
    "CE cost" = CE_arm - CE_none; scaffold-alternative bar = any non-row-0
    row with CE cost >= +1.4 (row-0's zero-arm CE cost) while expression
    intact (>= 0.8 x base).
  * probe-4 sink-keyed collapse = expr_d_r0(g-12) <= 0.5 x expr_none(g-12);
    content-keyed survival = >= 0.8 x.
  * probe-5 SOURCE-GRADED = linear fit R^2 >= 0.8 (slope < 0) WITH actual
    loss at t=1 (normalized expr(1.0) < 0.8 — both SOURCE-GRADED and
    GATE-THRESHOLD describe WAYS the fact dies along the curve; a curve that
    never drops below the survive bar reads NO-COLLAPSE, role-consistent);
    GATE-THRESHOLD = exists t* >= 0.25 with expr(t*) >= 0.8 x expr(0),
    expr(1.0) <= 0.5 x expr(0), and post/pre slope ratio >= 3x
    (segment slopes of the normalized curve around t*); GATE-THRESHOLD
    takes precedence when both fire.
  * overall ROLE-ROUTED vs WRITTEN-KEY is decided by probe (1) — the
    dispatched discriminator — with probes 2-5 as corroboration tallies;
    ambiguous => AMBIGUOUS with numbers.

INSTRUMENT PROVENANCE: battery/eval/surgery instruments are e131's
(lab/e131_rekeying_census.py) VERBATIM: battery_cell (e068/e113/e120),
battery_pz (e116), ce_fixed_cpu, val_windows (e065 seed 26502), deleted_wpe
(D2 subtractive row-zero + confinement gate), load_cpu/evl_load, and the
protocol rebuild (corpus seed 1337, SPLICE_RNG host shuffle, install-60 /
held-30 split, mix gate). Battery construction: ctx = train_text[p-PRE-j : p]
(read p(Z) at last position) exactly as e119 built its g-12 novel-geometry
cells.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import),
torch.set_num_threads(4) (modest — two other CPU agents live), all evals
sequential, no busy-waiting, no training.

Outputs: runs/e141/{metrics.json, sinkkey_mechanism.png}. No checkpoints
written (eval-only; every net consumed is on-disk and gated).

Run:  cd lab && python e141_sinkkey_mechanism.py     (E141_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (another agent owns the GPU)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(4)                              # modest (shared CPU)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E141_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
CONS_CK = CKPT_DIR / "e131_consolidated_e113.pt"   # consolidated (e113 recipe)
INST_CK = CKPT_DIR / "e048_repro.pt"               # install-phase (seed-42 line)
R150_CK = CKPT_DIR / "e119_r_jittered_s150.pt"     # calibrated road-R arm
R300_CK = CKPT_DIR / "e119_r_jittered_s300.pt"     # registered default road-R

# probe grids (smoke shrinks them)
T_GRID = (0.0, 0.5, 1.0) if SMOKE else (0.0, 0.25, 0.5, 0.75, 1.0)
DOSES = (1, 4, 16) if SMOKE else (1, 2, 4, 8, 16)
SCAFFOLD_ROWS = (2, 4) if SMOKE else (2, 3, 4, 5, 6)
R_NETS = ("R150",) if SMOKE else ("R150", "R300")

# geometry: g0 (trained by jitter road) + g-12 (e119's novel geometry,
# never trained by ANY arm; R@150 generalized 0.8130 there without deletion)
GEOS = (-12, 0)

# seeds (fixed before compute)
R_EVAL_SEED = 26502               # e065 CE_R bank seed (verbatim)
PERM_DIM_SEED = 14101             # direction-scramble permutation of wpe[0]
SCRAMBLE_SEED = 14102             # probe-2 per-context token permutations
NORMMATCH_SEED = 14103            # probe-3 norm-matched random row draw

# gates / references (full-precision, from stored metrics)
G_BIT_TOL = 5e-6                  # loaded-checkpoint reproduction gate
G_FALLBACK_TOL = 0.05             # e113 G_REPRO convention
G_CONS_REF_PZ = 0.7850371599197388          # e131 none__g+0__install60
G_CONS_REF_CE = 1.663516640663147           # e131 none__ce_r
G_E048_REF = 0.5563086867332458             # e131 G_E048 / e119 G_INST line
G_R150_REF_PZ = 0.5597274303436279          # e119 battery_table r_jittered g0
G_R150_REF_G12 = 0.8130165338516235         # e119 battery_table r_jittered g-12
G_R300_REF_PZ = 0.7753651738166809          # e119 calibration dial_default p_r
G_DELTA_REF = 0.05571012571454048           # e131 probe3 row0 delta_norm

# registered bars (numeric)
DIE_BAR, SURVIVE_BAR = 0.5, 0.8              # e131 collapse/survive convention
CE_FLAT_BAR = 0.35                           # ~25% of row-0 zero CE cost (1.4026)
PRESENCE_BAR, CONTENT_BAR = 0.20, 0.50       # probe-2 excess_pct thresholds
SCAFFOLD_CE_BAR = 1.4                        # row-0-level CE cost
R2_BAR, SLOPE_RATIO_BAR = 0.8, 3.0           # probe-5

REGISTERED_PREDICTION = {
    "queue_row_verbatim": (
        "On e131_consolidated + e048_repro + e119 R@150/R@300: (1) "
        "INSTALL-RESTORE SURGERY — swap consolidated wpe[0] <- install wpe[0] "
        "with t-interpolation dose curve (WRITTEN-KEY: fact dies with delta "
        "removed while CE flat; ROLE-ROUTED: survives t=1, dies only under "
        "direction-scramble); (2) PRESENCE-VS-CONTENT — scramble first 1-16 "
        "tokens of eval contexts vs mid-context matched control (PRESENCE-KEY "
        "<20% cost; CONTENT-KEY >=50% beyond control); (3) SCAFFOLD HARDENING "
        "— delete rows 2-6 individually + norm-matched random, CE_R + "
        "expression each (only-row-0-wrecks => sink uniqueness); (4) d_r0 at "
        "NOVEL geometry g-12 on R@150/R@300 (sink-keyed dies / content-keyed "
        "survives); (5) GATE-VS-SOURCE — graded interpolation of wpe[0] "
        "consolidated->install, expression vs r (SOURCE-GRADED R^2>=0.8; "
        "GATE-THRESHOLD >=80% retained to r* then collapse)."),
    "operationalizations": (
        "dies = expr <= 0.5 x expr(0); survives = >= 0.8 x; CE flat = "
        "|dCE| <= 0.35 (25% of row-0 zero CE cost 1.4026); direction-scramble "
        "= seeded 192-dim permutation of wpe[0] (norm-preserving), required to "
        "kill at t=0 for ROLE-ROUTED; probe-2 excess_pct = (front_cost - "
        "mid_cost)/base bar-read at k=16; probe-3 wrecks = min(zero,mean) "
        "expr <= 0.5 x base, scaffold bar = CE cost >= +1.4 with expression "
        "intact; probe-4 collapse = <= 0.5 x no-deletion g-12 level; probe-5 "
        "SOURCE-GRADED = linear R^2 >= 0.8 (slope<0) AND expr(1)/expr(0) < "
        "0.8 (actual loss — flat curves read NO-COLLAPSE), GATE-THRESHOLD = "
        "t*>=0.25 with >=80% retained, <=50% at t=1, post/pre slope ratio "
        ">= 3x; overall decided by probe 1, corroborated by 2-5 tallies."),
    "no_bar_shopping": "No bar shopping. Ambiguous => say AMBIGUOUS with "
                       "numbers.",
}

recipe_deviations: list[str] = [
    "Eval-only battery: every net is a LOADED, gated artifact (e131/"
    "e048/e119 checkpoints); nothing regenerated, nothing trained — the "
    "dispatch's eval-only constraint.",
    "Probe-4 R@300 gate reference (g0 = 0.7754) is e119's calibration "
    "dial_default p_r cell (mid-run eval of the default run) rather than a "
    "final battery-table cell — the battery table stored only the calibrated "
    "R@150 arm; fallback tolerance applies if the saved net drifted.",
    "Probe-2 scramble arm shuffles tokens WITHIN the window (permutation, no "
    "replacement): dose k=1 is an identity no-op by construction and serves "
    "as the in-arm sanity cell.",
    "SHAKEDOWN CLARIFICATION (before the full run; smoke exposed a branch "
    "bug): probe-5's SOURCE-GRADED originally read as bare 'linear R^2 >= "
    "0.8, slope < 0', which a FLAT curve trivially satisfies — internally "
    "contradicting probe 1 (nothing dies). Both probe-5 readings presuppose "
    "loss at t=1; the operationalization now requires normalized expr(1.0) "
    "< 0.8 for SOURCE-GRADED, with NO-COLLAPSE as the flat-curve reading. "
    "Verdict precedence: GATE-THRESHOLD > SOURCE-GRADED > NO-COLLAPSE.",
    "REPORT-ONLY RIDER (not a registered bar): probe-1 adds halfnorm/double-"
    "norm arms (wpe[0] x 0.5 / x 2.0, direction kept) to separate "
    "presence-only from norm-magnitude-keyed role; the rider never gates a "
    "verdict.",
]

trims: list[str] = []


# ------------------------------------------------------------------ instruments
# (e131 verbatim: load_cpu/evl_load/battery_cell/battery_pz/ce_fixed_cpu/
#  val_windows/deleted_wpe; provenance comments inline)

def evl_load(sd: dict) -> TinyGPT:
    m = TinyGPT(Cfg())
    m.load_state_dict(sd)
    m.eval()
    return m


def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e120 battery on CPU: p(Z) at the last position."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "std_pz": float(p.std()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0]}


@torch.no_grad()
def battery_pz(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> float:
    """e116's scalar battery (row census)."""
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, zid]) for k in range(pr.shape[0])]
    return float(np.mean(pzs))


@torch.no_grad()
def ce_fixed_cpu(net: TinyGPT, x, y, bs=64) -> float:
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    g = torch.Generator().manual_seed(seed)
    out_x, out_y = [], []
    tries = 0
    while len(out_x) < n and tries < 500 * n:
        i = int(torch.randint(len(val_ids) - block - 1, (1,), generator=g))
        txt = val_text[i: i + block + 1]
        tries += 1
        if "ZEPHYRA" in txt or "ZEPH" in txt:
            continue
        out_x.append(val_ids[i: i + block])
        out_y.append(val_ids[i + 1: i + 1 + block])
    return torch.stack(out_x), torch.stack(out_y)


def modified_wpe(sd: dict, row: int, value) -> tuple[dict, dict]:
    """e131 deleted_wpe's confinement gate, generalized to row-value surgery:
    at most `row`'s elements change, every other tensor bit-identical. The
    identity cell (value bit-equal to the original row, e.g. t=0) changes
    nothing and counts as confined by construction."""
    out = {k: v.clone() for k, v in sd.items()}
    out["wpe.weight"][row] = value
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    ok_rows = changed_rows in ([], [row])
    gate = {"row": row, "n_elements_changed": n,
            "changed_rows": changed_rows,
            "identity": bool(n == 0),
            "confined": bool(ok_rows),
            "others_bit_identical": bool(others),
            "pass": bool(ok_rows and others)}
    return out, gate


def deleted_wpe(sd: dict, rows: tuple[int, ...]) -> tuple[dict, dict]:
    """D2 subtractive row-zero with the e065/e113 confinement gate (verbatim)."""
    out = {k: v.clone() for k, v in sd.items()}
    for r in rows:
        out["wpe.weight"][r] = 0.0
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    gate = {"rows": list(rows), "n_elements_changed": n,
            "expected": len(rows) * sd["wpe.weight"].shape[1],
            "changed_rows": changed_rows,
            "confined": bool(changed_rows == sorted(rows)),
            "others_bit_identical": bool(others),
            "pass": bool(n == len(rows) * sd["wpe.weight"].shape[1]
                         and changed_rows == sorted(rows) and others)}
    return out, gate


def surgery_eval(sd: dict, label: str, bat: dict, zid: int, r_eval_xy,
                 net: TinyGPT | None = None) -> dict:
    """Load a surgical state, run the g0 batteries + CE_R, log, return."""
    if net is None:
        net = evl_load(sd)
    else:
        net.load_state_dict(sd)
    bz = battery_cell(net, bat[(0, "install60")], zid)
    hz = battery_cell(net, bat[(0, "held30")], zid)
    ce = ce_fixed_cpu(net, *r_eval_xy)
    log(f"  [{label:28s}] expr g0 {bz['mean_pz']:.4f} | held30 "
        f"{hz['mean_pz']:.4f} | CE_R {ce:.4f}")
    return {"install60_g0": bz, "held30_g0": hz, "ce_r": ce}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e141_smoke" if SMOKE else "e141")
    log(f"E141 SINK-KEY MECHANISM BATTERY (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e131 verbatim)
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

    # batteries: ctx = train_text[p-PRE-j : p] (e119 construction verbatim,
    # extended to j=-12 — the novel geometry cell)
    bat_ids = {}
    for j in GEOS:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[(0, "install60")]
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- nets + gates
    log("--- PHASE 0: load + gate the four artifacts ---")
    net_cons = load_cpu(CONS_CK)
    bz_cons = battery_cell(net_cons, ids130, zid)
    ce_cons = ce_fixed_cpu(net_cons, *r_eval_xy)
    G_CONS = {"battery_pz": bz_cons["mean_pz"], "ref_pz": G_CONS_REF_PZ,
              "ce_r": ce_cons, "ref_ce": G_CONS_REF_CE, "tol": G_BIT_TOL,
              "fallback_tol": G_FALLBACK_TOL,
              "pass": bool(abs(bz_cons["mean_pz"] - G_CONS_REF_PZ) < G_FALLBACK_TOL
                           and abs(ce_cons - G_CONS_REF_CE) < G_FALLBACK_TOL),
              "bit_reproducible": bool(
                  abs(bz_cons["mean_pz"] - G_CONS_REF_PZ) < G_BIT_TOL
                  and abs(ce_cons - G_CONS_REF_CE) < G_BIT_TOL)}
    log(f"G_CONS consolidated: p(Z) {bz_cons['mean_pz']:.10f} (ref "
        f"{G_CONS_REF_PZ:.10f}) CE_R {ce_cons:.6f} (ref {G_CONS_REF_CE:.6f}): "
        f"{'PASS' if G_CONS['pass'] else 'FAIL'}")
    if not G_CONS["pass"]:
        raise RuntimeError("e131 consolidated checkpoint failed its gate")

    net_inst = load_cpu(INST_CK)
    bz_inst = battery_cell(net_inst, ids130, zid)
    G_E048 = {"battery_pz": bz_inst["mean_pz"], "ref": G_E048_REF,
              "tol": G_BIT_TOL,
              "pass": bool(abs(bz_inst["mean_pz"] - G_E048_REF) < G_FALLBACK_TOL),
              "bit_reproducible": bool(abs(bz_inst["mean_pz"] - G_E048_REF)
                                       < G_BIT_TOL)}
    log(f"G_E048 install-phase: p(Z) {bz_inst['mean_pz']:.10f} (ref "
        f"{G_E048_REF:.10f}): {'PASS' if G_E048['pass'] else 'FAIL'}")
    if not G_E048["pass"]:
        raise RuntimeError("e048_repro checkpoint failed its gate")

    sd_cons = {k: v.clone() for k, v in net_cons.state_dict().items()}
    sd_inst = {k: v.clone() for k, v in net_inst.state_dict().items()}
    norms_all = sd_cons["wpe.weight"].norm(dim=1)   # row-norm census (probe 3)
    wpe0_cons = sd_cons["wpe.weight"][0].clone()
    wpe0_inst = sd_inst["wpe.weight"][0].clone()
    delta0 = wpe0_cons - wpe0_inst
    G_DELTA = {"delta_norm": float(delta0.norm()),
               "ref": G_DELTA_REF, "tol": G_BIT_TOL,
               "norm_cons": float(wpe0_cons.norm()),
               "norm_inst": float(wpe0_inst.norm()),
               "pass": bool(abs(float(delta0.norm()) - G_DELTA_REF)
                            < G_FALLBACK_TOL),
               "bit_reproducible": bool(abs(float(delta0.norm()) - G_DELTA_REF)
                                        < G_BIT_TOL)}
    log(f"G_DELTA wpe[0] cons-inst: |delta| {float(delta0.norm()):.10f} (ref "
        f"{G_DELTA_REF:.10f}; norms cons {float(wpe0_cons.norm()):.4f} / inst "
        f"{float(wpe0_inst.norm()):.4f}): "
        f"{'PASS' if G_DELTA['pass'] else 'FAIL'}")

    r_nets, r_gates = {}, {}
    for tag, path, ref_pz, ref_g12 in (
            ("R150", R150_CK, G_R150_REF_PZ, G_R150_REF_G12),
            ("R300", R300_CK, G_R300_REF_PZ, None)):
        if tag not in R_NETS:
            continue
        net_r = load_cpu(path)
        pz_g0 = battery_cell(net_r, bat_ids[(0, "install60")], zid)["mean_pz"]
        g = {"battery_pz_g0": pz_g0, "ref_g0": ref_pz, "tol": G_BIT_TOL,
             "fallback_tol": G_FALLBACK_TOL,
             "pass": bool(abs(pz_g0 - ref_pz) < G_FALLBACK_TOL),
             "bit_reproducible": bool(abs(pz_g0 - ref_pz) < G_BIT_TOL)}
        if ref_g12 is not None:
            pz_g12 = battery_cell(net_r, bat_ids[(-12, "install60")],
                                  zid)["mean_pz"]
            g["battery_pz_g12"] = pz_g12
            g["ref_g12"] = ref_g12
            g["pass"] = bool(g["pass"] and abs(pz_g12 - ref_g12)
                             < G_FALLBACK_TOL)
            g["bit_reproducible"] = bool(g["bit_reproducible"]
                                         and abs(pz_g12 - ref_g12) < G_BIT_TOL)
        r_nets[tag] = net_r
        r_gates[tag] = g
        log(f"G_{tag}: g0 {pz_g0:.10f} (ref {ref_pz:.10f})"
            + (f" g-12 {g['battery_pz_g12']:.10f} (ref {ref_g12:.10f})"
               if ref_g12 is not None else "")
            + f": {'PASS' if g['pass'] else 'FAIL'}")
        if not g["pass"]:
            raise RuntimeError(f"{tag} checkpoint failed its gate")

    base_expr = bz_cons["mean_pz"]           # consolidated, no surgery
    base_ce = ce_cons

    # =====================================================================
    # PROBE 1 — INSTALL-RESTORE SURGERY (t-interpolation) + direction-scramble
    # =====================================================================
    log("--- PROBE 1: install-restore surgery on wpe[0] (t-curve + scramble) ---")
    perm = torch.randperm(wpe0_cons.shape[0],
                          generator=torch.Generator().manual_seed(PERM_DIM_SEED))
    surgery_log = {}
    curve = []
    eval_net = evl_load(sd_cons)                   # reused eval twin (no RNG use)
    for t in T_GRID:
        sd_t, gate_t = modified_wpe(sd_cons, 0, t * wpe0_inst
                                    + (1.0 - t) * wpe0_cons)
        if not gate_t["pass"]:
            raise RuntimeError(f"t={t} surgery gate FAILED: {gate_t}")
        res = surgery_eval(sd_t, f"restore t={t:.2f}", bat_ids, zid,
                           r_eval_xy, eval_net)
        curve.append({"t": t, "gate": gate_t,
                      "expr": res["install60_g0"]["mean_pz"],
                      "held30": res["held30_g0"]["mean_pz"],
                      "ce_r": res["ce_r"], "battery": res["install60_g0"]})
        surgery_log[f"restore_t{t:g}"] = res

    expr0 = curve[0]["expr"]
    expr1 = curve[-1]["expr"]
    ce1 = curve[-1]["ce_r"]

    # direction-scramble (norm-preserving fixed permutation) at t=0 and t=1
    scr = {}
    for tag, row_val in (("perm@t0", wpe0_cons[perm]),
                         ("perm@t1", wpe0_inst[perm])):
        sd_s, gate_s = modified_wpe(sd_cons, 0, row_val)
        if not gate_s["pass"]:
            raise RuntimeError(f"{tag} surgery gate FAILED: {gate_s}")
        res = surgery_eval(sd_s, tag, bat_ids, zid, r_eval_xy, eval_net)
        scr[tag] = {"gate": gate_s,
                    "row_norm": float(row_val.norm()),
                    "norm_preserved": bool(abs(float(row_val.norm())
                                               - float(wpe0_cons.norm()))
                                           < 1e-5),
                    "expr": res["install60_g0"]["mean_pz"],
                    "held30": res["held30_g0"]["mean_pz"],
                    "ce_r": res["ce_r"], "battery": res["install60_g0"]}
        surgery_log[tag] = res

    # e116 reference arms on this net (zero / mean-replace row 0)
    mean_row = sd_cons["wpe.weight"].mean(0)
    refs16 = {}
    for tag, row_val in (("zero@t0", torch.zeros_like(wpe0_cons)),
                         ("mean@t0", mean_row)):
        sd_s, gate_s = modified_wpe(sd_cons, 0, row_val)
        res = surgery_eval(sd_s, tag, bat_ids, zid, r_eval_xy, eval_net)
        refs16[tag] = {"gate": gate_s, "expr": res["install60_g0"]["mean_pz"],
                       "ce_r": res["ce_r"],
                       "row_norm": float(row_val.norm())}
        surgery_log[tag] = res
    refs16["_context"] = {
        "row0_norm": float(wpe0_cons.norm()),
        "mean_row_norm": float(mean_row.norm()),
        "row_norm_rank_of_row0_descending": int((norms_all > norms_all[0]).sum()) + 1,
        "note": "mean-replace installs a norm-0.066 row (~8.6% of row 0's "
                "0.764) — it is near-REMOVAL, not a direction control "
                "(T077's point); zero/mean are removal-class arms, the perm "
                "and norm-dose rider are the presence/direction controls",
    }

    # REPORT-ONLY RIDER (not a registered bar): norm dose — direction kept,
    # norm halved/doubled — separates presence-only from norm-magnitude role
    rider = {}
    for tag, row_val in (("halfnorm@t0", 0.5 * wpe0_cons),
                         ("doublenorm@t0", 2.0 * wpe0_cons)):
        sd_s, gate_s = modified_wpe(sd_cons, 0, row_val)
        res = surgery_eval(sd_s, tag, bat_ids, zid, r_eval_xy, eval_net)
        rider[tag] = {"gate": gate_s,
                      "row_norm": float(row_val.norm()),
                      "expr": res["install60_g0"]["mean_pz"],
                      "ce_r": res["ce_r"]}
        surgery_log[tag] = res

    dies_t1 = expr1 <= DIE_BAR * expr0
    survives_t1 = expr1 >= SURVIVE_BAR * expr0
    ce_flat_t1 = abs(ce1 - base_ce) <= CE_FLAT_BAR
    scr_kills = scr["perm@t0"]["expr"] <= DIE_BAR * expr0
    scr_kills_t1 = scr["perm@t1"]["expr"] <= DIE_BAR * expr0
    written_key = bool(dies_t1 and ce_flat_t1)
    role_routed = bool(survives_t1 and scr_kills)
    presence_only = bool(survives_t1 and not scr_kills and not scr_kills_t1)
    if written_key:
        p1_verdict = "WRITTEN-KEY"
    elif role_routed:
        p1_verdict = "ROLE-ROUTED"
    elif presence_only:
        p1_verdict = "ROLE-ROUTED (presence-only: survives t=1 AND scramble)"
    elif dies_t1 and not ce_flat_t1:
        p1_verdict = "AMBIGUOUS (fact dies at t=1 but CE NOT flat — scaffold damage, not clean key removal)"
    else:
        p1_verdict = "AMBIGUOUS"
    probe1 = {
        "t_grid": list(T_GRID),
        "curve": curve,
        "direction_scramble": scr,
        "e116_reference_arms": refs16,
        "bars": {"die": DIE_BAR, "survive": SURVIVE_BAR, "ce_flat": CE_FLAT_BAR},
        "numbers": {"expr0": expr0, "expr1": expr1, "ce0": base_ce, "ce1": ce1,
                    "d_ce_t1": ce1 - base_ce,
                    "dies_t1": dies_t1, "survives_t1": survives_t1,
                    "ce_flat_t1": ce_flat_t1,
                    "perm_t0_expr": scr["perm@t0"]["expr"],
                    "perm_t1_expr": scr["perm@t1"]["expr"],
                    "zero_expr": refs16["zero@t0"]["expr"],
                    "mean_expr": refs16["mean@t0"]["expr"]},
        "written_key": written_key, "role_routed": role_routed,
        "presence_only": presence_only,
        "norm_dose_rider_report_only": rider,
        "verdict": p1_verdict,
    }
    log(f"PROBE1: expr(0) {expr0:.4f} -> expr(1) {expr1:.4f} "
        f"(x{expr1 / max(expr0, 1e-12):.3f}), dCE {ce1 - base_ce:+.4f} | "
        f"perm@t0 {scr['perm@t0']['expr']:.4f} perm@t1 "
        f"{scr['perm@t1']['expr']:.4f} | zero {refs16['zero@t0']['expr']:.4f} "
        f"mean {refs16['mean@t0']['expr']:.4f} | halfnorm "
        f"{rider['halfnorm@t0']['expr']:.4f} doublenorm "
        f"{rider['doublenorm@t0']['expr']:.4f} -> {p1_verdict}")

    # =====================================================================
    # PROBE 2 — PRESENCE-VS-CONTENT (front vs mid token scramble)
    # =====================================================================
    log("--- PROBE 2: presence-vs-content scramble dose-response ---")
    eval_net.load_state_dict(sd_cons)         # reload the unsurgerized baseline
    g_scr = torch.Generator().manual_seed(SCRAMBLE_SEED)
    n_ctx = ids130.shape[0]
    doses_out = []
    base2 = battery_pz(eval_net, ids130, zid)
    for k in DOSES:
        perms_k = [torch.randperm(k, generator=g_scr) for _ in range(n_ctx)]
        front = ids130.clone()
        mid = ids130.clone()
        for i in range(n_ctx):
            pm = perms_k[i]
            front[i, :k] = ids130[i, :k][pm]
            mid[i, 64:64 + k] = ids130[i, 64:64 + k][pm]
        pz_f = battery_pz(eval_net, front, zid)
        pz_m = battery_pz(eval_net, mid, zid)
        f_cost, m_cost = base2 - pz_f, base2 - pz_m
        excess = f_cost - m_cost
        doses_out.append({"k": k, "front_pz": pz_f, "mid_pz": pz_m,
                          "front_cost": f_cost, "mid_cost": m_cost,
                          "excess": excess,
                          "excess_pct": excess / max(base2, 1e-12)})
        log(f"  dose k={k:2d}: front {pz_f:.4f} mid {pz_m:.4f} | excess "
            f"{excess:+.4f} ({100 * excess / max(base2, 1e-12):+.1f}%)")
    head = doses_out[-1]                      # bar read at largest dose (k=16)
    max_exc = max(doses_out, key=lambda d: d["excess_pct"])
    if head["excess_pct"] < PRESENCE_BAR:
        p2_verdict = "PRESENCE-KEY"
    elif head["excess_pct"] >= CONTENT_BAR:
        p2_verdict = "CONTENT-KEY"
    else:
        p2_verdict = "AMBIGUOUS"
    probe2 = {
        "net": "consolidated (e131_consolidated_e113)",
        "battery": "install-60 g0 (130-token contexts)",
        "mid_window": "positions [64, 64+k) — matched count, SAME seeded "
                      "permutation as the front window",
        "base_pz": base2,
        "doses": doses_out,
        "bars": {"presence_lt": PRESENCE_BAR, "content_ge": CONTENT_BAR,
                 "read_at": f"k={head['k']} (largest dose)"},
        "max_excess_dose": {"k": max_exc["k"],
                            "excess_pct": max_exc["excess_pct"]},
        "verdict": p2_verdict,
        "note": "dose k=1 is an identity no-op by construction (in-arm sanity)",
    }
    log(f"PROBE2: k=16 excess {100 * head['excess_pct']:+.1f}% of base -> "
        f"{p2_verdict}")

    # =====================================================================
    # PROBE 3 — SCAFFOLD HARDENING (rows 2-6 + norm-matched random)
    # =====================================================================
    log("--- PROBE 3: scaffold hardening (rows 2-6 + norm-matched random) ---")
    norms = norms_all
    excluded = set(range(0, 7)) | set(range(121, 138))
    g_nm = torch.Generator().manual_seed(NORMMATCH_SEED)
    # all non-excluded rows in seeded permuted order; argmin |norm - norm0|
    # (randomness only orders near-ties — deterministic given the seed)
    cand_order = [r for r in torch.randperm(int(norms.shape[0]),
                                            generator=g_nm).tolist()
                  if r not in excluded]
    r_star = min(cand_order, key=lambda r: abs(float(norms[r])
                                               - float(norms[0])))
    log(f"norm-matched random row: {r_star} (norm {float(norms[r_star]):.4f} "
        f"vs row0 {float(norms[0]):.4f}; candidates {len(cand_order)})")
    probe3_rows = {}
    row0_arm = {"zero": refs16["zero@t0"], "mean": refs16["mean@t0"]}
    for r in list(SCAFFOLD_ROWS) + [r_star]:
        arms_r = {}
        for a_name, val in (("zero", torch.zeros_like(wpe0_cons)),
                            ("mean", mean_row)):
            sd_s, gate_s = modified_wpe(sd_cons, r, val)
            if not gate_s["pass"]:
                raise RuntimeError(f"row {r} {a_name} gate FAILED")
            res = surgery_eval(sd_s, f"row{r}_{a_name}", bat_ids, zid,
                               r_eval_xy, eval_net)
            arms_r[a_name] = {"gate": gate_s,
                              "expr": res["install60_g0"]["mean_pz"],
                              "ce_r": res["ce_r"],
                              "battery": res["install60_g0"]}
        worst_expr = min(arms_r["zero"]["expr"], arms_r["mean"]["expr"])
        max_ce = max(arms_r["zero"]["ce_r"], arms_r["mean"]["ce_r"])
        probe3_rows[str(r)] = {
            "arms": arms_r, "min_expr": worst_expr, "max_ce": max_ce,
            "ce_cost": max_ce - base_ce,
            "wrecks": bool(worst_expr <= DIE_BAR * base_expr),
            "expression_intact": bool(worst_expr >= SURVIVE_BAR * base_expr),
            "row_norm": float(norms[r]),
        }
        log(f"  row {r:3d} (norm {float(norms[r]):.3f}): min expr "
            f"{worst_expr:.4f} | max CE cost {max_ce - base_ce:+.4f} | "
            f"wrecks {probe3_rows[str(r)]['wrecks']}")
    row0_zero_cecost = refs16["zero@t0"]["ce_r"] - base_ce
    others = [probe3_rows[str(r)] for r in list(SCAFFOLD_ROWS) + [r_star]]
    only_row0_wrecks = bool(row0_arm["zero"]["expr"] <= DIE_BAR * base_expr
                            and not any(o["wrecks"] for o in others))
    any_other_big_ce = [str(r) for r in list(SCAFFOLD_ROWS) + [r_star]
                        if probe3_rows[str(r)]["ce_cost"] >= SCAFFOLD_CE_BAR]
    scaffold_alt = bool(len(any_other_big_ce) > 0
                        and all(probe3_rows[r]["expression_intact"]
                                for r in any_other_big_ce))
    probe3 = {
        "rows": probe3_rows,
        "row0_reference": {"zero": refs16["zero@t0"], "mean": refs16["mean@t0"],
                           "zero_ce_cost": row0_zero_cecost},
        "r_star": {"row": r_star, "norm": float(norms[r_star]),
                   "row0_norm": float(norms[0]),
                   "seed": NORMMATCH_SEED,
                   "exclusions": "rows 0-6 and band 121-137"},
        "bars": {"wrecks": DIE_BAR, "scaffold_ce": SCAFFOLD_CE_BAR},
        "only_row0_wrecks": only_row0_wrecks,
        "other_rows_ce_ge_1p4": any_other_big_ce,
        "scaffold_alternative_strengthened": scaffold_alt,
        "verdict": ("SINK-UNIQUENESS (only row-0 wrecks; all controls cheap)"
                    if only_row0_wrecks and not scaffold_alt else
                    f"SCAFFOLD-ALTERNATIVE STRENGTHENED (rows "
                    f"{any_other_big_ce} cost CE>=+1.4 with expression intact)"
                    if scaffold_alt else "AMBIGUOUS"),
    }
    log(f"PROBE3: only-row-0-wrecks {only_row0_wrecks} | other rows CE>=+1.4: "
        f"{any_other_big_ce} -> {probe3['verdict']}")

    # =====================================================================
    # PROBE 4 — d_r0 AT NOVEL GEOMETRY g-12 (R@150 / R@300; W011's missing cell)
    # =====================================================================
    log("--- PROBE 4: d_r0 at novel geometry g-12 on e119 R arms ---")
    probe4 = {}
    for tag in R_NETS:
        net_r = r_nets[tag]
        sd_r = {k: v.clone() for k, v in net_r.state_dict().items()}
        cells = {}
        for arm in ("none", "d_r0"):
            if arm == "none":
                sd_a = sd_r
                gate = {"rows": [], "pass": True, "note": "no deletion"}
            else:
                sd_a, gate = deleted_wpe(sd_r, (0,))
                if not gate["pass"]:
                    raise RuntimeError(f"{tag} d_r0 gate FAILED: {gate}")
            net_r.load_state_dict(sd_a)
            e12i = battery_cell(net_r, bat_ids[(-12, "install60")], zid)
            e12h = battery_cell(net_r, bat_ids[(-12, "held30")], zid)
            e0i = battery_cell(net_r, bat_ids[(0, "install60")], zid)
            ce = ce_fixed_cpu(net_r, *r_eval_xy)
            cells[arm] = {"gate": gate, "g-12_install60": e12i,
                          "g-12_held30": e12h, "g0_install60": e0i,
                          "ce_r": ce}
            log(f"  [{tag} {arm:4s}] g-12 expr {e12i['mean_pz']:.4f} "
                f"(held {e12h['mean_pz']:.4f}) | g0 {e0i['mean_pz']:.4f} | "
                f"CE_R {ce:.4f}")
        n12 = cells["none"]["g-12_install60"]["mean_pz"]
        d12 = cells["d_r0"]["g-12_install60"]["mean_pz"]
        cells["bars"] = {"collapse_le": DIE_BAR * n12,
                         "survive_ge": SURVIVE_BAR * n12}
        cells["collapses_at_g12"] = bool(d12 <= DIE_BAR * n12)
        cells["survives_at_g12"] = bool(d12 >= SURVIVE_BAR * n12)
        cells["verdict"] = ("SINK-KEYED COLLAPSE" if cells["collapses_at_g12"]
                            else "CONTENT-KEYED SURVIVAL"
                            if cells["survives_at_g12"] else "AMBIGUOUS")
        probe4[tag] = cells
        log(f"PROBE4 [{tag}]: g-12 {n12:.4f} -> {d12:.4f} "
            f"(x{d12 / max(n12, 1e-12):.3f}) -> {cells['verdict']}")

    # =====================================================================
    # PROBE 5 — GATE-VS-SOURCE (read probe-1's curve both ways)
    # =====================================================================
    log("--- PROBE 5: gate-vs-source reading of the t-curve ---")
    ts = [c["t"] for c in curve]
    es = [c["expr"] / max(expr0, 1e-12) for c in curve]
    slope, intercept = np.polyfit(ts, es, 1)
    yhat = np.polyval([slope, intercept], ts)
    ss_res = float(np.sum((np.array(es) - yhat) ** 2))
    ss_tot = float(np.sum((np.array(es) - np.mean(es)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    # gate threshold: t* = largest grid t retaining >= 80% expression
    retained = [t for t, e in zip(ts, es) if e >= SURVIVE_BAR]
    t_star = max(retained) if retained else 0.0
    pre = [(ts[i], ts[i + 1], es[i], es[i + 1]) for i in range(len(ts) - 1)
           if ts[i + 1] <= t_star + 1e-9]
    post = [(ts[i], ts[i + 1], es[i], es[i + 1]) for i in range(len(ts) - 1)
            if ts[i] >= t_star - 1e-9]
    sl = lambda p: (p[3] - p[2]) / (p[1] - p[0]) if p[1] > p[0] else 0.0
    slope_pre = float(np.mean([sl(p) for p in pre])) if pre else 0.0
    slope_post = float(np.mean([sl(p) for p in post])) if post else 0.0
    denom = max(abs(slope_pre), 1e-9)
    ratio = (abs(slope_post) / denom) if slope_post < 0 else 0.0
    no_collapse = bool(es[-1] >= SURVIVE_BAR)
    # both readings presuppose actual loss at t=1 (below the survive bar);
    # a flat / never-dropping curve reads NO-COLLAPSE (role-consistent)
    source_graded = bool(r2 >= R2_BAR and slope < 0 and not no_collapse)
    gate_threshold = bool(t_star >= 0.25 and es[-1] <= DIE_BAR
                          and ratio >= SLOPE_RATIO_BAR)
    probe5 = {
        "t": ts, "expr_normalized": [float(e) for e in es],
        "linear_fit": {"slope": float(slope), "intercept": float(intercept),
                       "r2": float(r2)},
        "gate_fit": {"t_star": float(t_star), "slope_pre": slope_pre,
                     "slope_post": slope_post,
                     "slope_ratio_post_over_pre": float(ratio),
                     "bars": {"retain": SURVIVE_BAR, "collapse": DIE_BAR,
                              "ratio_ge": SLOPE_RATIO_BAR}},
        "bars": {"r2_ge": R2_BAR, "slope_ratio_ge": SLOPE_RATIO_BAR},
        "source_graded": source_graded, "gate_threshold": gate_threshold,
        "no_collapse": no_collapse,
        "verdict": ("GATE-THRESHOLD" if gate_threshold else
                    "SOURCE-GRADED" if source_graded else
                    "NO-COLLAPSE (role-consistent: nothing lost across the "
                    "whole interpolation)" if no_collapse else "AMBIGUOUS"),
    }
    log(f"PROBE5: linear R^2 {r2:.3f} (slope {slope:+.3f}) | t* {t_star:.2f} "
        f"pre {slope_pre:+.3f} post {slope_post:+.3f} ratio {ratio:.1f} -> "
        f"{probe5['verdict']}")

    # =====================================================================
    # overall adjudication (registered: probe 1 decides, 2-5 corroborate)
    # =====================================================================
    corroboration = {
        "probe2_CONTENT_KEY_supports_WRITTEN_KEY": probe2["verdict"] == "CONTENT-KEY",
        "probe2_PRESENCE_KEY_supports_ROLE_ROUTED": probe2["verdict"] == "PRESENCE-KEY",
        "probe3_sink_uniqueness_supports_ROLE_ROUTED": probe3["only_row0_wrecks"],
        "probe3_scaffold_alternative": probe3["scaffold_alternative_strengthened"],
        "probe4_g12_collapse_supports_sink_keying": any(
            probe4[t]["collapses_at_g12"] for t in probe4),
        "probe4_g12_survival_supports_content_keyed_readout": any(
            probe4[t]["survives_at_g12"] for t in probe4),
        "probe5_GATE_THRESHOLD_supports_ROLE_ROUTED": probe5["gate_threshold"],
        "probe5_SOURCE_GRADED_supports_WRITTEN_KEY": probe5["source_graded"],
    }
    if written_key:
        overall = "WRITTEN-KEY"
    elif role_routed or presence_only:
        overall = "ROLE-ROUTED" + (" (presence-only)" if presence_only else "")
    else:
        overall = "AMBIGUOUS"
    role_votes = sum(bool(v) for k, v in corroboration.items()
                     if "supports_ROLE_ROUTED" in k or "supports_sink_keying" in k)
    written_votes = sum(bool(v) for k, v in corroboration.items()
                        if "supports_WRITTEN_KEY" in k)
    g12_bits = " ".join(
        f"{t} x{probe4[t]['d_r0']['g-12_install60']['mean_pz'] / max(probe4[t]['none']['g-12_install60']['mean_pz'], 1e-12):.2f}"
        for t in probe4)
    headline = (f"t=1 expr {expr1:.3f} vs base {expr0:.3f} "
                f"(x{expr1 / max(expr0, 1e-12):.2f}), dCE {ce1 - base_ce:+.3f}; "
                f"perm@t0 {scr['perm@t0']['expr']:.3f}; scramble excess "
                f"{100 * head['excess_pct']:+.0f}%; scaffold "
                f"{'only-row-0' if probe3['only_row0_wrecks'] else 'not-only-row-0'}; "
                f"g-12 {g12_bits}")
    adjudication = {
        "probe1_primary": p1_verdict,
        "probe2": p2_verdict,
        "probe3": probe3["verdict"],
        "probe4": {t: probe4[t]["verdict"] for t in probe4},
        "probe5": probe5["verdict"],
        "corroboration": corroboration,
        "role_routed_votes": role_votes,
        "written_key_votes": written_votes,
        "overall": overall,
        "headline": headline,
    }
    log("=" * 78)
    log(f"E141 VERDICT: {overall}")
    for k, v in corroboration.items():
        log(f"  {k}: {v}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e141_sinkkey_mechanism",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("R44 critic attacks 1/4/7 + ideator e141 merged; "
                         "T077 SECOND AMENDMENT discriminator. Docstring + "
                         "bars written before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is consolidated row 0's necessity a WRITTEN-KEY "
                     "(consolidation's delta in wpe[0] carries the fact) or "
                     "sink-ROLE (the pivot readouts route through)?"),
        "nets": {
            "consolidated": f"runs/checkpoints/{CONS_CK.name} (loaded, gated)",
            "install_phase": f"runs/checkpoints/{INST_CK.name} (loaded, gated)",
            "R150": f"runs/checkpoints/{R150_CK.name} (loaded, gated)",
            "R300": f"runs/checkpoints/{R300_CK.name} (loaded, gated)",
            "eval_only": True,
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_CONS": G_CONS, "G_E048": G_E048,
                  "G_DELTA": G_DELTA, "R_arms": r_gates},
        "probe1_install_restore": probe1,
        "probe1_surgery_log": {k: {"install60_g0": v["install60_g0"],
                                   "held30_g0": v["held30_g0"],
                                   "ce_r": v["ce_r"]}
                               for k, v in surgery_log.items()},
        "probe2_presence_vs_content": probe2,
        "probe3_scaffold_hardening": probe3,
        "probe4_dr0_novel_geometry": probe4,
        "probe5_gate_vs_source": probe5,
        "adjudication": adjudication,
        "honesty_reflex": {
            "interpolation_path_dependence": "the t-curve probes the LINEAR "
            "path consolidated->install only; a written key misaligned with "
            "that path could die at t=1 without being the fact's store "
            "(path-dependence cuts toward ROLE-ROUTED only if t=1 survives "
            "AND scramble kills; if t=1 dies, check CE before calling it "
            "WRITTEN-KEY)",
            "scramble_leakage": "probe-2 front scrambles tokens 0..k-1 — "
            "damage at k>1 conflates row-0 content with rows 1..k-1 content; "
            "the matched mid-window control absorbs generic context damage, "
            "and dose-1 is an identity no-op by construction",
            "single_net_caveats": "probe-1/2/3 run on ONE consolidated net "
            "(e113 recipe, one seed); probe-4 adds two more nets (e119 "
            "R@150/R@300) but at a different road/geometry — verdicts are "
            "line-specific until replicated",
            "direction_scramble_is_not_identity": "the perm arm preserves "
            "norm but destroys the direction; a role account that needs only "
            "'a large-norm row present' predicts survival, one that needs "
            "'row 0's particular direction as a pivot' predicts death — "
            "the perm arm cannot separate those two role subtypes from a "
            "written key by itself (t=1 does that); the REPORT-ONLY "
            "halfnorm/doublenorm rider separates presence-only from "
            "norm-magnitude-keyed role",
        },
        "trims": trims,
        "recipe_deviations": recipe_deviations,
        "ckpt_inventory": {
            "saved": {},
            "external_used": [f"runs/checkpoints/{p.name}"
                              for p in (CONS_CK, INST_CK)]
                             + [f"runs/checkpoints/e119_r_jittered_{t.lower()}.pt"
                                for t in R_NETS],
            "note": "eval-only: no checkpoints written or regenerated",
        },
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072,
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "sinkkey_mechanism.png", probe1, probe2, probe3, probe4,
         probe5, adjudication)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'sinkkey_mechanism.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def plot(path, probe1, probe2, probe3, probe4, probe5, adjudication):
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 13.0))

    # (0,0) probe 1: t-curve (expression + CE) + scramble reference levels
    ax = axes[0, 0]
    ts = [c["t"] for c in probe1["curve"]]
    ex = [c["expr"] for c in probe1["curve"]]
    ce = [c["ce_r"] for c in probe1["curve"]]
    ax.plot(ts, ex, "o-", color="crimson", lw=1.8, ms=6,
            label="expression p(Z) install-60 g0")
    ax.axhline(probe1["numbers"]["expr0"] * 0.8, ls=":", color="seagreen",
               lw=1.2, label="survive bar (0.8x)")
    ax.axhline(probe1["numbers"]["expr0"] * 0.5, ls="--", color="gray",
               lw=1.2, label="die bar (0.5x)")
    scr = probe1["direction_scramble"]
    for tag, col, mk in (("perm@t0", "purple", "x"), ("perm@t1", "purple", "+")):
        ax.axhline(scr[tag]["expr"], ls="-.", lw=0.9, color=col, alpha=0.6,
                   label=f"{tag} {scr[tag]['expr']:.3f}")
    for tag, col in (("zero@t0", "black"), ("mean@t0", "dimgray")):
        v = probe1["e116_reference_arms"][tag]["expr"]
        ax.axhline(v, ls=":", lw=0.9, color=col, alpha=0.6,
                   label=f"{tag} {v:.3f}")
    for tag in ("halfnorm@t0", "doublenorm@t0"):
        v = probe1["norm_dose_rider_report_only"][tag]["expr"]
        ax.axhline(v, ls="-.", lw=0.9, color="goldenrod", alpha=0.7,
                   label=f"rider {tag} {v:.3f}")
    ax2 = ax.twinx()
    ax2.plot(ts, ce, "s--", color="steelblue", lw=1.2, ms=5, alpha=0.8,
             label="CE_R (right)")
    ax2.set_ylabel("CE_R", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xlabel("t  (wpe[0] = t*install + (1-t)*consolidated)")
    ax.set_ylabel("expression p(Z)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"PROBE 1: install-restore surgery -> {probe1['verdict']}",
                 fontsize=10)
    ax.legend(fontsize=6.2, loc="center right")

    # (0,1) probe 2: scramble dose-response
    ax = axes[0, 1]
    ks = [d["k"] for d in probe2["doses"]]
    ax.plot(ks, [d["front_pz"] for d in probe2["doses"]], "o-",
            color="crimson", label="front window [0,k)")
    ax.plot(ks, [d["mid_pz"] for d in probe2["doses"]], "s-",
            color="steelblue", label="mid control [64,64+k)")
    ax.axhline(probe2["base_pz"], ls=":", color="k", lw=1.0,
               label=f"base {probe2['base_pz']:.3f}")
    ax.fill_between(ks, [d["front_pz"] for d in probe2["doses"]],
                    [d["mid_pz"] for d in probe2["doses"]], color="crimson",
                    alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k tokens scrambled (matched permutation)")
    ax.set_ylabel("expression p(Z)")
    head = probe2["doses"][-1]
    ax.annotate(f"k={head['k']} excess {100 * head['excess_pct']:+.1f}%\n"
                f"-> {probe2['verdict']}",
                xy=(head["k"], head["front_pz"]), xytext=(-90, -25),
                textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_title("PROBE 2: presence-vs-content scramble (consolidated net)",
                 fontsize=10)
    ax.legend(fontsize=7)

    # (1,0) probe 3: scaffold hardening controls grid
    ax = axes[1, 0]
    r_star_k = str(probe3["r_star"]["row"])
    rows_show = ["0", r_star_k] + [str(r) for r in sorted(SCAFFOLD_ROWS)]
    labels = [("row0" if k == "0" else f"r{k}" + ("*" if k == r_star_k else ""))
              for k in rows_show]

    def _arm(k, a):
        if k == "0":
            return probe3["row0_reference"][a]
        return probe3["rows"][k]["arms"][a]

    zero_v = [_arm(k, "zero")["expr"] for k in rows_show]
    mean_v = [_arm(k, "mean")["expr"] for k in rows_show]
    ce_cost = [max(_arm(k, "zero")["ce_r"], _arm(k, "mean")["ce_r"])
               - probe1["numbers"]["ce0"] for k in rows_show]
    xs = np.arange(len(rows_show))
    ax.bar(xs - 0.19, zero_v, 0.36, color="crimson", edgecolor="k", lw=0.4,
           label="zero arm expr")
    ax.bar(xs + 0.19, mean_v, 0.36, color="pink", edgecolor="k", lw=0.4,
           label="mean arm expr")
    base_expr = probe1["numbers"]["expr0"]
    ax.axhline(base_expr, ls=":", color="k", lw=1.0,
               label=f"base {base_expr:.3f}")
    ax.axhline(base_expr * 0.5, ls="--", color="gray", lw=1.0,
               label="wrecks bar (0.5x)")
    ax3 = ax.twinx()
    ax3.plot(xs, ce_cost, "D", color="steelblue", ms=6,
             label="max CE cost (right)")
    ax3.axhline(probe3["bars"]["scaffold_ce"], ls="-.", color="steelblue",
                lw=1.0, label=f"scaffold bar +{probe3['bars']['scaffold_ce']}")
    ax3.set_ylabel("CE cost vs none", color="steelblue")
    ax3.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("expression p(Z)")
    ax.set_title(f"PROBE 3: scaffold hardening -> {probe3['verdict']}",
                 fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.2, loc="lower left")

    # (1,1) probe 4: d_r0 at g-12
    ax = axes[1, 1]
    tags = list(probe4)
    w = 0.36
    for i, t in enumerate(tags):
        n12 = probe4[t]["none"]["g-12_install60"]["mean_pz"]
        d12 = probe4[t]["d_r0"]["g-12_install60"]["mean_pz"]
        ax.bar(i - w / 2, n12, w, color="steelblue", edgecolor="k", lw=0.5,
               label="none" if i == 0 else None)
        ax.bar(i + w / 2, d12, w, color="crimson", edgecolor="k", lw=0.5,
               label="d_r0" if i == 0 else None)
        ax.plot([i - 0.45, i + 0.45], [0.5 * n12] * 2, ls="--", color="gray",
                lw=1.1)
        ax.text(i + w / 2, d12 + 0.02,
                f"x{d12 / max(n12, 1e-12):.3f}\n{probe4[t]['verdict']}",
                ha="center", fontsize=7.5)
    ax.set_xticks(np.arange(len(tags)))
    ax.set_xticklabels([f"{t}\n(g-12 novel geom.)" for t in tags], fontsize=9)
    ax.set_ylabel("expression p(Z) at g-12 (install-60)")
    ax.set_ylim(0, 1.05)
    ax.set_title("PROBE 4: d_r0 at the novel geometry (e119 R arms) — "
                 "W011's missing cell", fontsize=10)
    ax.legend(fontsize=8)

    # (2,0) probe 5: threshold fit
    ax = axes[2, 0]
    ts5 = probe5["t"]
    es5 = probe5["expr_normalized"]
    ax.plot(ts5, es5, "o", ms=8, color="crimson", label="normalized expression")
    fit = probe5["linear_fit"]
    xs5 = np.linspace(0, 1, 50)
    ax.plot(xs5, fit["slope"] * xs5 + fit["intercept"], "--", color="gray",
            lw=1.2, label=f"linear fit R^2={fit['r2']:.3f}")
    gf = probe5["gate_fit"]
    ax.axvline(gf["t_star"], ls="-.", color="purple", lw=1.2,
               label=f"t*={gf['t_star']:.2f} (80% retained)")
    ax.axhline(0.8, ls=":", color="seagreen", lw=1.1)
    ax.axhline(0.5, ls=":", color="gray", lw=1.1)
    ax.set_xlabel("t (install fraction in wpe[0])")
    ax.set_ylabel("expression / expression(0)")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(f"PROBE 5: gate-vs-source — slope pre {gf['slope_pre']:+.2f} "
                 f"post {gf['slope_post']:+.2f} (ratio "
                 f"{gf['slope_ratio_post_over_pre']:.1f}x) -> "
                 f"{probe5['verdict']}", fontsize=9.5)
    ax.legend(fontsize=7.5)

    # (2,1) verdict text
    ax = axes[2, 1]
    ax.axis("off")
    txt = (f"OVERALL: {adjudication['overall']}\n\n"
           f"probe 1 (primary): {adjudication['probe1_primary']}\n"
           f"probe 2: {adjudication['probe2']}\n"
           f"probe 3: {adjudication['probe3']}\n"
           f"probe 4: " + " | ".join(f"{t}: {v}" for t, v in
                                     adjudication["probe4"].items())
           + f"\nprobe 5: {adjudication['probe5']}\n\n"
           f"corroboration votes: role-routed {adjudication['role_routed_votes']}"
           f" vs written-key {adjudication['written_key_votes']}\n\n"
           + "\n".join(f"  {k}: {v}" for k, v in
                       adjudication["corroboration"].items()))
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=7.6, va="top",
            family="monospace",
            bbox=dict(facecolor="lightyellow", alpha=0.94, edgecolor="gray"))

    fig.suptitle(f"E141 — sink-key mechanism battery: ROLE-ROUTED vs "
                 f"WRITTEN-KEY -> {adjudication['overall']}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
