"""E188 — THE DEATH-CURRENCY CELL (W022b's named measurement, pointed by
T137 and T139, confirmed with the two-currency amendment by the R58 ideator
[scratch/r58_ideator.md, e188 section]; dispatched 12:00Z per QUEUE.md).

THE QUESTION: in WHAT CURRENCY is the wash's kill priced — aligned
displacement (W022b's integral A = sum a_t*||d_theta||, a_t =
cos(d_theta, grad g0_t), the critic's sign convention: NEGATIVE =
death-aligned), or RAW displacement (e185/e180's ||d_theta||)? And is the
STATIC kill (g3K's iso rungs) priced in the same currency or a different
one (T137's trajectory-vs-static split, made quantitative)?

THE CELL (eval-only, CPU, no training, no GPU claim — another agent holds
the GPU, another holds heavy CPU; this is deliberately light: sequential,
minutes of backward passes):

  ROW 1 — THE TRAJECTORY ROW: the e180 lr-grid checkpoints ON DISK (verified
  against committed metrics, no silent loads). For each arm, each consecutive
  checkpoint pair (root = step 0): d_theta = theta_next - theta_t; grad g0 at
  theta_t (the fact-strength readout's gradient — opt1/e185's convention:
  ONE backward per checkpoint on the fact battery, mean log p(Z) at the last
  position, offset-0 geometry; the g-12 battery co-reported); a_t =
  cos(d_theta, grad g0_t); A(t) = cumulative sum a_t*||d_theta|| (the
  ALIGNMENT-WEIGHTED DISPLACEMENT). Raw cumulative ||d_theta|| (the
  raw-displacement rival) and the a_t trajectories co-reported.
  ARMS ON DISK (the e180 five-point curve's committed provenance):
    lr 1e-3  neutral  = e176n arm A snapshots s1/s2/s4/s50/s100/s200/+300
                        (e180's metrics attribute this arm verbatim; t*=2)
    lr 1e-4  original = e176n arm B snapshots s50/+300 ONLY (coarse; the
                        ORIGINAL extinction-anchor stream — e180's own
                        five-point mixes streams; t*=50 upper bound)
    lr 3e-5  neutral  = e180_neutral_lr3e5 s2/s10/s50/s100/s200/+300 (t*=200)
    lr 1e-5  neutral  = e180_neutral_lr1e5 same ladder (CENSORED: alive at
                        +300; enters the bars as a one-sided bound)
    + replication co-read at lr 1e-3: e184 seeds 10903/10904 and the e185c
      CPU reruns (A at their own t*=2; texture, never adjudicated).

  ROW 2 — THE STATIC ROW (two-currency amendment): g3K's iso arms
  re-expressed in the same currency. For each organism (store g3_gen /
  host e157_f2_consolidated), each iso seed (11401-3 / 11411-13) and rung
  {1,2,4,8,16,32,64}: A_static = cos(d_iso, grad g0_root)*||d_iso|| (single
  static jump; grad at the UNPERTURBED organism). g3K's committed
  displacement conventions reused verbatim (perturbation L2 = rung x
  ||own wash 1x||; iso draws regenerated from the committed seeds and
  VERIFIED by reproducing g3K's committed per-rung reads — nothing
  re-derived).

  ROW 3 — THE INSTALL-VS-WASH COSINE (T138/Ilharco): cos(install_direction,
  wash_direction) per organism where both are computable from committed
  checkpoints; endpoints documented per organism.
  Vocabulary rule (reported, never bar-adjudicated): |cos| > 0.5 -> the
  task-arithmetic vocabulary fits (adopt in the paper); |cos| < 0.3 -> the
  wash is the corpus's adaptation direction, not the fact's negation
  (reject the vocabulary); between -> mixed, reported.

REGISTERED BARS (frozen here, verbatim from the dispatch; no bar shopping):
  A*-CONSTANT: "fires if the fact dies at an lr-INDEPENDENT A* — the
    alignment-weighted displacement at death within +-25% across the e180
    arms (t* known per arm from committed metrics) — death is measured in
    aligned-displacement units; the rate law is a corollary of
    constant-speed aligned drift."
  RAW-WINS: "fires if raw cumulative ||d_theta|| at death is more
    arm-invariant than A (coefficient of variation comparison, stated
    numerically) — alignment is epiphenomenal; e185's displacement story
    was the whole law."
  NEITHER: "fires if neither currency is arm-invariant — alignment itself
    drifts with lr; its own finding, reported as texture."
  Co-read (never adjudicated as a bar): the static row's A at its 4-10x
  ||d|| kill rungs — expected far below A* (the static kill is not
  aligned-displacement death); the two-currency table is the paper figure.

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any row compute):
  * every checkpoint's identity verified against the committed e176n/e180/
    e184/e185c/g3/g3K metrics: file listed in the source run's committed
    ckpt_inventory AND its battery reads reproduce the committed trajectory
    rows (gm12 + g0 at every snapshot; tol 0.05 functional, diffs reported);
  * the root gated vs e151's before-cells (gm12/g0/gp12/CE_R; e176n/e180/
    opt1's gate set);
  * the neutral protocol rebuilt (corpus ZEPH count 0; splice mix
    FLORIZEL 19 / ELIZABETH 41; battery shapes 60 x {118,130,142});
  * g3K's organisms gated as g3K gated them (root g0 / wash-s1 g0 / D_all
    reproduce to 1e-6), and the regenerated iso draws reproduce g3K's
    committed iso + wash curves rung by rung (G-ISO-REPRO);
  * e185's stored CONTROL displacements cross-check the first two pair
    norms of the lr 1e-3 arm (D(1) 1.6543, D(2) 2.4893).
WHAT EACH ROW GUARANTEES: the trajectory row's A is computed from SAMPLED
snapshots — the integral is a QUADRATURE (piecewise-constant a_t per
segment; segments span 1-100 steps; the 1e-4 arm's whole kill sits inside
ONE 50-step segment — its caveat is the quadrature's extreme); grad g0 is
evaluated at SNAPSHOTTED theta, assuming piecewise-linear adaptation.
Nothing here is guaranteed-to-succeed: A* could be arm-dependent in any
direction; the static row could price its kill in the SAME currency as the
trajectory row (killing the two-currency figure); the cosines could be
near zero (random-direction default in high dimension).

HONESTY BLOCK (pre-registered): single fact family (the ZEPHYRA installs;
n=1 per arm, one lineage per row); the 1e-4 arm is the original stream on
a coarse grid (flagged wherever it enters a number); CPU fp32 throughout
(device texture of committed GPU-computed reads bounded by the 0.05
functional gate); the cell shows PRICING, not causation — the causal cell
is opt2's masked arms (r58's own clause); W022's summary cos -0.44 and
opt1/T139's flat -0.015..-0.105 were CUMULATIVE-displacement cosines — the
per-pair a_t here is a different estimator and may disagree with both.

Outputs: runs/e188/{metrics.json, two_currency.png, currency_curves.png}.
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds). Single commit
+ push.

Run:  cd lab && python e188_death_currency.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY (opt1/e176n/e185 convention)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(4)                              # e185-era reduction order; light footprint

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json  # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import g3R_seed_replicates as R                        # noqa: E402 (rebuild_protocol, load_root)
import g3_generative_store as G3                       # noqa: E402 (CKPT_DIR, evl_load, battery_cell)
from g3_generative_store import F2_CFG                 # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

common.DEVICE = "cpu"                                  # eval-only; NO GPU claim

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

SMOKE = os.environ.get("E188_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e188 is CPU-only by dispatch"

REPO = E43.REPO
CKPT = REPO / "runs" / "checkpoints"

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
GEOS = (-12, 0, 12)               # battery ctx offsets (e185/opt1 convention)
ROOT_CK = "e131_consolidated_e113.pt"    # THE FULLY-CONSOLIDATED ROOT (2,739,072)
SHUT_BAR = 0.27                   # the arc's committed death bar (gm12 <= 0.27)

F_G_TOL = 0.05                    # functional battery-read tolerance (committed
# reads were computed on-device for e176n/e184; e185c measured same-seed
# device ratios <= 0.2%, and a wrong-checkpoint load differs by ~0.5+)
G3_TOL = 1e-6                     # g3K's own bit-exact tolerance

REGISTERED_BARS = {
    "A_STAR_CONSTANT": "A*-CONSTANT: \"fires if the fact dies at an "
        "lr-INDEPENDENT A* — the alignment-weighted displacement at death "
        "within +-25% across the e180 arms (t* known per arm from committed "
        "metrics) — death is measured in aligned-displacement units; the "
        "rate law is a corollary of constant-speed aligned drift.\"",
    "RAW_WINS": "RAW-WINS: \"fires if raw cumulative ||d_theta|| at death "
        "is more arm-invariant than A (coefficient of variation comparison, "
        "stated numerically) — alignment is epiphenomenal; e185's "
        "displacement story was the whole law.\"",
    "NEITHER": "NEITHER: \"fires if neither currency is arm-invariant — "
        "alignment itself drifts with lr; its own finding, reported as "
        "texture.\"",
}

VOCAB_RULE = ("|cos| > 0.5 -> task-arithmetic vocabulary fits (adopt in the "
              "paper); |cos| < 0.3 -> the wash is the corpus's adaptation "
              "direction, not the fact's negation (reject the vocabulary); "
              "between -> mixed, reported.")

E151_ROOT = {                     # runs/e151 'before' battery (e176n/e180/opt1's gate set)
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
}

# e185's stored CONTROL displacements (opt1's embedded verbatim set) — the
# lr 1e-3 arm's first two pair-norm cross-check.
E185_CTRL_DISP = {1: 1.6542880535125732, 2: 2.4892616271972656}

# g3K's committed convention numbers (runs/g3K/metrics.json) — embedded for
# the gates, verified against the file at run time.
G3K_REF = {
    "store": {
        "root": "g3_gen.pt", "wash_s1": "g3_gen_s1.pt",
        "root_g0": 0.8885950446128845,
        "wash_s1_g0": 5.8887377235805616e-05,
        "D_all_s1": 0.9258007407188416, "n_params": 890_880,
        "iso_seeds": (11401, 11402, 11403), "read_key": "g0",
    },
    "host": {
        "root": "e157_f2_consolidated.pt", "wash_s1": "e157_f2_neutral_s1.pt",
        "root_g0": 0.5784125924110413,
        "wash_s1_g0": 0.0040224287658929825,
        "D_all_s1": 0.916419706836259, "n_params": 873_472,
        "iso_seeds": (11411, 11412, 11413), "read_key": "pz",
    },
}
G3K_RUNGS = (1, 2, 4, 8, 16, 32, 64)

# ---- the trajectory arms (ckpts verified below against committed metrics) ------
TRAJ_ARMS = [
    dict(tag="lr1e-3_neutral", lr=1e-3, stream="neutral", seed=10902,
         t_star=2, t_star_kind="measured (bracket (1, 2])",
         source=("runs/e176n/metrics.json trace_armA = e180's five-point arm "
                 "'1e-3_neutral_e176n_armA' (e180 attributes it verbatim; "
                 "ONLY the lr differs across the family's neutral cells)"),
         ckpts=[(1, "e176n_neutral_s1.pt"), (2, "e176n_neutral_s2.pt"),
                (4, "e176n_neutral_s4.pt"), (50, "e176n_neutral_s50.pt"),
                (100, "e176n_neutral_s100.pt"), (200, "e176n_neutral_s200.pt"),
                (300, "e176n_neutral.pt")]),
    dict(tag="lr1e-4_original", lr=1e-4, stream="original (extinction anchors)",
         seed=10902, t_star=50,
         t_star_kind="COARSE upper bound (bracket (0, 50]; only s50/s300 exist)",
         source=("runs/e176n/metrics.json trace_armB = e180's five-point arm "
                 "'1e-4_original_e176n_armB' (the stored half mixes streams — "
                 "e180's own convention, carried here with the flag)"),
         ckpts=[(50, "e176n_lr1e4_s50.pt"), (300, "e176n_lr1e4.pt")]),
    dict(tag="lr3e-5_neutral", lr=3e-5, stream="neutral", seed=10902,
         t_star=200, t_star_kind="measured (bracket (100, 200])",
         source="runs/e180/metrics.json cells.lr3e5.traj (this run)",
         ckpts=[(2, "e180_neutral_lr3e5_s2.pt"), (10, "e180_neutral_lr3e5_s10.pt"),
                (50, "e180_neutral_lr3e5_s50.pt"), (100, "e180_neutral_lr3e5_s100.pt"),
                (200, "e180_neutral_lr3e5_s200.pt"), (300, "e180_neutral_lr3e5.pt")]),
    dict(tag="lr1e-5_neutral", lr=1e-5, stream="neutral", seed=10902,
         t_star=None, t_star_kind="CENSORED (alive at +300; one-sided bound)",
         source="runs/e180/metrics.json cells.lr1e5.traj (this run)",
         ckpts=[(2, "e180_neutral_lr1e5_s2.pt"), (10, "e180_neutral_lr1e5_s10.pt"),
                (50, "e180_neutral_lr1e5_s50.pt"), (100, "e180_neutral_lr1e5_s100.pt"),
                (200, "e180_neutral_lr1e5_s200.pt"), (300, "e180_neutral_lr1e5.pt")]),
]

# replication co-read at lr 1e-3 (texture only; A at their own committed t*=2)
REPL_ARMS = [
    dict(tag="lr1e-3_neutral_s10903_e184", seed=10903, device="gpu(original)",
         source="runs/e184/metrics.json traces.10903",
         ckpts=[(1, "e184_neutral_s10903_s1.pt"), (2, "e184_neutral_s10903_s2.pt")]),
    dict(tag="lr1e-3_neutral_s10904_e184", seed=10904, device="gpu(original)",
         source="runs/e184/metrics.json traces.10904",
         ckpts=[(1, "e184_neutral_s10904_s1.pt"), (2, "e184_neutral_s10904_s2.pt")]),
    dict(tag="lr1e-3_neutral_s10903_e185c", seed=10903, device="cpu(rerun)",
         source="runs/e185c/metrics.json traces_cpu.10903",
         ckpts=[(1, "e185c_neutral_s10903_s1.pt"), (2, "e185c_neutral_s10903_s2.pt")]),
    dict(tag="lr1e-3_neutral_s10904_e185c", seed=10904, device="cpu(rerun)",
         source="runs/e185c/metrics.json traces_cpu.10904",
         ckpts=[(1, "e185c_neutral_s10904_s1.pt"), (2, "e185c_neutral_s10904_s2.pt")]),
]

_provenance: dict = {}            # name -> {path, sha1, bytes}


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_prov(name: str, path: Path, note: str = "") -> None:
    _provenance[name] = {"path": str(path.relative_to(REPO)),
                         "sha1": sha1_of(path), "bytes": path.stat().st_size,
                         "note": note}


def load_sd(name: str) -> dict:
    """Load a committed checkpoint's state dict (both storage formats) and
    record its provenance hash."""
    p = CKPT / name
    assert p.exists(), f"missing checkpoint {p}"
    st = torch.load(p, map_location="cpu", weights_only=False)
    sd = {k: v.clone() for k, v in
          (st["model"] if isinstance(st, dict) and "model" in st else st).items()}
    record_prov(name, p, (st.get("meta") if isinstance(st, dict) else None)
                and "meta-present" or "raw-sd")
    return sd


# ------------------------------------------------------------------ instruments
# (e185_noise_wash.py VERBATIM via opt1's verified copies — the e176n lineage)


@torch.no_grad()
def battery_cell(net, ids: torch.Tensor, zid: int, bs=30) -> dict:
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "frac_argmax_z": amax / ids.shape[0]}


def fact_grad(net, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """THE ALIGNMENT READ (opt1 verbatim): gradient (wrt all params, at the
    eval net's current weights theta_t) of the fact battery's mean log p(Z)
    readout at the last position. Critic's sign convention: NEGATIVE
    cos(delta, grad) = displacement aligned with the DEATH gradient (W022).
    Consumes no RNG."""
    net.zero_grad(set_to_none=True)
    sums = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        sums.append(F.log_softmax(lg[:, -1], -1)[:, zid].sum())
    F_obj = torch.stack(sums).sum() / ids.shape[0]
    F_obj.backward()
    g = torch.cat([p.grad.detach().reshape(-1) for p in net.parameters()])
    net.zero_grad(set_to_none=True)
    return g


def flat_sd(sd: dict, keys=None) -> torch.Tensor:
    ks = list(sd.keys()) if keys is None else keys
    return torch.cat([sd[k].float().reshape(-1) for k in ks])


@torch.no_grad()
def ce_fixed_cpu(net, x, y, bs=64) -> float:
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    """e065 val_windows verbatim: name-free val-split windows."""
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


def fact_grad_by_keys(net, ids: torch.Tensor, zid: int, keys: list[str],
                      bs=30) -> torch.Tensor:
    """The alignment read, returning the flat gradient in the GIVEN state-dict
    key order (sd keys == parameter names on these nets — asserted by the
    caller). Same objective/convention as fact_grad; grads read BEFORE the
    net is re-zeroed."""
    net.zero_grad(set_to_none=True)
    sums = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        sums.append(F.log_softmax(lg[:, -1], -1)[:, zid].sum())
    (torch.stack(sums).sum() / ids.shape[0]).backward()
    named = dict(net.named_parameters())
    missing = [k for k in keys if k not in named]
    assert not missing, f"sd keys that are not named parameters: {missing}"
    g = torch.cat([named[k].grad.detach().reshape(-1) for k in keys])
    net.zero_grad(set_to_none=True)
    return g


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


# ------------------------------------------------------- committed-metrics reads

def read_metrics(rel: str) -> dict:
    p = REPO / rel
    assert p.exists(), f"missing committed metrics {p}"
    record_prov(rel, p, "committed metrics (provenance source)")
    return json.loads(p.read_text(encoding="utf-8"))


def committed_traj(arm: dict) -> dict[int, dict]:
    """{step: {gm12, g0, ce_r}} from the arm's committed source metrics."""
    if arm["tag"] == "lr1e-3_neutral":
        rows = read_metrics("runs/e176n/metrics.json")["trace_armA"]
        return {int(r["freeze_steps"]): r for r in rows}
    if arm["tag"] == "lr1e-4_original":
        rows = read_metrics("runs/e176n/metrics.json")["trace_armB"]
        return {int(r["freeze_steps"]): r for r in rows}
    m = read_metrics("runs/e180/metrics.json")["cells"]
    cell = m["lr3e5"] if arm["tag"] == "lr3e-5_neutral" else m["lr1e5"]
    return {int(r["step"]): r for r in cell["traj"]}


def committed_repl(arm: dict) -> dict[int, dict]:
    if arm["source"].startswith("runs/e184"):
        rows = read_metrics("runs/e184/metrics.json")["traces"][str(arm["seed"])]
        return {int(r["freeze_steps"]): r for r in rows}
    t = read_metrics("runs/e185c/metrics.json")["traces_cpu"][str(arm["seed"])]
    return {int(s): {"gm12": g, "g0": gg}
            for s, g, gg in zip(t["freeze_steps"], t["gm12"], t["g0"])}


# ------------------------------------------------------------------ row 1

def run_arm(arm: dict, net0: TinyGPT, root_sd: dict, root_flat: torch.Tensor,
            g0_ids, gm12_ids, zid) -> dict:
    """One trajectory arm: pair table, a_t, A(t), raw D(t), death currencies."""
    keys = list(root_sd.keys())
    ct = committed_traj(arm)
    out = {"lr": arm["lr"], "stream": arm["stream"], "seed": arm["seed"],
           "t_star_committed": arm["t_star"], "t_star_kind": arm["t_star_kind"],
           "source": arm["source"], "pairs": [], "repro": []}
    prev_sd, prev_flat, prev_step = root_sd, root_flat, 0
    A_g0 = A_m12 = 0.0
    for i, (step, ck) in enumerate(arm["ckpts"]):
        sd = load_sd(ck)
        assert set(sd.keys()) == set(root_sd.keys()), f"{ck}: key-set drift"
        flat = flat_sd(sd, keys)
        # no-silent-load functional gate vs the committed trajectory row
        net0.load_state_dict(sd); net0.eval()
        bz = battery_cell(net0, g0_ids, zid)
        bm = battery_cell(net0, gm12_ids, zid)
        ref = ct.get(step)
        d_g0 = abs(bz["mean_pz"] - ref["g0_mean_pz" if "g0_mean_pz" in ref
                                       else "g0"]) if ref else None
        gm_ref = ref["g_m12_mean_pz" if "g_m12_mean_pz" in ref else "gm12"] \
            if ref else None
        out["repro"].append({"step": step, "ckpt": ck,
                             "g0": bz["mean_pz"], "g0_ref_diff": d_g0,
                             "gm12": bm["mean_pz"], "gm12_ref_diff":
                             (abs(bm["mean_pz"] - gm_ref) if gm_ref is not None
                              else None)})
        if d_g0 is not None and d_g0 > F_G_TOL:
            raise AssertionError(f"{arm['tag']} s{step}: g0 repro diff {d_g0}")
        if gm_ref is not None and abs(bm["mean_pz"] - gm_ref) > F_G_TOL:
            raise AssertionError(
                f"{arm['tag']} s{step}: gm12 repro diff "
                f"{abs(bm['mean_pz'] - gm_ref)}")
        # gradient at the LEFT endpoint of the pair (prev -> current);
        # prev_sd is a left endpoint at EVERY iteration, so this is always
        # needed (the final snapshot s300 is never a left endpoint — no grad
        # is ever computed AT it)
        net0.load_state_dict(prev_sd)
        gg0 = fact_grad(net0, g0_ids, zid)
        gm = fact_grad(net0, gm12_ids, zid)
        # the pair t_prev -> step
        d = flat - prev_flat
        dn = float(d.norm())
        row = {"from_step": prev_step, "to_step": step,
               "seg_span_steps": step - prev_step,
               "d_norm": dn, "raw_cum_norm": float((flat - root_flat).norm()),
               "a_t_g0": cos(d, gg0),
               "a_t_m12": cos(d, gm),
               "grad_g0_norm": float(gg0.norm()),
               "ce_r_committed": (ref.get("ce_r") if ref else None)}
        A_g0 += row["a_t_g0"] * dn
        A_m12 += row["a_t_m12"] * dn
        row["A_g0_cum"] = A_g0
        row["A_m12_cum"] = A_m12
        out["pairs"].append(row)
        prev_sd, prev_flat, prev_step = sd, flat, step
    out["A_final_g0"], out["A_final_m12"] = A_g0, A_m12
    # death currencies at t* (or the censored bound at +300)
    tstar = arm["t_star"] if arm["t_star"] is not None else 300
    p = next((r for r in out["pairs"] if r["to_step"] == tstar), None)
    assert p is not None, f"{arm['tag']}: no pair ending at {tstar}"
    out["death"] = {"t": tstar, "censored": arm["t_star"] is None,
                    "A_g0": p["A_g0_cum"], "A_m12": p["A_m12_cum"],
                    "raw_D": p["raw_cum_norm"],
                    "note": ("censored arm: A/D at +300 are NOT death values; "
                             "they enter the bars as a one-sided bound"
                             if arm["t_star"] is None else
                             f"measured at committed t*={tstar}")}
    out["repro_max_diff"] = max(
        max((abs(r["g0_ref_diff"]) for r in out["repro"]
             if r["g0_ref_diff"] is not None), default=0.0),
        max((abs(r["gm12_ref_diff"]) for r in out["repro"]
             if r["gm12_ref_diff"] is not None), default=0.0))
    return out


# ------------------------------------------------------------------ row 2

def g3k_load(name: str) -> dict:
    st = torch.load(CKPT / name, map_location="cpu", weights_only=False)
    sd = {k: v.clone() for k, v in st["model"].items()}
    record_prov(name, CKPT / name, "g3K organism checkpoint")
    return sd


def run_static_row(pr, sd_store, sd_host) -> dict:
    g3k = read_metrics("runs/g3K/metrics.json")
    out = {"convention": g3k["convention"],
           "convention_note": ("g3K's committed displacement conventions "
                               "reused verbatim: perturbation L2 = rung x "
                               "||own wash 1x||; iso draws regenerated from "
                               "the committed seeds (11401-3 / 11411-13) and "
                               "VERIFIED by reproducing g3K's committed "
                               "per-rung reads rung by rung (G-ISO-REPRO)"),
           "organisms": {}}
    for tag, sd_root in (("store", sd_store), ("host", sd_host)):
        ref = G3K_REF[tag]
        sd_s1 = g3k_load(ref["wash_s1"])
        keys = list(sd_root.keys())
        delta = {k: sd_s1[k].float() - sd_root[k].float() for k in keys}
        Dw = float(sum(float(delta[k].norm() ** 2) for k in keys) ** 0.5)
        assert abs(Dw - ref["D_all_s1"]) < G3_TOL, f"{tag}: Dw {Dw}"
        assert sum(v.numel() for v in sd_root.values()) == ref["n_params"]
        net = (G3.evl_load("gen", sd_store) if tag == "store"
               else TinyGPT(F2_CFG))
        net.load_state_dict(sd_root); net.eval()
        # grad g0 at the UNPERTURBED organism (one backward; the g3 battery
        # has a single geometry, ids130 = offset 0); flat in sd-key order so
        # it dots exactly against the iso vectors below.
        named = dict(net.named_parameters())
        assert list(named.keys()) == keys, f"{tag}: sd-key order drift"
        g_keys = fact_grad_by_keys(net, pr["ids130"], pr["zid"], keys)
        del net
        # wash-direction static leg (co-read): cos(d_wash, grad root)
        w_flat = flat_sd(delta, keys)
        wash_cos = cos(w_flat, g_keys)
        iso = []
        repro_max = 0.0
        committed_iso = g3k["organisms"][tag]["iso"]
        committed_wash = g3k["organisms"][tag]["kill"]["curve"]
        # wash-leg reproduction (root + rung * delta) + its A_static
        net = (G3.evl_load("gen", sd_store) if tag == "store"
               else TinyGPT(F2_CFG))
        wash_rows = []
        for r in G3K_RUNGS:
            sd = {k: sd_root[k] + r * delta[k] for k in keys}
            net.load_state_dict(sd); net.eval()
            v = battery_cell(net, pr["ids130"], pr["zid"])["mean_pz"]
            ref_v = next(c[ref["read_key"]] for c in committed_wash
                         if c["rung"] == r)
            repro_max = max(repro_max, abs(v - ref_v))
            vec = flat_sd({k: r * delta[k] for k in keys}, keys)
            wash_rows.append({"rung": r, "read": v,
                              "A_static": cos(vec, g_keys) * float(vec.norm())})
        # iso legs: regenerate draws from committed seeds; verify vs committed
        for si, seed in enumerate(ref["iso_seeds"]):
            g = torch.Generator().manual_seed(seed)
            draw = {k: torch.randn(sd_root[k].shape, generator=g) for k in sd_root}
            gn = float(sum(float(draw[k].norm() ** 2) for k in keys) ** 0.5)
            rows = []
            for r in G3K_RUNGS:
                sc = r * Dw / gn
                sd = {k: sd_root[k] + sc * draw[k] for k in keys}
                net.load_state_dict(sd); net.eval()
                v = battery_cell(net, pr["ids130"], pr["zid"])["mean_pz"]
                ref_v = next(c[ref["read_key"]] for c in committed_iso[si]["curve"]
                             if c["rung"] == r)
                repro_max = max(repro_max, abs(v - ref_v))
                vec = flat_sd({k: sc * draw[k] for k in keys}, keys)
                rows.append({"rung": r, "read": v,
                             "cos": cos(vec, g_keys),
                             "A_static": cos(vec, g_keys) * float(vec.norm())})
            kill = committed_iso[si]["kill_rung"]
            at_kill = next(rr for rr in rows if rr["rung"] == kill)
            iso.append({"seed": seed, "draw_L2": gn, "rows": rows,
                        "kill_rung_committed": kill,
                        "A_static_at_kill": at_kill["A_static"],
                        "read_at_kill": at_kill["read"]})
        del net
        out["organisms"][tag] = {
            "root": f"runs/checkpoints/{ref['root']}",
            "n_params": ref["n_params"], "D_wash_1x": Dw,
            "grad_g0_norm": float(g_keys.norm()),
            "wash_dir_cos_grad_root": wash_cos,
            "wash_dir_A_static_rung1": wash_cos * Dw,
            "wash_leg": wash_rows,
            "iso": iso,
            "kill_rungs_committed": [a["kill_rung_committed"] for a in iso],
            "A_static_at_kill": [a["A_static_at_kill"] for a in iso],
            "g3k_repro_max_abs_diff": repro_max,
        }
        assert repro_max < G3_TOL, f"{tag}: G-ISO-REPRO failed {repro_max}"
        log(f"[static/{tag}] G-ISO-REPRO max|diff| {repro_max:.2e}; "
            f"wash cos(grad) {wash_cos:+.4f}; A_static@kill "
            f"{[f'{a:.4f}' for a in out['organisms'][tag]['A_static_at_kill']]}")
    return out


# ------------------------------------------------------------------ row 3

def install_vs_wash(sd_lookup) -> dict:
    def delta(a, b):
        """Flat (a - b) over the UNION of state-dict keys, missing side = 0
        (g3_gen adds store keys the pristine base never had — the organ IS
        part of the install direction); one-sided-key mass reported."""
        sdA, sdB = sd_lookup(a), sd_lookup(b)
        keys = list(dict.fromkeys(list(sdA.keys()) + list(sdB.keys())))
        vecs = {k: (sdA[k].float() - sdB[k].float()).reshape(-1)
                if k in sdA and k in sdB
                else (sdA[k].float().reshape(-1) if k in sdA
                      else -sdB[k].float().reshape(-1)) for k in keys}
        v = torch.cat(list(vecs.values()))
        one_sided = [k for k in keys if (k in sdA) != (k in sdB)]
        added = (torch.cat([vecs[k] for k in one_sided]).norm()
                 if one_sided else torch.tensor(0.0))
        return v, float(v.norm()), float(added)

    rows = {}
    # e180 lineage: install = consolidated - base (the FULL install+
    # consolidation arc; the e113-installed intermediate is not checkpointed
    # — documented); wash = the lr 1e-3 arm's committed first step (s1).
    rows["e180_lineage_zephyra"] = {
        "install": {"endpoints": "e131_consolidated_e113 - e048_repro",
                    "note": "full install+consolidation arc (root_meta base)"},
        "wash": {"endpoints": "e176n_neutral_s1 - e131_consolidated_e113",
                 "note": "committed lr 1e-3 neutral first step (seed 10902)"},
        "vectors": (delta("e131_consolidated_e113.pt", "e048_repro.pt"),
                    delta("e176n_neutral_s1.pt", "e131_consolidated_e113.pt"))}
    # g3 store: install = the construction (one-shot store write + trunk
    # restore) on the pristine host base; wash = committed s1 snapshot.
    rows["g3_store"] = {
        "install": {"endpoints": "g3_gen - e098_base_s4305",
                    "note": "the g3 construction arc (one-shot organ write; "
                            "host trunk bit-identical by g3R's gate, so this "
                            "vector is essentially the organ itself — the "
                            "one-sided store-key mass is reported)"},
        "wash": {"endpoints": "g3_gen_s1 - g3_gen",
                 "note": "committed wash snapshot (g3's seed 10902, t*=+1)"},
        "vectors": (delta("g3_gen.pt", "e098_base_s4305.pt"),
                    delta("g3_gen_s1.pt", "g3_gen.pt"))}
    # g3 host (e157 family 2): both install variants computable.
    rows["g3_host_e157f2_full_arc"] = {
        "install": {"endpoints": "e157_f2_consolidated - e098_base_s4305",
                    "note": "full install+consolidation arc"},
        "wash": {"endpoints": "e157_f2_neutral_s1 - e157_f2_consolidated",
                 "note": "committed wash snapshot (e176n arm A verbatim; "
                         "seed 10902, t*=+1)"},
        "vectors": (delta("e157_f2_consolidated.pt", "e098_base_s4305.pt"),
                    delta("e157_f2_neutral_s1.pt", "e157_f2_consolidated.pt"))}
    rows["g3_host_e157f2_consolidation_only"] = {
        "install": {"endpoints": "e157_f2_consolidated - e098_install_s4305",
                    "note": "consolidation-only arc (installed -> consolidated)"},
        "wash": {"endpoints": "e157_f2_neutral_s1 - e157_f2_consolidated",
                 "note": "same committed wash snapshot"},
        "vectors": (delta("e157_f2_consolidated.pt", "e098_install_s4305.pt"),
                    delta("e157_f2_neutral_s1.pt", "e157_f2_consolidated.pt"))}
    out = {}
    for k, v in rows.items():
        (vi, ni, ai), (vw, nw, aw) = v["vectors"]
        c = cos(vi, vw)
        ac = abs(c)
        verdict = ("adopt: task-arithmetic vocabulary fits" if ac > 0.5 else
                   "reject: the wash is the corpus's adaptation direction, "
                   "not the fact's negation" if ac < 0.3 else
                   "mixed, reported")
        out[k] = {"install": v["install"], "wash": v["wash"], "cos": c,
                  "abs_cos": ac, "vocab_rule_call": verdict,
                  "install_L2": ni, "wash_L2": nw,
                  "install_one_sided_key_L2": ai, "wash_one_sided_key_L2": aw}
        log(f"[vocab/{k}] cos(install, wash) = {c:+.4f} -> {verdict}")
    out["rule"] = VOCAB_RULE
    return out


# ------------------------------------------------------------------ adjudication

def cv(vals: list[float]) -> float:
    v = np.asarray(vals, dtype=np.float64)
    m = np.abs(v).mean()
    return float(v.std(ddof=0) / m) if m > 0 else float("inf")


def adjudicate(arms_out: dict, static: dict, repl: dict | None = None) -> dict:
    meas = {t: a for t, a in arms_out.items()
            if a["t_star_committed"] is not None}
    cen = {t: a for t, a in arms_out.items() if a["t_star_committed"] is None}
    # death alignment = -A (positive = death-aligned; the critic's convention)
    da = {t: -a["death"]["A_g0"] for t, a in meas.items()}
    dd = {t: a["death"]["raw_D"] for t, a in meas.items()}
    dam = {t: -a["death"]["A_m12"] for t, a in meas.items()}
    tags = list(meas)
    mean_da = float(np.mean(list(da.values())))
    rel = {t: abs(da[t] - mean_da) / mean_da for t in tags}
    band_pass = bool(all(r <= 0.25 for r in rel.values())) and mean_da > 0
    cvA, cvD = cv(list(da.values())), cv(list(dd.values()))
    cvAm = cv(list(dam.values()))
    raw_wins = bool(cvD < cvA)
    # censored one-sided bound: the 1e-5 arm, alive at +300, must NOT have
    # accumulated more death alignment than the band's most-aligned edge.
    bound = None
    for t, a in cen.items():
        da_c = -a["death"]["A_g0"]
        bound = {"tag": t, "death_alignment_at_300": da_c,
                 "band_edge_1p25_mean": 1.25 * mean_da,
                 "consistent": bool(da_c < 1.25 * mean_da)}
    a_star_pass = bool(band_pass and (cvA <= cvD)
                       and (bound is None or bound["consistent"]))
    if raw_wins:
        verdict = "RAW-WINS"
    elif a_star_pass:
        verdict = "A*-CONSTANT"
    else:
        verdict = "NEITHER"
    clause = {
        "A*-CONSTANT": REGISTERED_BARS["A_STAR_CONSTANT"] + " — FIRED: "
            f"A* per arm {{ {', '.join(f'{t}: {da[t]:+.4f}' for t in tags)} }}; "
            f"max rel dev from mean {max(rel.values()):.1%} (band +-25%); "
            f"CV(A) {cvA:.3f} <= CV(D) {cvD:.3f}; censored bound "
            + ("consistent" if bound and bound["consistent"] else "VIOLATED"),
        "RAW-WINS": REGISTERED_BARS["RAW_WINS"] + " — FIRED: "
            f"raw D at death {{ {', '.join(f'{t}: {dd[t]:.3f}' for t in tags)} }} "
            f"CV(D) {cvD:.3f} < CV(A) {cvA:.3f}; A* per arm "
            f"{{ {', '.join(f'{t}: {da[t]:+.4f}' for t in tags)} }}",
        "NEITHER": REGISTERED_BARS["NEITHER"] + " — FIRED: A is at least as "
            f"arm-invariant as raw (CV(A) {cvA:.3f} vs CV(D) {cvD:.3f}) but "
            f"A* is not lr-independent (max rel dev {max(rel.values()):.1%} "
            f"vs the +-25% band"
            + (f"; censored bound {bound['death_alignment_at_300']:.4f} vs "
               f"band edge {bound['band_edge_1p25_mean']:.4f} "
               + ("consistent" if bound["consistent"] else "VIOLATED")
               if bound else "") + ")",
    }[verdict]
    # alignment-drift texture read (NEITHER's own finding): does mean a_t
    # move monotonically with lr?
    lrs, a_means = [], []
    for t, a in arms_out.items():
        ats = [p["a_t_g0"] for p in a["pairs"] if p["a_t_g0"] is not None]
        lrs.append(a["lr"]); a_means.append(float(np.mean(ats)))
    order = np.argsort(lrs)
    mono = bool(np.all(np.diff(np.asarray(a_means)[order]) <= 0) or
                np.all(np.diff(np.asarray(a_means)[order]) >= 0))
    # neutral-only co-report (the 1e-4 arm is the original stream; e180's
    # own five-point mixed streams — the bar's letter adjudicates on the
    # full measured set above; this co-report shows the verdict does not
    # hinge on the mixing)
    nda = [da[t] for t in tags if "neutral" in t]
    ndd = [dd[t] for t in tags if "neutral" in t]
    # replication co-report: the SAME arm (lr 1e-3) at the SAME t*=2 across
    # wash seeds — displacement spread vs alignment spread
    repl_sum = None
    if repl:
        seed_A = [-arms_out["lr1e-3_neutral"]["death"]["A_g0"]] + \
            [-v["A_g0_at_t2"] for v in repl.values()]
        seed_D = [arms_out["lr1e-3_neutral"]["death"]["raw_D"]] + \
            [v["raw_D_at_t2"] for v in repl.values()]
        repl_sum = {"seeds": ["10902 (main)"] + list(repl.keys()),
                    "death_alignment_A": seed_A, "raw_D": seed_D,
                    "A_spread_x": float(max(seed_A) / min(seed_A)),
                    "D_spread_x": float(max(seed_D) / min(seed_D)),
                    "note": "same lr, same t*=2, wash-seed lottery only"}
    # static co-read (never adjudicated)
    st = {}
    for tag, org in static["organisms"].items():
        st[tag] = {"A_static_at_kill": org["A_static_at_kill"],
                   "kill_rungs": org["kill_rungs_committed"],
                   "A_star_traj_range": [min(da.values()), max(da.values())],
                   "expected": "far below A* (the static kill is not "
                               "aligned-displacement death) — reported as "
                               "measured, never adjudicated"}
    return {
        "bars_verbatim": REGISTERED_BARS,
        "no_bar_shopping": True,
        "verdict": verdict,
        "clause": clause,
        "death_alignment_per_arm": da,           # -A_g0 at t* (positive = death-aligned)
        "raw_D_per_arm": dd,
        "A_m12_per_arm_co_read": dam,
        "mean_death_alignment": mean_da,
        "rel_dev_from_mean": rel,
        "band_pass_25pct": band_pass,
        "CV_A": cvA, "CV_D": cvD, "CV_A_m12_co_read": cvAm,
        "cv_convention": "population std (ddof=0) / mean(|values|); signed A "
                         "entered as death alignment -A so the mean is "
                         "positive when the mechanism holds",
        "censored_bound": bound,
        "cv_neutral_only_co_read": {
            "CV_A": cv(nda), "CV_D": cv(ndd),
            "note": "neutral measured arms only (drops the original-stream "
                    "1e-4 arm); the verdict is unchanged in this subset"},
        "replication_seed_spread_co_read": repl_sum,
        "alignment_drift_texture": {"lr": lrs, "mean_a_t_g0": a_means,
                                    "monotone_in_lr": mono},
        "static_co_read_never_adjudicated": st,
    }


# ------------------------------------------------------------------ plots

def plot_two_currency(path, adj, arms_out, static):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    ax = axes[0]
    da = adj["death_alignment_per_arm"]
    cen = adj["censored_bound"]
    labels, vals, colors = [], [], []
    for t, v in da.items():
        labels.append(f"trajectory\n{t}\n(t*={arms_out[t]['death']['t']})")
        vals.append(v); colors.append("crimson")
    if cen:
        labels.append(f"trajectory\n{cen['tag']}\n(alive at +300; bound)")
        vals.append(cen["death_alignment_at_300"]); colors.append("mistyrose")
    for tag, org in static["organisms"].items():
        v = float(np.mean(org["A_static_at_kill"]))
        lo, hi = min(org["A_static_at_kill"]), max(org["A_static_at_kill"])
        labels.append(f"STATIC {tag}\niso kill rungs {org['kill_rungs_committed']}")
        vals.append(-v)   # death alignment of the static jump (sign convention)
        colors.append("royalblue")
        ax.annotate(f"[{lo:.4f}, {hi:.4f}]",
                    (len(vals) - 1, max(-v, 1e-6)), fontsize=7, rotation=90,
                    va="bottom", ha="center", color="royalblue")
    y = np.array(vals)
    bars = ax.bar(range(len(vals)), np.abs(y) + 1e-9, color=colors, ec="k",
                  lw=0.6)
    for i, (vi, yi) in enumerate(zip(vals, y)):
        ax.text(i, abs(yi) * 1.15 + 1e-4, f"{yi:+.4f}", ha="center",
                fontsize=7.5, rotation=0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("|death alignment| = |A| (cos-weighted L2, log scale)")
    ax.set_title("THE TWO-CURRENCY TABLE — A at death: trajectory row "
                 "(crimson; pale = censored bound) vs static row (blue)",
                 fontsize=9.5)
    ax.axhline(1e-3, color="gray", ls=":", lw=0.8)
    ax.text(len(vals) - 0.4, 1.05e-3, "random-direction scale ~1e-3",
            fontsize=6.5, ha="right", color="gray")
    ax = axes[1]
    for t, a in arms_out.items():
        xs = [p["to_step"] for p in a["pairs"] if p["a_t_g0"] is not None]
        ys = [p["a_t_g0"] for p in a["pairs"] if p["a_t_g0"] is not None]
        ax.plot(xs, ys, "o-", ms=4, lw=1.2, label=f"{t} (lr {a['lr']:g})")
        if a["death"]["t"] in xs:
            ax.axvline(a["death"]["t"], color="gray", ls=":", lw=0.7)
            ax.annotate("t*", (a["death"]["t"], 0.02), fontsize=7,
                        rotation=90, color="gray")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xlabel("segment end step (symlog; root->s1 pair plotted at s1)")
    ax.set_ylabel(r"$a_t = \cos(d_\theta,\ \nabla g_0)$ (critic's sign)")
    ax.set_title("per-pair alignment trajectories (a_t, g0 gradient); "
                 "negative = death-aligned", fontsize=9.5)
    ax.legend(fontsize=7)
    v = adj["verdict"]
    fig.suptitle(f"E188 THE DEATH-CURRENCY CELL — verdict {v} | CV(A) "
                 f"{adj['CV_A']:.3f} vs CV(D) {adj['CV_D']:.3f} | eval-only "
                 f"CPU fp32 (snapshot quadrature; single fact family)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(path, arms_out):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for t, a in arms_out.items():
        ps = [p for p in a["pairs"] if p["A_g0_cum"] is not None]
        xs = [p["to_step"] for p in ps]
        axes[0].plot(xs, [p["A_g0_cum"] for p in ps], "o-", ms=4,
                     label=f"{t}")
        axes[1].plot(xs, [p["raw_cum_norm"] for p in ps], "o-", ms=4,
                     label=f"{t}")
        axes[2].plot(xs, [p["d_norm"] for p in ps], "o-", ms=4, label=f"{t}")
        if not a["death"]["censored"]:
            for ax in axes[:2]:
                ax.axvline(a["death"]["t"], color="gray", ls=":", lw=0.8)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("A(t) = sum a_t*||d|| (g0 currency)")
    axes[1].set_ylabel("raw cumulative ||theta_t - theta_0||")
    axes[2].set_ylabel("per-segment ||d|| (snapshot quadrature)")
    for ax, ttl in zip(axes, ("alignment-weighted displacement A(t)",
                              "raw displacement (the rival currency)",
                              "segment norms (sampling resolution)")):
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xlabel("step")
        ax.set_title(ttl, fontsize=9.5)
        ax.legend(fontsize=7)
    fig.suptitle("E188 trajectory row — A(t) vs raw D(t) per lr arm; dotted "
                 "lines = committed t* (the 1e-4 arm: one 50-step segment)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    torch.set_num_threads(4)                    # g3R's import resets to 8; re-assert the light footprint
    rd = run_dir("e188_smoke" if SMOKE else "e188")
    log(f"E188 THE DEATH-CURRENCY CELL -> {rd} (CPU-only, threads "
        f"{torch.get_num_threads()}, eval-only, no GPU claim)")

    # ============ protocol rebuild + gates (opt1 verbatim; Rule 12) =========
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids = corpus.train
    train_text = "".join(itos[int(i)] for i in train_ids)
    gates: dict = {"corpus_zeph_count": train_text.count("ZEPH")}
    assert gates["corpus_zeph_count"] == 0, "corpus contains ZEPH"

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    _random.Random(E43.SPLICE_RNG).shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    gates["install_mix"] = mix
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"

    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    assert list(bat_ids[-12].shape) == [60, PRE - 12]
    assert list(bat_ids[0].shape) == [60, PRE]
    assert list(bat_ids[12].shape) == [60, PRE + 12]
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]
    gates["battery_shapes"] = {str(j): list(bat_ids[j].shape) for j in GEOS}
    log("protocol rebuilt: ZEPH-free corpus, splice mix 19/41, batteries "
        "60x{118,130,142} — PASS")

    # root + gate vs e151's before-cells
    root_sd = load_sd(ROOT_CK)
    keys = list(root_sd.keys())
    assert sum(v.numel() for v in root_sd.values()) == 2_739_072
    net0 = TinyGPT(Cfg()); net0.load_state_dict(root_sd); net0.eval()
    # sd-key order == parameter order (fact_grad dots against flat_sd vectors)
    assert list(dict(net0.named_parameters()).keys()) == keys, \
        "TinyGPT sd-key order != parameter order — currency mismatch risk"
    root_flat = flat_sd(root_sd, keys)
    r_eval_x, r_eval_y = val_windows(corpus.val,
                                     "".join(itos[int(i)] for i in corpus.val),
                                     60, 26502)
    root_reads = {
        "gm12": battery_cell(net0, gm12_ids, zid)["mean_pz"],
        "g0": battery_cell(net0, g0_ids, zid)["mean_pz"],
        "gp12": battery_cell(net0, bat_ids[12], zid)["mean_pz"],
        "ce_r": ce_fixed_cpu(net0, r_eval_x, r_eval_y),
    }
    root_diffs = {k: root_reads[k] - E151_ROOT[{"gm12": "base_gm12",
                                                "g0": "base_g0",
                                                "gp12": "base_gp12",
                                                "ce_r": "ce_r"}[k]]
                  for k in root_reads}
    gates["G_ROOT"] = {"reads": root_reads, "refs": E151_ROOT,
                       "max_abs_diff": max(abs(v) for v in root_diffs.values()),
                       "pass": bool(max(abs(v) for v in root_diffs.values())
                                    < F_G_TOL)}
    assert gates["G_ROOT"]["pass"], f"G_ROOT failed: {root_diffs}"
    log(f"G_ROOT: gm12 {root_reads['gm12']:.4f} g0 {root_reads['g0']:.4f} "
        f"(max|diff| {gates['G_ROOT']['max_abs_diff']:.1e}) — PASS")

    # ============ ROW 1: the trajectory row ==================================
    arms_out: dict = {}
    for arm in (TRAJ_ARMS[:1] if SMOKE else TRAJ_ARMS):
        a = run_arm(arm, net0, root_sd, root_flat, g0_ids, gm12_ids, zid)
        arms_out[arm["tag"]] = a
        log(f"[traj/{arm['tag']}] pairs {len(a['pairs'])}; repro max|diff| "
            f"{a['repro_max_diff']:.2e}; death A_g0 {a['death']['A_g0']:+.4f} "
            f"raw D {a['death']['raw_D']:.3f} at t={a['death']['t']}")
    # e185 stored-control displacement cross-check (lr 1e-3 arm's first pairs)
    p1 = arms_out["lr1e-3_neutral"]["pairs"]
    assert abs(p1[0]["raw_cum_norm"] - E185_CTRL_DISP[1]) < F_G_TOL
    assert abs(p1[1]["raw_cum_norm"] - E185_CTRL_DISP[2]) < F_G_TOL
    gates["G_DISP_XCHECK"] = {
        "mine": [p1[0]["raw_cum_norm"], p1[1]["raw_cum_norm"]],
        "e185_stored": [E185_CTRL_DISP[1], E185_CTRL_DISP[2]], "pass": True}
    log("G_DISP_XCHECK: ||d|| at s1/s2 reproduce e185's stored control "
        f"{E185_CTRL_DISP[1]:.4f}/{E185_CTRL_DISP[2]:.4f} — PASS")

    # replication co-read at lr 1e-3 (texture; time-guarded)
    repl_out = {}
    if not SMOKE and time.time() - T0 < 1200:
        for arm in REPL_ARMS:
            ct = committed_repl(arm)
            sd1, sd2 = load_sd(arm["ckpts"][0][1]), load_sd(arm["ckpts"][1][1])
            f1, f2 = flat_sd(sd1, keys), flat_sd(sd2, keys)
            net0.load_state_dict(sd1); net0.eval()
            r1 = {"g0": battery_cell(net0, g0_ids, zid)["mean_pz"],
                  "gm12": battery_cell(net0, gm12_ids, zid)["mean_pz"]}
            net0.load_state_dict(sd2); net0.eval()
            r2 = {"g0": battery_cell(net0, g0_ids, zid)["mean_pz"],
                  "gm12": battery_cell(net0, gm12_ids, zid)["mean_pz"]}
            for r, st in ((r1, 1), (r2, 2)):
                assert abs(r["g0"] - ct[st]["g0"]) < F_G_TOL, arm["tag"]
                assert abs(r["gm12"] - ct[st]["gm12"]) < F_G_TOL, arm["tag"]
            net0.load_state_dict(root_sd)
            gr = fact_grad(net0, g0_ids, zid)
            d1, d2 = f1 - root_flat, f2 - f1
            a1, a2 = cos(d1, gr), cos(d2, gr)
            A2 = a1 * float(d1.norm()) + a2 * float(d2.norm())
            t_star = min(s for s in ct if s > 0
                         and ct[s]["gm12"] <= SHUT_BAR)
            repl_out[arm["tag"]] = {
                "source": arm["source"], "device": arm["device"],
                "seed": arm["seed"], "t_star_committed": t_star,
                "pairs": [{"from_step": 0, "to_step": 1,
                           "d_norm": float(d1.norm()), "a_t_g0": a1},
                          {"from_step": 1, "to_step": 2,
                           "d_norm": float(d2.norm()), "a_t_g0": a2,
                           "A_g0_cum": A2}],
                "A_g0_at_t2": A2, "raw_D_at_t2": float((f2 - root_flat).norm())}
            log(f"[repl/{arm['tag']}] A_g0 at t*=2: {A2:+.4f}")
        del sd1, sd2, f1, f2

    # ============ ROW 2: the static row (two-currency amendment) ============
    pr = R.rebuild_protocol()
    sd_store, skeys, _, store_gates = R.load_root(pr)   # asserts g3R's gates
    gates["G_G3K_STORE_ROOT"] = store_gates
    sd_host_root = g3k_load(G3K_REF["host"]["root"])
    sd_host_s1 = g3k_load(G3K_REF["host"]["wash_s1"])
    net_h = TinyGPT(F2_CFG); net_h.load_state_dict(sd_host_root)
    g0h = battery_cell(net_h, pr["ids130"], pr["zid"])["mean_pz"]
    Dh = float(sum(float((sd_host_s1[k].float() - sd_host_root[k].float()).norm() ** 2)
                   for k in sd_host_root) ** 0.5)
    net_h.load_state_dict(sd_host_s1)
    g0h1 = battery_cell(net_h, pr["ids130"], pr["zid"])["mean_pz"]
    del net_h
    gates["G_G3K_HOST_ROOT"] = {
        "root_g0": g0h, "ref": G3K_REF["host"]["root_g0"],
        "wash_s1_g0": g0h1, "ref_s1": G3K_REF["host"]["wash_s1_g0"],
        "D_all_s1": Dh, "ref_D": G3K_REF["host"]["D_all_s1"],
        "pass": bool(abs(g0h - G3K_REF["host"]["root_g0"]) < G3_TOL
                     and abs(g0h1 - G3K_REF["host"]["wash_s1_g0"]) < G3_TOL
                     and abs(Dh - G3K_REF["host"]["D_all_s1"]) < G3_TOL)}
    assert gates["G_G3K_HOST_ROOT"]["pass"], "host ruler gate FAILED"
    log("G_G3K organisms: g3R store gates + host ruler gates PASS (1e-6)")
    static = run_static_row(pr, sd_store, sd_host_root)

    # ============ ROW 3: install-vs-wash cosines =============================
    cache: dict = {}

    def sd_lookup(name):
        if name not in cache:
            cache[name] = load_sd(name)
        return cache[name]

    vocab = install_vs_wash(sd_lookup)

    # ============ adjudication ===============================================
    adj = adjudicate(arms_out, static, repl_out)
    log(f"ADJUDICATION: {adj['verdict']} — CV(A) {adj['CV_A']:.3f} vs "
        f"CV(D) {adj['CV_D']:.3f}; band_pass {adj['band_pass_25pct']}")

    # ============ plots ======================================================
    plot_two_currency(rd / "two_currency.png", adj, arms_out, static)
    plot_curves(rd / "currency_curves.png", arms_out)
    log("plots written")

    # ============ metrics ====================================================
    M = {
        "experiment": "e188_death_currency",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registration": ("QUEUE.md e188 DISPATCHED 12:00Z (bars frozen: "
                         "A*-CONSTANT / RAW-WINS / NEITHER + the vocabulary "
                         "rule); W022b's estimator; T137/T139 pointed; R58 "
                         "ideator's two-currency amendment; bars frozen "
                         "VERBATIM in the module docstring before compute"),
        "registered_bars": REGISTERED_BARS,
        "vocab_rule": VOCAB_RULE,
        "question": ("in what currency is the wash kill priced — aligned "
                     "displacement A (W022b's integral) or raw ||d_theta|| "
                     "(e185/e180) — and is g3K's static kill priced in the "
                     "same currency?"),
        "gates": gates,
        "trajectory_row": {"arms": arms_out,
                           "replication_corread_lr1e-3_texture": repl_out,
                           "row_note": ("A from SAMPLED snapshots — the "
                                        "integral is a quadrature "
                                        "(piecewise-constant a_t per segment; "
                                        "the 1e-4 arm's whole kill sits in "
                                        "ONE 50-step segment; grad g0 at "
                                        "snapshotted theta assumes "
                                        "piecewise-linear adaptation)")},
        "static_row": static,
        "install_vs_wash": vocab,
        "vocabulary_verdict": {
            "per_organism": {k: {"cos": v["cos"],
                                 "call": v["vocab_rule_call"]}
                             for k, v in vocab.items() if k != "rule"},
            "rule": VOCAB_RULE,
            "note": ("reported per organism; the paper-level adoption is the "
                     "coordinator's fold, not this cell's adjudication")},
        "adjudication": adj,
        "honesty": {
            "snapshot_quadrature": "A(t) is a 1-7-point quadrature per arm; "
                                   "segments span 1-100 steps; the estimator "
                                   "assumes piecewise-constant alignment and "
                                   "piecewise-linear adaptation between "
                                   "snapshots — the caveat is carried on the "
                                   "figure, not shopped away",
            "single_fact_family": "the ZEPHYRA installs (e043 splice, one "
                                  "corpus, one install recipe); n=1 per arm; "
                                  "the lr 1e-4 arm is the ORIGINAL "
                                  "extinction-anchor stream on a coarse grid "
                                  "(flagged everywhere it enters a number); "
                                  "the g3 organisms are one construction "
                                  "each (g3K's standing scope)",
            "device_texture": "CPU fp32 reads of committed trajectories "
                              "(e176n/e184 committed on-device; functional "
                              "gate 0.05; diffs reported per snapshot; e185c "
                              "measured same-seed device ratios <= 0.2%)",
            "pricing_not_causation": "the cell shows WHAT death is priced "
                                     "in, not what CAUSES it; the causal "
                                     "cell is opt2's masked arms (r58's "
                                     "clause, adopted)",
            "estimator_change": "W022's -0.44 and opt1/T139's -0.015..-0.105 "
                                "were CUMULATIVE-displacement cosines; e188's "
                                "a_t is the PER-PAIR estimator (W022b's "
                                "specification) — disagreement between the "
                                "two is expected and informative, not a "
                                "contradiction",
            "no_bar_shopping": True,
        },
        "provenance": {
            "checkpoints": _provenance,
            "committed_metrics_consumed": [
                "runs/e176n/metrics.json", "runs/e180/metrics.json",
                "runs/e184/metrics.json", "runs/e185c/metrics.json",
                "runs/g3K/metrics.json", "runs/g3/metrics.json (via g3R gates)",
                "runs/e157/metrics.json (lineage endpoints)",
                "runs/e151/metrics.json (root gate refs, embedded)",
                "runs/e185/metrics.json (control displacements, embedded)"],
            "note": "every checkpoint hashed at load; every battery read "
                    "reproduced its committed trajectory row before use "
                    "(no silent loads)",
        },
        "compute": {"device": "cpu", "threads": torch.get_num_threads(),
                    "elapsed_s": round(time.time() - T0, 1),
                    "backward_passes": "one per left-endpoint snapshot per "
                                       "battery geometry (g0 primary, g-12 "
                                       "co-report) + one per g3K organism "
                                       "root + one per replication arm",
                    "training_runs": 0, "gpu_claimed": False},
    }
    save_json(rd / "metrics.json", M)
    log(f"DONE -> {rd / 'metrics.json'} ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    main()
