"""E231 — THE BALL-SIDE OVERLAP JOIN (scratch/e231_design.md, frozen; T208's
registered instrument, ripened and dispatched).

THE QUESTION (the design's frozen form): does each root's flat-phase retention
track how much of its fact direction lives inside its own wash's ACTIVE SPAN —
the fourth currency after SNR (dead, e225's cons pair), strength/margin (dead,
e229's -0.179) and local fragility (dead, T207) all died? THE BALL REMAINS.

WHAT IT BUILDS ON (module-imported, never retyped): lab/e225_one_currency.py
VERBATIM as a module — the seven roots' loading conventions (ROSTER, load_body,
G_ROOT battery gate), the u0 convention (the t=0 post-clip sign-ray on the
seed-10902 stream), and the span/band instrument with the CHUNKED fp64 Gram-SVD
(svd_basis). runs/e225/metrics.json (the joined table: retentions + multiples,
hard-bound at 1e-12) and runs/e229/metrics.json (the margin aggregates +
rho = -0.17857142857142858, hard-bound at 1e-12). THINKING T208 (the
three-dead-currency ledger this cell resolves) and T206 (the regime read — the
REGIME-PROXY bar's story).

WHAT IS NEW: the overlap instrument itself. (1) the PRIMARY overlap
||P_span u0|| / ||u0|| with P the projection onto the top-k span PCs of the
root's OWN 20-step unwalled AdamW history (seed 10902, e209's convention with
e225's chunked fp64 fix); (2) the CO-REPORT max_j |<u0, pc_j>| (the one-door
flavor); (3) the random-basis null control (same k); (4) the join at n=7
against the committed retentions with the multiple's +0.607 and the
aggregate's -0.179 quoted on the same page (the three-currency ledger on one
figure); (5) the second-seed sensitivity co-read (stream seed 10914 — the next
free draw of the 109xx wash family, registry-grep'd: 10903/10904 g2d,
10905/10906 g3R/e152r, 10907/10908 g1bR, 10909-10911 g2g2, 10912/10913 the
cons draws; 10914+ clean).

REGISTERED BARS (scratch/e231_design.md's letter, VERBATIM, frozen before
compute; adjudicate against exactly this; no bar shopping):
  - BALL-OWNS-THE-FLAT-PHASE — "rho(primary overlap, retention) >= 0.714 at
    n=7 AND the named breakers (g1d, take6) sit ON the curve — the fourth
    currency named: the flat phase reads the wash's books"
  - REGIME-PROXY — "the overlap correlates with retention only through
    formation regime: within fresh-formation rows it holds, within exotic rows
    it flatlines or inverts — T206's regime read survives as the organizer"
  - NEITHER — "no relation at the line with the breakers unexplained — the
    flat phase's currency stays unnamed; the honest bound"

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do not
move the bars):
  * primary overlap = ||P_top-k u0|| (u0 unit by construction), k at the
    scree knee by the MASS-PLATEAU rule: the smallest k with cumulative
    eigenvalue mass M(k) = sum_{i<=k} lambda_i / sum lambda >= 0.90; k
    recorded per root; the design's registered expectation "k in 2..6" is
    TESTED (fired iff every root's k sits in 2..6), never adjudicated.
  * co-report = max_{j<=k} |<u0, pc_j>| over the SAME top-k basis, with its
    argmax j and the door-ratio co_report/primary. Registered prediction 3
    (single-door) fires iff door-ratio >= 0.90 for EVERY root.
  * null control = 3 fresh random orthonormal k-bases (gaussian columns +
    fp64 Cholesky orthonormalization; per-row seeds 129xx, registry-grep'd
    clean), overlap of the SAME u0 with each; the chance level ~ sqrt(k/P).
  * second-seed sensitivity (design's honesty guard, cheap here — no grid
    walks in this cell): the full instrument re-run at stream seed 10914
    (u0' + span'); co-reported are the full-10914 overlap, the mixed read
    span@10914 x u0@10902 (the span side alone) AND cos(u0, u0') (the fact
    side alone). NEVER adjudicated.
  * breakers ON the curve = |rank_overlap - rank_retention| <= 1 for BOTH
    J3 (g1d) and J7 (take6), ranks ascending by value, ties disclosed.
  * regime split (T206's own sentence — the 10M takes sit inside its
    "fresh-formation families"; exotic = the named cons-axis/half-expression
    members): FRESH = {J1, J2, J6, J7}; EXOTIC = {J3, J4, J5}. The
    alternative split (J7 exotic by T172's lottery-break stamp) is a
    co-read, never adjudicated.
  * REGIME-PROXY fires iff NOT BALL AND rho_fresh >= 0.714 AND
    rho_exotic <= 0 (n=3: the observable "flatline-or-invert" versions are
    non-positive; a positive rho_exotic is disclosed if it occurs).
  * composite order: BALL-OWNS-THE-FLAT-PHASE / REGIME-PROXY / NEITHER
    (frozen before compute; the first two mutually exclusive by
    construction).
  * the discriminating cell (prediction 2, verbatim from the design): g1d
    vs g1f — under BALL both exotic rows' retentions are set by overlap
    (g1d 1.327 needs the table's TOP overlap); under REGIME exotic rows
    ignore the overlap (g1d high-retention at ANY overlap). The cell's
    observable: within the exotic triple {J3, J4, J5}, BALL predicts
    rho_exotic = +1 with g1d at the top overlap; REGIME predicts
    rho_exotic <= 0.

REGISTERED PREDICTIONS (the design's letter, tested verbatim, no retrofit):
  1. T208's: overall rho > 0.714 with g1d HIGH overlap (the half-expressed
     fact lives where the wash already points — its 1.327 retention
     recaptured) and g1f's overlap ABOVE g1e's (the cons pair separating in
     the ball's favor, extending the pair's decision-order match into the
     ball's geometry). FIRES iff (rho > 0.714) AND (J3 at the table's TOP
     overlap rank) AND (overlap(J5) > overlap(J4)).
  2. The discriminating cell (above).
  3. The co-report's shape: if overlap concentrates in ONE wash PC for every
     root, the flat phase has a single-door geometry. FIRES iff door-ratio
     >= 0.90 for every root.

HONESTY GUARDS (the design's, made numerical — frozen before compute):
  - THE ARITHMETIC-IDENTITY DISCLOSURE (the design's "u0 and the span share
    the seed-10902 stream — not independent objects"): u0 IS the sign of the
    wash's first step's gradient, and AdamW's FIRST displacement is
    -lr*(sign(g) + wd*theta0) — so history segment 1 is (minus) u0 to
    ~1e-4 and the FULL-20 span contains u0 BY CONSTRUCTION
    (||P_full u0|| recorded per row as the disclosure's number, expected
    ~0.9999). The informative quantity is the TOP-k mass — how much of u0
    lives in the wash's DOMINANT subspace — and the null control fixes the
    chance level (~sqrt(k/P) ~ 1e-3). Disclosed on the figure.
  - the span basis is n=1 per root (one 20-step history, seed 10902) — the
    2-3x band lottery applies to the span too; the seed-10914 co-read
    bounds it.
  - this is neither the fact-edge object nor the argmax margin (e208/T207
    restated); it is a GEOMETRY object (subspace membership).
  - nothing guaranteed; n=7; the GRADED discipline of the e225/e229 family
    applies; J3 (g1d) remains OBSERVED-UNADJUDICATED (T186's stamp, carried).

COMPUTE ENVELOPE (dispatch): CPU-ONLY desk+eval (CUDA_VISIBLE_DEVICES=-1
forced before torch; the GPU is never claimed — other cells own it), torch
threads 4, load-check recorded between organisms (soft pause > 70%), no
perturb-and-eval grid walks in this cell (pure geometry: the only behavioral
read is the G_ROOT battery gate per organism), progressive metrics.json
writes after every phase and every organism, n=1 per organism, decisive.

Outputs: runs/e231/{metrics.json (PROGRESSIVE), e231_ball_overlap.png}.
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

Run:  cd lab && python e231_ball_overlap.py    (E231_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e225's convention)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(4)                              # the e185/e209/e225 reduction order

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, run_dir, save_json # noqa: E402

import e043_install as E43                            # noqa: E402 (REPO, find_occ, jsonable)

import e225_one_currency as E25                       # noqa: E402 (THE PORTED INSTRUMENT)
from e225_one_currency import (                       # noqa: E402
    ANCHOR_FORBIDDEN, ANCH_BS, BLOCK, CKPT_DIR, E185_XHASH_1, GEOS, HOSTS,
    LR_ADAMW, POST_CAP, PRE, RAND_BS, ROSTER, RULER_J, WASH_HIST_STEPS,
    WASH_SEED, battery_cell, ce_fixed_cpu, cpu_load_probe, flat_params,
    load_body, soft_load_pause, svd_basis, val_windows,
)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E231_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e231 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------------ frozen constants
SECOND_STREAM_SEED = 10914        # the next free 109xx wash-family draw (registry-grep'd)
KNEE_MASS = 0.90                  # the mass-plateau rule (frozen in the docstring)
DOOR_RATIO_BAR = 0.90             # prediction 3's single-door line
NULL_DRAWS = 3
NULL_SEED_BASE = 12900            # 129xx block registry-grep'd clean
RHO_BAR = 0.714                   # the one-sided 5% Spearman line at n=7 (frozen)

FRESH_ROWS = ["J1", "J2", "J6", "J7"]          # T206's fresh-formation families
EXOTIC_ROWS = ["J3", "J4", "J5"]               # g1d half-expression + g1e/g1f cons draws
BREAKERS = ["J3", "J7"]                        # the design's named breakers: g1d, take6
CONS_PAIR = ["J4", "J5"]                       # g1e / g1f

# THE COMMITTED LEDGER, hard-bound at 1e-12 (runs/e225/metrics.json joined_table
# + runs/e229/metrics.json join_table; G_PARENTS asserts every one):
BIND = {
    "J1": {"retention": 0.9979692766675983, "multiple": 3.7206885127780804,
           "aggregate": 1.3318806921021653},
    "J2": {"retention": 1.0263911969648605, "multiple": 1.2161096471128718,
           "aggregate": 0.9794073282786698},
    "J3": {"retention": 1.3266428034732003, "multiple": 1.3789356098383065,
           "aggregate": 0.26796058555057695},
    "J4": {"retention": 0.6153917590189493, "multiple": 0.5995813346448845,
           "aggregate": 0.6652406915887169},
    "J5": {"retention": 0.9317064573129114, "multiple": 0.5617260108399429,
           "aggregate": 1.7539447693197137},
    "J6": {"retention": 0.8949634049389881, "multiple": 0.5433506780298486,
           "aggregate": 0.9008607010514801},
    "J7": {"retention": 0.7897980962997516, "multiple": 1.0231918306111651,
           "aggregate": 1.3683092125843754},
}
RHO_MULTIPLE_COMMITTED = 0.6071428571428571    # e225's rank layer (quoted on the page)
RHO_AGGREGATE_COMMITTED = -0.17857142857142858 # e229's primary (quoted on the page)

# u0 bit-identity gates: the committed e225 sign-ray md5s (J1 binds to the
# e131-root bridge cell B1 — the same organism, the fresh read):
U0_MD5_EXPECT = {
    "J1": "aac6c6d643e327939f680148772dd180",   # = e225 cells.B1.directions.sign_ray_md5
    "J2": "4e97cac2975bf11a030a8eb2abe0cb05",
    "J3": "8869069a821636cdd9631677595eb2a9",
    "J4": "644fe6333239b6d6535c403b6343eb0c",
    "J5": "6072858bf2a0cb85cf351d960777c3d1",
    "J6": "7d2c738d5847fc038776a1c696b9f75a",
    "J7": "d60e052ee11869b0aea41a8f48d0953f",
}

REGISTERED_BARS = {
    "BALL_OWNS_THE_FLAT_PHASE": "BALL-OWNS-THE-FLAT-PHASE — \"rho(primary overlap, "
        "retention) >= 0.714 at n=7 AND the named breakers (g1d, take6) sit ON the "
        "curve — the fourth currency named: the flat phase reads the wash's books\"",
    "REGIME_PROXY": "REGIME-PROXY — \"the overlap correlates with retention only "
        "through formation regime: within fresh-formation rows it holds, within "
        "exotic rows it flatlines or inverts — T206's regime read survives as the "
        "organizer\"",
    "NEITHER": "NEITHER — \"no relation at the line with the breakers unexplained — "
        "the flat phase's currency stays unnamed; the honest bound\"",
    "operationalizations": "primary overlap = ||P_top-k u0||, k = the scree knee by "
        "the mass-plateau rule (smallest k with cumulative eigenvalue mass >= 0.90, "
        "k recorded, the k-in-2..6 expectation tested never adjudicated); co-report "
        "= max_{j<=k} |<u0, pc_j>| with argmax and door-ratio (prediction 3: "
        "door-ratio >= 0.90 for every root); null = 3 random orthonormal k-bases "
        "(129xx seeds) against the SAME u0; second-seed = the full instrument at "
        "stream 10914 + the mixed span@10914 x u0@10902 read (co-reads only); "
        "breakers ON the curve = |rank_ov - rank_ret| <= 1 for BOTH J3 and J7; "
        "regime split = T206's own sentence (FRESH {J1,J2,J6,J7} / EXOTIC "
        "{J3,J4,J5}; the J7-exotic alternative a co-read); REGIME-PROXY iff NOT "
        "BALL AND rho_fresh >= 0.714 AND rho_exotic <= 0; composite order BALL / "
        "REGIME-PROXY / NEITHER, the first two mutually exclusive by construction",
    "registration": "scratch/e231_design.md (frozen on paper before dispatch) + "
        "this docstring (frozen before compute). Adjudicate against exactly this; "
        "no bar shopping.",
}

deviations: list[str] = [
    "MODULE-IMPORT OF THE PARENT INSTRUMENT: lab/e225_one_currency.py is imported "
    "as a module (its loading conventions, u0 construction, stream recipe, and the "
    "CHUNKED fp64 Gram-SVD svd_basis are used VERBATIM, never retyped). Only the "
    "P0a protocol rebuild (corpus/splice/battery/anchor bank) is ported "
    "line-verbatim, because e225 keeps it inside main() — asserted against the "
    "same committed anchors (G_NAMEFREE/G_SPLICE/G_BATTERY/G_ANCHOR).",
    "NO GRID WALKS: this cell is pure geometry — the band instrument's "
    "perturb-and-eval walks are not needed (no edge, no band, no kill-D reads); "
    "the only behavioral read is the one G_ROOT battery cell per organism. The "
    "wash history's per-step ruler evals are likewise skipped (the span needs "
    "only the displacements; the fact-dies-at-step-1 co-read is already "
    "committed in e225's cells).",
    "u0 AND THE SPAN ARE NOT INDEPENDENT OBJECTS (the design's guard, quantified): "
    "u0 = sign of the 10902 stream's first wash-step gradient, and AdamW's first "
    "displacement is -lr*(sign(g) + wd*theta0), so history segment 1 IS (minus) "
    "u0 to ~1e-4 and ||P_full u0|| ~ 1 BY CONSTRUCTION. The informative read is "
    "the TOP-k mass (the wash's dominant subspace); the null control fixes the "
    "chance level. Recorded per row; drawn on the figure.",
    "SECOND STREAM SEED = 10914: the 109xx wash family is occupied through 10913 "
    "(10903/10904 g2d, 10905/10906 g3R/e152r, 10907/10908 g1bR, 10909-10911 "
    "g2g2, 10912/10913 the cons draws); 10914 is the next free draw, "
    "registry-grep'd (the only 10914 hits in lab/ are float literals).",
    "The null control orthonormalizes gaussian columns via fp64 Cholesky on the "
    "small k x k Gram (z' S^-1 z) instead of a P x k QR — lighter at 10M and "
    "numerically identical for a Haar-random k-basis.",
    "No new checkpoints; every fresh vector is bit-deterministic from (the "
    "pristine root, the gated stream, the frozen seeds); u md5s recorded; the "
    "PRIMARY u0 md5s are gated BIT-IDENTICAL to e225's committed cells (G_U0).",
    "Smoke mode trims: 2 organisms (J4 + J6), 1 null draw; nothing adjudicated "
    "(verdict stamped SMOKE).",
]


# ------------------------------------------------------------------ P0a (ported verbatim)

def protocol_rebuild():
    """e225's P0a block, line-verbatim (module main-scope, not importable)."""
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
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
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
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in GEOS},
        "pass": bool(list(bat_ids[-12].shape) == [60, PRE - 12]
                     and list(bat_ids[0].shape) == [60, PRE]
                     and list(bat_ids[12].shape) == [60, PRE + 12]),
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    ruler_ids = bat_ids[RULER_J]

    arng = _random.Random(E170_ANCHOR_SEED := E25.E170_ANCHOR_SEED)
    n_starts, tries = [], 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ANCHOR_FORBIDDEN):
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    G_ANCHOR = {"starts": n_starts, "tries": tries,
                "bank_starts_match_e185_stored": bool(
                    n_starts == E25.E170_BANK_STARTS)}
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"
    r_eval = val_windows(val_ids, val_text, 60, E25.R_EVAL_SEED)
    return {"zid": zid, "train_ids": train_ids, "ruler_ids": ruler_ids,
            "anchor_neutral": anchor_neutral, "r_eval": r_eval,
            "gates": {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                      "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR}}


# ------------------------------------------------------------------ the instrument

def scree_knee(sv: torch.Tensor, mass_bar: float = KNEE_MASS):
    """The mass-plateau knee: smallest k with cumulative eigen-mass >= mass_bar."""
    lam = (sv.double() ** 2)
    total = float(lam.sum())
    if total <= 0:
        return {"k": 0, "mass_curve": [], "total": 0.0}
    cum, run, k = [], 0.0, None
    for i, v in enumerate(lam.tolist(), start=1):
        run += v
        cum.append(run / total)
        if k is None and cum[-1] >= mass_bar:
            k = i
    return {"k": k if k is not None else len(lam), "mass_curve": cum,
            "total": total}


def stream_batch(generator, train_ids, anchor_neutral):
    """e225's stream recipe: one wash batch = 16 anchor draws + 16 random."""
    aj = torch.randint(16, (ANCH_BS,), generator=generator)
    rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,), generator=generator)
    anc = anchor_neutral[aj]
    rnd = torch.stack([train_ids[q: q + BLOCK] for q in rj])
    x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
    y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
    return x, y


def null_overlap(u0: torch.Tensor, k: int, seed: int) -> dict:
    """Overlap of u0 with a random orthonormal k-basis (gaussian + Cholesky)."""
    P = u0.numel()
    g = torch.Generator().manual_seed(seed)
    Gm = torch.randn(P, k, generator=g)                      # (P, k) fp32
    z = torch.zeros(k, dtype=torch.float64)
    S = torch.zeros(k, k, dtype=torch.float64)
    chunk = max(1, int(40e6 // max(k, 1)))
    u0d = u0.double()
    for s in range(0, P, chunk):
        Gc = Gm[s:s + chunk].double()
        z += Gc.T @ u0d[s:s + chunk]
        S += Gc.T @ Gc
    del Gm, u0d
    L = torch.linalg.cholesky(S)
    w = torch.linalg.solve_triangular(L, z.unsqueeze(1), upper=False).squeeze(1)
    return {"overlap": float(torch.norm(w)),
            "expected_level": float((k / P) ** 0.5), "seed": seed, "k": k, "P": P}


def organism_cell(spec, proto, metrics, write_partial):
    """The full overlap instrument at one pristine root, both stream seeds."""
    rid = spec["id"]
    cfg = Cfg(**spec["cfg"]) if spec["cfg"] else Cfg()
    net, sd, meta = load_body(CKPT_DIR / spec["ckpt"], cfg)
    n_par = sum(p.numel() for p in net.parameters())
    theta0 = flat_params(net)
    root_read = battery_cell(net, proto["ruler_ids"], proto["zid"])["mean_pz"]
    ce_r = ce_fixed_cpu(net, *proto["r_eval"])
    rdev = abs(root_read - spec["committed_root_gm12"])
    G_ROOT = {
        "checkpoint": f"runs/checkpoints/{spec['ckpt']}",
        "meta": {k: meta.get(k) for k in
                 ("experiment", "desc", "cons_seed", "root_gm12", "params",
                  "movement_rms", "steps", "lr", "base")},
        "n_params": n_par, "expected_params": spec["n_params"],
        "battery_read_measured": root_read,
        "battery_read_committed": spec["committed_root_gm12"],
        "abs_diff": rdev, "tol": E25.G_READ_TOL, "ce_r": ce_r,
        "flat_md5": hashlib.md5(theta0.numpy().tobytes()).hexdigest(),
        "pass": bool(n_par == spec["n_params"] and rdev < E25.G_READ_TOL),
    }
    log(f"G_ROOT[{rid}]: {n_par} params; battery {root_read:.6f} vs committed "
        f"{spec['committed_root_gm12']:.6f} (|d| {rdev:.1e}): "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    if not G_ROOT["pass"]:
        raise RuntimeError(f"root gate FAILED at {rid}")
    cell = {"id": rid, "ckpt": spec["ckpt"], "scale": spec["scale"],
            "organism": spec["organism"], "G_ROOT": G_ROOT}
    metrics.setdefault("cells", {})[rid] = cell
    write_partial(f"P1[{rid}] G_ROOT PASSED")

    def side(seed: int, is_primary: bool, u0_other=None):
        """One stream seed's full read: u0, 20-step history, span, overlaps."""
        # ---- u0 (e225's construction verbatim on this stream) ----------------
        gg = torch.Generator().manual_seed(seed)
        x1, y1 = stream_batch(gg, proto["train_ids"], proto["anchor_neutral"])
        x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()
        if seed == WASH_SEED:
            assert x1_md5 == E185_XHASH_1, "primary stream drift"
        gnet = copy.deepcopy(net)
        gnet.train()
        gnet.zero_grad(set_to_none=True)
        logits_g, _ = gnet(x1)
        F.cross_entropy(logits_g.reshape(-1, logits_g.shape[-1]),
                        y1.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(gnet.parameters(), 1.0)
        g0 = torch.cat([p.grad.detach().reshape(-1)
                        for p in gnet.parameters()]).clone()   # post-clip
        gnet.zero_grad(set_to_none=True)
        del gnet, logits_g
        u_g = (g0 / torch.norm(g0)).clone()
        u0 = torch.sign(g0)
        u0 = (u0 / torch.norm(u0)).clone()
        u0_md5 = hashlib.md5(u0.numpy().tobytes()).hexdigest()
        out = {"stream_seed": seed, "x1_md5": x1_md5,
               "g_ray_md5": hashlib.md5(u_g.numpy().tobytes()).hexdigest(),
               "sign_ray_md5": u0_md5,
               "cos_g_sign": E25.cos64(u_g, u0)}
        if is_primary:
            ok = u0_md5 == U0_MD5_EXPECT[rid]
            out["u0_bit_identical_to_e225"] = bool(ok)
            assert ok, f"G_U0 FAILED at {rid}: {u0_md5}"
            log(f"  G_U0[{rid}]: sign-ray md5 BIT-IDENTICAL to e225's "
                f"committed cell ({u0_md5[:12]}...)")

        # ---- the 20-step unwalled wash history (e225 verbatim) ---------------
        wh_gen = torch.Generator().manual_seed(seed)
        wh_net = copy.deepcopy(net)
        wh_net.train()
        wh_opt = torch.optim.AdamW(wh_net.parameters(), lr=LR_ADAMW,
                                   betas=(0.9, 0.95), weight_decay=0.1)
        wh_theta = theta0.clone()
        hist_disp = []
        for s_wh in range(1, WASH_HIST_STEPS + 1):
            x, y = stream_batch(wh_gen, proto["train_ids"],
                                proto["anchor_neutral"])
            if s_wh == 1 and seed == WASH_SEED:
                assert hashlib.md5(x.contiguous().numpy().tobytes()).hexdigest() \
                    == E185_XHASH_1, "history stream drift"
            logits_w, _ = wh_net(x)
            lw = F.cross_entropy(logits_w.reshape(-1, logits_w.shape[-1]),
                                 y.reshape(-1))
            wh_opt.zero_grad(set_to_none=True)
            lw.backward()
            torch.nn.utils.clip_grad_norm_(wh_net.parameters(), 1.0)
            wh_opt.step()
            th_new = flat_params(wh_net)
            hist_disp.append(th_new - wh_theta)
            wh_theta = th_new
        wh_net.eval()
        del wh_net, wh_opt, logits_w
        H = torch.stack(hist_disp)
        seg_norms = [float(torch.norm(d)) for d in hist_disp]

        # ---- the span (Gram-SVD, chunked fp64 — e225's svd_basis) ------------
        basis = svd_basis(H)
        del H
        Vp = basis["Vp"].contiguous()
        knee = scree_knee(basis["sv"])
        k = knee["k"]
        # coefficients <u0, pc_j> in fp64 (per-row dots — chunk-light at 10M):
        u0d, ugd = u0.double(), u_g.double()
        coefs_u0 = torch.tensor([float(torch.dot(Vp[i].double(), u0d))
                                 for i in range(Vp.shape[0])])
        coefs_ug = torch.tensor([float(torch.dot(Vp[i].double(), ugd))
                                 for i in range(Vp.shape[0])])
        coefs_mix = None
        if u0_other is not None:
            uod = u0_other.double()
            coefs_mix = torch.tensor([float(torch.dot(Vp[i].double(), uod))
                                      for i in range(Vp.shape[0])])
            del uod
        del Vp, basis["Vp"], u0d, ugd
        d1 = hist_disp[0] / hist_disp[0].norm()
        out.update({
            "span": {"n_segments": WASH_HIST_STEPS,
                     "rank_eff": basis["rank_eff"],
                     "participation_ratio": basis["pr"], "cond": basis["cond"],
                     "sv": [float(v) for v in basis["sv"]],
                     "seg1_L2": seg_norms[0],
                     "seg_norms_minmax": [min(seg_norms), max(seg_norms)]},
            "knee": {"k": k, "mass_bar": KNEE_MASS,
                     "mass_at_k": knee["mass_curve"][k - 1],
                     "mass_at_k_minus_1": (knee["mass_curve"][k - 2]
                                           if k >= 2 else 0.0),
                     "mass_curve": knee["mass_curve"],
                     "k_in_design_window_2_6": bool(2 <= k <= 6)},
            "identity_disclosure": {
                "cos_u0_seg1": E25.cos64(u0, d1),
                "overlap_full_rank20": float(torch.norm(coefs_u0)),
                "note": "segment 1 = -lr(sign(g)+wd*theta) IS (minus) u0 to "
                        "~1e-4 — the full-20 span contains u0 BY CONSTRUCTION; "
                        "the informative read is the top-k mass"},
            "overlap": {
                "primary": float(torch.norm(coefs_u0[:k])),
                "co_report_max_pc": float(torch.abs(coefs_u0[:k]).max()),
                "co_report_argmax_pc": int(torch.abs(coefs_u0[:k]).argmax()) + 1,
                "door_ratio": float(torch.abs(coefs_u0[:k]).max()
                                    / torch.norm(coefs_u0[:k])),
                "overlap_vs_k_1_to_8": [float(torch.norm(coefs_u0[:j]))
                                        for j in range(1, min(9, len(coefs_u0) + 1))],
                "coefs_u0_all_pcs": [float(c) for c in coefs_u0],
                "coefs_ug_all_pcs": [float(c) for c in coefs_ug],
                "ug_overlap_topk": float(torch.norm(coefs_ug[:k])),
            },
        })
        if coefs_mix is not None:
            out["overlap_mixed_u0_10902_on_this_span"] = {
                "topk": float(torch.norm(coefs_mix[:k])),
                "full20": float(torch.norm(coefs_mix)),
                "note": "the span side alone (this seed's span vs the PRIMARY "
                        "u0@10902); co-read, never adjudicated"}
        del hist_disp, d1
        log(f"  [{rid} s{seed}] k={k} (mass {knee['mass_curve'][k-1]:.3f}) | "
            f"PRIMARY overlap {out['overlap']['primary']:.5f} | door "
            f"{out['overlap']['door_ratio']:.3f} "
            f"(pc{out['overlap']['co_report_argmax_pc']}) | "
            f"cos(u0,seg1) {out['identity_disclosure']['cos_u0_seg1']:.5f} | "
            f"full20 {out['identity_disclosure']['overlap_full_rank20']:.6f}")
        return out, u0, u_g

    prim, u0, u_g = side(WASH_SEED, is_primary=True)
    cell["primary_stream"] = prim
    write_partial(f"P1[{rid}] PRIMARY span read (k={prim['knee']['k']}, overlap "
                  f"{prim['overlap']['primary']:.5f})")

    # ---- the null control (same k, random orthonormal bases) -----------------
    idx = [r["id"] for r in ROSTER].index(rid)
    n_null = 1 if SMOKE else NULL_DRAWS
    nulls = [null_overlap(u0, prim["knee"]["k"], NULL_SEED_BASE + 10 * idx + j)
             for j in range(n_null)]
    cell["null_control"] = {
        "draws": nulls,
        "median": sorted(n["overlap"] for n in nulls)[len(nulls) // 2],
        "note": "3 random orthonormal k-bases vs the SAME u0; chance level "
                "~ sqrt(k/P); the observed overlaps must clear it by orders "
                "of magnitude to mean anything",
    }
    log(f"  [{rid}] null overlaps "
        + ", ".join(f"{n['overlap']:.2e}" for n in nulls)
        + f" (expected ~{(prim['knee']['k'] / n_par) ** 0.5:.1e})")
    write_partial(f"P1[{rid}] null control done")

    # ---- the second-seed sensitivity (co-read, never adjudicated) ------------
    sec, u0s, _ = side(SECOND_STREAM_SEED, is_primary=False, u0_other=u0)
    cell["second_stream"] = sec
    cell["second_stream_coreads"] = {
        "cos_u0_10902_vs_10914": E25.cos64(u0, u0s),
        "overlap_full_10914": sec["overlap"]["primary"],
        "overlap_mixed_span10914_vs_u0_10902":
            sec["overlap_mixed_u0_10902_on_this_span"]["topk"],
        "note": "the full instrument re-run at 10914 (u0' + span'), the mixed "
                "span-side read, and the fact side's cos(u0,u0'); the span "
                "lottery and the fact lottery both visible; NEVER adjudicated",
    }
    log(f"  [{rid}] second stream {SECOND_STREAM_SEED}: overlap "
        f"{sec['overlap']['primary']:.5f} (k={sec['knee']['k']}), mixed "
        f"{sec['overlap_mixed_u0_10902_on_this_span']['topk']:.5f}, "
        f"cos(u0,u0') {cell['second_stream_coreads']['cos_u0_10902_vs_10914']:.4f}")
    write_partial(f"P1[{rid}] COMPLETE (primary k={prim['knee']['k']} overlap "
                  f"{prim['overlap']['primary']:.5f}; second-seed "
                  f"{sec['overlap']['primary']:.5f})")
    del net, u0, u_g, u0s
    return cell


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e231_smoke" if SMOKE else "e231")
    metrics: dict = {
        "experiment": "e231_ball_overlap",
        "date": common.now_iso(),
        "status": "PARTIAL (progressive)",
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "smoke": SMOKE,
        "envelope": {
            "device": "CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced; GPU never "
                      "claimed — other cells own it)",
            "torch_threads": torch.get_num_threads(),
            "phases": "P0 gates -> P1 per-organism overlap cells (both stream "
                      "seeds + null) -> P2 the joined table -> P3 adjudication "
                      "-> P4 figure; progressive writes",
            "roster": [r["id"] + ":" + r["ckpt"] for r in ROSTER],
        },
        "deviations": deviations,
    }

    def write_partial(note: str):
        metrics["date"] = common.now_iso()
        metrics["phase"] = note
        save_json(rd / "metrics.json", E43.jsonable(metrics))
        log(f"WROTE partial metrics ({note})")

    load0 = cpu_load_probe()
    metrics["envelope"]["cpu_load_pct_at_launch"] = load0
    log(f"E231 THE BALL-SIDE OVERLAP JOIN (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), no grid walks "
        f"(pure geometry), progressive writes, n=1 per organism, 7 rows")

    # ================= P0a: protocol rebuild (e225's gates) ====================
    proto = protocol_rebuild()
    metrics["gates"] = proto["gates"]
    log("P0a: protocol gates PASS (namefree / splice 19+41 / battery shapes / "
        "e170 bank bit-match)")
    write_partial("P0a protocol gates PASSED")

    # ================= P0b: parents hard-bound (1e-12) ========================
    RUNS = E43.REPO / "runs"
    PARENT_FILES = {"e225": RUNS / "e225" / "metrics.json",
                    "e229": RUNS / "e229" / "metrics.json"}

    def md5of(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    parents = {k: json.loads(p.read_text(encoding="utf-8"))
               for k, p in PARENT_FILES.items()}
    parents_md5 = {k: md5of(p) for k, p in PARENT_FILES.items()}
    e225_rows = {r["id"]: r for r in parents["e225"]["joined_table"]}
    e229_rows = {r["id"]: r for r in parents["e229"]["join_table"]}
    for rid, b in BIND.items():
        assert abs(e225_rows[rid]["retention_flat_min"] - b["retention"]) < 1e-12, rid
        assert abs(e225_rows[rid]["multiple"] - b["multiple"]) < 1e-12, rid
        assert abs(e229_rows[rid]["aggregate_median"] - b["aggregate"]) < 1e-12, rid
        assert abs(e229_rows[rid]["retention"] - b["retention"]) < 1e-12, rid
    assert abs(parents["e225"]["adjudication"]["rank_layer"]["spearman"]
               - RHO_MULTIPLE_COMMITTED) < 1e-12
    assert abs(parents["e229"]["join_statistics"]["primary"]
               ["spearman_aggregate_vs_retention"]
               - RHO_AGGREGATE_COMMITTED) < 1e-12
    e225_cells = parents["e225"]["cells"]
    for rid, h in U0_MD5_EXPECT.items():
        src = "B1" if rid == "J1" else rid
        assert e225_cells[src]["directions"]["sign_ray_md5"] == h, rid
    G_PARENTS = {
        "files_md5": parents_md5,
        "hardbound": {
            "retentions_multiples": "runs/e225/metrics.json joined_table (7 rows)",
            "aggregates_rho": "runs/e229/metrics.json join_table + "
                              "join_statistics.primary",
            "rho_multiple_e225": RHO_MULTIPLE_COMMITTED,
            "rho_aggregate_e229": RHO_AGGREGATE_COMMITTED,
            "u0_md5s": "e225 cells[].directions.sign_ray_md5 (J1 via the B1 "
                       "bridge cell — the same organism)",
        },
        "pass": True,
    }
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log("P0b: parents hard-bound at 1e-12 (e225 retentions/multiples/rho + "
        "e229 aggregates/rho + the seven committed sign-ray md5s)")
    write_partial("P0b G_PARENTS PASSED (the ledger side closed data)")

    # ================= P1: the per-organism overlap cells ======================
    roster_run = ROSTER if not SMOKE else [ROSTER[3], ROSTER[5]]
    for spec in roster_run:
        log("=" * 78)
        log(f"P1[{spec['id']}] {spec['ckpt']} ({spec['scale']})")
        soft_load_pause(spec["id"])
        t_cell = time.time()
        organism_cell(spec, proto, metrics, write_partial)
        log(f"P1[{spec['id']}] cell time {time.time() - t_cell:.1f}s")
        metrics["envelope"][f"cpu_load_pct_after_{spec['id']}"] = cpu_load_probe()
        time.sleep(10 if spec["scale"] == "2.74M" else 30)   # cooldown pause
    write_partial("P1 COMPLETE (all overlap cells, both seeds + nulls)")

    # ================= P2: the joined table ====================================
    rows = []
    for spec in roster_run:
        rid = spec["id"]
        c = metrics["cells"][rid]
        prim, sec = c["primary_stream"], c["second_stream"]
        b = BIND[rid]
        rows.append({
            "id": rid, "scale": spec["scale"], "organism": spec["organism"],
            "ckpt": f"runs/checkpoints/{spec['ckpt']}",
            "retention_flat_min": b["retention"],       # hard-bound (e225)
            "multiple": b["multiple"],                  # hard-bound (e225)
            "aggregate_median": b["aggregate"],         # hard-bound (e229)
            "overlap_primary": prim["overlap"]["primary"],
            "overlap_co_report": prim["overlap"]["co_report_max_pc"],
            "door_ratio": prim["overlap"]["door_ratio"],
            "door_pc": prim["overlap"]["co_report_argmax_pc"],
            "k": prim["knee"]["k"],
            "mass_at_k": prim["knee"]["mass_at_k"],
            "overlap_full_rank20": prim["identity_disclosure"]["overlap_full_rank20"],
            "cos_u0_seg1": prim["identity_disclosure"]["cos_u0_seg1"],
            "null_median": c["null_control"]["median"],
            "null_draws": [d["overlap"] for d in c["null_control"]["draws"]],
            "second_seed_overlap": sec["overlap"]["primary"],
            "second_seed_k": sec["knee"]["k"],
            "second_seed_mixed": sec["overlap_mixed_u0_10902_on_this_span"]["topk"],
            "cos_u0_seed_sensitivity":
                c["second_stream_coreads"]["cos_u0_10902_vs_10914"],
            "regime": "fresh" if rid in FRESH_ROWS else "exotic",
            "is_breaker": rid in BREAKERS, "is_cons_pair": rid in CONS_PAIR,
            "u0_bit_identical_to_e225": prim["u0_bit_identical_to_e225"],
            "row_notes": spec["row_notes"],
        })
    metrics["joined_table"] = rows
    write_partial(f"P2 the joined table assembled (n={len(rows)} rows)")

    # ================= P3: the adjudication (frozen bars) ======================
    n = len(rows)
    xs = [r["overlap_primary"] for r in rows]
    ys = [r["retention_flat_min"] for r in rows]
    rho = E25.spearman(xs, ys) if n >= 3 else float("nan")

    def rank_asc(vals):
        order_ = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0] * len(vals)
        for pos, i in enumerate(order_):
            rk[i] = pos + 1.0
        return rk

    rk_ov, rk_ret = rank_asc(xs), rank_asc(ys)
    ties_x = len(xs) != len(set(xs))
    ties_y = len(ys) != len(set(ys))
    for r, a, b_ in zip(rows, rk_ov, rk_ret):
        r["rank_overlap"], r["rank_retention"] = a, b_
        r["rank_residual"] = a - b_
    on_curve = {r["id"]: abs(r["rank_residual"]) <= 1 for r in rows}
    fr = [r for r in rows if r["id"] in FRESH_ROWS]
    ex = [r for r in rows if r["id"] in EXOTIC_ROWS]
    rho_fresh = (E25.spearman([r["overlap_primary"] for r in fr],
                              [r["retention_flat_min"] for r in fr])
                 if len(fr) >= 3 else float("nan"))
    rho_exotic = (E25.spearman([r["overlap_primary"] for r in ex],
                               [r["retention_flat_min"] for r in ex])
                  if len(ex) >= 3 else float("nan"))
    # co-read: the alternative regime split (J7 exotic by T172's stamp)
    fr_alt = [r for r in rows if r["id"] in ("J1", "J2", "J6")]
    ex_alt = [r for r in rows if r["id"] in ("J3", "J4", "J5", "J7")]
    rho_fresh_alt = (E25.spearman([r["overlap_primary"] for r in fr_alt],
                                  [r["retention_flat_min"] for r in fr_alt])
                     if len(fr_alt) >= 3 else float("nan"))
    rho_exotic_alt = (E25.spearman([r["overlap_primary"] for r in ex_alt],
                                   [r["retention_flat_min"] for r in ex_alt])
                      if len(ex_alt) >= 3 else float("nan"))
    rho_co = (E25.spearman([r["overlap_co_report"] for r in rows], ys)
              if n >= 3 else float("nan"))

    ball_fires = bool(n == 7 and rho >= RHO_BAR
                      and all(on_curve[b_] for b_ in BREAKERS))
    regime_fires = bool(not ball_fires and rho_fresh >= RHO_BAR
                        and rho_exotic <= 0)
    neither_fires = not ball_fires and not regime_fires
    assert not (ball_fires and regime_fires), "bars must be exclusive"

    # registered predictions, tested verbatim (need the full n=7 table)
    by_id = {r["id"]: r for r in rows}
    have_full = (n == 7 and set(by_id) >= {"J3", "J4", "J5", "J7"})
    if have_full:
        pred1 = bool(rho > RHO_BAR
                     and by_id["J3"]["rank_overlap"] == n
                     and by_id["J5"]["overlap_primary"]
                     > by_id["J4"]["overlap_primary"])
        pred2_cell = {
            "g1d_overlap": by_id["J3"]["overlap_primary"],
            "g1d_overlap_rank": by_id["J3"]["rank_overlap"],
            "g1f_overlap": by_id["J5"]["overlap_primary"],
            "g1e_overlap": by_id["J4"]["overlap_primary"],
            "rho_exotic": rho_exotic,
            "ball_reads": "g1d's 1.327 retention sits at the table's TOP overlap "
                          "and the exotic triple orders by overlap (rho_exotic "
                          f"= {rho_exotic:.3f}; BALL wants +1)",
            "regime_reads": f"rho_exotic = {rho_exotic:.3f} (REGIME wants <= 0 — "
                            "the exotic rows ignore the overlap)",
        }
    else:
        pred1, pred2_cell = None, {"note": "SMOKE / incomplete table — the "
                                           "prediction cells need n=7"}
    pred3 = bool(all(r["door_ratio"] >= DOOR_RATIO_BAR for r in rows))
    k_window = bool(all(2 <= r["k"] <= 6 for r in rows))

    if SMOKE:
        verdict, clause = "SMOKE (nothing adjudicated)", "shakedown only"
    elif ball_fires:
        verdict = "BALL-OWNS-THE-FLAT-PHASE"
        clause = (f"rho(primary overlap, retention) = {rho:.3f} >= {RHO_BAR} at "
                  f"n={n} AND the named breakers sit ON the curve (g1d rank "
                  f"residual {by_id['J3']['rank_residual']:+.0f}, take6 "
                  f"{by_id['J7']['rank_residual']:+.0f}; both |.| <= 1) — the "
                  "fourth currency named: the flat phase reads the wash's books "
                  "(against the multiple's +0.607 and the aggregate's -0.179 "
                  "quoted on the same page).")
    elif regime_fires:
        verdict = "REGIME-PROXY"
        clause = (f"the overall line missed (rho = {rho:.3f} < {RHO_BAR}) but "
                  f"the overlap correlates through formation regime: "
                  f"rho_fresh = {rho_fresh:.3f} (n={len(fr)}) holds while "
                  f"rho_exotic = {rho_exotic:.3f} (n={len(ex)}) flatlines or "
                  "inverts — T206's regime read survives as the organizer; the "
                  "overlap is its correlate, not the currency.")
    else:
        why = []
        if rho < RHO_BAR:
            why.append(f"rho = {rho:.3f} < {RHO_BAR} at the line (n={n})")
        if not all(on_curve[b_] for b_ in BREAKERS):
            why.append("the breakers do not sit on the curve (g1d residual "
                       f"{by_id['J3']['rank_residual']:+.0f}, take6 "
                       f"{by_id['J7']['rank_residual']:+.0f})")
        if not (rho_fresh >= RHO_BAR):
            why.append(f"rho_fresh = {rho_fresh:.3f} does not hold")
        if not (rho_exotic <= 0):
            why.append(f"rho_exotic = {rho_exotic:.3f} does not flatline/invert")
        verdict = "NEITHER"
        clause = ("no relation at the line with the breakers unexplained — "
                  + "; ".join(why)
                  + " — the flat phase's currency stays unnamed; the honest "
                    "bound recorded (with the multiple's +0.607 and the "
                    "aggregate's -0.179, the ledger remains 0-for-4).")

    metrics["adjudication"] = {
        "bars": {"BALL_OWNS_THE_FLAT_PHASE": {"fires": ball_fires},
                 "REGIME_PROXY": {"fires": regime_fires},
                 "NEITHER": {"fires": neither_fires}},
        "rank_layer": {"n": n, "spearman": rho, "bar": RHO_BAR,
                       "co_report_spearman": rho_co,
                       "ties_x_disclosed": ties_x, "ties_y_disclosed": ties_y},
        "regime_layer": {"fresh_ids": FRESH_ROWS, "exotic_ids": EXOTIC_ROWS,
                         "rho_fresh": rho_fresh, "rho_exotic": rho_exotic,
                         "alt_split_coread": {
                             "fresh_ids": ["J1", "J2", "J6"],
                             "exotic_ids": ["J3", "J4", "J5", "J7"],
                             "rho_fresh": rho_fresh_alt,
                             "rho_exotic": rho_exotic_alt,
                             "note": "J7 exotic by T172's lottery-break stamp; "
                                     "never adjudicated"}},
        "breakers": ({b_: {"rank_residual": by_id[b_]["rank_residual"],
                           "on_curve": on_curve[b_]} for b_ in BREAKERS}
                     if have_full else {"note": "SMOKE — incomplete table"}),
        "cons_pair": ({"g1e": by_id["J4"]["overlap_primary"],
                       "g1f": by_id["J5"]["overlap_primary"],
                       "g1f_above_g1e": bool(by_id["J5"]["overlap_primary"]
                                             > by_id["J4"]["overlap_primary"])}
                      if "J4" in by_id and "J5" in by_id
                      else {"note": "SMOKE — incomplete table"}),
        "registered_predictions": {
            "P1_t208": ({"fires": pred1,
                         "rho_over_bar": bool(rho > RHO_BAR),
                         "g1d_top_overlap": bool(by_id["J3"]["rank_overlap"] == n),
                         "g1f_above_g1e": bool(by_id["J5"]["overlap_primary"]
                                               > by_id["J4"]["overlap_primary"])}
                        if have_full else {"note": "SMOKE — incomplete table"}),
            "P2_discriminating_cell": pred2_cell,
            "P3_single_door": {"fires": pred3,
                               "door_ratios": {r["id"]: r["door_ratio"]
                                               for r in rows},
                               "bar": DOOR_RATIO_BAR},
            "design_expectation_k_in_2_6": {"fires": k_window,
                                            "ks": {r["id"]: r["k"] for r in rows}},
        },
        "verdict": verdict, "clause": clause,
        "composite_order": "BALL-OWNS-THE-FLAT-PHASE / REGIME-PROXY / NEITHER "
                           "(frozen before compute; the first two mutually "
                           "exclusive by construction)",
    }
    log("=" * 78)
    log(f"E231 VERDICT: {verdict}")
    log(f"  {clause}")
    log("  joined: " + "; ".join(
        f"{r['id']}({r['scale']}) ov {r['overlap_primary']:.5f} k{r['k']} "
        f"ret {r['retention_flat_min']:.3f} rr {r['rank_residual']:+.0f}"
        for r in rows))
    log("=" * 78)

    # ---- provenance + honesty ------------------------------------------------
    metrics["provenance"] = {
        "design": "scratch/e231_design.md (frozen on paper before dispatch)",
        "parents_files_md5": parents_md5,
        "machinery": {
            "u0": "lab/e225_one_currency.py module-imported; u0 = the t=0 "
                  "post-clip sign-ray on the seed-10902 stream, gated "
                  "BIT-IDENTICAL to e225's committed cells (G_U0; J1 via the "
                  "B1 bridge organism)",
            "span": "e225's svd_basis VERBATIM (the chunked fp64 Gram-SVD of "
                    "the root's OWN 20-step unwalled AdamW history, seed "
                    "10902; betas 0.9/0.95, wd 0.1, clip 1.0, lr 1e-3)",
            "knee": f"mass-plateau rule: smallest k with cumulative "
                    f"eigen-mass >= {KNEE_MASS}; k recorded per root",
            "retention": "runs/e225/metrics.json joined_table.retention_flat_min, "
                         "hard-bound at 1e-12 (the g1bS2+ flat-min convention)",
            "multiple": "runs/e225/metrics.json joined_table.multiple, "
                        "hard-bound at 1e-12",
            "aggregate": "runs/e229/metrics.json join_table.aggregate_median, "
                         "hard-bound at 1e-12",
            "seeds": f"stream 10902 (primary, gated) + {SECOND_STREAM_SEED} "
                     "(second-seed co-read, registry-grep'd free); null "
                     f"seeds {NULL_SEED_BASE}+ (129xx block clean)",
        },
        "checkpoints": {r["id"]: {
            "file": r["ckpt"],
            "flat_md5": metrics["cells"][r["id"]]["G_ROOT"]["flat_md5"]}
            for r in rows},
        "u_md5s": {rid: {"sign_ray_primary": c["primary_stream"]["sign_ray_md5"],
                         "sign_ray_second": c["second_stream"]["sign_ray_md5"],
                         "bit_identical_to_e225":
                             c["primary_stream"].get("u0_bit_identical_to_e225")}
                   for rid, c in metrics["cells"].items()},
    }
    metrics["honesty"] = {
        "identity_disclosure": ("u0 IS the sign of the wash's first step's "
                                "gradient and AdamW's first displacement is "
                                "-lr*(sign(g)+wd*theta): segment 1 is (minus) "
                                "u0 to ~1e-4, so ||P_full20 u0|| ~ 1 BY "
                                "CONSTRUCTION (recorded per row; drawn on the "
                                "figure). The informative object is the TOP-k "
                                "mass — the wash's dominant subspace — and the "
                                "null control (~sqrt(k/P)) fixes the chance "
                                "level. The design's 'not independent objects' "
                                "guard, quantified."),
        "n_and_scope": ("n=1 per organism, 7 rows: every overlap is a "
                        "single-u0, single-history read; the span lottery is "
                        "bounded by the 10914 co-read (the full and mixed "
                        "reads, per row); the 2-3x BAND lottery of the e209 "
                        "family does not enter this cell (no band is drawn) "
                        "but the same texture class applies to the span's "
                        "membership structure"),
        "retention_side": ("desk-fixed closed data (e225's committed "
                           "flat-min convention; J3 = g1d remains "
                           "OBSERVED-UNADJUDICATED, T186's stamp carried; "
                           "excluding it is a co-read never taken)"),
        "regime_split": ("T206's own sentence froze FRESH {J1,J2,J6,J7} / "
                         "EXOTIC {J3,J4,J5}; the J7-exotic alternative (T172's "
                         "lottery-break stamp) is co-reported, never "
                         "adjudicated — at n=4/n=3 (and n=3/n=4) these are "
                         "texture-level reads, disclosed"),
        "knee_sensitivity": ("the overlap-vs-k curve (k=1..8) is recorded per "
                             "row in cells[].primary_stream.overlap."
                             "overlap_vs_k_1_to_8 — the primary's k is the "
                             "frozen mass-plateau read, not shopped"),
        "nothing_guaranteed": ("the openness was the point: the overlaps could "
                               "have landed anywhere above the null; the "
                               f"observed outcome is '{verdict}'"),
    }
    metrics["status"] = "COMPLETE — adjudicated (this write replaces all " \
                        "PARTIAL progressive writes)"
    write_partial("P3 adjudicated (+ provenance + honesty)")

    # ================= P4: the figure ==========================================
    plot_ledger(rd / "e231_ball_overlap.png", rows, verdict, rho, clause,
                rho_fresh, rho_exotic, metrics)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'e231_ball_overlap.png'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

def plot_ledger(path, rows, verdict, rho, clause, rho_fresh, rho_exotic, metrics):
    fig = plt.figure(figsize=(20, 11.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0])

    def ring(ax, x, y, color, ls):
        ax.plot([x], [y], "o", ms=24, mfc="none", mec=color, mew=2.2,
                zorder=7, linestyle=ls)

    def scatter(ax, xkey, ykey, xlabel, title):
        for r in rows:
            is10 = r["scale"] == "10M"
            x, y = r[xkey], r[ykey]
            ax.plot(x, y, "*" if is10 else "o", ms=17 if is10 else 10,
                    color="darkorange" if is10 else "black", mec="k", zorder=6,
                    alpha=0.9)
            ax.annotate(r["id"], (x, y), textcoords="offset points",
                        xytext=(10, 5), fontsize=10, weight="bold")
            if r["is_breaker"]:
                ring(ax, x, y, "crimson", "-")
            if r["is_cons_pair"]:
                ring(ax, x, y, "seagreen", "--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("flat-phase retention (min g-12{10..300} / root g-12)")
        ax.axhline(1.0, ls=":", lw=1.0, color="gray")
        ax.set_title(title, fontsize=9)

    # (0,0) THE PRIMARY — retention vs overlap
    ax0 = fig.add_subplot(gs[0, 0])
    scatter(ax0, "overlap_primary", "retention_flat_min",
            "PRIMARY ball-overlap: ||P_top-k u0|| (k at the scree knee; the "
            "root's own 20-step unwalled wash span)",
            f"THE FOURTH-CURRENCY JOIN (n={len(rows)}) — o = 2.74M, * = 10M | "
            f"Spearman = {rho:.3f} vs the 0.714 line\n"
            f"rho_fresh = {rho_fresh:.2f} / rho_exotic = {rho_exotic:.2f} "
            f"(T206's split; the J7-exotic co-read in metrics)")
    null_med = max(r["null_median"] for r in rows)
    full20_min = min(r["overlap_full_rank20"] for r in rows)
    ax0.annotate("null level ~ sqrt(k/P) ~ 1e-3 (random k-basis; max median "
                 f"{null_med:.1e}) — every point sits orders above chance\n"
                 f"full-20 overlap >= {full20_min:.4f} everywhere: segment 1 "
                 "= -u0 BY CONSTRUCTION (the identity disclosure)",
                 (0.02, 0.04), xycoords="axes fraction", fontsize=8,
                 color="royalblue")
    ax0.annotate("breakers ringed solid red (g1d=J3, take6=J7); cons pair "
                 "ringed dashed green (g1e=J4, g1f=J5)",
                 (0.02, 0.96), xycoords="axes fraction", fontsize=8,
                 color="dimgray")
    ax0.set_ylim(0.45, 1.45)

    # (0,1) the second currency — retention vs multiple (e225, committed)
    ax1 = fig.add_subplot(gs[0, 1])
    scatter(ax1, "multiple", "retention_flat_min",
            "edge-multiple (e225 committed, hard-bound) — log scale",
            "THE SECOND CURRENCY — dead below the bar")
    ax1.axvline(1.0, ls="--", lw=1.4, color="crimson", alpha=0.7)
    ax1.annotate(f"committed Spearman = +{RHO_MULTIPLE_COMMITTED:.3f} "
                 "(e225; GRADED)", (0.03, 0.95), xycoords="axes fraction",
                 fontsize=9, color="black")
    ax1.set_ylim(0.45, 1.45)
    ax1.set_xscale("log")

    # (1,0) the third currency — retention vs aggregate (e229, committed)
    ax2 = fig.add_subplot(gs[1, 0])
    scatter(ax2, "aggregate_median", "retention_flat_min",
            "margin aggregate median (e229 committed, hard-bound)",
            "THE THIRD CURRENCY — dead (anti-tilted)")
    ax2.annotate(f"committed Spearman = {RHO_AGGREGATE_COMMITTED:.3f} (e229)",
                 (0.03, 0.95), xycoords="axes fraction", fontsize=9,
                 color="black")
    ax2.set_ylim(0.45, 1.45)

    # (1,1) the instrument — the door structure + the identity disclosure
    ax3 = fig.add_subplot(gs[1, 1])
    xoff = 0.0
    for r in rows:
        coefs = metrics["cells"][r["id"]]["primary_stream"]["overlap"][
            "coefs_u0_all_pcs"][:r["k"]]
        xs_ = [xoff + i * 0.09 for i in range(len(coefs))]
        ax3.bar(xs_, [abs(c) for c in coefs], width=0.075,
                color="darkorange" if r["scale"] == "10M" else "black",
                alpha=0.8)
        ax3.plot([xoff - 0.02, xoff + (len(coefs) - 1) * 0.09 + 0.02],
                 [r["overlap_primary"]] * 2, "_", color="crimson", ms=12,
                 mew=2.4, zorder=6)
        ax3.annotate(f"{r['id']}\nk={r['k']}",
                     (xoff + (len(coefs) - 1) * 0.09 / 2, 1.07),
                     fontsize=7.5, ha="center")
        xoff += len(coefs) * 0.09 + 0.14
    ax3.axhline(DOOR_RATIO_BAR, ls="--", lw=1.2, color="seagreen")
    ax3.annotate("the 0.90 single-door line (prediction 3: door-ratio = tallest "
                 "bar / red dash)", (0.0, DOOR_RATIO_BAR + 0.02), fontsize=8,
                 color="seagreen")
    ax3.set_ylim(0, 1.18)
    ax3.set_xticks([])
    ax3.set_xlim(-0.05, max(xoff, 1.0))
    ax3.set_xlabel("per root: |<u0, pc_j>| for j = 1..k (left to right); the "
                   "red dash = the primary overlap ||P_top-k u0||")
    ax3.set_ylabel("|cosine|")
    ax3.set_title("THE INSTRUMENT — the door structure (which wash PCs hold "
                  "the fact)\nfull-20 overlap ~1 by construction (segment 1 = "
                  "-u0): the read is the top-k mass; null ~ sqrt(k/P)",
                  fontsize=9)

    fig.suptitle(f"E231 — THE BALL-SIDE OVERLAP JOIN: the three-currency "
                 f"ledger on one page (overlap vs multiple +0.607 vs "
                 f"aggregate -0.179)    VERDICT: {verdict}", fontsize=12)
    fig.text(0.5, 0.012, clause, ha="center", fontsize=8.5, wrap=True,
             color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
