"""E114 — the W006 BRAKE-SIGNATURE probe: why does the address suppress its
own fact? Three mechanisms, three named signatures (REGISTERED).

WHY (W006, verbatim spirit): after e109/e113's jittered-replay consolidation,
deleting the address row RAISES expression at its own coordinate (e113: none
g+0 0.785 -> d129 0.917; d-all-addresses 0.905). The mature memory's old
index is mildly SUPPRESSIVE of its own content — an address that became a
brake. W006 names three mechanisms, each with its own signature:

  M1 READ-BUDGET DILUTION — the read is routing-only (T054/T059); if the
     address coordinate still attracts routing mass whose content is now
     redundant with the field, the net wastes read budget on a duplicate.
     The brake is OPPORTUNITY COST.
  M2 DUPLICATE-INTERFERENCE — the address path delivers an older version of
     the fact; two versions mixing is worse than either alone. The brake is
     CROSSTALK between the old and new carriers.
  M3 TRAINED INHIBITOR — replay's address+field double-evidence taught an
     actual inhibitory connection (calibration; don't over-commit). The
     brake is FUNCTIONAL, learned on purpose.

BASE (registered): rebuild e109 arm (a)'s consolidated net EXACTLY as e113
did — runs/checkpoints/e048_repro.pt + the 300-step jittered-replay
fine-tune (seed 10901, offsets {-8,-4,0,+4,+8}, batch 16 install + 16 anchor
(8 paired + 8 random), e043 token-level union CE, AdamW (0.9,0.95) wd 0.1
constant lr 1e-3 clip 1.0), CPU. GATES: G_SPLICE (install mix 19/41),
G_INST (e048 battery p(Z) 0.556313 +- 0.005), G_REPRO (post-none install-60
table vs e109's 0.99/0.95/0.78/0.99/0.99, tol 0.05/cell — e113's CPU rebuild
matched to <0.01/cell), G_FWD (the manual attention-recording forward vs
net(), max p(Z) dev < 1e-4, clean and wpe-zeroed paths).

ALL SIGNATURES measured on the consolidated net, ORIGINAL GEOMETRY (130-token
contexts, decision row = wpe row 129), install-60 primary / held-30 mirror.
The BRAKE per context: brake_i = p(Z)_D129,i - p(Z)_none,i (the e109/e113
registered single-address deletion; the d-all-addresses variant is a
robustness row).

  S1 (M1) BRAKE vs ROUTING MASS: per-decision attention mass on the address
      coordinate — e100-style, sum over all 36 layer-heads of the softmax
      attention probability from the decision row (last position, 129) to
      the position carrying wpe row 129, in the CLEAN consolidated net, one
      forward, pre-intervention. M1 predicts POSITIVE correlation: more
      routing mass on the address = more wasted budget = bigger brake.
  S2 (M2) BRAKE vs FIELD STRENGTH: the field read's strength per context =
      PRIMARY: the anchor-band attention mass at the decision — the same
      36-lh-summed attention of the decision row over the BODY BAND
      (context positions 1..128; row 0 excluded: generic window scaffold,
      e113's never-touch convention). SECONDARY (reported, CIRCULARITY-
      FLAGGED: it is one addend of the brake itself): the held-expression
      without the address, p(Z)_D129,i. M2 predicts POSITIVE correlation:
      crosstalk scales with the duplicate's presence.
  S3 (M3) ADDRESS-ONLY RE-EXPOSURE: the 2x2 {address wpe-129 present /
      deleted} x {field readable / ablated}. Field ablation = e075-style
      whole-position V-zero of the body band (v := 0 at context positions
      1..128, every layer, every head, all queries; K never touched; row 0
      and the address coordinate 129 untouched), applied at EVAL on the
      consolidated net. ADDR_ONLY (A, -F) vs NEITHER (-A, -F): M3 predicts
      the brake STILL OPERATES with no field — p(Z)_ADDR_ONLY <
      p(Z)_NEITHER (the residual address presence suppresses with nothing
      to dilate or crosstalk with). M1/M2 predict NO suppression there
      (no field = no duplicate = no cost).

STATISTICS: Pearson r and Spearman rho over the 60 contexts; bootstrap 95%
CIs (n=1000, context resampling, seeds frozen 0/1); S3 = paired per-context
difference ADDR_ONLY - NEITHER (negative = suppression), mean + bootstrap CI
+ two-sided sign test. Family honesty note: the three attention masses are
mechanically complementary (per layer-head the decision's probs sum to 1, so
addr_mass + row0_mass + field_mass = 36 exactly); r_S1 and r_S2 are near
mirror images, making S1/S2 effectively ONE sign-discriminator — reported
as such, not hidden.

REGISTERED BARS (frozen before compute):
  1. S3 PRECEDENCE (intervention beats correlation — the lab's honesty
     reflex): if p(Z)_ADDR_ONLY < p(Z)_NEITHER with the paired bootstrap CI
     excluding 0, M3 IS DIRECTLY SUPPORTED and wins (the correlations are
     then reported as covariation texture).
  2. Else among S1/S2 PRIMARY measures: the correlation with the LARGEST
     |r| whose CI excludes 0 AND whose SIGN matches its mechanism (positive)
     wins for M1/M2 respectively. A significant ANTI-signed correlation
     falsifies that mechanism's direction (texture, not a win).
  3. ALL NULL (no CI excluding 0 in a mechanism's direction; S3 CI covering
     0) => FOURTH-STORY NEEDED — the brake wants a mechanism W006 did not
     name; report honestly.

Report-only secondaries (never barred on): held-30 mirrors of S1/S2/S3; the
d-all-addresses brake correlations; jittered-geometry (+4/+8, where wpe row
129 is a genuine CONTEXT row read by decision rows 133/137) S1 mirrors; the
full 2x2 table; per-layer address-mass descriptives.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import;
torch.set_num_threads(8)); single run target <= 30 min (the 300-step
rebuild is ~9 min on this box, e113 precedent; every measurement forward is
seconds). No new automations.

Outputs: runs/e114/{metrics.json, brake_signature.png}.
No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).

Run:  cd lab && python e114_brake_signature.py        (E114_SMOKE=1 for smoke)
"""
from __future__ import annotations

import copy
import math
import random
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (operator)

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(8)                              # 8 threads max (operator)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E114_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
INSTALLED_CK = "e048_repro.pt"

JITTERS = (-8, -4, 0, 4, 8)       # e109's registered jitter set
GEO_ORDER = [-8, -4, 0, 4, 8]     # display order (original geometry in middle)
ADDR_ROWS = (121, 125, 129, 133, 137)   # 129 + the grown rows (T064)
DEC_ROW = {j: PRE - 1 + j for j in GEO_ORDER}   # decision (last-ctx) wpe row

# fine-tune envelope (registered — e109 arm (a) verbatim except CPU/caps)
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
FT_TIME_CAP = 1500.0              # CPU safety cap (e109's 180 s was GPU thermal)
EVAL_EVERY = 25 if not SMOKE else 2
NAME_BS = 16                      # install windows per step
ANCH_BS = 16                      # anchor windows per step (8 paired + 8 random)
CONS_SEED = 10901                 # e109 arm (a) seed VERBATIM

# batteries / guards
R_EVAL_SEED = 26502               # e065's CE_R eval-bank seed (verbatim)
G_INST_REF = 0.556313             # e065 arms_battery no_removal p_z_mean
G_INST_TOL = 0.005                # e091 convention

# e109 arm (a) reference cells (runs/e109/metrics.json, full precision) —
# the reproduction gate for this rebuild (the operator's 0.99/0.95/0.78/
# 0.99/0.99 table at 2 dp)
E109_REF_NONE = {-8: 0.9924036860466003, -4: 0.9496323466300964,
                 0: 0.776076078414917, 4: 0.9848979115486145,
                 8: 0.9854525923728943}
G_REPRO_TOL = 0.05                        # per-cell tolerance
G_FWD_TOL = 1e-4                          # manual-forward instrument gate

# the field band for S3's V-zero (frozen): body content rows of the 130-token
# context; row 0 excluded (generic scaffold, e113 never-touch), address
# coordinate 129 excluded by definition (it IS the S3 contrast)
FIELD_BAND = list(range(1, PRE - 1))               # positions 1..128
ADDR_COORD = PRE - 1                               # position 129

BOOT_N = 1000

REGISTERED_BARS = {
    "s3_precedence": "if p(Z)_ADDR_ONLY < p(Z)_NEITHER with paired bootstrap "
                     "CI excluding 0 => M3 TRAINED INHIBITOR directly "
                     "supported (intervention beats correlation)",
    "s1_m1": "corr(brake, address routing mass) POSITIVE with CI excluding 0 "
             "=> M1 READ-BUDGET DILUTION",
    "s2_m2": "corr(brake, field read strength) POSITIVE with CI excluding 0 "
             "=> M2 DUPLICATE-INTERFERENCE",
    "winner_rule": "S3 fires => M3 wins; else largest |r| among S1/S2 with CI "
                   "excluding 0 AND mechanism-consistent (positive) sign; "
                   "anti-signed significance = direction falsified, texture",
    "all_null": "no mechanism-consistent CI excludes 0 and S3 CI covers 0 => "
                "FOURTH-STORY NEEDED (honest report)",
    "complementarity_note": "addr_mass + row0_mass + field_mass = 36 exactly "
                            "(per-lh normalization): r_S1/r_S2 are near "
                            "mirror images — effectively one "
                            "sign-discriminator, reported as such",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-only (CUDA_VISIBLE_DEVICES=-1, 8 threads) per operator instruction; "
    "e109's arm (a) was fine-tuned on cuda — same seed/recipe/batches, but "
    "float arithmetic differs by device, so the rebuild is gated (G_REPRO, "
    "0.05/cell) against e109's post-none table rather than bit-compared "
    "(e113's CPU rebuild matched to <0.01/cell).",
    "e109's 180 s wall cap was a GPU thermal guard — raised to 1500 s (CPU "
    "envelope) so the registered 300 steps complete; the 300-step count and "
    "every other recipe element are verbatim.",
    "S2 has two admissible operationalizations (tasking: 'the anchor-band "
    "attention mass at the decision, or the held-expression without the "
    "address'); registered PRIMARY = attention mass (non-circular); the "
    "held-expression variant is reported with a circularity flag (it is one "
    "addend of the brake) and never adjudicated on.",
    "The S3 field band is frozen at context positions 1..128: row 0 excluded "
    "(generic window scaffold; e113 never touched it — including it would "
    "conflate field ablation with scaffold loss), the address coordinate 129 "
    "excluded by definition (it is the contrast).",
]


# ------------------------------------------------------------------ instruments

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30,
                 keep_per_ctx=False) -> dict:
    """e068-style battery on CPU: p(Z) at the last position over contexts."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    out = {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
           "std_pz": float(p.std()),
           "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
           "frac_argmax_z": amax / ids.shape[0]}
    if keep_per_ctx:
        out["pz_per_ctx"] = [float(v) for v in p.tolist()]
    return out


@torch.no_grad()
def ce_fixed_cpu(net: TinyGPT, x, y, bs=64) -> float:
    """e065 ce_fixed (CPU)."""
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


def deleted_wpe(sd: dict, rows: tuple[int, ...]) -> tuple[dict, dict]:
    """D2-style subtractive row-zero on wpe rows; confinement gate (e065
    G_SURG convention: exact element count, row confinement, everything
    else bit-identical)."""
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


@torch.no_grad()
def manual_forward(net: TinyGPT, ids: torch.Tensor, zid: int,
                   vzero=None, wpe_zero_rows=(), record_att=False, bs=30):
    """Manual TinyGPT forward (e100 query_features pattern, TinyGPT op
    order): returns (p(Z) at last position per row (N,), attention profile
    of the DECISION row (last position) per layer — tensor (L, N, H, T) —
    when record_att, else None). vzero: (N, T) bool — whole-position V-zero
    (v := 0 there, every layer, every head, all queries; K untouched;
    e075/e100 instrument). wpe_zero_rows: eval-time wpe row zeroing (same
    arithmetic as deleted_wpe, in-forward)."""
    L = net.cfg.n_layer
    H = net.cfg.n_head
    wpe_w = net.wpe.weight
    if wpe_zero_rows:
        w2 = wpe_w.clone()
        for r in wpe_zero_rows:
            w2[r] = 0.0
        wpe_w = w2
    outs = []
    att_all = [[] for _ in range(L)]
    for i in range(0, ids.shape[0], bs):
        idxs = ids[i:i + bs]
        vz = vzero[i:i + bs] if vzero is not None else None
        N, T = idxs.shape
        pos = torch.arange(T)
        x = net.wte(idxs) + wpe_w[pos].unsqueeze(0)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        for li, blk in enumerate(net.h):
            xh = blk.ln1(x)
            qkv = blk.attn.c_attn(xh)
            C = qkv.shape[-1] // 3
            d = C // H
            q, k, v = qkv.split(C, dim=2)
            q = q.view(N, T, H, d).transpose(1, 2)
            k = k.view(N, T, H, d).transpose(1, 2)
            v = v.view(N, T, H, d).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
            att = att.masked_fill(causal, float("-inf"))
            probs = torch.softmax(att, dim=-1)
            if record_att:
                att_all[li].append(probs[:, :, -1, :].clone())
            if vz is not None and bool(vz.any()):
                v = v.masked_fill(vz[:, None, :, None], 0.0)
            y = (probs @ v).transpose(1, 2).reshape(N, T, C)
            x = x + blk.attn.c_proj(y)
            x = x + blk.mlp(blk.ln2(x))
        pr = F.softmax(net.lm_head(net.ln_f(x[:, -1, :])), -1)[:, zid]
        outs.append(pr)
    p = torch.cat(outs)
    prof = torch.stack([torch.cat(a, 0) for a in att_all], 0) if record_att \
        else None        # (L, N, H, T)
    return p, prof


# ------------------------------------------------------------------ statistics

def rankdata_avg(v):
    """Ascending ranks 1..n with TIES AVERAGED (proper Mann-Whitney ranks)."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    ranks = np.empty(len(v), float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def spearman(x, y):
    return pearson(rankdata_avg(x), rankdata_avg(y))


def boot_corr_ci(x, y, n=BOOT_N, seed=0, method="pearson"):
    """Bootstrap 95% CI of the correlation (context resampling)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    f = pearson if method == "pearson" else spearman
    vals = []
    for _ in range(n):
        sel = rng.integers(0, len(x), len(x))
        r = f(x[sel], y[sel])
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def boot_mean_ci(d, n=BOOT_N, seed=1):
    """Bootstrap 95% CI of the mean (context resampling)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    vals = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def sign_test(d):
    """Two-sided sign test on paired diffs (vs 0)."""
    d = np.asarray(d, float)
    n = int(np.isfinite(d).sum())
    s = int((d > 0).sum())
    if n == 0:
        return {"n": 0, "n_pos": 0, "frac_pos": float("nan"),
                "p_two_sided": float("nan")}
    k = min(s, n - s)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** (n - 1))
    return {"n": n, "n_pos": s, "frac_pos": s / n,
            "p_two_sided": min(1.0, p)}


def corr_block(x, y, label, seed=0):
    """Pearson + Spearman with bootstrap CIs, packaged."""
    r_p = pearson(x, y)
    r_s = spearman(x, y)
    ci_p = boot_corr_ci(x, y, seed=seed, method="pearson")
    ci_s = boot_corr_ci(x, y, seed=seed, method="spearman")
    return {"label": label, "n": int(len(x)),
            "pearson_r": r_p, "pearson_ci95": list(ci_p),
            "spearman_rho": r_s, "spearman_ci95": list(ci_s)}


# ------------------------------------------------------------------ fine-tune

def finetune_arm(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                 pool_mask: torch.Tensor, anchor: torch.Tensor,
                 train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                 seed: int):
    """e109 arm-(a) fine-tune VERBATIM (recipe/seed/batch composition), CPU:
    batch 32 = 16 install windows from the jittered pool + 16 anchors
    (8 paired + 8 random); e043 token-level union CE; constant lr 1e-3
    AdamW (0.9,0.95) wd 0.1 clip 1.0; 300 steps. In-loop CPU evals every 25
    steps (original-geometry battery + CE_R)."""
    net = copy.deepcopy(net0).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    traj, t_start = [], time.time()
    step = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, FT_STEPS + 1):
        ix = torch.randint(n_pool, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                           generator=gen)
        nw = pool_x[ix].to(CPU)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])], 0).to(CPU)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(NAME_BS + ANCH_BS, x.shape[1], dtype=torch.bool, device=CPU)
        m[:NAME_BS] = pool_mask[ix].to(CPU)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              reduction="none").view(x.shape[0], x.shape[1])
        nm = nll[:NAME_BS][m[:NAME_BS]]
        cm = nll[NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == FT_STEPS or \
                (time.time() - t_start) > FT_TIME_CAP:
            evl.load_state_dict({k: v.detach().cpu().clone()
                                 for k, v in net.state_dict().items()})
            evl.eval()
            bz = battery_cell(evl, f_eval_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "p_z_mean": bz["mean_pz"],
                         "frac_argmax_z": bz["frac_argmax_z"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p_z {bz['mean_pz']:.4f} argmaxZ "
                f"{bz['frac_argmax_z']:.2f} CE_R {ce_r:.4f} "
                f"({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            break
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e114_smoke" if SMOKE else "e114")
    assert not torch.cuda.is_available(), "CPU-only violated: CUDA visible"
    log(f"E114 BRAKE-SIGNATURE probe (W006; smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e065/e091/e109/e113 verbatim)
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
    log(f"protocol rebuilt: install60 {mix}, held30 (SPLICE_RNG {E43.SPLICE_RNG})")

    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)

    # ---------------- jittered install windows (training pool) + batteries
    # (e109/e113 construction verbatim — only needed to drive the rebuild)
    jit_x, jit_mask = {}, {}
    for j in JITTERS:
        wins = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            if len(w) != BLOCK:
                raise RuntimeError(f"window len {len(w)} != {BLOCK} at offset {j}")
            wins.append(w)
        jit_x[j] = torch.stack(wins)
        m = torch.zeros(len(wins), BLOCK - 1, dtype=torch.bool)
        m[:, PRE - 1 + j: PRE - 1 + j + len(NAME)] = True
        jit_mask[j] = m
    pool_a_x = torch.cat([jit_x[j] for j in JITTERS])          # (300, 256)
    pool_a_mask = torch.cat([jit_mask[j] for j in JITTERS])
    log(f"jitter pool (arm a): {tuple(pool_a_x.shape)} "
        f"(offsets {list(JITTERS)}); anchor bank 16 paired originals")

    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])       # e065 anchor bank

    # batteries per geometry: ctx = train_text[p-PRE-j : p], readout p(Z) at
    # the last position (wpe row 129+j). install60 primary, held30 mirror.
    bat_ids = {}
    for j in GEO_ORDER:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    f_eval_ids = bat_ids[(0, "install60")]                     # the G_INST battery

    # CE_R eval bank (e065 verbatim, seed 26502)
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- base net + instrument gate
    net0 = load_cpu(E43.REPO / "runs" / "checkpoints" / INSTALLED_CK)
    evl = copy.deepcopy(net0)
    bz0 = battery_cell(evl, f_eval_ids, zid)
    G_INST = {"battery_pz": bz0["mean_pz"], "ref": G_INST_REF,
              "tol": G_INST_TOL,
              "pass": bool(abs(bz0["mean_pz"] - G_INST_REF) < G_INST_TOL)}
    log(f"G_INST installed battery p(Z) {bz0['mean_pz']:.6f} "
        f"(ref {G_INST_REF}): {'PASS' if G_INST['pass'] else 'FAIL'}")
    if not G_INST["pass"]:
        raise RuntimeError("instrument broken vs e065/e091/e109/e113")
    ce_r0 = ce_fixed_cpu(evl, *r_eval_xy)
    log(f"CE_R (60 name-free val windows): {ce_r0:.4f}")

    # ---------------- REBUILD e109 arm (a): jittered replay, seed 10901
    log(f"ARM (a) REBUILD: e109 consolidation fine-tune VERBATIM on CPU "
        f"(seed {CONS_SEED}, {FT_STEPS} steps, lr 1e-3, batch 32)")
    a = finetune_arm("a_consolidated", net0, pool_a_x, pool_a_mask, anchor,
                     train_ids, r_eval_xy, f_eval_ids, zid, CONS_SEED)
    sd_a = a["sd"]
    evl = copy.deepcopy(net0)
    evl.load_state_dict(sd_a)
    evl.eval()

    # ---------------- G_REPRO: post-none table vs e109 (5 geometries)
    none_means = {}
    for j in GEO_ORDER:
        none_means[j] = battery_cell(evl, bat_ids[(j, "install60")],
                                     zid)["mean_pz"]
    repro_cells = {f"g{g:+d}": {"this_run": none_means[g],
                                "e109_ref": E109_REF_NONE[g],
                                "diff": none_means[g] - E109_REF_NONE[g]}
                   for g in GEO_ORDER}
    G_REPRO = {"cells": repro_cells, "tol": G_REPRO_TOL,
               "pass": bool(all(abs(c["diff"]) < G_REPRO_TOL
                                for c in repro_cells.values()))}
    log(f"G_REPRO post-none table vs e109 (tol {G_REPRO_TOL}): "
        + " ".join(f"g{g:+d} {none_means[g]:.4f}/{E109_REF_NONE[g]:.4f}"
                   f"({none_means[g] - E109_REF_NONE[g]:+.3f})"
                   for g in GEO_ORDER)
        + f" -> {'PASS' if G_REPRO['pass'] else 'FAIL'}")
    if not G_REPRO["pass"] and not SMOKE:
        raise RuntimeError(f"rebuild failed to reproduce e109's post-none "
                           f"table within {G_REPRO_TOL}: {repro_cells}")

    # wpe probes (confinement/descriptive, e113 convention)
    orig = net0.state_dict()["wpe.weight"]
    w0 = sd_a["wpe.weight"]
    wpe_probes = {str(r): {"norm": float(w0[r].norm()),
                           "cos_to_e048": float(F.cosine_similarity(
                               w0[r], orig[r], dim=0)),
                           "e048_norm": float(orig[r].norm())}
                  for r in (0,) + ADDR_ROWS}
    log("wpe row probes (norm/cos-to-e048): "
        + " | ".join(f"r{r} {wpe_probes[str(r)]['norm']:.3f}/"
                     f"{wpe_probes[str(r)]['cos_to_e048']:+.2f}"
                     for r in ADDR_ROWS))

    # ---------------- S1/S2 inputs: decision-row attention (CLEAN, g0)
    # attention of the decision row (position 129) over positions 0..129,
    # per layer-head, in ONE clean forward of the consolidated net.
    att = {}                      # (battery,) -> (L, N, H, T) profile
    for tag in ("install60", "held30"):
        _, prof = manual_forward(evl, bat_ids[(0, tag)], zid, record_att=True)
        att[tag] = prof
    L, N, H, T = att["install60"].shape
    masses = {}
    for tag in ("install60", "held30"):
        pr = att[tag].sum(0).sum(1).numpy()          # (N, T) 36-lh summed
        masses[tag] = {
            "addr_mass": pr[:, ADDR_COORD].copy(),           # position 129
            "row0_mass": pr[:, 0].copy(),
            "field_mass": pr[:, 1:PRE - 1].sum(1).copy(),    # body band 1..128
            "total_mass": pr.sum(1).copy(),
        }
    tot = masses["install60"]["total_mass"]
    comp_gate = {
        "addr_plus_row0_plus_field_vs_total_max_dev":
            float(np.abs(masses["install60"]["addr_mass"]
                         + masses["install60"]["row0_mass"]
                         + masses["install60"]["field_mass"] - tot).max()),
        "total_mass_mean": float(tot.mean()), "expected_total": L * H,
        "rows_sum_to_one_max_dev": float(np.abs(
            att["install60"].sum(-1).numpy() - 1.0).max()),
    }
    comp_gate["pass"] = bool(
        comp_gate["addr_plus_row0_plus_field_vs_total_max_dev"] < 1e-4
        and comp_gate["rows_sum_to_one_max_dev"] < 1e-4)
    log("attention masses (install60 g0, 36-lh sums): addr "
        f"{masses['install60']['addr_mass'].mean():.3f} | row0 "
        f"{masses['install60']['row0_mass'].mean():.3f} | field "
        f"{masses['install60']['field_mass'].mean():.3f} | total "
        f"{tot.mean():.3f} (= L*H {L * H}) | complementarity gate "
        f"{'PASS' if comp_gate['pass'] else 'FAIL'}")
    per_layer_addr = {f"L{li}": float(att["install60"][li, :, :,
                                                      ADDR_COORD].sum(1).mean())
                      for li in range(L)}
    per_layer_field = {f"L{li}": float(att["install60"][li, :, :,
                                                       1:PRE - 1].sum(-1).sum(1).mean())
                       for li in range(L)}
    mean_profile = att["install60"].sum(0).sum(1).mean(0).numpy()   # (T,)

    # ---------------- G_FWD: manual forward vs net() (clean + wpe-zeroed)
    p_clean_manual, _ = manual_forward(evl, f_eval_ids, zid)
    p_clean_net = []
    with torch.no_grad():
        for i in range(0, f_eval_ids.shape[0], 30):
            lg, _ = evl(f_eval_ids[i:i + 30])
            p_clean_net.append(F.softmax(lg[:, -1], -1)[:, zid])
    dev_clean = float((p_clean_manual - torch.cat(p_clean_net)).abs().max())
    sd_del, gate_d129 = deleted_wpe(sd_a, (129,))
    evl_del = copy.deepcopy(net0)
    evl_del.load_state_dict(sd_del)
    p_del_net = []
    with torch.no_grad():
        for i in range(0, f_eval_ids.shape[0], 30):
            lg, _ = evl_del(f_eval_ids[i:i + 30])
            p_del_net.append(F.softmax(lg[:, -1], -1)[:, zid])
    p_del_manual, _ = manual_forward(evl, f_eval_ids, zid,
                                     wpe_zero_rows=(129,))
    dev_del = float((p_del_manual - torch.cat(p_del_net)).abs().max())
    G_FWD = {"clean_max_dev": dev_clean, "wpe_zero_max_dev": dev_del,
             "tol": G_FWD_TOL,
             "pass": bool(dev_clean < G_FWD_TOL and dev_del < G_FWD_TOL)}
    log(f"G_FWD manual vs net(): clean dev {dev_clean:.2e}, wpe-zeroed dev "
        f"{dev_del:.2e} -> {'PASS' if G_FWD['pass'] else 'FAIL'}")
    if not G_FWD["pass"]:
        raise RuntimeError("manual attention forward broken vs TinyGPT")

    # ---------------- per-context brake (g0 primary) + jittered secondaries
    # none / d129 / d_all per-context at g0 (install60 primary, held30 mirror)
    pz = {}
    for tag in ("install60", "held30"):
        pz[(tag, "none")] = battery_cell(evl, bat_ids[(0, tag)], zid,
                                         keep_per_ctx=True)["pz_per_ctx"]
        pz[(tag, "d129")] = battery_cell(evl_del, bat_ids[(0, tag)], zid,
                                         keep_per_ctx=True)["pz_per_ctx"]
        sd_dall, g_all = deleted_wpe(sd_a, ADDR_ROWS)
        evl_dall = copy.deepcopy(net0)
        evl_dall.load_state_dict(sd_dall)
        pz[(tag, "d_all")] = battery_cell(evl_dall, bat_ids[(0, tag)], zid,
                                          keep_per_ctx=True)["pz_per_ctx"]
        gates_all = g_all
    brake = {tag: (np.asarray(pz[(tag, "d129")]) - np.asarray(pz[(tag, "none")]))
             for tag in ("install60", "held30")}
    brake_all = {tag: (np.asarray(pz[(tag, "d_all")])
                       - np.asarray(pz[(tag, "none")]))
                 for tag in ("install60", "held30")}
    log(f"BRAKE (install60 g0): mean {brake['install60'].mean():+.4f} "
        f"[{boot_mean_ci(brake['install60'])[0]:+.3f},"
        f"{boot_mean_ci(brake['install60'])[1]:+.3f}] | frac>0 "
        f"{float((brake['install60'] > 0).mean()):.2f} | d_all variant mean "
        f"{brake_all['install60'].mean():+.4f}")

    # jittered-geometry S1 mirrors (report-only): d129 brake at +4/+8 vs
    # attention from decision row 133/137 onto context position 129
    jit_mirror = {}
    for j in (4, 8):
        sdj, gj = deleted_wpe(sd_a, (129,))
        evj = copy.deepcopy(net0)
        evj.load_state_dict(sdj)
        pz_j_del = battery_cell(evj, bat_ids[(j, "install60")], zid,
                                keep_per_ctx=True)["pz_per_ctx"]
        pz_j_non = battery_cell(evl, bat_ids[(j, "install60")], zid,
                                keep_per_ctx=True)["pz_per_ctx"]
        _, profj = manual_forward(evl, bat_ids[(j, "install60")], zid,
                                  record_att=True)
        am = profj.sum(0).sum(1).numpy()[:, 129]      # mass on context row 129
        brj = np.asarray(pz_j_del) - np.asarray(pz_j_non)
        jit_mirror[f"g{j:+d}"] = {
            "mean_brake": float(brj.mean()),
            "mean_addr_mass_on_ctx129": float(am.mean()),
            "corr": corr_block(am, brj, f"jit g{j:+d}: brake vs ctx-129 mass"),
            "gate_wpe": gj,
        }
        log(f"jit mirror g{j:+d}: brake {brj.mean():+.4f}, ctx-129 mass "
            f"{am.mean():.3f}, r {jit_mirror[f'g{j:+d}']['corr']['pearson_r']:+.3f}")

    # ---------------- S1 / S2 correlations (registered)
    S = {}
    for tag, seedoff in (("install60", 0), ("held30", 2)):
        am = masses[tag]["addr_mass"]
        fm = masses[tag]["field_mass"]
        S[tag] = {
            "S1_addr": corr_block(am, brake[tag], "S1: brake vs address routing mass",
                                  seed=seedoff),
            "S2_field": corr_block(fm, brake[tag], "S2: brake vs field read strength",
                                   seed=seedoff),
            "S2_alt_noaddr_expr": corr_block(
                np.asarray(pz[(tag, "d129")]), brake[tag],
                "S2 secondary (CIRCULAR): brake vs held-expression-no-address",
                seed=seedoff),
            "row0": corr_block(masses[tag]["row0_mass"], brake[tag],
                               "brake vs row-0 mass (scaffold control)",
                               seed=seedoff),
            "mass_anticorr": corr_block(am, fm,
                                        "addr_mass vs field_mass (complementarity)",
                                        seed=seedoff),
            "brake_dall_S1": corr_block(am, brake_all[tag],
                                        "robustness: d_all brake vs addr mass",
                                        seed=seedoff),
            "brake_dall_S2": corr_block(fm, brake_all[tag],
                                        "robustness: d_all brake vs field mass",
                                        seed=seedoff),
        }
        log(f"[{tag}] S1 r {S[tag]['S1_addr']['pearson_r']:+.3f} CI "
            f"{S[tag]['S1_addr']['pearson_ci95']} | S2 r "
            f"{S[tag]['S2_field']['pearson_r']:+.3f} CI "
            f"{S[tag]['S2_field']['pearson_ci95']} | rho "
            f"{S[tag]['S1_addr']['spearman_rho']:+.3f}/"
            f"{S[tag]['S2_field']['spearman_rho']:+.3f}")

    # ---------------- S3: the 2x2 (address x field), e075-style V-zero
    vz = {}
    for tag in ("install60", "held30"):
        n = bat_ids[(0, tag)].shape[0]
        m = torch.zeros(n, PRE, dtype=torch.bool)
        m[:, FIELD_BAND[0]:FIELD_BAND[-1] + 1] = True
        vz[tag] = m
    G_VZ = {"band": [FIELD_BAND[0], FIELD_BAND[-1]],
            "n_positions_per_ctx": len(FIELD_BAND),
            "counts_ok": bool(all(int(v.sum() / v.shape[0]) == len(FIELD_BAND)
                                  for v in vz.values())),
            "k_untouched": True, "row0_untouched": True,
            "addr_coord_129_untouched": True}
    G_VZ["pass"] = G_VZ["counts_ok"]

    s3 = {}
    for tag in ("install60", "held30"):
        p_AF, _ = manual_forward(evl, bat_ids[(0, tag)], zid)   # clean = A+F
        p_nAF, _ = manual_forward(evl, bat_ids[(0, tag)], zid,
                                  wpe_zero_rows=(129,))          # -A +F
        p_AnF, _ = manual_forward(evl, bat_ids[(0, tag)], zid,
                                  vzero=vz[tag])                 # A + -F
        p_nAnF, _ = manual_forward(evl, bat_ids[(0, tag)], zid,
                                   vzero=vz[tag],
                                   wpe_zero_rows=(129,))         # -A + -F
        # cross-check the manual wpe-zero path against the battery instrument
        dev_x = float(np.abs(np.asarray(p_nAF.tolist())
                             - np.asarray(pz[(tag, "d129")])).max())
        dev_clean_x = float(np.abs(np.asarray(p_AF.tolist())
                                   - np.asarray(pz[(tag, "none")])).max())
        d_supp = np.asarray(p_nAnF.tolist()) - np.asarray(p_AnF.tolist())
        ci = boot_mean_ci(d_supp, seed=1)
        s3[tag] = {
            "pz_AF_clean": [float(v) for v in p_AF.tolist()],
            "pz_nAF_d129": [float(v) for v in p_nAF.tolist()],
            "pz_AnF_addr_only": [float(v) for v in p_AnF.tolist()],
            "pz_nAnF_neither": [float(v) for v in p_nAnF.tolist()],
            "means": {"AF": float(p_AF.mean()), "nAF": float(p_nAF.mean()),
                      "AnF": float(p_AnF.mean()), "nAnF": float(p_nAnF.mean())},
            "suppression_diff_nAnF_minus_AnF": {
                "mean": float(d_supp.mean()), "ci95": list(ci),
                "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0),
                "positive_means_suppression": bool(d_supp.mean() > 0
                                                   and ci[0] > 0),
                "sign_test": sign_test(d_supp)},
            "field_effect_diff_AF_minus_AnF": {
                "mean": float((np.asarray(p_AF.tolist())
                               - np.asarray(p_AnF.tolist())).mean())},
            "addr_effect_with_field_diff_nAF_minus_AF": {
                "mean": float((np.asarray(p_nAF.tolist())
                               - np.asarray(p_AF.tolist())).mean()),
                "note": "the original brake, manual-path replication"},
            "instrument_cross_dev": {"d129_path": dev_x, "clean_path": dev_clean_x},
        }
        st = s3[tag]["suppression_diff_nAnF_minus_AnF"]
        log(f"[{tag}] S3 2x2: A+F {s3[tag]['means']['AF']:.3f} | -A+F "
            f"{s3[tag]['means']['nAF']:.3f} | A+-F(ADDR_ONLY) "
            f"{s3[tag]['means']['AnF']:.3f} | -A+-F(NEITHER) "
            f"{s3[tag]['means']['nAnF']:.3f} | suppression (NEITHER-ADDR_ONLY) "
            f"{st['mean']:+.4f} CI [{ci[0]:+.3f},{ci[1]:+.3f}] "
            f"sign p {st['sign_test']['p_two_sided']:.2e} | manual-vs-battery "
            f"dev {dev_x:.2e}/{dev_clean_x:.2e}")

    # ---------------- adjudication (registered precedence)
    s1 = S["install60"]["S1_addr"]
    s2 = S["install60"]["S2_field"]
    s3i = s3["install60"]["suppression_diff_nAnF_minus_AnF"]

    def fires_positive(cb):
        return (np.isfinite(cb["pearson_r"]) and cb["pearson_r"] > 0
                and cb["pearson_ci95"][0] > 0)

    def fires_negative(cb):
        return (np.isfinite(cb["pearson_r"]) and cb["pearson_r"] < 0
                and cb["pearson_ci95"][1] < 0)

    s3_fires = bool(s3i["positive_means_suppression"])
    s1_pos, s2_pos = fires_positive(s1), fires_positive(s2)
    s1_neg, s2_neg = fires_negative(s1), fires_negative(s2)

    if s3_fires:
        winner = "M3_TRAINED_INHIBITOR"
        clause = ("S3 PRECEDENCE FIRES: suppression-without-field — the "
                  "address's residual presence suppresses p(Z) even when "
                  "the field's values are dead; the brake is functional "
                  "(learned inhibition), not dilution or crosstalk. "
                  "Correlations reported as covariation texture.")
    else:
        cands = []
        if s1_pos:
            cands.append(("M1_READ_BUDGET_DILUTION", abs(s1["pearson_r"]), "S1"))
        if s2_pos:
            cands.append(("M2_DUPLICATE_INTERFERENCE", abs(s2["pearson_r"]), "S2"))
        if len(cands) >= 2:
            mech, _, sig = max(cands, key=lambda c: c[1])
            winner = mech
            clause = (f"BOTH correlation signatures fire positive; the "
                      f"registered largest-|r| rule gives the win to {sig} "
                      f"({mech}) — with the complementarity caveat that "
                      f"addr/field masses are mechanically anti-correlated, "
                      f"so this is one sign-discriminator measured twice.")
        elif len(cands) == 1:
            winner = cands[0][0]
            clause = (f"Exactly one mechanism-consistent correlation fires "
                      f"({cands[0][2]}): {cands[0][0]}.")
        else:
            winner = "FOURTH_STORY_NEEDED"
            clause = ("ALL NULL: no mechanism-consistent correlation CI "
                      "excludes 0 and S3 shows no suppression-without-field "
                      "(CI covering 0). The brake wants a mechanism W006 did "
                      "not name — reported honestly; texture below is the "
                      "raw material for the fourth story.")
        if fires_negative(s1):
            clause += " [S1 significant but ANTI-signed: M1 direction falsified.]"
        if fires_negative(s2):
            clause += " [S2 significant but ANTI-signed: M2 direction falsified.]"

    adjudication = {
        "s3_fires": s3_fires,
        "s1_positive_ci_excludes_0": bool(s1_pos),
        "s2_positive_ci_excludes_0": bool(s2_pos),
        "s1_anti_signed": bool(s1_neg), "s2_anti_signed": bool(s2_neg),
        "winner": winner, "clause": clause,
        "s3_ci_covers_0": bool(not s3i["ci_excludes_0"]),
    }
    log("=" * 78)
    log(f"E114 VERDICT: {winner}")
    log(f"  S1/M1 brake~addr_mass: r {s1['pearson_r']:+.3f} CI {s1['pearson_ci95']}"
        f" (rho {s1['spearman_rho']:+.3f}) -> "
        f"{'FIRES' if s1_pos else ('ANTI-SIGNED' if s1_neg else 'null')}")
    log(f"  S2/M2 brake~field_mass: r {s2['pearson_r']:+.3f} CI {s2['pearson_ci95']}"
        f" (rho {s2['spearman_rho']:+.3f}) -> "
        f"{'FIRES' if s2_pos else ('ANTI-SIGNED' if s2_neg else 'null')}")
    log(f"  S3/M3 ADDR_ONLY {s3['install60']['means']['AnF']:.4f} vs NEITHER "
        f"{s3['install60']['means']['nAnF']:.4f}: diff "
        f"{s3i['mean']:+.4f} CI {s3i['ci95']} -> "
        f"{'SUPPRESSES (M3)' if s3_fires else 'no suppression-without-field'}")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e114_brake_signature",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "W006 (three mechanisms, three named signatures; bars "
                        "verbatim below; docstring written before compute)",
        "registered_bars": REGISTERED_BARS,
        "question": "why is the mature memory's address row SUPPRESSIVE of "
                    "its own fact (e113: 0.785 -> 0.917 at g0 under D129)? "
                    "M1 read-budget dilution vs M2 duplicate-interference "
                    "vs M3 trained inhibitor",
        "net": f"e109 arm (a) rebuilt: runs/checkpoints/{INSTALLED_CK} "
               f"+ 300-step jittered replay (seed {CONS_SEED}), CPU",
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "jitters": list(JITTERS),
                     "battery_construction": "ctx = train_text[p-PRE-j:p], "
                                             "readout p(Z) at last position "
                                             "(wpe row 129+j); brake = "
                                             "p(Z)_D129 - p(Z)_none per ctx",
                     "field_band_vzero": "context positions 1..128, "
                                         "whole-position V-zero (all layers, "
                                         "all heads, all queries; K untouched; "
                                         "row 0 and coord 129 untouched)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_INST": G_INST, "G_REPRO": G_REPRO,
                  "G_FWD": G_FWD, "G_VZ": G_VZ, "G_SURG": {
                      "d129": gate_d129, "d_all_addresses": gates_all,
                      "jit_d129_g4": jit_mirror["g+4"]["gate_wpe"],
                      "jit_d129_g8": jit_mirror["g+8"]["gate_wpe"]},
                  "mass_complementarity": comp_gate, "G_CE_R0": ce_r0},
        "fine_tune": {"lr": FT_LR, "steps": FT_STEPS, "time_cap_s": FT_TIME_CAP,
                      "batch": f"{NAME_BS} install + {ANCH_BS} anchor "
                               f"({ANCH_BS // 2} paired + {ANCH_BS // 2} random)",
                      "loss": "e043 token-level union CE",
                      "optimizer": "AdamW (0.9,0.95) wd 0.1 clip 1.0 constant lr",
                      "seed": CONS_SEED, "device": "cpu",
                      "steps_ran": a["steps_ran"], "traj": a["traj"]},
        "wpe_row_probes": wpe_probes,
        "battery_none_means_g": {f"g{g:+d}": none_means[g] for g in GEO_ORDER},
        "attention_masses": {
            "definition": "decision row (last position, wpe 129) softmax "
                          "attention, summed over all 36 layer-heads, CLEAN "
                          "consolidated net, original geometry",
            "install60": {k: v.tolist() for k, v in masses["install60"].items()},
            "held30": {k: v.tolist() for k, v in masses["held30"].items()},
            "per_layer_addr_mass_mean": per_layer_addr,
            "per_layer_field_mass_mean": per_layer_field,
            "mean_profile_positions_0_129": mean_profile.tolist(),
        },
        "brake": {"install60": brake["install60"].tolist(),
                  "held30": brake["held30"].tolist(),
                  "d_all_variant_install60": brake_all["install60"].tolist(),
                  "install60_mean": float(brake["install60"].mean()),
                  "install60_ci95": list(boot_mean_ci(brake["install60"], seed=0)),
                  "d_all_install60_mean": float(brake_all["install60"].mean()),
                  "held30_mean": float(brake["held30"].mean())},
        "correlations": S,
        "s3_two_by_two": s3,
        "jit_mirrors_report_only": {k: {kk: vv for kk, vv in v.items()
                                        if kk != "gate_wpe"}
                                    for k, v in jit_mirror.items()},
        "adjudication": adjudication,
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net0.num_params()), "device": "cpu",
                   "torch_threads": torch.get_num_threads(), "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: brake_signature.png
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    inst = "install60"

    def fit_line(ax, x, y, col):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ax.scatter(x, y, s=30, alpha=0.75, color=col, edgecolor="k",
                   linewidth=0.3)
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=col, lw=2, ls="--")

    # (0,0) S1 scatter
    ax = axes[0, 0]
    fit_line(ax, masses[inst]["addr_mass"], brake[inst], "tab:blue")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("routing mass on address coordinate 129 (36-lh sum)")
    ax.set_ylabel("brake  p(Z)$_{D129}$ - p(Z)$_{none}$  (per context)")
    s1c = S[inst]["S1_addr"]
    ax.set_title(f"S1/M1 read-budget dilution: brake ~ addr mass\n"
                 f"Pearson r {s1c['pearson_r']:+.3f} CI {s1c['pearson_ci95']}"
                 f" | rho {s1c['spearman_rho']:+.3f} "
                 f"({'FIRES' if s1_pos else 'null' if not s1_neg else 'ANTI'})",
                 fontsize=9.5)

    # (0,1) S2 scatter
    ax = axes[0, 1]
    fit_line(ax, masses[inst]["field_mass"], brake[inst], "tab:orange")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("field read strength: body-band (pos 1..128) attention mass")
    ax.set_ylabel("brake (per context)")
    s2c = S[inst]["S2_field"]
    ax.set_title(f"S2/M2 duplicate-interference: brake ~ field mass\n"
                 f"Pearson r {s2c['pearson_r']:+.3f} CI {s2c['pearson_ci95']}"
                 f" | rho {s2c['spearman_rho']:+.3f} "
                 f"({'FIRES' if s2_pos else 'null' if not s2_neg else 'ANTI'})",
                 fontsize=9.5)

    # (0,2) decision-row attention profile
    ax = axes[0, 2]
    ax.plot(np.arange(len(mean_profile)), mean_profile, color="dimgray",
            lw=1.4, label="mean 36-lh mass per position")
    ax.axvline(ADDR_COORD, color="tab:blue", lw=2.2,
               label=f"address coord 129 (mean {mean_profile[ADDR_COORD]:.2f})")
    ax.axvline(0, color="tab:green", lw=1.4,
               label=f"row 0 scaffold (mean {mean_profile[0]:.2f})")
    ax.axvspan(1, PRE - 2, color="tab:orange", alpha=0.12,
               label=f"field band 1..128 (mean/pos {mean_profile[1:PRE-1].mean():.2f})")
    ax.set_xlabel("context position (decision row reads these)")
    ax.set_ylabel("mean attention mass (36-lh sum)")
    ax.set_yscale("log")
    ax.legend(fontsize=7.5)
    ax.set_title("where the decision row routes (clean consolidated net, g0)",
                 fontsize=9.5)

    # (1,0) S3 2x2
    ax = axes[1, 0]
    conds = [("A+F\nclean", s3[inst]["means"]["AF"], "tab:gray"),
             ("-A+F\nD129", s3[inst]["means"]["nAF"], "tab:purple"),
             ("A+-F\nADDR_ONLY", s3[inst]["means"]["AnF"], "tab:red"),
             ("-A+-F\nNEITHER", s3[inst]["means"]["nAnF"], "tab:green")]
    for k, (lbl, v, col) in enumerate(conds):
        ax.bar(k, v, 0.62, color=col, edgecolor="k", linewidth=0.5)
        ax.text(k, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds], fontsize=8.5)
    ax.set_ylabel("battery p(Z) mean (install-60, g0)")
    st = s3[inst]["suppression_diff_nAnF_minus_AnF"]
    ax.annotate("", xy=(3, s3[inst]["means"]["nAnF"]),
                xytext=(2, s3[inst]["means"]["AnF"]),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.6))
    ax.set_title(f"S3/M3 address-only re-exposure (field V-zeroed 1..128)\n"
                 f"suppression NEITHER-ADDR_ONLY {st['mean']:+.4f} CI "
                 f"{st['ci95']} sign p {st['sign_test']['p_two_sided']:.1e} "
                 f"-> {'M3 SUPPORTED' if s3_fires else 'no suppression'}",
                 fontsize=9.5)
    ax.set_ylim(0, max(1.05, max(c[1] for c in conds) * 1.25))

    # (1,1) correlation summary with CIs
    ax = axes[1, 1]
    rows = [("S1 addr (M1)", S["install60"]["S1_addr"], "tab:blue"),
            ("S2 field (M2)", S["install60"]["S2_field"], "tab:orange"),
            ("S2 alt no-addr expr\n(CIRCULAR)", S["install60"]["S2_alt_noaddr_expr"],
             "moccasin"),
            ("held30: S1 addr", S["held30"]["S1_addr"], "lightblue"),
            ("held30: S2 field", S["held30"]["S2_field"], "navajowhite"),
            ("row0 mass control", S["install60"]["row0"], "tab:gray")]
    for k, (lbl, cb, col) in enumerate(rows):
        r = cb["pearson_r"]
        lo, hi = cb["pearson_ci95"]
        ax.bar(k, r, 0.6, color=col, edgecolor="k", linewidth=0.4)
        ax.errorbar(k, r, yerr=[[max(0, r - lo)], [max(0, hi - r)]],
                    fmt="none", ecolor="k", capsize=4, lw=1.2)
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.2)
    ax.set_ylabel("Pearson r vs per-context brake")
    ax.set_title("correlation summary (95% bootstrap CI, contexts resampled);\n"
                 "M1 predicts r>0 blue, M2 predicts r>0 orange", fontsize=9.5)

    # (1,2) verdict
    ax = axes[1, 2]
    ax.axis("off")
    vlines = [
        "REGISTERED BARS (frozen pre-compute):",
        "  S3 precedence: ADDR_ONLY < NEITHER, paired CI excl 0 => M3",
        "  else largest |r| among S1/S2 with CI excl 0 AND positive sign",
        "  all null => FOURTH-STORY NEEDED",
        f"",
        f"MEANS: brake(g0) {metrics['brake']['install60_mean']:+.4f} "
        f"CI {metrics['brake']['install60_ci95']}",
        f"  S1 r {s1c['pearson_r']:+.3f} CI {s1c['pearson_ci95']}"
        f" | S2 r {s2c['pearson_r']:+.3f} CI {s2c['pearson_ci95']}",
        f"  S3 diff {st['mean']:+.4f} CI {st['ci95']} "
        f"(frac supp {st['sign_test']['frac_pos']:.2f})",
        f"  complementarity: addr {masses[inst]['addr_mass'].mean():.2f} + "
        f"row0 {masses[inst]['row0_mass'].mean():.2f} + field "
        f"{masses[inst]['field_mass'].mean():.2f} = 36 (L*H)",
        "",
        f"WINNER: {winner}",
    ] + [f"  {wd}" for wd in _wrap(clause, 62)]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.052, tx, fontsize=7.6, va="top",
                family="monospace")

    fig.suptitle(f"E114 — W006 brake-signature probe: why the address "
                 f"suppresses its own fact -> {winner}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "brake_signature.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'brake_signature.png'}")
    log(f"total {time.time() - T0:.1f}s")


def _wrap(text, width):
    import textwrap
    return textwrap.wrap(text, width=width)


if __name__ == "__main__":
    main()
