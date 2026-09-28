"""E115 — the T067-registered GRADED-ABLATION RESURRECTION CHECK for M3
(the one caveat between the coordinate-local fourth story and safety).
CPU-only.

WHY (T067, verbatim spirit): e114 killed all three W006 mechanisms — but
M3's kill had a caveat the agent flagged: S3's whole-band V-zero left NO
DYNAMIC RANGE (A+-F p(Z) ~ 2.1e-5 vs -A+-F ~ 1.9e-5 — both at the floor,
so "no suppression without the field" could not distinguish a trained
inhibitor that needs SOME field to act from a memory that simply cannot be
expressed at all). The fourth story (coordinate-local state modulation:
the address row's state at its own coordinate slightly reshapes the
decision there — a learned "this-was-installed-here" tag) is not safe
until M3 is dead along the graded axis too.

THE CHECK (registered): the brake is +0.132 (deleting the address RAISES
expression: 0.785 -> 0.917). M3 (trained inhibitor) predicts suppression
EVEN WITH THE FIELD WEAKENED — it only needs enough dynamic range for the
decision to be made at all. The graded ablation gives S3 that range.

ARMS (all at the ORIGINAL GEOMETRY — 130-token contexts, decision = wpe
row 129; battery = install-60, standard p(Z) readout at the last
position; held-30 mirror report-only):
  (a) A+F      — address intact, full field (the 0.785 reference);
  (b) -A+F     — no address (wpe row 129 zeroed, e113/e114 D129), full
                 field (the 0.917 brake reference);
  (c) A+graded — address INTACT, field scaled to r in {0.1, 0.25, 0.56}:
                 the e102/e110 scaling on the band's V, ENTRY-WISE
                 (V <- r*V at context positions 1..128, every layer,
                 every head, all queries; direction preserved exactly,
                 cos = 1, norm ratio = r; K never touched; row 0 and the
                 address coordinate 129 never touched);
  (d) -A+graded — the matched controls (row 129 zeroed + the same r);
  (e) floor     — A+no-field and -A+no-field (the e114 S3 cells,
                 whole-band V-zero), for continuity with e114.

REGISTERED BARS (frozen before compute):
  M3 RESURRECTED (the fourth story falls): p(Z) with address INTACT is
    BELOW the matched no-address cell at >= 2 of the 3 graded r levels
    WITH CI separation — i.e. the paired per-context brake
    brake(r) = p(Z)_(-A,r) - p(Z)_(A,r) is positive with its bootstrap
    95% CI excluding 0 at >= 2 of r in {0.1, 0.25, 0.56}. Texture rider:
    M3's strong form also wants the gap to WIDEN downward (brake(r)
    increasing as r falls) — reported, not required.
    [Operationalization registered: "CI separation" = the PAIRED
    bootstrap CI of brake(r) excludes 0 (primary, the lab's e114 S3
    convention); unpaired non-overlap of the two arm means' CIs is
    reported as a secondary flag. FLOOR EPSILON (added after the smoke
    instrument check, BEFORE the registered compute; see deviations): a
    brake counts as separating only if its mean also exceeds 0.001 —
    at the p ~ 1e-5 floor, float dust differences of ~1e-7 carry
    CIs that formally exclude 0 (observed in smoke) and would fire the
    bar on numerics, not suppression.]
  COORDINATE-LOCAL STORY SAFE: the brake SHRINKS TOWARD 0 OR INVERTS as
    r falls — the paired step brake(1.0) - brake(0.1) is positive with
    CI excluding 0 AND brake(0.1) itself is no longer significantly
    positive (CI covers 0 or lies below it): the address HELPS when the
    field is weak — it is content, not inhibitor. (The r=0 floor cells
    are continuity anchors, not adjudicated on — no dynamic range there.)
  MIXED: anything else — report the texture honestly.

BASE (registered): rebuild e109 arm (a)'s consolidated net EXACTLY as
e113/e114 did — runs/checkpoints/e048_repro.pt + the 300-step
jittered-replay fine-tune (seed 10901, offsets {-8,-4,0,+4,+8}, batch 16
install + 16 anchor (8 paired + 8 random), e043 token-level union CE,
AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0), CPU. GATES: G_SPLICE
(install mix 19/41), G_INST (e048 battery p(Z) 0.556313 +- 0.005),
G_REPRO (post-none install-60 table vs e109's 0.99/0.95/0.78/0.99/0.99,
tol 0.05/cell), G_FWD (manual attention-recording forward vs net(), max
p(Z) dev < 1e-4, clean and wpe-zeroed paths), G_SCALE (the graded
instrument: r=1.0 identity vs clean < 1e-6; r=0 scaling path vs the e114
V-zero path < 1e-7; vector-space identity norm ratio = r +- 1e-6 and
cos = 1; K/row-0/coord-129 untouched by construction), G_CONT
(continuity vs runs/e114/metrics.json: A+F / -A+F / floor cells within
0.02).

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import;
torch.set_num_threads(8)); single run target <= 30 min (the 300-step
rebuild is ~9 min, e113/e114 precedent; every measurement forward is
seconds). No new automations.

Outputs: runs/e115/{metrics.json, graded_ablation.png}.
No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).

Run:  cd lab && python e115_graded_ablation.py        (E115_SMOKE=1 for smoke)
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

# this torch build can report is_available()==True even with the env var
# empty (e110 precedent) — force CPU before common sniffs the device
torch.cuda.is_available = lambda: False               # noqa: E404
torch.cuda.device_count = lambda: 0                   # noqa: E404

torch.set_num_threads(8)                              # 8 threads max (operator)

import torch.nn.functional as F                        # noqa: E402

import json as _json                                   # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E115_SMOKE") == "1"
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
# the reproduction gate for this rebuild
E109_REF_NONE = {-8: 0.9924036860466003, -4: 0.9496323466300964,
                 0: 0.776076078414917, 4: 0.9848979115486145,
                 8: 0.9854525923728943}
G_REPRO_TOL = 0.05                        # per-cell tolerance
G_FWD_TOL = 1e-4                          # manual-forward instrument gate

# the graded field band (frozen, e114's S3 band): body content rows of the
# 130-token context; row 0 excluded (generic scaffold, e113 never-touch),
# address coordinate 129 excluded by definition (it is the A contrast)
FIELD_BAND = list(range(1, PRE - 1))               # positions 1..128
ADDR_COORD = PRE - 1                               # position 129

# the graded retention ladder (tasking, frozen — e102/e110's rungs below 1)
RS_GRADED = [0.56, 0.25, 0.1]
RS_ALL = [1.0] + RS_GRADED + [0.0]                 # anchors + graded + floor

# e114 continuity references (loaded at runtime if present)
E114_METRICS = E43.REPO / "runs" / "e114" / "metrics.json"
G_CONT_TOL = 0.02

BOOT_N = 1000
FLOOR_EPS = 0.001      # brake below this = instrument dust at the p~1e-5 floor

REGISTERED_BARS = {
    "m3_resurrected": "p(Z) with address INTACT < matched no-address at "
                      ">= 2 of the 3 graded r levels with CI separation "
                      "(paired brake(r) = p_nA - p_A positive, bootstrap CI "
                      "excluding 0) => M3 RESURRECTED, the fourth story falls",
    "ci_separation_operationalization": "primary = PAIRED bootstrap CI of "
                                        "brake(r) excludes 0 (e114 S3 "
                                        "convention); unpaired non-overlap of "
                                        "arm-mean CIs reported as secondary; "
                                        "FLOOR EPSILON: the brake mean must "
                                        "also exceed 0.001 to count (float "
                                        "dust at the p~1e-5 floor carries "
                                        "formally-separated CIs — observed "
                                        "in the smoke instrument check, "
                                        "epsilon added BEFORE the registered "
                                        "compute)",
    "coordinate_local_safe": "brake(1.0) - brake(0.1) paired CI positive "
                             "excluding 0 AND brake(0.1) not significantly "
                             "positive (CI covers 0 or below) => the brake "
                             "shrinks toward 0 / inverts as r falls; the "
                             "address helps when the field is weak — content, "
                             "not inhibitor",
    "mixed": "anything else — report the texture honestly",
    "floor_note": "the r=0 cells (e114 S3 floor) are continuity anchors only "
                  "(no dynamic range); never adjudicated on",
    "m3_texture_rider": "M3's strong form wants the gap to WIDEN downward "
                        "(brake increasing as r falls) — reported, not "
                        "required for the bar",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-only (CUDA_VISIBLE_DEVICES=-1, 8 threads) per operator instruction; "
    "e109's arm (a) was fine-tuned on cuda — same seed/recipe/batches, but "
    "float arithmetic differs by device, so the rebuild is gated (G_REPRO, "
    "0.05/cell) against e109's post-none table rather than bit-compared "
    "(e113/e114's CPU rebuilds matched to <0.01/cell).",
    "e109's 180 s wall cap was a GPU thermal guard — raised to 1500 s (CPU "
    "envelope) so the registered 300 steps complete; the 300-step count and "
    "every other recipe element are verbatim (e113/e114 precedent).",
    "'CI separation' in the tasking is operationalized as the PAIRED "
    "bootstrap CI of the per-context brake excluding 0 (primary; the lab's "
    "e114 S3 convention, and the tighter/more standard test); the unpaired "
    "non-overlap reading is computed and reported as a secondary flag.",
    "The graded band is frozen at context positions 1..128 with the e102/"
    "e110 entry-wise V <- r*V semantics (cos = 1, norm ratio = r, K never "
    "touched): row 0 excluded (generic window scaffold; e113 never touched "
    "it), address coordinate 129 excluded by definition (it is the A "
    "contrast). The r=0 floor is realized with e114's exact whole-band "
    "V-zero path (continuity), with the r=0.0 scaling path cross-checked "
    "against it (G_SCALE).",
    "FLOOR EPSILON (pre-compute addition after the smoke instrument check): "
    "a brake counts as CI-separated only if |mean| > 0.001 as well — in "
    "smoke, the r=0.1 floor cell (p ~ 1e-5) showed a +3e-7 paired diff "
    "whose bootstrap CI formally excludes 0; counting that as suppression "
    "would fire the M3 bar on float dust. Registered here before the real "
    "compute; flagged rather than silently applied.",
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
    G_SURG convention)."""
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
                   vzero=None, vscale=None, wpe_zero_rows=(), bs=30,
                   vstats=False):
    """Manual TinyGPT forward (e114's instrument, extended with the graded
    V-scaling): returns p(Z) at the last position per row (N,).

    vzero:   (N, T) bool — e114/e075 whole-position V-zero (v := 0 there,
             every layer, every head, all queries; K untouched).
    vscale:  (band: list[int], r: float) — the e102/e110 graded ablation:
             v := r * v ENTRY-WISE at the band positions, every layer, every
             head, all queries; direction preserved exactly (cos = 1), norm
             ratio = r; K never touched; all other positions untouched.
    wpe_zero_rows: eval-time wpe row zeroing (same arithmetic as deleted_wpe,
             in-forward — the -A arm).
    vstats:  collect vector-space identity stats of the scaling on the FIRST
             batch of the FIRST layer (norm ratio dev, min cos) when vscale
             is active."""
    L = net.cfg.n_layer
    wpe_w = net.wpe.weight
    if wpe_zero_rows:
        w2 = wpe_w.clone()
        for r in wpe_zero_rows:
            w2[r] = 0.0
        wpe_w = w2
    outs = []
    stats = {"norm_ratio_dev": None, "min_cos": None}
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
            d = C // net.cfg.n_head
            q, k, v = qkv.split(C, dim=2)
            q = q.view(N, T, net.cfg.n_head, d).transpose(1, 2)
            k = k.view(N, T, net.cfg.n_head, d).transpose(1, 2)
            v = v.view(N, T, net.cfg.n_head, d).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
            att = att.masked_fill(causal, float("-inf"))
            probs = torch.softmax(att, dim=-1)
            if vz is not None and bool(vz.any()):
                v = v.masked_fill(vz[:, None, :, None], 0.0)
            if vscale is not None:
                band, r = vscale
                cols = torch.as_tensor(band, dtype=torch.long)
                if vstats and li == 0 and i == 0:
                    old = v[:, :, cols, :].clone()
                    v = v.clone()
                    v[:, :, cols, :] = r * old
                    nrm = old.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    nnew = v[:, :, cols, :].norm(dim=-1, keepdim=True)
                    stats["norm_ratio_dev"] = float(
                        ((nnew / nrm) - r).abs().max())
                    cos = (old * v[:, :, cols, :]).sum(-1) / \
                        (nrm.squeeze(-1) * nnew.squeeze(-1)).clamp_min(1e-12)
                    stats["min_cos"] = float(cos.min())
                else:
                    v = v.clone()
                    v[:, :, cols, :] = r * v[:, :, cols, :]
            y = (probs @ v).transpose(1, 2).reshape(N, T, C)
            x = x + blk.attn.c_proj(y)
            x = x + blk.mlp(blk.ln2(x))
        pr = F.softmax(net.lm_head(net.ln_f(x[:, -1, :])), -1)[:, zid]
        outs.append(pr)
    p = torch.cat(outs)
    return (p, stats) if vstats else (p, None)


# ------------------------------------------------------------------ statistics

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
    rd = run_dir("e115_smoke" if SMOKE else "e115")
    assert not torch.cuda.is_available(), "CPU-only violated: CUDA visible"
    log(f"E115 GRADED-ABLATION resurrection check for M3 (T067; "
        f"smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e065/e091/e109/e113/e114 verbatim)
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
    # (e109/e113/e114 construction verbatim — only needed to drive the rebuild)
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

    # batteries: ctx = train_text[p-PRE:p], readout p(Z) at the last position
    # (wpe row 129) — ORIGINAL GEOMETRY ONLY. install60 primary, held30 mirror.
    bat_ids = {}
    for tag, occ in (("install60", install_occ), ("held30", held_occ)):
        cs = [train_text[p - PRE: p] for p, _ in occ]
        bat_ids[tag] = torch.stack([corpus.encode(c) for c in cs])
    f_eval_ids = bat_ids["install60"]                          # the G_INST battery

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
        raise RuntimeError("instrument broken vs e065/e091/e109/e113/e114")
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
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        ids_j = torch.stack([corpus.encode(c) for c in cs])
        none_means[j] = battery_cell(evl, ids_j, zid)["mean_pz"]
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

    # ---------------- THE GRADED GRID
    # arms at the original geometry: A (address intact) vs -A (wpe row 129
    # zeroed) x r in {1.0, 0.56, 0.25, 0.1, 0(floor, e114 V-zero path)}.
    vz = {}
    for tag in ("install60", "held30"):
        n = bat_ids[tag].shape[0]
        m = torch.zeros(n, PRE, dtype=torch.bool)
        m[:, FIELD_BAND[0]:FIELD_BAND[-1] + 1] = True
        vz[tag] = m

    def arm_p(tag, r, addr, floor_vzero=False):
        """p(Z) per context for one arm. r=1.0 clean; graded r via vscale;
        floor via e114's exact vzero path."""
        if floor_vzero:
            return manual_forward(evl, bat_ids[tag], zid,
                                  vzero=vz[tag],
                                  wpe_zero_rows=() if addr else (129,))[0]
        vs = None if r == 1.0 else (FIELD_BAND, r)
        return manual_forward(evl, bat_ids[tag], zid, vscale=vs,
                              wpe_zero_rows=() if addr else (129,))[0]

    # G_SCALE instrument checks (on install60, A side)
    p_id, _ = manual_forward(evl, bat_ids["install60"], zid,
                             vscale=(FIELD_BAND, 1.0))
    dev_r1 = float((p_id - p_clean_manual).abs().max())
    p_r0_scale, _ = manual_forward(evl, bat_ids["install60"], zid,
                                   vscale=(FIELD_BAND, 0.0))
    p_r0_vz, _ = manual_forward(evl, bat_ids["install60"], zid, vzero=vz["install60"])
    dev_r0 = float((p_r0_scale - p_r0_vz).abs().max())
    _, vst = manual_forward(evl, bat_ids["install60"], zid,
                            vscale=(FIELD_BAND, 0.56), vstats=True)
    G_SCALE = {
        "r1_identity_vs_clean_max_dev": dev_r1,
        "r0_scale_vs_vzero_max_dev": dev_r0,
        "vscale_norm_ratio_dev_r056": vst["norm_ratio_dev"],
        "vscale_min_cos_r056": vst["min_cos"],
        "norm_tol": 1e-6, "identity_tol": 1e-6, "r0_tol": 1e-7,
        "k_untouched": True, "row0_untouched": True,
        "addr_coord_129_untouched": True,
        "band": [FIELD_BAND[0], FIELD_BAND[-1]],
        "n_positions_per_ctx": len(FIELD_BAND),
    }
    G_SCALE["pass"] = bool(dev_r1 < 1e-6 and dev_r0 < 1e-7
                           and vst["norm_ratio_dev"] < 1e-6
                           and vst["min_cos"] > 1.0 - 1e-6)
    log(f"G_SCALE graded instrument: r=1.0 identity dev {dev_r1:.2e} | r=0 "
        f"scale-vs-vzero dev {dev_r0:.2e} | norm-ratio dev "
        f"{vst['norm_ratio_dev']:.2e} | min cos {vst['min_cos']:.9f} -> "
        f"{'PASS' if G_SCALE['pass'] else 'FAIL'}")
    if not G_SCALE["pass"]:
        raise RuntimeError("graded V-scaling instrument broken")

    # the grid itself: grid[tag][r] = {"A": [...], "nA": [...]}
    grid = {}
    for tag in ("install60", "held30"):
        grid[tag] = {}
        for r in RS_ALL:
            floor = (r == 0.0)
            pa = arm_p(tag, r, addr=True, floor_vzero=floor)
            pn = arm_p(tag, r, addr=False, floor_vzero=floor)
            grid[tag][r] = {"A": [float(v) for v in pa.tolist()],
                            "nA": [float(v) for v in pn.tolist()]}
        log(f"[{tag}] graded grid done: "
            + " | ".join(f"r={r:g}: A {np.mean(grid[tag][r]['A']):.4f} "
                         f"-A {np.mean(grid[tag][r]['nA']):.4f}"
                         for r in RS_ALL))

    # ---------------- per-level stats (install60 primary)
    levels = {}
    for tag in ("install60", "held30"):
        levels[tag] = {}
        for r in RS_ALL:
            pa = np.asarray(grid[tag][r]["A"])
            pn = np.asarray(grid[tag][r]["nA"])
            br = pn - pa                                   # brake(r) per ctx
            ci_br = boot_mean_ci(br, seed=1)
            ci_a = boot_mean_ci(pa, seed=1)
            ci_n = boot_mean_ci(pn, seed=1)
            levels[tag][r] = {
                "mean_p_A": float(pa.mean()), "mean_p_nA": float(pn.mean()),
                "ci_p_A": list(ci_a), "ci_p_nA": list(ci_n),
                "brake_mean": float(br.mean()), "brake_ci95": list(ci_br),
                "brake_ci_excludes_0_positive": bool(br.mean() > FLOOR_EPS
                                                     and ci_br[0] > 0),
                "brake_ci_excludes_0_negative": bool(br.mean() < -FLOOR_EPS
                                                     and ci_br[1] < 0),
                "at_floor": bool(min(pa.mean(), pn.mean()) < 0.01),
                "unpaired_ci_separation": bool(ci_a[1] < ci_n[0]),
                "sign_test": sign_test(br),
            }
    inst = levels["install60"]

    # ---------------- trend stats (paired step-downs, install60 primary)
    trend = {}
    pairs = [(1.0, 0.56), (0.56, 0.25), (0.25, 0.1), (0.56, 0.1), (1.0, 0.1)]
    for hi, lo in pairs:
        d = (np.asarray(grid["install60"][lo]["nA"])
             - np.asarray(grid["install60"][lo]["A"])) - \
            (np.asarray(grid["install60"][hi]["nA"])
             - np.asarray(grid["install60"][hi]["A"]))
        ci = boot_mean_ci(d, seed=0)
        trend[f"brake({lo:g})-brake({hi:g})"] = {
            "mean": float(d.mean()), "ci95": list(ci),
            "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0),
            "positive_means_gap_widens_as_r_falls": bool(d.mean() > 0
                                                         and ci[0] > 0),
            "negative_means_brake_shrinks_as_r_falls": bool(d.mean() < 0
                                                            and ci[1] < 0),
        }
    for k, v in trend.items():
        log(f"trend {k}: {v['mean']:+.4f} CI [{v['ci95'][0]:+.3f},"
            f"{v['ci95'][1]:+.3f}]")

    # ---------------- G_CONT: continuity vs e114's shipped cells
    g_cont = {"ref": str(E114_METRICS), "tol": G_CONT_TOL, "cells": {},
              "pass": True}
    if E114_METRICS.exists():
        e114 = _json.load(open(E114_METRICS))
        s3 = e114["s3_two_by_two"]["install60"]["means"]
        refs = {"A+F_full_field": s3["AF"], "nAF_D129_full_field": s3["nAF"],
                "A_no_field_floor": s3["AnF"], "nA_no_field_floor": s3["nAnF"]}
        mine = {"A+F_full_field": inst[1.0]["mean_p_A"],
                "nAF_D129_full_field": inst[1.0]["mean_p_nA"],
                "A_no_field_floor": inst[0.0]["mean_p_A"],
                "nA_no_field_floor": inst[0.0]["mean_p_nA"]}
        for k in refs:
            g_cont["cells"][k] = {"this_run": mine[k], "e114": refs[k],
                                  "diff": mine[k] - refs[k]}
            g_cont["pass"] &= bool(abs(mine[k] - refs[k]) < G_CONT_TOL)
        g_cont["pass"] = bool(g_cont["pass"])
        log(f"G_CONT vs e114 floor/full cells (tol {G_CONT_TOL}): "
            + " ".join(f"{k} {mine[k]:.4f}/{refs[k]:.4f}"
                       for k in refs)
            + f" -> {'PASS' if g_cont['pass'] else 'FAIL'}")
    else:
        g_cont["pass"] = False
        g_cont["note"] = "runs/e114/metrics.json missing (report-only)"
        log("G_CONT: runs/e114/metrics.json missing (continuity unchecked)")

    # ---------------- adjudication (registered, frozen)
    m3_count = sum(1 for r in RS_GRADED
                   if inst[r]["brake_ci_excludes_0_positive"])
    m3_count_raw = sum(1 for r in RS_GRADED
                       if inst[r]["brake_ci95"][0] > 0
                       and inst[r]["brake_mean"] > 0)
    m3_unpaired_count = sum(1 for r in RS_GRADED
                            if inst[r]["unpaired_ci_separation"])
    m3_fires = bool(m3_count >= 2)
    shrink = trend["brake(0.1)-brake(1)"]
    cl_fires = bool(shrink["negative_means_brake_shrinks_as_r_falls"]
                    and not inst[0.1]["brake_ci_excludes_0_positive"])
    floor_inverts = bool(inst[0.0]["brake_mean"] < 0)
    gap_widens = bool(trend["brake(0.1)-brake(0.56)"]
                      ["positive_means_gap_widens_as_r_falls"])

    if m3_fires and not cl_fires:
        winner = "M3_RESURRECTED"
        clause = ("M3 RESURRECTED: suppression (p_nA > p_A, paired CI "
                  "excluding 0) persists at >= 2 of 3 graded field levels — "
                  "the brake is not a floor artifact; the trained-inhibitor "
                  "story is back and the coordinate-local fourth story "
                  "FALLS.")
    elif cl_fires and not m3_fires:
        winner = "COORDINATE_LOCAL_STORY_SAFE"
        clause = ("COORDINATE-LOCAL STORY SAFE: the brake shrinks "
                  "significantly from full field to the weakest graded "
                  "field and is no longer significantly positive there "
                  "(shrinks toward 0 or inverts) — the address HELPS when "
                  "the field is weak: content, not inhibitor. M3 stays "
                  "dead; the fourth story stands."
                  + (" Floor rider: brake(0) inverts (sign)." if floor_inverts
                     else ""))
    elif m3_fires and cl_fires:
        winner = "MIXED (both bars partially fire)"
        clause = ("MIXED: significant suppression at >= 2 graded levels "
                  "(M3 leg) AND significant shrinkage with no significant "
                  "positive at r=0.1 (coordinate-local leg) — the brake "
                  "persists into weak fields but also demonstrably fades; "
                  "report the texture, no clean kill.")
    else:
        winner = "MIXED (no clean bar)"
        clause = ("MIXED: neither registered bar fires cleanly — suppression "
                  "appears at < 2 graded levels AND the shrinkage clause is "
                  "incomplete; report the texture honestly.")

    adjudication = {
        "m3_graded_positive_count": m3_count,
        "m3_graded_positive_count_raw_no_epsilon": m3_count_raw,
        "m3_unpaired_separation_count": m3_unpaired_count,
        "floor_epsilon": FLOOR_EPS,
        "m3_fires": m3_fires,
        "brake_shrinks_1_to_01": shrink["negative_means_brake_shrinks_as_r_falls"],
        "brake_positive_at_r01": inst[0.1]["brake_ci_excludes_0_positive"],
        "coordinate_local_fires": cl_fires,
        "floor_inverts_sign": floor_inverts,
        "gap_widens_downward_056_to_01": gap_widens,
        "winner": winner, "clause": clause,
    }
    log("=" * 78)
    log(f"E115 VERDICT: {winner}")
    for r in RS_ALL:
        L = inst[r]
        log(f"  r={r:<4g}: p_A {L['mean_p_A']:.4f} [{L['ci_p_A'][0]:.3f},"
            f"{L['ci_p_A'][1]:.3f}] | p_nA {L['mean_p_nA']:.4f} "
            f"[{L['ci_p_nA'][0]:.3f},{L['ci_p_nA'][1]:.3f}] | brake "
            f"{L['brake_mean']:+.4f} CI [{L['brake_ci95'][0]:+.3f},"
            f"{L['brake_ci95'][1]:+.3f}] "
            f"{'SUPPRESSES' if L['brake_ci_excludes_0_positive'] else ''}"
            f"{' [FLOOR: no dynamic range]' if L['at_floor'] else ''}")
    log(f"  graded positive count (M3): {m3_count}/3 (raw no-epsilon "
        f"{m3_count_raw}/3; unpaired reading {m3_unpaired_count}/3) | "
        f"shrink 1.0->0.1 "
        f"{shrink['negative_means_brake_shrinks_as_r_falls']} | floor "
        f"inverts {floor_inverts} | gap widens 0.56->0.1 {gap_widens}")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e115_graded_ablation",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "T067's registered follow-up (graded field ablation "
                        "to give S3 dynamic range before the fourth story "
                        "is safe from resurrection of M3); bars verbatim "
                        "below; docstring written before compute",
        "registered_bars": REGISTERED_BARS,
        "question": "does the address brake (deleting the address RAISES "
                    "expression: 0.785 -> 0.917, +0.132) persist when the "
                    "field is GRADEDLY weakened (r in {0.56, 0.25, 0.1} on "
                    "the band's V, entry-wise)? M3 trained-inhibitor "
                    "predicts suppression WITH a weakened field; the "
                    "coordinate-local fourth story predicts the brake fades "
                    "as the field weakens (the address is content, not "
                    "inhibitor)",
        "net": f"e109 arm (a) rebuilt: runs/checkpoints/{INSTALLED_CK} "
               f"+ 300-step jittered replay (seed {CONS_SEED}), CPU",
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "geometry": "ORIGINAL ONLY (g0; decision = wpe row "
                                 "129; battery install-60 primary, held-30 "
                                 "mirror report-only)",
                     "arms": "(a) A+F full field | (b) -A+F (wpe 129 "
                             "zeroed) | (c) A + graded field (V <- r*V "
                             "entry-wise on positions 1..128, all layers, "
                             "all heads, all queries, K untouched) | "
                             "(d) -A + graded | (e) floor: e114 whole-band "
                             "V-zero cells",
                     "graded_rs": RS_GRADED},
        "gates": {"G_SPLICE": G_SPLICE, "G_INST": G_INST, "G_REPRO": G_REPRO,
                  "G_FWD": G_FWD, "G_SCALE": G_SCALE, "G_CONT": g_cont,
                  "G_SURG_d129": gate_d129, "G_CE_R0": ce_r0},
        "fine_tune": {"lr": FT_LR, "steps": FT_STEPS, "time_cap_s": FT_TIME_CAP,
                      "batch": f"{NAME_BS} install + {ANCH_BS} anchor "
                               f"({ANCH_BS // 2} paired + {ANCH_BS // 2} random)",
                      "loss": "e043 token-level union CE",
                      "optimizer": "AdamW (0.9,0.95) wd 0.1 clip 1.0 constant lr",
                      "seed": CONS_SEED, "device": "cpu",
                      "steps_ran": a["steps_ran"], "traj": a["traj"]},
        "battery_none_means_g": {f"g{g:+d}": none_means[g] for g in GEO_ORDER},
        "grid": {tag: {f"r={r:g}": grid[tag][r] for r in RS_ALL}
                 for tag in ("install60", "held30")},
        "levels": {tag: {f"r={r:g}": levels[tag][r] for r in RS_ALL}
                   for tag in ("install60", "held30")},
        "trend_paired_install60": trend,
        "adjudication": adjudication,
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net0.num_params()), "device": "cpu",
                   "torch_threads": torch.get_num_threads(), "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: graded_ablation.png
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))

    # (0,0) p(Z) vs r, address vs no-address (linear, incl. floor)
    ax = axes[0, 0]
    xs = np.array(RS_ALL)
    for key, col, lbl in (("A", "tab:blue", "address INTACT (A)"),
                          ("nA", "tab:red", "no address (-A, wpe 129 zeroed)")):
        ys = [inst[r][f"mean_p_{key}"] for r in RS_ALL]
        lo = [inst[r][f"ci_p_{key}"][0] for r in RS_ALL]
        hi = [inst[r][f"ci_p_{key}"][1] for r in RS_ALL]
        ax.plot(xs, ys, "o-", lw=2.2, ms=7, color=col, label=lbl)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.15)
        ax.plot([0.0], [ys[-1]], "o", ms=9, mfc="none", mec=col, mew=1.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("field retention r on the band's V (entry-wise; 0 = e114 floor)")
    ax.set_ylabel("battery p(Z) mean (install-60, g0)")
    ax.set_title("E115-1 — p(Z) vs field retention: address vs no-address\n"
                 "(open marker = r=0 floor; bands = bootstrap 95% CI)",
                 fontsize=9.5)
    ax.legend(fontsize=8.5, loc="center left")

    # (0,1) graded zoom, log-x
    ax = axes[0, 1]
    xs_g = np.array([1.0] + RS_GRADED)
    for key, col in (("A", "tab:blue"), ("nA", "tab:red")):
        ys = [inst[r][f"mean_p_{key}"] for r in xs_g]
        ax.plot(xs_g, ys, "o-", lw=2.2, ms=8, color=col)
    for r in xs_g:
        ax.annotate(f"A {inst[r]['mean_p_A']:.3f}\n-A {inst[r]['mean_p_nA']:.3f}",
                    (r, max(inst[r]["mean_p_A"], inst[r]["mean_p_nA"])),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xticks([float(v) for v in xs_g])
    ax.set_xticklabels([str(v) for v in xs_g])
    ax.set_xlabel("retention r (log scale)")
    ax.set_ylabel("battery p(Z) mean")
    ax.set_title("E115-2 — graded zoom (dynamic range S3 never had)",
                 fontsize=9.5)

    # (0,2) brake(r) with CIs
    ax = axes[0, 2]
    ys = [inst[r]["brake_mean"] for r in RS_ALL]
    lo = [inst[r]["brake_ci95"][0] for r in RS_ALL]
    hi = [inst[r]["brake_ci95"][1] for r in RS_ALL]
    cols = ["tab:purple"] + ["tab:orange"] * 3 + ["tab:green"]
    ax.errorbar(xs, ys, yerr=[[max(0, y - l) for y, l in zip(ys, lo)],
                              [max(0, h - y) for y, h in zip(ys, hi)]],
                fmt="o-", lw=2.2, ms=8, capsize=5, color="k", zorder=2)
    for x, y, c in zip(xs, ys, cols):
        ax.scatter([x], [y], s=90, color=c, zorder=3, edgecolor="k",
                   linewidth=0.5)
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xlabel("retention r (1.0 full | graded | 0 floor)")
    ax.set_ylabel("brake(r) = p(Z)$_{-A}$ - p(Z)$_A$  (per-context paired)")
    ax.set_title("E115-3 — the brake vs field retention\n"
                 "M3: stays positive with CI separation | coord-local: "
                 "shrinks/inverts as r falls", fontsize=9.5)

    # (1,0) per-context scatter p_A vs p_nA, colored by r
    ax = axes[1, 0]
    rcol = {1.0: "tab:purple", 0.56: "tab:orange", 0.25: "tab:green",
            0.1: "tab:blue", 0.0: "tab:gray"}
    for r in RS_ALL:
        pa = np.asarray(grid["install60"][r]["A"])
        pn = np.asarray(grid["install60"][r]["nA"])
        ax.scatter(pa, pn, s=26, alpha=0.7, color=rcol[r],
                   edgecolor="k", linewidth=0.25,
                   label=f"r={r:g} (brake {inst[r]['brake_mean']:+.3f})")
    lims = [0, 1]
    ax.plot(lims, lims, "k--", lw=1, label="diagonal (no brake)")
    ax.set_xlabel("p(Z) with address INTACT")
    ax.set_ylabel("p(Z) with address DELETED")
    ax.set_xlim(-0.02, 1.02), ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("per-context pairing (install-60); points above the "
                 "diagonal = the address brakes that context", fontsize=9.5)

    # (1,1) trend step-downs
    ax = axes[1, 1]
    keys = list(trend.keys())
    vals = [trend[k]["mean"] for k in keys]
    cis = [trend[k]["ci95"] for k in keys]
    cols_t = ["tab:green" if (v < 0 and c[1] < 0)
              else "tab:red" if (v > 0 and c[0] > 0) else "tab:gray"
              for v, c in zip(vals, cis)]
    for k, (v, c, col) in enumerate(zip(vals, cis, cols_t)):
        ax.bar(k, v, 0.6, color=col, edgecolor="k", linewidth=0.4)
        ax.errorbar(k, v, yerr=[[max(0, v - c[0])], [max(0, c[1] - v)]],
                    fmt="none", ecolor="k", capsize=4, lw=1.2)
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("brake", "b").replace("(1)", "(1.0)")
                        for k in keys], fontsize=7.5)
    ax.set_ylabel("paired step-down (install-60)")
    ax.set_title("E115-4 — brake(r_low) - brake(r_high)\n"
                 "negative (green) = brake shrinks as field weakens | "
                 "positive (red) = gap widens downward (M3 texture)",
                 fontsize=9.5)

    # (1,2) verdict text
    ax = axes[1, 2]
    ax.axis("off")
    import textwrap
    vlines = [
        "REGISTERED BARS (frozen pre-compute):",
        "  M3: brake(r) > 0, paired CI excl 0, >= 2 of 3 graded r",
        "       => M3 RESURRECTED (fourth story falls)",
        "  COORD-LOCAL: brake(1.0)-brake(0.1) CI>0 shrinks AND",
        "       brake(0.1) not sig-positive => STORY SAFE",
        "  else MIXED (report texture)",
        "",
        "THE GRID (install-60, g0):",
    ] + [
        f"  r={r:<4g}: A {inst[r]['mean_p_A']:.4f}  -A "
        f"{inst[r]['mean_p_nA']:.4f}  brake {inst[r]['brake_mean']:+.4f} "
        f"CI [{inst[r]['brake_ci95'][0]:+.3f},{inst[r]['brake_ci95'][1]:+.3f}]"
        + ("  [FLOOR]" if inst[r]["at_floor"] else "")
        for r in RS_ALL
    ] + [
        "",
        f"graded positive count: {m3_count}/3 (raw no-epsilon "
        f"{m3_count_raw}/3; unpaired {m3_unpaired_count}/3)",
        f"shrink 1.0->0.1: {shrink['negative_means_brake_shrinks_as_r_falls']}"
        f" | floor inverts: {floor_inverts}",
        f"gap widens 0.56->0.1: {gap_widens}",
        "",
        f"WINNER: {winner}",
    ] + [f"  {wd}" for wd in textwrap.wrap(clause, 62)]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.043, tx, fontsize=7.6, va="top",
                family="monospace")

    fig.suptitle(f"E115 — T067 graded-ablation resurrection check for M3: "
                 f"does the brake survive a weakened field? -> {winner}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "graded_ablation.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'graded_ablation.png'}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
