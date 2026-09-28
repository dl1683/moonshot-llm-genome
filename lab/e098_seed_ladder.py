"""E098 — the SEED LADDER, dual mandate (T053-NOVELTY upgrade + T066 replication).

MANDATE 1 (address-universality, T053-NOVELTY's registered upgrade): fresh
base nets at NEW seeds s in {4305, 4306, 4307, 4308} (the e040 REF family
used 4304; 42/43 are the existing install-family seeds), built with the
standard e053c recipe (4L/4H/128d, block 512 -> 873,472 params, corpus seed
1337, batch 32, lr 1e-3, cosine, ~2000 steps, 180s cap, ckpt-resumable).
Then the e043 install protocol VERBATIM on each (e082 GATE-0 convention:
same 60 install windows, SPLICE_RNG 24301, masked token-weighted name loss,
16 paired + 32 random anchor mix, AdamW 0.9/0.95 wd 0.1, lr 1e-3, 100 steps,
total 100, house cosine warmup 100, clip 1.0; pass bars install-60 p(Z)
>= 0.20 AND R1i NLL <= 4.5; ONE fresh 200-step patience trajectory if the
100-step run misses). Then the single-row census (e067/e071/e082 method) on
each: every wpe row r <- mean(all rows), in place; Drop(r) = base p(Z) -
pZ(r) on the install-60 battery. Per-seed report: install battery p(Z),
census top-3 rows, single-row concentration (top1 >= 2x top2 over fed rows
0..129), decision-row-129 stats, null band (rows 130..511, never fed by the
130-char battery).

REGISTERED BAR M1: >=3/4 seeds show single-row concentration (top row >= 2x
the second) => ADDRESS-UNIVERSALITY GRADUATES to law (n >= 5 total with
seeds 42/43); mixed => stays anecdote. Secondary readings (reported, not
gating): the excl-row-0 top row (the address row proper, 42/43 both = 129)
and row 129's rank.

MANDATE 2 (the share constant's first replication, T066): on TWO of the new
install nets (or their bases if the install complicates the anchor — use
whichever has the standard free-run anchor per the e053c/e110 convention:
seed-202 8-draw 64-token val prompts, seed-7 free run 64->512; anchor
"complicated" = control continuation clean-judge tail CE > 2.5 nats), run
the e110 field-floor mini-grid r in {0.25, 0.56, 1.0} x k in {64, 128} at
e110's exact intervention point (g_int 250 / T_int 314, band = positions
64..217, single-shot V-scaling before the 314 decode, V<-r*V on the kept
k-subset + V=0 on the band complement, matched continuation seed 960250,
clean-judge tail 448..511 vs the matched control; subsets re-drawn with
e110's own RNG stream seed 110 in (k, draw, row) order -> the k=64/128
subsets are bitwise those of e110). r*(k) = the 0.3-nat crossing
interpolated in log2(r) on the mini-grid rungs; product = r*_cont(k) x k.

REGISTERED BAR M2: products within +-25% of 54 (i.e. [40.5, 67.5]) in BOTH
nets (all four values) => CONSTANT REPLICATES (parameter-dependence question
sharpens); outside => SEED-DEPENDENT (the 54 was one net's accident —
equally important). Off-grid columns count as outside (off-grid-low =>
product < 0.25k << 40.5; off-grid-high => r* > 1.0 => product > k), flagged.

Envelope (task spec): GPU-PERMITTED but thermally gated — gpu_ok()
double-poll before every launch; cooldown(120) between fine-tunes; each
install fine-tune <= 180s (measured); CPU for all evals. If nvidia-smi shows
the GPU user-occupied (double-poll keeps failing for >10 min), base trains
fall back to CPU at reduced steps (500) and installs run CPU-side —
deviation documented in metrics.

DEVIATIONS / DECISIONS (registered before compute):
  1. The e043 install batch is VERBATIM (16 name + 48 anchor = 64 windows/
     step, 100 steps); the task's "batch 32" envelope note cannot override
     VERBATIM (e082 deviation-2 precedent); measured runtimes recorded and
     all installs are far under the 180s cap.
  2. Base steps = 2000 (the tasking's "~2000 steps"; e053c's 4000-step
     target was 180s-cap-bound at 3133 — 2000 is a COMPLETED cosine
     schedule, identical treatment for every ladder seed; data order fixed
     by corpus seed 1337 so INIT is the only seed axis, e040 convention).
  3. The ladder lives in the 0.84M 4L/4H/128d/512-ctx e053c family, not the
     2.7M e001 family of e043/e067/e082 — the tasking's design: the
     universality claim is tested across seeds AND into a second
     architecture family; census therefore sweeps 512 wpe rows (null band
     130..511) instead of e067's 256.
  4. Install exposure generator seed = the base seed s (no e043 donor
     crosscheck target exists for these nets; e082 used E43's b43-donor seed
     for its crosscheck — nothing analogous here).
  5. M2 battery = e110's own (B=8, all eight seed-202 prompts; e053c's own
     onset measurement used B=4 — the mini-grid replicates e110, whose
     constant we are testing).

Outputs: runs/e098/{metrics.json, seed_ladder.png}  (smoke: runs/e098_smoke/)
Run:     python lab/e098_seed_ladder.py   (E098_SMOKE=1 -> shakedown)
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

import common
from common import (REPO, Cfg, CharCorpus, TinyGPT, cooldown, estimate_loss,
                    gpu_ok, gpu_status, run_dir, save_json, set_seed,
                    train_model)
import e043_install as E43                       # the install protocol, verbatim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402

SMOKE = os.environ.get("E098_SMOKE") == "1"

# ------------------------------------------------------------------ constants
NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
SEEDS = [4305, 4306] if SMOKE else [4305, 4306, 4307, 4308]

# base-net training (e053c recipe; tasking's ~2000 steps)
BASE_STEPS = 40 if SMOKE else 2000
BASE_STEPS_CPU_FLOOR = 8 if SMOKE else 500       # reduced-steps CPU fallback
BASE_BATCH = 4 if SMOKE else 32
BASE_CAP_S = 30.0 if SMOKE else 180.0
BASE_CAP_S_CPU = 60.0 if SMOKE else 900.0
EVAL_EVERY = 10 if SMOKE else 250
VCE_PLAUSIBLE = 1.70                             # e053c's plausibility gate

# install (e043 verbatim + e082 GATE-0 bars/patience)
INSTALL_STEPS = 8 if SMOKE else 100
INSTALL_EVAL_AT = {4, 8} if SMOKE else {25, 50, 100}
PATIENCE_STEPS = 16 if SMOKE else 200
GATE_PZ_FLOOR = 0.20
GATE_R1I_MAX = 4.5

# census (e067/e071/e082 method, widened to the 512-row wpe)
FED_ROWS = 130                                    # rows 0..129 fed by the battery
CONC_RATIO = 2.0                                  # M1 bar: top1 >= 2x top2

# M1 verdict bar
M1_GRADUATE_N = 3                                 # >= 3/4 seeds

# M2 (e110 mini-grid; machinery constants verbatim e110 — geometry IDENTICAL
# in smoke and full mode; smoke shrinks only B / draws / boot, never the band)
PROMPT_TOK = 64
T_TOTAL = 512
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE = 202, 7
SEED_CONT = 960250
SEED_SUB = 110
N_PROMPTS = 2 if SMOKE else 8
B = N_PROMPTS
BOOT_N = 50 if SMOKE else 1000
THREADS = 8                                       # T050: 12 thrashes the box

KEY_T = (T_TOTAL - 64, T_TOTAL - 1)               # judged tail (448, 511)
G_INT = 250
T_INT = PROMPT_TOK + G_INT                        # 314
GEN_FIRST = 64
BAND = list(range(GEN_FIRST, T_INT - 96))         # 64..217 (154 entries)
FIRST_AFFECTED = T_INT + 1                        # 315
TAIL_STEP0 = KEY_T[0] - FIRST_AFFECTED            # 133
BAND_ARR = np.arange(BAND[0], BAND[-1] + 1)

KS = [64, 128]                                    # the mini-grid field sizes
RS = [0.25, 0.56, 1.0]                            # the mini-grid rungs
D_DRAWS = 1 if SMOKE else 3
HEALTHY_BAR = 0.3
DAMAGED_BAR = 1.0
SHARE_REF = 54.0                                  # T066's constant
SHARE_BAND = 0.25                                 # +-25%
SHARE_LO, SHARE_HI = SHARE_REF * (1 - SHARE_BAND), SHARE_REF * (1 + SHARE_BAND)
ANCHOR_CJ_MAX = 2.5                               # "install complicates anchor"

GPU_WAIT_MAX_S = 60 if SMOKE else 600
GPU_POLL_S = 30
COOLDOWN_S = 5 if SMOKE else 120

E067_METRICS = REPO / "runs" / "e067" / "metrics.json"
E082_METRICS = REPO / "runs" / "e082" / "metrics.json"
E110_METRICS = REPO / "runs" / "e110" / "metrics.json"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

REGISTERED = {
    "m1_bar": (f">= {M1_GRADUATE_N}/4 seeds with single-row concentration "
               f"(top census row drop >= {CONC_RATIO:.0f}x the second, over fed "
               f"rows 0..{FED_ROWS - 1}) => ADDRESS-UNIVERSALITY GRADUATES to "
               "law (n >= 5 with seeds 42/43); mixed => stays anecdote"),
    "m1_secondary": ("excl-row-0 top row (the address row proper; 42/43 both "
                     "129) and row-129 rank reported per seed, not gating"),
    "m2_bar": (f"r*_cont(k) x k within [{SHARE_LO:.1f}, {SHARE_HI:.1f}] "
               f"(+-{SHARE_BAND:.0%} of {SHARE_REF:.0f}) for BOTH ks in BOTH "
               "nets => CONSTANT REPLICATES; any product outside (off-grid "
               "counts as outside) => SEED-DEPENDENT"),
    "m2_net_rule": ("first two seeds in ladder order whose INSTALL passes GATE "
                    "and whose free-run anchor is healthy (control-continuation "
                    "clean-judge tail CE <= 2.5); else that seed's BASE net "
                    "(flagged fallback); else next seed"),
}
log("REGISTERED: " + " | ".join(f"{k}: {v}" for k, v in REGISTERED.items()))


# ------------------------------------------------------------------ gpu gate
def gpu_ok_double(poll_gap_s: float = 2.0) -> bool:
    if not gpu_ok():
        return False
    time.sleep(poll_gap_s)
    return gpu_ok()


def acquire_gpu() -> tuple[str, list[dict]]:
    """Bounded wait for a thermally-clear, user-free GPU; else CPU fallback."""
    polls = []
    t0 = time.time()
    while time.time() - t0 < GPU_WAIT_MAX_S:
        s = gpu_status()
        polls.append(s)
        if gpu_ok_double():
            return "cuda", polls
        log(f"[gpu_guard] HOLD (double-poll failed): {s} — waiting "
            f"{GPU_POLL_S}s ({time.time() - t0:.0f}/{GPU_WAIT_MAX_S}s)")
        time.sleep(GPU_POLL_S)
    return "cpu", polls


# ------------------------------------------------------------------ batteries
def load_ref_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def ref_block() -> dict:
    """Seeds 42/43 census rows + e110's products, read from shipped metrics."""
    out = {}
    m67 = load_ref_json(E067_METRICS)
    if m67:
        p = m67["primary"]
        out["seed42"] = {
            "net": p["checkpoint"], "install_pz": p["base_pz"],
            "top5_rows": p["top5_rows"],
            "top5_drops": [round(d, 4) for d in p["top5_drops"]],
            "ratio_top1_top2": round(p["top5_drops"][0] / p["top5_drops"][1], 3),
        }
    m82 = load_ref_json(E082_METRICS)
    if m82:
        g1 = m82["gate1"]
        dr = g1["drops_install60"]
        srt = sorted(((float(v), int(k)) for k, v in dr.items()), reverse=True)
        out["seed43"] = {
            "net": "e082_b43_install.pt",
            "install_pz": m82["gate0"]["final"]["install60_pz"]
            if isinstance(m82["gate0"].get("final"), dict)
            else m82["refs_A0"]["install60"]["mean_pz"],
            "top3_rows": g1["top3_by_drop"],
            "top3_drops": [round(srt[i][0], 4) for i in range(3)],
            "ratio_top1_top2": round(srt[0][0] / srt[1][0], 3),
        }
    m110 = load_ref_json(E110_METRICS)
    if m110:
        out["e110_share_constant"] = {
            k: (round(v, 2) if v is not None else None)
            for k, v in m110["summary"]["product"]["values"].items()}
        out["e110_net"] = "e053c_ctx512.pt (873,472 params, val CE 1.5227)"
    return out


@torch.no_grad()
def battery_pz(m: TinyGPT, ids: torch.Tensor, zid: int, bs: int = 30):
    """e067's battery: per-window p(Z) at the last context position."""
    m.eval()
    pzs, amax = [], []
    for i in range(0, ids.shape[0], bs):
        lg, _ = m(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, zid]) for k in range(pr.shape[0])]
        amax += [bool(pr[k].argmax() == zid) for k in range(pr.shape[0])]
    return np.array(pzs), float(np.mean(amax))


def run_census(net_cpu: TinyGPT, ids: torch.Tensor, zid: int) -> dict:
    """e067 census widened to the 512-row wpe: every row <- mean(all rows)."""
    net_cpu.eval()
    w = net_cpu.wpe.weight.data
    wpe_orig = w.clone()
    mean_row = wpe_orig.mean(0)
    pzs0, amax0 = battery_pz(net_cpu, ids, zid)
    base_pz, base_amax = float(pzs0.mean()), amax0
    pz_rows = np.zeros(wpe_orig.shape[0])
    t1 = time.time()
    for r in range(wpe_orig.shape[0]):
        w.copy_(wpe_orig)
        w[r] = mean_row
        pzs, _ = battery_pz(net_cpu, ids, zid)
        pz_rows[r] = float(pzs.mean())
    w.copy_(wpe_orig)
    drops = base_pz - pz_rows
    fed = drops[:FED_ROWS]
    null = drops[FED_ROWS:]
    order = np.argsort(-fed)
    top3 = [int(r) for r in order[:3]]
    d1, d2, d3 = (float(fed[order[i]]) for i in range(3))
    ratio = d1 / max(d2, 1e-9)
    excl0 = [r for r in order if r != 0]
    top3_excl0 = [int(r) for r in excl0[:3]]
    d1x = float(fed[excl0[0]])
    d2x = float(fed[excl0[1]])
    # zero-arm confirmation on the top-3 fed rows (e067 convention)
    zero_drops = {}
    for r in top3:
        w.copy_(wpe_orig)
        w[r] = 0.0
        pzs, _ = battery_pz(net_cpu, ids, zid)
        zero_drops[int(r)] = float(base_pz - pzs.mean())
    w.copy_(wpe_orig)
    pos = np.clip(fed, 0, None)
    res = {
        "base_pz": float(base_pz), "base_argmax_frac": base_amax,
        "pz_rows": pz_rows.tolist(), "drops": drops.tolist(),
        "top3_rows": top3, "top3_drops": [d1, d2, d3],
        "ratio_top1_top2": ratio, "concentration_pass": bool(ratio >= CONC_RATIO),
        "row0_drop": float(fed[0]), "row129_drop": float(fed[129]),
        "row129_rank_excl0": (1 + int(np.where(np.array(excl0) == 129)[0][0])
                              if 129 in excl0 else None),
        "top3_rows_excl_row0": top3_excl0,
        "drops_excl_row0_top2": [d1x, d2x],
        "ratio_excl0": d1x / max(d2x, 1e-9),
        "top1_pos_mass_frac": float(pos[order[0]] / max(pos.sum(), 1e-9)),
        "null_band_rows_130_511": {"max": float(np.max(null)),
                                   "min": float(np.min(null)),
                                   "mean_abs": float(np.mean(np.abs(null)))},
        "zero_arm_top3": zero_drops,
        "row129_norm": float(wpe_orig[129].norm()),
        "census_wall_s": round(time.time() - t1, 1),
    }
    log(f"  census: base pZ {base_pz:.4f} | top3 {top3} drops "
        f"{[round(x, 4) for x in (d1, d2, d3)]} | ratio {ratio:.2f} -> "
        f"{'CONCENTRATED' if res['concentration_pass'] else 'not'} | "
        f"excl-row0 top {top3_excl0} | null max "
        f"{res['null_band_rows_130_511']['max']:.4f}")
    return res


# ------------------------------------------------- e110 machinery (VERBATIM)
@torch.no_grad()
def manual_all_logits(net: TinyGPT, idxs):
    """Clean forward returning logits at ALL positions (the clean judge)."""
    N, T = idxs.shape
    pos = torch.arange(T)
    x = net.wte(idxs) + net.wpe(pos).unsqueeze(0)
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    for blk in net.h:
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
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(N, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


def sample_and_ce(logits_clean: torch.Tensor, gen: torch.Generator):
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
    Bb, T = idx.shape
    x = net.wte(idx) + net.wpe(torch.arange(T))
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    kv = []
    for blk in net.h:
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=2)
        q = q.view(Bb, T, H, d).transpose(1, 2)
        k = k.view(Bb, T, H, d).transpose(1, 2)
        v = v.view(Bb, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(Bb, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
        kv.append((k, v))
    return net.lm_head(net.ln_f(x[:, -1, :])), kv


@torch.no_grad()
def decode_step_batch(net: TinyGPT, toks: torch.Tensor, pos: int, kv: list):
    Bb = toks.shape[0]
    x = net.wte(toks) + net.wpe(torch.full((Bb,), pos))
    H = net.cfg.n_head
    for li, blk in enumerate(net.h):
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=1)
        q = q.view(Bb, 1, H, d).transpose(1, 2)
        k = k.view(Bb, 1, H, d).transpose(1, 2)
        v = v.view(Bb, 1, H, d).transpose(1, 2)
        kp, vp = kv[li]
        k = torch.cat([kp, k], dim=2)
        v = torch.cat([vp, v], dim=2)
        kv[li] = (k, v)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(Bb, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


@torch.no_grad()
def generate_control(net: TinyGPT, prompts, gen: torch.Generator):
    Bb = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    G = T_TOTAL - PROMPT_TOK
    for g in range(G):
        t = PROMPT_TOK + g
        toks = torch.zeros(Bb, dtype=torch.long)
        for j in range(Bb):
            tok, _ce = sample_and_ce(logits[j], gen)
            toks[j] = tok
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < T_TOTAL - 1:
            logits = decode_step_batch(net, toks, t, kv)
    return dict(idx=idx, kv=kv)


@torch.no_grad()
def run_field_continuation(net: TinyGPT, prefix: torch.Tensor,
                           forced_tok: torch.Tensor, forced_pos: int,
                           seed: int, keep_per_row=None, r: float = 1.0):
    """e110's run_field_continuation VERBATIM (see its docstring): single-shot
    band transform (kept subset V<-r*V, band complement V=0, K untouched)
    before the decode at forced_pos; teacher-forced token; shared-seed
    matched free-run; per-row online CE + entropy tracked."""
    R = prefix.shape[0]
    gen = torch.Generator().manual_seed(seed)
    logits, kv = prefill_batch(net, prefix)
    whole_band = torch.tensor(BAND, dtype=torch.long)
    is_identity = (keep_per_row is None) or (
        r == 1.0 and all(torch.equal(torch.as_tensor(kp, dtype=torch.long),
                                     whole_band) for kp in keep_per_row))
    fired = dict(applied=not is_identity, r=r,
                 n_kept_per_row=[], n_removed_per_row=[],
                 norm_dev=0.0, min_cos=1.0, mean_cos=None,
                 mean_norm_ratio=None, removed_exactly_zero=True,
                 nonband_untouched=True)
    if not is_identity:
        pre_v = [v.clone() for (_k, v) in kv]
        coss, ratios = [], []
        band_set = set(BAND)
        for j in range(R):
            kp = torch.as_tensor(keep_per_row[j], dtype=torch.long)
            comp = torch.tensor(sorted(band_set - set(kp.tolist())),
                                dtype=torch.long)
            fired["n_kept_per_row"].append(int(kp.numel()))
            fired["n_removed_per_row"].append(int(comp.numel()))
            for (_k, v) in kv:
                old = v[j, :, kp, :].clone()
                if r != 1.0:
                    v[j, :, kp, :] = r * old
                if comp.numel():
                    v[j, :, comp, :] = 0.0
        for (_k, v), pv in zip(kv, pre_v):
            for j in range(R):
                kp = torch.as_tensor(keep_per_row[j], dtype=torch.long)
                comp = torch.tensor(sorted(band_set - set(kp.tolist())),
                                    dtype=torch.long)
                old = pv[j, :, kp, :].clone()
                new = v[j, :, kp, :]
                nrm = old.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                nnew = new.norm(dim=-1, keepdim=True)
                fired["norm_dev"] = max(fired["norm_dev"], float(
                    (nnew - r * nrm).abs().max()))
                cos = (old * new).sum(-1) / (nrm.squeeze(-1)
                                             * nnew.squeeze(-1)).clamp_min(1e-12)
                fired["min_cos"] = min(fired["min_cos"], float(cos.min()))
                coss.append(cos.reshape(-1))
                ratios.append((nnew / nrm).reshape(-1))
                if comp.numel():
                    fired["removed_exactly_zero"] &= bool(
                        (v[j, :, comp, :] == 0.0).all())
            nb_lo = torch.arange(0, GEN_FIRST)
            nb_hi = torch.arange(BAND[-1] + 1, forced_pos)
            fired["nonband_untouched"] &= bool(
                torch.equal(v[:, :, nb_lo, :], pv[:, :, nb_lo, :]))
            fired["nonband_untouched"] &= bool(
                torch.equal(v[:, :, nb_hi, :], pv[:, :, nb_hi, :]))
        fired["mean_cos"] = float(torch.cat(coss).mean())
        fired["mean_norm_ratio"] = float(torch.cat(ratios).mean())
    idx = torch.cat([prefix, forced_tok[:, None]], 1)
    logits = decode_step_batch(net, forced_tok, forced_pos, kv)
    n_free = T_TOTAL - 1 - forced_pos
    ce_s = np.zeros((R, n_free), float)
    ent_s = np.zeros((R, n_free), float)
    for s in range(n_free):
        pos = forced_pos + 1 + s
        new = torch.zeros(R, dtype=torch.long)
        for j in range(R):
            tok, ce = sample_and_ce(logits[j], gen)
            new[j] = tok
            ce_s[j, s] = ce
        p = torch.softmax(logits.float(), -1)
        ent_s[:, s] = (-(p * p.clamp_min(1e-12).log()).sum(-1)).numpy()
        idx = torch.cat([idx, new[:, None]], 1)
        if pos < T_TOTAL - 1:
            logits = decode_step_batch(net, new, pos, kv)
    return idx, kv, fired, ce_s, ent_s


def judge_windows(all_lg: torch.Tensor, idx: torch.Tensor, windows):
    out = {}
    for (lo, hi) in windows:
        lg = all_lg[:, lo - 1:hi, :]
        tgt = idx[:, lo:hi + 1]
        lp = torch.log_softmax(lg.float(), -1)
        out[(lo, hi)] = -lp.gather(2, tgt[:, :, None]).squeeze(2).mean(1).numpy()
    return out


def boot_ci(arrs, fn, n: int = BOOT_N, seed: int = 0):
    S = np.asarray(arrs[0]).shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, S, S)
        try:
            v = fn(*[np.asarray(a)[sel] for a in arrs])
        except Exception:
            v = None
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def health_of(cost: float) -> str:
    if cost < HEALTHY_BAR:
        return "healthy"
    if cost > DAMAGED_BAR:
        return "damaged"
    return "ambiguous"


def rstar_cont(col_means: dict) -> tuple:
    """e110's continuous 0.3-nat crossing in log2(r) — rightmost bracket."""
    if col_means[RS[-1]] > HEALTHY_BAR:
        return float("nan"), "off-grid-high"
    if col_means[RS[0]] < HEALTHY_BAR:
        return float("nan"), "off-grid-low"
    x = np.log2(np.array(RS))
    y = np.array([col_means[r] for r in RS])
    for i in range(len(RS) - 2, -1, -1):
        if y[i] >= HEALTHY_BAR > y[i + 1]:
            t = x[i] + (HEALTHY_BAR - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            return float(2.0 ** t), "interp"
    return float("nan"), "no-crossing"


# ------------------------------------------------------------------ helpers
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)) or x is None or isinstance(x, (int, str)):
        return bool(x) if isinstance(x, np.bool_) else x
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return "inf" if math.isinf(v) else ("nan" if math.isnan(v) else v)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, torch.Tensor):
        return jsonable(x.tolist())
    return str(x)


def to_cpu_net(sd: dict, cfg: Cfg) -> TinyGPT:
    m = TinyGPT(cfg)
    m.load_state_dict(sd)
    m.eval()
    return m


# ------------------------------------------------------------------ main
def main():
    torch.set_num_threads(THREADS)
    tag = "e098_smoke" if SMOKE else "e098"
    rd = run_dir(tag)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cfg = None
    per_seed: dict[int, dict] = {}
    deviations = []

    # ---- corpus + protocol rebuild (e055/e066/e066b/e067/e082 verbatim)
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=T_TOTAL)
    stoi = corpus.stoi
    train_ids = corpus.train
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    zid = stoi["Z"]
    assert not E43.find_occ(train_text, NAME)

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + E43.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"
    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)

    def build_win(p, host):
        return torch.cat([train_ids[p - PRE: p], name_ids,
                          train_ids[p + len(host): p + len(host) + E43.POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    inst_mask = torch.zeros(60, E43.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, PRE - 1:PRE - 1 + len(NAME)] = True
    anchor_full = torch.stack([train_ids[p - PRE: p - PRE + E43.BLOCK]
                               for p, _ in install_occ])
    bat_i_seq = win_i[:, :PRE + len(NAME)]
    ids_install = torch.stack([corpus.encode(train_text[p - PRE:p])
                               for p, _ in install_occ])
    ids_held = torch.stack([corpus.encode(train_text[p - PRE:p])
                            for p, _ in held_occ])
    log(f"protocol rebuilt: install60 {mix}, held30 (corpus 1337, "
        f"SPLICE_RNG {E43.SPLICE_RNG}) | cfg 4L/4H/128d/wpe-{T_TOTAL}")

    refs = ref_block()
    log(f"refs: {json.dumps(jsonable(refs), default=str)[:400]}")

    # ================================================== M1: the seed ladder
    for i, s in enumerate(SEEDS):
        log(f"===== seed {s} ({i + 1}/{len(SEEDS)}) =====")
        rec: dict = {"seed": s}

        # ---- base net (e053c recipe at a new seed)
        dev, polls = acquire_gpu()
        rec["base_gpu"] = {"device": dev, "n_polls": len(polls),
                           "last": polls[-1] if polls else None}
        steps = BASE_STEPS
        cap = BASE_CAP_S
        if dev == "cpu":
            steps = BASE_STEPS_CPU_FLOOR
            cap = BASE_CAP_S_CPU
            deviations.append(
                f"seed {s}: base train fell back to CPU at reduced steps "
                f"({steps}) — GPU user-occupied/thermal (task-spec fallback)")
        common.DEVICE = dev
        set_seed(s)                              # init on CPU (e040 rule)
        net = TinyGPT(cfg).to(dev)
        n_params = net.num_params()
        assert n_params == 873_472, f"params {n_params} != 873472"
        ck_base = REPO / "runs" / "checkpoints" / (
            f"e098_base_s{s}{'_smoke' if SMOKE else ''}.pt")
        t1 = time.time()
        hist = train_model(net, corpus, steps=steps, lr=1e-3,
                           batch_size=BASE_BATCH, max_seconds=cap,
                           eval_every=EVAL_EVERY, ckpt=ck_base)
        base_wall = time.time() - t1
        common.DEVICE = "cpu"
        net_cpu = net.cpu().eval()
        val_ce = estimate_loss(net_cpu, corpus, "val", n_batches=12)
        base_sd = {k: v.detach().clone() for k, v in net_cpu.state_dict().items()}
        rec["base_sd"] = base_sd
        rec["base"] = {
            "params": n_params, "steps_target": steps,
            "steps_done": int(hist[-1]["step"]) if hist else 0,
            "val_ce": val_ce, "plausible": bool(val_ce < VCE_PLAUSIBLE),
            "wall_s": round(base_wall, 1), "device": dev,
            "ckpt": str(ck_base.name),
            "history": [{"step": h["step"], "val_loss": h["val_loss"]}
                        for h in hist],
        }
        del net
        log(f"  base: steps {rec['base']['steps_done']}/{steps} on {dev} in "
            f"{base_wall:.0f}s | val CE {val_ce:.4f} "
            f"({'plausible' if rec['base']['plausible'] else 'OFF'})")
        if dev == "cuda":
            cooldown(COOLDOWN_S)

        # ---- install (e043 exposure VERBATIM; e082 GATE-0 bars/patience)
        dev2, polls2 = acquire_gpu()
        rec["install_gpu"] = {"device": dev2, "n_polls": len(polls2)}
        common.DEVICE = dev2
        E43.DEVICE = dev2
        if dev2 == "cpu":
            deviations.append(
                f"seed {s}: install ran CPU-side (GPU unavailable at launch; "
                f"full 100-step protocol kept — measured cost trivial)")
        inst = copy.deepcopy(net_cpu).to(dev2)

        def gate_readout(m, step):
            pz, am = battery_pz(m, ids_install.to(dev2), zid)
            pzh, _ = battery_pz(m, ids_held.to(dev2), zid)
            r1i = E43.eval_seq(m, bat_i_seq.to(dev2), len(NAME), PRE - 1)
            return {"step": step, "install60_pz": float(pz.mean()),
                    "install60_argmax": am, "held30_pz": float(pzh.mean()),
                    "r1i_nll": r1i["nll"], "r1i_acc": r1i["acc"],
                    "onset_acc": r1i["per_pos_acc"][0],
                    "pos36_acc": sum(r1i["per_pos_acc"][3:7]) / 4}

        ck_inst = REPO / "runs" / "checkpoints" / (
            f"e098_install_s{s}{'_smoke' if SMOKE else ''}.pt")
        t2 = time.time()
        traj = E43.exposure(
            inst, win_i.to(dev2), inst_mask.to(dev2), anchor_full.to(dev2),
            steps=INSTALL_STEPS, total=INSTALL_STEPS,
            gen=torch.Generator().manual_seed(s), tag=f"e098_install_s{s}",
            ckpt=ck_inst, eval_at=set(INSTALL_EVAL_AT), on_eval=gate_readout,
            mix_random=E43.MIX_RANDOM, train_ids=train_ids, log=log)
        inst_wall = time.time() - t2
        final_ro = gate_readout(inst, INSTALL_STEPS)
        gate_pass = bool(final_ro["install60_pz"] >= GATE_PZ_FLOOR
                         and final_ro["r1i_nll"] <= GATE_R1I_MAX)
        patience_used = False
        if not gate_pass and not SMOKE:
            log(f"  GATE-0 miss at s{INSTALL_STEPS} (pZ "
                f"{final_ro['install60_pz']:.3f} / R1i {final_ro['r1i_nll']:.2f})"
                f" — e082 patience branch: ONE fresh {PATIENCE_STEPS}-step run")
            inst = copy.deepcopy(net_cpu).to(dev2)
            ck_pat = REPO / "runs" / "checkpoints" / (
                f"e098_install_s{s}_pat{'_smoke' if SMOKE else ''}.pt")
            t3 = time.time()
            traj = E43.exposure(
                inst, win_i.to(dev2), inst_mask.to(dev2), anchor_full.to(dev2),
                steps=PATIENCE_STEPS, total=PATIENCE_STEPS,
                gen=torch.Generator().manual_seed(s + 1000),
                tag=f"e098_install_s{s}_pat", ckpt=ck_pat,
                eval_at={PATIENCE_STEPS // 2, PATIENCE_STEPS},
                on_eval=gate_readout, mix_random=E43.MIX_RANDOM,
                train_ids=train_ids, log=log)
            inst_wall += time.time() - t3
            final_ro = gate_readout(inst, PATIENCE_STEPS)
            gate_pass = bool(final_ro["install60_pz"] >= GATE_PZ_FLOOR
                             and final_ro["r1i_nll"] <= GATE_R1I_MAX)
            patience_used = True
        inst_sd = {k: v.detach().cpu().clone()
                   for k, v in inst.state_dict().items()}
        rec["install_sd"] = inst_sd
        rec["install"] = {
            "protocol": "e043 exposure VERBATIM (100 steps, 16+48 mix, "
                        "lr 1e-3, token-weighted masked loss)",
            "gen_seed": s if not patience_used else s + 1000,
            "steps": PATIENCE_STEPS if patience_used else INSTALL_STEPS,
            "traj": traj, "final": final_ro, "gate_pass": gate_pass,
            "patience_used": patience_used,
            "bars": {"pz_floor": GATE_PZ_FLOOR, "r1i_max": GATE_R1I_MAX},
            "wall_s": round(inst_wall, 1), "device": dev2,
        }
        log(f"  install (s{rec['install']['steps']}, {dev2}, "
            f"{inst_wall:.0f}s): pZ {final_ro['install60_pz']:.4f} held "
            f"{final_ro['held30_pz']:.4f} R1i {final_ro['r1i_nll']:.3f}/"
            f"{final_ro['r1i_acc']:.3f} -> GATE "
            f"{'PASS' if gate_pass else 'FAIL'}"
            + (" [patience]" if patience_used else ""))
        del inst
        common.DEVICE = "cpu"
        E43.DEVICE = "cpu"
        if dev2 == "cuda":
            cooldown(COOLDOWN_S)

        # ---- census (CPU; e067/e071 method on the 512-row wpe)
        inst_cpu = to_cpu_net(inst_sd, cfg)
        census = run_census(inst_cpu, ids_install, zid)
        census["floor_limited"] = bool(final_ro["install60_pz"] < 0.05)
        rec["census"] = census
        rec["counts_for_m1"] = bool(gate_pass and census["concentration_pass"])
        del inst_cpu
        per_seed[s] = rec

    # ---- M1 verdict
    n_conc = sum(1 for r in per_seed.values() if r["counts_for_m1"])
    n_gate = sum(1 for r in per_seed.values() if r["install"]["gate_pass"])
    m1 = {
        "n_seeds": len(SEEDS), "n_gate_pass": n_gate, "n_concentrated": n_conc,
        "graduate_bar": M1_GRADUATE_N,
        "verdict": ("ADDRESS-UNIVERSALITY GRADUATES to law"
                    if n_conc >= M1_GRADUATE_N else
                    "MIXED — stays anecdote"),
        "n_total_with_4243": 2 + n_conc,
        "secondary": {
            str(s): {"excl0_top": r["census"]["top3_rows_excl_row0"][0],
                     "row129_rank_excl0": r["census"]["row129_rank_excl0"],
                     "ratio_excl0": round(r["census"]["ratio_excl0"], 2)}
            for s, r in per_seed.items()},
    }
    log(f"M1: {n_conc}/{len(SEEDS)} seeds concentrated (gate-pass "
        f"{n_gate}/{len(SEEDS)}) -> {m1['verdict']}")

    # ================================================== M2: the share constant
    corp2 = corpus            # same corpus object; prompts from val
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp2.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts = [corp2.val[i:i + PROMPT_TOK] for i in ix]

    rng_sub = np.random.default_rng(SEED_SUB)
    subsets = {k: [[None] * B for _ in range(D_DRAWS)] for k in KS}
    for k in (32, *KS):                       # e110's exact (k,d,row) stream
        for d in range(D_DRAWS if k in KS else 3):
            for j in range(B):
                drawn = rng_sub.choice(BAND_ARR, size=k, replace=False)
                if k in KS:
                    subsets[k][d][j] = drawn

    def anchor_and_grid(sd: dict, which: str, seed: int) -> dict | None:
        net_m2 = to_cpu_net(sd, cfg)
        gen7 = torch.Generator().manual_seed(SEED_SAMPLE)
        run1 = generate_control(net_m2, prompts, gen7)
        idx1 = run1["idx"]
        prefix, forced = idx1[:, :T_INT], idx1[:, T_INT]
        idx_ctl, kv_ctl, fired_ctl, ce_ctl, ent_ctl = run_field_continuation(
            net_m2, prefix, forced, T_INT, SEED_CONT, keep_per_row=None, r=1.0)
        J_ctl = judge_windows(manual_all_logits(net_m2, idx_ctl), idx_ctl,
                              [KEY_T])[KEY_T]
        online_tail = float(np.asarray(ce_ctl[:, TAIL_STEP0:]).mean())
        anchor_ok = bool(np.isfinite(J_ctl).all() and J_ctl.mean() <= ANCHOR_CJ_MAX)
        log(f"  [s{seed} {which}] control tail cj {float(J_ctl.mean()):.4f} "
            f"online {online_tail:.4f} -> anchor "
            f"{'healthy' if anchor_ok else 'BROKEN'}")
        if not anchor_ok:
            return None
        cellJ, cell_info = {}, {}
        for k in KS:
            for r in RS:
                Js, fireds = [], []
                for d in range(D_DRAWS):
                    keep = [torch.as_tensor(subsets[k][d][j], dtype=torch.long)
                            for j in range(B)]
                    idx_a, _kv, fired_a, ce_a, _e = run_field_continuation(
                        net_m2, prefix, forced, T_INT, SEED_CONT,
                        keep_per_row=keep, r=r)
                    Js.append(judge_windows(manual_all_logits(net_m2, idx_a),
                                            idx_a, [KEY_T])[KEY_T])
                    fireds.append(fired_a)
                cellJ[(k, r)] = np.mean(np.stack(Js), axis=0)
                f0 = fireds[0]
                ok_inst = bool(
                    f0["applied"] and f0["norm_dev"] < 1e-5
                    and f0["min_cos"] > 1.0 - 1e-5
                    and f0["removed_exactly_zero"] and f0["nonband_untouched"]
                    and f0["n_kept_per_row"] == [k] * B)
                cell_info[(k, r)] = {"fired": f0, "ok": ok_inst}
                log(f"  [s{seed} {which}] cell k={k} r={r}: cj-tail cost "
                    f"{float((cellJ[(k, r)] - J_ctl).mean()):+.4f} "
                    f"[{health_of(float((cellJ[(k, r)] - J_ctl).mean()))}] "
                    f"inst-ok {ok_inst}")
        col_means = {k: {r: float((cellJ[(k, r)] - J_ctl).mean())
                         for r in RS} for k in KS}
        rstar, prod_ci = {}, {}
        rng_b = np.random.default_rng(0)
        conts_b = {k: [] for k in KS}
        for _ in range(BOOT_N):
            sel = rng_b.integers(0, B, B)
            for k in KS:
                cm = {r: float((cellJ[(k, r)] - J_ctl)[sel].mean()) for r in RS}
                cv, _cs = rstar_cont(cm)
                conts_b[k].append(cv * k if np.isfinite(cv) else np.nan)
        for k in KS:
            cval, cstatus = rstar_cont(col_means[k])
            prod = float(cval * k) if np.isfinite(cval) else None
            ci = ([float(np.nanpercentile(conts_b[k], 2.5)),
                   float(np.nanpercentile(conts_b[k], 97.5))]
                  if not all(np.isnan(x) for x in conts_b[k])
                  else [float("nan"), float("nan")])
            rstar[k] = {"cont_value": (None if not np.isfinite(cval)
                                       else float(cval)),
                        "cont_status": cstatus, "product": prod,
                        "product_ci": ci,
                        "in_band": (bool(SHARE_LO <= prod <= SHARE_HI)
                                    if prod is not None else None),
                        "col_means": {str(r): col_means[k][r] for r in RS}}
            log(f"  [s{seed} {which}] r*({k}) = "
                + (f"{cval:.3f}" if np.isfinite(cval) else cstatus)
                + (f" | r*k {prod:.1f} CI [{ci[0]:.1f},{ci[1]:.1f}] "
                   f"in-band {rstar[k]['in_band']}" if prod else ""))
        per_cell = {f"k={k},r={r}": {
            "cost_per_row": (cellJ[(k, r)] - J_ctl).tolist(),
            "mean": float((cellJ[(k, r)] - J_ctl).mean()),
            "ci": boot_ci([cellJ[(k, r)] - J_ctl], lambda a: float(a.mean())),
            "health": health_of(float((cellJ[(k, r)] - J_ctl).mean())),
            "instrument_ok": cell_info[(k, r)]["ok"]}
            for k in KS for r in RS}
        return {"which": which, "seed": seed,
                "control_tail_cj": float(J_ctl.mean()),
                "control_online_tail": online_tail,
                "control_not_applied": bool(not fired_ctl["applied"]),
                "rstar": {str(k): v for k, v in rstar.items()},
                "per_cell": per_cell,
                "grid_instrument_all_ok": bool(
                    all(v["ok"] for v in cell_info.values()))}

    m2_nets = []
    for s in SEEDS:
        if len(m2_nets) >= 2:
            break
        rec = per_seed[s]
        chosen = None
        if rec["install"]["gate_pass"]:
            chosen = anchor_and_grid(rec["install_sd"], "install", s)
        if chosen is None:
            log(f"  [s{s}] install unusable for M2 — trying the BASE net")
            chosen = anchor_and_grid(rec["base_sd"], "base", s)
            if chosen is not None:
                deviations.append(
                    f"M2 seed {s}: used the BASE net (install did not provide "
                    "a healthy free-run anchor or failed its gate)")
        if chosen is not None:
            m2_nets.append(chosen)
    if len(m2_nets) < 2:
        deviations.append(
            f"M2: only {len(m2_nets)} net(s) usable (anchor/gate failures)")

    prods = [net["rstar"][str(k)]["product"] for net in m2_nets for k in KS]
    evaluated = [p for p in prods if p is not None]
    all_in_band = bool(len(m2_nets) == 2 and len(evaluated) == 4
                       and all(SHARE_LO <= p <= SHARE_HI for p in evaluated))
    any_out = bool(len(evaluated) >= 1 and any(
        not (SHARE_LO <= p <= SHARE_HI) for p in evaluated))
    if all_in_band:
        m2_verdict = "CONSTANT REPLICATES (both nets, both ks within +-25% of 54)"
    elif any_out:
        m2_verdict = ("SEED-DEPENDENT (at least one product outside "
                      f"[{SHARE_LO:.1f}, {SHARE_HI:.1f}])")
    else:
        m2_verdict = "PARTIAL/UNRESOLVED (off-grid or unusable columns)"
    m2 = {"nets": m2_nets, "products": prods, "band": [SHARE_LO, SHARE_HI],
          "ref": SHARE_REF, "verdict": m2_verdict,
          "e110_reference": refs.get("e110_share_constant")}
    log(f"M2: products {[round(p, 1) if p else None for p in prods]} -> "
        f"{m2_verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = {
        "experiment": "e098_seed_ladder",
        "purpose": ("dual mandate: (M1) address-universality seed ladder "
                    "(T053-NOVELTY's registered upgrade to n>=5) — 4 fresh "
                    "seeds, e043 install verbatim, single-row census; "
                    "(M2) the T066 share constant r*k~54's first replication "
                    "on two new nets via the e110 mini-grid"),
        "started": started, "wall_s": round(time.time() - T0, 1),
        "smoke": SMOKE, "threads": THREADS,
        "registered": REGISTERED,
        "seeds": SEEDS,
        "config": {"base_recipe": "e053c: 4L/4H/128d block-512, corpus 1337, "
                                  f"batch {BASE_BATCH}, lr 1e-3, cosine, "
                                  f"{BASE_STEPS} steps, {BASE_CAP_S:.0f}s cap",
                   "install_recipe": "e043 exposure VERBATIM; e082 GATE-0 "
                                     "bars + 200-step patience",
                   "census": "e067: every wpe row r <- mean(all 512 rows), "
                             "install-60 battery, p(Z) at position 129",
                   "m2_grid": {"k": KS, "r": RS, "draws": D_DRAWS, "B": B,
                               "subset_seed": SEED_SUB,
                               "subset_provenance": "e110's exact RNG stream "
                               "(k,d,row) — k=64/128 subsets bitwise e110's",
                               "intervention": {"g_int": G_INT, "t_int": T_INT,
                                                "band": [BAND[0], BAND[-1]],
                                                "n_band": len(BAND)},
                               "continuation_seed": SEED_CONT,
                               "judge_tail": list(KEY_T)}},
        "references": refs,
        "per_seed": {str(s): {k: v for k, v in r.items()
                              if k not in ("base_sd", "install_sd")}
                     for s, r in per_seed.items()},
        "m1": m1,
        "m2": m2,
        "gates": {
            "splice_protocol": {"mix": mix, "pass": True},
            "base_params": {str(s): r["base"]["params"]
                            for s, r in per_seed.items()},
            "base_val_ce_plausible": {str(s): r["base"]["plausible"]
                                      for s, r in per_seed.items()},
            "install_gate": {str(s): r["install"]["gate_pass"]
                             for s, r in per_seed.items()},
            "install_under_180s": {str(s): r["install"]["wall_s"] <= 180
                                   for s, r in per_seed.items()},
            "census_null_band": {str(s): r["census"]["null_band_rows_130_511"]
                                 for s, r in per_seed.items()},
            "m2_control_not_applied": {f"s{n['seed']}/{n['which']}":
                                       n["control_not_applied"]
                                       for n in m2_nets},
            "m2_instrument_all_ok": {f"s{n['seed']}/{n['which']}":
                                     n["grid_instrument_all_ok"]
                                     for n in m2_nets},
        },
        "deviations": deviations,
    }
    save_json(rd / "metrics.json", jsonable(metrics))
    log("metrics.json written")
    plot(rd / "seed_ladder.png", metrics, per_seed)
    log(f"plot written -> {rd}")
    return 0


# ------------------------------------------------------------------ plot
def plot(path: Path, M: dict, per_seed: dict):
    seeds = list(per_seed)
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    for j, s in enumerate(seeds[:4]):
        ax = axes[j // 2, j % 2]
        c = per_seed[s]["census"]
        d = np.array(c["drops"][:FED_ROWS])
        ax.plot(np.arange(FED_ROWS), d, lw=0.9, color="steelblue")
        ax.plot(c["top3_rows"], [d[r] for r in c["top3_rows"]], "o",
                color="crimson", ms=6, label="top-3 rows")
        ax.plot([129], [d[129]], "x", color="k", ms=8, label="row 129")
        ax.axhspan(-abs(c["null_band_rows_130_511"]["max"]),
                   abs(c["null_band_rows_130_511"]["max"]),
                   color="gray", alpha=0.18,
                   label=f"null band |max| {abs(c['null_band_rows_130_511']['max']):.4f}")
        ax.set_xlabel("wpe row (fed 0..129)")
        ax.set_ylabel("drop in battery p(Z)")
        inst = per_seed[s]["install"]
        ax.set_title(
            f"seed {s}: install p(Z) {inst['final']['install60_pz']:.3f} "
            f"(gate {'PASS' if inst['gate_pass'] else 'FAIL'}) | top3 "
            f"{c['top3_rows']} drops {[round(x, 3) for x in c['top3_drops']]} "
            f"| ratio {c['ratio_top1_top2']:.2f} "
            f"({'CONC' if c['concentration_pass'] else 'no-conc'}; excl0 top "
            f"{c['top3_rows_excl_row0'][0]})", fontsize=9)
        ax.legend(fontsize=7)
    if len(seeds) < 4:
        for j in range(len(seeds), 4):
            axes[j // 2, j % 2].axis("off")

    ax = axes[2, 0]
    kcol = {64: "tab:orange", 128: "tab:green"}
    for n_i, net in enumerate(M["m2"]["nets"]):
        for k in KS:
            rs = M["m2"]["nets"][n_i]["rstar"][str(k)]["col_means"]
            ax.plot(RS, [rs[str(r)] for r in RS], "o-",
                    color=kcol[k], lw=2 - 0.4 * n_i, ms=7,
                    label=f"s{net['seed']}/{net['which']} k={k}")
    ax.axhline(HEALTHY_BAR, color="tab:green", ls="--", lw=1.2, label="0.3 healthy")
    ax.axhline(DAMAGED_BAR, color="tab:red", ls="--", lw=1.2, label="1.0 damaged")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks(RS)
    ax.set_xticklabels([str(r) for r in RS])
    ax.set_xlabel("retention r")
    ax.set_ylabel("clean-judge tail cost (nats)")
    ax.set_title("M2 mini-grid: cost vs r (k=64/128), both nets", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[2, 1]
    xlabels, xs, ys, los, his = [], [], [], [], []
    x = 0
    for net in M["m2"]["nets"]:
        for k in KS:
            r = net["rstar"][str(k)]
            xlabels.append(f"s{net['seed']}\n{net['which']}\nk={k}")
            xs.append(x)
            ys.append(r["product"] if r["product"] is not None else np.nan)
            ci = r["product_ci"]
            los.append(ys[-1] - ci[0] if r["product"] is not None else np.nan)
            his.append(ci[1] - ys[-1] if r["product"] is not None else np.nan)
            x += 1
    ax.bar(xs, ys, 0.55, color=["tab:orange" if "k=64" in l else "tab:green"
                                for l in xlabels])
    ax.errorbar(xs, ys, yerr=[np.nan_to_num(los), np.nan_to_num(his)],
                fmt="none", ecolor="k", capsize=4, lw=1.2)
    ax.axhline(SHARE_REF, color="k", ls="--", lw=1.6, label="54 (T066)")
    ax.axhspan(SHARE_LO, SHARE_HI, color="gold", alpha=0.2,
               label=f"±25% band [{SHARE_LO:.0f}, {SHARE_HI:.0f}]")
    for xv, yv, lb in zip(xs, ys, xlabels):
        if np.isfinite(yv):
            ax.text(xv, yv + 2, f"{yv:.1f}", ha="center", fontsize=9)
        else:
            ax.text(xv, 5, "off-grid", ha="center", fontsize=8, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("r*(k) x k")
    ax.set_title(f"M2 products — {M['m2']['verdict']}", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(
        f"E098 seed ladder — M1: {M['m1']['verdict']} "
        f"({M['m1']['n_concentrated']}/{M['m1']['n_seeds']}) | "
        f"M2: {M['m2']['verdict']}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
