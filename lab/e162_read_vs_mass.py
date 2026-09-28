"""E162 — THE READ-vs-MASS FORK (R47 critic's final attack on T093's
READ-coupled claim; REGISTERED — bars fixed by the dispatch before compute).

WHY: e159's joint cell (mask + poison) was numerically identical to
mask-alone, but it removed reads AND mass TOGETHER — it cannot tell
whether the poisoned row 0 kills by what attention READS off it (corrupt
value content; READ-coupled, T093's claim on trial) or by what it STEALS
(the degraded row 0 is a 10.8x attention absorber — total mass on key 0
0.157 -> 1.693 — starving informative positions; MASS-coupled, the
alternative). The two cells below separate CONTENT from ALLOCATION for
the first time.

THE TWO CELLS (+ one measurement):
  (i) VALUE-RESTORE-UNDER-POISON: wpe[0] norm -> 0.07 (the poison; the
      degraded key geometry preserved) BUT position-0's VALUE
      contribution is overridden with the healthy net's — a dual-stream
      forward keeps a clean copy (same weights, healthy wpe[0]) and at
      every layer replaces the V-projected vectors at key position 0
      with the clean stream's V-projection of the same input row.
      Measure fact (g-12 primary, g0) + CE. HEALS => the poison rode
      the CONTENT of reads (READ-coupled literal). KILLS => the poison
      is mass/allocation (MASS-coupled).
  (ii) MASS-INFLATE-ON-HEALTHY: healthy net (row 0 untouched) + an
      additive logit bias on key 0 (all layers/heads), swept to
      reproduce the poisoned net's attention-mass profile on key 0
      (target: e159's stored total, the 1.69x mean). Measure fact + CE.
      KILLS (>= 60% drop at moderate CE) => mass starvation ALONE
      suffices — MASS-coupled confirmed without any degradation of
      row 0's content. SPARES => mass alone is not sufficient; reads
      of degraded content matter.
  (iii) MEASUREMENT (report-only): under poison-only, the attention
      mass on the fact's read band — install-geometry windows with the
      name INSIDE the window: onset read at 129 + within-name reads
      130..135 (the dispatch's 128-137 band) and the novel geometry's
      shifted band (reads 141..148) — before/after the poison: direct
      evidence of starvation vs corruption.

REGISTERED PREDICTION (dispatch verbatim; no shopping):
  - READ-COUPLED-LITERAL fires if: (i) heals (retention >= 0.7) AND
    (ii) spares (retention >= 0.7) — the content of reads carries the
    kill.
  - MASS-COUPLED fires if: (i) kills AND/OR (ii) kills — allocation
    carries the kill; the intro's "dies of what attention reads"
    inverts to "dies of what the degraded pivot steals attention
    from"; T092's layer-4 rewrites.
  - MIXED if one heals and the other kills.
  - Texture => TEXTURE with numbers.

DECISION TABLE (the dispatch's "AND/OR" and "MIXED" sentences
reconciled BEFORE compute — written before any cell ran):
    (i)HEALS + (ii)HEALS  -> READ-COUPLED-LITERAL
    (i)KILLS + (ii)KILLS  -> MASS-COUPLED
    (i)KILLS + (ii)other  -> MASS-COUPLED  (cell (i) is the direct
                           test: healthy content at the absorber fails
                           to rescue => content is not the carrier)
    (i)HEALS + (ii)KILLS  -> MIXED
    anything else (a GAP in the decisive cell) -> TEXTURE with numbers.

OPERATIONALIZATIONS (fixed before compute):
  * retention = expr_arm / expr_base with expr_base = the SAME net's
    UNMODIFIED expression at that geometry; CE cost = CE_R(arm) -
    CE_R(same net, unmodified), e065 val-windows bank (seed 26502, 60
    windows) — a CE column on EVERY cell. Fact = install60 battery
    p(Z) at the last position; g-12 primary, g0 the consistency column.
  * a cell HEALS iff retention >= 0.70 at BOTH geometries (e159's
    MASK-HEALS convention: the health door must work everywhere); a
    cell KILLS iff retention <= 0.40 at the PRIMARY geometry g-12 (the
    dispatch's single criterion ">= 60% drop", read where the poison
    kills — poison-alone's own g-12 retention is 0.158); cell (ii)
    KILLS additionally requires CE cost <= +0.85 ("moderate CE" = no
    more wreckful than the state it emulates; poison-alone costs
    +0.843). g0 is the consistency column: if the g0 branch (same
    thresholds) disagrees with g-12's, the branch is suffixed -SPLIT
    and both numbers quoted — reported, never re-mapped. [BARS v2 —
    corrected after the smoke shakedown, before the full compute: v1
    had invented a g0 <= 0.50 kill conjunct absent from the dispatch,
    which the shakedown showed could acquit a 97% primary-geometry
    kill; the correction is the dispatch-literal reading. Documented
    in recipe_deviations.]
  * cell (ii) calibration: uniform additive bias b added to the key-0
    attention logit for queries >= 1, all layers/heads; swept then
    bisected to match e159's stored POISONED total mass on key 0 (sum
    over layers of mean-over-queries-ge1; read at runtime from
    runs/e159/metrics.json) on the install60@g0 battery — e159's exact
    instrument battery. The per-layer PROFILE is not matched (the
    poisoned profile is front-loaded; a uniform bias is flatter): the
    cell reproduces the allocation DOSE, not its distribution;
    profiles reported as texture. The bracketing coarse-grid biases
    run as full REPORT-ONLY riders (the dose dimension).
  * cell (i) instrument gates: the transplant forward with donor ==
    recipient (healthy row on the healthy net) reproduces the healthy
    battery bit-tight; on the poisoned net with ITS OWN values as
    donor it reproduces poison-alone (cross-gated vs e151's stored
    cells) — the dual-stream machinery is inert. The transplant
    replaces position-0's value vector for EVERY reader (queries
    1..T-1 and row 0's self-read), every layer, every head; the donor
    stream is the clean copy's own forward (same weights, healthy
    wpe[0]), so later-layer donations are healthy-context donations.
  * measurement (iii) builds NEW fact-in-window batteries (never
    gating): [pre-name context (130 tokens at g0 / 142 at g-12) +
    NAME + 11 continuation chars]; read queries = the onset read (the
    last pre-name position) + the 6 within-name reads + the first
    post-name read; key band = [onset-2, onset+7] (g0: 128..137,
    exactly the dispatch's band; the novel geometry's band shifts by
    +12). Per layer: the band queries' mean mass on key 0 / on the
    band keys / elsewhere, healthy vs poisoned.

INSTRUMENT PROVENANCE: e150/e159 verbatim — load_cpu/evl_load,
battery_fwd/ce_fwd (e068/e113/e120 lineage), val_windows (e065 seed
26502), modified_wpe (e131 confinement gate), forward_custom (causal
+ forced-off-sink; gated against the standard forward), the sink-mass
manual-softmax measurement (gated against e159's stored per-layer
values), and the protocol rebuild (corpus seed 1337, SPLICE_RNG host
shuffle, install-60 split, mix gate). e151's stored mask/poison cells
and e159's stored mass profile are read at runtime from
runs/{e151,e159}/metrics.json (auditable) and gate this run.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch
import; the GPU is user-occupied; e158 + e125a share the CPU — threads
capped at 4, all evals sequential, no busy-waiting), eval-only, no
training, minutes. Outputs: runs/e162/{metrics.json, read_vs_mass.png}.
No checkpoints written.

Run:  cd lab && python e162_read_vs_mass.py    (E162_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (GPU user-occupied)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(4)                              # LOW (shared CPU)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E162_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
CONS_CK = CKPT_DIR / "e131_consolidated_e113.pt"   # consolidated (primary)
E151_METRICS = E43.REPO / "runs" / "e151" / "metrics.json"   # stored joint-ingredient cells (gates)
E159_METRICS = E43.REPO / "runs" / "e159" / "metrics.json"   # stored sink-mass profile (gates + calibration target)

GEOS = (0, -12)                   # g0 (trained) + g-12 (novel; PRIMARY)
R_EVAL_SEED = 26502               # e065 CE_R bank seed (verbatim)

POISON_NORM = 0.07                # the poison bracket (e150/e151/e159)

# gates / references (full precision, from stored metrics)
G_BIT_TOL = 5e-6                  # loaded-checkpoint reproduction gate
G_CROSS_TOL = 1e-4                # cross-run gate (e151's own convention)
G_CONS_REF_PZ = 0.7850371599197388          # e131/e150/e151/e159 none g0 install60
G_CONS_REF_CE = 1.663516640663147           # e131/e150/e151/e159 none ce_r
G_CONS_REF_GM12 = 0.9155886173248291        # e150/e159 base g-12 install60
G_E151_MASK_G0 = 0.808849573135376          # e151 before.mask g+0 mask_pz
G_E151_MASK_GM12 = 0.9242185950279236       # e151 before.mask g-12 mask_pz
G_E151_MASK_CE = 1.6974624395370483         # e151 before.mask ce_mask
G_E151_P07_G0 = 0.3231039047241211          # e151 before.ladder 0.07 g+0
G_E151_P07_GM12 = 0.14433333277702332       # e151 before.ladder 0.07 g-12
G_E151_P07_CE = 2.506317377090454           # e151 before.ladder 0.07 ce_r
G_MASS_TOL = 1e-4                 # mass-instrument cross-gate vs e159

# registered bars (numeric; BARS v2 — dispatch-literal, see docstring)
HEAL_RET = 0.70                   # HEALS: retention >= 0.70 at BOTH geos
KILL_RET = 0.40                   # KILLS: retention <= 0.40 at g-12 (primary)
CELL2_KILL_CE = 0.85              # cell (ii) "moderate CE": no worse than
                                  # the emulated state (poison +0.843)

# bias calibration (cell ii)
BIAS_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
BIAS_GRID_SMOKE = (0.5, 1.5, 3.0)
BISECT_ITERS = 14 if not SMOKE else 7

REGISTERED_PREDICTION = {
    "dispatch_bars_verbatim": {
        "READ-COUPLED-LITERAL": "fires if: (i) heals (retention >= 0.7) "
                                "AND (ii) spares (retention >= 0.7) — the "
                                "content of reads carries the kill.",
        "MASS-COUPLED": "fires if: (i) kills AND/OR (ii) kills — allocation "
                        "carries the kill; the intro's 'dies of what "
                        "attention reads' inverts to 'dies of what the "
                        "degraded pivot steals attention from'; T092's "
                        "layer-4 rewrites.",
        "MIXED": "if one heals and the other kills.",
        "texture": "Texture => TEXTURE with numbers.",
    },
    "decision_table_pre_compute": {
        "(i)HEALS+(ii)HEALS": "READ-COUPLED-LITERAL",
        "(i)KILLS+(ii)KILLS": "MASS-COUPLED",
        "(i)KILLS+(ii)other": "MASS-COUPLED (cell (i) is the direct test: "
                              "healthy content at the absorber fails to "
                              "rescue => content is not the carrier)",
        "(i)HEALS+(ii)KILLS": "MIXED",
        "else": "TEXTURE with numbers (a GAP in the decisive cell)",
    },
    "operationalizations": (
        "retention = expr_arm / expr_base(healthy same net, same geometry); "
        "CE cost vs same-net unmodified CE_R (e065 bank seed 26502); a cell "
        "HEALS iff retention >= 0.70 at BOTH g0 and g-12; KILLS iff "
        "retention <= 0.40 at the PRIMARY geometry g-12 (the dispatch's "
        "' >= 60% drop', read where the poison kills: poison-alone's own "
        "g-12 retention is 0.158); cell (ii) KILLS additionally requires "
        "CE cost <= +0.85 ('moderate' = no more wreckful than the emulated "
        "poison at +0.843). g0 is the consistency column: a disagreeing g0 "
        "branch suffixes -SPLIT (reported, never re-mapped). BARS v2: "
        "corrected after the smoke shakedown and before the full compute "
        "(v1's invented g0<=0.50 kill conjunct is not in the dispatch and "
        "could acquit a 97% primary-geometry kill — the correction is the "
        "dispatch-literal reading; see recipe_deviations). Cell (ii) "
        "calibration = uniform additive key-0 logit bias (queries >= 1, "
        "all layers/heads) swept+bisected to e159's stored poisoned TOTAL "
        "key-0 mass on the install60@g0 battery; per-layer profile NOT "
        "matched (dose, not distribution); bracketing grid biases run as "
        "report-only riders. Cell (i) = dual-stream forward, poisoned q/k "
        "with the clean copy's v at key 0 every layer/head/reader; donor-"
        "identity and donor-self instrument gates must be bit-tight; a "
        "v0 content-delta audit quantifies whether the poison perturbs the "
        "value channel at all (cell (i)'s power). Measurement (iii) uses "
        "NEW fact-in-window batteries (name inside the window; g0 reads "
        "129..136, band 128..137; g-12 shifted +12) — report-only, never "
        "gating."),
    "no_bar_shopping": "No bar shopping. Texture => TEXTURE with numbers.",
}

recipe_deviations: list[str] = [
    "Eval-only battery: the consolidated net is a LOADED, gated artifact "
    "(e131 lineage); nothing regenerated, nothing trained.",
    "SHAKEDOWN CORRECTION 1 (found by E162_SMOKE, fixed before the full "
    "compute): the value-transplant forward's donor default was inverted — "
    "donor_wpe0=None fell back to the net's OWN (poisoned) row, making "
    "cell (i) a literal no-op (the smoke cell matched poison-alone "
    "exactly); fixed so the default donor is the CLEAN row. The smoke "
    "run's cell-(i) numbers are therefore invalid; the full run's are "
    "not affected by the bug.",
    "SHAKEDOWN CORRECTION 2 (bars v1 -> v2, before the full compute): v1 "
    "had invented a kill conjunct (g0 <= 0.50) absent from the dispatch, "
    "which the shakedown showed could acquit a 97% primary-geometry kill; "
    "v2 is the dispatch-literal reading (kill = >= 60% drop at the "
    "primary geometry g-12 at moderate CE; heal = >= 0.70 at BOTH geos; "
    "g0 disagreement => -SPLIT suffix, reported, never re-mapped). Both "
    "the docstring and REGISTERED_PREDICTION carry v2; this entry is the "
    "audit trail.",
    "arm_b secondary column OMITTED (dispatch: 'if cheap'): its fact "
    "survives the poison outright (e159 SITE-SPARED, x0.922 at norm 0.07), "
    "so the value-restore / mass-inflate cells have no kill to explain on "
    "that net; omitted for CPU thrift while e158/e125a share the machine — "
    "stated, not silent.",
    "Cell (i)'s transplant replaces position-0's value vector for EVERY "
    "reader (queries 1..T-1 and row 0's self-read), all layers/heads — the "
    "dispatch's 'override position-0's VALUE contribution' taken literally; "
    "the donor stream is the clean copy's own forward, so later-layer "
    "donations are healthy-context donations (the cell asks: is healthy "
    "CONTENT at the absorber sufficient?).",
    "Cell (ii) calibrates the uniform bias to the TOTAL key-0 mass on the "
    "g0 battery (e159's instrument battery); the per-layer profile is not "
    "matched — the cell reproduces the allocation DOSE, not its "
    "distribution; the bracketing coarse-grid biases run as full "
    "report-only riders.",
    "Measurement (iii) builds fact-in-window batteries (the name INSIDE the "
    "window — onset read 129 + within-name reads at g0, shifted +12 at "
    "g-12): a NEW battery for the mass decomposition, report-only; the "
    "registered fact cells remain the install60 last-position battery.",
    "held30 battery not rebuilt (no registered cell uses it); the e151 "
    "mask/poison cells and e159's mass profile are read at runtime from "
    "runs/{e151,e159}/metrics.json (auditable), not hand-copied.",
]

trims: list[str] = []


# ------------------------------------------------------------------ instruments
# (e150/e159 verbatim: load_cpu/evl_load/battery_fwd/ce_fwd/val_windows/
#  modified_wpe/forward_custom/fwd_causal/fwd_offsink; mass_profile is
#  e159's sink_mass_profile extended with (a) an additive key-0 logit bias
#  and (b) a band decomposition. New: forward_value_transplant,
#  forward_bias.)

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
def battery_fwd(net: TinyGPT, ids: torch.Tensor, zid: int, fwd=None,
                bs=30) -> dict:
    """e068/e113/e120 battery on CPU: p(Z) at the last position."""
    net.eval()
    f = fwd if fwd is not None else net
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = f(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "std_pz": float(p.std()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0]}


@torch.no_grad()
def ce_fwd(net: TinyGPT, x, y, fwd=None, bs=64) -> float:
    f = fwd if fwd is not None else net
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = f(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    g = torch.Generator().manual_seed(seed)
    out_x, out_y = [], []
    tries = 0
    while len(out_y) < n and tries < 500 * n:
        i = int(torch.randint(len(val_ids) - block - 1, (1,), generator=g))
        txt = val_text[i: i + block + 1]
        tries += 1
        if "ZEPHYRA" in txt or "ZEPH" in txt:
            continue
        out_x.append(val_ids[i: i + block])
        out_y.append(val_ids[i + 1: i + 1 + block])
    return torch.stack(out_x), torch.stack(out_y)


def modified_wpe(sd: dict, row: int, value) -> tuple[dict, dict]:
    """e131's confinement gate, generalized to row-value surgery: at most
    `row`'s elements change, every other tensor bit-identical."""
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


def _causal_mask(T: int, device) -> torch.Tensor:
    m = torch.zeros(T, T, device=device)
    m.masked_fill_(torch.triu(torch.ones(T, T, device=device,
                                         dtype=torch.bool), 1),
                   float("-inf"))
    return m


@torch.no_grad()
def forward_custom(net: TinyGPT, idx, targets=None, block_key0=False):
    """common.TinyGPT.forward replicated with an explicit additive attention
    mask (e150/e159 verbatim). block_key0=True: attention TO key position 0
    blocked for queries 1..T-1, ALL layers, ALL heads."""
    B, T = idx.shape
    C = net.cfg.n_embd
    H = net.cfg.n_head
    D = C // H
    pos = torch.arange(T, device=idx.device)
    x = net.wte(idx) + net.wpe(pos)
    mask = _causal_mask(T, idx.device)
    if block_key0:
        mask[:, 0] = float("-inf")
        mask[0, 0] = 0.0                 # keep row 0's only legal key
    for block in net.h:
        xin = block.ln1(x)
        q, k, v = block.attn.c_attn(xin).split(C, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + block.attn.c_proj(y)
        x = x + block.mlp(block.ln2(x))
    logits = net.lm_head(net.ln_f(x))
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1))
    return logits, loss


def fwd_causal(net):
    return lambda idx, targets=None: forward_custom(
        net, idx, targets=targets, block_key0=False)


def fwd_offsink(net):
    return lambda idx, targets=None: forward_custom(
        net, idx, targets=targets, block_key0=True)


@torch.no_grad()
def forward_value_transplant(net: TinyGPT, idx, clean_wpe0, targets=None,
                             donor_wpe0=None):
    """CELL (i)'s instrument — the dual-stream value transplant.

    The recipient stream uses the net's LOADED state (the poison: wpe[0]
    norm 0.07) for embeddings, queries and keys — the degraded key geometry
    is preserved. A donor stream, built with `donor_wpe0` in row 0 (default
    `clean_wpe0` = the healthy copy's row), runs the same blocks in
    lockstep; at every layer, the recipient's V-projected vectors at key
    position 0 are REPLACED by the donor's (all heads, all batch, every
    reader — queries 1..T-1 and row 0's self-read). The donor stream's row
    0 defaults to `clean_wpe0` (the healthy copy); passing donor_wpe0 =
    the net's own row is the instrument self-gate (transplant becomes a
    no-op; must reproduce the plain forward bit-tight)."""
    B, T = idx.shape
    C = net.cfg.n_embd
    H = net.cfg.n_head
    D = C // H
    pos = torch.arange(T, device=idx.device)
    donor_row = clean_wpe0 if donor_wpe0 is None else donor_wpe0
    wpe_d = net.wpe.weight.clone()
    wpe_d[0] = donor_row
    xp = net.wte(idx) + net.wpe(pos)            # recipient (poisoned) stream
    xd = net.wte(idx) + wpe_d[pos]              # donor (clean) stream
    mask = _causal_mask(T, idx.device)
    for block in net.h:
        qd, kd, vd = block.attn.c_attn(block.ln1(xd)).split(C, dim=2)
        qd = qd.view(B, T, H, D).transpose(1, 2)
        kd = kd.view(B, T, H, D).transpose(1, 2)
        vd = vd.view(B, T, H, D).transpose(1, 2)
        qp, kp, vp = block.attn.c_attn(block.ln1(xp)).split(C, dim=2)
        qp = qp.view(B, T, H, D).transpose(1, 2)
        kp = kp.view(B, T, H, D).transpose(1, 2)
        vp = vp.view(B, T, H, D).transpose(1, 2)
        vp[:, :, 0, :] = vd[:, :, 0, :]          # THE TRANSPLANT (key 0)
        y = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        xp = xp + block.attn.c_proj(y)
        xp = xp + block.mlp(block.ln2(xp))
        # the donor stream evolves under its own plain attention
        yd = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=mask)
        yd = yd.transpose(1, 2).contiguous().view(B, T, C)
        xd = xd + block.attn.c_proj(yd)
        xd = xd + block.mlp(block.ln2(xd))
    logits = net.lm_head(net.ln_f(xp))          # the recipient's logits
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1))
    return logits, loss


@torch.no_grad()
def forward_bias(net: TinyGPT, idx, targets=None, bias=0.0):
    """CELL (ii)'s instrument — the healthy net with an additive logit bias
    `bias` on key 0 for queries 1..T-1, ALL layers, ALL heads, eval only.
    Row 0 is untouched (its only legal key; a bias there is softmax-invariant
    anyway)."""
    B, T = idx.shape
    C = net.cfg.n_embd
    H = net.cfg.n_head
    D = C // H
    pos = torch.arange(T, device=idx.device)
    x = net.wte(idx) + net.wpe(pos)
    mask = _causal_mask(T, idx.device)
    mask[1:, 0] += bias
    for block in net.h:
        xin = block.ln1(x)
        q, k, v = block.attn.c_attn(xin).split(C, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + block.attn.c_proj(y)
        x = x + block.mlp(block.ln2(x))
    logits = net.lm_head(net.ln_f(x))
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1))
    return logits, loss


@torch.no_grad()
def mass_profile(net: TinyGPT, ids: torch.Tensor, bias=None,
                 read_queries=None, key_band=None, bs=30) -> dict:
    """e159's sink_mass_profile extended. Per layer (mean over heads/batch,
    manual-softmax lineage): read_pos = mass on key 0 at query T-1;
    mean_queries_ge1 = mass on key 0 averaged over queries 1..T-1 (e159's
    quantities, gated against its stored values). Optional `bias` adds the
    key-0 logit bias (cell ii's calibration knob). If read_queries +
    key_band=(lo,hi) are given: band queries' mean mass split into key 0 /
    band keys / elsewhere (measurement iii)."""
    cfg = net.cfg
    H, D = cfg.n_head, cfg.n_embd // cfg.n_head
    per_layer = {l: {"read": [], "allq": [], "b0": [], "bband": []}
                 for l in range(cfg.n_layer)}
    for i in range(0, ids.shape[0], bs):
        x = ids[i:i + bs]
        B, Tt = x.shape
        emb = net.wte(x) + net.wpe(torch.arange(Tt))
        causal = torch.triu(torch.ones(Tt, Tt, dtype=torch.bool), 1)
        for l, block in enumerate(net.h):
            xin = block.ln1(emb)
            q, k, v = block.attn.c_attn(xin).split(cfg.n_embd, dim=2)
            q = q.view(B, Tt, H, D).transpose(1, 2)
            k = k.view(B, Tt, H, D).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / (D ** 0.5)
            att = att.masked_fill(causal, float("-inf"))
            if bias is not None:
                att[:, :, 1:, 0] += bias
            att = F.softmax(att, dim=-1)
            per_layer[l]["read"].append(float(att[:, :, Tt - 1, 0].mean()))
            per_layer[l]["allq"].append(float(att[:, :, 1:, 0].mean()))
            if read_queries is not None and key_band is not None:
                lo, hi = key_band
                for qpos in read_queries:
                    m0 = att[:, :, qpos, 0]
                    mband = att[:, :, qpos, lo:min(qpos, hi) + 1].sum(-1)
                    per_layer[l]["b0"].append(float(m0.mean()))
                    per_layer[l]["bband"].append(float(mband.mean()))
            y = (att @ v.view(B, Tt, H, D).transpose(1, 2))
            emb = emb + block.attn.c_proj(
                y.transpose(1, 2).contiguous().view(B, Tt, cfg.n_embd))
            emb = emb + block.mlp(block.ln2(emb))
    out = {}
    for l, v in per_layer.items():
        ent = {"read_pos": float(np.mean(v["read"])),
               "mean_queries_ge1": float(np.mean(v["allq"]))}
        if v["b0"]:
            m0 = float(np.mean(v["b0"]))
            mb = float(np.mean(v["bband"]))
            ent.update({"band_q_key0": m0, "band_q_bandkeys": mb,
                        "band_q_elsewhere": 1.0 - m0 - mb})
        out[f"L{l}"] = ent
    return out


def mass_total(prof: dict) -> float:
    """e159's quantity: total attention mass on key 0 = sum over layers of
    mean-over-queries(>=1) mass (healthy ~0.157, poisoned ~1.693)."""
    return float(sum(v["mean_queries_ge1"] for v in prof.values()))


@torch.no_grad()
def v0_delta_profile(net: TinyGPT, ids: torch.Tensor, clean_wpe0,
                     bs=30) -> dict:
    """Content-channel audit (report-only): how much does the poison change
    WHAT IS READ at key 0? Dual-stream (poisoned vs clean row 0, same
    blocks); per layer the relative delta ||v0_clean - v0_pois|| / ||v0_clean||
    of the V-projected vectors at key position 0 (mean over heads/batch),
    plus ||v0_clean|| magnitudes. If this is ~0 the value content read off
    key 0 is barely perturbed and cell (i) is weakly powered by
    construction — the fork then leans on cell (ii)."""
    cfg = net.cfg
    H, D = cfg.n_head, cfg.n_embd // cfg.n_head
    wpe_d = net.wpe.weight.clone()
    wpe_d[0] = clean_wpe0
    per_layer = {l: {"rel": [], "norm": []} for l in range(cfg.n_layer)}
    for i in range(0, ids.shape[0], bs):
        x = ids[i:i + bs]
        B, Tt = x.shape
        pos = torch.arange(Tt)
        xp = net.wte(x) + net.wpe(pos)             # poisoned stream
        xd = net.wte(x) + wpe_d[pos]               # clean stream
        causal = torch.triu(torch.ones(Tt, Tt, dtype=torch.bool), 1)
        mask = torch.zeros(Tt, Tt)
        mask.masked_fill_(causal, float("-inf"))
        for l, block in enumerate(net.h):
            streams = {}
            for tag, st in (("pois", xp), ("clean", xd)):
                q, k, v = block.attn.c_attn(block.ln1(st)).split(cfg.n_embd,
                                                                 dim=2)
                streams[tag] = (q.view(B, Tt, H, D).transpose(1, 2),
                                k.view(B, Tt, H, D).transpose(1, 2),
                                v.view(B, Tt, H, D).transpose(1, 2))
            v0p = streams["pois"][2][:, :, 0]       # (B, H, D) at key 0
            v0d = streams["clean"][2][:, :, 0]
            nd = v0d.norm(dim=-1)                   # (B, H)
            per_layer[l]["norm"].append(float(nd.mean()))
            per_layer[l]["rel"].append(
                float(((v0d - v0p).norm(dim=-1) / (nd + 1e-12)).mean()))
            # evolve both streams plainly (no transplant — this is the audit)
            for st, (qq, kk, vv) in ((xp, streams["pois"]),
                                     (xd, streams["clean"])):
                y = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask)
                y = y.transpose(1, 2).contiguous().view(B, Tt, cfg.n_embd)
                st.copy_(st + block.attn.c_proj(y))
                st.copy_(st + block.mlp(block.ln2(st)))
    return {f"L{l}": {"v0_rel_delta": float(np.mean(v["rel"])),
                      "v0_clean_norm": float(np.mean(v["norm"]))}
            for l, v in per_layer.items()}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e162_smoke" if SMOKE else "e162")
    log(f"E162 THE READ-vs-MASS FORK (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e150/e159 verbatim)
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
    log(f"protocol rebuilt: install60 {mix}")

    name_ids = corpus.encode(NAME)
    assert len(name_ids) == 7

    n_install = 12 if SMOKE else 60
    occ_use = install_occ[:n_install]

    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in occ_use]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text,
                                     12 if SMOKE else 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- measurement-iii batteries: the fact INSIDE the window
    # g0: pre=130, name at 130..136, reads 129..136, band 128..137
    # g-12: pre=142, name at 142..148, reads 141..148, band 140..149
    band_win, band_meta = {}, {}
    for j in GEOS:
        wins, prelen = [], PRE - j
        for p, h in occ_use:
            pre = train_text[p - prelen: p]
            post = train_text[p + len(h): p + len(h) + 11]
            w = pre + NAME + post
            assert len(w) == prelen + len(NAME) + 11
            wins.append(corpus.encode(w))
        band_win[j] = torch.stack(wins)
        band_meta[j] = {
            "pre_len": prelen, "name_cols": [prelen, prelen + 6],
            "read_queries": list(range(prelen - 1, prelen + 7)),
            "key_band": [prelen - 2, prelen + 7],
            "window_len": prelen + len(NAME) + 11}
        log(f"band battery g{j:+d}: reads {band_meta[j]['read_queries'][0]}"
            f"..{band_meta[j]['read_queries'][-1]}, keys "
            f"{band_meta[j]['key_band']}, T={band_meta[j]['window_len']}")

    # ---------------- stored cells (read at runtime, auditable)
    e151 = json.loads(E151_METRICS.read_text(encoding="utf-8"))
    e159 = json.loads(E159_METRICS.read_text(encoding="utf-8"))
    sm_stored = e159["probe_a_mask_ladder_joint"]["sink_mass_premask"]
    mass_healthy_stored = {k: v["mean_queries_ge1"]
                           for k, v in sm_stored["healthy"].items()}
    mass_poison_stored = {k: v["mean_queries_ge1"]
                          for k, v in sm_stored["poison07"].items()}
    MASS_TARGET = float(sum(mass_poison_stored.values()))
    MASS_HEALTHY = float(sum(mass_healthy_stored.values()))
    log(f"e159 stored mass: healthy {MASS_HEALTHY:.4f} -> poisoned "
        f"{MASS_TARGET:.4f} (x{MASS_TARGET / MASS_HEALTHY:.2f}); "
        f"e151 stored mask/poison cells loaded for gating")

    # ---------------- nets + gates
    log("--- PHASE 0: load + gate the artifact ---")
    gates: dict = {}

    net_cons = load_cpu(CONS_CK)
    bz_cons = {j: battery_fwd(net_cons, bat_ids[j], zid) for j in GEOS}
    ce_cons = ce_fwd(net_cons, *r_eval_xy)
    if not SMOKE:
        gates["consolidated"] = {
            "battery_pz_g0": bz_cons[0]["mean_pz"], "ref_g0": G_CONS_REF_PZ,
            "battery_pz_gm12": bz_cons[-12]["mean_pz"],
            "ref_gm12": G_CONS_REF_GM12,
            "ce_r": ce_cons, "ref_ce": G_CONS_REF_CE,
            "pass": bool(abs(bz_cons[0]["mean_pz"] - G_CONS_REF_PZ) < G_BIT_TOL
                         and abs(bz_cons[-12]["mean_pz"] - G_CONS_REF_GM12)
                         < G_BIT_TOL
                         and abs(ce_cons - G_CONS_REF_CE) < G_BIT_TOL)}
    else:
        gates["consolidated"] = {"pass": True,
                                 "note": "smoke: reduced battery, no bit gate"}
    log(f"G_CONS: p(Z) g0 {bz_cons[0]['mean_pz']:.10f} g-12 "
        f"{bz_cons[-12]['mean_pz']:.10f} CE_R {ce_cons:.6f}: "
        f"{'PASS' if gates['consolidated']['pass'] else 'FAIL'}")
    if not gates["consolidated"]["pass"]:
        raise RuntimeError("consolidated checkpoint failed its gate")

    sd_cons = {k: v.clone() for k, v in net_cons.state_dict().items()}
    wpe0_cons = sd_cons["wpe.weight"][0].clone()
    n0_cons = float(wpe0_cons.norm())
    sd_p07, gate_p07 = modified_wpe(sd_cons, 0,
                                    wpe0_cons * (POISON_NORM / n0_cons))
    assert gate_p07["pass"], gate_p07
    ev = evl_load(sd_cons)                    # reusable eval twin
    evp = evl_load(sd_p07)                    # poisoned twin (cell i recipient)

    @torch.no_grad()
    def state_eval(label, fwd=None, net=None):
        """Battery at both geos + CE for the twin's current state."""
        m = net if net is not None else ev
        f = fwd
        out = {f"g{j}": battery_fwd(m, bat_ids[j], zid, fwd=f)["mean_pz"]
               for j in GEOS}
        out["ce_r"] = ce_fwd(m, *r_eval_xy, fwd=f)
        log(f"  [{label:30s}] " + " ".join(
            f"{k} {v:.4f}" for k, v in out.items()))
        return out

    # (0) healthy base + custom-causal instrument gate
    base_out = state_eval("cons none (healthy)")
    if not SMOKE:
        assert abs(base_out["g0"] - G_CONS_REF_PZ) < G_BIT_TOL
    with torch.no_grad():
        lg_std, _ = net_cons(bat_ids[0][:6])
        lg_cus, _ = forward_custom(net_cons, bat_ids[0][:6], block_key0=False)
        dmax = float((lg_std - lg_cus).abs().max())
    pz_cus = battery_fwd(net_cons, bat_ids[0], zid,
                         fwd=fwd_causal(net_cons))["mean_pz"]
    dpz = abs(pz_cus - bz_cons[0]["mean_pz"])
    gates["mask_instrument"] = {
        "max_abs_logit_diff_batch6": dmax, "abs_dpz_g0": dpz,
        "tol_logit": 1e-2, "tol_pz": 1e-3,
        "pass": bool(dmax < 1e-2 and dpz < 1e-3)}
    log(f"MASK-GATE: max|dlogit| {dmax:.3e}, |dp(Z)| {dpz:.3e}: "
        f"{'PASS' if gates['mask_instrument']['pass'] else 'FAIL'}")
    if not gates["mask_instrument"]["pass"]:
        raise RuntimeError("custom forward does not reproduce the standard one")

    # (1) transplant instrument gates: donor==recipient must be no-ops
    ev.load_state_dict(sd_cons)
    vt_h = state_eval("VT identity (healthy donor=recipient)",
                      fwd=lambda i, t=None: forward_value_transplant(
                          ev, i, wpe0_cons, targets=t, donor_wpe0=wpe0_cons))
    evp.load_state_dict(sd_p07)
    vt_p = state_eval("VT self (poison donor=recipient)",
                      fwd=lambda i, t=None: forward_value_transplant(
                          evp, i, wpe0_cons, targets=t,
                          donor_wpe0=sd_p07["wpe.weight"][0]))
    gates["vt_identity"] = {
        "g0": {"value": vt_h["g0"], "ref": base_out["g0"],
               "pass": bool(abs(vt_h["g0"] - base_out["g0"]) < G_BIT_TOL)},
        "gm12": {"value": vt_h["g-12"], "ref": base_out["g-12"],
                 "pass": bool(abs(vt_h["g-12"] - base_out["g-12"])
                              < G_BIT_TOL)},
        "ce": {"value": vt_h["ce_r"], "ref": base_out["ce_r"],
               "pass": bool(abs(vt_h["ce_r"] - base_out["ce_r"])
                            < G_BIT_TOL)}}
    log(f"VT-IDENTITY: g0 {vt_h['g0']:.10f} g-12 {vt_h['g-12']:.10f} CE "
        f"{vt_h['ce_r']:.6f}: "
        f"{'PASS' if all(v['pass'] for v in gates['vt_identity'].values()) else 'FAIL'}")
    if not all(v["pass"] for v in gates["vt_identity"].values()):
        raise RuntimeError("value-transplant machinery is not inert (identity)")

    CELLS: list[dict] = []

    def cell(probe, tag, geo, expr_base, expr_arm, ce_base, ce_arm,
             rider=False, extra=None):
        ret = expr_arm / max(expr_base, 1e-12)
        c = {"probe": probe, "tag": tag, "net": "consolidated", "geo": geo,
             "expr_base": expr_base, "expr_arm": expr_arm,
             "retention": ret, "fact_drop_pct": 100.0 * (1.0 - ret),
             "ce_base": ce_base, "ce_arm": ce_arm,
             "ce_cost": ce_arm - ce_base, "rider": bool(rider)}
        if extra:
            c.update(extra)
        CELLS.append(c)
        log(f"  [{probe} | {tag:30s}] expr {expr_base:.4f} -> {expr_arm:.4f} "
            f"(x{ret:.3f}, drop {100 * (1 - ret):5.1f}%) | CE {ce_base:.4f} "
            f"-> {ce_arm:.4f} (cost {ce_arm - ce_base:+.4f})"
            + ("  [rider]" if rider else ""))
        return c

    # (2) anchor cells rebuilt + gated vs e151 stored
    if not SMOKE:
        ev.load_state_dict(sd_cons)
        m_out = state_eval("mask alone", fwd=fwd_offsink(ev))
        g_e151_mask = {
            "g0": {"value": m_out["g0"], "ref": G_E151_MASK_G0,
                   "pass": bool(abs(m_out["g0"] - G_E151_MASK_G0)
                                < G_CROSS_TOL)},
            "gm12": {"value": m_out["g-12"], "ref": G_E151_MASK_GM12,
                     "pass": bool(abs(m_out["g-12"] - G_E151_MASK_GM12)
                                  < G_CROSS_TOL)},
            "ce": {"value": m_out["ce_r"], "ref": G_E151_MASK_CE,
                   "pass": bool(abs(m_out["ce_r"] - G_E151_MASK_CE)
                                < G_CROSS_TOL)}}
        gates["e151_mask_cells"] = g_e151_mask
        log(f"G_E151_MASK: g0 {m_out['g0']:.10f} g-12 {m_out['g-12']:.10f} "
            f"CE {m_out['ce_r']:.6f}: "
            f"{'PASS' if all(v['pass'] for v in g_e151_mask.values()) else 'FAIL'}")
        if not all(v["pass"] for v in g_e151_mask.values()):
            raise RuntimeError("mask-alone does not reproduce e151's cells")
        ev.load_state_dict(sd_p07)
        p07_out = state_eval(f"poison {POISON_NORM} alone")
        g_e151_p07 = {
            "g0": {"value": p07_out["g0"], "ref": G_E151_P07_G0,
                   "pass": bool(abs(p07_out["g0"] - G_E151_P07_G0)
                                < G_CROSS_TOL)},
            "gm12": {"value": p07_out["g-12"], "ref": G_E151_P07_GM12,
                     "pass": bool(abs(p07_out["g-12"] - G_E151_P07_GM12)
                                  < G_CROSS_TOL)},
            "ce": {"value": p07_out["ce_r"], "ref": G_E151_P07_CE,
                   "pass": bool(abs(p07_out["ce_r"] - G_E151_P07_CE)
                                < G_CROSS_TOL)}}
        gates["e151_poison07_cells"] = g_e151_p07
        log(f"G_E151_P07: g0 {p07_out['g0']:.10f} g-12 {p07_out['g-12']:.10f} "
            f"CE {p07_out['ce_r']:.6f}: "
            f"{'PASS' if all(v['pass'] for v in g_e151_p07.values()) else 'FAIL'}")
        if not all(v["pass"] for v in g_e151_p07.values()):
            raise RuntimeError("poison-alone does not reproduce e151's cells")
    else:
        ev.load_state_dict(sd_cons)
        m_out = state_eval("mask alone", fwd=fwd_offsink(ev))
        ev.load_state_dict(sd_p07)
        p07_out = state_eval(f"poison {POISON_NORM} alone")
        gates["e151_mask_cells"] = {"pass": True,
                                    "note": "smoke: no cross gate"}
        gates["e151_poison07_cells"] = {"pass": True,
                                        "note": "smoke: no cross gate"}

    # donor-self gate on the poisoned side: the transplant machinery with
    # the poisoned net's OWN values as donor must reproduce poison-alone
    # bit-tight (the dual-stream machinery is inert in the poisoned state)
    gates["vt_self"] = {
        "g0": {"value": vt_p["g0"], "ref": p07_out["g0"],
               "pass": bool(abs(vt_p["g0"] - p07_out["g0"]) < G_BIT_TOL)},
        "gm12": {"value": vt_p["g-12"], "ref": p07_out["g-12"],
                 "pass": bool(abs(vt_p["g-12"] - p07_out["g-12"])
                              < G_BIT_TOL)},
        "ce": {"value": vt_p["ce_r"], "ref": p07_out["ce_r"],
               "pass": bool(abs(vt_p["ce_r"] - p07_out["ce_r"])
                            < G_BIT_TOL)}}
    gates["vt_self"]["pass"] = all(v["pass"]
                                   for k, v in gates["vt_self"].items()
                                   if k != "pass")
    log(f"VT-SELF: g0 {vt_p['g0']:.10f} g-12 {vt_p['g-12']:.10f} CE "
        f"{vt_p['ce_r']:.6f} vs poison-alone: "
        f"{'PASS' if gates['vt_self']['pass'] else 'FAIL'}")
    if not gates["vt_self"]["pass"]:
        raise RuntimeError("value-transplant machinery is not inert (self)")

    # (3) mass-instrument gate + measurement (iii)
    ev.load_state_dict(sd_cons)
    prof_h = mass_profile(ev, bat_ids[0])
    ev.load_state_dict(sd_p07)
    prof_p = mass_profile(ev, bat_ids[0])
    gates["mass_instrument"] = {
        "healthy_max_abs_dev": max(abs(prof_h[k]["mean_queries_ge1"] - v)
                                   for k, v in mass_healthy_stored.items()),
        "poison_max_abs_dev": max(abs(prof_p[k]["mean_queries_ge1"] - v)
                                  for k, v in mass_poison_stored.items()),
        "tol": G_MASS_TOL,
        "pass": bool(max(abs(prof_h[k]["mean_queries_ge1"] - v)
                         for k, v in mass_healthy_stored.items()) < G_MASS_TOL
                     and max(abs(prof_p[k]["mean_queries_ge1"] - v)
                             for k, v in mass_poison_stored.items())
                     < G_MASS_TOL)}
    if SMOKE:
        gates["mass_instrument"]["pass"] = True
        gates["mass_instrument"]["note"] = (
            "smoke: reduced battery (12 windows), no cross gate")
    log(f"MASS-GATE: healthy total {mass_total(prof_h):.6f} poisoned total "
        f"{mass_total(prof_p):.6f} (stored {MASS_HEALTHY:.6f} / "
        f"{MASS_TARGET:.6f}): "
        f"{'PASS' if gates['mass_instrument']['pass'] else 'FAIL'}")
    if not gates["mass_instrument"]["pass"]:
        raise RuntimeError("mass instrument does not reproduce e159's profile")

    # anchor cells into the CELLS table
    for j in GEOS:
        cell("ANCHOR", f"mask@g{j:+d}", j, base_out[f"g{j}"], m_out[f"g{j}"],
             base_out["ce_r"], m_out["ce_r"],
             extra={"note": "forced-off-sink mask alone (e150 P2 / e151 "
                            "before.mask, rebuilt + gated)"})
        cell("ANCHOR", f"poison{POISON_NORM}@g{j:+d}", j,
             base_out[f"g{j}"], p07_out[f"g{j}"], base_out["ce_r"],
             p07_out["ce_r"],
             extra={"target_norm": POISON_NORM, "gate": gate_p07,
                    "note": "norm 0.07 alone (e150 P5 / e151 before.ladder, "
                            "rebuilt + gated)"})

    # =====================================================================
    # MEASUREMENT (iii) — the starvation read-out (report-only)
    # =====================================================================
    log("--- MEASUREMENT (iii): fact-band mass, healthy vs poisoned ---")
    meas: dict = {"geometries": {}, "note": (
        "per layer, the fact-band read queries' mean attention mass split "
        "into key 0 / band keys / elsewhere (fact-in-window batteries, "
        "name INSIDE the window; g0 band 128..137, g-12 band 140..149). "
        "REPORT-ONLY, never gating.")}
    for j in GEOS:
        meta = band_meta[j]
        ev.load_state_dict(sd_cons)
        mh = mass_profile(ev, band_win[j],
                          read_queries=meta["read_queries"],
                          key_band=tuple(meta["key_band"]))
        ev.load_state_dict(sd_p07)
        mp = mass_profile(ev, band_win[j],
                          read_queries=meta["read_queries"],
                          key_band=tuple(meta["key_band"]))
        meas["geometries"][f"g{j:+d}"] = {
            "battery": meta, "healthy": mh, "poison07": mp,
            "band_key0_mean_over_layers": {
                "healthy": float(np.mean([v["band_q_key0"]
                                          for v in mh.values()])),
                "poison07": float(np.mean([v["band_q_key0"]
                                           for v in mp.values()]))},
            "band_bandkeys_mean_over_layers": {
                "healthy": float(np.mean([v["band_q_bandkeys"]
                                          for v in mh.values()])),
                "poison07": float(np.mean([v["band_q_bandkeys"]
                                           for v in mp.values()]))},
            "sink_total_mean_queries_ge1": {
                "healthy": mass_total(mh), "poison07": mass_total(mp)}}
        g = meas["geometries"][f"g{j:+d}"]
        log(f"  g{j:+d}: band-queries' mass on key0 "
            f"{g['band_key0_mean_over_layers']['healthy']:.4f} -> "
            f"{g['band_key0_mean_over_layers']['poison07']:.4f}; on the band "
            f"{g['band_bandkeys_mean_over_layers']['healthy']:.4f} -> "
            f"{g['band_bandkeys_mean_over_layers']['poison07']:.4f}")
    ev.load_state_dict(sd_cons)

    # =====================================================================
    # CELL (i) — VALUE-RESTORE-UNDER-POISON
    # =====================================================================
    log("--- CELL (i): value-restore-under-poison (dual-stream) ---")
    # content-channel power audit (report-only): does the poison change WHAT
    # IS READ at key 0? (cell (i)'s discriminating power rides on this)
    evp.load_state_dict(sd_p07)
    v0_del = v0_delta_profile(evp, bat_ids[0], wpe0_cons)
    v0_rel_mean = float(np.mean([v["v0_rel_delta"] for v in v0_del.values()]))
    log(f"  v0 content-delta audit (g0 battery): mean rel delta "
        f"{v0_rel_mean:.4f} over layers ("
        + ", ".join(f"L{l} {v['v0_rel_delta']:.3f}"
                    for l, v in v0_del.items()) + ")")
    content_audit = {
        "profile": v0_del, "mean_rel_delta": v0_rel_mean,
        "note": ("relative delta ||v0_clean - v0_pois|| / ||v0_clean|| of "
                 "the V-projected vectors at key position 0, per layer "
                 "(dual-stream, no transplant). If ~0, the value content "
                 "read off key 0 is barely perturbed by the poison and "
                 "cell (i) is weakly powered by construction — the fork "
                 "then leans on cell (ii). REPORT-ONLY.")}
    evp.load_state_dict(sd_p07)
    vt_out = state_eval("VT clean values under poison",
                        fwd=lambda i, t=None: forward_value_transplant(
                            evp, i, wpe0_cons, targets=t),
                        net=evp)
    cell_i: dict = {"cells": {}, "v0_content_audit": content_audit}
    for j in GEOS:
        cell_i["cells"][f"g{j}"] = cell(
            "I", f"VALUE-RESTORE@g{j:+d}", j, base_out[f"g{j}"],
            vt_out[f"g{j}"], base_out["ce_r"], vt_out["ce_r"],
            extra={"instrument": "dual-stream: poisoned q/k (norm-0.07 key "
                                 "geometry) + clean-copy v at key 0, every "
                                 "layer/head/reader",
                   "note": "THE content cell: degraded geometry kept, the "
                           "CONTENT of every read off key 0 restored"})

    # =====================================================================
    # CELL (ii) — MASS-INFLATE-ON-HEALTHY (bias calibrated to the poison's
    # key-0 mass; then fact + CE)
    # =====================================================================
    log("--- CELL (ii): mass-inflate-on-healthy (bias calibration) ---")
    ev.load_state_dict(sd_cons)
    grid = list(BIAS_GRID_SMOKE if SMOKE else BIAS_GRID)
    sweep: list[dict] = []

    def tot_at(b):
        prof = mass_profile(ev, bat_ids[0], bias=b)
        t = mass_total(prof)
        sweep.append({"bias": b, "total_key0_mass": t})
        return t

    for b in grid:
        log(f"  bias {b:+.2f} -> total key-0 mass {tot_at(b):.4f} "
            f"(target {MASS_TARGET:.4f})")
    # bracket + extend if needed
    while (sweep[-1]["total_key0_mass"] < MASS_TARGET
           and grid[-1] < 10.0):
        grid.append(grid[-1] + 1.5)
        log(f"  bias {grid[-1]:+.2f} -> total key-0 mass "
            f"{tot_at(grid[-1]):.4f} (target {MASS_TARGET:.4f}) [extend]")
    lo_b, hi_b = None, None
    for a, b in zip(grid[:-1], grid[1:]):
        ta = sweep[[s["bias"] for s in sweep].index(a)]["total_key0_mass"]
        tb = sweep[[s["bias"] for s in sweep].index(b)]["total_key0_mass"]
        if ta < MASS_TARGET <= tb:
            lo_b, hi_b = a, b
            break
    if lo_b is None:
        lo_b, hi_b = grid[0], grid[-1]
        trims.append(f"calibration did not bracket the target on the grid; "
                     f"using full range [{lo_b}, {hi_b}] for bisection")
    best = {"bias": None, "total": None, "absdev": float("inf")}
    blo, bhi = lo_b, hi_b
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (blo + bhi)
        t = tot_at(mid)
        if abs(t - MASS_TARGET) < best["absdev"]:
            best = {"bias": mid, "total": t,
                    "absdev": abs(t - MASS_TARGET)}
        if t < MASS_TARGET:
            blo = mid
        else:
            bhi = mid
    b_star = best["bias"]
    log(f"calibration: b* = {b_star:.4f} -> total {best['total']:.4f} "
        f"(target {MASS_TARGET:.4f}, |dev| {best['absdev']:.4f}; healthy "
        f"{MASS_HEALTHY:.4f})")

    ev.load_state_dict(sd_cons)
    prof_bias = mass_profile(ev, bat_ids[0], bias=b_star)
    prof_bias_g12 = mass_profile(ev, bat_ids[-12], bias=b_star)
    ev.load_state_dict(sd_p07)
    prof_p_g12 = mass_profile(ev, bat_ids[-12])
    ev.load_state_dict(sd_cons)
    calibration = {
        "target_total_key0_mass": MASS_TARGET,
        "target_source": "runs/e159/metrics.json probe_a.sink_mass_premask "
                         "poison07 (sum over layers of mean_queries_ge1)",
        "healthy_total_key0_mass": MASS_HEALTHY,
        "ratio": MASS_TARGET / MASS_HEALTHY,
        "grid": grid, "sweep": sweep,
        "bracket": [lo_b, hi_b], "bisection_iterations": BISECT_ITERS,
        "bias_star": b_star, "achieved_total": best["total"],
        "abs_dev": best["absdev"],
        "per_layer_at_bstar_g0": {k: v["mean_queries_ge1"]
                                  for k, v in prof_bias.items()},
        "per_layer_poison_g0_stored": mass_poison_stored,
        "per_layer_at_bstar_gm12": {k: v["mean_queries_ge1"]
                                    for k, v in prof_bias_g12.items()},
        "per_layer_poison_gm12_measured": {k: v["mean_queries_ge1"]
                                           for k, v in prof_p_g12.items()},
        "total_gm12": {"bias_star": mass_total(prof_bias_g12),
                       "poison07": mass_total(prof_p_g12)},
        "note": ("uniform additive bias on the key-0 logit (queries >= 1, "
                 "all layers/heads), calibrated on install60@g0 to the "
                 "poisoned net's TOTAL key-0 mass; the per-layer profile is "
                 "NOT matched (the poison's profile is front-loaded) — the "
                 "cell reproduces the allocation DOSE, not its distribution."),
    }

    def bias_fwd(b):
        return lambda i, t=None: forward_bias(ev, i, targets=t, bias=b)

    ev.load_state_dict(sd_cons)
    mi_out = state_eval(f"MASS-INFLATE bias={b_star:.3f}", fwd=bias_fwd(b_star))
    cell_ii: dict = {"cells": {}, "calibration": calibration}
    for j in GEOS:
        cell_ii["cells"][f"g{j}"] = cell(
            "II", f"MASS-INFLATE@g{j:+d}", j, base_out[f"g{j}"],
            mi_out[f"g{j}"], base_out["ce_r"], mi_out["ce_r"],
            extra={"bias": b_star,
                   "achieved_total_key0_mass": best["total"],
                   "target_total_key0_mass": MASS_TARGET,
                   "note": "THE allocation cell: healthy row 0 (content "
                           "untouched), key-0 attention mass inflated to "
                           "the poison's dose by a uniform logit bias"})

    # riders: the bracketing coarse-grid biases (dose dimension)
    for b in sorted({lo_b, hi_b}):
        if abs(b - b_star) < 1e-9:
            continue
        ev.load_state_dict(sd_cons)
        o = state_eval(f"mass-inflate bias={b:+.2f} [rider]",
                       fwd=bias_fwd(b))
        for j in GEOS:
            cell("II", f"mass-inflate b={b:+.2f}@g{j:+d} [rider]", j,
                 base_out[f"g{j}"], o[f"g{j}"], base_out["ce_r"], o["ce_r"],
                 rider=True,
                 extra={"bias": b, "role": "REPORT-ONLY rider: the dose "
                          "dimension of the mass-inflate cell; never gates "
                          "a bar"})

    # =====================================================================
    # adjudication (registered decision table — no bar shopping)
    # =====================================================================
    def branch(cells_by_geo, ce_cost=None, ce_bar=False):
        r12 = cells_by_geo["g-12"]["retention"]
        r0 = cells_by_geo["g0"]["retention"]
        b12 = ("HEALS" if r12 >= HEAL_RET
               else "KILLS" if r12 <= KILL_RET else "GAP")
        b0 = ("HEALS" if r0 >= HEAL_RET
              else "KILLS" if r0 <= KILL_RET else "GAP")
        if b12 == "KILLS" and ce_bar and ce_cost > CELL2_KILL_CE:
            return {"branch": "KILL-AT-HEAVY-CE",
                    "detail": f"kills at the primary geometry (x{r12:.3f}@"
                              f"g-12) but CE cost {ce_cost:+.3f} > "
                              f"+{CELL2_KILL_CE} (moderate bar) — not a "
                              f"bar-eligible kill; g0 x{r0:.3f}"}
        if b12 == "HEALS" and b0 == "HEALS":
            return {"branch": "HEALS",
                    "detail": f"x{r12:.3f}@g-12, x{r0:.3f}@g0 (heal bar "
                              f">= {HEAL_RET} at both)"}
        if b12 == "HEALS":
            return {"branch": "GAP",
                    "detail": f"heals at the primary geometry (x{r12:.3f}@"
                              f"g-12) but NOT at g0 (x{r0:.3f}) — the heal "
                              f"bar requires both; split, texture"}
        if b12 == "KILLS":
            sfx = "-SPLIT" if b0 == "HEALS" else ""
            return {"branch": "KILLS" + sfx,
                    "detail": f"x{r12:.3f}@g-12 (<= {KILL_RET}, the primary "
                              f"geometry), x{r0:.3f}@g0"
                              + ("; g0 branch disagrees (strong rescue at "
                                 "g0)" if sfx else "")}
        return {"branch": "GAP",
                "detail": f"x{r12:.3f}@g-12, x{r0:.3f}@g0 — between the bars"}

    bi = branch(cell_i["cells"])
    bii = branch(cell_ii["cells"],
                 ce_cost=cell_ii["cells"]["g-12"]["ce_cost"], ce_bar=True)
    cell_i["verdict"] = bi
    cell_ii["verdict"] = bii

    table = {
        ("HEALS", "HEALS"): ("READ-COUPLED-LITERAL",
            "the content of reads carries the kill: restoring healthy VALUE "
            "content at the poisoned key 0 heals the fact, and inflating "
            "key-0 mass on a healthy net spares it — T093's claim stands "
            "in its literal form"),
        ("KILLS", "KILLS"): ("MASS-COUPLED",
            "allocation carries the kill twice over: healthy values fail to "
            "rescue under the poison, and mass inflation alone (healthy "
            "content) kills — the intro's 'dies of what attention reads' "
            "inverts to 'dies of what the degraded pivot steals attention "
            "from'; T092's layer 4 rewrites"),
        ("KILLS", "HEALS"): ("MASS-COUPLED",
            "cell (i) kills: healthy content at the absorber fails to "
            "rescue — content is not the carrier; allocation is"),
        ("KILLS", "GAP"): ("MASS-COUPLED",
            "cell (i) kills: healthy content at the absorber fails to "
            "rescue — content is not the carrier (cell (ii) texture: "
            "quoted)"),
        ("KILLS", "KILL-AT-HEAVY-CE"): ("MASS-COUPLED",
            "cell (i) kills: healthy content at the absorber fails to "
            "rescue (cell (ii) kills only beyond the moderate-CE bar — "
            "quoted)"),
        ("HEALS", "KILLS"): ("MIXED",
            "one heals and the other kills: healthy values rescue the fact "
            "under the poison AND mass inflation alone kills — content and "
            "allocation each suffice to move the outcome in opposite "
            "directions; texture with numbers"),
    }
    norm = lambda b: b.replace("-SPLIT", "")
    key = (norm(bi["branch"]), norm(bii["branch"]))
    if key in table:
        outcome, verdict_txt = table[key]
    elif bi["branch"] == "HEALS" and bii["branch"] in ("GAP",
                                                       "KILL-AT-HEAVY-CE"):
        outcome = "TEXTURE"
        verdict_txt = (f"cell (i) heals ({bi['detail']}) but cell (ii) "
                       f"lands {bii['branch']} ({bii['detail']}) — the "
                       f"sufficiency test did not resolve; numbers quoted")
    else:
        outcome = "TEXTURE"
        verdict_txt = (f"cell (i) {bi['branch']} ({bi['detail']}); cell (ii) "
                       f"{bii['branch']} ({bii['detail']}) — no registered "
                       f"branch fired cleanly; texture with numbers")

    verdict = {
        "cell_i_branch": bi, "cell_ii_branch": bii,
        "bars": {"heals": f"retention >= {HEAL_RET} at BOTH g-12 and g0",
                 "kills": f"retention <= {KILL_RET} at the PRIMARY geometry "
                          f"g-12 (dispatch '>= 60% drop'; poison-alone's "
                          f"own g-12 retention is 0.158); g0 = consistency "
                          f"column, disagreement => -SPLIT suffix (reported, "
                          f"never re-mapped)",
                 "cell_ii_moderate_ce": f"CE cost <= +{CELL2_KILL_CE} for a "
                                        f"bar-eligible cell-(ii) kill",
                 "bars_v2_note": "v1 (smoke shakedown) had an invented "
                                 "g0<=0.50 kill conjunct absent from the "
                                 "dispatch; corrected to the dispatch-literal "
                                 "form before the full compute (see "
                                 "recipe_deviations)"},
        "decision_table": REGISTERED_PREDICTION["decision_table_pre_compute"],
        "outcome": outcome, "verdict": verdict_txt,
    }
    log("=" * 78)
    log(f"E162 VERDICT: {outcome} — {verdict_txt}")
    log(f"  cell (i) {bi['branch']} ({bi['detail']}) | cell (ii) "
        f"{bii['branch']} ({bii['detail']})")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e162_read_vs_mass",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("R47 critic's final attack on T093's READ-coupled "
                         "claim — the content-vs-allocation fork. Docstring "
                         "+ bars + decision table written before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does the consolidated fact die of what attention "
                     "READS off the poisoned row 0 (value content; "
                     "READ-coupled literal) or of what the degraded pivot "
                     "STEALS (10.8x attention-mass absorption starving "
                     "informative positions; MASS-coupled)?"),
        "nets": {
            "consolidated": f"runs/checkpoints/{CONS_CK.name} (loaded, "
                            "gated vs e131/e150/e151/e159 + e151 stored "
                            "cells + e159 stored mass profile)",
            "site_stored_column": "OMITTED (dispatch 'if cheap': arm_b's "
                                  "fact survives the poison outright — "
                                  "e159 SITE-SPARED x0.922 — so the cells "
                                  "have no kill to explain there; CPU "
                                  "thrift while e158/e125a run)",
            "eval_only": True},
        "gates": {"G_SPLICE": G_SPLICE, "consolidated": gates["consolidated"],
                  "mask_instrument": gates["mask_instrument"],
                  "vt_identity": gates["vt_identity"],
                  "vt_self": gates["vt_self"],
                  "e151_mask_cells": gates["e151_mask_cells"],
                  "e151_poison07_cells": gates["e151_poison07_cells"],
                  "mass_instrument": gates["mass_instrument"]},
        "cell_i_value_restore": cell_i,
        "cell_ii_mass_inflate": cell_ii,
        "measurement_iii_starvation": meas,
        "cells": CELLS,
        "verdict": verdict,
        "honesty_reflex": {
            "value_transplant_pathologies": "the transplant restores the "
                "FULL healthy value channel at key 0 (every layer, head, and "
                "reader — including row 0's self-read), not a minimal "
                "content patch; the donor stream is the clean copy's own "
                "forward, so later-layer donations are healthy-context "
                "donations and the two streams diverge by design. The cell "
                "therefore answers 'is healthy CONTENT at the absorber "
                "sufficient?', not 'which read carries the kill?'. Donor-"
                "identity and donor-self gates prove the machinery is inert "
                "(bit-tight); the v0 content-delta audit (cell_i."
                "v0_content_audit) quantifies the channel's actual "
                "perturbation — if the poison barely moves v0, cell (i) is "
                "weakly powered by construction and says so; a partial heal "
                "is read as GAP texture, never shopped.",
            "bias_calibration_limits": "the uniform bias matches the TOTAL "
                "key-0 mass on the g0 battery (e159's instrument), not the "
                "per-layer profile (the poison's absorption is front-loaded "
                "L1-L3; the biased net's is flatter) and not the g-12 "
                "battery's dose (reported alongside: total_gm12). The cell "
                "is a sufficiency test for the DOSE of absorption, not an "
                "emulation of the poison; the bracketing riders expose the "
                "dose gradient.",
            "moderate_ce_bar": "cell (ii)'s kill bar requires CE <= +0.85 — "
                "'no more wreckful than the emulated state' (poison-alone "
                "+0.843); a kill priced beyond that reads as generic wreck, "
                "not mass starvation (KILL-AT-HEAVY-CE, non-eligible).",
            "single_net": "one consolidated net (e131 lineage, single "
                "recipe); the fork's verdicts are lineage-specific until "
                "replication seeds land; the R150 rider history (e150) "
                "agreed with the consolidated ladder at these brackets.",
            "off_distribution": "both cells are off the training "
                "distribution (as are the mask and poison anchors); every "
                "reading is anchored to the rebuilt + gated "
                "single-intervention cells, not to absolute health claims.",
            "ce_bank_is_position_blind": "CE_R windows are 256-token corpus "
                "windows that never contain the fact; the CE column prices "
                "general LM damage, not fact-geometry damage (e150's "
                "caveat, inherited).",
            "measurement_iii_is_report_only": "the band-mass decomposition "
                "runs on NEW fact-in-window batteries (the name inside the "
                "window); it motivates and illustrates but gates nothing.",
        },
        "trims": trims,
        "recipe_deviations": recipe_deviations,
        "ckpt_inventory": {
            "saved": {},
            "external_used": [f"runs/checkpoints/{CONS_CK.name}"],
            "metrics_read": [str(E151_METRICS.relative_to(E43.REPO)),
                             str(E159_METRICS.relative_to(E43.REPO))],
            "note": "eval-only: no checkpoints written or regenerated"},
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072,
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "read_vs_mass.png", CELLS, base_out, m_out, p07_out, cell_i,
         cell_ii, calibration, meas, verdict, b_star, MASS_TARGET,
         MASS_HEALTHY)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'read_vs_mass.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def plot(path, cells_all, base_out, m_out, p07_out, cell_i, cell_ii,
         calibration, meas, verdict, b_star, mass_target, mass_healthy):
    """The fork panel: the four states x two geometries, the fact-vs-CE
    plane, the calibration curve, and the starvation decomposition."""
    fig = plt.figure(figsize=(16.0, 11.0))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.1, 1.0))

    # ---- panel 1 (top, spans): the fork quartet per geometry
    ax = fig.add_subplot(gs[0, :])
    series = [
        ("base (healthy)", {j: 1.0 for j in GEOS}, None, None, "steelblue"),
        ("mask alone [anchor]", {j: m_out[f"g{j}"] / base_out[f"g{j}"]
                                 for j in GEOS},
         m_out["ce_r"] - base_out["ce_r"], None, "seagreen"),
        ("poison 0.07 alone [anchor]",
         {j: p07_out[f"g{j}"] / base_out[f"g{j}"] for j in GEOS},
         p07_out["ce_r"] - base_out["ce_r"], None, "firebrick"),
        ("(i) VALUE-RESTORE under poison",
         {j: cell_i["cells"][f"g{j}"]["retention"] for j in GEOS},
         cell_i["cells"]["g0"]["ce_cost"], "CELL (i)", "darkviolet"),
        ("(ii) MASS-INFLATE on healthy",
         {j: cell_ii["cells"][f"g{j}"]["retention"] for j in GEOS},
         cell_ii["cells"]["g0"]["ce_cost"], "CELL (ii)", "darkorange"),
    ]
    riders = [c for c in cells_all if c["probe"] == "II" and c["rider"]]
    all_series = series + [
        (f"mass-inflate b={c['bias']:+.2f} [rider]",
         {j: (c["retention"] if c["geo"] == j else None) for j in GEOS},
         c["ce_cost"], None, "moccasin") for c in riders]
    ng, ns = len(GEOS), len(all_series)
    w = 0.8 / ns
    for si, (lab, rets, ce, tagtxt, col) in enumerate(all_series):
        for gi, j in enumerate(GEOS):
            if rets.get(j) is None:
                continue
            x = gi - 0.4 + w * (si + 0.5)
            v = 100 * rets[j]
            ax.bar(x, v, w * 0.9, color=col, edgecolor="k", lw=0.6,
                   label=(lab if gi == 0 else None))
            txt = f"x{rets[j]:.2f}"
            if ce is not None and gi == 0:
                txt += f"\nCE{ce:+.2f}"
            ax.text(x, v + 2.0, txt, ha="center", fontsize=6.4,
                    color=("dimgray" if col == "moccasin" else "k"))
    ax.axhline(100 * HEAL_RET, ls="--", color="seagreen", lw=1.2)
    ax.axhline(100 * KILL_RET, ls="--", color="firebrick", lw=1.2)
    ax.text(ng - 0.42, 100 * HEAL_RET + 1.5, "heal bar 0.70 (both geos)",
            fontsize=7, color="seagreen")
    ax.text(ng - 0.42, 100 * KILL_RET - 6.5,
            "kill bar 0.40 @g-12 (primary)", fontsize=7, color="firebrick")
    ax.set_xticks(range(ng))
    ax.set_xticklabels([f"g{j:+d}  (base expr {base_out[f'g{j}']:.3f})"
                        for j in GEOS], fontsize=9)
    ax.set_ylabel("fact retention vs healthy same-net base (%)")
    ax.set_ylim(0, 118)
    ax.set_title("E162 — THE READ-vs-MASS FORK: (i) healthy VALUES under the "
                 "poison vs (ii) poisoned MASS on the healthy net",
                 fontsize=10)
    ax.legend(fontsize=7.6, loc="upper right", ncol=2)
    ax.grid(alpha=0.25, axis="y")

    # ---- panel 2 (bottom-left): fact-vs-CE plane
    ax = fig.add_subplot(gs[1, 0])
    ax.axhspan(100 * HEAL_RET, 112, color="seagreen", alpha=0.14)
    ax.axhspan(-4, 100 * KILL_RET, color="firebrick", alpha=0.10)
    ax.axhline(100 * HEAL_RET, ls="--", color="seagreen", lw=1.1)
    ax.axhline(100 * KILL_RET, ls="--", color="firebrick", lw=1.1)
    ax.text(0.02, 104, "HEAL/SPARE region (ret>=0.70)", fontsize=7.5,
            color="seagreen")
    ax.text(0.02, 3, "KILL region (ret<=0.40)", fontsize=7.5,
            color="firebrick")
    pts = []
    for j in GEOS:
        pts.append((f"mask@g{j:+d}", m_out[f"g{j}"] / base_out[f"g{j}"],
                    m_out["ce_r"] - base_out["ce_r"], "o", "seagreen"))
        pts.append((f"poison.07@g{j:+d}",
                    p07_out[f"g{j}"] / base_out[f"g{j}"],
                    p07_out["ce_r"] - base_out["ce_r"], "o", "firebrick"))
        pts.append((f"VALUE-RESTORE@g{j:+d}",
                    cell_i["cells"][f"g{j}"]["retention"],
                    cell_i["cells"][f"g{j}"]["ce_cost"], "*",
                    "darkviolet"))
        pts.append((f"MASS-INFLATE@g{j:+d}",
                    cell_ii["cells"][f"g{j}"]["retention"],
                    cell_ii["cells"][f"g{j}"]["ce_cost"], "*",
                    "darkorange"))
    for c in riders:
        pts.append((f"b={c['bias']:+.2f}@g{c['geo']:+d}", c["retention"],
                    c["ce_cost"], "s", "moccasin"))
    for lab, ret, ce, mk, col in pts:
        ax.scatter(ce, 100 * ret, s=(240 if mk == "*" else 50), marker=mk,
                   color=col, edgecolor="k", lw=0.6, zorder=3)
        ax.annotate(lab, (ce, 100 * ret), textcoords="offset points",
                    xytext=(5, 4), fontsize=6.2)
    ax.set_xlabel("CE cost vs same-net baseline (nats, e065 bank seed 26502)")
    ax.set_ylabel("fact retention (%)")
    ax.set_ylim(-4, 112)
    ax.set_title("the fork cells in the fact-vs-CE plane "
                 "(violet star = content cell; orange star = allocation cell)",
                 fontsize=9.5)
    ax.grid(alpha=0.25)

    # ---- panel 3 (bottom-middle... merged into bottom-right row): calibration
    ax = fig.add_subplot(gs[1, 1])
    sw = calibration["sweep"]
    ax.plot([s["bias"] for s in sw], [s["total_key0_mass"] for s in sw],
            "o-", color="tab:blue", lw=1.3, ms=4,
            label="healthy net + bias (sweep+bisect)")
    ax.axhline(mass_target, ls="--", color="firebrick", lw=1.3,
               label=f"poison's total mass {mass_target:.3f} (target)")
    ax.axhline(mass_healthy, ls=":", color="seagreen", lw=1.2,
               label=f"healthy total mass {mass_healthy:.3f}")
    ax.scatter([b_star], [calibration["achieved_total"]], marker="*",
               s=240, color="darkorange", edgecolor="k", lw=0.7, zorder=4,
               label=f"b*={b_star:.3f} (the cell)")
    ax.annotate(f"b*={b_star:.3f}\ntotal {calibration['achieved_total']:.3f}",
                (b_star, calibration["achieved_total"]),
                textcoords="offset points", xytext=(8, -14), fontsize=7.5)
    # per-layer profile inset as twin bars
    ax2 = ax.inset_axes([0.58, 0.08, 0.40, 0.42])
    layers = sorted(calibration["per_layer_at_bstar_g0"])
    x = np.arange(len(layers))
    ax2.bar(x - 0.2, [calibration["per_layer_poison_g0_stored"][l]
                      for l in layers], 0.38, color="firebrick",
            label="poison (e159)")
    ax2.bar(x + 0.2, [calibration["per_layer_at_bstar_g0"][l]
                      for l in layers], 0.38, color="darkorange",
            label=f"bias b* (g0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([l[1:] for l in layers], fontsize=6)
    ax2.tick_params(labelsize=6)
    ax2.set_title("per-layer key-0 mass (dose matched, profile not)",
                  fontsize=6.5)
    ax2.legend(fontsize=5.5)
    ax.set_xlabel("additive key-0 logit bias (queries >= 1, all layers/heads)")
    ax.set_ylabel("total key-0 attention mass (sum over layers)")
    ax.set_title("CELL (ii) calibration — reproduce the poison's 10.8x "
                 "absorption on a healthy net", fontsize=9.5)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)

    # ---- starvation measurement as suptitle-adjacent text block
    g12 = meas["geometries"]["g-12"]
    g0 = meas["geometries"]["g+0"]
    txt = ("starvation read-out (band queries' mass): "
           f"g-12 key0 {g12['band_key0_mean_over_layers']['healthy']:.3f}->"
           f"{g12['band_key0_mean_over_layers']['poison07']:.3f}, "
           f"band {g12['band_bandkeys_mean_over_layers']['healthy']:.3f}->"
           f"{g12['band_bandkeys_mean_over_layers']['poison07']:.3f} | "
           f"g0 key0 {g0['band_key0_mean_over_layers']['healthy']:.3f}->"
           f"{g0['band_key0_mean_over_layers']['poison07']:.3f}, "
           f"band {g0['band_bandkeys_mean_over_layers']['healthy']:.3f}->"
           f"{g0['band_bandkeys_mean_over_layers']['poison07']:.3f}")
    fig.suptitle(f"E162 — {verdict['outcome']}: {verdict['verdict'][:180]}\n"
                 f"{txt}", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
