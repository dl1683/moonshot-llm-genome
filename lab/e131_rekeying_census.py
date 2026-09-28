"""E131 — the RE-KEYING CENSUS (R43 critic's discriminator; REGISTERED).

WHY (Review 43, critic point 3, verbatim context): "Most damaging assumption:
D-all survival != fact left the wpe system. Row 0 (content-carrying in 6/6
installs per T069) never content-tested post-consolidation; 12 band rows left
intact in e113; no out-of-band row ever scanned. BODY-STORED vs
ADDRESS-MIGRATED-ELSEWHERE is OPEN." e113's D-all deleted only 5 of 17 band
rows on the consolidated net (survived 0.90 g0); e120's splice arms trained
the fact at address row 183, which no instrument ever read. This experiment
adjudicates BODY-STORED-GENUINE vs RE-KEYED.

NETS (rebuilt from deterministic recipes, CPU):
  * LINE-SPLICE (e120): runs/checkpoints/e082_b43_install.pt (B43 install,
    seed 24331; gate G_B43: install-60 g0 battery p(Z) = 0.3198219 +- 5e-6)
    -> arms (a) self-ctx-spliced and (b) corpus-ctx-spliced regenerated
    VERBATIM (e120 recipe: dream harvest seeds 12110-12113, corpus filler
    seed 12103, fact segments from install windows k%60, splice at cont-col
    42 -> ZEPHYRA x-cols 184..190 / address row 183; fine-tune 300 steps
    AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0, batch 16 exposure +
    16 anchor (8 paired + 8 random), full-continuation mask, seed 12101).
    GATE G_E120 (per arm): the post-none install-60 battery table must match
    runs/e120/metrics.json within 1e-6/cell (both runs CPU, 8 threads —
    bit-reproduction expected; if only the 0.05/cell e113 convention holds,
    record a recipe_deviation and continue).
  * LINE-CONSOLIDATED (e113): runs/checkpoints/e048_repro.pt (install-phase
    net, seed 42; gate G_E048: g0 battery p(Z) = 0.5563087 +- 5e-6) -> the
    e113 consolidated net regenerated VERBATIM (jittered replay {-8,-4,0,+4,
    +8}, 300 steps, seed 10901, same optimizer/batch/mask recipe). GATE
    G_E113: post-none install-60 table within 1e-6/cell of runs/e113/
    metrics.json (0.05/cell fallback -> deviation).

PROBES (priority order; drop from the bottom if CPU wall-clock forces it):
  (1) 183-GEOMETRY READ (T075): on the regenerated splice arms, BEFORE any
      deletion, read fact expression at the splice geometry: mean p of each
      true name char at positions 183..189 (wpe rows 183..189 predict window
      cols 184..190 = ZEPHYRA) over the arm's 120 exposure windows; headline
      = mean p(Z) at position 183 (the battery convention's char) + frac
      p>=0.5. References: the same read on the base net (pre-ft floor), the
      arm's install-60 g0 level, the arm's battery-band floor (min over the
      five g cells). REPORTED SEPARATELY (does not gate the verdict).
  (2) ROW-0 CONTENT TEST (W005/W008): e116 mean/zero-arm census on wpe row 0
      (mean arm: wpe[0] <- mean of all 512 rows; zero arm: wpe[0] <- 0; drop
      = base battery p(Z) - arm p(Z); install-60 g0 130-token battery), on
      the install-phase net AND the consolidated net. Controls: fed rows
      {1,2,3,4,5,6,60,100,118,119,120} (scaffold + ordinary rows, outside
      {0} and outside 121..137); band rows 121..137 reported for context.
      Install-phase gate: row-0 drops must reproduce e116 seed-42 stored
      values (tol 5e-6). Strength(r) = min(mean-drop, zero-drop).
  (3) FULL-512 wpe DELTA CENSUS: per-row delta = wpe_cons[r] - wpe_inst[r];
      content score = |delta[r] . normalize(wpe_inst[129])| (the install
      net's row-129 direction = the fact's original address/content axis,
      e067/e109). Scan ALL 512 rows; grown band = 121..137; out-of-band =
      everything outside {0} u 121..137. Riders: |delta| norms, projection
      on the row-0 axis, and the same census for the e120 splice arms vs
      their B43 base (row 183 context; report-only).
  (4) BAND-MINUS-ROW-0 DELETION (scaffold-matched): D2 subtractive row-zero
      on the consolidated net, install-60 battery x 5 geometries (+held-30
      report + CE_R): none | d_all_e113 {121,125,129,133,137} (repro) |
      d_all_r0 {0,121,125,129,133,137} | d_all_r1 {1,121,125,129,133,137}
      (row 1 = e116-established scaffolding row, content-negative — the
      scaffold-matched control) | d_r0 {0} | d_r1 {1}.

REGISTERED PREDICTION (coordinator, VERBATIM — no bar shopping):
  * RE-KEYED fires if ANY of: row-0 content test positive
    post-consolidation (>= its install-phase strength or above control-row
    band); band-minus-row-0 collapses (fact expression drops >=50% vs
    e113's D-all level); census finds an out-of-band row carrying
    content-carrying delta >= the grown-row band's median.
  * BODY-STORED-GENUINE fires if: row-0 test null (at/below control band),
    band-minus-row-0 survives like D-all did (within 20% of e113's D-all
    level), and the census shows no out-of-band home.
  * For probe (1) report separately: SIGNAL-AT-183 (p >= 0.5 x the arm's
    install-60 level) vs NO-SIGNAL-AT-183 (near battery floor).
  * No bar shopping. Ambiguous => say AMBIGUOUS with numbers.
OPERATIONALIZATIONS (fixed before compute):
  * row-0 "content test positive" = e116 criterion (mean-drop > 0,
    zero-drop > 0, min/max arm ratio >= 0.5) AND (strength >= install-phase
    strength OR strength > max control-row strength).
  * "band-minus-row-0 collapses" = mean over the 5 install-60 geometries of
    d_all_r0 <= 0.5 x mean of e113's STORED d_all cells (0.9064);
    "survives" = same mean >= 0.8 x 0.9064 = 0.7251.
  * census "content-carrying delta" = |delta[r] . normalize(wpe_inst[129])|
    (absolute projection length); fires if ANY out-of-band row's score >=
    median score over rows 121..137.
  * probe-1 "near battery floor" = p <= 2 x the arm's battery-band min;
    anything between the SIGNAL and NO-SIGNAL bars = AMBIGUOUS (numbers).

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import;
torch.set_num_threads(8), e113/e120's exact envelope — another agent owns
the GPU; fine-tunes run sequentially, no busy-wait polling). Per-fine-tune
wall cap 1500 s. Checkpoints saved to runs/checkpoints/e131_<phase>.pt
(existing convention; *.pt is gitignored — on-disk persistence is the
point; coordinator erratum honored: no runs/e131/ckpts/).

Outputs: runs/e131/{metrics.json, rekeying_census.png},
         runs/checkpoints/e131_<phase>.pt.
No NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e131_rekeying_census.py     (E131_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (e119 owns the GPU)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(8)                              # 8 threads max (lab convention)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E131_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
B43_CK = CKPT_DIR / "e082_b43_install.pt"      # e120 line base
E048_CK = CKPT_DIR / "e048_repro.pt"           # e113 line install-phase

# splice geometry (e120 verbatim)
FACT_PRE, FACT_POST = 12, 12
FACT_LEN = FACT_PRE + len(NAME) + FACT_POST      # 31
SPLICE_AT = 42
Z_XCOL = PRE + SPLICE_AT + FACT_PRE             # 184: ZEPHYRA onset x-col
SPLICE_ADDR_ROW = Z_XCOL - 1                    # 183: wpe row predicting 'Z'
GEN_T, GEN_TOPK = 0.8, 40
DREAM_SEEDS = (12110, 12111, 12112, 12113)
N_DREAMS_PER_PROMPT = 1 if SMOKE else 4
N_PROMPTS = 8 if SMOKE else 30
DREAM_LEN = BLOCK - PRE                          # 126
CORP_CONT_SEED = 12103

# fine-tune envelope (e109 arm-a / e113 / e120 verbatim)
FT_LR = 1e-3
FT_STEPS = 8 if SMOKE else 300
FT_TIME_CAP = 1500.0
EVAL_EVERY_SPLICE = 2 if SMOKE else 50           # e120 convention
EVAL_EVERY_JIT = 2 if SMOKE else 25              # e113 convention
NAME_BS, ANCH_BS = 16, 16
CONS_SEED_SPLICE = 12101                         # e120 arms a/b
CONS_SEED_JIT = 10901                            # e113 / e109 arm (a)

JITTERS = (-8, -4, 0, 4, 8)
GEO_ORDER = [-8, -4, 0, 4, 8]
ADDR_BAND = tuple(r for r in range(121, 138))
E113_ADDR_ROWS = (121, 125, 129, 133, 137)       # e113's D-all set
CONTROL_ROWS = (1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120)
CENSUS_ROWS = (0,) + CONTROL_ROWS + ADDR_BAND    # probe-2 row set

# gates / references
R_EVAL_SEED = 26502
G_B43_REF = 0.3198219250887632                   # e082 gate0 install-60 p(Z)
G_E048_REF = 0.5563086867332458                  # e113 G_INST battery_pz
G_BIT_TOL = 1e-6                                 # bit-reproduction gate
G_FALLBACK_TOL = 0.05                            # e113 G_REPRO convention
G_E116_TOL = 5e-6                                # e116 reproduction tol
E120_M = E43.REPO / "runs" / "e120" / "metrics.json"
E113_M = E43.REPO / "runs" / "e113" / "metrics.json"
E116_M = E43.REPO / "runs" / "e116" / "metrics.json"
HARVEST_REF_Z = 34                               # e121/e120 own-dream ZEPHYRA count

REGISTERED_PREDICTION = {
    "re_keyed": "RE-KEYED fires if ANY of: row-0 content test positive "
                "post-consolidation (>= its install-phase strength or above "
                "control-row band); band-minus-row-0 collapses (fact "
                "expression drops >=50% vs e113's D-all level); census finds "
                "an out-of-band row carrying content-carrying delta >= the "
                "grown-row band's median.",
    "body_stored_genuine": "BODY-STORED-GENUINE fires if: row-0 test null "
                           "(at/below control band), band-minus-row-0 "
                           "survives like D-all did (within 20% of e113's "
                           "D-all level), and the census shows no out-of-band "
                           "home.",
    "probe1": "For probe (1) report separately: SIGNAL-AT-183 (p >= 0.5 x the "
              "arm's install-60 level) vs NO-SIGNAL-AT-183 (near battery "
              "floor).",
    "no_bar_shopping": "No bar shopping. Ambiguous => say AMBIGUOUS with "
                       "numbers.",
    "operationalizations": "row-0 positive = e116 content criterion AND "
                           "(strength >= install-phase strength OR strength > "
                           "max control strength); strength = min(mean-drop, "
                           "zero-drop); collapse = mean_g(d_all_r0) <= 0.5 x "
                           "mean of e113's stored D-all cells; survives = >= "
                           "0.8 x that; census score = |delta[r] . "
                           "normalize(wpe_inst[129])|, fires if any "
                           "out-of-band row >= band median; probe-1 "
                           "no-signal = p <= 2 x battery-band min.",
}

trims: list[str] = []
recipe_deviations: list[str] = [
    "COORDINATOR ERRATUM honored: runs/checkpoints/ ships the recipe roots — "
    "e082_b43_install.pt (e120 line base) and e048_repro.pt (e113 line "
    "install-phase) were LOADED (not regenerated) and gated to their "
    "recorded battery values (5e-6). Only the fine-tuned nets were "
    "regenerated from recipe: e120 arms (a)/(b) (splice fine-tunes) and the "
    "e113 consolidated net (jittered replay).",
    "e044_b_zephyra.pt / e044_a_reinstall.pt inspected per the erratum and "
    "REJECTED as the consolidated-line root: they are e044 scar-tissue arms "
    "(meta: steps 400, seeds 24400/24401), not the e113 recipe's base — "
    "e113's metrics cite runs/checkpoints/e048_repro.pt + 300-step jittered "
    "replay (seed 10901) verbatim, and that root exists.",
    "Probe-3 'consolidated vs install-phase vs base': the pre-install "
    "corpus base of the e048 line is not persisted, so the census delta "
    "reference is the INSTALL-PHASE net (e048_repro) — the meaningful "
    "reference for 'what consolidation changed'. The e120-arm rider census "
    "uses their B43 base.",
    "Scaffold-matched control implemented as d_all_r1 (row 1 = e116 "
    "scaffolding row, content-test negative on the install net) alongside "
    "d_all_r0 and the row-only arms d_r0/d_r1, so generic-scaffold damage "
    "and fact-content damage are separable.",
    "Census universe is ALL 256 wpe rows of this line (block_size 256), not "
    "512 — the tasking's '512' assumes the e098 fresh-family block size; "
    "no row of the net is left unscanned either way.",
    "Checkpoints saved to runs/checkpoints/e131_<phase>.pt (existing "
    "convention, coordinator instruction); *.pt is gitignored repo-wide — "
    "on-disk persistence only. The install-phase net is NOT duplicated "
    "(it is e048_repro.pt, recorded under external_used).",
]


# ------------------------------------------------------------------ instruments

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
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30,
                 keep_per_ctx=False) -> dict:
    """e068/e113/e120 battery on CPU: p(Z) at the last position."""
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
def battery_pz(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> float:
    """e116's scalar battery (probe-2 arm census)."""
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


def deleted_wpe(sd: dict, rows: tuple[int, ...]) -> tuple[dict, dict]:
    """D2 subtractive row-zero with the e065/e113 confinement gate."""
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
def free_run_batch(net: TinyGPT, prompt_ids: torch.Tensor, n_new: int,
                   seed: int, temperature: float = GEN_T,
                   top_k: int = GEN_TOPK) -> torch.Tensor:
    """e121/e120 batched free-run, verbatim."""
    net.eval()
    torch.manual_seed(seed)
    idx = prompt_ids.clone()
    for _ in range(n_new):
        logits, _ = net(idx[:, -net.cfg.block_size:])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], 1)
    return idx


@torch.no_grad()
def read_fact_position(net: TinyGPT, pool_x: torch.Tensor, name_ids,
                       zid: int, bs=30) -> dict:
    """PROBE-1 instrument: p(true name char) at positions 183..189 over the
    exposure windows (position t predicts window col t+1; ZEPHYRA occupies
    cols 184..190). Headline = p(Z) at position 183 (the onset)."""
    net.eval()
    n_name = len(name_ids)
    per_pos = [[] for _ in range(n_name)]
    onset = []
    for i in range(0, pool_x.shape[0], bs):
        w = pool_x[i:i + bs]
        lg, _ = net(w)
        pr = F.softmax(lg, -1)
        for k in range(w.shape[0]):
            onset.append(float(pr[k, SPLICE_ADDR_ROW, int(zid)]))
            for j in range(n_name):
                per_pos[j].append(
                    float(pr[k, SPLICE_ADDR_ROW + j, int(w[k, Z_XCOL + j])]))
    onset_t = torch.tensor(onset)
    allp_t = torch.tensor([p for pos in per_pos for p in pos])
    return {"pz_onset_mean": float(onset_t.mean()),
            "pz_onset_median": float(onset_t.median()),
            "pz_onset_frac_ge_0.5": float((onset_t >= 0.5).float().mean()),
            "pname_mean_over7": float(allp_t.mean()),
            "pname_frac_ge_0.5": float((allp_t >= 0.5).float().mean()),
            "per_position_mean": [float(np.mean(pos)) for pos in per_pos]}


# ------------------------------------------------------------------ fine-tune

def finetune_arm(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                 pool_mask: torch.Tensor, anchor: torch.Tensor,
                 train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                 seed: int, eval_every: int):
    """e109/e113/e120 fine-tune recipe VERBATIM (composition/seed/loss), CPU.
    Evals are no-grad on a twin and consume no RNG, so eval_every does not
    touch the gradient trajectory."""
    net = copy.deepcopy(net0).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    traj, t_start = [], time.time()
    step = 0
    evl = copy.deepcopy(net0)
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
        if step % eval_every == 0 or step == FT_STEPS or \
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
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed}


# ------------------------------------------------------------------ splice pools

def build_fact_segments(install_occ, train_text, encode) -> torch.Tensor:
    segs = []
    for p, h in install_occ:
        s = (train_text[p - FACT_PRE: p] + NAME
             + train_text[p + len(h): p + len(h) + FACT_POST])
        if len(s) != FACT_LEN:
            raise RuntimeError(f"fact segment len {len(s)} != {FACT_LEN}")
        segs.append(encode(s))
    return torch.stack(segs)


def splice_pool(prompt_ids: torch.Tensor, filler: torch.Tensor,
                fact_segs: torch.Tensor) -> torch.Tensor:
    """e120 verbatim: [prompt 130][filler[:42] + fact[k%60] + filler[73:]]."""
    n = filler.shape[0]
    if filler.shape[1] != DREAM_LEN:
        raise RuntimeError(f"filler len {filler.shape[1]} != {DREAM_LEN}")
    fs = fact_segs[torch.arange(n) % fact_segs.shape[0]]
    cont = torch.cat([filler[:, :SPLICE_AT], fs,
                      filler[:, SPLICE_AT + FACT_LEN:]], 1)
    if cont.shape[1] != DREAM_LEN:
        raise RuntimeError("spliced continuation len mismatch")
    pr = torch.stack([prompt_ids[k % prompt_ids.shape[0]] for k in range(n)])
    return torch.cat([pr, cont], 1)


def save_ckpt(name: str, sd: dict, meta: dict, inventory: dict):
    """Save a phase net to runs/checkpoints/e131_<phase>.pt (convention)."""
    p = CKPT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "meta": meta}, p)
    inventory[name] = {"path": str(p.relative_to(E43.REPO)).replace("\\", "/"),
                       "bytes": p.stat().st_size, "meta": meta}
    log(f"ckpt saved: {p.name} ({p.stat().st_size / 1e6:.1f} MB)")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e131_smoke" if SMOKE else "e131")
    ckpt_inventory: dict = {}
    log(f"E131 RE-KEYING CENSUS (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda visible = {torch.cuda.is_available()}")

    refs = {}
    for key, path in (("e120", E120_M), ("e113", E113_M), ("e116", E116_M)):
        refs[key] = json.loads(Path(path).read_text(encoding="utf-8")) \
            if Path(path).exists() else None

    # ---------------- protocol rebuild (verbatim)
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

    name_ids = corpus.encode(NAME)

    # batteries per geometry (e113/e120 construction verbatim)
    bat_ids = {}
    for j in GEO_ORDER:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    f_eval = bat_ids[(0, "install60")]             # both lines' g0 battery
    ids130 = bat_ids[(0, "install60")]             # e116 130-token battery

    # CE_R eval bank (e065 verbatim, seed 26502)
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # anchor bank (e065/e109/e113/e120 verbatim)
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])

    # =====================================================================
    # PROBE 1 — regenerate e120 arms (a)/(b) verbatim; the 183-geometry read
    # =====================================================================
    log("--- PHASE A: e120 splice arms (a)/(b) regeneration + 183-read ---")
    net_b43 = load_cpu(B43_CK)
    evl = copy.deepcopy(net_b43)
    bz_b43 = battery_cell(evl, f_eval, zid)
    G_B43 = {"battery_pz": bz_b43["mean_pz"], "ref": G_B43_REF,
             "tol": G_BIT_TOL,
             "pass": bool(abs(bz_b43["mean_pz"] - G_B43_REF) < G_BIT_TOL)}
    log(f"G_B43 base battery p(Z) {bz_b43['mean_pz']:.10f} (ref "
        f"{G_B43_REF:.10f}): {'PASS' if G_B43['pass'] else 'FAIL'}")
    if not G_B43["pass"]:
        raise RuntimeError("B43 checkpoint failed its gate")

    prompts = [train_text[p - PRE: p] for p, _ in held_occ][:N_PROMPTS]
    prompt_ids = torch.stack([corpus.encode(c) for c in prompts])

    wins = []
    for s in DREAM_SEEDS[:N_DREAMS_PER_PROMPT]:
        out = free_run_batch(net_b43, prompt_ids, DREAM_LEN, seed=s)
        wins.append(out)
        log(f"  harvest seed {s}: {out.shape[0]} dreams x {DREAM_LEN} tokens")
    dream_ids = torch.cat(wins)
    z_in_dreams = sum("".join(itos[int(i)] for i in w[PRE:]).count(NAME)
                      for w in dream_ids)
    log(f"  harvest ZEPHYRA count {z_in_dreams} "
        f"(e120/e121 ref {HARVEST_REF_Z if not SMOKE else 'n/a'})")
    if not SMOKE and z_in_dreams != HARVEST_REF_Z:
        recipe_deviations.append(
            f"dream harvest ZEPHYRA count {z_in_dreams} != e120/e121 ref "
            f"{HARVEST_REF_Z} — regeneration drift flagged; arms gated "
            f"anyway (G_E120)")
        trims.append(f"harvest count drift: {z_in_dreams}")

    g = torch.Generator().manual_seed(CORP_CONT_SEED)
    src = torch.randint(len(train_ids) - DREAM_LEN - 1,
                        (N_DREAMS_PER_PROMPT, len(prompts)), generator=g)
    cont = torch.stack([train_ids[s: s + DREAM_LEN] for s in src.flatten()])

    fact_segs = build_fact_segments(install_occ, train_text, corpus.encode)
    dream_fill = dream_ids[:, PRE:]
    pool_a_x = splice_pool(prompt_ids, dream_fill, fact_segs)
    pool_b_x = splice_pool(prompt_ids, cont, fact_segs)
    G_GEO = {"z_xcols": [Z_XCOL, Z_XCOL + len(NAME) - 1],
             "a_all": bool(all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)],
                                           name_ids) for w in pool_a_x)),
             "b_all": bool(all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)],
                                           name_ids) for w in pool_b_x))}
    G_GEO["pass"] = bool(G_GEO["a_all"] and G_GEO["b_all"])
    if not G_GEO["pass"]:
        raise RuntimeError(f"splice geometry gate FAILED: {G_GEO}")
    log(f"G_GEO: ZEPHYRA at x-col {Z_XCOL} (address row {SPLICE_ADDR_ROW}) "
        f"in all windows: PASS")

    m = torch.zeros(pool_a_x.shape[0], BLOCK - 1, dtype=torch.bool)
    m[:, PRE - 1:] = True
    pool_mask = m

    read_base = {"a_pool": read_fact_position(net_b43, pool_a_x, name_ids, zid),
                 "b_pool": read_fact_position(net_b43, pool_b_x, name_ids, zid)}
    log(f"183-read on BASE (pre-ft): p(Z)@183 a-pool "
        f"{read_base['a_pool']['pz_onset_mean']:.4f} | b-pool "
        f"{read_base['b_pool']['pz_onset_mean']:.4f}")

    arms = {}
    for tag, pool in (("a_self_ctx_spliced", pool_a_x),
                      ("b_corpus_ctx_spliced", pool_b_x)):
        log(f"ARM {tag}: {FT_STEPS} steps, lr {FT_LR}, batch 32, seed "
            f"{CONS_SEED_SPLICE}")
        arms[tag] = finetune_arm(tag, net_b43, pool, pool_mask, anchor,
                                 train_ids, r_eval_xy, f_eval, zid,
                                 CONS_SEED_SPLICE, EVAL_EVERY_SPLICE)

    # gate vs e120's stored battery tables (install60 + held30, none deletion)
    e120_gate = {}
    for tag in arms:
        net_arm = evl_load(arms[tag]["sd"])
        diffs = []
        for j in GEO_ORDER:
            for bt in ("install60", "held30"):
                k = f"{tag}__none__g{j:+d}__{bt}"
                mine = battery_cell(net_arm, bat_ids[(j, bt)], zid)["mean_pz"]
                ref = refs["e120"]["battery_table"][k]["mean_pz"] \
                    if refs["e120"] and k in refs["e120"]["battery_table"] else None
                diffs.append(abs(mine - ref) if ref is not None
                             else float("nan"))
        md = max(diffs)
        e120_gate[tag] = {"max_abs_diff": md, "tol": G_BIT_TOL,
                          "fallback_tol": G_FALLBACK_TOL,
                          "bit_reproducible": bool(md < G_BIT_TOL),
                          "passes_e113_convention": bool(md < G_FALLBACK_TOL)}
        if not e120_gate[tag]["passes_e113_convention"] and not SMOKE:
            recipe_deviations.append(
                f"{tag} regeneration deviates from e120's stored table by "
                f"{md:.4f} (> {G_FALLBACK_TOL}) — REGENERATED-PRIMARY, "
                f"adjudication proceeds on this run's nets")
        log(f"G_E120[{tag}]: max |diff| vs e120 battery table {md:.3e} -> "
            f"{'BIT-REPRODUCIBLE' if md < G_BIT_TOL else ('CONVENTION-PASS' if md < G_FALLBACK_TOL else 'DEVIATION')}")
        del net_arm

    # the 183-geometry read on the regenerated arms (BEFORE any deletion)
    read183 = {}
    for tag in arms:
        net_arm = evl_load(arms[tag]["sd"])
        pool = pool_a_x if tag.startswith("a") else pool_b_x
        read183[tag] = read_fact_position(net_arm, pool, name_ids, zid)
        log(f"183-read [{tag}]: p(Z)@183 "
            f"{read183[tag]['pz_onset_mean']:.4f} | mean p(name char) "
            f"{read183[tag]['pname_mean_over7']:.4f} | frac>=0.5 "
            f"{read183[tag]['pname_frac_ge_0.5']:.3f}")

    # references for the probe-1 bars
    probe1 = {}
    for tag in arms:
        net_arm = evl_load(arms[tag]["sd"])
        arm_cells = {f"g{j:+d}": battery_cell(net_arm, bat_ids[(j, "install60")],
                                             zid)["mean_pz"]
                     for j in GEO_ORDER}
        del net_arm
        band = list(arm_cells.values())
        p = read183[tag]["pz_onset_mean"]
        inst60_g0 = arm_cells["g+0"]
        band_min, band_mean = min(band), float(np.mean(band))
        sig = p >= 0.5 * inst60_g0
        nosig = p <= 2.0 * band_min
        probe1[tag] = {
            "read183": read183[tag],
            "base_read": read_base["a_pool" if tag.startswith("a") else "b_pool"],
            "install60_g0_pre_deletion": inst60_g0,
            "battery_band_pre_deletion": arm_cells,
            "band_min": band_min, "band_mean": band_mean,
            "bar_signal": 0.5 * inst60_g0, "bar_nosignal": 2.0 * band_min,
            "signal_at_183": bool(sig), "no_signal_at_183": bool(nosig),
            "verdict": ("SIGNAL-AT-183" if sig else
                        "NO-SIGNAL-AT-183" if nosig else "AMBIGUOUS"),
        }
        log(f"PROBE1 [{tag}]: p@183 {p:.4f} vs 0.5x install-60 "
            f"({0.5 * inst60_g0:.4f}) and 2x band-min ({2.0 * band_min:.4f}) "
            f"-> {probe1[tag]['verdict']}")

    save_ckpt("e131_arm_a_self_spliced.pt", arms["a_self_ctx_spliced"]["sd"],
              {"recipe": "e120 arm (a) verbatim", "seed": CONS_SEED_SPLICE,
               "base": "runs/checkpoints/e082_b43_install.pt"},
              ckpt_inventory)
    save_ckpt("e131_arm_b_corpus_spliced.pt",
              arms["b_corpus_ctx_spliced"]["sd"],
              {"recipe": "e120 arm (b) verbatim", "seed": CONS_SEED_SPLICE,
               "base": "runs/checkpoints/e082_b43_install.pt"},
              ckpt_inventory)
    sd_b43 = {k: v.clone() for k, v in net_b43.state_dict().items()}
    del net_b43, evl

    # =====================================================================
    # PROBE 2/3/4 base — regenerate the e113 consolidated net verbatim
    # =====================================================================
    log("--- PHASE B: e113 consolidated net regeneration ---")
    net_inst = load_cpu(E048_CK)                    # the install-phase net
    evl = copy.deepcopy(net_inst)
    bz_inst = battery_cell(evl, f_eval, zid)
    G_E048 = {"battery_pz": bz_inst["mean_pz"], "ref": G_E048_REF,
              "tol": G_BIT_TOL,
              "pass": bool(abs(bz_inst["mean_pz"] - G_E048_REF) < G_BIT_TOL)}
    log(f"G_E048 install-phase battery p(Z) {bz_inst['mean_pz']:.10f} (ref "
        f"{G_E048_REF:.10f}): {'PASS' if G_E048['pass'] else 'FAIL'}")
    if not G_E048["pass"]:
        raise RuntimeError("e048_repro checkpoint failed its gate")
    del evl

    # e113 jittered-replay pool (verbatim construction)
    jit_x, jit_mask = {}, {}
    for j in JITTERS:
        wins_j = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            if len(w) != BLOCK:
                raise RuntimeError(f"window len {len(w)} != {BLOCK} at {j}")
            wins_j.append(w)
        jit_x[j] = torch.stack(wins_j)
        mm = torch.zeros(len(wins_j), BLOCK - 1, dtype=torch.bool)
        mm[:, PRE - 1 + j: PRE - 1 + j + len(NAME)] = True
        jit_mask[j] = mm
    pool_j_x = torch.cat([jit_x[j] for j in JITTERS])
    pool_j_mask = torch.cat([jit_mask[j] for j in JITTERS])
    log(f"jitter pool: {tuple(pool_j_x.shape)} (offsets {list(JITTERS)})")

    cons = finetune_arm("consolidated_e113", net_inst, pool_j_x, pool_j_mask,
                        anchor, train_ids, r_eval_xy, f_eval, zid,
                        CONS_SEED_JIT, EVAL_EVERY_JIT)
    sd_cons = cons["sd"]
    net_cons = evl_load(sd_cons)

    # gate vs e113's stored none-table
    diffs113 = []
    for j in GEO_ORDER:
        mine = battery_cell(net_cons, bat_ids[(j, "install60")], zid)["mean_pz"]
        ref = refs["e113"]["battery_table"][f"none__g{j:+d}__install60"]["mean_pz"]
        diffs113.append(abs(mine - ref))
    md113 = max(diffs113)
    G_E113 = {"max_abs_diff": md113, "tol": G_BIT_TOL,
              "fallback_tol": G_FALLBACK_TOL,
              "bit_reproducible": bool(md113 < G_BIT_TOL),
              "passes_e113_convention": bool(md113 < G_FALLBACK_TOL)}
    if not G_E113["passes_e113_convention"] and not SMOKE:
        recipe_deviations.append(
            f"e113 consolidated-net regeneration deviates from e113's stored "
            f"table by {md113:.4f} (> {G_FALLBACK_TOL}) — REGENERATED-PRIMARY")
    log(f"G_E113: max |diff| vs e113 none-table {md113:.3e} -> "
        f"{'BIT-REPRODUCIBLE' if md113 < G_BIT_TOL else ('CONVENTION-PASS' if md113 < G_FALLBACK_TOL else 'DEVIATION')}")

    save_ckpt("e131_consolidated_e113.pt", sd_cons,
              {"recipe": "e113 verbatim: jittered replay 300 steps",
               "seed": CONS_SEED_JIT, "base": "runs/checkpoints/e048_repro.pt"},
              ckpt_inventory)

    # =====================================================================
    # PROBE 2 — row-0 content test (e116 mean/zero-arm census)
    # =====================================================================
    log("--- PHASE C: row-0 content test (install-phase vs consolidated) ---")
    census = {}
    for phase, net in (("install_phase", net_inst), ("consolidated", net_cons)):
        base_pz = battery_pz(net, ids130, zid)
        rows = list(CENSUS_ROWS)
        m_d, z_d = {}, {}
        w = net.wpe.weight.data
        orig = w.clone()
        mean_row = orig.mean(0)
        for r in rows:
            w.copy_(orig); w[r] = mean_row
            m_d[r] = base_pz - battery_pz(net, ids130, zid)
            w.copy_(orig); w[r] = 0.0
            z_d[r] = base_pz - battery_pz(net, ids130, zid)
        w.copy_(orig)
        census[phase] = {
            "base_pz": base_pz,
            "rows": {str(r): {"mean": float(m_d[r]), "zero": float(z_d[r]),
                              "ratio": float(min(m_d[r], z_d[r]) /
                                             max(m_d[r], z_d[r]))
                                     if max(m_d[r], z_d[r]) > 0 else 0.0,
                              "strength": float(min(m_d[r], z_d[r])),
                              "content": bool(m_d[r] > 0 and z_d[r] > 0 and
                                              min(m_d[r], z_d[r]) /
                                              max(m_d[r], z_d[r]) >= 0.5)}
                     for r in rows}}
        log(f"census[{phase}]: base {base_pz:.4f} | row0 m/z "
            f"{m_d[0]:+.4f}/{z_d[0]:+.4f} | row1 m/z "
            f"{m_d[1]:+.4f}/{z_d[1]:+.4f}")

    # install-phase gate vs e116 seed-42 stored values
    G_E116 = {"note": "install-phase row-0 drops vs e116 seed-42 stored"}
    if refs["e116"] is not None:
        e116_r0 = refs["e116"]["per_seed"]["42"]["row0"]
        dm = abs(census["install_phase"]["rows"]["0"]["mean"] - e116_r0["mean"])
        dz = abs(census["install_phase"]["rows"]["0"]["zero"] - e116_r0["zero"])
        G_E116.update({"max_diff_mean": dm, "max_diff_zero": dz,
                       "tol": G_E116_TOL,
                       "pass": bool(max(dm, dz) < G_E116_TOL)})
        log(f"G_E116: row0 install-phase m/z diffs {dm:.2e}/{dz:.2e} -> "
            f"{'PASS' if G_E116['pass'] else 'FAIL'}")

    # probe-2 adjudication
    r0_i = census["install_phase"]["rows"]["0"]
    r0_c = census["consolidated"]["rows"]["0"]
    ctrl = [census["consolidated"]["rows"][str(r)] for r in CONTROL_ROWS]
    ctrl_band = {"rows": {str(r): census["consolidated"]["rows"][str(r)]
                          for r in CONTROL_ROWS},
                 "max_strength": max(c["strength"] for c in ctrl)}
    row0_positive = bool(r0_c["content"] and
                         (r0_c["strength"] >= r0_i["strength"] or
                          r0_c["strength"] > ctrl_band["max_strength"]))
    row0_null = bool((not r0_c["content"]) or
                     (r0_c["strength"] <= ctrl_band["max_strength"]))
    probe2 = {"install_row0": r0_i, "consolidated_row0": r0_c,
              "control_band": ctrl_band,
              "row0_content_positive": row0_positive,
              "row0_null": row0_null,
              "note": "RE-KEYED condition 1 fires iff row0_content_positive"}
    log(f"PROBE2: row0 consolidated content {r0_c['content']} strength "
        f"{r0_c['strength']:.4f} (install {r0_i['strength']:.4f}, control-max "
        f"{ctrl_band['max_strength']:.4f}) -> "
        f"{'POSITIVE (RE-KEYED cond.1)' if row0_positive else 'null' if row0_null else 'AMBIGUOUS'}")

    # =====================================================================
    # PROBE 3 — full-512 wpe delta census (weight-space)
    # =====================================================================
    log("--- PHASE D: full-512 wpe delta census ---")
    W_i = net_inst.state_dict()["wpe.weight"].clone()
    W_c = sd_cons["wpe.weight"].clone()
    delta = W_c - W_i
    u129 = W_i[129] / W_i[129].norm()
    u0 = W_i[0] / W_i[0].norm()
    dn = delta.norm(dim=1)
    p129 = (delta @ u129).abs()
    p0 = (delta @ u0).abs()
    n_rows = int(delta.shape[0])        # this line: block_size 256 -> 256 rows
    band_mask = torch.zeros(n_rows, dtype=torch.bool); band_mask[121:138] = True
    r0_mask = torch.zeros(n_rows, dtype=torch.bool); r0_mask[0] = True
    oob = ~(band_mask | r0_mask)
    band_med = float(p129[band_mask].median())
    band_med_norm = float(dn[band_mask].median())
    oob_rows = torch.nonzero(oob).flatten()
    oob_scores = p129[oob_rows]
    worst_i = int(oob_rows[torch.argmax(oob_scores)])
    worst = {"row": worst_i, "score": float(p129[worst_i]),
             "delta_norm": float(dn[worst_i]), "proj0": float(p0[worst_i])}
    top10_idx = torch.argsort(-p129)[:10]
    top10 = [{"row": int(i), "in_band": bool(band_mask[i]),
              "is_row0": bool(r0_mask[i]), "score129": float(p129[i]),
              "score0": float(p0[i]), "delta_norm": float(dn[i])}
             for i in top10_idx]
    oob_ge_med = sorted(int(r) for r in oob_rows if p129[r] >= band_med)
    census_fires = bool(len(oob_ge_med) > 0)
    probe3 = {
        "delta_reference": "consolidated minus install-phase (e048_repro)",
        "universe_rows": n_rows,
        "universe_note": "the tasking said 'full-512'; this line's wpe is "
                         "block_size 256 -> the census universe is ALL 256 "
                         "rows (512 was the e098 fresh-family block size); "
                         "recorded as a recipe deviation",
        "content_axis": "wpe_inst[129] unit vector (the fact's original "
                        "address row direction)",
        "score_definition": "|delta[r] . normalize(wpe_inst[129])|",
        "score129_all": [round(float(v), 6) for v in p129.tolist()],
        "score0_all": [round(float(v), 6) for v in p0.tolist()],
        "delta_norm_all": [round(float(v), 6) for v in dn.tolist()],
        "band_median_score129": band_med,
        "band_median_delta_norm": band_med_norm,
        "row0": {"score129": float(p129[0]), "score0": float(p0[0]),
                 "delta_norm": float(dn[0])},
        "row129": {"score129": float(p129[129]),
                   "delta_norm": float(dn[129])},
        "row183": {"score129": float(p129[183]),
                   "delta_norm": float(dn[183])},
        "top10_by_score129": top10,
        "worst_out_of_band": worst,
        "out_of_band_rows_ge_band_median": oob_ge_med[:20],
        "n_out_of_band_ge_band_median": len(oob_ge_med),
        "census_fires": census_fires,
        "note": "RE-KEYED condition 3 fires iff census_fires",
    }
    log(f"PROBE3: band median score {band_med:.4f} | worst OOB row "
        f"{worst['row']} score {worst['score']:.4f} | row0 score "
        f"{float(p129[0]):.4f} | n OOB>=median {len(oob_ge_med)} -> "
        f"{'FIRES (RE-KEYED cond.3)' if census_fires else 'no out-of-band home'}")

    # rider: same census for the e120 splice arms vs their B43 base
    rider = {}
    for tag in arms:
        W_arm = arms[tag]["sd"]["wpe.weight"]
        d = W_arm - sd_b43["wpe.weight"]
        dd = d.norm(dim=1)
        ua = sd_b43["wpe.weight"][129]
        ua = ua / ua.norm()
        pa = (d @ ua).abs()
        rider[tag] = {
            "axis": "B43 row-129 unit vector",
            "row183": {"delta_norm": float(dd[183]), "score129": float(pa[183])},
            "row0": {"delta_norm": float(dd[0]), "score129": float(pa[0])},
            "top10_by_delta_norm": [{"row": int(i), "delta_norm": float(dd[i])}
                                    for i in torch.argsort(-dd)[:10]],
            "note": "report-only (e120 metrics row-183 d_norm: a 0.163 / "
                    "b 0.158)",
        }
    del net_inst

    # =====================================================================
    # PROBE 4 — band-minus-row-0 deletion (scaffold-matched)
    # =====================================================================
    log("--- PHASE E: band-minus-row-0 deletion battery ---")
    DELS = {"none": (),
            "d_all_e113": E113_ADDR_ROWS,
            "d_all_r0": (0,) + E113_ADDR_ROWS,
            "d_all_r1": (1,) + E113_ADDR_ROWS,
            "d_r0": (0,),
            "d_r1": (1,)}
    table, gates_surg = {}, {}
    for dl_name, rows in DELS.items():
        if dl_name == "none":
            sd_del = {k: v.clone() for k, v in sd_cons.items()}
            gate = {"rows": [], "pass": True, "note": "no deletion"}
        else:
            sd_del, gate = deleted_wpe(sd_cons, rows)
        gates_surg[dl_name] = gate
        if not gate["pass"]:
            raise RuntimeError(f"deletion gate FAILED {dl_name}: {gate}")
        net_cons.load_state_dict(sd_del)
        for j in GEO_ORDER:
            for bt in ("install60", "held30"):
                table[(dl_name, j, bt)] = battery_cell(
                    net_cons, bat_ids[(j, bt)], zid,
                    keep_per_ctx=(bt == "install60" and dl_name != "none"))
        table[(dl_name, "ce_r", "-")] = {"ce_r": ce_fixed_cpu(net_cons,
                                                              *r_eval_xy)}
        log(f"deletion {dl_name:12s} (rows {list(rows) or '-'}): install60 "
            + " ".join(f"g{j_:+d} {table[(dl_name, j_, 'install60')]['mean_pz']:.3f}"
                     for j_ in GEO_ORDER)
            + f" | CE_R {table[(dl_name, 'ce_r', '-')]['ce_r']:.4f}")

    def geo_mean(dl, bt="install60"):
        return float(np.mean([table[(dl, j, bt)]["mean_pz"] for j in GEO_ORDER]))

    if refs["e113"] is not None:
        e113_dall_ref_cells = {
            f"g{j:+d}": refs["e113"]["battery_table"]
            [f"d_all_addresses__g{j:+d}__install60"]["mean_pz"]
            for j in GEO_ORDER}
    else:
        e113_dall_ref_cells = {f"g{j:+d}":
                               table[("d_all_e113", j, "install60")]["mean_pz"]
                               for j in GEO_ORDER}
    e113_dall_mean = float(np.mean(list(e113_dall_ref_cells.values())))
    gm_r0 = geo_mean("d_all_r0")
    collapse_bar, survive_bar = 0.5 * e113_dall_mean, 0.8 * e113_dall_mean
    collapses = bool(gm_r0 <= collapse_bar)
    survives = bool(gm_r0 >= survive_bar)
    probe4 = {"e113_dall_reference_cells": e113_dall_ref_cells,
              "e113_dall_reference_mean": e113_dall_mean,
              "this_run_d_all_e113_mean": geo_mean("d_all_e113"),
              "d_all_r0_mean": gm_r0,
              "d_all_r1_mean": geo_mean("d_all_r1"),
              "d_r0_mean": geo_mean("d_r0"),
              "d_r1_mean": geo_mean("d_r1"),
              "collapse_bar_50pct": collapse_bar,
              "survive_bar_20pct": survive_bar,
              "collapses": collapses, "survives": survives,
              "per_geometry": {dl: {f"g{j:+d}":
                                    table[(dl, j, "install60")]["mean_pz"]
                                    for j in GEO_ORDER}
                               for dl in DELS if dl != "none"},
              "note": "RE-KEYED condition 2 fires iff collapses"}
    log(f"PROBE4: e113 D-all ref mean {e113_dall_mean:.4f} | this-run D-all "
        f"{geo_mean('d_all_e113'):.4f} | +row0 {gm_r0:.4f} | +row1 "
        f"{geo_mean('d_all_r1'):.4f} | row0-only {geo_mean('d_r0'):.4f} "
        f"| row1-only {geo_mean('d_r1'):.4f} -> "
        f"{'COLLAPSES (RE-KEYED cond.2)' if collapses else 'survives' if survives else 'AMBIGUOUS'}")

    # =====================================================================
    # final adjudication (registered)
    # =====================================================================
    rekeyed_conditions = {"cond1_row0_content": probe2["row0_content_positive"],
                          "cond2_band_minus_row0_collapse": probe4["collapses"],
                          "cond3_census_out_of_band": probe3["census_fires"]}
    body_conditions = {"row0_null": probe2["row0_null"],
                       "band_minus_row0_survives": probe4["survives"],
                       "census_no_oob_home": not probe3["census_fires"]}
    n_rekeyed = sum(rekeyed_conditions.values())
    if n_rekeyed > 0:
        fired = "RE-KEYED" if n_rekeyed == 3 else \
            f"MIXED ({n_rekeyed}/3 re-keying conditions fire)"
    elif all(body_conditions.values()):
        fired = "BODY-STORED-GENUINE"
    else:
        fired = "AMBIGUOUS"
    log("=" * 78)
    log(f"E131 VERDICT: {fired}")
    for k, v in rekeyed_conditions.items():
        log(f"  {k}: {v}")
    log("  probe1 (reported separately): "
        + " | ".join(f"{t}: {probe1[t]['verdict']}" for t in probe1))
    log("=" * 78)

    # ---------------- outputs
    bt_json = {}
    for dl in DELS:
        bt_json[f"{dl}__ce_r"] = table[(dl, "ce_r", "-")]["ce_r"]
        for j in GEO_ORDER:
            for bt in ("install60", "held30"):
                bt_json[f"{dl}__g{j:+d}__{bt}"] = table[(dl, j, bt)]

    metrics = {
        "experiment": "e131_rekeying_census",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("R43 critic point 3 discriminator; coordinator "
                         "dispatch. Docstring + bars written before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does the consolidated fact live in the body, or did it "
                     "RE-KEY to undeleted positional rows (row 0, the 12 "
                     "intact band rows, or any out-of-band row incl. 183)?"),
        "nets": {
            "splice_line_base": f"runs/checkpoints/{B43_CK.name} (loaded, gated)",
            "splice_arms": "e120 arms (a)/(b) regenerated verbatim "
                           f"(seed {CONS_SEED_SPLICE}, {FT_STEPS} steps, CPU)",
            "consolidated_line_install": f"runs/checkpoints/{E048_CK.name} (loaded, gated)",
            "consolidated": f"e113 recipe verbatim (seed {CONS_SEED_JIT}, "
                            f"{FT_STEPS} steps, CPU)",
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_B43": G_B43, "G_E048": G_E048,
                  "G_GEO": G_GEO, "G_E120": e120_gate, "G_E113": G_E113,
                  "G_E116": G_E116, "G_SURG": gates_surg,
                  "harvest_zephyra_count": z_in_dreams,
                  "harvest_ref": None if SMOKE else HARVEST_REF_Z},
        "probe1_183_geometry_read": probe1,
        "probe2_row0_content": probe2,
        "probe2_census_tables": census,
        "probe3_delta_census": probe3,
        "probe3_rider_splice_arms": rider,
        "probe4_band_minus_row0": probe4,
        "probe4_battery_table": bt_json,
        "adjudication": {
            "rekeyed_conditions": rekeyed_conditions,
            "body_conditions": body_conditions,
            "fired": fired,
            "headline": (f"row0-content "
                         f"{'POSITIVE' if probe2['row0_content_positive'] else 'null'}; "
                         f"D-all+row0 {gm_r0:.3f} vs e113 D-all "
                         f"{e113_dall_mean:.3f} "
                         f"({'COLLAPSE' if collapses else 'survives' if survives else 'ambiguous'}); "
                         f"census OOB>=band-median "
                         f"{probe3['n_out_of_band_ge_band_median']} rows "
                         f"(worst r{worst['row']} {worst['score']:.4f} vs "
                         f"median {band_med:.4f}) | probe1: "
                         + " | ".join(f"{t.split('_')[0]}: "
                                      f"{probe1[t]['verdict']}"
                                      for t in probe1)),
        },
        "fine_tune": {"lr": FT_LR, "steps": FT_STEPS,
                      "time_cap_s": FT_TIME_CAP,
                      "batch": f"{NAME_BS} exposure + {ANCH_BS} anchor "
                               f"({ANCH_BS // 2} paired + {ANCH_BS // 2} random)",
                      "optimizer": "AdamW (0.9,0.95) wd 0.1 clip 1.0 constant lr",
                      "seeds": {"splice_arms": CONS_SEED_SPLICE,
                                "consolidated": CONS_SEED_JIT},
                      "device": "cpu",
                      "runs": {t: {"steps_ran": r["steps_ran"],
                                   "traj": r["traj"]}
                               for t, r in list(arms.items()) +
                               [("consolidated_e113", cons)]}},
        "trims": trims,
        "recipe_deviations": recipe_deviations,
        "ckpt_inventory": {"saved": ckpt_inventory,
                           "external_used": [
                               f"runs/checkpoints/{B43_CK.name}",
                               f"runs/checkpoints/{E048_CK.name}"],
                           "note": "saved under runs/checkpoints/ (convention, "
                                   "coordinator erratum); *.pt gitignored — "
                                   "on-disk only (R43 fix)"},
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072,
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot
    plot(rd / "rekeying_census.png", probe1, probe2, census, probe3, probe4,
         fired, rekeyed_conditions, worst, band_med)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'rekeying_census.png'}, "
        f"ckpts under runs/checkpoints/e131_*.pt")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def plot(path, probe1, probe2, census, probe3, probe4, fired,
         rekeyed, worst, band_med):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))

    # (0,0) probe 1: 183-read
    ax = axes[0, 0]
    tags = list(probe1)
    for k, t in enumerate(tags):
        v = probe1[t]
        ax.bar(k - 0.21, v["read183"]["pz_onset_mean"], 0.38,
               color="crimson" if t.startswith("a") else "darkorange",
               edgecolor="k", lw=0.5,
               label="arm p(Z)@183" if k == 0 else None)
        ax.bar(k + 0.21, v["base_read"]["pz_onset_mean"], 0.38,
               color="whitesmoke", edgecolor="k", lw=0.6, hatch="//",
               label="base (pre-ft) p(Z)@183" if k == 0 else None)
        ax.plot([k - 0.4, k + 0.4], [v["bar_signal"]] * 2, ls="--",
                color="seagreen", lw=1.4,
                label="SIGNAL bar (0.5x install-60)" if k == 0 else None)
        ax.plot([k - 0.4, k + 0.4], [v["bar_nosignal"]] * 2, ls=":",
                color="gray", lw=1.4,
                label="NO-SIGNAL bar (2x band-min)" if k == 0 else None)
        ax.text(k - 0.21, v["read183"]["pz_onset_mean"] + 0.004,
                f"{v['read183']['pz_onset_mean']:.4f}\n{v['verdict']}",
                ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(tags)))
    ax.set_xticklabels([f"({t.split('_')[0]}) "
                        f"{'self' if t.startswith('a') else 'corpus'}-ctx"
                        for t in tags], fontsize=8)
    ax.set_ylabel("p(Z) at position 183 (fact onset)")
    ax.set_title("PROBE 1: 183-geometry read on regenerated e120 arms "
                 "(before any deletion)", fontsize=10)
    ax.legend(fontsize=6.5, loc="upper right")

    # (0,1) probe 2: row-0 content test
    ax = axes[0, 1]
    rows_all = [0] + [r for r in (2, 3, 4, 5, 6, 60, 100, 118, 119, 120)] + [1]
    for k, (phase, col, al) in enumerate(
            (("install_phase", "steelblue", 0.45),
             ("consolidated", "crimson", 1.0))):
        vals = [census[phase]["rows"][str(r)]["strength"] for r in rows_all]
        ax.bar(np.arange(len(rows_all)) + (k - 0.5) * 0.38, vals, 0.36,
               color=col, alpha=al, edgecolor="k", lw=0.4, label=phase)
    ctrl_max = probe2["control_band"]["max_strength"]
    ax.axhline(ctrl_max, color="gray", ls=":", lw=1.2,
               label=f"control band max {ctrl_max:.3f}")
    ax.axhline(probe2["install_row0"]["strength"], color="steelblue",
               ls="--", lw=1.2,
               label=f"row0 install {probe2['install_row0']['strength']:.3f}")
    ax.set_xticks(np.arange(len(rows_all)))
    ax.set_xticklabels([("row0" if r == 0 else f"{r}") for r in rows_all],
                       fontsize=7, rotation=45)
    ax.set_ylabel("strength = min(mean-drop, zero-drop)")
    ax.set_title(f"PROBE 2: row-0 content test — consolidated row0 "
                 f"{'CONTENT-POSITIVE' if probe2['row0_content_positive'] else 'null'} "
                 f"(strength {probe2['consolidated_row0']['strength']:.3f}, "
                 f"arm ratio {probe2['consolidated_row0']['ratio']:.2f})",
                 fontsize=9.5)
    ax.legend(fontsize=6.5)

    # (1,0) probe 3: full census scan
    ax = axes[1, 0]
    scan = probe3["score129_all"]
    norms = probe3["delta_norm_all"]
    xs3 = np.arange(len(scan))
    ax.vlines(xs3, 0, scan, color="lightgray", lw=1.2)
    ax.plot(xs3[121:138], np.array(scan)[121:138], "o", ms=3.5,
            color="steelblue", label="grown band 121-137")
    ax.plot(0, scan[0], "o", ms=6, color="tab:red", label="row 0")
    ax.plot(183, scan[183], "s", ms=6, color="darkorange", label="row 183")
    for t in probe3["top10_by_score129"]:
        if not t["in_band"] and not t["is_row0"]:
            ax.plot(t["row"], t["score129"], "d", ms=5, color="purple")
            ax.annotate(f"r{t['row']}", (t["row"], t["score129"]),
                        xytext=(3, 2), textcoords="offset points", fontsize=6)
    ax.axhline(band_med, color="k", ls="--", lw=1.2,
               label=f"grown-band median {band_med:.4f}")
    ax.set_xlabel("wpe row (all 512)")
    ax.set_ylabel("|delta[r] . u129_inst| (content projection)")
    ax.set_title(f"PROBE 3: full-512 wpe delta census (consolidated - "
                 f"install-phase); worst OOB row {worst['row']} "
                 f"{worst['score']:.4f}; fires="
                 f"{probe3['census_fires']}", fontsize=9.5)
    ax.legend(fontsize=7)

    # (1,1) probe 4: deletions + verdict
    ax = axes[1, 1]
    dls = ["d_all_e113", "d_all_r0", "d_all_r1", "d_r0", "d_r1"]
    lbl = {"d_all_e113": "D-all (e113 5 rows)",
           "d_all_r0": "D-all + row0", "d_all_r1": "D-all + row1 (scaffold ctl)",
           "d_r0": "row0 only", "d_r1": "row1 only"}
    cols = {"d_all_e113": "steelblue", "d_all_r0": "crimson",
            "d_all_r1": "darkorange", "d_r0": "pink", "d_r1": "wheat"}
    geo_keys = ["g-8", "g-4", "g+0", "g+4", "g+8"]
    xs4 = np.arange(5)
    for dl in dls:
        vals = [probe4["per_geometry"][dl][kk] for kk in geo_keys]
        ax.plot(xs4, vals, "o-", ms=4, lw=1.1, color=cols[dl], label=lbl[dl])
    ax.axhspan(0, probe4["collapse_bar_50pct"], color="crimson", alpha=0.08)
    ax.axhline(probe4["collapse_bar_50pct"], color="crimson", ls="--", lw=1.1,
               label=f"collapse bar (50% of e113 D-all) "
                     f"{probe4['collapse_bar_50pct']:.3f}")
    ax.set_xticks(xs4)
    ax.set_xticklabels(geo_keys, fontsize=8)
    ax.set_ylabel("battery p(Z) install-60")
    ax.set_ylim(0, 1.05)
    ax.set_title("PROBE 4: band-minus-row-0 deletion (scaffold-matched)",
                 fontsize=10)
    ax.legend(fontsize=6.5, loc="lower left")

    txt = (f"VERDICT: {fired}\n"
           + "\n".join(f"  {k}: {v}" for k, v in rekeyed.items())
           + "\n  probe1: " + " | ".join(
               f"{t.split('_')[0]}={probe1[t]['verdict']}" for t in probe1))
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=7.2, va="top",
            family="monospace",
            bbox=dict(facecolor="lightyellow", alpha=0.94, edgecolor="gray"))

    fig.suptitle(f"E131 — the re-keying census: BODY-STORED-GENUINE vs "
                 f"RE-KEYED -> {fired}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
