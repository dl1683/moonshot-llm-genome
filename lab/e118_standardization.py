"""E118 — the STANDARDIZATION CONTROL (T063 claim-A's registered rebuttal).

[REGISTERED DESIGN — frozen in this docstring BEFORE any compute]

THE WORRY (T063, claim-A collapse risk): e108/e111's self/other structure —
the splice-time V-cos step 0.4032 (sibling, same trained net) vs 0.1410/
0.1383 (any differently-trained net) and the k*=7 subspace-occupancy
separation — could be a ROGUE-DIMENSION artifact (Timkey & Schütte 2021):
a few high-variance coordinates dominate every cosine, so the "self/other
step" is just SHARED ANISOTROPY (the recipient's own vectors and every
donor's vectors piling energy onto the same few dims), not family geometry.

THE CONTROL: z-score each of the 32 head-dim coordinates per (layer,head)
over the POOLED own+donor V-sets (recipient none-arm anchor-band set +
the three donor spliced sets, sibling/middle/foreign — ONE COMMON frame,
so every arm is standardized identically and cross-arm ratios stay exact),
then recompute on the standardized vectors:
  (a) the family cosines — mean|cos(old, donor)| per arm, the e099/e108
      instrument verbatim (BEFORE: must reproduce 0.4032/0.1410/0.1383);
  (b) the e111-style subsspace occupancy — uncentered PCA of the
      STANDARDIZED own set per group -> donor energy-fraction curves E(k)
      (headline k=8) + same-norm gaussian nulls (R=64, dedicated seeds
      8181/8182/8183 — never touching any arm stream).
Robustness variant: strict per-comparison Timkey frame (old_a + donor_a
pooled PER ARM — stats from the very vectors whose cosine is computed).

REGISTERED BARS (frozen):
  - SURVIVES => ROGUE-DIMENSION CONFOUND EXCLUDED (the family geometry is
    not anisotropy; T060/T061 claims hardened) iff post-standardization
    sibling >= 2x foreign in mean|cos| OR the energy gap persists
    (E_sibling(k=8) >= 2x E_foreign(k=8)).
  - COLLAPSES => ANISOTROPY-DRIVEN (the family signal is shared
    high-variance dims — a major reframe; report honestly) iff neither
    ratio reaches 2x AND sibling ~= foreign post (separation
    sib - max(mid, foreign) <= 0.05).
  - else PARTIAL (honest texture, reported).
Secondary (registered): which dims carry the most variance BEFORE
standardization — rogue-dim census (a raw coordinate dim holding > 25% of
pooled variance in a group), plus own-vs-donor variance-PROFILE
correlations (are the high-variance dims SHARED across nets?).

DATA: bit-exact regeneration per the rigs' own determinism pins — e111's
VERBATIM capture skeletons + e105/e108 donor machinery (imported, not
retyped); every stream a published seed; NO new sampling anywhere outside
the null replicates' dedicated generators.

GATES: G1 recipient val CE (tol 0.02); G2 params 873,472; G3 donor-window
determinism (bit-identical reruns); G4a published-artifact match vs
runs/e108/metrics.json (four arms' final-128 tail tokens + repl_stats
floats, tol 1e-12); G4b verbatim bit-identity (capture skeletons ==
e080.generate_arm / e099.generate_arm5, fresh seed-7 streams); G5 schedule
identity (324 replaced per arm, derangements 4343/4747/4646, native donor
sources, zero reuse); G6 analysis sanity (BEFORE-cos reproduces the
published step within 1e-6; z-frame identity — pooled standardized frame
mean~0/std~1 within 1e-9, zero degenerate dims; standardized own E(32)=1;
null mean curve within 0.01 of analytic k/32).

Run:     python lab/e118_standardization.py
Outputs: runs/e118/metrics.json + runs/e118/standardization.png
Envelope: NO training, NO new automations; CPU-only (CUDA masked pre-torch,
8 threads), single step, minutes. No NOTES/THINKING/QUEUE/STATE edits; no
commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e080)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import json  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import (  # noqa: E402
    REPO,
    Cfg,
    CharCorpus,
    TinyGPT,
    estimate_loss,
    run_dir,
    save_json,
)
import e080_prune_vs_replace as e080   # the verbatim rig (constants + arms)
import e099_attractor_identity as e099  # randomize arm + draw_donor
import e105_cross_family as e105        # crossfamily donors + position map
import e108_distance_ladder as e108     # middle donors (e040_ref)
import e111_self_signature as e111      # VERBATIM capture + PCA machinery

THREADS = 8                                # task spec / T050: 12 thrashes box
torch.set_num_threads(THREADS)             # (e080 sets 12; e099/e111 reset 8)

# ------------------------------------------------------------------ constants
ARMS = ["none", "randomize", "middle", "crossfamily"]
DONOR_ARMS = ["randomize", "middle", "crossfamily"]      # self/middle/foreign
K_MAX = 32                                 # head_dim; k axis = 1..32
KS_REPORT = (1, 2, 4, 8, 16, 32)           # headline table ks
K_HEADLINE = 8                             # the registered energy bar's k
R_NULL = 64                                # gaussian null replicates per arm
SEED_NULL = {"randomize": 8181, "middle": 8182, "crossfamily": 8183}
BAND_LO, BAND_HI = 64, 387                 # the 324 splice positions
N_BAND = BAND_HI - BAND_LO + 1             # 324

# e108/e111's published numbers (the step this control stress-tests)
E108_METRICS = REPO / "runs" / "e108" / "metrics.json"
E111_METRICS = REPO / "runs" / "e111" / "metrics.json"
PUB_V_COS = {"randomize": 0.4031668494478512,
             "middle": 0.14102163393464354,
             "crossfamily": 0.1383244868505884}
PUB_NORM_RATIO = {"randomize": 1.0262596847972385,
                  "middle": 1.0354408476455712,
                  "crossfamily": 1.2976166870858934}

# registered bars (frozen; see docstring)
RATIO_BAR = 2.0                            # sibling >= 2x foreign
COLLAPSE_SEP_BAR = 0.05                    # "sibling ~= foreign" separation
ROGUE_SHARE_BAR = 0.25                     # one raw dim > 25% of pooled var
VOCAB = 65
CHANCE_K = {k: float(np.sqrt(2.0 / (np.pi * k))) for k in range(1, K_MAX + 1)}

T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------ math helpers

def pair_cos(o: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """|cos| per row between paired (n, d) float64 vectors -> (n,)."""
    c = (o * s).sum(-1) / (o.norm(dim=-1) * s.norm(dim=-1)).clamp_min(1e-300)
    return c.abs()


def arm_cos_rows(G_old: dict, G_src: dict, groups: list) -> np.ndarray:
    """Per-row (8 battery sequences) mean|cos| pooled over groups+positions."""
    rows = []
    for g in groups:
        c = pair_cos(G_old[g], G_src[g]).view(-1, N_BAND).mean(1)  # (8,)
        rows.append(c[None])
    return torch.cat(rows, 0).mean(0).numpy()           # (8,)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    xm, ym = x - x.mean(), y - y.mean()
    return float((xm * ym).sum() / (xm.norm() * ym.norm()).clamp_min(1e-300))


def participation(w: torch.Tensor) -> float:
    """Effective rank (participation ratio) of an eigenspectrum."""
    return float((w.sum() ** 2) / (w * w).sum().clamp_min(1e-300))


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e118")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False,
                 threads_note="e080 module import sets 12; overridden to 8 "
                              "(task spec / T050)")

    # ---- recipient battery: e053c net EXACTLY as e080..e111 did
    corp = CharCorpus(REPO / "data" / "input.txt")       # seed 1337
    assert corp.vocab_size == VOCAB
    st = torch.load(e080.CKPT, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    cfg = Cfg(vocab=corp.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=e080.T_TOTAL)
    net = TinyGPT(cfg)
    net.load_state_dict(sd, strict=True)
    net.eval()
    n_params = net.num_params()
    gates["G2_params"] = dict(params=n_params, expected=873472,
                              ok=bool(n_params == 873472))
    val_ce = estimate_loss(net, corp, "val", n_batches=12)
    gates["G1_val_ce"] = dict(val_ce=val_ce, ref=e080.E053C_VAL_CE, tol=0.02,
                              ok=bool(abs(val_ce - e080.E053C_VAL_CE) <= 0.02))
    log(f"e053c recipient net loaded ({n_params:,} params) | val CE "
        f"{val_ce:.4f} vs e053c {e080.E053C_VAL_CE:.4f} -> G1 "
        f"{'PASS' if gates['G1_val_ce']['ok'] else 'FAIL'}")

    gen_p = torch.Generator().manual_seed(e080.SEED_PROMPT)
    ix = torch.randint(len(corp.val) - e080.PROMPT_TOK - 1, (e080.N_PROMPTS,),
                       generator=gen_p)
    prompts8 = [corp.val[i:i + e080.PROMPT_TOK] for i in ix]
    log(f"battery: {e080.N_PROMPTS} prompts (seed {e080.SEED_PROMPT})")

    # ================================================== G3: the donor nets
    def load_net(ckpt, arch, block, corpus_vocab):
        stx = torch.load(ckpt, map_location="cpu", weights_only=False)
        sdx = stx["model"] if isinstance(stx, dict) and "model" in stx else stx
        cfgx = Cfg(vocab=corpus_vocab, block_size=block, **arch)
        netx = TinyGPT(cfgx)
        netx.load_state_dict(sdx, strict=True)
        netx.eval()
        return netx, stx.get("step", None)

    net_m, step_m = load_net(e108.MID_CKPT, e108.MID_ARCH, e108.MID_BLOCK,
                             corp.vocab_size)
    val_ce_m = estimate_loss(net_m, corp, "val", n_batches=12)
    donors_m = {w: e108.donor_run_mid(net_m, corp, e108.MID_WIN[w]["prompt"],
                                      e108.MID_WIN[w]["sample"]) for w in (1, 2)}
    d1m = e108.donor_run_mid(net_m, corp, e108.MID_WIN[1]["prompt"],
                             e108.MID_WIN[1]["sample"])
    det_m = bool(torch.equal(d1m["idx"], donors_m[1]["idx"])
                 and all(torch.equal(a, b) for a, b in zip(d1m["V"],
                                                           donors_m[1]["V"])))
    corp21 = CharCorpus(e105.DONOR_CORPUS)                # seed 1337
    st21 = torch.load(e105.DONOR_CKPT, map_location="cpu", weights_only=False)
    sd21 = st21["model"] if isinstance(st21, dict) and "model" in st21 else st21
    cfg21 = Cfg(vocab=corp21.vocab_size, n_layer=6, n_head=6, n_embd=192,
                block_size=e105.DONOR_BLOCK)
    net21 = TinyGPT(cfg21)
    net21.load_state_dict(sd21, strict=True)
    net21.eval()
    val_ce21 = estimate_loss(net21, corp21, "val", n_batches=12)
    donors_f = {w: e105.donor_run(net21, corp21, e105.DONOR_WIN[w]["prompt"],
                                  e105.DONOR_WIN[w]["sample"]) for w in (1, 2)}
    d1f = e105.donor_run(net21, corp21, e105.DONOR_WIN[1]["prompt"],
                         e105.DONOR_WIN[1]["sample"])
    det_f = bool(torch.equal(d1f["idx"], donors_f[1]["idx"])
                 and all(torch.equal(a, b) for a, b in zip(d1f["V"],
                                                           donors_f[1]["V"])))
    gates["G3_donor_determinism"] = dict(
        middle=dict(ckpt=str(e108.MID_CKPT), step=step_m, val_ce=val_ce_m,
                    window1_rerun_bit_identical=det_m,
                    ref_e062_cached=e108.E062_BASE_CE["e040_ref"],
                    val_ce_ok=bool(abs(val_ce_m
                                       - e108.E062_BASE_CE["e040_ref"]) <= 0.05)),
        foreign=dict(ckpt=str(e105.DONOR_CKPT),
                     step=int(st21.get("step", -1)), val_ce=val_ce21,
                     window1_rerun_bit_identical=det_f,
                     ref_e063b=e105.E063B_VAL_CE_TASK,
                     val_ce_ok=bool(abs(val_ce21 - e105.E063B_VAL_CE_TASK)
                                    <= 0.15)),
        ok=bool(det_m and det_f))
    log(f"G3 donor determinism: middle {det_m} (val CE {val_ce_m:.4f}) | "
        f"foreign {det_f} (val CE {val_ce21:.4f}) -> "
        f"{'PASS' if gates['G3_donor_determinism']['ok'] else 'FAIL'}")

    # ============================================ THE FOUR ARMS (e111 VERBATIM)
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)          # none
    idx_none, V_none = e111.capture_none(net, prompts8, gen)
    donor_rz = e099.draw_donor(e099.SEED_DONOR, e080.N_PROMPTS)    # 4343
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)          # randomize
    C_rz = e111.capture_randomize(net, prompts8, gen, donor_rz)
    donor_mid = e099.draw_donor(e108.SEED_DONOR_MID, e080.N_PROMPTS)  # 4747
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)          # middle
    C_mid = e111.capture_donor_arm(net, prompts8, gen, donors_m, donor_mid)
    donor_cf = e099.draw_donor(e105.SEED_DONOR_MAP, e080.N_PROMPTS)   # 4646
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)          # crossfam
    C_cf = e111.capture_donor_arm(net, prompts8, gen, donors_f, donor_cf)
    CAP = {"none": dict(idx=idx_none), "randomize": C_rz,
           "middle": C_mid, "crossfamily": C_cf}
    for a in DONOR_ARMS:
        log(f"arm {a:11s}: {CAP[a]['n_replaced']} replaced | mean|cos(old,"
            f"donor)| {CAP[a]['mean_abs_cos']:.6f} (published "
            f"{PUB_V_COS[a]:.6f}) | norm ratio {CAP[a]['norm_ratio']:.4f}")

    # ---- G4b: verbatim bit-identity (e111's own gate, same rule)
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
    V0 = e080.generate_arm(net, prompts8, gen, "none")
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
    R5 = e099.generate_arm5(net, prompts8, gen, "randomize",
                            e099.draw_donor(e099.SEED_DONOR, e080.N_PROMPTS))
    g4b = dict(none_bit_identical=bool(torch.equal(idx_none, V0["idx"])),
               randomize_bit_identical=bool(torch.equal(C_rz["idx"],
                                                        R5["idx"])),
               rule="capture skeletons bit-identical to e080.generate_arm / "
                    "e099.generate_arm5 (fresh seed-7 streams)")
    g4b["ok"] = bool(g4b["none_bit_identical"] and g4b["randomize_bit_identical"])
    gates["G4b_verbatim_bit_identity"] = g4b
    log(f"G4b verbatim bit-identity: none {g4b['none_bit_identical']} | "
        f"randomize {g4b['randomize_bit_identical']} -> "
        f"{'PASS' if g4b['ok'] else 'FAIL'}")

    # ---- G4a: published-artifact match (e108's own outputs)
    g4a = dict(ref_file=str(E108_METRICS), tail_tokens_match={}, ok=None)
    if E108_METRICS.exists():
        with open(E108_METRICS) as f:
            m108 = json.load(f)
        devs = {}
        for a in ARMS:
            ref = np.asarray(m108["arms"][a]["tail_tokens"])
            new = CAP[a]["idx"][:, -128:].numpy()
            devs[a + "_tailtokens"] = float(np.abs(new - ref).max())
        for a in DONOR_ARMS:
            devs[a + "_cos"] = abs(CAP[a]["mean_abs_cos"] - PUB_V_COS[a])
            devs[a + "_normratio"] = abs(CAP[a]["norm_ratio"]
                                         - PUB_NORM_RATIO[a])
        max_dev = max(devs.values())
        g4a.update(per_quantity_max_dev=devs, ok=bool(max_dev <= 1e-12))
        g4a["note"] = (f"four arms vs runs/e108/metrics.json: tail tokens + "
                       f"repl_stats floats, max dev {max_dev:.2e} (bar 1e-12)")
    else:
        g4a["note"] = "runs/e108/metrics.json missing"
    gates["G4a_published_match"] = g4a
    log(f"G4a published match: {g4a['note']} -> "
        f"{'PASS' if g4a['ok'] else ('SKIPPED' if g4a['ok'] is None else 'FAIL')}")

    # ---- G5: schedule identity (e111's own gate, same rule)
    derange_ok = (all(d != b for b, d in enumerate(donor_rz))
                  and all(d != b for b, d in enumerate(donor_mid))
                  and all(d != b for b, d in enumerate(donor_cf)))
    src_pairs = [(w, ps) for p in range(64, 388)
                 for w, ps in [e105.donor_source(p)]]
    donor_pos_ok = all(64 <= ps <= 255 and w in (1, 2) for w, ps in src_pairs)
    reuse_ok = (len({ps for w, ps in src_pairs if w == 1}) == 192
                and len({ps for w, ps in src_pairs if w == 2}) == 132)
    counts_ok = all(CAP[a]["n_replaced"] == 324 for a in DONOR_ARMS)
    gates["G5_schedule_identity"] = dict(
        replaced_counts={a: CAP[a]["n_replaced"] for a in DONOR_ARMS},
        counts_ok=counts_ok,
        donor_maps=dict(randomize=list(donor_rz), middle=list(donor_mid),
                        crossfamily=list(donor_cf)),
        derangements_ok=derange_ok, donor_sources_native=donor_pos_ok,
        zero_reuse_ok=reuse_ok,
        ok=bool(counts_ok and derange_ok and donor_pos_ok and reuse_ok))
    log(f"G5 schedule: counts {counts_ok} | derangements {derange_ok} | "
        f"sources native {donor_pos_ok} | zero reuse {reuse_ok} -> "
        f"{'PASS' if gates['G5_schedule_identity']['ok'] else 'FAIL'}")

    # ================================================== ASSEMBLE THE V-SETS
    # recipient OWN set: none arm's final V at the 324 band positions,
    # (L, B, H, 324, 32) — e099/e111's representation, position-ascending.
    own = torch.stack([v[:, :, BAND_LO:BAND_HI + 1, :]
                       for v in V_none])                    # (L,8,H,324,32)
    log(f"own anchor-band V set: {tuple(own.shape)} = "
        f"{own.shape[1] * own.shape[3] * 16:,} vectors x 32-d")
    pairs = {a: e111.assemble_pairs(CAP[a]["caps"]) for a in DONOR_ARMS}
    for a in DONOR_ARMS:
        old_a, src_a = pairs[a]
        assert old_a.shape == own.shape, (a, tuple(old_a.shape))
        cs = ((old_a * src_a).sum(-1)
              / (old_a.norm(dim=-1) * src_a.norm(dim=-1)).clamp_min(1e-12))
        assert abs(float(cs.double().abs().mean())
                   - CAP[a]["mean_abs_cos"]) < 1e-6

    L, Bb, H = cfg.n_layer, e080.B, cfg.n_head
    groups = [(li, h) for li in range(L) for h in range(H)]  # 16 groups

    # per-group float64 matrices (the analysis representation)
    G_own = {g: own[g[0], :, g[1]].reshape(-1, K_MAX).to(torch.float64)
             for g in groups}
    G_old = {a: {g: pairs[a][0][g[0], :, g[1]].reshape(-1, K_MAX)
                 .to(torch.float64) for g in groups} for a in DONOR_ARMS}
    G_src = {a: {g: pairs[a][1][g[0], :, g[1]].reshape(-1, K_MAX)
                 .to(torch.float64) for g in groups} for a in DONOR_ARMS}

    # ============================================== BEFORE: the family cosines
    cos_pre_row = {a: arm_cos_rows(G_old[a], G_src[a], groups)
                   for a in DONOR_ARMS}
    cos_pre = {a: float(cos_pre_row[a].mean()) for a in DONOR_ARMS}
    cos_pre_ci = {a: e111.bootstrap_rows(cos_pre_row[a].reshape(-1, 1))
                  for a in DONOR_ARMS}
    k32_dev = max(abs(cos_pre[a] - PUB_V_COS[a]) for a in DONOR_ARMS)
    log("BEFORE (must reproduce the published step): "
        + " ".join(f"{a}:{cos_pre[a]:.4f}" for a in DONOR_ARMS)
        + f" | max dev vs published {k32_dev:.2e}")

    # ================================ VARIANCE PROFILE + the rogue-dim census
    # pooled frame per group: own + the three donor src sets (4 x 2592 rows)
    mu_g, sd_g, share_g = {}, {}, {}
    var_own_g, var_src_g = {}, {a: {} for a in DONOR_ARMS}
    max_share, argmax_dim, dims50, dims90, n_degen = [], [], [], [], 0
    for g in groups:
        Xp = torch.cat([G_own[g]] + [G_src[a][g] for a in DONOR_ARMS], 0)
        mu = Xp.mean(0)
        var = ((Xp - mu) ** 2).mean(0)               # population variance
        sd = var.sqrt()
        n_degen += int((sd < 1e-8).sum())
        mu_g[g], sd_g[g] = mu, sd.clamp_min(1e-12)
        sh = var / var.sum().clamp_min(1e-300)
        share_g[g] = sh
        srt, idx = torch.sort(sh, descending=True)
        max_share.append(float(srt[0]))
        argmax_dim.append(int(idx[0]))
        dims50.append(int((torch.cumsum(srt, 0) < 0.50).sum()) + 1)
        dims90.append(int((torch.cumsum(srt, 0) < 0.90).sum()) + 1)
        vo = G_own[g].mean(0)
        var_own_g[g] = ((G_own[g] - vo) ** 2).mean(0)
        for a in DONOR_ARMS:
            vs = G_src[a][g].mean(0)
            var_src_g[a][g] = ((G_src[a][g] - vs) ** 2).mean(0)
    n_rogue = int(sum(1 for m in max_share if m > ROGUE_SHARE_BAR))
    log(f"variance profile (pooled own+donors, per group): max raw-dim share "
        f"{min(max_share):.3f}..{max(max_share):.3f} (mean "
        f"{float(np.mean(max_share)):.3f}) | rogue dims (> "
        f"{ROGUE_SHARE_BAR:.0%}) in {n_rogue}/16 groups | dims to 50%: "
        f"{int(np.median(dims50))} (median) | degenerate dims {n_degen}")

    # own-vs-donor variance-PROFILE correlations (are high-var dims SHARED?)
    lv_corr = {}
    for a in DONOR_ARMS:
        cs_ = [pearson(torch.log(var_own_g[g] + 1e-12),
                       torch.log(var_src_g[a][g] + 1e-12)) for g in groups]
        lv_corr[a] = float(np.mean(cs_))
    lv_sf = float(np.mean([pearson(torch.log(var_src_g["randomize"][g] + 1e-12),
                                   torch.log(var_src_g["crossfamily"][g]
                                             + 1e-12)) for g in groups]))
    log("log-variance PROFILE correlations (mean over 16 groups): own~sibling "
        f"{lv_corr['randomize']:.3f} | own~middle {lv_corr['middle']:.3f} | "
        f"own~foreign {lv_corr['crossfamily']:.3f} | "
        f"sibling~foreign {lv_sf:.3f}")

    # ============================================== STANDARDIZE (the control)
    def zsc(X: torch.Tensor, g) -> torch.Tensor:
        return (X - mu_g[g]) / sd_g[g]

    Z_own = {g: zsc(G_own[g], g) for g in groups}
    Z_old = {a: {g: zsc(G_old[a][g], g) for g in groups} for a in DONOR_ARMS}
    Z_src = {a: {g: zsc(G_src[a][g], g) for g in groups} for a in DONOR_ARMS}
    # G6b: frame identity (pooled standardized frame is exactly mean-0/var-1)
    Zp_all = [torch.cat([Z_own[g]] + [Z_src[a][g] for a in DONOR_ARMS], 0)
              for g in groups]
    zmean_dev = max(float(Zp.mean(0).abs().max().item()) for Zp in Zp_all)
    zstd_dev = max(float((((Zp - Zp.mean(0)) ** 2).mean(0).sqrt() - 1.0)
                         .abs().max().item()) for Zp in Zp_all)
    log(f"standardized frame: max |per-dim pooled mean| {zmean_dev:.2e} | "
        f"max |per-dim pooled std - 1| {zstd_dev:.2e} (bar 1e-9) | "
        f"degenerate dims {n_degen}")

    # robustness variant: STRICT per-comparison Timkey frame (old_a + donor_a
    # pooled per arm — the standardization is fit on the very vectors whose
    # cosine the instrument computes)
    cos_rob = {}
    for a in DONOR_ARMS:
        rows = []
        for g in groups:
            Xa = torch.cat([G_old[a][g], G_src[a][g]], 0)
            mua, sda = Xa.mean(0), Xa.var(0, unbiased=False).sqrt() \
                .clamp_min(1e-12)
            o = (G_old[a][g] - mua) / sda
            s = (G_src[a][g] - mua) / sda
            rows.append(pair_cos(o, s).view(-1, N_BAND).mean(1)[None])
        cos_rob[a] = float(torch.cat(rows, 0).mean())
    log("robustness (strict per-comparison frame old+donor pooled): "
        + " ".join(f"{a}:{cos_rob[a]:.4f}" for a in DONOR_ARMS))

    # ============================================ AFTER (a): family cosines
    cos_post_row = {a: arm_cos_rows(Z_old[a], Z_src[a], groups)
                    for a in DONOR_ARMS}
    cos_post = {a: float(cos_post_row[a].mean()) for a in DONOR_ARMS}
    cos_post_ci = {a: e111.bootstrap_rows(cos_post_row[a].reshape(-1, 1))
                   for a in DONOR_ARMS}
    chance = CHANCE_K[K_MAX]
    log("AFTER (a) family cosines: "
        + " ".join(f"{a}:{cos_post[a]:.4f}" for a in DONOR_ARMS)
        + f" | isotropic chance {chance:.4f} | sib/foreign "
        f"{cos_post['randomize'] / cos_post['crossfamily']:.2f}x (was "
        f"{cos_pre['randomize'] / cos_pre['crossfamily']:.2f}x before)")

    # ================================= AFTER (b): subspace occupancy (e111)
    eig = {}
    own_curve_g, pr_pre, pr_post = [], [], []
    for g in groups:
        w_pre, _U = e111.pca_uncentered(G_own[g])
        pr_pre.append(participation(w_pre))
        w, U = e111.pca_uncentered(Z_own[g])          # standardized own PCA
        pr_post.append(participation(w))
        eig[g] = (w, U)
        own_curve_g.append(e111.energy_curve(Z_own[g], U))
    own_cum = torch.stack([torch.cumsum(eig[g][0], 0) for g in groups]).sum(0)
    own_curve_post = (own_cum / own_cum[-1]).numpy()
    top_share_post = np.array([[float(eig[(li, h)][0][0]
                                        / eig[(li, h)][0].sum())
                                for h in range(H)] for li in range(L)])
    log(f"own PCA after standardization: top-axis share "
        f"{top_share_post.min():.3f}..{top_share_post.max():.3f} (mean "
        f"{top_share_post.mean():.3f}; was 0.090..0.563 raw) | effective "
        f"rank {float(np.mean(pr_pre)):.1f} -> {float(np.mean(pr_post)):.1f}")

    E_post, E_row = {}, {}
    for a in DONOR_ARMS:
        cum_tot = torch.zeros(K_MAX, dtype=torch.float64)
        tot = 0.0
        row_num = torch.zeros(Bb, K_MAX, dtype=torch.float64)
        row_den = torch.zeros(Bb, dtype=torch.float64)
        for g in groups:
            U = eig[g][1]
            D = Z_src[a][g]
            e = ((D @ U) ** 2).sum(0)
            cum_tot += torch.cumsum(e, 0)
            tot += float(e.sum())
            eg = ((D.view(Bb, N_BAND, K_MAX) @ U) ** 2).sum(1)   # (8,32)
            row_num += torch.cumsum(eg, 1)
            row_den += eg.sum(1)
        E_post[a] = (cum_tot / tot).numpy()
        E_row[a] = (row_num / row_den[:, None]).numpy()
        log(f"AFTER (b) {a:11s} energy E(k): "
            + " ".join(f"k{k}:{E_post[a][k - 1]:.3f}" for k in KS_REPORT))

    # nulls: same-norm gaussian directions per standardized donor set
    E_null = {}
    for a in DONOR_ARMS:
        g_n = torch.Generator().manual_seed(SEED_NULL[a])
        curves = []
        for _ in range(R_NULL):
            cum_tot = torch.zeros(K_MAX, dtype=torch.float64)
            tot = 0.0
            for g in groups:
                U = eig[g][1]
                D = Z_src[a][g]
                z = torch.randn(D.shape, generator=g_n, dtype=torch.float64)
                z = z / z.norm(dim=-1, keepdim=True)
                zn = z * D.norm(dim=-1, keepdim=True)       # same norms
                e = ((zn @ U) ** 2).sum(0)
                cum_tot += torch.cumsum(e, 0)
                tot += float(e.sum())
            curves.append((cum_tot / tot).numpy())
        E_null[a] = np.stack(curves)                        # (R, 32)
        log(f"null[{a}] (std frame): mean E(k=8) "
            f"{E_null[a][:, 7].mean():.3f} "
            f"[{np.percentile(E_null[a][:, 7], 2.5):.3f},"
            f"{np.percentile(E_null[a][:, 7], 97.5):.3f}] (analytic "
            f"{8 / 32:.3f})")
    null_mean = {a: E_null[a].mean(0) for a in DONOR_ARMS}
    null_975 = {a: np.percentile(E_null[a], 97.5, axis=0) for a in DONOR_ARMS}
    E_ci = {a: e111.bootstrap_rows(E_row[a]) for a in DONOR_ARMS}

    # cos-in-subspace after standardization (texture; e111's (c) instrument)
    cosk_post, cosk_row = {}, {}
    for a in DONOR_ARMS:
        acc = torch.zeros(Bb, K_MAX, dtype=torch.float64)
        for g in groups:
            ck = e111.cos_curves(Z_old[a][g], Z_src[a][g], eig[g][1])
            acc += ck.view(Bb, N_BAND, K_MAX).mean(1)
        cosk_row[a] = (acc / len(groups)).numpy()
        cosk_post[a] = cosk_row[a].mean(0)
    cosk_ci = {a: e111.bootstrap_rows(cosk_row[a]) for a in DONOR_ARMS}

    # ---- G6: analysis sanity
    null_kdev = float(max(
        np.abs(E_null[a].mean(0) - np.arange(1, 33) / 32).max()
        for a in DONOR_ARMS))
    g6 = dict(
        pre_cos_max_dev=k32_dev, pre_cos_tol=1e-6,
        pre_cos_ok=bool(k32_dev < 1e-6),
        zframe_max_abs_pooled_mean=zmean_dev,
        zframe_max_abs_pooled_std_minus_1=zstd_dev, zframe_tol=1e-9,
        zframe_ok=bool(zmean_dev < 1e-9 and zstd_dev < 1e-9 and n_degen == 0),
        degenerate_dims=n_degen,
        own_E32_post=float(own_curve_post[31]),
        own_E32_ok=bool(abs(float(own_curve_post[31]) - 1.0) < 1e-9),
        null_mean_vs_k_over_32_maxdev=null_kdev, null_tol=0.01,
        ok=bool(k32_dev < 1e-6 and zmean_dev < 1e-9 and zstd_dev < 1e-9
                and n_degen == 0
                and abs(float(own_curve_post[31]) - 1.0) < 1e-9
                and null_kdev < 0.01))
    gates["G6_analysis_sanity"] = g6
    log(f"G6 analysis sanity: pre-cos dev {k32_dev:.2e} | z-frame dev "
        f"{zmean_dev:.2e} | own E(32) {float(own_curve_post[31]):.9f} | null "
        f"dev vs k/32 {null_kdev:.4f} -> "
        f"{'PASS' if g6['ok'] else 'FAIL'}")

    # ================================================ REGISTERED DECISION
    ks = np.arange(1, K_MAX + 1)
    ratio_cos_pre = cos_pre["randomize"] / cos_pre["crossfamily"]
    ratio_cos_post = cos_post["randomize"] / cos_post["crossfamily"]
    ratio_cos_post_mid = cos_post["randomize"] / cos_post["middle"]
    ratio_E_post = float(E_post["randomize"][K_HEADLINE - 1]
                         / E_post["crossfamily"][K_HEADLINE - 1])
    sep_post = cos_post["randomize"] - max(cos_post["middle"],
                                           cos_post["crossfamily"])
    survives_cos = bool(ratio_cos_post >= RATIO_BAR)
    survives_E = bool(ratio_E_post >= RATIO_BAR)
    collapses = bool((not survives_cos) and (not survives_E)
                     and sep_post <= COLLAPSE_SEP_BAR)
    # texture: the full e111 condition post-standardization
    ratio_post_full = E_post["randomize"] / E_post["crossfamily"]
    null_ok_post = np.array([bool(null_975["crossfamily"][k - 1]
                                  < E_post["crossfamily"][k - 1]
                                  and null_975["randomize"][k - 1]
                                  < E_post["randomize"][k - 1]) for k in ks])
    fires_post = [int(k) for k in ks[:8]
                  if ratio_post_full[k - 1] >= RATIO_BAR
                  and null_ok_post[k - 1]]
    excess_chance = {a: cos_post[a] - chance for a in DONOR_ARMS}
    clauses = dict(
        cos_ratio=dict(
            rule=f"post mean|cos| sibling >= {RATIO_BAR}x foreign",
            before=dict(sibling=cos_pre["randomize"],
                        middle=cos_pre["middle"],
                        foreign=cos_pre["crossfamily"], ratio=ratio_cos_pre),
            after=dict(sibling=cos_post["randomize"],
                       middle=cos_post["middle"],
                       foreign=cos_post["crossfamily"], ratio=ratio_cos_post),
            fires=survives_cos),
        energy_ratio=dict(
            rule=f"post E_sibling(k={K_HEADLINE}) >= {RATIO_BAR}x "
                 f"E_foreign(k={K_HEADLINE})",
            after=dict(sibling=float(E_post["randomize"][K_HEADLINE - 1]),
                       middle=float(E_post["middle"][K_HEADLINE - 1]),
                       foreign=float(E_post["crossfamily"][K_HEADLINE - 1]),
                       null=float(null_mean["crossfamily"]
                                  [K_HEADLINE - 1]),
                       ratio=ratio_E_post),
            full_condition_firing_ks=fires_post,
            fires=survives_E),
        collapse_sep=dict(
            rule=f"neither ratio >= {RATIO_BAR}x AND sib - max(mid,foreign) "
                 f"<= {COLLAPSE_SEP_BAR}",
            separation=float(sep_post), bar=COLLAPSE_SEP_BAR,
            fires=collapses))
    if survives_cos or survives_E:
        clause = "SURVIVES — ROGUE-DIMENSION CONFOUND EXCLUDED"
        fired = ("post mean|cos| sibling/foreign "
                 f"{ratio_cos_post:.2f}x (>= {RATIO_BAR}x)"
                 if survives_cos else "post mean|cos| sibling/foreign "
                 f"{ratio_cos_post:.2f}x (< {RATIO_BAR}x)")
        verdict = (
            f"SURVIVES: after per-dimension z-scoring over the pooled "
            f"own+donor frame (every coordinate's scale — the rogue-dimension "
            f"lever — removed; the frame is exactly mean-0/var-1 per dim per "
            f"group), the self/other separation persists: {fired}; "
            f"post E(8) sibling {E_post['randomize'][7]:.3f} vs foreign "
            f"{E_post['crossfamily'][7]:.3f} = {ratio_E_post:.2f}x "
            f"(null {null_mean['crossfamily'][7]:.3f}). "
            f"BEFORE (reproduced): sibling {cos_pre['randomize']:.4f} vs "
            f"{cos_pre['middle']:.4f}/{cos_pre['crossfamily']:.4f} "
            f"({ratio_cos_pre:.1f}x). The family geometry is NOT shared "
            f"anisotropy — whatever the standardized cosine reads is "
            f"correlation/SHAPE structure, not per-dim scale dominance. "
            f"T060's binary step and T061's subspace separation are hardened "
            f"against the Timkey rogue-dimension rebuttal. TEXTURE: "
            f"sibling post-cos {cos_post['randomize']:.4f} vs isotropic "
            f"chance {chance:.4f} (excess {excess_chance['randomize']:+.4f}); "
            f"foreign post-cos {cos_post['crossfamily']:.4f} (excess "
            f"{excess_chance['crossfamily']:+.4f}); strict per-comparison "
            f"frame gives sibling/foreign "
            f"{cos_rob['randomize'] / cos_rob['crossfamily']:.2f}x; own-PCA "
            f"effective rank {float(np.mean(pr_pre)):.1f} -> "
            f"{float(np.mean(pr_post)):.1f} (scale dominance removed from the "
            f"recipient's own manifold too).")
    elif collapses:
        clause = "COLLAPSES — ANISOTROPY-DRIVEN"
        verdict = (
            f"COLLAPSES: after per-dimension z-scoring over the pooled "
            f"own+donor frame, the self/other step is gone — post mean|cos| "
            f"sibling {cos_post['randomize']:.4f} vs foreign "
            f"{cos_post['crossfamily']:.4f} ({ratio_cos_post:.2f}x, bar "
            f"{RATIO_BAR}x) and post E(8) ratio {ratio_E_post:.2f}x, with "
            f"sibling - max(mid,foreign) = {sep_post:.4f} (<= "
            f"{COLLAPSE_SEP_BAR}). The family signal rides SHARED "
            f"HIGH-VARIANCE DIMENSIONS: the 0.40/0.14 step and the k*=7 "
            f"occupancy were per-coordinate scale dominance (the Timkey "
            f"rogue-dimension artifact), not family geometry. MAJOR REFRAME "
            f"of T060/T061 — report honestly; e112's causal splice results "
            f"(holographic identity check) are the remaining load-bearing "
            f"evidence and need their own re-reading under standardization.")
    else:
        clause = "PARTIAL — honest texture"
        verdict = (
            f"PARTIAL: post-standardization the separation neither clears the "
            f"{RATIO_BAR}x bars (cos ratio {ratio_cos_post:.2f}x; E(8) ratio "
            f"{ratio_E_post:.2f}x) nor fully collapses (sibling - "
            f"max(mid,foreign) = {sep_post:.4f} > {COLLAPSE_SEP_BAR}). "
            f"Post-cos: sibling {cos_post['randomize']:.4f} / middle "
            f"{cos_post['middle']:.4f} / foreign "
            f"{cos_post['crossfamily']:.4f} vs isotropic chance {chance:.4f}. "
            f"Substantial but scale-reduced family structure survives "
            f"z-scoring — report the texture and sharpen the bar before "
            f"claiming either direction.")
    rogue_present = bool(n_rogue > 0)
    secondary = dict(
        rule=f"a single RAW coordinate dim holding > {ROGUE_SHARE_BAR:.0%} of "
             f"pooled (own+donor) variance in a (layer,head) group",
        max_share_per_group=max_share, argmax_dim_per_group=argmax_dim,
        n_rogue_groups=n_rogue, n_groups=len(groups),
        rogue=rogue_present,
        mean_max_share=float(np.mean(max_share)),
        median_dims_to_50=int(np.median(dims50)),
        median_dims_to_90=int(np.median(dims90)),
        dims_to_50_per_group=dims50, dims_to_90_per_group=dims90,
        logvar_profile_corr=dict(
            own_vs={a: lv_corr[a] for a in DONOR_ARMS},
            sibling_vs_foreign=lv_sf,
            note="Pearson of log per-dim variance profiles, mean over 16 "
                 "groups; high own~donor correlations = high-variance dims "
                 "are SHARED across nets (the anisotropy the control "
                 "removes)"))
    log(f"SECONDARY rogue-dim census: {n_rogue}/16 groups have a raw dim > "
        f"{ROGUE_SHARE_BAR:.0%} of pooled variance -> "
        f"{'ROGUE DIM PRESENT' if rogue_present else 'no rogue dim'}; mean "
        f"top-dim share {float(np.mean(max_share)):.3f}; own~donor "
        f"log-var-profile corr "
        f"{lv_corr['randomize']:.3f}/{lv_corr['middle']:.3f}/"
        f"{lv_corr['crossfamily']:.3f} (sib/mid/for)")
    log(f"REGISTERED DECISION [{clause}]: {verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e118_standardization",
        purpose="T063 claim-A rebuttal hygiene: the rogue-dimension control. "
                "z-score each of the 32 head-dim coordinates per (layer,head) "
                "over the pooled own+donor V-sets (recipient none-arm "
                "anchor-band set + sibling/middle/foreign spliced sets — one "
                "common frame), then recompute (a) the family cosines "
                "mean|cos(old,donor)| (before: the published 0.4032/0.1410/"
                "0.1383 step) and (b) the e111-style subspace occupancy "
                "E_donor(k) in the standardized own-PCA (headline k=8) + "
                "same-norm gaussian nulls. SURVIVES (sibling >= 2x foreign "
                "in mean|cos| OR E(8) >= 2x) => rogue-dimension confound "
                "EXCLUDED, T060/T061 hardened; COLLAPSES (neither 2x and "
                "sib ~= foreign) => ANISOTROPY-DRIVEN, major reframe.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        recipient_net=dict(ckpt=str(e080.CKPT),
                           arch=dict(n_layer=4, n_head=4, n_embd=128,
                                     block_size=e080.T_TOTAL, vocab=VOCAB),
                           params=n_params, val_ce=val_ce,
                           val_ce_e053c=e080.E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=e080.SEED_PROMPT,
                   sampling=e080.SEED_SAMPLE,
                   donor_map_randomize=e099.SEED_DONOR,
                   donor_map_middle=e108.SEED_DONOR_MID,
                   donor_map_crossfamily=e105.SEED_DONOR_MAP,
                   middle_donor_windows=e108.MID_WIN,
                   foreign_donor_windows=e105.DONOR_WIN,
                   null_replicates=SEED_NULL, null_R=R_NULL,
                   note="every arm stream is a published seed (bit-exact "
                        "e111 regeneration); the ONLY new sampling is the "
                        "null replicates' dedicated generators"),
        protocol=dict(
            band=f"positions {BAND_LO}..{BAND_HI} ({N_BAND}), the e099/e105/"
                 f"e108 splice targets",
            v_representation="per-(layer,head) 32-d V vectors; 16 groups x "
                             "2592 vectors per set (own + per-arm old/donor)",
            standardization="per (layer,head), per coordinate: z = (x - "
                            "mu)/sd with mu/sd over the POOLED own+donor "
                            "sets (4 x 2592 rows per group; ONE COMMON frame "
                            "for all arms); strict per-comparison variant "
                            "(old_a + donor_a pooled) as robustness",
            energy_fraction="projected energy / total in the top-k axes of "
                            "the STANDARDIZED own uncentered PCA (e111 "
                            "convention, pooled-by-energy)",
            null=f"R={R_NULL} same-norm gaussian-direction replicates per "
                 f"standardized donor set; analytic reference k/32",
            published_step=dict(v_cos=PUB_V_COS, norm_ratio=PUB_NORM_RATIO)),
        gates=gates,
        before=dict(
            cos_mean={a: cos_pre[a] for a in DONOR_ARMS},
            cos_ci={a: dict(lo=float(cos_pre_ci[a][0][0]),
                            hi=float(cos_pre_ci[a][1][0]))
                    for a in DONOR_ARMS},
            cos_per_row={a: cos_pre_row[a].tolist() for a in DONOR_ARMS},
            ratio_sibling_over_foreign=ratio_cos_pre,
            isotropic_chance=chance),
        variance_profile=dict(
            frame="pooled own+{sibling,middle,foreign} src sets, centered "
                  "variance about the pooled mean, per (layer,head) group",
            share_per_group={f"L{li}H{h}": share_g[(li, h)].tolist()
                             for li, h in groups},
            max_share_per_group=max_share, argmax_dim_per_group=argmax_dim,
            dims_to_50_per_group=dims50, dims_to_90_per_group=dims90,
            degenerate_dims=n_degen,
            logvar_profile_corr=dict(own_vs={a: lv_corr[a]
                                             for a in DONOR_ARMS},
                                     sibling_vs_foreign=lv_sf),
            effective_rank_own_pca=dict(before=float(np.mean(pr_pre)),
                                        after=float(np.mean(pr_post)),
                                        per_group_before=pr_pre,
                                        per_group_after=pr_post),
            top_axis_share_after=top_share_post.tolist()),
        after=dict(
            cos_mean={a: cos_post[a] for a in DONOR_ARMS},
            cos_ci={a: dict(lo=float(cos_post_ci[a][0][0]),
                            hi=float(cos_post_ci[a][1][0]))
                    for a in DONOR_ARMS},
            cos_per_row={a: cos_post_row[a].tolist() for a in DONOR_ARMS},
            cos_excess_over_chance={a: excess_chance[a] for a in DONOR_ARMS},
            isotropic_chance=chance,
            cos_robust_percomparison_frame={a: cos_rob[a] for a in DONOR_ARMS},
            energy=dict(
                k=list(ks.tolist()),
                donor={a: E_post[a].tolist() for a in DONOR_ARMS},
                donor_ci={a: dict(lo=E_ci[a][0].tolist(), hi=E_ci[a][1].tolist())
                          for a in DONOR_ARMS},
                null_mean={a: null_mean[a].tolist() for a in DONOR_ARMS},
                null_975={a: null_975[a].tolist() for a in DONOR_ARMS},
                null_curves_note=f"{R_NULL} replicates per arm, seeds "
                                 f"{SEED_NULL}",
                own_insample=own_curve_post.tolist(),
                analytic_k_over_32=(ks / 32).tolist(),
                ratio_sibling_over_foreign=ratio_post_full.tolist(),
                null_below_both_ok=null_ok_post.tolist(),
                full_condition_firing_ks=fires_post),
            cos_in_subspace=dict(
                k=list(ks.tolist()),
                note="e111's (c) instrument on standardized vectors; k=1 "
                     "degenerate (|cos_1|=1); k=32 == the post-standard-"
                     "ization family cosine",
                mean={a: cosk_post[a].tolist() for a in DONOR_ARMS},
                ci={a: dict(lo=cosk_ci[a][0].tolist(), hi=cosk_ci[a][1].tolist())
                    for a in DONOR_ARMS},
                isotropic_chance=[CHANCE_K[k] for k in ks])),
        registered_decision=dict(
            frozen_rules=dict(
                survives=f"post sibling >= {RATIO_BAR}x foreign in mean|cos| "
                         f"OR E(8) >= {RATIO_BAR}x => ROGUE-DIMENSION "
                         f"CONFOUND EXCLUDED",
                collapses=f"neither ratio >= {RATIO_BAR}x AND sib - "
                          f"max(mid,foreign) <= {COLLAPSE_SEP_BAR} => "
                          f"ANISOTROPY-DRIVEN",
                else_="PARTIAL (honest texture)"),
            clauses=clauses, clause=clause, verdict=verdict),
        secondary_rogue_dim_census=secondary,
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "standardization.png", metrics)
    log(f"plot written; total wall {elapsed():0.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    B = M["before"]
    A = M["after"]
    E = A["energy"]
    C = A["cos_in_subspace"]
    dec = M["registered_decision"]
    sec = M["secondary_rogue_dim_census"]
    ks = np.asarray(E["k"])
    chance = B["isotropic_chance"]
    rB = B["cos_mean"]["randomize"] / B["cos_mean"]["crossfamily"]
    rA = A["cos_mean"]["randomize"] / A["cos_mean"]["crossfamily"]
    rE8 = E["donor"]["randomize"][7] / E["donor"]["crossfamily"][7]
    lv = M["variance_profile"]["logvar_profile_corr"]["own_vs"]
    lv_sf = M["variance_profile"]["logvar_profile_corr"]["sibling_vs_foreign"]
    eff = M["variance_profile"]["effective_rank_own_pca"]
    cols = {"randomize": "tab:green", "middle": "tab:purple",
            "crossfamily": "tab:red"}
    labs = {"randomize": "sibling (randomize, same net)",
            "middle": "middle (e040_ref)", "crossfamily": "foreign (e021_task)"}
    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: BEFORE/AFTER family cosines (THE control)
    x = np.arange(3)
    for i, a in enumerate(DONOR_ARMS):
        pre, post = B["cos_mean"][a], A["cos_mean"][a]
        ax1.plot([i, i], [pre, post], "-", color=cols[a], lw=2.2, alpha=0.7)
        ax1.scatter([i], [pre], facecolor="white", edgecolor=cols[a], s=110,
                    zorder=5, label="before z-scoring" if i == 0 else None)
        ax1.scatter([i], [post], color=cols[a], s=110, zorder=6,
                    label="after z-scoring (pooled own+donor frame)"
                    if i == 0 else None)
        lo, hi = A["cos_ci"][a]["lo"], A["cos_ci"][a]["hi"]
        ax1.plot([i, i], [lo, hi], color=cols[a], lw=1.2, alpha=0.6)
        rob = A["cos_robust_percomparison_frame"][a]
        ax1.scatter([i + 0.13], [rob], marker="^", color=cols[a], s=52,
                    alpha=0.75, zorder=5,
                    label="strict per-comparison frame" if i == 0 else None)
        ax1.annotate(f"{pre:.3f}", (i, pre), xytext=(-26, 4),
                     textcoords="offset points", fontsize=8, color=cols[a])
        ax1.annotate(f"{post:.3f}", (i, post), xytext=(-26, -12),
                     textcoords="offset points", fontsize=8, color=cols[a])
    ax1.axhline(chance, color="k", ls=":", lw=1.4,
                label=f"isotropic chance sqrt(2/pi/32) = {chance:.4f}")
    r_pre = rB
    r_post = rA
    ax1.text(0.02, 0.97, f"sib/foreign: {r_pre:.2f}x before -> {r_post:.2f}x "
                         f"after (bar {RATIO_BAR}x)", fontsize=9,
             transform=ax1.transAxes, va="top")
    ax1.set_xticks(x, ["sibling", "middle", "foreign"])
    ax1.set_xlim(-0.5, 2.6)
    ax1.set_ylabel("mean |cos(old, donor)|  (the e099/e108 instrument)")
    ax1.legend(fontsize=8, loc="center right")
    ax1.set_title("E118-1 — THE CONTROL: family cosines before/after "
                  "per-dimension z-scoring\n(pooled own+donor frame, one "
                  "common frame per (layer,head); bars = row-bootstrap 95% "
                  "CI)", fontsize=10)

    # ---- panel 2: post-standardization energy curves + null + e111 ghosts
    nm = np.asarray(E["null_mean"]["crossfamily"])
    nhi = np.asarray(E["null_975"]["crossfamily"])
    ax2.fill_between(ks, np.minimum(nm, np.asarray(E["null_mean"]
                                                   ["randomize"])), nhi,
                     color="gray", alpha=0.25,
                     label=f"same-norm gaussian null (R={R_NULL})")
    ax2.plot(ks, nm, color="gray", lw=1.6, label="null mean")
    ax2.plot(ks, np.asarray(E["analytic_k_over_32"]), ":", color="k", lw=1.2,
             label="isotropic analytic k/32")
    ax2.plot(ks, np.asarray(E["own_insample"]), "--", color="k", lw=2.0,
             label="recipient own set (in-sample, std)")
    if E111_METRICS.exists():
        with open(E111_METRICS) as f:
            m111 = json.load(f)["energy"]["donor"]
        for a in DONOR_ARMS:
            ax2.plot(ks, np.asarray(m111[a]), ls=(0, (1, 2)), lw=1.1,
                     color=cols[a], alpha=0.45,
                     label=f"{labs[a]} — e111 RAW (pre-z)" if a ==
                     "randomize" else None)
    for a in DONOR_ARMS:
        ax2.plot(ks, np.asarray(E["donor"][a]), "-o", color=cols[a], lw=2.2,
                 ms=4, label=labs[a])
    ax2.axvline(K_HEADLINE, color="tab:blue", ls="--", lw=1.2)
    ax2.text(K_HEADLINE - 0.4, 0.03, f"headline k={K_HEADLINE}", rotation=90,
             fontsize=8, color="tab:blue", va="bottom", ha="right")
    rE = rE8
    ax2.text(0.02, 0.97, f"E(8) sib/foreign = {rE:.2f}x after (bar "
                         f"{RATIO_BAR}x); e111 raw was 3.42x", fontsize=9,
             transform=ax2.transAxes, va="top")
    ax2.set_xlabel("k (top-k principal axes of the STANDARDIZED own V)")
    ax2.set_ylabel("energy fraction captured")
    ax2.set_xlim(1, 32)
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_title("E118-2 — donor subspace occupancy AFTER standardization "
                  "(e111's (b) recomputed)\nfaint dashed = published e111 raw "
                  "curves (same data, no z-scoring)", fontsize=10)

    # ---- panel 3: cos-in-subspace after standardization
    ch = np.asarray(C["isotropic_chance"])
    ax3.plot(ks, ch, ":", color="k", lw=1.4,
             label="isotropic chance sqrt(2/pi k)")
    for a in DONOR_ARMS:
        m = np.asarray(C["mean"][a])
        lo = np.asarray(C["ci"][a]["lo"])
        hi = np.asarray(C["ci"][a]["hi"])
        ax3.plot(ks, m, "-o", color=cols[a], lw=2.2, ms=4, label=labs[a])
        ax3.fill_between(ks, lo, hi, color=cols[a], alpha=0.18)
    ax3.axvline(K_HEADLINE, color="tab:blue", ls="--", lw=1.2)
    ax3.set_xlabel("k")
    ax3.set_ylabel("mean |cos_k(old, donor)| (standardized)")
    ax3.set_xlim(1, 32)
    ax3.set_ylim(0, 1.02)
    ax3.legend(fontsize=8)
    ax3.set_title("E118-3 — the V-cos instrument in the top-k subspace AFTER "
                  "standardization\n(k=32 == the post-z family cosine; bands = "
                  "row-bootstrap 95% CI)", fontsize=10)

    # ---- panel 4: the variance profile heat map (the rogue-dim census)
    share = np.asarray([M["variance_profile"]["share_per_group"][f"L{li}H{h}"]
                        for li in range(4) for h in range(4)])   # (16, 32)
    im = ax4.imshow(share, aspect="auto", cmap="viridis",
                    norm=LogNorm(vmin=max(share.min(), 1e-5),
                                 vmax=share.max()))
    rogue_mask = share > ROGUE_SHARE_BAR
    rr, cc = np.where(rogue_mask)
    ax4.scatter(cc, rr, marker="o", s=90, facecolor="none",
                edgecolor="red", lw=1.4,
                label=f"rogue dim (> {ROGUE_SHARE_BAR:.0%} of pooled var)")
    ax4.set_yticks(range(16), [f"L{li}H{h}" for li in range(4)
                               for h in range(4)], fontsize=7)
    ax4.set_xlabel("raw coordinate dim (0..31, pre-standardization)")
    ax4.set_title(f"E118-4 — pooled (own+donor) per-dim VARIANCE SHARE per "
                  f"(layer,head), log color\nrogue census: "
                  f"{sec['n_rogue_groups']}/16 groups encircle a dim > "
                  f"{ROGUE_SHARE_BAR:.0%}; mean top-dim share "
                  f"{sec['mean_max_share']:.3f}", fontsize=10)
    ax4.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax4, fraction=0.035, pad=0.01, label="variance share")

    # ---- panel 5: per-group max share + dims-to-50% (secondary)
    ms = np.asarray(sec["max_share_per_group"])
    ax5.bar(range(16), ms, color=["tab:red" if m > ROGUE_SHARE_BAR
                                  else "tab:blue" for m in ms], alpha=0.75)
    ax5.axhline(ROGUE_SHARE_BAR, color="tab:red", ls="--", lw=1.4,
                label=f"rogue bar {ROGUE_SHARE_BAR:.0%}")
    ax5.axhline(1.0 / 32, color="k", ls=":", lw=1.2,
                label="isotropic 1/32 = 3.1%")
    for i, (m, d) in enumerate(zip(ms, sec["argmax_dim_per_group"])):
        ax5.annotate(f"d{d}", (i, m), xytext=(0, 3), textcoords="offset "
                    "points", ha="center", fontsize=6)
    ax5.set_xticks(range(16), [f"L{li}H{h}" for li in range(4)
                               for h in range(4)], fontsize=7,
                   rotation=45)
    ax5.set_ylabel("max single-dim variance share (pooled own+donor)")
    ax5.set_ylim(0, max(0.6, ms.max() * 1.15))
    ax5b = ax5.twinx()
    ax5b.plot(range(16), sec["dims_to_50_per_group"], "k^--", ms=5, lw=1.0,
              alpha=0.7, label="dims to 50% of variance")
    ax5b.set_ylabel("dims needed for 50% of variance", color="k")
    ax5b.set_ylim(0, 32)
    h1, l1 = ax5.get_legend_handles_labels()
    h2, l2 = ax5b.get_legend_handles_labels()
    ax5.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax5.set_title(f"E118-5 — secondary: rogue-dim census + variance "
                  f"concentration\nlog-var-profile corr own~donor: sib "
                  f"{lv['randomize']:.2f} / mid {lv['middle']:.2f} / for "
                  f"{lv['crossfamily']:.2f} (sib~for {lv_sf:.2f})",
                  fontsize=10)

    # ---- panel 6: headline table + the registered decision
    ax6.axis("off")
    lines = ["headline (mean |cos| before -> after | E(k=8) after | null):",
             "arm       before   after   excess/chance |  E(8)   null  | "
             "ratio"]
    for a in DONOR_ARMS:
        lines.append(
            f"{a:11s} {B['cos_mean'][a]:7.4f} {A['cos_mean'][a]:7.4f} "
            f"{A['cos_excess_over_chance'][a]:+9.4f} | "
            f"{E['donor'][a][7]:5.3f} {E['null_mean'][a][7]:5.3f} | "
            f"{'sib/x' if a == 'randomize' else ''}")
    lines.append(f"sib/foreign cos: {rB:.2f}x -> {rA:.2f}x | "
                 f"E(8) ratio {rE8:.2f}x")
    lines.append("strict per-comparison frame cos: "
                 + " ".join(f"{A['cos_robust_percomparison_frame'][a]:.4f}"
                            for a in DONOR_ARMS))
    lines.append(f"effective rank of own PCA: {eff['before']:.1f} -> "
                 f"{eff['after']:.1f}")
    lines.append(f"rogue census: {sec['n_rogue_groups']}/16 groups | "
                 f"median dims to 50%: {sec['median_dims_to_50']}")
    lines.append("")
    lines.append("gates: " + " ".join(
        f"{g.split('_')[0]}:{'PASS' if v.get('ok') else 'FAIL'}"
        for g, v in M["gates"].items() if isinstance(v, dict)))
    lines.append("")
    lines.append(f"VERDICT [{dec['clause']}]:")
    lines += [f"  {wd}" for wd in textwrap.wrap(dec["verdict"], 96)]
    ax6.text(0.02, 0.97, "E118-6 — the registered decision (T063 rebuttal "
                         "hygiene)", fontsize=12, weight="bold", va="top")
    for i, t in enumerate(lines):
        ax6.text(0.02, 0.935 - i * 0.030, t, fontsize=8.3, va="top",
                 family="monospace")

    fig.suptitle(f"E118 — the standardization control (T063: rogue-dimension "
                 f"artifact?) | {dec['clause']} | sib/foreign cos {rB:.2f}x "
                 f"-> {rA:.2f}x (bar {RATIO_BAR}x) | published step "
                 f"0.403/0.141/0.138 reproduced (G4a/G6)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
