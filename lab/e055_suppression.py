"""E055 — THE SUPPRESSION LOCALIZER (REGISTERED; frozen in scratch/e055_design.md).

The first interventional depth-localization of an expressed-vs-installed
factual gap: transplant the model's OWN teacher-forced battery states into
its free run at incumbent-name onset decision-positions and measure the
depth-survival of the address.

Implements design §3 exactly:
 3.1 site harvesting: 4 seeds x 8 e048 gen-prompts x 350 chars (T=0.8,
     top-k 40) on e048_repro, BATCH-8 per seed; ELIZABETH/FLORIZEL onsets
     (registered primary host slots); per-site p(Z)/rank(Z)/argmax+p/in-model
     (cropped) position/terminal(t==120)-vs-deep(t>150) stratum; G3 counts
     gate per 8x350; seeds 4-7 extension if <12 onsets (registered fallback).
 3.2 arms: donors = the M5 four battery contexts (+2 held-out battery
     contexts as replication donors); for each site x depth d in {0..6}:
     A-TF (battery decision state at its pos-129), A-shuf (random-corpus
     130-char prefixes, Random(25501) family, n=4), A-rev (site state ->
     battery decision position, symmetry of the write); pad-shifted donors
     (left-pad 10 -> decision at wpe 139: same cue, no trained address) on
     4 sites (registered P1 discriminating check); write semantics one-shot
     AND held; readouts R1 (immediate P(Z-first) at next position), R2
     (one-shot downstream: 60 free chars x 4 samples, Z-words + full text),
     R3 (held downstream: 60 chars with the write held + p(Z) at the next
     name-onset inside the continuation). Reference curves: e048_direct800
     (its own sites + its own donors; the no-suppression reference), e001
     base (destruction reference). Sensitivity (unregistered, report-only):
     mean-donor state transplant (relay-direction analogue).
 3.3 stats: per depth AUC_d = P(p_z[TF] > p_z[shuf]) over all site x donor
     pairs (ties 0.5) with site-bootstrap CI; meanDelta_d with site-bootstrap
     95% CI (10k resamples); terminal/deep strata split. d* = shallowest d
     with site-mean R1 >= 0.30 and AUC_d >= 0.90.
 3.4 gates G1-G5 (any failure -> NO VERDICTS).
 3.5 budget <= 15 min CPU (batched; registered trim ladder: R2/R3 sites
     8 -> 6 if over).
 3.6 outputs: runs/e055/{metrics.json, depth_survival.png}.

Verdicts, frozen pre-run (design §0):
 P1 state-rescue: some d <= 5 with TF site-mean R1 >= 0.30 while the
    same-depth shuffled transplant stays <= 0.05 AND the site-bootstrap 95%
    CI on (TF - shuffled) excludes 0.
 P2 no-rescue / trajectory phenomenon: no d <= 5 reaches 0.30 on the
    immediate readout AND the one-shot-write downstream readout (Z-word
    count in 60 free chars) is 0 at every measured d for both one-shot and
    held writes.
 P3 mid-stack tie to the causal gate: d* in {2,3,4} AND the off-geometry
    non-monotonicity (d1 rescue >= 2x the d2 value) replicates at >= 2/3 of
    deep (t>150) sites.
 Discrimination bar ("rescue is knowledge-specific", per T028): AUC_d
 >= 0.90 AND meanDelta_d bootstrap CI excludes 0.

Addendum corrections incorporated (2026-09-25 takeover):
 - d6 is READOUT-DOMINATED (the uninstalled base net also shows d6 ~ 0.74
   under the same patch), so the meaningful rescue band is d <= 5 — which
   is already the registered P1/P3 bar; the d6 row is reported but flagged.
 - The onset decision is a three-way E/Z/M choice (site p0/t120:
   E 0.679 / Z 0.180 rank-2 / M 0.136); per-site p(E) and p(M) recorded.

No training. No edits to NOTES/THINKING/QUEUE/STATE. No git commit.

Run:  python lab/e055_suppression.py        (full run -> runs/e055/)
      E055_SMOKE=1 python lab/e055_suppression.py   (shakedown -> runs/e055_smoke/)
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")          # CPU experiment

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

torch.set_num_threads(min(16, os.cpu_count() or 8))

import torch.nn.functional as F                             # noqa: E402

import common                                               # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                                  # noqa: E402  (find_occ, SPLICE_RNG, jsonable)
import matplotlib.pyplot as plt                             # noqa: E402

SMOKE = os.environ.get("E055_SMOKE") == "1"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
HOST_RE = re.compile(r"(?<![A-Za-z])(ELIZABETH|FLORIZEL)(?![A-Za-z])")
ZW_RE = re.compile(r"(?<![A-Za-z])Z[A-Za-z]*")
ZEPH_RE = re.compile(r"(?<![A-Za-z])ZEPHYRA(?![A-Za-z])")
DEPTHS = list(range(7))                 # 0 = emb/input of block 0 ... 6 = final residual
MEANINGFUL_BAND = list(range(6))        # d <= 5 (d6 readout-dominated; addendum)
PAD = 10                                # pad-shifted donor left-pad -> decision at wpe 139
BUDGET_S = 900.0                        # registered <= 15 min
N_SEEDS = 1 if SMOKE else 4
N_PROMPTS = 8
GEN_TOK = 350
TEMP, TOPK = 0.8, 40
R23_SITES_FULL, R23_SITES_TRIM = (2, 2) if SMOKE else (8, 6)   # registered fallback: 8 -> 6
R23_SAMPLES = 2 if SMOKE else 4
N_BOOT = 10000
BOOT_SEED = 7
SITE_MIN = 12                           # <12 onsets -> extend seeds (registered fallback)
SITE_ENOUGH = 20                        # budget-ladder stop for adaptive seed count
SHUF_SEED = 25501                       # the M5 shuffled family (design §3.2)
N_SHUF = 4

trims: list[str] = []
deviations: list[str] = []


# ------------------------------------------------------------------ instruments

def load(path: Path):
    m = TinyGPT(Cfg())      # lab-standard 65/6/6/192/256
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def states_forward(net, toks):
    """toks: 1D -> (xs, logits); xs[0]=emb, xs[k]=output of block k-1."""
    toks = toks.unsqueeze(0)
    T = toks.shape[1]
    x = net.wte(toks) + net.wpe(torch.arange(T))
    xs = [x]
    for block in net.h:
        x = block(x)
        xs.append(x)
    return xs, net.lm_head(net.ln_f(x))


@torch.no_grad()
def patch_logits_batch(net, toks, pos, depth, states):
    """Overwrite the residual at `pos` with `states` (B, C) at depth d.
    Exact semantics of the probe's patched_logits: d=0 patches the embedding
    (input of block 0); d=k patches the output of block k-1; d=6 the final
    residual (pre-ln_f). Readout position = `pos` (the next-char logits)."""
    B, T = toks.shape
    x = net.wte(toks) + net.wpe(torch.arange(T))

    def put():
        x2 = x.clone()
        x2[:, pos] = states
        return x2

    if depth == 0:
        x = put()
    for i, block in enumerate(net.h):
        x = block(x)
        if i + 1 == depth:
            x = put()
    return net.lm_head(net.ln_f(x))


def pz_next(logits, zid):
    """P(Z) at each row's last position."""
    return F.softmax(logits[:, -1], -1)[:, zid]


@torch.no_grad()
def gen_batch(net, idx0, n_new, temperature=TEMP, top_k=TOPK):
    """Batched common.generate semantics (global RNG; seed externally)."""
    idx = idx0.clone()
    T0len = idx0.shape[1]
    for _ in range(n_new):
        cond = idx[:, -net.cfg.block_size:]
        logits, _ = net(cond)
        lg = logits[:, -1] / temperature
        v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
        lg = lg.masked_fill(lg < v[:, [-1]], float("-inf"))
        idx = torch.cat([idx, torch.multinomial(F.softmax(lg, -1), 1)], 1)
    return idx[:, T0len:]


@torch.no_grad()
def gen_held_batch(net, idx0, site_pos, depth, state, n_new,
                   temperature=TEMP, top_k=TOPK):
    """Generation with the write HELD at absolute sequence position
    `site_pos` (index within idx0; the window slides as the sequence grows)."""
    idx = idx0.clone()
    T0len = idx0.shape[1]
    for _ in range(n_new):
        cond = idx[:, -net.cfg.block_size:]
        p = site_pos - (idx.shape[1] - cond.shape[1])
        if p >= 0:
            logits = patch_logits_batch(net, cond, p, depth,
                                        state.unsqueeze(0).expand(cond.shape[0], -1))
        else:
            logits, _ = net(cond)
        lg = logits[:, -1] / temperature
        v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
        lg = lg.masked_fill(lg < v[:, [-1]], float("-inf"))
        nxt = torch.multinomial(F.softmax(lg, -1), 1)
        idx = torch.cat([idx, nxt], 1)
    return idx[:, T0len:]


@torch.no_grad()
def sample_first_batch(net, crop, dpos, depth, state, n, temperature=TEMP, top_k=TOPK):
    """Sample n onset chars from the patched next-char distribution."""
    logits = patch_logits_batch(net, crop.repeat(n, 1), dpos, depth,
                                state.unsqueeze(0).expand(n, -1))
    lg = logits[:, -1] / temperature
    v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
    lg = lg.masked_fill(lg < v[:, [-1]], float("-inf"))
    return torch.multinomial(F.softmax(lg, -1), 1)          # (n, 1)


# ------------------------------------------------------------------ statistics

def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auc_tf_shuf(tf: np.ndarray, sh: np.ndarray) -> float:
    """AUC = P(tf > sh) + 0.5 P(==) over all pairs (Mann-Whitney)."""
    if len(tf) == 0 or len(sh) == 0:
        return float("nan")
    r = _rankdata_avg(np.concatenate([tf, sh]))
    n1 = len(tf)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * len(sh)))


def site_bootstrap(tf_by_site, sh_by_site, n_boot=N_BOOT, seed=BOOT_SEED):
    """Resample SITES with replacement; recompute AUC and mean-delta.
    tf_by_site / sh_by_site: list of 1-D arrays (donor values per site)."""
    rng = np.random.default_rng(seed)
    n = len(tf_by_site)
    if n == 0:
        return {"auc_ci": [None, None], "delta_ci": [None, None]}
    aucs, deltas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        tf = np.concatenate([tf_by_site[i] for i in idx])
        sh = np.concatenate([sh_by_site[i] for i in idx])
        aucs.append(auc_tf_shuf(tf, sh))
        deltas.append(float(tf.mean() - sh.mean()))
    return {"auc_ci": [float(np.percentile(aucs, 2.5)),
                       float(np.percentile(aucs, 97.5))],
            "delta_ci": [float(np.percentile(deltas, 2.5)),
                         float(np.percentile(deltas, 97.5))]}


def depth_stats(tf_by_site, sh_by_site):
    tf_all = np.concatenate(tf_by_site) if tf_by_site else np.array([])
    sh_all = np.concatenate(sh_by_site) if sh_by_site else np.array([])
    out = {
        "n_sites": len(tf_by_site),
        "mean_tf": float(tf_all.mean()) if len(tf_all) else None,
        "mean_shuf": float(sh_all.mean()) if len(sh_all) else None,
        "mean_delta": float(tf_all.mean() - sh_all.mean()) if len(tf_all) else None,
        "auc": auc_tf_shuf(tf_all, sh_all),
    }
    out.update(site_bootstrap(tf_by_site, sh_by_site))
    return out


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e055_smoke" if SMOKE else "e055")
    log(f"E055 suppression localizer (smoke={SMOKE}) -> {rd}")

    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    stoi, itos = corpus.stoi, corpus.itos
    zid, eid, mid = stoi["Z"], stoi["E"], stoi["M"]
    train_ids = corpus.train
    train_text = "".join(itos[int(i)] for i in train_ids)

    CK = E43.REPO / "runs" / "checkpoints"
    net = load(CK / "e048_repro.pt")            # the G1-gated step-400 install
    net_base = load(CK / "e001.pt")             # destruction reference
    net_d8 = load(CK / "e048_direct800.pt")     # no-suppression reference
    log(f"nets loaded (repro/base/direct800); params {net.num_params():,}")

    # ---------------- protocol rebuild (e043-frozen; splice RNG 24301)
    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)
    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    assert (sum(1 for _, h in install_occ if h == "FLORIZEL"),
            sum(1 for _, h in install_occ if h == "ELIZABETH")) == (19, 41)
    gen_prompts = ([train_text[p - 120: p] for p, _ in install_occ[:4]]
                   + [train_text[p - 120: p] for p, _ in held_occ[:4]])[:N_PROMPTS]
    ctx130_i = [train_text[p - PRE: p] for p, _ in install_occ]
    log(f"protocol rebuilt: install-60 hosts F19/E41, {len(gen_prompts)} gen prompts")

    # ---------------- instrument check (M1: repro 0.556 / 0.817 / 1.18)
    @torch.no_grad()
    def battery_stats(m, ctxs, bs=30):
        rows = []
        for i in range(0, len(ctxs), bs):
            ids = torch.stack([corpus.encode(c) for c in ctxs[i:i + bs]])
            lg, _ = m(ids)
            lgl = lg[:, -1]
            pr = F.softmax(lgl, -1)
            for k in range(len(ctxs[i:i + bs])):
                rows.append({"p_z": float(pr[k, zid]),
                             "rank_z": int((lgl[k] > lgl[k, zid]).sum()) + 1,
                             "argmax": itos[int(lgl[k].argmax())]})
        return rows

    bat_repro = battery_stats(net, ctx130_i)
    bat_base = battery_stats(net_base, ctx130_i)
    bat_d8 = battery_stats(net_d8, ctx130_i)

    def _st(rows):
        return {"n": len(rows),
                "p_z_mean": sum(r["p_z"] for r in rows) / len(rows),
                "frac_argmax_z": sum(r["argmax"] == "Z" for r in rows) / len(rows),
                "mean_rank_z": sum(r["rank_z"] for r in rows) / len(rows)}

    instrument = {"repro": _st(bat_repro), "base": _st(bat_base), "d800": _st(bat_d8),
                  "repro_ref_e048": {"p_z": 0.556, "argmax_z": 0.817, "rank_z": 1.18},
                  "repro_matches_ref": bool(
                      abs(_st(bat_repro)["p_z_mean"] - 0.556) < 0.02)}
    log(f"instrument: repro p(Z) {instrument['repro']['p_z_mean']:.4f} "
        f"argmaxZ {instrument['repro']['frac_argmax_z']:.3f} "
        f"rank {instrument['repro']['mean_rank_z']:.2f} (e048 ref .556/.817/1.18) | "
        f"base {instrument['base']['p_z_mean']:.2e} d800 "
        f"{instrument['d800']['p_z_mean']:.3f}")

    # ---------------- donors (§3.2): M5 four + 2 held-out replication
    order = sorted(range(60), key=lambda i: -bat_repro[i]["p_z"])
    primary4 = list(dict.fromkeys(order[:2]
                                  + [i for i in order[2:] if install_occ[i][1] == "FLORIZEL"][:2]))[:4]
    replication2 = [i for i in order if i not in primary4][:2]
    donor_idx = primary4 + replication2

    def build_donors(m, idxs, tag):
        out = []
        for i in idxs:
            ids = corpus.encode(ctx130_i[i])
            xs, lg = states_forward(m, ids)
            pr = F.softmax(lg[0, -1], -1)
            out.append({"tag": tag, "ctx_i": i, "host": install_occ[i][1],
                        "p_z": float(pr[zid]), "argmax": itos[int(lg[0, -1].argmax())],
                        "states": [x[0, -1].clone() for x in xs]})
        return out

    donors = build_donors(net, donor_idx, "repro")
    for d in donors:
        d["family"] = "primary" if donors.index(d) < 4 else "replication"
        log(f"  donor ctx{d['ctx_i']:2d} ({d['host']:9s} {d['family']:11s}) "
            f"p(Z) {d['p_z']:.3f} argmax {d['argmax']}")

    # shuffled family: the M5 Random(25501) corpus prefixes
    _s = _random.Random(SHUF_SEED)
    shuf_ctxs = []
    while len(shuf_ctxs) < N_SHUF:
        q = _s.randrange(PRE + 1, len(train_ids) - 1)
        shuf_ctxs.append(train_text[q - PRE: q])

    def build_shuf(m):
        out = []
        for c in shuf_ctxs:
            xs, lg = states_forward(m, corpus.encode(c))
            out.append({"p_z": float(F.softmax(lg[0, -1], -1)[zid]),
                        "states": [x[0, -1].clone() for x in xs]})
        return out

    shuf = build_shuf(net)
    log("  shuffled donors own p(Z): " + " ".join(f"{d['p_z']:.2e}" for d in shuf))

    # pad-shifted donors (same cue, no trained address): left-pad 10 -> wpe 139
    pad_donors = []
    for i in donor_idx:
        ids = corpus.encode(" " * PAD + ctx130_i[i])
        xs, lg = states_forward(net, ids)
        pad_donors.append({"ctx_i": i, "host": install_occ[i][1],
                           "p_z_padded": float(F.softmax(lg[0, -1], -1)[zid]),
                           "states": [x[0, -1].clone() for x in xs]})
    log("  pad-shifted donors own p(Z): "
        + " ".join(f"{d['p_z_padded']:.3f}" for d in pad_donors))

    # ---------------- 3.1 trajectories + site harvesting
    prompt_ids = torch.stack([corpus.encode(p) for p in gen_prompts])

    def gen_trajectories(m, seed, tag):
        torch.manual_seed(seed)
        out = gen_batch(m, prompt_ids, GEN_TOK)
        conts = [corpus.decode(row.tolist()) for row in out]
        cnt = {w: sum(len(re.findall(r"(?<![A-Za-z])" + w + r"(?![A-Za-z])", c))
               for c in conts) for w in ("ZEPHYRA", "ELIZABETH", "FLORIZEL")}
        g3 = {"zephyra": cnt["ZEPHYRA"],
              "elizabeth_florizel": cnt["ELIZABETH"] + cnt["FLORIZEL"],
              "counts": cnt,
              "pass": bool(cnt["ZEPHYRA"] == 0 and cnt["ELIZABETH"] + cnt["FLORIZEL"] >= 3)}
        log(f"  traj[{tag}] counts {cnt} -> G3 {'PASS' if g3['pass'] else 'FAIL'}")
        return conts, g3

    sites = []
    traj_log = []
    seed_idx = 0
    while True:
        if seed_idx >= (1 if SMOKE else 8):   # registered fallback bound: seeds 4-7
            break
        conts, g3 = gen_trajectories(net, 5000 + 100 * seed_idx, f"seed{seed_idx}")
        traj_log.append({"seed_batch": seed_idx, "rng_seed": 5000 + 100 * seed_idx,
                         "g3": g3})
        for i, cont in enumerate(conts):
            for m in HOST_RE.finditer(cont):
                t = len(gen_prompts[i]) + m.start()
                sites.append({"prompt": i, "seed_batch": seed_idx, "t": t,
                              "name": m.group(1),
                              "ctx": gen_prompts[i] + cont[:m.start()]})
        seed_idx += 1
        if SMOKE:
            break
        if len(sites) >= SITE_ENOUGH and seed_idx >= 2:
            trims.append(f"adaptive_seed_stop_after_{seed_idx}_batches "
                         f"({len(sites)} sites >= {SITE_ENOUGH})")
            log(f"  adaptive stop: {len(sites)} sites after {seed_idx} seed batches")
            break
        if seed_idx >= N_SEEDS and len(sites) >= SITE_MIN:
            break
    if not sites:
        metrics = {"experiment": "e055_suppression", "smoke": SMOKE,
                   "error": "zero onset sites harvested", "sites_n": 0,
                   "gates": {"G3_traj_counts": {"per_batch":
                             [t["g3"] for t in traj_log]}}}
        save_json(rd / "metrics.json", E43.jsonable(metrics))
        log("ZERO SITES — aborting (metrics written)")
        return
    log(f"harvested {len(sites)} onset sites from {seed_idx}x8x{GEN_TOK} chars")

    # per-site instrument rows + crops + own states (for G1 + A-rev)
    for s in sites:
        crop = corpus.encode(s["ctx"])[-net.cfg.block_size:]
        dpos = len(crop) - 1
        with torch.no_grad():
            lg, _ = net(crop.unsqueeze(0))
        lgl = lg[0, -1]
        pr = F.softmax(lgl, -1)
        xs, _ = states_forward(net, crop)
        s.update({"crop_len": int(len(crop)), "dpos": int(dpos),
                  "in_model_pos": int(dpos),
                  "stratum": "terminal" if s["t"] == 120
                             else ("deep" if s["t"] > 150 else "mid"),
                  "base_p_z": float(pr[zid]), "base_p_e": float(pr[eid]),
                  "base_p_m": float(pr[mid]),
                  "base_rank_z": int((lgl > lgl[zid]).sum()) + 1,
                  "base_argmax": itos[int(lgl.argmax())],
                  "base_p_argmax": float(pr.max()),
                  "own_states": [x[0, dpos].clone() for x in xs]})
        s["crop"] = crop
    for s in sites:
        log(f"  site p{s['prompt']} t={s['t']:3d} {s['name']:9s} {s['stratum']:8s} "
            f"crop {s['crop_len']:3d} pos {s['dpos']:3d} p(Z) {s['base_p_z']:.4f} "
            f"r{s['base_rank_z']} argmax {s['base_argmax']}({s['base_p_argmax']:.2f}) "
            f"E {s['base_p_e']:.3f} M {s['base_p_m']:.3f}")
    n_term = sum(s["stratum"] == "terminal" for s in sites)
    n_mid = sum(s["stratum"] == "mid" for s in sites)
    n_deep = sum(s["stratum"] == "deep" for s in sites)
    log(f"strata: terminal {n_term} / mid {n_mid} / deep {n_deep}")

    # ---------------- gates G1/G2 (machinery, before arms)
    g1_max = 0.0
    for s in sites[:3]:
        lg0, _ = net(s["crop"].unsqueeze(0))
        for d in (0, 3, 6):
            lg = patch_logits_batch(net, s["crop"].unsqueeze(0), s["dpos"], d,
                                    s["own_states"][d].unsqueeze(0))
            g1_max = max(g1_max, float((lg[0, -1] - lg0[0, -1]).abs().max()))
    G1 = {"max_abs_dlogit": g1_max, "gate": 1e-4, "pass": bool(g1_max < 1e-4)}
    log(f"G1 self-patch identity max|dlogit| {g1_max:.2e} (<1e-4): "
        f"{'PASS' if G1['pass'] else 'FAIL'}")

    g2_rows = []
    s0 = sites[0]
    for dnr in donors:
        lg = patch_logits_batch(net, s0["crop"].unsqueeze(0), s0["dpos"], 6,
                                dnr["states"][6].unsqueeze(0))
        pz6 = float(pz_next(lg, zid))
        g2_rows.append({"donor_ctx": dnr["ctx_i"], "pz6": pz6,
                        "donor_pz": dnr["p_z"], "absdiff": abs(pz6 - dnr["p_z"])})
    G2 = {"rows": g2_rows, "max_absdiff": max(r["absdiff"] for r in g2_rows),
          "gate": 1e-5, "pass": bool(max(r["absdiff"] for r in g2_rows) < 1e-5)}
    log(f"G2 d6 == donor onset p(Z): max|diff| {G2['max_absdiff']:.2e}: "
        f"{'PASS' if G2['pass'] else 'FAIL'}")

    G3 = {"per_batch": [t["g3"] for t in traj_log],
          "pass": bool(all(t["g3"]["pass"] for t in traj_log))}
    G4 = {"shuf_own_pz": [d["p_z"] for d in shuf], "gate": 1e-4,
          "pass": bool(max(d["p_z"] for d in shuf) <= 1e-4)}
    log(f"G3 trajectory counts: {'PASS' if G3['pass'] else 'FAIL'} | "
        f"G4 shuffled floor max {max(d['p_z'] for d in shuf):.2e}: "
        f"{'PASS' if G4['pass'] else 'FAIL'}")

    # G5 determinism: rerun one continuation bit-identical (same-seed 60-char
    # batched generations on the install net)
    torch.manual_seed(97001)
    _r1 = [corpus.decode(r.tolist()) for r in gen_batch(net, prompt_ids, 60)]
    torch.manual_seed(97001)
    _r2 = [corpus.decode(r.tolist()) for r in gen_batch(net, prompt_ids, 60)]
    G5 = {"pass": bool(_r1 == _r2), "note": "same-seed 60-char batch reruns bit-identical"}
    log(f"G5 generation determinism: {'PASS' if G5['pass'] else 'FAIL'}")

    gates_pass = G1["pass"] and G2["pass"] and G3["pass"] and G4["pass"] and G5["pass"]

    # ---------------- 3.2 A-arms: the depth-survival curve (R1)
    log("A-arms R1: site x depth x {6 TF + 4 shuf + 6 pad + mean-donor} + base-net + A-rev")
    tf_vals = [[[] for _ in DEPTHS] for _ in sites]        # site -> depth -> donor vals
    shuf_vals = [[[] for _ in DEPTHS] for _ in sites]
    pad_vals = [[[] for _ in DEPTHS] for _ in sites]
    meandonor_vals = [[[] for _ in DEPTHS] for _ in sites]
    base_net_vals = [[[] for _ in DEPTHS] for _ in sites]
    mean_states = [torch.stack([d["states"][didx] for d in donors]).mean(0)
                   for didx in DEPTHS]

    for si, s in enumerate(sites):
        crop1 = s["crop"].unsqueeze(0)
        for d in DEPTHS:
            B = len(donors) + len(shuf) + len(pad_donors) + 1     # 6+4+6+1
            states = torch.stack([dnr["states"][d] for dnr in donors]
                                 + [x["states"][d] for x in shuf]
                                 + [x["states"][d] for x in pad_donors]
                                 + [mean_states[d]])
            lg = patch_logits_batch(net, crop1.repeat(B, 1), s["dpos"], d, states)
            pz = pz_next(lg, zid)
            tf_vals[si][d] = [float(v) for v in pz[:len(donors)]]
            shuf_vals[si][d] = [float(v) for v in pz[len(donors):len(donors) + N_SHUF]]
            pad_vals[si][d] = [float(v) for v in
                               pz[len(donors) + N_SHUF:len(donors) + N_SHUF + len(pad_donors)]]
            meandonor_vals[si][d] = [float(pz[-1])]
            lgb = patch_logits_batch(net_base, crop1, s["dpos"], d,
                                     donors[0]["states"][d].unsqueeze(0))
            base_net_vals[si][d] = [float(pz_next(lgb, zid))]

    # A-rev: site state -> battery decision position (donor0 ctx), symmetry arm
    rev_target = corpus.encode(ctx130_i[donors[0]["ctx_i"]])
    rev_base_pz = donors[0]["p_z"]
    rev_vals = [[] for _ in DEPTHS]                        # depth -> site vals
    for d in DEPTHS:
        states = torch.stack([s["own_states"][d] for s in sites])
        lg = patch_logits_batch(net, rev_target.unsqueeze(0).repeat(len(sites), 1),
                                PRE - 1, d, states)
        rev_vals[d] = [float(v) for v in pz_next(lg, zid)]
    log(f"A-rev (battery p(Z) {rev_base_pz:.3f} -> site-state writes): "
        + " ".join(f"d{d}:{np.mean(v):.3f}" for d, v in enumerate(rev_vals)))

    # ---------------- 3.3 statistics
    tf_by_depth_all = [[np.array(tf_vals[i][d]) for i in range(len(sites))] for d in DEPTHS]
    sh_by_depth_all = [[np.array(shuf_vals[i][d]) for i in range(len(sites))] for d in DEPTHS]
    idx_term = [i for i, s in enumerate(sites) if s["stratum"] == "terminal"]
    idx_deep = [i for i, s in enumerate(sites) if s["stratum"] == "deep"]
    tf_by_depth_term = [[np.array(tf_vals[i][d]) for i in idx_term] for d in DEPTHS]
    sh_by_depth_term = [[np.array(shuf_vals[i][d]) for i in idx_term] for d in DEPTHS]
    tf_by_depth_deep = [[np.array(tf_vals[i][d]) for i in idx_deep] for d in DEPTHS]
    sh_by_depth_deep = [[np.array(shuf_vals[i][d]) for i in idx_deep] for d in DEPTHS]

    stats_all = [depth_stats(tf_by_depth_all[d], sh_by_depth_all[d]) for d in DEPTHS]
    stats_term = [depth_stats(tf_by_depth_term[d], sh_by_depth_term[d])
                  if idx_term else None for d in DEPTHS]
    stats_deep = [depth_stats(tf_by_depth_deep[d], sh_by_depth_deep[d])
                  if idx_deep else None for d in DEPTHS]
    pad_curve = [float(np.mean([pad_vals[i][d] for i in range(len(sites))]))
                 for d in DEPTHS]
    meandonor_curve = [float(np.mean([meandonor_vals[i][d] for i in range(len(sites))]))
                       for d in DEPTHS]
    base_net_curve = [float(np.mean([base_net_vals[i][d] for i in range(len(sites))]))
                      for d in DEPTHS]
    base_site_curve = [float(np.mean([s["base_p_z"] for s in sites]))]

    log("R1 depth-survival (all sites):")
    for d in DEPTHS:
        st = stats_all[d]
        log(f"  d{d}: TF {st['mean_tf']:.3f} shuf {st['mean_shuf']:.4f} "
            f"dAUC {st['auc']:.4f} CI[{st['auc_ci'][0]:.3f},{st['auc_ci'][1]:.3f}] "
            f"delta {st['mean_delta']:.3f} CI[{st['delta_ci'][0]:.3f},"
            f"{st['delta_ci'][1]:.3f}] | pad {pad_curve[d]:.3f} "
            f"meandon {meandonor_curve[d]:.3f} base {base_net_curve[d]:.2e}")

    # d* = shallowest d (band d<=5) with site-mean R1 >= .30 AND AUC_d >= .90
    d_star = None
    for d in MEANINGFUL_BAND:
        if stats_all[d]["mean_tf"] is not None and stats_all[d]["mean_tf"] >= 0.30 \
                and stats_all[d]["auc"] >= 0.90:
            d_star = d
            break
    d_best = max(MEANINGFUL_BAND,
                 key=lambda d: stats_all[d]["mean_tf"] or 0.0)
    log(f"d* = {d_star} (shallowest d<=5 with TF mean >= .30 and AUC >= .90); "
        f"d_best(TF mean) = {d_best}")

    # ---------------- direct800 reference curve (its own sites + donors)
    d800 = {"instrument": instrument["d800"], "sites": [], "stats": None}
    if not SMOKE:
        log("d800 reference: trajectories + own sites + own donors")
        torch.manual_seed(9800)
        conts_d8 = [corpus.decode(r.tolist()) for r in
                    gen_batch(net_d8, prompt_ids, GEN_TOK)]
        d8_sites = []
        for i, cont in enumerate(conts_d8):
            for m in HOST_RE.finditer(cont):
                t = len(gen_prompts[i]) + m.start()
                d8_sites.append({"prompt": i, "t": t, "name": m.group(1),
                                 "ctx": gen_prompts[i] + cont[:m.start()]})
        order8 = sorted(range(60), key=lambda i: -bat_d8[i]["p_z"])
        d8_donor_idx = list(dict.fromkeys(
            order8[:2] + [i for i in order8[2:]
                          if install_occ[i][1] == "FLORIZEL"][:2]))[:4]
        d8_donors = build_donors(net_d8, d8_donor_idx, "d800")
        d8_shuf = build_shuf(net_d8)
        tf8 = [[[] for _ in DEPTHS] for _ in d8_sites]
        sh8 = [[[] for _ in DEPTHS] for _ in d8_sites]
        for si, s in enumerate(d8_sites):
            crop = corpus.encode(s["ctx"])[-net_d8.cfg.block_size:]
            dpos = len(crop) - 1
            lg, _ = net_d8(crop.unsqueeze(0))
            s["base_p_z"] = float(F.softmax(lg[0, -1], -1)[zid])
            s["stratum"] = "terminal" if s["t"] == 120 else ("deep" if s["t"] > 150 else "mid")
            s["crop_len"], s["dpos"] = int(len(crop)), int(dpos)
            for d in DEPTHS:
                B = len(d8_donors) + len(d8_shuf)
                states = torch.stack([x["states"][d] for x in d8_donors]
                                     + [x["states"][d] for x in d8_shuf])
                lgp = patch_logits_batch(net_d8, crop.unsqueeze(0).repeat(B, 1),
                                         dpos, d, states)
                pz = pz_next(lgp, zid)
                tf8[si][d] = [float(v) for v in pz[:len(d8_donors)]]
                sh8[si][d] = [float(v) for v in pz[len(d8_donors):]]
        st8 = [depth_stats([np.array(tf8[i][d]) for i in range(len(d8_sites))],
                           [np.array(sh8[i][d]) for i in range(len(d8_sites))])
               for d in DEPTHS] if d8_sites else None
        d800 = {"instrument": instrument["d800"], "n_sites": len(d8_sites),
                "sites": [{k: s[k] for k in ("prompt", "t", "name", "stratum",
                                             "crop_len", "base_p_z")} for s in d8_sites],
                "donors": [{k: d[k] for k in ("ctx_i", "host", "p_z", "argmax")}
                           for d in d8_donors],
                "stats": st8,
                "mean_base_pz": float(np.mean([s["base_p_z"] for s in d8_sites]))
                if d8_sites else None}
        if st8:
            log("  d800 reference curve: "
                + " ".join(f"d{d}:TF {st8[d]['mean_tf']:.3f}/sh "
                           f"{st8[d]['mean_shuf']:.3f}/AUC {st8[d]['auc']:.3f}"
                           for d in DEPTHS))

    # ---------------- R2/R3: one-shot vs held downstream readouts
    n_r23 = R23_SITES_FULL
    if not SMOKE and time.time() - T0 > 570:
        n_r23 = R23_SITES_TRIM
        trims.append(f"R2R3_sites_{R23_SITES_FULL}->{R23_SITES_TRIM} "
                     f"(elapsed {time.time()-T0:.0f}s > 570s)")
    n_samples = R23_SAMPLES
    if not SMOKE and time.time() - T0 > 690:
        n_samples = 3
        trims.append(f"R2R3_samples_{R23_SAMPLES}->3 (elapsed > 690s)")
    r23_depths = sorted(set([d_star if d_star is not None else d_best, 6]))

    order_pick = ([i for i, s in enumerate(sites) if s["stratum"] == "terminal"]
                  + [i for i, s in enumerate(sites) if s["stratum"] == "deep"]
                  + [i for i, s in enumerate(sites) if s["stratum"] == "mid"])
    r23_site_idx = order_pick[:n_r23]
    _picked = [(sites[i]["prompt"], sites[i]["t"], sites[i]["stratum"])
               for i in r23_site_idx]
    log(f"R2/R3: sites {_picked} x depths {r23_depths} x "
        f"[one-shot, held, base] x {n_samples} samples x 60 chars")

    donor0 = donors[0]
    r23 = []
    for si in r23_site_idx:
        s = sites[si]
        for d in r23_depths:
            for arm in ("oneshot", "held", "base"):
                if arm == "base" and d != r23_depths[0]:
                    continue          # base arm is depth-independent: run once
                torch.manual_seed(7100 + 97 * si + 13 * d + (0 if arm != "held" else 1))
                crop = s["crop"]
                if arm == "base":
                    first = None
                    idx1 = crop.unsqueeze(0).repeat(n_samples, 1)
                    out = gen_batch(net, idx1, 60)
                elif arm == "oneshot":
                    first = sample_first_batch(net, crop, s["dpos"], d,
                                               donor0["states"][d], n_samples)
                    idx1 = torch.cat([crop.unsqueeze(0).repeat(n_samples, 1), first], 1)
                    out = torch.cat([first, gen_batch(net, idx1, 59)], 1)
                else:   # held
                    out = gen_held_batch(net, crop.unsqueeze(0).repeat(n_samples, 1),
                                         s["dpos"], d, donor0["states"][d], 60)
                texts = [corpus.decode(row.tolist()) for row in out]
                counts = {"zephyra": sum(len(ZEPH_RE.findall(t)) for t in texts),
                          "z_words": sum(len(ZW_RE.findall(t)) for t in texts),
                          "first_z": sum(t[:1] == "Z" for t in texts),
                          "chars": sum(len(t) for t in texts)}
                row = {"site": si, "prompt": s["prompt"], "t": s["t"],
                       "stratum": s["stratum"], "depth": d, "arm": arm,
                       "n_samples": n_samples, "counts": counts, "texts": texts}
                # R3 extra: re-expression p(Z) at the next name-onset, write held
                if arm != "base":
                    reexp = []
                    for t in texts:
                        mm = HOST_RE.search(t)
                        if not mm:
                            continue
                        seq = torch.cat([crop, corpus.encode(t[:mm.start()])])
                        cond = seq[-net.cfg.block_size:]
                        p = s["dpos"] - (len(seq) - len(cond))
                        if p < 0:
                            continue
                        lgp = patch_logits_batch(net, cond.unsqueeze(0), p, d,
                                                 donor0["states"][d].unsqueeze(0))
                        pr = F.softmax(lgp[0, -1], -1)
                        reexp.append({"onset_at": int(mm.start()),
                                      "p_z": float(pr[zid]),
                                      "rank_z": int((lgp[0, -1] > lgp[0, -1, zid]).sum()) + 1})
                    row["reexpression"] = reexp or None
                r23.append(row)
                log(f"  R2/R3 site{si}(t={s['t']}) d{d} {arm:7s}: "
                    f"{counts} reexp "
                    f"{row.get('reexpression')}")

    # ---------------- verdicts (frozen §0 wording; no post-hoc bars)
    mean_tf = {d: stats_all[d]["mean_tf"] for d in DEPTHS}
    mean_sh = {d: stats_all[d]["mean_shuf"] for d in DEPTHS}

    p1_detail = {"rule": "some d<=5: TF site-mean >= .30 AND shuffled <= .05 "
                         "AND site-bootstrap 95% CI on TF-shuf excludes 0",
                 "per_depth": {}}
    p1_pass_depths = []
    for d in MEANINGFUL_BAND:
        ok = (mean_tf[d] is not None and mean_tf[d] >= 0.30
              and mean_sh[d] is not None and mean_sh[d] <= 0.05
              and stats_all[d]["delta_ci"][0] > 0)
        p1_detail["per_depth"][d] = {"mean_tf": mean_tf[d], "mean_shuf": mean_sh[d],
                                     "delta_ci": stats_all[d]["delta_ci"], "ok": bool(ok)}
        if ok:
            p1_pass_depths.append(d)
    p1_detail["pass_depths"] = p1_pass_depths
    P1 = bool(gates_pass and p1_pass_depths)

    # P2: no d<=5 reaches .30 on R1 AND R2/R3 Z-words 0 at every measured d,
    # both write semantics
    zw_nonzero = [r for r in r23 if r["arm"] in ("oneshot", "held")
                  and (r["counts"]["z_words"] > 0 or r["counts"]["zephyra"] > 0)]
    p2_detail = {"rule": "no d<=5 R1 TF site-mean >= .30 AND Z-word count == 0 "
                         "at every measured d for one-shot AND held",
                 "no_d5_reaches_030": all((mean_tf[d] or 0.0) < 0.30
                                          for d in MEANINGFUL_BAND),
                 "r2r3_zword_rows_nonzero": len(zw_nonzero),
                 "measured_depths": r23_depths}
    P2 = bool(gates_pass and p2_detail["no_d5_reaches_030"]
              and len(zw_nonzero) == 0)

    # P3: d* in {2,3,4} AND d1-peak/d2-crash (R1(d1) >= 2x R1(d2)) at >= 2/3
    # of deep (t>150) sites
    deep_idx = idx_deep
    d1d2 = []
    for i in deep_idx:
        r1 = float(np.mean(tf_vals[i][1]))       # TF mean over donors at d1
        r2 = float(np.mean(tf_vals[i][2]))
        raw = bool(r1 >= 2.0 * max(r2, 1e-6))
        guarded = bool(raw and r1 >= 0.01)
        d1d2.append({"site": i, "t": sites[i]["t"], "r1_d1": r1, "r1_d2": r2,
                     "ratio": r1 / max(r2, 1e-6), "raw": raw, "guarded": guarded})
    n_deep_sites = len(deep_idx)
    n_raw = sum(x["raw"] for x in d1d2)
    p3_detail = {"rule": "d* in {2,3,4} AND R1(d1) >= 2x R1(d2) at >= 2/3 of "
                         "deep (t>150) sites (raw rule, 1e-6 epsilon vs 0/0; "
                         "guarded rule also reported: peak >= 0.01)",
                 "d_star": d_star, "d_star_in_234": bool(d_star in (2, 3, 4)),
                 "deep_sites": d1d2, "n_deep": n_deep_sites,
                 "n_raw_replicate": n_raw,
                 "frac_raw_replicate": (n_raw / n_deep_sites) if n_deep_sites else None,
                 "n_guarded_replicate": sum(x["guarded"] for x in d1d2)}
    p3_clause2 = bool(n_deep_sites and n_raw / n_deep_sites >= 2 / 3)
    p3_detail["clause2_pass"] = p3_clause2
    P3 = bool(gates_pass and p3_detail["d_star_in_234"] and p3_clause2)

    disc = {"rule": "AUC_d >= 0.90 AND meanDelta_d bootstrap CI excludes 0 "
                    "(the causal discrimination bar per T028)",
            "per_depth": {d: {"auc": stats_all[d]["auc"],
                              "auc_ci": stats_all[d]["auc_ci"],
                              "mean_delta": stats_all[d]["mean_delta"],
                              "delta_ci": stats_all[d]["delta_ci"],
                              "knowledge_specific": bool(
                                  stats_all[d]["auc"] >= 0.90
                                  and stats_all[d]["delta_ci"][0] > 0)}
                          for d in DEPTHS}}
    pad_check = {"rule": "pad-shifted donors (same wpe cue, no trained address): "
                         "rescue dies -> knowledge-specific; rescue survives -> "
                         "position-cue leak (P1 would die)",
                 "pad_curve_4site": None}
    pad4_idx = ([i for i, s in enumerate(sites) if s["stratum"] == "terminal"][:2]
                + [i for i, s in enumerate(sites) if s["stratum"] == "deep"][:2])
    if len(pad4_idx) < 4:
        pad4_idx = list(range(len(sites)))[:4]
    pad4 = {d: float(np.mean([pad_vals[i][d] for i in pad4_idx])) for d in DEPTHS}
    tf4 = {d: float(np.mean([np.mean(tf_vals[i][d]) for i in pad4_idx])) for d in DEPTHS}
    sh4 = {d: float(np.mean([np.mean(shuf_vals[i][d]) for i in pad4_idx])) for d in DEPTHS}
    pad_check["sites"] = [(sites[i]["prompt"], sites[i]["t"]) for i in pad4_idx]
    pad_check["pad_curve_4site"] = pad4
    pad_check["tf_curve_4site"] = tf4
    pad_check["shuf_curve_4site"] = sh4
    pad_check["verdict_note"] = (
        "pad-shifted rescue at d<=5 comparable to TF (>= half the TF mean at any "
        "d<=5) would indicate position-cue donor leak; here max pad mean over "
        f"d<=5 = {max(pad4[d] for d in MEANINGFUL_BAND):.3f} vs TF "
        f"{max(tf4[d] for d in MEANINGFUL_BAND):.3f}")
    log(f"pad-shifted (4-site) d<=5: pad max {max(pad4[d] for d in MEANINGFUL_BAND):.3f} "
        f"vs TF max {max(tf4[d] for d in MEANINGFUL_BAND):.3f}")

    log("=" * 72)
    log(f"GATES {'ALL PASS' if gates_pass else 'FAIL -> NO VERDICTS'}")
    log(f"P1 state-rescue: {P1} (pass depths {p1_pass_depths})")
    log(f"P2 no-rescue/trajectory: {P2} ({p2_detail})")
    log(f"P3 mid-stack tie: {P3} (d*={d_star}, d1/d2 raw {n_raw}/{n_deep_sites})")

    # ---------------- outputs
    sites_out = [{k: s[k] for k in ("prompt", "seed_batch", "t", "name", "stratum",
                                    "crop_len", "in_model_pos", "base_p_z", "base_p_e",
                                    "base_p_m", "base_rank_z", "base_argmax",
                                    "base_p_argmax")} for s in sites]
    metrics = {
        "experiment": "e055_suppression",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design": "scratch/e055_design.md (frozen registration §0-§5 + addendum)",
        "smoke": SMOKE,
        "net": "runs/checkpoints/e048_repro.pt (G1-gated bit-level repro of "
               "e043 Dmix@s400, saved at step 400)",
        "addendum_corrections": [
            "d6 readout rescue is TRIVIAL (uninstalled base net d6 ~ 0.74): "
            "d6 is readout-dominated; the meaningful rescue band is d<=5 "
            "(already the registered P1/P3 bar).",
            "onset decision = three-way E/Z/M (p0/t120: E .679 / Z .180 rank2 / "
            "M .136); per-site p(E), p(M) recorded."],
        "instrument_check": instrument,
        "protocol": {"splice_rng": E43.SPLICE_RNG,
                     "install_hosts": {"FLORIZEL": 19, "ELIZABETH": 41},
                     "gen_prompts": len(gen_prompts), "gen_toks": GEN_TOK,
                     "temp": TEMP, "top_k": TOPK, "seed_batches": seed_idx,
                     "traj_seeds": [5000 + 100 * k for k in range(seed_idx)]},
        "gates": {"G1_selfpatch": G1, "G2_d6_construction": G2, "G3_traj_counts": G3,
                  "G4_shuffled_floor": G4, "G5_determinism": G5,
                  "all_pass": gates_pass},
        "sites": {"n": len(sites), "strata": {"terminal": n_term, "mid": n_mid,
                                              "deep": n_deep},
                  "table": sites_out,
                  "base_pz_range": [min(s["base_p_z"] for s in sites),
                                    max(s["base_p_z"] for s in sites)]},
        "donors": [{k: d[k] for k in ("tag", "ctx_i", "host", "p_z", "argmax",
                                      "family")} for d in donors],
        "shuffled": {"contexts_seed": SHUF_SEED, "n": N_SHUF,
                     "own_pz": [d["p_z"] for d in shuf]},
        "pad_shifted_donors": {"pad": PAD, "decision_wpe": PRE - 1 + PAD,
                               "own_pz_padded": [d["p_z_padded"] for d in pad_donors]},
        "r1_curves": {
            "tf_mean": {d: mean_tf[d] for d in DEPTHS},
            "shuf_mean": {d: mean_sh[d] for d in DEPTHS},
            "pad_mean": {d: pad_curve[d] for d in DEPTHS},
            "mean_donor_mean": {d: meandonor_curve[d] for d in DEPTHS},
            "base_net_mean": {d: base_net_curve[d] for d in DEPTHS},
            "base_site_mean": base_site_curve,
            "a_rev": {"target_ctx": donors[0]["ctx_i"], "battery_pz": rev_base_pz,
                      "mean_by_depth": {d: float(np.mean(v))
                                        for d, v in enumerate(rev_vals)}},
            "per_site_tf": {f"site{si}_t{sites[si]['t']}_"
                            f"{sites[si]['stratum']}":
                            {d: tf_vals[si][d] for d in DEPTHS}
                            for si in range(len(sites))},
            "per_site_shuf": {f"site{si}_t{sites[si]['t']}":
                              {d: shuf_vals[si][d] for d in DEPTHS}
                              for si in range(len(sites))},
            "per_site_pad": {f"site{si}_t{sites[si]['t']}":
                             {d: pad_vals[si][d] for d in DEPTHS}
                             for si in range(len(sites))},
            "per_site_base_net": {f"site{si}_t{sites[si]['t']}":
                                  {d: base_net_vals[si][d] for d in DEPTHS}
                                  for si in range(len(sites))},
        },
        "stats_by_depth": {"all": {d: stats_all[d] for d in DEPTHS},
                           "terminal": {d: stats_term[d] for d in DEPTHS},
                           "deep": {d: stats_deep[d] for d in DEPTHS}},
        "d_star": d_star, "d_best_tf_mean": d_best,
        "discrimination": disc,
        "pad_shifted_check": pad_check,
        "direct800_reference": d800,
        "r2r3": {"depths": r23_depths, "site_idx": r23_site_idx,
                 "samples_per_combo": n_samples, "rows": r23},
        "verdicts": {"gates_all_pass": gates_pass,
                     "P1_state_rescue": P1, "P1_detail": p1_detail,
                     "P2_no_rescue": P2, "P2_detail": p2_detail,
                     "P3_mid_stack": P3, "P3_detail": p3_detail},
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net.num_params())},
    }
    if not gates_pass:
        metrics["verdicts"]["note"] = ("GATE FAILURE: fix instrument, no verdicts "
                                       "(design §3.4)")
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: depth_survival.png
    ds = list(DEPTHS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    ax = axes[0, 0]
    tfm = [mean_tf[d] for d in ds]
    lo = [stats_all[d]["delta_ci"][0] + mean_sh[d] for d in ds]
    hi = [stats_all[d]["delta_ci"][1] + mean_sh[d] for d in ds]
    ax.fill_between(ds, lo, hi, color="crimson", alpha=0.15,
                    label="TF 95% site-bootstrap band")
    ax.plot(ds, tfm, "o-", color="crimson", label="A-TF (battery state)", lw=2)
    ax.plot(ds, [mean_sh[d] for d in ds], "s--", color="gray", label="A-shuf control")
    ax.plot(ds, base_net_curve, "^-", color="tab:blue", label="e001 base net (destruction)")
    ax.plot(ds, pad_curve, "v-", color="darkorange", label="pad-shifted donors")
    ax.plot(ds, meandonor_curve, ":", color="purple", label="mean-donor (report-only)")
    if d800.get("stats"):
        ax.plot(ds, [d800["stats"][d]["mean_tf"] for d in ds], "*-", color="seagreen",
                ms=9, label="e048_direct800 (own sites)")
    ax.axhline(0.30, color="k", ls=":", lw=1)
    ax.axvspan(5.5, 6.5, color="k", alpha=0.08)
    ax.text(6.02, 0.02, "d6 readout-dominated", rotation=90, fontsize=7, va="bottom")
    if d_star is not None:
        ax.axvline(d_star, color="crimson", ls=":", lw=1)
        ax.annotate(f"d*={d_star}", (d_star, 0.55), color="crimson", fontsize=9)
    ax.set_xlabel("write depth d (0=emb ... 6=final residual)")
    ax.set_ylabel("R1 = P(Z-first) at onset position")
    ax.set_title(f"Depth-survival of the address ({len(sites)} free-run onset sites)")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    if idx_term:
        ax.plot(ds, [stats_term[d]["mean_tf"] if stats_term[d] else float("nan")
                     for d in ds], "o--", color="darkred", label=f"TF terminal (n={n_term})")
    if idx_deep:
        ax.plot(ds, [stats_deep[d]["mean_tf"] if stats_deep[d] else float("nan")
                     for d in ds], "o-", color="crimson", label=f"TF deep (n={n_deep})")
    rev_curve = [float(np.mean(v)) for v in rev_vals]
    ax.plot(ds, rev_curve, "x-", color="teal",
            label="A-rev: site state -> battery (pz .775)")
    ax.axhline(0.30, color="k", ls=":", lw=1)
    ax.set_xlabel("write depth d")
    ax.set_ylabel("R1")
    ax.set_title("Strata + reverse-write symmetry")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    aucs = [stats_all[d]["auc"] for d in ds]
    auci = [[stats_all[d]["auc_ci"][0] for d in ds],
            [stats_all[d]["auc_ci"][1] for d in ds]]
    ax.fill_between(ds, auci[0], auci[1], color="navy", alpha=0.15)
    ax.plot(ds, aucs, "o-", color="navy", label="AUC_d (TF vs shuffled)")
    ax.axhline(0.90, color="navy", ls=":", lw=1, label="0.90 bar")
    ax.set_ylim(0.3, 1.03)
    ax.set_xlabel("write depth d")
    ax.set_ylabel("AUC")
    ax2 = ax.twinx()
    deltas = [stats_all[d]["mean_delta"] for d in ds]
    dci = [[stats_all[d]["delta_ci"][0] for d in ds],
           [stats_all[d]["delta_ci"][1] for d in ds]]
    ax2.fill_between(ds, dci[0], dci[1], color="crimson", alpha=0.12)
    ax2.plot(ds, deltas, "s--", color="crimson", label="meanDelta_d")
    ax2.axhline(0, color="crimson", lw=0.5)
    ax2.set_ylabel("TF - shuffled", color="crimson")
    ax.set_title("Discrimination metric (T028 causal bar)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, 1]
    if r23:
        combos = [(a, d) for d in r23_depths for a in ("base", "oneshot", "held")]
        xs = np.arange(len(combos))
        zw, cols = [], []
        cmap = {"base": "gray", "oneshot": "crimson", "held": "darkred"}
        for a, d in combos:
            if a == "base":
                rows = [r for r in r23 if r["arm"] == "base"]
            else:
                rows = [r for r in r23 if r["arm"] == a and r["depth"] == d]
            zw.append(sum(r["counts"]["z_words"] for r in rows) / max(len(rows), 1))
            cols.append(cmap[a])
        ax.bar(xs, zw, color=cols)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{a}\nd{d}" for a, d in combos], fontsize=8)
        ax.set_ylabel("Z-words per sample (60 chars)")
        ax.set_title(f"R2/R3 downstream ({len(r23_site_idx)} sites x "
                     f"{n_samples} samples; one-shot vs held)")
    fig.suptitle("E055 — the suppression localizer (e048_repro install; "
                 f"P1={P1} P2={P2} P3={P3}; d*={d_star})", fontsize=12)
    fig.tight_layout()
    fig.savefig(rd / "depth_survival.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'depth_survival.png'}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
