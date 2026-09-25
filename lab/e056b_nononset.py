"""E056B — THE CIRCULARITY KILLER for e055's headline (registered in
THINKING.md T033's audit block, 2026-09-25T22:30Z).

T033's open caveat: ALL 21 e055 onset sites are gap-selected (positions
where free-run diverged toward an incumbent-name onset). The registered
follow-up: re-run the SAME R1 depth-survival transplant at ~24 RANDOM
NON-ONSET positions from the e055 trajectories and ask whether the d4
rescue requires the onset POSITION or just the donor STATE
(position-specificity of the rescue).

Design (mirrors lab/e055_suppression.py exactly; no new machinery):
 1. Trajectories: the e055 proper seed batches [5000, 5100, 5200] x 8
    prompts x 350 chars are deterministic (e055 gate G5); regenerate them
    and GATE on exact reproduction of the onset (prompt, t) sets recorded
    in runs/e055/metrics.json sites.table — bit-identical trajectories ==
    "the e055 trajectory caches" (the probe2 cache used different seeds).
 2. Sample 24 random NON-onset positions (8 per seed batch, Random(56001)):
    ordinary mid-text decision positions t in [125, 465] at least 16 chars
    from any ELIZABETH/FLORIZEL/ZEPHYRA span. Record base p(Z) (expected
    at the ~2e-8 trajectory floor per e055 design M4).
 3. At each position x depth d in {0..6}: the SAME arms as e055 R1 —
    6 battery-TF donors (primary4 + replication2), 4 shuffled states
    (Random(25501) family), 6 pad-shifted donors, 1 mean-donor, and the
    e001 base-net twin (donor0); readout P(Z) at the next position.
 4. REGISTERED VERDICT (frozen pre-run):
    - d4-rescue states FAIL at non-onset positions (d* rule does not fire
      at any d<=5: site-mean TF < 0.30)  ->  rescue is POSITION-SPECIFIC,
      consistent with the wpe-130 binding; suppression story unchanged,
      sharpened.
    - they RESCUE anywhere (d* fires at some d<=5)  ->  "suppression" is
      a GLOBAL state property; site-selection circularity moot but the
      localization claim weakens.
    d6 is reported but flagged readout-dominated (trivial by construction
    at any position; e055 addendum).

No training. No edits to NOTES/THINKING/QUEUE/STATE. No git commit.

Run:  python lab/e056b_nononset.py     ->  runs/e056b/
"""
from __future__ import annotations

import json
import os
import random as _random
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
from common import CharCorpus, run_dir, save_json           # noqa: E402
import e043_install as E43                                  # noqa: E402  (find_occ, SPLICE_RNG, jsonable)
import e055_suppression as E55                              # noqa: E402  (the r1 machinery)
import matplotlib.pyplot as plt                             # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
HOST_RE = re.compile(r"(?<![A-Za-z])(ELIZABETH|FLORIZEL)(?![A-Za-z])")
NAME_RE = re.compile(r"(?<![A-Za-z])(ELIZABETH|FLORIZEL|ZEPHYRA)(?![A-Za-z])")
DEPTHS = list(range(7))
MEANINGFUL_BAND = list(range(6))        # d <= 5 (d6 readout-dominated; e055 addendum)
PAD = 10
GEN_TOK = 350
TEMP, TOPK = 0.8, 40
TRAJ_SEEDS = [5000, 5100, 5200]         # the e055 proper batches (protocol.traj_seeds)
N_POS = 24                              # ~24 registered
PER_BATCH = N_POS // len(TRAJ_SEEDS)    # 8 per batch
SAMPLE_RNG = 56001                      # frozen position-sampling seed
EXCL_WIN = 16                           # min chars from any name span
T_MIN, T_MAX = 120 + 5, 120 + GEN_TOK - 5   # ordinary decision positions
BUDGET_S = 900.0
E055_METRICS = E43.REPO / "runs" / "e055" / "metrics.json"

deviations: list[str] = []


def main():
    rd = run_dir("e056b")
    log(f"E056B non-onset circularity killer -> {rd}")

    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    stoi, itos = corpus.stoi, corpus.itos
    zid, eid = stoi["Z"], stoi["E"]
    train_ids = corpus.train
    train_text = "".join(itos[int(i)] for i in train_ids)

    net = E55.load(E43.REPO / "runs" / "checkpoints" / "e048_repro.pt")
    net_base = E55.load(E43.REPO / "runs" / "checkpoints" / "e001.pt")
    log(f"nets loaded (repro/base); params {net.num_params():,}")

    # ---------------- protocol rebuild (e043-frozen; e055-identical)
    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    assert (sum(1 for _, h in install_occ if h == "FLORIZEL"),
            sum(1 for _, h in install_occ if h == "ELIZABETH")) == (19, 41)
    gen_prompts = ([train_text[p - 120: p] for p, _ in install_occ[:4]]
                   + [train_text[p - 120: p] for p, _ in held_occ[:4]])[:8]
    ctx130_i = [train_text[p - PRE: p] for p, _ in install_occ]
    log("protocol rebuilt: install-60 hosts F19/E41, 8 gen prompts")

    # battery stats -> the e055 donor set (primary4 + replication2)
    @torch.no_grad()
    def battery_pz(m, ctxs, bs=30):
        rows = []
        for i in range(0, len(ctxs), bs):
            ids = torch.stack([corpus.encode(c) for c in ctxs[i:i + bs]])
            lg, _ = m(ids)
            pr = F.softmax(lg[:, -1], -1)
            rows.extend(float(pr[k, zid]) for k in range(ids.shape[0]))
        return rows

    bat_repro = battery_pz(net, ctx130_i)
    order = sorted(range(60), key=lambda i: -bat_repro[i])
    primary4 = list(dict.fromkeys(order[:2]
                                  + [i for i in order[2:] if install_occ[i][1] == "FLORIZEL"][:2]))[:4]
    replication2 = [i for i in order if i not in primary4][:2]
    donor_idx = primary4 + replication2
    assert donor_idx == [7, 29, 43, 15, 36, 26], f"donor set drift: {donor_idx}"

    donors = []
    for i in donor_idx:
        xs, lg = E55.states_forward(net, corpus.encode(ctx130_i[i]))
        donors.append({"ctx_i": i, "host": install_occ[i][1],
                       "p_z": float(F.softmax(lg[0, -1], -1)[zid]),
                       "states": [x[0, -1].clone() for x in xs]})
    log("donors (e055 set): " + " ".join(f"ctx{d['ctx_i']}:{d['p_z']:.3f}" for d in donors))

    # shuffled + pad-shifted families (e055-identical construction)
    _s = _random.Random(E55.SHUF_SEED)
    shuf_ctxs = []
    while len(shuf_ctxs) < E55.N_SHUF:
        q = _s.randrange(PRE + 1, len(train_ids) - 1)
        shuf_ctxs.append(train_text[q - PRE: q])
    shuf = []
    for c in shuf_ctxs:
        xs, lg = E55.states_forward(net, corpus.encode(c))
        shuf.append({"p_z": float(F.softmax(lg[0, -1], -1)[zid]),
                     "states": [x[0, -1].clone() for x in xs]})
    pad_donors = []
    for i in donor_idx:
        xs, _ = E55.states_forward(net, corpus.encode(" " * PAD + ctx130_i[i]))
        pad_donors.append({"ctx_i": i, "states": [x[0, -1].clone() for x in xs]})
    mean_states = [torch.stack([d["states"][di] for d in donors]).mean(0) for di in DEPTHS]
    log("shuffled own p(Z) max "
        f"{max(d['p_z'] for d in shuf):.2e} (e055 G4 family)")

    # ---------------- 1. regenerate the e055 trajectories + identity GATE
    prompt_ids = torch.stack([corpus.encode(p) for p in gen_prompts])
    trajs = []            # per batch: list of full texts (prompt + cont)
    onset_sets = []       # per batch: sorted [(prompt, t)]
    for seed in TRAJ_SEEDS:
        torch.manual_seed(seed)
        out = E55.gen_batch(net, prompt_ids, GEN_TOK)
        conts = [corpus.decode(r.tolist()) for r in out]
        trajs.append([gen_prompts[i] + conts[i] for i in range(len(gen_prompts))])
        onsets = []
        for i, cont in enumerate(conts):
            for m in HOST_RE.finditer(cont):
                onsets.append((i, len(gen_prompts[i]) + m.start()))
        onset_sets.append(sorted(onsets))

    with open(E055_METRICS, encoding="utf-8") as f:
        e055 = json.load(f)
    e055_onsets = {}
    for row in e055["sites"]["table"]:
        e055_onsets.setdefault(row["seed_batch"], []).append((row["prompt"], row["t"]))
    gate_traj = all(onset_sets[b] == sorted(e055_onsets.get(b, []))
                    for b in range(len(TRAJ_SEEDS)))
    G_traj = {"rule": "regenerated onset (prompt,t) sets == runs/e055 metrics "
                      "sites.table for all 3 seed batches (bit-identical "
                      "trajectories; e055 G5 determinism)",
              "regenerated": {str(b): onset_sets[b] for b in range(len(TRAJ_SEEDS))},
              "e055": {str(b): sorted(e055_onsets.get(b, [])) for b in range(len(TRAJ_SEEDS))},
              "pass": bool(gate_traj)}
    log(f"G-traj identity vs runs/e055: {'PASS' if gate_traj else 'FAIL'} "
        f"({sum(len(v) for v in onset_sets)} onsets regenerated)")
    if not gate_traj:
        deviations.append("trajectory identity gate FAILED — positions are "
                          "from e055-protocol regenerated trajectories, not "
                          "verified bit-identical ones")

    # ---------------- 2. sample 24 random NON-onset positions
    def valid_positions(text: str, prompt_len: int):
        """Decision positions t (crop=text[:t]) that are ordinary mid-text:
        inside [T_MIN, T_MAX] and >= EXCL_WIN chars from any name span."""
        spans = [(m.start(), m.end()) for m in NAME_RE.finditer(text)]
        out = []
        for t in range(T_MIN, min(T_MAX + 1, len(text))):
            if all(t < s - EXCL_WIN or t > e + EXCL_WIN for s, e in spans):
                out.append(t)
        return out, spans

    srng = _random.Random(SAMPLE_RNG)
    positions = []
    for b, text_batch in enumerate(trajs):
        cand = []
        for i, text in enumerate(text_batch):
            vp, spans = valid_positions(text, len(gen_prompts[i]))
            cand.extend((i, t) for t in vp)
        if len(cand) < PER_BATCH:
            raise RuntimeError(f"batch {b}: only {len(cand)} valid non-onset positions")
        for i, t in srng.sample(cand, PER_BATCH):
            positions.append({"batch": b, "prompt": i, "t": int(t),
                              "ctx": text_batch[i][:t]})
    assert len(positions) == N_POS
    log(f"sampled {N_POS} non-onset positions "
        f"(rng {SAMPLE_RNG}, exclusion window +-{EXCL_WIN} chars around any "
        f"ELIZABETH/FLORIZEL/ZEPHYRA span)")

    # ---------------- 3. per-position instruments + the SAME R1 arms
    for p in positions:
        crop = corpus.encode(p["ctx"])[-net.cfg.block_size:]
        dpos = len(crop) - 1
        with torch.no_grad():
            lg, _ = net(crop.unsqueeze(0))
        lgl = lg[0, -1]
        pr = F.softmax(lgl, -1)
        xs, _ = E55.states_forward(net, crop)
        p.update({"crop_len": int(len(crop)), "dpos": int(dpos),
                  "in_model_pos": int(dpos),
                  "stratum": "deep" if dpos == net.cfg.block_size - 1 else "mid",
                  "base_p_z": float(pr[zid]), "base_p_e": float(pr[eid]),
                  "base_rank_z": int((lgl > lgl[zid]).sum()) + 1,
                  "base_argmax": itos[int(lgl.argmax())],
                  "next_char": trajs[p["batch"]][p["prompt"]][p["t"]]})
        p["own_states"] = [x[0, dpos].clone() for x in xs]
        p["crop"] = crop

    # instrument: G1 self-patch identity at 2 sampled positions
    g1_max = 0.0
    for p in positions[:2]:
        lg0, _ = net(p["crop"].unsqueeze(0))
        for d in (0, 3, 6):
            lg = E55.patch_logits_batch(net, p["crop"].unsqueeze(0), p["dpos"], d,
                                        p["own_states"][d].unsqueeze(0))
            g1_max = max(g1_max, float((lg[0, -1] - lg0[0, -1]).abs().max()))
    G1 = {"max_abs_dlogit": g1_max, "gate": 1e-4, "pass": bool(g1_max < 1e-4)}
    log(f"G1 self-patch identity max|dlogit| {g1_max:.2e}: "
        f"{'PASS' if G1['pass'] else 'FAIL'}")
    gates_pass = G1["pass"] and gate_traj

    tf_vals = [[[] for _ in DEPTHS] for _ in positions]
    shuf_vals = [[[] for _ in DEPTHS] for _ in positions]
    pad_vals = [[[] for _ in DEPTHS] for _ in positions]
    meandonor_vals = [[[] for _ in DEPTHS] for _ in positions]
    base_net_vals = [[[] for _ in DEPTHS] for _ in positions]

    for pi, p in enumerate(positions):
        crop1 = p["crop"].unsqueeze(0)
        for d in DEPTHS:
            B = len(donors) + len(shuf) + len(pad_donors) + 1     # 6+4+6+1
            states = torch.stack([dnr["states"][d] for dnr in donors]
                                 + [x["states"][d] for x in shuf]
                                 + [x["states"][d] for x in pad_donors]
                                 + [mean_states[d]])
            lg = E55.patch_logits_batch(net, crop1.repeat(B, 1), p["dpos"], d, states)
            pz = E55.pz_next(lg, zid)
            tf_vals[pi][d] = [float(v) for v in pz[:len(donors)]]
            shuf_vals[pi][d] = [float(v) for v in pz[len(donors):len(donors) + len(shuf)]]
            pad_vals[pi][d] = [float(v) for v in
                               pz[len(donors) + len(shuf):len(donors) + len(shuf) + len(pad_donors)]]
            meandonor_vals[pi][d] = [float(pz[-1])]
            lgb = E55.patch_logits_batch(net_base, crop1, p["dpos"], d,
                                         donors[0]["states"][d].unsqueeze(0))
            base_net_vals[pi][d] = [float(E55.pz_next(lgb, zid))]
        log(f"  pos b{p['batch']} p{p['prompt']} t={p['t']:3d} pos{p['dpos']:3d} "
            f"{p['stratum']:4s} base p(Z) {p['base_p_z']:.2e} r{p['base_rank_z']} "
            f"argmax {p['base_argmax']} | TF d4 "
            f"{np.mean(tf_vals[pi][4]):.3f} d5 {np.mean(tf_vals[pi][5]):.3f} "
            f"d6 {np.mean(tf_vals[pi][6]):.3f} shuf4 {np.mean(shuf_vals[pi][4]):.1e}")

    # ---------------- 4. statistics (e055 machinery)
    stats = [E55.depth_stats([np.array(tf_vals[i][d]) for i in range(len(positions))],
                             [np.array(shuf_vals[i][d]) for i in range(len(positions))])
             for d in DEPTHS]
    pad_curve = [float(np.mean([pad_vals[i][d] for i in range(len(positions))]))
                 for d in DEPTHS]
    meandonor_curve = [float(np.mean([meandonor_vals[i][d] for i in range(len(positions))]))
                       for d in DEPTHS]
    base_net_curve = [float(np.mean([base_net_vals[i][d] for i in range(len(positions))]))
                      for d in DEPTHS]
    base_site_curve = [float(np.mean([p["base_p_z"] for p in positions]))]

    idx_deep = [i for i, p in enumerate(positions) if p["stratum"] == "deep"]
    idx_mid = [i for i, p in enumerate(positions) if p["stratum"] == "mid"]
    stats_deep = [E55.depth_stats([np.array(tf_vals[i][d]) for i in idx_deep],
                                  [np.array(shuf_vals[i][d]) for i in idx_deep])
                  for d in DEPTHS] if idx_deep else None
    stats_mid = [E55.depth_stats([np.array(tf_vals[i][d]) for i in idx_mid],
                                 [np.array(shuf_vals[i][d]) for i in idx_mid])
                 for d in DEPTHS] if idx_mid else None

    log("R1 depth curves at NON-onset positions (vs e055 onset-site TF):")
    e055_tf = {int(k): v for k, v in e055["r1_curves"]["tf_mean"].items()}
    e055_bn = {int(k): v for k, v in e055["r1_curves"]["base_net_mean"].items()}
    for d in DEPTHS:
        st = stats[d]
        log(f"  d{d}: TF {st['mean_tf']:.4f} (e055 onset {e055_tf[d]:.3f}) "
            f"shuf {st['mean_shuf']:.2e} AUC {st['auc']:.3f} "
            f"CI[{st['auc_ci'][0]:.3f},{st['auc_ci'][1]:.3f}] "
            f"delta {st['mean_delta']:.4f} CI[{st['delta_ci'][0]:.4f},"
            f"{st['delta_ci'][1]:.4f}] | pad {pad_curve[d]:.3f} "
            f"meandon {meandonor_curve[d]:.3f} basenet {base_net_curve[d]:.2e}")

    # d* rule (e055-identical): shallowest d<=5 with site-mean TF >= .30 AND AUC >= .90
    d_star = None
    for d in MEANINGFUL_BAND:
        if stats[d]["mean_tf"] is not None and stats[d]["mean_tf"] >= 0.30 \
                and stats[d]["auc"] >= 0.90:
            d_star = d
            break
    d_best = max(MEANINGFUL_BAND, key=lambda d: stats[d]["mean_tf"] or 0.0)

    # per-position rescue stats at the e055 headline depths (d4, d5) + d6
    per_pos = {}
    for d in (4, 5, 6):
        means = [float(np.mean(tf_vals[i][d])) for i in range(len(positions))]
        maxes = [float(np.max(tf_vals[i][d])) for i in range(len(positions))]
        per_pos[d] = {"donor_mean_by_pos": means,
                      "n_pos_mean_ge_030": sum(m >= 0.30 for m in means),
                      "n_pos_max_ge_030": sum(m >= 0.30 for m in maxes),
                      "min": float(np.min(means)), "median": float(np.median(means)),
                      "max": float(np.max(means))}

    # ---------------- REGISTERED VERDICT (frozen in the module docstring)
    fires = d_star is not None
    if not gates_pass:
        verdict = "NO VERDICT (gate failure)"
    elif fires:
        verdict = (f"GLOBAL STATE PROPERTY — the d{d_star} rescue fires at random "
                   f"non-onset positions (site-mean {stats[d_star]['mean_tf']:.3f} "
                   f">= .30, AUC {stats[d_star]['auc']:.3f}): the donor state "
                   "rescues anywhere; site-selection circularity moot, the "
                   "onset-position localization claim weakens")
    else:
        verdict = (f"POSITION-SPECIFIC — no d<=5 reaches .30 at non-onset "
                   f"positions (best d{d_best} = {stats[d_best]['mean_tf']:.4f} "
                   f"vs e055 onset d4 0.374): the d4/d5 rescue requires the "
                   "onset position; consistent with the wpe-130 binding — the "
                   "suppression story stands, sharpened")
    log("=" * 72)
    log(f"VERDICT: {verdict}")
    log(f"per-position d4: {per_pos[4]['n_pos_mean_ge_030']}/{N_POS} positions "
        f"with donor-mean >= .30 (median {per_pos[4]['median']:.4f}, "
        f"max {per_pos[4]['max']:.3f})")

    # ---------------- outputs
    pos_out = [{k: p[k] for k in ("batch", "prompt", "t", "crop_len", "dpos",
                                  "in_model_pos", "stratum", "base_p_z", "base_p_e",
                                  "base_rank_z", "base_argmax", "next_char")}
               for p in positions]
    metrics = {
        "experiment": "e056b_nononset",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design": "T033 audit-block registered follow-up (THINKING.md); mirrors "
                  "lab/e055_suppression.py R1 machinery exactly",
        "question": "does the e055 d4/d5 TF-state rescue require the onset "
                    "POSITION or just the donor STATE (position-specificity; "
                    "kills or confirms the gap-selection circularity)",
        "net": "runs/checkpoints/e048_repro.pt (+ e001.pt base twin)",
        "protocol": {"traj_seeds": TRAJ_SEEDS, "gen_toks": GEN_TOK,
                     "temp": TEMP, "top_k": TOPK,
                     "sample_rng": SAMPLE_RNG, "n_positions": N_POS,
                     "per_batch": PER_BATCH, "exclusion_window": EXCL_WIN,
                     "t_range": [T_MIN, T_MAX],
                     "excluded_names": ["ELIZABETH", "FLORIZEL", "ZEPHYRA"]},
        "gates": {"G_traj_identity": G_traj, "G1_selfpatch": G1,
                  "all_pass": gates_pass},
        "positions": {"n": N_POS,
                      "strata": {"deep(pos255)": len(idx_deep), "mid": len(idx_mid)},
                      "base_pz": {"median": float(np.median([p["base_p_z"] for p in positions])),
                                  "p90": float(np.percentile([p["base_p_z"] for p in positions], 90)),
                                  "max": float(np.max([p["base_p_z"] for p in positions]))},
                      "table": pos_out},
        "donors": [{"ctx_i": d["ctx_i"], "host": d["host"], "p_z": d["p_z"]}
                   for d in donors],
        "r1_curves": {
            "tf_mean": {d: stats[d]["mean_tf"] for d in DEPTHS},
            "shuf_mean": {d: stats[d]["mean_shuf"] for d in DEPTHS},
            "pad_mean": {d: pad_curve[d] for d in DEPTHS},
            "mean_donor_mean": {d: meandonor_curve[d] for d in DEPTHS},
            "base_net_mean": {d: base_net_curve[d] for d in DEPTHS},
            "base_site_mean": base_site_curve,
            "e055_onset_tf_mean": e055_tf,
            "e055_onset_base_net_mean": e055_bn,
            "per_pos_tf": {f"pos{pi}_b{p['batch']}p{p['prompt']}_t{p['t']}":
                            {d: tf_vals[pi][d] for d in DEPTHS}
                            for pi, p in enumerate(positions)},
            "per_pos_shuf": {f"pos{pi}_b{p['batch']}p{p['prompt']}_t{p['t']}":
                             {d: shuf_vals[pi][d] for d in DEPTHS}
                             for pi, p in enumerate(positions)},
        },
        "stats_by_depth": {"all": {d: stats[d] for d in DEPTHS},
                           "deep": {d: stats_deep[d] for d in DEPTHS} if stats_deep else None,
                           "mid": {d: stats_mid[d] for d in DEPTHS} if stats_mid else None},
        "d_star_nononset": d_star, "d_best_nononset": d_best,
        "per_position_rescue": per_pos,
        "ratio_to_e055": {d: {"nononset_tf": stats[d]["mean_tf"],
                              "onset_tf": e055_tf[d],
                              "ratio": (stats[d]["mean_tf"] / e055_tf[d])
                              if e055_tf[d] else None}
                          for d in DEPTHS},
        "verdict": {"rule": "GLOBAL if the e055 d* rule (site-mean TF >= .30 AND "
                            "AUC >= .90 at some d<=5) fires at non-onset "
                            "positions; POSITION-SPECIFIC otherwise. d6 "
                            "readout-dominated (trivial), reported separately.",
                    "verdict": verdict,
                    "d_star_fired": fires,
                    "d4_nononset_mean": stats[4]["mean_tf"],
                    "d4_onset_mean_e055": e055_tf[4],
                    "d4_ratio": (stats[4]["mean_tf"] / e055_tf[4]) if e055_tf[4] else None,
                    "d6_trivial_check": {"nononset_d6": stats[6]["mean_tf"],
                                         "donor_pz_mean": float(np.mean([d["p_z"] for d in donors]))}},
        "trims": [], "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net.num_params())},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot
    ds = list(DEPTHS)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    ax = axes[0]
    ax.plot(ds, [stats[d]["mean_tf"] for d in ds], "o-", color="crimson",
            lw=2, label="NON-onset TF (24 random positions)")
    lo = [stats[d]["delta_ci"][0] + stats[d]["mean_shuf"] for d in ds]
    hi = [stats[d]["delta_ci"][1] + stats[d]["mean_shuf"] for d in ds]
    ax.fill_between(ds, lo, hi, color="crimson", alpha=0.12,
                    label="TF 95% site-bootstrap band")
    ax.plot(ds, [stats[d]["mean_shuf"] for d in ds], "s--", color="gray",
            label="shuffled control")
    ax.plot(ds, [e055_tf[d] for d in ds], "k--", lw=2, marker="o", mfc="none",
            label="e055 ONSET sites TF (21 sites)")
    ax.plot(ds, base_net_curve, "^-", color="tab:blue", label="e001 base twin")
    ax.plot(ds, pad_curve, "v-", color="darkorange", label="pad-shifted donors")
    ax.plot(ds, meandonor_curve, ":", color="purple", label="mean-donor")
    ax.axhline(0.30, color="k", ls=":", lw=1)
    ax.axvspan(5.5, 6.5, color="k", alpha=0.08)
    ax.text(6.02, 0.02, "d6 readout-dominated", rotation=90, fontsize=7, va="bottom")
    ax.axvline(4, color="crimson", ls=":", lw=1)
    ax.annotate("e055 d* = 4", (4, 0.55), color="crimson", fontsize=9, ha="center")
    ax.set_xlabel("write depth d (0=emb ... 6=final residual)")
    ax.set_ylabel("R1 = P(Z-first) at next position")
    ax.set_title("Depth-survival: non-onset positions vs e055 onset sites")
    ax.legend(fontsize=7)

    ax = axes[1]
    d4_by_pos = [float(np.mean(tf_vals[i][4])) for i in range(len(positions))]
    d5_by_pos = [float(np.mean(tf_vals[i][5])) for i in range(len(positions))]
    xs = np.arange(len(positions))
    ax.bar(xs - 0.2, d4_by_pos, 0.4, color="crimson", label="d4 donor-mean")
    ax.bar(xs + 0.2, d5_by_pos, 0.4, color="darkred", label="d5 donor-mean")
    ax.axhline(0.30, color="k", ls=":", lw=1, label="0.30 rescue bar")
    ax.axhline(e055_tf[4], color="crimson", ls="--", lw=1,
               label=f"e055 onset d4 mean {e055_tf[4]:.3f}")
    ax.set_xlabel("non-onset position index")
    ax.set_ylabel("TF donor-mean p(Z)")
    ax.set_title(f"Per-position rescue (d4: {per_pos[4]['n_pos_mean_ge_030']}/24 "
                 f">= .30; d5: {per_pos[5]['n_pos_mean_ge_030']}/24)")
    ax.legend(fontsize=8)

    ax = axes[2]
    aucs = [stats[d]["auc"] for d in ds]
    ax.fill_between(ds, [stats[d]["auc_ci"][0] for d in ds],
                    [stats[d]["auc_ci"][1] for d in ds], color="navy", alpha=0.15)
    ax.plot(ds, aucs, "o-", color="navy", label="AUC_d (TF vs shuffled)")
    ax.axhline(0.90, color="navy", ls=":", lw=1, label="0.90 bar")
    ax.set_ylim(0.3, 1.03)
    ax.set_xlabel("write depth d")
    ax.set_ylabel("AUC")
    ax2 = ax.twinx()
    ax2.fill_between(ds, [stats[d]["delta_ci"][0] for d in ds],
                     [stats[d]["delta_ci"][1] for d in ds], color="crimson", alpha=0.12)
    ax2.plot(ds, [stats[d]["mean_delta"] for d in ds], "s--", color="crimson",
             label="meanDelta_d")
    ax2.axhline(0, color="crimson", lw=0.5)
    ax2.set_ylabel("TF - shuffled", color="crimson")
    ax.set_title("Discrimination at non-onset positions")
    ax.legend(fontsize=8, loc="center right")

    fig.suptitle("E056B — non-onset circularity killer "
                 f"({'GLOBAL' if fires else 'POSITION-SPECIFIC'}; "
                 f"non-onset d4 {stats[4]['mean_tf']:.4f} vs onset {e055_tf[4]:.3f})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(rd / "nononset_depth.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'nononset_depth.png'}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
