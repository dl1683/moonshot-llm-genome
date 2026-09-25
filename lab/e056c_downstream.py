"""E056C — e056b+R2: the R14-REGISTERED DISCRIMINATOR for the claim-split's
second leg (REVIEWS.md R14, 2026-09-25T23:00Z; THINKING.md T034 flag).

T034's open slot: e056b's position-general rescue is R1-ONLY (next-token
p(Z) at the write position). A "loud Z-logit paste" — a d4 write that
momentarily cranks the Z logit but installs nothing the model's own
dynamics reuse — produces the SAME R1 curve. The discriminator is the
DOWNSTREAM readout at the same 24 non-onset positions.

Design (frozen pre-run):
 1. Rebuild the e056b world EXACTLY (protocol -> donors [7,29,43,15,36,26]
    -> shuffled family -> the 3 seed-batch trajectories -> Random(56001)
    position sampling). Gates: trajectory identity vs runs/e055 onset
    sets, position identity vs runs/e056b positions.table, donor-TF
    cross-check vs e056b per_pos_tf at d4, G1 self-patch identity,
    G5 continuation determinism.
 2. At each of the 24 non-onset positions: ONE-SHOT d4 donor write at the
    decision position, then FREE-RUN continue 60 chars (T=0.8, top-k 40;
    e055 R2 semantics). Readouts:
    (a) ZEPHYRA-word count in the continuation — full word or prefix
        >= 3 chars (ZEP|ZEPH|ZEPHY|ZEPHYR|ZEPHYRA), standalone word;
        split offset-0 (mechanical completion of the written first char)
        vs later offsets (model re-instantiates ZEPHYRA on its own).
    (b) argmax-flip rate at +1 (patched vs base next-char argmax; and
        flip-TO-Z rate), per write arm, deterministic (no sampling).
    (c) p(Z) persistence: p(Z) at +1,+2,+3,+5,+10 under teacher-forced
        replay of the generated text (recorded at generation time — each
        step's distribution is conditioned on the emitted prefix; the
        write touches only step 0's distribution, so +2.. is the model's
        own dynamics with the write gone).
    Arms (rows batched per position; step-0 patch has per-row states):
      base      x4  no write (spontaneous-rate control)
      donor0    x4  d4 write, donor ctx7 (e055 R2-identical primary)
      meandonor x4  d4 write, mean of the 6 donors (the relay direction)
      bestdonor x4  d4 write, per-position argmax-TF donor (optimistic;
                    selected on the R1 readout itself — flagged)
      shuf      x4  d4 write, the 4 Random(25501) shuffled states (1 row
                    each) — the knowledge-specificity control
    d5 donor0 is reported as a +1 instrument only (continuations not run;
    d4 is the registered write depth).
 3. REGISTERED VERDICT (frozen, R14 wording):
    - recurrent Z-words >= 2 in a continuation (prefix-3 count) under a
      donor write, with all control (shuf+base) continuations below that
      ->  ADDRESS INSTALLED (leg-2 holds: a portable address).
    - Z blip at +1 that reverts (p(Z) at +2 back at the control floor)
      with zero downstream ZEPHYRA words beyond offset-0 completions
      ->  LOUD LOGIT PASTE (leg-2 reframes as transient state injection).

No training. No edits to NOTES/THINKING/QUEUE/STATE. No git commit.

Run:  python lab/e056c_downstream.py     ->  runs/e056c/
"""
from __future__ import annotations

import json
import os
import random as _random
import re
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")          # CPU experiment

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402

import common                                               # noqa: E402
common.DEVICE = "cpu"
from common import CharCorpus, run_dir, save_json           # noqa: E402
import e043_install as E43                                  # noqa: E402  (find_occ, SPLICE_RNG, jsonable)
import e055_suppression as E55                              # noqa: E402  (the machinery)

# Host-specific: the e055/e056b module import pins 16 torch threads, which is
# ~3x slower than 4 on this machine. Set AFTER imports; trajectory identity
# at 4 threads is re-gated below (pre-flight verified; gate is load-bearing).
torch.set_num_threads(4)

import matplotlib.pyplot as plt                             # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
HOST_RE = re.compile(r"(?<![A-Za-z])(ELIZABETH|FLORIZEL)(?![A-Za-z])")
NAME_RE = re.compile(r"(?<![A-Za-z])(ELIZABETH|FLORIZEL|ZEPHYRA)(?![A-Za-z])")
ZEPH_RE = re.compile(r"(?<![A-Za-z])ZEPHYRA(?![A-Za-z])")
ZEP3_RE = re.compile(r"(?<![A-Za-z])Z(?:EPHYRA|EPHYR|EPHY|EPH|EP)(?![A-Za-z])")
ZW_RE = re.compile(r"(?<![A-Za-z])Z[A-Za-z]*")
PAD = 10
GEN_TOK = 350
TEMP, TOPK = 0.8, 40
TRAJ_SEEDS = [5000, 5100, 5200]         # the e055/e056b proper batches
N_POS = 24
PER_BATCH = N_POS // len(TRAJ_SEEDS)    # 8
SAMPLE_RNG = 56001                      # e056b's frozen position-sampling seed
EXCL_WIN = 16
T_MIN, T_MAX = 120 + 5, 120 + GEN_TOK - 5
DEPTH_W = 4                             # the registered write depth (e055 d*)
CONT_CHARS = 60                         # "60+ chars" (e055 R2 protocol)
N_SAMPLES = 4                           # samples per arm per position
GEN_SEED = 91600                        # frozen: seed = GEN_SEED + 101 * pos_idx
OFFSETS = [1, 2, 3, 5, 10]              # p(Z) persistence readouts (+k)
REVERT_FOLD = 10.0                      # "reverted" = pz(+2) within 10x ctrl floor
BUDGET_S = 900.0                        # soft target (~10 min + slack)
TRIM_S = 640.0                          # drop optional arms past this
E055_METRICS = E43.REPO / "runs" / "e055" / "metrics.json"
E056B_METRICS = E43.REPO / "runs" / "e056b" / "metrics.json"

trims: list[str] = []
deviations: list[str] = [
    "torch threads = 4 (host-specific; e055/e056b pinned 16 at import, "
    "~3x slower here). Trajectory identity at 4 threads is gated below "
    "(G-traj) and donor TF cross-checked vs runs/e056b (G-xcheck); any "
    "numeric drift trips the gates -> NO VERDICT.",
    "Continuation RNG: one frozen seed per position (GEN_SEED + 101*idx) "
    "consumed jointly by all arms' rows (e055 R2 seeded per site x depth "
    "x arm instead). Determinism itself is gated (G-det).",
    "bestdonor arm is selected on the R1 readout it then continues from "
    "(optimistic upper bound; flagged in metrics).",
]


# ------------------------------------------------------------------ counting

def count_z(text: str) -> dict:
    p3 = [m.start() for m in ZEP3_RE.finditer(text)]
    return {
        "text": text,
        "first_char": text[0],
        "zephyra_full": len(ZEPH_RE.findall(text)),
        "zep3": len(p3),
        "zep3_offset0": sum(1 for s in p3 if s == 0),
        "zep3_later": sum(1 for s in p3 if s >= 1),
        "z_words": len(ZW_RE.findall(text)),
        "recurrent": bool(len(p3) >= 2),          # registered rule
        "later_z": bool(any(s >= 1 for s in p3)), # honest split: beyond char 0
    }


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e056c")
    log(f"E056C non-onset downstream discriminator -> {rd}")

    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids = corpus.train
    train_text = "".join(itos[int(i)] for i in train_ids)

    net = E55.load(E43.REPO / "runs" / "checkpoints" / "e048_repro.pt")
    log(f"net loaded (e048_repro); params {net.num_params():,}")

    # ---------------- protocol rebuild (e043-frozen; e055/e056b-identical)
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

    @torch.no_grad()
    def battery_pz(m, ctxs, bs=30):
        rows = []
        for i in range(0, len(ctxs), bs):
            ids = torch.stack([corpus.encode(c) for c in ctxs[i:i + bs]])
            lg, _ = m(ids)
            rows.extend(float(F.softmax(lg[:, -1], -1)[k, zid])
                        for k in range(ids.shape[0]))
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
    mean_states = [torch.stack([d["states"][di] for d in donors]).mean(0)
                   for di in range(7)]

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
    log("donors " + " ".join(f"ctx{d['ctx_i']}:{d['p_z']:.3f}" for d in donors)
        + f" | shuffled own p(Z) max {max(d['p_z'] for d in shuf):.2e}")

    # ---------------- 1. regenerate the e055/e056b trajectories + gates
    prompt_ids = torch.stack([corpus.encode(p) for p in gen_prompts])
    trajs, onset_sets = [], []
    for seed in TRAJ_SEEDS:
        torch.manual_seed(seed)
        out = E55.gen_batch(net, prompt_ids, GEN_TOK)
        conts = [corpus.decode(r.tolist()) for r in out]
        trajs.append([gen_prompts[i] + conts[i] for i in range(len(gen_prompts))])
        onset_sets.append(sorted((i, len(gen_prompts[i]) + m.start())
                                 for i, cont in enumerate(conts)
                                 for m in HOST_RE.finditer(cont)))

    with open(E055_METRICS, encoding="utf-8") as f:
        e055 = json.load(f)
    e055_onsets = {}
    for row in e055["sites"]["table"]:
        e055_onsets.setdefault(row["seed_batch"], []).append((row["prompt"], row["t"]))
    gate_traj = all(onset_sets[b] == sorted(e055_onsets.get(b, []))
                    for b in range(len(TRAJ_SEEDS)))
    log(f"G-traj identity vs runs/e055 @4 threads: "
        f"{'PASS' if gate_traj else 'FAIL'}")

    with open(E056B_METRICS, encoding="utf-8") as f:
        e56b = json.load(f)

    # ---------------- 2. the same 24 non-onset positions (e056b rule)
    def valid_positions(text: str):
        spans = [(m.start(), m.end()) for m in NAME_RE.finditer(text)]
        out = []
        for t in range(T_MIN, min(T_MAX + 1, len(text))):
            if all(t < s - EXCL_WIN or t > e + EXCL_WIN for s, e in spans):
                out.append(t)
        return out

    srng = _random.Random(SAMPLE_RNG)
    positions = []
    for b, text_batch in enumerate(trajs):
        cand = []
        for i, text in enumerate(text_batch):
            cand.extend((i, t) for t in valid_positions(text))
        for i, t in srng.sample(cand, PER_BATCH):
            positions.append({"batch": b, "prompt": i, "t": int(t),
                              "ctx": text_batch[i][:t]})

    e56b_rows = [(r["batch"], r["prompt"], r["t"]) for r in e56b["positions"]["table"]]
    my_rows = [(p["batch"], p["prompt"], p["t"]) for p in positions]
    gate_pos = my_rows == e56b_rows
    log(f"G-pos identity vs runs/e056b positions.table: "
        f"{'PASS' if gate_pos else 'FAIL'} ({len(positions)} positions)")

    for p in positions:
        crop = corpus.encode(p["ctx"])[-net.cfg.block_size:]
        dpos = len(crop) - 1
        with torch.no_grad():
            lg, _ = net(crop.unsqueeze(0))
        lgl = lg[0, -1]
        pr = F.softmax(lgl, -1)
        p.update({"crop_len": int(len(crop)), "dpos": int(dpos),
                  "stratum": "deep" if dpos == net.cfg.block_size - 1 else "mid",
                  "base_p_z": float(pr[zid]), "base_argmax": itos[int(lgl.argmax())],
                  "crop": crop})

    # G1 self-patch identity
    g1_max = 0.0
    for p in positions[:2]:
        with torch.no_grad():
            lg0, _ = net(p["crop"].unsqueeze(0))
        xs, _ = E55.states_forward(net, p["crop"])
        for d in (0, 3, 6):
            lg = E55.patch_logits_batch(net, p["crop"].unsqueeze(0), p["dpos"], d,
                                        xs[d][0, p["dpos"]].unsqueeze(0))
            g1_max = max(g1_max, float((lg[0, -1] - lg0[0, -1]).abs().max()))
    G1 = {"max_abs_dlogit": g1_max, "gate": 1e-4, "pass": bool(g1_max < 1e-4)}
    log(f"G1 self-patch identity max|dlogit| {g1_max:.2e}: "
        f"{'PASS' if G1['pass'] else 'FAIL'}")

    # ---------------- 3. instruments: +1 readouts + argmax flips (exact)
    log("instruments: d4 TF per donor (best selection + e056b cross-check), "
        "+1 readouts and argmax flips per write arm, donor0 d5")
    tf_x_maxdiff = 0.0
    for pi, p in enumerate(positions):
        crop1 = p["crop"].unsqueeze(0)
        # all 6 donors at d4 -> best-donor selection + cross-check vs e056b
        st6 = torch.stack([d["states"][DEPTH_W] for d in donors])
        lg6 = E55.patch_logits_batch(net, crop1.repeat(6, 1), p["dpos"], DEPTH_W, st6)
        pr6 = F.softmax(lg6[:, -1], -1)[:, zid]
        tf6 = [float(v) for v in pr6]
        best_i = int(torch.tensor(tf6).argmax())
        p["tf6_d4"] = tf6
        p["best_donor_i"] = best_i
        p["donor0_pz1"] = tf6[0]
        key = f"pos{pi}_b{p['batch']}p{p['prompt']}_t{p['t']}"
        ref = float(e56b["r1_curves"]["per_pos_tf"][key]["4"][0])
        tf_x_maxdiff = max(tf_x_maxdiff, abs(tf6[0] - ref))
        # write-arm +1 readouts at d4: donor0, meandonor, bestdonor, 4x shuf
        arms = [("donor0", donors[0]["states"][DEPTH_W]),
                ("meandonor", mean_states[DEPTH_W]),
                ("bestdonor", donors[best_i]["states"][DEPTH_W])] + \
               [(f"shuf{k}", shuf[k]["states"][DEPTH_W]) for k in range(E55.N_SHUF)]
        st = torch.stack([s for _, s in arms])
        lga = E55.patch_logits_batch(net, crop1.repeat(len(arms), 1),
                                     p["dpos"], DEPTH_W, st)
        lgla = lga[:, -1]
        p["write1"] = {name: {"p_z": float(F.softmax(lgla[k], -1)[zid]),
                              "argmax": itos[int(lgla[k].argmax())]}
                       for k, (name, _) in enumerate(arms)}
        # donor0 at d5 (report-only instrument)
        lg5 = E55.patch_logits_batch(net, crop1, p["dpos"], 5,
                                     donors[0]["states"][5].unsqueeze(0))
        p["donor0_d5_pz1"] = float(F.softmax(lg5[0, -1], -1)[zid])
        p["donor0_d5_argmax"] = itos[int(lg5[0, -1].argmax())]

    G_x = {"rule": "donor0 d4 TF per position vs runs/e056b r1_curves.per_pos_tf",
           "max_abs_diff": tf_x_maxdiff, "gate": 5e-4, "pass": bool(tf_x_maxdiff < 5e-4)}
    log(f"G-xcheck vs e056b donor0 TF(d4): max|diff| {tf_x_maxdiff:.2e}: "
        f"{'PASS' if G_x['pass'] else 'FAIL'}")

    flips = {}
    for name in ["donor0", "meandonor", "bestdonor", "shuf0", "shuf1", "shuf2", "shuf3"]:
        fl = [p["write1"][name]["argmax"] != p["base_argmax"] for p in positions]
        fz = [p["write1"][name]["argmax"] == "Z" for p in positions]
        flips[name] = {"flip_rate": float(np.mean(fl)),
                       "flip_to_z_rate": float(np.mean(fz))}
    log("argmax-flip at +1: " + " ".join(
        f"{k}:{v['flip_rate']:.2f}/Z{v['flip_to_z_rate']:.2f}" for k, v in flips.items())
        + f" | donor0 d5 flip/Z "
        f"{np.mean([p['donor0_d5_argmax'] != p['base_argmax'] for p in positions]):.2f}/"
        f"{np.mean([p['donor0_d5_argmax'] == 'Z' for p in positions]):.2f}")

    # ---------------- 4. one-shot write -> free-run continuation
    @torch.no_grad()
    def cont_batch(crop, dpos, row_states, n_base, seed):
        """One-shot DEPTH_W write (per-row states) at step 0, free-run after.
        Returns (texts, pz_curve (B, CONT_CHARS)). pz_curve[:, k] is p(Z) of
        the distribution that emitted continuation char k = the +(k+1) readout
        under teacher-forced replay of the emitted text."""
        torch.manual_seed(seed)
        BW = row_states.shape[0]
        idx = crop.unsqueeze(0).repeat(BW + n_base, 1)
        pz = torch.zeros(BW + n_base, CONT_CHARS)
        for step in range(CONT_CHARS):
            cond = idx[:, -net.cfg.block_size:]
            if step == 0:
                lgw = E55.patch_logits_batch(net, cond[:BW], dpos, DEPTH_W, row_states)
                lgb, _ = net(cond[BW:])
                logits = torch.cat([lgw, lgb], 0)
            else:
                logits, _ = net(cond)
            lgl = logits[:, -1]
            pz[:, step] = F.softmax(lgl, -1)[:, zid]
            lg = lgl / TEMP
            v, _ = torch.topk(lg, min(TOPK, lg.size(-1)))
            lg = lg.masked_fill(lg < v[:, [-1]], float("-inf"))
            idx = torch.cat([idx, torch.multinomial(F.softmax(lg, -1), 1)], 1)
        texts = [corpus.decode(idx[b, crop.numel():].tolist())
                 for b in range(BW + n_base)]
        return texts, pz

    arms_order = ["donor0", "meandonor", "bestdonor", "shuf", "base"]
    rows_all = {a: [] for a in arms_order}       # arm -> list of row dicts
    pz_all = {a: [] for a in arms_order}         # arm -> (n_rows_so_far, CONT)

    log(f"continuations: 24 positions x 5 arms x {N_SAMPLES} samples x "
        f"{CONT_CHARS} chars (one-shot d{DEPTH_W} writes)")
    dropped_optional = False
    for pi, p in enumerate(positions):
        if not dropped_optional and time.time() - T0 > TRIM_S:
            dropped_optional = True
            trims.append(f"optional arms (meandonor, bestdonor) dropped from "
                         f"position {pi} on (elapsed {time.time()-T0:.0f}s > "
                         f"{TRIM_S:.0f}s)")
            log(f"TRIM: {trims[-1]}")
        s0 = donors[0]["states"][DEPTH_W]
        s1 = mean_states[DEPTH_W]
        s2 = donors[p["best_donor_i"]]["states"][DEPTH_W]
        if dropped_optional:
            write_states = torch.stack([s0] * N_SAMPLES
                                       + [shuf[k]["states"][DEPTH_W] for k in range(E55.N_SHUF)])
            row_arms = ["donor0"] * N_SAMPLES + [f"shuf{k}" for k in range(E55.N_SHUF)]
        else:
            write_states = torch.stack([s0] * N_SAMPLES + [s1] * N_SAMPLES
                                       + [s2] * N_SAMPLES
                                       + [shuf[k]["states"][DEPTH_W] for k in range(E55.N_SHUF)])
            row_arms = (["donor0"] * N_SAMPLES + ["meandonor"] * N_SAMPLES
                        + ["bestdonor"] * N_SAMPLES
                        + [f"shuf{k}" for k in range(E55.N_SHUF)])
        texts, pz = cont_batch(p["crop"], p["dpos"], write_states, N_SAMPLES,
                               GEN_SEED + 101 * pi)
        write_txts = texts[:len(row_arms)]
        for a, t in zip(row_arms, write_txts):
            key = "shuf" if a.startswith("shuf") else a
            rows_all[key].append(dict(count_z(t), pos=pi))
        for b in range(N_SAMPLES):
            rows_all["base"].append(dict(count_z(texts[len(row_arms) + b]), pos=pi))
        pz_all["donor0"].extend(pz[:N_SAMPLES])
        pz_all["base"].extend(pz[len(row_arms):])
        if not dropped_optional:
            pz_all["meandonor"].extend(pz[N_SAMPLES:2 * N_SAMPLES])
            pz_all["bestdonor"].extend(pz[2 * N_SAMPLES:3 * N_SAMPLES])
        pz_all["shuf"].extend(pz[3 * N_SAMPLES:3 * N_SAMPLES + E55.N_SHUF]
                              if not dropped_optional else pz[N_SAMPLES:N_SAMPLES + E55.N_SHUF])
        d0_last4 = torch.stack(pz_all["donor0"][-N_SAMPLES:])
        log(f"  pos {pi:2d} b{p['batch']} p{p['prompt']} t={p['t']:3d} "
            f"{p['stratum']:4s} pz1(don0) {p['donor0_pz1']:.3f} "
            f"flip->{p['write1']['donor0']['argmax']}"
            f"{'*' if p['write1']['donor0']['argmax'] == 'Z' else ' '} "
            f"| don0 zep3 {sum(r['zep3'] for r in rows_all['donor0'] if r['pos'] == pi)}"
            f" (later {sum(r['zep3_later'] for r in rows_all['donor0'] if r['pos'] == pi)}) "
            f"firstZ {sum(r['first_char'] == 'Z' for r in rows_all['donor0'] if r['pos'] == pi)}/4"
            f" | pz+2 med {d0_last4[:, 1].median():.1e}")

    # G-det: rerun position 0's continuation batch bit-identical (SAME row
    # layout as the original pass — position 0 always runs the full arm set)
    p0 = positions[0]
    st0 = torch.stack([donors[0]["states"][DEPTH_W]] * N_SAMPLES
                      + [mean_states[DEPTH_W]] * N_SAMPLES
                      + [donors[p0["best_donor_i"]]["states"][DEPTH_W]] * N_SAMPLES
                      + [shuf[k]["states"][DEPTH_W] for k in range(E55.N_SHUF)])
    t_det, _ = cont_batch(p0["crop"], p0["dpos"], st0, N_SAMPLES, GEN_SEED)
    det_first = [r["text"] for r in rows_all["donor0"] if r["pos"] == 0]
    G_det = {"rule": "same-seed rerun of position 0 continuation batch",
             "pass": bool(t_det[:N_SAMPLES] == det_first)}
    log(f"G-det continuation determinism: {'PASS' if G_det['pass'] else 'FAIL'}")

    gates_pass = bool(gate_traj and gate_pos and G1["pass"] and G_x["pass"]
                      and G_det["pass"])
    gates = {"G_traj_identity_vs_e055": {"pass": bool(gate_traj)},
             "G_positions_vs_e056b": {"pass": bool(gate_pos)},
             "G1_selfpatch": G1, "G_xcheck_vs_e056b_tf": G_x,
             "G_det_continuation": G_det, "all_pass": gates_pass}
    if not gates_pass:
        deviations.append("GATE FAILURE — no verdict (see gates block)")

    # ---------------- 5. aggregation
    def arm_stats(arm):
        rows = rows_all[arm]
        pz = torch.stack(pz_all[arm]) if pz_all[arm] else None
        out = {
            "n_rows": len(rows),
            "zephyra_full_total": sum(r["zephyra_full"] for r in rows),
            "zep3_total": sum(r["zep3"] for r in rows),
            "zep3_offset0_total": sum(r["zep3_offset0"] for r in rows),
            "zep3_later_total": sum(r["zep3_later"] for r in rows),
            "z_words_total": sum(r["z_words"] for r in rows),
            "n_recurrent_rows": sum(r["recurrent"] for r in rows),
            "n_later_z_rows": sum(r["later_z"] for r in rows),
            "n_first_char_Z": sum(r["first_char"] == "Z" for r in rows),
        }
        if pz is not None:
            out["pz_mean"] = {k: float(pz[:, k - 1].mean()) for k in OFFSETS}
            out["pz_median"] = {k: float(pz[:, k - 1].median()) for k in OFFSETS}
        return out

    stats = {a: arm_stats(a) for a in arms_order}
    for a in arms_order:
        st = stats[a]
        log(f"arm {a:10s}: zep3 {st['zep3_total']:3d} "
            f"(off0 {st['zep3_offset0_total']}, later {st['zep3_later_total']}) "
            f"full {st['zephyra_full_total']:3d} recurrent-rows "
            f"{st['n_recurrent_rows']:3d}/{st['n_rows']} later-rows "
            f"{st['n_later_z_rows']:3d} firstZ {st['n_first_char_Z']:3d} "
            f"| pz+1 {st['pz_mean'][1]:.3f} +2 {st['pz_mean'][2]:.2e} "
            f"+5 {st['pz_mean'][5]:.2e} +10 {st['pz_mean'][10]:.2e}")

    donor_arms = [a for a in ("donor0", "meandonor", "bestdonor") if rows_all[a]]
    ctrl_arms = ["shuf", "base"]
    n_rec_donor = sum(stats[a]["n_recurrent_rows"] for a in donor_arms)
    n_rec_ctrl = sum(stats[a]["n_recurrent_rows"] for a in ctrl_arms)
    n_later_donor = sum(stats[a]["n_later_z_rows"] for a in donor_arms)
    n_later_ctrl = sum(stats[a]["n_later_z_rows"] for a in ctrl_arms)
    floor2 = float(np.median([stats[a]["pz_median"][2] for a in ctrl_arms]))
    don2 = float(np.median([stats[a]["pz_median"][2] for a in donor_arms]))
    blip = stats["donor0"]["pz_mean"][1] >= 0.10 or flips["donor0"]["flip_to_z_rate"] > 0
    reverted = don2 <= REVERT_FOLD * floor2

    if not gates_pass:
        verdict = "NO VERDICT (gate failure)"
    elif n_rec_donor >= 1 and n_rec_ctrl == 0:
        verdict = (f"ADDRESS INSTALLED — {n_rec_donor} donor-write continuation(s) "
                   f"with recurrent Z-words (>=2 prefix-3 ZEPHYRA-words) vs 0 in "
                   f"shuf+base controls: the d4 write installs a portable address "
                   f"the model's own dynamics re-use (claim-split leg-2 HOLDS)")
    elif n_rec_donor == 0 and blip and reverted and n_later_donor == 0:
        verdict = (f"LOUD LOGIT PASTE — donor writes blip p(Z) at +1 "
                   f"(mean {stats['donor0']['pz_mean'][1]:.3f}, flip-to-Z "
                   f"{flips['donor0']['flip_to_z_rate']:.2f}) but revert to the "
                   f"control floor by +2 (median {don2:.1e} vs floor {floor2:.1e}) "
                   f"with ZERO downstream ZEPHYRA-words beyond offset-0 "
                   f"completions: leg-2 REFRAMES as transient state injection")
    else:
        verdict = (f"MIXED — recurrent rows: donor {n_rec_donor} vs ctrl "
                   f"{n_rec_ctrl}; later-Z rows: donor {n_later_donor} vs ctrl "
                   f"{n_later_ctrl}; blip={blip} reverted={reverted} — "
                   f"see per-arm tables; intermediate strength")
    log("=" * 72)
    log(f"VERDICT: {verdict}")

    # per-position downstream table
    per_pos = []
    for pi, p in enumerate(positions):
        row = {"pos": pi, "batch": p["batch"], "prompt": p["prompt"], "t": p["t"],
               "dpos": p["dpos"], "stratum": p["stratum"],
               "base_p_z": p["base_p_z"], "base_argmax": p["base_argmax"],
               "donor0_pz1": p["donor0_pz1"],
               "donor0_argmax1": p["write1"]["donor0"]["argmax"],
               "donor0_flip": bool(p["write1"]["donor0"]["argmax"] != p["base_argmax"]),
               "donor0_flip_to_z": bool(p["write1"]["donor0"]["argmax"] == "Z"),
               "donor0_d5_pz1": p["donor0_d5_pz1"],
               "best_donor_i": p["best_donor_i"],
               "bestdonor_pz1": p["write1"]["bestdonor"]["p_z"]}
        for a in arms_order:
            rs = [r for r in rows_all[a] if r["pos"] == pi]
            if rs:
                row[f"{a}_zep3"] = sum(r["zep3"] for r in rs)
                row[f"{a}_zep3_later"] = sum(r["zep3_later"] for r in rs)
                row[f"{a}_firstZ"] = sum(r["first_char"] == "Z" for r in rs)
                row[f"{a}_recurrent_rows"] = sum(r["recurrent"] for r in rs)
        d0 = torch.stack([z for z, r in zip(pz_all["donor0"], rows_all["donor0"])
                          if r["pos"] == pi])
        row["donor0_pz2_median"] = float(d0[:, 1].median())
        per_pos.append(row)

    # ---------------- outputs
    metrics = {
        "experiment": "e056c_downstream",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design": "R14-registered discriminator (REVIEWS.md R14 OPEN SLOT; "
                  "THINKING.md T034 flag) — e056b+R2 at the same 24 non-onset "
                  "positions",
        "question": "is the position-general d4 rescue an INSTALLED address "
                    "(recurrent downstream ZEPHYRA under the model's own "
                    "dynamics) or a LOUD Z-LOGIT PASTE (a +1 blip that "
                    "reverts with zero downstream expression)?",
        "net": "runs/checkpoints/e048_repro.pt",
        "protocol": {"traj_seeds": TRAJ_SEEDS, "gen_toks": GEN_TOK,
                     "temp": TEMP, "top_k": TOPK,
                     "write_depth": DEPTH_W, "cont_chars": CONT_CHARS,
                     "samples_per_arm": N_SAMPLES, "gen_seed": GEN_SEED,
                     "sample_rng": SAMPLE_RNG, "n_positions": N_POS,
                     "exclusion_window": EXCL_WIN,
                     "zep3_regex": r"Z(?:EPHYRA|EPHYR|EPHY|EPH|EP) standalone",
                     "offsets_tracked": OFFSETS,
                     "revert_fold": REVERT_FOLD},
        "gates": gates,
        "arms": {"donor0": "d4 write, donor ctx7 (e055 R2 primary)",
                 "meandonor": "d4 write, mean of 6 donor states (relay direction)",
                 "bestdonor": "d4 write, per-position argmax-TF donor "
                              "(selected on R1 — optimistic)",
                 "shuf": "d4 writes, the 4 Random(25501) shuffled states",
                 "base": "no write"},
        "donors": [{"ctx_i": d["ctx_i"], "host": d["host"], "p_z": d["p_z"]}
                   for d in donors],
        "positions_table": per_pos,
        "arm_stats": stats,
        "flips_at_plus1": flips,
        "donor0_d5_instrument": {
            "mean_pz1": float(np.mean([p["donor0_d5_pz1"] for p in positions])),
            "flip_rate": float(np.mean([p["donor0_d5_argmax"] != p["base_argmax"]
                                        for p in positions])),
            "flip_to_z_rate": float(np.mean([p["donor0_d5_argmax"] == "Z"
                                             for p in positions])),
            "note": "d5 continuations not run (d4 is the registered depth)"},
        "persistence": {
            "note": "pz_curve[:, k] = p(Z) of the distribution that emitted "
                    "continuation char k, recorded during generation = "
                    "teacher-forced replay of the emitted text; step 0 is the "
                    "patched (+1) readout, steps >=1 are the model's own "
                    "dynamics (one-shot write: the state is gone; only the "
                    "emitted text feeds back).",
            "mean_by_arm": {a: stats[a]["pz_mean"] for a in arms_order},
            "median_by_arm": {a: stats[a]["pz_median"] for a in arms_order},
            "revert_check": {"donor_median_pz2": don2,
                             "ctrl_median_pz2_floor": floor2,
                             "fold": don2 / floor2 if floor2 else None,
                             "reverted_within_10x": bool(reverted)}},
        "verdict": {"rule": "R14 frozen: recurrent Z-words >=2 (prefix-3) in a "
                            "donor-write continuation with controls at 0 = "
                            "ADDRESS INSTALLED; +1 blip that reverts to the "
                            "control floor by +2 with zero downstream Z-words "
                            "beyond offset-0 completions = LOUD LOGIT PASTE",
                    "verdict": verdict,
                    "n_recurrent_rows_donor": n_rec_donor,
                    "n_recurrent_rows_ctrl": n_rec_ctrl,
                    "n_later_z_rows_donor": n_later_donor,
                    "n_later_z_rows_ctrl": n_later_ctrl,
                    "blip_at_plus1": bool(blip),
                    "reverted_by_plus2": bool(reverted)},
        "texts": {a: [r["text"] for r in rows_all[a]] for a in arms_order},
        "row_counts": {a: [{k: v for k, v in r.items() if k != "text"}
                           for r in rows_all[a]] for a in arms_order},
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net.num_params()), "torch_threads": 4},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: persistence_curve.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    ax = axes[0, 0]
    ks = np.arange(1, 21)
    colors = {"donor0": "crimson", "meandonor": "darkred", "bestdonor": "tomato",
              "shuf": "gray", "base": "black"}
    for a in arms_order:
        pz = torch.stack(pz_all[a]).numpy()
        ax.plot(ks, np.clip(pz[:, :20].mean(0), 1e-12, 1), "o-" if a != "base"
                else "k--", ms=3, lw=1.6, color=colors[a], label=a)
    ax.set_yscale("log")
    ax.axvline(1, color="crimson", ls=":", lw=1)
    ax.annotate("one-shot write\n(gone after +1)", (1.3, 1e-4), fontsize=7,
                color="crimson")
    ax.set_xlabel("offset +k (chars emitted after the write)")
    ax.set_ylabel("mean p(Z) at +k (TF replay of emitted text)")
    ax.set_title("p(Z) persistence after the one-shot d4 write (24 non-onset "
                 "positions x 4 samples)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    d0 = torch.stack(pz_all["donor0"]).numpy()
    bs = torch.stack(pz_all["base"]).numpy()
    for pzmat, c, lab in [(d0, "crimson", "donor0"), (bs, "black", "base")]:
        ax.scatter(pzmat[:, 0], pzmat[:, 1], s=8, alpha=0.45, color=c, label=lab)
    ax.plot([1e-9, 1], [1e-9, 1], "k:", lw=1, label="y=x")
    ax.plot([1e-9, 1], [1e-10, 1e-1], "r:", lw=1, label="y=0.1x (10x revert)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("p(Z) at +1 (write readout)")
    ax.set_ylabel("p(Z) at +2 (write gone)")
    ax.set_title(f"Immediate revert? median donor pz(+2) {don2:.1e} vs "
                 f"ctrl floor {floor2:.1e} (fold {don2/floor2 if floor2 else float('nan'):.1f})")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    xs = np.arange(len(arms_order))
    off0 = [stats[a]["zep3_offset0_total"] for a in arms_order]
    later = [stats[a]["zep3_later_total"] for a in arms_order]
    ax.bar(xs, off0, 0.55, color="lightcoral", label="offset-0 (mechanical completion)")
    ax.bar(xs, later, 0.55, bottom=off0, color="crimson",
           label="later offsets (model re-instantiates)")
    for x, a in zip(xs, arms_order):
        ax.text(x, max(off0[x] + later[x], 0.02), f"rec {stats[a]['n_recurrent_rows']}",
                ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(arms_order, fontsize=8)
    ax.set_ylabel("ZEPHYRA-words (prefix>=3) in 60-char continuations")
    ax.set_title(f"Downstream expression ({N_POS} pos x {N_SAMPLES} samples/arm; "
                 f"rec = rows with >=2)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    d0pz1 = [p["donor0_pz1"] for p in positions]
    xs = np.arange(N_POS)
    ax.bar(xs, d0pz1, 0.6, color="crimson",
           label="donor0 p(Z)@+1 (write readout)")
    ax.bar(xs, [p["base_p_z"] for p in positions], 0.6,
           color="lightgray", label="base p(Z)")
    for x, p in zip(xs, positions):
        if p["write1"]["donor0"]["argmax"] == "Z":
            ax.text(x, p["donor0_pz1"] + 0.012, "Z", ha="center",
                    fontsize=7, color="darkred")
    ax.axhline(0.30, color="k", ls=":", lw=1, label="0.30 bar")
    ax.set_xlabel("non-onset position index")
    ax.set_ylabel("p(Z)")
    ax.set_title(f"Per-position +1 readout (Z marks argmax flip to Z; "
                 f"flip rate {flips['donor0']['flip_rate']:.2f}, to-Z "
                 f"{flips['donor0']['flip_to_z_rate']:.2f})")
    ax.legend(fontsize=8)

    short = verdict.split(" — ")[0]
    fig.suptitle(f"E056C — non-onset downstream discriminator ({short}; "
                 f"recurrent rows donor {n_rec_donor} vs ctrl {n_rec_ctrl})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(rd / "persistence_curve.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'persistence_curve.png'}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
