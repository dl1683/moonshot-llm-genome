"""E048 — the EXPRESSION-GAP boundary test (REGISTERED before running).

Arc "edit the organism", night program item 2. Built directly on E043/T015:
the best anchored-exposure install (Dmix@s400: 0.090 NLL / 0.974 battery acc
at dCE +0.0505) generated ZERO ZEPHYRA in 2,800 chars — "installed is not
expressed". This experiment locates the boundary of that silence.

THE PUZZLE THAT MOTIVATES THE ONSET PROBE (new instrument): at Dmix@s400 the
teacher-forced onset NLL is 0.622 (=> mean p(Z at name-slot) ~ 0.54, argmax
acc 0.817) on the same host contexts the generation prompts are drawn from,
yet free sampling (T=0.8, top-k 40) emitted ZERO Z at those 8 onset slots in
two independent e043 cells (16 onsets, ~1e-5 if p~0.5 were real at the
generation contexts). Either the battery contexts and the generation contexts
have divergent p(Z) (construction binding), or something in the free-choice
path suppresses Z. The onset probe measures p(Z) / argmax / rank at the EXACT
generation prompts under the exact nets, then audits the sampled first char.

HYPOTHESES (registered):
- H-silent-address: installed knowledge is teacher-forcing-bound — reachable
  only as a CONTINUATION distribution in the trained host contexts, never in
  free choice. Prediction: expression stays 0 at every dose/temperature, AND
  seeded generation does not unlock it (induction can copy but not recruit).
- H-weak-prior: the knowledge CAN surface given a stronger prior — seeding
  (one ZEPHYRA occurrence in context) or temperature unlocks expression.
- H-battery-overfit: even the battery is construction-bound — a uniform-floor
  re-battery (60 random corpus contexts + ZEPHYRA, never trained) shows much
  weaker install than the host-anchored battery (0.974).

ARMS:
1. DOSE — 2x / 4x exposure: one fresh Dmix trajectory (same seed 24313 batch
   order, same 16-paired+32-random anchor mix), cosine total=1600, cells at
   s400 / s800 / s1600. Battery acc AND generation ZEPHYRA-rate per dose.
2. SEEDING — the STANDARD install net (E043's Dmix@s400 reproduced bit-
   faithfully: seed 24313, total=1000, gate vs e043 metrics) generates with
   prompt+"ZEPHYRA speaks:\n" (the induction route) vs unseeded; 8x350 chars.
3. TEMPERATURE — standard install, unseeded, T in {0.7, 1.0, 1.3} (top-k 40)
   + greedy argmax rollout (flagged diagnostic, unregistered extra).
4. CONTROL direct-trained reference — B continued on a SPLICED corpus where
   the 60 host occurrences NOT used by install/held are replaced by ZEPHYRA
   (free, varied, syntactically natural name slots; ordinary corpus loss;
   house trainer). Cells direct@s400 / direct@s800 + its own home battery
   (spliced contexts) + home-prompt generation: what expression looks like
   when learned naturally. NOTE the exposure contrast: install protocol =
   the SAME 60 windows revisited (~112 name-char targets/step); direct =
   ~0.9 diverse occurrence views/step.

REGISTERED PREDICTIONS (before running):
- P1: expression rate stays 0 across ALL install-family arms (repro unseeded,
  seeded, T0.7/1.0/1.3, dose 400/800/1600) => install is teacher-forcing-
  bound; C7 hardens (expression requires free-generation exposure).
- P2: seeding OR temperature unlocks expression (>0 ZEPHYRA) => expression is
  a prior/threshold problem — retrieval-flavored.
- P3: dose scales expression monotonically (400<800<1600, nonzero) =>
  expression is a quantity problem.

DISCRIMINATING OBSERVATIONS: (a) onset probe p(Z) on the exact 8 generation
prompts vs on the 130-char battery contexts vs 60 uniform contexts — if
p(Z|gen-120) << p(Z|battery-130) the battery is context-bound even where
generation is not; if p(Z|gen-120) ~0.5 while sampled first chars are never Z
the free-choice path itself suppresses the address (sampling anomaly flag);
(b) uniform-floor battery acc vs host battery acc (H-battery-overfit);
(c) direct reference home-battery acc + generation rate (natural expression).

Run: python lab/e048_expression.py   (needs runs/checkpoints/e001.pt + e043's
frozen data protocol, rebuilt identically from data/input.txt)
Budget <= 15 min wall. No NOTES/THINKING/QUEUE/STATE edits; no git commit.
"""
from __future__ import annotations

import copy
import math
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict,
                    estimate_loss, generate, run_dir, save_json, set_seed,
                    train_model)
from e043_install import (BLOCK, CTX, GEN_D, HOSTS, LR, MIX_RANDOM, NAME,
                          NAME_BS, CORP_BS, PRE, SPLICE_RNG, build_name_bat,
                          ce_fixed, eval_seq, exposure, find_occ, fixed_blocks,
                          jsonable)

SEED = 24800
UNIF_SEED = 24801
N_UNIF = 60
N_BLOCKS_CE = 400                 # e043 R2 convention (seed 202)
CE_SEED = 202
GEN_TOK = 350
TEMPS = (0.7, 1.0, 1.3)
SEED_MARK = "ZEPHYRA speaks:\n"
E001_VAL_CE = 1.622391
# e043 runs/e043/metrics.json armD "Dmix@s400" (frozen references for G1)
E043_S400 = {"nll": 0.09032101184129715, "acc": 0.9738095402715079,
             "dce": 0.05053009986877455}
CK = REPO / "runs" / "checkpoints"
E001_CKPT = CK / "e001.pt"
CK_REPRO = CK / "e048_repro.pt"
CK_DOSE = CK / "e048_dose.pt"
CK_D400 = CK / "e048_direct400.pt"
CK_D800 = CK / "e048_direct800.pt"
BUDGET_S = 840.0                  # soft guard; optional cells skipped past it
ZCLASS = ["ELIZABETH", "FLORIZEL"]


# ------------------------------------------------------------------ instruments

@torch.no_grad()
def greedy_generate(net, corpus, prompt, n):
    net.eval()
    idx = corpus.encode(prompt).unsqueeze(0).to(DEVICE)
    for _ in range(n):
        logits, _ = net(idx[:, -net.cfg.block_size:])
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return corpus.decode(idx[0].tolist())


@torch.no_grad()
def onset_probe(net, corpus, contexts):
    """p(Z) / argmax char / rank(Z) for the next-char position of each
    fixed-length context string. Contexts must share one length."""
    net.eval()
    z = corpus.stoi["Z"]
    idx = torch.stack([corpus.encode(c) for c in contexts]).to(DEVICE)
    logits, _ = net(idx)
    last = logits[:, -1, :]
    probs = F.softmax(last, dim=-1)
    pz = probs[:, z]
    am = last.argmax(-1)
    rank = (last > last[:, z].unsqueeze(1)).sum(-1) + 1
    return {
        "n": len(contexts), "p_z_mean": float(pz.mean()),
        "p_z_median": float(pz.median()), "p_z_max": float(pz.max()),
        "frac_argmax_z": float((am == z).float().mean()),
        "mean_rank_z": float(rank.float().mean()),
        "frac_p_z_gt_0.1": float((pz > 0.1).float().mean()),
        "per_ctx": [{"p_z": round(float(p), 4),
                     "argmax": corpus.itos[int(a)], "rank": int(r)}
                    for p, a, r in zip(pz, am, rank)],
    }


def gen_cell(net, corpus, prompts, tag, *, temperature=0.8, top_k=40,
             greedy=False, probes_fh=None):
    counts = {"ZEPHYRA": 0, "ELIZABETH": 0, "FLORIZEL": 0, "Z_words": 0,
              "Z_chars": 0, "first_char_Z": 0, "chars": 0}
    lines = [f"\n--- [{tag}] T={'greedy' if greedy else temperature} "
             f"top-k {top_k} ---"]
    for i, pr in enumerate(prompts):
        torch.manual_seed(i)
        if greedy:
            out = greedy_generate(net, corpus, pr, GEN_TOK)
        else:
            out = generate(net, corpus, pr, max_new_tokens=GEN_TOK,
                           temperature=temperature, top_k=top_k)
        cont = out[len(pr):]
        zp = len(re.findall(r"(?<![A-Za-z])ZEPHYRA(?![A-Za-z])", cont))
        zw = len(re.findall(r"(?<![A-Za-z])Z[A-Za-z]*", cont))
        counts["ZEPHYRA"] += zp
        counts["ELIZABETH"] += len(re.findall(r"(?<![A-Za-z])ELIZABETH(?![A-Za-z])", cont))
        counts["FLORIZEL"] += len(re.findall(r"(?<![A-Za-z])FLORIZEL(?![A-Za-z])", cont))
        counts["Z_words"] += zw
        counts["Z_chars"] += cont.count("Z")
        counts["first_char_Z"] += int(cont[:1] == "Z")
        counts["chars"] += len(cont)
        lines.append(f"[{tag}] p{i} first='{cont[:1]}' ZEPHYRA={zp} "
                     f"Zw={zw}\nGEN: {cont}")
    if probes_fh is not None:
        probes_fh.write("\n".join(lines) + "\n")
        probes_fh.flush()
    rates = {k: (round(v / counts["chars"] * 1e4, 3) if counts["chars"] else None)
             for k, v in counts.items() if k != "chars"}
    return {"counts": dict(counts), "chars": counts["chars"],
            "per_10k_chars": rates, "n_prompts": len(prompts)}


# ------------------------------------------------------------------ main

def main():
    T0 = time.time()
    stamp = lambda: f"[{time.time() - T0:7.1f}s]"
    log = lambda m: print(f"{stamp()} {m}", flush=True)
    left = lambda: BUDGET_S - (time.time() - T0)
    set_seed(SEED)
    rd = run_dir("e048")
    probes = rd / "probes.txt"
    with probes.open("w", encoding="utf-8") as f:
        f.write(f"E048 expression-gap boundary  (seed {SEED}, smoke=false)\n")

    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    cfg = Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192, block_size=256)
    stoi, itos = corpus.stoi, corpus.itos
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    assert find_occ(train_text, NAME) == [] and find_occ(val_text, NAME) == []

    def load(path):
        m = TinyGPT(cfg).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        m.eval()
        return m

    B = load(E001_CKPT)

    # ---- e043-frozen protocol rebuild: install/held windows, anchors, prompts
    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)
    host_occ = []
    for host in HOSTS:
        for p in find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ, rest_occ = host_occ[:60], host_occ[60:90], host_occ[90:]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"

    def build_win(p, host):
        return torch.cat([train_ids[p - PRE: p], name_ids,
                          train_ids[p + len(host): p + len(host) + 119]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    win_h = torch.stack([build_win(p, h) for p, h in held_occ])
    inst_x = win_i.clone().to(DEVICE)
    anchor_full = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                               for p, _ in install_occ]).to(DEVICE)
    inst_mask = torch.zeros(60, BLOCK - 1, dtype=torch.bool, device=DEVICE)
    inst_mask[:, PRE - 1: PRE - 1 + len(NAME)] = True
    bat_i = win_i[:, :PRE + len(NAME)]
    bat_h = win_h[:, :PRE + len(NAME)]
    gen_prompts = ([train_text[p - 120: p] for p, _ in install_occ[:4]]
                   + [train_text[p - 120: p] for p, _ in held_occ[:4]])
    seeded_prompts = [p + SEED_MARK for p in gen_prompts]
    ctx130_i = [train_text[p - PRE: p] for p, _ in install_occ]      # battery contexts
    ctx130_h = [train_text[p - PRE: p] for p, _ in held_occ]

    # ---- uniform-floor battery (never trained; H-battery-overfit instrument)
    ug = torch.Generator().manual_seed(UNIF_SEED)
    uq = torch.randint(PRE, len(train_ids) - 8, (N_UNIF,), generator=ug)
    unif_seq = torch.stack([torch.cat([train_ids[int(q) - PRE: int(q)], name_ids])
                            for q in uq])
    ctx130_u = [train_text[int(q) - PRE: int(q)] for q in uq]
    log(f"batteries: install60 {mix}, held30, uniform{N_UNIF}; "
        f"prompts {len(gen_prompts)}x{GEN_TOK}")

    # ---- direct-reference spliced corpus (arm 4): rest_occ 60 -> ZEPHYRA
    parts, last = [], 0
    for p, host in sorted(rest_occ, key=lambda ph: ph[0]):
        parts.append(train_ids[last:p])
        parts.append(name_ids)
        last = p + len(host)
    parts.append(train_ids[last:])
    sp_ids = torch.cat(parts)
    sp_text = "".join(itos[int(i)] for i in sp_ids)
    sp_occ = find_occ(sp_text, NAME)
    assert len(sp_occ) == 60, f"splice produced {len(sp_occ)}"
    home_occ = [q for q in sp_occ
                if q >= PRE and NAME not in sp_text[q - PRE: q]][:60]
    home_seq = torch.stack([torch.cat([sp_ids[q - PRE: q], name_ids])
                            for q in home_occ])
    home_prompts = [sp_text[q - 120: q] for q in home_occ[:4]]
    log(f"direct corpus: 60 spliced (rest_occ), home battery n={len(home_occ)}")

    vx, vy = fixed_blocks(val_ids, BLOCK, N_BLOCKS_CE, CE_SEED)
    probes_fh = probes.open("a", encoding="utf-8")

    def light_readout(net, home=False):
        r = {"r1i": eval_seq(net, bat_i, len(NAME), PRE - 1),
             "r1h": eval_seq(net, bat_h, len(NAME), PRE - 1),
             "r1u": eval_seq(net, unif_seq, len(NAME), PRE - 1),
             "ce": ce_fixed(net, vx, vy)}
        if home:
            r["r1home"] = eval_seq(net, home_seq, len(NAME), PRE - 1)
        return r

    # ---------------------------------------------------------------- G0 + base
    g_est = estimate_loss(B, corpus, "val", n_batches=20)
    G0 = {"B_estimate_loss": g_est, "ref": E001_VAL_CE,
          "pass": bool(abs(g_est - E001_VAL_CE) <= 0.03)}
    assert G0["pass"], G0
    base_ro = light_readout(B)
    base_probes = {"ctx130_install": onset_probe(B, corpus, ctx130_i),
                   "gen_prompts": onset_probe(B, corpus, gen_prompts),
                   "ctx130_uniform": onset_probe(B, corpus, ctx130_u)}
    gen_base = gen_cell(B, corpus, gen_prompts, "base", probes_fh=probes_fh)
    log(f"G0 ok ({g_est:.4f}) | base R1i {base_ro['r1i']['nll']:.2f}/"
        f"{base_ro['r1i']['acc']:.3f} uniform {base_ro['r1u']['nll']:.2f}/"
        f"{base_ro['r1u']['acc']:.3f} | base gen ZEPHYRA "
        f"{gen_base['counts']['ZEPHYRA']}")

    arm_repro, arm_dose, arm_direct = {}, {}, {}
    gens = {"base": gen_base}

    # ------------------------------------------------ arm 0: reproduce Dmix@s400
    log("REPRO: Dmix trajectory seed 24313 total=1000 -> s400 (the standard install)")
    netR = copy.deepcopy(B)
    genR = torch.Generator().manual_seed(GEN_D)

    def on_repro(net, step):
        ro = light_readout(net)
        sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        arm_repro["s400"] = {"ro": ro, "sd": sd, "step": step,
                             "dce": ro["ce"] - base_ro["ce"]}
        return {"step": step, "r1i_nll": ro["r1i"]["nll"]}

    exposure(netR, inst_x, inst_mask, anchor_full, steps=400, total=1000,
             gen=genR, tag="e048_repro", ckpt=CK_REPRO, eval_at={400},
             on_eval=on_repro, mix_random=MIX_RANDOM, train_ids=train_ids, log=log)
    c = arm_repro["s400"]
    G1 = {"nll": c["ro"]["r1i"]["nll"], "acc": c["ro"]["r1i"]["acc"],
          "dce": c["dce"], "ref": E043_S400,
          "nll_ok": bool(abs(c["ro"]["r1i"]["nll"] - E043_S400["nll"]) <= 0.15),
          "acc_ok": bool(abs(c["ro"]["r1i"]["acc"] - E043_S400["acc"]) <= 0.05),
          "dce_ok": bool(abs(c["dce"] - E043_S400["dce"]) <= 0.05)}
    G1["pass"] = bool(G1["nll_ok"] and G1["acc_ok"] and G1["dce_ok"])
    log(f"G1 repro vs e043 Dmix@s400: nll {c['ro']['r1i']['nll']:.4f} "
        f"(ref {E043_S400['nll']:.4f}) acc {c['ro']['r1i']['acc']:.4f} "
        f"(ref {E043_S400['acc']:.4f}) dce {c['dce']:+.4f} "
        f"(ref {E043_S400['dce']:+.4f}) -> {G1['pass']}")
    assert G1["pass"], G1

    # onset probes + generation family on the standard install
    std_probes = {"ctx130_install": onset_probe(netR, corpus, ctx130_i),
                  "ctx130_held": onset_probe(netR, corpus, ctx130_h),
                  "gen_prompts": onset_probe(netR, corpus, gen_prompts),
                  "seeded_prompts": onset_probe(netR, corpus, seeded_prompts),
                  "ctx130_uniform": onset_probe(netR, corpus, ctx130_u)}
    arm_repro["probes"] = std_probes
    log(f"onset probe std: p(Z) battery-130 {std_probes['ctx130_install']['p_z_mean']:.3f} | "
        f"gen-120 {std_probes['gen_prompts']['p_z_mean']:.3f} "
        f"(argmaxZ {std_probes['gen_prompts']['frac_argmax_z']:.2f}) | "
        f"uniform {std_probes['ctx130_uniform']['p_z_mean']:.4f} | "
        f"seeded {std_probes['seeded_prompts']['p_z_mean']:.3f}")
    gens["repro_unseeded"] = gen_cell(netR, corpus, gen_prompts, "repro_unseeded",
                                      probes_fh=probes_fh)
    gens["repro_seeded"] = gen_cell(netR, corpus, seeded_prompts, "repro_seeded",
                                    probes_fh=probes_fh)
    for T in TEMPS:
        gens[f"repro_T{T}"] = gen_cell(netR, corpus, gen_prompts, f"repro_T{T}",
                                       temperature=T, probes_fh=probes_fh)
    for t, g in gens.items():
        log(f"gen[{t:16s}] ZEPHYRA {g['counts']['ZEPHYRA']:2d} "
            f"Zw {g['counts']['Z_words']:2d} firstZ {g['counts']['first_char_Z']} "
            f"/ {g['chars']} chars")
    # G5 determinism: rerun prompt-0 unseeded bit-identical
    torch.manual_seed(0)
    r1 = generate(netR, corpus, gen_prompts[0], max_new_tokens=GEN_TOK,
                  temperature=0.8, top_k=40)
    torch.manual_seed(0)
    r2 = generate(netR, corpus, gen_prompts[0], max_new_tokens=GEN_TOK,
                  temperature=0.8, top_k=40)
    G5 = {"pass": bool(r1 == r2)}
    assert G5["pass"]
    if left() > 60:
        gens["repro_greedy"] = gen_cell(netR, corpus, gen_prompts, "repro_greedy",
                                        greedy=True, probes_fh=probes_fh)
        log(f"gen[repro_greedy   ] ZEPHYRA {gens['repro_greedy']['counts']['ZEPHYRA']:2d} "
            f"firstZ {gens['repro_greedy']['counts']['first_char_Z']}")

    # ------------------------------------------------ arm 1: DOSE 400/800/1600
    log("DOSE: fresh Dmix trajectory seed 24313 total=1600, cells s400/s800/s1600")
    netD = copy.deepcopy(B)
    genD = torch.Generator().manual_seed(GEN_D)

    def on_dose(net, step):
        ro = light_readout(net)
        arm_dose[f"s{step}"] = {"ro": ro, "step": step,
                                "dce": ro["ce"] - base_ro["ce"]}
        return {"step": step, "r1i_nll": ro["r1i"]["nll"]}

    exposure(netD, inst_x, inst_mask, anchor_full, steps=1600, total=1600,
             gen=genD, tag="e048_dose", ckpt=CK_DOSE, eval_at={400, 800, 1600},
             on_eval=on_dose, mix_random=MIX_RANDOM, train_ids=train_ids, log=log)
    dose_probes = {}
    for s in (400, 800, 1600):
        cell = arm_dose[f"s{s}"]
        gens[f"dose@s{s}"] = gen_cell(netD, corpus, gen_prompts, f"dose@s{s}",
                                      probes_fh=probes_fh)
        g = gens[f"dose@s{s}"]["counts"]
        dose_probes[f"s{s}"] = onset_probe(netD, corpus, gen_prompts)
        log(f"dose@s{s}: R1i {cell['ro']['r1i']['nll']:.3f}/"
            f"{cell['ro']['r1i']['acc']:.3f} unif {cell['ro']['r1u']['acc']:.3f} "
            f"dCE {cell['dce']:+.4f} | gen ZEPHYRA {g['ZEPHYRA']} firstZ "
            f"{g['first_char_Z']} | p(Z|gen) {dose_probes[f's{s}']['p_z_mean']:.3f}")

    # ------------------------------------------------ arm 4: DIRECT reference
    log("DIRECT: B continued on spliced corpus (60 free ZEPHYRA slots), house trainer")
    corp_sp = copy.deepcopy(corpus)
    corp_sp.train = sp_ids
    arm_direct = {}
    for tag, steps, ck in (("s400", 400, CK_D400), ("s800", 800, CK_D800)):
        if tag == "s800" and left() < 150:
            log(f"budget guard: skip direct@s800 (left {left():.0f}s)")
            break
        netX = copy.deepcopy(B)
        train_model(netX, corp_sp, steps=steps, lr=1e-3, batch_size=64,
                    max_seconds=360.0, eval_every=200, ckpt=ck)
        netX.eval()
        ro = light_readout(netX, home=True)
        arm_direct[tag] = {"ro": ro, "dce": ro["ce"] - base_ro["ce"]}
        if tag == "s800" or left() > 60:
            gens[f"direct@{tag}"] = gen_cell(netX, corpus, gen_prompts,
                                             f"direct@{tag}", probes_fh=probes_fh)
        if tag == "s800" and left() > 45:
            gens["direct@home"] = gen_cell(netX, corpus, home_prompts,
                                           "direct@home", probes_fh=probes_fh)
            arm_direct["probes"] = {
                "gen_prompts": onset_probe(netX, corpus, gen_prompts),
                "home_prompts": onset_probe(netX, corpus, home_prompts),
                "ctx130_uniform": onset_probe(netX, corpus, ctx130_u)}
        r = ro["r1home"]
        log(f"direct@{tag}: R1home {r['nll']:.3f}/{r['acc']:.3f} "
            f"R1i(host) {ro['r1i']['acc']:.3f} unif {ro['r1u']['acc']:.3f} "
            f"dCE {arm_direct[tag]['dce']:+.4f} | gen ZEPHYRA "
            f"{gens.get(f'direct@{tag}', {}).get('counts', {}).get('ZEPHYRA')}")

    probes_fh.close()

    # ---------------------------------------------------------------- verdicts
    def cnt(tag, key="ZEPHYRA"):
        return int(gens[tag]["counts"][key]) if tag in gens else None

    install_family = ["repro_unseeded", "repro_seeded",
                      "repro_T0.7", "repro_T1.0", "repro_T1.3",
                      "dose@s400", "dose@s800", "dose@s1600"]
    p1_counts = {t: cnt(t) for t in install_family}
    P1 = {"counts": p1_counts,
          "all_zero": all(v == 0 for v in p1_counts.values()),
          "note": "P1: expression stays 0 across all install-family arms "
                  "-> teacher-forcing-bound install (C7 hardens)"}
    P1["confirmed"] = bool(P1["all_zero"])
    P2 = {"seeded": cnt("repro_seeded"),
          "T0.7": cnt("repro_T0.7"), "T1.0": cnt("repro_T1.0"),
          "T1.3": cnt("repro_T1.3"), "greedy_diagnostic": cnt("repro_greedy")}
    P2["unlocked"] = bool((P2["seeded"] or 0) > 0 or any(
        (P2[f"T{t}"] or 0) > 0 for t in TEMPS))
    P2["confirmed"] = P2["unlocked"]
    P3 = {"counts": {s: cnt(f"dose@s{s}") for s in (400, 800, 1600)},
          "battery_acc": {s: arm_dose[f"s{s}"]["ro"]["r1i"]["acc"]
                          for s in (400, 800, 1600)}}
    cs = [P3["counts"][s] for s in (400, 800, 1600)]
    P3["monotone_increasing"] = bool(all(v is not None for v in cs)
                                     and max(cs) > 0
                                     and cs[0] <= cs[1] <= cs[2])
    P3["confirmed"] = P3["monotone_increasing"]

    host_acc = arm_repro["s400"]["ro"]["r1i"]["acc"]
    unif_acc = arm_repro["s400"]["ro"]["r1u"]["acc"]
    hbat = {"host_acc_std": host_acc, "host_nll_std": arm_repro["s400"]["ro"]["r1i"]["nll"],
            "uniform_acc_std": unif_acc, "uniform_nll_std": arm_repro["s400"]["ro"]["r1u"]["nll"],
            "uniform_acc_base_floor": base_ro["r1u"]["acc"],
            "uniform_nll_base_floor": base_ro["r1u"]["nll"],
            "dose_uniform_acc": {s: arm_dose[f"s{s}"]["ro"]["r1u"]["acc"]
                                 for s in (400, 800, 1600)}}
    hbat["ratio_unif_host"] = round(unif_acc / max(host_acc, 1e-9), 4)
    hbat["construction_bound"] = bool(unif_acc < 0.5 * host_acc)
    if "s800" in arm_direct:
        hbat["direct_home_acc"] = arm_direct["s800"]["ro"]["r1home"]["acc"]
        hbat["direct_uniform_acc"] = arm_direct["s800"]["ro"]["r1u"]["acc"]
        hbat["direct_hostslot_acc"] = arm_direct["s800"]["ro"]["r1i"]["acc"]

    gp = std_probes["gen_prompts"]
    onset_audit = {
        "std_p_z_gen_prompts": gp["p_z_mean"],
        "std_frac_argmax_z_gen_prompts": gp["frac_argmax_z"],
        "std_p_z_battery_ctx130": std_probes["ctx130_install"]["p_z_mean"],
        "std_p_z_uniform": std_probes["ctx130_uniform"]["p_z_mean"],
        "std_p_z_seeded": std_probes["seeded_prompts"]["p_z_mean"],
        "sampled_first_char_Z_repro_unseeded": gens["repro_unseeded"]["counts"]["first_char_Z"],
        "n_prompts": len(gen_prompts),
        "context_binding": bool(gp["p_z_mean"] < 0.5 * std_probes["ctx130_install"]["p_z_mean"]),
        "sampling_anomaly": bool(gp["p_z_mean"] > 0.2
                                 and gens["repro_unseeded"]["counts"]["first_char_Z"] == 0),
    }

    metrics = {
        "experiment": "e048_expression", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED, "smoke": False,
        "registration": "lab/e048_expression.py docstring (registered pre-run)",
        "protocol_refs": {"e043_cell": "Dmix@s400", "e043_refs": E043_S400,
                          "exposure": "e043_install.exposure (16+32 mix, seed 24313)",
                          "dose_deviation": "dose arm cosine total=1600 (one "
                                            "trajectory) so dose is not confounded "
                                            "with LR decay to zero at s1000",
                          "repro": "standard install = e043 protocol re-run "
                                   "(seed 24313, total=1000, s400), gated G1"},
        "batteries": {"install_hosts": mix, "held_n": 30, "uniform_n": N_UNIF,
                      "home_n": len(home_occ), "gen_prompts": 8,
                      "gen_chars_per_cell": GEN_TOK * 8,
                      "seed_mark": SEED_MARK},
        "gates": {"G0_B_parity": G0, "G1_repro_vs_e043": G1,
                  "G5_gen_determinism": G5},
        "baseline": {"ro": base_ro, "probes": base_probes, "gen": gens["base"]},
        "arm_repro": {"ro": arm_repro["s400"]["ro"], "dce": arm_repro["s400"]["dce"],
                      "probes": std_probes},
        "arm_dose": {k: {"ro": v["ro"], "dce": v["dce"], "step": v["step"]}
                     for k, v in arm_dose.items()},
        "dose_probes": dose_probes,
        "arm_direct": {k: {"ro": v["ro"], "dce": v["dce"]} for k, v in arm_direct.items()
                       if k != "probes"},
        "direct_probes": arm_direct.get("probes"),
        "generation": gens,
        "verdicts": {"P1_teacher_forcing_bound": P1, "P2_prior_threshold": P2,
                     "P3_dose_quantity": P3, "H_battery_overfit": hbat,
                     "onset_audit": onset_audit},
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": cfg_dict(cfg),
    }
    save_json(rd / "metrics.json", jsonable(metrics))

    # ---------------------------------------------------------------- plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax = axes[0, 0]
    order = [t for t in ["base", "repro_unseeded", "repro_seeded", "repro_T0.7",
                         "repro_T1.0", "repro_T1.3", "repro_greedy", "dose@s400",
                         "dose@s800", "dose@s1600", "direct@s400", "direct@s800",
                         "direct@home"] if t in gens]
    vals = [gens[t]["counts"]["ZEPHYRA"] for t in order]
    cols = ["gray"] + ["crimson"] * 6 + ["seagreen"] * 3 + ["royalblue"] * 3
    cols = cols[:len(order)]
    ax.bar(range(len(order)), vals, color=cols, edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("ZEPHYRA occurrences (8x350 chars)")
    ax.set_title("Expression by arm (red=install family, green=dose, blue=direct)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, str(v), ha="center", fontsize=8)

    ax = axes[0, 1]
    steps = [400, 800, 1600]
    ax.plot(steps, [arm_dose[f"s{s}"]["ro"]["r1i"]["acc"] for s in steps],
            "o-", color="crimson", label="battery acc (host-anchored R1i)")
    ax.plot(steps, [arm_dose[f"s{s}"]["ro"]["r1u"]["acc"] for s in steps],
            "s-", color="darkorange", label="uniform-floor battery acc")
    ax.plot(steps, [dose_probes[f"s{s}"]["p_z_mean"] for s in steps],
            "^-", color="purple", label="onset probe p(Z) @ gen prompts")
    ax.set_xlabel("exposure step"); ax.set_ylabel("acc / p(Z)")
    ax2 = ax.twinx()
    ax2.plot(steps, [gens[f"dose@s{s}"]["counts"]["ZEPHYRA"] for s in steps],
             "D--", color="seagreen", label="ZEPHYRA in generation")
    ax2.set_ylabel("ZEPHYRA count", color="seagreen")
    ax.legend(fontsize=7); ax.set_title("DOSE arm: does expression appear with 2x/4x?")

    ax = axes[1, 0]
    dpr = arm_direct.get("probes")
    labels = ["base", "std install", "dose s1600"] + (["direct s800"] if dpr else [])
    psets = [base_probes, std_probes,
             {"gen_prompts": dose_probes["s1600"]}] + ([dpr] if dpr else [])
    x = range(len(labels))
    ax.bar([i - 0.22 for i in x],
           [p["gen_prompts"]["p_z_mean"] for p in psets], 0.42,
           label="p(Z) @ 8 gen prompts", color="purple")
    ax.bar([i + 0.22 for i in x],
           [p.get("home_prompts", p["gen_prompts"])["p_z_mean"] for p in psets],
           0.42, label="p(Z) @ home prompts (direct only)", color="royalblue")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean p(Z) at onset")
    ax.legend(fontsize=7)
    ax.set_title("Onset probe: the prior at the address")

    ax = axes[1, 1]
    w = 0.35
    ax.bar([0 - w / 2, 1 - w / 2, 2 - w / 2],
           [arm_repro["s400"]["ro"]["r1i"]["acc"],
            arm_dose["s800"]["ro"]["r1i"]["acc"],
            arm_dose["s1600"]["ro"]["r1i"]["acc"]], w, color="crimson",
           label="host battery acc")
    ax.bar([0 + w / 2, 1 + w / 2, 2 + w / 2],
           [arm_repro["s400"]["ro"]["r1u"]["acc"],
            arm_dose["s800"]["ro"]["r1u"]["acc"],
            arm_dose["s1600"]["ro"]["r1u"]["acc"]], w, color="darkorange",
           label="uniform-floor acc")
    if "s800" in arm_direct:
        ax.bar([3 - w / 2], [arm_direct["s800"]["ro"]["r1i"]["acc"]], w,
               color="crimson")
        ax.bar([3 + w / 2], [arm_direct["s800"]["ro"]["r1u"]["acc"]], w,
               color="darkorange")
        ax.bar([3], [arm_direct["s800"]["ro"]["r1home"]["acc"]], 1.6 * w,
               color="royalblue", label="direct home acc", alpha=0.6)
    ax.axhline(base_ro["r1u"]["acc"], color="gray", ls=":", lw=1,
               label=f"base uniform floor {base_ro['r1u']['acc']:.3f}")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["std install", "dose s800", "dose s1600", "direct s800"],
                       fontsize=8)
    ax.set_ylabel("battery acc"); ax.legend(fontsize=7)
    ax.set_title("H-battery-overfit: host-anchored vs uniform-floor")
    fig.suptitle("E048 — the expression-gap boundary (installed != expressed)")
    fig.tight_layout()
    fig.savefig(rd / "expression_gap.png", dpi=130)
    plt.close(fig)

    log("=" * 70)
    log(f"P1 (teacher-forcing-bound, all install arms 0): {P1['confirmed']} {p1_counts}")
    log(f"P2 (prior/threshold unlock): {P2['confirmed']} "
        f"(seeded {P2['seeded']}, T {P2['T0.7']}/{P2['T1.0']}/{P2['T1.3']}, "
        f"greedy {P2['greedy_diagnostic']})")
    log(f"P3 (dose quantity): {P3['confirmed']} counts {P3['counts']} "
        f"battery {P3['battery_acc']}")
    log(f"H-battery-overfit: host {host_acc:.3f} vs uniform {unif_acc:.3f} "
        f"(ratio {hbat['ratio_unif_host']}, floor {hbat['uniform_acc_base_floor']:.3f}) "
        f"-> construction_bound {hbat['construction_bound']}")
    log(f"onset audit: p(Z|gen) {onset_audit['std_p_z_gen_prompts']:.3f} "
        f"argmaxZ {onset_audit['std_frac_argmax_z_gen_prompts']:.2f} "
        f"battery-130 {onset_audit['std_p_z_battery_ctx130']:.3f} "
        f"sampled-first-Z {onset_audit['sampled_first_char_Z_repro_unseeded']}/8 "
        f"context_binding {onset_audit['context_binding']} "
        f"sampling_anomaly {onset_audit['sampling_anomaly']}")
    log(f"outputs: {rd} | total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
