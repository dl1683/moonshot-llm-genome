"""E055 DESIGN PROBES part 2 — continuation after e055_probe.py P1-P4(part).

P1-P3 measured (see design doc). This script redoes P4 with the >256-context
fix (crop to last 256, exactly what generate() sees), then P5/P6.
Trajectories are cached to scratch/e055_traj_cache.pt.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lab"))

import torch
torch.set_num_threads(16)

import torch.nn.functional as F

import common
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT
import e043_install as E43

T0 = time.time()
log = lambda m: print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]

corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
cfg = Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192, block_size=256)
stoi, itos = corpus.stoi, corpus.itos
zid = stoi["Z"]
train_ids = corpus.train
train_text = "".join(itos[int(i)] for i in train_ids)


def load(path):
    m = TinyGPT(cfg)
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


CK = REPO / "runs" / "checkpoints"
net_repro = load(CK / "e048_repro.pt")
net_base = load(CK / "e001.pt")
net_d800 = load(CK / "e048_direct800.pt")
log("nets loaded (repro step-400 ckpt, base, direct800)")

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
gen_prompts = ([train_text[p - 120: p] for p, _ in install_occ[:4]]
               + [train_text[p - 120: p] for p, _ in held_occ[:4]])
ctx130_i = [train_text[p - PRE: p] for p, _ in install_occ]

# ------------------------------------------------------------------ instruments
@torch.no_grad()
def onset_ctx(net, ctx):
    """onset on the model's own view: crop to last 256 (what generate sees)."""
    ids = corpus.encode(ctx)[-net.cfg.block_size:]
    logits, _ = net(ids.unsqueeze(0))
    lg = logits[0, -1]
    pr = F.softmax(lg, -1)
    rank = int((lg > lg[zid]).sum()) + 1
    eid = stoi.get(ctx[-1], 0)
    pr_last = F.softmax(logits[0, -2], -1) if ids.shape[0] > 1 else None
    return {"p_z": float(pr[zid]), "rank_z": rank,
            "argmax": itos[int(lg.argmax())], "p_argmax": float(pr.max()),
            "crop_len": int(ids.shape[0])}


@torch.no_grad()
def states_forward(net, toks):
    toks = toks.unsqueeze(0)
    T = toks.shape[1]
    x = net.wte(toks) + net.wpe(torch.arange(T))
    xs = [x]
    for block in net.h:
        x = block(x)
        xs.append(x)
    return xs, net.lm_head(net.ln_f(x))


@torch.no_grad()
def patched_logits(net, toks, pos, depth, state):
    toks = toks.unsqueeze(0)
    T = toks.shape[1]
    x = net.wte(toks) + net.wpe(torch.arange(T))
    if depth == 0:
        x = x.clone()
        x[:, pos] = state
    for i, block in enumerate(net.h):
        x = block(x)
        if i + 1 == depth:
            x = x.clone()
            x[:, pos] = state
    return net.lm_head(net.ln_f(x))


@torch.no_grad()
def sample_from(net, ctx_ids, n_new, temperature=0.8, top_k=40):
    idx = ctx_ids.clone()
    for _ in range(n_new):
        idx_cond = idx[-net.cfg.block_size:]
        logits, _ = net(idx_cond.unsqueeze(0))
        lg = logits[0, -1] / temperature
        v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
        lg[lg < v[-1]] = -float("inf")
        idx = torch.cat([idx, torch.multinomial(F.softmax(lg, -1), 1)])
    return idx[len(ctx_ids):]


CACHE = REPO / "scratch" / "e055_traj_cache.pt"
if CACHE.exists():
    tr = torch.load(CACHE, weights_only=False)
    traj_repro = tr["repro"]
    conts_repro = tr["conts"]
    log(f"trajectories from cache ({len(traj_repro)} repro)")
else:
    traj_repro, conts_repro = [], []
    for i in range(8):
        torch.manual_seed(i)
        ids = corpus.encode(gen_prompts[i])
        out = sample_from(net_repro, ids, 350)
        cont = corpus.decode(out.tolist())
        traj_repro.append(gen_prompts[i] + cont)
        conts_repro.append(cont)
    torch.save({"repro": traj_repro, "conts": conts_repro}, CACHE)
    log("trajectories generated + cached")

cnt = {w: sum(len(re.findall(r"(?<![A-Za-z])" + w + r"(?![A-Za-z])", c))
       for c in conts_repro) for w in ("ZEPHYRA", "ELIZABETH", "FLORIZEL")}
log(f"P4 repro traj counts (e048 ref: Z0 E7 F2): {cnt}")

sites = []
for i, cont in enumerate(conts_repro):
    for m in re.finditer(r"(?<![A-Za-z])(ELIZABETH|FLORIZEL)(?![A-Za-z])", cont):
        t = len(gen_prompts[i]) + m.start()
        sites.append({"prompt": i, "t": t, "name": m.group(1),
                      "ctx": traj_repro[i][:t]})
log(f"onset sites: {len(sites)} {[(s['prompt'], s['t'], s['name']) for s in sites]}")

# ---- P4 sub-argmax table + TF completion in own context
sub = []
for s in sites:
    o = onset_ctx(net_repro, s["ctx"])
    ids = torch.cat([corpus.encode(s["ctx"])[-249:], name_ids])
    lg, _ = net_repro(ids.unsqueeze(0))
    pos0 = ids.shape[0] - len(NAME)
    tfp = [float(F.softmax(lg[0, pos0 - 1 + j], -1)[name_ids[j]])
           for j in range(len(NAME))]
    s.update(o)
    s["tf_complete"] = tfp
    sub.append(s)
for s in sub:
    log(f"  site p{s['prompt']} t={s['t']:3d} {s['name']:9s} p(Z) {s['p_z']:.4f} "
        f"rankZ {s['rank_z']:2d} argmax {s['argmax']}({s['p_argmax']:.2f}) "
        f"crop {s['crop_len']:3d} TFcomp[{' '.join(f'{p:.2f}' for p in s['tf_complete'])}]")

# ---- trajectory floor: p(Z) at 100 random non-onset positions
import random as _r3
fl = _r3.Random(9)
floor = []
for i, cont in enumerate(conts_repro):
    for _ in range(4):
        q = fl.randrange(30, len(cont) - 30)
        if any(abs((q + len(gen_prompts[i])) - s["t"]) < 15 for s in sub
               if s["prompt"] == i):
            continue
        floor.append(onset_ctx(net_repro, traj_repro[i][:len(gen_prompts[i]) + q])["p_z"])
floor.sort()
if floor:
    log(f"  trajectory floor p(Z) n={len(floor)}: median {floor[len(floor)//2]:.2e} "
        f"p90 {floor[int(len(floor)*0.9)]:.2e} max {floor[-1]:.2e}")

# ---- direct800 reference (4 prompts to bound time)
d8_conts = []
for i in range(4):
    torch.manual_seed(i)
    d8_conts.append(corpus.decode(
        sample_from(net_d800, corpus.encode(gen_prompts[i]), 350).tolist()))
cntd = {w: sum(len(re.findall(r"(?<![A-Za-z])" + w + r"(?![A-Za-z])", c))
        for c in d8_conts) for w in ("ZEPHYRA", "ELIZABETH", "FLORIZEL")}
d8_sites = []
for i, cont in enumerate(d8_conts):
    for m in re.finditer(r"(?<![A-Za-z])(ELIZABETH|FLORIZEL)(?![A-Za-z])", cont):
        t = 120 + m.start()
        d8_sites.append(onset_ctx(net_d800, (gen_prompts[i] + cont)[:t]))
log(f"  d800 traj counts (4 prompts, e048 8-prompt ref Z0 E2 F0): {cntd}")
if d8_sites:
    log("  d800 onset p(Z): " + " ".join(
        f"{r['p_z']:.4f}/r{r['rank_z']}/{r['argmax']}" for r in d8_sites))

# ================================================================== P5 donors
log("P5 donors + shuffled + state distances")
@torch.no_grad()
def bat_onset(i):
    ids = corpus.encode(ctx130_i[i])
    lg, _ = net_repro(ids.unsqueeze(0))
    pr = F.softmax(lg[0, -1], -1)
    return {"p_z": float(pr[zid]), "argmax": itos[int(lg[0, -1].argmax())]}


bat_rows = [bat_onset(i) for i in range(60)]
order = sorted(range(60), key=lambda i: -bat_rows[i]["p_z"])
donor_idx = order[:2] + [i for i in order[2:] if install_occ[i][1] == "FLORIZEL"][:2]
donor_idx = list(dict.fromkeys(donor_idx))[:4]
donor_states = []
for i in donor_idx:
    xs, lg = states_forward(net_repro, corpus.encode(ctx130_i[i]))
    donor_states.append({"xs": [x[0, -1].clone() for x in xs],
                         "p_z": bat_rows[i]["p_z"], "argmax": bat_rows[i]["argmax"],
                         "host": install_occ[i][1], "ctx_i": i})
    log(f"  donor ctx{i:2d} ({install_occ[i][1]:9s}) p(Z) {bat_rows[i]['p_z']:.3f} "
        f"argmax {bat_rows[i]['argmax']}")

import random as _r2
_s2 = _r2.Random(25501)
shuf_states = []
while len(shuf_states) < 4:
    q = _s2.randrange(PRE + 1, len(train_ids) - 1)
    c = train_text[q - PRE: q]
    xs, lg = states_forward(net_repro, corpus.encode(c))
    pr = F.softmax(lg[0, -1], -1)
    shuf_states.append({"xs": [x[0, -1].clone() for x in xs],
                        "p_z": float(pr[zid])})
log("  shuffled donors p(Z): " + " ".join(f"{d['p_z']:.2e}" for d in shuf_states))

for s in sub[:3]:
    crop = corpus.encode(s["ctx"])[-256:]
    dpos = len(crop) - 1
    xs, _ = states_forward(net_repro, crop)
    dists = [float((xs[d][0, dpos] - donor_states[0]["xs"][d]).norm()
                   / donor_states[0]["xs"][d].norm()) for d in range(7)]
    log(f"  site t={s['t']} rel.dist(free vs donor0): "
        + " ".join(f"d{d}:{v:.2f}" for d, v in enumerate(dists)))

# ================================================================== P6a
log("P6a MINI-TRANSPLANT cross-context (battery state -> onset decision pos)")
mini = []
for s in sub:
    crop = corpus.encode(s["ctx"])[-256:]
    dpos = len(crop) - 1
    _, lg0 = states_forward(net_repro, crop)
    base_pz = float(F.softmax(lg0[0, -1], -1)[zid])
    row = {"site": (s["prompt"], s["t"], s["name"]), "base_pz": base_pz, "depths": {}}
    for d in (0, 1, 2, 3, 4, 5, 6):
        tfp = [float(F.softmax(patched_logits(net_repro, crop, dpos, d,
                                              ds["xs"][d].unsqueeze(0))[0, -1], -1)[zid])
               for ds in donor_states]
        shp = [float(F.softmax(patched_logits(net_repro, crop, dpos, d,
                                              ds["xs"][d].unsqueeze(0))[0, -1], -1)[zid])
               for ds in shuf_states]
        row["depths"][d] = {"tf_mean": sum(tfp) / len(tfp), "tf_max": max(tfp),
                            "shuf_mean": sum(shp) / len(shp), "shuf_max": max(shp)}
    mini.append(row)
    log(f"  site {row['site']} base {base_pz:.4f} | " + " | ".join(
        f"d{d}:TF {v['tf_mean']:.3f}/{v['tf_max']:.3f} sh {v['shuf_mean']:.3f}"
        for d, v in row["depths"].items()))

crop = corpus.encode(sub[0]["ctx"])[-256:]
dpos = len(crop) - 1
xs, lg0 = states_forward(net_repro, crop)
for d in (0, 3, 6):
    lg = patched_logits(net_repro, crop, dpos, d, xs[d][0, dpos].unsqueeze(0))
    assert float((lg[0, -1] - lg0[0, -1]).abs().max()) < 1e-4
log("  gate self-patch==baseline: PASS")
lg = patched_logits(net_repro, crop, dpos, 6, donor_states[0]["xs"][6].unsqueeze(0))
pz6 = float(F.softmax(lg[0, -1], -1)[zid])
log(f"  gate d6 patch reproduces donor onset p(Z): {pz6:.4f} vs "
    f"{donor_states[0]['p_z']:.4f}")

# base net control: same transplant on the UNTRAINED-for-Z base net
log("P6a-base control: same patch on base B (expect no rescue anywhere)")
crop = corpus.encode(sub[0]["ctx"])[-256:]
dpos = len(crop) - 1
row = {}
for d in (0, 2, 3, 4, 6):
    tfp = [float(F.softmax(patched_logits(net_base, crop, dpos, d,
                                          ds["xs"][d].unsqueeze(0))[0, -1], -1)[zid])
           for ds in donor_states]
    row[d] = sum(tfp) / len(tfp)
log("  base-net patched p(Z): " + " ".join(f"d{d}:{v:.2e}" for d, v in row.items()))

# ---- P6a-cont: write HELD at the site, continue free-run (2 sites x 4 arms x 2 x 60)
log("P6a-cont: continued free-run, write held at site (60 chars x 2 samples)")
for s in sub[:2]:
    crop = corpus.encode(s["ctx"])[-256:]
    dpos = len(crop) - 1
    res = {}
    for arm in ("base", "tf_d0", "tf_d3", "shuf_d3"):
        donor = None if arm == "base" else (donor_states[0] if arm.startswith("tf")
                                            else shuf_states[0])
        d = 0 if arm.endswith("d0") else 3
        zp, fz, zch = 0, 0, 0
        for k in range(2):
            torch.manual_seed(7000 + 97 * k)
            ctx_ids = crop.clone()
            txt = []
            for _ in range(60):
                cc = ctx_ids[-256:]
                p = dpos - (len(ctx_ids) - len(cc))
                if donor is not None and p >= 0:
                    lg = patched_logits(net_repro, cc, p, d,
                                        donor["xs"][d].unsqueeze(0))
                else:
                    lg, _ = net_repro(cc.unsqueeze(0))
                lg = lg[0, -1] / 0.8
                v, _ = torch.topk(lg, 40)
                lg[lg < v[-1]] = -float("inf")
                nxt = int(torch.multinomial(F.softmax(lg, -1), 1))
                txt.append(itos[nxt])
                ctx_ids = torch.cat([ctx_ids, torch.tensor([nxt])])
            t = "".join(txt)
            zp += len(re.findall(r"(?<![A-Za-z])Z[A-Za-z]*", t))
            fz += (t[:1] == "Z")
            zch += t.count("Z")
        res[arm] = {"z_words": zp, "first_z": fz, "z_chars": zch}
    log(f"  site p{s['prompt']} t={s['t']}: {res}")

# ================================================================== P6b
log("P6b shared-prefix: ctx130+'E' stream, transplant TF(ctx130+'Z') state @130")
sp_rows = []
argz = [i for i in range(60) if bat_rows[i]["argmax"] == "Z"][:10]
for i in argz:
    c = ctx130_i[i]
    base_ids = torch.cat([corpus.encode(c), torch.tensor([stoi["E"]])])
    tf_ids = torch.cat([corpus.encode(c), torch.tensor([zid])])
    xs_tf, _ = states_forward(net_repro, tf_ids)
    lg0, _ = net_repro(base_ids.unsqueeze(0))
    row = {"ctx": i, "base": float(F.softmax(lg0[0, -1], -1)[stoi["E"]]), "depths": {}}
    for d in (0, 2, 3, 4, 6):
        lg = patched_logits(net_repro, base_ids, len(base_ids) - 1, d,
                            xs_tf[d][0, -1].unsqueeze(0))
        row["depths"][d] = float(F.softmax(lg[0, -1], -1)[stoi["E"]])
    sp_rows.append(row)
log("  baseline p(E@131 | ctx130+'E'): mean "
    f"{sum(r['base'] for r in sp_rows)/len(sp_rows):.3f}")
for d in (0, 2, 3, 4, 6):
    vals = [r["depths"][d] for r in sp_rows]
    log(f"  depth {d}: p(E@131|TF transplant) mean {sum(vals)/len(vals):.3f} "
        + " ".join(f"{v:.2f}" for v in vals))

log("PROBES-2 DONE")
