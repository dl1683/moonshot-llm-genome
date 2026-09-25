"""E055 DESIGN PROBES — measured CPU probes for the suppression localizer.

No training, no edits to lab files. Outputs printed to stdout (captured into
scratch/e055_design.md). Budget target <= 12 min CPU.

Probes:
  P1  battery TF onset on e048_repro (gate vs e048: p_z~0.556, argmaxZ~0.82)
  P2  position-vs-content binding curve (truncate-oldest / left-pad)
  P3  FREE-RUN FROM BATTERY GEOMETRY (ctx130): the number e048 never measured
  P4  free-run trajectories (e048 global-RNG seeds -> repro gate) -> onset
      sites: sub-argmax p(Z)/rank(Z) at each incumbent-name onset + TF
      completion-in-own-context + direct800 reference
  P5  free-vs-battery state distance per depth (the write size)
  P6a MINI-TRANSPLANT (cross-context): battery TF state at donor decision
      pos -> trajectory onset decision pos, depths {0,2,3,4,6}, TF vs
      shuffled donors; readout p(Z) at next position + continued free-run
  P6b MINI-TRANSPLANT (shared-prefix): TF state at onset token -> diverged
      free-run stream; readout p(correct 2nd char)
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # CPU-safe design work

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lab"))

import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402

import common                      # noqa: E402
common.DEVICE = "cpu"

from common import Cfg, CharCorpus, TinyGPT   # noqa: E402
import e043_install as E43                    # noqa: E402  (constants only)

T0 = time.time()
log = lambda m: print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]

# ------------------------------------------------------------------ corpus + nets
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
    return m, (st.get("step") if isinstance(st, dict) else None)


CK = REPO / "runs" / "checkpoints"
net_repro, s_repro = load(CK / "e048_repro.pt")
net_base, _ = load(CK / "e001.pt")
net_d800, _ = load(CK / "e048_direct800.pt")
log(f"loaded repro(step={s_repro}) base d800; params {net_repro.num_params():,}")

# ------------------------------------------------------------------ protocol rebuild (e048 verbatim)
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
               + [train_text[p - 120: p] for p, _ in held_occ[:4]])
ctx130_i = [train_text[p - PRE: p] for p, _ in install_occ]


# ------------------------------------------------------------------ instruments
@torch.no_grad()
def onset(net, contexts):
    """p(Z), argmax, rank(Z), p(argmax) for next char after each context."""
    out = []
    for c in contexts:
        idx = corpus.encode(c).unsqueeze(0)
        logits, _ = net(idx)
        lg = logits[0, -1]
        pr = F.softmax(lg, -1)
        rank = int((lg > lg[zid]).sum()) + 1
        out.append({"p_z": float(pr[zid]), "rank_z": rank,
                    "argmax": itos[int(lg.argmax())],
                    "p_argmax": float(pr.max())})
    return out


def stats(rows):
    pz = [r["p_z"] for r in rows]
    return {"n": len(rows), "p_z_mean": sum(pz) / len(pz),
            "p_z_max": max(pz),
            "frac_argmax_z": sum(r["argmax"] == "Z" for r in rows) / len(rows),
            "mean_rank_z": sum(r["rank_z"] for r in rows) / len(rows)}


@torch.no_grad()
def states_forward(net, toks):
    """toks: 1D tensor. Returns (xs, logits): xs[0]=emb, xs[k]=block k-1 output."""
    toks = toks.unsqueeze(0)
    T = toks.shape[1]
    x = net.wte(toks) + net.wpe(torch.arange(T))
    xs = [x]
    for block in net.h:
        x = block(x)
        xs.append(x)
    logits = net.lm_head(net.ln_f(x))
    return xs, logits


@torch.no_grad()
def patched_logits(net, toks, pos, depth, state):
    """Overwrite residual at `pos` at depth d (0=embedding/input of block 0,
    k=output of block k / input of block k+1, 6=final residual pre-ln_f)."""
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
    """Replicates common.generate on the GLOBAL torch RNG (call torch.manual_seed
    first to reproduce e048 generations). Returns new 1D ids."""
    idx = ctx_ids.clone()
    for _ in range(n_new):
        idx_cond = idx[-net.cfg.block_size:]
        logits, _ = net(idx_cond.unsqueeze(0))
        lg = logits[0, -1] / temperature
        v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
        lg[lg < v[-1]] = -float("inf")
        pr = F.softmax(lg, -1)
        idx = torch.cat([idx, torch.multinomial(pr, 1)])
    return idx[len(ctx_ids):]


# ================================================================== P1
log("P1 battery TF onset (gate vs e048: repro 0.556/0.817/rank~1.2)")
rows_bat = onset(net_repro, ctx130_i)
p1 = {"repro": stats(rows_bat),
      "base": stats(onset(net_base, ctx130_i)),
      "d800": stats(onset(net_d800, ctx130_i))}
for t, s in p1.items():
    log(f"  {t:6s} p(Z) {s['p_z_mean']:.4f} argmaxZ {s['frac_argmax_z']:.3f} "
        f"rankZ {s['mean_rank_z']:.2f} max {s['p_z_max']:.3f}")

# ================================================================== P2
log("P2 position-vs-content binding (repro): same 60 occurrences, context length varies")
bind = {}
for L in (110, 120, 125, 129, 130, 131, 140):
    ctxs = [train_text[p - L: p] for p, _ in install_occ]
    bind[f"trunc{L}"] = stats(onset(net_repro, ctxs))
for k in (5, 10):
    ctxs = [" " * k + train_text[p - PRE: p] for p, _ in install_occ]
    bind[f"pad{k}"] = stats(onset(net_repro, ctxs))
for k in sorted(bind):
    s = bind[k]
    log(f"  {k:8s} p(Z) {s['p_z_mean']:.4f} argmaxZ {s['frac_argmax_z']:.3f} "
        f"rankZ {s['mean_rank_z']:.2f}")

# ================================================================== P3
log("P3 free-run from ctx130 (60 ctx; greedy + 3 samples x 40 chars) — never measured")
first_chars = {}
zp_from_bat = 0
greedy_first_z = 0
greedy_complete = 0
n_samp = 0
for i, c in enumerate(ctx130_i):
    ids = corpus.encode(c)
    torch.manual_seed(300 + i)
    out = sample_from(net_repro, ids, 7, temperature=1e-9, top_k=1)
    gtxt = corpus.decode(out.tolist())
    greedy_first_z += (gtxt[0] == "Z")
    greedy_complete += (gtxt == NAME)
    for s in range(3):
        torch.manual_seed(1000 + 17 * i + s)
        o = sample_from(net_repro, ids, 40)
        txt = corpus.decode(o.tolist())
        first_chars[txt[0]] = first_chars.get(txt[0], 0) + 1
        zp_from_bat += len(re.findall(r"(?<![A-Za-z])ZEPHYRA(?![A-Za-z])", txt))
        n_samp += 1
log(f"  greedy: first-char Z {greedy_first_z}/60, full ZEPHYRA {greedy_complete}/60")
log(f"  sampled: ZEPHYRA {zp_from_bat} in {n_samp * 40} chars; "
    f"first-char dist {dict(sorted(first_chars.items(), key=lambda kv: -kv[1]))}")

# ================================================================== P4
log("P4 free-run trajectories (e048 seeds, global RNG) -> incumbent onset sites")


def trajectory(net, prompt, seed, n=350):
    torch.manual_seed(seed)
    ids = corpus.encode(prompt)
    out = sample_from(net, ids, n)
    return prompt + corpus.decode(out.tolist())


traj_repro = [trajectory(net_repro, gen_prompts[i], i) for i in range(8)]
cnt = {w: sum(len(re.findall(r"(?<![A-Za-z])" + w + r"(?![A-Za-z])",
                             t[len(gen_prompts[i]):]))
             for i, t in enumerate(traj_repro))
       for w in ("ZEPHYRA", "ELIZABETH", "FLORIZEL")}
log(f"  repro traj counts (e048 gate: Z0 E7 F2): {cnt}")

sites = []
for i, t in enumerate(traj_repro):
    for m in re.finditer(r"(?<![A-Za-z])(ELIZABETH|FLORIZEL)(?![A-Za-z])",
                         t[len(gen_prompts[i]):]):
        sites.append({"prompt": i, "t": len(gen_prompts[i]) + m.start(),
                      "name": m.group(1),
                      "ctx": t[:len(gen_prompts[i]) + m.start()]})
log(f"  onset sites: {len(sites)} {[(s['prompt'], s['t'], s['name']) for s in sites]}")

sub = []
for s in sites:
    o = onset(net_repro, [s["ctx"]])[0]
    ids = torch.cat([corpus.encode(s["ctx"]), name_ids])
    lg, _ = net_repro(ids.unsqueeze(0))
    pos0 = len(s["ctx"])
    tfp = []
    for j in range(len(NAME)):
        pr = F.softmax(lg[0, pos0 - 1 + j], -1)
        tfp.append(float(pr[name_ids[j]]))
    s.update(o)
    s["tf_complete"] = tfp
    sub.append(s)
for s in sub:
    log(f"  site p{s['prompt']} t={s['t']:3d} {s['name']:9s} p(Z) {s['p_z']:.4f} "
        f"rankZ {s['rank_z']:2d} argmax {s['argmax']}({s['p_argmax']:.2f}) "
        f"TFcomp[{' '.join(f'{p:.2f}' for p in s['tf_complete'])}]")

# trajectory floor: p(Z) at 120-char windows NOT at name onsets (20 per traj)
floor = []
for i, t in enumerate(traj_repro):
    cont = t[len(gen_prompts[i]):]
    taken = 0
    for q in range(20, len(cont) - 20, 7):
        if any(abs((q + len(gen_prompts[i])) - s["t"]) < 15 for s in sub
               if s["prompt"] == i):
            continue
        floor.append(onset(net_repro, [t[len(t) - 0 - (q + 120):q + 120]
                                       if False else cont[max(0, q - 120):q]])[0]["p_z"])
        taken += 1
        if taken >= 3:
            break
if floor:
    floor.sort()
    log(f"  trajectory floor p(Z) (n={len(floor)}): median {floor[len(floor)//2]:.2e} "
        f"max {floor[-1]:.2e}")

# direct800 reference trajectories + onset p(Z)
traj_d8 = [trajectory(net_d800, gen_prompts[i], i) for i in range(8)]
cntd = {}
for w in ("ZEPHYRA", "ELIZABETH", "FLORIZEL"):
    cntd[w] = sum(len(re.findall(r"(?<![A-Za-z])" + w + r"(?![A-Za-z])",
                                 t[len(gen_prompts[i]):]))
                  for i, t in enumerate(traj_d8))
d8_pz = []
for i, t in enumerate(traj_d8):
    for m in re.finditer(r"(?<![A-Za-z])(ELIZABETH|FLORIZEL)(?![A-Za-z])",
                         t[len(gen_prompts[i]):]):
        ctx = t[:len(gen_prompts[i]) + m.start()]
        d8_pz.append(onset(net_d800, [ctx])[0])
log(f"  d800 traj counts: {cntd}")
if d8_pz:
    log("  d800 onset p(Z): " + " ".join(f"{r['p_z']:.4f}/r{r['rank_z']}" for r in d8_pz))

# ================================================================== P5 donors + state distances
log("P5 donors (battery TF states) + shuffled (random corpus prefixes) + distances")
order = sorted(range(60), key=lambda i: -rows_bat[i]["p_z"])
donor_idx = order[:2] + [i for i in order[2:] if install_occ[i][1] == "FLORIZEL"][:2] + [order[2]]
donor_idx = list(dict.fromkeys(donor_idx))[:5]
donor_states = []
for i in donor_idx:
    xs, lg = states_forward(net_repro, corpus.encode(ctx130_i[i]))
    pr = F.softmax(lg[0, -1], -1)
    donor_states.append({"xs": [x[0, -1].clone() for x in xs],
                         "p_z": float(pr[zid]), "argmax": itos[int(lg[0, -1].argmax())],
                         "host": install_occ[i][1], "ctx_i": i})
for d in donor_states:
    log(f"  donor ctx{d['ctx_i']:2d} ({d['host']:9s}) p(Z) {d['p_z']:.3f} "
        f"argmax {d['argmax']}")

import random as _r2
_s2 = _r2.Random(25501)
shuf_ctxs = []
while len(shuf_ctxs) < 5:
    q = _s2.randrange(PRE + 1, len(train_ids) - 1)
    shuf_ctxs.append(train_text[q - PRE: q])
shuf_states = []
for c in shuf_ctxs:
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
    log(f"  site t={s['t']} rel.dist(free vs donor0)/depth: "
        + " ".join(f"d{d}:{v:.2f}" for d, v in enumerate(dists)))

# ================================================================== P6a
log("P6a MINI-TRANSPLANT cross-context: battery state -> onset decision pos (t-1)")
log("      readout: p(Z) at next position (the onset choice)")
mini = []
for s in sub:
    crop = corpus.encode(s["ctx"])[-256:]
    dpos = len(crop) - 1
    _, lg0 = states_forward(net_repro, crop)
    base_pz = float(F.softmax(lg0[0, -1], -1)[zid])
    row = {"site": (s["prompt"], s["t"], s["name"]), "base_pz": base_pz, "depths": {}}
    for d in (0, 2, 3, 4, 6):
        tfp = [float(F.softmax(patched_logits(net_repro, crop, dpos, d,
                                              ds["xs"][d].unsqueeze(0))[0, -1], -1)[zid])
               for ds in donor_states]
        shp = [float(F.softmax(patched_logits(net_repro, crop, dpos, d,
                                              ds["xs"][d].unsqueeze(0))[0, -1], -1)[zid])
               for ds in shuf_states]
        row["depths"][d] = {"tf_mean": sum(tfp) / len(tfp), "tf_max": max(tfp),
                            "tf_vals": [round(v, 3) for v in tfp],
                            "shuf_mean": sum(shp) / len(shp), "shuf_max": max(shp)}
    mini.append(row)
    log(f"  site {row['site']} base {base_pz:.4f} | " + " | ".join(
        f"d{d}: TF {v['tf_mean']:.3f}(max {v['tf_max']:.3f}) shuf {v['shuf_mean']:.3f}"
        for d, v in row["depths"].items()))

# machinery gates
crop = corpus.encode(sub[0]["ctx"])[-256:]
dpos = len(crop) - 1
xs, lg0 = states_forward(net_repro, crop)
for d in (0, 3, 6):
    lg = patched_logits(net_repro, crop, dpos, d, xs[d][0, dpos].unsqueeze(0))
    assert float((lg[0, -1] - lg0[0, -1]).abs().max()) < 1e-4
log("  gate self-patch==baseline (<1e-4): PASS")
lg = patched_logits(net_repro, crop, dpos, 6, donor_states[0]["xs"][6].unsqueeze(0))
pz6 = float(F.softmax(lg[0, -1], -1)[zid])
log(f"  gate d6 patch reproduces donor onset p(Z): {pz6:.4f} vs "
    f"{donor_states[0]['p_z']:.4f}")

# continuation (write HELD at the site through generation; see doc)
log("P6a-cont: continued free-run, write held (80 chars x 4 samples x 3 sites)")
cont_rows = []
for s in sub[:3]:
    crop = corpus.encode(s["ctx"])[-256:]
    dpos = len(crop) - 1
    res = {}
    for arm in ("base", "tf_d0", "tf_d3", "shuf_d3"):
        donor = None if arm == "base" else (donor_states[0] if arm.startswith("tf")
                                            else shuf_states[0])
        d = 0 if arm.endswith("d0") else 3
        zp, fz, pz_any = 0, 0, []
        for k in range(4):
            torch.manual_seed(7000 + 97 * k)
            ctx_ids = crop.clone()
            new_ids = []
            for _ in range(80):
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
                new_ids.append(nxt)
                ctx_ids = torch.cat([ctx_ids, torch.tensor([nxt])])
            txt = corpus.decode(new_ids)
            zp += len(re.findall(r"(?<![A-Za-z])ZEPHYRA(?![A-Za-z])", txt))
            fz += (txt[:1] == "Z")
            pz_any.append(txt.count("Z"))
        res[arm] = {"zephyra": zp, "first_z": fz, "z_chars": sum(pz_any)}
    cont_rows.append({"site": (s["prompt"], s["t"]), **res})
    log(f"  site p{s['prompt']} t={s['t']}: {res}")

# ================================================================== P6b
log("P6b shared-prefix: forced divergence at onset, transplant TF state @130, p(E@131)")
sp_rows = []
argz = [i for i in range(60) if rows_bat[i]["argmax"] == "Z"][:10]
for i in argz:
    c = ctx130_i[i]
    base_ids = torch.cat([corpus.encode(c), torch.tensor([stoi["E"]])])
    tf_ids = torch.cat([corpus.encode(c), torch.tensor([zid])])
    xs_tf, _ = states_forward(net_repro, tf_ids)
    lg0, _ = net_repro(base_ids.unsqueeze(0))
    row = {"ctx": i, "base": float(F.softmax(lg0[0, -1], -1)[stoi["E"]]), "depths": {}}
    for d in (0, 2, 4, 6):
        lg = patched_logits(net_repro, base_ids, len(base_ids) - 1, d,
                            xs_tf[d][0, -1].unsqueeze(0))
        row["depths"][d] = float(F.softmax(lg[0, -1], -1)[stoi["E"]])
    sp_rows.append(row)
log("  baseline p(E|ctx+'E') at 131: " + " ".join(f"{r['base']:.3f}" for r in sp_rows))
for d in (0, 2, 4, 6):
    vals = [r["depths"][d] for r in sp_rows]
    log(f"  depth {d}: p(E@131|transplant) mean {sum(vals)/len(vals):.3f} "
        + " ".join(f"{v:.3f}" for v in vals))

log("PROBES DONE")
