"""G5 PREFLIGHT (eval-only, CPU, stored g3 checkpoints; no training).

The design-critical questions before writing lab/g5_cone_wall.py:
  Q1 ROUTE — is the washed host's fact route dead only at +1 (the Adam
      overshoot transient) or also at +10/+50/+300? (g3's closure ran at
      t*=+1 only: root-store-into-washed-host 6.4e-5.)
  Q2 WALL END-STATE — approximating the W_q wall's terminal state by
      RESTORING root W_q inside the washed states (root W_q + washed K/V/W_o
      + washed host): what does g0 read? Same for the whole-store wall
      (= g3's bypass probe, root store + washed host).
  Q3 THE R_q CURRENCY — the W_q-ONLY lambda sweep along the measured +1
      wash direction (root everything else): where does g0 cross the bars?
      This is the dial the wall's radius must sit inside.

Run:  python scratch/g5_preflight.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab"))

import torch                                        # noqa: E402
import torch.nn.functional as F                     # noqa: E402

import common                                       # noqa: E402
from common import CharCorpus                       # noqa: E402

import e043_install as E43                          # noqa: E402
import g3_generative_store as G3                    # noqa: E402

common.DEVICE = "cpu"
CK = E43.REPO / "runs" / "checkpoints"

# ---- protocol rebuild (g3's main VERBATIM, batteries only) -----------------
corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
stoi, itos = corpus.stoi, corpus.itos
zid = stoi["Z"]
train_ids = corpus.train
train_text = "".join(itos[int(i)] for i in train_ids)

host_occ = []
for host in G3.HOSTS:
    for p in E43.find_occ(train_text, host):
        if p >= 280 and p + len(host) + G3.POST_CAP <= len(train_ids):
            host_occ.append((p, host))
rng = random.Random(E43.SPLICE_RNG)
rng.shuffle(host_occ)
install_occ = host_occ[:60]

bat = {}
for j in G3.GEOS:
    cs = [train_text[p - G3.PRE - j: p] for p, _ in install_occ]
    bat[j] = torch.stack([corpus.encode(c) for c in cs])
ids130 = bat[0]
name_ids = corpus.encode(G3.NAME)

# e143's offset-pool construction (g3's offset_pool(0), VERBATIM)
wins = []
for p, h in install_occ:
    pre = train_ids[p - G3.PRE: p]
    post = train_ids[p + len(h): p + len(h) + G3.POST_CAP]
    wins.append(torch.cat([pre, name_ids, post]))
pool_band_x = torch.stack(wins)


def retrieval_probe(sd_store, sd_q_src):
    """a_fact for the store in sd_store but with queries recomputed through
    the W_q of sd_q_src (root), on fact rows 129..135."""
    net_q = G3.evl_load("gen", sd_q_src)
    net_q.store.record = True
    _ = net_q(pool_band_x[:, :-1])
    q = net_q.store.cache["q"].clone()
    net_q.store.record = False
    net_s = G3.evl_load("gen", sd_store)
    K = F.normalize(net_s.store.K.detach(), dim=-1)
    fact_rows = slice(G3.PRE - 1, G3.PRE + 6)
    pat = torch.arange(7).unsqueeze(0).expand(q.shape[0], -1)
    a = F.softmax(G3.BETA * (F.normalize(q[:, fact_rows], dim=-1) @ K.t()),
                  dim=-1)
    return float(a.gather(-1, pat.unsqueeze(-1)).mean())


def load(name):
    st = torch.load(CK / name, map_location="cpu", weights_only=False)
    return {k: v.clone() for k, v in st["model"].items()}


sd_root = load("g3_gen.pt")
skeys = G3.store_keys("gen")
WQ = "store.W_q.weight"

print(f"root g0 = {G3.battery_cell(G3.evl_load('gen', sd_root), ids130, zid)['mean_pz']:.4f}")

results = {}
for s in (10, 50, 300):
    sd_w = load(f"g3_gen_s{s}.pt")
    g0_w = G3.battery_cell(G3.evl_load("gen", sd_w), ids130, zid)["mean_pz"]
    # bypass / whole-store-wall end-state: root store into washed host
    sd_byp, g = G3.restore_class(sd_w, sd_root, skeys, "store")
    assert g["pass"]
    g0_byp = G3.battery_cell(G3.evl_load("gen", sd_byp), ids130, zid)["mean_pz"]
    # W_q-wall end-state: root W_q only, washed K/V/W_o + washed host
    sd_wq, g = G3.restore_class(sd_w, sd_root, [WQ], "W_q")
    assert g["pass"]
    g0_wq = G3.battery_cell(G3.evl_load("gen", sd_wq), ids130, zid)["mean_pz"]
    # the walled store's retrieval: root-q x washed-K (washed V/W_o intact)
    a_fact = retrieval_probe(sd_wq, sd_root)
    # store-survival probe: {root W_q, washed K/V/W_o} into the ROOT host
    hkeys = [k for k in sd_w if k not in skeys]
    sd_sr, g = G3.restore_class(sd_root, sd_wq,
                                ["store.K", "store.V", "store.W_o.weight"],
                                "store_rest")
    assert g["pass"]
    g0_sr = G3.battery_cell(G3.evl_load("gen", sd_sr), ids130, zid)["mean_pz"]
    print(f"+{s}: g0 {g0_w:.5f} | bypass(root store) {g0_byp:.5f} | "
          f"root-W_q end-state {g0_wq:.5f} | root-q x washed-K a_fact "
          f"{a_fact:.4f} | walled-store-into-root-host {g0_sr:.4f}")
    results[s] = (g0_w, g0_byp, g0_wq, a_fact, g0_sr)

# ---- Q3: the W_q-only lambda sweep along the +1 wash direction -------------
sd_s1 = load("g3_gen_s1.pt")
print("\nW_q-ONLY lambda sweep along the +1 wash direction "
      "(root everything else):")
for lam in (0.125, 0.25, 0.5, 1.0, 2.0):
    sd_l = {k: v.clone() for k, v in sd_root.items()}
    sd_l[WQ] = sd_root[WQ] + lam * (sd_s1[WQ] - sd_root[WQ])
    g0 = G3.battery_cell(G3.evl_load("gen", sd_l), ids130, zid)["mean_pz"]
    dq = float((sd_l[WQ] - sd_root[WQ]).norm())
    print(f"  lam {lam:5.3f}  |dW_q| {dq:.4f}  g0 {g0:.4f}")
