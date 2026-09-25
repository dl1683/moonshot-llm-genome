"""E049 — The refrain threshold: at what repetition density does far-retrieval appear?

Motivated by T020's L4 erosion edge (16-token sufficiency erodes with scale:
far-value -0.001 at 2.7M -> +0.035 at 10M) and T007's far-value method.
E021 showed retrieval EXISTS when 100% of documents demand it (copy acc
99.96%, far-value +3.27 nats, retrieval head ID-mass 0.95). This experiment
maps the DENSITY boundary on naturalistic data: sharp threshold or graded?

DESIGN (registered before any run):
1. Four synthetic-naturalistic corpora (~620KB each) from Shakespeare filler.
   Documents = F1 filler (30-120c) + [if refrain doc] a distinctive refrain
   phrase R (24-32 chars, from a fixed pool of 10, e.g. "the night has been
   unkind to") + F2 filler gap (20-90c, i.e. >16 chars) + fixed completion C
   (8-18 chars, e.g. " us all.") + F3 filler (30-120c). Refrain inclusion is
   nested across corpora: p=60 -> i%5<3, p=20 -> i%5==0, p=5 -> i%20==0,
   p=0 -> never. All four corpora share the same doc-spec stream (rng 4910),
   so they are prefix-aligned; only refrain insertion differs. The refrain
   appears verbatim at distance >16 chars from its completion, so predicting
   C's content chars requires retrieving R from beyond the 16-char local
   window (or an implausibly strong filler-local n-gram).
2. Train a 2.7M net per corpus (B config: 6L/6H/192/ctx256, seed 42,
   steps=4000 lr=1e-3 batch=64 cosine, 252s wall cap, ckpt-resumable).
3. READOUTS per net:
   (a) far-value at refrain-completion positions = CE(trunc-16) - CE(full-256),
       per completion index k (k=0 is the leading space -> locally trivial;
       PRIMARY = prefix k=1..3, where the association must be retrieved).
       Own-corpus val events AND a shared probe set (p60 val events, same
       prompts for every net -> isolates learned knowledge from corpus
       sampling noise).
   (b) completion accuracy (top-1, full ctx) over k=1..3 and full span.
   (c) attention census at 40 shared completion prompts: per layer/head mass
       on the refrain SOURCE span (refrain-mass, e021 ID-mass style), plus a
       pre-refrain equal-length control span and far mass d>=17.
   (d) 16-token sufficiency at NON-refrain val positions (1500 random
       positions outside R and C spans, e013c machinery): does far-context
       value leak into ordinary text?
4. ONE extra 10M net (8L/8H/320, e005s config, seed 42, 300s cap) to test
   whether the threshold moves with scale.

REGISTERED PREDICTIONS:
  P1: retrieval appears between p=5% and p=60% at 2.7M: retrieval_shown :=
      fv_prefix(own-val) >= fv_prefix(p0 net, shared probes) + 1.0 nat AND
      acc_prefix(own-val) >= 0.5. P1 holds iff shown at 60% and NOT at 5%.
      (If shown at 5% too -> REFUTED-LOW, threshold <= 5%.)
  P2: the appearance is SHARP: max adjacent ratio of fv_prefix(own) over
      5->20 or 20->60 >= 10x (denominator < 0.05 nat counts as >= 10x) AND
      the absolute jump >= 1.0 nat. Else GRADED.
  P3: no leak — mean far-value at non-refrain positions < 0.1 nat at all p.
  P4 (conditional): threshold p* lower at 10M than 2.7M, tested directly:
      if p* in {20, 60}: 10M trained at the level BELOW p* (direct test —
      deviation from the card's literal "at p*", noted: the purpose clause
      "test whether the threshold moves" requires the below-threshold
      corpus); P4 holds iff the 10M net shows retrieval there.
      If p* = 5: 10M at p=5, P4 judged by weak magnitude proxy (10M fv_prefix
      >= 1.5x the 2.7M's). If no level shows retrieval: 10M at 60; P4 holds
      iff the 10M shows it where 2.7M did not.

Budget <= 25 min (4 x 252s + 300s trains + ~2 min readouts). smoke: false.

Run: python lab/e049_refrain_threshold.py
Outputs: runs/e049/{metrics.json, threshold_curve.png}
"""
from __future__ import annotations

import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, run_dir,
                    save_json, set_seed, train_model)

T0 = time.time()
SEED = 42            # model init (B/e001 convention)
GEN_SEED = 4910      # corpus doc-spec generation
CORPUS_SEED = 1337   # batch sampling (B convention)
DATA = REPO / "data"
CKPT_DIR = REPO / "runs" / "checkpoints"
FILLER_SRC = DATA / "input.txt"

LEVELS = [0, 5, 20, 60]
TARGET_BYTES = 620_000
F1_MIN, F1_MAX = 30, 120
F2_MIN, F2_MAX = 20, 90        # gap refrain->completion, > 16 chars
F3_MIN, F3_MAX = 30, 120
TRUNC = 16
CTX = 256
TRAIN_SECONDS = 252.0
TRAIN_SECONDS_10M = 300.0
N_NONREFRAIN = 1500
N_ATTN_PROMPTS = 40
NONREFRAIN_SEED = 15
BATCH = 64

S27 = {"n_layer": 6, "n_head": 6, "n_embd": 192}   # B config
S10 = {"n_layer": 8, "n_head": 8, "n_embd": 320}   # e005s large config

# 10 refrain/completion pairs; R 24-32 chars, C 8-18 chars, leading space,
# lowercase + space + period only (vocab-safe); none occur in input.txt.
REFRAIN_PAIRS = [
    ("the night has been unkind to", " us all."),
    ("when the cold stars gather", " we keep watch."),
    ("o bury me beneath the willow", " where she waits."),
    ("the king rides out at midnight", " and none follow."),
    ("hark the lark has forgotten", " her morning song"),
    ("the sea remembers every name", " given to the deep"),
    ("my heart is a winter garden", " locked in frost."),
    ("speak softly of the drowned", " they hear us."),
    ("the fox knows the orchard gate", " but not the hour."),
    ("bring torches to the water stair", " and be silent."),
]


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- corpus


def inclusion(p: int, i: int) -> bool:
    if p == 60:
        return i % 5 < 3
    if p == 20:
        return i % 5 == 0
    if p == 5:
        return i % 20 == 0
    return False


def build_corpora() -> tuple[dict, dict, dict]:
    """Returns ({p: path}, {p: [events]}, {p: stats}); an event has absolute
    char coords r_start/r_end (refrain) and c_start/c_end (completion).
    All corpora share one doc-spec stream (prefix-aligned). A pangram header
    (all 65 filler chars once) forces an identical vocab in every corpus —
    '$' (1x) and '&' (3x) are too rare to survive random 620KB slicing."""
    src = FILLER_SRC.read_text(encoding="utf-8")
    for r, c in REFRAIN_PAIRS:
        assert 24 <= len(r) <= 32 and 8 <= len(c) <= 18 and c[0] == " ", (r, c)
        assert r not in src, f"refrain already in filler: {r}"
    pangram = "".join(sorted(set(src)))
    rng = random.Random(GEN_SEED)
    parts = {p: [] for p in LEVELS}
    pos = {p: len(pangram) + 1 for p in LEVELS}  # coords are file-absolute
    events = {p: [] for p in LEVELS}
    done = {p: False for p in LEVELS}
    stats = {p: {"docs": 0, "events": 0} for p in LEVELS}
    i = 0
    while not all(done.values()):
        f1_len = rng.randint(F1_MIN, F1_MAX)
        f2_len = rng.randint(F2_MIN, F2_MAX)
        f3_len = rng.randint(F3_MIN, F3_MAX)
        s1 = rng.randrange(len(src) - f1_len - 1)
        s2 = rng.randrange(len(src) - f2_len - 1)
        s3 = rng.randrange(len(src) - f3_len - 1)
        f1, f2, f3 = src[s1:s1 + f1_len], src[s2:s2 + f2_len], src[s3:s3 + f3_len]
        refrain, compl = REFRAIN_PAIRS[i % 10]
        for p in LEVELS:
            if done[p]:
                continue
            if inclusion(p, i):
                doc = f1 + refrain + f2 + compl + f3 + "\n"
                r_start = pos[p] + len(f1)
                events[p].append({"r_start": r_start, "r_end": r_start + len(refrain),
                                  "c_start": r_start + len(refrain) + len(f2),
                                  "c_end": r_start + len(refrain) + len(f2) + len(compl),
                                  "doc": i, "phrase": i % 10})
                stats[p]["events"] += 1
            else:
                doc = f1 + f3 + "\n"
            parts[p].append(doc)
            pos[p] += len(doc)
            stats[p]["docs"] += 1
            if pos[p] >= TARGET_BYTES:
                done[p] = True
        i += 1
    paths = {}
    for p in LEVELS:
        path = DATA / f"e049_p{p}.txt"
        if not path.exists():
            path.write_text(pangram + "\n" + "".join(parts[p]), encoding="utf-8")
        paths[p] = path
        stats[p]["bytes"] = pos[p]  # pos already includes the pangram offset
    return paths, events, stats


# ---------------------------------------------------------------- training


def train_net(corpus_path, ckpt_final, ckpt_train, tag, cfg_kw, cap_s):
    set_seed(SEED)
    corpus = CharCorpus(corpus_path, seed=CORPUS_SEED)
    model = TinyGPT(Cfg(vocab=corpus.vocab_size, block_size=CTX, **cfg_kw)).to(DEVICE)
    if ckpt_final.exists():
        model.load_state_dict(torch.load(ckpt_final, map_location=DEVICE, weights_only=True))
        log(f"{tag}: final ckpt found ({model.num_params():,} params), skipping training")
        return model, corpus, None
    log(f"{tag}: training {model.num_params():,} params (cap {cap_s:.0f}s)")
    hist = train_model(model, corpus, steps=4000, lr=1e-3, batch_size=64,
                       max_seconds=cap_s, ckpt=ckpt_train)
    torch.save(model.state_dict(), ckpt_final)
    log(f"{tag}: done at step {hist[-1]['step']}, val {hist[-1]['val_loss']:.4f}")
    return model, corpus, {"steps": hist[-1]["step"], "val_loss": hist[-1]["val_loss"],
                           "cap_s": cap_s}


# ---------------------------------------------------------------- readouts


@torch.no_grad()
def pred_at(model, ids, ts, ctx):
    """CE and top-1 acc predicting ids[t] from ids[t-ctx:t], batched."""
    ces, accs = [], []
    for b0 in range(0, len(ts), BATCH):
        tb = ts[b0:b0 + BATCH]
        x = torch.stack([ids[t - ctx:t] for t in tb]).to(DEVICE)
        y = ids[tb].to(DEVICE)
        logits, _ = model(x)
        lg = logits[:, -1]
        ces.append(F.cross_entropy(lg, y, reduction="none").cpu())
        accs.append((lg.argmax(-1) == y).float().cpu())
    return torch.cat(ces), torch.cat(accs)


def event_positions(events, train_len):
    """Val events (completion fully after train_len): flat positions + k index."""
    ts, ks = [], []
    for ev in events:
        if ev["c_start"] < train_len:
            continue
        for k in range(ev["c_end"] - ev["c_start"]):
            ts.append(ev["c_start"] + k)
            ks.append(k)
    return torch.tensor(ts, dtype=torch.long), torch.tensor(ks, dtype=torch.long)


def refrain_readout(model, ids, events, train_len, tag):
    """(a)+(b): far-value and accuracy at completion positions."""
    ts, ks = event_positions(events, train_len)
    n_events = int((ks == 0).sum())
    if len(ts) == 0:
        return {"n_events": 0}
    ce_f, acc_f = pred_at(model, ids, ts, CTX)
    ce_t, _ = pred_at(model, ids, ts, TRUNC)
    fv = ce_t - ce_f
    pre = (ks >= 1) & (ks <= 3)
    k1 = ks == 1
    per_k = {}
    for k in sorted(set(ks.tolist())):
        m = ks == k
        per_k[k] = {"far_value": float(fv[m].mean()), "acc": float(acc_f[m].mean()),
                    "n": int(m.sum())}
    out = {
        "n_events": n_events, "n_positions": len(ts),
        "fv_prefix_k123": float(fv[pre].mean()),
        "fv_k1": float(fv[k1].mean()),
        "acc_prefix_k123": float(acc_f[pre].mean()),
        "acc_span_all": float(acc_f.mean()),
        "ce_full_prefix": float(ce_f[pre].mean()),
        "ce_trunc_prefix": float(ce_t[pre].mean()),
        "frac_prefix_fv_ge_05": float((fv[pre] >= 0.5).float().mean()),
        "per_k": per_k,
    }
    log(f"  {tag}: events {n_events} | fv_prefix {out['fv_prefix_k123']:+.3f} "
        f"(k1 {out['fv_k1']:+.3f}) | acc_prefix {out['acc_prefix_k123']:.3f} | "
        f"CE full {out['ce_full_prefix']:.3f} trunc {out['ce_trunc_prefix']:.3f}")
    return out


def non_refrain_readout(model, ids, mask, train_len, tag):
    """(d): far-value at ordinary (non-refrain, non-completion) val positions."""
    n = len(ids)
    cand = torch.tensor([t for t in range(train_len, n - 1) if not mask[t]], dtype=torch.long)
    gen = torch.Generator().manual_seed(NONREFRAIN_SEED)
    pick = cand[torch.randperm(len(cand), generator=gen)[:N_NONREFRAIN]]
    ce_f, _ = pred_at(model, ids, pick, CTX)
    ce_t, _ = pred_at(model, ids, pick, TRUNC)
    fv = ce_t - ce_f
    out = {"n": len(pick), "far_value_mean": float(fv.mean()),
           "far_value_median": float(fv.median()),
           "frac_ge_015": float((fv >= 0.15).float().mean())}
    log(f"  {tag}: non-refrain far-value {out['far_value_mean']:+.4f} "
        f"(frac>=0.15 {out['frac_ge_015']:.3f})")
    return out


@torch.no_grad()
def attention_census(model, ids, prompts):
    """(c): per layer/head attention from the completion-query position onto
    the refrain source span, a pre-refrain control span, and far d>=17.
    prompts: list of (q_pos, r_start, r_end)."""
    cfg = model.cfg
    L, H, hd = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
    win = CTX
    rmass = torch.zeros(L, H, len(prompts))
    cmass = torch.zeros(L, H, len(prompts))
    fmass = torch.zeros(L, H, len(prompts))
    ln_inputs, handles = {}, []
    for li, block in enumerate(model.h):
        def mk(li):
            def pre(m, args):
                ln_inputs[li] = args[0].detach()
                return None
            return pre
        handles.append(block.attn.register_forward_pre_hook(mk(li)))
    d = torch.arange(win)
    dist = (win - 1) - d
    far_mask = dist >= 17
    for p, (q_pos, r_start, r_end) in enumerate(prompts):
        start = q_pos - (win - 1)
        x = ids[start:q_pos + 1].unsqueeze(0).to(DEVICE)
        model(x)
        r_mask = torch.zeros(win, dtype=torch.bool)
        r_mask[r_start - start:r_end - start] = True
        c_mask = torch.zeros(win, dtype=torch.bool)
        c_mask[r_start - start - (r_end - r_start):r_start - start] = True  # equal-len pre-refrain
        for li in range(L):
            q, k, _ = model.h[li].attn.c_attn(ln_inputs[li]).split(cfg.n_embd, dim=2)
            q = q.view(1, win, H, hd).transpose(1, 2)
            k = k.view(1, win, H, hd).transpose(1, 2)
            att = F.softmax(q @ k.transpose(-2, -1) / (hd ** 0.5), dim=-1)[0, :, -1, :].cpu()
            for h in range(H):
                a = att[h]
                rmass[li, h, p] = float(a[r_mask].sum())
                cmass[li, h, p] = float(a[c_mask].sum())
                fmass[li, h, p] = float(a[far_mask].sum())
    for h_ in handles:
        h_.remove()
    head_r = rmass.mean(dim=2)
    best = torch.unravel_index(head_r.argmax(), head_r.shape)
    return {"layer_refrain_mass": head_r.mean(dim=1).tolist(),
            "layer_control_mass": cmass.mean(dim=2).mean(dim=1).tolist(),
            "layer_far_mass": fmass.mean(dim=2).mean(dim=1).tolist(),
            "best_head": {"layer": int(best[0]), "head": int(best[1]),
                          "refrain_mass": float(head_r[best]),
                          "control_mass": float(cmass.mean(dim=2)[best])},
            "n_prompts": len(prompts)}


# ---------------------------------------------------------------- main


def main():
    rd = run_dir("e049")
    log("=== corpus ===")
    paths, events, cstats = build_corpora()
    for p in LEVELS:
        log(f"p={p}%: {cstats[p]['docs']} docs, {cstats[p]['bytes']:,} bytes, "
            f"{cstats[p]['events']} refrain events")

    corpora, full_ids, masks = {}, {}, {}
    base_vocab = None
    for p in LEVELS:
        cp = CharCorpus(paths[p], seed=CORPUS_SEED)
        if base_vocab is None:
            base_vocab = cp.stoi
        assert cp.stoi == base_vocab and cp.vocab_size == 65, f"vocab mismatch at p={p}"
        corpora[p] = cp
        full_ids[p] = torch.cat([cp.train, cp.val])
        mask = torch.zeros(len(full_ids[p]), dtype=torch.bool)
        for ev in events[p]:
            mask[ev["r_start"]:ev["r_end"]] = True
            mask[ev["c_start"]:ev["c_end"]] = True
        masks[p] = mask
        assert cstats[p]["bytes"] == len(full_ids[p])

    # shared probe set: p60 val events (same prompts for every net)
    train_len60 = len(corpora[60].train)
    shared_events = [ev for ev in events[60] if ev["c_start"] >= train_len60]
    shared_ids = full_ids[60]
    attn_prompts = [(ev["c_start"] - 1, ev["r_start"], ev["r_end"])
                    for ev in shared_events[:N_ATTN_PROMPTS]]
    log(f"shared probes: {len(shared_events)} p60 val events, "
        f"{len(attn_prompts)} attention prompts")

    readouts, train_meta = {}, {}
    for p in LEVELS:
        tag = f"s27_p{p}"
        model, corpus, tmeta = train_net(
            paths[p], CKPT_DIR / f"e049_{tag}.pt", CKPT_DIR / f"e049_{tag}.train.pt",
            tag, S27, TRAIN_SECONDS)
        model.eval()
        train_meta[f"s27_p{p}"] = tmeta
        train_len = len(corpus.train)
        own = refrain_readout(model, full_ids[p], events[p], train_len, f"{tag} own-val")
        shared = refrain_readout(model, shared_ids, shared_events, train_len60, f"{tag} shared")
        nonref = non_refrain_readout(model, full_ids[p], masks[p], train_len, f"{tag} non-refrain")
        attn = attention_census(model, shared_ids, attn_prompts)
        bh = attn["best_head"]
        log(f"  {tag}: best head L{bh['layer']}H{bh['head']} refrain-mass "
            f"{bh['refrain_mass']:.3f} (control {bh['control_mass']:.3f})")
        readouts[f"s27_p{p}"] = {"own": own, "shared": shared,
                                 "non_refrain": nonref, "attention": attn}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- threshold decision ----------------
    base_fv = readouts["s27_p0"]["shared"]["fv_prefix_k123"]

    def shown(key):
        r = readouts[key]["own"]
        return (r["fv_prefix_k123"] - base_fv >= 1.0 and r["acc_prefix_k123"] >= 0.5,
                r["fv_prefix_k123"] - base_fv, r["acc_prefix_k123"])

    shown_at = {}
    for p in LEVELS:
        if p == 0:
            continue
        ok, margin, acc = shown(f"s27_p{p}")
        shown_at[p] = {"shown": bool(ok), "fv_margin": float(margin), "acc_prefix": float(acc)}
        log(f"p={p}%: retrieval_shown={ok} (fv margin {margin:+.3f}, acc {acc:.3f})")

    p_star = next((p for p in (5, 20, 60) if shown_at[p]["shown"]), None)
    if p_star == 5:
        s10_level, p4_mode = 5, "magnitude-proxy"
    elif p_star is not None:
        s10_level, p4_mode = {20: 5, 60: 20}[p_star], "below-threshold-direct"
    else:
        s10_level, p4_mode = 60, "no-2.7M-retrieval"

    # ---------------- conditional 10M net (always run: the scale contrast
    # is the point of P4; level chosen by the 2.7M threshold above) ----------------
    readouts_10m, p4 = None, "UNTESTABLE (no 10M run)"
    if True:
        tag = f"s10_p{s10_level}"
        model, corpus, tmeta = train_net(
            paths[s10_level], CKPT_DIR / f"e049_{tag}.pt", CKPT_DIR / f"e049_{tag}.train.pt",
            tag, S10, TRAIN_SECONDS_10M)
        model.eval()
        train_meta[tag] = tmeta
        train_len = len(corpus.train)
        own = refrain_readout(model, full_ids[s10_level], events[s10_level], train_len, f"{tag} own-val")
        shared = refrain_readout(model, shared_ids, shared_events, train_len60, f"{tag} shared")
        nonref = non_refrain_readout(model, full_ids[s10_level], masks[s10_level], train_len, f"{tag} non-refrain")
        attn = attention_census(model, shared_ids, attn_prompts)
        bh = attn["best_head"]
        log(f"  {tag}: best head L{bh['layer']}H{bh['head']} refrain-mass "
            f"{bh['refrain_mass']:.3f} (control {bh['control_mass']:.3f})")
        readouts_10m = {"level": s10_level, "own": own, "shared": shared,
                        "non_refrain": nonref, "attention": attn}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        s10_shown = (own["fv_prefix_k123"] - base_fv >= 1.0 and own["acc_prefix_k123"] >= 0.5)
        if p4_mode == "below-threshold-direct":
            p4 = f"{'CONFIRMED' if s10_shown else 'REFUTED'} (10M shows retrieval at p={s10_level}% where 2.7M does not: {s10_shown})"
        elif p4_mode == "magnitude-proxy":
            ratio = (own["fv_prefix_k123"] / readouts[f"s27_p5"]["own"]["fv_prefix_k123"]
                     if readouts[f"s27_p5"]["own"]["fv_prefix_k123"] > 0.05 else float("inf"))
            p4 = f"{'WEAK-SUPPORT' if (s10_shown and ratio >= 1.5) else 'NOT-SUPPORTED'} (weak proxy: 10M/2.7M fv ratio at p=5: {ratio:.2f})"
        else:  # no 2.7M retrieval anywhere
            p4 = f"{'CONFIRMED' if s10_shown else 'UNRESOLVED'} (2.7M shows no retrieval at any p; 10M at p=60: {s10_shown})"

    # ---------------- registered verdicts ----------------
    fv_own = {p: readouts[f"s27_p{p}"]["own"]["fv_prefix_k123"] for p in (5, 20, 60)}
    acc_ok = {p: readouts[f"s27_p{p}"]["own"]["acc_prefix_k123"] >= 0.5 for p in (5, 20, 60)}
    P1 = bool(shown_at[60]["shown"] and not shown_at[5]["shown"]) if 60 in shown_at else False
    if shown_at.get(5, {}).get("shown"):
        P1_label = "REFUTED-LOW (threshold <= 5%)"
    elif P1:
        P1_label = "CONFIRMED"
    else:
        P1_label = "REFUTED (no retrieval even at 60%)"

    ratios = []
    for a, b in ((5, 20), (20, 60)):
        lo, hi = fv_own[a], fv_own[b]
        ratios.append((f"{a}->{b}", hi / lo if lo > 0.05 else (float("inf") if hi - lo >= 1.0 else 0.0), hi - lo))
    best_ratio = max(r for _, r, _ in ratios)
    best_jump = max(j for _, _, j in ratios)
    P2 = bool(best_ratio >= 10 and best_jump >= 1.0)
    P3 = bool(all(readouts[f"s27_p{p}"]["non_refrain"]["far_value_mean"] < 0.1 for p in LEVELS))

    verdicts = {
        "P1_retrieval_between_5_and_60": P1_label,
        "P2_sharp_threshold": ("SHARP" if P2 else "GRADED"),
        "P2_adjacent_ratios": {lbl: (float("+inf") if r == float("inf") else round(r, 2))
                               for lbl, r, _ in ratios},
        "P3_no_leak_non_refrain": P3,
        "P3_means": {p: readouts[f"s27_p{p}"]["non_refrain"]["far_value_mean"] for p in LEVELS},
        "P4_threshold_lower_at_10M": p4,
        "p_star_2_7M": p_star,
        "retrieval_shown_by_level": {str(p): shown_at[p]["shown"] for p in shown_at},
        "fv_prefix_own": {str(p): fv_own[p] for p in fv_own},
        "fv_prefix_baseline_p0_shared": base_fv,
    }

    # ---------------- plot ----------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    xs = LEVELS
    lbl = lambda p: f"{p}%"

    ax = axes[0, 0]
    own_y = [readouts[f"s27_p{p}"]["own"]["fv_prefix_k123"] for p in xs]
    sh_y = [readouts[f"s27_p{p}"]["shared"]["fv_prefix_k123"] for p in xs]
    ax.plot(xs, own_y, "o-", color="steelblue", label="2.7M own-val")
    ax.plot(xs, sh_y, "s--", color="seagreen", label="2.7M shared p60 probes")
    ax.axhline(base_fv, color="gray", ls=":", label=f"p0 baseline {base_fv:+.3f}")
    ax.axhline(base_fv + 1.0, color="crimson", ls="--", label="retrieval bar (+1.0)")
    if readouts_10m is not None:
        ax.plot([readouts_10m["level"]], [readouts_10m["own"]["fv_prefix_k123"]], "*",
                color="purple", ms=16, label=f"10M p={readouts_10m['level']}%")
    for x, y in zip(xs, own_y):
        ax.annotate(f"{y:+.2f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xlabel("refrain density p (%)"); ax.set_ylabel("far-value at completions (nats)")
    ax.set_title(f"(a) far-value vs density — P1: {P1_label} / P2: {verdicts['P2_sharp_threshold']}")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    acc_y = [readouts[f"s27_p{p}"]["own"]["acc_prefix_k123"] for p in xs]
    acc_s = [readouts[f"s27_p{p}"]["shared"]["acc_prefix_k123"] for p in xs]
    ax.plot(xs, acc_y, "o-", color="steelblue", label="2.7M own-val")
    ax.plot(xs, acc_s, "s--", color="seagreen", label="2.7M shared")
    ax.axhline(0.5, color="crimson", ls="--", label="acc bar 0.5")
    if readouts_10m is not None:
        ax.plot([readouts_10m["level"]], [readouts_10m["own"]["acc_prefix_k123"]], "*",
                color="purple", ms=16, label=f"10M p={readouts_10m['level']}%")
    for x, y in zip(xs, acc_y):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xlabel("refrain density p (%)"); ax.set_ylabel("completion accuracy (k=1..3)")
    ax.set_title("(b) refrain-completion accuracy vs density")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bm = [readouts[f"s27_p{p}"]["attention"]["best_head"]["refrain_mass"] for p in xs]
    bc = [readouts[f"s27_p{p}"]["attention"]["best_head"]["control_mass"] for p in xs]
    ax.plot(xs, bm, "o-", color="crimson", label="2.7M best-head refrain-mass")
    ax.plot(xs, bc, "o--", color="orange", label="2.7M same-head control span")
    ax.axhline(0.5, color="gray", ls="--", label="e021 retrieval-head bar 0.5")
    if readouts_10m is not None:
        ax.plot([readouts_10m["level"]],
                [readouts_10m["attention"]["best_head"]["refrain_mass"]], "*",
                color="purple", ms=16, label="10M best-head")
    for p in xs:
        bh = readouts[f"s27_p{p}"]["attention"]["best_head"]
        ax.annotate(f"L{bh['layer']}H{bh['head']}", (p, bh["refrain_mass"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xlabel("refrain density p (%)"); ax.set_ylabel("attn mass on refrain source span")
    ax.set_title("(c) retrieval-head formation vs density (40 shared prompts)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    nr = [readouts[f"s27_p{p}"]["non_refrain"]["far_value_mean"] for p in xs]
    bars = ax.bar([lbl(p) for p in xs], nr, color=["gray", "steelblue", "steelblue", "steelblue"])
    ax.axhline(0.1, color="crimson", ls="--", label="P3 bar 0.1")
    if readouts_10m is not None:
        ax.axhline(readouts_10m["non_refrain"]["far_value_mean"], color="purple", ls=":",
                   label=f"10M p={readouts_10m['level']}%: {readouts_10m['non_refrain']['far_value_mean']:+.4f}")
    for b, v in zip(bars, nr):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.4f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("refrain density p (%)"); ax.set_ylabel("mean far-value, non-refrain positions")
    ax.set_title(f"(d) leak into ordinary text — P3: {'HOLDS' if P3 else 'REFUTED'}")
    ax.legend(fontsize=8)

    fig.suptitle("E049 — the refrain threshold: where does far-retrieval appear?")
    fig.tight_layout()
    fig.savefig(rd / "threshold_curve.png", dpi=140)
    plt.close(fig)

    metrics = {
        "experiment": "e049_refrain_threshold", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED, "smoke": False,
        "registered": "P1-P4 in lab/e049_refrain_threshold.py header (written before the run)",
        "refrain_pairs": [{"refrain": r, "completion": c} for r, c in REFRAIN_PAIRS],
        "corpora": {str(p): {"path": str(paths[p].name), **cstats[p],
                             "events_val": (len(shared_events) if p == 60 else
                                            len([e for e in events[p] if e["c_start"] >= len(corpora[p].train)]))}
                    for p in LEVELS},
        "training": train_meta,
        "readouts": readouts,
        "readouts_10m": readouts_10m,
        "verdicts": verdicts,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", metrics)

    log("=== REGISTERED VERDICTS ===")
    for k, v in verdicts.items():
        log(f"{k}: {v}")
    log(f"outputs: {rd}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
