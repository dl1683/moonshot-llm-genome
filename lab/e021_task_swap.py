"""E021 — Task swap (T009): does a retrieval-required task break the no-retrieval picture?

Synthetic corpus (~1MB): documents "ID: XXXXX\\n" + real Shakespeare filler
(30-120 chars, >16 tokens between the ID nonce and COPY) + "\\nCOPY: XXXXX\\n".
Nonce = 5 iid uppercase chars. Control corpus: byte-identical except each COPY
nonce is an independent random nonce (cue uncorrelated). Two fresh e001-config
nets (seed 42, 252s cap each, ckpt resume). Readouts at COPY nonce positions
on held-out val docs:
  (a) copy accuracy — P1: task net >= 80% next-char accuracy on nonce chars.
  (b) far-value = CE(trunc-16) - CE(full-256) — P2: task >= +1.0 nat, ctrl ~ 0.
  (c) decision-depth census, 1000 COPY positions, vs runs/e012 baseline — P4:
      new late mode => stage picture gains a task-dependent member; byte-identical
      => stages are corpus-trivial.
  (d) attention census, ~50 COPY prompts: local d1-3 / far d17+ / mass ON the ID
      nonce positions, task vs control — P3: attention concentrates on the ID.

Deviations from the task card (noted per instructions):
- Doc count ~10.6k (card said ~7-8k): with 30-120-char fillers a doc averages
  ~98 bytes, so ~1MB needs ~10.6k docs. The registered design pins "~1MB" +
  filler 30-120 + >16-token gap; all preserved, count follows.
- Depth census and attention census are ALSO run on the control net (free
  contrast; registration only requires the task net).
- Attention census window is 256 (e013a used 96) so the ID nonce (up to 136
  chars back) always fits in-window.
- P2 "control ~ 0" operationalized as |far-value_ctrl| <= 0.25 nat; P3 as
  max-head ID-mass >= 0.5 or max layer-mean ID-mass >= 0.30; P4 "new mode" as
  unstable frac > 5% (e012 baseline had 0%) or late(>=L5) frac >= 0.65 with
  JS(task, Shakespeare) > 0.10; "byte-identical" as JS <= 0.02.

Run: python lab/e021_task_swap.py    (corpora + checkpoints auto-created/resumed)
"""
from __future__ import annotations

import json
import random
import string
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict,
                    estimate_loss, run_dir, save_json, set_seed, train_model)

TASK_TXT = REPO / "data" / "e021_task.txt"
CTRL_TXT = REPO / "data" / "e021_control.txt"
FILLER_SRC = REPO / "data" / "input.txt"
TASK_CKPT = REPO / "runs" / "checkpoints" / "e021_task.train.pt"
CTRL_CKPT = REPO / "runs" / "checkpoints" / "e021_control.train.pt"
E012_METRICS = REPO / "runs" / "e012" / "metrics.json"
E013A_METRICS = REPO / "runs" / "e013a" / "metrics.json"

SEED = 42          # model init (same as e001)
GEN_SEED = 2100    # corpus generation
CORPUS_SEED = 1337 # batch sampling (same as e001)
TARGET_BYTES = 1_048_576
FILL_MIN, FILL_MAX = 30, 120
TRAIN_SECONDS = 252.0
N_EVAL_POS = 2500   # COPY-nonce target positions for accuracy + far-value
N_DEPTH_POS = 1000
N_ATTN_PROMPTS = 50
TRUNC = 16
BATCH = 64


# ---------------------------------------------------------------- corpus


def build_corpus() -> tuple[list[tuple[list[int], list[int]]], dict]:
    """Returns per-doc (id_nonce_positions, copy_nonce_positions) in char
    coordinates + stats. Layout of a doc (f = filler len):
        'ID: ' (0-3) nonce (4-8) '\\n' (9) filler (10..10+f-1)
        '\\nCOPY: ' (10+f..10+f+6) copy-nonce (10+f+7..+11) '\\n'
    """
    src = FILLER_SRC.read_text(encoding="utf-8")
    rng = random.Random(GEN_SEED)
    letters = string.ascii_uppercase

    def nonce5():
        return "".join(rng.choice(letters) for _ in range(5))

    task_parts, ctrl_parts, meta, nonces = [], [], [], []
    pos = total = 0
    while total < TARGET_BYTES:
        nonce = nonce5()
        nonces.append(nonce)
        f_len = rng.randint(FILL_MIN, FILL_MAX)
        s0 = rng.randrange(len(src) - f_len - 1)
        filler = src[s0 : s0 + f_len]
        head = f"ID: {nonce}\n{filler}\nCOPY: "
        id_pos = [pos + 4 + k for k in range(5)]
        copy_pos = [pos + 17 + f_len + k for k in range(5)]
        task_parts.append(head + nonce + "\n")
        cnonce = nonce5()
        while cnonce == nonce:  # uncorrelated replacement, never identical
            cnonce = nonce5()
        ctrl_parts.append(head + cnonce + "\n")
        meta.append((id_pos, copy_pos))
        pos += len(head) + 6
        total += len(head) + 6
    task_txt = "".join(task_parts)
    ctrl_txt = "".join(ctrl_parts)
    if not TASK_TXT.exists():
        TASK_TXT.write_text(task_txt, encoding="utf-8")
    if not CTRL_TXT.exists():
        CTRL_TXT.write_text(ctrl_txt, encoding="utf-8")
    stats = {"n_docs": len(meta), "bytes": total,
             "unique_nonces": len(set(nonces)),
             "min_gap_id_to_copy_chars": FILL_MIN + 7}
    return meta, stats


# ---------------------------------------------------------------- training


def train_net(corpus_path, ckpt, tag):
    set_seed(SEED)
    corpus = CharCorpus(corpus_path, seed=CORPUS_SEED)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    if ckpt.exists():
        st = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        done = st["step"] >= 4000 or st.get("history", [{}])[-1].get("elapsed_s", 0) > 250
        if done:
            model.load_state_dict(st["model"])
            print(f"[{tag}] ckpt complete (step {st['step']}), skipping training")
            return model, corpus, st["history"]
    print(f"[{tag}] training ({model.num_params():,} params)...")
    history = train_model(model, corpus, steps=4000, lr=1e-3, batch_size=64,
                          max_seconds=TRAIN_SECONDS, ckpt=ckpt)
    return model, corpus, history


# ---------------------------------------------------------------- readouts


@torch.no_grad()
def last_pred(model, xs, ys):
    """Per-position CE and top-1-correct for the last-token prediction."""
    ces, corr = [], []
    for b0 in range(0, len(xs), BATCH):
        xb, yb = xs[b0 : b0 + BATCH].to(DEVICE), ys[b0 : b0 + BATCH].to(DEVICE)
        logits, _ = model(xb)
        lg = logits[:, -1]
        ces.append(F.cross_entropy(lg, yb, reduction="none").cpu())
        corr.append((lg.argmax(-1) == yb).float().cpu())
    return torch.cat(ces), torch.cat(corr)


@torch.no_grad()
def batched_snapshots(model, xs):
    """xs: (B,T) -> list of (B, C) last-position stream vectors per depth 0..L."""
    snaps, handles = [], []

    def pre(module, args):
        snaps.append(args[0].detach())
        return None

    handles.append(model.h[0].register_forward_pre_hook(pre))
    for block in model.h:
        def h(module, args, out):
            snaps.append(out.detach())
        handles.append(block.register_forward_hook(h))
    model(xs)
    for h_ in handles:
        h_.remove()
    return [s[:, -1, :] for s in snaps]


@torch.no_grad()
def depth_census(model, ids, targets, n_pos, seed):
    """Decision depth at the given target positions (e012 pattern)."""
    gen = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(targets), generator=gen)[:n_pos]
    ts = targets[pick]
    depths = torch.full((len(ts),), -1, dtype=torch.long)
    for b0 in range(0, len(ts), BATCH):
        xb = torch.stack([ids[t - 256 : t] for t in ts[b0 : b0 + BATCH]]).to(DEVICE)
        last = batched_snapshots(model, xb)
        top1 = torch.stack([model.lm_head(model.ln_f(v)).argmax(-1) for v in last])
        final = top1[-1]
        for d in range(model.cfg.n_layer + 1):
            stable_d = (top1[d:] == final.unsqueeze(0)).all(dim=0)
            sel = (depths[b0 : b0 + BATCH] < 0) & stable_d.cpu()
            depths[b0 : b0 + BATCH][sel] = d
    hist = torch.bincount(depths[depths >= 0], minlength=model.cfg.n_layer + 1).float()
    return {"counts": hist.tolist(), "unstable": int((depths < 0).sum()),
            "n": len(ts)}


@torch.no_grad()
def attention_census(model, ids, prompts, win=256):
    """prompts: list of (q_pos, id_positions[5]). Attention of the query at
    q_pos (last window token) per layer/head: local d1-3, far d17+, and mass
    on the ID nonce key positions (e013a recompute pattern)."""
    L, H = model.cfg.n_layer, model.cfg.n_head
    cfg = model.cfg
    local = torch.zeros(L, H, len(prompts))
    far = torch.zeros(L, H, len(prompts))
    idm = torch.zeros(L, H, len(prompts))

    ln_inputs, handles = {}, []
    for i, block in enumerate(model.h):
        def mk(i):
            def pre(m, args):
                ln_inputs[i] = args[0].detach()
                return None
            return pre
        handles.append(block.attn.register_forward_pre_hook(mk(i)))

    d = torch.arange(win)
    dist = (win - 1) - d
    local_mask = (dist >= 1) & (dist <= 3)
    far_mask = dist >= 17
    for p, (q_pos, id_pos) in enumerate(prompts):
        start = q_pos - (win - 1)
        x = ids[start : q_pos + 1].unsqueeze(0).to(DEVICE)
        chars = ids[start : q_pos + 1]
        model(x)
        id_mask = torch.zeros(win, dtype=torch.bool)
        id_mask[[ip - start for ip in id_pos]] = True
        for i in range(L):
            qkv = model.h[i].attn.c_attn(ln_inputs[i])
            q, k, _ = qkv.split(cfg.n_embd, dim=2)
            hd = cfg.n_embd // cfg.n_head
            q = q.view(1, win, H, hd).transpose(1, 2)
            k = k.view(1, win, H, hd).transpose(1, 2)
            att = F.softmax(q @ k.transpose(-2, -1) / (hd ** 0.5), dim=-1)[0, :, -1, :]
            for h in range(H):
                a = att[h].cpu()
                local[i, h, p] = float(a[local_mask].sum())
                far[i, h, p] = float(a[far_mask].sum())
                idm[i, h, p] = float(a[id_mask].sum())
    for h_ in handles:
        h_.remove()
    return local, far, idm


def js_divergence(p, q, eps=1e-3):
    p = (torch.tensor(p, dtype=torch.float) + eps)
    q = (torch.tensor(q, dtype=torch.float) + eps)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum())


# ---------------------------------------------------------------- main


def main():
    t_start = time.time()
    rd = run_dir("e021")
    set_seed(SEED)

    print("=== corpus ===")
    meta, cstats = build_corpus()
    print(f"docs {cstats['n_docs']} | {cstats['bytes']:,} bytes | "
          f"unique nonces {cstats['unique_nonces']}/{cstats['n_docs']} | "
          f"min ID->COPY gap {cstats['min_gap_id_to_copy_chars']} chars")

    print("=== task net ===")
    tmodel, tcorpus, _ = train_net(TASK_TXT, TASK_CKPT, "task")
    print("=== control net ===")
    cmodel, ccorpus, _ = train_net(CTRL_TXT, CTRL_CKPT, "control")

    val_ce_task = estimate_loss(tmodel, tcorpus, "val", n_batches=30)
    val_ce_ctrl = estimate_loss(cmodel, ccorpus, "val", n_batches=30)
    print(f"val CE: task {val_ce_task:.4f} | control {val_ce_ctrl:.4f}")

    # full ids + eval positions: docs whose ID nonce lies fully in the val split
    t_ids = torch.cat([tcorpus.train, tcorpus.val])
    c_ids = torch.cat([ccorpus.train, ccorpus.val])
    val_start = len(tcorpus.train)
    eval_docs = [(i, c) for (i, c) in meta if i[0] >= val_start]
    copy_targets = torch.tensor([t for (_, c) in eval_docs for t in c])
    print(f"val docs {len(eval_docs)} | COPY nonce target positions {len(copy_targets)}")
    stride = max(1, len(copy_targets) // N_EVAL_POS)
    sel = copy_targets[::stride][:N_EVAL_POS]

    def gather(ids):
        xs_f = torch.stack([ids[t - 256 : t] for t in sel])
        xs_t = torch.stack([ids[t - TRUNC : t] for t in sel])
        ys = ids[sel]
        return xs_f, xs_t, ys

    # ---- (a) copy accuracy + (b) far-value ----
    txf, txt_, ty = gather(t_ids)
    cxf, cxt, cy = gather(c_ids)
    ce_f_t, acc_t = last_pred(tmodel, txf, ty)
    ce_t_t, _ = last_pred(tmodel, txt_, ty)
    ce_f_c, acc_c = last_pred(cmodel, cxf, cy)
    ce_t_c, _ = last_pred(cmodel, cxt, cy)
    fv_t, fv_c = ce_t_t - ce_f_t, ce_t_c - ce_f_c
    chance = 1.0 / 26.0
    print(f"copy acc: task {acc_t.mean():.3f} (chance {chance:.3f}) | control {acc_c.mean():.3f}")
    print(f"CE@COPY full-256: task {ce_f_t.mean():.3f} | control {ce_f_c.mean():.3f} (ln26={torch.log(torch.tensor(26.)):.3f})")
    print(f"far-value: task {fv_t.mean():+.3f} (med {fv_t.median():+.3f}) | "
          f"control {fv_c.mean():+.3f} (med {fv_c.median():+.3f})")

    # ---- (c) decision-depth census ----
    d_task = depth_census(tmodel, t_ids, copy_targets, N_DEPTH_POS, seed=21)
    d_ctrl = depth_census(cmodel, c_ids, copy_targets, N_DEPTH_POS, seed=21)
    e012 = json.loads(E012_METRICS.read_text())
    base_hist = e012["depth_histogram"] + [0.0]  # unstable bin (baseline had none)
    task_hist = d_task["counts"] + [float(d_task["unstable"])]
    ctrl_hist = d_ctrl["counts"] + [float(d_ctrl["unstable"])]
    js_task = js_divergence(task_hist, base_hist)
    js_ctrl = js_divergence(ctrl_hist, base_hist)
    late_task = sum(task_hist[5:]) / sum(task_hist)
    late_base = sum(base_hist[5:]) / sum(base_hist)
    print(f"depth@COPY task: {[int(v) for v in task_hist]} unstable {d_task['unstable']} "
          f"| JS vs Shakespeare {js_task:.3f} | late(>=L5) {late_task:.3f} (base {late_base:.3f})")
    print(f"depth@COPY ctrl: {[int(v) for v in ctrl_hist]} unstable {d_ctrl['unstable']} | JS {js_ctrl:.3f}")

    # ---- (d) attention census ----
    gen = torch.Generator().manual_seed(22)
    doc_pick = torch.randperm(len(eval_docs), generator=gen)[:N_ATTN_PROMPTS]
    prompts = []
    for j in doc_pick.tolist():
        id_pos, copy_pos = eval_docs[j]
        q_pos = copy_pos[0] - 1  # token predicting the FIRST COPY nonce char
        prompts.append((q_pos, id_pos))
    lcl_t, far_t, idm_t = attention_census(tmodel, t_ids, prompts)
    lcl_c, far_c, idm_c = attention_census(cmodel, c_ids, prompts)
    layer_local_t = lcl_t.mean(dim=(1, 2))
    layer_far_t = far_t.mean(dim=(1, 2))
    layer_idm_t = idm_t.mean(dim=(1, 2))
    layer_idm_c = idm_c.mean(dim=(1, 2))
    layer_local_c = lcl_c.mean(dim=(1, 2))
    head_idm = idm_t.mean(dim=2)  # (L, H)
    best = torch.unravel_index(head_idm.argmax(), head_idm.shape)
    best_val = float(head_idm[best])
    e013a = json.loads(E013A_METRICS.read_text())
    print("task layer local d1-3:", [round(v, 3) for v in layer_local_t.tolist()])
    print("task layer far d17+ : ", [round(v, 3) for v in layer_far_t.tolist()])
    print("task layer ID-mass  :", [round(v, 3) for v in layer_idm_t.tolist()])
    print("ctrl  layer ID-mass :", [round(v, 3) for v in layer_idm_c.tolist()])
    print(f"best retrieval head: L{best[0]} H{best[1]} ID-mass {best_val:.3f} "
          f"(ctrl same head {float(idm_c.mean(dim=2)[best]):.3f})")
    print(f"Shakespeare L5 local (e013a): {e013a['layer_local_mass'][5]:.3f} | "
          f"task L5 local here: {layer_local_t[5]:.3f}")

    # ---- verdicts (T009) ----
    P1 = bool(acc_t.mean() >= 0.80)
    P2 = bool(fv_t.mean() >= 1.0 and abs(fv_c.mean()) <= 0.25)
    P3 = bool(best_val >= 0.5 or layer_idm_t.max() >= 0.30)
    new_mode = (d_task["unstable"] / d_task["n"] > 0.05) or (late_task >= 0.65 and js_task > 0.10)
    byte_identical = js_task <= 0.02 and d_task["unstable"] == 0
    P4 = "NEW-MODE" if new_mode else ("IDENTICAL" if byte_identical else "SHIFTED")
    print("verdicts:", {"P1_copy_ge_80": P1, "P2_far_value": P2, "P3_id_focus": P3, "P4_depth": P4})

    # ---- plots ----
    labels = ["emb", "L0", "L1", "L2", "L3", "L4", "L5"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].hist(fv_t.tolist(), bins=50, alpha=0.65, color="seagreen", label=f"task (mean {fv_t.mean():+.2f})")
    axes[0].hist(fv_c.tolist(), bins=50, alpha=0.65, color="gray", label=f"control (mean {fv_c.mean():+.2f})")
    axes[0].axvline(1.0, color="crimson", ls="--", label="P2 threshold +1.0 nat")
    axes[0].set_xlabel("far-value = CE(16 ctx) − CE(256 ctx) at COPY nonce")
    axes[0].set_title("far-value distribution at COPY positions")
    axes[0].legend()
    x = [0, 1, 2, 3]
    w = 0.35
    axes[1].bar([i - w / 2 for i in x[:2]], [ce_f_t.mean(), ce_t_t.mean()], w, color="seagreen", label="task")
    axes[1].bar([i - w / 2 for i in [2]], [fv_t.mean()], w, color="seagreen")
    axes[1].bar([i + w / 2 for i in x[:2]], [ce_f_c.mean(), ce_t_c.mean()], w, color="gray", label="control")
    axes[1].bar([i + w / 2 for i in [2]], [fv_c.mean()], w, color="gray")
    axes[1].axhline(torch.log(torch.tensor(26.)).item(), color="k", ls=":", label="ln 26 (chance)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["CE full-256", "CE trunc-16", "far-value", ""])
    axes[1].set_ylabel("nats/char at COPY nonce positions")
    axes[1].set_title("CE by context length")
    axes[1].legend()
    fig.suptitle("E021 far-value: retrieval exists when the task requires it")
    fig.tight_layout(); fig.savefig(rd / "far_value.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs_ = torch.arange(8)
    w = 0.27
    ax.bar(xs_ - w, torch.tensor(base_hist) / sum(base_hist), w, color="steelblue", label="Shakespeare (e012)")
    ax.bar(xs_, torch.tensor(task_hist) / sum(task_hist), w, color="seagreen", label="task @ COPY")
    ax.bar(xs_ + w, torch.tensor(ctrl_hist) / sum(ctrl_hist), w, color="gray", label="control @ COPY")
    ax.set_xticks(xs_); ax.set_xticklabels(labels + ["unstable"])
    ax.set_ylabel("fraction of positions")
    ax.set_title(f"decision depth at COPY nonce positions (JS vs Shakespeare: {js_task:.3f})")
    ax.legend()
    fig.tight_layout(); fig.savefig(rd / "depth_hist.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    Ls = list(range(6))
    ax.plot(Ls, layer_local_t.tolist(), "o-", color="crimson", label="task local d1-3")
    ax.plot(Ls, layer_far_t.tolist(), "o-", color="steelblue", label="task far d17+")
    ax.plot(Ls, layer_idm_t.tolist(), "o-", color="seagreen", lw=2.5, label="task mass ON ID nonce")
    ax.plot(Ls, layer_idm_c.tolist(), "s--", color="gray", label="control mass ON ID nonce")
    ax.plot(Ls, e013a["layer_local_mass"], ":", color="crimson", alpha=0.5, label="Shakespeare local (e013a)")
    ax.set_xticks(Ls); ax.set_xticklabels([f"L{i}" for i in Ls])
    ax.set_ylabel("attention mass (query = COPY cue)")
    ax.set_title(f"attention by layer at COPY prompts — best head L{best[0]}H{best[1]} ID-mass {best_val:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(rd / "attention_by_layer.png", dpi=140); plt.close(fig)

    metrics = {
        "experiment": "e021_task_swap",
        "registered_design": "T009 (2026-09-24T12:40Z)",
        "corpus": cstats,
        "training": {
            "config": cfg_dict(tmodel.cfg), "seed": SEED,
            "params": tmodel.num_params(), "max_seconds": TRAIN_SECONDS,
            "val_ce_task": val_ce_task, "val_ce_control": val_ce_ctrl,
        },
        "copy_accuracy": {
            "task": float(acc_t.mean()), "control": float(acc_c.mean()),
            "chance_1_over_26": chance, "n_positions": len(sel),
        },
        "far_value": {
            "task_mean": float(fv_t.mean()), "task_median": float(fv_t.median()),
            "control_mean": float(fv_c.mean()), "control_median": float(fv_c.median()),
            "task_ce_full": float(ce_f_t.mean()), "task_ce_trunc": float(ce_t_t.mean()),
            "control_ce_full": float(ce_f_c.mean()), "control_ce_trunc": float(ce_t_c.mean()),
        },
        "depth_census": {
            "task_counts": task_hist, "control_counts": ctrl_hist,
            "shakespeare_baseline_counts": base_hist,
            "js_task_vs_shakespeare": js_task, "js_control_vs_shakespeare": js_ctrl,
            "task_unstable": d_task["unstable"], "control_unstable": d_ctrl["unstable"],
            "late_frac_task": late_task, "late_frac_shakespeare": late_base,
        },
        "attention_census": {
            "n_prompts": N_ATTN_PROMPTS,
            "task_layer_local": layer_local_t.tolist(),
            "task_layer_far": layer_far_t.tolist(),
            "task_layer_id_mass": layer_idm_t.tolist(),
            "control_layer_id_mass": layer_idm_c.tolist(),
            "control_layer_local": layer_local_c.tolist(),
            "best_head": {"layer": int(best[0]), "head": int(best[1]),
                          "id_mass": best_val,
                          "control_same_head": float(idm_c.mean(dim=2)[best])},
            "shakespeare_layer_local_e013a": e013a["layer_local_mass"],
        },
        "verdicts": {
            "P1_copy_ge_80": P1,
            "P2_far_value_ge_1_and_control_0": P2,
            "P3_attention_concentrates_on_ID": P3,
            "P4_depth_mode": P4,
        },
        "runtime_s": round(time.time() - t_start, 1),
    }
    save_json(rd / "metrics.json", metrics)
    print(f"outputs: {rd}  (runtime {metrics['runtime_s']}s)")


if __name__ == "__main__":
    main()
