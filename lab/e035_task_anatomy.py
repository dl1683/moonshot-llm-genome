"""E035 — Task-net anatomy: the stage picture under retrieval demand (e021 follow-up slot).

Eval-only, on the EXISTING e021 task-net checkpoint. Checkpoint is
runs/checkpoints/e021_task.train.pt (deviation note: slot card wrote
"e021_task.pt"; verified against e021_task_swap.py — actual name is
e021_task.train.pt, a dict ckpt with a "model" state dict). Corpus:
data/e021_task.txt val split; doc layout/meta rebuilt deterministically via
e021's build_corpus (GEN_SEED 2100).

  Q1  Full lesion map (zero attn / mlp block per layer, 30 batches) on the
      task corpus val vs the Shakespeare e001 profile. KEY: does L4-attn
      damage spike (the retrieval layer)? does the whole allocation
      reorganize?
  Q2  Decision-depth census on NON-COPY (filler) positions (~1000) vs
      Shakespeare e012. Pre-registered readout: "TWO-PROFILES" if
      Pearson(filler hist, Shakespeare hist) >= 0.8 AND JS(filler hist,
      task-COPY hist) > 0.10; "PIPELINE-SHIFTED" if Pearson < 0.5; else
      "PARTIAL". Free contrast: control-net filler census.
  Q3  ΔW quick pass: cos(dW_task, dW_B) at the shared seed-42 init for the
      stream-facing matrices W_in / W_out per layer (+ c_attn / c_proj as
      context), plus the v009 top-16 right-singular subspace alignment vs
      the B<->R same-seed values (W_out band ~0.15-0.25). Task pressure
      PRESERVES init-anchoring if task<->B alignment stays at/above the
      B<->R band (well above the random floor sqrt(K/d_in)); it DESTROYS it
      if it falls to the floor (0.144 @ d=768, 0.289 @ d=192).

Deviations from the slot card:
- census also records depth-entropy Spearman (e012 instrument, free).
- ΔW computed BOTH as raw flattened cosine and v009 subspace alignment
  (the card's "0.15-0.25 band" refers to the v009 B<->R subspace numbers;
  raw cos links to e029's full-organ +0.152).
- control-net filler census added as free contrast (registration only
  requires the task net).

Run: python lab/e035_task_anatomy.py
"""
from __future__ import annotations

import json
import time

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, estimate_loss,
                    lesion_loss, run_dir, save_json, set_seed)
from e021_task_swap import (CTRL_CKPT, CTRL_TXT, TASK_CKPT, TASK_TXT,
                            batched_snapshots, build_corpus, js_divergence)

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

E001_M = REPO / "runs" / "e001" / "metrics.json"
E012_M = REPO / "runs" / "e012" / "metrics.json"
E021_M = REPO / "runs" / "e021" / "metrics.json"
V009_M = REPO / "runs" / "v009" / "metrics.json"
B_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"

SEED = 42
N_BATCHES = 30
N_DEPTH_POS = 1000
CENSUS_SEED = 35
BATCH = 64
KSVD = 16


# ---------------------------------------------------------------- loading


def load_net(txt_path, ckpt_path, tag):
    corpus = CharCorpus(txt_path, seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    st = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(st["model"])
    model.eval()
    print(f"[{tag}] ckpt step {st['step']} (final hist val "
          f"{st['history'][-1]['val_loss']:.4f}) | vocab {corpus.vocab_size}")
    return model, corpus


# ---------------------------------------------------------------- Q1 lesion map


def q1_lesion(model, corpus):
    base = estimate_loss(model, corpus, "val", n_batches=N_BATCHES)
    print(f"baseline val CE on task corpus: {base:.4f}")
    attn_d, mlp_d = [], []
    for i in range(model.cfg.n_layer):
        attn_d.append(lesion_loss(model, corpus, "attn", i, n_batches=N_BATCHES) - base)
        mlp_d.append(lesion_loss(model, corpus, "mlp", i, n_batches=N_BATCHES) - base)
        print(f"  L{i}: attn-zero {attn_d[-1]:+.3f} | mlp-zero {mlp_d[-1]:+.3f}")
    return base, attn_d, mlp_d


# ---------------------------------------------------------------- Q2 depth census


def filler_positions(meta, val_start):
    """Target positions whose next char is a FILLER char (never ID/COPY nonce,
    never the 'ID: '/'COPY: ' structure): strictly inside the filler span."""
    out = []
    for id_pos, copy_pos in meta:
        if id_pos[0] < val_start:
            continue
        pos = id_pos[0] - 4
        f_len = copy_pos[0] - pos - 17
        out.extend(range(pos + 10, pos + 10 + f_len))
    return torch.tensor(out, dtype=torch.long)


@torch.no_grad()
def census(model, ids, targets, n_pos, seed):
    """e012-style decision depth + final entropy at the given targets."""
    gen = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(targets), generator=gen)[:n_pos]
    ts = targets[pick]
    depths = torch.full((len(ts),), -1, dtype=torch.long)
    entrop = torch.zeros(len(ts))
    for b0 in range(0, len(ts), BATCH):
        xb = torch.stack([ids[t - 256:t] for t in ts[b0:b0 + BATCH]]).to(DEVICE)
        last = batched_snapshots(model, xb)
        probs = [F.softmax(model.lm_head(model.ln_f(v)), -1) for v in last]
        top1 = torch.stack([p.argmax(-1) for p in probs])
        final = top1[-1]
        for d in range(model.cfg.n_layer + 1):
            stable_d = (top1[d:] == final.unsqueeze(0)).all(dim=0)
            sel = (depths[b0:b0 + BATCH] < 0) & stable_d.cpu()
            depths[b0:b0 + BATCH][sel] = d
        p6 = probs[-1]
        entrop[b0:b0 + BATCH] = -(p6 * (p6 + 1e-12).log()).sum(-1).cpu()
    hist = torch.bincount(depths[depths >= 0], minlength=model.cfg.n_layer + 1).float()
    return depths, entrop, hist, int((depths < 0).sum())


def ranks(x):
    return torch.argsort(torch.argsort(x)).float()


def spearman(a, b):
    ra, rb = ranks(torch.tensor(a, dtype=torch.float)), ranks(torch.tensor(b, dtype=torch.float))
    ra = (ra - ra.mean()) / ra.std().clamp_min(1e-8)
    rb = (rb - rb.mean()) / rb.std().clamp_min(1e-8)
    return float((ra * rb).mean())


def pearson(a, b):
    a = torch.tensor(a, dtype=torch.float)
    b = torch.tensor(b, dtype=torch.float)
    a = (a - a.mean()) / a.std().clamp_min(1e-8)
    b = (b - b.mean()) / b.std().clamp_min(1e-8)
    return float((a * b).mean())


# ---------------------------------------------------------------- Q3 dW atlas


def q3_dw():
    task_sd = torch.load(TASK_CKPT, map_location="cpu", weights_only=False)["model"]
    b_sd = torch.load(B_CKPT, map_location="cpu", weights_only=True)
    set_seed(SEED)  # e001/e021 both did set_seed(42) -> CharCorpus (no RNG) -> TinyGPT(cfg)
    init_model = TinyGPT(Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192, block_size=256))
    init = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
    subs = {"c_attn": "h.{l}.attn.c_attn.weight", "c_proj": "h.{l}.attn.c_proj.weight",
            "W_in": "h.{l}.mlp.0.weight", "W_out": "h.{l}.mlp.2.weight"}
    d_in = {"c_attn": 192, "c_proj": 192, "W_in": 192, "W_out": 768}
    ref = json.loads(V009_M.read_text())["alignment_pairs"]["B<->R"]["values"]
    rows = []
    for sub, pat in subs.items():
        for l in range(6):
            key = pat.format(l=l)
            dt = task_sd[key].double() - init[key].double()
            db = b_sd[key].double() - init[key].double()
            raw_cos = float(torch.dot(dt.flatten(), db.flatten()) / (dt.norm() * db.norm()))
            _, _, Vt = torch.linalg.svd(dt, full_matrices=False)
            _, _, Vb = torch.linalg.svd(db, full_matrices=False)
            align = float(torch.linalg.norm(Vt[:KSVD] @ Vb[:KSVD].T) / KSVD ** 0.5)
            rows.append({"sub": sub, "layer": l, "raw_cos": raw_cos,
                         "subspace_align_task_vs_B": align,
                         "v009_B_vs_R_same_seed": ref[f"L{l}|{sub}"],
                         "random_floor": (KSVD / d_in[sub]) ** 0.5,
                         "fro_task": float(dt.norm()), "fro_B": float(db.norm())})
            print(f"  {sub} L{l}: raw cos {raw_cos:+.3f} | subspace {align:.3f} "
                  f"(B-R ref {ref[f'L{l}|{sub}']:.3f}, floor {rows[-1]['random_floor']:.3f})")
    return rows


# ---------------------------------------------------------------- main


def main():
    t_start = time.time()
    rd = run_dir("e035")
    set_seed(SEED)
    e001 = json.loads(E001_M.read_text())
    e012 = json.loads(E012_M.read_text())
    e021 = json.loads(E021_M.read_text())

    tmodel, tcorpus = load_net(TASK_TXT, TASK_CKPT, "task")
    print("=== Q1: lesion map on task corpus ===")
    base, attn_d, mlp_d = q1_lesion(tmodel, tcorpus)

    shake_attn, shake_mlp = e001["attn_block_damage"], e001["mlp_block_damage"]
    l4_spike = attn_d[4] - shake_attn[4]
    reorg = {"spearman_attn": spearman(attn_d, shake_attn),
             "spearman_mlp": spearman(mlp_d, shake_mlp),
             "attn_l4_gt_l3_task": bool(attn_d[4] > attn_d[3]),
             "attn_l4_shakespeare": shake_attn[4]}
    print(f"L4-attn: task {attn_d[4]:+.3f} vs Shakespeare {shake_attn[4]:+.3f} "
          f"(spike {l4_spike:+.3f}) | profile corr attn {reorg['spearman_attn']:.2f} "
          f"mlp {reorg['spearman_mlp']:.2f}")

    print("=== Q2: decision depth at FILLER positions ===")
    meta, _ = build_corpus()
    t_ids = torch.cat([tcorpus.train, tcorpus.val])
    val_start = len(tcorpus.train)
    ftgt = filler_positions(meta, val_start)
    print(f"val filler target positions: {len(ftgt)}")
    d_t, e_t, hist_t, unst_t = census(tmodel, t_ids, ftgt, N_DEPTH_POS, CENSUS_SEED)
    decided_t = d_t >= 0
    rho_t = spearman(d_t[decided_t].float().tolist(), e_t[decided_t].tolist())

    cmodel, ccorpus = load_net(CTRL_TXT, CTRL_CKPT, "control")
    c_ids = torch.cat([ccorpus.train, ccorpus.val])
    d_c, e_c, hist_c, unst_c = census(cmodel, c_ids, ftgt, N_DEPTH_POS, CENSUS_SEED)
    decided_c = d_c >= 0
    rho_c = spearman(d_c[decided_c].float().tolist(), e_c[decided_c].tolist())

    shake_hist = e012["depth_histogram"] + [0.0]        # 8 bins incl. unstable
    fill_hist = hist_t.tolist() + [float(unst_t)]
    ctrl_hist = hist_c.tolist() + [float(unst_c)]
    copy_hist = e021["depth_census"]["task_counts"]      # already 8 bins
    js_fill_shake = js_divergence(fill_hist, shake_hist)
    js_fill_copy = js_divergence(fill_hist, copy_hist)
    r_fill_shake = pearson(fill_hist, shake_hist)
    r_fill_copy = pearson(fill_hist, copy_hist)
    r_copy_shake = pearson(copy_hist, shake_hist)
    if r_fill_shake >= 0.8 and js_fill_copy > 0.10:
        q2_verdict = "TWO-PROFILES"
    elif r_fill_shake < 0.5:
        q2_verdict = "PIPELINE-SHIFTED"
    else:
        q2_verdict = "PARTIAL"
    print(f"filler hist: {[int(v) for v in fill_hist]} unstable {unst_t} "
          f"(rho depth-entropy {rho_t:+.3f})")
    print(f"ctrl  hist: {[int(v) for v in ctrl_hist]} unstable {unst_c} "
          f"(rho {rho_c:+.3f})")
    print(f"JS(filler, Shakespeare) {js_fill_shake:.3f} | r {r_fill_shake:.3f} | "
          f"JS(filler, COPY) {js_fill_copy:.3f} | r {r_fill_copy:.3f} -> {q2_verdict}")

    print("=== Q3: dW atlas quick pass (task vs B, shared init s42) ===")
    dw_rows = q3_dw()
    for sub in ("W_in", "W_out"):
        a_tb = sum(r["subspace_align_task_vs_B"] for r in dw_rows if r["sub"] == sub) / 6
        a_br = sum(r["v009_B_vs_R_same_seed"] for r in dw_rows if r["sub"] == sub) / 6
        floor = (KSVD / (192 if sub == "W_in" else 768)) ** 0.5
        print(f"  {sub}: mean task-B {a_tb:.3f} vs mean B-R {a_br:.3f} (floor {floor:.3f})")

    # ---------------- plots ----------------
    Ls = list(range(6))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    w = 0.38
    axes[0].bar([i - w / 2 for i in Ls], attn_d, w, color="crimson", label="task net (task corpus)")
    axes[0].bar([i + w / 2 for i in Ls], shake_attn, w, color="steelblue", alpha=0.75,
                label="Shakespeare (e001)")
    axes[0].set_title(f"attention block zeroed — L4 spike {l4_spike:+.2f} nats")
    axes[1].bar([i - w / 2 for i in Ls], mlp_d, w, color="darkorange", label="task net")
    axes[1].bar([i + w / 2 for i in Ls], shake_mlp, w, color="steelblue", alpha=0.75,
                label="Shakespeare (e001)")
    axes[1].set_title("MLP block zeroed")
    for ax in axes:
        ax.set_xlabel("layer"); ax.set_ylabel("Δ val CE (nats)"); ax.set_xticks(Ls); ax.legend()
    fig.suptitle(f"E035 Q1 — lesion map under retrieval demand (task val CE {base:.3f})")
    fig.tight_layout(); fig.savefig(rd / "lesion_map.png", dpi=140); plt.close(fig)

    labels8 = ["emb", "L0", "L1", "L2", "L3", "L4", "L5", "unstable"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs8 = torch.arange(8)
    w = 0.2
    for off, h, lab, col in ((-1.5 * w, shake_hist, "Shakespeare (e012)", "steelblue"),
                             (-0.5 * w, fill_hist, "task @ FILLER", "seagreen"),
                             (0.5 * w, copy_hist, "task @ COPY (e021)", "crimson"),
                             (1.5 * w, ctrl_hist, "control @ FILLER", "gray")):
        ax.bar(xs8 + off, torch.tensor(h, dtype=torch.float) / sum(h), w, color=col, label=lab)
    ax.set_xticks(xs8); ax.set_xticklabels(labels8)
    ax.set_ylabel("fraction of positions")
    ax.set_title(f"E035 Q2 — filler depth vs Shakespeare: JS {js_fill_shake:.3f}, "
                 f"r {r_fill_shake:.3f}; vs COPY: JS {js_fill_copy:.3f} → {q2_verdict}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(rd / "filler_depth.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, sub in enumerate(("W_in", "W_out")):
        rows_s = [r for r in dw_rows if r["sub"] == sub]
        axes[0, j].plot(Ls, [r["subspace_align_task_vs_B"] for r in rows_s], "o-",
                        color="seagreen", label="task ↔ B (this run)")
        axes[0, j].plot(Ls, [r["v009_B_vs_R_same_seed"] for r in rows_s], "s--",
                        color="gray", label="B ↔ R (v009, same-seed)")
        axes[0, j].axhline(rows_s[0]["random_floor"], color="k", ls=":", label="random floor")
        axes[0, j].set_title(f"{sub}: top-16 right-singular alignment"); axes[0, j].legend(fontsize=8)
        axes[1, j].plot(Ls, [r["raw_cos"] for r in rows_s], "o-", color="crimson",
                        label="raw cos(ΔW_task, ΔW_B)")
        axes[1, j].axhline(0, color="k", lw=0.8)
        axes[1, j].set_title(f"{sub}: raw flattened cosine"); axes[1, j].legend(fontsize=8)
        for axr in (axes[0, j], axes[1, j]):
            axr.set_xlabel("layer"); axr.set_ylabel("alignment / cos"); axr.set_xticks(Ls)
    fig.suptitle("E035 Q3 — ΔW init-anchoring under task pressure (shared seed-42 init)")
    fig.tight_layout(); fig.savefig(rd / "dw_atlas_quick.png", dpi=140); plt.close(fig)

    metrics = {
        "experiment": "e035_task_anatomy",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": {"task": TASK_CKPT.name, "control": CTRL_CKPT.name,
                       "shakespeare_B": B_CKPT.name},
        "q1_lesion": {
            "baseline_val_ce_task_corpus": base,
            "attn_block_damage": attn_d, "mlp_block_damage": mlp_d,
            "shakespeare_attn_block_damage": shake_attn,
            "shakespeare_mlp_block_damage": shake_mlp,
            "l4_attn_spike_vs_shakespeare": l4_spike,
            "l4_gt_l3_in_task": bool(attn_d[4] > attn_d[3]),
            "profile_rank_corr": reorg,
        },
        "q2_filler_depth": {
            "n_positions": N_DEPTH_POS,
            "task_filler_counts": fill_hist, "task_filler_unstable": unst_t,
            "control_filler_counts": ctrl_hist, "control_filler_unstable": unst_c,
            "shakespeare_counts_e012": shake_hist, "task_copy_counts_e021": copy_hist,
            "js_filler_vs_shakespeare": js_fill_shake,
            "pearson_filler_vs_shakespeare": r_fill_shake,
            "js_filler_vs_copy": js_fill_copy,
            "pearson_filler_vs_copy": r_fill_copy,
            "pearson_copy_vs_shakespeare": r_copy_shake,
            "spearman_depth_entropy_task_filler": rho_t,
            "spearman_depth_entropy_control_filler": rho_c,
            "verdict": q2_verdict,
        },
        "q3_dw": {"k": KSVD, "rows": dw_rows,
                  "note": "raw_cos = flattened full-matrix cosine of ΔW; "
                          "subspace_align = v009 protocol (top-16 right singulars, "
                          "||PaᵀPb||_F/√K); v009_B_vs_R is the same-seed-42 "
                          "renorm-vs-base reference pair."},
        "runtime_s": round(time.time() - t_start, 1),
    }
    save_json(rd / "metrics.json", metrics)
    print(f"outputs: {rd}  (runtime {metrics['runtime_s']}s)")


if __name__ == "__main__":
    main()
