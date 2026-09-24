"""E038 — Causal lesion of the retrieval head L4-H1 (e021 follow-up slot).

E021 found head L4-H1 carries 95.1% of its attention mass on the ID nonce at
COPY cues (n=1 correlation). This experiment converts that into a causal
claim: zero ONLY head L4-H1 (its 32-dim slice of the c_proj input, the exact
e001 head-lesion hook) and measure on the task corpus val split:
  (a) copy accuracy at COPY nonce positions (was 99.96%);
  (b) CE at COPY nonce positions (was 0.007 nats; chance ~ ln 26 = 3.258);
  (c) filler-position CE (locality: does the rest of the net care?).
Also zero L4-H1 in the CONTROL net for contrast (its same head has 15% ID-mass).

REGISTERED verdict (slot card): if copy accuracy collapses (>50% drop) with
filler CE moving <0.15, L4-H1 is CAUSALLY the retrieval circuit; if copy
survives (<=50% drop), other heads carry it and the "dedicated head"
reading dies.

Deviations from the slot card:
- copy-accuracy measurement reuses e021's exact selection protocol (stride
  subsample of val COPY targets, n=2500) for comparability with the
  "was 100% / CE 0.007" reference numbers.
- per-nonce-index (k=0..4) accuracy recorded (free); whole-corpus val CE
  under the head-zero recorded (free, one estimate_loss per arm).

Run: python lab/e038_head_lesion.py
"""
from __future__ import annotations

import json
import time

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, estimate_loss,
                    lesion, run_dir, save_json, set_seed)
from e021_task_swap import (CTRL_CKPT, CTRL_TXT, TASK_CKPT, TASK_TXT,
                            build_corpus, last_pred)

import matplotlib.pyplot as plt
import torch

E021_M = REPO / "runs" / "e021" / "metrics.json"
HEAD_LAYER, HEAD_IDX = 4, 1
N_EVAL = 2500
N_FILLER = 2500
FILLER_SEED = 38
N_BATCHES = 30
LN26 = float(torch.log(torch.tensor(26.0)))


def load_net(txt_path, ckpt_path, tag):
    corpus = CharCorpus(txt_path, seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    st = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(st["model"])
    model.eval()
    print(f"[{tag}] ckpt step {st['step']}")
    return model, corpus


@torch.no_grad()
def arm(model, ids, sel, ks=None):
    """CE + top-1 accuracy at the selected target positions (256-token windows)."""
    xs = torch.stack([ids[t - 256:t] for t in sel])
    ys = ids[sel]
    ce, acc = last_pred(model, xs, ys)
    out = {"ce_mean": float(ce.mean()), "ce_median": float(ce.median()),
           "acc": float(acc.mean()), "n": len(sel)}
    if ks is not None:
        out["acc_by_k"] = [float(acc[ks == k].mean()) for k in range(5)]
        out["ce_by_k"] = [float(ce[ks == k].mean()) for k in range(5)]
    return out


def main():
    t_start = time.time()
    rd = run_dir("e038")
    set_seed(42)
    e021 = json.loads(E021_M.read_text())

    meta, _ = build_corpus()
    tmodel, tcorpus = load_net(TASK_TXT, TASK_CKPT, "task")
    cmodel, ccorpus = load_net(CTRL_TXT, CTRL_CKPT, "control")
    t_ids = torch.cat([tcorpus.train, tcorpus.val])
    c_ids = torch.cat([ccorpus.train, ccorpus.val])
    val_start = len(tcorpus.train)
    eval_docs = [(i, c) for (i, c) in meta if i[0] >= val_start]
    copy_targets = torch.tensor([t for (_, c) in eval_docs for t in c], dtype=torch.long)
    k_all = torch.arange(len(copy_targets)) % 5
    stride = max(1, len(copy_targets) // N_EVAL)
    sel = copy_targets[::stride][:N_EVAL]
    k_sel = k_all[::stride][:N_EVAL]

    filler = []
    for id_pos, copy_pos in eval_docs:
        pos = id_pos[0] - 4
        f_len = copy_pos[0] - pos - 17
        filler.extend(range(pos + 10, pos + 10 + f_len))
    filler = torch.tensor(filler, dtype=torch.long)
    gen = torch.Generator().manual_seed(FILLER_SEED)
    fill_sel = filler[torch.randperm(len(filler), generator=gen)[:N_FILLER]]
    print(f"val docs {len(eval_docs)} | COPY targets {len(sel)} | filler targets {len(fill_sel)}")

    results = {}
    for tag, model, corpus, ids in (("task", tmodel, tcorpus, t_ids),
                                    ("control", cmodel, ccorpus, c_ids)):
        base_copy = arm(model, ids, sel, k_sel)
        base_fill = arm(model, ids, fill_sel)
        base_val = estimate_loss(model, corpus, "val", n_batches=N_BATCHES)
        with lesion(model, "head", HEAD_LAYER, head=HEAD_IDX):
            les_copy = arm(model, ids, sel, k_sel)
            les_fill = arm(model, ids, fill_sel)
            les_val = estimate_loss(model, corpus, "val", n_batches=N_BATCHES)
        results[tag] = {"base_copy": base_copy, "les_copy": les_copy,
                        "base_fill": base_fill, "les_fill": les_fill,
                        "base_val_ce": base_val, "les_val_ce": les_val}
        print(f"[{tag}] copy acc {base_copy['acc']:.4f} -> {les_copy['acc']:.4f} | "
              f"CE@COPY {base_copy['ce_mean']:.4f} -> {les_copy['ce_mean']:.4f} "
              f"(chance {LN26:.3f}) | filler CE {base_fill['ce_mean']:.4f} -> "
              f"{les_fill['ce_mean']:.4f} (Δ {les_fill['ce_mean'] - base_fill['ce_mean']:+.4f}) "
              f"| val CE {base_val:.4f} -> {les_val:.4f}")

    t, c = results["task"], results["control"]
    acc_drop_frac = 1.0 - t["les_copy"]["acc"] / max(t["base_copy"]["acc"], 1e-9)
    filler_move = abs(t["les_fill"]["ce_mean"] - t["base_fill"]["ce_mean"])
    copy_ce_rise = t["les_copy"]["ce_mean"] - t["base_copy"]["ce_mean"]
    collapse = bool(acc_drop_frac > 0.50)
    local = bool(filler_move < 0.15)
    verdict = ("CAUSAL: L4-H1 is the retrieval circuit"
               if collapse and local else
               "NOT-CAUSAL: other heads carry the copy" if not collapse else
               "COLLAPSE-BUT-NONLOCAL: copy dies AND filler CE moves ≥0.15")
    print(f"\nREGISTERED VERDICT: {verdict}")
    print(f"  copy acc drop {acc_drop_frac:.1%} (>50%?) | filler CE move "
          f"{filler_move:.4f} (<0.15?) | CE@COPY rise {copy_ce_rise:+.3f} nats")

    # ---------------- plots ----------------
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    conds = [("task base", t["base_copy"]["acc"], "seagreen"),
             ("task L4H1=0", t["les_copy"]["acc"], "crimson"),
             ("ctrl base", c["base_copy"]["acc"], "lightgray"),
             ("ctrl L4H1=0", c["les_copy"]["acc"], "gray")]
    axes[0].bar(range(4), [v for _, v, _ in conds], color=[col for _, _, col in conds])
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels([n for n, _, _ in conds], fontsize=8)
    axes[0].axhline(1 / 26, color="k", ls=":", label="chance 1/26")
    axes[0].set_title("copy accuracy at COPY nonce"); axes[0].legend(fontsize=7)

    axes[1].bar([0, 1], [t["base_copy"]["ce_mean"], t["les_copy"]["ce_mean"]],
                color=["seagreen", "crimson"])
    axes[1].bar([2, 3], [c["base_copy"]["ce_mean"], c["les_copy"]["ce_mean"]],
                color=["lightgray", "gray"])
    axes[1].axhline(LN26, color="k", ls=":", label="ln 26 (chance)")
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels([n for n, _, _ in conds], fontsize=8)
    axes[1].set_title("CE at COPY nonce positions"); axes[1].legend(fontsize=7)

    axes[2].bar([0, 1], [t["base_fill"]["ce_mean"], t["les_fill"]["ce_mean"]],
                color=["seagreen", "crimson"], label="task")
    axes[2].bar([2, 3], [c["base_fill"]["ce_mean"], c["les_fill"]["ce_mean"]],
                color=["lightgray", "gray"], label="control")
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels(
        ["task base", "task L4H1=0", "ctrl base", "ctrl L4H1=0"], fontsize=8)
    axes[2].set_title(f"filler CE (locality: Δtask {filler_move:+.3f})"); axes[2].legend(fontsize=7)

    xs5 = range(5)
    axes[3].plot(xs5, t["base_copy"]["acc_by_k"], "o-", color="seagreen", label="task base")
    axes[3].plot(xs5, t["les_copy"]["acc_by_k"], "s-", color="crimson", label="task L4H1=0")
    axes[3].axhline(1 / 26, color="k", ls=":", label="chance")
    axes[3].set_xticks(xs5); axes[3].set_xticklabels([f"nonce[{k}]" for k in xs5])
    axes[3].set_title("copy accuracy by nonce index"); axes[3].legend(fontsize=7)

    for ax in axes:
        ax.set_ylabel("acc" if ax in (axes[0], axes[3]) else "CE (nats)")
    fig.suptitle(f"E038 — zero head L4-H1 only (e001 hook): {verdict}")
    fig.tight_layout(); fig.savefig(rd / "head_lesion.png", dpi=140); plt.close(fig)

    metrics = {
        "experiment": "e038_head_lesion",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "head": {"layer": HEAD_LAYER, "head": HEAD_IDX,
                 "e021_id_mass": e021["attention_census"]["best_head"]["id_mass"]},
        "e021_reference": {"copy_acc": e021["copy_accuracy"]["task"],
                           "ce_at_copy": e021["far_value"]["task_ce_full"]},
        "task": t, "control": c,
        "verdict": {
            "registered_rule": "collapse = copy acc drop >50% AND filler CE move <0.15 "
                               "-> CAUSAL; copy survives -> other heads carry it",
            "acc_drop_fraction": acc_drop_frac,
            "filler_ce_move": filler_move,
            "copy_ce_rise": copy_ce_rise,
            "collapse": collapse, "local": local,
            "verdict": verdict,
        },
        "runtime_s": round(time.time() - t_start, 1),
    }
    save_json(rd / "metrics.json", metrics)
    print(f"outputs: {rd}  (runtime {metrics['runtime_s']}s)")


if __name__ == "__main__":
    main()
