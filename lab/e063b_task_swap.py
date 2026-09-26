"""E063b — TASK-SWAP DISCRIMINATOR (T041): is the universal organ-reliance
template TASK-PINNED or an OPTIMIZER-ATTRACTOR?

T041 established (e063) that the own-organ load A profile — L0 huge / L1
trough / monotone rise — is a universal emergent template on the Shakespeare
char-LM: shape r = +1.000 across independent inits AND data orders (2.7M
cohort). Two live explanations:

  H-i  TASK-PINNED: the char-LM error structure fixes where capacity must
       sit; any competent optimizer lands there. Template tracks the TASK.
  H-ii OPTIMIZER-ATTRACTOR: SGD's basin on this architecture is that deep;
       the template is the attractor and corpus details don't matter.

REGISTERED DISCRIMINATOR (T041, frozen): re-measure A on the e021 task-swap
checkpoints — same 6L/6H/192d arch, trained on a corpus whose task structure
DIFFERS (docs "ID: XXXXX\\n<filler>\\nCOPY: XXXXX\\n"; the task net must copy
a nonce from >16 tokens back; the control net's cue is uncorrelated). Each
net is evaluated on ITS OWN corpus val split (e021's eval protocol: batches
sampled with the corpus seed from that corpus's val tail), A machinery reused
verbatim from e063 (lesion dCE per MLP site, 15 fixed val batches, bootstrap
se).

REGISTERED READOUT & PREDICTIONS:
  shape = Pearson r of the 6-site A profile vs the B (Shakespeare e001)
  template, measured with the identical instrument in this process.
    H-i  predicts r < 0.5 for at least the copy-task net (task re-weights
         allocation);
    H-ii predicts r >= 0.8 everywhere.
  Supportive (honesty, not registered): Spearman rho; Pearson over sites
  1-5 only (L0's magnitude dominates the 6-point r); mean |dA| rung vs the
  e063 noise floor 0.0142.

CHECKPOINT REALITY (verified before running — differs from T041's
parenthetical "copy-task vs word-shuffled vs Shakespeare"): there is NO
word-shuffled net. runs/checkpoints has exactly two e021-family ckpts:
  e021_task.train.pt    step 1980 (252s cap), copy acc 0.9996, far-val +3.27
  e021_control.train.pt step  913 (252s cap), copy acc 0.041, far-val -0.00
Both are trainstate dicts (model under ['model']), 6L/6H/192d, vocab 65 —
arch-compatible with the B cohort. The control (byte-identical corpus, cue
uncorrelated) is a bonus contrast; the registered decision rests on the
copy-task net. Caveat recorded: control saw ~half the steps (wall-clock cap).

Run: python lab/e063b_task_swap.py   (CPU-only; CUDA masked pre-torch)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"             # strict CPU-only

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sstats

import common

common.DEVICE = "cpu"
from common import (REPO, Cfg, CharCorpus, TinyGPT, lesion, run_dir,
                    save_json)

CORPUS_PATH = REPO / "data" / "input.txt"
CKPT = REPO / "runs" / "checkpoints"
N_EVAL = 15                                        # e052/e063 fixed-batch protocol
N_BOOT = 2000
T0 = time.time()
SITES = list(range(6))                             # B-family 2.7M = 6L/6H/192d
NOISE_B_E063 = 0.014242571756562152                # e063 cohort-B bootstrap floor

# name -> (group, corpus file, ckpt stem, note). Every net is 6L/6H/192d.
NETS = {
    "B_e001": ("anchor", "input.txt", "e001",
               "B template: Shakespeare char-LM (e001, init42/order1337)"),
    "e021_task": ("copy-task", "e021_task.txt", "e021_task.train",
                  "task net: ID->COPY nonce retrieval required (step 1980, "
                  "copy acc .9996, far-val +3.27)"),
    "e021_control": ("control", "e021_control.txt", "e021_control.train",
                     "control net: byte-identical corpus, COPY cue "
                     "uncorrelated (step 913, far-val ~0)"),
}

# e063 cached B template (instrument cross-check gate; re-measured here).
E063_B_TEMPLATE = {"base_ce": 1.6343892574310304,
                   "A": {0: 4.079657411575317, 1: 0.15235294500986735,
                         2: 0.2660760561625163, 3: 0.4587578614552816,
                         4: 0.5932433764132812, 5: 0.5844108581542968},
                   "se": {0: 0.04241007754254047, 1: 0.004947139125580957,
                          2: 0.006886085108418278, 3: 0.007462882892482038,
                          4: 0.010776002345552292, 5: 0.016309975568410262}}
E021_REF = {"val_ce_task_30batch": 1.4318429072697958,
            "val_ce_control_30batch": 1.551924721399943,
            "copy_acc_task": 0.9995999932289124,
            "far_value_task": 3.268735647201538,
            "far_value_control": -0.0018345688004046679}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def load_sd(name: str) -> dict:
    obj = torch.load(CKPT / f"{name}.pt", map_location="cpu", weights_only=False)
    return obj["model"] if "model" in obj else obj


@torch.no_grad()
def per_batch_losses_cpu(model, corpus, n_batches: int = N_EVAL) -> list[float]:
    """e052/e063 fixed-batch protocol (RNG seeded by corpus.seed=1337)."""
    model.eval()
    cfg = model.cfg
    gen = torch.Generator().manual_seed(corpus.seed)
    src = corpus.val
    losses = []
    for _ in range(n_batches):
        ix = torch.randint(len(src) - cfg.block_size - 1, (16,), generator=gen)
        x = torch.stack([src[i: i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1: i + 1 + cfg.block_size] for i in ix])
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return losses


def name_seed(name: str) -> int:
    return 30630 + sum(ord(c) * (i + 1) for i, c in enumerate(name))


def boot_se(diffs: list[float], seed: int) -> float:
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs)
    means = rng.choice(arr, size=(N_BOOT, arr.size), replace=True).mean(axis=1)
    return float(means.std(ddof=1))


def measure_net(ckpt_stem: str, group: str, note: str, corpus) -> dict:
    cfg = Cfg(vocab=corpus.vocab_size, block_size=256,
              n_layer=6, n_head=6, n_embd=192)
    net = TinyGPT(cfg)
    net.load_state_dict(load_sd(ckpt_stem))
    base = per_batch_losses_cpu(net, corpus)
    A, se, per_batch = {}, {}, {}
    for s in SITES:
        with lesion(net, "mlp", s):
            abl = per_batch_losses_cpu(net, corpus)
        diffs = [a - b for a, b in zip(abl, base)]
        A[s] = float(np.mean(diffs))
        se[s] = boot_se(diffs, seed=name_seed(ckpt_stem) + s)
        per_batch[s] = diffs
    return {"group": group, "note": note, "base_ce": float(np.mean(base)),
            "A": A, "se": se, "per_batch": per_batch}


def shape_stats(prof: dict[int, float], ref: dict[int, float]) -> dict:
    """Registered shape readout vs the B template + supportive variants."""
    a = np.array([prof[s] for s in SITES], dtype=float)
    b = np.array([ref[s] for s in SITES], dtype=float)
    r_all = float(sstats.pearsonr(a, b)[0])
    rho_all = float(sstats.spearmanr(a, b)[0])
    r_deep = float(sstats.pearsonr(a[1:], b[1:])[0])          # L0 excluded
    return {"pearson_r_all_sites": r_all, "spearman_rho_all_sites": rho_all,
            "pearson_r_sites1_5_excl_L0": r_deep,
            "mean_abs_dA": float(np.mean(np.abs(a - b))),
            "profile": {str(s): float(prof[s]) for s in SITES}}


def main():
    rd = run_dir("e063b")
    log("E063b task-swap discriminator (T041) — A profiles on the e021 "
        "task-swap checkpoints (CPU-only)")

    nets: dict[str, dict] = {}
    for name, (group, corpus_file, ckpt_stem, note) in NETS.items():
        corpus = CharCorpus(REPO / "data" / corpus_file, seed=1337)
        nets[name] = measure_net(ckpt_stem, group, note, corpus)
        m = nets[name]
        log(f"  {name:13s} [{group:9s}] vocab {corpus.vocab_size} | base CE "
            f"{m['base_ce']:.4f} | "
            + " ".join(f"A@L{s} {m['A'][s]:+.3f}" for s in SITES))

    # ---- instrument gates ------------------------------------------------------
    B = nets["B_e001"]
    dA_cache = float(np.mean([abs(B["A"][s] - E063_B_TEMPLATE["A"][s])
                              for s in SITES]))
    gate_cache = dA_cache <= 3.0 * float(np.mean(list(E063_B_TEMPLATE["se"].values())))
    gate_task_ce = abs(B["base_ce"] - E063_B_TEMPLATE["base_ce"]) < 0.05
    gate_task = abs(nets["e021_task"]["base_ce"] - E021_REF["val_ce_task_30batch"]) < 0.10
    gate_ctrl_ce = abs(nets["e021_control"]["base_ce"] - E021_REF["val_ce_control_30batch"]) < 0.10
    gates = {
        "B_remeasure_vs_e063_cache_mean_dA": dA_cache,
        "B_cache_se_mean": float(np.mean(list(E063_B_TEMPLATE["se"].values()))),
        "gate_B_matches_e063_cache(<=3se)": bool(gate_cache),
        "gate_B_base_ce_matches_e063(<0.05)": bool(gate_task_ce),
        "gate_task_base_ce_matches_e021(<0.10)": bool(gate_task),
        "gate_control_base_ce_matches_e021(<0.10)": bool(gate_ctrl_ce),
        "arch_all": "6L/6H/192d block256, vocab 65 — identical to B cohort",
        "checkpoint_reality": ("no word-shuffled net exists; e021 family = "
                               "task (step 1980) + control (step 913, "
                               "wall-clock-capped ~half exposure)"),
    }
    log(f"gates: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in gates.items()})}")

    # ---- registered readout: shape vs B template --------------------------------
    ref = B["A"]
    shapes = {n: shape_stats(m["A"], ref) for n, m in nets.items()}
    for n in ("e021_task", "e021_control"):
        s = shapes[n]
        log(f"shape {n:13s}: r {s['pearson_r_all_sites']:+.3f} | rho "
            f"{s['spearman_rho_all_sites']:+.3f} | r excl-L0 "
            f"{s['pearson_r_sites1_5_excl_L0']:+.3f} | mean|dA| "
            f"{s['mean_abs_dA']:.3f}")

    r_task = shapes["e021_task"]["pearson_r_all_sites"]
    r_ctrl = shapes["e021_control"]["pearson_r_all_sites"]
    r_all_nets = [shapes[n]["pearson_r_all_sites"] for n in ("e021_task", "e021_control")]

    # ---- registered verdict (T041) ------------------------------------------------
    if r_task < 0.5:
        verdict = ("H-i TASK-PINNED — the copy-task net's A profile shape "
                   f"r = {r_task:+.3f} < 0.5 vs the Shakespeare template; "
                   "the task's error structure determines where capacity sits")
    elif min(r_all_nets) >= 0.8:
        verdict = ("H-ii OPTIMIZER-ATTRACTOR — every task-swap net keeps the "
                   f"template (task r = {r_task:+.3f}, control r = "
                   f"{r_ctrl:+.3f}, both >= 0.8); corpus/task details don't "
                   "reshape organ-reliance")
    else:
        which = [f"{n} r={shapes[n]['pearson_r_all_sites']:+.3f}"
                 for n in ("e021_task", "e021_control")
                 if shapes[n]["pearson_r_all_sites"] < 0.8]
        verdict = ("INTERMEDIATE (neither registered branch) — copy-task r "
                   f"= {r_task:+.3f} (H-i needs < 0.5, H-ii needs >= 0.8); "
                   "below-attractor nets: " + "; ".join(which))
    log("=" * 78)
    log(f"VERDICT: {verdict}")

    # ---- outputs --------------------------------------------------------------------
    metrics = {
        "experiment": "e063b_task_swap",
        "registered_in": "T041 (2026-09-26T11:15Z)",
        "date": common.now_iso(),
        "status": "complete",
        "device": "cpu-only (CUDA_VISIBLE_DEVICES=-1; common.DEVICE=cpu)",
        "question": ("is the universal organ-reliance template task-pinned "
                     "(H-i: shape changes with task) or an optimizer-attractor "
                     "(H-ii: same shape regardless of corpus)?"),
        "protocol": {
            "A": "dCE of zeroing own mlp organ, mean over 15 fixed val "
                 "batches from EACH net's own corpus val tail (corpus seed "
                 "1337) — e063 machinery reused verbatim",
            "shape": "Pearson r of the 6-site A profile vs the B (e001 "
                     "Shakespeare) template re-measured in-process",
            "registered_rules": {"H-i": "r < 0.5 for at least the copy-task net",
                                 "H-ii": "r >= 0.8 everywhere"},
            "supportive": ["spearman rho", "pearson r over sites 1-5 (L0 "
                           "magnitude dominates the 6-point r)",
                           "mean |dA| vs e063 noise floor 0.0142"],
            "n_eval_batches": N_EVAL, "n_boot": N_BOOT,
        },
        "instrument_gates": gates,
        "checkpoints": {n: {"group": m["group"], "note": m["note"],
                            "base_ce": m["base_ce"],
                            "A_by_site": {str(s): m["A"][s] for s in SITES},
                            "bootstrap_se": {str(s): m["se"][s] for s in SITES}}
                        for n, m in nets.items()},
        "shape_vs_B_template": shapes,
        "e021_task_references": E021_REF,
        "verdict": verdict,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", metrics)

    # ---- figure -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    xs = np.arange(6)
    styles = {"B_e001": ("#2c6fbb", "-", "o", 2.4, "B: Shakespeare (e001 template)"),
              "e021_task": ("#2aa198", "-", "s", 2.2, "task: ID->COPY retrieval (step 1980)"),
              "e021_control": ("#888888", "--", "^", 1.8, "control: cue uncorrelated (step 913)")}

    ax = axes[0, 0]
    for n, (c, ls, mk, lw, lab) in styles.items():
        ax.plot(xs, [nets[n]["A"][s] for s in SITES], color=c, ls=ls,
                marker=mk, lw=lw, label=lab)
        ax.errorbar(xs, [nets[n]["A"][s] for s in SITES],
                    yerr=[nets[n]["se"][s] for s in SITES], color=c,
                    fmt="none", capsize=3, alpha=0.5)
    ax.set_xticks(xs); ax.set_xlabel("MLP site (layer)")
    ax.set_ylabel("A = own-ablation dCE (nats)")
    ax.set_title("A) A profiles at 6 MLP sites (each net on ITS OWN corpus val)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for n, (c, ls, mk, lw, lab) in styles.items():
        v = np.array([nets[n]["A"][s] for s in SITES])
        share = v / v.sum() * 100.0
        ax.plot(xs, share, color=c, ls=ls, marker=mk, lw=lw,
                label=lab.split(" (")[0])
    ax.set_xticks(xs); ax.set_xlabel("MLP site (layer)")
    ax.set_ylabel("share of total A (%)")
    ax.set_title("B) allocation shape (scale-free view)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    names = ["e021_task", "e021_control"]
    rvals = [shapes[n]["pearson_r_all_sites"] for n in names]
    rho = [shapes[n]["spearman_rho_all_sites"] for n in names]
    rdeep = [shapes[n]["pearson_r_sites1_5_excl_L0"] for n in names]
    xb = np.arange(2)
    ax.bar(xb - 0.25, rvals, 0.25, color="#2aa198", label="Pearson r (registered, 6 sites)")
    ax.bar(xb, rho, 0.25, color="#7b3fa0", label="Spearman rho")
    ax.bar(xb + 0.25, rdeep, 0.25, color="#e67e22", label="Pearson r sites 1-5 (excl L0)")
    ax.axhline(0.8, color="#c0392b", ls="--", lw=1.4, label="H-ii bar: r >= 0.8")
    ax.axhline(0.5, color="#444", ls=":", lw=1.4, label="H-i bar: r < 0.5")
    for i, (a, b_, c_) in enumerate(zip(rvals, rho, rdeep)):
        for off, v in ((-0.25, a), (0.0, b_), (0.25, c_)):
            ax.text(i + off, v + 0.02 * np.sign(v or 1), f"{v:+.2f}",
                    ha="center", fontsize=7)
    ax.set_xticks(xb); ax.set_xticklabels(names)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("shape correlation vs B template")
    ax.set_title("C) registered shape readout vs H-i / H-ii bars")
    ax.legend(fontsize=7); ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 1]
    ax.axhline(NOISE_B_E063, color="#666", ls="-.", lw=1.2,
               label=f"e063 noise floor {NOISE_B_E063:.3f}")
    for n in names:
        d = [abs(nets[n]["A"][s] - ref[s]) for s in SITES]
        ax.plot(xs, d, marker="o", color=styles[n][0], label=f"|dA| {n}")
    ax.set_xticks(xs); ax.set_xlabel("MLP site (layer)")
    ax.set_ylabel("|A(net) - A(B)| (nats)")
    ax.set_title("D) per-site distance from the Shakespeare template")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle("E063b — task-swap discriminator (T041): task-pinned or optimizer-attractor?\n"
                 f"VERDICT: {verdict.split(' — ')[0]}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(rd / "task_swap.png", dpi=140)
    plt.close(fig)
    log(f"outputs: {rd} (metrics.json, task_swap.png)")
    return metrics


if __name__ == "__main__":
    main()
