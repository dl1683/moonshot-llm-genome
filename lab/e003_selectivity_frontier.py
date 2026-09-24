"""E003 — Forgetting selectivity frontier: the T002 discriminator suite.

Four arms, each designed to kill a specific hypothesis from THINKING.md/T002:

  1. GRADIENT COSINE  (H2: structural non-separability) — cosine between
     loss-gradients on A vs B (same-corpus halves), within-half baselines, and
     A vs French (dissimilar content). Registered prediction: A↔B >= 0.85,
     A↔French far lower.
  2. FINE LR SWEEP    (H1: dose pathology) — ascent trajectories (dA, dB) at
     lr 1e-6..3e-5. Registered prediction: NO operating point reaches
     dA >= 1.0 with dB <= 0.1.
  3. DISSIMILAR ARM   (H2 again, causal side) — implant French by brief
     fine-tune, then ascend on French while measuring Shakespeare. Registered
     prediction: selectivity APPEARS for dissimilar content.
  4. FLUENCY vs CONTENT (H3: wrong measurement axis) — CE on A-unique vs
     B-unique lines after mild ascent vs generic val CE.

Run: python lab/e003_selectivity_frontier.py   (requires E001 checkpoint)
"""
import copy
import random
import unicodedata

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict, estimate_loss,
                    run_dir, save_json, set_seed)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
SEED = 31415
K_BATCH = 8          # batch pairs for cosine estimates
SWEEP_LRS = [1e-6, 3e-6, 1e-5, 3e-5]
SWEEP_STEPS = 200
IMPLANT_STEPS, IMPLANT_LR = 400, 5e-4
DIS_STEPS, DIS_LR = 300, 1e-5


# ---------------------------------------------------------------- helpers

def batch_from(src: torch.Tensor, block: int, bs: int, gen: torch.Generator):
    ix = torch.randint(len(src) - block - 1, (bs,), generator=gen)
    x = torch.stack([src[i : i + block] for i in ix]).to(DEVICE)
    y = torch.stack([src[i + 1 : i + 1 + block] for i in ix]).to(DEVICE)
    return x, y


def grad_cosine(model, src_a, src_b, block, gen):
    """Cosine between loss-gradients of one batch drawn from src_a and one from src_b."""
    def flat_grad(src):
        model.zero_grad(set_to_none=True)
        x, y = batch_from(src, block, 32, gen)
        _, l = model(x, y)
        l.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        model.zero_grad(set_to_none=True)
        return g
    g1, g2 = flat_grad(src_a), flat_grad(src_b)
    return F.cosine_similarity(g1, g2, dim=0).item()


def fit_other(text: str, corpus: CharCorpus) -> torch.Tensor:
    """Map foreign text onto the model's vocab (strip accents, drop rest)."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    ids = [corpus.stoi[c] for c in text if c in corpus.stoi]
    return torch.tensor(ids, dtype=torch.long)


def ascent_track(base, src, eval_sets, lr, steps, tag):
    """Ascend on `src`; track CE on each (name, tensor) in eval_sets."""
    m = copy.deepcopy(base)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(SEED)
    traj = []
    for s in range(steps + 1):
        if s % 25 == 0:
            point = {"step": s}
            for name, t in eval_sets:
                point[name] = estimate_loss(m, CORPUS, "val", n_batches=10, data=t)
            traj.append(point)
            print(f"  [{tag}] s{s:3d} " + " ".join(f"{k}={v:.3f}" for k, v in point.items() if k != "step"), flush=True)
        if s == steps:
            break
        x, y = batch_from(src, m.cfg.block_size, 32, gen)
        _, l = m(x, y)
        objective = -l
        opt.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m, traj


@torch.no_grad()
def ce_on_text(model, corpus, text: str) -> float:
    """Mean per-char CE over a held-out snippet (fluency/content probe)."""
    model.eval()
    ids = [corpus.stoi[c] for c in text if c in corpus.stoi]
    tot, n = 0.0, 0
    B = model.cfg.block_size
    for i in range(0, len(ids) - B - 1, B):
        chunk = torch.tensor(ids[i : i + B + 1]).unsqueeze(0).to(DEVICE)
        _, loss = model(chunk[:, :-1], chunk[:, 1:])
        tot += float(loss.item()) * B
        n += B
    model.train()
    return tot / max(n, 1)


# ---------------------------------------------------------------- main

CORPUS: CharCorpus  # global so helpers can use its seed for eval determinism


def main():
    global CORPUS
    assert E001_CKPT.exists(), "run e001 first"
    set_seed(SEED)
    rd = run_dir("e003")
    CORPUS = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=CORPUS.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    base = TinyGPT(cfg).to(DEVICE)
    base.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))

    train_a, train_b = CORPUS.slice("train", 0.0, 0.5), CORPUS.slice("train", 0.5, 1.0)
    val_a, val_b = CORPUS.slice("val", 0.0, 0.5), CORPUS.slice("val", 0.5, 1.0)
    base_a = estimate_loss(base, CORPUS, "val", n_batches=16, data=val_a)
    base_b = estimate_loss(base, CORPUS, "val", n_batches=16, data=val_b)

    # ---- ARM 1: gradient cosine (H2) ----
    fr_path = REPO / "data" / "input_french.txt"
    other_name = "french" if fr_path.exists() and fr_path.stat().st_size > 200_000 else "reversed"
    other_src = (fit_other(fr_path.read_text(encoding="utf-8"), CORPUS) if other_name == "french"
                 else fit_other((REPO / "data" / "input_reversed.txt").read_text(encoding="utf-8"), CORPUS))
    other_train = other_src[: int(0.9 * len(other_src))]
    other_val = other_src[int(0.9 * len(other_src)) :]

    print(f"other corpus: {other_name} ({len(other_src):,} chars in-vocab)")
    gen = torch.Generator().manual_seed(SEED)
    cos = {
        "within_A": sum(grad_cosine(base, train_a, train_a, cfg.block_size, gen) for _ in range(K_BATCH)) / K_BATCH,
        "within_B": sum(grad_cosine(base, train_b, train_b, cfg.block_size, gen) for _ in range(K_BATCH)) / K_BATCH,
        "A_vs_B": sum(grad_cosine(base, train_a, train_b, cfg.block_size, gen) for _ in range(K_BATCH)) / K_BATCH,
        "A_vs_other": sum(grad_cosine(base, train_a, other_train, cfg.block_size, gen) for _ in range(K_BATCH)) / K_BATCH,
    }
    print("gradient cosines:", {k: round(v, 4) for k, v in cos.items()}, flush=True)

    # ---- ARM 2: fine LR sweep (H1) ----
    sweeps = {}
    mild_model = None
    for lr in SWEEP_LRS:
        print(f"ascent sweep lr={lr}")
        m, traj = ascent_track(base, train_a, [("A", val_a), ("B", val_b)], lr, SWEEP_STEPS, f"lr{lr}")
        sweeps[str(lr)] = traj
        if lr == 1e-5:
            mild_model = copy.deepcopy(m)
    # best selective operating point across sweep
    best = None
    for lr, traj in sweeps.items():
        for p in traj:
            da, db = p["A"] - base_a, p["B"] - base_b
            score = da if db <= 0.1 else (da / db if db > 1e-6 else -99)
            if best is None or score > best["score"]:
                best = {"lr": lr, "step": p["step"], "dA": round(da, 3), "dB": round(db, 3), "score": score}
    print("best selective point:", best)

    # ---- ARM 3: dissimilar-content arm (H2, causal side) ----
    print(f"implanting {other_name} ({IMPLANT_STEPS} steps)...")
    implanted = copy.deepcopy(base)
    opt = torch.optim.AdamW(implanted.parameters(), lr=IMPLANT_LR, betas=(0.9, 0.95))
    gen2 = torch.Generator().manual_seed(SEED)
    impl_curve = []
    for s in range(IMPLANT_STEPS + 1):
        if s % 100 == 0:
            lo = estimate_loss(implanted, CORPUS, "val", n_batches=10, data=other_val)
            ls = estimate_loss(implanted, CORPUS, "val", n_batches=10, data=val_a)
            impl_curve.append({"step": s, "other": lo, "shake": ls})
            print(f"  [implant] s{s:3d} other={lo:.3f} shake={ls:.3f}", flush=True)
        if s == IMPLANT_STEPS:
            break
        x, y = batch_from(other_train, cfg.block_size, 32, gen2)  # DESCENT: implant = ordinary training
        _, l = implanted(x, y)
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(implanted.parameters(), 1.0)
        opt.step()
    imp_other0 = impl_curve[-1]["other"]
    imp_shake0 = estimate_loss(implanted, CORPUS, "val", n_batches=16, data=val_a)
    print(f"implanted: other={imp_other0:.3f} (from {impl_curve[0]['other']:.3f}), shake={imp_shake0:.3f} (base {base_a:.3f})")

    print(f"ascending on {other_name}...")
    _, dis_traj = ascent_track(implanted, other_train, [("other", other_val), ("shake", val_a)], DIS_LR, DIS_STEPS, "dissim")
    dis_best = None
    for p in dis_traj:
        do, ds = p["other"] - imp_other0, p["shake"] - imp_shake0
        score = do if ds <= 0.1 else (do / ds if ds > 1e-6 else -99)
        if dis_best is None or score > dis_best["score"]:
            dis_best = {"step": p["step"], "d_other": round(do, 3), "d_shake": round(ds, 3), "score": score}
    print("dissimilar best selective point:", dis_best)

    # ---- ARM 4: fluency vs content (H3) ----
    text = (REPO / "data" / "input.txt").read_text(encoding="utf-8")
    lines = text.split("\n")
    half = len(lines) // 2
    set_b = set(lines[half:])
    uniq_a = [l for l in lines[:half] if len(l) > 20 and l not in set_b]
    set_a = set(lines[:half])
    uniq_b = [l for l in lines[half:] if len(l) > 20 and l not in set_a]
    rng = random.Random(SEED)
    probe_a = "\n".join(rng.sample(uniq_a, min(40, len(uniq_a))))
    probe_b = "\n".join(rng.sample(uniq_b, min(40, len(uniq_b))))
    fluency = {
        "base": {"A_unique": ce_on_text(base, CORPUS, probe_a), "B_unique": ce_on_text(base, CORPUS, probe_b),
                 "generic": estimate_loss(base, CORPUS, "val", n_batches=16)},
        "after_mild_ascent": {"A_unique": ce_on_text(mild_model, CORPUS, probe_a),
                              "B_unique": ce_on_text(mild_model, CORPUS, probe_b),
                              "generic": estimate_loss(mild_model, CORPUS, "val", n_batches=16)},
    }
    print("fluency/content:", fluency)

    # ---- graphs ----
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    axes[0].bar(cos.keys(), cos.values(), color=["gray", "gray", "crimson", "steelblue"])
    axes[0].set_title(f"gradient cosine (H2)\nA_vs_B={cos['A_vs_B']:.3f} vs A_vs_other={cos['A_vs_other']:.3f}")
    axes[0].tick_params(axis="x", rotation=30); axes[0].set_ylabel("cosine")
    axes[0].axhline(0.85, ls="--", color="k", lw=0.8)
    for lr, traj in sweeps.items():
        axes[1].plot([p["B"] - base_b for p in traj], [p["A"] - base_a for p in traj], "o-", ms=3,
                     label=f"lr {lr}")
    axes[1].plot([0, 3], [0, 3], "k--", lw=0.8, label="anti-selective line")
    axes[1].axvline(0.1, color="g", ls=":", lw=1)
    axes[1].axhline(1.0, color="g", ls=":", lw=1)
    axes[1].set_xlabel("ΔB (collateral, nats)"); axes[1].set_ylabel("ΔA (target, nats)")
    axes[1].set_title("ascent trajectories (H1)\ngreen target: ΔA≥1 with ΔB≤0.1"); axes[1].legend(fontsize=7)
    do_ = [p["other"] - imp_other0 for p in dis_traj]
    ds_ = [p["shake"] - imp_shake0 for p in dis_traj]
    axes[2].plot(ds_, do_, "o-", color="purple", ms=3)
    axes[2].plot([0, 3], [0, 3], "k--", lw=0.8)
    axes[2].axvline(0.1, color="g", ls=":", lw=1); axes[2].axhline(1.0, color="g", ls=":", lw=1)
    axes[2].set_xlabel(f"Δ shakespeare (collateral)"); axes[2].set_ylabel(f"Δ {other_name} (target)")
    axes[2].set_title(f"dissimilar-content ascent (H2 causal)")
    labels = ["A-unique", "B-unique", "generic"]
    b0 = [fluency["base"]["A_unique"], fluency["base"]["B_unique"], fluency["base"]["generic"]]
    b1 = [fluency["after_mild_ascent"]["A_unique"], fluency["after_mild_ascent"]["B_unique"],
          fluency["after_mild_ascent"]["generic"]]
    x = range(3)
    axes[3].bar([i - 0.18 for i in x], b0, 0.36, label="base", color="gray")
    axes[3].bar([i + 0.18 for i in x], b1, 0.36, label="after mild ascent", color="crimson")
    axes[3].set_xticks(list(x)); axes[3].set_xticklabels(labels)
    axes[3].set_ylabel("CE (nats/char)"); axes[3].set_title("fluency vs content (H3)"); axes[3].legend()
    fig.suptitle("E003 selectivity frontier — T002 discriminator suite")
    fig.tight_layout()
    fig.savefig(rd / "selectivity_frontier.png", dpi=140)
    plt.close(fig)

    save_json(rd / "metrics.json", {
        "experiment": "e003_selectivity_frontier", "seed": SEED,
        "other_corpus": other_name, "config": cfg_dict(cfg),
        "grad_cosine": cos,
        "sweep": {"lrs": SWEEP_LRS, "trajectories": sweeps, "best_selective_point": best},
        "dissimilar": {"implant_curve": impl_curve, "implanted_other_ce": imp_other0,
                       "implanted_shake_ce": imp_shake0, "ascent_trajectory": dis_traj,
                       "best_selective_point": dis_best},
        "fluency_content": fluency,
        "registered_predictions_T002": {
            "grad_cosine_AB_ge_0.85": cos["A_vs_B"] >= 0.85,
            "no_selective_lr_point": best is None or best["dA"] < 1.0 or best["dB"] > 0.1,
            "dissimilar_selective": dis_best is not None and dis_best["d_other"] >= 1.0 and dis_best["d_shake"] <= 0.1,
        },
    })
    print("registered predictions:", {
        "grad_cosine_AB_ge_0.85": cos["A_vs_B"] >= 0.85,
        "no_selective_lr_point": best is None or best["dA"] < 1.0 or best["dB"] > 0.1,
        "dissimilar_selective": dis_best is not None and dis_best["d_other"] >= 1.0 and dis_best["d_shake"] <= 0.1,
    })
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
