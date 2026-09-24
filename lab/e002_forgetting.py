"""E002 — Forgetting pilot: can we cut out one memory without wrecking the rest?

Take the E001 net (trained on all of Tiny Shakespeare). Split the corpus by
position: A = first half (early plays), B = second half. Then unlearn A with
two arms:

  1. naive    : gradient ASCENT on A only (maximize CE on A)
  2. anchored : ascent on A + equal-weight descent on B (retain anchor)

Track CE on held-out A-text and B-text. Selectivity = how much B suffers while
A is erased. Also probe generation from an A prompt and a B prompt before and
after — the honesty check: CE is logit-level, generation is behavior.

Run: python lab/e002_forgetting.py   (requires E001 checkpoint)
"""
import copy

import matplotlib.pyplot as plt
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict, estimate_loss,
                    generate, run_dir, save_json, set_seed)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
SEED = 4242
STEPS = 400
EVAL_EVERY = 20
LR = 2e-5


def batch_from(src: torch.Tensor, block: int, gen: torch.Generator):
    ix = torch.randint(len(src) - block - 1, (32,), generator=gen)
    x = torch.stack([src[i : i + block] for i in ix]).to(DEVICE)
    y = torch.stack([src[i + 1 : i + 1 + block] for i in ix]).to(DEVICE)
    return x, y


def run_arm(model, corpus, src_a, src_b, val_a, val_b, anchored: bool, tag: str, rd):
    model = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(SEED)
    curve = []
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            la = estimate_loss(model, corpus, "val", n_batches=16, data=val_a)
            lb = estimate_loss(model, corpus, "val", n_batches=16, data=val_b)
            curve.append({"step": step, "loss_A": la, "loss_B": lb})
            print(f"  [{tag}] step {step:4d} | A {la:.4f} | B {lb:.4f}", flush=True)
        if step == STEPS:
            break
        xa, ya = batch_from(src_a, model.cfg.block_size, gen)
        _, loss_a = model(xa, ya)
        objective = -loss_a  # ascent on A
        if anchored:
            xb, yb = batch_from(src_b, model.cfg.block_size, gen)
            _, loss_b = model(xb, yb)
            objective = -loss_a + 1.0 * loss_b
        opt.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, curve


def main():
    assert E001_CKPT.exists(), "run e001 first (need runs/checkpoints/e001.pt)"
    set_seed(SEED)
    rd = run_dir("e002")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))

    train_a, train_b = corpus.slice("train", 0.0, 0.5), corpus.slice("train", 0.5, 1.0)
    val_a, val_b = corpus.slice("val", 0.0, 0.5), corpus.slice("val", 0.5, 1.0)
    base_a = estimate_loss(model, corpus, "val", n_batches=16, data=val_a)
    base_b = estimate_loss(model, corpus, "val", n_batches=16, data=val_b)
    print(f"baseline | A {base_a:.4f} | B {base_b:.4f}")

    prompt_a, prompt_b = "ROMEO:", "PROSPERO:"
    probes_pre = {p: generate(model, corpus, p, 240) for p in (prompt_a, prompt_b)}

    print("arm 1: naive gradient ascent on A")
    m_naive, cur_naive = run_arm(model, corpus, train_a, train_b, val_a, val_b, False, "naive", rd)
    print("arm 2: anchored (ascent A + retain B)")
    m_anchor, cur_anchor = run_arm(model, corpus, train_a, train_b, val_a, val_b, True, "anchored", rd)

    probes_post = {
        "naive/ROMEO:": generate(m_naive, corpus, prompt_a, 240),
        "naive/PROSPERO:": generate(m_naive, corpus, prompt_b, 240),
        "anchored/ROMEO:": generate(m_anchor, corpus, prompt_a, 240),
        "anchored/PROSPERO:": generate(m_anchor, corpus, prompt_b, 240),
    }
    (rd / "probes.txt").write_text(
        "=== BEFORE ===\n\n"
        + "\n\n".join(f"[{p}]\n{s}" for p, s in probes_pre.items())
        + "\n\n=== AFTER ===\n\n"
        + "\n\n".join(f"[{p}]\n{s}" for p, s in probes_post.items()),
        encoding="utf-8",
    )

    # selectivity: erase A by +1.0 nat, how much does B move?
    def damage(curve):
        end_a, end_b = curve[-1]["loss_A"], curve[-1]["loss_B"]
        da, db = end_a - base_a, end_b - base_b
        # first step where A rose >= 1.0 nat, if any
        step_1nat = next((c["step"] for c in curve if c["loss_A"] - base_a >= 1.0), None)
        b_at = next((c["loss_B"] - base_b for c in curve if c["step"] == step_1nat), None) if step_1nat else None
        return da, db, step_1nat, b_at

    da_n, db_n, s1_n, b1_n = damage(cur_naive)
    da_a, db_a, s1_a, b1_a = damage(cur_anchor)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, curve, name in ((axes[0], cur_naive, "naive ascent on A"), (axes[1], cur_anchor, "anchored (retain B)")):
        st = [c["step"] for c in curve]
        ax.plot(st, [c["loss_A"] for c in curve], "o-", color="crimson", label="A (to forget)")
        ax.plot(st, [c["loss_B"] for c in curve], "s-", color="steelblue", label="B (to keep)")
        ax.axhline(base_a, ls="--", color="crimson", alpha=0.4)
        ax.axhline(base_b, ls="--", color="steelblue", alpha=0.4)
        ax.set_title(name); ax.set_xlabel("unlearn step"); ax.legend()
    axes[0].set_ylabel("CE (nats/char)")
    fig.suptitle(f"E002 forgetting pilot — erase A (first half), keep B (second half)")
    fig.tight_layout()
    fig.savefig(rd / "forgetting_curve.png", dpi=140)
    plt.close(fig)

    metrics = {
        "experiment": "e002_forgetting",
        "seed": SEED, "steps": STEPS, "lr": LR,
        "baseline": {"A": base_a, "B": base_b},
        "naive": {"curve": cur_naive, "final_delta_A": da_n, "final_delta_B": db_n,
                  "B_damage_at_A_plus_1nat": b1_n},
        "anchored": {"curve": cur_anchor, "final_delta_A": da_a, "final_delta_B": db_a,
                     "B_damage_at_A_plus_1nat": b1_a},
        "config": cfg_dict(model.cfg),
    }
    save_json(rd / "metrics.json", metrics)
    print(f"\nnaive    : ΔA {da_n:+.3f}  ΔB {db_n:+.3f}  (B damage when A +1 nat: {b1_n})")
    print(f"anchored : ΔA {da_a:+.3f}  ΔB {db_a:+.3f}  (B damage when A +1 nat: {b1_a})")
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
