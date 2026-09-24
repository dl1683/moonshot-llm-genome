"""E013a — Attention census over 200 prompts: is L5's re-globalization real?

Review 1 weakened T005: the rare-token story came from ONE head of 36 on ONE
prompt (possible dialogue-vocative artifact). This census measures, across
200 held-out prompts and all 36 heads:

  - local mass (distance 1-3 from the last token) and far mass (d >= 17)
  - attended-token surprisal (bits, under the corpus unigram distribution)
  - concentration (max single-position mass; is it on a distant token?)

Verdict rules (pre-registered):
  - "L5 abandons local" is REAL if L5 local-mass < L4 local-mass - 0.05 in
    >= 70% of prompts.
  - "distant-rare reader heads" exist if any heads (report which layers) have
    far-mass >= 0.5 AND distant-concentration (max mass >= 0.3 at d >= 17)
    in >= 60% of prompts. Count them per layer.
  - e013 (causal mask) proceeds only if >= 2 such heads exist; their top
    attended positions across prompts become e013's targets.

Run: python lab/e013a_attention_census.py   (requires E001 checkpoint)
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json, set_seed)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
N_PROMPTS = 200
WIN = 96


@torch.no_grad()
def main():
    set_seed(13)
    rd = run_dir("e013a")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    # unigram surprisal (bits) from train split
    counts = torch.bincount(corpus.train, minlength=cfg.vocab).float()
    uni = counts / counts.sum()
    surprisal = -torch.log2(uni + 1e-12)

    gen = torch.Generator().manual_seed(13)
    starts = torch.randint(len(corpus.val) - WIN - 1, (N_PROMPTS,), generator=gen)

    # storage: per layer, per head: arrays over prompts
    far = torch.zeros(cfg.n_layer, cfg.n_head, N_PROMPTS)
    local = torch.zeros(cfg.n_layer, cfg.n_head, N_PROMPTS)
    surp_att = torch.zeros(cfg.n_layer, cfg.n_head, N_PROMPTS)
    conc_far = torch.zeros(cfg.n_layer, cfg.n_head, N_PROMPTS)  # max mass >= 0.3 at d>=17
    max_mass = torch.zeros(cfg.n_layer, cfg.n_head, N_PROMPTS)

    ln_inputs = {}
    handles = []
    for i, block in enumerate(model.h):
        def mk(i):
            def pre(m, args):
                ln_inputs[i] = args[0].detach()
                return None
            return pre
        handles.append(block.attn.register_forward_pre_hook(mk(i)))

    T = WIN
    d = torch.arange(T)
    dist = (T - 1) - d  # distance of each key from the last query
    local_mask = (dist >= 1) & (dist <= 3)
    far_mask = dist >= 17

    for p in range(N_PROMPTS):
        s = int(starts[p])
        x = corpus.val[s : s + WIN].unsqueeze(0).to(DEVICE)
        chars = corpus.val[s : s + WIN]
        model(x)
        for i in range(cfg.n_layer):
            xv = ln_inputs[i]
            qkv = model.h[i].attn.c_attn(xv)
            q, k, _ = qkv.split(cfg.n_embd, dim=2)
            hd = cfg.n_embd // cfg.n_head
            q = q.view(1, T, cfg.n_head, hd).transpose(1, 2)      # (1,H,T,hd)
            k = k.view(1, T, cfg.n_head, hd).transpose(1, 2)
            att = F.softmax(q @ k.transpose(-2, -1) / (hd ** 0.5), dim=-1)[0, :, -1, :]  # (H, T_key) last query
            for h in range(cfg.n_head):
                a = att[h]
                far[i, h, p] = float(a[far_mask].sum())
                local[i, h, p] = float(a[local_mask].sum())
                surp_att[i, h, p] = float((a * surprisal[chars].to(DEVICE)).sum())
                mm, am = float(a.max()), int(a.argmax())
                max_mass[i, h, p] = mm
                conc_far[i, h, p] = 1.0 if (mm >= 0.3 and dist[am] >= 17) else 0.0
    for h_ in handles:
        h_.remove()

    layer_local = local.mean(dim=(1, 2))
    layer_far = far.mean(dim=(1, 2))
    layer_surp = surp_att.mean(dim=(1, 2))
    print("per-layer mean local(d1-3):", [round(v, 3) for v in layer_local.tolist()])
    print("per-layer mean far(d17+):  ", [round(v, 3) for v in layer_far.tolist()])
    print("per-layer mean attended surprisal (bits):", [round(v, 2) for v in layer_surp.tolist()])

    # prompt-level L4 vs L5 deltas (averaged over heads within layer)
    l5_minus_l4_local = (local[5] - local[4]).mean(dim=0)  # (N_PROMPTS,)
    frac_abandons = float((l5_minus_l4_local <= -0.05).float().mean())
    print(f"L5 local-mass < L4 - 0.05 in {frac_abandons*100:.1f}% of prompts "
          f"[criterion >= 70%]")

    # distant-rare reader heads
    head_far = far.mean(dim=2)
    head_conc = conc_far.mean(dim=2)
    readers = []
    for i in range(cfg.n_layer):
        for h in range(cfg.n_head):
            if head_far[i, h] >= 0.5 and head_conc[i, h] >= 0.6:
                readers.append((i, h, round(float(head_far[i, h]), 2), round(float(head_conc[i, h]), 2)))
    print("distant-rare reader heads (far>=0.5, conc>=0.6):", readers if readers else "NONE")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    xs = list(range(cfg.n_layer))
    axes[0].errorbar(xs, layer_local.tolist(), yerr=(local.mean(dim=2).std(dim=1) / 6).tolist(), label="local d1-3", color="crimson")
    axes[0].errorbar(xs, layer_far.tolist(), yerr=(far.mean(dim=2).std(dim=1) / 6).tolist(), label="far d17+", color="steelblue")
    axes[0].set_xticks(xs); axes[0].set_xticklabels([f"L{i}" for i in xs])
    axes[0].set_title("attention mass by distance (mean over 200 prompts × heads)"); axes[0].legend()
    axes[1].hist(l5_minus_l4_local.tolist(), bins=30, color="gray")
    axes[1].axvline(-0.05, color="crimson", ls="--")
    axes[1].set_title(f"L5−L4 local-mass delta ({frac_abandons*100:.0f}% ≤ −0.05)")
    axes[1].set_xlabel("Δ local mass (L5 − L4)")
    for i in range(cfg.n_layer):
        axes[2].scatter(head_far[i].tolist(), head_conc[i].tolist(), s=18, label=f"L{i}")
    axes[2].axvline(0.5, color="k", ls=":"); axes[2].axhline(0.6, color="k", ls=":")
    axes[2].set_xlabel("mean far-mass (d≥17)"); axes[2].set_ylabel("frac prompts conc.≥0.3 at d≥17")
    axes[2].set_title(f"distant-rare readers: {len(readers)} heads"); axes[2].legend(fontsize=7, ncol=2)
    fig.suptitle("E013a — attention census over 200 prompts (T005 adjudication)")
    fig.tight_layout(); fig.savefig(rd / "attention_census.png", dpi=140); plt.close(fig)

    save_json(rd / "metrics.json", {
        "experiment": "e013a_attention_census", "n_prompts": N_PROMPTS,
        "layer_local_mass": layer_local.tolist(), "layer_far_mass": layer_far.tolist(),
        "layer_attended_surprisal_bits": layer_surp.tolist(),
        "frac_prompts_L5_abandons_local": frac_abandons,
        "distant_rare_reader_heads": readers,
        "per_head_far": head_far.tolist(), "per_head_concentration": head_conc.tolist(),
        "verdicts": {
            "L5_abandons_local_real": frac_abandons >= 0.70,
            "e013_proceeds": len(readers) >= 2,
        },
    })
    print("verdicts:", {"L5_abandons_local_real": frac_abandons >= 0.70,
                        "e013_proceeds": len(readers) >= 2})
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
