"""
genome_157c_3seed_canonical_verdict.py

CANONICAL 3-SEED VERDICT (conditional on g157b DIRECTIONAL_SUPPORT): embedding-layer prefix instead of same-layer.

Pre-reg LOCKED: research/prereg/genome_157b_eta_delta_probe_embedding_prefix_2026-04-26.md
Theory: research/derivations/prefix_information_transport.md
Predecessor: code/genome_157_eta_delta_probe.py (same-layer prefix variant)

Conditional launch: only if g157 PILOT returns PILOT_KILL or WEAK_SUPPORT.
The g157 PILOT uses h_<t at the same layer ell as the prefix probe's K/V.
That captures only info the residual stream has already merged. The proper
test of remaining transport gap uses fresh embedding-layer prefix.

Same compute envelope as g157 PILOT. Same checkpoints, same dedup audit.

If KILL_157b: theory's eta>delta criterion is wrong; g156 inversion still
stands as cross-axis evidence but the proposed mechanism is dead.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

ROOT = _THIS_DIR.parent

SEQ_LEN = 256
N_PROBE_TRAIN = 2048
N_PROBE_VAL = 256
N_PROBE_TEST = 256

MIDBAND_INDICES = {14: [5, 7, 9], 7: [2, 3, 4]}

# CANONICAL 3-seed verdict (g157c, conditional on g157b DIRECTIONAL_SUPPORT)
SEEDS = [42, 7, 13]
PROBE_LR = 1e-3
PROBE_STEPS = 500
PROBE_BATCH = 32
PROBE_BENCH_STEPS = 50
HARD_ABORT_HOURS = 3.5
EPS = 1e-6

CKPT_DIR = ROOT / "results" / "genome_156_checkpoints"


class ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def load_checkpoint(ckpt_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = payload["config"]
    no_mlp = cfg_dict.get("no_mlp", False)
    cfg = LlamaConfig(
        vocab_size=cfg_dict["vocab_size"], hidden_size=cfg_dict["hidden"],
        num_hidden_layers=cfg_dict["layers"], num_attention_heads=cfg_dict["heads"],
        num_key_value_heads=cfg_dict["heads"], intermediate_size=cfg_dict["ffn"],
        max_position_embeddings=SEQ_LEN + 64, rms_norm_eps=1e-6,
        tie_word_embeddings=True, attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)
    if no_mlp:
        for layer in model.model.layers:
            layer.mlp = ZeroMLP()
    model.load_state_dict(payload["state_dict"])
    model = model.to("cuda").eval()
    meta = {"cond": payload["cond"], "arm": payload["arm"], "seed": payload["seed"],
            "vocab_size": cfg_dict["vocab_size"], "n_layers": cfg_dict["layers"],
            "hidden": cfg_dict["hidden"]}
    return model, meta


def extract_layer_and_embed(model, ids, mask, layer_idx):
    """Capture both: (1) hidden state at layer_idx (post-block), (2) embedding (layer 0 input).
    Both moved to CPU per-batch."""
    captured_h = []
    captured_e = []

    def hook_h(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured_h.append(h.detach().cpu())

    handle_h = model.model.layers[layer_idx].register_forward_hook(hook_h)
    try:
        with torch.no_grad():
            for i in range(0, ids.size(0), PROBE_BATCH):
                ids_b = ids[i:i+PROBE_BATCH].to("cuda")
                # Get embedding directly via embed_tokens
                e_b = model.model.embed_tokens(ids_b).detach()
                captured_e.append(e_b.cpu())
                _ = model(input_ids=ids_b, attention_mask=mask[i:i+PROBE_BATCH].to("cuda"),
                          use_cache=False)
    finally:
        handle_h.remove()
    return torch.cat(captured_h, dim=0), torch.cat(captured_e, dim=0)


class LinearProbe(nn.Module):
    def __init__(self, hidden, vocab):
        super().__init__()
        self.W = nn.Linear(hidden, vocab, bias=True)
    def forward(self, h):
        return self.W(h)


class LocalMLPProbe(nn.Module):
    def __init__(self, hidden, vocab, mlp_hidden=None):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = max(1, (hidden * vocab) // (hidden + vocab))
        self.fc1 = nn.Linear(hidden, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, vocab)
    def forward(self, h):
        return self.fc2(F.gelu(self.fc1(h)))


class PrefixEmbedAttnProbe(nn.Module):
    """Causal cross-attention with K/V from EMBEDDINGS (not same-layer h).
    kv_dim chosen for ~1% param match with LinearProbe."""
    def __init__(self, hidden, vocab, embed_dim=None, kv_dim=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = hidden  # Llama: embed_tokens hidden = model hidden
        if kv_dim is None:
            kv_dim = max(1, (hidden * vocab) // (hidden + 2 * embed_dim + vocab))
        self.q_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        self.out = nn.Linear(kv_dim, vocab, bias=True)
        self.scale = kv_dim ** -0.5

    def forward(self, h, e, mask):
        # h: (B, T, hidden) at layer ell
        # e: (B, T, embed_dim) at layer 0
        # Causal cross-attention: q(h_t) over k(e_<t), v(e_<t)
        q = self.q_proj(h); k = self.k_proj(e); v = self.v_proj(e)
        B, T, _ = h.shape
        scores = torch.einsum("bth,bsh->bts", q, k) * self.scale
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        if mask is not None:
            pad = (mask == 0).unsqueeze(1).expand(B, T, T)
            scores = scores.masked_fill(pad, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        ctx = torch.einsum("bts,bsh->bth", attn, v)
        return self.out(ctx)


def train_probe(probe, h_train, e_train, ids_train, mask_train,
                 h_val, e_val, ids_val, mask_val, use_prefix=False):
    # Probe weights in FP32 for numerical stability; activations stay BF16.
    # Lesson from g157 v2 PILOT: BF16 probes blew up on shuffled distribution
    # (CE > 200) due to exploding lin-probe gradients without clipping.
    probe = probe.to("cuda").to(torch.float32)
    opt = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR, weight_decay=0.01)
    n_train = h_train.size(0)
    rng = np.random.default_rng(0)
    best_val = float("inf")
    best_state = None
    for step in range(PROBE_STEPS):
        idx = rng.integers(0, n_train, size=PROBE_BATCH)
        h_b = h_train[idx].to("cuda").to(torch.float32)
        ids_b = ids_train[idx].to("cuda")
        mask_b = mask_train[idx].to("cuda")
        if use_prefix:
            e_b = e_train[idx].to("cuda").to(torch.float32)
            logits = probe(h_b, e_b, mask_b)
        else:
            logits = probe(h_b)
        sl = logits[:, :-1].contiguous()
        lbl = ids_b[:, 1:].clone()
        sm = mask_b[:, 1:]
        lbl[sm == 0] = -100
        loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            continue  # Skip non-finite losses instead of poisoning the params
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            v = eval_probe(probe, h_val, e_val, ids_val, mask_val, use_prefix)
            if v < best_val:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in probe.state_dict().items()}
    # Per heartbeat code review Sev-7: refuse to return random/unvalidated probe
    if best_state is None:
        raise RuntimeError(
            "probe training never produced a finite validation checkpoint; "
            "all loss values were non-finite or no eval interval reached"
        )
    probe.load_state_dict(best_state)
    return probe


def eval_probe(probe, h, e, ids, mask, use_prefix=False):
    probe.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, h.size(0), PROBE_BATCH):
            h_b = h[i:i+PROBE_BATCH].to("cuda").to(torch.float32)
            ids_b = ids[i:i+PROBE_BATCH].to("cuda")
            mask_b = mask[i:i+PROBE_BATCH].to("cuda")
            if use_prefix:
                e_b = e[i:i+PROBE_BATCH].to("cuda").to(torch.float32)
                logits = probe(h_b, e_b, mask_b)
            else:
                logits = probe(h_b)
            sl = logits[:, :-1].contiguous()
            lbl = ids_b[:, 1:].clone()
            sm = mask_b[:, 1:]
            valid = (sm != 0)
            lbl[~valid] = -100
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1),
                                    ignore_index=-100, reduction="sum")
            total_loss += loss.item()
            total_tokens += valid.sum().item()
    probe.train()
    return total_loss / max(total_tokens, 1)


def shuffle_token_rows(ids, mask, shuffle_seed=43):
    rng = np.random.default_rng(shuffle_seed)
    out = ids.clone()
    for r in range(ids.shape[0]):
        valid_pos = (mask[r] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        if len(valid_pos) <= 1:
            continue
        perm = rng.permutation(len(valid_pos))
        out[r, valid_pos] = ids[r, valid_pos[perm]]
    return out


def thirteen_token_hashes(ids, mask):
    hashes = set()
    for r in range(ids.shape[0]):
        valid = mask[r].sum().item()
        if valid < 13:
            continue
        row = ids[r, :valid].tolist()
        for i in range(len(row) - 12):
            hashes.add(tuple(row[i:i+13]))
    return hashes


def load_validation_data(tok):
    """Same protocol as g157 PILOT — c4 val + Wikitext val + dedup audit."""
    from datasets import load_dataset
    print("Loading c4 validation...")
    c4_texts = []
    try:
        ds_c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        for ex in ds_c4:
            t = ex["text"]
            if len(t) > 200:
                c4_texts.append(t)
            if len(c4_texts) >= N_PROBE_TRAIN + N_PROBE_VAL + N_PROBE_TEST:
                break
    except Exception as e:
        print(f"  c4 streaming failed: {e}; trying file fallback")
        ds_c4 = load_dataset("allenai/c4", "en", split="validation",
                              data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"})
        for ex in ds_c4:
            t = ex["text"]
            if len(t) > 200:
                c4_texts.append(t)
            if len(c4_texts) >= N_PROBE_TRAIN + N_PROBE_VAL + N_PROBE_TEST:
                break
    print(f"  loaded {len(c4_texts)}")

    enc = tok(c4_texts[:N_PROBE_TRAIN + N_PROBE_VAL + N_PROBE_TEST],
               padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc["input_ids"][:N_PROBE_TRAIN]
    train_mask = enc["attention_mask"][:N_PROBE_TRAIN]
    val_ids = enc["input_ids"][N_PROBE_TRAIN:N_PROBE_TRAIN + N_PROBE_VAL]
    val_mask = enc["attention_mask"][N_PROBE_TRAIN:N_PROBE_TRAIN + N_PROBE_VAL]
    test_ids = enc["input_ids"][N_PROBE_TRAIN + N_PROBE_VAL:]
    test_mask = enc["attention_mask"][N_PROBE_TRAIN + N_PROBE_VAL:]

    # Dedup audit
    print("Dedup audit vs g156 train slice...")
    from stimulus_banks import c4_clean_v1
    g156_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=2048):
        g156_texts.append(rec["text"])
        if len(g156_texts) >= 2048:
            break
    enc_g156 = tok(g156_texts, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    g156_h = thirteen_token_hashes(enc_g156["input_ids"], enc_g156["attention_mask"])
    train_h = thirteen_token_hashes(train_ids, train_mask)
    overlap = train_h & g156_h
    pct = 100.0 * len(overlap) / max(len(train_h), 1)
    print(f"  13-gram overlap: {pct:.2f}%")
    if pct > 5.0:
        raise RuntimeError(f"dedup FAIL: {pct:.2f}% > 5%")

    return {
        "natural": {"train": (train_ids, train_mask), "val": (val_ids, val_mask), "test": (test_ids, test_mask)},
        "token_shuffled": {
            "train": (shuffle_token_rows(train_ids, train_mask, 43), train_mask),
            "val": (shuffle_token_rows(val_ids, val_mask, 44), val_mask),
            "test": (shuffle_token_rows(test_ids, test_mask, 45), test_mask),
        },
    }


def microbenchmark(hidden, vocab):
    print("\nMicrobenchmark...")
    fake_ids = torch.randint(0, vocab, (PROBE_BATCH, SEQ_LEN), device="cuda")
    fake_mask = torch.ones_like(fake_ids)
    fake_h = torch.randn(PROBE_BATCH * 4, SEQ_LEN, hidden, dtype=torch.bfloat16, device="cuda")
    fake_e = torch.randn(PROBE_BATCH * 4, SEQ_LEN, hidden, dtype=torch.bfloat16, device="cuda")
    rng = np.random.default_rng(0)
    times = {}
    for name, ProbeClass, use_prefix in [
        ("lin", LinearProbe, False),
        ("local", LocalMLPProbe, False),
        ("prefix_embed", PrefixEmbedAttnProbe, True),
    ]:
        probe = ProbeClass(hidden, vocab).to("cuda").to(torch.bfloat16)
        opt = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(PROBE_BENCH_STEPS):
            idx = rng.integers(0, fake_h.size(0), size=PROBE_BATCH)
            h_b = fake_h[idx]
            ids_b = fake_ids
            mask_b = fake_mask
            if use_prefix:
                e_b = fake_e[idx]
                logits = probe(h_b, e_b, mask_b)
            else:
                logits = probe(h_b)
            sl = logits[:, :-1].float()
            lbl = ids_b[:, 1:].clone()
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        per_step = (time.time() - t0) / PROBE_BENCH_STEPS
        times[name] = per_step
        n_params = sum(p.numel() for p in probe.parameters())
        print(f"  {name}: {1000*per_step:.1f} ms/step  params={n_params/1e6:.2f}M")
        del probe, opt; torch.cuda.empty_cache()
    n_layers_used = 3
    n_ckpts = 4
    total_seconds = sum(times.values()) * PROBE_STEPS * n_layers_used * n_ckpts + n_layers_used * n_ckpts * 30
    total_hours = total_seconds / 3600
    print(f"  PROJECTED total: {total_hours:.2f} hr")
    if total_hours > HARD_ABORT_HOURS:
        raise RuntimeError(f"projected {total_hours:.2f} hr > envelope {HARD_ABORT_HOURS}; abort")
    return total_hours


def main():
    t0 = time.time()
    print("genome_157c CANONICAL 3-SEED VERDICT: eta/delta probe (embedding-prefix)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    data = load_validation_data(tok)
    all_ckpts = []
    for s in SEEDS:
        all_ckpts.extend(sorted(CKPT_DIR.glob(f"*seed{s}.pt")))
    if len(all_ckpts) != 4 * len(SEEDS):
        raise RuntimeError(f"3-seed canonical requires 12 ckpts, found {len(all_ckpts)}")
    print(f"Canonical scope: {len(all_ckpts)} ckpts ({len(SEEDS)} seeds x 4 conditions/arms)")

    model_ref, meta_ref = load_checkpoint(all_ckpts[0])
    projected_hr = microbenchmark(meta_ref["hidden"], meta_ref["vocab_size"])
    del model_ref; torch.cuda.empty_cache()

    results = {}
    for ckpt_path in all_ckpts:
        ckpt_name = ckpt_path.stem
        cond = ckpt_name.split("__")[0]
        print(f"\n=== {ckpt_name} (cond={cond}) ===")
        model, meta = load_checkpoint(ckpt_path)
        n_layers = meta["n_layers"]
        hidden = meta["hidden"]; vocab = meta["vocab_size"]
        layer_indices = MIDBAND_INDICES[n_layers]
        print(f"  n_layers={n_layers}, midband: {layer_indices}")

        d_train = data[cond]["train"]; d_val = data[cond]["val"]; d_test = data[cond]["test"]
        per_layer = {}
        for li in layer_indices:
            t_l = time.time()
            print(f"  -- layer {li} (depth ~{li/(n_layers-1):.2f}) --")
            h_train, e_train = extract_layer_and_embed(model, d_train[0], d_train[1], li)
            h_val, e_val = extract_layer_and_embed(model, d_val[0], d_val[1], li)
            h_test, e_test = extract_layer_and_embed(model, d_test[0], d_test[1], li)

            ce = {}; params_M = {}
            for name, ProbeClass, use_prefix in [
                ("lin", LinearProbe, False),
                ("local", LocalMLPProbe, False),
                ("prefix_embed", PrefixEmbedAttnProbe, True),
            ]:
                probe = ProbeClass(hidden, vocab)
                params_M[name] = sum(p.numel() for p in probe.parameters()) / 1e6
                probe = train_probe(probe, h_train, e_train, d_train[0], d_train[1],
                                     h_val, e_val, d_val[0], d_val[1], use_prefix=use_prefix)
                ce[name] = eval_probe(probe, h_test, e_test, d_test[0], d_test[1], use_prefix=use_prefix)
                print(f"    {name}: CE={ce[name]:.4f}  params={params_M[name]:.2f}M")
                del probe; torch.cuda.empty_cache()

            delta = ce["lin"] - ce["local"]
            eta = ce["local"] - ce["prefix_embed"]
            G = eta - delta
            per_layer[li] = {
                "CE_lin": ce["lin"], "CE_local": ce["local"], "CE_prefix_embed": ce["prefix_embed"],
                "delta_hat_mlp": delta, "eta_hat": eta, "G_l": G,
                "depth_frac": li / (n_layers - 1), "params_M": params_M,
                "wallclock_s": time.time() - t_l,
            }
            print(f"    -> delta={delta:.4f} eta={eta:.4f} G={G:+.4f}")
            del h_train, h_val, h_test, e_train, e_val, e_test; torch.cuda.empty_cache()

        results[ckpt_name] = {"meta": meta, "per_layer": per_layer}
        del model; torch.cuda.empty_cache()

    if len(results) != 4:
        raise RuntimeError(f"PILOT incomplete: {len(results)}/4")

    # Summary
    print(f"\n=== ANALYSIS (157b, embedding-layer prefix) ===")
    summary = {}
    for cond in ["natural", "token_shuffled"]:
        for arm in ["baseline_200M_4k", "minimal_7L_200M_8k"]:
            G_per_seed = []
            for s in SEEDS:
                key = f"{cond}__{arm}__seed{s}"
                if key not in results:
                    continue
                G_vals = [v["G_l"] for v in results[key]["per_layer"].values()]
                G_per_seed.append(float(np.mean(G_vals)))
            if not G_per_seed:
                continue
            summary[f"{cond}__{arm}"] = {"G_mid_mean": float(np.mean(G_per_seed)),
                                          "G_mid_std": float(np.std(G_per_seed)),
                                          "G_per_seed": G_per_seed,
                                          "n_seeds": len(G_per_seed)}
            print(f"  {cond}__{arm}: G_mid_mean={np.mean(G_per_seed):+.4f} (+/-{np.std(G_per_seed):.4f}) n={len(G_per_seed)}")

    nat_min = summary.get("natural__minimal_7L_200M_8k", {})
    shuf_min = summary.get("token_shuffled__minimal_7L_200M_8k", {})
    nat_G = nat_min.get("G_mid_mean", 0.0)
    shuf_G = shuf_min.get("G_mid_mean", 0.0)
    contrast = nat_G - shuf_G

    nat_seeds_positive = sum(1 for g in nat_min.get("G_per_seed", []) if g >= 0.02)
    n_seeds_min = nat_min.get("n_seeds", 0)
    if nat_seeds_positive >= 2 and shuf_G <= 0.0 and contrast >= 0.03:
        verdict = (f"PASS_C: {nat_seeds_positive}/{n_seeds_min} seeds with nat_G>=0.02; pooled nat_G={nat_G:+.4f}; "
                   f"shuf_G={shuf_G:+.4f}; contrast={contrast:+.4f}. CANONICAL VERDICT: "
                   f"transport budget criterion eta>delta directly OBSERVED at 3-seed scale. "
                   f"Promote P12 -> C16.")
    elif nat_seeds_positive >= 1 and contrast >= 0.015:
        verdict = (f"PARTIAL_C: {nat_seeds_positive}/{n_seeds_min} seeds positive; pooled nat_G={nat_G:+.4f}; "
                   f"contrast={contrast:+.4f}. PILOT signal not fully replicated; consider 5-seed expansion.")
    else:
        verdict = (f"KILL_C: {nat_seeds_positive}/{n_seeds_min} seeds positive; pooled nat_G={nat_G:+.4f}; "
                   f"PILOT not replicable at 3 seeds. Falls back to Path C of post-g157b decision tree.")
    print(f"\n  verdict: {verdict}")

    out = {
        "genome": "157c", "name": "eta_delta_probe_3seed_canonical_verdict",
        "config": {"midband_indices": {str(k): v for k, v in MIDBAND_INDICES.items()},
                    "n_probe_train": N_PROBE_TRAIN, "probe_lr": PROBE_LR,
                    "probe_steps": PROBE_STEPS, "probe_batch": PROBE_BATCH,
                    "canonical_seeds": SEEDS, "projected_hours": projected_hr},
        "results": {k: {"meta": v["meta"], "per_layer": {str(li): pv for li, pv in v["per_layer"].items()}}
                    for k, v in results.items()},
        "summary": summary,
        "nat_G_mean": nat_G, "shuf_G_mean": shuf_G, "contrast": contrast,
        "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_157b_eta_delta_probe.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
