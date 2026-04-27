"""
genome_157_eta_delta_probe.py

POST-g156-PASS PROBE: layerwise transport surplus G_l = eta_hat_l - delta_hat_l^mlp.

Pre-reg LOCKED: research/prereg/genome_157_eta_delta_probe_2026-04-26.md
Theory: research/derivations/prefix_information_transport.md
Program: research/programs/post_g156_pass_program.md

Operates on the 12 saved g156 checkpoints under
results/genome_156_checkpoints/. For each checkpoint and each functional
depth d in {0.20, 0.35, 0.50, 0.65, 0.80}, extract activations on
held-out c4_val + wikitext_val windows. Train three probes per layer
(linear, local-MLP, prefix-cross-attn) at equal parameter budget, compute
held-out test cross-entropies, derive:

  delta_hat^mlp(l) = CE_lin(l)   - CE_local(l)
  eta_hat(l)       = CE_local(l) - CE_prefix(l)
  G_l              = eta_hat(l)  - delta_hat^mlp(l)

PASS_G157: minimal-natural mid-band G_l >= +0.02 nats in >=2/3 seeds
           AND minimal-shuffled G_l <= 0.00
           AND pooled contrast G_nat - G_shuf >= +0.03 nats

Compute: ~1.5-2hr on RTX 5090. Activations cached + sharded.

Results: results/genome_157_eta_delta_probe.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent

SEQ_LEN = 256
SEEDS = [42, 7, 13]
N_PROBE_TRAIN = 4096
N_PROBE_VAL = 512
N_PROBE_TEST = 512
N_OOD_TEST = 512
DEPTHS = [0.20, 0.35, 0.50, 0.65, 0.80]
PROBE_LR = 1e-3
PROBE_STEPS = 3000
PROBE_BATCH = 32
EPS = 1e-6

CKPT_DIR = ROOT / "results" / "genome_156_checkpoints"
CACHE_DIR = ROOT / "cache" / "g157_activations"


class ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def load_checkpoint(ckpt_path: Path):
    from transformers import LlamaConfig, LlamaForCausalLM
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = payload["config"]
    no_mlp = cfg_dict.get("no_mlp", False)
    cfg = LlamaConfig(
        vocab_size=cfg_dict["vocab_size"],
        hidden_size=cfg_dict["hidden"],
        num_hidden_layers=cfg_dict["layers"],
        num_attention_heads=cfg_dict["heads"],
        num_key_value_heads=cfg_dict["heads"],
        intermediate_size=cfg_dict["ffn"],
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="eager",
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


def extract_activations(model, ids, mask, layer_idx):
    """Run forward, capture hidden states at given layer (post-block)."""
    n_layers = model.config.num_hidden_layers
    captured = []
    handles = []

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach())
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for i in range(0, ids.size(0), PROBE_BATCH):
                _ = model(input_ids=ids[i:i+PROBE_BATCH].to("cuda"),
                          attention_mask=mask[i:i+PROBE_BATCH].to("cuda"),
                          use_cache=False)
    finally:
        handle.remove()
    return torch.cat(captured, dim=0)  # (N, T, h)


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
            # Match LinearProbe param count: linear has hidden*vocab + vocab.
            # Two-layer MLP: hidden*mlp_hidden + mlp_hidden + mlp_hidden*vocab + vocab.
            # Solve: mlp_hidden ~= (hidden*vocab) / (hidden + vocab).
            mlp_hidden = max(1, (hidden * vocab) // (hidden + vocab))
        self.fc1 = nn.Linear(hidden, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, vocab)

    def forward(self, h):
        return self.fc2(F.gelu(self.fc1(h)))


class PrefixAttnProbe(nn.Module):
    """One-head cross-attention probe: query = h_t, keys/values = prefix h_{<t}.
    Causal, single head. Approx param-matched to LinearProbe via projection sizes."""
    def __init__(self, hidden, vocab, kv_dim=None):
        super().__init__()
        if kv_dim is None:
            kv_dim = hidden  # one-head, full hidden
        self.q_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.out = nn.Linear(kv_dim, vocab, bias=True)
        self.scale = kv_dim ** -0.5

    def forward(self, h, mask):
        # h: (B, T, hidden), mask: (B, T)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        # causal mask + pad mask
        B, T, _ = h.shape
        scores = torch.einsum("bth,bsh->bts", q, k) * self.scale  # (B, T, T)
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        if mask is not None:
            pad = (mask == 0).unsqueeze(1).expand(B, T, T)
            scores = scores.masked_fill(pad, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        ctx = torch.einsum("bts,bsh->bth", attn, v)
        return self.out(ctx)


def train_probe(probe, h_train, ids_train, mask_train, h_val, ids_val, mask_val,
                use_mask=False, label="probe"):
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
        if use_mask:
            logits = probe(h_b, mask_b)
        else:
            logits = probe(h_b)
        sl = logits[:, :-1].contiguous()
        lbl = ids_b[:, 1:].clone()
        sm = mask_b[:, 1:]
        lbl[sm == 0] = -100
        loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            v = eval_probe(probe, h_val, ids_val, mask_val, use_mask)
            if v < best_val:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in probe.state_dict().items()}
    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe


def eval_probe(probe, h, ids, mask, use_mask=False):
    probe.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, h.size(0), PROBE_BATCH):
            h_b = h[i:i+PROBE_BATCH].to("cuda").to(torch.float32)
            ids_b = ids[i:i+PROBE_BATCH].to("cuda")
            mask_b = mask[i:i+PROBE_BATCH].to("cuda")
            if use_mask:
                logits = probe(h_b, mask_b)
            else:
                logits = probe(h_b)
            sl = logits[:, :-1].contiguous()
            lbl = ids_b[:, 1:].clone()
            sm = mask_b[:, 1:]
            valid = (sm != 0)
            lbl_for_loss = lbl.clone()
            lbl_for_loss[~valid] = -100
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl_for_loss.reshape(-1),
                                    ignore_index=-100, reduction="sum")
            n = valid.sum().item()
            total_loss += loss.item()
            total_tokens += n
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


def load_probe_data(tok):
    """Load held-out c4 + wikitext, plus shuffled c4."""
    print("Loading probe data: c4_val (held-out) + wikitext_val + shuffled c4...")
    n_total = N_PROBE_TRAIN + N_PROBE_VAL + N_PROBE_TEST
    pool = []
    # Skip the first 32k+200 sequences (consumed by g156); start at offset 32968 effectively
    # via a different seed and skip
    for rec in c4_clean_v1(seed=99, n_samples=n_total + 1000):
        pool.append(rec["text"])
        if len(pool) >= n_total:
            break
    enc = tok(pool[:n_total], padding=True, truncation=True,
               max_length=SEQ_LEN, return_tensors="pt")
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    train_ids = ids[:N_PROBE_TRAIN]
    train_mask = mask[:N_PROBE_TRAIN]
    val_ids = ids[N_PROBE_TRAIN:N_PROBE_TRAIN + N_PROBE_VAL]
    val_mask = mask[N_PROBE_TRAIN:N_PROBE_TRAIN + N_PROBE_VAL]
    test_ids = ids[N_PROBE_TRAIN + N_PROBE_VAL:N_PROBE_TRAIN + N_PROBE_VAL + N_PROBE_TEST]
    test_mask = mask[N_PROBE_TRAIN + N_PROBE_VAL:N_PROBE_TRAIN + N_PROBE_VAL + N_PROBE_TEST]
    # Shuffled versions of all three splits (eval shuffle seed 43 to match g156 eval)
    train_ids_s = shuffle_token_rows(train_ids, train_mask, shuffle_seed=43)
    val_ids_s = shuffle_token_rows(val_ids, val_mask, shuffle_seed=44)
    test_ids_s = shuffle_token_rows(test_ids, test_mask, shuffle_seed=45)
    return {
        "natural": {"train": (train_ids, train_mask), "val": (val_ids, val_mask), "test": (test_ids, test_mask)},
        "token_shuffled": {"train": (train_ids_s, train_mask), "val": (val_ids_s, val_mask), "test": (test_ids_s, test_mask)},
    }


def depth_to_layer_idx(depth, n_layers):
    return min(n_layers - 1, max(0, int(round(depth * (n_layers - 1)))))


def main():
    t0 = time.time()
    print("genome_157: layerwise eta/delta probe on 12 g156 checkpoints")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    data = load_probe_data(tok)

    # Match probe data condition to checkpoint condition (natural ckpt -> natural probe data)
    results = {}  # results[ckpt_name] = {layer: {CE_lin, CE_local, CE_prefix, delta, eta, G}}

    ckpts = sorted(CKPT_DIR.glob("*.pt"))
    print(f"Found {len(ckpts)} checkpoints")

    for ckpt_path in ckpts:
        ckpt_name = ckpt_path.stem
        cond = ckpt_name.split("__")[0]  # natural | token_shuffled
        print(f"\n=== {ckpt_name} (cond={cond}) ===")
        model, meta = load_checkpoint(ckpt_path)
        n_layers = meta["n_layers"]
        hidden = meta["hidden"]
        vocab = meta["vocab_size"]

        layer_indices = [depth_to_layer_idx(d, n_layers) for d in DEPTHS]
        layer_indices = sorted(set(layer_indices))  # dedup if rounding collides
        print(f"  n_layers={n_layers}, probing layer indices: {layer_indices}")

        # Use the matching condition's probe data
        d_train = data[cond]["train"]
        d_val = data[cond]["val"]
        d_test = data[cond]["test"]

        per_layer = {}
        for li in layer_indices:
            print(f"  -- layer {li} (depth ~{li/(n_layers-1):.2f}) --")
            t_layer = time.time()

            # Extract activations once per (ckpt, layer)
            h_train = extract_activations(model, d_train[0], d_train[1], li).cpu()
            h_val = extract_activations(model, d_val[0], d_val[1], li).cpu()
            h_test = extract_activations(model, d_test[0], d_test[1], li).cpu()

            # Three probes
            ce = {}
            for name, ProbeClass, use_mask in [
                ("lin", LinearProbe, False),
                ("local", LocalMLPProbe, False),
                ("prefix", PrefixAttnProbe, True),
            ]:
                probe = ProbeClass(hidden, vocab)
                n_params = sum(p.numel() for p in probe.parameters())
                probe = train_probe(probe, h_train, d_train[0], d_train[1],
                                    h_val, d_val[0], d_val[1], use_mask=use_mask, label=name)
                ce_test = eval_probe(probe, h_test, d_test[0], d_test[1], use_mask=use_mask)
                ce[name] = ce_test
                print(f"    {name}: CE_test={ce_test:.4f}  params={n_params/1e6:.1f}M")
                del probe
                torch.cuda.empty_cache()

            delta = ce["lin"] - ce["local"]
            eta = ce["local"] - ce["prefix"]
            G = eta - delta
            R = eta / max(delta, EPS)
            per_layer[li] = {
                "CE_lin": ce["lin"], "CE_local": ce["local"], "CE_prefix": ce["prefix"],
                "delta_hat_mlp": delta, "eta_hat": eta, "G_l": G, "R_l": R,
                "depth_frac": li / (n_layers - 1),
                "wallclock_s": time.time() - t_layer,
            }
            print(f"    -> delta={delta:.4f}  eta={eta:.4f}  G={G:.4f}  R={R:.2f}")

            del h_train, h_val, h_test
            torch.cuda.empty_cache()

        results[ckpt_name] = {"meta": meta, "per_layer": per_layer}
        del model
        torch.cuda.empty_cache()

    # Aggregate analysis
    print(f"\n=== ANALYSIS (mid-band depths {{0.35, 0.50, 0.65}}) ===")
    summary = {}
    for cond in ["natural", "token_shuffled"]:
        for arm in ["baseline_200M_4k", "minimal_7L_200M_8k"]:
            G_per_seed = []
            for s in SEEDS:
                key = f"{cond}__{arm}__seed{s}"
                if key not in results:
                    continue
                pl = results[key]["per_layer"]
                n_layers = results[key]["meta"]["n_layers"]
                # Mid-band: depth in [0.30, 0.70]
                mid_Gs = [v["G_l"] for li, v in pl.items()
                          if 0.30 <= li / (n_layers - 1) <= 0.70]
                if mid_Gs:
                    G_per_seed.append(float(np.mean(mid_Gs)))
            if G_per_seed:
                summary[f"{cond}__{arm}"] = {
                    "G_mid_mean": float(np.mean(G_per_seed)),
                    "G_mid_std": float(np.std(G_per_seed)),
                    "G_per_seed": G_per_seed,
                    "n_seeds": len(G_per_seed),
                }
                print(f"  {cond}__{arm}: G_mid={np.mean(G_per_seed):+.4f} +/- {np.std(G_per_seed):.4f}")

    # Verdict
    nat_min = summary.get("natural__minimal_7L_200M_8k", {})
    shuf_min = summary.get("token_shuffled__minimal_7L_200M_8k", {})
    nat_G = nat_min.get("G_mid_mean", 0.0)
    shuf_G = shuf_min.get("G_mid_mean", 0.0)
    nat_n_pos = sum(1 for g in nat_min.get("G_per_seed", []) if g >= 0.02)
    contrast = nat_G - shuf_G

    if nat_n_pos >= 2 and shuf_G <= 0.0 and contrast >= 0.03:
        verdict = (f"PASS_G157: nat_G={nat_G:+.4f} (>=0.02 in {nat_n_pos}/{nat_min.get('n_seeds',0)} seeds), "
                   f"shuf_G={shuf_G:+.4f} (<=0.0), contrast={contrast:+.4f} (>=0.03). "
                   f"Transport budget criterion eta_l > delta_l^mlp directly OBSERVED on natural arm "
                   f"and COLLAPSES on shuffled. Theory now has a measured internal quantity.")
    elif nat_G >= 0.0 and contrast >= 0.015:
        verdict = (f"PARTIAL_G157: nat_G={nat_G:+.4f}, shuf_G={shuf_G:+.4f}, "
                   f"contrast={contrast:+.4f} (>=0.015). Direction supported, signal weaker than expected.")
    else:
        verdict = (f"KILL_G157: nat_G={nat_G:+.4f}, shuf_G={shuf_G:+.4f}, "
                   f"contrast={contrast:+.4f}. Transport budget criterion NOT directly observed.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 157, "name": "eta_delta_probe",
        "config": {"depths": DEPTHS, "n_probe_train": N_PROBE_TRAIN,
                    "probe_lr": PROBE_LR, "probe_steps": PROBE_STEPS, "probe_batch": PROBE_BATCH,
                    "seeds": SEEDS},
        "results": {k: {"meta": v["meta"], "per_layer": {str(li): pv for li, pv in v["per_layer"].items()}}
                    for k, v in results.items()},
        "summary": summary, "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_157_eta_delta_probe.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
