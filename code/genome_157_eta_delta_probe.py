"""
genome_157_eta_delta_probe.py — PILOT (rewritten per Codex pre-flight)

POST-g156-PASS PROBE: layerwise transport surplus G_l = eta_hat_l - delta_hat_l^mlp.

Pre-reg LOCKED (relocked): research/prereg/genome_157_eta_delta_probe_pilot_2026-04-26.md
Theory: research/derivations/prefix_information_transport.md
Program: research/programs/post_g156_pass_program.md §g157
Codex pre-flight: codex_outputs/g157_pre_flight.md (Severity 7+ findings all addressed)

PILOT scope: seed=42 subset only (4 ckpts), 3 mid-band depths only, 500 probe-train steps,
BF16 throughout, true validation splits with 13-token dedup audit, hard preflight abort
if projected runtime > 3.5 hr.

If pilot DIRECTIONAL_SUPPORT: write 3-seed prereg and execute for locked verdict.
If pilot KILL: theory mechanism wrong; pivot to distillation track.

Compute target: ~1.5 hr.
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
N_OOD_TEST = 256

# Locked mid-band layer indices per prereg §2
MIDBAND_INDICES = {14: [5, 7, 9], 7: [2, 3, 4]}

PILOT_SEED = 42
PROBE_LR = 1e-3
PROBE_STEPS = 500
PROBE_BATCH = 32
PROBE_BENCH_STEPS = 50
HARD_ABORT_HOURS = 3.5
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


def extract_activations_to_cpu(model, ids, mask, layer_idx):
    """Run forward; capture residual-stream output post-block, move to CPU per-batch.
    Avoids GPU activation churn flagged by Codex Severity-6."""
    captured = []

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach().cpu())  # CPU immediately

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for i in range(0, ids.size(0), PROBE_BATCH):
                _ = model(input_ids=ids[i:i+PROBE_BATCH].to("cuda"),
                          attention_mask=mask[i:i+PROBE_BATCH].to("cuda"),
                          use_cache=False)
    finally:
        handle.remove()
    return torch.cat(captured, dim=0)


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
            # Match LinearProbe params: hidden*vocab + vocab.
            # MLP params: hidden*mlp_hidden + mlp_hidden + mlp_hidden*vocab + vocab.
            # Solve: mlp_hidden ≈ (hidden*vocab) / (hidden + vocab).
            mlp_hidden = max(1, (hidden * vocab) // (hidden + vocab))
        self.fc1 = nn.Linear(hidden, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, vocab)

    def forward(self, h):
        return self.fc2(F.gelu(self.fc1(h)))


class PrefixAttnProbe(nn.Module):
    """Causal self-attn over same-layer prefix activations. kv_dim chosen for ~1% param match."""
    def __init__(self, hidden, vocab, kv_dim=None):
        super().__init__()
        if kv_dim is None:
            # Match LinearProbe params: hidden*vocab + vocab.
            # PrefixAttn params: 3*hidden*kv_dim + kv_dim*vocab + vocab.
            # Solve: kv_dim ≈ (hidden*vocab) / (3*hidden + vocab).
            kv_dim = max(1, (hidden * vocab) // (3 * hidden + vocab))
        self.q_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.out = nn.Linear(kv_dim, vocab, bias=True)
        self.scale = kv_dim ** -0.5

    def forward(self, h, mask):
        q = self.q_proj(h); k = self.k_proj(h); v = self.v_proj(h)
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


def train_probe(probe, h_train, ids_train, mask_train, h_val, ids_val, mask_val, use_mask=False):
    # FP32 throughout for numerical stability on shuffled distribution
    # (BF16 caused lin probe CE > 200 on shuffled in v2 PILOT).
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
        logits = probe(h_b, mask_b) if use_mask else probe(h_b)
        sl = logits[:, :-1].contiguous()
        lbl = ids_b[:, 1:].clone()
        sm = mask_b[:, 1:]
        lbl[sm == 0] = -100
        loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            v = eval_probe(probe, h_val, ids_val, mask_val, use_mask)
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


def eval_probe(probe, h, ids, mask, use_mask=False):
    probe.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, h.size(0), PROBE_BATCH):
            h_b = h[i:i+PROBE_BATCH].to("cuda").to(torch.float32)
            ids_b = ids[i:i+PROBE_BATCH].to("cuda")
            mask_b = mask[i:i+PROBE_BATCH].to("cuda")
            logits = probe(h_b, mask_b) if use_mask else probe(h_b)
            sl = logits[:, :-1].contiguous()
            lbl = ids_b[:, 1:].clone()
            sm = mask_b[:, 1:]
            valid = (sm != 0)
            lbl_for_loss = lbl.clone()
            lbl_for_loss[~valid] = -100
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl_for_loss.reshape(-1),
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
    """Set of all 13-token windows present in (ids, mask)."""
    hashes = set()
    n, T = ids.shape
    for r in range(n):
        valid = mask[r].sum().item()
        if valid < 13:
            continue
        row = ids[r, :valid].tolist()
        for i in range(len(row) - 12):
            hashes.add(tuple(row[i:i+13]))
    return hashes


def load_validation_data(tok):
    """Load c4 + wikitext VALIDATION splits per locked prereg + dedup audit."""
    from datasets import load_dataset

    print("Loading c4 validation split...")
    try:
        ds_c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        c4_texts = []
        for ex in ds_c4:
            t = ex["text"]
            if len(t) > 200:
                c4_texts.append(t)
            if len(c4_texts) >= N_PROBE_TRAIN + N_PROBE_VAL + N_PROBE_TEST:
                break
    except Exception as e:
        print(f"  c4 validation load failed: {e}")
        print("  Trying smaller fallback (allenai/c4 with split=validation, no streaming)")
        # Try non-streaming as fallback
        ds_c4 = load_dataset("allenai/c4", "en", split="validation",
                              data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"})
        c4_texts = []
        for ex in ds_c4:
            t = ex["text"]
            if len(t) > 200:
                c4_texts.append(t)
            if len(c4_texts) >= N_PROBE_TRAIN + N_PROBE_VAL + N_PROBE_TEST:
                break
    print(f"  loaded {len(c4_texts)} c4-val sequences")

    print("Loading wikitext-103 validation split...")
    ds_wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    wt_texts = []
    for ex in ds_wt:
        t = ex["text"].strip()
        if len(t) > 200:
            wt_texts.append(t[:1500])
        if len(wt_texts) >= N_OOD_TEST:
            break
    print(f"  loaded {len(wt_texts)} wikitext-val sequences")

    enc_c4 = tok(c4_texts[:N_PROBE_TRAIN + N_PROBE_VAL + N_PROBE_TEST],
                  padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc_c4["input_ids"][:N_PROBE_TRAIN]
    train_mask = enc_c4["attention_mask"][:N_PROBE_TRAIN]
    val_ids = enc_c4["input_ids"][N_PROBE_TRAIN:N_PROBE_TRAIN + N_PROBE_VAL]
    val_mask = enc_c4["attention_mask"][N_PROBE_TRAIN:N_PROBE_TRAIN + N_PROBE_VAL]
    test_ids = enc_c4["input_ids"][N_PROBE_TRAIN + N_PROBE_VAL:]
    test_mask = enc_c4["attention_mask"][N_PROBE_TRAIN + N_PROBE_VAL:]

    # Dedup audit: regenerate g156 train slice, hash, check overlap with our train
    print("Running 13-token rolling-hash dedup audit vs g156 train slice...")
    sys.path.insert(0, str(_THIS_DIR))
    from stimulus_banks import c4_clean_v1
    g156_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=2048):  # subsample for hash audit
        g156_texts.append(rec["text"])
        if len(g156_texts) >= 2048:
            break
    enc_g156 = tok(g156_texts, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    g156_hashes = thirteen_token_hashes(enc_g156["input_ids"], enc_g156["attention_mask"])
    train_hashes = thirteen_token_hashes(train_ids, train_mask)
    overlap = train_hashes & g156_hashes
    overlap_pct = 100.0 * len(overlap) / max(len(train_hashes), 1)
    print(f"  13-gram overlap: {len(overlap)} / {len(train_hashes)} train windows = {overlap_pct:.2f}%")
    if overlap_pct > 5.0:
        raise RuntimeError(f"dedup audit FAILED: {overlap_pct:.2f}% > 5% overlap")

    # Shuffled versions
    train_ids_s = shuffle_token_rows(train_ids, train_mask, shuffle_seed=43)
    val_ids_s = shuffle_token_rows(val_ids, val_mask, shuffle_seed=44)
    test_ids_s = shuffle_token_rows(test_ids, test_mask, shuffle_seed=45)

    enc_wt = tok(wt_texts[:N_OOD_TEST], padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    return {
        "natural": {"train": (train_ids, train_mask), "val": (val_ids, val_mask), "test": (test_ids, test_mask)},
        "token_shuffled": {"train": (train_ids_s, train_mask), "val": (val_ids_s, val_mask), "test": (test_ids_s, test_mask)},
        "wikitext_test": (enc_wt["input_ids"], enc_wt["attention_mask"]),
    }


def microbenchmark(model, hidden, vocab, n_layers):
    """50-step probe train microbenchmark for hard-abort."""
    print("\nMicrobenchmark: 50 probe-train steps to estimate total runtime...")
    fake_ids = torch.randint(0, vocab, (PROBE_BATCH, SEQ_LEN), device="cuda")
    fake_mask = torch.ones_like(fake_ids)
    fake_h = torch.randn(PROBE_BATCH * 4, SEQ_LEN, hidden, dtype=torch.bfloat16, device="cuda")
    rng = np.random.default_rng(0)
    times = {}
    for name, ProbeClass, use_mask in [
        ("lin", LinearProbe, False),
        ("local", LocalMLPProbe, False),
        ("prefix", PrefixAttnProbe, True),
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
            logits = probe(h_b, mask_b) if use_mask else probe(h_b)
            sl = logits[:, :-1].float()
            lbl = ids_b[:, 1:].clone()
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        per_step = elapsed / PROBE_BENCH_STEPS
        times[name] = per_step
        n_params = sum(p.numel() for p in probe.parameters())
        print(f"  {name}: {1000*per_step:.1f} ms/step  params={n_params/1e6:.1f}M")
        del probe, opt
        torch.cuda.empty_cache()
    n_layers_used = len(MIDBAND_INDICES.get(n_layers, [0,1,2]))
    n_ckpts = 4  # PILOT scope
    total_probe_time = sum(times.values()) * PROBE_STEPS * n_layers_used * n_ckpts
    extract_overhead = n_layers_used * n_ckpts * 30  # ~30s per layer for activation extract
    total_seconds = total_probe_time + extract_overhead
    total_hours = total_seconds / 3600
    print(f"  PROJECTED total time: {total_hours:.2f} hr (probe={total_probe_time/3600:.2f}, extract={extract_overhead/3600:.2f})")
    if total_hours > HARD_ABORT_HOURS:
        raise RuntimeError(f"PROJECTED runtime {total_hours:.2f} hr > envelope {HARD_ABORT_HOURS} hr; abort and relock.")
    return total_hours


def main():
    t0 = time.time()
    print("genome_157 PILOT: layerwise eta/delta probe (1-seed, mid-band only, BF16)")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    # Load data first (Codex Severity-10 fix)
    data = load_validation_data(tok)

    # Find PILOT checkpoints (seed 42 only)
    all_ckpts = sorted(CKPT_DIR.glob("*seed42.pt"))
    if len(all_ckpts) != 4:
        raise RuntimeError(f"PILOT requires exactly 4 seed=42 checkpoints, found {len(all_ckpts)}")
    print(f"PILOT scope: {len(all_ckpts)} checkpoints (seed=42 only)")
    for c in all_ckpts:
        print(f"  - {c.name}")

    # Microbenchmark + hard-abort BEFORE loading first ckpt
    # (use baseline 14L hidden=1024 vocab=50277 as reference)
    model_ref, meta_ref = load_checkpoint(all_ckpts[0])
    projected_hr = microbenchmark(model_ref, meta_ref["hidden"], meta_ref["vocab_size"], meta_ref["n_layers"])
    del model_ref
    torch.cuda.empty_cache()

    results = {}
    for ckpt_path in all_ckpts:
        ckpt_name = ckpt_path.stem
        cond = ckpt_name.split("__")[0]
        print(f"\n=== {ckpt_name} (cond={cond}) ===")
        model, meta = load_checkpoint(ckpt_path)
        n_layers = meta["n_layers"]
        hidden = meta["hidden"]
        vocab = meta["vocab_size"]

        layer_indices = MIDBAND_INDICES.get(n_layers)
        if layer_indices is None:
            raise RuntimeError(f"no locked midband indices for n_layers={n_layers}")
        print(f"  n_layers={n_layers}, midband indices: {layer_indices}")

        d_train = data[cond]["train"]
        d_val = data[cond]["val"]
        d_test = data[cond]["test"]

        per_layer = {}
        for li in layer_indices:
            t_layer = time.time()
            print(f"  -- layer {li} (depth ~{li/(n_layers-1):.2f}) --")
            h_train = extract_activations_to_cpu(model, d_train[0], d_train[1], li)
            h_val = extract_activations_to_cpu(model, d_val[0], d_val[1], li)
            h_test = extract_activations_to_cpu(model, d_test[0], d_test[1], li)

            ce = {}
            params_M = {}
            for name, ProbeClass, use_mask in [
                ("lin", LinearProbe, False),
                ("local", LocalMLPProbe, False),
                ("prefix", PrefixAttnProbe, True),
            ]:
                probe = ProbeClass(hidden, vocab)
                params_M[name] = sum(p.numel() for p in probe.parameters()) / 1e6
                probe = train_probe(probe, h_train, d_train[0], d_train[1],
                                    h_val, d_val[0], d_val[1], use_mask=use_mask)
                ce[name] = eval_probe(probe, h_test, d_test[0], d_test[1], use_mask=use_mask)
                print(f"    {name}: CE={ce[name]:.4f}  params={params_M[name]:.2f}M")
                del probe
                torch.cuda.empty_cache()

            delta = ce["lin"] - ce["local"]
            eta = ce["local"] - ce["prefix"]
            G = eta - delta
            R = eta / max(delta, EPS)
            per_layer[li] = {
                "CE_lin": ce["lin"], "CE_local": ce["local"], "CE_prefix": ce["prefix"],
                "delta_hat_mlp": delta, "eta_hat": eta, "G_l": G, "R_l": R,
                "depth_frac": li / (n_layers - 1), "params_M": params_M,
                "wallclock_s": time.time() - t_layer,
            }
            print(f"    -> delta={delta:.4f} eta={eta:.4f} G={G:+.4f} R={R:.2f}")
            del h_train, h_val, h_test
            torch.cuda.empty_cache()

        results[ckpt_name] = {"meta": meta, "per_layer": per_layer}
        del model
        torch.cuda.empty_cache()

    # Completeness guard
    if len(results) != 4:
        raise RuntimeError(f"PILOT incomplete: only {len(results)} of 4 ckpts ran")

    # Aggregate (PILOT — single-seed, no seed-stat)
    print(f"\n=== ANALYSIS (PILOT, single seed=42) ===")
    summary = {}
    for cond in ["natural", "token_shuffled"]:
        for arm in ["baseline_200M_4k", "minimal_7L_200M_8k"]:
            key = f"{cond}__{arm}__seed42"
            if key not in results:
                summary[f"{cond}__{arm}"] = {"G_mid_mean": float("nan"), "n_layers": 0}
                continue
            pl = results[key]["per_layer"]
            G_vals = [v["G_l"] for v in pl.values()]
            summary[f"{cond}__{arm}"] = {
                "G_mid_mean": float(np.mean(G_vals)),
                "G_per_layer": {str(li): pl[li]["G_l"] for li in pl},
                "n_layers": len(G_vals),
            }
            print(f"  {cond}__{arm}: G_mid={summary[f'{cond}__{arm}']['G_mid_mean']:+.4f} "
                  f"(per-layer {[f'{g:+.3f}' for g in G_vals]})")

    nat_min = summary.get("natural__minimal_7L_200M_8k", {})
    shuf_min = summary.get("token_shuffled__minimal_7L_200M_8k", {})
    nat_G = nat_min.get("G_mid_mean", 0.0)
    shuf_G = shuf_min.get("G_mid_mean", 0.0)
    contrast = nat_G - shuf_G

    if nat_G >= 0.02 and shuf_G <= 0.0 and contrast >= 0.03:
        verdict = (f"DIRECTIONAL_SUPPORT: nat_G={nat_G:+.4f} (>=0.02), shuf_G={shuf_G:+.4f} (<=0.0), "
                   f"contrast={contrast:+.4f} (>=0.03). Pilot supports theory; write 3-seed prereg + run.")
    elif nat_G >= 0.0 and contrast >= 0.015:
        verdict = (f"WEAK_SUPPORT: nat_G={nat_G:+.4f}, contrast={contrast:+.4f}. "
                   f"Direction OK but signal weak; redesign probe scale before scaling to 3 seeds.")
    else:
        verdict = (f"PILOT_KILL: nat_G={nat_G:+.4f}, shuf_G={shuf_G:+.4f}, contrast={contrast:+.4f}. "
                   f"Theory mechanism not supported; pivot to distillation track.")
    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 157, "name": "eta_delta_probe_pilot",
        "config": {"midband_indices": {str(k): v for k, v in MIDBAND_INDICES.items()},
                    "n_probe_train": N_PROBE_TRAIN, "probe_lr": PROBE_LR,
                    "probe_steps": PROBE_STEPS, "probe_batch": PROBE_BATCH,
                    "pilot_seed": PILOT_SEED, "projected_hours": projected_hr},
        "results": {k: {"meta": v["meta"],
                         "per_layer": {str(li): pv for li, pv in v["per_layer"].items()}}
                    for k, v in results.items()},
        "summary": summary,
        "nat_G_mean": nat_G, "shuf_G_mean": shuf_G, "contrast": contrast,
        "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_157_eta_delta_probe.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
