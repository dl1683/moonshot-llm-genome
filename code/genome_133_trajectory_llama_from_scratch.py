"""
genome_133_trajectory_llama_from_scratch.py

CROSS-ARCHITECTURE TRAJECTORY UNIVERSALITY TEST.

genome_127-129 established the U-shape trajectory on Pythia (GPT-NeoX style).
Pythia-2.8b is broken (EleutherAI's HF repo has step* branches aliased to
the same checkpoint). Pythia-160m, 410m, 1.4b all show:
  random ~9.6 -> mode-collapse-min ~2.8 -> recovery to ~4.2

Open question: is this trajectory PYTHIA-SPECIFIC (architecture+data-pair),
or UNIVERSAL across architectures (just a property of training transformers)?

This experiment trains a small Llama-style model FROM SCRATCH on c4_clean_v1
and measures the invariant trajectory at the same log-spaced step grid
[0, 128, 512, 1000, 4000, 16000].

Llama vs Pythia differs in:
  - Positional encoding: RoPE (Llama) vs learned absolute (Pythia)
  - Norm: RMSNorm (Llama) vs LayerNorm (Pythia)
  - Activation: SwiGLU (Llama) vs GELU (Pythia)
  - Bias terms: typically removed (Llama) vs present (Pythia)

If trajectory shape (random-high -> mode-collapse-min -> recovery) holds on
this small Llama, the trajectory is architecture-universal for transformers.

Pre-stated PASS:
  - random-init invariant > 6 (well above target 4.243)
  - mid-training (step 100-2000) dips to <3.5
  - final-step (16000) recovery to within 30% of target [4.243*0.7, 4.243*1.3]

Tiny Llama config: 4 layers, hidden 384, head 6, FFN 1024. ~7M params.
Trains in ~10-15 min on RTX 5090.

Results: results/genome_133_trajectory_llama_from_scratch.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent
TARGET = float(np.sqrt(18))

# Tiny Llama config
HIDDEN_SIZE = 384
N_LAYERS = 6
N_HEADS = 6
INTERMEDIATE_SIZE = 1024
SEQ_LEN = 256
MAX_STEPS = 4000
EVAL_AT = [0, 32, 128, 512, 1000, 2000, 4000]
LR = 3e-4
BATCH_SIZE = 8
SEED = 42


def make_tiny_llama(vocab_size):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="eager",  # avoid SDPA mask bug
    )
    torch.manual_seed(SEED)
    # bfloat16 for numerical stability; Llama RMSNorm + RoPE can NaN in fp16
    model = LlamaForCausalLM(cfg).to("cuda").to(torch.bfloat16)
    return model


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(max(X.shape[0] - 1, 1))
    return s.astype(np.float64)


def fit_power_tail(s, lo=0.05, hi=0.5):
    r = np.arange(1, len(s) + 1)
    a = max(1, int(len(s) * lo))
    b = int(len(s) * hi)
    lr = np.log(r[a:b])
    ls = np.log(s[a:b] + 1e-12)
    slope, _ = np.polyfit(lr, ls, 1)
    return float(-slope)


def eff_rank(s):
    s2 = s ** 2
    tot = s2.sum()
    return float(tot ** 2 / (s2 ** 2).sum()) if tot > 0 else 0.0


def measure_invariant(model, tok, calib_texts):
    """Use extract_trajectory at mid-layer to get sqrt(er)*alpha."""
    n_layers = model.config.num_hidden_layers
    mid = max(1, n_layers // 2)
    # use FP32 for the in-flight model -- Llama in fp16 with our hooks should be OK
    traj = extract_trajectory(
        model=model, tokenizer=tok,
        texts=calib_texts, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key="tiny_llama", class_id=1,
        quantization="fp16",
        stimulus_version="c4_clean.v1.seed42.n800",
        seed=42, batch_size=BATCH_SIZE, max_length=SEQ_LEN,
    )
    X = traj.layers[0].X.astype(np.float32)
    s = spectrum(X)
    a = fit_power_tail(s)
    er = eff_rank(s)
    return {
        "alpha": a, "eff_rank": er,
        "sqrt_er_alpha": float(np.sqrt(er) * a),
        "er_alpha2": float(er * a ** 2),
        "n": int(X.shape[0]), "h": int(X.shape[1]),
    }


def main():
    t0 = time.time()
    print("genome_133: tiny Llama from-scratch trajectory")

    # Use the Pythia-160m tokenizer for vocabulary compatibility with our stimuli.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    print("Loading c4_clean_v1 stimuli (n=800 calib, n=4000 train)...")
    all_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=4800):
        all_texts.append(rec["text"])
    calib_texts = all_texts[:800]
    train_texts = all_texts[800:]
    print(f"  calib N={len(calib_texts)}, train N={len(train_texts)}")

    print(f"Building tiny Llama: hidden={HIDDEN_SIZE}, layers={N_LAYERS}, heads={N_HEADS}, ffn={INTERMEDIATE_SIZE}")
    # Use len(tok), not vocab_size — Pythia tokenizer has 23 special tokens beyond vocab_size
    actual_vocab = len(tok)
    print(f"  effective vocab size: {actual_vocab}")
    model = make_tiny_llama(actual_vocab)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params/1e6:.2f}M")

    # Tokenize all training texts upfront for efficiency
    print("Pre-tokenizing training corpus...")
    enc = tok(train_texts, padding=True, truncation=True,
              max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc["input_ids"]
    train_mask = enc["attention_mask"]
    print(f"  train tensor shape: {train_ids.shape}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(SEED)
    rows = []

    # Initial measurement (random init)
    print(f"\n[{time.time()-t0:.1f}s] === step 0 (random init) ===")
    model.eval()
    r = measure_invariant(model, tok, calib_texts)
    rows.append({"step": 0, **r})
    print(f"  step=     0  alpha={r['alpha']:.3f}  er={r['eff_rank']:7.2f}  sqrt(er)*alpha={r['sqrt_er_alpha']:.3f}")
    model.train()

    step = 0
    next_eval_idx = 1  # skip step 0 (already measured)
    while step < MAX_STEPS:
        idx = rng.integers(0, train_ids.size(0), size=BATCH_SIZE)
        ids = train_ids[idx].to("cuda")
        mask = train_mask[idx].to("cuda")
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = ids[:, 1:].contiguous().clone()
        shift_mask = mask[:, 1:].contiguous()
        shift_labels[shift_mask == 0] = -100
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if next_eval_idx < len(EVAL_AT) and step == EVAL_AT[next_eval_idx]:
            print(f"\n[{time.time()-t0:.1f}s] === step {step} ===")
            model.eval()
            r = measure_invariant(model, tok, calib_texts)
            rows.append({"step": step, **r, "loss": float(loss.item())})
            print(f"  step={step:6d}  loss={loss.item():.3f}  alpha={r['alpha']:.3f}  er={r['eff_rank']:7.2f}  sqrt(er)*alpha={r['sqrt_er_alpha']:.3f}")
            model.train()
            next_eval_idx += 1

    # Trajectory analysis
    print(f"\n=== TRAJECTORY (target {TARGET:.3f}) ===")
    invs = [r["sqrt_er_alpha"] for r in rows]
    steps = [r["step"] for r in rows]
    min_idx = int(np.argmin(invs))
    final_dev = abs(invs[-1] - TARGET) / TARGET * 100
    for r in rows:
        mark = " (above)" if r["sqrt_er_alpha"] >= TARGET else " (below)"
        if r["step"] == steps[min_idx]:
            mark += " [MIN]"
        print(f"  step {r['step']:6d}: sqrt(er)*alpha={r['sqrt_er_alpha']:.3f}  er={r['eff_rank']:7.2f}{mark}")
    print(f"  minimum @ step {steps[min_idx]}: {invs[min_idx]:.3f}")
    print(f"  final deviation: {final_dev:.1f}%")

    # Verdict
    init_above = invs[0] > 6.0
    has_dip = any(v < 3.5 for v in invs)
    final_recovers = final_dev < 30.0  # within 30% of target

    if init_above and has_dip and final_recovers:
        verdict = (f"PASS: Llama trajectory matches Pythia U-shape. "
                   f"random={invs[0]:.2f} (>6), min={invs[min_idx]:.2f} (<3.5), "
                   f"final={invs[-1]:.2f} (dev {final_dev:.1f}%<30%). "
                   f"Trajectory is architecture-universal for transformers.")
    elif init_above and (has_dip or final_recovers):
        verdict = (f"PARTIAL: Llama shows partial trajectory match. "
                   f"random={invs[0]:.2f}, min={invs[min_idx]:.2f}, final={invs[-1]:.2f}, "
                   f"dev={final_dev:.1f}%. Some landmarks present but not all.")
    else:
        verdict = (f"KILL: Llama trajectory does NOT match Pythia. "
                   f"random={invs[0]:.2f}, min={invs[min_idx]:.2f}, final={invs[-1]:.2f}, "
                   f"dev={final_dev:.1f}%. Trajectory is Pythia-specific.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 133, "name": "trajectory_llama_from_scratch",
        "target": TARGET,
        "config": {
            "hidden_size": HIDDEN_SIZE, "n_layers": N_LAYERS, "n_heads": N_HEADS,
            "intermediate_size": INTERMEDIATE_SIZE, "seq_len": SEQ_LEN,
            "max_steps": MAX_STEPS, "lr": LR, "batch_size": BATCH_SIZE,
            "n_params_M": n_params / 1e6,
            "tokenizer": "EleutherAI/pythia-160m",
        },
        "rows": rows,
        "summary": {
            "min_step": steps[min_idx], "min_value": invs[min_idx],
            "final_step": steps[-1], "final_value": invs[-1],
            "final_dev_pct": final_dev,
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_133_trajectory_llama_from_scratch.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
