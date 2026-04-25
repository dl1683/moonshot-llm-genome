"""
genome_138_arch_prior_decomposition.py

CODEX Q1 — ARCHITECTURE-PRIOR DECOMPOSITION.

After g135/g136/g137 closed three process-descriptor candidates, the only
unexplained POSITIVE datum in the surgery branch is g125: random Qwen3 +
glue-only training (26% trainable, attn+MLP frozen at random init) closes
42.66% of the gap in 100 steps. g134 confirmed glue-only training reaches
the same endpoint manifold via a smooth trajectory.

Question: WHICH ARCHITECTURAL FEATURES carry this prior? Run one-factor
ablations on a tiny Llama, measure gap closure under glue-only training.

Held fixed across ALL arms:
  - Same vocabulary (Pythia-160m tokenizer, ~50277 tokens)
  - Same training protocol (glue-only: train embed + lm_head + RMSNorm gammas;
    freeze attn+MLP at their respective random init)
  - 100 training steps, lr=3e-4, batch=8
  - Same eval set (200 c4_clean_v1 sequences)
  - Same number of layers (6) and hidden_size (384) where applicable
  - Same SEED=42

Arms (one-factor ablations from the BASELINE = full Llama):
  A) BASELINE — full Llama (RoPE + RMSNorm + SwiGLU + attention + residual + 6 layers)
  B) NO_ATTENTION — replace self-attention with identity (just MLP + residual)
  C) NO_MLP — replace MLP with identity (just attention + residual)
  D) NO_RESIDUAL — remove residual connections (sequential block output)
  E) NO_CAUSAL_MASK — bidirectional attention (encoder-style)
  F) DEPTH_HALVED — 3 layers instead of 6
  G) WIDTH_HALVED — hidden=192 instead of 384
  H) FROZEN_RANDOM_LINEAR — replace attn+MLP with FROZEN random LINEAR
                            projections (no nonlinearity)

Metric: NLL on eval after 100 glue-train steps. Compare to:
  - Full random init NLL (baseline_random_full = ~10.9)
  - Full Llama trained from scratch — for context

Pre-stated criteria:
  PASS (architecture decomposable):
    Identify >=1 ablation that drops capability >=20% AND >=1 that
    preserves it within 5%. Capability is then LOCALIZABLE.
  PARTIAL: 10-20% drops indicate which features matter most, but no
    single irreducible core.
  KILL: all ablations preserve gap closure within ±5%. Architecture-prior
    is HOLISTIC across these features (or none of them is the core).

Compute estimate: 8 arms x 100 steps + 8 init/eval = ~5-8 min total.

Results: results/genome_138_arch_prior_decomposition.json
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
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent

SEQ_LEN = 256
BATCH_SIZE = 8
LR = 3e-4
TRAIN_STEPS = 100
EVAL_AT = [0, 25, 50, 100]
SEED = 42
N_EVAL = 200
N_TRAIN = 800


def make_baseline_llama(vocab_size, hidden=384, layers=6, heads=6, ffn=1024,
                         no_causal=False, no_attn=False, no_mlp=False,
                         no_residual=False, frozen_linear=False, seed=SEED):
    """Build a Llama-style model with optional ablations.
    All ablations are baked in via subclassing or post-construction patching."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        intermediate_size=ffn,
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="eager",
        is_decoder=not no_causal,
    )
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg).to("cuda").to(torch.bfloat16)

    # Ablation patches (all post-construction)
    if no_attn or no_mlp or no_residual or frozen_linear:
        for layer in model.model.layers:
            if no_attn:
                # Replace self_attn with an identity — return inputs as residual
                layer.self_attn = IdentityAttention()
            if no_mlp:
                # Replace MLP with identity returning zeros (residual passes through)
                layer.mlp = ZeroMLP()
            if no_residual:
                # Wrap layer's forward to NOT add residual
                _wrap_no_residual(layer)
            if frozen_linear:
                # Replace attn and MLP with FROZEN random linear projections
                d = hidden
                layer.self_attn = FrozenLinear(d, d, seed=seed + 1).to("cuda").to(torch.bfloat16)
                layer.mlp = FrozenLinear(d, d, seed=seed + 2).to("cuda").to(torch.bfloat16)

    if no_causal:
        # Patch the attention to use full (bidirectional) mask
        # We'll do this by setting is_causal=False on each attention forward call
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "is_causal"):
                layer.self_attn.is_causal = False

    return model


class IdentityAttention(nn.Module):
    """Replaces self-attention with an identity (passes hidden_states through unchanged
    while obeying the (output, attn_weights, past_kv) tuple API expected by Llama)."""
    def __init__(self):
        super().__init__()
        self.is_causal = False

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        # Llama expects (attn_output, attn_weights), here pass identity
        attn_output = hidden_states
        return attn_output, None


class ZeroMLP(nn.Module):
    """Replaces MLP with zero output (residual passes through unchanged)."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


class FrozenLinear(nn.Module):
    """Replaces attn/MLP with frozen random linear (single matmul + bias=0)."""
    def __init__(self, d_in, d_out, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(d_in, d_out) * (1.0 / np.sqrt(d_in)),
                                     requires_grad=False)

    def forward(self, *args, **kwargs):
        # Take first positional arg as hidden_states
        x = args[0]
        out = torch.matmul(x.to(self.weight.dtype), self.weight)
        # Llama expects tuple (out, weights) for attention; check kwarg
        # Mimic the attention interface: return tuple
        if "attention_mask" in kwargs or len(args) > 1:
            return out, None
        return out


def _wrap_no_residual(layer):
    """Wrap a Llama decoder layer to skip residual connections.
    The default forward does:
      attn_out = self.input_layernorm(x); attn_out = self.self_attn(attn_out)
      x = x + attn_out  # residual 1
      mlp_out = self.post_attention_layernorm(x); mlp_out = self.mlp(mlp_out)
      x = x + mlp_out   # residual 2
    We replace with: x = attn_out; x = mlp_out (no add)."""
    orig_forward = layer.forward

    def new_forward(hidden_states, attention_mask=None, position_ids=None,
                     past_key_value=None, output_attentions=False, use_cache=False,
                     cache_position=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        h = layer.input_layernorm(hidden_states)
        attn_kwargs = dict(attention_mask=attention_mask,
                            position_ids=position_ids, past_key_value=past_key_value,
                            output_attentions=output_attentions, use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings)
        # Filter kwargs that not all attn variants accept
        try:
            attn_out = layer.self_attn(h, **attn_kwargs)
        except TypeError:
            attn_out = layer.self_attn(h)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        # NO RESIDUAL: do not add to residual; just take attn_out
        h2 = layer.post_attention_layernorm(attn_out)
        mlp_out = layer.mlp(h2)
        # NO RESIDUAL: do not add
        outputs = (mlp_out,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (None,)
        return outputs

    layer.forward = new_forward


def is_glue_param(name):
    if "embed_tokens" in name or name == "lm_head.weight":
        return True
    if "layernorm" in name.lower() or name == "model.norm.weight":
        return True
    return False


def freeze_non_glue(model):
    n_train, n_total = 0, 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        p.requires_grad = is_glue_param(name)
        if p.requires_grad:
            n_train += p.numel()
    return n_train, n_total


def measure_eval_nll(model, eval_ids, eval_mask):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, eval_ids.size(0), BATCH_SIZE):
            ids = eval_ids[i:i+BATCH_SIZE].to("cuda")
            mask = eval_mask[i:i+BATCH_SIZE].to("cuda")
            try:
                out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            except Exception as e:
                print(f"  forward fail: {e}")
                return float("nan")
            logits = out.logits
            sl = logits[:, :-1].contiguous()
            lbl = ids[:, 1:].contiguous().clone()
            sm = mask[:, 1:].contiguous()
            lbl[sm == 0] = -100
            loss = F.cross_entropy(
                sl.view(-1, sl.size(-1)), lbl.view(-1),
                ignore_index=-100, reduction="sum",
            )
            n = (sm != 0).sum().item()
            total_loss += loss.item()
            total_tokens += n
    model.train()
    return total_loss / max(total_tokens, 1)


def train_arm(arm_name, model, train_ids, train_mask, eval_ids, eval_mask):
    n_train, n_total = freeze_non_glue(model)
    print(f"  trainable: {n_train:,}/{n_total:,} ({100*n_train/n_total:.1f}%)")
    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=LR, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(SEED)
    rows = []

    nll0 = measure_eval_nll(model, eval_ids, eval_mask)
    rows.append({"step": 0, "nll": nll0})
    print(f"  step=0  NLL={nll0:.3f}")
    if not np.isfinite(nll0):
        return rows  # arm broke

    model.train()
    next_idx = 1
    cursor = 0
    n_train_seqs = train_ids.size(0)
    for step in range(1, TRAIN_STEPS + 1):
        if cursor + BATCH_SIZE > n_train_seqs:
            cursor = 0
        ids = train_ids[cursor:cursor + BATCH_SIZE].to("cuda")
        mask = train_mask[cursor:cursor + BATCH_SIZE].to("cuda")
        cursor += BATCH_SIZE
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits
        sl = logits[:, :-1].contiguous()
        lbl = ids[:, 1:].contiguous().clone()
        sm = mask[:, 1:].contiguous()
        lbl[sm == 0] = -100
        loss = F.cross_entropy(
            sl.view(-1, sl.size(-1)), lbl.view(-1), ignore_index=-100
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if next_idx < len(EVAL_AT) and step == EVAL_AT[next_idx]:
            nll = measure_eval_nll(model, eval_ids, eval_mask)
            rows.append({"step": step, "nll": nll, "loss": float(loss.item())})
            print(f"  step={step:3d}  NLL={nll:.3f}  loss={loss.item():.3f}")
            model.train()
            next_idx += 1
    return rows


def main():
    t0 = time.time()
    print("genome_138: architecture-prior decomposition (Codex Q1)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    actual_vocab = len(tok)

    print("Loading c4_clean_v1 stimuli...")
    pool_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=N_TRAIN + N_EVAL):
        pool_texts.append(rec["text"])
    train_texts = pool_texts[:N_TRAIN]
    eval_texts = pool_texts[N_TRAIN:N_TRAIN + N_EVAL]

    enc_t = tok(train_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc_t["input_ids"]; train_mask = enc_t["attention_mask"]
    enc_e = tok(eval_texts, padding=True, truncation=True,
                 max_length=SEQ_LEN, return_tensors="pt")
    eval_ids = enc_e["input_ids"]; eval_mask = enc_e["attention_mask"]
    print(f"  train: {train_ids.shape}, eval: {eval_ids.shape}")

    arms = [
        ("baseline_full",        dict()),
        ("no_attention",         dict(no_attn=True)),
        ("no_mlp",               dict(no_mlp=True)),
        ("no_residual",          dict(no_residual=True)),
        ("no_causal_mask",       dict(no_causal=True)),
        ("depth_halved",         dict(layers=3)),
        ("width_halved",         dict(hidden=192, ffn=512, heads=6)),
        ("frozen_random_linear", dict(frozen_linear=True)),
    ]

    all_results = {}
    for arm_name, kwargs in arms:
        print(f"\n=== Arm: {arm_name} ===")
        try:
            model = make_baseline_llama(actual_vocab, **kwargs)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  built model: {n_params/1e6:.2f}M params")
            rows = train_arm(arm_name, model, train_ids, train_mask, eval_ids, eval_mask)
        except Exception as e:
            print(f"  ARM FAILED: {e}")
            import traceback; traceback.print_exc()
            rows = [{"step": 0, "nll": float("nan"), "error": str(e)}]
        all_results[arm_name] = {"rows": rows,
                                  "n_params_M": n_params / 1e6 if 'n_params' in dir() else None}
        try:
            del model; torch.cuda.empty_cache()
        except Exception:
            pass

    # === ANALYSIS ===
    print(f"\n=== ANALYSIS ===")
    baseline_step0 = all_results["baseline_full"]["rows"][0]["nll"]
    baseline_final = all_results["baseline_full"]["rows"][-1]["nll"]
    baseline_drop = baseline_step0 - baseline_final
    print(f"  baseline_full: step0 NLL={baseline_step0:.3f}, step100 NLL={baseline_final:.3f}, drop={baseline_drop:.3f}")

    summary = {}
    for arm_name in [a[0] for a in arms]:
        rows = all_results[arm_name]["rows"]
        step0 = rows[0]["nll"]
        final = rows[-1]["nll"]
        drop = step0 - final if (np.isfinite(step0) and np.isfinite(final)) else float("nan")
        # Capability gain relative to BASELINE drop
        rel_capability = drop / baseline_drop if baseline_drop > 0 else float("nan")
        summary[arm_name] = {
            "step0_nll": float(step0),
            "final_nll": float(final),
            "nll_drop": float(drop),
            "relative_capability": float(rel_capability),
        }
        print(f"  {arm_name:25s} step0={step0:.3f}  final={final:.3f}  drop={drop:.3f}  rel_capability={rel_capability:+.3f}")

    # Verdict
    drops_pct_of_baseline = {n: summary[n]["relative_capability"] for n in summary if n != "baseline_full"}
    catastrophic = [n for n, r in drops_pct_of_baseline.items() if r < 0.80]  # >20% drop
    preserved = [n for n, r in drops_pct_of_baseline.items() if 0.95 <= r <= 1.05]
    moderate = [n for n, r in drops_pct_of_baseline.items() if 0.80 <= r < 0.95]

    print(f"\n  catastrophic drops (>20%): {catastrophic}")
    print(f"  preserved (within 5%):     {preserved}")
    print(f"  moderate drops (5-20%):    {moderate}")

    if len(catastrophic) >= 1 and len(preserved) >= 1:
        verdict = (f"PASS: architecture-prior is LOCALIZABLE. "
                   f"Catastrophic without {catastrophic}; preserved without {preserved}. "
                   f"Identifies the irreducible architectural cores for the g125 prior.")
    elif len(catastrophic) >= 1 or len(moderate) >= 2:
        verdict = (f"PARTIAL: some features matter. Catastrophic: {catastrophic}, "
                   f"moderate: {moderate}. No single irreducible core but ranking exists.")
    else:
        verdict = (f"KILL: all ablations preserve capability within +/-5%. "
                   f"Architecture-prior is HOLISTIC across these features.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 138, "name": "arch_prior_decomposition",
        "config": {"train_steps": TRAIN_STEPS, "lr": LR, "batch": BATCH_SIZE,
                    "seed": SEED, "tokenizer": "EleutherAI/pythia-160m"},
        "arms": [a[0] for a in arms],
        "rows_per_arm": all_results,
        "summary": summary,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_138_arch_prior_decomposition.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
