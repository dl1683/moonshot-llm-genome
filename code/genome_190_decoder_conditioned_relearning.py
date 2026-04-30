"""
genome_190_decoder_conditioned_relearning.py

Phase 1: Freeze real trained Qwen3-0.6B decoder (28 layers). Attach GPT-2
tokenizer. Initialize embed/lm_head randomly. Train ONLY embed/lm_head on C4
for 2000 steps. Result: decoder-conditioned embeddings in GPT-2 tokenizer space.

Phase 2: Use Phase 1 relearned embeddings as anchor target for a fresh
GPT-2-tokenizer Qwen3-arch model (full 28L, not 8L shell). Compare against
scratch, static OT bridge (g188 result), and relearned-anchor-only.

Motivation: g183 (PPMI SVD) and g188 (static OT bridge) both HARM. Static
bridges lack decoder-format alignment. If decoder-conditioned embeddings work
as anchors, it proves the decoder shapes embedding geometry during training.

Codex design gate: codex_outputs/g190_decoder_conditioned_relearning_design_gate_20260430.md
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_165_annealed_donor as g165
import genome_167_kd_canonical as g167
import genome_181a_tokenizer_isolation as g181a

OUT_PATH = ROOT / "results" / "genome_190_decoder_conditioned_relearning.json"
PHASE1_CACHE = ROOT / "results" / "cache" / "genome_190_phase1_embed.pt"

SEEDS = [42, 7, 13]
SEQ_LEN = g165.SEQ_LEN
BATCH_SIZE = g165.BATCH_SIZE
PHASE1_STEPS = 2000
PHASE1_CHECKPOINTS = [250, 500, 1000, 1500, 2000]
PHASE2_STEPS = 5000
N_TRAIN_WINDOWS = 16384
N_C4_VAL_WINDOWS = 256
LR = g165.LR
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOG_EVERY = 100
EVAL_EVERY = 500

C4_TRAIN_SEED = g165.C4_TRAIN_SEED
C4_VAL_SEED = g165.C4_VAL_SEED

ANCHOR_LAMBDA = 0.01

DEVICE = g165.DEVICE
FORWARD_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


PHASE2_ARMS = [
    "scratch_ce",
    "relearned_init_anchor",
    "relearned_anchor_only",
]


def print_flush(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ---------- Phase 1: Relearn embed/lm_head on frozen decoder ----------

def run_phase1(tok_gpt2, train_ids, train_mask, val_ids, val_mask) -> dict[str, Any]:
    print_flush("\n=== Phase 1: Decoder-Conditioned Embedding Relearning ===")

    # Load trained Qwen3 decoder
    print_flush("  Loading trained Qwen3-0.6B...")
    from transformers import AutoModelForCausalLM, AutoConfig
    model = AutoModelForCausalLM.from_pretrained(
        g165._MODEL_ID, torch_dtype=torch.bfloat16,
    ).to(DEVICE)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    gpt2_vocab_size = len(tok_gpt2)
    hidden_size = model.config.hidden_size  # 1024

    print_flush(f"  Original embed: {model.model.embed_tokens.weight.shape}")
    print_flush(f"  Target GPT-2 vocab: {gpt2_vocab_size}, hidden: {hidden_size}")

    # Replace embed_tokens with GPT-2-sized random init
    old_embed_fro = model.model.embed_tokens.weight.detach().float().norm().item()
    new_embed = nn.Embedding(gpt2_vocab_size, hidden_size, device=DEVICE, dtype=torch.bfloat16)
    nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)
    # Scale to match trained embed Frobenius norm
    with torch.no_grad():
        fro_actual = new_embed.weight.float().norm().item()
        if fro_actual > 0:
            new_embed.weight.mul_(old_embed_fro / fro_actual)
    model.model.embed_tokens = new_embed
    model.model.embed_tokens.weight.requires_grad_(True)

    # Replace lm_head (tied weights)
    new_lm_head = nn.Linear(hidden_size, gpt2_vocab_size, bias=False, device=DEVICE, dtype=torch.bfloat16)
    new_lm_head.weight = model.model.embed_tokens.weight  # tie weights
    model.lm_head = new_lm_head

    # Update config for new vocab
    model.config.vocab_size = gpt2_vocab_size

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print_flush(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print_flush(f"  New embed Fro: {model.model.embed_tokens.weight.detach().float().norm().item():.1f} (target: {old_embed_fro:.1f})")

    # Optimizer: only trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(190)
    schedule = rng.integers(0, int(train_ids.shape[0]), size=(PHASE1_STEPS, BATCH_SIZE), dtype=np.int64)

    trajectory = []
    checkpoints = {}

    # Initial eval
    model.eval()
    with torch.no_grad():
        init_nll = g181a.evaluate_nll(model, val_ids, val_mask)
    trajectory.append({"step": 0, **init_nll})
    print_flush(f"  step=0 val_nll={init_nll['nll']:.4f}")

    t0 = time.time()
    model.train()
    prev_embed = model.model.embed_tokens.weight.detach().float().clone()

    for step in range(1, PHASE1_STEPS + 1):
        batch_indices = schedule[step - 1]
        ids = train_ids[torch.as_tensor(batch_indices, dtype=torch.long)].to(DEVICE)
        mask = train_mask[torch.as_tensor(batch_indices, dtype=torch.long)].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            loss = g167.causal_ce_loss(logits, ids, mask)

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at Phase 1 step {step}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP,
        )
        optimizer.step()

        if step % LOG_EVERY == 0:
            row = {"step": step, "ce_loss": float(loss.item()), "elapsed_s": time.time() - t0}
            if step % EVAL_EVERY == 0 or step == PHASE1_STEPS:
                model.eval()
                with torch.no_grad():
                    metrics = g181a.evaluate_nll(model, val_ids, val_mask)
                row.update(metrics)
                model.train()

                # Movement metric
                curr_embed = model.model.embed_tokens.weight.detach().float()
                movement = (curr_embed - prev_embed).norm().item() / curr_embed.norm().item()
                row["embed_movement"] = movement
                prev_embed = curr_embed.clone()

                print_flush(f"  step={step} ce={loss.item():.4f} val_nll={metrics['nll']:.4f} move={movement:.4f} ({time.time()-t0:.0f}s)")
            elif step % (LOG_EVERY * 5) == 0:
                print_flush(f"  step={step} ce={loss.item():.4f} ({time.time()-t0:.0f}s)")
            trajectory.append(row)

        if step in PHASE1_CHECKPOINTS:
            embed_snap = model.model.embed_tokens.weight.detach().float().cpu()
            checkpoints[str(step)] = {
                "fro_norm": float(embed_snap.norm().item()),
                "mean_row_norm": float(embed_snap.norm(dim=1).mean().item()),
            }

    # Save relearned embeddings
    relearned_embed = model.model.embed_tokens.weight.detach().float().cpu().numpy()
    PHASE1_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(relearned_embed), PHASE1_CACHE)

    final_nll = trajectory[-1].get("nll", None)
    if final_nll is None:
        model.eval()
        with torch.no_grad():
            final_metrics = g181a.evaluate_nll(model, val_ids, val_mask)
        final_nll = final_metrics["nll"]

    result = {
        "phase1_steps": PHASE1_STEPS,
        "final_nll": float(final_nll),
        "initial_nll": float(init_nll["nll"]),
        "nll_improvement": float(init_nll["nll"]) - float(final_nll),
        "relearned_embed_shape": list(relearned_embed.shape),
        "relearned_fro": float(np.linalg.norm(relearned_embed, "fro")),
        "trajectory": trajectory,
        "checkpoints": checkpoints,
        "wallclock_s": time.time() - t0,
    }

    del model, optimizer
    cleanup_cuda()

    print_flush(f"\n  Phase 1 done: NLL {init_nll['nll']:.4f} → {final_nll:.4f} ({time.time()-t0:.0f}s)")
    print_flush(f"  Relearned embed: {relearned_embed.shape}, Fro={np.linalg.norm(relearned_embed, 'fro'):.1f}")
    return result


# ---------- Phase 2: Use relearned embeddings as anchor ----------

def make_full_qwen3_gpt2(tok_gpt2, seed: int):
    """Create a full 28-layer Qwen3 model with GPT-2 vocab size."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    base_cfg = AutoConfig.from_pretrained(g165._MODEL_ID)

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=len(tok_gpt2),
        hidden_size=base_cfg.hidden_size,
        num_hidden_layers=base_cfg.num_hidden_layers,
        num_attention_heads=base_cfg.num_attention_heads,
        num_key_value_heads=base_cfg.num_key_value_heads,
        intermediate_size=base_cfg.intermediate_size,
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=base_cfg.rms_norm_eps,
        tie_word_embeddings=True,
        head_dim=getattr(base_cfg, "head_dim", 64),
        rope_theta=getattr(base_cfg, "rope_theta", 10000.0),
        use_cache=False,
        bos_token_id=tok_gpt2.bos_token_id if tok_gpt2.bos_token_id is not None else tok_gpt2.eos_token_id,
        eos_token_id=tok_gpt2.eos_token_id,
        pad_token_id=tok_gpt2.pad_token_id if tok_gpt2.pad_token_id is not None else tok_gpt2.eos_token_id,
    )
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"

    from transformers import Qwen3ForCausalLM
    model = Qwen3ForCausalLM(cfg)
    model.tie_weights()
    model.to(DEVICE)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def train_phase2_cell(
    arm_label: str,
    seed: int,
    relearned_embed: np.ndarray | None,
    use_init: bool,
    use_anchor: bool,
    tok_gpt2,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_full_qwen3_gpt2(tok_gpt2, seed)

    # Inject relearned embeddings as init if requested
    if use_init and relearned_embed is not None:
        with torch.no_grad():
            target = torch.from_numpy(relearned_embed).to(model.model.embed_tokens.weight.device,
                                                          dtype=model.model.embed_tokens.weight.dtype)
            model.model.embed_tokens.weight.copy_(target)
            if not model.config.tie_word_embeddings and hasattr(model, "lm_head"):
                model.lm_head.weight.copy_(target)

    # Build anchor pairs
    anchor_target = None
    actual_lambda = 0.0
    if use_anchor and relearned_embed is not None:
        anchor_target = torch.from_numpy(relearned_embed).to(DEVICE, dtype=torch.float32)
        actual_lambda = ANCHOR_LAMBDA

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
    )
    rng = np.random.default_rng(seed)
    schedule = rng.integers(0, int(train_ids.shape[0]), size=(PHASE2_STEPS, BATCH_SIZE), dtype=np.int64)

    trajectory = []
    initial_metrics = g181a.evaluate_nll(model, val_ids, val_mask)
    trajectory.append({"step": 0, **initial_metrics})
    print_flush(f"    {arm_label} seed={seed} lambda={actual_lambda:.6f} step=0 nll={initial_metrics['nll']:.4f}")

    t0 = time.time()
    model.train()
    for step in range(1, PHASE2_STEPS + 1):
        batch_indices = schedule[step - 1]
        ids = train_ids[torch.as_tensor(batch_indices, dtype=torch.long)].to(DEVICE)
        mask = train_mask[torch.as_tensor(batch_indices, dtype=torch.long)].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            ce_loss = g167.causal_ce_loss(logits, ids, mask)

        if not torch.isfinite(ce_loss):
            raise RuntimeError(f"non-finite CE loss at step {step} arm={arm_label} seed={seed}")

        ce_loss.backward()

        if anchor_target is not None and actual_lambda > 0.0:
            with torch.no_grad():
                coeff = 2.0 * actual_lambda
                param = model.model.embed_tokens.weight
                if param.grad is not None:
                    param.grad.add_(param.detach().to(anchor_target.dtype) - anchor_target, alpha=coeff)

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == PHASE2_STEPS:
            row = {"step": step, "ce_loss": float(ce_loss.item()), "elapsed_s": time.time() - t0}
            if step % EVAL_EVERY == 0 or step == PHASE2_STEPS:
                model.eval()
                with torch.no_grad():
                    row.update(g181a.evaluate_nll(model, val_ids, val_mask))
                model.train()
                print_flush(f"    {arm_label} seed={seed} step={step} nll={row['nll']:.4f} ({row['elapsed_s']:.0f}s)")
            elif step % (LOG_EVERY * 5) == 0:
                print_flush(f"    {arm_label} seed={seed} step={step} ce={row['ce_loss']:.4f} ({row['elapsed_s']:.0f}s)")
            trajectory.append(row)

    final_metrics = trajectory[-1]
    result = {
        "seed": seed, "arm_label": arm_label,
        "anchor_lambda": actual_lambda,
        "use_init": use_init, "use_anchor": use_anchor,
        "initial_metrics": initial_metrics,
        "final_nll": float(final_metrics["nll"]),
        "trajectory": trajectory,
        "wallclock_s": time.time() - t0,
    }
    del model, optimizer
    cleanup_cuda()
    return result


# ---------- Verdict ----------

def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("phase2_results", {})
    if not all(arm in results and len(results[arm]) >= len(SEEDS) for arm in PHASE2_ARMS):
        return {"status": "incomplete"}

    scratch_nlls = {str(s): float(results["scratch_ce"][str(s)]["final_nll"]) for s in SEEDS}
    relearned_nlls = {str(s): float(results["relearned_init_anchor"][str(s)]["final_nll"]) for s in SEEDS}

    gains = [scratch_nlls[str(s)] - relearned_nlls[str(s)] for s in SEEDS]
    mean_gain = float(np.mean(gains))
    seeds_positive = sum(1 for g in gains if g > 0)

    anchor_only_nlls = {str(s): float(results["relearned_anchor_only"][str(s)]["final_nll"]) for s in SEEDS}
    ao_gains = [scratch_nlls[str(s)] - anchor_only_nlls[str(s)] for s in SEEDS]
    ao_mean = float(np.mean(ao_gains))

    if mean_gain >= 0.257 and seeds_positive >= 3 and ao_mean >= 0.15:
        verdict = "STRONG_PASS"
    elif mean_gain >= 0.15 and seeds_positive >= 3:
        verdict = "PASS"
    elif mean_gain >= 0.05 and seeds_positive >= 2:
        verdict = "PARTIAL"
    elif mean_gain < 0.05 or seeds_positive < 2:
        verdict = "FAIL"
    else:
        verdict = "UNCLEAR"

    return {
        "status": "complete",
        "verdict": verdict,
        "init_anchor_mean_gain": mean_gain,
        "init_anchor_per_seed": gains,
        "init_anchor_seeds_positive": seeds_positive,
        "anchor_only_mean_gain": ao_mean,
        "anchor_only_per_seed": ao_gains,
    }


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase1-only", action="store_true")
    parser.add_argument("--phase2-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    smoke = args.smoke
    phase1_steps = 50 if smoke else PHASE1_STEPS
    phase2_steps = 50 if smoke else PHASE2_STEPS
    seeds = [42] if smoke else SEEDS

    print_flush(f"=== g190 Decoder-Conditioned Embedding Relearning ===")
    print_flush(f"  smoke={smoke}, phase1={phase1_steps}, phase2={phase2_steps}, seeds={seeds}")

    # Load GPT-2 tokenizer
    from transformers import AutoTokenizer
    tok_gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tok_gpt2.pad_token is None:
        tok_gpt2.pad_token = tok_gpt2.eos_token

    # Load training data with GPT-2 tokenizer
    print_flush("\n--- Loading GPT-2-tokenizer training data ---")
    train_ids, train_mask, _ = g167.load_c4_windows(
        tok_gpt2, split="train", seed=C4_TRAIN_SEED, n_windows=N_TRAIN_WINDOWS,
    )
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
    val_ids, val_mask, _ = g167.load_c4_windows(
        tok_gpt2, split="train", seed=C4_VAL_SEED, n_windows=N_C4_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    print_flush(f"  Train: {train_ids.shape}, Val: {val_ids.shape}")

    # Resume
    if not args.no_resume and OUT_PATH.exists():
        payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        payload = {
            "genome": 190,
            "name": "decoder_conditioned_relearning",
            "timestamp_utc_started": now_utc(),
            "config": {
                "phase1_steps": phase1_steps,
                "phase2_steps": phase2_steps,
                "seeds": seeds,
                "anchor_lambda": ANCHOR_LAMBDA,
            },
            "phase1": {},
            "phase2_results": {},
            "summary": {},
            "verdict": "INCOMPLETE",
        }

    t_start = time.time()

    def save():
        payload["timestamp_utc_last_write"] = now_utc()
        payload["elapsed_s"] = time.time() - t_start
        tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, OUT_PATH)

    # Phase 1
    if not args.phase2_only:
        if "final_nll" not in payload.get("phase1", {}):
            payload["phase1"] = run_phase1(tok_gpt2, train_ids, train_mask, val_ids, val_mask)
            save()
        else:
            print_flush(f"\n  Phase 1 already done (NLL={payload['phase1']['final_nll']:.4f}), skipping")

    if args.phase1_only:
        print_flush("\n  --phase1-only, exiting")
        save()
        return

    # Load relearned embeddings
    if PHASE1_CACHE.exists():
        relearned_embed = torch.load(PHASE1_CACHE, weights_only=True).numpy()
        print_flush(f"\n  Loaded relearned embed: {relearned_embed.shape}, Fro={np.linalg.norm(relearned_embed, 'fro'):.1f}")
    else:
        raise RuntimeError("Phase 1 cache not found — run Phase 1 first")

    # Phase 2
    print_flush("\n=== Phase 2: Anchor Training ===")
    for arm in PHASE2_ARMS:
        payload["phase2_results"].setdefault(arm, {})

    for arm_label in PHASE2_ARMS:
        for seed in seeds:
            key = str(seed)
            if key in payload["phase2_results"].get(arm_label, {}) and not args.no_resume:
                cell = payload["phase2_results"][arm_label][key]
                if isinstance(cell, dict) and "final_nll" in cell:
                    print_flush(f"\n  Skipping {arm_label}/seed={seed} (done)")
                    continue

            print_flush(f"\n  === {arm_label} seed={seed} ===")

            if arm_label == "scratch_ce":
                result = train_phase2_cell(
                    arm_label, seed, None, use_init=False, use_anchor=False,
                    tok_gpt2=tok_gpt2, train_ids=train_ids, train_mask=train_mask,
                    val_ids=val_ids, val_mask=val_mask,
                )
            elif arm_label == "relearned_init_anchor":
                result = train_phase2_cell(
                    arm_label, seed, relearned_embed, use_init=True, use_anchor=True,
                    tok_gpt2=tok_gpt2, train_ids=train_ids, train_mask=train_mask,
                    val_ids=val_ids, val_mask=val_mask,
                )
            elif arm_label == "relearned_anchor_only":
                result = train_phase2_cell(
                    arm_label, seed, relearned_embed, use_init=False, use_anchor=True,
                    tok_gpt2=tok_gpt2, train_ids=train_ids, train_mask=train_mask,
                    val_ids=val_ids, val_mask=val_mask,
                )

            payload["phase2_results"][arm_label][key] = result
            save()
            print_flush(f"  {arm_label} seed={seed} final_nll={result['final_nll']:.4f} ({result['wallclock_s']:.0f}s)")

    summary = compute_verdict(payload)
    payload["summary"] = summary
    payload["verdict"] = summary.get("verdict", "INCOMPLETE")
    save()

    print_flush(f"\n*** g190 VERDICT: {summary.get('verdict', '?')} ***")
    if "init_anchor_mean_gain" in summary:
        print_flush(f"  init+anchor gain={summary['init_anchor_mean_gain']:+.4f} nats")
        print_flush(f"  anchor-only gain={summary['anchor_only_mean_gain']:+.4f} nats")


if __name__ == "__main__":
    main()
