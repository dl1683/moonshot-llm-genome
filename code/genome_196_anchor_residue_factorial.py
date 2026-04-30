"""
genome_196_anchor_residue_factorial.py

Resolves A18 SEV-10 #2 (anchor dominance) and #3 (scaffold-vs-content).
Tests whether trained row directions leave a persistent basin residue after
the anchor tether is removed, or only help as active regularization.

10 arms x 3 seeds = 30 cells.
Gated on g195 (determines intervention surface: input/output/both/tied).
Prereg: research/prereg/genome_196_anchor_residue_factorial_2026-04-30.md
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_165_annealed_donor as g165
import genome_167_kd_canonical as g167
import genome_188_tokenizer_flow_bridge as g188
import genome_191_string_match_decomposition as g191

OUT_PATH = ROOT / "results" / "genome_196_anchor_residue_factorial.json"

SEEDS = [42, 7, 13]
TRAIN_STEPS = 5000
ANCHOR_LAMBDA = 0.01
LOG_EVERY = 100
EVAL_EVERY = 500
DEVICE = g165.DEVICE

SCAFFOLD_SEED_ORTHO = 19601
SCAFFOLD_SEED_COV = 19602

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

ARMS = [
    "scratch",
    "init_only",
    "anchor_only_full",
    "init_anchor_full",
    "cutoff_50",
    "cutoff_500",
    "cutoff_2000",
    "late_anchor_only_2000",
    "orthogonal_scaffold_full",
    "cov_scaffold_full",
]


def print_flush(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- Scaffold construction ----------

def build_orthogonal_scaffold(
    target: np.ndarray, matched_mask: np.ndarray,
) -> np.ndarray:
    """Rotate all matched rows by a fixed random orthogonal matrix Q.
    Preserves all pairwise cosines; destroys trained coordinate basis."""
    d = target.shape[1]
    rng = np.random.default_rng(SCAFFOLD_SEED_ORTHO)
    A = rng.standard_normal((d, d)).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    Q = Q.astype(np.float32)

    out = np.zeros_like(target)
    matched_ids = np.where(matched_mask)[0]
    out[matched_ids] = target[matched_ids] @ Q
    return out


def build_covariance_scaffold(
    target: np.ndarray, matched_mask: np.ndarray,
) -> np.ndarray:
    """Draw rows from N(mean, cov + eps*I) of matched target rows.
    Preserves second-order statistics; destroys token identity."""
    matched_ids = np.where(matched_mask)[0]
    T_m = target[matched_ids].astype(np.float64)
    mean = T_m.mean(axis=0)
    cov = np.cov(T_m, rowvar=False)
    eps = 1e-6
    cov += eps * np.eye(cov.shape[0])

    rng = np.random.default_rng(SCAFFOLD_SEED_COV)
    X_m = rng.multivariate_normal(mean, cov, size=len(matched_ids)).astype(np.float32)

    norms = np.linalg.norm(X_m, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X_m = X_m / norms

    target_norms = np.linalg.norm(target[matched_ids], axis=1)
    uniform_norm = float(target_norms.mean())
    X_m = X_m * uniform_norm

    target_fro = float(np.linalg.norm(target[matched_ids], "fro"))
    current_fro = float(np.linalg.norm(X_m, "fro"))
    if current_fro > 1e-8:
        X_m = X_m * (target_fro / current_fro)

    out = np.zeros_like(target)
    out[matched_ids] = X_m
    return out


# ---------- Anchor schedule ----------

def get_anchor_lambda(arm_label: str, step: int) -> float:
    """Return the anchor lambda for this arm at this step."""
    if arm_label in ("scratch", "init_only"):
        return 0.0
    elif arm_label in ("anchor_only_full", "init_anchor_full",
                       "orthogonal_scaffold_full", "cov_scaffold_full"):
        return ANCHOR_LAMBDA
    elif arm_label == "cutoff_50":
        return ANCHOR_LAMBDA if step <= 50 else 0.0
    elif arm_label == "cutoff_500":
        return ANCHOR_LAMBDA if step <= 500 else 0.0
    elif arm_label == "cutoff_2000":
        return ANCHOR_LAMBDA if step <= 2000 else 0.0
    elif arm_label == "late_anchor_only_2000":
        return ANCHOR_LAMBDA if step > 2000 else 0.0
    else:
        raise ValueError(f"Unknown arm: {arm_label}")


# ---------- Training cell ----------

def train_cell(
    arm_label: str,
    seed: int,
    tok_gpt2,
    embed_init: np.ndarray | None,
    lm_head_init: np.ndarray | None,
    anchor_target: np.ndarray | None,
    anchor_mask: np.ndarray | None,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    n_steps: int = TRAIN_STEPS,
    tied: bool = True,
    surface: str = "tied",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if tied:
        model = g188.make_gpt2_qwen3_model(tok_gpt2, seed)
    else:
        from transformers import Qwen3ForCausalLM
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        torch.manual_seed(seed)
        cfg = Qwen3Config(
            vocab_size=len(tok_gpt2),
            hidden_size=1024,
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=2816,
            max_position_embeddings=g188.SEQ_LEN + 64,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            head_dim=64,
            rope_theta=10000.0,
            use_cache=False,
        )
        model = Qwen3ForCausalLM(cfg).to(device=DEVICE)
        model.config.pad_token_id = tok_gpt2.pad_token_id

    if embed_init is not None:
        emb_t = torch.from_numpy(embed_init).to(
            model.model.embed_tokens.weight.device,
            dtype=model.model.embed_tokens.weight.dtype,
        )
        with torch.no_grad():
            if anchor_mask is not None:
                mask_t = torch.from_numpy(anchor_mask).to(emb_t.device)
                model.model.embed_tokens.weight[mask_t] = emb_t[mask_t]
            else:
                model.model.embed_tokens.weight.copy_(emb_t)

    if lm_head_init is not None and not tied:
        head_t = torch.from_numpy(lm_head_init).to(
            model.lm_head.weight.device,
            dtype=model.lm_head.weight.dtype,
        )
        with torch.no_grad():
            if anchor_mask is not None:
                mask_t = torch.from_numpy(anchor_mask).to(head_t.device)
                model.lm_head.weight[mask_t] = head_t[mask_t]
            else:
                model.lm_head.weight.copy_(head_t)

    embed_anchor_t = None
    lm_head_anchor_t = None
    row_mask_t = None

    if anchor_target is not None and anchor_mask is not None:
        row_mask_t = torch.from_numpy(
            anchor_mask.astype(np.float32)
        ).to(DEVICE).unsqueeze(1)

        if tied or surface in ("tied", "input", "both"):
            embed_anchor_t = torch.from_numpy(anchor_target).to(
                DEVICE, dtype=torch.float32,
            )
        if not tied and surface in ("output", "both"):
            lm_head_anchor_t = torch.from_numpy(anchor_target).to(
                DEVICE, dtype=torch.float32,
            )

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=g188.LR, betas=g188.BETAS, weight_decay=g188.WEIGHT_DECAY,
    )

    n_train = train_ids.shape[0]
    trajectory = {}
    t0 = time.time()
    cutoff_eval_steps = {50, 500, 2000}

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_train, (g188.BATCH_SIZE,))
        batch_ids = train_ids[idx].to(DEVICE)
        batch_mask = train_mask[idx].to(DEVICE)

        with torch.amp.autocast("cuda", dtype=g188.FORWARD_DTYPE):
            out = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
            loss = out.loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step} arm={arm_label} seed={seed}")

        optimizer.zero_grad()
        loss.backward()

        lam = get_anchor_lambda(arm_label, step)
        coeff = 2.0 * lam
        if coeff > 0.0:
            with torch.no_grad():
                if embed_anchor_t is not None:
                    param = model.model.embed_tokens.weight
                    if param.grad is not None:
                        grad_add = (param.detach().to(embed_anchor_t.dtype) - embed_anchor_t) * coeff
                        if row_mask_t is not None:
                            grad_add = grad_add * row_mask_t
                        param.grad.add_(grad_add)

                if lm_head_anchor_t is not None:
                    param = model.lm_head.weight
                    if param.grad is not None:
                        grad_add = (param.detach().to(lm_head_anchor_t.dtype) - lm_head_anchor_t) * coeff
                        if row_mask_t is not None:
                            grad_add = grad_add * row_mask_t
                        param.grad.add_(grad_add)

        torch.nn.utils.clip_grad_norm_(model.parameters(), g188.GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            print_flush(f"    step {step}/{n_steps} loss={loss.item():.4f}")

        do_eval = (step % EVAL_EVERY == 0 or step == n_steps
                   or step in cutoff_eval_steps)
        if do_eval:
            model.eval()
            with torch.no_grad():
                val_nll = g188._eval_nll(model, val_ids, val_mask)
            trajectory[str(step)] = float(val_nll)
            if step % EVAL_EVERY == 0:
                print_flush(f"    eval step={step} val_nll={val_nll:.4f}")
            model.train()

    model.eval()
    with torch.no_grad():
        final_nll = g188._eval_nll(model, val_ids, val_mask)

    result = {
        "arm_label": arm_label,
        "seed": seed,
        "tied": tied,
        "surface": surface,
        "final_val_nll": float(final_nll),
        "trajectory": trajectory,
        "wallclock_s": time.time() - t0,
    }
    del model, optimizer
    cleanup_cuda()
    return result


# ---------- Verdict ----------

def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", {})
    if not all(arm in results and len(results[arm]) >= len(SEEDS) for arm in ARMS):
        return {"status": "incomplete"}

    scratch_nlls = {str(s): float(results["scratch"][str(s)]["final_val_nll"]) for s in SEEDS}

    def arm_stats(arm_name):
        nlls = {str(s): float(results[arm_name][str(s)]["final_val_nll"]) for s in SEEDS}
        gaps = [scratch_nlls[str(s)] - nlls[str(s)] for s in SEEDS]
        return float(np.mean(gaps)), gaps

    init_mean, init_gaps = arm_stats("init_only")
    anchor_full_mean, anchor_full_gaps = arm_stats("anchor_only_full")
    init_anchor_mean, init_anchor_gaps = arm_stats("init_anchor_full")
    cut50_mean, cut50_gaps = arm_stats("cutoff_50")
    cut500_mean, cut500_gaps = arm_stats("cutoff_500")
    cut2000_mean, cut2000_gaps = arm_stats("cutoff_2000")
    late_mean, late_gaps = arm_stats("late_anchor_only_2000")
    ortho_mean, ortho_gaps = arm_stats("orthogonal_scaffold_full")
    cov_mean, cov_gaps = arm_stats("cov_scaffold_full")

    replication_gate = (
        (anchor_full_mean >= 0.30 and all(g > 0 for g in anchor_full_gaps))
        or (init_anchor_mean >= 0.30 and all(g > 0 for g in init_anchor_gaps))
    )

    control_max_gain = max(ortho_mean, cov_mean)

    residue_fraction_2000 = (
        cut2000_mean / anchor_full_mean if anchor_full_mean > 0.01 else 0.0
    )

    scaffold_alt = (
        control_max_gain >= 0.20
        or (anchor_full_mean > 0.01 and control_max_gain >= 0.50 * anchor_full_mean)
    )

    if not replication_gate:
        verdict = "FAIL_REPLICATION"
    elif (
        cut2000_mean >= 0.20
        and all(g > 0 for g in cut2000_gaps)
        and residue_fraction_2000 >= 0.45
        and cut2000_mean >= init_mean + 0.10
        and control_max_gain <= 0.15
        and anchor_full_mean - control_max_gain >= 0.20
    ):
        verdict = "PASS_RESIDUE"
    elif scaffold_alt:
        verdict = "PASS_SCAFFOLD_ALT"
    elif (
        cut2000_mean >= 0.12
        and sum(1 for g in cut2000_gaps if g > 0) >= 2
        and residue_fraction_2000 >= 0.25
        and not scaffold_alt
    ):
        verdict = "PASS_PARTIAL_RESIDUE"
    elif (
        cut500_mean < 0.12
        and cut2000_mean < 0.12
        and residue_fraction_2000 < 0.25
        and late_mean >= 0.20
        and (anchor_full_mean < 0.01 or late_mean >= 0.50 * anchor_full_mean)
    ):
        verdict = "PASS_REGULARIZATION"
    elif (
        cut50_mean < 0.10
        and cut500_mean < 0.15
        and cut2000_mean >= 0.20
        and all(g > 0 for g in cut2000_gaps)
    ):
        verdict = "PASS_EARLY_WINDOW"
    else:
        if not replication_gate and max(init_mean, anchor_full_mean, cut2000_mean) < 0.10:
            verdict = "FAIL_REPLICATION"
        elif late_mean < 0.10 and cut2000_mean < 0.10:
            verdict = "FAIL_TIMING_AMBIGUOUS"
        elif any(g <= 0 for g in cut2000_gaps) and any(g > 0 for g in cut2000_gaps):
            verdict = "FAIL_NOISY"
        else:
            verdict = "FAIL"

    return {
        "status": "complete",
        "verdict": verdict,
        "replication_gate_passed": replication_gate,
        "init_only_mean_gain": init_mean,
        "init_only_per_seed": init_gaps,
        "anchor_only_full_mean_gain": anchor_full_mean,
        "anchor_only_full_per_seed": anchor_full_gaps,
        "init_anchor_full_mean_gain": init_anchor_mean,
        "init_anchor_full_per_seed": init_anchor_gaps,
        "cutoff_50_mean_gain": cut50_mean,
        "cutoff_50_per_seed": cut50_gaps,
        "cutoff_500_mean_gain": cut500_mean,
        "cutoff_500_per_seed": cut500_gaps,
        "cutoff_2000_mean_gain": cut2000_mean,
        "cutoff_2000_per_seed": cut2000_gaps,
        "residue_fraction_2000": residue_fraction_2000,
        "late_anchor_only_2000_mean_gain": late_mean,
        "late_anchor_only_2000_per_seed": late_gaps,
        "orthogonal_scaffold_full_mean_gain": ortho_mean,
        "orthogonal_scaffold_full_per_seed": ortho_gaps,
        "cov_scaffold_full_mean_gain": cov_mean,
        "cov_scaffold_full_per_seed": cov_gaps,
        "control_max_gain": control_max_gain,
        "scaffold_alt": scaffold_alt,
    }


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--surface", type=str, default=None,
                        help="Override g195 surface: input/output/both/tied")
    parser.add_argument("--tied", action="store_true", default=False,
                        help="Force tied fallback branch")
    args = parser.parse_args()

    smoke = args.smoke
    n_steps = 50 if smoke else TRAIN_STEPS
    seeds = [42] if smoke else SEEDS
    run_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_smoke.json") if smoke else OUT_PATH

    # Determine surface from g195 result or CLI override
    g195_path = ROOT / "results" / "genome_195_untied_input_output_factorial.json"
    surface = args.surface
    use_tied = args.tied

    if surface is None and not use_tied:
        if g195_path.exists():
            g195 = json.loads(g195_path.read_text(encoding="utf-8"))
            g195_verdict = g195.get("verdict", "INCOMPLETE")
            print_flush(f"  g195 verdict: {g195_verdict}")

            verdict_to_surface = {
                "PASS_INPUT": "input",
                "PASS_INPUT_DOMINANT": "input",
                "PASS_OUTPUT": "output",
                "PASS_OUTPUT_DOMINANT": "output",
                "PASS_BOTH_NEEDED": "both",
            }
            if g195_verdict in verdict_to_surface:
                surface = verdict_to_surface[g195_verdict]
                use_tied = False
            elif g195_verdict in ("AMBIGUOUS_POSITIVE", "FAIL"):
                tied_gain = g195.get("summary", {}).get("tied_mean_gain", 0)
                if tied_gain >= 0.30:
                    print_flush(f"  g195 ambiguous but tied_reference gain={tied_gain:+.3f} >= 0.30 => tied fallback")
                    use_tied = True
                    surface = "tied"
                else:
                    print_flush(f"  g195 FAIL and tied_reference gain={tied_gain:+.3f} < 0.30 => ABORT g196")
                    print_flush("  Cannot launch g196 without g195 signal. Exiting.")
                    return
            else:
                print_flush(f"  g195 verdict={g195_verdict} not recognized. Waiting for g195 to complete.")
                return
        else:
            print_flush("  g195 result not found. Cannot determine surface. Use --surface or --tied.")
            return

    if surface is None:
        surface = "tied"
    if use_tied:
        surface = "tied"

    print_flush(f"=== g196 Anchor-Residue Factorial ===")
    print_flush(f"  smoke={smoke}, steps={n_steps}, seeds={seeds}")
    print_flush(f"  surface={surface}, tied={use_tied}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok_qwen = AutoTokenizer.from_pretrained(g188.QWEN_MODEL_ID)
    tok_gpt2 = AutoTokenizer.from_pretrained(g188.GPT2_MODEL_ID)
    if tok_gpt2.pad_token is None:
        tok_gpt2.pad_token = tok_gpt2.eos_token

    print_flush("\n--- Loading data ---")
    train_ids, train_mask, _ = g167.load_c4_windows(
        tok_gpt2, split="train", seed=g188.C4_TRAIN_SEED, n_windows=g188.N_TRAIN_WINDOWS,
    )
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
    val_ids, val_mask, _ = g167.load_c4_windows(
        tok_gpt2, split="train", seed=g188.C4_VAL_SEED, n_windows=g188.N_C4_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    print_flush(f"  Train: {train_ids.shape}, Val: {val_ids.shape}")

    print_flush("\n--- Loading Qwen3 trained embeddings ---")
    qwen_model = AutoModelForCausalLM.from_pretrained(g188.QWEN_MODEL_ID, torch_dtype=torch.float32)
    trained_embed = qwen_model.model.embed_tokens.weight.detach().cpu().numpy()
    trained_fro = float(np.linalg.norm(trained_embed, "fro"))
    del qwen_model
    cleanup_cuda()
    print_flush(f"  Trained embed: {trained_embed.shape}, Fro={trained_fro:.1f}")

    print_flush("\n--- Building string-match base embeddings ---")
    gpt2_vocab = len(tok_gpt2)
    embed_dim = trained_embed.shape[1]
    full_embed, matched_mask = g191.build_string_match_with_mask(
        tok_qwen, tok_gpt2, trained_embed, gpt2_vocab, embed_dim,
    )
    full_embed = g188.normalize_to_fro_norm(full_embed, trained_fro)

    n_matched = int(matched_mask.sum())
    matched_fro = float(np.linalg.norm(full_embed[matched_mask], "fro"))
    print_flush(f"  Matched: {n_matched}, matched_fro={matched_fro:.2f}")

    # g194 direction-only target: correct directions, uniform norm, Fro-normalized
    from genome_194_scalar_direction_factorial import decompose_rows, build_correct_dir_uniform_norm
    norms, unit_dirs = decompose_rows(full_embed, matched_mask)
    primary_target = build_correct_dir_uniform_norm(unit_dirs, norms, matched_mask)
    primary_target = g188.normalize_to_fro_norm(primary_target, matched_fro)
    print_flush(f"  Primary target (cd_un) Fro={float(np.linalg.norm(primary_target[matched_mask], 'fro')):.2f}")

    # Scaffold controls
    print_flush("\n--- Building scaffold controls ---")
    ortho_target = build_orthogonal_scaffold(primary_target, matched_mask)
    ortho_target = g188.normalize_to_fro_norm(ortho_target, matched_fro)
    print_flush(f"  Orthogonal scaffold Fro={float(np.linalg.norm(ortho_target[matched_mask], 'fro')):.2f}")

    cov_target = build_covariance_scaffold(primary_target, matched_mask)
    print_flush(f"  Covariance scaffold Fro={float(np.linalg.norm(cov_target[matched_mask], 'fro')):.2f}")

    # Arm configurations
    # embed_init / lm_head_init: what to inject at step 0
    # anchor_target: the target matrix for the anchor loss term
    # anchor_mask: which rows to apply anchor to
    # For cutoff/late arms, the schedule is handled dynamically by get_anchor_lambda()
    def init_for_surface(target):
        if surface == "input":
            return target, None
        elif surface == "output":
            return None, target
        elif surface == "both":
            return target, target
        else:  # tied
            return target, None

    primary_embed_init, primary_lm_head_init = init_for_surface(primary_target)

    arm_configs = {
        "scratch": {
            "embed_init": None, "lm_head_init": None,
            "anchor_target": None, "anchor_mask": None,
        },
        "init_only": {
            "embed_init": primary_embed_init, "lm_head_init": primary_lm_head_init,
            "anchor_target": None, "anchor_mask": matched_mask,
        },
        "anchor_only_full": {
            "embed_init": None, "lm_head_init": None,
            "anchor_target": primary_target, "anchor_mask": matched_mask,
        },
        "init_anchor_full": {
            "embed_init": primary_embed_init, "lm_head_init": primary_lm_head_init,
            "anchor_target": primary_target, "anchor_mask": matched_mask,
        },
        "cutoff_50": {
            "embed_init": None, "lm_head_init": None,
            "anchor_target": primary_target, "anchor_mask": matched_mask,
        },
        "cutoff_500": {
            "embed_init": None, "lm_head_init": None,
            "anchor_target": primary_target, "anchor_mask": matched_mask,
        },
        "cutoff_2000": {
            "embed_init": None, "lm_head_init": None,
            "anchor_target": primary_target, "anchor_mask": matched_mask,
        },
        "late_anchor_only_2000": {
            "embed_init": None, "lm_head_init": None,
            "anchor_target": primary_target, "anchor_mask": matched_mask,
        },
        "orthogonal_scaffold_full": {
            "embed_init": None, "lm_head_init": None,
            "anchor_target": ortho_target, "anchor_mask": matched_mask,
        },
        "cov_scaffold_full": {
            "embed_init": None, "lm_head_init": None,
            "anchor_target": cov_target, "anchor_mask": matched_mask,
        },
    }

    if not args.no_resume and run_out_path.exists():
        payload = json.loads(run_out_path.read_text(encoding="utf-8"))
        prev_cfg = payload.get("config", {})
        if (prev_cfg.get("surface") != surface
                or prev_cfg.get("tied") != use_tied
                or prev_cfg.get("train_steps") != n_steps
                or prev_cfg.get("seeds") != seeds):
            print_flush(f"  WARNING: resume config mismatch (prev surface={prev_cfg.get('surface')}, "
                        f"tied={prev_cfg.get('tied')}, steps={prev_cfg.get('train_steps')}). "
                        f"Starting fresh.")
            payload = None
    else:
        payload = None

    if payload is None:
        payload = {
            "genome": 196,
            "name": "anchor_residue_factorial",
            "timestamp_utc_started": now_utc(),
            "config": {
                "train_steps": n_steps,
                "seeds": seeds,
                "anchor_lambda": ANCHOR_LAMBDA,
                "n_matched": n_matched,
                "trained_fro": trained_fro,
                "matched_fro": matched_fro,
                "surface": surface,
                "tied": use_tied,
                "scaffold_seed_ortho": SCAFFOLD_SEED_ORTHO,
                "scaffold_seed_cov": SCAFFOLD_SEED_COV,
            },
            "results": {},
            "summary": {},
            "verdict": "INCOMPLETE",
        }

    t_start = time.time()

    def save():
        payload["timestamp_utc_last_write"] = now_utc()
        payload["elapsed_s"] = time.time() - t_start
        tmp = run_out_path.with_suffix(run_out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
        os.replace(tmp, run_out_path)

    for arm_label in ARMS:
        payload["results"].setdefault(arm_label, {})
        cfg = arm_configs[arm_label]

        for seed in seeds:
            key = str(seed)
            if key in payload["results"][arm_label] and not args.no_resume:
                cell = payload["results"][arm_label][key]
                if isinstance(cell, dict) and "final_val_nll" in cell:
                    print_flush(f"\n  Skipping {arm_label}/seed={seed} (done)")
                    continue

            print_flush(f"\n  === {arm_label} seed={seed} ===")
            result = train_cell(
                arm_label=arm_label,
                seed=seed,
                tok_gpt2=tok_gpt2,
                embed_init=cfg["embed_init"],
                lm_head_init=cfg["lm_head_init"],
                anchor_target=cfg["anchor_target"],
                anchor_mask=cfg["anchor_mask"],
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
                n_steps=n_steps,
                tied=use_tied,
                surface=surface,
            )
            payload["results"][arm_label][key] = result
            save()
            print_flush(f"  {arm_label} seed={seed} nll={result['final_val_nll']:.4f} ({result['wallclock_s']:.0f}s)")

    summary = compute_verdict(payload)
    payload["summary"] = summary
    payload["verdict"] = summary.get("verdict", "INCOMPLETE")
    save()

    print_flush(f"\n*** g196 VERDICT: {summary.get('verdict', '?')} ***")
    for key, val in summary.items():
        if key.endswith("_mean_gain"):
            print_flush(f"  {key}: {val:+.4f}")
    if "residue_fraction_2000" in summary:
        print_flush(f"  residue_fraction_2000: {summary['residue_fraction_2000']:.3f}")
    print_flush(f"  replication_gate_passed: {summary.get('replication_gate_passed')}")
    print_flush(f"  scaffold_alt: {summary.get('scaffold_alt')}")


if __name__ == "__main__":
    main()
