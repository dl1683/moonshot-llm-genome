"""
genome_186_kd_dose_response.py

KD Dose-Response Delta Geometry experiment.

Pre-reg: research/prereg/genome_186_dose_response_2026-04-29.md (LOCKED)

60 cells = 2 architectures (Qwen3-arch, GPT-2-arch) x 5 KD doses x 6 seeds.
Primary test: seed-matched delta_geometry predicts delta_NLL on held-out seeds.
KD loss form: loss = CE(C4) + alpha * CE(teacher_text)

Outputs:
  - results/genome_186_kd_dose_response.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_182_triage_arena as g182

OUT_PATH = ROOT / "results" / "genome_186_kd_dose_response.json"

SEEDS = list(range(6))
KD_ALPHAS = [0.0, 0.3, 0.7, 1.0, 2.0]
ARCHS = ["qwen3", "gpt2"]
DOSE_TRAIN_STEPS = 1200  # prereg: ~90s/cell, shortened from g182's 3600

print_flush = g182.print_flush
now_utc = g182.now_utc
set_seed = g182.set_seed
make_model = g182.make_model
evaluate_nll = g182.evaluate_nll
autocast_context = g182.autocast_context
causal_ce_loss = g182.causal_ce_loss
warmup_lr = g182.warmup_lr
param_count = g182.param_count
DEVICE = g182.DEVICE


def train_one_cell_dose(
    arch: str,
    seed: int,
    kd_alpha: float,
    pools: dict,
    teacher_pools: dict | None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train one cell with additive KD loss: CE(C4) + alpha * CE(teacher)."""
    train_steps = 20 if smoke else DOSE_TRAIN_STEPS
    feature_step = max(1, int(math.ceil(0.03 * train_steps)))

    arm_label = f"alpha_{kd_alpha:.1f}"
    cell_id = f"{arch}_{arm_label}_s{seed}"
    print_flush(f"\n=== Cell {cell_id} ===")
    t_cell = time.time()

    set_seed(seed)
    model = make_model(arch, seed)
    counts = param_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=g182.LR, betas=g182.BETAS,
        weight_decay=g182.WEIGHT_DECAY,
    )

    c4_ids = pools["train_ids"]
    c4_mask = pools["train_mask"]
    n_c4 = c4_ids.shape[0]

    has_teacher = teacher_pools is not None and kd_alpha > 0
    if has_teacher:
        t_ids = teacher_pools["train_ids"]
        t_mask = teacher_pools["train_mask"]
        n_teacher = t_ids.shape[0]

    rng = np.random.default_rng(seed)
    c4_schedule = rng.integers(0, n_c4, size=(train_steps, g182.TRAIN_BATCH_SIZE), dtype=np.int64)
    if has_teacher:
        t_schedule = rng.integers(0, n_teacher, size=(train_steps, g182.TRAIN_BATCH_SIZE), dtype=np.int64)

    initial_metrics = evaluate_nll(model, pools["val_ids"], pools["val_mask"])
    print_flush(f"    params={counts['n_total_params']/1e6:.2f}M alpha={kd_alpha} "
                f"step=0 c4_nll={initial_metrics['nll']:.4f}")

    train_log = []
    trajectory_losses = {}
    features = None
    early_loss = float("nan")

    model.train()
    t_train = time.time()

    for step in range(1, train_steps + 1):
        current_lr = warmup_lr(step - 1)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        c4_idx = torch.as_tensor(c4_schedule[step - 1], dtype=torch.long)
        ids_c4 = c4_ids[c4_idx].to(DEVICE)
        mask_c4 = c4_mask[c4_idx].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits_c4 = model(input_ids=ids_c4, attention_mask=mask_c4, use_cache=False).logits
            ce_c4 = causal_ce_loss(logits_c4, ids_c4, mask_c4)

        loss = ce_c4

        if has_teacher:
            t_idx = torch.as_tensor(t_schedule[step - 1], dtype=torch.long)
            ids_t = t_ids[t_idx].to(DEVICE)
            mask_t = t_mask[t_idx].to(DEVICE)
            with autocast_context():
                logits_t = model(input_ids=ids_t, attention_mask=mask_t, use_cache=False).logits
                ce_t = causal_ce_loss(logits_t, ids_t, mask_t)
            loss = ce_c4 + kd_alpha * ce_t

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step} cell={cell_id}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), g182.GRAD_CLIP, error_if_nonfinite=False)
        optimizer.step()

        if step in g182.TRAJECTORY_STEPS:
            model.eval()
            with torch.no_grad():
                traj_metrics = evaluate_nll(model, pools["val_ids"], pools["val_mask"])
            trajectory_losses[str(step)] = traj_metrics["nll"]
            model.train()

        if step == feature_step:
            model.eval()
            with torch.no_grad():
                feat_metrics = evaluate_nll(model, pools["val_ids"], pools["val_mask"])
            early_loss = feat_metrics["nll"]
            trajectory_losses[str(step)] = early_loss
            try:
                probe = dict(pools["probe_batch"])
                probe["early_loss"] = early_loss
                features = g182.extract_features_for_cell(
                    model, probe, arch, include_qwen_ref=False,
                )
            except Exception as e:
                print_flush(f"    feature extraction failed at step {step}: {e}")
                features = {}
            model.train()

        if step % g182.LOG_EVERY == 0 or step == train_steps:
            elapsed = time.time() - t_train
            print_flush(f"    step={step}/{train_steps} loss={float(loss):.4f} "
                        f"time={elapsed:.1f}s")

    model.eval()
    with torch.no_grad():
        final_metrics = evaluate_nll(model, pools["val_ids"], pools["val_mask"])

    wallclock = time.time() - t_cell
    print_flush(f"    DONE: c4_nll={final_metrics['nll']:.4f} time={wallclock:.1f}s")

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "cell_id": cell_id,
        "arch": arch,
        "arm": arm_label,
        "kd_alpha": kd_alpha,
        "seed": seed,
        "train_steps": train_steps,
        "feature_step": feature_step,
        "n_total_params": counts["n_total_params"],
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "final_nll": final_metrics["nll"],
        "early_loss": early_loss,
        "trajectory_losses": trajectory_losses,
        "features": features or {},
        "wallclock_s": wallclock,
    }


def compute_dose_labels(cells: list[dict]) -> list[dict]:
    """Compute seed-matched delta labels: delta_NLL = scratch_nll - dose_nll."""
    labeled = []
    for arch in ARCHS:
        scratch_by_seed = {}
        for c in cells:
            if c["arch"] == arch and c["kd_alpha"] == 0.0:
                scratch_by_seed[c["seed"]] = c
        for c in cells:
            if c["arch"] == arch and c["kd_alpha"] > 0:
                sc = scratch_by_seed.get(c["seed"])
                if sc is None:
                    continue
                delta_nll = sc["final_nll"] - c["final_nll"]
                labeled.append({**c, "label": delta_nll, "scratch_nll": sc["final_nll"]})
    return labeled


TELEMETRY_FEAT_NAMES = [
    "grad_norm_mean", "grad_norm_var", "gradient_noise_scale",
    "hidden_norm_early_late_ratio", "hidden_var_early_late_ratio",
    "norm_param_early_late_ratio", "curvature_top_eigen_proxy",
]
SHESHA_FEAT_NAMES = [
    "shesha_anchor_stability", "shesha_feature_split", "shesha_sample_split",
]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def _cv_ridge_baseline(bX, dy, delta_meta, seed_folds, min_train=2):
    """Run cross-validated Ridge on feature matrix bX, return (preds, mask, r2)."""
    from sklearn.linear_model import RidgeCV
    b_preds = np.zeros(len(dy))
    b_mask = np.zeros(len(dy), dtype=bool)
    for fold_seeds in seed_folds:
        train_m = np.array([m["seed"] not in fold_seeds for m in delta_meta])
        test_m = ~train_m
        if train_m.sum() < max(min_train, bX.shape[1] + 1) or test_m.sum() < 1:
            continue
        mu, sd = bX[train_m].mean(0), bX[train_m].std(0)
        sd[sd < 1e-12] = 1.0
        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit((bX[train_m] - mu) / sd, dy[train_m])
        b_preds[test_m] = ridge.predict((bX[test_m] - mu) / sd)
        b_mask |= test_m
    if b_mask.sum() > 2:
        ss = float(np.sum((dy[b_mask] - b_preds[b_mask]) ** 2))
        st = float(np.sum((dy[b_mask] - dy[b_mask].mean()) ** 2))
        r2 = 1 - ss / st if st > 1e-12 else float("nan")
    else:
        r2 = float("nan")
    return b_preds, b_mask, r2


def _safe_delta(c_feats, sc_feats, names):
    """Compute feature delta, returning None if any feature is missing/NaN."""
    dx = []
    for fn in names:
        v_kd = float(c_feats.get(fn, float("nan")))
        v_sc = float(sc_feats.get(fn, float("nan")))
        if not (math.isfinite(v_kd) and math.isfinite(v_sc)):
            return None
        dx.append(v_kd - v_sc)
    return dx


def _safe_delta_partial(c_feats, sc_feats, names):
    """Compute feature delta, using 0.0 for features where either value is NaN."""
    dx = []
    for fn in names:
        v_kd = float(c_feats.get(fn, float("nan")))
        v_sc = float(sc_feats.get(fn, float("nan")))
        if math.isfinite(v_kd) and math.isfinite(v_sc):
            dx.append(v_kd - v_sc)
        else:
            dx.append(0.0)
    return dx


def _filter_available_features(cells, names):
    """Return subset of feature names that are finite in ALL cells (no imputation needed)."""
    counts = {fn: 0 for fn in names}
    for c in cells:
        feats = c.get("features") or {}
        for fn in names:
            v = feats.get(fn)
            if v is not None and math.isfinite(float(v)):
                counts[fn] += 1
    n = len(cells)
    return [fn for fn in names if counts[fn] == n]


def pairwise_dose_analysis(labeled: list[dict], all_cells: list[dict]) -> dict[str, Any]:
    """Primary analysis: seed-matched delta geometry predicts delta NLL."""
    from sklearn.linear_model import RidgeCV

    feat_names = list(g182.MANIFOLD_ONLY_FEATURE_NAMES)
    results = {}

    scratch_by = {}
    for c in all_cells:
        if c["kd_alpha"] == 0.0:
            scratch_by[(c["arch"], c["seed"])] = c

    # Pre-filter feature lists to only features with finite values in enough cells
    avail_tel = _filter_available_features(all_cells, TELEMETRY_FEAT_NAMES)
    avail_she = _filter_available_features(all_cells, SHESHA_FEAT_NAMES)
    dropped_tel = set(TELEMETRY_FEAT_NAMES) - set(avail_tel)
    dropped_she = set(SHESHA_FEAT_NAMES) - set(avail_she)
    if dropped_tel:
        print_flush(f"    Dropped unavailable telemetry features: {sorted(dropped_tel)}")
    if dropped_she:
        print_flush(f"    Dropped unavailable shesha features: {sorted(dropped_she)}")
    results["avail_telemetry_features"] = avail_tel
    results["avail_shesha_features"] = avail_she

    # Build aligned delta arrays for geometry, telemetry, shesha, early_loss
    delta_X, delta_y, delta_meta = [], [], []
    delta_tel, delta_shesha, delta_el = [], [], []
    skipped_feats = 0

    for c in labeled:
        sc = scratch_by.get((c["arch"], c["seed"]))
        if sc is None:
            continue
        c_feats = c.get("features") or {}
        sc_feats = sc.get("features") or {}
        if not c_feats or not sc_feats:
            skipped_feats += 1
            continue
        dx = _safe_delta(c_feats, sc_feats, feat_names)
        if dx is None:
            skipped_feats += 1
            continue
        delta_X.append(dx)
        delta_y.append(c["label"])
        delta_meta.append({"arch": c["arch"], "seed": c["seed"], "alpha": c["kd_alpha"]})

        # Telemetry deltas -- partial OK, zero for missing individual features
        delta_tel.append(_safe_delta_partial(c_feats, sc_feats, avail_tel) if avail_tel else [])

        # Shesha deltas -- partial OK
        delta_shesha.append(_safe_delta_partial(c_feats, sc_feats, avail_she) if avail_she else [])

        # Early loss delta (aligned)
        el_kd = float(c.get("early_loss", float("nan")))
        el_sc = float(sc.get("early_loss", float("nan")))
        if math.isfinite(el_kd) and math.isfinite(el_sc):
            delta_el.append(el_kd - el_sc)
        else:
            delta_el.append(0.0)

    if skipped_feats:
        print_flush(f"    WARNING: skipped {skipped_feats} rows with missing/nan features")

    # FIX #2: prereg requires 48 delta rows
    if len(delta_X) < 48:
        print_flush(f"    CRITICAL: only {len(delta_X)}/48 delta rows -- below prereg minimum")
        if len(delta_X) < 10:
            return {"error": f"too few delta pairs: {len(delta_X)}", "n_pairs": len(delta_X)}
    results["n_expected_pairs"] = 48
    results["rows_below_prereg_minimum"] = len(delta_X) < 48

    dX = np.array(delta_X)
    dy = np.array(delta_y)
    del_X = np.array(delta_el).reshape(-1, 1)
    tel_X = np.array(delta_tel)
    she_X = np.array(delta_shesha)

    results["n_pairs"] = len(delta_X)
    results["label_std"] = float(np.std(dy))
    results["label_range"] = [float(np.min(dy)), float(np.max(dy))]

    # --- Primary: leave-two-seeds-out (3 folds) ---
    seed_folds = [(0, 1), (2, 3), (4, 5)]
    fold_r2s = []
    all_preds = np.zeros(len(dy))
    all_test_mask = np.zeros(len(dy), dtype=bool)

    for fold_seeds in seed_folds:
        train_mask = np.array([m["seed"] not in fold_seeds for m in delta_meta])
        test_mask = ~train_mask

        X_tr, y_tr = dX[train_mask], dy[train_mask]
        X_te, y_te = dX[test_mask], dy[test_mask]

        if len(X_tr) < 4 or len(X_te) < 2:
            continue

        mu, sd = X_tr.mean(0), X_tr.std(0)
        sd[sd < 1e-12] = 1.0

        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit((X_tr - mu) / sd, y_tr)
        pred = ridge.predict((X_te - mu) / sd)
        all_preds[test_mask] = pred
        all_test_mask |= test_mask

        ss_res = float(np.sum((y_te - pred) ** 2))
        ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
        fold_r2s.append(r2)

    pooled_ss_res = float(np.sum((dy[all_test_mask] - all_preds[all_test_mask]) ** 2))
    pooled_ss_tot = float(np.sum((dy[all_test_mask] - dy[all_test_mask].mean()) ** 2))
    pooled_r2 = 1 - pooled_ss_res / pooled_ss_tot if pooled_ss_tot > 1e-12 else float("nan")

    results["held_out_seed"] = {
        "pooled_r2": pooled_r2,
        "fold_r2s": fold_r2s,
        "pooled_corr": float(np.corrcoef(dy[all_test_mask], all_preds[all_test_mask])[0, 1])
            if np.sum(all_test_mask) > 2 else float("nan"),
    }

    # --- Baselines (all aligned with delta_meta, no misalignment possible) ---
    baselines = {}
    baseline_preds = {}

    alphas = np.array([m["alpha"] for m in delta_meta]).reshape(-1, 1)
    alphas2 = np.column_stack([alphas, alphas**2])
    arch_indicator = np.array([1.0 if m["arch"] == "qwen3" else 0.0 for m in delta_meta]).reshape(-1, 1)

    baseline_specs = [
        ("alpha_only", alphas),
        ("alpha_quad", alphas2),
        ("delta_early_loss", del_X),
        ("alpha_plus_arch", np.column_stack([alphas, arch_indicator])),
    ]
    if avail_tel:
        baseline_specs.append(("delta_telemetry", tel_X))
    else:
        print_flush("    SKIP delta_telemetry baseline (no available features)")
    if avail_she:
        baseline_specs.append(("delta_shesha", she_X))
    else:
        print_flush("    SKIP delta_shesha baseline (no available features)")
    # combined includes telemetry only if available
    combined_parts = [alphas, alphas**2, del_X]
    if avail_tel:
        combined_parts.append(tel_X)
    baseline_specs.append(("combined_non_geometry", np.column_stack(combined_parts)))
    # strongest combined competitor: add arch indicator (adversarial attack #3)
    combined_plus_arch = combined_parts + [arch_indicator]
    baseline_specs.append(("combined_non_geometry_plus_arch", np.column_stack(combined_plus_arch)))

    for bname, bX in baseline_specs:
        bp, bm, br2 = _cv_ridge_baseline(bX, dy, delta_meta, seed_folds)
        if not math.isnan(br2):
            baselines[bname] = {"r2": br2}
            baseline_preds[bname] = bp.copy()

    # arm_mean baseline (cross-validated per prereg)
    am_preds = np.zeros(len(dy))
    am_mask = np.zeros(len(dy), dtype=bool)
    for fold_seeds in seed_folds:
        train_m = np.array([m["seed"] not in fold_seeds for m in delta_meta])
        test_m = ~train_m
        train_means = {}
        for i, m in enumerate(delta_meta):
            if train_m[i]:
                key = (m["arch"], m["alpha"])
                train_means.setdefault(key, []).append(dy[i])
        for k in train_means:
            train_means[k] = float(np.mean(train_means[k]))
        for i, m in enumerate(delta_meta):
            if test_m[i]:
                am_preds[i] = train_means.get((m["arch"], m["alpha"]), float(dy[train_m].mean()))
                am_mask[i] = True
    if am_mask.sum() > 2:
        ss = float(np.sum((dy[am_mask] - am_preds[am_mask]) ** 2))
        st = float(np.sum((dy[am_mask] - dy[am_mask].mean()) ** 2))
        baselines["arm_mean"] = {"r2": 1 - ss / st if st > 1e-12 else float("nan")}
        baseline_preds["arm_mean"] = am_preds.copy()

    results["baselines"] = baselines

    # geometry beats alpha-only?
    alpha_r2 = baselines.get("alpha_only", {}).get("r2", float("nan"))
    results["geometry_beats_alpha_only"] = pooled_r2 > alpha_r2 if not math.isnan(alpha_r2) else None

    # MSE reduction vs best non-geometry baseline
    geo_mse = pooled_ss_res / max(1, np.sum(all_test_mask))
    best_bl_name = max(baselines, key=lambda k: baselines[k].get("r2", float("-inf"))) if baselines else None
    best_bl_r2 = baselines[best_bl_name]["r2"] if best_bl_name else float("-inf")
    best_bl_mse = (1 - best_bl_r2) * pooled_ss_tot / max(1, np.sum(all_test_mask)) if not math.isinf(best_bl_r2) else float("inf")
    mse_reduction = 1 - geo_mse / best_bl_mse if best_bl_mse > 1e-12 else float("nan")
    results["mse_reduction_vs_best_baseline"] = mse_reduction
    results["best_baseline_name"] = best_bl_name

    # Permutation test -- two variants:
    # (a) shuffle within architecture (original prereg spec)
    # (b) shuffle within (arch, alpha) -- adversarial-demanded: isolates geometry beyond dose
    rng = np.random.RandomState(186)

    def _run_permutation(n_iter, group_keys):
        """Run permutation test shuffling geometry within specified groups."""
        nr2s = []
        for _ in range(n_iter):
            perm_dX = dX.copy()
            unique_keys = set(group_keys)
            for gk in unique_keys:
                gmask = np.array([k == gk for k in group_keys])
                perm_dX[gmask] = rng.permutation(perm_dX[gmask])
            p_preds = np.zeros(len(dy))
            p_mask = np.zeros(len(dy), dtype=bool)
            for fold_seeds in seed_folds:
                train_m = np.array([m["seed"] not in fold_seeds for m in delta_meta])
                test_m = ~train_m
                if train_m.sum() < 4:
                    continue
                mu, sd = perm_dX[train_m].mean(0), perm_dX[train_m].std(0)
                sd[sd < 1e-12] = 1.0
                ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                ridge.fit((perm_dX[train_m] - mu) / sd, dy[train_m])
                p_preds[test_m] = ridge.predict((perm_dX[test_m] - mu) / sd)
                p_mask |= test_m
            if p_mask.sum() > 2:
                ss = float(np.sum((dy[p_mask] - p_preds[p_mask]) ** 2))
                st = float(np.sum((dy[p_mask] - dy[p_mask].mean()) ** 2))
                nr2s.append(1 - ss / st if st > 1e-12 else float("nan"))
        return nr2s

    arch_keys = [m["arch"] for m in delta_meta]
    arch_alpha_keys = [(m["arch"], m["alpha"]) for m in delta_meta]

    null_r2s = _run_permutation(1000, arch_keys)
    null_r2s_cond = _run_permutation(1000, arch_alpha_keys)

    if null_r2s:
        results["permutation"] = {
            "p_value": float(np.mean([n >= pooled_r2 for n in null_r2s])),
            "null_mean_r2": float(np.mean(null_r2s)),
            "null_95th": float(np.percentile(null_r2s, 95)),
            "n_permutations": len(null_r2s),
        }
    if null_r2s_cond:
        results["permutation_cond_alpha"] = {
            "p_value": float(np.mean([n >= pooled_r2 for n in null_r2s_cond])),
            "null_mean_r2": float(np.mean(null_r2s_cond)),
            "null_95th": float(np.percentile(null_r2s_cond, 95)),
            "n_permutations": len(null_r2s_cond),
        }

    # --- Per-architecture R2 (prereg criterion 6) ---
    per_arch = {}
    for arch_name in ARCHS:
        arch_mask = np.array([m["arch"] == arch_name for m in delta_meta])
        if arch_mask.sum() < 4:
            continue
        a_preds = np.zeros(arch_mask.sum())
        a_tested = np.zeros(arch_mask.sum(), dtype=bool)
        a_dy = dy[arch_mask]
        a_dX = dX[arch_mask]
        a_meta = [m for m, am in zip(delta_meta, arch_mask) if am]
        for fold_seeds in seed_folds:
            tr = np.array([m["seed"] not in fold_seeds for m in a_meta])
            te = ~tr
            if tr.sum() < 2 or te.sum() < 1:
                continue
            mu, sd = a_dX[tr].mean(0), a_dX[tr].std(0)
            sd[sd < 1e-12] = 1.0
            ridge = RidgeCV(alphas=RIDGE_ALPHAS)
            ridge.fit((a_dX[tr] - mu) / sd, a_dy[tr])
            a_preds[te] = ridge.predict((a_dX[te] - mu) / sd)
            a_tested |= te
        if a_tested.sum() > 2:
            ss = float(np.sum((a_dy[a_tested] - a_preds[a_tested]) ** 2))
            st = float(np.sum((a_dy[a_tested] - a_dy[a_tested].mean()) ** 2))
            per_arch[arch_name] = {"r2": 1 - ss / st if st > 1e-12 else float("nan"),
                                   "n": int(a_tested.sum())}
    results["per_architecture"] = per_arch

    # --- Seed-block bootstrap CI (prereg criterion 3) ---
    seed_ids = np.array([m["seed"] for m in delta_meta])
    unique_seeds = np.unique(seed_ids)
    best_bl_preds_arr = baseline_preds.get(best_bl_name) if best_bl_name else None
    boot_diffs = []
    rng_boot = np.random.RandomState(1860)
    for _ in range(2000):
        boot_seeds = rng_boot.choice(unique_seeds, size=len(unique_seeds), replace=True)
        boot_idx = np.concatenate([np.where(seed_ids == s)[0] for s in boot_seeds])
        boot_idx = boot_idx[boot_idx < len(all_test_mask)]
        boot_idx = boot_idx[all_test_mask[boot_idx]]
        if len(boot_idx) < 4:
            continue
        boot_geo_mse = float(np.mean((dy[boot_idx] - all_preds[boot_idx]) ** 2))
        if best_bl_preds_arr is not None:
            boot_bl_mse = float(np.mean((dy[boot_idx] - best_bl_preds_arr[boot_idx]) ** 2))
        else:
            boot_bl_mse = float(np.var(dy[boot_idx]))
        boot_diffs.append(boot_bl_mse - boot_geo_mse)
    if boot_diffs:
        ci_lo = float(np.percentile(boot_diffs, 2.5))
        ci_hi = float(np.percentile(boot_diffs, 97.5))
        results["bootstrap_ci"] = {
            "best_baseline": best_bl_name,
            "ci_95_lower": ci_lo,
            "ci_95_upper": ci_hi,
            "ci_excludes_zero": ci_lo > 0,
            "n_bootstrap": len(boot_diffs),
        }

    # --- D5: Alpha decodability from geometry deltas ---
    alphas_arr = np.array([m["alpha"] for m in delta_meta])
    mu_d5, sd_d5 = dX.mean(0), dX.std(0)
    sd_d5[sd_d5 < 1e-12] = 1.0
    dX_s = (dX - mu_d5) / sd_d5
    ridge_d5 = RidgeCV(alphas=RIDGE_ALPHAS[:5])
    ridge_d5.fit(dX_s, alphas_arr)
    alpha_pred = ridge_d5.predict(dX_s)
    ss_d5 = float(np.sum((alphas_arr - alpha_pred) ** 2))
    st_d5 = float(np.sum((alphas_arr - alphas_arr.mean()) ** 2))
    results["d5_alpha_decodability"] = {
        "r2": 1 - ss_d5 / st_d5 if st_d5 > 1e-12 else float("nan"),
        "corr": float(np.corrcoef(alphas_arr, alpha_pred)[0, 1]),
    }

    # --- Held-out-dose stress test (prereg secondary) ---
    dose_stress = {}
    nonzero_alphas = sorted(set(m["alpha"] for m in delta_meta))
    for held_alpha in nonzero_alphas:
        tr_m = np.array([m["alpha"] != held_alpha for m in delta_meta])
        te_m = ~tr_m
        if tr_m.sum() < 4 or te_m.sum() < 2:
            continue
        mu_d, sd_d = dX[tr_m].mean(0), dX[tr_m].std(0)
        sd_d[sd_d < 1e-12] = 1.0
        ridge_d = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge_d.fit((dX[tr_m] - mu_d) / sd_d, dy[tr_m])
        pred_d = ridge_d.predict((dX[te_m] - mu_d) / sd_d)
        ss_d = float(np.sum((dy[te_m] - pred_d) ** 2))
        st_d = float(np.sum((dy[te_m] - dy[te_m].mean()) ** 2))
        dose_stress[str(held_alpha)] = {
            "r2": 1 - ss_d / st_d if st_d > 1e-12 else float("nan"),
            "n_test": int(te_m.sum()),
        }
    results["held_out_dose_stress"] = dose_stress

    # FIX #6: check if geometry only works for alpha=1.0
    alpha_1_only = False
    if dose_stress:
        non_1_doses = [v for k, v in dose_stress.items() if k != "1.0"]
        if non_1_doses and all(v["r2"] < 0 for v in non_1_doses):
            alpha_1_only = True
    results["alpha_1_only_flag"] = alpha_1_only

    # --- D1: Label variance check ---
    d1 = {"pooled_std": float(np.std(dy)), "pooled_range": [float(np.min(dy)), float(np.max(dy))]}
    for arch_name in ARCHS:
        am = [dy[i] for i, m in enumerate(delta_meta) if m["arch"] == arch_name]
        if am:
            d1[f"{arch_name}_std"] = float(np.std(am))
    for alpha_val in sorted(set(m["alpha"] for m in delta_meta)):
        dm = [dy[i] for i, m in enumerate(delta_meta) if m["alpha"] == alpha_val]
        if dm:
            d1[f"alpha_{alpha_val}_mean"] = float(np.mean(dm))
    results["d1_label_variance"] = d1
    if d1["pooled_std"] < 0.005:
        print_flush("    WARNING: pooled label std < 0.005 -- experiment may be under-identified")

    # --- D2: Dose monotonicity ---
    d2 = {}
    for arch_name in ARCHS:
        for alpha_val in sorted(set(m["alpha"] for m in delta_meta)):
            vals = [dy[i] for i, m in enumerate(delta_meta) if m["arch"] == arch_name and m["alpha"] == alpha_val]
            if vals:
                d2[f"{arch_name}_alpha_{alpha_val}"] = {"mean": float(np.mean(vals)), "n": len(vals)}
    results["d2_dose_monotonicity"] = d2

    # --- D4: Scratch denominator stability ---
    d4 = {}
    for arch_name in ARCHS:
        scratch_nlls = [c["final_nll"] for c in all_cells
                        if c["arch"] == arch_name and c["kd_alpha"] == 0.0
                        and math.isfinite(c.get("final_nll", float("nan")))]
        if scratch_nlls:
            d4[f"{arch_name}_scratch_std"] = float(np.std(scratch_nlls))
            d4[f"{arch_name}_scratch_mean"] = float(np.mean(scratch_nlls))
    results["d4_scratch_stability"] = d4

    # --- Verdict ---
    # Use the STRICTER conditioned permutation (within arch+alpha) for the verdict
    perm_p_raw = results.get("permutation", {}).get("p_value", 1.0)
    perm_p_cond = results.get("permutation_cond_alpha", {}).get("p_value", 1.0)
    perm_p = max(perm_p_raw, perm_p_cond)  # must pass BOTH
    ci_ok = results.get("bootstrap_ci", {}).get("ci_excludes_zero", False)
    arch_r2s = [v["r2"] for v in per_arch.values()
                if isinstance(v.get("r2"), (int, float)) and not math.isnan(v["r2"])]

    # FIX #5: require BOTH architectures present and neither negative
    arch_criterion = (
        len(arch_r2s) == len(ARCHS)
        and max(arch_r2s) >= 0.25
        and all(r >= 0 for r in arch_r2s)
    )

    passes_primary = (
        pooled_r2 >= 0.30
        and mse_reduction >= 0.20
        and perm_p <= 0.05
        and ci_ok
        and results.get("geometry_beats_alpha_only", False)
        and arch_criterion
        and not alpha_1_only
        and not results.get("rows_below_prereg_minimum", False)
    )
    weak_pass = (
        0.20 <= pooled_r2 < 0.30
        and mse_reduction >= 0.10
        and results.get("geometry_beats_alpha_only", False)
        and not alpha_1_only
        and not results.get("rows_below_prereg_minimum", False)
    )
    if passes_primary:
        verdict = "PASS"
    elif weak_pass:
        verdict = "WEAK PASS"
    else:
        verdict = "FAIL"

    results["verdict"] = verdict
    print_flush(f"\n*** g186 VERDICT: {verdict} ***")
    print_flush(f"    pooled R2={pooled_r2:.3f} mse_red={mse_reduction:.1%} "
                f"perm_p={perm_p:.3f} ci_ok={ci_ok} beats_alpha={results.get('geometry_beats_alpha_only')}")
    print_flush(f"    per-arch R2: {per_arch}")
    bl_str = ", ".join(f"{k}={v['r2']:.3f}" for k, v in baselines.items())
    print_flush(f"    baselines: {bl_str}")
    if results.get("d5_alpha_decodability"):
        print_flush(f"    D5 alpha-decodability R2={results['d5_alpha_decodability']['r2']:.3f}")
    if alpha_1_only:
        print_flush(f"    FAIL: geometry works only for alpha=1.0")
    if results.get("rows_below_prereg_minimum"):
        print_flush(f"    FAIL: only {results['n_pairs']}/48 rows (below prereg minimum)")

    return results


def main():
    parser = argparse.ArgumentParser(description="g186 KD dose-response")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reanalyze", action="store_true")
    parser.add_argument("--export-ridge", action="store_true",
                        help="Export frozen Ridge artifact for g185v2")
    parser.add_argument("--max-cells", type=int, default=999)
    args = parser.parse_args()

    t_start = time.time()

    seeds = [0] if args.smoke else SEEDS
    doses = [0.0, 1.0] if args.smoke else KD_ALPHAS
    archs = ARCHS

    print_flush(f"=== genome_186 KD Dose-Response ({now_utc()}) ===")
    print_flush(f"    doses: {doses}")
    print_flush(f"    seeds: {seeds}")
    print_flush(f"    archs: {archs}")
    print_flush(f"    total cells: {len(archs) * len(doses) * len(seeds)}")
    if args.smoke:
        print_flush("    *** SMOKE TEST MODE ***")

    if args.export_ridge:
        export_frozen_ridge()
        return

    if args.reanalyze:
        reanalyze_main()
        return

    existing = {}
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
    done_cells = existing.get("cells", [])
    done_ids = {c["cell_id"] for c in done_cells}

    any_kd = any(a > 0 for a in doses)
    teacher_texts = None
    if any_kd:
        cache_dir = ROOT / "results" / "cache" / "genome_182_features"
        teacher_cache_path = cache_dir / "teacher_texts.json"
        if teacher_cache_path.exists():
            n_teacher = 96 if args.smoke else g182.N_TRAIN_WINDOWS + 512
            teacher_texts = g182.load_teacher_text_cache(teacher_cache_path, n_teacher)
            print_flush(f"    Loaded {len(teacher_texts)} cached teacher texts")
        else:
            n_teacher = 96 if args.smoke else g182.N_TRAIN_WINDOWS + 512
            teacher_texts = g182.generate_teacher_texts(n_teacher)
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(teacher_cache_path, "w", encoding="utf-8") as f:
                json.dump(teacher_texts, f)
            print_flush(f"    Generated and cached {len(teacher_texts)} teacher texts")

    cells_run = 0
    for arch in archs:
        print_flush(f"\n=== Architecture: {arch} ===")

        tok = g182.get_tokenizer(arch)
        n_train = 96 if args.smoke else g182.N_TRAIN_WINDOWS
        n_val = 64 if args.smoke else g182.N_C4_VAL_WINDOWS
        pools = g182.load_c4_pools(tok, n_train, n_val, g182.SEQ_LEN)

        teacher_pools = None
        if teacher_texts is not None:
            teacher_enc = [tok(t, truncation=True, max_length=g182.SEQ_LEN,
                              padding="max_length", return_tensors="pt")
                           for t in teacher_texts]
            t_ids = torch.cat([e["input_ids"] for e in teacher_enc], dim=0)[:n_train]
            t_mask = torch.cat([e["attention_mask"] for e in teacher_enc], dim=0)[:n_train]
            teacher_pools = {"train_ids": t_ids, "train_mask": t_mask}

        stop_requested = False
        for alpha in doses:
            for seed in seeds:
                cell_id = f"{arch}_alpha_{alpha:.1f}_s{seed}"
                if cell_id in done_ids:
                    print_flush(f"    SKIP {cell_id} (already done)")
                    continue
                if cells_run >= args.max_cells:
                    print_flush(f"    MAX CELLS reached ({args.max_cells})")
                    stop_requested = True
                    break

                result = train_one_cell_dose(
                    arch=arch, seed=seed, kd_alpha=alpha,
                    pools=pools, teacher_pools=teacher_pools,
                    smoke=args.smoke,
                )
                done_cells.append(result)
                cells_run += 1

                existing["cells"] = done_cells
                existing["timestamp_utc"] = now_utc()
                g182.save_incremental(OUT_PATH, existing)
            if stop_requested:
                break

        del pools, teacher_pools, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if stop_requested:
            break

    # --- Analysis ---
    expected_total = len(archs) * len(doses) * len(seeds)
    run_complete = len({c["cell_id"] for c in done_cells}) >= expected_total
    all_cells = existing.get("cells", [])
    labeled = compute_dose_labels(all_cells)
    print_flush(f"\n=== Analysis: {len(labeled)} labeled cells ===")

    if run_complete and len(labeled) >= 10:
        analysis = pairwise_dose_analysis(labeled, all_cells)
        existing["dose_analysis"] = analysis

    total_time = time.time() - t_start
    existing["total_wallclock_s"] = total_time
    existing["status"] = "completed" if run_complete else "running"
    g182.save_incremental(OUT_PATH, existing)
    print_flush(f"\nTotal wallclock: {total_time/3600:.1f}h")


def reanalyze_main():
    if not OUT_PATH.exists():
        print_flush("No results file to reanalyze")
        return
    with open(OUT_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    cells = existing.get("cells", [])
    labeled = compute_dose_labels(cells)
    print_flush(f"=== Reanalysis: {len(labeled)} labeled cells ===")
    if len(labeled) >= 10:
        analysis = pairwise_dose_analysis(labeled, cells)
        existing["dose_analysis"] = analysis
        g182.save_incremental(OUT_PATH, existing)


def export_frozen_ridge():
    """Train Ridge on ALL g186 delta rows and export frozen artifact for g185v2."""
    from sklearn.linear_model import RidgeCV

    if not OUT_PATH.exists():
        print_flush("No results file to export from")
        return
    with open(OUT_PATH, encoding="utf-8") as f:
        existing = json.load(f)

    cells = existing.get("cells", [])
    labeled = compute_dose_labels(cells)
    feat_names = list(g182.MANIFOLD_ONLY_FEATURE_NAMES)

    scratch_by = {}
    for c in cells:
        if c["kd_alpha"] == 0.0:
            scratch_by[(c["arch"], c["seed"])] = c

    delta_X, delta_y, delta_meta = [], [], []
    for c in labeled:
        sc = scratch_by.get((c["arch"], c["seed"]))
        if sc is None:
            continue
        c_feats = c.get("features") or {}
        sc_feats = sc.get("features") or {}
        if not c_feats or not sc_feats:
            continue
        dx = _safe_delta(c_feats, sc_feats, feat_names)
        if dx is None:
            continue
        delta_X.append(dx)
        delta_y.append(c["label"])
        delta_meta.append({"arch": c["arch"], "seed": c["seed"], "alpha": c["kd_alpha"]})

    if len(delta_X) < 48:
        print_flush(f"Too few rows ({len(delta_X)}/48) -- need full g186 data to export")
        return

    with open(OUT_PATH, encoding="utf-8") as fcheck:
        check_data = json.load(fcheck)
    verdict = (check_data.get("dose_analysis") or {}).get("verdict", "")
    if verdict not in ("PASS", "WEAK PASS"):
        print_flush(f"g186 verdict is '{verdict}', not PASS -- frozen Ridge not scientifically valid")
        return

    dX = np.array(delta_X)
    dy = np.array(delta_y)

    mu = dX.mean(0)
    sd = dX.std(0)
    sd[sd < 1e-12] = 1.0

    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
    ridge.fit((dX - mu) / sd, dy)

    artifact = {
        "source": "genome_186_kd_dose_response",
        "n_training_rows": len(delta_X),
        "feature_names": feat_names,
        "feature_means": mu.tolist(),
        "feature_scales": sd.tolist(),
        "ridge_coef": ridge.coef_.tolist(),
        "ridge_intercept": float(ridge.intercept_),
        "ridge_alpha": float(ridge.alpha_),
        "training_r2": float(ridge.score((dX - mu) / sd, dy)),
        "training_label_mean": float(dy.mean()),
        "training_label_std": float(dy.std()),
    }

    out_path = ROOT / "results" / "genome_186_frozen_ridge.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print_flush(f"Frozen Ridge exported to {out_path}")
    print_flush(f"    n_rows={len(delta_X)} R2={artifact['training_r2']:.3f} "
                f"alpha={artifact['ridge_alpha']:.1f}")
    return artifact


def offline_dose_selection_replay():
    """C4 gate: simulate g185v2 dose selection on g186 data via leave-two-seeds-out."""
    from sklearn.linear_model import RidgeCV

    if not OUT_PATH.exists():
        print_flush("No results file for replay")
        return None
    with open(OUT_PATH, encoding="utf-8") as f:
        existing = json.load(f)

    cells = existing.get("cells", [])
    feat_names = list(g182.MANIFOLD_ONLY_FEATURE_NAMES)

    scratch_by = {}
    for c in cells:
        if c["kd_alpha"] == 0.0:
            scratch_by[(c["arch"], c["seed"])] = c

    # Build per-(arch, seed, alpha) lookup of features and final NLL
    cell_lookup = {}
    for c in cells:
        key = (c["arch"], c["seed"], c["kd_alpha"])
        cell_lookup[key] = c

    seed_folds = [(0, 1), (2, 3), (4, 5)]
    nonzero_alphas = [a for a in KD_ALPHAS if a > 0]

    # For each fold, train Ridge on training seeds, then select dose for held-out seeds
    selections = []  # list of dicts per (arch, seed)

    for fold_seeds in seed_folds:
        train_seeds = [s for s in SEEDS if s not in fold_seeds]
        test_seeds = list(fold_seeds)

        # Build training delta arrays
        tr_X, tr_y = [], []
        for arch in ARCHS:
            for seed in train_seeds:
                sc = scratch_by.get((arch, seed))
                if sc is None:
                    continue
                sc_feats = sc.get("features") or {}
                for alpha in nonzero_alphas:
                    c = cell_lookup.get((arch, seed, alpha))
                    if c is None:
                        continue
                    c_feats = c.get("features") or {}
                    dx = _safe_delta(c_feats, sc_feats, feat_names)
                    if dx is None:
                        continue
                    delta_nll = sc["final_nll"] - c["final_nll"]
                    tr_X.append(dx)
                    tr_y.append(delta_nll)

        if len(tr_X) < 8:
            continue

        tr_X = np.array(tr_X)
        tr_y = np.array(tr_y)
        mu, sd = tr_X.mean(0), tr_X.std(0)
        sd[sd < 1e-12] = 1.0

        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit((tr_X - mu) / sd, tr_y)

        # Select dose for each test (arch, seed)
        for arch in ARCHS:
            for seed in test_seeds:
                sc = scratch_by.get((arch, seed))
                if sc is None:
                    continue
                sc_feats = sc.get("features") or {}
                scratch_nll = sc["final_nll"]

                best_pred, best_alpha = -float("inf"), nonzero_alphas[0]
                preds_by_alpha = {}
                for alpha in nonzero_alphas:
                    c = cell_lookup.get((arch, seed, alpha))
                    if c is None:
                        continue
                    c_feats = c.get("features") or {}
                    dx = _safe_delta(c_feats, sc_feats, feat_names)
                    if dx is None:
                        continue
                    pred = ridge.predict(((np.array(dx) - mu) / sd).reshape(1, -1))[0]
                    preds_by_alpha[alpha] = float(pred)
                    if pred > best_pred:
                        best_pred = pred
                        best_alpha = alpha

                # Oracle: actually best dose
                actual_nlls = {}
                for alpha in nonzero_alphas:
                    c = cell_lookup.get((arch, seed, alpha))
                    if c:
                        actual_nlls[alpha] = c["final_nll"]

                if not actual_nlls:
                    continue
                oracle_alpha = min(actual_nlls, key=actual_nlls.get)
                oracle_nll = actual_nlls[oracle_alpha]
                selected_nll = actual_nlls.get(best_alpha, scratch_nll)

                selections.append({
                    "arch": arch, "seed": seed,
                    "geometry_alpha": best_alpha,
                    "oracle_alpha": oracle_alpha,
                    "alpha_heuristic": 1.0,
                    "scratch_nll": scratch_nll,
                    "selected_nll": selected_nll,
                    "oracle_nll": oracle_nll,
                    "heuristic_nll": actual_nlls.get(1.0, scratch_nll),
                })

    if not selections:
        print_flush("No selections made -- data insufficient for replay")
        return None

    # Score policies
    n = len(selections)
    geo_correct = sum(1 for s in selections if s["geometry_alpha"] == s["oracle_alpha"])
    geo_picks_1 = sum(1 for s in selections if s["geometry_alpha"] == 1.0)

    improvement_retentions = []
    regrets = []
    for s in selections:
        denom = s["scratch_nll"] - s["oracle_nll"]
        if abs(denom) < 1e-8:
            improvement_retentions.append(1.0)
        else:
            ir = (s["scratch_nll"] - s["selected_nll"]) / denom
            improvement_retentions.append(ir)
        regrets.append(s["oracle_nll"] - s["selected_nll"])

    heuristic_regrets = [s["oracle_nll"] - s["heuristic_nll"] for s in selections]

    result = {
        "n_selections": n,
        "dose_selection_accuracy": geo_correct / n,
        "alpha_1_agreement": geo_picks_1 / n,
        "mean_improvement_retention": float(np.mean(improvement_retentions)),
        "mean_regret": float(np.mean(regrets)),
        "mean_heuristic_regret": float(np.mean(heuristic_regrets)),
        "geometry_beats_heuristic": float(np.mean(regrets)) > float(np.mean(heuristic_regrets)),
        "selections": selections,
    }

    # Launch gate verdict
    launch_ok = (
        result["mean_improvement_retention"] >= 0.70
        and result["alpha_1_agreement"] < 0.80
        and result["geometry_beats_heuristic"]
    )
    result["launch_gate"] = "PASS" if launch_ok else "FAIL"

    print_flush(f"\n*** g185v2 Offline Replay Gate: {result['launch_gate']} ***")
    print_flush(f"    selections={n} accuracy={result['dose_selection_accuracy']:.1%} "
                f"alpha=1.0 agreement={result['alpha_1_agreement']:.1%}")
    print_flush(f"    improvement_retention={result['mean_improvement_retention']:.3f} "
                f"regret={result['mean_regret']:.4f}")
    print_flush(f"    heuristic_regret={result['mean_heuristic_regret']:.4f} "
                f"geometry_beats={result['geometry_beats_heuristic']}")

    return result


if __name__ == "__main__":
    main()
