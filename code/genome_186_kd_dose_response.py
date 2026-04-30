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
    train_steps = 20 if smoke else g182.TRAIN_STEPS
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


def pairwise_dose_analysis(labeled: list[dict], all_cells: list[dict]) -> dict[str, Any]:
    """Primary analysis: seed-matched delta geometry predicts delta NLL."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut

    feat_names = list(g182.MANIFOLD_ONLY_FEATURE_NAMES)
    results = {}

    scratch_by = {}
    for c in all_cells:
        if c["kd_alpha"] == 0.0:
            scratch_by[(c["arch"], c["seed"])] = c

    delta_X, delta_y, delta_meta = [], [], []
    for c in labeled:
        sc = scratch_by.get((c["arch"], c["seed"]))
        if sc is None:
            continue
        dx = []
        for fn in feat_names:
            v_kd = float(c["features"].get(fn, 0))
            v_sc = float(sc["features"].get(fn, 0))
            dx.append(v_kd - v_sc)
        delta_X.append(dx)
        delta_y.append(c["label"])
        delta_meta.append({"arch": c["arch"], "seed": c["seed"], "alpha": c["kd_alpha"]})

    if len(delta_X) < 10:
        return {"error": f"too few delta pairs: {len(delta_X)}"}

    dX = np.array(delta_X)
    dy = np.array(delta_y)

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

        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
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

    # --- Baselines ---
    baselines = {}

    # alpha-only
    alphas = np.array([m["alpha"] for m in delta_meta]).reshape(-1, 1)
    alphas2 = np.column_stack([alphas, alphas**2])
    for bname, bX in [("alpha_only", alphas), ("alpha_quad", alphas2)]:
        b_preds = np.zeros(len(dy))
        b_mask = np.zeros(len(dy), dtype=bool)
        for fold_seeds in seed_folds:
            train_m = np.array([m["seed"] not in fold_seeds for m in delta_meta])
            test_m = ~train_m
            if train_m.sum() < 2 or test_m.sum() < 1:
                continue
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
            mu, sd = bX[train_m].mean(0), bX[train_m].std(0)
            sd[sd < 1e-12] = 1.0
            ridge.fit((bX[train_m] - mu) / sd, dy[train_m])
            b_preds[test_m] = ridge.predict((bX[test_m] - mu) / sd)
            b_mask |= test_m
        if b_mask.sum() > 2:
            ss = float(np.sum((dy[b_mask] - b_preds[b_mask]) ** 2))
            st = float(np.sum((dy[b_mask] - dy[b_mask].mean()) ** 2))
            baselines[bname] = {"r2": 1 - ss / st if st > 1e-12 else float("nan")}

    # delta_early_loss baseline
    delta_el = []
    for c in labeled:
        sc = scratch_by.get((c["arch"], c["seed"]))
        if sc:
            delta_el.append(c.get("early_loss", 0) - sc.get("early_loss", 0))
        else:
            delta_el.append(0)
    del_X = np.array(delta_el).reshape(-1, 1)
    b_preds = np.zeros(len(dy))
    b_mask = np.zeros(len(dy), dtype=bool)
    for fold_seeds in seed_folds:
        train_m = np.array([m["seed"] not in fold_seeds for m in delta_meta])
        test_m = ~train_m
        if train_m.sum() < 2:
            continue
        mu, sd = del_X[train_m].mean(0), del_X[train_m].std(0)
        sd[sd < 1e-12] = 1.0
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        ridge.fit((del_X[train_m] - mu) / sd, dy[train_m])
        b_preds[test_m] = ridge.predict((del_X[test_m] - mu) / sd)
        b_mask |= test_m
    if b_mask.sum() > 2:
        ss = float(np.sum((dy[b_mask] - b_preds[b_mask]) ** 2))
        st = float(np.sum((dy[b_mask] - dy[b_mask].mean()) ** 2))
        baselines["delta_early_loss"] = {"r2": 1 - ss / st if st > 1e-12 else float("nan")}

    # arm_mean baseline
    arm_means = {}
    for i, m in enumerate(delta_meta):
        key = (m["arch"], m["alpha"])
        arm_means.setdefault(key, []).append(dy[i])
    for k in arm_means:
        arm_means[k] = np.mean(arm_means[k])
    am_preds = np.array([arm_means.get((m["arch"], m["alpha"]), 0) for m in delta_meta])
    ss = float(np.sum((dy - am_preds) ** 2))
    st = float(np.sum((dy - dy.mean()) ** 2))
    baselines["arm_mean"] = {"r2": 1 - ss / st if st > 1e-12 else float("nan")}

    results["baselines"] = baselines

    # geometry beats alpha-only?
    alpha_r2 = baselines.get("alpha_only", {}).get("r2", float("nan"))
    results["geometry_beats_alpha_only"] = pooled_r2 > alpha_r2 if not math.isnan(alpha_r2) else None

    # MSE reduction vs best baseline
    geo_mse = pooled_ss_res / max(1, np.sum(all_test_mask))
    best_bl_r2 = max((v.get("r2", float("-inf")) for v in baselines.values()), default=float("-inf"))
    best_bl_mse = (1 - best_bl_r2) * pooled_ss_tot / max(1, np.sum(all_test_mask)) if not math.isinf(best_bl_r2) else float("inf")
    mse_reduction = 1 - geo_mse / best_bl_mse if best_bl_mse > 1e-12 else float("nan")
    results["mse_reduction_vs_best_baseline"] = mse_reduction

    # --- Permutation test (500 iterations) ---
    rng = np.random.RandomState(186)
    null_r2s = []
    for _ in range(500):
        perm_dy = dy.copy()
        for arch_name in ARCHS:
            arch_mask = np.array([m["arch"] == arch_name for m in delta_meta])
            perm_dy[arch_mask] = rng.permutation(perm_dy[arch_mask])
        p_preds = np.zeros(len(perm_dy))
        p_mask = np.zeros(len(perm_dy), dtype=bool)
        for fold_seeds in seed_folds:
            train_m = np.array([m["seed"] not in fold_seeds for m in delta_meta])
            test_m = ~train_m
            if train_m.sum() < 4:
                continue
            mu, sd = dX[train_m].mean(0), dX[train_m].std(0)
            sd[sd < 1e-12] = 1.0
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
            ridge.fit((dX[train_m] - mu) / sd, perm_dy[train_m])
            p_preds[test_m] = ridge.predict((dX[test_m] - mu) / sd)
            p_mask |= test_m
        if p_mask.sum() > 2:
            ss = float(np.sum((perm_dy[p_mask] - p_preds[p_mask]) ** 2))
            st = float(np.sum((perm_dy[p_mask] - perm_dy[p_mask].mean()) ** 2))
            null_r2s.append(1 - ss / st if st > 1e-12 else float("nan"))

    if null_r2s:
        results["permutation"] = {
            "p_value": float(np.mean([n >= pooled_r2 for n in null_r2s])),
            "null_mean_r2": float(np.mean(null_r2s)),
            "null_95th": float(np.percentile(null_r2s, 95)),
            "n_permutations": len(null_r2s),
        }

    # --- Verdict ---
    perm_p = results.get("permutation", {}).get("p_value", 1.0)
    passes_primary = (
        pooled_r2 >= 0.30
        and mse_reduction >= 0.20
        and perm_p <= 0.05
        and results.get("geometry_beats_alpha_only", False)
    )
    weak_pass = (
        0.20 <= pooled_r2 < 0.30
        and mse_reduction >= 0.10
        and results.get("geometry_beats_alpha_only", False)
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
                f"perm_p={perm_p:.3f} beats_alpha={results.get('geometry_beats_alpha_only')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="g186 KD dose-response")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reanalyze", action="store_true")
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

        for alpha in doses:
            for seed in seeds:
                cell_id = f"{arch}_alpha_{alpha:.1f}_s{seed}"
                if cell_id in done_ids:
                    print_flush(f"    SKIP {cell_id} (already done)")
                    continue
                if cells_run >= args.max_cells:
                    print_flush(f"    MAX CELLS reached ({args.max_cells})")
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

        del pools, teacher_pools, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Analysis ---
    all_cells = existing.get("cells", [])
    labeled = compute_dose_labels(all_cells)
    print_flush(f"\n=== Analysis: {len(labeled)} labeled cells ===")

    if len(labeled) >= 10:
        analysis = pairwise_dose_analysis(labeled, all_cells)
        existing["dose_analysis"] = analysis

    total_time = time.time() - t_start
    existing["total_wallclock_s"] = total_time
    existing["status"] = "completed"
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


if __name__ == "__main__":
    main()
