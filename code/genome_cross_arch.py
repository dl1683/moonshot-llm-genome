"""Cross-architecture Batch-1 pilot runner — first actual test of the
universality axiom.

Runs the three Batch-1 systems (Qwen3-0.6B autoregressive LLM,
Mamba2-370M SSM, Falcon-H1-0.5B hybrid) on the SAME C4 stimulus bank
at the SAME sentinel depths (normalized, not raw-layer-index), computes
ID + PR + kNN-clustering at each depth, and emits one atlas_rows JSON
that covers all three systems.

Per `atlas_tl_session.md` section 3c Phase-3 Batch-1 plan, with caveats
documented in the ledger entry (still single resample, single quant).

Wall-clock target: ~90 seconds at n=500. Sequential system loads with
unload between to keep peak VRAM at ~1.3 GB.

Usage:
    python code/genome_cross_arch.py -n 500 --c4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from stimulus_banks import c4_clean_v1, imagenet_val_v1  # noqa: E402
from genome_loaders import load_system, SYSTEM_IDS  # noqa: E402
from genome_extractor import (  # noqa: E402
    extract_trajectory, extract_vision_trajectory, sentinel_layer_indices,
)
from genome_primitives import (  # noqa: E402
    twonn_id, mle_id, participation_ratio, knn_clustering_coefficient,
)


# -------------------- Fixed hardcoded fixture for n<=5 sanity runs --------------------

_FALLBACK_STIMULI = [
    "The cat sat on the mat and watched the rain fall outside the window all afternoon.",
    "She opened the book to a random page and began to read aloud to the sleeping dog.",
    "Scientists have long debated whether language shapes thought or thought shapes language.",
    "The old lighthouse keeper walked down the spiral staircase carrying a lantern and a heavy iron key.",
    "A thousand starlings rose from the field in a single coordinated motion that looked almost magical.",
]


def _current_commit_sha() -> str:
    git_head = _THIS_DIR.parent / ".git" / "HEAD"
    if not git_head.exists():
        return "unknown"
    try:
        ref = git_head.read_text().strip()
        if ref.startswith("ref:"):
            ref_path = _THIS_DIR.parent / ".git" / ref.split()[1]
            if ref_path.exists():
                return ref_path.read_text().strip()
        return ref
    except OSError:
        return "unknown"


def _measure_point_cloud(X):
    """Run all Batch-1 primitives + estimator variants on a single point cloud.

    Extended 2026-04-21 with k=3/20/30 neighborhood sizes — Gate-2 G2.3
    hierarchical-fit needs k ∈ {3, 5, 10, 20, 30} to identify α_d, β_d,
    κ, d_int independently (per prereg `genome_knn_k10_hierarchical_2026-04-21.md`
    §8). Smoke fit at k ∈ {5, 10} alone is underdetermined.
    """
    n = X.shape[0]
    return [
        twonn_id(X),
        mle_id(X, k=min(10, n - 1)),
        participation_ratio(X, centered=True),
        participation_ratio(X, centered=False),
        knn_clustering_coefficient(X, k=min(3, n - 2)),
        knn_clustering_coefficient(X, k=min(5, n - 2)),
        knn_clustering_coefficient(X, k=min(10, n - 2)),
        knn_clustering_coefficient(X, k=min(20, n - 2)),
        knn_clustering_coefficient(X, k=min(30, n - 2)),
    ]


def run_cross_arch(*, n_sentences: int, use_c4: bool, seed: int,
                   max_length: int, run_untrained: bool = False,
                   systems_filter: list[str] | None = None) -> dict:
    print(f"=== CROSS-ARCHITECTURE BATCH-1 PILOT ===")
    print(f"n_sentences: {n_sentences}, c4: {use_c4}, seed: {seed}, "
          f"untrained_twins: {run_untrained}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    t0 = time.time()

    # 1. Modality-specific stimulus banks. Both used if any vision system present.
    if use_c4:
        print(f"[{time.time()-t0:.1f}s] streaming {n_sentences} sentences from C4...")
        text_stimuli = [it["text"] for it in c4_clean_v1(seed=seed, n_samples=n_sentences)]
    else:
        base = _FALLBACK_STIMULI
        reps = max(1, (n_sentences + len(base) - 1) // len(base))
        text_stimuli = (base * reps)[:n_sentences]
    print(f"[{time.time()-t0:.1f}s] text stimuli ready ({len(text_stimuli)} sentences)")

    vision_stimuli = None
    has_vision = any(meta.get("modality") == "vision"
                     for meta in SYSTEM_IDS.values())
    if has_vision:
        print(f"[{time.time()-t0:.1f}s] streaming {n_sentences} images from ImageNet-val mirror...")
        try:
            vision_stimuli = [it["image"] for it in imagenet_val_v1(
                seed=seed, n_samples=n_sentences)]
            print(f"[{time.time()-t0:.1f}s] vision stimuli ready "
                  f"({len(vision_stimuli)} images)")
        except Exception as exc:
            print(f"  WARN: vision stimulus fetch failed "
                  f"({type(exc).__name__}: {exc}); vision systems will be skipped")
            vision_stimuli = None

    # 2. For each system, load -> extract -> measure -> unload -> collect.
    all_rows = []
    per_system_summary = {}
    commit_sha = _current_commit_sha()
    stimulus_version = f"{'c4_en' if use_c4 else 'fallback'}.seed{seed}.n{len(text_stimuli)}"

    # Build system list: trained + optional untrained twins for neg-control.
    system_plan: list[tuple[str, dict, bool]] = []
    for system_key, meta in SYSTEM_IDS.items():
        if systems_filter is not None and system_key not in systems_filter:
            continue
        system_plan.append((system_key, meta, False))  # trained
        if run_untrained:
            system_plan.append((system_key, meta, True))  # untrained twin

    for system_key, meta, untrained in system_plan:
        hf_id = meta["hf_id"]
        modality = meta.get("modality", "text")
        tag = f"{system_key}{'_untrained' if untrained else ''}"
        print(f"\n--- System: {tag} ({hf_id}{' random-init' if untrained else ''}; modality={modality}) ---")
        t_sys = time.time()

        # Skip vision systems if vision stimuli unavailable.
        if modality == "vision" and vision_stimuli is None:
            print(f"  SKIP {tag}: vision stimuli unavailable")
            per_system_summary[tag] = {
                "status": "skipped",
                "reason": "vision stimuli not fetched this run",
            }
            continue

        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=untrained,
                                  device="cuda")
        except Exception as exc:
            print(f"  SKIP {tag}: load failed ({type(exc).__name__}: {exc})")
            per_system_summary[tag] = {
                "status": "skipped",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            continue

        n_layers = sys_obj.n_hidden_layers()
        sentinel_idxs = sentinel_layer_indices(n_layers)
        print(f"  [{time.time()-t_sys:.1f}s] loaded {hf_id} (L={n_layers})")
        print(f"  sentinel layers: {sentinel_idxs} at depths "
              f"{[round(i/max(n_layers-1, 1), 3) for i in sentinel_idxs]}")

        try:
            if modality == "text":
                traj = extract_trajectory(
                    sys_obj.model, sys_obj.tokenizer, text_stimuli,
                    layer_indices=sentinel_idxs, pooling="seq_mean",
                    max_length=max_length, device="cuda",
                    system_key=sys_obj.system_key,
                    class_id=sys_obj.class_id,
                    quantization=sys_obj.quant,
                    stimulus_version=stimulus_version,
                    seed=seed,
                )
            else:  # vision
                traj = extract_vision_trajectory(
                    sys_obj.model, sys_obj.image_processor, vision_stimuli,
                    layer_indices=sentinel_idxs, pooling="cls_or_mean",
                    device="cuda",
                    system_key=sys_obj.system_key,
                    class_id=sys_obj.class_id,
                    quantization=sys_obj.quant,
                    stimulus_version="imagenet_val.v1.seed" + str(seed),
                    seed=seed,
                )
        except Exception as exc:
            print(f"  FAIL extraction for {tag}: {type(exc).__name__}: {exc}")
            sys_obj.unload()
            per_system_summary[tag] = {
                "status": "extract_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            continue

        print(f"  [{time.time()-t_sys:.1f}s] extracted {len(traj.layers)} sentinel layers")

        # Compute primitives at each sentinel depth.
        for lyr in traj.layers:
            measurements = _measure_point_cloud(lyr.X)
            for m in measurements:
                all_rows.append({
                    "system_key": system_key,
                    "system_tag": tag,
                    "class_id": sys_obj.class_id,
                    "class_name": sys_obj.class_name,
                    "hf_id": hf_id,
                    "untrained": sys_obj.untrained,
                    "quantization": sys_obj.quant,
                    "pooling": traj.pooling,
                    "stimulus_version": stimulus_version,
                    "seed": seed,
                    "k_index": lyr.k_index,
                    "k_normalized": round(lyr.k_normalized, 4),
                    "n_points": m.n_points,
                    "primitive_id": m.primitive_id,
                    "estimator": m.estimator,
                    "value": m.value,
                    "se": m.se,
                    "scope_label": (
                        f"(modality=text, stimulus_family={stimulus_version}, "
                        f"pooling=seq_mean, tokenizer=per-model-native)"
                    ),
                    "commit_sha": commit_sha,
                })

        per_system_summary[tag] = {
            "status": "ok",
            "n_layers_total": n_layers,
            "sentinel_layers": sentinel_idxs,
            "wall_clock_system_seconds": round(time.time() - t_sys, 2),
            "untrained": untrained,
        }

        # Free VRAM before loading the next system.
        sys_obj.unload()
        print(f"  [{time.time()-t_sys:.1f}s] unloaded; continuing")

    # 3. Emit.
    out_dir = _THIS_DIR.parent / "results" / "cross_arch"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_n{len(text_stimuli)}{'_c4' if use_c4 else ''}_seed{seed}"
    if systems_filter:
        # Don't clobber the canonical full-bestiary atlas; write partial-run
        # rows to a suffixed file so the caller can merge explicitly.
        filter_tag = "_".join(sorted(systems_filter))[:40]
        suffix += f"_only_{filter_tag}"
    out_path = out_dir / f"atlas_rows{suffix}.json"

    payload = {
        "experiment_id": "genome_005_cross_modal",
        "commit_sha": commit_sha,
        "n_sentences": len(text_stimuli),
        "stimulus_version": stimulus_version,
        "wall_clock_seconds": round(time.time() - t0, 2),
        "per_system_summary": per_system_summary,
        "n_rows": len(all_rows),
        "rows": all_rows,
    }
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=float)

    print(f"\n=== DONE in {payload['wall_clock_seconds']}s ===")
    print(f"systems_ok: {sum(1 for s in per_system_summary.values() if s['status']=='ok')}")
    print(f"n_rows: {len(all_rows)}")
    print(f"out: {out_path}")
    return payload


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--n-sentences", type=int, default=500)
    ap.add_argument("--c4", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--untrained", action="store_true",
                    help="also run untrained-twin negative-control for each system")
    ap.add_argument("--systems", type=str, nargs="*", default=None,
                    help="subset of SYSTEM_IDS keys to run; defaults to all")
    args = ap.parse_args()
    result = run_cross_arch(
        n_sentences=args.n_sentences, use_c4=args.c4, seed=args.seed,
        max_length=args.max_length, run_untrained=args.untrained,
        systems_filter=args.systems,
    )
    ok = sum(1 for s in result["per_system_summary"].values() if s["status"] == "ok")
    sys.exit(0 if ok >= 1 else 1)
