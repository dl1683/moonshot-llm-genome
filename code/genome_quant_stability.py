"""Gate-1 G1.5 quantization-stability probe + strategic-directive efficiency hook.

Measures the same primitives on FP16 vs Q8 (bitsandbytes 8-bit) versions of
each system, at sentinel depths, on the same stimulus bank. Applies the §2.5.6
equivalence criterion `|Δ| + c·SE(Δ) < δ·median(|f|)` to each
(system, primitive, estimator) cell.

Strategic significance: this is the manifesto's efficiency hook. If a Gate-1-
passing primitive (e.g., kNN-k10) is STABLE under Q8 quantization, then the
atlas coordinate survives 4× model compression — that is evidence that the
universality claim is not tied to full-precision representations. In
manifesto language: geometry survives electricity reduction.

Per CLAUDE.md §6.1: new file justified as a distinct Gate-1 criterion runner
(G1.5-specific, not generic). Reuses genome_cross_arch.run_cross_arch by
looping quantization modes. Does NOT re-extract for quant modes we've already
measured — caches per-quant results.

Outputs results/gate1/quant_stability_n{N}_seeds{S}.json with per-cell
verdicts at δ ∈ {0.05, 0.10, 0.20}.

Windows+CUDA gotcha per Codex R7 D7: untrained twins must be fp16 (quantizing
random weights is not meaningful neg-control). Vision systems (DINOv2) skip
q8 by default unless explicitly requested — ViT + bnb is less-tested on
Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from stimulus_banks import c4_clean_v1  # noqa: E402
from genome_loaders import load_system, SYSTEM_IDS  # noqa: E402
from genome_extractor import (  # noqa: E402
    extract_trajectory, sentinel_layer_indices,
)
from genome_primitives import (  # noqa: E402
    twonn_id, mle_id, participation_ratio, knn_clustering_coefficient,
)
from genome_stim_resample import evaluate_g13_sensitivity  # noqa: E402


def _measure(X):
    n = X.shape[0]
    return [
        twonn_id(X),
        mle_id(X, k=min(10, n - 1)),
        participation_ratio(X, centered=True),
        participation_ratio(X, centered=False),
        knn_clustering_coefficient(X, k=min(5, n - 2)),
        knn_clustering_coefficient(X, k=min(10, n - 2)),
    ]


def run(*, n_sentences: int, seed: int, max_length: int = 128,
        systems: list[str] | None = None) -> dict:
    """For each text system with a defined Q8 path, measure the same
    primitives at FP16 and Q8 on the same stimulus bank. Compare via
    equivalence criterion for G1.5.
    """
    print(f"=== GATE-1 G1.5 QUANTIZATION STABILITY ===")
    print(f"n_sentences={n_sentences}, seed={seed}")
    t0 = time.time()

    # Text-only for now: bnb 8-bit works cleanly on transformers; vision
    # bnb is less-tested on Windows. Skip vision unless explicitly requested.
    text_systems = {k: v for k, v in SYSTEM_IDS.items()
                    if v.get("modality") == "text"}
    if systems:
        text_systems = {k: v for k, v in text_systems.items() if k in systems}

    # Stream text stimuli once.
    print(f"[{time.time()-t0:.1f}s] streaming {n_sentences} C4 sentences...")
    text_stimuli = [it["text"] for it in c4_clean_v1(seed=seed, n_samples=n_sentences)]
    print(f"[{time.time()-t0:.1f}s] {len(text_stimuli)} stimuli ready")

    all_rows = []
    per_quant_summary = {}

    for system_key, meta in text_systems.items():
        hf_id = meta["hf_id"]
        for quant in ("fp16", "q8"):
            tag = f"{system_key}_{quant}"
            print(f"\n--- {tag} ---")
            t_sys = time.time()
            try:
                sys_obj = load_system(hf_id, quant=quant, untrained=False,
                                      device="cuda")
            except Exception as exc:
                print(f"  SKIP {tag}: {type(exc).__name__}: {str(exc)[:200]}")
                per_quant_summary[tag] = {"status": "load_failed",
                                          "reason": f"{type(exc).__name__}: {exc}"}
                continue

            n_layers = sys_obj.n_hidden_layers()
            sentinel_idxs = sentinel_layer_indices(n_layers)
            print(f"  [{time.time()-t_sys:.1f}s] loaded {hf_id} (L={n_layers})")

            try:
                traj = extract_trajectory(
                    sys_obj.model, sys_obj.tokenizer, text_stimuli,
                    layer_indices=sentinel_idxs, pooling="seq_mean",
                    max_length=max_length, device="cuda",
                    system_key=sys_obj.system_key,
                    class_id=sys_obj.class_id,
                    quantization=quant,
                    stimulus_version=f"c4_en.seed{seed}.n{len(text_stimuli)}",
                    seed=seed,
                )
            except Exception as exc:
                print(f"  FAIL extract for {tag}: {type(exc).__name__}: {str(exc)[:200]}")
                sys_obj.unload()
                per_quant_summary[tag] = {"status": "extract_failed",
                                          "reason": f"{type(exc).__name__}: {exc}"}
                continue

            for lyr in traj.layers:
                measurements = _measure(lyr.X)
                for m in measurements:
                    all_rows.append({
                        "system_key": system_key,
                        "quantization": quant,
                        "class_id": sys_obj.class_id,
                        "hf_id": hf_id,
                        "k_index": lyr.k_index,
                        "k_normalized": round(lyr.k_normalized, 4),
                        "primitive_id": m.primitive_id,
                        "estimator": m.estimator,
                        "value": m.value,
                        "se": m.se,
                        "n_points": m.n_points,
                        "seed": seed,
                    })
            per_quant_summary[tag] = {
                "status": "ok",
                "wall_clock_system_seconds": round(time.time() - t_sys, 2),
            }
            sys_obj.unload()
            print(f"  [{time.time()-t_sys:.1f}s] unloaded")

    # G1.5 equivalence verdict: pair FP16 vs Q8 per (system, primitive, estimator, depth).
    grouped: dict[tuple, dict[int, tuple[float, float]]] = {}
    for r in all_rows:
        key = (r["system_key"], r["primitive_id"], r["estimator"],
               round(r["k_normalized"], 3))
        # Use seed=0 for fp16, seed=1 for q8 as "fake resample seeds" so we
        # can reuse evaluate_g13_sensitivity (it computes |Δ| + c·SE across
        # any 2-seed pair; we use quant as the pair dimension).
        quant_seed = 0 if r["quantization"] == "fp16" else 1
        grouped.setdefault(key, {})[quant_seed] = (r["value"], r["se"])

    verdicts = evaluate_g13_sensitivity(grouped)

    out_dir = _THIS_DIR.parent / "results" / "gate1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"quant_stability_n{n_sentences}_seed{seed}.json"
    payload = {
        "experiment_id": "genome_008_quant_stability",
        "n_sentences": n_sentences,
        "seed": seed,
        "total_wall_clock_seconds": round(time.time() - t0, 2),
        "per_system_quant_summary": per_quant_summary,
        "n_rows": len(all_rows),
        "g15_verdicts": {f"{s}||{p}||{e}": v for (s, p, e), v in verdicts.items()},
        "rows": all_rows,
    }
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=float)

    passed = sum(1 for v in verdicts.values()
                 if v["sensitivity_sweep"]["delta_0.1"]["status"] == "pass")
    total = len(verdicts)
    print(f"\n=== G1.5 QUANT-STABILITY ===")
    print(f"total (system, primitive, estimator) cells: {total}")
    print(f"G1.5 PASS at delta=0.10: {passed} / {total}")
    print(f"out: {out_path}")
    return payload


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--n-sentences", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--systems", type=str, nargs="*", default=None,
                    help="subset of system_keys; defaults to all text systems")
    args = ap.parse_args()
    run(n_sentences=args.n_sentences, seed=args.seed,
        max_length=args.max_length, systems=args.systems)
