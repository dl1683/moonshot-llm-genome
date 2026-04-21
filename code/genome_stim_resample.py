"""Gate-1 G1.3 stimulus-resample stability probe.

Runs the cross-modal Batch-1 pipeline with THREE seed-disjoint stimulus
resamples (seeds 42, 123, 456 per prereg section 4) and evaluates whether
each primitive passes the G1.3 equivalence criterion:

    |Delta| + c * SE(Delta) < delta_relative * median(|f|)

with c = 2.77 (K=18 Bonferroni) and delta_relative = 0.10 (per prereg
section 5 defaults).

If a (primitive, system, depth) tuple PASSES G1.3 for all 3 pairwise resample
comparisons, that tuple is a 🟡 portability-gate-passed measurement at this
scope. Per Codex R5 audit + 2.5.6b, aggregate via max-over-pairs per
(primitive, system).

Outputs results/cross_arch/atlas_rows_resample_multiseed.json with per-pair
equivalence stats + aggregate per-system verdicts.

Runs within envelope: ~3 seeds * ~95s per cross-modal run = ~5 min wall-clock
(stimulus caching lowers this further after the first seed).

Per CLAUDE.md: new file justified because multi-seed evaluation is a distinct
reusable boundary (Gate-1 suite module).
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

from genome_cross_arch import run_cross_arch  # noqa: E402


def _group_rows(rows: list[dict]) -> dict:
    """Group atlas rows by (system_key, primitive_id, estimator, k_normalized)."""
    out: dict[tuple, dict[int, tuple[float, float]]] = {}
    for r in rows:
        key = (r["system_key"], r["primitive_id"], r["estimator"],
               round(r["k_normalized"], 3))
        out.setdefault(key, {})[int(r["seed"])] = (r["value"], r["se"])
    return out


def evaluate_g13_sensitivity(multiseed_rows: dict,
                              *, c: float = 2.77,
                              deltas: tuple[float, ...] = (0.05, 0.10, 0.20)
                              ) -> dict:
    """Mandatory δ sensitivity sweep per prereg §5.

    Returns per-(system, primitive, estimator) cell: max_stat + pass/fail at
    each delta_relative in the sweep. Claims promoted only if they pass at
    δ=0.10 (primary); δ=0.05 tightens for 🟢 candidacy; δ=0.20 is the loose
    "δ-sensitive" annotation threshold.
    """
    base = evaluate_g13(multiseed_rows, c=c, delta_relative=deltas[1])
    for key, v in base.items():
        max_stat = v["max_stat"]
        median_abs_f = v["median_abs_f"]
        v["sensitivity_sweep"] = {}
        for d in deltas:
            margin_d = d * median_abs_f
            v["sensitivity_sweep"][f"delta_{d}"] = {
                "margin": float(margin_d),
                "status": "pass" if max_stat < margin_d else "fail",
            }
    return base


def evaluate_g13(multiseed_rows: dict,
                 *, c: float = 2.77, delta_relative: float = 0.10
                 ) -> dict:
    """G1.3 equivalence verdicts per (system, primitive, estimator).

    Aggregate rule (per 2.5.6b): for each criterion, max over the sub-grid
    of `|Delta| + c * SE(Delta)`. Primitive is G1.3-stable on that system
    iff max_stat < delta_relative * median(|f|) across all depths.
    """
    # Organize: results_by_spk[(system, primitive, estimator)][depth][seed] = (v, se)
    results_by_spk: dict[tuple, dict] = {}
    for (system, prim, est, depth), per_seed in multiseed_rows.items():
        spk = (system, prim, est)
        results_by_spk.setdefault(spk, {})[depth] = per_seed

    verdicts = {}
    for (system, prim, est), by_depth in results_by_spk.items():
        # All primitive values across depths and seeds (for median).
        all_values = [v for d in by_depth.values() for (v, _se) in d.values()]
        all_finite = [abs(v) for v in all_values
                      if v == v and abs(v) != float("inf")]
        if not all_finite:
            continue
        median_abs_f = float(sorted(all_finite)[len(all_finite) // 2])
        if median_abs_f == 0:
            continue
        margin = delta_relative * median_abs_f

        max_stat = 0.0
        details = []
        for depth, per_seed in by_depth.items():
            seeds = sorted(per_seed.keys())
            if len(seeds) < 2:
                continue
            # Pairwise comparisons.
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    s_i, s_j = seeds[i], seeds[j]
                    v_i, se_i = per_seed[s_i]
                    v_j, se_j = per_seed[s_j]
                    if not (v_i == v_i and v_j == v_j):  # NaN guard
                        continue
                    delta = abs(v_i - v_j)
                    se_delta = (se_i ** 2 + se_j ** 2) ** 0.5
                    stat = delta + c * se_delta
                    details.append({
                        "depth": depth,
                        "seed_pair": (s_i, s_j),
                        "delta": float(delta),
                        "se_delta": float(se_delta),
                        "equiv_stat": float(stat),
                    })
                    if stat > max_stat:
                        max_stat = stat
        status = "pass" if max_stat < margin else "fail"
        verdicts[(system, prim, est)] = {
            "status": status,
            "max_stat": float(max_stat),
            "margin_abs": float(margin),
            "median_abs_f": float(median_abs_f),
            "delta_relative": float(delta_relative),
            "c_bonferroni": float(c),
            "n_pairs": len(details),
            "details_truncated": details[:9],  # first few for inspection
        }
    return verdicts


def run(*, n_sentences: int, max_length: int, seeds: list[int]) -> dict:
    t0 = time.time()
    all_rows: list[dict] = []
    per_seed_summary = {}
    for seed in seeds:
        print(f"\n======== SEED {seed} ========")
        seed_t0 = time.time()
        result = run_cross_arch(
            n_sentences=n_sentences, use_c4=True, seed=seed,
            max_length=max_length, run_untrained=False,
        )
        all_rows.extend(result.get("rows", []))
        per_seed_summary[seed] = {
            "systems_ok": sum(1 for s in result.get("per_system_summary", {}).values()
                              if s.get("status") == "ok"),
            "wall_clock": result.get("wall_clock_seconds"),
            "seed_wall_clock": round(time.time() - seed_t0, 2),
        }

    grouped = _group_rows(all_rows)
    verdicts = evaluate_g13_sensitivity(grouped)

    out_dir = _THIS_DIR.parent / "results" / "gate1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stim_resample_n{n_sentences}_seeds{'_'.join(str(s) for s in seeds)}.json"

    payload = {
        "experiment_id": "genome_006_stim_resample",
        "n_sentences": n_sentences,
        "seeds": seeds,
        "total_wall_clock_seconds": round(time.time() - t0, 2),
        "per_seed_summary": per_seed_summary,
        "n_rows": len(all_rows),
        "g13_verdicts": {
            f"{sys}||{prim}||{est}": v
            for (sys, prim, est), v in verdicts.items()
        },
        "rows": all_rows,
    }
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=float)

    passed = sum(1 for v in verdicts.values() if v["status"] == "pass")
    total = len(verdicts)
    print(f"\n=== G1.3 STIM-RESAMPLE VERDICT ===")
    print(f"total (system, primitive, estimator) cells: {total}")
    print(f"G1.3 PASS: {passed} / {total}")
    print(f"out: {out_path}")
    return payload


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--n-sentences", type=int, default=500)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = ap.parse_args()
    run(n_sentences=args.n_sentences, max_length=args.max_length,
        seeds=args.seeds)
