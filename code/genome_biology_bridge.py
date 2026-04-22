"""Candidate-8 spectral bridge on MOUSE V1 NEURONS.

Extends candidate-8 (`c ≈ eff_rank/d_rd` universal on 7/8 trained
artificial networks, genome_060/063/064) to biological neurons.
If the bridge also holds on Allen V1 Neuropixels recordings under
Natural Movie One, candidate-8 is an ML-independent finding about
LEARNING systems generally, not a backprop artifact.

Reuses infrastructure from `code/genome_biology_extractor.py` (session
enumeration, spike-count loading, stimulus-response cloud construction)
and `code/genome_svd_bridge_multimodel.py` (spectrum + eff_rank + alpha
+ d_rd + kNN power-law).

Measures: for each Allen V1 session, compute (c, ratio, alpha) on the
(stimulus-frame × neuron) firing-rate cloud. Target: ratio within 15%
of c on 2+ sessions, testing the bridge universally.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from genome_biology_extractor import (  # noqa: E402
    list_visual_coding_sessions,
    load_natural_movie_one_spike_counts,
    build_stimulus_response_cloud,
)
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def spectrum(X):
    if X.shape[0] < 2 or np.all(X == 0) or not np.all(np.isfinite(X)):
        return np.full(min(X.shape), np.nan)
    Xc = X - X.mean(axis=0)
    if np.all(Xc == 0):
        return np.full(min(X.shape), np.nan)
    return (np.linalg.svd(Xc, compute_uv=False) / np.sqrt(X.shape[0] - 1)).astype(np.float64)


def eff_rank_np(s):
    if not np.all(np.isfinite(s)):
        return float("nan")
    s2 = s ** 2
    total = s2.sum()
    return 0.0 if total <= 0 else float(total ** 2 / (s2 ** 2).sum())


def fit_alpha_tail(s, lo_frac=0.05, hi_frac=0.5):
    if not np.all(np.isfinite(s)):
        return float("nan")
    r = np.arange(1, len(s) + 1)
    lo = max(1, int(len(s) * lo_frac))
    hi = int(len(s) * hi_frac)
    if hi - lo < 3 or np.any(s[lo:hi] <= 0):
        return float("nan")
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    return float(-slope)


def analyze_cloud(X):
    s = spectrum(X)
    er = eff_rank_np(s)
    alpha = fit_alpha_tail(s)
    try:
        Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
        p, _, _ = fit_power_law(K_GRID, Cs)
        rd = rate_distortion_dim(X)
        c = p * rd["d_rd"]
        d_rd_val = float(rd["d_rd"])
        ratio = er / d_rd_val if d_rd_val > 0 else float("nan")
        rel_err = abs(ratio - c) / max(c, 1e-6) if np.isfinite(ratio) else float("nan")
    except Exception as e:
        p = float("nan"); d_rd_val = float("nan"); c = float("nan")
        ratio = float("nan"); rel_err = float("nan")
    return {"c": float(c), "p": float(p) if np.isfinite(p) else float("nan"),
            "d_rd": d_rd_val, "eff_rank": float(er), "alpha": float(alpha),
            "ratio": float(ratio),
            "rel_err_ratio_vs_c": float(rel_err) if np.isfinite(rel_err) else float("nan")}


def run_session(session_idx, n_neurons=50, t0=None):
    t0 = t0 if t0 is not None else time.time()
    print(f"[{time.time()-t0:.1f}s] enumerating Allen V1 sessions...")
    sessions = list_visual_coding_sessions(max_sessions=session_idx + 5)
    if session_idx >= len(sessions):
        print(f"ERROR: only {len(sessions)} sessions enumerated")
        return {"error": "session index out of range"}
    s = sessions[session_idx]
    print(f"[{time.time()-t0:.1f}s] streaming session {session_idx}: {s['path']}")
    data = load_natural_movie_one_spike_counts(s["asset_url"], n_neurons=n_neurons)
    print(f"[{time.time()-t0:.1f}s] loaded {data['n_units']} units, "
          f"{len(data['frame_onset_times'])} stimulus frames")
    cloud = build_stimulus_response_cloud(data)
    print(f"[{time.time()-t0:.1f}s] cloud shape: {cloud.shape}")
    r = analyze_cloud(cloud)
    r["session_idx"] = session_idx
    r["session_id"] = s.get("session_id", str(session_idx))
    r["n_units"] = int(data["n_units"])
    r["n_frames"] = int(len(data["frame_onset_times"]))
    print(f"[{time.time()-t0:.1f}s] c={r['c']:.3f}  ratio={r['ratio']:.3f}  "
          f"rel_err={r['rel_err_ratio_vs_c']:.3f}  alpha={r['alpha']:.3f}")
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--n-neurons", type=int, default=50)
    args = parser.parse_args()
    t0 = time.time()

    results = []
    for idx in args.sessions:
        try:
            r = run_session(idx, n_neurons=args.n_neurons, t0=t0)
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"session_idx": idx, "error": str(e)})

    print("\n=== CANDIDATE-8 BRIDGE ON MOUSE V1 ===")
    for r in results:
        if "error" in r:
            print(f"  session {r['session_idx']}: ERROR {r['error']}")
            continue
        tag = "PASS" if r["rel_err_ratio_vs_c"] < 0.15 else "FAIL"
        print(f"  session {r['session_idx']:3d} ({r.get('n_units','?')} units, "
              f"{r.get('n_frames','?')} frames):  "
              f"c={r['c']:.2f}  ratio={r['ratio']:.2f}  "
              f"rel_err={r['rel_err_ratio_vs_c']:.3f}  alpha={r['alpha']:.3f}  {tag}")

    out = {"purpose": "Candidate-8 bridge on Allen V1 Neuropixels",
           "n_neurons_per_session": args.n_neurons,
           "per_session": results}
    out_path = _ROOT / "results/gate2/candidate8_biology_bridge.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
