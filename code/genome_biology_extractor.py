"""Gate-2 G2.5 biology bridge: extract kNN-10 clustering coefficient from
Allen Brain Observatory Visual Coding Neuropixels recordings under Natural
Movie One, for comparison against DINOv2 kNN-10 on matched frames.

Per `research/prereg/genome_knn_k10_biology_2026-04-21.md` §3-4 and
`research/derivations/knn_clustering_universality.md` §7 (LOCKED):

Per stimulus frame `s_i` in Natural Movie One (~900 frames, 30fps):
    x_i ∈ R^{N_neurons} = trial-averaged, z-scored firing-rate vector
                           for frame s_i, time-locked to onset + 50 ms window
Point cloud X = {x_i} of size n_stimuli × N_neurons.
Compute C(X, k=10) on this cloud.

Per CLAUDE.md §6: use `remfile + h5py + dandi` for Windows+Python 3.13
compatibility. Do NOT use `allensdk` (incompatible with Python 3.13).

This module provides:
  - `list_visual_coding_sessions()` — enumerate DANDI dandiset 000021 sessions
  - `load_natural_movie_one_spike_counts(session_url)` — stream-load one session's
    spike raster + stimulus-frame timestamps
  - `build_stimulus_response_cloud(spike_counts, frame_onsets, ...)` — construct
    the (n_stimuli, N_neurons) firing-rate point cloud per §3 of prereg
  - `biology_knn_k10(cloud)` — compute the atlas coordinate

Usage (smoke):
    python code/genome_biology_extractor.py --list-only
    python code/genome_biology_extractor.py --session 0 --n-neurons 50

Full G2.5 run is a separate chain script once this module is validated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


# -------------------- DANDI enumeration --------------------

_DANDISET_VISUAL_CODING = "000021"
# Natural Movie One = 30 s movie at 30 fps = ~900 unique frames; each frame
# repeated ~10 times across trials for averaging.
_NATURAL_MOVIE_ONE_STIMULUS_NAME = "natural_movie_one_presentations"


def list_visual_coding_sessions(max_sessions: int = 30) -> list[dict]:
    """Enumerate available sessions in Visual Coding Neuropixels dandiset.

    Returns a list of dicts: {session_id, asset_url, size_bytes, area_list}.
    Network call — caches client between uses within one Python process.
    """
    from dandi.dandiapi import DandiAPIClient
    client = DandiAPIClient()
    dandiset = client.get_dandiset(_DANDISET_VISUAL_CODING, "draft")

    sessions: list[dict] = []
    for idx, asset in enumerate(dandiset.get_assets()):
        if idx >= max_sessions:
            break
        if not asset.path.endswith(".nwb"):
            continue
        sessions.append({
            "session_id": asset.path.replace("/", "_").replace(".nwb", ""),
            "asset_url": asset.get_content_url(
                follow_redirects=1, strip_query=False),
            "size_bytes": asset.size,
            "path": asset.path,
        })
    return sessions


# -------------------- Spike-count loading --------------------

def load_natural_movie_one_spike_counts(session_url: str, *,
                                         n_neurons: int | None = None,
                                         session_cache_dir: Path | None = None
                                         ) -> dict[str, Any]:
    """Stream-load spike counts + Natural Movie One stimulus timestamps.

    Returns:
        {"spikes_by_unit": dict[unit_id, np.ndarray[float]] (spike times in s),
         "frame_onset_times": np.ndarray[float] (one per frame, seconds),
         "frame_indices": np.ndarray[int] (which movie frame index 0..N-1),
         "unit_ids": list,
         "unit_brain_area": dict[unit_id, str],
         "n_units": int}

    Uses remfile + h5py for memory-efficient remote reading.
    """
    import h5py
    import remfile

    # Remote file handle — streaming, no full download
    remote = remfile.File(session_url, verbose=False)
    f = h5py.File(remote, mode="r")

    try:
        # Intervals / stimulus timestamps
        stim_group_path = f"intervals/{_NATURAL_MOVIE_ONE_STIMULUS_NAME}"
        if stim_group_path not in f:
            # Older NWB layout — search under processing
            found = None
            for candidate in f.get("intervals", {}).keys():
                if "movie" in candidate.lower() and "one" in candidate.lower():
                    found = f"intervals/{candidate}"
                    break
            if found is None:
                raise KeyError(
                    f"Natural Movie One stimulus not found in {session_url} "
                    f"(looked for {stim_group_path} + movie+one pattern)")
            stim_group_path = found
        stim_group = f[stim_group_path]
        start_times = np.array(stim_group["start_time"])
        stop_times = np.array(stim_group["stop_time"])
        # Frame index: some NWBs store `frame` column, otherwise derive from order
        if "frame" in stim_group:
            frame_indices = np.array(stim_group["frame"]).astype(np.int64)
        else:
            # Assume sorted, 30 fps × 30 s = 900 unique frames
            frame_indices = np.arange(len(start_times)) % 900

        # Units (neurons) + spike times
        units_group = f["units"]
        spike_times_flat = np.array(units_group["spike_times"])
        spike_times_index = np.array(units_group["spike_times_index"])
        unit_ids = list(np.array(units_group["id"]))
        unit_brain_area = {}
        if "electrode_group" in units_group:
            # Map each unit to a brain area via electrode-group lookup
            eg_col = units_group["electrode_group"]
            # Best-effort — brain area may live in electrodes table
            pass  # Leave as TBD in the scaffold
        # spike_times_index[i] = cumulative index into spike_times_flat for unit i
        spikes_by_unit: dict[int, np.ndarray] = {}
        prev = 0
        for i, (uid, idx_end) in enumerate(zip(unit_ids, spike_times_index)):
            spikes_by_unit[int(uid)] = spike_times_flat[prev:int(idx_end)]
            prev = int(idx_end)

        # Optional: subsample neurons
        if n_neurons is not None and n_neurons < len(unit_ids):
            keep = sorted(unit_ids)[:n_neurons]
            spikes_by_unit = {int(u): spikes_by_unit[int(u)] for u in keep}
            unit_ids = keep

        return {
            "spikes_by_unit": spikes_by_unit,
            "frame_onset_times": start_times,
            "frame_stop_times": stop_times,
            "frame_indices": frame_indices,
            "unit_ids": [int(u) for u in unit_ids],
            "unit_brain_area": unit_brain_area,
            "n_units": len(unit_ids),
        }
    finally:
        f.close()
        remote.close()


# -------------------- Point-cloud construction --------------------

def build_stimulus_response_cloud(data: dict[str, Any], *,
                                   integration_window_s: float = 0.05,
                                   n_frames: int = 900,
                                   z_score: bool = True) -> np.ndarray:
    """Build the stimulus-indexed firing-rate point cloud per prereg §4.

    For each of n_frames unique movie frames:
      - find every trial presentation of that frame
      - count spikes per neuron in [onset, onset + integration_window_s]
      - average across trials to get trial-averaged firing rate
    Yields X ∈ R^{n_frames × N_neurons}.

    z_score=True subtracts per-neuron mean + divides per-neuron std — matches
    prereg §4 "trial-averaged, z-scored firing rate vector." Prevents a few
    high-FR neurons from dominating the kNN graph geometry.
    """
    spikes_by_unit = data["spikes_by_unit"]
    frame_onsets = data["frame_onset_times"]
    frame_indices = data["frame_indices"]
    unit_ids = data["unit_ids"]

    N_neurons = len(unit_ids)
    cloud = np.zeros((n_frames, N_neurons), dtype=np.float64)
    trial_counts = np.zeros(n_frames, dtype=np.int64)

    # Build per-frame-index list of trial-onset times
    frame_to_trials: dict[int, list[float]] = {}
    for onset, fidx in zip(frame_onsets, frame_indices):
        fidx = int(fidx) % n_frames
        frame_to_trials.setdefault(fidx, []).append(float(onset))

    # Accumulate firing rate per (frame, neuron) over all trials
    for frame_idx, trial_onsets in frame_to_trials.items():
        if frame_idx >= n_frames:
            continue
        trial_counts[frame_idx] = len(trial_onsets)
        for n_i, uid in enumerate(unit_ids):
            spikes = spikes_by_unit[uid]
            # Count spikes in [onset, onset + window] for each trial
            total = 0
            for onset in trial_onsets:
                start, end = onset, onset + integration_window_s
                # Binary search for speed
                lo = np.searchsorted(spikes, start, side="left")
                hi = np.searchsorted(spikes, end, side="right")
                total += (hi - lo)
            if len(trial_onsets) > 0:
                cloud[frame_idx, n_i] = total / (len(trial_onsets) * integration_window_s)

    # Drop frames with zero trials (missing data)
    keep = trial_counts > 0
    cloud = cloud[keep]

    if z_score:
        mean = cloud.mean(axis=0, keepdims=True)
        std = cloud.std(axis=0, keepdims=True) + 1e-6
        cloud = (cloud - mean) / std

    return cloud


# -------------------- kNN-10 on biology cloud --------------------

def biology_knn_k10(cloud: np.ndarray) -> dict[str, float]:
    """Compute the atlas coordinate on the biology point cloud."""
    from genome_primitives import knn_clustering_coefficient
    m = knn_clustering_coefficient(cloud, k=10)
    return {
        "knn_k10_value": float(m.value),
        "knn_k10_se": float(m.se),
        "n_points": int(cloud.shape[0]),
        "n_neurons": int(cloud.shape[1]),
    }


# -------------------- Smoke CLI --------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-only", action="store_true",
                    help="just enumerate DANDI sessions, don't download anything")
    ap.add_argument("--session", type=int, default=0,
                    help="0-indexed into the session list")
    ap.add_argument("--n-neurons", type=int, default=50,
                    help="subsample to first N neurons for smoke speed")
    ap.add_argument("--max-sessions", type=int, default=5,
                    help="max sessions to list / try")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] enumerating DANDI dandiset {_DANDISET_VISUAL_CODING}...")
    sessions = list_visual_coding_sessions(max_sessions=args.max_sessions)
    print(f"[{time.time()-t0:.1f}s] {len(sessions)} sessions found")
    for i, s in enumerate(sessions[:args.max_sessions]):
        print(f"  [{i}] {s['path']}  ({s['size_bytes']/1e9:.1f} GB)")

    if args.list_only:
        out_path = _THIS_DIR.parent / "results" / "gate2" / "biology_session_list.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"sessions": sessions[:args.max_sessions],
                       "max_sessions": args.max_sessions}, f, indent=2)
        print(f"wrote: {out_path}")
        sys.exit(0)

    if args.session >= len(sessions):
        print(f"ERROR: session index {args.session} out of range (0..{len(sessions)-1})")
        sys.exit(1)

    s = sessions[args.session]
    print(f"[{time.time()-t0:.1f}s] streaming session {args.session}: {s['path']}")
    print(f"  asset URL: {s['asset_url'][:80]}...")
    data = load_natural_movie_one_spike_counts(
        s["asset_url"], n_neurons=args.n_neurons)
    print(f"[{time.time()-t0:.1f}s] loaded {data['n_units']} units, "
          f"{len(data['frame_onset_times'])} stimulus frames")

    print(f"[{time.time()-t0:.1f}s] building stimulus-response cloud...")
    cloud = build_stimulus_response_cloud(data)
    print(f"[{time.time()-t0:.1f}s] cloud shape: {cloud.shape}")

    print(f"[{time.time()-t0:.1f}s] computing kNN-10 clustering...")
    result = biology_knn_k10(cloud)
    print(f"[{time.time()-t0:.1f}s] biology kNN-k10 = {result['knn_k10_value']:.4f} "
          f"(SE {result['knn_k10_se']:.4f}, n={result['n_points']} stimuli, "
          f"d={result['n_neurons']} neurons)")
    print(f"DINOv2 mid-depth kNN-k10 reference: ~0.30-0.35 (from genome_007)")
    print(f"Compare: if biology value is also ~0.30-0.35, G2.5 passes at "
          f"delta=0.10 tolerance. Preliminary only — full test requires "
          f"matched stimuli + 30+ sessions + shuffle controls.")
