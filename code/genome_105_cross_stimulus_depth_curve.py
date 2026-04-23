"""Genome_105: cross-stimulus depth-curve universality test.

Genome_104 established f(normalized_depth) is a shared curve across 5 systems
on C4. Does the same curve hold on OTHER stimulus distributions, or is it
C4-specific?

Test: Qwen3-0.6B + BERT + DeepSeek on 3 stimuli:
  - C4 clean (baseline, natural web text)
  - wikitext-103 (different natural-text distribution)
  - scrambled-C4 (shuffled-word C4, destroys syntax, keeps vocabulary)

For each (system, stimulus), compute per-layer invariant. Overlay curves.

Outcomes:
 A. Curves overlay → f(depth) is architecturally-intrinsic, stimulus-invariant
 B. Curves overlay for natural-text but break for scrambled → f(depth) is
    'the shared response to well-formed text'
 C. Curves all different → stimulus is the dominant factor

Budget: 3 models x 3 stimuli = 9 extractions, ~1-2 min each = ~15 min.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1, wikitext_v1  # noqa: E402

_ROOT = _THIS_DIR.parent

SYSTEMS = [
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    ("bert-base-uncased", "bert-base-uncased"),
    ("deepseek-r1-distill-qwen-1.5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
]


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(max(X.shape[0] - 1, 1))
    return s.astype(np.float64)


def stats(s):
    s2 = s ** 2
    er = float(s2.sum() ** 2 / (s2 ** 2).sum()) if s2.sum() > 0 else 0.0
    h = len(s)
    r = np.arange(1, h + 1)
    lo, hi = max(1, int(h * 0.05)), int(h * 0.5)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    alpha = float(-slope)
    return float(np.sqrt(er) * alpha)


def load_stimuli():
    stims = {}
    # C4
    c4 = []
    for r in c4_clean_v1(seed=42, n_samples=3000):
        c4.append(r["text"])
        if len(c4) >= 800: break
    stims["c4"] = c4
    # Wikitext
    try:
        wt = []
        for r in wikitext_v1(seed=42, n_samples=3000):
            wt.append(r["text"])
            if len(wt) >= 800: break
        stims["wikitext"] = wt
    except Exception as e:
        print(f"  wikitext load fail: {e}")
    # Scrambled C4: shuffle word order in each sentence
    rng = np.random.default_rng(42)
    scr = []
    for txt in c4[:800]:
        words = txt.split()
        rng.shuffle(words)
        scr.append(" ".join(words))
    stims["c4_scrambled"] = scr
    return stims


def main():
    t0 = time.time()
    stims = load_stimuli()
    print(f"[{time.time()-t0:.1f}s] stimuli loaded: {list(stims.keys())}")

    # Common normalized-depth grid for interpolation
    grid = np.linspace(0.0, 1.0, 21)

    # traces[sys][stim] = per-layer invariant array interpolated onto grid
    traces = {}
    for label, hf_id in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {label} =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
        n_layers = sys_obj.n_hidden_layers()
        layer_indices = list(range(n_layers))
        traces[label] = {}
        for stim_name, texts in stims.items():
            try:
                traj = extract_trajectory(
                    model=sys_obj.model, tokenizer=sys_obj.tokenizer,
                    texts=texts, layer_indices=layer_indices, pooling="seq_mean",
                    device="cuda", system_key=f"{label}_{stim_name}", class_id=1,
                    quantization="fp16",
                    stimulus_version=f"{stim_name}.v1.seed42.n800",
                    seed=42, batch_size=16, max_length=256,
                )
            except Exception as e:
                print(f"  FAIL {stim_name}: {e}"); continue
            per_layer = np.array([stats(spectrum(traj.layers[L].X.astype(np.float32)))
                                    for L in layer_indices])
            depths = np.arange(n_layers) / max(n_layers - 1, 1)
            interp = np.interp(grid, depths, per_layer)
            traces[label][stim_name] = interp.tolist()
            # report a few key bins
            idx_mid = np.argmin(np.abs(grid - 0.5))
            idx_upper = np.argmin(np.abs(grid - 0.75))
            print(f"  {stim_name:15s}  depth0.5={interp[idx_mid]:.3f}  "
                  f"depth0.75={interp[idx_upper]:.3f}")
        sys_obj.unload(); torch.cuda.empty_cache()

    # Analysis: per (stim, depth), CV across systems
    stims_avail = set()
    for sys_key in traces:
        stims_avail.update(traces[sys_key].keys())
    print(f"\n\n=== CV across 3 SYSTEMS per (stimulus, depth) ===")
    for stim in sorted(stims_avail):
        print(f"\n--- stimulus: {stim} ---")
        print(f"  {'depth':>6} {'mean':>7} {'CV%':>6}")
        arrs = [traces[s][stim] for s in traces if stim in traces[s]]
        if len(arrs) < 2: continue
        arrs = np.array(arrs)
        for i, d in enumerate(grid):
            vals = arrs[:, i]
            m, s_ = float(np.mean(vals)), float(np.std(vals))
            cv = 100*s_/m if m else 0
            if 0.3 <= d <= 0.85:
                mark = "  <-- TIGHT" if 0 < cv < 10 else ""
                print(f"  {d:>6.2f} {m:>7.3f} {cv:>5.2f}%{mark}")

    # Analysis: per (system, depth), CV across stimuli
    print(f"\n\n=== CV across 3 STIMULI per (system, depth) ===")
    for sys_key in traces:
        if len(traces[sys_key]) < 2: continue
        print(f"\n--- system: {sys_key} ---")
        arrs = np.array([traces[sys_key][stim] for stim in sorted(traces[sys_key].keys())])
        for i, d in enumerate(grid):
            if not (0.3 <= d <= 0.85): continue
            vals = arrs[:, i]
            m, s_ = float(np.mean(vals)), float(np.std(vals))
            cv = 100*s_/m if m else 0
            mark = "  <-- TIGHT" if 0 < cv < 10 else ""
            print(f"  {d:>6.2f} {m:>7.3f} {cv:>5.2f}%{mark}")

    out = {"traces": traces, "grid": grid.tolist()}
    out_path = _ROOT / "results/gate2/cross_stimulus_depth.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
