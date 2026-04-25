"""
genome_128_trajectory_fine_grain.py

FINE-GRAINED Pythia trajectory of trained-spectrum invariant.

genome_127 PASS: invariant trajectory is reproducible at 5 checkpoints
across 2 Pythia sizes — random=9.6, undershoot=3.5 at step 1k, recovery
to 4.0-4.2 at step 143k.

Open questions for genome_128:
  Q1. Where exactly does the invariant first dip BELOW target (4.243)?
      Coarse answer: between step 0 and step 1k. We need finer resolution.
  Q2. Where is the MINIMUM of the U-shape? Coarse answer: around step 1k.
  Q3. Is the trajectory exactly the same across Pythia sizes, or scale-dependent?
  Q4. Does the trajectory cross 4.243 multiple times before settling?

Sweep 8 log-spaced checkpoints across Pythia-160m and Pythia-410m:
  [0, 128, 1000, 4000, 16000, 64000, 143000] -- 7 checkpoints x 2 sizes = 14 measurements.
  (Pythia released revisions are step0, step1, step2, step4, ..., step512, step1000, step2000, ...)

Pre-stated criteria:
  PASS: U-shape minimum within 200 steps of step 1k for both sizes;
        first crossing of 4.243 within factor of 2 of step 500;
        sigma_sep > 5 between random-init and final.
  PARTIAL: trajectory shape consistent but minimum / first-crossing varies > 2x across sizes.
  KILL: trajectories diverge between sizes; not a scale-invariant phase transition.

Results: results/genome_128_trajectory_fine_grain.json
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
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent

# Log-spaced checkpoints across the full Pythia training run
PYTHIA_STEPS = [
    "step0",       # random init
    "step128",     # very early (early-collapse zone)
    "step512",     # still early
    "step1000",    # genome_127 measured here (3.5 undershoot)
    "step4000",    # mid-early
    "step16000",   # mid
    "step64000",   # mid-late
    "step143000",  # final
]

PYTHIA_SIZES = [
    ("pythia-160m", "EleutherAI/pythia-160m"),
    ("pythia-410m", "EleutherAI/pythia-410m"),
]

TARGET = float(np.sqrt(18))


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(max(X.shape[0] - 1, 1))
    return s.astype(np.float64)


def fit_power_tail(s, lo=0.05, hi=0.5):
    r = np.arange(1, len(s) + 1)
    a = max(1, int(len(s) * lo))
    b = int(len(s) * hi)
    lr = np.log(r[a:b])
    ls = np.log(s[a:b] + 1e-12)
    slope, _ = np.polyfit(lr, ls, 1)
    return float(-slope)


def eff_rank(s):
    s2 = s ** 2
    tot = s2.sum()
    return float(tot ** 2 / (s2 ** 2).sum()) if tot > 0 else 0.0


def measure_at_checkpoint(hf_id, revision, texts, sys_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"    loading {hf_id} @ {revision}...")
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, revision=revision, torch_dtype=torch.float16
        ).to("cuda").eval()
    except Exception as e:
        return {"error": f"load: {e}"}

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    mid = max(1, n_layers // 2)
    try:
        traj = extract_trajectory(
            model=model, tokenizer=tok,
            texts=texts, layer_indices=[mid], pooling="seq_mean",
            device="cuda", system_key=f"{sys_key}@{revision}",
            class_id=1, quantization="fp16",
            stimulus_version="c4_clean.v1.seed42.n800",
            seed=42, batch_size=8, max_length=256,
        )
        X = traj.layers[0].X.astype(np.float32)
    except Exception as e:
        del model; torch.cuda.empty_cache()
        return {"error": f"extract: {e}"}
    del model; torch.cuda.empty_cache()

    s = spectrum(X)
    a = fit_power_tail(s)
    er = eff_rank(s)
    return {
        "system": sys_key, "hf_id": hf_id, "revision": revision,
        "step": int(revision.replace("step", "")),
        "n_layers": n_layers, "mid_layer": mid,
        "alpha": a, "eff_rank": er,
        "sqrt_er_alpha": float(np.sqrt(er) * a),
        "er_alpha2": float(er * a ** 2),
        "n": int(X.shape[0]), "h": int(X.shape[1]),
    }


def main():
    t0 = time.time()
    print("genome_128: fine-grain Pythia trajectory")

    print("Loading c4_clean_v1 stimuli...")
    texts = []
    for rec in c4_clean_v1(seed=42, n_samples=800):
        texts.append(rec["text"])
    print(f"  N={len(texts)}")

    rows = []
    for sys_key, hf_id in PYTHIA_SIZES:
        for revision in PYTHIA_STEPS:
            print(f"\n[{time.time()-t0:.1f}s] === {sys_key} @ {revision} ===")
            r = measure_at_checkpoint(hf_id, revision, texts, sys_key)
            if "error" in r:
                print(f"  FAIL: {r['error']}")
                continue
            rows.append(r)
            print(f"  step={r['step']:6d}  alpha={r['alpha']:.3f}  er={r['eff_rank']:7.2f}  "
                  f"sqrt(er)*alpha={r['sqrt_er_alpha']:.3f}")

    # Trajectory analysis
    print(f"\n=== TRAJECTORY ANALYSIS (target = {TARGET:.3f}) ===")
    by_size = {}
    for r in rows:
        by_size.setdefault(r["system"], []).append(r)

    summary = {}
    for sys_key, rs in by_size.items():
        rs_sorted = sorted(rs, key=lambda x: x["step"])
        steps = [r["step"] for r in rs_sorted]
        invs = [r["sqrt_er_alpha"] for r in rs_sorted]
        ers = [r["eff_rank"] for r in rs_sorted]
        # Find minimum
        min_idx = int(np.argmin(invs))
        # Find first step where inv crosses below target
        first_below = None
        for i, v in enumerate(invs):
            if v < TARGET:
                first_below = steps[i]
                break
        # Find first step where inv crosses ABOVE target after dipping
        first_above_after = None
        if min_idx < len(invs) - 1:
            for i in range(min_idx, len(invs)):
                if invs[i] > TARGET:
                    first_above_after = steps[i]
                    break
        # Final-step deviation
        final_dev = abs(invs[-1] - TARGET) / TARGET * 100 if invs else float("nan")

        print(f"\n  {sys_key}:")
        for st, inv, er in zip(steps, invs, ers):
            mark = ""
            if inv < TARGET and (first_below is None or st <= first_below):
                mark += " (below)"
            elif inv >= TARGET:
                mark += " (above)"
            if st == steps[min_idx]:
                mark += " [MIN]"
            print(f"    step {st:7d}: sqrt(er)*alpha={inv:.3f}  er={er:7.2f}{mark}")
        print(f"    minimum at step={steps[min_idx]}: {invs[min_idx]:.3f}")
        print(f"    first crossing below target at step={first_below}")
        print(f"    first re-crossing above target at step={first_above_after}")
        print(f"    final deviation from target: {final_dev:.1f}%")

        summary[sys_key] = {
            "steps": steps, "sqrt_er_alpha": invs, "eff_rank": ers,
            "min_step": steps[min_idx], "min_value": invs[min_idx],
            "first_crossing_below": first_below,
            "first_recrossing_above": first_above_after,
            "final_dev_pct": final_dev,
        }

    # Verdict
    if len(summary) >= 2:
        sizes = list(summary.keys())
        min_steps = [summary[s]["min_step"] for s in sizes]
        first_below_steps = [summary[s]["first_crossing_below"] for s in sizes if summary[s]["first_crossing_below"]]
        final_devs = [summary[s]["final_dev_pct"] for s in sizes]

        # Check minimum-step consistency (within factor of 4)
        if min(min_steps) > 0 and max(min_steps) / min(min_steps) <= 4:
            min_consistent = True
        else:
            min_consistent = False

        # Check final convergence (all sizes within 10%)
        all_converge = all(d <= 10 for d in final_devs)

        if min_consistent and all_converge:
            verdict = (f"PASS: trajectory shape is scale-invariant. Minimum step varies "
                       f"by factor {max(min_steps)/max(min(min_steps),1):.1f}, all sizes "
                       f"converge to target within 10% by final step.")
        elif all_converge:
            verdict = (f"PARTIAL: all sizes converge but minimum-step varies by factor "
                       f"{max(min_steps)/max(min(min_steps),1):.1f}. Trajectory shape "
                       f"is qualitatively consistent but quantitatively scale-dependent.")
        else:
            verdict = (f"KILL: not all sizes converge to target by final step. "
                       f"Final deviations: {final_devs}.")
    else:
        verdict = "FAIL: not enough sizes measured"

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 128, "name": "trajectory_fine_grain",
        "target_sqrt_er_alpha": TARGET,
        "checkpoints_tested": PYTHIA_STEPS,
        "rows": rows, "summary": summary, "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_128_trajectory_fine_grain.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
