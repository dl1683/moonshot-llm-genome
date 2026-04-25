"""
genome_130_trajectory_scaling_law.py

PYTHIA-2.8B TRAJECTORY + SCALING LAW FIT.

genome_127-129 established:
  - Trajectory U-shape (random=9.6 -> mode-collapse minimum -> recovery to ~4.2)
    is universal across Pythia 160m/410m/1.4b
  - Minimum step SHIFTS EARLIER with capacity:
      Pythia-160m: min @ step 512
      Pythia-410m: min @ step 512
      Pythia-1.4b: min @ step 128 (4x earlier than 160m at 9x capacity)
  - Asymptotic value ~4.243 reached by all sizes (final dev 0.9-5.6%)

Open question: is min_step ~ 1/N (parameter count)?
  160m -> 512
  410m -> 512 (NOT consistent with 1/N)
  1.4b -> 128

The 160m and 410m minimum-step alignment despite 2.5x capacity ratio is
suspicious — possibly we missed the true minimum (only sampled at step 128,
512, 1000). 410m might actually peak earlier than 512.

This experiment:
  1. Tests Pythia-2.8b across 9 checkpoints (extends to 17.5x capacity vs 160m)
  2. Adds finer early-step resolution at [32, 64, 256] for 2.8b
  3. Provides 4th data point for log-log fit of min_step vs N

Pre-stated PASS:
  - 2.8b trajectory matches U-shape qualitatively
  - 2.8b final-step value within 10% of 4.243
  - 4-point fit min_step ~ N^k has R^2 > 0.7

Results: results/genome_130_trajectory_scaling_law.json
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

PYTHIA_STEPS = [
    "step0", "step32", "step64", "step128", "step256",
    "step512", "step1000", "step4000", "step16000",
    "step64000", "step143000",
]
SYS_KEY = "pythia-2.8b"
HF_ID = "EleutherAI/pythia-2.8b"
TARGET = float(np.sqrt(18))

# Pythia parameter counts (approximate, total params)
PYTHIA_PARAM_COUNTS = {
    "pythia-160m": 162e6,
    "pythia-410m": 405e6,
    "pythia-1.4b": 1.4e9,
    "pythia-2.8b": 2.8e9,
}

# Min steps observed in g128/129 (these are sampled points; true min may be elsewhere)
G128_G129_MINIMA = {
    "pythia-160m": 512,
    "pythia-410m": 512,
    "pythia-1.4b": 128,
}


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


def measure_at_checkpoint(hf_id, revision, texts, sys_key, batch_size=2):
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
            seed=42, batch_size=batch_size, max_length=256,
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
    print(f"genome_130: Pythia-2.8b trajectory ({len(PYTHIA_STEPS)} checkpoints)")

    print("Loading c4_clean_v1 stimuli...")
    texts = []
    for rec in c4_clean_v1(seed=42, n_samples=800):
        texts.append(rec["text"])
    print(f"  N={len(texts)}")

    rows = []
    for revision in PYTHIA_STEPS:
        print(f"\n[{time.time()-t0:.1f}s] === {SYS_KEY} @ {revision} ===")
        r = measure_at_checkpoint(HF_ID, revision, texts, SYS_KEY)
        if "error" in r:
            print(f"  FAIL: {r['error']}")
            continue
        rows.append(r)
        print(f"  step={r['step']:6d}  alpha={r['alpha']:.3f}  er={r['eff_rank']:7.2f}  "
              f"sqrt(er)*alpha={r['sqrt_er_alpha']:.3f}")

    rows_sorted = sorted(rows, key=lambda x: x["step"])
    if not rows_sorted:
        print("no measurements")
        return
    steps = [r["step"] for r in rows_sorted]
    invs = [r["sqrt_er_alpha"] for r in rows_sorted]
    ers = [r["eff_rank"] for r in rows_sorted]
    min_idx = int(np.argmin(invs))
    final_dev = abs(invs[-1] - TARGET) / TARGET * 100

    print(f"\n=== Pythia-2.8b TRAJECTORY (target {TARGET:.3f}) ===")
    for st, inv, er in zip(steps, invs, ers):
        mark = " (above)" if inv >= TARGET else " (below)"
        if st == steps[min_idx]:
            mark += " [MIN]"
        print(f"  step {st:7d}: sqrt(er)*alpha={inv:.3f}  er={er:7.2f}{mark}")
    print(f"  minimum at step {steps[min_idx]}: {invs[min_idx]:.3f}")
    print(f"  final deviation: {final_dev:.1f}%")

    # Scaling law fit: min_step vs N
    minima = dict(G128_G129_MINIMA)
    minima[SYS_KEY] = steps[min_idx]
    sizes_kept = list(minima.keys())
    Ns = np.array([PYTHIA_PARAM_COUNTS[s] for s in sizes_kept])
    mins = np.array([minima[s] for s in sizes_kept])

    print(f"\n=== SCALING LAW FIT: min_step vs N ===")
    print(f"  Pythia-160m: N=1.62e8, min_step={minima['pythia-160m']}")
    print(f"  Pythia-410m: N=4.05e8, min_step={minima['pythia-410m']}")
    print(f"  Pythia-1.4b: N=1.4e9,  min_step={minima['pythia-1.4b']}")
    print(f"  Pythia-2.8b: N=2.8e9,  min_step={steps[min_idx]}")

    # log-log fit min_step = c * N^k -> log(min) = log(c) + k * log(N)
    log_n = np.log(Ns)
    log_m = np.log(mins)
    k_fit, log_c = np.polyfit(log_n, log_m, 1)
    c_fit = np.exp(log_c)
    pred = c_fit * Ns ** k_fit
    ss_res = np.sum((mins - pred) ** 2)
    ss_tot = np.sum((mins - mins.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"\n  fit: min_step = {c_fit:.2e} * N^{k_fit:.3f}")
    print(f"  R^2 = {r2:.4f}")
    print(f"  predictions:")
    for s, n, mp, mm in zip(sizes_kept, Ns, pred, mins):
        print(f"    {s}: pred={mp:.0f}  obs={mm:.0f}")

    # Verdict
    if final_dev < 10 and r2 > 0.7:
        verdict = (f"PASS: 2.8b converges (final_dev={final_dev:.1f}%) AND "
                   f"min_step ~ N^{k_fit:.2f} fits with R^2={r2:.3f}.")
    elif final_dev < 10:
        verdict = (f"PARTIAL: 2.8b converges (final_dev={final_dev:.1f}%) but "
                   f"scaling fit R^2={r2:.3f} < 0.7.")
    else:
        verdict = (f"KILL: 2.8b final_dev={final_dev:.1f}% > 10%.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 130, "name": "trajectory_scaling_law",
        "target_sqrt_er_alpha": TARGET,
        "checkpoints_tested": PYTHIA_STEPS,
        "rows": rows,
        "scaling_law_fit": {
            "k_exponent": float(k_fit),
            "c_prefactor": float(c_fit),
            "r_squared": float(r2),
            "sizes": sizes_kept,
            "param_counts": [float(x) for x in Ns],
            "min_steps_observed": [int(x) for x in mins],
            "min_steps_predicted": [float(x) for x in pred],
        },
        "summary": {
            "min_step": steps[min_idx],
            "min_value": invs[min_idx],
            "final_dev_pct": final_dev,
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_130_trajectory_scaling_law.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
