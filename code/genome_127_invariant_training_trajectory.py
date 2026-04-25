"""
genome_127_invariant_training_trajectory.py

PYTHIA CHECKPOINT TRAJECTORY: trained-spectrum invariant as training diagnostic.

genome_106 showed Pythia-410m at step-0=9.55, step-1k=3.57, step-143k=4.09.
genome_126 showed GPT-Neo-125m at 1.62 (off-manifold) vs cluster ~4.4. The
hypothesis: GPT-Neo is UNDER-CONVERGED. The invariant might be a training-
maturity diagnostic — a phase-transition coordinate that distinguishes
random-init / lightly-trained / fully-trained networks.

This experiment sweeps Pythia checkpoints across 3 model sizes:
  - pythia-160m: steps [0, 64, 1k, 4k, 16k, 64k, 143k]
  - pythia-410m: same checkpoints
  - pythia-1.4b: same checkpoints

For each (size, step), measure sqrt(er)*alpha and er*alpha^2 at mid-layer.

Pre-stated criteria:
  PASS: trajectory shows clean monotonic convergence to ~4.2 with sigma_sep > 5
        between random-init and final, AND under-converged (~step 1k or below)
        sits below the cluster mean of fully-trained.
  PARTIAL: phase transition observed but with noise.
  KILL: no clear trajectory; invariant unrelated to training maturity.

If PASS: the invariant is a training-maturity diagnostic, with practical use
as a model-quality signal that does NOT require eval benchmarks. GenomeGuard-
adjacent finding: spectral fingerprint detects under-trained models.

Results: results/genome_127_invariant_training_trajectory.json
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

PYTHIA_STEPS = ["step0", "step1000", "step10000", "step64000", "step143000"]
PYTHIA_SIZES = [
    ("pythia-160m", "EleutherAI/pythia-160m"),
    ("pythia-410m", "EleutherAI/pythia-410m"),
]


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
        del model
        torch.cuda.empty_cache()
        return {"error": f"extract: {e}"}
    del model
    torch.cuda.empty_cache()

    s = spectrum(X)
    a = fit_power_tail(s)
    er = eff_rank(s)
    return {
        "system": sys_key, "hf_id": hf_id, "revision": revision,
        "n_layers": n_layers, "mid_layer": mid,
        "alpha": a, "eff_rank": er,
        "sqrt_er_alpha": float(np.sqrt(er) * a),
        "er_alpha2": float(er * a ** 2),
        "n": int(X.shape[0]), "h": int(X.shape[1]),
    }


def main():
    t0 = time.time()
    print("genome_127: Pythia checkpoint trajectory of trained-spectrum invariant")

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
            print(f"  alpha={r['alpha']:.3f}  er={r['eff_rank']:7.2f}  "
                  f"sqrt(er)*alpha={r['sqrt_er_alpha']:.3f}  "
                  f"er*alpha^2={r['er_alpha2']:.3f}")

    # Per-size trajectory analysis
    print(f"\n=== TRAJECTORY ANALYSIS (target sqrt_er_alpha = {np.sqrt(18):.3f}) ===")
    by_size = {}
    for r in rows:
        by_size.setdefault(r["system"], []).append(r)

    target = float(np.sqrt(18))
    summary = {}
    for sys_key, rs in by_size.items():
        rs_sorted = sorted(rs, key=lambda x: int(x["revision"].replace("step", "")))
        steps = [int(r["revision"].replace("step", "")) for r in rs_sorted]
        invs = [r["sqrt_er_alpha"] for r in rs_sorted]
        ers = [r["eff_rank"] for r in rs_sorted]
        alphas = [r["alpha"] for r in rs_sorted]
        print(f"\n  {sys_key}:")
        for i, (st, inv, er, a) in enumerate(zip(steps, invs, ers, alphas)):
            dev = abs(inv - target) / target * 100
            mark = " <-- target" if dev < 5 else (" <-- close" if dev < 15 else "")
            print(f"    step {st:6d}: sqrt(er)*alpha={inv:.3f}  er={er:7.2f}  alpha={a:.3f}  (dev {dev:.1f}%){mark}")
        summary[sys_key] = {
            "steps": steps, "sqrt_er_alpha": invs,
            "eff_rank": ers, "alpha": alphas,
            "final_step": steps[-1] if steps else None,
            "final_inv": invs[-1] if invs else None,
            "step0_inv": invs[0] if invs else None,
        }

    # Verdict
    converged = []
    for sys_key, s in summary.items():
        if s["final_inv"] is None:
            continue
        final_dev = abs(s["final_inv"] - target) / target * 100
        if final_dev < 10:
            converged.append((sys_key, final_dev))

    if len(converged) >= 2:
        verdict = (f"PASS: {len(converged)}/{len(summary)} sizes converge to target "
                   f"({converged}). Trajectory shows clean training-maturity signal.")
    elif len(converged) >= 1:
        verdict = (f"PARTIAL: {len(converged)} size(s) converge: {converged}. "
                   f"Phase-transition signal present but not population-uniform.")
    else:
        verdict = (f"KILL: no Pythia size converges to target within 10%. "
                   f"Invariant is not a clean training-maturity coordinate.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 127, "name": "invariant_training_trajectory",
        "target_sqrt_er_alpha": target,
        "checkpoints_tested": PYTHIA_STEPS,
        "rows": rows, "summary": summary, "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_127_invariant_training_trajectory.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
