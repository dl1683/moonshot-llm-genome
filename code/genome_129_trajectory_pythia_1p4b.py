"""
genome_129_trajectory_pythia_1p4b.py

EXTEND TRAJECTORY TO PYTHIA-1.4B — scale-invariance test across 9x capacity.

genome_128 PASS: Pythia-160m and Pythia-410m (2.6x capacity ratio) had
IDENTICAL trajectory landmarks at every checkpoint. This experiment tests
whether the alignment holds at Pythia-1.4b (9x ratio relative to 160m).

If 1.4b shows the same minimum step (512), same first-crossing-below
(128), same first-recovery-above (4000), and converges to ~4.2 by step
143k, the scale-invariance claim becomes population-level (3 sizes,
9x capacity range).

Pre-stated PASS: Pythia-1.4b minimum step in [128, 1000], final step value
within 10% of 4.243, and first re-crossing above target in [1000, 16000].

Same 8 checkpoints as genome_128:
[0, 128, 512, 1000, 4000, 16000, 64000, 143000]

Results: results/genome_129_trajectory_pythia_1p4b.json
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
    "step0", "step128", "step512", "step1000",
    "step4000", "step16000", "step64000", "step143000",
]
SYS_KEY = "pythia-1.4b"
HF_ID = "EleutherAI/pythia-1.4b"
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
            seed=42, batch_size=4, max_length=256,  # smaller batch for 1.4b
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
    print(f"genome_129: Pythia-1.4b trajectory ({len(PYTHIA_STEPS)} checkpoints)")

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

    # Trajectory analysis
    print(f"\n=== TRAJECTORY ANALYSIS (target = {TARGET:.3f}) ===")
    rows_sorted = sorted(rows, key=lambda x: x["step"])
    steps = [r["step"] for r in rows_sorted]
    invs = [r["sqrt_er_alpha"] for r in rows_sorted]
    ers = [r["eff_rank"] for r in rows_sorted]
    if not invs:
        print("  no measurements collected")
        return

    min_idx = int(np.argmin(invs))
    first_below = next((s for s, v in zip(steps, invs) if v < TARGET), None)
    first_above_after = None
    for i in range(min_idx, len(invs)):
        if invs[i] > TARGET:
            first_above_after = steps[i]
            break
    final_dev = abs(invs[-1] - TARGET) / TARGET * 100

    print(f"\n  pythia-1.4b:")
    for st, inv, er in zip(steps, invs, ers):
        mark = " (above)" if inv >= TARGET else " (below)"
        if st == steps[min_idx]:
            mark += " [MIN]"
        print(f"    step {st:7d}: sqrt(er)*alpha={inv:.3f}  er={er:7.2f}{mark}")
    print(f"  minimum at step={steps[min_idx]}: {invs[min_idx]:.3f}")
    print(f"  first crossing below target at step={first_below}")
    print(f"  first re-crossing above target at step={first_above_after}")
    print(f"  final deviation from target: {final_dev:.1f}%")

    # Compare to genome_128 landmarks (Pythia-160m and -410m)
    g128_min_step = 512
    g128_first_below = 128
    g128_first_above_after = 4000

    pass_min = (steps[min_idx] in PYTHIA_STEPS_INTS_FOR_MATCH)
    pass_min_close = abs(steps[min_idx] - g128_min_step) <= max(g128_min_step // 2, 256)
    pass_first_below = first_below is not None and abs(first_below - g128_first_below) <= max(g128_first_below // 2, 64)
    pass_first_above = first_above_after is not None and abs(first_above_after - g128_first_above_after) <= max(g128_first_above_after // 2, 1000)
    pass_final = final_dev < 10

    landmarks_match = pass_min_close and pass_first_below and pass_first_above and pass_final
    if landmarks_match:
        verdict = (f"PASS: Pythia-1.4b matches genome_128 trajectory. "
                   f"Min@{steps[min_idx]} (g128: 512), first-below@{first_below} (g128: 128), "
                   f"first-above@{first_above_after} (g128: 4000), final_dev={final_dev:.1f}%. "
                   f"Scale-invariance holds across 9x capacity range.")
    elif pass_final:
        verdict = (f"PARTIAL: 1.4b converges to target (final_dev={final_dev:.1f}%) but "
                   f"trajectory landmarks differ from g128: "
                   f"min@{steps[min_idx]}, first-below@{first_below}, "
                   f"first-above@{first_above_after}.")
    else:
        verdict = (f"KILL: 1.4b final_dev={final_dev:.1f}% > 10%. Trajectory does not converge.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 129, "name": "trajectory_pythia_1p4b",
        "target_sqrt_er_alpha": TARGET,
        "checkpoints_tested": PYTHIA_STEPS,
        "rows": rows,
        "summary": {
            "steps": steps, "sqrt_er_alpha": invs, "eff_rank": ers,
            "min_step": steps[min_idx], "min_value": invs[min_idx],
            "first_crossing_below": first_below,
            "first_recrossing_above": first_above_after,
            "final_dev_pct": final_dev,
        },
        "g128_reference": {
            "min_step": g128_min_step,
            "first_below": g128_first_below,
            "first_above_after": g128_first_above_after,
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_129_trajectory_pythia_1p4b.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


PYTHIA_STEPS_INTS_FOR_MATCH = {int(s.replace("step", "")) for s in PYTHIA_STEPS}

if __name__ == "__main__":
    main()
