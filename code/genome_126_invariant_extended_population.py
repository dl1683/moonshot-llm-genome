"""
genome_126_invariant_extended_population.py

EXTEND TRAINED-SPECTRUM INVARIANT TO LARGER POPULATION (N>=12 text systems).

Codex direction Y. Reuses genome_088's proven extraction protocol via
genome_loaders + genome_extractor (NOT reimplemented — earlier attempt
diverged at the cloud-statistics level despite ostensibly-identical hooks).

Adds 5 new text systems on top of genome_088's 5:
  - Pythia-160m, Pythia-410m, GPT-Neo-125m, DistilBERT, ALBERT-base-v2

Pre-stated criteria:
  PASS: N>=10, mean(sqrt_er_alpha) within 5% of 4.243, CV<7%, sigma_sep>5
  PARTIAL: deviation <10%, CV<15%
  KILL: doesn't generalize past genome_088's N=5

Results: results/genome_126_invariant_extended_population.json
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
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent

# Already in genome_088 — re-measure for consistency check + extend population
ALREADY_TESTED = [
    ("qwen3-0.6b",                   "Qwen/Qwen3-0.6B"),
    ("deepseek-r1-distill-qwen-1.5b","deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("bert-base-uncased",            "bert-base-uncased"),
    ("roberta-base",                 "FacebookAI/roberta-base"),
    ("minilm-l6-contrastive",        "sentence-transformers/all-MiniLM-L6-v2"),
]
NEW_SYSTEMS = [
    ("pythia-160m",                  "EleutherAI/pythia-160m"),
    ("pythia-410m",                  "EleutherAI/pythia-410m"),
    ("gpt-neo-125m",                 "EleutherAI/gpt-neo-125m"),
    ("distilbert-base-uncased",      "distilbert-base-uncased"),
    ("albert-base-v2",               "albert-base-v2"),
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


def measure(sys_key, hf_id, texts):
    """Returns dict with trained + shuffled rows, or None on failure."""
    try:
        sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    except Exception as e:
        return {"error": f"load: {e}"}
    mid = max(1, sys_obj.n_hidden_layers() // 2)
    try:
        traj = extract_trajectory(
            model=sys_obj.model, tokenizer=sys_obj.tokenizer,
            texts=texts, layer_indices=[mid], pooling="seq_mean",
            device="cuda", system_key=sys_key, class_id=1,
            quantization="fp16",
            stimulus_version="c4_clean.v1.seed42.n800",
            seed=42, batch_size=8, max_length=256,
        )
        X = traj.layers[0].X.astype(np.float32)
    except Exception as e:
        sys_obj.unload(); torch.cuda.empty_cache()
        return {"error": f"extract: {e}"}
    sys_obj.unload(); torch.cuda.empty_cache()

    rows = []
    s_tr = spectrum(X)
    rng = np.random.default_rng(42)
    Xs = X.copy()
    for j in range(Xs.shape[1]):
        rng.shuffle(Xs[:, j])
    s_sh = spectrum(Xs)
    for cond, s in [("trained", s_tr), ("shuffled", s_sh)]:
        a = fit_power_tail(s)
        er = eff_rank(s)
        rows.append({
            "system": sys_key, "hf_id": hf_id, "condition": cond,
            "alpha": a, "eff_rank": er,
            "sqrt_er_alpha": float(np.sqrt(er) * a),
            "er_alpha2": float(er * a ** 2),
            "n": int(X.shape[0]), "h": int(X.shape[1]),
        })
    return {"rows": rows}


def main():
    t0 = time.time()
    print("genome_126: extended invariant population (target N>=10)")

    print("Loading c4_clean_v1 stimuli (seed=42, n=800)...")
    texts = []
    for rec in c4_clean_v1(seed=42, n_samples=800):
        texts.append(rec["text"])
    print(f"  N={len(texts)}")

    rows = []
    all_systems = ALREADY_TESTED + NEW_SYSTEMS
    for sys_key, hf_id in all_systems:
        print(f"\n[{time.time()-t0:.1f}s] === {sys_key} ===")
        result = measure(sys_key, hf_id, texts)
        if "error" in result:
            print(f"  FAIL: {result['error']}")
            continue
        for r in result["rows"]:
            rows.append(r)
            print(f"  {r['condition']:9s}  alpha={r['alpha']:.3f}  er={r['eff_rank']:6.2f}  "
                  f"sqrt(er)*alpha={r['sqrt_er_alpha']:.3f}  er*alpha^2={r['er_alpha2']:.3f}")

    # Aggregate
    target = float(np.sqrt(18))
    print(f"\n=== AGGREGATE (target sqrt_er_alpha = {target:.3f}) ===")
    summary = {}
    for cond in ("trained", "shuffled"):
        vals = [r["sqrt_er_alpha"] for r in rows if r["condition"] == cond]
        if not vals:
            continue
        m = float(np.mean(vals))
        sd = float(np.std(vals))
        cv = 100 * sd / m if m else 0
        summary[cond] = {"N": len(vals), "mean": m, "std": sd, "cv_pct": cv}
        print(f"  {cond:9s}  N={len(vals):2d}  mean={m:.3f}  std={sd:.3f}  CV={cv:.2f}%")

    # Verdict
    if "trained" in summary:
        t = summary["trained"]
        dev_pct = abs(t["mean"] - target) / target * 100
        if "shuffled" in summary:
            sh_mean = summary["shuffled"]["mean"]
            sigma = abs(t["mean"] - sh_mean) / max(t["std"], 1e-6)
        else:
            sigma = float("nan")

        if t["N"] >= 10 and dev_pct < 5 and t["cv_pct"] < 7 and sigma > 5:
            verdict = (f"PASS: N={t['N']} mean={t['mean']:.3f} "
                       f"(deviation {dev_pct:.1f}%, CV {t['cv_pct']:.2f}%, "
                       f"sigma_sep={sigma:.1f}). Invariant scales — variational "
                       f"derivation worth pursuing.")
        elif dev_pct < 10 and t["cv_pct"] < 15:
            verdict = (f"PARTIAL: N={t['N']} mean={t['mean']:.3f} "
                       f"(deviation {dev_pct:.1f}%, CV {t['cv_pct']:.2f}%). "
                       f"Holds approximately.")
        else:
            verdict = (f"KILL: N={t['N']} mean={t['mean']:.3f} "
                       f"(deviation {dev_pct:.1f}%, CV {t['cv_pct']:.2f}%). "
                       f"Invariant doesn't scale past genome_088's N=5.")
    else:
        verdict = "FAIL: no trained measurements"

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 126, "name": "invariant_extended_population",
        "target_sqrt_er_alpha": target, "rows": rows, "summary": summary,
        "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_126_invariant_extended_population.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
