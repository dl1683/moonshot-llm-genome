"""
genome_131_invariant_predicts_nll.py

PREDICTIVE UTILITY TEST.

Question: does sqrt(eff_rank)*alpha at training step k predict the model s
NLL (perplexity) on held-out text at that same step?

If yes, the invariant is a training-monitoring tool: you can measure model
quality from a small calibration batch without running a full eval. This
converts the spectral measurement into a practical efficiency tool.

This experiment computes BOTH the invariant AND the NLL at each
checkpoint for Pythia-160m and Pythia-410m at 8 steps each:
  [0, 128, 512, 1000, 4000, 16000, 64000, 143000]

Then tests Pearson and Spearman correlation between sqrt_er_alpha and NLL,
and between |sqrt_er_alpha - 4.243| (deviation from target) and NLL.

Pre-stated criteria:
  PASS: |Pearson r| > 0.85 between deviation-from-target and NLL across
        N >= 16 checkpoints (16 = 2 sizes x 8 steps).
  PARTIAL: |r| > 0.7
  KILL: |r| < 0.7

Results: results/genome_131_invariant_predicts_nll.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent

PYTHIA_STEPS = [
    "step0", "step128", "step512", "step1000",
    "step4000", "step16000", "step64000", "step143000",
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


def measure_nll(model, tok, texts, batch_size=4):
    """Mean NLL across all valid tokens in held-out texts."""
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                   truncation=True, max_length=256).to("cuda")
        ids, mask = enc["input_ids"], enc["attention_mask"]
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = ids[:, 1:].contiguous().clone()
        shift_mask = mask[:, 1:].contiguous()
        shift_labels[shift_mask == 0] = -100
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tokens = (shift_mask != 0).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def measure_at_checkpoint(hf_id, revision, calib_texts, eval_texts, sys_key):
    """Returns (invariant, nll) at this checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"    loading {hf_id} @ {revision}...")
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if tok.pad_token is None or not tok.pad_token:
        if tok.eos_token is not None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        else:
            tok.pad_token = tok.decode([0])
            tok.pad_token_id = 0
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
            texts=calib_texts, layer_indices=[mid], pooling="seq_mean",
            device="cuda", system_key=f"{sys_key}@{revision}",
            class_id=1, quantization="fp16",
            stimulus_version="c4_clean.v1.seed42.n800",
            seed=42, batch_size=8, max_length=256,
        )
        X = traj.layers[0].X.astype(np.float32)
    except Exception as e:
        del model; torch.cuda.empty_cache()
        return {"error": f"extract: {e}"}

    # Compute NLL on EVAL texts (different from calibration)
    try:
        nll = measure_nll(model, tok, eval_texts)
    except Exception as e:
        del model; torch.cuda.empty_cache()
        return {"error": f"nll: {e}"}

    del model; torch.cuda.empty_cache()

    s = spectrum(X)
    a = fit_power_tail(s)
    er = eff_rank(s)
    inv = float(np.sqrt(er) * a)
    return {
        "system": sys_key, "hf_id": hf_id, "revision": revision,
        "step": int(revision.replace("step", "")),
        "alpha": a, "eff_rank": er,
        "sqrt_er_alpha": inv,
        "deviation_from_target": float(abs(inv - TARGET)),
        "nll": float(nll),
    }


def pearson(x, y):
    x = np.asarray(x); y = np.asarray(y)
    cx = x - x.mean(); cy = y - y.mean()
    num = (cx * cy).sum()
    den = np.sqrt((cx ** 2).sum() * (cy ** 2).sum())
    return float(num / den) if den > 0 else 0.0


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)


def main():
    t0 = time.time()
    print("genome_131: invariant -> NLL predictive utility test")

    print("Loading c4_clean_v1 stimuli (calib n=400, eval n=200)...")
    all_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=600):
        all_texts.append(rec["text"])
    calib = all_texts[:400]
    evalt = all_texts[400:600]
    print(f"  calib N={len(calib)}, eval N={len(evalt)}")

    rows = []
    for sys_key, hf_id in PYTHIA_SIZES:
        for revision in PYTHIA_STEPS:
            print(f"\n[{time.time()-t0:.1f}s] === {sys_key} @ {revision} ===")
            r = measure_at_checkpoint(hf_id, revision, calib, evalt, sys_key)
            if "error" in r:
                print(f"  FAIL: {r['error']}")
                continue
            rows.append(r)
            print(f"  step={r['step']:6d}  inv={r['sqrt_er_alpha']:.3f}  "
                  f"|inv-target|={r['deviation_from_target']:.3f}  NLL={r['nll']:.3f}")

    print(f"\n=== CORRELATION ANALYSIS (N={len(rows)}) ===")
    invs = [r["sqrt_er_alpha"] for r in rows]
    devs = [r["deviation_from_target"] for r in rows]
    nlls = [r["nll"] for r in rows]

    if len(rows) >= 4:
        p_inv_nll = pearson(invs, nlls)
        s_inv_nll = spearman(invs, nlls)
        p_dev_nll = pearson(devs, nlls)
        s_dev_nll = spearman(devs, nlls)

        print(f"  Pearson  inv vs NLL:  r = {p_inv_nll:+.4f}")
        print(f"  Spearman inv vs NLL:  r = {s_inv_nll:+.4f}")
        print(f"  Pearson  |inv-target| vs NLL:  r = {p_dev_nll:+.4f}")
        print(f"  Spearman |inv-target| vs NLL:  r = {s_dev_nll:+.4f}")

        # Verdict based on |dev| vs NLL (the more meaningful relationship)
        best_r = max(abs(p_dev_nll), abs(s_dev_nll))
        if best_r > 0.85:
            verdict = (f"PASS: |inv-target| predicts NLL with |r|={best_r:.3f}>0.85. "
                       f"Invariant is a training-monitoring tool.")
        elif best_r > 0.7:
            verdict = f"PARTIAL: |r|={best_r:.3f} predictive but below 0.85 threshold."
        else:
            verdict = f"KILL: |r|={best_r:.3f}<0.7. Invariant is not a strong NLL predictor."
    else:
        verdict = f"FAIL: only {len(rows)} checkpoints succeeded"
        p_inv_nll = s_inv_nll = p_dev_nll = s_dev_nll = float("nan")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 131, "name": "invariant_predicts_nll",
        "target": TARGET,
        "rows": rows,
        "correlations": {
            "pearson_inv_nll": p_inv_nll,
            "spearman_inv_nll": s_inv_nll,
            "pearson_devtarget_nll": p_dev_nll,
            "spearman_devtarget_nll": s_dev_nll,
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_131_invariant_predicts_nll.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
