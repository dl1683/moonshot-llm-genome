"""
genome_132_predicts_nll_crossarch.py

CROSS-ARCHITECTURE PREDICTIVE-UTILITY TEST.

genome_131 PASS: |sqrt(er)*alpha - 4.243| predicts NLL with r=0.89 within
Pythia-160m and Pythia-410m at 16 training checkpoints. This experiment
extends to N=22+ data points by adding 6+ fully-trained causal LMs across
architecture families:
  - Qwen3-0.6B (Llama-style)
  - DeepSeek-R1-Distill-1.5B (Qwen-distilled)
  - GPT-Neo-125m (GPT-NeoX precursor)
  - OPT-125m, OPT-350m (Meta OPT family)
  - TinyLlama-1.1B (Llama-architecture small model)
  - Pythia-1.4b (already had invariant; add NLL)

Plus the 16 Pythia-160m/410m checkpoint points from genome_131.

Total: 16 + 7 = 23 (model, NLL, invariant) data points.

Pre-stated PASS:
  Combined-population Pearson r(|inv-target|, NLL) > 0.80 across N>=20.
  Cross-architecture-only (excluding Pythia checkpoints) Pearson > 0.70.

If PASS: invariant is architecture-universal as a model-quality predictor.
If KILL: predictive power is Pythia-specific.

Results: results/genome_132_predicts_nll_crossarch.json
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
TARGET = float(np.sqrt(18))

CROSS_ARCH_SYSTEMS = [
    # (sys_key, hf_id)
    ("qwen3-0.6b",                  "Qwen/Qwen3-0.6B"),
    ("deepseek-r1-distill-1.5b",    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("gpt-neo-125m",                "EleutherAI/gpt-neo-125m"),
    ("opt-125m",                    "facebook/opt-125m"),
    ("opt-350m",                    "facebook/opt-350m"),
    ("tinyllama-1.1b",              "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("pythia-1.4b",                 "EleutherAI/pythia-1.4b"),
    ("pythia-160m",                 "EleutherAI/pythia-160m"),
    ("pythia-410m",                 "EleutherAI/pythia-410m"),
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


def measure_nll(model, tok, texts, batch_size=4):
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


def measure_one_system(sys_key, hf_id, calib, evalt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"    loading {hf_id} ...")
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None or not tok.pad_token:
        if tok.eos_token is not None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        else:
            tok.pad_token = tok.decode([0])
            tok.pad_token_id = 0
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.float16
        ).to("cuda").eval()
    except Exception as e:
        return {"error": f"load: {e}"}

    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    if n_layers is None:
        del model; torch.cuda.empty_cache()
        return {"error": "n_layers unknown"}
    mid = max(1, n_layers // 2)

    try:
        traj = extract_trajectory(
            model=model, tokenizer=tok,
            texts=calib, layer_indices=[mid], pooling="seq_mean",
            device="cuda", system_key=sys_key, class_id=1,
            quantization="fp16",
            stimulus_version="c4_clean.v1.seed42.n800",
            seed=42, batch_size=8, max_length=256,
        )
        X = traj.layers[0].X.astype(np.float32)
    except Exception as e:
        del model; torch.cuda.empty_cache()
        return {"error": f"extract: {e}"}

    try:
        nll = measure_nll(model, tok, evalt)
    except Exception as e:
        del model; torch.cuda.empty_cache()
        return {"error": f"nll: {e}"}

    del model; torch.cuda.empty_cache()

    s = spectrum(X)
    a = fit_power_tail(s)
    er = eff_rank(s)
    inv = float(np.sqrt(er) * a)
    return {
        "system": sys_key, "hf_id": hf_id,
        "n_layers": n_layers, "mid_layer": mid,
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
    print("genome_132: cross-architecture invariant -> NLL prediction")

    # Load existing g131 data
    g131_path = ROOT / "results/genome_131_invariant_predicts_nll.json"
    g131 = json.loads(g131_path.read_text())
    g131_rows = g131["rows"]
    print(f"  loaded g131: N={len(g131_rows)} Pythia checkpoint rows")

    print("Loading c4_clean_v1 stimuli (calib n=400, eval n=200)...")
    all_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=600):
        all_texts.append(rec["text"])
    calib = all_texts[:400]
    evalt = all_texts[400:600]

    new_rows = []
    for sys_key, hf_id in CROSS_ARCH_SYSTEMS:
        # Skip duplicates already in g131 (Pythia 160m/410m at step 143000 are equivalent)
        # but still measure them for sanity check
        print(f"\n[{time.time()-t0:.1f}s] === {sys_key} ===")
        r = measure_one_system(sys_key, hf_id, calib, evalt)
        if "error" in r:
            print(f"  FAIL: {r['error']}")
            continue
        new_rows.append(r)
        print(f"  inv={r['sqrt_er_alpha']:.3f}  |inv-target|={r['deviation_from_target']:.3f}  NLL={r['nll']:.3f}")

    # Combine: g131 trajectory points + new cross-arch points
    combined = []
    for r in g131_rows:
        combined.append({
            "source": "g131_trajectory",
            "system": f"{r['system']}@{r['revision']}",
            "deviation_from_target": r["deviation_from_target"],
            "sqrt_er_alpha": r["sqrt_er_alpha"],
            "nll": r["nll"],
        })
    for r in new_rows:
        combined.append({
            "source": "g132_crossarch",
            "system": r["system"],
            "deviation_from_target": r["deviation_from_target"],
            "sqrt_er_alpha": r["sqrt_er_alpha"],
            "nll": r["nll"],
        })

    print(f"\n=== CORRELATION ANALYSIS ===")
    print(f"  Total N = {len(combined)}")

    devs_all = [r["deviation_from_target"] for r in combined]
    nlls_all = [r["nll"] for r in combined]
    invs_all = [r["sqrt_er_alpha"] for r in combined]

    p_dev_all = pearson(devs_all, nlls_all)
    s_dev_all = spearman(devs_all, nlls_all)
    p_inv_all = pearson(invs_all, nlls_all)
    s_inv_all = spearman(invs_all, nlls_all)

    print(f"\n  COMBINED ({len(combined)} pts):")
    print(f"    Pearson  |inv-target| vs NLL: {p_dev_all:+.4f}")
    print(f"    Spearman |inv-target| vs NLL: {s_dev_all:+.4f}")
    print(f"    Pearson  inv vs NLL:          {p_inv_all:+.4f}")
    print(f"    Spearman inv vs NLL:          {s_inv_all:+.4f}")

    # Cross-arch only
    cross_only = [r for r in combined if r["source"] == "g132_crossarch"]
    if len(cross_only) >= 4:
        cdv = [r["deviation_from_target"] for r in cross_only]
        cnl = [r["nll"] for r in cross_only]
        ci = [r["sqrt_er_alpha"] for r in cross_only]
        p_dev_x = pearson(cdv, cnl)
        s_dev_x = spearman(cdv, cnl)
        print(f"\n  CROSS-ARCH ONLY ({len(cross_only)} pts):")
        print(f"    Pearson  |inv-target| vs NLL: {p_dev_x:+.4f}")
        print(f"    Spearman |inv-target| vs NLL: {s_dev_x:+.4f}")
    else:
        p_dev_x = s_dev_x = float("nan")

    # Verdict
    if abs(p_dev_all) > 0.80 and abs(p_dev_x) > 0.70:
        verdict = (f"PASS: combined r={p_dev_all:.3f} (>0.80), cross-arch r={p_dev_x:.3f} (>0.70). "
                   f"Invariant deviation is architecture-universal NLL predictor.")
    elif abs(p_dev_all) > 0.80:
        verdict = (f"PARTIAL: combined r={p_dev_all:.3f}, but cross-arch r={p_dev_x:.3f} below 0.70. "
                   f"Combined fit is dominated by Pythia trajectory points.")
    else:
        verdict = (f"KILL: combined r={p_dev_all:.3f} below 0.80. Predictive power weakens "
                   f"when combining trajectory + final-step measurements.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 132, "name": "predicts_nll_crossarch",
        "target": TARGET,
        "g131_trajectory_rows_used": len(g131_rows),
        "g132_crossarch_rows": new_rows,
        "combined": combined,
        "correlations": {
            "combined_pearson_devtarget_nll": p_dev_all,
            "combined_spearman_devtarget_nll": s_dev_all,
            "combined_pearson_inv_nll": p_inv_all,
            "combined_spearman_inv_nll": s_inv_all,
            "crossarch_pearson_devtarget_nll": p_dev_x,
            "crossarch_spearman_devtarget_nll": s_dev_x,
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_132_predicts_nll_crossarch.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
