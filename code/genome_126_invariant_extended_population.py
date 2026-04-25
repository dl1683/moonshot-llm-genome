"""
genome_126_invariant_extended_population.py

EXTEND TRAINED-SPECTRUM INVARIANT TO LARGER POPULATION.

Codex direction Y: derive zero-fit predictions for eff_rank * alpha^2 = 18
on held-out architectures. Before deriving, validate the invariant scales:
genome_088 had N=5 text systems (mean 4.268, CV 5.09%). genome_095 has 1
held-out trained system (Falcon-H1 inv=4.192, er*a^2=17.57). DINOv2 / CLIP
extraction failed in genome_095 due to a NoneType bug.

This experiment uses a SELF-CONTAINED measurement (no genome_extractor dep)
and tests the invariant on a broader population of text systems:

  Already tested (genome_088 + 095):
    - Qwen3-0.6B, DeepSeek-R1-Distill-1.5B, BERT, RoBERTa, MiniLM, Falcon-H1

  New systems for extension:
    - Pythia-160M  (small autoregressive)
    - Pythia-410M  (mid autoregressive)
    - Qwen3-1.7B   (Qwen3 scale)
    - GPT-Neo-125M (GPT-style)
    - DistilBERT   (distilled MLM)
    - ALBERT-base-v2 (parameter-shared MLM)

Total target: N >= 12 text systems. Compute er, alpha, sqrt(er)*alpha,
er*alpha^2 at mid-depth on C4-style stimuli.

Pre-stated criteria for the invariant:
  PASS: N>=8, mean(sqrt_er_alpha) within 5% of 4.243, CV < 7%, 5 sigma+
        separation from random-init/shuffled controls.
  PARTIAL: CV 7-15% but mean still close to 4.243.
  KILL: CV > 15% or mean off by > 10% (invariant not universal).

If PASS, the er*alpha^2 = 18 invariant is established as a population-level
trained-text-LM constraint and merits the variational derivation pursuit.
If KILL, the genome_088 result was system-specific and we need a new
direction.

Results: results/genome_126_invariant_extended_population.json
"""

import json
import pathlib
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

N_STIM = 800
SEQ_LEN = 256

# (system_key, hf_id, is_causal_lm, mid_layer_index_or_None)
# When mid_layer_index is None, use n_hidden_layers // 2
SYSTEMS = [
    # Already-tested systems (re-measured for consistency check)
    ("qwen3-0.6b",                "Qwen/Qwen3-0.6B",                          True,  None),
    ("deepseek-r1-distill-1.5b",  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", True,  None),
    ("bert-base",                 "bert-base-uncased",                         False, None),
    ("roberta-base",              "FacebookAI/roberta-base",                   False, None),
    ("minilm-l6",                 "sentence-transformers/all-MiniLM-L6-v2",    False, None),
    # New systems
    ("pythia-160m",               "EleutherAI/pythia-160m",                    True,  None),
    ("pythia-410m",               "EleutherAI/pythia-410m",                    True,  None),
    ("gpt-neo-125m",              "EleutherAI/gpt-neo-125m",                   True,  None),
    ("distilbert",                "distilbert-base-uncased",                   False, None),
    ("albert-base-v2",            "albert-base-v2",                            False, None),
]


def load_c4_texts(n=N_STIM, seed=SEED):
    """Return list of N text strings from C4 (filtered for length)."""
    try:
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        rng = np.random.default_rng(seed)
        out = []
        seen = 0
        for rec in ds:
            text = rec["text"].strip()
            if len(text) < 200:
                continue
            out.append(text[:1500])
            seen += 1
            if seen >= n:
                break
        if len(out) >= n:
            return out
    except Exception as e:
        print(f"  c4 streaming failed: {e}; falling back to wikitext")
    # Fallback to wikitext
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    out = []
    for idx in perm:
        text = ds[int(idx)]["text"].strip()
        if len(text) < 200:
            continue
        out.append(text[:1500])
        if len(out) >= n:
            break
    return out


def load_model_and_tok(hf_id, is_causal_lm):
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.cls_token
    if is_causal_lm:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.float16, output_hidden_states=True
        )
    else:
        model = AutoModel.from_pretrained(
            hf_id, dtype=torch.float16, output_hidden_states=True
        )
    model = model.to(DEVICE).eval()
    return model, tok


def n_hidden_layers(model):
    """Return number of transformer layers, robustly across architectures."""
    cfg = model.config
    for name in ("num_hidden_layers", "n_layer", "n_layers"):
        if hasattr(cfg, name):
            return getattr(cfg, name)
    raise ValueError(f"can't find num_layers in config: {cfg}")


def collect_mid_layer_activations(model, tok, texts, batch_size=8):
    """Run model on texts, return seq-mean-pooled mid-layer hidden states.
       Returns array of shape (n_texts, hidden_dim)."""
    n_layers = n_hidden_layers(model)
    mid_layer = max(1, n_layers // 2)
    print(f"    n_layers={n_layers}, mid_layer={mid_layer}")

    all_pooled = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                   truncation=True, max_length=SEQ_LEN).to(DEVICE)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # hidden_states is tuple of (n_layers+1) tensors of shape (B, T, D)
        # index 0 = embeddings, 1..n_layers = layer outputs
        h = out.hidden_states[mid_layer]  # (B, T, D)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        pooled = (h.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_pooled.append(pooled.cpu().numpy())
    return np.concatenate(all_pooled, axis=0)


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


def shuffle_columns(X, seed=SEED):
    rng = np.random.default_rng(seed)
    Xs = X.copy()
    for j in range(Xs.shape[1]):
        rng.shuffle(Xs[:, j])
    return Xs


def main():
    t0 = time.time()
    print(f"genome_126: extended invariant population (target N>=12 text systems)")

    print("Loading C4 stimuli...")
    texts = load_c4_texts(N_STIM)
    print(f"  N={len(texts)}, mean length={np.mean([len(t) for t in texts]):.0f}")

    rows = []
    for sys_key, hf_id, is_causal, _ in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] === {sys_key} ({hf_id}) ===")
        try:
            model, tok = load_model_and_tok(hf_id, is_causal)
            X = collect_mid_layer_activations(model, tok, texts)
            print(f"    activation cloud: {X.shape}")
        except Exception as e:
            print(f"    FAIL: {e}")
            continue
        finally:
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass

        for cond, cloud in [("trained", X), ("shuffled", shuffle_columns(X))]:
            s = spectrum(cloud)
            alpha = fit_power_tail(s)
            er = eff_rank(s)
            inv = float(np.sqrt(er) * alpha)
            er_a2 = float(er * alpha ** 2)
            rows.append({
                "system": sys_key, "hf_id": hf_id, "condition": cond,
                "alpha": alpha, "eff_rank": er,
                "sqrt_er_alpha": inv, "er_alpha2": er_a2,
                "n": int(X.shape[0]), "h": int(X.shape[1]),
            })
            print(f"    {cond:9s}  alpha={alpha:.3f}  er={er:6.2f}  "
                  f"sqrt(er)*alpha={inv:.3f}  er*alpha^2={er_a2:.3f}")

    # Aggregate
    print("\n=== INVARIANT AGGREGATE ===")
    target_inv = float(np.sqrt(18))
    target_er_a2 = 18.0
    summary = {}
    for cond in ("trained", "shuffled"):
        vals_inv = [r["sqrt_er_alpha"] for r in rows if r["condition"] == cond]
        vals_er_a2 = [r["er_alpha2"] for r in rows if r["condition"] == cond]
        if not vals_inv:
            continue
        summary[cond] = {
            "N": len(vals_inv),
            "sqrt_er_alpha_mean": float(np.mean(vals_inv)),
            "sqrt_er_alpha_std":  float(np.std(vals_inv)),
            "sqrt_er_alpha_cv_pct": float(100 * np.std(vals_inv) / np.mean(vals_inv)) if vals_inv else 0,
            "er_alpha2_mean":     float(np.mean(vals_er_a2)),
            "er_alpha2_std":      float(np.std(vals_er_a2)),
            "er_alpha2_cv_pct":   float(100 * np.std(vals_er_a2) / np.mean(vals_er_a2)) if vals_er_a2 else 0,
        }
        s = summary[cond]
        print(f"  {cond:9s}  N={s['N']:2d}  "
              f"sqrt(er)*alpha={s['sqrt_er_alpha_mean']:.3f} +/- {s['sqrt_er_alpha_std']:.3f}  "
              f"CV={s['sqrt_er_alpha_cv_pct']:.2f}%   "
              f"er*a^2={s['er_alpha2_mean']:.2f} +/- {s['er_alpha2_std']:.2f}  "
              f"CV={s['er_alpha2_cv_pct']:.2f}%")

    # Verdict
    if "trained" in summary:
        t_mean = summary["trained"]["sqrt_er_alpha_mean"]
        t_cv = summary["trained"]["sqrt_er_alpha_cv_pct"]
        t_n = summary["trained"]["N"]
        deviation = abs(t_mean - target_inv) / target_inv * 100

        if "shuffled" in summary:
            sh_mean = summary["shuffled"]["sqrt_er_alpha_mean"]
            t_std = summary["trained"]["sqrt_er_alpha_std"]
            sigma_sep = abs(t_mean - sh_mean) / max(t_std, 1e-6)
        else:
            sigma_sep = float("nan")

        if t_n >= 8 and deviation < 5 and t_cv < 7 and sigma_sep > 5:
            verdict = (f"PASS: N={t_n} sqrt(er)*alpha={t_mean:.3f} "
                       f"({deviation:.1f}% from 3sqrt(2)), CV={t_cv:.2f}%, "
                       f"sigma_sep={sigma_sep:.1f}. Invariant scales to extended population.")
        elif deviation < 10 and t_cv < 15:
            verdict = (f"PARTIAL: N={t_n} mean={t_mean:.3f} "
                       f"(deviation {deviation:.1f}%, CV {t_cv:.2f}%). "
                       f"Invariant holds approximately but with looser tolerance.")
        else:
            verdict = (f"KILL: N={t_n} mean={t_mean:.3f} "
                       f"(deviation {deviation:.1f}%, CV {t_cv:.2f}%). "
                       f"genome_088 result was system-specific. Invariant does not generalize.")
    else:
        verdict = "FAIL: no trained measurements collected"

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 126, "name": "invariant_extended_population",
        "n_stim": N_STIM, "seq_len": SEQ_LEN,
        "target_sqrt_er_alpha": target_inv, "target_er_alpha2": target_er_a2,
        "rows": rows, "summary": summary, "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = RESULTS / "genome_126_invariant_extended_population.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
