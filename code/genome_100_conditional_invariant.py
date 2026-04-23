"""Adversarial #3: invariant on topic-homogeneous (conditional) vs unconditional text.

Codex: atlas works only on fully-destroyed models; unconditional spectra
may track a distribution prior rather than context-conditional computation.

Test: extract Qwen3 + BERT mid-depth activations on:
 - unconditional C4 (baseline, 800 sentences)
 - pseudo-conditional subsets: split C4 into 4 topic clusters via TF-IDF
   k-means on the sentences themselves, then take per-cluster activation
   cloud separately

Compute sqrt(er)*alpha on each. If the invariant is identical across
conditional subsets and unconditional, invariant is genuinely representation-
intrinsic. If it SHIFTS when the stimulus becomes topic-homogeneous, the
'trained attractor' is partly an artifact of marginal stimulus distribution.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


SYSTEMS = [
    ("qwen3-0.6b",       "Qwen/Qwen3-0.6B"),
    ("bert-base-uncased","bert-base-uncased"),
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
    return {"eff_rank": er, "alpha": alpha,
            "sqrt_er_alpha": float(np.sqrt(er) * alpha)}


def main():
    t0 = time.time()
    # Pull LOTS of sentences so we can subset by topic
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=8000):
        sents.append(rec["text"])
        if len(sents) >= 3200:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} sentences loaded")

    # Topic-cluster via TF-IDF + KMeans
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = vec.fit_transform(sents)
    K = 4
    km = MiniBatchKMeans(n_clusters=K, n_init=5, random_state=42, batch_size=512)
    labels = km.fit_predict(X_tfidf)
    print(f"[{time.time()-t0:.1f}s] clustered into {K} topics: sizes={[int(np.sum(labels==i)) for i in range(K)]}")

    # Build subsets: cluster-homogeneous (up to 800 sentences each) + unconditional
    subsets = {"unconditional": sents[:800]}
    for c in range(K):
        idx = np.where(labels == c)[0]
        np.random.default_rng(42).shuffle(idx)
        take = [sents[i] for i in idx[:800]]
        subsets[f"topic_{c}"] = take
        print(f"  topic_{c}: n={len(take)}")

    rows = []
    for sys_key, hf_id in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {sys_key} =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
        mid = max(1, sys_obj.n_hidden_layers() // 2)
        for subset_name, texts in subsets.items():
            try:
                traj = extract_trajectory(
                    model=sys_obj.model, tokenizer=sys_obj.tokenizer,
                    texts=texts, layer_indices=[mid], pooling="seq_mean",
                    device="cuda", system_key=f"{sys_key}_{subset_name}", class_id=1,
                    quantization="fp16",
                    stimulus_version=f"c4_{subset_name}.v1.seed42.n{len(texts)}",
                    seed=42, batch_size=16, max_length=256,
                )
                X = traj.layers[0].X.astype(np.float32)
            except Exception as e:
                print(f"  FAIL extract {subset_name}: {e}"); continue
            s = spectrum(X)
            st = stats(s)
            rows.append({"system": sys_key, "subset": subset_name,
                         "n": int(X.shape[0]), "h": int(X.shape[1]), **st})
            print(f"  {subset_name:15s}  eff_rank={st['eff_rank']:6.2f}  "
                  f"alpha={st['alpha']:.3f}  sqrt(er)*a={st['sqrt_er_alpha']:.3f}")
        sys_obj.unload(); torch.cuda.empty_cache()

    print("\n\n=== COMPARISON: per-system invariant across topic subsets ===")
    for sys_key, _ in SYSTEMS:
        sys_rows = [r for r in rows if r["system"] == sys_key]
        if not sys_rows: continue
        invs = [r["sqrt_er_alpha"] for r in sys_rows]
        m, s = float(np.mean(invs)), float(np.std(invs))
        cv = 100*s/m if m else 0
        print(f"  {sys_key}  N={len(invs)}  mean={m:.3f}  std={s:.3f}  CV={cv:.2f}%")

    out = {"rows": rows}
    out_path = _ROOT / "results/gate2/conditional_invariant.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
