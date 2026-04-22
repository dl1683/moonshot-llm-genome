"""Codex Move 3: stimulus-axis sweep to discriminate candidate-5's stimulus-sensitivity mechanism.

Per Codex strategic verdict 2026-04-22-T43h (DeepMind-publishability 9.3/10).

Hypothesis. genome_058 showed c drops when moving text stimuli from C4 to
wikitext-103-raw. Why? Two candidate mechanisms:

  (A) SEMANTIC-STRUCTURE: c tracks something about syntactic / topic /
      discourse structure that differs between web-scraped C4 and
      article-style Wikipedia.
  (B) COMPRESSIBILITY: c tracks average token-level compressibility of the
      stimulus. Wikipedia is more domain-coherent, hence lower-entropy,
      hence produces activations with different spectrum.

Decisive probe. Re-measure c on scrambled versions of wikitext:

  - wikitext_v1 baseline (genome_058 reproduces c~1.16 on Qwen3, 0.55 on BERT)
  - wikitext word-shuffled WITHIN each sample (preserves unigram dist,
    destroys n-gram/syntactic structure completely)
  - wikitext token-reversed (preserves forward-bigram distribution if
    tokenizer is BPE; destroys directionality)

If semantic/syntactic structure is what drives the C4-vs-wikitext gap,
scrambled versions of wikitext should SHIFT c back toward C4 (restoring
the structure-destroying property of typical web crawl). If compressibility,
scrambled wikitext should not shift c (unigram dist preserved).

Two models: Qwen3-0.6B (CLM, text_base 2 at C4) and BERT-base (MLM, text
_base 2.65 at C4, outlier). Candidate-5 at C4 says text=2, but stimulus-
sensitivity reframed it to a (model x stimulus) property. This experiment
gives derivation-class signal.

Also records eff_rank + alpha per condition so candidate-8 bridge can be
checked point-by-point across stimuli.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import c4_clean_v1, wikitext_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(X.shape[0] - 1)
    return s.astype(np.float64)


def eff_rank(s):
    s2 = s ** 2
    total = s2.sum()
    return 0.0 if total <= 0 else float(total ** 2 / (s2 ** 2).sum())


def fit_alpha_tail(s, lo_frac=0.05, hi_frac=0.5):
    r = np.arange(1, len(s) + 1)
    lo = max(1, int(len(s) * lo_frac))
    hi = int(len(s) * hi_frac)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    return float(-slope)


def get_sents_c4(n, seed):
    out = []
    for rec in c4_clean_v1(seed=seed, n_samples=5 * n):
        out.append(rec["text"])
        if len(out) >= n:
            break
    return out


def get_sents_wiki(n, seed):
    out = []
    for rec in wikitext_v1(seed=seed, n_samples=5 * n):
        out.append(rec["text"])
        if len(out) >= n:
            break
    return out


def scramble_words(texts, seed):
    rng = random.Random(seed)
    out = []
    for t in texts:
        words = t.split()
        rng.shuffle(words)
        out.append(" ".join(words))
    return out


def reverse_words(texts):
    return [" ".join(reversed(t.split())) for t in texts]


def measure_cloud(sys_obj, texts, sk, class_id, seed, mid, stim_tag):
    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=texts, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=sk, class_id=class_id,
        quantization="fp16",
        stimulus_version=f"stim_axis.{stim_tag}.seed{seed}.n{len(texts)}",
        seed=seed, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    s = spectrum(X)
    er = eff_rank(s)
    alpha = fit_alpha_tail(s)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, _, _ = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    return {"stim": stim_tag, "c": float(c),
            "p": float(p), "d_rd": float(rd["d_rd"]),
            "eff_rank": er, "alpha": alpha,
            "ratio_eff_rank_over_d_rd": er / rd["d_rd"],
            "rel_err_ratio_vs_c": abs(er / rd["d_rd"] - c) / c}


def main():
    seed = 42
    n = 1000
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] fetching C4 and wikitext stimuli...")
    c4_sents = get_sents_c4(n, seed)
    wiki_sents = get_sents_wiki(n, seed)
    wiki_scramb = scramble_words(wiki_sents, seed=seed + 1)
    wiki_rev = reverse_words(wiki_sents)
    print(f"  c4: {len(c4_sents)}, wiki: {len(wiki_sents)}, "
          f"wiki_scramb: {len(wiki_scramb)}, wiki_rev: {len(wiki_rev)}")

    targets = [
        ("Qwen/Qwen3-0.6B", "qwen3-0.6b", 1),
        ("bert-base-uncased", "bert-base-uncased", 7),
    ]
    conditions = [
        (c4_sents, "c4"),
        (wiki_sents, "wiki_raw"),
        (wiki_scramb, "wiki_word_shuffled"),
        (wiki_rev, "wiki_word_reversed"),
    ]

    results = []
    out_path = _ROOT / "results/gate2/stimulus_axis_sweep.json"

    def flush():
        out_path.write_text(json.dumps(
            {"purpose": "Candidate-5 stimulus-axis discrimination (Codex Move 3)",
             "per_system_stim": results,
             "verdict": "IN_PROGRESS"}, indent=2))

    for hf, sk, cid in targets:
        print(f"\n=== {sk} ===")
        sys_obj = load_system(hf, quant="fp16", untrained=False, device="cuda")
        mid = sys_obj.n_hidden_layers() // 2
        for texts, stim_tag in conditions:
            t1 = time.time()
            try:
                m = measure_cloud(sys_obj, texts, sk, cid, seed, mid, stim_tag)
                m["system"] = sk
                print(f"  {stim_tag:24s}  c={m['c']:.3f}  "
                      f"ratio={m['ratio_eff_rank_over_d_rd']:.3f}  "
                      f"rel_err={m['rel_err_ratio_vs_c']:.3f}  "
                      f"alpha={m['alpha']:.3f}  "
                      f"(t={time.time()-t1:.1f}s)")
                results.append(m)
            except Exception as e:
                import traceback; traceback.print_exc()
                results.append({"system": sk, "stim": stim_tag, "error": str(e)})
            flush()
        sys_obj.unload(); torch.cuda.empty_cache()

    # Verdict
    def c_of(sk, stim):
        for r in results:
            if r.get("system") == sk and r.get("stim") == stim and "error" not in r:
                return r["c"]
        return None

    for sk in ["qwen3-0.6b", "bert-base-uncased"]:
        c4 = c_of(sk, "c4")
        wi = c_of(sk, "wiki_raw")
        sc = c_of(sk, "wiki_word_shuffled")
        rv = c_of(sk, "wiki_word_reversed")
        if None not in (c4, wi, sc, rv):
            print(f"\n  {sk}: c4={c4:.2f}  wiki={wi:.2f}  shuf={sc:.2f}  rev={rv:.2f}")
            # If scramble restores c back toward c4, semantic/syntactic structure drives it.
            # If scramble stays near wiki, compressibility (unigram) drives it.
            scramble_restores = abs(sc - c4) < abs(sc - wi)
            print(f"    scramble_restores_toward_c4: {scramble_restores}")

    out_path.write_text(json.dumps(
        {"purpose": "Candidate-5 stimulus-axis discrimination (Codex Move 3)",
         "per_system_stim": results,
         "verdict": "COMPLETED"}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
