"""BERT-on-Wikipedia distribution-confound test (P-M-I prediction slot).

Candidate-5 predicts c ~= 2.0 for pure-text MLM encoders. BERT-base measured
on C4-clean produces c=2.65 (15-30% high, the sole outlier in the 12-system
scorecard). RoBERTa measured on C4-clean produces c=2.25 (fits). BERT was
trained on Wikipedia+BooksCorpus (not web-scraped C4); RoBERTa was trained on
CC-100/OpenWebText, closer to C4.

Working hypothesis from BREAKTHROUGH_SYNTHESIS: the BERT outlier is a
stimulus-vs-training-distribution confound, not an MLM-base failure of
candidate-5. This probe re-measures BERT on wikitext-103 (proxy for
BERT's training distribution). If c drops from 2.65 -> ~2.0, distribution
confound is confirmed and candidate-5's 11/12 becomes 12/12.

Controls:
  - RoBERTa-base on wikitext: should move little (already fits on C4).
  - MiniLM-L6 on wikitext: stimulus-distribution null (already 2.03 on C4,
    should stay ~2.0; if it shifts, distribution broadly rescales c).
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
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import wikitext_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def measure_c_wikitext(hf_id, sk, class_id, n=1000, seed=42):
    print(f"\n=== {sk} on wikitext-103 ===")
    t0 = time.time()
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2
    sents = []
    for rec in wikitext_v1(seed=seed, n_samples=n * 5):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    print(f"  collected {len(sents)} wikitext stimuli")
    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=sk, class_id=class_id,
        quantization="fp16",
        stimulus_version=f"wikitext103.v1.seed{seed}.n{n}", seed=seed,
        batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, _, r2 = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    print(f"  p={p:.3f}  d_rd={rd['d_rd']:.2f}  c={c:.2f}  Ck_R2={r2:.3f}  (t={time.time()-t0:.1f}s)")
    sys_obj.unload()
    torch.cuda.empty_cache()
    return {"system": sk, "p": float(p), "d_rd": float(rd["d_rd"]),
            "c": float(c), "Ck_R2": float(r2), "n": n}


def main():
    # C4 reference values from committed ledger (genome_052/053)
    c4_reference = {
        "bert-base-uncased": 2.65,
        "roberta-base": 2.25,
        "minilm-l6-contrastive": 2.03,
    }
    targets = [
        ("bert-base-uncased", "bert-base-uncased", 7),
        ("FacebookAI/roberta-base", "roberta-base", 7),
        ("sentence-transformers/all-MiniLM-L6-v2", "minilm-l6-contrastive", 8),
    ]
    results = []
    for hf, sk, cid in targets:
        try:
            r = measure_c_wikitext(hf, sk, cid)
            r["c_on_c4"] = c4_reference.get(sk)
            r["delta_c_wiki_minus_c4"] = (r["c"] - r["c_on_c4"]
                                          if r["c_on_c4"] is not None else None)
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"system": sk, "error": str(e)})

    print("\n=== BERT DISTRIBUTION-CONFOUND TEST ===")
    print(f"  Hypothesis: BERT c drops from 2.65 (C4) -> ~2.0 (wikitext)")
    for r in results:
        if "error" in r:
            print(f"  {r['system']}: ERROR")
            continue
        rel_err_vs_2 = abs(r["c"] - 2.0) / 2.0
        tag = "FITS_C5" if rel_err_vs_2 < 0.15 else "OUTLIER"
        d = r.get("delta_c_wiki_minus_c4")
        d_str = f"{d:+.2f}" if d is not None else "n/a"
        print(f"  {r['system']}: c_wiki={r['c']:.2f}  "
              f"(c_c4={r['c_on_c4']})  delta={d_str}  "
              f"rel_err_vs_2={rel_err_vs_2:.3f}  {tag}")

    bert = next((r for r in results if r["system"] == "bert-base-uncased"
                 and "error" not in r), None)
    if bert is not None:
        delta = bert.get("delta_c_wiki_minus_c4")
        if delta is not None and delta < -0.3:
            verdict = ("DISTRIBUTION_CONFOUND_CONFIRMED — BERT c drops "
                       f">{-delta:.2f} on training distribution; "
                       "candidate-5 passes 12/12 with distribution-match")
        elif delta is not None and abs(delta) < 0.15:
            verdict = ("DISTRIBUTION_NEUTRAL — BERT c unchanged on wikitext; "
                       "the 2.65 value is a real BERT-specific effect, "
                       "candidate-5 MLM story needs refinement")
        else:
            verdict = f"MIXED — BERT c moved {delta:+.2f}, ambiguous"
    else:
        verdict = "BERT_RUN_FAILED"
    print(f"\n  verdict: {verdict}")

    out = {
        "purpose": "BERT-on-Wikipedia distribution-confound test (candidate-5)",
        "stimulus": "wikitext-103-raw-v1",
        "per_system": results,
        "c4_reference": c4_reference,
        "verdict": verdict,
    }
    out_path = _ROOT / "results/gate2/bert_wikipedia_c.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
