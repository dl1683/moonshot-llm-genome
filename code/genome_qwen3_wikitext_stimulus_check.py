"""Stimulus-sensitivity sanity check on Qwen3-0.6B (candidate-5's anchor system).

genome_bert_wikipedia_c showed BERT + RoBERTa + MiniLM all drop ~1.1 when
measured on wikitext-103-raw. That was intended as a BERT distribution-
confound test but all three text encoders moved. Two possible readings:

 (A) Candidate-5 is a property of (model x stimulus), not model alone.
     All text encoders drop on wikitext because wikitext differs from C4
     in intrinsic dim / topic coherence / formatting.

 (B) Wikitext-103-raw has formatting artifacts (@,@ section markers, leading
     spaces) that specifically derail *encoder* models (BERT/RoBERTa/MiniLM
     bidirectional), but autoregressive decoder models (Qwen3) see through
     them.

Decisive test: measure Qwen3-0.6B (our most-characterized text CLM system,
c=1.89 on C4) on wikitext. Predictions:

 - If Qwen3 on wikitext still c=~2: BERT-family encoders are
   stimulus-sensitive in a way CLMs are not; the wikitext-103 raw format
   breaks MLM evaluation specifically. Candidate-5 remains a model property
   under "clean" stimuli.

 - If Qwen3 also drops to c=~0.8-1.0: candidate-5 is fundamentally
   stimulus-dependent; c = f(model, stimulus), not a model-only invariant.
   This would FALSIFY the stimulus-agnostic framing of candidate-5.

Also tests DeepSeek-R1-Distill-Qwen-1.5B (second text CLM system).
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
from stimulus_banks import c4_clean_v1, wikitext_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def measure_c(hf_id, sk, class_id, stim_fn, stim_tag, n=1000, seed=42):
    print(f"\n=== {sk} on {stim_tag} ===")
    t0 = time.time()
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    mid = sys_obj.n_hidden_layers() // 2
    sents = []
    for rec in stim_fn(seed=seed, n_samples=n * 5):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    print(f"  collected {len(sents)} {stim_tag} stimuli")
    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=sk, class_id=class_id,
        quantization="fp16",
        stimulus_version=f"{stim_tag}.seed{seed}.n{n}", seed=seed,
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
    return {"system": sk, "stimulus": stim_tag, "p": float(p),
            "d_rd": float(rd["d_rd"]), "c": float(c), "Ck_R2": float(r2)}


def main():
    targets = [
        ("Qwen/Qwen3-0.6B", "qwen3-0.6b", 1),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
         "deepseek-r1-distill-qwen-1.5b", 2),
    ]
    stims = [
        (c4_clean_v1, "c4_clean.v1"),
        (wikitext_v1, "wikitext103.v1"),
    ]
    results = []
    for hf, sk, cid in targets:
        for stim_fn, stim_tag in stims:
            try:
                results.append(measure_c(hf, sk, cid, stim_fn, stim_tag))
            except Exception as e:
                import traceback; traceback.print_exc()
                results.append({"system": sk, "stimulus": stim_tag,
                                "error": str(e)})

    print("\n=== STIMULUS-SENSITIVITY TEST (Qwen3/DeepSeek c4 vs wikitext) ===")
    by_system = {}
    for r in results:
        if "error" in r:
            continue
        by_system.setdefault(r["system"], {})[r["stimulus"]] = r["c"]
    for sk, cs in by_system.items():
        c4 = cs.get("c4_clean.v1")
        wi = cs.get("wikitext103.v1")
        if c4 is not None and wi is not None:
            print(f"  {sk}: c4={c4:.2f}  wikitext={wi:.2f}  delta={wi-c4:+.2f}")

    # Verdict
    deltas = []
    for sk, cs in by_system.items():
        c4 = cs.get("c4_clean.v1"); wi = cs.get("wikitext103.v1")
        if c4 is not None and wi is not None:
            deltas.append(wi - c4)
    mean_delta = np.mean(deltas) if deltas else None

    if mean_delta is not None and mean_delta < -0.5:
        verdict = ("STIMULUS_DOMINATES — decoder-CLM c also drops ~"
                   f"{-mean_delta:.2f} on wikitext. Candidate-5 is NOT a "
                   "model-only invariant; it's model x stimulus. "
                   "Reframes the claim.")
    elif mean_delta is not None and abs(mean_delta) < 0.25:
        verdict = ("CLM_STABLE_MLM_FRAGILE — Qwen3/DeepSeek largely stable "
                   "on wikitext. BERT/RoBERTa/MiniLM wikitext drop is "
                   "encoder-or-formatting-specific, not a general c property.")
    else:
        verdict = f"MIXED — mean delta {mean_delta:+.2f}"
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Candidate-5 stimulus-sensitivity test on text CLM anchors",
           "per_system_stimulus": results,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/qwen3_deepseek_wikitext_stimulus_check.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
