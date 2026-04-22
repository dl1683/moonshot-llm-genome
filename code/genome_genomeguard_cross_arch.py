"""GenomeGuard cross-architecture swap-detection benchmark.

genome_067 proved GenomeGuard works on Qwen3 (7.29x swap spike) and
BERT (post-hoc from genome_062 = 45x). This script formally confirms
swap-detection across the full 7/8 candidate-8 bridge scorecard
systems so the shipping claim is cross-architecture, not Qwen3-specific.

Systems tested:
  Qwen3-0.6B, DeepSeek-R1-Distill-1.5B, BERT-base, RoBERTa-base,
  MiniLM-L6-contrastive. (CLIP and DINOv2 excluded because their
  extractors need different pipelines; those were measured separately
  in genome_063/064.)

For each system:
  - Probe 1: C4 baseline (n=1000, seed 42)
  - Probe 2: wikitext-word-shuffled (n=1000, seed 42+1)
  - Compute: baseline rel_err, swap rel_err, swap_spike = swap/base

Spike >=2x on >=4/5 text systems = shippable cross-arch claim.
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
from genome_genomeguard import probe_health  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1, wikitext_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


def scramble_words(texts, seed):
    rng = random.Random(seed)
    out = []
    for t in texts:
        words = t.split()
        rng.shuffle(words)
        out.append(" ".join(words))
    return out


def main():
    c4 = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        c4.append(rec["text"])
        if len(c4) >= 1000:
            break
    wiki = []
    for rec in wikitext_v1(seed=42, n_samples=3000):
        wiki.append(rec["text"])
        if len(wiki) >= 1000:
            break
    swap = scramble_words(wiki, seed=43)

    systems = [
        ("Qwen/Qwen3-0.6B", "qwen3-0.6b", 1),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-r1-distill-qwen-1.5b", 2),
        ("bert-base-uncased", "bert-base-uncased", 7),
        ("FacebookAI/roberta-base", "roberta-base", 7),
        ("sentence-transformers/all-MiniLM-L6-v2", "minilm-l6-contrastive", 8),
    ]

    rows = []
    for hf_id, sk, cid in systems:
        print(f"\n=== {sk} ===")
        t0 = time.time()
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
            mid = sys_obj.n_hidden_layers() // 2

            hb = probe_health(sys_obj.model, sys_obj.tokenizer, c4, "cuda", mid, tag=f"{sk}_base")
            hs = probe_health(sys_obj.model, sys_obj.tokenizer, swap, "cuda", mid, tag=f"{sk}_swap")
            spike = hs["bridge_rel_err"] / max(hb["bridge_rel_err"], 1e-6)
            row = {"system": sk,
                   "baseline_rel_err": hb["bridge_rel_err"],
                   "swap_rel_err": hs["bridge_rel_err"],
                   "swap_spike": float(spike),
                   "PASS_2x": bool(spike >= 2.0),
                   "wall_s": time.time() - t0}
            rows.append(row)
            print(f"  baseline rel_err={hb['bridge_rel_err']:.3f}  "
                  f"swap rel_err={hs['bridge_rel_err']:.3f}  "
                  f"spike={spike:.2f}x  {'PASS' if spike >= 2.0 else 'FAIL'}  "
                  f"(t={time.time()-t0:.1f}s)")
            sys_obj.unload(); torch.cuda.empty_cache()
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({"system": sk, "error": str(e)})

    passes = sum(1 for r in rows if r.get("PASS_2x", False))
    tested = sum(1 for r in rows if "error" not in r)
    print(f"\n=== CROSS-ARCH GENOMEGUARD SWAP DETECTION ===")
    for r in rows:
        if "error" in r:
            print(f"  {r['system']:30s} ERROR")
        else:
            tag = "PASS" if r["PASS_2x"] else "FAIL"
            print(f"  {r['system']:30s} base={r['baseline_rel_err']:.3f}  "
                  f"swap={r['swap_rel_err']:.3f}  spike={r['swap_spike']:.1f}x  {tag}")
    print(f"\n  {passes}/{tested} PASS 2x spike threshold")

    if passes >= 4:
        verdict = (f"CROSS_ARCH_LANDS - {passes}/{tested} systems spike "
                   f">=2x. GenomeGuard is cross-architecture.")
    else:
        verdict = f"PARTIAL - only {passes}/{tested} pass. Architecture-dependent."
    print(f"  verdict: {verdict}")

    out = {"purpose": "GenomeGuard cross-architecture swap detection",
           "rows": rows, "passes": passes, "tested": tested,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/genomeguard_cross_arch.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
