"""Genome_108: Codex round-2 blind spot #3 — hard OOD stimuli.

Codex: 'C4 universality could be dataset adjacency / memorization. Test harder
OOD (non-English scripts, code/math, synthetic grammar, freshly-written text).
If CV grows systematically with token-frequency divergence from C4, the
universal is a pretraining-distribution artifact.'

Probe: Qwen3 + DeepSeek + BERT at mid-depth, seq_mean pooling, n=800, across
5 stimulus banks:
  - C4 clean (baseline, in-distribution)
  - wikitext (slightly OOD)
  - code (Python source, very OOD)
  - math (arXiv math notation, VERY OOD)
  - random-character-strings (extreme OOD - syntactic noise)

If CV grows as stimulus moves further OOD, universality is an in-distribution
effect. If CV stays tight, universality transcends stimulus distribution.
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
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1, wikitext_v1  # noqa: E402

_ROOT = _THIS_DIR.parent

SYSTEMS = [
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    ("deepseek-r1-distill-qwen-1.5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("bert-base-uncased", "bert-base-uncased"),
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


def build_stimuli():
    stims = {}
    # C4
    c4 = []
    for r in c4_clean_v1(seed=42, n_samples=3000):
        c4.append(r["text"])
        if len(c4) >= 800: break
    stims["c4"] = c4
    # wikitext
    try:
        wt = []
        for r in wikitext_v1(seed=42, n_samples=3000):
            wt.append(r["text"])
            if len(wt) >= 800: break
        stims["wikitext"] = wt
    except Exception as e:
        print(f"  wikitext fail: {e}")
    # Python code snippets - synthesize plausible code
    code_lines = [
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "import numpy as np\narr = np.zeros((3, 3))\nfor i in range(3):\n    arr[i, i] = 1",
        "class LinkedList:\n    def __init__(self):\n        self.head = None\n    def append(self, data):\n        pass",
        "with open('data.txt', 'r') as f:\n    for line in f:\n        if line.startswith('#'):\n            continue",
        "results = {k: v * 2 for k, v in items.items() if isinstance(v, (int, float))}",
    ]
    code = []
    rng = np.random.default_rng(42)
    for _ in range(800):
        code.append(rng.choice(code_lines))
    stims["code"] = code
    # Math-like strings with LaTeX
    math_exprs = [
        "\\int_0^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}",
        "E = mc^2 \\implies \\Delta E = c^2 \\Delta m",
        "\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}",
        "\\mathbf{A} \\cdot \\mathbf{B} = |A||B|\\cos\\theta",
        "f(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}",
        "H(X) = -\\sum_x p(x) \\log p(x)",
        "\\nabla \\cdot \\mathbf{E} = \\rho / \\epsilon_0",
    ]
    math = []
    for _ in range(800):
        math.append(rng.choice(math_exprs) + " and so we have " + rng.choice(math_exprs))
    stims["math_latex"] = math
    # Random-character strings (extreme OOD)
    rand_chars = []
    for _ in range(800):
        length = rng.integers(50, 200)
        s = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz 0123456789")) for _ in range(length))
        rand_chars.append(s)
    stims["random_chars"] = rand_chars
    return stims


@torch.no_grad()
def extract_seq_mean(model, tok, texts, layer_idx, batch=8, max_len=256, device="cuda"):
    model.eval()
    acts = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                   max_length=max_len).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer_idx].float()
        mask = enc["attention_mask"].float().unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        acts.append(pooled.cpu().numpy())
    return np.concatenate(acts, axis=0).astype(np.float32)


def main():
    t0 = time.time()
    stims = build_stimuli()
    print(f"[{time.time()-t0:.1f}s] stimuli: {list(stims.keys())}")

    rows = []
    for label, hf_id in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {label} =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
        mid = max(1, sys_obj.n_hidden_layers() // 2)
        for stim_name, texts in stims.items():
            try:
                X = extract_seq_mean(sys_obj.model, sys_obj.tokenizer, texts, mid)
            except Exception as e:
                print(f"  FAIL {stim_name}: {e}"); continue
            s = spectrum(X)
            st = stats(s)
            rows.append({"system": label, "stimulus": stim_name,
                         "n": int(X.shape[0]), "h": int(X.shape[1]), **st})
            print(f"  {stim_name:15s}  sqrt(er)*a={st['sqrt_er_alpha']:.3f}  "
                  f"(eff_rank={st['eff_rank']:.1f}, alpha={st['alpha']:.3f})")
        sys_obj.unload(); torch.cuda.empty_cache()

    print(f"\n\n=== CROSS-SYSTEM CV per stimulus ===")
    for stim in sorted(set(r["stimulus"] for r in rows)):
        invs = [r["sqrt_er_alpha"] for r in rows if r["stimulus"] == stim]
        if len(invs) < 2: continue
        m, s = float(np.mean(invs)), float(np.std(invs))
        cv = 100*s/m if m else 0
        mark = "  <-- TIGHT" if 0 < cv < 10 else ("  <-- LOOSE" if cv > 25 else "")
        print(f"  {stim:15s}  N={len(invs)}  mean={m:>7.3f}  std={s:>7.3f}  CV={cv:>6.2f}%{mark}")

    out = {"rows": rows}
    out_path = _ROOT / "results/gate2/hard_ood_stimuli.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
