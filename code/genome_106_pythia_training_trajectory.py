"""Genome_106: does sqrt(er)*alpha converge toward the trained attractor DURING training?

Pythia has public checkpoints at every 1000 training steps (0, 1k, 2k, ..., 143k).
EleutherAI/pythia-410m-v0 / EleutherAI/pythia-410m (with revision='step{N}').

Plan: measure invariant on Pythia-410M at 5 checkpoints:
  step 0          (random-init equivalent)
  step 1000       (early training)
  step 10000      (middle)
  step 50000      (late)
  step 143000     (fully trained, aka 'main')

If the invariant traces:
  random-init-like (~7.5) -> trained attractor (~4.3) monotonically
then the attractor is a DYNAMICAL fixed point of training, not just a
post-hoc observation.

This probes blind spot not explicitly in Codex list: is the universal
attractor a FIXED POINT of the training dynamics? If yes, huge story.
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

_ROOT = _THIS_DIR.parent


# Pythia-410m checkpoints at various training steps
# We'll use transformers directly with revision kwarg
CHECKPOINTS = [
    ("pythia-step0", "step0"),
    ("pythia-step1000", "step1000"),
    ("pythia-step10000", "step10000"),
    ("pythia-step50000", "step50000"),
    ("pythia-step143000", "step143000"),  # fully trained
]
HF_ID = "EleutherAI/pythia-410m"


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


def load_pythia_checkpoint(hf_id, revision, device="cuda"):
    """Load Pythia at specific checkpoint revision."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision,
                                                  torch_dtype=torch.float16).to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def extract_mid(model, tok, texts, batch=16, max_len=256, device="cuda"):
    model.eval()
    acts = []
    n_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else len(model.gpt_neox.layers)
    mid = max(1, n_layers // 2)
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                   max_length=max_len).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[mid].float()
        mask = enc["attention_mask"].float().unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        acts.append(pooled.cpu().numpy())
    X = np.concatenate(acts, axis=0).astype(np.float32)
    return X


def main():
    t0 = time.time()
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=2000):
        sents.append(rec["text"])
        if len(sents) >= 800: break

    rows = []
    for label, rev in CHECKPOINTS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {label} ({rev}) =====")
        try:
            model, tok = load_pythia_checkpoint(HF_ID, rev)
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
        try:
            X = extract_mid(model, tok, sents)
        except Exception as e:
            print(f"  FAIL extract: {e}")
            del model, tok; torch.cuda.empty_cache(); continue
        del model, tok; torch.cuda.empty_cache()
        s = spectrum(X)
        st = stats(s)
        st.update({"label": label, "revision": rev, "n": int(X.shape[0]), "h": int(X.shape[1])})
        rows.append(st)
        print(f"  eff_rank={st['eff_rank']:.2f}  alpha={st['alpha']:.3f}  "
              f"sqrt(er)*a={st['sqrt_er_alpha']:.3f}")

    print("\n\n=== PYTHIA-410M TRAINING TRAJECTORY ===")
    print(f"  {'revision':20s} {'sqrt(er)*alpha':>15s}  {'vs-trained-attractor':>22s}")
    attractor = 4.27
    for r in rows:
        delta = r["sqrt_er_alpha"] - attractor
        print(f"  {r['revision']:20s} {r['sqrt_er_alpha']:>15.3f}  {delta:>+22.3f}")

    out = {"rows": rows, "attractor_ref": attractor}
    out_path = _ROOT / "results/gate2/pythia_training_trajectory.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
