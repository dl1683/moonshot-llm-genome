"""
grafting_004_ridge_overdetermined.py

Diagnose and fix the grafting_003 PARTIAL result (grafted NLL 9.64, ceiling gap 5.80 nats).

grafting_003 failure mode: underdetermined lstsq (n=1500 < d=3072) returns the
minimum-norm solution, which perfectly fits training activations (R2=1.0) but
generalises poorly to test distribution. Two candidate fixes:

  Fix A — Ridge regularisation: add lambda*I to the normal equations, shrinking
           null-space weight components toward zero -> better generalisation from
           underdetermined system.

  Fix B — Overdetermined regime: collect n=4096 > d=3072 training samples so
           the system becomes overdetermined and the least-squares solution is
           unique (minimum-residual, not minimum-norm).

Both fixes are tested here via a single data collection pass at n=4096.

Conditions
----------
  n1500_l1e-4  : n=1500, lambda=0.0001  (near-identical to grafting_003, verify)
  n1500_l0.01  : n=1500, lambda=0.01    (mild Ridge)
  n1500_l0.1   : n=1500, lambda=0.1     (moderate Ridge)
  n1500_l1.0   : n=1500, lambda=1.0     (heavy Ridge)
  n1500_l10.0  : n=1500, lambda=10.0    (very heavy Ridge)
  n4096_l1e-4  : n=4096, lambda=0.0001  (overdetermined, primary test)
  n4096_l0.1   : n=4096, lambda=0.1     (overdetermined + mild Ridge)

Pass threshold: any condition achieves grafted NLL within 0.5 nats of donor (3.8446).
"""

import json
import pathlib
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

ROOT    = pathlib.Path(__file__).parent.parent.parent
RESULTS = pathlib.Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SEED     = 42
N_TRAIN  = 4096   # collect at max n; slice per condition
N_TEST   = 300
N_TOTAL  = N_TRAIN + N_TEST
SEQ_LEN  = 128
BATCH    = 16
MODEL_ID = "Qwen/Qwen3-0.6B"

CONDITIONS = [
    (1500, 1e-4,  "n1500_l1e-4"),
    (1500, 0.01,  "n1500_l0.01"),
    (1500, 0.1,   "n1500_l0.1"),
    (1500, 1.0,   "n1500_l1.0"),
    (1500, 10.0,  "n1500_l10.0"),
    (4096, 1e-4,  "n4096_l1e-4"),
    (4096, 0.1,   "n4096_l0.1"),
]


def load_texts(n, seed=42):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n * 4, len(ds)), replace=False)
    texts = []
    for i in indices:
        t = ds[int(i)]["text"].strip()
        if len(t) >= 80:
            texts.append(t[:512])
        if len(texts) >= n:
            break
    return texts


def measure_nll(model, tokenizer, texts, max_len=SEQ_LEN, batch=8):
    model.eval()
    total_nll, total_toks = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(DEVICE)
            out   = model(**enc)
            logits = out.logits[:, :-1].float()
            labels = enc["input_ids"][:, 1:].clone()
            labels[enc["attention_mask"][:, 1:] == 0] = -100
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1), ignore_index=-100, reduction="sum"
            )
            total_nll  += loss.item()
            total_toks += (labels != -100).sum().item()
    return total_nll / max(total_toks, 1)


def collect_hidden(model, tokenizer, texts, n_layers):
    hidden = {l: [] for l in range(n_layers + 1)}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=SEQ_LEN).to(DEVICE)
            out  = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            for l, h in enumerate(out.hidden_states):
                pooled = (h.float() * mask).sum(1) / mask.sum(1)
                hidden[l].append(pooled.cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in hidden.items()}


def collect_mlp_intermediates(model, tokenizer, texts, n_layers):
    mlp_interm = {l: [] for l in range(n_layers)}
    hooks = []

    def make_pre_hook(layer_idx):
        def pre_hook(mod, inp):
            x = inp[0].float().mean(dim=1).detach().cpu().numpy()
            mlp_interm[layer_idx].append(x)
        return pre_hook

    for l in range(n_layers):
        h = model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
            make_pre_hook(l)
        )
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=SEQ_LEN).to(DEVICE)
            model(**enc)

    for h in hooks:
        h.remove()
    return {l: np.concatenate(v, axis=0) for l, v in mlp_interm.items()}


def ridge_transplant(model_lesion, h_donor, h_lesion, mlp_interm,
                     n_layers, n_train, lambda_val):
    """
    Ridge regression transplant via normal equations:
      sol = (F^T F + lambda * I)^{-1} F^T r
    where F = mlp_interm[:n_train], r = (h_donor - h_lesion)[:n_train].

    For lambda=1e-4 and n<d: equivalent to lstsq but with tiny shrinkage.
    For n>d: overdetermined system; adding tiny lambda makes solve numerically safe.
    """
    layer_results = {}
    for L in range(n_layers):
        r = (h_donor[L + 1][:n_train] - h_lesion[L + 1][:n_train]).astype(np.float64)
        f = mlp_interm[L][:n_train].astype(np.float64)

        FtF = f.T @ f                                        # (d, d)
        FtR = f.T @ r                                        # (d, h)
        A   = FtF + lambda_val * np.eye(f.shape[1])         # (d, d) regularised
        sol = np.linalg.solve(A, FtR)                       # (d, h)

        pred    = f @ sol
        r_norm  = float(np.linalg.norm(r))
        res_norm = float(np.linalg.norm(r - pred))
        r2 = 1.0 - (res_norm ** 2) / (r_norm ** 2 + 1e-12)

        with torch.no_grad():
            w = torch.tensor(sol.T, dtype=torch.float16).to(DEVICE)
            model_lesion.model.layers[L].mlp.down_proj.weight.copy_(w)

        layer_results[L] = {"r2_train": float(r2), "res_norm": float(res_norm)}
        if L % 4 == 0:
            print(f"    L{L:02d}: R2={r2:.4f}  |r|={r_norm:.1f}  |res|={res_norm:.2f}")

    return layer_results


def reset_lesion(model, n_layers):
    with torch.no_grad():
        for L in range(n_layers):
            model.model.layers[L].mlp.down_proj.weight.zero_()


def main():
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading texts...")
    texts       = load_texts(N_TOTAL, SEED)
    train_texts = texts[:N_TRAIN]
    test_texts  = texts[N_TRAIN:N_TOTAL]
    print(f"  train={len(train_texts)} test={len(test_texts)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- Pass 1: donor ----
    print(f"\n[{time.time()-t0:.1f}s] Loading donor ({MODEL_ID})...")
    donor = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    n_layers = donor.config.num_hidden_layers
    print(f"  n_layers={n_layers}")

    nll_donor = measure_nll(donor, tok, test_texts)
    print(f"  Donor NLL (test): {nll_donor:.4f}")

    print(f"[{time.time()-t0:.1f}s] Collecting donor hidden states (n={N_TRAIN})...")
    h_donor = collect_hidden(donor, tok, train_texts, n_layers)
    print(f"  shape: {h_donor[0].shape}")
    del donor
    torch.cuda.empty_cache()

    # ---- Pass 2: lesioned recipient ----
    print(f"\n[{time.time()-t0:.1f}s] Creating lesioned recipient (down_proj=0)...")
    recipient = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    reset_lesion(recipient, n_layers)

    nll_lesion = measure_nll(recipient, tok, test_texts)
    print(f"  Lesioned NLL (test): {nll_lesion:.4f}")

    print(f"[{time.time()-t0:.1f}s] Collecting lesion hidden states + MLP intermediates (n={N_TRAIN})...")
    h_lesion   = collect_hidden(recipient, tok, train_texts, n_layers)
    mlp_interm = collect_mlp_intermediates(recipient, tok, train_texts, n_layers)
    print(f"  hidden: {h_lesion[0].shape}  interm: {mlp_interm[0].shape}")

    # ---- Condition sweep ----
    condition_results = {}
    for n_train, lam, name in CONDITIONS:
        print(f"\n[{time.time()-t0:.1f}s] Condition {name}: n={n_train}, lambda={lam}")
        reset_lesion(recipient, n_layers)

        t_cond = time.time()
        layer_res = ridge_transplant(
            recipient, h_donor, h_lesion, mlp_interm, n_layers, n_train, lam
        )
        transplant_s = time.time() - t_cond

        nll_grafted = measure_nll(recipient, tok, test_texts)
        improvement = nll_lesion - nll_grafted
        ceiling_gap = nll_grafted - nll_donor

        r2_vals = [v["r2_train"] for v in layer_res.values()]
        mean_r2 = float(np.mean(r2_vals))

        verdict = "PASS" if ceiling_gap < 0.5 else (
                  "PARTIAL" if improvement > 1.0 else (
                  "WEAK" if improvement > 0.1 else "KILL"))

        print(f"  Grafted NLL:   {nll_grafted:.4f}")
        print(f"  Improvement:   {improvement:.4f} nats  ({100*improvement/(nll_lesion-nll_donor):.1f}% recovered)")
        print(f"  Ceiling gap:   {ceiling_gap:.4f} nats")
        print(f"  Mean R2_train: {mean_r2:.6f}")
        print(f"  Transplant:    {transplant_s:.1f}s")
        print(f"  Verdict:       {verdict}")

        condition_results[name] = {
            "n_train": n_train, "lambda": lam,
            "nll_grafted": float(nll_grafted),
            "improvement": float(improvement),
            "ceiling_gap": float(ceiling_gap),
            "pct_recovered": float(100 * improvement / (nll_lesion - nll_donor)),
            "mean_r2_train": mean_r2,
            "transplant_s": float(transplant_s),
            "verdict": verdict,
        }

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print(f"SUMMARY  (Donor {nll_donor:.4f}  Lesion {nll_lesion:.4f}  gap {nll_lesion-nll_donor:.4f})")
    print(f"  grafting_003 baseline: NLL 9.6436  gap 5.7990  59.5% recovered")
    print(f"{'='*70}")
    print(f"  {'Condition':<18} {'NLL':>7}  {'Gap':>7}  {'%Rec':>7}  {'R2':>8}  Verdict")
    for name, r in condition_results.items():
        print(f"  {name:<18} {r['nll_grafted']:>7.4f}  "
              f"{r['ceiling_gap']:>7.4f}  {r['pct_recovered']:>6.1f}%  "
              f"{r['mean_r2_train']:>8.6f}  {r['verdict']}")
    print(f"{'='*70}")

    best = min(condition_results.items(), key=lambda kv: kv[1]["ceiling_gap"])
    print(f"Best condition: {best[0]}  NLL={best[1]['nll_grafted']:.4f}  "
          f"gap={best[1]['ceiling_gap']:.4f}")

    summary = {
        "model": MODEL_ID, "n_layers": n_layers,
        "nll_donor": float(nll_donor), "nll_lesion": float(nll_lesion),
        "grafting_003_reference": {
            "nll_grafted": 9.6436, "ceiling_gap": 5.7990, "pct_recovered": 59.5,
            "method": "lstsq_minimum_norm", "n_train": 1500
        },
        "conditions": condition_results,
        "best_condition": best[0],
        "elapsed_s": float(time.time() - t0),
    }

    out = RESULTS / "grafting_004_ridge_overdetermined.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
