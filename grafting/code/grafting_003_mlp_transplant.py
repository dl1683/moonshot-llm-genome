"""
grafting_003_mlp_transplant.py

Kill test for analytical MLP transplant (zero gradient steps).

Hypothesis: if T_l is a shared transition law (grafting_002 PASS, ratio=0.93),
we can analytically recover a lesioned model's MLP weights by solving a
linear system from donor activations — bypassing gradient descent entirely.

Protocol:
  Donor:      trained Qwen3-0.6B (all weights intact)
  Recipient:  same model with all MLP down_proj weights zeroed (MLP-lesioned)

  Grafting (zero gradient steps):
    For each layer L:
      1. Run N_TRAIN texts through donor + lesioned model in parallel
      2. MLP target:       r_L = h_{L+1}^donor - h_{L+1}^lesion  (N, hidden)
         (= donor MLP contribution, since attention weights are identical)
      3. MLP intermediate: f_L = silu(gate_proj @ h_L.T) * (up_proj @ h_L.T)
         (N, intermediate) — computed inside lesioned model using its intact gate/up
      4. Solve:            down_proj_new = lstsq(f_L, r_L)
         (intermediate, hidden) — installed as down_proj.weight.T
      5. Install in recipient (no gradient steps, pure linear algebra)

  Evaluation on N_TEST held-out texts:
    - Lesioned NLL  (baseline — before grafting)
    - Grafted NLL   (after transplant, 0 steps)
    - Donor NLL     (ceiling — not achievable for same model!)
    - genome_087 FM+KL 200-step reference: NLL ≈ 3.3-3.5 (from ledger)

Kill: grafted NLL not better than lesioned NLL -> analytical transplant fails
Pass: grafted NLL within 0.5 nats of donor NLL -> analytical transplant succeeds
"""

import json
import pathlib
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

ROOT = pathlib.Path(__file__).parent.parent.parent
RESULTS = pathlib.Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_TRAIN = 1500   # texts for lstsq solve
N_TEST  = 300    # held-out evaluation texts
N_TOTAL = N_TRAIN + N_TEST
SEQ_LEN = 128
BATCH   = 16
MODEL_ID = "Qwen/Qwen3-0.6B"


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
    """Mean token-level NLL on texts (lower is better)."""
    model.eval()
    total_nll, total_toks = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(DEVICE)
            out = model(**enc)
            logits = out.logits[:, :-1].float()
            labels = enc["input_ids"][:, 1:].clone()
            labels[enc["attention_mask"][:, 1:] == 0] = -100
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100,
                                   reduction="sum")
            n_toks = (labels != -100).sum().item()
            total_nll += loss.item()
            total_toks += n_toks
    return total_nll / max(total_toks, 1)


def collect_hidden(model, tokenizer, texts, n_layers):
    """Collect mean-pooled hidden states at every layer. No hooks needed."""
    hidden = {l: [] for l in range(n_layers + 1)}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=SEQ_LEN).to(DEVICE)
            out = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            for l, h in enumerate(out.hidden_states):
                pooled = (h.float() * mask).sum(1) / mask.sum(1)
                hidden[l].append(pooled.cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in hidden.items()}


def collect_mlp_intermediates(model, tokenizer, texts, n_layers):
    """
    Collect mean-pooled pre-down_proj intermediates at every MLP layer.
    Hook pools immediately and moves to CPU to avoid GPU OOM.
    """
    mlp_interm = {l: [] for l in range(n_layers)}
    hooks = []

    def make_pre_hook(layer_idx):
        def pre_hook(mod, inp):
            # inp[0]: (B, T, intermediate_dim) — pool over T, move to CPU
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
            model(**enc)   # forward only — hooks capture intermediates

    for h in hooks:
        h.remove()

    return {l: np.concatenate(v, axis=0) for l, v in mlp_interm.items()}


def analytical_transplant(model_lesion, hidden_donor, hidden_lesion,
                           mlp_interm_lesion, n_layers, n_train):
    """
    For each layer L, solve:
        down_proj @ f_L.T ≈ r_L.T
    where:
        r_L = h_{L+1}^donor - h_{L+1}^lesion   (MLP target)
        f_L = mlp_interm_lesion[L]               (intermediate activations)

    Install the solved down_proj back into the lesioned model.
    Only uses training split (first n_train rows).
    """
    results_per_layer = {}
    for L in range(n_layers):
        # MLP target: (n_train, hidden_dim)
        r = (hidden_donor[L + 1][:n_train] - hidden_lesion[L + 1][:n_train]).astype(np.float32)
        # MLP intermediate: (n_train, intermediate_dim)
        f = mlp_interm_lesion[L][:n_train].astype(np.float32)

        # Solve: down_proj.weight is (hidden, intermediate)
        # We want: f @ down_proj.weight.T ≈ r
        # i.e., down_proj.weight.T = lstsq(f, r) → shape (intermediate, hidden)
        # so down_proj.weight = solution.T → (hidden, intermediate)
        sol, residuals, rank, sv = np.linalg.lstsq(f, r, rcond=None)
        # sol: (intermediate, hidden)

        r_norm  = float(np.linalg.norm(r))
        pred    = f @ sol
        res_norm = float(np.linalg.norm(r - pred))
        r2 = 1.0 - (res_norm ** 2) / (r_norm ** 2 + 1e-12)

        with torch.no_grad():
            w_tensor = torch.tensor(sol.T, dtype=torch.float16).to(DEVICE)
            model_lesion.model.layers[L].mlp.down_proj.weight.copy_(w_tensor)

        results_per_layer[L] = {"rank": int(rank), "r2_train": float(r2)}
        if L % 4 == 0:
            print(f"  Layer {L:2d}: rank={rank}, train R²={r2:.4f}, "
                  f"|r|={r_norm:.2f}, |res|={res_norm:.2f}")

    return results_per_layer


def main():
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading texts...")
    texts = load_texts(N_TOTAL, SEED)
    train_texts = texts[:N_TRAIN]
    test_texts  = texts[N_TRAIN:N_TOTAL]
    print(f"  train={len(train_texts)} test={len(test_texts)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- Pass 1: donor (trained model) ----
    print(f"\n[{time.time()-t0:.1f}s] Loading donor (trained {MODEL_ID})...")
    donor = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    n_layers = donor.config.num_hidden_layers
    print(f"  n_layers={n_layers}")

    nll_donor = measure_nll(donor, tok, test_texts)
    print(f"  Donor NLL (test): {nll_donor:.4f}")

    print(f"  Collecting donor hidden states (train)...")
    h_donor = collect_hidden(donor, tok, train_texts, n_layers)
    print(f"  Done. shape: {h_donor[0].shape}")

    del donor
    torch.cuda.empty_cache()

    # ---- Pass 2: lesioned recipient ----
    print(f"\n[{time.time()-t0:.1f}s] Creating lesioned recipient (down_proj=0)...")
    recipient = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)

    with torch.no_grad():
        for L in range(n_layers):
            recipient.model.layers[L].mlp.down_proj.weight.zero_()

    nll_lesion = measure_nll(recipient, tok, test_texts)
    print(f"  Lesioned NLL (test): {nll_lesion:.4f}")

    print(f"\n[{time.time()-t0:.1f}s] Collecting lesion hidden states + MLP intermediates...")
    h_lesion   = collect_hidden(recipient, tok, train_texts, n_layers)
    mlp_interm = collect_mlp_intermediates(recipient, tok, train_texts, n_layers)
    print(f"  Done. hidden: {h_lesion[0].shape}, interm: {mlp_interm[0].shape}")

    print(f"\n[{time.time()-t0:.1f}s] Analytical transplant (0 gradient steps)...")
    layer_results = analytical_transplant(
        recipient, h_donor, h_lesion, mlp_interm, n_layers, N_TRAIN
    )

    # Reload donor NLL on test using recipient model (same weights, different object)
    # donor was deleted above — load fresh copy just for ceiling measurement
    print(f"\n[{time.time()-t0:.1f}s] Measuring grafted NLL...")
    nll_grafted = measure_nll(recipient, tok, test_texts)
    print(f"\n[{time.time()-t0:.1f}s] NLL after grafting (0 steps): {nll_grafted:.4f}")

    # Sanity: also measure on training texts to check for overfitting
    nll_grafted_train = measure_nll(recipient, tok, train_texts[:200])
    print(f"  NLL on train sample (200 texts): {nll_grafted_train:.4f}")

    # ---- Summary ----
    improvement = nll_lesion - nll_grafted
    ceiling_gap = nll_grafted - nll_donor

    if ceiling_gap < 0.5:
        verdict = "PASS: grafted NLL within 0.5 nats of donor — transplant succeeds"
    elif improvement > 1.0:
        verdict = "PARTIAL: significant improvement but not within ceiling gap"
    elif improvement > 0.1:
        verdict = "WEAK: minor improvement — transplant partially works"
    else:
        verdict = "KILL: no improvement from analytical transplant"

    summary = {
        "model":         MODEL_ID,
        "n_train":       N_TRAIN,
        "n_test":        N_TEST,
        "n_layers":      n_layers,
        "nll_donor":     float(nll_donor),
        "nll_lesion":    float(nll_lesion),
        "nll_grafted":   float(nll_grafted),
        "improvement":   float(improvement),
        "ceiling_gap":   float(ceiling_gap),
        "reference_genome_087_fm_kl_200step": "~3.3-3.5 nats (from ledger)",
        "verdict":       verdict,
        "layer_results": {str(k): v for k, v in layer_results.items()},
        "elapsed_s":     float(time.time() - t0),
    }

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"  Donor NLL:   {nll_donor:.4f}")
    print(f"  Lesion NLL:  {nll_lesion:.4f}")
    print(f"  Grafted NLL: {nll_grafted:.4f}")
    print(f"  Improvement: {improvement:.4f} nats")
    print(f"  Ceiling gap: {ceiling_gap:.4f} nats (0 = perfect recovery)")
    print(f"{'='*60}")

    out = RESULTS / "grafting_003_mlp_transplant.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
