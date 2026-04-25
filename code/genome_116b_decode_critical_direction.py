"""
genome_116b_decode_critical_direction.py

What IS the critical direction at layer 5?

genome_115 showed PC1 at layer 5 carries 4.46 nats of capability (906x vs
random). Before surgery we need to know WHAT this direction encodes. Is it:
  (a) A generic language-model axis (frequency / position / token-type)?
  (b) A syntactic structure axis?
  (c) A semantic content axis (topics, entities)?
  (d) A computation-state axis (attention sink / output-gate activations)?

Protocol
--------
1. Feed 2000 diverse wikitext tokens through the model.
2. At layer 5, collect per-TOKEN (not per-sequence) hidden states.
3. Project each token's hidden state onto PC1.
4. Find top-50 (high) and bottom-50 (low) scoring tokens.
5. Report: token strings, their contexts, and PC1 projection values.
6. Additionally: fit vocabulary logit lens at layer 5 and project vocab onto PC1
   to find which output tokens the direction "points toward."

This takes ~5 minutes on RTX 5090 and answers the EV=8.0 Codex question.

Results: results/genome_116b_decode_critical_direction.json
"""

import json
import pathlib
import time
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT    = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED     = 42

N_FIT    = 200   # for PCA fit (same as genome_115)
N_PROBE  = 500   # sequences for token-level projections
SEQ_LEN  = 64
BATCH    = 8
SURGERY_LAYER = 5
TOP_K    = 50


def load_wikitext_split(n, offset, seed=SEED):
    ds  = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    out, count = [], 0
    for idx in perm:
        t = ds[int(idx)]["text"].strip()
        if len(t) < 60:
            continue
        if count >= offset:
            out.append(t[:300])
        count += 1
        if len(out) >= n:
            break
    return out


def tokenize(texts, tok):
    enc = tok(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=SEQ_LEN)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def main():
    t0 = time.time()
    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    n_layers = model.config.num_hidden_layers
    d_model  = model.config.hidden_size
    print(f"  {MODEL_ID}: {n_layers} layers, d={d_model}")

    # -----------------------------------------------------------------------
    # Step 1: Fit PCA on layer-5 SEQUENCE-MEAN activations (same as genome_115)
    # -----------------------------------------------------------------------
    print(f"\nFitting PCA on layer-{SURGERY_LAYER} activations (n={N_FIT})...")
    fit_texts = load_wikitext_split(N_FIT, offset=0)
    seq_pooled = []

    def hook_seq(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        seq_pooled.append(h.detach().float().mean(dim=1).cpu())

    handle = model.model.layers[SURGERY_LAYER].register_forward_hook(hook_seq)
    for i in range(0, len(fit_texts), BATCH):
        ids, mask = tokenize(fit_texts[i:i+BATCH], tok)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
    handle.remove()

    fit_acts = torch.cat(seq_pooled, dim=0).numpy()
    pca = PCA(n_components=10)
    pca.fit(fit_acts)
    dir_pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-8)
    var_pc1 = float(pca.explained_variance_ratio_[0])
    print(f"  PC1 explains {var_pc1:.1%} of sequence-mean variance")

    # -----------------------------------------------------------------------
    # Step 2: Per-TOKEN projections on probe split
    # -----------------------------------------------------------------------
    print(f"\nCollecting per-token projections (n={N_PROBE} seqs)...")
    probe_texts = load_wikitext_split(N_PROBE, offset=N_FIT)

    token_acts  = []   # (n_total_tokens, d_model)
    token_ids_list = []
    token_ctx_list = []  # (seq_idx, pos_in_seq)

    per_tok_acts = []

    def hook_tok(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out  # (B, T, D)
        per_tok_acts.append(h.detach().float().cpu())

    handle = model.model.layers[SURGERY_LAYER].register_forward_hook(hook_tok)
    seq_ids_storage = []

    for i in range(0, len(probe_texts), BATCH):
        batch = probe_texts[i:i+BATCH]
        ids, mask = tokenize(batch, tok)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        seq_ids_storage.append(ids.cpu())

    handle.remove()

    # Flatten: for each batch, each (seq, pos) with non-pad token
    dir_pc1_t = torch.tensor(dir_pc1, dtype=torch.float32)
    projections = []
    proj_token_ids = []
    proj_contexts = []

    for batch_idx, (acts_b, ids_b) in enumerate(zip(per_tok_acts, seq_ids_storage)):
        # acts_b: (B, T, D), ids_b: (B, T)
        B, T, D = acts_b.shape
        projs = (acts_b @ dir_pc1_t).numpy()  # (B, T)
        for b in range(B):
            for t in range(T):
                tid = ids_b[b, t].item()
                if tid == tok.pad_token_id:
                    continue
                projections.append(float(projs[b, t]))
                proj_token_ids.append(tid)
                # context: a few tokens around this position
                start = max(0, t-3)
                end   = min(T, t+4)
                ctx_ids = ids_b[b, start:end].tolist()
                proj_contexts.append(tok.decode(ctx_ids, skip_special_tokens=True))

    projections = np.array(projections)
    print(f"  Collected {len(projections)} token projections")
    print(f"  Projection range: [{projections.min():.3f}, {projections.max():.3f}]")
    print(f"  Mean: {projections.mean():.3f}, Std: {projections.std():.3f}")

    # -----------------------------------------------------------------------
    # Step 3: Top and bottom tokens by projection
    # -----------------------------------------------------------------------
    top_idxs    = np.argsort(projections)[-TOP_K:][::-1]
    bottom_idxs = np.argsort(projections)[:TOP_K]

    def make_entry(idx):
        return {
            "token": tok.decode([proj_token_ids[idx]]).strip(),
            "token_id": proj_token_ids[idx],
            "projection": float(projections[idx]),
            "context": proj_contexts[idx],
        }

    top_tokens    = [make_entry(i) for i in top_idxs]
    bottom_tokens = [make_entry(i) for i in bottom_idxs]

    # Aggregate: which token TYPES have highest mean projection
    token_proj_map = defaultdict(list)
    for tid, proj in zip(proj_token_ids, projections):
        token_proj_map[tid].append(proj)

    token_mean_proj = {
        tok.decode([tid]).strip(): float(np.mean(vals))
        for tid, vals in token_proj_map.items()
        if len(vals) >= 5  # only tokens seen >=5 times
    }
    top_type_tokens    = sorted(token_mean_proj.items(), key=lambda x: x[1], reverse=True)[:30]
    bottom_type_tokens = sorted(token_mean_proj.items(), key=lambda x: x[1])[:30]

    def safe(s):
        return s.encode("ascii", errors="replace").decode("ascii")

    print("\n=== TOP TOKEN TYPES (high PC1 projection) ===")
    for tok_str, proj in top_type_tokens[:15]:
        print(f"  {repr(safe(tok_str)):20s}  mean_proj={proj:.3f}")

    print("\n=== BOTTOM TOKEN TYPES (low PC1 projection) ===")
    for tok_str, proj in bottom_type_tokens[:15]:
        print(f"  {repr(safe(tok_str)):20s}  mean_proj={proj:.3f}")

    # -----------------------------------------------------------------------
    # Step 4: Vocabulary logit-lens projection
    # -----------------------------------------------------------------------
    print("\nComputing vocabulary logit-lens projections...")
    # Project embedding matrix onto PC1 to find which output tokens dir aligns with
    with torch.no_grad():
        embed = model.model.embed_tokens.weight.float().cpu().numpy()  # (V, D)
    vocab_proj = embed @ dir_pc1  # (V,)
    top_vocab_idxs    = np.argsort(vocab_proj)[-30:][::-1]
    bottom_vocab_idxs = np.argsort(vocab_proj)[:30]

    top_vocab    = [{"token": tok.decode([int(i)]).strip(), "proj": float(vocab_proj[i])} for i in top_vocab_idxs]
    bottom_vocab = [{"token": tok.decode([int(i)]).strip(), "proj": float(vocab_proj[i])} for i in bottom_vocab_idxs]

    def safe(s):
        return s.encode("ascii", errors="replace").decode("ascii")

    print("Top vocab tokens (high PC1 in embedding space):")
    for e in top_vocab[:10]:
        print(f"  {repr(safe(e['token'])):20s}  proj={e['proj']:.3f}")
    print("Bottom vocab tokens:")
    for e in bottom_vocab[:10]:
        print(f"  {repr(safe(e['token'])):20s}  proj={e['proj']:.3f}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out = {
        "model": MODEL_ID,
        "surgery_layer": SURGERY_LAYER,
        "pca_var_pc1": var_pc1,
        "n_tokens_probed": len(projections),
        "projection_stats": {
            "mean": float(projections.mean()),
            "std": float(projections.std()),
            "min": float(projections.min()),
            "max": float(projections.max()),
        },
        "top_instance_tokens": top_tokens,
        "bottom_instance_tokens": bottom_tokens,
        "top_type_tokens": [{"token": t, "mean_proj": p} for t, p in top_type_tokens],
        "bottom_type_tokens": [{"token": t, "mean_proj": p} for t, p in bottom_type_tokens],
        "top_vocab_projection": top_vocab,
        "bottom_vocab_projection": bottom_vocab,
        "elapsed_s": time.time() - t0,
    }

    out_path = RESULTS / "genome_116b_decode_critical_direction.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
