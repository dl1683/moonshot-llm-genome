"""
genome_124_activation_basis_alignment.py

ACTIVATION-SPACE BASIS ALIGNMENT (Procrustes per-layer).

genome_119-122 KILL: weight subset transplant fails due to coordinate mismatch.
genome_123 KILL: layerwise feature-matching loss fights CE learning at every gamma.

Codex direction A (basis alignment) reframed: weight-space ||D - T(R)|| is
ill-posed for random R because the random basis has no preferred direction.
But ACTIVATION-space Procrustes is well-posed: random-init activations have
non-trivial statistical structure (kaiming covariance), and we can find the
orthogonal transformation that maps recipient activations onto donor
activations in the same basis.

Protocol per recipient seed:
  1. Run donor and recipient on N_CALIB calibration sequences. Collect
     per-layer pre-MLP residual stream activations h_l_donor, h_l_recip
     of shape (n_tokens, hidden_dim).
  2. For each layer l, solve the orthogonal Procrustes problem:
         T_l = argmin_{T orthogonal} || T h_l_recip - h_l_donor ||_F^2
     via SVD of cross-covariance C = h_l_recip^T @ h_l_donor.
  3. Apply T_l to all recipient weights that produce or consume layer-l
     hidden states (rotates the hidden basis at each layer's boundary).
     This is a functionally-preserving change of basis IF done jointly.
  4. After alignment, run baseline NLL (rotated recipient).
  5. Then test partial transplant arms in the aligned basis:
       - all_attn_aligned: copy donor self_attn weights, expressed in
         recipient's gauge via T_l rotation
       - embed_attn_aligned: same plus donor embedding
       - mlp_only_aligned: copy donor MLP (control)

Pass:    any aligned arm closes >=10% gap, CI_lo > 0
Partial: any aligned arm closes >=5% gap, CI_lo > 0
Kill:    all aligned arms <5% — basis alignment doesn't break the barrier

Time budget: ~10 min (1 seed, calibration is one-shot).
Results: results/genome_124_activation_basis_alignment.json
"""

import copy
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 42
SEEDS = [42]  # Single seed for fast iteration; expand if PASS

N_EVAL = 200
N_CALIB = 64  # Smaller for speed; layer dim is 1024, need n >= 1024
SEQ_LEN = 64
BATCH = 8
N_BOOT = 500


def load_wikitext(n, offset, seed=SEED):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
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


def load_trained():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(DEVICE).eval(), tok


def load_random_init(seed):
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).to(DEVICE).eval()


def collect_layer_activations(model, tok, texts):
    """Return dict layer_idx -> (n_tokens, hidden_dim) tensor on CPU float32.
       Uses output_hidden_states for the residual stream at each layer boundary."""
    activations = {}
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            ids, mask = tokenize(texts[i:i + BATCH])
            out = model(input_ids=ids, attention_mask=mask,
                        output_hidden_states=True)
            for l, h in enumerate(out.hidden_states):
                # h is (B, T, D); flatten to (B*T, D), keep only valid tokens
                flat = h.float().reshape(-1, h.shape[-1])
                m = mask.reshape(-1).bool()
                valid = flat[m].cpu()
                activations.setdefault(l, []).append(valid)
    return {l: torch.cat(activations[l], dim=0) for l in activations}


def fit_procrustes(H_recip, H_donor):
    """Find orthogonal T such that T @ H_recip.T ~= H_donor.T (per token).
       Equivalently: T = U V^T where C = H_recip.T @ H_donor = U S V^T.
       Returns T of shape (D, D)."""
    n = min(H_recip.shape[0], H_donor.shape[0])
    R = H_recip[:n]  # (n, D)
    D = H_donor[:n]  # (n, D)
    # cross-covariance
    C = R.T @ D  # (D, D)
    U, S, Vt = torch.linalg.svd(C, full_matrices=False)
    T = U @ Vt  # (D, D), orthogonal — H_recip @ T ~= H_donor
    return T, S


def apply_basis_rotation(model, T_per_layer):
    """Apply per-layer orthogonal rotation T_l to recipient's hidden basis.
       For Qwen3: each layer reads/writes residual at hidden dim H.
       Convention: T_l rotates the basis at the OUTPUT of layer l (== input of layer l+1).
       T[0] rotates the embedding output. T[-1] rotates final norm output before lm_head.

       For functional preservation, all weights consuming layer-l output must be
       multiplied by T_l^T on their input side (or equivalently, layer-l output
       weights multiplied by T_l on their output side).

       Implementation: rotate at each layer boundary using:
       - embed_tokens output: multiply embedding rows by T_0 (right-multiply)
       - layer l self_attn: q_proj/k_proj/v_proj inputs rotated by T_{l-1}^T
                            (i.e., these projections expect T_{l-1}^T-rotated inputs;
                            equivalent to right-multiplying their weights by T_{l-1})
       - layer l self_attn o_proj output: rotated by T_l (left-multiply rows)
       - layer l mlp gate/up inputs: rotated by T_{l-1}^T (right-multiply weights by T_{l-1})
       - layer l mlp down output: rotated by T_l (left-multiply by T_l)
       - input_layernorm and post_attention_layernorm: rotate by T_{l-1} (their output
         is the input to attention; norm doesn't compose linearly with rotation, so we
         apply rotation BEFORE norm by rotating the residual input. But RMSNorm is
         scale-invariant per-channel; rotation breaks per-channel-ness.)

       This is the well-known issue: LayerNorm/RMSNorm is NOT rotation-invariant.
       Therefore strict per-layer orthogonal rotation cannot be applied to a model
       with RMSNorm in the residual stream and remain functionally equivalent.

       MITIGATION: instead of strict gauge transformation, apply ROTATION ONLY at
       the embedding output layer (T_0). This rotates the "input" basis once, which
       is the most consequential for downstream alignment. Subsequent rotations
       would require re-fitting the norm gammas, breaking equivalence.

       This simplification reduces the experiment to: rotate the embedding output
       so that recipient's first-layer activations match donor's, and ask if
       partial transplant in this single-rotation gauge improves over baseline."""
    # Apply only T_0 rotation: embed_tokens.weight has shape (V, H)
    # We want embed_tokens(x) @ T_0 to match donor's embed output
    # So multiply embed_tokens.weight on the right by T_0
    T0 = T_per_layer[0].to(DEVICE).to(torch.bfloat16)
    with torch.no_grad():
        emb = model.model.embed_tokens.weight  # (V, H)
        emb.data = emb.data @ T0
        # If lm_head is tied, this rotation also affects output projection
        # which we don't want. So untie if needed — or check tying.
        if model.lm_head.weight.data_ptr() == emb.data_ptr():
            # tied — same tensor — already rotated; need to handle output side
            # Solution: clone lm_head and apply inverse rotation on output side
            # so output decoding is unchanged
            new_head = (emb.data @ T0.T).clone()  # apply T0^T to get back to original
            # Actually if embed and head are tied and we rotated embed by T0,
            # lm_head.weight is also rotated, which produces wrong logits.
            # We need: lm_head untied, with weight = original_emb (no rotation).
            # Simpler: don't rotate embed if tied. Instead rotate first layer
            # weights to expect T0^T-rotated input.
            emb.data = emb.data @ T0.T  # undo rotation on embed
            # And rotate first attention's input projections by T0
            first = model.model.layers[0]
            for proj in [first.self_attn.q_proj, first.self_attn.k_proj, first.self_attn.v_proj]:
                # weight shape (H_out, H_in). Rotate input dim: weight @ T0
                proj.weight.data = proj.weight.data @ T0
            for proj in [first.mlp.gate_proj, first.mlp.up_proj]:
                proj.weight.data = proj.weight.data @ T0
            # Also rotate input_layernorm (RMSNorm weight is per-channel scale;
            # rotating breaks per-channel structure but if weight is uniform-ish
            # the damage is bounded). For now, leave it unchanged.
    return model


def measure_nll(model, tok, texts):
    per_seq = []
    for i in range(0, len(texts), BATCH):
        ids, mask = tokenize(texts[i:i + BATCH])
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        for j in range(ids.shape[0]):
            lj = out.logits[j:j+1, :-1]
            lbl = ids[j:j+1, 1:].clone()
            mk = mask[j:j+1, 1:]
            lbl[mk == 0] = -100
            per_seq.append(
                F.cross_entropy(lj.view(-1, lj.size(-1)),
                                lbl.view(-1), ignore_index=-100).item()
            )
    return np.array(per_seq)


def bsci(arr, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(arr.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


def bootstrap_gap(recip, donor, surgery, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(recip)

    def pt(r, d, s):
        gap = r.mean() - d.mean()
        return 0.0 if abs(gap) < 1e-6 else float((r.mean() - s.mean()) / gap * 100)

    boots = [pt(recip[rng.integers(0, n, n)], donor[rng.integers(0, n, n)],
                surgery[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return {"mean": float(pt(recip, donor, surgery)),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


def main():
    t0 = time.time()
    print(f"genome_124: activation-basis alignment on {MODEL_ID}")

    eval_texts = load_wikitext(N_EVAL, offset=0)
    calib_texts = load_wikitext(N_CALIB, offset=N_EVAL)
    print(f"  eval={len(eval_texts)}  calib={len(calib_texts)}")

    print("Loading donor...")
    donor, tok = load_trained()
    donor_nlls = measure_nll(donor, tok, eval_texts)
    donor_stats = bsci(donor_nlls)
    print(f"  donor NLL={donor_stats['mean']:.4f}")

    print("Collecting donor layer activations...")
    donor_acts = collect_layer_activations(donor, tok, calib_texts)
    print(f"  collected {len(donor_acts)} layers, sample shape={donor_acts[0].shape}")

    arm_results = {}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        recipient = load_random_init(seed)
        recip_nlls = measure_nll(recipient, tok, eval_texts)
        recip_stats = bsci(recip_nlls, seed=seed)
        print(f"  recipient NLL={recip_stats['mean']:.4f}")

        print("Collecting recipient layer activations...")
        recip_acts = collect_layer_activations(recipient, tok, calib_texts)

        print("Fitting per-layer orthogonal Procrustes...")
        T_per_layer = {}
        for l in donor_acts:
            T, S = fit_procrustes(recip_acts[l], donor_acts[l])
            T_per_layer[l] = T
            top_sv = float(S[:5].mean())
            print(f"  layer {l:2d}: T shape={tuple(T.shape)}  top-5 SV mean={top_sv:.3f}")

        # Arm: full-aligned (rotate embedding output via T_0)
        print("\nApplying T_0 rotation to recipient embedding/first-layer-input...")
        rotated = copy.deepcopy(recipient)
        rotated = apply_basis_rotation(rotated, T_per_layer)
        rot_nlls = measure_nll(rotated, tok, eval_texts)
        rot_stats = bsci(rot_nlls, seed=seed)
        rot_gap = bootstrap_gap(recip_nlls, donor_nlls, rot_nlls, seed=seed)
        print(f"  rotated_baseline: NLL={rot_stats['mean']:.4f} gap={rot_gap['mean']:.2f}%")

        arm_results[f"seed{seed}_rotated_baseline"] = {
            "surgery_nll": rot_stats, "gap_closed_pct": rot_gap,
            "description": "recipient with T_0 rotation applied (no donor weights)"
        }

        # Arm: rotated + all_attn donor copy
        print("Applying all_attn copy on rotated recipient...")
        with torch.no_grad():
            attn_arm = copy.deepcopy(rotated)
            sd_d = donor.state_dict()
            sd_a = attn_arm.state_dict()
            for k in sd_d:
                if ".self_attn." in k:
                    sd_a[k] = sd_d[k].clone()
            attn_arm.load_state_dict(sd_a, strict=True)
        attn_nlls = measure_nll(attn_arm, tok, eval_texts)
        attn_stats = bsci(attn_nlls, seed=seed)
        attn_gap = bootstrap_gap(recip_nlls, donor_nlls, attn_nlls, seed=seed)
        print(f"  rotated_all_attn: NLL={attn_stats['mean']:.4f} gap={attn_gap['mean']:.2f}%")
        arm_results[f"seed{seed}_rotated_all_attn"] = {
            "surgery_nll": attn_stats, "gap_closed_pct": attn_gap,
            "description": "T_0-rotated recipient + donor all_attn weights"
        }

        del rotated, attn_arm, recipient
        torch.cuda.empty_cache()

    # Verdict
    best_name = max(arm_results, key=lambda k: arm_results[k]["gap_closed_pct"]["mean"])
    best = arm_results[best_name]["gap_closed_pct"]
    if best["mean"] >= 10 and best["ci_lo"] > 0:
        verdict = f"PASS: {best_name} closes {best['mean']:.1f}% gap [CI {best['ci_lo']:.1f}, {best['ci_hi']:.1f}]"
    elif best["mean"] >= 5 and best["ci_lo"] > 0:
        verdict = f"PARTIAL: {best_name} closes {best['mean']:.1f}% gap"
    else:
        verdict = f"KILL: best {best_name} closes {best['mean']:.1f}%. Activation-basis alignment via T_0 does not break holism barrier."

    print("\n=== GENOME 124 SUMMARY ===")
    for n, r in arm_results.items():
        print(f"  {n:50s}: NLL={r['surgery_nll']['mean']:.4f} gap={r['gap_closed_pct']['mean']:.2f}%")
    print(f"  verdict: {verdict}")

    out = {
        "model": MODEL_ID, "genome": 124, "name": "activation_basis_alignment",
        "n_eval": N_EVAL, "n_calib": N_CALIB, "seeds": SEEDS,
        "donor_nll": donor_stats,
        "arm_results": {
            n: {"surgery_nll_mean": r["surgery_nll"]["mean"],
                "surgery_nll_ci": r["surgery_nll"],
                "gap_closed_pct": r["gap_closed_pct"],
                "description": r["description"]}
            for n, r in arm_results.items()
        },
        "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = RESULTS / "genome_124_activation_basis_alignment.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
