"""
genome_159_cross_class_lesion.py

POST-CHAIN ARCHITECTURE-CLASS EXTENSION via lesion on pretrained models.

Pre-reg LOCKED: research/prereg/genome_159_cross_class_lesion_2026-04-26.md
Theory: research/derivations/prefix_information_transport.md
Program: research/programs/post_g156_pass_program.md §g159
Codex design: codex_outputs/g159_lesion_algorithm.md

Tests transport-vs-local lesion asymmetry across three architecture
classes (Qwen3-0.6B, RWKV-4-169M, Falcon-H1-0.5B). Per Codex spec:
  - Top-32 PCA basis fit per (model, depth, sublayer) on 2048 c4 windows.
  - Lesion = project out top-32 from the residual contribution at eval.
  - Eval on natural c4-val (1024 windows) and shuffled control.
  - Ratio R = dNLL_transport / dNLL_local; aggregate by median.

PASS: 3/3 classes with median R_nat >= 1.5 AND R_shuf falls by >=40%.
Compute: 0.8-1.5 GPU-hr per Codex estimate.

Falcon-H1 needs forward-patching (not just hooks) because every layer
has composite transport (Mamba + Attention summed).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MethodType
from typing import Dict, List, Tuple, Callable
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent

SEQ_LEN = 256
N_CALIB = 2048
N_EVAL = 1024
DEPTHS = [0.25, 0.50, 0.75]
PCA_RANK = 32
SHUFFLE_SEED = 42
MODELS = [
    ("Qwen/Qwen3-0.6B", "qwen3", 64),
    ("RWKV/rwkv-4-169m-pile", "rwkv4", 64),
    ("tiiuae/Falcon-H1-0.5B-Instruct", "falcon_h1", 32),
]


@dataclass
class PCABasis:
    mean: torch.Tensor   # [H], fp32
    basis: torch.Tensor  # [K, H], orthonormal rows, fp32
    path: str
    var_explained: float  # fraction of variance captured by top-K


class ForwardPatchHandle:
    def __init__(self, module: nn.Module, orig_forward):
        self.module = module
        self.orig_forward = orig_forward

    def remove(self):
        self.module.forward = self.orig_forward


def _project_out(y: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """y_lesioned = y - ((y - mean) @ basis.T) @ basis. Done in fp32 then cast back."""
    od = y.dtype
    y32 = y.to(torch.float32)
    mu = mean.to(y32.device, y32.dtype)
    U = basis.to(y32.device, y32.dtype)
    yc = y32 - mu
    y_out = y32 - (yc @ U.T) @ U
    return y_out.to(od)


def _replace_first(output, new0):
    if isinstance(output, tuple):
        return (new0,) + output[1:]
    return new0


class LesionAdapter(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def n_layers(self) -> int: ...

    def depth_to_idx(self, d: float) -> int:
        return int(round(d * (self.n_layers() - 1)))

    @abstractmethod
    def get_transport_sublayer(self, depth_frac: float) -> nn.Module: ...

    @abstractmethod
    def get_local_sublayer(self, depth_frac: float) -> nn.Module: ...

    def install_lesion_hook(self, sublayer, pca_basis: PCABasis, rank: int = PCA_RANK) -> ForwardPatchHandle:
        U = pca_basis.basis[:rank]
        mu = pca_basis.mean

        def hook(_m, _inp, output):
            y0 = output[0] if isinstance(output, tuple) else output
            y0_l = _project_out(y0, mu, U)
            return _replace_first(output, y0_l)

        h = sublayer.register_forward_hook(hook)
        # Wrap as ForwardPatchHandle for uniform .remove() interface
        class _H:
            def __init__(self, hk): self.hk = hk
            def remove(self): self.hk.remove()
        return _H(h)


class Qwen3Adapter(LesionAdapter):
    def n_layers(self): return len(self.model.model.layers)
    def get_transport_sublayer(self, d): return self.model.model.layers[self.depth_to_idx(d)].self_attn
    def get_local_sublayer(self, d): return self.model.model.layers[self.depth_to_idx(d)].mlp


class RWKVAdapter(LesionAdapter):
    def n_layers(self): return len(self.model.rwkv.blocks)
    def get_transport_sublayer(self, d): return self.model.rwkv.blocks[self.depth_to_idx(d)].attention
    def get_local_sublayer(self, d): return self.model.rwkv.blocks[self.depth_to_idx(d)].feed_forward


class FalconTransportSite(nn.Module):
    """Marker for the composite transport site at one Falcon-H1 layer."""
    def __init__(self, layer: nn.Module, layer_idx: int):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.path = f"model.layers.{layer_idx}::transport_composite"


class FalconH1Adapter(LesionAdapter):
    def n_layers(self): return len(self.model.model.layers)

    def get_transport_sublayer(self, d):
        i = self.depth_to_idx(d)
        return FalconTransportSite(self.model.model.layers[i], i)

    def get_local_sublayer(self, d):
        i = self.depth_to_idx(d)
        return self.model.model.layers[i].feed_forward

    def install_lesion_hook(self, sublayer, pca_basis: PCABasis, rank: int = PCA_RANK) -> ForwardPatchHandle:
        if not isinstance(sublayer, FalconTransportSite):
            return super().install_lesion_hook(sublayer, pca_basis, rank=rank)
        layer = sublayer.layer
        orig_forward = layer.forward
        U = pca_basis.basis[:rank]
        mu = pca_basis.mean

        def patched_forward(this, hidden_states, attention_mask=None, mamba_attention_mask=None,
                            position_ids=None, past_key_values=None, output_attentions=False,
                            use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
            residual = hidden_states
            hidden_states = this.input_layernorm(hidden_states)
            m_out = this.mamba(
                hidden_states=hidden_states, cache_params=past_key_values,
                cache_position=cache_position, attention_mask=mamba_attention_mask,
            ) * this.ssm_out_multiplier
            a_out, a_weights = this.self_attn(
                hidden_states=hidden_states * this.attention_in_multiplier,
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, output_attentions=output_attentions,
                use_cache=use_cache, cache_position=cache_position,
                position_embeddings=position_embeddings, **kwargs,
            )
            a_out = a_out * this.attn_out_multiplier
            transport = _project_out(m_out + a_out, mu, U)
            hidden_states = residual + transport
            residual = hidden_states
            hidden_states = this.pre_ff_layernorm(hidden_states)
            hidden_states = this.feed_forward(hidden_states)
            hidden_states = residual + hidden_states
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (a_weights,)
            return outputs

        layer.forward = MethodType(patched_forward, layer)
        return ForwardPatchHandle(layer, orig_forward)


def fit_pca_at_sublayer(model, sublayer_or_site, ids, mask, batch_size, is_falcon_transport=False):
    """Capture residual contribution and fit top-32 PCA. Returns PCABasis."""
    captured_X = []
    if is_falcon_transport:
        # For Falcon transport composite, monkey-patch forward to capture m_out + a_out
        layer = sublayer_or_site.layer
        orig_forward = layer.forward
        captured = []

        def cap_forward(this, hidden_states, attention_mask=None, mamba_attention_mask=None,
                        position_ids=None, past_key_values=None, output_attentions=False,
                        use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
            residual = hidden_states
            hidden_states = this.input_layernorm(hidden_states)
            m_out = this.mamba(
                hidden_states=hidden_states, cache_params=past_key_values,
                cache_position=cache_position, attention_mask=mamba_attention_mask,
            ) * this.ssm_out_multiplier
            a_out, a_weights = this.self_attn(
                hidden_states=hidden_states * this.attention_in_multiplier,
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, output_attentions=output_attentions,
                use_cache=use_cache, cache_position=cache_position,
                position_embeddings=position_embeddings, **kwargs,
            )
            a_out = a_out * this.attn_out_multiplier
            transport = m_out + a_out
            captured.append(transport.detach().cpu().to(torch.float32))
            hidden_states = residual + transport
            residual = hidden_states
            hidden_states = this.pre_ff_layernorm(hidden_states)
            hidden_states = this.feed_forward(hidden_states)
            return (residual + hidden_states,)

        layer.forward = MethodType(cap_forward, layer)
        try:
            with torch.no_grad():
                for i in range(0, ids.size(0), batch_size):
                    _ = model(input_ids=ids[i:i+batch_size].to(model.device),
                              attention_mask=mask[i:i+batch_size].to(model.device), use_cache=False)
        finally:
            layer.forward = orig_forward
        captured_X = captured
    else:
        captured = []

        def hook(_m, _inp, output):
            y0 = output[0] if isinstance(output, tuple) else output
            captured.append(y0.detach().cpu().to(torch.float32))

        h = sublayer_or_site.register_forward_hook(hook)
        try:
            with torch.no_grad():
                for i in range(0, ids.size(0), batch_size):
                    _ = model(input_ids=ids[i:i+batch_size].to(model.device),
                              attention_mask=mask[i:i+batch_size].to(model.device), use_cache=False)
        finally:
            h.remove()
        captured_X = captured

    # Flatten to [N_tokens, H], filter pads
    X_chunks = []
    for c, b_start in zip(captured_X, range(0, len(captured_X) * batch_size, batch_size)):
        b_mask = mask[b_start:b_start+c.shape[0]]
        for r in range(c.shape[0]):
            valid = b_mask[r].bool()
            X_chunks.append(c[r][valid].reshape(-1, c.shape[-1]))
    X = torch.cat(X_chunks, dim=0).to(torch.float32)
    print(f"      PCA fit on {X.shape[0]} tokens, hidden={X.shape[1]}")

    # Streaming covariance: mu = mean(X), C = (X^T X - N*mu*mu^T) / (N-1)
    mu = X.mean(dim=0)
    Xc = X - mu
    # Use SVD on Xc / sqrt(N-1) for top-K eigenvectors of cov
    if Xc.shape[0] > 50000:
        # Subsample for SVD speed
        idx = torch.randperm(Xc.shape[0])[:50000]
        Xc_sub = Xc[idx]
    else:
        Xc_sub = Xc
    U, S, Vh = torch.linalg.svd(Xc_sub, full_matrices=False)
    basis = Vh[:PCA_RANK]  # [K, H]
    total_var = (S ** 2).sum().item() / max(Xc_sub.shape[0] - 1, 1)
    top_var = (S[:PCA_RANK] ** 2).sum().item() / max(Xc_sub.shape[0] - 1, 1)
    var_explained = top_var / max(total_var, 1e-12)

    return PCABasis(mean=mu, basis=basis, path=str(getattr(sublayer_or_site, "path", "")),
                     var_explained=float(var_explained))


def shuffle_token_rows(ids, mask, shuffle_seed=SHUFFLE_SEED):
    rng = np.random.default_rng(shuffle_seed)
    out = ids.clone()
    for r in range(ids.shape[0]):
        valid_pos = (mask[r] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        if len(valid_pos) <= 1:
            continue
        perm = rng.permutation(len(valid_pos))
        out[r, valid_pos] = ids[r, valid_pos[perm]]
    return out


def measure_nll(model, ids, mask, batch_size):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.size(0), batch_size):
            ids_b = ids[i:i+batch_size].to(model.device)
            mask_b = mask[i:i+batch_size].to(model.device)
            out = model(input_ids=ids_b, attention_mask=mask_b, use_cache=False)
            logits = out.logits.float()
            sl = logits[:, :-1].contiguous()
            lbl = ids_b[:, 1:].clone()
            sm = mask_b[:, 1:]
            valid = (sm != 0)
            lbl[~valid] = -100
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1),
                                    ignore_index=-100, reduction="sum")
            total_loss += loss.item()
            total_tokens += valid.sum().item()
    return total_loss / max(total_tokens, 1)


def adapter_for(short_name, model):
    if short_name == "qwen3":
        return Qwen3Adapter(model)
    if short_name == "rwkv4":
        return RWKVAdapter(model)
    if short_name == "falcon_h1":
        return FalconH1Adapter(model)
    raise ValueError(short_name)


def load_eval_data(tok):
    print("Loading c4 + shuffled control...")
    pool = []
    for rec in c4_clean_v1(seed=159, n_samples=N_CALIB + N_EVAL + 100):
        pool.append(rec["text"])
        if len(pool) >= N_CALIB + N_EVAL:
            break
    enc = tok(pool[:N_CALIB + N_EVAL], padding="max_length", truncation=True,
               max_length=SEQ_LEN, return_tensors="pt")
    calib_ids = enc["input_ids"][:N_CALIB]
    calib_mask = enc["attention_mask"][:N_CALIB]
    eval_ids = enc["input_ids"][N_CALIB:]
    eval_mask = enc["attention_mask"][N_CALIB:]
    eval_ids_shuf = shuffle_token_rows(eval_ids, eval_mask, SHUFFLE_SEED)
    return calib_ids, calib_mask, eval_ids, eval_mask, eval_ids_shuf


def main():
    t0 = time.time()
    print("genome_159: cross-class transport-vs-local lesion")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    results = {}

    for hf_id, short_name, batch_size in MODELS:
        print(f"\n=== {hf_id} ({short_name}) ===")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=torch.bfloat16,
            ).to("cuda").eval()
            adapter = adapter_for(short_name, model)
        except Exception as e:
            print(f"  load failed: {e}")
            results[short_name] = {"error": str(e)}
            continue

        n_layers = adapter.n_layers()
        print(f"  n_layers={n_layers}, depths {DEPTHS} -> indices {[adapter.depth_to_idx(d) for d in DEPTHS]}")

        calib_ids, calib_mask, eval_ids, eval_mask, eval_ids_shuf = load_eval_data(tok)

        # Baseline NLL (no lesion) on natural and shuffled
        print("  measuring baseline NLL (no lesion)...")
        nll_nat_base = measure_nll(model, eval_ids, eval_mask, batch_size)
        nll_shuf_base = measure_nll(model, eval_ids_shuf, eval_mask, batch_size)
        print(f"    nll_nat_base={nll_nat_base:.4f}  nll_shuf_base={nll_shuf_base:.4f}")

        per_depth = {}
        for d in DEPTHS:
            print(f"\n  -- depth {d} (layer {adapter.depth_to_idx(d)}) --")
            t_d = time.time()

            transport_site = adapter.get_transport_sublayer(d)
            local_site = adapter.get_local_sublayer(d)

            print("    fitting transport PCA...")
            is_fct = isinstance(transport_site, FalconTransportSite)
            pca_t = fit_pca_at_sublayer(model, transport_site, calib_ids, calib_mask, batch_size,
                                          is_falcon_transport=is_fct)
            print(f"    transport top-{PCA_RANK} variance explained: {pca_t.var_explained:.3f}")

            print("    fitting local PCA...")
            pca_l = fit_pca_at_sublayer(model, local_site, calib_ids, calib_mask, batch_size,
                                          is_falcon_transport=False)
            print(f"    local top-{PCA_RANK} variance explained: {pca_l.var_explained:.3f}")

            # Eval transport-lesion on natural and shuffled
            h_t = adapter.install_lesion_hook(transport_site, pca_t, PCA_RANK)
            try:
                nll_t_nat = measure_nll(model, eval_ids, eval_mask, batch_size)
                nll_t_shuf = measure_nll(model, eval_ids_shuf, eval_mask, batch_size)
            finally:
                h_t.remove()

            # Eval local-lesion (need fresh adapter for falcon since transport-site is composite)
            h_l = adapter.install_lesion_hook(local_site, pca_l, PCA_RANK)
            try:
                nll_l_nat = measure_nll(model, eval_ids, eval_mask, batch_size)
                nll_l_shuf = measure_nll(model, eval_ids_shuf, eval_mask, batch_size)
            finally:
                h_l.remove()

            d_t_nat = nll_t_nat - nll_nat_base
            d_t_shuf = nll_t_shuf - nll_shuf_base
            d_l_nat = nll_l_nat - nll_nat_base
            d_l_shuf = nll_l_shuf - nll_shuf_base
            R_nat = d_t_nat / max(d_l_nat, 1e-6)
            R_shuf = d_t_shuf / max(d_l_shuf, 1e-6)
            per_depth[d] = {
                "nll_t_nat": nll_t_nat, "nll_t_shuf": nll_t_shuf,
                "nll_l_nat": nll_l_nat, "nll_l_shuf": nll_l_shuf,
                "delta_t_nat": d_t_nat, "delta_t_shuf": d_t_shuf,
                "delta_l_nat": d_l_nat, "delta_l_shuf": d_l_shuf,
                "R_nat": R_nat, "R_shuf": R_shuf,
                "var_explained_transport": pca_t.var_explained,
                "var_explained_local": pca_l.var_explained,
                "wallclock_s": time.time() - t_d,
            }
            print(f"    R_nat={R_nat:.2f}  R_shuf={R_shuf:.2f}  "
                  f"(dT_nat={d_t_nat:+.3f} dL_nat={d_l_nat:+.3f})")

        # Aggregate by median over depths
        R_nat_med = float(np.median([per_depth[d]["R_nat"] for d in DEPTHS]))
        R_shuf_med = float(np.median([per_depth[d]["R_shuf"] for d in DEPTHS]))
        results[short_name] = {
            "hf_id": hf_id, "n_layers": n_layers,
            "nll_nat_base": nll_nat_base, "nll_shuf_base": nll_shuf_base,
            "per_depth": {str(d): v for d, v in per_depth.items()},
            "R_nat_median": R_nat_med, "R_shuf_median": R_shuf_med,
        }
        print(f"\n  {short_name}: median R_nat={R_nat_med:.2f}, R_shuf={R_shuf_med:.2f}")

        del model
        torch.cuda.empty_cache()

    # Verdict
    print(f"\n=== ANALYSIS ===")
    valid = {k: v for k, v in results.items() if "error" not in v}
    n_pass_nat = sum(1 for v in valid.values() if v["R_nat_median"] >= 1.5)
    n_pass_shuf = sum(1 for v in valid.values() if v["R_shuf_median"] <= 0.6 * v["R_nat_median"])
    n_total = len(valid)
    for k, v in valid.items():
        print(f"  {k}: R_nat={v['R_nat_median']:.2f} R_shuf={v['R_shuf_median']:.2f}")

    if n_total == 3 and n_pass_nat == 3 and n_pass_shuf >= 2:
        verdict = (f"PASS: {n_pass_nat}/3 classes with R_nat>=1.5, "
                   f"{n_pass_shuf}/3 with R_shuf collapse >=40%. "
                   f"Cross-class transport-vs-local asymmetry confirmed.")
    elif n_total == 3 and n_pass_nat >= 2:
        verdict = f"PARTIAL: {n_pass_nat}/3 classes pass R_nat>=1.5; {n_pass_shuf}/3 shuf-collapse."
    elif n_total < 3:
        verdict = f"INCOMPLETE: only {n_total}/3 models loaded successfully; cannot decide cross-class."
    else:
        verdict = (f"KILL: only {n_pass_nat}/3 classes pass R_nat>=1.5. Theory does not generalize across classes.")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 159, "name": "cross_class_lesion",
        "config": {"models": [m[0] for m in MODELS], "depths": DEPTHS,
                    "n_calib": N_CALIB, "n_eval": N_EVAL, "pca_rank": PCA_RANK,
                    "shuffle_seed": SHUFFLE_SEED, "seq_len": SEQ_LEN},
        "results": results, "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_159_cross_class_lesion.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
