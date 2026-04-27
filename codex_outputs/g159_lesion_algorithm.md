**g159 lesion spec**

This is for the local environment actually installed here: `transformers 4.56.2`, `torch 2.9.1+cu128`. Sentinel-depth mapping in this repo is `int(round(depth_frac * (L - 1)))` per [genome_extractor.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_extractor.py:76>). The relevant forward equations are in [modeling_qwen3.py](</C:/Users/devan/AppData/Local/Programs/Python/Python313/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py:233>), [modeling_rwkv.py](</C:/Users/devan/AppData/Local/Programs/Python/Python313/Lib/site-packages/transformers/models/rwkv/modeling_rwkv.py:348>), and [modeling_falcon_h1.py](</C:/Users/devan/AppData/Local/Programs/Python/Python313/Lib/site-packages/transformers/models/falcon_h1/modeling_falcon_h1.py:1048>).

**A. Module Paths**

`Qwen/Qwen3-0.6B`  
`L=28`, sentinel layers `{0.25,0.50,0.75} -> {7,14,20}`.
- Transport sublayer: `model.model.layers[i].self_attn`
- Local sublayer: `model.model.layers[i].mlp`
- Exact transport delta: post-`o_proj` attention output, i.e. the first element returned by `self_attn(...)`
- Exact local delta: `mlp(...)` output
- `get_transport_sublayers(0.5)` returns `[model.model.layers[14].self_attn]`

`RWKV/rwkv-4-169m-pile`  
`L=12`, sentinel layers `{0.25,0.50,0.75} -> {3,6,8}`.
- Transport sublayer: `model.rwkv.blocks[i].attention`
- Local sublayer: `model.rwkv.blocks[i].feed_forward`
- Exact transport delta: first element of `attention(...)`, i.e. post-`output` linear on `sigmoid(receptance) * rwkv_linear_attention(...)`
- Exact local delta: first element of `feed_forward(...)`, i.e. `sigmoid(receptance) * value(square(relu(key(...))))`
- `get_transport_sublayers(0.5)` returns `[model.rwkv.blocks[6].attention]`

`tiiuae/Falcon-H1-0.5B-Instruct`  
`L=36`, sentinel layers `{0.25,0.50,0.75} -> {9,18,26}`.
- Every layer is hybrid. There are not separate “attention layers” and “Mamba layers” in this installed implementation.
- Each `model.model.layers[i]` contains both `mamba` and `self_attn`, and `FalconH1DecoderLayer.forward` always runs both.
- Transport site is the composite inside `model.model.layers[i]`:
  `T_i = ssm_out_multiplier * model.model.layers[i].mamba(...) + attention_out_multiplier * model.model.layers[i].self_attn(...)[0]`
- Local sublayer: `model.model.layers[i].feed_forward`
- `get_transport_sublayers(0.5)` should return a wrapper site anchored on `model.model.layers[18]`; the child paths involved are `model.model.layers[18].mamba` and `model.model.layers[18].self_attn`

**B. Exact Lesion Algorithm**

For each `(model, depth)`:
1. Map depth to layer index with `int(round(d * (L - 1)))`.
2. Run `model.eval()` and `torch.no_grad()`.
3. Use independent fixed windows; do not carry cache/state across windows. Set `use_cache=False`. For RWKV pass `state=None`.
4. Calibration set: 2048 natural `c4` windows only.
5. Capture the chosen residual contribution tensor before residual add. Shape is `[B,T,H]`.
6. Flatten only valid non-pad token positions to `X in R^{N_tokens x H}`. If tokenizer inserts BOS/EOS, keep them as valid tokens; only mask pads.
7. Fit exact mean-centered PCA in output space, separately for transport and local, using a streaming covariance:
   `mu = mean(X)`, `C = (X^T X - N * mu mu^T) / (N - 1)`, top-32 eigenvectors of `C`.
8. Eval set: 1024 natural windows and 1024 shuffled-control windows.
9. Shuffled control: tokenize first, then permute non-pad token IDs independently within each row; keep pad positions fixed, and keep BOS/EOS fixed if present.
10. At eval time, lesion the same residual contribution:
   `y_lesioned = y - ((y - mu) @ U^T) @ U`
   where `U` is the top-32 orthonormal basis with shape `[32,H]`.
11. Compute `ΔNLL_transport` and `ΔNLL_local` against no-lesion baseline on the same eval condition.
12. Compute `R = ΔNLL_transport / max(ΔNLL_local, 1e-6)`.
13. Aggregate by median over the three sentinel depths.

Use float32 for projection math even if the model runs BF16/FP16, then cast back to the original dtype.

**C. Edge-Case Decisions**

- Qwen attention lesion target is the post-`o_proj` output, not pre-`o_proj`. The lesion must act on what is actually added to the residual stream.
- RWKV is hookable in the same sense as a transformer if you lesion the first return value of `attention` or `feed_forward` and leave the returned `state` untouched.
- RWKV caveat: `feed_forward` is not purely token-local. It uses `time_shift` / previous-token state before the nonlinear channel mix. So the prereg’s “local” label is an approximation. This makes RWKV’s transport-vs-local contrast conservative, not cleaner.
- Falcon-H1 exact transport lesion is not representable as a single child-module forward hook, because the residual receives a sum of two transport branches. The exact workaround is to patch/wrap `FalconH1DecoderLayer.forward` at the parent layer and lesion the summed transport delta before `residual + transport`.
- In Falcon-H1, do not treat `mamba` as “the transport sublayer” and ignore `self_attn`. The layer has both, every time.
- In Falcon-H1, the local lesion target is `feed_forward`, not any internal gated path inside `mamba`.

**D. Symmetric Rank Budget**

Fixed rank-32 is acceptable as the prereg primary spec, but it is not perfectly cross-architecture fair.
- Qwen/Falcon: `32 / 1024 = 3.125%` of residual width
- RWKV: `32 / 768 = 4.167%` of residual width

Within each model, transport and local lesions are matched exactly. Across models, the stronger fairness issue is RWKV local-not-quite-local, not the rank fraction. Keep fixed 32 for prereg, but log `variance_explained_top32` for every site so the cross-class comparison is audit-visible.

**E. Code Skeleton**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MethodType
from typing import Dict, List
import torch
import torch.nn as nn

@dataclass
class PCABasis:
    mean: torch.Tensor      # [H]
    basis: torch.Tensor     # [K, H], orthonormal rows
    path: str

class ForwardPatchHandle:
    def __init__(self, module: nn.Module, orig_forward):
        self.module = module
        self.orig_forward = orig_forward
    def remove(self):
        self.module.forward = self.orig_forward

def _project_out(y, mean, basis):
    od = y.dtype
    y32 = y.to(torch.float32)
    mu = mean.to(y32.device, y32.dtype)
    U = basis.to(y32.device, y32.dtype)
    yc = y32 - mu
    y32 = y32 - (yc @ U.T) @ U
    return y32.to(od)

def _replace_first(output, new0):
    if isinstance(output, tuple):
        return (new0,) + output[1:]
    return new0

class LesionAdapter(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def n_layers(self) -> int: ...

    def depth_to_idx(self, depth_frac: float) -> int:
        return int(round(depth_frac * (self.n_layers() - 1)))

    @abstractmethod
    def get_transport_sublayers(self, depth_frac: float) -> List[nn.Module]: ...

    @abstractmethod
    def get_local_sublayers(self, depth_frac: float) -> List[nn.Module]: ...

    @abstractmethod
    def fit_pca(self, sublayers, calib_data) -> Dict[str, PCABasis]: ...

    def install_lesion_hook(self, sublayer, pca_basis: PCABasis, rank=32):
        U = pca_basis.basis[:rank]
        mu = pca_basis.mean
        def hook(_m, _inp, output):
            y0 = output[0] if isinstance(output, tuple) else output
            y0 = _project_out(y0, mu, U)
            return _replace_first(output, y0)
        return sublayer.register_forward_hook(hook)

class Qwen3Adapter(LesionAdapter):
    def n_layers(self): return len(self.model.model.layers)
    def get_transport_sublayers(self, depth_frac): return [self.model.model.layers[self.depth_to_idx(depth_frac)].self_attn]
    def get_local_sublayers(self, depth_frac): return [self.model.model.layers[self.depth_to_idx(depth_frac)].mlp]

class RWKVAdapter(LesionAdapter):
    def n_layers(self): return len(self.model.rwkv.blocks)
    def get_transport_sublayers(self, depth_frac): return [self.model.rwkv.blocks[self.depth_to_idx(depth_frac)].attention]
    def get_local_sublayers(self, depth_frac): return [self.model.rwkv.blocks[self.depth_to_idx(depth_frac)].feed_forward]

class FalconTransportSite(nn.Module):
    def __init__(self, layer: nn.Module, layer_idx: int):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.path = f"model.layers.{layer_idx}::<transport_composite>"

class FalconH1Adapter(LesionAdapter):
    def n_layers(self): return len(self.model.model.layers)

    def get_transport_sublayers(self, depth_frac):
        i = self.depth_to_idx(depth_frac)
        return [FalconTransportSite(self.model.model.layers[i], i)]

    def get_local_sublayers(self, depth_frac):
        i = self.depth_to_idx(depth_frac)
        return [self.model.model.layers[i].feed_forward]

    def install_lesion_hook(self, sublayer, pca_basis: PCABasis, rank=32):
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
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=mamba_attention_mask,
            ) * this.ssm_out_multiplier

            a_out, a_weights = this.self_attn(
                hidden_states=hidden_states * this.attention_in_multiplier,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
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
```

**F. Compute**

The user’s eval-only arithmetic is right:
- `3 models x 3 depths x 2 lesions x 2 eval conditions x 1024 windows = 36,864` lesion-eval windows

The full run is larger because you also need calibration and baselines:
- Calibration: `3 x 3 x 2048 = 18,432` windows
- No-lesion baselines: `3 x 2 x 1024 = 6,144` windows
- Total: `61,440` window-forwards

At `256` tokens/window that is about `15.7M` token positions through the models. With batch sizes `{64, 64, 32}` for `{Qwen, RWKV, Falcon}`, that is about `1280` batched forwards. Falcon will dominate wall clock because this machine is using the naive Mamba path, so a realistic envelope is closer to `0.8-1.5 GPU-hours` than the optimistic “few minutes” estimate.