"""
genome_161_rwkv_training_extension.py

Self-contained small RWKV-4 implementation for g161.

This file intentionally does not depend on the official RWKV repo. It
implements the minimal model family needed for the locked g161 contrast:

  baseline_rwkv:
      12 layers, hidden=512, channel-mix enabled

  transport_heavy:
      18 layers, hidden=512, channel-mix removed

The channel-mix width is set to 336 so that the baseline and
transport-heavy arms match core model forward FLOPs within +/- 2%.

The forward signature is compatible with the train/eval helpers used in
g156 / g158:

    out = model(input_ids=ids, attention_mask=mask, use_cache=False)
    logits = out.logits

For numerical stability, the recurrent WKV scan is always evaluated in
float32 even when the module weights and activations are bfloat16.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 50277
HIDDEN_SIZE = 512
BASELINE_LAYERS = 12
TRANSPORT_LAYERS = 18
CHANNEL_MIX_HIDDEN = 336
SEQ_LEN = 256

@dataclass
class RWKV4Output:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    state: list[tuple[torch.Tensor, ...]] | None = None

def _layer_ratio(layer_id: int, n_layers: int) -> tuple[float, float]:
    if n_layers <= 1:
        return 0.0, 1.0
    ratio_0_to_1 = layer_id / (n_layers - 1)
    ratio_1_to_0 = 1.0 - (layer_id / n_layers)
    return ratio_0_to_1, ratio_1_to_0

def _shift_with_state(x: torch.Tensor, prev_x: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, hidden = x.shape
    if prev_x is None:
        first = torch.zeros(batch, 1, hidden, dtype=x.dtype, device=x.device)
    else:
        first = prev_x.unsqueeze(1).to(dtype=x.dtype, device=x.device)
    shifted = torch.cat([first, x[:, :-1, :]], dim=1)
    return shifted, x[:, -1, :].detach()

def _mix_current_prev(x: torch.Tensor, x_prev: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    return x_prev + (x - x_prev) * mix

@torch.jit.script
def _wkv_scan_script(
    k: torch.Tensor,
    v: torch.Tensor,
    aa: torch.Tensor,
    bb: torch.Tensor,
    pp: torch.Tensor,
    decay: torch.Tensor,
    bonus: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outs = []
    steps = k.size(1)
    for t in range(steps):
        kt = k[:, t, :]
        vt = v[:, t, :]
        ww = bonus + kt
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        numer = e1 * aa + e2 * vt
        denom = e1 * bb + e2
        outs.append((numer / denom).unsqueeze(1))
        ww = pp + decay
        p = torch.maximum(ww, kt)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kt - p)
        aa = e1 * aa + e2 * vt
        bb = e1 * bb + e2
        pp = p
    return torch.cat(outs, dim=1), aa, bb, pp

class TimeMixV4(nn.Module):
    def __init__(self, hidden_size: int, layer_id: int, n_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.key = nn.Linear(hidden_size, hidden_size, bias=True)
        self.value = nn.Linear(hidden_size, hidden_size, bias=True)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output = nn.Linear(hidden_size, hidden_size, bias=True)
        self.time_mix_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_v = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_decay = nn.Parameter(torch.empty(hidden_size))
        self.time_first = nn.Parameter(torch.empty(hidden_size))
        self._reset_parameters(layer_id, n_layers)

    def _reset_parameters(self, layer_id: int, n_layers: int) -> None:
        ratio_0_to_1, ratio_1_to_0 = _layer_ratio(layer_id, n_layers)
        d = torch.arange(self.hidden_size, dtype=torch.float32) / max(self.hidden_size - 1, 1)
        d = d.view(1, 1, self.hidden_size)
        with torch.no_grad():
            self.time_mix_k.copy_(torch.pow(d, ratio_1_to_0))
            self.time_mix_v.copy_(torch.clamp(torch.pow(d, ratio_1_to_0) + 0.3 * ratio_0_to_1, 0.0, 1.0))
            self.time_mix_r.copy_(torch.pow(d, 0.5 * ratio_1_to_0))
            decay_speed = torch.empty(self.hidden_size, dtype=torch.float32)
            for i in range(self.hidden_size):
                frac = i / max(self.hidden_size - 1, 1)
                decay_speed[i] = -5.0 + 8.0 * (frac ** (0.7 + 1.3 * ratio_0_to_1))
            self.time_decay.copy_(decay_speed)
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(self.hidden_size)], dtype=torch.float32) * 0.5
            self.time_first.copy_(torch.full((self.hidden_size,), math.log(0.3), dtype=torch.float32) + zigzag)
        for linear in (self.key, self.value, self.receptance):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02 / math.sqrt(max(n_layers, 1)))
        nn.init.zeros_(self.output.bias)

    def _wkv_scan(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch, steps, hidden = k.shape
        kf = k.float()
        vf = v.float()
        if state is None:
            aa = torch.zeros(batch, hidden, dtype=torch.float32, device=k.device)
            bb = torch.zeros(batch, hidden, dtype=torch.float32, device=k.device)
            pp = torch.full((batch, hidden), -1.0e30, dtype=torch.float32, device=k.device)
        else:
            aa, bb, pp = state
            aa = aa.to(device=k.device, dtype=torch.float32)
            bb = bb.to(device=k.device, dtype=torch.float32)
            pp = pp.to(device=k.device, dtype=torch.float32)
        decay = (-torch.exp(self.time_decay.float())).view(1, hidden)
        bonus = self.time_first.float().view(1, hidden)
        out, aa, bb, pp = _wkv_scan_script(kf, vf, aa, bb, pp, decay, bonus)
        return out, (aa.detach(), bb.detach(), pp.detach())

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        prev_x = None if state is None else state[0]
        wkv_state = None if state is None else state[1:]
        x_prev, next_prev_x = _shift_with_state(x, prev_x)
        xk = _mix_current_prev(x, x_prev, self.time_mix_k)
        xv = _mix_current_prev(x, x_prev, self.time_mix_v)
        xr = _mix_current_prev(x, x_prev, self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        wkv, next_wkv_state = self._wkv_scan(k, v, wkv_state)
        rwkv = torch.sigmoid(r.float()) * wkv
        out = self.output(rwkv.to(dtype=x.dtype))
        return out, (next_prev_x, *next_wkv_state)

class ChannelMixV4(nn.Module):
    def __init__(self, hidden_size: int, channel_mix_hidden: int, layer_id: int, n_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.channel_mix_hidden = channel_mix_hidden
        self.key = nn.Linear(hidden_size, channel_mix_hidden, bias=True)
        self.gate = nn.Linear(hidden_size, channel_mix_hidden, bias=True)
        self.value = nn.Linear(channel_mix_hidden, hidden_size, bias=True)
        self.time_mix_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_g = nn.Parameter(torch.empty(1, 1, hidden_size))
        self._reset_parameters(layer_id, n_layers)

    def _reset_parameters(self, layer_id: int, n_layers: int) -> None:
        _, ratio_1_to_0 = _layer_ratio(layer_id, n_layers)
        d = torch.arange(self.hidden_size, dtype=torch.float32) / max(self.hidden_size - 1, 1)
        d = d.view(1, 1, self.hidden_size)
        with torch.no_grad():
            self.time_mix_k.copy_(torch.pow(d, ratio_1_to_0))
            self.time_mix_g.copy_(torch.pow(d, 0.5 * ratio_1_to_0))
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.zeros_(self.key.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.value.weight, mean=0.0, std=0.02 / math.sqrt(max(n_layers, 1)))
        nn.init.zeros_(self.value.bias)

    def forward(
        self,
        x: torch.Tensor,
        prev_x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_prev, next_prev_x = _shift_with_state(x, prev_x)
        xk = _mix_current_prev(x, x_prev, self.time_mix_k)
        xg = _mix_current_prev(x, x_prev, self.time_mix_g)
        k = self.key(xk)
        g = self.gate(xg)
        out = self.value(F.silu(g) * k)
        return out, next_prev_x

class RWKV4Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        channel_mix_hidden: int,
        layer_id: int,
        n_layers: int,
        no_channel_mix: bool = False,
    ):
        super().__init__()
        self.no_channel_mix = no_channel_mix
        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.time_mix = TimeMixV4(hidden_size, layer_id, n_layers)
        if not no_channel_mix:
            self.ln2 = nn.LayerNorm(hidden_size, eps=1e-5)
            self.channel_mix = ChannelMixV4(hidden_size, channel_mix_hidden, layer_id, n_layers)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[tuple[torch.Tensor, ...], torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, ...], torch.Tensor | None]]:
        time_state = None if state is None else state[0]
        channel_state = None if state is None else state[1]
        dx, next_time_state = self.time_mix(self.ln1(x), time_state)
        x = x + dx
        next_channel_state = channel_state
        if not self.no_channel_mix:
            dc, next_channel_state = self.channel_mix(self.ln2(x), channel_state)
            x = x + dc
        return x, (next_time_state, next_channel_state)

class RWKV4ForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        n_layers: int = BASELINE_LAYERS,
        channel_mix_hidden: int = CHANNEL_MIX_HIDDEN,
        no_channel_mix: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.channel_mix_hidden = channel_mix_hidden
        self.no_channel_mix = no_channel_mix
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([
            RWKV4Block(
                hidden_size=hidden_size,
                channel_mix_hidden=channel_mix_hidden,
                layer_id=i,
                n_layers=n_layers,
                no_channel_mix=no_channel_mix,
            )
            for i in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.emb.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.ln_out.weight)
        nn.init.zeros_(self.ln_out.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        state: list[tuple[torch.Tensor, ...]] | None = None,
    ) -> RWKV4Output:
        x = self.emb(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        next_state = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            block_state = None if state is None else state[i]
            x, block_next_state = block(x, block_state)
            if use_cache:
                next_state.append(block_next_state)
        x = self.ln_out(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().clone()
            if attention_mask is not None:
                shift_labels[attention_mask[:, 1:] == 0] = -100
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return RWKV4Output(logits=logits, loss=loss, state=next_state)

def time_mix_flops_per_token(hidden_size: int) -> int:
    return 8 * hidden_size * hidden_size + 35 * hidden_size

def channel_mix_flops_per_token(hidden_size: int, channel_mix_hidden: int) -> int:
    return 6 * hidden_size * channel_mix_hidden + 6 * hidden_size + 5 * channel_mix_hidden

def lm_head_flops_per_token(hidden_size: int, vocab_size: int) -> int:
    return 2 * hidden_size * vocab_size


def arm_forward_flops_per_token(
    hidden_size: int,
    n_layers: int,
    channel_mix_hidden: int,
    no_channel_mix: bool,
    vocab_size: int = VOCAB_SIZE,
    include_lm_head: bool = False,
) -> int:
    per_layer = time_mix_flops_per_token(hidden_size)
    if not no_channel_mix:
        per_layer += channel_mix_flops_per_token(hidden_size, channel_mix_hidden)
    total = n_layers * per_layer
    if include_lm_head:
        total += lm_head_flops_per_token(hidden_size, vocab_size)
    return total

def flop_match_ratio(
    baseline_layers: int = BASELINE_LAYERS,
    transport_layers: int = TRANSPORT_LAYERS,
    hidden_size: int = HIDDEN_SIZE,
    channel_mix_hidden: int = CHANNEL_MIX_HIDDEN,
) -> float:
    baseline = arm_forward_flops_per_token(
        hidden_size=hidden_size,
        n_layers=baseline_layers,
        channel_mix_hidden=channel_mix_hidden,
        no_channel_mix=False,
    )
    transport = arm_forward_flops_per_token(
        hidden_size=hidden_size,
        n_layers=transport_layers,
        channel_mix_hidden=channel_mix_hidden,
        no_channel_mix=True,
    )
    return abs(baseline - transport) / ((baseline + transport) / 2.0)

def build_rwkv_arm(
    arm_name: str,
    vocab_size: int = VOCAB_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    channel_mix_hidden: int = CHANNEL_MIX_HIDDEN,
    seed: int = 42,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> RWKV4ForCausalLM:
    if arm_name == "baseline_rwkv":
        n_layers = BASELINE_LAYERS
        no_channel_mix = False
    elif arm_name == "transport_heavy":
        n_layers = TRANSPORT_LAYERS
        no_channel_mix = True
    else:
        raise ValueError(f"Unknown arm_name={arm_name}")

    torch.manual_seed(seed)
    model = RWKV4ForCausalLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        channel_mix_hidden=channel_mix_hidden,
        no_channel_mix=no_channel_mix,
    )
    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model

def _arm_summary(arm_name: str) -> dict[str, float | int | bool]:
    model = build_rwkv_arm(arm_name, device="cpu", dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    if arm_name == "baseline_rwkv":
        n_layers = BASELINE_LAYERS
        no_channel_mix = False
    else:
        n_layers = TRANSPORT_LAYERS
        no_channel_mix = True
    core_flops = arm_forward_flops_per_token(
        hidden_size=HIDDEN_SIZE,
        n_layers=n_layers,
        channel_mix_hidden=CHANNEL_MIX_HIDDEN,
        no_channel_mix=no_channel_mix,
        include_lm_head=False,
    )
    total_flops = arm_forward_flops_per_token(
        hidden_size=HIDDEN_SIZE,
        n_layers=n_layers,
        channel_mix_hidden=CHANNEL_MIX_HIDDEN,
        no_channel_mix=no_channel_mix,
        include_lm_head=True,
    )
    return {
        "params": n_params,
        "layers": n_layers,
        "no_channel_mix": no_channel_mix,
        "core_forward_flops_per_token": core_flops,
        "forward_flops_per_token_with_lm_head": total_flops,
    }

def _print_summary() -> None:
    print("g161 small RWKV-4 summary")
    print(f"  vocab_size={VOCAB_SIZE} hidden_size={HIDDEN_SIZE} channel_mix_hidden={CHANNEL_MIX_HIDDEN}")
    print(f"  time_mix_flops_per_token={time_mix_flops_per_token(HIDDEN_SIZE):,}")
    print(f"  channel_mix_flops_per_token={channel_mix_flops_per_token(HIDDEN_SIZE, CHANNEL_MIX_HIDDEN):,}")
    print(f"  lm_head_flops_per_token={lm_head_flops_per_token(HIDDEN_SIZE, VOCAB_SIZE):,}")
    for arm_name in ("baseline_rwkv", "transport_heavy"):
        s = _arm_summary(arm_name)
        print(
            f"  {arm_name}: layers={s['layers']} params={s['params'] / 1e6:.2f}M "
            f"core_flops/token={s['core_forward_flops_per_token']:,} "
            f"with_lm_head={s['forward_flops_per_token_with_lm_head']:,}"
        )
    print(f"  core_flop_gap={100.0 * flop_match_ratio():.2f}%")

def _run_smoke(device: str) -> None:
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    for arm_name in ("baseline_rwkv", "transport_heavy"):
        model = build_rwkv_arm(arm_name, device=device, dtype=dtype)
        model.train()
        ids = torch.randint(0, VOCAB_SIZE, (2, 32), device=device)
        mask = torch.ones_like(ids)
        out = model(input_ids=ids, attention_mask=mask, labels=ids, use_cache=False)
        if out.loss is None or not torch.isfinite(out.loss):
            raise RuntimeError(f"{arm_name} smoke loss invalid: {out.loss}")
        out.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"  smoke {arm_name}: loss={out.loss.item():.4f} grad_norm={float(grad_norm):.4f}")

# ============================================================
# Training loop integration (added 2026-04-26 by Devansh, post-Codex model)
# ============================================================

import json
import sys
import time
from pathlib import Path
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

ROOT_DIR = _THIS_DIR.parent

# Locked prereg constants
SEEDS_FOR_TRAIN = [42, 7, 13]
N_C4_EVAL = 256
N_OOD_EVAL = 256
N_TRAIN = 32768
TRAIN_STEPS = 4000
LR_WARMUP_STEPS = 200
SHUFFLE_SEED = 42
BATCH_SIZE = 8
LR_GRID = [2e-4, 3e-4, 4e-4]


def _shuffle_token_rows(ids, mask, shuffle_seed=SHUFFLE_SEED):
    rng = np.random.default_rng(shuffle_seed)
    out = ids.clone()
    for r in range(ids.shape[0]):
        valid_pos = (mask[r] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        if len(valid_pos) <= 1:
            continue
        perm = rng.permutation(len(valid_pos))
        out[r, valid_pos] = ids[r, valid_pos[perm]]
    return out


def _warmup_lr(step, target_lr, warmup_steps):
    if step < warmup_steps:
        return target_lr * (step + 1) / warmup_steps
    return target_lr


def _measure(model, eval_ids, eval_mask, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    with torch.no_grad():
        for i in range(0, eval_ids.size(0), BATCH_SIZE):
            ids = eval_ids[i:i+BATCH_SIZE].to(device)
            mask = eval_mask[i:i+BATCH_SIZE].to(device)
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = out.logits.float() if hasattr(out, "logits") else out["logits"].float()
            sl = logits[:, :-1].contiguous()
            lbl = ids[:, 1:].clone()
            sm = mask[:, 1:]
            valid = (sm != 0)
            lbl[~valid] = -100
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1),
                                    ignore_index=-100, reduction="sum")
            total_loss += loss.item()
            total_tokens += valid.sum().item()
            preds = sl.argmax(dim=-1)
            correct_top1 += ((preds == lbl) & valid).sum().item()
    model.train()
    return {"nll": total_loss / max(total_tokens, 1),
            "top1_acc": correct_top1 / max(total_tokens, 1)}


def _train_arm(arm_name, lr, model, train_ids, train_mask, n_steps, seed, device="cuda"):
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name} seed={seed} lr={lr}: params={n_total/1e6:.2f}M steps={n_steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(seed)
    t_arm = time.time()
    model.train()
    n_train = train_ids.size(0)
    nan_seen = False
    for step in range(1, n_steps + 1):
        cur_lr = _warmup_lr(step, lr, LR_WARMUP_STEPS)
        for g in opt.param_groups:
            g['lr'] = cur_lr
        idx = rng.integers(0, n_train, size=BATCH_SIZE)
        ids = train_ids[idx].to(device)
        mask = train_mask[idx].to(device)
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits.float() if hasattr(out, "logits") else out["logits"].float()
        sl = logits[:, :-1].contiguous()
        lbl = ids[:, 1:].clone()
        sm = mask[:, 1:]
        lbl[sm == 0] = -100
        loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            nan_seen = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"    step={step:5d} loss={loss.item():.3f} ({time.time()-t_arm:.0f}s)")
    return n_total, time.time() - t_arm, nan_seen


def run_full_experiment():
    """Full g161 training experiment per locked prereg."""
    t0 = time.time()
    print("genome_161: RWKV training-time transport extension")
    print(f"  arms: baseline_rwkv ({BASELINE_LAYERS}L+chmix) vs transport_heavy ({TRANSPORT_LAYERS}L noch)")
    print(f"  conditions: natural + token_shuffled (shuffle_seed={SHUFFLE_SEED})")
    print(f"  seeds: {SEEDS_FOR_TRAIN}")

    # Pre-flight FLOP-match check
    flop_diff = flop_match_ratio() * 100
    print(f"  forward FLOP match: {flop_diff:.2f}% (target <=2.0%)")
    if flop_diff > 2.0:
        raise RuntimeError(f"FLOP match violated: {flop_diff:.2f}%")

    from transformers import AutoTokenizer
    from stimulus_banks import c4_clean_v1
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # Use c4_clean_v1 seed=161 to avoid g141..g160 train slices
    print(f"\nLoading {N_TRAIN}+{N_C4_EVAL} c4 sequences (seed=161)...")
    pool = []
    for rec in c4_clean_v1(seed=161, n_samples=N_TRAIN + N_C4_EVAL):
        pool.append(rec["text"])
        if len(pool) >= N_TRAIN + N_C4_EVAL:
            break
    enc = tok(pool, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    train_ids_nat = enc["input_ids"][:N_TRAIN]
    train_mask = enc["attention_mask"][:N_TRAIN]
    eval_ids_nat = enc["input_ids"][N_TRAIN:]
    eval_mask = enc["attention_mask"][N_TRAIN:]

    # Build shuffled corpus
    train_ids_shuf = _shuffle_token_rows(train_ids_nat, train_mask, SHUFFLE_SEED)
    eval_ids_shuf = _shuffle_token_rows(eval_ids_nat, eval_mask, SHUFFLE_SEED + 1)

    # OOD eval from wikitext-103 VAL split
    from datasets import load_dataset
    ds_ood = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    ood_texts = []
    rng_ood = np.random.default_rng(12345)
    for idx in rng_ood.permutation(len(ds_ood)):
        t = ds_ood[int(idx)]["text"].strip()
        if len(t) > 200:
            ood_texts.append(t[:1500])
        if len(ood_texts) >= N_OOD_EVAL:
            break
    enc_ood = tok(ood_texts, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    ood_ids_nat = enc_ood["input_ids"]
    ood_mask = enc_ood["attention_mask"]
    ood_ids_shuf = _shuffle_token_rows(ood_ids_nat, ood_mask, SHUFFLE_SEED + 2)

    arms = ["baseline_rwkv", "transport_heavy"]
    conditions = [
        ("natural", train_ids_nat, train_mask, eval_ids_nat, eval_mask, ood_ids_nat, ood_mask),
        ("token_shuffled", train_ids_shuf, train_mask, eval_ids_shuf, eval_mask, ood_ids_shuf, ood_mask),
    ]

    # NOTE: locked prereg expects per-arm best LR chosen on a separate val bank.
    # For PILOT-fast, use lr=3e-4 for both (matches g141..g158 default).
    # If pilot passes, write a g161b prereg with proper LR selection.
    arm_lr = {"baseline_rwkv": 3e-4, "transport_heavy": 3e-4}

    results = {}
    for cond_name, t_ids, t_mask, e_ids, e_mask, o_ids, o_mask in conditions:
        results[cond_name] = {}
        for arm_name in arms:
            results[cond_name][arm_name] = {}
            lr = arm_lr[arm_name]
            for seed in SEEDS_FOR_TRAIN:
                print(f"\n=== cond={cond_name} arm={arm_name} seed={seed} ===")
                model = build_rwkv_arm(arm_name, vocab_size=tok.vocab_size if hasattr(tok, 'vocab_size') and tok.vocab_size else VOCAB_SIZE,
                                          seed=seed, device="cuda", dtype=torch.bfloat16)
                n_total, elapsed, nan_seen = _train_arm(arm_name, lr, model, t_ids, t_mask, TRAIN_STEPS, seed)
                metrics = {"nan_seen": nan_seen, "wallclock_s": elapsed, "params_M": n_total / 1e6}
                if not nan_seen:
                    metrics["c4"] = _measure(model, e_ids, e_mask)
                    metrics["ood"] = _measure(model, o_ids, o_mask)
                    print(f"    c4 top1={100*metrics['c4']['top1_acc']:.2f}%  "
                          f"ood top1={100*metrics['ood']['top1_acc']:.2f}%")
                else:
                    metrics["c4"] = {"top1_acc": float("nan"), "nll": float("nan")}
                    metrics["ood"] = {"top1_acc": float("nan"), "nll": float("nan")}
                results[cond_name][arm_name][seed] = metrics
                del model
                torch.cuda.empty_cache()

    # Analysis
    print(f"\n=== ANALYSIS ===")
    deltas = {}
    for cond_name in ["natural", "token_shuffled"]:
        b_c4 = [results[cond_name]["baseline_rwkv"][s]["c4"]["top1_acc"] for s in SEEDS_FOR_TRAIN
                if not results[cond_name]["baseline_rwkv"][s]["nan_seen"]]
        t_c4 = [results[cond_name]["transport_heavy"][s]["c4"]["top1_acc"] for s in SEEDS_FOR_TRAIN
                if not results[cond_name]["transport_heavy"][s]["nan_seen"]]
        b_ood = [results[cond_name]["baseline_rwkv"][s]["ood"]["top1_acc"] for s in SEEDS_FOR_TRAIN
                 if not results[cond_name]["baseline_rwkv"][s]["nan_seen"]]
        t_ood = [results[cond_name]["transport_heavy"][s]["ood"]["top1_acc"] for s in SEEDS_FOR_TRAIN
                 if not results[cond_name]["transport_heavy"][s]["nan_seen"]]
        d_c4 = (np.mean(t_c4) - np.mean(b_c4)) * 100 if b_c4 and t_c4 else float("nan")
        d_ood = (np.mean(t_ood) - np.mean(b_ood)) * 100 if b_ood and t_ood else float("nan")
        deltas[cond_name] = {"c4": d_c4, "ood": d_ood}
        print(f"  {cond_name}: delta_c4={d_c4:+.2f}pp, delta_ood={d_ood:+.2f}pp")

    nat_c4 = deltas["natural"]["c4"]
    shuf_c4 = deltas["token_shuffled"]["c4"]
    nat_ood = deltas["natural"]["ood"]
    shuf_ood = deltas["token_shuffled"]["ood"]
    contrast_c4 = nat_c4 - shuf_c4
    contrast_ood = nat_ood - shuf_ood

    if (nat_c4 >= 0.3 and nat_ood >= 0.3 and shuf_c4 <= 0.1 and shuf_ood <= 0.1
        and contrast_c4 >= 0.3 and contrast_ood >= 0.3):
        verdict = (f"PASS_RWKV: Δ_nat_c4={nat_c4:+.2f}pp / ood={nat_ood:+.2f}pp; "
                   f"Δ_shuf_c4={shuf_c4:+.2f}pp / ood={shuf_ood:+.2f}pp; "
                   f"contrast_c4={contrast_c4:+.2f}pp / ood={contrast_ood:+.2f}pp. "
                   f"RWKV transport extension confirmed.")
    elif nat_c4 >= 0.2 and contrast_c4 >= 0.2:
        verdict = (f"PARTIAL_RWKV: contrast_c4={contrast_c4:+.2f}pp.")
    else:
        verdict = (f"KILL_RWKV: contrast_c4={contrast_c4:+.2f}pp / ood={contrast_ood:+.2f}pp; "
                   f"theory does not extend to RWKV training-time.")
    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 161, "name": "rwkv_training_extension",
        "config": {"baseline_layers": BASELINE_LAYERS, "transport_layers": TRANSPORT_LAYERS,
                    "hidden_size": HIDDEN_SIZE, "channel_mix_hidden": CHANNEL_MIX_HIDDEN,
                    "seeds": SEEDS_FOR_TRAIN, "shuffle_seed": SHUFFLE_SEED,
                    "n_train": N_TRAIN, "train_steps": TRAIN_STEPS,
                    "arm_lr": arm_lr, "flop_diff_pct": float(flop_diff)},
        "results": results, "deltas": deltas,
        "contrast_c4": float(contrast_c4), "contrast_ood": float(contrast_ood),
        "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = ROOT_DIR / "results" / "genome_161_rwkv_training_extension.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="g161 small RWKV-4 implementation (model module)")
    parser.add_argument("--summary", action="store_true", help="Print param and FLOP summaries")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny forward/backward smoke test")
    parser.add_argument("--run", action="store_true",
                        help="DEPRECATED: use python code/genome_161_run.py")
    args = parser.parse_args()
    if not (args.summary or args.smoke or args.run):
        args.summary = True
    if args.summary:
        _print_summary()
    if args.smoke:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"running smoke on device={device}")
        _run_smoke(device)
    if args.run:
        # Per cycle 6 code review Sev-8: this stale path used N_C4_EVAL=256, fixed arm_lr,
        # no microbenchmark/completeness guard. canonical runner is genome_161_run.py.
        raise RuntimeError(
            "Deprecated: --run on the model module is removed; "
            "use `python code/genome_161_run.py` for the locked-prereg training experiment."
        )


if __name__ == "__main__":
    main()
