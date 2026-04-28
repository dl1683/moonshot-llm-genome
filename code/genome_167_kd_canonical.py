"""
genome_167_kd_canonical.py

Canonical 3-seed distillation scale-up of genome 154.

Locked question
---------------
Does top-k logit KD transfer teacher next-token capability to a random-init
recipient at canonical multi-seed scale?

Teacher:
  - Qwen/Qwen3-0.6B

Student:
  - Same minimal_3L no-MLP Llama-family student used in genome 154
  - Qwen3 tokenizer / vocab so KD is defined on one shared token space

Arms:
  - scratch_ce
  - kd_topk64_t2_g0p5

Data:
  - 8192 train windows from deduped C4 train
  - 1000 eval windows from deduped C4 validation
  - 1000 eval windows from Wikitext-103 validation
  - fixed window length 256

Training:
  - seeds = [42, 7, 13]
  - 6000 steps
  - batch size 8
  - BF16 forward, FP32 recipient weights

PASS:
  - mean(KD - scratch) C4-val top1 >= +0.40 pp
  - paired 95% CI on that mean excludes zero
  - mean NLL gain (scratch - KD) >= +0.03 on both C4-val and Wikitext-val
  - positive primary effect in at least 2/3 seeds

FAIL:
  - any threshold above fails

Outputs:
  - results/genome_167_kd_canonical.json
  - results/cache/genome_167_teacher_topk_cache_<train_hash>.pt
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "models"))
try:
    from registry import resolve as _resolve_model  # type: ignore

    TEACHER_HF = _resolve_model("qwen3-0.6b").get("hf_id", "Qwen/Qwen3-0.6B")
except Exception:
    TEACHER_HF = "Qwen/Qwen3-0.6B"

OUT_PATH = ROOT / "results" / "genome_167_kd_canonical.json"
CACHE_DIR = ROOT / "results" / "cache"

SEEDS = [42, 7, 13]
SEQ_LEN = 256
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
TRAIN_STEPS = 6000
LR = 3e-4
LR_WARMUP_STEPS = 200
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
LOG_EVERY = 1000

N_TRAIN_WINDOWS = 8192
N_C4_VAL_WINDOWS = 1000
N_WIKITEXT_VAL_WINDOWS = 1000

KD_TOPK = 64
KD_TEMP = 2.0
KD_GAMMA = 0.5
N_BOOT = 10000

C4_TRAIN_SEED = 167001
C4_VAL_SEED = 167101
WIKITEXT_VAL_SEED = 167201

PASS_TOP1_GAIN_PP = 0.40
PASS_NLL_GAIN = 0.03
PASS_MIN_POSITIVE_SEEDS = 2

HASH_LEN = 13
HASH_MULT = np.uint64(1099511628211)
HASH_OFFSET = np.uint64(1469598103934665603)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORWARD_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class ArmSpec:
    label: str
    use_kd: bool
    description: str


ARM_SPECS = [
    ArmSpec(
        label="scratch_ce",
        use_kd=False,
        description="Scratch CE baseline from identical random initialization.",
    ),
    ArmSpec(
        label="kd_topk64_t2_g0p5",
        use_kd=True,
        description="CE plus top-k KD with k=64, T=2.0, gamma=0.5.",
    ),
]


class ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def warmup_lr(step: int, target_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return target_lr
    if step < warmup_steps:
        return target_lr * float(step + 1) / float(warmup_steps)
    return target_lr


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        TEACHER_HF,
        trust_remote_code=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_trained_teacher(tok=None):
    if tok is None:
        tok = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_HF,
        torch_dtype=FORWARD_DTYPE,
        trust_remote_code=False,
    ).to(DEVICE).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tok


def make_minimal_student(vocab_size: int, seed: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        num_hidden_layers=3,
        num_attention_heads=6,
        num_key_value_heads=6,
        intermediate_size=1024,
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"

    set_seed(seed)
    model = LlamaForCausalLM(cfg)
    for layer in model.model.layers:
        layer.mlp = ZeroMLP()
    model = model.to(DEVICE)
    model.train()
    return model


def _load_streaming_dataset(dataset_candidates, config_name: str, split: str, seed: int):
    last_error = None
    for dataset_name in dataset_candidates:
        try:
            ds = load_dataset(
                dataset_name,
                config_name,
                split=split,
                streaming=True,
                trust_remote_code=False,
            )
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
            return ds, dataset_name
        except Exception as exc:
            last_error = exc
    joined = ", ".join(dataset_candidates)
    raise RuntimeError(
        f"failed to load dataset candidates [{joined}] split={split}: {last_error}"
    )


def rolling_13gram_hashes(tokens: np.ndarray) -> np.ndarray:
    n_hashes = int(tokens.shape[0]) - HASH_LEN + 1
    if n_hashes <= 0:
        return np.empty(0, dtype=np.uint64)
    tok_u64 = tokens.astype(np.uint64, copy=False)
    acc = np.zeros(n_hashes, dtype=np.uint64)
    for offset in range(HASH_LEN):
        acc = acc * HASH_MULT + (
            tok_u64[offset : offset + n_hashes] + np.uint64(offset + 1) * HASH_OFFSET
        )
    return acc


def collect_13gram_hashes(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> set[int]:
    ids_np = input_ids.cpu().numpy()
    mask_np = attention_mask.cpu().numpy()
    hashes: set[int] = set()
    for row, mask in zip(ids_np, mask_np):
        valid = int(mask.sum())
        if valid < HASH_LEN:
            continue
        row_hashes = rolling_13gram_hashes(np.asarray(row[:valid], dtype=np.int64))
        hashes.update(int(h) for h in row_hashes.tolist())
    return hashes


def sample_document_windows(
    tok,
    *,
    dataset_candidates,
    config_name: str,
    split: str,
    seed: int,
    n_windows: int,
    seq_len: int,
    forbidden_hashes: set[int] | None = None,
):
    print(f"  loading document windows split={split} seed={seed} n={n_windows} len={seq_len}")
    ds, chosen_name = _load_streaming_dataset(dataset_candidates, config_name, split, seed)
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    seen_prefixes: set[str] = set()
    overlap_rejects = 0

    for record in ds:
        text = record.get("text", "")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < 100:
            continue
        prefix = text[:200]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        token_ids = tok(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        if len(token_ids) < seq_len:
            continue

        if len(token_ids) == seq_len:
            start = 0
        else:
            start = int(rng.integers(0, len(token_ids) - seq_len + 1))
        window = np.asarray(token_ids[start : start + seq_len], dtype=np.int64)
        if window.shape[0] != seq_len:
            continue

        if forbidden_hashes is not None:
            row_hashes = rolling_13gram_hashes(window)
            if any(int(h) in forbidden_hashes for h in row_hashes.tolist()):
                overlap_rejects += 1
                continue

        windows.append(window)
        if len(windows) >= n_windows:
            break

    if len(windows) < n_windows:
        raise RuntimeError(
            f"only sampled {len(windows)} / {n_windows} document windows from "
            f"{chosen_name}:{config_name}:{split} seed={seed}"
        )

    input_ids = torch.tensor(np.stack(windows, axis=0), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    meta = {
        "dataset_name": chosen_name,
        "config_name": config_name,
        "split": split,
        "seed": seed,
        "n_windows": int(input_ids.shape[0]),
        "seq_len": seq_len,
        "sampling": "document_random_window",
        "prefix_dedup": True,
        "forbidden_overlap_rejects": overlap_rejects,
    }
    print(f"  got {input_ids.shape[0]} windows from {chosen_name}")
    return input_ids, attention_mask, meta


def sample_stream_windows(
    tok,
    *,
    dataset_candidates,
    config_name: str,
    split: str,
    seed: int,
    n_windows: int,
    seq_len: int,
    forbidden_hashes: set[int] | None = None,
):
    print(f"  loading stream windows split={split} seed={seed} n={n_windows} len={seq_len}")
    ds, chosen_name = _load_streaming_dataset(dataset_candidates, config_name, split, seed)
    windows: list[np.ndarray] = []
    token_buffer: list[int] = []
    buffer_cursor = 0
    overlap_rejects = 0
    records_seen = 0

    for record in ds:
        text = record.get("text", "")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        records_seen += 1

        token_ids = tok(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        if not token_ids:
            continue

        token_buffer.extend(token_ids)
        while (len(token_buffer) - buffer_cursor) >= seq_len and len(windows) < n_windows:
            window = np.asarray(
                token_buffer[buffer_cursor : buffer_cursor + seq_len],
                dtype=np.int64,
            )
            buffer_cursor += seq_len

            if forbidden_hashes is not None:
                row_hashes = rolling_13gram_hashes(window)
                if any(int(h) in forbidden_hashes for h in row_hashes.tolist()):
                    overlap_rejects += 1
                    continue

            windows.append(window)

        if buffer_cursor >= 8192:
            token_buffer = token_buffer[buffer_cursor:]
            buffer_cursor = 0

        if len(windows) >= n_windows:
            break

    if len(windows) < n_windows:
        raise RuntimeError(
            f"only sampled {len(windows)} / {n_windows} stream windows from "
            f"{chosen_name}:{config_name}:{split} seed={seed}; records_seen={records_seen}"
        )

    input_ids = torch.tensor(np.stack(windows, axis=0), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    meta = {
        "dataset_name": chosen_name,
        "config_name": config_name,
        "split": split,
        "seed": seed,
        "n_windows": int(input_ids.shape[0]),
        "seq_len": seq_len,
        "sampling": "concatenated_nonoverlapping_windows",
        "prefix_dedup": False,
        "forbidden_overlap_rejects": overlap_rejects,
        "records_seen": records_seen,
    }
    print(f"  got {input_ids.shape[0]} windows from {chosen_name}")
    return input_ids, attention_mask, meta


def load_c4_windows(tok, *, split: str, seed: int, n_windows: int, forbidden_hashes=None):
    return sample_document_windows(
        tok,
        dataset_candidates=["allenai/c4"],
        config_name="en",
        split=split,
        seed=seed,
        n_windows=n_windows,
        seq_len=SEQ_LEN,
        forbidden_hashes=forbidden_hashes,
    )


def load_wikitext_windows(tok, *, split: str, seed: int, n_windows: int, forbidden_hashes=None):
    return sample_stream_windows(
        tok,
        dataset_candidates=["wikitext", "Salesforce/wikitext"],
        config_name="wikitext-103-raw-v1",
        split=split,
        seed=seed,
        n_windows=n_windows,
        seq_len=SEQ_LEN,
        forbidden_hashes=forbidden_hashes,
    )


def causal_ce_loss(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous().clone()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels[shift_mask == 0] = -100
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate_model(model, eval_ids: torch.Tensor, eval_mask: torch.Tensor) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_top1 = 0
    for start in range(0, eval_ids.shape[0], EVAL_BATCH_SIZE):
        ids = eval_ids[start : start + EVAL_BATCH_SIZE].to(DEVICE)
        mask = eval_mask[start : start + EVAL_BATCH_SIZE].to(DEVICE)
        with autocast_context():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        shift_logits = logits[:, :-1].contiguous().float()
        shift_labels = ids[:, 1:].contiguous().clone()
        shift_mask = mask[:, 1:].contiguous().bool()
        shift_labels_for_loss = shift_labels.clone()
        shift_labels_for_loss[~shift_mask] = -100
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels_for_loss.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tokens = int(shift_mask.sum().item())
        total_loss += float(loss_sum.item())
        total_tokens += n_tokens
        preds = shift_logits.argmax(dim=-1)
        total_top1 += int(((preds == shift_labels) & shift_mask).sum().item())
    model.train()
    return {
        "nll": total_loss / max(total_tokens, 1),
        "top1_acc": total_top1 / max(total_tokens, 1),
        "n_tokens": float(total_tokens),
    }


def tensor_sha1(tensor: torch.Tensor) -> str:
    array = tensor.cpu().numpy()
    digest = hashlib.sha1(array.tobytes()).hexdigest()
    return digest


def estimate_teacher_cache_bytes(n_windows: int, seq_len: int, top_k: int) -> int:
    positions = n_windows * (seq_len - 1) * top_k
    idx_bytes = positions * 4
    logit_bytes = positions * 2
    return idx_bytes + logit_bytes


def precompute_teacher_topk_cache(
    teacher,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    train_signature: str,
):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"genome_167_teacher_topk_cache_{train_signature[:16]}.pt"
    expected_meta = {
        "teacher_hf": TEACHER_HF,
        "train_signature": train_signature,
        "n_train_windows": int(train_ids.shape[0]),
        "seq_len": int(train_ids.shape[1]),
        "top_k": KD_TOPK,
        "dtype_logits": "bfloat16",
        "dtype_indices": "int32",
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("meta") == expected_meta:
            print(f"  teacher cache hit: {cache_path}")
            return {
                "topk_idx": payload["topk_idx"],
                "topk_logits": payload["topk_logits"],
                "path": str(cache_path),
                "cache_hit": True,
                "estimated_bytes": estimate_teacher_cache_bytes(
                    int(train_ids.shape[0]),
                    int(train_ids.shape[1]),
                    KD_TOPK,
                ),
            }
        print(f"  teacher cache mismatch, regenerating: {cache_path}")

    print(
        f"  precomputing teacher top-{KD_TOPK} cache over {train_ids.shape[0]} windows "
        f"(approx {estimate_teacher_cache_bytes(int(train_ids.shape[0]), int(train_ids.shape[1]), KD_TOPK) / (1024 ** 3):.2f} GiB)"
    )
    teacher.eval()
    n_windows = int(train_ids.shape[0])
    n_pos = int(train_ids.shape[1]) - 1
    topk_idx = torch.empty((n_windows, n_pos, KD_TOPK), dtype=torch.int32)
    topk_logits = torch.empty((n_windows, n_pos, KD_TOPK), dtype=torch.bfloat16)
    t0 = time.time()

    with torch.inference_mode():
        for start in range(0, n_windows, TRAIN_BATCH_SIZE):
            ids = train_ids[start : start + TRAIN_BATCH_SIZE].to(DEVICE)
            mask = train_mask[start : start + TRAIN_BATCH_SIZE].to(DEVICE)
            with autocast_context():
                logits = teacher(input_ids=ids, attention_mask=mask, use_cache=False).logits
            shift_logits = logits[:, :-1].contiguous().float()
            values, indices = shift_logits.topk(KD_TOPK, dim=-1)
            batch_n = ids.shape[0]
            topk_idx[start : start + batch_n] = indices.to(torch.int32).cpu()
            topk_logits[start : start + batch_n] = values.to(torch.bfloat16).cpu()
            if start == 0 or (start // TRAIN_BATCH_SIZE) % 200 == 0:
                elapsed = time.time() - t0
                print(f"    cache {start:5d}/{n_windows} windows ({elapsed:.0f}s)")

    payload = {
        "meta": expected_meta,
        "topk_idx": topk_idx,
        "topk_logits": topk_logits,
    }
    torch.save(payload, cache_path)
    print(f"  saved teacher cache: {cache_path} ({time.time() - t0:.0f}s)")
    return {
        "topk_idx": topk_idx,
        "topk_logits": topk_logits,
        "path": str(cache_path),
        "cache_hit": False,
        "estimated_bytes": estimate_teacher_cache_bytes(
            int(train_ids.shape[0]),
            int(train_ids.shape[1]),
            KD_TOPK,
        ),
    }


def build_train_schedule(seed: int, n_examples: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_examples, size=(TRAIN_STEPS, TRAIN_BATCH_SIZE), dtype=np.int64)


def topk_kd_loss(
    student_shift_logits: torch.Tensor,
    teacher_topk_idx: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
) -> torch.Tensor:
    student_at_topk = student_shift_logits.gather(2, teacher_topk_idx)
    student_log_probs = F.log_softmax(student_at_topk / KD_TEMP, dim=-1)
    teacher_probs = F.softmax(teacher_topk_logits / KD_TEMP, dim=-1)
    teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-12))
    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return kl.mean() * (KD_TEMP ** 2)


def train_one_arm(
    arm: ArmSpec,
    *,
    seed: int,
    vocab_size: int,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    c4_val_ids: torch.Tensor,
    c4_val_mask: torch.Tensor,
    wiki_val_ids: torch.Tensor,
    wiki_val_mask: torch.Tensor,
    train_schedule: np.ndarray,
    teacher_cache: dict | None,
):
    set_seed(seed)
    model = make_minimal_student(vocab_size=vocab_size, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_log = []
    t0 = time.time()

    print(
        f"  {arm.label} seed={seed} params={n_total_params / 1e6:.2f}M "
        f"use_kd={arm.use_kd}"
    )

    for step in range(1, TRAIN_STEPS + 1):
        current_lr = warmup_lr(step - 1, LR, LR_WARMUP_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        batch_indices = train_schedule[step - 1]
        batch_index_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
        ids = train_ids[batch_index_tensor].to(DEVICE)
        mask = train_mask[batch_index_tensor].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            ce_loss = causal_ce_loss(logits, ids, mask)
            kd_loss_value = None
            if arm.use_kd:
                if teacher_cache is None:
                    raise RuntimeError("teacher cache missing for KD arm")
                batch_topk_idx = teacher_cache["topk_idx"][batch_index_tensor].to(
                    DEVICE,
                    dtype=torch.long,
                )
                batch_topk_logits = teacher_cache["topk_logits"][batch_index_tensor].to(
                    DEVICE,
                    dtype=torch.float32,
                )
                kd_loss_value = topk_kd_loss(
                    logits[:, :-1].contiguous().float(),
                    batch_topk_idx,
                    batch_topk_logits,
                )
                total_loss = (1.0 - KD_GAMMA) * ce_loss + KD_GAMMA * kd_loss_value
            else:
                total_loss = ce_loss

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite loss at step {step} arm={arm.label} seed={seed}")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == TRAIN_STEPS:
            row = {
                "step": step,
                "lr": float(current_lr),
                "ce_loss": float(ce_loss.item()),
                "kd_loss": float(kd_loss_value.item()) if kd_loss_value is not None else None,
                "total_loss": float(total_loss.item()),
                "elapsed_s": time.time() - t0,
            }
            train_log.append(row)
            kd_text = "na" if row["kd_loss"] is None else f"{row['kd_loss']:.4f}"
            print(
                f"    step={step:5d} ce={row['ce_loss']:.4f} "
                f"kd={kd_text} total={row['total_loss']:.4f} ({row['elapsed_s']:.0f}s)"
            )

    final_metrics = {
        "c4_val": evaluate_model(model, c4_val_ids, c4_val_mask),
        "wikitext_val": evaluate_model(model, wiki_val_ids, wiki_val_mask),
    }
    wallclock_s = time.time() - t0
    print(
        f"    final c4_val: nll={final_metrics['c4_val']['nll']:.4f} "
        f"top1={100.0 * final_metrics['c4_val']['top1_acc']:.2f}%"
    )
    print(
        f"    final wikitext_val: nll={final_metrics['wikitext_val']['nll']:.4f} "
        f"top1={100.0 * final_metrics['wikitext_val']['top1_acc']:.2f}%"
    )

    payload = {
        "seed": seed,
        "arm_label": arm.label,
        "description": arm.description,
        "use_kd": arm.use_kd,
        "n_total_params": int(n_total_params),
        "n_trainable_params": int(n_trainable_params),
        "train_log": train_log,
        "final_metrics": final_metrics,
        "wallclock_s": wallclock_s,
    }

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def paired_bootstrap_ci(values: list[float], *, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        boot_means[idx] = sample.mean()
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _metric_summary(values_by_seed: dict[str, float], *, seed: int) -> dict[str, object]:
    values = [float(values_by_seed[str(s)]) for s in SEEDS]
    ci_lo, ci_hi = paired_bootstrap_ci(values, seed=seed)
    return {
        "per_seed": values_by_seed,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
    }


def build_summary(results: dict) -> dict[str, object]:
    scratch = results["scratch_ce"]
    kd = results["kd_topk64_t2_g0p5"]

    c4_top1_delta_pp = {}
    c4_nll_gain = {}
    wiki_top1_delta_pp = {}
    wiki_nll_gain = {}

    for seed in SEEDS:
        seed_key = str(seed)
        scratch_metrics = scratch[seed_key]["final_metrics"]
        kd_metrics = kd[seed_key]["final_metrics"]

        c4_top1_delta_pp[seed_key] = (
            float(kd_metrics["c4_val"]["top1_acc"]) - float(scratch_metrics["c4_val"]["top1_acc"])
        ) * 100.0
        c4_nll_gain[seed_key] = (
            float(scratch_metrics["c4_val"]["nll"]) - float(kd_metrics["c4_val"]["nll"])
        )
        wiki_top1_delta_pp[seed_key] = (
            float(kd_metrics["wikitext_val"]["top1_acc"]) - float(scratch_metrics["wikitext_val"]["top1_acc"])
        ) * 100.0
        wiki_nll_gain[seed_key] = (
            float(scratch_metrics["wikitext_val"]["nll"]) - float(kd_metrics["wikitext_val"]["nll"])
        )

    active = {
        "c4_val_top1_delta_pp": _metric_summary(c4_top1_delta_pp, seed=167101),
        "c4_val_nll_gain": _metric_summary(c4_nll_gain, seed=167102),
        "wikitext_val_top1_delta_pp": _metric_summary(wiki_top1_delta_pp, seed=167103),
        "wikitext_val_nll_gain": _metric_summary(wiki_nll_gain, seed=167104),
    }

    c4_top1_mean = float(active["c4_val_top1_delta_pp"]["mean"])
    c4_top1_ci_lo = float(active["c4_val_top1_delta_pp"]["ci_95_lo"])
    c4_nll_mean = float(active["c4_val_nll_gain"]["mean"])
    wiki_nll_mean = float(active["wikitext_val_nll_gain"]["mean"])
    positive_primary = int(sum(float(c4_top1_delta_pp[str(s)]) > 0.0 for s in SEEDS))

    dataset_wins = {
        "c4_val": bool(
            float(active["c4_val_top1_delta_pp"]["mean"]) > 0.0
            and float(active["c4_val_nll_gain"]["mean"]) > 0.0
        ),
        "wikitext_val": bool(
            float(active["wikitext_val_top1_delta_pp"]["mean"]) > 0.0
            and float(active["wikitext_val_nll_gain"]["mean"]) > 0.0
        ),
    }

    criteria = {
        "mean_c4_top1_delta_pp_ge_0p40": c4_top1_mean >= PASS_TOP1_GAIN_PP,
        "paired_c4_top1_ci_excludes_zero": c4_top1_ci_lo > 0.0,
        "mean_c4_nll_gain_ge_0p03": c4_nll_mean >= PASS_NLL_GAIN,
        "mean_wikitext_nll_gain_ge_0p03": wiki_nll_mean >= PASS_NLL_GAIN,
        "positive_primary_effect_in_at_least_2_of_3_seeds": positive_primary >= PASS_MIN_POSITIVE_SEEDS,
    }

    if all(criteria.values()):
        verdict = (
            "PASS: KD beats scratch at canonical scale. "
            f"C4 top1 delta={c4_top1_mean:+.3f} pp with 95% CI "
            f"[{float(active['c4_val_top1_delta_pp']['ci_95_lo']):+.3f}, "
            f"{float(active['c4_val_top1_delta_pp']['ci_95_hi']):+.3f}], "
            f"C4 NLL gain={c4_nll_mean:+.4f}, Wikitext NLL gain={wiki_nll_mean:+.4f}, "
            f"positive seeds={positive_primary}/3."
        )
    else:
        failed = [name for name, ok in criteria.items() if not ok]
        verdict = (
            "FAIL: locked KD thresholds not met. "
            f"Failed checks: {', '.join(failed)}. "
            f"Observed C4 top1 delta={c4_top1_mean:+.3f} pp, "
            f"C4 top1 CI=[{float(active['c4_val_top1_delta_pp']['ci_95_lo']):+.3f}, "
            f"{float(active['c4_val_top1_delta_pp']['ci_95_hi']):+.3f}], "
            f"C4 NLL gain={c4_nll_mean:+.4f}, Wikitext NLL gain={wiki_nll_mean:+.4f}, "
            f"positive seeds={positive_primary}/3."
        )

    return {
        "verdict": verdict,
        "criteria": criteria,
        "positive_primary_effect_seeds": positive_primary,
        "dataset_wins": dataset_wins,
        "active_ingredient_analysis": active,
    }


def save_incremental(
    *,
    results: dict,
    teacher_reference: dict,
    data_meta: dict,
    teacher_cache_meta: dict,
    t_start: float,
) -> None:
    payload = {
        "genome": 167,
        "name": "kd_canonical",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_model_id": TEACHER_HF,
        "results": results,
        "teacher_reference": teacher_reference,
        "data": data_meta,
        "teacher_cache": teacher_cache_meta,
        "elapsed_s": time.time() - t_start,
        "incremental": True,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def print_active_ingredient_summary(summary: dict[str, object]) -> None:
    active = summary["active_ingredient_analysis"]

    def _print_metric(label: str, metric: dict[str, object], precision: int) -> None:
        print(f"  {label}:")
        for seed in SEEDS:
            value = float(metric["per_seed"][str(seed)])
            print(f"    seed {seed:>2d}: {value:+.{precision}f}")
        print(
            f"    mean: {float(metric['mean']):+.{precision}f}  "
            f"95% CI [{float(metric['ci_95_lo']):+.{precision}f}, {float(metric['ci_95_hi']):+.{precision}f}]"
        )

    print("\n=== ACTIVE INGREDIENT ANALYSIS ===")
    _print_metric("C4-val top1 delta KD-scratch (pp)", active["c4_val_top1_delta_pp"], 3)
    _print_metric("C4-val NLL gain scratch-KD", active["c4_val_nll_gain"], 4)
    _print_metric("Wikitext-val top1 delta KD-scratch (pp)", active["wikitext_val_top1_delta_pp"], 3)
    _print_metric("Wikitext-val NLL gain scratch-KD", active["wikitext_val_nll_gain"], 4)
    print(
        f"  KD wins C4-val: {'YES' if summary['dataset_wins']['c4_val'] else 'NO'}"
    )
    print(
        f"  KD wins Wikitext-val: {'YES' if summary['dataset_wins']['wikitext_val'] else 'NO'}"
    )
    print(
        f"  Positive primary seeds: {int(summary['positive_primary_effect_seeds'])}/{len(SEEDS)}"
    )


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("genome_167: canonical 3-seed KD scale-up of genome 154")
    print(f"  teacher={TEACHER_HF}")
    print(f"  device={DEVICE} forward_dtype={FORWARD_DTYPE} recipient_dtype=torch.float32")
    print(f"  seeds={SEEDS} steps={TRAIN_STEPS} batch={TRAIN_BATCH_SIZE}")
    print(
        f"  train_windows={N_TRAIN_WINDOWS} c4_val_windows={N_C4_VAL_WINDOWS} "
        f"wikitext_val_windows={N_WIKITEXT_VAL_WINDOWS} seq_len={SEQ_LEN}"
    )

    t_start = time.time()
    tok = load_tokenizer()
    vocab_size = len(tok)
    print(f"  tokenizer_vocab={vocab_size}")

    train_ids, train_mask, train_meta = load_c4_windows(
        tok,
        split="train",
        seed=C4_TRAIN_SEED,
        n_windows=N_TRAIN_WINDOWS,
    )
    train_hashes = collect_13gram_hashes(train_ids, train_mask)
    print(f"  train 13-gram hashes: {len(train_hashes)}")

    c4_val_ids, c4_val_mask, c4_val_meta = load_c4_windows(
        tok,
        split="validation",
        seed=C4_VAL_SEED,
        n_windows=N_C4_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    wiki_val_ids, wiki_val_mask, wiki_val_meta = load_wikitext_windows(
        tok,
        split="validation",
        seed=WIKITEXT_VAL_SEED,
        n_windows=N_WIKITEXT_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )

    data_meta = {
        "train": train_meta,
        "c4_val": c4_val_meta,
        "wikitext_val": wiki_val_meta,
        "train_13gram_hash_count": len(train_hashes),
    }

    teacher, _ = load_trained_teacher(tok)
    teacher_reference = {
        "c4_val": evaluate_model(teacher, c4_val_ids, c4_val_mask),
        "wikitext_val": evaluate_model(teacher, wiki_val_ids, wiki_val_mask),
    }
    print(
        f"  teacher c4_val: nll={teacher_reference['c4_val']['nll']:.4f} "
        f"top1={100.0 * teacher_reference['c4_val']['top1_acc']:.2f}%"
    )
    print(
        f"  teacher wikitext_val: nll={teacher_reference['wikitext_val']['nll']:.4f} "
        f"top1={100.0 * teacher_reference['wikitext_val']['top1_acc']:.2f}%"
    )

    train_signature = tensor_sha1(train_ids)
    teacher_cache = precompute_teacher_topk_cache(
        teacher,
        train_ids,
        train_mask,
        train_signature=train_signature,
    )

    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher_cache_meta = {
        "path": teacher_cache["path"],
        "cache_hit": teacher_cache["cache_hit"],
        "estimated_bytes": teacher_cache["estimated_bytes"],
        "estimated_gib": teacher_cache["estimated_bytes"] / (1024 ** 3),
        "train_signature": train_signature,
    }

    results = {arm.label: {} for arm in ARM_SPECS}
    print(f"\n=== Running {len(ARM_SPECS)} arms x {len(SEEDS)} seeds = {len(ARM_SPECS) * len(SEEDS)} cells ===")
    print("=== Pairing rule: same seed -> same init and same batch schedule for scratch and KD ===")

    for seed in SEEDS:
        train_schedule = build_train_schedule(seed, n_examples=int(train_ids.shape[0]))
        for arm in ARM_SPECS:
            print(f"\n--- arm={arm.label} seed={seed} ---")
            payload = train_one_arm(
                arm,
                seed=seed,
                vocab_size=vocab_size,
                train_ids=train_ids,
                train_mask=train_mask,
                c4_val_ids=c4_val_ids,
                c4_val_mask=c4_val_mask,
                wiki_val_ids=wiki_val_ids,
                wiki_val_mask=wiki_val_mask,
                train_schedule=train_schedule,
                teacher_cache=teacher_cache if arm.use_kd else None,
            )
            results[arm.label][str(seed)] = payload
            save_incremental(
                results=results,
                teacher_reference=teacher_reference,
                data_meta=data_meta,
                teacher_cache_meta=teacher_cache_meta,
                t_start=t_start,
            )

    summary = build_summary(results)
    print_active_ingredient_summary(summary)

    out = {
        "genome": 167,
        "name": "kd_canonical",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_model_id": TEACHER_HF,
        "student_family": "minimal_3L_noMLP_llama_qwen_vocab",
        "device": DEVICE,
        "forward_dtype": str(FORWARD_DTYPE),
        "recipient_param_dtype": "torch.float32",
        "config": {
            "seeds": SEEDS,
            "seq_len": SEQ_LEN,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "train_steps": TRAIN_STEPS,
            "lr": LR,
            "lr_warmup_steps": LR_WARMUP_STEPS,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "n_train_windows": N_TRAIN_WINDOWS,
            "n_c4_val_windows": N_C4_VAL_WINDOWS,
            "n_wikitext_val_windows": N_WIKITEXT_VAL_WINDOWS,
            "kd_top_k": KD_TOPK,
            "kd_temp": KD_TEMP,
            "kd_gamma": KD_GAMMA,
            "score_on_pass": 7.0,
            "compute_envelope": {
                "soft_hours": 3.5,
                "hard_hours": 4.0,
                "max_vram_gb": 22.0,
            },
            "data_seeds": {
                "c4_train": C4_TRAIN_SEED,
                "c4_val": C4_VAL_SEED,
                "wikitext_val": WIKITEXT_VAL_SEED,
            },
        },
        "data": data_meta,
        "teacher_cache": teacher_cache_meta,
        "teacher_reference": teacher_reference,
        "results": results,
        "summary": summary,
        "verdict": summary["verdict"],
        "elapsed_s": time.time() - t_start,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\n=== verdict: {summary['verdict']} ===")
    print(f"Saved: {OUT_PATH} ({out['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()