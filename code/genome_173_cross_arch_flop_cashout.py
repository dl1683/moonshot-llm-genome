"""
genome_173_cross_arch_flop_cashout.py

Preregistered cross-architecture KD cash-out in FLOPs.

Cycle 42 direction review recommendation:
  Replace nearby mechanism variants with a real end-task cash-out:
  Qwen teacher -> Llama-architecture student on C3_macro, with honest
  train+inference compute accounting.

Locked design
-------------
Teacher:
  - Qwen/Qwen3-0.6B, frozen, BF16 forward on GPU

Students:
  - Llama-arch random-init student
      hidden=768, layers=8, heads=12, ffn=2048
  - Qwen3-arch random-init control student
      matched width/depth/tokenizer budget to isolate architecture family

Tokenizer:
  - Shared Qwen3 tokenizer / vocab for every arm so logit KD is defined on a
    single vocabulary. This inflates total parameter count because the shared
    vocab is 151,936 tokens, but it keeps the comparison well-posed.

Arms (4 x 3 seeds = 12 cells):
  - scratch_ce_llama
  - kd_logit_llama
  - scratch_ce_qwen_arch
  - kd_logit_qwen_arch

Training:
  - 8192 C4 train windows, len 256
  - 1000 C4 validation windows, len 256
  - 6000 steps, batch 8, lr 3e-4
  - BF16 forward autocast, FP32 master weights
  - top-k KD with k=64, T=2.0, gamma=0.5

End-task eval at step 6000:
  - C4 validation NLL / top-1
  - HellaSwag / PIQA / Winogrande, 500 validation examples each
  - C3_macro = mean(task accuracies)

FLOP accounting
---------------
Primary accounting follows the cycle-42 brief:
  - teacher_cache_flops  ~= n_windows * seq_len * 2 * teacher_params
  - student_train_flops  ~= steps * batch * seq_len * 6 * student_params
  - student_eval_flops   ~= eval_forward_tokens * 2 * student_params

For the matched-capability cash-out, we also report:
  matched_qwen_kd_compute = llama_kd_total_compute / retention_vs_qwen_kd
where:
  retention_vs_qwen_kd = C3_llama_kd / C3_qwen_kd

Outputs:
  - results/genome_173_cross_arch_flop_cashout.json
  - reuses results/cache/genome_167_teacher_topk_cache_<train_hash>.pt when
    the train signature matches genome_167.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "models"))
try:
    from registry import resolve as _resolve_model  # type: ignore

    TEACHER_HF = _resolve_model("qwen3-0.6b").get("hf_id", "Qwen/Qwen3-0.6B")
except Exception:
    TEACHER_HF = "Qwen/Qwen3-0.6B"

OUT_PATH = ROOT / "results" / "genome_173_cross_arch_flop_cashout.json"
CACHE_DIR = ROOT / "results" / "cache"

SEEDS = [42, 7, 13]
SEQ_LEN = 256
MC_MAX_LENGTH = SEQ_LEN + 128
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
N_C3_PER_TASK = 500

KD_TOPK = 64
KD_TEMP = 2.0
KD_GAMMA = 0.5
N_BOOT = 10000

C4_TRAIN_SEED = 167001
C4_VAL_SEED = 167101
HELLASWAG_SEED = 173401
PIQA_SEED = 173402
WINOGRANDE_SEED = 173403

PASS_MIN_RETENTION = 0.75
PASS_MAX_MATCHED_COMPUTE_FRACTION = 0.40
PASS_MIN_TRANSFER_RATIO = 2.5
FAIL_MIN_RETENTION = 0.50
FAIL_MIN_TRANSFER_RATIO = 1.5

HASH_LEN = 13
HASH_MULT = np.uint64(1099511628211)
HASH_OFFSET = np.uint64(1099511628211 * 1469598103934665603 % (2**64))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORWARD_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class StudentSpec:
    label: str
    architecture: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    head_dim: int | None
    description: str


@dataclass(frozen=True)
class ArmSpec:
    label: str
    use_kd: bool
    student: StudentSpec
    description: str


LLAMA_SPEC = StudentSpec(
    label="llama_8L_768h_12a_2048ffn_qwen_vocab",
    architecture="llama",
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=12,
    num_key_value_heads=12,
    intermediate_size=2048,
    head_dim=None,
    description=(
        "Cross-family student: Llama block structure, random init, shared Qwen3 "
        "tokenizer / vocab so top-k KD is defined on one logit space."
    ),
)

QWEN_SPEC = StudentSpec(
    label="qwen3_8L_768h_12a_2048ffn_qwen_vocab",
    architecture="qwen3",
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=12,
    num_key_value_heads=6,
    intermediate_size=2048,
    head_dim=64,
    description=(
        "Same-tokenizer Qwen3-family control student at comparable width/depth "
        "budget, used to isolate architecture-family effects from tokenizer and KD."
    ),
)

ARM_SPECS = [
    ArmSpec(
        label="scratch_ce_llama",
        use_kd=False,
        student=LLAMA_SPEC,
        description="Llama-arch student trained from scratch with CE only.",
    ),
    ArmSpec(
        label="kd_logit_llama",
        use_kd=True,
        student=LLAMA_SPEC,
        description="Llama-arch student with CE + top-k logit KD from Qwen teacher.",
    ),
    ArmSpec(
        label="scratch_ce_qwen_arch",
        use_kd=False,
        student=QWEN_SPEC,
        description="Qwen3-arch student control trained from scratch with CE only.",
    ),
    ArmSpec(
        label="kd_logit_qwen_arch",
        use_kd=True,
        student=QWEN_SPEC,
        description="Qwen3-arch student control with CE + top-k logit KD.",
    ),
]


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
        local_files_only=True,
        trust_remote_code=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    if tok.pad_token_id is None:
        raise RuntimeError("tokenizer pad_token_id unresolved")
    if tok.eos_token_id is None:
        raise RuntimeError("tokenizer eos_token_id unresolved")
    return tok


def load_trained_teacher(tok=None):
    if tok is None:
        tok = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_HF,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=FORWARD_DTYPE,
    ).to(DEVICE).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tok


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


def _prepare_multiple_choice_item(
    tok,
    *,
    context: str,
    choices: list[str],
    label: int,
    benchmark: str,
    item_id: str,
) -> dict[str, Any]:
    candidates = []
    total_forward_tokens = 0
    context = context or ""
    for choice_idx, choice in enumerate(choices):
        if context.strip():
            full_text = context + choice
            full_ids = tok(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=MC_MAX_LENGTH,
            )["input_ids"]
            ctx_ids = tok(
                context,
                add_special_tokens=False,
                truncation=True,
                max_length=MC_MAX_LENGTH,
            )["input_ids"]
            choice_start = len(ctx_ids)
        else:
            choice_ids = tok(
                choice,
                add_special_tokens=False,
                truncation=True,
                max_length=MC_MAX_LENGTH - 1,
            )["input_ids"]
            full_ids = [tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id] + choice_ids
            choice_start = 1

        if len(full_ids) < 2:
            full_ids = [tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id, tok.eos_token_id]
            choice_start = 1

        total_forward_tokens += len(full_ids)
        candidates.append(
            {
                "choice_idx": choice_idx,
                "full_ids": full_ids,
                "choice_start": choice_start,
            }
        )

    return {
        "benchmark": benchmark,
        "item_id": item_id,
        "label": int(label),
        "n_choices": len(candidates),
        "candidates": candidates,
        "total_forward_tokens": int(total_forward_tokens),
    }


def load_c3_validation(tok, *, n_per_task: int = N_C3_PER_TASK):
    print("Loading C3 validation sets...")
    out: dict[str, list[dict[str, Any]]] = {}

    # HellaSwag
    hs_items: list[dict[str, Any]] = []
    try:
        try:
            hs = load_dataset("hellaswag", split="validation", trust_remote_code=True)
        except Exception:
            hs = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
        hs = hs.shuffle(seed=HELLASWAG_SEED)
        for idx, ex in enumerate(hs):
            ctx = ex["ctx"] if ex.get("ctx") else (
                ex.get("activity_label", "") + " " + ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")
            )
            endings = list(ex["endings"])
            label = int(ex["label"])
            hs_items.append(
                _prepare_multiple_choice_item(
                    tok,
                    context=ctx,
                    choices=endings,
                    label=label,
                    benchmark="hellaswag",
                    item_id=f"hellaswag_{idx}",
                )
            )
            if len(hs_items) >= n_per_task:
                break
    except Exception as exc:
        raise RuntimeError(f"hellaswag load failed: {exc}") from exc
    out["hellaswag"] = hs_items
    print(f"  hellaswag: {len(hs_items)}")

    # PIQA
    piqa_items: list[dict[str, Any]] = []
    try:
        piqa = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
        piqa = piqa.shuffle(seed=PIQA_SEED)
        for idx, ex in enumerate(piqa):
            piqa_items.append(
                _prepare_multiple_choice_item(
                    tok,
                    context=ex["goal"] + " ",
                    choices=[ex["sol1"], ex["sol2"]],
                    label=int(ex["label"]),
                    benchmark="piqa",
                    item_id=f"piqa_{idx}",
                )
            )
            if len(piqa_items) >= n_per_task:
                break
    except Exception as exc:
        raise RuntimeError(f"piqa load failed: {exc}") from exc
    out["piqa"] = piqa_items
    print(f"  piqa: {len(piqa_items)}")

    # Winogrande
    wino_items: list[dict[str, Any]] = []
    try:
        wino = load_dataset(
            "allenai/winogrande",
            "winogrande_debiased",
            split="validation",
            trust_remote_code=True,
        )
        wino = wino.shuffle(seed=WINOGRANDE_SEED)
        for idx, ex in enumerate(wino):
            template = ex["sentence"].replace("_", "{}")
            choices = [template.format(ex["option1"]), template.format(ex["option2"])]
            wino_items.append(
                _prepare_multiple_choice_item(
                    tok,
                    context="",
                    choices=choices,
                    label=int(ex["answer"]) - 1,
                    benchmark="winogrande",
                    item_id=f"winogrande_{idx}",
                )
            )
            if len(wino_items) >= n_per_task:
                break
    except Exception as exc:
        raise RuntimeError(f"winogrande load failed: {exc}") from exc
    out["winogrande"] = wino_items
    print(f"  winogrande: {len(wino_items)}")

    required = ("hellaswag", "piqa", "winogrande")
    missing = [name for name in required if len(out.get(name, [])) == 0]
    if missing:
        raise RuntimeError(f"C3 validation incomplete; missing/empty tasks: {missing}")

    token_meta = {}
    for name, items in out.items():
        token_meta[name] = {
            "n_items": len(items),
            "n_choices_total": int(sum(item["n_choices"] for item in items)),
            "forward_tokens": int(sum(item["total_forward_tokens"] for item in items)),
        }

    return out, token_meta


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
def evaluate_c4(model, eval_ids: torch.Tensor, eval_mask: torch.Tensor) -> dict[str, float]:
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


@torch.no_grad()
def evaluate_c3(model, c3_data: dict[str, list[dict[str, Any]]], *, pad_token_id: int) -> dict[str, Any]:
    model.eval()
    per_task = {}
    total_forward_tokens = 0
    total_items = 0

    for task_name, items in c3_data.items():
        correct = 0
        for item in items:
            n_choices = int(item["n_choices"])
            max_len = max(len(candidate["full_ids"]) for candidate in item["candidates"])

            ids = torch.full((n_choices, max_len), pad_token_id, dtype=torch.long)
            mask = torch.zeros((n_choices, max_len), dtype=torch.long)
            for row_idx, candidate in enumerate(item["candidates"]):
                full_ids = candidate["full_ids"]
                ids[row_idx, : len(full_ids)] = torch.tensor(full_ids, dtype=torch.long)
                mask[row_idx, : len(full_ids)] = 1

            total_forward_tokens += int(mask.sum().item())
            ids = ids.to(DEVICE)
            mask = mask.to(DEVICE)

            with autocast_context():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()

            mean_nlls = []
            for row_idx, candidate in enumerate(item["candidates"]):
                valid_end = int(mask[row_idx].sum().item())
                choice_start = int(candidate["choice_start"])
                if valid_end <= choice_start:
                    mean_nlls.append(float("inf"))
                    continue
                pred_logits = logits[row_idx, choice_start - 1 : valid_end - 1]
                targets = ids[row_idx, choice_start:valid_end]
                if pred_logits.shape[0] != targets.shape[0] or targets.numel() == 0:
                    mean_nlls.append(float("inf"))
                    continue
                log_probs = F.log_softmax(pred_logits, dim=-1)
                tok_log_probs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
                mean_nlls.append(float((-tok_log_probs.mean()).item()))

            pred = int(np.argmin(np.asarray(mean_nlls, dtype=np.float64)))
            if pred == int(item["label"]):
                correct += 1

        total_items += len(items)
        per_task[task_name] = correct / max(len(items), 1)

    model.train()
    c3_macro = float(np.mean(list(per_task.values()))) if per_task else 0.0
    return {
        "per_task": per_task,
        "C3_macro": c3_macro,
        "n_items_total": total_items,
        "forward_tokens_measured": int(total_forward_tokens),
    }


def tensor_sha1(tensor: torch.Tensor) -> str:
    array = tensor.cpu().numpy()
    return hashlib.sha1(array.tobytes()).hexdigest()


def estimate_teacher_cache_bytes(n_windows: int, seq_len: int, top_k: int) -> int:
    positions = n_windows * (seq_len - 1) * top_k
    idx_bytes = positions * 4
    logit_bytes = positions * 2
    return idx_bytes + logit_bytes


def estimate_teacher_cache_flops(teacher_params: int, *, n_windows: int, seq_len: int) -> int:
    return int(n_windows * seq_len * 2 * teacher_params)


def estimate_train_flops(student_params: int) -> int:
    return int(TRAIN_STEPS * TRAIN_BATCH_SIZE * SEQ_LEN * 6 * student_params)


def estimate_eval_flops(student_params: int, *, eval_forward_tokens: int) -> int:
    return int(eval_forward_tokens * 2 * student_params)


def _teacher_cache_meta_expected(train_ids: torch.Tensor, train_signature: str) -> dict[str, Any]:
    return {
        "teacher_hf": TEACHER_HF,
        "train_signature": train_signature,
        "n_train_windows": int(train_ids.shape[0]),
        "seq_len": int(train_ids.shape[1]),
        "top_k": KD_TOPK,
        "dtype_logits": "bfloat16",
        "dtype_indices": "int32",
    }


def _load_teacher_cache_path(path: Path, *, expected_meta: dict[str, Any]):
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if payload.get("meta") != expected_meta:
        return None
    return {
        "topk_idx": payload["topk_idx"],
        "topk_logits": payload["topk_logits"],
        "path": str(path),
    }


def precompute_teacher_topk_cache(
    teacher,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    train_signature: str,
):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    expected_meta = _teacher_cache_meta_expected(train_ids, train_signature)
    g167_path = CACHE_DIR / f"genome_167_teacher_topk_cache_{train_signature[:16]}.pt"
    own_path = CACHE_DIR / f"genome_173_teacher_topk_cache_{train_signature[:16]}.pt"

    for cache_path, cache_source in ((g167_path, "genome_167"), (own_path, "genome_173")):
        loaded = _load_teacher_cache_path(cache_path, expected_meta=expected_meta)
        if loaded is not None:
            print(f"  teacher cache hit ({cache_source}): {cache_path}")
            estimated_bytes = estimate_teacher_cache_bytes(
                int(train_ids.shape[0]),
                int(train_ids.shape[1]),
                KD_TOPK,
            )
            return {
                "topk_idx": loaded["topk_idx"],
                "topk_logits": loaded["topk_logits"],
                "path": loaded["path"],
                "cache_hit": True,
                "cache_source": cache_source,
                "estimated_bytes": estimated_bytes,
            }

    estimated_bytes = estimate_teacher_cache_bytes(
        int(train_ids.shape[0]),
        int(train_ids.shape[1]),
        KD_TOPK,
    )
    print(
        f"  precomputing teacher top-{KD_TOPK} cache over {train_ids.shape[0]} windows "
        f"(approx {estimated_bytes / (1024 ** 3):.2f} GiB)"
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
    torch.save(payload, own_path)
    print(f"  saved teacher cache: {own_path} ({time.time() - t0:.0f}s)")
    return {
        "topk_idx": topk_idx,
        "topk_logits": topk_logits,
        "path": str(own_path),
        "cache_hit": False,
        "cache_source": "genome_173",
        "estimated_bytes": estimated_bytes,
    }


def build_llama_student(vocab_size: int, seed: int):
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=LLAMA_SPEC.hidden_size,
        num_hidden_layers=LLAMA_SPEC.num_hidden_layers,
        num_attention_heads=LLAMA_SPEC.num_attention_heads,
        num_key_value_heads=LLAMA_SPEC.num_key_value_heads,
        intermediate_size=LLAMA_SPEC.intermediate_size,
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    set_seed(seed)
    model = LlamaForCausalLM(cfg).to(DEVICE)
    model.train()
    return model


def build_qwen_student(vocab_size: int, seed: int):
    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=QWEN_SPEC.hidden_size,
        intermediate_size=QWEN_SPEC.intermediate_size,
        num_hidden_layers=QWEN_SPEC.num_hidden_layers,
        num_attention_heads=QWEN_SPEC.num_attention_heads,
        num_key_value_heads=QWEN_SPEC.num_key_value_heads,
        head_dim=QWEN_SPEC.head_dim,
        max_position_embeddings=SEQ_LEN + 64,
        max_window_layers=QWEN_SPEC.num_hidden_layers,
        layer_types=["full_attention"] * QWEN_SPEC.num_hidden_layers,
        tie_word_embeddings=True,
        use_sliding_window=False,
        sliding_window=None,
        rope_theta=1_000_000,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_dropout=0.0,
        attention_bias=False,
    )
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    set_seed(seed)
    model = Qwen3ForCausalLM(cfg).to(DEVICE)
    model.train()
    return model


def build_student(student: StudentSpec, *, vocab_size: int, seed: int):
    if student.architecture == "llama":
        return build_llama_student(vocab_size, seed)
    if student.architecture == "qwen3":
        return build_qwen_student(vocab_size, seed)
    raise ValueError(f"unknown student architecture: {student.architecture}")


def count_model_parameters(model) -> dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed_params = 0
    emb = model.get_input_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        embed_params = int(emb.weight.numel())
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "token_embedding_params": int(embed_params),
        "non_embedding_params": int(total_params - embed_params),
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


def compute_arm_flops(
    *,
    student_params: int,
    c4_eval_forward_tokens: int,
    c3_eval_forward_tokens: int,
    teacher_cache_flops: int,
    use_kd: bool,
) -> dict[str, Any]:
    train_flops = estimate_train_flops(student_params)
    c4_eval_flops = estimate_eval_flops(student_params, eval_forward_tokens=c4_eval_forward_tokens)
    c3_eval_flops = estimate_eval_flops(student_params, eval_forward_tokens=c3_eval_forward_tokens)
    student_eval_flops = c4_eval_flops + c3_eval_flops
    program_contribution_flops = train_flops + student_eval_flops
    standalone_total_flops = program_contribution_flops + (teacher_cache_flops if use_kd else 0)
    return {
        "teacher_cache_flops_included": int(teacher_cache_flops if use_kd else 0),
        "student_train_flops": int(train_flops),
        "student_eval_flops": int(student_eval_flops),
        "student_eval_c4_flops": int(c4_eval_flops),
        "student_eval_c3_flops": int(c3_eval_flops),
        "program_contribution_flops": int(program_contribution_flops),
        "standalone_total_flops": int(standalone_total_flops),
        "teacher_cache_pflops_included": float((teacher_cache_flops if use_kd else 0) / 1e15),
        "student_train_pflops": float(train_flops / 1e15),
        "student_eval_pflops": float(student_eval_flops / 1e15),
        "program_contribution_pflops": float(program_contribution_flops / 1e15),
        "standalone_total_pflops": float(standalone_total_flops / 1e15),
    }


def train_one_arm(
    arm: ArmSpec,
    *,
    seed: int,
    vocab_size: int,
    pad_token_id: int,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    c4_val_ids: torch.Tensor,
    c4_val_mask: torch.Tensor,
    c3_data: dict[str, list[dict[str, Any]]],
    c4_eval_forward_tokens: int,
    c3_eval_forward_tokens: int,
    train_schedule: np.ndarray,
    teacher_cache: dict | None,
    teacher_cache_flops: int,
):
    set_seed(seed)
    model = build_student(arm.student, vocab_size=vocab_size, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    param_info = count_model_parameters(model)
    train_log = []
    t0 = time.time()

    print(
        f"  {arm.label} seed={seed} arch={arm.student.architecture} "
        f"params={param_info['total_params'] / 1e6:.2f}M use_kd={arm.use_kd}"
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
                "step": int(step),
                "lr": float(current_lr),
                "ce_loss": float(ce_loss.item()),
                "kd_loss": float(kd_loss_value.item()) if kd_loss_value is not None else None,
                "total_loss": float(total_loss.item()),
                "elapsed_s": float(time.time() - t0),
            }
            train_log.append(row)
            kd_text = "na" if row["kd_loss"] is None else f"{row['kd_loss']:.4f}"
            print(
                f"    step={step:5d} ce={row['ce_loss']:.4f} "
                f"kd={kd_text} total={row['total_loss']:.4f} ({row['elapsed_s']:.0f}s)"
            )

    final_metrics = {
        "c4_val": evaluate_c4(model, c4_val_ids, c4_val_mask),
        "c3_val": evaluate_c3(model, c3_data, pad_token_id=pad_token_id),
    }
    wallclock_s = time.time() - t0
    compute = compute_arm_flops(
        student_params=param_info["total_params"],
        c4_eval_forward_tokens=c4_eval_forward_tokens,
        c3_eval_forward_tokens=c3_eval_forward_tokens,
        teacher_cache_flops=teacher_cache_flops,
        use_kd=arm.use_kd,
    )
    print(
        f"    final c4_val: nll={final_metrics['c4_val']['nll']:.4f} "
        f"top1={100.0 * final_metrics['c4_val']['top1_acc']:.2f}%"
    )
    print(
        f"    final c3_macro: {100.0 * final_metrics['c3_val']['C3_macro']:.2f}% "
        f"(HS={100.0 * final_metrics['c3_val']['per_task']['hellaswag']:.2f} "
        f"PIQA={100.0 * final_metrics['c3_val']['per_task']['piqa']:.2f} "
        f"Wino={100.0 * final_metrics['c3_val']['per_task']['winogrande']:.2f})"
    )

    payload = {
        "seed": int(seed),
        "arm_label": arm.label,
        "description": arm.description,
        "student": asdict(arm.student),
        "use_kd": bool(arm.use_kd),
        "param_info": param_info,
        "train_log": train_log,
        "final_metrics": final_metrics,
        "compute": compute,
        "wallclock_s": float(wallclock_s),
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


def _metric_summary(values_by_seed: dict[str, float], *, seed: int) -> dict[str, Any]:
    values = [float(values_by_seed[str(s)]) for s in SEEDS]
    ci_lo, ci_hi = paired_bootstrap_ci(values, seed=seed)
    return {
        "per_seed": values_by_seed,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
    }


def build_summary(
    *,
    results: dict[str, dict[str, Any]],
    teacher_reference: dict[str, Any],
    teacher_cache_meta: dict[str, Any],
) -> dict[str, Any]:
    arm_summary: dict[str, Any] = {}
    for arm in ARM_SPECS:
        c3_by_seed = {}
        c4_nll_by_seed = {}
        total_compute_by_seed = {}
        per_task_by_seed = {
            "hellaswag": {},
            "piqa": {},
            "winogrande": {},
        }
        for seed in SEEDS:
            seed_key = str(seed)
            payload = results[arm.label][seed_key]
            c3_by_seed[seed_key] = float(payload["final_metrics"]["c3_val"]["C3_macro"])
            c4_nll_by_seed[seed_key] = float(payload["final_metrics"]["c4_val"]["nll"])
            total_compute_by_seed[seed_key] = float(payload["compute"]["standalone_total_flops"])
            for task_name in per_task_by_seed:
                per_task_by_seed[task_name][seed_key] = float(
                    payload["final_metrics"]["c3_val"]["per_task"][task_name]
                )

        arm_summary[arm.label] = {
            "description": arm.description,
            "student": asdict(arm.student),
            "c3_macro": _metric_summary(c3_by_seed, seed=173001 + len(arm.label)),
            "c4_val_nll": _metric_summary(c4_nll_by_seed, seed=173101 + len(arm.label)),
            "standalone_total_flops": _metric_summary(total_compute_by_seed, seed=173201 + len(arm.label)),
            "standalone_total_pflops": {
                "per_seed": {k: v / 1e15 for k, v in total_compute_by_seed.items()},
                "mean": float(np.mean(list(total_compute_by_seed.values())) / 1e15),
                "std": float(np.std(list(total_compute_by_seed.values())) / 1e15),
            },
            "per_task": {
                task_name: _metric_summary(per_task_by_seed[task_name], seed=173301 + idx * 10 + len(arm.label))
                for idx, task_name in enumerate(("hellaswag", "piqa", "winogrande"))
            },
        }

    retention_vs_qwen_by_seed = {}
    retention_vs_teacher_by_seed = {}
    llama_matched_compute_by_seed = {}
    qwen_total_compute_by_seed = {}
    matched_compute_fraction_by_seed = {}
    transfer_ratio_by_seed = {}
    llama_positive_direction_by_seed = {}
    qwen_positive_direction_by_seed = {}

    teacher_c3 = float(teacher_reference["c3_val"]["C3_macro"])
    for seed in SEEDS:
        seed_key = str(seed)
        llama_kd = results["kd_logit_llama"][seed_key]
        qwen_kd = results["kd_logit_qwen_arch"][seed_key]
        llama_scratch = results["scratch_ce_llama"][seed_key]
        qwen_scratch = results["scratch_ce_qwen_arch"][seed_key]

        llama_c3 = float(llama_kd["final_metrics"]["c3_val"]["C3_macro"])
        qwen_c3 = float(qwen_kd["final_metrics"]["c3_val"]["C3_macro"])
        llama_total = float(llama_kd["compute"]["standalone_total_flops"])
        qwen_total = float(qwen_kd["compute"]["standalone_total_flops"])

        retention_qwen = llama_c3 / max(qwen_c3, 1e-12)
        matched_compute = llama_total / max(retention_qwen, 1e-12)
        matched_fraction = matched_compute / max(qwen_total, 1.0)
        transfer_ratio = qwen_total / max(matched_compute, 1.0)

        retention_vs_qwen_by_seed[seed_key] = float(retention_qwen)
        retention_vs_teacher_by_seed[seed_key] = float(llama_c3 / max(teacher_c3, 1e-12))
        llama_matched_compute_by_seed[seed_key] = float(matched_compute)
        qwen_total_compute_by_seed[seed_key] = float(qwen_total)
        matched_compute_fraction_by_seed[seed_key] = float(matched_fraction)
        transfer_ratio_by_seed[seed_key] = float(transfer_ratio)
        llama_positive_direction_by_seed[seed_key] = float(llama_c3 > float(llama_scratch["final_metrics"]["c3_val"]["C3_macro"]))
        qwen_positive_direction_by_seed[seed_key] = float(qwen_c3 > float(qwen_scratch["final_metrics"]["c3_val"]["C3_macro"]))

    criteria = {
        "llama_kd_retention_vs_qwen_kd_ge_0p75": float(np.mean(list(retention_vs_qwen_by_seed.values()))) >= PASS_MIN_RETENTION,
        "matched_compute_fraction_le_0p40": float(np.mean(list(matched_compute_fraction_by_seed.values()))) <= PASS_MAX_MATCHED_COMPUTE_FRACTION,
        "cross_arch_transfer_ratio_ge_2p5": float(np.mean(list(transfer_ratio_by_seed.values()))) >= PASS_MIN_TRANSFER_RATIO,
        "all_llama_seeds_positive_direction_vs_scratch": all(bool(v) for v in llama_positive_direction_by_seed.values()),
    }

    fail_conditions = {
        "llama_kd_retention_vs_qwen_kd_lt_0p50": float(np.mean(list(retention_vs_qwen_by_seed.values()))) < FAIL_MIN_RETENTION,
        "cross_arch_transfer_ratio_lt_1p5": float(np.mean(list(transfer_ratio_by_seed.values()))) < FAIL_MIN_TRANSFER_RATIO,
    }

    if all(criteria.values()):
        verdict = (
            "PASS: cross-architecture KD cash-out lands. "
            f"Llama KD retains {100.0 * float(np.mean(list(retention_vs_qwen_by_seed.values()))):.1f}% "
            f"of Qwen-arch KD C3_macro, matched-capability compute fraction="
            f"{float(np.mean(list(matched_compute_fraction_by_seed.values()))):.3f}, "
            f"transfer ratio={float(np.mean(list(transfer_ratio_by_seed.values()))):.2f}x."
        )
    elif any(fail_conditions.values()):
        failed = [name for name, ok in fail_conditions.items() if ok]
        verdict = (
            "FAIL: cross-architecture KD cash-out misses the locked floor. "
            f"Failed conditions: {', '.join(failed)}. "
            f"Retention={100.0 * float(np.mean(list(retention_vs_qwen_by_seed.values()))):.1f}%, "
            f"transfer ratio={float(np.mean(list(transfer_ratio_by_seed.values()))):.2f}x."
        )
    else:
        missed = [name for name, ok in criteria.items() if not ok]
        verdict = (
            "PARTIAL: cross-architecture KD is directional but below flagship cash-out. "
            f"Missed pass checks: {', '.join(missed)}. "
            f"Retention={100.0 * float(np.mean(list(retention_vs_qwen_by_seed.values()))):.1f}%, "
            f"transfer ratio={float(np.mean(list(transfer_ratio_by_seed.values()))):.2f}x."
        )

    full_program_flops = int(teacher_cache_meta["teacher_cache_flops"])
    for arm in ARM_SPECS:
        for seed in SEEDS:
            full_program_flops += int(results[arm.label][str(seed)]["compute"]["program_contribution_flops"])

    one_seed_panel_flops = int(teacher_cache_meta["teacher_cache_flops"])
    for arm in ARM_SPECS:
        mean_contrib = np.mean(
            [results[arm.label][str(seed)]["compute"]["program_contribution_flops"] for seed in SEEDS]
        )
        one_seed_panel_flops += int(mean_contrib)

    return {
        "verdict": verdict,
        "criteria": criteria,
        "fail_conditions": fail_conditions,
        "arm_summary": arm_summary,
        "teacher_reference_c3_macro": teacher_c3,
        "cross_arch_cashout": {
            "retention_vs_qwen_kd": _metric_summary(retention_vs_qwen_by_seed, seed=173501),
            "retention_vs_teacher": _metric_summary(retention_vs_teacher_by_seed, seed=173502),
            "llama_matched_qwen_kd_compute_flops": _metric_summary(llama_matched_compute_by_seed, seed=173503),
            "llama_matched_qwen_kd_compute_pflops": {
                "per_seed": {k: v / 1e15 for k, v in llama_matched_compute_by_seed.items()},
                "mean": float(np.mean(list(llama_matched_compute_by_seed.values())) / 1e15),
                "std": float(np.std(list(llama_matched_compute_by_seed.values())) / 1e15),
            },
            "qwen_kd_total_compute_flops": _metric_summary(qwen_total_compute_by_seed, seed=173504),
            "matched_compute_fraction_vs_qwen_kd": _metric_summary(matched_compute_fraction_by_seed, seed=173505),
            "transfer_ratio_compute_saved_at_matched_capability": _metric_summary(transfer_ratio_by_seed, seed=173506),
            "llama_positive_direction_vs_scratch": {
                "per_seed": llama_positive_direction_by_seed,
                "all_positive": all(bool(v) for v in llama_positive_direction_by_seed.values()),
            },
            "qwen_positive_direction_vs_scratch": {
                "per_seed": qwen_positive_direction_by_seed,
                "all_positive": all(bool(v) for v in qwen_positive_direction_by_seed.values()),
            },
            "note": (
                "Matched-capability compute is defined as arm_total_compute / retention_vs_qwen_kd. "
                "This is the honest capability-normalized FLOP cash-out requested by cycle 42."
            ),
        },
        "program_compute": {
            "teacher_cache_flops_once": int(teacher_cache_meta["teacher_cache_flops"]),
            "teacher_cache_pflops_once": float(teacher_cache_meta["teacher_cache_flops"] / 1e15),
            "one_seed_4arm_panel_flops": int(one_seed_panel_flops),
            "one_seed_4arm_panel_pflops": float(one_seed_panel_flops / 1e15),
            "full_12_cell_program_flops": int(full_program_flops),
            "full_12_cell_program_pflops": float(full_program_flops / 1e15),
        },
    }


def save_incremental(
    *,
    results: dict[str, dict[str, Any]],
    teacher_reference: dict[str, Any],
    data_meta: dict[str, Any],
    c3_meta: dict[str, Any],
    teacher_cache_meta: dict[str, Any],
    t_start: float,
) -> None:
    payload = {
        "genome": 173,
        "name": "cross_arch_flop_cashout",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_model_id": TEACHER_HF,
        "results": results,
        "teacher_reference": teacher_reference,
        "data": data_meta,
        "c3_eval_meta": c3_meta,
        "teacher_cache": teacher_cache_meta,
        "elapsed_s": time.time() - t_start,
        "incremental": True,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("genome_173: preregistered cross-architecture KD cash-out scored in FLOPs")
    print(f"  teacher={TEACHER_HF}")
    print(f"  device={DEVICE} forward_dtype={FORWARD_DTYPE} recipient_dtype=torch.float32")
    print(f"  seeds={SEEDS} steps={TRAIN_STEPS} batch={TRAIN_BATCH_SIZE}")
    print(
        f"  train_windows={N_TRAIN_WINDOWS} c4_val_windows={N_C4_VAL_WINDOWS} "
        f"c3_examples_per_task={N_C3_PER_TASK} seq_len={SEQ_LEN}"
    )

    t_start = time.time()
    tok = load_tokenizer()
    vocab_size = len(tok)
    pad_token_id = int(tok.pad_token_id)
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
    c3_data, c3_meta = load_c3_validation(tok, n_per_task=N_C3_PER_TASK)

    data_meta = {
        "train": train_meta,
        "c4_val": c4_val_meta,
        "train_13gram_hash_count": len(train_hashes),
    }

    c4_eval_forward_tokens = int(c4_val_mask.sum().item())
    c3_eval_forward_tokens = int(
        sum(task_meta["forward_tokens"] for task_meta in c3_meta.values())
    )

    teacher, _ = load_trained_teacher(tok)
    teacher_param_count = sum(p.numel() for p in teacher.parameters())
    teacher_reference = {
        "param_count": int(teacher_param_count),
        "c4_val": evaluate_c4(teacher, c4_val_ids, c4_val_mask),
        "c3_val": evaluate_c3(teacher, c3_data, pad_token_id=pad_token_id),
    }
    print(
        f"  teacher c4_val: nll={teacher_reference['c4_val']['nll']:.4f} "
        f"top1={100.0 * teacher_reference['c4_val']['top1_acc']:.2f}%"
    )
    print(
        f"  teacher c3_macro: {100.0 * teacher_reference['c3_val']['C3_macro']:.2f}% "
        f"(HS={100.0 * teacher_reference['c3_val']['per_task']['hellaswag']:.2f} "
        f"PIQA={100.0 * teacher_reference['c3_val']['per_task']['piqa']:.2f} "
        f"Wino={100.0 * teacher_reference['c3_val']['per_task']['winogrande']:.2f})"
    )

    train_signature = tensor_sha1(train_ids)
    teacher_cache = precompute_teacher_topk_cache(
        teacher,
        train_ids,
        train_mask,
        train_signature=train_signature,
    )

    teacher_cache_flops = estimate_teacher_cache_flops(
        teacher_param_count,
        n_windows=int(train_ids.shape[0]),
        seq_len=int(train_ids.shape[1]),
    )

    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher_cache_meta = {
        "path": teacher_cache["path"],
        "cache_hit": teacher_cache["cache_hit"],
        "cache_source": teacher_cache["cache_source"],
        "estimated_bytes": teacher_cache["estimated_bytes"],
        "estimated_gib": teacher_cache["estimated_bytes"] / (1024 ** 3),
        "train_signature": train_signature,
        "teacher_cache_flops": int(teacher_cache_flops),
        "teacher_cache_pflops": float(teacher_cache_flops / 1e15),
        "teacher_param_count": int(teacher_param_count),
    }

    results = {arm.label: {} for arm in ARM_SPECS}
    print(f"\n=== Running {len(ARM_SPECS)} arms x {len(SEEDS)} seeds = {len(ARM_SPECS) * len(SEEDS)} cells ===")
    print("=== Pairing rule: same seed -> same batch schedule across scratch/KD and across architectures ===")

    for seed in SEEDS:
        train_schedule = build_train_schedule(seed, n_examples=int(train_ids.shape[0]))
        for arm in ARM_SPECS:
            print(f"\n--- arm={arm.label} seed={seed} ---")
            payload = train_one_arm(
                arm,
                seed=seed,
                vocab_size=vocab_size,
                pad_token_id=pad_token_id,
                train_ids=train_ids,
                train_mask=train_mask,
                c4_val_ids=c4_val_ids,
                c4_val_mask=c4_val_mask,
                c3_data=c3_data,
                c4_eval_forward_tokens=c4_eval_forward_tokens,
                c3_eval_forward_tokens=c3_eval_forward_tokens,
                train_schedule=train_schedule,
                teacher_cache=teacher_cache if arm.use_kd else None,
                teacher_cache_flops=teacher_cache_flops,
            )
            results[arm.label][str(seed)] = payload
            save_incremental(
                results=results,
                teacher_reference=teacher_reference,
                data_meta=data_meta,
                c3_meta=c3_meta,
                teacher_cache_meta=teacher_cache_meta,
                t_start=t_start,
            )

    summary = build_summary(
        results=results,
        teacher_reference=teacher_reference,
        teacher_cache_meta=teacher_cache_meta,
    )

    out = {
        "genome": 173,
        "name": "cross_arch_flop_cashout",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_model_id": TEACHER_HF,
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
            "n_c3_per_task": N_C3_PER_TASK,
            "kd_top_k": KD_TOPK,
            "kd_temp": KD_TEMP,
            "kd_gamma": KD_GAMMA,
            "c4_train_seed": C4_TRAIN_SEED,
            "c4_val_seed": C4_VAL_SEED,
            "c3_eval_seeds": {
                "hellaswag": HELLASWAG_SEED,
                "piqa": PIQA_SEED,
                "winogrande": WINOGRANDE_SEED,
            },
            "arm_specs": [
                {
                    "label": arm.label,
                    "use_kd": arm.use_kd,
                    "description": arm.description,
                    "student": asdict(arm.student),
                }
                for arm in ARM_SPECS
            ],
            "pass_criteria_locked_cycle42": {
                "llama_kd_retention_vs_qwen_kd_ge": PASS_MIN_RETENTION,
                "matched_compute_fraction_le": PASS_MAX_MATCHED_COMPUTE_FRACTION,
                "cross_arch_transfer_ratio_ge": PASS_MIN_TRANSFER_RATIO,
                "all_llama_seeds_positive_direction": True,
            },
            "fail_floor_locked_cycle42": {
                "llama_kd_retention_vs_qwen_kd_lt": FAIL_MIN_RETENTION,
                "cross_arch_transfer_ratio_lt": FAIL_MIN_TRANSFER_RATIO,
            },
            "compute_envelope": {
                "soft_hours": 3.5,
                "hard_hours": 4.0,
                "max_vram_gb": 22.0,
            },
            "notes": [
                "Teacher cache train signature intentionally matches genome_167 so cached top-k logits can be reused.",
                "Shared Qwen tokenizer means embedding / lm_head parameters dominate total student params; this is expected.",
                "Matched-capability compute divides standalone total FLOPs by retention_vs_qwen_kd.",
            ],
        },
        "data": data_meta,
        "c3_eval_meta": c3_meta,
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
