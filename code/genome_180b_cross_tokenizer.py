"""
genome_180b_cross_tokenizer.py

Cross-tokenizer lock test for the g180 Genome Forecast model.

The experiment keeps the recipient architecture fixed to a small Qwen3-style
decoder and swaps only the tokenizer/vocabulary interface. Sequence-level KD
is implemented by generating raw teacher text with Qwen3-0.6B, retokenizing the
text under each recipient tokenizer, and training with ordinary causal CE.

Outputs:
  - results/genome_180b_cross_tokenizer.json
  - results/cache/genome_180b_features/*.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_165_annealed_donor as g165
import genome_167_kd_canonical as g167
import genome_180_forecast as g180


OUT_PATH = ROOT / "results" / "genome_180b_cross_tokenizer.json"
G180_PATH = ROOT / "results" / "genome_180_forecast.json"
CACHE_DIR = ROOT / "results" / "cache" / "genome_180b_features"
RAW_POOL_CACHE = CACHE_DIR / "raw_c4_pools.json"
TEACHER_TEXT_CACHE = CACHE_DIR / "qwen3_teacher_texts.json"

PREREG_PATH = ROOT / "research" / "prereg" / "genome_180b_cross_tokenizer_2026-04-29.md"
DESIGN_GATE_PATH = ROOT / "codex_outputs" / "g180b_design_gate_20260429.md"

SEEDS = [42, 7, 13]
SEQ_LEN = 256
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
TRAIN_STEPS = 3600
TARGET_STEP = int(math.ceil(0.03 * TRAIN_STEPS))
SEQ_KD_LATE_START_STEP = 2401
LR = 3e-4
LR_WARMUP_STEPS = 200
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
LOG_EVERY = 200
EVAL_EVERY = 1200

N_TRAIN_WINDOWS = 8192
N_C4_VAL_WINDOWS = 1000
PROBE_WINDOWS = 16
PROBE_RAW_SEED = 180180
C4_TRAIN_SEED = 180201
C4_VAL_SEED = 180301

TEACHER_PREFIX_TOKENS = 64
TEACHER_NEW_TOKENS = 192
TEACHER_TEXT_COUNT = N_TRAIN_WINDOWS + 512
TEACHER_GENERATE_BATCH = 8

SMOKE_STEPS = 20
SMOKE_TRAIN_WINDOWS = 96
SMOKE_VAL_WINDOWS = 64
SMOKE_TEACHER_TEXTS = 96
SMOKE_PROJECT_LIMIT_S = 3.75 * 3600.0

BOOTSTRAP_N = 10_000
PASS_MSE_REDUCTION = 0.15
WEAK_PASS_MSE_REDUCTION = 0.10
ACTIONABLE_GAIN_NATS = g180.ACTIONABLE_GAIN_NATS
STOP_RECOMMEND_GAIN_THRESHOLD = g180.STOP_RECOMMEND_GAIN_THRESHOLD
RANDOM_STATE = 18020

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
class TokenizerSpec:
    label: str
    hf_id: str
    vocab_family: str
    eos_policy: str


@dataclass(frozen=True)
class ArmSpec:
    label: str
    description: str


@dataclass
class TokenizedPools:
    train_ids: torch.Tensor
    train_mask: torch.Tensor
    teacher_ids: torch.Tensor
    teacher_mask: torch.Tensor
    val_ids: torch.Tensor
    val_mask: torch.Tensor
    probe_batch: dict[str, Any]
    metadata: dict[str, Any]


TOKENIZER_SPECS = [
    TokenizerSpec(
        label="bert_wordpiece",
        hf_id="bert-base-uncased",
        vocab_family="WordPiece",
        eos_policy="use [SEP] as EOS, [PAD] as pad, no [CLS] injection",
    ),
    TokenizerSpec(
        label="t5_sentencepiece",
        hf_id="google-t5/t5-small",
        vocab_family="SentencePiece unigram",
        eos_policy="use </s> as EOS and <pad> as pad",
    ),
    TokenizerSpec(
        label="gpt2_bpe",
        hf_id="gpt2",
        vocab_family="GPT-2 BPE",
        eos_policy="use <|endoftext|> as EOS; pad is EOS (absent native pad)",
    ),
]

ARMS = [
    ArmSpec("scratch_ce", "Native-tokenizer C4 CE baseline."),
    ArmSpec("seq_kd_full", "Sequence-level teacher-text CE for all steps."),
    ArmSpec("seq_kd_late_only", "C4 CE through step 2400, teacher-text CE from step 2401."),
]

TOKENIZER_BY_LABEL = {spec.label: spec for spec in TOKENIZER_SPECS}
ARM_BY_LABEL = {arm.label: arm for arm in ARMS}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def print_flush(message: str = "") -> None:
    print(message, flush=True)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha1_texts(texts: Sequence[str], *, limit: int | None = None) -> str:
    h = hashlib.sha1()
    n = len(texts) if limit is None else min(len(texts), limit)
    for text in texts[:n]:
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def safe_id(*parts: Any) -> str:
    raw = "_".join(str(part) for part in parts)
    out = []
    for ch in raw:
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(out).strip("_")


def warmup_lr(step_zero_based: int, target_lr: float = LR, warmup_steps: int = LR_WARMUP_STEPS) -> float:
    if warmup_steps <= 0:
        return target_lr
    if step_zero_based < warmup_steps:
        return target_lr * float(step_zero_based + 1) / float(warmup_steps)
    return target_lr


def configure_tokenizer(spec: TokenizerSpec):
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=False, use_fast=True)
    tok.model_max_length = max(int(getattr(tok, "model_max_length", 0) or 0), SEQ_LEN)

    if spec.label == "bert_wordpiece":
        if tok.sep_token is None or tok.sep_token_id is None:
            raise RuntimeError("bert-base-uncased tokenizer has no [SEP] token")
        tok.eos_token = tok.sep_token
        tok.eos_token_id = tok.sep_token_id
        if tok.pad_token is None:
            tok.pad_token = "[PAD]"
    elif spec.label == "t5_sentencepiece":
        if tok.eos_token is None:
            tok.eos_token = "</s>"
        if tok.pad_token is None:
            tok.pad_token = "<pad>"
    elif spec.label == "gpt2_bpe":
        if tok.eos_token is None or tok.eos_token_id is None:
            raise RuntimeError("GPT-2 tokenizer has no EOS token")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
    else:
        raise ValueError(f"unknown tokenizer spec {spec.label}")

    if tok.pad_token_id is None:
        raise RuntimeError(f"{spec.label} has no pad token id after configuration")
    if tok.eos_token_id is None:
        raise RuntimeError(f"{spec.label} has no eos token id after configuration")
    return tok


def qwen3_recipient_config(tok) -> Qwen3Config:
    cfg = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=12,
        num_key_value_heads=6,
        intermediate_size=2048,
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        head_dim=64,
        rope_theta=10000.0,
        use_cache=False,
        bos_token_id=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    return cfg


def make_recipient(tok, seed: int) -> Qwen3ForCausalLM:
    set_seed(seed)
    model = Qwen3ForCausalLM(qwen3_recipient_config(tok))
    model.tie_weights()
    model.to(DEVICE)
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def param_count(model: torch.nn.Module) -> dict[str, int]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {"n_total_params": total, "n_trainable_params": trainable}


def load_raw_c4_pool(
    *,
    split: str,
    seed: int,
    min_docs: int,
    min_chars: int,
    label: str,
) -> dict[str, Any]:
    print_flush(
        f"  loading raw C4 pool {label}: split={split} seed={seed} "
        f"min_docs={min_docs} min_chars={min_chars}"
    )
    ds = load_dataset("allenai/c4", "en", split=split, streaming=True, trust_remote_code=False)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    texts: list[str] = []
    seen_prefixes: set[str] = set()
    total_chars = 0
    records_seen = 0

    for record in ds:
        records_seen += 1
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
        texts.append(text)
        total_chars += len(text)
        if len(texts) >= min_docs and total_chars >= min_chars:
            break

    if len(texts) < min_docs or total_chars < min_chars:
        raise RuntimeError(
            f"raw C4 pool {label} underfilled: docs={len(texts)}/{min_docs} "
            f"chars={total_chars}/{min_chars}"
        )

    return {
        "label": label,
        "dataset_name": "allenai/c4",
        "config_name": "en",
        "split": split,
        "seed": seed,
        "texts": texts,
        "n_texts": len(texts),
        "total_chars": total_chars,
        "records_seen": records_seen,
        "prefix_dedup_chars": 200,
        "sha1": sha1_texts(texts),
    }


def load_or_create_raw_pools(*, force: bool = False) -> dict[str, Any]:
    expected = {
        "train_seed": C4_TRAIN_SEED,
        "val_seed": C4_VAL_SEED,
        "probe_seed": PROBE_RAW_SEED,
        "seq_len": SEQ_LEN,
        "n_train_windows": N_TRAIN_WINDOWS,
        "n_c4_val_windows": N_C4_VAL_WINDOWS,
        "probe_windows": PROBE_WINDOWS,
        "teacher_text_count": TEACHER_TEXT_COUNT,
    }
    if not force and RAW_POOL_CACHE.exists():
        try:
            payload = read_json(RAW_POOL_CACHE)
            if payload.get("expected") == expected:
                print_flush(f"  raw C4 pool cache hit: {RAW_POOL_CACHE}")
                return payload
            print_flush("  raw C4 pool cache mismatch; rebuilding")
        except Exception as exc:
            print_flush(f"  raw C4 pool cache unreadable; rebuilding: {exc}")

    train_min_docs = max(TEACHER_TEXT_COUNT + 128, N_TRAIN_WINDOWS // 2)
    train_min_chars = int(N_TRAIN_WINDOWS * SEQ_LEN * 8)
    val_min_docs = max(1200, N_C4_VAL_WINDOWS // 2)
    val_min_chars = int(N_C4_VAL_WINDOWS * SEQ_LEN * 8)
    probe_min_docs = 64
    probe_min_chars = int(PROBE_WINDOWS * SEQ_LEN * 12)

    payload = {
        "genome": "180b",
        "created_utc": now_utc(),
        "expected": expected,
        "train": load_raw_c4_pool(
            split="train",
            seed=C4_TRAIN_SEED,
            min_docs=train_min_docs,
            min_chars=train_min_chars,
            label="train",
        ),
        "c4_val": load_raw_c4_pool(
            split="validation",
            seed=C4_VAL_SEED,
            min_docs=val_min_docs,
            min_chars=val_min_chars,
            label="c4_val",
        ),
        "probe": load_raw_c4_pool(
            split="validation",
            seed=PROBE_RAW_SEED,
            min_docs=probe_min_docs,
            min_chars=probe_min_chars,
            label="probe",
        ),
    }
    atomic_write_json(RAW_POOL_CACHE, payload)
    print_flush(f"  saved raw C4 pool cache: {RAW_POOL_CACHE}")
    return payload


def load_smoke_raw_pools() -> dict[str, Any]:
    return {
        "train": load_raw_c4_pool(
            split="train",
            seed=C4_TRAIN_SEED,
            min_docs=SMOKE_TEACHER_TEXTS + 16,
            min_chars=int(SMOKE_TRAIN_WINDOWS * SEQ_LEN * 8),
            label="smoke_train",
        ),
        "c4_val": load_raw_c4_pool(
            split="validation",
            seed=C4_VAL_SEED,
            min_docs=128,
            min_chars=int(SMOKE_VAL_WINDOWS * SEQ_LEN * 8),
            label="smoke_c4_val",
        ),
        "probe": load_raw_c4_pool(
            split="validation",
            seed=PROBE_RAW_SEED,
            min_docs=32,
            min_chars=int(PROBE_WINDOWS * SEQ_LEN * 8),
            label="smoke_probe",
        ),
    }


def tokenizer_to_ids(tok, text: str) -> list[int]:
    ids = tok(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    return [int(x) for x in ids]


def windows_from_texts(
    tok,
    texts: Sequence[str],
    *,
    n_windows: int,
    seq_len: int,
    forbidden_hashes: set[int] | None = None,
    source_label: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    eos_id = int(tok.eos_token_id)
    token_buffer: list[int] = []
    docs_used = 0
    tokens_total = 0

    for text in texts:
        ids = tokenizer_to_ids(tok, text)
        if not ids:
            continue
        token_buffer.extend(ids)
        token_buffer.append(eos_id)
        docs_used += 1
        tokens_total += len(ids) + 1

    windows: list[np.ndarray] = []
    overlap_rejects = 0
    cursor = 0
    while cursor + seq_len <= len(token_buffer) and len(windows) < n_windows:
        window = np.asarray(token_buffer[cursor : cursor + seq_len], dtype=np.int64)
        cursor += seq_len
        if forbidden_hashes is not None:
            row_hashes = g167.rolling_13gram_hashes(window)
            if any(int(h) in forbidden_hashes for h in row_hashes.tolist()):
                overlap_rejects += 1
                continue
        windows.append(window)

    if len(windows) < n_windows:
        raise RuntimeError(
            f"{source_label}: only built {len(windows)} / {n_windows} windows "
            f"from {docs_used} docs and {tokens_total} tokens for tokenizer={tok.name_or_path}"
        )

    input_ids = torch.tensor(np.stack(windows, axis=0), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    meta = {
        "source_label": source_label,
        "tokenizer_name_or_path": str(tok.name_or_path),
        "n_windows": int(input_ids.shape[0]),
        "seq_len": int(input_ids.shape[1]),
        "docs_used": docs_used,
        "tokens_total_with_eos": tokens_total,
        "sampling": "raw_pool_concat_eos_nonoverlapping_windows",
        "forbidden_overlap_rejects": overlap_rejects,
        "eos_token_id": int(tok.eos_token_id),
        "pad_token_id": int(tok.pad_token_id),
    }
    return input_ids, attention_mask, meta


def generate_teacher_texts(
    raw_train_texts: Sequence[str],
    *,
    n_texts: int,
    cache_path: Path | None,
    force: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    expected = {
        "teacher_model": g165._MODEL_ID,
        "raw_train_sha1_prefix": sha1_texts(raw_train_texts, limit=n_texts),
        "n_texts": n_texts,
        "prefix_tokens": TEACHER_PREFIX_TOKENS,
        "new_tokens": TEACHER_NEW_TOKENS,
        "generate_batch": TEACHER_GENERATE_BATCH,
        "do_sample": False,
    }
    if cache_path is not None and not force and cache_path.exists():
        try:
            payload = read_json(cache_path)
            if payload.get("expected") == expected and len(payload.get("texts", [])) >= n_texts:
                print_flush(f"  teacher text cache hit: {cache_path}")
                return list(payload["texts"][:n_texts]), payload["metadata"]
            print_flush("  teacher text cache mismatch; regenerating")
        except Exception as exc:
            print_flush(f"  teacher text cache unreadable; regenerating: {exc}")

    print_flush(f"  generating Qwen3 teacher texts: n={n_texts}")
    t0 = time.time()
    teacher, tok = g165.load_trained_donor()
    if hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = True
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    prefix_rows: list[list[int]] = []
    source_docs = 0
    for text in raw_train_texts:
        ids = tokenizer_to_ids(tok, text)
        if len(ids) < 8:
            continue
        prefix_rows.append(ids[:TEACHER_PREFIX_TOKENS])
        source_docs += 1
        if len(prefix_rows) >= n_texts:
            break
    if len(prefix_rows) < n_texts:
        raise RuntimeError(f"only got {len(prefix_rows)} teacher prefixes / {n_texts}")

    decoded: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(prefix_rows), TEACHER_GENERATE_BATCH):
            rows = prefix_rows[start : start + TEACHER_GENERATE_BATCH]
            max_len = max(len(row) for row in rows)
            input_ids = torch.full(
                (len(rows), max_len),
                int(tok.pad_token_id),
                dtype=torch.long,
                device=DEVICE,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_idx, row in enumerate(rows):
                row_tensor = torch.tensor(row, dtype=torch.long, device=DEVICE)
                input_ids[row_idx, -len(row) :] = row_tensor
                attention_mask[row_idx, -len(row) :] = 1
            with autocast_context():
                generated = teacher.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=TEACHER_NEW_TOKENS,
                    min_new_tokens=TEACHER_NEW_TOKENS,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    use_cache=True,
                )
            for row in generated.detach().cpu().tolist():
                text = tok.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                if text:
                    decoded.append(text)
            if start == 0 or (start // TEACHER_GENERATE_BATCH) % 50 == 0:
                print_flush(f"    teacher generated {len(decoded):5d}/{n_texts}")

    del teacher
    cleanup_cuda()
    if len(decoded) < n_texts:
        raise RuntimeError(f"teacher produced only {len(decoded)} non-empty texts / {n_texts}")

    metadata = {
        "expected": expected,
        "source_docs_scanned": source_docs,
        "elapsed_s": time.time() - t0,
        "texts_per_s": len(decoded) / max(time.time() - t0, 1e-9),
        "created_utc": now_utc(),
    }
    payload = {"expected": expected, "metadata": metadata, "texts": decoded[:n_texts]}
    if cache_path is not None:
        atomic_write_json(cache_path, payload)
        print_flush(f"  saved teacher text cache: {cache_path}")
    return decoded[:n_texts], metadata


def build_qwen_reference(raw_probe_texts: Sequence[str]) -> dict[str, Any]:
    print_flush("  building shared Qwen3 reference geometry")
    t0 = time.time()
    qwen_tok = g167.load_tokenizer()
    probe_ids, probe_mask, probe_meta = windows_from_texts(
        qwen_tok,
        raw_probe_texts,
        n_windows=PROBE_WINDOWS,
        seq_len=SEQ_LEN,
        source_label="qwen_reference_probe",
    )
    qwen_probe = {
        "input_ids": probe_ids,
        "attention_mask": probe_mask,
        "labels": probe_ids,
        "meta": probe_meta,
    }
    ref = g180._load_qwen_reference_geometry(qwen_probe, layer_indices=[1, 14, 28])
    ref["reference_probe_meta"] = probe_meta
    ref["reference_build_elapsed_s"] = time.time() - t0
    return ref


def build_tokenized_pools(
    spec: TokenizerSpec,
    tok,
    raw_pools: Mapping[str, Any],
    teacher_texts: Sequence[str],
    qwen_reference: Mapping[str, Any],
    *,
    n_train_windows: int = N_TRAIN_WINDOWS,
    n_val_windows: int = N_C4_VAL_WINDOWS,
    n_probe_windows: int = PROBE_WINDOWS,
) -> TokenizedPools:
    print_flush(f"  tokenizing pools for {spec.label} ({spec.hf_id})")
    train_ids, train_mask, train_meta = windows_from_texts(
        tok,
        raw_pools["train"]["texts"],
        n_windows=n_train_windows,
        seq_len=SEQ_LEN,
        source_label=f"{spec.label}_c4_train",
    )
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)

    val_ids, val_mask, val_meta = windows_from_texts(
        tok,
        raw_pools["c4_val"]["texts"],
        n_windows=n_val_windows,
        seq_len=SEQ_LEN,
        forbidden_hashes=train_hashes,
        source_label=f"{spec.label}_c4_val",
    )
    val_hashes = g167.collect_13gram_hashes(val_ids, val_mask)
    overlap = len(train_hashes.intersection(val_hashes))
    if overlap != 0:
        raise RuntimeError(f"{spec.label}: C4 train/val token 13-gram overlap detected: {overlap}")

    teacher_ids, teacher_mask, teacher_meta = windows_from_texts(
        tok,
        teacher_texts,
        n_windows=n_train_windows,
        seq_len=SEQ_LEN,
        forbidden_hashes=val_hashes,
        source_label=f"{spec.label}_qwen_teacher_text",
    )
    teacher_val_overlap = len(
        g167.collect_13gram_hashes(teacher_ids, teacher_mask).intersection(val_hashes)
    )
    if teacher_val_overlap != 0:
        raise RuntimeError(
            f"{spec.label}: teacher-text/val 13-gram overlap: {teacher_val_overlap}"
        )

    probe_ids, probe_mask, probe_meta = windows_from_texts(
        tok,
        raw_pools["probe"]["texts"],
        n_windows=n_probe_windows,
        seq_len=SEQ_LEN,
        source_label=f"{spec.label}_probe",
    )
    probe_batch = {
        "input_ids": probe_ids,
        "attention_mask": probe_mask,
        "labels": probe_ids,
        "meta": probe_meta,
        "reference_hidden": qwen_reference.get("reference_hidden"),
        "reference_embedding": qwen_reference.get("reference_embedding"),
        "reference_lm_head": qwen_reference.get("reference_lm_head"),
    }

    metadata = {
        "tokenizer": asdict(spec),
        "vocab_size": int(len(tok)),
        "special_tokens": {
            "eos_token": tok.eos_token,
            "eos_token_id": int(tok.eos_token_id),
            "pad_token": tok.pad_token,
            "pad_token_id": int(tok.pad_token_id),
        },
        "train": train_meta,
        "teacher_text": teacher_meta,
        "c4_val": val_meta,
        "probe": probe_meta,
        "train_13gram_hash_count": len(train_hashes),
        "val_13gram_hash_count": len(val_hashes),
        "train_val_13gram_overlap_count": overlap,
        "raw_pool_sha1": {
            "train": raw_pools["train"]["sha1"],
            "c4_val": raw_pools["c4_val"]["sha1"],
            "probe": raw_pools["probe"]["sha1"],
        },
    }
    return TokenizedPools(
        train_ids=train_ids,
        train_mask=train_mask,
        teacher_ids=teacher_ids,
        teacher_mask=teacher_mask,
        val_ids=val_ids,
        val_mask=val_mask,
        probe_batch=probe_batch,
        metadata=metadata,
    )


@torch.no_grad()
def evaluate_nll(model: torch.nn.Module, eval_ids: torch.Tensor, eval_mask: torch.Tensor) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_top1 = 0
    for start in range(0, int(eval_ids.shape[0]), EVAL_BATCH_SIZE):
        ids = eval_ids[start : start + EVAL_BATCH_SIZE].to(DEVICE)
        mask = eval_mask[start : start + EVAL_BATCH_SIZE].to(DEVICE)
        with autocast_context():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        shift_logits = logits[:, :-1].contiguous().float()
        shift_labels = ids[:, 1:].contiguous().clone()
        shift_mask = mask[:, 1:].contiguous().bool()
        labels_for_loss = shift_labels.clone()
        labels_for_loss[~shift_mask] = -100
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            labels_for_loss.view(-1),
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


def feature_cache_path(tokenizer_label: str, arm_label: str, seed: int) -> Path:
    cell_id = safe_id("g180b", tokenizer_label, arm_label, seed)
    digest = hashlib.sha1(cell_id.encode("utf-8")).hexdigest()[:10]
    return CACHE_DIR / f"{cell_id}_{digest}.json"


def load_feature_cache(tokenizer_label: str, arm_label: str, seed: int) -> dict[str, Any] | None:
    path = feature_cache_path(tokenizer_label, arm_label, seed)
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if not (
        payload.get("tokenizer_label") == tokenizer_label
        and payload.get("arm_label") == arm_label
        and int(payload.get("seed", -1)) == int(seed)
        and int(payload.get("target_step", -1)) == TARGET_STEP
        and payload.get("genome") == "180b"
    ):
        return None
    return payload


def write_feature_cache(
    tokenizer_label: str,
    arm_label: str,
    seed: int,
    payload: Mapping[str, Any],
) -> Path:
    path = feature_cache_path(tokenizer_label, arm_label, seed)
    atomic_write_json(path, payload)
    return path


def select_training_source(
    arm_label: str,
    step: int,
    pools: TokenizedPools,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if arm_label == "scratch_ce":
        return pools.train_ids, pools.train_mask, "c4"
    if arm_label == "seq_kd_full":
        return pools.teacher_ids, pools.teacher_mask, "teacher_text"
    if arm_label == "seq_kd_late_only":
        if step >= SEQ_KD_LATE_START_STEP:
            return pools.teacher_ids, pools.teacher_mask, "teacher_text"
        return pools.train_ids, pools.train_mask, "c4"
    raise ValueError(f"unknown arm {arm_label}")


def train_cell(
    *,
    tokenizer_spec: TokenizerSpec,
    tok,
    arm: ArmSpec,
    seed: int,
    pools: TokenizedPools,
    train_steps: int = TRAIN_STEPS,
    target_step: int = TARGET_STEP,
    write_cache: bool = True,
    force_feature: bool = False,
) -> tuple[dict[str, Any], dict[str, float]]:
    print_flush(f"\n--- tokenizer={tokenizer_spec.label} arm={arm.label} seed={seed} ---")
    set_seed(seed)
    t_cell = time.time()
    model = make_recipient(tok, seed)
    counts = param_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    n_examples = int(pools.train_ids.shape[0])
    n_teacher = int(pools.teacher_ids.shape[0])
    if n_examples != n_teacher:
        raise RuntimeError(f"train/teacher window count mismatch: {n_examples} vs {n_teacher}")
    rng = np.random.default_rng(seed)
    train_schedule = rng.integers(
        0,
        n_examples,
        size=(train_steps, TRAIN_BATCH_SIZE),
        dtype=np.int64,
    )

    train_log: list[dict[str, Any]] = []
    initial_metrics = evaluate_nll(model, pools.val_ids, pools.val_mask)
    print_flush(
        f"    params={counts['n_total_params'] / 1e6:.2f}M "
        f"step=0 c4_nll={initial_metrics['nll']:.4f}"
    )

    cached = None if force_feature else load_feature_cache(tokenizer_spec.label, arm.label, seed)
    features: dict[str, float] | None = None
    feature_cache = None
    if cached is not None:
        features = {str(k): float(v) for k, v in cached.get("features", {}).items() if v is not None}
        feature_cache = str(feature_cache_path(tokenizer_spec.label, arm.label, seed))

    timing = {
        "model_init_and_initial_eval_s": time.time() - t_cell,
        "feature_extract_s": 0.0,
        "final_eval_s": 0.0,
        "train_loop_s": 0.0,
    }
    t_train = time.time()
    early_loss = float("nan")
    last_source = "none"
    trajectory_checkpoints = {20, 40, 60, 80, target_step}
    trajectory_losses: dict[int, float] = {}
    model.train()

    try:
        for step in range(1, train_steps + 1):
            current_lr = warmup_lr(step - 1)
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            source_ids, source_mask, source_name = select_training_source(arm.label, step, pools)
            last_source = source_name
            batch_index_tensor = torch.as_tensor(train_schedule[step - 1], dtype=torch.long)
            ids = source_ids[batch_index_tensor].to(DEVICE)
            mask = source_mask[batch_index_tensor].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                ce_loss = g167.causal_ce_loss(logits, ids, mask)

            if not torch.isfinite(ce_loss):
                raise RuntimeError(
                    f"non-finite CE loss at step {step} tokenizer={tokenizer_spec.label} "
                    f"arm={arm.label} seed={seed}"
                )

            ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            early_loss = float(ce_loss.detach().float().cpu().item())
            if step in trajectory_checkpoints:
                trajectory_losses[step] = early_loss

            if step == target_step and features is None:
                t_feat = time.time()
                local_probe = dict(pools.probe_batch)
                local_probe["early_loss"] = early_loss
                features = g180.extract_features(model, local_probe, layer_indices=[1, 1 + 8 // 2, 8])
                timing["feature_extract_s"] = time.time() - t_feat
                cache_payload = {
                    "genome": "180b",
                    "timestamp_utc": now_utc(),
                    "tokenizer_label": tokenizer_spec.label,
                    "tokenizer_hf_id": tokenizer_spec.hf_id,
                    "arm_label": arm.label,
                    "seed": seed,
                    "target_step": target_step,
                    "features": features,
                    "probe_meta": pools.metadata.get("probe", {}),
                    "qwen_reference_model": g165._MODEL_ID,
                }
                if write_cache:
                    path = write_feature_cache(tokenizer_spec.label, arm.label, seed, cache_payload)
                    feature_cache = str(path)
                    print_flush(f"    feature cache saved: {path.name}")
                model.train()

            if step % LOG_EVERY == 0 or step == train_steps:
                row = {
                    "step": step,
                    "lr": float(current_lr),
                    "ce_loss": early_loss,
                    "source": source_name,
                    "elapsed_s": time.time() - t_train,
                }
                if step % EVAL_EVERY == 0 or step == train_steps:
                    row.update(evaluate_nll(model, pools.val_ids, pools.val_mask))
                    model.train()
                    print_flush(
                        f"    step={step:4d} src={source_name} ce={early_loss:.4f} "
                        f"c4_nll={row['nll']:.4f} ({row['elapsed_s']:.0f}s)"
                    )
                elif step % (LOG_EVERY * 2) == 0:
                    print_flush(
                        f"    step={step:4d} src={source_name} ce={early_loss:.4f} "
                        f"({row['elapsed_s']:.0f}s)"
                    )
                train_log.append(row)

        if features is None:
            t_feat = time.time()
            local_probe = dict(pools.probe_batch)
            local_probe["early_loss"] = early_loss
            features = g180.extract_features(model, local_probe, layer_indices=[1, 1 + 8 // 2, 8])
            timing["feature_extract_s"] = time.time() - t_feat
            if write_cache:
                path = write_feature_cache(
                    tokenizer_spec.label,
                    arm.label,
                    seed,
                    {
                        "genome": "180b",
                        "timestamp_utc": now_utc(),
                        "tokenizer_label": tokenizer_spec.label,
                        "tokenizer_hf_id": tokenizer_spec.hf_id,
                        "arm_label": arm.label,
                        "seed": seed,
                        "target_step": target_step,
                        "features": features,
                        "probe_meta": pools.metadata.get("probe", {}),
                        "qwen_reference_model": g165._MODEL_ID,
                    },
                )
                feature_cache = str(path)

        timing["train_loop_s"] = time.time() - t_train
        t_eval = time.time()
        final_metrics = evaluate_nll(model, pools.val_ids, pools.val_mask)
        timing["final_eval_s"] = time.time() - t_eval
        wallclock_s = time.time() - t_cell

        result = {
            "seed": seed,
            "tokenizer_label": tokenizer_spec.label,
            "tokenizer_hf_id": tokenizer_spec.hf_id,
            "vocab_family": tokenizer_spec.vocab_family,
            "arm_label": arm.label,
            "description": arm.description,
            "train_steps": train_steps,
            "target_step": target_step,
            "seq_len": SEQ_LEN,
            "batch_size": TRAIN_BATCH_SIZE,
            "lr": LR,
            "lr_warmup_steps": LR_WARMUP_STEPS,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "initial_metrics": initial_metrics,
            "early_loss": float(features.get("early_loss", early_loss)),
            "final_metrics": final_metrics,
            "final_nll": float(final_metrics["nll"]),
            "final_top1_acc": float(final_metrics["top1_acc"]),
            "last_training_source": last_source,
            "train_log": train_log,
            "trajectory_losses": trajectory_losses,
            "feature_cache_path": feature_cache,
            "n_total_params": counts["n_total_params"],
            "n_trainable_params": counts["n_trainable_params"],
            "timing": timing,
            "wallclock_s": wallclock_s,
        }
        return result, features
    finally:
        del model, optimizer
        cleanup_cuda()


def base_payload() -> dict[str, Any]:
    return {
        "genome": "180b",
        "name": "cross_tokenizer_forecast",
        "timestamp_utc_started": now_utc(),
        "prereg_path": str(PREREG_PATH),
        "design_gate_path": str(DESIGN_GATE_PATH),
        "device": DEVICE,
        "forward_dtype": str(FORWARD_DTYPE),
        "config": {
            "tokenizers": [asdict(spec) for spec in TOKENIZER_SPECS],
            "arms": [asdict(arm) for arm in ARMS],
            "seeds": SEEDS,
            "train_steps": TRAIN_STEPS,
            "target_step": TARGET_STEP,
            "seq_kd_late_start_step": SEQ_KD_LATE_START_STEP,
            "seq_len": SEQ_LEN,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "lr": LR,
            "lr_warmup_steps": LR_WARMUP_STEPS,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "n_train_windows": N_TRAIN_WINDOWS,
            "n_c4_val_windows": N_C4_VAL_WINDOWS,
            "probe_windows": PROBE_WINDOWS,
            "probe_raw_seed": PROBE_RAW_SEED,
            "c4_train_seed": C4_TRAIN_SEED,
            "c4_val_seed": C4_VAL_SEED,
            "teacher": {
                "model": g165._MODEL_ID,
                "prefix_tokens": TEACHER_PREFIX_TOKENS,
                "new_tokens": TEACHER_NEW_TOKENS,
                "text_count": TEACHER_TEXT_COUNT,
                "generate_batch": TEACHER_GENERATE_BATCH,
                "do_sample": False,
            },
            "qwen3_recipient_config": {
                "hidden_size": 768,
                "num_hidden_layers": 8,
                "num_attention_heads": 12,
                "num_key_value_heads": 6,
                "intermediate_size": 2048,
                "tie_word_embeddings": True,
                "vocab_size": "len(tokenizer)",
            },
            "windows_cuda_rules": {
                "num_workers": 0,
                "pin_memory": False,
                "sklearn_n_jobs": 1,
            },
            "smoke_project_limit_s": SMOKE_PROJECT_LIMIT_S,
        },
        "smoke_test": {},
        "data": {},
        "results": {spec.label: {arm.label: {} for arm in ARMS} for spec in TOKENIZER_SPECS},
        "rows": [],
        "analysis": {},
        "summary": {"status": "INCOMPLETE", "verdict": "INCOMPLETE: no cells complete."},
        "verdict": "INCOMPLETE",
    }


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("results", {})
    for spec in TOKENIZER_SPECS:
        payload["results"].setdefault(spec.label, {})
        for arm in ARMS:
            payload["results"][spec.label].setdefault(arm.label, {})
    payload.setdefault("rows", [])
    payload.setdefault("analysis", {})
    payload.setdefault("summary", {"status": "INCOMPLETE"})
    payload.setdefault("verdict", "INCOMPLETE")
    payload.setdefault("smoke_test", {})
    payload.setdefault("data", {})
    return payload


def load_or_create_payload(*, resume: bool) -> dict[str, Any]:
    if resume and OUT_PATH.exists():
        try:
            return normalize_payload(read_json(OUT_PATH))
        except Exception as exc:
            raise RuntimeError(f"failed to read existing {OUT_PATH}: {exc}") from exc
    return base_payload()


def write_payload(payload: dict[str, Any], *, t_start: float, incremental: bool) -> None:
    payload["timestamp_utc_last_write"] = now_utc()
    payload["elapsed_s"] = time.time() - t_start
    payload["incremental"] = bool(incremental)
    atomic_write_json(OUT_PATH, payload)


def cell_result(payload: Mapping[str, Any], tokenizer_label: str, arm_label: str, seed: int) -> dict[str, Any] | None:
    raw = (
        payload.get("results", {})
        .get(tokenizer_label, {})
        .get(arm_label, {})
        .get(str(seed))
    )
    return raw if isinstance(raw, dict) else None


def cell_done(payload: Mapping[str, Any], tokenizer_label: str, arm_label: str, seed: int) -> bool:
    result = cell_result(payload, tokenizer_label, arm_label, seed)
    if not result or "final_nll" not in result:
        return False
    cached = load_feature_cache(tokenizer_label, arm_label, seed)
    return cached is not None


def all_cells_done(payload: Mapping[str, Any]) -> bool:
    return all(
        cell_done(payload, spec.label, arm.label, seed)
        for spec in TOKENIZER_SPECS
        for seed in SEEDS
        for arm in ARMS
    )


def g180_train_rows_and_models() -> dict[str, Any]:
    if not G180_PATH.exists():
        raise FileNotFoundError(f"missing required g180 result JSON: {G180_PATH}")
    g180_payload = read_json(G180_PATH)
    rows = list(g180_payload.get("rows", []))
    train_rows = [row for row in rows if row.get("split") == "train"]
    if len(train_rows) < 4:
        raise RuntimeError(f"not enough g180 train rows to fit frozen forecasters: {len(train_rows)}")
    y_train = [float(row["label"]) for row in train_rows]
    baseline = g180.fit_baseline(train_rows, y_train)
    full = g180.fit_full(train_rows, y_train)
    return {
        "g180_payload": g180_payload,
        "train_rows": train_rows,
        "baseline": baseline,
        "full": full,
    }


def serialize_ridge(model: Any) -> dict[str, Any]:
    return {
        "alpha": float(getattr(model, "alpha", 1.0)),
        "feature_names": list(getattr(model, "_forecast_feature_names")),
        "medians": np.asarray(getattr(model, "_forecast_medians"), dtype=np.float64).tolist(),
        "mean": np.asarray(getattr(model, "_forecast_mean"), dtype=np.float64).tolist(),
        "scale": np.asarray(getattr(model, "_forecast_scale"), dtype=np.float64).tolist(),
        "coef": np.asarray(getattr(model, "coef_"), dtype=np.float64).tolist(),
        "intercept": float(getattr(model, "intercept_")),
    }


def load_features_for_row(tokenizer_label: str, arm_label: str, seed: int) -> dict[str, float] | None:
    cached = load_feature_cache(tokenizer_label, arm_label, seed)
    if cached is None:
        return None
    features = cached.get("features")
    if not isinstance(features, dict):
        return None
    out: dict[str, float] = {}
    for key, value in features.items():
        if value is None:
            out[str(key)] = float("nan")
        else:
            out[str(key)] = float(value)
    return out


def build_completed_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in TOKENIZER_SPECS:
        for seed in SEEDS:
            scratch = cell_result(payload, spec.label, "scratch_ce", seed)
            if scratch is None or "final_nll" not in scratch:
                continue
            scratch_nll = float(scratch["final_nll"])
            for arm in ARMS:
                result = cell_result(payload, spec.label, arm.label, seed)
                if result is None or "final_nll" not in result:
                    continue
                features = load_features_for_row(spec.label, arm.label, seed)
                if features is None:
                    continue
                final_nll = float(result["final_nll"])
                label = 0.0 if arm.label == "scratch_ce" else scratch_nll - final_nll
                row = {
                    "cell_id": safe_id("g180b", spec.label, arm.label, seed),
                    "source": "g180b",
                    "arm": arm.label,
                    "seed": seed,
                    "tokenizer": spec.label,
                    "tokenizer_hf_id": spec.hf_id,
                    "family": spec.label,
                    "split": "test",
                    "protocol": "cross_tokenizer_seq_kd",
                    "target_steps": TARGET_STEP,
                    "final_steps": TRAIN_STEPS,
                    "label": float(label),
                    "scratch_final_nll": scratch_nll,
                    "final_nll": final_nll,
                    "feature_source": "cache",
                }
                row.update(features)
                rows.append(row)
    return rows


def mse(y: np.ndarray, pred: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    return float(np.mean((y - pred) ** 2))


def paired_bootstrap_improvement(y: np.ndarray, pred_base: np.ndarray, pred_full: np.ndarray) -> dict[str, Any]:
    if y.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "p_gt_0": float("nan")}
    rng = np.random.default_rng(RANDOM_STATE)
    values = np.empty(BOOTSTRAP_N, dtype=np.float64)
    for idx in range(BOOTSTRAP_N):
        sample = rng.integers(0, y.size, size=y.size)
        values[idx] = np.mean((y[sample] - pred_base[sample]) ** 2) - np.mean(
            (y[sample] - pred_full[sample]) ** 2
        )
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "p_gt_0": float(np.mean(values > 0.0)),
    }


def auroc_bad_run(y: np.ndarray, pred_full: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        labels = (y <= 0.0).astype(np.int64)
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, -pred_full))
    except Exception:
        return float("nan")


def reduced_no_interface_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    drop_prefixes = ("embed_to_qwen_ref_", "lm_head_to_qwen_ref_")
    drop_keys = {"embed_reference_rows_used", "lm_head_reference_rows_used"}
    reduced = []
    for row in rows:
        reduced.append(
            {
                key: value
                for key, value in row.items()
                if key not in drop_keys and not any(key.startswith(prefix) for prefix in drop_prefixes)
            }
        )
    return reduced


def exploratory_no_interface_eval(
    g180_state: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    pred_base: np.ndarray,
) -> dict[str, Any]:
    if not rows:
        return {}
    train_rows = list(g180_state["train_rows"])
    train_reduced = reduced_no_interface_rows(train_rows)
    test_reduced = reduced_no_interface_rows(rows)
    y_train = [float(row["label"]) for row in train_rows]
    try:
        model = g180._fit_ridge(train_reduced, y_train, early_loss_only=False)
        pred = g180._predict(model, test_reduced)
        reduced_mse = mse(y, pred)
        base_mse = mse(y, pred_base)
        return {
            "note": "Exploratory only; embedding/lm_head Qwen-reference features removed and Ridge refit on g180 train rows.",
            "mse": reduced_mse,
            "mse_reduction_vs_early_loss": (
                (base_mse - reduced_mse) / base_mse if base_mse > 1e-12 else float("nan")
            ),
            "feature_names": list(getattr(model, "_forecast_feature_names")),
            "predictions": pred.tolist(),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


_ROW_META_KEYS = frozenset({
    "cell_id", "source", "arm", "seed", "tokenizer", "tokenizer_hf_id",
    "family", "split", "protocol", "target_steps", "final_steps",
    "label", "scratch_final_nll", "final_nll", "feature_source",
    "pred_early_loss_only", "pred_geometry_plus_early_loss",
    "residual_geometry_plus_early_loss",
})
_EARLY_LOSS_KEY = "early_loss_at_target_step"
_PERM_N = 1000


def shuffled_geometry_control(
    g180_state: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    y: np.ndarray,
) -> dict[str, Any]:
    if len(rows) < 4:
        return {"note": "too few rows for permutation test"}
    rng = np.random.default_rng(RANDOM_STATE)
    real_pred = g180._predict(g180_state["full"], rows)
    real_mse_val = float(np.mean((y - real_pred) ** 2))
    geom_keys = [k for k in rows[0] if k not in _ROW_META_KEYS and k != _EARLY_LOSS_KEY]
    perm_mses: list[float] = []
    for _ in range(_PERM_N):
        perm_rows = [dict(row) for row in rows]
        perm_idx = rng.permutation(len(rows)).tolist()
        for fk in geom_keys:
            vals = [rows[i][fk] for i in perm_idx]
            for j, row in enumerate(perm_rows):
                row[fk] = vals[j]
        try:
            pf = g180._predict(g180_state["full"], perm_rows)
            perm_mses.append(float(np.mean((y - pf) ** 2)))
        except Exception:
            continue
    if not perm_mses:
        return {"error": "all permutations failed"}
    arr = np.asarray(perm_mses, dtype=np.float64)
    return {
        "note": "Permute geometry features (not early_loss) across rows; repredict with frozen Ridge. p < 0.05 means real geometry ordering is informative.",
        "real_mse": real_mse_val,
        "n_permutations": len(perm_mses),
        "permuted_mse_mean": float(arr.mean()),
        "permuted_mse_std": float(arr.std()),
        "permuted_mse_p05": float(np.percentile(arr, 5)),
        "permuted_mse_p95": float(np.percentile(arr, 95)),
        "p_value_real_le_permuted": float(np.mean(arr <= real_mse_val)),
        "geometry_feature_keys_permuted": geom_keys,
    }


def refresh_analysis(payload: dict[str, Any], g180_state: Mapping[str, Any]) -> None:
    rows = build_completed_rows(payload)
    payload["rows"] = rows
    total_expected = len(TOKENIZER_SPECS) * len(ARMS) * len(SEEDS)
    completed = len(rows)

    model_meta = {
        "source": str(G180_PATH),
        "rule": "refit Ridge on original g180 train rows only, then freeze for all g180b rows",
        "g180_train_rows": len(g180_state["train_rows"]),
        "baseline_model": serialize_ridge(g180_state["baseline"]),
        "full_model": serialize_ridge(g180_state["full"]),
    }

    if not rows:
        payload["analysis"] = {"frozen_g180": model_meta}
        payload["summary"] = {
            "status": "INCOMPLETE",
            "verdict": "INCOMPLETE: no scored g180b rows yet.",
            "counts": {"completed_rows": 0, "expected_rows": total_expected},
        }
        payload["verdict"] = "INCOMPLETE"
        return

    y = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    pred_base = g180._predict(g180_state["baseline"], rows)
    pred_full = g180._predict(g180_state["full"], rows)
    for row, pb, pf in zip(rows, pred_base, pred_full):
        row["pred_early_loss_only"] = float(pb)
        row["pred_geometry_plus_early_loss"] = float(pf)
        row["residual_geometry_plus_early_loss"] = float(float(row["label"]) - pf)

    base_mse = mse(y, pred_base)
    full_mse = mse(y, pred_full)
    mse_reduction = (base_mse - full_mse) / base_mse if base_mse > 1e-12 else float("nan")
    paired = paired_bootstrap_improvement(y, pred_base, pred_full)

    false_stop_rows = []
    for row, pred, truth in zip(rows, pred_full, y):
        recommend_stop = bool(pred < STOP_RECOMMEND_GAIN_THRESHOLD)
        if recommend_stop and truth >= ACTIONABLE_GAIN_NATS:
            false_stop_rows.append(
                {
                    "cell_id": row["cell_id"],
                    "tokenizer": row["tokenizer"],
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "predicted_gain": float(pred),
                    "true_gain": float(truth),
                }
            )

    per_tokenizer = {}
    for spec in TOKENIZER_SPECS:
        idxs = [idx for idx, row in enumerate(rows) if row["tokenizer"] == spec.label]
        if not idxs:
            continue
        yy = y[idxs]
        pb = pred_base[idxs]
        pf = pred_full[idxs]
        b = mse(yy, pb)
        f = mse(yy, pf)
        per_tokenizer[spec.label] = {
            "n": len(idxs),
            "mse_early_loss_only": b,
            "mse_geometry_plus_early_loss": f,
            "mse_reduction": (b - f) / b if b > 1e-12 else float("nan"),
            "residual_mean_full": float(np.mean(yy - pf)),
            "truth_mean": float(np.mean(yy)),
            "prediction_mean_full": float(np.mean(pf)),
        }

    ci_lo = float(paired["ci95"][0])
    no_false_stop = len(false_stop_rows) == 0
    all_complete = completed == total_expected
    if not all_complete:
        status = "INCOMPLETE"
        verdict = f"INCOMPLETE: {completed} / {total_expected} scored rows complete."
    elif mse_reduction >= PASS_MSE_REDUCTION and ci_lo > 0.0 and no_false_stop:
        status = "PASS"
        verdict = (
            "PASS: frozen g180 geometry+early-loss Ridge beats frozen early-loss-only "
            f"by {100.0 * mse_reduction:.1f}% MSE reduction with paired CI above 0."
        )
    elif mse_reduction >= WEAK_PASS_MSE_REDUCTION and ci_lo > 0.0 and no_false_stop:
        status = "WEAK_PASS"
        verdict = (
            "WEAK_PASS: frozen g180 geometry helps on held-out tokenizers but clears only "
            f"{100.0 * mse_reduction:.1f}% MSE reduction."
        )
    else:
        status = "FAIL"
        if false_stop_rows:
            reason = "false stop on actionable high-gain arm"
        elif not (ci_lo > 0.0):
            reason = "paired bootstrap CI crosses zero"
        else:
            reason = f"MSE reduction below {100.0 * WEAK_PASS_MSE_REDUCTION:.0f}%"
        verdict = f"FAIL: {reason}."

    payload["analysis"] = {
        "frozen_g180": model_meta,
        "primary": {
            "n": int(y.size),
            "mse_early_loss_only": base_mse,
            "mse_geometry_plus_early_loss": full_mse,
            "mse_reduction": float(mse_reduction),
            "paired_bootstrap_mse_improvement": paired,
            "predictions_early_loss_only": pred_base.tolist(),
            "predictions_geometry_plus_early_loss": pred_full.tolist(),
            "truth": y.tolist(),
            "residuals_geometry_plus_early_loss": (y - pred_full).tolist(),
        },
        "secondary": {
            "per_tokenizer": per_tokenizer,
            "auroc_bad_run_stop_score": auroc_bad_run(y, pred_full),
            "no_interface_feature_exploratory": exploratory_no_interface_eval(
                g180_state,
                rows,
                y,
                pred_base,
            ),
            "shuffled_geometry_permutation_test": shuffled_geometry_control(
                g180_state, rows, y
            ),
        },
    }
    payload["summary"] = {
        "status": status,
        "verdict": verdict,
        "counts": {
            "completed_rows": completed,
            "expected_rows": total_expected,
            "tokenizers": len(TOKENIZER_SPECS),
            "arms": len(ARMS),
            "seeds": len(SEEDS),
        },
        "criteria": {
            "pass_mse_reduction_ge": PASS_MSE_REDUCTION,
            "weak_pass_mse_reduction_ge": WEAK_PASS_MSE_REDUCTION,
            "observed_mse_reduction": float(mse_reduction),
            "paired_bootstrap_improvement_ci95": paired["ci95"],
            "paired_bootstrap_ci_low_gt_0": bool(ci_lo > 0.0) if math.isfinite(ci_lo) else False,
            "stop_threshold_predicted_gain_lt": STOP_RECOMMEND_GAIN_THRESHOLD,
            "actionable_true_gain_ge": ACTIONABLE_GAIN_NATS,
            "no_false_stop_on_actionable_gain": no_false_stop,
            "all_rows_complete": all_complete,
        },
        "false_stop_rows": false_stop_rows,
    }
    payload["verdict"] = status


def run_smoke_test(*, force: bool = False) -> dict[str, Any]:
    print_flush("\n=== genome_180b smoke test: 1 tokenizer x 1 seed x 20 steps ===")
    t0 = time.time()
    raw = load_smoke_raw_pools()
    spec = TOKENIZER_SPECS[0]
    tok = configure_tokenizer(spec)

    t_teacher = time.time()
    teacher_texts, teacher_meta = generate_teacher_texts(
        raw["train"]["texts"],
        n_texts=SMOKE_TEACHER_TEXTS,
        cache_path=None,
        force=force,
    )
    teacher_elapsed = time.time() - t_teacher

    t_ref = time.time()
    qwen_ref = build_qwen_reference(raw["probe"]["texts"])
    reference_elapsed = time.time() - t_ref

    t_tok = time.time()
    pools = build_tokenized_pools(
        spec,
        tok,
        raw,
        teacher_texts,
        qwen_ref,
        n_train_windows=SMOKE_TRAIN_WINDOWS,
        n_val_windows=SMOKE_VAL_WINDOWS,
        n_probe_windows=PROBE_WINDOWS,
    )
    tokenization_elapsed = time.time() - t_tok

    result, features = train_cell(
        tokenizer_spec=spec,
        tok=tok,
        arm=ARM_BY_LABEL["seq_kd_full"],
        seed=SEEDS[0],
        pools=pools,
        train_steps=SMOKE_STEPS,
        target_step=SMOKE_STEPS,
        write_cache=False,
        force_feature=True,
    )
    elapsed = time.time() - t0
    pure_train_loop_s = max(
        0.0,
        result["timing"]["train_loop_s"] - result["timing"]["feature_extract_s"],
    )
    train_step_s = pure_train_loop_s / max(SMOKE_STEPS, 1)
    teacher_per_text_s = teacher_elapsed / max(SMOKE_TEACHER_TEXTS, 1)
    projected_s = (
        teacher_per_text_s * TEACHER_TEXT_COUNT
        + reference_elapsed
        + tokenization_elapsed * (len(TOKENIZER_SPECS) * N_TRAIN_WINDOWS / max(SMOKE_TRAIN_WINDOWS, 1))
        + len(TOKENIZER_SPECS)
        * len(ARMS)
        * len(SEEDS)
        * (
            result["timing"]["model_init_and_initial_eval_s"]
            + TRAIN_STEPS * train_step_s
            + result["timing"]["feature_extract_s"]
            + result["timing"]["final_eval_s"]
        )
    )
    passed = bool(projected_s <= SMOKE_PROJECT_LIMIT_S)
    smoke = {
        "timestamp_utc": now_utc(),
        "tokenizer": spec.label,
        "arm": "seq_kd_full",
        "seed": SEEDS[0],
        "steps": SMOKE_STEPS,
        "elapsed_s": elapsed,
        "teacher_elapsed_s": teacher_elapsed,
        "reference_elapsed_s": reference_elapsed,
        "tokenization_elapsed_s": tokenization_elapsed,
        "pure_train_loop_s": pure_train_loop_s,
        "train_step_s": train_step_s,
        "feature_extract_s": result["timing"]["feature_extract_s"],
        "final_eval_s": result["timing"]["final_eval_s"],
        "projected_full_run_s": projected_s,
        "projected_full_run_h": projected_s / 3600.0,
        "limit_s": SMOKE_PROJECT_LIMIT_S,
        "passed": passed,
        "teacher_meta": teacher_meta,
        "smoke_final_nll": result["final_nll"],
        "feature_names": sorted(features.keys()),
    }
    print_flush(
        f"=== smoke projected full run: {projected_s / 3600.0:.2f}h "
        f"(limit {SMOKE_PROJECT_LIMIT_S / 3600.0:.2f}h) passed={passed} ==="
    )
    return smoke


def selected_specs(args: argparse.Namespace) -> tuple[list[TokenizerSpec], list[ArmSpec], list[int]]:
    tokenizers = TOKENIZER_SPECS
    arms = ARMS
    seeds = SEEDS
    if args.limit_tokenizers is not None:
        tokenizers = tokenizers[: args.limit_tokenizers]
    if args.limit_arms is not None:
        arms = arms[: args.limit_arms]
    if args.limit_seeds is not None:
        seeds = seeds[: args.limit_seeds]
    return list(tokenizers), list(arms), list(seeds)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    print_flush("genome_180b: cross-tokenizer forecast lock test")
    print_flush(f"  device={DEVICE} forward_dtype={FORWARD_DTYPE}")
    print_flush(f"  output={OUT_PATH}")
    t_start = time.time()
    payload = load_or_create_payload(resume=not args.no_resume)
    g180_state = g180_train_rows_and_models()

    smoke = payload.get("smoke_test", {})
    if args.force_smoke or not smoke.get("passed"):
        smoke = run_smoke_test(force=args.force_smoke)
        payload["smoke_test"] = smoke
        refresh_analysis(payload, g180_state)
        write_payload(payload, t_start=t_start, incremental=True)
        if not smoke["passed"]:
            payload["verdict"] = "ABORTED_SMOKE"
            payload["summary"] = {
                "status": "ABORTED_SMOKE",
                "verdict": (
                    "ABORTED_SMOKE: 20-step smoke projection exceeds the locked "
                    "3.75 hour wall-clock cap."
                ),
                "smoke_test": smoke,
            }
            write_payload(payload, t_start=t_start, incremental=False)
            raise SystemExit(2)
    else:
        print_flush("  smoke test already passed in existing payload; resume continues")

    if args.smoke_only:
        print_flush("  smoke-only requested; stopping after smoke test")
        return payload

    if all_cells_done(payload) and not args.force_rerun:
        print_flush("  all cells already complete; refreshing analysis only")
        refresh_analysis(payload, g180_state)
        write_payload(payload, t_start=t_start, incremental=False)
        return payload

    raw_pools = load_or_create_raw_pools(force=args.force_data)
    payload["data"]["raw_pools"] = {
        key: {k: v for k, v in value.items() if k != "texts"}
        for key, value in raw_pools.items()
        if isinstance(value, dict)
    }

    teacher_texts, teacher_meta = generate_teacher_texts(
        raw_pools["train"]["texts"],
        n_texts=TEACHER_TEXT_COUNT,
        cache_path=TEACHER_TEXT_CACHE,
        force=args.force_teacher,
    )
    payload["data"]["teacher_texts"] = teacher_meta

    qwen_reference = build_qwen_reference(raw_pools["probe"]["texts"])
    payload["data"]["qwen_reference"] = {
        key: value
        for key, value in qwen_reference.items()
        if key not in {"reference_hidden", "reference_embedding", "reference_lm_head"}
    }

    tokenizers_to_run, arms_to_run, seeds_to_run = selected_specs(args)
    for tokenizer_spec in tokenizers_to_run:
        tok = configure_tokenizer(tokenizer_spec)
        pools = build_tokenized_pools(tokenizer_spec, tok, raw_pools, teacher_texts, qwen_reference)
        payload["data"].setdefault("tokenized_pools", {})[tokenizer_spec.label] = pools.metadata
        refresh_analysis(payload, g180_state)
        write_payload(payload, t_start=t_start, incremental=True)

        for seed in seeds_to_run:
            for arm in arms_to_run:
                if not args.force_rerun and cell_done(payload, tokenizer_spec.label, arm.label, seed):
                    print_flush(f"  skip complete cell {tokenizer_spec.label}/{arm.label}/seed{seed}")
                    continue
                result, _ = train_cell(
                    tokenizer_spec=tokenizer_spec,
                    tok=tok,
                    arm=arm,
                    seed=seed,
                    pools=pools,
                    train_steps=TRAIN_STEPS,
                    target_step=TARGET_STEP,
                    write_cache=True,
                    force_feature=args.force_features,
                )
                payload["results"][tokenizer_spec.label][arm.label][str(seed)] = result
                refresh_analysis(payload, g180_state)
                write_payload(payload, t_start=t_start, incremental=True)

        del pools, tok
        cleanup_cuda()

    refresh_analysis(payload, g180_state)
    write_payload(payload, t_start=t_start, incremental=False)
    print_flush(f"\n=== verdict: {payload['summary'].get('verdict', payload['verdict'])} ===")
    print_flush(f"Saved: {OUT_PATH} ({payload['elapsed_s']:.1f}s)")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genome 180b cross-tokenizer forecast experiment.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore an existing result JSON.")
    parser.add_argument("--force-rerun", action="store_true", help="Rerun cells even if result and feature cache exist.")
    parser.add_argument("--force-features", action="store_true", help="Recompute per-cell features during rerun.")
    parser.add_argument("--force-data", action="store_true", help="Rebuild raw C4 text pools.")
    parser.add_argument("--force-teacher", action="store_true", help="Regenerate Qwen3 teacher text cache.")
    parser.add_argument("--force-smoke", action="store_true", help="Run smoke projection even if payload has a pass.")
    parser.add_argument("--smoke-only", action="store_true", help="Run the required smoke test and stop.")
    parser.add_argument("--limit-tokenizers", type=int, default=None, help="Debug limit; default runs all 3.")
    parser.add_argument("--limit-arms", type=int, default=None, help="Debug limit; default runs all 3.")
    parser.add_argument("--limit-seeds", type=int, default=None, help="Debug limit; default runs all 3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
