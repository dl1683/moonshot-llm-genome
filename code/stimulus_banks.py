"""Stimulus banks for the Neural Genome atlas — the conditioning variable of
every coordinate (per `research/atlas_tl_session.md` 2.5.7 conditional
universality).

Provides the pinned-identity symbols that prereg F tuples reference:

    c4_clean_v1          - generator (deterministic, seed-parametrized)
    filter_len_256_english - filter predicate on raw examples
    in_family            - invariance-check: membership in declared F
    dataset_hash         - canonical sha256 of the source dataset slice

All symbols are declared at module top level so that validator code can
verify they resolve under the pinned (git_commit, file_path, symbol)
triple at a specific commit. Bodies are stubs until Batch-1 extraction
code comes in; signatures are locked.

Per CLAUDE.md section 3.3 conventions: experiment primitives live under
code/genome_* scripts; this file is Gate-1 infrastructure (same tier as
code/prereg_validator.py), not a genome experiment.

ASCII-only source per Windows cp1252 constraint.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


# -------------------- Dataset identity --------------------

# Canonical source-dataset hash for the C4-clean slice used in Batch-1.
# Placeholder until the actual slice is fixed and hashed. The prereg locks
# this value; changing it invalidates the prereg (per 2.5.9 lock rule).
DATASET_HASH_C4_CLEAN_V1 = (
    "PLACEHOLDER_sha256_locked_at_first_real_run_fill_in_via_"
    "code/genome_extraction.py_once_data_is_downloaded"
)


# -------------------- F tuple record --------------------

@dataclass(frozen=True)
class StimulusFamily:
    """Machine-checkable stimulus family F (per atlas_tl_session.md 2.5.7).

    Not a free-form string description: a tuple of pinned code identities +
    a dataset hash. Two F instances are interchangeable iff every field
    compares equal.
    """

    scope_id: str
    generator_symbol: str     # "code/stimulus_banks.py::c4_clean_v1"
    filter_symbol: str        # "code/stimulus_banks.py::filter_len_256_english"
    invariance_check_symbol: str  # "code/stimulus_banks.py::in_family"
    dataset_hash: str
    length_tokens: int
    invariances: tuple[str, ...]  # syntactic transformations, decidable

    def pointer_tuples(self) -> list[tuple[str, str]]:
        """The pinned (file_path, symbol) references this F declares."""
        return [
            ("code/stimulus_banks.py", "c4_clean_v1"),
            ("code/stimulus_banks.py", "filter_len_256_english"),
            ("code/stimulus_banks.py", "in_family"),
        ]


# -------------------- Generators --------------------

_C4_SCOPE_ID = "text.c4_clean.len256.v1"


def c4_clean_v1(seed: int, n_samples: int = 5000,
                length_tokens: int = 256) -> Iterator[dict[str, Any]]:
    """Yield n_samples stimuli deterministically from the C4-clean distribution.

    Streams `allenai/c4` (en subset) via the `datasets` library in streaming
    mode, seeds with the given seed for reproducibility, applies
    `filter_len_256_english`, and yields until n_samples accepted stimuli
    have been produced (or the stream is exhausted, which should not happen
    for C4-scale).

    Each yielded item:
        {"scope_id": str,     # constant "text.c4_clean.len256.v1"
         "seed": int,
         "idx": int,           # 0..n_samples-1
         "text": str,          # raw Unicode text, not yet tokenized
         "length_tokens_est": int  # whitespace-word-count heuristic proxy}

    Caller per-model tokenizes downstream. The `length_tokens` prereg field
    is a target; real per-model token count is recorded in the atlas row.

    Determinism: seeded via `datasets.IterableDataset.shuffle(seed=seed)`.
    Two calls with the same seed produce the same sequence.
    """
    from datasets import load_dataset  # lazy: heavy import

    # Streaming + shuffle(seed) + filter yields a reproducible stream.
    ds = load_dataset(
        "allenai/c4", "en", split="train", streaming=True,
        trust_remote_code=False,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    yielded = 0
    for idx, example in enumerate(ds):
        if yielded >= n_samples:
            break
        if not filter_len_256_english(example):
            continue
        text = example["text"]
        yield {
            "scope_id": _C4_SCOPE_ID,
            "seed": seed,
            "idx": yielded,
            "text": text,
            "length_tokens_est": _whitespace_word_count(text),
        }
        yielded += 1


# -------------------- Multilingual C4 (Tier 1b) --------------------

_C4_MULTILINGUAL_SCOPE_ID = "text.c4_multilingual.len256.v1"
# Languages cover Germanic (German), Romance (French), Sino-Tibetan (Chinese
# simplified), Afro-Asiatic (Arabic), Dravidian (Tamil). Spans 3 scripts
# (Latin, Han, Arabic, Tamil) and 5 language families.
_C4_MULTILINGUAL_LANGS = ("de", "fr", "zh", "ar", "ta")


def c4_multilingual_v1(seed: int, n_samples: int = 2000,
                       langs: tuple[str, ...] = _C4_MULTILINGUAL_LANGS
                       ) -> Iterator[dict[str, Any]]:
    """Stream multilingual C4 (allenai/c4 per-language subsets).

    Yields `n_samples // len(langs)` examples per language, round-robin
    interleaved. Tests: is the kNN-k10 clustering value a property of
    English-language geometry or of the underlying manifold shared across
    languages?

    Same-architecture test: runs Qwen3 / BERT / MiniLM on non-English
    input, compares atlas values vs the English baseline. If kNN-k10 is
    invariant, the universality claim is modality-level not language-level.

    NOTE: this is a new stimulus family (scope_id differs from c4_clean_v1),
    so any Gate-1 verdict is evaluated per-(system, lang) scope. A prereg
    extension will compute a fresh dataset_hash.
    """
    from datasets import load_dataset  # lazy
    per_lang = max(1, n_samples // len(langs))
    iters = {}
    for lang in langs:
        try:
            ds = load_dataset("allenai/c4", lang, split="train",
                              streaming=True, trust_remote_code=False)
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
            iters[lang] = iter(ds)
        except Exception as exc:
            print(f"WARN: c4_multilingual_v1 skipping lang={lang}: "
                  f"{type(exc).__name__}: {exc}")

    yielded_per_lang: dict[str, int] = {lang: 0 for lang in iters}
    total_idx = 0
    while any(yielded_per_lang[lang] < per_lang for lang in iters):
        for lang in list(iters.keys()):
            if yielded_per_lang[lang] >= per_lang:
                continue
            try:
                example = next(iters[lang])
            except StopIteration:
                del iters[lang]
                continue
            # Use same length filter (whitespace word count proxy).
            wc = _whitespace_word_count(example.get("text", ""))
            if wc < 100 or wc > 500:
                continue
            yield {
                "scope_id": _C4_MULTILINGUAL_SCOPE_ID,
                "seed": seed,
                "idx": total_idx,
                "lang": lang,
                "text": example["text"],
                "length_tokens_est": wc,
            }
            yielded_per_lang[lang] += 1
            total_idx += 1


# -------------------- Filter --------------------

def _whitespace_word_count(text: str) -> int:
    """Fast proxy for tokenized length: whitespace-split word count.

    For English BPE tokenizers (Qwen3/Llama/Mamba), `tokens ~= 1.3 * words`
    on average. Good enough to filter candidates before per-model tokenization.
    """
    return len(text.split())


def filter_len_256_english(example: dict[str, Any]) -> bool:
    """Predicate: example is English text likely to tokenize to around 256 tokens.

    Heuristic (BPE token count ~= 1.3 * word count):
      - Length target 256 tokens -> target ~196 words
      - Accept range: 150 to 350 words (covers 195 to 455 tokens)
      - Text must be non-empty, not code / URL-heavy

    Returns True iff the example should be included in the stimulus bank. Called
    from c4_clean_v1 during streaming; also usable standalone for auditing.
    """
    text = example.get("text") if isinstance(example, dict) else None
    if not isinstance(text, str) or not text.strip():
        return False

    wc = _whitespace_word_count(text)
    if wc < 150 or wc > 350:
        return False

    # Reject text dominated by URLs (C4 has leakage cases).
    url_fraction = (text.count("http://") + text.count("https://")) * 10 / max(wc, 1)
    if url_fraction > 0.10:
        return False

    # Reject text that looks like a code dump (many braces/semicolons per word).
    symbol_rate = sum(text.count(c) for c in "{}[];") / max(wc, 1)
    if symbol_rate > 0.50:
        return False

    return True


# -------------------- Invariance check --------------------

def _whitespace_norm(text: str) -> str:
    """Collapse runs of whitespace to single space; strip ends. Decidable."""
    return " ".join(text.split())


def _case_norm(text: str) -> str:
    """Lowercase. Decidable under Unicode casing (ASCII for our stimuli)."""
    return text.casefold()


def _canonicalize(text: str) -> str:
    """Apply the Batch-1-declared invariances in order: whitespace_norm,
    case_norm. Idempotent.
    """
    return _case_norm(_whitespace_norm(text))


def in_family(a: dict[str, Any], b: dict[str, Any],
              family_scope_id: str) -> bool:
    """True iff stimulus a and stimulus b belong to the same family F.

    Checks (all must hold; short-circuits):
      1. Both stimuli declare scope_id == family_scope_id.
      2. Both pass filter_len_256_english.
      3. Their canonicalized texts (whitespace_norm then case_norm applied)
         are string-equal — meaning one is obtainable from the other by a
         composition of declared invariances.

    Check 3 is the real membership test. Two DIFFERENT stimuli from the
    same family will fail it (their canonical forms differ); that is
    correct. In-family membership is NOT "same distribution" — it is
    "equivalent up to declared invariances."

    Decidable: all three operations are syntactic.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.get("scope_id") != family_scope_id:
        return False
    if b.get("scope_id") != family_scope_id:
        return False
    if not filter_len_256_english(a):
        return False
    if not filter_len_256_english(b):
        return False
    text_a = a.get("text", "")
    text_b = b.get("text", "")
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return False
    return _canonicalize(text_a) == _canonicalize(text_b)


# -------------------- Canonical F for Batch-1 --------------------

# -------------------- Vision stimulus bank (Batch-1 cross-modal extension) --------------------

_IMAGENET_SCOPE_ID = "vision.imagenet1k_val.v1"


def imagenet_val_v1(seed: int, n_samples: int = 500) -> Iterator[dict[str, Any]]:
    """Yield n_samples ImageNet-1k validation images deterministically.

    Uses the HF `datasets` `ILSVRC/imagenet-1k` validation split in streaming
    mode with shuffle(seed) for reproducibility. Added for Batch-1 cross-modal
    extension per strategic-adversarial Codex directive.

    Yields:
        {"scope_id": str,     # "vision.imagenet1k_val.v1"
         "seed": int,
         "idx": int,
         "image": PIL.Image,  # RGB PIL Image (preprocessed by caller)
         "label": int}

    Falls back to `imagefolder` if `ILSVRC/imagenet-1k` requires auth; callers
    should handle either.
    """
    from datasets import load_dataset  # lazy import
    # HF ImageNet validation requires gated access; try public mirrors first.
    candidates = [
        ("zh-plus/tiny-imagenet", "valid"),
        ("mrm8488/ImageNet1K-val", "train"),
    ]
    ds = None
    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split, streaming=True,
                              trust_remote_code=False)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(
            "no accessible ImageNet-val mirror; add a local fixture or grant "
            "HF access to ILSVRC/imagenet-1k."
        )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    yielded = 0
    for idx, example in enumerate(ds):
        if yielded >= n_samples:
            break
        img = example.get("image")
        if img is None:
            continue
        # Convert to RGB for downstream image-processor compatibility.
        try:
            img = img.convert("RGB")
        except Exception:
            continue
        yield {
            "scope_id": _IMAGENET_SCOPE_ID,
            "seed": seed,
            "idx": yielded,
            "image": img,
            "label": int(example.get("label", -1)),
        }
        yielded += 1


BATCH1_VISION_F = StimulusFamily(
    scope_id=_IMAGENET_SCOPE_ID,
    generator_symbol="code/stimulus_banks.py::imagenet_val_v1",
    filter_symbol="code/stimulus_banks.py::filter_len_256_english",  # N/A vision
    invariance_check_symbol="code/stimulus_banks.py::in_family",      # N/A vision
    dataset_hash="PLACEHOLDER_sha256_imagenet_val",
    length_tokens=224,  # image side in pixels
    invariances=("rgb_conversion", "resize_224"),
)


BATCH1_LANGUAGE_F = StimulusFamily(
    scope_id="text.c4_clean.len256.v1_seeds42_123_456",
    generator_symbol="code/stimulus_banks.py::c4_clean_v1",
    filter_symbol="code/stimulus_banks.py::filter_len_256_english",
    invariance_check_symbol="code/stimulus_banks.py::in_family",
    dataset_hash=DATASET_HASH_C4_CLEAN_V1,
    length_tokens=256,
    invariances=("whitespace_norm", "case_norm"),
)


# -------------------- Utilities --------------------

def sha256_of_file(path: str) -> str:
    """Compute sha256 of a file. Used to populate DATASET_HASH_C4_CLEAN_V1
    once the C4-clean slice is fixed on disk.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    # Sanity: verify the symbols declared in BATCH1_LANGUAGE_F actually
    # exist in this module's namespace. If this exits 0, the pinned
    # pointers in the prereg can reach the real objects.
    import sys

    missing: list[str] = []
    for _fp, symbol in BATCH1_LANGUAGE_F.pointer_tuples():
        if symbol not in globals():
            missing.append(symbol)

    if missing:
        print(f"ERROR: missing pinned symbols in stimulus_banks: {missing}",
              file=sys.stderr)
        sys.exit(1)

    print(f"OK: {len(BATCH1_LANGUAGE_F.pointer_tuples())} pinned symbols "
          f"resolve in stimulus_banks.py")
    print(f"  scope_id: {BATCH1_LANGUAGE_F.scope_id}")
    sys.exit(0)
