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

def c4_clean_v1(seed: int, n_samples: int = 5000,
                length_tokens: int = 256) -> Iterator[dict[str, Any]]:
    """Yield n_samples stimuli deterministically from the C4-clean distribution.

    Each yielded item:
        {"scope_id": "text.c4_clean.len256.v1",
         "seed": int,
         "idx": int,
         "text": str,  # raw Unicode text, not yet tokenized
         "length_tokens": int  # after per-model tokenization}

    STUB: raises NotImplementedError until Batch-1 extraction lands. The
    signature is locked so prereg pointers can reference it now.
    """
    raise NotImplementedError(
        "c4_clean_v1 is a pinned-identity stub for prereg F references. "
        "Implementation lands with code/genome_extraction.py after prereg "
        "lock and smoke-test approval per atlas_tl_session.md section 3g."
    )


# -------------------- Filter --------------------

def filter_len_256_english(example: dict[str, Any]) -> bool:
    """Predicate: example is English text with tokenized length == 256.

    STUB: returns False and raises NotImplementedError so any call fails loudly
    before Batch-1 implementation lands. Signature locked.
    """
    raise NotImplementedError(
        "filter_len_256_english is a pinned-identity stub. Implementation "
        "lands with code/genome_extraction.py."
    )


# -------------------- Invariance check --------------------

def in_family(a: dict[str, Any], b: dict[str, Any],
              family_scope_id: str) -> bool:
    """True iff stimulus a and stimulus b belong to the same F under
    declared invariances (whitespace-norm, case-norm for the Batch-1
    scope).

    Per prereg 2.5.7: invariances must be syntactic / canonicalizable so that
    in_family is decidable. Semantic invariances are forbidden.

    STUB: raises NotImplementedError. Signature locked.
    """
    raise NotImplementedError(
        "in_family is a pinned-identity stub. Implementation lands with "
        "code/genome_extraction.py."
    )


# -------------------- Canonical F for Batch-1 --------------------

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
