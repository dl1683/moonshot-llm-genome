"""Audit-grade deterministic IO primitives for the Neural Genome pipeline.

Per Codex R7 impl-design D1/D3/D8: sha256 hashing, JSON/JSONL/NPZ writers,
git-HEAD resolver, UTC timestamp. Centralized so the ledger + atlas rows +
smoke test + batch-1 runners all use identical audit logic. Prevents
subtly-different hash / JSON formatting across callers.

ASCII-only source per Windows cp1252 constraint. Deterministic newline=\n.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# -------------------- Hashing --------------------

def sha256_file(path: Path | str) -> str:
    """Compute sha256 hex digest of a file on disk. Streams in 1 MB chunks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"sha256_file: {p} does not exist")
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


# -------------------- JSON / JSONL writers --------------------

def write_json(path: Path | str, obj: dict | list, *, sort_keys: bool = True,
               indent: int = 2) -> None:
    """Deterministic JSON writer: UTF-8, LF line endings, sorted keys."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # default=float converts numpy scalars; lists/dicts pass through.
    with open(p, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(obj, fp, indent=indent, sort_keys=sort_keys, default=float)
        fp.write("\n")


def append_jsonl(path: Path | str, obj: dict) -> None:
    """Append ONE JSON object as a single LF-terminated line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, sort_keys=True, default=float)
    with open(p, "a", encoding="utf-8", newline="\n") as fp:
        fp.write(line + "\n")


def read_jsonl(path: Path | str) -> list[dict]:
    """Read a JSONL file back into a list (newest last)."""
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# -------------------- NPZ writer --------------------

def write_npz(path: Path | str, arrays: dict[str, np.ndarray],
              *, compressed: bool = True) -> None:
    """Write a dict of numpy arrays to .npz. Compressed by default."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(str(p), **arrays)
    else:
        np.savez(str(p), **arrays)


# -------------------- Git HEAD resolution --------------------

def git_head_sha(repo_root: Path | str | None = None) -> str:
    """Read current HEAD commit SHA without subprocess.

    If repo_root is None, walk up from this file to find a .git dir.
    Returns 'unknown' if no git dir resolvable or reading fails.
    """
    root = Path(repo_root) if repo_root is not None else _walk_up_to_git(
        Path(__file__).resolve().parent)
    if root is None:
        return "unknown"
    head = root / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            ref_path = root / ".git" / ref.split()[1]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
        return ref
    except OSError:
        return "unknown"


def _walk_up_to_git(start: Path) -> Path | None:
    probe = start.resolve()
    while probe.parent != probe:
        if (probe / ".git").exists():
            return probe
        probe = probe.parent
    return None


# -------------------- Time --------------------

def utc_now_iso() -> str:
    """YYYY-MM-DDTHH:MM:SSZ in UTC (no microseconds, no timezone offset)."""
    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


# -------------------- CLI sanity --------------------

if __name__ == "__main__":
    print("git HEAD:", git_head_sha())
    print("utc now:", utc_now_iso())
    # Hash this file as a self-check.
    h = sha256_file(__file__)
    print(f"sha256({Path(__file__).name}): {h[:16]}...")
    print("OK: genome_io sanity")
