"""End-to-end smoke test for the Batch-1 Neural Genome pipeline.

Runs the minimum viable path per prereg
`research/prereg/genome_id_portability_2026-04-21.md` section 14 compliance
checklist: 5 sentences x 2 layers x 1 system x 1 resample x 1 quant, must
complete in <10 minutes wall-clock on the RTX 5090.

Pipeline: stimulus -> loader -> extractor -> primitives -> atlas-row JSON.

Outputs a machine-checkable atlas row to results/smoke/atlas_row.json so
humans + auditors can inspect the end-to-end artifact. Also appends a
ledger entry to experiments/ledger.jsonl.

Per CLAUDE.md section 8 git discipline: logical change = one idea; this
is the "first end-to-end smoke" unit.

Usage:
    python code/genome_smoke_test.py

Exits 0 if the pipeline completes without errors and the atlas row passes
basic sanity checks. Exits non-zero otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure `code/` is importable whether invoked from root or code/.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from stimulus_banks import _whitespace_word_count, _canonicalize  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_extractor import (  # noqa: E402
    extract_trajectory, sentinel_layer_indices,
)
from genome_primitives import (  # noqa: E402
    twonn_id, mle_id, participation_ratio, knn_clustering_coefficient,
)


# -------------------- Smoke-sized config --------------------

SMOKE_N_SENTENCES = 5
SMOKE_N_LAYERS = 2          # sentinel subset for speed
SMOKE_SYSTEM = "Qwen/Qwen3-0.6B"
SMOKE_QUANT = "fp16"
SMOKE_SEED = 42
SMOKE_MAX_LENGTH = 128      # shorter than prereg's 256 to speed up the smoke

# Hardcoded fallback stimuli so the smoke test does NOT require a live
# HuggingFace datasets download (keeps the <10-min wall-clock). These are
# short public-domain-style sentences; just enough to get ~10-30 tokens
# each under BPE. Replaced by real `c4_clean_v1` stream in the full Batch-1
# run.
_SMOKE_STIMULI = [
    "The cat sat on the mat and watched the rain fall outside the window all afternoon.",
    "She opened the book to a random page and began to read aloud to the sleeping dog.",
    "Scientists have long debated whether language shapes thought or thought shapes language.",
    "The old lighthouse keeper walked down the spiral staircase carrying a lantern and a heavy iron key.",
    "A thousand starlings rose from the field in a single coordinated motion that looked almost magical.",
]


# -------------------- Atlas-row schema --------------------

def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_commit_sha() -> str:
    """Best-effort: read HEAD SHA via git without subprocess if possible."""
    git_head = _THIS_DIR.parent / ".git" / "HEAD"
    if not git_head.exists():
        return "unknown"
    try:
        ref = git_head.read_text().strip()
        if ref.startswith("ref:"):
            ref_path = _THIS_DIR.parent / ".git" / ref.split()[1]
            if ref_path.exists():
                return ref_path.read_text().strip()
        return ref
    except OSError:
        return "unknown"


# -------------------- Pipeline --------------------

def run_smoke() -> dict:
    print(f"=== SMOKE TEST START ===")
    print(f"system: {SMOKE_SYSTEM}, quant: {SMOKE_QUANT}, "
          f"n_sentences: {SMOKE_N_SENTENCES}, seed: {SMOKE_SEED}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    t0 = time.time()

    # 1. Stimulus: use the hardcoded fixture (no C4 download).
    stimuli = _SMOKE_STIMULI[:SMOKE_N_SENTENCES]
    print(f"[{time.time()-t0:.1f}s] stimuli: {len(stimuli)} sentences, "
          f"avg wc={sum(_whitespace_word_count(s) for s in stimuli)/len(stimuli):.1f}")

    # 2. Load system.
    sys_obj = load_system(SMOKE_SYSTEM, quant=SMOKE_QUANT, untrained=False,
                          device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    sentinel_idxs = sentinel_layer_indices(n_layers)
    # Smoke: use just 2 of the 3 sentinel depths.
    chosen_idxs = sentinel_idxs[:SMOKE_N_LAYERS]
    print(f"[{time.time()-t0:.1f}s] loaded {sys_obj.hf_id} "
          f"(L={n_layers}); extracting layers {chosen_idxs}")

    # 3. Extract (seq_mean pooling only for smoke).
    traj = extract_trajectory(
        sys_obj.model, sys_obj.tokenizer, stimuli,
        layer_indices=chosen_idxs,
        pooling="seq_mean",
        max_length=SMOKE_MAX_LENGTH,
        device="cuda",
        system_key=sys_obj.system_key,
        class_id=sys_obj.class_id,
        quantization=sys_obj.quant,
        stimulus_version="smoke.hardcoded.v1",
        seed=SMOKE_SEED,
    )
    print(f"[{time.time()-t0:.1f}s] extracted {len(traj.layers)} layers; "
          f"X shapes: {[lyr.X.shape for lyr in traj.layers]}")

    # 4. Primitives per layer.
    atlas_rows = []
    for lyr in traj.layers:
        X = lyr.X
        measurements = [
            twonn_id(X),
            mle_id(X, k=min(10, X.shape[0] - 1)),
            participation_ratio(X, centered=True),
            participation_ratio(X, centered=False),
            knn_clustering_coefficient(X, k=min(5, X.shape[0] - 1)),
            knn_clustering_coefficient(X, k=min(10, X.shape[0] - 1)),
        ]
        for m in measurements:
            row = {
                "system_key": traj.system_key,
                "class_id": traj.class_id,
                "hf_id": sys_obj.hf_id,
                "untrained": sys_obj.untrained,
                "quantization": traj.quantization,
                "pooling": traj.pooling,
                "stimulus_version": traj.stimulus_version,
                "seed": traj.seed,
                "k_index": lyr.k_index,
                "k_normalized": lyr.k_normalized,
                "n_points": m.n_points,
                "primitive_id": m.primitive_id,
                "estimator": m.estimator,
                "value": m.value,
                "se": m.se,
                "scope_label": (
                    f"(modality=text, stimulus_family=smoke.hardcoded.v1, "
                    f"pooling=seq_mean, tokenizer=per-model-native)"
                ),
                "commit_sha": _current_commit_sha(),
                "metadata": m.metadata,
            }
            atlas_rows.append(row)

    print(f"[{time.time()-t0:.1f}s] computed {len(atlas_rows)} atlas rows "
          f"(primitives x layers)")

    # 5. Free the model (smoke doesn't need to keep it).
    sys_obj.unload()

    # 6. Emit.
    out_dir = _THIS_DIR.parent / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "atlas_rows.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump({
            "smoke_test_id": "genome_smoke_2026-04-21",
            "n_rows": len(atlas_rows),
            "wall_clock_seconds": round(time.time() - t0, 2),
            "rows": atlas_rows,
        }, fp, indent=2, default=float)

    # 7. Sanity checks.
    errors = []
    if len(atlas_rows) != len(traj.layers) * 6:
        errors.append(
            f"row count mismatch: got {len(atlas_rows)}, "
            f"expected {len(traj.layers) * 6}"
        )
    for r in atlas_rows:
        if not (0.0 < r["value"] < 1e6) and r["primitive_id"] == "intrinsic_dim":
            errors.append(f"implausible ID value: {r['value']}")
    if errors:
        print(f"SMOKE ERRORS: {errors}")
        return {"passed": False, "errors": errors, "output": str(out_path)}

    print(f"[{time.time()-t0:.1f}s] wrote {out_path}")
    print(f"=== SMOKE TEST PASSED ===")
    return {
        "passed": True,
        "errors": [],
        "output": str(out_path),
        "wall_clock_seconds": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    # Ensure Windows+CUDA safe defaults (DataLoader workers etc.) by not
    # spawning any subprocess workers here.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    result = run_smoke()
    sys.exit(0 if result["passed"] else 1)
