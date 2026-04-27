"""
genome_161_rwkv_training_extension.py

POST-CHAIN ARCHITECTURE-CLASS EXTENSION: train-time test in non-transformer.

Pre-reg LOCKED: research/prereg/genome_161_rwkv_training_extension_2026-04-26.md
Theory: research/derivations/prefix_information_transport.md
Program: research/programs/post_g156_pass_program.md §g161

Tests whether the natural-vs-shuffled contrast that g156 observed in
transformer architectures replicates at TRAINING TIME in a small RWKV-4
variant. We use a custom small RWKV implementation (not pretrained
weights) so we can vary architecture cleanly:

  baseline_rwkv:    12L + channel-mix
  transport_heavy:  18L no-channel-mix (more depth/time-mix)

FLOP-matched within +/- 2%.

Two conditions x 2 arms x 3 seeds = 12 cells.

PASS_RWKV_INVERSION:
  delta_nat  >= +0.3pp on both c4-val and wikitext-val
  AND delta_shuf <= +0.1pp on both
  AND contrast >= +0.3pp on both
KILL: no contrast or reversed contrast.

Compute: ~2-3 hr per program estimate.

NOTE: This is a STUB implementation. Custom RWKV-4 from scratch is
nontrivial (time-mix recurrence, init conventions). For a real launch
this needs either:
  (a) reuse RWKV's reference implementation as a starting point
  (b) use an existing RWKV-4 PyTorch port and modify

We will fire a Codex consult (codex_prompts/g161_rwkv_implementation.txt)
to design the exact small-RWKV-4 codebase before launching this script.
The skeleton below documents the plan; do NOT launch as-is.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PRELAUNCH_GUARD = """
g161 IMPLEMENTATION INCOMPLETE.

This file is a STUB. RWKV-4 from-scratch training requires careful
architecture choices that should be designed by Codex first. Do not
launch until:

1. codex_outputs/g161_rwkv_implementation.md exists with the agreed
   small-RWKV-4 codebase (channel-mix module, time-mix module,
   exact-FLOP-counter for the comparison).
2. The transport_heavy variant's "no channel-mix" replacement is
   defined precisely (does it use identity? a parameterless MLP?
   re-invest those params into more depth/width?).
3. PILOT smoke test: 1 seed, 100 steps, baseline + transport on
   natural c4 only, verify both arms train and produce sane loss.

After those steps, this stub gets replaced with the real
implementation. Until then, raise to prevent accidental launches.
"""


def main():
    print(PRELAUNCH_GUARD)
    raise RuntimeError("g161 stub: fire Codex design consult first; do not launch.")


if __name__ == "__main__":
    main()
