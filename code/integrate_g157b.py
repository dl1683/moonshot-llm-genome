"""
integrate_g157b.py

Mechanical integration of g157b PILOT verdict into ledger + WIKI snippet.
Per research/programs/post_g157b_decision_tree.md.

Usage: python code/integrate_g157b.py
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "results" / "genome_157b_eta_delta_probe.json"
LEDGER = ROOT / "experiments" / "ledger.jsonl"


def main():
    if not RESULT.exists():
        print(f"ERROR: {RESULT} not found; g157b not yet complete.")
        sys.exit(1)
    r = json.loads(RESULT.read_text(encoding="utf-8"))

    nat_G = r.get("nat_G_mean", float("nan"))
    shuf_G = r.get("shuf_G_mean", float("nan"))
    contrast = r.get("contrast", float("nan"))
    verdict = r.get("verdict", "(missing)")

    # Choose path per decision tree
    if "DIRECTIONAL_SUPPORT_157b" in verdict:
        path = "A"
        path_text = "Path A: theory mechanism validated; next = write 3-seed g157c prereg + run + launch g158 in parallel"
    elif "WEAK_SUPPORT_157b" in verdict:
        path = "B"
        path_text = "Path B: direction OK signal weak; next = run g157d (probe-budget expansion: 2000 steps, 5 depths) + launch g158 in parallel"
    elif "KILL_157b" in verdict:
        path = "C"
        path_text = "Path C: mechanism likely wrong; next = run g157 v3 (same-layer FP32 control) + launch g158 in parallel"
    else:
        path = "?"
        path_text = "UNKNOWN verdict; manual investigation required"

    # Build ledger entry
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "id": "genome_157b_eta_delta_probe_embedding_prefix",
        "purpose": "Codex post-g157-v2-KILL alternative probe: embedding-layer prefix instead of same-layer. FP32 + grad clip + skip-non-finite-loss.",
        "git_commit": "(set at commit time)",
        "config_path": "code/genome_157b_eta_delta_probe_embedding_prefix.py",
        "prereg_path": "research/prereg/genome_157b_eta_delta_probe_embedding_prefix_2026-04-26.md",
        "systems": ["12 g156 ckpts (PILOT subset: seed=42 only, 4 ckpts)"],
        "primitive": "eta_delta_probe_embedding_prefix",
        "universality_level_claimed": None,
        "metrics": {
            "nat_G_mean": nat_G,
            "shuf_G_mean": shuf_G,
            "contrast": contrast,
            "verdict": verdict,
        },
        "artifacts": [
            "code/genome_157b_eta_delta_probe_embedding_prefix.py",
            "results/genome_157b_eta_delta_probe.json",
            "results/genome_157b_run.log",
        ],
        "notes": f"PILOT ({path}). {path_text}. Per locked decision tree research/programs/post_g157b_decision_tree.md.",
        "status": "completed",
    }

    print("=== g157b verdict ===")
    print(f"  nat_G_mean = {nat_G:+.4f}")
    print(f"  shuf_G_mean = {shuf_G:+.4f}")
    print(f"  contrast = {contrast:+.4f}")
    print(f"  verdict = {verdict[:120]}")
    print(f"\n  Decision path: {path}")
    print(f"  {path_text}")

    print("\n=== Ledger entry to append ===")
    print(json.dumps(entry, indent=2))

    print("\nTo append to ledger.jsonl, run:")
    print(f"  python code/integrate_g157b.py --commit")

    if "--commit" in sys.argv:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"\nAppended to {LEDGER}")


if __name__ == "__main__":
    main()
