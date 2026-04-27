"""
integrate_g165.py — mechanical g165 verdict integration.

Reads results/genome_165_annealed_donor.json and:
  - prints per-arm summary (mean/CI of final-step C4 NLL advantage vs scratch)
  - applies decision tree from research/prereg/genome_165_annealed_donor_2026-04-27.md
  - emits ledger entry + WIKI patch

Decision rules (locked):
  PASS:       at least one (lambda_0, schedule) arm has mean delta >= +0.5 nats
              AND paired bootstrap 95% CI excludes zero
  WEAK_PASS:  at least one arm has mean delta in [+0.2, +0.5) nats with CI > 0
  FAIL:       all 12 anchored arms wash out (delta < +0.2 OR CI crosses zero)

  Special note: if the constant-lambda arm shows comparable persistence to a
  decay-schedule arm, that's evidence the active ingredient is anchor presence
  (not decay schedule). Codex direction review interpretation: if PASS_constant
  >= PASS_decay, the result is g125-like (anchor-rate-zero suffices) and the
  decay schedule is NOT the rescue mechanism.
"""
from __future__ import annotations
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "results" / "genome_165_annealed_donor.json"
LEDGER = ROOT / "experiments" / "ledger.jsonl"


def main():
    if not RESULT.exists():
        print(f"ERROR: {RESULT} not found; g165 not yet complete.")
        sys.exit(1)
    r = json.loads(RESULT.read_text(encoding="utf-8"))

    # If incremental save still has 'incremental': True flag, abort
    if r.get("incremental"):
        print("ERROR: g165 result is still incremental (run not yet finished).")
        sys.exit(2)

    summary = r.get("summary", {})
    verdict = r.get("verdict", "(missing)")
    config = r.get("config", {})
    seeds = config.get("seeds", [])
    elapsed_s = r.get("elapsed_s", 0)

    per_arm = summary.get("per_arm", {})
    pass_arms = summary.get("pass_arms", [])
    weak_arms = summary.get("weak_arms", [])

    # Categorize verdict by Path
    if "PASS:" in verdict:
        path = "PASS"
    elif "WEAK_PASS:" in verdict:
        path = "WEAK_PASS"
    elif "FAIL:" in verdict:
        path = "FAIL"
    else:
        path = "AMBIGUOUS"

    # Active-ingredient analysis: is the persistence due to decay schedule or
    # just anchor presence? Compare PASS arms by schedule type.
    decay_pass = [a for a in pass_arms if "constant" not in a]
    constant_pass = [a for a in pass_arms if "constant" in a]
    if path == "PASS":
        if decay_pass and not constant_pass:
            interpretation = (
                "Decay schedule IS the active ingredient: only decay arms persist, "
                "constant-lambda arms wash out. This is the canonical annealed-donor result."
            )
        elif constant_pass and not decay_pass:
            interpretation = (
                "Constant anchor (anchor-rate-zero) is sufficient; decay does NOT add value. "
                "g125-like: freezing donor weights produces persistence; the active ingredient "
                "is anchor PRESENCE not anchor DECAY."
            )
        else:
            interpretation = (
                "Both constant and decay arms persist. Decay schedule may be helpful but not "
                "necessary; constant anchor is the load-bearing mechanism. Mixed."
            )
    elif path == "FAIL":
        interpretation = (
            "Annealed-donor hypothesis NOT supported at canonical scale. "
            "Pivot: (1) g166 = optimizer-state + decay-anchor combined (the g137 outlier "
            "follow-up using ~1420-step half-life), or (2) g155 procurement (wall-power meter) "
            "to unblock the 8.2/10 path."
        )
    elif path == "WEAK_PASS":
        interpretation = (
            "Weak signal at canonical scale; expand seeds (g165e endpoint expansion) or "
            "expand decay-schedule grid before locking the headline. Codex sign-off required."
        )
    else:
        interpretation = "AMBIGUOUS verdict; manual investigation needed."

    print("=== g165 annealed-donor verdict ===")
    print(f"  seeds: {seeds}  elapsed: {elapsed_s/3600:.2f}h")
    print(f"  PATH: {path}")
    print(f"  VERDICT: {verdict}")
    print(f"\n  pass_arms ({len(pass_arms)}): {pass_arms}")
    print(f"  weak_arms ({len(weak_arms)}): {weak_arms}")
    print(f"\n  Interpretation: {interpretation}")
    print(f"\n=== per-arm final advantages ===")
    # Sort by mean advantage descending
    sorted_arms = sorted(
        per_arm.items(),
        key=lambda kv: kv[1].get("mean_final_advantage_nats", -1e9),
        reverse=True,
    )
    for label, m in sorted_arms[:15]:
        adv = m.get("mean_final_advantage_nats", float("nan"))
        ci_lo = m.get("ci_95_lo", float("nan"))
        ci_hi = m.get("ci_95_hi", float("nan"))
        print(f"    {label:32s} mean={adv:+.3f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "id": "genome_165_annealed_donor",
        "purpose": "g165 annealed-donor / decaying-anchor washout test (cycle 24 strategic pivot — first §0-axis experiment).",
        "git_commit": "(set at commit time)",
        "config_path": "code/genome_165_annealed_donor.py",
        "prereg_path": "research/prereg/genome_165_annealed_donor_2026-04-27.md",
        "systems": ["Qwen3-0.6B donor (frozen)", "Qwen3-0.6B-arch random-init recipient"],
        "primitive": "annealed_donor",
        "universality_level_claimed": None,
        "metrics": {
            "verdict": verdict,
            "path": path,
            "n_pass_arms": len(pass_arms),
            "n_weak_arms": len(weak_arms),
            "pass_arms": pass_arms,
            "weak_arms": weak_arms,
            "per_arm": per_arm,
        },
        "artifacts": [
            "code/genome_165_annealed_donor.py",
            "results/genome_165_annealed_donor.json",
        ],
        "notes": f"Path {path}: {interpretation}. Wall {elapsed_s/3600:.2f}h.",
        "status": "completed",
    }

    wiki_patch = f"""**genome_165 COMPLETED ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}): {path}**
- Verdict: {verdict}
- pass_arms: {len(pass_arms)}; weak_arms: {len(weak_arms)}
- **Interpretation:** {interpretation}
- Wall: {elapsed_s/3600:.2f}h.
- `code/genome_165_annealed_donor.py` -> `results/genome_165_annealed_donor.json`
"""
    print("\n=== WIKI patch ===")
    print(wiki_patch)
    print("\nTo append to ledger.jsonl + stash WIKI patch:")
    print("  python code/integrate_g165.py --commit")

    if "--commit" in sys.argv:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(entry) + "\n")
        wiki_path = ROOT / "research" / "g165_wiki_patch.md"
        wiki_path.write_text(wiki_patch, encoding="utf-8")
        print(f"\nAppended to {LEDGER}")
        print(f"WIKI patch stashed at {wiki_path}")


if __name__ == "__main__":
    main()
