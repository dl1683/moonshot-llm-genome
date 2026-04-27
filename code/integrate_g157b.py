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

    # Compute the eta-only criterion as a probe-pathology-robust alternative
    # (per research/programs/post_g157b_decision_tree.md update 2026-04-26 22:37).
    eta_nat_minimal_per_layer = []
    eta_shuf_minimal_per_layer = []
    eta_nat_baseline_per_layer = []
    eta_shuf_baseline_per_layer = []
    res = r.get("results", {})
    for ckpt_key, ckpt_data in res.items():
        per_layer = ckpt_data.get("per_layer", {})
        for layer_str, lv in per_layer.items():
            eta = lv.get("eta_hat", float("nan"))
            if "natural__minimal" in ckpt_key:
                eta_nat_minimal_per_layer.append(eta)
            elif "token_shuffled__minimal" in ckpt_key:
                eta_shuf_minimal_per_layer.append(eta)
            elif "natural__baseline" in ckpt_key:
                eta_nat_baseline_per_layer.append(eta)
            elif "token_shuffled__baseline" in ckpt_key:
                eta_shuf_baseline_per_layer.append(eta)

    def _mean(xs):
        if not xs:
            return float("nan")
        import math
        return sum(xs) / len(xs)

    eta_nat_min = _mean(eta_nat_minimal_per_layer)
    eta_shuf_min = _mean(eta_shuf_minimal_per_layer)
    eta_nat_base = _mean(eta_nat_baseline_per_layer)
    eta_shuf_base = _mean(eta_shuf_baseline_per_layer)
    eta_contrast_min = eta_nat_min - eta_shuf_min

    # Probe pathology check: lin probe blew up if delta is >100 nats anywhere
    pathology_flagged = False
    for ckpt_key, ckpt_data in res.items():
        for layer_str, lv in ckpt_data.get("per_layer", {}).items():
            d = lv.get("delta_hat_mlp", 0)
            if abs(d) > 100:
                pathology_flagged = True
                break
        if pathology_flagged:
            break

    print("\n=== Eta-only (probe-pathology robust) ===")
    print(f"  eta nat-minimal mid-band: {eta_nat_min:+.4f}")
    print(f"  eta shuf-minimal mid-band: {eta_shuf_min:+.4f}")
    print(f"  eta contrast (nat-shuf): {eta_contrast_min:+.4f}")
    print(f"  pathology flagged (delta > 100 anywhere): {pathology_flagged}")

    eta_only_verdict = ""
    if eta_nat_min > 0 and eta_shuf_min <= 0 and eta_contrast_min >= 0.05:
        eta_only_verdict = "ETA_ONLY_PASS: prefix-info signal exists on natural-minimal AND collapses on shuffled"
    elif eta_nat_min > 0:
        eta_only_verdict = "ETA_ONLY_WEAK: prefix beats local on natural-minimal but contrast small"
    else:
        eta_only_verdict = "ETA_ONLY_KILL: no prefix-beats-local signal on natural-minimal"
    print(f"  eta-only verdict: {eta_only_verdict}")

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

    # Build WIKI patch text
    wiki_patch = f"""**genome_157b PILOT COMPLETED ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}): {verdict.split(':',1)[0]}**
- nat_G_mean = {nat_G:+.4f}, shuf_G_mean = {shuf_G:+.4f}, contrast = {contrast:+.4f}
- eta-only criterion: nat-min={eta_nat_min:+.4f}, shuf-min={eta_shuf_min:+.4f}, contrast={eta_contrast_min:+.4f}
  ({eta_only_verdict})
- Pathology flagged: {pathology_flagged} (lin probe |delta|>100 anywhere)
- **Path {path}:** {path_text}
- `code/genome_157b_eta_delta_probe_embedding_prefix.py` -> `results/genome_157b_eta_delta_probe.json`
"""
    print("\n=== WIKI patch (paste into ACTIVE EXPERIMENT QUEUE section) ===")
    print(wiki_patch)

    if "--commit" in sys.argv:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"\nAppended to {LEDGER}")
        # Also write the wiki patch to a stash file
        wiki_path = ROOT / "research" / "g157b_wiki_patch.md"
        wiki_path.write_text(wiki_patch, encoding="utf-8")
        print(f"WIKI patch stashed at {wiki_path}; manual integration required")


if __name__ == "__main__":
    main()
