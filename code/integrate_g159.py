"""
integrate_g159.py — mechanical g159 verdict integration.

Reads results/genome_159_cross_class_lesion.json and:
  - prints summary (per-model R_nat/R_shuf medians, verdict)
  - prints ledger entry to append (with --commit appends)
  - generates WIKI patch text

Per locked decision tree: PASS -> Level-2 candidate, promote in CLAIM_EVIDENCE_MAP;
PARTIAL/KILL -> theory's class-extension prediction questioned.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "results" / "genome_159_cross_class_lesion.json"
LEDGER = ROOT / "experiments" / "ledger.jsonl"


def main():
    if not RESULT.exists():
        print(f"ERROR: {RESULT} not found; g159 not yet complete.")
        sys.exit(1)
    r = json.loads(RESULT.read_text(encoding="utf-8"))

    res = r.get("results", {})
    verdict = r.get("verdict", "(missing)")
    elapsed_s = r.get("elapsed_s", 0)

    if "PASS:" in verdict:
        path = "PASS"
        path_text = ("PASS: 3/3 classes show transport-vs-local asymmetry on natural data + "
                     "shuf-collapse in >=2/3 classes. Transport-vs-local asymmetry is class-general at "
                     "Level-2. Update CLAIM_EVIDENCE_MAP and §0.1 score 6/10 -> 7/10.")
    elif "PARTIAL:" in verdict:
        path = "PARTIAL"
        path_text = ("PARTIAL: 2/3 classes pass; class-extension is partial. Acceptable for "
                     "Llama-family + 1 other architecture; pivots to g160 with caveat.")
    elif "INCOMPLETE:" in verdict:
        path = "INCOMPLETE"
        path_text = ("INCOMPLETE: not all 3 models loaded. Investigate which class failed and "
                     "either fix or reduce scope.")
    elif "KILL:" in verdict:
        path = "KILL"
        path_text = ("KILL: theory does NOT generalize across architecture classes; result is "
                     "Llama-only. Combined with g157+g157b mechanism rejection, the empirical "
                     "thesis is narrow and architecturally specific. Pivot to distillation track "
                     "(g160 -> g155) for the manifesto cash-out.")
    else:
        path = "AMBIGUOUS"
        path_text = "AMBIGUOUS: see verdict text. Manual investigation."

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "id": "genome_159_cross_class_lesion",
        "purpose": "g159 cross-class causal transport-vs-local lesion: tests whether transport-vs-local asymmetry replicates across Qwen3 (transformer) + RWKV-4 (linear-recurrent) + Falcon-H1 (hybrid). Independent of η/δ probe.",
        "git_commit": "(set at commit time)",
        "config_path": "code/genome_159_cross_class_lesion.py",
        "prereg_path": "research/prereg/genome_159_cross_class_lesion_2026-04-26.md",
        "systems": list(res.keys()),
        "primitive": "cross_class_lesion_pca32",
        "universality_level_claimed": None,
        "metrics": {
            "per_model": {k: {"R_nat_median": v.get("R_nat_median"),
                              "R_shuf_median": v.get("R_shuf_median"),
                              "n_layers": v.get("n_layers")}
                          for k, v in res.items() if "error" not in v},
            "errors": {k: v.get("error") for k, v in res.items() if "error" in v},
            "verdict": verdict,
            "elapsed_s": elapsed_s,
        },
        "artifacts": [
            "code/genome_159_cross_class_lesion.py",
            "results/genome_159_cross_class_lesion.json",
            "results/genome_159_run.log",
        ],
        "notes": f"Path {path}: {path_text}",
        "status": "completed",
    }

    print("=== g159 verdict ===")
    print(f"  verdict: {verdict[:120]}")
    print(f"  elapsed: {elapsed_s/60:.1f} min")
    print(f"\n  Decision: {path}")
    print(f"  {path_text}")

    print("\n=== Per-model results ===")
    for model_name, v in res.items():
        if "error" in v:
            print(f"  {model_name}: ERROR ({v['error']})")
        else:
            R_n = v.get("R_nat_median")
            R_s = v.get("R_shuf_median")
            print(f"  {model_name}: R_nat_median={R_n} R_shuf_median={R_s}")

    wiki_patch = f"""**genome_159 COMPLETED ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}): {verdict.split(':',1)[0]}**
- Per-model R_nat_median: {[f"{k}: {v.get('R_nat_median')}" for k, v in res.items() if 'error' not in v]}
- Per-model R_shuf_median: {[f"{k}: {v.get('R_shuf_median')}" for k, v in res.items() if 'error' not in v]}
- **Path {path}:** {path_text}
- `code/genome_159_cross_class_lesion.py` -> `results/genome_159_cross_class_lesion.json`
"""
    print("\n=== WIKI patch ===")
    print(wiki_patch)

    print("\nTo append to ledger.jsonl:")
    print("  python code/integrate_g159.py --commit")

    if "--commit" in sys.argv:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(entry) + "\n")
        wiki_path = ROOT / "research" / "g159_wiki_patch.md"
        wiki_path.write_text(wiki_patch, encoding="utf-8")
        print(f"\nAppended to {LEDGER}")
        print(f"WIKI patch stashed at {wiki_path}")


if __name__ == "__main__":
    main()
