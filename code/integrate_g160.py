"""
integrate_g160.py — mechanical g160 verdict integration.

Reads results/genome_160_transport_guided_student.json and:
  - prints summary (per-student final C3_macro, CtQ_90_flops, c3_gap_pp)
  - prints ledger entry to append (with --commit appends)
  - generates WIKI patch text

g160 is the manifesto cash-out: matched-cost transport-heavy student vs local-heavy student.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "results" / "genome_160_transport_guided_student.json"
LEDGER = ROOT / "experiments" / "ledger.jsonl"


def main():
    if not RESULT.exists():
        print(f"ERROR: {RESULT} not found; g160 not yet complete.")
        sys.exit(1)
    r = json.loads(RESULT.read_text(encoding="utf-8"))

    summary = r.get("summary", {})
    verdict = r.get("verdict", "(missing)")
    c3_gap_pp = r.get("c3_gap_pp", float("nan"))
    ctq_ratio = r.get("ctq_ratio", float("nan"))
    elapsed = r.get("elapsed_s", 0)

    if "PASS:" in verdict:
        path = "PASS"
        path_text = ("PASS: transport-heavy student beats local-heavy on C3_macro AND CtQ_90 "
                     "at matched inference FLOPs. Theory becomes a model-selection rule. "
                     "MANIFESTO CASH-OUT — proceed to g155 edge benchmark.")
    elif "PARTIAL:" in verdict:
        path = "PARTIAL"
        path_text = ("PARTIAL: only one of (C3_macro, CtQ_90) lands. Architecture-prior "
                     "+ distillation are useful but not a clean design law. Proceed to g155 with caveat.")
    elif "KILL:" in verdict:
        path = "KILL"
        path_text = ("KILL: local-heavy ties or wins on both metrics. The architecture-prior "
                     "advantage observed at random-init does NOT translate to distilled students. "
                     "Theory does not guide design selection. Pivot to: drop transport theory, "
                     "ride empirical g156 PASS as a narrow ablation paper.")
    else:
        path = "AMBIGUOUS"
        path_text = "AMBIGUOUS: see verdict. Manual investigation."

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "id": "genome_160_transport_guided_student",
        "purpose": "g160 manifesto cash-out: matched-inference-FLOPs transport-heavy student (6L noMLP) vs local-heavy student (4L MLP), distilled from Qwen3-0.6B teacher. Tests whether transport theory selects a better matched-cost design.",
        "git_commit": "(set at commit time)",
        "config_path": "code/genome_160_transport_guided_student.py",
        "prereg_path": "research/prereg/genome_160_transport_guided_student_2026-04-26.md",
        "systems": ["transport_heavy 6L_noMLP_h512", "local_heavy 4L_MLP_h384_ffn1024"],
        "primitive": "matched_flops_distillation",
        "universality_level_claimed": None,
        "metrics": {
            "summary": summary,
            "c3_gap_pp": c3_gap_pp,
            "ctq_ratio": ctq_ratio,
            "elapsed_s": elapsed,
            "verdict": verdict,
        },
        "artifacts": [
            "code/genome_160_transport_guided_student.py",
            "results/genome_160_transport_guided_student.json",
            "results/genome_160_run.log",
        ],
        "notes": f"Path {path}: {path_text}",
        "status": "completed",
    }

    print("=== g160 verdict ===")
    print(f"  c3_gap_pp = {c3_gap_pp:+.2f}")
    print(f"  ctq_ratio = {ctq_ratio:.3f}")
    print(f"  elapsed: {elapsed/60:.1f} min")
    print(f"  verdict: {verdict[:120]}")
    print(f"\n  Decision: {path}")
    print(f"  {path_text}")

    print("\n=== Per-student summary ===")
    for sname, s in summary.items():
        print(f"  {sname}: C3_macro={s.get('C3_final_mean'):.4f} CtQ_90_flops={s.get('CtQ_90_flops_mean')/1e12:.2f} TFLOP")

    wiki_patch = f"""**genome_160 COMPLETED ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}): {verdict.split(':',1)[0]}**
- c3_gap_pp = {c3_gap_pp:+.2f}pp; ctq_ratio = {ctq_ratio:.3f}
- transport_heavy: C3_macro = {summary.get('transport_heavy', {}).get('C3_final_mean')}
- local_heavy: C3_macro = {summary.get('local_heavy', {}).get('C3_final_mean')}
- **Path {path}:** {path_text}
- `code/genome_160_transport_guided_student.py` -> `results/genome_160_transport_guided_student.json`
"""
    print("\n=== WIKI patch ===")
    print(wiki_patch)

    if "--commit" in sys.argv:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(entry) + "\n")
        wiki_path = ROOT / "research" / "g160_wiki_patch.md"
        wiki_path.write_text(wiki_patch, encoding="utf-8")
        print(f"\nAppended to {LEDGER}")
        print(f"WIKI patch stashed at {wiki_path}")


if __name__ == "__main__":
    main()
