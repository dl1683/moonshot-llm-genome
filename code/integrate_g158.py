"""
integrate_g158.py — mechanical g158 verdict integration.

Reads results/genome_158_context_length_inversion.json and:
  - prints summary (delta_per_L, Spearman rho, sign-pattern, verdict)
  - prints ledger entry to append (with --commit appends + stashes WIKI patch)

Per locked decision tree: PASS_INVERSION -> g158 promoted in CLAIM_EVIDENCE_MAP;
PARTIAL/KILL -> theory's input-side prediction questioned.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "results" / "genome_158_context_length_inversion.json"
LEDGER = ROOT / "experiments" / "ledger.jsonl"


def main():
    if not RESULT.exists():
        print(f"ERROR: {RESULT} not found; g158 not yet complete.")
        sys.exit(1)
    r = json.loads(RESULT.read_text(encoding="utf-8"))

    rho_c4 = r.get("spearman_rho_c4")
    rho_ood = r.get("spearman_rho_ood")
    d32_c4 = r.get("delta_32_c4")
    d256_c4 = r.get("delta_256_c4")
    sign_consistent = r.get("sign_consistent", False)
    verdict = r.get("verdict", "(missing)")
    delta_per_L = r.get("delta_per_L", {})

    if "PASS_INVERSION" in verdict:
        path = "PASS"
        path_text = ("PASS: theory's monotone-attenuation prediction validated. "
                     "Promote P13 -> C in CLAIM_EVIDENCE_MAP. §0.1 score 6/10 -> 7/10. "
                     "Next: g159 cross-class lesion.")
    elif "PARTIAL_INVERSION" in verdict:
        path = "PARTIAL"
        path_text = ("PARTIAL: direction supported but no clean sign flip. "
                     "g158 partial; consider extending to L=16 to find inversion "
                     "OR move on to g159 with caveat.")
    elif "KILL_INVERSION" in verdict:
        path = "KILL"
        path_text = ("KILL: theory's input-side prediction does NOT hold. "
                     "Both internal-mechanism (g157+g157b) AND input-side prediction "
                     "(g158) are rejected. The empirical g156 PASS becomes a single-axis "
                     "result; pivot to distillation track immediately.")
    else:
        path = "AMBIGUOUS"
        path_text = "AMBIGUOUS: see verdict text. Manual investigation."

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "id": "genome_158_context_length_inversion",
        "purpose": "g158 context-length inversion: tests theory's input-side prediction (transport demand control variable). Independent of η/δ probe (which was rejected by g157+g157b).",
        "git_commit": "(set at commit time)",
        "config_path": "code/genome_158_context_length_inversion.py",
        "prereg_path": "research/prereg/genome_158_context_length_inversion_2026-04-26.md",
        "systems": ["baseline_6L+MLP x 4 L x 3 seeds", "minimal_3L_noMLP x 4 L x 3 seeds"],
        "primitive": "context_length_inversion",
        "universality_level_claimed": None,
        "metrics": {
            "spearman_rho_c4": rho_c4,
            "spearman_rho_ood": rho_ood,
            "delta_32_c4": d32_c4,
            "delta_256_c4": d256_c4,
            "sign_consistent": sign_consistent,
            "delta_per_L_c4": {L: v.get("c4") for L, v in delta_per_L.items()},
            "delta_per_L_ood": {L: v.get("ood") for L, v in delta_per_L.items()},
            "verdict": verdict,
        },
        "artifacts": [
            "code/genome_158_context_length_inversion.py",
            "results/genome_158_context_length_inversion.json",
            "results/genome_158_run.log",
        ],
        "notes": f"Path {path}: {path_text}",
        "status": "completed",
    }

    print("=== g158 verdict ===")
    print(f"  rho_c4 = {rho_c4}")
    print(f"  rho_ood = {rho_ood}")
    print(f"  delta_32_c4 = {d32_c4}")
    print(f"  delta_256_c4 = {d256_c4}")
    print(f"  sign_consistent = {sign_consistent}")
    print(f"  verdict = {verdict[:120]}")
    print(f"\n  Decision: {path}")
    print(f"  {path_text}")

    print("\n=== delta_per_L ===")
    for L in sorted(delta_per_L.keys(), key=lambda x: int(x)):
        d = delta_per_L[L]
        print(f"  L={L:>3s}: dc4={d.get('c4'):+.2f}pp  dood={d.get('ood'):+.2f}pp  n_b={d.get('n_b')}  n_m={d.get('n_m')}")

    wiki_patch = f"""**genome_158 COMPLETED ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}): {verdict.split(':',1)[0]}**
- Spearman rho(L, delta_c4) = {rho_c4:+.3f}; rho(L, delta_ood) = {rho_ood:+.3f}
- delta at L=32: dc4={d32_c4:+.2f}pp; at L=256: dc4={d256_c4:+.2f}pp
- sign consistent across c4-val and wikitext-val: {sign_consistent}
- **Path {path}:** {path_text}
- `code/genome_158_context_length_inversion.py` -> `results/genome_158_context_length_inversion.json`
"""
    print("\n=== WIKI patch ===")
    print(wiki_patch)

    print("\nTo append to ledger.jsonl:")
    print("  python code/integrate_g158.py --commit")

    if "--commit" in sys.argv:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(entry) + "\n")
        wiki_path = ROOT / "research" / "g158_wiki_patch.md"
        wiki_path.write_text(wiki_patch, encoding="utf-8")
        print(f"\nAppended to {LEDGER}")
        print(f"WIKI patch stashed at {wiki_path}")


if __name__ == "__main__":
    main()
