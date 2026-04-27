"""
integrate_g158c.py — mechanical g158c canonical-verdict integration.

Reads results/genome_158c_3seed_canonical.json and:
  - prints summary (per-seed delta_per_L, mean rho, mean Delta_256, sign of Delta_32)
  - applies decision tree from research/programs/post_g158c_decision_tree.md
  - emits ledger entry + WIKI patch (with --commit appends ledger and stashes patch)

Decision rules (carried from PILOT, now over 3 seeds):
  PASS_canonical:
    rho mean across 3 seeds >= +0.8 AND
    Delta_256(c4) 95% CI excludes zero AND mean >= +2.0pp AND
    Delta_32(c4) sign matches PILOT (negative or zero, never strongly positive)
  WEAK_canonical:
    rho mean >= +0.5 AND Delta_256 mean >= +1.0pp AND Delta_256 CI does not flip sign
  PILOT_FRAGILE:
    otherwise (rho < +0.5, OR Delta_256 CI crosses zero, OR Delta_32 strongly positive)
"""
from __future__ import annotations
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "results" / "genome_158c_3seed_canonical.json"
LEDGER = ROOT / "experiments" / "ledger.jsonl"


def t_ci(values, alpha=0.05):
    """Approx 95% CI on mean using student-t at n=3 (t=4.303). Returns (lo, hi)."""
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    m = mean(values)
    s = stdev(values)
    se = s / math.sqrt(n)
    # t critical at n=3 (df=2) alpha=0.05 two-sided = 4.303
    t_crit = {2: 4.303, 3: 3.182, 4: 2.776}.get(n - 1, 2.0)
    return (m - t_crit * se, m + t_crit * se)


def main():
    if not RESULT.exists():
        print(f"ERROR: {RESULT} not found; g158c not yet complete.")
        sys.exit(1)
    r = json.loads(RESULT.read_text(encoding="utf-8"))

    results = r.get("results", {})
    seeds = r.get("config", {}).get("seeds", [])
    Ls = sorted(int(L) for L in results.keys())

    # Compute per-seed delta_per_L on c4, then aggregate
    per_seed_deltas = {}  # seed -> {L: delta_c4}
    per_seed_rho = {}  # seed -> spearman approx (sign-rank correlation)
    for seed in seeds:
        s_str = str(seed)
        per_seed_deltas[seed] = {}
        for L in Ls:
            cell = results.get(str(L), {})
            base = cell.get("baseline_6L+MLP", {}).get(s_str, {})
            mini = cell.get("minimal_3L_noMLP", {}).get(s_str, {})
            base_top1 = base.get("c4", {}).get("top1_acc", float("nan"))
            mini_top1 = mini.get("c4", {}).get("top1_acc", float("nan"))
            # Codex cycle 24 SEV7 fix: fail loudly on NaN/missing rather than
            # silently emitting PILOT_FRAGILE on a corrupt JSON.
            if not math.isfinite(base_top1) or not math.isfinite(mini_top1):
                raise RuntimeError(
                    f"incomplete/corrupt g158c result: L={L} seed={seed} "
                    f"baseline_c4={base_top1!r} minimal_c4={mini_top1!r}"
                )
            delta = (mini_top1 - base_top1) * 100.0  # pp
            per_seed_deltas[seed][L] = delta
        # Spearman rho between L (rank) and delta (rank) — for monotone with 4 points
        ranks_L = list(range(len(Ls)))
        deltas = [per_seed_deltas[seed][L] for L in Ls]
        ranks_d = sorted(range(len(deltas)), key=lambda i: deltas[i])
        rank_of_d = [0] * len(deltas)
        for r_i, idx in enumerate(ranks_d):
            rank_of_d[idx] = r_i
        # Spearman = 1 - 6*sum(d_i^2) / (n*(n^2-1))
        n = len(Ls)
        d2 = sum((ranks_L[i] - rank_of_d[i]) ** 2 for i in range(n))
        rho = 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else float("nan")
        per_seed_rho[seed] = rho

    # Aggregate
    mean_rho = mean(per_seed_rho.values())
    mean_d256 = mean(per_seed_deltas[s][256] for s in seeds)
    mean_d32 = mean(per_seed_deltas[s][32] for s in seeds)
    ci_d256 = t_ci([per_seed_deltas[s][256] for s in seeds])
    ci_d32 = t_ci([per_seed_deltas[s][32] for s in seeds])

    # Decision rule.
    # Codex audit SEV8 (2026-04-27): the d32 threshold here previously allowed
    # +0.5pp, which conflicts with both the script's own PASS gate (d32 <= -0.2pp)
    # and the decision-tree wording ("negative or zero"). Tighten to <= 0 to match
    # the decision tree exactly (Path A: "Delta_32 sign matches PILOT (negative
    # or zero, never strongly positive)"). This is the authoritative rule.
    if (
        mean_rho >= 0.8
        and ci_d256[0] > 0
        and mean_d256 >= 2.0
        and mean_d32 <= 0.0  # negative or zero per decision tree (no positive d32 allowed)
    ):
        path = "PASS_canonical"
        path_text = (
            "PASS: theory's input-side prediction LOCKS at canonical 3-seed scale. "
            "Promote P13 -> C17 (canonical). §0.1 score 6.8 -> 7.2. "
            "Next: causal mechanism probe (g162 or g163 — Codex direction review)."
        )
    elif mean_rho >= 0.5 and mean_d256 >= 1.0 and ci_d256[0] > 0:
        path = "WEAK_canonical"
        path_text = (
            "WEAK: signal softens at canonical scale but holds. "
            "Mark P13 -> WEAK_CANONICAL. §0.1 score 6.8 -> 7.0. "
            "Consider g158d budget-expansion (Codex sign-off required)."
        )
    else:
        path = "PILOT_FRAGILE"
        path_text = (
            "PILOT_FRAGILE: PILOT signal does NOT survive multi-seed. "
            "Mark P13 -> R8b (rejected). §0.1 score 6.8 -> 6.0. "
            "Pivot: abandon transport theory; treat g156 as standalone narrow ablation."
        )

    verdict = f"{path}: mean_rho={mean_rho:+.2f}, mean_D256={mean_d256:+.2f}pp, mean_D32={mean_d32:+.2f}pp, CI_D256=[{ci_d256[0]:+.2f}, {ci_d256[1]:+.2f}]"

    print("=== g158c 3-seed canonical verdict ===")
    print(f"  seeds = {seeds}")
    for s in seeds:
        print(f"  seed={s}: rho={per_seed_rho[s]:+.2f}  "
              f"D32={per_seed_deltas[s][32]:+.2f}pp  "
              f"D64={per_seed_deltas[s][64]:+.2f}pp  "
              f"D128={per_seed_deltas[s][128]:+.2f}pp  "
              f"D256={per_seed_deltas[s][256]:+.2f}pp")
    print(f"\n  mean_rho      = {mean_rho:+.3f}")
    print(f"  mean_D256(c4) = {mean_d256:+.2f}pp  (95% CI: [{ci_d256[0]:+.2f}, {ci_d256[1]:+.2f}])")
    print(f"  mean_D32(c4)  = {mean_d32:+.2f}pp  (95% CI: [{ci_d32[0]:+.2f}, {ci_d32[1]:+.2f}])")
    print(f"\n  VERDICT: {verdict}")
    print(f"  PATH   : {path_text}")

    elapsed_s = r.get("elapsed_s", 0)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "id": "genome_158c_3seed_canonical",
        "purpose": "g158c canonical 3-seed verdict of context-length inversion: confirms theory's input-side prediction (transport demand as control variable) at canonical scale.",
        "git_commit": "(set at commit time)",
        "config_path": "code/genome_158c_3seed_canonical.py",
        "prereg_path": "research/prereg/genome_158c_3seed_canonical_2026-04-27.md",
        "systems": ["baseline_6L+MLP x 4 L x 3 seeds", "minimal_3L_noMLP x 4 L x 3 seeds"],
        "primitive": "context_length_inversion",
        "universality_level_claimed": None,
        "metrics": {
            "mean_rho": mean_rho,
            "mean_D256_c4": mean_d256,
            "mean_D32_c4": mean_d32,
            "CI_D256_c4_lo": ci_d256[0],
            "CI_D256_c4_hi": ci_d256[1],
            "CI_D32_c4_lo": ci_d32[0],
            "CI_D32_c4_hi": ci_d32[1],
            "per_seed_rho": per_seed_rho,
            "per_seed_D256_c4": {s: per_seed_deltas[s][256] for s in seeds},
            "verdict": verdict,
        },
        "artifacts": [
            "code/genome_158c_3seed_canonical.py",
            "results/genome_158c_3seed_canonical.json",
            "results/genome_158c_run.log",
        ],
        "notes": f"Path {path}: {path_text}. Wall {elapsed_s/3600:.1f}h.",
        "status": "completed",
    }

    wiki_patch = f"""**genome_158c COMPLETED ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}): {path}**
- mean_rho across 3 seeds = {mean_rho:+.3f}
- mean Delta_256(c4) = {mean_d256:+.2f}pp (95% CI: [{ci_d256[0]:+.2f}, {ci_d256[1]:+.2f}])
- mean Delta_32(c4)  = {mean_d32:+.2f}pp (95% CI: [{ci_d32[0]:+.2f}, {ci_d32[1]:+.2f}])
- **Path {path}:** {path_text}
- Wall: {elapsed_s/3600:.1f}h.
- `code/genome_158c_3seed_canonical.py` -> `results/genome_158c_3seed_canonical.json`
"""
    print("\n=== WIKI patch ===")
    print(wiki_patch)

    print("\nTo append to ledger.jsonl + stash WIKI patch:")
    print("  python code/integrate_g158c.py --commit")

    if "--commit" in sys.argv:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(entry) + "\n")
        wiki_path = ROOT / "research" / "g158c_wiki_patch.md"
        wiki_path.write_text(wiki_patch, encoding="utf-8")
        print(f"\nAppended to {LEDGER}")
        print(f"WIKI patch stashed at {wiki_path}")


if __name__ == "__main__":
    main()
