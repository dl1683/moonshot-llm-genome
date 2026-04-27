"""
Early-help / no-persistence meta-audit.

Per Codex 2026-04-27 data-mining consult Section B#2: pool grafting_005-009 +
genome_125 + genome_134 + genome_137 to locate the maturity window where donor
signal helps before washout. CPU-only, runs while g158c is in flight.

Goal: identify the step range where donor-aided trajectories beat scratch
trajectories AND the step range where the gap closes ("washout"). If a clean
maturity window exists across experiments, the receiver-first / annealed-transfer
hypothesis is alive and a GPU experiment that decays donor signal at exactly that
step is the next high-leverage move.

Output: research/EARLY_HELP_META_AUDIT_2026-04-27.md (markdown table of per-
experiment peak/washout + pooled summary).
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent

# Per-experiment schema:
#   each entry has: file path, donor_arm_key, scratch_arm_key, label
#   donor_arm = the trajectory that received donor signal (mean-shift, init,
#               glue, optimizer state, etc.)
#   scratch_arm = the lesion / random-init / no-donor reference
EXPERIMENTS = [
    # grafting series (subproject)
    {
        "path": "grafting/results/grafting_005_ce_training_speedup.json",
        "donor_key": "trajectory_grafted",
        "scratch_key": "trajectory_lesion",
        "label": "g005_ce_grafted_init",
        "donor_mech": "ridge_grafted_init",
    },
    {
        "path": "grafting/results/grafting_006_tokenlevel_rank30_adapter_bootstrap.json",
        "donor_key": "trajectory_arm_b",
        "scratch_key": "trajectory_arm_a",
        "label": "g006_rank30_adapter",
        "donor_mech": "rank30_adapter",
    },
    {
        "path": "grafting/results/grafting_007_meanshift_speedup.json",
        "donor_key": "trajectory_arm_b",
        "scratch_key": "trajectory_arm_a",
        "label": "g007_meanshift",
        "donor_mech": "meanshift_init",
    },
    {
        "path": "grafting/results/grafting_008_trainable_meanshift_persistence.json",
        "donor_key": "trajectory_arm_b",
        "scratch_key": "trajectory_arm_a",
        "label": "g008_trainable_meanshift",
        "donor_mech": "trainable_meanshift",
    },
    {
        "path": "grafting/results/grafting_009_weightspace_seed.json",
        "donor_key": "trajectory_arm_b",
        "scratch_key": "trajectory_arm_a",
        "label": "g009_weightspace_seed",
        "donor_mech": "weightspace_seed",
    },
    # genome series (main)
    {
        "path": "results/genome_125_frozen_attn_glue_train.json",
        "donor_key": None,  # custom shape; handled below
        "scratch_key": None,
        "label": "g125_frozen_attn_glue",
        "donor_mech": "frozen_attn_glue",
    },
    {
        "path": "results/genome_134_glue_only_trajectory.json",
        "donor_key": None,
        "scratch_key": None,
        "label": "g134_glue_only",
        "donor_mech": "glue_only",
    },
    {
        "path": "results/genome_137_optimizer_state_transfer.json",
        "donor_key": None,
        "scratch_key": None,
        "label": "g137_optimizer_state",
        "donor_mech": "optimizer_state",
    },
]


def to_step_indexed(traj_obj):
    """Coerce a trajectory dict-or-list into {int_step: float_nll}."""
    if traj_obj is None:
        return None
    if isinstance(traj_obj, dict):
        # Keys may be strings of ints
        out = {}
        for k, v in traj_obj.items():
            try:
                kk = int(k)
            except (ValueError, TypeError):
                continue
            if isinstance(v, (int, float)):
                out[kk] = float(v)
            elif isinstance(v, dict):
                # try common keys
                for pref in ("nll", "loss", "val_nll", "mean_nll"):
                    if pref in v:
                        out[kk] = float(v[pref])
                        break
        return out if out else None
    if isinstance(traj_obj, list):
        # Common: list of (step, nll) pairs OR list of {step, nll} dicts OR list of nlls
        if not traj_obj:
            return None
        out = {}
        for i, item in enumerate(traj_obj):
            if isinstance(item, (int, float)):
                out[i] = float(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out[int(item[0])] = float(item[1])
            elif isinstance(item, dict):
                step = item.get("step", i)
                for pref in ("nll", "loss", "val_nll", "mean_nll"):
                    if pref in item:
                        out[int(step)] = float(item[pref])
                        break
        return out if out else None
    return None


def extract_genome_125(d):
    """g125 has donor_nll (single value) + random_init_nll (single) + eval_steps + (presumably) per-step values somewhere."""
    out_donor = {}
    out_scratch = {}
    # Look for arrays
    for k, v in d.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
            if "donor" in k.lower() and "nll" in k.lower():
                out_donor = {i: float(x) for i, x in enumerate(v)}
            elif ("random" in k.lower() or "scratch" in k.lower() or "init" in k.lower()) and "nll" in k.lower():
                out_scratch = {i: float(x) for i, x in enumerate(v)}
    return (out_donor or None, out_scratch or None)


def extract_generic(d, donor_key, scratch_key):
    return to_step_indexed(d.get(donor_key)), to_step_indexed(d.get(scratch_key))


def analyze_pair(donor, scratch, label):
    """Compute donor_advantage(step) = scratch[step] - donor[step] (positive = donor helps).
    Find peak step + washout step.
    """
    if not donor or not scratch:
        return None
    common = sorted(set(donor.keys()) & set(scratch.keys()))
    if len(common) < 3:
        return None
    advantages = [(s, scratch[s] - donor[s]) for s in common]
    # Peak: step with max positive advantage
    peak_step, peak_adv = max(advantages, key=lambda x: x[1])
    # Washout: first step AFTER peak where advantage drops to <= 25% of peak_adv
    washout_step = None
    if peak_adv > 0:
        threshold = 0.25 * peak_adv
        for s, adv in advantages:
            if s > peak_step and adv <= threshold:
                washout_step = s
                break
    # Also report final-step advantage
    final_step, final_adv = advantages[-1]
    initial_adv = advantages[0][1]
    return {
        "label": label,
        "n_points": len(common),
        "step_range": (common[0], common[-1]),
        "initial_advantage_nats": initial_adv,
        "peak_step": peak_step,
        "peak_advantage_nats": peak_adv,
        "washout_step": washout_step,
        "final_step": final_step,
        "final_advantage_nats": final_adv,
        "fraction_decayed": (peak_adv - final_adv) / peak_adv if peak_adv > 0 else None,
    }


def main():
    rows = []
    for ex in EXPERIMENTS:
        path = ROOT / ex["path"]
        if not path.exists():
            rows.append({"label": ex["label"], "status": "DATA_MISSING"})
            continue
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            rows.append({"label": ex["label"], "status": f"PARSE_ERROR: {e}"})
            continue

        if ex["label"].startswith("g125"):
            donor, scratch = extract_genome_125(d)
        elif ex["label"].startswith("g134") or ex["label"].startswith("g137"):
            # g134/g137 schemas are unusual; try a generic scan
            donor, scratch = None, None
            # g137 has donor_nll_K_per_seed
            if "donor_nll_K_per_seed" in d:
                # Format: {seed: {K: nll}} or similar
                v = d["donor_nll_K_per_seed"]
                if isinstance(v, dict):
                    for s_key, k_dict in v.items():
                        if isinstance(k_dict, dict):
                            for k_key, nll in k_dict.items():
                                try:
                                    donor = donor or {}
                                    donor[int(k_key)] = float(nll)
                                except (ValueError, TypeError):
                                    continue
                            break
        else:
            donor, scratch = extract_generic(d, ex["donor_key"], ex["scratch_key"])

        if donor is None or scratch is None:
            rows.append({"label": ex["label"], "status": "TRAJECTORY_UNAVAILABLE", "donor_mech": ex["donor_mech"]})
            continue

        result = analyze_pair(donor, scratch, ex["label"])
        if result is None:
            rows.append({"label": ex["label"], "status": "INSUFFICIENT_OVERLAP", "donor_mech": ex["donor_mech"]})
            continue
        result["status"] = "OK"
        result["donor_mech"] = ex["donor_mech"]
        rows.append(result)

    # Pooled stats over experiments with status OK
    ok_rows = [r for r in rows if r.get("status") == "OK"]
    pooled = {}
    if ok_rows:
        peaks = [r["peak_advantage_nats"] for r in ok_rows]
        peak_steps = [r["peak_step"] for r in ok_rows]
        finals = [r["final_advantage_nats"] for r in ok_rows]
        decay_fracs = [r["fraction_decayed"] for r in ok_rows if r.get("fraction_decayed") is not None]
        pooled = {
            "n": len(ok_rows),
            "mean_peak_advantage_nats": mean(peaks),
            "stdev_peak_advantage_nats": stdev(peaks) if len(peaks) > 1 else None,
            "mean_peak_step": mean(peak_steps),
            "stdev_peak_step": stdev(peak_steps) if len(peak_steps) > 1 else None,
            "mean_final_advantage_nats": mean(finals),
            "mean_decay_fraction": mean(decay_fracs) if decay_fracs else None,
            "n_with_washout": sum(1 for r in ok_rows if r.get("washout_step") is not None),
            "n_persisting": sum(1 for r in ok_rows if r.get("washout_step") is None and r["final_advantage_nats"] > 0.05),
        }

    return {"rows": rows, "pooled": pooled}


if __name__ == "__main__":
    out = main()
    import pprint
    pprint.pprint(out, width=120)
