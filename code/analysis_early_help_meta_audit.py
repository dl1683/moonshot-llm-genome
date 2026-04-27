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
    """g125 schema: arm_results[arm].nll_curve = list-of-{step,nll} OR list-of-floats keyed
    by eval_steps. Donor arm = frozen_attn_glue; scratch arm = matched_param_ctrl
    (the like-for-like no-donor lesion). full_train_ctrl is a stronger sanity baseline,
    NOT the right scratch comparator. Codex cycle 27 SEV8 fix 2026-04-27."""
    arm_results = d.get("arm_results", {})
    eval_steps = d.get("eval_steps", [])
    donor_arm = arm_results.get("frozen_attn_glue", {})
    scratch_arm = arm_results.get("matched_param_ctrl", {})

    def _curve(arm_data):
        curve = arm_data.get("nll_curve")
        if curve is None:
            return None
        if isinstance(curve, dict):
            # {step: {mean, ci_lo, ci_hi, ...}} OR {step: float}
            out = {}
            for k, v in curve.items():
                try:
                    step = int(k)
                except (ValueError, TypeError):
                    continue
                if isinstance(v, dict) and "mean" in v:
                    out[step] = float(v["mean"])
                elif isinstance(v, (int, float)):
                    out[step] = float(v)
            return out if out else None
        if isinstance(curve, list) and curve:
            if isinstance(curve[0], dict):
                return {int(item.get("step", i)): float(item["nll"]) for i, item in enumerate(curve) if "nll" in item}
            if isinstance(curve[0], (int, float)):
                if len(curve) == len(eval_steps):
                    return {int(eval_steps[i]): float(curve[i]) for i in range(len(curve))}
                return {i: float(x) for i, x in enumerate(curve)}
        return None

    return (_curve(donor_arm), _curve(scratch_arm))


def extract_genome_137(d):
    """g137 schema: rows_per_seed_per_arm[seed][arm] = list-of-{step,nll}.
    Codex cycle 27 SEV-direction fix 2026-04-27: scratch arm should be
    resume_reset (state-zeroed, isolates the value of optimizer state), NOT
    state_only (catastrophic baseline). The clean comparison is
    donor=resume_true vs scratch=resume_reset. With this comparator g137 shows
    monotone decay (1064: +0.0456 -> 4000: -0.0004), consistent with washout."""
    rps = d.get("rows_per_seed_per_arm", {})
    if not rps:
        return None, None

    def _avg_curve(arm_name):
        per_seed_curves = {}
        for seed, arms in rps.items():
            rows = arms.get(arm_name, [])
            if not rows:
                continue
            for item in rows:
                if not isinstance(item, dict) or "nll" not in item:
                    continue
                step = int(item.get("step", 0))
                per_seed_curves.setdefault(step, []).append(float(item["nll"]))
        if not per_seed_curves:
            return None
        return {step: sum(vs) / len(vs) for step, vs in per_seed_curves.items()}

    return (_avg_curve("resume_true"), _avg_curve("resume_reset"))


def extract_genome_134(d):
    """g134 has only a single-arm 'rows' trajectory. No donor vs scratch comparison.
    Skip — does not fit the audit pattern."""
    return (None, None)


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
        elif ex["label"].startswith("g134"):
            donor, scratch = extract_genome_134(d)
        elif ex["label"].startswith("g137"):
            donor, scratch = extract_genome_137(d)
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
