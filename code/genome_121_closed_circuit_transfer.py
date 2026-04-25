"""
genome_121_closed_circuit_transfer.py

CLOSED-CIRCUIT COMPOUND TRANSFER ON QWEN3-0.6B.

genome_116-120 exhausted the simple surgery ladder:
  - activation PC1 injection transfers within-model only
  - cross-model / cross-checkpoint injection is null
  - every single-component weight transplant is null on Pythia-160M
  - every single-component weight transplant is null on Qwen3-0.6B
  - all_attn is the only non-negative Qwen3 component (+0.63%, CI_lo > 0)

Interpretation: donor weights need BOTH the donor token basis and a closed
subcircuit. A donor attention stack dropped into a random model is forced to
consume random embeddings and coexist with random MLP branches. That is an
open-circuit transplant.

Genome 121 tests the next strongest zero-step hypothesis:
  donor embeddings + donor attention + donor norm glue + silenced random MLP
  may form a viable partial compiler, even when the rest of the model remains
  untrained.

Primary arm:
  - embed_attn_ln_zero_mlp

Secondary arms:
  - all_attn                    (genome_120 anchor)
  - embed_only                  (negative anchor)
  - embed_attn                  (does donor token basis rescue attention?)
  - embed_attn_ln               (do norm tensors matter?)
  - zero_mlp_only               (is silencing alone helpful?)
  - embed_mlp                   (contrast: donor MLP with donor embedding)
  - embed_mlp_ln                (contrast + norm glue)
  - embed_mlp_ln_zero_attn      (closed-circuit MLP contrast)
  - zero_attn_only              (is silencing alone helpful?)
  - full_exact                  (exact donor copy positive control)

Pass:
  - primary arm closes >=20% of the donor-recipient NLL gap with CI_lo > 0
  - and beats all_attn by >=5 percentage points
  - and full_exact reproduces donor NLL within 0.02 nats

Partial:
  - any embed+attn arm closes >=5% gap with CI_lo > 0
  - and beats all_attn by >=3 percentage points

Kill:
  - all non-full compound arms close <5% of the gap or fail to beat all_attn

Results:
  results/genome_121_closed_circuit_transfer.json
"""

import copy
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
BASE_SEED = 42

SEEDS = [7, 13, 23, 42, 101]
N_EVAL = 200
SEQ_LEN = 64
BATCH = 8
N_BOOT = 500

PASS_GAP = 20.0
PARTIAL_GAP = 5.0
PRIMARY_DELTA_PASS = 5.0
PRIMARY_DELTA_PARTIAL = 3.0
FULL_EXACT_TOL = 0.02

GROUPS = {
    "embed": [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ],
    "all_attn": [
        ".self_attn.",
    ],
    "all_mlp": [
        ".mlp.",
    ],
    "all_ln": [
        "input_layernorm",
        "post_attention_layernorm",
        "model.norm.weight",
    ],
}

ARMS = [
    {
        "name": "embed_only",
        "copy_groups": ["embed"],
        "zero_groups": [],
        "kind": "compound",
    },
    {
        "name": "all_attn",
        "copy_groups": ["all_attn"],
        "zero_groups": [],
        "kind": "compound",
    },
    {
        "name": "embed_attn",
        "copy_groups": ["embed", "all_attn"],
        "zero_groups": [],
        "kind": "compound",
    },
    {
        "name": "embed_attn_ln",
        "copy_groups": ["embed", "all_attn", "all_ln"],
        "zero_groups": [],
        "kind": "compound",
    },
    {
        "name": "embed_attn_ln_zero_mlp",
        "copy_groups": ["embed", "all_attn", "all_ln"],
        "zero_groups": ["all_mlp"],
        "kind": "primary",
    },
    {
        "name": "zero_mlp_only",
        "copy_groups": [],
        "zero_groups": ["all_mlp"],
        "kind": "control",
    },
    {
        "name": "embed_mlp",
        "copy_groups": ["embed", "all_mlp"],
        "zero_groups": [],
        "kind": "contrast",
    },
    {
        "name": "embed_mlp_ln",
        "copy_groups": ["embed", "all_mlp", "all_ln"],
        "zero_groups": [],
        "kind": "contrast",
    },
    {
        "name": "embed_mlp_ln_zero_attn",
        "copy_groups": ["embed", "all_mlp", "all_ln"],
        "zero_groups": ["all_attn"],
        "kind": "contrast",
    },
    {
        "name": "zero_attn_only",
        "copy_groups": [],
        "zero_groups": ["all_attn"],
        "kind": "control",
    },
    {
        "name": "full_exact",
        "copy_groups": ["__full_exact__"],
        "zero_groups": [],
        "kind": "control",
    },
]


def load_wikitext(n, offset, seed=BASE_SEED):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    out, count = [], 0
    for idx in perm:
        text = ds[int(idx)]["text"].strip()
        if len(text) < 60:
            continue
        if count >= offset:
            out.append(text[:300])
        count += 1
        if len(out) >= n:
            break
    return out


def tokenize(texts, tok):
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN,
    )
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def load_trained():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(DEVICE).eval()
    return model, tok


def load_random_init(seed):
    cfg = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).to(DEVICE).eval()
    return model


def _matches_group(key, group_name):
    for token in GROUPS[group_name]:
        if key == token or token in key:
            return True
    return False


def _unique_numel(tensor, seen_ptrs):
    ptr = tensor.untyped_storage().data_ptr()
    if ptr in seen_ptrs:
        return 0
    seen_ptrs.add(ptr)
    return tensor.numel()


def apply_arm(donor, recipient, arm):
    model_copy = copy.deepcopy(recipient)

    if arm["name"] == "full_exact":
        model_copy.load_state_dict(donor.state_dict(), strict=True)
        total = sum(p.numel() for p in donor.parameters())
        return model_copy, total, 0, total

    sd_donor = donor.state_dict()
    sd_model = model_copy.state_dict()
    copied = 0
    zeroed = 0
    seen_copied = set()
    seen_zeroed = set()

    for key in list(sd_model.keys()):
        if any(_matches_group(key, grp) for grp in arm["copy_groups"]):
            sd_model[key] = sd_donor[key].clone()
            copied += _unique_numel(sd_donor[key], seen_copied)

    for key in list(sd_model.keys()):
        if any(_matches_group(key, grp) for grp in arm["zero_groups"]):
            sd_model[key] = torch.zeros_like(sd_model[key])
            zeroed += _unique_numel(sd_model[key], seen_zeroed)

    model_copy.load_state_dict(sd_model, strict=True)
    total = sum(p.numel() for p in donor.parameters())
    return model_copy, copied, zeroed, total


def logits_to_nll(logits, input_ids, attention_mask):
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous().clone()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels[shift_mask == 0] = -100
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
        ignore_index=-100,
    ).item()


def measure_nll(model, tok, texts):
    per_seq = []
    for i in range(0, len(texts), BATCH):
        ids, mask = tokenize(texts[i:i + BATCH], tok)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        for j in range(ids.shape[0]):
            per_seq.append(logits_to_nll(out.logits[j:j + 1], ids[j:j + 1], mask[j:j + 1]))
    return np.array(per_seq)


def bsci(arr, n_boot=N_BOOT, seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    boots = [
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ]
    return {
        "mean": float(arr.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def bootstrap_gap_closed(recip, donor, surgery, n_boot=N_BOOT, seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    n = len(recip)

    def point(r, d, s):
        gap = r.mean() - d.mean()
        if abs(gap) < 1e-6:
            return 0.0
        return float((r.mean() - s.mean()) / gap * 100.0)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(point(recip[idx], donor[idx], surgery[idx]))
    return {
        "mean": float(point(recip, donor, surgery)),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def make_verdict(aggregate_results, donor_nll, full_exact_stats):
    all_attn_gap = aggregate_results["all_attn"]["gap_closed_pct"]["mean"]

    primary = aggregate_results["embed_attn_ln_zero_mlp"]["gap_closed_pct"]
    primary_delta = primary["mean"] - all_attn_gap

    embed_attn_names = ("embed_attn", "embed_attn_ln", "embed_attn_ln_zero_mlp")
    best_embed_attn_name = max(
        embed_attn_names,
        key=lambda name: aggregate_results[name]["gap_closed_pct"]["mean"],
    )
    best_embed_attn = aggregate_results[best_embed_attn_name]["gap_closed_pct"]
    best_embed_attn_delta = best_embed_attn["mean"] - all_attn_gap
    best_nonfull = max(
        (
            res["gap_closed_pct"]
            for name, res in aggregate_results.items()
            if name != "full_exact"
        ),
        key=lambda x: x["mean"],
    )

    full_exact_error = abs(full_exact_stats["mean"] - donor_nll["mean"])

    if (
        primary["mean"] >= PASS_GAP
        and primary["ci_lo"] > 0
        and primary_delta >= PRIMARY_DELTA_PASS
        and full_exact_error <= FULL_EXACT_TOL
    ):
        return (
            "PASS: closed donor attention circuit closes "
            f"{primary['mean']:.1f}% of the gap "
            f"(CI [{primary['ci_lo']:.1f}, {primary['ci_hi']:.1f}]), "
            f"beats all_attn by {primary_delta:.1f} points, "
            f"and full_exact matches donor within {full_exact_error:.4f} nats."
        )

    if (
        best_embed_attn["mean"] >= PARTIAL_GAP
        and best_embed_attn["ci_lo"] > 0
        and best_embed_attn_delta >= PRIMARY_DELTA_PARTIAL
    ):
        return (
            "PARTIAL: donor-basis attention transfer shows a real rescue signal. "
            f"Best embed+attn arm ({best_embed_attn_name}) closes "
            f"{best_embed_attn['mean']:.1f}% of the gap and beats all_attn by "
            f"{best_embed_attn_delta:.1f} points."
        )

    return (
        "KILL: closed-circuit compound transfer does not break the holism barrier. "
        f"Best non-full arm closes {best_nonfull['mean']:.1f}% of the gap; "
        f"primary delta vs all_attn = {primary_delta:.1f} points."
    )


def main():
    t0 = time.time()
    print(f"genome_121: closed-circuit compound transfer on {MODEL_ID}")
    print(f"  seeds={SEEDS}  n_eval={N_EVAL}  batch={BATCH}")

    eval_texts = load_wikitext(N_EVAL, offset=0)
    print(f"  loaded {len(eval_texts)} eval sequences")

    print("Loading donor...")
    donor, tok = load_trained()
    donor_nlls = measure_nll(donor, tok, eval_texts)
    donor_stats = bsci(donor_nlls)
    print(f"  donor NLL={donor_stats['mean']:.4f}")

    per_seed = []
    pooled_recip = []
    pooled_donor = []
    pooled_by_arm = {arm["name"]: [] for arm in ARMS}
    arm_copy_stats = {}

    for seed in SEEDS:
        print(f"\n=== Recipient seed {seed} ===")
        recipient = load_random_init(seed)
        recip_nlls = measure_nll(recipient, tok, eval_texts)
        recip_stats = bsci(recip_nlls, seed=seed)
        print(f"  recipient NLL={recip_stats['mean']:.4f}")

        seed_row = {
            "seed": seed,
            "recipient_nll": recip_stats,
            "results_by_arm": {},
        }

        pooled_recip.append(recip_nlls)
        pooled_donor.append(donor_nlls)

        for arm in ARMS:
            print(f"  -- arm: {arm['name']}")
            model_copy, n_copied, n_zeroed, total = apply_arm(donor, recipient, arm)
            surg_nlls = measure_nll(model_copy, tok, eval_texts)
            surg_stats = bsci(surg_nlls, seed=seed)
            gap = bootstrap_gap_closed(recip_nlls, donor_nlls, surg_nlls, seed=seed)

            copied_pct = n_copied / total * 100.0
            zeroed_pct = n_zeroed / total * 100.0
            print(
                f"     copied={copied_pct:6.2f}% zeroed={zeroed_pct:6.2f}% "
                f"NLL={surg_stats['mean']:.4f} gap_closed={gap['mean']:.2f}%"
            )

            if arm["name"] not in arm_copy_stats:
                arm_copy_stats[arm["name"]] = {
                    "params_copied_pct": copied_pct,
                    "params_zeroed_pct": zeroed_pct,
                    "kind": arm["kind"],
                }

            seed_row["results_by_arm"][arm["name"]] = {
                "params_copied_pct": copied_pct,
                "params_zeroed_pct": zeroed_pct,
                "surgery_nll": surg_stats,
                "gap_closed_pct": gap,
            }
            pooled_by_arm[arm["name"]].append(surg_nlls)

            del model_copy
            torch.cuda.empty_cache()

        per_seed.append(seed_row)
        del recipient
        torch.cuda.empty_cache()

    pooled_recip = np.concatenate(pooled_recip)
    pooled_donor = np.concatenate(pooled_donor)
    aggregate_results = {}

    for arm in ARMS:
        name = arm["name"]
        pooled_arm = np.concatenate(pooled_by_arm[name])
        aggregate_results[name] = {
            "kind": arm_copy_stats[name]["kind"],
            "params_copied_pct": arm_copy_stats[name]["params_copied_pct"],
            "params_zeroed_pct": arm_copy_stats[name]["params_zeroed_pct"],
            "surgery_nll": bsci(pooled_arm),
            "gap_closed_pct": bootstrap_gap_closed(pooled_recip, pooled_donor, pooled_arm),
        }

    recip_seed_means = np.array([row["recipient_nll"]["mean"] for row in per_seed])
    recipient_stats = {
        "mean_of_seed_means": float(recip_seed_means.mean()),
        "seed_means": [float(x) for x in recip_seed_means],
        "ci_across_seeds_lo": float(np.percentile(recip_seed_means, 2.5)),
        "ci_across_seeds_hi": float(np.percentile(recip_seed_means, 97.5)),
    }

    full_exact_stats = aggregate_results["full_exact"]["surgery_nll"]
    verdict = make_verdict(aggregate_results, donor_stats, full_exact_stats)

    print("\n=== GENOME 121 SUMMARY ===")
    print(f"  donor NLL:     {donor_stats['mean']:.4f}")
    print(f"  recipient NLL: {recipient_stats['mean_of_seed_means']:.4f} (seed-mean)")
    for name, res in aggregate_results.items():
        print(
            f"  {name:26s} copied={res['params_copied_pct']:6.2f}% "
            f"zeroed={res['params_zeroed_pct']:6.2f}% "
            f"NLL={res['surgery_nll']['mean']:.4f} "
            f"gap={res['gap_closed_pct']['mean']:.2f}%"
        )
    print(f"  verdict: {verdict}")

    out = {
        "model": MODEL_ID,
        "genome": 121,
        "name": "closed_circuit_transfer",
        "motivation": "test whether donor basis + donor attention + norm glue + silenced random branch breaks the holism barrier",
        "n_eval": N_EVAL,
        "seeds": SEEDS,
        "arms": ARMS,
        "donor_nll": donor_stats,
        "recipient_nll": recipient_stats,
        "criteria": {
            "pass_gap_pct": PASS_GAP,
            "partial_gap_pct": PARTIAL_GAP,
            "primary_delta_pass_pct_points": PRIMARY_DELTA_PASS,
            "primary_delta_partial_pct_points": PRIMARY_DELTA_PARTIAL,
            "full_exact_tolerance_nll": FULL_EXACT_TOL,
        },
        "aggregate_results": {
            name: {
                "kind": res["kind"],
                "params_copied_pct": res["params_copied_pct"],
                "params_zeroed_pct": res["params_zeroed_pct"],
                "surgery_nll": res["surgery_nll"],
                "gap_closed_pct": res["gap_closed_pct"],
            }
            for name, res in aggregate_results.items()
        },
        "per_seed": per_seed,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }

    out_path = RESULTS / "genome_121_closed_circuit_transfer.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
