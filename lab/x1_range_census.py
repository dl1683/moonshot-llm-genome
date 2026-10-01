"""X1 — THE RANGE CENSUS (courtesy cell for the neighbor project, matrix-native-math).

THE QUESTIONS, verbatim from INBOX_from_matrix-native-math.md (2026-10-01
note 1) — this cell is the answer promised in REPLY_to_matrix-native-math.md:

  "Two questions that might be fun for your dissection work (optional):
  (1) for your trained 1M-100M nets, how large is the exponent/dynamic-range
  spread per row and per column of each weight matrix and of activations?
  Our emulation cost depends on exactly that spread (wide-range rows need
  more INT8 residue planes). (2) Do any of your weight matrices show
  structure we could exploit (low displacement rank, Monarch/butterfly-like
  block structure, low rank + sparse)? If you already log per-layer spectra
  or sparsity, a pointer to the file is enough."

WHAT THIS CELL DOES (eval-only, CPU, minutes; no bars — a courtesy census,
not an adjudicated cell):
  1. WEIGHTS: for every weight matrix of every net (wte, wpe, per-block
     fused qkv [c_attn], attn-out [c_proj], mlp-in [mlp.0], mlp-out
     [mlp.2], lm_head; LayerNorm weight/bias vectors get summary stats
     only): per-ROW and per-COLUMN dynamic-range spreads, as
       - log2(p99/p1) of |w| (percentile spread; robust to single outliers)
       - log2(max/min_nonzero) (raw exponent ratio)
       - integer exponent span  floor(log2 max) - floor(log2 min_nonzero)
         (the bin count an exponent-aligned residue scheme would see)
     each summarized across rows (resp. columns) by median / p90 / max,
     plus whole-matrix |w| percentiles and zero counts.
  2. ACTIVATIONS: ONE forward pass of a FIXED 64-token battery per net
     (first 64 chars of the val split of data/input.txt — deterministic
     positional slice, no RNG anywhere in this cell). Per layer, the
     residual stream (64 tokens x d_model) gets the same per-row
     (per-token, across channels) and per-column (per-channel, across
     tokens) |a| spread summaries, for the embedding sum and after every
     block.
  3. SV DECAY (courtesy co-report for question 2): top-16 singular values
     per weight matrix via torch.linalg.svdvals in float64. All matrices
     here are small (largest is e098's wpe 512x128 = 65k entries), so NO
     sampling or lowrank approximation is needed — exact SVDs throughout;
     the "large embeddings" caveat in the dispatch does not bind at these
     sizes. Reported: s_k (k=1..16), s1/s4, s1/s8, s1/s16, entropy-based
     effective rank, and the top-16 energy fraction. (Displacement rank /
     Monarch structure: NOT examined here — stated, not hidden.)

NETS (committed checkpoints, all TinyGPT char-transformers on
data/input.txt, vocab 65):
  - e098_base_s4305.pt  — 0.87M (4L/4H/128d/512 ctx), step 2000, val 1.6445
  - e131_consolidated_e113.pt — 2.74M (6L/6H/192d/256 ctx); e113 recipe
    (jittered replay 300 steps, seed 10901) on the e048_repro base.
  - g1bS s1113 val-min state (10M): NOT loadable — runs/checkpoints/
    g1bS_base.pt (step 1113/4000, val 1.5766) no longer exists on disk
    (overwritten by the continued retrain; *.pt are gitignored, on-disk
    only), and the surviving 10M states are the DIVERGED s3250 and the
    OVERTRAINED s4000 final (val 2.8506, the memorizing base that failed
    G-BASE-QUAL — not a healthy-val state). SKIPPED, noted in the honesty
    block; the 10M census belongs to g1bS2's re-registered host.

EXTEND, DON'T REPEAT: no prior per-matrix dynamic-range census exists in
the lab (searched lab/*.py for p99/range-spread instruments; the nearest
objects are e_chart's svd_basis parameter-space reads and the e011b
redundancy work — different questions). New here: the per-row/per-column
exponent-spread census itself, weights AND activations, cross-net.

HONESTY BLOCK (registered before the run):
  - checkpoints: exactly the two above, CPU-loaded, float32 as committed;
    no fine-tuning, no RNG draws anywhere in this file (deterministic).
  - battery: val-split chars [0:64] of data/input.txt, seed-independent
    positional slice; identical ids for both nets (same corpus/vocab).
  - float precision: weights stay float32; percentiles/SVD computed after
    float64 upcast (stability), documented per-field.
  - zeros: exact zeros handled by min_nonzero flooring; zero counts
    reported (trained nets: expected ~none outside biases).
  - percentile caveat: per-row percentiles over 65-512 entries are noisy
    for a single row; the census reports the DISTRIBUTION (median/p90/max
    across rows), which is the neighbor-relevant quantity.
  - no adjudicated bars: this is a courtesy read, not a claim.

OUTPUTS: runs/x1/metrics.json (PROGRESSIVE partial writes after every
phase), runs/x1/range_census.csv (neighbor-friendly flat table),
runs/x1/x1_range_spread_summary.png (per-layer range-spread summary, both
nets). This cell does NOT edit NOTES/THINKING/QUEUE/STATE (executor
scope); the neighbor-facing summary goes to scratch/x1_summary_for_neighbor.md.
"""
from __future__ import annotations

import csv
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import REPO, Cfg, CharCorpus, TinyGPT, now_iso

RUN = REPO / "runs" / "x1"
CKPT = REPO / "runs" / "checkpoints"
BATTERY_TOKENS = 64          # fixed 64-token battery, dispatch-specified
Q_LO, Q_HI = 0.01, 0.99      # p1 / p99 of |w|
SV_TOPK = 16

NETS = [
    ("e098_base_s4305", CKPT / "e098_base_s4305.pt",
     Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512),
     {"params": 873472, "step": 2000, "val_loss": 1.6445,
      "note": "seed ladder s4305 base; 4L/4H/128d/512ctx"}),
    ("e131_consolidated_e113", CKPT / "e131_consolidated_e113.pt",
     Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192, block_size=256),
     {"params": 2739072, "step": None,
      "note": "e113 recipe (jittered replay 300, seed 10901) on e048_repro; "
              "6L/6H/192d/256ctx"}),
]

SKIPPED = {
    "g1bS_s1113_val_min": {
        "requested": "10M host healthy-val state (step 1113, val 1.5766)",
        "status": "SKIPPED — not loadable",
        "reason": "runs/checkpoints/g1bS_base.pt (the s1113 resume state) no "
                  "longer exists on disk; *.pt are gitignored (on-disk only). "
                  "Surviving 10M states are g1bS_base_diverged_s3250.pt and "
                  "g1bS_base_overtrained_s4000_lr4e-4.pt (val 2.8506 — the "
                  "memorizing base that FAILED G-BASE-QUAL; not healthy-val). "
                  "The 10M census is licensed to g1bS2's re-registered host.",
    }
}

T0 = time.time()
PHASES = []


def log(msg: str) -> None:
    print(f"[x1 {time.time()-T0:7.1f}s] {msg}", flush=True)


def write_partial(metrics: dict, phase: str) -> None:
    PHASES.append(phase)
    metrics["phases"] = list(PHASES)
    metrics["partial"] = phase != "done"
    metrics["phase"] = phase
    metrics["progressive_writes"] = len(PHASES)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "metrics.json").write_text(
        json_dumps(metrics), encoding="utf-8")


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2, default=float)


def _q(v: torch.Tensor, q: float) -> float:
    return float(torch.quantile(v, q).item())


def spread_stats(x: torch.Tensor) -> dict:
    """Dynamic-range spreads of |x| (1-D, float). p1 floored at min nonzero
    if zeros push the quantile to 0; zero count always reported."""
    a = x.detach().double().abs().flatten()
    n = a.numel()
    zeros = int((a == 0).sum().item())
    nz = a[a > 0]
    if nz.numel() == 0:
        return {"n": n, "n_zero": zeros, "all_zero": True}
    p1 = _q(a, Q_LO) if zeros == 0 else min(_q(nz, Q_LO), _q(a, Q_LO))
    p1 = max(p1, float(nz.min().item())) if p1 > 0 else float(nz.min().item())
    p99 = max(_q(a, Q_HI), float(nz.min().item()))
    mx, mn = float(nz.max().item()), float(nz.min().item())
    lo2 = math.floor(math.log2(mn))
    return {
        "n": n, "n_zero": zeros,
        "abs_p1": p1, "abs_p50": _q(a, 0.5), "abs_p99": p99,
        "abs_max": mx, "abs_min_nonzero": mn,
        "spread_p99_p1_log2": math.log2(p99 / p1),
        "ratio_max_min_log2": math.log2(mx / mn),
        "exponent_span_bits": int(math.floor(math.log2(mx)) - lo2),
    }


def _rowcol(W: torch.Tensor, dim: int) -> dict:
    """Spread stats per row (dim=1 keeps rows) / per column, summarized by
    median / p90 / max across the axis distribution."""
    A = W.detach().double().abs()
    rows = [spread_stats(A[i]) for i in range(A.shape[0])]
    good = [r for r in rows if not r.get("all_zero")]
    def agg(key):
        v = torch.tensor([r[key] for r in good], dtype=torch.float64)
        return {"median": _q(v, 0.5), "p90": _q(v, 0.9), "max": float(v.max())}
    return {
        "count": len(rows), "all_zero_rows": len(rows) - len(good),
        "spread_p99_p1_log2": agg("spread_p99_p1_log2"),
        "ratio_max_min_log2": agg("ratio_max_min_log2"),
        "exponent_span_bits": agg("exponent_span_bits"),
    }


def matrix_census(W: torch.Tensor) -> dict:
    W = W.detach().float()
    out = {"shape": list(W.shape)}
    out["global"] = spread_stats(W)
    out["per_row"] = _rowcol(W, 0)          # rows = output units
    out["per_col"] = _rowcol(W.T, 0)        # cols = input units
    return out


def vector_census(v: torch.Tensor) -> dict:
    s = spread_stats(v)
    return {"shape": list(v.shape), "global": s,
            "per_row": None, "per_col": None,
            "note": "vector (LayerNorm weight/bias): global stats only"}


def sv_census(W: torch.Tensor) -> dict:
    """Exact top-16 SVs, float64 upcast. All lab matrices are small enough
    that no sampling/lowrank is needed (largest: 512x128)."""
    W = W.detach().double()
    s = torch.linalg.svdvals(W)              # descending
    k = min(SV_TOPK, s.numel())
    top = s[:k]
    e = s ** 2
    p = e / e.sum()
    erank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum()).item())
    g = lambda i: float(top[i].item()) if i < k else None
    return {
        "top_singular_values": [float(x) for x in top.tolist()],
        "s1_over_s4": g(0) / g(3) if k >= 4 else None,
        "s1_over_s8": g(0) / g(7) if k >= 8 else None,
        "s1_over_s16": g(0) / g(15) if k >= 16 else None,
        "effective_rank_entropy": erank,
        "energy_frac_top16": float((e[:k].sum() / e.sum()).item()),
        "method": f"torch.linalg.svdvals, float64 upcast, exact (no sampling; "
                  f"largest matrix {tuple(W.shape)})",
    }


def weight_sites(model: TinyGPT):
    """(site_label, kind, tensor) in canonical layer order."""
    sites = [("wte", "matrix", model.wte.weight),
             ("wpe", "matrix", model.wpe.weight)]
    for i, blk in enumerate(model.h):
        sites += [
            (f"h{i}.ln1", "vector", blk.ln1.weight),
            (f"h{i}.ln1.bias", "vector", blk.ln1.bias),
            (f"h{i}.attn.c_attn[qkv]", "matrix", blk.attn.c_attn.weight),
            (f"h{i}.attn.c_proj[attn-out]", "matrix", blk.attn.c_proj.weight),
            (f"h{i}.ln2", "vector", blk.ln2.weight),
            (f"h{i}.ln2.bias", "vector", blk.ln2.bias),
            (f"h{i}.mlp.0[mlp-in]", "matrix", blk.mlp[0].weight),
            (f"h{i}.mlp.0.bias", "vector", blk.mlp[0].bias),
            (f"h{i}.mlp.2[mlp-out]", "matrix", blk.mlp[2].weight),
            (f"h{i}.mlp.2.bias", "vector", blk.mlp[2].bias),
        ]
    sites += [("ln_f", "vector", model.ln_f.weight),
              ("ln_f.bias", "vector", model.ln_f.bias),
              ("lm_head", "matrix", model.lm_head.weight)]
    return sites


def load_net(name, path, cfg):
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = state["model"] if "model" in state else state
    net = TinyGPT(cfg)
    net.load_state_dict(sd)
    net.eval()
    return net, state


@torch.no_grad()
def activation_census(net: TinyGPT, ids: torch.Tensor) -> dict:
    """One forward pass; residual stream after embedding sum and after each
    block. Rows = tokens (across channels), cols = channels (across tokens)."""
    T = ids.numel()
    x = net.wte(ids.unsqueeze(0)) + net.wpe(torch.arange(T))
    streams = {"stream0_embed": x[0]}
    for i, blk in enumerate(net.h):
        x = blk(x)
        streams[f"stream{i+1}_block{i}_out"] = x[0]
    per_layer = {}
    for label, A in streams.items():
        per_layer[label] = {
            "shape": [A.shape[0], A.shape[1]],
            "global": spread_stats(A),
            "per_row_tokens_across_channels": _rowcol(A, 0),
            "per_col_channels_across_tokens": _rowcol(A.T, 0),
        }
    return per_layer


def main() -> None:
    metrics = {
        "experiment": "x1_range_census",
        "date": now_iso(),
        "question_verbatim": (
            "(1) for your trained 1M-100M nets, how large is the "
            "exponent/dynamic-range spread per row and per column of each "
            "weight matrix and of activations? Our emulation cost depends on "
            "exactly that spread (wide-range rows need more INT8 residue "
            "planes). (2) Do any of your weight matrices show structure we "
            "could exploit (low displacement rank, Monarch/butterfly-like "
            "block structure, low rank + sparse)?"),
        "purpose": "courtesy census for matrix-native-math (INBOX 2026-10-01 "
                   "note 1); answer promised in REPLY_to_matrix-native-math.md; "
                   "no adjudicated bars",
        "honesty_block": {
            "checkpoints": [f"runs/checkpoints/{n}.pt" for n, _, _, _ in NETS],
            "skipped": SKIPPED,
            "battery": f"first {BATTERY_TOKENS} chars of the VAL split of "
                       "data/input.txt (deterministic positional slice; no RNG "
                       "anywhere in this cell; identical ids for both nets)",
            "float_precision": "checkpoints loaded float32 as committed; "
                               "percentiles and SVD computed after float64 upcast",
            "sv_method": f"exact torch.linalg.svdvals (float64) on every "
                         "matrix — all smaller than 513x129, no sampling or "
                         "lowrank needed; top-{SV_TOPK} reported",
            "row_percentile_caveat": "per-row p99/p1 over 65-512 entries is "
                                     "noisy per single row; distributions "
                                     "(median/p90/max across rows) are the "
                                     "reported quantity",
            "structure_question_scope": "SV decay co-reported as the nearest "
                                        "object we have; displacement rank and "
                                        "Monarch structure NOT examined",
            "determinism": "fully deterministic (no seeds, no RNG calls)",
        },
        "nets": {},
        "weights": {},
        "activations": {},
        "svd": {},
        "summary": {},
    }
    try:
        metrics["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            capture_output=True, text=True).stdout.strip()
    except Exception:
        metrics["git_commit"] = None
    write_partial(metrics, "start")
    log(f"start (commit {metrics['git_commit']})")

    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    battery = corpus.val[:BATTERY_TOKENS]
    metrics["battery"] = {
        "source": "data/input.txt val split [0:64]",
        "n_tokens": int(battery.numel()),
        "first_chars": corpus.decode(battery[:40]),
    }
    log(f"battery: {battery.numel()} tokens, starts "
        f"{metrics['battery']['first_chars'][:30]!r}")

    csv_rows = []

    for name, path, cfg, meta in NETS:
        net, state = load_net(name, path, cfg)
        metrics["nets"][name] = {**meta, "cfg": {
            "n_layer": cfg.n_layer, "n_head": cfg.n_head,
            "n_embd": cfg.n_embd, "block_size": cfg.block_size,
            "vocab": cfg.vocab},
            "ckpt_keys": sorted(state.keys()) if isinstance(state, dict) else str(type(state))}
        assert net.num_params() == meta["params"], \
            f"{name}: param mismatch {net.num_params()} != {meta['params']}"
        log(f"{name}: {net.num_params()} params loaded")

        # ---- weights + svd ----
        metrics["weights"][name] = {}
        metrics["svd"][name] = {}
        for label, kind, tensor in weight_sites(net):
            cens = matrix_census(tensor) if kind == "matrix" else vector_census(tensor)
            metrics["weights"][name][label] = cens
            if kind == "matrix":
                metrics["svd"][name][label] = sv_census(tensor)
            csv_rows.append(csv_row(name, "weight", label, cens,
                                    metrics["svd"][name].get(label)))
        write_partial(metrics, f"weights:{name}")
        log(f"weights:{name} census done ({len(weight_sites(net))} sites)")

        # ---- activations ----
        metrics["activations"][name] = activation_census(net, battery)
        for label, cens in metrics["activations"][name].items():
            csv_rows.append(csv_row(name, "activation", label, cens, None))
        write_partial(metrics, f"activations:{name}")
        log(f"activations:{name} done ({len(metrics['activations'][name])} streams)")

    # ---- CSV (neighbor-friendly flat artifact) ----
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "range_census.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    metrics["artifacts"] = {"csv": "runs/x1/range_census.csv",
                            "rows": len(csv_rows)}
    write_partial(metrics, "csv")
    log(f"csv written: {len(csv_rows)} rows")

    # ---- summary (the neighbor's headline numbers) ----
    metrics["summary"] = build_summary(metrics)
    write_partial(metrics, "summary")

    # ---- PNG ----
    plot_summary(metrics, RUN / "x1_range_spread_summary.png")
    metrics["artifacts"]["png"] = "runs/x1/x1_range_spread_summary.png"
    write_partial(metrics, "done")
    log("DONE")

    # console digest for the log
    for name in metrics["summary"]["nets"]:
        s = metrics["summary"]["nets"][name]
        print(f"\n=== {name} ({s['params']} params) digest ===")
        print(f"  weights: worst row-spread median {s['weights_worst_row_median']['site']} "
              f"{s['weights_worst_row_median']['log2']:.2f} bits | worst col "
              f"{s['weights_worst_col_median']['site']} "
              f"{s['weights_worst_col_median']['log2']:.2f} bits")
        print(f"  activations: token-spread median {s['act_row_median_bits']:.2f} bits | "
              f"channel-spread median {s['act_col_median_bits']:.2f} bits (max over layers)")
        print(f"  sv decay: fastest {s['sv_fastest_decay']['site']} "
              f"s1/s16={s['sv_fastest_decay']['s1_over_s16']:.0f} | slowest "
              f"{s['sv_slowest_decay']['site']} s1/s16="
              f"{s['sv_slowest_decay']['s1_over_s16']:.1f}")


def csv_row(net, kind, label, cens, sv):
    g = cens["global"]
    row = {
        "net": net, "kind": kind, "site": label,
        "rows": cens["shape"][0] if isinstance(cens["shape"], list) and len(cens["shape"]) > 1 else cens["shape"][0],
        "cols": cens["shape"][1] if isinstance(cens["shape"], list) and len(cens["shape"]) > 1 else "",
        "n_zero": g.get("n_zero", ""),
        "abs_p1": round(g["abs_p1"], 6) if "abs_p1" in g else "",
        "abs_p99": round(g["abs_p99"], 6) if "abs_p99" in g else "",
        "abs_max": round(g["abs_max"], 6) if "abs_max" in g else "",
        "global_spread_p99_p1_log2": round(g["spread_p99_p1_log2"], 3) if "spread_p99_p1_log2" in g else "",
        "global_ratio_max_min_log2": round(g["ratio_max_min_log2"], 3) if "ratio_max_min_log2" in g else "",
        "exponent_span_bits": g.get("exponent_span_bits", ""),
    }
    for prefix, key in (("row", "per_row"), ("col", "per_col")):
        pc = cens.get(key) or cens.get(
            "per_row_tokens_across_channels" if key == "per_row"
            else "per_col_channels_across_tokens")
        for stat in ("median", "p90", "max"):
            row[f"{prefix}_spread_p99p1_{stat}_log2"] = (
                round(pc["spread_p99_p1_log2"][stat], 3)
                if pc and "spread_p99_p1_log2" in pc else "")
        row[f"{prefix}_ratio_maxmin_median_log2"] = (
            round(pc["ratio_max_min_log2"]["median"], 3)
            if pc and "ratio_max_min_log2" in pc else "")
    if sv:
        row["s1"] = round(sv["top_singular_values"][0], 5)
        row["s16"] = round(sv["top_singular_values"][15], 5) if len(sv["top_singular_values"]) >= 16 else ""
        row["s1_over_s16"] = round(sv["s1_over_s16"], 2) if sv["s1_over_s16"] else ""
        row["effective_rank_entropy"] = round(sv["effective_rank_entropy"], 2)
        row["energy_frac_top16"] = round(sv["energy_frac_top16"], 5)
    else:
        for k in ("s1", "s16", "s1_over_s16", "effective_rank_entropy", "energy_frac_top16"):
            row[k] = ""
    return row


def build_summary(m):
    out = {"nets": {}}
    for name in m["nets"]:
        s = {"params": m["nets"][name]["params"]}
        worst_r = worst_c = None
        for site, c in m["weights"][name].items():
            if not c.get("per_row"):
                continue
            r = c["per_row"]["spread_p99_p1_log2"]["median"]
            cc = c["per_col"]["spread_p99_p1_log2"]["median"]
            if worst_r is None or r > worst_r["log2"]:
                worst_r = {"site": site, "log2": r}
            if worst_c is None or cc > worst_c["log2"]:
                worst_c = {"site": site, "log2": cc}
        s["weights_worst_row_median"] = worst_r
        s["weights_worst_col_median"] = worst_c
        acts = m["activations"][name]
        s["act_row_median_bits"] = max(
            v["per_row_tokens_across_channels"]["spread_p99_p1_log2"]["median"]
            for v in acts.values())
        s["act_col_median_bits"] = max(
            v["per_col_channels_across_tokens"]["spread_p99_p1_log2"]["median"]
            for v in acts.values())
        s["act_row_max_bits"] = max(
            v["per_row_tokens_across_channels"]["spread_p99_p1_log2"]["max"]
            for v in acts.values())
        fastest = slowest = None
        for site, sv in m["svd"][name].items():
            if not sv.get("s1_over_s16"):
                continue
            d = {"site": site, "s1_over_s16": sv["s1_over_s16"],
                 "erank": sv["effective_rank_entropy"],
                 "e16": sv["energy_frac_top16"]}
            if fastest is None or d["s1_over_s16"] > fastest["s1_over_s16"]:
                fastest = d
            if slowest is None or d["s1_over_s16"] < slowest["s1_over_s16"]:
                slowest = d
        s["sv_fastest_decay"] = fastest
        s["sv_slowest_decay"] = slowest
        out["nets"][name] = s
    return out


def plot_summary(m, out_png):
    """4x2 grid: cols = nets; rows = (1) weight median spreads per site,
    (2) weight p90 spreads, (3) activation residual-stream spreads per
    layer, (4) SV decay curves for representative matrices."""
    names = list(m["nets"].keys())
    fig, axes = plt.subplots(4, len(names), figsize=(13.5, 15.5))
    if len(names) == 1:
        axes = axes.reshape(-1, 1)
    block_keys = ["attn.c_attn[qkv]", "attn.c_proj[attn-out]",
                  "mlp.0[mlp-in]", "mlp.2[mlp-out]"]
    for ci, name in enumerate(names):
        W = m["weights"][name]
        L = m["nets"][name]["cfg"]["n_layer"]
        # rows 1-2: weights. x slots: wte, wpe, then per-layer 4 block sites
        x0, xwpe, xoff = 0, 1, 2
        for row_i, stat in ((0, "median"), (1, "p90")):
            ax = axes[row_i][ci]
            for bi, bkey in enumerate(block_keys):
                xs = [xoff + i * 4 + bi for i in range(L)]
                ys_r = [W[f"h{i}.{bkey}"]["per_row"]["spread_p99_p1_log2"][stat] for i in range(L)]
                ys_c = [W[f"h{i}.{bkey}"]["per_col"]["spread_p99_p1_log2"][stat] for i in range(L)]
                ax.plot(xs, ys_r, "o-", ms=4, label=f"row {bkey.split('[')[0]}")
                ax.plot(xs, ys_c, "^--", ms=4, alpha=0.55)
            for lbl, key, mk, dx, dy in (("wte", "wte", "o", 6, -3),
                                         ("wpe", "wpe", "s", 6, 3),
                                         ("lm_head", "lm_head", "D", 6, 4)):
                ax.scatter([x0 if key != "wpe" else xwpe],
                           [W[key]["per_row"]["spread_p99_p1_log2"][stat]],
                           marker=mk, c="k", zorder=5)
                ax.scatter([x0 if key != "wpe" else xwpe],
                           [W[key]["per_col"]["spread_p99_p1_log2"][stat]],
                           marker=mk, facecolors="none", edgecolors="k", zorder=5)
                ax.annotate(lbl, (x0 if key != "wpe" else xwpe,
                                  W[key]["per_row"]["spread_p99_p1_log2"][stat]),
                            textcoords="offset points", xytext=(dx, dy), fontsize=7)
            ax.set_title(f"{name} — weight spreads ({stat} across rows/cols)", fontsize=9)
            ax.set_ylabel("log2(p99/p1) of |w|")
            ax.set_xticks([x0, xwpe] + [xoff + i * 4 + 1.5 for i in range(L)])
            ax.set_xticklabels(["wte", "wpe"] + [f"L{i}" for i in range(L)],
                               fontsize=7)
            if row_i == 0 and ci == 0:
                ax.legend(fontsize=6, ncol=2, loc="upper left")
            ax.grid(alpha=0.25)
        # row 3: activations
        ax = axes[2][ci]
        acts = m["activations"][name]
        labels = sorted(acts.keys())
        xs = list(range(len(labels)))
        yr = [acts[l]["per_row_tokens_across_channels"]["spread_p99_p1_log2"]["median"] for l in labels]
        yc = [acts[l]["per_col_channels_across_tokens"]["spread_p99_p1_log2"]["median"] for l in labels]
        yr90 = [acts[l]["per_row_tokens_across_channels"]["spread_p99_p1_log2"]["p90"] for l in labels]
        ax.plot(xs, yr, "o-", label="per-token (across channels) median")
        ax.plot(xs, yc, "^--", label="per-channel (across tokens) median")
        ax.plot(xs, yr90, ":", alpha=0.6, label="per-token p90")
        ax.set_xticks(xs)
        ax.set_xticklabels([l.replace("stream", "s").replace("_block", "/b").replace("_out", "").replace("_embed", " emb").replace("_out", "") for l in labels], fontsize=7)
        ax.set_title(f"{name} — residual-stream |a| spreads (64-token battery)", fontsize=9)
        ax.set_ylabel("log2(p99/p1) of |a|")
        if ci == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=1.0)
        ax.grid(alpha=0.25)
        # row 4: sv decay curves, representative mid-layer matrices
        ax = axes[3][ci]
        mid = L // 2
        for site in (f"h{mid}.attn.c_attn[qkv]", f"h{mid}.attn.c_proj[attn-out]",
                     f"h{mid}.mlp.0[mlp-in]", f"h{mid}.mlp.2[mlp-out]",
                     "wte", "wpe", "lm_head"):
            sv = m["svd"][name][site]["top_singular_values"]
            ax.plot(range(1, len(sv) + 1), [s / sv[0] for s in sv],
                    marker=".", ms=3, label=site.replace(f"h{mid}.", f"h{mid}/"))
        ax.set_yscale("log")
        ax.set_title(f"{name} — SV decay s_k/s_1 (courtesy co-report)", fontsize=9)
        ax.set_xlabel("k"); ax.set_ylabel("s_k / s_1")
        ax.legend(fontsize=5.5, ncol=4, loc="lower left", framealpha=1.0,
                  columnspacing=0.9, handlelength=1.4)
        ax.grid(alpha=0.25, which="both")
    fig.suptitle("X1 range census — dynamic-range spreads (log2 p99/p1) + SV decay",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=2.2)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
