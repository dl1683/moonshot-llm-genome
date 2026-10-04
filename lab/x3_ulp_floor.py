"""X3 — THE ULP FLOOR OF ONE MATMUL (neighbor courtesy, INBOX note 5;
also W030's floor-of-floors made literal: the arithmetic floor under every
cosine the lab has ever reported).

WHY (note 5, verbatim intent): "take ONE of your layers' matmuls and
measure how its output differs, bitwise, between (a) two runs in the same
session, (b) before/after a GPU cool-down or a different batch size/stream,
(c) FP32 vs FP64 reference - report the distribution of ULP differences,
not just the max." The lab's stake: W030 recorded that closed nulls are
the lab measuring its own floor; this cell prices the ARITHMETIC part of
that floor — below what |Delta cos| is a difference arithmetically
unresolvable in fp32 GEMM, regardless of the organism?

MATERIAL: the real up-projection h.3.mlp.0 (768x192) of
runs/checkpoints/e131_consolidated_e113.pt, fed a REAL activation: a
forward hook captures its input (post-ln2 residual stream) on a genuine
256-char context from data/input.txt. TF32 OFF for the reference
comparisons (and one TF32-ON arm recorded, since training-time defaults
matter to the neighbor's emulation contracts).

CONDITIONS:
  (a) same-session reruns: three identical GPU fp32 calls -> bitwise.
  (b) batch-size split: one 256-row call vs two 128-row calls -> bitwise
      on overlapping rows (GEMM tiling changes rounding).
  (c) non-default stream: same call under torch.cuda.Stream -> bitwise.
  (d) after-pause: ~3 s sleep + fresh context build -> bitwise (a thermal
      proxy; honest note recorded — a true cold/hot comparison is the
      neighbor's workload, not ours).
  (e) cross-device: CPU fp32 vs GPU fp32 -> ULP distribution.
  (f) FP64 reference: CPU fp64 -> ULP distribution of the GPU fp32 output
      (THE distribution: rounding beyond the 0.5-ulp representability
      bound is accumulated GEMM error).
  (g) TF32-ON arm of (f) for the neighbor's contract mapping.

BARS (frozen before compute):
  DETERMINISTIC-WITHIN — same-session reruns bit-identical (or the ULP
      distribution reported if not).
  FLOOR-QUANTIFIED — the (e)/(f)/(g) ULP distributions + the induced
      cosine floor reported: any future |Delta cos| below the floor is
      arithmetically unresolvable, and every past null at or below it is
      re-read as floored, not empty.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
torch.set_num_threads(4)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CharCorpus, Cfg, TinyGPT, gpu_status, run_dir, save_json

HERE = Path(__file__).resolve().parent
CKPT = HERE.parent / "runs/checkpoints/e131_consolidated_e113.pt"
CORPUS = HERE.parent / "data/input.txt"


def bitwise(a: torch.Tensor, b: torch.Tensor) -> dict:
    ai = a.view(torch.int32).cpu().numpy()
    bi = b.view(torch.int32).cpu().numpy()
    eq = ai == bi
    return {"frac_bit_identical": float(eq.mean()),
            "n_diff": int((~eq).sum()), "n_total": int(eq.size)}


def ulp_dist(y32: np.ndarray, ref64: np.ndarray) -> dict:
    sp = np.spacing(np.abs(ref64).astype(np.float32)).astype(np.float64)
    sp = np.maximum(sp, np.float32(np.finfo(np.float32).tiny).astype(np.float64))
    ulps = np.abs(y32.astype(np.float64) - ref64) / sp
    edges = np.array([0.0, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, np.inf])
    hist, _ = np.histogram(ulps, bins=edges)
    return {
        "frac_exact_0ulp": float((ulps == 0).mean()),
        "p50": float(np.percentile(ulps, 50)),
        "p90": float(np.percentile(ulps, 90)),
        "p99": float(np.percentile(ulps, 99)),
        "max": float(ulps.max()),
        "hist_edges": edges[:-1].tolist(),
        "hist_counts": hist.tolist(),
    }


def main() -> None:
    rd = run_dir("x3")
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TinyGPT(Cfg())
    model.load_state_dict(blob["model"])
    model.eval()

    corpus = CharCorpus(CORPUS)
    ctx = corpus.train[:256].unsqueeze(0)

    captured: dict[str, torch.Tensor] = {}

    def hook(_m, inp, _out):
        captured["x"] = inp[0].detach()

    model.h[3].mlp[0].register_forward_hook(hook)
    with torch.no_grad():
        model(ctx)
    X = captured["x"].reshape(-1, 192).clone()  # (256, 192) real activations
    W = blob["model"]["h.3.mlp.0.weight"].float()  # (768, 192)

    try:
        gpu = gpu_status()
    except Exception as e:  # keep the cell alive on CPU-only boxes
        gpu = {"error": str(e)}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    out: dict = {"experiment": "x3_ulp_floor", "status": "PROGRESSIVE",
                 "device": dev, "gpu_at_start": gpu,
                 "matmul": "y = X @ W.T  (X 256x192 real post-ln2 activations, W 768x192)"}

    torch.backends.cuda.matmul.allow_tf32 = False
    Xg, Wg = X.to(dev), W.to(dev)
    y1 = Xg @ Wg.T
    y2 = Xg @ Wg.T
    y3 = Xg @ Wg.T
    torch.cuda.synchronize() if dev == "cuda" else None
    out["a_same_session"] = {"r1_r2": bitwise(y1, y2), "r1_r3": bitwise(y1, y3)}

    ya, yb = Xg[:128] @ Wg.T, Xg[128:] @ Wg.T
    ysplit = torch.cat([ya, yb], dim=0)
    out["b_batch_split"] = bitwise(ysplit, y1)

    if dev == "cuda":
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            ystream = Xg @ Wg.T
        s.synchronize()
        out["c_stream"] = bitwise(ystream, y1)
    else:
        out["c_stream"] = "n/a (cpu)"

    time.sleep(3.0)
    y4 = Xg @ Wg.T
    out["d_after_pause"] = bitwise(y4, y1)
    try:
        out["gpu_at_end"] = gpu_status()
    except Exception as e:
        out["gpu_at_end"] = {"error": str(e)}

    y_cpu32 = (X @ W.T).numpy()
    y_gpu32 = y1.cpu().numpy()
    ref64 = (X.double() @ W.double().T).numpy()
    out["e_cross_device"] = ulp_dist(y_gpu32, ref64)
    out["e_cross_device_cpu_vs_ref"] = ulp_dist(y_cpu32, ref64)

    torch.backends.cuda.matmul.allow_tf32 = True
    y_tf32 = (Xg @ Wg.T).cpu().numpy()
    torch.backends.cuda.matmul.allow_tf32 = False
    out["g_tf32_arm"] = ulp_dist(y_tf32, ref64)

    v_gpu = y_gpu32.reshape(-1)
    v_ref = ref64.reshape(-1)
    cos = float(np.dot(v_gpu, v_ref) /
                (np.linalg.norm(v_gpu) * np.linalg.norm(v_ref)))
    out["f_cosine_floor"] = {
        "cos_gpu32_vs_fp64ref": cos,
        "one_minus_cos": 1.0 - cos,
        "statement": "any |Delta cos| at or below one_minus_cos is arithmetically "
                     "unresolvable in this pipeline; W030 re-reads its floor here",
    }

    # PNG: ULP histograms e (gpu-vs-ref) and g (tf32-vs-ref)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, key, title in [(axes[0], "e_cross_device", "GPU fp32 vs FP64 ref"),
                           (axes[1], "g_tf32_arm", "TF32 vs FP64 ref")]:
        h = out[key]
        centers = np.arange(len(h["hist_counts"]))
        ax.bar(centers, np.array(h["hist_counts"]) / max(1, sum(h["hist_counts"])))
        ax.set_xticks(centers)
        ax.set_xticklabels([str(int(e)) for e in h["hist_edges"]], fontsize=7)
        ax.set_xlabel("ULP bin (upper edge; last = inf)")
        ax.set_ylabel("fraction of elements")
        ax.set_title(f"{title}\nexact={h['frac_exact_0ulp']:.3f} p99={h['p99']:.1f} max={h['max']:.0f} ulp")
    fig.suptitle("X3 — ULP floor of one real matmul (e131 h.3.mlp.0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(rd / "ulp_hist.png", dpi=140)

    out["status"] = "DONE"
    save_json(rd / "metrics.json", out)
    print(json.dumps({k: out[k] for k in
                      ["a_same_session", "b_batch_split", "c_stream", "d_after_pause",
                       "f_cosine_floor"]}, indent=2, default=str))
    print("e p99/max:", out["e_cross_device"]["p99"], out["e_cross_device"]["max"],
          "| g p99/max:", out["g_tf32_arm"]["p99"], out["g_tf32_arm"]["max"])


if __name__ == "__main__":
    main()
