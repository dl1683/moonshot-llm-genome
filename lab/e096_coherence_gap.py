"""E096 — the COHERENCE-GAP DOSE LADDER (scratch/next_wave_programs.md item 6;
P-B from scratch/explorations_harvest_20260926.md, sharpened by T051's
mass-action/threshold law). CPU-only.

CONTEXT. e089 closed P3's arc: the anchor's dose-response law in DISCRETE
entries is threshold-shaped (k=200 continuous whole-band V-zero removal costs
+2.153 nats; e075's K=32 lump schedule of the same band +0.262). e080 showed
norm-matched noise REPLACEMENT of the band is cheap (H-statistics-scaffold /
mixed). The harvest's open promise (table #4, Waddington/Rozum "coherence
gap": canalized systems robust to large basin-crossing perturbations,
sensitive to small within-valley ones) is the CONTINUOUS-dose face of the
same geometry: does mid-generation cache corruption damage the run
NON-monotonically per nat of perturbation?

DESIGN (registered here BEFORE compute; bars frozen from the tasking).
Battery: the e053c ctx-512 net (873,472 params, val CE 1.5227), the seed-202
8-draw prompts, B=8, the seed-7 control free run (e075/e080/e085/e088/e089
A-none convention; G3 gated against e080's AND e089's stored control runs).

INTERVENTION (the e089 matched-stream continuation rig, single-shot variant):
at generation step g_int=250 (prefix = positions 0..313; the token at
position 314 is teacher-forced from the control run), corrupt ALL anchor-band
entries present at that frame — self-generated positions 64..217, i.e. ages
97..250 relative to the query at 314 (the e075 dead-band rule age > 96
evaluated at the intervention frame; the e089 final-frame anchor band
64..414 = ages 97..447 exists only in part mid-run, and its extant prefix —
154 entries — is corrupted) — all SIMULTANEOUSLY, ONCE, immediately before
the decode at 314 (the e085/e088/e089 top-of-the-event timing; single-shot is
feasible here because every band entry already exists at t_int, which is
exactly what forced e089 to crossing-time scheduling for SPREAD removals).
Corruption: V-only (K untouched, the e075/e080/e085/e089 instrument), per
(row, layer, head, position) d=32 vector:
    V <- V + eps * m_layer * u,    u = unit-L2 gaussian direction,
drawn from a dedicated CPU generator (seed 9696, the e080 seed-4242
convention) that never touches the sampling stream; m_layer = MEDIAN ||V||
over all 8*4*154 band vectors of that layer in the pristine cache
("eps x median-entry-norm": eps=1.0 displaces every vector by one typical
entry norm; eps=0.05 by 5% of it). First affected sample = position 315; the
judged tail 448..511 is fully free-sampled after the intervention (exposure
197 >= 64, e089's criterion).

DOSE LADDER (tasking, frozen): eps in {0.05, 0.1, 0.25, 0.5, 1.0} (the
next-wave draft's 0.75 rung dropped by the tasking) PLUS the eps=0 identity
arm (same code path, applies nothing; MUST come out bitwise identical to the
matched control continuation — the instrument gate).

READOUT per arm: cost = clean-judge tail CE (positions 448..511, clean net,
full clean recompute — e089's judge_windows verbatim) of the corruption
stream minus the MATCHED control stream (same prefix, same continuation seed
960250, shared row-order generator; streams row-matched until sampled
divergence), mean over B=8 rows; paired-row bootstrap CIs. Attractor
signature per dose (the tasking's secondary readout): the clean-judge GAP
(clean judge minus the run's own online tail CE; e080's off-manifold bar:
gap > 1.0 nats; e080 vzero calibration: cj 6.40, gap +5.02) plus tail entropy
(fluency spot-check — basin re-entry = cj near control AND entropy near
control). Reference anchors carried on the curve: e075 whole-band V-zero
+0.2619 (K=32 schedule) and e089 k=200 continuous removal +2.1530.

REGISTERED BARS (frozen from the tasking; evaluated on the per-eps MEANS of
the paired costs, bootstrap CIs reported alongside):
  - NON-MONOTONE (U-shape; canalization made causal): per-nat cost at the
    smallest dose exceeds 2x the mid-dose per-nat cost —
        cost(0.05)/0.05 > 2 * cost(0.5)/0.5
    (equivalently cost(0.05) > 0.2 * cost(0.5)) — AND cost(1.0) <=
    cost(0.25) (the run re-enters a coherent basin under large corruption).
  - MONOTONE (the registered kill; canalization-continuity dead): cost rises
    with eps throughout — every consecutive pair of the ordered means
    increases (0.05 < 0.1 < 0.25 < 0.5 < 1.0).
  - The two clauses are mutually exclusive by construction (the re-entry
    clause contradicts strict rise). Anything else -> honest texture; if
    every |mean cost| <= 0.01 nats -> NULL (no dose separates from control).
  - HONESTY GUARD (added after the first pass exposed the case, before the
    rerun that shipped these artifacts; the frozen inequalities themselves
    are unchanged and still evaluated + reported literally): if EVERY ladder
    rung's mean cost is <= 0 — no dose damages at all — the U/monotone
    DAMAGE shapes are undefined and the reported clause is NO-DAMAGE
    (robustness/facilitation texture), with any vacuously-firing sub-clause
    (per-nat clause with cost(0.5) <= 0; re-entry clause with cost(0.25) <= 0)
    flagged as such. A verdict of "re-enters a coherent basin" is only
    honest when there was a cost peak to come down from.

Gates: G1 val CE vs e053c 1.5227 +-0.02; G2 params 873472; G3 control
battery identity vs e080's stored none arm (clean-judge per-seq, 1e-4, bitwise
flag) and bitwise vs e089's stored control run; G4 instrument identity
(eps=0 stream bitwise == control continuation; every arm prefix-identical
through position 314; band delta norms == eps*m_layer within 1e-5; non-band
cache columns bitwise untouched at application; matched continuation seed
across arms; single-shot by construction).

Run:     python lab/e096_coherence_gap.py
Outputs: runs/e096/metrics.json + runs/e096/coherence_gap.png
Envelope: NO training, NO new automations; CPU-only (CUDA_VISIBLE_DEVICES=-1),
8 threads (T050), single step, target minutes.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e089)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import (  # noqa: E402
    REPO,
    Cfg,
    CharCorpus,
    TinyGPT,
    estimate_loss,
    run_dir,
    save_json,
)

# ------------------------------------------------------------------ constants
PROMPT_TOK = 64
T_TOTAL = 512                             # e053c's ctx-512 window
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069..e089 seeds
SEED_NOISE = 9696                         # e096: dedicated noise generator
SEED_CONT = 960250                        # e096: the single matched
                                          # continuation seed (g_int=250)
B = 8
N_PROMPTS = 8
BOOT_N = 1000
THREADS = 8                               # T050: 12 spin-thrashes the box
torch.set_num_threads(THREADS)

P512 = T_TOTAL - 1                        # 511
TAIL = 64                                 # the e075..e089 tail window
KEY_T = (T_TOTAL - TAIL, T_TOTAL - 1)     # judged tail window (448, 511)
G = T_TOTAL - PROMPT_TOK                  # 448 generation steps

# ---- the intervention (single-shot, one frame) --------------------------------
G_INT = 250                               # generation step of the dose shot
T_INT = PROMPT_TOK + G_INT                # 314: teacher-forced token position
GEN_FIRST = 64                            # first generated position
BAND = list(range(GEN_FIRST, T_INT - 96)) # 64..217: ages 97..250 at query 314
                                          # (e075's age>96 rule at the frame;
                                          # 154 entries — the extant prefix of
                                          # the e089 final-frame band 64..414)
FIRST_AFFECTED = T_INT + 1                # 315 (first free-sampled position)
TAIL_STEP0 = KEY_T[0] - FIRST_AFFECTED    # 133: tail slice start in free steps

# ---- the dose ladder (tasking, frozen) ----------------------------------------
EPS_LADDER = [0.05, 0.1, 0.25, 0.5, 1.0]
ARMS = [0.0] + EPS_LADDER                 # eps=0 = bitwise identity gate arm

# ---- REGISTERED decision numbers (frozen, docstring verbatim) -----------------
PERNAT_FACTOR = 2.0                       # cost(0.05)/0.05 > 2 * cost(0.5)/0.5
NULL_LEVEL = 0.01                         # every |mean cost| <= this -> NULL
CJ_GAP_BAR = 1.0                          # e080's off-manifold bar (nats)

# ---- reference numbers (protocol-identity gates + curve anchors) --------------
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
E080_METRICS = REPO / "runs" / "e080" / "metrics.json"
E089_METRICS = REPO / "runs" / "e089" / "metrics.json"
E075_METRICS = REPO / "runs" / "e075" / "metrics.json"
E075_REF_DELTA = 0.26191402220749405      # whole-band V-zero, K=32 schedule
E089_REF_K200 = 2.1530486822128294        # k=200 continuous (K=1) removal
E080_VZERO_CJ = 6.402048110361914         # off-manifold calibration
E080_VZERO_CJ_GAP = 5.024754047397799
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e089)

@torch.no_grad()
def manual_all_logits(net: TinyGPT, idxs):
    """Clean forward returning logits at ALL positions (N, T, vocab) — the
    clean-judge instrument (scores a stream without any pruning)."""
    N, T = idxs.shape
    pos = torch.arange(T)
    x = net.wte(idxs) + net.wpe(pos).unsqueeze(0)
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    for blk in net.h:
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=2)
        q = q.view(N, T, H, d).transpose(1, 2)
        k = k.view(N, T, H, d).transpose(1, 2)
        v = v.view(N, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(N, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


def sample_and_ce(logits_clean: torch.Tensor, gen: torch.Generator):
    """Sample (temp 0.8, top-k 40) from filtered probs; CE from FULL softmax.
    [VERBATIM e053b — CPU tensors + CPU generator, stream-identical]"""
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
    """Batched prefill, (B, Tp) -> last-position logits (B, V) + KV cache.
    [VERBATIM e075/e080/e085/e088/e089]"""
    Bb, T = idx.shape
    x = net.wte(idx) + net.wpe(torch.arange(T))
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    kv = []
    for blk in net.h:
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=2)
        q = q.view(Bb, T, H, d).transpose(1, 2)
        k = k.view(Bb, T, H, d).transpose(1, 2)
        v = v.view(Bb, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(Bb, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
        kv.append((k, v))
    return net.lm_head(net.ln_f(x[:, -1, :])), kv


@torch.no_grad()
def decode_step_batch(net: TinyGPT, toks: torch.Tensor, pos: int, kv: list):
    """Batched incremental decode: (B,) tokens at position pos -> (B, V).
    [VERBATIM e075/e080/e085/e088/e089]"""
    Bb = toks.shape[0]
    x = net.wte(toks) + net.wpe(torch.full((Bb,), pos))
    H = net.cfg.n_head
    for li, blk in enumerate(net.h):
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=1)
        q = q.view(Bb, 1, H, d).transpose(1, 2)
        k = k.view(Bb, 1, H, d).transpose(1, 2)
        v = v.view(Bb, 1, H, d).transpose(1, 2)
        kp, vp = kv[li]
        k = torch.cat([kp, k], dim=2)
        v = torch.cat([vp, v], dim=2)
        kv[li] = (k, v)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))   # (B,H,1,t+1)
        probs = torch.softmax(att, -1)
        y = (probs @ v).transpose(1, 2).reshape(Bb, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


@torch.no_grad()
def generate_control(net: TinyGPT, prompts, gen: torch.Generator):
    """Free-run 64->512 for B sequences (e075/e080/e085/e088/e089 A-none
    VERBATIM stream math)."""
    Bb = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    for g in range(G):
        t = PROMPT_TOK + g
        toks = torch.zeros(Bb, dtype=torch.long)
        for j in range(Bb):
            tok, _ce = sample_and_ce(logits[j], gen)
            toks[j] = tok
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < T_TOTAL - 1:
            logits = decode_step_batch(net, toks, t, kv)
    return dict(idx=idx, kv=kv)


def band_median_norms(kv: list) -> list:
    """Per-layer MEDIAN ||V|| over all band vectors (rows x heads x 154
    positions) of a PRISTINE post-prefill cache — the dose unit's scale."""
    sel = torch.tensor(BAND, dtype=torch.long)
    meds = []
    for (_k, v) in kv:
        nrm = v[:, :, sel, :].norm(dim=-1)          # (B, H, n_band)
        meds.append(float(nrm.median()))
    return meds


@torch.no_grad()
def run_continuation(net: TinyGPT, prefix: torch.Tensor, forced_tok: torch.Tensor,
                     forced_pos: int, seed: int, eps: float = 0.0,
                     noise_gen: torch.Generator | None = None,
                     med_norm: list | None = None):
    """Single-shot dose-corruption continuation (eps=0: the matched control /
    bitwise identity arm). [e089's run_continuation with the e096
    single-shot corruption instead of crossing-time removal:]

    1. prefill(prefix) — prefix = control positions 0..forced_pos-1.
    2. if eps > 0: corrupt ALL band entries at once — every (row, layer,
       head, band position) d=32 V vector gets V <- V + eps * m_layer * u
       with u an independent unit-L2 gaussian direction from noise_gen
       (dedicated generator; the sampling stream never sees it). K is never
       touched. Applied BEFORE the decode at forced_pos (top-of-the-event
       timing), so the first affected SAMPLE is forced_pos+1.
    3. teacher-force the token at forced_pos (clean control token), then
       free-run to 511 with the shared row-order generator (e053b stream
       math) — all arms + control share the seed, so streams are row-by-row
       matched until sampled divergence.

    Tracks online CE + full-softmax entropy of every emitted free token (no
    rng cost). Returns idx, kv, fired-info dict (identity stats + bookkeeping).
    """
    R = prefix.shape[0]
    gen = torch.Generator().manual_seed(seed)
    logits, kv = prefill_batch(net, prefix)
    fired = dict(applied=False, n_vec=0, max_norm_dev=0.0, cos_abs_sum=0.0,
                 nonband_untouched=True)
    if eps > 0:
        assert noise_gen is not None and med_norm is not None
        sel = torch.tensor(BAND, dtype=torch.long)
        pre_v = [v.clone() for (_k, v) in kv]       # identity snapshot
        for li, (_k, v) in enumerate(kv):
            old = v[:, :, sel, :].clone()                     # (B,H,n,d)
            g = torch.randn(old.shape, generator=noise_gen)
            u = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            v[:, :, sel, :] = old + eps * med_norm[li] * u
            # ---- intervention identity: delta norm == eps*m_layer exactly;
            #      non-band columns bitwise untouched
            dn = (v[:, :, sel, :] - old).norm(dim=-1)
            fired["max_norm_dev"] = max(fired["max_norm_dev"], float(
                (dn - eps * med_norm[li]).abs().max()))
            cos = (old * u).sum(-1) / old.norm(dim=-1).clamp_min(1e-12)
            fired["cos_abs_sum"] += float(cos.abs().sum())
            fired["n_vec"] += int(cos.numel())
        nb_lo = torch.arange(0, GEN_FIRST)                    # prompt cols
        nb_hi = torch.arange(BAND[-1] + 1, forced_pos)        # young cols
        for v, pv in zip([v for (_k, v) in kv], pre_v):
            fired["nonband_untouched"] &= bool(
                torch.equal(v[:, :, nb_lo, :], pv[:, :, nb_lo, :]))
            fired["nonband_untouched"] &= bool(
                torch.equal(v[:, :, nb_hi, :], pv[:, :, nb_hi, :]))
        fired["applied"] = True
    idx = torch.cat([prefix, forced_tok[:, None]], 1)
    logits = decode_step_batch(net, forced_tok, forced_pos, kv)
    n_free = T_TOTAL - 1 - forced_pos
    ce_s = np.zeros((R, n_free), float)
    ent_s = np.zeros((R, n_free), float)
    for s in range(n_free):
        pos = forced_pos + 1 + s
        new = torch.zeros(R, dtype=torch.long)
        for j in range(R):
            tok, ce = sample_and_ce(logits[j], gen)
            new[j] = tok
            ce_s[j, s] = ce
        p = torch.softmax(logits.float(), -1)
        ent_s[:, s] = (-(p * p.clamp_min(1e-12).log()).sum(-1)).numpy()
        idx = torch.cat([idx, new[:, None]], 1)
        if pos < T_TOTAL - 1:
            logits = decode_step_batch(net, new, pos, kv)
    return idx, kv, fired, ce_s, ent_s


def judge_windows(all_lg: torch.Tensor, idx: torch.Tensor, windows):
    """Clean-net CE of target windows [(lo, hi), ...] (queries lo-1..hi-1).
    Returns dict window -> (R,) mean CE per row. [VERBATIM e085/e088/e089]"""
    out = {}
    for (lo, hi) in windows:
        lg = all_lg[:, lo - 1:hi, :]
        tgt = idx[:, lo:hi + 1]
        lp = torch.log_softmax(lg.float(), -1)
        out[(lo, hi)] = -lp.gather(2, tgt[:, :, None]).squeeze(2).mean(1).numpy()
    return out


# ------------------------------------------------------------- statistics

def boot_rows(arrs, fn, n: int = BOOT_N, seed: int = 0):
    """Paired bootstrap over the 8 rows (runs): resample row ids with
    replacement, recompute fn on the resampled arrays."""
    S = np.asarray(arrs[0]).shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, S, S)
        try:
            v = fn(*[np.asarray(a)[sel] for a in arrs])
        except Exception:
            v = None
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _fmt_ci(ci):
    return f"[{ci[0]:+.3f},{ci[1]:+.3f}]"


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width)


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e096")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e089 did
    corp = CharCorpus(REPO / "data" / "input.txt")       # seed 1337
    assert corp.vocab_size == 65
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    cfg = Cfg(vocab=corp.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=T_TOTAL)
    net = TinyGPT(cfg)
    net.load_state_dict(sd, strict=True)
    net.eval()
    n_params = net.num_params()
    gates["G2_params"] = dict(params=n_params, expected=873472,
                              ok=bool(n_params == 873472))
    val_ce = estimate_loss(net, corp, "val", n_batches=12)
    gates["G1_val_ce"] = dict(val_ce=val_ce, ref=E053C_VAL_CE, tol=0.02,
                              ok=bool(abs(val_ce - E053C_VAL_CE) <= 0.02))
    log(f"e053c net loaded ({n_params:,} params) | val CE {val_ce:.4f} vs "
        f"{E053C_VAL_CE:.4f} -> G1 {'PASS' if gates['G1_val_ce']['ok'] else 'FAIL'}")

    # ---- battery: the seed-202 8-draw, ALL 8
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts8 = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"battery: {N_PROMPTS} prompts (seed {SEED_PROMPT}); prompt0 prefix: "
        f"{corp.decode(prompts8[0])[:32]!r}")

    # ---- control free run (seed-7 A-none convention)
    log("control battery: seed-7 free run (e075/e080/e085/e088/e089 A-none)")
    gen7 = torch.Generator().manual_seed(SEED_SAMPLE)
    run1 = generate_control(net, prompts8, gen7)
    idx1 = run1["idx"]
    log(f"  control run done ({T_TOTAL} positions x {B} rows)")

    # ---- G3 (scope: control battery) vs e080's stored A-none + e089's control
    cj1 = judge_windows(manual_all_logits(net, idx1), idx1, [KEY_T])[KEY_T]
    g3 = dict(ref_files=[str(E080_METRICS), str(E089_METRICS)],
              note="e096 has no static arm; G3 scope = control-battery "
                   "identity vs e080's stored none arm (clean-judge per-seq, "
                   "1e-4) AND bitwise vs e089's stored control_run "
                   "[e088/e089's scoped G3 convention]")
    import json as _json
    ok3, dev3 = True, float("nan")
    if E080_METRICS.exists():
        e080 = _json.load(open(E080_METRICS))
        ref_cj = np.asarray(e080["arms"]["none"]["clean_judge_tail_ce"]["per_seq"])
        dev3 = float(np.abs(cj1 - ref_cj).max())
        bit80 = bool(np.array_equal(cj1, ref_cj))
        g3.update(e080_clean_judge_max_dev=dev3, e080_bitwise=bit80)
        ok3 &= bool(dev3 < 1e-4)
        log(f"G3 vs e080 A-none: clean-judge dev {dev3:.2e} (bitwise {bit80})")
    else:
        ok3 = False
        g3["note"] += " | runs/e080/metrics.json missing"
    if E089_METRICS.exists():
        e089 = _json.load(open(E089_METRICS))
        ref89 = np.asarray(e089["control_run"]["clean_judge_tail_ce"])
        dev89 = float(np.abs(cj1 - ref89).max())
        bit89 = bool(np.array_equal(cj1, ref89))
        g3.update(e089_clean_judge_max_dev=dev89, e089_bitwise=bit89)
        ok3 &= bool(dev89 < 1e-4)
        log(f"G3 vs e089 control: clean-judge dev {dev89:.2e} (bitwise {bit89})")
    else:
        g3["note"] += " | runs/e089/metrics.json missing (bitwise leg skipped)"
    g3["ok"] = bool(ok3)
    gates["G3_protocol_identity"] = g3
    log(f"G3 -> {'PASS' if g3['ok'] else 'FAIL'}")

    # ---- reference anchors from disk (drift-checked constants)
    e075_ref, e089_ref = E075_REF_DELTA, E089_REF_K200
    if E075_METRICS.exists():
        e75 = _json.load(open(E075_METRICS))
        r1 = e75["registered_decision"]["r1_tail_clean_ce"]
        e075_ref = float(r1["self"] - r1["none"])
    if E089_METRICS.exists():
        e89 = _json.load(open(E089_METRICS))
        e089_ref = float(e89["summary"]["per_k"]["200"]["mean"])
    log(f"reference anchors: e075 whole-band V-zero {e075_ref:+.4f} (K=32) | "
        f"e089 k=200 continuous {e089_ref:+.4f} (K=1)")

    # ================================================== the matched continuations
    prefix = idx1[:, :T_INT]
    forced = idx1[:, T_INT]
    log(f"continuations: prefix 0..{T_INT - 1}, forced token at {T_INT}, "
        f"band = positions {BAND[0]}..{BAND[-1]} ({len(BAND)} entries, ages "
        f"97..{T_INT - BAND[0]} at query {T_INT}), seed {SEED_CONT}")

    # the matched control (eps=0 code path, no draws) fixes the dose unit
    idx_ctl, kv_ctl, fired_ctl, ce_ctl, ent_ctl = run_continuation(
        net, prefix, forced, T_INT, SEED_CONT, eps=0.0)
    med_norm = band_median_norms(kv_ctl)
    log(f"matched control done | median band ||V|| per layer: "
        f"{[round(m, 4) for m in med_norm]} (the dose unit's scale)")

    noise_gen = torch.Generator().manual_seed(SEED_NOISE)
    S = {}
    for eps in ARMS:
        idx_a, kv_a, fired_a, ce_a, ent_a = run_continuation(
            net, prefix, forced, T_INT, SEED_CONT, eps=eps,
            noise_gen=noise_gen, med_norm=med_norm)
        J = judge_windows(manual_all_logits(net, idx_a), idx_a, [KEY_T])[KEY_T]
        eq = torch.eq(idx_a, idx_ctl)
        fdr = []
        for r in range(idx_a.shape[0]):
            nz = (~eq[r]).nonzero().flatten()
            fdr.append(int(nz[0].item()) if len(nz) else None)
        S[eps] = dict(
            idx=idx_a, ce=ce_a, ent=ent_a, J=J, fired=fired_a,
            stream_identical=bool(eq.all().item()),
            first_div_per_row=fdr,
            first_div=(min(d for d in fdr if d is not None)
                       if any(d is not None for d in fdr) else None),
            n_rows_diverged=sum(d is not None for d in fdr),
            online_tail=ce_a[:, TAIL_STEP0:].mean(1),
            ent_tail=ent_a[:, TAIL_STEP0:].mean(1),
        )
        tag = (f"|cos(v,u)| {fired_a['cos_abs_sum'] / max(fired_a['n_vec'], 1):.3f}"
               if fired_a["applied"] else "no-op")
        log(f"  arm eps={eps:<4}: cj tail {float(S[eps]['J'].mean()):.4f} | "
            f"online tail {float(S[eps]['online_tail'].mean()):.4f} | "
            f"ent tail {float(S[eps]['ent_tail'].mean()):.4f} | first_div "
            f"{S[eps]['first_div']} | {tag}")

    # ---- G4 instrument identity
    J_ctl = judge_windows(manual_all_logits(net, idx_ctl), idx_ctl,
                          [KEY_T])[KEY_T]
    bit0 = bool(torch.equal(S[0.0]["idx"], idx_ctl))
    pre_ident = all(bool(torch.equal(S[eps]["idx"][:, :T_INT + 1],
                                     idx_ctl[:, :T_INT + 1])) for eps in ARMS)
    fired_ok = all(S[eps]["fired"]["max_norm_dev"] < 1e-5
                   and S[eps]["fired"]["nonband_untouched"] for eps in EPS_LADDER)
    eps0_nofired = (not S[0.0]["fired"]["applied"])
    g4 = dict(
        eps0_bitwise_control=bit0,
        all_arms_prefix_identical_through=T_INT + 1 if pre_ident else None,
        prefix_identical=pre_ident,
        band_delta_norm_max_dev={str(eps): S[eps]["fired"]["max_norm_dev"]
                                 for eps in EPS_LADDER},
        nonband_untouched={str(eps): S[eps]["fired"]["nonband_untouched"]
                           for eps in EPS_LADDER},
        eps0_applied_nothing=eps0_nofired,
        matched_seed=SEED_CONT, single_shot=True,
        band=dict(positions=[BAND[0], BAND[-1]], n=len(BAND),
                  ages_at_query=[T_INT - BAND[-1], T_INT - BAND[0]]),
        noise=dict(seed=SEED_NOISE, rule="unit-L2 gaussian direction per "
                   "(row,layer,head,band-position) d=32 V vector; arms draw "
                   "independently in ladder order; eps=0 draws nothing",
                   median_norm_per_layer=med_norm))
    g4["ok"] = bool(bit0 and pre_ident and fired_ok and eps0_nofired)
    gates["G4_instrument_identity"] = g4
    log(f"G4 instrument identity: eps0 bitwise {bit0} | prefix identity "
        f"{pre_ident} | band delta norms exact {fired_ok} | "
        f"eps0 no-op {eps0_nofired} -> {'PASS' if g4['ok'] else 'FAIL'}")

    # ================================================== the dose curve + bars
    per = {}
    for eps in ARMS:
        c = S[eps]["J"] - J_ctl                 # eps=0: bitwise-zero by G4
        ci = boot_rows([c], lambda a: float(a.mean()))
        per[eps] = dict(
            cost_per_row=c.tolist(), mean=float(c.mean()), ci=ci,
            per_nat=(float(c.mean()) / eps) if eps > 0 else None,
            n_worse=int((c > 0).sum()),
            n_stream_identical=B - S[eps]["n_rows_diverged"],
            first_div=S[eps]["first_div"],
            first_div_per_row=S[eps]["first_div_per_row"],
            n_rows_diverged=S[eps]["n_rows_diverged"])
    # clean-judge attractor signature per dose (control baseline + e080 calib)
    cj_gap_ctl = float((J_ctl - S[0.0]["online_tail"]).mean())
    for eps in ARMS:
        per[eps]["clean_judge"] = dict(
            mean=float(S[eps]["J"].mean()),
            per_row=S[eps]["J"].tolist(),
            online_tail=float(S[eps]["online_tail"].mean()),
            gap_vs_online=float((S[eps]["J"] - S[eps]["online_tail"]).mean()),
            gap_per_row=(S[eps]["J"] - S[eps]["online_tail"]).tolist(),
            ent_tail=float(S[eps]["ent_tail"].mean()),
            ent_drift_pct=float((S[eps]["ent_tail"].mean()
                                 / float(ent_ctl[:, TAIL_STEP0:].mean()) - 1.0)
                                * 100.0))
        per[eps]["clean_judge"]["signature_fires"] = bool(
            per[eps]["clean_judge"]["gap_vs_online"] > CJ_GAP_BAR)

    m = {eps: per[eps]["mean"] for eps in ARMS}
    c05, c1, c25, c5, c10 = m[0.05], m[0.1], m[0.25], m[0.5], m[1.0]
    # registered contrasts (point rule on means; CIs alongside)
    clauseA = bool((c05 / 0.05) > PERNAT_FACTOR * (c5 / 0.5))
    clauseB = bool(c10 <= c25)
    nonmonotone = bool(clauseA and clauseB)
    monotone = bool(c05 < c1 and c1 < c25 and c25 < c5 and c5 < c10)
    null_tex = bool(all(abs(m[e]) <= NULL_LEVEL for e in ARMS))
    # honesty guard (docstring): vacuous sub-clauses + the no-damage case
    all_nonpos = bool(all(m[e] <= 0 for e in EPS_LADDER))
    vacuous_A = bool(c5 <= 0 or c05 <= 0)
    vacuous_B = bool(c25 <= 0)
    n_fac = sum(1 for e in EPS_LADDER if per[e]["ci"][1] < 0)
    # bootstrap CIs for the two registered contrasts (paired over rows)
    ci_A = boot_rows([per[0.05]["cost_per_row"], per[0.5]["cost_per_row"]],
                     lambda a, b_: (float(a.mean()) / 0.05)
                     - PERNAT_FACTOR * (float(b_.mean()) / 0.5))
    ci_B = boot_rows([per[1.0]["cost_per_row"], per[0.25]["cost_per_row"]],
                     lambda a, b_: float(a.mean()) - float(b_.mean()))
    ci_cons = {f"{a}->{b_}": boot_rows(
        [per[a]["cost_per_row"], per[b_]["cost_per_row"]],
        lambda x, y, _a=a, _b=b_: float(y.mean() - x.mean()))
        for a, b_ in zip(EPS_LADDER[:-1], EPS_LADDER[1:])}

    if all_nonpos:
        clause = ("NO-DAMAGE (all rungs non-positive; U/monotone damage "
                  "shapes undefined)")
        verdict = (
            f"NO rung of the ladder damages the run: every mean cost is "
            f"<= 0 ("
            + ", ".join(f"eps={e}: {m[e]:+.4f} {_fmt_ci(per[e]['ci'])}"
                        for e in EPS_LADDER)
            + f"). The frozen inequalities technically "
            f"{'FIRE' if nonmonotone else 'do not fire'} but "
            f"{'VACUOUSLY' if nonmonotone else ''}: the per-nat clause "
            f"compares against a non-positive mid-dose cost and the "
            f"re-entry clause has no positive cost peak to come down from — "
            f"a 're-enters a coherent basin' story would be false because "
            f"there was never a cost to recover from. Honest reading: "
            f"additive V corruption of the WHOLE anchor band (154 entries, "
            f"single shot at g=250) at doses up to one median entry norm "
            f"produces no tail-CE damage at any dose (point estimates "
            f"{min(m[e] for e in EPS_LADDER):+.4f}.."
            f"{max(m[e] for e in EPS_LADDER):+.4f} — zero to slightly "
            f"facilitative; {n_fac} of 5 rungs' CIs exclude 0 on the "
            f"facilitative side), and no off-manifold escape (clean-judge "
            f"gap ~0 at every dose vs e080 vzero calibration "
            f"{E080_VZERO_CJ_GAP:+.2f}). "
            + (f"eps=0.05 is FULLY ABSORBED: all {B} rows continue "
               f"bitwise-identically to the matched control for the full "
               f"197 post-corruption steps. " if per[0.05]["n_rows_diverged"] == 0
               else "")
            + f"The coherence-gap hypothesis is killed at its premise: "
            f"damage is not a function of perturbation magnitude anywhere "
            f"in [0.05, 1.0] x median-norm. The same mass REMOVED (V-zero) "
            f"costs {e089_ref:+.3f} nats on the K=1 schedule (e089 k=200) "
            f"and {e075_ref:+.3f} on the K=32 schedule (e075): the anchor "
            f"is fragile to removal, ROBUST to perturbation — the e080 "
            f"noise-replacement cheapness generalizes from replacement to "
            f"additive corruption.")
    elif nonmonotone:
        clause = "NON-MONOTONE (U-shape)"
        if vacuous_A or vacuous_B:
            clause += " [DEGENERATE sub-clause — see verdict]"
        verdict = (f"Both coherence-gap clauses fire: per-nat cost at eps=0.05 "
                   f"is {(c05 / 0.05) / (c5 / 0.5):.2f}x the eps=0.5 per-nat "
                   f"cost (bar > {PERNAT_FACTOR}x; contrast "
                   f"{c05 / 0.05 - PERNAT_FACTOR * (c5 / 0.5):+.3f} "
                   f"{_fmt_ci(ci_A)}) AND cost(1.0) {c10:+.4f} <= cost(0.25) "
                   f"{c25:+.4f} (contrast {c10 - c25:+.4f} {_fmt_ci(ci_B)}) — "
                   f"the run RE-ENTERS a coherent basin under corruption "
                   f"larger than the mid-dose peak. Small doses drift more "
                   f"per nat than mid doses; canalization is causal "
                   f"(canalization-continuity survives)."
                   + (" NOTE: a compared rung is non-positive — this firing "
                      "is partially vacuous; apply the honesty guard."
                      if (vacuous_A or vacuous_B) else ""))
    elif monotone:
        clause = "MONOTONE (the registered kill)"
        verdict = (f"Cost rises with eps throughout the ladder "
                   f"({c05:+.4f} < {c1:+.4f} < {c25:+.4f} < {c5:+.4f} < "
                   f"{c10:+.4f}) — per-nat damage does NOT peak at small "
                   f"dose (per-nat 0.05: {c05 / 0.05:+.3f} vs 0.5: "
                   f"{c5 / 0.5:+.3f} nats/nat) and the largest dose does not "
                   f"re-enter a cheap basin. Canalization-continuity is "
                   f"KILLED: the threshold law (e089) stays an entry-level "
                   f"fact, disconnected from continuous corruption; P-B takes "
                   f"its second haircut.")
    elif null_tex:
        clause = "NULL (no dose separates)"
        verdict = (f"Every |mean cost| <= {NULL_LEVEL} nats: "
                   + ", ".join(f"eps={e}: {m[e]:+.4f}" for e in ARMS)
                   + ". No corruption dose in the ladder separates the run "
                     f"from its matched control at B=8 — the anchor band "
                     f"tolerates additive noise up to one median norm per "
                     f"vector with no tail-CE consequence. Neither "
                     f"registered shape is evaluable.")
    else:
        clause = "TEXTURE (no registered shape fires cleanly)"
        verdict = (f"Curve: "
                   + " | ".join(f"eps={e}: {m[e]:+.4f} {_fmt_ci(per[e]['ci'])}"
                                for e in ARMS)
                   + f". Per-nat: " + " | ".join(
                       f"{e}: {m[e] / e:+.3f}" for e in EPS_LADDER)
                   + f". Registered contrasts: per-nat 0.05 vs "
                   f"{PERNAT_FACTOR}x per-nat 0.5 -> "
                   f"{c05 / 0.05 - PERNAT_FACTOR * (c5 / 0.5):+.3f} "
                   f"{_fmt_ci(ci_A)} [{'FIRES' if clauseA else 'no'}]; "
                   f"cost(1.0)-cost(0.25) -> {c10 - c25:+.4f} {_fmt_ci(ci_B)} "
                   f"[{'FIRES' if clauseB else 'no'}].")
    gap_fire = {str(e): per[e]["clean_judge"]["signature_fires"] for e in ARMS}
    first_gap = next((e for e in ARMS
                      if per[e]["clean_judge"]["signature_fires"]), None)
    attractor_note = (f"clean-judge attractor signature (gap > {CJ_GAP_BAR} "
                      f"nats): control gap {cj_gap_ctl:+.3f}; fires at "
                      + ("NO dose" if first_gap is None
                         else f"eps >= {first_gap} first")
                      + f" | gaps: "
                      + " ".join(f"{e}:{per[e]['clean_judge']['gap_vs_online']:+.2f}"
                                 for e in ARMS)
                      + f" (e080 vzero calibration {E080_VZERO_CJ_GAP:+.2f})")
    log("curve: " + " | ".join(f"eps={e} {m[e]:+.4f}" for e in ARMS))
    log("per-nat: " + " | ".join(f"eps={e} {m[e] / e:+.3f}" for e in EPS_LADDER))
    log(f"contrasts: A(per-nat 2x) {clauseA} CI {_fmt_ci(ci_A)} | "
        f"B(re-entry) {clauseB} CI {_fmt_ci(ci_B)}")
    log(attractor_note)
    log(f"REGISTERED DECISION [{clause}]: {verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e096_coherence_gap",
        purpose="the coherence-gap dose ladder (P-B / next_wave item 6, "
                "sharpened by T051's threshold law): single-shot mid-run V "
                "corruption of the whole anchor band at a dose ladder eps x "
                "median-entry-norm, matched-stream continuation + clean-net "
                "final-64 tail judgment (the e089 instrument, single-shot "
                "variant). NON-MONOTONE (per-nat cost peaks at small dose + "
                "large-dose basin re-entry) => canalization causal; MONOTONE "
                "=> the registered kill.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   noise=SEED_NOISE, continuation=SEED_CONT, bootstrap=0),
        protocol=dict(
            B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
            topk=TOPK, tail_window=list(KEY_T), boot_n=BOOT_N,
            intervention=dict(
                g_int=G_INT, t_int=T_INT, timing="corruption applied ONCE, "
                "immediately before the decode at t_int (top-of-the-event "
                "convention); token at t_int teacher-forced from the control "
                "run; first affected sample = t_int+1 = "
                f"{FIRST_AFFECTED}; tail exposure "
                f"{T_TOTAL - 1 - FIRST_AFFECTED + 1} free steps",
                band=dict(positions=[BAND[0], BAND[-1]], n=len(BAND),
                          rule="self-generated entries with age > 96 at the "
                               "intervention frame (e075 dead-band rule): "
                               "positions 64..217, ages 97..250 at query 314; "
                               "the extant prefix of the e089 final-frame "
                               "band 64..414 (ages 97..447)",
                          note="single-shot feasible because every band "
                               "entry already exists at t_int (e089 needed "
                               "crossing-time scheduling only for spread "
                               "removals)"),
            corruption="V-only (K untouched); per (row,layer,head,position) "
                       "d=32 vector: V <- V + eps * m_layer * u, u = "
                       "unit-L2 gaussian direction from the dedicated "
                       "generator (seed 9696); m_layer = median ||V|| over "
                       "the 8x4x154 band vectors of the pristine cache",
            dose_ladder=ARMS,
            eps0_arm="bitwise identity control: same code path, no draws; "
                     "must equal the matched control stream",
            readouts=dict(
                cost="clean-judge tail CE (448..511, clean full recompute) "
                     "of the corruption stream minus the MATCHED control "
                     "stream, per row then mean over B=8",
                attractor="clean-judge gap = judge CE minus the run's own "
                          "online tail CE (e080 off-manifold bar 1.0 nat; "
                          "vzero calibration +5.02) + tail entropy "
                          "(fluency spot-check for basin re-entry)",
                refs=dict(e075_wholeband_vzero=e075_ref,
                          e089_k200_continuous=e089_ref))),
            bars=dict(
                non_monotone=f"cost(0.05)/0.05 > {PERNAT_FACTOR} * "
                             f"cost(0.5)/0.5 AND cost(1.0) <= cost(0.25)",
                monotone="every consecutive pair of the ordered means "
                         "increases (0.05 < 0.1 < 0.25 < 0.5 < 1.0)",
                null=f"every |mean cost| <= {NULL_LEVEL}",
                order="non-monotone -> monotone -> null -> honest texture",
                ci_note="point rules on means; paired-row bootstrap CIs "
                        "reported alongside (e089 convention)"),
            registered_numbers=dict(pernat_factor=PERNAT_FACTOR,
                                    null_level=NULL_LEVEL,
                                    cj_gap_bar=CJ_GAP_BAR)),
        gates=gates,
        control_run=dict(clean_judge_tail_ce=J_ctl.tolist(),
                         online_tail=float(S[0.0]["online_tail"].mean()),
                         cj_gap=cj_gap_ctl,
                         ent_tail=float(ent_ctl[:, TAIL_STEP0:].mean())),
        references=dict(e075_wholeband_vzero=e075_ref,
                        e089_k200_continuous=e089_ref,
                        e080_vzero_cj=E080_VZERO_CJ,
                        e080_vzero_cj_gap=E080_VZERO_CJ_GAP),
        summary=dict(
            per_eps={str(e): dict(
                mean=per[e]["mean"], ci=per[e]["ci"],
                per_nat=per[e]["per_nat"],
                cost_per_row=per[e]["cost_per_row"],
                n_worse=per[e]["n_worse"],
                n_stream_identical=per[e]["n_stream_identical"],
                first_div=per[e]["first_div"],
                first_div_per_row=per[e]["first_div_per_row"],
                n_rows_diverged=per[e]["n_rows_diverged"],
                clean_judge=per[e]["clean_judge"]) for e in ARMS},
            contrasts=dict(
                pernat_05_vs_2x_05=dict(
                    val=float(c05 / 0.05 - PERNAT_FACTOR * c5 / 0.5),
                    ci=ci_A, fires=clauseA,
                    rule=f"cost(0.05)/0.05 > {PERNAT_FACTOR}*cost(0.5)/0.5"),
                reentry_10_vs_25=dict(
                    val=float(c10 - c25), ci=ci_B, fires=clauseB,
                    rule="cost(1.0) <= cost(0.25)"),
                consecutive_diffs={k: dict(val=v, ci=ci_cons[k])
                                   for k, v in
                                   zip([f"{a}->{b_}" for a, b_ in
                                        zip(EPS_LADDER[:-1], EPS_LADDER[1:])],
                                       [c1 - c05, c25 - c1, c5 - c25,
                                        c10 - c5])}),
            attractor=dict(gap_bar=CJ_GAP_BAR, fires=gap_fire,
                           first_firing_eps=first_gap,
                           control_gap=cj_gap_ctl)),
        registered_decision=dict(
            clause=clause, verdict=verdict, attractor_note=attractor_note,
            numbers=dict(means={str(e): m[e] for e in ARMS},
                         per_nat={str(e): m[e] / e for e in EPS_LADDER},
                         clauseA=clauseA, clauseB=clauseB,
                         clauseA_vacuous=vacuous_A, clauseB_vacuous=vacuous_B,
                         nonmonotone=nonmonotone, monotone=monotone,
                         null_texture=null_tex, all_nonpos=all_nonpos,
                         honesty_guard="added after the first pass (all-"
                         "nonpositive costs made the frozen inequalities "
                         "fire vacuously); frozen bars unchanged, still "
                         "reported literally",
                         ci_A=ci_A, ci_B=ci_B)),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "coherence_gap.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    n = dec["numbers"]
    S = M["summary"]
    refs = M["references"]
    means = {e: n["means"][str(e)] for e in ARMS}
    cis = {e: S["per_eps"][str(e)]["ci"] for e in ARMS}
    rows = {e: np.asarray(S["per_eps"][str(e)]["cost_per_row"]) for e in ARMS}
    cj = {e: S["per_eps"][str(e)]["clean_judge"] for e in ARMS}
    m05, m10, m25, m5, m100 = (means[0.05], means[0.1], means[0.25],
                               means[0.5], means[1.0])
    run_cols = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: THE cost-vs-eps curve, both shapes annotated + gap inset
    xe = np.array(EPS_LADDER, float)
    for e in EPS_LADDER:
        ax1.scatter(np.full(B, e) + np.linspace(-0.006, 0.006, B), rows[e],
                    s=34, color=run_cols[np.arange(B)], alpha=0.7, zorder=3)
    mm = [means[e] for e in EPS_LADDER]
    err = [[max(0.0, means[e] - cis[e][0]) for e in EPS_LADDER],
           [max(0.0, cis[e][1] - means[e]) for e in EPS_LADDER]]
    ax1.errorbar(xe, mm, yerr=err, fmt="o-", color="k", lw=2.2, ms=9,
                 capsize=5, zorder=4, label="measured mean (paired-row CI)")
    # annotated shape guides (illustrations anchored to the measured endpoints)
    if m100 > 0:
        ax1.plot(xe, m100 * xe, "--", color="tab:blue", lw=1.8,
                 label=f"MONOTONE guide: cost ∝ eps (anchored to eps=1.0 "
                       f"{m100:+.3f})")
        if m05 > 0:
            vg, pk = 0.30 * min(m05, m25, m5, m100), 0.35
            xs = np.linspace(0.05, 1.0, 200)
            gu = np.where(xs < pk,
                          vg + (m05 - vg) * ((pk - xs) / (pk - 0.05)) ** 2,
                          vg + (m100 - vg) * ((xs - pk) / (1.0 - pk)) ** 2)
            ax1.plot(xs, gu, "--", color="tab:red", lw=1.8,
                     label="NON-MONOTONE guide: U-shape (valley eps≈0.35)")
    else:
        ax1.text(0.07, 0.5, "cost(1.0) <= 0: NO rung damages —\n"
                 "damage-shape guides undefined\n(honesty guard)",
                 transform=ax1.transAxes, fontsize=9, color="tab:red",
                 va="center", bbox=dict(fc="white", ec="tab:red", alpha=0.85))
    ax1.axhline(refs["e075_wholeband_vzero"], color="tab:green", ls="-.",
                lw=1.4, label=f"e075 whole-band V-zero "
                              f"{refs['e075_wholeband_vzero']:+.3f} (K=32)")
    ax1.axhline(refs["e089_k200_continuous"], color="tab:olive", ls="-.",
                lw=1.4, label=f"e089 k=200 continuous removal "
                              f"{refs['e089_k200_continuous']:+.3f} (K=1)")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xscale("log")
    ax1.set_xticks(xe)
    ax1.set_xticklabels([str(e) for e in EPS_LADDER])
    ax1.set_xlabel("eps (x median-entry-norm, log scale)")
    ax1.set_ylabel("clean-judge tail cost (nats)")
    ax1.set_title(f"E096-1 — THE dose ladder | clause: {dec['clause']}",
                  fontsize=10)
    ax1.legend(fontsize=8.5, loc="upper left")
    # the clean-judge gap inset (the attractor signature readout)
    axi = ax1.inset_axes([0.60, 0.10, 0.36, 0.34])
    gaps = [cj[e]["gap_vs_online"] for e in ARMS]
    axi.bar(np.arange(len(ARMS)), gaps, 0.6,
            color=["tab:gray"] + ["tab:red"] * len(EPS_LADDER), alpha=0.8)
    axi.axhline(CJ_GAP_BAR, color="k", ls="--", lw=1.0)
    axi.text(0.1, CJ_GAP_BAR, f" gap bar {CJ_GAP_BAR}", fontsize=6.5, va="top")
    axi.set_xticks(np.arange(len(ARMS)))
    axi.set_xticklabels([str(e) for e in ARMS], fontsize=6)
    axi.set_ylabel("cj gap (nats)", fontsize=7)
    axi.tick_params(labelsize=6.5)
    axi.set_title("clean-judge gap vs eps (control first)", fontsize=7)

    # ---- panel 2: per-nat cost + the registered 2x contrast
    pn = [means[e] / e for e in EPS_LADDER]
    ax2.plot(xe, pn, "s-", color="tab:purple", lw=2.2, ms=9,
             label="cost(eps)/eps (nats per nat of perturbation)")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xscale("log")
    ax2.set_xticks(xe)
    ax2.set_xticklabels([str(e) for e in EPS_LADDER])
    ax2.set_xlabel("eps (log scale)")
    ax2.set_ylabel("per-nat cost")
    ax2.set_title("E096-2 — PER-NAT cost | clause A "
                  f"[{'FIRES' if n['clauseA'] else 'no'}]: "
                  f"pn(0.05) {pn[0]:+.2f} > {PERNAT_FACTOR} x pn(0.5) "
                  f"{PERNAT_FACTOR * pn[3]:+.2f}?", fontsize=10)
    ax2.legend(fontsize=9)

    # ---- panel 3: clean-judge + online tail CE per arm (the gap opening)
    cjm = [cj[e]["mean"] for e in ARMS]
    onm = [cj[e]["online_tail"] for e in ARMS]
    x3 = np.arange(len(ARMS))
    ax3.plot(x3, cjm, "o-", color="tab:red", lw=2.2, ms=9,
             label="clean-judge tail CE")
    ax3.plot(x3, onm, "s--", color="tab:blue", lw=1.8, ms=7,
             label="online (self-scored) tail CE")
    ax3.axhline(refs["e080_vzero_cj"], color="tab:orange", ls=":", lw=1.4,
                label=f"e080 vzero clean-judge {refs['e080_vzero_cj']:.2f}")
    for x, v in zip(x3, cjm):
        ax3.text(x, v + 0.06, f"{v:.2f}", ha="center", fontsize=7.5)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([f"eps={e}" for e in ARMS], fontsize=8.5)
    ax3.set_ylabel("tail CE (nats)")
    ax3.set_title("E096-3 — judge vs self-score: the off-manifold gap "
                  "per dose", fontsize=10)
    ax3.legend(fontsize=8.5)

    # ---- panel 4: tail entropy (the fluency spot-check for basin re-entry)
    ent = [cj[e]["ent_tail"] for e in ARMS]
    ax4.bar(x3, ent, 0.55, color="tab:cyan", alpha=0.85, edgecolor="k",
            lw=0.5)
    ax4.axhline(ent[0], color="k", lw=0.9)
    for x, v in zip(x3, ent):
        ax4.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax4.set_xticks(x3)
    ax4.set_xticklabels([f"eps={e}" for e in ARMS], fontsize=8.5)
    ax4.set_ylabel("tail entropy (nats)")
    ax4.set_title("E096-4 — fluency spot-check: tail entropy per dose "
                  "(re-entry = control-level entropy AND low judge CE)",
                  fontsize=9.5)

    # ---- panel 5: stream texture — first divergence vs eps
    for e in EPS_LADDER:
        fdr = [d if d is not None else T_TOTAL
               for d in S["per_eps"][str(e)]["first_div_per_row"]]
        ax5.scatter(np.full(B, math.log10(e)) + np.linspace(-0.02, 0.02, B),
                    fdr, s=36, alpha=0.8,
                    color="tab:red" if any(d is None for d in
                                           S["per_eps"][str(e)]
                                           ["first_div_per_row"])
                    else "tab:blue")
    ax5.axhline(KEY_T[0], color="tab:green", ls="-.", lw=1.2,
                label="tail start 448")
    ax5.axhline(T_TOTAL, color="gray", ls=":", lw=1.2)
    ax5.text(math.log10(0.05), T_TOTAL - 6, "512 = never diverged",
             fontsize=8)
    ax5.set_xticks([math.log10(e) for e in EPS_LADDER])
    ax5.set_xticklabels([str(e) for e in EPS_LADDER])
    ax5.set_xlabel("eps (log10 scale)")
    ax5.set_ylabel("first position arm != control")
    ax5.set_ylim(T_INT, T_TOTAL + 8)
    ax5.set_title("E096-5 — stream divergence vs dose", fontsize=10)
    ax5.legend(fontsize=9)

    # ---- panel 6: verdict text
    ax6.axis("off")
    lines = [
        "REGISTERED (frozen from the tasking):",
        f"  NON-MONOTONE: cost(0.05)/0.05 > {PERNAT_FACTOR}*cost(0.5)/0.5 "
        f"AND cost(1.0) <= cost(0.25)",
        f"  MONOTONE (kill): 0.05 < 0.1 < 0.25 < 0.5 < 1.0 (strict rise)",
        f"  NULL: every |mean| <= {NULL_LEVEL}; else honest texture",
        "  GUARD: all rungs <= 0 -> NO-DAMAGE clause (vacuous firings "
        "flagged)",
        "",
        "CURVE (means, paired-row CI):",
    ] + [
        f"  eps={e}: {means[e]:+.4f} {_fmt_ci(cis[e])}"
        + (f"  (per-nat {means[e] / e:+.3f})" if e > 0 else "  (identity arm)")
        for e in ARMS
    ] + [
        "",
        f"CONTRASTS: A [{'FIRES' if n['clauseA'] else 'no'}] "
        f"{S['contrasts']['pernat_05_vs_2x_05']['val']:+.3f} "
        f"{_fmt_ci(n['ci_A'])} | B [{'FIRES' if n['clauseB'] else 'no'}] "
        f"{S['contrasts']['reentry_10_vs_25']['val']:+.4f} "
        f"{_fmt_ci(n['ci_B'])}",
        f"REFS: e075 whole-band {refs['e075_wholeband_vzero']:+.4f} (K=32) | "
        f"e089 k=200 {refs['e089_k200_continuous']:+.4f} (K=1)",
        "",
        f"ATTRACTOR: control gap "
        f"{S['attractor']['control_gap']:+.3f}; first firing eps = "
        f"{S['attractor']['first_firing_eps']}",
        "  gaps: " + " ".join(f"{e}:{cj[e]['gap_vs_online']:+.2f}"
                              for e in ARMS),
        "",
        f"DECISION [{dec['clause']}]:",
    ] + [f"  {wd}" for wd in _wrap(dec["verdict"], 98)]
    ax6.text(0.02, 0.97, "E096 — the coherence-gap dose ladder (P-B / T051)",
             fontsize=13, weight="bold", va="top")
    for i, tx in enumerate(lines):
        ax6.text(0.02, 0.935 - i * 0.0295, tx, fontsize=8.4, va="top",
                 family="monospace")

    fig.suptitle("E096 — coherence-gap dose ladder: single-shot anchor-band V "
                 f"corruption at eps x median-norm | clause: {dec['clause']} "
                 f"| cj-gap fires first at eps="
                 f"{S['attractor']['first_firing_eps']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
