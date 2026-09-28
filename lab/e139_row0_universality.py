"""E139 — ROW-0 UNIVERSALITY + 183-ROBUSTNESS (T077's central open question;
REGISTERED before compute — bars verbatim from the coordinator dispatch).

WHY (T077, verbatim context): e131 showed (a) a jitter-consolidated fact's
positional key is wpe ROW 0 (content strength 0.732, the only content-positive
row; D-all+row-0 collapses expression -97%; the old address row 129 became a
suppressive brake), and (b) the e120 corpus/self splice arms consolidated at
row 183 to p(Z) 0.989/0.988 (an address the old battery never read). OPEN:
is row 0 the UNIVERSAL readout key for any consolidated fact (including a
183-site consolidation), or was re-keying-to-0 a property of the jitter road
alone? This run adjudicates on e131's saved nets (eval-only) + one permitted
regeneration (arm c, the rider).

NETS (loaded from disk, gated bit-level vs e131's stored tables — NOTHING
regenerated for probes 1-4):
  * runs/checkpoints/e131_arm_a_self_spliced.pt   (e120 arm a verbatim)
  * runs/checkpoints/e131_arm_b_corpus_spliced.pt (e120 arm b verbatim)
  * runs/checkpoints/e131_consolidated_e113.pt    (row-0-positive reference)
  * runs/checkpoints/e082_b43_install.pt          (the splice-line base)

PROBES:
  (1) ROW-0 CONTENT TEST on the splice arms — e131's e116 mean/zero-arm
      census instrument verbatim (mean arm: wpe[r] <- mean of all rows; zero
      arm: wpe[r] <- 0; drop = base - arm; strength = min(mean-drop,
      zero-drop); content = both drops > 0 and ratio >= 0.5; controls =
      e131's CONTROL_ROWS), with the readout at the arms' OWN fact geometry
      (read_fact_position, p(Z) at position 183) because the old install-60
      battery readout sits at floor on these arms (0.016/0.008). Census rows
      = e131's CENSUS_ROWS + row 183 (the splice site: the site-locked side
      of the content test).
  (2) DELETION BATTERY at the 183-geometry on the splice arms: none /
      D-183 / D-band{121,125,129,133,137} / D-band+183 / D-row-0 /
      D-row-0+183 — p(Z)@183 per cell. Report-only scaffold-matched controls
      (e131's d_r1 convention): D-row-1, D-row-1+183, and D-129 (probe 4).
  (3) NOVEL-CONTEXT GENERALIZATION at 183: held-out corpus windows (never in
      the splice exposure stream) with fact segments at the same 183 columns
      (e120's splice construction machinery); p(Z)@183. Closes e131's
      honesty note (d): training-geometry read vs generalization.
  (4) ROW-129 BRAKE SCOPE on the splice arms: row-129 mean/zero replacement
      drops at the 183-readout (from probe 1's census) + the D-129 deletion
      cell — does the brake reach across homes?
  (5) RIDER (the ONE permitted regeneration): arm-c verbatim-dreams net
      (e120 recipe verbatim: dream harvest seeds 12110-12113, corpus-free
      self dreams, full-continuation mask, AdamW (0.9,0.95) wd 0.1 constant
      lr 1e-3 clip 1.0, batch 16+16, 300 steps, seed 12101, CPU); read the
      fact's expression at the ACTUAL dream ZEPHYRA positions the training
      used — is arm c ALSO instrument-blind (did the dreams consolidate
      somewhere the band-battery never read)?

REGISTERED PREDICTION (coordinator, VERBATIM — no bar shopping):
  * ROW-0-UNIVERSAL fires if: splice-arm fact expression drops >= 50% under
    D-row-0 (or row-0 content test positive at >= 2x control band) — row 0
    is the universal readout key.
  * SITE-LOCKED fires if: D-183 kills >= 80% of splice expression AND
    row-0 test null — the splice arms were ordinary installs at a new
    address; re-keying-to-0 belongs to the jitter road alone.
  * HYBRID fires if: both partial (two-door access).
  * No bar shopping; texture => say TEXTURE with numbers.

OPERATIONALIZATIONS (fixed before compute):
  * "splice fact expression" = pz_onset_mean (mean p(Z) at position 183)
    from e131's read_fact_position on the arm's own training pool (a-pool
    for arm a, b-pool for arm b; pools rebuilt with the e120 construction
    verbatim and gated against e131's stored numbers).
  * "drops >= 50% under D-row-0" = (none - d_r0)/none >= 0.50, credited
    only when it EXCEEDS the scaffold-matched D-row-1 control by the same
    bar: (d_r1 - d_r0)/none >= 0.50 (e131's scaffold-matched convention —
    guards against generic position-0 embedding damage; raw drops always
    reported alongside).
  * "row-0 content test positive at >= 2x control band" = e116 criterion
    (mean-drop > 0, zero-drop > 0, min/max arm ratio >= 0.5) at the
    183-geometry readout AND strength >= 2 x max control-row strength.
  * "row-0 test null" = not the above (fails the criterion or the 2x bar).
  * "D-183 kills >= 80%" = (none - d183)/none >= 0.80.
  * "both partial" = row-0 partial: (0.10 <= net r0 drop < 0.50, net of the
    row-1 control) OR (content criterion met but strength < 2x control
    max); 183 partial: 0.10 <= D-183 kill < 0.80. If BOTH full conditions'
    ingredients co-occur (row-0 full AND D-183 kill >= 0.80 — both doors
    individually lethal) => HYBRID-STRONG, reported under HYBRID with
    numbers.
  * Adjudication order per arm: (1) r0_full and not site_full ->
    ROW-0-UNIVERSAL; (2) site_full and not r0_full -> SITE-LOCKED;
    (3) r0_full and site_full -> HYBRID-STRONG; (4) both partial -> HYBRID;
    (5) else TEXTURE. Overall verdict = the per-arm verdict IFF both arms
    agree; arms disagree -> TEXTURE with per-arm numbers.
  * Probe-3 bars: GENERALIZES if pz@183(novel) >= 0.5 AND >= 0.5 x the
    training-pool pz@183; PARTIAL if >= 2 x the base-net floor on the same
    pool but below GENERALIZES; FAILS otherwise.
  * Rider bars: RIDER-SIGNAL if arm-c p(Z)@dream-onsets >= 0.5 AND >= 2x
    base-net-at-the-same-positions; RIDER-NULL if <= 2x base; else PARTIAL.
  * Honesty guards (report-only, never bars): D-row-1 / D-row-1+183
    scaffold controls, CE_R per deletion cell, probe-3 construction-leakage
    gates (see recipe notes 6).

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import;
torch.set_num_threads(8), sequential — another agent owns the GPU). Only the
arm-c rider trains (300 steps, ~6 min; the one permitted regeneration).
Arm-c checkpoint saved to runs/checkpoints/e139_arm_c_verbatim_dreams.pt.

Outputs: runs/e139/{metrics.json, row0_universality.png}.
No NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e139_row0_universality.py     (E139_SMOKE=1 -> plumbing shakedown)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (e119 owns the GPU)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(8)                              # 8 threads max (lab convention)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

# INSTRUMENTS REUSED VERBATIM FROM E131 (import = provenance):
#   read_fact_position (183-geometry read), deleted_wpe (D2 subtractive
#   row-zero + confinement gate), battery_cell / battery_pz, ce_fixed_cpu,
#   val_windows, load_cpu / evl_load, free_run_batch (e121/e120 harvest),
#   finetune_arm (e109/e113/e120 recipe), build_fact_segments + splice_pool
#   (e120 splice construction), and the geometry/seed constants.
import e131_rekeying_census as E131                   # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E139_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE, BLOCK = E131.PRE, E131.BLOCK
HOSTS = E131.HOSTS
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
CK_A = CKPT_DIR / "e131_arm_a_self_spliced.pt"
CK_B = CKPT_DIR / "e131_arm_b_corpus_spliced.pt"
CK_CONS = CKPT_DIR / "e131_consolidated_e113.pt"
CK_B43 = CKPT_DIR / "e082_b43_install.pt"

# geometry (e120/e131 verbatim — also read from the E131 module)
SPLICE_ADDR_ROW, Z_XCOL = E131.SPLICE_ADDR_ROW, E131.Z_XCOL
FACT_PRE, FACT_POST = E131.FACT_PRE, E131.FACT_POST
DREAM_SEEDS, N_DREAMS_PER_PROMPT, N_PROMPTS = E131.DREAM_SEEDS, 4, 30
DREAM_LEN = E131.DREAM_LEN
CORP_CONT_SEED = E131.CORP_CONT_SEED
CONS_SEED_SPLICE = E131.CONS_SEED_SPLICE            # 12101 (e120 arms a/b/c)
FT_LR = E131.FT_LR

CONTROL_ROWS = E131.CONTROL_ROWS                    # (1,2,3,4,5,6,60,100,118,119,120)
E113_ADDR_ROWS = E131.E113_ADDR_ROWS                # (121,125,129,133,137)
ADDR_BAND = E131.ADDR_BAND                          # 121..137
CENSUS_ROWS = (0,) + CONTROL_ROWS + ADDR_BAND + (SPLICE_ADDR_ROW,)
SMOKE_CENSUS = (0, 1, 2, 60, 118, 120, 121, 125, 129, 133, 137, 183)

# registered deletion battery at the 183-geometry
DELS_REG = {"none": (),
            "d_183": (SPLICE_ADDR_ROW,),
            "d_band": E113_ADDR_ROWS,
            "d_band_183": tuple(sorted(set(E113_ADDR_ROWS) | {SPLICE_ADDR_ROW})),
            "d_r0": (0,),
            "d_r0_183": (0, SPLICE_ADDR_ROW)}
# report-only controls (never enter adjudication)
DELS_CTL = {"d_129": (129,),
            "d_r1": (1,),
            "d_r1_183": (1, SPLICE_ADDR_ROW)}

# novel-context pool seeds (fresh; e-series used 10901/12101-12113/1337/24331/26502)
NOV_TRAIN_SEED, NOV_VAL_SEED = 13903, 13905
NOV_N = 12 if SMOKE else 120
PROTECT_MARGIN = 140

# gates / references
G_BIT_TOL = 1e-6                                    # bit-identity vs e131's stored tables
G_FALLBACK_TOL = 0.05                               # e113 convention fallback
G_B43_REF = E131.G_B43_REF
E131_M = E43.REPO / "runs" / "e131" / "metrics.json"
E120_M = E43.REPO / "runs" / "e120" / "metrics.json"
HARVEST_REF_Z = 34                                  # e120/e121/e131 own-dream ZEPHYRA count

REGISTERED_PREDICTION = {
    "row_0_universal": "ROW-0-UNIVERSAL fires if: splice-arm fact expression "
                       "drops >= 50% under D-row-0 (or row-0 content test "
                       "positive at >= 2x control band) — row 0 is the "
                       "universal readout key.",
    "site_locked": "SITE-LOCKED fires if: D-183 kills >= 80% of splice "
                   "expression AND row-0 test null — the splice arms were "
                   "ordinary installs at a new address; re-keying-to-0 "
                   "belongs to the jitter road alone.",
    "hybrid": "HYBRID fires if: both partial (two-door access).",
    "no_bar_shopping": "No bar shopping; texture => say TEXTURE with numbers.",
    "operationalizations": "expression = pz_onset_mean at position 183 "
                           "(e131 read_fact_position) on the arm's own "
                           "training pool; r0 drop credited only if it "
                           "exceeds the scaffold-matched D-row-1 control by "
                           "the same bar; row-0 content-positive = e116 "
                           "criterion AND strength >= 2x max control-row "
                           "strength; both-full (row-0 full AND D-183 kill "
                           ">= 80%) => HYBRID-STRONG under HYBRID; arms must "
                           "agree or the verdict is TEXTURE.",
}

recipe_notes: list[str] = [
    "Instruments imported VERBATIM from lab/e131_rekeying_census.py "
    "(read_fact_position, deleted_wpe, battery_cell, ce_fixed_cpu, "
    "val_windows, load_cpu/evl_load, free_run_batch, finetune_arm, "
    "build_fact_segments, splice_pool, constants) — the import is the "
    "provenance.",
    "Nets for probes 1-4 are e131's saved checkpoints, gated bit-level "
    "(1e-6) against e131's stored tables; nothing regenerated. The ONLY "
    "training in e139 is the permitted arm-c rider regeneration.",
    "Readout adaptation (registered): e131's probe-2 census read the "
    "install-60 g0 battery (the consolidated line's own address); the splice "
    "arms' battery-band sits at floor (0.016/0.008), so the census readout "
    "here is the 183-geometry read — the arms' own fact address. Criterion "
    "arithmetic (mean/zero arms, strength=min, ratio>=0.5, control rows) is "
    "e116/e131 verbatim.",
    "Census row set = e131's CENSUS_ROWS + row 183 (the splice site — gives "
    "the SITE-LOCKED side a content test: mean-arm vs zero-arm on row 183 "
    "separates identity-carrying from presence-needed).",
    "Report-only deletion controls beyond the 6 registered cells: D-129 "
    "(probe 4 brake scope), D-row-1 / D-row-1+183 (scaffold-matched, e131's "
    "d_r1 convention). Adjudication uses only the registered cells.",
    "Novel-context pools (probe 3): fresh seeds 13903 (train split) / 13905 "
    "(val split); windows rejected if they contain 'Z' or overlap any "
    "protected span: all 90 install/held protocol occurrences (+/-140 "
    "chars), the arm-b training filler spans (seed-12103 draws reproduced "
    "exactly), and the val r_eval spans (seed-26502 reproduction). Honesty "
    "note: exclusion covers the EXPOSURE stream + protocol spans; the "
    "fine-tune's RANDOM anchor draws sample the whole train split (carry no "
    "fact; ~300x8x256 chars ~ 0.1% coverage) and cannot be exhaustively "
    "excluded.",
    "Rider: arm c regenerated verbatim from the B43 base (e120 recipe: "
    "harvest seeds 12110-12113 on the base net, full-continuation mask, "
    "AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0, batch 16 exposure + "
    "16 anchor, 300 steps, seed 12101, CPU) and gated vs e120's stored "
    "battery table (bit-repro expected; e131 precedent: arms a/b gated 0.0).",
    "wpe universe: 256 rows (block_size 256, e131's note) — same line.",
]


# ------------------------------------------------------------------ instruments

def read183(net: TinyGPT, pool: torch.Tensor, name_ids, zid: int) -> dict:
    """e131's read_fact_position VERBATIM (via import) — full dict."""
    return E131.read_fact_position(net, pool, name_ids, zid)


def read183_scalar(net: TinyGPT, pool: torch.Tensor, name_ids, zid: int) -> float:
    """Scalar headline: mean p(Z) at position 183 (e131's probe-1 headline)."""
    return read183(net, pool, name_ids, zid)["pz_onset_mean"]


def row_census_at183(net: TinyGPT, pool: torch.Tensor, name_ids, zid: int,
                     rows) -> dict:
    """e131's Phase-C census loop VERBATIM (mean-arm / zero-arm / restore),
    readout swapped to the 183-geometry read (registered adaptation)."""
    net.eval()
    base = read183_scalar(net, pool, name_ids, zid)
    w = net.wpe.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    m_d, z_d = {}, {}
    for r in rows:
        w.copy_(orig); w[r] = mean_row
        m_d[r] = base - read183_scalar(net, pool, name_ids, zid)
        w.copy_(orig); w[r] = 0.0
        z_d[r] = base - read183_scalar(net, pool, name_ids, zid)
    w.copy_(orig)
    rows_d = {str(r): {"mean": float(m_d[r]), "zero": float(z_d[r]),
                       "ratio": float(min(m_d[r], z_d[r]) / max(m_d[r], z_d[r]))
                              if max(m_d[r], z_d[r]) > 0 else 0.0,
                       "strength": float(min(m_d[r], z_d[r])),
                       "content": bool(m_d[r] > 0 and z_d[r] > 0 and
                                       min(m_d[r], z_d[r]) /
                                       max(m_d[r], z_d[r]) >= 0.5)}
              for r in rows}
    assert torch.equal(w, orig), "census failed to restore wpe"
    return {"base_pz_onset": base, "rows": rows_d}


@torch.no_grad()
def read_at_occurrences(net: TinyGPT, pool_x: torch.Tensor, occs, zid: int,
                        bs=30) -> dict:
    """RIDER instrument (modeled on e131's read_fact_position, readout
    arithmetic verbatim): p(true name char) at each dream-ZEPHYRA occurrence.
    occs = list of (window_idx, onset_col); name spans cols c..c+6; read
    positions c-1..c+5 (position t predicts col t+1)."""
    net.eval()
    by_win: dict[int, list[int]] = {}
    for k, c in occs:
        by_win.setdefault(int(k), []).append(int(c))
    wins = sorted(by_win)
    pzs, allp = [], []
    for i in range(0, len(wins), bs):
        chunk = wins[i:i + bs]
        w = pool_x[torch.tensor(chunk)]
        lg, _ = net(w)
        pr = F.softmax(lg, -1)
        for j, k in enumerate(chunk):
            for c in by_win[k]:
                pzs.append(float(pr[j, c - 1, int(zid)]))
                for t in range(len(NAME)):
                    allp.append(float(pr[j, c - 1 + t, int(pool_x[k, c + t])]))
    pz_t = torch.tensor(pzs)
    allp_t = torch.tensor(allp)
    return {"n_occ": len(occs),
            "pz_onset_mean": float(pz_t.mean()) if len(pzs) else float("nan"),
            "pz_onset_frac_ge_0.5": float((pz_t >= 0.5).float().mean()) if len(pzs) else float("nan"),
            "pname_mean_over7": float(allp_t.mean()) if len(allp) else float("nan"),
            "pname_frac_ge_0.5": float((allp_t >= 0.5).float().mean()) if len(allp) else float("nan")}


def find_name_occ(pool_x: torch.Tensor, itos, name: str = NAME,
                  lo: int = PRE, hi: int = BLOCK) -> list[tuple[int, int]]:
    """ZEPHYRA occurrences fully inside the continuation region [lo, hi)."""
    occs = []
    for k, w in enumerate(pool_x):
        t = "".join(itos[int(i)] for i in w.tolist())
        s = 0
        while True:
            i = t.find(name, s)
            if i < 0:
                break
            if i >= lo and i + len(name) <= hi:
                occs.append((k, i))
            s = i + 1
    return occs


def novel_windows(ids: torch.Tensor, text: str, protected: list[tuple[int, int]],
                  n: int, seed: int) -> tuple[torch.Tensor, dict]:
    """Probe-3 pool source: n corpus windows of BLOCK chars, Z-free, outside
    every protected span. Returns (windows, construction report)."""
    g = torch.Generator().manual_seed(seed)
    out, tries, rej_z, rej_ov = [], 0, 0, 0
    while len(out) < n and tries < 2000 * n:
        s = int(torch.randint(len(ids) - BLOCK - 1, (1,), generator=g))
        tries += 1
        txt = text[s: s + BLOCK + 1]
        if "Z" in txt:
            rej_z += 1
            continue
        if any(s < b and s + BLOCK > a for a, b in protected):
            rej_ov += 1
            continue
        out.append(ids[s: s + BLOCK])
    if len(out) < n:
        raise RuntimeError(f"novel pool: only {len(out)}/{n} windows in {tries} tries")
    rep = {"n": len(out), "tries": tries, "rejected_contains_Z": rej_z,
           "rejected_overlap_protected": rej_ov, "seed": seed,
           "source_chars": f"{len(ids)}", "note": "Z-free, outside protected spans"}
    return torch.stack(out), rep


def novel_splice(windows: torch.Tensor, fact_segs: torch.Tensor) -> torch.Tensor:
    """e120 splice geometry on novel windows: prompt = win[:130], filler =
    win[130:] with cols 42..73 replaced by fact[k] — ZEPHYRA lands at x-cols
    184..190 exactly as in training (e131.splice_pool semantics, coherent
    prompt+filler from one corpus window)."""
    n = windows.shape[0]
    pr = windows[:, :PRE]
    filler = windows[:, PRE:]
    if filler.shape[1] != DREAM_LEN:
        raise RuntimeError("novel filler width mismatch")
    fs = fact_segs[torch.arange(n) % fact_segs.shape[0]]
    cont = torch.cat([filler[:, :E131.SPLICE_AT], fs,
                      filler[:, E131.SPLICE_AT + E131.FACT_LEN:]], 1)
    return torch.cat([pr, cont], 1)


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e139_smoke" if SMOKE else "e139")
    log(f"E139 ROW-0 UNIVERSALITY + 183-ROBUSTNESS (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, threads {torch.get_num_threads()}, "
        f"cuda visible = {torch.cuda.is_available()}")

    refs = {}
    for key, path in (("e131", E131_M), ("e120", E120_M)):
        refs[key] = json.loads(Path(path).read_text(encoding="utf-8")) \
            if Path(path).exists() else None
        if refs[key] is None:
            raise RuntimeError(f"missing reference metrics {path}")

    # ---------------- protocol rebuild (e131 Phase A construction verbatim)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + E131.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30")

    name_ids = corpus.encode(NAME)
    bat_ids = {}
    for j in E131.GEO_ORDER:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    f_eval = bat_ids[(0, "install60")]
    r_eval_x, r_eval_y = E131.val_windows(val_ids, val_text, 60, E131.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])

    # ---------------- load the gated nets
    net_b43 = E131.load_cpu(CK_B43)
    evl = E131.evl_load(net_b43.state_dict())
    bz_b43 = E131.battery_cell(evl, f_eval, zid)
    G_B43 = {"battery_pz": bz_b43["mean_pz"], "ref": G_B43_REF,
             "tol": G_BIT_TOL,
             "pass": bool(abs(bz_b43["mean_pz"] - G_B43_REF) < G_BIT_TOL)}
    log(f"G_B43 base battery p(Z) {bz_b43['mean_pz']:.10f} (ref "
        f"{G_B43_REF:.10f}): {'PASS' if G_B43['pass'] else 'FAIL'}")
    if not G_B43["pass"]:
        raise RuntimeError("B43 checkpoint failed its gate")
    del evl

    nets = {"a_self_ctx_spliced": E131.load_cpu(CK_A),
            "b_corpus_ctx_spliced": E131.load_cpu(CK_B),
            "consolidated_e113": E131.load_cpu(CK_CONS)}

    # ---------------- rebuild the training pools (e120 construction verbatim)
    prompts = [train_text[p - PRE: p] for p, _ in held_occ][:N_PROMPTS]
    prompt_ids = torch.stack([corpus.encode(c) for c in prompts])
    wins = []
    for s in DREAM_SEEDS[:N_DREAMS_PER_PROMPT]:
        wins.append(E131.free_run_batch(net_b43, prompt_ids, DREAM_LEN, seed=s))
    dream_ids = torch.cat(wins)                        # arm-c pool (verbatim)
    z_in_dreams = sum("".join(itos[int(i)] for i in w[PRE:]).count(NAME)
                      for w in dream_ids)
    log(f"dream harvest: {dream_ids.shape[0]} windows, ZEPHYRA count "
        f"{z_in_dreams} (ref {HARVEST_REF_Z})")

    g = torch.Generator().manual_seed(CORP_CONT_SEED)
    src = torch.randint(len(train_ids) - DREAM_LEN - 1,
                        (N_DREAMS_PER_PROMPT, len(prompts)), generator=g)
    cont = torch.stack([train_ids[s: s + DREAM_LEN] for s in src.flatten()])
    fact_segs = E131.build_fact_segments(install_occ, train_text, corpus.encode)
    fact_segs_held = E131.build_fact_segments(held_occ, train_text, corpus.encode)
    pool_a_x = E131.splice_pool(prompt_ids, dream_ids[:, PRE:], fact_segs)
    pool_b_x = E131.splice_pool(prompt_ids, cont, fact_segs)
    pools = {"a_self_ctx_spliced": pool_a_x, "b_corpus_ctx_spliced": pool_b_x}

    def geo_ok(pool):
        return bool(all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)], name_ids)
                        for w in pool))
    G_GEO = {"z_xcols": [Z_XCOL, Z_XCOL + len(NAME) - 1],
             "a_all": geo_ok(pool_a_x), "b_all": geo_ok(pool_b_x)}
    G_GEO["pass"] = bool(G_GEO["a_all"] and G_GEO["b_all"])
    if not G_GEO["pass"]:
        raise RuntimeError(f"splice geometry gate FAILED: {G_GEO}")
    log(f"G_GEO: ZEPHYRA at x-col {Z_XCOL} (address row {SPLICE_ADDR_ROW}) in "
        f"all training windows: PASS")

    # ---------------- gate the loaded arms vs e131's stored numbers
    G_ARMS = {}
    for tag in ("a_self_ctx_spliced", "b_corpus_ctx_spliced"):
        r = refs["e131"]["probe1_183_geometry_read"][tag]
        d_read = abs(read183_scalar(nets[tag], pools[tag], name_ids, zid)
                     - r["read183"]["pz_onset_mean"])
        d_band = max(abs(E131.battery_cell(nets[tag], bat_ids[(j, "install60")],
                                           zid)["mean_pz"] - v)
                     for j, v in ((j, r["battery_band_pre_deletion"][f"g{j:+d}"])
                                  for j in E131.GEO_ORDER))
        G_ARMS[tag] = {"read183_diff": d_read, "band_max_diff": d_band,
                       "tol": G_BIT_TOL,
                       "pass": bool(max(d_read, d_band) < G_BIT_TOL)}
        log(f"G_ARMS[{tag}]: read183 |d| {d_read:.2e}, battery-band |d| "
            f"{d_band:.2e} -> {'PASS' if G_ARMS[tag]['pass'] else 'FAIL'}")
        if not G_ARMS[tag]["pass"]:
            raise RuntimeError(f"checkpoint gate failed: {tag}")

    cons_ref_cells = {f"g{j:+d}": refs["e131"]["probe4_battery_table"]
                      [f"none__g{j:+d}__install60"]["mean_pz"]
                      for j in E131.GEO_ORDER}
    d_cons = max(abs(E131.battery_cell(nets["consolidated_e113"],
                                       bat_ids[(j, "install60")], zid)["mean_pz"] - v)
                 for j, v in ((j, cons_ref_cells[f"g{j:+d}"])
                              for j in E131.GEO_ORDER))
    G_CONS = {"battery_max_diff": d_cons, "tol": G_BIT_TOL,
              "pass": bool(d_cons < G_BIT_TOL)}
    log(f"G_CONS: consolidated battery |d| {d_cons:.2e} -> "
        f"{'PASS' if G_CONS['pass'] else 'FAIL'}")
    if not G_CONS["pass"]:
        raise RuntimeError("consolidated checkpoint gate failed")

    # =====================================================================
    # PROBE 1 — row-0 content test on the splice arms (census at 183-readout)
    # =====================================================================
    log("--- PROBE 1: row-0 content test on splice arms (183-geometry readout) ---")
    rows_used = SMOKE_CENSUS if SMOKE else CENSUS_ROWS
    census, probe1 = {}, {}
    for tag in ("a_self_ctx_spliced", "b_corpus_ctx_spliced"):
        cen = row_census_at183(nets[tag], pools[tag], name_ids, zid, rows_used)
        census[tag] = cen
        r0 = cen["rows"]["0"]
        ctrl_max = max(cen["rows"][str(r)]["strength"] for r in CONTROL_ROWS
                       if str(r) in cen["rows"])
        content_crit = r0["content"]
        row0_pos_2x = bool(content_crit and r0["strength"] >= 2.0 * max(ctrl_max, 0.0))
        r183 = cen["rows"].get(str(SPLICE_ADDR_ROW))
        r183_pos_2x = bool(r183 and r183["content"] and
                           r183["strength"] >= 2.0 * max(ctrl_max, 0.0))
        probe1[tag] = {"base_pz_onset": cen["base_pz_onset"],
                       "row0": r0,
                       "control_max_strength": ctrl_max,
                       "row0_content_criterion": content_crit,
                       "row0_content_positive_2x": row0_pos_2x,
                       "row0_null": not row0_pos_2x,
                       "row183": r183,
                       "row183_content_positive_2x": r183_pos_2x,
                       "row129": cen["rows"].get("129"),
                       "bar_2x_control": 2.0 * max(ctrl_max, 0.0)}
        log(f"PROBE1[{tag}]: base p@183 {cen['base_pz_onset']:.4f} | row0 "
            f"m/z {r0['mean']:+.4f}/{r0['zero']:+.4f} strength "
            f"{r0['strength']:.4f} vs ctrl-max {ctrl_max:.4f} "
            f"(2x bar {2.0 * max(ctrl_max, 0.0):.4f}) -> "
            f"{'POSITIVE-2X' if row0_pos_2x else 'null' if not content_crit else 'content-but-weak'}"
            f" | row183 m/z "
            f"{cen['rows'][str(SPLICE_ADDR_ROW)]['mean']:+.4f}/"
            f"{cen['rows'][str(SPLICE_ADDR_ROW)]['zero']:+.4f} strength "
            f"{r183['strength']:.4f} -> "
            f"{'SITE-CONTENT-POSITIVE-2X' if r183_pos_2x else 'not-2x'}")

    # reference: the consolidated (jitter-road) net read at the 183 geometry
    # + its row-0 mean/zero there (does row 0 gate even an untrained geometry?)
    cons_ref = {"read183": read183(nets["consolidated_e113"], pool_a_x,
                                   name_ids, zid)}
    cons_cen = row_census_at183(nets["consolidated_e113"], pool_a_x, name_ids,
                                zid, (0,) + (SPLICE_ADDR_ROW,) + (1,))
    cons_ref["row0_mean_zero_at183_readout"] = {
        "base": cons_cen["base_pz_onset"], "row0": cons_cen["rows"]["0"],
        "row1": cons_cen["rows"]["1"], "row183": cons_cen["rows"]["183"]}
    log(f"consolidated-net reference: p(Z)@183 {cons_ref['read183']['pz_onset_mean']:.4f} "
        f"| row0 m/z there {cons_cen['rows']['0']['mean']:+.4f}/"
        f"{cons_cen['rows']['0']['zero']:+.4f}")

    # =====================================================================
    # PROBE 2 — deletion battery at the 183-geometry (+ report-only controls)
    # =====================================================================
    log("--- PROBE 2: deletion battery at the 183-geometry ---")
    table, gates_surg = {}, {}
    for tag in ("a_self_ctx_spliced", "b_corpus_ctx_spliced"):
        sd = nets[tag].state_dict()
        for dl_name, drows in {**DELS_REG, **DELS_CTL}.items():
            if dl_name == "none":
                sd_del = {k: v.clone() for k, v in sd.items()}
                gate = {"rows": [], "pass": True, "note": "no deletion"}
            else:
                sd_del, gate = E131.deleted_wpe(sd, drows)
            gates_surg[(tag, dl_name)] = gate
            if not gate["pass"]:
                raise RuntimeError(f"deletion gate FAILED {tag}/{dl_name}: {gate}")
            net_del = E131.evl_load(sd_del)
            cell = read183(net_del, pools[tag], name_ids, zid)
            cell["ce_r"] = E131.ce_fixed_cpu(net_del, *r_eval_xy)
            table[(tag, dl_name)] = cell
            del net_del
        n0 = table[(tag, "none")]["pz_onset_mean"]
        log(f"[{tag}] " + " | ".join(
            f"{dl} {table[(tag, dl)]['pz_onset_mean']:.4f}"
            f"({100 * (n0 - table[(tag, dl)]['pz_onset_mean']) / n0:+.0f}%)"
            for dl in {**DELS_REG, **DELS_CTL}) + " (pz@183, drop% vs none)")

    # =====================================================================
    # PROBE 3 — novel-context generalization at the 183 geometry
    # =====================================================================
    log("--- PROBE 3: novel-context generalization at 183 ---")
    # protected spans: all 90 install/held protocol occurrences (+/- margin),
    # the arm-b training filler spans (seed-12103 draws, reproduced exactly),
    # and the val r_eval spans (seed-26502 windows, starts recovered exactly).
    protected = [[max(0, p - PROTECT_MARGIN), p + len(h) + PROTECT_MARGIN]
                 for p, h in install_occ + held_occ]
    protected += [[int(s), int(s) + DREAM_LEN] for s in src.flatten()]
    r_eval_starts = []
    for x in r_eval_x:
        eq = (val_ids == x[0]).nonzero().flatten()
        hit = [int(i) for i in eq.tolist()
               if torch.equal(val_ids[i:i + BLOCK], x)]
        r_eval_starts.append(hit[0] if hit else -1)
    val_protected = [tuple([s, s + BLOCK]) for s in r_eval_starts if s >= 0]
    protected = sorted([tuple(sp) for sp in protected] + val_protected)
    merged = []
    for a, b in protected:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    protected = [tuple(sp) for sp in merged]

    nov_train, rep_tr = novel_windows(train_ids, train_text, protected,
                                      NOV_N, NOV_TRAIN_SEED)
    nov_val, rep_va = novel_windows(val_ids, val_text, val_protected,
                                    NOV_N, NOV_VAL_SEED)
    nov_pools = {"nov_train__train_fact": novel_splice(nov_train, fact_segs),
                 "nov_train__held_fact": novel_splice(nov_train, fact_segs_held),
                 "nov_val__train_fact": novel_splice(nov_val, fact_segs),
                 "nov_val__held_fact": novel_splice(nov_val, fact_segs_held)}
    G_NOV = {k: geo_ok(v) for k, v in nov_pools.items()}
    G_NOV["pass"] = all(G_NOV.values())
    if not G_NOV["pass"]:
        raise RuntimeError(f"novel pool geometry gate FAILED: {G_NOV}")
    log(f"novel pools: {NOV_N} windows each | train {rep_tr} | val {rep_va} | "
        f"geometry PASS")

    probe3 = {"construction": {"train_split": rep_tr, "val_split": rep_va,
                               "protected_spans_n": len(protected),
                               "fact_variants": {"train_fact": "install60 "
                                                "segments (trained content)",
                                                "held_fact": "held30 segments "
                                                "(never trained)"},
                               "leakage_gates": {
                                   "novel_windows_z_free": True,
                                   "novel_windows_outside_exposure_spans": True,
                                   "honesty_note": recipe_notes[5]}},
              "reads": {}}
    for tag in ("a_self_ctx_spliced", "b_corpus_ctx_spliced"):
        train_pz = table[(tag, "none")]["pz_onset_mean"]
        cells = {"training_pool": {"pz_onset_mean": train_pz}}
        for pname, pool in nov_pools.items():
            cells[pname] = read183(nets[tag], pool, name_ids, zid)
        base_cells = {pname: read183(net_b43, pool, name_ids, zid)
                      for pname, pool in nov_pools.items()}
        for pname in nov_pools:
            p = cells[pname]["pz_onset_mean"]
            fl = base_cells[pname]["pz_onset_mean"]
            gen = bool(p >= 0.5 and p >= 0.5 * train_pz)
            part = bool(p >= 2.0 * fl and not gen)
            cells[pname]["base_floor_pz"] = fl
            cells[pname]["verdict"] = ("GENERALIZES" if gen else
                                       "PARTIAL" if part else "FAILS")
        probe3["reads"][tag] = cells
        log(f"PROBE3[{tag}]: " + " | ".join(
            f"{pn.replace('nov_', '').replace('__', '/')} "
            f"{cells[pn]['pz_onset_mean']:.3f} "
            f"(floor {cells[pn]['base_floor_pz']:.3f}, "
            f"{cells[pn]['verdict']})" for pn in nov_pools))
    probe3["consolidated_reference"] = {
        pn: read183(nets["consolidated_e113"], pool, name_ids, zid)
        for pn, pool in list(nov_pools.items())[:2]}
    log(f"PROBE3 consolidated-net at 183 (report-only): "
        + " | ".join(f"{pn} {v['pz_onset_mean']:.3f}"
                     for pn, v in probe3["consolidated_reference"].items()))

    # =====================================================================
    # PROBE 4 — row-129 brake scope on the splice arms
    # =====================================================================
    log("--- PROBE 4: row-129 brake scope ---")
    probe4 = {}
    for tag in ("a_self_ctx_spliced", "b_corpus_ctx_spliced"):
        r129 = census[tag]["rows"].get("129", {"mean": float("nan"),
                                               "zero": float("nan")})
        d129_cell = table[(tag, "d_129")]
        n0 = table[(tag, "none")]["pz_onset_mean"]
        probe4[tag] = {"row129_mean_drop": r129["mean"],
                       "row129_zero_drop": r129["zero"],
                       "d129_pz_onset": d129_cell["pz_onset_mean"],
                       "d129_change_vs_none": d129_cell["pz_onset_mean"] - n0,
                       "consolidated_reference": {"mean": -0.1085,
                                                  "zero": -0.1324,
                                                  "note": "e131: replacement "
                                                          "RAISES battery p(Z) "
                                                          "= suppressive brake"},
                       "brake_present": bool(min(r129["mean"], r129["zero"]) < -0.02)}
        log(f"PROBE4[{tag}]: row129 replacement drops m/z "
            f"{r129['mean']:+.4f}/{r129['zero']:+.4f} | D-129 pz@183 "
            f"{d129_cell['pz_onset_mean']:.4f} ({d129_cell['pz_onset_mean'] - n0:+.4f} "
            f"vs none) -> brake "
            f"{'PRESENT' if probe4[tag]['brake_present'] else 'absent'}")

    # =====================================================================
    # RIDER — arm-c regeneration + read at the actual dream positions
    # =====================================================================
    log("--- RIDER: arm-c verbatim-dreams regeneration + dream-position read ---")
    rider: dict = {"mode": "smoke (skipped)" if SMOKE else "full"}
    net_c = None
    if not SMOKE:
        G_HARV = {"zephyra_count": z_in_dreams, "ref": HARVEST_REF_Z,
                  "pass": bool(z_in_dreams == HARVEST_REF_Z)}
        log(f"G_HARV: {z_in_dreams} ZEPHYRA in harvest (ref {HARVEST_REF_Z}): "
            f"{'PASS' if G_HARV['pass'] else 'DRIFT'}")
        rider["harvest_gate"] = G_HARV
        pool_c_x = dream_ids
        m_c = torch.zeros(pool_c_x.shape[0], BLOCK - 1, dtype=torch.bool)
        m_c[:, PRE - 1:] = True
        res_c = E131.finetune_arm("arm_c_rider", net_b43, pool_c_x, m_c,
                                  anchor, train_ids, r_eval_xy, f_eval, zid,
                                  CONS_SEED_SPLICE, 50)
        sd_c = res_c["sd"]
        net_c = E131.evl_load(sd_c)
        diffs, e120_bt = [], refs["e120"]["battery_table"]
        for j in E131.GEO_ORDER:
            k = f"c_verbatim_dreams__none__g{j:+d}__install60"
            mine = E131.battery_cell(net_c, bat_ids[(j, "install60")],
                                     zid)["mean_pz"]
            diffs.append(abs(mine - e120_bt[k]["mean_pz"]))
        md = max(diffs)
        G_ARMC = {"max_abs_diff": md, "tol": G_BIT_TOL,
                  "fallback_tol": G_FALLBACK_TOL,
                  "bit_reproducible": bool(md < G_BIT_TOL),
                  "passes_convention": bool(md < G_FALLBACK_TOL)}
        rider["armc_gate"] = G_ARMC
        log(f"G_ARMC: arm-c battery vs e120 stored table |d| {md:.2e} -> "
            f"{'BIT-REPRODUCIBLE' if G_ARMC['bit_reproducible'] else ('CONVENTION-PASS' if G_ARMC['passes_convention'] else 'DEVIATION')}")
        if not G_ARMC["passes_convention"]:
            recipe_notes.append(f"arm-c regeneration deviates from e120 by "
                                f"{md:.4f} — RERUN-PRIMARY reported")
        p_c = CKPT_DIR / "e139_arm_c_verbatim_dreams.pt"
        torch.save({"model": sd_c, "meta": {"recipe": "e120 arm (c) verbatim",
                                            "seed": CONS_SEED_SPLICE,
                                            "base": str(CK_B43)}}, p_c)
        log(f"ckpt saved: {p_c.name}")

        occs = find_name_occ(pool_c_x, itos)
        rider["n_dream_occurrences"] = len(occs)
        rider["onset_cols"] = [c for _, c in occs]
        hist: dict[int, int] = {}
        for _, c in occs:
            hist[c] = hist.get(c, 0) + 1
        rider["onset_col_histogram"] = {str(k): v for k, v in sorted(hist.items())}
        rider["onset_position_note"] = (
            "dream ZEPHYRA onsets concentrate at x-col 130 (the host-"
            "continuation slot: the 130-char prompts end exactly at host-name "
            "positions, so the net dreams its name as the FIRST continuation "
            "token) — read position c-1 = 129 = the OLD address row, i.e. "
            "the dream positions are NOT position-diverse")
        read_c = read_at_occurrences(net_c, pool_c_x, occs, zid)
        read_base = read_at_occurrences(net_b43, pool_c_x, occs, zid)
        gen = bool(read_c["pz_onset_mean"] >= 0.5 and
                   read_c["pz_onset_mean"] >= 2.0 * max(read_base["pz_onset_mean"], 1e-9))
        nul = bool(read_c["pz_onset_mean"] <= 2.0 * max(read_base["pz_onset_mean"], 1e-9))
        rider.update({"read_arm_c": read_c, "read_base_same_positions": read_base,
                      "verdict": ("RIDER-SIGNAL (arm c was instrument-blind "
                                  "— the dreams DID consolidate at their own "
                                  "positions)" if gen else
                                  "RIDER-NULL (no consolidation at dream "
                                  "positions either — genuine decay)" if nul
                                  else "RIDER-PARTIAL")})
        log(f"RIDER: {len(occs)} dream ZEPHYRA occurrences | arm-c p(Z)@onset "
            f"{read_c['pz_onset_mean']:.4f} (frac>=0.5 "
            f"{read_c['pz_onset_frac_ge_0.5']:.3f}, p(name7) "
            f"{read_c['pname_mean_over7']:.4f}) vs base-at-same-positions "
            f"{read_base['pz_onset_mean']:.4f} -> {rider['verdict']}")
        del net_c
    else:
        log("RIDER: skipped in smoke")

    # =====================================================================
    # ADJUDICATION (registered order; per-arm then combined)
    # =====================================================================
    adjudication = {"per_arm": {}, "bars": REGISTERED_PREDICTION}
    for tag in ("a_self_ctx_spliced", "b_corpus_ctx_spliced"):
        n0 = table[(tag, "none")]["pz_onset_mean"]
        pz = {dl: table[(tag, dl)]["pz_onset_mean"]
              for dl in {**DELS_REG, **DELS_CTL}}
        r0_drop = (n0 - pz["d_r0"]) / n0
        r1_drop = (n0 - pz["d_r1"]) / n0
        net_r0_drop = r0_drop - r1_drop                 # scaffold-matched
        d183_kill = (n0 - pz["d_183"]) / n0
        row0_pos = probe1[tag]["row0_content_positive_2x"]
        row0_null = probe1[tag]["row0_null"]
        r0_full = bool(net_r0_drop >= 0.50 or row0_pos)
        site_full = bool(d183_kill >= 0.80 and row0_null)
        partial_r0 = bool((0.10 <= net_r0_drop < 0.50) or
                          (probe1[tag]["row0_content_criterion"] and not row0_pos))
        partial_183 = bool(0.10 <= d183_kill < 0.80)
        if r0_full and not site_full:
            v = "ROW-0-UNIVERSAL"
        elif site_full and not r0_full:
            v = "SITE-LOCKED"
        elif r0_full and site_full:
            v = "HYBRID-STRONG"
        elif partial_r0 and partial_183:
            v = "HYBRID"
        else:
            v = "TEXTURE"
        adjudication["per_arm"][tag] = {
            "pz_onset_cells": pz, "none": n0,
            "r0_drop_raw": r0_drop, "r1_drop_control": r1_drop,
            "r0_drop_net_of_control": net_r0_drop,
            "d183_kill": d183_kill,
            "row0_content_positive_2x": row0_pos, "row0_null": row0_null,
            "row0_strength": probe1[tag]["row0"]["strength"],
            "control_max_strength": probe1[tag]["control_max_strength"],
            "r0_full": r0_full, "site_full": site_full,
            "partial_r0": partial_r0, "partial_183": partial_183,
            "arm_verdict": v}
        log(f"ADJ[{tag}]: r0 drop {r0_drop:+.1%} (net {net_r0_drop:+.1%}) | "
            f"D-183 kill {d183_kill:+.1%} | row0 content-2x {row0_pos} -> {v}")

    vs = {v["arm_verdict"] for v in adjudication["per_arm"].values()}
    verdict = next(iter(vs)) if len(vs) == 1 else "TEXTURE"
    adjudication["verdict"] = verdict
    adjudication["arms_agree"] = len(vs) == 1
    log("=" * 78)
    log(f"E139 VERDICT: {verdict}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e139_row0_universality",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("T077's registered follow-up (dispatch 07:05Z). "
                         "Docstring + bars verbatim; operationalizations "
                         "fixed before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is row 0 the UNIVERSAL readout key for any consolidated "
                     "fact (including a 183-site consolidation), or was "
                     "re-keying-to-0 a property of the jitter road alone?"),
        "nets": {"arms": f"runs/checkpoints/{CK_A.name}, {CK_B.name} "
                         "(loaded, gated 1e-6 vs e131 tables)",
                 "consolidated": f"runs/checkpoints/{CK_CONS.name} (gated)",
                 "base": f"runs/checkpoints/{CK_B43.name} (gated)",
                 "rider_arm_c": "regenerated (e120 recipe verbatim) -> "
                                "runs/checkpoints/e139_arm_c_verbatim_dreams.pt"},
        "gates": {"G_SPLICE": G_SPLICE, "G_B43": G_B43, "G_ARMS": G_ARMS,
                  "G_CONS": G_CONS, "G_GEO": G_GEO,
                  "G_NOV": G_NOV, "G_SURG": {f"{k[0]}/{k[1]}": v
                                             for k, v in gates_surg.items()},
                  "harvest_zephyra_count": z_in_dreams},
        "probe1_row0_content": {"census": census, "adjudication": probe1,
                                "consolidated_reference": cons_ref,
                                "rows_censused": list(rows_used)},
        "probe2_deletion_battery": {
            f"{tag}__{dl}": {"pz_onset_mean": cell["pz_onset_mean"],
                             "pz_onset_frac_ge_0.5": cell["pz_onset_frac_ge_0.5"],
                             "pname_mean_over7": cell["pname_mean_over7"],
                             "ce_r": cell["ce_r"],
                             "registered_cell": dl in DELS_REG}
            for (tag, dl), cell in table.items()},
        "probe3_novel_contexts": probe3,
        "probe4_row129_brake": probe4,
        "rider_arm_c": rider,
        "adjudication": adjudication,
        "recipe_notes": recipe_notes,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072, "device": "cpu",
                   "torch_threads": torch.get_num_threads(), "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "row0_universality.png", table, census, probe1, probe3, probe4,
         rider, adjudication, verdict)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'row0_universality.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def plot(path, table, census, probe1, probe3, probe4, rider, adjudication,
         verdict):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    arms = ["a_self_ctx_spliced", "b_corpus_ctx_spliced"]
    col = {"a_self_ctx_spliced": "crimson", "b_corpus_ctx_spliced": "darkorange"}

    # (0,0) MAIN: arms x deletions at the 183-geometry
    ax = axes[0, 0]
    cells = ["none", "d_183", "d_band", "d_band_183", "d_r0", "d_r0_183",
             "d_129", "d_r1", "d_r1_183"]
    lbl = {"none": "none", "d_183": "D-183", "d_band": "D-band",
           "d_band_183": "D-band+183", "d_r0": "D-row-0",
           "d_r0_183": "D-row-0+183", "d_129": "D-129 (ctl)",
           "d_r1": "D-row-1 (ctl)", "d_r1_183": "D-r1+183 (ctl)"}
    xs = np.arange(len(cells))
    for k, tag in enumerate(arms):
        vals = [table[(tag, dl)]["pz_onset_mean"] for dl in cells]
        reg = [dl in DELS_REG for dl in cells]
        ax.bar(xs + (k - 0.5) * 0.38,
               [v if r else 0 for v, r in zip(vals, reg)], 0.36,
               color=col[tag], edgecolor="k", lw=0.5,
               label=f"arm ({tag.split('_')[0]})" + (" registered" if k == 0 else ""))
        ax.bar(xs + (k - 0.5) * 0.38,
               [v if not r else 0 for v, r in zip(vals, reg)], 0.36,
               color=col[tag], alpha=0.35, edgecolor="k", lw=0.5, hatch="//")
        n0 = table[(tag, "none")]["pz_onset_mean"]
        for x, dl, v in zip(xs, cells, vals):
            if dl != "none":
                ax.text(x + (k - 0.5) * 0.38, v + 0.015,
                        f"{100 * (n0 - v) / n0:+.0f}%", ha="center", fontsize=5.6,
                        rotation=90)
    for k, tag in enumerate(arms):
        n0 = table[(tag, "none")]["pz_onset_mean"]
        ax.axhline(0.5 * n0, ls="--", lw=1.0, color=col[tag], alpha=0.7)
        ax.axhline(0.2 * n0, ls=":", lw=1.0, color=col[tag], alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl[c] for c in cells], fontsize=6.5, rotation=30)
    ax.set_ylabel("p(Z) at position 183 (fact onset)")
    ax.set_ylim(0, 1.08)
    ax.set_title("PROBE 2 MAIN: splice-arm fact expression @183 x deletion "
                 "(dashed = -50% bar, dotted = -80% bar; hatched = ctl)",
                 fontsize=9.5)
    ax.legend(fontsize=7, loc="upper right")

    # (0,1) row-0 content test census
    ax = axes[0, 1]
    rows_x = [0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120, 121, 125, 129,
              133, 137, 183]
    for k, tag in enumerate(arms):
        vals = [census[tag]["rows"][str(r)]["strength"] if str(r) in
                census[tag]["rows"] else 0.0 for r in rows_x]
        ax.bar(np.arange(len(rows_x)) + (k - 0.5) * 0.38, vals, 0.36,
               color=col[tag], edgecolor="k", lw=0.4,
               label=f"arm ({tag.split('_')[0]})")
    for k, tag in enumerate(arms):
        b = probe1[tag]["bar_2x_control"]
        ax.axhline(b, ls="--", lw=0.9, color=col[tag], alpha=0.6,
                   label=f"2x ctrl-max ({tag.split('_')[0]}) {b:.4f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(rows_x)))
    ax.set_xticklabels([("row0" if r == 0 else "r183" if r == 183 else str(r))
                        for r in rows_x], fontsize=6.5, rotation=45)
    ax.set_ylabel("strength = min(mean-drop, zero-drop) @183-readout")
    ax.set_title("PROBE 1: row-0 content test on splice arms "
                 "(negative drop = replacement RAISES p = brake)", fontsize=9.5)
    ax.legend(fontsize=6.5)

    # (1,0) novel-context generalization
    ax = axes[1, 0]
    pools_p = ["training_pool", "nov_train__train_fact",
               "nov_train__held_fact", "nov_val__train_fact",
               "nov_val__held_fact"]
    plbl = ["training pool", "novel train ctx\n+ trained fact",
            "novel train ctx\n+ held fact", "novel val ctx\n+ trained fact",
            "novel val ctx\n+ held fact"]
    xs3 = np.arange(len(pools_p))
    for k, tag in enumerate(arms):
        vals = [probe3["reads"][tag][p]["pz_onset_mean"] for p in pools_p]
        ax.bar(xs3 + (k - 0.5) * 0.38, vals, 0.36, color=col[tag],
               edgecolor="k", lw=0.4, label=f"arm ({tag.split('_')[0]})")
        for x, p in zip(xs3, pools_p):
            if p != "training_pool":
                fl = probe3["reads"][tag][p]["base_floor_pz"]
                ax.plot(x + (k - 0.5) * 0.38, fl, "k_", ms=7, mew=1.4)
        for x, p, v in zip(xs3, pools_p, vals):
            vd = probe3["reads"][tag][p].get("verdict", "")
            if vd:
                ax.text(x + (k - 0.5) * 0.38, v + 0.012, vd[:4], ha="center",
                        fontsize=5.6, rotation=90)
    ax.set_xticks(xs3)
    ax.set_xticklabels(plbl, fontsize=6.5)
    ax.set_ylabel("p(Z) at position 183")
    ax.set_ylim(0, 1.08)
    ax.set_title("PROBE 3: novel-context generalization at 183 "
                 "(black dash = base-net floor)", fontsize=9.5)
    ax.legend(fontsize=7, loc="upper right")

    # (1,1) rider + verdict
    ax = axes[1, 1]
    if rider.get("read_arm_c"):
        rc, rb = rider["read_arm_c"], rider["read_base_same_positions"]
        bars = [rb["pz_onset_mean"], rc["pz_onset_mean"],
                rb["pname_mean_over7"], rc["pname_mean_over7"]]
        ax.bar(np.arange(4), bars, 0.55,
               color=["tab:gray", "steelblue", "tab:gray", "steelblue"],
               edgecolor="k", lw=0.5)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["base p(Z)\n@dream onsets", "arm-c p(Z)\n@dream onsets",
                            "base p(name7)", "arm-c p(name7)"], fontsize=6.5)
        for i, v in enumerate(bars):
            ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_ylabel("p")
        ax.set_ylim(0, 1.15)
        ax.set_title(f"RIDER (arm c, {rider.get('n_dream_occurrences', '-')} dream "
                     f"ZEPHYRA positions): {rider['verdict']}", fontsize=8.5)
        txt_x, txt_y, fs = 0.52, 0.97, 7.0
    else:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        txt_x, txt_y, fs = 0.03, 0.97, 8.0
    pa = adjudication["per_arm"]
    lines = [f"VERDICT: {verdict}"
             + ("" if adjudication.get("arms_agree", True)
                else " (ARMS DISAGREE -> texture)")]
    for tag in pa:
        v = pa[tag]
        lines += [f"arm {tag.split('_')[0]}: D-r0 drop {v['r0_drop_raw']:+.0%} "
                  f"(net of r1 ctl {v['r0_drop_net_of_control']:+.0%}) | "
                  f"D-183 kill {v['d183_kill']:+.0%}",
                  f"   row0 content-2x {v['row0_content_positive_2x']} "
                  f"(strength {v.get('row0_strength', float('nan')):.3f} vs "
                  f"2x-ctrl {2 * v.get('control_max_strength', float('nan')):.3f})"
                  f" -> {v['arm_verdict']}"]
    for tag in probe4:
        lines.append(f"brake[{tag.split('_')[0]}]: row129 m/z "
                     f"{probe4[tag]['row129_mean_drop']:+.3f}/"
                     f"{probe4[tag]['row129_zero_drop']:+.3f} "
                     f"({'present' if probe4[tag]['brake_present'] else 'absent'})")
    ax.text(txt_x, txt_y, "\n".join(lines), transform=ax.transAxes,
            fontsize=fs, va="top", family="monospace",
            bbox=dict(facecolor="lightyellow", alpha=0.94, edgecolor="gray"))

    fig.suptitle(f"E139 — ROW-0 UNIVERSALITY + 183-ROBUSTNESS on the e131 "
                 f"splice arms -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
