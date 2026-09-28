"""E163 — THE SATURATION CONTROL (R44 ideator; REGISTERED; the last
submission blocker).

WHY: the intro's first sentence ("born with one memory organ — row 0
carries every install, 13/13 nets") sits on the TRAINED-GEOMETRY row-0
dial (e131 probe-2 / e142 census: rel = S_0/base in 0.878..1.000 across
the 13 gate-passing nets). T083 declared that dial SATURATING at trained
geometries — it measures sink-load, which every readout has, so in
principle it cannot detect NON-carriage. If so, the 13/13 is a truism
("every readout loads the sink") and the intro's sentence collapses.

THE LICENSING CELL: run the SAME dial on a net whose fact is KNOWN to be
only ~7% row-0-dependent — e131_arm_b_corpus_spliced.pt (e133's
site_locked: wpe_r0 share 0.0706 of attributable load; row-0 removal
leaves the site fact at survivor 0.725, i.e. 0.988 -> 0.716 onset p(Z)).
  * DIAL-BLIND: arm_b's dial ALSO reads ~1.0 there -> the dial cannot
    detect non-carriage; the 13/13 collapses to T083's truism; the
    intro's first sentence rewrites.
  * DIAL-VALID: arm_b's dial reads 0.7-0.8 -> the dial discriminates
    carriage; the intro stands as licensed.

INSTRUMENTS (reused VERBATIM with provenance):
  * THE DIAL — e142 row_census / e131 probe-2, arm mechanics VERBATIM:
    mean arm wpe[r] <- mean(all rows); zero arm wpe[r] <- 0;
    drop = base - arm; strength(r) = min(mean-drop, zero-drop);
    e116 content criterion (mean>0 AND zero>0 AND min/max >= 0.5);
    rel = S/base reported everywhere (T083: trained-geometry absolutes
    are sink-load quantities).
  * BATTERY per net (each net's OWN instrument context, e142's
    convention): the CARRIER (e131_consolidated_e113.pt) on the e131
    probe-2 battery — install-60 g0 130-token contexts, mean p(Z) at
    the last position (e131 battery_pz); the NON-CARRIER (arm_b) on the
    e133 site battery — the e120 arm-(b) splice pool VERBATIM (30
    held-prompts x 4 corpus fillers, seed 12103 = 120 x 256-token
    windows, ZEPHYRA at x-col 184 / address row 183), read = onset
    p(Z) at position 183 (e133 battery_site). This is the only battery
    where arm_b's fact is expressed (std install-60 g0 = 0.0078) and it
    is exactly the battery of the 7.06%/0.725 ground truth.
  * GATES: carrier vs e131's CPU-stored probe-2 cells (base, row-0
    mean/zero, row-1, row-129; tol 5e-6, e113 fallback 0.05); arm_b vs
    e131's CPU-stored probe-1 cells (std g0 0.007800613064318895; site
    onset 0.9880021214485168) and e133's stored zero-arm cell
    (site onset 0.7161287069320679, GPU-evaluated -> tol 1e-3).

REGISTERED PREDICTION (QUEUE e163 / dispatch, VERBATIM — no shopping):
  * DIAL-BLIND fires if: arm_b's dial reads ~1.0 (>= 0.9) — the dial
    cannot detect non-carriage; the 13/13 collapses to "every readout
    loads the sink" (T083's truism); the intro's first sentence
    rewrites.
  * DIAL-VALID fires if: arm_b's dial reads 0.7-0.8 — the dial
    discriminates carriage; the intro stands as licensed.
  * No bar shopping; texture => TEXTURE with numbers.

OPERATIONALIZATIONS (fixed before compute — the two bar figures live on
the dial's two faces, both reported for every cell):
  * The dial's census headline (the quantity behind "13/13" and "~1.0")
    is rel = S_0/base, the DROP fraction; the e142 census band is
    0.878..1.000. DIAL-BLIND = rel_S0(arm_b) >= 0.9 (arm_b reads like
    a carrier despite 7% ground truth).
  * The registered 0.7-0.8 DIAL-VALID figure is pinned by the same
    QUEUE row's ground-truth parenthetical "(survives row-0 removal at
    0.725)": it is the dial's SURVIVOR — the fraction of base the fact
    still reads after the row-0 arm. DIAL-VALID = 1 - rel_S0(arm_b) in
    [0.7, 0.8] (equivalently rel_S0 in [0.2, 0.3]), i.e. the dial
    reproduces the known 7%-dependent profile instead of saturating.
  * Reading "0.7-0.8" as rel in [0.7, 0.8] instead would be a THIRD
    outcome; it is neither registered face and fires nothing — any
    outcome other than the two registered bars => TEXTURE with numbers.
  * Carrier control (context, not a bar): the dial should read ~1.0 on
    e131_consolidated_e113 (e131 stored cells -> rel 0.932; census band
    0.878..1.000).

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch
import); LOW threads (4); eval-only, minutes; sequential net loads; no
busy-waiting.

Outputs: runs/e163/{metrics.json, saturation_control.png}.
No NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e163_saturation_control.py     (E163_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"         # CPU-ONLY (constraint)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(4)                              # LOW threads (<=4)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = _os.environ.get("E163_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
POST_CAP = 119                                         # e043 deviation-1
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

# splice geometry + site battery (e120 arm-(b) / e133 verbatim)
FACT_PRE, FACT_POST = 12, 12
FACT_LEN = FACT_PRE + len(NAME) + FACT_POST           # 31
SPLICE_AT = 42
Z_XCOL = PRE + SPLICE_AT + FACT_PRE                   # 184
SPLICE_ADDR_ROW = Z_XCOL - 1                          # 183
CORP_CONT_SEED = 12103
N_PROMPTS = 8 if SMOKE else 30
N_FILL = 1 if SMOKE else 4

CONTROL_ROWS_256 = (1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120)   # e131 verbatim
BAND = tuple(range(121, 138))                         # report/context rows

G_BIT_TOL = 5e-6                                       # CPU-stored refs
G_FALLBACK_TOL = 0.05                                  # e113 convention
G_GPU_TOL = 1e-3                                       # GPU-evaluated stored refs (e142 convention)

REGISTERED_PREDICTION = {
    "dial_blind": "DIAL-BLIND fires if: arm_b's dial reads ~1.0 "
                  "(>= 0.9) — the dial cannot detect non-carriage; the "
                  "13/13 collapses to 'every readout loads the sink' "
                  "(T083's truism); the intro's first sentence rewrites.",
    "dial_valid": "DIAL-VALID fires if: arm_b's dial reads 0.7-0.8 — the "
                  "dial discriminates carriage; the intro stands as "
                  "licensed.",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with numbers.",
    "operationalizations": "dial headline = rel = S_0/base (drop fraction; "
                           "e142 census band 0.878..1.000): DIAL-BLIND = "
                           "rel_S0(arm_b) >= 0.9. The 0.7-0.8 DIAL-VALID "
                           "figure is the dial's SURVIVOR (1 - rel_S0), "
                           "pinned by the QUEUE row's own parenthetical "
                           "'(survives row-0 removal at 0.725)': DIAL-VALID "
                           "= 1 - rel_S0(arm_b) in [0.7, 0.8] (rel in "
                           "[0.2, 0.3]). Both faces reported for every "
                           "cell; any other outcome => TEXTURE with "
                           "numbers; no post-hoc bar movement.",
}

# ground truth / gate refs (stored; embedded as fallback if metrics move)
GT = {
    "arm_b": {
        "ckpt": "runs/checkpoints/e131_arm_b_corpus_spliced.pt",
        "e133_tag": "site_locked",
        "std_g0_ref_e131_cpu": 0.007800613064318895,
        "site_onset_ref_e131_cpu": 0.9880021214485168,
        "site_onset_e133_run": 0.9880020022392273,
        "wpe_r0_zero_arm_site_onset_e133": 0.7161287069320679,
        "wpe_r0_share_e133": 0.07063399829336497,
        "survivor_known": 0.7161287069320679 / 0.9880020022392273,
    },
    "carrier": {
        "ckpt": "runs/checkpoints/e131_consolidated_e113.pt",
        "e133_tag": "graduated",
        "probe2_base_e131_cpu": 0.7850372118875384,
        "probe2_row0_mean_e131_cpu": 0.7842019017236945,
        "probe2_row0_zero_e131_cpu": 0.7316772222270098,
        "probe2_row1_mean_e131_cpu": -0.034818108193576336,
        "probe2_row1_zero_e131_cpu": -0.02616492702315254,
        "probe2_row129_mean_e131_cpu": -0.1084650677318375,
        "probe2_row129_zero_e131_cpu": -0.13237020391970877,
        "rel_known": 0.7316772222270098 / 0.7850372118875384,
    },
}


# ------------------------------------------------------------------ instruments

def load_cpu(path: Path, cfg: Cfg) -> TinyGPT:
    m = TinyGPT(cfg)
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_pz(net: TinyGPT, ids: torch.Tensor, cid: int, bs=30) -> float:
    """e131/e142 battery_pz VERBATIM: mean p(char) at the last position."""
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, cid]) for k in range(pr.shape[0])]
    return float(np.mean(pzs))


@torch.no_grad()
def battery_site_onset(net: TinyGPT, pool_x: torch.Tensor, name_ids,
                       zid: int, bs=30) -> float:
    """e133 battery_site VERBATIM (the onset scalar): p(Z) at position 183
    (the address row) over the splice windows — arm_b's own battery read."""
    net.eval()
    onset = []
    for i in range(0, pool_x.shape[0], bs):
        w = pool_x[i:i + bs]
        lg, _ = net(w)
        pr = F.softmax(lg, -1)
        for k in range(w.shape[0]):
            onset.append(float(pr[k, SPLICE_ADDR_ROW, int(zid)]))
    return float(np.mean(onset))


@torch.no_grad()
def row_census(net: TinyGPT, read, rows: list[int]) -> dict:
    """e142 row_census VERBATIM (arm mechanics + conventions), generalized
    only over the battery read fn (each net's own instrument context):
    mean arm wpe[r]<-mean(all rows); zero arm wpe[r]<-0;
    strength = min(mean-drop, zero-drop); e116 content criterion."""
    base = read(net)
    w = net.wpe.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    out: dict[int, dict] = {}
    for r in rows:
        w.copy_(orig); w[r] = mean_row
        m_d = base - read(net)
        w.copy_(orig); w[r] = 0.0
        z_d = base - read(net)
        hi = max(m_d, z_d)
        out[int(r)] = {
            "mean": float(m_d), "zero": float(z_d),
            "ratio": float(min(m_d, z_d) / hi) if hi > 0 else 0.0,
            "strength": float(min(m_d, z_d)),
            "content": bool(m_d > 0 and z_d > 0 and
                            (min(m_d, z_d) / hi) >= 0.5),
        }
    w.copy_(orig)
    return {"base": base, "rows": out}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e163_smoke" if SMOKE else "e163")
    cfg = Cfg()                                        # 6L/6H/192d/256 (2.7M)
    log(f"E163 THE SATURATION CONTROL (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda visible = {torch.cuda.is_available()}")

    refs = {}
    for key, path in (("e131", E43.REPO / "runs" / "e131" / "metrics.json"),
                      ("e133", E43.REPO / "runs" / "e133" / "metrics.json"),
                      ("e142", E43.REPO / "runs" / "e142" / "metrics.json")):
        refs[key] = json.loads(Path(path).read_text(encoding="utf-8")) \
            if Path(path).exists() else None

    # ---------------- protocol rebuild (e131/e133 verbatim)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids = corpus.train
    train_text = "".join(itos[int(i)] for i in train_ids)

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30")

    name_ids = corpus.encode(NAME)

    # carrier battery: install-60 g0 130-token contexts (e131 probe-2)
    bat_install = torch.stack([corpus.encode(train_text[p - PRE: p])
                               for p, _ in install_occ])

    # arm_b battery: e133 site pool (e120 arm-(b) construction VERBATIM)
    prompts = [train_text[p - PRE: p] for p, _ in held_occ][:N_PROMPTS]
    prompt_ids = torch.stack([corpus.encode(c) for c in prompts])
    gc_ = torch.Generator().manual_seed(CORP_CONT_SEED)
    src = torch.randint(len(train_ids) - (BLOCK - PRE) - 1,
                        (N_FILL, len(prompts)), generator=gc_)
    filler = torch.stack([train_ids[s: s + BLOCK - PRE]
                          for s in src.flatten()])
    segs = []
    for p, h in install_occ:
        s = (train_text[p - FACT_PRE: p] + NAME
             + train_text[p + len(h): p + len(h) + FACT_POST])
        assert len(s) == FACT_LEN
        segs.append(corpus.encode(s))
    fact_segs = torch.stack(segs)
    fs = fact_segs[torch.arange(filler.shape[0]) % fact_segs.shape[0]]
    cont = torch.cat([filler[:, :SPLICE_AT], fs,
                      filler[:, SPLICE_AT + FACT_LEN:]], 1)
    pool_b = torch.cat([torch.stack(
        [prompt_ids[k % prompt_ids.shape[0]] for k in range(filler.shape[0])]),
        cont], 1)
    assert all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)], name_ids)
               for w in pool_b), "site battery geometry broken"
    log(f"site battery rebuilt: {tuple(pool_b.shape)}, ZEPHYRA at x-col "
        f"{Z_XCOL} (address row {SPLICE_ADDR_ROW}) — e133 ref 120x256")

    # ---------------- the two dials
    results, gates = {}, {}

    # ---- CARRIER CONTROL (e131_consolidated_e113): std install battery
    log("--- CARRIER: e131_consolidated_e113 on its own install battery ---")
    net_c = load_cpu(CKPT_DIR / "e131_consolidated_e113.pt", cfg)
    read_c = lambda n: battery_pz(n, bat_install, zid)
    rows_c = sorted({0, 129} | set(CONTROL_ROWS_256) | set(BAND))
    cen_c = row_census(net_c, read_c, rows_c)

    g = {"kind": "e131_probe2_cells", "tol": G_BIT_TOL,
         "fallback_tol": G_FALLBACK_TOL}
    if refs["e131"] is not None and not SMOKE:
        tab = refs["e131"]["probe2_census_tables"]["consolidated"]
        checks = {
            "base": (cen_c["base"], tab["base_pz"]),
            "row0_mean": (cen_c["rows"][0]["mean"], tab["rows"]["0"]["mean"]),
            "row0_zero": (cen_c["rows"][0]["zero"], tab["rows"]["0"]["zero"]),
            "row1_mean": (cen_c["rows"][1]["mean"], tab["rows"]["1"]["mean"]),
            "row1_zero": (cen_c["rows"][1]["zero"], tab["rows"]["1"]["zero"]),
            "row129_mean": (cen_c["rows"][129]["mean"], tab["rows"]["129"]["mean"]),
            "row129_zero": (cen_c["rows"][129]["zero"], tab["rows"]["129"]["zero"]),
        }
        g["cells"] = {k: {"mine": v[0], "ref": v[1], "diff": abs(v[0] - v[1])}
                      for k, v in checks.items()}
        g["max_diff"] = max(c["diff"] for c in g["cells"].values())
        g["bit_reproducible"] = bool(g["max_diff"] < G_BIT_TOL)
        g["passes_e113_convention"] = bool(g["max_diff"] < G_FALLBACK_TOL)
        g["base_ok"] = g["passes_e113_convention"]
    else:
        g["note"] = "e131 metrics unavailable or smoke — embedded GT refs used"
        g["cells"] = {
            "base": {"mine": cen_c["base"], "ref": GT["carrier"]["probe2_base_e131_cpu"],
                     "diff": abs(cen_c["base"] - GT["carrier"]["probe2_base_e131_cpu"])},
            "row0_mean": {"mine": cen_c["rows"][0]["mean"],
                          "ref": GT["carrier"]["probe2_row0_mean_e131_cpu"],
                          "diff": abs(cen_c["rows"][0]["mean"] - GT["carrier"]["probe2_row0_mean_e131_cpu"])},
            "row0_zero": {"mine": cen_c["rows"][0]["zero"],
                          "ref": GT["carrier"]["probe2_row0_zero_e131_cpu"],
                          "diff": abs(cen_c["rows"][0]["zero"] - GT["carrier"]["probe2_row0_zero_e131_cpu"])},
        }
        g["max_diff"] = max(c["diff"] for c in g["cells"].values())
        g["bit_reproducible"] = bool(g["max_diff"] < G_BIT_TOL)
        g["passes_e113_convention"] = bool(g["max_diff"] < G_FALLBACK_TOL)
        g["base_ok"] = g["passes_e113_convention"]
    gates["carrier"] = g
    log(f"GATE carrier vs e131 stored cells: max diff {g['max_diff']:.3e} -> "
        f"{'BIT' if g['bit_reproducible'] else ('CONVENTION' if g['base_ok'] else 'FAIL')}")
    if not g["base_ok"] and not SMOKE:
        raise RuntimeError("carrier failed its e131 probe-2 gate")

    r0c = cen_c["rows"][0]
    # off-geometry context: the carrier read through arm_b's site battery
    off_c = row_census(net_c, lambda n: battery_site_onset(n, pool_b, name_ids, zid),
                       [0])
    carrier = {
        "ckpt": "runs/checkpoints/e131_consolidated_e113.pt",
        "role": "carrier control (e133 graduated; the dial should read ~1.0)",
        "battery": "install-60 g0 130-token (e131 probe-2 verbatim)",
        "battery_n": int(bat_install.shape[0]), "char": "Z",
        "base": cen_c["base"],
        "row0": r0c,
        "S0": r0c["strength"], "S0_content": r0c["content"],
        "rel_S0": float(r0c["strength"] / cen_c["base"]),
        "survivor": float(1.0 - r0c["strength"] / cen_c["base"]),
        "decision_row_129": cen_c["rows"][129],
        "controls": {"rows": {str(r): cen_c["rows"][r] for r in CONTROL_ROWS_256},
                     "max_strength": max(cen_c["rows"][r]["strength"]
                                         for r in CONTROL_ROWS_256)},
        "census_rows": {str(r): cen_c["rows"][r] for r in rows_c},
        "off_geometry_site_row0": off_c["rows"][0],
        "off_geometry_site_base": off_c["base"],
        "gate": g,
    }
    results["carrier"] = carrier
    log(f"  carrier dial: base {cen_c['base']:.4f} | S0 {r0c['strength']:.4f} "
        f"| rel {carrier['rel_S0']:.4f} | survivor {carrier['survivor']:.4f} "
        f"(e131 stored rel {GT['carrier']['rel_known']:.4f})")
    del net_c
    gc.collect()

    # ---- NON-CARRIER (e131_arm_b_corpus_spliced): its OWN site battery
    log("--- NON-CARRIER: e131_arm_b_corpus_spliced on its own site battery ---")
    net_b = load_cpu(CKPT_DIR / "e131_arm_b_corpus_spliced.pt", cfg)
    read_b = lambda n: battery_site_onset(n, pool_b, name_ids, zid)
    rows_b = sorted({0, 183} | set(CONTROL_ROWS_256) | set(BAND))
    cen_b = row_census(net_b, read_b, rows_b)

    gb = {"kind": "e131_probe1_plus_e133_cells", "tol": G_BIT_TOL,
          "gpu_tol": G_GPU_TOL}
    std_g0 = battery_pz(net_b, bat_install, zid)
    gb["std_g0"] = {"mine": std_g0,
                    "ref": GT["arm_b"]["std_g0_ref_e131_cpu"],
                    "diff": abs(std_g0 - GT["arm_b"]["std_g0_ref_e131_cpu"]),
                    "tol": G_BIT_TOL}
    gb["std_g0"]["ok"] = bool(gb["std_g0"]["diff"] < G_BIT_TOL)
    gb["site_onset_base"] = {"mine": cen_b["base"],
                             "ref": GT["arm_b"]["site_onset_ref_e131_cpu"],
                             "diff": abs(cen_b["base"] - GT["arm_b"]["site_onset_ref_e131_cpu"]),
                             "tol": G_BIT_TOL}
    gb["site_onset_base"]["ok"] = bool(gb["site_onset_base"]["diff"] < G_BIT_TOL)
    # the ground-truth cross-check: e133's stored zero-arm cell (GPU-evaluated).
    # e133's stored 0.7161... is the ARM VALUE (onset after zeroing row 0),
    # so compare base - zero_drop against it, not the drop itself.
    zero_arm_pz = float(cen_b["base"] - cen_b["rows"][0]["zero"])
    gb["row0_zero_vs_e133"] = {
        "mine_arm_pz": zero_arm_pz,
        "ref_arm_pz": GT["arm_b"]["wpe_r0_zero_arm_site_onset_e133"],
        "mine_zero_drop": float(cen_b["rows"][0]["zero"]),
        "tol": G_GPU_TOL,
        "ok": bool(abs(zero_arm_pz
                       - GT["arm_b"]["wpe_r0_zero_arm_site_onset_e133"])
                   < G_GPU_TOL)}
    gb["base_ok"] = bool(gb["std_g0"]["ok"] and gb["site_onset_base"]["ok"])
    gb["ground_truth_ok"] = gb["row0_zero_vs_e133"]["ok"]
    gates["arm_b"] = gb
    log(f"GATE arm_b: std_g0 diff {gb['std_g0']['diff']:.3e} | site base diff "
        f"{gb['site_onset_base']['diff']:.3e} | zero-arm pz vs e133 diff "
        f"{abs(zero_arm_pz - GT['arm_b']['wpe_r0_zero_arm_site_onset_e133']):.3e} -> "
        f"{'PASS' if gb['base_ok'] and gb['ground_truth_ok'] else 'FAIL'}")
    if not (gb["base_ok"] and gb["ground_truth_ok"]) and not SMOKE:
        raise RuntimeError("arm_b failed its e131/e133 gates")

    r0b = cen_b["rows"][0]
    # off-geometry context: the std-battery dial (the fact is absent there)
    off_b = row_census(net_b, lambda n: battery_pz(n, bat_install, zid), [0])
    arm_b = {
        "ckpt": "runs/checkpoints/e131_arm_b_corpus_spliced.pt",
        "role": "non-carrier (e133 site_locked: 7.06% wpe_r0 share; "
                "row-0-removal survivor 0.725)",
        "battery": "e133 site battery (e120 arm-(b) splice pool, onset p(Z)@183)",
        "battery_n": int(pool_b.shape[0]), "char": "Z",
        "base": cen_b["base"],
        "row0": r0b,
        "S0": r0b["strength"], "S0_content": r0b["content"],
        "rel_S0": float(r0b["strength"] / cen_b["base"]),
        "survivor": float(1.0 - r0b["strength"] / cen_b["base"]),
        "site_row_183": cen_b["rows"][183],
        "controls": {"rows": {str(r): cen_b["rows"][r] for r in CONTROL_ROWS_256},
                     "max_strength": max(cen_b["rows"][r]["strength"]
                                         for r in CONTROL_ROWS_256)},
        "census_rows": {str(r): cen_b["rows"][r] for r in rows_b},
        "off_geometry_std_row0": off_b["rows"][0],
        "off_geometry_std_base": off_b["base"],
        "gate": gb,
    }
    results["arm_b"] = arm_b
    log(f"  arm_b dial: base {cen_b['base']:.4f} | mean-arm {r0b['mean']:.4f} "
        f"| zero-arm {r0b['zero']:.4f} | S0 {r0b['strength']:.4f} "
        f"| rel {arm_b['rel_S0']:.4f} | SURVIVOR {arm_b['survivor']:.4f} "
        f"(known survivor {GT['arm_b']['survivor_known']:.4f})")
    del net_b
    gc.collect()

    # ---------------- 13/13 census context (from e142's stored metrics)
    # NOTE: e142's own adjudication ("ROW-0-ALWAYS, 13/13") runs over ALL 13
    # censused nets — s4307's gate failed but the net was kept per its
    # registered first-trajectory deviation note. The band here reproduces
    # e142's adjudicated set (no re-gating), with per-net flags recorded.
    census_ctx = {"source": "runs/e142/metrics.json (e142_row0_at_birth)",
                  "verdict": None, "nets": [], "band": None}
    if refs["e142"] is not None:
        census_ctx["verdict"] = refs["e142"]["adjudication"]["verdict"]
        for n in refs["e142"]["nets"]:
            census_ctx["nets"].append(
                {"file": n["file"], "rel_S0": n["rel_S0"], "S0": n["S0"],
                 "base": n["base_pz"],
                 "gate_ok": bool(n["gate"].get(
                     "all_ok", n["gate"].get("base_ok", True)))})
        census_ctx["band"] = {
            "n_nets": len(census_ctx["nets"]),
            "note": "e142's adjudicated 13-net set verbatim "
                    "(s4307 gate-flagged, kept per its deviation note)",
            "rel_S0_min": min(n["rel_S0"] for n in census_ctx["nets"]),
            "rel_S0_max": max(n["rel_S0"] for n in census_ctx["nets"]),
            "gate_flagged": [n["file"] for n in census_ctx["nets"]
                             if not n["gate_ok"]]}
        log(f"13/13 census context: {len(census_ctx['nets'])} censused nets "
            f"(e142 adjudication), rel_S0 band "
            f"[{census_ctx['band']['rel_S0_min']:.4f}, "
            f"{census_ctx['band']['rel_S0_max']:.4f}]")

    # ---------------- adjudication (registered; no bar shopping)
    rel_b = arm_b["rel_S0"]
    surv_b = arm_b["survivor"]
    dial_blind = bool(rel_b >= 0.9)
    dial_valid = bool(0.7 <= surv_b <= 0.8)
    carrier_in_band = None
    if census_ctx["band"] is not None:
        b = census_ctx["band"]
        carrier_in_band = bool(b["rel_S0_min"] <= carrier["rel_S0"]
                               <= b["rel_S0_max"])

    if dial_blind:
        fired = "DIAL-BLIND"
        intro = ("the dial cannot detect non-carriage; the 13/13 collapses "
                 "to T083's truism ('every readout loads the sink'); the "
                 "intro's first sentence rewrites.")
    elif dial_valid:
        fired = "DIAL-VALID"
        intro = ("the dial discriminates carriage (arm_b reads far below "
                 "the census band and reproduces its known 7%-dependent "
                 "profile); the intro stands as licensed.")
    else:
        fired = "TEXTURE"
        intro = (f"neither registered bar: arm_b rel_S0 {rel_b:.4f} "
                 f"(survivor {surv_b:.4f}) — numbers above; no bar movement.")

    adjudication = {
        "dial_blind": {"fires": dial_blind,
                       "bar": "rel_S0(arm_b) >= 0.9",
                       "rel_S0_arm_b": rel_b},
        "dial_valid": {"fires": dial_valid,
                       "bar": "1 - rel_S0(arm_b) in [0.7, 0.8] "
                              "(the dial's survivor; QUEUE's '0.7-8')",
                       "survivor_arm_b": surv_b},
        "verdict": fired,
        "what_happens_to_the_intro": intro,
        "carrier_control": {"rel_S0": carrier["rel_S0"],
                            "e131_stored_rel": GT["carrier"]["rel_known"],
                            "within_census_band": carrier_in_band},
        "ground_truth": {"wpe_r0_share_e133": GT["arm_b"]["wpe_r0_share_e133"],
                         "survivor_known": GT["arm_b"]["survivor_known"],
                         "zero_arm_reproduced": gb["row0_zero_vs_e133"]["ok"]},
    }
    log("=" * 78)
    log(f"E163 VERDICT: {fired}")
    log(f"  arm_b dial:   rel_S0 {rel_b:.4f} | survivor {surv_b:.4f} "
        f"(ground truth survivor {GT['arm_b']['survivor_known']:.4f}, "
        f"share {GT['arm_b']['wpe_r0_share_e133']:.3f})")
    log(f"  carrier dial: rel_S0 {carrier['rel_S0']:.4f} "
        f"(e131 stored {GT['carrier']['rel_known']:.4f}; census band "
        + (f"[{census_ctx['band']['rel_S0_min']:.3f}, {census_ctx['band']['rel_S0_max']:.3f}]"
           if census_ctx['band'] else "n/a") + ")")
    log(f"  DIAL-BLIND (>=0.9): {dial_blind} | DIAL-VALID (survivor 0.7-0.8): {dial_valid}")
    log(f"  intro: {intro}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e163_saturation_control",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("R44 ideator (never run); QUEUE e163; coordinator "
                         "dispatch. Docstring + bars written before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does the trained-geometry row-0 dial (the instrument "
                     "behind the intro's 'row 0 carries every install, "
                     "13/13') detect NON-carriage, or does it saturate for "
                     "every readout (T083)?"),
        "instruments": {
            "dial": "e142 row_census / e131 probe-2 VERBATIM (mean arm "
                    "wpe[r]<-mean(all rows); zero arm wpe[r]<-0; strength = "
                    "min(mean-drop, zero-drop); e116 content criterion)",
            "battery": "each net's OWN instrument context (e142 convention): "
                       "carrier on install-60 g0 130-token (e131 probe-2); "
                       "arm_b on the e133 site battery (e120 arm-(b) splice "
                       "pool, onset p(Z)@183 — the battery of the 7.06%/0.725 "
                       "ground truth)",
            "rel": "S/base reported everywhere (T083: trained-geometry "
                   "absolutes measure sink-load); the registered 0.7-0.8 bar "
                   "is the survivor 1 - rel, pinned by the QUEUE row's own "
                   "'survives row-0 removal at 0.725' parenthetical",
        },
        "gates": gates,
        "nets": results,
        "census_context_13": census_ctx,
        "adjudication": adjudication,
        "recipe_deviations": [
            "arm_b's own battery is its SITE battery (the only one where "
            "its fact is expressed: std install-60 g0 = 0.0078); the dial's "
            "arm mechanics are verbatim e142/e131 — only the battery read "
            "differs, per e142's own 'each net's OWN install battery' "
            "convention.",
            "The two registered bar figures live on the dial's two faces "
            "(drop rel vs survivor); both are operationalized in the "
            "docstring BEFORE compute and both are reported for every cell. "
            "Reading '0.7-0.8' as rel in [0.7,0.8] would be a third, "
            "unregistered outcome and fires nothing.",
            "The carrier's site-battery read and arm_b's std-battery read "
            "are recorded as off-geometry context (not adjudicated).",
        ],
        "honesty_reflex": (
            "arm_b's ground truth is itself a member of the dial family: the "
            "7.06% share and the 0.725 survivor come from e133's zero-arm on "
            "this very battery, so the zero-arm here is a REPRODUCTION "
            "(gated at 1e-3), not new evidence; the independent content is "
            "the mean-arm, the min-convention S_0, and the carrier control's "
            "reproduction of e131's CPU-stored cells. Battery asymmetry: the "
            "carrier is read at the last position of 130-token install "
            "contexts, arm_b at position 183 of 256-token splice windows — "
            "each is the net's own instrument context (the dimensionless "
            "rel=S/base is within-net), and arm_b's std-battery off-geometry "
            "read shows the dial has nothing to read where the fact is "
            "absent (base 0.0078). Survivor at 0.7-0.8 still attributes up "
            "to a quarter of base expression to row-0 sink load in a "
            "7%-share net — the dial's absolute floor is sink-load, exactly "
            "as T083 said; the licensing question is only whether its scale "
            "SEPARATES carriage levels, which is what DIAL-VALID vs "
            "DIAL-BLIND adjudicates."),
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"threads": torch.get_num_threads(), "device": "cpu",
                   "smoke": SMOKE,
                   "cfg": "6L/6H/192d/256 (2.7M)"},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    plot(rd / "saturation_control.png", results, census_ctx, adjudication)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'saturation_control.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def plot(path, results, census_ctx, adj):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    carrier, arm_b = results["carrier"], results["arm_b"]

    # (0,0) the headline: rel_S0 across the census + the two dials
    ax = axes[0, 0]
    cens = list(census_ctx["nets"]) if census_ctx["nets"] else []
    xs = np.arange(len(cens) + 2)
    vals = [n["rel_S0"] for n in cens] + [carrier["rel_S0"], arm_b["rel_S0"]]
    cols = ["lightgray"] * len(cens) + ["steelblue", "crimson"]
    ax.bar(xs, vals, 0.72, color=cols, edgecolor="k", lw=0.5)
    ax.axhline(0.9, color="crimson", ls="--", lw=1.4,
               label="DIAL-BLIND bar (rel >= 0.9)")
    ax.axhspan(0.2, 0.3, color="seagreen", alpha=0.15,
               label="DIAL-VALID rel-window (survivor 0.7-0.8)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=6.0)
    labels = [n["file"].replace(".pt", "").replace("e098_install_", "s")
              .replace("e048_", "").replace("e044_", "").replace("_install", "")
              for n in cens] + ["CARRIER\ne131_consol", "NON-CARRIER\narm_b"]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5.6, rotation=45, ha="right")
    ax.set_ylabel("dial reading: rel = S_0 / base")
    ax.set_ylim(0, 1.12)
    ax.set_title("THE SATURATION CONTROL: the row-0 dial on a known "
                 "non-carrier (gray = e142's 13/13 census)", fontsize=9.5)
    ax.legend(fontsize=6.5, loc="lower right")

    # (0,1) both faces of the dial for the two nets
    ax = axes[0, 1]
    xs2 = np.arange(2)
    rels = [carrier["rel_S0"], arm_b["rel_S0"]]
    surv = [carrier["survivor"], arm_b["survivor"]]
    ax.bar(xs2 - 0.19, rels, 0.34, color=["steelblue", "crimson"],
           edgecolor="k", lw=0.5, label="drop face: rel = S_0/base")
    ax.bar(xs2 + 0.19, surv, 0.34, color=["lightskyblue", "mistyrose"],
           edgecolor="k", lw=0.5, label="survivor face: 1 - rel")
    ax.axhline(0.9, color="crimson", ls="--", lw=1.2)
    ax.axhspan(0.7, 0.8, color="seagreen", alpha=0.15,
               label="registered VALID window (survivor)")
    for i in range(2):
        ax.text(i - 0.19, rels[i] + 0.02, f"{rels[i]:.3f}", ha="center",
                fontsize=8)
        ax.text(i + 0.19, surv[i] + 0.02, f"{surv[i]:.3f}", ha="center",
                fontsize=8)
    ax.set_xticks(xs2)
    ax.set_xticklabels(["CARRIER (e131_consolidated_e113)\n"
                        "known rel 0.932 (e131 stored)",
                        "NON-CARRIER (arm_b)\nknown survivor 0.725 (e133)"],
                       fontsize=7.5)
    ax.set_ylim(0, 1.12)
    ax.set_title("both faces of the dial vs the known ground truths",
                 fontsize=10)
    ax.legend(fontsize=6.5, loc="center right")

    # (1,0) arm_b's row census (its own battery)
    ax = axes[1, 0]
    rows = arm_b["census_rows"]
    rids = sorted(int(r) for r in rows)
    strengths = [rows[str(r)]["strength"] for r in rids]
    cols3 = []
    for r in rids:
        cols3.append("crimson" if r == 0 else
                     "darkorange" if r == 183 else
                     "steelblue" if r in BAND else "lightgray")
    ax.bar(np.arange(len(rids)), strengths, 0.75, color=cols3,
           edgecolor="k", lw=0.3)
    ctrl_max = arm_b["controls"]["max_strength"]
    ax.axhline(ctrl_max, color="gray", ls=":", lw=1.2,
               label=f"control-max {ctrl_max:.4f}")
    ax.set_xticks(np.arange(len(rids)))
    ax.set_xticklabels([str(r) for r in rids], fontsize=5.5, rotation=45)
    ax.set_ylabel("strength = min(mean-drop, zero-drop)")
    ax.set_title("arm_b row census on its site battery (row 0 vs controls, "
                 "band 121-137, site 183)", fontsize=9.5)
    ax.legend(fontsize=7)

    # (1,1) verdict
    a = adj
    cc = a["carrier_control"]
    txt = (f"VERDICT: {a['verdict']}\n"
           f"arm_b dial:      rel_S0 {a['dial_blind']['rel_S0_arm_b']:.4f} | "
           f"survivor {a['dial_valid']['survivor_arm_b']:.4f}\n"
           f"  ground truth (e133): share 0.0706, survivor "
           f"{a['ground_truth']['survivor_known']:.4f} "
           f"(reproduced: {a['ground_truth']['zero_arm_reproduced']})\n"
           f"carrier dial:    rel_S0 {cc['rel_S0']:.4f} "
           f"(e131 stored {cc['e131_stored_rel']:.4f}; in census band: "
           f"{cc['within_census_band']})\n"
           f"DIAL-BLIND (rel >= 0.9): {a['dial_blind']['fires']}\n"
           f"DIAL-VALID (survivor 0.7-0.8): {a['dial_valid']['fires']}\n"
           f"intro: {a['what_happens_to_the_intro']}")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=7.2, va="top",
            family="monospace", wrap=True,
            bbox=dict(facecolor="lightyellow", alpha=0.94, edgecolor="gray"))
    ax.set_axis_off()
    ax.set_title("adjudication (registered bars; no shopping)", fontsize=10)

    fig.suptitle(f"E163 — THE SATURATION CONTROL -> {adj['verdict']} "
                 f"(arm_b rel {arm_b['rel_S0']:.3f} / survivor "
                 f"{arm_b['survivor']:.3f} vs census band "
                 f"{census_ctx['band']['rel_S0_min']:.3f}-"
                 f"{census_ctx['band']['rel_S0_max']:.3f})"
                 if census_ctx["band"] else
                 f"E163 — THE SATURATION CONTROL -> {adj['verdict']}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
