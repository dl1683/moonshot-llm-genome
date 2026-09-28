"""E142 — ROW-0 AT BIRTH: the install-dose census (R44 ideator; REGISTERED).

WHY (T082's taxonomy, origin question): the five-day "address -> field"
arc (install binds a local address -> consolidation re-keys to the row-0
hub, W011) assumes row 0 arrived LATE. But T069/e116 found row 0
content-carrying in 6/6 installs, and W011's savor (c) says consolidation
may merely PROMOTE the already-largest seed. This census asks the ORIGIN
question retroactively across the whole install-dose ladder: was there
EVER an address-only phase (row 0 null at birth), or has row 0 co-carried
every install from birth? A ROW-0-ALWAYS verdict rewrites the lab's
history: the "address -> field" arc was row-0-co-carried throughout,
mistaken for address-binding because batteries read the band.

INSTRUMENTS (reused VERBATIM with provenance):
  * ROW-0 CONTENT TEST — e131 probe-2 / e116 census: on each net's OWN
    install battery (its own instrument context — e140/T083's lesson: the
    row-0 dial at trained geometries measures sink-load, so absolutes are
    per-net quantities and the verdict rides on the ACROSS-DOSE shape and
    on rel = S/base), arms wpe[r] <- mean(all rows) and wpe[r] <- 0;
    drop = base p(name-char) - arm p; strength(r) = min(mean-drop,
    zero-drop); e116 content criterion = mean>0 AND zero>0 AND
    min/max ratio >= 0.5. (e098's census used the same mean-arm on the
    fresh family; the zero arm is e131's addition.)
  * DECISION-ROW READ — same census on each net's known address row
    (129 for the B43/e048/e044 lines; per e098's stored metrics for the
    fresh family: 129/129/129/127; e117 has NO stored census -> its
    address row is DISCOVERED in-run by an e098-method mean-arm scan over
    rows 0..137, noted as such).
  * BATTERY per net (own geometry): e043/e131 install-60 130-token
    contexts (SPLICE_RNG 24301, corpus seed 1337) for all protocol installs;
    e044 arm (a) reads p(J) (JULIET re-install) on the same contexts;
    e048 direct cells read p(Z) on their OWN home battery (the spliced
    corpus of e048 arm 4, 54 contexts) — with the install-battery read
    recorded as off-geometry context.

NETS (runs/checkpoints/, all gated where a stored gate value exists):
  e048 dose ladder (2.7M): e048_repro.pt (install s400, 1x),
    e048_dose.pt (install s1600, 4x), e048_direct400.pt / e048_direct800.pt
    (natural/direct exposure s400/s800).
  e044 line installs (2.7M, on the erased/scarred base, s400):
    e044_b_zephyra.pt, e044_b2_zephyra.pt, e044_a_reinstall.pt (JULIET).
  e098 fresh family (0.84M, block 512): e098_install_s4305..s4308.pt
    (T069's row-0-heavy line; NOTE s4307.pt is the first-trajectory s100
    net — its run's gate-passing twin is the _pat file, not substituted
    per tasking).
  cross-family: e117_install_s4309.pt (0.84M), e082_b43_install.pt (2.7M).
  Missing files are skipped with a note; no substitutions.

REGISTERED PREDICTION (adjudicated against exactly this; no bar shopping):
  * ADDRESS-ONLY-EVER fires if: any checkpoint (especially lowest dose)
    shows row-0 <= 2x max control row while its decision row clears the
    content bar — a genuine address-bound phase existed; migration is real
    history.
  * ROW-0-ALWAYS fires if: row-0 clears the content bar at every dose —
    the address->field story was row-0-co-carried throughout;
    consolidation compresses to share-growth of the largest seed (W011's
    (c) promoted from savor to law).
  * HUB-FIRST fires if: low-dose installs are row-0-DOMINANT (S_0 >
    decision-row S) and the decision row takes over at higher doses.
  * Texture => TEXTURE with numbers (e.g., mixed families).

OPERATIONALIZATIONS (fixed before compute):
  * "row-0 clears the content bar" = e116 criterion (mean>0, zero>0,
    min/max arm ratio >= 0.5) on that net's own battery.
  * "row-0 <= 2x max control row": control rows are family-specific nulls
    (the e131 control list for the 2.7M line: 1,2,3,4,5,6,60,100,118,119,
    120; for the fresh 512-row family: fed ordinary rows 40,60,80,100,110,
    120 + unfed null-band rows 200,300,400 — row 1 is EXCLUDED there
    because e098's stored census identifies it as a top-2 carrier, and is
    reported separately). Control max = max measured control strength.
  * "decision row clears the content bar" = e116 criterion on the known
    address row (129 for 2.7M lines; 129/127 per e098's stored excl-0 tops
    for the fresh seeds; e117's discovered row).
  * "low-dose / higher doses" = the e048 dose ladder (the only true dose
    axis): repro s400 (lowest) vs dose s1600 (highest); the direct pair is
    a natural-exposure contrast, reported, not gating.
  * rel = S/base reported everywhere alongside absolutes.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import;
e143 owns the GPU — never touched). Eval-only, no training. Modest
threads (4), sequential net loads, no busy-waiting.

Outputs: runs/e142/{metrics.json, row0_at_birth.png}.
No NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e142_row0_at_birth.py     (E142_SMOKE=1 -> shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"         # CPU-ONLY (e143 owns GPU)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(4)                              # modest threads

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = _os.environ.get("E142_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
JULIET = "JULIET"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
POST_CAP = 119                                         # e043 deviation-1
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

CFG_256 = None                                         # filled in main (vocab 65 default: 6L/6H/192d/256)
CFG_512 = None                                         # 4L/4H/128d/512 (e053c fresh family)

CONTROL_ROWS_256 = (1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120)   # e131 verbatim
CONTROL_ROWS_512 = (40, 60, 80, 100, 110, 120, 200, 300, 400)   # fresh-family nulls
BAND = tuple(range(121, 138))                          # report/context rows
EXTRA_ROWS_512 = (1,)                                  # row 1: fresh-family top-2 carrier (report)

G_BIT_TOL = 5e-6                                       # CPU-stored refs (e131 G_E116 convention)
G_FALLBACK_TOL = 0.05                                  # e113 convention fallback
G_GPU_TOL = 1e-3                                       # GPU-evaluated stored refs (e117/e098 traj)

REGISTERED_PREDICTION = {
    "address_only_ever": "ADDRESS-ONLY-EVER fires if: any checkpoint "
                         "(especially lowest dose) shows row-0 <= 2x max control "
                         "row while its decision row clears the content bar — a "
                         "genuine address-bound phase existed; migration is real "
                         "history.",
    "row0_always": "ROW-0-ALWAYS fires if: row-0 clears the content bar at "
                   "every dose — the address->field story was row-0-co-carried "
                   "throughout; consolidation compresses to share-growth of the "
                   "largest seed (W011's (c) promoted from savor to law).",
    "hub_first": "HUB-FIRST fires if: low-dose installs are row-0-DOMINANT "
                 "(S_0 > decision-row S) and the decision row takes over at "
                 "higher doses.",
    "no_bar_shopping": "No bar shopping. Texture => TEXTURE with numbers.",
    "operationalizations": "content bar = e116 criterion (mean>0, zero>0, "
                           "min/max arm ratio >= 0.5); control rows = e131 list "
                           "(2.7M) / fresh-family nulls (0.84M, row 1 excluded "
                           "there as a known carrier and reported separately); "
                           "decision row = known address row (129 for 2.7M "
                           "lines, 129/127 per e098 stored excl-0 tops, e117 "
                           "discovered in-run); dose axis = e048 ladder s400 vs "
                           "s1600 (direct pair = contrast, not gating); "
                           "rel = S/base.",
}

recipe_deviations: list[str] = [
    "e098_install_s4307.pt is the FIRST-TRAJECTORY s100 net (its run's "
    "gate-passing twin is e098_install_s4307_pat.pt, which e116 censused "
    "instead); the tasked file is run as listed, gated by sanity + the s100 "
    "trajectory value, with the ambiguity noted.",
    "e117_install_s4309.pt has no stored census (e117 was the M2/maturity "
    "run); its address row is DISCOVERED in-run by an e098-method mean-arm "
    "scan over rows 0..137 (top excl-0 row), with the family-convention row "
    "129 reported alongside.",
    "e044 arm gates: the saved checkpoints are plain weights (the L3H5 patch "
    "was a runtime hook), so all reads here are patch-off; arm (a) is gated "
    "against its stored patch-off battery value, arms (b)/(b2) against their "
    "stored (patch-on) battery values with the 0.05 fallback convention — "
    "any diff beyond 5e-6 is flagged as patch-state convention, not drift.",
    "Controls are family-specific (e131 list for the 2.7M line; fed-ordinary "
    "+ unfed null rows for the 512-row fresh family, where e098's stored "
    "census shows the null band 130..511 is bit-inert under the 130-token "
    "battery and row 1 is a top-2 carrier). Bars compare within-net, so the "
    "different null conventions do not cross family lines.",
    "The row-0 dial at trained geometries measures sink-load and saturates "
    "(T083); per design this census reads trained geometries only — "
    "absolutes are per-net sink-load quantities, rel=S/base is reported "
    "everywhere, and the verdict rides on the across-dose shape.",
]


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
    """e131's battery_pz VERBATIM: mean p(char) at the last position."""
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, cid]) for k in range(pr.shape[0])]
    return float(np.mean(pzs))


@torch.no_grad()
def battery_nll_acc(net: TinyGPT, seq: torch.Tensor, L: int, lo: int,
                    chunk=60) -> tuple[float, float]:
    """e043 eval_seq / e044 eval_bat44 metric path, CPU, patch-off:
    mean NLL + argmax acc over the L name chars at slice [lo, lo+L)."""
    net.eval()
    x, y = seq[:, :-1], seq[:, 1:]
    nlls, accs = [], []
    for i in range(0, len(x), chunk):
        logits, _ = net(x[i:i + chunk])
        lg = logits[:, lo:lo + L, :]
        tg = y[i:i + chunk][:, lo:lo + L]
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                              reduction="none").view(-1, L)
        nlls.append(nll)
        accs.append((lg.argmax(-1) == tg).float())
    n = torch.cat(nlls)
    return float(n.mean()), float(torch.cat(accs).mean())


@torch.no_grad()
def row_census(net: TinyGPT, ids: torch.Tensor, cid: int,
               rows: list[int]) -> dict:
    """e131 probe-2 census VERBATIM (mean arm + zero arm + e116 criterion)."""
    base_pz = battery_pz(net, ids, cid)
    w = net.wpe.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    out: dict[int, dict] = {}
    for r in rows:
        w.copy_(orig); w[r] = mean_row
        m_d = base_pz - battery_pz(net, ids, cid)
        w.copy_(orig); w[r] = 0.0
        z_d = base_pz - battery_pz(net, ids, cid)
        hi = max(m_d, z_d)
        out[int(r)] = {
            "mean": float(m_d), "zero": float(z_d),
            "ratio": float(min(m_d, z_d) / hi) if hi > 0 else 0.0,
            "strength": float(min(m_d, z_d)),
            "content": bool(m_d > 0 and z_d > 0 and
                            (min(m_d, z_d) / hi) >= 0.5),
        }
    w.copy_(orig)
    return {"base_pz": base_pz, "rows": out}


@torch.no_grad()
def mean_arm_scan(net: TinyGPT, ids: torch.Tensor, cid: int,
                  rows: range) -> dict:
    """e098's census arm (mean only): drop(r) = base p - p(r<-mean)."""
    base_pz = battery_pz(net, ids, cid)
    w = net.wpe.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    drops = {}
    for r in rows:
        w.copy_(orig); w[r] = mean_row
        drops[int(r)] = float(base_pz - battery_pz(net, ids, cid))
    w.copy_(orig)
    return {"base_pz": base_pz, "drops": drops}


# ------------------------------------------------------------------ net table

NETS = [
    # --- e048 dose ladder (the dose axis; 2.7M B43-line) ---
    dict(file="e048_repro.pt", cfg="256", group="e048_dose_ladder",
         family="B43-line 2.7M (6L/6H/192d/256)", dose=1,
         dose_label="install s400 (1x)", battery="install60", char="Z",
         dec_row=129, gate=dict(kind="e116", key="42")),
    dict(file="e048_dose.pt", cfg="256", group="e048_dose_ladder",
         family="B43-line 2.7M (6L/6H/192d/256)", dose=2,
         dose_label="install s1600 (4x)", battery="install60", char="Z",
         dec_row=129,
         gate=dict(kind="nll_acc", ref="e048:arm_dose.s1600.ro.r1i",
                   seq="bat_i_seq")),
    dict(file="e048_direct400.pt", cfg="256", group="e048_dose_ladder",
         family="B43-line 2.7M (6L/6H/192d/256)", dose=3,
         dose_label="direct/natural s400", battery="home54", char="Z",
         dec_row=129,
         gate=dict(kind="nll_acc", ref="e048:arm_direct.s400.ro.r1home",
                   seq="home_seq")),
    dict(file="e048_direct800.pt", cfg="256", group="e048_dose_ladder",
         family="B43-line 2.7M (6L/6H/192d/256)", dose=4,
         dose_label="direct/natural s800", battery="home54", char="Z",
         dec_row=129,
         gate=dict(kind="nll_acc", ref="e048:arm_direct.s800.ro.r1home",
                   seq="home_seq")),
    # --- e044 line installs (2.7M, on the erased/scarred base) ---
    dict(file="e044_b_zephyra.pt", cfg="256", group="e044_line",
         family="B43-line 2.7M (6L/6H/192d/256)", dose=1,
         dose_label="fresh ZEPHYRA on erased base, s400",
         battery="install60", char="Z", dec_row=129,
         gate=dict(kind="nll_acc", ref="e044:arms.b.traj[-1].spliced",
                   seq="bat_i_seq_z")),
    dict(file="e044_b2_zephyra.pt", cfg="256", group="e044_line",
         family="B43-line 2.7M (6L/6H/192d/256)", dose=1,
         dose_label="fresh ZEPHYRA on erased base (Z rows pre-zeroed), s400",
         battery="install60", char="Z", dec_row=129,
         gate=dict(kind="nll_acc", ref="e044:arms.b2.traj[-1].spliced",
                   seq="bat_i_seq_z")),
    dict(file="e044_a_reinstall.pt", cfg="256", group="e044_line",
         family="B43-line 2.7M (6L/6H/192d/256)", dose=1,
         dose_label="JULIET re-install on scar, s400",
         battery="install60", char="J", dec_row=129,
         gate=dict(kind="nll_acc", ref="e044:arms.a.traj[-1].spliced_patchoff",
                   seq="bat_i_seq_j")),
    # --- e098 fresh family (0.84M, block 512) ---
    dict(file="e098_install_s4305.pt", cfg="512", group="e098_fresh",
         family="fresh 0.84M (4L/4H/128d/512)", dose=1,
         dose_label="install s100, seed 4305", battery="install60", char="Z",
         dec_row=129, gate=dict(kind="e116", key="4305")),
    dict(file="e098_install_s4306.pt", cfg="512", group="e098_fresh",
         family="fresh 0.84M (4L/4H/128d/512)", dose=1,
         dose_label="install s100, seed 4306", battery="install60", char="Z",
         dec_row=129, gate=dict(kind="e116", key="4306")),
    dict(file="e098_install_s4307.pt", cfg="512", group="e098_fresh",
         family="fresh 0.84M (4L/4H/128d/512)", dose=1,
         dose_label="install s100 first trajectory, seed 4307",
         battery="install60", char="Z", dec_row=129,
         gate=dict(kind="pz_ref",
                   ref="e098:per_seed.4307.install.traj[0].install60_pz",
                   tol=G_GPU_TOL,
                   note="first-trajectory net; gate-passing twin is _pat")),
    dict(file="e098_install_s4308.pt", cfg="512", group="e098_fresh",
         family="fresh 0.84M (4L/4H/128d/512)", dose=1,
         dose_label="install s100, seed 4308", battery="install60", char="Z",
         dec_row=127, gate=dict(kind="e116", key="4308")),
    # --- cross-family ---
    dict(file="e117_install_s4309.pt", cfg="512", group="cross_family",
         family="fresh 0.84M (4L/4H/128d/512)", dose=1,
         dose_label="install s100, seed 4309 (3133-step base)",
         battery="install60", char="Z", dec_row=None,      # discovered in-run
         gate=dict(kind="pz_ref", ref="e117:install.final.install60_pz",
                   tol=G_GPU_TOL)),
    dict(file="e082_b43_install.pt", cfg="256", group="cross_family",
         family="B43-line 2.7M (6L/6H/192d/256)", dose=1,
         dose_label="B43 install (cross-seed transplant line)",
         battery="install60", char="Z", dec_row=129,
         gate=dict(kind="e116", key="43")),
]


def dig(dic, path):
    for part in path.split("."):
        if part.endswith("]"):
            key, idx = part[:-1].split("[")
            dic = dic[key][int(idx)]
        else:
            dic = dic[part]
    return dic


# ------------------------------------------------------------------ main

def main():
    global CFG_256, CFG_512
    rd = run_dir("e142_smoke" if SMOKE else "e142")
    CFG_256 = Cfg()
    CFG_512 = Cfg(n_layer=4, n_head=4, n_embd=128, block_size=512)
    log(f"E142 ROW-0 AT BIRTH (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda visible = {torch.cuda.is_available()}")

    refs = {}
    for key, path in (("e116", E43.REPO / "runs" / "e116" / "metrics.json"),
                      ("e048", E43.REPO / "runs" / "e048" / "metrics.json"),
                      ("e044", E43.REPO / "runs" / "e044" / "metrics.json"),
                      ("e098", E43.REPO / "runs" / "e098" / "metrics.json"),
                      ("e117", E43.REPO / "runs" / "e117" / "metrics.json")):
        refs[key] = json.loads(Path(path).read_text(encoding="utf-8")) \
            if Path(path).exists() else None

    # ---------------- protocol rebuild (e043/e048 verbatim)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid, jid = stoi["Z"], stoi["J"]
    train_ids = corpus.train
    train_text = "".join(itos[int(i)] for i in train_ids)

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ, rest_occ = host_occ[:60], host_occ[60:90], host_occ[90:]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30, rest{len(rest_occ)}")

    name_ids = torch.tensor([stoi[c] for c in NAME])
    jul_ids = torch.tensor([stoi[c] for c in JULIET])

    # install-60 130-token battery (e131/e116 construction verbatim)
    bat_install = torch.stack([corpus.encode(train_text[p - PRE: p])
                               for p, _ in install_occ])
    # gate sequences: ctx130 + name (e043 bat_i / e044 bat_i)
    bat_i_seq_z = torch.stack(
        [torch.cat([train_ids[p - PRE: p], name_ids]) for p, _ in install_occ])
    bat_i_seq_j = torch.stack(
        [torch.cat([train_ids[p - PRE: p], jul_ids]) for p, _ in install_occ])

    # e048 direct home battery (arm 4 spliced corpus, verbatim)
    parts, last = [], 0
    for p, host in sorted(rest_occ, key=lambda ph: ph[0]):
        parts.append(train_ids[last:p]); parts.append(name_ids)
        last = p + len(host)
    parts.append(train_ids[last:])
    sp_ids = torch.cat(parts)
    sp_text = "".join(itos[int(i)] for i in sp_ids)
    sp_occ = E43.find_occ(sp_text, NAME)
    assert len(sp_occ) == 60, f"splice produced {len(sp_occ)}"
    home_occ = [q for q in sp_occ
                if q >= PRE and NAME not in sp_text[q - PRE: q]][:60]
    bat_home = torch.stack([corpus.encode(sp_text[q - PRE: q]) for q in home_occ])
    home_seq = torch.stack([torch.cat([sp_ids[q - PRE: q], name_ids])
                            for q in home_occ])
    log(f"home battery rebuilt: {len(home_occ)} contexts (e048 ref 54)")

    batteries = {"install60": (bat_install, zid),
                 "home54": (bat_home, zid)}
    gate_seqs = {"bat_i_seq": (bat_i_seq_z, 7), "home_seq": (home_seq, 7),
                 "bat_i_seq_z": (bat_i_seq_z, 7), "bat_i_seq_j": (bat_i_seq_j, 6)}

    # ---------------- the census
    results, gates_all, skipped = {}, {}, []
    for spec in NETS:
        path = CKPT_DIR / spec["file"]
        if not path.exists():
            skipped.append(spec["file"])
            log(f"SKIP {spec['file']} (missing; not substituted)")
            continue
        cfg = CFG_256 if spec["cfg"] == "256" else CFG_512
        net = load_cpu(path, cfg)
        ids, cid = batteries[spec["battery"]]
        if spec["char"] == "J":
            cid = jid

        # ---- gate
        g = spec["gate"]
        gate_rec = {"kind": g["kind"]}
        if g["kind"] == "e116":
            ref = refs["e116"]["per_seed"][g["key"]]
            base_now = battery_pz(net, ids, cid)
            gate_rec.update({"ref_base_pz": ref["base_pz"],
                             "base_pz": base_now,
                             "base_diff": abs(base_now - ref["base_pz"]),
                             "tol": G_BIT_TOL})
            gate_rec["base_ok"] = bool(gate_rec["base_diff"] < G_BIT_TOL)
        elif g["kind"] == "pz_ref":
            run_key, ref_path = g["ref"].split(":", 1)
            ref_val = float(dig(refs[run_key], ref_path))
            base_now = battery_pz(net, ids, cid)
            gate_rec.update({"ref_base_pz": ref_val, "base_pz": base_now,
                             "base_diff": abs(base_now - ref_val),
                             "tol": g.get("tol", G_GPU_TOL)})
            gate_rec["base_ok"] = bool(gate_rec["base_diff"] < gate_rec["tol"])
            if "note" in g:
                gate_rec["note"] = g["note"]
        elif g["kind"] == "nll_acc":
            run_key, ref_path = g["ref"].split(":", 1)
            ref = dig(refs[run_key], ref_path)
            seq, L = gate_seqs[g["seq"]]
            nll, acc = battery_nll_acc(net, seq, L, PRE - 1)
            gate_rec.update({"ref_nll": ref["nll"], "ref_acc": ref["acc"],
                             "nll": nll, "acc": acc,
                             "nll_diff": abs(nll - ref["nll"]),
                             "acc_diff": abs(acc - ref["acc"]),
                             "tol": G_BIT_TOL, "fallback_tol": G_FALLBACK_TOL})
            gate_rec["bit_reproducible"] = bool(
                gate_rec["nll_diff"] < G_BIT_TOL and
                gate_rec["acc_diff"] < G_BIT_TOL)
            gate_rec["passes_e113_convention"] = bool(
                gate_rec["nll_diff"] < G_FALLBACK_TOL and
                gate_rec["acc_diff"] < G_FALLBACK_TOL)
            gate_rec["base_ok"] = gate_rec["passes_e113_convention"]
        gates_all[spec["file"]] = gate_rec
        log(f"GATE {spec['file']}: " + json.dumps(
            {k: (round(v, 8) if isinstance(v, float) else v)
             for k, v in gate_rec.items() if k != "kind"})[:220])

        # ---- e117 address-row discovery (e098 method, rows 0..137)
        dec_row = spec["dec_row"]
        discovery = None
        if dec_row is None:
            scan = mean_arm_scan(net, ids, cid, range(0, 138))
            cand = {r: d for r, d in scan["drops"].items() if r != 0}
            top = max(cand, key=cand.get)
            discovery = {"method": "e098 mean-arm scan rows 0..137, top excl-0",
                         "scan_base_pz": scan["base_pz"],
                         "top_excl0_row": int(top),
                         "top_excl0_drop": cand[top],
                         "fallback_convention_row": 129,
                         "scan_drops": scan["drops"]}
            dec_row = int(top)
            log(f"  e117 address-row discovery: top excl-0 row {top} "
                f"(drop {cand[top]:.4f}); convention row 129 reported too")

        # ---- census rows
        controls = CONTROL_ROWS_256 if spec["cfg"] == "256" else CONTROL_ROWS_512
        extras = EXTRA_ROWS_512 if spec["cfg"] == "512" else ()
        rows = sorted({0, dec_row, 129} | set(controls) | set(BAND) | set(extras))
        cen = row_census(net, ids, cid, rows)

        # ---- e116 row-0 twin gate (kind e116 only)
        if g["kind"] == "e116":
            ref0 = refs["e116"]["per_seed"][g["key"]]["row0"]
            r0 = cen["rows"][0]
            gate_rec.update({
                "ref_row0_mean": ref0["mean"], "ref_row0_zero": ref0["zero"],
                "row0_mean_diff": abs(r0["mean"] - ref0["mean"]),
                "row0_zero_diff": abs(r0["zero"] - ref0["zero"]),
                "row0_tol": G_BIT_TOL,
                "row0_ok": bool(max(abs(r0["mean"] - ref0["mean"]),
                                    abs(r0["zero"] - ref0["zero"])) < G_BIT_TOL)})
            gate_rec["all_ok"] = bool(gate_rec.get("base_ok") and
                                      gate_rec.get("row0_ok"))

        r0 = cen["rows"][0]
        dec = cen["rows"][dec_row]
        ctrl_rows = [cen["rows"][r] for r in controls]
        ctrl_max = max(c["strength"] for c in ctrl_rows)
        cand2 = {r: v for r, v in cen["rows"].items() if r != 0}
        top_excl0 = max(cand2, key=lambda r: cand2[r]["strength"])
        base_pz = cen["base_pz"]
        rec = {
            "file": spec["file"], "family": spec["family"],
            "group": spec["group"], "dose": spec["dose"],
            "dose_label": spec["dose_label"], "battery": spec["battery"],
            "battery_n": int(ids.shape[0]), "char": spec["char"],
            "dec_row": dec_row,
            "base_pz": base_pz,
            "row0": r0,
            "decision_row": dec,
            "controls": {"rows": {str(r): cen["rows"][r] for r in controls},
                         "max_strength": ctrl_max},
            "top_excl0_row_measured": int(top_excl0),
            "top_excl0_strength_measured": cand2[top_excl0]["strength"],
            "S0": r0["strength"], "S0_content": r0["content"],
            "Sdec": dec["strength"], "Sdec_content": dec["content"],
            "rel_S0": float(r0["strength"] / base_pz) if base_pz > 0 else None,
            "rel_Sdec": float(dec["strength"] / base_pz) if base_pz > 0 else None,
            "S0_le_2x_control": bool(r0["strength"] <= 2 * ctrl_max),
            "census_rows": {str(r): cen["rows"][r] for r in rows},
            "gate": gate_rec,
        }
        if discovery:
            rec["address_row_discovery"] = discovery
        # off-geometry context read for the direct cells
        if spec["battery"] == "home54":
            rec["off_geometry_install60_pz"] = battery_pz(net, bat_install, zid)
        results[spec["file"]] = rec
        log(f"  {spec['file']}: base {base_pz:.4f} | S0 {r0['strength']:.4f} "
            f"(content {r0['content']}, rel {rec['rel_S0']:.3f}) | "
            f"Sdec[r{dec_row}] {dec['strength']:.4f} (content "
            f"{dec['content']}, rel {rec['rel_Sdec']:.3f}) | ctrl-max "
            f"{ctrl_max:.4f} | top excl-0 r{top_excl0}")
        del net
        gc.collect()

    # ---------------- adjudication (registered)
    nets = list(results.values())
    addr_hits = [r["file"] for r in nets
                 if r["S0_le_2x_control"] and r["Sdec_content"]]
    address_only_ever = bool(addr_hits)
    row0_always = bool(nets) and all(r["S0_content"] for r in nets)
    ladder = [r for r in nets if r["group"] == "e048_dose_ladder"
              and "install" in r["dose_label"]]
    ladder = sorted(ladder, key=lambda r: r["dose"])
    hub_first = None
    if len(ladder) >= 2:
        lo, hi = ladder[0], ladder[-1]
        hub_first = bool(lo["S0"] > lo["Sdec"] and hi["Sdec"] > hi["S0"])
    n_row0_dominant = sum(1 for r in nets if r["S0"] > r["Sdec"])

    if address_only_ever:
        fired = "ADDRESS-ONLY-EVER"
    elif row0_always and hub_first:
        fired = "ROW-0-ALWAYS (+ HUB-FIRST shape at birth: row-0-dominant low-dose)"
    elif row0_always:
        fired = "ROW-0-ALWAYS"
    elif hub_first:
        fired = "HUB-FIRST"
    else:
        fired = "TEXTURE"

    adjudication = {
        "address_only_ever": {"fires": address_only_ever,
                              "hits": addr_hits},
        "row0_always": {"fires": row0_always,
                        "n_nets": len(nets),
                        "n_row0_content": sum(1 for r in nets if r["S0_content"])},
        "hub_first": {"fires": hub_first,
                      "ladder": [r["file"] for r in ladder],
                      "n_row0_dominant_all_nets": n_row0_dominant},
        "verdict": fired,
    }
    log("=" * 78)
    log(f"E142 VERDICT: {fired}")
    log(f"  ADDRESS-ONLY-EVER: {address_only_ever} (hits: {addr_hits})")
    log(f"  ROW-0-ALWAYS: {row0_always} "
        f"({sum(1 for r in nets if r['S0_content'])}/{len(nets)} row-0 content-positive)")
    log(f"  HUB-FIRST: {hub_first} | row-0-dominant in {n_row0_dominant}/{len(nets)} nets")
    log("=" * 78)

    # ---------------- outputs
    order = {"e048_dose_ladder": 0, "e044_line": 1, "e098_fresh": 2,
             "cross_family": 3}
    table = sorted(nets, key=lambda r: (order[r["group"]], r["dose"],
                                        r["file"]))
    metrics = {
        "experiment": "e142_row0_at_birth",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("R44 ideator; coordinator dispatch. Docstring + bars "
                         "written before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("was there ever an address-only phase at birth, or has "
                     "row 0 co-carried every install from the start (making "
                     "the five-day address->field arc row-0-co-carried "
                     "throughout)?"),
        "instruments": {
            "row0_content_test": "e131 probe-2 / e116 census VERBATIM "
                                 "(mean arm + zero arm; strength = "
                                 "min(mean-drop, zero-drop); content = e116 "
                                 "criterion)",
            "battery": "each net's OWN install battery (e043/e131 install-60 "
                       "130-token contexts; e048 direct cells on their "
                       "spliced-corpus home battery; e044 arm (a) reads "
                       "p(J))",
            "decision_row": "known address row (129 for 2.7M lines; "
                            "129/127 per e098 stored excl-0 tops; e117 "
                            "discovered in-run)",
            "rel": "S/base reported for every net (T083: trained-geometry "
                   "absolutes measure sink-load)",
        },
        "gates": gates_all,
        "nets": table,
        "adjudication": adjudication,
        "skipped_missing": skipped,
        "recipe_deviations": recipe_deviations,
        "honesty_reflex": (
            "Instrument context is per net: the 2.7M B43-line and the 0.84M "
            "fresh family differ in geometry (block 256 vs 512), battery "
            "provenance (protocol installs vs spliced-corpus home for the "
            "direct cells), control nulls (e131 list vs fresh-family nulls), "
            "and name char (Z vs J for the e044 JULIET re-install). "
            "Cross-family comparisons of absolute strengths are therefore "
            "NOT like-for-like; every bar here is adjudicated WITHIN a net "
            "against its own controls/decision row, and the dose-axis "
            "verdict rests on the within-family e048 ladder. Trained-"
            "geometry row-0 strength measures sink-load and saturates "
            "(T083) — this census cannot see routing, only carriage; the "
            "at-birth question (was row 0 EVER null) is nevertheless "
            "well-posed on this dial because a null row-0 (<= 2x controls) "
            "would show even at a trained geometry."),
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"threads": torch.get_num_threads(), "device": "cpu",
                   "smoke": SMOKE,
                   "cfg_256": "6L/6H/192d/256 (2.7M)",
                   "cfg_512": "4L/4H/128d/512 (0.84M)"},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    plot(rd / "row0_at_birth.png", table, adjudication)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'row0_at_birth.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def plot(path, table, adj):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    groups = [("e048_dose_ladder", "E048 DOSE LADDER (2.7M): the dose axis",
               axes[0, 0]),
              ("e044_line", "E044 LINE INSTALLS + cross-family (2.7M)",
               axes[0, 1]),
              ("e098_fresh", "E098 FRESH FAMILY (0.84M, block 512)",
               axes[1, 0])]

    for gname, title, ax in groups:
        rows = [r for r in table if r["group"] == gname]
        if not rows:
            continue
        xs = np.arange(len(rows))
        s0 = [r["S0"] for r in rows]
        sd = [r["Sdec"] for r in rows]
        cm = [r["controls"]["max_strength"] for r in rows]
        ax.bar(xs - 0.21, s0, 0.38, color="crimson", edgecolor="k", lw=0.5,
               label="S_0 (row 0)")
        ax.bar(xs + 0.21, sd, 0.38, color="steelblue", edgecolor="k", lw=0.5,
               label="S_decision (address row)")
        ax.plot(xs, cm, "o", ms=4, color="gray",
                label="max control strength")
        for i, r in enumerate(rows):
            ax.text(i - 0.21, max(s0[i], 0) + 0.012,
                    f"{s0[i]:.3f}\nrel {r['rel_S0']:.2f}", ha="center",
                    fontsize=6.2)
            ax.text(i + 0.21, max(sd[i], 0) + 0.012,
                    f"{sd[i]:.3f}\nrel {r['rel_Sdec']:.2f}", ha="center",
                    fontsize=6.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['file'].replace('.pt','')}\n[r{r['dec_row']}] "
                            f"{r['dose_label']}" for r in rows], fontsize=6.2)
        ax.set_ylabel("strength = min(mean-drop, zero-drop)")
        ax.set_ylim(bottom=min(0, min(s0 + sd) - 0.05))
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)

    # cross-family rel view + verdict
    ax = axes[1, 1]
    xs = np.arange(len(table))
    rel0 = [r["rel_S0"] for r in table]
    reld = [r["rel_Sdec"] for r in table]
    ax.bar(xs - 0.21, rel0, 0.38, color="crimson", edgecolor="k", lw=0.5,
           label="rel S_0 = S_0 / base")
    ax.bar(xs + 0.21, reld, 0.38, color="steelblue", edgecolor="k", lw=0.5,
           label="rel S_dec / base")
    fam_col = {"e048_dose_ladder": "k", "e044_line": "darkorange",
               "e098_fresh": "seagreen", "cross_family": "purple"}
    for i, r in enumerate(table):
        ax.plot([i, i], [-0.03, -0.012], color=fam_col[r["group"]], lw=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["file"].replace(".pt", "").replace("e098_install_", "")
                        .replace("e048_", "").replace("e044_", "")
                        for r in table], fontsize=6.2, rotation=45,
                       ha="right")
    ax.set_ylabel("share of base expression")
    ax.set_title("CO-CARRY SHARES, all nets (ticks colored by family)", fontsize=10)
    ax.legend(fontsize=7)

    a = adj
    txt = (f"VERDICT: {a['verdict']}\n"
           f"ADDRESS-ONLY-EVER: {a['address_only_ever']['fires']} "
           f"(hits: {a['address_only_ever']['hits'] or 'none'})\n"
           f"ROW-0-ALWAYS: {a['row0_always']['fires']} "
           f"({a['row0_always']['n_row0_content']}/{a['row0_always']['n_nets']} "
           f"row-0 content-positive)\n"
           f"HUB-FIRST: {a['hub_first']['fires']} | row-0-dominant in "
           f"{a['hub_first']['n_row0_dominant_all_nets']}/{a['row0_always']['n_nets']} nets\n"
           f"(bars adjudicated within-net; trained-geometry strengths = "
           f"sink-load, T083)")
    ax.text(0.02, 0.60, txt, transform=ax.transAxes, fontsize=7.0, va="top",
            family="monospace",
            bbox=dict(facecolor="lightyellow", alpha=0.94, edgecolor="gray"))

    fig.suptitle(f"E142 — ROW-0 AT BIRTH: the install-dose census -> "
                 f"{adj['verdict']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
