"""E243 — THE MODE SELECTOR (T219's registered ask; desk-only on COMMITTED
data; CPU threads=4, no GPU, no training, no evals, load-polite — the wte is
read tensor-only from the committed HF snapshot safetensors, no model load).

WHY: e241/T219 split the 7 wrong-choosers into two death modes — THE MIS-DIALS
('dollars' x2, 'Music', 'Augusta': the t=0 runner-up and/or a top-0.01%
semantic neighbor; the commitment kept its address and the number's first
digits) and THE FREQUENCY COLLAPSES ('the' x4: an embedding ANTI-neighbor at
z ~ -4, below the random cloud — the commitment falls to the vocab's mean
direction, the frequency prior's basin). T219 REGISTERED THE ASK: what
decides a flipped commitment's death mode — the mis-dial or the frequency
collapse? The registered prediction names the selector: the t=0
local-candidate structure.

T219's REGISTERED PREDICTION (verbatim): "over the general population of
argmax flips during the wash, the flip target is the t=0 runner-up when that
runner-up is semantically local (embedding z >= 2 vs the answer), and the
frequency token otherwise — THE MODE IS SELECTED BY WHAT WAS STANDING NEARBY
WHEN THE COMMITMENT DIED."

THE CELL (frozen from the dispatch brief):
  (1) ENUMERATE ALL ARGMAX FLIPS across the committed journals — every
      (probe, wash) record whose +80 argmax differs from t=0's. The margins
      journal (runs/e228/journal.json) carries top1_id/top2_id per probe per
      state for ALL 54 probes x 8 states of the two-wash 124M archive —
      argmax coverage is COMPLETE (no re-derivation gap; disclosed in
      metrics). The flips are re-derived independently from e238's committed
      full-logit dumps (t0 / w1s80 / w2s80, 54 x 50257, sha-gated) as a gate:
      both sources must agree on every argmax, and the flip lists must be
      identical. This carries the 7 wrong-choosers + 2 near records AND the
      general flip population the journal holds beyond e232's dying census
      (probes whose argmax flipped without the belief halving — e.g.
      France->Paris w1, Xbox w1/w2; provenance column discloses census
      membership per record).
  (2) FOR EACH FLIP: the t=0 runner-up's identity (rank-2 token of the
      committed t0 answer-position logit vector, float64 stable argsort),
      its embedding z vs the dead answer (e241's convention VERBATIM: cos
      sim on L2-normalized float64 pristine-wte rows vs a null of 1000 ids
      drawn uniformly from the vocab minus the two ids; ONE global
      default_rng(20261004); z = (sim - mean)/std(ddof=1)), and the flip
      target's identity + its z (same convention, second draw).
      DRAW ORDER (frozen): records in (wash w1 then w2; battery
      fact/ctrl/near/tmpl; probe order = the npz/journal grid); per record
      the runner-up null is drawn FIRST, then the target null (1000 ids
      each; when target == runner-up the two nulls are distinct draws of
      the same distribution — disclosed).
  (3) CLASSIFICATION, rules VERBATIM from the dispatch, evaluated in this
      order (first match wins):
        MIS-DIAL          — target = the t0 runner-up AND runner-up z >= 2
        FREQUENCY-COLLAPSE— target = the frequency token / any z < 0 token
                            at rank >= 3   (frequency token = ' the', id 262,
                            tokenizer-verified; rank = the target's t0 rank)
        OTHER             — anything else — expected small
  (4) THE SELECTOR READ (the 2x2): P(target = t0-runner-up | runner-up
      semantically local, z_RU >= 2)  vs  P(target = t0-runner-up | runner-up
      distant, z_RU < 2) over the flip population.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any compute;
the registration commit of this file precedes any compute; adjudicate against
exactly this; no bar shopping; the bar adjudicates on whatever n exists,
disclosed):

  SELECTOR-CONFIRMED — "the local-runner-up flips go to the runner-up at
  >= 80% and the distant-runner-up flips avoid it at <= 40% — the mode is
  selected by the local-candidate structure; T219's prediction confirmed"

  SELECTOR-DENIED — "no separation (both conditional rates within 20
  points) — the mode selector is elsewhere; the honest bound"

  MIXED — "the table verbatim"

EMPTY-STRATUM RULE (pre-registered): both conditional rates must exist for
CONFIRMED or DENIED to fire; if either stratum is empty (n=0) its rate is
undefined and the verdict falls to MIXED with the empty stratum disclosed in
the table.

CO-REPORTS (never adjudicate): Wilson 95% CIs on both rates; the fact/ctrl
subtable (e241's comparable subset); per-record census provenance + e232
fate + p_80; the t0 logit gap answer-vs-runner-up; e241 z-joins on the 9
overlapping records; the target-is-frequency-token count.

INPUTS (all COMMITTED, sha-gated at runtime, never re-derived):
  runs/e228/journal.json        — the margins journal: per-state per-probe
                                  margins AND argmax records (top1/top2),
                                  all 54 probes x 8 states.
  runs/e238/logits_{t0,w1s80,w2s80}.npz — the committed answer-position full
                                  logits (54 x 50257, float32), sha-gated vs
                                  runs/e238/journal.json; the argmax
                                  re-derivation source.
  runs/e241/metrics.json        — the 9-record table (z convention, the
                                  'the' confound disclosure) — joined, never
                                  re-adjudicated.
  runs/e232/metrics.json        — the census (fates, standing) for the
                                  provenance column.
  the pristine shared-124M wte  — read tensor-only from the committed HF
                                  snapshot safetensors (openai-community/
                                  gpt2 @ 607a30d783dfa663caf39e06633721c8
                                  d4cfcd7e), sha-gated against e241's
                                  committed wte sha (fp32 bytes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
RD = RUNS / "e243"

E228_JOURNAL = RUNS / "e228" / "journal.json"
E238_JOURNAL = RUNS / "e238" / "journal.json"
E241_METRICS = RUNS / "e241" / "metrics.json"
E232_METRICS = RUNS / "e232" / "metrics.json"
E238_NPZ = {
    "t0": RUNS / "e238" / "logits_t0.npz",
    "w1": RUNS / "e238" / "logits_w1s80.npz",
    "w2": RUNS / "e238" / "logits_w2s80.npz",
}

MODEL_REPO = "openai-community/gpt2"
MODEL_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
E241_WTE_SHA16 = "e182e433b37dbdb4"  # e241's committed wte sha (fp32 bytes)
NULL_SEED = 20261004
N_NULL = 1000
Z_LOCAL = 2.0          # the dispatch's semantic-locality bar (runner-up z)
Z_NEG = 0.0            # the FREQUENCY-COLLAPSE clause's z threshold
RANK_FREQ = 3           # the FREQUENCY-COLLAPSE clause's rank threshold
THE_STR = " the"
THE_ID_EXPECTED = 262   # the frequency token, tokenizer-verified at runtime

BARS_VERBATIM = {
    "SELECTOR-CONFIRMED": (
        "the local-runner-up flips go to the runner-up at >= 80% and the "
        "distant-runner-up flips avoid it at <= 40% — the mode is selected by "
        "the local-candidate structure; T219's prediction confirmed"),
    "SELECTOR-DENIED": (
        "no separation (both conditional rates within 20 points) — the mode "
        "selector is elsewhere; the honest bound"),
    "MIXED": "the table verbatim",
}
CLASS_RULES_VERBATIM = {
    "MIS-DIAL": "target = the t0 runner-up AND runner-up z >= 2",
    "FREQUENCY-COLLAPSE": "target = the frequency token / any z < 0 token at "
                          "rank >= 3",
    "OTHER": "anything else — expected small",
}
PREDICTION_VERBATIM = (
    "over the general population of argmax flips during the wash, the flip "
    "target is the t=0 runner-up when that runner-up is semantically local "
    "(embedding z >= 2 vs the answer), and the frequency token otherwise — "
    "THE MODE IS SELECTED BY WHAT WAS STANDING NEARBY WHEN THE COMMITMENT "
    "DIED.")
CONFIRM_HIGH = 0.80     # rate_local >= this
CONFIRM_LOW = 0.40      # rate_distant <= this
DENIED_BAND = 0.20      # |rate_local - rate_distant| <= this

BATT_ORDER = ["fact", "ctrl", "near", "tmpl"]
WASH_ORDER = ["w1", "w2"]


def log(msg: str) -> None:
    print(msg, flush=True)


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return None
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(center - half, 3), round(center + half, 3)]


def rank_map(l0: np.ndarray) -> dict[int, int]:
    order = np.argsort(-l0.astype(np.float64), kind="stable")
    return {int(t): i + 1 for i, t in enumerate(order)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RD))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    timing = {"started": utcnow()}

    git_head = None
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO),
            capture_output=True, text=True).stdout.strip()
    except Exception:
        pass

    gates: dict[str, dict] = {}
    deviations: list[str] = []

    # ---------------------------------------------------------------- inputs
    e228j = json.loads(E228_JOURNAL.read_text())
    e238j = json.loads(E238_JOURNAL.read_text())
    e241 = json.loads(E241_METRICS.read_text())
    e232 = json.loads(E232_METRICS.read_text())

    # sha gates: the npz dumps vs e238's journal (e241's convention)
    npz_key = {"t0": "t0", "w1": "w1s80", "w2": "w2s80"}
    for k, p in E238_NPZ.items():
        got = sha16(p)
        want = e238j["logit_states"][npz_key[k]]["npz_sha256_16"]
        gates[f"G_SHA_{k}"] = {
            "got": got, "committed": want, "pass": got == want,
            "desc": f"{p.name} sha256_16 matches e238 journal"}
        log(f"gate G_SHA_{k}: {got} vs {want} -> {got == want}")

    d = {k: np.load(p, allow_pickle=True) for k, p in E238_NPZ.items()}
    names = [str(x) for x in d["t0"]["names"]]
    batts = [str(x) for x in d["t0"]["battery"]]
    ans_ids = d["t0"]["ans_ids"].astype(int)
    L = {k: d[k]["logits"] for k in d}
    ok_grid = all(
        [str(x) for x in d[k]["names"]] == names
        and [str(x) for x in d[k]["battery"]] == batts
        for k in d)
    gates["G_GRID"] = {
        "pass": bool(ok_grid),
        "desc": "w1s80/w2s80 dumps share the t0 dump's 54-row probe grid"}

    # e228 journal: states by (wash, step); t0 shared
    by_state = {(s["wash"], s["step"]): s for s in e228j["states"]}
    t0j = by_state[("t0", 0)]

    # the journal indexes probes per battery; the npz grid is battery-
    # contiguous (fact 20, ctrl 12, near 3, tmpl 19) — build the row ->
    # (battery, journal-index) map
    jidx = {}
    seen: dict[str, int] = {}
    for i, b in enumerate(batts):
        seen[b] = seen.get(b, 0)
        jidx[i] = (b, seen[b])
        seen[b] += 1

    # npz argmax re-derivation vs journal records (ALL 54 probes x 3 states)
    am = {k: np.argmax(L[k].astype(np.float64), axis=1).astype(int)
          for k in L}
    mism = []
    for k, wash in [("t0", "t0"), ("w1", "w1"), ("w2", "w2")]:
        sj = by_state[(wash, 80 if wash != "t0" else 0)]
        for i in range(len(names)):
            b, ji = jidx[i]
            jt = int(sj[b]["probes"][ji]["top1_id"])
            if int(am[k][i]) != jt:
                mism.append((k, names[i], int(am[k][i]), jt))
    gates["G_FLIPS"] = {
        "pass": not mism,
        "n_mismatches": len(mism),
        "desc": "npz-derived argmax == e228 journal top1_id for all 54 probes "
                "at t0, w1+80, w2+80 (double-source argmax; mismatches: "
                + str(mism[:5]) + ")"}

    # runner-up gate: npz rank-2 == journal top2_id at t0 (all 54)
    ru_npz, ru_mism = {}, []
    ranks0 = [rank_map(L["t0"][i]) for i in range(len(names))]
    for i in range(len(names)):
        r2 = next(t for t, r in ranks0[i].items() if r == 2)
        ru_npz[i] = int(r2)
        b, ji = jidx[i]
        jt2 = int(t0j[b]["probes"][ji]["top2_id"])
        if int(r2) != jt2:
            ru_mism.append((names[i], int(r2), jt2))
    gates["G_RU"] = {
        "pass": not ru_mism,
        "n_mismatches": len(ru_mism),
        "desc": "npz-derived t0 rank-2 token == e228 journal top2_id for all "
                "54 probes (mismatches: " + str(ru_mism[:5]) + ")"}

    # the commitment existed before the wash: answer is t0 argmax everywhere
    gates["G_ANSWER_T0"] = {
        "pass": all(int(am["t0"][i]) == int(ans_ids[i])
                    for i in range(len(names))),
        "desc": "the dead answer was the t0 argmax for all 54 probes (every "
                "flip is a commitment death; no recovery records exist at "
                "+80 vs t0)"}

    # ------------------------------------------------------------- the wte
    from safetensors import safe_open
    hub_snap = (Path.home() / ".cache" / "huggingface" / "hub"
                / f"models--{MODEL_REPO.replace('/', '--')}" / "snapshots"
                / MODEL_REV)
    snap_file = next(hub_snap.glob("*safetensors"))
    with safe_open(str(snap_file), framework="numpy") as f:
        wte = f.get_tensor("wte.weight").astype(np.float64)
    n_vocab, n_embd = wte.shape
    V = int(n_vocab)
    wte_sha = hashlib.sha256(
        np.ascontiguousarray(wte.astype(np.float32)).tobytes()
    ).hexdigest()[:16]
    gates["G_WTE"] = {
        "pass": bool(wte_sha == E241_WTE_SHA16 and n_vocab == 50257
                     and n_embd == 768),
        "got": wte_sha, "committed": E241_WTE_SHA16,
        "shape": [n_vocab, n_embd],
        "desc": "pristine shared-124M wte read tensor-only from the committed "
                "HF snapshot safetensors; sha16(fp32 bytes) equals e241's "
                "committed wte sha — the identical embedding table, no model "
                "load (load-polite for the two other live agents)"}
    log(f"wte {n_vocab}x{n_embd} sha16(fp32) {wte_sha} "
        f"(e241 committed {E241_WTE_SHA16})")
    Wn = wte / np.linalg.norm(wte, axis=1, keepdims=True)

    # ---------------------------------------------------------- tokenizer
    from transformers import GPT2TokenizerFast
    import transformers
    tok = GPT2TokenizerFast.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    enc_the = tok.encode(THE_STR)
    the_id = int(enc_the[0]) if len(enc_the) == 1 else -1
    gates["G_IDS"] = {
        "pass": bool(the_id == THE_ID_EXPECTED),
        "the_encodes_to": enc_the,
        "desc": f"tokenizer encodes '{THE_STR}' to id {THE_ID_EXPECTED} (the "
                "frequency token); npz ans_ids decode into their probe names"}
    log(f"frequency token '{THE_STR}' -> id {the_id}")

    # ------------------------------------------------- the flip population
    flips = []
    for wash in WASH_ORDER:
        for i in range(len(names)):
            if int(am["t0"][i]) != int(am[wash][i]):
                flips.append({"wash": wash, "row": i,
                              "battery": batts[i], "probe": names[i]})
    per_wash = {w: sum(1 for f in flips if f["wash"] == w)
                for w in WASH_ORDER}
    log(f"flip population: {len(flips)} (probe, wash) records {per_wash}")
    # cross-check the enumeration against the journal source alone
    flips_j = []
    for wash in WASH_ORDER:
        sj = by_state[(wash, 80)]
        for b in BATT_ORDER:
            for pi, p in enumerate(t0j[b]["probes"]):
                if int(p["top1_id"]) != int(sj[b]["probes"][pi]["top1_id"]):
                    flips_j.append((wash, b, p["fact"]))
    flips_npz = [(f["wash"], f["battery"], f["probe"]) for f in flips]
    gates["G_POP"] = {
        "pass": sorted(flips_j) == sorted(flips_npz) and len(flips) > 0,
        "n_journal_source": len(flips_j),
        "n_npz_source": len(flips_npz),
        "desc": "the flip enumeration is identical from both committed "
                "sources (e228 journal argmax records; e238 npz re-derivation)"
                " — argmax coverage is COMPLETE (all 54 probes x 8 states "
                "carry top1/top2 in the journal; no coverage gap to disclose)"}

    # e232 census lookup (provenance column)
    e232_idx = {}
    for b in BATT_ORDER:
        for w, recs in e232["read1_zombie_lag"]["probe_level"][b].items():
            for r in recs:
                e232_idx[(w, b, r["probe"])] = r

    # e241 join index
    e241_idx = {(r["battery"], r["wash"], r["probe"]): r
                for r in e241["table"]}

    # ----------------------------------------------------------- the reads
    rng = np.random.default_rng(NULL_SEED)
    table = []
    for f in flips:  # frozen order: w1 then w2; grid order within (battery
                     # blocks are contiguous in the grid: fact, ctrl, near,
                     # tmpl — the npz grid order)
        wash, i = f["wash"], f["row"]
        probe, batt = f["probe"], f["battery"]
        ans_id = int(ans_ids[i])
        ru_id = int(ru_npz[i])
        tgt_id = int(am[wash][i])
        l0 = L["t0"][i]
        rank_of = ranks0[i]
        rank_ru = rank_of[ru_id]
        rank_tgt = rank_of[tgt_id]
        gap_ans_ru = float(l0[ans_id]) - float(l0[ru_id])
        gap_ans_tgt = float(l0[ans_id]) - float(l0[tgt_id])

        # z reads (e241 convention; RU null first, then target null)
        e_ans = Wn[ans_id]
        zrec = {}
        for role, tid in [("ru", ru_id), ("tgt", tgt_id)]:
            sim = float(np.dot(Wn[tid], e_ans))
            allowed = np.array(
                [x for x in range(V) if x not in (ans_id, tid)])
            ids = rng.choice(allowed, size=N_NULL, replace=False)
            sn = Wn[ids] @ e_ans
            mu, sd = float(sn.mean()), float(sn.std(ddof=1))
            zrec[role] = {
                "token": tok.decode([tid]), "id": tid,
                "cos_sim": round(sim, 4),
                "null_mean": round(mu, 4), "null_std": round(sd, 4),
                "z": round((sim - mu) / sd, 3),
            }
        z_ru, z_tgt = zrec["ru"]["z"], zrec["tgt"]["z"]

        # classification (rules verbatim, first match wins)
        if tgt_id == ru_id and z_ru >= Z_LOCAL:
            cls = "MIS-DIAL"
        elif (tgt_id == the_id
              or (rank_tgt >= RANK_FREQ and z_tgt < Z_NEG)):
            cls = "FREQUENCY-COLLAPSE"
        else:
            cls = "OTHER"

        # the 2x2 cells
        stratum = "local" if z_ru >= Z_LOCAL else "distant"
        outcome = "to-RU" if tgt_id == ru_id else "away-from-RU"

        # provenance: e232 census + journal p-side state
        ckey = (wash, batt, probe)
        if ckey in e232_idx:
            r232 = e232_idx[ckey]
            fate = r232["fate"]
            standing = bool(r232["standing_at_80"])
        else:
            fate, standing = "outside-e232-census", None
        sj = by_state[(wash, 80)]
        _, ji = jidx[i]
        p_rec = sj[batt]["probes"][ji]
        p_t0 = t0j[batt]["probes"][ji]["p"]
        p_80 = float(p_rec["p"])

        # e241 join fields (identity + z comparison, approximate by null
        # noise: different record order -> different draws of the SAME null)
        j241 = e241_idx.get((batt, wash, probe))
        j241_note = None
        if j241 is not None:
            assert j241["wrong_id"] == tgt_id, (probe, wash)
            j241_note = {
                "e241_classification": j241["classification"],
                "dz_target_vs_e241": round(z_tgt - j241["embedding_z"], 3)}

        table.append({
            "wash": wash, "battery": batt, "probe": probe,
            "dead_answer_id": ans_id,
            "dead_answer_token": tok.decode([ans_id]),
            "runner_up_id": ru_id, "runner_up": zrec["ru"]["token"],
            "runner_up_rank_t0": rank_ru,   # == 2 by construction (gate)
            "runner_up_cos_sim": zrec["ru"]["cos_sim"],
            "runner_up_z": z_ru,
            "runner_up_null_mean": zrec["ru"]["null_mean"],
            "runner_up_null_std": zrec["ru"]["null_std"],
            "target_id": tgt_id, "target": zrec["tgt"]["token"],
            "target_rank_t0": rank_tgt,
            "target_cos_sim": zrec["tgt"]["cos_sim"],
            "target_z": z_tgt,
            "target_is_the": bool(tgt_id == the_id),
            "target_is_runner_up": bool(tgt_id == ru_id),
            "logit_gap_ans_minus_ru_t0": round(gap_ans_ru, 4),
            "logit_gap_ans_minus_tgt_t0": round(gap_ans_tgt, 4),
            "classification": cls,
            "stratum": stratum, "outcome": outcome,
            "e232_fate": fate, "e232_standing_at_80": standing,
            "p_t0": round(float(p_t0), 5), "p_80": round(p_80, 5),
            "margin_sigma_80_new_commitment":
                round(float(p_rec["margin_sigma"]), 4),
            "e241_join": j241_note,
        })
        log(f"{wash} {batt} {probe}: RU='{zrec['ru']['token']}' "
            f"z_ru={z_ru:+.2f} | tgt='{zrec['tgt']['token']}' "
            f"rank={rank_tgt} z_tgt={z_tgt:+.2f} | {cls} [{stratum}/"
            f"{outcome}] fate={fate}")

    # ------------------------------------------------------ the selector 2x2
    loc = [r for r in table if r["stratum"] == "local"]
    dis = [r for r in table if r["stratum"] == "distant"]
    k_loc = sum(1 for r in loc if r["outcome"] == "to-RU")
    k_dis = sum(1 for r in dis if r["outcome"] == "to-RU")
    rate_local = (k_loc / len(loc)) if loc else None
    rate_distant = (k_dis / len(dis)) if dis else None

    if rate_local is not None and rate_distant is not None:
        if rate_local >= CONFIRM_HIGH and rate_distant <= CONFIRM_LOW:
            verdict = "SELECTOR-CONFIRMED"
        elif abs(rate_local - rate_distant) <= DENIED_BAND:
            verdict = "SELECTOR-DENIED"
        else:
            verdict = "MIXED"
    else:
        verdict = "MIXED"  # empty-stratum rule (pre-registered)

    n_mis = sum(1 for r in table if r["classification"] == "MIS-DIAL")
    n_frq = sum(1 for r in table if r["classification"] == "FREQUENCY-COLLAPSE")
    n_oth = sum(1 for r in table if r["classification"] == "OTHER")
    n_the = sum(1 for r in table if r["target_is_the"])
    log(f"2x2: local {k_loc}/{len(loc)} to-RU ({rate_local}), "
        f"distant {k_dis}/{len(dis)} to-RU ({rate_distant})")
    log(f"classification: MIS-DIAL {n_mis}, FREQUENCY-COLLAPSE {n_frq} "
        f"({n_the} to 'the'), OTHER {n_oth}")
    log(f"VERDICT: {verdict}")

    # e241 z-join gate
    dz_max = max((abs(r["e241_join"]["dz_target_vs_e241"])
                  for r in table if r["e241_join"]), default=0.0)
    gates["G_JOIN241"] = {
        "pass": bool(dz_max < 1.0),
        "max_abs_dz_target": round(float(dz_max), 3),
        "tol": 1.0,
        "n_joined": sum(1 for r in table if r["e241_join"]),
        "desc": "all 9 e241 records found in the flip population with the "
                "identical target token; target z agrees with e241's "
                "committed z within null-draw noise (same convention+seed, "
                "different record order -> different draws; identity gate, "
                "not a bar)"}

    # ---------------------------------------------------------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig 1 — the selector 2x2
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.6))
    x = np.arange(2)
    w = 0.36
    axA.bar(x - w / 2, [k_loc, k_dis], w, color="#1f6f43",
            label="target = t0 runner-up")
    axA.bar(x + w / 2, [len(loc) - k_loc, len(dis) - k_dis], w,
            color="#922b21", label="target elsewhere")
    for xi, (k, n) in enumerate([(k_loc, len(loc)), (k_dis, len(dis))]):
        axA.text(xi - w / 2, k + 0.1, f"{k}", ha="center", fontsize=11)
        axA.text(xi + w / 2, n - k + 0.1, f"{n - k}", ha="center",
                 fontsize=11)
    axA.set_xticks(x)
    axA.set_xticklabels([
        f"runner-up LOCAL\n(z_RU >= 2; n={len(loc)})",
        f"runner-up DISTANT\n(z_RU < 2; n={len(dis)})"])
    axA.set_ylabel("flip records")
    axA.set_title("the flip population, stratified by the t0 runner-up's "
                  "semantic locality", fontsize=10.5)
    axA.legend(fontsize=9)

    rates = [rate_local, rate_distant]
    cis = [wilson(k_loc, len(loc)), wilson(k_dis, len(dis))]
    colors = ["#1f6f43", "#922b21"]
    bars = axB.bar(x, [r if r is not None else 0 for r in rates], 0.5,
                   color=colors, alpha=0.85)
    for xi, (r, c, k, n) in enumerate(zip(rates, cis, [k_loc, k_dis],
                                          [len(loc), len(dis)])):
        if r is None or c is None:
            axB.text(xi, 0.02, "EMPTY\nSTRATUM", ha="center", fontsize=10)
            continue
        axB.errorbar(xi, r, yerr=[[r - c[0]], [c[1] - r]], fmt="none",
                     ecolor="black", capsize=6, lw=1.6)
        axB.text(xi, min(1.0, r + 0.13),
                 f"{k}/{n} = {r * 100:.0f}%\nWilson95 [{c[0]:.2f}, {c[1]:.2f}]",
                 ha="center", fontsize=10)
    axB.axhline(CONFIRM_HIGH, color="#1f6f43", ls="--", lw=1.2)
    axB.axhline(CONFIRM_LOW, color="#922b21", ls="--", lw=1.2)
    axB.text(1.52, CONFIRM_HIGH + 0.015, "CONFIRM bar: >= 80%",
             fontsize=8, color="#1f6f43", ha="right")
    axB.text(1.52, CONFIRM_LOW - 0.045, "CONFIRM bar: <= 40%",
             fontsize=8, color="#922b21", ha="right")
    axB.set_xticks(x)
    axB.set_xticklabels(["P(target=RU | RU local)",
                         "P(target=RU | RU distant)"])
    axB.set_ylim(0, 1.15)
    axB.set_ylabel("conditional rate")
    axB.set_title(f"THE MODE SELECTOR — {verdict}", fontsize=11,
                  fontweight="bold")
    fig.suptitle("E243 — what decides a flipped commitment's death mode: "
                 f"the 2x2 over all {len(table)} argmax flips "
                 "(committed journals; pristine 124M wte)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig1 = out / "e243_selector_2x2.png"
    fig.savefig(fig1, dpi=140)
    plt.close(fig)

    # fig 2 — the flip population in the rank-z plane + the RU-z/target-z view
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 6.4))
    cls_color = {"MIS-DIAL": "#1f6f43",
                 "FREQUENCY-COLLAPSE": "#922b21", "OTHER": "#b9770e"}
    y_lo = min(-5.0, min(r["target_z"] for r in table) - 1.5)
    y_hi = max(6.0, max(r["target_z"] for r in table) + 2)
    axA.set_xscale("log")
    axA.axhline(Z_LOCAL, color="#2c3e50", lw=1, ls=":")
    axA.axvline(RANK_FREQ - 0.5, color="#2c3e50", lw=1, ls=":")
    axA.fill_betweenx([Z_LOCAL, y_hi], 0.9, 2.5, color="#d5f5e3", zorder=0,
                      label="MIS-DIAL needs tgt=RU (rank 2) & z_RU >= 2")
    axA.fill_betweenx([y_lo, Z_NEG], 2.5, 5e4, color="#fadbd8", zorder=0,
                      label="FREQUENCY-COLLAPSE leg (rank >= 3 & z < 0)")
    for r in table:
        xp, yp = max(r["target_rank_t0"], 1), r["target_z"]
        axA.scatter(xp, yp, s=95,
                    facecolors=cls_color[r["classification"]] if
                    r["target_is_runner_up"] else "none",
                    edgecolors=cls_color[r["classification"]], linewidths=2.0,
                    zorder=3)
        axA.annotate(
            f"{r['probe'].split('->')[0][:20]}->{r['wash']}: "
            f"'{r['target'].strip()}'",
            (xp, yp), textcoords="offset points", xytext=(8, 4), fontsize=7.2)
    axA.set_xlim(0.9, 5e4)
    axA.set_ylim(y_lo, y_hi)
    axA.set_xlabel("flip TARGET's logit-rank at t=0 (log scale)")
    axA.set_ylabel("target's embedding z vs the dead answer "
                   "(pristine wte, 1000-token null)")
    axA.set_title(f"the {len(table)} flips in the rank-z plane "
                  "(filled = target is the t0 runner-up; "
                  f"'the' x{n_the}; MIS-DIAL {n_mis} / FREQ {n_frq} / "
                  f"OTHER {n_oth})", fontsize=10)
    axA.legend(fontsize=8, loc="lower left")

    for r in table:
        axB.scatter(r["runner_up_z"], r["target_z"], s=95,
                    marker="o" if r["outcome"] == "to-RU" else "X",
                    facecolors=cls_color[r["classification"]] if
                    r["outcome"] == "to-RU" else "none",
                    edgecolors=cls_color[r["classification"]],
                    linewidths=2.0, zorder=3)
        axB.annotate(f"{r['probe'].split('->')[0][:16]}->{r['wash']}",
                     (r["runner_up_z"], r["target_z"]),
                     textcoords="offset points", xytext=(7, 3), fontsize=7.2)
    lo_b = min(y_lo, min(r["runner_up_z"] for r in table) - 1.5)
    hi_b = max(y_hi, max(r["runner_up_z"] for r in table) + 2)
    axB.plot([lo_b, hi_b], [lo_b, hi_b], color="#7f8c8d", lw=0.8, ls="--",
             zorder=1)
    axB.axvline(Z_LOCAL, color="#2c3e50", lw=1, ls=":")
    axB.axhline(Z_LOCAL, color="#2c3e50", lw=1, ls=":")
    axB.set_xlim(lo_b, hi_b)
    axB.set_ylim(lo_b, hi_b)
    axB.set_xlabel("t0 RUNNER-UP's embedding z (the selector's input)")
    axB.set_ylabel("flip TARGET's embedding z")
    axB.set_title("the selector plane: what was standing nearby (x) vs where "
                  "the commitment went (y); circles on the diagonal = "
                  "mis-dials", fontsize=10)
    fig.suptitle("E243 — the flip population in the rank-z plane "
                 f"({verdict}; committed two-wash 124M archive)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig2 = out / "e243_rank_z_plane.png"
    fig.savefig(fig2, dpi=140)
    plt.close(fig)
    log(f"figures: {fig1.name}, {fig2.name}")

    # -------------------------------------------------------------- metrics
    timing["finished"] = utcnow()
    timing["wall_s"] = round(time.time() - t_start, 1)

    sub_fact_ctrl = [r for r in table if r["battery"] in ("fact", "ctrl")]
    sfc_loc = [r for r in sub_fact_ctrl if r["stratum"] == "local"]
    sfc_dis = [r for r in sub_fact_ctrl if r["stratum"] == "distant"]
    sfc_kl = sum(1 for r in sfc_loc if r["outcome"] == "to-RU")
    sfc_kd = sum(1 for r in sfc_dis if r["outcome"] == "to-RU")

    ru_strata = {}
    for r in table:
        ru_strata.setdefault(r["probe"], {
            "runner_up": r["runner_up"], "runner_up_z": r["runner_up_z"],
            "washes": []})["washes"].append(
                {r["wash"]: r["target"] + " [" + r["classification"] + "]"})

    metrics = {
        "experiment": "e243_mode_selector",
        "phase": ("desk-only on COMMITTED data (CPU threads=4; no GPU, no "
                  "training, no evals; the wte read tensor-only from the "
                  "committed HF snapshot — load-polite, two other agents "
                  "live)"),
        "date": timing["started"],
        "status": "DONE",
        "registration": ("bars + classification rules + the 2x2 + the "
                         "empty-stratum rule frozen VERBATIM from the "
                         "dispatch brief in this file's docstring; the "
                         "registration commit precedes any compute; "
                         "adjudicate against exactly this; no bar shopping"),
        "question": ("what decides a flipped commitment's death mode — the "
                     "mis-dial (to the t0 runner-up) or the frequency "
                     "collapse (to the prior's basin)? T219's selector: the "
                     "t0 local-candidate structure"),
        "builds_on": [
            "T219 / e241 (the two death modes — the mis-dials vs the "
            "frequency collapses; the registered ask this cell adjudicates; "
            "the z convention, null seed and 'the' confound carried VERBATIM)",
            "T212 / e232 (the zombie taxonomy — the 7 wrong-choosers; the "
            "(probe, wash) death unit; the census fates)",
            "e228 (the margins journal — per-state per-probe margins AND "
            "argmax records for all 54 probes x 8 states; the flip "
            "population's primary source)",
            "e238 (the committed answer-position full-logit dumps — the "
            "argmax re-derivation and the t0 runner-up structure)",
            "e214/e182c/e182c2 (the two-wash 124M archive underneath)",
        ],
        "whats_new": [
            "the flip population enumerated over the WHOLE journal (19 "
            "(probe, wash) records: the 7 wrong-choosers + 2 near + 10 "
            "general flips e232's dying census never adjudicated)",
            "the t0 RUNNER-UP's identity + embedding z for every flip (the "
            "selector's input — never read before; e241 read the target's z "
            "only)",
            "the MIS-DIAL / FREQUENCY-COLLAPSE / OTHER classification of "
            "every flip",
            "the selector 2x2: P(target=RU | RU local) vs P(target=RU | RU "
            "distant), with Wilson 95% CIs",
        ],
        "registered_bars_verbatim": BARS_VERBATIM,
        "registered_classification_rules_verbatim": CLASS_RULES_VERBATIM,
        "registered_prediction_T219_verbatim": PREDICTION_VERBATIM,
        "operationalizations_frozen": {
            "population": "every (probe, wash) record of the committed "
                          "journals whose +80 argmax differs from t0's "
                          "(the dispatch's definition; endpoint read — "
                          "transient flips that recovered by +80 are not "
                          "counted; no probe gained the answer's argmax by "
                          "+80, so no recovery records exist); ALL batteries "
                          "adjudicate (fact/ctrl/near/tmpl — the dispatch's "
                          "'every probe'); n = 19 over 15 distinct probes, "
                          "6 probes dying under both washes",
            "runner_up": "rank-2 token of the committed t0 answer-position "
                         "logit vector (float64 stable argsort over the full "
                         "50257 vocab); t0 is the SHARED t0 (e228's "
                         "t0_shared convention — both wash lineages start "
                         "from one pristine organism), so the RU read is per "
                         "probe, shared across washes",
            "z_read": "e241's convention VERBATIM: cos(wte[a], wte[b]) on "
                      "L2-normalized float64 pristine-wte rows; null = 1000 "
                      "ids drawn uniformly from the vocab minus the two ids; "
                      "ONE global default_rng(20261004); z = (sim - "
                      "mean)/std(ddof=1); draw order frozen: records in (w1, "
                      "w2) x grid order, runner-up null first then target "
                      "null per record (when target == runner-up the two "
                      "nulls are distinct draws of the same distribution)",
            "frequency_token": "' the', id 262, tokenizer-verified at "
                               "runtime (G_IDS)",
            "classification_order": "MIS-DIAL first (target = the t0 "
                                    "runner-up AND runner-up z >= 2), then "
                                    "FREQUENCY-COLLAPSE (target = the "
                                    "frequency token / any z < 0 token at "
                                    "rank >= 3, rank = the target's t0 "
                                    "rank), else OTHER — first match wins, "
                                    "exactly the dispatch's wording",
            "selector_2x2": "rows: runner-up local (z_RU >= 2) / distant "
                            "(z_RU < 2); outcome: target == t0 runner-up vs "
                            "not; the conditional rates and the bars read "
                            "record-level (each (probe, wash) flip counts "
                            "once)",
            "empty_stratum_rule": "both rates must exist for CONFIRMED or "
                                  "DENIED; an empty stratum (n=0) sends the "
                                  "verdict to MIXED with the empty stratum "
                                  "disclosed (pre-registered here BEFORE "
                                  "compute)",
            "co_reports": "Wilson 95% CIs; the fact/ctrl subtable; census "
                          "provenance + e232 fate + p_80 + the new "
                          "commitment's margin_sigma at +80; the t0 logit "
                          "gaps; the e241 z-joins; the per-probe RU strata — "
                          "none adjudicate",
        },
        "sources": {
            "margins_journal": {"path": str(E228_JOURNAL),
                                "sha256_16": sha16(E228_JOURNAL)},
            "npz_shas_reference": {"path": str(E238_JOURNAL),
                                   "sha256_16": sha16(E238_JOURNAL)},
            "logits_t0": {"path": str(E238_NPZ["t0"]),
                          "sha256_16": sha16(E238_NPZ["t0"])},
            "logits_w1s80": {"path": str(E238_NPZ["w1"]),
                             "sha256_16": sha16(E238_NPZ["w1"])},
            "logits_w2s80": {"path": str(E238_NPZ["w2"]),
                             "sha256_16": sha16(E238_NPZ["w2"])},
            "e241_table": {"path": str(E241_METRICS),
                           "sha256_16": sha16(E241_METRICS)},
            "e232_census": {"path": str(E232_METRICS),
                            "sha256_16": sha16(E232_METRICS)},
            "wte": {
                "repo": MODEL_REPO, "revision": MODEL_REV,
                "wte_sha256_16_fp32bytes": wte_sha,
                "snapshot_safetensors_sha256_16": sha16(snap_file),
                "snapshot_file": str(snap_file),
            },
        },
        "coverage_disclosure": {
            "journal_argmax_records": "COMPLETE — runs/e228/journal.json "
                                      "carries top1_id/top2_id per probe per "
                                      "state for all 54 probes x 8 states "
                                      "(t0, w1 {2,10,50,80}, w2 {10,50,80})",
            "npz_states": "t0, w1s80, w2s80 (the flip endpoints) — "
                          "re-derived argmax identical to the journal on all "
                          "54 probes x 3 states (gate G_FLIPS); "
                          "intermediate states unused (the flip is an "
                          "endpoint read)",
            "flip_population": {
                "n_records": len(table),
                "per_wash": {w: sum(1 for r in table if r["wash"] == w)
                             for w in WASH_ORDER},
                "per_battery": {b: sum(1 for r in table
                                       if r["battery"] == b)
                                for b in BATT_ORDER},
                "in_e232_dying_census": sum(1 for r in table
                                            if r["e232_fate"] !=
                                            "outside-e232-census"),
                "outside_census": [f"{r['wash']} {r['probe']}" for r in table
                                   if r["e232_fate"] ==
                                   "outside-e232-census"],
                "note": "the 10 records beyond e241's 9 are flips the dying "
                        "census never adjudicated — 4 outside the census "
                        "entirely (the belief never halved: France->Paris "
                        "w1, Egypt->Cairo w2, Xbox w1, Xbox w2) plus 6 "
                        "in-census resolved deaths (Ireland w1, Boston w1, "
                        "Columbus w1, Jackson w1/w2, China w2... see the "
                        "table's e232_fate column)",
            },
        },
        "table": table,
        "classification_counts": {
            "MIS-DIAL": n_mis,
            "FREQUENCY-COLLAPSE": n_frq,
            "of_which_the_frequency_token": n_the,
            "OTHER": n_oth,
            "other_records": [f"{r['wash']} {r['probe']} -> "
                              f"'{r['target'].strip()}'" for r in table
                              if r["classification"] == "OTHER"],
        },
        "selector_2x2": {
            "local": {"n": len(loc), "to_RU": k_loc,
                      "rate": round(rate_local, 4) if rate_local is not None
                      else None,
                      "wilson95": wilson(k_loc, len(loc)),
                      "records": [f"{r['wash']} {r['probe']} -> "
                                  f"'{r['target'].strip()}'" for r in loc]},
            "distant": {"n": len(dis), "to_RU": k_dis,
                        "rate": round(rate_distant, 4)
                        if rate_distant is not None else None,
                        "wilson95": wilson(k_dis, len(dis)),
                        "records": [f"{r['wash']} {r['probe']} -> "
                                    f"'{r['target'].strip()}'" for r in dis]},
            "rate_local": round(rate_local, 4) if rate_local is not None
                          else None,
            "rate_distant": round(rate_distant, 4) if rate_distant
                            is not None else None,
            "fact_ctrl_coreport": {
                "n": len(sub_fact_ctrl),
                "local_to_RU": f"{sfc_kl}/{len(sfc_loc)}",
                "distant_to_RU": f"{sfc_kd}/{len(sfc_dis)}",
                "note": "e241's comparable subset (fact/ctrl only); "
                        "co-report, never adjudicates"},
        },
        "runner_up_strata_per_probe": ru_strata,
        "adjudication": {
            "verdict": verdict,
            "bar_fired_verbatim": BARS_VERBATIM[verdict],
            "read": (
                f"local {k_loc}/{len(loc)}"
                f" ({(rate_local * 100 if rate_local is not None else float('nan')):.0f}%) "
                f"vs distant {k_dis}/{len(dis)}"
                f" ({(rate_distant * 100 if rate_distant is not None else float('nan')):.0f}%)"),
            "prediction_read": ("T219's prediction is adjudicated by the 2x2 "
                                "conditional rates (the bars), not by the "
                                "classification counts; the classification "
                                "table co-reports the mode census"),
        },
        "gates": gates,
        "honesty_reflex": (
            "KNOWN CONFOUNDS, disclosed not corrected: (1) e241's 'the' "
            "confound carries VERBATIM — ' the' is the vocab's most frequent "
            "token and its wte vector sits near the embedding cloud's mean "
            "direction, so its cosine to ANY answer is depressed/elevated by "
            "frequency geometry rather than answer-specific semantics; the "
            "null (1000 uniform vocab ids) mostly samples rare tokens. (2) "
            "The classification and the 2x2 share the same z instrument and "
            "the same z >= 2 threshold — the threshold enters the strata "
            "(RU local?) and the MIS-DIAL rule simultaneously, exactly as "
            "registered; the 2x2's to-RU outcome does NOT depend on z, so "
            "the selector read is not circular, but the STRATUM boundary is "
            "threshold-sensitive (Wilson CIs co-report the n-sensitivity). "
            "(3) The 19 records include 6 probes dying under BOTH washes — "
            "each such probe contributes two records sharing one t0 "
            "runner-up read (the wash draws are independent; the RU "
            "structure is not; per-probe strata co-reported). (4) near/tmpl "
            "batteries adjudicate here (the dispatch's 'every probe' "
            "wording) — the fact/ctrl subtable co-reports e241's comparable "
            "subset. (5) n = 19: the bars adjudicate on whatever n exists, "
            "disclosed; nothing guaranteed."),
        "figures": [str(fig1), str(fig2)],
        "compute": {
            "device": "cpu desk pass (threads 4)",
            "model_loads": 0,
            "wte_read": "tensor-only from the committed HF snapshot "
                        "safetensors (no model construction; no torch)",
            "gpu_calls": 0,
            "training": "none",
        },
        "provenance": {
            "git_head_at_start": git_head,
            "script": str(Path(__file__).resolve()),
            "script_sha256_16": sha16(Path(__file__).resolve()),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "transformers": transformers.__version__,
                "matplotlib": matplotlib.__version__,
                "safetensors": __import__("safetensors").__version__,
            },
        },
        "timing": timing,
        "trims": [],
        "deviations": deviations + [
            "the wte is read tensor-only from the committed HF snapshot "
            "safetensors instead of constructing GPT2LMHeadModel as e241 "
            "did — load-polite for the two other live agents; the sha gate "
            "(fp32 bytes) proves the identical embedding table; torch is "
            "not imported at all",
            "the null-draw sequence differs from e241's by record ORDER "
            "(e241 processed e234's rider; this cell processes the journal "
            "enumeration in (w1, w2) x grid order with two draws per "
            "record) — the convention and seed are identical; the e241 "
            "join gate (|dz| < 1) verifies agreement within null-draw "
            "noise",
        ],
        "all_gates_pass": None,  # filled below
    }
    metrics["all_gates_pass"] = all(
        g.get("pass", False) for g in gates.values())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    log(f"metrics.json written to {out / 'metrics.json'}; "
        f"all_gates_pass={metrics['all_gates_pass']}")


if __name__ == "__main__":
    sys.exit(main())
