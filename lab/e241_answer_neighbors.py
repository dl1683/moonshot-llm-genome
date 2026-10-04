"""E241 — THE ANSWER-NEIGHBOR QUANTIFICATION (T217's registered ask; desk-only,
CPU threads=4, COMMITTED data + one read-only load of the pristine organism's
wte; no training, no evals, no GPU).

WHY: T212/e232 found the 7 wrong-choice zombies (standing margins on a WRONG
argmax at +80); e234's rider killed H-i RE-TEACHING (corpus-mode 1/7 — CITED,
not re-derived here) and T217 named the texture: the new commitments are the
dead answers' own semantic neighbors ('dollars' for US->dollar, ' Music' for
iTunes, ' Augusta' for Georgia->Atlanta, ' the' x4) — "THE ZOMBIE'S
COMMITMENT DOES NOT DIE AND IS NOT REWRITTEN — IT WANDERS NEXT DOOR." This
cell quantifies that texture on committed data.

THE QUESTION (verbatim from the dispatch): is the wrong-choosers' collapse
LOCAL (into the dead answer's semantic neighborhood) or GLOBAL (scattered)?

T217's REGISTERED PREDICTION (verbatim): "the neighbors sit in the dead
answer's top-1% semantic neighborhood (a LOCAL collapse, not a global one —
the commitment keeps its ADDRESS but loses its NUMBER)."

THE CELL (frozen from the dispatch, per record = one (probe, wash) death as
in e232's convention — 7 adjudicated records over 5 distinct probes; the 2
Georgia->Atlanta near records are co-report only, counted separately, as in
e232/e234):
  (1) the wrong token's logit-rank at t=0 (was it already the runner-up?)
      — rank among all 50257 entries of the COMMITTED t=0 answer-position
      logit vector (runs/e238/logits_t0.npz, sha-gated); rank 1 = argmax.
  (2) the embedding-space cos sim — wte of the PRISTINE shared 124M
      (openai-community/gpt2 @ 607a30d783dfa663caf39e06633721c8d4cfcd7e,
      the archive's t0 organism, loaded read-only on CPU) — wrong-token vs
      dead-answer vector, read against the null distribution of dead-answer
      vs 1000 random vocab tokens: z = (sim - mean(null)) / std(null, ddof=1).
      NULL SPEC (frozen): ONE global numpy default_rng(seed=20261004);
      records processed in e234 rider order; each record draws 1000 ids
      uniformly from the 50257 vocab minus {answer_id, wrong_id}; cosines in
      float64 on L2-normalized wte rows.
  (3) the p-side distance: p(wrong | t=0) vs p(answer | t=0) (softmax in
      float64 of the committed t=0 logits) and the raw logit gap
      l(answer) - l(wrong) — the runner-up gap at the start.
  (4) classification, rules VERBATIM from the dispatch, evaluated in this
      order (first match wins):
        NEXT-DOOR       — embedding z >= 2 AND logit-rank <= 3 at t=0
        RUNNER-UP-PURE  — rank 2 only, embedding z < 2
        DISTANT         — neither

CO-REPORTS (never adjudicate): the full-vocab embedding percentile of the
wrong token in the dead answer's neighborhood (1 - (rank-1)/(N-1), the
direct "top-1%" read); the dead answer's top-5 wte neighbors (texture);
the wrong token's null percentile; everything on the 2 near records.

REGISTERED BARS (frozen VERBATIM from the dispatch brief BEFORE any compute;
the registration commit of this file precedes any compute; adjudicate
against exactly this; no bar shopping; adjudication counts the 7 fact/ctrl
records only):

  LOCAL-COLLAPSE — ">= 5/7 NEXT-DOOR — the commitment collapses into the
  dead answer's own neighborhood; T217's prediction confirmed; the zombie's
  new commitment is a mis-dial, not a re-write"

  GLOBAL-SCATTER — ">= 5/7 DISTANT — the flips are unrelated to the dead
  answers; the neighborhood texture was anecdotal"

  MIXED — "the classification table verbatim, no narrative inflation"

INPUTS (all COMMITTED, sha-gated at runtime, never re-derived):
  runs/e234/metrics.json  — the rider: the 9 (probe, wash, dead answer,
                            +80 argmax token id/string) identities; the
                            1/7 corpus-mode count CITED from it.
  runs/e238/logits_t0.npz + logits_w1s80.npz + logits_w2s80.npz — the
                            committed answer-position full logits (t0 for
                            reads; +80 dumps ONLY for the argmax-consistency
                            gate re-reading the committed flip).
  runs/e232/metrics.json  — the wrong-choosers' standing-zombie records
                            (p_t0, margin_80) for join gates.
  runs/e238/journal.json  — the committed npz sha256_16 records (the gate's
                            reference).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "lab"
RUNS = REPO / "runs"
RD = RUNS / "e241"

E234_METRICS = RUNS / "e234" / "metrics.json"
E232_METRICS = RUNS / "e232" / "metrics.json"
E238_JOURNAL = RUNS / "e238" / "journal.json"
E238_NPZ = {
    "t0": RUNS / "e238" / "logits_t0.npz",
    "w1s80": RUNS / "e238" / "logits_w1s80.npz",
    "w2s80": RUNS / "e238" / "logits_w2s80.npz",
}

MODEL_REPO = "openai-community/gpt2"
MODEL_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
NULL_SEED = 20261004
N_NULL = 1000
Z_BAR = 2.0
RANK_BAR = 3

BARS_VERBATIM = {
    "LOCAL-COLLAPSE": ">= 5/7 NEXT-DOOR — the commitment collapses into the dead answer's own neighborhood; T217's prediction confirmed; the zombie's new commitment is a mis-dial, not a re-write",
    "GLOBAL-SCATTER": ">= 5/7 DISTANT — the flips are unrelated to the dead answers; the neighborhood texture was anecdotal",
    "MIXED": "the classification table verbatim, no narrative inflation",
}
CLASS_RULES_VERBATIM = {
    "NEXT-DOOR": "embedding z >= 2 AND logit-rank <= 3 at t=0",
    "RUNNER-UP-PURE": "rank 2 only, embedding z < 2",
    "DISTANT": "neither",
}
PREDICTION_VERBATIM = ("the neighbors sit in the dead answer's top-1% semantic "
                       "neighborhood (a LOCAL collapse, not a global one — the "
                       "commitment keeps its ADDRESS but loses its NUMBER)")


def log(msg: str) -> None:
    print(msg, flush=True)


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def softmax64(l: np.ndarray) -> np.ndarray:
    x = l.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


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
        import subprocess
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO),
            capture_output=True, text=True).stdout.strip()
    except Exception:
        pass

    gates: dict[str, dict] = {}
    deviations: list[str] = []

    # ---------------------------------------------------------------- inputs
    e234 = json.loads(E234_METRICS.read_text())
    e232 = json.loads(E232_METRICS.read_text())
    e238j = json.loads(E238_JOURNAL.read_text())
    rider = e234["rider"]["targets"]
    log(f"rider records: {len(rider)} "
        f"(adjudicated {sum(1 for r in rider if r['battery'] in ('fact','ctrl'))}, "
        f"near co-report {sum(1 for r in rider if r['battery'] == 'near')}); "
        f"committed corpus-mode count (CITED): "
        f"bigram {e234['rider']['h_i_count_bigram']}/7, "
        f"trigram {e234['rider']['h_i_count_trigram']}/7")

    # sha gates on the committed npz dumps vs e238's journal
    for k, p in E238_NPZ.items():
        got, want = sha16(p), e238j["logit_states"][k]["npz_sha256_16"]
        gates[f"G_SHA_{k}"] = {
            "got": got, "committed": want, "pass": got == want,
            "desc": f"{p.name} sha256_16 matches e238 journal"}
        log(f"gate G_SHA_{k}: {got} vs {want} -> {got == want}")

    d_t0 = np.load(E238_NPZ["t0"], allow_pickle=True)
    d_w1 = np.load(E238_NPZ["w1s80"], allow_pickle=True)
    d_w2 = np.load(E238_NPZ["w2s80"], allow_pickle=True)
    names = [str(x) for x in d_t0["names"]]
    batts = [str(x) for x in d_t0["battery"]]
    ans_ids = d_t0["ans_ids"].astype(int)
    L_t0 = d_t0["logits"]
    L_80 = {"w1": d_w1["logits"], "w2": d_w2["logits"]}
    # the npz row index is shared across states (same 54-probe grid)
    ok_grid = (all(str(a) == str(b) for a, b in zip(d_w1["names"], names))
               and all(str(a) == str(b) for a, b in zip(d_w2["names"], names)))
    gates["G_GRID"] = {
        "pass": bool(ok_grid),
        "desc": "w1s80/w2s80 dumps share the t0 dump's 54-row probe grid"}

    # ------------------------------------------------------------- organism
    import torch
    torch.set_num_threads(4)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import transformers
    tok = GPT2TokenizerFast.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    net = GPT2LMHeadModel.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    net.eval()
    wte = net.transformer.wte.weight.detach().to(torch.float64).numpy().copy()
    n_vocab, n_embd = wte.shape
    V = int(n_vocab)
    gates["G_WTE"] = {
        "pass": bool(n_vocab == 50257 and n_embd == 768),
        "shape": [int(n_vocab), int(n_embd)],
        "repo": MODEL_REPO, "revision": MODEL_REV,
        "desc": "pristine shared-124M wte (the archive's t0 organism), read-only"}
    wte_sha = hashlib.sha256(
        np.ascontiguousarray(wte.astype(np.float32)).tobytes()).hexdigest()[:16]
    # the HF snapshot file, for provenance
    snap_sha, snap_file = None, None
    try:
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        for cand in (hub / f"models--{MODEL_REPO.replace('/', '--')}"
                     / "snapshots" / MODEL_REV).glob("*safetensors"):
            snap_sha = sha16(cand)
            snap_file = str(cand)
            break
    except Exception:
        pass
    log(f"wte {n_vocab}x{n_embd} sha16(fp32 bytes) {wte_sha}; "
        f"snapshot safetensors sha16 {snap_sha}")

    Wn = wte / np.linalg.norm(wte, axis=1, keepdims=True)

    # ------------------------------------------------------------- the cell
    rng = np.random.default_rng(NULL_SEED)
    e232_pl = e232["read1_zombie_lag"]["probe_level"]
    table = []
    for r in rider:  # e234 rider order (frozen): 7 fact/ctrl, then 2 near
        probe, wsh, batt = r["fact"], r["wash"], r["battery"]
        wrong_str, wrong_id = r["argmax_token"], int(r["argmax_80_committed"])
        row = next(i for i in range(len(names))
                   if names[i] == probe and batts[i] == batt)
        ans_id = int(ans_ids[row])
        l0 = L_t0[row]

        # --- identity gates
        enc = tok.encode(wrong_str)
        tok_ok = len(enc) == 1 and enc[0] == wrong_id
        ans_dec = tok.decode([ans_id])
        ans_in_name = ans_dec.strip().lower() in probe.split("->")[-1].strip().lower()
        # --- decision-layer reads at t=0 (committed logits)
        order = np.argsort(-l0.astype(np.float64), kind="stable")
        rank_of = {int(t): i + 1 for i, t in enumerate(order)}
        rank_wrong = rank_of[wrong_id]
        rank_ans = rank_of[ans_id]
        arg80 = int(np.argmax(L_80[wsh][row].astype(np.float64)))
        # --- p-side reads at t=0
        p64 = softmax64(l0)
        p_ans, p_wrong = float(p64[ans_id]), float(p64[wrong_id])
        gap = float(l0[ans_id]) - float(l0[wrong_id])
        # --- embedding reads (pristine wte)
        e_ans = Wn[ans_id]
        sim = float(np.dot(Wn[wrong_id], e_ans))
        allowed = np.array([i for i in range(V) if i not in (ans_id, wrong_id)])
        null_ids = rng.choice(allowed, size=N_NULL, replace=False)
        sims_null = Wn[null_ids] @ e_ans
        mu, sd = float(sims_null.mean()), float(sims_null.std(ddof=1))
        z = (sim - mu) / sd
        pct_null = float((sims_null < sim).mean())
        sims_all = Wn @ e_ans
        emb_rank_wrong = int(np.sum(sims_all > sims_all[wrong_id])) + 1
        emb_pct = 100.0 * (1.0 - (emb_rank_wrong - 1) / (V - 1))
        top5_idx = np.argsort(-sims_all, kind="stable")[:5]
        top5 = [(tok.decode([int(i)]), round(float(sims_all[i]), 4))
                for i in top5_idx]

        # --- classification (rules verbatim, first match wins)
        if z >= Z_BAR and rank_wrong <= RANK_BAR:
            cls = "NEXT-DOOR"
        elif rank_wrong == 2 and z < Z_BAR:
            cls = "RUNNER-UP-PURE"
        else:
            cls = "DISTANT"

        # --- join gates vs e232's committed wrong-chooser records
        e2rec = next(x for x in e232_pl[batt][wsh] if x["probe"] == probe)
        dp = abs(p_ans - e2rec["p_t0"])
        dm = abs(r["margin_80"] - e2rec["margin_80"])

        table.append({
            "battery": batt, "wash": wsh, "probe": probe,
            "dead_answer_token": ans_dec, "dead_answer_id": ans_id,
            "wrong_token": wrong_str, "wrong_id": wrong_id,
            "adjudicates": batt in ("fact", "ctrl"),
            # (1) decision layer, t=0
            "logit_rank_wrong_t0": rank_wrong,
            "logit_rank_answer_t0": rank_ans,
            "logit_gap_answer_minus_wrong_t0": round(gap, 4),
            # (2) embedding layer (pristine wte)
            "cos_sim_wrong_vs_answer": round(sim, 4),
            "null_mean": round(mu, 4), "null_std": round(sd, 4),
            "embedding_z": round(float(z), 3),
            "null_percentile": round(pct_null, 3),
            "full_vocab_emb_rank_of_wrong": emb_rank_wrong,
            "full_vocab_emb_percentile": round(emb_pct, 2),
            "answer_top5_emb_neighbors": top5,
            # (3) p-side, t=0
            "p_answer_t0": round(p_ans, 6),
            "p_wrong_t0": round(p_wrong, 6),
            "p_ratio_answer_over_wrong": (round(p_ans / p_wrong, 2)
                                          if p_wrong > 0 else None),
            # (4) the classification
            "classification": cls,
            # rider fields carried for the table's readability (CITED)
            "rider_margin_80": r["margin_80"], "rider_p_ans_80": r["p_ans_80"],
            "rider_is_bigram_mode": r["is_bigram_mode"],
            # gates for this record
            "_gates": {
                "tokenizer_id_matches_committed": bool(tok_ok),
                "answer_token_in_probe_name": bool(ans_in_name),
                "answer_is_argmax_t0": rank_ans == 1,
                "argmax80_reread_equals_wrong": arg80 == wrong_id,
                "dp_vs_e232_p_t0": dp, "dm_vs_e232_margin_80": dm,
            },
        })
        log(f"{batt}/{wsh} {probe}: wrong='{wrong_str}' rank_t0={rank_wrong} "
            f"z={z:+.2f} pct_null={pct_null:.3f} emb_rank={emb_rank_wrong} "
            f"p_ans={p_ans:.3f} p_wrong={p_wrong:.3e} -> {cls}")

    # gate roll-up
    # dp tolerance 1e-4 (set AFTER the first run tripped at 1e-6): the npz
    # dumps are float32 round-trips of the live forward — softmax-of-dump vs
    # the committed journal p differs by a SYSTEMATIC ~0.8-1.1e-5 on every
    # record (e238's own gate compared its LIVE forward to e214 and read
    # exactly 0.0). This is an identity gate, not a bar; no read in this cell
    # sits within orders of magnitude of that scale. dm is exact everywhere.
    for rec in table:
        g = rec["_gates"]
        assert g["tokenizer_id_matches_committed"], rec["probe"]
        assert g["answer_is_argmax_t0"], rec["probe"]
        assert g["argmax80_reread_equals_wrong"], rec["probe"]
        assert g["dp_vs_e232_p_t0"] < 1e-4, rec["probe"]
        assert g["dm_vs_e232_margin_80"] == 0.0, rec["probe"]
    gates["G_IDS"] = {
        "pass": all(r["_gates"]["tokenizer_id_matches_committed"]
                    and r["_gates"]["answer_token_in_probe_name"] for r in table),
        "desc": "tokenizer re-encodes every committed wrong-token string to its "
                "committed id; npz ans_ids decode into their probe names"}
    gates["G_ANSWER_T0"] = {
        "pass": all(r["_gates"]["answer_is_argmax_t0"] for r in table),
        "desc": "the dead answer was argmax at t=0 in every record (the "
                "commitment existed before the wash)"}
    gates["G_FLIP80"] = {
        "pass": all(r["_gates"]["argmax80_reread_equals_wrong"] for r in table),
        "desc": "re-read argmax of the committed +80 dump equals the committed "
                "wrong token in all 9 records"}
    gates["G_JOIN232"] = {
        "pass": all(r["_gates"]["dp_vs_e232_p_t0"] < 1e-4
                    and r["_gates"]["dm_vs_e232_margin_80"] == 0.0
                    for r in table),
        "max_dp_observed": max(r["_gates"]["dp_vs_e232_p_t0"] for r in table),
        "tol_dp": 1e-4,
        "desc": "t0 p (float64 softmax of the committed float32 t0 logit "
                "dump) reproduces e232's committed p_t0 within the npz "
                "round-trip scale (observed max ~1.1e-5, systematic — see "
                "deviations); rider margin_80 equals e232's committed "
                "margin_80 exactly on all 9 records"}

    # ------------------------------------------------------- adjudication
    adj = [r for r in table if r["adjudicates"]]
    n_next = sum(1 for r in adj if r["classification"] == "NEXT-DOOR")
    n_run = sum(1 for r in adj if r["classification"] == "RUNNER-UP-PURE")
    n_dis = sum(1 for r in adj if r["classification"] == "DISTANT")
    if n_next >= 5:
        verdict = "LOCAL-COLLAPSE"
    elif n_dis >= 5:
        verdict = "GLOBAL-SCATTER"
    else:
        verdict = "MIXED"
    near_tbl = [r for r in table if not r["adjudicates"]]
    log(f"ADJUDICATION: NEXT-DOOR {n_next}/7, RUNNER-UP-PURE {n_run}/7, "
        f"DISTANT {n_dis}/7 -> {verdict} "
        f"(near co-report: {[r['classification'] for r in near_tbl]})")

    # ---------------------------------------------------------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # re-run the null draws EXACTLY as registered (same global rng, same
    # rider order, deterministic) so the histogram is the honest in-run null
    rng2 = np.random.default_rng(NULL_SEED)
    null_draws = []
    for r in rider:
        row = next(i for i in range(len(names))
                   if names[i] == r["fact"] and batts[i] == r["battery"])
        a_id = int(ans_ids[row])
        w_id = int(r["argmax_80_committed"])
        allowed = np.array([i for i in range(V) if i not in (a_id, w_id)])
        ids = rng2.choice(allowed, size=N_NULL, replace=False)
        null_draws.append(Wn[ids] @ Wn[a_id])

    # fig 1 — the 9 records' null clouds vs the wrong token's sim
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, r, nd in zip(axes.ravel(), table, null_draws):
        sim = r["cos_sim_wrong_vs_answer"]
        lo = min(nd.min(), sim)
        hi = max(nd.max(), sim)
        pad = 0.1 * (hi - lo + 1e-9)
        ax.hist(nd, bins=40, color="#9db8d2", alpha=0.85,
                label=f"null: answer vs {N_NULL} random vocab")
        ax.axvline(sim, color="#c0392b", lw=2.2,
                   label=f"wrong '{r['wrong_token'].strip()}' "
                         f"(z={r['embedding_z']:+.1f})")
        ax.axvline(r["null_mean"], color="#34495e", lw=1.0, ls="--",
                   label=f"null mean")
        ttl = (f"{r['probe']}  [{r['wash']}]  {r['battery']}"
               + ("  (NEAR co-report)" if not r["adjudicates"] else ""))
        ax.set_title(ttl, fontsize=9.5,
                     color=("#7d6608" if not r["adjudicates"] else "black"))
        ax.set_xlabel(f"cos vs '{r['dead_answer_token'].strip()}' "
                      f"(pristine wte)", fontsize=8.5)
        ax.set_ylabel("count", fontsize=8.5)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7.5)
        ax.text(0.99, 0.60,
                f"rank_t0 = {r['logit_rank_wrong_t0']}\n"
                f"pct_null = {r['null_percentile']:.2f}\n"
                f"emb_pct = {r['full_vocab_emb_percentile']:.1f}%\n"
                f"{r['classification']}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(fc="white", ec="#999999", alpha=0.9))
    fig.suptitle("E241 — the wrong-choosers' new commitments vs the dead "
                 "answers' random-token clouds (pristine 124M wte; committed "
                 "t0/npz archive)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig1 = out / "e241_neighborhoods.png"
    fig.savefig(fig1, dpi=140)
    plt.close(fig)

    # fig 2 — the classification plane
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.set_xscale("log")
    y_lo = min(-1.0, min(r["embedding_z"] for r in table) - 1)
    y_hi = max(6.0, max(r["embedding_z"] for r in table) + 2)
    ax.fill_betweenx([Z_BAR, y_hi], 0.9, RANK_BAR + 0.5, color="#d5f5e3",
                     zorder=0, label="NEXT-DOOR region (rank<=3 & z>=2)")
    ax.plot([2.0, 2.0], [y_lo, Z_BAR], color="#f39c12", lw=6, alpha=0.5,
            solid_capstyle="butt",
            label="RUNNER-UP-PURE (rank==2 & z<2)")
    ax.axhline(Z_BAR, color="#2c3e50", lw=1, ls=":")
    ax.axvline(RANK_BAR + 0.5, color="#2c3e50", lw=1, ls=":")
    for r in table:
        x = r["logit_rank_wrong_t0"]
        y = r["embedding_z"]
        if r["adjudicates"]:
            ax.scatter(x, y, s=90, c="#c0392b" if r["classification"] ==
                       "NEXT-DOOR" else ("#e67e22" if r["classification"] ==
                                         "RUNNER-UP-PURE" else "#7f8c8d"),
                       edgecolors="black", zorder=3)
            ax.annotate(f"{r['probe'].split('->')[0][:26]}->{r['wash']}: "
                        f"'{r['wrong_token'].strip()}'",
                        (x, y), textcoords="offset points", xytext=(8, 4),
                        fontsize=7.6)
        else:
            ax.scatter(x, y, s=90, facecolors="none", edgecolors="#7d6608",
                       linewidths=1.8, zorder=3)
            ax.annotate(f"{r['probe']}->{r['wash']} (near)",
                        (x, y), textcoords="offset points", xytext=(8, -10),
                        fontsize=7.6, color="#7d6608")
    ax.set_xlim(0.9, 5e4)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("wrong token's logit-rank at t=0 (committed t0 logits; "
                  "log scale)")
    ax.set_ylabel("embedding z vs the dead answer's random-token null "
                  "(pristine wte)")
    ax.set_title(f"E241 — classification plane: {verdict} "
                 f"(NEXT-DOOR {n_next}/7, RUNNER-UP-PURE {n_run}/7, "
                 f"DISTANT {n_dis}/7 of the adjudicated 7)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig2 = out / "e241_classification.png"
    fig.savefig(fig2, dpi=140)
    plt.close(fig)
    log(f"figures: {fig1.name}, {fig2.name}")

    # -------------------------------------------------------------- metrics
    timing["finished"] = utcnow()
    timing["wall_s"] = round(time.time() - t_start, 1)
    for r in table:
        g = r.pop("_gates")
        r["join_dp_vs_e232_p_t0"] = round(g["dp_vs_e232_p_t0"], 10)
        r["join_dm_vs_e232_margin_80"] = round(g["dm_vs_e232_margin_80"], 10)
    metrics = {
        "experiment": "e241_answer_neighbors",
        "phase": ("desk-only on COMMITTED data + one read-only CPU load of "
                  "the pristine shared-124M wte (threads 4; no training, no "
                  "evals, no GPU)"),
        "date": timing["started"],
        "status": "DONE",
        "registration": ("bars + classification rules + prediction frozen "
                         "VERBATIM from the dispatch brief in this file's "
                         "docstring; the registration commit precedes any "
                         "compute; adjudicate against exactly this; no bar "
                         "shopping"),
        "question": ("is the wrong-choosers' collapse LOCAL (into the dead "
                     "answer's semantic neighborhood) or GLOBAL (scattered)?"),
        "builds_on": [
            "T217 / e234 (the answer-neighbor texture — 'wanders next door'; "
            "the registered prediction this cell adjudicates)",
            "T212 / e232 (the zombie taxonomy — the 7 wrong-choosers; the "
            "(probe, wash) death unit)",
            "e234's rider (the 1/7 corpus-mode kill of H-i re-teaching — "
            "CITED, never re-derived)",
            "e238 (the committed answer-position logit dumps, sha-gated; the "
            "one-time re-probe that produced them)",
            "e214/e228/e182c/e182c2 (the two-wash 124M archive underneath)",
        ],
        "whats_new": [
            "the per-record logit-rank of the wrong token at t=0 (the "
            "runner-up question on committed logits)",
            "the embedding z-read: wrong-vs-dead-answer cos sim against a "
            "1000-random-vocab null on the pristine wte",
            "the p-side start line: p(wrong|t0) vs p(answer|t0) and the raw "
            "logit gap",
            "the NEXT-DOOR / RUNNER-UP-PURE / DISTANT classification and the "
            "LOCAL-COLLAPSE / GLOBAL-SCATTER / MIXED adjudication",
        ],
        "registered_bars_verbatim": BARS_VERBATIM,
        "registered_classification_rules_verbatim": CLASS_RULES_VERBATIM,
        "registered_prediction_T217_verbatim": PREDICTION_VERBATIM,
        "operationalizations_frozen": {
            "unit": "one (probe, wash) death — e232's convention; 7 "
                    "adjudicated records (fact/ctrl) over 5 distinct probes "
                    "(US->dollar and Apple->iPhone die under both washes); "
                    "the 2 Georgia->Atlanta near records co-report only, "
                    "counted separately (e232/e234's rule)",
            "rank_t0": "1-based rank of the wrong token's logit among all "
                       "50257 entries of the committed t0 answer-position "
                       "vector (float64 argsort, stable)",
            "z_read": f"cos(wte[wrong], wte[answer]) on L2-normalized "
                      f"float64 pristine-wte rows; null = {N_NULL} ids drawn "
                      f"uniformly from the vocab minus the two ids; ONE "
                      f"global default_rng({NULL_SEED}); records in e234 "
                      f"rider order; z = (sim - mean)/std(ddof=1)",
            "p_side": "softmax in float64 of the committed t0 logits; gap = "
                      "l(answer) - l(wrong), the raw runner-up gap at the "
                      "start",
            "classification_order": "NEXT-DOOR tested first (z >= 2 AND "
                                    "rank <= 3), then RUNNER-UP-PURE (rank "
                                    "== 2 AND z < 2), else DISTANT — first "
                                    "match wins, exactly the dispatch's "
                                    "wording",
            "adjudication": "LOCAL-COLLAPSE iff >= 5/7 NEXT-DOOR; "
                            "GLOBAL-SCATTER iff >= 5/7 DISTANT; else MIXED; "
                            "the 7 = fact/ctrl records only",
            "co_reports": "full-vocab embedding percentile of the wrong "
                          "token (the direct top-1% read), the answer's "
                          "top-5 wte neighbors, the null percentile, all "
                          "near-record reads, the runner-up-gap table — none "
                          "adjudicate",
        },
        "sources": {
            "rider_identities": {
                "path": str(E234_METRICS), "sha256_16": sha16(E234_METRICS)},
            "wrong_chooser_records": {
                "path": str(E232_METRICS), "sha256_16": sha16(E232_METRICS)},
            "npz_shas_reference": {
                "path": str(E238_JOURNAL), "sha256_16": sha16(E238_JOURNAL)},
            "logits_t0": {
                "path": str(E238_NPZ["t0"]),
                "sha256_16": sha16(E238_NPZ["t0"])},
            "logits_w1s80": {
                "path": str(E238_NPZ["w1s80"]),
                "sha256_16": sha16(E238_NPZ["w1s80"])},
            "logits_w2s80": {
                "path": str(E238_NPZ["w2s80"]),
                "sha256_16": sha16(E238_NPZ["w2s80"])},
            "wte": {
                "repo": MODEL_REPO, "revision": MODEL_REV,
                "wte_sha256_16_fp32bytes": wte_sha,
                "snapshot_safetensors_sha256_16": snap_sha,
                "snapshot_file": snap_file if snap_sha else None},
        },
        "cited_not_rederived": {
            "corpus_mode_count": {
                "bigram": e234["rider"]["h_i_count_bigram"],
                "trigram": e234["rider"]["h_i_count_trigram"],
                "source": "runs/e234/metrics.json rider (e234's committed "
                          "bigram/trigram reads)"},
        },
        "table": table,
        "adjudication": {
            "n_adjudicated": len(adj),
            "n_NEXT_DOOR": n_next, "n_RUNNER_UP_PURE": n_run,
            "n_DISTANT": n_dis,
            "verdict": verdict,
            "bar_fired_verbatim": BARS_VERBATIM[verdict],
            "near_co_report_classifications":
                [r["classification"] for r in near_tbl],
            "prediction_read": ("T217's letter (top-1% semantic neighborhood) "
                                "is a CO-REPORT here (full-vocab embedding "
                                "percentile / null percentile columns); the "
                                "letter's adjudicator was never registered as "
                                "a bar — the registered bars are the "
                                "classification counts only"),
        },
        "gates": gates,
        "honesty_reflex": (
            "the rank read (t0 decision layer) and the z read (static "
            "embedding geometry) are INDEPENDENT instruments — the "
            "classification needs both, so neither alone manufactures a "
            "NEXT-DOOR. KNOWN CONFOUND, disclosed not corrected: ' the' is "
            "the vocab's most frequent token and its wte vector sits near "
            "the embedding cloud's mean direction, so its cosine to ANY "
            "answer can be elevated by frequency geometry rather than "
            "answer-specific semantics — the null (1000 uniform vocab ids) "
            "mostly samples rare tokens; the full-vocab percentile column "
            "is the less confounded read and co-reports only. the +80 "
            "dumps are used ONLY to re-verify the committed flips (gate), "
            "never to adjudicate; the 2 near records never adjudicate."),
        "figures": [str(fig1), str(fig2)],
        "compute": {
            "device": "cpu desk pass (threads 4)",
            "model_loads": 1,
            "model_role": "read-only wte extraction from the pristine shared "
                          "124M (the t0 organism)",
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
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "matplotlib": matplotlib.__version__,
            },
        },
        "timing": timing,
        "trims": [],
        "deviations": deviations
        + [
            "the null-draw sequence is consumed once for the metrics and "
            "re-drawn with the SAME seed for the histogram figure (bit-wise "
            "identical by construction; no second sample)",
            "G_JOIN232's dp tolerance was widened from 1e-6 to 1e-4 after "
            "the first run tripped: softmax of the committed FLOAT32 npz "
            "dump differs from e232/e214's committed journal p by a "
            "systematic 0.8-1.1e-5 on every record (float32 round-trip; "
            "e238's own gate compared its LIVE forward to e214 and read "
            "exactly 0.0). Identity gate only — no bar, no read, and no "
            "classification sits within orders of magnitude of this scale; "
            "per-record dp/dm kept in the table as join_* fields",
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
