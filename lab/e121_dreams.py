"""E121 — THE DREAMS PROBE: does the net's OWN spontaneous free-run replay
consolidate the installed fact? (REGISTERED before compute)

Arc six opener (P5's registered third road — scratch/next_arc_programs.md
section 1, the e120-designated cell; the coordinator's dispatch assigns slot
e121 and renames it THE DREAMS PROBE). Frame: W004 (THINKING.md) — identity
is a fixed-point property ("X is self iff X looks like what my weights
produce"); this experiment asks whether that fixed point reaches into
LEARNING dynamics: can the net consolidate an installed fact from nothing
but its own dreams — self-generated exposure, with NO re-teaching — and does
it consolidate its OWN dreams better than its twin's identical-statistics
dreams?

BASE (frozen): the B43 install line, runs/checkpoints/e082_b43_install.pt
(B43 + e043 exposure verbatim, seed 24331; e082 GATE-0 install-60 p(Z)
0.3198, R1i 0.191; GATE-1 address row r* = 129, NOT retargeted).
REPRODUCTION GATE (G_INST): base install-60 battery p(Z) = 0.3198 +- 0.005
(the coordinator's stated gate; e091 tolerance convention).

ARMS — matched-exposure fine-tunes from the SAME base net; same step budget
(300 steps), same optimizer (AdamW 0.9/0.95 wd 0.1, constant lr 1e-3, clip
1.0), same batch 32 (16 exposure windows + 16 anchors = 8 paired originals +
8 random corpus, e109 arm-(a) composition verbatim), same draw generator
(seed 12101 — arms (a)/(b) draw the SAME pool slots: slot k = same held
prompt, same sampling seed, different generator). The ONLY difference is the
exposure text:
  (a) OWN-DREAMS: free-run the BASE NET ITSELF from the 30 held prompts
      (held-30 contexts, never trained in any form), 4 dreams per prompt
      (seeds 12110..12113), T=0.8 top_k=40 (the lab free-run standard,
      e043-R3/e048-repro convention), 126 new tokens each (window = 130
      prompt + 126 dream = 256). The generated continuations are used
      VERBATIM as fine-tune text — name-bearing or not (the e120 tier-1
      splice fallback is explicitly NOT taken; the coordinator's instruction
      overrides). Loss on the 126 continuation columns only (the net's own
      text; prompt columns are context, e109 mask convention).
  (b) TWIN-DREAMS: e043_donor_bdo.pt's free-run continuations from the SAME
      30 held prompts, same 4 sampling seeds, same T/top-k, same pool layout
      (120 windows) — exposure-matched; only the GENERATOR of the dream text
      differs. TWIN CHOICE (documented deviation 2): the coordinator's
      first choice e041_bdo is (i) NOT B43-line-matched (seed-42 lineage)
      and (ii) BARE — no install, so its dreams have ZERO name-content by
      construction (not "identical-statistics dreams"); the only other
      B43-family member with the install (e043_donor_b43) is a
      device-replicate of the base itself (same exposure seed 24331, e082
      xcheck max|d| ~1e-6 — near-identical dream text, the own-vs-twin
      contrast would be vacuous). Chosen: e043_donor_bdo (same-init-as-B /
      different-batch-order lineage, same cfg 6L/6H/192/256, matched
      install: R1i 0.202/0.890 vs base 0.191/0.881). Gate G_TWIN: its
      install-60 battery p(Z) >= 0.20 (e067 informativeness floor; expected
      ~0.3) else PARK (document; the twin choice would need re-tasking).
  (c) NO-EXPOSURE control: equal steps, same batch slots, same loss-mask
      geometry — but the 16 exposure windows are retain-style CORPUS text:
      the SAME 30 held prompts with continuations spliced from random
      corpus positions (4 per prompt, seed 12103; matched context exposure,
      loss on the same 126 continuation columns). Pure-retention reference:
      the ONLY difference from (a) is that the continuation text is corpus
      rather than self-generated.

THEN (eval-only, all CPU): delete the address rows — D2-STYLE SUBTRACTIVE
ROW-ZERO on the wpe address rows with the e065/e113 confinement gate
(e113's deleted_wpe quotes this as "D2-style subtractive row-zero on wpe
rows"; the e042/e083 TOKEN-space D2 reset wte[Z]/lm_head[Z] is NOT used
because zeroing lm_head[Z] forces logit-Z = 0 and makes p(Z) uninformative
as a readout — deviation 4). Address sets are PER-NET (e113 convention):
D129 = {129}; D-ALL-ITS-ADDRESSES = {129} + grown rows, grown row r in
{121..137} minus {129} with norm(arm, r) - norm(base, r) > 0.02 (e109's grown rows
inflated 0.05-0.09 above baseline; 0.02 is half the smallest observed
inflation and above wd-drift noise). Row 0 is NEVER touched. Measured on
each net: battery p(Z) at the e113 arm-set geometries (-8,-4,0,+4,+8;
readout at wpe row 129+g), install-60 primary + held-30 report-only, plus
CE_R (60 name-free val windows, e065 seed 26502) for retain health.

REGISTERED BARS (coordinator, verbatim; adjudicated in this priority order):
  1. (a) post-D-all p(Z) >= 0.20 (e113 operationalization: >= 3 of 5
     geometries on install-60; geometry-0 reported) while (c) collapses
     (max over geometries <= 0.05)  => DREAMS CONSOLIDATE (spontaneous
     replay alone graduates the fact — W004's self reaches into learning).
  2. (a) > (b) with CI separation (non-overlapping bootstrap 95% CIs on the
     post-D-all install-60 mean, geometry 0; direction a > b required)
     => OWN-DREAM PREFERENCE (the net needs ITS dreams, not just any
     same-family text — the fixed point gates learning itself).
  3. (a) ~ (b) (overlapping CIs) AND both survive (each >= 0.20 at >= 3/5
     geometries) => FAMILY-TEXT SUFFICES (dreams work but self-specificity
     stops at the field).
  4. (a) collapses like (c) (both max_g <= 0.05) => NO DREAM CONSOLIDATION
     (replay needs jitter or teaching signal — report honestly).
  5. none of the above => TEXTURE (honest report). CALIBRATED SECONDARY
     (always reported): the unconsolidated line's known D129 floor is
     0.1945 at geometry 0 (e082 A1-zero) and its shifted-geometry no-copy
     plateau is ~0.123 (e082 free rider), so the <= 0.05 collapse bar may
     be unreachable on this line under wpe-only deletion; the base net's
     OWN post-D-all row is measured live in-run and every arm is
     additionally read as a CI-separated delta vs that base reference.
COVARIATE (not a gate): the harvest's name-rate — ZEPHYRA occurrences,
Z chars, per-10k-char rates, windows-with-name — for own and twin dreams
(e048 counting convention). If the dreams carry little name-content, that
is reported honestly and the adjudication is annotated with it (bar 4's
"replay needs jitter or teaching signal" then has a name-poverty reading).

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import;
torch.set_num_threads(8); e113's exact envelope — the GPU thermal machinery
(180 s caps, double-polls, cooldowns) is void without a GPU). Fine-tune
wall cap 1500 s each (e113's CPU envelope). Single-run target ~35 min.

Outputs: runs/e121/{metrics.json, dreams.png}.
No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).

Run:  cd lab && python e121_dreams.py        (E121_SMOKE=1 for smoke)
"""
from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (coordinator)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(8)                              # 8 threads max

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E121_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
BASE_CK = E43.REPO / "runs" / "checkpoints" / "e082_b43_install.pt"
TWIN_CK = E43.REPO / "runs" / "checkpoints" / "e043_donor_bdo.pt"

JITTERS = (-8, -4, 0, 4, 8)       # the e113 arm-set geometries
GEO_ORDER = [-8, -4, 0, 4, 8]
ADDR_BAND = tuple(r for r in range(121, 138))   # grown-row census band
GROWN_D_NORM = 0.02               # registered grown-row inflation threshold

# dream harvest (registered)
GEN_T, GEN_TOPK = 0.8, 40         # lab free-run standard
DREAM_SEEDS = (12110, 12111, 12112, 12113)
N_DREAMS_PER_PROMPT = 1 if SMOKE else 4
N_PROMPTS = 8 if SMOKE else 30    # the held-30 prompts (smoke: first 8)
DREAM_LEN = BLOCK - PRE           # 126 new tokens -> 256-token windows
CORP_CONT_SEED = 12103            # arm-(c) random corpus-continuation draws

# fine-tune envelope (e109 arm (a) verbatim except CPU caps)
FT_LR = 1e-3
FT_STEPS = 8 if SMOKE else 300
FT_TIME_CAP = 1500.0              # CPU safety cap (e113)
EVAL_EVERY = 2 if SMOKE else 50
NAME_BS = 16                      # exposure windows per step
ANCH_BS = 16                      # anchor windows per step (8 paired + 8 random)
CONS_SEED = 12101                 # SAME draw generator for all three arms

# batteries / guards
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_INST_REF = 0.3198219250887632   # e082 gate0 final install-60 p(Z)
G_INST_TOL = 0.005
G_TWIN_FLOOR = 0.20               # e067 informativeness floor

# registered bars
BAR_SURVIVE = 0.20
BAR_COLLAPSE = 0.05
BOOT_N = 10000

REGISTERED_BARS = {
    "dreams_consolidate": "arm (a) post-D-all p(Z) >= 0.20 at >= 3/5 "
                          "geometries (install-60) while arm (c) collapses "
                          "(max_g <= 0.05)",
    "own_dream_preference": "post-D-all install-60 geometry-0 mean p(Z): "
                            "(a) > (b) with non-overlapping bootstrap 95% CIs",
    "family_text_suffices": "(a) ~ (b) (overlapping CIs) AND both survive "
                            "(each >= 0.20 at >= 3/5 geometries)",
    "no_dream_consolidation": "(a) collapses like (c) (both max_g <= 0.05)",
    "texture": "none of the above — honest report + calibrated base-referenced "
               "deltas (base's own post-D-all row measured live)",
    "priority": "adjudicated in order 1-4; first match fires",
}

trims: list[str] = []
deviations: list[str] = [
    "Numbering: scratch/next_arc_programs.md numbers the dream-road cell e120 "
    "and e121 as the store-sharing probe; the coordinator's dispatch assigns "
    "slot e121 to THE DREAMS PROBE (the e120 design verbatim otherwise). "
    "Outputs under runs/e121/.",
    "TWIN = e043_donor_bdo.pt (not e041_bdo): e041_bdo is seed-42-line (not "
    "B43-matched) AND bare (no install -> zero name-content dreams, not "
    "identical-statistics); the only installed B43-family alternative "
    "(e043_donor_b43) is a device-replicate of the base itself (same exposure "
    "seed 24331, e082 xcheck ~1e-6) whose dreams would be near-identical text. "
    "e043_donor_bdo: same-init-as-B/different-batch-order lineage, same cfg, "
    "matched install strength (R1i 0.202/0.890 vs base 0.191/0.881).",
    "CPU-only (CUDA_VISIBLE_DEVICES=-1, 8 threads) per the coordinator's "
    "'CPU-ONLY preferred'; the GPU 180 s cap / double-poll / cooldown(120) "
    "machinery is void without a GPU (e113 precedent); per-fine-tune CPU wall "
    "cap 1500 s so the registered 300 steps complete.",
    "'D2 row-reset' implemented as e113's D2-style subtractive row-zero on "
    "the WPE address rows with the e065/e113 confinement gate (e113's own "
    "docstring names deleted_wpe 'D2-style subtractive row-zero on wpe "
    "rows'); the e042/e083 token-space D2 (wte[Z]=lm_head[Z]=0) is NOT used "
    "because a zeroed lm_head[Z] pins logit-Z at 0 for every context, making "
    "p(Z) uninformative as a consolidation readout.",
    "Coordinator bars kept verbatim as the primary adjudication; a calibrated "
    "secondary (live-measured base post-D-all reference + bootstrap CI "
    "deltas) is reported because this line's unconsolidated D129 floor is "
    "0.1945 (e082 A1-zero) — the <= 0.05 collapse bar may be unreachable on "
    "this line under wpe-only deletion.",
    "Harvest temperature T=0.8 top_k=40 (lab standard); e048's T-sweep "
    "(0.7/1.0/1.3/greedy/seeded) on the seed-42 install line found ZERO "
    "expression at every temperature, so temperature is not the binding "
    "variable — the name-rate covariate is measured on THIS line instead.",
    "Arm (c) construction: same 30 held prompts, continuations spliced from "
    "random corpus positions (retain-style text) with the identical "
    "loss-mask geometry — keeps 'the ONLY difference is the exposure text' "
    "exact (dream continuations vs corpus continuations on the same "
    "contexts).",
    "Batched free-run sampler (all prompts sampled in one batch with "
    "per-row independent multinomial draws) — statistically equivalent to "
    "common.generate's single-row loop at the same T/top-k; ~15x faster on "
    "CPU.",
]


# ------------------------------------------------------------------ instruments

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30,
                 keep_per_ctx=False) -> dict:
    """e068-style battery on CPU: p(Z) at the last position over contexts."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    out = {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
           "std_pz": float(p.std()),
           "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
           "frac_argmax_z": amax / ids.shape[0]}
    if keep_per_ctx:
        out["pz_per_ctx"] = [float(v) for v in p.tolist()]
    return out


@torch.no_grad()
def ce_fixed_cpu(net: TinyGPT, x, y, bs=64) -> float:
    """e065 ce_fixed (CPU)."""
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    """e065 val_windows verbatim: name-free val-split windows."""
    g = torch.Generator().manual_seed(seed)
    out_x, out_y = [], []
    tries = 0
    while len(out_x) < n and tries < 500 * n:
        i = int(torch.randint(len(val_ids) - block - 1, (1,), generator=g))
        txt = val_text[i: i + block + 1]
        tries += 1
        if "ZEPHYRA" in txt or "ZEPH" in txt:
            continue
        out_x.append(val_ids[i: i + block])
        out_y.append(val_ids[i + 1: i + 1 + block])
    return torch.stack(out_x), torch.stack(out_y)


def deleted_wpe(sd: dict, rows: tuple[int, ...]) -> tuple[dict, dict]:
    """D2-style subtractive row-zero on wpe rows; confinement gate (e065
    G_SURG convention: exact element count, row confinement, everything else
    bit-identical). Row 0 is NEVER touched by construction of the row sets."""
    out = {k: v.clone() for k, v in sd.items()}
    for r in rows:
        out["wpe.weight"][r] = 0.0
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    gate = {"rows": list(rows), "n_elements_changed": n,
            "expected": len(rows) * sd["wpe.weight"].shape[1],
            "changed_rows": changed_rows,
            "confined": bool(changed_rows == sorted(rows)),
            "others_bit_identical": bool(others),
            "pass": bool(n == len(rows) * sd["wpe.weight"].shape[1]
                         and changed_rows == sorted(rows) and others)}
    return out, gate


@torch.no_grad()
def free_run_batch(net: TinyGPT, prompt_ids: torch.Tensor, n_new: int,
                   seed: int, temperature: float = GEN_T,
                   top_k: int = GEN_TOPK) -> torch.Tensor:
    """Batched free-run (common.generate semantics: T, top-k, per-row
    independent multinomial). Returns (n_prompts, PRE+n_new) ids."""
    net.eval()
    torch.manual_seed(seed)
    idx = prompt_ids.clone()
    for _ in range(n_new):
        logits, _ = net(idx[:, -net.cfg.block_size:])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx


def dream_covariates(windows: torch.Tensor, itos) -> dict:
    """e048 counting convention over the continuation region of each window."""
    texts = ["".join(itos[int(i)] for i in w[PRE:].tolist()) for w in windows]
    chars = sum(len(t) for t in texts)
    zephyra = sum(t.count("ZEPHYRA") for t in texts)
    zephs = sum(t.count("ZEPH") for t in texts)
    z_chars = sum(t.count("Z") for t in texts)
    first_z = sum(1 for t in texts if t.startswith("Z"))
    return {
        "n_windows": len(windows), "n_chars": chars,
        "zephyra_count": zephyra, "zeph_count": zephs,
        "z_char_count": z_chars, "first_char_z_windows": first_z,
        "windows_with_zephyra": sum(1 for t in texts if "ZEPHYRA" in t),
        "per_10k_chars": {
            "ZEPHYRA": 1e4 * zephyra / max(chars, 1),
            "ZEPH": 1e4 * zephs / max(chars, 1),
            "Z_chars": 1e4 * z_chars / max(chars, 1)},
        "sample_continuation_heads": texts[:3],
    }


# ------------------------------------------------------------------ fine-tune

def finetune_arm(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                 pool_mask: torch.Tensor, anchor: torch.Tensor,
                 train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                 seed: int):
    """e109 arm-(a) fine-tune recipe VERBATIM (composition/seed/loss), CPU:
    batch 32 = 16 exposure windows from the pool + 16 anchors (8 paired + 8
    random); token-level union CE over the exposure mask + full anchor CE;
    AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0; 300 steps."""
    net = copy.deepcopy(net0).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    traj, t_start = [], time.time()
    step = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, FT_STEPS + 1):
        ix = torch.randint(n_pool, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                           generator=gen)
        nw = pool_x[ix].to(CPU)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])], 0).to(CPU)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(NAME_BS + ANCH_BS, x.shape[1], dtype=torch.bool, device=CPU)
        m[:NAME_BS] = pool_mask[ix].to(CPU)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              reduction="none").view(x.shape[0], x.shape[1])
        nm = nll[:NAME_BS][m[:NAME_BS]]
        cm = nll[NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == FT_STEPS or \
                (time.time() - t_start) > FT_TIME_CAP:
            evl.load_state_dict({k: v.detach().cpu().clone()
                                 for k, v in net.state_dict().items()})
            evl.eval()
            bz = battery_cell(evl, f_eval_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "p_z_mean": bz["mean_pz"],
                         "frac_argmax_z": bz["frac_argmax_z"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p_z {bz['mean_pz']:.4f} argmaxZ "
                f"{bz['frac_argmax_z']:.2f} CE_R {ce_r:.4f} "
                f"({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            break
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e121_smoke" if SMOKE else "e121")
    log(f"E121 THE DREAMS PROBE (own dreams vs twin dreams vs no exposure; "
        f"smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda visible = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e082/e113 verbatim)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)

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
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30 "
        f"(SPLICE_RNG {E43.SPLICE_RNG})")

    # batteries per geometry (e113 construction verbatim): ctx =
    # train_text[p-PRE-j:p], readout p(Z) at the last position (row 129+j).
    bat_ids = {}
    for j in GEO_ORDER:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    f_eval_ids = bat_ids[(0, "install60")]          # the G_INST battery

    # CE_R eval bank (e065 verbatim, seed 26502)
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # held prompts (the dream-harvest contexts; never trained in any form)
    prompts = [train_text[p - PRE: p] for p, _ in held_occ][:N_PROMPTS]
    prompt_ids = torch.stack([corpus.encode(c) for c in prompts])
    log(f"held prompts: {prompt_ids.shape[0]} x {PRE} chars")

    # anchor bank (e065/e109 verbatim): first 16 install-position corpus windows
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])

    # ---------------- base net + instrument gate
    net0 = load_cpu(BASE_CK)
    evl = copy.deepcopy(net0)
    bz0 = battery_cell(evl, f_eval_ids, zid)
    G_INST = {"battery_pz": bz0["mean_pz"], "ref": G_INST_REF,
              "tol": G_INST_TOL,
              "pass": bool(abs(bz0["mean_pz"] - G_INST_REF) < G_INST_TOL)}
    log(f"G_INST base installed battery p(Z) {bz0['mean_pz']:.6f} "
        f"(ref {G_INST_REF:.4f}): {'PASS' if G_INST['pass'] else 'FAIL'}")
    if not G_INST["pass"]:
        raise RuntimeError("base net/battery mismatch vs e082 (gate failed)")
    ce_r0 = ce_fixed_cpu(evl, *r_eval_xy)
    log(f"CE_R (60 name-free val windows): {ce_r0:.4f}")

    # ---------------- twin + gate
    twin = load_cpu(TWIN_CK)
    bz_tw = battery_cell(twin, f_eval_ids, zid)
    G_TWIN = {"ckpt": TWIN_CK.name, "battery_pz": bz_tw["mean_pz"],
              "floor": G_TWIN_FLOOR, "pass": bool(bz_tw["mean_pz"] >= G_TWIN_FLOOR)}
    log(f"G_TWIN {TWIN_CK.name} install-60 battery p(Z) "
        f"{bz_tw['mean_pz']:.4f} (floor {G_TWIN_FLOOR}): "
        f"{'PASS' if G_TWIN['pass'] else 'FAIL'}")
    if not G_TWIN["pass"]:
        raise RuntimeError("twin lacks a comparable install (gate failed)")

    # ---------------- dream harvest (own + twin) + covariates
    dream_ids = {}
    cov = {}
    for tag, net in (("own", net0), ("twin", twin)):
        wins = []
        for s in DREAM_SEEDS[:N_DREAMS_PER_PROMPT]:
            out = free_run_batch(net, prompt_ids, DREAM_LEN, seed=s)
            wins.append(out)
            log(f"  harvest[{tag}] seed {s}: {out.shape[0]} dreams x "
                f"{DREAM_LEN} new tokens")
        dream_ids[tag] = torch.cat(wins)            # (n_prompts*K, 256)
        cov[tag] = dream_covariates(dream_ids[tag], itos)
        c = cov[tag]
        log(f"  harvest[{tag}]: ZEPHYRA {c['zephyra_count']} in "
            f"{c['n_chars']} chars ({c['per_10k_chars']['ZEPHYRA']:.1f}/10k) "
            f"| Z chars {c['z_char_count']} | windows w/ name "
            f"{c['windows_with_zephyra']}/{c['n_windows']}")

    # ---------------- arm (c) pool: corpus continuations on the same prompts
    # seed-major slot layout (seed blocks of all prompts) matching the dream
    # pools, so slot k of (a)/(b)/(c) is the SAME held prompt everywhere.
    g = torch.Generator().manual_seed(CORP_CONT_SEED)
    src = torch.randint(len(train_ids) - DREAM_LEN - 1,
                        (N_DREAMS_PER_PROMPT, len(prompts)), generator=g)
    cont = torch.stack([train_ids[s: s + DREAM_LEN] for s in src.flatten()])
    pool_c_x = torch.cat([prompt_ids.repeat(N_DREAMS_PER_PROMPT, 1), cont], 1)
    if pool_c_x.shape[1] != BLOCK:
        raise RuntimeError(f"arm-c window len {pool_c_x.shape[1]}")

    # pools + masks (slot layout: prompt-major, seed-minor — arms a/b slot-matched)
    pool_a_x = dream_ids["own"]
    pool_b_x = dream_ids["twin"]
    m = torch.zeros(pool_a_x.shape[0], BLOCK - 1, dtype=torch.bool)
    m[:, PRE - 1:] = True                          # y-cols 129..254 = the
    # 126 continuation tokens (x-positions 130..255): loss on exposure text only
    pool_mask = m
    log(f"pools: own {tuple(pool_a_x.shape)} | twin {tuple(pool_b_x.shape)} "
        f"| corpus-ctl {tuple(pool_c_x.shape)} | mask cols "
        f"{int(pool_mask[0].sum())}/window")

    # ---------------- the three matched-exposure fine-tunes
    arms = {}
    for tag, pool, desc in (
            ("a_own_dreams", pool_a_x,
             f"own free-run dreams ({N_PROMPTS} held prompts x "
             f"{N_DREAMS_PER_PROMPT} seeds), verbatim"),
            ("b_twin_dreams", pool_b_x,
             f"twin (e043_donor_bdo) free-run dreams, same prompts/seeds"),
            ("c_no_exposure", pool_c_x,
             "corpus continuations spliced after the same held prompts")):
        log(f"ARM {tag}: {FT_STEPS} steps, lr {FT_LR}, batch 32, "
            f"seed {CONS_SEED} — {desc}")
        res = finetune_arm(tag, net0, pool, pool_mask, anchor, train_ids,
                           r_eval_xy, f_eval_ids, zid, CONS_SEED)
        res["desc"] = desc
        arms[tag] = res

    # ---------------- nets to measure: base + the three arms
    nets = {"base_no_ft": {"sd": net0.state_dict(),
                           "desc": "e082_b43_install reference (no fine-tune)"}}
    for tag, res in arms.items():
        nets[tag] = {"sd": res["sd"], "desc": res["desc"]}

    # ---------------- wpe probes + grown-row detection (registered rule)
    w_base = net0.state_dict()["wpe.weight"]
    wpe_probes = {}
    addr_sets = {}
    for tag, n in nets.items():
        w = n["sd"]["wpe.weight"]
        wpe_probes[tag] = {str(r): {"norm": float(w[r].norm()),
                                    "base_norm": float(w_base[r].norm()),
                                    "d_norm": float(w[r].norm() - w_base[r].norm())}
                           for r in (0,) + ADDR_BAND}
        grown = tuple(r for r in ADDR_BAND
                      if r != 129 and w[r].norm() - w_base[r].norm() > GROWN_D_NORM)
        addr_sets[tag] = {"d129": (129,), "d_all_addresses": (129,) + grown,
                          "grown_rows": list(grown)}
        log(f"wpe[{tag}]: grown rows (d_norm > {GROWN_D_NORM}) = "
            f"{list(grown) or 'none'} | d129 d_norm "
            f"{wpe_probes[tag]['129']['d_norm']:+.3f}")

    # ---------------- deletion x geometry battery (e113 machinery)
    table = {}                     # (net, deletion, geometry, battery) -> cell
    gates_surg = {}
    evl = copy.deepcopy(net0)
    for tag, n in nets.items():
        for dl_name, rows in (("none", ()), ("d129", addr_sets[tag]["d129"]),
                              ("d_all_addresses", addr_sets[tag]["d_all_addresses"])):
            if dl_name == "none":
                sd_del, gate = {k: v.clone() for k, v in n["sd"].items()}, \
                    {"rows": [], "pass": True, "note": "no deletion"}
            else:
                sd_del, gate = deleted_wpe(n["sd"], rows)
            gates_surg[(tag, dl_name)] = gate
            if not gate["pass"]:
                raise RuntimeError(f"deletion gate FAILED {tag}/{dl_name}: {gate}")
            evl.load_state_dict(sd_del)
            for j in GEO_ORDER:
                for bt in ("install60", "held30"):
                    table[(tag, dl_name, j, bt)] = battery_cell(
                        evl, bat_ids[(j, bt)], zid,
                        keep_per_ctx=(bt == "install60"))
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            table[(tag, dl_name, "ce_r", "-")] = {"ce_r": ce_r}
            log(f"net {tag:14s} deletion {dl_name:17s} (rows "
                f"{list(rows) or '-'}): install60 "
                + " ".join(f"g{g_:+d} {table[(tag, dl_name, g_, 'install60')]['mean_pz']:.3f}"
                           for g_ in GEO_ORDER)
                + f" | CE_R {ce_r:.4f}")

    # ---------------- bootstrap CIs (post-D-all, geometry 0, install-60)
    def boot_ci(per_ctx):
        arr = np.asarray(per_ctx, dtype=np.float64)
        rg = np.random.default_rng(12105)
        idx = rg.integers(0, len(arr), size=(BOOT_N if not SMOKE else 200,
                                             len(arr)))
        means = arr[idx].mean(1)
        return {"mean": float(arr.mean()),
                "lo": float(np.percentile(means, 2.5)),
                "hi": float(np.percentile(means, 97.5))}

    ci = {tag: boot_ci(table[(tag, "d_all_addresses", 0, "install60")]
                       ["pz_per_ctx"]) for tag in nets}
    ci_d129 = {tag: boot_ci(table[(tag, "d129", 0, "install60")]
                            ["pz_per_ctx"]) for tag in nets}
    for tag in nets:
        log(f"post-D-all g0 install60 [{tag}]: mean {ci[tag]['mean']:.4f} "
            f"95% CI [{ci[tag]['lo']:.4f}, {ci[tag]['hi']:.4f}]")

    # report-only crosscheck: base post-D129 g0 vs e082's A1-zero reference
    X_D129_REF = 0.1945319922024038      # e082 A1 own-row destroy (zero)
    x_d129 = {"this_run": ci_d129["base_no_ft"]["mean"], "e082_ref": X_D129_REF,
              "diff": ci_d129["base_no_ft"]["mean"] - X_D129_REF,
              "note": "same net, same battery, same surgery — device-exact "
                      "crosscheck of the deletion/battery machinery "
                      "(report-only)"}
    log(f"crosscheck base post-D129 g0: {x_d129['this_run']:.4f} "
        f"(e082 A1 ref {X_D129_REF:.4f}, diff {x_d129['diff']:+.4f})")

    # ---------------- adjudication (registered)
    def counts(tag, dl, bt="install60"):
        vals = [table[(tag, dl, g_, bt)]["mean_pz"] for g_ in GEO_ORDER]
        return {"per_geometry": {f"g{g_:+d}": table[(tag, dl, g_, bt)]["mean_pz"]
                                 for g_ in GEO_ORDER},
                "max": max(vals), "min": min(vals),
                "n_ge_020": sum(v >= BAR_SURVIVE for v in vals),
                "n_le_005": sum(v <= BAR_COLLAPSE for v in vals)}

    cd = {(tag, dl): counts(tag, dl)
          for tag in nets for dl in ("none", "d129", "d_all_addresses")}
    held = {(tag, dl): counts(tag, dl, "held30")
            for tag in nets for dl in ("none", "d129", "d_all_addresses")}

    a_surv = cd[("a_own_dreams", "d_all_addresses")]["n_ge_020"] >= 3
    b_surv = cd[("b_twin_dreams", "d_all_addresses")]["n_ge_020"] >= 3
    c_coll = cd[("c_no_exposure", "d_all_addresses")]["max"] <= BAR_COLLAPSE
    a_coll = cd[("a_own_dreams", "d_all_addresses")]["max"] <= BAR_COLLAPSE
    a_gt_b_ci = bool(ci["a_own_dreams"]["lo"] > ci["b_twin_dreams"]["hi"])
    a_b_overlap = not (ci["a_own_dreams"]["lo"] > ci["b_twin_dreams"]["hi"] or
                       ci["b_twin_dreams"]["lo"] > ci["a_own_dreams"]["hi"])

    if a_surv and c_coll:
        fired = "DREAMS CONSOLIDATE"
    elif a_gt_b_ci:
        fired = "OWN-DREAM PREFERENCE"
    elif a_b_overlap and a_surv and b_surv:
        fired = "FAMILY-TEXT SUFFICES"
    elif a_coll and c_coll:
        fired = "NO DREAM CONSOLIDATION"
    else:
        fired = "TEXTURE"

    # calibrated secondary: deltas vs the live base post-D-all reference
    cal = {}
    for tag in ("a_own_dreams", "b_twin_dreams", "c_no_exposure"):
        cal[tag] = {"post_dall_g0": ci[tag]["mean"],
                    "base_post_dall_g0": ci["base_no_ft"]["mean"],
                    "delta_vs_base": ci[tag]["mean"] - ci["base_no_ft"]["mean"],
                    "ci_excludes_base_mean": bool(
                        ci[tag]["lo"] > ci["base_no_ft"]["mean"] or
                        ci[tag]["hi"] < ci["base_no_ft"]["mean"])}
    arm_a_above_base = bool(ci["a_own_dreams"]["lo"] > ci["base_no_ft"]["mean"])
    arm_c_above_base = bool(ci["c_no_exposure"]["lo"] > ci["base_no_ft"]["mean"])

    adjudication = {
        "counts_post_d_all": {t: cd[(t, "d_all_addresses")] for t in nets},
        "counts_post_d129": {t: cd[(t, "d129")] for t in nets},
        "counts_pre_none": {t: cd[(t, "none")] for t in nets},
        "held30_post_d_all": {t: held[(t, "d_all_addresses")] for t in nets},
        "a_survives": bool(a_surv), "b_survives": bool(b_surv),
        "c_collapses": bool(c_coll), "a_collapses": bool(a_coll),
        "a_gt_b_ci_separated": a_gt_b_ci, "a_b_ci_overlap": bool(a_b_overlap),
        "bootstrap_ci_g0_post_dall": ci, "bootstrap_ci_g0_post_d129": ci_d129,
        "calibrated_vs_base": cal,
        "a_above_base_ci": bool(arm_a_above_base),
        "c_above_base_ci": bool(arm_c_above_base),
        "fired": fired,
        "name_rate_covariate": cov,
        "headline": (f"post-D-all install-60: own {ci['a_own_dreams']['mean']:.3f} "
                     f"[{ci['a_own_dreams']['lo']:.3f},{ci['a_own_dreams']['hi']:.3f}] "
                     f"| twin {ci['b_twin_dreams']['mean']:.3f} "
                     f"[{ci['b_twin_dreams']['lo']:.3f},{ci['b_twin_dreams']['hi']:.3f}] "
                     f"| no-exp {ci['c_no_exposure']['mean']:.3f} "
                     f"| base {ci['base_no_ft']['mean']:.3f} -> {fired}"),
    }
    log("=" * 78)
    log(f"E121 VERDICT: {fired}")
    log(f"  (a) own-dreams post-D-all: max {cd[('a_own_dreams', 'd_all_addresses')]['max']:.3f} "
        f"n>=0.20 {cd[('a_own_dreams', 'd_all_addresses')]['n_ge_020']}/5 "
        f"g0 {ci['a_own_dreams']['mean']:.3f}")
    log(f"  (b) twin-dreams post-D-all: max {cd[('b_twin_dreams', 'd_all_addresses')]['max']:.3f} "
        f"n>=0.20 {cd[('b_twin_dreams', 'd_all_addresses')]['n_ge_020']}/5 "
        f"g0 {ci['b_twin_dreams']['mean']:.3f}")
    log(f"  (c) no-exposure post-D-all: max {cd[('c_no_exposure', 'd_all_addresses')]['max']:.3f} "
        f"collapses {c_coll} | base post-D-all g0 "
        f"{ci['base_no_ft']['mean']:.3f}")
    log(f"  harvest name-rate: own {cov['own']['zephyra_count']} ZEPHYRA / "
        f"{cov['own']['n_chars']} chars | twin {cov['twin']['zephyra_count']} / "
        f"{cov['twin']['n_chars']}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e121_dreams",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("coordinator dispatch (P5 registered third road / "
                         "e120-designated cell; slot e121). Docstring written "
                         "before compute; bars verbatim below"),
        "registered_bars": REGISTERED_BARS,
        "question": ("can the net's OWN spontaneous free-run dreams — with NO "
                     "re-teaching — consolidate the installed fact into the "
                     "field, and better than its twin's identical-statistics "
                     "dreams?"),
        "base": f"runs/checkpoints/{BASE_CK.name} (e082 B43 install line)",
        "twin": {"ckpt": TWIN_CK.name, "gate": G_TWIN,
                 "choice_note": deviations[1]},
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "geometries": list(GEO_ORDER),
                     "battery_construction": "ctx = train_text[p-PRE-j:p], "
                                             "readout p(Z) at last position "
                                             "(wpe row 129+j); install-60 "
                                             "primary, held-30 report-only",
                     "held_prompts": f"{N_PROMPTS} held-30 contexts x "
                                     f"{N_DREAMS_PER_PROMPT} dreams",
                     "dream_harvest": {"T": GEN_T, "top_k": GEN_TOPK,
                                       "seeds": list(DREAM_SEEDS[:N_DREAMS_PER_PROMPT]),
                                       "new_tokens": DREAM_LEN}},
        "gates": {"G_SPLICE": G_SPLICE, "G_INST": G_INST, "G_TWIN": G_TWIN,
                  "G_SURG": {f"{k[0]}/{k[1]}": v for k, v in gates_surg.items()},
                  "ce_r_base": ce_r0, "xcheck_base_post_d129": x_d129},
        "fine_tune": {"lr": FT_LR, "steps": FT_STEPS, "time_cap_s": FT_TIME_CAP,
                      "batch": f"{NAME_BS} exposure + {ANCH_BS} anchor "
                               f"({ANCH_BS // 2} paired + {ANCH_BS // 2} random)",
                      "loss": "e043 token-level union CE (exposure mask = the "
                              "126 continuation cols; anchors full)",
                      "optimizer": "AdamW (0.9,0.95) wd 0.1 clip 1.0 constant lr",
                      "seed": CONS_SEED, "device": "cpu",
                      "arms": {t: {"desc": arms[t]["desc"],
                                   "steps_ran": arms[t]["steps_ran"],
                                   "traj": arms[t]["traj"]}
                               for t in arms}},
        "harvest_covariates": cov,
        "wpe_row_probes": wpe_probes,
        "address_sets": {t: {"d129": list(v["d129"]),
                             "d_all_addresses": list(v["d_all_addresses"]),
                             "grown_rows": v["grown_rows"]}
                         for t, v in addr_sets.items()},
        "battery_table": {f"{t}__{dl}__g{g_:+d}__{bt}":
                          table[(t, dl, g_, bt)]
                          for t in nets for dl in ("none", "d129",
                                                   "d_all_addresses")
                          for g_ in GEO_ORDER for bt in ("install60", "held30")},
        "ce_r_table": {f"{t}__{dl}": table[(t, dl, "ce_r", "-")]["ce_r"]
                       for t in nets for dl in ("none", "d129",
                                                "d_all_addresses")},
        "adjudication": adjudication,
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: dreams.png
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    net_order = ["base_no_ft", "a_own_dreams", "b_twin_dreams", "c_no_exposure"]
    net_lbl = {"base_no_ft": "base (no fine-tune)", "a_own_dreams": "(a) OWN dreams",
               "b_twin_dreams": "(b) TWIN dreams", "c_no_exposure": "(c) NO exposure"}
    net_col = {"base_no_ft": "tab:gray", "a_own_dreams": "crimson",
               "b_twin_dreams": "darkorange", "c_no_exposure": "steelblue"}
    xs = np.arange(len(GEO_ORDER))

    # (0,0) MAIN: arm x geometry, pre (none) vs post (D-all) deletion
    ax = axes[0, 0]
    bw = 0.8 / 8
    k = 0
    for t in net_order:
        pre = [table[(t, "none", g_, "install60")]["mean_pz"] for g_ in GEO_ORDER]
        post = [table[(t, "d_all_addresses", g_, "install60")]["mean_pz"] for g_ in GEO_ORDER]
        ax.bar(xs + (k - 3.5) * bw, pre, bw * 0.92, color=net_col[t],
               alpha=0.35, edgecolor="k", linewidth=0.3,
               label=f"{net_lbl[t]}: pre-del (light)")
        ax.bar(xs + (k - 2.5) * bw, post, bw * 0.92, color=net_col[t],
               edgecolor="k", linewidth=0.4,
               label=f"post-D-all (solid)")
        k += 2
    ax.axhline(BAR_SURVIVE, color="seagreen", ls="--", lw=1.1,
               label=f"survive bar {BAR_SURVIVE}")
    ax.axhline(BAR_COLLAPSE, color="gray", ls=":", lw=1.1,
               label=f"collapse bar {BAR_COLLAPSE}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"g{g_:+d}\n(row {129 + g_})" for g_ in GEO_ORDER],
                       fontsize=8)
    ax.set_ylabel("battery p(Z) (install-60)")
    ax.set_ylim(0, 1.05)
    ax.set_title("MAIN: arm x geometry, pre-deletion vs post-D-all "
                 "(light = pre, solid = post)", fontsize=10)
    ax.legend(fontsize=6.2, ncol=2, loc="upper left")

    # (0,1) post-D-all g0 with bootstrap CIs + D129 comparison
    ax = axes[0, 1]
    xs2 = np.arange(len(net_order))
    for k, t in enumerate(net_order):
        v = ci[t]["mean"]
        e = [[v - ci[t]["lo"]], [ci[t]["hi"] - v]]
        ax.bar(k - 0.19, v, 0.36, color=net_col[t], edgecolor="k", lw=0.4,
               yerr=e, capsize=3, error_kw={"lw": 1.0})
        v2 = ci_d129[t]["mean"]
        ax.bar(k + 0.19, v2, 0.36, color=net_col[t], alpha=0.45,
               edgecolor="k", lw=0.4)
        ax.text(k - 0.19, ci[t]["hi"] + 0.012, f"{v:.3f}", ha="center",
                fontsize=7)
        ax.text(k + 0.19, ci_d129[t]["mean"] + 0.012, f"{v2:.3f}", ha="center",
                fontsize=7)
    ax.axhline(BAR_SURVIVE, color="seagreen", ls="--", lw=1.1)
    ax.axhline(BAR_COLLAPSE, color="gray", ls=":", lw=1.1)
    ax.set_xticks(xs2)
    ax.set_xticklabels([net_lbl[t].replace(" (", "\n(") for t in net_order],
                       fontsize=8)
    ax.set_ylabel("post-deletion p(Z), geometry 0 (install-60)")
    ax.set_ylim(0, 1.0)
    ax.set_title("solid = post-D-ALL (95% bootstrap CI) | light = post-D129",
                 fontsize=10)

    # (1,0) harvest: name-rate + dream stats + wpe grown rows
    ax = axes[1, 0]
    axr = ax.twinx()
    tags_h = ["own", "twin"]
    rates = [cov[t]["per_10k_chars"]["ZEPHYRA"] for t in tags_h]
    zch = [cov[t]["per_10k_chars"]["Z_chars"] for t in tags_h]
    xs3 = np.arange(2)
    ax.bar(xs3 - 0.17, rates, 0.34, color="crimson", edgecolor="k", lw=0.4,
           label="ZEPHYRA / 10k chars")
    ax.bar(xs3 + 0.17, zch, 0.34, color="pink", edgecolor="k", lw=0.4,
           label="Z chars / 10k chars")
    ax.set_xticks(xs3)
    ax.set_xticklabels([f"{t} dreams\n({cov[t]['n_windows']} x {DREAM_LEN} tok)"
                        for t in tags_h], fontsize=8)
    ax.set_ylabel("harvest name-rate (per 10k chars)")
    ax.legend(fontsize=7, loc="upper left")
    rows_x = np.arange(len(ADDR_BAND))
    for k, t in enumerate(net_order):
        dn = [wpe_probes[t][str(r)]["d_norm"] for r in ADDR_BAND]
        axr.plot(rows_x, dn, "o-", ms=3, lw=1.0, color=net_col[t],
                 label=f"{net_lbl[t]} d_norm")
    axr.axhline(GROWN_D_NORM, color="k", ls=":", lw=0.9)
    axr.set_ylabel("wpe row d_norm vs base (band 121-137)")
    axr.legend(fontsize=6.2, loc="upper right")
    ax.set_title("harvest covariates (not a gate) + grown-row detection "
                 f"(threshold {GROWN_D_NORM})", fontsize=9)

    # (1,1) verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    a_ = cd[("a_own_dreams", "d_all_addresses")]
    b_ = cd[("b_twin_dreams", "d_all_addresses")]
    c_ = cd[("c_no_exposure", "d_all_addresses")]
    lines = [
        "E121 THE DREAMS PROBE — base e082_b43_install (p(Z) 0.3198 gate "
        f"{'PASS' if G_INST['pass'] else 'FAIL'})",
        "",
        f"harvest: own {cov['own']['zephyra_count']} ZEPHYRA / "
        f"{cov['own']['n_chars']} chars | twin {cov['twin']['zephyra_count']} / "
        f"{cov['twin']['n_chars']}",
        "",
        "post-D-all install-60 (95% CI, geometry 0):",
        f"  (a) own  {ci['a_own_dreams']['mean']:.3f} "
        f"[{ci['a_own_dreams']['lo']:.3f}, {ci['a_own_dreams']['hi']:.3f}] "
        f"max_g {a_['max']:.3f} n>=.20 {a_['n_ge_020']}/5",
        f"  (b) twin {ci['b_twin_dreams']['mean']:.3f} "
        f"[{ci['b_twin_dreams']['lo']:.3f}, {ci['b_twin_dreams']['hi']:.3f}] "
        f"max_g {b_['max']:.3f} n>=.20 {b_['n_ge_020']}/5",
        f"  (c) none {ci['c_no_exposure']['mean']:.3f} "
        f"max_g {c_['max']:.3f} collapses {c_coll}",
        f"  base    {ci['base_no_ft']['mean']:.3f} "
        f"[{ci['base_no_ft']['lo']:.3f}, {ci['base_no_ft']['hi']:.3f}] "
        "(live no-ft reference)",
        "",
        f"a>b CI-separated: {a_gt_b_ci} | a survives: {a_surv} | "
        f"b survives: {b_surv} | c collapses: {c_coll}",
        f"address sets: " + " | ".join(
            f"{t.replace('_no_ft', '')}: D-all {{{','.join(map(str, addr_sets[t]['d_all_addresses']))}}}"
            for t in net_order),
        "",
        f"FIRED BAR: {fired}",
    ]
    ax.text(0.02, 0.97, "\n".join(lines), va="top", ha="left", fontsize=8.0,
            family="monospace", transform=ax.transAxes,
            bbox=dict(facecolor="lightyellow", alpha=0.94, edgecolor="gray"))

    fig.suptitle(f"E121 — THE DREAMS PROBE: do the net's own free-run dreams "
                 f"consolidate the fact? -> {fired}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "dreams.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'dreams.png'}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
