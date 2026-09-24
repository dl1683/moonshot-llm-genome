"""E023 — surgical forgetting at entity granularity: letter-row surgery vs
entity-granular ascent (REGISTERED).

Design: scratch/e023_design.md (2026-09-24, built from measured probes on the
real E001 checkpoint). Answers T010 card C5's "next family" pointer and cashes
the tombstone inheritance (lm_head row directions are causal coordinates).

Target JULIET (125 train / 0 val; memorized, NLL/char ~0.35). Anchor PROSPERO
(0 train / 63 val, NLL ~13.2). Granularity ladder: D1 = full letter set
{J,U,L,I,E,T} (6 rows x 2 matrices = 2,304 params) vs D2 = rare core {J}
(1 row x 2 = 384 params).
Arms: A = zero/resample/shuffle(J<->Q) x both/wte-only/lm-only;
      B = span projection onto non-target uppercase rows;
      C = entity-window ascent, e003c protocol verbatim, 600 steps, ckpt'd.
Readouts R1 name battery + J-census, R2 corpus CE channels (val-All seed 202,
J-free, JULIET- vs ROMEO-anchored windows, PETRUCHIO-containing vs -free val
blocks), R3 generation (12 fixed prompts, 350 tok, temp 0.8, top-k 40, torch
seed = prompt index), R4 row geometry. Frozen selectivity S_name / S_letter /
S_corpus; bars Bar-1 (+2.0 nats/char) and Bar-2 (NLL >= ln65 AND acc <= 0.10);
predictions P1-P3; revive triggers; gates G0-G4. Outputs runs/e023/.

DEVIATIONS / SPEC-GAP FILLS (all registered cells present, none dropped):
 1. D1-shuffle partner map is unspecified in the design beyond J<->Q: J<->Q is
    kept; U,L,I,E,T are each paired with the unused non-target UPPERCASE
    letter of nearest train frequency (alphabetical tie-break), computed
    in-run and recorded in metrics.json (arm_ab.d1_shuffle_partners).
 2. Resample seeds: wte rows use seed 23100+row (as registered); lm rows use
    23200+row (one seed family is named; distinct draws per matrix are the
    only sane reading of an untied wte/lm pair).
 3. R1 battery = first 125 train occurrences per name (design: "every train
    occurrence (<=125)"); J-census uncapped (its max is JULIET's 125 anyway).
    PROSPERO (0 train occurrences) uses its 63 VAL occurrences — it is the
    anchor by definition. Computed overlap o(PROSPERO)=0.20 vs the design
    table's 0.17 (PROSPERO is excluded from the P2a Spearman set either way;
    all 8 in-range o values match the table exactly).
 4. Added D2/zero/lm to the generation-probe set (P2b's "J vanishes from
    generation while battery acc stays high" is otherwise untestable); all 4
    registered gen cells run. D2-proj-wte and D2-proj-lm also run (design:
    "if time"; time allowed). "Nurse"/"NURSE" counted as a free non-J R&J
    generation control.
 5. Arm C window index: one 256-char window anchor per JULIET occurrence,
    anchor offset ~ Uniform{0..250} with seed 23001 (reading of "window index
    built once, seed 23001"); sampler draws window indices with replacement,
    generator seed 23002. Every window contains >= 1 full JULIET occurrence.
 6. Occurrence matching is word-bounded (no adjacent alphabetic char),
    case-sensitive; battery forwards are batched in chunks of 128 (equivalent
    to the design's "one forward each", bit-deterministic).
 7. NOTES.md entry NOT written (operator instruction: no NOTES/THINKING/
    QUEUE/STATE edits in this run). No git commit.
 8. Registered runtime fallbacks implemented as time-triggered paths; whether
    each fired is recorded in metrics.timing.
 9. G4: the registered [10,16] PROSPERO range was calibrated on a design-MP
    probe that turned out context-construction-sensitive; under the registered
    battery (val occurrences, local val ctx 120) the anchor reads ~ln65
    (uniform floor = zero knowledge). Recorded as failed-with-explanation;
    all anchor USES (no-knowledge reference, ascent early-stop 13.2) unaffected.

Run: python lab/e023_surgical_forgetting.py   (requires runs/checkpoints/e001.pt)
E023_SMOKE=1 runs a fast end-to-end shakedown (separate outputs, not the
registered run).
"""
from __future__ import annotations

import copy
import math
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict,
                    estimate_loss, generate, run_dir, save_json, set_seed)

SMOKE = os.environ.get("E023_SMOKE") == "1"

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
ASCENT_CKPT = REPO / "runs" / "checkpoints" / ("e023_ascent_smoke.pt" if SMOKE else "e023_ascent.pt")
D2ZB_CKPT = REPO / "runs" / "checkpoints" / ("e023_d2_zero_both_smoke.pt" if SMOKE else "e023_d2_zero_both.pt")

SEED = 23000
CTX = 120                 # name-battery context
BATT_CAP = 125            # first N train occurrences per battery name
EVAL_BS = 64
BLOCK = 256
N_BLOCKS = 400            # e003c convention

# arm C (e003c protocol verbatim except the batch source)
LR = 1e-5
STEPS = 600
EVAL_EVERY = 50
ASC_BATCH = 32
CKPT_EVERY = 100
STOP_NLL = 13.2           # anchor-level early stop

BAR1 = 2.0                # nats/char above this run's baseline (partial forgetting)
BAR2_NLL = math.log(65)   # 4.174 — uniform-floor erasure
BAR2_ACC = 0.10
FLOOR_CE = 0.01           # significance floor, dCE_corpus
FLOOR_NAME = 0.10         # significance floor, dNLL_name

BATTERY = ["JULIET", "JOHN", "ROMEO", "GLOUCESTER", "MENENIUS",
           "CORIOLANUS", "ISABELLA", "LUCIO", "PETRUCHIO", "PROSPERO"]
PURE_CONTROLS = ["ROMEO", "GLOUCESTER", "CORIOLANUS"]
P2A_CONTROLS = ["JOHN", "ROMEO", "GLOUCESTER", "MENENIUS", "CORIOLANUS",
                "ISABELLA", "LUCIO", "PETRUCHIO"]      # 8 controls, o in [0.25, 0.60]
CENSUS = ["JULIET", "Juliet", "JOHN", "John", "Jove", "Jack", "Jesu",
          "Justice", "Jupiter", "Join", "Juno", "Julius"]
TARGET_LETTERS_D1 = list("JULIET")
TARGET_LETTERS_D2 = ["J"]
TARGET_SET = set("JULIET")

RJ_LO, RJ_HI = 458053, 886378    # measured JULIET region in train
E001_VAL_CE = 1.622391           # e001 training-history val CE (G0 reference)

if SMOKE:
    STEPS, EVAL_EVERY, CKPT_EVERY = 6, 3, 3
    N_GEN_PROMPTS, GEN_TOK = 2, 30
else:
    N_GEN_PROMPTS, GEN_TOK = 12, 350


# ------------------------------------------------------------------ helpers

def find_occ(text: str, word: str) -> list[int]:
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])")
    return [m.start() for m in pat.finditer(text)]


def overlap_o(name: str) -> float:
    u = set(name)
    return len(u & TARGET_SET) / len(u)


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 0 else float("nan")


def fixed_blocks(src: torch.Tensor, block: int, n: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(src) - block - 1, (n,), generator=gen)
    x = torch.stack([src[i: i + block] for i in ix])
    y = torch.stack([src[i + 1: i + 1 + block] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def ce_fixed(model: TinyGPT, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), EVAL_BS):
        xb, yb = x[i: i + EVAL_BS], y[i: i + EVAL_BS]
        _, loss = model(xb, yb)
        tot += float(loss.item()) * len(xb)
        n += len(xb)
    return tot / max(n, 1)


def sample_windows(src_ids, src_text, n, seed, accept):
    """Deterministically sample block anchors accepted by pred on the text window."""
    gen = torch.Generator().manual_seed(seed)
    starts, tries = [], 0
    while len(starts) < n and tries < 500 * n:
        i = int(torch.randint(len(src_ids) - BLOCK - 1, (1,), generator=gen))
        tries += 1
        if accept(src_text[i: i + BLOCK + 1]):
            starts.append(i)
    x = torch.stack([src_ids[s: s + BLOCK] for s in starts])
    y = torch.stack([src_ids[s + 1: s + 1 + BLOCK] for s in starts])
    return x.to(DEVICE), y.to(DEVICE), len(starts)


def anchored_windows(src_ids, occs, back, n):
    starts = [p - back for p in occs
              if p >= back and p + BLOCK + 1 <= len(src_ids)][:n]
    x = torch.stack([src_ids[s: s + BLOCK] for s in starts])
    y = torch.stack([src_ids[s + 1: s + 1 + BLOCK] for s in starts])
    return x.to(DEVICE), y.to(DEVICE), len(starts)


def build_seq_bat(ids: torch.Tensor, text: str, stoi, words, cap=None):
    bats = {}
    for w in words:
        all_occ = find_occ(text, w)
        occs = all_occ[:cap] if cap else all_occ
        keep = [p for p in occs if p >= CTX]
        wid = torch.tensor([stoi[c] for c in w], dtype=torch.long)
        seqs = [torch.cat([ids[p - CTX: p], wid]) for p in keep]
        bats[w] = {
            "seq": torch.stack(seqs) if seqs else None,
            "n": len(keep), "n_all": len(all_occ),
            "n_skipped_ctx": len(occs) - len(keep),
        }
    return bats


@torch.no_grad()
def eval_bats(model: TinyGPT, bats: dict, jid: int) -> dict:
    """ctx-CTX battery: per-name NLL/char, argmax acc, per-position profiles,
    first-char vs chars-2+ split, mean logit of 'J' at J-target positions."""
    model.eval()
    V = model.cfg.vocab
    out = {}
    for w, b in bats.items():
        if b["seq"] is None or b["n"] == 0:
            out[w] = {"n": 0}
            continue
        L = len(w)
        seq = b["seq"].to(DEVICE)
        x, y = seq[:, :-1], seq[:, 1:]
        nlls, accs = [], []
        lj_sum, lj_n = 0.0, 0
        for i in range(0, len(seq), 128):
            logits, _ = model(x[i: i + 128])
            lg = logits[:, CTX - 1: CTX - 1 + L, :]      # logits[119+i] predicts name[i]=seq[120+i]
            tg = y[i: i + 128][:, CTX - 1: CTX - 1 + L]
            nll = F.cross_entropy(lg.reshape(-1, V), tg.reshape(-1),
                                  reduction="none").view(-1, L)
            acc = (lg.argmax(-1) == tg).float()
            nlls.append(nll)
            accs.append(acc)
            m = tg == jid
            if m.any():
                lj_sum += float(lg[:, :, jid][m].sum().item())
                lj_n += int(m.sum().item())
        nll_m, acc_m = torch.cat(nlls), torch.cat(accs)
        out[w] = {
            "n": int(nll_m.shape[0]),
            "nll": float(nll_m.mean().item()),
            "acc": float(acc_m.mean().item()),
            "per_pos_acc": [float(v) for v in acc_m.mean(0).tolist()],
            "per_pos_nll": [float(v) for v in nll_m.mean(0).tolist()],
            "first_char_nll": float(nll_m[:, 0].mean().item()),
            "rest_nll": float(nll_m[:, 1:].mean().item()) if L > 1 else None,
            "logit_J_at_J_pos": (lj_sum / lj_n) if lj_n else None,
        }
    return out


# ------------------------------------------------------------------ surgery

def proj_rows_for(base_sd, letters_span, target_letters, key):
    """Orthonormal-projection replacement rows: r' = Qb Qb^T r onto the span
    of the non-target uppercase rows (float64 CPU for determinism)."""
    W = base_sd[key]
    rows = W[[ch_row[c] for c in letters_span], :].to(torch.float64).cpu()   # (k, d)
    Qb, _ = torch.linalg.qr(rows.T)                                          # (d, k)
    out = {}
    for ch in target_letters:
        r = W[ch_row[ch]].to(torch.float64).cpu()
        out[ch] = (Qb @ (Qb.T @ r)).to(W.dtype).to(W.device)
    return out


def surgery_sd(base_sd, letters, variant, matrix, partner, proj_map):
    sd = {k: v.clone() for k, v in base_sd.items()}
    mats = {"both": ("wte.weight", "lm_head.weight"),
            "wte": ("wte.weight",), "lm": ("lm_head.weight",)}[matrix]
    for key in mats:
        W = sd[key]
        for ch in letters:
            r = int(ch_row[ch])
            if variant == "zero":
                W[r] = 0.0
            elif variant == "resample":
                g = torch.Generator().manual_seed(
                    (23100 if key.startswith("wte") else 23200) + r)
                W[r] = (torch.randn(W.shape[1], generator=g) * 0.02).to(W.device)
            elif variant == "shuffle":
                q = int(ch_row[partner[ch]])
                tmp = W[r].clone()
                W[r] = W[q].clone()
                W[q] = tmp
            elif variant == "proj":
                W[r] = proj_map[(key, ch)]
            else:
                raise ValueError(variant)
    return sd


def g2_check(base_sd, sd, expected_elements, expected_rows, n_embd):
    rows_changed, total, confined = {}, 0, True
    for key in ("wte.weight", "lm_head.weight"):
        d = sd[key] != base_sd[key]
        nd = int(d.sum().item())
        total += nd
        rows = sorted(set(torch.nonzero(d)[:, 0].tolist()))
        rows_changed[key] = rows
        if nd != len(rows) * n_embd:
            confined = False
    others = all(torch.equal(sd[k], base_sd[k])
                 for k in sd if k not in ("wte.weight", "lm_head.weight"))
    want_rows = {k: sorted(expected_rows.get(k, [])) for k in rows_changed}
    rows_exact = all(rows_changed[k] == want_rows[k] for k in rows_changed)
    return {"rows_changed": rows_changed, "expected_rows": want_rows,
            "n_elements_changed": total, "expected_elements": expected_elements,
            "confined_to_full_rows": confined,
            "others_bit_identical": bool(others), "rows_exact": bool(rows_exact),
            "pass": bool(total == expected_elements and confined and others and rows_exact)}


# ------------------------------------------------------------------ generation

GEN_CELLS = ["D2/zero/both", "D2/resample/lm", "D2/shuffle/both", "D1/zero/both",
             "D2/zero/lm"]     # 4 registered + zero-lm (deviation 4)


def run_gen(model, corpus, prompts, tag, probes_path, n_tok):
    """12 fixed prompts (6 R&J-region, 6 generic), temp 0.8, top-k 40,
    torch seed = prompt index. Counts on the GENERATED continuation only."""
    counts = {k: 0 for k in ["JULIET", "ROMEO", "JOHN", "Nurse/NURSE",
                             "J_chars", "QULIET", "Q_words", "chars"]}
    per_group = {"rj": dict(counts), "generic": dict(counts)}
    lines = [f"\n{'=' * 70}\nGENERATION PROBES — {tag}  ({len(prompts)} prompts x {n_tok} tok, "
             f"temp 0.8, top-k 40, seed=prompt index)\n{'=' * 70}"]
    for i, prompt in enumerate(prompts):
        torch.manual_seed(i)
        out = generate(model, corpus, prompt, max_new_tokens=n_tok,
                       temperature=0.8, top_k=40)
        cont = out[len(prompt):]
        grp = "rj" if i < 6 else "generic"
        for bag in (counts, per_group[grp]):
            bag["chars"] += len(cont)
            bag["J_chars"] += cont.count("J")
            bag["QULIET"] += cont.count("QULIET")
            bag["Q_words"] += len(re.findall(r"(?<![A-Za-z])Q[a-z]+", cont))
            for w, k in [("JULIET", "JULIET"), ("ROMEO", "ROMEO"), ("JOHN", "JOHN")]:
                bag[k] += cont.count(w)
            bag["Nurse/NURSE"] += cont.count("Nurse") + cont.count("NURSE")
        lines.append(f"\n--- [{tag}] prompt {i} ({grp}) seed={i} ---\nPROMPT: {prompt}\n"
                     f"GEN:    {cont}")
    probes_path.write_text(probes_path.read_text(encoding="utf-8") + "\n".join(lines) + "\n",
                           encoding="utf-8")
    def rates(bag):
        r = {k: (round(v / bag["chars"] * 1e4, 2) if bag["chars"] else None)
             for k, v in bag.items() if k != "chars"}
        return {"counts": {k: v for k, v in bag.items() if k != "chars"},
                "chars": bag["chars"], "per_10k_chars": r}
    return {"total": rates(counts), "rj_prompts": rates(per_group["rj"]),
            "generic_prompts": rates(per_group["generic"])}


# ------------------------------------------------------------------ verdict utils

def interp_cross(pairs, level):
    """First-crossing linear interpolation of (dose, value) pairs."""
    prev = None
    for d, v in pairs:
        if prev is not None and prev[0] < level <= d:
            f = (level - prev[0]) / max(d - prev[0], 1e-9)
            return prev[1] + f * (v - prev[1])
        prev = (d, v)
    return None


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if math.isnan(x):
            return "nan"
        return x
    if isinstance(x, torch.Tensor):
        return jsonable(x.tolist())
    return str(x)


# ------------------------------------------------------------------ main

ch_row: dict[str, int] = {}


def main():
    T0 = time.time()
    stamp = lambda: f"[{time.time() - T0:7.1f}s]"
    set_seed(SEED)
    rd = run_dir("e023_smoke" if SMOKE else "e023")
    probes_path = rd / "probes.txt"
    probes_path.write_text("", encoding="utf-8")

    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    base = TinyGPT(cfg).to(DEVICE)
    base.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    stoi, itos = corpus.stoi, corpus.itos
    for c, i in stoi.items():
        ch_row[c] = int(i)
    jid = stoi["J"]
    qid = stoi["Q"]
    N_EMBD = cfg.n_embd

    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    uppercase = sorted([c for c in stoi if c.isupper()])
    assert len(uppercase) == 26

    # ---------------------------------------------------------------- P0: batteries
    print(f"{stamp()} P0: building batteries (seed-stamped)", flush=True)
    name_bats = {}
    for w in BATTERY:
        ids, txt = (val_ids, val_text) if w == "PROSPERO" else (train_ids, train_text)
        name_bats.update(build_seq_bat(ids, txt, stoi, [w], cap=BATT_CAP))
    census_bats = build_seq_bat(train_ids, train_text, stoi, CENSUS, cap=None)

    design_check = {w: {"train_all": name_bats[w]["n_all"] if w != "PROSPERO" else
                        len(find_occ(train_text, w)),
                        "val_all": len(find_occ(val_text, w)),
                        "battery_n": name_bats[w]["n"]}
                    for w in BATTERY}
    assert len(find_occ(train_text, "JULIET")) == 125, "target integrity"
    assert len(find_occ(train_text, "PROSPERO")) == 0

    r2_blocks = {}
    r2_blocks["val_all"] = fixed_blocks(val_ids, BLOCK, N_BLOCKS, seed=202)
    xf, yf, njf = sample_windows(val_ids, val_text, N_BLOCKS, 23103,
                                 lambda t: "J" not in t)
    r2_blocks["val_jfree"] = (xf, yf)
    jul_occs = find_occ(train_text, "JULIET")
    rom_occs = find_occ(train_text, "ROMEO")
    xw, yw, nw1 = anchored_windows(train_ids, jul_occs, 128, 125)
    r2_blocks["jul_windows"] = (xw, yw)
    xw, yw, nw2 = anchored_windows(train_ids, rom_occs, 128, 125)
    r2_blocks["rom_windows"] = (xw, yw)
    petr_val = find_occ(val_text, "PETRUCHIO")
    xw, yw, np1 = anchored_windows(val_ids, petr_val, 128, 400)
    r2_blocks["petr_contain"] = (xw, yw)
    xf, yf, np2 = sample_windows(val_ids, val_text, N_BLOCKS, 23104,
                                 lambda t: "PETRUCHIO" not in t)
    r2_blocks["petr_free"] = (xf, yf)
    r2_meta = {"val_jfree_n": njf, "jul_windows_n": nw1, "rom_windows_n": nw2,
               "petr_contain_n": np1, "petr_free_n": np2,
               "petr_val_occs": len(petr_val)}
    print(f"{stamp()} R2 channels: {r2_meta}", flush=True)

    # prompts: 6 evenly spaced R&J-region train offsets + 6 generic train offsets
    prompts = []
    for k in range(6):
        s = RJ_LO + k * (RJ_HI - 128 - RJ_LO) // 5
        prompts.append(train_text[s: s + 128])
    for k in range(6):
        s = k * (RJ_LO - 128) // 5
        prompts.append(train_text[s: s + 128])

    def eval_r2(model):
        return {k: ce_fixed(model, x, y) for k, (x, y) in r2_blocks.items()}

    # ---------------------------------------------------------------- P0: baselines + gates
    g_val = estimate_loss(base, corpus, "val", n_batches=20)
    G0 = {"val_ce_estimate_loss": g_val, "ref": E001_VAL_CE,
          "pass": bool(abs(g_val - E001_VAL_CE) <= 0.03)}
    print(f"{stamp()} G0 val CE {g_val:.4f} vs {E001_VAL_CE} -> {G0['pass']}", flush=True)

    r1_base = eval_bats(base, name_bats, jid)
    r1_base_2 = eval_bats(base, name_bats, jid)
    G1 = bool(all(r1_base[w][k] == r1_base_2[w][k]
                  for w in r1_base if r1_base[w].get("n")
                  for k in ("nll", "acc", "first_char_nll", "rest_nll")))
    print(f"{stamp()} G1 baseline battery bit-identical: {G1}", flush=True)
    r1c_base = eval_bats(base, census_bats, jid)
    ce_base = eval_r2(base)
    print(f"{stamp()} baselines: JULIET {r1_base['JULIET']['nll']:.3f}/"
          f"{r1_base['JULIET']['acc']:.2f}  JOHN {r1_base['JOHN']['nll']:.3f}  "
          f"ROMEO {r1_base['ROMEO']['nll']:.3f}  PROSPERO "
          f"{r1_base['PROSPERO']['nll']:.2f}  val_all {ce_base['val_all']:.4f}", flush=True)

    G4 = {"prospero_nll": r1_base["PROSPERO"]["nll"],
          "prospero_acc": r1_base["PROSPERO"]["acc"],
          "pass": bool(10.0 <= r1_base["PROSPERO"]["nll"] <= 16.0)}
    print(f"{stamp()} G4 PROSPERO anchor NLL {G4['prospero_nll']:.2f} -> {G4['pass']}", flush=True)
    if not G4["pass"]:
        G4["note"] = ("FAILED against the registered [10,16] range, but the anchor's "
                      "SUBSTANCE holds: 0 train occurrences, acc 0.31, NLL ~ ln65=4.17 "
                      "(uniform floor = zero knowledge). The design's MP 13.23 was "
                      "context-construction-sensitive (diagnosed 2026-09-24: after a "
                      "confident train mid-sentence context the same name reads 9.0 "
                      "nats/char, after a speaker-slot context 4.4; the registered "
                      "battery uses LOCAL val contexts where the model is already "
                      "uncertain on unseen Tempest text).")

    # R4 row geometry (descriptive)
    span_d2 = [c for c in uppercase if c != "J"]
    span_d1 = [c for c in uppercase if c not in TARGET_LETTERS_D1]
    r4 = {}
    for key in ("wte.weight", "lm_head.weight"):
        W = base_sd[key]
        norms = W.norm(dim=1)
        r4[key] = {
            "mean_row_norm": float(norms.mean().item()),
            "norms_uppercase": {c: float(norms[ch_row[c]].item()) for c in uppercase},
            "cos_J_Q": float(F.cosine_similarity(W[jid], W[qid], dim=0).item()),
            "cos_target_vs_uppercase": {
                c: {d: float(F.cosine_similarity(W[ch_row[c]], W[ch_row[d]], dim=0).item())
                    for d in uppercase} for c in TARGET_LETTERS_D1},
        }
    proj_rows = {}
    for tag, span, lets in (("D2", span_d2, TARGET_LETTERS_D2),
                            ("D1", span_d1, TARGET_LETTERS_D1)):
        for key in ("wte.weight", "lm_head.weight"):
            pm = proj_rows_for(base_sd, span, lets, key)
            proj_rows.update({(tag, key): pm})
            for ch, newr in pm.items():
                r = base_sd[key][ch_row[ch]]
                res = float((r - newr).norm().item()) / float(r.norm().item())
                r4.setdefault("residual_frac_outside_span", {})[f"{tag}/{key}/{ch}"] = res
    print(f"{stamp()} R4: |r_J| wte {r4['wte.weight']['norms_uppercase']['J']:.3f} "
          f"(mean {r4['wte.weight']['mean_row_norm']:.3f})  lm "
          f"{r4['lm_head.weight']['norms_uppercase']['J']:.3f} "
          f"(mean {r4['lm_head.weight']['mean_row_norm']:.3f})  cos(J,Q) lm "
          f"{r4['lm_head.weight']['cos_J_Q']:.2f}  residuals "
          f"{ {k: round(v, 3) for k, v in r4['residual_frac_outside_span'].items() if k.startswith('D2')} }",
          flush=True)

    # base generation
    gen_base = run_gen(base, corpus, prompts[:N_GEN_PROMPTS], "base", probes_path, GEN_TOK)
    print(f"{stamp()} R3 base gen: JULIET {gen_base['total']['counts']['JULIET']} "
          f"ROMEO {gen_base['total']['counts']['ROMEO']} J-chars "
          f"{gen_base['total']['counts']['J_chars']} per "
          f"{gen_base['total']['chars']} chars", flush=True)

    # ---------------------------------------------------------------- arm A + B cells
    print(f"{stamp()} arms A+B: surgery ladder", flush=True)
    partner_d2 = {"J": "Q"}
    freq_up = {c: train_text.count(c) for c in uppercase}
    partner_d1 = {"J": "Q"}
    used = {"Q"}
    for ch in [c for c in TARGET_LETTERS_D1 if c != "J"]:
        cands = sorted([c for c in uppercase
                        if c not in TARGET_LETTERS_D1 and c not in used],
                       key=lambda c: (abs(freq_up[c] - freq_up[ch]), c))
        partner_d1[ch] = cands[0]
        used.add(cands[0])
    print(f"{stamp()} D1 shuffle partners: {partner_d1}", flush=True)

    cells = []
    for depth, lets in (("D2", TARGET_LETTERS_D2), ("D1", TARGET_LETTERS_D1)):
        for variant in ("zero", "resample", "shuffle"):
            mats = ["both", "wte", "lm"] if depth == "D2" else ["both"]
            for matrix in mats:
                cells.append({"depth": depth, "letters": lets, "variant": variant,
                              "matrix": matrix,
                              "partner": partner_d2 if depth == "D2" else partner_d1})
    for depth, lets, tag in (("D2", TARGET_LETTERS_D2, "D2"), ("D1", TARGET_LETTERS_D1, "D1")):
        mats = ["both", "wte", "lm"] if depth == "D2" else ["both"]
        pmap = {}
        for (tg, mkey), d in proj_rows.items():
            if tg == tag:
                for ch, row in d.items():
                    pmap[(mkey, ch)] = row
        for matrix in mats:
            cells.append({"depth": depth, "letters": lets, "variant": "proj",
                          "matrix": matrix, "partner": None, "proj": pmap})

    cell_results: dict[str, dict] = {}
    d2zb_sd = None
    for cell in cells:
        key = f"{cell['depth']}/{cell['variant']}/{cell['matrix']}"
        if SMOKE and key not in ("D2/zero/both", "D2/zero/wte", "D2/zero/lm",
                                 "D1/zero/both", "D2/proj/both"):
            continue
        letters = cell["letters"]
        variant, matrix = cell["variant"], cell["matrix"]
        pm = cell.get("proj") or {}
        sd = surgery_sd(base_sd, letters, variant, matrix,
                        cell["partner"] or {}, pm)
        affected = set(letters)
        if variant == "shuffle":
            affected |= {cell["partner"][c] for c in letters}
        mats = {"both": ("wte.weight", "lm_head.weight"), "wte": ("wte.weight",),
                "lm": ("lm_head.weight",)}[matrix]
        exp_rows = {k: [ch_row[c] for c in sorted(affected)] for k in mats}
        exp_elem = len(affected) * len(mats) * N_EMBD
        g2 = g2_check(base_sd, sd, exp_elem, exp_rows, N_EMBD)
        assert g2["pass"], f"G2 FAILED for {key}: {g2}"
        if key == "D2/zero/both":
            d2zb_sd = sd
            torch.save(sd, D2ZB_CKPT)
        m = copy.deepcopy(base)
        m.load_state_dict(sd)
        r1 = eval_bats(m, name_bats, jid)
        r1c = eval_bats(m, census_bats, jid)
        ce = eval_r2(m)
        dnll = {w: (r1[w]["nll"] - r1_base[w]["nll"]) for w in BATTERY if r1[w].get("n")}
        dce = {k: ce[k] - ce_base[k] for k in ce}
        dj = dnll["JULIET"]
        pure = [dnll[w] for w in PURE_CONTROLS]
        mpure = sum(pure) / len(pure)
        s_name = dj / mpure if mpure > 1e-9 else float("inf")
        s_letter = dj / dnll["JOHN"] if abs(dnll["JOHN"]) > 1e-9 else float("inf")
        dva = dce["val_all"]
        s_corpus = dj / dva if dva >= FLOOR_CE else None
        s_corpus_bound = dj / FLOOR_CE if dj > 0 else None
        res = {
            "letters": letters, "variant": variant, "matrix": matrix,
            "partner": cell["partner"], "g2": g2,
            "r1": {w: {"nll": r1[w]["nll"], "acc": r1[w]["acc"],
                       "n": r1[w]["n"]} for w in BATTERY},
            "juliet_per_pos_acc": r1["JULIET"]["per_pos_acc"],
            "juliet_logit_J": r1["JULIET"]["logit_J_at_J_pos"],
            "census": {w: {"nll": r1c[w]["nll"], "acc": r1c[w]["acc"]} for w in CENSUS},
            "r2": ce, "d_nll": dnll, "d_ce": dce,
            "s_name": s_name, "s_letter": s_letter,
            "s_corpus": s_corpus, "s_corpus_lower_bound": s_corpus_bound,
            "bar1": bool(dj >= BAR1),
            "bar2": bool(r1["JULIET"]["nll"] >= BAR2_NLL and r1["JULIET"]["acc"] <= BAR2_ACC),
            "bar2_nll_only": bool(r1["JULIET"]["nll"] >= BAR2_NLL),
        }
        cell_results[key] = res
        print(f"{stamp()} {key:18s} G2 ok ({g2['n_elements_changed']} elems) "
              f"JULIET {r1['JULIET']['nll']:6.2f}/{r1['JULIET']['acc']:.2f} "
              f"JOHN {r1['JOHN']['nll']:6.2f} ROMEO {r1['ROMEO']['nll']:5.2f} "
              f"dCE_val {dva:+.4f} S_name "
              f"{'inf' if math.isinf(s_name) else f'{s_name:8.1f}'} S_letter "
              f"{'inf' if math.isinf(s_letter) else f'{s_letter:5.2f}'} "
              f"bar1={int(res['bar1'])} bar2={int(res['bar2'])}", flush=True)
        if key in GEN_CELLS:
            gp = prompts[:N_GEN_PROMPTS]
            gt = GEN_TOK
            if not SMOKE and time.time() - T0 > 600:   # registered fallback
                gp, gt = gp[:8], 300
                print(f"{stamp()}   gen fallback 8x300 fired", flush=True)
            res["gen"] = run_gen(m, corpus, gp, key, probes_path, gt)
            gc = res["gen"]["total"]["counts"]
            print(f"{stamp()}   gen: JULIET {gc['JULIET']} J-chars {gc['J_chars']} "
                  f"QULIET {gc['QULIET']} Q-words {gc['Q_words']}", flush=True)
        del m

    # ---------------------------------------------------------------- arm C: ascent
    elapsed = time.time() - T0
    eval_every, steps_cap = EVAL_EVERY, STEPS
    fb = []
    if not SMOKE and elapsed > 630:
        eval_every, steps_cap = 75, 400
        fb.append("armC eval75/cap400")
        print(f"{stamp()} arm C fallback fired: eval_every=75 cap=400", flush=True)
    print(f"{stamp()} arm C: entity-window ascent (600 steps max, "
          f"windows seeded 23001/23002)", flush=True)

    gix = torch.Generator().manual_seed(23001)
    win_starts = []
    for p in jul_occs:
        u = int(torch.randint(0, 251, (1,), generator=gix))
        s = p - u
        if 0 <= s and s + BLOCK + 1 <= len(train_ids):
            win_starts.append(s)
    win_x = torch.stack([train_ids[s: s + BLOCK] for s in win_starts])
    win_y = torch.stack([train_ids[s + 1: s + 1 + BLOCK] for s in win_starts])
    print(f"{stamp()} window index: {len(win_starts)} JULIET-containing windows", flush=True)

    m = copy.deepcopy(base)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.95))
    sgen = torch.Generator().manual_seed(23002)
    traj, start = [], 0
    if ASCENT_CKPT.exists():
        st = torch.load(ASCENT_CKPT, map_location=DEVICE, weights_only=False)
        m.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sgen.set_state(st["gen_state"])
        start, traj = st["step"], st["traj"]
        print(f"{stamp()} resumed ascent from step {start}", flush=True)

    def ascent_eval(step):
        r1 = eval_bats(m, name_bats, jid)
        r1c = eval_bats(m, census_bats, jid)
        ce = eval_r2(m)
        pt = {"step": step,
              "names": {w: {"nll": r1[w]["nll"], "acc": r1[w]["acc"]} for w in BATTERY},
              "juliet": {"per_pos_acc": r1["JULIET"]["per_pos_acc"],
                         "logit_J": r1["JULIET"]["logit_J_at_J_pos"]},
              "census": {w: {"nll": r1c[w]["nll"], "acc": r1c[w]["acc"]} for w in CENSUS},
              "ce": ce}
        traj.append(pt)
        print(f"{stamp()} [ascent] s{step:4d} JULIET {r1['JULIET']['nll']:6.2f}/"
              f"{r1['JULIET']['acc']:.2f} JOHN {r1['JOHN']['nll']:5.2f} "
              f"ROMEO {r1['ROMEO']['nll']:5.2f} GLOU {r1['GLOUCESTER']['nll']:5.2f} "
              f"val_all {ce['val_all']:.4f} julwin {ce['jul_windows']:.3f} "
              f"logitJ {r1['JULIET']['logit_J_at_J_pos']:.1f}", flush=True)
        m.train()

    G3 = None
    if start == 0:
        ascent_eval(0)
        G3 = {"max_abs_diff": max(abs(traj[0]["names"][w][k] - r1_base[w][k])
                                  for w in BATTERY if r1_base[w].get("n")
                                  for k in ("nll", "acc"))}
        G3["pass"] = bool(G3["max_abs_diff"] == 0.0)
        if not G3["pass"] and G3["max_abs_diff"] < 1e-6:
            G3["pass_note"] = "within 1e-6, not bit-identical"
        print(f"{stamp()} G3 ascent step-0 == baseline: {G3}", flush=True)

    stop_reason = "step_cap"
    m.train()
    last_step = start
    for step in range(start + 1, steps_cap + 1):
        ix = torch.randint(len(win_starts), (ASC_BATCH,), generator=sgen)
        x = win_x[ix].to(DEVICE)
        y = win_y[ix].to(DEVICE)
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        (-loss).backward()                       # ascent: maximize window CE
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        last_step = step
        if step % eval_every == 0:
            ascent_eval(step)
            if traj[-1]["names"]["JULIET"]["nll"] >= STOP_NLL:
                stop_reason = "anchor_level_13.2"
                break
        if step % CKPT_EVERY == 0 or step == steps_cap:
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                        "gen_state": sgen.get_state(), "step": step,
                        "traj": traj, "stop_reason": None},
                       ASCENT_CKPT)
    if traj[-1]["step"] != last_step:
        ascent_eval(last_step)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                "gen_state": sgen.get_state(), "step": traj[-1]["step"],
                "traj": traj, "stop_reason": stop_reason}, ASCENT_CKPT)
    print(f"{stamp()} ascent done at step {traj[-1]['step']} ({stop_reason})", flush=True)

    gen_asc = run_gen(m, corpus, prompts[:N_GEN_PROMPTS], "ascent-final",
                      probes_path, GEN_TOK)

    # arm C derived quantities
    jul = lambda p: p["names"]["JULIET"]["nll"]
    john = lambda p: p["names"]["JOHN"]["nll"]
    cev = lambda p: p["ce"]["val_all"]
    jwin = lambda p: p["ce"]["jul_windows"]
    pure_mean = lambda p: sum(p["names"][w]["nll"] for w in PURE_CONTROLS) / 3
    logitJ = lambda p: p["juliet"]["logit_J"]

    r_points = []
    for p in traj:
        dt, dc = jul(p) - jul(traj[0]), cev(p) - cev(traj[0])
        r_points.append(dt / dc if (dt >= 0.05 and dc > 1e-4) else None)

    base_jul = jul(traj[0])
    cross = {}
    if any(jul(p) - base_jul >= BAR1 for p in traj):
        cross = {
            "step": None,
            "r_entity": None, "s_name": None, "s_letter": None,
            "dce_val": None, "dce_jul_windows": None, "logit_J": None,
        }
        dc_val_i = interp_cross([(jul(p) - base_jul, cev(p) - cev(traj[0])) for p in traj], BAR1)
        dc_jw_i = interp_cross([(jul(p) - base_jul, jwin(p) - jwin(traj[0])) for p in traj], BAR1)
        djohn_i = interp_cross([(jul(p) - base_jul, john(p) - john(traj[0])) for p in traj], BAR1)
        dpure_i = interp_cross([(jul(p) - base_jul, pure_mean(p) - pure_mean(traj[0])) for p in traj], BAR1)
        lJ_i = interp_cross([(jul(p) - base_jul, logitJ(p) - logitJ(traj[0])) for p in traj], BAR1)
        cross["dce_val"] = dc_val_i
        cross["dce_jul_windows"] = dc_jw_i
        cross["d_nll_john"] = djohn_i
        cross["d_nll_pure_mean"] = dpure_i
        cross["d_logit_J"] = lJ_i
        cross["r_entity"] = (BAR1 / dc_val_i) if (dc_val_i is not None and dc_val_i > 1e-6) else None
        cross["s_name"] = (BAR1 / dpure_i) if (dpure_i is not None and dpure_i > 1e-9) else float("inf")
        cross["s_letter"] = (BAR1 / djohn_i) if (djohn_i is not None and abs(djohn_i) > 1e-9) else float("inf")
        for p in traj:
            if jul(p) - base_jul >= BAR1:
                cross["step"] = p["step"]
                break
    bar1_reached = bool(cross)

    d2zb = cell_results.get("D2/zero/both")
    surg_cost_scaled = None
    coll_ratio = None
    if d2zb is not None and cross.get("dce_val") is not None:
        dnll_zb = d2zb["r1"]["JULIET"]["nll"] - r1_base["JULIET"]["nll"]
        surg_cost_scaled = d2zb["d_ce"]["val_all"] * (BAR1 / dnll_zb) if dnll_zb else None
        if surg_cost_scaled and surg_cost_scaled > 0:
            coll_ratio = cross["dce_val"] / surg_cost_scaled

    # revive triggers (any one reopens C5)
    revive = {}
    revive["1_context_restriction"] = bool(
        bar1_reached and cross["r_entity"] is not None
        and cross["r_entity"] >= 1.3 and cross["dce_val"] is not None
        and cross["dce_val"] <= 0.10)
    trig2 = False
    for p in traj:
        if 0 < p["step"] <= 100:
            dj, dc = jul(p) - base_jul, cev(p) - cev(traj[0])
            dp = pure_mean(p) - pure_mean(traj[0])
            if dj >= BAR1 and dc < 0.05 and dp < 0.3:
                trig2 = True
    revive["2_early_selective_transient"] = bool(trig2)
    trig3_step = None
    for p in traj:
        n = p["names"]["JULIET"]
        if (n["nll"] >= BAR2_NLL and n["acc"] <= BAR2_ACC
                and cev(p) - cev(traj[0]) <= 0.5):
            trig3_step = p["step"]
            break
    revive["3_bar2_with_low_collateral"] = trig3_step

    # ---------------------------------------------------------------- verdicts
    def sname_ok(res):
        return (math.isinf(res["s_name"]) and res["s_name"] > 0) or res["s_name"] > 50

    d2zb_res = cell_results.get("D2/zero/both")
    d2rl_res = cell_results.get("D2/resample/lm")
    d1zb_res = cell_results.get("D1/zero/both")
    p1 = {"cells": {}}
    if d2zb_res:
        a = {}
        a["bar2"] = d2zb_res["bar2"]
        a["dce_val"] = d2zb_res["d_ce"]["val_all"]
        a["dce_ok"] = bool(a["dce_val"] <= FLOOR_CE)
        a["s_name"] = d2zb_res["s_name"]
        a["s_name_ok"] = sname_ok(d2zb_res)
        a["nll_ge_bar"] = d2zb_res["bar2_nll_only"]
        a["acc"] = d2zb_res["r1"]["JULIET"]["acc"]
        if a["bar2"]:
            a["verdict"] = "clean erasure (Bar-2)"
            a["holds"] = bool(a["dce_ok"] and a["s_name_ok"])
        elif a["nll_ge_bar"] and 0.10 < a["acc"] <= 0.30:
            fb_ok = bool(d2rl_res and d2rl_res["bar2"] and d2rl_res["d_ce"]["val_all"] <= FLOOR_CE
                         and sname_ok(d2rl_res))
            a["verdict"] = ("damaged-not-erased (acc in 0.10-0.30); "
                            + ("resample-lm IS the erasure cell" if fb_ok
                               else "resample-lm also fails Bar-2"))
            a["holds"] = bool(a["dce_ok"] and a["s_name_ok"] and fb_ok)
        else:
            a["verdict"] = "fails Bar-2"
            a["holds"] = False
        p1["cells"]["a_D2_zero_both"] = a
    if d1zb_res:
        b = {}
        dj = d1zb_res["r1"]["JULIET"]["nll"] - r1_base["JULIET"]["nll"]
        b["erases"] = d1zb_res["bar2"]
        b["dce_val"] = d1zb_res["d_ce"]["val_all"]
        b["dce_fail_ok"] = bool(b["dce_val"] >= 0.30)
        b["s_name"] = d1zb_res["s_name"]
        b["s_name_fail_ok"] = bool((not math.isinf(b["s_name"])) and b["s_name"] < 5)
        b["holds"] = bool(b["erases"] and b["dce_fail_ok"] and b["s_name_fail_ok"])
        p1["cells"]["b_D1_zero_both"] = b
    p1["confirmed"] = bool(p1["cells"].get("a_D2_zero_both", {}).get("holds")
                           and p1["cells"].get("b_D1_zero_both", {}).get("holds"))

    # P2
    p2 = {}
    if d1zb_res:
        xs = [overlap_o(w) for w in P2A_CONTROLS]
        ys = [d1zb_res["d_nll"][w] for w in P2A_CONTROLS]
        rho = spearman(xs, ys)
        p2["a_collateral_vs_overlap"] = {
            "spearman": rho, "o": dict(zip(P2A_CONTROLS, xs)),
            "d_nll": dict(zip(P2A_CONTROLS, ys)),
            "pass": bool(rho >= 0.8)}
    lzw, wzw = cell_results.get("D2/zero/lm"), cell_results.get("D2/zero/wte")
    if lzw and wzw:
        jc_l = lzw["gen"]["total"]["counts"]["J_chars"] if "gen" in lzw else None
        p2["b_untied_split"] = {
            "lm_zero": {"nll": lzw["r1"]["JULIET"]["nll"], "acc": lzw["r1"]["JULIET"]["acc"],
                        "gen_J_chars": jc_l},
            "wte_zero": {"nll": wzw["r1"]["JULIET"]["nll"], "acc": wzw["r1"]["JULIET"]["acc"]},
            "write_damage": {"lm_acc_stays_high": bool(lzw["r1"]["JULIET"]["acc"] >= 0.5),
                             "J_vanishes_from_gen": bool(jc_l is not None and jc_l == 0)},
            "read_damage": {"wte_nll_gt_lm": bool(wzw["r1"]["JULIET"]["nll"] > lzw["r1"]["JULIET"]["nll"]),
                            "wte_acc_lt_lm": bool(wzw["r1"]["JULIET"]["acc"] < lzw["r1"]["JULIET"]["acc"])},
        }
        p2["b_untied_split"]["pass"] = bool(
            p2["b_untied_split"]["write_damage"]["lm_acc_stays_high"]
            and p2["b_untied_split"]["write_damage"]["J_vanishes_from_gen"]
            and p2["b_untied_split"]["read_damage"]["wte_nll_gt_lm"])
    p2c = {}
    if d2rl_res:
        p2c["resample_lm_acc"] = d2rl_res["r1"]["JULIET"]["acc"]
        p2c["resample_lm_kills_argmax_artifact"] = bool(d2rl_res["r1"]["JULIET"]["acc"] <= BAR2_ACC)
        p2c["resample_lm_bar2"] = d2rl_res["bar2"]
    sh = cell_results.get("D2/shuffle/both")
    if sh and "gen" in sh:
        gc = sh["gen"]["total"]["counts"]
        p2c["shuffle_gen_QULIET"] = gc["QULIET"]
        p2c["shuffle_gen_Q_words"] = gc["Q_words"]
        p2c["shuffle_Q_substitutions"] = bool(gc["QULIET"] >= 1 or gc["Q_words"] >
                                              gen_base["total"]["counts"]["Q_words"])
    p2["c_honest_erasure_and_shuffle"] = p2c
    pj = cell_results.get("D2/proj/both")
    if pj:
        nll = pj["r1"]["JULIET"]["nll"]
        p2["d_span_projection"] = {
            "nll": nll, "acc": pj["r1"]["JULIET"]["acc"], "bar2": pj["bar2"],
            "branch": ("erased (private component carries identity)" if pj["bar2"]
                       else "spared (shared component carries identity)" if nll < 1.0
                       else "intermediate")}

    # P3
    p3 = {"bar1_reached": bar1_reached, "at_bar1": cross}
    if bar1_reached and cross.get("r_entity") is not None:
        p3["r_entity_ok"] = bool(cross["r_entity"] < 1.3)
        p3["s_name_ok"] = bool((not math.isinf(cross["s_name"])) and cross["s_name"] < 2)
        p3["local_fluency_ok"] = bool(cross["dce_jul_windows"] is not None
                                      and cross["dce_jul_windows"] >= 0.5 * BAR1)
        p3["confirmed"] = bool(p3["r_entity_ok"] and p3["s_name_ok"] and p3["local_fluency_ok"])
    elif not bar1_reached:
        p3["confirmed"] = None
        p3["note"] = ("Bar-1 never reached within the step cap — ascent fails to "
                      "forget even partially; P3's failure mode holds a fortiori "
                      "(registered numeric tests n/a)")
    else:
        p3["confirmed"] = None
        p3["note"] = "dCE_val never exceeded 1e-4 before Bar-1 (r undefined)"
    p3["sub_logit_J_falls"] = (cross.get("d_logit_J") is not None
                               and cross["d_logit_J"] < 0)
    p3["sub_collateral_ratio_vs_surgery"] = coll_ratio
    p3["sub_collateral_ratio_ge_30"] = bool(coll_ratio is not None and coll_ratio >= 30)
    p3["surgery_cost_scaled_to_bar1"] = surg_cost_scaled
    # third-outcome check: neither instrument entity-granular
    asc_sletter_final = None
    djf = jul(traj[-1]) - base_jul
    djof = john(traj[-1]) - john(traj[0])
    if abs(djof) > 1e-9:
        asc_sletter_final = djf / djof
    p3["ascent_s_letter_final"] = asc_sletter_final
    p3["third_outcome"] = bool(
        asc_sletter_final is not None and 0.5 <= asc_sletter_final <= 2.0
        and d2zb_res is not None and 0.5 <= d2zb_res["s_letter"] <= 2.0)

    # ---------------------------------------------------------------- R0 + outputs
    r0 = sorted([{"name": w, "train_occ": design_check[w]["train_all"],
                  "nll": r1_base[w]["nll"], "acc": r1_base[w]["acc"]}
                 for w in BATTERY if w != "PROSPERO" and r1_base[w].get("n")],
                key=lambda r: r["train_occ"])

    metrics = {
        "experiment": "e023_surgical_forgetting", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED, "smoke": SMOKE,
        "design": "scratch/e023_design.md",
        "deviations": [l.strip() for l in __doc__.split("DEVIATIONS")[1].split("Run:")[0].splitlines() if l.strip()][:12],
        "gates": {"G0": G0, "G1": G1, "G2_all_cells": True, "G3": G3, "G4": G4},
        "design_crosscheck_occurrences": design_check,
        "r2_block_meta": r2_meta,
        "baseline": {"r1": r1_base, "census": r1c_base, "r2": ce_base,
                     "gen": gen_base, "r4_geometry": r4,
                     "juliet_region": [RJ_LO, RJ_HI]},
        "r0_dose_response": r0,
        "d1_shuffle_partners": partner_d1,
        "arm_ab": cell_results,
        "arm_c": {"traj": traj, "r_points": r_points, "at_bar1": cross,
                  "stop_reason": stop_reason, "steps_done": traj[-1]["step"],
                  "n_windows": len(win_starts), "gen_final": gen_asc,
                  "revive_triggers": revive,
                  "collateral_ratio_vs_surgery": coll_ratio},
        "verdicts": {"P1": p1, "P2": p2, "P3": p3,
                     "revive_fired": [k for k, v in revive.items() if v]},
        "timing": {"total_s": round(time.time() - T0, 1), "fallbacks_fired": fb},
        "config": cfg_dict(cfg),
    }

    # ---- plots
    if not SMOKE:
        keys = list(cell_results.keys())
        short = {"zero": "z", "resample": "r", "shuffle": "s", "proj": "p"}
        labels = [f"{k.split('/')[0]}-{short[k.split('/')[1]]}-{k.split('/')[2][0]}" for k in keys]

        def numlist(f):
            vals, caps = [], []
            for k in keys:
                v = f(cell_results[k])
                if v is None:
                    v = 1e-2
                if isinstance(v, float) and math.isinf(v):
                    vals.append(1e4)
                    caps.append(True)
                else:
                    vals.append(v)
                    caps.append(False)
            return vals, caps

        sn, cn = numlist(lambda r: r["s_name"])
        sl, _ = numlist(lambda r: r["s_letter"])
        sc, _ = numlist(lambda r: r["s_corpus"] if r["s_corpus"] is not None
                        else r["s_corpus_lower_bound"])
        scb = [r["s_corpus"] is None for r in cell_results.values()]
        x = range(len(keys))
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
        ax = axes[0]
        w = 0.27
        ax.bar([i - w for i in x], [max(v, 1e-2) for v in sn], w, color="crimson", label="S_name")
        ax.bar(list(x), [max(v, 1e-2) for v in sl], w, color="darkorange", label="S_letter")
        ax.bar([i + w for i in x], [max(v, 1e-2) for v in sc], w, color="seagreen", label="S_corpus")
        for i, (v, c) in enumerate(zip(sn, cn)):
            if c:
                ax.text(i, 10, ">", ha="center", fontsize=7, color="crimson")
        for i, cb in enumerate(scb):
            if cb:
                ax.text(i + w, 0.02, ">", ha="center", fontsize=7, color="seagreen")
        ax.set_yscale("log")
        ax.axhline(5, color="k", ls="--", lw=0.8, label="S_name=5 selective bar")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=60, fontsize=7)
        ax.set_title("selectivity by surgery cell")
        ax.legend(fontsize=7)
        ax = axes[1]
        dj = [cell_results[k]["r1"]["JULIET"]["nll"] - r1_base["JULIET"]["nll"] for k in keys]
        dc = [cell_results[k]["d_ce"]["val_all"] for k in keys]
        ax.bar([i - w / 2 for i in x], dj, w, color="navy", label="dNLL JULIET")
        ax.bar([i + w / 2 for i in x], dc, w, color="gray", label="dCE val-All")
        ax.axhline(BAR1, color="navy", ls=":", lw=1)
        ax.axhline(0.01, color="gray", ls=":", lw=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=60, fontsize=7)
        ax.set_title("target damage vs corpus collateral")
        ax.legend(fontsize=7)
        ax = axes[2]
        rho_x = [overlap_o(wm) for wm in P2A_CONTROLS]
        rho_y = [cell_results["D1/zero/both"]["d_nll"][wm] for wm in P2A_CONTROLS]
        ax.plot(rho_x, rho_y, "o")
        for xx, yy, wm in zip(rho_x, rho_y, P2A_CONTROLS):
            ax.annotate(wm, (xx, yy), fontsize=7)
        ax.set_xlabel("letter overlap o vs JULIET")
        ax.set_ylabel("dNLL after D1-zero-both")
        ax.set_title(f"P2a collateral vs shared rows (spearman={p2.get('a_collateral_vs_overlap', {}).get('spearman', float('nan')):.2f})")
        fig.suptitle("E023 surgery ladder")
        fig.tight_layout()
        fig.savefig(rd / "surgery_ladder.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
        doses = [jul(p) - base_jul for p in traj]
        pts = [(d, r) for d, r in zip(doses, r_points) if r is not None]
        if pts:
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, color="crimson")
        axes[0].axhline(1.3, color="crimson", ls=":", lw=1, label="revive bar 1.3")
        axes[0].axhline(2.0, color="seagreen", ls=":", lw=1, label="e003c upgrade 2.0")
        axes[0].axvline(BAR1, color="navy", ls="-.", lw=1.2, label="Bar-1 dose")
        axes[0].set_xlabel("dose = dNLL JULIET (nats/char)")
        axes[0].set_ylabel("r_entity = dNLL_JULIET / dCE_val")
        axes[0].legend(fontsize=7)
        axes[0].set_title("r_entity(dose)")
        st = [p["step"] for p in traj]
        axes[1].plot(st, [jul(p) for p in traj], "o-", ms=2.5, label="JULIET")
        axes[1].plot(st, [john(p) for p in traj], "o-", ms=2.5, label="JOHN")
        axes[1].plot(st, [pure_mean(p) for p in traj], "o-", ms=2.5, label="pure controls (mean)")
        axes[1].axhline(BAR2_NLL, color="k", ls=":", lw=0.8)
        axes[1].axhline(base_jul + BAR1, color="navy", ls="-.", lw=0.8, label="Bar-1")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("NLL/char")
        axes[1].legend(fontsize=7)
        axes[1].set_title("name battery under ascent")
        axes[2].plot(st, [cev(p) for p in traj], "o-", ms=2.5, label="val-All")
        axes[2].plot(st, [jwin(p) for p in traj], "o-", ms=2.5, label="JULIET windows")
        axes[2].plot(st, [p["ce"]["rom_windows"] for p in traj], "o-", ms=2.5, label="ROMEO windows")
        axes[2].set_xlabel("step")
        axes[2].set_ylabel("CE")
        axes[2].legend(fontsize=7)
        axes[2].set_title("corpus channels under ascent")
        axes[3].plot(st, [logitJ(p) for p in traj], "o-", ms=2.5, color="purple")
        axes[3].set_xlabel("step")
        axes[3].set_ylabel("mean logit(J) at J positions")
        axes[3].set_title("the J-row channel")
        fig.suptitle("E023 arm C: entity-granular ascent")
        fig.tight_layout()
        fig.savefig(rd / "ascent_traj.png", dpi=130)
        plt.close(fig)

    save_json(rd / "metrics.json", jsonable(metrics))
    print(f"\n{stamp()} === VERDICTS ===")
    print(f"G0 {G0['pass']}  G1 {G1}  G3 {G3['pass'] if G3 else 'n/a'}  G4 {G4['pass']}")
    print(f"P1 confirmed: {p1['confirmed']}")
    if "a_D2_zero_both" in p1["cells"]:
        a = p1["cells"]["a_D2_zero_both"]
        print(f"  (a) D2-zero-both: {a['verdict']} dCE={a['dce_val']:+.4f} "
              f"S_name={'inf' if math.isinf(a['s_name']) else format(a['s_name'], '.1f')} holds={a['holds']}")
    if "b_D1_zero_both" in p1["cells"]:
        b = p1["cells"]["b_D1_zero_both"]
        print(f"  (b) D1-zero-both: erases={b['erases']} dCE={b['dce_val']:+.4f} "
              f"S_name={b['s_name']:.2f} holds={b['holds']}")
    print(f"P2a spearman: {p2.get('a_collateral_vs_overlap', {}).get('spearman')} "
          f"(pass {p2.get('a_collateral_vs_overlap', {}).get('pass')})")
    print(f"P2b: {p2.get('b_untied_split', {}).get('write_damage')} "
          f"{p2.get('b_untied_split', {}).get('read_damage')}")
    print(f"P2c: resample-lm acc {p2c.get('resample_lm_acc')} "
          f"(kills artifact: {p2c.get('resample_lm_kills_argmax_artifact')}), "
          f"shuffle Q-subs: {p2c.get('shuffle_Q_substitutions')}")
    print(f"P2d: {p2.get('d_span_projection')}")
    print(f"P3 confirmed: {p3.get('confirmed')} (bar1_reached={bar1_reached})")
    if cross:
        print(f"  at Bar-1: r_entity={cross.get('r_entity')} S_name={cross.get('s_name')} "
              f"S_letter={cross.get('s_letter')} dCE_val={cross.get('dce_val')} "
              f"dCE_julwin={cross.get('dce_jul_windows')}")
    print(f"  collateral ratio vs surgery: {coll_ratio} (>=30: {p3.get('sub_collateral_ratio_ge_30')})")
    print(f"  revive triggers: {metrics['verdicts']['revive_fired'] or 'NONE fired'}")
    print(f"outputs: {rd}")
    print(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
