# e028 design — organ transplants across anatomies: within- vs cross-anatomy swap damage

Status: DESIGN (implementable as `lab/e028_transplant.py`). Date: 2026-09-24.
Answers T006 hypothesis **PL3** and settles registered prediction **P3**
("cross-anatomy swaps cost ≥ 2× same-anatomy swap damage"). Builds on E011b
(orthogonal-innovation rung, hook style), E014b (the second anatomy), E001.

**Question.** E014b showed two trained nets with IDENTICAL architecture
(2,739,072 params, 6L/6H/192d) and near-identical function (val 1.622 vs 1.610)
but DIFFERENT anatomies: baseline MLP-0 is a keystone (+4.08 nats) that the
renorm net does not have (+0.10); the MLP damage arrangement is INVERTED
(baseline early-heavy [4.08, 0.15, 0.27, 0.47, 0.60, 0.59] vs renorm
late-heavy [0.10, 0.08, 0.23, 0.45, 0.78, 0.70]); mid-stack roles swapped
(L2 1.06→0.48, L3 0.38→0.81). Are the organs (attention/MLP sublayers)
interchangeable across these two anatomies, or is an organ only an organ in
its own body?

**Core logic.** Take a trained host net, replace one sublayer's parameters
with a donor's, measure Δ val CE. Normalize by the SAME organ's zero-ablation
damage in the SAME host (from E001/E014b, verified): R = ΔCE_transplant /
ΔCE_ablate. R indexes whether a foreign organ is better than no organ
(R < 1, compatibility), equivalent to no organ (R ≈ 1, inert), or worse than
no organ (R > 1, active interference — cf. E011b: same-norm random writes hurt
MORE than zeroing everywhere). On the baseline host, E011b already measured
the same-norm-random rung (R_rand: attn [1.52, 1.39, 1.68, 2.06, 1.97, 2.94],
mlp [1.09, 5.47, 2.72, 1.80, 1.49, 1.69]) — the transplant R lands on an
existing three-rung ladder: zero < foreign-trained < random.

---

## 1. TRANSPLANT UNITS (state_dict key surgery; shapes identical by construction)

| unit | keys moved (at site i) | # tensors |
|---|---|---|
| (a) MLP organ | `h.{i}.mlp.0.weight/.bias`, `h.{i}.mlp.2.weight/.bias` | 4 |
| (b) attention organ | `h.{i}.attn.c_attn.weight`, `h.{i}.attn.c_proj.weight` | 2 (no biases) |
| (c) layer-pair (block organ) | (a) + (b) at the same i; host LNs kept | 6 |
| (d) whole-prefix 0..k | (a)+(b)+block LNs for all layers ≤ k; `wte/wpe/ln_f/lm_head` ALWAYS host-native; k ∈ {1, 3}; k = 5 excluded (trivial = the donor net) | 11·(k+1) |

**LN decision (primary):** `ln1/ln2` affines stay with the HOST. Reasons:
(i) `lesion()` semantics — zero-ablation (our normalization denominator)
removes only the sublayer write, so transplant must also move only the
sublayer to keep R well-defined; (ii) LN affine = the host's interface
calibration, host tissue by definition. Secondary variant (O2, optional):
co-transplant the organ's input LNs (`ln1` for attn, `ln2` for mlp) at the
two L0 sites only — if O2 repairs a large fraction of cross damage, the
incompatibility lives in the INTERFACE (calibration), not the write content.
All swaps are one-directional grafts (donor unchanged; we copy tensors, never
exchange) — reciprocal double-transplants confound two failures and are out.

**Sites: L0, L2, L3, L5 — and why exactly those four:**
- **L0 — the divergence epicenter.** Baseline block-0 sees stream 0.67 and
  MLP-0 bootstraps the 8.4× norm jump (+4.08 keystone); renorm host pins the
  block-0 input to 5.6 and that job is deleted (+0.10); attn-L0 writes differ
  2.9× (2.7 vs 7.8). Highest absolute damage anywhere → best SNR, and the
  sharpest theory-driven cell: baseline organs were calibrated for a 0.67
  stream, renorm organs for a 5.6 one.
- **L2 AND L3 — the crossed roles.** The two anatomies SWAPPED mid-stack
  (attention damage L2 1.06→0.48, L3 0.38→0.81). Same-stage-different-address
  in both directions; each site informs the other. If PL3 is about stage
  compatibility (not index compatibility), L2/L3 cross grafts should be the
  mid-stack failure case in BOTH directions.
- **L5 — the calibrator + the inverted MLP arrangement.** Renorm MLP-5 is
  critical (0.78) where baseline MLP-5 is moderate (0.59); attention-L5 is
  nearly dead in BOTH anatomies (0.03). A near-dead site is the purest
  interference probe: ablation there costs ~nothing, so ANY transplant damage
  at attn-L5 is active harm from the foreign write, not lost function.
- **L1/L4 excluded (budget):** L1 has tiny ablation damage (mlp 0.15) — poor
  normalization denominator; L4 is the shoulder of L5. They are the first
  fill-ins if the stop-rule (§5) leaves slack.

## 2. ARMS + CONTROLS

Nets available: **B** = `runs/checkpoints/e001.pt` (baseline anatomy, seed 42,
val 1.6224); **R** = `runs/checkpoints/e014b.pt` (renorm anatomy, seed 42 —
same init as B, differs only by the training-time renorm hooks, val 1.6100).
IMPORTANT: R's state_dict was saved WITHOUT hooks; every R evaluation must
reinstall `register_renorm(model, c=5.6)` first (eval without hooks is
off-manifold: +3.56 nats, E014b control). B is always evaluated bare.

**Single-seed limitation (stated plainly):** we have exactly ONE net per
anatomy. A "within-anatomy swap" between two blocks of the same net at
different sites is NOT a control (it changes the site, not the donor); a
self-transplant is exactly zero by construction (that is its job — wiring
sanity, not anatomy). The only true within-anatomy control needs a second
seed of the same regime. Therefore:

| arm | host ← donor | purpose | evals |
|---|---|---|---|
| **C0 self-transplant** | B←B, R←R (organ's own params through the full surgery path) | wiring sanity: ΔCE < 1e-4, and bitwise-identical state_dict check; validates the surgery helper end-to-end. Sites: attn-L0, mlp-L0, each host | 4 |
| **C1 within-anatomy (the P3 denominator)** | B ← B43 (fresh baseline, NEW seed 43) | same regime, same anatomy family, different seed — the honest "same-anatomy" damage. Trained in-run: `train_model(steps=4000, lr=1e-3, bs=64, max_seconds=252, ckpt resumable)`, val gate ≤ 1.7224 (else flag; still usable — damage deltas, not absolute loss, are the readout) | 8 (4 sites × 2 kinds) |
| **X cross-anatomy, B host** | B ← R | the PL3 test, direction 1 | 8 |
| **X cross-anatomy, R host** | R ← B | direction 2 (hooks active) | 8 |
| **C2 lottery (random-tissue)** | R ← fresh-init organ (seed 99, std-0.02 init) | is a TRAINED-foreign organ better than a random one? Sites: attn/mlp × L0/L5 on the R host (the B host already has E011b's norm-matched random rung for free) | 4 |
| **(c) layer-pair** | cross only, L0 and L5, both directions | does moving attn+mlp together cost more than either alone (coordination)? | 4 |
| **(d) whole-prefix** | cross only, k ∈ {1, 3}, both directions | anatomy as a coordinated whole | 4 |

Seed-43 rationale: B and R share init seed 42 and differ only by the renorm
regime, so X is attributable to REGIME, not init. B43 must therefore be a
different seed (43) — it varies BOTH init and stochastic training order,
which is the correct within-anatomy variability estimate (it includes
everything "same anatomy" can mean). Renorm-anatomy within-control (a second
renorm seed) is NOT affordable inside 15 min; registered as the e028.1
replication (add seed-43 renorm arm, +4.2 min, same script flag) — until
then, P3 is evaluated on the B host only, and R-host results are directional.

## 3. EVAL PROTOCOL

- **30 fixed batches, paired:** `estimate_loss(model, corpus, "val",
  n_batches=30)` — deterministic, generator seeded from `corpus.seed`, so
  EVERY arm sees the SAME 30 batches; batch-sampling noise cancels in every
  difference. Per-batch losses stored for bootstrap (2000 resamples) on every
  headline cell (E011c noise discipline); P3 thresholds must hold with the
  ratio CI excluding 1 — see §4.
- **Hooks:** R-host arms run inside `register_renorm(model, c=5.6)` /
  remove (copy the 8-line helper from `lab/e014b_stream_renorm.py` — do not
  import the experiment module). A renorm-host liveness assert: one probe
  forward must show block-input norms = 5.6 ± 1e-3 before the first eval.
- **Base CEs recomputed in-run:** B (expect 1.6224), R with hooks (expect
  1.6100), B43 (record). Ablation references REUSED from `runs/e001/
  metrics.json` (attn [2.401, 1.737, 1.063, 0.379, 0.193, 0.034], mlp
  [4.079, 0.150, 0.265, 0.467, 0.599, 0.592]) and `runs/e014b/metrics.json`
  (attn [2.799, 1.693, 0.484, 0.812, 0.207, 0.031], mlp [0.097, 0.083, 0.228,
  0.451, 0.781, 0.704]) — both were measured on this same deterministic
  30-batch protocol; spot-recompute 2 of them as a cross-check.
- **Damage normalization:** for every cell report ΔCE and
  **R = ΔCE_transplant / ΔCE_ablate(host, site, kind)**, with the bands:
  R < 0.5 strong compatibility (foreign organ does usable local work);
  0.5–0.9 partial; 0.9–1.1 inert (organ output is noise to the host);
  R > 1.1 active interference. On the B host also report
  R/R_rand (position on the E011b zero–foreign–random ladder). Negative ΔCE
  (graft IMPROVES the host, plausible only at near-dead sites like attn-L5)
  is a valid observation — log it as "graft takes".

## 4. PREDICTIONS OPERATIONALIZED (frozen before running)

**P3 (T006 verbatim: "cross-anatomy swaps cost ≥ 2× same-anatomy swap
damage").** On the B host, per cell (4 sites × 2 kinds):
ρ = ΔCE_cross(B←R) / ΔCE_within(B←B43).
- **CONFIRMED** iff median ρ over the 8 cells ≥ 2 AND ρ ≥ 2 in ≥ 6/8 cells
  AND in ≥ 6/8 cells the bootstrap 95% CI of ΔCE_cross − 2·ΔCE_within
  excludes 0.
- **REFUTED** iff median ρ ≤ 1.2 AND ΔCE_cross ≤ ΔCE_within in ≥ 4/8 cells.
- Partial otherwise: report the per-site ρ table (the SITE STRUCTURE is the
  finding — see S1).
- Denominator guard: cells with ΔCE_within < 0.05 (noise floor) are excluded
  from the median and listed separately (their honest statement is the
  additive one: ΔCE_cross ≥ ΔCE_within + 0.30).

**Compatibility vs interference (registered meaning of R):**
- **Foreign beats no organ (R < 1):** the sublayer's local computation is
  anatomy-portable — organs implement generic local transforms; degeneracy
  holds ACROSS anatomies at the sublayer level (weakens PL3, supports
  stage-level PL2).
- **Foreign ≈ no organ (R ≈ 1):** the host treats the foreign write as
  unreadable noise — incompatibility without interference. The organ's value
  is body-specific.
- **Foreign worse than ablation (R > 1):** active interference — the foreign
  organ writes structured, confident, WRONG-for-this-body content that
  downstream computation trusts and is misled by (the transplant analogue of
  E011b's random-write excess; distinguish from it via C2: if R_cross ≈
  R_lottery, the excess is generic off-manifold energy; if R_cross < R_lottery,
  trained content is partially legible even when harmful).

**Secondary registrations:**
- **S1 — keystone asymmetry (the sharpest single cell):** B's MLP-0 → R host:
  R ≥ 1 (its norm-bootstrapping function is deleted by the pin; its write is
  baseline-calibrated → predicted interference on top of a lost keystone).
  R's MLP-0 → B host: R ≈ 1 with ΔCE ≈ B's own MLP-0 ablation (4.08) — a
  near-inert organ (home damage +0.10) replacing a keystone should behave
  like removing the keystone, i.e. quietness transfers even when function
  does not. If BOTH directions show R ≈ 1 that is already informative:
  transplants fail by silence, not violence.
- **S2 — prefix superadditivity:** ΔCE(prefix-0..3 cross) > Σ of its four
  single-block cross damages → cross-anatomy incompatibility compounds;
  anatomy is a coordinated whole, not a bag of organs. (Directional; no gate.)
- **S3 — lottery placement:** on the R host, R_lottery ≥ R_cross at ≥ 3/4
  sites (a trained-foreign organ beats random tissue). Failure (lottery ≈
  cross) would mean trained structure contributes nothing beyond generic
  off-manifold energy.
- **Caveat registered with P3:** index-matched swaps assume stage addresses
  align across anatomies (PL2). If stages MOVED (e012b census will tell),
  index-matched cross grafts mispair function and inflate ρ — P3 conflates
  "different anatomy" with "different addresses". The B←B43 within arm has
  the same index-matching convention, so the CONTRAST stays clean, but the
  interpretation of any single cross cell must wait for the stage map.

## 5. TIME BUDGET (hard cap 15 min; single steps ≤ 30 min; all resumable)

Timing base (e014b realized): one 30-batch deterministic eval ≈ 13–15 s;
R-host evals +~5% (hook overhead).

| step | wall clock | evals |
|---|---|---|
| load B, R, B43-if-exists; base CEs; liveness assert; spot-check 2 ablations | ~1.2 min | 4 |
| train B43 (fresh baseline seed 43, `max_seconds=252`, ckpt `runs/checkpoints/e028_b43.train.pt` → save `e028_b43.pt`) | 4.2 min | — |
| C0 self-transplant sanity (4) | ~1.0 min | 4 |
| X cross B host (8) + C1 within (8) — the P3 core | ~3.8 min | 16 |
| X cross R host (8, hooks) | ~2.0 min | 8 |
| C2 lottery on R host (4) | ~1.0 min | 4 |
| layer-pair (4), then prefix k=1,3 (4) | ~2.0 min | 8 |
| bootstrap CIs, metrics.json, 2 figures | ~0.5 min | — |
| **total** | **≈ 15.5 min worst case → stop-rule below** | 44 |

**Priority + stop-rule (pre-registered):** run in the table's order. Before
each optional block, check elapsed: if > 12 min, drop PREFIX first, then
layer-pair, then C2 (E011b's random rung partially substitutes on the B
host). NEVER drop: C0, X both hosts, C1. If the wall clock exceeds 13 min at
the R-host block, finish the L0 + L5 cells and drop L2/L3 there (P3 itself
does not need R-host cells). If B43 training is interrupted, resume from its
ckpt (a partial-budget B43 with val ≤ 1.72 is still a valid donor — record
its val and steps).

## 6. FAILURE MODES → interpretation (pre-registered)

| outcome | meaning for PL3 |
|---|---|
| P3 confirmed (median ρ ≥ 2) | Anatomy-level degeneracy is REAL: organs interchangeable within, not across. T006's PL3 advances; follow-up = which interface fails (O2 LN co-transplant; e014c-style write rescaling) |
| P3 refuted (cross ≈ within) | Sublayer organs are anatomy-portable; the E014b "different anatomies" differ mainly in scheduling/addresses, not organ mechanics. Supports stage-level PL2 — "which stage does X, and where did it land" |
| R > 1 widely (interference) | Foreign trained organs actively mislead — transplant failure is violent, not silent; connects to E011b's random-write excess; check S3 to separate content from energy |
| R ≈ 1 everywhere (inert) | Transplants fail by silence: hosts ignore foreign writes. Damage is dominated by the LOST native organ, i.e. transplant ≈ ablation everywhere |
| R < 1 at some sites | Partial graft-take: those sublayers implement portable local transforms — the first direct evidence of cross-anatomy organ compatibility (headline-worthy either way) |
| B43 fails val gate (> 1.72) | Within-denominator still used (deltas, not levels, matter) but flagged; if B43 diverges, retrain once at lr 6e-4 (fallback ladder, e014b §2 style) before dropping C1 |
| self-transplant ΔCE > 1e-4 | Do not interpret ANYTHING; the surgery helper is broken (fix first) |
| renorm liveness assert fires | R-host numbers are off-manifold; stop and fix hooks before reading direction-2 cells |
| R-host L2/L3 dropped by stop-rule | P3 unaffected (B host only); report direction-2 as L0/L5-only |

## 7. PSEUDOCODE — parameter-swap surgery + eval

```python
# ---- key sets (verified against both ckpts: 65 keys, identical shapes) ----
def mlp_keys(i):   return [f"h.{i}.mlp.0.weight", f"h.{i}.mlp.0.bias",
                           f"h.{i}.mlp.2.weight", f"h.{i}.mlp.2.bias"]
def attn_keys(i):  return [f"h.{i}.attn.c_attn.weight", f"h.{i}.attn.c_proj.weight"]
def block_keys(i): return [f"h.{i}.ln1.weight", f"h.{i}.ln1.bias"] + attn_keys(i) \
                      + [f"h.{i}.ln2.weight", f"h.{i}.ln2.bias"] + mlp_keys(i)
def organ_keys(site, kind):  # the SURGERY unit (LNs stay with host)
    return mlp_keys(site) if kind == "mlp" else attn_keys(site)

def transplant(host: TinyGPT, donor_sd: dict, site: int, kind: str,
               keys=None) -> None:
    """Graft donor organ into host IN PLACE (host's other params untouched).
    One-way copy; donor_sd never modified."""
    sd = host.state_dict()                       # OrderedDict, on DEVICE
    for k in (keys or organ_keys(site, kind)):
        assert sd[k].shape == donor_sd[k].shape, k
        sd[k] = donor_sd[k].clone().to(sd[k].dtype)
    host.load_state_dict(sd, strict=True)        # strict: key/shape sanity
    # prefix variant: keys = [k for j in range(k_max+1) for k in block_keys(j)]

def snapshot(host: TinyGPT) -> dict:             # undo = load snapshot
    return {k: v.detach().clone() for k, v in host.state_dict().items()}

def restore(host, snap): host.load_state_dict(snap, strict=True)

# ---- lottery organ (random tissue, seed 99) ------------------------------
set_seed(99)
lottery_net = TinyGPT(cfg).to(DEVICE)            # fresh init only, never trained
lottery_sd = lottery_net.state_dict()

# ---- renorm hooks for R-host evals (copy from lab/e014b_stream_renorm.py) -
def register_renorm(model, c=5.6):               # 6 block-input pre-hooks
    hooks = []
    for block in model.h:
        def pre(m, args, _c=c):
            x = args[0]
            n = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return (x * (_c / n),)
        hooks.append(block.register_forward_pre_hook(pre))
    return hooks

# ---- one cell -------------------------------------------------------------
def eval_cell(host, corpus, donor_sd, site, kind, base_ce, renorm=False):
    snap = snapshot(host)
    if donor_sd == "self":
        pass                                     # C0: surgery on own params
    else:
        transplant(host, donor_sd, site, kind)
    hs = register_renorm(host) if renorm else []
    ce = estimate_loss(host, corpus, "val", n_batches=30)   # same 30 batches
    for h in hs: h.remove()
    restore(host, snap)
    return ce - base_ce                          # ΔCE; R computed vs ablation ref

# ---- C0 bitwise check (before any other cell) ----------------------------
sd0 = snapshot(B); transplant(B, {k: v for k, v in sd0.items()}, 0, "mlp")
assert all(torch.equal(a, b) for a, b in zip(B.state_dict().values(),
                                             sd0.values()))

# ---- main grid -------------------------------------------------------------
for host, donor, label, renorm in [
        (B, R_sd,  "cross_B",  False), (B, B43_sd, "within", False),
        (R, B_sd,  "cross_R",  True),  (R, lottery_sd, "lottery", True)]:
    for site in (0, 2, 3, 5):
        for kind in ("attn", "mlp"):
            d = eval_cell(host, corpus, donor, site, kind, base[host], renorm)
            record(label, site, kind, d)         # + bootstrap over 30 batches
# layer-pair: keys=attn_keys(s)+mlp_keys(s) at s in (0,5), cross both hosts
# prefix:     keys=block_keys(0..k)      for k in (1,3), cross both hosts
```

Notes: (1) `state_dict()` returns REFERENCES — always `snapshot()` before and
`restore()` after; every cell leaves the host bit-identical (assert once).
(2) The transplant composes with nothing else — no lesion hooks in e028; the
ablation references come from prior runs. (3) `estimate_loss` flips
`model.train()` at exit — irrelevant here (pure eval, no hooks that tag by
mode besides renorm, which must fire in BOTH modes). (4) Optional
instrumentation (cheap, 8 probe forwards): log the grafted organ's in-situ
write norm vs its at-home write norm at L0/L5 — the geometric half of the
compatibility story.

## 8. RUN ORDER + BOOKKEEPING

1. loads + C0 (gate: ΔCE < 1e-4) → 2. train B43 → 3. P3 core (X_B + within)
→ 4. X_R → 5. lottery → 6. layer-pair → 7. prefix (stop-rule §5) →
8. bootstrap CIs, `runs/e028/metrics.json` (schema: per arm/host — base CE,
per {site, kind} ΔCE with CI, R and R/R_rand, ρ table with P3 verdict
booleans, S1–S3 booleans, B43 val + steps), 2 figures (R heatmap over the
site×kind grid per host with the R=1 contour; ρ per site with the 2× line).
9. NOTES.md entry + THINKING T006 PL3 resolution BEFORE anything builds on
it. ckpts: `runs/checkpoints/e028_b43.pt` (+ `.train.pt` resumable).
Replication debt: e028.1 = renorm seed-43 arm (within-control for the R host,
+4.2 min) — same script, `--replicate` flag.
