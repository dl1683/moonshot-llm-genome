# e040 design — graft-evolution lineage: is init-anchoring EVOLVABLE?

Status: DESIGN (implementable as `lab/e040_graft_evolution.py`). Date: 2026-09-25.
The lab's first EVOLUTION-thread experiment. Answers the question Review 8 scoped
("the steps lesson rewrites e040's protocol — step-matched lineages, designed with
the user") and stress-tests card v3 **C3** (partial anchoring: same-init dW
alignment 0.15 vs diff-init 0.00, ceiling 0.53) with the only instrument C3 has
never faced: **selection**. T003/T006's anchoring ladder and T006-P3's mechanism
(dW orthogonality across inits) are static descriptions of one training draw
each. e040 asks whether the property they describe — cross-seed MLP graft
violence — has **heritable variation a lineage can respond on**, while holding
function at parity. Builds on e028/e029 (organ surgery protocol, verbatim),
e041 (alignment ladder), e005s (the 0.84M config + its val 1.5581 anchor),
e023_design (the measured-probe method: design numbers are MEASURED, marked MP).

**Compute envelope (PERMANENT, STATE.json 2026-09-25):** every lineage member
≤1M params (we use 840,704); strictly serial GPU (no concurrent jobs); batch 32;
≤180s training caps; `gpu_ok()` guard before every launch; `cooldown(60–90s)`
between any two GPU phases; **all assays (grafts, ablations, ΔW) run CPU-only**
(`CUDA_VISIBLE_DEVICES=""` — validated by the design probes below), so thermal
budget is spent on training only.

---

## 1. MEASURED PROBES (2026-09-25, eval-only, CPU, no files written)

Instrument = e028/e029 verbatim: surgery units `mlp_keys(i)` (MLP fc weights+biases;
**LNs stay with the host**); 30 fixed deterministic val batches (generator seeded
from corpus.seed 1337, bs 16×256); host snapshotted and restored bitwise after
every cell (C0 self-transplant gate: dCE exactly 0.0, passed); paired bootstrap
CI (n=2000). Hosts/donors are the existing 6L/6H/192 nets (2.7M, under the 5M
ceiling for eval; zero GPU used).

### 1a. Cross-seed graft damage at the L2/L3 MLP sites (MP)

| host ← donor | mlp-L2 dCE (CI95) | mlp-L3 dCE (CI95) | lineage tag |
|---|---|---|---|
| B ← B43 | **+0.820** (+0.801, +0.839) | **+1.990** (+1.960, +2.016) | cross-seed |
| B43 ← B | **+0.779** (+0.760, +0.802) | **+1.912** (+1.887, +1.939) | cross-seed (reverse) |
| B ← R | +0.425 (+0.411, +0.437) | +0.647 (+0.629, +0.662) | same-init diff-regime |
| B ← BDO | +0.106 (+0.099, +0.113) | +0.367 (+0.356, +0.378) | same-init diff-order |
| ablate ref (zero own organ, host B) | +0.265 | +0.467 | — |

Readings that fix the design:
- **Trait magnitude:** a foreign trained MLP costs +0.8…+2.0 nats — 3.1× (L2) to
  4.3× (L3) the cost of **deleting** the host's own organ. Grafting is not
  mildly-off-manifold; it is worse than absence (trained-foreign tissue
  misleads — e028's law, now quantified at the exact e040 sites).
- **Measurement noise is negligible:** CI half-widths ±0.02–0.03 nats ≈ 2–3% of
  trait. Differences ≥0.06 nats between members are resolvable → selection has
  an instrument, not a coin flip.
- **Trait variance across the existing axes:** host-direction asymmetry ~4–5%
  (B←B43 vs B43←B), site spread L3/L2 ≈ 2.4×, and at L2 the lineage rungs run
  0.106 (same-init diff-order) → 0.425 (diff-regime) → 0.820 (diff-seed). The
  selection trait varies over an ~8× range across known init/regime axes — the
  question is whether it varies over *selectable, heritable* init perturbations.

### 1b. ΔW-alignment at the same sites (MP)

cos(dW_a, dW_b), dW = W_trained − W_init(own seed), flattened per MLP organ:

| pair | L2 | L3 | rung |
|---|---|---|---|
| B ↔ BDO | **+0.422** | **+0.356** | same-init diff-ORDER (ceiling) |
| B ↔ R | +0.087 | +0.090 | same-init diff-regime |
| B43 ↔ R43 | +0.108 | +0.111 | same-init diff-regime |
| B ↔ B43, B ↔ R43, R ↔ B43, BDO ↔ B43 | −0.013 … +0.005 | ≈ **0.000** | diff-init (floor) |

Consistent with e029 (pooled same-init +0.152 / diff-init ≈ 0) and e041's
~0.53 pooled ceiling; new fact: **at the L2/L3 MLP graft sites the diff-regime
rung is only ~0.09–0.11** — the anchoring ladder is site-specific, and the
floor-to-ceiling room at the violent sites is the full 0→0.4 range. Nothing in
the ladder says the trait can MOVE under selection — that is e040's question.

---

## 2. EVOLUTION PROTOCOL (under the envelope)

**Genotype = the init; phenotype = the trained net; selection = graft
compatibility.** Every member trains FROM SCRATCH, step-matched; heredity flows
through inherited init + fresh mutation. No member is ever fine-tuned (that
would re-import the steps confound Review 8 flagged in T020: 4000/2226/1086
steps anti-correlated with scale — here, unequal steps would masquerade as
evolution).

### 2.1 Architecture and training (frozen for every member, every generation)

- Config = e005s SMALL verbatim: 4L / 4H / 128d, block 256, vocab 65 →
  **840,704 params ≤ 1M** ✓. Corpus `data/input.txt`, CharCorpus(seed=1337) —
  identical batch order for all members (init is the ONLY axis that varies).
- `train_model(steps=4000, lr=1e-3, batch_size=32, max_seconds=180, ckpt=…)` —
  **step-matched at exactly 4000 steps**; cosine schedule completes for everyone.
  (e005s measured 4000 steps ≈ 63s at batch 64; expect ~60–110s at batch 32.)
- Step-completion rule (G5): a member that hits the 180s cap before step 4000 is
  **excluded and reported**, never extended. If >1 member per cohort is
  excluded → abort the experiment (envelope violated; retry when cooler).

### 2.2 The reference foreign body (fixed, never selected)

- **REF**: fresh init from `set_seed(4304)` (a seed sharing no family relation),
  trained under §2.1. Its MLP-L2 and MLP-L3 organs are THE standard graft for
  every assay in every generation. REF is measured once for parity (G3) and then
  frozen. Scope limit registered: compatibility is donor-specific (e029's
  organ-type specificity); all claims are w.r.t. this one reference.

### 2.3 Cohorts (11 nets total, all 0.84M)

- **gen-0 (founders, 4 members + REF):**
  - `W` — wildtype, exact `set_seed(42)` init (the family anchor). NOTE:
    e005s_small (0.84M, seed 42) already exists but trained at batch 64 — per
    the step/batch-match lesson it is used ONLY as an external val anchor
    (G0), not as a population member; W is retrained at batch 32.
  - `M1..M3` — mutants: `init = init42 + ε`, ε ~ N(0, σ_mut²) elementwise on
    ALL weights, σ_mut = 0.005 (25% of the 0.02 init std), draw seeds
    4021/4022/4023. Same seed family, perturbed.
- **gen-1 (3 children):** parents = two best eligible gen-0 members (§2.4).
  Children = parent init + fresh ε (σ_mut, seeds 4031/4032/4033): two from the
  top parent, one from the second. If <2 eligible members: all children from
  the single best parent (bottleneck logged).
- **gen-2 (3 children):** same rule from gen-1's best two (seeds 4041/4042/4043).

Two selection events (gen-0→1, gen-1→2); P1's bar is at gen-2.

### 2.4 Selection criterion (frozen)

Per member i, on the 30 fixed val batches (CPU):

- **D_i** = mean over sites {L2, L3} of dCE(host_i ← REF mlp-Lsite) — the raw
  trait (headline).
- **A_is** = ablate ref: dCE(host_i, zero own mlp-Lsite) — the organ's
  load-bearing weight, recomputed per member per generation.
- **Selection index R_i** = mean over sites of D_is / A_is — foreign-organ
  damage **relative to deleting your own organ**. This is the selected scalar.
  R is the honest trait because D can be gamed two ways (see degenerate routes
  §4): by de-valuing the organ (A shrinks) or by damaging the host.
- **Eligibility gates** (a member ranks only if both hold):
  (a) parity: val CE_i ≤ val(W) + 0.05;
  (b) organ-band: A_is ∈ [0.5×, 2.0×] A_Ws for both sites (organ still
  load-bearing).
- Rank by R_i ascending; top two breed.

**Variance gate (G7):** if the gen-0 mutant spread (max−min of D over M1..M3)
< 3× the bootstrap CI half-width (~0.06 nats), σ_mut was too small to see:
gen-1 escalates the pre-registered ladder σ_mut ×2 (0.010), logged as
`sigma_escalation`, and P1/P2 verdicts carry that caveat.

---

## 3. ASSAY PROTOCOL (every cohort; identical to the §1 probes)

1. Base CE per member (30 fixed batches, CPU) + G2 determinism double-run.
2. Graft cells: member ← REF organs at mlp-L2, mlp-L3 (8 cells/cohort for
   gen-0 incl. W; 6 for gen-1/2). Paired per-batch ΔCE + bootstrap CI;
   host restored bitwise + asserted after every cell.
3. Ablation refs A_is per member (lesion hook, same 30 batches).
4. ΔW-alignment per member: cos(dW_member, dW_REF) at both graft organs
   (dW from the member's OWN init; raw cos(W_member, W_REF) reported as
   secondary, immune to init bookkeeping).
5. Reverse-graft control on the gen-2 winner only: REF ← winner organs
   (host-direction asymmetry check, MP says expect ~5%).
6. Outputs → `runs/e040/metrics.json`, `lineage_trait.png` (D and R by
   generation, CI bars, parity panel), `dw_alignment.png` (cos vs REF by
   generation).

---

## 4. REGISTERED PREDICTIONS (frozen before any training)

**P1 — EVOLVABLE.** By gen-2: mean D_gen2 ≤ 0.75 × mean D_gen0 (**−25%**) AND
mean R_gen2 ≤ 0.75 × mean R_gen0, with the paired-bootstrap CI of the gen-2 vs
gen-0 difference excluding 0, AND every contributing member inside both
eligibility gates. → cross-seed MLP graft compatibility has heritable
variation responsive to selection at 0.84M: **init-anchoring is evolvable**.
(C3 gains its first dynamic stamp; the ladder's rungs are movable by selection.)

**P2 — FROZEN.** No response: |mean D_gen2 − mean D_gen0| < 10% or CI includes
0, AND alignment shift < +0.05. → at σ_mut = 25% of init std the anchoring
trait has no selectable heritable variation — the ladder's rungs are fixed at
this scale/mutation size; evolution would need larger moves (new-seed-scale
mutation) or cannot move it at all (anchoring is a crystallized property of
early optimization).

**P3 — DEGENERATION-ROUTE.** Mean cos(dW_member, dW_REF) at the graft organs
RISES by ≥ +0.05 from gen-0 to gen-2 **while** damage does NOT fall (P1 fails).
→ the lineage drifts toward the donor's parameter-space motion without the
graft interface improving: alignment is the observable selection can pump but
is NOT sufficient for compatibility — the mechanism story (C3) would need the
interface (LN statistics / stream geometry, e031's alternative) upgraded from
"unexcluded" to "necessary".

**Gate-failure verdicts (distinct from P1–P3, reported as-is, no rounding):**
- *Parity-broken route*: D falls ≥25% but mean val CE of selected members
  exceeds parity — damage fell by degrading the host.
- *Organ-devaluing route*: D falls while A refs exit the band — the organ
  stopped mattering; R did not fall.
- *Instrument-failure*: G1/G2 bitwise or determinism gates fail → nothing is
  interpreted (e046 smoke-flag lesson).

**Interpretation discipline:** with n=1 donor, n=3 mutants, n=2 generations,
P1's claim is scoped to "this reference, this σ, this scale" — the registered
follow-up on a P1 fire is a second REF donor (e040b), never a re-rolled e040.

---

## 5. BUDGET (thermal blocks; expected total ≈ 23.8 min ≤ 25 min cap)

All GPU work strictly serial; `cooldown(60s)` (90s if `gpu_status().temp > 70C`)
after every training run; CPU assays run INSIDE the cooldown windows (zero
added wall time). 180s is a CAP, not a target — expected ~60–110s per run.

| # | phase | device | est | thermal block after |
|---|---|---|---|---|
| 1 | REF train (seed 4304) | GPU | ~75s | cooldown 60s |
| 2 | W train (seed 42, b32) | GPU | ~75s | cooldown 60s |
| 3–5 | M1–M3 train | GPU | 3×~75s | 60s after each |
| — | gen-0 assay: 8 grafts + 8 ablates + bases + ΔW | CPU (in cooldowns) | ~90s | none |
| 6–8 | gen-1 children train (post-selection) | GPU | 3×~75s | 60s after each |
| — | gen-1 assay | CPU | ~70s | none |
| 9–11 | gen-2 children train | GPU | 3×~75s | 60s after each |
| — | gen-2 assay + reverse-graft + figures | CPU | ~70s | none |
| | **totals** | | **11×75s train + 10×60s cooldown ≈ 1425s** | |

**Stop-rules (pre-registered):** before launching child k, if elapsed > 1140s
(19 min) skip it — minimum viable lineage = REF + gen-0 + ≥2 gen-1 children
(still adjudicates P2's direction with a caveat). If any single training run
exceeds 150s wall, treat the envelope as violated: finish the current cohort's
assay and stop (report partial). `gpu_ok()` returns False at any launch point →
wait for GPU_IDLE_TEMP_TARGET (65C) then retry once; a second failure parks the
run for the heartbeat to resume (ckpts make everything resumable).

---

## 6. VERIFICATION GATES

- **G0:** val(W_b32) within 0.03 of e005s_small's 1.5581 (protocol sanity;
  batch-size shift tolerated).
- **G1:** C0 self-transplant bitwise-identical, dCE exactly 0.0, once per cohort.
- **G2:** two consecutive base-CE evals bit-identical (determinism).
- **G3:** REF val CE ∈ [1.50, 1.66] (a competent foreign body, not a broken one).
- **G4:** per-member ablate refs inside [0.5×, 2.0×] W's (eligibility band).
- **G5:** all members reach exactly step 4000 (else exclusion/abort, §2.1).
- **G6:** `gpu_ok()` before every GPU launch; serial-only; cooldowns logged with
  temp into metrics (`thermal_log`).
- **G7:** gen-0 variance gate (§2.4) — σ-escalation decision point.

## 7. CHECKPOINT / OUTPUT DISCIPLINE

Every member: resumable train ckpt `runs/checkpoints/e040_<name>.train.pt`
(written at every eval point by `train_model`; map_location="cpu" resume bug
already fixed in common.py) + final `runs/checkpoints/e040_<name>.pt`
(names: e040_ref, e040_w, e040_m1..3, e040_g1a..c, e040_g2a..c). Nothing in
`runs/` or `data/` is ever deleted. Outputs: `runs/e040/metrics.json`
(full cell table, per-generation trait summaries, thermal log, verdicts),
`lineage_trait.png`, `dw_alignment.png`. NOTES.md entry + THINKING.md
interpretation entry follow the run (Rule 0). **Pre-run gate:** register P1–P3
as a THINKING.md entry citing this memo before `lab/e040_graft_evolution.py`
trains anything.

## 8. HONESTY CAVEATS (pre-registered)

- Selection sees ONE donor's TWO organs — "evolvable compatibility" claims are
  donor-scoped (n=1 REF; e029 showed organ-type specificity).
- Trait readout is on val text with a fixed 30-batch panel: deltas are exact,
  but the panel is one draw of positions; CI reflects batch resampling only.
- σ_mut = 25% of init std is a guess calibrated to sit between the order-rung
  and the seed-rung of the MP ladder; G7's escalation is the pre-registered
  correction, not a retrofit.
- Parity gate (+0.05) is tighter than e029's (+0.10) on purpose: selection
  pressure toward low graft damage is exactly the pressure that rewards damaged
  nets; the tight gate plus the R-normalization plus the ablate-band are three
  independent guards against the same degenerate solution.
- Founder effects: 2 parents × 2 generations from a 4-member founder cohort is
  a demonstration of response-to-selection, not a population-genetics estimate;
  no claim about selection strength or convergence is made or tested.

## 9. PSEUDOCODE

```python
# init42 = snapshot(TinyGPT(cfg)); REF: set_seed(4304) -> train (G3) -> freeze.
# for cohort, members in [(g0, [W, M1..3]), (g1, children(g0)), (g2, children(g1))]:
#   for m in members: gpu_ok(); train(m, steps=4000, bs=32, cap=180, ckpt);
#                     cooldown(60); assert step==4000 (G5)
#   CPU assay (CUDA_VISIBLE_DEVICES=""): base x2 (G2), C0 (G1), graft m<-REF
#     at mlp L2/L3, ablate refs, dW cos vs REF; restore+assert per cell.
#   selection: eligible = parity & band; parents = two lowest R.
# verdicts per section 4 on frozen numbers; figures + metrics.json + NOTES.
```
