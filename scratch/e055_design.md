# E055 — THE SUPPRESSION LOCALIZER (design memo, measured; REGISTERED)

Date 2026-09-25. Strategist item #1 double-down / paper-candidate upgrade
(T025-C, T028 sharpened). Net: `runs/checkpoints/e048_repro.pt` (the G1-gated
bit-level repro of e043 Dmix@s400, saved exactly at step 400 — NOTE
`e048_dose.pt` holds the s1600 endpoint, not s400; do not use as "dose@s400").
Design probes (CPU, no training): `scratch/e055_probe.py` (P1–P3),
`scratch/e055_probe2.py` (P4–P6, trajectories cached in
`scratch/e055_traj_cache.pt`); raw log `scratch/e055_probe2_out.log`.

---

## 0. REGISTRATION (frozen before the e055 run)

**Claim (T028 wording):** "factual-recall elicitation failure, causally
localized" — the first interventional depth-localization of an *expressed-vs-
installed* factual gap. Cite proactively: YOPO "You Only Pass Once"
(arXiv:2608.14465 — relay steering flips silently-encoded verdicts into speech
in the ABSTENTION domain; learned direction, not own-state transplant; elicitation,
not suppression-depth); Yan & Jia EMNLP-25 (arXiv:2502.20475 — promote-then-
suppress circuits exist, enumeration domain); Orgad ICLR-25 (probe-only; their
C/D cells are this population, AUC 0.59–0.68 — hence our pre-registered causal
discrimination metric below); Buckmann et al. coin "elicitation failure".
No prior transplants the model's OWN teacher-forced states into its free run
and reports a depth-survival curve (scratch/expression_gap_lit.md).

**Registered predictions:**

- **P1 (state-rescue — expression is a STATE phenomenon):** at some depth
  d ≤ 5, transplanting the battery-TF residual state at a free-run onset
  decision-position raises P(Z-first-char at the next position) to site-mean
  ≥ 0.30 (from baseline 0.004–0.234), while the same-depth shuffled-state
  transplant stays ≤ 0.05, with site-bootstrap 95% CI on (TF − shuffled)
  excluding 0. Verdict: knowledge present-but-suppressed; a one-position
  state write at depth d suffices.
- **P2 (no rescue — expression is a COMPUTATION/TRAJECTORY phenomenon):** no
  d ≤ 5 reaches 0.30 on the immediate readout AND the one-shot-write
  downstream readout (Z-word count in 60 free chars) is 0 at every d for
  both one-shot and held writes. Verdict: the free-run residual stream at no
  depth carries usable address state — expression cannot be state-injected.
- **P3 (mid-stack suppression — the causal-gate tie):** the rescue depth
  d* (shallowest d with off-geometry site-mean ≥ 0.30) lies in {2,3,4}, AND
  the off-geometry non-monotonicity (d1 rescue ≥ 2× the d2 value) replicates
  at ≥ 2/3 of deep-trajectory sites. Verdict: the suppression write lives
  between blocks 1 and 3 — the same region as the causal gate (T012 mode-3,
  T014 modes 3–5, R/B/R43 census).

**Discrimination metric (pre-registered per T028):** per depth d,
AUC_d = P(p_z[TF transplant] > p_z[shuffled transplant]) over all
site × donor pairs (ties 0.5); report AUC_d with site-bootstrap CI. Bar for
"rescue is knowledge-specific": AUC_d ≥ 0.90 AND meanΔ_d = mean_TF − mean_shuf
with bootstrap CI excluding 0. This is the causal analogue Orgad lacks.

**Gates (run-killers):** G1 self-patch identity (patched-with-own-state
logits == baseline, max|Δlogit| < 1e-4 — PASSED in probe); G2 d6 construction
(d6 patch reproduces the donor context's own onset p(Z) exactly — probe:
0.7749 vs 0.7749, PASSED); G3 e048-count gate (trajectory ZEPHYRA == 0,
ELIZABETH+FLORIZEL ≥ 3 per 8×350 — probe: 0 and 5, PASSED); G4 shuffled-floor
(unpatched random-prefix p(Z) ≤ 1e-4 — probe: ≤ 2.7e-6, PASSED).

---

## 1. MEASURED PROBES (what this design stands on)

All numbers: e048_repro (step-400 standard install), CPU, e043-frozen protocol
(splice RNG 24301; install-60 hosts FLORIZEL 19 / ELIZABETH 41).

### M1 — instrument gate (reproduce e048's battery readout)

| net | p(Z) @ ctx130 | argmax-Z | mean rank(Z) |
|---|---|---|---|
| e048_repro | **0.5563** | **0.817** | **1.18** |
| e001 base  | ~0 | 0 | 30.3 |
| e048_direct800 | 0.185 | 0.117 | 6.15 |

Matches runs/e048/metrics.json (0.556/0.817/1.18) — the install ckpt and the
onset instrument reproduce the published numbers bit-consistently.

### M2 — the address is a POSITION knife-edge (T019's open edge, resolved)

Same 60 occurrences, context length varied (terminal text identical except
where noted):

| context | p(Z) | argmax-Z | rank(Z) |
|---|---|---|---|
| truncate→130 (trained geometry) | **0.556** | **0.817** | 1.18 |
| truncate→129 / 131 | 0.120 / 0.124 | 0 / 0 | 3.7 / 3.1 |
| truncate→125 / 120 / 110 | 0.133 / 0.101 / 0.167 | 0 / 0 / 0.02 | 3.6 / 4.7 / 3.7 |
| truncate→140 | 0.145 | 0 | 2.9 |
| left-pad 5 / 10 (content identical, positions shifted) | 0.104 / 0.110 | 0 / 0 | 4.1 / 4.0 |

A ONE-character shift in either direction collapses p(Z) 4.5× and kills
argmax; left-padding (content fixed, only wpe indices move) collapses
identically. **Geometry binding is POSITIONAL (wpe 130 = the install-trained
name slot), not content.** T019's "battery geometry" now has a coordinate.

### M3 — free-run FROM battery geometry (the number e048 never measured)

From the 60 ctx130 prefixes (greedy): first char Z **49/60**, full ZEPHYRA
**49/60**. Sampled (T=0.8, top-k 40, 3×40 chars): **10 ZEPHYRA in 7,200
chars**; first-char dist Z 98 / E 47 / F 35 (of 180). The install DOES
express freely — inside the trained geometry. The e043/e048 zero-expression
results were measured from 120-char prompts = off-geometry by 10 positions.

### M4 — THE DIVERGENCE SITES + SUB-ARGMAX RANKS (the T028 measurement)

8 trajectories (e048 seeds/protocol; counts Z0/E5/F0 vs e048's Z0/E7/F2 —
same rates, not bit-identical: device RNG). Five incumbent onsets, all
ELIZABETH — these are the transplant sites. Divergence token = the onset
choice position (the model emits 'E'; the correct char 'Z' is sub-argmax):

| site | in-model pos | p(Z) | rank(Z) | argmax (p) | TF-complete p(Z·E·P·H·Y·R·A) |
|---|---|---|---|---|---|
| p0 t=120 | 120 | **0.180** | **2** | E (0.68) | [.18, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00] |
| p1 t=120 | 120 | **0.167** | **2** | E (0.78) | [.17, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00] |
| p7 t=120 | 120 | **0.234** | **2** | E (0.73) | [.23, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00] |
| p7 t=245 | 244 | **0.007** | **3** | E (0.94) | [.01, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00] |
| p7 t=415 | 255 | **0.004** | **3** | E (0.97) | [.01, 1.00, 1.00, 1.00, 0.99, 1.00, 1.00] |

Trajectory floor at non-onset positions (n=31): median 2.0e-8, p90 8.4e-7,
max 5.5e-5 → onset sites sit 2–5 orders above floor: the address is
site-selective even off-geometry. **Completion is context-general**: given
'Z', positions 2–7 of the name are ~1.00 everywhere, including deep
self-generated context. Direct800 reference (4 prompts): 3 onsets with
p(Z) 0.425 (rank 1, argmax Z!), 0.117 (rank 3), 0.278 (rank 2) — natural
learning pushes the prior past argmax at some slots; install leaves it
rank-2/3.

### M5 — donors, shuffled controls, write sizes

Donors (battery contexts, decision state at pos 129): ctx7 p(Z) .775,
ctx29 .752 (ELIZABETH hosts), ctx43 .672, ctx15 .666 (FLORIZEL hosts), all
argmax-Z. Shuffled (random corpus 130-char prefixes, same position): p(Z)
1e-8–2.7e-6. Free-site vs donor state relative distance by depth:
d0 .45, d1 .33–.38, d2 .32–.36, d3 .40–.45, d4 .37–.45, d5 .47–.55, d6 .45–.51
— a mid-stack dip; the write is 0.3–0.5 relative units at every depth.

### M6 — THE MINI-TRANSPLANT (feasibility; the money probe)

Cross-context: overwrite the residual state at the onset decision-position
(t−1) with the donor battery state, depth d (0=embedding input of block 0 …
6=final residual). Readout P(Z at next position). Mean over 4 donors (max):

| site (base) | d0 | d1 | d2 | d3 | d4 | d5 | d6 | shuffled max (any d) |
|---|---|---|---|---|---|---|---|---|
| (0,120) .180 | .221 | .161 | .296 | .342 | .391 | .479 | .716 | ≤ .001 |
| (1,120) .167 | .200 | .180 | .348 | .272 | .387 | .416 | .716 | ≤ .015 |
| (7,120) .234 | .264 | .270 | .194 | .279 | .325 | .458 | .716 | ≤ .029 |
| (7,245) .007 | .010 | **.163** | **.014** | .161 | .276 | .360 | .716 | ≤ .001 |
| (7,415) .004 | .004 | **.160** | **.003** | .105 | .238 | .398 | .716 | ≤ .000 |

1. **Rescue is knowledge-specific**: TF donors rescue to 0.1–0.7 at d3–d6;
   same-depth shuffled writes sit at floor (~0.000–0.03) — AUC 1.0 at every
   depth in this sample.
2. **d6 hits the donor's own onset p(Z) exactly** (construction gate).
3. **Mid-stack shape**: rescue grows through d3–d5 at all sites. At the two
   OFF-geometry (deep-trajectory) sites there is a **d1 peak → d2 crash**
   (.163→.014, .160→.003): the address survives in the block-0 output but is
   destroyed across blocks 1→2 in off-position context, then re-emerges as
   the donor state becomes self-sufficient (d4+). Prompt-terminal sites
   (in-model pos 120) show no crash.
4. **Base-net control** (same patch on e001): d0 3e-5, d2 3e-4, d3 3e-3,
   d4 1.6e-2 — the mid-stack rescue requires the INSTALLED knowledge; the
   untrained-for-Z stack destroys the address at every d ≤ 4 (d6 0.74 is the
   construction write).
5. **Downstream hint** (write HELD at the site, 60 chars × 2 samples × 2
   sites): base 0 Z-words; tf_d0 1/2 samples surface a Z-word; tf_d3 1/2;
   shuffled 0. Z-words appear in continued generation after the write
   (identity logging registered below; n tiny — this is feasibility, not a
   verdict).

### M7 — shared-prefix control (completion is invulnerable)

Force the divergence at the onset (ctx130+'E' where TF forced 'Z'), transplant
the TF state at the onset position: p(E at 131) = 0.999–1.000 at EVERY depth
{0,2,3,4,6} (baseline 0.001). Flat, by construction. **Given the 'Z', no
suppression exists anywhere in the stack — the entire expression gap lives
at the onset choice.** (This kills any "within-name suppression" reading and
concentrates e055 on the onset-decision state.)

---

## 2. The reframed phenomenon (what T019 meant, now with coordinates)

The expression gap = two measured mechanisms, not one:

1. **Position knife-edge** (M2): the installed address is read out only at
   wpe-130 geometry; free generation is off-geometry by construction
   (120-char prompts, then self-shifted positions).
2. **Sub-argmax prior at onset** (M4): off-geometry, the knowledge survives
   as a rank-2/3 prior at incumbent-name onsets (0.17–0.23 terminal,
   0.004–0.007 deep — both far above the 2e-8 floor), never argmax. THIS is
   the suppression e055 localizes.

Completion (given 'Z') is context-general and depth-invariant (M4 col 7,
M7). The direct-trained reference puts Z at argmax at some onsets — the
installed-vs-natural difference is precisely argmax-vs-sub-argmax at the
choice point (M4 reference row).

The transplant is the intervention that separates position-locked READOUT
from state-carried CONTENT: a battery state written at wpe-244/255 (off
geometry) still produces p(Z) 0.1–0.7 (M6) — the address content is portable
even though its native readout is not.

---

## 3. THE EXPERIMENT PROPER (lab/e055_suppression.py → runs/e055/)

### 3.1 Site harvesting

Generate 4 seeds × 8 e048 gen-prompts × 350 chars (T=0.8, top-k 40) on
e048_repro (batch the 8 prompts per seed — probe cost was 0.14 s/forward
single-sequence on CPU; batch-8 makes trajectories ~3 min total). Harvest
ELIZABETH/FLORIZEL onsets (registered primary: the install-host slots);
record for each site: p(Z), rank(Z), argmax+p, in-model (cropped) position,
terminal (t=120) vs deep (t>150) stratum. Expect ~20–30 sites (probe rate:
5 per 8 trajectories). Gate G3 as registered.

### 3.2 Transplant arms (the depth-survival curve)

Donors: the M5 four battery contexts (+2 held-out battery contexts as
replication donors). For each site × depth d ∈ {0..6} × donor:

- **A-TF**: overwrite residual at the onset decision-position with the
  donor battery state at its pos-129 decision state. Readout R1
  = P(Z-first-char at next position).
- **A-shuf** (control): same-depth overwrite with random-corpus-prefix
  states (M5 shuffled family, n=4).
- **A-rev** (reverse, secondary): donor = free-run site state, target =
  battery-context decision position — does off-geometry state SUPPRESS the
  battery readout (symmetry of the write)?

Write semantics, both registered:
- **one-shot**: state written only for the forward that emits the onset
  char, then released (tests persistence through the model's own dynamics);
- **held**: write persists at the site position for the whole continuation
  (the probe's semantics; the honest "state is in the stream" reading).

Readouts:
- **R1 (immediate)**: P(Z-first) at the next position (M6 instrument).
- **R2 (one-shot downstream)**: after a one-shot write, free-run 60 chars
  (4 samples, batched), count Z-words + record full text (probe gap: the
  mini did not log identities).
- **R3 (held downstream)**: same with the write held; plus p(Z) at the next
  name-onset inside the continuation (the "re-expression" readout).

Reference curves: identical protocol on e048_direct800 (its own onset
sites; expected: high baseline prior, shallow/trivial rescue — the
no-suppression reference) and on e001 base (destruction reference, M6.4).
Sensitivity (unregistered, report-only): mean-donor state transplant (the
"relay direction" analogue contrasting own-state vs direction methods).

### 3.3 Statistical plan

Per depth: AUC_d (TF vs shuffled over site×donor pairs), meanΔ_d with
site-bootstrap 95% CI (10k resamples), terminal/deep strata split. d* =
shallowest d with off-geometry site-mean R1 ≥ 0.30 and AUC_d ≥ 0.90.
Verdicts exactly per §0 P1/P2/P3 wording — no post-hoc bars.

### 3.4 Gates

G1 self-patch identity <1e-4; G2 d6 == donor p(Z); G3 trajectory counts;
G4 shuffled floor ≤1e-4; G5 generation determinism (rerun one continuation
bit-identical). Any gate failure → fix instrument, no verdicts.

### 3.5 Budget (CPU; ≤15 min)

trajectories ~3; harvest/onsets <1; A-arms R1: 24 sites × 7 depths ×
(6 TF + 4 shuf + 1 base + rev 7) ≈ 4.3k forwards ≈ 3 min batched; R2/R3:
8 sites × {d*, 6} × {one-shot, held, base} × 4 samples × 60 chars batched
≈ 7; reference nets ≈ 2. Total ≈ 16 min worst case → trim R2/R3 sites to 6
if over (registered fallback). GPU run permitted under the standard thermal
envelope (eval-only, e048 precedent) but NOT required.

### 3.6 Outputs

runs/e055/metrics.json (sites table, R1/R2/R3 curves, AUC/CI tables,
verdicts P1/P2/P3) + runs/e055/depth_survival.png (R1 vs d, TF vs shuffled
vs base-net, terminal/deep strata; R2/R3 bar inset). NOTES.md entry +
THINKING.md interpretation (≥2 alternative readings + discriminating
observation + registered follow-up) before anything builds on it. No new
automations; no edits to runs/ or data/.

---

## 4. Failure modes (what kills what)

- **P1 dies** if R1 rescue fails to separate from shuffled with CIs — then
  the mini's 0.1–0.7 vs ~0 was donor-leak (position-129 wpe content acting
  as a position cue, not knowledge). Discriminating check: A-rev arm +
  pad-shifted donors (donor at pos 139 by left-pad — same cue, no trained
  address; registered donor variant, run on 4 sites).
- **P2's persistence leg dies trivially** if held writes express but
  one-shot writes wash out at the next attention read — that outcome IS the
  "trajectory phenomenon" verdict for downstream expression even if P1
  holds for the immediate readout; report both, they are not in tension.
- **P3 dies** if d* lands at 0–1 or 5, or the d1/d2 crash fails to
  replicate — then suppression is not gate-tied and the paper claim shrinks
  to "state-rescuable, depth-unlocalized".
- **Site poverty** (<12 onsets): extend to seeds 4–7 (registered fallback);
  never relax the host-slot definition.

## 5. Scope & honesty

Single 2.7M char-transformer, one install protocol, one name — the claim is
mechanism-scoped ("an elicitation failure, causally localized, in a
controlled install"), not a universal. The battery-vs-free gap decomposition
(knife-edge + sub-argmax) is itself a finding T019 under-specified: e055's
THINKING entry must reconcile "p(Z) 0.556→1.7e-6" with the new coordinates
(wpe-130; rank-2 at 0.18 at terminal sites). Honesty reflex: every rescue
claimed here is an INTERVENTION changing the next-token distribution (M6)
and downstream sampling (M6.5), not a probe correlate — the Orgad gap.

---

## ADDENDUM (2026-09-25 ~22:00Z, takeover session — appended AFTER the frozen
registration; §0–§5 above unaltered)

**Provenance:** this memo DID land (file mtime 15:19:37 local, 3 min after
`e055_probe2_out.log` completed at 15:16:41) — the "stalled" design agent had
finished; the memo was simply never committed (untracked in git until this
session). Registration integrity preserved: nothing above was edited.

**Two single-forward measurements** (the registration's one explicit gap —
sub-argmax identities around 'Z' at the site; e048_repro, crop-256):

1. **Site p0/t120 onset distribution, full:** E 0.679 / **Z 0.180 (rank 2)** /
   M 0.136, then nothing above 9e-4. **'P' is rank 8 at 4e-4** — ZEPHYRA's
   interior chars carry no onset mass; the suppression is a three-way E/Z/M
   decision the install never wins. Free-run continues `"ELIZABETH:\nA"`.
2. **Same-occurrence binding twin** (single-prefix form of M2): prompt0 is
   `train_text[p-120:p]` of install occurrence 0 (ELIZABETH @175539); its
   ctx130 battery twin (same occurrence, 10 more context chars, terminal
   "QUEEN ") reads **p(Z)=0.535, rank 1, argmax Z**. Ten chars flip argmax
   E(0.68)→Z(0.54) on the same net/occurrence — the knife-edge is
   per-occurrence real, and the p0/t120 transplant site sits exactly on its
   own battery twin's 130↔120 boundary.

**Honesty cross-check of the lost probe1 stdout** (M1–M3 rows exist only in
this memo): M3 greedy first-char-Z 49/60 = 0.8167 EXACTLY equals M1's
frac_argmax_z 0.8167 (greedy follows argmax — bit-consistency passes); M2's
trunc→120 (0.101, n=60) is magnitude-consistent with e048's 8-prompt 0.0898.
Only M1's direct800 row (0.185/6.15) has no surviving cross-check — it is
reference-only, not load-bearing for any verdict.
