# e065 design — RMU-style unlearning vs address surgery: obfuscation head-to-head

Status: DESIGN (Rule-0 memo; not registered until reviewed). GPU-gated behind
e053c (no concurrent GPU jobs). Net: existing 0.84M B43-family checkpoints
only — no new training beyond the RMU fine-tune itself.

## Why now

- The paper's removal claim is parametric (embedding+lm_head row surgery =
  ASYMMETRIC-CHEAP-REMOVE, e043) with a causal scar (e044/e044b: relearn
  2.92× slower, address-direction cos 0.728 replicated n=2).
- 2025 consensus at scale (frontier scan): RMU-style representation-level
  unlearning OBFUSCATES rather than deletes — reverts in ~50 retraining
  steps ("LLM Unlearning Under the Microscope"; "Beyond Data Filtering").
- Nobody has run both on the same net with causal instruments. Our rig can:
  suppression-depth transplant (e055), relearn scar battery (e044),
  probe-vs-generation gap (Orgad). If RMU-at-toy-scale reproduces the
  obfuscation signature and surgery does not, the head-to-head becomes the
  paper's strongest "parametric surgery is the minority alternative" claim,
  now with a controlled comparison.

## Setup (all existing tooling)

Base: e043's name-installed net (or re-install via e043 protocol on a fresh
0.84M seed if ckpt absent). Forget-set F = prompts containing the installed
name windows; retain-set R = stratified val windows without the name.

Three removal conditions + two controls, matched for "forget-set p(name)
TF logit-gap within a pre-registered band" (critic fix 2: continuous
band, not a floor — threshold-matching hides 50× suppression-depth
differences):

1. **RMU-analogue (e065a):** direction u = normalized mean Δh_layer(
   forget − retain) at the causal depth; fine-tune (≤2000 steps, ≤180 s
   cap, lr 1e-3, batch 32) with RMU loss on F (‖h′ − α·u‖² + retain CE
   on R). **Step 0 (critic fix 3): measure ‖h‖ at the target depth
   first (eval-only, seconds); set α ∈ {1×, 2×} of the median
   activation norm — do NOT import the 7B constant; keep {2,5} only if
   norms happen to agree.** Mis-set α fails asymmetrically (too large →
   LN saturation → false kill; too small → no suppression → false
   "infeasible"), so the norm measurement gates the whole run.
   Layer grid {d4, d5, both} REQUIRED (critic fix 5: e056b resolved the
   site-selection circularity — rescue is position-general R1 — but the
   d4-vs-d5 stratum dependence stands; the grid is the honest answer).
2. **Address surgery (e065b):** removal = e042's D2 subtractive row-reset
   (critic feasibility note: e043 packages the INSTALL direction only;
   port the ~10-line row-reset from e042).
3. **Gradient ascent control (e065c):** matched-budget projected ascent
   on F only (calibrates the instruments).
4. **Retain-only fine-tune control (critic fix 2):** same steps/LR/batch
   on R only, minus any forget-set loss — measures the generic
   fine-tune/LR scar that the RMU arm also carries; its readouts are the
   null lines R1–R4 are compared AGAINST.
5. **No-removal control (critic fix 1):** the unmodified installed net —
   guards the expression-gap confound (T015: 0.974 TF accuracy but ZERO
   free-run expression PRE-removal, so R2's generation leg starts at
   floor for every condition).

## Pre-registered readouts (freeze before running)

R1 **Suppression-depth transplant (e055 rig):** TF-state donor rescue of
p(name) at onset position, d1–d6 sweep, per condition.
   - Bars inherited from e055 verbatim (critic fix 4): rescuable =
     site-mean R1 ≥ 0.30 with shuffled ≤ 0.05 and site-bootstrap 95% CI
     excluding 0; NOT rescuable = ≤ 2× base-net twin (0.0062).
   - H-obfuscation: RMU net still rescuable at ≥1 depth (knowledge intact,
     readout misdirected); surgery net not rescuable anywhere.
R2 **Probe-vs-generation gap (Orgad instrument):** linear probe for the
   name-token on layer activations vs actual free-run generation rate.
   - **Conditional pre-registration (critic fix 1):** if the no-removal
     control's generation rate is 0 (expected, per T015), R2 is PROBE-ONLY
     — the gap arm is uninformative at floor and is not counted as
     evidence for or against obfuscation; report dynamic range openly.
   - H-obfuscation (if generation > 0 in any arm): RMU gap WIDE (probe ≫
     generation); surgery gap ≈ 0.
R3 **Relearn scar (e044 battery):** 50-step and matched-CE re-exposure
   on F; recovery curve + address-direction regrowth cos.
   - H-obfuscation: RMU reverts fast (full recovery ≤ 50 steps, cos
     returns to fresh-from-zero baseline); surgery scarred (≥2× steps,
     cos stays high — replicates e044b, which is n=2; e065's own relearn
     cells are n=1 per condition, flagged as such).
R4 **Collateral (critic fix 4 — explicit bounds):** val_R CE drift with
   accept band ≤ +0.05 nats ("≈0"), 0.05–0.15 "small", > 0.15 fails the
   condition's comparability; fluency entropy drift band ±5%.
   Predicted: surgery ≈ 0; RMU small-but-nonzero at single-depth arms.

Discriminating structure: H-obfuscation predicts an INVERSION between R1
and R3 (rescuable-but-reverts-fast vs unrescuable-but-relearns-slow).
Any other pattern → write the alternative before rerunning.

## Cost & envelope

α-norm step + α{1×,2×} × depth{d4,d5,both} RMU grid (up to 6 × ≤180 s)
+ surgery + ascent + retain-only arms + eval-only rigs ≈ **≤ 40 min
wall** (critic fix 6 revision), single 0.84M net, sequential GPU with
cooldown(60–120) between fine-tunes. Launch only when gpu_ok() and no
other GPU job live (after e053c).

## Kill criteria

- RMU fine-tune cannot reach p(name) ≤ 0.05 without destroying val_R CE
  (> +0.3 nats) at any α/depth → toy-scale RMU infeasible; report as
  negative result, park head-to-head.
- If surgery ALSO shows wide probe gap / fast revert → our own removal
  claim weakens; escalate to full audit before any paper text change.

## Queue row (added on registration, not before)

| e065 | RMU-vs-surgery head-to-head | GATED (e053c) | obfuscation signature inversion: rescuable-but-reverts vs unrescuable-but-scarred |
