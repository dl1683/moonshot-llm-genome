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

Three removal conditions, matched for "forget-set p(name) ≤ 0.05 under TF":

1. **RMU-analogue (e065a):** direction u = normalized mean Δh_layer(
   forget − retain) at the causal depth d* (use e055's d4); fine-tune
   (≤2000 steps, ≤180 s cap, lr 1e-3, batch 32) with RMU loss on F
   (‖h′ − α·u‖², α≈20 toy-scaled down to keep LN sane — pilot α ∈ {2,5})
   + retain CE on R. Layer choice ablation: {d4, d5, both}.
2. **Address surgery (e065b):** the e043 removal recipe verbatim.
3. **Gradient ascent control (e065c):** matched-budget projected ascent on
   F only (we already know this fails; here it calibrates the instruments).

## Pre-registered readouts (freeze before running)

R1 **Suppression-depth transplant (e055 rig):** TF-state donor rescue of
p(name) at onset position, d1–d6 sweep, per condition.
   - H-obfuscation: RMU net still rescuable at ≥1 depth (knowledge intact,
     readout misdirected); surgery net not rescuable anywhere.
R2 **Probe-vs-generation gap (Orgad instrument):** linear probe for the
   name-token on layer activations vs actual free-run generation rate.
   - H-obfuscation: RMU gap WIDE (probe ≫ generation); surgery gap ≈ 0
     (both near floor — it's gone, not hidden).
R3 **Relearn scar (e044 battery, n=1 flagged):** 50-step and matched-CE
   re-exposure on F; recovery curve + address-direction regrowth cos.
   - H-obfuscation: RMU reverts fast (full recovery ≤ 50 steps, cos
     returns to fresh-from-zero baseline); surgery scarred (≥2× steps,
     cos stays high — replicates e044b).
R4 **Collateral:** val_R CE drift + fluency (entropy/top-k) per condition.
   Predicted: surgery ≈ 0 (e043), RMU small but nonzero at d5-only.

Discriminating structure: H-obfuscation predicts an INVERSION between R1
and R3 (rescuable-but-reverts-fast vs unrescuable-but-relearns-slow).
Any other pattern → write the alternative before rerunning.

## Cost & envelope

One RMU fine-tune ≈ ≤180 s GPU + eval-only rigs ≈ 5–8 min. Total wall
≤ 25 min, single net, no new training infra. Launch only when gpu_ok()
and no other GPU job live.

## Kill criteria

- RMU fine-tune cannot reach p(name) ≤ 0.05 without destroying val_R CE
  (> +0.3 nats) at any α/depth → toy-scale RMU infeasible; report as
  negative result, park head-to-head.
- If surgery ALSO shows wide probe gap / fast revert → our own removal
  claim weakens; escalate to full audit before any paper text change.

## Queue row (added on registration, not before)

| e065 | RMU-vs-surgery head-to-head | GATED (e053c) | obfuscation signature inversion: rescuable-but-reverts vs unrescuable-but-scarred |
