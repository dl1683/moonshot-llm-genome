# Post-g157b decision tree

**Date locked:** 2026-04-26 (before g157b verdict known)
**Purpose:** record the decision logic so the verdict-to-action transition is mechanical, not vibes-based.

**TOMBSTONE 2026-04-27.** g157b returned KILL_157b. Path C (mechanism rejection) was activated. The η > δ^mlp criterion is REJECTED (R7 in CLAIM_EVIDENCE_MAP). Implementation files `code/genome_157c_3seed_canonical_verdict.py` and `code/genome_157d_probe_budget_expansion.py` referenced below were deleted 2026-04-27 (entropy reduction; preregs retained per audit trail). This document is now a historical record of the decision logic — do NOT attempt to execute Path A/B paths from here. Active decision tree for the current branch is `post_g158c_decision_tree.md`.

## Inputs

g157b PILOT verdict is one of: DIRECTIONAL_SUPPORT_157b / WEAK_SUPPORT_157b / KILL_157b.

The locked criteria (reproduced from `research/prereg/genome_157b_eta_delta_probe_embedding_prefix_2026-04-26.md`):
- DIRECTIONAL_SUPPORT_157b: nat_G ≥ +0.02 nats (mean over 3 mid-band layers, seed=42 only) AND shuf_G ≤ 0 AND contrast ≥ +0.03 nats
- WEAK_SUPPORT_157b: nat_G ≥ 0 AND contrast ≥ +0.015 nats
- KILL_157b: otherwise

Per Codex `codex_outputs/g157_pilot_interpretation.md`, the v2 PILOT_KILL was rejected as not a real theory falsification. g157b is the canonical PILOT.

## Path A: g157b returns DIRECTIONAL_SUPPORT

The transport budget criterion η > δ^mlp is observed on the proper probe. Theory mechanism validated at single-seed pilot scale.

**Next actions (in order):**
1. Write `research/prereg/genome_157c_3seed_verdict_2026-04-26.md` LOCKED at 3 seeds × 3 mid-band depths × 2 conditions × 2 arms = 36 cells. Same FP32 + grad clip + embedding-prefix protocol. Compute estimate: ~3.5x the pilot = ~100 min, within envelope.
2. Launch g157c (3-seed verdict run).
3. In parallel: launch g158 (context-length inversion) since it's independent of g157c outcome.
4. After g157c verdict: if PASS, write CLAIM_EVIDENCE_MAP C16 promoting P12 to a confirmed claim. Update WIKI's §0.1 score from 6/10 to 7/10.

**Decision tree branch in this case:** the §0.1 framing claim becomes "Within 256-token English LM at 200M scale, the prefix-information transport budget criterion η_l > δ_l^mlp is directly measurable via a 3-probe layerwise paradigm AND collapses on order-destroyed controls."

## Path B: g157b returns WEAK_SUPPORT

Direction is right but signal weaker than threshold. Suggests:
- Probe budget too small (500 steps) — try 2000 steps
- Probe rank too small (kv_dim ~965) — try 1500
- Mid-band wrong — sweep all 5 depths

**Next actions:**
1. Run g157d (probe-budget expansion): IMPLEMENTED at `code/genome_157d_probe_budget_expansion.py`. Same scope as g157b but PROBE_STEPS=2000, all 5 depths, kv_dim=1500. Compute estimate: ~3.5x the pilot = ~100 min.
2. Launch g158 in parallel (independent).
3. If g157d still WEAK, the underlying η/δ probe paradigm cannot reach the 0.02-nat threshold at our scale. Pivot to: give up on g157 series, treat g156 PASS as the discrimination evidence, claim §0.1 score 6/10 (what we have).

## Path C: g157b returns KILL_157b

The η > δ^mlp criterion is empirically wrong on natural-minimal even with the proper embedding-prefix probe. The theory's proposed mechanism dies. g156's empirical inversion still stands.

**Next actions:**
1. Run g157 v3 (same-layer prefix with FP32 + grad clip) as a CONTROL. If g157 v3 also KILL → both probe variants reject the criterion. Mechanism is genuinely wrong.
2. Run g158 (context-length inversion) — does NOT depend on η/δ, tests the theory's other unique prediction.
3. If g158 PASSes: theory survives at the input-side prediction (transport demand is the control variable) even if the η > δ mechanism is wrong.
4. If g158 also KILLs: theory's two unique predictions both fail. The g156 inversion stands but is no longer part of a derivation chain — it becomes an isolated empirical observation.
5. Update CLAIM_EVIDENCE_MAP: P12 → REJECTED. Reframe §0.1 claim to "empirical inversion observed; proposed mechanism rejected; further work on alternative mechanisms."
6. **Pivot decision:** if both g157 v3 KILL AND g158 KILL, ABANDON the η/δ derivation route entirely. The empirical chain (g156) becomes a "narrow ablation paper" per Codex's adversarial audit and the project §0.1 axis pivots to:
   - Distillation product (g160 + g155 edge benchmark) as the manifesto-aligned cash-out
   - g159 (cross-class lesion) as the empirical extension
   - g161 (RWKV training) as the architecture-class extension

## Update 2026-04-26 22:37: lin probe pathology persists on shuffled

g157b mid-run observation (shuffled-baseline layer 5): even with FP32 + grad clip + skip-non-finite, the lin probe still produces CE=249 on shuffled distribution. The local + prefix probes are well-behaved (CE ~8 and ~7.8 respectively).

**Diagnosis:** the issue is NOT BF16 numerics. It's probe undertraining on out-of-distribution activations. A linear probe with 50M params at vocab=50277 and only 500 training steps × 32 batch = 16k samples is severely under-fit. On natural data the probe works; on shuffled the model's hidden states have very different statistics and the same probe config fails.

**Implication:** the `delta = CE_lin - CE_local` term is dominated by lin-pathology on shuffled. The G_l calculation will be massively negative on shuffled-baseline regardless of theory.

**Workaround for verdict logic:**
- If shuf_G_baseline is more than 10x more negative than nat_G_baseline (current g157b: ~-200 vs ~-3.7), the shuffled-arm `delta` is corrupted by lin pathology. Use ONLY the eta term (`CE_local - CE_prefix_embed`) for the discrimination criterion in this case.
- New criterion candidate: `eta_nat_minimal > 0 AND eta_shuf_minimal <= 0` — robust to lin-pathology because eta uses local + prefix CE only.

**Action:** when g157b finishes, compute BOTH the locked criterion (G_l based) AND this eta-only criterion. Report both. If they disagree, treat as AMBIGUOUS_PROBE_PATHOLOGY.

## What does NOT change regardless of g157b outcome

- g154 PASS (distillation pipeline validated) stands.
- g156 PASS (data-order inversion at 200M, 3 seeds, sharp) stands.
- g152 PARTIAL (compute-axis attenuation, CIs include zero) stands.
- All three are independent empirical observations.

## Critical: do not overinterpret single PILOT outcomes

The locked PILOT spec used 1 seed. ANY PILOT verdict is preliminary. The canonical 3-seed verdict requires a separate prereg + run. Per CLAUDE.md §4.1, PILOT data alone CANNOT promote claims in CLAIM_EVIDENCE_MAP from P-to-C status.

## Status tracking

When g157b finishes, update:
- `experiments/ledger.jsonl` (entry 172)
- `experiments/EXPERIMENTS.md`
- `research/CLAIM_EVIDENCE_MAP.md` P12 row
- `WIKI.md` ACTIVE EXPERIMENT QUEUE
- This decision document with realized verdict and chosen path
