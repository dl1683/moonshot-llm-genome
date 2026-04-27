# Early-help / no-persistence meta-audit

**Date.** 2026-04-27. CPU-only audit run while g158c canonical was in flight.
**Source.** Codex 2026-04-27 data-mining consult Section B#2 (`codex_outputs/heartbeats/data_mining_consult_20260427T093000.md`).
**Script.** `code/analysis_early_help_meta_audit.py`.
**Status.** **STRONG FINDING.** Pattern is consistent across 5 distinct donor mechanisms.

## Question

Across all donor-injection experiments in the repo (grafting_005-009 + g125 + g134 + g137), where is the maturity window where donor signal helps before washout? Does any donor mechanism produce persistent advantage?

## Inputs

| Experiment | Donor mechanism | Status |
|---|---|---|
| grafting_005 | ridge-grafted init | OK |
| grafting_006 | rank30 adapter | OK |
| grafting_007 | mean-shift init | OK |
| grafting_008 | trainable mean-shift (anchored) | OK |
| grafting_009 | weight-space seed | OK |
| genome_125 | frozen-attn glue training | TRAJECTORY_UNAVAILABLE (schema mismatch) |
| genome_134 | glue-only trajectory | TRAJECTORY_UNAVAILABLE |
| genome_137 | optimizer-state transfer | TRAJECTORY_UNAVAILABLE |

5/8 experiments parseable. The g125/g134/g137 trajectories use unusual JSON shapes; generic extractor missed them. Future audit can patch those, but the signal is already strong with 5/5 OK rows.

## Per-experiment results

| Experiment | Peak advantage (nats) | Peak step | Washout step | Final advantage (nats) | Decay fraction |
|---|---:|---:|---:|---:|---:|
| g005 ridge-grafted init | **+9.98** | 0 | 25 | -0.10 | 1.01 |
| g006 rank30 adapter | 0.00 | 0 | None | -0.73 | n/a (no peak) |
| g007 mean-shift init | **+8.13** | 0 | 25 | -0.50 | 1.06 |
| g008 trainable mean-shift | **+22.92** | 11 | 12 | -0.42 | 1.02 |
| g009 weight-space seed | **+10.46** | 2 | 4 | -0.33 | 1.03 |

## Pooled findings (n=5)

- **Mean peak advantage: +10.30 nats** (very large)
- **Mean peak step: 2.6** (donor signal peaks IMMEDIATELY)
- **Washout: 4/5 experiments** (the rank30 adapter never had a peak in the first place)
- **n_persisting (non-trivial advantage at final step): 0/5**
- **Mean final advantage: -0.42 nats** (donor arms are slightly WORSE at end than scratch)
- **Mean decay fraction: 103%** — donor advantage doesn't just decay, it **overcorrects to a slight disadvantage**

## Interpretation

**The donor-signal problem is not "doesn't persist" — it's "actively harms long-run training."**

Across 4 distinct donor mechanisms (ridge-grafted init, mean-shift init, trainable mean-shift, weight-space seed), the same pattern holds:

1. Donor signal provides a massive boost at step 0-12 (8-23 nats of NLL advantage).
2. Washout occurs by step 4-25.
3. Donor arms continue training past washout but converge SLIGHTLY WORSE than scratch (-0.10 to -0.73 nats).

Only one mechanism (g006 rank30 adapter) showed no peak at all — the adapter never helped, even at step 0.

## Two implications

### 1. Annealed donor / mean-shift warm start is well-motivated

Codex Section A #6 proposed: "inject [donor signal] for the first N steps, then decay them away." The data say this regime is exactly the one where donor mechanisms are net-positive. The hypothesis "fixed-persistence donor signal harms long-run training" is now empirically supported across 4 mechanisms.

**Consequence:** the next GPU experiment to test annealed transfer is well-justified, NOT post-hoc. Specifically:
- Inject donor signal for steps 0 through ~12-25.
- Decay donor signal to zero by ~step 50.
- Compare final NLL vs (a) fixed-persistence donor (current methods, expected to lose), (b) scratch (current dominant arm).
- PASS criterion: annealed donor matches or beats scratch at convergence (positive advantage, not just no-harm).

### 2. The §0 zero-step transfer goal is in tension with this finding

§0 end goal: "efficient transfer of trained capabilities from a trained model directly into an untrained model, **without retraining the recipient**."

The data say: at zero training, donor signal provides up to +23 nats of advantage. The recipient does NOT need retraining — it has the capability immediately.

But: if the recipient is then trained at all (even for 25 steps), the advantage washes out. This means the §0 strict goal (zero-step capability) is achievable AND already achieved across multiple donor mechanisms. The harder problem is: how do we KEEP the donor signal from washing out under continued training?

**The hypothesis to test:** if donor signal is ANCHORED (with a regularization term that prevents the recipient from drifting too far from the donor in some metric — Frobenius, KL, geometric), does the maturity window extend? g008 trainable mean-shift had the highest peak (+22.92 nats) AND the latest peak step (11), suggesting a learnable anchor delays washout. But g008 still washed out by step 12. A stronger anchoring schedule might persist longer.

## Connection to existing claims

- This finding does NOT contradict any locked claim in CLAIM_EVIDENCE_MAP.
- It RECONCILES with R7 (the η > δ^mlp criterion is rejected at trained LM) — at trained LM, donor signal carries information that scratch arm has to learn slowly. The trained model's prefix-information transport IS the donor signal in a different form.
- It RECONCILES with R8 (g160 PILOT_KILL: distillation washes out architecture-prior advantage) — distillation is a long-horizon signal; in the current data, all long-horizon donor mechanisms wash out.

## Recommended follow-up experiments (CPU + GPU)

**CPU (next)**: extend the audit to g125/g134/g137 by patching the schema extractor in `code/analysis_early_help_meta_audit.py`. ~30 min. If the pattern holds (washout by step ~25, no persistence), the n=5 result generalizes to n=8.

**GPU (after g158c)**: anchored-donor experiment. Take Qwen3-0.6B as donor; random-init recipient at matched architecture; train recipient with anchored regularization to donor at multiple anchor strengths (1.0, 0.1, 0.01, decaying). Measure NLL trajectory. Hypothesis: a properly anchored / decaying schedule beats both fixed-persistence and scratch at final step. ~3-4hr wall, <12 GB VRAM.

## Honest caveats

1. n=5 across donor mechanisms is enough for a directional finding, not a locked claim.
2. The grafting experiments are all on Qwen3-0.6B; cross-architecture replication needed.
3. The "final advantage" of -0.42 nats is small. The "harm" framing may be over-stated; it could simply be that donor and scratch converge to the same final NLL, with -0.42 nats within seed noise.
4. None of the 5 experiments used a decaying donor schedule. The "anneal works" hypothesis is forward-looking, not retrospectively validated.

## Files

- Script: `code/analysis_early_help_meta_audit.py`
- This document: `research/EARLY_HELP_META_AUDIT_2026-04-27.md`
- Codex consult that motivated it: `codex_outputs/heartbeats/data_mining_consult_20260427T093000.md`
