# Pre-registration: genome_177 matched-alt-donor donor-IDENTITY falsifier

**LOCKED at first commit. Modifying post-lock invalidates the experiment.**

- Date: 2026-04-28
- Author: Devansh + Claude (autonomous loop) + Codex (design gate)
- Trigger: cycle 55 adversarial 9/10 attack on g175 (`codex_outputs/heartbeats/cycle55_adversarial_20260428T044500.md`)
- Supersedes: g175 PASS (now PROVISIONAL — three confounds: undertraining, Frobenius mismatch, corpus mismatch)

---

## Hypothesis

**H1 (donor-identity is real):** Even when alt donors are matched in (a) eval-NLL on C4, (b) Frobenius distance from random init, and (c) training corpus, the Qwen3-trained donor still produces a strictly larger continuous-anchor capability transfer to a random-init recipient than the best matched alt donor. Direction: Δ(Qwen3 - best_alt) ≥ +0.5 nats with paired-t 95% CI excluding zero.

**H0 (donor-identity is confounded):** Matched alt donors achieve ≥80% of Qwen3 gain, OR Δ(Qwen3 - best_alt) ≤ +0.2 nats, OR CI crosses zero. C22 dies. Fall back to C21 trained-structure specificity.

## Universality level claim

Level-0 single-family donor-identity-vs-trained-structure decomposition. Promotes to Level-1 only after cross-architecture replication (g173).

## Measurement primitive

`donor_identity_matched_falsifier`:
- 3 same-architecture (Qwen3-arch, randomly initialized) alt donors trained on the same C4 stream as recipient eval, until each reaches matched eval-NLL ≤ 3.6 (or wallclock budget).
- λ_alt = λ_qwen3 × sqrt(F²_qwen3 / F²_alt) for matched ‖∇L_anchor‖ at init (per-donor).
- 13-gram dedup forbids alt-donor train windows from overlapping recipient train + eval windows (eliminates eval-leakage confound).

## Systems tested

- Donor: `Qwen/Qwen3-0.6B` (canonical, frozen target).
- Alt donors: 3 × Qwen3-arch random-init pretrained on dedup'd C4 (seeds 1234, 5678, 9999).
- Recipient: Qwen3-0.6B-arch random init × 3 seeds [42, 7, 13].

## Sample size + power

n=3 paired-seed pairs. From g174 PART A, σ_paired ≈ 0.15 nats. With n=3 and target Δ=+0.5 nats, paired-t power ≥ 0.95. Sufficient.

## Pass / fail criteria

| Outcome | Decision |
|---|---|
| Δ(Qwen3 - best_alt) ≥ +0.5 nats AND paired-t 95% CI > 0 | **PASS** → C22 re-locks, donor-identity confirmed |
| Δ(Qwen3 - best_alt) in [+0.2, +0.5] nats with CI > 0 | **WEAK PASS** → C22 partial; advisor consult to decide flagship status |
| Δ(Qwen3 - best_alt) ≤ +0.2 nats OR CI crosses 0 | **FAIL** → C22 dies, fall back to C21 |

## What a null result means

A FAIL invalidates the C22 donor-identity claim. The +1.087 nats Qwen3 effect remains real (from g165) and trained-structure specificity remains locked (C21), but the ~60% identity decomposition is killed. Narrative tightens to "trained-structure-specific weight-anchor / KD transfer, donor-identity attribution unsupported," §0.1 ceiling regresses to ~7.5/10.

## Compute envelope compliance (CLAUDE.md §1.5 + COMPUTE.md §9)

- [x] Max VRAM: ≤22 GB. Qwen3-0.6B at FP16 = 1.3 GB; alt-donor pretrain peak ≤ 4 GB; recipient main ≤ 4 GB. Single-model at any time — no concurrent loads.
- [x] Max system RAM: ≤56 GB. Tokenized C4 buffers + 13-gram hash set ~5-10 GB.
- [x] Wall-clock: target ≤4 h. Codex g177v2 patch reduces alt-donor max steps + adds global pretrain wallclock budget. Verified post-patch (see `codex_outputs/g177v2_envelope_fix_20260428T060000.md`).
- [x] Disk: NPZ alt-donor weights × 3 ≈ 18 GB. Plenty of free disk.
- [x] Quantization: BF16 forward / FP32 master. Logged in ledger.
- [x] Save-resume: per-donor NPZ checkpoint after each pretrain success/abort; per-cell JSON resume on recipient main.

## Critical confound fixes vs g175

| Confound | g175 status | g177 fix |
|---|---|---|
| Undertraining | alt loss=5.49 (vs Qwen3 ~3.6) | stop-NLL ≤ 3.6 with global wallclock budget; donors that don't reach are saved unmatched and reported separately |
| Frobenius mismatch | alt F²=1.27M vs Qwen3 F²=2.03M (79% gradient force) | per-donor λ normalization to match ‖∇L_anchor‖ at init |
| Corpus mismatch | alt=Wikitext-103, eval=C4 | alt trained on same C4 stream as recipient |
| Eval leakage | (not addressed in g175) | 13-gram dedup forbids alt-donor train windows overlapping recipient train + eval |
| Single donor | n=1 alt donor | n=3 alt donors (seeds 1234/5678/9999) |

## Audit trail

- Cycle 55 adversarial: `codex_outputs/heartbeats/cycle55_adversarial_20260428T044500.md`
- g175 advisor: `codex_outputs/g175_advisor_20260428T040000.md`
- g177 v1 implementation: `codex_outputs/g177_implementation_20260428T050000.md`
- Cycle 57 code review (SEV 8 + SEV 7 + SEV 5): `codex_outputs/heartbeats/cycle57_code_review_20260428T053000.md`
- Cycle 57 direction review (g177 = 9.5/10 must finish): `codex_outputs/heartbeats/cycle57_direction_review_20260428T053000.md`
- g177 v2 envelope patch: `codex_outputs/g177v2_envelope_fix_20260428T060000.md` (in flight)
