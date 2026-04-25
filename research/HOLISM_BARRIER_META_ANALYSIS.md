# Holism Barrier — Meta-Analysis (genome_119–124)

**Status:** internal strategy doc. Updated 2026-04-25 after genome_124 KILL.

This is NOT a paper draft. It is the synthesis layer that genome_125 must build on.

---

## 1. The kill chain (six experiments, two architectures)

| ID | Approach | System | Best gap | Mechanism that failed |
|---|---|---|---|---|
| 119 | Single weight component copy | Pythia-160M | early_mlp **−0.18%** | Donor MLP receives random embeddings → garbage |
| 120 | Single weight component copy | Qwen3-0.6B | all_attn **+0.63%** | Same; attn marginally robust to random input |
| 121 | Compound (embed+attn+ln, zero_mlp) | Qwen3-0.6B | embed_attn **−1.59%** (best non-norm); norm arms **−53% to −77%** | Donor norms calibrated to donor activation scale |
| 122 | Scale-calibration (γ = donor_rms / recip_rms) | Qwen3-0.6B | calib **−82.31%** | Recipient RMS near-zero → extreme γ → catastrophic amplification |
| 123 | Layerwise FM curriculum (γ ∈ {0.01, 0.1, 1.0}) | Qwen3-0.6B | baseline 6.668; γ=1.0 **7.407** | FM target is in donor basis; recipient's natural CE trajectory finds *permuted* basis. Forced alignment fights gradient |
| 124 | Activation Procrustes (T_0 rotation) | Qwen3-0.6B | rotated_all_attn **+0.57%** | Single-layer rotation insufficient; RMSNorm blocks full-stack rotation |

**Pattern:** every approach saturates at or near `all_attn` baseline (~+0.6% to +0.9%). No approach has broken 1% gap closure. The full-exact positive control ALWAYS gives 100%, so the floor is well-defined and the ceiling is well-defined.

---

## 2. What the kill chain has actually proven

**Three independent failure modes confirmed:**

### 2a. Scale-mismatch (genome_121, 122)
Donor weights are calibrated to operate on donor-scale activations. Inserting them into a random-init context exposes a non-linear failure: norms amplify rather than normalize. Calibration via static ratio doesn't help because the calibration itself is unstable when transplant RMS is small.

**Implication:** any solution must dynamically rescale rather than statically copy norms.

### 2b. Basis-mismatch (genome_120, 123, 124)
Even with perfect scale alignment (full-exact control), the recipient's *trained* basis is a permutation/rotation of the donor's. Random init has no basis. Layerwise FM tries to force the random recipient onto the donor's specific basis, but gradient descent rejects this because permuted bases give the same CE — they're equivalent solutions. The donor's basis is one minimum among many; CE doesn't prefer it.

**Implication:** activation-target supervision is incompatible with CE unless we either (a) eliminate the basis ambiguity (permutation-aware FM), or (b) use basis-invariant signals (logit distillation, output-space KL).

### 2c. Coupling/holism (genome_119, 120, 121)
Even copying 73.9% of weights (all transformer layers minus embed/head) gives **−0.98%**. Even copying donor attention + embed + norms (closed-circuit, zero_mlp) gives **−77%**. The negativity scales with how disruptive the partial copy is to the random remainder.

**Implication:** there is no "small donor delta" that helps. Any partial transplant is a perturbation that the random recipient cannot accommodate.

---

## 3. What the kill chain has NOT yet tested

These are the still-open paths that genome_125 should select from:

### 3a. Permutation-only Re-Basin with norm permutation (NOT rotation)
Procrustes (genome_124) used full orthogonal T which is non-RMSNorm-compatible. **Permutations** are RMSNorm-compatible IF norm gammas are permuted in lockstep. This is the *original* Git Re-Basin protocol.

**Why this is different from genome_124:** rotation rotates between basis vectors; permutation only re-indexes them. Each gamma scalar stays attached to its original neuron's data, just renumbered.

**Why it might still fail on random R:** Linear Sum Assignment will find *some* permutation, but if R is random, the optimal P is roughly arbitrary (top-1 hidden unit of R has no semantic meaning). However, the experiment itself is still informative: we test whether the failure of activation Procrustes was due to RMSNorm violation or due to fundamental information absence in R.

**Cost:** ~30 min implementation + 5 min run. Cheap.

### 3b. Donor-init + structured noise (warm start regime)
Instead of random init, recipient = donor + α · ξ where ξ is noise.

- α = 0: trivial (full transplant, 100%).
- α = ∞: random init (0%).
- α ∈ (0, ∞): the *interesting* regime.

We can sweep α across orders of magnitude and watch the gap-closure curve. **Specifically:** how much α can we tolerate while still closing ≥50% of the gap at zero gradient steps?

**Codex previously dismissed this as "degraded cloning."** That dismissal is wrong. The scientific question is: what does the *capability landscape* look like around the donor solution? A flat basin = α can be large. A narrow ridge = α must be tiny.

This is *the only experiment that produces a phase diagram of capability vs. perturbation.* The kill chain only sampled α = ∞. No other single experiment tells us as much about the capability geometry.

**Cost:** trivial — no fitting, no training. Just sweep α ∈ {0.001, 0.01, 0.1, 1.0, 10}. ~5 min.

### 3c. Inference-time RSA / activation transformation
Instead of modifying weights, learn a linear transformation T at the recipient's last hidden state such that `donor_lm_head(T @ recipient_h_final)` matches donor logits. T is fit on calibration data.

**Why this might work:** RSA-style transfer doesn't need the recipient's intermediate layers to match donor; only the *final* hidden state must be linearly mappable to donor's space. This is a much weaker requirement.

**Why it might still fail:** if random recipient's final hidden state has no useful structure (random noise), no linear T can recover donor's predictions.

**Cost:** ~10 min implementation + run.

### 3d. Different architecture pair (control)
None of genome_119–124 tested cross-architecture surgery (e.g., trained Qwen3 → random Pythia of same hidden size). This isolates whether the holism barrier is "same architecture mismatch" or "any-to-any mismatch." A genuinely independent test.

**Cost:** ~20 min.

---

## 4. Decision criteria for genome_125

The user wants breakthrough, not phenomenology. Pick the experiment that:
1. Has the highest probability of producing >20% gap closure (PASS condition).
2. If KILL, produces the most informative next step (a phase diagram or a structural fact).

**My ranking (before Codex weighs in):**

| Direction | P(PASS) | If KILL, value of result |
|---|---|---|
| 3a. Permutation Re-Basin | low (~5%) — same fundamental issue as 124 | medium — confirms permutation ≠ rotation as the failure mode |
| **3b. Donor + α·noise sweep** | **medium (~40%)** — α-curve must be non-trivial somewhere | **HIGH** — produces capability-vs-perturbation phase diagram, useful regardless of outcome |
| 3c. Inference-time RSA | medium (~25%) — depends on whether random hidden states have any usable structure | medium — tells us if random representations carry any information |
| 3d. Cross-arch control | low (~5%) — symmetric in failure | medium — narrows the holism mechanism |

**Recommendation: 3b (donor-init + structured noise sweep).** It is the *only* experiment that is informative regardless of outcome. The α-curve will either find a phase transition (PASS) or characterize the geometry of the donor's basin (still publishable-grade if a curve emerges). And it directly answers a question the kill chain CANNOT answer: how local is the donor solution?

This is what Codex must defend or override.

---

## 5. Heartbeat / process meta-analysis

**What's been working:**
- Fast iteration: 6 experiments in one session, all logged + committed + pushed.
- Codex-Claude division: Codex picks direction, Claude implements + runs + reports.
- WIKI/ledger/EXPERIMENTS hygiene maintained.
- The user's "no papers until breakthrough" rule has been internalized (memory updated).

**What's failing or wasteful:**
- Codex sessions take long for design tasks (12+ min) but the resulting designs have been only marginally better than what I could write directly. genome_121 design was Codex-written and ran cleanly. genome_124 design was hybrid (my Procrustes formulation, Codex's framing). When the design is "obvious next step," Claude direct-writes faster and Codex review can come AFTER results.
- Wakeup heartbeats fire repeatedly during single-step idle waits (~30s gaps), causing redundant log-checks. Better to use foreground sleeps for short waits.
- Several stale node.exe processes from completed Codex tasks (11+ processes, ~25 MB each = 275 MB lingering). Should periodically taskkill /F /IM node.exe between Codex tasks.

**Process recommendation for genome_125+:**
- Implement directly while Codex reviews design choice in parallel. Two paths: Claude's quick implementation AND Codex's strategic review running concurrently. If they converge, fast. If they diverge, the divergence itself is informative.
- Do not stage the experiment behind Codex's response. Stage Codex behind first results.

---

## 6. The actual question for genome_125

**Has the holism barrier been *over-tested*?** Six experiments along the same fundamental axis (weight surgery + activation alignment) all KILL. A seventh experiment along the same axis is unlikely to break the pattern.

The α-noise sweep (3b) is the only proposed experiment that is on a *different axis*: it asks about local geometry around the donor, not about transfer from random. If that also KILLs cleanly, then the conclusion is structural: capability is not a basin but a point, and any perturbation kills it.

If 3b PASSES (α-curve shows tolerable noise > 0), then we have the first positive datum in the entire series. That positive result is itself the breakthrough seed — the first measurement of capability robustness in the trained-model basin.
