# WIKI — Neural Genome

**The living project registry. Agents read this first, always. Agents update this after every experiment, finding, or doc change — in the same commit. Never stale.**

Entries are pointers (≤500 chars). If an entry needs more depth, link to the canonical doc. This file is an **index**, not a document.

Codex's Cross-System Auditor checks WIKI consistency at every PR gate. A commit that changes experiment results / docs / primitives without a corresponding WIKI patch is rejected.

---

## THE END GOAL (POST-PIVOT 2026-04-29 cycle 72)

**Live framing:** *"the earliest token/embedding/interface geometry predicts whether training will be healthy, wasteful, or doomed."* Forecast/Diagnostic as the headline; falsification-discipline as the integrity story.

The strong-form transfer claim was tested and falsified on 3 axes (g177v2 / g173 / g181a). Surviving locked findings: tokenizer-prior trained-init effect within Qwen3-arch family. The pivot turns g181a's negative result into the mechanism for a training-triage diagnostic.

**Pre-pivot end goal (RETIRED 2026-04-29, retained as audit trail):** ~~"Efficient transfer of trained capabilities from a trained model directly into an untrained model, without retraining the recipient."~~ Falsified by g177v2 (donor identity 96% from undertrained alts) + g173 (cross-arch failed locked criterion) + g181a (tokenizer-prior dominates; transformer-block anchor HARMS).

**§0.1 honest baseline:** 5.0-5.3/10 (post g180 WEAK PASS). Full branch projections in CURRENT STATUS block below.

---

## ⚠ SCOPE LOCK — CS/AI/MATH ONLY (read first) ⚠

We are a CS / AI / math research group. End goal: **map the learning of every AI model** so we can diagnose capability, perform model surgery (transfer a capability from Model A into Model B without retraining), and ship tools for ML practitioners. **Biology experiments are DEPRIORITIZED.** We borrow biological principles as inspiration but do not replicate biology in this repo. See `CLAUDE.md §0.05` for the full scope lock. Any experiment, outreach, or synthesis that drifts into "let's also test on mouse V1 / organoids / cortex" — **stop and redirect**. Partners: Martian / Furiosa / Weka / Liquid AI / VERSES / NVIDIA. They care about capability transfer + efficient inference + geometry of *learned ML representations*.

---

## ⚡ CURRENT STATUS (2026-04-29, cycle 105) ⚡

**§0.1 honest score: 5.5-6.0/10** (post g180b FAIL). Branch projections:
- Current (g180 WEAK PASS + g180b FAIL): **5.5-6.0/10** (geometry forecast tokenizer-specific, not universal)
- g182 Model C'/C (manifold-only/pure geometry) beats arm_mean + telemetry: **8.6-9.0/10** (geometry alone predicts training health)
- g182 Model B beats arm_mean + combined_telemetry: **7.5-8.0/10** (confounded with telemetry — weaker claim)
- g182 PASS + phase 2 SSM/hybrid: **9.0/10** (adds non-attention family)
- g182 partial (one fold or only Model A): **6.5-7.0/10**
- g182 FAIL: **4.0-4.5/10** (geometry diagnostic is dead)

**g180b COMPLETE (27/27 cells) — FAIL.** Frozen g180 geometry model is tokenizer-specific. Primary: geometry+early_loss MSE=0.323 vs early_loss_only MSE=0.232, reduction **-39.4%** (geometry HURTS). Per-tokenizer: BERT -42.9%, T5 -96.4%, GPT-2 **+44.0%** (geometry wins ONLY on closest tokenizer). Shuffled permutation p=0.999 (anti-informative). KD universally harmful across all 3 families. Confirms g181a tokenizer-prior dominance. Source: `results/genome_180b_cross_tokenizer.json`.

**g182 Triage Arena RUNNING (cycle 104 — third restart).** Smoke PASS (12/12). Cycle 93 SEV-8 padding_side fixed. Cycle 96: scratch excluded from labeled set, added Model C/D ablation. Cycle 100 (A15): added Model C' (MANIFOLD_ONLY, 8 pure manifold features). **Cycle 102:** 6 bug fixes (NaN handling, tied-weight lm_head, P3 per-arm threshold, UTF-8 I/O, teacher cache). **Cycle 104:** prior process died silently during teacher gen (~13:10Z–15:58Z, no error in log). Restarted ~15:58Z, teacher gen in progress (GPU 78%, ~60-90 min for 8704 texts). Added progress logging for teacher gen. 7-model analysis: A/B/C/C'/D/E. Source: `codex_outputs/heartbeats/cycle102_code_correctness_20260429.md`.

**g184 pre-staging (cycle 94–101):** SSM compatibility verified. Mamba-370M BLOCKED (requires Triton, Linux-only). **Falcon-H1-0.5B WORKS** on Windows (naive SSM fallback, 1024d/36L, output_hidden_states=37 layers). Granite-4.0-Tiny also loads (hybrid MoE, 1536d/40L). g184 third architecture = Falcon-H1-0.5B (hybrid attention+SSM). **Cycle 101:** `frozen_eval_main()` fully implemented — Phase 1 (train frozen Ridge on g182 cells), Phase 2 (run 24 Falcon-H1 cells with native-tokenizer teacher), Phase 3 (frozen evaluation with bootstrap + permutation). Prereg: `research/prereg/genome_184_falcon_frozen_geometry_2026-04-29.md` (DRAFT, locks after g182 analysis). Ready to fire: `--frozen-eval falcon_h1` after g182 stage 1 completes.

**Cycle 95 adversarial (A13):** 6 attacks, 2× S10. (1) Shesha may erase moat — same-step geometry features from published library could match g182. (2) Scratch label=0 leak — FIXED cycle 96. (3) Model B not pure geometry — FIXED: added Model C/D ablation. (4) C23 narrows story to interface prior. (5) Umwelt = ceiling on cross-family claims. (6) Resolving: g182-Shesha Residual Kill experiment. Prior cycle 90 adversarial (A12) arm/protocol confound addressed by arm_mean baseline. Source: `codex_outputs/heartbeats/cycle95_adversarial_20260429.md`.

**Cycle 96 code review (A14):** S10 verdict gate fixed — compute_verdict was gating ALL models (C/D/E could fail and incorrectly make verdict FAIL). Now only co-primary A/B gate the verdict per prereg. Shesha augmentation bugs fixed (tokenizer loading, anchor loss replay, teacher text replay for seq_kd_full). Performance: C/D/E add ~66s to analysis (trivial vs training). Source: `codex_outputs/heartbeats/cycle96_code_review_20260429.md`.

**Theory backbone (cycle 105):** Derivation at `research/derivations/early_geometry_predicts_training_health.md` — 3 routes: (1) Fisher/NTK, (2) Rate-distortion, (3) Stat-physics symmetry breaking (most testable). **Cycle 105:** Route 3 Verdict Matrix PRE-LOCKED (8 outcome scenarios with §0.1 scores, interpretation fixed before data arrives). Added P6 (Landau nonlinearity: quadratic features test) and A16 arm-identity diagnostics (arm decodability + within-arm residualized Ridge). Codex adversarial SEV-10: arm identity may masquerade as geometry. Source: `codex_outputs/heartbeats/cycle105_adversarial_20260429.md`, `codex_outputs/heartbeats/cycle105_direction_review_20260429.md`.

**Competitive intel (cycle 93):** DIRECT COMPETITOR: "The Geometric Canary" (arXiv 2604.17698) — "Shesha" metric predicts steerability/drift from representational geometry (rho=0.89-0.97, 35-69 embedding models, detects drift before CKA in 73%). Code released (`shesha-geometry` on PyPI). Overlaps with our training-health pivot but focuses on POST-training steerability, not EARLY-training triage. Also: "Umwelt Representation Hypothesis" (2604.17960) directly challenges universality claims — modalities as local Umwelten, not converging to universal optimum. In-training probes competitor (2604.01025) achieves AUROC>0.75 on OLMo3-7B. Differentiator for g182: cross-architecture (Qwen3+GPT-2), pre-registered falsification discipline, early-stage geometry (not mid-training probes).

**Framing pivot (cycle 72 Q2): from "efficient transfer of trained capabilities" → "the earliest token/embedding/interface geometry predicts whether training will be healthy, wasteful, or doomed."** Forecast/Diagnostic is the new headline; falsification-discipline is the integrity story in the intro. The manifesto §0 wording overclaims against g177v2/g173/g181a and must be rewritten. **C18+C19+C21 dramatically narrowed**: the +1 nat effect is ~100% Qwen3-tokenizer+lm_head trained-init; anchoring transformer blocks HARMS. C22 REJECTED 08:50. C18/C19/C21 SURVIVE only as "tokenizer-prior trained-init transfer at recipient initialization" — not as "neural genome transfer of internal structure."

**★ g181a VERDICT: tokenizer-prior dominates (cycle 65 A7 9/10 attack CONFIRMED) — 2026-04-28 ~17:00 UTC ★**

| Arm | C4 NLL gain vs scratch | CI |
|---|---:|---:|
| full_anchor | ~+0.99 nats (reproduces g165 +1.087) | — |
| **embed_lm_head_only** (λ=0.0323, matched ‖∇L‖) | **+0.483 nats** | — |
| **no_embed_lm_head** (λ=0.0105, matched ‖∇L‖) | **−0.439 nats (HARMS)** | — |
| **no_embed − embed paired** | **−0.923 nats** | **[−1.055, −0.835] excludes 0 strongly negative** |

The continuous SGD anchor on transformer block weights actively HURTS performance vs scratch. Only the embed+lm_head anchor delivers gain. The +1 nat "transfer mechanism" is essentially Qwen3-tokenizer trained-vocabulary initialization being held in place during recipient training. This is NOT what "neural genome transfer" should mean — it is a tokenizer-init prior. Source: `results/genome_181a_tokenizer_isolation.json`, cycle 65 adversarial `codex_outputs/heartbeats/cycle65_adversarial_20260428T091500.md`.

**★ g177v2 VERDICT: FAIL ★ (2026-04-28 ~08:50 UTC, wall ~2.92h)**

3 alt donors on C4 (matched corpus, λ-normalized, 13-gram dedup'd) at NLL ~5.72 each give 95-96% of Qwen3's +1.087 nat anchor effect. Donor-identity-specific component is tiny: Qwen3-minus-best-alt = +0.038 nats [CI +0.018, +0.068]. Cycle 55 + cycle 60 adversarial A6 attacks confirmed: the active ingredient is "any sufficiently trained Qwen3-arch checkpoint" — NOT donor-identity. **C22 REJECTED.** Source: `results/genome_177_matched_alt_donor.json`, `research/CLAIM_EVIDENCE_MAP.md`.

**g173 LAUNCHED** (cycle 63 direction Q1: regardless of g177v2 verdict). Tests cross-arch generalization → falsifies/confirms A6 same-family-basin attack at the cross-family level. ~3.85h ETA.

**Cycle 60 limitations logged:**
- A5 (cycle 57 desk audit): C20 late-KD has soft matched-compute-null gap; kept at Level-0.
- A6 (cycle 60 adversarial 8/10): C18/C19/C21 are single-architecture/task/λ. Honest framing is "trained-structure-specific continuous constraint within Qwen3-arch family" — the strong "neural genome transfer" framing requires g173 cross-arch PASS.

**Locked claims:** C17 (g158c transport-demand) + C18 (g165 weight-anchor) + C19 (g167 KD canonical) + C21 (g174 trained-structure specificity, both axes) — all matched-null backed.

**Live falsifier:** g177v2 RUNNING with `--allow-unmatched-donors` (alt-donor pretrain ~2.5h, then main 5×3=15 cells ~50min). Reframed per Codex sanity check (`codex_outputs/g177v2_unmatched_decision_20260428T062000.md`): NOT matched-condition parity (computationally infeasible at NLL 3.6 on RTX 5090) but **matched-corpus, force-normalized sensitivity probe**. Pass requires Δ(Qwen3 - best_alt) ≥ +0.5 nats AND Qwen3 above 95% PI of NLL×Δ fit extrapolated from 3 alt donors. If Qwen3 within PI → undertraining-dominant, claim dies. Active fixes vs g175: corpus parity + 13-gram dedup + per-donor λ normalization + n=3 same-arch.

**Queued post-g177v2 (revised per cycle 63 direction review 2026-04-28 ~08:24):**

**★ g173 VERDICT: FAIL on locked criterion (cycle 70 adversarial 10/10 rejected reframe) — 2026-04-28 ~13:10 UTC ★**

Locked PASS = final-accuracy ratio ≥1.5x → got **0.99x** (fail). Per-arm c3_macro means (3 seeds, near random chance):

| Arm | C3 mean | Gain |
|---|---:|---:|
| scratch_ce_llama (173M) | 39.87% | — |
| kd_logit_llama | 42.16% | +2.29pp |
| kd_late_only_llama | 41.18% | +1.31pp |
| scratch_ce_qwen_arch (596M) | 40.91% | — |
| kd_logit_qwen_arch | 41.71% | +0.80pp |
| kd_late_only_qwen_arch | 41.31% | +0.40pp |

**Cycle 70 adversarial REJECTED the post-hoc gain-ratio reframe** (10/10 methodology drift, 9/10 underpowered near-chance benchmarks, 8/10 param-count + tokenizer confound). Honest external claim: *"the preregistered criterion failed; a post-hoc gain-normalized analysis suggests Llama may benefit more from KD than Qwen-arch — hypothesis-generating, not confirmatory."* Resolving experiment requires fresh prereg with ≥10 paired seeds, matched param-count, same-tokenizer + native-tokenizer arms.

§0.1 honest read drops from cycle-60-projected 8.0-8.4 → **5.5-6.0** (g173 fails its own criterion; cross-arch transfer remains hypothetical). Source: `results/genome_173_cross_arch_flop_cashout.json`, `codex_outputs/heartbeats/cycle70_adversarial_20260428T131000.md`.

**★ g180 VERDICT: WEAK PASS — geometry forecast beats early-loss baseline by 62% MSE reduction, but bootstrap CI crosses zero (cycle 73, 2026-04-29 ~06:34 UTC) ★**

| Metric | Baseline (early-loss only) | Full (geometry + early-loss) |
|---|---|---|
| Held-out MSE (n=9 Llama cells) | 0.01548 | 0.00595 |
| R² | −0.941 | +0.254 |
| MSE reduction | — | **61.6%** |
| Paired bootstrap 95% CI | — | [−0.0009, +0.021] (crosses zero) |
| p(improvement > 0) | — | 96.3% |

The 25% MSE reduction threshold is cleared (61.6%), but the paired bootstrap CI lower bound is −0.0009 — just barely crosses zero. With only 9 test cells, the CI problem is sample size, not signal strength. Baseline R²=−0.94 means early loss alone ANTI-predicts cross-family runs; geometry is the only useful signal.

**g180b FAIL (cycle 91):** Frozen g180 geometry model is tokenizer-specific. MSE reduction -39.4% (geometry HURTS). Per-tokenizer: BERT -42.9%, T5 -96.4%, GPT-2 +44.0% (wins only on closest tokenizer). Permutation p=0.999. P17 does NOT promote. Pivot to g182 Triage Arena (architecture-explicit, LOAO CV, 9 baselines).

**g181b long-horizon attenuation PASS** (embed_lm_head_only_anchor × 3 seeds + scratch × 3 seeds, 5000 steps each). All 6/6 cells complete. **Mean gap +0.513 nats** (per-seed: +0.531, +0.486, +0.523). Gap trajectory: +0.387@500, +0.481@2000, +0.502@3000, +0.518@4000, +0.513@5000 — stable plateau. **Resolves A8 (500-step horizon artifact).** Claim C23 locked. Source: `results/genome_181b_long_horizon.json`.

**g180b ENHANCEMENTS (cycle 75):** shuffled-geometry permutation test (1000 iterations, p-value for real vs random feature ordering), trajectory loss logging at steps {20,40,60,80,108} for post-hoc trajectory-baseline analysis. Both address adversarial A9 attacks.

**Adversarial A9 (cycle 75, severity 9/10):** effective n=3 not n=9 (3 seeds × 3 arms); baseline too weak (scalar loss only, need trajectory); g180b = Qwen-shell tokenizer swap not cross-arch; big labs view as internal telemetry. Resolving: g182 "Blinded Training Triage Arena" (72 cells, Qwen3+GPT-2 arch, 12 seeds, 5 baselines + combined telemetry, block-bootstrap). Source: `codex_outputs/heartbeats/cycle75_adversarial_20260429.md`, `codex_outputs/cycle75_direction_review_20260429.md`.

**g182 prereg LOCKED (cycle 76):** Codex design gate APPROVED after 3 iterative reviews (v1→v2→v3). All 8 fixes applied: embed_anchor arm, fresh matrix, combined telemetry, normalized labels, train-fold-only Ridge, reference-free co-primary geometry, expanded 9-baseline suite, §9 compliance checklist. Minor advisories resolved: smoke-test includes interrupt/resume, Ridge alpha grid predeclared [0.01–1000], wall-clock gate corrected to 4.2h/12-cell. Implementation gated on g180b + g181b completion. Source: `research/prereg/genome_182_triage_arena_2026-04-29.md`, `codex_outputs/g182_design_gate_v3_20260429.md`.

**g180b blocker fixes (cycle 76):** (1) teacher-text vs C4-val 13-gram overlap check added + forbidden_hashes passed to teacher tokenization, (2) feature cache validation strengthened (checks target_step + genome version). Both from Codex correctness review. Source: `codex_outputs/g180b_correctness_perf_20260429.md`.

Source: `results/genome_180_forecast.json`, `research/prereg/genome_180b_cross_tokenizer_2026-04-29.md`, `codex_outputs/g180b_design_gate_20260429.md`.

**★ PIVOT 2026-04-28 ~17:10: transfer-mechanism story dead → Genome Forecast / Diagnostic ★**

Per g181a advisor (`codex_outputs/g181a_next_direction_20260428.md`): "the transfer-mechanism story is dead in the strong form ... the highest section 0.1 move is to pivot the headline to Forecast/Diagnostic." 9.0/10 advisor pick. New flagship question: **"Can we predict final run failure or compute efficiency from zero-to-3% tokenizer/embed geometry better than early loss?"**

The pivot turns negative findings into mechanism: g181a showed tokenizer/embed init dominates the +1 nat anchor effect. If early tokenizer/embed geometry also predicts run health, that's a training triage instrument — practitioner tool, cross-arch from day one, cheap, falsifiable. Bar: held-out runs across multiple tokenizers/architectures, AUROC for bad-run risk, simulated compute-savings policy, must beat early-loss baseline.

**Sequencing revised (post-pivot 2026-04-28 ~17:10):**
1. **g180 Genome Forecast (refocused on tokenizer/embed geometry)** — biggest §0.1 uplift if the geometry beats early-loss baseline. Codex implementation prompt staged. Launch after Codex writes the script.
2. **g181a tokenizer-isolation control (NEW, cycle 65 9/10 attack)** — CRITICAL. Tests whether the surviving "+1 nat" claim is just Qwen3-tokenizer-prior. 4 arms × 3 seeds × 2000 steps = ~2.6h. Survives only if no_embed_lm_head_anchor retains ≥+0.5 nats AND beats embed-only by ≥+0.3 nats. **If FAIL → C18+C19+C21 collapse to "Qwen3-tokenizer init artifact."**
3. **g181b long-horizon (cycle 65 8/10 attack)** — scratch + full_anchor × 3 seeds × 5000 steps = ~3.3h. Survives only if gap at step 5000 ≥+0.5 nats. **If FAIL → claim narrows to "short-horizon acceleration" only.** Order: AFTER g181a survives.
4. **g180 Genome Forecast (advisor pick 8.6/10)** — biggest §0.1 uplift candidate. NEW direction: early-checkpoint destiny predictor. Cross-arch from start. **Order: gated on g181a+g181b survival.** If g181 series kills the transfer claim, g180 becomes the pivot anyway.
5. **g179 λ-sweep derivation** — gated on g181a+g181b survival.
6. **g178 layer-family** — DEFERRED indefinitely.

**g177v2 NLL × Δ scatter formal verdict (advisor 08:48):** Fit `Δ = 3.116 − 0.362·NLL`. At Qwen3 NLL=3.565, predicted Δ=+1.825, 95% PI = [-0.998, +4.648] (huge due to df=1, 3 alt donors clustered at NLL ~5.72). **Qwen3's observed +1.087 is INSIDE PI and even below the point estimate.** Identity-residual NOT confirmed. C22 fully REJECTED, not just downgraded.

**Moat (cycle 63 Q3):** the adversarial negative-control discipline — "random-init recipient + continuous donor constraint + matched nulls + kill criteria, trained structure works as an active basin force during SGD while zero-step transplant and decay fail." Big labs optimize deployed transfer; they don't publish adversarial nulls.

**g177v2 verdict (cycle 60-63 reframed)**: NLL × Δ scatter with n=3 alt donors clustered at NLL ~5.72 cannot honestly relock C22. Verdict reads as "Qwen3 exceeds local undertrained-alt trend" — sensitivity decomposition. **Concerning early signal**: anchor_alt_donor_seed_1234 seed=42 step 425 reaching NLL 4.46 — if scratch ≈ 5.5, alt donor gives Δ ≈ +1.04 nats vs Qwen3's expected +1.087. Undertraining-dominant attack from cycle 60 may play out.

**Hardware blocker:** wall-power meter for g155 (only honest 8.5+ path). Procurement priority #1.

**Codex cadence:** cycle-3 dual review + cycle-5 adversarial + advisor on every experiment completion (per HEARTBEAT.md). Cycle 55 adversarial caught the g175 confound; cycle 57 code review caught g177 SEV 8/7 before the run wasted 6+ hours.

---

## Historical Experiment Queue (ARCHIVED — see EXPERIMENTS.md)

> **Anti-entropy note (cycle 77):** The detailed experiment queue from the pre-pivot era (g152–g161, architecture-prior chain) has been archived. Code files genome_087–161 were deleted. All results preserved in git, , and . Key verdicts: g156 PASS_TRANSPORT, g157/g157b PILOT_KILL (eta mechanism rejected), g158 PILOT DIRECTIONAL_SUPPORT, g158c PASS_canonical (context-length inversion locked), g159 INCOMPLETE/scale-limited, g160 PILOT_KILL.

## ⚠️ BLOCKERS — surface for procurement / unblocking

1. **External AC wall-power meter** — only hard prerequisite gating g155 (8.2/10 PASS, the only direction that BREAKS the §0.1 ceiling). Yokogawa WT310E gold; Tasmota-flashed Sonoff Pow R3 / Shelly Plug S Plus practical. Without it the locked g155 prereg cannot execute honestly per the integrity bar (nvidia-smi proxy explicitly disallowed in the headline). Acquisition is the project's #1 procurement priority.

## ★ NARRATIVE LOCK (Codex cycle 42, 2026-04-27 — PRE-PIVOT historical context)

> **NOTE:** This narrative was SUPERSEDED by the cycle 72 pivot to Forecast/Diagnostic (see CURRENT STATUS above). The "continuous donor-information-in-the-loss" theory was tested and partially falsified: g177v2 showed donor identity is 96% from undertrained alts; g173 cross-arch FAIL; g181a showed tokenizer-prior dominates. Retained as audit trail for how findings evolved.

**Pre-pivot locked findings (g158c→g174, cycle 42 framing):**
- g158c: transport-demanded positions (long context) = WHERE donor info matters (PASS_canonical, rho=+0.933)
- g165: continuous weight-anchor = +1.088 nats persistent (PASS)
- g167: continuous KD logits = +1.014 pp persistent (PASS, second independent axis)
- g168: zero-step transplant KILL (alignment-loophole dead)
- g169: decay scaffold KILL (temporary-scaffold dead)
- g170: transport-gated KD KILL (uniform KD beats transport-aware weighting)
- g172: late-KD (33% compute) yields 69% of full KD effect (rich finding)
- g174: trained-structure donor-specificity PASS on BOTH weight-anchor and KD axes
- g177v2: alt donors give 95-96% of Qwen3 effect → C22 REJECTED (donor identity dead)

---

## ⚡ TIER-0 FRAMING — READ BEFORE EVERY ACTION ⚡

**We are ONE independent researcher competing against DeepMind, Anthropic, OpenAI, Google, Meta.** Every action must advance toward: (a) **first-principles derivation**, not phenomenology, (b) a finding the big labs architecturally cannot/will not publish, or (c) **electricity-grade efficiency** on a real task. See `CLAUDE.md §0.1`.

**Current distinctive direction (post-pivot 2026-04-29):** Early-training geometry predicts run health across architectures (g182 Blinded Triage Arena). Big labs don't pre-register falsification-discipline experiments with strict kill criteria. Our moat is adversarial integrity + cross-architecture generalization.

**When in doubt, fire Codex with:** *"given current state, which action has the highest probability of producing a finding DeepMind cannot or will not produce?"*

---

## How to read this file

- Every section is scannable. Scan it.
- Every claim has a pointer (`→ file:section` or `→ ledger:<id>`).
- Anything undated is current. Anything dated is as-of that date.
- If a pointer is stale, fix it in the same session you noticed it.

## How to update this file

After any of the following, patch the relevant section(s) in the **same commit** as the change:

- Experiment run (any `genome_*.py` with a ledger entry)
- Primitive added, promoted, or demoted
- Bestiary change (system added, marked broken, reclassified)
- Mystery progress (hypothesis confirmed/falsified, new priority, scar flag)
- Finding promoted to a universality level
- Doc created, renamed, or deleted
- Anti-entropy pass (files deleted, merged, renamed)

No "update WIKI later." If the change exists in git, WIKI reflects it.

---

## 1. Project state at a glance

| Field | Value |
|---|---|
| **Phase** | 1 — Instrument live; **first 🟡 coordinate promoted 2026-04-21** |
| **Axiom status** | **G1 + G2.4-text + G2.5-biology all PASS as of genome_034 2026-04-21.** 9 trained neural networks across 7 training objectives produce `C(X,k) = c_0·k^p` with `p = 0.179 ± 0.021 (CV 12.0%), R² mean 0.997` across 27 cells. The 12% CV decomposes into: **text systems converge to `p ∈ [0.158, 0.177]`, vision systems to `p ∈ [0.210, 0.223]`** (Δ ≈ 0.06 modality gap; verified via 4 systems × 3 stim seeds = 12 cells, per-system CV 1.6–3.4%). **Random-init twins span `p ∈ [0, 0.37]` (22× wider)**, across 15 cells → training is a modality-stratified convergence operation toward a shared fixed point. **Biology bridge passes (genome_034, 10/10 Allen V1 sessions at δ=0.10, 8/10 at δ=0.05, prereg criterion cleared by 40 and 20 points respectively)**. Only open criterion for Level-1 is G2.3 theoretical re-derivation (v1 FALSIFIED, 3 of 4 v2 sketches FALSIFIED, framework D untested). |
| **Bestiary coverage** | **9 / ~13 classes measured** (through genome_022 2026-04-21): classes 1 transformer / 2 reasoning / 3 recurrent / 4 hybrid / 6 vision ViT / 7 BERT-MLM / 8 MiniLM-contrastive-text / 9 I-JEPA-predictive-masked / 10 CLIP-contrastive-vision + **NEW: 11 DiT-XL/2-256 class-conditional diffusion transformer (genome_021+022, 3-seed n=2000 cluster-join)**. **kNN-k10 + power-law passes on 9 classes** (Falcon narrow-fail at n=2000, tips at n=4000). Spans **7 distinct training objectives** (CLM + reasoning-distilled + MLM + contrastive-text + self-supervised-ViT + contrastive-vision + predictive-masked + diffusion-denoising). |
| **Promoted primitives (🟢¹/🟢²)** | 0 |
| **Gate-1 passed (🟡 coordinate)** | **1 CLEAN 🟡 + now 9-class extended**: kNN-10 clustering coefficient + power-law form `C(X,k)=c_0·k^p`, scope `(modality ∈ {text, vision}, stimulus_family ∈ {c4_clean.len256.v1, imagenet1k_val.v1}, pooling ∈ {seq_mean, cls_or_mean})` on **Qwen3-0.6B + RWKV-4-169M + DINOv2-small + DeepSeek-R1-Distill-Qwen-1.5B + Falcon-H1-0.5B + BERT-base + MiniLM-L6 + CLIP-ViT-B/32 + I-JEPA-ViT-H/14 + DiT-XL/2-256**. Prereg `research/prereg/genome_knn_k10_portability_2026-04-21.md` LOCKED. **Biology interim (genome_027, n=4/10 sessions)**: 100% pass rate at δ=0.10 vs DINOv2 band. |
| **Active mysteries** | 7 (unchanged; H11-H13 are hypotheses, not mysteries) |
| **Scars (🩹)** | 0 |
| **Active hypotheses (H-register)** | 14 — H1..H10 original + H11 Koopman + H12 stimulus-dominance + H13 quantization-stability + H14 subsample-stability. H15 retired to governance rule (modality-scope is policy, not falsifiable). Atlas TL session file deleted in anti-entropy; hypotheses are historical context from pre-pivot era. |
| **Open pre-registrations** | **4 locked:** `research/prereg/genome_180b_cross_tokenizer_2026-04-29.md` (cross-tokenizer forecast), `research/prereg/genome_182_triage_arena_2026-04-29.md` (Blinded Training Triage Arena, §0.1=8.6), plus 2 atlas-era prereg from 2026-04-21 (superseded by post-pivot focus). |
| **Phase-3 claims** | 0 (Gate-1 ≠ Level-1; v1 derivation FALSIFIED; empirical power law `C(X,k)=c_0·k^p` with **p=0.179±0.021 (CV 12.0%), R²>0.989 mean 0.997 across 27 cells (9 architectures × 3 depths × seeds)** stands as stronger-than-originally-claimed replacement. 2026-04-21 v2-derivation pilots RULED OUT 3 of 4 simple algebraic sketches: **framework A (fractal d_2/d_int) FALSIFIED** wrong-sign structurally (genome_024); **framework B (doubling-dim ratio) FALSIFIED** magnitude-absurd (genome_026); **framework C (heavy-tailed NN-degree) FALSIFIED** wrong-sign (genome_020). Only **framework D (rate-distortion) untested**. All 3 falsifications predict wrong sign or huge magnitude → v2 mechanism likely needs non-dimensional / information-theoretic / correction-to-leading-order class of argument. Pilot details: `research/derivations/power_law_v2_candidates.md`. **Separately (genome_028 negative control, 2026-04-21):** untrained-twin power-law exponents span `p ∈ [0.021, 0.355]` (16.9× spread) on 3 systems vs trained 1.1× spread → training is a CONVERGENCE operation toward the cross-arch universal, not an architectural constant. This is the strongest single manifesto-claim datum collected to date. |
| **Active TL session** | ARCHIVED — atlas TL session file deleted in anti-entropy (cycle 77); atlas work paused for Forecast/Diagnostic direction |
| **Gate semantics** | LOCKED in pre-pivot atlas session (gate spec retained in `research/MEASUREMENT_PRIMITIVES.md`) |
| **Next phase trigger** | Phase 1 begins when TL session converges to blueprint AND a Gate-1 prereg is locked AND smoke test passes |

→ Phase definitions: `README.md` §Status.

---

## 2. Canonical docs index

Single source of truth per topic. If a topic's pointer goes stale, update here first, then fix the pointer.

| Topic | File |
|---|---|
| Public face | `README.md` |
| Agent operating manual | `CLAUDE.md` |
| **This index** | `WIKI.md` (you are here) |
| **Compute envelope (binding)** | `COMPUTE.md` — read at every design gate |
| Intellectual framing | `research/MANIFESTO.md` |
| Universality framework | `research/UNIVERSALITY_LEVELS.md` |
| Measurement toolkit | `research/MEASUREMENT_PRIMITIVES.md` |
| System bestiary | `research/SYSTEM_BESTIARY.md` |
| Unresolved phenomena | `research/OPEN_MYSTERIES.md` |
| Experiment log (human) | `experiments/EXPERIMENTS.md` |
| Experiment ledger (JSONL) | `experiments/ledger.jsonl` |
| Pre-registrations | `research/prereg/` (one file per experiment, dated, locked) |
| Claim-to-evidence map | `research/CLAIM_EVIDENCE_MAP.md` — every public claim maps to a ledger entry + locked prereg |
| Repo-wide model registry | `../../models/MODEL_DIRECTORY.md` + `../../models/registry.py` |
| **Grafting subproject** | `grafting/OBJECTIVE.md` — geometry-first initialization for efficient training |

Any markdown file not in this table either feeds one of these or should be deleted. (CLAUDE.md §3.4.)

---

## 3. Measurement primitives status

→ Full catalog: `research/MEASUREMENT_PRIMITIVES.md`. Gate semantics from pre-pivot atlas session (file deleted in anti-entropy).

**Legend (four-tier per §2.5).** 🟢¹ Level-1 universal (Gate 2 passed) · 🟢² Level-2 family-local (Gate 1 on ≥5 classes + family constants) · 🟡 coordinate (Gate 1 on ≥3 classes, portability only, no universality claim) · ⚪ diagnostic (Level-0; class-local or fails semantic comparability) · ⚫ untested.

| Primitive | Status | Classes tested | Last used | Notes |
|---|---|---|---|---|
| Intrinsic dimension (TwoNN + MLE estimator pair) | ⚪ | 4 (class 1,3,4,6) | genome_007 | **DEMOTED to diagnostic.** TwoNN and MLE-k10 fail G1.3 at δ=0.10 on all systems at n=2000 (max_stat 2.4-4.4 vs margin 1.8-2.3). Negative-control test (genome_004) showed only 6-13% trained-vs-untrained gap — measures architecture more than learned geometry. |
| Participation ratio centered | ⚪ | 4 (class 1,3,4,6) | genome_007 | Fails G1.3 at δ=0.10 on all systems (max_stat 3.9-5.9 vs margin 0.8-2.8). Neg-control passes (92% gap) but stimulus-resample too noisy. |
| Participation ratio uncentered | ⚪ | 5 (class 1,2,3,4,6) | genome_009 | **DEMOTED to DC-artifact diagnostic 2026-04-21.** Passes G1.3 5/5 classes at δ=0.10 but values are all ≈1.0 (range 1.01-1.60); PR_centered is 13-39× larger across the same systems. Its 5/5 pass is documenting that all trained networks have a dominant DC activation component (top eigenvector captures ≥95% of uncentered variance), not that they share substantive learned geometry. |
| kNN-10 clustering coefficient | **🟡→🟢¹ text-scope** pending DINOv2+biology | 5 Batch-1 + 3 Batch-2 | genome_013 | **G2.4 CAUSAL PROVISIONAL PASS 2026-04-21.** Gate-1: 5/5 Batch-1 + 7/8 Batch-2 (BERT/MiniLM/CLIP) pass G1.3 at δ=0.10 across 5 training objectives. G1.5 Q8-stability δ=0.05 on 4 text. G2.4 full-grid 3/3 text PASS (Qwen3/RWKV/DeepSeek): topk-ablation effect 7.8-443% at λ=1.0, monotonic, 20-60× specific vs random-10d and PCA-10. Random-Gaussian baseline: trained 0.28-0.36 vs random 0.05-0.08 (4-7× ratio) → NOT an artifact. Gate-2 derivation LOCKED 62338b8. **Remaining for clean green-1:** DINOv2 causal test (needs linear-probe loss), G2.3 hierarchical fit (needs extended k-sweep), G2.5 biology (Allen Neuropixels, STAGED). |
| kNN-5 clustering coefficient | ⚪ | 4 (class 1,3,4,6) | genome_007 | **DEMOTED.** Fails G1.3 at δ=0.10 on 3/4 systems at n=2000 (too noisy at k=5). k=10 is the stable neighborhood size. |
| Koopman spectrum (DMD) | ⚫ | — | — | **NEW** H11 (conf medium). Strongest cross-class candidate by literature (transformer+SSM+diffusion 2025-2026). Deferred to Batch 2 per parsimony. |
| Persistent homology | ⚫ | — | — | Deferred to Batch 2; needs subsampling-stability control. |
| Ricci curvature (Ollivier) | ⚫ | — | — | Deferred to Batch 2. H3a. Null result on SSM/diffusion in 2025-2026 lit — new science opportunity. |
| Fisher info matrix trace | ⚫ | — | — | Harder on JEPAs (no probability output). |
| Lyapunov spectrum | ⚫ | — | — | Needs layer-wise adaptation for feedforward. |
| CKA (linear / RBF) | ⚪ | — | — | **DEMOTED** from 🟡 to ⚪ per Round 1 (scale-confound Feb-2026 Aristotelian-View; PC-dominance). Diagnostic only. |
| NNGS (kNN Jaccard between two embeddings) | ⚪ | — | — | **NEW** cross-system diagnostic (not a per-system coordinate). Level-0. |
| Procrustes / CCA / SVCCA | ⚫ | — | — | Prior: fails cross-arch (p=0.82); informative bound. |
| RSA (cross-system RDM) | ⚫ | — | — | Canonical bridge to biology — stimulus-bank-conditional. |
| SAE (feature decomposition) | ⚫ | — | — | High-risk, family-local per dark-matter lit; see Mystery 7. Phase-N not Phase-1. |
| PCA / SVD spectral | 🟡 | LLM, vision | — | Spectral-slope retired per Codex R2 (fragile, redundant with PR). PCA/SVD still a generic diagnostic. |
| Activation ablation | ⚫ | — | — | Minimum class-agnostic causal primitive; Gate-2 G2.4 requirement. |
| Path / activation patching | ⚪ | transformer | — | **DEMOTED** per Round 1 — transformer-native; Level-0 until class-agnostic extension derived. |
| CAA (direction steering) | ⚪ | transformer-only | — | Demoted — fails on hybrids/SSMs; see Mystery 4. |
| Linear probes | ⚪ | — | — | Diagnostic; cannot imply usability (see Mystery 2). |
| MDL probes | ⚫ | — | — | Pure information-theoretic — works anywhere. |
| Non-linear / MLP probes | ⚫ | — | — | Pair with linear probes for manifold hypothesis test. |
| Task-conditional compression | ⚫ | — | — | Carries over the 99.7%→3D coherent-divergence finding. |
| Successive-refinement D(R) curves | ⚫ | — | — | Direct CTI lineage — strong Level-1 candidate. |
| Diffusion noise-step representations | ⚫ | — | — | No analogue in transformer "layer depth". |
| JEPA predictor/encoder alignment | ⚫ | — | — | Dual-network geometry. |
| World-model latent rollout | ⚫ | — | — | Dynamics-aware, not static. |

---

## 4. System bestiary status

→ Full bestiary: `research/SYSTEM_BESTIARY.md`. Model IDs always pulled from `../../models/registry.py`.

| Class | In atlas? | Phase-1 anchor | Status |
|---|---|---|---|
| 1 Autoregressive LLM | **measured genome_001..007** | `Qwen/Qwen3-0.6B` | ACTIVE in Batch 1 at n∈{5, 500, 2000} |
| 2 Reasoning | 0/N | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Not yet loaded |
| 3 SSM / linear-attention | **measured genome_003..007** | ~~`state-spaces/mamba2-370m-hf`~~ → `RWKV/rwkv-4-169m-pile` | ACTIVE. Mamba2 Windows-blocked (mamba-ssm kernel build fail); RWKV-4 substituted as linear-attention recurrent. |
| 4 Hybrid | **2/3 seeds at n=2000** | `tiiuae/Falcon-H1-0.5B-Instruct` | **UNBLOCKED 2026-04-21** via micro-batched extractor. Natural-fallback Mamba layers work for Falcon at n=2000 with batch_size=64 (was 93 GB OOM in single-batch). |
| 5 Diffusion | 0/N | `GSAI-ML/LLaDA-8B-Instruct` | Not yet loaded |
| 6 Vision encoder | 0/N | `facebook/dinov2-small` | Not in canonical registry yet — add Phase 1 |
| 7 JEPA | 0/N | `facebook/ijepa-vit-huge-14-448` | Not in canonical registry yet — add Phase 1 |
| 8 World model | 0/N | Dreamer-V3 small | Not in canonical registry — may need to build/port |
| 9 Controls | 0/N | Untrained Qwen3-0.6B + Allen V1 | Allen access pattern known from CTI |

---

## 5. Open mysteries — status board

→ Full descriptions: `research/OPEN_MYSTERIES.md`.

| # | Mystery | Hyp. falsified | Hyp. remaining | Status |
|---|---|---|---|---|
| 1 | Orthogonality coupling at 7B+ | 3 | phase-transition, global manifold, depth-capacity | ⚫ unresolved — 1 more falsified hyp → 🩹 scar |
| 2 | Reading/writing asymmetry | 0 | 3 (manifold, distributed, activation-vs-weight) | 🔥 high priority — unlocks Phase 5 |
| 3 | 2 tokens of noise fixes reasoning | 0 | 3 (attention-sink, phase-space, derivable-2) | Cross-class replication is the genome angle |
| 4 | CAA works on tuned transformers only | 0 | 3 (post-tune artifact, non-linear concepts, wrong-layer) | Touches every causal primitive design |
| 5 | Coherent divergence (99.7% content destruction, fluency preserved) | 0 | Cross-class replication pending | Strong Level-1 candidate if replicates |
| 6 | CLIP modality gap | 0 | "gap as Level-2 family constant" | Counter-example to naive universality |
| 7 | SAE feature universality across models | 0 | Feature-vocabulary shared or family-local | Determines if SAEs are coordinate or diagnostic |

---

## 6. Active experiments

→ Running or queued. Full log: `experiments/EXPERIMENTS.md`. Raw: `experiments/ledger.jsonl`.

| ID | Status | Purpose | Pre-reg |
|---|---|---|---|
| `genome_181b` | **PASS** | Long-horizon attenuation: +0.513 nats at 5000 steps | — |
| `genome_180b` | **FAIL** | Cross-tokenizer forecast (27/27 cells, geometry tokenizer-specific) | `research/prereg/genome_180b_cross_tokenizer_2026-04-29.md` |
| `genome_182` | **RUNNING** | Blinded Training Triage Arena (48 cells stage 1, 5 models: A/B co-primary + C pure-geometry + D pure-telemetry + E Shesha) | `research/prereg/genome_182_triage_arena_2026-04-29.md` |
| `genome_183` | DRAFT | Corpus-derived init (8 arms × 3 seeds, gated on g182) | `research/prereg/genome_183_corpus_derived_init_2026-04-29.md` |
| `genome_180` | WEAK PASS | Forecast/diagnostic (24 geometry features, MSE -61.6%) | — |
| `genome_181a` | COMPLETED | Tokenizer isolation (embed anchor = tokenizer prior) | — |

Earlier experiments (g001-g177) documented in `experiments/EXPERIMENTS.md` and `experiments/ledger.jsonl`.

---

## 7. Findings (by universality level)

→ Full derivations and pre-registrations under `research/`.

### Level 1 (functional-form universal)
*(none)*

### Level 2 (family constants)
*(none)*

### Level 3 (task/data intercepts)
*(none)*

### Phase-2 atlas observations (null level — not yet claimed)
*(none)*

---

## 8. Cross-project connections

→ Full context: `_meta/insights/` (sibling path outside this moonshot).

| Other project / moonshot | What it gives us | What we give it |
|---|---|---|
| `moonshot-cti-universal-law` | 3-tier framework; EVT derivation template; biology validation pattern; `cti_allen_*` access scripts | Generalizes CTI's law of representation *quality* into laws of representation *structure* |
| `moonshot-fractal-embeddings` | Hierarchical scale-separated embeddings as a prior on content-subspace structure | Test of whether fractal structure is Level-2 family-local or Level-1 universal |
| `moonshot-sutra` | Byte-level small model — control for "tokenizer-induced geometry" vs. fundamental | Identifies which atlas coordinates are tokenizer artifacts |
| `moonshot-fractal-mind` | Adaptive-depth reasoning — test of "depth-as-geometric-invariant" | Framework for interpreting its adaptive-depth measurements geometrically |
| `moonshot-j-self-construction` | Untrained networks that solve XOR/parity — pure "architecture-vs-weights" control | Rigorous characterization of what their emergent networks represent |
| `LLM exploration/` | Intrinsic-dim and orthogonality-coupling prior data | Inherits Mystery 1; provides cross-class replication |
| `llm-rosetta-stone/` | CAA/steering prior work; cross-arch linear-alignment failure (p=0.82) | Inherits Mysteries 2 and 4; reframes steering as a primitive-agnosticism question |
| `knowledge-surgeon/` | LoRA-based weight-space editing (100% on geography) | Activation-vs-weight asymmetry lens for Mystery 2 |
| `Latent-Space-Reasoning/` | 2-token noise → +19.6pp arithmetic | Inherits Mystery 3 for cross-class replication |
| `llm-platonic-geometry/` | Positive Ricci curvature in LLM embedding spaces, Lyapunov ≈ 0 | Primary prior on atlas-shape expectations |

---

## 9. Decisions log

Architectural / methodological decisions. One line each. Irreversible choices get highlighted.

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-20 | **Axiom-first framing, atlas as instrument.** | CTI template; avoids pre-specified-curve-fit trap. README §The Axiom. |
| 2026-04-20 | **Inherit CTI's 3-tier universality framework.** | Prevents "universality collapsed because one constant varied" failure. `research/UNIVERSALITY_LEVELS.md`. |
| 2026-04-20 | **Architecture-agnosticism gate: ≥3 classes before a primitive becomes a coordinate.** | Keeps LLM-specific tools from masquerading as atlas coordinates. CLAUDE.md §4.3. |
| 2026-04-20 | **Biological validation mandatory for every Level-1 claim.** | Separates "fact about trained NNs" from "fact about learning systems." CLAUDE.md §4.5. |
| 2026-04-20 | **Anti-entropy is a Tier-1 rule, not a cleanup chore.** | Atlas dies under file-bloat within 6 months without this. CLAUDE.md §3. |
| 2026-04-20 | **Repo-wide model registry at `Projects/models/` is canonical; no local model lists.** | One source of truth across every moonshot and research project. |
| 2026-04-20 | **WIKI.md is read first, updated every commit touching state.** | Agents bootstrap fast; nothing goes stale. |
| 2026-04-20 | **9 system classes + Phase-1 minimum viable bestiary defined.** | Concrete, constrained first atlas iteration. `research/SYSTEM_BESTIARY.md` §Phase 1. |
| 2026-04-20 | **`COMPUTE.md` is the binding hardware envelope.** | RTX 5090 Laptop (24 GB VRAM, ≤22 GB usable), Ultra 9 285HX, 64 GB RAM. Every Codex prompt, every prereg must comply with §9 checklist. Out-of-envelope proposals are rejected at design gate. No cloud available. |
| 2026-04-20 | **`moonshot-llm-genome` becomes its own GitHub repo** (pattern matches `moonshot-sutra`). Process docs (`CLAUDE.md`, `WIKI.md`, codex review artifacts) excluded from public push. | Separates mission-public content from agent-process scaffolding. |
| 2026-04-26 | **Architecture-prior thread treated as breakthrough-axis seed** (g138-g151 → C10-C13). Pursue derivation, not phenomenology, per CLAUDE.md §0.1. | A workshop paper saying "minimal-3L wins by 0.8pp" is publishable but not a breakthrough. The derivation is what's distinctive. |
| 2026-04-26 | **Prefix-Information Transport Principle is the committed first-principles route** (research/derivations/prefix_information_transport.md). | Codex audit: of 5 candidates evaluated, only Candidate 5 is mechanistically right-shaped + brutally falsifiable + product-conflict for big labs. g156 is its locked killer test. |
| 2026-04-26 | **C3-TEI/kJ (HellaSwag+PIQA+Winogrande items per kJ wall power) is the committed edge-benchmark headline metric**, not tokens/sec/joule. | Codex Competitive-Analyst: tokens/sec/joule is gameable across tokenizers; an integrity audit would tear it apart. Wall-power required, GPU-only never the headline. g155 prereg LOCKED. |

---

## 10. Anti-entropy log

Every deletion, merge, and rename. Demonstrates the repo is getting simpler, not just bigger.

| Date | Action | Target | Reason |
|---|---|---|---|
| 2026-04-20 | Migrated | Old `LLM Genome Project/` → deleted in full | Superseded by axiom-first scope; content had diverged from the vision |
| 2026-04-20 | Scaffolded | `moonshot-llm-genome/` | New home under AI Moonshots umbrella |
| 2026-04-26 | Deleted | `code/genome_130_trajectory_scaling_law.py` + result JSON + run.log | Pythia-2.8b checkpoint aliasing meant all step* branches mapped to same file; experiment was skipped, never had a ledger entry. |
| 2026-04-26 | Deleted | `code/genome_124_kd_logit_distillation.py` (untracked, never committed) | Dead duplicate of the now-canonical g154 distillation pipeline. |
| 2026-04-29 | Deleted | 54 code files (genome_100–150) + 44 result JSONs (genome_110–150) | Pre-pivot experiments with zero imports from active code (g165+). |
| 2026-04-29 | Deleted | 15 code files (genome_087–099 + 161–162) | Pre-pivot era, zero WIKI refs (161/162) or pre-transfer-mechanism invariant exploration (087–099). |
| 2026-04-29 | Deleted | 10 one-shot scripts (integrate_g*, make_*, assemble_paper, analysis_*) | Run-once verdict integrators; results already committed. |
| 2026-04-29 | Deleted | 89 dead utility/probe/figure scripts (non-numbered genome_*.py) | Self-referencing cluster with zero active (g165+) imports except genome_primitives. Kept: genome_primitives, stimulus_banks, prereg_validator. Repo: 189→21 code files. |
| 2026-04-29 | Deleted | 9 orphaned pre-pivot scripts (g151-g159, stimulus_banks) | Import graph analysis: none imported by active chain (g165→g180→g182). stimulus_banks only imported by deleted files. Results preserved in git. Repo: 21→12 code files. |

---

## 11. Retired primitives, archived mysteries, dead ends

Kept for institutional memory. Do not resurrect without reading the retirement reason first.

*(none yet — project just scaffolded)*

---

## 12. Next actions

*(Updated 2026-04-29 cycle 82)*

**Active:**
1. **g180b cross-tokenizer forecast** — COMPLETE, FAIL. 27/27 cells. `results/genome_180b_cross_tokenizer.json`.
2. **g182 Blinded Training Triage Arena** — prereg LOCKED, implementation READY (`code/genome_182_triage_arena.py`). Smoke test next, then full 72-cell run. GPU free.
3. **g183 corpus-derived init** — DRAFT prereg pre-staged (`research/prereg/genome_183_corpus_derived_init_2026-04-29.md`). 24 cells, 8 arms. Gated on g182 Codex design gate.

**Completed this session:**
- **g181b long-horizon attenuation** — PASS. +0.513 nats at 5000 steps (3-seed mean). C23 locked. A8 resolved.

**Queue (post-g182):**
- g155 production distill + C3-TEI/kJ — HARDWARE-BLOCKED on wall-power meter
- Phase 2 SSM/hybrid for g182 — if phase 1 PASS

**Historical (2026-04-22 era: genome_068–g087, GenomeGuard, candidate-8 bridge, grafting series):**
Detailed in `experiments/EXPERIMENTS.md` and `experiments/ledger.jsonl`. Code deleted in cycle 77 anti-entropy pass.

## §13–§15 Historical Experiment Details (ARCHIVED)

> Experiments g110–g156 (mental models, critical subspace, surgery series, architecture-prior chain, prefix-destruction) are fully documented in `experiments/EXPERIMENTS.md` and `experiments/ledger.jsonl`. Code files were deleted in cycle 77 anti-entropy pass (189→21 files). Key verdicts preserved in git history.
>
> **Summary:** g110-g113 mental models (3 KILL, 1 PARTIAL). g114-g118 critical subspace (causal, cross-arch, but surgery KILL). g119-g125 surgery series (holism barrier confirmed). g126-g137 invariant population + grafting (12 KILL/NULL). g138-g156 architecture-prior chain (g156 PASS_TRANSPORT, g158c PASS_canonical).


*End of WIKI. If anything here surprised you, fix the docs — not the wiki — and then patch the wiki pointer.*
