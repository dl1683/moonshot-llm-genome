# Explorations harvest — mining the standing idea directories for dissection questions

**Date:** 2026-09-26. **Author:** EXPLORER angle (frontier review pass).
**Sources read (CPU-only):**
- (a) `C:\Users\devan\OneDrive\Desktop\Projects\Market Reports\Open Exploration\` — 14 files across 10 topic folders (Memory Architecture, Sleep and Downtime, Attention Everywhere, Developmental Computation, Death and Renewal, Aging and Information Decay, Cognitive Science, Error Correction Across Scales, Small Models and the Verifier Wall, Grand Unified Patterns) + folder READMEs.
- (b) `C:\Users\devan\OneDrive\Desktop\Projects\_meta\` — INDEX.md, README.md, inquiry/THESIS.md (v14), insights/ (open-mysteries, cross-domain-mechanisms, connections, second-order, implicit-bets, blind-spots, the 2026-08-30 LSR structured negative), experiments/proposed.md, harvest/moonshots.md, projects/llm-genome.md.

**Path shorthand below:** `OE/…` = Open Exploration root; `_meta/…` = the `_meta` root.
**Live-hypothesis anchors:** T037 (coordinate-keyed memory / canalization / read policy), T038–T039 (relay-not-row; absolute ~6-token spike horizon), T040–T041 (selection moves nothing; universal organ-reliance template), T042–T043 (row 129 address vs row 0 scaffolding), T044 (K/V dissociation, value-side readers), T045 (self-generated cache junk), T046 (pre-graft crossmatch instrument, AUC 0.919), T047 (critic audit), T048 (run-specific trajectory anchor; free-run-honesty principle, 3rd appearance).

---

## 1. MINED-IDEAS TABLE (16 ideas)

| # | Source file | Concept (one line) | Dissection question (one line, our scale) |
|---|---|---|---|
| 1 | OE/Death and Renewal/forgetting_as_intelligence.md | Retrieval-induced forgetting (RIF): retrieving one item actively suppresses its category competitors | Does *eliciting* an installed fact (prompt-only, no training) suppress its coordinate neighbors — other names at adjacent wpe rows / same vocab cluster — below usage-matched controls? |
| 2 | same | Anderson–Schooler: forgetting curve mirrors environmental need-probability (recency/frequency) | Does the utility onset a* (exposure axis 3→21→86, r=0.82) *track* the corpus recency structure — do synthetic corpora with stretched dependency length stretch a* at matched steps? |
| 3 | same | Active forgetting has dedicated machinery (DAMB ≠ learning receptor) | Is there a dissociable "forgetting direction" in weight space: does descent on a scramble objective erase via the same subspace that installs used, or an orthogonal one? |
| 4 | OE/Developmental Computation/what_development_teaches_ai.md | Rozum/Waddington "coherence gap": canalized systems robust to large (basin-crossing) but sensitive to small (within-valley) perturbations | Does mid-generation cache corruption show *non-monotone* dose-response — do small doses drift the run permanently while large doses re-enter a basin cleanly? |
| 5 | same | Degeneracy ≠ redundancy: different structures, same function, uncorrelated failure modes | After e033-style constrained retraining (anatomy reorganization at parity), does the relocated function fail on *different* inputs than the original circuit (degenerate) or the same ones (redundant)? |
| 6 | same | Planarian regeneration: repair targets *function* (target morphology), not original coordinates | After erase→re-learn, does the new circuit reconverge on the same deep-relay object (T038's "third thing") though the address moved — regeneration converges on the feature, not the coordinate? |
| 7 | OE/Memory Architecture/what_would_real_ai_memory_look_like.md | Six-component memory: fast binding, consolidation, content-addressing, salience tagging, forgetting policy, schema formation | Is the *price* of content-addressing (T037) paid in exposure-mass or in surprise — does error-weighted (surprise-gated) install exposure buy stronger binding than uniform exposure at matched gradient mass? |
| 8 | same | "Attention is a read-only Modern Hopfield; writable attention = fast binding" | Can we *write* to the Hopfield directly — transplant donor KV entries (position-matched) into a host run and get donor-flavored continuations, gated by the e062 crossmatch cosine? |
| 9 | OE/Memory Architecture/hippocampal_replay_and_ai.md | Prioritized (surprise-biased) + interleaved + bidirectional replay as consolidation | Does interleaved replay (install windows mixed with original corpus) during re-learn beat massed install exposure on surgical resistance — the spacing effect at 1M params? |
| 10 | same | Backward replay serves credit assignment | Is install exposure direction-sensitive — does reverse-order presentation of install windows bind the same address with the same expression? |
| 11 | OE/Cognitive Science/dual_process_theory_and_llms.md | Gigerenzer "take-the-best": ordered cues, stop at first discriminator (satisficing vs integration) | Is the read policy satisficing or integrating — do the first ~6 positions causally *gate* (their lesion flips argmax) while the tail only *modulates* (lesion shifts confidence), a two-regime signature? |
| 12 | OE/Cognitive Science/memory_systems.md | Method of loci: install content at *spatial coordinates* to exploit the oldest binding system | Capacity curve of the palace: how many facts can successive wpe rows carry before interference — does fact k degrade fact k−1's expression monotonically or at a threshold? |
| 13 | OE/Attention Everywhere/meta_attention.md | Monitoring–control loop: self-report without control is commentary | Does the net have *any* closed loop over its read policy — does intervening on late-layer state change *where* attention reads at the next step (control), or nothing (open-loop)? |
| 14 | OE/Aging/model_degradation_in_ai.md + agent-drift lit | Drift mitigation taxonomy: consolidation, drift-aware routing, adaptive behavioral anchoring | Dose-response of anchoring: what fraction of prompt-copy / self-echo content keeps a free run on-manifold (e080's prompt-copy arm at +0.069 is one point on the curve)? |
| 15 | _meta/inquiry/THESIS.md (v14) | ρ = tanh(1/2) universality; temperature scaling ρ(τ)=tanh(1/(2τ)); K_eff = 3.164 percolation | Eval-only on cached activations (e055_traj_cache.pt): does layer-to-layer state correlation sit near 0.462, and does softmax-temperature scaling track tanh(1/(2τ))? |
| 16 | _meta/projects/llm-genome.md + insights/connections.md | Candidate-8 spectral identity c ≈ eff_rank/d_rd (7/8 nets); kNN-10 portable coordinate; Forecast/Diagnostic pivot ("early geometry predicts training health") | Do our 0.84M–10M nets satisfy c ≈ eff_rank/d_rd, and does *mismatch* of that quantity between host and donor predict graft damage on top of the stream-cosine crossmatch? |

Also-mined but table-overflow (kept for QUEUE trawling): stress-induced mutagenesis + error fencing (OE/Error Correction/when_errors_are_features.md) → does elevated LR during install increase drift *selectively at the address row*; quasispecies error threshold → the graft-size-vs-lineage-survival curve (ties to T024/T027); "route ladder" cheapest-reliable-route discipline (OE/Small Models FIELD_NOTE.md) → our edit law prices the ladder mechanistically (row-copy = the "five-line rule" rung); one-shot vs gradient gap + "refinement is intelligence" (OE/Grand Unified Patterns/what_we_still_dont_understand.md) → does iterated self-refinement (feeding the run its own best-of-k continuations) extend coherent generation length; attention-sink elimination by gated attention (flagged in _meta/insights/cross-domain-mechanisms.md as a 12–18-month obsolescence window) → our K/V dissociation (T044) is already a sink-detector: is row-0 load K-side (sink-like) or V-side (value read) in the *untruncated* window?

---

## 2. THE TOP 5 (fully worked proposals)

### P-A — RIF at our scale: does reading write? (table #1)

**Hypothesis.** A tiny char-LM has retrieval-induced forgetting: eliciting one installed fact by prompt exposure (no gradient step) measurably suppresses the expression probability of its coordinate neighbors (adjacent-row / adjacent-window installs), beyond any usage-matched sham control. If true, the read policy has *side effects* — reads write — the first dynamical (not structural) plasticity the lab would have found.

**Cheapest discriminating experiment (eval-only, minutes, existing artifacts).** Use the two-install net (e048_dose.pt; T047's e078 already re-runs its battery — add this readout to the same pass). Arms: (i) 64 elicitation prompts for fact A (the address row 129/130 window), (ii) 64 sham prompts matched in length/charset, (iii) no-op. Readout: p(fact B) at B's canonical probe (its own knife-edge geometry) after each arm, plus KL-lens surgicality at B's address. No training; B=32 resamples of elicitation sets for CIs.

**Registered prediction.** H-RIF: p(B) drops by ≥0.05 absolute after A-elicitation with the sham arm flat (CI-separated); H-null (reads are pure): both arms flat. Secondary: the drop, if any, concentrates on B's *address row* perturbation sensitivity (KL 0.2–0.5 surgical band), not distribution-wide.

**Why-now.** T037 #5 names the read policy as the never-edited component; T043 gives us the cleanest address object (row 129) and the KL lens to see surgical vs collateral damage; T047's e078 is already touching this net this week. RIF would also reframe T018's scar (erasure burns the address) as possibly *read-induced*, not gradient-induced.

### P-B — The coherence gap: non-monotone dose-response around the trajectory anchor (table #4)

**Hypothesis.** The free-run attractor of T048 is canalized in Waddington's sense: robust to large perturbations (basin-crossing kicks get corrected back onto *a* coherent trajectory) but sensitive to small ones (within-valley drift accumulates). Concrete signature: per-nat damage and non-return rate are *non-monotone* in corruption dose — maximal at small ε, decreasing at moderate ε.

**Cheapest discriminating experiment (eval-only, ≤10 min, existing rig).** The e080 rig (same nets, B=8) with dose as the new axis: at matched prune/corrupt events (age>96), apply Gaussian cache perturbations at ε ∈ {0.05, 0.2, 0.5, 1.0} × entry-norm plus the existing V-zero arm. Readouts: clean-judge CE at +64 tokens; *return-to-trajectory* rate = 1 − normalized distance between the corrupted continuation and the uncorrupted continuation (a new, cheap statistic); junk census.

**Registered prediction.** H-coherence-gap: return-rate is U-shaped or per-nat damage peaks at ε=0.05 (small doses drift; large doses land in a coherent basin — possibly the *wrong* one, visible as clean-judge-off-manifold-but-low-entropy). H-flat: damage scales linearly with ε — the anchor is a simple additive buffer, not a landscape. The two hypotheses make opposite signed predictions at the smallest dose; one pass discriminates.

**Why-now.** T048 just closed e080 with "the anchor is run-specific trajectory content" but only replacement-style arms; P3 pivoted from replacement to *description of the anchor* — the dose-response shape IS the first structural description (basin vs buffer). T045 (junk is self-generated = drift) predicts the small-dose sensitivity arm; this is its discriminating test.

### P-C — Surprise-gated writes: pricing the purchase of content-addressing (table #7)

**Hypothesis.** The currency that buys content-addressing (T037 #1: "purchased by exposure") is *surprise*, not raw exposure mass: install exposure sampled proportional to the net's own error at each window binds faster and more surgically than uniform exposure at matched total gradient norm. (Biology's emotional/salience tagging; RL's prioritized replay.)

**Cheapest discriminating experiment (≤180 s training, 0.84M net).** Two install arms, matched total steps and matched Σ‖grad‖: (i) uniform window sampling (the standard install), (ii) error-weighted sampling (first pass computes per-window loss; sampling ∝ loss; refresh every ~50 steps). Readouts: p(Z) learning curve; knife-edge behavior at the new address; surgical resistance via the T018 erase/re-learn protocol on the winning arm; KL-lens surgicality of the address row.

**Registered prediction.** H-surprise: error-weighted reaches p(Z)≥0.5 in ≤60% of the steps AND its address perturbation lands in the surgical KL band (0.2–0.5) while uniform sits collateral-heavy; H-mass: both arms indistinguishable at matched gradient mass — the purchase is priced in raw exposure only, and "salience tagging" has no analogue at this scale (itself a clean negative: the write interface is dumber than biology's).

**Why-now.** Law 7 (expression needs free-shaped exposure) leaves the *shape* unexplored; T037's purchase metaphor is unpriced; T047 demands cheap n=2 replication work (e078) and this slots after it. If H-surprise fires, installs get cheaper and P1's address atlas gets a better instrument.

### P-D — Cross-net KV transplant: writing to the read-only Hopfield (table #8/#16)

**Hypothesis.** Attention is a read-only associative memory over KV pairs; if we write to it directly — splicing a donor net's cache entries (K and V at position-matched coordinates) into a host free-run — the host's continuation drifts donor-ward, and the drift magnitude is predicted by the e062 crossmatch cosine (fires above the 0.4459 threshold). This tests whether the T046 instrument generalizes from weight-level grafts to *state-level* grafts, and whether T048's non-substitutability (within-net) is a content fact or a geometry fact.

**Cheapest discriminating experiment (eval-only, minutes).** Host free-run (B=8); at age>96 replace 32 cache entries with the donor's entries from the donor's own run at the *same ages* (position-matched). Arms over ≥6 donors spanning the crossmatch cosine range (0.2–0.8); K-only vs V-only vs K+V dissociation (the T044 instrument). Readouts: clean-judge CE, continuation divergence, junk census.

**Registered prediction.** H-geometry: drift/damage is monotone in crossmatch cosine with the decision rule transferring (out-of-sample AUC ≥ 0.8); K+V ≫ either alone above threshold, ≈ nothing below. H-content (T048 generalized): even geometrically-matched donor content is inert — the anchor is *run*-specific, not *net*-specific — which would sharpen T048 from "run-specific" to "trajectory-specific" and close the P2 question of whether any state-level graft channel exists.

**Why-now.** T046 gave P2-immunology a screening instrument and explicitly registered "if tolerance raises D-compatibility without raising cosine, instrument and mechanism diverge"; P-D is the state-level version of that divergence test. T044's K/V dissociation instrument is sitting ready. This is the only proposal that touches P1 (address), P2 (crossmatch), and P3 (anchor) at once.

### P-E — The consolidation cycle law: monotone closure vs replay-reversal (table #9; runs T037's registered prediction)

**Hypothesis.** T037's registered canalization prediction is still un-run: a third erase/re-learn cycle is slower and more surgical-proof than the second (monotone closure, never oscillation). The archive's counter-mechanism is consolidation-as-phase: interleaved replay (install windows mixed with original-corpus text during re-learn) should flatten or reverse the closure trend — the spacing/interleaving effect at 1M params.

**Cheapest discriminating experiment (≤180 s per arm, existing dose-net lineage).** Three erase→re-learn cycles (Z-erase at the address, re-learn to criterion), measuring per cycle: steps-to-criterion, address-row surgicality (KL lens), and route (which layer carries the re-learned read). Arm B: identical but re-learn exposure interleaves 50% original-corpus batches. Total ≤6 short runs.

**Registered prediction.** H-canalization: steps-to-criterion and surgical resistance increase monotonically across cycles 1→2→3 in arm A. H-consolidation: arm B's cycle-3 is ≤ arm A's cycle-2 (interleaving buys back plasticity). If BOTH fail (no monotone trend at all), T037's canalization construct takes its second haircut (after the live-window correction) and the scar story (T018/T036) is re-scoped to first-erase only.

**Why-now.** It discharges an *already-registered falsifiable prediction* (T037 #2) — the cheapest kind of experiment the lab can run, morally and computationally; T036 replicated the scar at seed 43, so the n=2 base exists; and the archive independently nominates interleaved replay as biology's answer, giving the two arms opposite theoretical sponsors (canalization vs consolidation).

---

## 3. DIRECT SYNERGIES (what our instruments can answer for *them*)

1. **"State over output" needs a causal micro-foundation — we have it.** `_meta/insights/open-mysteries.md` tracks "state over output" as an emerging universal principle (ant-irys, MapU, legal-rlm, iqidis, Open Exploration, _meta itself — 6 instantiations, all behavioral/architectural). T048/e075/e080 is the same claim with causal teeth at model scale: cache-state interventions change outcomes in ways output metrics (single teacher-forced CE) do not predict. We can write the 7th instantiation *and* the mechanism.
2. **The self-consistent-attractor diagnosis is our clean-judge dissociation.** `_meta/harvest/moonshots.md` post-mortems a 122-version autonomous loop that "fell off criticality into a self-consistent attractor — internally coherent, externally unmoored (0.98 internal confidence, 102 unresolved P2 errors)," grouped with SynFlow bridge-severing and LSR's attention sink. Our free-run-honesty principle (self-score fine / clean-judge off-manifold; 3rd independent appearance) is the *measurable, intervenable* version of that failure — and P-B/P-D are exactly the instruments that could test mitigations for it.
3. **The Forecast/Diagnostic pivot cashed out — tell _meta.** `_meta/projects/llm-genome.md` open question: "Can the Forecast/Diagnostic pivot (early geometry predicts training health) actually be cashed out predictively?" e062/T046 answers yes at the graft level (pre-graft stream-cosine, AUC 0.919, transfer hint at 2.7M). The card is also stale (pre-tombstone, cites the falsified-transfer era as "live"); a refresh would point at our README's 10 laws.
4. **LSR's "why ~2 tokens, not 1 or 4" and the non-monotone dose-response.** `_meta/open-mysteries.md` Mystery 3 needs a system where dose-response can be measured *and* the interior observed. Our cache-corruption dose ladder (P-B) and the LSR-structured-negative lesson ("readout instrument validity first" — gate the instrument on the unmodified base) is a discipline we already practice (protocol-identity gates in T044); the non-monotone dose question is directly portable.
5. **The gated-attention obsolescence window (NeurIPS 2025 best paper eliminates sinks).** `_meta/insights/cross-domain-mechanisms.md` flags sink-dependent mechanisms going stale. Our K/V dissociation (T044: V≫K at window start under truncation — the *opposite* of the sink signature K≫V) is a per-net sink-detector at tiny scale; running it on the *untruncated* window would tell us whether our net family even has sinks — a one-line prerequisite for extrapolating any of our row-0/anchor results into the gated-attention era.
6. **Agent-drift mitigation taxonomy wants its quantitative curve.** `OE/Aging/model_degradation_in_ai.md` (agent drift, arXiv 2601.04170) lists "adaptive behavioral anchoring" as a proposed mitigation with no dose-response data. e080's prompt-copy arm (+0.069 nats, ~3/4 recovery) is the first point on that curve; P-B's dose axis completes it.
7. **Anderson–Schooler environmental mirroring vs our exposure axis.** The archive's "forgetting tracks need-probability" (table #2) and our "a* grows with training exposure" (r=0.82) are the same law candidate from two directions — theirs environmental-statistics, ours training-dynamics. One synthetic-corpus run decides whether a* is fitted to the corpus or to the exposure count.
8. **The lab fills two named portfolio blind spots.** `_meta/insights/implicit-bets.md` and `blind-spots.md` name "no circuit-level mechanistic interpretability" and "no RLHF/preference" as the portfolio's conspicuous gaps (blind-spot #9 explicitly asks for a circuit-level pass). We are that pass; worth stating in any cross-portfolio writeup, and the falsification-trilogy essay (proposed.md #13) gains a 4th case study (our tombstone history).
9. **Metric discipline: our generous/strict pair is teacher-forced vs free-run.** The archive's universal mechanism #1 (local-vs-integration metric split, named 17 ways) is instantiated in our methodology as the T019/T048 split: teacher-forced read = generous metric, free-run expression = strict metric. Framing our laws in their vocabulary makes both sides sharper — and their rule "strict number first" is already our paper's 5.5 boundary sentence.

---

## Bottom line

16 table-grade ideas (+6 overflow), 5 worked proposals. Ranking for QUEUE trawling: **P-A** (cheapest, rides e078), **P-B** (P3's registered next step, new statistic), **P-E** (discharges T037's standing prediction). P-D is the highest-information but needs the crossmatch grid; P-C is the only one requiring fresh training (~180 s).
