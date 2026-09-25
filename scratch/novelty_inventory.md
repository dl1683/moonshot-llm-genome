# Novelty Inventory — STRATEGIST, 2026-09-25 (CPU-only)

Inputs: DAY_ONE/TWO reports, THINKING.md T015-T026, scratch/frontier_research_20260925.md.
Grades: FRONTIER-NOVEL (no counterpart found in scan) / PARTIALLY-KNOWN (adjacent
literature exists, our measurement adds a new axis) / REPLICATION-OF-KNOWN (the field
already established the direction; our value is confirmation/scoping).

## 1. The ten strongest results, graded

| # | Result | Novelty vs 2025-26 field | Evidence today | Single paper-grade upgrade |
|---|--------|--------------------------|----------------|----------------------------|
| 1 | **Four-faculty edit law** (address/ability/expression/history; T015/T018/T019) | **FRONTIER-NOVEL** — scan §2: "nobody separates address/ability/expression/history as distinct failure modes"; Guo ICML-25 + RMU-obfuscation thread confirm only that first-order fails | Expression zero confirmed across dose/temp/seeding arms; install cheapness 7 cells; but scar clause **n=1 (R7-flagged)**, install cells single-net-family, dose legs instrument-suspect (bit-identical) | **e055**: transplant teacher-forced residual at the divergence token into free-run — flips behavior → causal locus of suppression; localizes the faculty |
| 2 | **Expression-gap decomposition** (geometry binding p→0.089/1.7e-6, 6× collapse on 10-char deletion; sub-argmax prior) | **FRONTIER-NOVEL** — scan §1e: zero interventional studies; Orgad ICLR-25 is probe-level (with reproduction caveats) | Single net family; battery-overstat 3.4×; open confound: content-vs-position binding not discriminated | Same e055 transplant + a content-vs-position binding probe (swap vs shift the binding context) |
| 3 | **Basis frozenness under selection** (e040/T024: damage −5.6%, alignment +0.0015 vs 0.53 ceiling; T026: assay reads organ-load r=0.807, not basis-fit) | **FRONTIER-NOVEL** — scan §3 verdict: "the mechanistic-evolvability experiment the universality literature lacks"; publishable as-is per researcher | n=1 lineage, 1 donor, 2 generations; audit flags (G0 anchor failed; trickle is one member); confound now characterized not removed | **e050 directed-mutation** with T026's registered A-residualized / L2-weighted readout → reachability vs visibility verdict |
| 4 | **Init-anchoring ladder** (1.0/0.53/0.15/0.00; stream-facing violent, MLP-hidden free; e028/e029/e031/e041) | **PARTIALLY-KNOWN** — condensation, LMC, weak-universality are neighbors; interface-specific *causal transplant* ladder is the new measurement | Strong: replicated seed×regime matrix, causal grafts | Fold into #3 as one paper (statics + selection dynamics); add 1 big-model or crosscoder cross-check |
| 5 | **Address row-surgery** (384 params → 13% name acc, +0.0008 nats, class-exact, 500× selectivity, scale-invariant 0.84M/2.7M/10M) | **PARTIALLY-KNOWN** — Guo ICML-25 localizes circuits; parametric row surgery is the minority alternative; our class-exactness + 3-scale invariance quantification is new | **Best-replicated result in the lab** (5/5 nets row-half; 3-scale) | Head-to-head vs RMU/first-order on the same suite incl. relearning attack (uses our ~3× surgical-proof datum) |
| 6 | **First-order ascent cannot selectively forget** (r=1.08–1.8 vs bar; r≈6 retracted internally) | **REPLICATION-OF-KNOWN** — Guo + RMU-reverts line already established the direction | Excellent and honest (internal retraction of the chaotic false positive) | None needed — demote to motivation for #1/#5 |
| 7 | **Retrieval threshold curve** (refrain density flips far context −2.24 nats interference → +0.89 retrieval; graded, compartmentalized; head forms discretely in late-attention slot) | **PARTIALLY-KNOWN** — induction-head formation literature (Mușat; Predicting-Formation NeurIPS-25) is adjacent; density-threshold + discrete-slot formation on naturalistic corpora is a new curve | R8 flags: p0 net off-parity (val 2.82), "≤5%" rests on one n=20 cell, 10M wall-matched (steps confound) | Parity-matched p0 net + ≥3 events/cell + second seed (~4 cheap trainings) |
| 8 | **Mid-stack causal gate; depth non-invariant** (exists in every net; slides with seed/regime/architecture; census ρ 0.50-0.80) | **PARTIALLY-KNOWN** — staged/emergence literature exists; the *non-invariance* census is a useful negative | 4 nets + 3-scale ladder; steps-confound on multipliers | Not a paper; a framing figure ("laws are ensemble properties") for any of the above |
| 9 | **Late-MLP energy carrier** (zero/rotate 0.25-0.34 in 10/10 cells, 5/5 nets; equalizer: schedule decorative, allocation defended) | **PARTIALLY-KNOWN** — rhymes with massive-activation/outlier literature (not in scan scope); matched-energy causal dissociation is solid but small | 5/5 causal; equalizer leg 0.84M n=1 | One 2.7M equalizer run would close it — but low paper yield; keep as supporting result |
| 10 | **L5-calibrator** (KL 0.91-1.08 at ≤+0.046 removal cost, 5/5 — most robust mechanism in lab) | **UNKNOWN-GRADE** — scan never covered it; plausibly sink/outlier-adjacent | 5/5, causal, replicated | One mechanism story (is it logit-norm/temperature? one afternoon, eval-only) — then it either joins #9 or dies |

## 2. Top-3 paper candidates

**P-A. "Four faculties of model editing" (results 1+2+5, with 6 as motivation).**
Claim: editing failures decompose into address (concentrated, surgically removable,
scale-invariant), ability (distributed, train-only), expression (teacher-forcing-bound;
causally localized by transplant), history (erasure burns the address, not the attractor;
re-learned memory is ~3× surgical-proof). Missing for submission: (i) causal localization
of the expression failure — e055 residual transplant at the divergence token; (ii) scar
clause replicated ≥3 nets (e046-grade, ~3 cheap retrains); (iii) one head-to-head
RMU/first-order cell on our suite so the decomposition is anchored to the known baseline.
Cheapest path: all three reuse existing transplant/battery tooling; e055 is CPU-light.

**P-B. "The stream basis is written once" (results 3+4).**
Claim: the residual basis is init-anchored (causal interface-specific ladder), and
compatibility selection walks a landscape that never passes through donor alignment
(−5.6% damage vs +0.0015 alignment vs 0.53 ceiling) — plus the methodological punchline
that graft-damage assays read organ-reliance (r=0.807), not basis-fit. Missing: e050
directed-mutation arm with the T026 registered A-residualized readout (reachability vs
visibility), one second lineage/donor, and the G0-anchor fix. Cheapest path: e050 +
one residual-selection generation — days, existing harness.

**P-C. "The cache utility timeline" (e053, running).**
Claim: first per-position causal utility curve of a KV cache over long generation —
where cached entries die relative to sink + recency window (scan §1a: "nobody has
published this curve at any scale"; sinks proven universal even at 14M = our size class).
Missing: the curve itself at 2-3 context lengths + a sink-controlled variant
(lesion-sink-only arm) to separate dilution from dead weight. Cheapest path: e053 already
reuses lesion/patch tooling; add the sink-only arm.

## 3. Stop / double-down

**STOP: chasing the two-factor COMPLETE-erasure "second head" (C6's completion half).**
Demoted 2/5 nets; every new net is a documented seed lottery (B43 L4H4 vs BDO L3H1);
each attempt costs a full honesty battery for zero transferable claim. The address half
survives and is already in P-A. Novelty-to-cost is the worst in the lab.

**DOUBLE-DOWN: the expression-gap arc (results 1+2, experiment e055).** Highest
novelty (zero interventional studies anywhere), cheapest discriminating experiment
(transplant tooling exists), and it upgrades two inventory items at once.

## 4. Market line

Practitioners editing/unlearning models are blocked exactly where we are: evaluation
lies (continuation batteries overstate install 3.4×; RMU hides rather than deletes) —
the free-generation honesty probe + four-faculty decomposition is directly actionable;
the cache-utility timeline targets inference cost for long-context serving.
