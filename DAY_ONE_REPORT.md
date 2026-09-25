# Day One Report — The Functional Anatomy of a 2.7M-Parameter Character Transformer

**Neural Dissection Lab · 2026-09-24 → 09-25 · one RTX 5090 laptop · ~40
experiments, 11 visualizations, 7 frontier reviews, 19 thinking entries.**

*Da Vinci's question, asked of a small network: what is actually in there,
what does each part do, and what can be cut, moved, or written?*

---

## 0. The one-paragraph summary

A 2.7M-parameter char-level transformer trained on Shakespeare was
dissected along five axes — lesions, causal patching, training
interventions, transplants, and surgical edits — under registered
predictions with adversarial review. The day's central discovery is
methodological as much as scientific: **the laws are ensemble properties;
the mechanisms are samples.** Claims about *which specific component*
carries a function died under replication (4 of them); claims about *what
kind of structure* exists held across every net tested. What survives is a
set of seven graded laws, one complete editing doctrine, and a long list
of honestly-earned negative results.

## 1. The laws (card v3 — graded, replicated where stamped)

**L1. A mid-stack causal gate exists in every anatomy.** Decisions become
causally committed at a mid-stack point (activation patching; suffix-
monotone flip curves; 15% of positions distributed). The gate's DEPTH is
not invariant — it slides with seed and training regime (modes 3→5 across
four nets; cross-net correlation 0.50–0.80, failing the 0.8 bar).
*Qualitative universal / quantitative non-invariant. 4 nets.*

**L2. Anatomy is plastic; damage tracks energy.** Lesion maps reorganize
under architectural constraint (the MLP-0 keystone dissolves +4.08→+0.10
under stream-renorm training at parity loss; replicated on a second seed)
while front-loaded attention allocation persists. Damage tracks
perturbation ENERGY, not content (matched-energy rotation ≈ removal;
random-at-√2 worse). **The late-MLP energy carrier replicates 5/5 nets**
(zeroing costs 3–4× direction-scrambling; magnitude load-bearing,
direction nearly free).

**L3. The residual-stream basis is init-anchored — partially.** Organ
transplant compatibility follows initialization lineage, not training
regime; the alignment ladder: 1.0 (identical) → 0.53 (same-init,
different data order) → 0.15 (same-init, different regime) → 0.00
(different init). Stream-facing matrices (W_in reads, W_out writes) are
the violent grafts; attention matrices are mild (weak anchoring, not
shared subspaces). MLP hidden space is barely anchored at all.

**L4. Far-context retrieval is task-elicited, not architectural.** On
natural char data at this scale: no specific far retrieval (L5's far
attention is idle grazing; its calibration is local; far-value is bimodal
but structure-insensitive — shuffled far context hurts MORE). Given a
retrieval-required task (ID→COPY with nonce >16 tokens back): noiseless
retrieval (CE 0.007), a dominant head (95% ID-mass) inside a redundant
fan, and a new decision mode at L4. *Scoped: this architecture, this
scale, natural char data.*

**L5. First-order ascent cannot selectively forget — anywhere.** At the
real forgetting bar: naive r=1.23, projection 1.43, masked 1.8; against
memorization-symmetric collateral r=1.08 — the two memories are forgotten
at identical rates. An apparent r≈6 breakthrough failed to reproduce
against its own code and seed (chaotic event).

**L6. Entity knowledge is address-plus-body; removal is surgical,
completion is not.** Zeroing one rare letter's embedding+head rows (384 of
2.7M params) damages a name to ~13% accuracy at +0.0008 corpus nats with
class-exact collateral (S_name 573; the ascent control is 4,900× more
expensive). COMPLETE erasure was achieved once (rows + one head) but the
second factor does not replicate — the residual's structure is
net-specific. *Universal address-half / sample-level completion-half.*

**L7. The edit law: address, ability, expression, history.** REMOVAL is
surgical and robust. INSTALLATION is never surgical (best graft closes
6% of the gap; cross-init rows damage while installing nothing) but
exposure-trainable cheaply and selectively — EXCEPT that teacher-forced
install never achieves free-generation expression: zero occurrences
across all doses, temperatures, and seeding (battery 92–96% the whole
time); the installed address is bound to the trained continuation
geometry (p collapses 6× with 10 oldest context chars deleted) and its
prior is sub-argmax. The day's single expression event came from
naturally-trained free-context exposure. And HISTORY: after erasure and
re-learning, the address re-grows along its original direction (cos 0.76;
the attractor survived) while the new route is different (atlas ρ 0.21;
the old carrier head becomes an anti-carrier) and ~3× more resistant to
the original surgical key. *n=1 flagged.*

## 2. The meta-law (the day's deepest finding)

Every positive mechanism claim that died was n=1 and discovered-in-run;
every claim tested on ≥3 nets survived in scoped form. Small networks
are a degenerate ensemble: 36 heads at 7.6× superadditive redundancy,
orthogonal training motion across inits — WHICH component carries a
function is a seed lottery; THAT the coarse allocation exists is forced.
**Method adopted: mechanism claims enter the card at high confidence only
after ≥3 nets, stated as distributions.**

## 3. Instruments — validated and retired

Validated: causal-depth patching (replaced the argmax-stability lens,
which proved per-position UNCORRELATED with causal depth — ρ=−0.009 — its
dominant bin selecting causally random positions); matched-perturbation
controls; transplant R-bands; ΔW subspace atlases; free-generation probes
(the honesty check: continuation batteries overstate install by 3.4×).
Retired: lens-based decision depth; single-run head attributions;
continuation batteries alone.

## 4. The negative-results catalog (each a closed door with the key left in it)

Ascent unlearning (all granularities); far retrieval on natural data;
surgical installation; cross-net transfer of the two-factor erasure;
single-head localization of retrieval (redundant fan); stream-norm
geometry as the cause of lesion front-loading (front-loading survives
renorm); "one body head" transplantability (net-specific); attention
subspace sharing (portability = weak anchoring, not shared directions).

## 5. Process ledger (what the day taught about doing this)

The cadence — 10-min heartbeats, hourly adversarial panels, registered
predictions before every run, thinking-gates before launches — caught:
one foundation instrument (the depth lens), one false-positive verdict
(cosine of a near-zero vector), two non-reproducing "breakthroughs"
(r≈6 ascent; the rare-token head), one never-executed run flagged
"confirmed" (a smoke env-var), and two silent ledger losses repaired from
git. Rule adopted: verify every edit in the same breath; never trust a
commit message over the file; harvest headlines only from full reports.

## 6. Open edges (the honest frontier)

Does L1's qualitative gate survive scale (e005s: 0.7M/8M, registered,
frozen readouts)? Does geometry-bound expression generalize (content-vs-
position binding)? Is init-anchoring evolvable under lineage selection
(e040)? Where is the far-retrieval threshold on naturalistic refrains
(e049)? Can a body circuit — not rows — be transplanted at the BLOCK
level across lineage (e036's re-scoped successor)?

## 7. The tombstone inheritance, honored

The prior project's one surviving finding — output lm_head token-row
directions are causal training coordinates — became this day's surgical
instrument (L6/L7 are its behavioral proof). The old lab's discipline —
preregistration, matched nulls, adversarial review — became this day's
operating system. Nothing else was inherited as evidence. Everything here
was earned today, on one laptop GPU, by a lab that audited itself harder
than any outsider would.

---

*Artifacts: runs/e001-e048 + v001-v010 (metrics.json + figures per
experiment); THINKING.md (T001-T019, the full interpretation journal);
REVIEWS.md (R0.5-R7); scratch/ (design memos with measured probes).
Everything reproducible from lab/ scripts; everything recoverable from
git. The heartbeat continues.*
