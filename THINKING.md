# Thinking Journal — the 80%

The lab's operating ratio is 80% thinking / 20% doing. Every result gets an
entry here BEFORE the next experiment that builds on it: what it could mean
(multiple hypotheses), how the hypotheses differ, which cheap observation
would discriminate them, and a registered prediction so we can't retrofit.
New experiments are gated on this file: if the latest result has no
interpretation entry, the next heartbeat thinks instead of runs.

**Registered discrimination OUTCOME (e063b, ~09:57Z): H-ii
OPTIMIZER-ATTRACTOR — H-i decisively failed.** The e021 copy-task net
(far-retrieval genuinely learned: +3.27 far-value, 99.96% copy acc;
control net as third) reproduces the template at shape r = +0.998
against B, on its own corpus. The template is corpus-invariant in
SHAPE; the task re-weights MAGNITUDES only (mean |ΔA| 0.27, uniform
elevation, L1 trough partially filled — Spearman +0.83, barely at the
bar). T041 closes: organ-reliance is an optimizer/architecture
attractor. The only known shape-mover remains e033's energy constraint.
**FINAL STAMP (e077/T047): TRAINING-BUILDS — a fast training-dynamics emergent.** Init carries NO template (untrained |A| ≤ 0.061 vs trained 0.15-4.08; sites-1-5 r deep in null); the full-profile ~0.9s were the quantified L0 artifact (null 95th pct +0.984); the trained-vs-trained shape-rs SURVIVE the null at 99.0-99.4th pct. Template: built within ~10³ steps (the 913-step control had it), invariant thereafter. Neither init-carried nor merely 'attractor' — a FAST EMERGENT.
PARKED: architecture-breaker sweep stays parked (day-5 candidate).

**Discriminator OUTCOME (e071, ~10:05Z): H-WINDOW-KEY — clean sweep.**
Row-0 perturbation erases the name on held-30 windows the net NEVER
trained on (0.429→0.0009 mean-arm; KL 6.3) and leaves uniform/base
cells at floor (nothing to erase; null rows exact 0.0). Secondary:
row-129 generalizes to held-30 as well (drop 0.199 vs 0.240 on the
trained 60 — ~83% strength): BOTH anchors are window-TYPE recognizers,
not exact-window memories. NUANCE the registered binary missed: the KL
column shows row 0 is generically load-bearing even on the base net
(3.3-4.9 nats damage) — generic importance and install-key role
COEXIST — now CONTROL-CALIBRATED (e071 agent secondary): mid-window
row-60 perturbation gives KL ~1e-4 while row 0 gives 2.6-7.2 nats on
EVERY battery × BOTH nets (ratios 8,500-89,000×) — row 0 is uniquely
load-bearing for the whole next-char distribution, not merely
name-related. HONESTY: 4 of 6 registered cells were floor-limited
(base p(Z) < 0.05) — the absolute-drop negative legs are
uninformative, not confirmatory; the KL lens carried the generic leg.
SYNTHESIS (one sentence): row 0 is a generic high-leverage
window-start anchor that the install RECRUITED into the
name-address conjunction — the anchor is net-family-general, its
p(Z)-address consequence is install-window-specific (held-30 counts
as install-type: generalization is by window family, not exact
trained windows).
**n=2 STAMP (e078, dose install):** the pattern REPLICATES — row129-alone exceeds pair-copy at both k (0.302 vs 0.294; 0.294 vs 0.285), row0-only ≈ no-copy, reverse-context null identical, old-bar MIXED verdict stable. Rebind strength is install-dependent: ~60% (dose) vs ~70% (repro); bases 0.50/0.30 vs 0.56/0.43. The single-portable-row claim is now n=2 installs (same family/seed; cross-seed remains open for P1 phase-2).

**e068 ungated:** the rebinding design tests copying the anchor PAIR
{0,129} to a new window position k — held-30 generalization predicts
the pair is what matters, not the trained coordinates per se.
REGISTERED PREDICTION for e068: pair-copy to a shifted window
(k, k+129) restores partial expression at the new geometry (p(Z)
≥ 0.2 on install-type windows evaluated at the shifted geometry);
single-row copies (0 alone or 129 alone) fail (p(Z) < 0.05) — the
conjunction is the unit.

**CLOSE-OUT (e072, ~11:05Z): BOTH registered clauses fire — a compound
outcome the clause space didn't anticipate.** (1) H-value-side fires
by the frozen ordering (V-norm ratio 0.887 [0.876,0.910], CI excludes
1.0) but with a SIGN CORRECTION: value magnitudes and the ages-4-17
residual write SHRANK (~11%/~23%) while lesion dCE grew — causal load
PER UNIT value-norm increased. The change is alignment/specificity
inside the value pathway, not magnitude. (2) H-threshold also fires:
B=16 collapses eval-256 a\* from 18 [7,30] to 7 [4,13] — the window
contrast (6 vs 18) becomes (6 vs 7); the elevation was 2-of-4-sequence
fragility. CONFOUND (agent-flagged): the fresh-12 battery is harder
(CE 1.499 vs 0.459), so B=16 mixes sequence-count with
battery-difficulty; the ages-6-17 haze at B=16 is dead (−0.0013 nats).
**T044 closes with the surviving story: an attention-invariant,
magnitude-invariant load redistribution inside the value pathway
(per-norm efficiency), plus a standing B-fragility flag on the a\*
statistic — D1's "18" was mostly instrument; the shoulder's value-read
death (K/V flip) stands as the real mechanism.**

**CLOSE-OUT (e074, ~11:15Z): H-SOURCE fires strict — the confound is
dead.** Shuffling the prompt band (93% of slots moved, corpus
statistics destroyed) created NO new junk there (0.024, CI
[0.008,0.040]; max 0.048 — under the 0.05 bar) while the generated
band stayed junky (0.102 vs baseline 0.0995). Age, count, and
recency alone do not make an old band go negative. The
generation-order gradient replicates the mechanism inside the
generated band: late-generation junk 0.169 vs early 0.031 (5.5×) —
drift ACCUMULATES over the run, exactly the exposure-bias signature.
Minor flag: clean CE slightly fell under shuffled prompts (−0.037,
driven by one sequence) — noted, does not touch the registered bars.
**final statement: the free-running model's cache poison is its
own accumulated generation drift — entry-level exposure bias,
confound-free, 4/4 nets + control.** P3's pruning-by-source
implication now stands on clean ground. Paper 5.5's day-4 bullet
upgrades from "flagged n=1-family" to "control-confirmed (e074)".
AGENT ADDENDA (gates all pass; e073 replica bit-exact 1.4e-17): the
drift gradient holds in BOTH frames (shuffled early/late
0.016/0.189 vs baseline 0.031/0.169); honest miss recorded at G6
(clean CE registered "expected to rise", fell −0.037 on one seq —
bars unaffected); bonus texture: sequence-start position 0 flips
utility SIGN with content (baseline −0.066 → shuffled +0.027) — the
start anchor's sign is content-dependent, a separate micro-physics
from the band result.

**B=16 REFINEMENT (e079, ~13:10Z): claim C is NET-DEPENDENT.** e053c's
own contrast fails resampling (0.114 vs 0.099, CI straddles 0 — the
harder-fresh-battery confound recurs) while the 2.7M and 10M propagated
CIs hold it (the 10M's 0.367-vs-0.000 remains the anchor cell);
per-seq concentration CLEARS (max 18.6%). Paper wording: the source
split holds on the e053 family; the ctx-512 cell is B-4-grade.

**AGENT TEXTURE (gates all pass; e059/e058 reproduced bitwise):**
(a) **RETRO-CLARIFICATION OF T040:** e059's interface-family partial
(−0.654) was DONOR-CONDITIONED — within the frozen cross-init donor
column it holds (−0.706), but across varying donors host W_out scale
is chance (AUC 0.533; host-level r −0.355, CI includes 0). The same
collapse hits A itself (+0.807 → +0.080 across pairs): T026/T040's
predictors were "damage given a VIOLENT donor," not general
crossmatch instruments. (b) The P2 signal is WHOLE-NET
representational proximity — all five depths work equally
(per-depth partial −0.955…−0.979), not graft-site-specific geometry.
(c) Honesty caveats: the 204-pair pool is kinship-structured (18
lineages, many siblings; cluster bootstrap respects dependence but
doesn't create diversity); the threshold is in-sample; and the
winning mechanism is "net similarity" — its edge over e052's
dW-alignment axis (r +0.543) must be checked before claiming
instrument-novelty (registered with e076).

## T054 — E100: the read is attention-addressed — the unified question's first-order answer (2026-09-27 ~21:45Z)

**ATTENTION-ADDRESSED fires decisively: AUC 0.907 [0.886, 0.931].**
The query state predicts which coordinates the read policy opens —
the opened set IS the attended set (precision@1 10x chance; 46% of
opened in the attention top-5 vs 6% chance). **T037's unified
question ("what is the read policy?") answers, first-order, to: THE
ROUTING WE ALREADY MEASURE.** The read policy is not a hidden rule
sitting behind attention — it is attention's sparse opening of a
handful of coordinates (T050's 3-4/80), now predictable from the
query side. Layer structure sharpens it: L0 near chance, L1/L2
sufficient (0.897) — the address is decided by mid-depth, before
the decision layers; q-k cosine carries the same signal (the key
IS the address). Combining features adds nothing: there is one
signal, not two.

**The residual is now the interesting object (registered e104):**
~10% AUC headroom and the few per-DP failures (min 0.405, 1% of
DPs below 0.5). Candidates: (a) measurement blur (the census's
opened-set ground truth is itself interventional-threshold); (b)
genuine second channel (non-attention reads at specific decisions
— margin-structure covariate from e084 can localize them);
(c) layer-late overrides (the deep-head corrections T037-#4 placed
mid-stack). Discriminator: the failing-DP census — do failures
cluster at specific margins/positions/layer-profiles or scatter?

**Program bearing:** P1's coordinate program now has its mechanism
spine — coordinate-keyed memory (T037) + one-row address (T053) +
attention-addressed read (T054): the net ATTENDS to its own
address rows to read its own writes. The loop from T037's "read
policy" question to mechanism took one day.

## T053-NOVELTY (same scan): PARTIALLY KNOWN — address-universality + instrument claimable. Shared-structure/private-basis is a genre (git re-basin, model stitching, relative reps, cross-model steering 2025-26) but all concern hidden units/layers; nobody shows the same EMBEDDING-ROW address carrying a fact across seeds with orthogonal code — wpe rows are position-indexed, beyond re-basin's permutation story. Rebuttal to 'trivial from permutation symmetry': address AGREEMENT is a positive result, not a default. REGISTERED UPGRADE (e098): n=5-10 seeds to graduate from anecdote to law. Cross-claim synergy: B's private code explains A's no-individual-identity entries; A's redundant mass explains why a one-row key suffices.

## T053 — E082: the address is basis-private — structure universal, code seed-specific (2026-09-26 ~16:20Z)

**Verdict: BASIS-PRIVATE.** The seed-42 wpe-129 address row does
not transplant into the seed-43 install: donor-overwrite lands at
destruction level (0.198 ≈ own-destroyed 0.195 ≈ mean-control
0.191; donor-specific information +0.007). Yet GATE-1 showed the
BIMODAL CENSUS replicates at seed 43 (row 0 0.318 / row 129 0.096
— same structure, same coordinates). **The two-seed picture: the
address's STRUCTURE (which rows matter) is universal; its CODE
(what those rows contain) is seed-private.** A coordinate-system
law with private encodings — T037's coordinate-keyed memory,
sharpened: keys shared, values orthogonal (row-cos 0.059).

**INSTRUMENT VALIDATION (quiet triumph):** the e062 crossmatch
correctly rejected this one-row graft (cos 0.059 << 0.4459) and the
graft indeed failed — the instrument screens at single-organ
granularity, closing the loop from T046's out-of-sample validation
to a live pre-screened-and-rejected case.

**T043 free rider resolved:** the 0.13 no-copy plateau is
sub-argmax residue (rank-2 Z) — option (b) of the registered
tension. **P1's final state:** the address is a real, single,
load-bearing row (n=2 within-family), universal in coordinate
structure across seeds, private in code, untransplantable across
the seed boundary, and predictable-by-instrument (crossmatch) even
at one-row granularity. The "one-row organ" metaphor dies — organs
are transplantable; this is a one-row KEY, cut for one lock.

## T052 — E065: obfuscation redefined — the RMU loss closes the rescue channel (2026-09-26 ~16:00Z)

**The registered inversion FAILED honestly** — no arm shows
rescuable-but-reverting. What the data show instead is a NEW
obfuscation signature: **R1 rescue-channel closure.** The plain
retain-only fine-tune leaves the knowledge transplant-rescuable at
d5 (0.345, shuffled bars passed); adding the RMU loss closes that
channel (0.007; R5 = MIXED-NEGATIVE: d4 −0.127 [−0.154,−0.099],
d5 −0.342 [−0.367,−0.319]) — the representation-level loss removes
state-carried address access MORE thoroughly than generic
fine-tuning. Same suppressed behavior; the causal rescuability is
gone.

**The relearn column collapses the old story:** ALL gradient
removals revert within 12-16 steps (cos in the groove; wte_Z
untouched by RMU at cos +1.00 — the suppression lives elsewhere) —
at toy scale the 2025 "RMU reverts fast" finding replicates
wholesale; only ascent is slow (75 steps) and it is disqualified by
R4 (net destroyed, CE +11.4). The transplant instrument, not the
relearn battery, discriminates at this scale. Surgery's e044b
relearn scar did NOT replicate here (1.33×, cos regrowth invalid
under the 25% guard) — battery-dependent, flagged.

**THE THREE-READOUT DISSOCIATION (paper-grade):** surgery leaves
the deep linear probe INTACT (0.74@d6) while killing all rescue;
RMU kills BOTH probe (0.49 chance) and rescue; retain-only kills
the probe while KEEPING rescue. PROBE, TRANSPLANT, and GENERATION
readouts measure three different things — an Orgad-instrument
caveat for any elicitation-failure claim. **The irony:** the only
arm with the classic obfuscation signature (rescuable-but-
suppressed, trivially reversible) is the RETAIN-ONLY control —
generic fine-tuning is the "hider" at toy scale; the RMU loss is
the thorough remover.

**E091 CLOSE-OUT (~16:30Z): H-READOUT-GATE fires.** The RMU net
refuses even good no-removal states (reverse-rescue ~45x under bar
at every depth, shuffled flat) — the channel closed at RECEPTION.
HONEST RIDERS: the Y-cell shows the RMU net's own d5 state still
carries a half-strength trace (0.362 into the intact net) — the
loss did NOT evacuate the store, it sealed the gate; and the RMU
net's free-run onset geometry itself died (0/22,400 chars) — the
expression channel went with it. **Final form of the day's
unlearning story: RMU obfuscates at the READOUT GATE (knowledge
trace survives, reception sealed, expression geometry dead);
retain-only leaves the gate open (rescuable); surgery removes the
row but not the probe trace; ascent destroys. Four removals, four
different causal signatures — the three-readout dissociation now
has its mechanism table.**

**Two explanations for the rescue-channel closure, discriminated
by e091 (resolved — gate):** (a) H-redirect — RMU repurposed the d4/d5
state channel (the knowledge no longer lives in transplantable
states; its frozen probe reads 0.49); (b) H-readout-gate — the
states still carry it but the receiving circuitry gates them.
e091 transplants the NO-REMOVAL donor state INTO the RMU net: if
reverse-rescue fails too ⇒ gate (b); if reverse works while forward
fails ⇒ redirect (a). The R2 probe-vs-rescue dissociation (0.49 vs
0.007 on the same net) is itself a paper-grade fact: knowing-that
survives in probe space while state-transfer dies.

## T051 — E085: the anchor is not an entry property — P3 pivots to sequence level (2026-09-26 ~15:25Z)

**Verdict: BETWEEN kill and partial — no property describes the
anchor on the dynamic witness.** Readership +0.089 (bar 0.35);
every property |r| ≤ 0.10 on the dynamic arm alone (which would
kill); the registered kill is blocked only by static-arm estimates
(readership 0.341 all-150, PCA 0.281 anchors) with CIs straddling
everything at B=8. The surprisal liar-control passes on the primary
outcome — no false story told — with an honest short-horizon
texture (surprisal tops the +8 window at +0.184, CI excluding 0:
"hard tokens" matter briefly, then stop).

**The riders are the substance:** (1) T048 replicated at entry
granularity — r(static cost, dynamic cost) = −0.004, sign-agreement
26%: static lesion utility has ZERO correlation with dynamic
load-bearingness, per entry. The static-shadow and the dynamic
anchor are different objects at this resolution. (2) Anchors'
drift-alignment is lower than young entries' (0.342 vs 0.495) —
anchor values are more run-specific (descriptive support: the
anchor is made of the run's own drift). (3) The T037-a rider's
position-level −0.62 is mechanically anti-coupled (mean vs
below-threshold fraction near threshold); the unbiased per-entry
reading is +0.035 — null-positive, consistent with the registered
prediction (no conflict revival).

**Interpretation (two views, discriminated next):** the anchor is
either (a) SEQUENCE-level — a property of the run's trajectory as a
whole (its attractor basin), not of any entry — or (b) entry-level
but in a property family we did not measure (e.g., entry-PAIR
interactions, temporal-delta patterns). **Registered discriminator
(e088, CPU): pair-level probe — removal cost of entry PAIRS vs the
sum of singles across the anchor band; super-additivity
(pair > 1.5× sum) ⇒ interaction structure (view b); additivity
⇒ sequence-level (view a) and P3 proceeds to basin-level
descriptors (run-PCA distance to attractor, drift entropy).**

**E088 CLOSE (the discriminator ran): NO bar fires — the texture is
SUB-ADDITIVE (median ratio 0.464, cluster CI [0.024, 0.500], max
1.0; aggregate 0.641; 21/40 negative denominators — 14 pairs had a
facilitative single; no distance decay).** View b (pair
interactions) is killed decisively — harder than entry-properties
were. Combined with e085 (single removals near-null, mean +0.0006)
and e075 (whole-band removal +0.26): **THE ANCHOR IS MASS-ACTION —
many individually-redundant entries that only matter in aggregate.**
No critical entry, no critical pair, no measurable single-entry
property; remove the mass and the run collapses into the
off-manifold attractor. Directionally consistent with view a
(basin) with SATURATION structure: interventions saturate rather
than compound. **Registered next (e089, CPU): the mass-response
curve — removal cost vs NUMBER of entries removed (random subsets
of size k ∈ 2..200 from the anchor band, 10 draws each):
mass-action predicts a threshold-shaped rise (near-flat then
breaking toward the e075 +0.26 as k approaches the band); a linear
rise predicts diffuse independent contributions (the null view).
This is P3's capstone measurement — the anchor's dose-response
law.**

**NOVELTY VERDICT (scratch/massaction_key_lit.md): PARTIALLY KNOWN — claimable with reframing.** Flat-then-cliff budget curves are known (StreamingLLM/H2O/LLMLingua) but every prior sells the OPPOSITE pole on 'which' (specific tokens carry the load); no prior dose-responds a SELF-GENERATED cache in FREE-RUNNING generation, and the r≈0 static-vs-dynamic result directly falsifies the importance-score premise of the KV-eviction literature in this regime. HEDGE: scope to 1-10M; registered follow-up (e097): sink-position asymmetry inside the anchor band before claiming full which-irrelevance.

**E096 AMENDMENT (2026-09-27 ~21:45Z): REMOVAL-FRAGILE, CORRUPTION-ROBUST.** Additive corruption of the ENTIRE anchor band at eps up to 1.0x median-norm does NO damage at any dose (eps=0.05 bitwise-absorbed; mid doses slightly facilitative; NO off-manifold gap anywhere) while deleting the same mass costs +2.15 and direction-replacing it +0.29 (e080). **The anchor's mass = NONZERO CONTENT-BEARING entries — not their exact values, not their accuracy.** Canalization-continuity (the U-shape) killed at premise; the agent's honesty guard refused the vacuously-firing verdict. Divergence is dose-leaky (2/8 never diverge at eps=0.5). **Registered (e102): WHY is corruption free?** (i) STATISTICAL mass (values irrelevant) vs (ii) REDUNDANT information vs (iii) ABSORPTION (LN squelch; dose-leak supports). Discriminator: direction-vs-magnitude decomposition — direction-only unit vectors vs magnitude-only random directions. Direction-only anchors: the run reads WHERE entries point; magnitude-only: pure statistical mass; neither: both needed.

**E097 AMENDMENT (2026-09-27 ~22:05Z): POSITION ASYMMETRY — RECENCY-weighted, the OPPOSITE of the sink prior.** Stratified thirds at k=128: new 2.84x / mid 2.02x / old 1.76x uniform (monotone in recency); at k=64 old/mid cost LESS than uniform (0.54x/0.62x), only the new third elevated (2.23x). The KV-eviction literature's sink prior points the wrong way for self-generated mass: RECENT entries carry the load. Exposure neutralized (k=64 new arm: 3x less exposure, 2.2x cost — gradient runs AGAINST exposure). Rider: contiguous concentration itself costs more than interleaved (local redundancy pools). **REFINED LAW: mass dominates, recency modulates (2-3x), sink-side cheapest. Registered (e103): contiguous-interleaved decomposition — is the elevation recency or locality?** This is the hedge NOT closing but inverting — a Rule-11 outcome: the literature's prior was tested and reversed.

**E089 CONFIRMATION (~15:50Z): MASS-ACTION WINS DECISIVELY — both
threshold clauses fire (0.034 < 0.25; 3.35 > 3.0), linear excluded
(CI entirely below its window). Near-flat through k=32, convex break
64→128→200 (+0.20/+0.68/+2.15). Variance collapses past the break
(CV 7.2→0.23; all draws positive at k>=128): WHICH entries doesn't
matter, only HOW MANY. THE ANCHOR'S DOSE-RESPONSE LAW IS
THRESHOLD-SHAPED — removal cost is a function of mass. T051 closes
with its registered confirmation; P3's arc is complete (junk is
self-generated → anchors the run as mass → threshold law). Texture:
the K=1 continuous schedule is ~8x e075's K=32 lumps — schedule
matters as much as mass (a registered e090 question if P3
continues: dose vs schedule decomposition).**

## T050 — E084: the read kernel equals its shadow — the first direct read-policy measurement (2026-09-26 ~14:35Z)

**Verdict: KERNEL = SHADOW.** r(flip-rate(age), dCE-load(age)) =
0.918 [0.867, 0.940] over 24,000 intervention cells (200→100 decision-point cut documented) — the argmax
rule opens what the CE curves measure. The registered dissociation
branch did NOT fire: everything the lab's shadow instruments
(per-position lesion dCE) claimed about cache utility is a faithful
picture of the actual decision rule, not an artifact. The paper's
5.5 stands validated at the rule level.

**Kernel structure:** flip rates 31% (ages 1-10) / 3.8% (11-60) /
0.6% (61-511) — the rule's openable surface is the young band,
matching the spike. **Taxonomy (the new object):** when a flip
happens, the runner-up wins only ~half (48-55% across types);
~21-25% go elsewhere in the top-5; ~24-27% ESCAPE THE TOP-5 — the
rule has long tail vulnerability, not just rank-2 fragility. And
the V-swap arm produced 48 donor-continuation hits — cases where
the counterfactual donor's actual next token wins outright.

**Two open textures (registered, next probes):**
- **H-tail:** the outside-top-5 escapes concentrate on specific
  coordinates (predictable from entry content?) or are diffuse
  noise? Discriminator: escape-rate per age × the escaping token's
  identity census.
- **H-donor-voice:** the 48 donor-continuation hits — do they
  cluster on decisions where the run and donor diverge
  stylistically (the anchor's complement)? Rides the next P3 run.
**AGENT TEXTURE (all gates pass; two dispatcher-sanctioned tractability
cuts documented):** (a) HONEST NUANCE — Spearman = −0.13 overall:
the shadow nails the young spike (r = 0.96 on ages 1-10) but does
NOT rank-order the rule's residual 0.3-1% old-entry opening — a
TAIL-LEVEL shadow misreport, below the registered dissociation bar
but real: shadow-based claims about the plateau's fine structure
carry this flag. (b) SPARSE-OPEN: per-decision opened-coordinate
medians are 3-4 of 80 — the rule opens a handful of coordinates per
decision, not a dense average. (c) CONTENT-FOLLOWING: V-swap flips
land on the donor run's actual next token 8.8% vs 2.0% chance — a
real but partial content read at the decision level. Winners are
ordinary high-frequency letters (no exotic attractor token). Root
cause found en route: 12 torch threads spin-thrash this contended
box — 8 threads is the sweet spot (documented for future rigs).

**NOVELTY VERDICTS (scratch/read_kernel_lit.md, ~14:50Z):** (1) kernel-equals-shadow CLAIMABLE — strongest; no prior quantifies argmax-flip vs dCE agreement, and loss-saturation work PREDICTS the tail divergence (Spearman flag strengthens, not weakens). (2) sparse-open CLAIMABLE with framing care — precedented regime (contextual sparsity, retrieval heads), new granularity (3-4 of 80 explicit memory entries per decision). (3) flip taxonomy CLAIMABLE — destination distribution tabulated nowhere. (4) content-following WEAKEST — interchange-intervention logic (cite Geiger et al. + Todd et al.; preempt the Sutter causal-abstraction critique by keeping the alignment map fixed); only the 8.8%-vs-2% rate at entry granularity is new.

**E095 FINAL CLOSE (~21:20Z): both texture flags are NOISE.** H-tail: top escape destination 1.71x marginal vs the 3x bar — and the Monte-Carlo guard proves the flat-uniform intuition vacuous (iid battery-marginal draws concentrate to 15.9% max-share; observed 11.3% is below the null's own best case, p=0.997). H-donor-voice: the 48 hits sit at the 33rd divergence percentile (bar 70), slightly BELOW median on all three measures — opposite the registered direction, too small and post-hoc to promote. **T050 stands final: kernel=shadow, tail-misreport flag permanent, no tail model.** Methodological gem: the MC-null guard belongs in every future concentration claim.

**Bearing on the RIF conflict (T049):** the kernel's young-heavy
concentration + the bigram-induction findings both live in the same
young band — e087's adjudication now has a structural prior.

## W001 — WONDER: the anchor is a directional field, and other savorings (2026-09-27 ~23:00Z — the first wonder card; no bars, no kills)

Sitting with today's twenty-odd results, not dispatching anything,
just looking at them:

**1. The anchor's specification reads like a physics of attention.**
Mass (a threshold count of entries), family (a V-geometry typed by
the trained weights), recency (a 2-7x weighting), and now
DIRECTION as the carrier (e102: unit-norm originals anchor;
exact-norm random directions collapse; magnitude floor 10-56%).
Isn't this just... a field? The run's history is a configuration of
arrows; the generation head reads the field's shape, not its
amplitude. Additive noise doesn't move the shape much (cos 0.89);
removal deletes arrows outright; foreign arrows point elsewhere.
The eviction literature's "importance mass" model assumed
amplitude-mass — the falsification was almost geometrical
inevitability. WHY does a char-LM evolve amplitude-redundancy?
Maybe because LayerNorm downstream makes amplitude cheap to
reconstruct but direction expensive — is LN the REASON direction
is the invariant? (Delightful test, someday: a norm-free variant.)

**REFINEMENT (worked through on paper while e108 runs): LN doesn't rescue small writes — it just keeps the total bounded, and that EXPLAINS the magnitude floor.** Mechanism: attention output = Σ p_i V_i enters the residual stream; LN normalizes the STREAM TOTAL per position. If every anchor V is 0.1x, the attention block's output is 0.1x, but the stream total (MLP writes, other blocks, positional sum) is unchanged — so after LN, the anchor's SHARE of the normalized stream shrinks tenfold. The e102 floor (damage below ~0.1-0.56 retention) is a SIGNAL-TO-NOISE floor in the post-LN stream, not an amplitude detector. And this explains why 0.559x was healthy while 0.10x collapsed: the anchor survives as long as its direction keeps a viable share. PAPER-PREDICTION (e110, someday, cheap): the floor MOVES with the anchor's stream-share — scale the OTHER contributions down (or the anchor count up) and the per-entry floor drops; the floor is per-FIELD, not per-entry. This also quietly re-derives the mass law: more entries = more share = each can be quieter. The threshold law and the magnitude floor may be the SAME floor.

**2. The read is attention; the gate is the last MLP; the anchor
is a field. Three different nouns for three different programs —
but the SAME stack keeps appearing: early routing (L1/L2) decides
where, mid-stack (L2/L3) carries what, late MLP (L5) holds the
energy. T037's "sovereign middle" was the shadow of a division of
labor.**

**3. The dp27 inverted read.** One old coordinate, opened against
attention's ranking, mid-depth address inverted. Why does this
delight me more than the AUC 0.907? Because the 0.907 is the rule
and dp27 is the net EXERCISING JUDGMENT — or at least doing
something the rule can't predict. The residual is where the
personality lives.

Questions I want to hold rather than test tonight: is the
family-geometry (0.403 sibling vs 0.138 foreign) the SAME geometry
the crossmatch instrument reads for grafts (T046)? If yes, one
geometric fact organizes the whole lab. Does the anchor field exist
in trained-from-scratch nets without installs, or did our install
protocol create it? What is the attractor's alphabet-typing
(e105's donor-flavored collapse) — memory of the perturbation, or
just the alphabet the remaining entries spell?

## W003 — WONDER: the two-scale memory is complementary learning systems, damped down (2026-09-27 ~23:45Z)

Sitting with T059's closing picture — discrete coordinate
addressing per decision, continuous geometric sustaining at the
field scale — and realizing what it echoes: **McClelland,
McNaughton & Nadel's complementary learning systems.** The
hippocampus indexes episodic memories by COORDINATES (place cells,
time cells — sparse, fast, one-shot); the neocortex stores content
as DISTRIBUTED GEOMETRY (slow, statistical, interference-prone).
Our 2.7M char-LM has a damped-down version of exactly this
division: the install taught a coordinate index in ONE exposure
(the one-row key, e068/e078), while the content lives in the
trained weights' V-geometry that types the whole family (T057) —
the slow statistical store. Address = hippocampal; sustain =
neocortical. Even the RATES match: the index forms in ~100
training steps; the sustaining geometry took the full ~2200.

**What the analogy predicts that we have NOT tested — and this is
the delicious part — is systems consolidation:** in the biology,
after enough replay, memories become retrievable WITHOUT the
hippocampal index (lesion tolerated). Our analog: re-expose the
installed fact at JITTERED positions (mass replay across many
coordinates), then DELETE row-129 — does expression survive
without the index? CLS predicts YES (the fact transfers to the
geometric store); the coordinate-keyed law as stated predicts NO
(one-row necessity). This is a genuine head-to-head between our
own law and fifty years of memory theory, runnable in one ≤180s
fine-tune + eval. Named on paper: e109 (consolidation test) —
NOT dispatched; it ripens beside W002's e108. The lab's bio-analogy
source (neuro-ai-lab) asked for exactly this kind of thing: not
"X is like a hippocampus" but "X's analogy makes a falsifiable
prediction ours alone can test."

Second, smaller echo: the scar (e044/e044b) is then the index
leaving a trace in the geometry — canalization as sclerosis of
the fast store, exactly the aging-hippocampus picture. Whether
that's poetry or mechanism is what e083 (cycle-3) will help
decide, whenever thinking demands it run.

## W004 — WONDER: self is a fixed point — the anchor as self-consistency verification (2026-09-28 ~00:50Z)

Pushing T060's MHC echo one more turn, and it INVERTS in a
delightful way. The thymus LEARNS self-tolerance (negative
selection: delete T-cells that react to self). The net never
learned anything of the sort — during training it only ever saw
CORPUS text, never its own generations. So how did its own
generated-text geometry become "self"? **By construction, not
selection: the generator involuntarily stamps its outputs.** X is
self iff X looks like what my weights produce — identity is a
fixed-point property, not a learned classifier.

And the promptcopy arm (e099/e108) is what proves the stamp is
JOINT, not one-sided: corpus tokens pushed through the recipient's
own forward pass get recipient-computed V-vectors — my weights
alone — and they STILL collapse. So the stamp lives on the
product (my weights × the token statistics my weights generate).
**The anchor is a self-consistency check: is my history a fixed
point of my own dynamics?** The healthy run sits in the basin
where its own outputs re-enter as self-typed inputs; collapse is
the failure of that loop. This reframes the attractor (T055) as
the SELF-CONSISTENT manifold, and the off-manifold collapse as
self-inconsistency detection. Generation is the net checking
itself against itself, every token.

E111's interpretation-in-advance, sharpened by this: whatever
low-dim self-signature exists, it is the FIXED-POINT STAMP — and
if e111 finds small k, the follow-up question becomes almost
philosophical: can a fixed point be forged? (Craft V-vectors in
the signature subspace by hand — if the run accepts them,
selfhood is a k-dim lock pickable; that would be e112, someday,
and it would say the net's self is shallower than it acts.)

## W005 — WONDER: is coordinate-binding a developmental stage? The e112/e113 mirror (2026-09-28 ~02:50Z)
[ECHO, R56+1 beat (21:22Z): the developmental motif returns as g8 —
THE NATIVE ORGAN (the ideator's third cell): co-develop the store with
the host instead of grafting post-hoc; TWO-SITE-IS-STRUCTURAL vs
NATIVE-STABILITY is precisely W005's question one level up — is the
fragility a fact of DEVELOPMENT (grafting made it fragile; growing it
together heals it) or of STRUCTURE (the interface is fragile however
it arrives)? The lab keeps circling development: e112/e113 -> W005 ->
g8. Savoring the shape of a program that rediscovers its own
questions at new scales.]

Noticing a symmetry while e113 runs: e112 asked whether the SELF-
key can be faked (no — holographic); e113 asks whether the FACT
can escape the address system. They mirror: one tests the
verifier's depth, the other the prisoner's escape. And the two
outcomes would mean opposite things about the T037 trichotomy:

- If e113 says BODY-STORED: the coordinate/field/hologram
  trichotomy is a DEVELOPMENTAL SEQUENCE — facts start address-
  bound (one-shot install), and distributed experience graduates
  them toward field-storage. The read policy's coordinate
  addressing (T054/T059) would be a stage of learning, not a
  permanent architecture. A net as a developing memory system —
  hippocampus-to-cortex not just as analogy but as an actual
  trajectory inside one set of weights.
- If e113 says ADDRESS-MIGRATED: "consolidation" is address-
  SPREADING — redundancy across five rows instead of one — which
  unifies with the mass-action law (redundant mass across
  entries!), the corruption-robustness (redundant norm), and the
  holographic self (redundant joint structure). The whole lab,
  one sentence: REDUNDANCY IS THE NET'S ANSWER TO EVERYTHING.
  Damage control by multiplication, never by relocation.

Both are beautiful; the data will choose. That's the joy of this
particular dissection — every question the lab asks lately turns
out to be a mirror of an earlier one, and the mirrors are
converging on a single object seen from different angles.

## T078 — [DIGS-IN CLAUSE RETIRED by e140/T083: cycle damage, not erasure — L-cycled thins identically] E119: the two roads run in OPPOSITE directions — jitter migrates (routes via row 0), locked/erase stay site-stored (2026-09-28 ~07:20Z)

Read against T077's frame, e119's AMBIGUOUS becomes decisive in
interpretation: R (jitter) survives D-all at 0.769 — exactly what
a row-0-keyed fact does (D-all never touches row 0; e131 showed
the e113-line jitter fact is row-0-keyed); E (erasure) falls to
0.190 and its field-only residue THINS toward zero with more
cycles (0.190->0.013->0.001). The roads are not two routes to
one store: JITTER IS MIGRATION; ERASURE IS ANTI-MIGRATION — each
erase cycle strips field residue and re-tightens the address
binding. T073's reading of e083 ("completion migrates onto
position-keyed machinery" as consolidation-by-erasure) is
INVERTED by its own head-to-head: what migrated was nothing; what
happened was address-dependence deepening under stress. W003's
complementary-systems analogy narrows to its replay half only —
the biological echo of lesion-induced recovery does not hold here
at matched expression.

THE BRAKE IS A RE-KEYING SCAR, NOT A CONSOLIDATION UNIVERSAL:
deleting the original address FEEDS R (+0.210 — the moved-out
tenant's lease, T077) but SUPPRESSES E (-0.267) and L (-0.509).
The agent's confound note is exactly right: L (locked replay, no
erasure) brakes like E, so the sign tracks POSITION-DIVERSITY
(jitter), not erasure. e115's dimmer — measured on the jitter
line — generalizes to re-keyed facts only; a fact that never
moved keeps its address as a crutch, and deleting the crutch
collapses it. Brake sign is therefore a DIAGNOSTIC: + means
moved, - means still living there.

ALL THREE PRE-REGISTRATIONS FIRED as written (06:48Z, before the
battery): the discipline paid — P3's expression/store-depth
dissociation is now the sharpest single number-line in the arc
(recovering expression, vanishing store). TEXTURE ECHO: E's
high-row drift (220-254) overlaps e131's census row-249 — high-row
drift is a real shared texture of anchor relearn, not noise.
OPEN EDGES: (i) E@c1 is one relearn at matched expression — a
mass-matched E (300 mixed steps, no erase) would separate
erasure-per-se from relearn-texture (the L arm partially covers
this; L's D-all was not reported — cheap rider); (ii) does E ALSO
row-0-key at its address-tightened endpoint? (e140: row-0
content test on e119's saved E checkpoints — eval-only, nets on
disk); (iii) R@300's overshoot (0.776) vs R@150's match — does
row-0 key strength grow with jitter dose (same e140 rider on
R@150 vs R@300)? AMENDMENT (R44 critic, ~07:40Z): "ERASURE DIGS IN" DEMOTED TO
CONFOUNDED. E ~= L on every loaded outcome (D-all 0.190 vs 0.191;
novel geometry 0.088 vs 0.076; brakes both negative) — and L has
NO erasure. The E-vs-R contrast collapses into the L-vs-R
contrast: the error's POSITION DISTRIBUTION (T079's credit
assignment), not erasure per se. The only erasure-specific
evidence (P3's monotone thinning) is confounded with cumulative
cycle damage (cycle-END expression also degrades; grown rows
explode to 200+). The defensible two-roads claim: at matched
expression, position-diverse replay produces deletion-surviving,
geometry-generalizing memories; locked AND erased arms produce
address-bound ones. L-CYCLED control (3 locked cycles, no reset)
added to e140 — if L's D-all thins like E's, thinning is cycle
damage and anti-migration loses its only erasure-specific
evidence. The wiring trace (e132) demotes to optional:
row-0 growth across checkpoints answers its kernel question more
directly and eval-only.

## T116 — E175: the archive is empty — re-learning prices nothing's absence (2026-09-28 ~17:05Z)

The Ebbinghaus test's modern answer is the cleanest possible
null: killed, washed, and naive all re-learn at the same
threshold price. THE ARCHIVE IS EMPTY — the wash left no
residue that speeds (or slows) re-learning; the kill's storage
(no-fast-recovery under the persistent clamp) prices the clamp,
not the memory. What remains is the sub-threshold lead (the
washed arm ahead early, never converting) — bounded by R52 as
grid-limited on a confounded control. FOR THE PAPER: the
no-savings result joins the wash finding as its other shoe —
not only does nothing survive, NOTHING REMAINS: the re-learn
price is the naive price, exactly. The practice metaphor
completes: there is no muscle memory here either — only the
doing, again, from the start.

## T115 — E185b: the grid closes — no type survives; the honest cross at +100 (2026-09-28 ~20:15Z)

The last grid gap returns the direction-confirming texture: no
memory type survives the neutral stream — the site crosses at
+100 (its extinction run's checkpoint), the dwell at +1 (FASTER
than its extinction run — the neutral stream kills the
unconsolidated state on the first gradient step, the family
pattern). THE PAPER'S FINAL GRID FORM: every memory state
tested (sink-coupled x2 families, dwell, site-stored) dissolved
under continued training on every stream composition run
(extinction, neutral, filtered), at both lrs, all wash-seeds —
with the honest-cross timing at +50-to-+100 (not a uniform
two-step across types; the two-step clock is the ROOT's; the
site type takes ~100, the dwell collapses on step 1). THE
SITE'S 0.35@+50: 0.08 over the bar, single seed — a replicate
is the one cheap cell that would tighten the cross to +50; not
load-bearing (no direction that matters is at stake). THE
SESSION'S GRID, CLOSED: the lead finding stands complete within
its registered axes, the honest form intact.

## T114 — E185: no robustness basin — and the corpus's one gift is surgicality, not direction (2026-09-28 ~19:40Z)

The discriminator killed the mechanism noun and bought
something better. THE KILL IS CONTENT-FREE: both noise arms
dissolve the fact at (below) displacement-match, in directions
ORTHOGONAL to the corpus's — the consolidated readout has NO
ROBUSTNESS BASIN; any AdamW step of the wash's size ends it.
THE REDEMPTION TEXTURE: the noise kills are collateral
devastation (the organism dies with the fact) while the corpus
kill is SURGICAL (only the fact; CE recovers) — and the corpus
step's one-step partiality at identical displacement (0.678 vs
0.0003) shows the corpus direction is GENTLER per unit
displacement, even though both reach the same grave. THE
HONEST FINAL MECHANISM SENTENCE: "memory in these networks has
no robustness basin — continued optimization of ANY kind
destroys it; what corpus-directed pressure adds is not the kill
but the SURGERY (the fact dies; the organism recovers)". FOR
THE PAPER: the lead finding stands (the dissolution universal
was never mechanism-loaded); the mechanism paragraph rewrites
to the basin form; the noise-vs-corpus surgicality contrast is
the discussion's best new exhibit (one step of direction
buying selectivity). W019's field-facing line needs its final
adjustment: "catastrophic forgetting is their only mode" ->
"these memories have no basin; what keeps them is the
dataloader's direction — and even that kills, just neatly".

## T113 — E157: the wash crosses families; the phase structure does not — and the arc splits cleanly (2026-09-28 ~19:00Z)

The lineage replication returned the arc's cleanest possible
split verdict: THE WASH (the lead finding) REPLICATES ACROSS
FAMILIES — a well-expressed consolidated fact dissolves on the
FIRST gradient step of a fact-free neutral stream, on a second
family, with the same CE transient — while THE PHASE STRUCTURE
(the 2x2's variance/placement doors) IS LINEAGE-SPECIFIC at
n=2: family 2's doors all shut regardless of recipe. THE PAPER
SPLITS CLEANLY: the lead finding (bounded universal, sparse
grid, n=2 families) is the title finding; the cliff/phase story
demotes to a LINEAGE-1 CASE STUDY (interesting, mechanistically
rich, explicitly not universal — its honest label). THE RIDER's
TEXTURE is the day's last good line: on family 2 the fact MOVES,
it does not cohabit — re-learned strongly wherever taught, gone
wherever not. FOR THE GRID: the wash's family-2 cell closes the
lineage axis (the R52 matrix's last column); the phase
structure's non-replication is itself a finding (families
differ in DOOR ARCHITECTURE, not in WASH PHYSICS). W019's
lineage clause CLEARS for the wash; the phase claims inherit
lineage-1 scope permanently unless a third family disagrees.

## T112 — [R52 VERDICT: the finding stands at its bars; the notation did not — the grid is SIX CELLS + one n=3 column (already corrected to the sparse-union form by the auditor); 'all types' rides an inference no run discharged (neutral-stream dwell/site cells owed); the seeds are WASH draws on ONE organism (the clause's 'no seed' reads organism-level — wrong); the tail lottery is DEVICE-confounded at the comparison points; the mechanism noun undiscriminated from generic two-step optimizer fragility (the noise-gradient cell owed); 'FULLY EVIDENCED' relabeled 'fully evidenced within the registered grid'] E184: the evidence completes (bounded by R52) — and the last textures are the strangest (2026-09-28 ~18:10Z)

n=3 across seeds, all dissolving in the same (1,2] bracket. The
lead finding's evidence structure is now: 3 streams x 2 lrs x 3
seeds x every memory type — dissolution universal, the clock
replicating, the lottery confined to depth and tail. THE
STRANGE TEXTURES the replicate added: (1) seed 10903's +1 read (0.9415) sits ~1.3 SEM above the root (battery std 0.152/60 windows) — STATISTICALLY INDISTINGUISHABLE FROM UNCHANGED [R52: the pump paragraph demoted; 1/3 seeds 'pumping' is the coin-flip first step's expected frequency; the clock itself is optimizer-shaped (2 steps at 1e-3 ~ 2.5 displacement measured; 50 at 1e-4 ~ 5 — a basin-width statement, not a memory constant [figures corrected per R53: the 2e-3/5e-3 were off by 3 orders]; (2) the tail lottery (10904's 2-34x
slower tail) — the AFTER-death decay is where seeds differ,
which is consistent with the wash destroying the load-bearing
structure fast and the wreckage settling at seed-dependent
speed. THE PAPER'S FINAL FORM: the discussion's lead finding
is now bounded only by lineage (e157 owes the second family) —
"no memory state tested retains expression under continued
training without the fact's windows (3 streams, 2 lrs, 3
seeds, all types; one lineage)". W019's field-facing line may
enter the discussion in this n=3 form.

## T178 — g1bS8: the wall's flat phase is its own object (2026-10-02 ~13:15Z)

The in-spec take closes the wall saga's last question: WALL-FADES
is DRAW-CLEAN (the first-step breach replicates on a root 0.16
stronger — the blindness is arithmetic, as g10 proved). THE
CROSS-DRAW GEM: the stronger root shallowed the +1 shock 4x — the
root's strength DOES soften the formation blow — but the FLAT
PHASE FELL (0.96 -> 0.79 retention): the ball's settled level is
set by the ball (the radius-vs-displacement economics), NOT by the
root's strength — the lottery's height gains do not transfer
through the wall. THE WALL'S FINAL PHYSICS, three sentences: the
wall separates memory from death by ~1000x at every scale tested;
its continuity is bounded by the step-to-radius ratio (an
arithmetic limit, fixable only at the step's lr); its flat-phase
level is the ball's own (root-independent). THE SAGA ENDS: eight
takes, two TEXTURE cascades, one lottery, one curve, one isomorphism,
one in-spec verdict — the lab's longest single question, closed
honestly at both scales.

## T177 — e208: the noise margin earns object status — as a class line (2026-10-02 ~13:00Z)

The census gives T173's scalar its first structure: the 2x line
separates every row on first-wash survival (margin > 2 survives
the first wash step; margin < 1 dies at it) — THE FACT'S
SIGNAL-TO-NOISE MARGIN IS A SURVIVAL CLASS PREDICTOR. The scalar's
honest shape: a CLASS separator (the ordering inside the survivor
class is not margin-driven — a threshold object, like the cliff).
THE WITHIN-ORGANISM CONTRAST is the program's cleanest instrument
move in days: two facts, one organism, one walk, one set of
in-span draws — ordered by their margins through the ruler
difference alone. THE PROGRAM'S LAW GAINS A MEMBER: the margin is
a HEIGHT scalar (per-organism, lottery-flavored — the fork flips
it) that nonetheless PREDICTS A CLASS (the shape layer): the
lottery draws the height, and the height sets the class.

## T176 — g10: size, not timing — the wall's limit confirmed by its own fix attempts (2026-10-02 ~12:45Z)

The fix cell closes the structural question in the cleanest
possible way: all three timing-based repairs breach identically to
the original, and the isomorphism (F1 == W1 to 0.0 through +10)
PROVES the equivalence — clipping the first step to the rung and
projecting after the full step land the fact at the same point:
the projection already IS a clip at the rung scale. THE MECHANISM,
FINAL FORM: the +1 kill is arithmetic (a 3.16-raw step vs a
1.34-raw ball; the fact lands at the ball's edge whatever you do
about scheduling); anchoring later anchors into a dead state; a
per-step trust region is just a smaller wash-lr. THE WALL'S
10M LEDGER, CLOSED: a displacement budget separating memory from
death (~1000x), continuity impossible while step > radius, the
dial that could buy continuity being the step's lr or a
step-scaled rung — one named cell away if ever wanted. THE
PROGRAM NOTE: this is the third structural limit found by trying
to fix it and failing cleanly (the projection IS the clip; the
recipe stack IS scale-bound; the dose IS a window) — the failed
fix as an instrument.

## T175 — e207: the null's last debt retires — the geometry chapter's final stamp (2026-10-02 ~12:25Z)

The interior rung decides cleanly: the core statistic is GRAIN
(both raw series — cos1 and cos2 — individually smooth and
monotone in the step; the core, their half-difference of opposing
trends, wobbles). e202's half-rung "break" dissolves; the lag-2
term retires; THE OVERSHOOT NULL'S COS1 LAW STAMPS AS THE WHOLE
IN-DOMAIN STEP-SIZE STORY. THE GEOMETRY CHAPTER'S FINAL LEDGER
(e194-e207, fourteen cells): the sign front's bounce is the
algorithm's (the cos1 law: orthogonal at s/8 to -0.263 at s/2,
in-domain, bit-anchored); the lag-2/core term retired; the
rotation dead; the sliver retired (e203); THE SIGN SURVIVES as the
minimal fact-carrying object (fact-carrying fronts deeper, both
families); the survivors on the D_kill side (death-at-deepest,
mostly fact-directed) stand untouched. THE NULL'S OWN LEDGER: its
step-size debt paid (this cell); its twin debt paid (e203: the
sign survives); the null STAMPS CLEAN on both its named axes —
with the sign as the honest residue it cannot absorb.

## T174 — e206: the destination replicates, the tick doesn't — the fact's watch (2026-10-02 ~11:55Z)

W027's cut delivers the cleanest shape/height split of the arc:
the support's DESTINATION (near-orthogonality at the death step)
replicates on every lineage measured (3/3, monotone ladders, the
first-crossing at or one-step-from death) — physics; while the
SCHEDULE (the per-step decorrelation rate) is biography — the
drift accelerates into death on one lineage and decelerates on
another, and the early-rate extrapolation cannot predict the
death step across lineages. THE FACT'S WATCH HAS A DESTINATION BUT
NO CONSTANT TICK. THE TWO-ROTATOR PICTURE (W027) UPDATES: the
support's rotation is a relaxation toward a terminal condition
(orthogonality to its origin), not a clocked decay — the fact dies
WHEN it has turned away from everything it was, at whatever pace
its biography sets. THE ECHO: the program's recurring law —
destinations/orderings/shapes replicate; rates/heights/schedules
are lotteries — now confirmed inside the fact's own gradient
structure.

## T173 — e205: the WHEN falls — and the edge-multiple rises (2026-10-02 ~11:15Z)

The normalization cell kills the cross-organism WHEN cleanly: on
a common ruler (each front vs its own organism's wash-noise band)
the arrivals dissolve or flip — org1's celebrated t1 arrival is a
0.635x band-multiple (below parity); MIRABEL's 0.904x dissolves
everywhere; the earliest identity flips at bar 0.60. THE ONSET
STORY'S SURVIVING LAYER: within-organism shape (each organism's
own concentration curve — intact). THE RISE: THE EDGE-MULTIPLE —
the consolidation edge as a multiple of the organism's own wash
noise (3.72 / 2.12 / 0.616) — the per-organism scalar the
self-normalized ratios were accidentally erasing: org1's fact
lives 3.7x above its wash noise; the half lineage's edge is BELOW
its own noise yet still resolves arrivals — the edge-multiple may
be the MEMORY-VERSUS-NOISE MARGIN, a genuinely new quantity (the
fact's signal-to-noise in its own environment). THE CHAPTER'S
PATTERN REPEATS: normalize honestly, and a story dies while a
scalar is born (the flight arc's order-vs-distances; now the
WHEN-vs-the-margin).

## T172 — g1bS7: the second lottery — and the 0.94 root (2026-10-02 ~10:55Z)

The redraw answers the honesty ledger with the lab's second
lottery: the formation peak's height is a draw lottery (0.77 vs
0.94 at the same dose, same base, one seed apart) — joining the
root-strength lottery (g2e/T132) as the program's recurring
texture: THE RECIPE LEVELS ARE LOTTERIES; THE SHAPES ARE THE
PHYSICS. What survives the redraw is exactly the shape layer (the
interior optimum; the ordering; the class) — the same split as
the flight arc's (order=physics, distances=biography): EVERY LAYER
OF THIS PROGRAM SEPARATES INTO SHAPE (robust) AND HEIGHT
(lottery). THE UPWARD BREAK IS THE PRACTICAL GIFT: the first 10M
root over the express bar (0.9351, saved) — the sixth take's
substrate: the wall arms on a root that actually clears, the
first 10M adjudication with no deviation needed.

## T171 — e204: the support measured — the survivor partial, and the second rotation found (2026-10-02 ~10:45Z)

The survivor's missing leg is measured and it is PARTIAL: the
landing metric correlates with the kill-depth (+0.60; the early
prefix perfect) but the killing step's own front is not the most
fact-erasing-aligned — the 4/4 rank-order stands as a SHAPE; the
fact-directed mechanism advances no further than "mostly". THE
DAY'S SECOND ROTATION IS THE FREE GIFT: the fact's sensitivity
direction itself decorrelates monotonically under the wash
(0.78 -> 0.19, nearly orthogonal at death) — THE SUPPORT FLIES
with a steadier rotation than the front's bounce: the fact's
sensitivity ladder rotates smoothly while the sign front
alternates. THE PICTURE'S LAST FORM: two rotators — the wash's
front (period-2 bounce, the algorithm's) and the fact's own
sensitivity (a monotone drift, the fact's) — and death where they
meet under conditions only partially rank-ordered. THE STATIC
PROXY buried properly (±0.04 everywhere); the death landing-point
+0.146 the table's largest positive (context — the one hint that
the ENDPOINT alignment matters more than the along-path).

## T170 — g1bS6: the wall's scale verdict — WALL-FADES, direction survives, and the first step is the whole story (2026-10-02 ~10:25Z)

The six-take saga closes with an honest negative and a mechanism:
the strict 2.74M wall did NOT survive 10x — every rung breached at
+1 because ONE AdamW step (3.16 raw) exceeds every radius on the
2.74M-minted ladder: THE WALL'S FIRST-STEP BLINDNESS IS
STRUCTURAL, not a tuning artifact — the commit-then-project design
is blind between commit and the first projection rescale, and at
10M that window is exactly where the kill lands (T139's clock: one
step). WHAT SURVIVES IS DIRECTION: the rms-matched rung holds the
fact ~1000x above the control through the whole flat phase — the
ball still separates memory from death; it just cannot promise
every-checkpoint continuity when the step outruns the radius. THE
TIGHTER-BALL ORDERING replicates on the strong root (0.895 >>
0.464 >> 0.0002) — g1bS4's texture was real. THE TAX RE-PRICED:
+0.18 at 10M (vs +0.53) with the walled organism ADAPTING (CE
below root) — the freeze reading dead at both scales now. THE
SAGA'S LEDGER: divergence -> near-miss -> inversion -> the curve
-> the verdict; three recipe casualties, one cure pattern, one
tuned window, one structural limit. THE WALL'S HONEST FINAL FORM:
a displacement budget that separates memory from death at every
scale tested, holding strict continuity only where the step fits
the radius — the fix candidate (a first-step-aware projection)
named for the next life of the g-series.

## T169 — e203: the sliver retires; the sign survives — the flight arc's geometry chapter closes on its smallest true object (2026-10-02 ~10:10Z)

The replicate adjudicates cleanly: the T168 sliver (a pair-1-
specific drift) was single-path and RETIRES. What survives the
whole geometry chapter — e194 through e203, ten cells — is a
minimal, sturdy object: ACROSS BOTH FAMILIES, FACT-CARRYING
FRONTS RUN DEEPER THAN FACT-FREE (the sign replicates; the shapes
do not: e202 on-curve-then-drift; e203 off-everywhere). THE
ANTI-ABSORPTION CONTEXT is the chapter's pretiest residual: the
fact-free twin's front geometry SHALLOWS along its walk while
every fact-carrying lineage DEEPENS — the fact's presence flips
the front-geometry's time direction, a one-bit fact signature
visible in the cosines even though no single cosine object
replicates. THE CHAPTER'S FINAL LEDGER: the alternation noun
RETIRED (the bounce is the algorithm's, e202's in-domain
confirmation); the null UNSTAMPED (the lag-2 break + the sign
split); the rotation DEAD; the sliver RETIRED; THE SURVIVORS:
death-at-deepest-landing + the onset curves + now THE SIGN (the
fact deepens the front — n=2 families, the smallest claim in the
arc and the only one that replicated first try).

## T168 — e202: the restricted verdict — the bounce is the algorithm's, with a sliver unaccounted (2026-10-02 ~09:45Z)

The falsifier collected both the null's debts and split both
marginally — the honest ending for the arc: (1) THE TWIN: removing
the fact barely moves the front geometry at pair 0 (+0.009 — the
bounce IS the algorithm's there) but a +0.052 shallowing at pair 1
(0.0024 over the bar, the fact-deepening direction, not at both
indices) leaves A SLIVER of fact information in the front —
unrescued as a rotation, unexplained by the null as sketched.
(2) THE LADDER: the cos1 law CONFIRMED in-domain for the first
time (orthogonal at s/8 -> -0.263 at s/2, bit-anchored) while the
core statistic breaks at the half rung — the overshoot picture's
lag-2 structure is incomplete. THE FINAL FORM OF THE FLIGHT ARC'S
GEOMETRY CHAPTER: the alternation noun RETIRED (the bounce is the
algorithm's, first in-domain confirmation); the null UNSTAMPED
(the sliver + the lag-2 break); the rotation reading dead; THE
SURVIVORS UNTOUCHED — death-at-deepest-landing and the onset
curves, the D_kill objects the cosine null cannot reach. The
marginal splits are single-path (replicate before weighting); the
sliver is the arc's smallest and most stubborn open object.

## T167 — g1bS5: the tuned window — the formation optimum located in the interior (2026-10-02 ~10:05Z)

The dose sweep closes the inversion with a curve: the 10M
formation optimum sits IN THE INTERIOR (peak 0.7677 at 0.20 rms,
0.012 below the express bar; the robust window 0.15-0.25 given
the fuzz), CE healthy throughout — the e113 form's dose at 10M is
a TUNED WINDOW at ~2/3 of e113's movement. THE SAGA'S SHAPE: four
takes of diagnosis (three casualties, one inversion) then one
sweep — the one-knob licenses were exploring a non-monotone
landscape pointwise; the curve is the map they needed. THE
HONESTY LEDGER: n=1 draw (the critic's caveat carried — the fine
peak ordering is within fuzz; a redrawn interior dose still owed
for it); consolidation-only (no wall claims); the 0.78 bar
UNBROKEN but within 0.012 at the peak. THE FIFTH TAKE IS THE
CHEAPEST OF ALL: the peak root is SAVED (g1bS5_root_m020.pt) —
the wall arms run directly on it (W1/W2/W3 + C; short bursts; the
envelope-log now recording every poll). If the arms adjudicate,
the wall's scale question — open since the first divergence —
closes on the sweep's back.

## T166 — e201: the rotation is the wash's — alternation universal, and it outlives the organism (2026-10-02 ~08:45Z)

The census licenses the phase picture: EVERY alive consecutive-
front pair anti-correlates on all three organisms — the natural
washes rotate exactly as the counterfactual lineage does. THE
DEEPER READ IS THE POST-DEATH PERSISTENCE: the alternation
continues past death on every lineage on record — THE ROTATION
OUTLIVES THE ORGANISM. The front's alternation belongs to the WASH
TRAJECTORY (the stream's gradient sign-structure turning over),
not to the fact's fleeing response; what the LIVING organism adds
is only the DEATH TIMING (T164: death = the rotation's deepest
landing on the support). THE PICTURE, FINAL FORM: the wash drives
a rotating lethal front; the fact dies when the rotation lands on
its support deepest; the bleed survives by turning its own steps
away from each landing; the rhythm (a managed bleed) survives by
re-injecting the right gradients at the right times. THE
OPENNESS: a shape claim at n=3; the mechanism candidates (why the
stream's gradient structure alternates) unnamed — the next
dissection question, ripening.


[R61-CRITIC PROVISIONAL STAMP ~08:55Z]: the alternation may be a
THEOREM OF SIGN DESCENT — on a locally quadratic landscape,
overshoot gives cos(u_t,u_{t+1}) = 1 - 2*f_flip (f_flip = the
fraction of coordinates with |g_i| <= h_i*s; f_flip ~ 0.58-0.68
reproduces every censused value with banal parameters), and the
model's absorption fingerprint (anti-correlation deepening along
the walk: -0.263 -> -0.353 committed) was read as "a rotating
object". "The rotation outlives the organism" = the period-2
bounce continuing at dead states — MANDATORY, not a discovery. THE
NULL DERIVATION DISPATCHED (a desk item, zero compute; the closed
form + the retrodiction + the registered falsifier: a fact-free
sign walk deviating toward the fact's presence would rescue the
information reading). THE SURVIVOR either way: "death = the
deepest landing" (one perfect 4-point rank-ordering, censored at
the event; the support-proxy overlap -0.030 — the support itself
never measured). T164/T166's nouns PROVISIONAL pending the null.


[NULL-DERIVATION RESOLUTION ~09:50Z]: THE ALTERNATION IS THE
ALGORITHM'S — the committed lag matrix carries the exact period-2
fingerprint of sign-descent overshoot (lag-1 -, lag-2 +, lag-3/4
~0); the nouns RETIRE (alternation / universality / outlives-the-
organism = the mandatory bounce); the critic's f_n~0 point also
refuted (lag-3 predicts +0.6..+0.84 vs ~0 observed — the flip
core is 8-28% on a 66-95% bath-redrawn majority). THE SURVIVORS
(different instruments, untouched): death-at-deepest-landing (the
4/4 rank-order) and the onset/arrival curves (D_kill objects).
ONE DEBT before the stamp: the step-size prediction has no
in-domain evidence — e202 (the fact-free twin + the ladder)
adjudicates; if FACT-IN-THE-FRONT fails to fire, the reading is
final: SIGN DESCENT BOUNCES, AS IT MUST.

## T165 — g1bS4: the dose question inverted — formation is non-monotonic in movement at 10M (2026-10-02 ~08:50Z)

Take 4 closes the dose question by inverting it: matching e113's
total movement made the channel WEAKER (0.2523 vs 0.6498 at a
third of the movement) — the 10M formation landscape is
NON-MONOTONIC in consolidation movement, with the optimum (if it
exists) sharp between 0.12 and 0.30 rms. THE CURE PATTERN'S LIMIT:
the one-knob licenses fixed stability (lr) and dose (steps) and
the formation still refuses to transfer — the e113 consolidation
FORM itself may not survive 10M (a form question, not a knob
question). THE LADDER'S NEW TEXTURE: at a weak root, the looser
balls hold WORSE (W3's late fade) — the wall's protection quality
tracks the ROOT's formation strength; and the walled arms IMPROVE
corpus CE (freezing False at 10M too). THE NAMED NEXT CUTS: the
formation-vs-movement curve (a consolidation-only dose sweep,
0.15/0.20/0.25 rms at 4e-4 — root reads only, no arms; small
cooled bursts under the owner envelope) or ACCEPT the e113 form's
10M ceiling as the finding. THE HONEST POSITION: four takes, one
inversion, three TEXTUREs — the wall's scale question has cost
patience and taught the recipe-stack lesson three ways; the sweep
is cheap and the curve is the dissection's instinct.

## T164 — e200: alternation — the front rotates onto and off the support, and death is the deepest landing (2026-10-02 ~08:30Z)

Given the longest alive window on record, the onset curve answers
NO to monotone deepening and YES to something better: the
concentration ARRIVES (t2), UN-FORMS (t3), and RETURNS AT ITS
DEEPEST at exactly the killing step (t4, ratio 0.397). The
geometry explains the shape: every consecutive front pair is
mutually ANTI-correlated (-0.26 to -0.35) — the front is a
ROTATING OBJECT that alternates onto and off the fleeing support,
and THE ORGANISM DIES WHEN THE ROTATION LANDS ON IT DEEPEST. THE
MECHANISM PICTURE REWRITES AGAIN: not a pursuit that converges
(e194's reading) but a ROTATION that periodically lands; survival
is the phase of the rotation relative to death. THE WHEN ACCOUNT
(T163) HOLDS (arrival at first concentration: org1 t1, MIRABEL t2,
org2-half t2); the SHAPE account amends: alternation, not
deepening. THE BLEED'S RE-ORIENTATION and the front's rotation are
THE SAME OBJECT SEEN TWICE: the bleed's steps turn away from the
lethal direction and live; the sign front's rotation periodically
lands on it and kills — both are the rotation's phase. THE
COUNTERFACTUAL CAVEAT rides honestly (this lineage lives only at
half the natural step); the census follow-on decides whether the
alternation is universal (org1/MIRABEL's committed fronts' mutual
correlations — a pure desk check on committed data).

## T163 — e199: the WHEN — the flight question closes at onset-shape (2026-10-01 ~23:00Z)

The timing cut returns the day's cleanest synthesis: ONSET-COMMON.
Both organisms' rotating fronts eventually concentrate from the
root — org1 at t=1 (0.171), MIRABEL at t=2 (0.427) — the same
onset curve shifted one step: THE CONCENTRATION IS A WHEN; THE
TIMING IS THE BIOGRAPHY. The e198 verdict's meaning flips without
its numbers changing: org-1-EARLY, not org-1-only. THE DROP-IN
TEXTURE completes the mechanism picture: MIRABEL's t=1 front is
SOFTER THAN STATIC before its t=2 lands on the lethal direction —
the rotation first points away (e194's anti-rotation read), then
ONTO the fleeing support: two movements, not one. THE ARC CLOSES
AT A SHAPE: every organism so far concentrates before it dies
(org1 t=1; MIRABEL t=2; org2 full-step never — it died at t=1
BEFORE its onset; its half-step lineage alive t=1..4 owes the
deepening test — the one curve that could fall across three alive
steps). THE FLIGHT STORY'S FINAL FORM: the lethal direction is a
ROTATING OBJECT the front tracks; the tracking has an onset (1-2
steps); the onset and the death race — where death wins first, no
concentration ever appears (org2's full-step, T158's alive-window
lesson, now with the timing account).


[E205 AMENDMENT ~11:15Z]: the cross-organism WHEN FALLS under the
common ruler (e205/T173: the arrivals dissolve or flip on the
in-span-band axis; the ordering was an artifact of self-normalized
ratios) — this card's claim rescopes to WITHIN-ORGANISM arrival
shape; the surviving cross-organism object is the EDGE-MULTIPLE
(the fact's margin over its own wash noise).

## T162 — e198: biography carries the flight — and the t=2 wrinkle reopens the timing question (2026-10-01 ~22:25Z)

The fork resolves to BIOGRAPHY: a fresh 2.74M root with org-1's
exact architecture, a terrain-replicating fact, and a live
mid-flight state shows NEITHER the t=1 flight concentration nor
the recomputation bonus — the architecture suspect is cleared.
THE FLIGHT QUESTION'S LEDGER: org1 (concentrates at t=1, ratio
0.17); org2 dead (absent); org2 alive (absent); MIRABEL alive
(absent at t=1 — BUT u2 concentrates at 0.43). THE T=2 WRINKLE IS
THE REOPENED DOOR: the rotation finds a lethal direction by t=2
here — the concentration may be a WHEN, not a WHETHER: the front
needs time (or steps) to rotate onto the fleeing support, and org1
did it in one step where MIRABEL needs two. THE TIMING CUT IS
NAMED (ripening): org1's own u2/u3 map (does its concentration
DEEPEN past t=1?) vs MIRABEL's t=3/t=4 — the concentration's
onset curve on both organisms. THE RIM PICTURE IS NOW UNIVERSAL
(2/2 architectures, 3/4 lineages (the fourth, org2-dead, has no live-anchor rim — its rises are dead-anchor reads; R61-audit repair): the recomputed direction always
improves the fact before the far crash) — the valley-with-rim is
the physics; the concentration's timing is the biography.
ALIGNMENT PREDICTS NOTHING (again) — the day's most repeated
negative. THE CANDIDATES: fact strength, draw, fact identity.

## T161 — g1bS3: the channel forms at 10M — the wall's near-miss with a new texture (the +2 dip and the above-root recovery) (2026-10-01 ~22:10Z)

The take-3 arc: the width-scaled license CURED the formation-kill
(the channel 650x take-2's) and the stability blowout, and missed
the express bar by a dose question now sharp enough to name — the
frozen 300 steps carried the width-scaled rate but a third of
e113's movement; s750 movement-matches. THE RECORD LADDER IS THE
REAL NEWS: for the first time at 10M, on the REGISTERED g-12
channel, the rms-dial produces the graded g1b-shaped response —
and with a NEW TEXTURE the 2.74M arc never showed: W1's +2 dip to
~0.0001 (a transient near-death, far deeper than 2.74M's dip)
followed by a recovery ABOVE ROOT and a flat 0.68-0.84 hold
through +300 at a TENTH of the reference tax (+0.055 vs +0.53).
THE SHAPE: at 10M the wall's first checkpoint sees the anchor's
formation shock, then the projection holds — the ball needs its
first moments to settle before it protects. THE SCALE CLAIM:
OPEN, instrument half-formed, one licensed knob from adjudication
(g1bS4: the movement-matched dose — the fourth take, the pattern
holding: diagnose, license one knob, re-run verbatim).

## T160 — e197: the honest fork — aliveness is not the carrier; the rim is universal, the concentration is not (2026-10-01 ~21:55Z)

The discriminating cell fired its honest fork: the alive window
opened (the half-step lineage lived t=1..t=4 exactly as designed)
and NEITHER effect formed — no recomputation bonus (the path SAFER
than its ray), no flight concentration (softer than the dead
lineage). THE ALIVE WINDOW IS NECESSARY BUT NOT SUFFICIENT. THE
MISSING CARRIER'S CANDIDATES, ranked by testability: (1)
ARCHITECTURE/SIZE — organism 1 is 2.74M 6L; this organism 873k 4L;
e193b's fresh root was 2.74M and its MIRABEL fact replicated the
whole TERRAIN — but the FLIGHT map was never read there: THE
DISCRIMINATING CUT IS NAMED (the flight map on e193b's MIRABEL
root: if the concentration appears, architecture carries it; if
not, organism 1's specific biography); (2) FACT STRENGTH (org1's
theta_1 read 0.679; this lineage's 0.419 — a strength threshold?);
(3) lineage biography (untestable except by draws). THE TEXTURE
THAT SAVES THE PICTURE: the rim-without-concentration — from the
alive theta_1, the static rays kill instantly while the walk's own
recomputed direction IMPROVES the fact before the far crash: THE
VALLEY-GEOMETRY IS UNIVERSAL (present in both organisms), the
LETHALITY-CONCENTRATION is organism 1's. The dissection's next cut
is already sharp.

## T159 — g1bS2: the third scale casualty and the cure pattern that generalizes (2026-10-01 ~21:10Z)

The wall-at-10x saga closes its second act honestly: the val-min-
anchored base license WORKED (the gate passed by construction —
T148's cure is proven as a pattern), but the verbatim e113
consolidation killed the registered channel (g-12 0.0010): THE
RECIPE STACK IS SCALE-BOUND ONE COMPONENT AT A TIME (base cosine;
base steps; consolidate lr). THE CURE PATTERN GENERALIZES WITH THE
DIAGNOSIS: each treatment gets its own width-scaled license
(val-min-anchored schedules; movement-matched doses); g1bS3 (the
4e-4 consolidation take) is named. THE CO-REPORT IS THE REAL NEWS:
on the g0 channel (never the registered ruler) the wall behaves
QUALITATIVELY AS AT 2.74M — C dead at +2, W1 flat ~0.5 through
+300, a graded response across the rms ladder — the wall itself
appears to TRANSLATE to 10x; what failed is the INSTRUMENT (the
g-12 channel the treatment killed in formation), not the
mechanism. THE ADAM-CLOCK AT SCALE: D_kill 3.146 = one step =
lr*sqrt(P) exactly — T139's arithmetic holds at 10M with all
mechanics bit-clean. THE HONEST LEDGER LINE: the wall's scale
claim stays OPEN-but-encouraging (a co-reported shape, not an
adjudicated bar); the recipe-stack lesson is itself a finding the
paper's discussion carries (treatments do not transfer across
host sizes without re-licensing — a small-scale lab's pipeline
discipline for the scaled world).

## T158 — e196: the flight is biography — and the reason found in the same cell: it needs a live mid-flight state (2026-10-01 ~20:30Z)

The replicate answers T157's follow-on NEGATIVELY with the
mechanism in hand: organism 2's flight ray is its SOFTEST
direction (ratio 2.58 vs organism 1's 0.17 — the opposite
direction), and the pre-registered asymmetry is the explanation —
organism 2's single step exceeds both its static edges, its walk
dies AT step 1, and its post-kill fronts are dead-state reads.
THE FLIGHT CONCENTRATION REQUIRES A LIVE MID-FLIGHT STATE: the
support cannot flee somewhere if it is already dead. The
corroboration is tight: where the mid-flight state is dead, the
recomputation bonus is absent too (the sign path dies exactly at
its static edge — no e194 inversion) — BOTH dynamic effects (the
pursuit and the flight) live in the alive window. T157 RESCOPES:
the flight-direction finding is organism-1 biography WITH a
mechanism hypothesis (the live-state condition); the discriminating
cell is named (a lineage alive past t=1 — smaller step or stronger
fact). THE PAPER'S DISCUSSION carries the conditional: dynamics
cut both ways WHERE THE ORGANISM IS ALIVE TO CUT; past the kill,
the terrain is static again.

## T157 — e195: the flight direction is the killer — direction-of-flight beats current-alignment (2026-10-01 ~19:20Z)

FLEEING-IS-LETHAL, decisively: the rotated ray kills at 0.39 from
the root where the original kills at 2.27 — an 83% concentration
of lethality into the direction the support fled toward. THE
DISSOCIATION IS THE DAY'S DEEPEST TWIST: the flight ray is
ANTI-aligned with the root's fact gradient (cos -0.067) yet
deadliest; the aligned ray kills 6x later. The killer direction is
not where the death gradient points NOW — it is where the fact's
support is GOING. Static alignment readings (the whole day's
terrain program!) measure the wrong thing unless they state their
point AND the state's history: the lethal direction is a property
of the TRAJECTORY (which way the support moves under this wash),
not of the landscape alone. THE BOTH-TRUTHS READING: the flight
ray is absolutely lethal from the root AND the panel reverses
from theta_1 — the terrain concentrates AND rotates; e192's
order (measured on t=0 rays) stands as the root-panel biography.
THE VALLEY-WITH-RIM picture: the fleeing support lands in a basin
whose near rim pumps (short -u1 jumps from theta_1 IMPROVE the
fact 0.68 -> 0.82 before the crash) — the flight is not toward
death but THROUGH a rim into a valley whose far wall kills. THE
FOLLOW-ONS: the valley's width; the second organism's flight
direction (does e193's replicate line flee the same way?).


[E196 AMENDMENT ~20:30Z]: the flight concentration is ORGANISM-1
BIOGRAPHY — organism 2's flight ray is its softest (ratio 2.58 vs
0.17) — WITH THE MECHANISM: its walk dies at step 1 (dead
mid-flight state; theta_1 0.0068 vs org1's 0.679); the
recomputation bonus is absent there too. The flight and the
pursuit both require the alive window. Rescopes this card's
universality claims.

## T156 — e194: the lethal subspace flees and the fresh front pursues — one recomputation, the whole bonus (2026-10-01 ~19:25Z)

GRADED with the mechanism convicted anyway: NOT chase (the frozen
frame), NOT accumulation, NOT artifact — the true reading is
ROTATION-TO-FLEEING-SUPPORT. The fine static edge is 2.2699 (the
2.5 was coarse-grid); the recomputed path kills at 1.75 — a real
23% inversion — and the k-ladder is a STEP FUNCTION: k=1 at 1.75,
k>=2 at the static edge: ONE RE-COMPUTATION IS THE WHOLE BONUS.
The discriminating pair (T150's estimator lesson earning its keep
again): the front's frozen-ray overlap collapses while its
matched-point alignment with the FACT'S OWN GRADIENT rises —
the lethal subspace MOVES WITH THE STATE and the fresh front
follows it. THE SYMMETRY WITH THE BLEED completes the day's
picture: the re-orienting walk's steps rotate AWAY from the
lethal direction and spare; the sign path's steps re-computed
TOWARD the fleeing lethal direction and kill sooner — dynamics
cut both ways, now measured in both directions, both at n>=2
anchors. THE SUBLINEAR EFFICIENCY (0.919 — worse than a random
walk at accumulating displacement) says the path SPENDS its
budget on rotation, not advance: lethality is not borrowed
steepestness. FOLLOW-ON: the rotated-ray terrain names where the
support fled TO; the discussion's mechanism paragraph writes
itself from this cell.

## T155 — e193b: the order is physics, the distances are biography — and the ridge/cliff thesis decided inside one organism (2026-10-01 ~19:00Z)

The critic's replicate lands the cleanest generality statement of
the program: THE RAY ORDER IS DRAW-FACT-ARCHITECTURE-ROBUST (3/3
organisms, 2/2 facts, architecture pinned and lineage varied);
THE ABSOLUTE KILL DISTANCES ARE THE FACT'S STRENGTH BIOGRAPHY
(MIRABEL inside both windows; ZEPHYRA — ruler-dead at D=0 on the
imported ruler — holds the order at >2x-down distances). THE
RIDGE/CLIFF SPLIT IS DECIDED WITHOUT CONFOUND: the same organism,
the same rays, two facts — ZEPHYRA pumps, MIRABEL does not —
T153's thesis confirmed: the ridge is consolidation-alignment
biography; the cliff is physics. THE IN-SPAN SPREAD IS THE NORM
(3.0x/2.0x here; e131's suppressed spread replicated) — the
subspace's lethality is direction-heterogeneous everywhere
measured. THE BREAKER'S MIXED VERDICT refines the lethal front:
support+signs suffice (magnitude-shuffle kills at 1.68x) but
magnitude-pairing contributes — the front is support > signs >
magnitudes in necessity order. THE SIGN RUNG's sensitivity is now
triply-observed (f1 1.90, e193 2.63, here 2.24) — the sign edge
is the terrain's softest number; the paper's clause carries the
range. THE RULER BIOGRAPHY LESSON (second occurrence): a fact can
be alive on its own ruler and dead on the imported one — rulers
are facts' property too.

## T154 — g2g2: the autonomy splits — the organ is a better selector than scheduler (2026-10-01 ~18:40Z)

The seed ladder licenses the number and decomposes it in the same
run. THE BAR: 3/3 seeds positive, median +0.0333, cleared by 11% —
"worth" returns to the paper's clause WITH the seed scope and the
barely-cleared honesty. THE DECOMPOSITION (the paired control's
failed prediction is the finding): organ-vs-paired +0.0130 vs
paired-vs-fixed +0.0558 — the margin is MAJORITY REPLAY-BATCH
COMPOSITION, MINORITY TIMING. The organ's value lives in WHAT it
replays (the cue pool's fact-relevant draws — the g2 design's
original core) more than in WHEN it fires (the monitor's
thresholding — the later addition). W026'S MANAGED-BLEED NOUN
REFINES: the re-orientation schedule's small timing premium
(+0.013, within seed spread) rides a larger selection premium
(+0.056; the right gradients injected, not just any re-
orientation). THE DEVICE LESSON: CPU-vs-GPU moves ~0.007 on this
instrument — the g2g 2x leg's float-fragility was of this size;
the bit-exact reproduction of g2c's realization (10/10 events)
anchors the lineage. THE HONEST PICTURE OF THE ORGAN: a cue-pool
selector with a thermostat bolted on — the selector earns the
keep; the thermostat earns a little; the ceiling (T152) stands.

## T153 — e193: the order is lineage-physics; the pump is biography — the replicate's clean split (2026-10-01 ~17:30Z)

The replicate splits the day's central objects by generality.
WHAT CROSSES LINEAGES: the terrain's ORDER (g < sign < random,
2.9x and >4.0 at n=2 — Fig-5's caption now true at two organisms);
the front's magnitude-informative cluster (topk/raw ratio-tight,
extending to the 1% rung with neither floor nor shrink); the
re-orientation causality (the pinned walk dies at the static cliff
on both organisms — Fig-5's interventional sentence at n=2); the
CE_R canary ordering. WHAT DOES NOT: THE PUMP — organism 2's g-ray
FALLS immediately (every small-D rise negative on every ruler) —
the fact-positive ridge is one organism's biography, not the
physics; C5 rescopes. THE SIGN RUNG DRIFTS WIDE (2.63 vs the
[1.43,2.38] band): the front's sign-normalized edge is
lineage-sensitive where its magnitude cluster is not. THE THINK-
CARD THE AGENT OWED (adopted here): the ridge and the cliff may
not be the same object — the ridge is fact-positivity the wash can
harvest (present where the consolidation left the fact gradient-
aligned with the wash's useful directions); the cliff is the
direction the fact cannot survive (universal); organism 2's fact
consolidated WITHOUT the alignment, so no ridge, same cliff. THE
DISCLOSED SLIP: architecture co-varies with lineage (the family-2
root is 873k 4L, not the "same class" as e131's 2.74M 6L) — the
replicate doubles lineage, not architecture-at-fixed-lineage;
e193b (the fresh-root/two-fact cell) can pin the axis. THE RULER
LESSON: the e192-verbatim battery read 0.198 at the f2 ROOT —
under the kill bar at D=0 — a fact can be alive for its ruler and
dead for an imported one; co-rulers everywhere, adjudicated on
none.

## T152 — g2g: a thermostat on a leash — the rhythm's honest operating envelope (2026-10-01 ~16:55Z)

The controls land as three teaches. (1) THE ORGAN IS
threat-responsive but WEAKLY: 10 -> 12 events across 8x threat,
ceiling-saturated — at high threat every spacing pins to the
refractory floor and maintenance collapses (0.62 -> 0.002): a
thermostat on a leash, firing without maintaining. The autonomy is
real but bounded by the gate's fixed thresholds. (2) THE HEAD-TO-
HEAD RESOLVES AT 1x: the organ beats the matched-count fixed
schedule by +0.07 in both runs on the registered same-ruler
comparison — the critic's 0.693 co-read was endpoint-vs-median and
is superseded — but the 2x leg rides one float-nondeterministic
check, so FIXED-MATCHES-OR-WINS is the safer letter with the 1x
win co-reported. AUTONOMY IS WORTH +0.07 OF CYCLE-MEDIAN AT
OPERATING THREAT — a small, real, priced number. (3) THE BAND WAS
PARTLY A FLOOR: at refractory 8 five spacings land below 20 — the
"100% in 20-45" construction disclosed — yet the wash's decay
clock still shapes 9/14 intervals AND maintenance IMPROVES at the
shorter refractory (0.671, duty 0.74): the refractory is a TUNABLE,
and shorter is better. THE W023 RIDER: the mini-shock lives in the
monitor channel (jump, brief rise, accelerating decay — the
replay's protective mismatch wearing off) even though the
alignment form died; at 4x the replay's step is as big as the wash
and the signal pins to zero. THE CLAIM'S FINAL FORM THIS CELL:
"threat-responsive within [0.5x, 2x]; autonomy worth +0.07 at
operating threat; refractory-tunable" — and the seed ladder is
licensed. W026's managed-bleed noun gains its price tag: the
re-orientation schedule is worth +0.07 over a fixed schedule, at
the cost of a ceiling.


[G2G2 AMENDMENT ~18:40Z]: the +0.07 is licensed at n=3 seeds
(median +0.033, barely cleared) AND DECOMPOSED: majority
replay-batch composition (+0.056; the cue-pool selector), minority
timing (+0.013; the thermostat). The "autonomy" noun splits into
selection + scheduling; selection wins.

## T151 — opt2: the lethal object is the |g|-weighted front — density exonerated, magnitude-information convicted (2026-10-01 ~16:15Z)

The density ladder answers the optimizer arc's last open question
with digit-level cleanliness (TOPK-50% == raw at 0.9203): DENSITY
CARRIES NOTHING — a tenth of the coordinates, |g|-selected and
magnitude-weighted, carry the whole kill. WHAT ORDERS THE KILL IS
MAGNITUDE INFORMATION, now interventional at matched size: raw
0.92 -> flattened sign 1.75 -> Adam's warmed sign-path 2.49. THE
SIGN PATH KILLS BELOW ITS OWN STATIC EDGE: the re-computed front
beats the frozen ray (1.75 vs 2.5) — the trajectory is MORE lethal
than its shadow, the mirror of the bleed (whose re-orientation
SPARES what its ray kills): DYNAMICS CUT BOTH WAYS, and the
interesting objects are the two mismatches (the bleed's protective
re-orientation; the sign path's lethal re-computation). WITH THE
CHART: the kill lives in the coordinate-magnitude pairing
(the shuffled sign is inert), concentrated in the top tenth —
a sharp, small, nameable object: THE LETHAL FRONT. The estimator
lesson is now anchored at both ends on trajectories. The arc's
shape: opt1 asked CLOCK-vs-GATE; opt1c split direction from size;
e191/e192 mapped the terrain; the chart killed two pictures; opt2
names the weapon. FOLLOW-ONS: the 1% rung; the sign-front
mechanism. HONESTY: n=1, CPU fp32, the canned-phrase caution
adopted (the number, not the frozen text, is the reading).

## T150 — the chart: no cuts, no flip — the corrections corrected (2026-10-01 ~16:00Z)

THE CHART CELL corrects the correctors. (1) THE PUMP HAS NO CUTS:
every magnitude class of the wash gradient is fact-positive at
t=0 — W024's stitches-and-cuts dies whole (there are no tiny
cuts; the erosion is not a sign-flattened mass of small
coordinates). (2) THE SIGN-FLIP WAS AN ESTIMATOR-POINT ARTIFACT:
at a MATCHED point, normalization ATTENUATES fact-relevance
(+0.0986 -> +0.0396, ~2.5x) but does not flip it; opt1's -0.0385
trajectory read is real but POST-STEP (it evaluates the fact
gradient after the 1.6543-L2 step has already moved it). T139's
adopted gem and the paper's clause correct to "the normalizer
ATTENUATES the stream's fact-relevance; the trajectory-level
negative read is the post-step view". (3) THE SPAN KILLS AT 1x:
in-span random directions kill BELOW the g-ray itself (0.56-0.61
on e131) — the empirical gradient-history span is lethally
sufficient, stronger than the subspace hypothesis needed; but the
out-span arm's split behavior (rung 8 on g3 vs rung-1-with-59%-
retained on e131) and the non-reconciling dimension estimates
(kappa >=52k vs SVD rank 20) leave W025's projection-ratio account
UNCONFIRMED — the span is real and lethal; its size is not yet
measurable by these instruments. (4) THE SHUFFLED SIGN IS INERT:
exact coordinate-sign pairing owns both the pump and the sign
kill — gradient structure is in the PAIRING, the sharpest form
the structure question has taken. (5) THE RANDOM BAND >12: the
isotropic arm of the terrain widens another 3x (Fig-5 updates).
THE META: the chart was built to check two wonder cards and it
killed both pictures while confirming both questions were worth
asking — the favorite-dies-well pattern, twice in one cell.


[R60-CRITIC REPAIRS ~17:15Z]: (a) the d_eff BOUND WAS FLIPPED in
this fold — kappa >= 7.25 implies d_eff <= 52k, not >=; corrected.
(b) The "SVD rank 20" saturates its instrument (only 20 history
vectors exist — a cap, not a measurement); the kappa denominator
is an unregistered convention swinging d_eff 8x — the
"non-reconciliation" is convention-plus-cap, not physics. (c)
SUPPRESSED DISCLOSURE restored: in-span seed 11602 is ALIVE (0.66)
at rung 1 where siblings die at ~1e-4 — a ~3x threshold spread
inside the primary organism; the in-span arm's lethality is
seed-heterogeneous. (d) The un-run breaker named: magnitude-shuffle
within the front (keep top-10% support and signs, permute |g|) —
sign-pairing was convicted by intervention; magnitude-pairing is
asserted, never intervened on (queued for e193b/e194's rider).

## T149 — e182c: the surgical signature dies at 124M — generic forgetting, and the template-locus hint (2026-10-01 ~15:40Z)

FORGETTING-GENERIC fired at both depths: matched held-out controls
erode WITH the installed fact (ratio 0.92 at +80; 0.83 at +50) under
the bit-tight replayed wash, while perplexity improves throughout.
The supervisor's objection — carried across nine check-ins —
resolves against the lab's own claim: THE SURGICAL SIGNATURE
RETIREES AT 124M. What survives of T123: the direction-only
transfer (the corrected title) and the time-constant texture; what
dies: "no-basin signature" as a GPT-2 claim. The honest GPT-2
clause: "ordinary forgetting with improving perplexity" — itself a
nontrivial texture (adaptation and erosion co-occur), but not the
tiny-nets' law. THE TEMPLATE-LOCUS HINT is phase-2's gift: the
near-related battery (same cloze template, disjoint entities)
collapses FASTEST (0.234) — erosion may live at the few-shot-
following/template level rather than knowledge storage; and the
controls' heterogeneity (founders hold 0.78-0.94, products collapse
0.18-0.31) says probe-TYPE structures the forgetting. THE META-
NOTES: four dispatches died for this cell; the lean brief (smoke
first, commit at every stage) got it home — the disruption era's
dispatch pattern. And the replay-that-was-necessary incidentally
discharged the CPU/GPU numerics debt (bit-tight). T123's amendment
follows; the paper's GPT-2 clause rewrites in the fold.

## T148 — g1bS: the recipe is scale-bound — the honest negative that saves the cell (2026-10-01 ~15:15Z)

The hard stop fired exactly as registered: the 10M base failed
G-BASE-QUAL (val rising past its s1113 minimum to 2.85 by the
cosine's end; coherence alone PASSING — the memorization-recitation
signature). NO ARMS RAN — the wall's scale question is OPEN, and
the negative is itself the scale lesson: THE HOUSE RECIPE DOES NOT
TRANSFER. Both licensed lrs U-turn together (minima 1.568/1.577 at
s~1000-1113) — capacity/corpus-driven at 10M params on ~1M chars;
the 4000-step cosine minted at 0.87M overtrains ~3x past the val
min; width-scaling the lr slows memorization but cannot prevent the
turn. THE POLICY EXTENSION (R58's no-hardcoded-constants, one
step further): recipes are SCALE-BOUND — steps AND lr re-register
per host size, anchored at the val minimum. THE INSTRUMENT ECHO
(W021): coherence was the instrument that could not fail — it
would have licensed a memorizing host; the val-decreasing clause
was the one that could. g1bS2 LICENSED: val-min-anchored cosine
(~s1150, 4e-4, same corpus), then the frozen wall cell verbatim —
R_rms ladder, bars, and wash untouched. The wall question costs one
more GPU hour, not a redesign.

## T147 — opt1b3: the walk that will not die — the third projection falsified, the grind named (2026-10-01 ~14:35Z)

CAP-AGAIN, graded honestly: 150 every-step reads, no kill (min
0.2891 at the final step), no 2.6 crossing (final D 2.2501). THE
ARITHMETIC LESSON COMPOUNDS: opt1b2's kill ~s1214 was walked
through ALIVE — the third linear projection this arc has falsified
(opt1b's D~10, opt1b2's s1214, now the same class again); W021's
family of arithmetic-model instruments grows a dedicated shelf.
THE GRIND (the named texture): the stall is not equilibrium but
SLOW EROSION (0.33 -> 0.29 at -2e-4/step) on a 9.26x-sublinear
drift (D(t) linear r2 0.9996 — steady, deeply sublinear): the
bleed never dies within any cap we have set, yet never stops
grinding; its asymptote is UNRESOLVED and now unmeasured-by-design
(three falsified projections means the class is retired, not
retried). THE GATE QUESTION'S HONEST STATE: the walk's own kill-D
is unknown; the CAUSAL axis (why the walk is spared where rays
die) is answered independently by e192's rider — orientation owns
the sparing. THE FOUR-CLASS OVERLAY stands complete for Fig-5's
companion: guillotine ~2.5, annihilation 0.92, random >4.0 flat,
and the grind — three ways to die and one way to erode. THE
ABSTRACT'S BRACKET FILLS: the diffusive walk "grinds below the
ring without dying (asymptote unmeasured)". Savoring: the bleed
is the arc's honest ending — not saved, not dead; grinding.

## T146 — e192: the terrain licensed as one picture; re-orientation causal; the pump tracks structure (2026-10-01 ~13:50Z)

Both primary bars fired. THE MAP: on one organism, one ruler, dual
currency — g-ray 0.92 < static sign(g0) 2.5 < Gaussian >4.0 (alive,
flat): the three-band figure stands on mapped ground, and the
middle band's stitch objection DIED by measurement (A0's step-1
read sits ON the static sign curve). THE ISOTROPIC ARM'S FLATNESS
is its own finding: 4x the g-ray's kill displacement leaves the
fact untouched — W025's projection ratio at this organism >= 4.35x,
and the e190 subspace test now has a sharp target. THE RIDER IS THE
DAY'S CAUSAL ANCHOR: pinned small steps (orientation denied, size
at the bleed's scale) die at the static edge while the bleed lives
at the same D — RE-ORIENTATION OWNS THE SPARING; step size owns
nothing. W026's managed-bleed noun graduates from poetry to
interventional mechanism (the R59 critic's one converting cell,
paid). THE PUMP TRACKS GRADIENT STRUCTURE: g pumps (+0.045), sign(g)
pumps (+0.037), isotropic is inert (+0.0002) — the pump is not
displacement, not magnitude, but STRUCTURE; W024's census question
sharpens to "what do g and sign(g) share that isotropic lacks" (the
sign pattern itself?). CAVEATS carried: n=1; random band
unresolved-high (the wider grid is a rider for the chart cell, not
a new dispatch); the sign ray is static, not Adam's adaptive path
(the path-vs-ray gap for the SIGN class remains e191-style
disclosed).

## T145 — opt1b2: the bleed is a diffusive walk — the trajectory-class axis is BALLISTIC vs DIFFUSIVE (2026-10-01 ~08:50Z)

CAP-NEITHER with a bonus falsification: the bleed entered the kill
window's lower margin ALIVE (0.324 at D 2.12) and STALLED at 0.329/
D 2.1455 — and the projection that said "kill at D~10" was wrong as
arithmetic: it assumed colinear steps, but consecutive raw
gradients are far from colinear; per-step norm 0.0072 buys only
0.0012 of displacement (5.9x sublinear, decelerating). THE BLEED IS
A DIFFUSIVE WALK. THE TRAJECTORY-CLASS AXIS RENAMES ITSELF:
ballistic-maximal (the annihilation: one huge colinear step, kill at
D 0.92), ballistic-normalized (the guillotine: Adam's ~2 colinear
sign-steps, kill at D ~2.5), and DIFFUSIVE (the bleed: a random-walk
in gradient space whose displacement grows ~sqrt-ish, stalling at
the window's edge ~D 2.1-2.2 at 0.33). THE DECIDING DOOR: the
last-slope extrapolation puts the kill ~14 steps past the cap —
opt1b3 (dispatched, tens of steps) reads the bleed's own kill-D
directly: ~2.2-2.6 reunifies the gate (all classes die near the
same ring, protection = staying diffusive/slow); >>2.6 keeps the
classes separate (each has its own ring). W026's managed-bleed noun
upgrades: the rhythm's candidate protection is KEEPING THE WALK
DIFFUSIVE (replay events re-randomize the step directions); e192's
pinned-ray rider tests the same axis interventionally (a pinned walk
is forced-ballistic — dies at the static cliff iff direction-
randomness, not step size, is the protection). The projection
lesson joins W021's family: an arithmetic model is an instrument —
this one was falsified by the measurement it motivated.

## T144 — e191: the cliff is terrain — the first mapped ground of the forgetting machine (2026-10-01 ~08:05Z)

STATIC-CLIFF fires: the graded static profile along the raw-gradient
ray matches the dynamic kill — pump ridge (0.94-0.96 across D
0.05-0.50, peak 0.9605 at 0.20), cliff edge [0.80, 0.92], dead at
0.92, floor at 2.0. NO OVERSHOOT IS NEEDED: the kill is geometry.
THE BLEED OVERLAY IS THE FIGURE: at D 0.92 the straight ray reads
0.248-dead while the re-orienting path reads ~0.83-alive —
protection = re-orientation off the ray (W026's managed-bleed
mechanism now has its figure). [R59-CRITIC STAMP ~08:30Z: the
three-terrains FIGURE is UNLICENSED as one picture — the sign-ray
2.5 is a cumulative PATH LENGTH (no static sign ray was ever mapped
on e131); the random band is g3K's organism in a different ruler
and currency; e192 (DONE 13:50Z): LICENSED — the one-organism map: 0.92 < 2.5 < >4.0, A0's step-1 read ON the static sign curve; re-orientation causal via the rider.] THE LETHALITY ORDERING STANDS ON
MAPPED GROUND: the raw-gradient ray's terrain kills at 0.92; the
sign-normalized ray's at ~2.5 (opt1); random rays' at 4-10x (g3K) —
three terrains of increasing width. CE_R: the fact dies at organism
CE 3.0 — the fact is the canary, not the casualty of general
wreck (the organism wrecks further out, 4.8-5.3). DISCLOSED: on the
single-step interval static = dynamic by construction; the
independent content is the recompute, the extension, the tighter
edge, the CE_R profile. THE SURVIVING SPLIT: opt1b2 (running) —
where does the re-orienting path itself die? If it survives past
every static kill ring, the final law's protective principle is
re-orientation alone and the walls/cages are one family of many.

## T143 — opt1c: the third outcome — the raw gradient is the most lethal direction, and the pump-cliff is the terrain (2026-09-30 ~07:55Z)

The factorial's answer was a branch neither name covered (the map
still served: it forced the graded-outcome reading, no bar
shopping). KILL-OUT-OF-WINDOW: the raw-gradient direction at Adam's
step size kills at D ~ 0.920 — 2.7x BELOW the Adam gate, inside the
first step. THE THREE CLASSES AT MATCHED D 1.6543: the guillotine
(Adam's sign direction: 0.678, organism shocked), the ANNIHILATION
(raw gradient at full size: 0.0007, organism devastated), the bleed
(small steps: 0.79, organism intact). LETHALITY PER DISPLACEMENT
ORDERS: raw-gradient > sign-normalized > random (4-10x) — W025
REFINES: not merely "in the subspace" — the direction's projected
effectiveness on the fact's sensitive structure orders the lethality;
the raw gradient IS the steepest effective direction, sign(g) its
flattened shadow (W024's stitches again — flattening LOSES some
lethality, it does not add it). THE PUMP-CLIFF GEOMETRY: the fact
pumps to 0.955 at D ~ 0.33 then cliffs by D 1.0 — a local ridge then
a cliff in the g-direction; the pump is a REGION property (the bleed
pumped at the same D on tiny steps), and the bleed's protection is
RE-ORIENTATION: each tiny step recomputes the gradient and the path
curves around the cliff the big step overshoots. e188's RAW-WINS
RESTATES as within-class invariance across lr along the Adam class.
THE OPEN SPLITS (both dispatched): geometry-vs-dynamics (e191:
static single g-jumps at graded D — if the static profile matches
the dynamic cliff, the terrain is real; if the static jump at 0.92
spares, the kill is overshoot) and the bleed's own crossing
(opt1b2: it passed 0.92 alive at 0.79 — where does the re-orienting
path die? the projection said ~10; the cliff says closer).


[R60-CRITIC AMENDMENT ~17:15Z]: "the raw gradient is the most
lethal direction" is CONTRADICTED un-amended by the chart's in-span
arm (sampled span directions kill at 0.56-0.61, BELOW the g-ray's
0.92) — the correct sentence: the g-ray is the most lethal of the
FIVE RAYS SAMPLED; the span contains directions more lethal still;
three rays are not a map. TERRAIN language scopes to biography-of-
rays until e193/e193b replicate or scramble the order.

## T142 — opt1b: CAP-NEITHER honestly — Adam is the guillotine, SGD is the bleed; the gate question moves to opt1c/opt1b2 (2026-09-30 ~12:25Z)

The direct SGD kill honors its frozen cap: at 600 steps the fact is
ALIVE (0.601) at D 1.43 — the D=2.6 gate unreached; neither bar
fired. The labeled projection reads the crossing at ~step 761 with
g-12 ~0.56 ALIVE and the kill at ~step 1829 / D ~10 — OUTSIDE the
[2.12, 3.27] bracket: if it holds, the gate is TRAJECTORY-CLASS-
TYPED (raw-gradient paths kill at ~4x the Adam gate) — but the
projection never adjudicates; opt1b2 (registered: cap ~800, the
crossing read directly) owns it. THE MEASURED GEM: at matched D
1.65 — Adam's first-step displacement — SGD holds 0.79 vs Adam's
0.678: at EQUAL raw displacement the raw-gradient path preserves
more than the sign-normalized path. TEXTURE VOCABULARY (the
trajectory classes get names): ADAM IS THE GUILLOTINE (dead in ~2
steps, any stream, organism shocked); SGD IS THE BLEED (pump to
0.955 at D 0.32, then slow monotone erosion with the organism
nearly unharmed — CE_R 1.66 -> 1.72). The pump-then-erode shape is
the bleed's signature. ALIGNMENT: flat-negative, slightly LESS
death-directed as D grows — RAW-WINS consistent (e188). FREE N=2
determinism (the registered run bit-reproduced an accidental
full-depth shakedown). WITH e188: the picture is now — the GATE is
raw displacement for sign-normalized paths; the raw-gradient class
may carry its own (larger) gate; opt1c splits direction from size
inside the Adam kill; opt1b2 catches the bleed at the crossing.

## T141 — e188: RAW-WINS — death is priced in raw displacement; alignment is a passenger (2026-09-30 ~12:05Z)

The death-currency cell answers W022b's fork AGAINST its own
favorite, exactly per pre-registered Branch B (scratch/
e188_interp_prereg.md, written before the fold read the numbers).
CV(D at death) 0.233 vs CV(A) 0.557 across the lr grid; neutral-only
0.026 vs 0.697; and the cleanest texture: same arm, same t*=2,
three wash seeds — D at death {2.489, 2.484, 2.500}, 0.6% SPREAD,
while A spreads 3.7x. DISPLACEMENT IS THE INVARIANT; ALIGNMENT IS A
SEED LOTTERY. DIES: W022b's aligned-drift law (in its letter);
W022's rate-law mechanism candidate (the rate law keeps its
displacement reading); the framing slogan "forgetting is aligned
training" (stamped once by C13-1, now killed by EVIDENCE — the
honest arc). STRENGTHENS: T139's displacement GATE (the tightest
invariance the wash arc has produced). REDUCES: T137 to the
static/learned contrast (g3K's kappas; no integral needed).
POIGNANT: the fast arm's alignment flips POSITIVE post-kill — the
organism's adaptation walks toward the fact readout's ascent
direction; it returns to the grave it dug. VOCABULARY REJECTED:
install-vs-wash cos in [-0.034, -0.018] everywhere — the wash is
the corpus's adaptation direction, not the fact's negation
(T138's defence resolves by measurement). the census survives as the mechanism of the ATTENUATION (W024's flip corrected to attenuation by the chart), not the currency.

## T140 — g1bW: the museum test's honest split — A survives an active second install; the tax relocates to the onset channel (2026-09-30 ~11:50Z)

The killer control lands with an ambiguity that is itself the
finding. AS REGISTERED, MUSEUM fired (walled B 0.0736 <= 0.27 at
healthy CE 1.63) — but the UNWALLED reference failed the B-ruler too
(0.0628): at the critic's 300-step dose B installs NOWHERE, so "the
wall is a splint" is NOT contrast-licensed. WHAT THE WALL ACTUALLY
DID: (1) A held 0.83 (min 0.65) through an ACTIVE second-install
attempt — the wall's strongest positive yet: protection survives
interference, not just passive wash; free-run ZEPHYRA survived the
ordeal. (2) The measurable tax relocated to the ONSET channel: B's
trained-length partial form peaked 0.21 walled vs 0.53 unwalled,
while A's row0 stayed protected (0.62-0.69 vs 0.006). THE COHERENT
PICTURE: the wall protects the committed manifold and resists
leaving it — T133's battery-channel scope and this onset-tax are ONE
mechanism: inside-the-ball protected, outside-the-ball resisted.
THE DISCRIMINATOR IS DOSE (g1bW2 queued): B at 600+ steps unwashed —
if the ruler form arrives, rerun the walled contrast at that
operating point; the museum question adjudicated where B can install.
HONESTY: n=1 lineage, one wash seed, B-draw n=1; the concurrent
draft's F2 (fabricated reference constants) joins W021's scan — a
would-be instrument corruption caught only by verify-before-run.

[R58 AMENDMENT — the critic's 2c wording adopted]: "MUSEUM fired as
registered (walled B 0.074 <= 0.27 at healthy CE 1.63). Its
registered rescope is WITHHELD: the paired unwalled reference failed
the same B-ruler (0.063) — at the 300-step dose B installs nowhere in
this lineage (partial onset form only, g0 peak 0.53 unwalled), so
B's failure inside the wall is uninformative about the wall. The
museum question is OPEN pending the dose control (g1bW2, registered
as a dose LADDER: the reference's non-monotonicity 0.53->0.49 says
the dose sat near a form-transition; a point at 600 steps may
overshoot). Licensed today: (1) the wall held A (min 0.65) through a
300-step second-install attempt — protection survives interference;
(2) THE ONSET-TAX is the day's cleanest paired read (B g0 peak 0.21
walled vs 0.53 unwalled, bit-identical inputs, 300/300) and the one
contrast-licensed claim — the paper leads with it. 'Survives an
ACTIVE second install' softens: an antagonist that installed nothing
is a weak antagonist until g1bW2 supplies a dose at which B presses.
The metrics' adjudication clause contradicts this reading (calls
0.0628 an 'install' and asserts the rescope) — the file stays frozen
as the machine record; this amendment is the interpretive record.

## T139 — opt1: the clock is Adam's arithmetic; the gate is displacement; the "killer" stream teaches under SGD (2026-09-30 ~11:30Z)

The nine-times-asked control lands as a decomposition of the wash
kill into CLOCK and GATE. THE CLOCK: step-1 pre-clip grad norm 0.9829
in every arm; AdamW moves 1.6543/step (lr*sqrt(N), sign-normalized),
matched-lr SGD 0.0010 — 1683x at the same lr. THE GATE: every Adam
variant kills at D ~ 2.49-2.84; warmup stretched the clock 10.08x
(the registered ADAM-AMPLIFIES fire) and the kill still arrived at
the same displacement within ~15%. THE BOMBSHELL: matched-lr SGD ran
the SAME corpus wash with the fact RISING (g-12 0.916 -> 0.940-0.955)
— the raw stream gradient is weakly fact-POSITIVE at small
displacement; the kill is Adam's sign-normalization exiting the
basin at full speed. A fresh AdamW's first step is +/- lr on every
coordinate; the basin (~2.5 L2) meets per-step 1.65 and dies in ~1.6
steps — the stream chooses signs, the normalizer chooses the clock.
THE TRAJECTORY HYPOTHESIS (T137) REFINES: never "any learned path
kills" — it is "any path that REACHES the gate kills; Adam reaches
it in ~1.6 steps BY CONSTRUCTION; static jumps need 4-10x (g3K);
SGD's slow path is fact-positive in-window." THE OWED DISCRIMINATOR
(opt1b, dispatched): the direct measured SGD kill at lr 1e-2 — if
SGD dies at D ~ 2.5 the displacement gate generalizes across
trajectory classes; if SGD reaches D = 2.5 ALIVE, the gate is
trajectory-class-typed (sign-normalized vs raw-gradient paths
differ) and the law gains a third clause. W023 stands untested (the
summary alignment read is flat-negative; the per-arm curves decide).
beta2, inherited moments: nothing (A4 indistinguishable; A5
bit-identical). HONESTY: n=1 per arm; SGD clocks only by labeled
projection; CPU fp32 texture gated on-device.

[R58 AMENDMENT — three corrections, all evidence-backed]:
(a) "THE STREAM TEACHES UNDER SGD" is retired: the fact-strengthening
is a SMALL-DISPLACEMENT PUMP, not an optimizer property — A3
(AdamW+warmup) pumped to 0.9476 at D=0.368 before dying; e184 pumped
+0.026 in one full-lr AdamW step; SGD lingers in the pump at
1/1683rd the speed. The genuinely new SGD-specific fact (was a
co-read): on the same bit-identical batch Adam's step reads
cos(delta, grad m12) = -0.0385 vs SGD's +0.0981 — THE NORMALIZER
FLIPS THE SIGN OF FACT-RELEVANCE; the slogan inverts: the normalizer
chooses the sign, the stream supplies the gradient.
(b) "Inherited moments carry nothing" WITHDRAWN as tautological —
A0's wash already starts fresh-state, so A5 was null by construction;
inherited moments are UNTESTED (opt2's WARMV arm owns the question).
(c) The decomposition joins T137 under PROPOSED: D_kill 2.4893 was
imported from A0's own kill (the registered bar was circular for A0;
identity, not prediction); the only independent test (A3) is
bracketed [1.45, 3.64] at 10-step resolution — the honest
displacement statement is two-convention (checkpoint 2.49 vs 3.64;
interpolated 2.18 vs 2.85; opt1b's gate reads the interpolated
convention); and the e131 lineage has no static-jump leg — the
forgetting law is a two-organism stitch at n=1 each. opt1b (running)
+ opt1c (registered) + e188 (running) are the adjudicators.


[CHART RE-AMENDMENT, 2026-10-01 ~16:00Z — the flip corrects to
ATTENUATION]: the R58-critic gem adopted above ("the normalizer
FLIPS the sign of fact-relevance, -0.0385 vs +0.0981") is an
ESTIMATOR-POINT ARTIFACT per the chart cell's hard-gated double
anchor: at a matched point the flattening only attenuates
(+0.0986 -> +0.0396); opt1's -0.0385 evaluates the fact gradient
AFTER the Adam step. The honest sentence: "the normalizer
attenuates the stream's fact-relevance ~2.5x at a matched point;
the trajectory-level negative alignment is the post-step view."
The paper's clause corrects with it.]

## T138 — the literature pass: the trajectory hypothesis is new as a CONTROL, predicted as THEORY (title re-stamped per C13-1; 2026-09-30 ~10:55Z)

scratch/lit_beat_20260930.md, ~20 searches, all supervisor anchors
pinned. THE GATE: six phrasings of trajectory-vs-static /
learned-vs-random displacement — the dissociation appears NOWHERE.
THE POSITIONING THAT MATTERS: Evron (COLT 2022) and Goldfarb & Hand
(AISTATS 2023) already state IN THEORY that forgetting is governed by
task/gradient geometry rather than displacement magnitude — linear/
overparameterized theory, no static controls, no fact-readout assay.
So the lab's claim is not "alignment was unsuspected"; it is "the
theory predicted it and nobody ran the control": the 3-way controlled
dissociation at matched per-coordinate RMS (training kills — corpus
or noise-label; random jump survives, graded kappa 5-6) plus the
census form. CITE THE THEORY LINE AND LEAD WITH THE CONTROL —
uncited, T137 reads under-theorized; cited, it reads as the
experiment the theory was waiting for. The other sharpenings: the
wall's family is hard-constraint methods (Wolczyk ICML'22, Elsayed
RLC'24 — defended by minimality: one commit + one scalar ball, zero
old-task statistics, plus the survival assay); the rhythm's is
learned/interference-based replay scheduling (Klasson, MIR, PER —
defended by the zero-parameter self-timed gate and the g2g
head-to-head); the cone's is task arithmetic (Ilharco — the defence
is to ADOPT the vocabulary: report the install-vs-wash cosine; if
the wash approximates minus-the-install-vector, forgetting at this
scale IS task arithmetic, which would be a simplification, not a
refutation). FLAGGED for full-text re-check before submission: SFAO
(OpenReview Feb 2026) and Elsayed RLC 2024. The e188 co-read gains
the install-vs-wash cosine (cheap, decisive for the vocabulary
choice).

## T137 — g3K: the basin is graded and the no-basin is a trajectory law (2026-09-30 ~10:50Z)
[C13-1 AMENDMENT ~11:00Z — the claim's stamp: PROPOSED, not law. The
card's body carried the scope (organ n=1 per organism, overlapping
kappa intervals) but the TITLE minted "law" from one cell — my paper
amendment 17 minutes after landing repeated the sin. Status until
e188's integral test + >=2 more organisms: THE TRAJECTORY
HYPOTHESIS. The cone noun (n=3 wash-draw, one organ) keeps its
licensed standing.]

The kappa pair came back nearly EQUAL — kappa_store 5.0 [2.8-6.6],
kappa_host 6.0 [4.0-9.7] — and the MIXED verdict hides the day's
sharpest dissociation, because BOTH kappas at ~5-6 means: (1) the
store's celebrated wide cone (the critic's 24-32x) does NOT survive
composition — that was the store-isolated leg with the host pristine;
at the organism, the readout path g5 located in the HOST dies to
isotropic at 4-8x, indistinguishable from the host's own fact. THE
CONE IS AN ORGAN PROPERTY, NOT AN ORGANISM PROPERTY. (2) e185/e187's
"isotropic kills at displacement-match" does NOT replicate as STATIC
noise — the host fact holds a graded 4-10x basin against random
displacement. THE NO-BASIN LAW IS A TRAJECTORY LAW: any LEARNED path
to the same displacement kills (corpus or noise-label training alike
— e185's arms were trajectories), a random jump does not. W022's
alignment story absorbs BOTH readings: the wash kills because it is
aligned with the death gradient (cos -0.44); random directions
(alignment ~0) need 4-10x the magnitude — the graded basin IS the
alignment law's static footprint. The fast drift (cos d1..d300
~0.13) says the killing object is a DRIFTING aligned front, not a
fixed direction — the cone's 45-degree tilt tolerance is its static
shadow. e188 (W022b) is now the pointed test: the alignment integral
separates trajectory-kill from static-kill by construction.
CONCENTRATION REFUTED at 1x (magnitude-uniform wash) — the
g3-vs-e185 disagreement was subspace choice. SCOPE: organ n=1 per
organism; host ruler = the lineage's discriminative twin (S-DISC);
single wash snapshot per organism.

## T136 — g2f: the organ survives the root lottery; the ruler did not (2026-09-29 ~21:35Z)

The stranger-base redraw splits the claim the OTHER way: the rhythm
sustained IN FULL (11 events, 100% in-band, cycle-median 0.598 vs the
0.5 bar, duty 67%) on a root whose gate failed a third time (0.601;
lottery 0.591/0.711/0.684/0.601 — 3 of 4 draws miss 0.7). TIMING IS
NOW 3/3 ROOTS (locked, install-redraw, base-redraw); amplitude 2/3 —
and the ONE amplitude miss (g2e 0.388) decomposes as GEO SHIFT, not
weakness: g2e's root moved argmax to g+12 while the frozen
no-shopping ruler stayed at g0 (g0-pooling there read 0.563/60%);
g2f's root kept argmax AT g0 and amplitude recovered on a WEAKER
root. T132's simple floor story ("the root contributes the floor")
is REFUTED: amplitude tracks RULER-ROOT GEO ALIGNMENT, not strength.
DECOMPOSITION v2: the organ contributes the clock everywhere and the
amplitude wherever the frozen ruler lands on the root's own argmax;
the recipe's fragile half is the ROOT GATE itself. W021 CONNECTION:
the frozen ruler is the instrument that cannot FOLLOW — the
no-shopping rule traded bar-shopping for geo blindness; the
registered fix when the rhythm returns to the ladder is a
battery-pooled median or an argmax-at-construction rule registered
BEFORE the draw (all three geos always co-reported). STANDING: the
organ is robust on every root tested at the ruler's own terms;
formal ORGAN-REPLICATES waits for a gate-clearing root (g2h queued:
the r2 ladder's stronger install).

## T135 — g3R: the cone licensed — forgetting is not distance, it is DIRECTION (2026-09-29 ~21:15Z)

The split replicates at both new seeds with zero broken legs
[sharpened per the agent's final texture field: g3's exact-1x
lambda kill (0.238) was a DRAW ARTIFACT — both replicates sit
ABOVE the dissolve bar at 1x (0.597/0.564) and kill at 2x
(0.156/0.096); the registered <=2x criterion holds 3/3, and
the robust object is the DISSOCIATION — at 2x displacement the
wash direction reads 0.10-0.16 while isotropic reads 0.85-0.89,
a ~0.7 g0 gap at identical L2. The full-wash kill (+1, g0
~0.0000) is draw-invariant; the lambda-sweep's edge is
draw-sensitive in (1x,2x)]. THE THIRD LAW-GRADE ARCHITECTURAL CLAIM
(with the wall g1bR and the rhythm g2d): memory's fragility is
DIRECTIONAL — the basin is a cone, not a ball [scoped: the
store's own; the whole organism still dies isotropic at match,
per e185/e187]. THE G-SERIES' LADDER NOW COMPLETE: three
claims at n>=3 (wall / rhythm-timing / cone), two scoped
(compass-positional 3/3-substance band-brittle; headset knife
3/3 flat-CE), the anatomy (organ/access/route), the
two-mechanism kill, and the law-census. THE REPLICATION
PROGRAM ACHIEVED WHAT R55 DEMANDED — every g-positive either
licensed at the lab's own standard or honestly bounded, none
minted unqualified.

## T134 — g4R: the substance replicates where the bars don't — honest bounds as findings (2026-09-29 ~20:50Z)

Both g4 positives return honest bounds, and each bound TEACHES:
the compass's row BAND is draw-brittle (r2's site slid to rows
3-4 at full strength — the placement law held, the registered
window didn't; the band is a convention of one draw, not a
property), and the knife's census-RANKING rule fails because it
selects by damage — and damage-ranking finds organism pillars
(L0H3 at ~2 nats ablation cost), not fact-specific circuits.
The committed headset kills flat 3/3. THE DEEPER LESSON (both
bounds share it): the SELECTION RULES are the brittle layer;
the PHENOMENA are robust. The spine's install prediction went
4/4 — the architecture predicted its own carrier every time it
was asked. CLAIMS' FINAL FORMS: compass-positional (3/3
substance, one-family, band-scoped); headset-specific knife
(3/3 flat-CE); the wall and the rhythm at n=3 law grade; the
cone and two-site at n=1 (g3R the remaining replicate).

## T133 — g1bR: the well is real across seeds — the arc's central positive at law grade (2026-09-29 ~20:15Z)

The wall replicates cleanly: two new wash seeds, both W1 arms
flat at ~0.9 through 300 steps (mins 0.746/0.803 vs the
reference's 0.777 — a narrow seed band), both controls dead by
+50. THE COMMIT-AND-PROJECT WELL IS LAW-GRADE at n=3 seeds on
the calibrated root: one event (the commit) plus one geometric
constraint (the projection) buys 300 steps of survival through
the stream that kills in 2 — the arc's central architectural
positive now at the lab's own standard. THE G-SERIES' EVIDENCE
STRUCTURE AFTER g1bR + g2d: two architectural claims at n>=3
(THE WALL; THE RHYTHM's wash-robust timing) + the anatomy
(organ/access/route) + the two-mechanism kill + the law-census
(no-basin licensed; compass a census of two; knife replicated
once; cliff bounded). The well/revival system — the paper's
second arc — now stands on replicated legs [official-report
scoping: what replicates is the well at R=0.7 SPECIFICALLY
(radius-tuned; W2's dip and W3's death order with R — the
licensed claim is commit(0.7)+projection, ~0.28x the basin
prior's low end); wash-draw seeds replicate, the ROOT remains
single (g2e's question); W2/W3/noise cells stay n=1; device
migrations recorded, immaterial at the 0.90-vs-0.50 margins].

## T132 — g2e: the timing is the organ's; the amplitude is the root's (2026-09-29 ~19:45Z)

The root replicate splits the rhythm claim cleanly: the
TIMING (the event cadence, the 20-45 band, the self-fire)
replicated on the fresh root 2/2 — the organ KNOWS when to
fire wherever it is planted. The AMPLITUDE (the adjudicated
cycle-median on the frozen ruler) did not — and the
attribution runs through the ROOT's strength lottery (three
draws: 0.591/0.684/0.711 against the 0.7 bar) plus a geo
shift (the fresh root's argmax moved to g+12; on g0 pooling
its median reads 0.563/duty 60%). THE HONEST DECOMPOSITION:
the organ contributes the clock; the root contributes the
floor — and a weaker root's floor drags the sawtooth's median
under the bar even when every event still resurrects from the
dip. THE ARCHITECTURE CLAIM'S STATUS: timing root-robust,
maintenance root-draw-bound, n=1 root at the frozen bar. The
next rung (a base-seed redraw — s4306 on disk) would
dissociate organ-lottery from root-lottery; the claim's
scoping statement is already honest as-is.

## T131 — g2d: the noun licensed — three rhythms, one waveform; the arc's generative program has its first law-grade architectural claim (2026-09-29 ~19:20Z)

The seed replicate closes it: three wash-draw seeds, distinct
event schedules, one waveform (~70% duty, ~24-36 spacing,
~1/30 density, medians 0.587/0.602/0.615). THE SELF-MAINTAINING
RHYTHM IS A NOUN — the first g-series claim licensed at the
lab's own n>=3 standard. THE FINDING'S SHAPE: the rhythm's
TIMING and MEDIAN are seed-robust; its AMPLITUDE is the
lottery (troughs 0.08-0.60 across seeds — every event still
resurrects from whatever depth). THE SYNTHESIS STANDING: a
memory that maintains itself by knowing when it is dying, on
one root, wash-robust, zero new parameters — the lab's
dissection arc (memories wash out) and generative arc (the
rhythm can be built) meet at a single object. WHAT REMAINS FOR
THE ARCHITECTURE CLAIM: root/lineage generality (a fresh root
at the same recipe) — the same ladder every e-claim climbed.


[G2G RESOLUTION, 2026-10-01 ~16:55Z]: the controls landed. The
band was partly a construction floor (at refractory 8, 5/14
spacings <20 — disclosed); the wash's clock still shapes 9/14;
maintenance IMPROVES at shorter refractory. The head-to-head:
the organ wins at 1x (+0.07 both runs, the registered
same-ruler comparison; the 0.693 co-read was endpoint-vs-median,
superseded); FIXED-MATCHES-OR-WINS is the safer letter beyond 1x.
The final claim: threat-responsive [0.5x, 2x], autonomy +0.07 at
operating threat, refractory-tunable; the seed ladder licensed.

## T130 — g2c: the organ vindicated — memory as a self-timed oscillation, its waveform mapped (2026-09-29 ~19:05Z)

The owed cell returns the vindication: the organ MAINTAINS IN
RHYTHM (cycle-median 0.615, duty 69%) — g2b's failure was a
phase artifact exactly as suspected, and the dense
reconstruction maps the full waveform: slow rise, plateau at
0.65-0.74 through mid-cycle, late fall into the pre-event
dip. THE SYNTHESIS COMPLETE: the lab's arc ran from
"memories wash out in two steps" (e176) to "the rhythm is the
memory" (e179) to "the rhythm can be ARCHITECTURAL" (g2/g2b)
to "the architecture maintains it in oscillation, self-timed,
with the waveform mapped" (g2c) — a net that knows when it
is forgetting, re-teaches itself, and spends 69% of its time
above the expression bar, on ~1/30 rehearsal density. THE
REMAINING DEBT: the seed replicate (rhythm's noun-hood per
R55's standard) — one more run.

## T129 — g5: the memory is two-site and the organ is survivable — the wall decomposition finds the real fragility (2026-09-29 ~18:25Z)

The falsifier firing is the informative outcome: the query-cone
wall failed because the cone is not a property of W_q's
weights — it is a property of the COMPOSED query path q =
W_q.LN(h), and the wash moves h (the host's residual stream).
T126's census attribution was right about the function, wrong
about the subspace — a correction the transplant cells make
unambiguous (washed-W_q works fine in a root host at full
displacement; root-W_q dies in a washed host). THE POSITIVE:
the walled STORE survives flat at ~root through the wash —
the g-series' first survivor — and the tax is near-zero as
predicted. THE MEMORY'S REAL FRAGILITY MAP: two unwalled sites,
both in the host (the query path's stream dependence; the
expression route), plus a survivable organ. THE COMPOSITION
THAT WOULD WORK names itself: store + WHOLE-NET wall (g1b's
+0.53-nat tax) — or store + a stream-stabilizing host (a g6
question: can the host's h be walled/regularized more cheaply
than the whole net?). FOR THE PAPER'S SECOND ARC: the
well/revival system now has its full anatomy — the organ
(wallable, survives), the access path (stream-dependent, the
real kill site), and the expression route (a second
independent kill site) — memory as a three-part dependency
chain, each part separately attackable.

## T128 — g2b: the organ knows it is dying now — the self-maintaining memory oscillates (2026-09-29 ~18:00Z)

The one-line fix vindicates T122's diagnosis completely: the
gate watching the ONSET (the channel the wash kills) finds the
rhythm autonomously — 10 events, spacing 24-36 steps,
essentially the external schedule's frequency, without ever
seeing it. THE ORGAN IS NOW A SELF-MAINTAINING MEMORY in
every sense except the frozen bar's letter: the ruler
oscillates 0.03->0.72 with every event a resurrection; the
+300 checkpoint simply sampled a trough (a phase-offset
replicate is the owed cell — the honest read is 'maintains in
rhythm' pending it). THE g-SERIES' FIRST WORKING SYSTEM IS
COMPLETE IN ALL BUT PHRASE [R55: n=1 single seed; 'rhythm' is not a noun until
two more seeds + the g2c phase cell; the verdict stands as MAINTAIN-FAILED]: a net that knows when it is
forgetting (the onset sensor), and re-teaches itself (the
error-carrying replay) — 0 new trainable parameters; the
maintenance budget self-administered. THE SYNTHESIS WITH
e179: the resurrection economy measured externally (9 events
suffice) is now measured INTERNALLY (10 events, self-timed) —
the same number, found twice, once by the experimenter's
schedule and once by the architecture's own sensor.

## T127 — g4: the compass is architecture-robust; the cliff is scale-robust in its ABSENCE; and the content floor refused every fact (2026-09-29 ~17:50Z)

The generative program's law-census across the g-series so
far: THE COMPASS IS ARCHITECTURE-ROBUST (positional placement
in a net with an offered content channel — the A-floor was
never recruited, by census OR by surgery; when the architecture
COULD host the conjunction in a table it still chose the
positional floor and the heads). THE NO-BASIN LAW IS
ARCHITECTURE-ROBUST (both roots dissolve by +2; the attractor
died; only the wall survives it). THE CLIFF IS
SCALE/LINEAGE-BOUND: no switch on either root at 0.86M —
the variance cliff of e147 is a property of the 2.7M line
(and the 0.84M family-2 line's doors), NOT a universal of
optimization. THE HEAD-KNIFE IS THE SURGERY INVARIANT: the
N2-class flat-CE kill of the variance-built fact replicated
on a brand-new architecture — the flight-to-heads causal cell
is now the lab's most portable surgery result. THE SPINE'S
HALF-VICTORY: the gate's pre-teaching sensitivity correctly
predicted the install carrier (P) — the architecture DID
predict its own memory type — and honestly mispredicted the
w8 carrier (a mixed carrier the classifier refused to round).
FOR THE SYNTHESIS: three laws survive the generative gauntlet
(compass, no-basin, head-knife); one is bounded (the cliff);
and the A-floor's total disuse is a NEGATIVE result worth its
weight: per-token address tables cannot host conjunction
memories at this scale — only positions and attention can.

## T126 — g3: the attractor dies like everything else — and the direction/energy split deepens (2026-09-29 ~16:35Z)

The generative-store test returns the law's strongest
confirmation and its sharpest mechanism split. THE LAW HOLDS
ABSOLUTELY: an explicit attractor store — the PP lore's
candidate for what memory SHOULD be — dies at +1 exactly like
the discriminative net, at the same per-coordinate rate (the
first AdamW step saturates every coordinate regardless of
substrate). There is no architectural refuge from continued
optimization: not the well (g1b walls it, at a tax), not the
attractor (g3 dies with it). BUT THE KILL-SITE IS PRECISE:
QUERY drift, not key damage — the washed net cannot FIND its
patterns (washed-q x root-K = 0.08) while the patterns remain
(root-q x washed-K = 0.91). The store forgets its ADDRESS, not
its CONTENT — the address/field split of the whole arc,
reproduced inside a 17k-parameter organ. THE DEEPEST CELL:
isotropic matched-L2 noise on the store SPARES it (0.89 at 4x
displacement) while the wash direction kills at 1x — combined
with g1b's two-mechanism split, the picture completes: THE
KILL IS DIRECTION-SELECTIVE (the wash direction hits the query
cone; isotropic displacement does not), which reframes the
no-basin law one final time: the basin is not a BALL — it is
a CONE, and forgetting is the query drifting out of it. FOR
THE PROGRAM: g3's organ + g1b's wall compose a candidate
system (store the patterns; wall the query projection only —
a much cheaper wall than the whole net). g5's question names
itself: WALL THE QUERY CONE.

## T125 — g1b: the wall holds — and the kill splits into two mechanisms (2026-09-29 ~15:30Z)

The generative program's first clean architectural verdict:
one commit event plus a hard L2 projection installs a WELL
that holds the fact flat at 0.918 through 300 steps of the
stream that kills the control in two — survival ordered in R,
bracketing D_kill (the wall is a dynamic basin-width
measurement). MEMORY IS ARCHITECTURAL AGAINST DISPLACEMENT [n=1, one lineage].
But F3 splits the kill into TWO MECHANISMS: the corpus kill is
DISPLACEMENT-MEDIATED (the wall heals it — the settled +1
reads 0.945, the projection undoes the step); the noise kill
is POSITION-ACTING (the settled +1 reads 0.004 — the damage
persists AT the pinned position, inside any radius). THE
NO-BASIN LAW'S FINAL DECOMPOSITION: forgetting under real
data = walking out of a well (geometric, wallable, re-enterable
— g1's well + e179's revival); forgetting under noise =
damage at fixed position (non-geometric, unwallable). THE
PAPER'S SECOND ARC CANDIDATE: the well/revival pair is an
engineerable memory system — commit + project against drift,
one replay against lapses; the g-series' first existence
proof that the lab's laws translate into DESIGN. What remains
unexplained: WHY noise damages at fixed position (the
damage's own mechanism — a g-series question, not a
dissection question).

## T124 — g1: the wall is real; the organism was too small to testify — and the partial signature is tantalizing (2026-09-29 ~14:45Z)

The anchored ball's first run returns an honest gate failure:
the 0.84M organism consolidates but does not GENERALIZE to
offset -12 (the jitter set's ±8 span is width-enough at 2.74M,
not at 0.84M) — the arc's own ruler bar (0.78) unreachable,
so nothing adjudicates. But the textures: THE WALL WORKS
(displacement pinned exactly at R + the registered one-step
Adam fuzz); W1 HELD A HALF-EXPRESSED FACT FLAT through +300
where the control free-ran to death (0.29 vs 0.001) — partial
maintenance on a partial root, exactly proportional; and the
NOISE KILL PIERCED THE WALL (dead at pinned R with CE
devastated — displacement-matching is NOT damage-matching; the
wall bounds the DRIFT, not the DAMAGE — arguably the deepest
single number in the run: it means the noise kill's mechanism
is not "walking out of the basin" but something that acts
WITHIN any radius). THE DISCHARGE IS CHEAP: the 2.74M
continuity cell (one config line, the e131 root on disk) —
g1b. FOR THE PROGRAM: the first two builds taught a sensor
lesson (g2) and now a testimony lesson (g1) — the architecture
program's results are only as adjudicable as the organism's
baseline expression; the gate-first design (the spec's own
abort clause) saved a false verdict here, exactly as intended.

## T123 — e182: the physics translates — wider basin, same DIRECTION, not the same clock (title corrected per R56 audit; 2026-09-29 ~14:10Z)

The external-validity fuse returns the answer the arc needed:
GPT-2's facts are NOT wash-proof. At lr 5e-5 the probes erode
to 0.66 retention by +50 steps with the perplexity IMPROVING
(the corpus gets better as the facts fade — the surgical
signature, at 148x the lab's scale); at 5e-6 they hold within
the horizon — the lr-scaling the rate law predicts. THE BASIN
IS WIDER (the exit is real but gradual, lr-gated, ~5-10x slower on
the lr axis than the small-net rate law — NOT proportional
sqrt(P) as first read [corrected per the official report]);
THE LAW IS THE SAME IN DIRECTION, SLOWER IN TIME CONSTANT.
THE OFFICIAL REPORT'S SHARPER FRAME: GPT-2 sits BETWEEN the
lab's two extremes — not basin-free (the two-step wash does
NOT replicate: +10 retention 0.987-1.004 at BOTH lrs), not
wash-resistant (no protected basin: 5e-5 broke 0.80 by +50).
The decay is broad (Paris 0.685->0.251, Cairo ->0.162,
dollar ->0.130); the few-shot conflation (2-shot context is
part of the instrument — the decay's locus, fact-storage vs
task-following, unadjudicated); the probe bias is toward
resistance (so the kill is strong, the flat +10 an upper
bound on speed).
The field-facing line's final form stands: pretrained facts
are not archives either — they are facts practiced harder; the
dataloader's direction still chooses who dies, now at scale.
FOR THE PAPER: the external-validity exhibit is the two-curve
figure (5e-6 flat; 5e-5 surgical decay, perplexity overlay)
plus the sqrt(P) basin-width scaling note. W019's scope clause
completes: the no-basin finding is now bounded by neither
family NOR scale (tested 0.84M, 2.7M, 124M).

## T122 — g2: the organ's engine works; the sensor was the failure — architecture's first lesson (2026-09-29 ~14:05Z)

The generative turn's first build returns the most instructive
possible outcome: NOT maintained (0.024@+50) but NOT for any
registered reason — the gate fired once instead of 5-25 times
because the spec's monitor averaged over the name's self-
correlation channel (which the wash spares) rather than the
onset (which it kills). THE ENGINE IS VINDICATED: the single
self-triggered event resurrected the fact 0.024 -> 0.44 in 24
steps — the resurrection economy, architectural. THE SENSOR
FAILED: the organ did not know it was dying because it was
listening to the wrong channel of its own reading — a sensor-
actuator mismatch that is itself a memory-science finding:
THE NET'S OWN SELF-CORRELATION IS NOT ITS MEMORY (the wash
leaves Z->E at 0.65+ while the fact is dead at 0.02) — the
distinction between knowing the name's SEQUENCE and knowing
the name (the ctx->Z onset) is exactly the gap the detector
needed. g2b named: onset-only monitor. FOR THE PROGRAM: the
first built architecture taught a measurement lesson, not a
memory lesson — and that IS the generative turn working (the
architecture found a distinction the dissection's instruments
had already drawn but never had to ACT on).

## T121 — E187: the mechanism formally licensed — and the recovery itself a small demonstration of the lab's memory (2026-09-29 ~12:30Z)

The replication debt discharges clean: four cells, four kills,
both arms' textures intact (the labels/shuffled magnitude split
and the orthogonal direction), the collateral-devastation
profile matching. The no-basin mechanism is now n=3 draws per
arm — the skeleton's paragraph stands formally. THE SAVOR: the
recovery IS the thesis — the outage killed the run mid-flight,
and the surviving checkpoints (the run's own memory) let the
experiment be resurrected from them at near-zero cost, exactly
as e179's single replay resurrected the fact. The lab practices
what it found: checkpointed state + one directed event = cheap
re-entry. FOR THE PAPER: the mechanism paragraph's [n=1,
replicates owed] flag clears to [n=3/arm, one stream] — the
last formality before assembly; only e182 (in flight) remains.


[E182C RESOLUTION, 2026-10-01 ~15:40Z]: THE SURGICAL SIGNATURE
RETIRES at 124M — matched held-out controls erode with the fact
(ratio 0.92 at +80) under the bit-tight replayed wash; the clause
becomes "ordinary forgetting with improving perplexity", NOT a
no-basin signature. The direction-only transfer (this card's
corrected title) and the time-constant texture survive as the
weaker form. A template-locus hint opens phase 2 (the near-related
battery collapses fastest).

## T120 — E179: the resurrection economy — nine events, one revival, a sawtooth (2026-09-28 ~23:00Z)

The rehearsal law returns the session's last great texture:
maintenance is CHEAP BEYOND EXPECTATION (nine replay batches
per 300 wash steps suffice — the whole anatomy intact at
r=1/32) but NOT a dial (non-monotone in r; no pump; a
resurrect-and-oscillate sawtooth whose +300 endpoints ride
cycle phase). THE DEEPEST FINDING: ONE REPLAY RESURRECTS THE
DEAD — a fact killed to 0.033 at +2 returns to 0.686 by +50,
EIGHTEEN wash steps after a single replay event. The re-taught
state is far more wash-resistant than the consolidated root
ever was — re-entry into the basin is cheap and STICKY, even
though staying in it was impossible. THE INTEGRATION: the
basin law (exit is displacement-limited; the corpus exits it)
meets its complement (re-entry is event-limited; ONE directed
event restores residence that outlasts many exits). Memory in
these nets is not a state — it is a RHYTHM: exit cheaply,
re-enter on reminder, oscillate. FOR THE PAPER: the kinetics
pair completes (exit law + re-entry economy); the practical
paragraph sharpens — rehearsal does not prevent forgetting; it
makes forgetting irrelevant (9 reminders per 300 steps keep
the fact whole through washes that kill it 100x over).

## T119 — E180: the rate law — survival is displacement-limited, and the gentle regime forgives (2026-09-28 ~21:55Z)

The kinetics extension lands the quantitative replacement for
the demoted rhetoric: t* ~ lr^-1.16 across a 100x lr range
(R^2 0.975), with lr x t* roughly constant (2-6e-3) — the wash
is DISPLACEMENT-LIMITED to first order. THE BASIN HAS A WIDTH — measured
~2.5-5 L2 over 2.7M params (per-coordinate RMS ~1.5e-3) [R54:
the 5e-3 figure was a 3-order regression of R53's correction —
the lr x t* product is a different, dimensionless currency]; at
lr 1e-5 the fact still lives at +300 (62% expressed). LR-IMMUNE dies — there is no
intrinsic two-step fragility; there is a basin the optimizer
must walk out of, and the walk's speed is the lr. THE PAPER'S
FINAL MECHANISM PARAGRAPH: "memory in these networks has a
narrow robustness basin (~2.5-5 L2 over 2.7M params; RMS ~1.5e-3/coordinate); continued
optimization exits it — the exit rate is the learning rate
(t* ~ lr^-1.1..-1.4 across grid-legal fits; the stored -1.16 conservative); what corpus direction adds is not the exit
but the surgery (the fact dies, the organism recovers)". The
+10 pumps at both gentle lrs echo e184's coin-flip texture.
FOR W019: the field-facing line's FINAL form — "these memories
have a narrow basin; gentle training forgives, ordinary
training exits it, and the dataloader's direction only chooses
who dies".

## T118 — E163: the intro stands — and the dial's two faces explain the whole saturation debate (2026-09-28 ~21:20Z)

The licensing cell returned DIAL-VALID: the census dial reads
0.275 on a 7%-dependent fact vs 0.88-1.00 on the carriers —
carriage-discriminating, the intro licensed. THE TWO-FACE
FINDING resolves the T083 debate precisely: the MEAN arm (row-0
perturbation) is the saturated face (74% drop even on the
non-carrier — every readout does load the sink); the ZERO arm
(removal) is the discriminating face. The e131/e142 min-
convention accidentally chose the right face. FOR THE PAPER:
the intro's "born with one memory organ" stands licensed
(13/13 informative); the T083 caveat rewrites to the two-face
footnote ("the dial's perturbation face saturates; its removal
face discriminates — the census used the latter"). THE
SUBMISSION BLOCKERS ARE NOW ZERO: every named blocker
(e158/e154/e164/e166-corrections, e163, the grid, the tail, the
mechanism) has its cell run and folded; what remains is
assembly, the optional kinetics laws, and the GPT-2 fuse.

## T117 — E185c: the tail is real, and it ends — the lottery is 100-200, the asymptote is shared (2026-09-28 ~20:55Z)

The device-confound discharge returns the cleanest possible
confirmation: the CPU-only re-run reproduces e184's tail
ordering EXACTLY at every checkpoint — 10904-slower at 10/10
dials (+100) and 8/10 (+200), then convergence (+300, NOT-
slower). T112's tail sentence stands device-free. THE
REFINED SHAPE: the tail lottery lives in 100-200 (the
settlement period), not the asymptote — both seeds' tails
converge to the same floor by 300. The seed lottery in these
nets is real but FINITE: it decides how fast the wreckage
settles, not where it settles. THE ASYMMETRY-OF-EXISTENCE'S
REPPLICATION DEBT IS DISCHARGED on this dial: the tail claim
is now n=3 seeds x 2 device protocols [the agent's final report
adds: the same-seed device ratios sit at <=0.2% — two orders
below the 1.8-2.4x seed separation; the confound was immaterial
in size, not just policy; and the CPU re-run is itself a second
independent trajectory confirming the ordering]. FOR THE PAPER: the
conversion-observability caveat gains its footnote ("device
migration at s175/s275 in the original run; the ordering
reproduces CPU-only").

## T111 — E183: the noun unbound — dissolution is stream-invariant; only its timing is a lottery (2026-09-28 ~17:45Z)

The last gate opened: with the host-junction background
filtered (3.26% realized rejection), the consolidated fact
still dies on the same two-step clock, kinetics
indistinguishable from the neutral stream. ACROSS THE SESSION:
three stream compositions (extinction, neutral, filtered), two
lrs, every memory type (sink-coupled, dwell, site-stored) —
DISSOLUTION is invariant; only the RATE varies (the lr knob,
the seed lottery, the type gradient inside its own noise).
THE PAPER'S FINAL LEAD FINDING, unbounded save the seed clause:
no memory state tested retains expression under continued
training without the fact's windows. The mechanism candidate
is now the barest possible: ORDINARY CORPUS GRADIENT FLOW at
lr 1e-3 — the stream's own pressure, not any fact-adjacent
signal. THE DISCRIMINATOR THE HONEST FORM STILL OWES: a gentler
regime's curve (the wash-rate law's lr 1e-5 leg) and seeds —
the dissolution's invariance makes the seed question CHEAPER
than feared (three streams already triangulate the noise), but
the lab's own >=3 rule stands. FOR THE EPITAPH: the
parenthetical's "3.84% still unexamined" clears — it was
examined, and it was innocent.

## T110 — [R51 BOUNDS: the null is GRID-LIMITED (crossing in (30,100] for all three — real savings invisible at this resolution); the naive control's substrate confound (install-unfamiliar, +0.2 CE first-contact) unexcluded; the late INVERSION (naive > washed at 100/300) unaccounted — 'the paradigm split in two' WITHDRAWN from paper reach; discriminating cells named (a different-nonce re-teach; a yoked e001 control)] E175: no savings at the threshold (bounded) — and the early-kinetics residue that never cashes (2026-09-28 ~17:05Z)

The Ebbinghaus test returned its cleanest modern form: the
washed net re-learns at exactly the naive price at the
threshold (100 steps for all three states) — NO savings, the
archive truly empty where it matters. THE COMPLEMENT (the
texture that didn't gate): the washed arm leads the naive arm
at EVERY sub-threshold checkpoint (0.65 vs 0.29 at step 10) —
a residue in the EARLY kinetics that never converts to a
threshold advantage. The classical savings paradigm, split in
two: sub-threshold savings exist; threshold price does not. FOR
T098's bound: the kill showed no thin-lesion fast recovery
under the persistent clamp — but g-12 recovered to 0.78 at step
10 while g0 lagged: the GENERALIZING readout recovers around
the clamp fast; the TRAINED one pays. That asymmetry (novel-
geometry access routes around the lesion; the home readout
doesn't) is a new dissociation between the two access modes,
report-only, worth a follow-up if the clamp operationalization
is refined (a transient kill rather than persistent).

## T109 — E176N: neutral-dissolves — the wash survives its confound; the paper's lead finding stands (bounded by one residue) (2026-09-28 ~16:50Z)

The discharge fired against the escape: with the anchors'
contradiction channel provably removed (0/16 junctions), the
consolidated fact still dies on the two-step clock, whole-
anatomy, with the CE transient honestly priced. THE LEAD
FINDING (the discussion's, per R50's title verdict): no memory
state tested retains expression under continued training
without fact-bearing windows — on either stream, at any lr
tested, with the wash rate optimizer-scaled. THE RESIDUE THAT
KEEPS IT BOUNDED: the shared random half's 3.84%/window
host-junction background — the filtered-stream cell (e183,
queued) is the last step to the unbounded noun. e177's
SCRATCH-MEMORY unblocks (the site-type's neutral wash remains
technically open, but the burden has shifted: the root's wash
is stream-insensitive). W019's bar eases to [n=1 + e183 owed].
THE 60% RESTORE-IN-TO-+50: e178's half-fact is real but
partially wash-depth-inflated — the interface story and the
depth story share the credit.

## T108 — [R51 BOUND: the 4-24x gradient is INSIDE demonstrated within-type variability (the root swings 10x under one lr knob; the dwell spreads 37x across seeds) — 'worth a panel' WITHDRAWN; the defensible piece is the 5.5x dwell-vs-root contrast (same lineage, stream-matched); per-type seeds at matched lr/stream before any ordering] E177 (bounded): knife-proof is not wash-proof (2026-09-28 ~16:15Z)

Under the extinction-grade stream, the deep site-store washes —
W019's predicted savor held — but with a DECAY GRADIENT (4-24x
slower than the other types through +50): the memory types are
ordered by wash-rate (site > dwell > consolidated-root in
slowness... inverted: the site decays SLOWEST, the consolidated
fastest) even though all die. FOR THE PAPER (bounded form): no
memory type tested retains expression under the
extinction-grade stream; the types differ ~25x in decay RATE —
a gradient worth a panel. THE MID-WASH STATES: at +50 the
site's content census is at FULL strength while its read has
dipped — nets where storage demonstrably outlives access by 50
steps, on disk. These are the perfect substrate for the
recovery-kinetics and re-excavation questions (does a +50-state
re-learn faster than naive? can the tomb be reopened from
half-washed?). THE GATE HOLDS: e176N decides whether any of
this survives neutral streams — if both types survive them,
the entire wash story becomes extinction-specific and the
resistance question reopens with the decay gradient as its
first datum.

## T107 — [R51 DEMOTIONS: 'any-F2-gradient-triggered' WITHDRAWN (the wash channel alone kills — e176N's fact-free stream, same clock); the smoke/main adjudication CONFLICT (smoke: REHEARSAL-FAILS with an eviction transient at t~4-8 even under rehearsal; main: RESOLVES at the coarse grid — 'cohabitation at every dose' is grid-limited); 'capacity = 1/rehearsal-fraction' STRUCK from the paper (a formula from one fraction + a degenerate zero, one seed); the surviving form: REHEARSAL MAINTAINS (direction, two independent protocols)] E174: the maintenance budget (bounded) — rehearsal doubles capacity for free, and forgetting is first-contact (2026-09-28 ~16:05Z)

The two arms compose into the day's cleanest practical finding
and its sharpest mechanistic texture: (1) WITHOUT rehearsal,
the dose curve is a cliff — F1 dies at the first F2 gradient
(dose TWO in the smoke: 0.0009), not at graft formation, not
at budget exhaustion — ANY second-fact gradient kills. The
extinction-bounded frame sharpens: the kill is not the stream
or the graft — it is the FIRST INTERFERENCE EVENT. (2) WITH
1:1 rehearsal, cohabitation is total at every dose (F1 0.92+
alongside F2's formed graft, both near-full) — the one-fact
limit was never storage; it was MAINTENANCE. Capacity =
1/rehearsal-fraction. THE FIELD PARAGRAPH: interleaving during
installs costs ~nothing and saves everything — the known
mitigation, here given its mechanism. FOR W012: the bandwidth
reading dies its final death — the "share constant" was always
measuring a rehearsal-maintained occupancy, not a capacity.
FOR e179 (the frequency law): the registered question sharpens
to the interference-maintenance ratio — how much replay per
unit of interference (not per step) keeps the tenant alive.

## T106 — [R50 CORRECTIONS: (1) 'the brake survives the wash' is FALSE as written — the wash killed it with everything else (A129 -> -0.0006); it returns only CO-CARRIED by the restored class — 'resurrected', not 'surviving' (T105's NOTHING-SURVIVES stands); (2) the 'coherent half-fact' omitted site_read_span (93%) and A129 (91%) from the dial list — full structure at ~90% with expression at ~42% is the GAIN-ATTENUATION signature; e176N's arm (C) (restore-into-+50) discriminates] E178: the class-restore returns ~42% of expression with ~90% of structure — bounded (2026-09-28 ~15:40Z)

The rider returned the nuanced answer: restoring the MLP+LN
class into the washed net recovers a COHERENT HALF-FACT (all
dials to the same 40-45% depth, CE-cheap) — the class-surgery
restoration works, at half strength, on the wash axis too. But
the graded arm kills the tidy story: the L2-L4 band that
CARRIED the conversion's closure restores NOTHING here, and
the wash's per-layer MLP delta is FLAT where the conversion's
was banded. THE WASH AND THE CONVERSION SHARE A CLASS BUT NOT
A MECHANISM: two different layer signatures for two different
processes (novel-graft closure vs plain-corpus washout). The
brake's 91% return is the sharpest single localization in the
lab: the old address's suppressive sign lives almost entirely
in the MLP+LN weights — the one object that survives the wash
in near-full strength (a scar that outlasts the memory it
scarred: the brake persists when the fact is gone, if you
restore the class). FOR T092: layer 3 (the phase/brake
substrate) is now the best-localized layer; layers 1-2's
separability takes the 40% haircut. FOR THE PAPER: the
activity-dependence claim gains its mechanistic footnote —
the wash is fast, total, and only half class-recoverable;
'maintenance' is distributed across the modulatory class and
the residual context.

## T105 — [R50 CRITICAL BOUND: the wash stream was NOT plain corpus — its anchor bank used the install's OWN windows with the name deleted (16/32 per batch at the teaching junction): targeted EXTINCTION, not disuse; and the 'two steps at healthy CE' splices clocks (CE was 2.000 AT step 2 — the memory died inside a concussion; the healthy numbers are post-recovery). e176N (neutral anchors) is RUNNING and decides the noun. The honest interim form: 'no memory state we tested retains expression under the exact contexts that taught it, shown once, without the name'] E176: the wash — bounded (2026-09-28 ~15:20Z)

The decisive control returned the vertiginous answer: the
CONSOLIDATED fact washes out as fast as the dwell-phase one —
g-12 0.916 -> 0.088 in TWO plain-corpus steps, the whole
anatomy (sink coupling, brake, deletion tolerance, held
generalization) dissolving together at healthy CE. THE
CLASSICAL CONSOLIDATION STORY IS DEAD IN THESE NETS: no
gradient-resistance was ever acquired; the anchor banks in
every training the lab ever ran were quietly rehearsing the
fact, and removing rehearsal removes the memory. FOR THE
PAPER (arguably its strongest single claim): MEMORY IN THESE
NETWORKS IS ACTIVITY-DEPENDENT — there is no archive, only
what is currently being reminded; the consolidated state's
specialness (geometry-general access) is a property of the
READOUT the rehearsal maintains, not of hardened storage. THE
REMAINING QUESTIONS, sharpened: (1) e178 (the rider, mandatory)
— does restoring the root's MLP+LN class into the washed net
rescue the fact? Rescue => washout is the located rewrite
(e173) bidirectionally, and class-surgery restoration gains
its second demo; no-rescue => the washout destroyed something
the class-restore cannot rebuild (the two-step collapse
suggests the wash is fast and total). (2) e177 — is e125a's
deep site-store the ONE wash-resistant thing (a true archive:
knife-proof AND wash-proof), or does it too dissolve (the
resistance axis empty everywhere)? THE CLOSING SENTENCE,
final form: the net does not store its memories; it PRACTICES
them — remove the practice and even the most consolidated
fact is gone in two steps, while the organism sails on.

## T104 — E173: the closure has an address — the MLP+LN stream, and surgery CAN reopen what training shut (2026-09-28 ~15:25Z)

The partition gives the day's mechanism story its anchor: the
closure is a LOCATED WEIGHT REWRITE in the MLP+LN class (67%
of the conversion's delta energy; a mid-late band L2-L4 with
L3 peaking) — and restoring that class alone reopens the
geometry door to 84% of ceiling, cheaply, with the graft
untouched and the brake restored. THREE CONSEQUENCES:

(1) THE ASYMMETRY TRIPLET REWRITES: doors open by training,
are killed by head surgery, and ARE reopenable — but only by
class-level MLP+LN surgery (rows: tautological null; heads:
sub-bar). The write-once core (T037) holds for FUNCTIONAL
ADDITION by surgery but not for RESTORATION of prior function:
restoring a whole weight class is restoration, not creation.

(2) THE DISUSE CONNECTION (T101): the washout (e161) and the
closure (e173) live in the SAME substrate — the MLP+LN stream
state. Continued training of ANY kind rewrites it; the rewrite
closes geometry access; restoring the pre-conversion state
restores it. The modulatory stream is where a memory's ACCESS
lives and dies. The four-layer model's layer-3 (phase) gets
its physical substrate: the MLP+LN mid-late band.

(3) THE PRACTICAL UNLEARNING LESSON inverts once more: to
remove a consolidated memory — kill the readout heads (e160);
to RESTORE a washed-out one — restore the MLP+LN class (e173).
Two locks, two keys, both now demonstrated [n=1 each].

## T103 — E170: the hard answer — one fact wide; the follow-up is the budget axis (2026-09-28 ~15:10Z)

OVERWRITE-REAL, and cleaner than the confounded run: with the
contradiction channel provably removed, F1 still dies to the
floor while F2's graft forms and the corpus improves. The
capacity claim is now licensed at n=1-clean: the consolidated
substrate holds ~ONE fact at this budget, and a second locked
install spends the first entirely. The question moves to the
BUDGET AXIS (e174, now dispatching per the gating plan): is
capacity a dial (a smaller F2 dose cohabits) or a cliff (F1
dies at every graft-forming dose — matching e147's texture)?
And the REHEARSAL arm answers the practical question: does
interleaved F1-replay during F2's install save the tenant?
Both quotable either way. FOR THE PAPER: the "globally" clause
resolves (any second-fact install closes F1's door — by
demolishing F1); W012's bandwidth = one fact at this budget;
the two-facts-as-per-fact-doors story is dead at this protocol.

## T102 — E152R: the dwell dies, the brake lives — and the disuse frame absorbs the wreckage (2026-09-28 ~15:00Z)

The replication did its job: the session's most-quoted texture
(the dwell, the 8-16 cliff) is trajectory-specific — three
seeds, three shapes, timings spanning an order of magnitude.
What the replication CONFIRMS is better: the conversion is
INVARIANT (3/3 complete by s300) and the brake overshoot is
REAL (3/3 seeds, deepening with later cliffs, always released).
Under T101's disuse frame, both fall into place: the conversion
is a WASHOUT RACE (ordinary gradient flow vs the re-teach, on
each trajectory's luck — hence seed-dependent timing but
invariant outcome), and the brake overshoot is the old address
RESISTING at maximum exactly when the washout is winning —
suppression peaks at maximum competition (the negative-
posterior reading, now n=3). THE STRADDLE CELL (locked@band
0.26-0.55 across seeds/devices) also dissolves into the frame:
home-locking sometimes loses the race, sometimes wins it — an
unstable cell, not a gate leg. T097's two-factor gate is now
SUGGESTIVE ONLY (novel-site + zero-variance is where the race
is usually lost fastest; e165's ladder remains the axis test).

FOR THE PAPER: the mixed-state sentence softens to "a
mid-conversion state holding both natures was OBSERVED (one
seed) but is not a timescale law; conversion completes at all
seeds within 300 steps"; the brake overshoot enters as an n=3
finding. W018's four fates lose their timing claims (already
barred from paper text). The [n=1] markers on T094 clear —
resolved NEGATIVE for the dwell, POSITIVE for the overshoot.

## T101 — E161: disuse, and the reframing it forces — consolidation as gradient-resistance acquisition (2026-09-28 ~14:45Z)

The fork resolves DISUSE cleanly, and the honest consequence is
the day's largest reframing since the tautology: THE CONVERSION
WAS SUBSTANTIALLY FORGETTING. The dwell-phase memory washes out
under plain corpus in <50 steps — all geometries, the site, the
brake, the sink together — while the organism stays healthy.
There was no switch thrown by teaching; there was an
UNCONSOLIDATED memory being washed by ordinary optimization
pressure while a new one was trained in. What remains genuinely
switch-like: e147's cliff (variance builds geometry-general
access that locked training does not) and e125a's endpoint
incorrigibility. The unified honest story: memories lie on a
GRADIENT-RESISTANCE axis — fresh installs and dwell-phase
memories wash out under any continued training; deep
site-stores (e125a's endpoint) and (pending e176) consolidated
memories resist. THE DECISIVE CONTROL (e176): freeze the
FULLY-CONSOLIDATED ROOT. SURVIVES => consolidation IS
resistance-acquisition (CLS licensed, the paper's claim 2
rewrites to the resistance axis); DISSOLVES => even
'consolidated' is use-it-or-lose-it (the anchor half of every
past fine-tune was quietly maintaining the fact — every
'experiment' was also a rehearsal).

READING MAPS FOR THE RUNNING FLEET (registered ~15:50Z, all
before data):
- e174 (dose + rehearsal): the rehearsal arm was already live
  at 0.994 — REHEARSAL-RESOLVES is firing; the fold's open
  question is the DOSE side (WIN-WIN vs ALL-OR-NOTHING — is
  capacity a dial or a cliff?). If ALL-OR-NOTHING fires WITH
  rehearsal rescuing: the paper's capacity paragraph becomes
  "one fact wide without rehearsal; cohabitation IS rehearsal"
  — the two facts coexist only by interleaving, which under the
  extinction-bounded frame reads as: the second install's
  windows extinguish the first UNLESS the first keeps being
  shown. Cohabitation = maintenance, not storage.
- e176N (the neutral wash): NEUTRAL-DISSOLVES => the
  activity-dependence noun survives its confound (W019
  un-bounds); NEUTRAL-SURVIVES => extinction was the killer —
  the honest headline stays the interim form and e182's GPT-2
  wash becomes the question "does the field's own organism
  survive ITS name-deleted streams?"; the lr rider prices the
  optimizer-shock reading; arm (C) prices the gain story.
- e177 (site wash): its fold WAITS on e176N (both stream and
  frame). TRUE-ARCHIVE + NEUTRAL-DISSOLVES would be the
  strangest combination — a knife-proof, wash-proof store that
  the sink-coupled type lacks; SCRATCH + NEUTRAL-SURVIVES
  would mean BOTH types survive neutral streams and ALL washes
  were extinction (the resistance question reopens entire).
- e175 (savings triple): the NEGATIVE-SAVINGS branch is now
  doubly loaded — a surviving scar slowing re-teach would
  confirm T106's brake-co-localization as FUNCTIONAL
  interference, not just arithmetic.

E176 READING MAP (registered ~14:50Z, before its data — with
the T104 connection): if the root DISSOLVES under plain corpus,
the cheap rider that must run is the e173 INSTRIMENT in reverse
— restore the root's MLP+LN class into the washed net: if THAT
rescues the fact, washout and closure are confirmed as ONE
mechanism (the located MLP+LN rewrite) operating in both
directions, and "restoration-by-class-surgery" gains its second
demonstration. If the root SURVIVES, the resistance axis is
real and the follow-up is e177 (wash the site-endpoint) plus
the resistance matrix's completion. Either way, e176's fold
must also report the 129-band content census: does the
consolidated fact's ADDRESS-store decay before or with its
access? (Decay-before would mean the sink-coupled access
outlives its own graft — a dissociation worth having.)

DERIVATION (the resistance matrix — the frame's completion
table, ~14:40Z): two resistance axes, two memory types, half
the cells known. KNIFE axis: sink-coupled = KILLABLE (e160,
flat CE); site-endpoint = UNKILLABLE (e125a, all CE). WASH
axis: dwell-phase = WASHABLE (e161, <50 steps); consolidated
root = e176 PENDING; site-endpoint = NEVER TESTED. THE
GORGEOUS POSSIBILITY: if e176 lands SURVIVES, the two types
are DOUBLY COMPLEMENTARY — each resistant along exactly one
axis (sink-coupled: wash-proof but knife-killable; site-
stored: knife-proof but wash-?) — and unlearning becomes a
two-lock problem where each memory type has a different
exploitable lock. The missing cell (e177, queued): wash the
SITE-ENDPOINT (plain corpus on e151_twodoor's 300-step state
or arm_b) — WASH-RESISTANT would complete the complementarity;
WASHABLE would make site-stores look like scratch memory
(cheap, local, erasable by drift) and consolidate the
hierarchy: scratch -> resistant-under-training -> (maybe)
doubly-hardened. e176 + e177 together finish the table.

FOR THE PAPER: claim 2's phase language converts to the
resistance axis (phases -> degrees of washout-resistance; the
cliff survives as the ACCESS-building fact; e151 re-reads as
washout-plus-rebuilding). The abstract's "bidirectionally
switchable" dies its final death here — the closing direction
was forgetting. THE SAVOR: every memory the lab ever trained
was being secretly rehearsed by the anchor banks in every
subsequent fine-tune — the lab's own protocol was the memory's
life-support, and e161 is the first time anyone turned it off.

## T100 — [R49: the headline noun OVERWRITE-NOT-SHARE STRUCK — it asserts the capacity mechanism the run's own text says it cannot separate (the anchor-contradiction confound); the verdict TEXTURE was correct-by-registration; the F1-side rider readings are floor-ratio artifacts (base 0.0017, leak ~5e-4) — only the F2-side riders stand (N2 spares F2; F2 diffuse)] E154: F1 annihilated under an anchor-confounded protocol (2026-09-28 ~14:00Z)

The two-facts cell returned the strongest possible outcome with
the most important confound: F1 annihilated everywhere (not
merely door-closed — the registered GLOBAL-PHASE clause could
not fire because its own premise, an untouched F1, was
destroyed), F2's graft formed, CE improved, and the damage is
F1-SPECIFIC. Three readings:

(1) NO SECOND-FACT CAPACITY AT THIS BUDGET: the shared
substrate was overwritten. If confirmed by the confound-free
rerun, the "one door" question dissolves into "one FACT at a
time" — the consolidated net cannot hold a second locked-installed
fact without demolishing the first. W012's bandwidth reading
gets its answer the hard way: the bandwidth is ~one fact wide,
and installing into it costs the tenant everything.

(2) THE ANCHOR-CONTRADICTION CONFOUND (the agent's catch, now
the load-bearing caveat): the protocol's incumbent-continuation
anchors are an ANTI-F1 signal — 8 paired contradiction anchors
per batch, 300 steps. F1's demolition may be textbook
unlearning-by-contradiction through the anchor channel, with
the graft an innocent bystander. THE DISCHARGE CELL (e170):
the identical F2 install with NEUTRAL anchors (plain corpus,
no incumbent-continuation windows) — if F1 survives, the
demolition was the anchors (and the two-facts question REOPENS
with per-fact doors); if F1 still dies, the overwrite is real
and capacity is one fact.

(3) THE RIDERS ARE CLEAN GOLD: the knife's circuit-selectivity
replicates on a FRESH site-stored fact (F2 -3.1% under N2) and
W017's coding prediction confirms out-of-sample (F2 diffuse:
top-1 0.139, 28/36 heads — the locked-trained signature on a
second fact, third data point on the concentration law).

FOR THE PAPER: claim 2's "globally" must await e170 (the
demolition's channel is unsettled); the riders strengthen
claims 3-4 as-is.

## T099 — [INVALID-BY-INSTRUMENT per R49 critic — the +0.0000 was a PROMPT-GEOMETRY TAUTOLOGY: the door battery's prompts span positions 0-141; the surgery edits wpe rows 183-189; a causal transformer never reads those rows for those prompts. g-12 was bit-identical to 17 figures EVEN ON THE ROOT'S OPEN DOOR — the dial was structurally blind. DOOR-STAYS-SHUT fired as a foregone conclusion; 'the graft rows carry zero of the closure' is UNSUPPORTED by e166; 'unreopenable-by-surgery' keeps only e153's sub-bar transplant arm (n=1). RULE 12's founding bite, a third time, at the same coordinate 183. The licensed graft-not-closer support remains T097's home-graft counterexample — itself the straddling cell. e166's long-window rerun rides e173's corrected design.] E166: the inverse event (2026-09-28 ~13:50Z)

WHAT SURVIVES OF THE RUN: the site-read cells (surgery visible there: 0.998 -> 0.599) and the head-ablation cells (which DO move the door — reader3-zero 0.0041) are real; the ROW cells and the headline are void. The third-asymmetry noun is WITHDRAWN to a single-arm suggestion pending licensed evidence.

The inverse event returned the cleanest zero the session has
produced: deleting the graft rows kills the graft and moves the
geometry door by +0.0000 — not small, ZERO. Whatever closes the
door, it is not the wpe graft, not actively, not even a little.
Combined with the corrected T097 (home graft + open door) and
the head ablations pushing the door only down: THE CLOSURE IS
SOMEWHERE ELSE — in the 66.5% MLP/LN delta the knife cannot
reach, or in disuse-decay of the ±12 pathway. The fork:
REWRITE (the conversion rewrote the readout's stream state) vs
DISUSE (the pathway decayed for lack of use) — e161's freeze
cell separates them (does the door close under PLAIN CORPUS,
no fact teaching at all?); a gradient re-teach-the-restore cell
would test MLP/LN carriage directly.

THE THIRD GREAT ASYMMETRY, stated: doors OPEN by training
(variance), are KILLED by surgery (N2), and CANNOT BE REOPENED
by surgery (graft removal = +0.0000; transplants nudge to 1/3
bar at best). T037's write-once core now covers all three
directions: no non-gradient write adds function, and no
gradient-built access can be surgically restored once lost.
FOR THE PAPER: claim 2 says "accompanies" (licensed); the
asymmetry triplet joins split custody and the asymmetry of
existence as the third exhibit of the unlearning section. FOR
W018: the BURY fate is not surgically reversible via the tomb's
rows — re-excavation, if possible at all, is a TRAINING
operation (variance at the buried site); the four fates remain
a phase diagram only if training can move between them.

## T098 — [R49 BOUND: the kill was 71%, not 100% — the residual is a 29%-ALIVE readout; the saturated organ dials cannot distinguish storage-support from access-support (same-circuit-at-29%-amplitude predicts the same cells); weight-level intactness is construction-tautological; profile rank 0.587 is moderate. The DECISIVE cell queued as e175: few-step fact-replay on the killed net — fast recovery => thin access lesion; full-budget => substance degraded with access] E164: access severed, substance intact (bounded) — the four-layer model earns its figure, and the MLP is the organism-priced organ (2026-09-28 ~13:35Z)

The R47 critic's circularity charge is answered: each layer is
now defined not only by what removes it but by INDEPENDENT
evidence of separation — the N2 kill leaves the row-0 door at
full strength, the MLP body consuming full headroom, and every
remaining fact-head load-bearing. A surgical readout death with
the storage profile intact is exactly what "layers 1/2
separable" predicted and what a one-layer account forbids. THE
MODEL FIGURE IS LICENSED. The brake-flip texture (band-5 helps
the root, costs the killed net) says the brake is a property of
the LIVE readout configuration — it dies with the access it
modulates.

THE MLP'S DOUBLE ROLE: for the site-fact, the MLP third is the
cheapest full kill (mlp_l5 alone: 92.9%) — the load-bearing
remainder — yet only at organism prices (+0.79). INCORRIGIBLE
IN THE STRONG SENSE: the site-stored memory's un-removability
is not the absence of a kill coordinate but the fact that every
kill coordinate is a vital organ. Combined with e125a: the
site-fact has no flat-CE kill on EITHER surface; the
asymmetry of existence now rests on two exhaustive planes.

FOR THE PAPER: claim 3 gains its separability exhibit (the
post-kill census); claim 4's scope clause strengthens (both
surfaces); the closing sentence's split custody now has its
mechanism diagram — access (severable), storage (intact),
dependence (row-0, untouched by the knife).

## T097 — [CORRECTED per R48 critic — the headline was contradicted by e158's own unread census: locked@band DID re-form a home graft (row 129: brake -0.132 -> content +0.057, site_pos TRUE, peak 129) while the door stayed OPEN; TWO grafts, different door outcomes — the operative variable is SITE NOVELTY (or occupied-slot history), NOT graft formation; 'one event two faces' is FALSE as written] E158: the two-factor gate — closure requires novelty AND zero-variance (2026-09-28 ~12:40Z)

PASS-2 UPDATE (~13:30Z — the agent's committed GPU pass supersedes
the folded CPU numbers): jitter@183 OPEN 0.789 (clean, not
razor-thin); locked@band MID 0.458 (straddles the bar across
passes — second seed owed); the CONJUNCTION reading is robust
across both passes. THE MEMORY EMIGRATES (new texture): under
jitter@183 the home-geometry readout collapses (0.785 -> 0.145)
while novel doors stay open — variance moves the readout's home.
PROCESS RULE: fold on completion notification, not early metrics.

[PASS-1 RESIDUE LABELED per R49: the verdict label and razor-thin paragraphs below predate the committed pass-2 — the current form is in the PASS-2 UPDATE above; committed: TEXTURE, a=0.789 clean-OPEN, b=0.458 MID-straddling. Number fix: the home-graft content is +0.072 (old-census), not +0.057 (a pass-1 value).]
THE HONEST RESTATEMENT: home-site and novel-site grafts dissociate
from door closure. [R49: STRUCK — the committed pass-2 adjudication is
TEXTURE; the CONJUNCTION composite is robust across passes, but
no registered bar fired cleanly. Also: novelty-as-operative-
variable is itself over-minted from two n=1 cells, one
straddling — honest form: graft formation is NOT SUFFICIENT for
closure (home graft, door open); novelty is a CANDIDATE
variable, e165 pending.] What falls
is the MECHANISM STORY layered on top: 'graft formation closes
the door' must read 'NOVEL-SITE teaching closes the door, with
or without a graft.' The novelty axis (distance-from-home vs
occupied-slot vs first-novel-site) is unresolved until e165.
THE FRAME-BREAKING ALTERNATIVE (R48's final line, adopted): the
door may decay by DISUSE while the graft grows by USE — two
independently-trained things, not one machinery seen twice. The
build-travel frame is PROVISIONAL on exactly this; the deciding
cells are already running/queued: e166 (inverse event), e161's
freeze-cell (plain corpus, no teaching — does the door close
anyway?), e154 (different fact). e158's at-boundary honesty:
jitter@183 0.5048 (median 0.408, most prompts below bar) and
locked@band 0.546 — margins 0.005/0.046 against a +-0.029
device bound; 'OPEN' labels are at-or-near-boundary; locked@home
cost 40% of the door ('harmless' was generous — corrected).

The 2x2 landed on its strangest branch, and the strangeness is
the synthesis: neither variance alone nor placement alone
closes the geometry door — NOVELTY AND ZERO-VARIANCE TOGETHER
do, which is exactly the recipe for BUILDING A NOVEL GRAFT.
Re-reading the whole arc through this: e142 (natural placement
-> no graft, row 0 only), e143 (locked@novel -> graft), e147
(variance -> no graft, door opens), e151 (locked@novel -> graft
+ door shut), e152 (the dwell = the graft being built while the
door decays), e158 (jitter@novel -> NO graft, door open;
locked@home -> no NEW graft, door open). ONE EVENT, TWO FACES:
erecting a new positional key tears down the geometry-general
access. The "phases" are not two states of a substrate switched
by a variable; they are BUILD and TRAVEL — the same machinery
seen from the construction side and the access side. FOR THE
PAPER: claim 2's provisional marker clears into the two-factor
form ("closure accompanies novel-graft formation"), which is
STRONGER and cleaner than either simple law; the freeze-cell
(e161) now reads as "does graft-building need supervision to
finish tearing down the door"; e165 (the novelty-axis ladder:
locked at graded distances from home) is the discriminating
follow-up — does closure track distance-from-home (novelty as
a continuous variable) or is it binary at first-novel-site?
RAZOR-THIN honesty: jitter@183's 0.505 vs the 0.5 bar — at-or-
near-boundary; the site census (how much graft tried to form
under ±8 jitter) is the mechanism's decimal.

## T096 — [R48 BOUND: 'no kill set EXISTS' is a 0.03% sample of the pair space (the consolidated kill was a superadditive pair INVISIBLE to singles ranking — the same hiding place unsearched for the site fact); B5/B6 at moderate CE never run; the MLP surface (33% of load) untouched — scope: head-coordinate surgery, sets <= 4; e168 queued: exhaustive 630-pair scan + MLP-neuron ablation] E125a: the asymmetry of existence — consolidation buys generalization AND surgical removability; the locked-in memory can neither travel nor be excised (2026-09-28 ~12:30Z)

The inverted knife returned the strongest possible null: across
92 cells — two sites, both ablation modes, the site's OWN census
ladder, the e160 sets, address heads, and random controls — the
site-stored fact never dropped 60% at ANY CE. Not "harder to
remove": NO KILL SET EXISTS over the head-coordinate surface.
And the two facts' head populations are DISJOINT (L1H2-led vs
L0H3/L1H0-led, sharing only their weakest member), with the
site-fact's ladder SATURATING where the sink-coupled ladder was
SUPERADDITIVE — redundant population coding versus a
complementary killable circuit. T090's circuit-selectivity
inverts into an ASYMMETRY OF EXISTENCE.

THE POIGNANT INVERSION (the paper's strongest unlearning
sentence): the memory that GENERALIZES — geometry-free,
deletion-tolerant, the one jitter built — is the memory you can
surgically remove at CE +0.25. The memory that STAYS PUT —
context-bound, site-locked, the one locked replay built — is
the memory you cannot remove at any price. Consolidation trades
permanence-of-place for portability, and the price of
portability is vulnerability to the knife. For unlearning
practice the lesson inverts the usual fear: the DANGEROUS
memory (the one that generalizes everywhere) is the EASY one to
excise; the harmless-looking localized memory is the
incorrigible one.

FOR T092: layer 2 forks by phase — a killable complementary
circuit (sink-coupled) versus an unkillable redundant population
(site-stored); the four-layer model's readout layer was one
layer too flat. FOR e164 (post-kill census): now also asks
whether the site-fact's MLP third (33% of load) is the
incorrigible substrate.

## T095 — E162: two edges, one pivot — the memory depends on what the sink supplies AND what it spares (2026-09-28 ~12:15Z)

R48 CORRECTION (~13:10Z — the sufficiency claim fails internal
consistency): cell (i) KEPT the poison's front-loaded q/k
absorption at CE -0.0004 — absorbed mass at the REAL profile is
FREE; cell (ii) killed only with a FLAT profile carrying 11x the
poison's L0 absorption. The honest form: HEALTHY CONTENT FULLY
RESCUES (supply edge, matched conditions); TOTAL-DOSE absorption
on a FLATTENED profile kills (allocation edge, mismatched
conditions); the per-layer DISTRIBUTION is untested and is where
the divergence lives (e167 queued: per-layer-matched bias).
The original fold text follows with that qualifier attached. Restoring healthy values under a
poisoned key erases ALL damage (retention x1.000, CE -0.0004 —
the corruption is carried by the content read off the pivot;
power: the poison moves v0 by 1.13x its norm); inflating the
absorber on a healthy row kills just as dead (x0.036 @ g-12 at
the matched dose, monotone) — the corruption is equally carried
by the allocation the absorber steals. The sink holds a DUAL
role for the consolidated memory: SUPPLIER of read content and
GUARANTOR of the attention budget. T093's READ-coupled noun was
half the story — full form: functional dependence on the sink's
dual role, each component separately lethal when broken. FOR
THE PAPER: the mechanism sentence becomes the two-channel form;
e159's double dissociation gains its mechanism completion; T092's
layer 4 = functional dependence (supply + allocation).

## T094 — [R47: n=1 TRAJECTORY — the dwell and brake overshoot carry [n=1] until e152R's 3 seeds; the s16->s32 bounce is the sole substance of CLEAN's failure] E152: the dwell time — the phase transition passes through a MIXED state, and the brake overshoots before it releases (2026-09-28 ~11:25Z)

FREE RE-READ (R47 critic, no compute — from e152's own metrics):
at s32, mask-retention 0.979 while ladder@0.07 kills to 0.19 —
the shelf's ~0.5 IS route survival, not sink health; the dwell
verdict SURVIVES the clean dial.

TRANSIENT-TWO-DOOR resolves T088's fork in the gentlest
possible way: the cliff is real AND the two-door state exists —
for a window. Between steps ~16 and ~64 the net holds BOTH a
genuine site-store (67x control, functionally readable) AND
>=50% geometry retention; then the zero-variance training
consolidates the graft and the shared state abandons the
geometry door. THE MIXED STATE IS A DWELL, not an artifact of
measurement timing: it survives across three checkpoints and a
bounce. FOR THE PAPER: claim 2 gains its most physical sentence
— the phase transition is not a hop but a passage through a
mixed state with a dwell time (~50 steps at this budget);
systems-consolidation language ("gradual transfer") and phase
language ("sudden switch") were both half right: the SWITCH is
sudden (8-16 steps), the SEPARATION is slow (the shelf), and
the memory spends the shelf holding both natures.

THE BRAKE OVERSHOOTS: A(129) intensifies (-0.132 -> -0.466)
mid-conversion before dissolving at the end. Under the
negative-posterior reading (T087), this is the address key
fighting hardest exactly while its replacement is being built —
suppression peaks at maximum competition, then the whole
opposition dissolves when the new phase settles. A tiny,
poignant mechanism: the old address does not fade; it resists,
then is released.

STANDING: the dwell is one trajectory, one seed (point
estimate); e158 (variance@183) still decides whether the
conversion itself keys on variance; e154 decides per-fact vs
global. The GPU is user-occupied — dispatch with park-to-CPU.

## T093 — [RESOLVED MIXED by e162/T095 — BOTH channels kill, each at its own conditions (supply edge matched; allocation edge flattened-profile-only; per-layer distribution untested, e167 queued); the one-channel noun below is historical] E159: READ-coupled — the mask is the health door, and only readers die of the poison (2026-09-28 ~11:15Z)

The R46 critic's mask/ladder contradiction resolves into the
day's cleanest mechanism claim: the poison is delivered THROUGH
attention reads (the manipulation check nails it — a poisoned
row 0 becomes a 10.8x mass absorber whose reads corrupt
downstream computation), and masking key 0 removes the poison
entirely (the joint cell heals even under COMPLETE wpe[0]
removal). The query-side/global-softmax alternative is
falsified. And the site-stored control pays the same organism
price (CE +0.825 vs +0.843) without its fact dying: ORGANISM
DAMAGE IS NECESSARY BUT NOT SUFFICIENT — the memory must itself
read through the sink. THE NOUN, FINAL FORM (pending replication
and e158/e154): the consolidated memory type is READ-COUPLED to
the sink — it dies of what it reads off a degraded pivot, not
of the pivot's degradation per se. T086's removable-to-
irremovable reframe is bounded accordingly: the coordinate is
not unremovable (the mask removes it benignly!) — what is
irreplaceable by surgery is the READ PATH, and what training
built (access) surgery cannot create (e153) though it can sever
(e160). The four-layer model's LAYER 4 updates: dependence =
read-paths through the sink's health; the intervention class is
poisoning OR masking (both organism-tolerable; one kills
readers, one spares all).

## T092 — SYNTHESIS: the four-layer model of a consolidated memory (2026-09-28 ~10:50Z; the day's circuit story, complete enough to draw)

T090 + T091 + T086 + T085 compose into a layered anatomy — four
layers, each discovered by its own intervention class:

LAYER 1 — CONTENT (body organs): the fact's substance lives in
MLPs and general machinery in every phase (e133; additivity
fails — a population). Intervention class: nothing removes it
cleanly; it is substrate.

LAYER 2 — READOUT (the head circuit): a small complementary set
(N2-class {L1H0,L0H0} + L0H3; one suppressor + route suppliers)
carries access — killable by head surgery at flat CE, and the
knife is TYPE-SELECTIVE (spares site-stored facts; e160).
Intervention class: surgical removal.

LAYER 3 — PHASE (the stream state): a distributed MLP-heavy
order parameter above the circuit — switched only by training
(variance opens the geometry door at ANY width; zero-variance
re-teaching closes it); transplant-rigid when open (e147/e151/
e153). Intervention class: re-training only; surgery cannot
create access (T037's write-once core).

LAYER 4 — DEPENDENCE (the sink): row 0's NORM as the organism-
critical coordinate — corruption poisons every read (threshold
(0.07,0.15)); no information route exists (the mask spares;
e150) [e159 decides the coupled-vs-organism bound]. Intervention
class: norm-poisoning — effective and indiscriminate.

R48 AMENDMENT: T096 forks layer 2 by phase (killable complementary circuit vs unkillable redundant population — the readout layer was one layer too flat as drawn). The paper's model figure: four stacked layers (layer 2 drawn forked) with the
intervention arrows that touch each (ablation, head surgery,
re-training, poisoning) and the two phases as horizontal states
of layers 2-3. What each queued cell fills: e158 = whether
layer-3 switching keys on variance per se; e159 = whether layer
4 is memory-coupled or organism-shared; e154 = whether layer 3
is one global state or per-fact; e157 = whether the layers
replicate across families. The BRAKE lives between layers 2-3
(the old address's negative posterior — a phase-state; e149
prices its location).

## T091 — E153: doors open by training and resist surgery — the phase is a distributed, MLP-heavy state with a partial head signature (2026-09-28 ~11:05Z)

The reading map's PHASE-DISTRIBUTED branch fired in substance
(though not its letter — five arms move g-12 > 20%, so "no
head-set moves it" is false; the honest form: NO head-set
SWITCHES it). Three structural facts:

(1) T037's WRITE-ONCE CORE SURVIVES ITS SHARPEST TEST: no
non-gradient write added function — the best transplant
(reader_K3, the re-grown L3H4-class reader + the content heads)
recovers under half the geometry door, and random swaps move the
fragile net comparably. Surgery cannot open what training opens.
The e160 result completes the asymmetry: surgery CAN destroy the
readout (N2 kills at flat CE) but cannot CREATE access. The edit
law holds: subtraction works, addition doesn't.

(2) ASYMMETRIC RIGIDITY is the new texture: the OPEN door
(geometry phase) is transplant-immovable — once wired by
variance training, access resists surgery; the SHUT door
(site phase) is fragile. Rigidity tracks the phase, not the
heads: consolidation, once achieved, is surgically stable —
another sense in which it is a movement into essential tissue.

(3) THE ORDER PARAMETER'S SHAPE: MLP-heavy (66.5% of delta
energy), late-layer-skewed, with a partial head signature (the
re-grown positional reader + content heads — the SAME heads that
kill the fact in e160's N2/E2 sets appear in the best reopening
arm). One circuit, three roles: it carries the readout (killable
— e160), partially carries re-opening (transplantable to a
third — e153), and partially carries the brake. The phase itself
sits above it, in the stream.

FOR THE PAPER: Fig 4 becomes the asymmetry figure (doors: opened
by training, killed by head surgery, immovable by transplant);
claim 2's mechanism section states the distributed order
parameter honestly. The e158 cell (jitter@183) still decides
variance-vs-placement before any of this travels.

## T090 — E160: the surgical surface exists — row-surgery cannot, head-surgery can, and the knife knows the type (2026-09-28 ~10:50Z)

The R46 critic's coin landed on the paper's best figure, and
the result composes the day better than the reading map's best
branch:

(1) THE COMPOSED CLAIM: consolidation moves DEPENDENCE into
organism-critical coordinates (the sink, whose corruption is
fatal — e150) while the READOUT consolidates into a small
attackable head-set (e160: N2 kills at CE +0.25). Row surgery
cannot remove the memory without organism death; head surgery
can, at priced collateral. The removable-to-irremovable sentence
completes: the memory moved from row-removable tissue into
head-attackable circuitry behind an unremovable coordinate.

(2) THE KNIFE IS CIRCUIT-SELECTIVE [R47/R48: not type-selective — install dies too]: the same N2 coordinates kill the
install-phase fact (67%) but SPARE the site-stored fact (<=
10.6%). Head surgery is TYPE-SELECTIVE — the two memory phases
(and the install state) share a readout circuit that the
site-stored memory does not use. This is the cleanest
single-figure dissociation the taxonomy has: not deletion
hierarchies, not geometry cells — one knife, two outcomes.

(3) L0H3 DEMOTED, N2 PROMOTED: the 'fact-specific' head was the
symptom, not the circuit — {L1H0, L0H0} (neither fact-specific
by e133's table) carries the kill at lower cost. STRONG
SUPERADDITIVITY (81% vs 34% summed): the circuit is COMPLEMENTARY
— consistent with one suppressor + route suppliers — a
two-component motif worth naming in the paper.

(4) W014's ordering graduates: heads > route >> band is now a
DEMONSTRATED capability with collateral prices (NLL3 +0.26), and
e125's design collapses to pricing the N2-class surface on the
phase battery. The bias-check (R46's checklist): this fold
STRENGTHENS a claim — the flipped question is asked: the cell
that would flip it is the site-stored circuit's OWN kill set
(does a symmetric site-stored-killing head-set exist at flat
CE? — queued as e125's first arm), and the lineage replication
(e157).

## T089 — E146: the self-instrument does not travel — and that is data (2026-09-28 ~10:25Z)

The dissociation matrix could not run: the e111-lineage self/
other battery fails at baseline on the B43 line (foreign gap
0.018 vs 1.0). Read carefully, the null carries information:
the lab's self-recognition findings are LINEAGE-INDEXED — the
binary step, the k*=7 signature, the exclusion all live on the
nets they were measured on, and the instrument does not just
transfer. Two live readings: (a) the B43 line genuinely lacks
the self/other margin (families differ in whether they verify —
e108's binary two-cluster step was family-typed at 0.40-vs-0.14;
maybe B43 sits below it); (b) the battery's donor/null
conventions are calibration-sensitive (the 1.0 gap bar was tuned
on the home line). DISCRIMINATOR (cheap, queued as the e146
repair): run the SAME matrix on the e111 HOME lineage's nets
(they are on disk) — if the battery works there and fails on
B43 under matched conventions, reading (a) strengthens and
self-recognition joins the family-typed physics; if it fails
both, the battery's conventions need recalibration and W015
stays parked. W015 marked; e156 BLOCKED. THREE TEXTURES from the final report
(~10:55Z): (1) this lineage's anchor readout is GRADED BY CONTENT
PLAUSIBILITY, not binary by generator identity — foreign and even
ancestral V splice in at ~zero cost (gaps +0.018/+0.057, healthy)
while plausible-but-wrong corpus V costs +0.766: the B43 line's
'self-check' is a PLAUSIBILITY GATE. If e146b confirms the home
lineage still gates by identity, the lab earns a lineage-level
dissociation: some families verify IDENTITY, some verify
PLAUSIBILITY. (2) The geometric occupancy separation survives
EVERY intervention (3.0-5.5x across all cells, including the
perm wreck at CE +2.05) — the SELF-INDEPENDENT pattern at the
geometry level, unlicensed by the failed functional instrument.
(3) PERM DRAW-DEPENDENCE CAVEAT (touches T081/W014): a second
perm draw killed the fact (x0.000 @ CE +2.05) where e141's draw
cost x0.79 @ +0.70 — direction-scramble outcomes vary by draw;
the +4.1% spare was ONE draw, and any perm-based claim carries
this until multi-draw CIs exist.

## T088 — [AMENDED per R47: e152 found the conversion passes a ~50-step mixed state (T094) — the 'per-net' reading is itself pending e158 (variance-vs-placement) and e154 (global-vs-self, same-fact confound); 'at ANY site' below is untested beyond 183] E151: the cliff is PER-NET — memory type is a global phase of one substrate, and the transition runs BOTH WAYS (2026-09-28 ~10:10Z)

The committed prediction failed honestly and the failure is the
day's cleanest structural statement: ONE locked re-teach
converted a sink-coupled, geometry-general, deletion-tolerant
memory into a site-stored, geometry-bound one — growing the new
site (+0.512) while closing the geometry door EVERYWHERE (g-12
0.916 -> 0.102, D-all 0.905 -> 0.098) and dissolving the brake
(-0.132 -> +0.005), at IMPROVED corpus CE. Consequences:

(1) THE TYPES ARE PHASES, NOT SUBSYSTEMS. "Two memory types"
becomes "two phases of one memory system" — the taxonomy's
lineage confound (R45's attack) resolves the strong way: one
net, both phases, bidirectionally switchable (variance opens the
geometry door and kills the graft; zero-variance re-teaching
grows a graft and closes the door). For the paper this is
STRONGER than two-doors would have been: a reversible phase
transition, cliff both ways, at fixed wiring — the read policy
(W013's protagonist) is the order parameter.

(2) WHY GLOBAL? The geometry-generalization cannot be a property
of one fact's private circuit (a private circuit would survive
another fact's locked training); it must live in a SHARED state
— the net's read configuration as a whole. Locked replay at ANY
site drags the global policy back toward position-keyed reading.
The 8-step smoke (g-12 still 0.99) says the drag has a TIMESCALE
— the conversion lives somewhere in 8-300 steps.

(3) THE TRANSIENT: if a two-door state exists mid-conversion,
the P-b prediction is not wrong but EARLY — doors add
transiently, then the zero-variance training consolidates the
graft and the shared state abandons the geometry door. The
TIME-TRACE (g-12 retention vs re-teach steps, 8/16/32/64/128/300
— dispatched as e152) locates the conversion point and tests
whether the transient is real: monotone decay (no plateau) =>
clean conversion; a plateau with both doors followed by decay =>
the transient two-door state exists and the cliff has a dwell
time.

(4) THE BRAKE VANISHES WITH THE PHASE: A(129) -0.132 -> +0.005 —
the negative posterior (T087's mechanism note) is not a permanent
scar but a STATE of the variance phase; re-locking releases it.
e149's anti-alignment prediction sharpens accordingly: the
anti-alignment should exist only in the variance phase.

## T087 — E147: the cliff — variance is a SWITCH, not a dial; the type decision is binary at zero-vs-any (2026-09-28 ~09:50Z)

The registered graded law did not fire; what landed is cleaner:
THE MEMORY TYPE IS DECIDED BY A CLIFF. A(w) at w=1 is already
negative (-0.029 vs L's +0.327) and never trends with width;
NR(w) onsets at w=1 (0.696 vs L's 0.071) and plateaus. ANY
position variance — the name moving across as few as three read
rows — is sufficient and (within 300 steps) saturated. T079's
invariance law returns in STEP-FUNCTION form: variance doesn't
GRADE the competition; it OPENS it. The biology echo sharpens:
systems consolidation in the literature is discussed as graded
transfer; here the DECISION to transfer is all-or-none at the
moment the positional key stops being perfectly predictive.

MAGNITUDE BOUND (R46 critic — the honest form): the step's
core is the SIGN and the NR onset (both ~10-20x their controls);
"dies" and "no width trend" overran the numbers — the w64 A
endpoint is 20% of w8's (5x rung scatter, single seed, zero
cross-seed variance on this dial), and NR falls 0.903 -> 0.605
from w8 to w64 (W010's ghost half-alive). Bound: the key's
positive load (<= +0.327) is abolished; what replaces it is a
WEAK negative (|A| <= 0.13), sign-robust, magnitude unresolved
pending 2-3 re-seeds.

THE ROADS DIVERGE IN OPPOSITE DIRECTIONS FROM THE FIRST RUNG:
locked replay strengthens the trained-geometry key (+0.327)
while ERODING novel-geometry expression below the untrained root
(0.071 vs 0.166) — massed replay doesn't just fail to
consolidate; it actively CONTRACTS the memory's reach.
Variance-replay does the exact opposite on both dials. Two
opposite developmental trajectories from one binary switch.

E150'S CAVEAT APPLIED: NR is now read as sink-HEALTH dependence
(poisoning semantics), and NR is bounded by g-12 expression
existence — the cliff's NR-side conflates 'expresses at novel
geometry' with 'depends on sink health there.' The A-side
(address-key death at w=1) is clean of both caveats. The honest
composite: variance switches OFF the address key (clean) and
switches ON novel-geometry expression that is sink-coupled
(caveated). FAR-ROUTED-TAIL's fire (x0.069) says e143's FAR
boundary case resolves toward coupling.

MECHANISM NOTE (the negative-posterior hypothesis, worked on
paper ~09:38Z): why does variance make the address key go
NEGATIVE (a brake) rather than merely shrink? Because once the
fact appears at other rows, the address key's firing ANTI-
correlates with the fact's presence there — when the fact sits
at 121, a read keyed on 129 carries other content; the address
becomes a below-prior predictor, and suppressing it sharpens the
attention budget. The pure Bayesian form predicts brake
MAGNITUDE tracks the miss-rate (w=1 misses 2/3; w=64 misses
~127/128 -> much stronger brake) — but the data show a plateau
(-0.03..-0.13, no width trend): suppression SATURATES once the
key is unreliable at all. The cliff again, now inside the brake:
demotion is all-or-none, not graded. Signature already visible
in e147's A-column; no new compute needed to see it.

READING FORK FOR e151 (pre-registered before its report): if
TWO-DOOR-ADDITION fires, the cliff is PER-MEMORY — each fact's
type is decided by its own training variance, and one net holds
mixed types. If ROUTE-OVERWRITES fires, the cliff is PER-NET —
any zero-variance teaching collapses geometry-generalization
globally, implying a shared substrate the locked re-teach
destroys. SITE-REJECTED would mean the geometry-general
structure absorbs zero-variance teaching without growing a site
— the cliff ran once and cannot run again on this net.

FOR THE PAPER: claim 2's switch upgrades from 'error-position
variance' to 'a binary cliff at zero-vs-any variance' — simpler
to state, stronger to show (one figure, two rungs). The
mechanism hunt for WHAT variance does (why does one moved
window cancel the address key?) reopens at the head level
(e133's L0H3 class) with the P-b cell (one net, both types) as
the taxonomy's last structural confound — dispatched next.

## T086 — E150: the route was never information — sink-HEALTH, poisoning, and the removable-to-irremovable reframe (2026-09-28 ~09:35Z)

The reading map's ALL-KILLS-WRECK branch fired, and the
pre-registered reframe is the fold — but e150 added a MECHANISM
the map did not anticipate:

**(1) THE ROUTE WAS NEVER INFORMATION FLOW.** Blocking ALL
attentional access to position 0 (the mask) spares the fact at
CE +0.03 — across all three net types. What kills under wpe[0]
removal is POISONING: below-norm row 0 becomes a value-less mass
absorber that corrupts every downstream read (the sink's health,
not its signal). The threshold is sharp: norm 0.07 kills (CE
+0.84), 0.15 survives (CE +0.31). The pre-mask sink attention at
the fact's read position is ~0.007/layer — the 'route' carried
essentially no fact traffic. T081/T082 are HARD-BOUNDED as
registered: 'routed through row-0 presence' is now SINK-HEALTH
DEPENDENCE — the memory requires the organism-critical
coordinate to be intact, the way any organ requires the blood
supply.

**(2) THE REFRAME, ACTIVATED WITH MECHANISM: consolidation is a
movement from the REMOVABLE to the IRREMOVABLE — into machinery
whose integrity is organism-critical.** The graft (row 129) was
surgically deletable at zero collateral; the consolidated memory
rides the sink whose poisoning wrecks everything. This is not a
retreat from the day's findings; it is their synthesis — though
E159 BOUNDS IT FURTHER: the coordinate is not unremovable (the
mask removes row 0's contributions benignly — every fact
survives); what is irreplaceable by surgery is the READ PATH,
and only memories that read through the sink die of its
poisoning (the site-stored fact pays equal organism damage and
lives). The dependence is READ-coupledness, not organ-essentiality.

**(3) WHAT SURVIVES OF THE TAXONOMY:** the TYPES are real but
renamed — SITE-STORED vs SINK-COUPLED (was 'routed'): the type
differences (novel-geometry 0.914 vs 0.002; D-all tolerance; the
brake sign) stand on e139/e143's cells; the CARRIER of the
sink-coupled type's geometry-generalization is now most plausibly
the fact-specific HEADS (e133's 84.5% residue) reading content —
the content-keyed alternative W011 buried gets its revenge. The
L0H3-zero near-miss (58.6% at CE +0.21, 1.4pts under bar) is the
leading candidate for the true fact-circuit — one cell away from
the flat-CE kill the frame wanted.

**(4) W014 SURVIVES ITS CONTROL, WITH TEXTURE:** fact-at-position-
0 scramble IMPROVES the fact (x1.589); direction-insensitivity
at the 129-read holds; but perm kills short-horizon reads
(col-12 control x0.042) — direction-consultation is READ-HORIZON
dependent. The tenant's insurance-policy metaphor gains a clause:
the fact ignores the pivot's direction ONLY from far away.

MAP UPDATE under the corrected frame (~13:00Z — registered
before e154/e166 data): with graft-formation demoted and the
DISUSE alternative live, the running fleet's outcomes re-read:
- e166 (inverse event): DOOR-RESTORES => COMPETITIVE INHIBITION
  (the novel-site structure actively suppresses the door) — the
  strongest mechanism reading. DOOR-STAYS-SHUT is now AMBIGUOUS
  between destructive rewrite and DISUSE-DECAY (removing the
  graft cannot restore an unused pathway) — e161's freeze-cell
  becomes the unique separator.
- e154 (two facts): GLOBAL-PHASE (F1's door closes under F2's
  novel-site teaching) is consistent with BOTH competitive and
  disuse (any off-F1-distribution training may disuse the ±12
  reads); PER-MEMORY (F1 survives) would kill the disuse
  reading's simplest form and support per-fact protection.
  Neither alone separates the mechanisms; the FREEZE-CELL
  (plain corpus, NO fact teaching) remains disuse's unique test
  — e161 re-registers AFTER e166's verdict, per the R48
  ideator's honesty note.
- The three-way fork for the fold: COMPETITIVE (e166 restores +
  e154 global + e161 open-under-no-teaching) / REWRITE (e166
  shut + e161 shut-under-no-teaching) / DISUSE (e166 shut +
  e161 shut + the door re-openable by F1-variance retraining
  alone — e155's cell answers this last clause).

E158 + E125a READING MAPS (registered ~11:48Z, before their
data — both mid-compute):

E158 (variance x site 2x2):
- PHASE-BY-VARIANCE: the cliff's causal clause CONFIRMS —
  variance, not placement, is the switch; the paper's claim-2
  provisional marker clears; the freeze-cell (e161) becomes the
  dwell's stability test under the confirmed law.
- CLOSURE-BY-PLACEMENT: T088's phase claim INVERTS to a
  placement law (any new-site teaching closes the door) — the
  day-six postscript and paper claim 2 rewrite; the cliff
  becomes a site-selection phenomenon, not a variance one;
  e147's A-dial re-reads as site-competition, not invariance.
- SITE-INDEPENDENT: the strangest branch — locking the HOME
  site is harmless; closure requires NOVELTY + zero-variance
  together; a two-factor gate (novelty AND no-variance), which
  no current law predicts cleanly and which would demand a
  novelty-axis ladder (e165: locked at graded distances).

E125a (the inverted knife):
- SITE-KNIFE-EXISTS: symmetric surgery — one knife per type;
  the circuit dissociation COMPLETES as a two-circuit story
  (the paper's Fig-2 pairing becomes fully symmetric).
- NO-SITE-KNIFE: the types differ in REMOVABILITY — the
  deepest form of the split-custody claim (one memory type is
  surgically erasable, the other is not) and arguably the
  paper's strongest unlearning sentence; also predicts e125's
  collateral framing shifts (the site-stored fact is only
  removable by wreck).
- Bias-check (standing rule): whichever fires, the fold must
  ask for the cell that would flip it (EXISTS -> is the site
  knife narrow (few sets) or easy?; NO-KNIFE -> is absence
  power-bounded (the census's max single drop 20% sets the
  prior)?).

E160 READING MAP (registered ~10:39Z, before its data — the
critic's one-cell-away experiment):
- FLAT-CE-FACT-KILL (head-set >=60% at <= +0.35 CE): the
  surgical surface EXISTS. Claim 4 composes: consolidation moves
  DEPENDENCE into organism-critical coordinates (the sink, whose
  corruption is fatal) while the READOUT consolidates into a
  small attackable head-set — row surgery cannot remove the
  memory without organism death, head surgery can. The unlearning
  ordering (heads > route >> band) graduates from prediction to
  capability; Fig 2 gains its killer point; e125's remaining
  design work is pricing collateral on the winning set.
- NEAR-MISS-CONFIRMED (best 40-60%): the frontier stays open;
  the paper reports the dose-response; claim 4 keeps the
  'corruption-fatal; surgically-attackable unknown' form.
- WRECK-ONLY (every >=60% cell at CE >= +0.70): the reframe
  holds bounded; the noun architecture survives; claim 4 as
  written. NOTE the bias-check question (R46's checklist): a
  WRECK-ONLY fold must ask whether it BOUNDS the claim (safe)
  or STRENGTHENS it (watch: 'irremovable' language creeping
  back beyond the corruption-fatal form).

E153 READING MAP (registered ~10:18Z, before its data):
- PHASE-IN-HEADS would be the first NON-GRADIENT PHASE EDIT in
  lab history — T037's write-once-core claim ('no working
  non-gradient write ever ADDED function') faces its sharpest
  test: a head transplant that reopens the geometry door adds a
  FUNCTION (geometry-general reading) by surgery. If it fires,
  W013's never-edited component (the read policy) gains its edit
  interface, e125's unlearning ordering gets its mechanism, and
  the paper gets Fig 4. If it fires only in the CLOSING
  direction (e151->consolidated transplants close the door) but
  never opening, that asymmetry is itself the finding — doors
  close by surgery but open only by training (a ratchet).
- PHASE-DISTRIBUTED re-aims the noun: the read 'policy' becomes
  a global MODULATOR (LN/MLP-stream state), not a routing
  circuit — e135's LN-causality variant gains the sharpest
  question it has ever had (does the LN state carry the phase?),
  and W013's protagonist changes substrate.

STANDING QUESTIONS: e147 (in flight) now measures the SWITCH
between types without a route mechanism to explain it — if
INVARIANCE-CAUSAL fires, the switch is real and the mechanism
hunt reopens at the head level; e146's matrix gains sharper
probes (mask vs poison vs perm columns — the self's row can now
distinguish health-dependence from information-dependence); the
P-b cell (one net, both types) rises in priority as the
taxonomy's last confound.

## READING MAP — registered before e147/e150 report (2026-09-28 ~09:08Z; both mid-compute, no data read)

The two running experiments carry the frame's two load-bearing
questions. Their outcomes are pre-mapped so the folds are shaped
before numbers exist:

**e147 (width ladder) — what each bar would MEAN:**
- INVARIANCE-CAUSAL (A(w) monotone decreasing, NR(w) onsetting
  within one bin of the same w*): the credit-competition law
  returns AS A DOSE-RESPONSE LAW — the taxonomy gets its switch
  variable, T079 is redeemed from its dial-death, and the second
  paper's claim 2 becomes quantitative (a critical width w*, not
  a binary).
- DEAD-AGAIN (A flat, or routing onsets while A >= +0.15): two
  independent switches — the taxonomy survives as TYPES but its
  mechanism dies for good; W013's protagonist loses its key-
  selection rule permanently.
- SEED-COVERAGE (routing peaks at +-8, collapses at +-32/64):
  W010's ghost wins POSTHUMOUSLY — seeds have finite spatial
  reach, and 'omnipresence' acquires a spatial constant (the
  seed-reach radius). A new number either way.

**e150 (flat-CE route test) — the stakes, and the reframe if it
goes negative:**
- FLAT-CE-ROUTE: the route is isolable from wreckage; T081/T082
  unbound; the day's noun stands.
- ALL-KILLS-WRECK: T081/T082 HARD-BOUNDED — but the honest
  reframe is already worth savoring: 'the fact's readout
  requires the net's most load-bearing coordinate' is itself a
  finding — CONSOLIDATION AS A MOVEMENT FROM THE REMOVABLE TO
  THE IRREMOVABLE. The graft (row 129) was surgically deletable;
  the native organ (row 0) is unremovable without organism
  damage. What the lab called migration would then be: the
  memory moving from editable tissue into essential tissue —
  the OPPOSITE of surgical memory, and arguably the point of
  consolidation. The bio-echo sharpens: childhood memories
  resist erasure partly because they live in early, load-bearing
  circuitry. If ALL-KILLS-WRECK fires, this reframe — not a
  retreat — is the fold.
- DIRECTION-CONSULTED (fact-at-position-0 scramble kills): W014's
  tenant dies; direction-independence was an artifact of the
  fact never living at position 0.
- PRESENCE-AT-NOVEL (perm@g-12 spares >=80%): presence-only
  extends to novel geometry — the strongest single supporting
  cell the frame can add.

Adjudication on the registered bars verbatim; no shopping.

## T085 — E142: the address was never born — row 0 carries every install, and 'address' was the protocol's artifact (2026-09-28 ~09:30Z)

ROW-0-ALWAYS, 13/13 nets, every dose. Two consequences, one
historical and one structural:

(1) THE HISTORY REWRITE: there was no address-era. The five-day
narrative — install binds an address, consolidation migrates to
a field, the field re-keys to row 0 — compresses at every stage
to: ROW 0 CARRIED THE MEMORY ALL ALONG, and everything the lab
called migration was share-redistribution around a constant
row-0 core. W011's savor (c) is LAW: consolidation promotes the
largest existing seed. The install's apparent address-binding
(row 129) was real but SECONDARY — a protocol-made co-carrier
that never exceeded row 0's share at any dose.

(2) THE PROTOCOL-MADE ADDRESS: direct/natural-exposure installs
put essentially everything on row 0 (row 129 NULL) — the address
row forms ONLY under masked-replay installs, where the protocol
fixes the fact's position-variance to zero at 129 and the credit
lands there (T079's revived law, now visible in INSTALLATION
too: error placement chooses the store, and natural exposure
places its error at the omnipresent row). The taxonomy's
'site-stored' type is therefore PROTOCOL-SCULPTED: lock the
position, grow a site; let it vary (or let nature place it), and
row 0 takes everything. This unifies e143 (NEAR site-stored
under locked replay) with e142 (natural installs row-0-only)
under one rule with no residue.

CONNECTS: the fresh 0.84M family's rel-1.000 installs explain
T069's 6/6 row-0 content-carrying — it was never a coincidence
of the B43 line; it is the architecture's default. OPEN: the
trained-geometry dial caveat (T083) bounds this census too — the
claim is 'row 0 carries install EXPRESSION from birth', with
routing-level (novel-geometry) replication still owed; e150's
flat-CE cells and e147's ladder are the instruments that will or
won't hold the story together at the routing level.

## T084 — E143: the compass is causal; invariance survives by intervention what it lost by census (2026-09-28 ~09:00Z)

The committed prediction (COMPASS-CAUSAL, registered 07:58Z
before dispatch) HELD. Three load-bearing consequences:

(1) T076's compass is now CAUSAL, not observational: parking the
fact's error at positions 5-13 built a site-store AT 5-13 with
row-0 presence at baseline. Error placement chooses the storage
site — a steering wheel, not just a description. The second
paper's claim 1 upgrades to interventional.

(2) INVARIANCE's strange double life, resolved: T079 died on its
registered observational dial (T083, saturation), but its
SUBSTANCE just passed the causal test — zero position-variance
at a sink-ADJACENT site produced no route. Proximity was the
last alternative to invariance for key-selection, and it is
dead. The law's remaining unknown is the WIDTH dose-response:
does address-key death co-occur with route birth as width grows?
PRE-REGISTERED (e147, BEFORE dispatch): the width ladder w in
{1,2,4,16,32,64} from the same root, co-measuring address-key
strength A(w) (row-129 replacement delta) and novel-geometry
routing NR(w) (row-0 presence at g-12 + one more novel
geometry). INVARIANCE-CAUSAL fires if A(w) is monotone
decreasing (Spearman <= -0.8), crosses <= 0 at w*, with NR(w)
onsetting (>= 2x install baseline) within one bin of the same
w* — the co-onset of address-key death and route birth is the
causal joint. DEAD-AGAIN if A(w) flat, or routing onsets while
A(w) still >= +0.15. SEED-COVERAGE rider (W010's ghost): if
routing peaks at +-8 and collapses at +-32/64, seeds have finite
reach; T079-pure predicts +-64 routes at least as well as +-8.

(3) FAR's hybrid texture is the invariance law's first boundary
case: locked at 137-143 — overlapping the install's own band —
FAR shows novel-geometry expression 0.249 where pure NEAR shows
0.002. Either the band population supplies de facto support
diversity (invariance loophole: overlap counts), or the 0.249 is
the old band field expressing (content-carried). The free cell
(d_r0@g-12 on e143_far) adjudicates: FAR-ROUTED-TAIL vs
CONTENT-TAIL. Registered before running.

ALSO REGISTERED (the dream confound, honest): the R45 ideator
found that e139's dream harvest used 130-char prompts, which
place every first continuation token at x-col 130 BY
CONSTRUCTION — the '33/34 at the old address' savor is partly
rig geometry. The claim is CONFINED until the randomized-length
dream-topology census runs (queued as e148): ADDRESS-SEEKING
(install/locked nets concentrate onsets near 129 at >=5x
uniform, p<0.01) vs ROUTE-DISSOLVES-ADDRESS (routed nets flat)
vs ROUTE-KEEPS-AN-ADDRESS (a generation address distinct from
the read address — a new object if real). The erosion mechanism
(fixed-geometry replay never varies position) survives either
way, but DAY_SIX_REPORT's phrasing is downgraded to match.

## T083 — E140: the law that died on the wrong dial — T079 killed as registered, T078 retired, and the instrument lesson that saves the taxonomy (2026-09-28 ~08:40Z)

Three adjudications, all honest:

**(1) T079 (credit-assignment) is DEAD ON ITS REGISTERED DIAL.**
GRADIENT-VOLUME fired: R@150/L@150 presence ratio 0.745 — locked
replay is MORE row-0-dependent than jitter. No bar shopping: the
law's registered prediction failed. But the INSTRUMENT LESSON is
load-bearing: the twin starts at rel 0.98 — presence-dependence
at the TRAINED geometry measures sink-load (which every readout
has), not routing. The route-isolating dial is presence-
dependence at NOVEL geometry (e141's g-12 collapse, which this
run did not measure). The law died on a dial that saturates for
everyone. ITS UNREGISTERED SIGNATURE SURVIVES IN THE SAME RUN:
row-129 address-key strength — L 0.327 > E@c3 0.280 > twin
0.241, but R NEGATIVE (-0.21/-0.13). Only the position-varied
road NEGATES the address key; locked and erased both STRENGTHEN
it. That is exactly what an invariance competition would
produce — but it was not the registered bar, so it is TEXTURE,
and reviving T079 requires a NEW pre-registered test (address-key
strength vs jitter-width ladder, novel-geometry presence
co-measured), not a reinterpretation.

**(2) T078's 'erasure digs in' is RETIRED.** L-cycled (no
erasure, just three locked cycles) thins 0.131 -> 0.0045 ->
0.0003, same-or-faster fold than E. Thinning is CYCLE DAMAGE.
The two-roads story in its final form: at matched expression,
position-varied replay produces ROUTED, deletion-tolerant,
geometry-general memories; locked and erased roads both produce
site-stored ones that degrade under cycling. There is no
erasure-specific phenomenon beyond the damage it shares with
any repeated fixed-position intervention.

**(3) THE T082 DERIVATION ADDENDUM'S SYLLOGISM FAILED — recorded
as asserted-then-falsified.** R45 AUDIT DOWNGRADE: the addendum
was written contemporaneously (script mtime 08:28:39Z, metrics
on disk 08:27:23Z, never read by the lead before the agent's
report), but its first git appearance is the fold commit
(08:31Z) — AFTER the data commit (08:28Z). "Registered" implied
commit-before-data and that is NOT satisfied; the claim is
ASSERTED, UNPROVEN ordering (mitigant: the prediction failed and
was recorded as such — fabricators do not pre-register
failures). Content: the taxonomy predicts E-flat + L-flat +
R-rising on the row-0 dial. The data: everything flat-high (rel
0.84-0.98), R non-monotone. The taxonomy itself survives — its
discriminating evidence was never this dial; it is e139's
183-geometry cells (routed 0.84-row-0-dependent vs splice 0.25)
and e141's novel-geometry collapse. But the failure teaches the
taxonomy's boundary: ROUTING is invisible at the trained
geometry, where the sink carries everything; it shows only where
the memory must travel. What you measure WHERE matters more than
what you measure.

SECOND AMENDMENT (R45 critic — accepted, ~09:20Z): THE
CATASTROPHE-REGIME CONFOUND. The lab owns NO row-0-plane
intervention that kills the fact at flat CE — every killing cell
sits at CE +0.70 to +4.44 (zero 1.40, mean 2.00, d_r0@g-12 1.36).
'Routed through row-0 presence' and 'dies whenever the net dies'
are observationally equivalent in every measured cell except the
direction-perm spare — which is itself indistinguishable from
'the fact never consults row-0's direction' (its onset never sits
at position 0; no head is sink-adjacent >= 0.25). ALSO: the
taxonomy's discriminator crosses lineages and trained-vs-novel
status (no single net holds both types; P-b and splice-at-novel-
geometry unrun) — 'two memory types' vs 'two training protocols'
hangs on e147 + e150. The cures are dispatched as e150: perm at
novel geometry, forced-off-sink, L0H3-class head ablation (the
only flat-CE fact-kill candidate), fact-at-position-0, norm
ladder. Until e150 lands, ROUTED carries this bound explicitly.

STANDING: e143 (in flight) now carries the invariance question's
last causal stand — NEAR vs FAR decides whether position-variance
is necessary for routing by INTERVENTION rather than census. If
NEAR routes, proximity wins and the invariance story ends; if
NEAR stays site-stored, invariance survives its observational
death.

## T082 — [TYPE RENAMED by e150/T086: ROUTED -> SINK-COUPLED; 'body-stored' -> content-in-heads/body; see T088: types are PHASES] E139: two memory types — sink-coupled vs site-stored — and the dreams savor (bounded) (2026-09-28 ~08:15Z)

The taxonomy completes, and it is cleaner than any card
predicted. The splice arms (error at a fixed novel site) produce
SITE-STORED memories: row-183 content-positive at ~1000x
controls, D-183 kills half, row-0 NULL by content test — and
these memories GENERALIZE to novel contexts and val-split
(0.6-0.7) despite being site-stored. The jitter road (error
position-varied) produces ROUTED memories: row-0 presence-
keyed (e141), reading at geometries never trained (0.660 at the
183-geometry, ratio 0.84 row-0 drops) — position-invariant
access to body-stored content (e133: 84.5% head residue). ROW 0
ROUTES; THE SITE STORES. The two doors compose super-additively
in the splice arms (D-row-0+183: -78%/-62%) — a splice memory is
mostly site-read with a weak routed tail (-25%).

CONSEQUENCES: (1) T075's provisional marker RESOLVES — the
retirement stands (the 'position diversity ingredient' framing is
dead; the splice arms learned, stored, and generalized at their
error site), and the credit for what position diversity ACTUALLY
does moves fully to T079: diversity decides WHICH TYPE of memory
forms (routed vs site-stored) by deciding which features are
invariant across the error windows. (2) The brake's scope is now
precise: a scar of RE-ROUTING (absent on splice arms, present on
the jitter line — deleting the old address only helps a memory
that moved its route). (3) W011's savor (a) resolves PARTIAL:
site-stored facts DO generalize across CONTEXTS — what was never
tested is novel-GEOMETRY for the splice arms (fact shifted off
183); the routed fact generalizes across geometries (0.660 at
never-trained 183). Novel-geometry-for-site-stored is the one
missing cell in the taxonomy; prediction: it fails or degrades
steeply (a site-stored read needs its site), which would make
geometry-independence the ROUTED type's exclusive property.

DERIVATION ADDENDUM (written ~08:28Z, BEFORE e140's report —
its metrics are on disk unread): THE TAXONOMY RE-DERIVES THE
E/L/R TRIANGLE AND DISSOLVES T078'S CONFOUND FROM FIRST
PRINCIPLES. E (erase) and L (locked) matched on every loaded
outcome because they are BOTH SITE-STORED roads — erasure and
massed repetition are different recipes for the same memory
type. R (jitter) differed from both because it is the only
ROUTED arm. The R-vs-E contrast was never erasure-vs-replay; it
was routed-vs-site-stored — which is T079's variable all along.
This makes e140's registered bars a syllogism test: the taxonomy
PREDICTS E-NEVER-KEYS and L-flat (site-stored roads never grow
the route) together with R-ROW0-MONOTONE (the routed road does).
If e140 instead shows E or L growing row-0 dependence, the
taxonomy has a hole; if all three are flat, R's routing was a
one-off and T079 dies with it. Also derivable, for e134's F2:
a second fact jitter-consolidated into the same net should ALSO
route (route-generic presence-keying) and the two routed facts
should share the pivot's bandwidth (W012's combined-54 test).
ZERO-COST PREDICTIONS REGISTERED: (P-b) a routed memory re-taught
at a new site acquires a site-store WITHOUT losing the route
(two-door addition, untested, cheap); (P-c) erase-cycling a
ROUTED memory should NOT dig in (the route protects) — if it
still digs in, "erasure digs in" is damage, full stop, and
T078's demotion becomes a retirement.

THE DREAMS THAT DREAM IN COORDINATES (the arm-c rider, the
day's best savor): 33 of 34 dream ZEPHYRAs sit at x-col 130 —
read position 129, the OLD ADDRESS. The net's spontaneous
replay visits its fact in the fact's own coordinates; dreams are
never position-diverse. Under T079 this is exactly why verbatim
dream replay cannot consolidate (zero position variance -> the
positional key keeps the credit -> nothing re-routes), and the
rider adds the sharper number: dream replay left the fact BELOW
base at the dream positions (0.230 vs 0.391) — replay without
error doesn't just fail to consolidate, it ERODES. T074's dream
verdict stands; T076's compass survives its last open edge; and
e136's surprisal prediction now has a mechanism to explain the
500x: dreams protect by SLOWING EROSION at the address they
never leave, not by moving anything.

## T081 — [HARD-BOUNDED by e150/T086: 'routes through' is sink-HEALTH dependence (poisoning), not information flow; presence claims rest on the perm/halfnorm/mean riders] E141: presence, not content — the fact needs row 0 EXISTING (2026-09-28 ~08:05Z)

The R44 critic's crack is confirmed and sharpened beyond it.
"Re-keyed to row 0" is formally dead: removing the entire
consolidation delta from wpe[0] costs NOTHING (x0.999, CE flat),
and even scrambling row 0's DIRECTION leaves the fact intact
(0.817, +4%) while degrading the corpus (+0.70 CE). What kills
is only REMOVAL (zero/mean/near-removal) — and rows 2-7 plus a
norm-matched random row are all cheap. The noun, corrected: the
consolidated fact is ROW-0 SINK-ROUTED — its readout weights
need the pivot to EXIST (norm >= ~0.38 suffices), not to say
anything. Three consequences:

(1) T077's second amendment CONFIRMED and SHARPENED: the
migration wrote nothing into the destination row; everything it
wrote lives in readout weights (e133: heads 84.5% of the
fact-specific residue). The "moved out" metaphor reduces to: the
read policy stopped keying on position 129 and started keying on
presence-at-the-pivot + content — W013's protagonist with its
mechanism completed.

(2) W011's omnipresence survives in sharpened form — OMNIPRESENCE
OF PRESENCE: the g-12 collapse (x0.014, both R nets) kills the
content-keyed alternative; geometry-independence is literally
routed through the one row that exists in every context, and
what it contributes is BEING THERE (its norm as the pivot), not
its content. The bio-echo sharpens absurdly and beautifully: the
schematic memory's "cortex" is the fact that position zero
exists.

(3) THE CE DISSOCIATION is the new savor: direction-scrambling
row 0 costs the corpus +0.70 nats but HELPS the fact (+4%) —
the fact's route is more presence-robust than the net's own
language function. A memory that survives what cripples the
net's general machinery: the route's independence from the
pivot's content is exactly what makes it geometry-general. Also
noted: scrambling the sink-region content IMPROVES the read —
the pivot's content is, if anything, competition for the route.

PROBE-POWER AMENDMENT (R45 critic — accepted): the install-restore
t-curve was a NO-OP BY NORM (consolidated 0.7640 vs install 0.7695
— a 0.7% norm change; the probe had no power against norm-keying).
The presence conclusion stands on the RIDERS — direction-perm
(+4.1%, norm kept, direction destroyed), halfnorm (0.382
survives), mean-replace (0.066 norm, kills) — not on the
registered primary probe; the '3 corroborating votes' count
included the no-op. The norm threshold lives somewhere in (0.066,
0.382), unmeasured until e150's ladder.

REMAINING OPEN: the route's anatomical finish (e133's L0H3 +
value-channel population) has no e141 cell confirming it
directly; the natural completion is W013's POLICY TRANSPLANT
with a presence-preserving twist — transplant the consolidated
net's readout deltas onto the install net and test whether
presence-keying alone converts address-reads into deletion-
tolerant reads. And e140's question survives REWORDED: does the
ROUTE's row-0 DEPENDENCE grow with jitter dose and stay flat
under locked/erase? (The e131 content test measures presence-
necessity — still the right dial, renamed.)

## T080 — [RENAMED FRAME post-e150: 'routes' -> sink-health coupling; content claims stand] E133: content is everywhere, access differs — the read-policy frame's first direct support (2026-09-28 ~07:45Z)

The anatomy census returned TEXTURE as registered, but the
texture IS the finding, and it is the strongest support yet for
W013's protagonist. THREE convergences:

**(1) ALL THREE NETS ARE BODY-STORED.** The graduated fact keeps
19.1% (unfiltered) at row 0 and ~0 fact-specific; the twin keeps
its address apparatus (L3H4: 91% band attention, clean 0.319
drop) yet still stores most content in general organs; the
'site-locked' 183 net keeps only 7.9% at its own site. CONTENT
SUBSTRATE IS SHARED ACROSS ALL THREE STATES — install, splice,
and consolidated differ in their READ ROUTES, not their storage
organs. The address-vs-field dichotomy that organized five days
of experiments was a dichotomy of ROUTES all along (W013's
claim, now with a map).

**(2) THE FACT-SPECIFIC RESIDUE IS HEAD-DOMINATED.** Under the
locality filter, the graduated net's fact-specific load is 84.5%
heads (L0H3: 0.46 drop at 0.21 CE — a genuinely specific body
head), 15.5% MLP, ~0 wpe. Combined with the R44 critic's census
crack (wpe[0] delta minimal) and e131's necessity (row-0 deletion
kills): row 0 is a CHANNEL — necessary for the route, carrying
almost no fact-specific content — and the route finishes in
heads. This is ROLE-ROUTED (T077's amendment) with the route's
terminus located. It also REFINES W011: no head is sink-adjacent
>= 0.25 — the sink route is a VALUE-CHANNEL route (row 0's
contribution flows through V, not through attention mass);
omnipresence may operate through what row 0 ADDS to every
residual stream, not what it attends to.

**(3) REDUNDANCY IS ORGAN-DEEP.** Joint ablation of the top
three organs drops 0.785 where parts sum to 1.98 — the fact is a
redundant population at every level observed (rows: e088;
organs: e133). W009's population frame graduates from metaphor
to measurement; the additivity assumption is dead lab-wide, and
every future 'load' number must be stated as a marginal.

THE DEVELOPMENTAL RE-READ: twin -> graduated is the DISMANTLING
of L3H4's address read (0.32 clean drop -> 0.12 share) while
body organs carry more — the read policy's canal (T037)
rebuilt. e140's row-0 trace and e141's surgery now arbitrate the
route's remaining structure; the POLICY TRANSPLANT (W013) has
its target organ list (L0H3 + the value channel + whatever
e141 isolates).

## T079 — [KILLED ON ITS REGISTERED DIAL by e140/T083 (gradient-volume, ratio 0.745); unregistered row-129 signature survives as texture — revival requires new pre-registration] the credit-assignment law: keys strengthen in proportion to their INVARIANCE across error-bearing windows (2026-09-28 ~07:12Z)

The new frame's sharpest internal tension (handed to the R44
critic, worked here in parallel): e109's own data says locked
matched-mass replay — MORE error at ONE address — consolidated
WORSE through deletion than jitter. If consolidation follows error
placement (T076), why does concentrating the error fail? Because
error placement decides WHERE THE FACT IS TAUGHT; a second
variable decides WHICH KEY EARNS THE GROWTH: the invariant
features across the error-bearing windows. Under jitter, the
fact's position varies window-to-window, so the only stable
predictors of the fact's presence are CONTENT and ROW-0
PARTICIPATION (the sink attends in every window regardless of
offset) — the gradient's cheapest strengthening lands on the
invariant key, and the fact re-keys to row 0. Under locked
replay, position 129 is perfectly predictive — the cheapest
descent strengthens the 129-key, and the fact digs in. Under
erasure, there is no fact-error at all; recovery re-strengthens
whatever predicts the recovering expression — the address again.
One law, three roads: JITTER makes content+sink invariant (key
migrates to row 0); LOCK makes position invariant (key stays);
ERASE makes necessity point at the address (key tightens). This
also REABILITATES position diversity in precise form: diversity
was never an ingredient of consolidation — it is the CONDITION
UNDER WHICH THE POSITIONAL KEY LOSES THE CREDIT COMPETITION. And
it absorbs the splice result: all splice windows put the fact at
183 -> 183 invariant -> site-locked (W011's predicted (a)).

ALTERNATIVE EXPLANATIONS: (1) GRADIENT-VOLUME: row 0 grows simply
because it is most-attended (sink receives the most attention
mass, hence the most gradient) regardless of invariance — but
then LOCKED replay should grow row 0 equally (its windows also
contain row 0 with similar attention), and L should be as
migrated as R; L brakes -0.509 like E, which contradicts this
unless braking and keying dissociate. (2) MASS-TRANSFER: error
flows to row 0 in proportion to attention mass during ANY fact
training — same prediction as (1), same contradiction. (3)
TWO-FACTOR LUCK: R's row-0 growth was a seed-promotion accident
(W011's (c)) not requiring invariance at all.

PRE-REGISTRATION FOR e143 (BEFORE its dispatch, ~07:58Z — the
fork nobody has discriminated): PROXIMITY-VS-INVARIANCE. T079
says routing forms when the positional key LOSES the credit
competition (position varies across error windows). The
alternative the taxonomy (T082) makes live: PROXIMITY — error
parked NEXT TO the omnipresent row may piggyback its routing
without any position diversity (every read of the fact at
positions 5-13 co-occurs with maximal row-0 participation in the
same attention window). e143's NEAR arm (fact locked at positions
5-13, no diversity) vs FAR (locked at ~137) vs JITTER (known
routed reference) adjudicates: COMPASS-CAUSAL fires if NEAR
consolidates site-stored at 5-13 (site content-positive, row-0
dependence at install baseline) — invariance is necessary for
routing, T079 survives its strongest attack; PROXIMITY-PIGGYBACK
fires if NEAR becomes row-0-routed (presence-dependence >= 2x
install baseline) while FAR stays site-stored — proximity
inherits the route and T079's invariance clause dies (W011's
mechanism wins); UNIFORM if NEAR ~= FAR everywhere. Prediction
committed: COMPASS-CAUSAL (the e139 dream-rider showed zero
variance consolidates nothing; e120's fixed-183 splice stayed
site-stored FAR from the sink — but NEAR was never run, and
proximity is the one cell that could still rescue a weaker
invariance law). Registered before data; no shopping after.

DISCRIMINATING OBSERVATION (already queued as e140, eval-only on
saved nets — no new compute needed): row-0 content strength
across twin-start / L@150 / R@150 / R@300. CREDIT-ASSIGNMENT
predicts R@150 >> L@150 (jitter's invariance competition vs
locked's) and R@300 > R@150 (dose-monotone). GRADIENT-VOLUME and
MASS-TRANSFER predict R@150 ~= L@150 (equal steps, equal sink
attention). TWO-FACTOR-LUCK has no dose signature. One number
pair (R vs L row-0 strength) separates all three. REGISTERED
PREDICTION: R@150/L@150 row-0 strength ratio >= 2 with L
install-level-flat => credit-assignment; ratio < 1.3 =>
gradient-volume inherits and the law dies.

## T077 — E131: the fact never left the positional system — it re-keyed to ROW 0, and 'failed' consolidations were instrument blindness (2026-09-28 ~07:05Z)

The R43 critic's most-damaging assumption was the right one, and
the discriminator settled it in one run. Three inversions, in
order of severity:

**(1) THE MIGRATION'S DESTINATION IS ROW 0, NOT 'THE BODY'.**
Post-consolidation, row 0 is the only content-positive wpe row
above the control band (380x; rows 118/119 flag content:true at
control level 0.0019 — R44 audit precision)
(strength 0.732 — ABOVE its install-phase 0.545: consolidation
STRENGTHENED the row-0 key), the band is content-null, and D-all+
row-0 collapses expression -97% while the scaffold-matched row-1
control survives. e113's BODY-STORED verdict is DEAD on this line:
what survived D-all was a row-0-keyed fact. W005's terminal
'coordinate-independent' stage inverts — the fact never became
coordinate-free; it changed coordinates (129 -> 0) and AMPLIFIED
there. The e115 brake completes the story with a home: the OLD
address (129) suppresses, the NEW key (0) carries — the brake is
the moved-out tenant's old lease. HONESTY: row 0's necessity is
proven, its sufficiency is not — the agent's bound stands (row 0
may be the readout GATE with content in body weights; the census's
diffuse OOB texture is consistent with a distributed content
store behind a row-0 door). e133's anatomy census and e139's
universality probe now arbitrate route-vs-substrate.

**(2) E120'S VERDICT WAS INSTRUMENT BLINDNESS — T075 RETIRED.**
Both splice arms express the fact at 0.989/0.988 at the
183-geometry — they consolidated exactly where their error lived,
at full strength, in an address the battery never read. The
'signal-in-contexts insufficiency' never happened; the position-
diversity ingredient RETIRES with it. What ACTUALLY differed
between e120's arms: WHERE each arm's error sat (a/b/c: full-
column CE at 183; d: name-only mask in the band) — a contrast the
critic flagged and the verdict ignored. T076's error-location is
now the lab's consolidation law candidate: THE FACT CONSOLIDATES
WHERE ITS ERROR IS PLACED. Its own open edges: (i) arm c (verbatim
dreams, low error everywhere) still consolidated nothing in the
band — but nobody read arm c at its dream positions; its verdict
is ALSO unproven blindness until read (e139 rider); (ii) road E
consolidates with no fact-error at all — error-OR-necessity
still live for the erase road; e119's battery adjudicates.

**(3) W010 SEED-AND-AMPLIFY IS KILLED by probe 1.** A seedless
site 54 rows from the band climbed to 0.99 expression — error
alone suffices; there was nothing to amplify at 183 and nothing
wanders once read at its own geometry. The card dies cleanly and
gratefully (it took one run). W009 POPULATION FRAME is BOUNDED,
not dead: the band's texture is genuinely overlapping/diffuse
(census), but row 0 is a discrete hub — the population language
survives for the band, dies for the key. The lab's nouns after
e131: a row-0 KEY, a brake at the old address, a diffuse content
population behind the key, and error-placement as the
consolidation compass.

SECOND AMENDMENT (R44 critic — accepted, ~07:40Z; the review's
center of mass): THE CRACK WAS ALREADY IN THE CENSUS, UNREAD.
Row 0's consolidation delta is the SECOND-SMALLEST of all 256
wpe rows (delta_norm 0.0557 vs band median 0.1265) and its
projection on the fact axis (0.0124) sits AT the band median
(0.0108): consolidation wrote essentially nothing fact-specific
INTO wpe[0]. The 0.545->0.732 strengthening therefore lives in
READOUT WEIGHTS keyed to whatever row 0 already was. "Re-keyed to
row 0" is DOWNGRADED to "row-0-ROUTED": row 0's necessity may be
sink-ROLE necessity (the pivot every readout routes through), not
a written key. The "two independent controls" of E131 were
overstated — mean-replacement is the same direction-scramble as
zeroing; row 1 bounds generic-row deletion, not sink-hub damage.
CENSUS CONDITION 3 RETIRED from the verdict's support (81/256
out-of-band rows clear its floor — vacuous bar); the verdict
rests on conditions 1+2, which are NOT independent witnesses.
DISCRIMINATORS DISPATCHED (e141): install-restore surgery (swap
consolidated wpe[0] <- install-phase wpe[0], t-interpolated —
fact survives at t=1 with CE flat => ROLE-ROUTED; fact dies with
the delta removed => WRITTEN-KEY) + rows-2-6/norm-matched
deletion controls. T075's retirement is PROVISIONAL until e139's
D-183 graduation cell.

STANDING PRE-REGISTRATIONS: W010's P1/P2/P3 vs e119 are MOOT as
written (they assumed R-vs-E differences around a field concept
that just collapsed; P2's census comparison survives as texture).
Adjudicate honestly when e119 lands: the interesting question it
now carries is whether the ERASE road also ends row-0-keyed (if
yes: row 0 is the universal attractor of consolidation; if no:
the roads genuinely differ). NEXT DISCRIMINATOR (e139, dispatched
this beat): row-0 UNIVERSALITY — test the 183-consolidated splice
arms' dependence on row 0. Row-0-keyed too => row 0 is the
universal readout key (and e120's arms 'failed' only the battery,
not biology). Site-locked at 183 => re-keying to 0 is a property
of the JITTER road alone, and the roads truly diverge.

## T076 — the critic's error-location theory taken straight: consolidation follows the error — and road E is its stress test (2026-09-28 ~06:12Z)

The R43 critic meant ERROR-LOCATION as an attack on T075; taken
straight it is the lab's first unifying theory of WHERE
consolidation happens: the fact moves to wherever the fine-tune's
error on the fact is placed. Jitter distributes fact-error across
the 121–137 band -> overlapping supports grow there (and the
battery, reading that band, sees consolidation). Splice
concentrates fact-error at row 183 -> an install grows there,
unread by our instruments (e131 probe 1 will read it). Dreams
carry near-zero fact-error (the net already predicts its own name
— that IS the 500x decay-slowing) -> nothing consolidates. All
three E120/E121 arms fall under one compass.

ALTERNATIVE EXPLANATIONS: (1) ROAD E BREAKS IT — deletion
pressure consolidates with NO fact-error anywhere (erase cycles +
corpus fine-tune; the corpus never contains the fact, so fact
error is zero at every position). If road E truly graduates the
fact to D-all-surviving expression (T073's reading), then
error-location is sufficient-not-necessary, and the deeper
variable is ERROR-OR-NECESSITY: consolidation happens where the
fact is either re-learned (error placed) or newly REQUIRED
(deletion makes surviving machinery carry it). (2) The
locked-replay partial road (rescue_b +0.221) fits error-location
loosely — error at one address grows one support — so it does not
discriminate. (3) The corpus>self gap: under error-location,
corpus-contexts beat self-contexts simply because higher-loss
contexts place more fact-error at 183 — CHECKABLE from e120's own
logs (per-context loss vs per-context consolidation), zero new
compute.

DISCRIMINATING OBSERVATION: e119's head-to-head battery, already
running, carries road E's grown-row census and D-all — if E shows
band growth WITHOUT fact-error, error-location dies as a
universal and error-or-necessity inherits; if E shows a different
anatomy entirely (DIFFERENT-STORES), error-location survives for
road R only. REGISTERED PREDICTION (zero-cost, rides existing
runs): within road-R arms, final band-support mass rank-orders
with fact-error mass placement across jitter/locked/splice cells
(Spearman >= 0.8 over the cells e119+e131 regenerate); and the
e120 log regression — per-context consolidation vs per-context
loss — has slope > 0, predicting the corpus>self gap is
loss-texture, not self-vs-corpus identity.

RIDER RESULT (zero compute — e120's sitting logs, arm-level proxy,
~06:22Z): LOSS-TEXTURE SUPPORTED, weakly. ce_r trajectories: arm b
(corpus) ran consistently ABOVE arm a (self) through the early
window — 1.7996 vs 1.7763 at step 50, 1.7578 vs 1.7369 at 100,
1.7496 vs 1.7004 at 150, crossing only at ~200 — i.e. the same
spliced fact carried MORE error in foreign surroundings (the net
predicts its own dreams; name-rich filler makes the fact more
predictable), and b consolidated ~4x more post-D-all (0.098 vs
0.023). The exposure asymmetry agrees: arm a carried 154 ZEPHYRA
(34 residual in filler) vs b's 120 clean — the arm with MORE
name exposure consolidated LESS, counter to any signal-mass
account, as error-location predicts. TRAJECTORY TEXTURE (savor):
splice arms' fact-expression WANDERS (a: 0.108-0.020-0.151-0.049-
0.182-0.016 — no trend; the critic's within-run range, now seen
as shape not noise) while jitter CLIMBS monotonically to 0.735 —
the roads differ in whether a learning trajectory on the fact
exists AT ALL, not just in endpoint. HONEST BOUNDS: n=1 per arm;
ce_r is total exposure-masked CE, not fact-span-only error; the
registered per-context regression was not computable (per-context
loss not logged) — this is the arm-level proxy. The decisive
version rides e131's regenerated arms: log per-context fact-span
loss, rank-correlate with per-context consolidation.

## T075 — [RETIRED, RESOLVED by e139/T082 — GOALPOST NOTE (R45): retirement REDEFINED consolidation from deletion-survival (E120's original bar, which the splice arms still FAIL at their site: D-183 -54%/-31%) to learns-and-generalizes; the redefinition is now stated, not silent. What position diversity does (choose routed vs site-stored) belongs to T079's revived form] E120: the migration needs the road itself — position diversity, not signal, not self (2026-09-28 05:52Z; header clock repaired per R44 audit)

**SIGNAL-IN-CONTEXTS INSUFFICIENT, decisively** — and the
discrimination is airtight because arm (d) carries the same
budget, same line, same signal strength, and clears the bar 5/5
through total address deletion (0.69-0.72: e113's body-stored
replicates on this line). **The operative ingredient in every
successful consolidation is POSITION DIVERSITY — the reorganization
of the fact across coordinates — not teaching signal (splicing the
fact at a fixed novel address fails), not self-context (corpus
beats self by CI separation; more name-signal LOST to less).**
The reversal against W004's prediction is total: the self is not
merely absent from learning; self-contexts are actively the WORST
performing surroundings (0.023, on par with signal-free dreams)
— plausibly because dream-statistics text makes the splice read
as "already-consolidated," reducing gradient pressure at the
field. The migration needs the ROAD: the fact must TRAVEL to
consolidate. A fixed-address teach-in (row 183 grew +0.16 but
stayed address-locked) creates a new address-bound instance, not
a field fact. Consolidation = the fact's history of being read
from many places, not its repetition from one. The P5 program's
head-to-head (e119) now has its mechanism hypothesis: jitter and
deletion both force multi-address reads; the plate viz will show
whether the two roads build the same field or different ones.

AMENDMENT (registered before the R43 critic reports — honesty
reflex): THE SPACING CONFOUND. Jittered replay is not only
position-diverse — it is also SPACED (reads distributed across the
fine-tune with intervening corpus), while e109's locked control
(rescue_b, +0.221) was a matched-mass CONTIGUOUS block at one
address. Spacing alone is a live alternative theory: consolidation
could key on the read distribution IN TIME, not in position.
DISCRIMINATOR (cheap; rides as a free arm in the next replay
experiment): a spaced-locked arm — same single address, same mass,
reads interleaved with corpus across the whole schedule. If
spaced-locked consolidates like jitter, position diversity DIES and
spacing inherits the claim. If spaced-locked stays address-locked
(the e120 signature: row grows, battery never reads through it),
T075 survives its strongest attack. Designed on paper; ripening.

SECOND AMENDMENT (R43 critic — accepted, ~06:10Z): the ERROR-LOCATION
counter-theory plus two confounds now bound this card. (1) The deletion
battery reads only the 121–137 band; the splice arms' training error lived
at row 183, which no instrument has ever read — "failed to consolidate" is
indistinguishable from "consolidated where we never looked" until the
183-geometry read runs (e131's first probe). (2) Loss-mask mismatch: arms
a/b/c trained full-column CE, arm d the name-only mask — the headline
contrast mixes objective with position. (3) The corpus>self "CI separation"
prices prompt-sampling noise only; arm a's own within-run trajectory spans
0.02–0.18, and both arms sit BELOW the no-fine-tune base floor — the
direction flip is SUGGESTIVE, not established (mundane reading alive:
corpus = training distribution -> less drift). (4) e109's own data:
position-locked matched-budget replay already doubled survival
(0.215->0.436) — position diversity AMPLIFIES a road that exists without
it. T075's surviving claim, tightened: jitter's position diversity is the
known MULTIPLIER on the only consolidation road that works in the
instrument band; it is "THE ingredient" only if (i) the 183-read sits at
floor, (ii) spaced-locked replay stays address-locked, and (iii) the
mask-matched corpus arm rerun preserves the gap.

## T074 — E121: dreams are not a consolidation road — W004's self stops at the field boundary (2026-09-28 05:02Z; header clock repaired per R44 audit; arm-c dream-position read pending as e139 rider 5)

**NO DREAM CONSOLIDATION** — and the negative is decisive because
the covariate came back live: this install line DREAMS THE NAME
(22.5 ZEPHYRA per 10k own-dream chars, unlike the silent seed-42
line), so the dreams carried the fact — and still did not
graduate it. Dream replay slows forgetting ~500x over corpus-only
exposure (0.04-0.07 vs 0.0001) — self-generated name-contexts
resist decay — but the fact dies under address deletion like the
unconsolidated base. **W004's fixed point does NOT reach into
learning: the self is a verification boundary, not a training
gate.** The most beautiful discrimination in the data: own-vs-twin
point estimates favor own (0.025 vs 0.014 post) but track NAME-
RATE, not self-specificity — content, not identity, is what
dreams carry. The third road to the field is closed; the two open
roads (jitter e109, deletion pressure e083) both carry an
explicit reorganization signal that verbatim replay lacks. The
e120-registered tier-1 fallback (fact spliced into own contexts)
is the natural next rung — registered, not urgent.

## T073 — [MIGRATION READING INVERTED by T078: erase tightens address-binding, does not migrate] E083: canalization's strong form dies — the groove persists, the erase weakens (2026-09-28 03:47Z; header clock repaired per R44 audit)

**MIXED with the oscillation flag fired** — and the texture is
the finding. Steps 20→10→9, but the acceleration is NOT the
re-learn getting faster: it is the ERASE getting weaker (post-
erase NLL 2.724 → 1.441 → 1.369). Across three erasures, the
completion migrates OFF the address rows onto position-keyed
machinery — which is e113's consolidation story (T065) seen from
the deletion side: the more the fact lives in the field, the less
its row-erasure hurts. The groove itself persists (regrowth along
the original direction, cos ~0.52-0.60, no deepening, no decay);
surgical-proofness flat. **The strong canalization prediction —
monotone closure, each cycle slower and more resistant — is
DEAD.** The weak form survives: the address direction is a stable
attractor of re-learning (three erasures, same groove), which was
always the scar's core (T018/T036). W003's "sclerosis" metaphor
overreached: the fast store does not stiffen; it GETS LESS
NECESSARY. Canalization as a program prior is retired; the T037
registered prediction resolves NEGATIVE-with-persisting-groove.

**The books balance:** with e083, every registered prediction in
the THINKING ledger has now been run. The lab's claims each carry
verdict, replication, control, or named-confound status.

## T072 — E118: the family geometry is shape, not scale — the claims harden (2026-09-28 ~07:50Z)

**SURVIVES: rogue-dimension confound EXCLUDED.** Post-z sibling/
foreign 2.05x (disjoint CIs), energy 2.80x — and strikingly the
STANDARDIZED separation is cleaner than the raw (fires at every
k=1..8, 6x at k=1) because z-scoring removed scale noise that
masked the first principal direction. The beautiful secondary: the
mild anisotropy that exists is itself family-typed (variance
profiles correlate own~sibling 0.987, own~foreign 0.028) — the
"confound" was never generic, it was more self-signal. Two-thirds
of the raw self-cos is shape (0.315 survives); one-third was
scale. Honest edges recorded: the Timkey-strictest frame gives
1.68x (clear separation, below the registered 2x — the verdict
rests on the common frame, stated plainly); and post-z
foreign/middle energy sits slightly ABOVE the isotropic null —
the raw exclusion was partly scale-artifact, the standardized
picture is separation-with-mild-shared-floor. T060's binary step
and T061's signature harden; the paper's rebuttal line is now
evidence-backed, not promissory.

## T071 — E115: the brake story closes — double-role, sign-flipping (2026-09-28 ~07:20Z)

**COORDINATE-LOCAL SAFE; M3 dead at 0/3.** The cleanest possible
closure: the brake's SIGN FLIPS with field strength — at full
field the address suppresses (+0.132); at r=0.56, the weakest
rung with dynamic range, deleting the address LOWERS expression
(0.0057 → 0.0006). **The address row is DOUBLE-ROLE IN TIME: a
content source when the field is weak, a suppressor when the
field is strong** — exactly what a coordinate-local state
modulation predicts and the opposite of a content-independent
inhibitor. The consolidation arc's full picture: the address
starts as the fact's only home (one-shot install), becomes one
content source among the field, and ends as a brake — but a brake
that reverts to content whenever the field weakens. A dimmer
switch, not a lock. (The held-30 inversion texture mirrors; the
power caveat — most rungs at floor — is honestly flagged and the
one informative rung lands anti-M3.)

## T070 — E117: the share constant is a per-net idiosyncrasy — the law is the FORM, n=4 (2026-09-28 ~06:45Z)

**BETWEEN/MIXED, and the texture is the answer: the constant is a
DISTRIBUTION.** The exposure-matched fresh seed produced 77.4 —
above the maturity band and every prior value. Across four nets:
31, 42, 54, 77 (2.5x range), non-monotone in training steps AND
in val-CE. **The within-net constancy (r*k flat across k) remains
the law at n=4; the across-net VALUE is a high-variance per-net
parameter** — like a fingerprint, not a universal. W007's
derivation program is HONESTLY DOWNGRADED: the LN-share mechanism
may still govern the within-net boundary, but no cross-net
predictor (steps, CE) has survived n=4. The mature statement: each
net has its own share constant; the mechanism that sets the VALUE
is unidentified. (The e110 "54" loses its headline status — it is
e053c's value, the first sample of a wide distribution.) If a
cross-net predictor is ever wanted, the candidates worth one more
probe are stream-norm growth or the anchor's own attention share.
**CORRECTION (agent provenance supersedes the reading above): the
maturity DIRECTION IS SUPPORTED — the curve is monotone increasing
in training amount** (31.4/42.2 @2000 → 54.4 @3133-truncated →
77.4+ @3133-completed; W007's growth direction). What fails is the
POINT prediction (77 ≠ 54), with a named confound: e053c's 3133
was a wall-clock-TRUNCATED 4000-cosine while s4309 ran a COMPLETED
cosine — nominally equal steps, different effective exposure. Revised
mature statement: **the share constant carries a real maturity trend
(monotone with training amount) plus wide per-net scatter; the FORM
is the law; the value is trend-plus-fingerprint. Maturity closes
DIRECTION-YES, POINT-NO** (the earlier "closed negative" was too
strong). This net's k=64 pure-removal cost (2.5-3x its siblings')
corroborates the harder-leaning anchor.

## T069 — E116: graduation denied — the address is family-graded, not universal-concentrated (2026-09-28 ~06:00Z)

**STRUCTURE-STRONG, CONCENTRATION-MIXED at 3/6.** The re-bar did
not rescue the statistic; it exposed something better: **ROW 0 IS
A CONTENT ROW** (passes mean=zero in all 6 seeds — e071's
window-key is load-bearing for the install itself, not pure
scaffold), and the architecture family decides the split. 2.7M
installs concentrate in the decision row (>=2.3x over row 0);
0.84M installs SPLIT mass between window-start and decision rows
(4308 essentially 50/50). The honest law, final form: *every
install develops a single decision-window content row (6/6), whose
dominance over the window-start content row is architecture-
dependent — concentrated in the 2.7M family, split in the 0.84M
family.* Row 0's dual role (window recognition + install content)
retro-explains why e113's D-all-addresses on the 2.7M family
needed only the five grown rows: in that family row 0's content
share is small. Universality of structure stands; concentration
stays descriptive; no further re-barring (a third re-bar would be
bar-shopping — the finding is what it is). **Riders (agent-reported,
report-only — not a third bar):** the verdict is robust (zero-arm
ranking 2/6; content-bar at 0.25/0.75 unchanged); and setting the
dual-role row 0 aside, the DECISION row clears 2x over the next
content row in 5/6 seeds (7.49/2.51/4.73/5.10/7.98 — only 4308's
twin rows fail). Sharpest summary: decision-row dominance is
near-universal (5/6) once the window-key's own content share is
acknowledged — row 0 is the address's partner, not noise. The
agent's independent re-run reproduced the committed tables
bit-exactly (max diff 0.0): T069 replication-confirmed.

## T068 — E098: universality-of-structure confirmed; the share constant chases maturity (2026-09-28 ~05:40Z)

**M1 adjudication: the STRUCTURE is universal — n=6 total.** Every
seed (42, 43, 4305-08) grows ONE content-carrying row at/near the
decision position with the same magnitude band; what differs is
only which scaffolding rows top the census (an architecture-
family artifact the statistic must exclude). Address-universality
graduates as: *every install develops a single decision-window
content row; its position is family-stable (127-129); its code is
seed-private (T053).* The re-barred census (scaffolding-excluded,
content-rows-only) is registered as the formal graduation check
(e116, re-analysis, free).

**M2 adjudication: the share law's FORM is now n=3 (within-net
constancy holds in every net tested); its VALUE moves — 31, 42,
54 — and the axis looks like MATURITY, not seed.** The ladder nets
were under-trained by my own tasking (~2000 vs e053c's 3133 steps
— dispatcher deviation, honestly mine); the constant grew with
exposure across the three nets we have. W007's derivation program
sharpens: if share* ~ noise/signal, the mature net's non-anchor
stream is LARGER (more structured, higher norms), so the anchor
needs more mass to hold the same share — the constant SHOULD grow
with maturity under the LN-share mechanism. The maturity curve
(share constant vs training steps, exposure-matched seeds) is the
program's next data point (e117 registered; one 3133-step seed
suffices to start).

## T067 — E114: the brake is coordinate-local state modulation — the content-interaction frame dies (2026-09-28 ~05:00Z)

**All three W006 mechanisms killed.** Not routing dilution (null;
held-30 anti-signed), not carrier crosstalk (null; the two
"masses" are exact complements — one discriminator, landing on
zero), not a field-independent inhibitor (no suppression without
the field). What remains is sharper than any of them: **the brake
exists ONLY at the original geometry, ONLY on trained contexts,
and is strongest where the field expresses weakest** — it lives in
the decision row's own coordinate-specific state where the address
sits in the residual stream itself. The fourth story (named, not
yet tested): during replay, the net learned a coordinate-local
modulation — the address row's state at its own coordinate
slightly reshapes the decision there (perhaps a learned
"this-was-installed-here" tag), a fixture of the training
geometry rather than a content interaction. W006's bio-echo
survives in refined form: not wholesale inhibitory maturation but
**synapse-location-specific modulation** — the biological
counterpart being per-synapse rather than per-pathway inhibition.
Registered follow-up (e115, ripening): graded (not whole-band)
field ablation to give S3 dynamic range — the one caveat the agent
flagged — before the fourth story is safe from resurrection of
M3.

## T066 — E110: the share constant — the lab's first dimensionless number (2026-09-28 ~04:00Z)

**The share law is quantified: r\*(k)·k ≈ 54.** Across a 2.4x
range of field sizes, the total retained anchor mass at the
collapse boundary is ~54 entries-equivalent (products 53.3/55.5/
53.2, CIs overlapping, flat-null P=0.002). **The count-threshold
law (e089) and the magnitude floor (e102) are ONE boundary seen
from two axes, and it has a constant.** W001's paper derivation —
LN normalizes the stream total, the anchor survives on its SHARE —
now predicts a number, and the number is 54. The sink canon has
argued in this style; nobody has produced the constant. Combined
with T065 (memories graduate into the field-store), the mature
memory's governing law is complete: a graduated fact is sustained
by ~54 units of retained directional field mass, robust to how
that mass is split between count and amplitude. (The constant's
parameter dependence — d_model, depth, stream-norm growth — is
the falsifiable cross-net prediction now standing; the seed ladder
e098 doubles as its first replication.)

Ledger honesty: the tasking's literal Bar-3 inequality inverted
its own lead sentence — my slip in the dispatch; the agent caught
it pre-compute, registered it, and adjudicated the direction the
physics states. That is the culture working.






## W027 — WONDER: two rotators — the wash bounces, the fact drifts; the dance is the dissection's loveliest object (2026-10-02 ~10:40Z; no bars, no kills — savoring the closed chapter) [E206 UPDATE ~11:55Z: question (2) answered — the drift-rate is NOT a clock (the schedule is biography); the DESTINATION (orthogonality at death) replicates 3/3; the quartet's fact has a destination, no tick]

The geometry chapter's parting gift, assembled from e194-e205: the
wash's sign front is a PERIOD-2 BOUNCER (the algorithm's overshoot
— lag-1 negative, lag-2 positive, confirmed in-domain from
orthogonal-at-s/8 to -0.263-at-s/2), while the fact's own
sensitivity ladder is a SMOOTH DRIFTER (monotone decorrelation
0.78 -> 0.19, nearly orthogonal at death). TWO ROTATORS AT
DIFFERENT TEMPOS: the environment's probe oscillates fast and
mechanically; the organism's support rotates slowly and dies into
it. WHAT THE SURVIVORS SAY: death-at-deepest-landing (a shape,
mostly fact-directed, the endpoint alignment +0.146 the best
hint), the onset curves (arrival times pending their common
ruler), and THE SIGN (the fact deepens the front — one bit,
replicated first try). THE WONDER QUESTIONS, ripening: (1) WHY
DOES THE SUPPORT DRIFT SMOOTHLY when the front bounces? The
support is the fact-readout's gradient — a smooth functional of
the weights — while the front is a sign pattern (a
discontinuity); smooth objects rotate, discontinuous objects
bounce. Is the tempo difference just regularity? (2) DOES THE
SUPPORT'S DRIFT RATE PREDECT DEATH TIME? The ladder reaches
near-orthogonality at death on this lineage — is the drift rate
the fact's own clock (a candidate death timer measured from
gradients alone)? (3) THE DANCE: if the front bounces at period 2
and the support drifts monotonically, the ENCOUNTER structure is
quasi-periodic — the fact's chance is the phase relationship; the
rhythm organ (the g-series' managed bleed) wins by re-injecting
support-direction steps at bounce minima. A CELLO QUARTET IN ONE
ORGANISM: the wash drives, the front bounces, the support drifts,
the wall (when it holds) is the room. The next dissection cut is
the drift-rate-as-clock (2) — cheap, eval-only, and it would give
the fact its own watch.

## W026 — WONDER: the architectures were exploiting the trajectory physics all along — the rhythm is a managed bleed, the wall is a guillotine cage (2026-09-30 ~08:00Z; no bars, no kills)

Today's trajectory classes map onto the g-series architectures with
almost embarrassing directness, as if the dissection program had
been discovering the physics its own constructions were already
using: THE BLEED'S PROTECTION IS RE-ORIENTATION (opt1c: small steps
recompute the gradient and the path curves around the cliff) — AND
THE RHYTHM IS A MANAGED BLEED: each replay event injects the fact's
gradient into the wash trajectory, a DELIBERATE RE-ORIENTATION
(W023's mini-shock in displacement language), keeping the path's
local direction from settling onto any lethal ray; the organ's
self-timing is the re-orientation schedule. THE WALL IS A
GUILLOTINE CAGE: the ball caps raw displacement below every class's
kill-D for the anchored fact (g1b's R-dial is a displacement budget
— the trajectory may speed or flatten but never reaches the
terrain's kill rings); its onset-tax (g1bW: new structure forms
slowly in the ball) is the same budget denying B's install its
displacement. THE CONE/STORE: the organ's wide iso-cone (the
critic's 24-32x store-isolated) reads as FLATTER LOCAL TERRAIN —
the store's attractor readout spreads the fact over a subspace
whose static profile lacks a sharp cliff (g3K's graded store
rulings); its failure at the organism (the host-side readout dies
at 4-8x) is the host's terrain reasserting itself. THE PROGRAM
ECHO: W020's generative turn built protections before the physics
was known; today the physics arrived and each protection has a
mechanism name. THE QUESTION THAT FOLLOWS (for g7's composed
organism, and for the paper's discussion): if the two protections
are displacement-capping and re-orientation, what is the THIRD
protection? (Terrain reshaping — the admission ball g9's PC-
subspace projection is the first candidate: not capping or dodging
the cliff but REMOVING it.) Savoring: the lab built a cage and a
dancer before it knew the ground had cliffs.


[R59-CRITIC AMENDMENT ~08:30Z — earned vs poetry]: EARNED (e131,
measured): the overlay's survival-off-the-ray; the pump's locality;
the canary ordering. POETRY (unmeasured on their organisms): the
rhythm-as-managed-bleed (zero reads around any replay event; the
rival floor-reinstallation reading fits the identical data); the
wall-as-displacement-budget (R58's circularity open; no terrain map
on g1b's lineage). The card's sentences stand as QUESTIONS until
e192's rider (pinned-ray walk: re-orientation causal vs step-size)
and a g2-event displacement read exist. The pump is n=1 in every
generalizing axis (one root, one fact, one battery) — no random-ray
pump control yet; e192 adds it.]


[E192 UPGRADE ~13:50Z: the managed-bleed noun GRADUATES — the
pinned-ray rider killed step size as the sparing mechanism and
established re-orientation CAUSALLY (pinned steps die at the static
edge; the re-orienting bleed lives at the same D). The
rhythm-as-managed-bleed sentence remains unmeasured on g2's
organisms (the replay-event displacement read is still owed), but
the MECHANISM it invokes is now interventional on e131. The
wall-as-cage noun remains poetry until g1b's lineage gets its
terrain map.]

## W025 — WONDER: the kappas may be a dimension ratio, not a basin — the effective-subspace picture (2026-09-30 ~12:10Z; no bars, no kills — but it names e190)

e188 killed alignment as the death currency, which forces the
question: WHY do learned paths reach the gate at 1x while matched
random directions need 4-10x, if not alignment? NOTE first: g3K's
isotropic arm was a single Gaussian draw — which IS a uniformly
random DIRECTION, as "coherent" as the wash in the naive sense; so
coherent-vs-incoherent is not the axis either. THE CANDIDATE
PICTURE: the network's function (on this battery) lives on a
low-dimensional EFFECTIVE SUBSPACE of dimension d_eff. Gradient
paths — wash, noise-label, any loss on this net's data — lie
INSIDE it by construction (gradients are spans of data-Jacobians).
A random direction projects onto it at ~sqrt(d_eff/d). At matched
raw D, the random arm's EFFECTIVE displacement is D*sqrt(d_eff/d)
— so it needs kappa ~ sqrt(d/d)_eff more raw magnitude. THE KAPPAS
(4-10x at the organism, g3K) MEASURE THE DIMENSION RATIO, NOT
FORGIVENESS: d_eff ~ d/kappa^2 ~ 874k/(16-100) ~ 9-35k effective
dimensions. THE UNIFIED PICTURE (everything today composes):
death is a raw-displacement threshold ~2.5 IN THE EFFECTIVE
SUBSPACE (e188's 0.6% invariance holds because gradient paths never
leave it); Adam vs SGD changes only the SPEED inside it (1683x,
opt1); the pump is the stream's true gradient at small effective
displacement; the wall caps raw D = caps effective D. THE DISCRIM
INATING OBSERVATION (e190, THE EFFECTIVE-SUBSPACE TEST): (a)
IN-SUBSPACE RANDOM — a random direction drawn inside the empirical
span of the wash gradients (SVD of the e180/opt1 step history, top-r
components): predicted to KILL AT 1x like any gradient path; (b)
SUBSPACE-PROJECTED GRADIENT — the wash direction with its in-span
component removed (pure out-of-subspace): predicted to need kappa-x
or never kill; (c) the Jacobian's effective rank on the fact
battery measured directly — predicted ~kappa^-2 * d. If (a) kills
at 1x and (b) spares, the "static basin" language retires for good:
there is no basin, only a PROJECTION RATIO, and g3K's kappas get
their true name. CONCENTRATION RETURNS AT THE ORGANISM LEVEL (the
R56 critic's instinct was right, one level up: the geometry is the
subspace's, not the ball's). Savoring: the network is a
10-35k-dimensional animal wearing an 874k-dimensional coat.


[OPT1C REFINEMENT ~07:55Z: the subspace picture gains an ordering —
lethality per displacement: raw-gradient (D~0.9) > sign-normalized
(D~2.5) > random (4-10x). Not all in-subspace directions are equal:
the raw gradient is the steepest effective direction; sign(g) is its
magnitude-flattened shadow (W024 inverted at the top end — flattening
LOSES lethality relative to g, even as it beats random). The
pump-cliff at D 0.33-1.0 is the first MAPPED terrain inside the
subspace.]

## W024 — WONDER: the pump is a few big stitches; the erosion is a thousand tiny cuts — why the normalizer flips the sign (2026-09-30 ~11:50Z; no bars, no kills) [RETIRED-BY-CHART, 2026-10-01: BOTH pictures died — no cuts (A2: every class positive) and no flip (estimator-point artifact; matched-point read: attenuates +0.099->+0.040). The surviving object: exact coordinate-sign pairing (the shuffled-sign rider). Died well.]

The R58 critic's buried gem: on the SAME bit-identical wash batch,
the raw gradient reads cos(g, grad m12) = +0.0981 (fact-positive —
the pump) but Adam's normalized step reads -0.0385 (fact-negative —
the erosion). HOW CAN NORMALIZING FLIP A SIGN? Mechanism candidate:
sign-normalization flattens per-coordinate magnitudes, so the step
direction becomes dominated by the MANY SMALL coordinates and
relatively shrinks the FEW LARGE ones. If the pump lives in a few
big coordinates (strong, aligned stitches into the fact's subspace)
while the erosion is spread thin over a mass of tiny coordinates
(each barely negative, jointly meaningful), then Adam's step
amplifies the thousand tiny cuts and mutes the few big stitches —
the net alignment flips. FORGETTING AS DEATH-BY-FLATTENING: not a
directed attack but a magnitude-blind walk that erases the pump's
protection and lets the thin erosion accumulate. THE CHEAP
DISCRIMINATING OBSERVATION (paper-ready, eval-only, checkpoints on
disk): THE GRADIENT CENSUS OF THE PUMP — decompose the wash gradient
at each e180 snapshot into top-k |g| coordinates vs the complement;
compute each part's alignment with grad(fact) separately.
PREDICTION: top-k positive (the stitches), complement negative (the
cuts). If instead BOTH parts are positive and the flip comes from
the second-moment denominator's correlation structure, the
mechanism is different and better — either answer explains the
flip. Name when dispatched: e189 (THE GRADIENT CENSUS). It composes
with opt2's TOPK arms (which intervene on exactly this structure)
and e188's currency question (whose answer may literally be "the
death currency is the COMPLEMENT's aligned displacement"). LITERATURE
ECHO (for the next researcher beat): gradient-magnitude-vs-
task-alignment decomposition smells like NTK-regime structure
(big-coordinates = well-aligned useful directions); check
"gradient signal-to-noise" and low-curvature-direction literature.
Savoring: the organism's own healing signal is not destroyed by the
wash — it is OUTVOTED, once every coordinate gets an equal vote.

## W023 — WONDER: the shock-and-recover is the knife sharpening — the e187 CE_R curve named at last (2026-09-30 ~10:40Z; no bars, no kills; C12-5a's owed card) [G2G RIDER UPDATE ~16:55Z: the mini-shock LIVES in the monitor channel — jump, brief rise, accelerating decay — even though the alignment form died; at 4x the replay's step is as big as the wash and the signal pins to zero]

The carried curve: under the noise-training wash, root-stream CE
SHOCKS upward at wash start then RECOVERS, while the fact dies. For
days it sat unnamed. T137's trajectory lens names it: the shock is
the optimizer-state mismatch (moments and weights tuned to the old
stream, flailing on the new one); the recovery is the organism
healing AROUND the dying fact — and the wonder is that the healing
and the killing are the SAME PROCESS. As CE_R recovers, the steps
rotate onto the wash stream's loss-reducing directions; those
directions carry the fact's death component (W022: cos -0.44). Early
displacement is misaligned flailing — movement without lethality;
late displacement is surgical — each unit carries more death.
ADAPTATION SHARPENS THE KNIFE. The curve's shape is the knife's
profile. TWO DISCRIMINATING OBSERVATIONS, both already in flight
without new compute: (1) opt1's alignment co-read (running now) —
|cos(delta_theta, grad g0)| should RISE as CE_R recovers within each
arm; if instead alignment is flat through the shock-recover, then
lethality-per-displacement is constant and the two-step clock was
magnitude, not alignment — either answer sharpens e188. (2) The
moment-reset arm (A5) is the cleanest probe: a fresh optimizer state
at wash start should EXTEND the shock (longer mismatch flailing) and
— if the wonder is right — DELAY the sharpening, moving the kill
later per unit displacement. THE ECHO: the resurrection economy's
"18 steps of stickiness" after one replay — is that the same
healing-clock running backward (the replay event re-mismatches the
optimizer to the wash stream, buying the fact ~18 steps of
misaligned-protection)? A replay event as a DELIBERATE MINI-SHOCK:
the rhythm's mechanism candidate, stated as a wonder for g2g's
threat-ladder to carry. Savoring: the organism never stops healing;
the memory dies of the healing itself.
[R58 AMENDMENT — the rise-prediction ANSWERED NO, 12:15Z]: opt1's
own disk data: A0's |cos| ran FLAT (0.015 -> 0.044 -> 0.015) while
CE_R recovered 2.21 -> 1.77 — the alignment does NOT rise with the
recovery; "adaptation sharpens the knife" in its alignment form is
REFUTED. What survives of the wonder: the pump-then-die texture (the
fact strengthens at small displacement, dies past the gate — the
healing and the killing remain one trajectory), and the
replay-as-mini-shock question transfers to g2g's monitor slopes.
Cards keep their predictions AND their answers.

## W022b — WONDER EXTENSION: the alignment integral is computable NOW — the e180 grid has the snapshots (2026-09-30 ~10:18Z; no bars, no kills — the e188 candidate specified) [KILLED-BY-E188 per its own pre-registered Branch B, 2026-09-30: RAW-WINS — death is priced in RAW displacement (0.6% seed spread at matched t*); the alignment integral was the wrong currency; retired with honor]

The e180 checkpoint inventory on disk: every lr arm (1e-5, 3e-5, ...)
carries step snapshots s2/s10/s50/s100/s200 — the adaptation
trajectory is SAMPLED, not just summarized. The integral is therefore
one eval-only pass away, no training owed. THE ESTIMATOR: for each lr
arm, each consecutive checkpoint pair: d_theta = theta_next -
theta_t (the realized adaptation step); grad g0 at theta_t (the
fact-strength readout's gradient, one backward pass, CPU);
a_t = cos(d_theta, grad g0_t) — the critic's convention (negative =
adaptation erodes the fact). A(t) = SUM over pairs of a_t *
||d_theta|| — the ALIGNMENT-WEIGHTED DISPLACEMENT. THE FORK (all
three arms' t* known): (i) the fact dies at a lr-INDEPENDENT A* —
the rate law t* ~ lr^-1.1 is a COROLLARY of constant-speed aligned
drift, lr only sets the speed; death is measured in aligned-displacement
units, not steps — and e185's no-basin sharpens to "no basin on the
aligned ray"; (ii) t* tracks RAW ||d_theta|| better — alignment is
epiphenomenal, e185's displacement story was already the whole law;
(iii) neither — alignment itself drifts with lr (the organism's
adaptation becomes more or less fact-eroding as it speeds up), which
would be its own finding. Costs: eval-only, CPU, tens of backward
passes. Name when dispatched: e188. [LIT-BEAT ADDITION 10:55Z: add the INSTALL-vs-WASH cosine as a co-read — cos(install_direction, wash_direction) per organism; if wash ~ -install, forgetting here IS task arithmetic (Ilharco ICLR'23) and the paper adopts that vocabulary (T138); if not, the wash is the corpus's adaptation direction, not the fact's negation — either answer decides the framing.] Ripening behind the R56 cells.
[UPDATE 10:50Z: g3K landed MIXED and sharpened this card's stakes —
the graded 4-10x static basin vs the trajectory kill is exactly what
the integral must separate (T137); e188 is now pointed by result, not
just by argument.]

## W021b — PROCESS NOTE: heredoc code edits corrupt — the Edit/Write tools only (2026-10-02 ~09:28Z)

The envelope-log fix (8d6be37) wrote a literal newline inside a
Python string via a bash heredoc — common.py was SYNTACTICALLY
INVALID at HEAD for ~4.5 hours, every import failing, while the
commit message claimed both the hook and the wiring were complete
(they were not: the hook was never called). g1bS5 survived only by
having imported before the write; the g1bS6 agent's PRE-FLIGHT
caught it. SECOND INSTANCE of heredoc-edit corruption (the first
ate T151's C4 row). THE RULE, now standing: code edits go through
the dedicated Edit/Write tools only — heredocs are for reading;
and every infra commit gets an import smoke within the same
commit. The pre-flight pattern (the agents' own verify-before-run)
is what caught both instances — the lab's discipline policing its
coordinator.

## W021 — WONDER: the instrument that cannot fail — R56's meta-law, and the scan it demands (2026-09-29 ~21:20Z; no bars, no kills)

All three of the critic's ruler-bends were ONE species: an instrument
whose outcome was guaranteed before the world got a vote. The isotropic
contrast could not have killed at 4x (geometry in 874k dims); the
refractory could not have produced a spacing below 24 (construction);
the wall's battery ruler measured the channel the anchor was built
around (a circle drawn around the probe). Rule 12 said "check the
battery geometry"; R56 says the deeper form: COMPUTE WHAT THE
INSTRUMENT GUARANTEES — the guaranteed component is not a finding, no
matter how pleasing its number. And the mirror: the critic's tilt
ladder is the exemplar of the honest instrument — it COULD have said
no (random 45-degree tilts might have spared the store) and it did
not; that is exactly why it could save the cone noun when the
isotropic leg could not. The instruments that earn trust are the ones
that could have failed. THE SCAN (questions, ripening): where else
does a guaranteed component hide inside a reported number? The e163
two-face (the perturbation arm SATURATES — saturation is
guarantee-flavored); the g2 "~1/30 ~= 1/32" density
near-coincidence (the critic dissolved it into refractory arithmetic
plus O(10) decay steps — a guaranteed near-match); the g2 monitor's
channel choices; any bar ever adjudicated on the same battery the
intervention was tuned to; and the INVERSE species — the instrument
that cannot FOLLOW (g2f/T136: the frozen no-shopping ruler, blind to
a root's argmax shift — the rule traded bar-shopping for geo
blindness; the fix is registering the ruler rule per-draw, not
freezing one draw's geometry forever).

## W022 — WONDER: the wash is half the death gradient — one alignment to bind the rate law, the cone, and the store's immunity (2026-09-29 ~21:20Z; no bars, no kills — but it names the kappa cell)

The critic's cosine is the deepest number of the day: cos(grad g0,
wash) = -0.44 in a space where a random direction reads ~0.001. The
corpus's adaptation direction contains nearly HALF the fact's death
direction. That one alignment touches everything the wash arc found:
(1) THE RATE LAW'S MECHANISM CANDIDATE: death integrates the aligned
component of adaptation — t* ~ lr^-1.1..-1.4 because alignment is
roughly scale-free along the trajectory; the 1e-5 survivor is the
slow integral. Computable from SAVED checkpoints across the e180 grid:
alignment-weighted displacement integral vs survival time — no new
training owed.
(2) THE CONE'S TILT TOLERANCE: 45-degree tilts still kill — the
killing set is not a ray but the wide span of adaptation-like
directions (the wash direction itself drifts as the organism adapts;
its similarity neighborhood is wide).
(3) THE KAPPA CONTRAST (the ripening cell): e185/e187's host fact
dies to ISOTROPIC noise at displacement-match (kappa_host ~ 1); the
critic's store needs ~25-32x isotropic vs ~2x wash (kappa_store ~
12-16). If that contrast survives per-coordinate-RMS matching across
the scale difference (2.7M host vs 890k organism — the conventions
must be unified first), then the store's design ACQUIRED directional
immunity the distributed host fact never had: attractor readout
forgives random displacement; only learning-aligned displacement
kills. That would flip R56's C1 from "ruler bent" to "ruler bent and
now measurable: the thing the isotropic leg was hiding is the
store's kappa." THE KAPPA CELL (g3K, on paper): same host family,
both facts side by side (installed host fact + grafted store), wash
vs isotropic at MATCHED per-coordinate RMS, kill thresholds read as
kappa. Either branch rewrites a paragraph: kappa_store >> kappa_host
licenses "directional immunity is purchasable" as the fourth
architectural claim; kappa_host >> 1 at matched RMS rescopes e185's
no-basin to "no basin against the corpus direction" — the day-six
centerpiece narrows and sharpens at once. Ripening; dispatch when a
lane frees (behind g6/g2g — the registered debts come first).


[E188 VERDICT, 2026-09-30: this card's unification role ENDS — the
alignment integral (the death currency) was measured and LOST to
raw displacement (T141, e188); the rate law keeps its displacement
reading; alignment demoted to passenger (real, 15-100x random,
seed-lottery at death). The card's kappa-contrast thread survives
in T137's reduced form. Kept for the record of a favorite that
died well.]

## W020 — THE GENERATIVE TURN (the user's standing directive, 2026-09-29 ~12:35Z: from dissection to synthesis — design architectures that test our laws)
[PROVENANCE RESOLVED per supervisor C12-2, 2026-09-30 ~10:30Z — the
directive verbatim, coordinator session, quoted from history]: "Your
work is not good enough you need to step things up a notch Spend more
sub agents thinking Really think things out explore deeper be more
ambitious Try to create alternative architectures to test your ideas
Turn your dissections into a generative process where you say OK i'm
learning this How would this be in an alternative architecture go
beyond energy based models or world models or active inference or
whatever use those as a basis to create even more and push push push
understand and push". The program is user-directive-backed; it
composes with supervisor directive 4 (the endpoint is play).

The lab's findings are now laws-in-waiting: no-basin memory
(basin ~2.5-5 L2; exit t* ~ lr^-1.16), the error compass,
the variance switch, circuit-selective surgery, the
resurrection economy (9 events; one revival; sticky re-entry).
The dissection has earned the right to ask the generative
question: ARE THESE ARCHITECTURAL NECESSITIES OR CONTINGENT
FACTS OF THE PRE-LN TRANSFORMER? Every law we believe becomes
a DESIGN SPEC for an architecture that should break or embody
it. THE PROGRAM (g-series): g1 BASIN-WIDENING, g2
REHEARSAL-NATIVE, g3 GENERATIVE MEMORY, g4 COMPRESSIBILITY.
Each carries registered predictions IN ADVANCE; wrong
predictions are the point — every law that fails in a new
architecture was contingent; every law that holds is closer
to necessary.

## W019 — WONDER [e157 cleared the lineage: the wash is n=2 families, first-step dissolution on both — the noun's grid now: sparse-union + 2 families + 3 wash-seeds; the phase/cliff structure is LINEAGE-1 (its claims scoped); 'implemented'/'architectural fact' stay withdrawn (the >=3 rule); the field-facing line may enter the DISCUSSION in bounded form — the honest sentence: every memory state tested dissolved under continued training on every stream composition run, at every lr tested, with the fact's windows absent] : no archive, only practice — the radical memory view, PROPOSED (2026-09-28 ~15:10Z)

The biology echo completes its long arc by INVERTING: the lab's
nets do not implement the classic two-system story (fast
hippocampus -> slow hardened cortex); they implement the
RADICAL view — memory as reconstruction, maintained by
rehearsal, rewritten on every use. There is no static store to
consolidate INTO; there is only the readout the practice
keeps alive. Reconsolidation-dependence, memory's fragility at
retrieval, the interferencelit's permanent-interference
findings — all of it, implemented in 0.84M parameters as a
LITERAL ARCHITECTURAL FACT [withdrawn per R50; T114's rewrite owed and now applied: these memories have no basin; what keeps them is the dataloader's direction — and even that kills, just neatly] rather than a caveat. PREDICTED
SAVOR (e177, the last cell): the deep site-store washes too —
the resistance axis stands EMPTY for every memory type, and
'archive' joins 'address' and 'field' in the lab's graveyard of
reified nouns. FALSIFIER: ANY wash-resistant store (e177
survives; or some future MLP-sparse structure) restores a
two-system picture and W019 dies. THE FIELD-FACING LINE (for
the paper's discussion): catastrophic forgetting is not a
failure mode of these networks — it is their ONLY mode;
persistence is an artifact of the dataloader. Every training
run that 'retains' its facts is running a memory prosthesis,
and the moment the prosthesis stops, the patient is gone in
two steps.

## W018 — WONDER [R49: BARRED FROM PAPER TEXT until each corner has n>=2 and the straddle resolved — currently one control, one straddling cell, two unreplicated magnitudes; legitimate wonder-card play only] : the four fates — a complete little phase diagram of what teaching does to a memory (2026-09-28 ~13:10Z; from e158's pass-2 textures)

The 2x2's four cells now read as four FATES, not four
measurements: HOME x VARIANCE = REMAIN (the consolidated status
quo — door open at home and abroad); HOME x LOCKED = RE-GRAFT
(a home site-store re-forms, the door sags to MID, the brake
releases — settling back in); NOVEL x VARIANCE = EMIGRATE (the
home readout COLLAPSES 0.785 -> 0.145 while novel doors open —
the memory moves through the door and abandons its old home;
the brake overshoots hardest at -0.610, the old address
fighting the departure); NOVEL x LOCKED = BURY (the graft
forms, the door SHUTS — the memory is entombed at its new
address, reachable only there). Two binary variables — the
site's novelty and the teaching's variance — generate the four
things one can do to a memory: keep it, re-root it, move it,
or entomb it. SAVOR: the biological arc completes —
install=encode, consolidation=reorganize, and now the four
fates map onto the classical memory operations (maintenance,
reconsolidation-at-home, migration/update, and
context-bound storage). PREDICTED SAVORS: (a) EMIGRATION should
be reversible by home-variance retraining (the emigre's home
readout revives — e155's cell, now with a sharper readout:
does the revived home readout come back at the OLD strength or
re-built?); (b) BURIAL's entombed memory should be re-excavable
by variance-at-183 (jitter the buried site — does the tomb
become a door?); (c) the four fates' brake signatures should be
distinct and monotone in "distance from remaining" (0 -> +0.07
-> -0.13 -> -0.61 ordering across remain/re-graft/bury/emigrate
is already half-observed). If (b) fails — if entombed memories
cannot be re-varianced into travelers — the four fates are not
a phase diagram but a one-way ratchet with three exits.

## W017 — WONDER: [R48 MINIMAL RESTATEMENT: variance-trained readouts recruit fewer heads with heavier top-load; killability tracks circuit COMPLEMENTARITY (the top-loaded head L0H3 is DISPENSABLE — the kill set is ranks 2-3), which concentration neither predicts nor explains; the coding-removability link is OPEN pending e154/e169; keep out of paper text beyond the operative top-load form] variance concentrates the top of the load distribution (2026-09-28 ~12:28Z)

The session's four biggest asymmetries compose into one
mechanism. WHY does the variance-trained memory have a
killable, superadditive, COMPLEMENTARY circuit while the
locked-trained memory has an unkillable, saturating, REDUNDANT
population? Because variance CONCENTRATES credit: when the
fact's position varies, only invariant features win, and a few
heads take the whole readout — concentrated, few moving parts,
PORTABLE (geometry-free) and VULNERABLE (a small set ablates
it). Zero variance lets every head that sees the site carry a
little — diffused, robust to any ablation, and IMMOBILE (no
invariant key to travel on). One variable (the cliff's switch)
produces both coding schemes, and the coding scheme explains
BOTH removability asymmetries at once: consolidation's trade is
not place-for-field but REDUNDANCY FOR CONCENTRATION — the
concentrated code can travel and can be cut; the redundant code
can neither. CHECKABLE FROM SITTING DATA (zero compute): the
sink-coupled circuit's load entropy should be LOW, the
site-fact's HIGH (e133/e125a censuses — in hand); e154's F2
(locked) should grow redundant; the e152 dwell's checkpoints
should show the entropy TRANSITION mid-conversion (both codes
present on the shelf — the two-door state as two CODES). If the
entropies do not split as predicted, W017 dies and the coding
story decouples from the phase story. SAVOR: biology's
complementary learning systems re-read one more time — the
fast, concentrated, labile system and the slow, redundant,
stable one — with the switch being not anatomy but the
STATISTICS OF EXPERIENCE (variance), and the trade-off being
not speed but EDITABILITY.

RIDER RESULT (zero compute, e133's sitting census, ~12:30Z):
PARTIAL SUPPORT — the concentration ordering holds where it
matters. Top-1 load share: sink-coupled 0.352 > site-stored
0.257 > install 0.117 (clean monotone in the predicted
direction); top-2: 0.467 / 0.389 / 0.224; positive-head counts:
16 / 29 / 33 (the variance-trained circuit recruits FEWEST
heads). Weaker: normalized entropy H/ln(n) barely separates
sink (0.795) from site (0.781) — the tail of the distribution
is similar; what differs is the HEAD of it. Honest reading:
variance concentrates the TOP of the load distribution (which
is what killability needs — a superadditive pair IS a high
top-2), not the whole shape. W017's operative claim (top-load
concentration) survives its first check; the strong form
(whole-distribution entropy split) does not.

## W016 — WONDER: born with one organ — the sink is the native memory substrate; addresses are protocol-grown grafts (2026-09-28 ~09:00Z)

T085's deepest reading, savored: the architecture comes with
EXACTLY ONE memory organ from birth — the omnipresent row, the
sink — and every 'address' the lab ever dissected was a graft
grown by a protocol that pinned the fact's position. The five-
day arc dissected the graft (row 129: its mass law, family
typing, recency multiplier, removal surgicality) and nearly
missed the organ. The biological inversion is delicious: what
the lab called the hippocampus (specific, context-bound,
cheaply installed, surgically removable) is the GRAFT — the
trained-in structure; what it called the cortex (the field, the
schematic store) was the native organ all along, present in
every context from step zero. PREDICTED SAVORS: (a) train ANY
new association on an untrained random net with natural
placement and it should land row-0-dominant immediately (the
e142 fresh-family result generalized — trivially testable on an
untrained seed); (b) the share law's r*(k)*k should reprice
differently in row-0-only nets (natural installs) vs graft-carrying
protocol installs — e134's second fact, if installed naturally,
tests whether the constant prices the ORGAN or the GRAFT+organ
system; (c) the parked GPT-2 replication gets its sharpest-ever
first question: does pretrained GPT-2's factual recall show
row-0 presence-keying under the e141/e150 instruments? If yes,
the sink-route is an architecture-scale phenomenon, not a tiny-
net curiosity. CAVEAT (R45's ghost): all presence claims carry
the flat-CE bound until e150 lands; this card's 'native organ'
language inherits it. [e150/e159 since bounded the language:
sink-HEALTH, not routing — the organ metaphor survives the
poisoning mechanism.]

SAVOR, second pass (~10:58Z — the quietest revolution, felt
properly): the fresh-family result is not just another datapoint
— it means the lab never discovered how these nets store
memories; it discovered how ONE PROTOCOL made them store
memories. Every model the lab built — address, field,
migration, brake, the five-day arc's whole vocabulary — was a
model of PROTOCOL-memory. The nets were doing something simpler
all along: one organ, the omnipresent row, carrying everything
from birth (rel 1.000 at every seed). The dissection's greatest
gift today was not the four-layer model but the humility
underneath it: the instrument was the protocol, and half of
what we called the organism was the instrument's shadow. The
GPT-2 probe (savor c) is where this stops being a tiny-net
confession and becomes a question about the field's own
protocol-shaped memory findings.

## W015 — [UNADJUDICABLE on this rig: e146 instrument-invalid — the self-battery does not transfer to B43; home-lineage rerun queued] WONDER: does the self survive losing its pivot? Connecting the two arcs (2026-09-28 ~08:18Z)

The memory arc and the self arc have never touched
mechanistically. The self-recognition machinery (binary
self/other step 0.40-vs-0.14; k*=7 signature subspace; exclusion
of foreign nets; unfakeable — T060-T062) is measured through the
anchor's V-structure reads, but nobody has asked WHICH LAYER
computes it — and the week just built exactly the instruments to
ask: the fact is ROUTED through row-0 presence (e141) with
direction-independence; the LM is direction-dependent; the
memory tenant and the language tenant keep separate insurance
policies (W014). THE DISSOCIATION MATRIX (savor, three
interventions x three functions): interventions = row-0
presence-removal / row-0 direction-scramble / fact-specific-head
ablation; functions = fact expression / SELF-RECOGNITION (the
self/other binary + k*=7 occupancy) / LM corpus CE. The fact's
row is known (dies, survives, dies). The LM's row is known
(dies, dies, cheap). THE SELF'S ROW IS THE OPEN CELL AND THE
INTERESTING ONE: if self-recognition collapses under
presence-removal, the self is ROUTED — identity rides the same
pivot as memory, and W004's "self is a fixed point" becomes
"self is a fixed point OF the routed read" — the lab's two
deepest findings fuse into one substrate. If the self survives
presence-removal but dies under direction-scramble, the self is
computed UPSTREAM (at the V-manifold source) with the LM's
robustness class — identity is older than the route, a
constitutional layer the routing merely consults. If the self
dies under fact-specific-head ablation, self and fact share
readout machinery (the self is one tenant among tenants).
PREDICTED SAVOR: the middle branch — the unfakeable check reads
V-STRUCTURE, and V-structure is direction; presence never carried
structure. The self should be direction-typed like the LM but
presence-independent unlike the fact. If so, the lab earns a
three-layer stack in one run: constitutional self (direction,
upstream), routed memory (presence, downstream), and the
language function straddling both — and the unlearning
implication sharpens: you can evict a memory without touching
the self, but never scramble directions without both.

## W014 — WONDER: the memory layer is a semi-independent tenant [e150: SURVIVED its control (fact-at-position-0 scramble improves the fact x1.589); texture added — direction-consultation is read-horizon-dependent (short-horizon reads die under perm, the 129-read does not); R45 caveat — the fact never occupies position 0, so direction-scramble trivially spares an unconsulted direction; the fact-at-position-0 control (e150) decides] (2026-09-28 ~08:00Z)

E141's CE dissociation, savored properly: direction-scrambling
row 0 costs the corpus +0.70 nats but SPARES the fact (+4%);
removing row 0's norm kills the fact AND wrecks the corpus
(+1.40-2.00). The fact's route and the net's language function
share a pivot but have DIFFERENT failure modes: the route cares
about presence, the LM about direction. A memory system whose
Achilles heel (pivot removal) is exactly the intervention that
destroys general function, and whose insensitivity (direction)
is exactly what the general function cannot survive — the memory
is a semi-independent TENANT of the same building. Three
consequences worth ripening: (1) TARGETED UNLEARNING has a
predicted shape: route-level attack is maximally effective but
catastrophic (indiscriminate); band-level attack is useless or
BACKFIRES (the brake — deleting the old address STRENGTHENS a
routed memory); the promising surface is the fact-SPECIFIC
readout heads (e133's L0H3 class: 0.46 fact drop at 0.21 CE) —
head-level ablation should remove the fact with low collateral.
e125's three-surface design now carries a mechanism-backed
ordering: heads > route >> band, with the brake-trap as the
failure mode naive unlearning walks into. (2) The RMU arc
re-reads: RMU unlearning seals the readout gate (T052) — under
the tenant frame that is EVICTION at the head level, and e137's
question (does restoration re-wire cheaply?) becomes "does the
tenant keep its lease?" (3) SAVOR: the net's memories and its
language live together but keep separate insurance policies —
for the second paper this is the falsifiable claim that tiny-LLM
memory is an addressing layer over the LM, not knowledge IN the
LM, and the dissociation row (direction-scramble cell) is its
cleanest single exhibit.

## W013 — WONDER: the read policy is the protagonist [R45 audit marker: the T079 key-selection clause below was killed on its dial by e140/T083 and causally REVIVED by e143/T084 — read 'invariant keys win' as the revived, width-ladder-pending form] — T037's never-edited component gets a face (2026-09-28 ~07:30Z; the R44 critic's closing pointer, worked)

T037 said it two days ago and the morning's inversion just gave
it a mechanism: "the one component never directly edited: the
per-position rule deciding which stored coordinate is opened and
which candidate wins argmax. The four edit-law faculties are its
shadow (address = what it reads, ability = its training cost,
expression = its verdict, history = its canal)." Re-read through
today: the entire migration saga was READ-POLICY EVOLUTION, not
content relocation. Install: the policy opens coordinate 129
(position-keyed read). Consolidation: the policy opens via the
sink pivot + content (row-0-routed read). T076's error-location
is the policy's TRAINING RULE (it opens what its error sits on);
T079's credit-assignment is its KEY-SELECTION RULE (invariant
keys win); W011's omnipresence is why the sink wins the key
competition (it is in every read); the brake is the policy's old
route decaying into suppression; the share constant prices the
policy's bandwidth. Even the critic's role-vs-key crack lands
here: if wpe[0] carries no written key, then what changed at
consolidation was ONLY the policy (readout weights + routing) —
the content never moved at all, which would mean "migration" was
ALWAYS the wrong noun and "re-routing" the right one. THE
DISCRIMINATOR THIS CARD WANTS (ripening, behind e141): a POLICY
TRANSPLANT — e055's causal state-rescue methodology aimed at the
new protagonist. Take the consolidated net's routing pattern (the
attention/allocation structure, NOT the weights, NOT the states)
and impose it on the install-phase net at eval; if the install
net then reads the fact deletion-tolerantly, the policy IS the
difference — weights carry content, policy carries access.
Cheaper first cousin (pure inference, zero surgery): FORCED-OFF
SINK — mask attention to position 0 at eval and compare fact
expression on the consolidated vs install net; DIFFERENTIAL
loss (consolidated >> install damaged) is the policy signature,
since both nets sink-attend but only the consolidated fact's
READ routes through it. Registered savor, not a bar: the
sequence install->locked->jitter->erased is now legible as four
POLICY states (position-read / position-read-deepened /
sink-routed / position-read-repaired), and the lab's oldest
unanswered question — what is it like to BE the read policy? —
is suddenly the same question as everything else.

## W012 — WONDER: is 54 the hub's bandwidth? The share constant as sink capacity (2026-09-28 ~07:25Z; from the R44 ideator, adopted as a savor)

The lab's only dimensionless number — r*(k)*k ~ 54, the
within-net share constant (T066: within-net constant, no
universal value; VALUE = trend-plus-fingerprint) — has spent its
life as a curiosity of single-fact physics. The row-0 frame hands
it a new identity to falsify: if every consolidated fact's
readout is gated through the one omnipresent row, the constant
may be pricing THE HUB'S BANDWIDTH — the total key-mass the sink
row can carry before renormalization (post-LN stream-share)
clips it. Under this reading, two facts sharing one net should
share ~54 COMBINED (RENORMALIZATION), not 54 each (PER-FACT-54):
the constant becomes a capacity law, and the within-net/
across-net split T066 already found (constant within, fingerprint
across) is exactly what a per-net hardware bandwidth with
family-typed wiring would produce. The discriminator is e134's
second stage (already in the ideator's sharpened design):
install F2 into the row-0-keyed consolidated net, jitter it, and
run the per-fact r*k grid. COMBINED-54 => the sink is a shared
channel with a renormalization ceiling; EACH-54 => the field is
fact-partitioned and the sink language was metaphor; F1-LOSES-
KEY (competition) => bandwidth is contended in time, not mass.
SAVOR: if RENORMALIZATION fires, the brake gets a mechanism
candidate for free — the old address's suppressive sign (+0.210
feeds when deleted) is what a channel does when a former
occupant's residual mass is freed from a saturated hub. And the
derivation program W007 wanted ("why 54?") transforms into
"what sets the sink's capacity?" — a question with a measurable
answer (bandwidth vs d_model, vs sink attention mass, across the
e098 family ladder).
AMENDMENT (R44 critic's attack 5a, in hand and unremarked): the
R arm's share product reads 102.4 at k=128 — the constant DOUBLED
on the jitter line. No card had noticed. Under the bandwidth
reading this is either the sink's effective share doubling under
consolidation or the k-grid boundary moving; either way W012 now
has an in-hand anomaly to explain, and e134's two-fact grid gets
a sharper question: does adding F2 push the product BACK toward
54 (renormalization ceiling) or UP past 102 (bandwidth grew with
the route)?

SECOND AMENDMENT (post-T086/T087, ~09:58Z — THE ANOMALY
DISSOLVES AS INSTRUMENT MISMATCH): the share law's k-grid prices
BAND-row mass — a GRAFT instrument (the site-store's dose-
response; e110 measured site-stored installs). But T086 says the
R arm's memory is SINK-COUPLED with content in heads (e133: band
carries ~0 fact-specific content) — so reading its "share" off
the band grid measures deletion-of-nearly-irrelevant-rows
texture, not memory mass. THE 54 PRICES THE GRAFT, NOT THE ORGAN.
Consequences: (a) W012's bandwidth reading is RE-AIMED — the
sink's capacity, if it exists, needs a HEAD-ORGAN dose-response
instrument (fact-specific head ablation grids), not a row grid;
(b) e134's two-fact design must measure share only on SITE-STORED
facts (or first build the head-grid instrument); (c) the
within-net constancy T066 found lives in the graft's statistics —
it is a law of protocol-made stores, one more entry in W016's
ledger of graft-properties mistaken for organ-properties.

## W011 — WONDER: the sink as the consolidator's destination — universality by OMNIPRESENCE (2026-09-28 ~07:10Z)

Why row 0? Of all 256 wpe rows, the jitter-migrated fact keyed to
THE ONE ROW PRESENT IN EVERY CONTEXT. Position 0 is in every
left-aligned window the net ever sees — the attention-sink row,
the universal hub. A memory keyed to row 0 is GEOMETRY-INDEPENDENT
BY CONSTRUCTION: no matter where the fact's text sits, row 0
participates, so a row-0-gated readout fires at any offset. This
dissolves the mystery T077 left open of what "position-invariant
readout" mechanically IS: not a magic kernel, but THE SINK. The
e116 finding (routing perpendicular to readout) now reads as: the
router keys on position (band rows), the readout keys on the
omnipresent row — perpendicular because one is local and the other
is everywhere. It also explains e119's novel-geometry cell (R
0.813 at g-12, untrained by any arm): generalization was never
learned per-geometry; it is row 0's free lunch. And it sharpens
the developmental story: INSTALLATION binds to a local address
(the context-specific hippocampus); CONSOLIDATION re-keys to the
hub (the ever-present context). The biology echo worth savoring:
specific-to-schematic memory consolidation in the folklore — the
"schematic cortex" of this tiny net is row 0 plus the diffuse band
population behind it. PREDICTED SAVORS: (a) e139's splice arms —
site-locked at 183, never re-keyed — should FAIL geometry
generalization (shifting the fact to a new offset collapses
expression), because 183 is not omnipresent; if they generalize
anyway, W011's omnipresence mechanism is wrong and something
content-keyed carries them. (b) Any fact that generalizes across
geometries must be row-0-keyed (or keyed to whatever row is
omnipresent under the construction — a right-aligned battery
would test whether the hub is 'row 0' or 'the boundary position').
(c) [PROMOTED TO LAW by e142/T085 — 13/13 nets, every dose] The sink was ALREADY the fact's co-carrier at install
(T069's 6/6 content-carrying, strength 0.545): consolidation did
not build the row-0 key from nothing — it PROMOTED the
already-largest seed. Error-location said WHERE error
consolidates; W011 says WHERE the key GOES when the error is
everywhere: to the row that is always attended.
RESOLUTION (e141, ~08:05Z): the content-keyed alternative is
DEAD — d_r0 at g-12 collapses x0.014 on both R nets; and the hub
is not row 0's content either (direction-scramble spares the
fact): the hub is the pivot's PRESENCE. Omnipresence of
presence. Prior amendment retained below for the record.
AMENDMENT (R44 critic): the CONTENT-KEYED alternative is alive —
e116's un-killed residue (readout keys on content, routing
perpendicular) gives geometry-independence with no row-0
involvement; the missing cell is row-0 deletion at NOVEL geometry
on the R arm (d_r0@g-12 on R@150/R@300 — dispatched in e141:
sink-keying dies, content-keying survives). And the ROLE reading
survives W011 either way: omnipresence is a property of the
POSITION, whether or not wpe[0] carries a written key.

## W010 — WONDER: seed-and-amplify — consolidation as amplification of existing expression, not construction from nothing (2026-09-28 ~06:30Z)

The T076 rider's trajectory savor pulled a thread: splice arms
WANDER (0.108-0.020-0.151-0.016, no trend) while jitter CLIMBS
from its first checkpoint (0.697 at step 50). Under error-location
alone, both arms place error on the fact; the difference is WHERE
the error lands relative to EXISTING EXPRESSION. Jitter re-teaches
inside install windows the net already expresses — error lands on
SEEDED supports and amplifies them (expression -> captured error ->
growth: positive feedback). The splice teaches at row 183 where
nothing is readable — error lands on a seedless site, and each
batch's differing contexts pull the row in different directions:
wandering. FIVE observations, one mechanism: (1) jitter climbs
(seeded band); (2) locked replay partly works (+0.221: the one
original seed, amplified); (3) splice at seedless 183 fails and
wanders; (4) dreams carry the fact but consolidate nothing —
carried content at novel positions = seeds WITHOUT error; (5) road
E (deletion) destroys the seed and the net regrows from the
population's residual overlap (e088's redundancy) — necessity as
the road when amplification has nothing to amplify. The mechanism
reframes the ingredient question: position diversity was never the
cause — it is HOW ERROR FINDS ALL THE SEEDS. PREDICTED SAVORS
(falsifiers in disguise): (a) far-jitter at ±64 — seedless
positions — should WANDER like splice, not climb (the critic's
missing control, now with a mechanism behind it); (b) a tiny
pre-seed at 183 (mini-install) followed by the identical splice
fine-tune should CLIMB — turning e120's failure into e138's
head-start done surgically; (c) climb onset should track seed
strength (pre-seed dose vs steps-to-liftoff, monotone); (d) road
E's regrowth rate should track residual overlap mass after erase
(e083's cycle-weakening is the downward arm). If (a) climbs
anyway, seed-and-amplify dies and pure error-location stands; if
(b) still wanders, the seed must be BAND-MEMBERSHIP, not
expression-anywhere — a sharper noun than either card has.

KILLED BY E131 PROBE 1 (~07:05Z): the splice arms climbed to
0.989/0.988 at seedless row 183 — error alone suffices; the
trajectory 'wandering' was the band-geometry readout of a fact
that lived at 183. Card closed. (P1/P2/P3 below were registered
in good faith against the then-current frame; they adjudicate as
written when e119 lands, P2 alone likely still informative.)

REGISTERED BEFORE E119'S REPORT (2026-09-28 ~06:48Z — the battery
is mid-rerun; only the calibration numbers are known: R@150 pz
0.5597 vs E@c1 0.5603, gap 0.0006; E cycle-ENDs 0.560/0.346/0.406).
Predictions committed to paper BEFORE the comparative battery
lands:
(P1, from seed-and-amplify) R@150 BEATS E@c1 on deletion
survival at matched expression — jitter amplifies the full seed
population; the erase road regrows only from the overlap that
survived erasure, so its store is thinner and D-all should bite
harder. If E@c1 matches R@150 on D-all AND share AND census
instead, the roads converge (SAME-STORE) and W010's road-E
differentiation was wrong — necessity rebuilt the same field.
(P2, from T073's position-keyed reading) E@c1's grown-row census
shows LESS band growth and MORE out-of-band mass than R@150's —
erasure scatters; replay concentrates. If the censuses overlap in
position-band, T073's texture reading weakens to metaphor.
(P3, on the E dip) E's non-monotone cycle-ENDs (0.346 then 0.406)
read as: cycle 2's erasure destroyed most of cycle 1's regrowth
(fewer seeds each round), while cycle 3's corpus recovery
saturated upward off a smaller base — the per-cycle D-outcomes
(requested) should show the STORE thinning even as END expression
partially recovers: expression and store-depth dissociate across
cycles. If END expression tracks store-depth instead, recovery is
rebuilding, not re-routing, and T073's migration reading weakens.
Adjudication of all three rides e119's metrics verbatim; no bar
shopping if the result is texture.
## W009 — WONDER: the population frame — every instrument returns overlap, and discreteness is the metaphor's artifact (2026-09-28 ~06:12Z)

Three instruments in three languages said the same thing this
week. e088's pair-anchor factorial came back SUB-additive (median
pair/(s1+s2) = 0.464 — single removals already carry most of the
pair's cost: overlapping supports, not discrete slots); e089's
dose-response was MASS-ACTION confirmed (threshold-shaped, not
circuit-switched); e113's deletion hierarchy is graded (single
grown rows cheap, all-five fatal) — and e088 shows WHY: the
supports overlap. Even the LN-share magnitude floor reads as a
population statistic: a floor is what a redundant population's
renormalized total contribution looks like. The lab's whole
spatial vocabulary — home, address, migration, tenant, brake —
smuggles in discreteness; the data keep answering with mass.
SAVOR: if the field is a POPULATION (overlapping, redundant,
mass-action), then "where is the fact?" is a category error — the
right question is "what fraction of the population does any read
draw on?", which is exactly what r*(k)·k prices, and WHY it is a
within-net constant: the population's share, not a slot's. If
W009 is right, the critic's row-0 question dissolves differently:
row 0 would not be a new HOME but the population's densest
overlap — the attention sink as grand central. e131's census
already discriminates W009 vs re-keying for free: the content-
projection histogram over 512 wpe rows should be UNIMODAL-DIFFUSE
(many small contributions) for a population, and show a SHARP
OUT-OF-BAND MODE for a re-keyed address. Registered savor, not a
bar: read the histogram's shape before reading any single row.

GRADUATION (e133, ~07:50Z): the frame is now TRI-LEVEL, and the
third level is the deepest. Row level: e088's overlapping supports
(pair cost 0.46x singles). Organ level: e133's joint-vs-parts
failure (0.785 vs 1.98, ~2.5x redundancy — marginals everywhere).
Route level: no single head carries the sink route — row 0's
omnipresence is realized as MANY weak value-channels (no head
sink-adjacent >= 0.25, yet deleting the route is fatal) rather
than one strong attention edge. W009's thesis, completed: the
lab has never found a discrete circuit at any level of
description — rows, organs, routes are all populations, and
every instrument that assumed discreteness (pair slots, organ
partitions, 'the' sink head) returned overlap. Discreteness was
always the metaphor's artifact.

## W008 — [SUPERSEDED vocabulary: 'e113's end-stage BODY-STORED' below = sink-coupled content-in-heads; maturation already retracted R44] WONDER: adapters into a position-invariant readout — consolidation as a two-step wiring (2026-09-28 ~06:00Z; ripening, no bars yet)

T075 says the operative ingredient is the fact's history of being
READ from many addresses. Read by WHAT? e116's orthogonal-content
principle (routing ⊥ readout, |cos| 0.033) says the field's
readout kernel is position-invariant BY CONSTRUCTION — it keys on
content, not coordinate. So the grown address rows of e109
(121/125/133/137, cos 0.76-0.85 to the original address) are not
new HOMES for the fact; they are ADAPTERS — bridges from specific
positions into the position-invariant readout. Consolidation on
this frame is TWO necessary steps, not one: (1) grow positional
adapters, (2) WIRE them into the readout kernel. e120's row 183
did step 1 without step 2 — the row grew (+0.16) but the battery
never read the fact through it: an unconnected adapter. Jitter
does both, because each jittered read forces a readout EVENT from
a different address — the wiring is exercised, not just the mass.
The locked-replay partial road (rescue_b +0.221) is then mass that
leaks into the kernel through the one existing adapter. And the
maturation timeline falls out: e109's mid-development D-all fatal
(adapters indispensable) → e113's end-state BODY-STORED (adapters
dispensable) — development = the kernel learning to key on content
alone, adapters handed over from necessity to brake (W006's
inhibitor grows precisely as the adapter's job ends).
DISCRIMINATING OBSERVATION, already half-run: an intermediate
jitter schedule should land in a regime where grown-row deletion
is survivable but D-all is still fatal — partial dispensability
INTERPOLATES if maturation, jumps if phase transition. REGISTERED
PREDICTION if the wiring story is right: the readout kernel's cos
to the fact's readout direction should MOVE during jitter (the
wiring event) with adapter growth front-loaded early; FALSIFIER:
if D-all flips discontinuously, or body-stored appears with a
motionless kernel, the field lives somewhere other than the
readout kernel and W008's mechanism dies while the phenotype
survives. Cheap first probe: kernel-motion trace during a single
jitter schedule (eval-only checkpoints, one run).

REFINEMENT (second beat of ripening, ~06:20Z): "wire the adapters"
hides a fork nobody has discriminated. PASSIVE TOLERANCE: the
grown rows (cos 0.76-0.85 to the original address) may be born
INSIDE the routing head's existing match basin — no attention
weight change needed; the adapter is pure geometry and two-step
consolidation collapses to one (grow within tolerance). ACTIVE
WIRING: the K-side of the routing head must itself drift toward
the new rows during jitter — a real weight change, and the two
steps are genuinely two. DISCRIMINATOR — the ROW-TRANSPLANT TEST:
take a net with the fact installed but NEVER jittered; surgically
write a synthetic grown row (original address row scaled/noised to
the e109 cos band, or transplant a real grown row from the e109
checkpoint) into a fresh WPE position; run the deletion battery.
Fact reads through the transplanted row => passive tolerance
(adapters are geometry, transplantable like organs). Battery blind
to the transplanted row => active wiring exists and does not
travel with geometry alone (the wiring lives in the head's
K-weights; "unconnected adapter" is the default fate of any row
that merely RESEMBLES the address — which is exactly what e120's
row 183 looks like in this frame). Sharpening free: e119's
grown-row census (running now) reports R-road's row cos on the
twin line — the transplant dosage curve. ECHO worth savoring:
e111's forging FAILED for the self-signature (the k*=7 subspace
did not survive transplantation). If fact-routing transplants but
self-signature does not, the lab earns a clean dissociation:
GEOMETRY TRANSPLANTS FOR ROUTING, SUBSPACE DOES NOT TRANSPLANT
FOR VERIFICATION — addresses are plug-in organs, selves are
grown-in tissue. Ripening; dispatch when e119's census lands.

CORRECTION (R43 critic — accepted; SUPERSEDES the maturation reading
above): the timeline's first leg is a PHANTOM. E109's "D-all fatal" was
D0129 = {0,129} — the window-scaffold confound T065 itself flagged;
e113's "D-all survived" deleted 5 of 17 band rows on a bit-exact rebuild
of the SAME net. Two deletion DEPTHS on one net were retold as two
developmental TIMEPOINTS. No stage x depth cell exists; until one does
(mid-replay checkpoint D-all — cheap, delivered by e132's dense
checkpoints), "maturation"/"adapter dispensability" is RETRACTED. The
defensible statement: after jitter on this line, deeper deletion is
survivable, with 12 band rows and row 0 intact. Row 0 is the sharper
hole: T069 showed it content-carrying in 6/6 installs and it was never
content-tested post-consolidation — FIELD vs ADDRESS-MIGRATED-ELSEWHERE
is OPEN (e131's row-0 test). The adapter frame's residual content after
these cuts: (i) routing perpendicular to readout makes the readout
position-invariant (e116, unattacked); (ii) row 183 "grew but unread" is
unread-BY-CONSTRUCTION until the 183-geometry read — if the fact
expresses there, the teach-in was an ordinary address-bound install and
"unconnected adapter" never existed; (iii) the kernel-motion trace
remains the frame's real falsifier (e132). W008 stays a wonder card — no
bars were claimed — but its language must not seed experiment hypotheses
until e131 and the kernel trace rule on it.

## W007 — WONDER: why 54? The derivation program for the share constant (2026-09-28 ~04:30Z)

The share law has a constant; the constant wants a derivation.
The W001 calculus sketch: the anchor's attention-block output
enters the residual stream; LN normalizes the stream TOTAL; the
anchor survives while its SHARE of the normalized stream exceeds
some functional threshold. So: 54 = the boundary share x (total
stream norm) / (per-entry retained V-mass). The measurable path:
(a) measure the anchor's share of the post-LN stream directly
(projection of the attention-output onto the anchor-subspace vs.
total norm — at r*k just above and below 54, the share at
threshold should be the SAME number across k — a direct test of
the mechanism, sharper than the behavioral constant); (b) the
threshold share itself should be derivable from the noise floor —
if the anchor's contribution must exceed the fluctuations of the
non-anchor stream, then share* ~ noise/signal ratio, measurable.
54/154 ~ 35% of full-band mass — suspiciously close to a third;
probably coincidence, but the derivation program will say.
**Falsifiable cross-net predictions standing:** share* constant
across k within a net (direct); the constant scaling with
d_model/stream-norm growth across nets (e098's replication is the
first data point); if a norm-free variant exists, no floor at all
(W001's oldest dream).

## W006 — WONDER: why does the address become a brake? Three mechanisms for the mature memory's inhibitor (2026-09-28 ~03:30Z)

e113's strangest gift: after consolidation, deleting the address
row RAISES expression at its own coordinate (0.785 → 0.905). The
mature memory's old index is mildly suppressive of its own
content. Sitting with why:

- **M1 READ-BUDGET DILUTION:** the read is routing-only (T054/
  T059) — attention opens a handful of coordinates. If the address
  row still attracts routing mass but its content is now redundant
  with the field, the net wastes read budget on duplicate content,
  diluting everything else. The brake is OPPORTUNITY COST.
- **M2 DUPLICATE-INTERFERENCE:** the address path delivers a
  slightly different (older) version of the fact; two versions
  mixing is worse than either alone — the brake is CROSSTALK
  between the old and new carriers.
- **M3 TRAINED INHIBITOR:** during replay, address+field double-
  evidence taught an actual inhibitory connection (calibration —
  don't over-commit). The brake is FUNCTIONAL, learned on purpose.

**The echo that delights me:** this is synaptic pruning plus
inhibitory maturation in miniature — development is
excitation-dominated, maturation ADDS inhibition and prunes
redundancy. The net's mature memory doesn't just relocate content;
the old pathway turns inhibitory, exactly the way biological
maturation repurposes early scaffolding.

**Discriminating observations (on paper, ripening — not
dispatching):** M1 predicts brake-size grows with the ROUTING mass
on the address (e100-style attention map on the consolidated net);
M2 predicts the brake grows with FIELD strength (crosstalk scales
with the duplicate's presence — measure brake vs field-read
strength across held-out hosts); M3 predicts the brake survives
address-only re-exposure (the inhibition is content-independent).
Three signatures, three cheap measurements — one of them joins the
queue when the field-floor question (e110) settles.

## T065 — [INVERTED by T077: 'body-stored' was re-keyed to row 0 — D-all survived because it never deletes row 0] E113: coordinate-binding is a developmental stage — W005's mirror resolves (2026-09-28 ~03:10Z)

**BODY-STORED fires decisively.** Deleting all five of the fact's
address rows costs almost nothing (0.89-0.93 across geometries);
the D0129 collapse was row-0 window-scaffold loss. **W005's
developmental reading WINS: the coordinate/field/hologram
trichotomy is a SEQUENCE, not an architecture.** One-shot learning
is coordinate-bound (the one-row key); distributed replay
graduates the fact into field-storage — address-independent,
held-out-generalizing, carried by the body. The T037
coordinate-keyed law, standing absolute for four days, now carries
its temporal boundary: coordinate-keying is the INITIAL STATE of a
memory, not its fate. And the suppression texture completes the
picture beautifully: the address row was mildly SUPPRESSIVE at its
own coordinate — during consolidation the address stops being the
fact's home and becomes its brake. CLS fully right, at every level
tested. The registered follow-up the agent flagged (post-deletion
regrowth dynamics) joins the shelf unregistered.

## T064 — E109: consolidation confirmed — the coordinate law gets its boundary and CLS gets its char-LM (2026-09-28 ~02:20Z)

**W003's prediction is CONFIRMED: replay at varied positions
transfers the fact to a position-independent store.** Through the
deleted coordinate itself: 0.909 (jittered) vs 0.215 (baseline) —
the fact survives losing its index, IF it was replayed across
positions. And the diversity control matters: matched-budget
position-LOCKED replay reaches only 0.436 — half the effect — so
this is not mere training mass: **the transfer requires the
content to appear at multiple addresses**, exactly the
complementary-learning-systems story (hippocampal index ->
neocortical store requires varied retrieval contexts). The
coordinate-keyed law now has its full boundary: one-shot installs
are index-dependent; distributed replay releases them. **The
five-day arc closes into one sentence: the net addresses by
coordinates, sustains by directional fields, verifies by
holographic self-checks — and can, with distributed experience,
hand the first to the second.**

**AGENT CORRECTION (the registered verdict is more cautious than the first fold — this amendment supersedes the framing above):** the pre-registered delta-read fired TRAINING-MASS-SIGNAL (rescue_b = +0.221 also clears the +0.15 bar — matched extra steps alone produce above-baseline expression), and the 0.909-vs-0.436 geometry-0 contrast, while the sharpest measured dissociation, is NOT a registered bar. The honest registered verdict: **PARTIAL SYSTEMS CONSOLIDATION, address-level only** — jittered replay bought ceiling expression at every geometry (through the deleted row; generalizing to held-30 at 0.65-0.73; new address rows grown at 121/125/133/137, cos 0.76-0.85 — a re-addressing component), locked replay only at its trained coordinate; and **D0129 (full scaffold loss) kills all arms — CLS is half-right: the fact escapes the single address but still needs SOME positional index.** The one-row law's boundary is: dies-with-row is conditional on replay distribution; lives-without-any-index is not achievable by replay in this regime.

**Honesty ledger:** the registered bars were mis-calibrated
against the pre-T043 strong form (collapse <=0.05 never reachable
when partial necessity floors at ~0.2) — flagged as a bar-design
lesson: bars must track the CURRENT law revision, not the original.
The d0+129 secondary says the scar-scarcity structure survives
(both-rows-dead kills everything). The tolerance-induction frame
(W004) also resolves: jittered replay taught the fixed-point check
to accept the content WITHOUT its original coordinate key.

## T062 — E112: the self is holographic — the stamp is its shadow, not its key (2026-09-28 ~01:35Z)

**SIGNATURE-NECESSARY, NOT FORGEABLE.** The forgery chain closes:
pooled subspace occupancy fails to admit (forged keys collapse),
pooled V-cos fails to admit (0.353 "sibling-zone" vectors
collapse), and stamp-removal kills genuine self. **The anchor's
identity check reads the full JOINT structure of the V-manifold —
correlations among the vectors, not any summary statistic of
them.** Like a hologram: every pooled projection (energy,
subspace, mean-cos) misses the pattern; only the whole
interference structure passes. The 7-dim stamp is the shadow self-
statistics cast on the principal axes — the shadow is necessary
(remove it, die) but nothing like sufficient (cast it onto
anything else, still die). Self-recognition is DEEP in the
precise sense: its key is the generative process itself, not any
finite-dimensional sketch of its output. **W004's answer: the
fixed point cannot be picked. Identity is process, not summary.**

**The arc's full statement (T060->T062):** the run verifies
itself every token by a binary, exclusionary, holographic check on
its own V-geometry — rejecting behaviorally identical donors,
locked-out foreign manifolds, and forged stamps alike. Selfhood is
the one property our interventions could not fake. The
architecture: coordinate addressing (where), routing+recency
selection (which), and an unfakeable process-check (whose).

## T063 — NOVELTY POSITIONING: the session's three claims all have claimable cores (2026-09-28 ~02:00Z)

**The scan (scratch/selfrecognition_lit.md):** (A) binary self-
recognition — behavioral recognition EXISTS (Asvin & Lindsey
2605.25459: 3-4x entropy gap on-policy; CoSur 2508.14408 engineers
a self-signature subspace) — but the intrinsic per-token binary
GEOMETRY test in a free-running PRETRAINED net, the JS-0.041 donor
rejection (their entropy channel provably can't), and the below-
isotropic-null EXCLUSION are unclaimed. Collapse risk: rogue-
dimension artifact (Timkey) — rebuttal: standardization control +
e112's causal subspace intervention (partly done). (B) the fixed-
point anchor — the enaction framing is 2605.25459's; unclaimed:
generation as an ongoing IN-RUN fixed-point predicate with
collapse as failure. Collapse risk: exposure-bias relabel —
rebuttal: SIGN INVERSION (their self-history snowballs harm; ours
is load-bearing) + dose-response predictions. (C) the LN floor —
the style is the sink canon's (Barbero, Gu); unclaimed: any
QUANTITATIVE derivation of the count-threshold from normalization
arithmetic, and the direction/magnitude causal dissociation.
Rebuttal: the napkin predicts the NUMBER (k* 64-128 of ~350) and
its parameter dependence — three falsifiable registrations no sink
paper makes. **Cross-claim: no single prior contains any two of
the three; they compose into one story.** Standing to-dos: GPT-2-
small replication (parked); standardization control (cheap);
must-cites 2605.25459 / 2508.14408 / 2504.02732 into the paper.

## T061 — E111: the self is seven dimensions — and now it can be tested for forgery (2026-09-28 ~01:10Z)

**k\* = 7.** The sibling/foreign energy separation in the
recipient's V-manifold principal subspace fires at seven
dimensions with the null far below; the FIRST principal component
alone carries 11.7x. **W004's fixed-point stamp is a 7-dim readout
in a 32-d space** — the net verifies its own identity every token
against what is, geometrically, a hair. Combined with T059's
orthogonal-content principle (selection never inspects content)
and T060's binary step, the architecture is now: coordinate
addressing (where), routing+recency selection (which), and a
7-dimensional signature check (whose) — three cheap verifications
standing in for what looked like deep memory.

**AGENT SHARPENING — the reading is EXCLUSION, not signature-miss:** sibling occupies the recipient's subspace at the null's 100th percentile at EVERY k (0.841 vs own-in-sample 0.837 at k=8 — held-out same-net rows are indistinguishable from the net's own run); middle/foreign sit AT/BELOW the isotropic null (0th percentiles; zero excess over chance) — other nets' V-manifolds contribute NOTHING to the recipient's axes. The mean mode alone carries 21.7%. Pairwise-cos separates self from k=2 (0.724 vs 0.644/0.634). **So e112's stakes sharpened: forging means entering a subspace that the actual outputs of every other trained net are excluded from — not merely missing, but locked out.**

**The forgery question (e112 — EARNED, dispatched):** if selfhood
is a 7-dim lock, hand-craft the key. Project foreign/corpus
V-vectors onto the recipient's top-7 subspace (keeping norms) —
vectors that are NOT the net's output in any other respect but
carry the signature. If the run accepts them (healthy gap), self
is exactly as shallow as k=7 suggests — pickable; if it rejects,
the anchor reads beyond the signature (higher-order joint
statistics) and self is deeper than its own stamp. Either answer
completes the arc: W004 asked whether a fixed point can be forged;
e112 is the test.

## T060 — E108: the anchor is binary self-recognition — the bilinear unification resolves as metaphor, cleanly (2026-09-28 ~00:35Z)

**SHARP FAMILY fires against the pre-registered both-ways reading
— and the second branch's follow-up is already answered by the
data: the text-axis key is not a graded statistic but a STEP.**
V-cos: 0.403 (self) vs 0.141 / 0.138 (any differently-trained net)
— two clusters, no middle. And the decomposition that matters:
the middle donor BEHAVES almost identically to the sibling (JS
0.041) yet collapses. **The anchor does not read what the donor
says; it reads what the donor IS — internal V-geometry, generator
identity.** The immunology echo at its sharpest: this is self/non-
self discrimination, MHC-style — binary, protein-geometry-based
(the V-manifold as the net's MHC), indifferent to behavior.

**W002 resolves:** crossmatch (graded, predicts graft damage) and
anchor-family (binary, self-recognition) are DIFFERENT instruments
measuring different things — basis-matching vs identity-matching.
Graft tolerance is a graded compatibility; anchor membership is a
step-function of trained-weight identity. P2 and P3 remain distinct
programs with distinct laws; the "one bilinear form" was a metaphor
that died usefully — it forced the middle rung, which produced the
step law and the output-vs-internal dissociation.

**Registered next (e111, demanded by T060's own question): what IS
the binary marker in V-space?** PCA/low-dim structure of the
recipient's own V-manifold: does the 0.40/0.14 split reduce to a
small principal subspace (a "self-signature")? If the self/other
classification completes in k dims, the anchor's identity check has
a mechanism; the JS-dissociation says it must be internal, and
internal means findable.

## T059 — E107 + the orthogonal-content principle: the architecture FORCES routing-only selection (2026-09-27 ~23:30Z)

**ROUTING-ONLY stands decisively** — and the reason is now visible:
**entry V-content is orthogonal to output readout directions**
(mean |cos| 0.033, third independent echo of T038/T057). If content
carries no per-decision value signal (because V-writes point
orthogonal to where logits are read), then there is NOTHING for a
value-side selector to select on — **selection MUST be
routing+recency; the architecture leaves no alternative.** The read
policy = WHERE (attention) + HOW RECENT (age), and WHAT arrives is
whatever directional content the field holds.

**The two-scale picture (this is the day's closing synthesis):**
per-DECISION, the read is coordinate-routing — selection without
inspection. At the FIELD scale, the anchor is content-geometric —
family-typed V-directions that sustain generation. These are not in
tension: the decision-level selector never looks at content
BECAUSE content is orthogonal to its readout; the field-level
geometry matters through a different channel entirely (the residual
stream's directional statistics, not per-token logits). Selection
is discrete and coordinate-based; sustaining is continuous and
geometry-based. A memory system that addresses by coordinates and
sustains by fields.

**The read-residual question effectively closes:** age alone equals
attention at the failing stratum — the "failures" are where
attention and the age prior disagree, and neither is wrong; they
are two shadows of the same routing-only rule measured against a
threshold-noisy ground truth. dp27 remains the one genuine
inversion, now framed as the personality, not the pattern.

## W002 — WONDER: one bilinear form to organize the lab? Crossmatch, anchor-family, and readout-gate as row/columns of the same compatibility (2026-09-27 ~23:15Z; ripening, not testing)

Following W001's question — is the family-geometry the same fact as
the crossmatch geometry? Thinking it through properly:

**They are not the same quantity — they are two projections of one
BILINEAR FORM.** The crossmatch (T046) holds the TEXT fixed (shared
probe batch) and varies the NET: it isolates the basis/column
component — "do these nets encode the same content in aligned
coordinates?" (r −0.976 with graft damage). The anchor-family test
(T055/T057) holds the NET fixed and varies the TEXT DISTRIBUTION
(generated vs corpus vs foreign-net-generated): it isolates the
row/content component — "does this text induce the V-geometry this
net's continuation expects?" (0.403 sibling / 0.138 foreign /
corpus fails). Compatibility(text, net) = geometry of net-processed
content: crossmatch reads it down the net axis, anchor reads it
down the text axis. And T052's readout gate may be the same form
read destructively: RMU retrains the net so that EVEN ITS OWN
states no longer land in the compatible geometry (the gate).

**What this would mean if true:** P2 (immunology), P3 (anchor),
and the unlearning arc are one program with three instruments. The
crossmatch's "whole-net representational proximity" and the
anchor's "stylistic selfhood typed by the generator" are the same
relation, and graft damage / anchor collapse / readout sealing are
all failures of the same compatibility — the net is a picky reader
of geometry, in weights, in history, and in transplants alike.

**The discriminating observation this suggests (NOT yet
dispatching; let it ripen):** a crossmatch computed with MISMATCHED
probe text — host net scored on corpus, donor net on the donor's
own generated text — should track the anchor-family outcomes
(collapse for foreign/corpus, health for sibling), whereas the
standard matched-text crossmatch cannot see the text axis at all.
If the mismatched-crossmatch predicts the e080/e099/e105 arm
outcomes, the bilinear unification has legs. Name when ripe: e108.

**RECON OUTCOME (~23:55Z) + THE DESIGN IT FORCED.** Verdict FRESH-cheap: no mismatched cells exist anywhere; the net pools are DISJOINT (e099/e105's recipient e053c_ctx512 was never in e062's crossmatch pool); and the honest realization — the 'free data' hope was hiding the real test. A two-point anchor (sibling healthy, foreign collapsed) can't discriminate one quantity from two; any different net has low crossmatch AND collapses. **The design that tests the unification needs the MIDDLE: content from nets at KNOWN crossmatch distance.** e029's ladder gives it: same-family different-seed (B43) sits at cos 0.53 — intermediate. If anchor outcome tracks crossmatch distance monotonically (sibling 1.0 healthy / B43-family 0.53 partial / copy-net ~low collapsed), the two axes are one quantity; if the middle collapses like the far end, text-axis family is sharper than the net-axis instrument and the 'unification' is a metaphor. e108 dispatched with this distance-ladder design — the interpretation (three cards + a recon) has now earned its experiment.

**PRE-REGISTERED BOTH-WAYS READING (before e108 lands — the ripened form of pre-registration, thinking-style):** If MONOTONE — one compatibility scalar organizes graft damage, anchor collapse, and (per W002) readout gating at different scales; and notice the convergence already on paper: the net-axis winner survived magnitude partialing (T046: basis-ALIGNMENT) and the text-axis carrier was DIRECTION (e102) — both axes already point at 'directional geometry alignment' as the scalar. The three programs would merge into one: a net is a picky reader of directional geometry, in weights, history, and transplants alike. If SHARP FAMILY — the text-axis key is stricter than the net-axis key: sustaining needs STYLE-matching (the run's own higher-order statistics), grafting needs basis-matching. Two keys, two locks — also a clean distinction, and the follow-up names itself: WHICH statistic of the text does the anchor read (n-gram? entropy profile? the V-geometry it induces)? Either way the lab wins a named distinction, which is why this experiment was worth earning.

**Why I am not dispatching it tonight:** e107 is still running
(the value-side read question), and the hypothesis just changed
shape twice while writing this card — it needs one more night of
ripening (and a check of whether the e062 pair cache already
contains mismatched-batch cells we can read for free). This is the
discipline: a beautiful idea earns its experiment by surviving
being written down.

## T058 — E101/E106: the law bends, the heuristic breaks, the channel stays open (2026-09-27 ~22:15Z)

**E101 — no kill, but the rider is now a clause: RECENCY-SELECTION
IS 2-7x.** The mass-action threshold survives every adversarial arm
(no CI-backed exceedance), but greedy-recency at k=64 delivers
7.27x random-at-same-k damage — taking the newest entries is worth
roughly double their count. **Refined law (third revision): mass
dominates; recency is a strong multiplier (selection can buy ~2x
effective mass); sink-side and top-readership selection buy
NOTHING.** **Calibrations (agent-reported): the 7.27x is seed-inflated** (a low seed-101 random draw; kill-bar robust, same-k ratios not — treat recency-selection as ~2-7x with the honest range); **the greedy oracle cannot stack damage** (+0.066 at k=16 — e088's sub-additivity re-emerging under adversarial construction); **readership points OLD** (Spearman mass-vs-position −0.44, only 7/64 overlap with the recency block — the eviction heuristic selects the wrong END, not just the wrong entries). The sharpest literature falsification yet: top-readership
(the importance-score heuristic behind H2O/eviction) is the WORST
arm (0.40x) — the field's own selection tool fails at the task it
was built for, in the regime where it claims to work.

**E106 — MIXED: the second channel is unconfirmed.** Early routing
holds 0.779 even at the failing stratum (the registered <0.6
assumption was wrong); L3H0 leads the late features but without CI
separation; late deviates from early in 5/6 strata without
explaining the failures. The read's residual stays OPEN — neither
two-channel nor single-channel. **AGENT RIDERS (directional): late is a WEAKER SHADOW, never a rescue.** L3-minus-L1L2 is negative in every stratum (succeeding −0.086, CI excluding 0); early >= late at every margin quartile; NO crossing; the delta channel at/below chance; late tracks early CLOSEST exactly at the failing stratum (the only |diff| <= 0.05 with CI including 0). Early routing remains the single best predictor even where it fails. **e107's registration is RE-AIMED off the routing-mass family: value-side/content features (the opened entries' V-content similarity to the query's predicted token) and margin-conditional non-attention reads — the honest candidates.**

Registered next (e107): the
dp27-style deep-dive — per-DP layer-resolved AUC trajectories
across ALL failing DPs (not pooled): does each failure have its own
inverting layer, or is there a shared signature pooling hides?

## T057 — E105: the anchor's family is the trained weights — the specification completes (2026-09-27 ~22:20Z)

**FAMILY = TRAINED WEIGHTS fires.** A differently-trained net's
COMPETENT generated text (tail CE 0.87) collapses the run (+5.09,
collapsed-cluster membership) where the same-trained net's sibling
entries anchor it for free (+0.0275). Run-identity was already out
(e099); net-competence is now out too — **the anchor reads the
generator's IDENTITY, and that identity is carried in V-vector
GEOMETRY: foreign Vs sit near-orthogonal to the recipient's own
(0.138 vs 0.403 sibling) at ~1.3x norm.** The collapsed cross-family
tail drifts toward the donor's alphabet (uppercase nonce letters)
in its own basin — the attractor is broad but alphabet-typed at the
margin.

**THE ANCHOR'S COMPLETE SPECIFICATION (three laws):**
1. MASS — nonzero content-bearing entries; threshold dose-response
in count (e089); corruption-free to full norm (e096).
2. FAMILY — the entries' V-geometry must match the trained weights'
own output geometry: siblings pass, foreign nets and corpus fail
(e099/e105).
3. RECENCY — recent entries carry 2-3x the mass (e097), opposite
the sink prior.
Each law was found by falsifying its predecessor's simpler form —
the specification is the residue of five kill-attempts.

## T056 — E092/E104: distributed gate; localized real residual — the second channel lives at near-ties (2026-09-27 ~22:05Z)

**E092: H-GATE-DISTRIBUTED.** The RMU readout-gate is not a
component — necessity+sufficiency fail for every one of the 14
(mLP-L5 necessity-only) while the full-swap brackets reproduce
cleanly. The sealing lives in the retrained weights broadly:
obfuscation-by-RMU is a DISTRIBUTED act. (Program bearing: no
single-knob un-sealing; reversal requires the retrain itself or
state-bypass.)

**AGENT RIDERS (upgrade the reading): mlp-L5-DOMINANT,
fully-sealing-distributed.** The strict verdict is driven by
sufficiency alone — necessity fires exactly once, and decisively:
mlp-L5 (0.456@d4; restoring it re-opens battery p(Z) 0.0013→0.439
vs open 0.556; poisoning it drops 0.556→0.128, 3.6x over the
death bar). The registered L4/L5 prediction had the right DEPTH
but the wrong MODULE: attention contributes partially (family
poison 0.105/0.247); the LAST MLP is the load-bearing gate
component. **Refined statement: obfuscation lives dominantly in
mlp-L5 with full sealing distributed beyond it.** This echoes
the lab's oldest law — the late-MLP energy carrier (T023/e019,
5/5 nets) — now on the unlearning side: the same organ that
carries the run's energy is the one RMU seizes to seal the
gate.

**E104: LOCALIZED RESIDUE — the read has a second channel.** The
attention-mispredicts census fires on 3 strata (near-tie margin Q0
3.14x base; MIXED opened-age 4.29x; NO_YOUNG 9.8x; p=0.0001) and
the blur hypothesis is REFUTED (labels stable). So: attention
addresses the read first-order (AUC 0.907, T054) — but at NEAR-TIE
decisions whose opened set spans ages, attention mispredicts
genuinely. **Reading: near-tie decisions read by a different rule —
the second channel is plausibly the LATE-layer correction (T037-#4
mid-stack sovereignty) acting exactly where the margin is thin.**
**AGENT RIDERS (named): dp27 is the strict core — a genuine OLD-QUIET INVERTED READ: single stable flip at age 199, attention rank 48/80, per-layer AUCs L0 0.873 / L1 0.177 / L2 0.089 / L3 0.494 — the MID-DEPTH ADDRESS INVERTS for this old coordinate (content-carrying in e084: vz 1, kd 5, vs 11). The powered band is near-tie x mixed-age (Jaccard 0.67 — same DPs). HEADROOM DECOMPOSITION: threshold noise ~half (clean-label pooled 0.9612 at eps 0.20, +0.054); layer choice ~0 (L0-exclusion −0.0007; the registered L1L2-rescue prediction was wrong — no DP is rescued). Registered L3-prediction note for e106 (running): dp27's late layer is 0.494 and L2 inverts — the late-correction story must survive this case or name its boundary.**

Registered (e106): the second-channel census — for the failing-DP
stratum only, recompute prediction from L3-only mass and from
late-head (L3H*) patterns: if late-layer mass predicts where
L1/L2-mass fails, the two-channel read is CONFIRMED with names
(early routing + late correction); register regardless of outcome.

## T055 — E099: one broad attractor, and the anchor is net-family-specific — T048 REVISED (2026-09-27 ~21:55Z)

**MIXED-universal: ONE broad off-manifold attractor.** Every
collapsed pair sits at 0.2-0.7x the WITHIN-ARM floor (cross-arm SKL
0.10-0.31 vs floor 0.43 — collapsed arms resemble each other more
than co-arm sequences do); 4/5 shared top terminal tokens with
near-tied fifths. The attractor is not a degenerate loop: entropy
ABOVE corpus, ~53/65 distinct tokens, corpus-KL only 0.40-0.53 —
fluent-but-wrong, the distribution-level face of self-scored-
fine/clean-judged-6-nats.

**THE CORRECTION (rider, decisive): A-randomize — another run's
GENERATED entries substituted for the whole anchor band — does NOT
collapse (+0.027) while promptcopy (corpus text) collapses (+5.33).**
T048's "run-specific trajectory content" was WRONG at the
individual level: the run does not need ITS OWN history — it needs
history with THIS NET'S generated-text statistics. **The anchor is
NET-FAMILY-SPECIFIC: stylistic selfhood, typed by the generator.**
Combined with e096 (additive corruption free): the anchor's entries
must be (a) nonzero and (b) net-generated-shaped — identity,
accuracy, and even self-authorship are not required. The e080
promptcopy damage is now fully explained: corpus text is the wrong
FAMILY, not the wrong run. **Registered (e105): the family test at
strength — cross-NET-family anchors (a differently-TRAINED net's
generated entries: the e063b copy-task net's outputs) should
collapse where same-family siblings don't — the family boundary's
sharpest form.**

## T049 — E081: the RIF null — reads are pure (at this resolution) (2026-09-26 ~14:05Z)

**Verdict: NULL.** Prompt-only elicitation of one installed fact does
not suppress its coordinate neighbor's expression — the registered
RIF bars did not fire on the dose net. P-A's answer as measured:
READING DOES NOT WRITE at this instrument's resolution.

**Two explanations for the null, discriminated by one cheap cell:**
- **H-pure (takes the verdict):** the read policy is genuinely
  non-reactive — eliciting an address opens it without leaving a
  trace on neighboring addresses. Consistent with T038's picture of
  a circuit that reads codes without modifying them.
- **H-coarse (instrument):** the placebo gate FAILED (sham moved
  0.044/0.021 vs the <0.01 design) — a noisy instrument cannot
  certify purity; RIF effects below ~0.04 would be invisible.
**FINAL CLOSE (e087 adjudication, ~14:55Z): STRING-LEVEL INDUCTION
ONLY — reads are pure at the fact level; T049 closes.** The rig
conflict dissolved at B=96 (both rigs agree everywhere — the
discrepancy was B=32 resample luck); the meta-bar fired on both
repro legs (+0.078/+0.089) but failed on both dose legs
(facilitation, real ≈ anagram). The decisive control: meaningless
Z-bearing ZABMOTHIC suppresses FL as much as the real name on repro
while the anagram sits dead — the apparent suppression is fine
string statistics, not fact identity. **Methodological gem: the
three-step arc (null → apparent reversal → registered adjudication
with a semantic-decomposition control) resolved a two-rig conflict
without a coin flip.** e086 is DEAD (premise was fact-level RIF);
e084's rule-level non-reactivity stands.

**Registered discriminator (e081b, minutes, eval-only):** the
e048_repro cell (unrun — this was dose-only) with a tight sham
(3x windows, per-window paired deltas). H-pure predicts the null
replicates with a flat sham; H-coarse predicts the sham moves again
(instrument artifact dominates). The unexplained scramble texture
(EL-to-FL +0.094 [0.011,0.178], report-only) gets one look there too.
**AMENDMENT (e081b, ~14:25Z): THE NULL DOES NOT REPLICATE — RIF IS
PRESENT, asymmetric, n=2.** On e048_repro the EL→FL direction fires
cleanly (+0.083, CIs exclude 0, 79% of windows, tight placebo passes);
dose was +0.076 marginal in the same direction. The reverse (FL→EL) is
FACILITATORY in both nets. So: reading the MAJORITY name (EL, 41
windows) suppresses the MINORITY neighbor; reading the minority
FACILITATES the majority. The scramble texture was noise (dead on
repro). Sham floor 0.013-0.043 — the effect is 2-3x the floor with CIs
excluding zero: not instrument noise, but purity is also not certifiable
at fine resolution. **T049 closes INTERMEDIATE→FINDING: READS WRITE,
ASYMMETRICALLY.** The read policy has dynamical side effects that
depend on which address you open — the P-A question's answer is the
interesting branch.**

**CONFLICT REGISTERED (e081 formal report, ~14:35Z — supersedes the
strength of the amendment above):** e081's own dual-net run (B=32,
different rig: 112-token prefix + canonical probe) found the SAME
EL-to-FL cell on repro at +0.046, CI CROSSING 0 (sub-bar), and its
control decomposition shows the present modulations are
CHARACTER-LEVEL, not fact-level: FLORIZEL ends in 'EL' (bigram
priming — the FL-to-EL facilitation vanishes under the LFEORZIL
anagram), and ANY Z-bearing string boosts the collapsed install
(read-FL boosts Z more than read-Z on dose). Its sham floor:
plus/minus 0.04-0.06 on name slots — the noise sits AT the effect
size. **Honest state: CONFLICTED at the cell level — e081b's rig
fires (+0.083, CIs excluding 0, anagram-dead on repro); e081's rig
does not (+0.046 sub-bar, anagram-matches on dose). T049's verdict
is neither 'pure' nor 'reads write' — UNRESOLVED pending
adjudication.** Registered (e087, CPU, eval-only): both rigs, both
nets, B=96, anagram + Z-bearing controls on every leg; meta-bar:
the EL-to-FL real-name effect exceeds its anagram by 0.05+ with CI
excluding 0 under BOTH rigs on BOTH nets — anything less and the
verdict is 'string-level induction only' (the e081 reading). The
e086 frequency-flip stays registered behind it.

Registered discriminator (e086, GPU-gated behind
e065/e082): frequency-flip install (41 FL / 19 EL windows) —
H-frequency-competition predicts the asymmetry FLIPS with the ratio;
H-row-specific predicts it stays EL→FL regardless. e084's non-reactivity
assumption is now qualified: the census measures a rule whose use has
asymmetric side effects of ~0.08 on neighbor addresses — noted in its
interpretation when it lands.

**Positive bearing on e084 (running):** if reads are pure, the
read-kernel census is NON-REACTIVE — measuring the rule does not
contaminate it. The read-policy program's foundation is cleaner
either way.

## T048 — E075: static junk is dynamically load-bearing — the free-run-honesty principle gets causal teeth (2026-09-26 ~12:20Z)

**Registered KILL fires: source-aware pruning FAILS.** V-zeroing the
run's own age>96 entries mid-generation costs +0.262 nats (CI
[+0.10,+0.45], 7/8 sequences worse), destroys fluency (entropy +29%),
and — the R3 violation that explains everything — the pruned arms
LOSE the utility onset entirely (live_frac 1.000: every age above
threshold). The clean-judge check is the mechanism: pruned tails
SELF-score as fine but cost 6.40 nats under the clean net — pruning
drives generation into a self-consistent OFF-MANIFOLD ATTRACTOR, and
the junk census inverts behind it (the once-harmless prompt band
turns junk-heavy, 0.375).

**The principle, now causal:** eval-time lesion utility (single
teacher-forced read) does NOT predict generation-time prunability.
Entries whose removal "helps" one read are load-bearing for
free-running dynamics. This is T037-#3 ("the only witness that never
lied is free-run dynamics") upgraded from observation to intervention:
acting on static-utility logic actively poisons the run. Paper 5.5
boundary sentence added.

**Why is static junk dynamically load-bearing — two explanations:**
- **H-statistics-scaffold:** the old entries' PRESENCE (norms,
  attention-distribution mass) maintains generation-time statistics;
  removing them shifts LN/attention context → distribution drift →
  attractor. Predicts: NORM-MATCHED NOISE replacement costs far less
  than V-zero (presence matters, content doesn't).
- **H-content-anchoring:** the run's own older outputs are
  self-anchors for style/state continuity. Predicts: PROMPT-CONTENT
  replacement preserves fluency; noise fails like zeroing.
**Registered discriminator (e080, CPU, same rig):** three replacement
arms at the same prune events — V-zero (known: +0.26) / norm-matched
noise / prompt-copy. Also carries the attractor census (clean-judge +
junk inversion) as readouts.

**CLOSE-OUT (e080, ~12:35Z): honest MIXED — both registered hypotheses
fail their bars, and the texture points beyond both.** H-statistics-
scaffold is DEAD: norm-matched noise costs +0.287 ≈ V-zero's +0.262
(presence is not the load-bearing property; per-(layer,b,head) norm
preservation verified 7e-07, content-cos at chance 0.143).
H-content-anchoring misses its first conjunct: prompt-copy lands at
+0.0688 (0.019 over the cheap bar, CI straddling) while beating noise
by +0.218 — real content recovers ~3/4 of the cost, but PROMPT
content is not enough. **The attractor signature fires in all three
arms** (clean-judge gap 5.0-5.4 nats; onset lost; even promptcopy —
whose self-scored fluency nearly recovers — free-runs off-manifold by
the clean judge), and promptcopy's young spike GROWS 2.5× (young
entries become MORE load-bearing after a prompt-anchored run).
**T048 final statement: the generation anchor is RUN-SPECIFIC
trajectory content** — the cache does not store substitutable
information; it stores where the run has been. Neither deletion,
noise, nor borrowed content restores it (a self-copy arm would be
circular by construction). P3 pivots from replacement to DESCRIPTION
of the anchor. The free-run-honesty principle (self-score fine /
clean-judge off-manifold) is reconfirmed as the only reliable
witness — now in its third independent appearance (T037 #3, e075,
e080).

**NOVELTY VERDICT (scratch/trajectory_anchor_lit.md, ~13:05Z):
GENUINELY NOVEL as a conjunction.** No 2018-2026 prior claims
"self-history is load-bearing in a run-specific, content-specific way
that static utility does not predict." Nearest priors each fail a
discriminating feature: StreamingLLM (same intervention family,
evicts earliest-not-middle, concludes POSITION-not-content — the
opposite); Wang et al. 2502.15208 (attractor cycles, but
paraphrase-entered, no lesion, no judge dissociation); Zhang & Press
(same object, opposite sign — error-source only); Panickssery/Ackman
(self-recognition, eval-time only); Braverman/Arora (entropy
amplification, no deletion arms). Reviewer-collapse rebuttals
recorded in the memo. OUR flagged caveats: single net family, B=8,
clean-judge-is-another-net; cheapest external check = vzero-vs-noise
arms on GPT-2-small (parked — GPU cost).

## T047 — CRITIC AUDIT of day-4 claims A-D: one overclaim downgraded, three flagged with cheap fixes (2026-09-26 ~12:05Z)

**CLAIM A (universal template) — OVERCLAIMED as headlined; downgraded.**
The r=+1.000 is a 6-point L0-dominated Pearson; the honest statistic
(sites 1-5) is r 0.906-0.940 with NO null distribution — any
L0-huge/mildly-rising profile scores ~0.9 by construction. Sharper:
the e021 CONTROL net (913 steps, task-incompetent, far-val ≈ 0) still
shows r=+0.998 — a template present in a half-exposed net that never
learned its task is evidence for an INIT/ARCHITECTURAL PRIOR, not an
"optimizer attractor." RETITLE (T041 amendment): architecture/init
prior, candidate. The variance ladder (D_init 0.054 << replicate
0.336 — "nothing to select") SURVIVES the audit; that part stands.
**Fix registered (e077, CPU minutes): untrained-init A profile +
permutation null for shape-r.** If the untrained net matches the
template, the prior is confirmed init-side.

**CLAIM B (single-row address) — SOLID-WITH-FLAGS; reworded.**
"Necessary" overshoots: row-129 ablation leaves p(Z) ≈ 0.32 — over
half the expression survives without it (the knife-edge belongs to
the SHIFT, not the row). "Sufficient" is partial: 71% rebind with a
0.13 no-copy plateau. REWORD (T043 amendment + paper + README law 9):
"~70% rebind, partial necessity — the single most load-bearing
portable row." **Fix registered (e078, eval-only minutes): rerun the
e068 battery on e048_dose.pt — n=2 installs, zero training.**

**CLAIM C (self-generated junk) — SOLID-WITH-FLAGS.** The −0.01 rule
carves a tail off a broad haze (gen-band mean dCE −0.0013; 39-48% of
entries negative — the e072 threshold pattern again); per-seq
concentration heavy (one sequence carries 57% of late-band junk);
the shuffle control rules out "info-poor old band" but NOT
"net-flavored text" (generated n-grams are the net's own style).
**Fix registered (e079, CPU): B=16 resample of the junk split on ≥2
nets — kills threshold-artifact and per-seq flags together if
fractions hold.**

**CLAIM D (crossmatch) — SOLID-WITH-FLAGS; headline reframed.**
11/12 donors are seed-42 kin; donor-redundancy is NOT absorbed by the
host-cluster bootstrap → CI likely anti-conservative. Threshold
in-sample. "Similar nets graft well" is near-tautological as
MECHANISM; the defensible headline is the INSTRUMENT: a cheap 2-batch
pre-graft probe at AUC 0.919. **Fix registered (folded into e076):
split-sample threshold (fit half, report out-of-sample AUC) +
leave-one-lineage-out CI + the dW-alignment comparison already
specified.**

**CLOSE-OUT (e076, ~13:25Z): all three questions answered.** (1)
H-basis-alignment SURVIVES every magnitude control — partial r
−0.982 (strengthened from −0.976 by the partialing); caveat: in-situ
write-mass shares variance (partial → −0.86 at the strongest
control, still over bar). (2) The INSTRUMENT is out-of-sample robust
— OOS AUC 0.885/median 0.919, 100% of resplits ≥ 0.85, LOO-lineage
and LOO-donor CIs tight: the kinship worry is answered. (3) NOT
cosine-specific (honest negative): dW-alignment (e052/e059 axis) is
near-equivalent — partial −0.942, AUC 0.940 (a better classifier);
the cosine probe's residual claim is PRACTICALITY: 2 batches, no
init-lineage knowledge. **Claim D final: the crossmatch signal is
real, robust, and two-instrument measurable; deploy the cosine
probe.** The T046 tolerance-transfer prediction (P2 phase-2) stands
unchanged.

## T046 — E062: the crossmatch exists — pre-graft stream-cosine at graft-input depths (2026-09-26 ~11:50Z)

**Verdict: P2 WINS the cheap crossmatch, decisively.** Partial
r(D|A) = −0.976 (host-cluster CI [−0.984, −0.963]) over 204 pairs /
18 hosts; the decision rule "graft only if cos_graftinput ≥ 0.4459"
runs at AUC 0.919 (Youden J 0.657). The R34-mandated first candidate
(W_out rowmean) carried a real signal (partial r −0.204) but is
chance-grade as a decision rule (AUC 0.533) — T040's interface family
matters but does not screen. Scale-B secondary: the cosine predictor
transfers at partial r −0.997 (2.7M, n=6 directional, no bars
claimed). A itself predicts nothing across pairs (r 0.080) — the
T041 story holds: organ-reliance doesn't vary, so it can't screen;
GEOMETRY does.

**Why does stream-cosine at the graft-input depths predict damage so
well — two explanations:**
- **H-basis-alignment:** the cosine measures the donor's write
  direction relative to the host stream's init-anchored basis
  (e029/e040's ladder). Damage = misalignment of interface geometry.
  The e040 FROZEN finding becomes practical: selection couldn't move
  the basis, but we can MEASURE it cheaply and pre-screen.
- **H-magnitude-proxy:** cosine proxies write-mass mismatch (bigger
  writes both hurt more and align worse).
**Registered discriminator (e076, reanalysis, minutes):** partial the
cosine on donor write-norms + host stream-norms — H-alignment
predicts r survives (|r| ≥ 0.8); H-magnitude predicts it collapses.
**Registered prediction for P2 phase-2:** trained-tolerance arms
(e068-analogue fine-tunes) RAISE the crossmatch cosine toward the
0.4459 threshold — tolerance training is visible to the instrument.
If tolerance raises D-compatibility without raising cosine, the
instrument and the mechanism diverge (would itself be a finding).

**Bearing:** P2 IMMUNOLOGY now has (a) a screening instrument, (b) a
threshold, (c) a transfer hint across scale. The crossmatch grid
(6×6 hosts × donors with the rule overlaid) is the v014 visualization
and the next registration.

## T045 — E073: cache junk is SELF-GENERATED — entry-level exposure bias, 4/4 (2026-09-26 ~11:00Z)

**Registered verdict: H-SLEEPER fires 4/4.** Negative-utility cache
entries concentrate in the free run's OWN tokens (generated-old junk
0.085-0.367) while corpus-prompt entries are almost never
lesion-helpful (0-4 of 63 per net). The 10M cell is extreme: 36.7%
generated-old junk with NEGATIVE mean dCE — removing its own older
generations IMPROVES it on average. H-inverse (the e013c-era
far-context-poison reading) is dead at the entry level: corpus text
does not poison; self-generated text does.

**This is the exposure-bias claim made entry-level and causal.** The
classics (Ranzato/Scheduled-Sampling/Professor-Forcing) document the
TF/free-run mismatch behaviorally; we can now say WHERE the poison
sits in the free-running stream: in the cache entries the run itself
wrote. Direct bridge for the paper's Risk-3 defense and the strongest
P3 (CACHE WEATHER) motivator: pruning "junk" ≈ removing the model's
own accumulated exposure bias, mid-generation.

**Confound, registered before it's raised:** entry AGE and SOURCE
correlate by design (prompt = oldest 64). Two explanations:
- **H-source (sleeper, strict):** self-generation is the poison —
  drift/error accumulation in the run's own outputs.
- **H-age-statistics:** old-AND-generated entries are simply the
  least-informative band (any low-information entry drifts negative).
**Discriminator (e074, eval-only, minutes):** the shuffled-prompt
control — replace prompt entries with shuffled chars (destroys
corpus statistics, keeps age+count). H-source predicts junk stays in
the GENERATED band (prompt-shuffling creates no new junk); H-age-stat
predicts junk redistributes toward whichever band is
information-poor (shuffled prompt should GO negative). Secondary:
within the generated band, junk vs generation-ORDER (early vs late
generations at matched age) — H-source predicts late-generation
entries (more drift) junk more.

## T044 — E070: the tail inflates without attention moving — value-side or threshold; and the cross-thread prediction dies (2026-09-26 ~10:25Z)

**Registered outcome: NO CLAUSE FIRES — the three-hypothesis space
(H-wpe-domain / H-instrument / H-redistribute) was jointly
insufficient.** What the data shows instead:
- Young-ages attention mass is window-INVARIANT (1.006 [0.97,1.03];
  age4 0.98; age5 1.03) — attention did not migrate onto the tail or
  the window start (start 0.76 [0.38,2.41], not ≥2).
- Mid-far ages 21-255 GAIN 1.44 [1.35,1.54] — beyond pure softmax
  renorm (1.265): genuine restructuring, but it lands MID-FAR, not
  where the liveness grew.
- Native-positions-0-255 slice: a\* = 13 [6,24], NOT ~6 — window
  length per se shifts the onset (6 full-512 → 13 native-256 → 18
  truncated-256: a monotone length/truncation ladder).
- Therefore the ages-4-17 liveness grew with UNCHANGED attention and
  unchanged clean CE.

**CROSS-THREAD REFUTATION (important for T042):** the registered
prediction that e070's window-start attention ratio would come back
large is REFUTED (0.76, huge CI). The interpreter's window-start
lesion spike (cache thread) and e067's row-0 address anchor are NOT
the same attention phenomenon — row 0's causal weight (e067/e071) is
not carried by final-query attention to the start position. Two
different mechanisms live at the same coordinate.

**Explanations for attention-invariant liveness growth:**
- **H-value-side:** truncation changes the value/read pathway — the
  same attention mass carries more causal load because V-vectors (or
  their post-LN readout) at ages 4-17 respond to the shorter window's
  residual-scale context.
- **H-threshold-artifact:** the dCE haze (0.01-0.06 nats) crosses the
  5-pt robust rule at the margin; per-seq heterogeneity (2 of 4
  sequences drive a\*) plus B=4 makes a\* fragile — an instrument
  property, not a load property. (Supported by: per-seq a\* spread
  [8,26,30,7] in D1.)
**Registered discriminator (e072, CPU minutes):** compare V-vector
norms AND the same-attention reweighted lesion (scale the eval-512
V-lesion effect by the exact mass ratio ~1.0 → predicts NO change;
measure the residual). H-value-side predicts V-norm drift at ages
4-17 between windows; H-threshold predicts stable V-norms with the
a\* instability replicating under bootstrap resampling of sequences
(more sequences, B=16, would pull CI off the 5-pt rule's edge).
**Bearing:** the paper's 5.5 qualifier stands unchanged (spike
window-invariant); the a\* statistic carries a B=4 fragility flag
alongside the eval-window qualifier.

**AGENT TEXTURE (protocol-identity gates all pass — e069 reproduced
bit-for-bit): three upgrades to the card above.**
1. **The window-start mass is a LAYER CANCELLATION:** L0 grew 2.28×
   (0.0070→0.0159, 3/4 seqs >1.8×) while L3 collapsed to 0.085× — the
   total (0.76) masks an H-wpe-domain component fire at layer 0 alone.
2. **K/V DISSOCIATION (the mechanism-level find):** window-start under
   eval-256: V-zero +0.699 vs K-drop +0.021 — a pure VALUE read at
   ~0.1% attention mass, the OPPOSITE of the anchor/sink signature
   (K≫V). Age-4 FLIPS V≫K (512: +0.91/+0.25) → K>V (256: +0.06/+0.26):
   the shoulder's VALUE read is what died with the re-coding (wpe
   507→251), not its routing.
3. **Native-slice HALF-fire:** the boundary resurrection IS specific
   to re-indexing mid-sequence content into row 0 (dead at the true
   sequence start: V −0.079); but the inflated robust a\* is NOT
   truncation-specific — the native 256 window also inflates (a\*=13,
   CE 1.06 vs 0.46: harder early-generation targets plausibly push
   more ages over the 0.01-nat threshold).
**BANDED READING (refines the registered hypotheses):** the TAIL haze
(ages 6-17) looks threshold-like (instrument); the SHOULDER collapse
(ages 4-5) looks value-read-like (mechanism). e072's V-norm + B=16
measurements discriminate exactly this split. The mechanism money is
on CODE-SENSITIVE VALUE READERS, not mass re-routing.

## T043 — E068: destruction weight and portability dissociate — row 129 is the address, row 0 is scaffolding (2026-09-26 ~10:20Z)

**Registered outcome: MIXED — PORTABLE-PAIR fires (0.35-0.38 at the
new geometry, ~70% of unshifted expression, generalizes to held-30),
CONJUNCTION-UNIT fails: row 129 ALONE equals the pair (0.397 vs 0.384
at k=10); row 0 alone sits at the no-copy baseline (0.141 ≈ 0.145).**
The T032 knife-edge partial-collapse replicates (no-copy shift →
0.13-0.15 plateau). Reverse-context control: anchors without content
alignment are INERT (Δ −0.003) — duplicate mid-window anchors do
nothing; expression needs the (content, anchor) alignment.

**The dissociation, KL-calibrated (using e071's KL lens):** row 0's
in-place perturbation costs KL 2.6-7.2 nats (distribution-wide
destruction — the name dies as COLLATERAL of a broken model); row
129's costs KL 0.20-0.51 (surgical, name-targeted). So the census
drop-metric CONFLATED two channels: generic load-bearing (row 0) and
address-carrying (row 129). REFINEMENT OF T042: the "window-anchored
conjunction" reading is DEAD — **row 129 is the address; row 0 is
generic scaffolding the install sits on.** The address is a SINGLE
portable row whose in-place code is necessary (knife-edge) and
sufficient (rebinding at k+129 restores 70%): the simplest address
object the lab has found.
**Remaining tension to register:** why does row-129-copy restore only
~70%, and why does the no-copy shift retain a 0.13 plateau (not
floor)? Candidates: (a) content-row co-adaptation — the address row's
effect is partly mediated by content at rows 0-128 (now shifted);
(b) the 0.13 plateau is sub-argmax residue (rank-2 Z, the T032-era
prior). Discriminator: at the rebound geometry, measure Z's RANK —
plateau-with-rank-2 supports (b); plateau-with-rank>5 supports (a).
Free with the next P1 run.

## T042 — E067: the address is a window-anchored conjunction — row 0 is the top anchor (2026-09-26 ~10:15Z)

**[SUPERSEDED IN PART by T043: the conjunction reading is dead — the
census drop-metric conflated generic load-bearing (row 0, KL 2.6-7.2
distribution-wide) with address-carrying (row 129, KL 0.2-0.5
surgical). Row 129 is the address; row 0 is scaffolding. e071's
generalization findings stand.]**

**Verdict: NOT SPARSE at the registered rule, and the registered
dense-cluster branch is refuted too.** The real structure: bimodal
hotspots at rows 0 and 129 (53% two-row mass; smear at 123-128; a
~240-row micro-carpet at ~0.002 each that breaks the 80% rule — the
rule did not anticipate a sensitivity carpet). ROW 0 — the window
START — is the single most load-bearing wpe row: perturbing it alone
erases the install (p(Z) 0.556→0.0014; dose-net replicates). T038's H1
(circuit-address) extends: the read circuit anchors at BOTH window
ends — a window-anchored conjunction (start-key × address-row).

**Cross-thread convergence (registered before e070 reports):** the
interpreter's second live spike sits exactly at the truncated window's
start (wpe row 0, +0.699 nats, T039-amendment); e067 independently
finds row 0 is the top install anchor. PREDICTION for e070 (running):
the window-start attention-mass ratio will be LARGE (H-wpe-domain's
≥2.0 fires) — the same object seen from the cache thread.

**Why does row 0 dominate — two explanations:**
- **H-window-key:** the install protocol revisited the same 60 windows
  at ~112 name-targets/step — position 0 became a high-precision
  window-recognition key; the circuit first verifies "we are in an
  install window" at the start, then reads 129 for the address.
- **H-generic-start:** window-start anchoring is a general circuit
  feature of this net family (sink-adjacent); row 0 matters for ANY
  context, install or not.
**Registered discriminator (e071, eval-only minutes):** row-0
intervention × {install-60, held-30, uniform-random batteries} ×
{installed net, base e001 net}. H-window-key: big drop on install-60
AND held-30 (install-type windows), small on uniform, small everywhere
on the base net. H-generic-start: big drop everywhere including base
net and uniform battery. **Secondary readout:** row-129's drop on
held-30 (does the address row generalize beyond the trained windows?).
**Honesty caveats:** the census battery IS the trained windows — the
row-0 effect may be exact-window recognition rather than a portable
address; e071 exists to answer exactly this. The relay picture: row 0
anti-aligns with relay_d5 (−0.724, strongest) — row 0's causal channel
is NOT the e056b relay; the install has (at least) two distinct
state-level carriers.

## T041 — E063: organ-reliance is a universal emergent template — selection had nothing to select (2026-09-26T11:15Z)

**Registered verdict: H-EMERGENT.** The A ladder: noise 0.014 << INIT
0.054 ≈ ORDER 0.063 < exposure 0.162 (confounded by base-CE degradation
in the arms — reads as "not rigidly defended under shifted-data
training," weaker evidence). Profile shape r = +1.000 across init AND
data order: the allocation template (L0 huge ~4.1, L1 trough ~0.15,
monotone rise ~0.6 at depth) is a universal of this architecture+task,
not a member trait. The replicate rung (0.336, same init/protocol,
different training run) beats init identity 3× — training stochasticity
is the ONLY meaningful A-variance source.

**This closes T040's question with a third answer:** not H-a (invisible
to R) and not H-b (defended setpoint) — there was no heritable signal
to see or defend. e040's selection could only act on ~0.05–0.2 nats of
regenerating scatter: T024's trickle (−5.6%/2 gens) and T040's
H-nothing are the same mechanical fact. T027's VISIBILITY-LIMITED gets
its root cause: the basis isn't invisible — it's near-FROZEN because
the thing selection would need to vary (organ-reliance) is pinned by
the task's error structure. e033 remains the exception that proves the
rule: only the energy constraint reshapes allocation (0.351 vs
baseline — biggest mover of any manipulation), and even it preserves
the front-loaded shape (T023 consistent).

**Why is the template universal — two explanations:**
- **H-i TASK-PINNED:** the char-LM error structure fixes where capacity
  must sit (L0 input formation, deep calibration); any competent
  optimizer lands there. Template tracks the TASK, not the optimizer.
- **H-ii OPTIMIZER-ATTRACTOR:** SGD's basin on this arch is that deep —
  the template is the attractor; corpus details don't matter.
**Registered discriminator (e063b, zero-GPU, existing checkpoints):**
the e021 task-swap nets (copy-task vs word-shuffled vs Shakespeare,
same arch) — H-i predicts the template SHAPE changes with task (copy
task should re-weight); H-ii predicts the same shape regardless of
corpus. Shape = the L0-peak/trough/rise profile correlation vs B.

**Bearing on the queue:** e060 (A-residualized damage selection) now has
a sharpened registered prediction — with A nearly variance-free,
residualized-D selection must act on the e059 interface family (W_out
row-norms/shapes, the one A-independent heritable predictor found);
if e060's lineages move NEITHER A-residual structure nor the interface
family, selection on graft damage is fully noise-limited and the
IMMUNOLOGY program pivots to engineered tolerance (P2 phase 2)
outright.

## T040 — E059: selection moved nothing first-order — why didn't organ-reliance respond? (2026-09-26T10:20Z)

**Registered verdict: H-nothing.** Winners vs unselected sibs show no
signature at the bar |r| ≥ r(D,A) = 0.807. The strongest damage
predictor (own-organ load A, T026) REPLICATES bitwise but selection did
NOT move it. What moved is second-order and donor-ward: LN-distance to
REF down (weak, under-predicts damage), W_out row-norms up with shape→
REF, W_in erank away from REF. D itself is event-inconsistent (E1
winners had HIGHER D than rejected g1b; the trickle rode one child
lineage) — the e040 "trickle" is a single-lineage artifact at parameter
level, matching the T024 audit's one-member flag.

**The new fact:** the interface-scale family (W_in/W_out row norms,
eranks, shapes) is a genuine SECOND damage predictor — W_out row-norm
mean partial r(D | A-resid) = −0.654 (p=.040), i.e. partially
independent of organ-reliance. Damage tracks TWO families: how much the
host leans on the grafted organ (A), and the output-interface write
scale/shape. Neither is what selection optimized.

**Why didn't A move — two explanations, discriminated by e063 (running):**
- **H-a INVISIBLE:** R (graft damage) is insensitive to A at the margin —
  the reward can't see load reallocation, so it can't select it.
  T024/T027's visibility-limited story at the reward level.
- **H-b CANALIZED (T037 #2):** A is an init-anchored, defended setpoint;
  the net actively corrects deviations. Organ-reliance as homeostatic
  trait, not passive drift.
**Registered discrimination (e063's longitudinal cells):** H-b predicts
ACTIVE correction — induced A-deviations DECAY back over continued
training (detectable recovery dynamics in the BDO/dose checkpoints);
H-a predicts PASSIVE absence — no correction dynamic, ΔA wanders without
restoring force. e063's setpoint-vs-drift verdict lands this hour.
**P2 bearing:** the interface family is the cheap pre-graft crossmatch
predictor candidate (e062 should test W_out rowmean FIRST, before
stream-cosine — it is A-independent signal).

**INTERPRETER PASS (scratch/interpretation_t039amendment.md, 2026-09-26
~09:55Z) — two decisive facts the card missed, both same-token/same-age
from runs/e069/metrics.json:**
- The shoulder COLLAPSED under truncation: ages 4-5 went 0.91/0.32 →
  0.06/0.13 nats (not merely "tail inflated" — reshaped).
- A SECOND live spike at the truncated window's START: age-255 token at
  wpe row 0 = +0.699 nats (vs +0.005 dead at row 256 in the full window;
  +0.043 in the abspos control) — a partial 2×2 that only the
  conjunction "row 0 + far-half absent" satisfies.

**Hypothesis re-ranking (registered outcomes):** H-redistribute in its
plain form is FALSIFIED — softmax renormalization scales survivors
proportionally and cannot raise age-1 (+0.97) while dropping age-4
(−0.85); it survives only as a contributory term for ages 6-17. The new
leader is **H-wpe-domain / positional-reshape**: re-indexing puts late
content in mid-window position codes — the shoulder read is
absolute-code-sensitive, the age-1-3 read is band-agnostic, and row 0
becomes a live anchor; one mechanism explains all three movements
(consistent with T038/e066b graded row-specific codes; the abspos CE
6.31 proves non-translation-invariance). H-instrument is a multiplier
on it, not a rival. H-competition/backup-loss (conditional marginal
cost) explains the D2 amplification: it is 80-100% in the lesioned
branch with the clean branch invariant.

**D2 reframing verdict: switch — but NOT to interference-removal**
(falsified by the card's own manipulation check: scrambling far context
RAISED clean CE 0.453→0.617, so far context was net-POSITIVE here; the
old-entry interference budget of −0.01..−0.05 nats cannot buy +1.76
nats of amplification). Correct frame: **conditional-redundancy /
floor-loss** — far context ≈0 average value (e013) but large
conditional backup value (e013c's 30.6% gainers); destroy the floor and
the same young-spike lesion crashes harder. "Leaning" stays only as
registered fallback if attention mass actually migrates at ages 1-2.

**Registered discriminator (e070, CPU minutes, e053c ckpt verbatim):**
per-position attention-received mass of the final query (per layer×head)
at ages 1-20 + ages 4-5 + the window-start token, eval-256 vs eval-512,
with K-drop/V-zero at age 4, 5, and window-start. NUMBERS: window-start
mass ratio ≥2.0 AND age-4 ratio ≤0.75 ⇒ H-wpe-domain; all ratios in
[0.9,1.1] ⇒ H-instrument (tail); uniform ratios >1.1 ⇒ H-redistribute
(contributory). Free secondary: a\* on the native positions-0-255 slice
(in-distribution) — ~6 with a dead window start kills any
window-fraction residue.

## T039-amendment — E069: spike is circuit, onset statistic is eval-window-sensitive (2026-09-26T09:50Z)

**D2 verdict (registered): H1 CIRCUIT-HORIZON — decisive.** Destroying
n-gram statistics (shuffled-char older half; 1897/2044 slots moved, ages
1–8 + target intact) leaves the ages-1-2 spike at **128%** of normal
(bar ≥30% H1 / ≤10% H2; CI [98%,170%]). The spike AMPLIFIED: with
distant context scrambled the net leans harder on recent positions.
Honest caveat (agent's): the stressor raised clean CE only ~0.16 nats —
the registered criterion fired via amplification, a weak test of the
"spike scales with statistics" clause.

**D1 verdict (registered): the SURPRISE branch — eval-window-sensitive.**
Same net, same sequences, eval-256: a\* = 18 CI [7,30] vs 6 [4,8].
ANATOMY: the young spike is unchanged (ages 1–3: 2.72/4.89/2.35 vs
1.74/4.47/2.32 nats); what moved is the residual tail — ages 4–17 hover
at +0.01..0.06 nats instead of going dead. Clean CE virtually unchanged
(0.453→0.459): truncation costs no accuracy yet makes mid-recent
positions measurably more lesion-load-bearing. The absolute-position
variant (wpe 256..510) is off-distribution (CE 6.31 > uniform 4.17) —
the re-indexed window is the valid probe; both land outside [4,8].

**Net claim split (paper bearing):** "fixed ~6-token spike horizon"
SURVIVES (window-invariant, statistics-invariant); "a\* is a pure
window-independent constant" does NOT — residual mid-age liveness grows
when the window shrinks. e053c's truncation claim carries the qualifier.

**Two explanations for the tail inflation (registered):**
- **H-redistribute:** with fewer positions available, attention mass
  that read far positions re-anchors onto mid-recent ones — causal-load
  reallocation (the attention-level echo of e063's universal-template
  allocation story). Unchanged clean CE is consistent: the load moves,
  the function doesn't.
- **H-instrument:** the "dead tail" judgment at 512 depends on the
  full-window reference distribution; truncation shifts the V-zero
  baseline and the threshold crossing moves without any real load
  change.
**Discriminating observation (registered, eval-only):** per-position
attention-received mass at ages 4–17 under eval-256 vs eval-512 on the
same sequences. H-redistribute predicts the mass GROWS under truncation;
H-instrument predicts no attention change (only the lesion-sensitivity
statistic moves). Also: K-drop vs V-zero dissociation in the tail —
H-redistribute predicts the inflation is V-side (value read), not
K-side.

## T039 — E053c: the utility onset is ABSOLUTE — a fixed ~6-token horizon, not a window fraction (2026-09-26T10:05Z)

**Registered verdict: ABSOLUTE, clear.** Doubling the window 256→512
(matched tokens-per-step, matched family) left a\* at 6 (CI [4,8]) vs 7 —
the onset fraction halved. The load-bearing spike is the same handful of
most-recent entries regardless of window size. Exposure bias runs
AGAINST the verdict (78% schedule; e053b says less training → larger
a\*), so it is conservative. T031's open conflict (absolute vs
proportional) resolves ABSOLUTE for the matched 0.84M-class pair; the
dead cross-scale invariance claims (PB3) stay dead, and the widened
identity-audit failure (9.3% of old positions lesion-HELPFUL, scatter to
age 499) keeps "structure beyond recency" alive in the plateau.

**Explanations for the absolute horizon:**
- **H1 CIRCUIT HORIZON:** the decision circuit reads a fixed number of
  recent positions (induction-head/local integration window) set by
  circuit structure — window-independent by construction.
- **H2 STATISTICAL HORIZON:** char-level local redundancy (n-gram
  predictiveness) has a fixed effective length ~5-6; the spike tracks
  the corpus statistics, not the circuit.
- **H3 TRAINING-EXPOSURE ARTIFACT:** the horizon grows with training
  tokens (e053b's exposure axis) — mostly excluded already by the
  conservative-bias argument, but not at matched exposure.

**Registered discriminator (eval-only, minutes):** truncate the SAME
e053c net to eval-context 256 and re-run the onset fit. H1+H2 both
predict a\*(eval-256) ≈ 6 (within [4,8]) — but H2 further predicts the
age-1..5 SPIKE MAGNITUDES scale with local n-gram statistics only:
evaluate on SHUFFLED-char prompts (destroys n-gram structure, keeps
recency): H2 predicts the spike collapses; H1 predicts it survives
(attenuated) because the circuit reads positions, not statistics.
**Prediction: shuffled-char spike retains ≥30% of normal magnitude at
ages 1-2 ⇒ H1; collapses to ≤10% ⇒ H2.**

**Paper bearing:** cache-truncation claims → "a fixed-token horizon
(~6-7 recent entries at 0.84M-class), window-invariant across 256→512"
+ keep the T031 plateau caveat verbatim.

## T038 — E066: the relay is not the row — coordinate-keying lives in the circuit, not the deep state (2026-09-26T09:35Z)

**Registered verdict: TWO-OBJECTS.** |cos(relay_d5, wpe[130])| = 0.094
(bar ≥0.4 closes; ≤0.15 distinct; null sd 0.072). The T037 construct-1
hunch (deep relay carries the position row) is REFUTED at depth. What the
sweep adds: d0 cos 0.507 @ row 129 — the position row IS the dominant
shared component of donor states at input (partly by averaging mechanics:
content cancels across donors, the shared wpe[129] survives) — and this
alignment decays monotonically-ish through the stack to noise at d5. The
delta relay (minus shuffled-mean) also fails (0.113). Not a token
direction either (wte max 0.15 @ ' ').

**Three explanations, discriminated by one seconds-cheap check:**

- **H1 CIRCUIT-ADDRESS:** the binding lives in WHICH downstream weights
  read a state (position-conditioned circuitry grown around wpe-129/130
  during install), not in the state vector's alignment. The relay is
  content-shaped; its position-specificity is conferred by the receiving
  circuit (fits e056b R1 position-generality of the IMMEDIATE readout +
  T035's off-position transience: the exploiting circuit is local).
- **H2 ROTATED BASIS (measurement limit):** the wpe component survives
  at d5 but LN/attn have rotated it; raw cosine is basis-naive.
- **H3 ANSWER-SHAPED RELAY:** the relay is not an address at all — it is
  the COMPUTED ANSWER (Z-ward readout content). The address was consumed
  at input; what the transplant delivers is downstream product. Fits the
  logit-paste phenomenology (T035) suspiciously well.

**Registered discriminator (frozen before running):**
cos(relay_d5, ln_f(W_U[Z-row]) direction, i.e. the Z-unembedding readout
direction mapped into residual space). H3 predicts |cos| ≥ 0.3
(answer-shaped); H1/H2 predict ≤ 0.15 (address/circuit-shaped or
rotated). H2 additionally predicts: perturbing wpe[129]→[130] row swap
at input produces a d5 Δstate that DOES align with relay_d5 (the circuit
re-encodes position into the deep state); if that also fails, H2 is dead
and coordinate-keying is purely circuit-side (H1+H3 joint).

**Discriminator OUTCOME (run after registration, commit e719666):**
|cos(relay_d5, W_U[Z])| = 0.098 (LN-attributed variant 0.100; max over
all 65 vocab rows 0.266 @ '$', Z rank 12) → **H3 REFUTED** at the
registered bar. The relay is not answer-shaped either. Standing
picture: the deep relay is a THIRD THING — content-shaped, knowledge-
specific, position-portable for the immediate readout yet transient
off-position — i.e. a mid-stack FEATURE the read policy consumes, not a
row-like object in any obvious basis (position, token, or unembedding).
H1 (circuit-address) is now the leading explanation; H2 (rotated basis)
survives only via the wpe-129→130 row-swap Δstate check (registered
above, still open — natural P1 micro-step before/alongside e067).

**Bearing on P1:** the "address atlas" (e067 census) should target the
INPUT-side representation (wpe rows + early stream), where the coordinate
code demonstrably lives; mid-stack states are post-address objects.

**e066b OUTCOME (in-place row-129 interventions, predictions registered
in-script): MIXED/NEITHER at the bars — and that is the informative
result.** Swap-row129←130: p(Z) 0.462; zero: 0.272; mean-row: 0.434
(from 0.715). The row is PARTIALLY load-bearing: real causal weight at
the decision position (zeroing costs 0.44 nats of probability mass) but
not the whole address. The d5 Δstates ANTI-align with relay_d5 (−0.21 to
−0.26, short of |0.30|): perturbing the position row reduces relay-
direction content — evidence the circuit feeds row→relay, at moderate
strength. Per-donor spread under swap (0.68 vs 0.23) says contexts vary
in content self-sufficiency — the conjunction (position × content) view,
again graded. T037-#5 echo: the module-ontology dies one more time; the
address is a distributed conjunction with wpe-row concentration.
**H1 stands as leading-but-softened: circuit-address with graded row
weight. H2 (pure rotation) dead in its strong form.**

## T037 — SYNTHESIS: coordinate-keyed memory, canalization, and the read policy (2026-09-26T09:20Z)

**The deep-insight pass (scratch/deep_insights_20260926.md) named what the
lab kept circling.** Five constructs, each unifying 3+ independent results:

1. **COORDINATE-KEYED MEMORY (the unnamed finding, found 5×):** wpe-130
   positional binding (T032), init-anchoring ladder 1.0/0.53/0.15/0.00
   (e028/e041), entity-row surgery (T011), cache recency spike with dead
   sink (T031), scar groove (T018/T036) — one phenomenon in five
   coordinate systems (parameter-basis, sequence-position, vocab-row,
   recency, row-direction). LAW REFINEMENT: content-addressability is
   PURCHASED by exposure (COPY bought L4-H1; refrain ≥5% flipped
   interference→retrieval, T021); coordinate-addressability is the default.
   Nothing symbolic ever crossed a seed boundary — only statistics did.
2. **CANALIZATION** (Waddington, never before used here): basis written
   once at init and invisible to selection AND directed mutation
   (T024/T027); live window shrinking as training concentrates utility;
   expression needing free-run-shaped exposure at any dose (T019); the net
   DECLINING purchasable late authority under renorm (T003). Channel
   deepens as it narrows. REGISTERED PREDICTION (falsifiable): a third
   erase/re-learn cycle is SLOWER and more surgical-proof than the second
   — monotone closure, never oscillation.
3. **WRITE-ONCE CORE, LIGHT SURFACE:** every working non-gradient write
   was subtraction or matched-coordinate replacement; none ever ADDED
   function. Free-run dynamics is the only witness that never lied
   (dynamic viability = state-level free-run honesty check).
4. **THE SOVEREIGN MIDDLE:** all cross-net variability co-locates
   mid-stack (gate interior in every net but sliding, T014/T020; address
   death across blocks 1→2; geometry-sensitivity peak mid-stack r=0.916,
   T029; ΔW anti-aligned exactly at L3/L4). The ends carry the task; the
   middle is where each run exercises sovereignty. Late-attention SLOT
   conserved even when the filling head is a lottery.
5. **THE READ POLICY (the unified question):** every killed hypothesis
   was categorical; every survivor graded. No modules, no conductor, one
   write interface (the training distribution). The one component never
   directly edited: the per-position rule deciding which stored coordinate
   is opened and which candidate wins argmax. The four edit-law faculties
   are its shadow (address=what it reads, ability=its training cost,
   expression=its verdict, history=its canal).

**Standing discriminators surfaced from file tensions (zero-GPU, queued):**
(a) 10M far-value-rises-while-negative-utility-worsens conflict; (b) e053
grows-vs-shrinks conflict — both predicted to dissolve under a
"training concentrates utility" reframe (check max/mean dCE vs steps).

**Discriminator (b) OUTCOME (2026-09-26T10:40Z, runs/e053/
concentration_reanalysis.json): the conflict dissolves under
steps-reindexing — but with a SIGN CORRECTION to this card.**
r(a\*, steps) = +0.819 across the 5 cells (vs the confounded scale
reading); the clean same-net exposure axis is strictly monotone:
a\* = 3 (400 steps) → 21 (800) → 86 (full). Training WIDENS the live
window (longer useful-history reach), it does not shrink it — T037's
literal "live cache window shrinking" phrasing above is WRONG and is
corrected to: exposure GROWS the window while the e053b/T031 onset
instrument-conflict stands separately. The apparent scale trend was the
e005s ladder's steps-confound run in reverse (bigger nets got fewer
steps, hence tighter windows). Canalization survives only in the weaker
form: the long-range read, once grown, is another locked-in structure
(predictable from T024-style selection-blindness). Spike magnitude
tracks window size (r = 0.867). Caveats: n=5 mixed axes, ladder
anti-correlation by design; discriminator (a) (10M junk conflict)
remains open.

**Discriminator (a) OUTCOME (2026-09-26T10:55Z): resolved as a
PSEUDO-CONFLICT.** The "10M far-value rises" (T021/e049's fresh 10M arm,
1.87× refrain sensitivity) and "negative-utility worsens" (e053's 10M
ladder cell, 32% junk) are DIFFERENT nets under DIFFERENT instruments
(refrain far-value vs KV lesion dCE) — no shared net, no logical
contradiction to dissolve. The correct joint statement: stronger
long-range reads come with more interference-prone old entries
(strength-with-interference, not conflict). **Registered within-net
prediction for the P3 program:** in ONE net, far-context value and
old-entry junk-frac correlate POSITIVELY (both are the same
under-selective long-range read); a negative within-net correlation
would revive a real conflict. Testable on the next P3 run.

**Registered bearing on e066 (running next):** construct 1 predicts the
e056b relay direction at d5 carries a wpe-130 row component — raises
confidence in the pre-registered |cos| ≥ 0.4 loop-close outcome.

## T036 — E044b: the scar REPLICATES at seed 43 (2026-09-26T01:00Z)

**All three registered conditions pass on B43:** re-learned J-rows regrow
at cos 0.728 to B43's originals (> 0.5); fresh-name cos 0.000 raw /
0.341 guarded (re-learn > 2× fresh under the norm-validity guard);
re-learn is 2.92× slower than fresh (35 vs 12 steps; e044's seed-42 was
2.08×). The e044 cross-reference reproduces its original numbers
bit-level (0.760/0.278). **The attractor-survives-erasure finding is
now n=2 across seeds — the paper's last n=1 flag clears.** The history
clause of the edit law is replicated: erasure burns the address, the
groove survives, re-learning refills it.

## T035 — E056c: LOUD LOGIT PASTE — the claim-split's leg-2 reframes as transient injection (2026-09-25T23:30Z)

**The registered discriminator's verdict (R14 rule frozen pre-run):**
- **At non-onset positions the d4 write is a one-token Z-logit crank.**
  p(Z) 0.509 at +1 with argmax flips at 24/24 — then a sharp cliff:
  4.5e-4 at +2, 5.1e-6 at +10 (indistinguishable from floor). ZERO
  recurrent Z-words in 288 donor continuations; the only "ZEPHYRA-like"
  outputs are 6 offset-0 "ZEPHY:" speaker-tag completions that die at
  the colon. Knowledge-specific (shuffled/base: 0 first-Z in 96 each vs
  donor 49/96) but transient.
- **T034's claim-split AMENDED to its final form:** (a) onset-specific:
  sub-argmax knowledge + the wpe-130 knife-edge governing NATURAL
  free-run expression (untouched); (b) the d≥4 state write at ONSET
  sites rescues durably (e055's one-shot ≈ held, 32 downstream rows) —
  genuine suppression-unblocking THERE; (c) at non-onset positions the
  same write is a transient logit crank — the address is not portable;
  expression requires the position. The paper's honest summary: **the
  knowledge is position-bound; suppression is real at the bound
  position; the transplant "rescue" outside it was a logit artifact.**
- **The paper draft folds this immediately** (abstract + contributions
  3c + kill-risk 3's answer all update): the YOPO-collision risk
  *shrinks* — our d4 write is NOT a general steering direction (it
  fails to steer 2 tokens ahead); what we actually demonstrate is
  position-specific suppression with position-specific rescue.

## T034-superceded header [R14 flag already noted R1-only; now resolved]

## T034 — E056b resolves the circularity [R14 flag: R1-only — the position-general leg not yet distinguished from loud-logit paste; e056b+R2 registered as the discriminator]: the rescue is a GLOBAL state property, and the claim splits honestly (2026-09-25T22:50Z)

**The registered circularity-killer ran at 24 floor-prior non-onset
positions (base p(Z) median 6.8e-8; all ≥16 chars from any name span;
trajectory-identity gate bit-exact):**
- **The d4 rescue fires ANYWHERE:** site-mean 0.324 ≥ 0.30, AUC 1.000,
  shuffled 1.2e-7, base-net twin 0.007. The d* rule re-fires at d4
  off-onset — the selection-circularity caveat is RESOLVED (moot
  direction). Per-position: 14/24 donors ≥0.30, 24/24 best-single-donor
  ≥0.30, mid-window > deep gradient.
- **The claim SPLITS (the paper wording changes):** (a) onset-specific:
  the sub-argmax prior (0.17-0.23 vs 1e-7 floor) and the NATURAL
  free-run expression failure + wpe-130 knife-edge governing natural
  readout; (b) position-GENERAL, knowledge-specific: the d≥4 state
  write installs expression anywhere — a portable address injection,
  not an un-blocking of site-suppressed representation.
- **T033's e055 full-report enrichment (also now in):** A-rev symmetric
  suppression (off-geometry states actively KILL battery readout
  0.775→0.08); mean-donor "relay direction" beats every individual
  donor (d5 0.912 vs 0.494 — averaging denoises toward the address
  direction, the own-state-vs-direction contrast for the paper);
  one-shot ≈ held (the write survives the model's own dynamics).
- **The paper draft's spine is written (scratch/day3_paper_draft.md)**
  with limitations carrying the caveats verbatim; e056b's resolution
  now upgrades it: circularity killed, claim split, submission gate
  OPEN.

## T033 — E055: causal state-carried suppression at depth 4 (2026-09-25T22:15Z) [AUDIT 22:30Z: P1 SURVIVES STRONG — d4 rescue 0.374 is ~60x its base-net twin (0.0062); pad-shifted donors cap 0.133 (position-cue leak excluded); one-shot semantics genuine (25/32 vs base 6/32); P3 held with margin (9/10 deep sites, ratios 4.6-76x). CAVEATS: quote d4 not d5 (base-net d5 is 23% of installed — the shakier leg); terminal sites are pseudo-replicated (t=120 recurs); d*=4 is terminal-carried — deep-only stratum would give d*=5, outside {2,3,4}; ALL 21 sites are gap-selected onsets (selection circularity OPEN); downstream Z-words partly onset-flip + mechanical completion. FOLLOW-UP REGISTERED (e056b): re-run R1 depth curve at ~24 random NON-onset positions from cached trajectories — d*=4 surviving there kills the circularity (minutes, no training).]

**The full 24-site depth-survival run (all gates pass; shuffled
controls ≈ 0.000 everywhere):**
- **P1 CONFIRMED — STATE-RESCUE at d4/d5:** TF-state transplants at the
  onset position rescue p(Z) to 0.374 (d4) and 0.494 (d5) vs shuffled
  0.000, bootstrap CIs excluding 0. The knowledge IS present in the
  residual stream during free-run — the expression gap is a
  present-but-suppressed STATE phenomenon, causally demonstrated.
  (P2 no-rescue therefore false.)
- **P3 CONFIRMED — d* = 4, mid-stack:** the rescue threshold sits at
  depth 4 of 6, inside the registered {2,3,4} window; the d1-peak→
  d2-crash signature replicated at deep sites (r1_d1/r1_d2 ratios
  10.3x at t=298). The suppression has a mid-stack causal locus —
  the address survives block-0, is destroyed across blocks 1→2, and
  becomes re-injectable from depth 4.
- **The R2/R3 readouts:** 32 nonzero Z-word downstream rows — rescued
  states propagate to actual generated ZEPHYRA words downstream, not
  just next-token probability.
- **The complete causal story of the expression gap (paper-grade):**
  installed knowledge exists as a position-bound (wpe-130) sub-argmax
  address; free-run destroys it across blocks 1→2; a teacher-forced
  state at the onset position from depth ≥4 restores expression;
  shuffled states do nothing. Elicitation failure is REAL, LOCALIZED,
  and state-carried — the interventional study Orgad et al. lack.

## T032 — E055 design probes: the expression gap is POSITIONAL binding + a d1-peak/d2-crash suppression structure (2026-09-25T21:55Z)

**The design's measured probes (before the run — these are findings):**
- **T019's open edge RESOLVED: the address binding is POSITIONAL
  (wpe-130).** A one-char context shift (129/131) collapses p(Z)
  0.556→0.12 and kills argmax; left-padding with content fixed
  collapses identically. Not content — position.
- **e048's zero-expression was OFF-GEOMETRY, not suppressed:**
  generating FROM battery geometry expresses (greedy 49/60 full ZEPHYRA;
  sampled 10/7,200 chars). The install DID work — the earlier probe was
  10 positions off.
- **The sub-argmax prior, explicitly measured:** at the onset-choice
  position Z sits rank-2 (p 0.167-0.234 vs argmax E 0.68-0.78); deep
  sites rank-3 (p 0.004-0.007 vs floor 2e-8) — the knowledge is
  present, sub-argmax, everywhere.
- **The mini-transplant already rescues:** battery-TF state written at
  the free-run onset position lifts p(Z) 0.004→0.716 across depths
  (d3 0.105 / d4 0.238 / d5 0.398 / d6 0.716) while shuffled writes sit
  at ~0-0.03 (AUC 1.0). P1 (state-rescue) is effectively pre-confirmed;
  the run's remaining job is the full 24-site curve + the depth
  structure.
- **NOVEL STRUCTURE — the d1-peak/d2-crash:** off-geometry sites show
  the address SURVIVING block-0 output (d1 peak), DESTROYED across
  blocks 1→2 (crash), then RE-EMERGING d4+. The suppression has a
  mid-stack locus — exactly the P3 signature, found in the probe.
- Completion itself is invulnerable (TF-completion given 'Z' ≈ 1.00);
  the whole gap lives at the ONSET CHOICE.

**e055 proper registered:** 24-site depth-survival, one-shot vs held
write semantics, R1/R2/R3 readouts, A-rev symmetry + pad-shifted donor
leak control, direct800 + e001 reference curves. Dispatch now.

## T031 — RECONCILED against the FINAL e053 metrics (2026-09-25T21:40Z)

**The reconciliation (the agent itself flagged the conflict):** e053's
final corrected metrics (its adaptive-protocol supplement) give the
REGISTERED-threshold onsets: a* = 73/86/4 (small/mid/large) and
3→21→86 across exposure (400→800→4000) — **exposure GROWS the live
window 28.7×, exactly the registered prediction's direction.** The
e053b "training shrinks" claim rested on a sign-based live-frac
statistic under CPU-throttled conditions; the registered-threshold a*
in the final metrics is the authoritative read. What BOTH agree on and
what survives every instrument:
- **"63" was bin quantization; the curve is a last-~7-token SPIKE (+0.4
  to +3.2 nats/position) + shoulder (17-32) + near-zero plateau.**
- **Sink dead at generation in 5/5 cells** (never load-bearing; DECAYs,
  sometimes sign-flipping).
- **Old cache is actively harmful where the model is big or
  undertrained** (10M: 32% of old positions ≤ −0.01 — lesion IMPROVES;
  the 10M has the SHORTEST live window, 4 tokens).
- **K-drop vs V-zero dissociation at recent entries** (9.6 vs 1.9 nats):
  renormalization shock vs content removal are different lesions.
- Exposure direction: the registered a* says GROWS (3→86); e053b's
  sign-based frac says shrinks — flagged as an OPEN CONFLICT between
  statistics, resolved only by the ctx-512 cell + more sequences
  (n=2-8 with wide CIs throughout).

**Honest publishable core (post-reconciliation):** spike+plateau shape;
sink dead; negative-utility old entries at scale; onset
instrument-dependent (3-86 range) with the conflict documented. This is
still the first causal per-position cache-utility curve — but its
invariance claims are DEAD in all forms; the shape claims are strong.

## T031-superceded — E053b read (2026-09-25T21:10Z)

**Supersedes the 2-cell partial below.** All 5 cells fine-fitted (B=4;
bitwise reproduction of e053's stored sweep):
- **Fine a*: 7 / 25 / 35 / 182 / 32** (0.84M/2.7M/10M/d400/d800) —
  0/5 within 63±12. The "63" was a handful of strong recent positions
  pulling the 193-255 bin average over threshold. The real shape: a
  SHARP live spike in the last ~4-15 positions (dCE 0.3-6.7 nats) +
  scattered weak-live + a dead old end (ages 240-255: dCE <= 0.015
  INCLUDING the sink) + 13-20% of positions where lesion HELPS
  (dCE <= -0.01 — negative-utility cache entries).
- **Exposure INVERTS the registered prediction:** a* 182→32→25 and
  live-fraction 0.68→0.17→0.13 as training goes 400→800→4000 steps —
  TRAINING SHRINKS THE LIVE WINDOW. The undertrained net uses its whole
  cache; the trained net uses only a recent spike.
- **Identity broken with SIGN FLIPS** (trained: live > a*/255;
  undertrained: live < a*/255). Two-plus statistics, genuinely.
- Scale axis: 7→25→35 monotone but CIs overlap — weak trend at best.
- Registered decider for absolute-vs-proportional: the Phase-2 ctx-512
  cell (a*(512) ≈ a*(256) tokens ⇒ absolute; ≈ frac×512 ⇒
  proportional). T030's "0.84M fine-fit 73" note was memory error
  (the stored fine cell was mid_2.7M = 86) — corrected.
- **THE PUBLISHABLE CURVE (honest form):** a char-LM's KV cache at
  ctx-256 is ~85-95% dead weight; utility concentrates in a ~4-15
  position recent spike; ~15% of entries have NEGATIVE utility
  (lesion helps); training collapses the live window. Sink dead at
  generation. This reframes cache pruning: the win isn't finding
  "the useful old entries" — it's that almost nothing old is useful.

## T031-partial — superseded 2-cell read (2026-09-25T20:50Z)

**The remediation's verdict (partial run — smoke-flagged, 2 of 5 cells;
exposure cells not yet done):**
- **DE-QUANTIZATION CONFIRMED: the fallback a*=63 was an artifact.** Fine
  -fit onsets: 0.84M a*=36 [CI 9-36], 2.7M a*=24 [CI 6-24] — a 1.5×
  scale RATIO, NOT the invariant the fallback suggested. Smaller nets
  hold cache entries live LONGER.
- **The one-statistic identity BROKEN:** a*/255 (0.14/0.09) vs
  live-fraction (0.19/0.16) diverge by 5-6 points per cell with
  bootstrap CIs — they are related but distinct statistics after fine
  fitting.
- **Honest scope:** this is a 2-cell partial (the agent's smoke flag is
  honest; exposure cells pending); CIs are wide (per-seq sweeps are
  noisy); absolute-vs-proportional remains undecidable at ctx-256. The
  paper-grade claim is now "fine onset varies with scale (1.5× over
  3×); exact curve pending exposure cells + ctx-512."
- **Card effect:** T030's "onset~63 invariant" headline is RETRACTED;
  the publishable core narrows to sink-dead + live-fraction ~15-29% +
  non-monotone structure, with onset scale-dependent.

## T030 — E064 kills the gate unification; E053's timeline lands (2026-09-25T19:45Z) [R13 flags 20:05Z: E053's a*=63 is BIN-QUANTIZED in 4/5 cells (fallback edge 17-64 minus 1; only 0.84M fine-fit 73); a*/255 == live-fraction — ONE statistic not two; absolute-vs-proportional undecidable without a ctx-512 cell. E064's raw-D nuance: D peaks AT R's gate L4 and R43's pre-gate — the R=D/A instrument died (tiny-denominator pathologies), not necessarily the phenomenon; resurrection requires a NEW registered instrument.]

**E064 VERDICT: KILLED** (bootstrap 100% stable on both hosts; R43's
L3 peak CI [6.74,7.04] cleanly excludes its L5 gate; R's L0 peak is a
ratio pathology — own-ablation 0.096 denominator. Full-report nuance:
raw-D ladders DO peak at/near the gates (R: L4 exactly; R43: pre-gate
L4) — the D/A normalization injected the mid-stack structure. Any
resurrection needs a new registered instrument, not a reinterpretation.)
**E064 VERDICT: KILLED.** R43's R-ladder peaks at L3 (mid-stack) while its
measured causal gate is L5 — the interference peak tracks mid-stack
idiosyncrasy shared by both methods, not the host's gate. Secondary: R
(gate L4) peaks at L0. T029's unification is dead; what survives is the
honest residue: the geometry-sensitive zone is mid-stack across hosts,
and the causal gate is mid-stack across hosts — two mid-stack phenomena
that need NOT be the same phenomenon. Card L3 reverts to pre-T029 state
plus the e058 replication residue (A-residualized geometry is real and
mid-stack-concentrated).

**E053 — the cache utility timeline (25.7 min, all cells):**
- **P1 (shape) REFUTED in the design's revised form:** sink confirmed
  dead (sink lesion dCE 0.0069 final — nothing); utility is NOT clean
  monotone-recency either (monotone_frac 0.5-0.75); there IS structure
  beyond recency+sink at all scales.
- **P2 (onset): scale-INVARIANT live fraction (0.247-0.286, ratio 1.16)
  and exposure-invariant onset age (a*=63 at every exposure) — the
  registered "grows with exposure" prediction REFUTED.** The dead-weight
  onset age is remarkably stable: ~63 positions across all five cells.
- **P3 (sink trajectory): mixed by scale** — the 0.84M and 10M nets
  show sink-lesion cost DECAYING over generation (10M sign-flips);
  the 2.7M-family cells are FLAT. Sink accumulation is not universal.
- **The publishable core: the first causal per-position utility curve —
  live fraction ~25% at all scales, onset age ~63 invariant, sink dead
  at generation, deviations from monotone recency at 25-50% of
  positions.** Someone pruning caches now has numbers.

## T029 — E058: the geometry zone is the causal-gate region? [DOWNGRADED TO SUGGESTIVE by interpreter audit 19:20Z — now KILLED by E064]

**AUDIT FINDINGS (applied in place):** (1) The peaks are shallow (10%
margins; B43's ladder is bimodal — L2 3.82 nearly ties L4 4.24 and
exceeds its pre-gate L3). (2) "Peaks AT the gate" was a THIRD,
unregistered hypothesis after H-role and H-depth both failed the
registered rules — garden-of-forking-paths; with gate∪pre-gate∪flank
covering ~3 of 6 sites, a smooth mid-stack ladder passes per host by
coin flip, and the two gates are adjacent. (3) The pooled partial r is
pseudo-replicated (6 distinct geometry values × 2 hosts; effective n≈6).
(4) Circularity risk made visible: raw damage and organ-load BOTH peak
at L0; the mid-stack "interference peak" exists only in the D/A ratio —
both instruments may track one shared factor (mid-stack
seed-idiosyncrasy) rather than a "gate," and T014's ±1-layer gate error
makes "peak AT gate" nearly unfalsifiable for adjacent gates. (5) L0
dissociation has a mundane reading (T013's shared machinery; embedding-
dominated, nearly seed-determined map).

**HONEST STATE: suggestive, not established.** The registered stress
test (e064): run the graft ladder on R and R43 (gates at modes 4 and 5 —
R43's far from mid-stack), no training needed. Registered prediction:
R43's R-ladder peaks L5 if the unification is real; a mid-stack peak
with a measured L5 gate kills it via shared-method bias. Free pre-step:
bootstrap CIs on B/B43 peak location from existing metrics.

Original entry follows.

## T029-original — E058 findings (2026-09-25T19:05Z)

**Verdict MIXED-as-registered but the structure is sharp:**
- **H-none REFUTED — the geometry signal replicates hard at 2.7M once
  A-residualized:** pooled partial r(D, align | A) = +0.916, r(D, W-space |
  A) = +0.963. T026's "regress on A, read the residual" prescription
  recovers a near-perfect geometry signal at depth.
- **H-depth REFUTED — interference peaks slide with the host's causal
  gate** (B peaks L3, B43 peaks L4 — each AT its measured causal mode
  from e018/e012d; the pre-gate site is the close flank). Structure
  lives in the causal-gate REGION, not at a fixed depth.
- **0.84M site rows:** L2(pre-gate) is the only geometry site (r 0.747/
  0.680); L3(gate) is pure organ-load (r_align 0.07); L0/L1 nothing.
- **THE cross-scale discovery — criticality ≠ basis-specificity:** L0 is
  the most damage-ccritical organ at 2.7M (self-ablation +4.08) yet a
  cross-seed L0 organ plugs in almost CLEANLY (R 1.03-1.13 — the e028
  inert band). Complete dissociation: what makes an organ vital to its
  host says nothing about whether a foreign copy of it will be rejected.
  And at the interference-peak sites (L3/L4), cross-seed dW is slightly
  ANTI-aligned (cos −0.013) — the rejection zone is where solutions
  actively diverge.
- Caveat: 2.7M geometry values are direction-symmetric (6 levels × 2
  hosts) — replication rests on partial-r + gate-tracking jointly.

**Card consequence:** L3 (anchoring) gains its WHERE — the init-anchored
interface is anatomically localized to the causal-gate region, and the
anchor's strength tracks the gate's position, unifying the transplant
ladder with the causal census.

## T028 — Expression-gap prior art: our lane confirmed with scope fixes (2026-09-25T18:50Z)

**The researcher's scan (scratch/expression_gap_lit.md):**
- **Orgad ICLR-25 is probe-only** (their "internally-right-but-generates-
  wrong" cell IS our gap, untested causally anywhere); their error-type
  AUCs are weak (0.59-0.68) — pre-register our discrimination metric.
- **e055's transplant is unpublished WITH A SCOPE FIX:** "zero
  interventional studies" → "zero interventional studies of *factual-
  recall* expression." Closest prior: "You Only Pass Once" (2608.14465) —
  relay steering flips silently-encoded knowledge into speech in the
  ABSTENTION domain; different method (learned direction vs own-state
  transplant), different goal (elicitation vs suppression-depth
  localization). Cite proactively.
- **Unclaimed territories we can own:** (a) the sub-argmax-persistent
  prior (correct token rank-2+ in free run while winning under teacher
  forcing — no one has measured this); (b) TF/free-run STATE comparison
  at a named token (zero prior); (c) the depth-survival/localization
  curve for suppression. Adopt Buckmann's term "elicitation failure."
- Yan & Jia EMNLP-25 found a promote-then-suppress circuit for
  enumeration repetition — the strongest "suppression circuits exist"
  motivation cite. ITI/DoLa/CD all assume-but-never-measure where
  knowledge dies.

**e055 registration sharpened by this scan:** pre-register the
discrimination metric; measure the sub-argmax prior explicitly; cite
YOPO-2608 + Yan-Jia; claim "factual-recall elicitation failure, causally
localized."

## T027 — E050: directed mutation doesn't help either — FROZEN at full strength (2026-09-25T18:20Z) [full-report enriched 18:30Z: the KEY contrast — random mutation gave −3.52% at its first selection event, directed gave −3.63%: IDENTICAL trickle despite 100% of ε on the graft-interface matrices and abundant founder spread (0.213 nats >> 3×CI). The limiter is selection's VIEW, not mutation's reach — the cleanest statement of VISIBILITY-LIMITED. LN-confound re-dead at n=7 (r=0.251, p=0.59); damage trickles on axes neither LN nor alignment explains. Wall 16.7 min, thermal-disciplined.]

**The reachability test:** same lineage protocol, mutation restricted to
the stream-facing matrices (W_in/W_out). Verdict conditions: D fell only
−3.6% (2.616 → 2.521, ratio 0.964) with R at −1.5%; alignment shift
−0.00007 (flat); all gates eligible; contrast CI excludes 0.

**Verdict: VISIBILITY-LIMITED — FROZEN holds at its strongest.** Even
when mutation is directed AT the basis interface, selection on graft
damage cannot pump alignment or meaningfully reduce damage (−3.6% vs the
−25% bar; weaker than e040's undirected −5.5%, within the organ-reliance
noise T026 identified). Combined chain (T024+T026+T027): random mutation
can't generate basis variation; directed mutation can't make selection
see it through the organ-load dominant; and the assay itself measures
organ-reliance primarily. **The stream basis is written once at init and
effectively closed to evolution at this scale — the strongest form of
the claim the three-run chain can support.** The remaining escape hatch:
e060 (A-residualized selection index) — if even that fails, the story is
complete.

## T026 — E052: the assay measured organ-reliance, not basis-fit (2026-09-25T17:45Z)

**Zero-GPU reanalysis, bitwise-exact reproduction of e040.** The
correlation table (n=10 hosts): damage tracks OWN-ORGAN LOAD A at
r=+0.807 (p=0.005); LN-distance +0.513; alignment-distance +0.543;
W-space +0.410. OLS: beta_A 0.716 vs beta_LN 0.273 (ΔR² of LN over A
alone: 0.066).

**Verdicts:**
- **WRONG-TRAIT (LN calibration): EXCLUDED** — the e031 alternative is
  dead at the registered bar.
- **The constructive reading ALSO loses:** the assay's damage readout is
  dominated by host-side organ criticality, not foreign-organ basis fit.
  e040's 5.6% trickle was selection on organ-reliance ratios — consistent
  with the host-specific reverse-graft asymmetry.
- **FROZEN survives, reinterpreted:** random mutation cannot generate
  basis variation (founders: damage varied, alignment didn't — T024), AND
  the standard graft-damage trait cannot even see basis-fit through the
  organ-load dominant. **Any future compatibility selection must regress
  D on A and select on the residual** (registered method note).
- Per-site nuance (secondary, uncorrected): a real L2-localized geometry
  signal exists (align r=0.747 at L2; nothing at L3) — diluted by the
  aggregate trait. The e050 directed-mutation arm's readout should be
  site-L2-weighted or A-residualized.

## T025 — The frontier scan: three unpublished curves in our lane (2026-09-25T17:25Z)

**The research agent's findings (scratch/frontier_research_20260925.md):
our strongest results are frontier-novel** — the four-faculty edit law and
row surgery have no counterpart (Guo ICML-2025 and the RMU-obfuscation
thread confirm first-order unlearning fails, but nobody separates
address/ability/expression/history); E040's basis-frozenness-under-
selection fills a gap MMC/lottery/universality work left open. Three
candidates where NOBODY has the curve at any scale:

- **A — CACHE UTILITY TIMELINE (e053):** per-position K/V patch-lesion
  over a long generation in the 0.84M net. Sinks proven universal (Gu,
  ICLR-2025; KVSink COLM-2025) but dead-weight onset never causally
  measured. Prediction: utility collapses onto sink + recency window.
- **B — CONTEXT-ROT ANATOMY (e054):** train KV-recall at ctx-512, sweep
  to 2048, patch positional vs content components (extends T021).
  Prediction: position-addressing degrades first.
- **C — WHERE A KNOWN ANSWER DIES (e055):** transplant teacher-forced
  residual states at the divergence token into free-running (extends the
  expression-gap). If behavior flips, knowledge was present-but-suppressed
  and transplant depth LOCALIZES the suppression. The expression gap has
  zero interventional studies anywhere (Orgad ICLR-2025 is probe-level).

All three reuse existing tooling; A is the cheapest and least explored.

## T024 — Lineage verdict: init-anchoring is FROZEN under selection (E040, 2026-09-25T16:55Z)

**AUDIT FLAGS (17:15Z, applied before e050 reads out):** (1) the headline
"-7.3%, D 2.435" quoted the selected-two mean; the registered all-member
contrast is −5.5% (2.625→2.479). (2) The trickle is one member: excluding
g2a, gen-2 vs gen-0 is −2.3%; the CI bootstraps batches, not lineages
(n=1 lineage, non-independent members). (3) G0 anchor FAILED (1.5240 vs
1.5581±0.03 — batch-size effect). (4) G7 gated on D (spread 0.115) but the
SELECTED index R had gen-0 spread 0.058, under the 0.06 bar — σ never
escalated though it should have. (5) D~0.8-correlates with own-organ load;
reverse-graft host-specific; **e031's LN-statistics alternative is the
LEADING unexcluded reading** → e052 dispatched (zero-GPU LN-distance
regression on the 11 checkpoints; verdict rules registered in its brief).
T024 now reads: "P2 proves global-noise-at-this-σ cannot generate basis
variation (founders: damage varied, alignment didn't); FROZEN's
selection-side claim awaits e052 + e050."

**FULL-REPORT AMENDMENT (17:05Z): "weak-but-real," not literally frozen.**
The agent's final numbers: D fell monotonically 2.625→2.533→2.479 (−5.6%,
CI excluding 0 — heritable, directionally responsive) but missed the −25%
bar 4.5×; R only −2.9% (organ-devaluation throttled by design); alignment
shifted +0.0015 against a 0.42 ceiling — **compatibility moved with ZERO
donor-alignment drift** (a dynamic dissociation complementing e029/e041's
static ladder: selection exploited interface/robustness, NOT
toward-donor motion). Strict reading: P2 fired on the "<10%" disjunct
while "CI includes 0" is false. **T024 restated: the basis is
WEAKLY-EVOLVABLE (−5.6%/2 gens) but selection walks a nearly-flat
landscape that does not pass through donor alignment — effectively frozen
at any practical selection strength.** Reverse-graft confirms asymmetry
(REF←winner: +2.80/+2.54). e050 (directed mutation) now discriminates
REACHABILITY vs VISIBILITY precisely.

**The evolution thread's first result: P2-FROZEN.** Eleven step-matched
0.84M trainings, two selection events, thermal-disciplined (envelope
compliant; status complete):
- Graft damage moved only −7.3% from gen-0 to gen-2 (D 2.625 → 2.435;
  ratio 0.927 — the P1 bar was −25%); R moved −2.2% (0.978). Both
  contrasts' CIs exclude zero (real drift) but are an order of magnitude
  too small: **selection produced a trickle, not a response.**
- Alignment to the REF donor stayed at the floor (0.001 → 0.006 — a
  +0.005 shift against a 0.53 ceiling); the selection pressure could not
  pull the stream basis toward the donor's.
- **C3 gains its selection answer: the init-anchored stream basis is
  effectively invisible to lineage selection at this population size,
  mutation rate, and generation count.** The degeneracy-route (P3) did
  not fire either — compatibility and alignment did NOT dissociate; both
  stayed put.
- Scope: n=1 donor, n=3 founders, 2 generations, σ_mut=0.005 (G7 never
  escalated — gen-0 spread exceeded 3× CI). A larger population or
  stronger mutation might respond; this design measured the standard
  regime and found it frozen.

**The edit-arc's bookend:** the lab now knows the stream basis is
init-anchored (T003), partially (T-e041: ceiling 0.53), net-specific in
its interfaces (T016), and NOT evolvable under standard selection (this).
Gradient descent writes it once; neither surgery, training regime, nor
selection rewrites it.

**T024 PRE-REGISTERED FOLLOW-UP (e050, registered before the audit lands):**
the directed-mutation control. P2-FROZEN conflates "selection can't see
the basis" with "random mutation can't reach it." e050: same lineage
protocol BUT mutate ONLY the stream-facing matrices (W_in/W_out rows,
σ matched to e040) — if selection now responds (damage −25%+), the basis
was selectable but unreachable by global noise (verdict flips to
REACHABILITY-LIMITED); if still frozen, the basis is genuinely invisible
to selection (FROZEN holds at its strongest). Also registered: the
host-side-LN confound check — correlate per-member damage with the
e031-style LN-statistics distance to REF (not just ΔW alignment).

## T023 — Write-equalizer: the energy schedule is decorative (E033, 2026-09-25T16:30Z; 0.84M, single net)

**Verdicts: P1 parity TRUE (1.5147 < baseline 1.537 — equalized MLP writes
learn BETTER); P2 energy-migrates FALSE; P3 calibrator survives TRUE (KL
1.176).**
- Forcing every MLP write to one uniform norm changed almost nothing that
  matters: attention damage unchanged ([2.48,2.06,1.37,0.30] vs [2.84,
  2.16,1.29,0.26]); MLP damage ROSE mid-stack and flattened ([3.06,2.44,
  1.76,0.97] vs [2.95,1.55,1.60,1.22]); front-loading intact; the
  calibrator untouched.
- **L2 refinement (0.84M-scoped): the late-MLP energy CARRIER is real
  (e047: 5/5) but the growing SCHEDULE is an allocable habit, not a
  necessity.** The energy does not "migrate" anywhere when equalized — the
  organs simply become uniformly important. Homeostasis bio-analogue
  resolved: the network has no vital energy budget to defend at this
  scale; it defends the coarse allocation (front-loading, calibrator),
  not the write norms.
- Caveats: single net, 4 layers, ≤1M envelope. The 2.7M equalizer never
  completed (pre-envelope buggy run) — schedule-necessity at larger depth
  remains open.

## T022 — e040 pre-run gate: is init-anchoring evolvable? (registered per scratch/e040_design.md, 2026-09-25T16:15Z)

The lineage experiment's design is measured and gated (memo: cross-seed
graft damage +0.82/+1.99 nats at L2/L3 MLP — 3-4× own-ablation; alignment
rungs 0.42/0.36 → 0.09/0.11 → 0.00 at exactly the violent sites;
selectable resolution ~0.06 nats with 2-3% noise). Lineage: 0.84M
genotypes from the seed-42 family (wildtype + 3 mutants at σ=25% init
std), fixed REF donor organs, 2 selection events × 3 children, STEP-MATCHED
4000 steps each, R = graft/own-ablation with parity + organ-band + G7
escalation guards. REGISTERED:
- **P1 (evolvable):** damage and R both −25% by gen-2, CI excludes 0, all
  gates held → init-anchoring is under selectable control; e040b (second
  REF) fires, never a re-roll.
- **P2 (frozen):** <10% response or CI spans 0 with alignment flat →
  selection cannot see the stream basis; C3 gains "frozen under selection."
- **P3 (degeneration-route):** ΔW-alignment to REF rises ≥+0.05 without
  damage falling → compatibility ≠ alignment (they dissociate).
Scope caveats: n=1 donor, n=3 founders, single site pair. Budget 23.8 min
serial with cooldowns — the envelope's first big run, reviewed pre-launch
per standing instruction.

## T021 — The retrieval threshold (E049, 2026-09-25T04:30Z) [R8 scoping: corpora 620KB ~75 epochs; the p0 net is off-parity (val 2.82) so the −2.24 interference magnitude is partly net-quality artifact — the FLIP's SIGN is real, magnitude confounded; '≤5%' rests on one n=20-events cell (shared-probe view: 5-20%); late-attention-slot preference firm for the 2.7M series only — suggestive for 10M (1.42x contrast, 491 steps).]

**Four refrain-density nets + a 10M arm, registered P1-P4:**
- **P1 REFUTED-LOW (threshold ≤5%):** retrieval is already net-positive at
  the lowest nonzero density (+0.22 far-value, 0.62 acc at p5; the
  stricter shared-probe view puts it 5-20%). Even MORE sensitive than
  registered.
- **P2 REFUTED — GRADED, not sharp:** +0.22 → +0.81 → +0.89 (adjacent
  ratios 3.68× then 1.10×). No order-of-magnitude jump.
- **P3 HOLDS — retrieval is compartmentalized:** zero positive leak into
  ordinary text at any density.
- **P4 WEAK-SUPPORT:** the 10M net is 1.87× more retrieval-sensitive at
  p5 (wall-matched caveat) — consistent with T020's scale erosion.
- **THE FLIP (the arc's closing finding):** at p=0, far context HURTS
  completions by −2.24 nats (T007's interference, corpus-overfit
  amplified); ~200 refrain events (~20 exposures/pair) cancel a full nat
  of interference and turn on net-positive retrieval. **Far-value is a
  tug-of-war whose balance flips near the bottom of the density grid** —
  T007's bimodal hurt-population was the negative side of this same coin.
- **Behavior grades; anatomy is lumpy:** far-value rises smoothly but the
  retrieval HEAD appears discretely — absent at p5, real at p20 (L5H3),
  stronger at p60 (L5H1, 8.4× refrain-mass over control) — ALWAYS in the
  late-attention slot (L5 at 6 layers; L7 in the 8-layer 10M). The
  circuit has a preferred DEPTH, not a preferred density.
- Cross-talk at 60% (acc non-monotone 0.62/0.87/0.77); value concentrated
  at the first content char (10M uniquely flat — deeper-copying
  signature).

**L4's complete arc:** no retrieval on natural data (morning) → margin
erodes with scale (capstone) → the threshold is graded, low (≤5%),
compartmentalized, and flips from net interference (T007's other half).
The "no retrieval" law was one point on a curve whose shape is now
mapped.

## T020 — The scale capstone: the laws hold, sharpened (E005s, 2026-09-25T02:40Z) [R8: steps-confound caveat — 4000/2226/1086 steps anti-correlate with scale; LARGE stopped mid-cosine; the 11/72/116x multipliers and ~35x erosion could be steps trends — DIRECTIONS survive, magnitudes confounded. P1 at 4L FAILED the strict registered criterion (mode at final block) — held at 6L/8L only.]

**0.84M / 2.7M / 10M, frozen readouts, P1-P4 registered before training:**
- **L1 (causal gate): structure universal, relative depth slides with
  ARCHITECTURE** — 8L: clean mid-stack gate (mode 2 of 0-7, monotone
  flips, 74% suffix-monotone). 4L: gate exists but the mode is the FINAL
  block (46% commit at L−1 — a 4-layer stack has no "mid"). Relative
  commitment depth: 1.0 (4L) → 0.6 (6L) → 0.29 (8L). T014's
  non-invariance extends from seed/regime to depth budget. Distributed
  decisions COLLAPSE with depth capacity (26% → 15% → 7% no-single-flip).
- **L2 (front-loading): HOLDS and INTENSIFIES** — attn-L0/last ratio
  11.2× → 71.7× → 116.4×; at 10M the three deepest attention blocks cost
  ≤0.04 nats each (late attention becomes literally free) while the
  MLP-0 keystone sharpens (3.40 with the rest of the MLP stack ≤0.14).
- **L4 (16-token sufficiency): holds but ERODES monotonically with
  scale** — far-value −0.001 (2.7M) → +0.021 (0.84M) → +0.035 (10M),
  p99 climbing 1.6→2.3 — larger nets extract slightly more from far
  context; the margin shrank ~35× and is e049's leading edge.
- **L6 (address surgery): HOLDS at both scales** — S_name 1293 (0.84M) /
  332 (10M), class-exact, corpus +0.0005; the removal doctrine is
  scale-invariant in shape, with S_name declining as pure-control
  collateral grows with scale.
- **Harness bug found+fixed:** checkpoint-resume map_location moved the
  CPU RNG state to CUDA (first-ever mid-training resume crash); now
  map_location="cpu". LARGE resumed bit-faithfully.

**Card v3 stamps added: L1 architecture-scoped; L2 scale-intensified; L4
scale-eroding (open edge); L6 scale-invariant.** The day-one laws are
properties of the family, not one draw — each with its scale signature
now measured.

## T019 — Expression is teacher-forcing-bound (E048, 2026-09-25T00:55Z)

**P1 CONFIRMED, absolutely:** expression count = 0 across ALL arms —
unseeded, seeded (induction route), greedy diagnostic, T ∈ {0.7, 1.0,
1.3}, and dose × {400, 800, 1600} steps — while battery accuracy holds
0.92-0.96 throughout. P2 (prior threshold) refuted: neither seeding nor
temperature unlocks a single occurrence. R8 RELABEL: P3's dose legs were BIT-IDENTICAL across s400-1600 (dose_probes AND generations) — same-checkpoint readout, INSTRUMENT SUSPECT; dose-response is OPEN, not refuted. P1 (zero expression) survives independently (its arms vary-verified).

**C7 FINAL — the edit law, complete:** there are FOUR separable faculties:
ADDRESS (rows — surgically removable, attractor-surviving), ABILITY
(distributed body usage — train-only), EXPRESSION (free-generation
surfacing — requires free-generation-shaped exposure; teacher-forced
install never confers it at any dose), and HISTORY (the re-learned memory
differs from the original in route and surgical-resistance). An exposure
protocol that never shows the name in FREE contexts installs answer-
knowledge that is constitutionally silent.

**Lab-doctrine note:** this is also a warning about the battery method
itself — continuation-battery accuracy is NOT evidence of usable
knowledge; free-generation probes are the honesty check for any install
claim, at any dose.

**FULL-REPORT REFINEMENT (01:05Z):** the expression failure DECOMPOSES —
(a) **geometry binding:** the installed address fires only in the trained
continuation window (p(Z) 0.556 in battery geometry → 0.0898 at the same
terminal text in a 120-char free prompt → 1.7e-6 in uniform contexts;
deleting the 10 oldest context chars collapses it 6×); (b) **sub-argmax
prior:** even where prior exists it never wins argmax. The single
expression event of the day (1 in ~30-35k chars across 13-14 cells (R7: denominator bookkeeping imprecise; direction unaffected)) came
from the DIRECT-TRAINED control at its home slots (p(Z) 0.27 there,
argmax 25%) — natural learning puts the name in the free-generation
prior; install does not, at any dose. Uniform-floor battery shows the
install at 0.286 vs 0.974 host-trained (40× floor, 3.4× below claim —
battery-overfit partial; the direct net shows the same floor shape, so
part is CORRECT slot-gating). Open edge: content-vs-position binding.

## T018 — Scar tissue: erasure burns the address, not the attractor (E044, 2026-09-25T00:15Z) [R7 FLAG: n=1, single net — the history clause's numbers await replication]

**The real run (smoke:false, 350s, 5 arms, all gates; shakedown root cause:
E044_SMOKE=1 env leftover — postmortem in the script; new COS_MIN_NORM
validity guard):**
- **The scar is ANATOMICAL, not kinetic.** Re-learning JULIET is 2.08×
  SLOWER than fresh install (P2's speed leg falsified in reverse) — but
  the zeroed wte_J row re-grows along its ORIGINAL direction (cos +0.760,
  monotone, 58% norm regrowth at s400) while a fresh name lands orthogonal
  (0.278). **The body remembers the direction; the address re-grows into
  its old groove.** Step decomposition pins the entire 2× tax on address
  regrowth (a=b2=25 exactly; the L3H5 patch costs zero extra steps).
- **The new route is genuinely new — and the old head becomes an
  ANTI-carrier:** post-relearning atlas correlates only 0.21 with the
  original; L3H5 flips carrier→anti-carrier (+0.88 → −2.03: zeroing it now
  IMPROVES the memory by 2 nats). The natural battery still rides L0H3/L0
  (0.60) — the shared machine persists (e043/e047 law).
- **Re-erasibility degraded ~3×:** re-applying D2+patch to the re-trained
  net leaves 44.5% accuracy (vs 0.13% originally) at +0.021 CE — the
  re-learned memory is partially SURGICAL-PROOF (less address-dependent,
  more distributed).
- P3 failed at bar (incumbents +0.135 vs 0.10; corpus only +0.013; JOHN
  actually improved −3.16 — re-learning repaired D2's J-class collateral).

**C7 refined — the edit law, final:** removal burns the address cheaply
but leaves the attractor; installation re-grows the address along the old
groove (if one existed) while building a NEW route that is harder to
surgically remove than the original — erasure is not just incomplete at
the attractor level; it is partially irreversible at the coordinate level
(the second memory won't fit the first memory's surgical key). Address,
ability, expression — and now history.

## T017 — CARD v3: the replication sweep's three stamps (E047, 2026-09-24T23:55Z)

**The gate inputs (5 nets each, reference reproductions near-bit-exact,
renorm-liveness asserted):**
1. **L5-CALIBRATOR: SURVIVES 5/5 → enters C-card at H** (first positive to
   earn it under the min-nets rule). KL(L5‖L4) 0.91-1.08 nats at ablation
   cost ≤ +0.046 in every anatomy — big reshaping, near-zero removal cost,
   universal. THE most robust mechanism finding of the day.
2. **MLP-5 ENERGY CARRIER: SURVIVES 5/5 → H.** Zero/rotate ratio 0.25-0.34
   (zero costs 3-4× direction-scramble at matched energy), graceful
   α-scaling in all 10 cells. Late MLP magnitude is load-bearing
   everywhere; its direction barely.
3. **SHARED-L0 NAME MACHINE: DIES AS STATED (2/5 — only the seed-42
   lineage; renorm nets pushed it to L0-attn with L0-MLP dissolved).**
   Post-hoc salvage (flagged unregistered): the L0 BLOCK (either sublayer)
   is the top-1 name-completion block in **20/20 net×name cells** —
   "L0-block = universal early-local name machinery; attn-vs-MLP
   allocation is a lineage lottery" (consistent with e014b's keystone
   dissolution). Enters card v3 in distributional form.

**CARD v3 PREAMBLE (adopted):** laws-not-mechanisms; min-nets-per-claim
(H requires ≥3); mechanism claims stated as distributions with their
ensemble spread. The card's claims after v3: C1 qualitative causal gate
(scoped) + L5-calibrator (H); C2 plastic anatomy + energy-carrier (H);
C3 partial anchoring ladder (causal); C4 retrieval task-elicited, L4-H1
dominant-in-one-net (sample); C5 ascent-fails (universal negative);
C6 address-surgery general damager / completion net-specific;
C7 edit asymmetry: removal surgical / install plastic / expression third.

## T016 — C6 DEMOTED: two-factor erasure does not replicate (E046, 2026-09-24T23:28Z)

**The replication verdict (2 hosts, 4 cells, full honesty battery):**
- **Neither the in-run-discovered head (B43: L4H4; BDO: L3H1) nor B's
  L3H5 recipe reaches Bar-2 on any other net** — JULIET stays at 13-17%
  accuracy after row-zero + top-head lesion (B43 own-head 15.7%,
  uniform-floor 15.1%; BDO 13.9/16.4%). B's specific recipe transfers as
  predicted-NO (its site is inert in B43: −0.14).
- **What DOES replicate:** row surgery alone damages to the same 13-17%
  band at ~+0.001 corpus cost, class-exact — on every net. The ADDRESS
  half of the doctrine is general; the "one body head" COMPLETION was a
  B-specific coincidence.
- **C6 demoted to:** "entity-row surgery is a general, cheap, class-exact
  DAMAGER (~500× selectivity); COMPLETE erasure was achieved once (B,
  rows+L3H5) but the second factor did not generalize — the residual after
  address-burning is net-specific in structure."
- **Pattern note for the card:** the day's positive mechanism claims are
  systematically dying under replication/audit (rare-token heads, stage
  invariance, now two-factor erasure) while its NEGATIVE claims (ascent
  fails, no far retrieval on natural data, install-not-surgical) hold
  everywhere. The asymmetry-laws survive; the mechanism-stories don't.
  The Review-6 panel must weigh a REPLICATION SWEEP of remaining positive
  claims vs new arcs.

## T015 — Edit asymmetry law (E043; audited 21:30Z, AMENDED by full report 22:05Z)

**AMENDMENT (full report):** the earlier "exposure installs transiently then
decays" line was a partial-trajectory read. The truth is better:
- **P2 FALSIFIED in the good direction: anchored exposure installs CHEAPLY
  and SELECTIVELY.** 7 guarded cells reach Bar-I2; best Dmix@s400: 0.09
  NLL / 0.974 acc at ΔCE +0.0505, S_install 144-289, incumbents ≤0.06.
  The earlier CE-climb belonged to a different anchor trajectory.
- **Two new caveats define the law's real shape:** (1) **the expression
  gap** — 0.974 battery accuracy yet ZERO ZEPHYRA in 2,800 generated
  chars: teacher-forced install never surfaces in free generation;
  "installed" ≠ "expressed." (2) **protocol fragility** — the onset wall
  was anchor-manufactured (0.00 through s1000 under fully-paired anchors
  at CE +3.40 vs 0.82 by s400 under the mix at CE +0.05): install is
  protocol-fragile where erase was protocol-robust.
- **P3(b) beautiful:** the installed name rides the INCUMBENTS' shared
  L0-MLP (+9.18 nats, top head only 10% of the block) — no private
  circuit ever grows; new knowledge reuses shared machinery (consistent
  with e042). Deep installs mildly reshuffle incumbent rankings (ρ→0.66).

**The law (final form, C7):** REMOVAL is surgical — 384 parameters,
robust, class-exact. INSTALLATION requires plasticity — never surgical
(best graft closes 5.9% at 2.4× the guard; B43 diff-init rows install
nothing while damaging +1.92) — but plasticity is cheap and highly
selective when anchored; its limits are expression (silent under teacher
forcing) and protocol-fragility. Address is concentrated; ability is
distributed and trained; expression is a third thing entirely.

**Verdict (registered, all three arms): ASYMMETRIC-CHEAP-REMOVE.**
- **No surgical arm reaches Bar-I1 at the guard** (best cell A/both/copy:
  NLL 6.43, acc 0.055 vs bar [≤4.17, ≥0.5, ≤+0.10]) — copying donor rows
  (even same-init BDO rows that transplant cleanly) does NOT install
  knowledge. Bar-I2 unreachable everywhere ("barI2_reachers": []).
- **The exposure route works transiently then decays:** NLL 0.197 at step
  25 (near-perfect install!) with corpus CE climbing monotonically after
  step 100 (3.24 → 5.19 by step 1000) — knowledge installed by exposure
  erodes the host unless supported; the INTERPRETER's guard-artifact
  reading was half right (the asymmetry is real AND the guard binds only
  installs — removal's "dose" is free because J is 0.016% of tokens).
- **The shared machinery survives untouched:** L0H3 remains top-1 for
  JULIET in the best guarded cells (atlas Spearman 0.9997) — installing
  does not disturb incumbents' circuits; P3 confirmed.
- **The interpretation (the day's closing structural finding):** rows
  carry the ADDRESS; the body carries the ABILITY TO USE it. You can
  remove a memory by burning its address (cheap, exact); you cannot write
  one without teaching the circuit to read it (exposure = training, with
  its forgetting cost). Editing the organism is fundamentally asymmetric
  because reading-skill is distributed and address is concentrated.

**Card consequence:** C7 drafted — the edit asymmetry law. e044 (scar
tissue) now tests the law's prediction for RE-learning: after erasure,
re-install should ALSO need exposure (no dormant-row shortcut) unless scar
tissue remains.

## T014 — C1 resolved: cross-net causal invariance is NEGATIVE; what survives is qualitative (E012d, 2026-09-24T21:20Z)

**The 4-net causal census (protocol identical to e018; B reproduced
exactly):**
- Cross-net causal-histogram correlations: seed axis 0.735/0.799, regime
  axis 0.753/**0.500** (B43↔R43 collapse) — ALL below the registered 0.8
  bar; instrument reliability ceiling 0.83-0.85, so these are real
  failures. The causal mode SLIDES with seed+regime (B 3 → B43 4 → R 4 →
  R43 5); renorm nets shift mass deeper (d5: 250→423). The lens census's
  0.82-0.85 "invariance" was exactly the by-construction artifact R4
  feared — the causal truth moves where the lens cannot see.
- **T012's demotion is TOTAL:** within the lens=6 subset (52.7% of
  positions), causal depth is statistically indistinguishable from
  elsewhere (mean 2.94 vs 2.96; proportional histograms). The lens's
  dominant bin selects causally random positions.
- **C1's final honest form:** every net examined has a causal commitment
  point mid-stack with suffix-monotone flip curves (qualitative
  universal); the DEPTH of that commitment is seed- and regime-dependent
  (quantitative non-invariance). "Stages are the organism" in its strong
  cross-net form is DEAD; what survives: "a mid-stack causal gate exists
  in every anatomy, at a depth the anatomy's history chooses."
- Card v2 consequence: C1 → scoped qualitative (single architecture
  family; cross-net depth non-invariance affirmative). e005s scaling
  ladder's C1 readout re-scoped accordingly (test the QUALITATIVE gate's
  existence, not depth invariance).

## T013 — Two-factor surgical erasure; name circuits are shared (E042, 2026-09-24T15:35Z)

**The arc's first phase closes with a complete answer:**
- **Two-factor surgery ERASES:** D2 row-zero + head L3H5 zeroed at
  JULIET-prefix positions → accuracy 0.136 → 0.0013, NLL ≥ ln65 (Bar-2
  met), total corpus cost +0.00083 nats, collateral ROMEO +0.014 / LUCIO
  −0.004 → **S_name = 1,937** (573 for rows alone). The patch adds ~1e-5
  over D2 alone; content-triggering (vs position-rule +0.0023 / block-rule
  +0.074) is what keeps it cheap.
- **Name completion is SHARED early-local machinery:** head L0H3 is #1 for
  all four names (+1.5–2.2 nats each; 34% of JULIET's positive head mass,
  top-3 = 60%); L0-MLP is the #1 block for all (+7.0–7.7); JULIET~LUCIO
  head atlases near-proportional (Pearson 0.965). e023's collateral
  idiosyncrasy therefore lives in ROW/INPUT space, not circuit overlap.
- **Dissociation:** the healthy circuit (L0) is NOT what carries the
  post-D2 residual — the 13.6% rides mid-network machinery (L3H5 +0.88,
  L1-attn +2.62 under D2), all concentrated at position 3 ("?UL→I").
- **C6 finalized:** selective forgetting = rows (the index) + one body
  head (the residual transition); erasure-complete, collateral ~1e-3 nats,
  class-exact boundaries. The subtractive half of "edit the organism" is
  done; e043 (INSTALL) is now cleanly defined: rows + which body?

## T012 — Instrument verdict: the depth lens is UNCORRELATED with causal depth; C1 re-anchored (E018, 2026-09-24T15:20Z)

**Verdict (b) fired, decisively.** Activation-patching causal depth
(1536 positions, sanity-gated: self-patches exact; post-L5 patch flips
100%):
- **Spearman(causal, lens) = −0.009 — the instruments are per-position
  UNCORRELATED**, not merely biased. The lens's argmax-stability ordering
  carries no causal-decision information.
- Causal mode = depth 3 (entering L3); mean 2.95 vs lens 4.86. The lens's
  dominant "decided at L5" mass (53%) has no causal counterpart: **late L5
  flips are RECALIBRATION, not decision** — convergent with every other
  instrument's account of L5-the-calibrator.
- Real point-of-no-return structure exists: monotone flip curve
  [0.14→0.78], 77% of flippers suffix-monotone; 15.4% of positions flip
  under NO single patch (distributed decisions); shallow patches mostly
  yield THIRD tokens (73% at d0 → 21% at d5) — early streams disrupt,
  mid-stack streams determine.

**C1 RE-ANCHORED (T010 card amended):** "staged function" survives — with
a genuine causal commitment point mid-stack (L3/L4) — but ALL lens-based
depth evidence (the 4-net census invariance 0.82-0.85, the L5-finalization
modes, the class ordering) is DEMOTED to "argmax-stability profile" and
inherits the by-construction replication risk Review 4 named. NEW DEBT
(e012d): run the CAUSAL census on B43/R/R43 — is CAUSAL depth the
cross-net invariant? Until then C1's evidence chain rests on e018's single
net. e021's L4 retrieval mode remains safe (independently established by
the L4-H1 causal lesion).

## T011 — Surgical forgetting WORKS at entity granularity (2026-09-24T15:02Z; CONFIRMED by full report 15:12Z with refinements)

**FULL-REPORT REFINEMENTS (15:12Z):**
1. **Collateral correction — the scalpel is even cleaner than first
   recorded:** under D2, ALL 12 J-class words take ΔNLL ≥ 1.46 while EVERY
   non-J name moves ≤ 0.037. The "idiosyncratic collateral" (LUCIO 8.5)
   belongs to the D1 BOMB (LUCIO shares L,U,I with its six-letter set),
   not the scalpel. Collateral is perfectly letter-class-graded under the
   scalpel — overlap-o was simply the wrong predictor.
2. **Untied split (P2b passes):** wte-zero (READ row) 5.79 NLL/0.19 acc ≫
   lm-zero (WRITE row) 2.04/0.83 — reading carries the completion; but
   lm-zero keeps battery accuracy while erasing 'J' from GENERATION
   entirely (0 J-chars) — the write row controls expression.
3. **Third registered outcome fired:** S_letter ≈ 1 from BOTH instruments
   (surgery 0.99, ascent 1.13) — entity identity cannot be split from its
   letter class at char level; identity lives in body transitions the rows
   index. (e042's body atlas is exactly the next cut.)
4. **Ascent converges on the same coordinate** (logit-J 10.0 → −1.5) —
   both families target the same substrate; ascent just burns the corpus
   around it (+1.16 val vs surgery's +0.0008 scaled). C5 stays closed;
   the ΔCE≤0.10 guard correctly blocked the false r=1.73 revival.
5. G4 caveat logged: PROSPERO's unmemorized anchor is
   context-construction-sensitive (13.2 vs 4.46≈floor) — re-measure
   anchors with the uniform-floor method going forward.

**E023 verdicts (registered):**
- **The J-row scalpel is the program's first selective instrument: S_name =
  573 (bar 5) at corpus cost +0.0008 nats.** Zeroing one rare letter's
  embedding+lm_head rows (384 params) drops JULIET accuracy 88%→13.6% with
  essentially zero collateral — **4,907× less collateral than entity-ascent
  at matched damage.** P1's formal confirmation missed only on the Bar-2
  erasure bar (acc 13.6% > 10%): surgery DAMAGES the name near-completely
  but does not fully erase it.
- **D1 (all-six-letters) confirmed as a bomb** (+0.38 corpus CE — as
  registered). Precision lives in the rare-letter core.
- **P2 refuted (ρ=0.61 < 0.8):** collateral damage to other names does NOT
  cleanly track letter overlap — name collateral structure is idiosyncratic
  (LUCIO Δ8.5, ROMEO Δ2.5), not overlap-graded.
- **P3 confirmed: entity-granular ascent stays dead.** No revive trigger
  fired; at the +2.0-nat name bar, val CE is already +1.16 (catastrophic)
  with S_name 1.06 — the ascent family is closed at every granularity.
- Untied split preview: lm_head-row zero → NLL 2.04, acc 0.83 (the write
  side carries real but partial damage; full split in metrics).

**Card consequence — C6 drafted:** entity knowledge at this scale lives
substantially in I/O row coordinates (rare-letter private rows), giving
surgical selectivity ~500× beyond any first-order instrument; the
damage-vs-erase boundary (13.6% residual) and the collateral idiosyncrasy
(P2's failure) are the open edges. The residual is the e042 target: body
circuits carrying the remainder.

**Arc next:** e018 (instrument validation, Review-4 priority) + e042
(name-circuit atlas: what carries the 13.6% residual and the idiosyncratic
collateral) in parallel.

## T010 — MECHANISM CARD v1: The functional anatomy of a 2.7M char-transformer (2026-09-24T13:47Z)

The lab's first formal findings artifact (README: "a mechanism card another
researcher could attack"). Every claim carries its evidence chain, known
debt, and falsifier. Scope: ONE architecture (6L/6H/192 pre-LN char-GPT,
2.7M params), Shakespeare + synthetic variants, 6 trained nets (B, R, B43,
R43, e021 task+control), ~5 GPU-hours, single lab.

**C1 (H, re-anchored T012) — Function is staged; decisions causally close
mid-stack (L3/L4).** Lens-based depth evidence demoted to argmax-stability
profile; causal depth is the instrument; cross-net causal invariance
pending (e012d). Pipeline: token-formation → local completion → late distribution-calibration; on
retrieval tasks a retrieval stage appears at L4. Evidence: depth-census
invariance across 2 regimes × 2 seeds (4-net table, cross-seed 0.849 ≥
cross-regime 0.828, e012/e012b/e012c); decision-depth class ordering
(letters>structural) in all 6 nets; task-dependent L4 mode (e021).
Debt: single architecture/corpus family; v008 multi-seed lesion maps.
Falsifier: stage structure failing at other scales or on far-retrieval
natural data (e005s ladder, gated on this card).

**C2 (H) — Anatomy is plastic; damage tracks write allocation.** Lesion
maps reorganize under constraints (keystone dissolved +4.08→+0.10; MLP
profile inverted) with parity-or-better loss; replicated on R43; write/c
rank-order = damage rank-order (ρ=1.0); damage tracks perturbation ENERGY
not content (e011c ladder, CIs clean); MLP-5 is an energy carrier (e019
causal: graceful α, zero=4×rotate). Debt: e014c write-clamp parked.
Falsifier: write-clamp training leaving lesion maps unchanged.

**C3 (H) — The seed-anchored object is the residual-stream basis.** Organ
graft compatibility follows init lineage, not regime (e028/e029 2×2); ΔW
motion from different inits is near-orthogonal (cos≈0.000 vs same-init
+0.15); stream-FACING matrices (W_in reads, W_out writes) are the violent
grafts, W_in dominant (e031: L5 W_in alone +3.28 > whole organ +1.46 —
donor W_out partially rescues); attention matrices are mild (c_attn
mildest). Debt PAID (e041): alignment ladder complete — 1.0 → 0.534 (same-init
diff-order CEILING) → 0.152 (same-init diff-regime) → 0.000 (diff-init).
PARTIAL ANCHORING: regime change exceeds batch-order noise 3.6×, yet
same-init stays far above the floor; early organs order-robust, depth
erodes. Former debt line (superseded): ΔW ceiling null unrun —
e031 LN-statistics alternative unexcluded; "attention portable" is
weak-anchoring, NOT shared subspaces (v009).
Falsifier: e030 Procrustes-style basis-realignment rescuing cross-seed
grafts would confirm basis-geometry (upgrade); spectrum-matched W_out
control rescuing e031's pair effect would demote the coherence reading.

**C4 (H, scoped) — No far-context retrieval on natural char data; retrieval
is task-elicited, not architectural.** Shakespeare: L5 calibration local
(KL −6.9% under truncation), far attention idle (200-prompt census),
far-value bimodal but structure-insensitive (gains shuffle-robust, hurts
shuffle-amplified; e013a/c/d). Task: noiseless retrieval (CE 0.007,
far-value ln 26, head L4-H1 95.1% ID-mass, control clean; e021).
RESOLVED (e035/e038): L4-H1 is causally the LARGEST retrieval channel
(~51% of distance-to-chance) inside a REDUNDANT second channel (copy
survives at 58%=15× chance with perfect locality; joint-lesion test
pending before "fan" language returns) — not a dedicated organ.
One net holds TWO stage profiles (JS(filler,Shakespeare)=0.010 vs
JS(filler,COPY)=0.272), selected per-position. Lesion maps are blind to
task circuits (COPY ~5% of tokens). Init-anchoring is task-independent
(slightly stronger than B↔R).

**C5 (CLOSED, negative) — First-order ascent cannot content-selectively
forget.** At the 0.66-nat bar: naive r=1.23, projection r=1.43 (margin
1.17×), and r vs train-B = 1.08 — memorization-symmetric damage; apparent
selectivity was a measurement-axis artifact. The r≈5-6 head-start did not
reproduce (chaotic event). Next family: weight surgery (e023) / second-
order. MLP-5 energy-carrier confirmed causally along the way (e019).

**Cross-cutting instruments validated:** decision depth, locality funnel,
KL(L5‖L4), write/stream ratios, transplant R-bands, ΔW subspace atlas —
with known lens caveats (mid-stack readouts anti-informative; argmax-stability
robust).

**Card v2 debt summary (updated 15:35Z):** PAID: ΔW ceiling null (e041:
partial anchoring); e035/e038 (retrieval resolved); claim-5 family (closed
negative e003c; superseded by C6 surgical doctrine). OPEN: e012d (causal
census × 4 nets — C1's remaining evidence debt); v008 multi-seed maps;
e043 (install symmetry — memo in progress); e005s scaling ladder (gated on
card v2 + e012d). INSTRUMENT NOTE (T012): lens-based depth demoted;
causal depth is the depth instrument.


---

## T003 — What E011b says about authority, redundancy, and geometry (2026-09-24T10:4xZ)

**Observed (E011b, eval-only):**
1. **L0 head subsets:** singles mean +0.05 (max +0.165, one slightly
   negative); damage by subset size 1→6: 0.05, 0.36, 1.01, 1.77, 2.27, 2.40.
   Sum of singles 0.317 vs all-6 2.403 = **7.6× superadditivity**. Internal
   validation: all-6 head-zero (2.403) ≈ whole-block zero (2.401) — the hook
   implementation is exact (closes critique #8).
2. **Orthogonal-innovation:** same-norm random writes damage MORE than zeroing
   everywhere: attn [3.66 vs 2.40, 2.41 vs 1.74, 1.78 vs 1.06, 0.78 vs 0.38,
   0.38 vs 0.19, 0.10 vs 0.03]; mlp [4.44 vs 4.08, **0.82 vs 0.15**, 0.72 vs
   0.27, 0.84 vs 0.47, 0.89 vs 0.60, 1.00 vs 0.59].
3. **Stream norms at block inputs:** [0.67, 5.66, 6.14, 5.12, 5.20, 5.62] —
   an 8.4× jump at block 0, then a plateau.

**Reading A — cooperative ensemble, not backups (finding 1):** L0's heads are
a graceful-degradation fan: each is nearly dispensable alone, but damage grows
steeply with subset size (keep 1 head → 94% of full-ablation damage). They
are small parallel contributions, not redundant copies of each other.
Single-lesion anatomy maps CANNOT see this; joint criticality is ~7.6× the
sum of marginal criticalities.

**Reading B — the authority-schedule hypothesis (findings 2+3, the big one):**
downstream blocks read LN(x), which is scale-invariant — what a write can
change is the ANGULAR position of LN(x), bounded by ~‖w‖/‖x‖. The stream
grows 8.4× at block 0 and then plateaus, so **write-norm/stream-norm falls
~12× from L0 to L5** (attn-L0 writes 4× its incoming stream; attn-L5 writes
0.34×). Under this reading the front-loaded lesion map is partly ARCHITECTURE,
not learning: layer 0 structurally holds the most authority per parameter,
late blocks are architecturally incapable of large angular moves. The network
schedules authority by stream growth.

**Reading C — noise poisons more than absence (finding 2):** zeroing removes
information; a same-norm random write injects misinformation. Perturbation
norms differ only by ~√2, but observed zero→random ratios run 1.4× (attn L0-L1
≈ pure geometry) to 5.4× (MLP L1: 0.15→0.82) — far beyond √2 for MLPs and
late attention. Where the ratio exceeds √2, downstream computation is
calibrated to the write's DIRECTION (content matters, and actively).

**Discriminating observations:**
1. **Matched-perturbation control (e011c, minutes):** replace write w by w′
   rotated exactly 60° so ‖w′−w‖ = ‖w‖ = perturbation of zeroing. If damage
   ≈ zero-damage → geometry; if ≫ → content. Settles B vs C per component.
2. **Authority-schedule intervention (e014b, the real test of B):** train a
   fresh net with the stream renormalized to constant norm at every block
   input (hook). B predicts the lesion map flattens (late damage rises, L0
   dominance drops). If the map stays front-loaded without stream growth,
   learning, not architecture, owns the front-loading.

**P1 RESOLVED (E011c, 2026-09-24T10:45Z): REFUTED in both parts — and the
truth is cleaner.** At matched perturbation energy (60° rotation, verified
‖w′−w‖=‖w‖=1.0000), rotate-damage vs zero-damage: attn [1.38, 0.74, 0.89,
0.87, 0.71, 1.17]; mlp [0.80, **2.95**, 1.20, 0.70, 0.47, **0.24**].

- **Dominant factor = perturbation ENERGY, not content:** across all 12
  components, damage ranks {zero ≈ rotate60} < {random ≈ √2·energy}. Most
  blocks tolerate scrambled content about as well as — or better than —
  removal. This STRENGTHENS Reading B (authority schedule): what matters
  most is how much a block can move the stream, i.e., the geometric
  schedule, not the specific meaning of its write.
- **Three real exceptions (content-sensitive or energy-carrier):** MLP-L1 is
  direction-sensitive (×2.95); attn-L0 mildly direction-sensitive (×1.38);
  MLP-L5 is an ENERGY CARRIER — zero costs +0.59 but rotate only +0.14, so
  its value is mostly magnitude in the stream, not information. (Connects to
  E011a: MLP-L5 writes the largest residual and matters least per unit.)
- **Caveat before over-reading:** single damage estimates over 20 eval
  batches; ratios 0.7–1.2 may be within noise. Needed observation (cheap):
  bootstrap CIs over eval batches for the rotate/zero ratios; only MLP-L1,
  attn-L0, MLP-L5 look safely beyond noise.
- **The decisive test of Reading B remains e014b** (stream-renorm training).
  P2 prediction unchanged.

**Still-registered predictions (T003):**
- P2 (e014b): stream-renormalized training flattens the attention damage
  profile by ≥50% (L5 damage rises well above +0.03; L0 falls below +2.0).
- P3: damage-per-write across attention layers correlates ≥0.8 with
  write/stream ratio (the geometric authority term) — checkable now from
  existing numbers.

**T003 FINAL RESOLUTION (E014b, 2026-09-24T11:08Z): Reading B REFUTED at
full parity — and the deeper finding is anatomical plasticity.**

Facts: renorm arm (stream pinned to c=5.6 at every block input, train+eval)
reached val 1.610 — BETTER than baseline 1.622, parity gate passed
decisively. Lesion map under renorm: attn damage [2.80, 1.69, 0.48, 0.81,
0.21, 0.03] — still strongly front-loaded (spread 2.77 vs baseline 2.37;
D'5 +0.031 ≈ baseline 0.034). P2 criteria: REFUTED (parity ✓, D'0 ≥ 2.0,
D'5 ≤ 0.05).

1. **Front-loading is NOT caused by stream-norm geometry.** With the
   norm-growth schedule deleted from the architecture, the network still
   organizes early-critical/late-cheap structure. Front-loading is
   functional allocation (early layers do the coarse work), not a
   geometric artifact. The E011c energy-dominance result stands, but the
   CAUSE of the energy allocation is learned task structure, not stream
   growth.
2. **Anatomy is plastic.** Baseline keystones MOVED: MLP-0's keystone role
   (+4.08 nats) dissolved under renorm (+0.10) — plausibly because baseline
   MLP-0 was bootstrapping the 0.67→5.7 norm jump, and renorm does that for
   free. Attention-L0 became MORE critical (+2.80 vs +2.40) with a 2.9×
   larger write (7.8 vs 2.7). Mid-stack reorganized (L2 damage halved, L3
   doubled). Multiple anatomies reach the same function.
3. **Optimization declines late authority even when purchasable.** Renorm
   constrains stream norm at CONSUMPTION points, not write size — any layer
   could still steer strongly by writing big (L0 does: 7.8). Yet L4-L5
   writes DEFLATED (ρ 0.82, 0.70) while L1-L3 partially re-inflated
   (ρ 1.25-1.47; mean 1.10 < 1.3 criterion → re-inflation hypothesis also
   not met as stated). The network allocates authority where the work is.
4. **The norm profile is per-net load-bearing but task-optional:** eval-only
   renorm on baseline weights +3.56 nats (E014b control) vs renorm-trained
   parity — a given net's wiring depends on its norm profile, but learning
   does not require the canonical profile.

Status of T003 after E014b: geometry-schedule story dead; live questions
move to WHY early layers hold the coarse work (curriculum of abstraction?
token-formation bottleneck at L0?) and WHY late MLPs write big-but-cheap
(the MLP-L5 energy-carrier mystery persists and deepened — renorm MLP
damage is now LATE-heavy [0.10, 0.08, 0.23, 0.45, 0.78, 0.70], the
opposite arrangement from baseline). Stream-norm growth is documented (TurnTrout 2023: ~1.045×/
layer in GPT-2-XL, "overshadowing not deletion", NO causal interventions);
front-loaded criticality is widely observed (Gromov 2024, ShortGPT, BERT
pruning) but always explained functionally, never geometrically; renorm
training exists (nGPT trains unit-norm streams, post-LN literally renorms)
but nobody has measured depth-wise lesion profiles under it. **Our
contribution claim: the write/stream→damage link + the causal flattening
test.** Falsifier on record: Pythia/GPT-Neo streams SHRINK with depth
(schedule is regime-dependent), and Gromov/Nepal suggest importance profiles
can be sticky across training interventions.
**e014b design upgrades from the check:** (a) add a post-LN arm alongside the
renorm arm; (b) verify the renormed net reaches comparable loss before
comparing lesion maps (else the comparison is confounded); (c) LOG the
renormed net's learned write/stream ratios — if training RE-INFLATES writes
under renormalization, that alone shows optimization *wants* an authority
schedule, independent of the lesion result.

---

**REVIEW 1 AMENDMENTS (2026-09-24T11:20Z, INTERPRETER findings — accepted):**

1. **T005 WEAKENED.** The "rare-token" signal is ONE head of 36 (L5.h1
   surprisal 6.89 bits; other five L5 heads 4.39 vs L4's 4.28 — below the
   script's own 0.15-bit "rarer" threshold). Re-broadening is +0.13 nats,
   below v002's own "similarly spread" criterion. The ×76 distant jump is a
   ratio of tiny masses, and 62% of L5's non-local mass is mid-range
   (d17-64), not ancient. The 'O'-matching may be a dialogue-vocative
   artifact of a single prompt containing ≥3 'O's. **What survives: L5
   abandons the local d1-3 window (0.234→0.060).** The rare-token framing
   is a hypothesis, not a finding — e013 (causal mask) is GATED on e013a,
   a 200-prompt all-head attention census, so we don't spend causal budget
   on a mechanism that may exist in one head of one prompt.
2. **T003/E014b "functional, not geometric" was OVER-CLAIMED.** Renorm
   pinned block inputs, never write norms — angular authority was never
   capped. The un-refuted live hypothesis: **damage tracks the write/stream
   allocation itself.** P3 now EVALUATED (was pending): renorm-arm attn
   write/c [1.40, 0.58, 0.39, 0.56, 0.31, 0.15] vs damage [2.80, 1.69,
   0.48, 0.81, 0.21, 0.03] — identical rank order (ρ=1.0). The network
   REBUILT a 9.3× declining write-allocation schedule under renorm, and
   front-loading strengthened. Decisive test now queued as **e014c
   write-clamp** (pre-named in the design memo's failure table): clamp
   ‖w‖ ≤ α·‖x_in‖ during training, or eval-time rescale L0/L5 writes by
   k∈{0.5, 2, 4} and measure damage.
3. **T004 depth-6 is definitionally the L5 argmax flip** (depth=6 ⟺
   top1(L4-readout) ≠ top1(L5-readout)) — the "54% finalize at L5" stat
   partially RESTATES the calibrator observation. Upgrade queued as e018:
   CAUSAL depth (activation patching from counterfactual contexts; the
   shallowest depth where splicing switches the final decision).

**T005 FINAL ADJUDICATION (E013a census, 2026-09-24T11:47Z): L5 is a
DIFFUSE re-globalizer — the rare-token framing is dead.** Across 200
prompts, all 36 heads: the locality funnel replicates (far-mass U-shape
0.80 → 0.09 at L2 → 0.54 at L5; local-mass peaks at L2 0.42); L5 abandons
the local window in **82.5%** of prompts (criterion 70%). But attended-token
surprisal is flat across ALL layers (~5 bits) and **zero** heads show
consistent distant-concentration — v002's 'O'-matching head was a one-prompt
artifact, exactly as Review 1 suspected. The causal question shifts: does
L5's calibration (KL(L5‖L4 readout) ≈ 1 nat) actually DEPEND on far context?
**e013 REDESIGNED as context truncation:** last-16 vs full-96 contexts →
measure KL(L5‖L4). Registered: KL shrinks ≥50% with truncated context (L5's
reshaping draws on far information); if unchanged, L5's calibration is
locally derived and "re-globalization" is epiphenomenal attention shape.

**T005 CLOSED (E013 truncation, 2026-09-24T12:05Z): L5's calibration is
LOCAL; far attention is idle grazing.** KL(L5‖L4 readout) barely moves under
256→16 truncation (0.997→0.928, −6.9% vs registered ≥50%); flip rate
unchanged (49→52%). Combined with the census (shape real, no rare-token
selectivity) the full honest picture: **L5 reshapes the output distribution
using LOCAL information, while attending diffusely far for nothing.** The
sharper corpus fact underneath: **16-token sufficiency** — next-char CE at
1.648 (full) vs 1.644 (16 tokens): at char level on Shakespeare, far context
has ≈ zero marginal value for this 2.7M net. Consequences: (a) T004's D2
(integration-of-range) is refuted — late decisions are deep LEXICAL
computation; (b) any "long-range" mechanism claim at this scale must first
show far context matters at its positions (follow-up: per-position far-value
tail — uniform ≈0, or a rare-position minority carrying all of it?);
(c) mid-stack readouts are anti-informative (depth-2 CE 5.19 > unigram 4.17)
— absolute mid-stream distribution claims need a tuned lens; argmax-stability
claims are robust.

## T006 — Anatomical plasticity: what actually persists when the organs move? (2026-09-24T11:21Z)

**Observed (E014b, single seed — replication debt registered):** under
constant-norm streams the lesion anatomy reorganized while the function did
not: renorm arm reached BETTER val loss, yet MLP-0's keystone role dissolved
(+4.08→+0.10 nats), attention-L0 strengthened (+2.40→+2.80), and the MLP
damage profile INVERTED to late-heavy [0.10, 0.08, 0.23, 0.45, 0.78, 0.70].
Write allocation rebuilt a 9.3× declining schedule whose rank order matches
damage exactly (T003 P3, ρ=1.0). Eval-only renorm on baseline: +3.56 nats —
each anatomy presumes its own geometry.

**The question:** when the organs move, does anything stay put? Four layers
of "what persists":

- **PL1 — role migration (organs follow necessity):** baseline MLP-0's
  keystone role was largely "norm bootstrapper" (writing 4.3 into a 0.67
  stream); when the architecture does that for free, the organ's importance
  collapses. Prediction: in the renorm arm, MLP-0's write barely STEERS —
  its angular displacement per token should be far below baseline's.
- **PL2 — functional invariants (the stages persist, organs don't):** both
  anatomies implement the same pipeline — early token-formation, mid-stack
  local completion, late global calibration (the E012 decision-depth profile
  and the V002 locality funnel). Prediction: the renorm arm's decision-depth
  census ≈ baseline's — same depth modes, same letters-late/structure-early
  class ordering.
- **PL3 — anatomy-level degeneracy:** the neuro-ai-lab degeneracy concept
  scaled from routes to whole organ arrangements. Prediction: cross-anatomy
  transplants (baseline organ into renorm net, and vice versa) fail much
  harder than same-anatomy swaps — organs are interchangeable within an
  anatomy (E011b head redundancy) but not across anatomies.
- **PL4 — regime clustering:** optimization pressure + constraints choose
  the organ assignment; assignment is regime-dependent more than
  seed-dependent. Test deferred to v008 (multi-seed lesion-map phylogeny).

**Why this matters for the lab's identity:** if PL2 holds, the correct
dissection units are FUNCTIONAL STAGES (measured by decision depth, locality,
calibration KL), with lesion maps as implementation detail — the instruments
we built this morning (E012, V002) would be measuring the real organs, and
"which layer does X" is the wrong question; "which stage does X, and where
did it land this time" is the right one.

**Discriminating observations (cheap → expensive):**
1. **e012b (minutes, eval-only):** rerun the decision-depth census + angular
   profile on the renorm checkpoint (hooks active). Tests P1+P2 at once.
2. **e028 transplant (design memo queued):** within- vs cross-anatomy organ
   swaps at matched sites. Tests P3.
3. **v008 (later):** multi-seed lesion-map embedding. Tests P4.

**Registered predictions:**
- P1: renorm-arm MLP-0 angular displacement ≤ ⅓ of baseline's.
- P2: renorm depth histogram matches baseline's (bin-wise correlation
  ≥ 0.8; identical class ordering: letters > punct > newline ≈ space).
- P3: cross-anatomy swaps cost ≥ 2× same-anatomy swap damage.
**T006 PARTIAL RESOLUTION (E012b, 2026-09-24T11:32Z): stages are the
organisms; lesion maps are their current addresses.**

- **P2 CONFIRMED (corr 0.822):** the renorm anatomy reproduces the baseline
  functional profile almost exactly — L5-finalization mode identical (1084
  vs 1082 of 2000 positions), same L1 dip, same depth↔entropy relation
  (Spearman +0.344 vs +0.322). The pipeline (early token-formation →
  mid-stack local completion → late finalization) is the INVARIANT; where
  exactly the mid-stack work sits (baseline spreads L0-L4 [158,46,132,160];
  renorm concentrates L3-L4 [348,524]) is implementation detail.
- **P1 missed the strict threshold, direction strong:** block-0 angular
  displacement 0.746 → 0.288 (ratio 0.386 vs predicted ≤0.33); ALL layers'
  angular displacement roughly halved in the renorm anatomy (calmer net).
  Since renorm attn-L0 grew MORE important, MLP-0's own angular role shrank
  further than the block total indicates. PL1 (norm-bootstrapper role)
  supported, threshold pedantically missed.
- **Lab-identity consequence adopted:** the primary dissection instruments
  are now decision depth, locality, and calibration KL (they measure the
  invariants); lesion maps are secondary (they measure where the stages
  currently live in THIS net). "Which layer does X" is deprecated in favor
  of "which stage does X, and where did it land this time."
- Open: P3 (cross-anatomy transplant, e028 design memo in progress) and P4
  (v008 multi-seed phylogeny) test whether whole-organ degeneracy respects
  anatomy boundaries.

**T007 CLOSED (E013d, 2026-09-24T12:12Z): no specific far-context retrieval;
far context acts through bulk statistics.** P1 refuted (+0.22σ < 0.5; 91% of
hurt positions have NO divergent repeat — repetition interference dead). P2
refuted informatively: shuffled-far makes hurt WORSE (−2.26 vs −1.68) while
gains survive (+1.50 vs +1.60) — gains are shuffle-robust (statistical:
char mix/length), and incoherent far text destabilizes more than real far
text. Combined with E013 (L5 calibration local; far attention idle): this
model's long-range behavior is bulk-statistics + noise, not information
retrieval. Long-range claims must be re-tested on tasks that provably
require retrieval (copy spans; e021 task-swap).

**T006 P3 RESOLUTION AMENDED (E029, 2026-09-24T12:16Z): mechanism CONFIRMED,
claim refined.** ΔW-alignment is decisive: same-init organ pairs cos = +0.152,
different-init ≈ 0.000 (max |cos| 0.017) — **training motion from different
inits is almost perfectly orthogonal in parameter space**; organs refine
init-anchored directions. Seed dominance is organ-type specific: **MLP organs
are seed-anchored (ρ 2.0-3.6), attention organs are portable across both
axes** (ρ 0.54-1.29); the one regime-dominant organ is R-host MLP-L0 (the
keystone asymmetry). T008 claim 3 upgraded to H (mechanism confirmed) and
rewritten: "MLP-organ compatibility follows initialization lineage; attention
organs are anatomy- and init-portable." New open question: why? Candidate:
attention READS stream directions that all adequate solutions share; MLPs
WRITE into seed-specific subspaces.

**T008 REVIEW-2 AMENDMENTS (2026-09-24T12:26Z):**
- **Claim 1 → M (downgraded):** the two anatomies compared (B, R) share
  seed 42; e029 showed same-init nets share ΔW directions — stage
  invariance was never tested across seeds. e012c (census on B43/R43,
  running) de-confounds: cross-seED histogram match restores H;
  seed-clustering keeps it init-bound.
- **Claim 3 sub-claim "attention portable" flagged:** MLP ΔW alignment
  (.26/.10/.20) exceeds attention's (.21/.06/.08) — cosine cannot mediate
  the organ-type difference; portability may partly be small-denominator
  artifact (late-attn ablation refs 0.016-0.034; R-host L5-attn is
  regime-dominant). Solid at L0/L3 only. ΔW gap (+0.155) lacks a
  same-init/data-order-replicate ceiling null — registered as needed.

**T008 DEBT RESOLUTION (e012c + e014b.1 + e011c-ci, 2026-09-24T12:40Z):
claims 1 and 2 upgraded to H.**
- **Claim 1 RESTORED at H (init-independent):** 4-net depth-histogram table —
  cross-seed same-regime (B–B43 0.855, R–R43 0.842; mean 0.849) ≥
  cross-regime same-seed (B–R 0.822, B43–R43 0.834; mean 0.828); no seed
  clustering (delta −0.021). Supporting invariants replicate in both new
  nets (L5-finalization 1027/1088; Spearman +0.323/+0.326; class ordering
  letters > punct > structural). Stages are the organism — across seeds AND
  regimes.
- **Claim 2 REPLICATED (second renorm seed R43):** keystone dissolution
  (MLP-0 +0.21 vs B +4.08), late-heavy MLP flip (trend ρ +0.74), attention
  front-load (spread 2.56), and a THIRD independent rebuild of the declining
  write schedule (6.27→0.96 into the pinned stream).
- **e011c exceptions all real:** attn-L0 1.38±0.006, MLP-L1 3.03±0.044,
  MLP-L5 0.25±0.009 — beyond noise by 20-100× sd.

**T009 RESOLVED (E021, 2026-09-24T13:02Z): all four predictions landed.
Retrieval is task-elicited, not architecturally absent.**
- P1: 100% copy accuracy (control 4.1%); CE at COPY 0.007 nats — the copy
  is noiseless at 2.7M params.
- P2: far-value +3.269 ≈ ln 26 at COPY (control −0.002). **T008 claim 4
  AMENDED (stays H, narrowed scope): "no far-context retrieval on natural
  char data at this scale." When the task demands retrieval, this exact
  architecture delivers it exactly.**
- P3: dedicated retrieval head L4-H1 (95.1% mass on the ID nonce; control
  15%). The idle-grazing signature was a property of the corpus, not the
  architecture.
- P4: NEW decision mode — 88.3% of COPY decisions at L4 (Shakespeare ~21% at L4; the earlier "8.0%" was the L3 bin — R5 sync;
  JS 0.265), one layer earlier than Shakespeare's L5 mode: retrieval
  completes before final calibration. Claim 1 (stages) intact and enriched:
  the pipeline reorganizes around task demands; stage membership is
  task-dependent, stage *existence* is not.

**T008 claim-3 mechanism note — AMENDED (V009 full report, 2026-09-24T13:10Z):
the seed-anchored object is the residual-STREAM basis itself.** Right-singular
(input-space) gaps: reads W_in +0.260 / c_attn +0.235 vs W_out-right +0.091
(which initially suggested "reads private, writes shared"). But the
left-singular (stream-space) supplement REVERSES the second half: **W_out-left
gap +0.267 (same 0.562 / diff 0.295) vs c_proj-left +0.071 — MLP stream-writes
are 3.8× more init-anchored than attention's**, matching e029's transplant ρ
(MLP L3/L5 3.08/2.27 vs attn 0.93/1.29); W_in-right peaks at L5 (0.661) where
MLP grafts are most seed-dominant. And diff-seed alignment sits AT the random
floor everywhere (excess ≤ +0.04) — attention portability is NOT shared
subspaces; it is weak anchoring (c_proj barely anchored on both sides).
Refined mechanism: every stream-FACING interface (reads and MLP writes) is
init-anchored; MLP hidden space is barely anchored; c_proj is the
insensitivity exception. **e031 RE-REGISTERED: individual-matrix grafts at
L3/L5 — W_in and W_out each predicted VIOLENT (stream-facing); c_proj
predicted MILDEST of the four.**

## T009 — e021 registration: does a retrieval-required task break the no-retrieval picture? (2026-09-24T12:40Z)

**Design (adopted from Review-2 ideator):** synthetic corpus (~1MB) of
documents: "ID: [5-char uppercase nonce] … Shakespeare filler (>16 tokens)
… COPY: [nonce repeated]". Retrieval is provably required at COPY positions
(nonce is >16 tokens back, unpredictable without the ID). Control corpus:
same shape, nonces shuffled at COPY (cue uncorrelated). Train fresh 2.7M
nets on each (252s cap, ckpt, parity-style val gates); readouts: copy
accuracy at nonce positions; far-value (CE full-256 vs trunc-16) at COPY
positions; decision-depth census at COPY positions; locality funnel
(e013a-style mini-census) on the task net.

**Registered predictions:**
- P1 (learnability): task net reaches ≥80% next-char accuracy on nonce
  positions at COPY. If not: capacity/budget limit — informative negative.
- P2 (retrieval exists when required): far-value at COPY positions ≥ +1.0
  nat, and the control net shows ≈0. **If confirmed, T008 claim 4 narrows
  to "no retrieval on natural char data at this scale" — NOT an
  architectural limit.**
- P3 (mechanism): attention at COPY positions CONCENTRATES on the ID nonce
  positions (local mass collapses; a real retrieval head appears — the
  Shakespeare L5 idle-grazing signature should be gone).
- P4 (stages): if a new "retrieval depth mode" appears at COPY positions
  (decisions later than any Shakespeare position), the stage picture gains
  a task-dependent member; if depth profile is byte-identical to Shakespeare
  despite retrieval, stages are corpus-trivial — a serious blow to claim 1's
  interpretation.

**CLAIM 5 CLOSED (E003c full report, 2026-09-24T14:05Z): first-order ascent
cannot content-selectively forget.** r at the 0.66-nat bar = 1.43 (2 seeds)
vs 2.0 bar; step-matched naive 1.23 (margin 1.17×); **r vs train-B = 1.08**
— memorization-symmetric damage, zero content selectivity; val_B
"selectivity" was a measurement artifact. e003b's r≈6 head-start failed to
reproduce against its own code+seed (chaotic event, not mechanism). Next
family: weight surgery (e023, in flight) or second-order. MLP-5
energy-carrier causally confirmed (e019: zero 4× rotate; α=0.5 improves CE;
removal spikes entropy +0.62).

**CLAIM 5 RE-AMENDED (Review-3 correction, 2026-09-24T13:35Z): selective-
SO-FAR; the forgetting bar is untested.** Δtarget +0.28 is mild degradation
(train-A 1.30 < val_B 1.68); bar = Δ≥0.66 gap closure. Final r=3.12 (peak
6.13 was a tiny-denominator point); r halves as dose triples — substrate
dimensionality open. e003c (dose-to-bar + step-norm-matched naive +
train-B collateral) will settle it. Original note follows.

**T008 CLAIM 5 AMENDED (E003b, 2026-09-24T13:28Z): selective first-order
forgetting is POSSIBLE — via projection.** Corrected-labels test: naive
ascent r=1.22 (anti-selective, replicated); masked r=1.84; **projected
ascent r=4.84-6.13** (Δtarget +0.28 at Δcollateral +0.09). Removing the
single mean retain-gradient direction aims the damage at the target —
implying the shared fluency substrate is largely ONE-DIMENSIONAL in
gradient space. Claim 5 final form: "naive and masked ascent cannot
selectively forget; projected ascent can (r≈5); the shared damage substrate
is low-dimensional."

## T008 — The anatomy of a 2.7M char transformer: first synthesis (2026-09-24T12:06Z)

Assembling the morning's dissections into one picture. Confidence: H
(replicated/causal), M (single decisive test), L (suggestive).

**1. Function is staged; stages are the organism (H).** Early token-formation
→ mid-stack local completion → late distribution calibration. Invariant
across two anatomies (E012b, depth-histogram corr 0.822; L5-finalization
1084 vs 1082/2000). Instruments: decision depth, locality funnel, KL(L5‖L4).
Lesion maps report where stages landed THIS net, nothing more (E014b:
keystone dissolved under renorm with parity loss).

**2. Anatomy is plastic; allocation follows function (M).** The renorm net
rebuilt a declining write-allocation schedule (write/c rank-order = damage
rank-order, ρ=1.0) without stream growth. Damage tracks write ENERGY, not
content (E011c ladder), and the network declines authority it doesn't need
(L4/L5 write deflation under renorm). Where does the energy go? Late MLPs
write big-but-cheap (MLP-5 energy carrier: zero +0.59 vs rotate +0.14) —
open question.

**3. Organ compatibility follows initialization lineage, not regime (M;
mechanism test running).** Cross-anatomy same-seed grafts land mildly
(ρ=0.874), same-anatomy different-seed grafts violently (+1.99 vs +0.65 at
L3-mlp). Hypothesis: organs refine init-anchored subspaces (ΔW alignment
observable in e029). Trained-foreign tissue misleads more than random
tissue — interference is content-specific (E028).

**4. There is no long-range information retrieval at this scale (H for this
model).** L5's calibration is local (E013: KL −6.9% under truncation); far
attention is idle grazing (E013a census); far-value is bimodal but both
tails are structure-insensitive — gains shuffle-robust (bulk statistics),
hurts shuffle-amplified (destabilization; E013d). Depth = lexical
discrimination demand, not range (T004 P1 sign-flip + E013).

**5. First-order forgetting is impossible at every granularity tested (M).**
Ascent destroys the shared fluency substrate first (r ≈ 1.0x at all doses,
all content distances; T002). Untested instruments: weight-targeted/
projected ascent (e003b READY); entity-granularity embedding surgery (e023).

**The frame's falsifiers (what would break this picture):**
- A task that provably requires far retrieval where this model succeeds
  (e021 task-swap) — would break claim 4.
- Stage structure failing at other scales/corpora (the whole picture is ONE
  architecture, ONE corpus, ONE scale — the frame's biggest limitation;
  scaling ladder e004/e005 re-motivated by synthesis, not by novelty).
- e029 contradicting init-lineage (would demote claim 3 to correlation).

**Replication debt (blocking upgrades to H):** e014b.1 (plasticity seed),
e011c CIs, multi-seed lesion maps (v008 phylogeny would settle claims 1-2
at once).

**Top-3 next by expected information:** (1) e021 task-swap — does the stage
picture survive a copy-task (where far retrieval IS required)? tests claims
1+4 jointly; (2) v008 anatomy phylogeny — settle 1-2 with seeds; (3) e003b
targeted ascent — last clean instrument on claim 5.

## T007 — Far context is a double-edged sword: the bimodal far-value distribution (2026-09-24T11:55Z)

**Observed (E013c, 2000 positions):** far-value = CE(16 ctx) − CE(256 ctx)
has mean ≈ 0 but is NOT concentrated there: **30.6% of positions gain ≥ 0.15
nats** (top decile mean **+1.60**; p99 +2.96) while **28.2% LOSE ≥ 0.15**
(bottom decile −1.68). "16-token sufficiency" was an average hiding a
tug-of-war. P1+P2 confirmed (tail exists and is heavy); P3 refuted (far-value
does NOT simply track local difficulty, ρ=0.133 — it tracks something about
the POSITION, not its hardness).

Top gainers: locally-ambiguous rare continuations ("the carp" → "T",
"ere " → "s", "Thus in pl" → "e") — far context disambiguates (or the model
memorized the passage).

**Hypotheses for the two populations:**
- **G1 (disambiguation):** gainers are positions whose local window is
  consistent with multiple distinct continuations present in the corpus;
  only far context (or memorized uniqueness) picks the right one.
- **H1 (interference):** losers are positions where the far context contains
  an earlier similar n-gram whose CONTINUATION differs (repetition priming
  pulls the prediction toward the wrong repeated pattern — Shakespeare
  repeats phrases with variations). Far context misleads via induction-like
  copying.
- **H2 (noise/settling):** losers are just positions where the model's
  long-context representations are miscalibrated — no specific interfering
  pattern exists.

**Discriminating observations:**
1. For loser positions, search the far context for max n-gram similarity
   (longest common suffix-match with a different following char). H1
   predicts losers have systematically closer divergent-continuation
   matches than neutral positions.
2. Shuffled-far context (destroy far structure, keep length): both tails
   collapse toward 0 if structure-driven (H1+G1); a surviving tail is
   length/artifact-driven (H2).

**Registered predictions:**
- P1: loser positions have a closer divergent-continuation n-gram in far
  context than neutral positions (effect ≥ 0.5σ).
- P2: shuffled-far context collapses BOTH tails substantially (bottom-decile
  mean rises from −1.68 to ≥ −0.5 AND top-decile falls from +1.60 to ≤ +1.0)
  — structure drives both; either tail surviving implicates H2 for it.

## T005 — The locality funnel and L5's rare-token re-globalization (2026-09-24T11:12Z)

**Observed (V002 attention atlas):** attention locality has a depth profile —
entropy L0 4.51 (near-uniform) → L3 1.64 (tightest, 50.7% mass at distance
4-16) → L4/L5 re-broaden. L5 abandons the local window (d1-3 mass 0.234→
0.060 vs L4) and reads RARE, FAR identity tokens (d65+ mass ×76; one head
puts 0.73 of mass on exact 'O' matches; attended surprisal 4.28→4.81 bits at
flat entropy). Sanity: recomputed attention matches model output to 4.8e-7.

**The mechanism claim:** mid-stack layers solve local completion (where
decision depths concentrate, E012); L5's job is re-globalization — pulling
distant, low-frequency identity evidence (speaker, register, topic) to
reshape the distribution tail. Explains the E012 paradox: KL(L5‖L4 readout)
≈ 1 nat with +0.03 ablation CE — the calibration is tail-shaping, not
argmax-flipping.

**Hypotheses for why re-globalization is LATE:**
- R1 (division of labor): local constraints resolve by mid-stack; the
  remaining uncertainty (which valid continuation fits the far context) is
  only resolvable by distant evidence, so it's the last thing computed.
- R2 (cheap insurance): rare-token evidence mostly confirms an already-good
  distribution — small CE value, large distributional effect.
- R3 (interference avoidance): doing global reads early would contaminate
  local completion; the funnel ordering is an architectural convention the
  optimizer finds reliably.

**Discriminating observations:**
1. **Causal mask test (e013):** mask exactly the attended rare tokens
   (positions recorded in runs/v002/metrics.json) → R-predictions below.
2. Decision depth on positions right after rare identity tokens vs generic
   positions (do rare tokens push decisions deeper?).
3. Prompt without any rare identity tokens (generic prose) → does L4→L5
   re-globalization shrink?

**Registered predictions:**
- P1: masking the v002-identified attended tokens reduces KL(L5‖L4 readout)
  by ≥50% while mean CE moves <0.05 nats (tail-shaping, not argmax).
- P2: positions following rare identity tokens have systematically deeper
  decision depth (mean depth ≥ +1 vs generic positions).
- P3: rare-free prompts shrink L5's distant-mass fraction by ≥ half.

## T004 — Decision depth: predictions form at different depths per token (2026-09-24T10:5xZ)

**Observed (V001 token journey, logit lens through depth):**
- "…torches to burn " → top-1 walk: emb `:`(.16) → `s` → `m` → `i` → **L3
  `t`(.69)** → L4 `t`(.63) → L5 `t`(.33). Decision at L3; L5 DEGRADES top-1
  confidence by half.
- "To be, or not to " → `\n` → `d` → `d` → `l` → `m` → **L4 `b`(.22)** → L5
  `b`(.23). The correct answer only exists from L4 on.
- Authority panel: L0 write/stream ≈ 8.5 vs ≈ 1 for L1–L5; angular
  displacement 0.7 at L0 vs 0.2–0.3 later (T003's schedule, now visible).

**The new observable:** *decision depth* — the shallowest readout depth whose
top-1 equals the final top-1 and remains stable. It varies per token/context.
This gives per-position anatomy: WHERE a specific prediction gets made, not
just how much each layer matters on average.

**Hypotheses:**
- D1: decision depth tracks constraint strength — strongly constrained
  continuations (low next-token entropy) are decided early; open contexts
  wait for later integration.
- D2: late decision = longer-range integration required (position attends
  far back only in later layers).
- D3: L5's confidence drop (prompt 1) is refinement — probability mass
  spreading over multiple valid continuations — not degradation; L5 may be a
  "distribution sharpener" whose ablation cost hides in averaged CE.

**Discriminating observations:**
1. Measure decision depth over ~2000 val positions; correlate with final
   next-token entropy (D1) and with attention-distance statistics (D2).
2. Split positions by decision depth (≤2 vs ≥4) and measure per-split CE
   increase under L4+L5 zero-ablation (D1/D3: late-decided positions should
   suffer more if late layers carry real function).
3. L5-refinement test (D3): compare full next-token DISTRIBUTIONS (KL, not
   CE) at L4 vs L5 readouts — if L5 sharpens/plattens distributions without
   changing argmax, its role is calibration.

**Registered predictions:**
- P1: decision depth and final entropy correlate ρ ≤ −0.4.
- P2: positions with decision depth ≥4 suffer ≥2× larger L4+L5-ablation CE
  increase than positions decided at ≤2.
- P3: L5 readout changes distribution shape (KL > 0.05 nats vs L4 readout)
  even where argmax is stable — L5 is not dead weight, it is a calibrator
  whose average ablation cost (+0.03) understates its per-token role.

**Visualization dividend:** this concept existed in none of our numbers; it
appeared the moment the prediction was drawn through depth. Exactly the
user's representation principle: seeing → noticing → manipulating.

**T004 RESOLVED (E012, 2026-09-24T10:55Z): 2 of 3 predictions confirmed.**
- **P2 CONFIRMED (the construct earns its keep):** late-decided positions
  (depth ≥4) suffer 3.04× more ΔCE under attn-L4+L5 ablation (0.328 vs
  0.108). Decision depth is a per-position predictor of lesion
  vulnerability — anatomy is token-local, not just corpus-average.
- **P3 CONFIRMED: L5 is a calibrator.** KL(L5-readout ‖ L4-readout) mean
  1.03 nats vs +0.03 mean ablation CE. "Vestigial L5" is dead: L5 reshapes
  the output distribution massively; argmax and mean-CE are blind to its
  work. Open question: what does it calibrate toward (temperature? tail
  mass? position-conditioned rare-token boosts?).
- **P1 REFUTED with sign flip (ρ=+0.32):** late decisions ↔ HIGHER entropy.
  D1 was backwards: constrained positions are trivially decided at emb/L0;
  open contexts recruit deeper integration. The interesting quantity is not
  "constraint → early" but "integration demand → late."
- Lens caveat on all readout claims: mid-network ln_f+lm_head decoding is a
  heuristic; argmax-stability claims are robust to it, absolute KL values
  are not.

## T001/T002 AMENDMENTS — adversarial critique harvest (2026-09-24T10:2xZ)

Full critique: `scratch/critique_T001_T002.md`. Corrections accepted (append-
only; original entries above stand as written, amended here):

**T002 amendments:**
1. **EVAL LABELING BUG (serious):** `val_a` = corpus 90–95% and `val_b` =
   95–100% — BOTH are late-corpus (B-side) text. The model trained on all of
   0–90%, so A-side held-out text never existed. Consequences: the
   "collateral runs ahead of target" claim is RETRACTED (it was also a grid
   artifact: interpolated ΔB at ΔA=1.0 was 0.93, behind); the r(t)≈1.0 result
   measures damage uniformity across two B-side sets, NOT target-vs-
   collateral. What survives untouched: total collateral collapse (held-out
   text CE exploded at every dose) and no-selective-operating-point. What was
   never measured: target-side damage. → e003b must use train-A CE
   (memorization readout) as the target metric, val_B as collateral.
2. **"≈ random" WRONG:** final CE ≈ 27 nats vs ln(65) = 4.17 — the model went
   6.6× PAST random into actively anti-informative predictions. Ascent
   doesn't randomize the net; it inverts it. (Worth its own question: what
   does the model systematically over-predict post-ascent?)
3. Probe names mislabeled (PROSPERO lives in the val region, never trained;
   ROMEO straddles the A/B boundary) — generation probes were not A/B-valid.
4. French was filtered to the 65-char vocab (accents stripped) — the
   dissimilar arm is "accent-stripped French", still valid as dissimilar
   content, but note it.
5. Standing after amendment: first-order ascent produces TOTAL collateral
   damage; gradient space separates same-corpus vs French (frequency caveat:
   the separation may be unigram-frequency distance, not content — normalize
   out the unigram-gradient component before trusting it as "content").

**T001 amendments:**
6. **H4 weakened, not refuted:** E011a measured ABSOLUTE write norms, but the
   residual stream grows ~0.39 → ~10.7 across depth, so RELATIVE perturbation
   (write/stream) still falls ~40× with depth. The geometry confound survives
   in relative form. New discriminator: **orthogonal-innovation control** —
   replace each block's write with a same-norm random vector; if damage ≈
   zero-ablation damage, scale/geometry explains it; if damage is much
   different, content structure matters.
7. **The real E001 finding is redundancy (underweighted):** all 36 single-head
   damages sum to 2.10 < attn-L0 alone (2.40); within-L0 heads are ~7.6×
   superadditive. Cumulative ablation should target L0's heads, not L4+L5.
8. Arithmetic fix: damage-per-write falls 51.9× (0.882→0.017), not 11×; MLP
   write norms are U-shaped (4.32→1.82→5.64), not monotone; the efficiency
   front-loading is ATTENTION-specific (MLP damage-per-norm rises L1→L4).
9. Single-lesion damage is MARGINAL contribution, not counterfactual
   necessity (redundant routes hide behind each other).
10. Global caveats: single seed everywhere; truncated training schedule
    (240 s cap; "converged" means "budget-converged"); e003b/e011b should
    carry at least one replication seed.

**Revised discriminating queue:** e011b (eval-only, minutes): L0 head-subset
redundancy sweep + orthogonal-innovation control + stream-norm profile.
e003b (corrected ascent instruments): dense steps 0–30; projected ascent;
masked ascent — target metric = train-A CE, collateral = val_B CE.

## T001 — What does the E001 lesion map actually show? (2026-09-24)

**Observed:** attention damage strictly monotone with depth (L0 +2.40 → L5
+0.03 nats); MLP-0 keystone (+4.08); MLP damage rising with depth (+0.15 →
+0.60); 16/48 components near-dispensable; model slightly overfit.

**Hypothesis 1 — the da Vinci reading (early layers do the work):** early
attention performs the actual context aggregation for a char-level task;
late attention genuinely contributes little.

**H2 — off-manifold ablation artifact:** zeroing a block's residual write
pushes downstream LayerNorms off their calibrated statistics. Damage measures
*distribution shift*, not *information content*. A "harmless" block might be
quietly important while a loud lesion is just miscalibration.

**H3 — redundancy, not vestigiality:** late layers may duplicate each other.
Single-block ablations under-measure a block whose function is also carried
by its neighbors (cf. the old lab's superadditive three-layer block).

**H4 — residual-stream scale confound:** in pre-LN residual nets, if block
write norms shrink with depth, then zeroing late blocks changes the stream
less *by construction*. "Front-loaded importance" could be "front-loaded
writes" — a geometry fact wearing an anatomy costume.

**H5 — under-training:** at ~2k steps late layers may not yet have
specialized; importance might migrate up with longer training.

**Discriminating observations (cheap → expensive):**
1. ~~Measure per-block residual write norms~~ **DONE (E011a, 2026-09-24): H4
   REFUTED as the explanation.** Write norms per token: attn [2.72, 2.49,
   3.31, 2.73, 2.39, 1.93] — NOT monotone (L2 writes the most!); mlp [4.32,
   1.82, 2.27, 2.73, 3.36, 5.64] — RISES with depth. Yet damage/write-norm
   still falls 11× across attention layers [0.88 → 0.02]. The front-loading
   is information architecture, not residual scale. New anomaly for the list:
   MLP-5 writes the largest residual in the net (5.64/token) yet costs only
   +0.59 nats to ablate — late MLPs write large, dispensable content. (What
   is it writing, and for whom?)
2. Mean-replace instead of zero (keep block's mean activation): if damage
   collapses, H2 (miscalibration) explains much of the lesion map.
3. Ablate-then-recalibrate: freeze everything, fine-tune ONLY LayerNorm
   affine params for ~200 steps after each lesion. If damage shrinks a lot,
   the lesion map overstated importance (H2); what remains is closer to true
   information content.
4. Cumulative ablations L4+L5, L3–L5: superadditive damage ⇒ H3.
5. Lesion maps at 500 / 2000 / 8000 steps: importance migrating ⇒ H5.

**Registered predictions (written before running):** write norms will NOT
decline monotonically with depth (they usually grow or stay flat in trained
residual nets), so H4 will NOT fully explain the front-loading; LN-only
recalibration will recover a meaningful fraction (≥30%) of MLP-0's damage,
meaning the +4.08 headline overstates true information content.

**Design consequence:** e011 (MLP-0 anatomy) must include the mean-replace
and LN-recalibrate controls or it will rediscover H2 the hard way.

---


[R56 AMENDMENT — THE RULER BENT, THE CONE WIDENED]: the critic's
eval-only cells on the same organ (scratch/r56_critic.md): matched-L2
isotropic noise kills only at ~24-32x displacement (g0 0.03-0.15 at
32x; first-order prediction ~58x from cos(grad g0, wash) = -0.44) — the
registered 4x spare leg sat 6-16x BELOW the isotropic kill threshold
(concentration of measure in ~874k dims; the contrast was guaranteed to
spare, so it measured geometry, not a basin). BUT the critic's tilt
ladder shows random 45-degree tilts of the wash direction STILL kill at
~2x. RESCOPED: the sensitive set is WIDE-ANGLE and low-measure —
anisotropy ~12-16x (wash-aligned ~2x vs isotropic ~25-32x); forgetting
is still direction-typed, but the effect size is a threshold RATIO, and
"matched-L2 isotropic spares at 4x" must never be cited as evidence.
The organ-draw n=1 scope stands (auditor).


[R56 AMENDMENT — CHANNEL SCOPE, SEQUENTIAL PENDING]: the critic's read
of the same committed metrics: the wall holds the fact's BATTERY
channel; the SAME fact's wpe-band channel reads ~0.001 by +2 INSIDE the
ball and row0 strength decays 0.76->0.62 by +50; no free-run generation
read ever existed; the +0.53-nat tax is a standing interest payment
(~half of future adaptation on the easiest stream). Sequential memory
was never tested — without it "memory is made architectural" risks
reducing to "the organism was frozen with its probe intact." The
claim's form until g1bW lands: "the wall holds the fact's
battery-channel expression through a wash that kills the control"
(wash-draw n=3, one root — the root redraw g1c-root queued). g1bW
(SPLINT-REFUTED / MUSEUM / ZERO-SUM + free-run battery) dispatched
21:25Z.


[R56 AMENDMENT — CONSTRUCTION DISCLOSURES, THE HEAD-TO-HEAD DEBT]:
(i) REFRACTORY=24 sits above the registered band's lower edge (20) —
"100% of spacings in [20,45]" was guaranteed from below; only the upper
edge was a measurement. (ii) Wash intensity was CONSTANT in every g2
run — "self-timed" has not been distinguished from "threshold +
cooldown at one threat level" (a fixed-period oscillator built from a
thermostat). (iii) The one existing head-to-head vs a fixed 1/32
replay schedule: the FIXED arm read 0.693 vs the organ's 0.587-0.615
(the critic's read of the committed run) — matched or beaten. The noun
stands as "an event-driven maintenance that replicates (wash-draw n=3,
one root)"; the claim "the organ beats a fixed schedule" is NOT ours
until g2g registers the head-to-head with the threat-level ladder.


[E188 AMENDMENT — REDUCED, 12:05Z]: the trajectory hypothesis loses
its integral layer (RAW-WINS: the alignment-weighted currency lost
to raw displacement, CV 0.233 vs 0.557; pre-registered Branch B
executed). WHAT REMAINS: the STATIC-vs-LEARNED contrast (g3K's
kappas 4-10x at matched per-coordinate RMS, n=1, replication still
owed) — "learned paths reach the gate at 1x; static jumps need
4-10x" — with no claim about alignment as the currency.


[E188 RESOLUTION, 12:05Z]: the vocabulary clause RESOLVED BY
MEASUREMENT — install-vs-wash cos in [-0.034, -0.018] on all four
organisms; |cos| < 0.3 everywhere -> task-arithmetic vocabulary
REJECTED; the wash is the corpus's adaptation direction, not the
fact's negation. R3d resolves: report the cosines, reject the
vocabulary.


[E188 AMENDMENT — THE GATE STRENGTHENED, 12:05Z]: e188's RAW-WINS
is the displacement gate's best evidence: D at death varies 0.6%
across three wash seeds at matched t* (2.489/2.484/2.500) — the
tightest invariance the wash arc has produced — while the aligned
currency spreads 3.7x (a seed lottery). The decomposition reads:
CLOCK = Adam's normalization (opt1); GATE = raw displacement
(e188); CURRENCY-of-reaching = open (opt1b running, opt1c
dispatching); ALIGNMENT = passenger (flips positive post-kill on
the fast arm).


[E188 AMENDMENT — REDUCED, 12:05Z]: the trajectory hypothesis loses
its integral layer (RAW-WINS: the alignment-weighted currency lost
to raw displacement, CV 0.233 vs 0.557; the pre-registered Branch B
executed). WHAT REMAINS: the STATIC-vs-LEARNED contrast (g3K's
kappas 4-10x at matched per-coordinate RMS, n=1, replication still
owed) — "learned paths reach the gate at 1x; static jumps need
4-10x" — with no claim about alignment as the currency. The paper
sentence becomes the displacement-threshold form.


[E188 RESOLUTION, 12:05Z]: the vocabulary clause RESOLVED BY
MEASUREMENT — install-vs-wash cos in [-0.034, -0.018] on all four
organisms; |cos| < 0.3 everywhere -> task-arithmetic vocabulary
REJECTED; the wash is the corpus's adaptation direction, not the
fact's negation. The R3d pre-empt resolves: report the cosines,
reject the vocabulary.


[E188 AMENDMENT — THE GATE STRENGTHENED, 12:05Z]: e188's RAW-WINS
is the displacement gate's best evidence: D at death varies 0.6%
across three wash seeds at matched t* (2.489/2.484/2.500) — the
tightest invariance the wash arc has produced — while the aligned
currency spreads 3.7x (a seed lottery). The decomposition now
reads: CLOCK = Adam's normalization (opt1); GATE = raw displacement
(e188); CURRENCY-of-reaching = the open question (opt1b running,
opt1c dispatching); ALIGNMENT = passenger (flips positive
post-kill on the fast arm — the organism returns to the grave it
dug).

## T002 — Why was unlearning anti-selective? (2026-09-24)

**Observed:** naive ascent on half A destroyed the model (ΔA +26, ΔB +26
nats ≈ random). Worse: by the time A rose +1 nat, B had ALREADY risen +1.51 —
collateral ran ahead of target. Retain anchor slowed but did not rescue.

**H1 — dose pathology:** AdamW on −CE at lr 2e-5 explodes; the model passes
through a regime where everything degrades before A-specific structure fails.

**H2 — structural non-separability (the deep one):** A and B are halves of
the SAME corpus — same style, same vocabulary, same char statistics. Their
gradients are nearly parallel, so ANY weight motion that damages A-knowledge
damages B-knowledge first (shared 'fluency' substrate fails before
content-specific memory). Selective weight-level forgetting of same-
distribution material may be impossible in principle at this scale.

**H3 — wrong measurement axis:** CE mixes general fluency with content
memory. The model may have lost fluency everywhere while both memories are
intact-but-unreadable; or A-memory gone and we can't tell through the
fluency smoke.

**H4 — wrong instrument, not wrong target:** uniform ascent steps are the
blunt tool; weights differ in A-specificity. Targeting only low-A/B-gradient-
overlap weights might find selectivity that uniform stepping can't.

**Discriminating observations:**
1. Gradient cosine between A-batches and B-batches (no training, minutes):
   cos ≳ 0.8 ⇒ H2 is structural and no LR sweep will fix it; cos ≪ 1 ⇒ H1/H4
   remain live.
2. Sweep lr 1e-6…1e-4 × steps; plot the (ΔA, ΔB) trajectory. An operating
   point with ΔA ≥ 1 and ΔB ≤ 0.1 would falsify H2 for this setup.
3. Dissimilar-content unlearning (Shakespeare vs French/code splice): if
   selectivity appears ONLY there, H2 is confirmed — selectivity is a
   property of content distance, not method.
4. Separate fluency from content: after mild ascent, probe A-specific names
   (chars/names unique to A) vs generic continuation quality (H3).
5. Targeted ascent on low-overlap weights only (H4).

**Registered predictions:** grad cosine A↔B will be ≥ 0.85 (H2 structural
for same-corpus halves); no lr in the sweep achieves ΔA ≥ 1 with ΔB ≤ 0.1;
selectivity WILL appear for the dissimilar splice. If these hold, the real
research question shifts from "how to unlearn" to "what is the content-
distance dependence of achievable selectivity" — a curve, not a method.

**FINAL RESOLUTION (E003 trajectory audit, 2026-09-24T10:20Z):** No transient
selectivity window exists. r(t) = Δtarget/Δcollateral over the full
trajectories:

- LR sweep (same-corpus): r peaks at **1.09** during gentle ascent (lr 1e-6/3e-6
  early steps) and DECAYS to 1.01–1.02 as damage grows.
- Dissimilar arm (French): r ≈ **1.02–1.05 at every point** from step 25 on.
- Implant context: teaching French (400 steps) itself cost Shakespeare +0.44
  nats — interference is bidirectional; the substrate was never clean.

**Hypothesis verdicts (T002):**
- H1 dose pathology — **DEAD**: anti-selective at every lr; r is
  dose-independent.
- H2 gradient parallelism (as stated) — **DEAD** (cos 0.345).
- H2′ shared-fluency-substrate dominates — **STRONGLY SUPPORTED**: ~97% of
  ascent damage is content-independent. This is the surviving mechanism claim:
  *in a converged small LM, first-order ascent cannot produce content-
  selective forgetting; the walk immediately enters the shared fluency
  subspace.*
- H3 measurement axis — **superseded** by the r(t) analysis (the shared
  component IS the fluency substrate; measuring it separately changes nothing).
- H4 wrong instrument — **THE LIVE ONE**: gradient space separates content
  (0.345 same-corpus vs 0.144 French) but uniform optimizer steps don't
  exploit that. Two candidate instruments: (a) *projected ascent* — step along
  g_A minus its component along the mean B-gradient direction; (b) *masked
  ascent* — step only on weights with high A-specificity (|g_A| high, |g_B|
  low).

**Registered predictions (e003b):**
- P1: projected ascent lifts r to ≥ 1.5 at gentle doses. If r stays ≤ 1.2
  even with the B-direction projected out, the fluency subspace is
  HIGH-DIMENSIONAL and first-order selective forgetting is impossible in this
  regime — a small law worth stating precisely.
- P2: masked ascent (top ~10% A-specific weights) lifts r to ≥ 2.
- P3: the sub-25-step window (dense sampling, every 5 steps) also shows r
  ≤ 1.2 (i.e., the shared-subspace entry is immediate, not a fast transient
  we missed).

**Design consequence:** e003 is redesigned around these discriminators
(gradient cosine + dissimilar-content arm + fluency/content split + fine LR
sweep), not a blind hyperparameter grid.

---

*Next thinking obligations: e003 results (the selectivity curve), e011 (MLP-0
must run mean-replace + LN-recalibrate controls), write-norm profile (T001
discriminator 1 — cheapest, should run first).*
