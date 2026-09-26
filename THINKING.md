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

**Bearing on the RIF conflict (T049):** the kernel's young-heavy
concentration + the bigram-induction findings both live in the same
young band — e087's adjudication now has a structural prior.

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
