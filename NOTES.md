# Lab Notebook

Append-only. Newest entries at the top. Format per experiment:

```
## E### — name (date)
WHAT WE DID / WHAT WE SAW / WHAT'S NEXT
```

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

## E063b — task-swap discriminator: H-ii OPTIMIZER-ATTRACTOR (2026-09-26) — DONE

WHAT WE DID: A-profiles on the e021 family (copy-task net: far-retrieval
learned, +3.27 far-value, 99.96% copy acc; + its control net as bonus
third), own-corpus val, e063 machinery verbatim, B anchor bitwise.

WHAT WE SAW (T041 amendment): copy-net shape r = +0.998 vs B template
(control +0.998 too); H-i's registered r < 0.5 prediction decisively
failed. The L0-huge/trough/rise template is corpus-invariant — an
optimizer/architecture attractor, not task-pinned. Nuance: the task
re-weights MAGNITUDES (mean |ΔA| 0.27, uniform elevation, L1 trough
partially filled 0.42 vs 0.15 — Spearman dips to +0.83 barely at bar);
shape is preserved, scale is not.

WHAT'S NEXT: T041 closes (template = attractor). Remaining open: what
WOULD move the shape — e033's energy constraint is the only known
mover; depth/architecture sweep parked.

---

## E069 — T039 onset discriminators: H1 circuit-horizon DECISIVE; D1 surprise — eval-window-sensitive (2026-09-26) — DONE

WHAT WE DID: the two T039-registered eval-only discriminators on the
frozen e053c net (CPU, 282 s, gates G0b/G1/G2/G3 all pass; a*(512)
reproduces 6 [4,8] bitwise).

WHAT WE SAW (T039 amendment): **D2 = H1 CIRCUIT-HORIZON, decisive** —
shuffled-char contexts (n-gram statistics destroyed, recency kept) leave
the ages-1-2 spike at 128% of normal (bar: ≥30% H1 / ≤10% H2; CI
[0.98, 1.70], per-seq 0.68-1.95). The recent-token spike is circuit
structure, not corpus statistics. **D1 = EVAL-WINDOW-SENSITIVE
(surprise, recorded as-is)** — the SAME net + sequences truncated to
eval-256 give a\* = 18 (CI [7,30]), outside the registered [4,8];
absolute-position-indexed variant gives 12. The horizon is absolute
w.r.t. the TRAINED window (e053c's cross-window verdict stands) but
shifts under eval-window truncation.

WHAT'S NEXT: T039 amendment card registers the D1 interpretation
question (attention re-anchoring vs instrument reference-shift);
e053c's window-invariant truncation claim needs the eval-window
qualifier in the paper.

---

## E063 — load homeostasis: H-EMERGENT — organ-reliance is a universal template (2026-09-26) — DONE

WHAT WE DID: own-organ load A tracked across cohorts (2.7M/6L: B vs BDO
[same init, diff order] vs B43 [diff init] + exposure arms; 0.84M/4L:
all-11 e040 lineage + e005s + e033), bootstrap noise floor, profile
correlations. CPU-only. Premise corrected en route: e041_bdo/e048 are
2.7M trainstates.

WHAT WE SAW (T041): H-EMERGENT at the registered ladder. Ladder: noise
0.014 | ORDER 0.063 | INIT 0.054 | exposure max 0.162 (base-CE confound
flagged). Profile shape r = +1.000 across order AND init — the
allocation template (L0 ~4 nats, L1 trough ~0.15, monotone rise to
~0.6) is universal. H-setpoint DEAD (order moves A more than fresh
init, 1.17×; replicate rung 0.336 beats init 3×). e033's energy
constraint is the biggest single mover (and even it keeps the shape).

WHAT'S NEXT: mechanically explains T024's trickle and T040's H-nothing
— A has no heritable variance to select; what scatter exists regenerates
through training noise. T041 registers the task-swap discriminator
(e063b, zero-GPU) and the e060 prediction (A-residualized selection
should move the e059 interface family instead).

---

## E059 — winner differencing: H-NOTHING at the bars; interface family is a second predictor (2026-09-26) — DONE

WHAT WE DID: zero-GPU ΔW audit of e040 lineage winners ({g1a,g1c}+children
vs unselected sibs), 8 axes, bootstrap CIs, gates clean (repro 8.7e-08,
REF self-graft 0, organ-band verbatim). 8.4 min CPU.

WHAT WE SAW (T040): no consistent winner signature at the registered bar
(|r| ≥ r(D,A)=0.807). A did NOT move (H-load's directional claim fails).
What winners DID: LN→closer to donor REF (weak), W_out row-norms +2.5%
with shape→REF, W_in erank→AWAY from REF. Sharpest: D itself is
inconsistent as a winner property (E1 −0.05 vs E2 +0.10); the gen-2
trickle was carried by the children — single-lineage artifact at
parameter level. SECOND PREDICTOR FOUND: interface-scale family
(W_out row-norm mean partial r(D|A) = −0.654 p=.040) — genuine
damage axis beyond organ-reliance, but not what selection changed.

WHAT'S NEXT: e063 (running) discriminates WHY A didn't move — defended
setpoint (canalization) vs invisible-to-R; T040 registers the
prediction. Interface family → P2 IMMUNOLOGY crossmatch predictor
candidate (e062).

---

## E053c — ctx-512 onset decider: ABSOLUTE verdict (2026-09-26) — DONE

WHAT WE DID: trained 4L/4H/128d/wpe-512 = 873k params (seed 42,
tokens-per-step matched to the e005s comparator; 180s cap hit at
3133/4000 steps = 78% exposure), then the e053b fine-onset machinery
verbatim (same seeds/rules, 511-position V-zero sweep, 1000× bootstrap).
GPU envelope respected (88°C peak → cooldown + re-check before eval).

WHAT WE SAW (T039): **a\*(512) = 6, CI [4,8], naive=robust=6** —
overlaps the ABSOLUTE window [5,13], entirely below PROPORTIONAL
[10.0,26.1]. Onset fraction HALVED (0.012 vs 0.027). Spike shape
preserved (ages 1–5: +1.7/+4.5/+2.3/+0.9/+0.3 nats; ≤0.007 after age
6). Val CE 1.5227 (beats both anchors). Gates G0b/G0-dev pass
(1.8e-06 / 1.6e-05). Honest caveats: 78% exposure biases AGAINST this
verdict (e053b: less training → larger a*, yet 6 ≤ 7); B=4 spread 3–8;
identity audit stays broken and widened (9.3% of positions
lesion-helpful, scattered to age 499) — plateau ≠ pure recency noise.

WHAT'S NEXT: paper cache-truncation claims restated window-invariantly
(fixed-token horizon). P3 junk-split now ungated (GPU free after
cooldown). Registered T039 discriminator: eval-window truncation on the
SAME net (a* at eval-256 on the e053c net) to separate circuit-horizon
from statistical-horizon readings.

---

## E066b — in-place row-129 interventions: the address is GRADED, not row-pure (2026-09-26) — DONE

WHAT WE DID: T038's H1-vs-H2 discriminator — in-place wpe-row-129
interventions on the 6 donor contexts (swap←130 / zero / mean-row vs
normal), p(Z) readout + d5 Δstate alignment vs relay. 29 s CPU.
Predictions registered in-script before compute.

WHAT WE SAW (T038 update): MIXED/NEITHER at the registered bars —
swap 0.462, zero 0.272, mean-row 0.434 (from 0.715): the row is
PARTIALLY load-bearing in place. Δstates anti-align with relay_d5
(−0.21…−0.26): breaking the address REDUCES relay content (pro-circuit
flavor, moderate). Per-donor spread 0.68→0.23 under swap — contexts
differ in how self-sufficient their content is (conjunction view).

WHAT'S NEXT: e067 address census — full single-row perturbation sweep
maps this partial-weight structure; e066/e066b justify it (the address
is distributed but wpe-row-concentrated).

---

## E066 — close the loop: relay direction vs wpe address row = TWO OBJECTS (2026-09-26) — DONE

WHAT WE DID: P1-COORDINATE ramp step 1 (ideator-registered, T037-refined).
Exact e055 protocol rebuild on e048_repro; mean-donor relay per depth
d0–d6; cosine sweep against all 256 wpe rows; delta-relay (minus shuffled
family), wte sweep + adjacent-row controls. 14 s CPU.

WHAT WE SAW (T038): registered metric |cos(relay_d5, wpe[130])| = 0.094
→ TWO-OBJECTS verdict (bar: ≥0.4 closes, ≤0.15 = distinct; null sd 0.072,
mean |row~row130| baseline 0.237). Depth profile: d0 cos 0.507 @ row 129
(decision position; partly mechanical — shared wpe survives donor
averaging), decaying through the stack to noise at d5 (max anywhere 0.121
@ row 152; rank of row 130 = 19). Delta variant 0.113 — same verdict.
wte control max 0.15 @ ' '. The T037 construct-1 optimism is REFUTED at
depth: the deep relay that rescues p(Z) at 0.912 is NOT the position row
itself.

WHAT'S NEXT: T038 discriminator — is the relay ANSWER-shaped (cos vs
Z-unembedding row) or circuit-address-shaped? Registered in T038; then
e067 address census (is the input-side code sparse?).

---

## E056c — downstream check: LOUD LOGIT PASTE at non-onset (2026-09-25) — DONE

WHAT WE DID: the R14-registered discriminator — free-run continuations
after one-shot d4 writes at the 24 non-onset positions; persistence
curves; 480 continuations total.

WHAT WE SAW (T035): the rescue is TRANSIENT off-onset — p(Z) 0.509 at
+1, floor by +2, zero recurrent Z-words (only 6 offset-0 "ZEPHY:" tag
  completions). Knowledge-specific but not portable. Claim-split final:
position-bound knowledge; real suppression at the bound position;
durable rescue there (e055); logit artifact elsewhere. The YOPO
collision risk shrinks — our write fails to steer even 2 tokens ahead.

## E056b — the circularity killer: rescue is position-general (2026-09-25) — DONE

WHAT WE DID: the d4-rescue donors transplanted at 24 floor-prior
non-onset positions (trajectory-identity verified).

WHAT WE SAW (T034): the rescue fires ANYWHERE (0.324 site-mean, AUC
1.000, shuffled ~0) — circularity resolved moot; the claim SPLITS:
onset-specific sub-argmax knowledge + natural wpe-130 knife-edge;
position-general knowledge-specific d≥4 state injection. Plus e055
full-report: A-rev symmetric suppression; mean-donor relay direction
beats all individuals. Paper draft spine written; submission gate open.

## E055 — the suppression localizer: causal, state-carried, depth-4 (2026-09-25) — DONE

WHAT WE DID: 24-site depth-survival transplant (TF-state at the onset
position, one-shot + held; shuffled/A-rev/base-net/direct800/e001
controls; R1/R2/R3 readouts), all gates pass.

WHAT WE SAW (T033): P1 CONFIRMED (state-rescue d4 0.374 / d5 0.494 vs
shuffled 0.000); P3 CONFIRMED (d*=4 mid-stack; d1-peak/d2-crash
replicated 10.3x); 32 downstream Z-word rows — rescued states express.
The expression gap is causally localized: position-bound knowledge,
destroyed across blocks 1-2, restorable from depth 4. The interventional
study the literature lacks.

## E053b — fine-onset remediation: the invariance was quantization (2026-09-25) — DONE (2-cell partial, honest smoke flag)

WHAT WE DID: fine-grid onset fitting on the cache-timeline cells (no
bin-edge fallback), bootstrap CIs, identity test.

RECONCILED vs final e053 metrics: registered-threshold a* = 73/86/4 and 3→21→86 across exposure (GROWS 28.7x, the registered direction); the shrink claim was a sign-based statistic — OPEN CONFLICT documented, ctx-512 + more seqs to resolve. Shape claims robust: spike+plateau, sink dead 5/5, 32% negative-utility old positions at 10M.

Superseded partial read: fine a* = 7/25/35/182/32 — 0/5
near "63"; the curve is a sharp recent spike (~4-15 positions, dCE
0.3-6.7) + dead old end (240-255: dCE <= 0.015, sink included) + 13-20%
NEGATIVE-utility entries (lesion helps); exposure INVERTS the prediction
(training SHRINKS the live window: live-frac 0.68→0.17→0.13 across
400→800→4000 steps); identity broken with sign flips. Honest publishable
core: ~85-95% of the ctx-256 cache is dead weight; sink dead at
generation; training collapses the live window. ctx-512 = the registered
absolute-vs-proportional decider.

## E064 + E053 — the stress test kills the unification; the timeline lands (2026-09-25) — DONE

E064 (gate stress): **KILLED** — R43's ladder peaks L3 with its gate at
L5 (and R peaks L0 with its gate at L4): the mid-stack interference peak
is shared-method idiosyncrasy, not gate-tracking. T029's unification
dead; two-mid-stack-phenomena residue recorded.

E053 (cache utility timeline, 25.7 min, 5 cells): sink DEAD at
generation (0.007 dCE); utility NOT clean recency (25-50% non-monotone
positions); **live fraction scale-invariant ~25%; onset age ~63
invariant across scale AND exposure** (the grows-with-exposure
prediction refuted); sink-cost decays at 0.84M/10M, flat at 2.7M. First
causal per-position utility curve — prunable-cache numbers.

## E058 — geometry-site anatomy: the causal-gate region carries the anchor (2026-09-25) — DONE

WHAT WE DID: per-site r(damage, geometry) across 11 ckpts at 0.84M + 12
B/B43 grafts at 2.7M; R-ladders by site; all instrument gates bitwise.

WHAT WE SAW (T029): A-residualized geometry replicates at depth (partial
r 0.92-0.96); interference peaks SLIDE with each host's causal gate (B
L3, B43 L4); L0 shows complete criticality/basis-specificity dissociation
(self-ablation +4.08 yet cross-seed R=1.03); anti-alignment exactly at
the rejection zone. The anchor lives in the causal-gate region.

WHAT'S NEXT: e053 (cache timeline) still computing; then interpretation
block before new dispatches per the critic's ratio flag.

## E050 — directed-mutation lineage: VISIBILITY-LIMITED (2026-09-25) — DONE

WHAT WE DID: the e040 protocol with mutation restricted to stream-facing
matrices (W_in/W_out) — the reachability-vs-visibility discriminator
registered in T024.

WHAT WE SAW (T027): D fell only -3.6% (bar -25%), alignment flat, gates
clean. Even directed-at-the-basis mutation gives selection nothing to
work with through the graft-damage trait. FROZEN holds at its strongest;
e060 (A-residualized index) is the last escape hatch.

## E052 — LN/geometry reanalysis: damage tracks organ-reliance (2026-09-25) — DONE

WHAT WE DID: zero-GPU regression of e040's 11-checkpoint graft damage on
four predictors (own-ablation A, LN-distance, dW-alignment, W-space),
bitwise-exact reproduction.

WHAT WE SAW (T026): A dominates (r=0.807); LN excluded (ΔR²=0.066);
geometry aggregate weak but a real L2-localized signal (r=0.747).
FROZEN survives reinterpreted — the trait measured organ-reliance, and
future compatibility selection must use A-residualized damage.

WHAT'S NEXT: e050 (directed mutation) should read out A-residualized or
L2-weighted — method note registered before its results land.

## E040 — graft-evolution lineage: init-anchoring is FROZEN (2026-09-25) — DONE

WHAT WE DID: 11 step-matched 0.84M trainings (seed-42 family wildtype + 3
mutants; fixed REF donor organs; 2 selection events x 3 children;
selection on cross-seed MLP graft damage with parity + organ-band gates;
thermal blocks throughout).

WHAT WE SAW (T024): **P2-FROZEN.** Damage fell only 7.3% over two
generations (P1 bar: 25%); R fell 2.2%; REF-alignment stayed at the floor
(0.001 -> 0.006 vs a 0.53 ceiling). CIs exclude zero — a real trickle,
not noise — but selection cannot see the stream basis at this regime.
The degeneration-route did not fire. C3 complete: the basis is
init-anchored, partial, interface-specific, and not evolvable under
standard selection.

WHAT'S NEXT: day-two report's last slot fills; the day-two arc is
complete. The lab's three stories (anatomy, editing, evolution) all have
first data.

## V012 — portrait v2: the 48-hour self-portrait (2026-09-25) — DONE

WHAT WE DID: v010's four panels refreshed with post-audit numbers (C1
re-anchored to the causal census + its non-invariance) + two new panels:
the retrieval-threshold curve (T021 flip from interference) and the
four-box edit law (T015/T018/T019, n=1 flags boxed). CPU-only.

WHAT WE SAW: the lab's third flagship artifact — one figure that carries
the whole 48-hour story with its confound flags visible.

## E033 — write-equalizer: the energy schedule is decorative (2026-09-25) — DONE

WHAT WE DID: fresh 0.84M net with every MLP write renormalized to one
uniform norm (1.64; hooks train+eval); baseline lesion map + equalized
lesion map + calibrator KL. Envelope-compliant (batch 32, cooldown).

WHAT WE SAW (T023): P1 parity TRUE (1.5147 < baseline 1.537 — BETTER);
P2 energy-migrates FALSE (attention unchanged; MLP damage rose/flatten);
P3 calibrator survives TRUE (KL 1.176). The late-MLP energy carrier is
real but the growing schedule is an allocable habit — the net defends
the coarse allocation, not the write norms. 0.84M-scoped, single net.

WHAT'S NEXT: e040 lineage RUNNING (11 step-matched trainings, thermal
blocks). README updated with the day-one/two results summary.

## E049 — the retrieval threshold (2026-09-25) — DONE

WHAT WE DID: refrain corpora at p ∈ {0,5,20,60}% (24-32-char verbatim
refrains in Shakespeare filler), 4 fresh 2.7M nets + a 10M arm at p5;
far-value / accuracy / retrieval-head readouts at refrain AND ordinary
positions.

WHAT WE SAW (T021): threshold ≤5% (refuted-low), GRADED not sharp, ZERO
leak (compartmentalized), 10M 1.87x more sensitive (weak support). THE
FLIP: far context HURTS at p0 (−2.24 nats, T007's interference) and turns
net-positive by p5 — far-value is a tug-of-war flipped by ~200 refrain
events. Retrieval heads form discretely in the LATE-ATTENTION slot (L5 at
6 layers, L7 at 8) — preferred depth, not preferred density. Cross-talk
at 60%.

WHAT'S NEXT: L4's arc complete (no-retrieval → scale erosion → threshold
mapped). Overnight program: e040 graft-evolution re-scoped remains; then
session review.

## E005s — the scaling capstone (0.84M / 2.7M / 10M) (2026-09-25) — DONE

WHAT WE DID: two new nets (4L/4H/128d = 0.84M; 8L/8H/320d = 9.98M), same
corpus/seed, frozen readouts P1-P4 registered before training.

WHAT WE SAW (T020): gate structure universal with relative depth sliding
by architecture (1.0 -> 0.6 -> 0.29); front-loading INTENSIFIES (11x ->
72x -> 116x; at 10M the three deepest attention blocks cost <=0.04 nats);
16-token sufficiency erodes monotonically with scale (+0.021/-0.001/+0.035);
address surgery scale-invariant in shape (S_name 1293/332, class-exact,
+0.0005 corpus). Distributed decisions collapse with depth (26->15->7%).
Harness bug fixed (resume map_location).

WHAT'S NEXT: card v3 scale-stamped; e049 (far-retrieval threshold) gains
priority from L4's erosion.

## V011 — the edit film (2026-09-25) — DONE

WHAT WE DID: six-frame filmstrip of law L7 from saved metrics (VISUALIZER
agent, CPU-only). runs/v011/edit_film.png.

WHAT WE SAW: baseline → address burn (S_name 573, real burned-net
generation) → complete erasure (n=1 boxed) → cheap install beside the
expression collapse (p(Z) 0.556→1.7e-6 log bars; "0 × ZEPHYRA in 2,800
chars"; the installed net opens ELIZABETH) → the scar (groove cos 0.760
vs 0.278; anti-carrier flip; 44.5% key-resistance, n=1) → the four-box
law. The lab's second user-facing artifact.

## E048 — expression gap: teacher-forcing-bound, at every dose (2026-09-25) — DONE

WHAT WE DID: dose x3, seeding (induction route), temperature x3, greedy
diagnostic on the installed cell; battery + free-generation readouts at
every arm.

WHAT WE SAW (T019): expression = 0 in ALL arms while battery holds
0.92-0.96. P1 confirmed (teacher-forcing-bound); P2/P3 refuted (no
threshold, no dose response). C7 final: address / ability / expression /
history — four separable faculties; install-by-teacher-forcing is
constitutionally silent. Doctrine: continuation batteries are not evidence
of usable knowledge; free generation is the honesty check.

WHAT'S NEXT: night program continues (e049 retrieval dose-response;
e040/e032/e005s gated). Review due.

## E044 — scar tissue (REAL run): erasure burns the address, not the attractor (2026-09-25) — DONE

WHAT WE DID: full 5-arm battery (re-install vs fresh vs patch-controls,
400 steps each); root cause of the earlier shakedown = E044_SMOKE=1 env
leftover; new COS_MIN_NORM validity guard.

WHAT WE SAW (T018): re-learn is 2.08x SLOWER but the address re-grows
along its ORIGINAL direction (cos 0.760 vs fresh 0.278) — the attractor
survived erasure; the new route is new (atlas rho 0.21; L3H5 flips
carrier->ANTI-carrier, -2.03); the re-learned memory is ~3x more
surgical-RESISTANT (44.5% vs 0.13% under the same D2+patch). P3 failed at
bar (incumbents +0.135). JOHN improved (re-learning repaired J-class
collateral).

WHAT'S NEXT: C7 final: address/ability/expression/history. Night program:
e048 expression gap next.

## E047 — positive-claims replication sweep: card v3's gate (2026-09-24) — DONE

WHAT WE DID: 3 surviving positives × 5 nets (references near-bit-exact;
renorm liveness asserted; uniform-floor batteries).

WHAT WE SAW (T017 = card v3):
- **L5-calibrator SURVIVES 5/5** (KL 0.91-1.08, ablation ≤0.046) → first
  positive at H under the min-nets rule.
- **MLP-5 energy carrier SURVIVES 5/5** (zero/rotate 0.25-0.34, graceful
  α everywhere) → H.
- **Shared-L0 name machine DIES as stated** (2/5, seed-42 only) →
  distributional form: L0 BLOCK top-1 in 20/20 cells; sublayer allocation
  is a lineage lottery.

WHAT'S NEXT: card v3 declared (T017 preamble). Night program continues
(e048 expression gap next; e044 rerun in flight).

## E046 — C6 replication: two-factor erasure does NOT replicate (2026-09-24) — DONE

WHAT WE DID: the two-factor recipe on B43 + BDO (own-head and B's-recipe
cells, 4 total), full honesty battery (train + uniform-floor contexts,
J-census, incumbents, corpus CE).

WHAT WE SAW (T016): no Bar-2 anywhere — JULIET stays 13-17% after row-zero
+ top-head lesion on both nets; B's L3H5 transfers as predicted-NO. Row
surgery alone (the address half) replicates exactly (13-17% band at
~+0.001 CE, class-exact). C6 DEMOTED: general cheap DAMAGER; complete
erasure was B-specific luck.

WHAT'S NEXT: Review 6 (overdue) must weigh a replication sweep of the
card's positive claims vs new arcs — the pattern of
positive-claims-die/negative-claims-hold is now itself the biggest fact.

## E043 — install a name: ASYMMETRIC-CHEAP-REMOVE confirmed (2026-09-24) — DONE (audited from raw metrics)

WHAT WE DID: the registered install battery per scratch/e043_design.md —
rows-only arms (BDO same-init donor; wte/lm/both x copy/delta), exposure
ladder, L0-MLP block graft, direct-training ceiling; gates G0-G6.

WHAT WE SAW (T015):
- **No surgical install reaches Bar-I1 at the guard** (best: NLL 6.43 /
  acc 0.055 vs bar 4.17/0.5); Bar-I2 unreachable in every arm.
- **AMENDED by full report: anchored exposure installs CHEAPLY** — 7
  guarded cells reach Bar-I2; best 0.09 NLL / 0.974 acc at +0.05 CE,
  S_install 144-289 (the earlier decay read was a partial trajectory).
  New caveats: the EXPRESSION GAP (0 ZEPHYRA in 2,800 generated chars at
  97% battery acc) and PROTOCOL FRAGILITY (onset wall anchor-manufactured).
- **Shared machinery conserved** (L0H3 top-1, atlas 0.9997).
- Interpretation: address is concentrated (rows, removable); usage-ability
  is distributed (body, needs training). Edit asymmetry law (C7).

WHAT'S NEXT: e044 scar tissue tests the law's re-learning prediction.

## E012d — 4-net causal census: C1's strong form is dead (2026-09-24) — DONE

WHAT WE DID: the e018 causal-depth protocol on B43/R/R43 (B reproduced
exactly); cross-net histogram correlations + the lens=6 scoping check.

WHAT WE SAW (T014):
- **Causal depth is NOT cross-net invariant:** seed axis 0.735, regime
  axis 0.500 (B43-R43 collapse) vs the 0.8 bar; mode slides 3->4->4->5
  across B/B43/R/R43; renorm shifts mass deeper. The lens census's
  0.82-0.85 was the by-construction artifact.
- **T012's demotion total:** the lens=6 bin (52.7% of positions) selects
  causally indistinguishable positions.
- C1 final: qualitative mid-stack causal gate in every net; quantitative
  depth non-invariant. e005s readout re-scoped to the qualitative gate.

WHAT'S NEXT: card C1 updated. e043 still running; audit slot next when it
lands. Evening program continues (e044 scar gated).

## E042 — name-circuit atlas + two-factor erasure (2026-09-24) — DONE

WHAT WE DID: position-resolved lesion atlas (36 heads + 12 blocks) at name
positions for JULIET/JOHN/ROMEO/LUCIO; residual atlas under D2; two-factor
erasure cells. All 6 gates pass; e023 numbers reproduced exactly.

WHAT WE SAW (T013):
- **Two-factor erasure works:** D2 + L3H5@JULIET-prefix → acc 0.0013,
  NLL ≥ ln65, corpus +0.00083 nats, S_name 1,937. Complete selective
  forgetting achieved.
- **Shared name machinery:** L0H3 #1 head for ALL names; L0-MLP #1 block;
  JULIET~LUCIO atlas correlation 0.965. Collateral idiosyncrasy lives in
  row space, not circuits.
- **Dissociation:** post-D2 residual (13.6%, all at position 3) rides
  mid-network machinery (L3H5, L1-attn), NOT the healthy L0 circuit.

WHAT'S NEXT: C6 finalized. e043 (INSTALL a name) now cleanly defined:
rows + which body. Review ~15:50Z.

## E018 — causal depth: the lens is UNCORRELATED with causal depth (2026-09-24) — DONE

WHAT WE DID: activation-patching causal depth over 1536 positions
(counterfactual last-position stream spliced at each depth; sanity gates:
self-patch exact, post-L5 patch flips 100%).

WHAT WE SAW (verdict (b), T012 written):
- **Spearman(causal, lens) = −0.009** — per-position UNCORRELATED. The
  lens's depth ordering carries no causal-decision information.
- Causal mode depth 3 (mean 2.95 vs lens 4.86); the lens's 53% "decided at
  L5" mass has no causal counterpart — L5 flips are RECALIBRATION
  (convergent with L5-the-calibrator from every other instrument).
- Genuine point-of-no-return exists mid-stack (monotone flip curve, 77%
  suffix-monotone); 15.4% distributed decisions; shallow patches → third
  tokens (73% at d0), deep patches → the counterfactual answer.

WHAT'S NEXT: C1 re-anchored (T012); e012d debt registered (causal census on
the other 3 nets — is CAUSAL depth the invariant?). e042 still building.

## E023 — surgical forgetting at entity granularity (2026-09-24) — DONE (confirmed by full report 15:12Z)

WHAT WE DID: the full registered battery per scratch/e023_design.md — D1/D2
granularity ladder, arms A/B/C, G0-G4 gates, frozen selectivity metrics.

WHAT WE SAW (T011 written):
- **The J-row scalpel: S_name = 573 (bar 5) at corpus cost +0.0008 nats**
  — the program's first selective instrument, ~4907× less collateral than
  entity-ascent at matched damage. Bar-2 erasure missed narrowly (acc
  13.6% > 10%): surgery damages near-completely, does not fully erase.
- D1 all-letters bomb confirmed (+0.38 CE). P2 refuted (collateral-vs-
  overlap ρ=0.61; idiosyncratic per-name collateral). P3 confirmed (no
  revive trigger; ascent catastrophic at the name bar: val +1.16, S_name
  1.06). lm_head-row zero: NLL 2.04/acc 0.83 (write-side partial).
- Card consequence: C6 drafted (entity knowledge in I/O row coordinates;
  damage-vs-erase boundary open).

WHAT'S NEXT: e018 (causal depth — instrument validation) + e042 (name-
circuit atlas: the 13.6% residual + collateral idiosyncrasy) dispatched.

## E041 — ΔW ceiling null: PARTIAL ANCHORING (card C3 debt paid) (2026-09-24) — DONE

WHAT WE DID: trained BDO = seed-42 init, different data order (corpus seed
7777 changes every batch; init bitwise-verified identical); computed the
ΔW-alignment ceiling cos(ΔW_B, ΔW_BDO) with the e029 protocol.

WHAT WE SAW:
- **The full ladder: 1.0 (same everything) → 0.534 (same init, diff data
  order = CEILING) → 0.141-0.152 (same init, diff regime, B↔R) → −0.002
  (diff init).** Verdict: PARTIAL ANCHORING — the regime change moves a net
  well beyond batch-order noise (0.15 is only 0.26× the ceiling), yet
  same-init anchoring remains far above the diff-init floor.
- Per-organ ceiling: early organs order-robust (L0-attn 0.73, L0-mlp 0.79),
  depth erodes alignment (L5 0.34-0.46) — deep layers are where both order
  noise AND regime pressure act.
- BDO val 1.595 (parity PASS; batch order alone shifts final CE by −0.03 —
  data order is a real training variable). Motion magnitudes identical
  (‖ΔW‖ ratio 0.98-1.01) — only directions differ.

WHAT'S NEXT: card C3 updated (ceiling paid). e023 design memo in progress.
Review ~14:35Z.

## E035 + E038 — task-net anatomy + causal head lesion (2026-09-24) — DONE

E035 (eval-only on the e021 task net):
- **Q1: allocation NOT reorganized; lesion maps are blind to task circuits.**
  Attn damage [3.03, 2.04, 0.64, 0.45, 0.24, 0.03] vs Shakespeare [2.40,
  1.74, 1.06, 0.38, 0.19, 0.03] — no L4 spike (COPY is ~5% of tokens, so
  the whole L4-attn block costs +0.24 corpus nats while ONE head inside it
  costs +1.67 at COPY positions). Position-resolved instruments are
  mandatory for task-elicited circuits. MLP-0 keystone even larger (4.90).
- **Q2: ONE net holds TWO stage profiles.** JS(filler, Shakespeare) = 0.010
  (filler pipeline ≈ Shakespeare's; L5 61.9%) vs JS(filler, COPY) = 0.272
  (COPY at L4, 88.3%). Profiles are selected per-position, not a global
  rewiring. Control net's filler census also Shakespeare-like.
- **Q3: init-anchoring task-independent — slightly STRONGER than the B↔R
  band** (W_in 0.622 vs 0.531; W_out 0.313 vs 0.261). Growing a retrieval
  circuit did not pull the net off the shared init trajectory.

E038 (causal lesion of retrieval head L4-H1):
- **Registered collapse verdict: NOT-CAUSAL (42% acc drop, bar >50%) — the
  dedicated-head reading dies; retrieval is a redundant cooperative fan.**
  Boundary reading: zeroing one head of 36 takes COPY CE 0.007→1.681
  (+1.674 = 51.3% of the distance to chance) and accuracy 99.96%→58.0%
  (15× chance; residual uniform across nonce positions) with PERFECT
  locality (filler CE −0.0006; whole-corpus +0.056). L4-H1 is causally the
  single largest retrieval channel — about half the nonce information —
  the other half in a distributed backup. E011b's cooperative-fan doctrine,
  now at the retrieval layer. Control head: nothing moves.

WHAT'S NEXT: card v2 updates (C4 language, C1 two-profile refinement, C3
task-independence) folded next edit. Review due ~14:35Z.

## E003c + E019 — dose-to-bar and the MLP-5 thermostat (2026-09-24) — DONE (record corrected per full report)

E003c (exact projection dosed to the 0.66-nat forgetting bar; 2 seeds +
step-norm-matched naive control verified at ratio 0.99):
- **DOWNGRADE FIRES.** r at the bar = 1.43/1.39 (needed ≥2.0); matched
  naive = 1.23 — projection's entire reproducible margin is 1.17×. r(dose)
  declines monotonically 1.52 → 1.41 → ~1.15 (the earlier 'flat ≈1.7'
  quick-read was wrong; apparent r>1.5 recovery at high dose is ratios of
  destroyed-model CEs).
- **THE KILLER: r vs train-B at the bar = 1.08 ≈ naive.** A and B trained
  memories are forgotten at IDENTICAL rates — zero content selectivity in
  the memorization channel; the val_B 'selectivity' was a measurement-axis
  artifact (held-out fluency text is more robust than any trained text).
- **e003b's r≈5-6 head-start DOES NOT REPRODUCE** (agent re-ran e003b's own
  code+seed: peak 1.455 vs recorded 6.128; s280 target matches bit-for-bit)
  — a chaotic trajectory event, not an Adam-anchor mechanism. The
  'accidental hybrid' story is dead too.

E019 (MLP-5 thermostat): **energy carrier CONFIRMED** — zero +0.60 vs
rotate60 +0.15 (4×); α=0.5 slightly IMPROVES CE (−0.0065 — MLP-5 marginally
over-writes); α=2 graceful (+0.12); zeroing spikes output entropy +0.62.
Magnitude keeps the distribution sharp; direction worth ~¼ of presence.

WHAT'S NEXT: claim 5 closes as 'first-order ascent cannot content-
selectively forget (r=1.08 memorization-symmetric)'. Next family: weight
surgery (e023 entity-level embedding+lm_head rows — the genome-era
inheritance) or second-order. Card C5 updated.
## E003b — targeted/projected ascent: registered instruments FAIL; an Adam-anchor accident soars (2026-09-24) — DONE (record corrected)

WHAT WE DID: 300-step ascent arms with corrected labels (target = train-A
memorization CE, baseline 1.018 vs val_B 1.681 — a 0.66-nat gap; collateral
= val_B): naive anchors, exact projection (unit B-direction, 4-batch mean,
refresh/10), top-10% masked, combined, plus an accidental variant
(unnormalized B-direction = 70%-strength projection, kept for the record).

WHAT WE SAW (registered verdicts):
- **Naive r = 1.22 (NOT 1.0)** — under corrected labels even plain ascent
  is mildly selective; the old anti-selectivity constant was partly the
  val_a mislabeling.
- **Exact projection FAILS the 1.5 bar (peak 1.455, decays to 1.09)** —
  removing the full first-order B-component converges to the naive
  signature. Masked fails 2.0 (1.80); combined 1.93 (gentlest: +0.034
  collateral at +0.064 target).
- **The ACCIDENTAL 70%-projection soars (r 3.12-6.13, target +0.28 at
  collateral +0.09).** Mechanism (agent's reading): through Adam's
  sign-like updates, the residual B-component acts as a weak implicit
  B-DESCENT anchor — projection-before-Adam ≠ projection-of-the-step.
  This is E002's explicit retain-anchor, rediscovered implicitly at the
  right dose.
- Mean-level gradient cosine cos(g_A, g_B) = 0.799 vs batch-level 0.345 —
  averaging collapses both onto the shared fluency direction.

WHAT'S NEXT: e003d REGISTERED — deliberate partial projection + explicit
small retain-descent term (the accidental winner made explicit), dose-to-
the-0.66-bar, step-norm-matched naive control, train-B second collateral.
NOTE: the running e003c uses the EXACT projection — interpret its
projected arm knowing it is the failing variant.
## E031 — stream-facing matrix grafts: W_in is the violent one (2026-09-24) — DONE

WHAT WE DID: host B received one matrix at a time from B43 (cross-seed,
same regime) at L3/L5: W_in, W_out, c_attn, c_proj + full-organ references.
Registered v2 predictions (v009-corrected stream-basis mechanism).

WHAT WE SAW (P1 CONFIRMED, P2 REFUTED):
- **P1 CONFIRMED: both stream-facing MLP matrices are violent.** L3: W_in
  +1.186 + W_out +0.795 ≈ full-mlp +1.907 (near-additive). L5: W_in +3.284
  — MORE violent alone than the whole organ (+1.46): donor W_out partially
  RESCUES donor W_in (the pair is internally coherent; the host punishes a
  foreign read more than a foreign read+matching-write).
- **P2 REFUTED:** c_attn is the mildest at both sites (L3 +0.397 vs c_proj
  +0.506; L5 +0.044 vs +0.068) — the weak-anchoring functional exception is
  the attention QUERY/KEY side, not c_proj. Attention portability is
  carried by both its matrices being mild.
- The seed-anchored object is confirmed as the STREAM-FACING interface, with
  the read half (W_in) dominant.

WHAT'S NEXT: T008 claim-3 mechanism now causally supported. e003b targeted
ascent remains the last open instrument (claim 5). Review 13:26Z.

## V009 — ΔW portability atlas: reads are seed-anchored, writes converge (2026-09-24) — DONE

WHAT WE DID (VISUALIZER agent, CPU-only): top-16 singular subspaces of every
organ's ΔW across the 4 nets; same-seed vs diff-seed subspace alignment
(random baseline 0.289); sanity vs e029 exact. runs/v009/dw_atlas.png.

WHAT WE SAW:
- **Candidate mechanism REFUTED in reverse:** the same/diff-seed alignment
  gap is largest on the READ side (W_in 0.260, c_attn 0.235) and smallest
  for MLP W_out (0.091). Diff-seed alignment ≈ random for all reads
  (excess ~0.003); W_out keeps a small positive excess (+0.036).
- **AMENDED after full report: the seed-anchored object is the residual-
  STREAM basis.** MLP stream-writes (W_out-LEFT gap +0.267) are 3.8x more
  init-anchored than attention's (c_proj-left +0.071) — matching e029's
  transplant rho. Diff-seed alignment is at the random floor EVERYWHERE
  (excess <= +0.04): attention portability = weak anchoring, not shared
  subspaces. e031 re-registered: W_in and W_out grafts each VIOLENT
  (stream-facing); c_proj mildest.

## E021 — task-swap: retrieval exists when required; new L4 decision mode (2026-09-24) — DONE

WHAT WE DID (T009 registered design, background agent): retrieval-required
corpus (10.7k docs, ID→COPY gap ≥37 chars) + shuffled-nonce control; two
fresh nets (val: task 1.432, control 1.552); copy accuracy, far-value at
COPY, depth census at COPY, attention ID-mass. runs/e021/*.png.

WHAT WE SAW (all four registered predictions resolved):
- **P1 CONFIRMED: 100% copy accuracy** (2500 held-out nonce chars; chance
  3.8%; control net 4.1%). CE at COPY positions 0.007 nats — noiseless.
- **P2 CONFIRMED: far-value at COPY +3.269 nats ≈ ln 26** (control
  −0.002). The full nonce information is retrieved from far context.
  **T008 claim 4 NARROWS: "no retrieval on natural char data at this
  scale" — not an architectural limit.**
- **P3 CONFIRMED: a dedicated retrieval head.** L4-H1 puts 95.1% of its
  attention mass on the 5 ID nonce chars (control same head 15.0%);
  layer-mean ID-mass peaks L4 (0.364 vs 0.061); local mass collapses
  (0.038/0.042 vs Shakespeare L5 0.106).
- **P4 = NEW-MODE:** 88.3% of COPY decisions at L4 vs 8.0% on Shakespeare
  (JS divergence 0.265). A sharply concentrated task-dependent decision
  mode — at L4, one layer EARLIER than Shakespeare's L5-centered profile:
  retrieval completes before final calibration. The stage picture gains a
  task-dependent member; stages remain the organism (claim 1 intact).

WHAT'S NEXT: e031 write/read-path split grafts (v009-flipped prediction);
e003b targeted ascent still queued. Review ~13:26Z gets this full ledger.

## E030 debt slot — claims 1+2 upgraded to H; e011c CIs clean (2026-09-24) — DONE

WHAT WE DID: one eval-only slot on existing checkpoints (background agent,
21.7s compute after setup): e012c 4-net depth-census table; e014b.1 R43
lesion-map replication; e011c bootstrap CIs.

WHAT WE SAW:
- **Claim 1 (stages) RESTORED at H:** cross-seed same-regime histogram
  correlation (0.849 mean) ≥ cross-regime same-seed (0.828); all six pairs
  in 0.822-0.855; invariants replicate in both seed-43 nets (L5-finalization
  1027/1088; depth-entropy Spearman; class ordering). Stages are
  init-independent AND regime-independent.
- **Claim 2 (plastic anatomy) REPLICATED on R43:** keystone dissolution
  (+0.21 vs B +4.08), late-heavy MLP flip (ρ +0.74), attn front-load, third
  independent rebuild of the declining write schedule.
- **e011c exceptions all real:** attn-L0 1.38±0.006, MLP-L1 3.03±0.044,
  MLP-L5 0.25±0.009 — 20-100× beyond sd.

WHAT'S NEXT: T009/e021 (retrieval-required task) REGISTERED and running in
background — P1-P4 break-conditions for T008 claims 1 and 4.

## E029 — seed×regime 2×2 transplant + ΔW alignment: mechanism CONFIRMED (2026-09-24) — DONE

WHAT WE DID: trained R43 (seed-43 renorm, parity PASS val 1.5596), then the
full 2×2 matrix (54 cells, 3 hosts × 6 organs, paired bootstrap CIs, C0
bitwise gates clean) + the ΔW-alignment observable: cos(ΔW_donor, ΔW_host),
ΔW = W_trained − W_init(seed). runs/e029/*.png.

WHAT WE SAW:
- **Mechanism CONFIRMED decisively: same-init pairs mean cos(ΔW) = +0.152;
  different-init pairs ≈ 0.000 (max |cos| = 0.017 across 24 pairs).**
  Training motion from different inits lives in almost perfectly ORTHOGONAL
  parameter subspaces — organs refine init-anchored directions.
- **Seed dominance is ORGAN-TYPE SPECIFIC:** MLP organs show strong seed
  dominance (ρ = dCE(seed)/dCE(regime) 2.0-3.6 at L3/L5 across all hosts;
  e028's violent cell replicates exactly: +1.990 vs +0.646); attention
  organs are axis-insensitive (ρ 0.54-1.29 — portable either way). The one
  regime-dominant organ: R-host MLP-L0 (keystone asymmetry pinned to the
  regime axis, R=7.95 vs 1.49).
- Both-axes changes are SUB-additive (median 0.49) — the two interference
  modes overlap.
- All-cell median ρ 1.007 (L0 cells saturate at the ablation ceiling;
  ratio-of-medians 2.48) — the registered "seed dominates everywhere"
  prediction refines to "MLP organs are seed-anchored; attention organs are
  portable."

WHAT'S NEXT: T008 claim 3 upgraded + refined. Open: why are attention organs
portable across inits while MLP organs are not? (candidate: attention reads
stream directions shared by all adequate solutions; MLP writes into
seed-specific subspaces.)

## E028 — cross-anatomy transplant: P3 REFUTED reversed — organs are portable; incompatibility follows SEED (2026-09-24) — DONE

WHAT WE DID: implemented scratch/e028_transplant_design.md (background agent):
MLP/attn organ swaps at L0/L2/L3/L5; hosts: baseline-B (seed 42) with
donors B43 (same anatomy, seed 43) and R (renorm anatomy, SAME seed 42);
C0 bitwise self-transplant gate (exactly 0.0 ✓); renorm liveness ✓; R =
ΔCE_transplant/ΔCE_ablation. runs/e028/*.png.

WHAT WE SAW:
- **P3 REFUTED in reverse: cross-anatomy swaps cost LESS than same-anatomy
  different-seed swaps** (median ρ = cross/within = 0.874; ρ≥2 in 0/8;
  cross ≤ within in 6/8; paired CI excludes 0 negatively in 8/8).
  L3-mlp: within +1.99 nats vs cross +0.65 (ρ=0.32). **Organ compatibility
  tracks initialization lineage (B and R share seed 42) more than training
  regime** — the two anatomies differ in scheduling/addresses, not organ
  mechanics. Supports T006/PL2 (stages) over PL3.
- **S1 keystone asymmetry confirmed both ways:** B's keystone MLP-0 → R host
  = worst interference anywhere (R=7.95); R's near-dead MLP-0 → B host ≈
  inert (+4.20 ≈ B's own ablation 4.08) — quietness transfers even when
  function doesn't.
- **S3 failed informatively:** trained foreign tissue misleads MORE than
  random tissue (lottery R=1.14 vs cross R=7.95 at L0-mlp) — interference
  is content-specific, not off-manifold energy.
- Prefix 0..3 cross strongly SUBadditive (+2.83 vs 8.92 summed) — host
  layers compensate for whole foreign prefixes.

WHAT'S NEXT: e029 — the clean 2×2: seed(42/43) × regime(base/renorm)
transplant matrix to isolate the compatibility axis (init lineage vs
regime); T006 P4 (v008 phylogeny) gains a new question: do lesion maps
cluster by seed or by regime?

## E013d — interference audit: both T007 stories dead; far context acts through bulk statistics (2026-09-24) — DONE

WHAT WE DID: divergent-continuation n-gram proximity for loser positions
(P1); shuffled-far context collapse test (P2). Same 2000 positions as
E013c.

WHAT WE SAW:
- **P1 REFUTED (effect +0.22σ < 0.5):** 91% of hurt positions have NO
  divergent repeat (≥4 chars) in far context at all — the
  repetition-interference story (H1) is dead in its simple form.
- **P2 REFUTED, informatively:** shuffled-far makes the hurt population
  WORSE (bottom decile −2.26 vs −1.68) while the gain tail survives nearly
  intact (+1.50 vs +1.60). Real far gains are shuffle-ROBUST (statistical,
  not informational); incoherent far text destabilizes MORE than real far
  text.
- **Net conclusion (T007 closed): this 2.7M char model shows no evidence of
  SPECIFIC far-context information retrieval — far context acts through
  bulk statistics (char mix / length) and can destabilize predictions when
  incoherent.** Consistent with E013's finding that L5's far attention is
  idle grazing.

WHAT'S NEXT: closed. If long-range retrieval is wanted, it must be tested
on a task that provably requires it (copy-span probes, e021 task-swap).

## E013c — far-value tail: far context is a double-edged sword (2026-09-24) — DONE

WHAT WE DID: per-position far-value = CE(16-ctx) − CE(256-ctx) over 2000
held-out positions; distribution, tails, correlation with local difficulty;
top/bottom context examples.

WHAT WE SAW:
- **Bimodal, not average-zero:** 30.6% of positions gain ≥+0.15 nats (top
  decile +1.60, p99 +2.96); 28.2% LOSE ≥0.15 (bottom decile −1.68).
  "16-token sufficiency" hid a tug-of-war.
- P1, P2 confirmed; P3 refuted (ρ=0.133 — far-value tracks the position,
  not its local difficulty).
- Top gainers = locally-ambiguous rare continuations resolved by far context
  ("the carp"→T, "ere "→s). Losers = far context actively misleading
  (candidate mechanism: interference from earlier similar n-grams with
  different continuations — T007/H1).

WHAT'S NEXT: T007 discriminators — divergent-continuation n-gram proximity
for losers; shuffled-far context collapse test. e028 running in background.

## E013 — context truncation: L5's calibration is LOCAL; far context is worth ~0 (2026-09-24) — DONE

WHAT WE DID: same 300 held-out windows at full-256 vs last-16 tokens; measured
KL(L5‖L4 readout), L4→L5 argmax-flip rate, and per-position CE.

WHAT WE SAW:
- **Registered prediction REFUTED:** KL 0.997 → 0.928 (−6.9%, predicted
  ≥50%). L5's distribution reshaping does NOT depend on far context; the
  census's diffuse far attention is idle grazing, not evidence gathering.
  "Re-globalization" is epiphenomenal attention shape.
- **16-token sufficiency (the bigger finding):** CE full-256 1.648 vs
  trunc-16 1.644 — far context adds ≈ NOTHING to next-char prediction on
  Shakespeare at this scale. Depth ≠ range: late decisions (T004) are deep
  lexical computation, not long-range integration; D2 is refuted.
- Flip rate 49.3% → 52.0% (unchanged): L5's argmax work is local too.
- BUG (found + fixed): double-softmax CE (probs fed to F.cross_entropy)
  inflated the first run's CE to 3.73; verified E012 unaffected (its CEs
  came from forward logits).
- BONUS (from the debug check): the depth-CE readout ladder is
  anti-informative mid-stack — [4.62, 4.70, **5.19**, 3.99, 3.45, 2.50,
  1.74]: depth-2 readouts are WORSE than unigram (4.17). Absolute mid-stream
  distributions are not lens-aligned (needs a tuned lens); argmax-stability
  claims are order-robust and unaffected.

WHAT'S NEXT: far-value TAIL distribution (per-position full−trunc ΔCE): is
the ≈0 average uniform, or do a few positions (after rare names?) carry all
the far-context value? e028 transplant running in background.

## E013a — attention census over 200 prompts (2026-09-24) — DONE

WHAT WE DID: all 36 heads' last-token attention across 200 held-out
prompts: local/far mass, attended-token surprisal, distant concentration.
Adjudicates T005 (Review 1 required this before any causal spend).

WHAT WE SAW:
- **The locality funnel replicates at scale:** far-mass U-shape
  [L0 0.80, L1 0.23, L2 0.09, L3 0.10, L4 0.23, L5 0.54]; local-mass peaks
  at L2 (0.42). L5 abandons the local window in 82.5% of prompts.
- **The rare-token story is DEAD:** attended surprisal flat (~5 bits) at
  every layer; ZERO heads with consistent distant-concentration. v002's
  'O'-head was a one-prompt artifact — Review 1's suspicion confirmed.
- Final T005 form: L5 = diffuse re-globalizer. e013 redesigned as context
  truncation (does L5's calibration KL depend on far context?).

WHAT'S NEXT: e013-redesigned (truncation, minutes); e028 transplant (design
memo ready at scratch/e028_transplant_design.md — T006 P3); e019 thermostat.

## E012b — renorm-anatomy census: the function persists (2026-09-24) — DONE

WHAT WE DID: reran the decision-depth census + angular-displacement profile
on the E014b renorm checkpoint (hooks active). T006 P1+P2 discriminators.

WHAT WE SAW:
- **P2 CONFIRMED (histogram corr 0.822):** two different anatomies, one
  functional profile — L5-finalization 1084/2000 vs 1082/2000, same L1 dip,
  same depth↔entropy Spearman (+0.344 vs +0.322). Mid-stack timing
  reshuffled (baseline spreads early; renorm concentrates L3-L4) but the
  pipeline shape held.
- **P1 near-miss (0.386 vs ≤0.33 threshold), direction strong:** block-0
  angular displacement 0.746→0.288; every layer's angular displacement
  roughly halved in the renorm anatomy.
- Adopted as lab doctrine: decision depth / locality / calibration KL are
  the primary anatomy instruments (invariants); lesion maps are the
  secondary "where do the stages live this time" instrument.

WHAT'S NEXT: e028 transplant (P3 — design memo in progress in background);
e013a census; e019 MLP-5 thermostat.

## V002 — attention atlas: L5 is a rare-token re-globalizer (2026-09-24) — DONE

WHAT WE DID: VISUALIZER subagent built lab/v002_attention_atlas.py: last-token
attention of all 36 heads on a 96-token dialogue window; per-head entropy,
distance profiles, attended-token surprisal; attention recomputation sanity-
checked against the model's own output (4.8e-7). runs/v002/*.png.

WHAT WE SAW:
- **Locality funnel across depth:** attention entropy L0 4.51 (near-uniform)
  → L3 1.64 (tightest; 50.7% mass at distance 4-16) → L4/L5 re-broaden.
  Matches E012's decision-depth structure (local completion peaks mid-stack).
- **L5 reads rare identity tokens far away:** local d1-3 mass collapses
  0.234→0.060 (L4→L5) while ancient d65+ jumps 0.001→0.091 (~76×);
  attended-token surprisal 4.28→4.81 bits at flat entropy. L5 head 1 puts
  0.499+0.146+0.085 ≈ 0.73 of its mass on exact 'O' character matches.
- This is the L5-calibrator mechanism: distant rare evidence reshapes the
  distribution tail (KL ~1 nat) while local argmax was already settled
  (ablation +0.03).

WHAT'S NEXT: causal test (e013): mask exactly the rare tokens L5 heads
attend to (positions recorded in runs/v002/metrics.json) — predict KL(L5‖L4)
collapses while mean CE barely moves. See THINKING T005.

## E014b — stream-renorm training: the decisive authority-schedule test (2026-09-24) — DONE

WHAT WE DID: implemented scratch/e014b_design.md exactly — renorm arm pins
every block-input stream to c=5.6 per token (hooks active train+eval), seed
42, same budget; baseline arm reuses E001 weights; eval-only-renorm control;
write/stream instrumentation; P2 criteria operationalized.

WHAT WE SAW (FINAL):
- **P2 REFUTED at full parity.** Renorm arm val 1.610 (BETTER than baseline
  1.622; gate 1.7224). Lesion map stayed front-loaded: attn damage
  [2.80, 1.69, 0.48, 0.81, 0.21, 0.03] (spread 2.77 vs baseline 2.37).
  Front-loading is functional allocation, NOT stream-geometry.
- **Anatomy is plastic:** baseline MLP-0 keystone (+4.08) dissolved (+0.10)
  under renorm; attn-L0 grew MORE critical (+2.80) with 2.9× larger write
  (7.8 vs 2.7); renorm MLP damage flipped late-heavy [0.10, 0.08, 0.23,
  0.45, 0.78, 0.70]. Multiple anatomies reach the same function.
- **Optimization declines late authority:** renorm caps the stream at
  consumption points, not write size — L4/L5 COULD write big but deflated
  (ρ 0.82, 0.70) while L1-L3 partially re-inflated (ρ 1.25-1.47; mean 1.10
  < 1.3 criterion → not met).
- **Norm profile is per-net load-bearing but task-optional:** eval-only
  renorm on baseline +3.56 nats, yet renorm-trained learning is unimpaired.

WHAT'S NEXT: T003 final resolution written (THINKING.md). Live questions:
why do early layers hold the coarse work? Why do late MLPs write big-but-
cheap (now LATE-heavy in renorm arm — the arrangement flipped)?

## V006 — decision-depth passage map (2026-09-24) — DONE

WHAT WE DID: colored 480 chars of held-out text by decision depth
(runs/v006/depth_passage.png); profiled depth by character class.

WHAT WE SAW:
- **Letters decided LATE (uppercase 5.32, lowercase 5.15); structural chars
  EARLY (newline 3.69, space 3.82); punct mid (4.52).** Depth tracks the
  TYPE of discrimination — structural/syntactic decisions finish early,
  lexical identity needs the deep pipeline (and L5's calibration).
- 51% of positions finalize only at the L5 readout — consistent with E012's
  L5-calibrator finding (it often settles the final argmax).
- Zero positions decided at emb in this passage; top-1 acc 0.61.

WHAT'S NEXT: depth clustering by position-in-line/speaker-turn; consider
depth as a manipulable surface (can we force a position to decide late/early
by context surgery?).

## E012 — decision-depth census (2026-09-24) — DONE

WHAT WE DID: over 2000 held-out positions, decoded the last-token stream
(ln_f+lm_head lens) at every depth; decision depth = shallowest depth whose
top-1 survives to the end. Correlated depth with final entropy (P1); split
positions early(≤2)/late(≥4) and measured per-position ΔCE under joint
attn-L4+L5 zero-ablation (P2); measured KL between L5 and L4 readout
distributions (P3). runs/e012/decision_depth.png.

WHAT WE SAW (2 confirmed, 1 refuted):
- **P2 CONFIRMED — decision depth predicts per-position vulnerability:**
  late-decided positions suffer 3.04× more ablation damage than early-decided
  (ΔCE 0.328 vs 0.108). Per-token anatomy is real; corpus-averaged lesion
  maps hide it.
- **P3 CONFIRMED strongly — L5 is a calibrator, not vestigial:** KL between
  L5 and L4 readout distributions averages 1.03 nats (median 0.70), yet
  zeroing L5 costs only +0.03 mean CE. L5 reshapes the output distribution
  massively in ways argmax and mean-CE cannot see. (Caveat: mid-network lens
  is heuristic.)
- **P1 REFUTED with sign flip:** Spearman(depth, entropy) = +0.32, not
  ≤ −0.4. Late decisions associate with HIGHER entropy — open contexts need
  deeper integration; constrained positions are decided at emb/L0. D1 was
  backwards.
- All 2000 positions eventually stable (no chronic wobble).

WHAT'S NEXT: viz — color a passage by per-token decision depth (v006): does
depth cluster structurally (names? line-ends? dialogue turns)? Think — if
L5 calibrates, what does it calibrate TOWARD (temperature? rare-token boost?
top-k shape?)? e014b design memo incoming from background agent.

## E011c — matched-perturbation control: geometry vs meaning (2026-09-24) — DONE

WHAT WE DID: replaced each attention/MLP write w with a 60°-rotated w′
(‖w′−w‖ = ‖w‖ exactly, verified 1.0000) — same perturbation energy as
zeroing, destroyed content. Three-rung ladder: zero vs rotate60 vs
same-norm-random (√2 energy). runs/e011c/geometry_ladder.png.

WHAT WE SAW:
- **P1 refuted both ways; energy is the dominant factor.** Damage ranks
  {zero ≈ rotate} < {random} across components — at matched energy, most
  blocks tolerate scrambled writes as well as or better than removal.
  The lesion map is mostly about HOW MUCH a block moves the stream
  (authority schedule), not what it says.
- **Exceptions:** MLP-L1 direction-sensitive (rotate ×2.95 zero); attn-L0
  mildly (×1.38); MLP-L5 is an ENERGY CARRIER (zero +0.59 vs rotate +0.14 —
  its magnitude matters, its direction barely).
- Caveat: single-run ratios 0.7–1.2 need bootstrap CIs; only the three
  exceptions look safely beyond noise.

WHAT'S NEXT: bootstrap CIs on rotate/zero ratios (cheap); e014b
stream-renorm training stays the decisive test of the authority schedule;
e012 decision-depth census queued.

## V001 — token journey visualization (2026-09-24) — DONE

WHAT WE DID: first artifact of the standing VISUALIZER thread: one forward
pass per prompt, three panels — logit lens through depth (final LN+head
applied to the last token's stream after emb and each block), PCA-2D token
trajectory with write arrows, per-layer angular authority (1−cos and
write/stream). runs/v001/token_journey.png.

WHAT WE SAW:
- **New observable: decision depth.** "…torches to burn " is decided at L3
  (top-1 `t`, p 0.69) and L5 HALVES its confidence (0.69→0.33); "To be, or
  not to " only surfaces the correct `b` at L4. Predictions form at
  different depths per token — T004 written with hypotheses D1-D3 and three
  registered predictions.
- Authority panel makes T003 visible: L0 write/stream ≈ 8.5 vs ≈ 1 later;
  angular displacement 0.7 (L0) vs 0.2-0.3 (later).

WHAT'S NEXT: e012 redesigned → measure decision depth over ~2000 val
positions; correlate with next-token entropy and L4/L5 ablation damage
(T004 discriminators). Viz polish backlog: arrowheads on trajectory, L0 bar
headroom in panel 3 (v001.1).

## E011b — L0 redundancy sweep + orthogonal-innovation control (2026-09-24) — DONE

WHAT WE DID: eval-only discriminators from the critique harvest: (1) all 63
L0 head-subset lesions; (2) same-norm random replacements of every attention/
MLP write (orthogonal-innovation control); (3) residual-stream norm profile at
block inputs. runs/e011b/redundancy_ortho.png.

WHAT WE SAW:
- **L0 heads are a cooperative ensemble with graceful degradation:** singles
  mean +0.05 nats (one slightly negative), yet all-6 = +2.40. Sum of singles
  0.317 vs joint 2.403 = 7.6× superadditivity. Keeping 1 of 6 heads still
  leaves 94% of full-ablation damage. Hook implementation validated exactly
  (all-6 head-zero 2.403 ≈ block-zero 2.401).
- **Same-norm random writes hurt more than zeroing** everywhere — attn L0
  +3.66 vs +2.40; MLP L1 0.82 vs 0.15 (5.4×). Ratios exceed the √2
  perturbation-scale prediction for MLPs and late attention → downstream is
  calibrated to write direction, not just magnitude.
- **Stream norm:** 0.67 at block 0 → 8.4× jump → plateau ~5.5. Write/stream
  ratio falls ~12× L0→L5.
- Emerging mechanism (see THINKING T003): the residual stream's norm growth
  may SCHEDULE each block's angular authority — front-loaded lesion maps
  could be partly architecture, not learning.

WHAT'S NEXT: e011c matched-perturbation control (60°-rotated writes) settles
geometry-vs-content per component; e014b stream-renorm training tests the
authority-schedule hypothesis directly. e003b (corrected ascent instruments)
still queued.

## E003 — forgetting selectivity frontier (2026-09-24) — DONE

WHAT WE DID: T002 discriminator suite: (1) gradient cosines (A↔B vs within-half
vs A↔French), (2) LR ascent sweep 1e-6…3e-5 with (ΔA,ΔB) trajectories, (3)
implant-French (400 steps, 5e-4) then ascend-on-French arm, (4) fluency-vs-
content CE probes (A-unique/B-unique lines vs generic) after mild ascent.
runs/e003/selectivity_frontier.png.

WHAT WE SAW (all three registered predictions resolved):
- **P1 REFUTED:** cos(A,B) = 0.345 ≈ within-half baselines (0.363/0.390);
  French is 2.4× lower (0.144). Same-corpus halves are NOT gradient-parallel.
- **P2 CONFIRMED:** no LR reaches ΔA ≥ 1 with ΔB ≤ 0.1. Even lr 1e-6 (the
  gentle walk along grad_A itself) gives ΔA +0.29 / ΔB +0.27 at step 200 —
  perfectly anti-selective.
- **P3 REFUTED:** unlearning French also destroyed Shakespeare
  (dissimilar_selective = false). Content distance does not rescue ascent.
- Fluency probes: after "mild" ascent (1e-5×200) everything ≈ 24 nats
  (random): A-unique 1.26→24.2, B-unique 1.35→24.0, generic 1.63→23.7.
- **Unified story forming:** first-order ascent — any dose, any content —
  destroys the shared fluency substrate first and content memories only with
  it. Gradient geometry (P1) shows content IS distinguishable in gradient
  space; the failure is that ascent trajectories do not follow it.

WHAT'S NEXT: T002 final resolution (thinking, not running): read the early
trajectory steps — was there ANY transient selectivity window (French rising
faster in the first 25 steps) before collapse? If yes: early-stopped ascent +
fluency-anchored objective is the repair hypothesis. If no: first-order
methods are structurally dead here; next family = weight-targeted surgery
(ascend only low-overlap weights) or second-order directions. Also reconcile
with the independent critique (scratch/critique_T001_T002.md) when it lands.

## E011a — write norms vs lesion damage (2026-09-24) — DONE

WHAT WE DID: measured mean residual write norms (per token) of every
attention/MLP block on val batches; compared to E001 lesion damage (T001
discriminator 1, zero training).

WHAT WE SAW:
- **H4 (write-norm confound) REFUTED.** Attention write norms are NOT
  monotone in depth ([2.72, 2.49, 3.31, 2.73, 2.39, 1.93] — layer 2 writes
  the most), yet damage still falls monotonically. Damage per unit write:
  attn [0.88, 0.70, 0.32, 0.14, 0.08, 0.02] — an 11× efficiency gradient.
  The front-loading is information architecture, not geometry.
- MLP write norms RISE with depth (1.82 → 5.64); MLP-5 writes the largest
  residual in the net yet ablation costs only +0.59 nats. Late MLPs write
  large, dispensable content — new open anomaly (for whom/what is it
  writing?). MLP-0 damage-per-write (0.95) is 5-10× any other MLP.
- Registered prediction "write norms will not decline monotonically" was
  CONFIRMED — first register-then-run success of the discipline.

WHAT'S NEXT: T001's remaining discriminators: mean-replace ablations and
LN-only recalibration (H2, off-manifold artifact) — e011 proper. And the new
anomaly: what does MLP-5 write? (logit-lens on its output direction space).

---

## 2026-09-24 — The pivot (context entry)

After three programs (neural genome transplants, LLM control surfaces,
HANDLE), we got too ambitious and drifted from the original spirit: da
Vinci-style dissection of neural networks for its own sake. Today the repo was
reset to a clean lab (tombstone commit `106aeff` preserves everything prior)
and the mission narrowed to the original one: small nets (1–10M params), many
experiments, curiosity first, graphs for everything, play.

The three questions opening the program: (1) what does a lesion map of a
freshly trained tiny transformer look like? (2) can we make a network forget
one part of its training data without wrecking the rest? (3) everything after
that is whatever the first two cuts turn up.

---

## E002 — forgetting pilot: naive vs anchored unlearning (2026-09-24) — DONE

WHAT WE DID: split the corpus positionally (A = first half of Shakespeare, B =
second half). From the E001 checkpoint, unlearned A with two arms: (1) naive
gradient ascent on A only; (2) anchored ascent (retain loss on B, weight 1.0).
AdamW lr 2e-5, 400 steps, batch 32×256. Tracked held-out CE on A and B;
generation probes before/after (runs/e002/probes.txt).

WHAT WE SAW:
- **Naive ascent is a bomb, not a scalpel.** ΔA +26.1, ΔB +25.9 nats — the
  entire model is destroyed (final CE ≈ random guessing over 65 chars).
- **Anti-selectivity:** to raise A by just +1 nat, B had ALREADY risen +1.51
  nats. Collateral damage runs ahead of the target damage.
- **The retain anchor only slows the destruction** (ΔA +6.4, ΔB +5.5 at equal
  steps; B damage at A+1 nat still +1.04). It does not create selectivity at
  this dose.
- Honesty reflex: the intervention changes behavior (massively) but with zero
  targeting value at this scale of step size.

WHAT'S NEXT: map the *selectivity frontier*: sweep ascent LR × steps × retain
weight; unlearn a small subset (one play) instead of half the corpus; try
Fisher/EWC-style parameter anchor instead of replay anchor; try weight-space
surgery (rank parameters by gradient overlap between A and B). → e003.

## E001 — first cut: lesion map (2026-09-24) — DONE

WHAT WE DID: trained a 2.74M-param char GPT (6 layers, 6 heads, 192 dim,
block 256) on Tiny Shakespeare (val loss 1.622 after ~2000 steps / 200 s; val
bottomed ~1.55 at step 1500 then overfit slightly). Then zeroed every
attention block, every MLP block, and each of the 36 heads individually;
measured deterministic val-loss delta per lesion (runs/e001/lesion_map.png).

WHAT WE SAW:
- **MLP-0 is the keystone organ:** zeroing it costs +4.08 nats — the single
  most damaging lesion, worse than any attention ablation.
- **Attention is strictly front-loaded:** L0 +2.40, L1 +1.74, L2 +1.06,
  L3 +0.38, L4 +0.19, **L5 +0.03 — the last attention layer is almost dead
  weight** in this model.
- **MLP damage grows with depth** after L0: L1 +0.15 → L4 +0.60, L5 +0.59.
- **16/48 components are ~dispensable** (Δ<0.02) — mostly late-layer heads.
- Net shape: early attention + layer-0 MLP carry the load; the top of the net
  is attention-light, MLP-heavy, and partly vestigial.

WHAT'S NEXT: what does MLP-0 actually store (char/unigram statistics? test by
probing / ablate-then-finetune recovery cost)? Why is attn-5 dispensable —
vestigial or quietly specialized (punctuation/newline)? Damage ≠ necessity:
how cheaply can the net re-learn around a lesion (zero-finetune recovery)?
Does this shape hold at 1M/10M/30M params (→ e004 ladder)?
