# R56 CRITIC — the three law-grade claims, attacked (2026-09-29 ~21:30Z)

Critic cells run for this review (marked **[C56]**): eval-only, CPU, organ reused
from disk (`runs/checkpoints/g3_gen.pt` + `g3_gen_s1.pt`), root g0 reproduced
0.8885950 exactly; fresh RNG seeds 22201-22203 / 22301-22303 (no collisions with
any registered block); no training, no repo files touched. No bar was shopped:
these are extensions past the registered grids, reported as measurements.

---

## 1. THE WALL (g1b + g1bR) — a splint that protects the ruler, at a price measured only on the ruler's terms

**What is replicated:** wash-draw seeds only, n=3, ONE root, ONE lineage, ONE
fact. W1(R=0.7) holds light g-12 at ~0.9 (mins 0.746/0.777/0.803) through +300
while same-draw controls die at +2..+4. Mechanically beautiful (bit-identical
roots, md5'd streams, G-PIN, G-STEP1 inert).

**Attack 1a — the splint objection is correct as stated, and the data prove it.**
The wall pins ALL 2,739,072 parameters in a 0.7-L2 ball around the commit;
D_kill ≈ 2.49. "Memory is made architectural against displacement" is close to
"displacement was made impossible." The non-trivial residue is two numbers:
(i) 0.7 L2 of slack leaves the native stream intact (ce_r 1.646 → 1.684 at
+300, no degradation); (ii) no second clock at pin (FLAT-AT-PIN 0.0026). Both
are about the RULER. That is all the wall has been shown to protect — see 1b.

**Attack 1b — the secret degradation is already in the metrics, unremarked.**
The wall's own anatomy battery shows channels of the SAME fact collapsing
inside the ball: the wpe band readout (band121_129) reads ~0.001 from +2
onward in W1 (vs the registered cost "band content=True" — NOTES reports "band
content lost" as a cost and moves on); g1bR's W1 old-band row0_strength decays
0.762 (+2) → 0.615 (+50); base_readout 0.81 → 0.66. The root's band value is
NOT in metrics.json — the one comparison the claim needs (how much band was
there at commit?) was never stored. As measured, the wall holds the light
probe flat while the underlying address anatomy partially collapses at
displacements ≪ R. The honest reflex (README 7b) has been applied to the
intervention but never to the READOUT SIDE: nothing in the g1-series reads
anything but the ruler family and CE. **No generation/free-run expression
read exists for the walled fact anywhere in the arc.** A wall that preserves a
teacher-forced probe while free-run expression degrades would still read
"WALL-HOLDS" at every bar.

**Attack 1c — the +0.53-nat tax is understated and mis-framed.** The
registered tax (W1 wash-CE@300 1.150 vs C 0.624) is measured against a control
adapted for 300 steps. Read against root (ce_r 1.66): the free net banks 1.04
nats of adaptation on the new stream; the walled net banks ~0.51 and its
root-stream CE drifts slightly WORSE (1.684). The wall forfeits ~half of all
future learning on the easiest possible stream, permanently — the tax is a
standing interest payment, not a one-time fee. Whether that is "acceptable" is
not the lab's call yet, because the tax's WORST case has never been priced:
**sequential memory.** Can the walled net install a SECOND fact? If not, the
wall is not a memory architecture; it is a museum with one exhibit, and the
paper's central positive is a frozen net that keeps its probe.

**KILLER CONTROL (cheap, <30 min, machinery exists verbatim):**
`g1bW-second-fact` — W1 machinery; after commit + 50 wash steps, install a
second fact (fresh host windows + fresh name token, e043-Dmix install
verbatim, 300 steps) UNDER the projection. Read A-ruler, B-ruler, CE_r, and
free-run completions of BOTH facts. Bars: SPLINT-REFUTED = B installs (≥0.7)
with A ≥0.5 (a real memory architecture); MUSEUM = B fails (≤0.27) at healthy
CE (the wall is a splint — rescope the claim to "the wall freezes the
organism; memory survives freezing"); ZERO-SUM = B installs and A dies
(displacement inside the ball kills — the ball is a single-exhibit museum by
geometry). Also free, zero compute: free-run expression battery on the
EXISTING W1_10907_s300 checkpoint.

---

## 2. THE RHYTHM (g2b/c/d/e; g2f in flight) — a threshold, a refractory, and a constant-threat wash do not make a law

**What is replicated:** wash-draw seeds n=3 on ONE root (g2d), timing 2/2 on a
fresh sibling root (g2e). The waveform (medians 0.587-0.615, duty ~70%,
spacing 24-36) is real on the locked root.

**Attack 2a — the timing band is partially true by construction.** The organ
is: monitor = mean p(Z|ctx) over 8 onset positions; gate opens below
THETA_OPEN=0.5, checked on a 4-step grid; REFRACTORY=24 hard lockout; event =
one error-replay batch. So spacing ∈ {24, 28, 32, ...} mechanically, and the
registered band is [20, 45]: **the lower edge sits BELOW the refractory.**
"100% of spacings in band" is guaranteed from below; only the upper edge (time
for the wash to push the monitor under 0.5) is a measurement. The celebrated
"~1/30 ≈ the external 1/32" is not a coincidence to marvel at — it is
refractory 24 + O(10) steps of decay-to-threshold at ONE wash intensity,
which was never varied in any g2 run. At constant threat this is a fixed-
period oscillator built from a thermostat; "self-timed" has not been shown to
mean anything beyond "threshold + cooldown."

**Attack 2b — the control that would license the noun has never been run
inside the replicated cells.** The fixed-period schedule MATCHES OR BEATS the
organ on every number the arc quotes: g2's SCHED r=1/32 maintained 0.693@+300
(n=1, family 2) vs the organ's cycle-medians 0.587-0.615 with mid-cycle
troughs to 0.08-0.28 — same event budget, higher floor, no monitor, no cue
pool. The organ's only possible virtue — firing earlier under stronger drift,
later (cheaper) under weaker — is precisely the axis no g2 cell varies. Until
a wash-intensity ladder shows the organ's period ADAPT and beat a matched
schedule somewhere, "THE RHYTHM" is "a self-triggered timer that replicates a
fixed schedule at constant threat," an engineering demo, not a law-grade
architectural claim.

**Attack 2c — the frozen argmax ruler already bit once, and g2f is about to
adjudicate on it again.** g2e: the fresh root's argmax moved to g+12; the
frozen g-12 rule read median 0.388 while g0 pooling reads 0.563 — a Rule 12
instrument-geometry failure, recorded but not fixed: g2f's registered bars
still adjudicate the amplitude on the frozen ruler. The root-strength gate
itself is a coin flip (draws 0.591/0.684/0.711 vs the 0.7 bar), so one more
base draw (g2f) adds nearly zero evidence about generality: strong-root-
strong-rhythm does NOT widen the claim beyond "the organ works on roots that
clear a gate that coin-flips" — the conditioning is doing the work, not the
architecture. The gate failure mode (fresh roots at 0.59) should be treated as
the primary object: either consolidate to a strength bar the recipe meets
reliably, or run 4-6 root draws and report root-strength-conditioned
medians.

**KILLER CONTROL (cheap, ~30-60 min CPU; trim to 4 cells for <30):**
`g2x-threat-ladder` — the g2d cell at wash lr {5e-4, 2e-3} x {organ,
SCHED r=1/32, SCHED matched-cumulative-budget} on the locked root, one seed.
Discriminating reads: (i) organ spacing vs lr (adaptation?); (ii) organ vs
schedule on cycle-median AND min-trough at each lr. If organ ≈ schedule
everywhere → rescope the noun to engineering; if the organ dominates at 2e-3
→ the claim earns "self-timed" for the first time. Free fix for g2f before it
lands: pool the ruler over geometries (g0/g-12/g+12 + per-root argmax
re-derivation) — costs nothing, prevents the third single-geometry misread.

---

## 3. THE CONE (g3 + g3R) — the dissociation is real; the instrument that
licensed it could not have failed; the cone is wider than claimed and the L2
metric is the wrong ruler

**What is replicated:** wash/noise draw seeds n=3 on ONE constructed organ
(n=1 construction): at 2x store-displacement the wash direction reads
0.10-0.16 while matched-L2 isotropic reads 0.85-0.89.

**Attack 3a — the registered ISO-SPARES bracket was structurally unfireable.**
[C56] I extended the isotropic ladder past the registered {2,4}x grid on the
same organ, same construct (3 fresh gaussian draws per level):

| iso level (of Dstore(t*)=0.1318) | 4x | 8x | 16x | 32x | 64x |
|---|---|---|---|---|---|
| g0 (3 draws) | 0.87-0.89 | 0.81-0.86 | 0.60-0.63 | **0.03-0.15** | 0.0006-0.02 |

**Isotropic noise KILLS — at ~24-32x the wash displacement (L2 ≈ 4.2 ≈ 55% of
the store's own norm).** The registered grid stopped at 4x, i.e., 6-16x BELOW
the isotropic kill level. "Matched-L2 isotropic spares through 4x" was
guaranteed by high-dimensional geometry before the first draw: [C56] the
gradient of g0 w.r.t. the store has ||∇g0||=5.44 and cos(∇g0, wash)=-0.44, so
the first-order wash effect at 1x is -0.32 (measured -0.65 — superlinear, the
retrieval cliff) while the iso first-order RMS at 1x is 0.005 — a predicted
~58x anisotropy, the same order as the measured 25-32x kill ratio. Most of
the licensed "dissociation" is concentration of measure, exactly the
instrument artifact the mandate asked about. The registered effect-size
sentence ("~0.7 g0 gap at identical L2") is arithmetically true and
rhetorically rigged: at the iso kill level the gap is ~0 in the other
direction.

**Attack 3b — and yet the cone noun survives, in amended form — by my own
falsifier.** [C56] Tilt ladder at FIXED L2 = 2x (0.2637), direction
cos(θ)·ŵ + sin(θ)·û (3 random û per angle): θ ≤ 20° → g0 ≈ 0.003-0.005;
45° → 0.04-0.06 (still under the 0.27 kill bar); 89° → 0.87-0.90. So the kill
region is NOT a ray: it is a cone with half-angle between ~45° and ~89°
around the wash direction. A random direction sits at 90° ± 0.7° in 17,408
dims, i.e., just outside it. The honest object is: **the fragility is a
low-measure, wide-angle cone that AdamW on real data enters on the first
step; the L2 ball is the wrong metric; "spares at 4x" is a geometry statement
about random directions, not a basin width.** Note also the leg's own
weakness: in g3R the replicates' store-only wash displacement at 1x reads
0.597/0.564 — ABOVE the kill bar in 2/3 draws (g3's 1x kill was the draw
artifact the run itself concedes); the store-only cone edge lives at 2x, and
the +1 full-wash kill is host-dominated (g5: both kill sites in the host).
The "forgetting is DIRECTION" slogan is really a statement about the composed
system; the store's own cone is the weaker half of the evidence.

**KILLER CONTROL (cheap, <30 min, eval-only — half already run above):**
absorb [C56]'s iso-extension and tilt ladders as g3R's effect-size cells; add
subspace-random directions (random vectors in the span of {the three seeds'
wash directions + ∇g0 at root and s1}) at matched L2 {1,2}x. If subspace-
random kills like the wash (likely — cos(∇g0,wash) is only 0.44, so the cone
axis is not the gradient and not the wash ray), the claim collapses one more
step to "the function lives in a low-dim sensitive subspace; the wash moves
inside it" — which is the arc's OLD address/content split wearing a cone
costume. Also worth one cell: pure −∇g0 at 0.13 L2 — if the steepest descent
direction does NOT kill, the cone's axis is something genuinely interesting.

---

## 4. CROSS-CLAIM — three mechanisms that cannot share a net, priced as alternatives

**Wall vs rhythm: substitution, not composition.** The wall's W1 floor
(g-12 ≥ 0.746 at every checkpoint) sits far above the gate's opening
condition (monitor < 0.5): inside the wall the organ's gate never opens — the
rhythm is dormant precisely where the wall works. Conversely the rhythm's
evidence (resurrections from 0.03-0.08 troughs) exists only because there is
no wall. They solve the SAME problem at opposite prices: the wall charges
~0.53+ nats of permanent plasticity and protects everything; the organ
charges ~30% of time below the bar plus a monitor wired to ONE fact's cue
pool. The paper must present a cost-frontier (min-expression vs nats
forfeited vs % time below bar vs parameters committed), not three
"architectural laws."

**Wall vs cone: the cone predicts a better wall nobody built.** The wall is
an isotropic L2 ball; the cone says only a low-measure direction set kills.
If the cone is real, a DIRECTIONAL wall (project updates out of the fragile
subspace only) should protect at a fraction of the 0.53-nat tax. g5 says the
cheap versions fail (W_q-only and store-only walls leave both host kill
sites live), but the gradient-subspace/PC-subspace wall was never tried. This
is THE discriminating experiment for the pair: directional-wall-works → the
claims compose into a design law; directional-wall-fails → the cone only
tells you WHAT kills, and protection necessarily costs the whole organism —
which would make "memory is made architectural" sound much less like
architecture and much more like amputation.

**Rhythm vs cone: compatible (replay = cheap cone re-entry, e179), but the
monitor reads the probe, and probes are exactly what the wall holds flat.**

**The three claims have never shared a substrate:** wall on the 2.74M line,
rhythm on the 0.873M family-2 line, cone on the 0.89M g3 organ's 17.4k store.
Three existence proofs on three different organisms, each n=1 root/organ
underneath its n=3 draw replication. The one composition attempt (g5) had its
cheap composition falsified by the run itself.

**META — the minting standard drifted.** R55 demanded "seeds ≥3" and the
g-claims met it with draw seeds on ONE net each. The lab's own meta-law
(README: "mechanism claims enter the card at H only after ≥3 nets") sets a
different bar that none of the three meets. Either the meta-law governs
mechanism claims or it does not; currently the g-series is minted on the
weaker of the lab's two standards. And every claim adjudicates on ONE
installed fact object (the e043 ZEPH lineage) — no wall, organ, or cone has
ever been asked to hold TWO facts in one net (e174's cohabitation exists but
was never run under any of the three mechanisms).

---

## The forced experiment (before any paper claim)

**g1bW-second-fact** (the wall's MUSEUM test; <30 min; g1b machinery + e043
install verbatim; free-run expression battery on the existing W1 checkpoints
thrown in at zero cost). The wall is the arc's self-declared central positive
(T133: "law-grade"); if the walled net cannot install fact B, or installs B
and kills A inside the ball, then "memory is made architectural" reduces to
"the organism was frozen with its probe intact" and the paper's second arc
must be re-led (by the no-basin law, which is honestly licensed). If B
installs beside a holding A, the wall becomes the first genuinely sequential
memory architecture in the lab's history and deserves the headline.

Cheap follow-ons in priority order: (1) absorb [C56] iso/tilt cells into
g3R's record (0 min — they are in this file); (2) g2f multi-geometry ruler
before landing (0 min); (3) g2x threat ladder, 4 cells (~30 min); (4) cone
subspace-random + pure-gradient cells (~15 min); (5) directional wall
(gradient-subspace projection at matched protection; ~1-2 h, the only
expensive item, and the one that decides whether the three claims compose
into a system).
