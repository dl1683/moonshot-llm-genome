# e043 design — INSTALL a name: the additive symmetry test of "edit the organism"

Status: DESIGN (implementable as `lab/e043_install_name.py`). Date: 2026-09-24.
Cashes T013's closing pointer: *"the subtractive half of 'edit the organism' is done;
e043 (INSTALL) is now cleanly defined: rows + which body?"* Builds on e023 (row surgery,
bars/metrics conventions), e042 (pos-resolved lesion atlas, pos_lesion machinery, L0H3/
L0-MLP shared-completion finding, L3H5 residual transition), C3 (graft compatibility
follows init lineage; e041 alignment ladder 1.0 → 0.534 → 0.152 → 0.000).

**Question.** e023/e042 showed a name can be SURGICALLY REMOVED (D2+L3H5: Bar-2 at
+0.0008 nats). Can a name B does not know be SURGICALLY INSTALLED — rows-only, rows +
body-graft, or rows + brief exposure — at comparable selectivity? When is editing the
organism bidirectional?

**Design probes already run (2026-09-24, eval-only + in-memory training on
`runs/checkpoints/{e001,e028_b43,e041_bdo}.pt`, NO files written; every number below
is measured, marked MP).** These ground the target choice, the donor choice, the
exposure protocol, and all three predictions.

---

## 1. TARGET — ZEPHYRA (made-up, Z rare core); the B43-borrow candidate is DEAD (MP)

**Candidate A — borrow a corpus name a donor knows but B does not: no such name exists.**
MP battery (e023 construction, ctx 120, NLL/acc, n = all-caps train occurrences):

| name | train occ | B | B43 | BDO |
|---|---|---|---|---|
| FITZWATER | 6 | 0.12/0.98 | 0.19/0.96 | 0.40/0.89 |
| EXTON | 6 | 1.13/0.63 | 1.21/0.53 | 1.47/0.57 |
| OVERDONE | 15 | 0.02/1.00 | 0.03/1.00 | 0.03/1.00 |
| VALERIA | 14 | 0.29/0.87 | 0.19/0.96 | 0.46/0.81 |
| VIRGILIA | 25 | 0.45/0.81 | 0.48/0.85 | 0.42/0.88 |
| PETRUCHIO | 21 | 0.55 | 0.44 | 0.41 |
| JOHN | 33 | 1.09/0.71 | 1.30/0.62 | 0.94/0.61 |
| JULIET | 125 | 0.38/0.90 | 0.33/0.92 | 0.34/0.91 |

Train-side memorization is seed-robust down to SIX occurrences (e023's "21 exposures
did not memorize" was a val-axis artifact — PETRUCHIO's train occurrences are at 0.44-0.55).
There is no B43-only name to borrow. **And B43's rows are basis-incompatible anyway** (C3
in row space, MP): cos(wte row, B row) Z +0.068 / X +0.049 vs same-init BDO Z +0.792;
lm_head Z −0.035 vs +0.828. Naive Z-row copy into B: B←B43 BREAKS the Z-class
(ELIZABETH 0.03→1.87, FLORIZEL 0.41→2.08, FITZWATER 0.12→2.19) while B←BDO is clean
(0.05/0.43/0.52) at dCE +0.0021 vs +0.0007. **Verdict: the donor must be CREATED in-run
by brief exposure, and the donor ladder is BDO (same-init, compatible) > B43 (diff-init,
control), per C3's lineage rule.**

**Candidate B (CHOSEN) — a made-up rare-letter name: ZEPHYRA** (Z,E,P,H,Y,R,A). Rare
core = Z: 161 train chars = 0.016%, carried by exactly 4 types — ELIZABETH 105, FLORIZEL
45 (+8 mixed), FITZWATER 6 (+6), Zounds 5 (MP census). ZEPHYRA occurs 0 times. Y is
common-class (1,648 chars) and untouched by surgery. Z rows are long (MP: |wte_Z| 0.991
vs mean 0.754; |lm_Z| 1.91 vs mean 1.38 — rare letters hold the biggest write rows).
**B does not know it (MP):** spliced batteries (name spliced over host-name train
contexts, e023 ctx convention): FLORIZEL-host n=45: **6.81 nats / 0.130 acc**, per-pos
[0, .91, 0, 0, 0, 0, 0] (only Z→E survives — FLORIZEL's own transition); ELIZABETH-host
n=60: 6.54/0.081; the FROZEN mixed-60 battery used in-run: **9.94/0.008**, per-pos
[0, .01, 0, 0, .05, 0, 0]. Above the uniform floor (ln 65 = 4.17) and the PROSPERO
anchor (4.46, MP): B is CONFIDENTLY WRONG — the incumbents own the slot. X-alternative
noted (X = 112 chars, purely names+numerals: POLIXENES 57, XI 21, OXFORD 14, EXETER 13,
EXTON 6) — Z preferred because its class is 3 names + 1 word (tighter collateral census).

## 2. INSTALL CONTEXTS + BATTERIES (frozen, seed-stamped)

Splice construction (MP): take train occurrences of hosts {FLORIZEL, ELIZABETH} with
p ≥ 280; rng Random(24301); shuffle; first 60 = **install set** (training contexts),
next 30 = **held-out set** (generalization axis — the PETRUCHIO-val channel). Window =
130 pre + ZEPHYRA + ≤120 post. Batteries (teacher-forced, name positions scored):
`R1i` install-60, `R1h` held-30, `R1z` Z-class {ELIZABETH, FLORIZEL, FITZWATER,
Zounds(cap 5)}, `R1n` the e023 10-name incumbent battery, anchor PROSPERO (val).
Corpus: `R2` = 400×256 val blocks seed 202 (MP: base 1.8206 on e001.pt; estimate_loss
protocol reads 1.6211 — gate on estimate_loss, report both) + Z-free fluency channel.

## 3. ARMS (each with measured preview)

**P0 donor prep (in-run, ckpt):** brief exposure of BDO and B43 copies on the install
set. Registered protocol — **masked exposure**: loss = CE at the 7 name-char positions
of name windows + full CE on 48 interleaved corpus windows per step (16+48 @ bs 64),
AdamW (0.9, 0.95) wd 0.1, lr 1e-3 cosine, clip 1.0, ≤100 steps, resumable ckpts
`runs/checkpoints/e043_donor_{bdo,b43}.pt`. Gate G6: donor R1i NLL ≤ 4.5. MP basis
(B-lineage, 300 steps): 3.44/0.62 install, **3.38/0.60 held-out** — the knowledge is
context-general. Protocol was chosen by measurement: pure full-loss replay collapses
(CE 1.79→2.3+, MP); full-loss interleave stalls at 4.6-5.4; masked-pure (no corpus
anchor) installs to 0.09/0.99 (held 0.03/0.99!) but CE → 18.6 — the knowledge is
representable, the organism is the cost; masked+anchor is the only frontier rider.

**A — rows-only (the index hypothesis):** copy donor Z rows {wte, lm, both} ×
{full-copy, Δ-add} into B (384 params max, G2-asserted). Cells: BDO-donor × 6,
B43-donor × 1 (copy-both, the C3 row-space control). **MP preview (B-lineage donor):
NOTHING installs** — 9.82/9.93/9.82 vs base 9.94 (≈0% of the 9.94→3.44 gap), dCE
−0.0001, Z-class flat. The e023-mirror expectation inverts: rows carried the ERASE
(384 params, S_name 573) but carry ~none of the INSTALL.

**B — rows + brief exposure (the plasticity arm):** masked exposure of B itself
(100 steps, ckpt every 25), two cells: bare, and rows-preinstalled (A-both from BDO
donor first). The composition question: do installed rows buy exposure acceleration?
MP preview: the whole effect is the exposure; knee transient peaks EARLY (see D).

**C — rows + body-graft (the transplant arm):** from the exposed BDO donor into B:
{L0-MLP, L0-attn, L0-MLP + rows, donor's top ZEPHYRA head from its own in-run atlas}.
**Does the donor have an L3H5-analog? NO — measured.** JULIET head atlas (global head
zeroing, MP): B = L0H3 +1.52 #1 / L3H5 +0.58 #2; **B43 = L0H5 #1 (+1.18), L0H3 +0.01,
L3H5 −0.02; BDO = L1H0 #1 (+1.11), L0H3 +0.26.** Head IDs are seed-specific; only the
BLOCK structure is invariant (L0-MLP #1 in all three: +7.6/+8.3/+8.4). So the portable
body unit is L0-MLP, and any head graft must be identified from the donor's own atlas
in-run (P0 includes a 1 s ZEPHYRA mini-atlas per donor). **MP preview (B-lineage
donor P): L0-MLP graft alone = 8.49 NLL (~15% of the gap) at dCE −0.0045; rows+L0-MLP
8.28; +L0-attn 8.85 (dCE +0.0144).** Partial, cheap, sub-additive.

**D — full-context control (the ceiling):** masked exposure of B on the install set,
dose ladder {25, 50, 100, 200, 400, 600} steps, one long-horizon cell to 1000.
**MP: the ceiling is a TRANSIENT.** s25: **2.72/0.77 at dCE +0.092** (inside the e023
guard); s50: 3.17/0.70 at dCE −0.02; s100: 3.64/0.64 at −0.03; then DECAY — s400 4.48,
s2000 5.39 while CE climbs +0.37. Best guarded point ≈ 2.7 nats / 0.77 acc. External
"knowing" references: JULIET 0.38/0.90 (a memorized name), PROSPERO 4.46 (anchor),
masked-pure 0.09 (unguarded representability null). **Install bars (frozen):**
Bar-I1 = R1i NLL ≤ 4.17 AND acc ≥ 0.50 at ΔCE ≤ +0.10 (MP reachable: 2.72@25);
Bar-I2 = NLL ≤ 1.0 AND acc ≥ 0.90 at ΔCE ≤ +0.10 (MP NOT reachable; only unguarded,
CE +8-17). Per-position anatomy (MP, s25): [0.01, 0.47, 1.0, 1.0, 1.0, 0.93, 1.0] —
the ONSET (context→Z) is the wall; incumbents ELIZABETH/FLORIZEL keep the slot at
every dose (pos-0 stays ~0 to s2000).

## 4. READOUTS (deterministic, e023/e042 conventions)

- **R1** batteries above; report NLL/acc + per-position for R1i/R1h.
- **R2** corpus CE (400 blocks seed 202) + Z-free channel.
- **R3 generation:** 8 fixed prompts = the 120-char pre-name prefixes of install
  windows 0-3 and held-out windows 0-3; 350 tokens, temp 0.8, top-k 40, per-prompt
  torch seed. Count ZEPHYRA / ELIZABETH / FLORIZEL / stray Z emissions per 10k chars.
  Base expectation (unmeasured, registered): 0 ZEPHYRA, incumbents > 0.
- **R4 atlas re-run** (e042 pos_lesion verbatim, uniform name-position slice):
  post-install atlases for {JULIET, ROMEO, LUCIO, ZEPHYRA} on the best install cell
  + the s25 transient cell (MP: atlas costs 0.9-2 s per name-net).
- **Metrics:** gap-closed G = (NLL_base − NLL_arm)/(NLL_base − NLL_D-best-guarded);
  S_install = ΔNLL_ZEPHYRA / mean|ΔNLL| over incumbents {ROMEO, GLOUCESTER,
  CORIOLANUS} (mirror of S_name; bar ≥ 5); C_install = ΔNLL_ZEPHYRA / ΔCE_val;
  Z-class collateral mean|ΔNLL|; onset-acc (pos 0) as its own row.

## 5. REGISTERED PREDICTIONS (3)

**P1 — rows-only installs (almost) nothing; rows are init-anchored, not content.**
(a) Every A cell (BDO donor): gap-closed G < 10% (preview ≈ 0%), |ΔCE| ≤ 0.005,
incumbent + Z-class |ΔNLL| ≤ 0.05. (b) The B43 copy-both cell fails BOTH ways: G < 10%
AND Z-class mean ΔNLL ≥ +1.0 (preview +1.8-2.2). Verdict rule: *the row index that
sufficed for erasure does not carry installation; row transplantability follows init
lineage (C3) even for the 384-parameter interface.* If (a) fails — some rows-only cell
reaches Bar-I1 — the tombstone's "row directions are causal coordinates" upgrades from
removal to bidirectional causality and C6 is amended on the spot.

**P2 — the additive wall: Bar-I2 is unreachable under the e023 cost guard, and the
best guarded install is an early TRANSIENT.** No arm (A-D) reaches Bar-I2 at
ΔCE ≤ +0.10 (preview: guarded best 2.72/0.77 at s25; unguarded 0.09 at CE +8-17 —
knowledge is representable, the organism is the price). Sub-predictions: (i) the
exposure knee occurs at ≤ 50 steps and DECAYS: NLL(s400) > NLL(s25) + 1.0 with CE
monotone rising after s100 (MP: 2.72 → 4.48 by s400) — continued anchored training
erases the new attractor; (ii) the wall is positional: onset acc (pos 0) stays ≤ 0.10
in every guarded cell while positions 3-6 reach ≥ 0.9 — incumbents own context→Z;
(iii) rows-preinstall (arm B composition) shifts the knee by < 2× steps (rows do not
seed the attractor). If some cell DOES reach Bar-I2 at ΔCE ≤ +0.10, installation is
cheap and the arc's symmetry verdict flips to bidirectional — headline, not failure.

**P3 — install does not disturb the shared L0 machinery, and the new name rides it
at BLOCK level.** Post-install (best cell + s25 cell): (a) incumbents keep L0H3 #1
and their top-3 head IDs; head-atlas Spearman vs base ≥ 0.8 for JULIET/ROMEO/LUCIO
(MP preview on the B-lineage donor P: JULIET L0H3 +1.13 still #1, top-3 IDs unchanged,
L0-MLP +6.62 #1); (b) ZEPHYRA's own carriers: L0-MLP #1 block and L0-attn #1 attn —
the SHARED completion block — with NO dominant single head (MP preview: top head +0.17,
L0H3 −0.49 for ZEPHYRA) — install reuses the organism's local-completion machinery
through a new channel rather than growing a dedicated circuit; (c) base B's ZEPHYRA
atlas is empty (MP: max head +0.34, top MLP +0.09). If ZEPHYRA instead shows a NEW
dominant head, installation grew private circuitry — a stronger "edit" than erasure
ever was.

## 6. THE SYMMETRY VERDICT RULE (pre-registered)

"Editing the organism is bidirectional" iff ALL of:
1. **Additive:** some SURGICAL arm (A or C — no host training) reaches Bar-I1 at
   ΔCE ≤ +0.10 with incumbent collateral mean ≤ +0.05 and Z-class ≤ +0.05.
2. **Subtractive mirror:** already established (e042: Bar-2 at +0.00083, S_name 1,937).
3. **Coordinate identity:** the additive channel is the substrate class the subtractive
   edit touched (rows and/or one body site — not a whole-organism retrain), AND the
   shared L0 machinery is conserved (P3a).

Outcome names (frozen): **BIDIRECTIONAL** (1-3 all hold) / **ASYMMETRIC-CHEAP-REMOVE**
(1 fails — expected from previews: no surgical install; only exposure installs, only
partially, only transiently: *the organism loses a name by index surgery but gains one
only by plasticity, and never fully while the incumbents hold the onset slot*) /
**ASYMMETRIC-BASIS** (1 holds for BDO-donor but fails for B43: C3's init-lineage
boundary IS the symmetry boundary — editing is bidirectional only within an init
lineage). Report which, with the measured frontier (NLL vs ΔCE Pareto for all cells).

## 7. BUDGET, DETERMINISM, CHECKPOINTS (all costs MP)

0.12 s/exposure-step @ bs 64; battery eval ~0.1 s; pos-atlas 0.9-2 s per name-net;
CE battery 0.5 s; generation ≈ 34 s per 12×350 (e023).

| phase | cost |
|---|---|
| P0: batteries, baselines, gates G0-G4, donor prep ×2 (100 steps) + donor mini-atlases | ~1.5 min |
| A: 7 cells × (R1+R2) | ~1 min |
| B: 2 cells × 100 steps + evals @25/50/100 | ~1.5 min |
| C: 4 graft cells + locality asserts | ~0.5 min |
| D: dose ladder (25-600) + 1000-step long cell | ~2.5 min |
| R3 generation (8×350, base + 2 best cells) + R4 atlases (4 names × 2 nets) | ~3 min |
| **total** | **~10 min (≤ 15 hard cap; 5 min slack)** |

Fallbacks if slow: drop Δ-add row variants (−0.3 min); D ladder caps at 400 with the
1000-step cell as optional (−1 min); generation 4 prompts × 300 (−0.5 min).
Determinism: set_seed(24300); splice rng Random(24301); exposure generators 24310+
(donor BDO/B43 get disjoint 2432x/2433x); CE battery seed 202 (e023/e042 convention);
per-prompt generation seeds; G1/G2 bit-identity gates; donor ckpts resumable (Rule 10);
e023 honesty caveat on cross-build bit-determinism inherited verbatim. Outputs:
`runs/e043/metrics.json`, `install_frontier.png` (NLL vs ΔCE Pareto, bars marked),
`machinery_atlas.png` (pre/post heatmaps), `probes.txt`, NOTES.md entry after the run.

## 8. VERIFICATION GATES

- **G0:** estimate_loss(B, val) within 0.03 of 1.6224 (e042 convention; MP 1.6211);
  BDO/B43 reloaded and re-gated against their e041/e028 val parity (≤ 1.7224-class).
- **G1:** two consecutive baseline battery calls bit-identical.
- **G2:** per surgery cell: exactly the target entries differ (row cells: 2×192 at
  row Z; graft cells: the grafted organ's parameter set), all else bit-identical.
- **G3:** donor step-0 eval == donor base battery (same code path).
- **G4:** pre-install ZEPHYRA R1i NLL ≥ 5.0 (anchor sanity; MP 9.94) AND PROSPERO
  anchor in [3.5, 6.0] (MP 4.46).
- **G5:** one generation prompt rerun bit-identical.
- **G6:** donor prep gate: donor R1i NLL ≤ 4.5 after 100 steps (else switch to the
  MP fallback config: lr 3e-4, 16+16, 300 steps — knee 4.01/0.57 — and note it).

## 9. PSEUDOCODE

```python
# batteries (once): spliced install/held sets via rng Random(24301) over host occs;
#   zcensus + e023 10-name battery + PROSPERO anchor; CE blocks seed 202.
# P0: donors = exposure(deepcopy(BDO|B43), masked+anchor protocol, 100 steps, ckpt);
#   donor mini-atlas (ZEPHYRA, pos-resolved) -> donor top head id.
# A: for cell in rows-cells: sd = deepcopy(B); sd[wte|lm][Z] <- donor row(s);
#   assert G2; eval R1+R2. C: same with organ grafts (L0-mlp, L0-attn, top head).
# B: exposure(deepcopy(B), 100 steps, ckpt@25) x {bare, rows-preinstalled}.
# D: dose ladder exposure(deepcopy(B), s in 25..600) + long cell 1000; track knee.
# eval all cells: R1 (i/h/z/n) + R2 + per-pos; frontier = (NLL_i, dCE).
# best two cells -> R3 generation + R4 atlas re-run (pos_lesion, name slice).
# verdicts: P1/P2/P3 + symmetry rule -> metrics.json.
```

## 10. HONESTY CAVEATS (pre-registered)

- Base ZEPHYRA NLL is battery-construction-dependent (MP: 6.5-6.8 single-host, 9.94
  mixed-60) — the battery is frozen and seed-stamped in-run; deltas and gap-closed are
  the registered objects, not absolute NLLs.
- The B-lineage pseudo-donor used for A/C previews is the UPPER BOUND for graft
  transfer (identical residual basis); the registered BDO donor shares only the init —
  expect weaker; B43 weaker still (that ladder is the point).
- Install contexts REPLACE incumbents (ELIZABETH/FLORIZEL) in their own slots — a
  competition original training never staged; Z-class collateral on R1z is therefore
  an outcome, not a confound.
- The masked-pure cell (unguarded) is reported as the representability null, never as
  an install success (CE +8-17 = organism destroyed; e023's ascent-honesty rule).
- The D-arm transient means "ceiling" is step-indexed; the registered ceiling is the
  best guarded point over the whole ladder, and the decay itself is P2(i) evidence.
- Exposure arms are brief (≤100 steps) per the arc definition; D's longer ladder is
  the reference trajectory, not the install instrument.
- R (renorm net) unusable as donor without re-pinned hooks (e028 caveat); excluded.
- Generation counts are small integers: report exact counts + per-10k rates.
