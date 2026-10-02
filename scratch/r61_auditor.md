# R61 AUDITOR — the flight arc, the scale saga, and the envelope (2026-10-02)

Mandate: recompute the headline numbers of the R60→R61 arc (T156–T166,
g1bS2/3/4, g2g2, e193b) from committed metrics; verify the amendments'
consistency; check the ledger and QUEUE; verify owner-envelope compliance;
flag overclaims. Method: pure CPU reads of `runs/*/metrics.json` + run logs
+ git history. No GPU touched (owner envelope honored). Every number below
was re-derived from the artifact, not copied from the fold.

## VERDICT TABLE

| # | Claim (card) | Verdict | Basis (all recomputed) |
|---|---|---|---|
| 1 | e194/T156 sign-front: NOT chase/accumulate/artifact; one recomputation = the whole bonus | **SOUND** | overlap +0.5950→−0.1622; mean front-cos 0.1545 = 255.7× iso floor 0.000604 ("256x" ✓); eff 0.9191; static edge 2.26992; path kill 1.74964 (22.9% below ✓); k-ladder 1.7496 / 2.2744×3, bracket-vs-grid 0.195% ✓; matched-point 0.0396→0.0602→0.134 ✓; 53.2s ✓ |
| 2 | e195/T157 FLEEING-IS-LETHAL (83% drop; direction-of-flight beats alignment; rim) | **SOUND at its stamp** (rescope handled, see A1) | 0.38750 vs 2.26992 → 82.93% ✓; θ₁ panel reverses 0.6156<0.8280 ✓; cos −0.0672 vs +0.0396, 5.86×≈6x ✓; rim 0.6786→0.8243 @ D0.30 ✓; 237.7s ✓ |
| 3 | e196/T158 flight does NOT replicate; live-mid-flight-state mechanism | **SOUND** | u1/u0 2.5847 vs bar 0.60 ✓; spread 66.1% ✓; θ₁ 0.0068 dead ✓; walk dies at step 1, kill 0.52574 vs static 0.52523 ✓ (no bonus); +0.0008 orthogonal ✓; cos(u0,u1) −0.1653 vs −0.1545 ✓; 207.8s ✓ |
| 4 | e197/T160 ALIVE-BUT-NO-FLIGHT | **SOUND** | s*=0.45821; alive t1–t4 = 0.4192/0.8506/0.5281/0.6482, kill t5 D 0.83872 ✓; bonus absent ratio 1.5969 (bar 0.85) ✓; flight absent u1 ratio 3.1690 ✓; th1_u0/u2 ≈0.0670/0.0676 ✓; rim 0.419→0.8537 @ D0.4, crash 1.8273 ✓; cos(u0,u1) −0.2632 ✓; 15/15 gates; 190.1s ✓ |
| 5 | e198/T162 BIOGRAPHY-CARRIES; t=2 wrinkle | **SOUND** | u1 unresolved (min 0.51687 @ D3.0; edge 1.94711) ✓; walk kill 2.41277, ratio 1.2392 ≥ 0.85 ✓; u2 0.83134 → ratio 0.4269 < 0.7 ✓; θ₁ 0.3673, θ₂ 0.6911 ✓; max alignment-cos 0.0918 ✓ ("|cos|≤0.092"); 310.6s ✓ |
| 6 | e199/T163 ONSET-COMMON — a WHEN | **SOUND** | org1 t1 0.17071 BIT-exact vs e195; MIRABEL t1 SOFT floor 1.5408 → t2 0.42692 (committed crosscheck 0.42696, identical verdict) ✓; org1 post-death ctx 0.39558 ✓; both die < t6 (t2/t3) ✓; 23/23 gates ✓ |
| 7 | e200/T164 alternation; death = deepest landing | **SOUND** | curve 3.16902/0.66312/1.29210/0.39727; t1 reproduces e197's 3.1690184 to 1.5e-8 (8 decimals ✓); death recompute 1.59686 = e197's 1.59686 ✓; front cos −0.2632/−0.3129/−0.3413/−0.3528 ✓; both registered bars fail → GRADED ✓; 15/15 gates; 140.2s ✓ |
| 8 | e201/T166 ALTERNATION-UNIVERSAL; rotation outlives the organism | **SOUND with one wording repair (R4)** | alive pairs −0.15453/−0.17511/−0.17814 + half −0.2632/−0.3129/−0.3413/−0.3528, all < −0.10 ✓; G_CROSSFILE 17/17 checks ✓; G_FILES 7 parents ✓; fresh u3: cos(u2,u3) −0.22036 ✓; root-point overlaps org1 0.595/−0.162/0.034, MIR 0.5998/−0.1742/−0.0210/+0.0069, half 0.5472/−0.2686/0.1482/−0.0770/−0.0305 ✓; envelope block (CPU load 36%, threads 4, 23.7s) ✓ |
| 9 | g1bS2/T159 third scale casualty; g0 co-report g1b-like; Adam-clock at 10M | **SOUND-WITH-NUANCE (R6)** | G-ROOT 0.0010485 vs 0.78 ✓; base 2.1299→1.5690 monotone ✓; root g0 0.65835, held30_g0 0.82934 ✓; C g0 +2 = 0.00434 ✓; W1 flat 0.49–0.54 ✓; W2 0.069→0.453 ✓; CE 1.5401→2.5784 ✓; tax W1−C 1.23215 ✓; step-1 3.14556 vs lr·√P = 3.15873 (0.42% — "exactly" too strong; g1bS3/4 at 3.15699 = 0.05%) |
| 10 | g1bS3/T161 channel forms; +2 dip; above-root flat hold at tenth-tax | **SOUND-WITH-NUANCE (R5)** | root 0.64980 vs 0.78 → TEXTURE ✓; CE_R 2.0348@s25→~1.70 settled ✓; W1 dip 0.00010374@+2, flat 0.68–0.84 through +300, flat-retention 1.0515 (0.9x bar held) ✓; W2 0.4877 ✓; tax 0.0549 vs ref 0.5263 ✓; "650x" is 619.8x on exact values (0.6498/0.0010485) |
| 11 | g1bS4/T165 dose question INVERTED; formation non-monotonic; W3 late fade | **SOUND-WITH-REPAIR (R3)** | root 0.2523 ✓; landscape 0.0010/0.6498/0.2523 at (0.30 rms @1e-3)/(0.12 @4e-4)/(0.30 @4e-4) — movement is registered arithmetic (steps×lr), disclosed ✓; W1 0.0400@+1, flat-ret 1.4859 ("1.49x") ✓; W2 1.1986 ✓; W3 0.3061@+200→0.1233@+300 ✓; tax 0.23210 ✓; freezing False (W1 CE 0.952 < root 1.6372) ✓; recovery provenance disclosed. Envelope claim mis-scoped — see R3 |
| 12 | g2g2/T154 AUTONOMY-REPLICATES +0.0333, 3/3; decomposition 0.056/0.013 | **SOUND** | 3/3 positive ✓; median 0.033338 ≥ 0.03 (+11.1%; clears by 0.0033) ✓; organ−paired +0.01301, paired−fixed +0.05577 ✓; device co-read +0.069 vs GPU +0.0758/+0.072 (≈0.007) ✓; g2c xcheck 10/10 identical events, cycle-median diff 0.0 ✓ (but see R8) |
| 13 | e193b/T155 order is physics, distances biography; ridge/cliff split | **SOUND** | ZEPHYRA pZ 0.37671, MIRABEL 0.53288 ✓; crush 0.000306 ✓; ZEPHYRA g-12 0.14864 (G_CONS'd), fallback g+0 0.46184 ✓; MIRABEL g-ray 0.8 ✓, rider dies at 0.82719 vs static cliff 0.8 ✓; pump split +0.0184 / +0.0004 ✓; in-span spreads 3.017x/2.027x ✓; magshuf 0.6651 = 1.677× topk10 (0.3966) ✓; ladder ratios 1.081/0.991/0.986/0.988, sign 2.2424 ✓; STEP_L2 1.65441 vs 1.65429 ✓; 89C hold honored ✓ |
| 14 | Ledger (claims_ledger.md) vs newest cards | **CONSISTENT, PRE-PARK** | internally consistent at its timestamp (~2026-10-01 21:10Z, updated through g1bS2/g2g2); C4 carries the corrected estimator form; no post-park edits (git-verified). Predates the entire flight arc (T156–T166) and g1bS3/4 — acceptable under the user's park; must be extended before any unpark |
| 15 | QUEUE vs runs | **TWO DEFECTS (R2)** | e201 has NO row (STATE/NOTES/git carry it); g1bS3's status cell reads "SUPERSEDED by g1bS3 (DONE 22:10Z…)" — self-referential typo; g1bS5 "DISPATCHING" while live (minor, in flight) |
| 16 | Owner-envelope compliance | **MOSTLY VERIFIED; one mis-scoped claim (R3)** | e200/e201: CPU-only, threads 4, load-checked (metrics envelope blocks) ✓. g1bS4 recovery: owner-gate + 181s owner-cooldown in log, double-poll visible, zero migrations ✓. g1bS5: "[owner envelope] cooldown 180s" in live log ✓. But g1bS4's ORIGINAL legs predate the envelope (e17bbe8) — they ran the older 120s-cooldown guard — and the run rode ≥24 pauses, not 5 |

## AMENDMENT CONSISTENCY (all verified)

- **A1 — T157's e196 rescope:** the `[E196 AMENDMENT ~20:30Z]` block is
  present in-card and numerically accurate (2.58 vs 0.17; dead mid-flight
  state 0.0068 vs 0.679; bonus absent). Propagated: T158, NOTES e196,
  QUEUE e196 row, ledger C4's biography clause. **Gap:** QUEUE's e195 row
  still reads the unscoped "DIRECTION-OF-FLIGHT BEATS CURRENT-ALIGNMENT"
  one-liner with no org-1-biography stamp — violates R56's standing form
  (law-grade one-liners carry n-scope). Repair R7.
- **A2 — T162's meaning-flip:** accurate and correctly attributed — e198's
  numbers unchanged, e199's commit says "e198's meaning flips without its
  numbers changing"; org-1-EARLY carried in T163, QUEUE e198/e199 rows. ✓
- **A3 — T163's WHEN:** consistent everywhere; the "onset-and-death race"
  account matches e196 (org2 full-step died at t=1 before onset). ✓
- **A4 — T164/T166 alternation:** T164 amends the SHAPE (deepening →
  alternation) while explicitly holding T163's WHEN; e200's registered
  bars (ONSET-DEEPENS/ONSET-STALLS) both fail → GRADED, honestly. T166's
  census upgrade to n=3 is gated and stamped "shape claim, not mechanism
  proof". ✓ except R4 below.
- **A5 — estimator-artifact corrections everywhere:** no surviving flip
  copies. Ledger C4, T139 (card text at ~L1459–63), T150, W024
  (RETIRED-BY-CHART with the attenuation numbers), day6 skeleton L331 all
  carry the corrected "attenuates ~2.5x at a matched point; the
  trajectory-level negative is the post-step view" form. The e194 G_SAMEPOINT
  gate re-anchors the dual-estimator convention in the newest cells. ✓

## OVERCLAIM FLAGS

- **F1 (the one substantive finding) — T162's "THE RIM PICTURE IS NOW
  UNIVERSAL … 4/4 lineages: the recomputed direction always improves the
  fact before the far crash".** Recomputed per lineage: org1 rim
  0.679→0.824 (e195 th1_u1) ✓; org2-half 0.419→0.854 (e197) ✓; MIRABEL
  0.367→0.713 (e198 th1_u1, best at D1.1) ✓ — but the fourth lineage
  (org2-dead, e196) has NO live-anchor rim: its θ₁ is dead (0.0068) and
  the only rises in its profiles are 0.0068→0.0868 (the fact stays dead,
  13× relative on a dead read) and root_u2 0.887→0.943 (a post-kill
  dead-state front, context-grade by e196's own disclosure). "Improves
  the fact" is not true of a fact that is dead at both ends. The honest
  scope is 3/4 lineages with live anchors. T160's earlier wording
  ("present in both organisms") was the defensible form; the 4/4 upgrade
  slipped in at T162.
- **F2 — T166's "THE ROTATION OUTLIVES THE ORGANISM"** as a headline:
  the post-death persistence is context-only by the cell's own rules
  (NOTES carries "(never adjudicated)" inline; the census flags every
  post-death pair `adjudicates: false`). The card's final-form paragraph
  does not repeat the qualifier. Earned-as-co-read, not as a noun.
- **F3 — g1bS4's envelope sentence** (NOTES + commit a837028): "the
  owner envelope HELD (launch gates util<=20% AND temp<=70C
  double-polled; 181s cooldowns; 5 heat pauses ridden; never
  migrated)". Log evidence: the owner-gate/owner-cooldown tags appear
  ONLY in `g1bS4r_recovery_run.log`; the original run (pre-e17bbe8) used
  the older single-poll gate and 120s cooldowns. And "5 heat pauses"
  matches only the C arm (5); W1 rode 6, W2 rode 6, W3 ≥2 pre-kill, the
  recovery W3 rode 7 — ≥24 total. "Never migrated" is TRUE (zero
  migration lines in both logs). Science unaffected (pause-and-wait
  preserves numerics), but the compliance sentence over-applies the
  envelope and undercounts pauses.
- **F4 — minor numeric inflation:** "650x" (T161/NOTES/QUEUE g1bS3) is
  619.8x on exact values; "lr·√P exactly/CONFIRMED" (T159) is 0.42% off
  at g1bS2 (3.1456 vs 3.1587), 0.05% at g1bS3/4 (3.1570).

## LEDGER / QUEUE / STATE CROSS-CHECKS

- Claims ledger: frozen pre-park (last substantive update ~21:10Z Oct 1,
  g2g2 + g1bS2 rows in, g1bS3-named in C6). No flight-arc rows — correct
  under the park; a debt to discharge only if drafting resumes.
- Paper-lane park (user directive 2026-10-01 ~21:30Z): HELD — zero
  commits touching the skeletons/ledger/drafts since 21:00Z Oct 1
  (git-verified); QUEUE header carries the directive; no paper items
  promoted since.
- QUEUE↔runs: e201 row missing; g1bS3 status typo; all other DONE rows
  (e193–e200, g1bS2/4, g2g2, e193b) match their runs' verdicts and
  metrics.
- STATE.json: `current_experiment` (e201 FOLDED / fleet g1bS5 gated)
  matches reality — g1bS5 is live under the envelope (log tail shows
  m015 root 0.7285 below the 0.78 bar at mid-sweep, m020/m025 pending).
  `last_review` was ~15.7h stale vs the 75-min cadence (the ninth
  disruption window); R61 discharges it.
- Review cadence honesty: the overnight heat shutdown (ninth disruption)
  and the owner-envelope transition explain the gap; ordering + commits
  remained the reliable record per the clock note.

## PRIORITIZED ANCHORED REPAIRS

- **R1 (P2, F1):** THINKING.md T162 (~L792): rescope "THE RIM PICTURE IS
  NOW UNIVERSAL (2/2 architectures, 4/4 lineages…)" to "3/4 lineages with
  a live anchor (org1, org2-half, MIRABEL); the dead lineage's rim is
  context-grade (0.0068→0.0868, the fact stays dead; root_u2 0.887→0.943
  is a dead-state front)". Mirror in NOTES e198's rim line ("4/4
  lineages") and QUEUE e198's row.
- **R2 (P2, QUEUE):** add the missing e201 row (DONE — T166
  ALTERNATION-UNIVERSAL at n=3, rotation outlives the organism as
  context-graded co-read); fix g1bS3's status cell from "SUPERSEDED by
  g1bS3 (DONE 22:10Z…)" to "DONE 22:10Z (T161…)".
- **R3 (P2, F3):** NOTES g1bS4 fold + one clause in T165: reword the
  envelope sentence to "the RECOVERY legs ran under the owner envelope
  (util≤20%/temp≤70C double-poll, 181s cooldown); the original legs
  predate the envelope (e17bbe8) and ran the 120s guard; pause-and-wait
  held throughout with zero migrations; ≥24 pauses ridden (C 5, W1 6,
  W2 6, W3 ≥2, recovery 7)".
- **R4 (P3, F2):** T166 final-form paragraph: append "(post-death
  persistence: context-graded co-read, never adjudicated)" to THE
  ROTATION OUTLIVES THE ORGANISM.
- **R5 (P3, F4):** T161 + NOTES g1bS3 + QUEUE g1bS3: "650x" → "~620x
  (0.6498/0.0010)".
- **R6 (P3, F4):** T159/NOTES g1bS2: "one AdamW step = lr*sqrt(P)
  exactly" → "within 0.4% of lr·√P (3.1456 vs 3.1587; g1bS3/4's 3.1570
  within 0.05%)".
- **R7 (P3, A1):** QUEUE e195 row: append "[org-1 biography — rescoped
  by e196/T158]" to the one-liner, per R56's n-scope standing form.
- **R8 (P4):** T154/NOTES g2g2: "bit-exact reproduction of g2c's
  realization" → "exact at read precision (10/10 events, cycle-median
  diff 0.0; soft xcheck, never a gate)".

## WHAT HELD (credit where due)

The e194→e201 chain's provenance engineering is the strongest in the
lab's record: md5-gated streams and rays across seven parent files,
tiered BIT/TEXTURE identity gates with the tier stamped per gate, the
cross-file constants check (17/17), and post-kill rows structurally
barred from adjudication. Every one of ~70 recomputed headline figures
reproduced to print precision except the four flagged above (three
wording inflations, one mis-scoped compliance sentence, one 4/4 that
should be 3/4). The verdict discipline (TEXTURE on every gate failure,
nothing adjudicated, no bar shopping) was followed verbatim in all four
g1bS takes. No e166-class invalidity. No estimator-flip copy survives.
