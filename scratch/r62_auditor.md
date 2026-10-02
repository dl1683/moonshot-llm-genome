# R62 AUDITOR — the T172-T185 arc recomputed (2026-10-02)

Mandate: pure recomputation over the committed metrics of
{g1bS5,g1bS6,g1bS7,g1bS8,g10,g1c_root,e202,e203,e205,e206,e207,e208,
e209,e210,e211,e212,e213,e182c2} + the ledger/QUEUE/W028 consistency
pass. Owner envelope honored: no GPU, CPU reads only, no writes outside
this file.

## Method

Every headline number in the NOTES entries was recomputed from the
run's own metrics.json (ratios, bars, Spearman correlations with
tie-aware ranks, min/max statistics), not re-read from the prose. 34
distinct headline quantities checked; 31 reproduce exactly; 3 are
wording-level misquotations (direction robust under every consistent
statistic — none changes a verdict). One staleness/harvest debt found
(g1d). Two ledger hygiene items.

## Verdict table

| # | Claim (anchored) | Recomputed | Verdict |
|---|---|---|---|
| 1 | g1bS5 curve 0.6498/0.7285/0.7677/0.7431/0.2523; SHARP-OPTIMUM (peak >= 0.6998); peak 0.0123 under the 0.78 bar; held30 0.39->0.58->collapse | curve array verbatim; 0.7677-0.6998=+0.0679; 0.78-0.7677=0.0123; held30 [0.392,0.386,0.557,0.583,0.060] | VERIFIED |
| 2 | g1bS7 PEAK-LOTTERY: 0.9351 vs 0.7677, \|d\| 0.1674 > 0.10; first 10M root over 0.78; redraw still clears 0.6998 and outranks 0.7431 | 0.93508-0.76766=0.16742; 0.9351>0.78; 0.9351>0.6998; 0.9351>0.7431 | VERIFIED |
| 3 | g1bS7 genuine-redraw gates: s25 divergence 4.6x fuzz; ~100% elements differ; seed axis ~6x device fuzz | s25 \|dg0\|=0.09968 = 4.57x the 0.0218 floor; 99.997% differ; bits NOT identical (max_abs_diff 0.0573). "~6x": recomputable ratios are 4.6x (s25) and 7.7x (final root g-12); 6x sits between — LOOSE | VERIFIED (one loose texture) |
| 4 | g1bS6 WALL-FADES: bar 0.7546 (=0.983x root); C dead +1 (D_kill = one AdamW step); every rung breaches at +1 (W1 0.0940 / W2 0.0015 / W3 ~C); flat retention 0.895>>0.464>>0.0002; W1 misses 0.9x-root by 0.005; tax +0.18; adapting (W1 CE 0.968 < root 1.697) | bar=0.7677*0.9/0.9156=0.75459; D_kill 3.1568 vs one-step 3.1587 (0.06%); first_ck_below_bar=1 x3; 0.895/0.464/1.8e-4; 0.9-0.89496=0.00504; W1-C=0.17969; CE holds | VERIFIED |
| 5 | g1bS6 "direction survives: ~1000x separation at every checkpoint >= +10" | ratios: +10 59,098x / +50 2,244x / +100 **276x** / +200 2,115x / +300 4,325x (g1bS8: 8,199/8,663/1,396/664/487) | **OVERSTATED AS A FLOOR** — order 10^3 fair, but "every checkpoint" fails at +100 (276x) and +300 (487x); honest form "276x-59,000x (order 10^3)" |
| 6 | g1bS8 WALL-FADES draw-clean: bar 0.9192; all rungs breach at +1; dip shallowed 4x (0.3741 vs 0.0940); flat phase ~0.74-0.80 root-independent; ordering 0.79>>0.30>>0.00; tax +0.10 | bar=0.9351*0.983=0.91916; first_ck_below_bar=1 x3; 0.3741/0.0940=3.98x; flat [0.7385-0.7990]; 0.7898/0.2958/1.2e-5; W1-C=0.1025 | VERIFIED |
| 7 | g1bS8/NOTES "retention FELL 0.96x -> 0.79x" | committed `flat_phase_retention_min`: **0.895 -> 0.790**; 0.96 matches take-5's median/+50 retention (0.9646), 0.79 take-6's min — mixed statistics. Consistent pairs: min 0.895->0.790, median 0.965->0.833, mean 0.997->0.828 — direction robust to the choice | **MISQUOTED PAIR** (finding robust) |
| 8 | g10 FIX-IMPOTENT: F1==W1 to 0.0 at +1/+2/+4/+10; F2@+1 == C@+1 to 0.0; F3 walk unbounded to 32 raw, dead by +4, CE 0.726 < C's 0.788; s=R/||du1||=0.4230 | deltas 5.2e-8/1.2e-7/1.2e-7/-1.8e-7 (=0.0 at display precision); F2@+1 = C@+1 = 0.00067278 (exact 0.0); max walk 31.992; F3 min 1.26e-5@+4; 0.72608<0.78802; 1.336/3.1587=0.4230 | VERIFIED (note: F1 diverges from W1 by 0.0895 at +300 — disclosed in metrics as CUDA fuzz; NOTES correctly scopes the isomorphism to <=+10) |
| 9 | g1c_root ROOT-WALL-HOLDS: root 0.9026 vs locked 0.9156 (ruler 0.9289; L2 18.6); W1 0.8214@+1 -> 0.9406@+300, flat min 0.9265 >= 0.9x-root 0.8124 AND strict 0.8873; ref dips 0.746-0.803; held30 0.653->0.758; anatomy: d183-independent g-12, negative A129; C dies by +1 | root_cells gm12 0.90263/gp12 0.92894/l2 18.603; flat min 0.92646; bar_flat 0.81237, strict 0.88727 both hold; ref 0.7768/0.8028/0.7460; held30 0.6529->0.7579; d183_gm12==gm12 (0.9026); A129 -0.1327; C first below 0.27 at +1 | VERIFIED |
| 10 | e202 GRADED: twin on curve at pair 0 (+0.009 vs org1), +0.052 shallower at pair 1 (0.0024 over the 0.05 bar); cos1 law +0.002 -> -0.206 -> -0.263 (bit-exact e197 anchor); core statistic breaks at the half rung (lag-2 collapses to +0.141) | devs 0.00925/0.05244; 0.05244-0.05=0.0024; anchor abs_diff 0.0; half cos2_t0 0.14055; cores 0.1339/0.2332/0.2019 non-monotone | VERIFIED |
| 11 | e203 GRADED: twin series off the family curve at pair 0 (+0.185 vs -0.263), shallower at pair 1 (+0.264); step-matched dead-full +0.087/+0.026 | deltas 0.18511/0.26380; context 0.08724/0.02639 | VERIFIED |
| 12 | e205 ARRIVALS-CONTINGENT (desk-forced): org1 t1 0.635x > 0.60; MIRABEL 0.904x > all bars; half t2 0.408x bar-robust; earliest flips org1->half at 0.60; edge multiples 3.72/2.12/0.616; half band median 0.8526 | 0.3875/0.6101=0.63517; 0.8312/0.92=0.90353; 0.3483/0.8526=0.40849; COMMON@0.60 {org1 null, half 2}; 2.2699/0.6101=3.7207, 1.9471/0.92=2.1164, 0.5252/0.8526=0.61602; 0.3483/0.40849=0.85260 | VERIFIED EXACTLY (the desk-forced arithmetic reproduces to 5 decimals) |
| 13 | e206 CLOCK-ONE-LINEAGE: ladders org1 0.386->0.020, MIRABEL 0.440->0.182->0.064, half 0.776->...->0.192; c_death all <= tau 0.2 (3/3); MIRABEL lands (-1), half misses (+2), org1 unsettable | all series verbatim; 0.020/0.064/0.192 <= 0.2; err -1/+2/undefined | VERIFIED |
| 14 | e207 CORE-GRAINY: cores 0.13387->0.23319->0.23013->0.20186; interior -0.0031 below quarter, +0.0283 above half; cos1 monotone to -0.26318; cos2 declines 0.26970->0.14055 | series verbatim; 0.23013-0.23319=-0.00306; 0.23013-0.20186=+0.02827; cos1_t0 monotone; cos2_t0 monotone | VERIFIED |
| 15 | e208 MARGIN-PREDICTS (desk-forced): 4 rows 3.72/2.12/1.24/0.616 with survivals 1/2/0/0; every >2 row outlives every <1 row; Spearman 0.738; fork: counterfactual survival 4 -> DECORRELATED, Spearman -0.400 | margins recompute from edge/band (0.92/0.74=1.2432; 0.5252/0.8526=0.6160); min(surv\|>2)=1 > max(surv\|<1)=0; Spearman (tie-aware) = 0.7379; fork -0.400 | VERIFIED |
| 16 | e209 MARGIN-BREAKS at n=7: R5 0.744x surv 1, R6 0.786x surv 3 (deaths +2/+4); Spearman 0.738->0.225; "walled bands GROW 2-3x" | 1.1757/1.5798=0.74419; 1.3413/1.7055=0.78642; 0.22454; per-root ratios vs pristine 0.61: **2.59x / 2.80x / 1.50x (R7=0.9141)** | VERIFIED except "2-3x" — R7 was already 1.5x at mint ("2 of 3 roots"); moot after the e211/e212 retirement |
| 17 | e210 SAME-EPISODE-BREAKS: W5 (0.616x <1) survived 4; n=6 natural-clock table separates (the hinge); Spearman 0.124; W3 step-1 read 1.20x the bar | W5 margin 0.61602/survival 4; n=6 violations 0; Spearman (tie-aware) = 0.1239; 0.32266/0.27=1.195 | VERIFIED |
| 18 | e211 GRADED: spectra match (PRs 1.02-1.04x, shape 0.058 bits); curvatures geomean 1.076, 0/3 under bar; same-instrument medians pristine 1.03 vs walled 0.85/0.90/0.92; the 2-3x gap = instrument shadow | ratios 1.024/1.035/1.035; shape 0.0577; geomean 1.0761, count 0; 1.0315 vs 0.8462/0.8955/0.9202 | VERIFIED |
| 19 | e212 GRADED: pristine 0.756 [0.435-2.128, 0 censored] vs walled 0.846/0.896/0.920 -> ratio 0.844x; neither in window [0.809, 0.957] nor below the 0.7x bar (threshold 0.627); bounded <= 16%; SE window [0.533, 1.233] | median 0.75592, kills [0.4347, 2.1284, ...]; 0.7559/0.8955=0.84411; window [0.80925, 0.95719]; threshold 0.62687=0.7*0.8955; 1-0.844=15.6%; [0.53328, 1.23316] | VERIFIED |
| 20 | e182c2: free find 0.4212/0.4202 (ratio 0.9978); reversed 0.561@+80 vs 0.766 (ratio 0.73, band [0.510,1.148]); +50 co-adjudication flips SPECIFIC by 0.011; lag-then-converge series; tmpl 1.41x ctrl; draw ratios 0.889/0.818; ppl 71.3->34.6; p0 0.907; n=19 | 0.42119 vs 0.42025 = 0.99777; 0.56126/0.76554=0.73316; band_lo 0.4320-0.4212=0.0108; series [0.0029,0.0682,0.4212,0.5613] vs [0.0157,0.2169,0.6479,0.7655]; 1.41394; 0.88934/0.81756; 71.336->34.596; 0.90749; 19 facts | VERIFIED |
| 21 | e213 PATH-PARTIAL: 6/11 non-floor match (all four at +50; fact+ctrl at +80); 5 wander (+10 all >1.1; near 0.857/tmpl 0.804 at +80); Spearman -0.60; depth not size/base-rate; free find at dp 0.0 | 6 MATCH / 5 WANDER / 1 floor-match; +50 ratios 0.912/0.979/0.912/0.998; +10 non-floor 1.289/1.164/1.125; 0.85743/0.80356; Spearman(w1,ratio)@+80 n=4 = -0.60 (committed stat; my 11-cell analog -0.81, same sign); near n=3 & tmpl n=19 wander; R0 0.589 & 0.907 wander; ratio 0.99777 | VERIFIED ("-0.60" is the +80/n=4 statistic; NOTES omits the n — add it) |
| 22 | Amendments propagated: T163 rescoped WHEN / T166 retired nouns / T179 retired free find / T149 template-locus | THINKING.md carries [E205 AMENDMENT ~11:15Z] on T163; [NULL-DERIVATION RESOLUTION ~09:50Z: "the nouns RETIRE"] on T166; [E211 AMENDMENT ~15:45Z: "RETIRED"] on T179; [E182C2 AMENDMENT ~16:05Z: "LICENSED"] on T149 | VERIFIED — all four in place |
| 23 | Claims ledger C4/C5/C7 vs the arc | untouched claims, no contradictions introduced by T172-T185 | CONSISTENT |
| 24 | Claims ledger C6 (the wall) vs the newest cards | see repairs: stale "(g1c-root queued)" fragment contradicting the same cell's result; take-5-only numbers (retention 0.895, tax +0.18) with g1bS8 (draw-clean, 0.790, +0.10) and g10 (FIX-IMPOTENT) never folded in; "~1000x" floor issue (#5); header date 2026-10-01 predates the 2026-10-02 amendment, unstamped | **STALE/SELF-CONTRADICTORY — repair owed** |
| 25 | QUEUE vs runs | g1bS7 (T172) and g1bS8 (T178): DONE with runs/ + NOTES, but NO QUEUE rows. g1d: registered (f3ab1be) + amended (65dfb44), finished 14:23:20Z — no QUEUE row, no NOTES entry, runs/g1d/ UNCOMMITTED. e209's one-liner still carries the retired "noise ball grows" free find with no retirement pointer (e211/e212 rows do record it) | **GAP — repair owed** |
| 26 | STATE/W028 vs the disk record | STATE.last_heartbeat 14:17Z, current_experiment "g1d mid-run"; W028 (~17:15Z) "g1d running" — but runs/g1d/metrics.json is COMPLETE at 14:23:20Z: verdict TEXTURE (GATE FAILURE: G-CONS; fresh base val 1.5113, root g-12 0.5235, full C trace). The run finished 3h before W028 and was never harvested/committed | **STALENESS + HARVEST DEBT — top item** |
| 27 | W028 shape/height citations | 7 of 8 instances accurate (formation shape/height T172; onset within-organism T163/T173; 124M mid-dose T185; gradients destination/schedule T174; margin triad T177-T180; band triad T179-T184; ray order T155). The wall sentence "protection replicates across wash x root x **base x scale**" is wrong twice: at scale the STRICT protection faded (WALL-FADES is the committed verdict — what replicates is direction + the root-independent flat phase); the BASE axis is untested (g1d G-CONS failure, unharvested) | **ONE CONFLATION — repair owed at harvest** |
| 28 | g1bS5 "CE_R healthy 1.66-1.70 across the sweep" | actual 1.6372-1.7035 — endpoints 0.02 outside the quoted band | TRIVIAL |

## Prioritized anchored repairs

**P1 — harvest g1d (the record is ahead of the lab).**
runs/g1d/metrics.json is complete (14:23:20Z): TEXTURE, GATE FAILURE
G-CONS (root g-12 0.5235 on the fresh seed-44 base; base val 1.5113;
the STEPS AMENDMENT 65dfb44 already on the git record). Owed: NOTES.md
entry, THINKING card (or fold into T181's open-rungs list), QUEUE row,
STATE refresh, `git add runs/g1d + commit + push`. Without this the
lab's memory lacks its newest cell and W028's "g1d running" is
untrue against the disk.

**P2 — claims ledger C6 (scratch/claims_ledger.md).**
(a) delete the stale "(g1c-root queued)" fragment (line 18 — the same
cell already records g1c-root's ROOT-WALL-HOLDS); (b) fold the arc's
close into the scale clause: g1bS8 = WALL-FADES draw-clean at the
in-spec 0.9351 root (bar 0.9192; every rung first-below at +1; the
fifth take's lottery caveat discharged), retention 0.895 -> 0.790,
tax +0.18 -> +0.10, g10 = FIX-IMPOTENT (the kill is the first step's
SIZE 3.16 raw vs the 1.34 rung — F1==W1 to 1e-7); (c) re-anchor
"~1000x fact-control separation" to "276x-59,000x (order 10^3) —
minimum at +100/+300"; (d) stamp the amendment date.

**P3 — QUEUE.md.** Add rows: g1bS7 DONE (T172: PEAK-LOTTERY — 0.9351
vs 0.7677, the first 10M root over the bar), g1bS8 DONE (T178:
WALL-FADES draw-clean; the flat phase root-independent), g1d DONE
(TEXTURE, G-CONS gate stop — the base axis still untested). Append
"(retired by e211/e212)" to e209's "noise ball grows" clause.

**P4 — wording, one-line edits, none verdict-touching.**
- T178/NOTES g1bS8: "0.96x -> 0.79x" -> quote one consistent
  statistic: min 0.895 -> 0.790 (or medians 0.965 -> 0.833).
- "~1000x at every checkpoint" (NOTES g1bS6/g10, T176, T178, C6,
  W028) -> "order 10^3 (276x-59,000x)".
- W028's wall sentence -> "protection replicates across wash x root
  (strict); at scale in the direction/flat-phase form (the strict bar
  fades); the base axis untested (g1d G-CONS)" — at harvest.
- e213 NOTES: mark "-0.60" as the +80, n=4 statistic.
- g1bS7 NOTES: "seed axis ~6x the device fuzz" -> "4.6x (s25) to 7.7x
  (final root)".
- e209 (historical note only): the mint-time "2-3x" was 2.59/2.80/
  1.50x per root — R7 already off-band; superseded by the retirement.
- Hygiene: scratch/r61_auditor.md + r61_critic.md untracked;
  runs/_envelope_log.jsonl + INBOX_from_matrix-native-math.md modified
  uncommitted.

## Honest bounds of this audit

Recomputation only — no instrument re-ran, no gate re-derived on
device. Cross-run bit-identity claims (BIT anchors, md5 gates) were
read from the committed gate tables, not re-executed. The CPU fp32
probes vs GPU wash textures are outside a read-only pass. Spearman
values were re-derived with tie-aware ranks and match to 3 decimals.
