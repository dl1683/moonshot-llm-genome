# R58 AUDITOR — frontier review 2026-09-30 (~12:00Z)

Scope per mandate: recompute g3K / opt1 / g1bW headline numbers from
`runs/{g3K,opt1,g1bW}/metrics.json`; verify the C13-1 repair; ledger
integrity; double-session provenance; opt1 soft spots. Read-only except
this file. Every number below was recomputed from the committed metrics.

---

## 1. Recomputed headline numbers

### g3K (runs/g3K/metrics.json; NOTES.md:90-123, T137)

| claim | recomputation | verdict |
|---|---|---|
| kappa_store 5.0, iso rungs {8,4,8}/wash 1 | per-seed thresholds 6.561/2.807/5.770, kill_rungs 8/4/8, mean 5.046; wash kill_rung 1 (g0 5.9e-5 at 1x) | **SOUND** |
| kappa_host 6.0, iso rungs {8,8,16}/wash 1 | per-seed 4.222/4.032/9.740, kill_rungs 8/8/16, mean 5.998; wash kill_rung 1 (pz 0.0040 at 1x) | **SOUND** |
| intervals [2.8,6.6] / [4.0,9.7] | store min/max 2.8068/6.5610; host 4.0319/9.7400 | **SOUND** |
| verdict aggregation-robust (min/mean/max all MIXED) | SPLIT needs store>=8 (max 6.56 — never); UNIVERSAL needs both<=4 (host min 4.0319 — never) | **SOUND** (note: host min misses the <=4 bar by 0.03; safe only because the store leg can never reach 8) |
| "static iso shows a graded 4-10x basin" (NOTES.md:109, QUEUE.md:197, skeleton:215) | host seeds 4.03-9.74 = "4-10x" TRUE; store seeds include 2.81 (seed 11402) — full per-seed range 2.81-9.74 | **SOUNDS-WITH-SCOPE** — "4-10x" is host-true, store-optimistic; T137:788 scopes it correctly ("the HOST fact holds a graded 4-10x basin") |
| "Kills precede organism wreck (CE_R 1.84-2.62 at kills)" | iso kill-rung CEs: 2.219/1.836/2.313 (store), 2.126/2.114/2.617 (host) = 1.84-2.62 | **SOUNDS-WITH-SCOPE** — true for the ISO arms only; the wash kill at 1x comes WITH wreck (CE_R 3.79 store / 2.84 host, over the 3.0 co-read bar) |
| wash-1x magnitude-uniform (0.324 vs 0.663; 0.142 vs 0.139) | 0.3245 vs 0.6627; 0.1424 vs 0.1391 | **SOUND** |
| wash drift cos ~0.13 | cos_d01_d0_s300 0.146 (store) / 0.122 (host) | **SOUND** |
| provenance gates (g3R bit-exact; S-DISC committed values) | store root g0 0.8885950 reproduces, host bit-identical, store-off 0.0012; host root/wash/D_all all reproduce bit-exact | **SOUND** |

### opt1 (runs/opt1/metrics.json; NOTES.md:51-86, T139)

| claim | recomputation | verdict |
|---|---|---|
| step-1 pre-clip grad norm 0.9829 in EVERY arm | all 7 arms: 0.9829167127609253 | **SOUND** |
| AdamW 1.6543/step vs SGD-1e-3 0.0010 = "1687x" (NOTES.md:68, THINKING.md:716, QUEUE.md:198, skeleton:286) | 1.6542880535125732 / 0.0009829070186242461 = **1683.06x** (lr*sqrt(N)/lr*min(g,1) = 1655.01/0.98292 = 1683.8; no derivation in the metrics yields 1687) | **OVERCLAIMED** (arithmetic slip, +0.24%; story unchanged, number is not derivable) |
| "every Adam variant kills at D ~ 2.49-2.84" (NOTES.md:69-70, T139:717, QUEUE.md:198) | checkpoint D_at_kill: A0 2.489, A4 2.485, A5 2.489 — but **A3 (warmup) 3.636**; 2.84 is A3's D *linearly interpolated at t_x 16.39* (1.4494 + 0.639x(3.6361-1.4494) = 2.847), i.e. the quoted range mixes two conventions; the metrics' own adjudication clause lists 3.636 verbatim | **SOUNDS-WITH-SCOPE** — the range is real under the interpolated reading; under the checkpoint reading everywhere else in the table it is 2.49-3.64 |
| "the kill still arrived at the same displacement within ~15%" (NOTES.md:71, T139:718-719) | 2.847/2.489 = 1.144 (+14.4%) — valid ONLY as A3-interpolated vs A0-checkpoint; checkpoint-vs-checkpoint = 3.636/2.489 = +46% | **SOUNDS-WITH-SCOPE** (same convention mix; "same displacement" survives only on the interpolated reading) |
| warmup 10.08x | t_x(A3)/t_x(A0) = 16.3941/1.6268 = 10.077; ADAM-AMPLIFIES fires measured (>=2x) | **SOUND** |
| SGD fact rise 0.916 -> 0.940-0.955 at |d| <= 0.29 | root 0.9156; A1 0.9399 @ d 0.0565; A2a 0.9530 @ 0.1216; A2b 0.9548 @ 0.2851 | **SOUND** |
| A5 bit-identical to A0 | G_A5_NULL all_bit_identical true; direct field-by-field recomputation of traj/ckpt/tstar: max metric diff 0.0 | **SOUND** (and correctly framed in deviations as null-by-construction gate, n=2 determinism replicate) |
| A0 reproduces e185 control max|diff| 3.2e-13 | G_A0_E185 max_abs_diff 3.2000097e-13 (disp/gm12/ce_batch all 0.0) | **SOUND** |
| OPT-AGNOSTIC could not fire; SGD clocks ~379/1174/3228 extrapolated | extrapolated_steps 379.4/1174.3/3228.4, all reached_D_kill false, cap-limited | **SOUND** |

### g1bW (runs/g1bW/metrics.json; NOTES.md:12-47, T140)

| claim | recomputation | verdict |
|---|---|---|
| walled final A 0.83 (min 0.65, every checkpoint) / B 0.0736 / CE_r 1.63 | A_gm12 0.8324; min over cells 0.6505 (s10); B 0.07360; CE_r 1.6285 <= root+0.30 = 1.9635 | **SOUND** |
| reference B 0.0628 (B fails everywhere at this dose) | ref final B_gm12 0.06284; walled 0.0736 — both <= 0.27 | **SOUND** |
| MUSEUM fires as registered; NOT contrast-licensed | adjudication: MUSEUM true, SPLINT-REFUTED/ZERO-SUM false; NOTES/T140 report the reference-leg failure and refuse the splint conclusion | **SOUND** |
| onset channel: B partial-form peak 0.21 walled vs 0.53 unwalled; A row0 0.62-0.69 vs 0.006 | walled B_g0 max 0.2101 (s100); ref B_g0 max 0.5328 (s100); walled row0 0.624-0.691; ref final row0 0.0058 | **SOUND** |
| wash reproduces g1bR to 7 decimals (0.9156978 vs 0.9156979) | G_WASHREP mine 0.9156977534 vs g1bR 0.9156978726, diff 1.19e-7; both values disclosed | **SOUND** ("to 7 decimals" is a hair loose — they differ by 1 ulp in the 7th decimal; the parenthetical discloses it) |
| free-run ZEPHYRA survived (3/2800 chars) | freerun.walled_final A: name count 3, chars 2800 | **SOUND** |
| all 9 gates green | G_BITEXACT, G_ROOT, G_BITROOT, G_BFRESH, G_CTRL, G_WASHREP, G_PIN, G_INPUTS, G_ANCHOR — 9, all pass | **SOUND** |
| (metrics artifact wording) adjudication clause "the reference (unwalled) leg **installs** B at 0.0628" | 0.0628 <= 0.27 = fails, does not install; the clause then states the splint rescope unqualified — internally inconsistent with its own number | **OVERCLAIMED** (artifact wording only; NOTES/T140 already correct it at the interpretation layer)

---

## 2. C13-1 repair — trajectory HYPOTHESIS vs law

Carried correctly at:
- THINKING.md:771-777 — amendment bracket atop T137, sets status "THE
  TRAJECTORY HYPOTHESIS ... until e188's integral test + >=2 more organisms".
- THINKING.md:726-729 (T139) — "THE TRAJECTORY HYPOTHESIS (T137) REFINES:
  never 'any learned path kills'".
- skeleton R6(c) (scratch/day6_paper_skeleton.md:209-223) — "THE TRAJECTORY
  HYPOTHESIS (PROPOSED — g3K n=1 per organism, kappa intervals overlap ...
  replication owed: e188 + two more organisms; supervisor C13-1)".
- skeleton framing sentence (:224-232) — "the hypothesis's slogan, PROPOSED
  per C13-1 — not law-graded until replicated".
- skeleton abstract clause 4 (:248-260) — "a direction-typed forgetting law
  with a PROPOSED trajectory hypothesis ... n=1, replication owed" (the
  "law" attaches to g3R's licensed direction-typing; the trajectory part is
  hypothesis — consistent).
- skeleton paragraph 2 (:18) — "(+g3K, PROPOSED n=1): ... the trajectory
  hypothesis — ... replication owed per C13-1".
- skeleton gaps item 11 (:292-294) and item 10 (:295-301) — the law-grade
  gate and "Never state the no-basin law against random displacement".

Still minting "law" (residuals):
- QUEUE.md:197 — "THE NO-BASIN LAW IS A TRAJECTORY LAW" (living ledger,
  rewritten at reviews — should carry the C13-1 stamp).
- scratch/day6_paper_skeleton.md:161 (R3b) — "the trajectory law's nearest
  prior is THEORY".
- THINKING.md:740 (T138 title) — "the trajectory law is new as a CONTROL"
  (written 10:55Z, 5 min before the amendment; journal is append-only — an
  amendment line is the honest fix).
- NOTES.md:90/109 (g3K entry) — historical, pre-amendment by ~15 min;
  append-only, acceptable as history but currently carries no forward
  pointer to the C13-1 amendment (T137 does).

Verdict: **SOUNDS-WITH-SCOPE** — the repair is real and load-bearing in the
paper skeleton and on T137/T139; three residuals (QUEUE row, skeleton R3b,
T138 title) still say "law" and post-date or ignore the amendment.

Related skeleton nit: paragraph 2 (:18) still contains "content-free noise
at displacement-match exits it identically" adjacent to the g3K correction.
Read loosely this restates the static-noise-kills reading g3K refuted; the
sentence is defensible (e185's noise arms were noise-LABEL TRAINING) but the
two wordings sit un-reconciled until gaps-item-10's rewrite is applied.

---

## 3. Ledger integrity

- g3K/opt1/g1bW rows: DONE, metrics + PNG + NOTES entries all exist. SOUND.
- opt1b: QUEUE.md:199 "DISPATCHED 11:30Z" — backed by lab/opt1b_sgd_kill.py
  (untracked, in flight), runs/opt1b_smoke/chunk_state.pt (the ckpt-resumable
  smoke), no runs/opt1b metrics yet (expected mid-run). Bars in the script
  (SGD-KILLS-AT-GATE / SGD-SPARED-AT-GATE, D=2.6 spared gate) match the QUEUE
  row verbatim. SOUND.
- g1bW2: QUEUE.md:196 QUEUED — no run dir, scratch/fold_g1bw2.py exists (fold
  prep). Correct: queued, not claimed. SOUND.
- g1bS: STATE.json:8 says "g1bS DISPATCHING (GPU freed)"; QUEUE.md:201 says
  "NEXT GPU SLOT ... **design owed FIRST** (this is the design step)". The
  design draft v1 exists and is committed (088ca91, scratch/g1bS_design.md,
  status DESIGN-DRAFT), but there is no lab/g1bS script and no runs/g1bS —
  STATE asserts a dispatch the artifact record does not show. **SOUNDS-WITH-
  SCOPE** (STATE ahead of ledger; claimed-but-missing marker).
- Cadence: heartbeat 10 min fresh; review 100 min stale (R58 = this review,
  discharges it); **novelty 875 min (~14.6 h) stale vs the 2 h bar** — the
  novelty beat is the most overdue automation input; R58 should trigger it.

---

## 4. Double-session era provenance

- g1bW: the concurrent draft's five bugs (F1 KeyError, F2 fabricated g1bR
  constants, F3 None runtimes, F4 idle-deadlocking GPU gate, F5 fp32 == assert)
  are documented in the executed script's docstring (lab/g1bW_second_fact.py:129-156),
  in NOTES.md:21-25, and the orphaned 10:07Z smoke process is explicitly
  quarantined ("its artifacts are not results and were overwritten"). T140:707-709
  adds F2 to W021's scan. **SOUND** — clearly quarantined.
- g3K: lab/g3K_kappa.py is the executed script (matches metrics: e157_f2 host,
  S-DISC offset-0 ruler, iso seeds 11401-3/11411-13, rungs 1-64).
  lab/g3K_kappa_cell.py is the earlier wrong-ruler draft (kappa_host on the
  e185 organism, offset -12 ruler, sub-1 rungs, iso seeds 112xx) — it is
  **untracked, has no SUPERSEDED/DO-NOT-RUN banner, and its docstring still
  reads "Run: python lab/g3K_kappa_cell.py"** while claiming the same
  experiment name. runs/g3K/kappa_cell.png (untracked) is its orphaned
  artifact sitting inside the REAL run's output directory. No NOTES/THINKING/
  QUEUE mention quarantines it. **NOT clearly quarantined** — this is the
  one double-session artifact a future session could mistake for the cell.

---

## 5. opt1 soft spots

- Never-adjudicated labeling: present in every appearance — NOTES.md:63-65
  ("clearly-labeled extrapolation ~379/1174/3228 steps, never adjudicated"),
  T139:737-738 ("SGD clocks only by labeled projection"), QUEUE.md:198
  ("projected only (~379-3228 steps, never adjudicated)"), opt1b QUEUE row,
  metrics tstar extrapolation_flag + honesty_reflex.projections_never_adjudicate,
  and lab/opt1b_sgd_kill.py's own registration. **SOUND**.
- CPU fp32 texture caveat: carried in NOTES.md:82 ("CPU fp32 texture (gated
  on-device via A0)"), T139:738, metrics honesty_reflex.float_texture.
  **Missing** from QUEUE.md:198 and from skeleton item 12 (:285-291) — the
  paper's optimizer clause would ship without the texture/n=1 scoping.
- Skeleton item 12 flags that paragraph (2)'s "at every lr tested" must
  become "under AdamW at every lr tested" — the edit is flagged (:288-289)
  but NOT yet applied at :18. Honest, pending.

---

## 6. Prioritized repairs (anchored old -> new)

R1 (HIGH — numbers that will be quoted forward): fix the opt1
kill-displacement convention mix.
- NOTES.md:69-71 old: "every Adam variant kills at D ~ 2.49-2.84; warmup
  stretched the step clock 10x and the kill still arrived at the same
  displacement within ~15%" -> new: "every Adam variant kills at D ~ 2.49
  at the checkpoint reading (A3-warmup 3.64 checkpoint / ~2.85 at its
  interpolated t_x — a ~15% gate shift only under the interpolated reading;
  the checkpoint shift is +46%); the gate is displacement-typed, its
  tolerance is convention-bound".
- Same correction at THINKING.md:717-719 and QUEUE.md:198 ("the GATE is
  displacement (~2.49-2.84 every Adam arm)" -> "(checkpoint 2.49, warmup
  3.64; interpolated 2.18 vs 2.85 — gate tolerant, not invariant)").
- This matters twice over: opt1b's SGD-SPARED bar (D=2.6 = "15% past the
  Adam gate") is calibrated on the interpolated reading; say so in the bar.

R2 (HIGH — arithmetic): 1687x -> 1683x.
- NOTES.md:68 "1687x at the same lr" -> "1683x at the same lr (1.6543 /
  9.829e-4)"; THINKING.md:716 "— 1687x" -> "— 1683x"; QUEUE.md:198
  "(1687x/step at matched lr)" -> "(1683x/step at matched lr)";
  scratch/day6_paper_skeleton.md:286 "Adam's sign-normalization (1687x/step"
  -> "(1683x/step".

R3 (MED — C13-1 residuals): re-stamp the three surviving "law" mintings.
- QUEUE.md:197 "THE NO-BASIN LAW IS A TRAJECTORY LAW" -> "the no-basin is a
  TRAJECTORY effect (C13-1: PROPOSED hypothesis until e188 + >=2 organisms)".
- scratch/day6_paper_skeleton.md:161 "the trajectory law's nearest prior" ->
  "the trajectory hypothesis's nearest prior".
- THINKING.md:740: add a bracket under T138's title mirroring T137's ("[the
  title's 'law' predates C13-1 by 5 min; read as hypothesis]").
- Optional: NOTES g3K entry takes a one-line pointer "(superseded stamp:
  C13-1 -> trajectory HYPOTHESIS, see T137 amendment)".

R4 (MED — quarantine the wrong-ruler g3K draft): add a SUPERSEDED banner to
lab/g3K_kappa_cell.py line 1 ("SUPERSEDED by lab/g3K_kappa.py (host ruler
corrected e185->S-DISC per the pre-dispatch instrument check); DO NOT RUN;
its outputs are not results"), commit it, and either delete
runs/g3K/kappa_cell.png or move it to runs/g3K/quarantine_kappa_cell.png with
a README line. (Alternatively delete the script if the coordinator prefers —
but silently leaving it runnable is the current hazard.)

R5 (MED — STATE precision): STATE.json:8 "g1bS DISPATCHING (GPU freed)" ->
"g1bS design v1 committed (088ca91); dispatch next (C13-2)" until a
lab/g1bS script + runs/g1bS exist.

R6 (LOW): scope "4-10x": NOTES.md:109 and QUEUE.md:197 "a graded 4-10x
basin" -> "a graded basin (host 4.0-9.7x; store seeds 2.8-6.6x, mean 5.0)";
skeleton:215 "static random displacement is 4-10x more forgivable" ->
"3-10x more forgivable (per-organism means 5.0/6.0)".

R7 (LOW): g1bW metrics clause wording — note in the next NOTES fold that
runs/g1bW/metrics.json adjudication.clause's "the reference (unwalled) leg
installs B at 0.0628" is a wording bug (0.0628 is a FAIL; the clause's
unqualified "the wall is a splint" rescope is what T140 already retracted).

R8 (LOW): carry the CPU fp32 + n=1 scoping into QUEUE.md:198 (append "CPU
fp32, n=1/arm") and into skeleton item 12 before any abstract use; apply the
flagged paragraph-2 edit ("at every lr tested" -> "under AdamW at every lr
tested") at skeleton:18.

R9 (LOW, process): trigger the novelty beat (last_novelty ~14.6 h stale) —
T138's literature pass is 2026-09-30 10:55Z content that a novelty trigger
should already have consumed.

---

## Verdict summary

| claim | verdict |
|---|---|
| g3K kappa pair 5.0/6.0 + iso rungs | SOUND |
| g3K aggregation-robust MIXED | SOUND (host min 4.03 misses <=4 bar by 0.03) |
| g3K "4-10x basin" | SOUNDS-WITH-SCOPE (store seed 2.81) |
| g3K CE-at-kills 1.84-2.62 | SOUNDS-WITH-SCOPE (iso arms; wash kill wrecks, CE 3.79) |
| opt1 1687x step ratio | OVERCLAIMED (actual 1683x; not derivable) |
| opt1 D-at-kill 2.49-2.84 / "within ~15%" | SOUNDS-WITH-SCOPE (convention mix; A3 checkpoint 3.64 in metrics only) |
| opt1 warmup 10.08x; SGD rise; A5; e185 repro; projection labels | SOUND |
| g1bW verdict + all sub-numbers | SOUND (metrics clause wording bug only) |
| C13-1 repair | SOUNDS-WITH-SCOPE (3 "law" residuals) |
| Ledger (opt1b/g1bW2/g1bS) | SOUNDS-WITH-SCOPE (STATE ahead of g1bS artifacts) |
| Double-session quarantine | g1bW SOUND; g3K draft NOT quarantined |
| opt1 soft spots | SOUND (fp32 caveat missing from QUEUE + skeleton) |
