# R83 AUDITOR — the two-cell window (2026-10-11T03:41:12Z, datetime.now(UTC))

Window: R82 (33c0155, 01:05Z) -> HEAD (2ceaa58, 03:27Z), 25 commits, two cells.
Every headline traced key-by-key into runs/*/metrics.json + REPORT.md; birth chains
by `git show --stat`; registration integrity by literal-eval diff of the REGISTERED
dict, birth commit vs final script.

## 1. Birth chains, parity, bars — SOUND

- x52: birth f7ff8e9 01:23:43Z (script only, 2401 ln) -> smoke e65028f -> COMPLETE
  7a9bd76; metrics date 01:29:43Z (post-birth). e346: birth b92f096 02:10:44Z ->
  hash backfill 1fd41aa (BIRTH_COMMIT=b92f096, pre-compute) -> smoke 1c3e231 ->
  ADJUDICATED 06a8760; metrics date 03:08:21Z. Artifacts first appear at COMPLETE.
- REGISTERED dict BYTE-IDENTICAL birth->final in BOTH cells (ast.literal_eval
  compare: x52 7232 chars, e346 7608 chars) — no bar/parity/P-card edit survived
  any smoke repair, thermal repair, or honesty patch. metrics.registered ==
  birth literals on every key (x52 7/7, e346 6/6).
- Bars honored: x52 rescore_verdict.walk_bar {pooled_ok: true, stratum_ok: true,
  clearing_strata: ["C"]}; e346 composite order frozen at birth (REPORT.md line
  21: TEXTURE->NON-MONOTONE->THRESHOLD->GRADED->honest shapes), NEVER-PROTECTS
  fired from the honest-shapes tier, fate cuts DEAD<=0.10 / ALIVE>=0.2859 with
  k=7 scored by the frozen min-over-rungs. No bar shopping.
- NICK: x52's metrics.birth_commit = e65028f (runtime head at launch = the smoke
  commit), not f7ff8e9; e346 pinned its true birth via the backfill. Harmless
  here (REGISTERED byte-identical across both), but the weaker pin should not
  become the convention. Receipts: runs/x52/metrics.json birth_commit;
  lab/e346_name_key_ladder.py:2641.

## 2. Headline numbers — SOUND (all exact at source)

- x52 strata: r_g-12|g0 -/0.8333/0.42389 and r_g0|g+12 -/0.8333/0.524615, 12/12
  in-window (metrics.rescore.walk.scored, .n_in_window); stratum C n=9 at
  0.8889/0.4278 + 0.8889/0.5048 (strata_scores.C); A/B n=3 at 0.6667; REPORT.md
  lines 3, 31, 34 match. Rider: dense 60 pairs r1 +0.50 / r2 -0.6667, full 40
  pairs +0.525/+0.55, verdict RULER-INTERNALLY-COHERENT
  (rider.adjudication.ruler_internal.*); anneal co-report -/0.85/0.307 and
  -/0.825/0.279 n=40 (rescore.anneal.scored); the two positive pairs are exactly
  walk_cons_s25/swap_s175 + walk_cons_s50/zeph_s200 (rescore.walk.pairs); the
  95-step oscillation = rider.curve emergence window [10,105] shape OSCILLATION
  (REPORT.md:72); P-x52a hit (composite.P-x52a.hit true); shape sub-prediction
  half-miss disclosed in catches[1]. Gates 9/9, CPU desk envelope.
- e346: k-curve 0.000473 -> 0.010666 -> 0.069899 -> 0.091134 -> 0.099412 ->
  0.309011 strictly monotone (adjudication.ladder[k].score, k=0 e345-committed /
  k=16 x45-ruler via G_ENDPOINTS); k7 knife-edge 0.10-0.099412=0.000588, per-rung
  ALIVE/DEAD/ALIVE (0.4348/0.0994/0.2994 vs cuts 0.2859/0.10). ZxT EXECUTES:
  p(Z) 6.70e-4/6.72e-4/3.93e-4, guest p(T) 0.378->0.673->0.7814. Neutral
  EXECUTES: p(T) 1.86e-3/1.25e-4/5.62e-4, guest p(Q) 0.513->0.505->0.692.
  NAME-BLIND: five-arm w_spread 0.001-0.0688, room_spread <=0.003963; ZxT arm
  14.225->21.558 / in_room 0.0602 == walk_s400 class 14.1/0.060
  (adjudication.zxt/.neutral/.rider_name_blindness). Slot-split co-report: guest
  p(Z) s300 = 0.6573/0.6388/0.6042 (k<=4) vs 0.4072 (k7). s1 micro-ladder host
  0.0063-0.0141 vs ruler 0.7490. Gates 15/15; G_DRAWS 300 draws bit-equal +
  first_ix == e345; G_XDEVICE |d| 2.37e-08..1.01e-05. All in REPORT.md lines 3-36.

## 3. THE THERMAL EVENT — SOUND-WITH-REPAIRS

- Breach numbers exact: envelope bursts k7.b1/b2/b3 = 85.0/86.0/86.0C, zxt.b2
  87.0C (t_start 02:33-02:36Z, aborts firing at 84-85C under 10-step polls);
  commit 4784dd1's "k7 burst3 86.0 / zxt burst2 87.0" matches metrics verbatim.
- Response verified in code + envelope: run STOPPED at zxt s200; phase-2 guard
  (cap 150->90s, cooldown 40->60s, abort 83.5->82.0C, launch 80->70C, poll every
  step) applied to POST_BREACH_ARMS=("zxt","neu") only; neu.burst1 [0,300]
  max 79.0C, cooldown 60, envelope_phase 2 — the guard demonstrably worked.
- No adjudicative content touched: REGISTERED byte-identical (item 1); the full
  post-smoke diff is guard code + deviation text + display-key repairs
  (zz.get(300), str-key coercion) + the NEVER-PROTECTS clause-text patch — no
  bar, arm, stream, gate criterion, or scored path.
- DEFECT (the repair): the restart's zxt-remainder phase-2 burst is MISSING from
  the committed envelope. Reconstruction: 5 passes (run.log lines 1/148/268/365/
  462); pass 2 reconciled 12 prior records, then appended its zxt remainder as
  tag "zxt.burst1" [200,300] phase 2 (run.log: launch polls 59C OK, "replay
  finished ... 1 burst(s), max temp 68.0C"); pass 3's dedupe-by-tag dropped it
  (tag collides with prior zxt.burst1 [0,50]) — "reconciled 13" thereafter. So
  the committed envelope holds 12 phase-1 + neu only, and the deviation's "the
  committed envelope carries both phases verbatim" is one record short; the zxt
  remainder's phase-2 evidence survives only in run.log. RECOMMENDED TEXT (for
  the next ledger touch, metrics.catches or NOTES e346): "ENVELOPE REPAIR
  DISCLOSURE: the restart pass's zxt-remainder phase-2 burst record (steps
  [200,300], max 68.0C, cooldown 60s) was dropped by the reconciliation's
  tag-collision dedupe (its tag zxt.burst1 collided with the prior pass's
  [0,50] record); runs/e346/run.log retains the launch polls and the finish line
  verbatim; no adjudicative content involved."
- NICKS: (a) commit 4784dd1 says "cooled 87->57C idle" vs the registered
  deviation/script "87->71C" — two different idle observations, narrative only;
  (b) the 12 phase-1 records lack the envelope_phase field (it postdates them;
  cooldown_s 40 is the phase-1 signature) — worth one clause in the same repair.

## 4. Fold edits — SOUND

- G3-dead clause (1a3d96e, THE_LAWS_V3.md ~255): numbers 0.833/12/12/
  stratum-cleared + 95-step oscillation all match item 2; class caveat carried
  (warm-vs-cold residue) exactly as the one-sided clause requires.
- Maintained-vs-not re-word (1b71d4c, ~248): grounded in the neutral + ZxT
  executions; "the walk, n=1, now isolated as THE OWN-KEY PHENOMENON" matches
  T336's composite. T335 (THINKING.md:53) and T336 (:10) both present with
  registered prediction P-T336a (lean + counter + discriminator).
- METHODS 5 (THE_LAWS_V3.md:306-311): 23-for-43 -> 24-for-44 (x52 fold) ->
  24-for-45 (e346 fold); "fourteen cells since R80 went 6-for-14" arithmetic
  exact.

## 5. Ledger re-derivation — SOUND

23-for-43 (R82) + x52 P-x52a HIT = 24-for-44; + e346 = 45th total, head missed:
the LEAN's registered head was SHARP/THRESHOLD (P_e346a's own ladder guess was
GRADED against it) and NEVER-PROTECTS is the honest fourth shape outside both —
head scores 0, the ZxT-EXECUTES rider hit is disclosed in the parenthetical:
"e346 (its SHARP head missed, the ZxT rider hit, disclosed)". X51 HEAD-CONVENTION
CONSISTENT: the same line counts "x51 (primary; rider missed, disclosed)" as a
hit — both directions count the head and disclose the rider split; e346 is the
mirror image of x51, not a new rule. QUEUE.md:201 and NOTES.md:23 carry the same
split. No double-count, no rider laundering.

## VERDICT

x52 SOUND; e346 SOUND (with the thermal-envelope repair owed); folds SOUND;
ledger SOUND. WINDOW: SOUND-WITH-REPAIRS (1 repair: the dropped zxt phase-2
envelope record, recommended text above; 3 nicks: x52's birth-commit pin, the
57C/71C narrative, the phase-1 field gap). No e166-class invalidity, no bar
shopping, no phantom — the honest fourth shape (NEVER-PROTECTS) was scored
against frozen bars that did not anticipate it, exactly as registered.
