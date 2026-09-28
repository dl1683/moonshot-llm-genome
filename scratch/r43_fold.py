import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- REVIEWS.md ----------
r = open("REVIEWS.md", encoding="utf-8").read()
anchor = "## Review 42 — the graduation-and-verification window"
entry = """## Review 43 — the re-keying ambush (2026-09-28T06:55Z; covering 05:50–06:55Z; e119 dispatched mid-review, RUNNING)

Context: E120/T075/W008 freshest; e119 (migration head-to-head) dispatched at
05:50Z and running on GPU through this review.

### AUDITOR (agent failed at 95s — model request error; audit performed directly by lead, resilience clause)
- Numbers: e109/e113/e116/e120/e121 headline values all verified against
  metrics.json (2 apparent misses were rounding: 22.487→22.5, 0.0321→0.033).
- Ledger debt found AND repaired in-beat: E019/E078/E088 late-folded (orphaned
  runs with registered rules but no NOTES entries). e078 replicates T047's
  row-129 rebind at 4x dose (0.97 ratio, both k); e088 pair-removal is
  SUB-additive (median 0.464) — overlapping redundant supports, the texture
  W008's adapter family implies; e019 energy-carrier held via the e011c rule.
- 2 stale READY rows (e065, e083) marked; combined-heading false positives
  cleared (E035+E038, E092+E104, E101+E106 covered).
- NEW audit finding: NO model weights are persisted anywhere in the repo
  (only an unrelated e055 scratch cache). The critic's "eval-only on existing
  checkpoints" discriminator is therefore NOT free — nets must be regenerated
  (deterministic recipes exist). Process fix adopted: every future experiment
  saves phase-boundary checkpoints under runs/eNNN/ckpts/ (small, loadable).

### IDEATOR (full 7-experiment list folded into QUEUE as e132–e138)
Top pick: the wiring trace (kernel-motion x brake x self-acceptance at dense
checkpoints of one jitter schedule — W008's own falsifier instrument, upgraded
to a three-dial conjunction with temporal-order predictions). Also: field
anatomy census (free-rides e119's twins), two-facts-one-field (first
multi-fact ecology; share-law quantitative prediction), LN-causality variant
(W001/W007's deferred causal test), dream-protection decomposition (the sole
positive self-effect), RMU rewiring speed (bridges edit-law and consolidation
programs), adapter head-start (e120's row-183 promoted to a discriminator).

### CRITIC — accepted nearly in full; this review's center of mass
1. T075 headline attacked (HIGH): ERROR-LOCATION counter-theory — every arm
   consolidated where its training error lived; the battery reads only the
   121–137 band and row 183 was NEVER read; plus loss-mask mismatch (a/b/c
   full-CE vs d name-only) and locked-replay already doubling survival
   (diversity is an amplifier, not a switch). ACCEPTED -> T075 second
   amendment: headline downgraded to multiplier-language; corpus>self
   downgraded to SUGGESTIVE (CI prices prompt-sampling only; arm a's
   within-run range 0.02–0.18; both arms below base floor = damage regime).
2. W008 timeline (HIGH): phantom leg — e109's "fatal D-all" was
   D0129={0,129} (window-scaffold confound T065 itself flagged); e113's
   "survived D-all" deleted 5-of-17 band rows on a bit-exact rebuild of the
   SAME net. Depths retold as timepoints; zero stage x depth cells exist.
   ACCEPTED -> W008 corrected: maturation RETRACTED pending a real cell.
3. Most damaging assumption: D-all survival != fact left the wpe system.
   Row 0 (content-carrying in 6/6 installs per T069) never content-tested
   post-consolidation; 12 band rows left intact in e113; no out-of-band row
   ever scanned. BODY-STORED vs ADDRESS-MIGRATED-ELSEWHERE is OPEN. ->
   e131 RE-KEYING CENSUS dispatched (regenerate + probe: 183-geometry read
   on e120 a/b arms; row-0 content test; band-minus-row-0 deletion with
   scaffold-matched control; full-512 wpe delta census). It gates the
   reading of e119, e122, e125 and both developmental arrows (W005/W008).
4. Queue honesty bars adopted (pre-registered): e123 must control against
   trivial output-similarity drift (rule 7a); e122 pre-commits the
   same-run-different-window falsifier; e125 needs collateral-matched
   specificity vs the pre-consolidation fact's removability; e128 needs the
   e095 Monte-Carlo null guard on apparent clustering.

### Decisions
1. Ledger corrections landed BEFORE the dependent dispatch (THINKING gate
   held): T075 second amendment, W008 correction, this entry.
2. e131 dispatched CPU-only, parallel with e119's GPU (envelope: no
   concurrent GPU — honored).
3. Ideator list ranked in as e132–e138; wiring trace (e132) takes the next
   GPU slot after e119.
4. Process: checkpoint-discipline instruction added to all future taskings.
5. Ratio this window: thinking-heavy (one earned experiment dispatch e119,
   one cadence review, three direct audit/repair commits) — BOTH LANES held.

---

"""
assert anchor in r, "REVIEWS anchor"
r = r.replace(anchor, entry + anchor, 1)
open("REVIEWS.md", "w", encoding="utf-8").write(r)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
o1 = "Designed on paper; ripening.\n"
n1 = """Designed on paper; ripening.

SECOND AMENDMENT (R43 critic — accepted, ~06:55Z): the ERROR-LOCATION
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
"""
assert o1 in t, "T075 anchor"
t = t.replace(o1, n1, 1)

o2 = "Ripening; dispatch when e119's census lands.\n"
n2 = """Ripening; dispatch when e119's census lands.

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
"""
assert o2 in t, "W008 anchor"
t = t.replace(o2, n2, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE.md ----------
q = open("QUEUE.md", encoding="utf-8").read()
anchor2 = "| e119 | migration head-to-head (P5 road 1v2) | READY |"
newrows = """| e131 | RE-KEYING CENSUS (R43 critic's discriminator — gates the READING of e119/e122/e125 and the W005/W008 arrows) | DISPATCHED 06:55Z (CPU, parallel w/ e119 GPU) | regenerate e120 a/b arms + e113-style consolidated net (deterministic recipes), then: (1) 183-geometry read on splice arms; (2) row-0 content test post-consolidation (e116 instrument); (3) band-minus-row-0 deletion vs scaffold-matched control; (4) full-512 wpe delta census. Bars: RE-KEYED fires if row-0 (or any out-of-band row) carries fact content post-consolidation, or band-minus-row-0 collapses where 5-row D-all didn't, or census finds a new positional home; BODY-STORED-GENUINE fires if row-0 test null, band-minus-row-0 survives like D-all, census clean. Saves ckpts under runs/e131/ckpts/ (R43 process fix) |
| e132 | the wiring trace (ideator top pick = W008's falsifier) | QUEUED — next GPU slot after e119 | one 300-step jitter replay, ~10 dense checkpoints, three dials: kernel-motion cos (e107 instrument), brake delta (e115), k=7 self-acceptance of t0 field; D-all survival per checkpoint. Registered: kernel-fact cos >=2x step-0 by step 100, plateaued by 200; brake crosses +0.05 only post-plateau; D-all survival monotone (rho>=0.8). Falsified by brake-leading-kernel, D-all jump, or self rekeying at the wiring event. Also delivers the critic's mid-replay stage x depth cell |
| e133 | field anatomy census (per-organ ablation on graduated vs address-phase twins — free-rides e119) | QUEUED | >=60% causal load post-graduation in non-address organs (W008) vs >=60% address-adjacent (address-phase mirror); W005-alt address-spreading killed if grown-row attention <50% |
| e134 | two facts, one field (first multi-fact ecology) | QUEUED | ADDITIVE: F1's r*(k).k within per-net band (max/min<1.2) with F2 consolidated; SHARED: >=25% drop. Riders: F2-at-F1's-old-address interference; brake-tag fact-vs-address specificity (T067/T071) |
| e135 | LN-causality variant (W001/W007 deferred test) | QUEUED | norm-variant twin: share product destabilizes (max/min>2) or collapse boundary vanishes => LN CAUSES the field physics; within 1.2 => LN decorative |
| e136 | dream-protection decomposition (e121's sole positive self-effect) | QUEUED | surprisal-carried: matched lowest-surprisal corpus reproduces within 1.5x; generator-carried: dreams protect >=3x at matched loss |
| e137 | RMU rewiring speed (bridges edit-law x consolidation) | QUEUED | restoration <=150 steps with 121-137 band retained (adapters dormant, not dead); falsified if >=250 steps or band regrows from <0.05 |
| e138 | adapter head-start (e120's row-183 net vs fresh twin, identical jitter) | QUEUED | head-start: pre-grown adapter wires in <=60% of fresh steps; within 10% => wiring is everything, mass-growth not the bottleneck |
""" + anchor2
assert anchor2 in q, "QUEUE anchor"
q = q.replace(anchor2, newrows, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE.json ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_review"] = now
s["last_novelty"] = now
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e119 (GPU, migration head-to-head) + e131 (CPU, re-keying census — R43 discriminator)."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)

print("R43 fold complete")
