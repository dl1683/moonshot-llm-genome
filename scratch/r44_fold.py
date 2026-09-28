import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- REVIEWS.md: R44 entry ----------
r = open("REVIEWS.md", encoding="utf-8").read()
anchor = "## Review 43 — the re-keying ambush"
entry = """## Review 44 — the sink-role counterattack (2026-09-28T07:40Z; covering 06:10–07:40Z; e139 + e133 running throughout)

### AUDITOR — VERDICT: ISSUES FOUND (no fabrication; every headline number
traces to metrics; pre-registration integrity CONFIRMED via git hashes:
W010 P1/P2/P3 at 479d816/06:47:52Z precede e119 results d0395ee/07:00:49Z;
e131 bars landed with the R43 fold 414860f/06:06Z before results 31e520c/06:53Z).
Findings, all repaired this beat: (1) e133 dispatch's ledger lag (dispatched
07:08Z, stamped QUEUED — the lead's bookkeeping miss, now DISPATCHED);
(2) three DONE queue rows asserting dead claims (e083/e113/e120 — markers
added); (3) seven rows needing row-0-frame bar updates (e123/e125/e132/
e133/e134/e137/e138 — updated; e138 RETIRED: its premise died with probe 1);
(4) NOTES E119 paired R's cross-geometry max 0.709@g-8 against E's g+0
(corrected to matched 0.663 vs 0.071); (5) 'the ONLY content-positive row'
was false on the registered criterion (rows 118/119 at control level —
phrasing corrected in NOTES and T077); (6) hygiene: e139 script untracked
(added), T075/T065/T073 header markers, T073-T075 clock repairs (~4h drift),
parking-lot table jam fixed.

### IDEATOR — 7 candidates ranked; queue updated
Top pick e141 (what kind of key is the sink: presence-vs-content scramble
dial + gate-vs-source interpolation, eval-only minutes). e142 row-0-at-birth
install-dose census (11 saved nets — rewrites the origin story: ADDRESS-ONLY-
EVER vs ROW-0-ALWAYS vs HUB-FIRST). e143 error-placement steering (NEAR/FAR/
jitter — the causal test of T076; claims e132's training slot). e134
sharpened into hub-bandwidth (W012 born from this: is 54 the sink's
capacity?). e125 re-scoped to three surfaces (key/band/brake — the brake is
an 'unlearning' move that STRENGTHENS). e144 frozen-sink install, e145
family-universality — sequenced after e142/e139. Queue hygiene adopted:
e138 retired, e132 demoted.

### CRITIC — the center of mass; nearly all accepted
1. ATTACK 1 (HIGH) — ROLE VS WRITTEN KEY, and the crack is ALREADY IN HAND:
   e131's census shows row 0's consolidation delta is the SECOND-SMALLEST
   of 256 rows (delta_norm 0.0557 vs band median 0.1265) with fact-axis
   projection at the band median (0.0124 vs 0.0108) — consolidation wrote
   nothing fact-specific INTO wpe[0]; the 0.545->0.732 strengthening lives
   in READOUT WEIGHTS. 'Re-keyed to row 0' may be 'the read policy became
   sink-ROUTED' (role necessity, not key storage). 'Two independent
   controls' overstated — mean-replacement is the same direction-scramble.
   ACCEPTED -> T077 second amendment; install-restore surgery + rows-2-6
   hardening dispatched in e141.
2. ATTACK 2 (HIGH) — retirement premature: probe 1 proves learning-at-183,
   not graduation (the D-183 survival cell is e139's, in flight). ACCEPTED
   -> T075 marker softened to RETIRED-PROVISIONAL.
3. ATTACK 3 (MED-HIGH) — E~=L on every loaded outcome (dall 0.190/0.191,
   geometry 0.088/0.076, brakes both negative): the E-vs-R contrast IS the
   L-vs-R contrast (position diversity, again); 'erasure digs in' demoted
   to CONFOUNDED (P3 thinning is confounded with cumulative cycle damage).
   ACCEPTED -> T078 amendment; L-CYCLED control added to e140.
4. ATTACK 4 (MED) — content-keyed alternative alive (e116's un-killed
   residue); the missing cell is d_r0 at g-12 on R@150/R@300 (added to
   e141). The 500x dream number unreached by the frame: e136 gets the
   pre-registered surprisal prediction (protection scales with fact-token
   surprisal mass, not self-generation).
5. ATTACK 5a (MED) — R arm's share product = 102.4 at k=128: the constant
   DOUBLED on the jitter line, unremarked. ACCEPTED -> W012 amendment
   (bandwidth reading gets direct input).
6. ATTACK 6 (LOW-MED) — census condition 3 is vacuous (81/256 rows clear
   the floor); RETIRED from the verdict's support; conditions 1+2 carry it
   (and they are not independent witnesses — see attack 1).
7. ATTACK 7 (MED) — one control row cannot bound the scaffold; rows-2-6 +
   norm-matched random deletions added to e141.
R43 ADJUDICATION: one over-correction — T075 retired too eagerly (mask
confound never discharged; retirement announced pre-graduation-cell);
otherwise correctly handled (W008 retraction, e131 dispatch, queue bars).

### Decisions
1. All audit repairs + critic corrections applied BEFORE the dependent
   dispatch (gate held).
2. e141 DISPATCHED (CPU eval-only, merged mechanism battery: install-restore
   surgery with t-curve; presence-vs-content; rows-2-6 + norm-matched
   controls; d_r0 at g-12 on e119's R checkpoints; gate-vs-source).
3. e140 gains the L-CYCLED rider (locked-replay cycles, no reset — does L's
   dall thin like E's?); e136 gains the surprisal pre-registration.
4. Frame status: T077 bounded (role-vs-key open), T078 demoted (confounded),
   T075 provisional, W011 amended (content-keyed alive), W012 amended
   (102.4 anomaly). The lab's honesty machinery caught its own second
   overclaim in one morning — the critic's in-hand crack (unread census
   cell) is the review system working.
5. Fleet through this window: e139 (CPU) + e133 (GPU) + e141 (CPU) — full.

---

"""
assert anchor in r
r = r.replace(anchor, entry + anchor, 1)
open("REVIEWS.md", "w", encoding="utf-8").write(r)

# ---------- THINKING.md amendments ----------
t = open("THINKING.md", encoding="utf-8").read()

# T077 second amendment — anchor: end of its "(3) W010 SEED-AND-AMPLIFY..." paragraph's card closer
o1 = """STANDING PRE-REGISTRATIONS: W010's P1/P2/P3 vs e119 are MOOT as
written (they assumed R-vs-E differences around a field concept
that just collapsed; P2's census comparison survives as texture)."""
n1 = """SECOND AMENDMENT (R44 critic — accepted, ~07:40Z; the review's
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
that just collapsed; P2's census comparison survives as texture)."""
assert o1 in t, "T077 2nd amendment anchor"
t = t.replace(o1, n1, 1)

# T078 amendment — anchor its closing open-edges paragraph
o2 = """The wiring trace (e132) demotes to optional:
row-0 growth across checkpoints answers its kernel question more
directly and eval-only."""
n2 = """AMENDMENT (R44 critic, ~07:40Z): "ERASURE DIGS IN" DEMOTED TO
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
directly and eval-only."""
assert o2 in t, "T078 amendment anchor"
t = t.replace(o2, n2, 1)

# W011 amendment
o3 = """(c) The sink was ALREADY the fact's co-carrier at install
(T069's 6/6 content-carrying, strength 0.545): consolidation did
not build the row-0 key from nothing — it PROMOTED the
already-largest seed. Error-location said WHERE error
consolidates; W011 says WHERE the key GOES when the error is
everywhere: to the row that is always attended."""
n3 = """(c) The sink was ALREADY the fact's co-carrier at install
(T069's 6/6 content-carrying, strength 0.545): consolidation did
not build the row-0 key from nothing — it PROMOTED the
already-largest seed. Error-location said WHERE error
consolidates; W011 says WHERE the key GOES when the error is
everywhere: to the row that is always attended.
AMENDMENT (R44 critic): the CONTENT-KEYED alternative is alive —
e116's un-killed residue (readout keys on content, routing
perpendicular) gives geometry-independence with no row-0
involvement; the missing cell is row-0 deletion at NOVEL geometry
on the R arm (d_r0@g-12 on R@150/R@300 — dispatched in e141:
sink-keying dies, content-keying survives). And the ROLE reading
survives W011 either way: omnipresence is a property of the
POSITION, whether or not wpe[0] carries a written key."""
assert o3 in t, "W011 amendment anchor"
t = t.replace(o3, n3, 1)

# W012 amendment
o4 = """answer (bandwidth vs d_model, vs sink attention mass, across the
e098 family ladder)."""
n4 = """answer (bandwidth vs d_model, vs sink attention mass, across the
e098 family ladder).
AMENDMENT (R44 critic's attack 5a, in hand and unremarked): the
R arm's share product reads 102.4 at k=128 — the constant DOUBLED
on the jitter line. No card had noticed. Under the bandwidth
reading this is either the sink's effective share doubling under
consolidation or the k-grid boundary moving; either way W012 now
has an in-hand anomaly to explain, and e134's two-fact grid gets
a sharper question: does adding F2 push the product BACK toward
54 (renormalization ceiling) or UP past 102 (bandwidth grew with
the route)?"""
assert o4 in t, "W012 amendment anchor"
t = t.replace(o4, n4, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE.md: e141 dispatched, e140 rider, e136 prereg ----------
q = open("QUEUE.md", encoding="utf-8").read()
o_q = "| e140 | ROW-0 GROWTH TRACE"
e141 = """| e141 | SINK-KEY MECHANISM BATTERY (R44 critic attacks 1/4/7 + ideator e141 merged) | DISPATCHED 07:40Z (CPU eval-only; stagger behind e139) | On e131_consolidated + e048_repro + e119 R@150/R@300: (1) INSTALL-RESTORE SURGERY — swap consolidated wpe[0] <- install wpe[0] with t-interpolation dose curve (WRITTEN-KEY: fact dies with delta removed while CE flat; ROLE-ROUTED: survives t=1, dies only under direction-scramble); (2) PRESENCE-VS-CONTENT — scramble first 1-16 tokens of eval contexts vs mid-context matched control (PRESENCE-KEY <20% cost; CONTENT-KEY >=50% beyond control); (3) SCAFFOLD HARDENING — delete rows 2-6 individually + norm-matched random, CE_R + expression each (only-row-0-wrecks => sink uniqueness); (4) d_r0 at NOVEL geometry g-12 on R@150/R@300 (sink-keyed dies / content-keyed survives); (5) GATE-VS-SOURCE — graded interpolation of wpe[0] consolidated->install, expression vs r (SOURCE-GRADED R^2>=0.8; GATE-THRESHOLD >=80% retained to r* then collapse) |
""" + o_q
assert o_q in q
q = q.replace(o_q, e141, 1)

o_e140 = "PARTIAL if E keys late |"
n_e140 = "PARTIAL if E keys late. [R44 rider: L-CYCLED arm — 3 locked-replay cycles with battery structure, NO row reset; if L's D-all thins monotonically like E's, 'erasure digs in' loses its only erasure-specific evidence (cycle damage explains it)] |"
assert o_e140 in q
q = q.replace(o_e140, n_e140, 1)

o_e136 = "generator-carried: dreams protect >=3x at matched loss |"
n_e136 = "generator-carried: dreams protect >=3x at matched loss. [R44 pre-registration: protection scales with fact-token SURPRISAL mass, not self-generation — the frame's extension: error-presence pins, error-absence frees; the 500x number is otherwise unreached by the row-0 frame] |"
assert o_e136 in q
q = q.replace(o_e136, n_e136, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE.json ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["last_review"] = now
s["last_novelty"] = now
s["current_experiment"] = "Fleet 3: e139 (CPU, universality) + e133 (GPU, anatomy) + e141 (CPU, sink-key mechanism battery — role-vs-written-key). R44 folded: T077 bounded, T078 demoted-confounded, T075 provisional."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("R44 fold complete")
