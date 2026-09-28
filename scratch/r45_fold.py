import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- REVIEWS.md R45 ----------
r = open("REVIEWS.md", encoding="utf-8").read()
anchor = "## Review 44 — the sink-role counterattack"
entry = """## Review 45 — the flat-CE ultimatum (2026-09-28T09:20Z; covering 07:40–09:20Z; e142 + e143 + e147 running through it)

### AUDITOR — ISSUES FOUND (2 significant, 3 minor), ALL REPAIRED IN-BEAT
40+ number traces verified across e133/e139/e140/e141. Significant: (1) the
T082 addendum's pre-registration is NOT git-verifiable (first appearance
08:31Z postdates the data commit 08:28Z) — downgraded in T083 to
asserted-unproven (mitigant: it failed and was recorded); (2) e143 complete
on disk but uncommitted (agent's completion notice never arrived) — committed
this beat. Minor: '+6%' was actually +4.1% (fixed in all five spots); W013
asserted the dead T079 clause (marker added); NOTES newest-first order broken
by late folds (E091/E119 relocated). Positive: e143's proximity-vs-invariance
pre-registration IS git-verified (cc9fc8d 07:58:34Z precedes all compute).

### IDEATOR — 7 candidates; e147 (width ladder) dispatched with bars
pre-registered in T084; route-vs-scar surgery (e149), dream-topology census
with randomized harvest (e148 — the 130-char prompt confound), e146
sharpened (4th outcome SELF-INDEPENDENT + dose columns + novel-geometry
primary), e145 promoted (replication backbone), e136 redesigned (position x
source 2x2). Second-paper structure registered: 4 claims + the ROAD->TYPE
plate; missing for submission = replication seeds (e145), the width
dose-response (e147), dream confound discharge (e148).

### CRITIC — the sharpest attack of the day; accepted in full
1. (HIGH) TAXONOMY CONFOUNDED: the routed-vs-site-stored discriminator
   crosses net lineages AND trained-vs-novel status; no single net holds
   both types (P-b unrun; splice-at-novel-geometry under D-r0 missing);
   'two memory types' may be 'two training protocols' until e147/e150 land.
2. (HIGH) THE FLAT-CE ULTIMATUM (the frame-breaking assumption): the lab
   owns NO row-0-plane intervention that kills the fact without wrecking the
   LM — every killing cell sits at CE +0.70 to +4.44. 'Routed through row-0
   presence' and 'dies whenever the net dies' are observationally equivalent
   except the perm spare, which is itself indistinguishable from 'the fact
   never consults row-0's direction.' Cures named, cheap: perm@novel-geometry,
   forced-off-sink mask, L0H3-class head ablation (the only flat-CE
   fact-kill candidate, 0.46 drop at 0.21 CE). -> e150 DISPATCHED.
3. (MED) T081's install-restore was a NO-OP BY NORM (0.7640 vs 0.7695, 0.7%
   change — the probe had no power against the norm-key hypothesis); the
   presence conclusion survives on the RIDERS (perm +4.1%, halfnorm, mean)
   not the registered primary. Norm threshold lives in (0.066, 0.382)
   unmeasured -> norm ladder in e150. T081 amended.
4. (MED) W014's tenant framing has a never-consults null (the fact never
   occupies position 0; scramble trivially spares an unconsulted direction)
   -> fact-at-position-0 control in e150. W014 amended.
5. (HIGH for the claim) DREAMS: the harvest note CONCEDES the artifact —
   130-char prompts ending at host-name positions make name-first
   continuations land at col-130 BY CONSTRUCTION. 'Visits its fact in its
   own coordinates' unsupported; erosion number (paired base) stands. e148
   must run before the claim travels. DAY_SIX already bounded.
6. T075 GOALPOST NOTE (attack 7b, accepted): the retirement quietly
   redefined 'consolidation' from deletion-survival (the original E120 bar,
   which the splice arms still FAIL at their site: D-183 -54%/-31%) to
   learns-and-generalizes. Stated as such in T075's marker now.
7. R44 adjudication otherwise faithful; the g-12 cell became e141's
   strongest result; the saturation-immunization risk (2a) is acknowledged
   in T083's downgrade.

### Decisions
1. e150 — THE FLAT-CE ROUTE TEST — dispatched (CPU eval-only): perm@g-12 +
   perm@g0; forced-off-sink attention mask; L0H3-class head ablation;
   fact-at-position-0 scramble; norm ladder (0.07/0.15/0.25/0.35). It alone
   decides whether 'routed' is a memory property or a wreck artifact.
2. Ledger amendments applied BEFORE dispatch (gate held): T081 (no-op-by-
   norm + riders), T082 (catastrophe-regime confound), W014 (never-consults
   null), T075 (goalpost statement).
3. NOTES ordering restored; e143 artifacts in version control.
4. Fleet through this window: e142 (CPU) + e147 (GPU) + e150 (CPU).

---

"""
assert anchor in r
r = r.replace(anchor, entry + anchor, 1)
open("REVIEWS.md", "w", encoding="utf-8").write(r)

# ---------- THINKING amendments ----------
t = open("THINKING.md", encoding="utf-8").read()

o1 = "STANDING: e143 (in flight) now carries the invariance question's"
n1 = """SECOND AMENDMENT (R45 critic — accepted, ~09:20Z): THE
CATASTROPHE-REGIME CONFOUND. The lab owns NO row-0-plane
intervention that kills the fact at flat CE — every killing cell
sits at CE +0.70 to +4.44 (zero 1.40, mean 2.00, d_r0@g-12 1.36).
'Routed through row-0 presence' and 'dies whenever the net dies'
are observationally equivalent in every measured cell except the
direction-perm spare — which is itself indistinguishable from
'the fact never consults row-0's direction' (its onset never sits
at position 0; no head is sink-adjacent >= 0.25). ALSO: the
taxonomy's discriminator crosses lineages and trained-vs-novel
status (no single net holds both types; P-b and splice-at-novel-
geometry unrun) — 'two memory types' vs 'two training protocols'
hangs on e147 + e150. The cures are dispatched as e150: perm at
novel geometry, forced-off-sink, L0H3-class head ablation (the
only flat-CE fact-kill candidate), fact-at-position-0, norm
ladder. Until e150 lands, ROUTED carries this bound explicitly.

STANDING: e143 (in flight) now carries the invariance question's"""
assert o1 in t, "T082 2nd amendment anchor"
t = t.replace(o1, n1, 1)

o2 = """REMAINING OPEN: the route's anatomical finish (e133's L0H3 +"""
n2 = """PROBE-POWER AMENDMENT (R45 critic — accepted): the install-restore
t-curve was a NO-OP BY NORM (consolidated 0.7640 vs install 0.7695
— a 0.7% norm change; the probe had no power against norm-keying).
The presence conclusion stands on the RIDERS — direction-perm
(+4.1%, norm kept, direction destroyed), halfnorm (0.382
survives), mean-replace (0.066 norm, kills) — not on the
registered primary probe; the '3 corroborating votes' count
included the no-op. The norm threshold lives somewhere in (0.066,
0.382), unmeasured until e150's ladder.

REMAINING OPEN: the route's anatomical finish (e133's L0H3 +"""
assert o2 in t, "T081 amendment anchor"
t = t.replace(o2, n2, 1)

o3 = "## W014 — WONDER: the memory layer is a semi-independent tenant — it dies to what the corpus ignores and ignores what the corpus dies to (2026-09-28 ~08:00Z)"
n3 = "## W014 — WONDER: the memory layer is a semi-independent tenant [R45 caveat: the never-consults null is alive — the fact never occupies position 0, so direction-scramble trivially spares an unconsulted direction; the fact-at-position-0 control (e150) decides] (2026-09-28 ~08:00Z)"
assert o3 in t
t = t.replace(o3, n3, 1)

o4 = "## T075 — [RETIRED, RESOLVED by e139/T082: the splice arms learned, stored (row-183 content ~1000x), and generalized (0.6-0.7) at their error site — retirement stands; what position diversity actually does (choose routed vs site-stored) belongs to T079]"
n4 = "## T075 — [RETIRED, RESOLVED by e139/T082 — GOALPOST NOTE (R45): retirement REDEFINED consolidation from deletion-survival (E120's original bar, which the splice arms still FAIL at their site: D-183 -54%/-31%) to learns-and-generalizes; the redefinition is now stated, not silent. What position diversity does (choose routed vs site-stored) belongs to T079's revived form]"
assert o4 in t
t = t.replace(o4, n4, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE: e150 ----------
q = open("QUEUE.md", encoding="utf-8").read()
o_q = "| e147 | THE WIDTH LADDER"
row = """| e150 | THE FLAT-CE ROUTE TEST (R45 critic's ultimatum — decides whether 'routed' is a memory property or a wreck artifact) | DISPATCHED 09:20Z (CPU eval-only) | On consolidated + R nets: (1) perm@g-12 AND perm@g0 (direction-permutation — fact survives => presence-only holds at both geometries); (2) FORCED-OFF-SINK attention mask (mask attention to position 0 at eval — removes the sink's attentional role at flat-ish CE); (3) L0H3-class fact-specific head ablation (e133's flat-CE fact-kill candidate, alone and as a set); (4) FACT-AT-POSITION-0 scramble (W014's never-consults control); (5) NORM LADDER on wpe[0] (0.07/0.15/0.25/0.35 — the threshold in (0.066,0.382)). Bars: FLAT-CE-ROUTE = any intervention kills fact >=60% at CE cost <=+0.35; ALL-KILLS-WRECK = every fact-kill costs CE >=+0.70 (T081/T082 hard-bounded — 'routed' becomes a wreck statement); DIRECTION-CONSULTED = fact-at-position-0 scramble kills (tenant reading dies); PRESENCE-AT-NOVEL = perm@g-12 spares >=80% of g-12 level |
""" + o_q
assert o_q in q
q = q.replace(o_q, row, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["last_review"] = now
s["last_novelty"] = now
s["current_experiment"] = "Fleet 3: e142 (CPU, origin census) + e147 (GPU, width ladder) + e150 (CPU, flat-CE route test — the frame's decisive cleaner). R45 folded."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("R45 fold complete")
