import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

r = open("REVIEWS.md", encoding="utf-8").read()
anchor = "## Review 45 — the flat-CE ultimatum"
entry = """## Review 46 — the missing cell and the flattering-direction bias (2026-09-28T10:45Z folded, entry written ~10:30Z; covering 09:20–10:45Z; e146/e152/e153/e160 ran through it)

### AUDITOR — ISSUES FOUND, all repaired in-beat
Five folds' numbers verified clean (e142/e147/e150/e151/e143-final). Findings:
(1) e146 finished INSTRUMENT-INVALID and un-folded — folded as T089 with the
e146b home-lineage rerun queued; (2) retro-markers missing on the day's newest
inversions (T080/T081/T082/W015/W008 carried live routed/body-stored language)
— all bracketed; (3) future-dated dispatch stamps (e153) — corrected with a
note; (4) untracked scripts (e146/e152) — committed; (5) paper skeleton: two
vocabularies + Fig-1 brake column transposed — fixed. Pre-registration
integrity CLEAN and git-verified (reading map < e150/e147 data; T087 fork <
e151 fold; e152 bars < dispatch).

### IDEATOR — phase-frame harvest; e153 dispatched, e154-e157 queued
Top pick e153 (phase-switch surgery — the wiring diff between the two phase
nets IS the conversion; PHASE-IN-HEADS vs PHASE-DISTRIBUTED). e154 two-facts-
one-door (decides global-vs-self-conversion — became the paper's new R2);
e155 hysteresis loop; e156 self-across-flip (forked on e146 — later BLOCKED by
instrument failure); e157 replication. Staleness: e134 superseded by e154
(graft instrument), e144 parked (dead-noun bars), e145 folded into e157,
e149 upgraded with the phase control.

### CRITIC — the day's sternest report; accepted in full
1. (HIGH) THE PHASE CLAIM'S MISSING CELL: e151 entangles variance with site;
   jitter@183 never queued by anyone — the 2x2 was one arm from complete
   while five new lines spawned off T088. -> e158 TOP priority (before e154);
   "globally" marked provisional; "fixed wiring" corrected to "fixed
   architecture" (both directions are 300 trained steps).
2. (HIGH) MASK/LADDER CONTRADICTION unreconciled: total sink removal benign,
   partial shrink catastrophic — only a query-side global-softmax collapse
   reconciles, making "COUPLED" possibly organism-death. -> e159 (mask+ladder
   joint cell; site-stored ladder control).
3. (MED) E147 WORDING: "dies"/"no width trend" overran — bound applied (weak
   negative |A|<=0.13 sign-robust; NR 0.903->0.605 = W010's ghost half-alive);
   e147R re-seeds queued.
4. (HIGH) THE CLOSING SENTENCE's "zero collateral" borrowed from day-2 native-
   name surgery — honest form applied (corruption-fatal vs surgically-
   attackable frontier); the irremovable half rests on a near-miss its own
   bar missed by 1.4 points. -> e160 dispatched (head-set escalation).
5. (MED) PAPER drift-prone in the flattering direction (numbers absorbed
   within minutes; caveats didn't make the cut) — three edits applied; the
   bias itself is now a named watch-item.
6. R45 adjudication: four real dispositions, TWO OVER-CORRECTIONS both in the
   claim-strengthening direction (hard-bounded -> "never information flow";
   P-b failure -> "phases of one substrate" from a same-fact cell).
7. Process: provenance rot back (e150 date local-as-Z; future stamps);
   NOTES separators + DAY_SIX splice — repaired.
FINAL LINE: "the lab is one head-ablation away from either the paper's best
figure or its retraction — and that cell has been a near-miss for three
entries running." -> e160 in flight.

### Decisions
1. e160 dispatched (head-set escalation); e158/e159 top-queued ahead of e154;
   e147R queued. 2. All wording/precision repairs applied before the e160
   dispatch (gate held). 3. The flattering-direction bias is named and added
   to the fold checklist (every inversion now asks: did the correction
   strengthen or bound the claim, and is the cell that would flip it queued?).
4. Fleet through this window: e146 (invalid, folded) + e152 + e153 + e160.

---

"""
assert anchor in r
r = r.replace(anchor, entry + anchor, 1)
open("REVIEWS.md", "w", encoding="utf-8").write(r)
print("R46 entry written")
