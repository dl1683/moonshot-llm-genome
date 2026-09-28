import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- THINKING: T093 amendment (the silent loss) ----------
t = open("THINKING.md", encoding="utf-8").read()
o1 = "## T093 — E159: READ-coupled — the mask is the health door, and only readers die of the poison (2026-09-28 ~11:15Z)"
n1 = "## T093 — [RESOLVED MIXED by e162/T095 — BOTH channels kill, each at its own conditions (supply edge matched; allocation edge flattened-profile-only; per-layer distribution untested, e167 queued); the one-channel noun below is historical] E159: READ-coupled — the mask is the health door, and only readers die of the poison (2026-09-28 ~11:15Z)"
assert o1 in t, "T093"
t = t.replace(o1, n1, 1)

# T090/NOTES type-selective markers
o2 = "(2) THE KNIFE IS CIRCUIT-SELECTIVE, ONE BOUNDARY"
if o2 not in t:
    o2b = "(2) THE KNIFE KNOWS THE TYPE:"
    assert o2b in t, "T090 alt"
    t = t.replace(o2b, "(2) THE KNIFE IS CIRCUIT-SELECTIVE [R47/R48: not type-selective — install dies too]:", 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

n = open("NOTES.md", encoding="utf-8").read()
n = n.replace("HEAD SURGERY DISSOCIATES THE MEMORY TYPES.", "HEAD SURGERY DISSOCIATES THE MEMORY CIRCUITS [R47/R48: circuit-, not type-, selective].", 1)
n = n.replace("b=3.0: x0.347", "b=3.0: x0.337", 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- paper fixes ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
fixes = [
 ("removable (e160) — and the dependence\nitself is READ-coupled (e159: masking the sink heals a poisoned net\ncompletely while the site-stored fact pays the same organism damage\nand lives — only readers die of the poison;",
  "removable (e160) — and the dependence is\nCOUPLING with the sink through two channels (e159's double dissociation\ngates it; e162: healthy content fully rescues; total-dose absorption on\na flattened profile kills — the per-layer distribution untested);"),
 ("dies 79-95% at CE +0.25 via a superadditive\n    complementary circuit",
  "dies 70.7-97.4% at CE 0.245-0.280 (N2, both modes) via a superadditive\n    complementary circuit"),
 ("67x control\nAND >=50% geometry retention (derivation stated; n=1 until e152R)",
  "66.8x the 2x-control bar across dwell peaks\nAND >=50% geometry retention (derivation stated; n=1 until e152R)"),
 ("[* = e158 pending]", "[e158 DONE: SITE-INDEPENDENT]"),
 ("e162*/e125a*", "e162/e125a (DONE)"),
]
for old, new in fixes:
    if old in p:
        p = p.replace(old, new, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

# ---------- queue status fixes ----------
q = open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| e163 | THE SATURATION CONTROL (licenses or collapses the intro's first sentence) | READY", "| e163 | THE SATURATION CONTROL | READY (next fill)", 1)
q = q.replace("| e164 | THE POST-KILL CENSUS (de-circularizes T092's layering) | READY", "| e164 | THE POST-KILL CENSUS | RUNNING ~12:40Z (CPU)", 1)
q = q.replace("| QUEUED — first GPU slot after e152 |", "| RUNNING ~12:35Z (CPU park) |", 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- R48 REVIEWS entry ----------
r = open("REVIEWS.md", encoding="utf-8").read()
anchor = "## Review 47 — the mass fork and the unread census"
entry = """## Review 48 — the home-graft counterexample and the disuse alternative (2026-09-28T13:15Z; covering 12:00–13:15Z; e164/e154/e166 running through it)

### AUDITOR — ISSUES FOUND (numbers ALL clean; defects marking/process)
W017 rider recomputed EXACTLY (0.352/0.257/0.117; entropy 0.795/0.7818);
66.8x derivation verified ((89.2+48.3+62.8)/3); all three folds' numbers
trace. Findings repaired: (1) ONE-CLOCK violations in e125a/e162 metrics
(local-EDT-as-Z, both written AFTER the R47 rule — documented here as the
correction; metrics left untouched as-written); (2) T093's RESOLVED-MIXED
amendment was a SILENT LOSS (fold claimed it; the replace failed to match)
— now applied; (3) "type-selectively" persisted in the intro (abstract-only
fix) + T090/NOTES/QUEUE markers — propagated; (4) abstract's number pairing
loosened (79-95%@+0.25 mixed modes) — now 70.7-97.4% @ 0.245-0.280 + the
allocation-edge condition qualifier; (5) QUEUE status drift (e164/e154
RUNNING) — fixed; STATE/review stamp mismatch noted; (6) minor misquote
(x0.337), 67x-bar qualifier, stale outline asterisks.

### IDEATOR — e166 dispatched (the inverse event); e169/e157r defined
Top pick e166: graft-REMOVAL door-restore — DOOR-RESTORES = closure is
active competitive inhibition (causal); DOOR-STAYS-SHUT = the third great
asymmetry (unreopenable-by-surgery). e161's freeze-cell re-registers AFTER
e166's verdict (honesty note). e165's concrete arm list (distance ladder
with the {1,4,8} arms inside jitter-traveled territory); e169 (codes on the
shelf); e157r (the 2x2 on family 2). Paper completion list: exactly e163/
e166/e154/e157r/e152R. Staleness: e125 retired, e148 parked, e144/e093/
e103 relabeled.

### CRITIC — accepted in full; the session's most instructive error
1. (HIGH) T097's headline CONTRADICTED BY ITS OWN RUN'S CENSUS: locked@band
   DID re-form a home graft (row 129: brake -> content, site_pos TRUE) with
   the door OPEN — two grafts, different outcomes; the operative variable
   is SITE NOVELTY, not graft formation. "One event two faces" WITHDRAWN;
   the registered verdict (two-factor gate) stands. The question was parked
   in NOTES and then answered without reading the census.
2. (MED-HIGH) T096's "NO kill set EXISTS" = a 0.03% sample (the consolidated
   kill was a superadditive pair INVISIBLE to singles — the same hiding
   place unsearched); B5/B6 and the MLP surface untouched -> e168 (exhaustive
   630-pair scan + MLP ablation).
3. (HIGH) T095's "each sufficient" fails internal consistency: allocation
   edge shown only at a flattened profile (11.4x L0); supply edge matched
   -> e167 (per-layer-matched bias).
4. (MED) W017 minimal restatement: killability tracks COMPLEMENTARITY (the
   top-loaded head is dispensable), not concentration.
5. Paper licensing gaps fixed: "bidirectionally" -> lineage-clause; the
   graft sentence -> "accompanies novel-site teaching"; nouns propagated.
6. R47 adjudication: handled EXCEPT noun propagation; the flattering-
   direction bias recurred ABOVE honest runs (narrative-layer overreach:
   three nouns minted, two outrun) — the correction discipline now targets
   the narration layer specifically.
7. Process: verdicts pre-committed and honest; the failure was post-
   adjudication storytelling answering questions the metrics had already
   answered differently.

### Decisions
1. The build-travel frame is PROVISIONAL on the DISUSE alternative; the
   three-way fork (COMPETITIVE/REWRITE/DISUSE) is mapped with deciding
   cells (e166/e161/e154/e155) — reading maps updated BEFORE their data.
2. e167/e168 queued as the bounding cells for T095/T096's strongest forms.
3. All repairs applied before this entry. 4. Fleet: e164 + e154 + e166.

---

"""
assert anchor in r
r = r.replace(anchor, entry + anchor, 1)
open("REVIEWS.md", "w", encoding="utf-8").write(r)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["last_review"] = now
s["last_novelty"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("R48 synthesis + repairs complete")
