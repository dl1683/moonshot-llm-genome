import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- Repairs ----------
t = open("THINKING.md", encoding="utf-8").read()
# T088 amendment (auditor finding 2)
o1 = "## T088 — E151: the cliff is PER-NET — memory type is a global phase of one substrate, and the transition runs BOTH WAYS (2026-09-28 ~10:10Z)"
n1 = "## T088 — [AMENDED per R47: e152 found the conversion passes a ~50-step mixed state (T094) — the 'per-net' reading is itself pending e158 (variance-vs-placement) and e154 (global-vs-self, same-fact confound); 'at ANY site' below is untested beyond 183] E151: the cliff is PER-NET — memory type is a global phase of one substrate, and the transition runs BOTH WAYS (2026-09-28 ~10:10Z)"
assert o1 in t, "T088"
t = t.replace(o1, n1, 1)
# 67x provenance fix (T094)
o2 = "a genuine, functional site-store (67× its control bar, readable at 0.962)"
if o2 in t:
    t = t.replace(o2, "a genuine, functional site-store (strength/bar mean across the three dwell peaks = 66.8x, ~89x at s8; derivation stated per R47 audit; readable at 0.962)", 1)
# CE range (T090)
o3 = "N2 {L1H0,L0H0} kills at CE +0.25 in both ablation modes"
if o3 in t:
    t = t.replace(o3, "N2 {L1H0,L0H0} kills at CE +0.245 (mean) / +0.276 (zero) in both ablation modes", 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

n = open("NOTES.md", encoding="utf-8").read()
n = n.replace("67x control, site read p_Z 0.962", "66.8x bar-mean across dwell peaks (89x at s8; derivation per R47 audit), site read p_Z 0.962", 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
p = p.replace("67x control\nAND >=50% geometry retention", "66.8x bar-mean across dwell peaks\nAND >=50% geometry retention (derivation stated; n=1 until e152R)", 1)
p = p.replace("kills the fact at CE +0.25 in both\n    ablation modes", "kills the fact at CE +0.245/+0.276 (mean/zero) in both\n    ablation modes", 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

q = open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| e158 | THE 2x2 COMPLETION (R46 critic attack 1 — THE missing cell: variance x site) | TOP PRIORITY — next GPU slot (before e154) |",
              "| e158 | THE 2x2 COMPLETION (variance x site) | RUNNING ~11:29Z (CPU park; bars frozen 10:20:51Z per R47 audit) |", 1)
q = q.replace("| e156 | THE SELF ACROSS THE PHASE FLIP (e146 follow-up fork, pre-registered) | QUEUED — rides e146's rig after its gates pass |",
              "| e156 | [GATED: e146's gates FAILED (instrument-invalid) — blocked on e146b] THE SELF ACROSS THE PHASE FLIP | GATED |", 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- R47 REVIEWS entry ----------
r = open("REVIEWS.md", encoding="utf-8").read()
anchor = "## Review 46 — the missing cell and the flattering-direction bias"
entry = """## Review 47 — the mass fork and the unread census (2026-09-28T12:00Z; covering 10:45–12:00Z; e158/e125a/e162 running through it, all CPU, GPU user-occupied)

### AUDITOR — ISSUES FOUND, all repaired in-beat
Five folds verified clean number-by-number (e146/e153/e159/e160/e152; the
10.8x absorber independently re-derived from sink-mass sums). Findings:
(1) "67x control" lacked provenance (actual: 89.2x/bar at s8; 66.8x = bar-mean
across the three dwell peaks — derivation now stated everywhere); (2) T088
never amended (e152-dwell + e158/e154-pending markers added); (3) "CE +0.25"
rounded toward the bar (+0.245/+0.276 range stated); (4) e160 reading-map
timing marginal (28s post-mtime, 65s pre-data-commit — QUEUE bars were 11 min
before run start; adjudicative pre-reg safe); (5) timestamp hygiene: forward-
dated labels recur, metrics date conventions inconsistent (some true-UTC,
some local-as-Z) — ONE CLOCK required in all future taskings; (6) queue drift
(e158 RUNNING, e156 GATED — fixed).

### IDEATOR — CPU-window plan; e125a dispatched, e161-e164/e152R defined
Top pick e125a (the inverted knife — the site-stored fact's own kill set;
NO-SITE-KNIFE = the types differ in REMOVABILITY). e161 (the dwell dissected:
knife-at-dwell, brake-at-peak, THE FREEZE-CELL), e146b (self-battery home
rerun — the W015 gate), e154's N2 rider (layer attribution), e149+s128,
finish-line ranked (e158 > e154 > e125a > e157 > e147R; GPT-2 = optional
crown). Staleness: e144 retired, e145 absorbed, e137 re-aimed.

### CRITIC — accepted in full; the frame's newest fork dispatched
1. (HIGH) THE MASS FORK: e159's joint cell was identical to mask-alone and
   removed reads AND mass together — "dies of what it reads" vs "dies of what
   the absorber steals" is OPEN. e162 DISPATCHED (value-restore-under-poison;
   mass-inflate-on-healthy). READ-coupled marked PROVISIONAL everywhere.
2. (MED) "Type-selective" overreached — corrected to CIRCUIT-selective
   (install dies too; one boundary); number attribution fixed (67.3 was
   E2-mean); the site-stored census SAT UNREAD in e133's metrics (L1H2 et al,
   different coordinates, max drop 20% — type-ASYMMETRIC surgery possible;
   e125a decides).
3. (MED-HIGH) The dwell is n=1 — markers added; the FREE mask-column re-read
   taken (dwell SURVIVES the clean dial: s32 mask-retention 0.979 vs ladder
   kill 0.19); e152R re-seeds queued.
4. (HIGH) The intro's first sentence sits on the saturating dial — PROVISIONAL
   marker; e163 (the saturation control: the dial on arm_b, 7% row-0-share)
   queued and deciding.
5. (MED) T092's layering circular until the post-kill census — e164 queued.
6. R46 adjudication: handled, with the flattering-direction bias RECURRING
   with better paperwork (e159's fold minted a new positive noun from a
   bounding cell within ~30 min; symmetric audit: three strengthening folds,
   zero bounding cells queued for the new claims — now corrected: e162/e163/
   e164 ARE the bounding cells).
7. Process: brake overshoot narrativized on sight (n=1, noise-floor known) —
   flagged; new standing rule: no narrativized texture enters paper text
   without a replication or a marker.

### Decisions
1. e162/e163/e164/e152R = the bounding cells for the window's three strongest
   claims; e125a/e158 carry the dissociation completions. 2. All repairs
   applied before this entry. 3. ONE-CLOCK rule added to future taskings
   (UTC, true, in metrics AND queue stamps). 4. Fleet: e158 + e125a + e162
   (CPU); R47 stamps below.

---

"""
assert anchor in r
r = r.replace(anchor, entry + anchor, 1)
open("REVIEWS.md", "w", encoding="utf-8").write(r)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["last_review"] = now
s["last_novelty"] = now
s["current_experiment"] = "Fleet 3 (all CPU): e158 (2x2) + e125a (inverted knife) + e162 (READ-vs-MASS fork). R47 folded and synthesized."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("R47 synthesis + repairs complete")
