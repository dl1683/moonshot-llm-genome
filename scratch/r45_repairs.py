import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# THINKING.md repairs
t = open("THINKING.md", encoding="utf-8").read()
# +6% -> +4% (T081 and W014)
fixes = [
    ("direction-scramble 0.817 — not even the direction matters", "direction-scramble 0.817 (+4% vs base 0.785) — not even the direction matters"),
    ("direction-scrambling\nrow 0 costs the corpus +0.70 nats but SPARES the fact (+6%)", "direction-scrambling\nrow 0 costs the corpus +0.70 nats but SPARES the fact (+4%)"),
    ("direction-scrambling row 0 costs the corpus +0.70 nats while\n*helping* the fact", "direction-scrambling row 0 costs the corpus +0.70 nats while\n*helping* the fact (+4%)"),
]
for old, new in fixes:
    if old in t:
        t = t.replace(old, new, 1)

# T083 addendum claim downgrade
o1 = """**(3) THE T082 DERIVATION ADDENDUM'S SYLLOGISM FAILED — recorded
as predicted-then-falsified.** Registered at ~08:28Z (before
reading e140): the taxonomy predicts E-flat + L-flat + R-rising
on the row-0 dial."""
n1 = """**(3) THE T082 DERIVATION ADDENDUM'S SYLLOGISM FAILED — recorded
as asserted-then-falsified.** R45 AUDIT DOWNGRADE: the addendum
was written contemporaneously (script mtime 08:28:39Z, metrics
on disk 08:27:23Z, never read by the lead before the agent's
report), but its first git appearance is the fold commit
(08:31Z) — AFTER the data commit (08:28Z). "Registered" implied
commit-before-data and that is NOT satisfied; the claim is
ASSERTED, UNPROVEN ordering (mitigant: the prediction failed and
was recorded as such — fabricators do not pre-register
failures). Content: the taxonomy predicts E-flat + L-flat +
R-rising on the row-0 dial."""
assert o1 in t, "T083 downgrade anchor"
t = t.replace(o1, n1, 1)

# W013 marker
o2 = "## W013 — WONDER: the read policy is the protagonist"
n2 = "## W013 — WONDER: the read policy is the protagonist [R45 audit marker: the T079 key-selection clause below was killed on its dial by e140/T083 and causally REVIVED by e143/T084 — read 'invariant keys win' as the revived, width-ladder-pending form]"
assert o2 in t
t = t.replace(o2, n2, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# QUEUE +4%
q = open("QUEUE.md", encoding="utf-8").read()
q = q.replace("direction-scramble SPARES fact (+6%)", "direction-scramble SPARES fact (+4%)")
open("QUEUE.md", "w", encoding="utf-8").write(q)

# DAY_SIX_REPORT repairs
r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
r = r.replace("direction-scrambling row 0 SPARES\n   the fact (+6%)", "direction-scrambling row 0 SPARES\n   the fact (+4%)")
r = r.replace("row-183 content ~1000x controls", "row-183 content 250-5900x controls")
r = r.replace("fact-specific residue 84.5% heads", "fact-specific residue 84.5% heads (locality-filtered, report-only table)")
o_r = """Single-lineage caveats stand where
noted (probes 1-3 one seed; probe 4 two independent nets). Open
cells: e140 (T079's law), e143 (the proximity fork) — this report
updates when they land."""
n_r = """Single-lineage caveats stand where
noted (probes 1-3 one seed; probe 4 two independent nets). Both
open cells have since landed: e140 killed T079's law on its
registered dial; e143's COMPASS-CAUSAL revived invariance
causally (see finding 5)."""
assert o_r in r, "coda anchor"
r = r.replace(o_r, n_r, 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("R45 audit repairs applied")
