import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o1 = "held-30-under-D-all 0.709 vs"
n1 = "held-30-under-D-all 0.663 vs 0.071 at matched g+0 (R44 audit: geometry-matched pair; R's cross-geometry max is 0.709 at g-8 where E reads"
# reconstruct: original line was "...; held-30-under-D-all 0.709 vs\n0.071; novel geometry..." -> fix precisely below
old_full = "E sits 0.01 under the 0.20 bar; held-30-under-D-all 0.709 vs\n0.071; novel geometry g-12 R 0.813 vs E 0.092"
new_full = "E sits 0.01 under the 0.20 bar; held-30-under-D-all 0.663 vs\n0.071 at MATCHED g+0 (R44 audit correction — the first fold paired R's\ncross-geometry max 0.709@g-8 against E's g+0; R@g-8 vs E@g-8 is\n0.709 vs 0.021); novel geometry g-12 R 0.813 vs E 0.092"
assert old_full in n, "NOTES pair anchor"
n = n.replace(old_full, new_full, 1)

o2 = "the\nONLY content-positive row in the census; the whole 121-137 band"
n2 = "the\nonly row ABOVE the control band (380x max control; two control-level\nrows 118/119 flag content:true at 0.0019 — R44 audit precision);\nthe whole 121-137 band"
assert o2 in n, "NOTES only-row anchor"
n = n.replace(o2, n2, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
o3 = "Post-consolidation, row 0 is the only content-positive wpe row"
n3 = "Post-consolidation, row 0 is the only content-positive wpe row\nabove the control band (380x; rows 118/119 flag content:true at\ncontrol level 0.0019 — R44 audit precision)"
assert o3 in t, "T077 precision anchor"
t = t.replace(o3, n3, 1)

hdrs = [
 ("## T075 — E120: the migration needs the road itself — position diversity, not signal, not self (2026-09-28 ~10:10Z)",
  "## T075 — [RETIRED-PROVISIONAL per R44 critic: retirement announced on probe-1 (learning-at-183) before the D-183 graduation cell; e139 adjudicates] E120: the migration needs the road itself — position diversity, not signal, not self (2026-09-28 05:52Z; header clock repaired per R44 audit)"),
 ("## T074 — E121: dreams are not a consolidation road — W004's self stops at the field boundary (2026-09-28 ~09:20Z)",
  "## T074 — E121: dreams are not a consolidation road — W004's self stops at the field boundary (2026-09-28 05:02Z; header clock repaired per R44 audit; arm-c dream-position read pending as e139 rider 5)"),
 ("## T073 — E083: canalization's strong form dies — the groove persists, the erase weakens, the memory migrates (2026-09-28 ~08:20Z)",
  "## T073 — [MIGRATION READING INVERTED by T078: erase tightens address-binding, does not migrate] E083: canalization's strong form dies — the groove persists, the erase weakens (2026-09-28 03:47Z; header clock repaired per R44 audit)"),
 ("## T065 — E113: coordinate-binding is a developmental stage — W005's mirror resolves (2026-09-28 ~03:10Z)",
  "## T065 — [INVERTED by T077: 'body-stored' was re-keyed to row 0 — D-all survived because it never deletes row 0] E113: coordinate-binding is a developmental stage — W005's mirror resolves (2026-09-28 ~03:10Z)"),
]
for old, new in hdrs:
    assert old in t, old[:40]
    t = t.replace(old, new, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE.md ----------
q = open("QUEUE.md", encoding="utf-8").read()
rows = [
 ("| e083 | canalization cycle-3 | DONE (T073: MIXED + oscillation — groove persists, erase weakens, memory migrates to field; strong canalization DEAD, weak form survives) |",
  "| e083 | canalization cycle-3 | DONE (T073: MIXED + oscillation — groove persists, erase weakens; strong canalization DEAD, weak form survives) [R44 marker: 'migrates to field' INVERTED by T078 — erase tightens address-binding] |"),
 ("| e120 | fact-in-contexts | DONE (T075: SIGNAL-IN-CONTEXTS INSUFFICIENT — position diversity is the ingredient; corpus-ctx > self-ctx (CI); e113 body-stored replicates 5/5) |",
  "| e120 | fact-in-contexts | DONE (verdict RETIRED-PROVISIONAL per R44 critic — e131 probe 1 showed the arms learned at 183 (0.989/0.988), but graduation needs the D-183 survival cell (e139); corpus>self downgraded to suggestive) |"),
 ("| e133 | field anatomy census (per-organ ablation on graduated vs address-phase twins — free-rides e119) | QUEUED | >=60% causal load post-graduation in non-address organs (W008) vs >=60% address-adjacent (address-phase mirror); W005-alt address-spreading killed if grown-row attention <50% |",
  "| e133 | field anatomy census (per-organ ablation: graduated vs address-phase twin vs 183-site splice) | DISPATCHED 07:08Z (GPU evals; ledger-lag fixed per R44 audit — the dispatch preceded the stamp) | SUBSTRATE-IN-BODY: >=60% graduated load in non-address organs + address-phase mirror; ROUTE-ONLY: body <30%, row-0/wpe carries; W005-alt killed if grown-row attention <50% |"),
 ("| e138 | adapter head-start (e120's row-183 net vs fresh twin, identical jitter) | QUEUED | head-start: pre-grown adapter wires in <=60% of fresh steps; within 10% => wiring is everything, mass-growth not the bottleneck |",
  "| e138 | adapter head-start | RETIRED (R44: premise false — e120's row-183 net expresses 0.99 at 183; the 'unconnected adapter' never existed) |"),
 ("| e113 | all-addresses deletion | DONE (T065: BODY-STORED — fact survives all-five-address zeroing; D0129 was scaffold loss; coordinate-binding is a developmental stage) |",
  "| e113 | all-addresses deletion | DONE (T065: fact survives all-five-address zeroing; D0129 was scaffold loss) [R44 marker: 'BODY-STORED'/'developmental stage' INVERTED by T077 — re-keyed to row 0; survival because D-all never deletes row 0] |"),
]
for old, new in rows:
    assert old in q, old[:40]
    q = q.replace(old, new, 1)

# e123/e125/e132/e134/e137 bar updates
more = [
 ("identity half-life across checkpoints; doubles as P5's rekeying probe. ",
  "identity half-life across checkpoints. [R44: 'rekeying probe' clause moot — e119/e131 answered re-keying; drift-vs-output-similarity bar stands] "),
 ("| e125 | attack the graduated fact (P7) | READY | what removes a field-stored fact — segregated ~54 units or woven into self? ",
  "| e125 | attack the moved fact (P7; R44 re-scope) | READY | three surfaces under the row-0 frame: the row-0 key (known -97% — now price collateral), the band (expected ~null), the old address (brake +0.210 — an 'unlearning' move that STRENGTHENS the memory); removability ratio vs pre-consolidation fact at matched collateral "),
 ("| e132 | the wiring trace (ideator top pick = W008's falsifier) | QUEUED — next GPU slot after e119 | ",
  "| e132 | the wiring trace | DEMOTED-OPTIONAL (T078/R44: e140 answers its kernel question eval-only; next TRAINING slot goes to e143 error-steering, the causal test) | was: "),
 ("| e134 | two facts, one field (first multi-fact ecology) | QUEUED | ",
  "| e134 | two facts, one sink (first multi-fact ecology; W012) | QUEUED | "),
 ("restoration <=150 steps with 121-137 band retained (adapters dormant, not dead); falsified if >=250 steps or band regrows from <0.05 |",
  "restoration <=150 steps (falsified if >=250). [R44 re-scope, W008 language barred: measure whether RMU leaves the ROW-0 KEY intact and whether restoration regrows it — 'adapters dormant' framing retired] |"),
]
for old, new in more:
    assert old in q, old[:40]
    q = q.replace(old, new, 1)

# table-jam fix at the parking-lot boundary
o_jam = "lines when a load-bearing claim is single-seed.| e108 |"
n_jam = "lines when a load-bearing claim is single-seed.\n\n| ID | experiment | status | notes |\n|---|---|---|---|\n| e108 |"
if o_jam in q:
    q = q.replace(o_jam, n_jam, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE.json ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e139 (CPU, row-0 universality) + e133 (GPU evals, anatomy census — R44 audit caught this dispatch's ledger lag, now stamped) + R44 critic running. e140 READY-held; e141 (sink-key mechanism) held for critic's position-role discriminator."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("R44 audit repairs applied")
