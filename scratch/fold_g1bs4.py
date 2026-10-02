# -*- coding: utf-8 -*-
"""Fold g1bS4: NOTES, T165, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g1bS4 — the wall at 10x, take 4 (movement-matched dose): TEXTURE — and the dose question INVERTED; the formation landscape is NON-MONOTONIC in movement (2026-10-02 ~08:50Z) — DONE

WHAT WE DID: T161's licensed knob (750 steps @ 4e-4 = e113's 0.30
rms), everything else loaded verbatim; the ninth-disruption
recovery (W2 complete; C/W1 recovered from the run log; W3
fresh-replayed bit-tight through +50); the owner envelope HELD
(launch gates util<=20% AND temp<=70C double-polled; 181s
cooldowns; 5 heat pauses ridden; never migrated).

WHAT WE SAW (T165): G-ROOT FAIL — root g-12 0.2523 < 0.78: the
movement-matched dose produced a WEAKER root than g1bS3's
near-miss at a THIRD of the movement. THE 10M FORMATION LANDSCAPE,
three points: 0.0010 (0.30 rms @ 1e-3) / 0.6498 (0.12 rms @ 4e-4)
/ 0.2523 (0.30 rms @ 4e-4) — NON-MONOTONIC IN MOVEMENT; the
near-miss was not a dose shortfall; movement-matching is not the
cure; the formation optimum (if any) sits BETWEEN 0.12 and 0.30
rms and is SHARP. No wall bar adjudicated (TEXTURE, the registered
rule). THE RECORD LADDER: C dead at +1 (D_kill = one AdamW step =
9.99e-4 rms, T139 at 10x again); W1's formation shock (0.04@+1)
then recovery ABOVE root (flat 1.49x; the +1 dip alone fails the
strict bar); W2 1.20x; W3's LATE FADE (0.31@+200 -> 0.12@+300) —
AT A WEAK ROOT THE LOOSER BALLS HOLD WORSE (the opposite of
WALL-TIGHTENS texture); tax +0.232; freezing FALSE (the walled
arm IMPROVED corpus CE). HONESTY: n=1 host/seed/fact; the ladder
carries no bar; the recovery provenance disclosed (the run-log
parse; the fresh replay's device-fuzz band).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T164 —"
card = """## T165 — g1bS4: the dose question inverted — formation is non-monotonic in movement at 10M (2026-10-02 ~08:50Z)

Take 4 closes the dose question by inverting it: matching e113's
total movement made the channel WEAKER (0.2523 vs 0.6498 at a
third of the movement) — the 10M formation landscape is
NON-MONOTONIC in consolidation movement, with the optimum (if it
exists) sharp between 0.12 and 0.30 rms. THE CURE PATTERN'S LIMIT:
the one-knob licenses fixed stability (lr) and dose (steps) and
the formation still refuses to transfer — the e113 consolidation
FORM itself may not survive 10M (a form question, not a knob
question). THE LADDER'S NEW TEXTURE: at a weak root, the looser
balls hold WORSE (W3's late fade) — the wall's protection quality
tracks the ROOT's formation strength; and the walled arms IMPROVE
corpus CE (freezing False at 10M too). THE NAMED NEXT CUTS: the
formation-vs-movement curve (a consolidation-only dose sweep,
0.15/0.20/0.25 rms at 4e-4 — root reads only, no arms; small
cooled bursts under the owner envelope) or ACCEPT the e113 form's
10M ceiling as the finding. THE HONEST POSITION: four takes, one
inversion, three TEXTUREs — the wall's scale question has cost
patience and taught the recipe-stack lesson three ways; the sweep
is cheap and the curve is the dissection's instinct.

"""
assert anchor in t and "## T165" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHING 22:11Z — the one-knob pattern's fourth application; base+install+nothing else reused |",
              "| DONE 08:50Z (T165: TEXTURE — the dose question INVERTED; formation non-monotonic in movement: 0.0010/0.6498/0.2523 at 0.30-hot/0.12/0.30 rms; the optimum sharp between; W3's late fade = looser balls hold worse at weak roots; freezing False) |", 1)
i = q.index("| g1bS4 |"); j = q.index("\n", i) + 1
row = ("| g1bS5 | THE FORMATION CURVE (consolidation-only dose sweep: 0.15/0.20/0.25 rms @ 4e-4 — root reads only, "
       "no arms; the sharp-optimum question) | DISPATCHING 08:51Z — small cooled bursts under the owner envelope |\n")
q = q[:j] + row + q[j:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T08:51:00Z"
st["current_experiment"] = ("g1bS4 FOLDED (T165: the dose question inverted — formation non-monotonic; the optimum "
                            "sharp between 0.12-0.30 rms). Fleet: e201 (CPU, the rotation census) + g1bS5 DISPATCHING "
                            "(the formation curve, consolidation-only, small cooled bursts)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g1bS4 folded")
