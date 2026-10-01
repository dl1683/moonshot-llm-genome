# -*- coding: utf-8 -*-
"""Fold g1bS: NOTES, T148, claims-ledger update, QUEUE (DONE-BLOCKED +
g1bS2 licensed), STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g1bS — the wall at 10x BLOCKED AT THE BASE GATE: the house recipe overtrains the 10M host — G-BASE-QUAL FAIL, hard stop, no arms, no bars (2026-10-01 ~15:15Z) — DONE (BLOCKED)

WHAT WE DID: fourth dispatch. The 4e-4 retrain's s1113 resume state
SURVIVED the shutdown — continued agent-3's schedule seamlessly (13
chunks total, one trajectory) to the completed 4000-step cosine.
The G-BASE-QUAL gate (registered BEFORE the retrain) then read:
g1 cosine-complete PASS; g2 val-decreasing FAIL (2.1609 -> min
1.5766 @ s1113 -> 2.8506; first violation chunk 5); g3 final<=1.70
FAIL (2.8506); g4 coherence PASS (ls 0.71, mwl 4.25). THE HARD
STOP FIRED EXACTLY AS REGISTERED: no install, no arms, no wall bar
adjudicated, no dial search; the failed base ARCHIVED (never
deleted); four-dispatch provenance + the lr deviation documented in
13 progressive metric writes.

WHAT WE SAW (T148): THE COHERENCE-PASS/VAL-FAIL SPLIT is the
memorization-recitation signature — the base "reads" plausibly
while generalizing worse (train 0.32 falling, val 2.85 rising); a
coherence check ALONE would have licensed a memorizing host (W021
echo: the gate that cannot fail). THE SCALE FINDING THE CELL DID
YIELD: BOTH licensed lrs (1e-3 archived; 4e-4) U-TURN in the same
s~1000-1200 window with near-identical minima (1.568 vs 1.577) —
the turn is CAPACITY/CORPUS-driven (10M params x ~1M-char corpus);
width-scaled lr slows the memorization slope but cannot prevent the
turn; the 0.87M-minted 4000-step recipe overtrains the 10M host ~3x
past its val min. THE WALL'S SCALE QUESTION IS OPEN, NOT ANSWERED.
WHAT'S LICENSED (g1bS2): the HOST recipe re-registration — a
val-min-anchored cosine (~s1150 at 4e-4, same corpus for
comparability), then the frozen wall cell VERBATIM (R_rms
convention, ladder, bars untouched). The frozen wall machinery
awaits; do NOT relaunch the current script unmodified (it would
retrain into the identical failure).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T147 —"
card = """## T148 — g1bS: the recipe is scale-bound — the honest negative that saves the cell (2026-10-01 ~15:15Z)

The hard stop fired exactly as registered: the 10M base failed
G-BASE-QUAL (val rising past its s1113 minimum to 2.85 by the
cosine's end; coherence alone PASSING — the memorization-recitation
signature). NO ARMS RAN — the wall's scale question is OPEN, and
the negative is itself the scale lesson: THE HOUSE RECIPE DOES NOT
TRANSFER. Both licensed lrs U-turn together (minima 1.568/1.577 at
s~1000-1113) — capacity/corpus-driven at 10M params on ~1M chars;
the 4000-step cosine minted at 0.87M overtrains ~3x past the val
min; width-scaling the lr slows memorization but cannot prevent the
turn. THE POLICY EXTENSION (R58's no-hardcoded-constants, one
step further): recipes are SCALE-BOUND — steps AND lr re-register
per host size, anchored at the val minimum. THE INSTRUMENT ECHO
(W021): coherence was the instrument that could not fail — it
would have licensed a memorizing host; the val-decreasing clause
was the one that could. g1bS2 LICENSED: val-min-anchored cosine
(~s1150, 4e-4, same corpus), then the frozen wall cell verbatim —
R_rms ladder, bars, and wash untouched. The wall question costs one
more GPU hour, not a redesign.

"""
assert anchor in t and "## T148" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

p = "scratch/claims_ledger.md"
s = io.open(p, encoding="utf-8").read()
old = "| C6 | THE WALL: commit-and-project holds the battery-channel through the killing wash (n=3 seeds, one root); A survived an active second-install attempt; the tax is the onset channel (0.21 vs 0.53); museum question open pending the dose ladder | g1b/g1bR/g1bW | licensed n=3 wash-draws ONE root (g1c-root queued); scale pending g1bS |"
new = "| C6 | THE WALL: commit-and-project holds the battery-channel through the killing wash (n=3 seeds, one root); A survived an active second-install attempt; the tax is the onset channel (0.21 vs 0.53); museum question open pending the dose ladder | g1b/g1bR/g1bW | licensed n=3 wash-draws ONE root (g1c-root queued); scale OPEN — g1bS BLOCKED at the base gate (the house recipe overtrains the 10M host ~3x past val-min; g1bS2 licensed with the val-min-anchored recipe) |"
assert old in s, "c6"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| g1bS |"):]; row = row[:row.index("\n")]
new_row = ("| g1bS | the wall at >=10x (~10M) | DONE-BLOCKED 15:15Z (T148: G-BASE-QUAL FAIL — the hard stop fired, "
           "no arms, no bars; BOTH lrs U-turn at s~1000-1113 (min 1.568/1.577): capacity/corpus-driven; the "
           "0.87M-minted 4000-step recipe overtrains the 10M host ~3x; coherence-PASS/val-FAIL = the memorization "
           "signature; the wall's scale question OPEN) |")
q = q.replace(row, new_row, 1)
i = q.index("\n", q.index(new_row)) + 1
g1bs2 = ("| g1bS2 | THE WALL AT 10x, TAKE 2 (host recipe RE-REGISTERED: val-min-anchored cosine ~s1150 at 4e-4, "
         "same corpus for comparability; then the frozen wall cell VERBATIM — R_rms ladder, bars, wash untouched) | "
         "QUEUED — GPU behind g2g; do NOT relaunch the current script unmodified |\n")
q = q[:i] + g1bs2 + q[i:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T15:16:00Z"
st["current_experiment"] = ("g1bS FOLDED (T148: DONE-BLOCKED — the recipe is scale-bound; g1bS2 licensed with the "
                            "val-min-anchored host recipe). Fleet: e_chart + e182c-r4 (CPU) + g2g DISPATCHING (GPU, "
                            "the freed slot — the rhythm's controls).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g1bS folded")
