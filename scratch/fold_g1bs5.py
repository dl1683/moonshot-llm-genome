# -*- coding: utf-8 -*-
"""Fold g1bS5: NOTES, T167, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g1bS5 — the formation curve: SHARP-OPTIMUM — the 10M formation optimum located IN THE INTERIOR (peak 0.7677 at 0.20 rms, 0.012 below the express bar); the dose is a tuned window; CE healthy throughout (2026-10-02 ~10:05Z) — DONE

WHAT WE DID: the consolidation-only dose sweep (3 new doses at
4e-4 from the same loaded base+install, seed 10901; the two
committed points joined); the owner envelope held throughout
(3 launches double-polled at 0%/55-65C; bursts 21-32s; 180s
cooldowns; 32 pause-and-waits).

WHAT WE SAW (T167): THE CURVE — 0.12 -> 0.6498 / 0.15 -> 0.7285 /
0.20 -> 0.7677 (PEAK) / 0.25 -> 0.7431 / 0.30 -> 0.2523 (the 1e-3
casualty 0.0010 kept SEPARATE, off the curve). SHARP-OPTIMUM
FIRES: the max interior reading >= the registered threshold; the
optimum is IN THE INTERIOR (argmax 0.20 rms / s500, 0.012 below
the 0.78 express bar); MONOTONE-DECLINE dead. THE E113 FORM'S
DOSE AT 10M IS A TUNED WINDOW (~2/3 of e113's 0.30 rms) — not
"more is better", not "less is safer". CE_R HEALTHY 1.66-1.70
across the sweep (a dose effect, not stability); held30 rises
0.39 -> 0.58 through the peak then collapses at 0.30, tracking
the channel. HONESTY: n=1 host/seed/fact; consolidation-only —
NO wall claims; the same-recipe replay fuzz re-measured in-cell
(mean |dg0| 2-4e-2, max 2.7e-1) — the curve SHAPE is 2-4x the max
fuzz but the 0.20-vs-0.25 ordering (gap 0.025) is WITHIN it: the
robust statement is "the optimum sits in the 0.15-0.25 window"
(the critic's one-draw caveat carried; the redrawn-dose leg still
owed for the fine ordering). THE REGISTERED FOLLOW-ON: the fifth
take — the WALL ARMS on the peak root (runs/checkpoints/
g1bS5_root_m020.pt, already 0.012 below the bar) — the actual
adjudication.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T166 —"
card = """## T167 — g1bS5: the tuned window — the formation optimum located in the interior (2026-10-02 ~10:05Z)

The dose sweep closes the inversion with a curve: the 10M
formation optimum sits IN THE INTERIOR (peak 0.7677 at 0.20 rms,
0.012 below the express bar; the robust window 0.15-0.25 given
the fuzz), CE healthy throughout — the e113 form's dose at 10M is
a TUNED WINDOW at ~2/3 of e113's movement. THE SAGA'S SHAPE: four
takes of diagnosis (three casualties, one inversion) then one
sweep — the one-knob licenses were exploring a non-monotone
landscape pointwise; the curve is the map they needed. THE
HONESTY LEDGER: n=1 draw (the critic's caveat carried — the fine
peak ordering is within fuzz; a redrawn interior dose still owed
for it); consolidation-only (no wall claims); the 0.78 bar
UNBROKEN but within 0.012 at the peak. THE FIFTH TAKE IS THE
CHEAPEST OF ALL: the peak root is SAVED (g1bS5_root_m020.pt) —
the wall arms run directly on it (W1/W2/W3 + C; short bursts; the
envelope-log now recording every poll). If the arms adjudicate,
the wall's scale question — open since the first divergence —
closes on the sweep's back.

"""
assert anchor in t and "## T167" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHING 08:51Z — small cooled bursts under the owner envelope |",
              "| DONE 10:05Z (T167: SHARP-OPTIMUM — the peak 0.7677 at 0.20 rms, 0.012 below the bar; the window 0.15-0.25 robust to fuzz; CE healthy; a tuned window at ~2/3 of e113's dose; the peak root SAVED) |", 1)
i = q.index("| g1bS5 |"); j = q.index("\n", i) + 1
row = ("| g1bS6 | THE FIFTH TAKE — THE WALL ARMS ON THE PEAK ROOT (g1bS5_root_m020.pt: commit -> W1/W2/W3 {R_rms "
       "ladder} vs C; the actual scale adjudication) | DISPATCHING 10:06Z — short bursts; the envelope-log live |\n")
q = q[:j] + row + q[j:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T10:06:00Z"
st["current_experiment"] = ("g1bS5 FOLDED (T167: SHARP-OPTIMUM — the interior peak at 0.20 rms; the tuned window). "
                            "Fleet: e202 (CPU, the falsifier) + g1bS6 DISPATCHING (GPU: the wall arms on the peak "
                            "root — the fifth take, the actual adjudication)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g1bS5 folded")
