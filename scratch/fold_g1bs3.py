# -*- coding: utf-8 -*-
"""Fold g1bS3: NOTES, T161, ledger C6, QUEUE (DONE + g1bS4 licensed), STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g1bS3 — the take-3 (the width-scaled e113 license): TEXTURE, and the strongest texture yet — the channel FORMED (650x take-2), the blowout cured, a graded rms ladder on the REGISTERED channel at 10M; the dose question sharp (2026-10-01 ~22:10Z) — DONE

WHAT WE DID: one knob (CONS_LR 1e-3 -> 4e-4) on g1bS2's LOADED
bit-faithful base+install (|d| = 0.0 vs record); all gates PASS
except G-ROOT: root g-12 0.6498 vs the 0.78 express bar ->
TEXTURE (gate failure), arms-for-the-record per the registered
failure_action; nothing adjudicated; no shopping.

WHAT WE SAW (T161): (1) THE CHANNEL FORMED — 0.6498 vs g1bS2's
0.0010 (~650x); the formation-kill cured. (2) THE BLOWOUT CURED —
CE_R 2.03@s25 -> 1.70 settled (vs 2.97 -> 2.58 stuck at 1e-3).
(3) THE MISSED BAR'S SHARP DOSE QUESTION: 300@4e-4 moves 0.12 rms
total vs e113's 0.30 — the frozen dose carried the width-scaled
RATE, not the DISTANCE; s750 would movement-match. Is 0.65 a dose
shortfall or the e113 form's 10M ceiling? g1bS4 LICENSED (the
movement-matched dose). (4) THE RECORD LADDER (never adjudicated):
C dead at +2 (D_kill 3.157 = one AdamW step = lr*sqrt(P), T139 at
10x); W1 (1x rms) DIPS TO ~0.0001 AT +2 — far deeper than g1b's
0.848x-root dip — THEN RECOVERS ABOVE ROOT and sits flat 0.68-0.84
through +300 (flat-phase retention 1.05x, holding the 0.9x
secondary bar; only the +2 dip fails the strict form); W2 partial
(0.49x flat); W3 near-kill-then-weak-recovery — A GRADED
RMS-LADDER RESPONSE ON THE REGISTERED CHANNEL AT 10M, the first.
(5) TAX +0.055 (ref +0.53 at 2.74M; freezing False). HONESTY: n=1
host/seed/fact; the co-reported ladder carries no bar; the +2 dip
(deeper than 2.74M's) is itself a finding — the wall's first
checkpoint at 10M sees a transient near-death before the recovery.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T160 —"
card = """## T161 — g1bS3: the channel forms at 10M — the wall's near-miss with a new texture (the +2 dip and the above-root recovery) (2026-10-01 ~22:10Z)

The take-3 arc: the width-scaled license CURED the formation-kill
(the channel 650x take-2's) and the stability blowout, and missed
the express bar by a dose question now sharp enough to name — the
frozen 300 steps carried the width-scaled rate but a third of
e113's movement; s750 movement-matches. THE RECORD LADDER IS THE
REAL NEWS: for the first time at 10M, on the REGISTERED g-12
channel, the rms-dial produces the graded g1b-shaped response —
and with a NEW TEXTURE the 2.74M arc never showed: W1's +2 dip to
~0.0001 (a transient near-death, far deeper than 2.74M's dip)
followed by a recovery ABOVE ROOT and a flat 0.68-0.84 hold
through +300 at a TENTH of the reference tax (+0.055 vs +0.53).
THE SHAPE: at 10M the wall's first checkpoint sees the anchor's
formation shock, then the projection holds — the ball needs its
first moments to settle before it protects. THE SCALE CLAIM:
OPEN, instrument half-formed, one licensed knob from adjudication
(g1bS4: the movement-matched dose — the fourth take, the pattern
holding: diagnose, license one knob, re-run verbatim).

"""
assert anchor in t and "## T161" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| QUEUED — GPU; the cure pattern's third application |",
              "| SUPERSEDED by g1bS3 (DONE 22:10Z: the channel FORMED at 0.6498 — 650x take-2; TEXTURE, nothing adjudicated; the graded ladder's new texture: W1's +2 dip to ~0 then above-root flat hold at tax +0.055) |", 1)
i = q.index("| g1bS3 |"); j = q.index("\n", i) + 1
row = ("| g1bS4 | THE MOVEMENT-MATCHED DOSE (the fourth take: 750 steps @ 4e-4 = e113's 0.30 rms movement — the "
       "dose-vs-ceiling question decided; then the wall verbatim) | DISPATCHING 22:11Z — the one-knob pattern's "
       "fourth application; base+install+nothing else reused |\n")
q = q[:j] + row + q[j:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T22:11:00Z"
st["current_experiment"] = ("g1bS3 FOLDED (T161: the channel forms at 10M — the wall's near-miss with the +2-dip/"
                            "above-root-recovery texture; the dose question sharp). g1bS4 DISPATCHING (the "
                            "movement-matched dose). Fleet: g1bS4 (GPU) + e198 (CPU)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g1bS3 folded")
