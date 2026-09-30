# -*- coding: utf-8 -*-
"""Fold g1bW: NOTES entry, T140 card, QUEUE rows (g1bW DONE + g1bW2 queued),
paper gap-8/R6(a)/clause-4 updates, STATE stamp."""
import io, json

# --- NOTES ---
n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g1bW — the wall's museum test: MUSEUM as registered, but the reference leg fails too — B's 300-step dose installs NOWHERE; A survives an ACTIVE second install; the tax relocates to the onset channel (2026-09-30 ~11:45Z) — DONE

WHAT WE DID: R56's killer control (spec verbatim, scratch/
r56_critic.md 54-64): W1 machinery, seed-10907 lineage; commit(0.7)
+ 50 wash -> install fact B (QUORINA on CAMILLO/AUTOLYCUS hosts,
e043-Dmix verbatim, 300 steps) UNDER the projection, anchor at A's
commit; reference = the same install on the unwalled washed control
(bit-identical inputs, 300/300 md5); zero-compute rider on the
g1bR +300 checkpoint. All 9 gates green; the wash reproduces g1bR
to 7 decimals (0.9156978 vs 0.9156979). The script was the
concurrent session's draft, verified then FIVE-bug-fixed before
running (F1 a guaranteed KeyError — it could never complete as
written; F2 fabricated g1bR constants; F4 a GPU gate deadlocking at
idle temperature; documented in the docstring).

WHAT WE SAW (T140): walled final A 0.8324 (min 0.6505, held at every
checkpoint) | B 0.0736 | CE_r 1.6285 healthy -> MUSEUM fires per its
registered letter (B <= 0.27 at healthy CE); SPLINT-REFUTED and
ZERO-SUM did not. BUT the reference leg ALSO fails the B-ruler
(0.0628): at this dose B installs nowhere, so "the wall is a
splint" is NOT contrast-licensed — reported as ambiguity inside the
fired bar. THE WALL'S MEASURED EFFECT: (1) A held through an ACTIVE
second-install attempt — the wall's strongest positive yet
(protection survives interference, not just passive wash); free-run
ZEPHYRA survived the whole ordeal (3/2800 chars, walled final).
(2) The tax relocated to the ONSET channel: B's partial-form peak
0.21 walled vs 0.53 unwalled, while A's row0 stayed protected
(0.62-0.69 vs reference's 0.006) — inside-the-ball protected,
outside-the-ball resisted. WHAT'S NEXT (queued g1bW2): the
discriminator is DOSE — B at 600+ steps on the unwashed control
(does the ruler form ever arrive without the wall? A needed ~400
steps), then the walled contrast rerun at that dose; plus a B-draw
replicate. HONESTY: n=1 lineage, one wash seed (10907), B-draw n=1
(the walled-vs-reference contrast is draw-controlled; B's absolute
level is not); all trainings cuda with pauses under external GPU
contention (logged; no migration).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

# --- T140 ---
t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T139 —"
card = """## T140 — g1bW: the museum test's honest split — A survives an active second install; the tax relocates to the onset channel (2026-09-30 ~11:50Z)

The killer control lands with an ambiguity that is itself the
finding. AS REGISTERED, MUSEUM fired (walled B 0.0736 <= 0.27 at
healthy CE 1.63) — but the UNWALLED reference failed the B-ruler too
(0.0628): at the critic's 300-step dose B installs NOWHERE, so "the
wall is a splint" is NOT contrast-licensed. WHAT THE WALL ACTUALLY
DID: (1) A held 0.83 (min 0.65) through an ACTIVE second-install
attempt — the wall's strongest positive yet: protection survives
interference, not just passive wash; free-run ZEPHYRA survived the
ordeal. (2) The measurable tax relocated to the ONSET channel: B's
trained-length partial form peaked 0.21 walled vs 0.53 unwalled,
while A's row0 stayed protected (0.62-0.69 vs 0.006). THE COHERENT
PICTURE: the wall protects the committed manifold and resists
leaving it — T133's battery-channel scope and this onset-tax are ONE
mechanism: inside-the-ball protected, outside-the-ball resisted.
THE DISCRIMINATOR IS DOSE (g1bW2 queued): B at 600+ steps unwashed —
if the ruler form arrives, rerun the walled contrast at that
operating point; the museum question adjudicated where B can install.
HONESTY: n=1 lineage, one wash seed, B-draw n=1; the concurrent
draft's F2 (fabricated reference constants) joins W021's scan — a
would-be instrument corruption caught only by verify-before-run.

"""
assert anchor in t and "## T140" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# --- QUEUE: g1bW DONE + g1bW2 queued ---
q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| g1bW |"):]
row = row[:row.index("\n")]
new = ("| g1bW | THE WALL'S MUSEUM TEST | DONE 11:45Z (T140: MUSEUM as registered but NOT contrast-licensed — "
       "B fails everywhere at the 300-step dose (walled 0.0736 / unwalled 0.0628); A SURVIVED an active second "
       "install (0.83, min 0.65, free-run intact); the tax relocated to the ONSET channel (B partial-form 0.21 "
       "walled vs 0.53 unwalled; A's row0 protected); wash reproduced g1bR to 7 decimals; the concurrent draft's "
       "script five-bug-fixed incl. fabricated constants) |")
q = q.replace(row, new, 1)
g1bw2 = ("| g1bW2 | THE DOSE CELL (g1bW's discriminator: B at 600+ steps on the unwashed control — does the ruler "
         "form ever arrive without the wall? then the walled contrast rerun at that dose; + B-draw replicate) | "
         "QUEUED — behind g1bS/g2g (GPU; C13-2 order) |\n")
j = q.index("\n", q.index(new)) + 1
q = q[:j] + g1bw2 + q[j:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

# --- paper ---
p = "scratch/day6_paper_skeleton.md"
t2 = io.open(p, encoding="utf-8").read()
old = """8. Wall's sequential test (g1bW, in flight): SPLINT-REFUTED licenses
   "memory architecture"; MUSEUM/ZERO-SUM rescopes the central positive
   to channel-scoped protection. The free-run battery (wpe-band
   collapse channels) must be reported either way."""
new = """8. g1bW DONE (T140): A SURVIVED an active second-install attempt
   (0.83 through the install; free-run expression intact) — the
   wall's strongest positive. The MUSEUM contrast itself is
   unlicensed at this dose (B installs nowhere, walled or unwalled;
   disclosed); the measured tax is the ONSET channel (B partial-form
   0.21 vs 0.53). g1bW2 (the dose cell) adjudicates where B can
   install; until then the paper says "sequential: A held through an
   active second install; the dose-adequate contrast owed"."""
assert old in t2
t2 = t2.replace(old, new, 1)
old2 = "(g1, battery-channel; sequential pending g1bW)"
new2 = "(g1, battery-channel; sequential: A held through an active second install, dose-adequate contrast owed)"
assert old2 in t2
t2 = t2.replace(old2, new2, 1)
io.open(p, "w", encoding="utf-8").write(t2)

# --- STATE ---
s = json.load(io.open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = "2026-09-30T11:50:00Z"
s["current_experiment"] = ("g1bW FOLDED (T140). Fleet: opt1b (CPU) + g1bS DISPATCHING (GPU freed). "
                           "R58 trio launching (review due). g1bW2 queued (the dose cell).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(s, indent=2, ensure_ascii=False))
print("g1bW folded")
