# -*- coding: utf-8 -*-
"""Fold g1bS2: NOTES, T159, ledger C6, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g1bS2 — the wall at 10x, take 2: TEXTURE (G-ROOT failure) — the e113 CONSOLIDATE recipe is scale-bound too; T148's cure pattern works (the base passed by construction); the g0 co-report is g1b-SHAPED at 10x; the Adam-clock holds at lr*sqrt(P) (2026-10-01 ~21:10Z) — DONE

WHAT WE DID: sixth-dispatch lineage; three external process kills
survived (the consolidate rerun x3 under the neighbor's INT8
bursts; progressive metrics carried every phase; pause-and-wait
held, never migrated). The licensed val-min-anchored base PASSED
ITS GATE BY CONSTRUCTION (1200-step cosine; val 2.13 -> 1.569
monotone; final BELOW g1bS's own 4000-step minimum — T148's cure
works). The wall cell ran VERBATIM: install (movement-matched;
post-install g0 0.5013) -> consolidate (e113 verbatim lr 1e-3) ->
commit -> W1/W2/W3 {R_rms ladder} vs C.

WHAT WE SAW (T159): NO WALL BAR ADJUDICATED — VERDICT TEXTURE
(GATE FAILURE: G-ROOT): the registered g-12 channel DIED INSIDE
THE CONSOLIDATION (root g-12 0.0010 vs the 0.78 bar; post-install
was 0.042) while corpus CE degraded 1.54 -> 2.58 under the
verbatim lr-1e-3 treatment — THE e113 CONSOLIDATE RECIPE IS THE
THIRD SCALE CASUALTY (base cosine; base steps; now the consolidate
lr): the house pipeline's treatments are scale-bound one by one.
THE FACT ITSELF INSTALLED AND GENERALIZED AT THE SITE (root g0
0.6583, held30 0.8293; the carrier distributed across the old wpe
band, not row 183). THE CO-REPORT THAT TRAVELS (never adjudicated):
the wall's g0 behavior is QUALITATIVELY g1b-LIKE AT 10x — C dies at
+2 (0.658 -> 0.004); W1 (1x rms) HOLDS FLAT ~0.5 THROUGH +300; W2
dips 0.069@+2 and recovers ~0.45; W3 near-killed — the
rms-convention dial produced a GRADED g1b-SHAPED RESPONSE at 10x;
D_kill = 3.146 raw = ONE AdamW step = lr*sqrt(P) (T139's Adam-clock
CONFIRMED at 10M); G-PIN/INPUTS/STEP1/BITROOT all bit-clean at
scale; the tax co-report W1-C dCE@300 +1.23 (ref +0.53, with the
degraded-root caveat). TAKE-3 NAMED (g1bS3): the width-scaled e113
license (4e-4 movement-matched consolidation — the same cure
pattern applied to the third casualty); the g0 co-report says the
wall itself will translate once the instrument does. HONESTY: n=1
host/seed/fact; the co-reported g0 behavior carries no bar.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T158 —"
card = """## T159 — g1bS2: the third scale casualty and the cure pattern that generalizes (2026-10-01 ~21:10Z)

The wall-at-10x saga closes its second act honestly: the val-min-
anchored base license WORKED (the gate passed by construction —
T148's cure is proven as a pattern), but the verbatim e113
consolidation killed the registered channel (g-12 0.0010): THE
RECIPE STACK IS SCALE-BOUND ONE COMPONENT AT A TIME (base cosine;
base steps; consolidate lr). THE CURE PATTERN GENERALIZES WITH THE
DIAGNOSIS: each treatment gets its own width-scaled license
(val-min-anchored schedules; movement-matched doses); g1bS3 (the
4e-4 consolidation take) is named. THE CO-REPORT IS THE REAL NEWS:
on the g0 channel (never the registered ruler) the wall behaves
QUALITATIVELY AS AT 2.74M — C dead at +2, W1 flat ~0.5 through
+300, a graded response across the rms ladder — the wall itself
appears to TRANSLATE to 10x; what failed is the INSTRUMENT (the
g-12 channel the treatment killed in formation), not the
mechanism. THE ADAM-CLOCK AT SCALE: D_kill 3.146 = one step =
lr*sqrt(P) exactly — T139's arithmetic holds at 10M with all
mechanics bit-clean. THE HONEST LEDGER LINE: the wall's scale
claim stays OPEN-but-encouraging (a co-reported shape, not an
adjudicated bar); the recipe-stack lesson is itself a finding the
paper's discussion carries (treatments do not transfer across
host sizes without re-licensing — a small-scale lab's pipeline
discipline for the scaled world).

"""
assert anchor in t and "## T159" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

c = "scratch/claims_ledger.md"
s = io.open(c, encoding="utf-8").read()
old = "scale OPEN — g1bS BLOCKED at the base gate (the house recipe overtrains the 10M host ~3x past val-min; g1bS2 licensed with the val-min-anchored recipe)"
assert old in s, "c6"
s = s.replace(old, "scale OPEN-ENCOURAGING — g1bS BLOCKED at the base gate; g1bS2's base license WORKED but the e113 consolidate recipe is the THIRD scale casualty (the recipe stack re-licenses component-by-component; g1bS3 named); the g0 co-report is g1b-shaped at 10x (W1 flat ~0.5 through +300; the Adam-clock = lr*sqrt(P) confirmed at 10M) — a shape, not an adjudicated bar", 1)
io.open(c, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHED 17:17Z (fifth-dispatch lineage; the licensed val-min-anchored recipe: lr 4e-4, cosine ~1200 steps) |",
              "| DONE 21:10Z (T159: TEXTURE — the consolidate recipe is the THIRD scale casualty; the base license WORKED by construction; the g0 co-report g1b-shaped at 10x (W1 flat through +300; the Adam-clock = lr*sqrt(P) at 10M); g1bS3 named — the width-scaled e113 license) |", 1)
i = q.index("| g1bS2 |"); j = q.index("\n", i) + 1
row = ("| g1bS3 | THE TAKE-3 (the width-scaled e113 license: 4e-4 movement-matched consolidation, then the wall "
       "verbatim) | QUEUED — GPU; the cure pattern's third application |\n")
q = q[:j] + row + q[j:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T21:11:00Z"
st["current_experiment"] = ("g1bS2 FOLDED (T159: the third scale casualty; the cure pattern works; the g0 co-report "
                            "g1b-shaped at 10x) - THE WAVE IS FULLY CLOSED. Fleet 0. The paper fully drafted. Next "
                            "wave: g1bS3 + g1c-root (GPU); the discriminating lineage; then the trim + assembly.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g1bS2 folded - wave closed")
