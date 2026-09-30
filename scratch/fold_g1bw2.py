# -*- coding: utf-8 -*-
"""g1bW fold part 2: paper (gap-8 + framing + clause 4) + STATE."""
import io, json

p = "scratch/day6_paper_skeleton.md"
t = io.open(p, encoding="utf-8").read()

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
assert old in t, "gap8"
t = t.replace(old, new, 1)

old2 = "battery-channel; sequential pending g1bW), detected (g2, with the"
new2 = "battery-channel; sequential: A held\n   through an active second install, dose-adequate contrast owed),\n   detected (g2, with the"
assert old2 in t, "framing"
t = t.replace(old2, new2, 1)

old3 = "standing tax; sequential memory tested in g1bW), a self-timed"
new3 = "standing tax; sequential: A held through an active second\n   install, dose-adequate contrast owed), a self-timed"
assert old3 in t, "clause4"
t = t.replace(old3, new3, 1)
io.open(p, "w", encoding="utf-8").write(t)

s = json.load(io.open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = "2026-09-30T11:50:00Z"
s["current_experiment"] = ("g1bW FOLDED (T140). Fleet: opt1b (CPU) + g1bS DISPATCHING (GPU freed). "
                           "R58 trio launching (review due). g1bW2 queued (the dose cell).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(s, indent=2, ensure_ascii=False))
print("part2 OK")
