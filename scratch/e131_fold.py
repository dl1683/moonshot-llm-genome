import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E120 — fact in contexts:"
entry = """## E131 — the re-keying census: RE-KEYED — the key is ROW 0, and the e120 splice arms never failed (2026-09-28 ~07:05Z) — DONE

WHAT WE DID: R43 critic's discriminator, as dispatched. Nets
regenerated bit-exactly from gated roots (G_E120/G_E113 max diff
0.0 vs stored tables; e082_b43_install + e048_repro loaded, e044
scar line inspected and rejected); CPU-only, 1538s; phase nets
saved runs/checkpoints/e131_*.pt. Four probes.

WHAT WE SAW (T077): all three RE-KEYED conditions fire.
(1) ROW-0 CONTENT TEST: post-consolidation strength 0.732 — ABOVE
its install-phase 0.545, 380x the max control row (0.0019); the
ONLY content-positive row in the census; the whole 121-137 band
including 129 sits at ~0, and row-129 replacement RAISES p(Z)
(the e115 brake, now with the new home identified). (2) BAND-
MINUS-ROW-0: D-all reproduces e113 exactly (0.906); +row-0
collapses expression to 0.024 (-97%); scaffold-matched D-all+ROW-1
survives untouched (0.908). (3) CENSUS (weakest): worst OOB row
249 merely matches row 129's own delta — conditions 1+2 carry the
verdict. PROBE 1 (183-geometry read): both splice arms express
the fact at address 183 — p(Z) 0.989 (self) / 0.988 (corpus),
frac p>=0.5 = 1.000 over 840 name-char reads — vs band 0.015-0.035
and base pre-ft 0.061/0.066. E120's SIGNAL-IN-CONTEXTS
INSUFFICIENT was INSTRUMENT BLINDNESS: the arms consolidated
exactly where their training error lived, at an address no battery
read. T076's error-location wins outright. Honesty bounds: row-0
collapse has a scaffold component (CE_R +1.40) but two independent
controls bound it (mean-replacement arm kills identically; row-1
control null); necessity != storage (row 0 may be the ROUTE, with
content in body weights — e133's question); probe 1 is a
training-geometry read (learning-at-183 proven, novel-context
generalization at 183 untested -> e139).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T076 — the critic's error-location theory taken straight:"
t077 = """## T077 — E131: the fact never left the positional system — it re-keyed to ROW 0, and 'failed' consolidations were instrument blindness (2026-09-28 ~07:05Z)

The R43 critic's most-damaging assumption was the right one, and
the discriminator settled it in one run. Three inversions, in
order of severity:

**(1) THE MIGRATION'S DESTINATION IS ROW 0, NOT 'THE BODY'.**
Post-consolidation, row 0 is the only content-positive wpe row
(strength 0.732 — ABOVE its install-phase 0.545: consolidation
STRENGTHENED the row-0 key), the band is content-null, and D-all+
row-0 collapses expression -97% while the scaffold-matched row-1
control survives. e113's BODY-STORED verdict is DEAD on this line:
what survived D-all was a row-0-keyed fact. W005's terminal
'coordinate-independent' stage inverts — the fact never became
coordinate-free; it changed coordinates (129 -> 0) and AMPLIFIED
there. The e115 brake completes the story with a home: the OLD
address (129) suppresses, the NEW key (0) carries — the brake is
the moved-out tenant's old lease. HONESTY: row 0's necessity is
proven, its sufficiency is not — the agent's bound stands (row 0
may be the readout GATE with content in body weights; the census's
diffuse OOB texture is consistent with a distributed content
store behind a row-0 door). e133's anatomy census and e139's
universality probe now arbitrate route-vs-substrate.

**(2) E120'S VERDICT WAS INSTRUMENT BLINDNESS — T075 RETIRED.**
Both splice arms express the fact at 0.989/0.988 at the
183-geometry — they consolidated exactly where their error lived,
at full strength, in an address the battery never read. The
'signal-in-contexts insufficiency' never happened; the position-
diversity ingredient RETIRES with it. What ACTUALLY differed
between e120's arms: WHERE each arm's error sat (a/b/c: full-
column CE at 183; d: name-only mask in the band) — a contrast the
critic flagged and the verdict ignored. T076's error-location is
now the lab's consolidation law candidate: THE FACT CONSOLIDATES
WHERE ITS ERROR IS PLACED. Its own open edges: (i) arm c (verbatim
dreams, low error everywhere) still consolidated nothing in the
band — but nobody read arm c at its dream positions; its verdict
is ALSO unproven blindness until read (e139 rider); (ii) road E
consolidates with no fact-error at all — error-OR-necessity
still live for the erase road; e119's battery adjudicates.

**(3) W010 SEED-AND-AMPLIFY IS KILLED by probe 1.** A seedless
site 54 rows from the band climbed to 0.99 expression — error
alone suffices; there was nothing to amplify at 183 and nothing
wanders once read at its own geometry. The card dies cleanly and
gratefully (it took one run). W009 POPULATION FRAME is BOUNDED,
not dead: the band's texture is genuinely overlapping/diffuse
(census), but row 0 is a discrete hub — the population language
survives for the band, dies for the key. The lab's nouns after
e131: a row-0 KEY, a brake at the old address, a diffuse content
population behind the key, and error-placement as the
consolidation compass.

STANDING PRE-REGISTRATIONS: W010's P1/P2/P3 vs e119 are MOOT as
written (they assumed R-vs-E differences around a field concept
that just collapsed; P2's census comparison survives as texture).
Adjudicate honestly when e119 lands: the interesting question it
now carries is whether the ERASE road also ends row-0-keyed (if
yes: row 0 is the universal attractor of consolidation; if no:
the roads genuinely differ). NEXT DISCRIMINATOR (e139, dispatched
this beat): row-0 UNIVERSALITY — test the 183-consolidated splice
arms' dependence on row 0. Row-0-keyed too => row 0 is the
universal readout key (and e120's arms 'failed' only the battery,
not biology). Site-locked at 183 => re-keying to 0 is a property
of the JITTER road alone, and the roads truly diverge.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t077, 1)

# W010 death marker
o_w010 = "REGISTERED BEFORE E119'S REPORT (2026-09-28 ~06:48Z"
n_w010 = """KILLED BY E131 PROBE 1 (~07:05Z): the splice arms climbed to
0.989/0.988 at seedless row 183 — error alone suffices; the
trajectory 'wandering' was the band-geometry readout of a fact
that lived at 183. Card closed. (P1/P2/P3 below were registered
in good faith against the then-current frame; they adjudicate as
written when e119 lands, P2 alone likely still informative.)

""" + o_w010
assert o_w010 in t
t = t.replace(o_w010, n_w010, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE.md ----------
q = open("QUEUE.md", encoding="utf-8").read()
o_q = "| e131 | RE-KEYING CENSUS"
import re
m = re.search(r"^\| e131 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e131 | RE-KEYING CENSUS | DONE 07:05Z (T077: RE-KEYED 3/3 — key is ROW 0, strength 0.732>install 0.545, D-all+row0 -97% vs row-1 control null; probe 1: splice arms 0.989/0.988 at 183 — E120 verdict was instrument blindness; T075 retired, W010 killed, W009 bounded) |\n" + q[m.end():]

# e139 row, inserted before e132
o_q2 = "| e132 | the wiring trace"
e139 = """| e139 | ROW-0 UNIVERSALITY + 183-ROBUSTNESS (T077's central open question; branch B of scratch/w010_preseed_design.md, updated) | DISPATCHED 07:05Z (eval-only CPU on e131's saved nets) | probes on e131_arm_{a,b} + consolidated: (1) row-0 content test on the 183-consolidated splice arms; (2) deletion battery at 183-geometry: none/D-183/D-band/D-band+183/D-row-0/D-row-0+183; (3) NOVEL-context generalization at 183 (held-out contexts shifted, fact at 183 — closes e131's honesty note d); (4) row-129 brake scope on splice arms; (5) rider: read arm-c-verbatim-dreams fact expression at its actual dream positions (is arm c ALSO instrument-blind?). Bars: ROW-0-UNIVERSAL fires if splice fact drops >=50% under row-0 deletion; SITE-LOCKED fires if D-183 kills >=80% and row-0 test null; HYBRID if both partial. |
""" + o_q2
assert o_q2 in q
q = q.replace(o_q2, e139, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE.json ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e119 (GPU rerun w/ ckpts, battery ~due) + e139 (CPU eval-only, row-0 universality). e131 DONE: RE-KEYED — key is row 0; splice arms consolidated at 183 (E120 verdict was instrument blindness)."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e131 fold complete")
