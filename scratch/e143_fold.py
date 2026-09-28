import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E140 — route-dependence trace:"
entry = """## E143 — error-placement steering: COMPASS-CAUSAL — the committed prediction holds; invariance survives its observational death by intervention (2026-09-28 ~09:00Z) — DONE

WHAT WE DID: three 300-step arms from the e048_repro root —
NEAR (fact locked at positions 5-13, zero diversity), FAR
(locked ~137-143), JITTER (e113 recipe); full battery; gates
clean; 427s.

WHAT WE SAW (T084): COMPASS-CAUSAL fires — NEAR consolidates
SITE-STORED at 5-13 (site strength 0.280) with row-0 presence
at/below install baseline (0.232 vs midpoint 0.634; PIGGY bar
1.091 — proximity piggybacking DEAD). The error-placement
compass is now CAUSAL: choose the error's site, choose the
store's site. Invariance survived its strongest attack by
INTERVENTION despite T079's observational death (T083): zero
position-variance at a sink-adjacent site did NOT create a
route. TEXTURE: FAR is a HYBRID — novel-geometry expression
0.249 (vs NEAR 0.002, JITTER 0.915), row-0 strength 0.943 at
trained geometry (sink-load per T083's saturation lesson, not
necessarily routing); FAR's site overlaps the install band
121-137, so the band population may supply de facto support
diversity — the invariance law's width-0 boundary may have a
band-overlap loophole, OR FAR's 0.249 is the old band field
(content-carried). Discriminator queued (free, minutes):
d_r0@g-12 on e143_far/jitter — FAR-ROUTED-TAIL (collapses >=70%)
vs CONTENT-TAIL (drops <=30%).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T083 — E140:"
t084 = """## T084 — E143: the compass is causal; invariance survives by intervention what it lost by census (2026-09-28 ~09:00Z)

The committed prediction (COMPASS-CAUSAL, registered 07:58Z
before dispatch) HELD. Three load-bearing consequences:

(1) T076's compass is now CAUSAL, not observational: parking the
fact's error at positions 5-13 built a site-store AT 5-13 with
row-0 presence at baseline. Error placement chooses the storage
site — a steering wheel, not just a description. The second
paper's claim 1 upgrades to interventional.

(2) INVARIANCE's strange double life, resolved: T079 died on its
registered observational dial (T083, saturation), but its
SUBSTANCE just passed the causal test — zero position-variance
at a sink-ADJACENT site produced no route. Proximity was the
last alternative to invariance for key-selection, and it is
dead. The law's remaining unknown is the WIDTH dose-response:
does address-key death co-occur with route birth as width grows?
PRE-REGISTERED (e147, BEFORE dispatch): the width ladder w in
{1,2,4,16,32,64} from the same root, co-measuring address-key
strength A(w) (row-129 replacement delta) and novel-geometry
routing NR(w) (row-0 presence at g-12 + one more novel
geometry). INVARIANCE-CAUSAL fires if A(w) is monotone
decreasing (Spearman <= -0.8), crosses <= 0 at w*, with NR(w)
onsetting (>= 2x install baseline) within one bin of the same
w* — the co-onset of address-key death and route birth is the
causal joint. DEAD-AGAIN if A(w) flat, or routing onsets while
A(w) still >= +0.15. SEED-COVERAGE rider (W010's ghost): if
routing peaks at +-8 and collapses at +-32/64, seeds have finite
reach; T079-pure predicts +-64 routes at least as well as +-8.

(3) FAR's hybrid texture is the invariance law's first boundary
case: locked at 137-143 — overlapping the install's own band —
FAR shows novel-geometry expression 0.249 where pure NEAR shows
0.002. Either the band population supplies de facto support
diversity (invariance loophole: overlap counts), or the 0.249 is
the old band field expressing (content-carried). The free cell
(d_r0@g-12 on e143_far) adjudicates: FAR-ROUTED-TAIL vs
CONTENT-TAIL. Registered before running.

ALSO REGISTERED (the dream confound, honest): the R45 ideator
found that e139's dream harvest used 130-char prompts, which
place every first continuation token at x-col 130 BY
CONSTRUCTION — the '33/34 at the old address' savor is partly
rig geometry. The claim is CONFINED until the randomized-length
dream-topology census runs (queued as e148): ADDRESS-SEEKING
(install/locked nets concentrate onsets near 129 at >=5x
uniform, p<0.01) vs ROUTE-DISSOLVES-ADDRESS (routed nets flat)
vs ROUTE-KEEPS-AN-ADDRESS (a generation address distinct from
the read address — a new object if real). The erosion mechanism
(fixed-geometry replay never varies position) survives either
way, but DAY_SIX_REPORT's phrasing is downgraded to match.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t084, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- DAY_SIX_REPORT dream caveat ----------
r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
o_r = """And the day's best savor:
   the net's dreams put 33/34 name occurrences at the OLD ADDRESS
   — dreams dream in the fact's coordinates, which is why replay
   without error erodes (0.230, below base 0.391) instead of
   consolidating."""
n_r = """And the day's savor, now
   bounded: the net's dreams put 33/34 name occurrences at the
   old address's column — though the harvest's 130-char prompts
   place onsets there BY CONSTRUCTION (R45 ideator's confound),
   so the address-seeking claim awaits the randomized-length
   census (e148); what survives either way is the mechanism —
   fixed-geometry replay never varies position, and replay
   without error erodes (0.230, below base 0.391) instead of
   consolidating."""
assert o_r in r
r = r.replace(o_r, n_r, 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

# ---------- QUEUE.md ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e143 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e143 | error-placement steering | DONE 09:00Z (T084: COMPASS-CAUSAL — committed prediction HELD; NEAR site-stored at 5-13 (0.280) with row-0 at baseline (0.232, PIGGY bar 1.091 dead); FAR hybrid texture — novel-geom 0.249 vs NEAR 0.002; free d_r0@g-12 cell queued) |\n" + q[m.end():]

o_q = "| e144 | FROZEN-SINK INSTALL"
rows = """| e147 | THE WIDTH LADDER (T079's pre-registered revival — bars registered in T084 BEFORE this dispatch) | DISPATCHED 09:00Z (GPU free after e143) | from e048_repro: 300-step replay arms at jitter widths w in {1,2,4,16,32,64} (w=0 locked, +-8, NEAR/FAR exist on disk as free endpoints); co-measure A(w) = row-129 address-key strength AND NR(w) = row-0 presence at g-12 + one more novel geometry; D-all tertiary. Bars: INVARIANCE-CAUSAL = A(w) monotone dec (rho<=-0.8), crosses <=0 at w*, NR onsets within one bin of same w* (co-onset = causal joint); DEAD-AGAIN = A flat or routing onsets while A>=+0.15; SEED-COVERAGE = routing peaks +-8, collapses +-32/64 |
| e148 | DREAM-TOPOLOGY CENSUS with randomized harvest (the dream-address confound must be discharged before the claim travels) | READY (CPU ~30-60 min; after e142) | regenerate matched dream samples from 7 saved nets (twin/L@150/R@150/R@300/consolidated/L-cycled/site-stored) with RANDOMIZED prompt lengths 60-200; census name-onset read positions. Bars: ADDRESS-SEEKING = install/locked nets concentrate onsets near 129 >=5x uniform (p<0.01); ROUTE-DISSOLVES-ADDRESS = routed nets flat <2x; ROUTE-KEEPS-AN-ADDRESS = significant mode in a routed net (new object) |
| e149 | ROUTE-VS-SCAR SURGERY on row 129 (W006's brake mechanism; e141's instrument moved to the origin row) | READY (CPU eval-only ~10-20 min) | restore install-phase wpe[129] into routed nets (R@150/R@300/consolidated), measure brake before/after; co-measure cos(dwp129, install fact direction) + attention mass at readout; E/L as brake-negative controls. Bars: SCAR-IN-CONTENT = restore removes >=50% of brake AND anti-alignment cos<=-0.3 in routed nets only; SCAR-IN-POLICY = brake persists under full restore (suppression lives in readout weights — symmetric with e141's destination finding) |
""" + o_q
assert o_q in q
q = q.replace(o_q, rows, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e142 (CPU, row-0-at-birth) + e147 (GPU, width ladder — T079 revival, bars pre-registered in T084). e143 DONE: COMPASS-CAUSAL."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e143 fold complete")
