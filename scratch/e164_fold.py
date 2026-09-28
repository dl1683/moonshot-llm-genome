import json, datetime, re

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E158 — the 2x2 completion: TEXTURE"
entry = """## E164 — the post-kill census: SUBSTANCE-SURVIVES + MLP-WRECK-ONLY — the knife severed ACCESS not STORAGE; the four-layer model's circularity is broken (2026-09-28 ~13:35Z) — DONE

WHAT WE DID: the N2-killed state rebuilt in-memory (bit-exact vs
e160's stored cell: 70.75% @ +0.245); the full organ census on
the killed net; the MLP-coordinate plane on the site-fact.

WHAT WE SAW (T098): PART A — SUBSTANCE-SURVIVES: behind the
dead readout, the fact's machinery is intact and organized as in
the root — row-0 dependence full strength (0.918-1.03), MLP body
still consumes ~100% of the killed net's headroom, ALL root
top-5 non-N2 heads still load-bearing (L0H3 0.218...), with
sham contrasts 2.9-3.9x noise. THE KNIFE SEVERED ACCESS, NOT
STORAGE — layers 1/2 are genuinely separable. No organ or third
head restores the readout (best: L2H5 partial re-open +0.0875 —
a weak suppressor texture). TEXTURE: the band-5 brake sign FLIPS
post-kill (root -0.120 HELPS -> killed +0.140 COSTS). PART B —
MLP-WRECK-ONLY: the head plane reproduces NO-SITE-KNIFE (best
flat 21.8%; ladder saturates 43.6% @ +1.15); the MLP plane HAS
an any-CE kill (mlp_l5 alone: 92.9% @ +0.793 — the cheapest
full kill of the site-fact) but NO flat-CE kill exists ({l1,l2}
39.5% @ +0.24). The MLP third IS the incorrigible substrate —
removing it is indistinguishable from wrecking the organism;
un-killability at flat CE holds on BOTH surfaces. Honesty:
in-memory kill bit-exact (5e-6) but stateless (no dynamics);
"unchanged weights" is a construction tautology — SUBSTANCE
rests on functional probes; floor compression on the killed
baseline (bars >= 50% headroom + shams + absolutes); mode/
bank conventions differ across planes (priced).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING ----------
t = open("THINKING.md", encoding="utf-8").read()
anchor = "## T097 — [CORRECTED per R48 critic"
t098 = """## T098 — E164: access severed, substance intact — the four-layer model earns its figure, and the MLP is the organism-priced organ (2026-09-28 ~13:35Z)

The R47 critic's circularity charge is answered: each layer is
now defined not only by what removes it but by INDEPENDENT
evidence of separation — the N2 kill leaves the row-0 door at
full strength, the MLP body consuming full headroom, and every
remaining fact-head load-bearing. A surgical readout death with
the storage profile intact is exactly what "layers 1/2
separable" predicted and what a one-layer account forbids. THE
MODEL FIGURE IS LICENSED. The brake-flip texture (band-5 helps
the root, costs the killed net) says the brake is a property of
the LIVE readout configuration — it dies with the access it
modulates.

THE MLP'S DOUBLE ROLE: for the site-fact, the MLP third is the
cheapest full kill (mlp_l5 alone: 92.9%) — the load-bearing
remainder — yet only at organism prices (+0.79). INCORRIGIBLE
IN THE STRONG SENSE: the site-stored memory's un-removability
is not the absence of a kill coordinate but the fact that every
kill coordinate is a vital organ. Combined with e125a: the
site-fact has no flat-CE kill on EITHER surface; the
asymmetry of existence now rests on two exhaustive planes.

FOR THE PAPER: claim 3 gains its separability exhibit (the
post-kill census); claim 4's scope clause strengthens (both
surfaces); the closing sentence's split custody now has its
mechanism diagram — access (severable), storage (intact),
dependence (row-0, untouched by the knife).

""" + anchor
assert anchor in t
t = t.replace(anchor, t098, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- paper ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "R3 Content everywhere, access differs (e133/e141/e142): the\n   three-net maps; install-restore/perm/halfnorm riders; the\n   origin census (13/13) [e163 pending for the dial license]."
n_p = "R3 Content everywhere, access differs (e133/e141/e142): the\n   three-net maps; install-restore/perm/halfnorm riders; the\n   origin census (13/13) [e163 pending for the dial license];\n   the POST-KILL CENSUS (e164): behind the N2-killed readout,\n   storage intact and organized — access severed, substance\n   survives; the MLP plane answers the site-fact's remainder\n   (any-CE kill at organism prices only)."
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

# ---------- queue + state ----------
q = open("QUEUE.md", encoding="utf-8").read()
m = re.search(r"^\| e164 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e164 | post-kill census | DONE 13:35Z (T098: SUBSTANCE-SURVIVES — access severed, storage intact (row-0 full, MLP full headroom, all heads load-bearing, shams 2.9-3.9x); brake flips sign post-kill; MLP-WRECK-ONLY — the site-fact's MLP third is the cheapest kill at organism prices (+0.79); no flat-CE kill on either surface; the four-layer figure licensed) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 3: e154 (two facts) + e166 (inverse event) + e152R (dwell re-seeds). e164 DONE: SUBSTANCE-SURVIVES — the layering validated."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e164 fold complete")
