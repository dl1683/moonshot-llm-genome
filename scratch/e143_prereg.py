import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

t = open("THINKING.md", encoding="utf-8").read()
if "PROXIMITY-VS-INVARIANCE" not in t:
    o = "DISCRIMINATING OBSERVATION (already queued as e140, eval-only on\nsaved nets — no new compute needed):"
    n = """PRE-REGISTRATION FOR e143 (BEFORE its dispatch, ~07:58Z — the
fork nobody has discriminated): PROXIMITY-VS-INVARIANCE. T079
says routing forms when the positional key LOSES the credit
competition (position varies across error windows). The
alternative the taxonomy (T082) makes live: PROXIMITY — error
parked NEXT TO the omnipresent row may piggyback its routing
without any position diversity (every read of the fact at
positions 5-13 co-occurs with maximal row-0 participation in the
same attention window). e143's NEAR arm (fact locked at positions
5-13, no diversity) vs FAR (locked at ~137) vs JITTER (known
routed reference) adjudicates: COMPASS-CAUSAL fires if NEAR
consolidates site-stored at 5-13 (site content-positive, row-0
dependence at install baseline) — invariance is necessary for
routing, T079 survives its strongest attack; PROXIMITY-PIGGYBACK
fires if NEAR becomes row-0-routed (presence-dependence >= 2x
install baseline) while FAR stays site-stored — proximity
inherits the route and T079's invariance clause dies (W011's
mechanism wins); UNIFORM if NEAR ~= FAR everywhere. Prediction
committed: COMPASS-CAUSAL (the e139 dream-rider showed zero
variance consolidates nothing; e120's fixed-183 splice stayed
site-stored FAR from the sink — but NEAR was never run, and
proximity is the one cell that could still rescue a weaker
invariance law). Registered before data; no shopping after.

""" + o
    assert o in t, "T079 anchor"
    t = t.replace(o, n, 1)
    open("THINKING.md", "w", encoding="utf-8").write(t)

q = open("QUEUE.md", encoding="utf-8").read()
# e140 status fix
o_e140 = "| e140 | ROUTE-DEPENDENCE TRACE (reworded per T081: the e131 test measures row-0 PRESENCE-necessity; eval-only on e119's saved phase nets) | READY (CPU) |"
n_e140 = "| e140 | ROUTE-DEPENDENCE TRACE (reworded per T081: the e131 test measures row-0 PRESENCE-necessity; eval-only on e119's saved phase nets) | DISPATCHED 08:05Z (CPU) |"
assert o_e140 in q, "e140 status"
q = q.replace(o_e140, n_e140, 1)

# e143 + the missing ideator rows, inserted after the e141 row
o_q = "| e140 | ROUTE-DEPENDENCE TRACE"
rows = """| e143 | ERROR-PLACEMENT STEERING (T076's only causal test; T079's proximity-vs-invariance fork, pre-registered 07:58Z) | DISPATCHED 07:58Z (GPU; 3x300-step fine-tunes) | root e048_repro: NEAR (fact locked at positions 5-13, no diversity) vs FAR (locked ~137) vs JITTER (e113 recipe control). Readouts: site content test at trained site, row-0 presence-dependence (e131 instrument), D-all battery, novel-geometry. Bars: COMPASS-CAUSAL = NEAR site-stored at 5-13 with row-0 at baseline; PROXIMITY-PIGGYBACK = NEAR row-0-routed >=2x baseline while FAR site-stored; UNIFORM = NEAR~=FAR |
| e142 | ROW-0 AT BIRTH — install-dose census (R44 ideator; was there ever an address-only phase?) | READY (CPU eval-only, ~15-20 min, 11 saved nets) | row-0 content test across e048_dose/direct400/direct800/repro, e044 zephyra installs, e098_install_s4305-4308, e117_install_s4309, e082_b43_install. Bars: ADDRESS-ONLY-EVER = any checkpoint row-0 <=2x control while decision row content-positive; ROW-0-ALWAYS = row-0 clears at every dose (address->field story was row-0-co-carried throughout; W011c promoted to law); HUB-FIRST = low-dose row-0-dominant, 129 takes over later |
| e144 | FROZEN-SINK INSTALL (interventional twin of e142; sequence after e142 reads) | QUEUED | install with wpe row 0 gradient-masked; COMPENSATION = decision-row content >=1.5x family norm; SINK-ATTRACTOR = post-jitter key lands on virgin row 0 anyway; ALTERNATE-HUB = key lands elsewhere |
| e145 | HUB ACROSS FAMILIES (after e139) | QUEUED | jitter-consolidate on e098 s4305/s4308 + e082 B43 (+ R43 optional); HUB-UNIVERSAL = >=3/3 row-0-keyed; FAMILY-MODULATED = strength order matches T069's install-time row-0 share (4308>4305>B43); DIVERGENT = any family keys elsewhere |
"""
idx = q.index(o_q)
q = q[:idx] + rows + q[idx:]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e140 (CPU, route-dependence + L-CYCLED) + e143 (GPU, error-placement steering — T079's proximity-vs-invariance fork, pre-registered)."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e143 pre-registered; e142/e144/e145 rows added; e140 status fixed")
