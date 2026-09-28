import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E131 — the re-keying census:"
entry = """## E133 — field anatomy census: TEXTURE — the fact is BODY-stored everywhere; what differs is the READ ROUTE (2026-09-28 ~07:45Z) — DONE

WHAT WE DID: per-organ causal ablation sweep (MLPs by layer,
attention heads, wpe row sets; mean-replace mode picked by
pre-registered CE_R rule over zero by 0.022 nats; both recorded)
on three gated nets: graduated (e131_consolidated, bit-exact
0.7850374), address-phase twin (e119 twin start, 0.5563082),
site-locked (e131 arm_b, site onset bit-exact 0.9880021).

WHAT WE SAW (T080): verdict TEXTURE — SUBSTRATE-IN-BODY half-
fired (graduated body share 0.689 >= 0.60 TRUE; twin address
mirror 0.388 < 0.60 FALSE — general organs dominate the marginal
denominator even pre-consolidation); ROUTE-ONLY dead (body 0.810
>> 0.30); W005-ALT (address-spreading) KILLED (band-adjacent
heads 0.120 < 0.50). Instrument cross-validated: wpe_r0 drops
0.732/0.546 reproduce e131/e116 stored strengths exactly. TEXTURE
FINDINGS: (1) ALL THREE NETS store the fact mostly in the body —
even the 'site-locked' 183 net keeps only 7.9% at row 183 (MLP
64.4%); content substrate is shared, READ ROUTES differ (W013).
(2) LOCALITY FILTER (CE damage <= 0.75): graduated fact-specific
residue is HEAD-dominated (84.5% heads / 15.5% MLP / wpe ~0) with
L0H3 a genuinely specific body head (0.46 drop at 0.21 CE); the
unfiltered registered map is dominated by load-bearing-for-
everything machinery (mlp_l0/l5, 2.4-4.1 nats CE). (3) ADDITIVITY
FAILS: joint top-MLP+top-head+key-row drops 0.785 vs parts-sum
1.98 — ~2.5x redundancy; W009's population frame confirmed at
the ORGAN level; all shares are marginals, not a partition.
(4) The TWIN runs a genuine positional-address apparatus — L3H4
reads the band with 91% attention mass (0.319 drop at 0.008 CE
damage) — and the graduated net has DISMANTLED it (address-head
share 0.12; band5 deletion now RAISES p(Z) 0.120 — brake
replicates; twin band5 drop +0.364 at ~0 CE = fact-specific
address read). (5) NO head is sink-adjacent >= 0.25 on the
scored position: row 0's causal load flows through SMALL-
attention VALUE channels — the sink route is not a high-sink-
attention route at this readout. Honesty: ablation-mode near-tie;
additivity assumption stated as failed; expression-drop
conflates fact load with general wreckage (both views recorded);
address-head label coarse (26-27/36 clear bar; sensitivity table
shows the two kills unchanged).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T079 — the credit-assignment law:"
t080 = """## T080 — E133: content is everywhere, routes are the difference — the read-policy frame's first direct support (2026-09-28 ~07:45Z)

The anatomy census returned TEXTURE as registered, but the
texture IS the finding, and it is the strongest support yet for
W013's protagonist. THREE convergences:

**(1) ALL THREE NETS ARE BODY-STORED.** The graduated fact keeps
19.1% (unfiltered) at row 0 and ~0 fact-specific; the twin keeps
its address apparatus (L3H4: 91% band attention, clean 0.319
drop) yet still stores most content in general organs; the
'site-locked' 183 net keeps only 7.9% at its own site. CONTENT
SUBSTRATE IS SHARED ACROSS ALL THREE STATES — install, splice,
and consolidated differ in their READ ROUTES, not their storage
organs. The address-vs-field dichotomy that organized five days
of experiments was a dichotomy of ROUTES all along (W013's
claim, now with a map).

**(2) THE FACT-SPECIFIC RESIDUE IS HEAD-DOMINATED.** Under the
locality filter, the graduated net's fact-specific load is 84.5%
heads (L0H3: 0.46 drop at 0.21 CE — a genuinely specific body
head), 15.5% MLP, ~0 wpe. Combined with the R44 critic's census
crack (wpe[0] delta minimal) and e131's necessity (row-0 deletion
kills): row 0 is a CHANNEL — necessary for the route, carrying
almost no fact-specific content — and the route finishes in
heads. This is ROLE-ROUTED (T077's amendment) with the route's
terminus located. It also REFINES W011: no head is sink-adjacent
>= 0.25 — the sink route is a VALUE-CHANNEL route (row 0's
contribution flows through V, not through attention mass);
omnipresence may operate through what row 0 ADDS to every
residual stream, not what it attends to.

**(3) REDUNDANCY IS ORGAN-DEEP.** Joint ablation of the top
three organs drops 0.785 where parts sum to 1.98 — the fact is a
redundant population at every level observed (rows: e088;
organs: e133). W009's population frame graduates from metaphor
to measurement; the additivity assumption is dead lab-wide, and
every future 'load' number must be stated as a marginal.

THE DEVELOPMENTAL RE-READ: twin -> graduated is the DISMANTLING
of L3H4's address read (0.32 clean drop -> 0.12 share) while
body organs carry more — the read policy's canal (T037)
rebuilt. e140's row-0 trace and e141's surgery now arbitrate the
route's remaining structure; the POLICY TRANSPLANT (W013) has
its target organ list (L0H3 + the value channel + whatever
e141 isolates).

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t080, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE.md ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e133 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e133 | field anatomy census | DONE 07:45Z (T080: TEXTURE — all three nets body-stored (site-locked keeps 7.9% at 183); fact-specific residue HEAD-dominated 84.5% (L0H3); additivity FAILS 0.785 vs 1.98 (population confirmed organ-level); twin's L3H4 address reader dismantled by graduation; no sink-attention head — row-0 route is a VALUE channel; W005-alt killed, ROUTE-ONLY dead) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE.json ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e139 (CPU, universality) + e141 (CPU, sink-key mechanism). e133 DONE: TEXTURE — content everywhere, routes differ (T080)."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e133 fold complete")
