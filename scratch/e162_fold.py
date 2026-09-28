import json, datetime, re

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E152 — the conversion time-trace:"
entry = """## E162 — the READ-vs-MASS fork: MIXED — the poison is a TWO-EDGED weapon, each channel independently sufficient (2026-09-28 ~12:15Z) — DONE

WHAT WE DID: the two discriminating cells + the starvation
measurement; 186s CPU; all gates bit-tight (incl. cross-run
reproduction of e159's mass profile).

WHAT WE SAW (T095): (i) VALUE-RESTORE-UNDER-POISON HEALS
COMPLETELY — poisoned key/query geometry kept, clean values at
key 0: retention x1.000 @ g-12, x1.002 @ g0, CE cost -0.0004
(restoring value content alone erases the ENTIRE organism
damage; power audit: the poison changes v0 by 1.13x its own
norm — the cell had full power). (ii) MASS-INFLATE-ON-HEALTHY
KILLS — healthy row 0, bias swept to the exact absorber dose
(1.6927): x0.036 @ g-12, x0.301 @ g0, CE +0.572, monotone
dose-response (b=3.0: x0.347; b=4.0: x0.005). (iii) Both modes
coexist in real poison: band queries' key-0 mass explodes
~60x while band keys lose ~23% relative mass. THE NOUN'S FULL
FORM: the memory depends on the sink's DUAL ROLE — what it
SUPPLIES (content read off the pivot) and what it SPARES (the
allocation the absorber would steal); either corruption alone
kills. Honesty: the value-transplant restores the full channel
(not a minimal patch); the bias matches total dose not the
per-layer profile (a sufficiency test); single net.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING ----------
t = open("THINKING.md", encoding="utf-8").read()
anchor = "## T094 — [R47: n=1 TRAJECTORY"
t095 = """## T095 — E162: two edges, one pivot — the memory depends on what the sink supplies AND what it spares (2026-09-28 ~12:15Z)

The fork resolves as MIXED, and the resolution is better than
either branch: the poison kills through BOTH channels, EACH
INDIVIDUALLY SUFFICIENT. Restoring healthy values under a
poisoned key erases ALL damage (CE -0.0004 — the corruption is
carried by the content read off the pivot); inflating the
absorber on a healthy row kills just as dead (x0.036 — the
corruption is equally carried by the allocation the absorber
steals). The sink holds a DUAL role for the consolidated memory:
SUPPLIER of read content and GUARANTOR of the attention budget.
T093's READ-coupled noun was half the story — the full form:
the memory's dependence is on the sink's FUNCTION, whose two
components are separately lethal when broken. FOR THE PAPER:
the intro's mechanism sentence becomes the two-channel form
("dies of what the degraded pivot supplies AND of what it
steals — either alone suffices"); e159's double dissociation
gains its mechanism completion; the closing sentence's split
custody is untouched. FOR T092: layer 4 = functional dependence
on the pivot (supply + allocation), not a single channel.

""" + anchor
assert anchor in t
t = t.replace(anchor, t095, 1)
# resolve T093's provisional marker
o2 = "R47-CRITIC AMENDMENT (~11:55Z — the noun is PROVISIONAL; the\nfork is live)"
n2 = "R47-CRITIC AMENDMENT (~11:55Z — RESOLVED MIXED by e162/T095:\nBOTH channels kill, each sufficient; the noun's full form is\nfunctional dependence on the sink's dual role. Original\namendment kept for record)"
assert o2 in t
t = t.replace(o2, n2, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- paper + report ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
p = p.replace("READ- vs MASS-coupled fork pending e162", "RESOLVED MIXED by e162: BOTH channels kill, each sufficient — the memory depends on the sink's dual role (what it supplies and what it spares)", 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
r = r.replace("READ-vs-MASS (e162), ", "", 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

# ---------- QUEUE + STATE ----------
q = open("QUEUE.md", encoding="utf-8").read()
m = re.search(r"^\| e162 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e162 | READ-vs-MASS fork | DONE 12:15Z (T095: MIXED — both channels kill, each sufficient; value-restore heals at CE -0.0004; mass-inflate kills x0.036 at matched dose; noun = functional dependence on the sink's dual role) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2 (CPU): e158 (2x2) + e125a (inverted knife). e162 DONE: MIXED — the two-edged poison."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e162 fold complete")
