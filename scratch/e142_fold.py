import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E143 — error-placement steering:"
entry = """## E142 — row-0 at birth: ROW-0-ALWAYS — there was never an address-only phase, and the address itself was protocol-made (2026-09-28 ~09:30Z) — DONE

WHAT WE DID: row-0 content census (e131 instrument, within-net
adjudication) across 13 install checkpoints spanning the e048
dose ladder (1x-4x), direct/natural-exposure installs, e044
re-installs, the e098 fresh family (5 seeds), e117, e082 B43.

WHAT WE SAW (T085): ROW-0-ALWAYS fires at every dose in all 13
nets — no net's row-0 strength comes within two orders of its
2x-control bar (most conservative: 2.6x over). HUB-FIRST fails
(row 0 dominates the decision dial in 13/13; no dose where the
decision row takes over). THE HISTORY REWRITE: the five-day
'address -> field' arc was row-0-co-carried throughout — W011's
savor (c) PROMOTED TO LAW (consolidation = share-growth of the
largest seed). SHARPER: the ADDRESS ITSELF is protocol-
contingent — direct/natural-exposure installs (e048_direct400/
800) are almost purely row-0-carried (rel 0.947/0.983) with row
129 NULL (-0.006/+0.007): the 'address' was a property of the
masked-replay protocol, not of birth. In the fresh 0.84M family
row 0 carries the ENTIRE install (rel exactly 1.000 at every
seed). DOSE moves share, not presence (e048 ladder: row-0 share
0.981->0.952 as decision-row share grows 0.432->0.522 — the
address grows INTO an already-row-0-carried memory, never
overtaking). Honesty: per-net instruments (families/batteries
differ — all bars within-net); trained-geometry dial caveat
stands (the at-birth question is well-posed on its own dial;
routing-level replication needs novel-geometry arms); one
unresolved gate (e098_s4307 is an earlier-trajectory state, not
the stored patience twin — census internally valid, exclusion
would not change the verdict).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T084 — E143:"
t085 = """## T085 — E142: the address was never born — row 0 carries every install, and 'address' was the protocol's artifact (2026-09-28 ~09:30Z)

ROW-0-ALWAYS, 13/13 nets, every dose. Two consequences, one
historical and one structural:

(1) THE HISTORY REWRITE: there was no address-era. The five-day
narrative — install binds an address, consolidation migrates to
a field, the field re-keys to row 0 — compresses at every stage
to: ROW 0 CARRIED THE MEMORY ALL ALONG, and everything the lab
called migration was share-redistribution around a constant
row-0 core. W011's savor (c) is LAW: consolidation promotes the
largest existing seed. The install's apparent address-binding
(row 129) was real but SECONDARY — a protocol-made co-carrier
that never exceeded row 0's share at any dose.

(2) THE PROTOCOL-MADE ADDRESS: direct/natural-exposure installs
put essentially everything on row 0 (row 129 NULL) — the address
row forms ONLY under masked-replay installs, where the protocol
fixes the fact's position-variance to zero at 129 and the credit
lands there (T079's revived law, now visible in INSTALLATION
too: error placement chooses the store, and natural exposure
places its error at the omnipresent row). The taxonomy's
'site-stored' type is therefore PROTOCOL-SCULPTED: lock the
position, grow a site; let it vary (or let nature place it), and
row 0 takes everything. This unifies e143 (NEAR site-stored
under locked replay) with e142 (natural installs row-0-only)
under one rule with no residue.

CONNECTS: the fresh 0.84M family's rel-1.000 installs explain
T069's 6/6 row-0 content-carrying — it was never a coincidence
of the B43 line; it is the architecture's default. OPEN: the
trained-geometry dial caveat (T083) bounds this census too — the
claim is 'row 0 carries install EXPRESSION from birth', with
routing-level (novel-geometry) replication still owed; e150's
flat-CE cells and e147's ladder are the instruments that will or
won't hold the story together at the routing level.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t085, 1)

# W011 (c) promotion marker
o_w = "(c) The sink was ALREADY the fact's co-carrier at install"
n_w = "(c) [PROMOTED TO LAW by e142/T085 — 13/13 nets, every dose] The sink was ALREADY the fact's co-carrier at install"
assert o_w in t
t = t.replace(o_w, n_w, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e142 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e142 | row-0 at birth census | DONE 09:30Z (T085: ROW-0-ALWAYS 13/13 every dose; address was protocol-made — direct installs row-129-NULL; fresh family rel 1.000; W011c promoted to LAW; dose moves share not presence) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e147 (GPU, width ladder) + e150 (dispatching: flat-CE route test). e142 DONE: ROW-0-ALWAYS — the address was protocol-made."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e142 fold complete")
